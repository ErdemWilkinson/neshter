"""Yan yana karşılaştırma, kategori bazlı test seti ve düz dilde özet.

Üretim her zaman greedy'dir (do_sample=False). Orijinal ve müdahaleli cevap aynı çağrıdan gelir:
iki satırlık yığında yalnızca 1. satıra müdahale uygulanır (apply(..., batch_index=1)). Bellek
yetmezse aynı ayarlarla sıralı üretime geçilir; çıktılar aynıdır.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel

from neshter.concepts.detectors import _lower, detect_code, detect_refusal
from neshter.interventions import Intervention, apply
from neshter.loader import LoadedModel
from neshter.metrics import Diff, rate_diff, wilson

# --- Üretim -----------------------------------------------------------------------------------


def _encode(lm: LoadedModel, prompt: str, copies: int):
    """Sohbet şablonlu (tek kullanıcı mesajı) ya da ham girdi; `copies` kopya, sola dolgulu."""
    tok = lm.tokenizer
    templated = bool(getattr(tok, "chat_template", None))
    if templated:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt
    device = next(lm.model.parameters()).device
    # Şablon BOS gibi özel token'ları zaten yazar; ikinci kez eklenmesin.
    return tok([text] * copies, return_tensors="pt", padding=True, add_special_tokens=not templated).to(device)


def _generate(lm: LoadedModel, enc, max_new_tokens: int, seed: int) -> list[str]:
    torch.manual_seed(seed)
    with torch.no_grad():
        out = lm.model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    return lm.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


_OOM_MARKERS = ("out of memory", "not enough memory", "can't allocate memory")  # CUDA ve CPU ayırıcıları


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(m in msg for m in _OOM_MARKERS)


def generate_pair(
    lm: LoadedModel,
    prompt: str,
    interventions: Sequence[Intervention],
    max_new_tokens: int = 200,
    seed: int = 42,
) -> tuple[str, str, Literal["batch", "sequential"]]:
    """(orijinal, müdahaleli, kip). Kip: "batch" (tek yığın, 2 satır) ya da "sequential" (OOM yedeği)."""
    try:
        enc = _encode(lm, prompt, 2)
        with apply(lm, interventions, batch_index=1):
            a, b = _generate(lm, enc, max_new_tokens, seed)
        return a, b, "batch"
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if not _is_oom(exc):
            raise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    enc = _encode(lm, prompt, 1)
    (a,) = _generate(lm, enc, max_new_tokens, seed)
    with apply(lm, interventions):
        (b,) = _generate(lm, enc, max_new_tokens, seed)
    return a, b, "sequential"


# --- Test seti --------------------------------------------------------------------------------


class SuiteItem(BaseModel):
    category: str
    prompt: str
    check: Literal["code", "refusal", "contains"]
    answer: str | None = None
    answers: list[str] | None = None  # herhangi biri geçerse doğru (answer ile birlikte)

    def accepted(self) -> list[str]:
        return ([self.answer] if self.answer is not None else []) + list(self.answers or [])


def load_suite(path: Path | None = None) -> list[SuiteItem]:
    if path is None:
        text = files("neshter").joinpath("suites", "default.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return [SuiteItem.model_validate(d) for d in data]


@dataclass
class CategoryResult:
    category: str
    n: int
    k_orig: int
    k_int: int
    diff: Diff
    orig_ci: tuple[float, float] = field(init=False)  # Wilson %95 (orijinal oran)
    int_ci: tuple[float, float] = field(init=False)  # Wilson %95 (müdahaleli oran)

    def __post_init__(self) -> None:
        self.orig_ci = wilson(self.k_orig, self.n)[1:]
        self.int_ci = wilson(self.k_int, self.n)[1:]


def _norm_text(text: str) -> str:
    """NFKC (H₂O → H2O) + Türkçe duyarlı küçük harf (İ → i)."""
    return _lower(unicodedata.normalize("NFKC", text))


def answer_matches(answer: str, text: str) -> bool:
    """Cevap metinde bağımsız olarak geçiyor mu?

    Sayılar iki yandan sınırlıdır ("12", "120"de ya da "3.12"de değil, "12 ay"da eşleşir). Kelimelerde
    yalnızca baş sınır aranır: Türkçe ekler ("doğudan", "Ankara'dır") geçerli kalsın, ama "stanbul"
    gibi kelime içi parçalar eşleşmesin.
    """
    ans, txt = _norm_text(answer).strip(), _norm_text(text)
    if not ans:
        return False
    if ans.isdigit():
        pattern = rf"(?<![\w.,]){re.escape(ans)}(?![\w]|[.,]\d)"
    else:
        pattern = rf"(?<!\w){re.escape(ans)}"
    return re.search(pattern, txt) is not None


def _passes(item: SuiteItem, answer: str) -> bool:
    if item.check == "code":
        return detect_code(answer)
    if item.check == "refusal":
        return detect_refusal(answer)
    accepted = item.accepted()
    if not accepted:
        raise ValueError(f"'contains' öğesinde answer eksik: {item.prompt!r}")
    return any(answer_matches(a, answer) for a in accepted)


def run_suite(
    lm: LoadedModel,
    items: list[SuiteItem],
    interventions: Sequence[Intervention],
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    max_new_tokens: int = 200,
) -> list[CategoryResult]:
    """Kategori başına orijinal/müdahaleli başarı sayıları. İptalde kısmi sonuç atılır ([] döner)."""
    counts: dict[str, list[int]] = {}  # kategori -> [n, k_orig, k_int]; ekleme sırası = ilk görünüş
    total = len(items)
    for done, item in enumerate(items):
        if cancel is not None and cancel.is_set():
            return []
        orig, mod, _ = generate_pair(lm, item.prompt, interventions, max_new_tokens=max_new_tokens)
        c = counts.setdefault(item.category, [0, 0, 0])
        c[0] += 1
        c[1] += _passes(item, orig)
        c[2] += _passes(item, mod)
        if progress is not None:
            progress(done + 1, total)
    if cancel is not None and cancel.is_set():
        return []
    return [
        CategoryResult(cat, n, k_orig, k_int, rate_diff(k_orig, n, k_int, n))
        for cat, (n, k_orig, k_int) in counts.items()
    ]


# --- Özet -------------------------------------------------------------------------------------


@dataclass
class Summary:
    level: Literal["good", "mixed", "bad"]
    tr: str
    en: str


def _points(r: CategoryResult) -> float:
    return round(r.diff.diff * 100, 9)  # kayan nokta gürültüsü "tam 2 puan" sınırını bozmasın


def _significant_decrease(r: CategoryResult) -> bool:
    return not r.diff.uncertain and r.diff.hi < 0


def _names(rs: list[CategoryResult]) -> str:
    return ", ".join(f"{r.category} ({_points(r):.0f} puan)" for r in rs)


def summarize(results: list[CategoryResult], target: str) -> Summary:
    by_name = {r.category: r for r in results}
    if target not in by_name:
        raise ValueError(f"Hedef kategori '{target}' sonuçlarda yok.")
    if not _significant_decrease(by_name[target]):
        return Summary(
            "bad",
            "Hedef davranış belirgin şekilde azalmadı; bu müdahale istenen etkiyi vermedi.",
            "The target behavior did not drop clearly; this change did not have the intended effect.",
        )
    others = [r for r in results if r.category != target]
    hard = [r for r in others if _significant_decrease(r)]
    if hard:
        names = _names(hard)
        return Summary(
            "bad",
            f"Hedef davranış azaldı, ama şu kategorilerde de belirgin düşüş var: {names}.",
            f"The target behavior dropped, but these categories also dropped clearly: {names}.",
        )
    soft = [r for r in others if _points(r) < -2]
    if soft:
        names = _names(soft)
        return Summary(
            "mixed",
            f"Hedef davranış belirgin şekilde azaldı. Bazı kategorilerde hafif düşüş var ama "
            f"hata payı içinde: {names}. Daha fazla test önerilir.",
            f"The target behavior dropped clearly. Some categories dropped slightly but within "
            f"the margin of error: {names}. More tests are advised.",
        )
    return Summary(
        "good",
        "Hedef davranış belirgin şekilde azaldı, genel yetenekte anlamlı bir kayıp görünmüyor.",
        "The target behavior dropped clearly, with no meaningful loss in general ability.",
    )
