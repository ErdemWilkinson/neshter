"""Yan yana karşılaştırma, kategori bazlı test seti ve düz dilde özet.

Üretim her zaman greedy'dir (do_sample=False). Orijinal ve müdahaleli cevap aynı çağrıdan gelir:
iki satırlık yığında yalnızca 1. satıra müdahale uygulanır (apply(..., batch_index=1)). Bellek
yetmezse aynı ayarlarla sıralı üretime geçilir; çıktılar aynıdır.

Yön içeren tarifler (DirectionAblation) kayıtta ağırlıklara gömülür; bu yüzden karşılaştırma ve test
seti, kullanıcının gerçekten kaydedeceği modeli ölçmek için gömülü ve kayıt dtype'ına yuvarlanmış
ağırlıklarla çalışır ("baked" kipi). Gömülü ve özgün ağırlıklar tek bir ileri geçişte karışamayacağı için
bu kipte üretim her zaman sıralıdır (önce özgün, sonra gömülü). Yalnız-head tarifleri kanca yolunda kalır.
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

from neshter.concepts.detectors import DETECTORS, _lower, is_broken
from neshter.interventions import DirectionAblation, Intervention, apply
from neshter.loader import LoadedModel
from neshter.metrics import ALPHA, Diff, holm, min_detectable_drop, paired_diff, wilson

# --- Üretim -----------------------------------------------------------------------------------


def _prompt_text(lm: LoadedModel, prompt: str) -> tuple[str, bool]:
    """(modele verilecek metin, sohbet şablonlu mu)."""
    tok = lm.tokenizer
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return text, True
    return prompt, False


def _encode(lm: LoadedModel, prompt: str, copies: int):
    """Sohbet şablonlu (tek kullanıcı mesajı) ya da ham girdi; `copies` kopya, sola dolgulu."""
    tok = lm.tokenizer
    text, templated = _prompt_text(lm, prompt)
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


def _has_direction(interventions: Sequence[Intervention]) -> bool:
    return any(isinstance(iv, DirectionAblation) for iv in interventions)


def generate_pair(
    lm: LoadedModel,
    prompt: str,
    interventions: Sequence[Intervention],
    max_new_tokens: int = 200,
    seed: int = 42,
    baked: bool | None = None,
) -> tuple[str, str, Literal["batch", "sequential", "baked"]]:
    """(orijinal, müdahaleli, kip).

    Kip: "batch" (tek yığın, 2 satır, kancalı), "sequential" (OOM yedeği, kancalı) ya da "baked".
    `baked=None`: otomatik, yön (DirectionAblation) varsa gömülü. Gömülü yolda orijinal kancasız/gömüsüz
    üretilir; sonra bake(..., round_to=save_dtype) ile müdahaleli cevap kaydedilecek modelin ağırlıklarıyla
    üretilir ve undo() her durumda (hata, iptal) `finally`de çalışır. Gömülü ve özgün ağırlıklar tek yığında
    karışamaz, bu yüzden bu yol her zaman sıralıdır.
    """
    if _has_direction(interventions) if baked is None else baked:
        from neshter.export import bake, ensure_bake_ram, save_dtype  # export, compare'i içe aktarır: yerel import

        ensure_bake_ram(lm, interventions, what="Gömme")
        enc = _encode(lm, prompt, 1)
        (a,) = _generate(lm, enc, max_new_tokens, seed)
        undo = bake(lm, list(interventions), round_to=save_dtype(lm))
        try:
            (b,) = _generate(lm, enc, max_new_tokens, seed)
        finally:
            undo()
        return a, b, "baked"
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


def answer_nll(lm: LoadedModel, prompt: str, answer: str) -> float | None:
    """Cevap token'larının, (şablonlu) istemden sonra, token başına ortalama NLL'i (nat).

    Orijinal modelle ölçülür: çağıran, müdahale kancaları/gömülü ağırlıklar geri alınmışken (apply ve
    bake dışında) çağırmalıdır. Cevap ya da istem 0 token ise None.
    """
    tok = lm.tokenizer
    text, templated = _prompt_text(lm, prompt)
    p_ids = tok(text, add_special_tokens=not templated)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    if not a_ids or not p_ids:  # ilk cevap token'ını tahmin edecek konum yok
        return None
    device = next(lm.model.parameters()).device
    ids = torch.tensor([p_ids + a_ids], device=device)
    with torch.no_grad():
        logits = lm.model(input_ids=ids, use_cache=False, logits_to_keep=len(a_ids) + 1).logits[0, :-1]
    target = torch.tensor(a_ids, device=device)
    return torch.nn.functional.cross_entropy(logits.to(torch.float32), target).item()


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
    """Kategori başına eşli sonuç. a = ikisi de geçti, b = yalnız orijinal, c = yalnız müdahaleli,
    d = ikisi de kaldı (k_orig = a + b, k_int = a + c). broken_*: bozuk cevap sayısı."""

    category: str
    n: int
    k_orig: int
    k_int: int
    a: int
    b: int
    c: int
    d: int
    broken_orig: int
    broken_int: int
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
    if item.check in DETECTORS:
        return DETECTORS[item.check](answer)
    accepted = item.accepted()
    if not accepted:
        raise ValueError(f"'contains' öğesinde answer eksik: {item.prompt!r}")
    return any(answer_matches(a, answer) for a in accepted)


def _outcome(item: SuiteItem, answer: str, broken: bool, is_target: bool, original: bool) -> bool:
    """Bozuk cevap kesimi asla başarılı göstermez.

    Müdahaleli bozuk: hedefte geçti (davranış sürüyor), diğerlerinde kaldı (yetenek kaybı).
    Orijinal bozuk: hedefte kaldı (b hücresine düşmez), diğerlerinde geçti (c hücresine düşmez).
    """
    if broken:
        return is_target != original
    return _passes(item, answer)


def _apply_holm(results: list[CategoryResult], target: str | None) -> None:
    """Hedef dışı kategorilerde Holm düzeltmeli p'yi (p_adj) ve belirsizliği doldurur; hedefe dokunmaz."""
    others = [r for r in results if r.category != target]
    for r, p_adj in zip(others, holm([r.diff.p_value for r in others])):
        r.diff.p_adj = p_adj
        r.diff.uncertain = p_adj >= ALPHA


def _tally(
    cells: dict[str, list[int]],
    item: SuiteItem,
    target: str | None,
    orig: str,
    mod: str,
    nll_o: float | None,
    nll_m: float | None,
) -> None:
    broken_o, broken_m = is_broken(orig, nll_o), is_broken(mod, nll_m)
    is_target = item.category == target
    pass_o = _outcome(item, orig, broken_o, is_target, original=True)
    pass_m = _outcome(item, mod, broken_m, is_target, original=False)
    cell = cells.setdefault(item.category, [0, 0, 0, 0, 0, 0])
    cell[(0 if pass_m else 1) if pass_o else (2 if pass_m else 3)] += 1
    cell[4] += broken_o
    cell[5] += broken_m


def _finish(cells: dict[str, list[int]], target: str | None) -> list[CategoryResult]:
    results = [
        CategoryResult(cat, a + b + c + d, a + b, a + c, a, b, c, d, bo, bi, paired_diff(a, b, c, d))
        for cat, (a, b, c, d, bo, bi) in cells.items()
    ]
    _apply_holm(results, target)
    return results


def _suite_hooked(
    lm: LoadedModel,
    items: list[SuiteItem],
    interventions: Sequence[Intervention],
    progress: Callable[[int, int], None] | None,
    cancel: threading.Event | None,
    max_new_tokens: int,
    target: str | None,
) -> list[CategoryResult]:
    cells: dict[str, list[int]] = {}  # kategori -> [a, b, c, d, bozuk_orig, bozuk_int]; sıra = ilk görünüş
    total = len(items)
    for done, item in enumerate(items):
        if cancel is not None and cancel.is_set():
            return []
        orig, mod, _ = generate_pair(lm, item.prompt, interventions, max_new_tokens=max_new_tokens, baked=False)
        nll_o = answer_nll(lm, item.prompt, orig)
        nll_m = answer_nll(lm, item.prompt, mod)
        _tally(cells, item, target, orig, mod, nll_o, nll_m)
        if progress is not None:
            progress(done + 1, total)
    if cancel is not None and cancel.is_set():
        return []
    return _finish(cells, target)


def _suite_baked(
    lm: LoadedModel,
    items: list[SuiteItem],
    interventions: Sequence[Intervention],
    progress: Callable[[int, int], None] | None,
    cancel: threading.Event | None,
    max_new_tokens: int,
    target: str | None,
    seed: int = 42,
) -> list[CategoryResult]:
    """İki geçiş: önce tüm orijinaller, sonra BİR kez bake, tüm gömülü cevaplar, undo (finally) ve en son,
    ağırlıklar geri gelmişken, tüm NLL'ler (answer_nll orijinal ağırlıklarla ölçülür)."""
    from neshter.export import bake, ensure_bake_ram, save_dtype  # export, compare'i içe aktarır: yerel import

    ensure_bake_ram(lm, interventions, what="Gömme")  # uzun ilk geçişten ÖNCE

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def gen(item: SuiteItem) -> str:
        (text,) = _generate(lm, _encode(lm, item.prompt, 1), max_new_tokens, seed)
        return text

    n = len(items)
    total = 2 * n
    origs: list[str] = []
    for i, item in enumerate(items):
        if cancelled():
            return []
        origs.append(gen(item))
        if progress is not None:
            progress(i + 1, total)
    undo = bake(lm, list(interventions), round_to=save_dtype(lm))
    mods: list[str] = []
    try:
        for i, item in enumerate(items):
            if cancelled():
                return []
            mods.append(gen(item))
            if progress is not None:
                progress(n + i + 1, total)
    finally:
        undo()
    cells: dict[str, list[int]] = {}
    for item, orig, mod in zip(items, origs, mods):
        if cancelled():
            return []
        nll_o = answer_nll(lm, item.prompt, orig)
        nll_m = answer_nll(lm, item.prompt, mod)
        _tally(cells, item, target, orig, mod, nll_o, nll_m)
    if cancelled():
        return []
    return _finish(cells, target)


def run_suite(
    lm: LoadedModel,
    items: list[SuiteItem],
    interventions: Sequence[Intervention],
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    max_new_tokens: int = 200,
    target: str | None = None,
    baked: bool | None = None,
) -> list[CategoryResult]:
    """Kategori başına eşli (orijinal, müdahaleli) sonuç tablosu. İptalde kısmi sonuç atılır ([] döner).

    `progress(done, total)`: tamamlanan adım sayısı. Kancalı yolda adım = prompt (total = n). Gömülü yolda
    (`baked=None`: yön varsa) iki geçiş vardır, total = 2n: geçiş 1 (orijinaller) 1..n, geçiş 2 (kayıt
    dtype'ına yuvarlanmış gömülü ağırlıklar) n+1..2n. Bozukluk için NLL her iki yolda da müdahale
    kancaları/gömülü ağırlıklar kalkmışken orijinal modelle ölçülür. Holm düzeltmesi hedef dışı
    kategorilere uygulanır (hedef yoksa tümüne). Not: üretim dışındaki ölçümler (KL, perplexity:
    _recipe_metrics) gömülü ağırlıklara değil kancalara dayanır.
    """
    if _has_direction(interventions) if baked is None else baked:
        return _suite_baked(lm, items, interventions, progress, cancel, max_new_tokens, target)
    return _suite_hooked(lm, items, interventions, progress, cancel, max_new_tokens, target)


# --- Özet -------------------------------------------------------------------------------------


@dataclass
class Summary:
    level: Literal["good", "mixed", "bad"]
    tr: str
    en: str
    mde_points: int  # testin görebileceği en küçük düşüş (puan): hedef dışı en küçük n, m = hedef dışı sayısı


def _points(r: CategoryResult) -> float:
    return round(r.diff.diff * 100, 9)  # kayan nokta gürültüsü "tam 2 puan" sınırını bozmasın


def _significant_decrease(r: CategoryResult) -> bool:
    """Anlamlı düşüş: kullanılan p < 0.05 ve fark negatif. Hedef dışı kategoride p = Holm düzeltmeli p_adj
    (run_suite doldurur), hedefte ham McNemar p."""
    p_used = r.diff.p_adj if r.diff.p_adj is not None else r.diff.p_value
    return p_used is not None and p_used < ALPHA and r.diff.diff < 0


def _names(rs: list[CategoryResult]) -> str:
    return ", ".join(f"{r.category} ({_points(r):.0f} puan)" for r in rs)


def summarize(results: list[CategoryResult], target: str) -> Summary:
    """Düz dilde özet. Anlamlılık run_suite'in doldurduğu p / p_adj değerlerinden okunur."""
    by_name = {r.category: r for r in results}
    if target not in by_name:
        raise ValueError(f"Hedef kategori '{target}' sonuçlarda yok.")
    tgt = by_name[target]
    others = [r for r in results if r.category != target]
    min_n = min((r.n for r in others), default=tgt.n)
    mde = round(100 * min_detectable_drop(min_n, m=len(others)))
    mde_tr = f"Bu test yaklaşık {mde} puandan küçük düşüşleri göremez."
    mde_en = f"This test cannot detect drops smaller than about {mde} points."
    broken_tr = broken_en = ""
    if tgt.broken_int > tgt.broken_orig:
        bi, bo = tgt.broken_int, tgt.broken_orig
        broken_tr = f" Hedef kategoride {bi} (orijinalde {bo}) cevap bozuk çıktı (bozuk cevaplar başarı sayılmaz)."
        broken_en = (
            f" {bi} (original: {bo}) answers in the target category were broken "
            "(broken answers never count as success)."
        )

    def out(level: Literal["good", "mixed", "bad"], tr: str, en: str) -> Summary:
        return Summary(level, tr + broken_tr, en + broken_en, mde)

    if not _significant_decrease(tgt):
        return out(
            "bad",
            "Hedef davranış belirgin şekilde azalmadı; bu müdahale istenen etkiyi vermedi.",
            "The target behavior did not drop clearly; this change did not have the intended effect.",
        )
    hard = [r for r in others if _significant_decrease(r)]
    if hard:
        names = _names(hard)
        return out(
            "bad",
            f"Hedef davranış azaldı, ama şu kategorilerde de belirgin düşüş var: {names}.",
            f"The target behavior dropped, but these categories also dropped clearly: {names}.",
        )
    # Özet yalnızca gerçekten denetlenen kontrol kategorilerini iddia eder (quick kipte yalnız "general").
    checked = ", ".join(r.category for r in others)
    soft = [r for r in others if _points(r) < -2]
    if soft:
        names = _names(soft)
        return out(
            "mixed",
            f"Hedef davranış belirgin şekilde azaldı. Kontrol edilen kategorilerden ({checked}) bazılarında hafif "
            f"düşüş var ama hata payı içinde: {names}. Daha fazla test önerilir. {mde_tr}",
            f"The target behavior dropped clearly. Of the checked categories ({checked}), some dropped slightly "
            f"but within the margin of error: {names}. More tests are advised. {mde_en}",
        )
    if not others:
        return out(
            "good",
            "Hedef davranış belirgin şekilde azaldı. Kontrol kategorisi denetlenmedi.",
            "The target behavior dropped clearly. No control category was checked.",
        )
    return out(
        "good",
        f"Hedef davranış belirgin şekilde azaldı; kontrol edilen diğer kategorilerde ({checked}) anlamlı bir "
        f"düşüş görülmedi. {mde_tr}",
        f"The target behavior dropped clearly; no other checked category ({checked}) dropped significantly. "
        f"{mde_en}",
    )
