"""Metrikler: Wilson aralığı, oran farkı, KL etkisi, rastgele kontrollü gerçek etki, perplexity.

Tüm model çalıştırmaları ham metinle (sohbet şablonu yok), prompt başına tek tek (dolgu yok),
torch.no_grad altında ve use_cache=False ile yapılır. KL/CE float32'de log_softmax'tan hesaplanır.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import torch
import yaml

from neshter.interventions import DirectionAblation, HeadAblation, Intervention, apply
from neshter.loader import LoadedModel
from neshter.progress import ProgressFn, check_cancel


def load_neutral() -> list[str]:
    """Gerçek etki (KL) ve perplexity için ayrılmış nötr istekler (neshter/concepts/neutral.yaml).

    Bu istekler test setinde ve kavram örneklerinde yer almaz; asla test setinden türetilmez.
    """
    text = files("neshter.concepts").joinpath("neutral.yaml").read_text(encoding="utf-8")
    prompts = yaml.safe_load(text)["prompts"]
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("neutral.yaml 'prompts' boş olmayan metinlerden oluşan bir liste olmalı.")
    return prompts


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson skor aralığı: (p, alt, üst). n == 0 -> (0.0, 0.0, 1.0)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"Geçersiz sayımlar: k={k}, n={n} (0 <= k <= n olmalı).")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return p, lo, hi


@dataclass
class Diff:
    diff: float
    lo: float
    hi: float
    uncertain: bool


def rate_diff(k1: int, n1: int, k2: int, n2: int) -> Diff:
    """p2 - p1 ve %95 aralığı (normal yaklaşım); aralık 0'ı kapsıyorsa belirsiz."""
    for k, n in ((k1, n1), (k2, n2)):
        if n <= 0 or not 0 <= k <= n:
            raise ValueError(f"Geçersiz sayımlar: k={k}, n={n} (n > 0 ve 0 <= k <= n olmalı).")
    p1, p2 = k1 / n1, k2 / n2
    diff = p2 - p1
    half = 1.96 * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = diff - half, diff + half
    return Diff(diff=diff, lo=lo, hi=hi, uncertain=lo <= 0 <= hi)


def _device(lm: LoadedModel) -> torch.device:
    return next(lm.model.parameters()).device


def _log_probs(lm: LoadedModel, text: str, last_k: int) -> torch.Tensor:
    """(min(T, last_k), V) float32 log-olasılıklar, son pozisyonlar; mevcut kancalar etkindir.

    Yalnızca son last_k pozisyonun logit'i üretilir (logits_to_keep): T × V float32 tutulmaz.
    """
    enc = lm.tokenizer(text, return_tensors="pt").to(_device(lm))
    if enc["input_ids"].numel() == 0:
        raise ValueError("Boş metin (token yok) ölçülemez.")
    with torch.no_grad():
        out = lm.model(**enc, use_cache=False, logits_to_keep=last_k)
    return out.logits[0, -last_k:].to(torch.float32).log_softmax(-1)


def _kl_from(
    lm: LoadedModel,
    base: list[torch.Tensor],
    prompts: list[str],
    interventions: Sequence[Intervention],
    last_k: int,
) -> list[float]:
    """Önceden hesaplanmış orijinal log-olasılıklara (base) karşı prompt başına ortalama KL."""
    result: list[float] = []
    with apply(lm, interventions):
        for logp, prompt in zip(base, prompts):
            logq = _log_probs(lm, prompt, last_k)
            result.append((logp.exp() * (logp - logq)).sum(-1).mean().item())
    return result


def kl_effect(
    lm: LoadedModel,
    prompts: list[str],
    interventions: Sequence[Intervention],
    last_k: int = 16,
) -> list[float]:
    """Prompt başına ortalama KL(orijinal || müdahaleli); son last_k pozisyon (kısaysa hepsi)."""
    if last_k < 1:
        raise ValueError(f"last_k en az 1 olmalı: {last_k}.")
    base = [_log_probs(lm, p, last_k) for p in prompts]
    return _kl_from(lm, base, prompts, interventions, last_k)


DEFAULT_N_RANDOM = 30  # rastgele kontrol çekilişi sayısı (kullanıcıya dönük alt sınır sunucuda: 20)


@dataclass
class EffectResult:
    kl: float
    kl_std: float
    kl_se: float  # kl_std / sqrt(n_prompts)
    n_prompts: int
    random_kl: float  # çekilişlerin ortalaması
    random_kls: list[float]
    random_p05: float
    random_p95: float
    percentile: float  # 100 * #{rastgele < kl} / N
    p_value: float  # (1 + #{rastgele >= kl}) / (1 + N): permütasyon p
    n_random: int
    layer_matched: bool  # False: bir katmanda havuz yetmedi, kalan en yakın katmanlardan alındı


def _dedupe(interventions: Sequence[Intervention]) -> list[Intervention]:
    """Aynı head iki kez sayılmasın (yönler olduğu gibi kalır)."""
    seen: set[HeadAblation] = set()
    out: list[Intervention] = []
    for iv in interventions:
        if isinstance(iv, HeadAblation):
            if iv in seen:
                continue
            seen.add(iv)
        out.append(iv)
    return out


def layer_matched_heads(
    lm: LoadedModel, heads: list[HeadAblation], rng: random.Random
) -> tuple[list[HeadAblation], bool]:
    """Her katmandaki seçili head sayısı kadar, aynı katmandan seçilmemiş head örnekler (tekrarsız).

    Bir katmanın havuzu yetmezse eksik, |katman farkı| en küçük katmanlardan (eşitlikte alt katman
    önce) tamamlanır ve `matched=False` döner. Tüm modelde seçilmemiş head yetmezse ValueError.
    """
    n_layers, n_heads = lm.arch.n_layers, lm.arch.n_heads
    chosen = set(heads)
    pools = {
        layer: [HeadAblation(layer, h) for h in range(n_heads) if HeadAblation(layer, h) not in chosen]
        for layer in range(n_layers)
    }
    if sum(len(p) for p in pools.values()) < len(heads):
        raise ValueError("Rastgele kontrol için yeterli seçilmemiş head yok.")
    counts: dict[int, int] = {}
    for h in heads:
        counts[h.layer] = counts.get(h.layer, 0) + 1

    picked: list[HeadAblation] = []
    deficits: dict[int, int] = {}
    for layer in sorted(counts):  # önce her katman kendi havuzundan
        take = min(counts[layer], len(pools[layer]))
        got = rng.sample(pools[layer], take)
        pools[layer] = [x for x in pools[layer] if x not in got]
        picked += got
        if take < counts[layer]:
            deficits[layer] = counts[layer] - take
    for layer in sorted(deficits):  # sonra eksikler en yakın katmanlardan
        need = deficits[layer]
        for other in sorted(range(n_layers), key=lambda x: (abs(x - layer), x)):
            if need == 0:
                break
            if other == layer or not pools[other]:
                continue
            got = rng.sample(pools[other], min(need, len(pools[other])))
            pools[other] = [x for x in pools[other] if x not in got]
            picked += got
            need -= len(got)
    return picked, not deficits


def random_control(
    lm: LoadedModel,
    interventions: Sequence[Intervention],
    rng: random.Random,
    gen: torch.Generator,
) -> tuple[list[Intervention], bool]:
    """Aynı biçimde rastgele kontrol — "rastgele kontrol"ün tek tanımı. (kontrol, katman_eşleşti_mi) döner.

    Her HeadAblation yerine katman eşleşmeli, seçilmemiş rastgele head (`layer_matched_heads`);
    her DirectionAblation yerine rastgele izotropik birim yön.

    Yön kontrolü: v0.1.1'de rastgele izotropik birim vektör kalır. Daha iyi kontrol (nötr
    istemlerin rastgele bölünmesinden ortalama fark yönü, AT §4) v0.2'dedir.
    """
    heads = [iv for iv in interventions if isinstance(iv, HeadAblation)]
    n_dirs = sum(isinstance(iv, DirectionAblation) for iv in interventions)
    picked, matched = layer_matched_heads(lm, heads, rng)
    control: list[Intervention] = list(picked)
    for _ in range(n_dirs):
        control.append(DirectionAblation(torch.randn(lm.arch.d_model, generator=gen)))
    return control, matched


def recipe_effect(
    lm: LoadedModel,
    interventions: Sequence[Intervention],
    prompts: list[str],
    n_random: int = DEFAULT_N_RANDOM,
    seed: int = 0,
    last_k: int = 16,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> EffectResult:
    """Müdahalelerin (head ve/veya yön) KL etkisi; n_random katman eşleşmeli rastgele çekilişle karşılaştırılır.

    Orijinal log-olasılıklar prompt başına bir kez hesaplanır. Rastgelelik yalnız yerel
    random.Random / torch.Generator'dan gelir (global tohum etkilenmez). Çekilişler içinde
    tekrar yok; çekilişler arasında tekrar olabilir.
    """
    if not prompts:
        raise ValueError("En az bir prompt gerekli.")
    if n_random < 1:
        raise ValueError(f"n_random en az 1 olmalı: {n_random}.")
    check_cancel(cancel)
    interventions = _dedupe(interventions)
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    drawn = [random_control(lm, interventions, rng, gen) for _ in range(n_random)]  # önce doğrula
    layer_matched = all(matched for _, matched in drawn)

    base = [_log_probs(lm, p, last_k) for p in prompts]
    per_prompt = _kl_from(lm, base, prompts, interventions, last_k)
    kl = sum(per_prompt) / len(per_prompt)
    kl_std = math.sqrt(sum((v - kl) ** 2 for v in per_prompt) / len(per_prompt))  # popülasyon
    if progress is not None:
        progress(0.05, "Seçili müdahalenin etkisi ölçüldü")

    random_kls: list[float] = []
    for i, (control, _) in enumerate(drawn):
        check_cancel(cancel)
        values = _kl_from(lm, base, prompts, control, last_k)
        random_kls.append(sum(values) / len(values))
        if progress is not None:
            progress(0.05 + 0.95 * (i + 1) / n_random, f"Rastgele kontrol {i + 1}/{n_random}")
    p05, p95 = (float(x) for x in np.percentile(random_kls, [5, 95]))
    return EffectResult(
        kl=kl,
        kl_std=kl_std,
        kl_se=kl_std / math.sqrt(len(prompts)),
        n_prompts=len(prompts),
        random_kl=sum(random_kls) / n_random,
        random_kls=random_kls,
        random_p05=p05,
        random_p95=p95,
        percentile=100.0 * sum(r < kl for r in random_kls) / n_random,
        p_value=(1 + sum(r >= kl for r in random_kls)) / (1 + n_random),
        n_random=n_random,
        layer_matched=layer_matched,
    )


def real_effect(
    lm: LoadedModel,
    heads: list[HeadAblation],
    prompts: list[str],
    n_random: int = DEFAULT_N_RANDOM,
    seed: int = 0,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> EffectResult:
    """Seçili head'lerin KL etkisi; katman eşleşmeli rastgele head kümeleriyle (seçilenler hariç) kontrol."""
    return recipe_effect(lm, heads, prompts, n_random=n_random, seed=seed, progress=progress, cancel=cancel)


def perplexity(
    lm: LoadedModel, texts: list[str], interventions: Sequence[Intervention] = ()
) -> float:
    """exp(tüm metinlerin tüm tahmin edilen token'larının ortalama çapraz entropisi)."""
    total, count = 0.0, 0
    with apply(lm, interventions):
        for text in texts:
            enc = lm.tokenizer(text, return_tensors="pt").to(_device(lm))
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue  # tahmin edilecek hedef yok
            with torch.no_grad():
                logits = lm.model(**enc, use_cache=False).logits[0, :-1].to(torch.float32)
            nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
            total += nll.item()
            count += ids.shape[1] - 1
    if count == 0:
        raise ValueError("Perplexity için tahmin edilecek token yok (bir metin ≥ 2 token olmalı).")
    return math.exp(total / count)
