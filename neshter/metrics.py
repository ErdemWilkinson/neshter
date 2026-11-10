"""Metrikler: Wilson aralığı, oran farkı, KL etkisi, rastgele kontrollü gerçek etki, perplexity.

Tüm model çalıştırmaları ham metinle (sohbet şablonu yok), prompt başına tek tek (dolgu yok),
torch.no_grad altında ve use_cache=False ile yapılır. KL/CE float32'de log_softmax'tan hesaplanır.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from neshter.interventions import HeadAblation, Intervention, apply
from neshter.loader import LoadedModel


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


def _log_probs(lm: LoadedModel, text: str) -> torch.Tensor:
    """(T, V) float32 log-olasılıklar; mevcut kancalar (müdahaleler) etkindir."""
    enc = lm.tokenizer(text, return_tensors="pt").to(_device(lm))
    if enc["input_ids"].numel() == 0:
        raise ValueError("Boş metin (token yok) ölçülemez.")
    with torch.no_grad():
        out = lm.model(**enc, use_cache=False)
    return out.logits[0].to(torch.float32).log_softmax(-1)


def kl_effect(
    lm: LoadedModel,
    prompts: list[str],
    interventions: Sequence[Intervention],
    last_k: int = 16,
) -> list[float]:
    """Prompt başına ortalama KL(orijinal || müdahaleli); son last_k pozisyon (kısaysa hepsi)."""
    if last_k < 1:
        raise ValueError(f"last_k en az 1 olmalı: {last_k}.")
    result: list[float] = []
    for prompt in prompts:
        logp = _log_probs(lm, prompt)[-last_k:]
        with apply(lm, interventions):
            logq = _log_probs(lm, prompt)[-last_k:]
        kl = (logp.exp() * (logp - logq)).sum(-1)  # (k,)
        result.append(kl.mean().item())
    return result


@dataclass
class EffectResult:
    kl: float
    kl_std: float
    random_kl: float
    n_prompts: int


def real_effect(
    lm: LoadedModel,
    heads: list[HeadAblation],
    prompts: list[str],
    n_random: int = 3,
    seed: int = 0,
) -> EffectResult:
    """Seçili head'lerin KL etkisi ve aynı sayıda rastgele head'lerle kontrol (seçilenler hariç)."""
    if not prompts:
        raise ValueError("En az bir prompt gerekli.")
    if n_random < 1:
        raise ValueError(f"n_random en az 1 olmalı: {n_random}.")

    per_prompt = kl_effect(lm, prompts, heads)
    kl = sum(per_prompt) / len(per_prompt)
    kl_std = math.sqrt(sum((v - kl) ** 2 for v in per_prompt) / len(per_prompt))  # popülasyon

    chosen = set(heads)
    pool = [
        HeadAblation(layer, head)
        for layer in range(lm.arch.n_layers)
        for head in range(lm.arch.n_heads)
        if HeadAblation(layer, head) not in chosen
    ]
    size = min(len(heads), len(pool))
    rng = random.Random(seed)
    control: list[float] = []
    for _ in range(n_random):
        sample = rng.sample(pool, size)
        values = kl_effect(lm, prompts, sample)
        control.append(sum(values) / len(values))

    return EffectResult(
        kl=kl, kl_std=kl_std, random_kl=sum(control) / len(control), n_prompts=len(prompts)
    )


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
