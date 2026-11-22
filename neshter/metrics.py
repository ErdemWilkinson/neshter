"""Metrikler: Wilson aralığı, eşli fark (McNemar kesin + Newcombe, Holm), KL etkisi, rastgele kontrollü gerçek etki, perplexity.

Tüm model çalıştırmaları ham metinle (sohbet şablonu yok), torch.no_grad altında ve use_cache=False ile
yapılır. KL istekleri KL_BATCH'lik sola dolgulu yığınlarla (dikkat maskesi + maskeden position_ids) işlenir;
her istek yalnızca kendi gerçek pozisyonları üzerinden ortalanır, sonuç tek tek çalıştırmayla aynıdır.
Perplexity metin başına tek geçiştir. KL/CE float32'de log_softmax'tan hesaplanır.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

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
    """Eşli fark p_int − p_orig, Newcombe eşli skor aralığı ve McNemar kesin p.

    `uncertain`: kullanılan p ≥ 0.05 (hedef kategoride ham p, diğerlerinde Holm düzeltmeli p_adj).
    """

    diff: float
    lo: float
    hi: float
    uncertain: bool
    p_value: float | None = None  # McNemar kesin, iki yönlü
    p_adj: float | None = None  # Holm düzeltmeli (yalnız hedef dışı kategoriler; run_suite doldurur)


ALPHA = 0.05


def _check_cells(*cells: int) -> None:
    if any(x < 0 for x in cells):
        raise ValueError(f"Geçersiz sayımlar: {cells} (negatif olamaz).")


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar kesin testi (iki yönlü): p = min(1, 2 · Σ_{i ≤ min(b,c)} C(b+c, i) · 0.5^(b+c)); b+c = 0 → 1."""
    _check_cells(b, c)
    nd = b + c
    if nd == 0:
        return 1.0
    tail = sum(math.comb(nd, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**nd)  # tam sayı / 2^nd: büyük nd'de de taşma yok


def paired_ci(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Δ = p_int − p_orig için %95 Newcombe eşli skor aralığı (yöntem 10; Wilson sınırları + φ düzeltmesi).

    Hücreler: a = ikisi de geçti, b = yalnız orijinal, c = yalnız müdahaleli, d = ikisi de kaldı.
    """
    _check_cells(a, b, c, d)
    n = a + b + c + d
    if n == 0:
        raise ValueError("Eşli aralık için en az bir prompt gerekli.")
    p1, l1, u1 = wilson(a + b, n)
    p2, l2, u2 = wilson(a + c, n)
    delta = p2 - p1
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else (a * d - b * c) / math.sqrt(den)
    lo = delta - math.sqrt(max(0.0, (p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2))
    hi = delta + math.sqrt(max(0.0, (u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2))
    return max(-1.0, lo), min(1.0, hi)


def paired_diff(a: int, b: int, c: int, d: int) -> Diff:
    """Eşli fark (c − b)/n, Newcombe aralığı ve McNemar p; p ≥ 0.05 ise belirsiz (Holm öncesi)."""
    lo, hi = paired_ci(a, b, c, d)
    p = mcnemar_exact(b, c)
    return Diff(diff=(c - b) / (a + b + c + d), lo=lo, hi=hi, uncertain=p >= ALPHA, p_value=p)


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni düzeltmesi; sonuç girdi sırasıyla döner."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):  # rank = j − 1
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def min_detectable_drop(n: int, alpha: float = ALPHA, m: int = 1) -> float:
    """En küçük saptanabilir düşüş: en iyi durumda (c = 0) McNemar'ın α/m'de anlamlı olması için gereken
    en az b = k, oran olarak k/n (en çok 1). m en az 1 sayılır."""
    if n <= 0:
        raise ValueError(f"n pozitif olmalı: {n}.")
    m = max(1, m)
    k = 1
    while 2 * 0.5**k >= alpha / m:
        k += 1
    return min(1.0, k / n)


def _device(lm: LoadedModel) -> torch.device:
    return next(lm.model.parameters()).device


KL_BATCH = 8  # KL ileri geçişlerinde aynı anda işlenen istek sayısı (hunt.HUNT_BATCH ile aynı)


def _log_probs_batch(
    lm: LoadedModel, texts: Sequence[str], last_k: int, cancel: threading.Event | None = None
) -> list[torch.Tensor]:
    """İstek başına (min(T_i, last_k), V) float32 log-olasılıklar, son gerçek pozisyonlar; kancalar etkindir.

    KL_BATCH'lik yığınlar sola dolgulanır; dikkat maskesi dolguyu gizler, position_ids maskenin kümülatif
    toplamından gelir (ilk gerçek token 0): her satır dolgusuz tek geçişle aynı sayıları verir. Yalnızca son
    last_k pozisyonun logit'i üretilir (logits_to_keep); dolgu pozisyonları atılır.
    """
    device = _device(lm)
    out: list[torch.Tensor] = []
    for start in range(0, len(texts), KL_BATCH):
        check_cancel(cancel)
        chunk = list(texts[start : start + KL_BATCH])
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True, padding_side="left")
        lengths = enc["attention_mask"].sum(-1).tolist()
        if enc["input_ids"].numel() == 0 or min(lengths) == 0:
            raise ValueError("Boş metin (token yok) ölçülemez.")
        enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        enc = enc.to(device)
        with torch.no_grad():
            logits = lm.model(**enc, use_cache=False, logits_to_keep=last_k).logits
        logp = logits.to(torch.float32).log_softmax(-1)  # [B, min(T_pad, last_k), V]
        k = logp.shape[1]
        for i, n in enumerate(lengths):
            out.append(logp[i, k - min(n, k) :])
    return out


def _log_probs(lm: LoadedModel, text: str, last_k: int) -> torch.Tensor:
    """Tek istek: (min(T, last_k), V) float32 log-olasılıklar (bkz. _log_probs_batch)."""
    return _log_probs_batch(lm, [text], last_k)[0]


def _kl_from(
    lm: LoadedModel,
    base: list[torch.Tensor],
    prompts: list[str],
    interventions: Sequence[Intervention],
    last_k: int,
    cancel: threading.Event | None = None,
) -> list[float]:
    """Önceden hesaplanmış orijinal log-olasılıklara (base) karşı prompt başına ortalama KL (yığınlı).

    KL her KL_BATCH yığınında hemen indirgenir: müdahaleli log-olasılıklar tüm istekler için tutulmaz
    (n_prompts × last_k × V float32 ek tepe yok).
    """
    out: list[float] = []
    with apply(lm, interventions):
        for start in range(0, len(prompts), KL_BATCH):
            cut = _log_probs_batch(lm, prompts[start : start + KL_BATCH], last_k, cancel)
            for logp, logq in zip(base[start : start + KL_BATCH], cut):
                out.append((logp.exp() * (logp - logq)).sum(-1).mean().item())
    return out


def kl_effect(
    lm: LoadedModel,
    prompts: list[str],
    interventions: Sequence[Intervention],
    last_k: int = 16,
) -> list[float]:
    """Prompt başına ortalama KL(orijinal || müdahaleli); son last_k pozisyon (kısaysa hepsi)."""
    if last_k < 1:
        raise ValueError(f"last_k en az 1 olmalı: {last_k}.")
    base = _log_probs_batch(lm, prompts, last_k)
    return _kl_from(lm, base, prompts, interventions, last_k)


DEFAULT_N_RANDOM = 30  # suite/tarif yolunun sabit rastgele kontrol çekiş sayısı (/api/effect sıralı test kullanır)


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
    layer_matched: bool  # tüm HEAD kontrolleri katman eşleşmeli mi (False: havuz yetmedi, en yakın katmanlardan tamamlandı)
    direction_control: Literal["empirical_diff", "isotropic"] | None  # yön içeren tarifte "empirical_diff" (≥2 istem, model içi fark vektörü) ya da "isotropic" (yedek); aksi halde None


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


def _match_direction_layer(resid: torch.Tensor, direction: torch.Tensor, window: Sequence[int]) -> int:
    """Hedef yönün modelin artık akımında en çok varyans gösterdiği aday katmanı seçer."""
    r = direction.detach().to(device="cpu", dtype=torch.float32)
    norm = r.norm()
    if norm <= 0:
        return window[len(window) // 2]
    r = r / norm
    return max(window, key=lambda layer: float((resid[layer] @ r).var(unbiased=False)))


def _sample_empirical_direction(acts: torch.Tensor, rng: random.Random) -> torch.Tensor | None:
    """Aynı katmandaki istem aktivasyonlarını ([N, d]) rastgele ikiye bölüp birim ortalama fark yönü üretir."""
    n = acts.shape[0]
    if n < 2:
        return None
    mid = n // 2
    idx = list(range(n))
    for _ in range(8):
        rng.shuffle(idx)
        diff = acts[idx[:mid]].mean(dim=0) - acts[idx[mid:]].mean(dim=0)
        norm = diff.norm()
        if norm > 1e-8:
            return diff / norm
    return None


def _build_dir_pools(
    lm: LoadedModel, interventions: Sequence[Intervention], prompts: list[str]
) -> tuple[list[torch.Tensor] | None, Literal["empirical_diff", "isotropic"] | None]:
    """Yön içeren tariflerde katman eşleşmeli ampirik fark havuzu; <2 istemde ya da sabit aktivasyonda izotropik yedek."""
    dirs = [iv for iv in interventions if isinstance(iv, DirectionAblation)]
    if not dirs:
        return None, None
    if len(prompts) < 2:
        return None, "isotropic"
    from neshter.concepts.hunt import candidate_layers, last_token_resid

    resid = last_token_resid(lm, prompts)
    window = list(candidate_layers(lm.arch.n_layers))
    pools = [resid[_match_direction_layer(resid, iv.direction, window)] for iv in dirs]
    if not all((p - p.mean(dim=0, keepdim=True)).norm() > 1e-7 for p in pools):
        return None, "isotropic"
    return pools, "empirical_diff"


def random_control(
    lm: LoadedModel,
    interventions: Sequence[Intervention],
    rng: random.Random,
    gen: torch.Generator,
    dir_pools: Sequence[torch.Tensor] | None = None,
) -> tuple[list[Intervention], bool]:
    """Aynı biçimde rastgele kontrol — "rastgele kontrol"ün tek tanımı. (kontrol, katman_eşleşti_mi) döner.

    Her HeadAblation yerine katman eşleşmeli, seçilmemiş rastgele head (`layer_matched_heads`);
    her DirectionAblation yerine `dir_pools` verildiyse aynı katmandaki istem çiftlerinden birim
    fark yönü (`empirical_diff`), verilmediyse rastgele izotropik birim yön (`isotropic`).
    """
    heads = [iv for iv in interventions if isinstance(iv, HeadAblation)]
    n_dirs = sum(isinstance(iv, DirectionAblation) for iv in interventions)
    picked, matched = layer_matched_heads(lm, heads, rng)
    control: list[Intervention] = list(picked)
    for idx in range(n_dirs):
        vec = _sample_empirical_direction(dir_pools[idx], rng) if dir_pools is not None else None
        if vec is None:
            vec = torch.randn(lm.arch.d_model, generator=gen)
        control.append(DirectionAblation(vec))
    return control, matched


def _draw_controls(
    lm: LoadedModel,
    interventions: Sequence[Intervention],
    n: int,
    seed: int,
    dir_pools: Sequence[torch.Tensor] | None = None,
) -> tuple[list[list[Intervention]], bool]:
    """n rastgele kontrol (tek random.Random + torch.Generator, tohum `seed`; çekiş sırası sabit) ve
    hepsinin katman eşleşmeli olup olmadığı. Havuz yetmezse ValueError (ölçümden önce)."""
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    drawn = [random_control(lm, interventions, rng, gen, dir_pools=dir_pools) for _ in range(n)]
    return [control for control, _ in drawn], all(matched for _, matched in drawn)


def _target_kl(
    lm: LoadedModel,
    base: list[torch.Tensor],
    prompts: list[str],
    interventions: Sequence[Intervention],
    last_k: int,
    progress: ProgressFn | None,
) -> tuple[float, float]:
    """Seçili müdahalenin ortalama KL'si ve popülasyon standart sapması (prompt başına değerlerden)."""
    per_prompt = _kl_from(lm, base, prompts, interventions, last_k)
    kl = sum(per_prompt) / len(per_prompt)
    kl_std = math.sqrt(sum((v - kl) ** 2 for v in per_prompt) / len(per_prompt))  # popülasyon
    if progress is not None:
        progress(0.05, "Seçili müdahalenin etkisi ölçüldü")
    return kl, kl_std


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
    dir_pools, dir_control = _build_dir_pools(lm, interventions, prompts)
    drawn, layer_matched = _draw_controls(lm, interventions, n_random, seed, dir_pools=dir_pools)  # önce doğrula

    base = _log_probs_batch(lm, prompts, last_k, cancel)
    kl, kl_std = _target_kl(lm, base, prompts, interventions, last_k, progress)

    random_kls: list[float] = []
    for i, control in enumerate(drawn):
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
        direction_control=dir_control,
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


# --- Sıralı (Besag–Clifford) permütasyon testi: /api/effect ----------------------------------------------
#
# N_max çekiş baştan seçilir (sonuca bakıp uzatılmaz); karar tam N_max kuralıdır: p = (1 + k)/(1 + N_max) <= τ.
# k, bu kuralın hâlâ sağlanabileceği en büyük değeri (futility_k) aşınca kalan çekişler kararı değiştiremez:
# çekim durur ve Besag–Clifford p = k/n raporlanır (asla yeniden hesaplanmış (1 + k)/(1 + n) değil).

DEFAULT_N_MAX = 39  # p_min = 1/40 = 0.025: m = 2 ölçümde Holm'a dayanır
MIN_N_MAX = 19  # p_min = 0.05: tek ölçümde anlamlılık gösterebilen en küçük N
CPU_N_MAX_CAP = 199  # CPU'da en çok çekiş (sunucu 422 verir)
TAU = 0.05


def futility_k(n_max: int, tau: float = TAU) -> int:
    """(1 + k)/(1 + n_max) <= tau koşulunu sağlayan en büyük k (hiçbiri sağlamıyorsa -1)."""
    return math.floor(tau * (n_max + 1) + 1e-9) - 1


@dataclass
class PermutationOutcome:
    n_drawn: int
    k_exceed: int  # #{çekiş >= hedef}
    p_value: float  # "full": (1 + k)/(1 + n_max); "besag_clifford": k/n_drawn
    p_kind: Literal["full", "besag_clifford"]
    significant: bool  # her zaman tam n_max kuralının kararı
    stopped_early: bool


def sequential_permutation(
    target: float, draws: Iterable[float], n_max: int, tau: float = TAU
) -> tuple[PermutationOutcome, list[float]]:
    """Çekişleri teker teker tüketir; durduktan sonra bir sonraki değeri istemez. (sonuç, görülen değerler).

    Karar, n_max çekişin hepsiyle verilecek kararla aynıdır (anlamlı ⇔ k <= futility_k). Son çekişte durma
    eşiği aşılırsa tüm çekişler yapılmış sayılır ve tam p raporlanır.
    """
    if n_max < 1:
        raise ValueError(f"n_max en az 1 olmalı: {n_max}.")
    k_max = futility_k(n_max, tau)
    if k_max < 0:
        raise ValueError(f"{n_max} çekişle p en az {1 / (n_max + 1):.4g} olur; τ = {tau} hiç sağlanamaz.")
    it = iter(draws)
    seen: list[float] = []
    k = 0
    while len(seen) < n_max:
        try:
            value = next(it)
        except StopIteration:
            raise ValueError(f"Çekişler bitti: {len(seen)}/{n_max}.") from None
        seen.append(value)
        k += value >= target  # eşitlik aşım sayılır (muhafazakâr)
        if k > k_max and len(seen) < n_max:
            return PermutationOutcome(len(seen), k, k / len(seen), "besag_clifford", False, True), seen
    return PermutationOutcome(n_max, k, (1 + k) / (1 + n_max), "full", k <= k_max, False), seen


@dataclass
class SequentialEffect:
    kl: float
    kl_std: float
    kl_se: float  # kl_std / sqrt(n_prompts)
    n_prompts: int
    random_kls: list[float]  # yalnız çekilenler (n_drawn tane), çekiş sırasıyla
    random_kl: float | None  # çekilenlerin ortalaması; n_drawn < 2 ise None
    random_p05: float | None
    random_p95: float | None
    percentile: float | None  # 100 * #{rastgele < kl} / n_drawn; n_drawn < 2 ise None
    n_drawn: int
    n_max: int
    k_exceed: int
    tau: float
    p_value: float
    p_kind: str  # "full" | "besag_clifford"
    p_min: float  # 1/(n_max + 1): bu N ile ulaşılabilecek en küçük p
    significant: bool
    stopped_early: bool
    layer_matched: bool
    seed: int


def sequential_real_effect(
    lm: LoadedModel,
    heads: list[HeadAblation],
    prompts: list[str],
    n_max: int = DEFAULT_N_MAX,
    tau: float = TAU,
    seed: int = 0,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> SequentialEffect:
    """Seçili head'lerin KL etkisi; katman eşleşmeli rastgele kontrollerle sıralı permütasyon testi.

    Kontroller recipe_effect ile aynı sırada çekilir (aynı tohum ⇒ aynı önek). Çekişler teker teker ölçülür;
    sonuç belli olunca (futility) durulur.
    """
    if n_max < MIN_N_MAX:
        raise ValueError(f"n_max en az {MIN_N_MAX} olmalı: {n_max}.")
    if not prompts:
        raise ValueError("En az bir prompt gerekli.")
    if futility_k(n_max, tau) < 0:  # ölçmeden önce: bu N ile τ hiç sağlanamaz
        raise ValueError(f"{n_max} çekişle p en az {1 / (n_max + 1):.4g} olur; τ = {tau} hiç sağlanamaz.")
    check_cancel(cancel)
    last_k = 16  # recipe_effect varsayılanıyla aynı (önek eşliği)
    interventions = _dedupe(heads)
    controls, layer_matched = _draw_controls(lm, interventions, n_max, seed)  # önce doğrula

    base = _log_probs_batch(lm, prompts, last_k, cancel)
    kl, kl_std = _target_kl(lm, base, prompts, interventions, last_k, progress)

    def draws() -> Iterator[float]:
        for i, control in enumerate(controls):
            check_cancel(cancel)
            values = _kl_from(lm, base, prompts, control, last_k)
            if progress is not None:
                progress(0.05 + 0.95 * (i + 1) / n_max, f"Rastgele kontrol {i + 1}/{n_max} (erken durabilir)")
            yield sum(values) / len(values)

    out, random_kls = sequential_permutation(kl, draws(), n_max, tau)
    band = len(random_kls) >= 2
    p05, p95 = (float(x) for x in np.percentile(random_kls, [5, 95])) if band else (None, None)
    return SequentialEffect(
        kl=kl,
        kl_std=kl_std,
        kl_se=kl_std / math.sqrt(len(prompts)),
        n_prompts=len(prompts),
        random_kls=random_kls,
        random_kl=sum(random_kls) / len(random_kls) if band else None,
        random_p05=p05,
        random_p95=p95,
        percentile=100.0 * sum(r < kl for r in random_kls) / len(random_kls) if band else None,
        n_drawn=out.n_drawn,
        n_max=n_max,
        k_exceed=out.k_exceed,
        tau=tau,
        p_value=out.p_value,
        p_kind=out.p_kind,
        p_min=1 / (n_max + 1),
        significant=out.significant,
        stopped_early=out.stopped_early,
        layer_matched=layer_matched,
        seed=seed,
    )


def probe_answer(lm: LoadedModel, heads: list[HeadAblation], text: str, k: int = 3) -> dict:
    """Bu istemin bir sonraki token'ı: modelin cevabı (head'ler açıkken en olası token) ve olasılığı, head'ler açıkken
    ve kapalıyken; iki durumda da ilk k tahmin. Ölçüm kartının yüz noktası bunu gösterir (tek istem, tahmin değil)."""
    enc = lm.tokenizer(text, return_tensors="pt").to(_device(lm))
    if enc["input_ids"].shape[1] < 1:
        raise ValueError("Boş istem: cevabı okunacak token yok.")

    def dist(interventions: Sequence[Intervention]) -> torch.Tensor:
        with apply(lm, interventions), torch.no_grad():
            logits = lm.model(**enc, use_cache=False).logits[0, -1]
        return torch.softmax(logits.float(), -1)

    def piece(token_id: int) -> str:
        tok = lm.tokenizer
        return tok.convert_tokens_to_string([tok.convert_ids_to_tokens(token_id)])

    def top(p: torch.Tensor) -> list[dict]:
        t = torch.topk(p, k)
        return [{"token_id": int(i), "text": piece(int(i)), "p": float(v)} for v, i in zip(t.values, t.indices)]

    before = dist([])
    after = dist(_dedupe(heads))
    y = int(before.argmax())
    return {
        "answer": {"token_id": y, "text": piece(y)},
        "p_before": float(before[y]),
        "p_after": float(after[y]),
        "top_before": top(before),
        "top_after": top(after),
    }


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
