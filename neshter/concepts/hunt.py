"""Davranış avı: bir kavramı taşıyan katmanı, yönü ve head'leri sıralar.

Tanımlar:
  - Her istek tek bir kullanıcı mesajı olarak sohbet şablonuna sokulur (add_generation_prompt=True);
    "son token" bu biçimlenmiş metnin son token'ıdır. Şablon yoksa ham istek kullanılır.
  - Yön (katman başına): mean(pos) − mean(neg) son token artık akımında, birim normlu.
  - Katman skoru: train yönü üzerinde val izdüşümlerinin Cohen d'si; aday penceresi (candidate_layers)
    içindeki en büyüğe bölünmüş, pencere dışı 0. ref_layer yalnızca pencereden seçilir.
  - Doğrulama doğruluğu: train izdüşümlerinin orta noktası eşik; val doğru sınıflama sayısı + Wilson aralığı.
  - Ölçüm: en iyi MEASURE_TOP_K head gerçekten sıfırlanır; sinyal düşüşü ve nötr KL ölçülür.
  - Head atıfı (attribution patching): m = mean_pos(resid[ref, son] · r_ref);
    skor_h = −mean_pos(∇_{z_h} m · (μ_h^neg − z_h)); z_h = o_proj girdisinin head dilimi, son token.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Literal

import torch

from neshter.activity import head_grams, last_token_head_power
from neshter.arch import decoder_layers, embed, o_proj
from neshter.concepts.model import Concept, split
from neshter.interventions import DirectionAblation, HeadAblation, apply
from neshter.loader import LoadedModel
from neshter.metrics import _kl_from, _log_probs_batch, load_neutral, wilson
from neshter.progress import ProgressFn, check_cancel, sub_progress

HUNT_BATCH = 8  # ileri geçişlerde (grad gerekmeyenler) aynı anda işlenen istek sayısı
LAYER_WINDOW_FRAC = 0.8  # aday katmanlar: 1 .. floor(0.8·L) (son katmanlar çıktıya hazırlanır, yön bulanıklaşır)
MEASURE_TOP_K = 20  # gerçekten sıfırlanıp ölçülen head sayısı
MEASURE_NEUTRAL = 8  # nötr KL için kullanılan nötr istek sayısı
NEUTRAL_LAST_K = 16  # nötr KL için son pozisyon sayısı (metrics.real_effect ile aynı)


@dataclass
class HeadMeasure:
    layer: int
    head: int
    estimated: float  # av yönteminin tahmini (heads'teki skor)
    # (m_pos - m_pos_h) / (m_pos - m_neg); 1.0: pozitifler negatif ortalamaya çöktü. None: ölçülmedi —
    # head ref_layer'da ya da sonrasında, m'yi yapısal olarak etkileyemez ("etkisiz" ölçümü DEĞİL).
    signal_drop: float | None
    neutral_kl: float  # head sıfırlanınca nötr istekler üzerinde ortalama KL(orijinal || müdahaleli)


@dataclass
class HuntResult:
    layer_scores: list[float]
    heads: list[tuple[int, int, float]]
    direction: torch.Tensor | None
    direction_layer: int | None
    ref_layer: int
    val_correct: int
    val_n: int
    val_accuracy: float
    val_ci: tuple[float, float]
    measured: list[HeadMeasure]
    # Tekrarlı süpürme (INLP benzeri): ilk yönden sonra bulunan ek yönler (ortonormal) ve her adımdaki val doğruluğu.
    extra_directions: list[torch.Tensor] = field(default_factory=list)
    sweep: list[dict] = field(default_factory=list)


SWEEP_STOP_ACC = 0.6  # kalan sinyalin val doğruluğu bunun altına inince süpürme durur (şansa yakın)


def sweep_directions(
    lm: LoadedModel,
    train: Concept,
    val: Concept,
    r1: torch.Tensor,
    max_k: int,
    stop_acc: float = SWEEP_STOP_ACC,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Model düzeyinde tekrarlı süpürme (INLP fikri, modelin kendi telafisine karşı).

    Bulunan yönler modelden gerçekten silinir (kaydedilecek olanla aynı `DirectionAblation`), değişmiş model
    yeniden çalıştırılır ve aday katmanların her birinde kavram yönü yeniden bulunur. Kavram herhangi bir
    katmanda hâlâ ayrılıyorsa (val doğruluğu >= stop_acc), o katmanın yönü (öncekilere dik) listeye eklenir.

    Neden aynı katmanda tekrar bakmak yetmez: iki sınıflı bir kavramda ortalama farkı yönü silinince sınıf
    ortalamaları eşitlenir ve o katmanda hiçbir doğrusal sınıflandırıcı ayıramaz (LEACE). Sızıntı, sonraki
    katmanların kavramı başka bir yönde yeniden kurmasından gelir; bu ancak değiştirilmiş modelde görülür.

    Dönen yönler ortonormal, ilki r1. `steps[i]`: o ana dek bulunan i+1 yön silinmişken en çok ayrışan katman ve
    val doğruluğu (kalan sızıntı)."""
    found = [r1.detach().to(torch.float32) / r1.norm()]
    steps: list[dict] = []
    window = list(candidate_layers(lm.arch.n_layers))
    while True:
        check_cancel(cancel)
        with apply(lm, [DirectionAblation(d) for d in found]):
            tp, tn, vp, vn = (last_token_resid(lm, x) for x in (train.positive, train.negative, val.positive, val.negative))
        dirs = _directions_from(tp, tn)
        accs = {}
        for layer in window:
            if dirs[layer].norm() == 0:
                continue
            correct, n = val_accuracy(dirs[layer], tp[layer], tn[layer], vp[layer], vn[layer])
            accs[layer] = (correct / n if n else 0.0, n)
        best = max(accs, key=lambda k: accs[k][0]) if accs else None
        acc, n = accs[best] if best is not None else (0.0, 0)
        steps.append({"removed": len(found), "layer": best, "residual_accuracy": acc, "n": n})
        if progress is not None:
            progress(min(1.0, len(found) / max_k), f"Süpürme: {len(found)} yön silindi")
        if best is None or acc < stop_acc or len(found) >= max_k:
            break
        q = torch.stack(found, dim=1)
        r = dirs[best] - q @ (q.T @ dirs[best])
        if r.norm() < 1e-6:
            break
        found.append(r / r.norm())
    return found, steps


def candidate_layers(n_layers: int) -> range:
    """layer_scores indisleri (i = katman i-1 sonrası artık akım): 1 .. max(1, floor(0.8·n_layers))."""
    return range(1, max(1, math.floor(LAYER_WINDOW_FRAC * n_layers + 1e-9)) + 1)


# --- Girdi biçimlendirme ----------------------------------------------------------------------


def format_prompt(lm: LoadedModel, prompt: str) -> str:
    tok = lm.tokenizer
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def _encode_batch(lm: LoadedModel, prompts: list[str]):
    """Sohbet şablonlu (şablon varsa), sola dolgulu yığın; tek şablon/dolgu uygulaması budur.

    position_ids dolguyu atlar (ilk gerçek token 0): dolgulu satırlar dolgusuz tek istekle aynı sayıyı verir.
    """
    templated = bool(getattr(lm.tokenizer, "chat_template", None))
    texts = [format_prompt(lm, p) for p in prompts]
    device = next(lm.model.parameters()).device
    # Şablon BOS gibi özel token'ları zaten metne yazar; tokenizer ikinci kez eklemesin.
    enc = lm.tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=not templated
    )
    enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    return enc.to(device)


def _encode(lm: LoadedModel, prompt: str):
    return _encode_batch(lm, [prompt])


# --- Artık akım ve yönler ---------------------------------------------------------------------


class _StopForward(Exception):
    """İstenen durum üretildikten sonra ileri geçişi kesmek için (yalnızca _hidden_states içinde yakalanır)."""


def _hidden_states(lm: LoadedModel, enc, stop_at: int | None = None) -> list[torch.Tensor]:
    """n_layers+1 norm-öncesi artık akım durumu: gömme çıktısı ve her decoder katmanının çıktısı
    ([B, T, d_model], graf korunur). 0..L−1 HF `hidden_states` ile aynıdır; L ise HF'nin aksine son
    norm ÖNCESİDİR — DirectionAblation'ın düzenlediği taban budur (norm sonrası γ ile bükülür).

    `stop_at` verilirse yalnızca 0..stop_at döner ve ileri geçiş o durumu üreten kancadan fırlatılan
    özel bir istisnayla orada kesilir (sonraki katmanlar hiç çalışmaz).

    output_hidden_states=True kullanılmaz: transformers 5 ilk kullanımda katmanlara kalıcı yakalama
    kancaları ekler; burada kancalar kendimizce eklenip her çağrıdan sonra kaldırılır.
    """
    states: dict[int, torch.Tensor] = {}

    def make_hook(i: int):
        def hook(module, args, output):
            states[i] = output[0] if isinstance(output, tuple) else output
            if i == stop_at:
                raise _StopForward

        return hook

    mods = [embed(lm.model)] + list(decoder_layers(lm.model))
    if stop_at is not None:
        mods = mods[: stop_at + 1]
    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(mods)]
    try:
        lm.model.model(**enc, use_cache=False)  # lm_head atlanır
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()
    return [states[i] for i in range(len(mods))]


def _last_token_states(
    lm: LoadedModel,
    prompts: list[str],
    capture_z: bool = False,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(resid [L+1, N, d_model], z [N, L, H, head_dim] ya da None); ikisi de float32 CPU.

    İstekler HUNT_BATCH'lik sola dolgulu yığınlarla işlenir; resid norm-öncesi artık akımın son token'ı
    (bkz. _hidden_states), z o_proj girdisinin son token'daki head dilimleridir (capture_z).
    """
    arch = lm.arch
    resid_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    total = len(prompts)
    for start in range(0, total, HUNT_BATCH):
        check_cancel(cancel)
        chunk = prompts[start : start + HUNT_BATCH]
        enc = _encode_batch(lm, chunk)
        zs: dict[int, torch.Tensor] = {}
        handles = []
        if capture_z:

            def make_hook(layer: int):
                def hook(module, args):
                    zs[layer] = args[0].detach()[:, -1].reshape(len(chunk), arch.n_heads, arch.head_dim).to(
                        "cpu", torch.float32
                    )

                return hook

            handles = [o_proj(lm.model, l).register_forward_pre_hook(make_hook(l)) for l in range(arch.n_layers)]
        try:
            with torch.no_grad():
                hs = _hidden_states(lm, enc)
        finally:
            for h in handles:
                h.remove()
        resid_parts.append(torch.stack([h[:, -1] for h in hs]).to("cpu", torch.float32))  # [L+1, B, d]
        if capture_z:
            z_parts.append(torch.stack([zs[l] for l in range(arch.n_layers)], dim=1))  # [B, L, H, hd]
        if progress is not None:
            progress(min(start + len(chunk), total) / total, f"{min(start + len(chunk), total)}/{total} istek işlendi")
    resid = torch.cat(resid_parts, dim=1)
    return resid, (torch.cat(z_parts, dim=0) if capture_z else None)


def last_token_resid(lm: LoadedModel, prompts: list[str]) -> torch.Tensor:
    """[n_layers+1, N, d_model] float32 CPU; norm-öncesi artık akımın son token'ı (bkz. _hidden_states)."""
    return _last_token_states(lm, prompts)[0]


def _unit(x: torch.Tensor) -> torch.Tensor:
    """Satır satır birim norm; sıfır satır sıfır kalır."""
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, x / torch.where(norm > 0, norm, torch.ones_like(norm)), torch.zeros_like(x))


def _directions_from(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return _unit(pos.mean(dim=1) - neg.mean(dim=1))


def directions(lm: LoadedModel, c: Concept) -> torch.Tensor:
    """[n_layers+1, d_model]; katman başına birim mean(pos) − mean(neg). Fark sıfırsa satır sıfırdır."""
    return _directions_from(last_token_resid(lm, c.positive), last_token_resid(lm, c.negative))


# --- Katman skorları --------------------------------------------------------------------------


def _cohen_d(dirs: torch.Tensor, val_pos: torch.Tensor, val_neg: torch.Tensor) -> torch.Tensor:
    """[L+1] ham Cohen d: (ort_pos − ort_neg) / sqrt((var_pos + var_neg)/2) (popülasyon varyansı)."""
    pp = torch.einsum("lnd,ld->ln", val_pos, dirs)
    pn = torch.einsum("lnd,ld->ln", val_neg, dirs)
    pooled = ((pp.var(dim=1, unbiased=False) + pn.var(dim=1, unbiased=False)) / 2).sqrt()
    pooled = torch.where(pooled > 0, pooled, torch.full_like(pooled, 1e-8))
    return (pp.mean(dim=1) - pn.mean(dim=1)) / pooled


def _normalize_scores(d: torch.Tensor) -> list[float]:
    """max(d) > 0 ise d / max(d) (en büyük = 1). Hiç pozitif yoksa d / max|d| (değerler [−1, 0]);
    hepsi sıfırsa sıfırlar. Negatif d, yönün val'de ters çıktığı (tutarsız) katmandır."""
    top = d.max()
    if top > 0:
        return (d / top).tolist()
    scale = d.abs().max()
    if scale > 0:
        return (d / scale).tolist()
    return torch.zeros_like(d).tolist()


def _window_scores(d: torch.Tensor, n_layers: int) -> list[float]:
    """Cohen d'yi aday penceresi içindeki en büyüğe böler; pencere dışı girdiler 0.0 olur."""
    window = list(candidate_layers(n_layers))
    scores = [0.0] * len(d)
    for i, v in zip(window, _normalize_scores(d[window])):
        scores[i] = v
    return scores


def layer_scores(lm: LoadedModel, train: Concept, val: Concept) -> list[float]:
    """Katman başına, train yönü üzerinde val pos/neg izdüşümlerinin Cohen d'si; aday penceresindeki
    en büyüğe bölünmüş, pencere dışı 0.0 (bkz. candidate_layers)."""
    dirs = directions(lm, train)
    d = _cohen_d(dirs, last_token_resid(lm, val.positive), last_token_resid(lm, val.negative))
    return _window_scores(d, lm.arch.n_layers)


def val_accuracy(
    r: torch.Tensor,
    train_pos: torch.Tensor,
    train_neg: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
) -> tuple[int, int]:
    """(doğru, n): eşik t = train pos/neg ortalama izdüşümlerinin orta noktası; val pozitifi izdüşüm > t,
    val negatifi izdüşüm <= t ise doğru. Girdiler ref katmandaki son-token durumları [N, d_model]."""
    t = ((train_pos @ r).mean() + (train_neg @ r).mean()) / 2
    correct = int(((val_pos @ r) > t).sum()) + int(((val_neg @ r) <= t).sum())
    return correct, val_pos.shape[0] + val_neg.shape[0]


# --- Head skorları ----------------------------------------------------------------------------


def _normalize_abs(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max()
    return x / scale if scale > 0 else torch.zeros_like(x)


def _attribution(
    lm: LoadedModel,
    c: Concept,
    ref_layer: int,
    r_ref: torch.Tensor,
    mu_neg: torch.Tensor | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> torch.Tensor:
    """Ham (normalize edilmemiş) atıf skorları [n_layers, n_heads].

    mu_neg [L, H, head_dim] negatiflerin ortalama z'sidir; verilmezse negatiflerden hesaplanır (hunt
    bunu train negatiflerinin artık akım geçişinden bedavaya verir). Yalnızca ref_layer'dan ÖNCEKİ
    katmanlar m'yi etkileyebilir: δ kancaları onlara konur ve ileri geçiş ref_layer'da kesilir.
    İstekler tek tek işlenir (geri yayılım yığınlanmaz).
    """
    arch = lm.arch
    scores = torch.zeros(arch.n_layers, arch.n_heads, dtype=torch.float32)
    if ref_layer == 0:  # gömme çıktısı: hiçbir head onu etkileyemez
        return scores
    if mu_neg is None:
        _, z_neg = _last_token_states(lm, c.negative, capture_z=True, cancel=cancel)
        mu_neg = z_neg.mean(dim=0)
    device = next(lm.model.parameters()).device
    r = r_ref.to(device, torch.float32)

    for i, p in enumerate(c.positive):
        check_cancel(cancel)
        deltas: dict[int, torch.Tensor] = {}
        zs: dict[int, torch.Tensor] = {}

        def make_hook(layer: int):
            def hook(module, args):
                x = args[0]
                # x + δ (δ = 0, grad ister): ∂m/∂δ = ∂m/∂z, sonraki head'ler üzerinden geçen yollar dahil
                # (parametreler requires_grad=False olduğu için x'in kendisi grad taşımayabilir).
                delta = torch.zeros_like(x, requires_grad=True)
                deltas[layer] = delta
                zs[layer] = x.detach()[0, -1].reshape(arch.n_heads, arch.head_dim).to("cpu", torch.float32)
                return (x + delta,) + tuple(args[1:])

            return hook

        handles = [o_proj(lm.model, l).register_forward_pre_hook(make_hook(l)) for l in range(ref_layer)]
        try:
            with torch.enable_grad():
                hs = _hidden_states(lm, _encode(lm, p), stop_at=ref_layer)
                m = hs[ref_layer][0, -1].to(torch.float32) @ r
                inputs = [deltas[l] for l in range(ref_layer)]
                grads = torch.autograd.grad(m, inputs, allow_unused=True)
        finally:
            for h in handles:
                h.remove()

        for layer, g in enumerate(grads):
            if g is None:
                continue
            g_last = g[0, -1].reshape(arch.n_heads, arch.head_dim).to("cpu", torch.float32)
            scores[layer] += (g_last * (mu_neg[layer] - zs[layer])).sum(dim=-1)
        if progress is not None:
            progress((i + 1) / len(c.positive), f"{i + 1}/{len(c.positive)} atıf geçişi")

    return -scores / len(c.positive)


def head_attribution(lm: LoadedModel, c: Concept, ref_layer: int, normalize: bool = True) -> torch.Tensor:
    """[n_layers, n_heads]; −mean_pos(∇_{z_h} m · (μ_h^neg − z_h)), en büyük mutlak değere bölünmüş.

    Pozitif skor: head'i negatif ortalamasına yamamak m'yi düşürür (head kavramı taşır).
    r_ref = directions(lm, c)[ref_layer]. ref_layer == 0 ise sıfırlar.
    """
    r_ref = directions(lm, c)[ref_layer] if ref_layer > 0 else torch.zeros(lm.arch.d_model)
    raw = _attribution(lm, c, ref_layer, r_ref)
    return _normalize_abs(raw) if normalize else raw


def _power_from_z(lm: LoadedModel, z: torch.Tensor, grams: dict[int, torch.Tensor]) -> torch.Tensor:
    """z [N, L, H, head_dim] (son token, float32 CPU) → head çıktı gücü ‖W_h z_h‖ [N, L, H].

    last_token_head_power ile aynı formül (Gram matrisleri, activity.head_grams); ileri geçiş yok.
    """
    arch = lm.arch
    out = torch.empty(z.shape[0], arch.n_layers, arch.n_heads, dtype=torch.float32)
    for layer in range(arch.n_layers):
        if layer not in grams:
            grams[layer] = head_grams(o_proj(lm.model, layer), arch.n_heads, arch.head_dim).to("cpu")
        zl = z[:, layer]
        out[:, layer] = torch.einsum("nhd,hde,nhe->nh", zl, grams[layer], zl).clamp_min(0).sqrt()
    return out


def _activation_diff_from_z(lm: LoadedModel, z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
    """activation_diff'in aynısı, artık akım geçişlerinde yakalanmış z'lerden (ek ileri geçiş yok)."""
    grams: dict[int, torch.Tensor] = {}
    diff = (_power_from_z(lm, z_pos, grams).mean(dim=0) - _power_from_z(lm, z_neg, grams).mean(dim=0)).abs()
    top = diff.max()
    return diff / top if top > 0 else torch.zeros_like(diff)


def activation_diff(
    lm: LoadedModel,
    c: Concept,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> torch.Tensor:
    """[n_layers, n_heads]; |ort. head çıktı gücü(pos) − ort.(neg)| son token'da, en büyüğe bölünmüş.

    Kendi ileri geçişlerini yapar (tek başına kullanım). `hunt` aynı değeri artık akım geçişlerinde yakaladığı
    z'lerden hesaplar (_activation_diff_from_z); pozitif/negatifler ikinci kez geçirilmez.
    """
    grams: dict[int, torch.Tensor] = {}  # o_proj Gram matrisleri bu av boyunca bir kez hesaplanır
    total = len(c.positive) + len(c.negative)
    done = 0

    def mean_power(prompts: list[str]) -> torch.Tensor:
        nonlocal done
        parts = []
        for start in range(0, len(prompts), HUNT_BATCH):
            check_cancel(cancel)
            chunk = prompts[start : start + HUNT_BATCH]
            parts.append(last_token_head_power(lm, _encode_batch(lm, chunk), grams))
            done += len(chunk)
            if progress is not None:
                progress(done / total, f"{done}/{total} istek işlendi")
        return torch.cat(parts, dim=0).mean(dim=0)

    diff = (mean_power(c.positive) - mean_power(c.negative)).abs()
    top = diff.max()
    return diff / top if top > 0 else torch.zeros_like(diff)


# --- Ölçüm: tahmin edilen en iyi head'leri gerçekten sıfırla ----------------------------------


def _ref_projection(
    lm: LoadedModel, prompts: list[str], ref_layer: int, r: torch.Tensor, cancel: threading.Event | None
) -> torch.Tensor:
    """[N] resid[ref, son token] . r (float32 CPU); ileri geçiş ref_layer'da kesilir, etkin kancalar korunur."""
    parts = []
    for start in range(0, len(prompts), HUNT_BATCH):
        check_cancel(cancel)
        enc = _encode_batch(lm, prompts[start : start + HUNT_BATCH])
        with torch.no_grad():
            h = _hidden_states(lm, enc, stop_at=ref_layer)[ref_layer]
        parts.append(h[:, -1].to("cpu", torch.float32) @ r)
    return torch.cat(parts)


def measure_top_heads(
    lm: LoadedModel,
    heads: list[tuple[int, int, float]],
    val_pos: list[str],
    ref_layer: int,
    r_ref: torch.Tensor,
    m_pos: float,
    m_neg: float,
    k: int = MEASURE_TOP_K,
    n_neutral: int = MEASURE_NEUTRAL,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> list[HeadMeasure]:
    """`heads`in (tahmine göre sıralı) ilk k head'ini tek tek sıfırlayıp ölçer; signal_drop azalan sıralı.

    signal_drop = (m_pos - m_pos_h) / (m_pos - m_neg); yalnızca ref_layer'dan ÖNCEKİ katmanlar m'yi etkiler,
    bu yüzden layer >= ref_layer için None (yapısal olarak ölçülemez, ileri geçiş yok; API'de `null`).
    Sıralama: ölçülenler signal_drop'a göre azalan, None'lar en sonda. neutral_kl: ilk n_neutral nötr
    istekte, son NEUTRAL_LAST_K pozisyonda ortalama KL; orijinal log-olasılıklar tek kez hesaplanır.
    """
    top = heads[:k]
    if not top:
        return []
    neutral = load_neutral()[:n_neutral]
    r = r_ref.to("cpu", torch.float32)
    denom = m_pos - m_neg
    base = _log_probs_batch(lm, neutral, NEUTRAL_LAST_K, cancel)  # yığınlı (metrics.KL_BATCH)
    out: list[HeadMeasure] = []
    for i, (layer, head, est) in enumerate(top):
        check_cancel(cancel)
        iv = [HeadAblation(layer, head)]
        drop: float | None = None
        if layer < ref_layer and denom > 0:
            with apply(lm, iv):
                m_h = float(_ref_projection(lm, val_pos, ref_layer, r, cancel).mean())
            drop = float((m_pos - m_h) / denom)
        kls = _kl_from(lm, base, neutral, iv, NEUTRAL_LAST_K, cancel)
        out.append(HeadMeasure(layer, head, est, drop, sum(kls) / len(kls)))
        if progress is not None:
            progress((i + 1) / len(top), f"{i + 1}/{len(top)} head ölçüldü")
    out.sort(key=lambda m: (m.signal_drop is not None, m.signal_drop or 0.0), reverse=True)  # None en sonda
    return out


# --- Av ---------------------------------------------------------------------------------------


def default_method(lm: LoadedModel) -> Literal["attribution", "activation_diff"]:
    """Varsayılan yöntem: CUDA'da atıf (geri yayılım), CPU'da etkinlik farkı (spec varsayılanı).

    v0.1.1 ölçümlerinde atıf CPU'da da kullanılabilir hızdadır; iki yöntemin süresine çoğunlukla ölçüm
    aşaması (nötr KL) hakimdir. Varsayılanı değiştirmek, yığınlı KL'den sonra yeniden ölçüm ister.
    """
    on_cuda = next(lm.model.parameters()).device.type == "cuda"
    return "attribution" if on_cuda else "activation_diff"


def hunt(
    lm: LoadedModel,
    c: Concept,
    method: Literal["attribution", "activation_diff", "direction"],
    seed: int = 0,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    sweeps: int = 1,
) -> HuntResult:
    """split → katman skorları → ref_layer = pencere içi argmax → head sıralaması (train kümesinde)
    → en iyi MEASURE_TOP_K head'in gerçek sıfırlama ölçümü (her yöntem için).

    "direction" yönteminde head sırası activation_diff skorlarıdır; yön, ref_layer'daki train yönüdür.
    İlerleme tek yönlü artar: artık akım geçişleri 0–0.4 (istek sayısına orantılı), head sıralaması 0.4–0.8,
    ölçüm 0.8–1.
    """
    if method not in ("attribution", "activation_diff", "direction"):
        raise ValueError(f"Bilinmeyen yöntem: {method!r}")
    check_cancel(cancel)
    train, val = split(c, seed=seed)
    sets = [train.positive, train.negative, val.positive, val.negative]
    total = sum(len(x) for x in sets)
    bounds, acc = [], 0
    for x in sets:
        bounds.append((0.4 * acc / total, 0.4 * (acc + len(x)) / total))
        acc += len(x)

    def resid(i: int, capture_z: bool = False):
        return _last_token_states(
            lm, sets[i], capture_z=capture_z, progress=sub_progress(progress, *bounds[i]), cancel=cancel
        )

    # z (o_proj girdisi, son token) artık akım geçişlerinde bedavaya yakalanır: atıf negatiflerin ortalamasını,
    # activation_diff/direction pozitif + negatiflerin head gücünü ondan alır (ikinci ileri geçiş yok).
    pos, z_pos = resid(0, capture_z=method != "attribution")
    neg, z_neg = resid(1, capture_z=True)
    dirs = _directions_from(pos, neg)
    val_pos, _ = resid(2)
    val_neg, _ = resid(3)
    d = _cohen_d(dirs, val_pos, val_neg)
    window = candidate_layers(lm.arch.n_layers)
    if not bool((d[list(window)] > 0).any()):
        raise ValueError("Kavram doğrulama kümesinde ayrışmıyor: daha fazla ya da daha belirgin örnek gerekli.")
    scores = _window_scores(d, lm.arch.n_layers)
    ref_layer = max(window, key=lambda i: float(d[i]))

    head_progress = sub_progress(progress, 0.4, 0.8)
    if method == "attribution":
        head_scores = _normalize_abs(
            _attribution(
                lm, train, ref_layer, dirs[ref_layer], mu_neg=z_neg.mean(dim=0), progress=head_progress, cancel=cancel
            )
        )
    else:
        check_cancel(cancel)
        head_scores = _activation_diff_from_z(lm, z_pos, z_neg)
        if head_progress is not None:
            head_progress(1.0, "Head gücü farkı hesaplandı")
    n_heads = head_scores.shape[1]
    heads = sorted(
        ((layer, head, float(head_scores[layer, head])) for layer in range(head_scores.shape[0]) for head in range(n_heads)),
        key=lambda t: t[2],
        reverse=True,
    )
    r_ref = dirs[ref_layer]
    val_correct, val_n = val_accuracy(r_ref, pos[ref_layer], neg[ref_layer], val_pos[ref_layer], val_neg[ref_layer])
    is_dir = method == "direction"
    sweeping = is_dir and sweeps > 1
    measured = measure_top_heads(
        lm,
        heads,
        sets[2],
        ref_layer,
        r_ref,
        float((val_pos[ref_layer] @ r_ref).mean()),
        float((val_neg[ref_layer] @ r_ref).mean()),
        k=MEASURE_TOP_K,
        n_neutral=MEASURE_NEUTRAL,
        progress=sub_progress(progress, 0.8, 0.9 if sweeping else 1.0),
        cancel=cancel,
    )
    extra: list[torch.Tensor] = []
    sweep: list[dict] = []
    if sweeping:
        # Her süpürme adımı modeli yeniden çalıştırır: çubuk %100'e ancak süpürme bitince varır.
        found, sweep = sweep_directions(
            lm, train, val, r_ref, max_k=sweeps, progress=sub_progress(progress, 0.9, 1.0), cancel=cancel
        )
        extra = [d.clone() for d in found[1:]]
    if progress is not None:
        progress(1.0, "Ölçüm tamamlandı")
    return HuntResult(
        extra_directions=extra,
        sweep=sweep,
        layer_scores=scores,
        heads=heads,
        direction=dirs[ref_layer].clone() if is_dir else None,
        direction_layer=ref_layer if is_dir else None,
        ref_layer=ref_layer,
        val_correct=val_correct,
        val_n=val_n,
        val_accuracy=val_correct / val_n if val_n else 0.0,
        val_ci=wilson(val_correct, val_n)[1:],
        measured=measured,
    )
