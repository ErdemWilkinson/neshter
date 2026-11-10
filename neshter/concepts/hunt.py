"""Davranış avı: bir kavramı taşıyan katmanı, yönü ve head'leri sıralar.

Tanımlar:
  - Her istek tek bir kullanıcı mesajı olarak sohbet şablonuna sokulur (add_generation_prompt=True);
    "son token" bu biçimlenmiş metnin son token'ıdır. Şablon yoksa ham istek kullanılır.
  - Yön (katman başına): mean(pos) − mean(neg) son token artık akımında, birim normlu.
  - Katman skoru: train yönü üzerinde val izdüşümlerinin Cohen d'si, en büyüğe bölünmüş.
  - Head atıfı (attribution patching): m = mean_pos(resid[ref, son] · r_ref);
    skor_h = −mean_pos(∇_{z_h} m · (μ_h^neg − z_h)); z_h = o_proj girdisinin head dilimi, son token.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Literal

import torch

from neshter.activity import last_token_head_power
from neshter.arch import decoder_layers, embed, o_proj
from neshter.concepts.model import Concept, split
from neshter.loader import LoadedModel
from neshter.progress import ProgressFn, check_cancel, sub_progress

HUNT_BATCH = 8  # ileri geçişlerde (grad gerekmeyenler) aynı anda işlenen istek sayısı


@dataclass
class HuntResult:
    layer_scores: list[float]
    heads: list[tuple[int, int, float]]
    direction: torch.Tensor | None
    direction_layer: int | None
    val_consistency: float


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
    enc = lm.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=not templated)
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


def layer_scores(lm: LoadedModel, train: Concept, val: Concept) -> list[float]:
    """Katman başına, train yönü üzerinde val pos/neg izdüşümlerinin Cohen d'si; en büyüğe bölünmüş."""
    dirs = directions(lm, train)
    d = _cohen_d(dirs, last_token_resid(lm, val.positive), last_token_resid(lm, val.negative))
    return _normalize_scores(d)


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


def activation_diff(
    lm: LoadedModel,
    c: Concept,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> torch.Tensor:
    """[n_layers, n_heads]; |ort. head çıktı gücü(pos) − ort.(neg)| son token'da, en büyüğe bölünmüş."""
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


# --- Av ---------------------------------------------------------------------------------------


def default_method(lm: LoadedModel) -> Literal["attribution", "activation_diff"]:
    """Varsayılan yöntem: atıf (geri yayılım) CUDA'da; CPU'da çok yavaş olduğundan etkinlik farkı."""
    on_cuda = next(lm.model.parameters()).device.type == "cuda"
    return "attribution" if on_cuda else "activation_diff"


def hunt(
    lm: LoadedModel,
    c: Concept,
    method: Literal["attribution", "activation_diff", "direction"],
    seed: int = 0,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> HuntResult:
    """split → katman skorları → ref_layer = argmax → head sıralaması (train kümesinde).

    "direction" yönteminde head sırası activation_diff skorlarıdır; yön, ref_layer'daki train yönüdür.
    İlerleme tek yönlü artar: artık akım geçişleri 0–0.4 (istek sayısına orantılı), head sıralaması 0.4–1.
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

    pos, _ = resid(0)
    neg, z_neg = resid(1, capture_z=method == "attribution")  # negatif z, atıf için ek geçiş olmadan gelir
    dirs = _directions_from(pos, neg)
    val_pos, _ = resid(2)
    val_neg, _ = resid(3)
    d = _cohen_d(dirs, val_pos, val_neg)
    if not bool((d > 0).any()):
        raise ValueError("Kavram doğrulama kümesinde ayrışmıyor: daha fazla ya da daha belirgin örnek gerekli.")
    scores = _normalize_scores(d)
    ref_layer = max(range(len(scores)), key=lambda i: scores[i])

    head_progress = sub_progress(progress, 0.4, 1.0)
    if method == "attribution":
        head_scores = _normalize_abs(
            _attribution(
                lm, train, ref_layer, dirs[ref_layer], mu_neg=z_neg.mean(dim=0), progress=head_progress, cancel=cancel
            )
        )
    else:
        head_scores = activation_diff(lm, train, progress=head_progress, cancel=cancel)
    if progress is not None:
        progress(1.0, "Head'ler sıralanıyor")

    n_heads = head_scores.shape[1]
    heads = sorted(
        ((layer, head, float(head_scores[layer, head])) for layer in range(head_scores.shape[0]) for head in range(n_heads)),
        key=lambda t: t[2],
        reverse=True,
    )
    is_dir = method == "direction"
    return HuntResult(
        layer_scores=scores,
        heads=heads,
        direction=dirs[ref_layer].clone() if is_dir else None,
        direction_layer=ref_layer if is_dir else None,
        val_consistency=math.tanh(float(d[ref_layer]) / 2),
    )
