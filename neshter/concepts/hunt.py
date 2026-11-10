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
from dataclasses import dataclass
from typing import Literal

import torch

from neshter.activity import head_output_power
from neshter.arch import decoder_layers, embed, o_proj
from neshter.concepts.model import Concept, split
from neshter.loader import LoadedModel


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


def _encode(lm: LoadedModel, prompt: str):
    templated = bool(getattr(lm.tokenizer, "chat_template", None))
    text = format_prompt(lm, prompt)
    device = next(lm.model.parameters()).device
    # Şablon BOS gibi özel token'ları zaten metne yazar; tokenizer ikinci kez eklemesin.
    return lm.tokenizer(text, return_tensors="pt", add_special_tokens=not templated).to(device)


# --- Artık akım ve yönler ---------------------------------------------------------------------


def _hidden_states(lm: LoadedModel, enc) -> list[torch.Tensor]:
    """HF `output_hidden_states` ile birebir aynı n_layers+1 durum: gömme çıktısı, katman 0..L−2
    çıktıları ve son norm çıktısı ([1, T, d_model], graf korunur).

    output_hidden_states=True kullanılmaz: transformers 5 ilk kullanımda katmanlara kalıcı yakalama
    kancaları ekler; burada kancalar kendimizce eklenip her çağrıdan sonra kaldırılır.
    """
    states: dict[int, torch.Tensor] = {}

    def make_hook(i: int):
        def hook(module, args, output):
            states[i] = output[0] if isinstance(output, tuple) else output

        return hook

    mods = [embed(lm.model)] + list(decoder_layers(lm.model))[:-1]
    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(mods)]
    try:
        out = lm.model.model(**enc, use_cache=False)  # lm_head atlanır
    finally:
        for h in handles:
            h.remove()
    return [states[i] for i in range(len(mods))] + [out.last_hidden_state]


def last_token_resid(lm: LoadedModel, prompts: list[str]) -> torch.Tensor:
    """[n_layers+1, N, d_model] float32 CPU; hidden_states'in son token'ı (istekler tek tek, dolgu yok)."""
    rows = []
    with torch.no_grad():
        for p in prompts:
            hs = _hidden_states(lm, _encode(lm, p))
            rows.append(torch.stack([h[0, -1] for h in hs]).to("cpu", torch.float32))
    return torch.stack(rows, dim=1)


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


def _last_token_z(lm: LoadedModel, prompts: list[str]) -> torch.Tensor:
    """[N, n_layers, n_heads, head_dim] o_proj girdisi (son token), grad yok."""
    arch = lm.arch
    out = []
    for p in prompts:
        zs: dict[int, torch.Tensor] = {}

        def make_hook(layer: int):
            def hook(module, args):
                zs[layer] = args[0].detach()[0, -1].reshape(arch.n_heads, arch.head_dim).to("cpu", torch.float32)

            return hook

        handles = [o_proj(lm.model, l).register_forward_pre_hook(make_hook(l)) for l in range(arch.n_layers)]
        try:
            with torch.no_grad():
                lm.model.model(**_encode(lm, p), use_cache=False)  # lm_head gereksiz
        finally:
            for h in handles:
                h.remove()
        out.append(torch.stack([zs[l] for l in range(arch.n_layers)]))
    return torch.stack(out)


def _normalize_abs(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max()
    return x / scale if scale > 0 else torch.zeros_like(x)


def _attribution(lm: LoadedModel, c: Concept, ref_layer: int, r_ref: torch.Tensor) -> torch.Tensor:
    """Ham (normalize edilmemiş) atıf skorları [n_layers, n_heads]."""
    arch = lm.arch
    scores = torch.zeros(arch.n_layers, arch.n_heads, dtype=torch.float32)
    if ref_layer == 0:  # gömme çıktısı: hiçbir head onu etkileyemez
        return scores
    mu_neg = _last_token_z(lm, c.negative).mean(dim=0)  # [L, H, d]
    device = next(lm.model.parameters()).device
    r = r_ref.to(device, torch.float32)

    for p in c.positive:
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

        handles = [o_proj(lm.model, l).register_forward_pre_hook(make_hook(l)) for l in range(arch.n_layers)]
        try:
            with torch.enable_grad():
                hs = _hidden_states(lm, _encode(lm, p))
                m = hs[ref_layer][0, -1].to(torch.float32) @ r
                inputs = [deltas[l] for l in range(arch.n_layers)]
                grads = torch.autograd.grad(m, inputs, allow_unused=True)
        finally:
            for h in handles:
                h.remove()

        for layer, g in enumerate(grads):
            if g is None:  # ref_layer'dan sonraki katmanlar m'yi etkilemez
                continue
            g_last = g[0, -1].reshape(arch.n_heads, arch.head_dim).to("cpu", torch.float32)
            scores[layer] += (g_last * (mu_neg[layer] - zs[layer])).sum(dim=-1)

    return -scores / len(c.positive)


def head_attribution(lm: LoadedModel, c: Concept, ref_layer: int, normalize: bool = True) -> torch.Tensor:
    """[n_layers, n_heads]; −mean_pos(∇_{z_h} m · (μ_h^neg − z_h)), en büyük mutlak değere bölünmüş.

    Pozitif skor: head'i negatif ortalamasına yamamak m'yi düşürür (head kavramı taşır).
    r_ref = directions(lm, c)[ref_layer]. ref_layer == 0 ise sıfırlar.
    """
    r_ref = directions(lm, c)[ref_layer] if ref_layer > 0 else torch.zeros(lm.arch.d_model)
    raw = _attribution(lm, c, ref_layer, r_ref)
    return _normalize_abs(raw) if normalize else raw


def activation_diff(lm: LoadedModel, c: Concept) -> torch.Tensor:
    """[n_layers, n_heads]; |ort. head çıktı gücü(pos) − ort.(neg)| son token'da, en büyüğe bölünmüş."""

    def mean_power(prompts: list[str]) -> torch.Tensor:
        return torch.stack([head_output_power(lm, format_prompt(lm, p))[:, :, -1] for p in prompts]).mean(dim=0)

    diff = (mean_power(c.positive) - mean_power(c.negative)).abs()
    top = diff.max()
    return diff / top if top > 0 else torch.zeros_like(diff)


# --- Av ---------------------------------------------------------------------------------------


def hunt(
    lm: LoadedModel,
    c: Concept,
    method: Literal["attribution", "activation_diff", "direction"],
    seed: int = 0,
) -> HuntResult:
    """split → katman skorları → ref_layer = argmax → head sıralaması (train kümesinde).

    "direction" yönteminde head sırası activation_diff skorlarıdır; yön, ref_layer'daki train yönüdür.
    """
    if method not in ("attribution", "activation_diff", "direction"):
        raise ValueError(f"Bilinmeyen yöntem: {method!r}")
    train, val = split(c, seed=seed)
    dirs = _directions_from(last_token_resid(lm, train.positive), last_token_resid(lm, train.negative))
    d = _cohen_d(dirs, last_token_resid(lm, val.positive), last_token_resid(lm, val.negative))
    scores = _normalize_scores(d)
    ref_layer = max(range(len(scores)), key=lambda i: scores[i])

    if method == "attribution":
        head_scores = _normalize_abs(_attribution(lm, train, ref_layer, dirs[ref_layer]))
    else:
        head_scores = activation_diff(lm, train)

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
