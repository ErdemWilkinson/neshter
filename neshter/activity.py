"""Head etkinlik haritası ve en etkin MLP nöronları (arayüz ısı haritası ve 'etkinlik farkı' avı için)."""

from __future__ import annotations

import torch

from neshter.arch import down_proj, o_proj
from neshter.loader import LoadedModel


def _encode(lm: LoadedModel, prompt: str, add_special_tokens: bool = True):
    device = next(lm.model.parameters()).device
    return lm.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)


def tokens(lm: LoadedModel, prompt: str, add_special_tokens: bool = True) -> list[str]:
    """Prompt'un token'ları, görüntülenebilir metin olarak (bayt düzeyi işaretler boşluğa çevrilir)."""
    ids = lm.tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    pieces = lm.tokenizer.convert_ids_to_tokens(ids)
    return [lm.tokenizer.convert_tokens_to_string([p]) for p in pieces]


def head_grams(module, n_heads: int, head_dim: int) -> torch.Tensor:
    """o_proj için head başına Gram matrisi G_h = W_hᵀ W_h; şekil [H, d, d], float32.

    float32 ağırlık kopyası yalnızca bu çağrı içinde yaşar (katman başına, hepsi birden değil).
    """
    weight = module.weight.detach()
    w = weight.to(torch.float32).reshape(weight.shape[0], n_heads, head_dim)  # [d_model, H, d]
    return torch.einsum("mhd,mhe->hde", w, w)


def head_output_power(
    lm: LoadedModel,
    prompt: str,
    add_special_tokens: bool = True,
    grams: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Her (katman, head, token) için ‖W_o[:, h·d:(h+1)·d] · z_h‖₂; şekil [n_layers, n_heads, T], CPU float32.

    ‖W_h z_h‖² = z_hᵀ (W_hᵀ W_h) z_h olduğundan head başına küçük Gram matrisleri kullanılır: tüm
    katmanların float32 W_o kopyası ve [T, H, d_model] ara tensörü hiç oluşmaz. `grams` sözlüğü
    verilirse katman başına G önbelleğe alınır (aynı ağırlıklarla art arda çağrılar için; ağırlıklar
    değişirse çağıran yeni sözlük vermelidir).

    z, o_proj girdisinden okunur. Kanca çağrı içinde (varsa dıştaki apply() kancalarından sonra)
    kaydedilir; bu yüzden müdahale sonrası değerler ölçülür.
    """
    arch = lm.arch
    enc = _encode(lm, prompt, add_special_tokens)
    n_tokens = enc["input_ids"].shape[-1]
    power = torch.zeros(arch.n_layers, arch.n_heads, n_tokens, dtype=torch.float32)

    def make_hook(layer: int):
        def hook(module, args):
            if grams is not None and layer in grams:
                g = grams[layer]
            else:
                g = head_grams(module, arch.n_heads, arch.head_dim)
                if grams is not None:
                    grams[layer] = g
            z = args[0].detach().to(torch.float32)[0].reshape(-1, arch.n_heads, arch.head_dim)
            sq = torch.einsum("thd,hde,the->th", z, g.to(z.device), z)
            power[layer] = sq.clamp_min(0).sqrt().T.cpu()

        return hook

    handles = [
        o_proj(lm.model, layer).register_forward_pre_hook(make_hook(layer))
        for layer in range(arch.n_layers)
    ]
    try:
        with torch.no_grad():
            lm.model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return power


def normalize_per_layer(x: torch.Tensor) -> torch.Tensor:
    """[L, H] -> her satır kendi içinde min–max ile [0, 1]; sabit satır sıfır olur."""
    lo = x.min(dim=1, keepdim=True).values
    span = x.max(dim=1, keepdim=True).values - lo
    safe = torch.where(span > 0, span, torch.ones_like(span))
    return torch.where(span > 0, (x - lo) / safe, torch.zeros_like(x))


def normalize_global(x: torch.Tensor) -> torch.Tensor:
    """Tüm matris için tek min–max; sabit matris sıfır olur."""
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def top_mlp_neurons(
    lm: LoadedModel, prompt: str, token_index: int = -1, k: int = 50
) -> list[tuple[int, int, float]]:
    """down_proj girdisinin (aktivasyon sonrası MLP nöronları) token_index'teki mutlak değerine göre
    tüm katmanlardan en büyük k nöron: (katman, nöron, değer), azalan."""
    arch = lm.arch
    enc = _encode(lm, prompt)
    acts: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(module, args):
            acts[layer] = args[0].detach().to(torch.float32)[0, token_index].abs().cpu()

        return hook

    handles = [
        down_proj(lm.model, layer).register_forward_pre_hook(make_hook(layer))
        for layer in range(arch.n_layers)
    ]
    try:
        with torch.no_grad():
            lm.model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    all_acts = torch.stack([acts[layer] for layer in range(arch.n_layers)])  # [L, d_mlp]
    values, flat = all_acts.flatten().topk(min(k, all_acts.numel()))
    d_mlp = all_acts.shape[1]
    return [(int(i) // d_mlp, int(i) % d_mlp, float(v)) for i, v in zip(flat, values)]
