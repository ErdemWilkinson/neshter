"""Head etkinlik haritası ve en etkin MLP nöronları (arayüz ısı haritası ve 'etkinlik farkı' avı için)."""

from __future__ import annotations

import torch

from neshter.arch import down_proj, o_proj
from neshter.loader import LoadedModel


def _encode(lm: LoadedModel, prompt: str):
    device = next(lm.model.parameters()).device
    return lm.tokenizer(prompt, return_tensors="pt").to(device)


def tokens(lm: LoadedModel, prompt: str) -> list[str]:
    """Prompt'un token'ları, görüntülenebilir metin olarak (bayt düzeyi işaretler boşluğa çevrilir)."""
    ids = lm.tokenizer(prompt)["input_ids"]
    pieces = lm.tokenizer.convert_ids_to_tokens(ids)
    return [lm.tokenizer.convert_tokens_to_string([p]) for p in pieces]


def head_output_power(lm: LoadedModel, prompt: str) -> torch.Tensor:
    """Her (katman, head, token) için ‖W_o[:, h·d:(h+1)·d] · z_h‖₂; şekil [n_layers, n_heads, T], CPU float32.

    z, o_proj girdisinden okunur. Kanca çağrı içinde (varsa dıştaki apply() kancalarından sonra)
    kaydedilir; bu yüzden müdahale sonrası değerler ölçülür.
    """
    arch = lm.arch
    enc = _encode(lm, prompt)
    n_tokens = enc["input_ids"].shape[-1]
    power = torch.zeros(arch.n_layers, arch.n_heads, n_tokens, dtype=torch.float32)

    def make_hook(layer: int):
        weight = o_proj(lm.model, layer).weight.detach().to(torch.float32)
        w = weight.reshape(weight.shape[0], arch.n_heads, arch.head_dim)  # [d_model, H, d]

        def hook(module, args):
            z = args[0].detach().to(torch.float32)[0].reshape(-1, arch.n_heads, arch.head_dim)
            out = torch.einsum("thd,mhd->thm", z, w)  # her head'in artık akışa kattığı vektör
            power[layer] = out.norm(dim=-1).T.cpu()

        return hook

    handles = [
        o_proj(lm.model, layer).register_forward_pre_hook(make_hook(layer))
        for layer in range(arch.n_layers)
    ]
    try:
        with torch.no_grad():
            lm.model(**enc)
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
            lm.model(**enc)
    finally:
        for h in handles:
            h.remove()

    all_acts = torch.stack([acts[layer] for layer in range(arch.n_layers)])  # [L, d_mlp]
    values, flat = all_acts.flatten().topk(min(k, all_acts.numel()))
    d_mlp = all_acts.shape[1]
    return [(int(i) // d_mlp, int(i) % d_mlp, float(v)) for i, v in zip(flat, values)]
