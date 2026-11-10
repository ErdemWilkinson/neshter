"""Hook tabanlı müdahaleler: head susturma ve rezidüel akımda yön silme.

Bu modüldeki matematik, kayıt (export) aşamasında ağırlıklara işlenecek olanla birebir aynı olmalıdır:
  - HeadAblation: o_proj girdisinin ilgili head dilimi 0 (ya da ortalama) -> o_proj sütunlarını sıfırlamaya denk.
  - DirectionAblation: rezidüel akıma YAZAN her modülün çıktısında x <- x - (x . r) r:
    embed, her katmanda o_proj ve down_proj (gemma2: post_attention_layernorm ve
    post_feedforward_layernorm). Böylece katman içi ara rezidüel (MLP girdisi) dahil rezidüel akım
    her noktada r'ye diktir ve sonuç ağırlık dikleştirmesiyle birebir aynıdır.
    Son norm çıktısına dokunulmaz.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

from torch import nn

from neshter.arch import decoder_layers, down_proj, embed, o_proj
from neshter.loader import LoadedModel


def residual_writers(model, family: str) -> list[nn.Module]:
    """Çıktısı doğrudan rezidüel akıma eklenen modüller (yön silme kancalarının yeri)."""
    mods: list[nn.Module] = [embed(model)]
    for i, layer in enumerate(decoder_layers(model)):
        if family == "gemma2":
            mods += [layer.post_attention_layernorm, layer.post_feedforward_layernorm]
        else:
            mods += [o_proj(model, i), down_proj(model, i)]
    return mods


@dataclass(frozen=True)
class HeadAblation:
    layer: int
    head: int


@dataclass(eq=False)
class DirectionAblation:
    direction: torch.Tensor  # (d_model,), kurucuda birim uzunluğa normalize edilir

    def __post_init__(self) -> None:
        d = torch.as_tensor(self.direction).detach().to(torch.float32).clone()
        if d.ndim != 1:
            raise ValueError(f"Yön vektörü tek boyutlu olmalı, şekil: {tuple(d.shape)}.")
        norm = d.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("Yön vektörünün normu sıfır ya da sonlu değil; normalize edilemez.")
        self.direction = d / norm


Intervention = HeadAblation | DirectionAblation


def _check_batch(x: torch.Tensor, batch_index: int) -> None:
    if batch_index >= x.shape[0]:
        raise ValueError(f"batch_index={batch_index}, ancak yığın boyutu {x.shape[0]}.")


def _project_out(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # float32'de hesaplanır, girdinin dtype'ına döndürülür.
    xf = x.to(torch.float32)
    r = r.to(device=x.device, dtype=torch.float32)
    return (xf - (xf @ r).unsqueeze(-1) * r).to(x.dtype)


def _direction_hook(direction: torch.Tensor, batch_index: int | None):
    """Forward hook: çıktı Tensor ya da tuple (ilk eleman gizli durum) olabilir; aynı tip döner."""

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if batch_index is None:
            new = _project_out(h, direction)
        else:
            _check_batch(h, batch_index)
            new = h.clone()
            new[batch_index] = _project_out(h[batch_index], direction)
        if isinstance(output, tuple):
            return (new, *output[1:])
        return new

    return hook


def _head_pre_hook(
    replacements: dict[int, torch.Tensor | None], head_dim: int, batch_index: int | None
):
    """o_proj forward pre-hook. replacements: head -> ortalama vektörü (None = sıfır)."""

    def hook(module, args):
        x = args[0]
        new = x.clone()
        if batch_index is not None:
            _check_batch(x, batch_index)
        for head, mean in replacements.items():
            sl = slice(head * head_dim, (head + 1) * head_dim)
            value = 0 if mean is None else mean.to(device=x.device, dtype=x.dtype)
            if batch_index is None:
                new[..., sl] = value
            else:
                new[batch_index, ..., sl] = value
        return (new, *args[1:])

    return hook


def _validate(
    lm: LoadedModel,
    interventions: Sequence[Intervention],
    batch_index: int | None,
    head_mode: str,
    head_means_: dict[tuple[int, int], torch.Tensor] | None,
) -> None:
    arch = lm.arch
    if head_mode not in ("zero", "mean"):
        raise ValueError(f"Geçersiz head_mode '{head_mode}'; 'zero' ya da 'mean' olmalı.")
    if batch_index is not None and batch_index < 0:
        raise ValueError(f"batch_index negatif olamaz: {batch_index}.")
    for iv in interventions:
        if isinstance(iv, HeadAblation):
            if not 0 <= iv.layer < arch.n_layers:
                raise ValueError(f"Katman {iv.layer} aralık dışı (0..{arch.n_layers - 1}).")
            if not 0 <= iv.head < arch.n_heads:
                raise ValueError(f"Head {iv.head} aralık dışı (0..{arch.n_heads - 1}).")
            if head_mode == "mean":
                mean = None if head_means_ is None else head_means_.get((iv.layer, iv.head))
                if mean is None:
                    raise ValueError(
                        f"'mean' kipi için ({iv.layer}, {iv.head}) head'inin ortalaması verilmemiş."
                    )
                if tuple(mean.shape) != (arch.head_dim,):
                    raise ValueError(
                        f"({iv.layer}, {iv.head}) ortalamasının şekli {tuple(mean.shape)}; "
                        f"({arch.head_dim},) olmalı."
                    )
        elif isinstance(iv, DirectionAblation):
            if iv.direction.shape != (arch.d_model,):
                raise ValueError(
                    f"Yön vektörü uzunluğu {iv.direction.numel()}; model boyutu {arch.d_model} olmalı."
                )
        else:
            raise ValueError(f"Bilinmeyen müdahale türü: {type(iv).__name__}.")


@contextmanager
def apply(
    lm: LoadedModel,
    interventions: Sequence[Intervention],
    batch_index: int | None = None,
    head_mode: Literal["zero", "mean"] = "zero",
    head_means: dict[tuple[int, int], torch.Tensor] | None = None,
):
    """Müdahaleleri kanca olarak kurar; çıkışta (istisna dahil) hepsini söker.

    Tüm doğrulama hiçbir kanca kurulmadan önce yapılır (ValueError).
    """
    _validate(lm, interventions, batch_index, head_mode, head_means)
    model, arch = lm.model, lm.arch

    per_layer: dict[int, dict[int, torch.Tensor | None]] = {}
    directions: list[torch.Tensor] = []
    for iv in interventions:
        if isinstance(iv, HeadAblation):
            mean = head_means[(iv.layer, iv.head)] if head_mode == "mean" else None
            per_layer.setdefault(iv.layer, {})[iv.head] = mean
        else:
            directions.append(iv.direction)

    handles = []
    try:
        for layer, replacements in per_layer.items():
            handles.append(
                o_proj(model, layer).register_forward_pre_hook(
                    _head_pre_hook(replacements, arch.head_dim, batch_index)
                )
            )
        writers = residual_writers(model, arch.family) if directions else []
        for direction in directions:  # her modülde kayıt sırası = liste sırası (sıralı bileşim)
            hook = _direction_hook(direction, batch_index)
            for module in writers:
                handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def head_means(lm: LoadedModel, prompts: list[str]) -> dict[tuple[int, int], torch.Tensor]:
    """Her (layer, head) için o_proj girdisi diliminin, tüm prompt'ların tüm token'ları üzerinden ortalaması.

    Ham metin kullanılır (sohbet şablonu yok). Dönen vektörler float32, şekil (head_dim,).
    """
    if not prompts:
        raise ValueError("En az bir prompt gerekli.")
    arch = lm.arch
    sums = {
        layer: torch.zeros(arch.n_heads, arch.head_dim, dtype=torch.float32)
        for layer in range(arch.n_layers)
    }
    count = 0

    def make_hook(layer: int):
        def hook(module, args):
            x = args[0].detach().to(torch.float32)
            sums[layer] += x.reshape(-1, arch.n_heads, arch.head_dim).sum(0).cpu()

        return hook

    handles = [
        o_proj(lm.model, layer).register_forward_pre_hook(make_hook(layer))
        for layer in range(arch.n_layers)
    ]
    try:
        with torch.no_grad():
            device = next(lm.model.parameters()).device
            for prompt in prompts:
                enc = lm.tokenizer(prompt, return_tensors="pt").to(device)
                n_tokens = enc["input_ids"].numel()  # tek prompt, dolgu yok
                if n_tokens == 0:
                    raise ValueError("Boş prompt (token yok) ortalamaya katılamaz.")
                lm.model.model(**enc)  # lm_head gerekmez
                count += n_tokens
    finally:
        for h in handles:
            h.remove()

    return {
        (layer, head): sums[layer][head] / count
        for layer in range(arch.n_layers)
        for head in range(arch.n_heads)
    }
