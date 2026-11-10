"""Hook tabanlı müdahaleler: head susturma ve rezidüel akımda yön silme.

Bu modüldeki matematik, kayıt (export) aşamasında ağırlıklara işlenecek olanla birebir aynı olmalıdır:
  - HeadAblation: o_proj girdisinin ilgili head dilimi 0 (ya da ortalama) -> o_proj sütunlarını sıfırlamaya denk.
  - DirectionAblation: tüm yönler TEK izdüşümle silinir. Q = direction_basis(r_1..r_k), yönlerin
    gerdiği uzayın dik birim tabanıdır (SVD; kopya/bağımlı yönler tabanı büyütmez). Rezidüel akıma
    YAZAN her modülün çıktısında x <- x - (x S) S^T: embed, her katmanda o_proj ve down_proj
    (her aile için aynı modüller; taban seçimi writer_bases). Böylece katman içi ara rezidüel
    (MLP girdisi) dahil rezidüel akım her noktada HER r_i'ye diktir; yönlerin sırası önemsiz
    (eski "liste sırasında sıralı bileşim" kuralının yerini alır: ilintili yönlerde sıralı izdüşüm
    önceki yönü geri getirebiliyordu). Sonuç ağırlık dikleştirmesiyle birebir aynıdır.
    gemma2: o_proj/down_proj çıktısı rezidüele γ ⊙ o / rms(o) olarak eklenir (post-norm,
    γ = 1 + weight, float32). rᵀ(γ ⊙ v) = (γ ⊙ r)ᵀ v ve rms(o) > 0 skaler olduğundan
    çıktı ⟂ r_i ⇔ o ⟂ γ ⊙ r_i (eps ve rms ne olursa olsun, tam). Bu yüzden o_proj/down_proj için
    taban S = direction_basis([γ ⊙ r_i]) kullanılır (post-attention / post-feedforward normunun γ'sı);
    gömme için düz Q (embed_scale skalerdir). Norm, izdüşülmüş vektörün rms'ini hesaplar, böylece
    kanca ağırlık dikleştirmesiyle birebir aynıdır.
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
    """Çıktısı rezidüel akıma yazan modüller (yön silme kancalarının yeri), her aile için
    [embed, o_proj_0, down_proj_0, ...]. gemma2'de çıktı rezidüele post-norm üzerinden gider;
    bunu writer_bases γ-ölçekli tabanla hesaba katar. `family` imza uyumu için tutulur."""
    mods: list[nn.Module] = [embed(model)]
    for i in range(len(decoder_layers(model))):
        mods += [o_proj(model, i), down_proj(model, i)]
    return mods


def direction_basis(directions: Sequence[torch.Tensor], rtol: float = 1e-6) -> torch.Tensor:
    """Yönlerin gerdiği uzayın dik birim tabanı Q [d, k'] (float32, sütunlar ortonormal).

    Girdiler önce birim uzunluğa getirilir (normalize edilmemiş vektör kabul edilir); normu sıfır
    olanlar atlanır (k' = 0 olabilir). Ayrıştırma float64'te SVD ile yapılır: S_j > rtol · max(S)
    olan tekil vektörler tutulur, böylece kopya ve doğrusal bağımlı yönler tabanı büyütmez ve
    sıralarından bağımsız olarak sonraki yönler kaybolmaz.
    """
    if not directions:
        raise ValueError("En az bir yön vektörü gerekli.")
    cols: list[torch.Tensor] = []
    size: int | None = None
    for v in directions:
        v = torch.as_tensor(v).detach().to(device="cpu", dtype=torch.float64)
        if v.ndim != 1:
            raise ValueError(f"Yön vektörü tek boyutlu olmalı, şekil: {tuple(v.shape)}.")
        if size is None:
            size = v.numel()
        elif v.numel() != size:
            raise ValueError(f"Yön vektörlerinin uzunlukları farklı: {size} ve {v.numel()}.")
        norm = v.norm()
        if not torch.isfinite(norm):
            raise ValueError("Yön vektörünün normu sonlu değil.")
        if norm > 0:
            cols.append(v / norm)
    if not cols:
        return torch.zeros(size, 0, dtype=torch.float32)
    U, S, _ = torch.linalg.svd(torch.stack(cols, dim=1), full_matrices=False)
    keep = S > rtol * S.max()
    return U[:, keep].to(torch.float32).contiguous()


def _gamma(norm: nn.Module) -> torch.Tensor:
    """Gemma2RMSNorm çarpanı: γ = 1 + weight (float32, transformers Gemma2RMSNorm.forward ile aynı)."""
    return 1.0 + norm.weight.detach().to(device="cpu", dtype=torch.float32)


def writer_bases(
    model, family: str, directions: Sequence[torch.Tensor]
) -> list[tuple[nn.Module, torch.Tensor]]:
    """residual_writers(model, family) içindeki her modül için (modül, taban) çiftleri.

    Kanca (apply) ve gömme (export.bake) aynı listeyi kullanır; böylece ikisi birebir eşdeğer kalır.
    llama/qwen2: her modül Q = direction_basis(directions) alır. gemma2: gömme Q alır; katman ℓ'de
    o_proj S = direction_basis([γ_attn,ℓ ⊙ r_i]), down_proj S = direction_basis([γ_mlp,ℓ ⊙ r_i])
    (γ = 1 + post_attention_layernorm / post_feedforward_layernorm ağırlığı). γ'nın sıfır olduğu
    eksenler SVD rank düşüşüyle ele alınır (taban k' = 0 sütun olabilir).
    """
    basis = direction_basis(directions)
    if family != "gemma2":
        return [(module, basis) for module in residual_writers(model, family)]
    rs = [torch.as_tensor(r).detach().to(device="cpu", dtype=torch.float32) for r in directions]
    out: list[tuple[nn.Module, torch.Tensor]] = [(embed(model), basis)]
    for i, layer in enumerate(decoder_layers(model)):
        for module, norm in (
            (o_proj(model, i), layer.post_attention_layernorm),
            (down_proj(model, i), layer.post_feedforward_layernorm),
        ):
            gamma = _gamma(norm)
            out.append((module, direction_basis([gamma * r for r in rs])))
    return out


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


def _project_out(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    # x <- x - (x Q) Q^T; float32'de hesaplanır, girdinin dtype'ına döndürülür.
    xf = x.to(torch.float32)
    q = basis.to(device=x.device, dtype=torch.float32)
    return (xf - (xf @ q) @ q.T).to(x.dtype)


def _direction_hook(basis: torch.Tensor, batch_index: int | None):
    """Forward hook (basis: [d, k'] dik birim taban). Çıktı Tensor ya da tuple (ilk eleman gizli
    durum) olabilir; aynı tip döner."""

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if batch_index is None:
            new = _project_out(h, basis)
        else:
            _check_batch(h, batch_index)
            new = h.clone()
            new[batch_index] = _project_out(h[batch_index], basis)
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
        bases = writer_bases(model, arch.family, directions) if directions else []
        for module, basis in bases:  # yazıcı başına tek kanca: tüm yönlerin gerdiği uzay silinir
            if basis.shape[1]:
                handles.append(module.register_forward_hook(_direction_hook(basis, batch_index)))
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
