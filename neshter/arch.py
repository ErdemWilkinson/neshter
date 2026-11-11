"""Model mimarisi bilgisi ve katman erişimcileri (llama, qwen2, gemma2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn
from transformers import PreTrainedModel


@dataclass(frozen=True)
class ArchInfo:
    family: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_model: int
    d_mlp: int
    n_params: int


def decoder_layers(model) -> nn.ModuleList:
    return model.model.layers


def o_proj(model, layer: int) -> nn.Linear:
    return decoder_layers(model)[layer].self_attn.o_proj


def down_proj(model, layer: int) -> nn.Linear:
    return decoder_layers(model)[layer].mlp.down_proj


def embed(model) -> nn.Embedding:
    return model.model.embed_tokens


def arch_info(model: PreTrainedModel) -> ArchInfo:
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return ArchInfo(
        family=cfg.model_type,
        n_layers=cfg.num_hidden_layers,
        n_heads=n_heads,
        n_kv_heads=getattr(cfg, "num_key_value_heads", None) or n_heads,
        head_dim=head_dim,
        d_model=cfg.hidden_size,
        d_mlp=cfg.intermediate_size,
        n_params=sum(p.numel() for p in model.parameters()),
    )


def estimate_memory_bytes(n_params: int, dtype_bytes: int = 2) -> int:
    return int(n_params * dtype_bytes * 1.15)


def fit_badge(required: int, vram: int, ram: int) -> Literal["ok", "slow", "no"]:
    if required <= 0.9 * vram:
        return "ok"
    if vram <= 0:
        # No GPU: RAM is the only place the weights can live, so "slow, RAM backed" would only be noise.
        return "ok" if required <= 0.8 * ram else "no"
    if required <= 0.8 * (vram + ram):
        return "slow"
    return "no"
