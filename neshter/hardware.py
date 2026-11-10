"""Donanım yoklaması: GPU adı/VRAM ve RAM."""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import torch


@dataclass
class SystemInfo:
    gpu_name: str | None
    vram_bytes: int
    ram_bytes: int


def system_info() -> SystemInfo:
    ram = int(psutil.virtual_memory().total)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return SystemInfo(gpu_name=props.name, vram_bytes=int(props.total_memory), ram_bytes=ram)
    return SystemInfo(gpu_name=None, vram_bytes=0, ram_bytes=ram)


def cpu_has_native_bf16() -> bool:
    """CPU bf16 matmul'unu donanımda yapabiliyor mu (AVX512_BF16 veya AMX)? Yoklama başarısızsa False."""
    for name in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        try:
            if getattr(torch.cpu, name)():
                return True
        except Exception:
            continue
    return False


def available_ram_bytes() -> int:
    """Şu an kullanılabilir RAM (bayt)."""
    return int(psutil.virtual_memory().available)
