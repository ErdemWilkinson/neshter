"""Yerel model tarayıcı: Ollama, LM Studio ve Hugging Face önbelleği."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from neshter.licenses import (
    _MAX_CONFIG_BYTES,
    MAX_MANIFEST_BYTES,
    _read_folder_file,
    OLLAMA_MODEL_MEDIA,
    folder_license,
    license_id_from_text,
    ollama_layer_text,
    gguf_license_from,
    valid_digest,
)
from neshter.gguf_header import read_gguf_header


@dataclass
class ModelEntry:
    name: str
    source: Literal["ollama", "lmstudio", "huggingface", "file"]
    path: Path
    fmt: Literal["gguf", "safetensors", "pt"]
    size_bytes: int
    license: str | None = None
    arch: dict | None = None  # {family, n_layers, n_heads, n_kv_heads, d_mlp}; tarama sırasında bir kez okunur


def scan_ollama(root: Path) -> list[ModelEntry]:
    lib = Path(root) / "manifests" / "registry.ollama.ai" / "library"
    if not lib.is_dir():
        return []
    entries: list[ModelEntry] = []
    for manifest in sorted(p for p in lib.glob("*/*") if p.is_file()):
        try:
            if manifest.stat().st_size > MAX_MANIFEST_BYTES:
                continue
            data = json.loads(manifest.read_text(encoding="utf-8"))
            layer = next(l for l in data["layers"] if l["mediaType"] == OLLAMA_MODEL_MEDIA)
            digest = layer["digest"]
            if not valid_digest(digest):  # AS L7: blob yolu yalnızca doğrulanmış digest'ten kurulur
                continue
            blob = Path(root) / "blobs" / ("sha256-" + digest.removeprefix("sha256:"))
            size = blob.stat().st_size
        except (OSError, ValueError, KeyError, TypeError, StopIteration, AttributeError, RecursionError):
            continue
        lid, arch = gguf_info(blob)
        if lid is None:
            text = ollama_layer_text(Path(root), data)
            lid = license_id_from_text(text) if text else None
        entries.append(ModelEntry(f"{manifest.parent.name}:{manifest.name}", "ollama", blob, "gguf", size, lid, arch))
    return entries


def scan_lmstudio(root: Path) -> list[ModelEntry]:
    root = Path(root)
    if not root.is_dir():
        return []
    entries: list[ModelEntry] = []
    for f in sorted(root.rglob("*.gguf")):
        try:
            size = f.stat().st_size
        except OSError:
            continue
        lid, arch = gguf_info(f)
        entries.append(ModelEntry(f.stem, "lmstudio", f, "gguf", size, lid, arch))
    return entries


def scan_hf_cache(root: Path) -> list[ModelEntry]:
    root = Path(root)
    if not root.is_dir():
        return []
    entries: list[ModelEntry] = []
    for repo in sorted(root.glob("models--*")):
        name = repo.name.removeprefix("models--").replace("--", "/")
        for snap in sorted(p for p in (repo / "snapshots").glob("*") if p.is_dir()):
            weights = sorted(snap.glob("*.safetensors"))
            if not (snap / "config.json").is_file() or not weights:
                continue
            try:
                size = sum(w.stat().st_size for w in weights)
            except OSError:
                continue
            entries.append(ModelEntry(name, "huggingface", snap, "safetensors", size, folder_license(snap)[0], model_arch(snap, "safetensors")))
    return entries


def default_roots() -> dict[str, list[Path]]:
    home = Path.home()
    ollama = os.environ.get("OLLAMA_MODELS")
    hub_cache = os.environ.get("HF_HUB_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if hub_cache:
        hf = Path(hub_cache)
    elif hf_home:
        hf = Path(hf_home) / "hub"
    else:
        hf = home / ".cache" / "huggingface" / "hub"
    return {
        "ollama": [Path(ollama) if ollama else home / ".ollama" / "models"],
        "lmstudio": [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"],
        "huggingface": [hf],
    }


_ARCH_SUFFIXES = (".block_count", ".attention.head_count", ".attention.head_count_kv", ".feed_forward_length")
_LICENSE_KEYS = ("general.license", "general.license.name")


def _gguf_done(found: dict) -> bool:
    family = found.get("general.architecture")
    return (
        "general.license" in found
        and isinstance(family, str)
        and all(f"{family}{suffix}" in found for suffix in _ARCH_SUFFIXES)
    )


_MAX_ARCH_INT = 2**31


def _arch_int(v: object) -> int | None:
    """Güvenilmez üst veri sayısı: yalnızca sonlu, tam sayı değerli, 0..2^31 aralığında; aksi halde None.

    (inf/NaN için `int()` OverflowError/ValueError fırlatır; tek bir düşmanca değer listeyi düşürmemeli.)
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and not (math.isfinite(v) and v.is_integer()):
        return None
    return int(v) if 0 <= v <= _MAX_ARCH_INT else None


def _clean_arch(arch: dict) -> dict | None:
    if not isinstance(arch["family"], str):
        return None
    for k in ("n_layers", "n_heads", "n_kv_heads", "d_mlp"):
        arch[k] = _arch_int(arch[k])
    return arch


def _reject_constant(name: str):
    raise ValueError(f"JSON'da sayı olmayan sabit: {name}")


def gguf_info(path: Path) -> tuple[str | None, dict | None]:
    """(lisans kimliği, yapı bilgisi) — TEK sınırlı üst veri okumasıyla (gguf.GGUFReader kullanılmaz)."""
    meta = read_gguf_header(
        path, keys=("general.architecture", *_LICENSE_KEYS), suffixes=_ARCH_SUFFIXES, done=_gguf_done
    )
    if meta is None:
        return None, None
    family = meta.get("general.architecture")
    n_heads = meta.get(f"{family}.attention.head_count")
    arch = {
        "family": family,
        "n_layers": meta.get(f"{family}.block_count"),
        "n_heads": n_heads,
        "n_kv_heads": meta.get(f"{family}.attention.head_count_kv") or n_heads,
        "d_mlp": meta.get(f"{family}.feed_forward_length"),
    }
    return gguf_license_from(meta), _clean_arch(arch)


def model_arch(path: Path, fmt: str) -> dict | None:
    """Yükleme yapmadan yapı bilgisi: {family, n_layers, n_heads, n_kv_heads, d_mlp} ya da None.

    Klasör: config.json. GGUF: yalnızca sınırlı üst veri yürüyüşü (bkz. gguf_header).
    """
    if fmt == "gguf":
        return gguf_info(path)[1]
    try:
        folder = Path(path) if Path(path).is_dir() else Path(path).parent
        text = _read_folder_file(folder, "config.json", _MAX_CONFIG_BYTES)  # sınırlı, klasör dışına çıkmaz
        if text is None:
            return None
        cfg = json.loads(text, parse_constant=_reject_constant)  # NaN / Infinity reddedilir
        n_heads = cfg.get("num_attention_heads")
        return _clean_arch(
            {
                "family": cfg.get("model_type"),
                "n_layers": cfg.get("num_hidden_layers"),
                "n_heads": n_heads,
                "n_kv_heads": cfg.get("num_key_value_heads") or n_heads,
                "d_mlp": cfg.get("intermediate_size"),
            }
        )
    except Exception:  # bozuk/eksik/düşmanca üst veri: liste yine gösterilir, yapı bilinmiyor
        return None


def scan_all(roots: dict[str, list[Path]] | None = None) -> list[ModelEntry]:
    roots = default_roots() if roots is None else roots
    scanners = {"ollama": scan_ollama, "lmstudio": scan_lmstudio, "huggingface": scan_hf_cache}
    entries: list[ModelEntry] = []
    for source, scan in scanners.items():
        for root in roots.get(source, []):
            try:
                entries.extend(scan(Path(root)))
            except (OSError, ValueError, OverflowError, RecursionError):  # son savunma: diğer kaynaklar listelenir
                continue
    return entries
