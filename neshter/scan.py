"""Yerel model tarayıcı: Ollama, LM Studio ve Hugging Face önbelleği."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_OLLAMA_MODEL_MEDIA = "application/vnd.ollama.image.model"


@dataclass
class ModelEntry:
    name: str
    source: Literal["ollama", "lmstudio", "huggingface", "file"]
    path: Path
    fmt: Literal["gguf", "safetensors", "pt"]
    size_bytes: int


def scan_ollama(root: Path) -> list[ModelEntry]:
    lib = Path(root) / "manifests" / "registry.ollama.ai" / "library"
    if not lib.is_dir():
        return []
    entries: list[ModelEntry] = []
    for manifest in sorted(p for p in lib.glob("*/*") if p.is_file()):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            layer = next(l for l in data["layers"] if l["mediaType"] == _OLLAMA_MODEL_MEDIA)
            digest = layer["digest"]
            if not digest.startswith("sha256:"):
                continue
            blob = Path(root) / "blobs" / ("sha256-" + digest.removeprefix("sha256:"))
            size = blob.stat().st_size
        except (OSError, ValueError, KeyError, TypeError, StopIteration, AttributeError):
            continue
        entries.append(ModelEntry(f"{manifest.parent.name}:{manifest.name}", "ollama", blob, "gguf", size))
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
        entries.append(ModelEntry(f.stem, "lmstudio", f, "gguf", size))
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
            entries.append(ModelEntry(name, "huggingface", snap, "safetensors", size))
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


def _gguf_value(reader, key: str):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def model_arch(path: Path, fmt: str) -> dict | None:
    """Yükleme yapmadan yapı bilgisi: {family, n_layers, n_heads, n_kv_heads, d_mlp} ya da None.

    Klasör: config.json. GGUF: yalnızca üst veri (gguf.GGUFReader dosyayı belleğe eşler, okumaz).
    """
    try:
        if fmt == "gguf":
            from gguf import GGUFReader

            reader = GGUFReader(str(path))
            family = _gguf_value(reader, "general.architecture")
            n_heads = _gguf_value(reader, f"{family}.attention.head_count")
            arch = {
                "family": family,
                "n_layers": _gguf_value(reader, f"{family}.block_count"),
                "n_heads": n_heads,
                "n_kv_heads": _gguf_value(reader, f"{family}.attention.head_count_kv") or n_heads,
                "d_mlp": _gguf_value(reader, f"{family}.feed_forward_length"),
            }
        else:
            folder = Path(path) if Path(path).is_dir() else Path(path).parent
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            n_heads = cfg.get("num_attention_heads")
            arch = {
                "family": cfg.get("model_type"),
                "n_layers": cfg.get("num_hidden_layers"),
                "n_heads": n_heads,
                "n_kv_heads": cfg.get("num_key_value_heads") or n_heads,
                "d_mlp": cfg.get("intermediate_size"),
            }
    except Exception:  # bozuk/eksik üst veri: liste yine gösterilir, yapı bilinmiyor
        return None
    if not isinstance(arch["family"], str):
        return None
    for k in ("n_layers", "n_heads", "n_kv_heads", "d_mlp"):
        v = arch[k]
        arch[k] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return arch


def scan_all(roots: dict[str, list[Path]] | None = None) -> list[ModelEntry]:
    roots = default_roots() if roots is None else roots
    scanners = {"ollama": scan_ollama, "lmstudio": scan_lmstudio, "huggingface": scan_hf_cache}
    entries: list[ModelEntry] = []
    for source, scan in scanners.items():
        for root in roots.get(source, []):
            try:
                entries.extend(scan(Path(root)))
            except OSError:
                continue
    return entries
