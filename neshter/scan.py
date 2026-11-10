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
    hf_home = os.environ.get("HF_HOME")
    return {
        "ollama": [Path(ollama) if ollama else home / ".ollama" / "models"],
        "lmstudio": [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"],
        "huggingface": [Path(hf_home) / "hub" if hf_home else home / ".cache" / "huggingface" / "hub"],
    }


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
