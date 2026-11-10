"""Güvenli model yükleyici: klasör (safetensors), .gguf ve yalnızca-ağırlık .pt."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from neshter.arch import ArchInfo, arch_info

SUPPORTED_FAMILIES = ("llama", "qwen2", "gemma2")
_CHUNK = 1024 * 1024


class UnsafeModelFile(Exception):
    """Dosya çalıştırılabilir kod içeriyor olabilir; yüklenmedi."""


class UnsupportedArchitecture(Exception):
    """Model mimarisi desteklenmiyor."""


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    arch: ArchInfo
    source_path: Path
    source_sha256: str
    license: str | None
    name: str


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """Dosya: içerik SHA-256. Klasör: sıralı (göreli yol + dosya hash'i) birleşiminin SHA-256'sı."""
    path = Path(path)
    if path.is_file():
        return _file_sha256(path)
    h = hashlib.sha256()
    files = sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix())
    for p in files:
        h.update(f"{p.relative_to(path).as_posix()}\0{_file_sha256(p)}\n".encode())
    return h.hexdigest()


def _unsupported(family: str) -> UnsupportedArchitecture:
    return UnsupportedArchitecture(
        f"'{family}' mimarisi desteklenmiyor. Desteklenen aileler: {', '.join(SUPPORTED_FAMILIES)}."
    )


def _check_family(family: str) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise _unsupported(family)


def _check_config_family(folder: Path) -> None:
    """config.json'daki model_type'ı, config sınıfı yüklenmeden önce kontrol eder."""
    try:
        family = json.loads((folder / "config.json").read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError, AttributeError) as e:
        raise UnsupportedArchitecture(f"config.json okunamadı, mimari desteklenmiyor: {e}") from e
    _check_family(str(family))


def _load_config(folder: Path):
    _check_config_family(folder)
    return AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)


def _read_license(folder: Path) -> str | None:
    try:
        value = json.loads((folder / "config.json").read_text(encoding="utf-8")).get("license")
        if isinstance(value, str) and value.strip():
            return value.strip()
    except (OSError, ValueError, AttributeError):
        pass
    try:
        text = (folder / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    front = re.match(r"\A---\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if front:
        m = re.search(r"^license:\s*['\"]?([^'\"\r\n#]+?)['\"]?\s*$", front.group(1), re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def _is_unsupported_gguf_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "architecture" in msg or "not supported" in msg


def _load_gguf(path: Path, dtype: torch.dtype):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path.parent, gguf_file=path.name, dtype=dtype, local_files_only=True, trust_remote_code=False
        )
    except (ValueError, NotImplementedError) as e:
        if _is_unsupported_gguf_error(e):
            raise UnsupportedArchitecture(f"Bu GGUF mimarisi desteklenmiyor ({e}).") from e
        raise
    tokenizer = AutoTokenizer.from_pretrained(
        path.parent, gguf_file=path.name, local_files_only=True, trust_remote_code=False
    )
    return model, tokenizer


def _load_pt(path: Path, dtype: torch.dtype):
    folder = path.parent
    if not (folder / "config.json").is_file():
        raise UnsupportedArchitecture(
            "Bu .pt dosyasının yanında config.json bulunamadı; mimari belirlenemediği için desteklenmiyor. "
            "Modeli config.json ve tokenizer ile aynı klasöre koyun."
        )
    config = _load_config(folder)
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except pickle.UnpicklingError as e:
        raise UnsafeModelFile(
            "Bu .pt dosyası çalıştırılabilir kod içeriyor (pickle) ve güvenli değil; yüklenmedi. "
            "Kaynağına güveniyorsanız önce safetensors biçimine dönüştürün, sonra yükleyin."
        ) from e
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=dtype)
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    return model, tokenizer


def _load_dir(path: Path, dtype: torch.dtype):
    _check_config_family(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tokenizer


def load_model(path: Path, device: str = "auto", dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    path = Path(path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if path.is_dir():
        model, tokenizer = _load_dir(path, dtype)
        name, license_ = path.name, _read_license(path)
    elif path.suffix.lower() == ".gguf":
        model, tokenizer = _load_gguf(path, dtype)
        name, license_ = path.stem, None
    elif path.suffix.lower() == ".pt":
        model, tokenizer = _load_pt(path, dtype)
        name, license_ = path.stem, _read_license(path.parent)
    else:
        raise UnsupportedArchitecture(
            f"'{path.suffix}' dosya biçimi desteklenmiyor. Klasör (safetensors), .gguf veya .pt kullanın."
        )

    _check_family(model.config.model_type)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        arch=arch_info(model),
        source_path=path,
        source_sha256=sha256_path(path),
        license=license_,
        name=name,
    )
