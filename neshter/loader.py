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
from neshter.hardware import available_ram_bytes, cpu_has_native_bf16

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
    device_note: str | None = None


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


def read_license(folder: Path) -> str | None:
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


_read_license = read_license  # geriye uyumluluk (testler)


_GGUF_MAGIC = b"GGUF"


def is_gguf(path: Path) -> bool:
    """Dosya GGUF sihirli baytlarıyla mı başlıyor? (Ollama blob'larının uzantısı yoktur.)"""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _GGUF_MAGIC
    except OSError:
        return False


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
    # Klasör = safetensors: yalnızca pytorch_model.bin içeren klasör pickle yoluna düşmesin.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
    except OSError as e:
        if not any(path.glob("*.safetensors")):
            raise UnsupportedArchitecture(
                "Bu klasörde .safetensors ağırlık dosyası yok; klasörler yalnızca safetensors biçiminde "
                "yüklenir. Modeli safetensors'a dönüştürüp yeniden deneyin."
            ) from e
        raise
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tokenizer


GPU_FALLBACK_NOTE = "GPU belleği yetmedi; CPU'da çalışıyor (yavaş)."


SLOW_BF16_NOTE = "Bu işlemcide bf16 hesaplama yavaş ve float32 için RAM yetmiyor; işlemler uzun sürebilir."


def pick_compute_dtype(device: str, n_params: int, available_ram: int, native_bf16: bool) -> torch.dtype:
    """Hesaplama dtype'ı: yerel bf16'sı olmayan CPU'da (7-54x yavaş) RAM yeterse float32, aksi halde bf16."""
    if device == "cpu" and not native_bf16 and n_params * 4 * 1.2 < 0.7 * available_ram:
        return torch.float32
    return torch.bfloat16


def _probe_model(path: Path, dtype: torch.dtype):
    """Parametre sayısı/boyutu için meta cihazda iskelet model (ağırlık okunmaz).

    GGUF'ta config ağırlık dosyasının içindedir; iskelet kurulamaz, None döner.
    """
    if path.is_dir():
        folder = path
    elif path.suffix.lower() == ".pt" and (path.parent / "config.json").is_file():
        folder = path.parent
    else:
        return None
    config = _load_config(folder)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=dtype)


def _pick_device(model, device: str) -> tuple[str, str | None]:
    """device="auto": CUDA varsa ve model tahmini boş VRAM'in %90'ına sığıyorsa cuda, değilse cpu.

    Sığmazsa CPU'da kalınır ve kullanıcıya gösterilecek not döner (v0.1'de katman bölme yok).
    """
    if device != "auto":
        return device, None
    if not torch.cuda.is_available():
        return "cpu", None
    from neshter.arch import estimate_memory_bytes

    n_params = sum(p.numel() for p in model.parameters())
    dtype_bytes = next(model.parameters()).element_size()
    need = estimate_memory_bytes(n_params, dtype_bytes)
    free, _total = torch.cuda.mem_get_info()
    if need > 0.9 * free:
        return "cpu", GPU_FALLBACK_NOTE
    return "cuda", None


def load_model(
    path: Path, device: str = "auto", dtype: torch.dtype | None = None, name: str | None = None
) -> LoadedModel:
    """Modeli yükler. `dtype=None` hesaplama politikasıdır (`pick_compute_dtype`): GPU'da bf16; yerel bf16'sı
    olmayan CPU'da RAM yeterse float32, değilse bf16 (+ yavaşlık notu). Açık dtype aynen uygulanır.

    Ağırlıklar doğrudan hedef dtype'ta okunur (bf16'ya yuvarlayıp yükseltmek yok). Kayıt dtype'ı bundan
    bağımsızdır (`export.save_dtype`). `name` verilirse görünen ad odur (ör. Ollama'nın "qwen2.5:7b"si);
    yoksa dosya/klasör adı. Uzantısız dosyalar (Ollama blob'ları) GGUF sihirli baytlarıyla tanınır.
    """
    path = Path(path)
    is_dir = path.is_dir()
    is_gguf_file = not is_dir and (path.suffix.lower() == ".gguf" or (path.is_file() and is_gguf(path)))
    is_pt = not is_dir and not is_gguf_file and path.suffix.lower() == ".pt"
    if not (is_dir or is_gguf_file or is_pt):
        raise UnsupportedArchitecture(
            f"'{path.suffix}' dosya biçimi desteklenmiyor. Klasör (safetensors), .gguf veya .pt kullanın."
        )

    # Klasör/.pt: cihaz ve dtype ağırlıklar okunmadan, meta iskeletinden seçilir.
    probe = _probe_model(path, dtype or torch.bfloat16)
    device_note = None
    if probe is not None:
        device, device_note = _pick_device(probe, device)
        if dtype is None:
            n_params = sum(p.numel() for p in probe.parameters())
            native = cpu_has_native_bf16()
            dtype = pick_compute_dtype(device, n_params, available_ram_bytes(), native)
            if device == "cpu" and dtype == torch.bfloat16 and not native:
                device_note = f"{device_note} {SLOW_BF16_NOTE}" if device_note else SLOW_BF16_NOTE
        del probe
    elif dtype is None:
        dtype = torch.bfloat16  # GGUF: bf16'ya açılır

    if is_dir:
        model, tokenizer = _load_dir(path, dtype)
        default_name, license_ = path.name, read_license(path)
    elif is_gguf_file:
        model, tokenizer = _load_gguf(path, dtype)
        default_name, license_ = path.stem, None
    else:
        model, tokenizer = _load_pt(path, dtype)
        default_name, license_ = path.stem, read_license(path.parent)

    _check_family(model.config.model_type)
    if not (is_dir or is_pt):  # GGUF: iskelet yok, cihaz yüklemeden sonra seçilir
        device, device_note = _pick_device(model, device)
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
        name=name or default_name,
        device_note=device_note,
    )
