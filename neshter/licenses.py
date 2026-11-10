"""Lisans okuma: HF klasörü, GGUF üst verisi, Ollama lisans katmanı.

Yaprak modül: `loader` ve `scan` bunu içe aktarır, kendisi ikisinden de bir şey almaz.
Modelin kendi dosyalarından gelen lisans metni güvenilmez görüntü verisidir: hiçbir yol bundan kurulmaz,
okunan metin sınırlıdır (256 KiB) ve yalnızca modelin kendi klasöründen / Ollama blob'larından okunur.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_LICENSE_BYTES = 256 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ID_CHARS = 128

OLLAMA_MODEL_MEDIA = "application/vnd.ollama.image.model"
OLLAMA_LICENSE_MEDIA = "application/vnd.ollama.image.license"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_BLOB_NAME = re.compile(r"sha256-[0-9a-f]{64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

# Sıra önemlidir: ilk eşleşme kazanır. \b: "permit license" / "submit license" "mit license" sayılmaz.
_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    ((r"\bapache\s+license\b", r"\bversion\s+2\.0\b"), "apache-2.0"),
    ((r"\bmit\s+license\b",), "mit"),
    ((r"\bllama\s+3\.3\b",), "llama3.3"),
    ((r"\bllama\s+3\.2\b",), "llama3.2"),
    ((r"\bllama\s+3\.1\b",), "llama3.1"),
    ((r"\bmeta\s+llama\s+3\b",), "llama3"),
    ((r"\bllama\s+3\s+community\b",), "llama3"),
    ((r"\bgemma\s+terms\s+of\s+use\b",), "gemma"),
    ((r"\bqwen\s+research\s+license\b",), "qwen-research"),
    ((r"\bqwen\s+license\b",), "qwen"),
    ((r"\btongyi\s+qianwen\b",), "qwen"),
)


def license_id_from_text(text: str) -> str:
    """Lisans metninden kısa kimlik; tanınmazsa "other"."""
    for patterns, lid in _RULES:
        if all(re.search(p, text, re.IGNORECASE) for p in patterns):
            return lid
    return "other"


def clean_id(value: str) -> str:
    """Üst veriden gelen kimlik: denetim karakterleri atılır, uzunluk sınırlanır."""
    return _CONTROL.sub(" ", value).strip()[:_MAX_ID_CHARS].strip()


def valid_digest(digest: object) -> bool:
    """AS L7: yalnızca ^sha256:[0-9a-f]{64}$ (sondaki satır sonu da geçersiz)."""
    return isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None


def read_license(folder: Path) -> str | None:
    """config.json'daki `license`, yoksa README.md ön yazısındaki `license:`."""
    try:
        value = json.loads((folder / "config.json").read_text(encoding="utf-8")).get("license")
        if isinstance(value, str) and value.strip():
            return clean_id(value) or None
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
            return clean_id(m.group(1)) or None
    return None


def _read_bounded(path: Path, limit: int = MAX_LICENSE_BYTES) -> str:
    with open(path, "rb") as f:
        return f.read(limit).decode("utf-8", errors="replace")


def _inside_model(folder: Path, f: Path) -> bool:
    """Dosya gerçekten modelin klasöründe mi? HF snapshot dosyaları kardeş `blobs/` dizinine sembolik
    bağdır (repo/snapshots/rev/F -> repo/blobs/hash); yalnızca o düzene izin verilir."""
    try:
        real = f.resolve(strict=True)
        if not real.is_file():
            return False
        if real.is_relative_to(folder.resolve()):
            return True
        if folder.parent.name == "snapshots":
            return real.is_relative_to((folder.parent.parent / "blobs").resolve())
    except (OSError, ValueError):
        pass
    return False


def read_license_text(folder: Path) -> str | None:
    """İlk `LICENSE*` dosyası (en çok 256 KiB, UTF-8, bozuk baytlar değiştirilir); yoksa None."""
    folder = Path(folder)
    try:
        candidates = sorted(folder.glob("LICENSE*"))
    except OSError:
        return None
    for f in candidates:
        if not _inside_model(folder, f):
            continue
        try:
            return _read_bounded(f)
        except OSError:
            continue
    return None


def folder_license(folder: Path) -> tuple[str | None, str | None]:
    """(kimlik, metin). README/config kimliği yoksa kimlik `LICENSE*` metninden çıkarılır."""
    text = read_license_text(folder)
    lid = read_license(folder)
    if lid is None and text and text.strip():
        lid = license_id_from_text(text)
    return lid, text


def _gguf_value(reader, key: str):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def read_gguf_license(path: Path) -> str | None:
    """GGUF üst verisinden `general.license`, yoksa `general.license.name`."""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        for key in ("general.license", "general.license.name"):
            value = _gguf_value(reader, key)
            if isinstance(value, str) and clean_id(value):
                return clean_id(value)
    except Exception:  # bozuk/eksik üst veri: lisans bilinmiyor
        pass
    return None


def ollama_layer_text(root: Path, manifest: object) -> str | None:
    """Manifest'teki `application/vnd.ollama.image.license` katmanının metni (en çok 256 KiB)."""
    try:
        layers = manifest["layers"]  # type: ignore[index]
        for layer in layers:
            if layer["mediaType"] != OLLAMA_LICENSE_MEDIA or not valid_digest(layer["digest"]):
                continue
            blob = Path(root) / "blobs" / ("sha256-" + layer["digest"].removeprefix("sha256:"))
            if blob.is_file():
                text = _read_bounded(blob)
                if text.strip():
                    return text
    except (OSError, KeyError, TypeError):
        pass
    return None


def ollama_license_text(blob: Path) -> str | None:
    """Model blob'una ait Ollama manifest'inin lisans katmanı metni; bulunamazsa None.

    Kök `blob.parent.parent`; yalnızca scan'in bulduğu `manifests/registry.ollama.ai/library/*/*` okunur.
    """
    blob = Path(blob)
    if not _BLOB_NAME.fullmatch(blob.name) or blob.parent.name != "blobs":
        return None
    root = blob.parent.parent
    want = "sha256:" + blob.name.removeprefix("sha256-")
    try:
        manifests = sorted(p for p in (root / "manifests" / "registry.ollama.ai" / "library").glob("*/*") if p.is_file())
    except OSError:
        return None
    for manifest in manifests:
        try:
            if manifest.stat().st_size > _MAX_MANIFEST_BYTES:
                continue
            data = json.loads(manifest.read_text(encoding="utf-8"))
            model = next(l for l in data["layers"] if l["mediaType"] == OLLAMA_MODEL_MEDIA)
            if not valid_digest(model["digest"]) or model["digest"] != want:
                continue
        except (OSError, ValueError, KeyError, TypeError, StopIteration, AttributeError):
            continue
        text = ollama_layer_text(root, data)
        if text is not None:
            return text
    return None
