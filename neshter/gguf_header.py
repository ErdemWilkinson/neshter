"""Sınırlı GGUF üst veri yürüyücüsü (yaprak modül).

`gguf.GGUFReader` güvenilmeyen dosyada kullanılmaz: tüm dosyayı ayrıştırır ve 149 baytlık bir dosya
(uint8 dizisi, sayı 2^62) sonsuz döngüye + sınırsız bellek büyümesine yol açabilir. Bu yürüyücü:

- yalnızca ilk 16 MiB'ı eşler; sihirli baytları ve sürümü (2 ya da 3) doğrular;
- KV sayısını (<= 4096), anahtar uzunluğunu ve her dize uzunluğunu / dizi sayısını, okumadan ya da
  ayırmadan ÖNCE kalan baytlarla karşılaştırır;
- sabit genişlikli dizileri hesaplanan boyutla atlar; iç içe diziler derinlik sınırlıdır; toplam adım
  sınırı vardır;
- istenen anahtarların hepsi bulunur bulunmaz durur ve yalnızca istenen anahtarları döndürür.

Bozuk/kesik üst veri istisna fırlatmaz: o ana dek bulunanlar (sihirli bayt/sürüm geçersizse None) döner.

`validate_gguf` ise yüklemeden önce TÜM başlığı ve tensör bilgi tablosunu aynı sınırlarla yürür ve geçersiz
dosyada GGUFInvalid fırlatır (transformers'ın `gguf_file=` yüklemesi dosyayı sınırsız ayrıştırır).
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_KV = 4096
MAX_KEY_BYTES = 256
MAX_STRING_VALUE_BYTES = 64 * 1024
MAX_STEPS = 1_000_000  # dize/iç dizi öğesi başına bir adım
MAX_DEPTH = 3
MAX_KEPT_SUFFIX_HITS = 64

_SCALARS = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_STRING, _ARRAY = 8, 9


class _Stop(Exception):
    """Üst veri bozuk/sınır aşıldı: yürüyüş burada biter."""


class _Walker:
    def __init__(self, mm: mmap.mmap):
        self.mm = mm
        self.n = len(mm)
        self.pos = 0
        self.steps = 0

    def need(self, k: int) -> None:
        if k < 0 or k > self.n - self.pos:
            raise _Stop

    def unpack(self, fmt: str, size: int):
        self.need(size)
        (v,) = struct.unpack_from(fmt, self.mm, self.pos)
        self.pos += size
        return v

    def string(self, keep: bool, limit: int) -> str | None:
        length = self.unpack("<Q", 8)
        self.need(length)  # ayırmadan önce kalan baytlarla karşılaştır
        start = self.pos
        self.pos += length
        if not keep or length > limit:
            return None
        return bytes(self.mm[start : start + length]).decode("utf-8", errors="replace")

    def array(self, depth: int) -> None:
        if depth >= MAX_DEPTH:
            raise _Stop
        elem = self.unpack("<I", 4)
        count = self.unpack("<Q", 8)
        remaining = self.n - self.pos
        if elem in _SCALARS:
            size = _SCALARS[elem][1]
            if count > remaining // size:
                raise _Stop
            self.pos += count * size
        elif elem == _STRING:
            if count > remaining // 8:  # her dize en az 8 bayt (uzunluk alanı)
                raise _Stop
            self.steps += count
            if self.steps > MAX_STEPS:
                raise _Stop
            mm, pos, n = self.mm, self.pos, self.n
            for _ in range(count):
                if n - pos < 8:
                    raise _Stop
                (length,) = struct.unpack_from("<Q", mm, pos)
                pos += 8
                if length > n - pos:
                    raise _Stop
                pos += length
            self.pos = pos
        elif elem == _ARRAY:
            if count > remaining // 12:  # iç dizi başlığı: 4 + 8 bayt
                raise _Stop
            self.steps += count
            if self.steps > MAX_STEPS:
                raise _Stop
            for _ in range(count):
                self.array(depth + 1)
        else:
            raise _Stop

    def value(self, vtype: int, keep: bool):
        if vtype in _SCALARS:
            fmt, size = _SCALARS[vtype]
            return self.unpack(fmt, size)
        if vtype == _STRING:
            return self.string(keep, MAX_STRING_VALUE_BYTES)
        if vtype == _ARRAY:
            self.array(0)
            return None
        raise _Stop


def _walk(
    mm: mmap.mmap, keys: Collection[str], suffixes: tuple[str, ...], done: Callable[[dict], bool] | None
) -> dict | None:
    if len(mm) < 24 or mm[:4] != b"GGUF":
        return None
    version, _n_tensors, n_kv = struct.unpack_from("<IQQ", mm, 4)
    if version not in (2, 3) or n_kv > MAX_KV:
        return None
    w = _Walker(mm)
    w.pos = 24
    found: dict = {}
    suffix_hits = 0
    wanted = set(keys)
    if not wanted and not suffixes:
        return found
    try:
        for _ in range(n_kv):
            klen = w.unpack("<Q", 8)
            if klen > MAX_KEY_BYTES:
                raise _Stop
            w.need(klen)
            key = bytes(mm[w.pos : w.pos + klen]).decode("utf-8", errors="replace")
            w.pos += klen
            vtype = w.unpack("<I", 4)
            by_suffix = bool(suffixes) and key.endswith(suffixes) and key not in wanted
            keep = key in wanted or (by_suffix and suffix_hits < MAX_KEPT_SUFFIX_HITS)
            value = w.value(vtype, keep)
            if keep and value is not None and key not in found:
                found[key] = value
                suffix_hits += by_suffix
            if done is not None:
                if done(found):
                    break
            elif not suffixes and wanted <= found.keys():
                break
    except (_Stop, struct.error, ValueError, OverflowError):
        pass  # bulunanlar geçerli; kalanı okunamadı
    return found


def read_gguf_header(
    path: Path,
    keys: Collection[str] = (),
    suffixes: tuple[str, ...] = (),
    done: Callable[[dict], bool] | None = None,
) -> dict | None:
    """İstenen üst veri anahtarlarını (tam ad ya da `suffixes` ile biten) döndürür; geçersiz dosyada None.

    Değerler: dize, sayı ya da bool. Diziler ve 64 KiB'tan uzun dizeler atlanır (değer olarak verilmez).
    `done(found)` doğruysa ya da (done yoksa, soneksiz) tüm `keys` bulunduysa yürüyüş durur.
    """
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < 24:
                return None
            with mmap.mmap(f.fileno(), min(size, MAX_HEADER_BYTES), access=mmap.ACCESS_READ) as mm:
                return _walk(mm, keys, tuple(suffixes), done)
    except (OSError, ValueError, OverflowError):
        return None


# --- Tam doğrulama (yüklemeden önce) -----------------------------------------------------------------
#
# transformers `gguf_file=` yüklemesinde `gguf.GGUFReader` ile TÜM dosyayı ayrıştırır; kötü niyetli bir başlık
# (dev dizi sayısı, dev tensör sayısı, dosya sonunu aşan konum) onu sonsuz döngüye ya da sınırsız belleğe
# sokar. validate_gguf aynı yapıyı sınırlı biçimde baştan sona yürür; geçemeyen dosya ayrıştırıcıya hiç
# verilmez.

MAX_VALIDATE_HEADER_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 65_536
MAX_TENSOR_NAME_BYTES = 256
MAX_TENSOR_DIMS = 4
MAX_TENSOR_ELEMS = 2**40
MAX_ALIGNMENT = 1 << 20
_DEFAULT_ALIGNMENT = 32
_MIN_TENSOR_INFO_BYTES = 24
INVALID_GGUF_MESSAGE = "GGUF dosyası bozuk ya da desteklenmeyen biçimde"


class GGUFInvalid(ValueError):
    """GGUF başlığı ya da tensör tablosu bozuk/sınır dışı; dosya yüklenmemeli."""


@dataclass(frozen=True)
class GGUFInfo:
    architecture: str | None
    n_params: int
    n_tensors: int
    alignment: int
    data_start: int


def _quant_sizes() -> dict[int, tuple[int, int]]:
    from gguf.constants import GGML_QUANT_SIZES  # yalnızca sabitler; GGUFReader kullanılmaz

    return {int(k): v for k, v in GGML_QUANT_SIZES.items()}


def _invalid(reason: str) -> GGUFInvalid:
    return GGUFInvalid(f"{INVALID_GGUF_MESSAGE} ({reason}).")


def _validate(mm: mmap.mmap, file_size: int) -> GGUFInfo:
    if len(mm) < 24 or mm[:4] != b"GGUF":
        raise _invalid("sihirli baytlar yok")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", mm, 4)
    if version not in (2, 3):
        raise _invalid(f"sürüm {version}")
    if n_kv > MAX_KV:
        raise _invalid("üst veri anahtarı sayısı sınırı aşıyor")
    if n_tensors > MAX_TENSORS:
        raise _invalid("tensör sayısı sınırı aşıyor")
    w = _Walker(mm)
    w.pos = 24
    architecture: str | None = None
    alignment = _DEFAULT_ALIGNMENT
    try:
        for _ in range(n_kv):
            klen = w.unpack("<Q", 8)
            if klen > MAX_KEY_BYTES:
                raise _Stop
            w.need(klen)
            key = bytes(mm[w.pos : w.pos + klen]).decode("utf-8", errors="replace")
            w.pos += klen
            vtype = w.unpack("<I", 4)
            value = w.value(vtype, keep=key == "general.architecture")
            if key == "general.architecture" and isinstance(value, str):
                architecture = value
            elif key == "general.alignment":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise _invalid("general.alignment tam sayı değil")
                alignment = value
        if alignment < 1 or alignment > MAX_ALIGNMENT or alignment & (alignment - 1):
            raise _invalid(f"hizalama {alignment}")

        if n_tensors > (w.n - w.pos) // _MIN_TENSOR_INFO_BYTES:
            raise _invalid("tensör tablosu dosyadan büyük")
        sizes = _quant_sizes()
        tensors: list[tuple[int, int]] = []  # (konum, bayt)
        n_params = 0
        for _ in range(n_tensors):
            nlen = w.unpack("<Q", 8)
            if nlen > MAX_TENSOR_NAME_BYTES:
                raise _invalid("tensör adı çok uzun")
            w.need(nlen)
            w.pos += nlen
            n_dims = w.unpack("<I", 4)
            if not 1 <= n_dims <= MAX_TENSOR_DIMS:
                raise _invalid(f"tensör boyut sayısı {n_dims}")
            dims = [w.unpack("<Q", 8) for _ in range(n_dims)]
            elems = 1
            for d in dims:
                if d < 1:
                    raise _invalid("tensör boyutu sıfır")
                elems *= d
                if elems > MAX_TENSOR_ELEMS:
                    raise _invalid("tensör çok büyük")
            gtype = w.unpack("<I", 4)
            if gtype not in sizes:
                raise _invalid(f"bilinmeyen ggml türü {gtype}")
            block, type_size = sizes[gtype]
            if dims[0] % block:
                raise _invalid("tensör boyutu blok boyutunun katı değil")
            offset = w.unpack("<Q", 8)
            tensors.append((offset, elems // block * type_size))
            n_params += elems
    except _Stop as e:
        raise _invalid("başlık kesik ya da sınır dışı") from e
    except (struct.error, OverflowError) as e:
        raise _invalid("başlık okunamadı") from e

    data_start = (w.pos + alignment - 1) // alignment * alignment
    data_size = file_size - data_start
    if tensors and data_size < 0:
        raise _invalid("veri bölgesi dosya sonunu aşıyor")
    for offset, nbytes in tensors:
        if offset > data_size or nbytes > data_size - offset:
            raise _invalid("tensör verisi dosya sonunu aşıyor")
    return GGUFInfo(architecture, n_params, n_tensors, alignment, data_start)


def validate_gguf(path: Path) -> GGUFInfo:
    """GGUF'un tüm üst verisini ve tensör bilgi tablosunu sınırlı yürür; geçersizse GGUFInvalid.

    Sınırlar: başlık bölgesi ilk 64 MiB; KV <= 4096 (her biri `_Walker` sınırlarıyla); tensör sayısı
    <= 65536 ve kalan bayt / 24; tensör adı <= 256 bayt; 1..4 boyut, her boyut >= 1, eleman sayısı <= 2^40;
    bilinen ggml türü; `konum + bayt <= dosya boyu - veri başlangıcı` (`general.alignment`, varsayılan 32).
    Dönüş: mimari adı, parametre sayısı (Σ eleman), tensör sayısı, hizalama, veri başlangıcı.
    """
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < 24:
                raise _invalid("dosya çok kısa")
            with mmap.mmap(f.fileno(), min(size, MAX_VALIDATE_HEADER_BYTES), access=mmap.ACCESS_READ) as mm:
                return _validate(mm, size)
    except GGUFInvalid:
        raise
    except (OSError, ValueError, OverflowError) as e:
        raise _invalid("dosya okunamadı") from e
