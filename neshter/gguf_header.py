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
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Callable, Collection
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
