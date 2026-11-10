"""Sınırlı GGUF üst veri yürüyücüsü: kötü niyetli/bozuk dosyalar hızlı ve istisnasız işlenir.

Zamanlama testleri alt süreçte çalışır: gerileme olursa takılan süreç öldürülür, test takımı donmaz.
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import gguf
import pytest

from neshter.gguf_header import read_gguf_header
from neshter.licenses import read_gguf_license
from neshter.scan import gguf_info, model_arch

REPO = Path(__file__).resolve().parent.parent


def hdr(n_tensors: int, n_kv: int, version: int = 3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, n_tensors, n_kv)


def s(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def kv_str(key: bytes, val: bytes) -> bytes:
    return s(key) + struct.pack("<I", 8) + s(val)


def crafted_cases() -> dict[str, bytes]:
    lic = kv_str(b"general.license", b"mit")
    cases = {
        "empty": b"",
        "magic_only": b"GGUF",
        "bad_magic": b"GGUX" + hdr(0, 0)[4:] + b"\0" * 64,
        "bad_version": hdr(0, 1, version=77) + lic,
        "v1_unsupported": hdr(0, 1, version=1) + lic,
        "truncated_header": hdr(1, 1)[:10],
        "truncated_after_count": hdr(0, 1),
        "huge_counts": hdr(2**63, 2**63),
        "huge_n_kv": hdr(0, 2**62) + b"\0" * 100,
        "n_kv_over_cap": hdr(0, 5000) + lic * 5000,
        "key_len_huge": hdr(0, 1) + struct.pack("<Q", 2**62) + b"abc",
        "string_len_huge": hdr(0, 1) + s(b"general.license") + struct.pack("<I", 8) + struct.pack("<Q", 2**62) + b"x",
        "string_len_past_eof": hdr(0, 1) + s(b"general.license") + struct.pack("<I", 8) + struct.pack("<Q", 1000) + b"mit",
        "unknown_value_type": hdr(0, 1) + s(b"k") + struct.pack("<I", 99) + b"\0" * 16,
        "array_u8_2_62": hdr(0, 1) + s(b"a") + struct.pack("<I", 9) + struct.pack("<IQ", 0, 2**62) + b"\0" * 100,
        "array_str_2_62": hdr(0, 1) + s(b"a") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2**62) + b"\0" * 100,
        "array_nested_deep": hdr(0, 1)
        + s(b"a")
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 9, 1) * 50000
        + b"\0" * 100,
        "array_u8_1M": hdr(0, 1) + s(b"a") + struct.pack("<I", 9) + struct.pack("<IQ", 0, 1_000_000) + b"\0" * 1_000_000,
        "array_u8_4M": hdr(0, 1) + s(b"a") + struct.pack("<I", 9) + struct.pack("<IQ", 0, 4_000_000) + b"\0" * 4_000_000,
        "array_str_zero_len_1M": hdr(0, 1) + s(b"a") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 1_000_000) + b"\0" * 8_000_000,
        "bad_utf8_key": hdr(0, 1) + s(b"\xff\xfe") + struct.pack("<I", 4) + b"\0" * 4,
        "bad_utf8_value": hdr(0, 1) + kv_str(b"general.license", b"\xff\xfe\xfd"),
    }
    return cases


# Alt süreç betiği: her dosya için süre, tracemalloc tepe belleği ve sonucu JSON olarak yazar.
_RUNNER = r"""
import json, sys, time, tracemalloc
from pathlib import Path
from neshter.gguf_header import read_gguf_header
from neshter.licenses import read_gguf_license
from neshter.scan import gguf_info
out = {}
for p in sys.argv[1:]:
    tracemalloc.start()
    t = time.perf_counter()
    try:
        r = [read_gguf_header(Path(p), keys=("general.license",)), read_gguf_license(Path(p)), gguf_info(Path(p))]
        err = None
    except BaseException as e:
        r, err = None, repr(e)
    dt = time.perf_counter() - t
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out[Path(p).stem] = {"err": err, "secs": dt, "peak_mb": peak / 1e6, "result": r}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def crafted_results(tmp_path_factory):
    d = tmp_path_factory.mktemp("crafted")
    paths = []
    for name, data in crafted_cases().items():
        p = d / f"{name}.gguf"
        p.write_bytes(data)
        paths.append(str(p))
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,  # 3 sınama x 20 dosya; gerileme (sonsuz döngü) burada öldürülür
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


@pytest.mark.parametrize("name", list(crafted_cases()))
def test_crafted_file_is_fast_bounded_and_never_raises(crafted_results, name):
    r = crafted_results[name]
    assert r["err"] is None
    assert r["secs"] < 2.0, f"{name}: {r['secs']:.2f}s"
    assert r["peak_mb"] < 64, f"{name}: peak {r['peak_mb']:.0f} MB"


@pytest.mark.parametrize(
    "name",
    [
        "empty", "magic_only", "bad_magic", "bad_version", "v1_unsupported", "truncated_header",
        "huge_counts", "huge_n_kv", "n_kv_over_cap",
    ],
)
def test_invalid_headers_yield_none(crafted_results, name):
    header, lic, info = crafted_results[name]["result"]
    assert header is None and lic is None and info == [None, None]


@pytest.mark.parametrize(
    "name",
    [
        "truncated_after_count", "key_len_huge", "string_len_huge", "string_len_past_eof", "unknown_value_type", "array_u8_2_62",
        "array_str_2_62", "array_nested_deep", "array_u8_1M", "array_u8_4M", "array_str_zero_len_1M",
        "bad_utf8_key",
    ],
)
def test_malicious_bodies_yield_no_license(crafted_results, name):
    header, lic, _info = crafted_results[name]["result"]
    assert header == {} and lic is None


def test_non_utf8_license_value_is_replaced_not_raised(crafted_results):
    header, lic, _ = crafted_results["bad_utf8_value"]["result"]
    assert header == {"general.license": "���"} and lic == "���"


def test_missing_and_directory_paths_return_none(tmp_path):
    assert read_gguf_header(tmp_path / "yok.gguf") is None
    assert read_gguf_header(tmp_path) is None
    assert read_gguf_license(tmp_path / "yok.gguf") is None


def test_oversize_string_value_is_dropped(tmp_path):
    p = tmp_path / "big.gguf"
    p.write_bytes(hdr(0, 1) + kv_str(b"general.license", b"A" * 100_000))
    assert read_gguf_header(p, keys=("general.license",)) == {}


def test_finds_license_after_big_arrays_and_stops_early(tmp_path):
    p = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(p), "llama")
    w.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(2000)])
    w.add_array("tokenizer.ggml.scores", [float(i) for i in range(2000)])
    w.add_string("general.license", "apache-2.0")
    w.add_block_count(32)
    w.add_head_count(32)
    w.add_head_count_kv(8)
    w.add_feed_forward_length(14336)
    w.add_string("tokenizer.after", "never read")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert read_gguf_license(p) == "apache-2.0"
    lic, arch = gguf_info(p)
    assert lic == "apache-2.0"
    assert arch == {"family": "llama", "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "d_mlp": 14336}
    assert model_arch(p, "gguf") == arch
    # Tüm anahtarlar bulununca yürüyüş durur: sonraki anahtar istenmediği için zaten dönmez; erken durma
    # `done` ile sınanır.
    seen = read_gguf_header(p, keys=("general.license", "tokenizer.after"), done=lambda f: "general.license" in f)
    assert seen == {"general.license": "apache-2.0"}


def test_license_name_fallback_and_no_license(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    for p, kv in ((a, {"general.license.name": "Qwen License"}), (b, {})):
        w = gguf.GGUFWriter(str(p), "qwen2")
        for k, v in kv.items():
            w.add_string(k, v)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    assert read_gguf_license(a) == "Qwen License"
    assert read_gguf_license(b) is None
    assert gguf_info(b) == (None, {"family": "qwen2", "n_layers": None, "n_heads": None, "n_kv_heads": None, "d_mlp": None})


def test_reader_is_never_used(monkeypatch, tmp_path):
    # GGUFReader (sınırsız ayrıştırıcı) yüklenmemeli: erişilirse test patlar.
    def boom(*a, **k):
        raise AssertionError("GGUFReader kullanıldı")

    monkeypatch.setattr(gguf, "GGUFReader", boom)
    p = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(p), "llama")
    w.add_string("general.license", "mit")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert read_gguf_license(p) == "mit" and gguf_info(p)[1]["family"] == "llama"
