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


# --- I1: validate_gguf — tüm üst veri + tensör bilgi tablosu, transformers'ın ayrıştırıcısından ÖNCE ---------


def ti(name: bytes, dims, gtype: int = 0, offset: int = 0) -> bytes:
    """Tensör bilgisi: ad, n_dims, boyutlar, ggml türü, veri bölgesine göre konum."""
    return s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack(
        "<IQ", gtype, offset
    )


def kv_u32(key: bytes, val: int) -> bytes:
    return s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def tensor_cases() -> dict[str, bytes]:
    pad = b"\0" * 256
    return {
        "t_count_huge": hdr(2**40, 0) + pad,
        "t_count_over_cap": hdr(70_000, 0) + ti(b"t", [1]) * 70_000 + pad,
        "t_count_over_remaining": hdr(1000, 0) + ti(b"t", [1]) * 3 + pad,
        "t_info_truncated": hdr(2, 0) + ti(b"a", [4]),
        "t_ndims_9": hdr(1, 0) + ti(b"t", [1] * 9) + pad,
        "t_ndims_0": hdr(1, 0) + ti(b"t", []) + pad,
        "t_dim_zero": hdr(1, 0) + ti(b"t", [0]) + pad,
        "t_dims_overflow": hdr(1, 0) + ti(b"t", [2**32, 2**32]) + pad,
        "t_unknown_type": hdr(1, 0) + ti(b"t", [4], gtype=99) + pad,
        "t_name_too_long": hdr(1, 0) + ti(b"n" * 300, [4]) + pad,
        "t_name_len_huge": hdr(1, 0) + struct.pack("<Q", 2**62) + pad,
        "t_offset_past_eof": hdr(1, 0) + ti(b"t", [4], offset=2**40) + pad,
        "t_data_past_eof": hdr(1, 0) + ti(b"t", [1000]) + pad,  # f32 x 1000 = 4000 bayt; dosyada ~250
        "t_quant_not_block_multiple": hdr(1, 0) + ti(b"t", [33], gtype=2) + pad,  # Q4_0 blok 32
        "t_bad_alignment": hdr(1, 1) + kv_u32(b"general.alignment", 3) + ti(b"t", [1]) + pad,
        "t_zero_alignment": hdr(1, 1) + kv_u32(b"general.alignment", 0) + ti(b"t", [1]) + pad,
    }


_VALIDATE_RUNNER = r"""
import json, sys, time
from pathlib import Path
from neshter.gguf_header import GGUFInvalid, validate_gguf
out = {}
for p in sys.argv[1:]:
    t = time.perf_counter()
    try:
        info = validate_gguf(Path(p))
        r = {"ok": True, "n_params": info.n_params, "n_tensors": info.n_tensors, "arch": info.architecture}
    except GGUFInvalid as e:
        r = {"ok": False, "msg": str(e)}
    except BaseException as e:
        r = {"ok": False, "unexpected": repr(e)}
    r["secs"] = time.perf_counter() - t
    out[Path(p).stem] = r
print(json.dumps(out))
"""

_STRUCTURALLY_VALID = {"bad_utf8_value", "bad_utf8_key", "array_u8_1M", "array_u8_4M", "array_str_zero_len_1M"}


@pytest.fixture(scope="module")
def validate_results(tmp_path_factory):
    d = tmp_path_factory.mktemp("validate")
    paths = []
    for name, data in {**crafted_cases(), **tensor_cases()}.items():
        p = d / f"{name}.gguf"
        p.write_bytes(data)
        paths.append(str(p))
    proc = subprocess.run(
        [sys.executable, "-c", _VALIDATE_RUNNER, *paths], cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


@pytest.mark.parametrize("name", [n for n in crafted_cases() if n not in _STRUCTURALLY_VALID] + list(tensor_cases()))
def test_validate_rejects_crafted_files_fast(validate_results, name):
    r = validate_results[name]
    assert r["ok"] is False and "unexpected" not in r, r
    assert "GGUF dosyası bozuk ya da desteklenmeyen biçimde" in r["msg"]
    assert r["secs"] < 2.0, f"{name}: {r['secs']:.2f}s"


@pytest.mark.parametrize("name", sorted(_STRUCTURALLY_VALID))
def test_validate_accepts_structurally_valid_headers(validate_results, name):
    r = validate_results[name]
    assert r["ok"] is True and r["n_tensors"] == 0 and r["n_params"] == 0, r
    assert r["secs"] < 2.0


def _tiny_gguf(path, arch="llama", alignment=None):
    import numpy as np

    w = gguf.GGUFWriter(str(path), arch)
    if alignment is not None:
        w.data_alignment = alignment
        w.add_custom_alignment(alignment)
    w.add_block_count(1)
    w.add_tensor("token_embd.weight", np.zeros((16, 8), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.ones((8, 8), dtype=np.float16))
    w.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_validate_accepts_writer_output_and_counts_params(tmp_path):
    from neshter.gguf_header import validate_gguf

    info = validate_gguf(_tiny_gguf(tmp_path / "m.gguf"))
    assert info.architecture == "llama"
    assert info.n_tensors == 3 and info.n_params == 16 * 8 + 8 * 8 + 8
    info64 = validate_gguf(_tiny_gguf(tmp_path / "a.gguf", alignment=64))
    assert info64.n_params == info.n_params and info64.data_start % 64 == 0


def test_validate_rejects_writer_output_truncated_in_tensor_data(tmp_path):
    from neshter.gguf_header import GGUFInvalid, validate_gguf

    p = _tiny_gguf(tmp_path / "m.gguf")
    data = p.read_bytes()
    p.write_bytes(data[:-40])  # son tensörün verisi eksik
    with pytest.raises(GGUFInvalid):
        validate_gguf(p)


def test_validate_missing_file_is_invalid(tmp_path):
    from neshter.gguf_header import GGUFInvalid, validate_gguf

    with pytest.raises(GGUFInvalid):
        validate_gguf(tmp_path / "yok.gguf")


# --- Follow-up R1: örtüşen / takma adlı / hizasız tensör verisi reddedilir -------------------------------------


def _with_data(tensors: list[bytes], data: bytes) -> bytes:
    raw = hdr(len(tensors), 0) + b"".join(tensors)
    return raw + b"\0" * ((-len(raw)) % 32) + data


def alias_cases() -> dict[str, bytes]:
    data = b"\0" * 4096
    return {
        # İnceleyicinin denemesi: 60000 x 1 MiB tensör, hepsi konum 0'da, 1 MiB veri (3.5 MB dosya, "15.7G parametre")
        "a_probe_60000_at_zero": _with_data([ti(b"blk.%d.w" % i, [262144]) for i in range(60_000)], b"\0" * (1 << 20)),
        "a_overlap_same_offset": _with_data([ti(b"a", [16]), ti(b"b", [16])], data),
        "a_overlap_partial": _with_data([ti(b"a", [16]), ti(b"b", [16], offset=32)], data),  # 64 bayt, 32'de başlar
        "a_unaligned_offset": _with_data([ti(b"a", [16], offset=3)], data),
        "a_sum_exceeds_data": _with_data([ti(b"a", [512]), ti(b"b", [512], offset=2048)], b"\0" * 3072),
    }


_ALIAS_RUNNER = _VALIDATE_RUNNER


@pytest.fixture(scope="module")
def alias_results(tmp_path_factory):
    d = tmp_path_factory.mktemp("alias")
    paths = []
    for name, data in alias_cases().items():
        p = d / f"{name}.gguf"
        p.write_bytes(data)
        paths.append(str(p))
    proc = subprocess.run(
        [sys.executable, "-c", _ALIAS_RUNNER, *paths], cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


@pytest.mark.parametrize("name", list(alias_cases()))
def test_validate_rejects_overlapping_or_unaligned_tensor_data(alias_results, name):
    r = alias_results[name]
    assert r["ok"] is False and "unexpected" not in r, r
    assert "GGUF dosyası bozuk ya da desteklenmeyen biçimde" in r["msg"]
    assert r["secs"] < 2.0, f"{name}: {r['secs']:.2f}s"


def test_validate_accepts_adjacent_aligned_tensors_in_any_table_order(tmp_path):
    from neshter.gguf_header import validate_gguf

    # tablo sırası konum sırasından farklı; bitişik, hizalı, örtüşmesiz
    p = tmp_path / "ok.gguf"
    p.write_bytes(_with_data([ti(b"b", [16], offset=64), ti(b"a", [16]), ti(b"c", [8], gtype=1, offset=128)], b"\0" * 160))
    info = validate_gguf(p)
    assert info.n_tensors == 3 and info.n_params == 40
