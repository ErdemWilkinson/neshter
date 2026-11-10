import json

import gguf
import pytest

from neshter.hardware import SystemInfo, system_info
from neshter.scan import default_roots, scan_all, scan_hf_cache, scan_lmstudio, scan_ollama

DIGEST = "ab" * 32


def _make_ollama(root, name="qwen2.5", tag="7b", digest=DIGEST, blob_bytes=12):
    mdir = root / "manifests/registry.ollama.ai/library" / name
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layers": [
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:zzz", "size": 3},
            {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}", "size": blob_bytes},
        ]
    }
    (mdir / tag).write_text(json.dumps(manifest))
    blobs = root / "blobs"
    blobs.mkdir(exist_ok=True)
    (blobs / f"sha256-{digest}").write_bytes(b"x" * blob_bytes)


def test_scan_ollama(tmp_path):
    _make_ollama(tmp_path)
    [e] = scan_ollama(tmp_path)
    assert (e.name, e.fmt, e.size_bytes) == ("qwen2.5:7b", "gguf", 12)
    assert e.source == "ollama"
    assert e.path == tmp_path / "blobs" / ("sha256-" + DIGEST)


def test_scan_ollama_skips_missing_blob(tmp_path):
    _make_ollama(tmp_path)
    (tmp_path / "blobs" / ("sha256-" + DIGEST)).unlink()
    assert scan_ollama(tmp_path) == []


def test_scan_skips_broken_manifest_and_missing_roots(tmp_path):
    (tmp_path / "manifests/registry.ollama.ai/library/x/y").parent.mkdir(parents=True)
    (tmp_path / "manifests/registry.ollama.ai/library/x/y").write_text("{bozuk")
    assert scan_ollama(tmp_path) == []
    assert scan_all({"ollama": [tmp_path / "yok"], "lmstudio": [], "huggingface": []}) == []


def test_scan_lmstudio(tmp_path):
    d = tmp_path / "lmstudio-community" / "Foo-GGUF"
    d.mkdir(parents=True)
    (d / "foo-q4.gguf").write_bytes(b"y" * 7)
    (d / "readme.md").write_text("not a model")
    [e] = scan_lmstudio(tmp_path)
    assert (e.name, e.source, e.fmt, e.size_bytes) == ("foo-q4", "lmstudio", "gguf", 7)
    assert e.path == d / "foo-q4.gguf"
    assert scan_lmstudio(tmp_path / "yok") == []


def test_scan_hf_cache(tmp_path):
    snap = tmp_path / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"z" * 20)
    # config yok -> atlanır
    bad = tmp_path / "models--a--b" / "snapshots" / "r1"
    bad.mkdir(parents=True)
    (bad / "model.safetensors").write_bytes(b"z")
    # safetensors yok -> atlanır
    bad2 = tmp_path / "models--c--d" / "snapshots" / "r1"
    bad2.mkdir(parents=True)
    (bad2 / "config.json").write_text("{}")
    [e] = scan_hf_cache(tmp_path)
    assert (e.name, e.source, e.fmt, e.size_bytes) == ("Qwen/Qwen2.5-0.5B", "huggingface", "safetensors", 20)
    assert e.path == snap


def test_scan_all_combines_sources(tmp_path):
    ol, lm, hf = tmp_path / "ol", tmp_path / "lm", tmp_path / "hf"
    _make_ollama(ol)
    lm.mkdir()
    (lm / "m.gguf").write_bytes(b"1")
    snap = hf / "models--o--n" / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "w.safetensors").write_bytes(b"1")
    got = scan_all({"ollama": [ol], "lmstudio": [lm], "huggingface": [hf]})
    assert sorted(e.source for e in got) == ["huggingface", "lmstudio", "ollama"]


def test_default_roots_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "o"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "h"))
    r = default_roots()
    assert r["ollama"] == [tmp_path / "o"]
    assert r["huggingface"] == [tmp_path / "h" / "hub"]
    assert len(r["lmstudio"]) == 2
    monkeypatch.delenv("OLLAMA_MODELS")
    monkeypatch.delenv("HF_HOME")
    r = default_roots()
    assert r["ollama"][0].parts[-2:] == (".ollama", "models")
    assert r["huggingface"][0].parts[-3:] == (".cache", "huggingface", "hub")


def test_system_info_cpu_only():
    s = system_info()
    assert isinstance(s, SystemInfo)
    assert s.ram_bytes > 0
    assert s.gpu_name is None and s.vram_bytes == 0  # CUDA_VISIBLE_DEVICES boş


def test_hf_hub_cache_env_takes_precedence(monkeypatch, tmp_path):
    from neshter.scan import default_roots

    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hubcache"))
    assert default_roots()["huggingface"] == [tmp_path / "hubcache"]
    monkeypatch.delenv("HF_HUB_CACHE")
    assert default_roots()["huggingface"] == [tmp_path / "home" / "hub"]


def test_scan_entries_carry_license(tmp_path):
    snap = tmp_path / "hf" / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"z" * 20)
    (snap / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    assert [e.license for e in scan_hf_cache(tmp_path / "hf")] == ["apache-2.0"]
    d = tmp_path / "lm" / "pub" / "Foo-GGUF"
    d.mkdir(parents=True)
    w = gguf.GGUFWriter(str(d / "foo.gguf"), "qwen2")
    w.add_string("general.license", "mit")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert [e.license for e in scan_lmstudio(tmp_path / "lm")] == ["mit"]


def test_scan_hf_license_file_fallback(tmp_path):
    snap = tmp_path / "models--a--b" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "w.safetensors").write_bytes(b"z")
    (snap / "LICENSE").write_text("Gemma Terms of Use", encoding="utf-8")
    assert [e.license for e in scan_hf_cache(tmp_path)] == ["gemma"]


def test_scan_ollama_license_from_gguf_then_layer(tmp_path):
    lic_digest = "cd" * 32
    _make_ollama(tmp_path)
    mf = tmp_path / "manifests/registry.ollama.ai/library/qwen2.5/7b"
    data = json.loads(mf.read_text())
    data["layers"].append(
        {"mediaType": "application/vnd.ollama.image.license", "digest": f"sha256:{lic_digest}", "size": 5}
    )
    mf.write_text(json.dumps(data))
    (tmp_path / "blobs" / f"sha256-{lic_digest}").write_text("Qwen RESEARCH LICENSE AGREEMENT", encoding="utf-8")
    assert [e.license for e in scan_ollama(tmp_path)] == ["qwen-research"]  # blob GGUF değil -> katman metni
    blob = tmp_path / "blobs" / f"sha256-{DIGEST}"
    w = gguf.GGUFWriter(str(blob), "qwen2")
    w.add_string("general.license", "apache-2.0")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert [e.license for e in scan_ollama(tmp_path)] == ["apache-2.0"]  # üst veri kazanır


def test_scan_ollama_rejects_invalid_digest(tmp_path):
    # AS L7: model katmanı digest'i ^sha256:[0-9a-f]{64}$ olmalı; yoksa girdi atlanır (yol kurulmaz).
    _make_ollama(tmp_path, name="m", tag="ok")
    (tmp_path / "blobs" / "x").write_bytes(b"gizli")  # '../x' hilesi buna ulaşmaya çalışırdı
    mdir = tmp_path / "manifests/registry.ollama.ai/library/m"
    for i, bad in enumerate(("sha256:../x", "sha256:" + "AB" * 32, "sha256:abc", f"sha256:{DIGEST}\n")):
        layer = {"mediaType": "application/vnd.ollama.image.model", "digest": bad, "size": 1}
        (mdir / f"bad{i}").write_text(json.dumps({"layers": [layer]}))
    assert [e.name for e in scan_ollama(tmp_path)] == ["m:ok"]


# --- Task 9 fix round 1 ---------------------------------------------------------------------------


def _gguf(path, arch="llama", **kv):
    w = gguf.GGUFWriter(str(path), arch)
    for k, v in kv.items():
        w.add_string(k.replace("__", "."), v)
    w.add_block_count(32)
    w.add_head_count(32)
    w.add_head_count_kv(8)
    w.add_feed_forward_length(14336)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_scan_reads_each_gguf_header_once_for_license_and_arch(tmp_path, monkeypatch):
    from neshter import scan

    d = tmp_path / "pub" / "Foo-GGUF"
    d.mkdir(parents=True)
    _gguf(d / "foo.gguf", general__license="mit")
    calls = []
    real = scan.read_gguf_header
    monkeypatch.setattr(scan, "read_gguf_header", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    [e] = scan_lmstudio(tmp_path)
    assert e.license == "mit"
    assert e.arch == {"family": "llama", "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "d_mlp": 14336}
    assert len(calls) == 1


def test_scan_ollama_arch_and_license_from_blob(tmp_path):
    _make_ollama(tmp_path)
    _gguf(tmp_path / "blobs" / f"sha256-{DIGEST}", general__license="apache-2.0")
    [e] = scan_ollama(tmp_path)
    assert e.license == "apache-2.0" and e.arch["family"] == "llama" and e.arch["n_layers"] == 32


def test_scan_survives_malicious_gguf(tmp_path):
    # 149 baytlık, uint8 dizisi sayısı 2^62 olan dosya: tarama sonsuza dek takılmaz ve girdi yine listelenir.
    import struct

    d = tmp_path / "pub"
    d.mkdir()
    bomb = (
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1) + b"a" + struct.pack("<I", 9)
        + struct.pack("<IQ", 0, 2**62) + b"\0" * 100
    )
    (d / "bomb.gguf").write_bytes(bomb)
    [e] = scan_lmstudio(tmp_path)
    assert e.name == "bomb" and e.license is None and e.arch is None


def test_scan_ollama_survives_deeply_nested_manifest(tmp_path):
    _make_ollama(tmp_path, name="x", tag="ok")
    lib = tmp_path / "manifests/registry.ollama.ai/library/x"
    (lib / "deep").write_text("[" * 200_000 + "]" * 200_000)  # ~400 KB, RecursionError üretir
    (lib / "deeper").write_text('{"layers":' + "[" * 200_000 + "]" * 200_000 + "}")
    assert [e.name for e in scan_ollama(tmp_path)] == ["x:ok"]


def test_scan_ollama_skips_oversize_manifest(tmp_path):
    _make_ollama(tmp_path, name="x", tag="ok")
    lib = tmp_path / "manifests/registry.ollama.ai/library/x"
    (lib / "big").write_text(" " * (2 * 1024 * 1024) + "{}")
    assert [e.name for e in scan_ollama(tmp_path)] == ["x:ok"]


def test_scan_hf_entry_carries_arch_and_license_name(tmp_path):
    snap = tmp_path / "models--Qwen--Qwen2.5-3B" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 2}))
    (snap / "model.safetensors").write_bytes(b"z")
    (snap / "README.md").write_text("---\nlicense: other\nlicense_name: qwen-research\n---\n", encoding="utf-8")
    [e] = scan_hf_cache(tmp_path)
    assert e.license == "qwen-research" and e.arch["family"] == "qwen2" and e.arch["n_layers"] == 2


# --- Final fix wave I2: tek bir düşmanca config/başlık değeri model listesini düşüremez -------------------


def _hf_snap(root, repo, config_text):
    snap = root / f"models--{repo}" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(config_text, encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"z")
    return snap


def test_hf_config_with_infinity_nan_or_huge_is_listed_without_arch(tmp_path):
    from neshter.scan import model_arch

    _hf_snap(tmp_path, "a--inf", '{"model_type":"llama","num_hidden_layers":Infinity}')
    _hf_snap(tmp_path, "b--nan", '{"model_type":"llama","num_attention_heads":NaN}')
    _hf_snap(tmp_path, "c--neginf", '{"model_type":"llama","intermediate_size":-Infinity}')
    _hf_snap(tmp_path, "d--big", '{"model_type":"llama","pad":"' + "x" * (2 * 1024 * 1024) + '"}')
    _hf_snap(tmp_path, "e--ok", json.dumps({"model_type": "llama", "num_hidden_layers": 2}))
    entries = {e.name: e for e in scan_hf_cache(tmp_path)}
    assert set(entries) == {"a/inf", "b/nan", "c/neginf", "d/big", "e/ok"}
    for name in ("a/inf", "b/nan", "c/neginf", "d/big"):
        assert entries[name].arch is None, name
    assert entries["e/ok"].arch["n_layers"] == 2
    # sınırlı okuma (1 MiB): 2 MiB'lık config hiç tam okunmaz, kesik JSON -> yapı bilinmiyor
    assert model_arch(tmp_path / "models--d--big" / "snapshots" / "r1", "safetensors") is None


@pytest.mark.parametrize("value", [1e300, -1, 2**40, True, "32", 3.5])
def test_clean_arch_rejects_out_of_range_numbers(tmp_path, value):
    _hf_snap(tmp_path, "x--y", json.dumps({"model_type": "llama", "num_hidden_layers": value, "num_attention_heads": 4}))
    [e] = scan_hf_cache(tmp_path)
    assert e.arch["n_layers"] is None and e.arch["n_heads"] == 4


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_gguf_float_inf_nan_header_value_is_listed(tmp_path, bad):
    d = tmp_path / "pub"
    d.mkdir()
    w = gguf.GGUFWriter(str(d / "kotu.gguf"), "llama")
    w.add_float32("llama.block_count", bad)
    w.add_float32("llama.attention.head_count", bad)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    _gguf(d / "iyi.gguf")
    entries = {e.name: e for e in scan_lmstudio(tmp_path)}
    assert entries["kotu"].arch["n_layers"] is None and entries["kotu"].arch["n_heads"] is None
    assert entries["iyi"].arch["n_layers"] == 32


def test_api_models_survives_hostile_entries(tmp_path):
    from fastapi.testclient import TestClient

    from neshter.server.app import create_app

    hf = tmp_path / "hf"
    _hf_snap(hf, "a--inf", '{"model_type":"llama","num_hidden_layers":Infinity}')
    _hf_snap(hf, "b--ok", json.dumps({"model_type": "llama", "num_hidden_layers": 2}))
    lms = tmp_path / "lms"
    lms.mkdir()
    w = gguf.GGUFWriter(str(lms / "nan.gguf"), "llama")
    w.add_float32("llama.block_count", float("nan"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    c = TestClient(create_app("t", 47998, roots={"huggingface": [hf], "lmstudio": [lms]}), base_url="http://127.0.0.1:47998")
    r = c.get("/api/models", headers={"X-Neshter-Token": "t"})
    assert r.status_code == 200
    assert {e["name"] for e in r.json()} == {"a/inf", "b/ok", "nan"}
