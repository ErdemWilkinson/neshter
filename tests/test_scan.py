import json

import gguf

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
