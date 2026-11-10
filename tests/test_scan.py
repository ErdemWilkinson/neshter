import json

from neshter.hardware import SystemInfo, system_info
from neshter.scan import default_roots, scan_all, scan_hf_cache, scan_lmstudio, scan_ollama


def _make_ollama(root, name="qwen2.5", tag="7b", digest="abc", blob_bytes=12):
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
    assert e.path == tmp_path / "blobs" / "sha256-abc"


def test_scan_ollama_skips_missing_blob(tmp_path):
    _make_ollama(tmp_path)
    (tmp_path / "blobs" / "sha256-abc").unlink()
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
