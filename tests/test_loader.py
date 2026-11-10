import json
import shutil

import pytest
import torch

from neshter import loader
from neshter.loader import (
    UnsafeModelFile,
    UnsupportedArchitecture,
    load_model,
    sha256_path,
)


def test_load_dir(tiny_model_dir):
    lm = load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    assert lm.arch.n_heads == 4 and len(lm.source_sha256) == 64
    assert lm.tokenizer.padding_side == "left"
    assert lm.name == tiny_model_dir.name
    assert lm.source_path == tiny_model_dir
    assert not lm.model.training
    assert all(not p.requires_grad for p in lm.model.parameters())


def test_sha_unchanged_after_load(tiny_model_dir):
    before = sha256_path(tiny_model_dir)
    load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    assert sha256_path(tiny_model_dir) == before


def test_sha256_file_and_dir(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert sha256_path(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    d = tmp_path / "d"
    d.mkdir()
    (d / "x").write_bytes(b"1")
    first = sha256_path(d)
    assert len(first) == 64
    (d / "x").write_bytes(b"2")
    assert sha256_path(d) != first
    (d / "x").write_bytes(b"1")
    (d / "y").write_bytes(b"")
    assert sha256_path(d) != first


def test_pickle_with_code_rejected(tmp_path, tiny_model_dir):
    shutil.copy(tiny_model_dir / "config.json", tmp_path)

    class Evil:  # weights_only'nin reddedeceği nesne
        def __reduce__(self):
            return (print, ("x",))

    torch.save({"a": Evil()}, tmp_path / "m.pt")
    with pytest.raises(UnsafeModelFile, match="safetensors"):
        load_model(tmp_path / "m.pt", device="cpu")


def test_pt_without_config_rejected(tmp_path):
    torch.save({"a": torch.zeros(1)}, tmp_path / "m.pt")
    with pytest.raises(UnsupportedArchitecture, match="config.json"):
        load_model(tmp_path / "m.pt", device="cpu")


def test_pt_state_dict_loads(tmp_path, tiny_model_dir):
    ref = load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if (tiny_model_dir / name).exists():
            shutil.copy(tiny_model_dir / name, tmp_path)
    torch.save(ref.model.state_dict(), tmp_path / "weights.pt")
    lm = load_model(tmp_path / "weights.pt", device="cpu", dtype=torch.float32)
    assert lm.name == "weights"
    assert lm.arch.n_heads == 4
    assert len(lm.source_sha256) == 64
    for (n1, p1), (n2, p2) in zip(ref.model.state_dict().items(), lm.model.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_unsupported_family(tmp_path, tiny_model_dir):
    shutil.copytree(tiny_model_dir, tmp_path / "m")
    cfg = json.loads((tmp_path / "m/config.json").read_text())
    cfg["model_type"] = "mamba"
    (tmp_path / "m/config.json").write_text(json.dumps(cfg))
    with pytest.raises(UnsupportedArchitecture, match="desteklenmiyor"):
        load_model(tmp_path / "m", device="cpu")


def test_gguf_unsupported_arch_message(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("GGUF model with architecture xyz is not supported")

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", boom)
    with pytest.raises(UnsupportedArchitecture, match="desteklenmiyor"):
        load_model(tmp_path / "x.gguf", device="cpu")


def test_gguf_passes_gguf_file_and_safety_flags(tmp_path, monkeypatch):
    seen = {}

    def boom(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise NotImplementedError("not supported")

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", boom)
    with pytest.raises(UnsupportedArchitecture):
        load_model(tmp_path / "x.gguf", device="cpu", dtype=torch.float32)
    assert seen["args"] == (tmp_path,)
    assert seen["kwargs"]["gguf_file"] == "x.gguf"
    assert seen["kwargs"]["trust_remote_code"] is False
    assert seen["kwargs"]["local_files_only"] is True
    assert seen["kwargs"]["dtype"] == torch.float32


def test_pad_token_falls_back_to_eos(tmp_path, tiny_model_dir):
    shutil.copytree(tiny_model_dir, tmp_path / "m")
    tok_cfg = json.loads((tmp_path / "m/tokenizer_config.json").read_text())
    tok_cfg["pad_token"] = None
    (tmp_path / "m/tokenizer_config.json").write_text(json.dumps(tok_cfg))
    lm = load_model(tmp_path / "m", device="cpu", dtype=torch.float32)
    assert lm.tokenizer.pad_token == lm.tokenizer.eos_token


def test_license_from_readme_and_config(tmp_path, tiny_model_dir):
    assert load_model(tiny_model_dir, device="cpu", dtype=torch.float32).license is None
    shutil.copytree(tiny_model_dir, tmp_path / "m")
    (tmp_path / "m/README.md").write_text("---\nlanguage: tr\nlicense: apache-2.0\n---\n# x\n")
    lm = load_model(tmp_path / "m", device="cpu", dtype=torch.float32)
    assert lm.license == "apache-2.0"
