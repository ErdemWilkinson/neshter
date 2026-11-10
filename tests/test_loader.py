import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

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


# --- Fix wave: C1 (Ollama blob), I7 (cihaz politikası), Minor 13/17 ---------------------------------


def test_suffixless_gguf_blob_routes_to_gguf_loader(tmp_path, monkeypatch):
    blob = tmp_path / "blobs" / ("sha256-" + "a" * 64)
    blob.parent.mkdir()
    blob.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 64)  # Ollama blob: uzantı yok
    seen = {}

    def boom(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise NotImplementedError("not supported")

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", boom)
    with pytest.raises(UnsupportedArchitecture, match="GGUF"):
        load_model(blob, device="cpu")
    assert seen["args"] == (blob.parent,)
    assert seen["kwargs"]["gguf_file"] == blob.name
    assert seen["kwargs"]["trust_remote_code"] is False


def test_suffixless_non_gguf_file_still_unsupported(tmp_path):
    f = tmp_path / ("sha256-" + "b" * 64)
    f.write_bytes(b"PK\x03\x04 bir zip")
    with pytest.raises(UnsupportedArchitecture, match="desteklenmiyor"):
        load_model(f, device="cpu")


def test_is_gguf_magic(tmp_path):
    (tmp_path / "a").write_bytes(b"GGUFxxxx")
    (tmp_path / "b").write_bytes(b"GGU")
    assert loader.is_gguf(tmp_path / "a")
    assert not loader.is_gguf(tmp_path / "b") and not loader.is_gguf(tmp_path / "yok")


def test_display_name_override(tiny_model_dir):
    lm = load_model(tiny_model_dir, device="cpu", dtype=torch.float32, name="qwen2.5:7b")
    assert lm.name == "qwen2.5:7b"


@pytest.mark.parametrize("device,n,ram,native,expected", [
    ("cuda", 10**9, 64 * 2**30, False, torch.bfloat16),
    ("cpu", 5 * 10**8, 16 * 2**30, False, torch.float32),   # 0.5B·4·1.2 = 2.4 GB < 0.7·16 GB
    ("cpu", 8 * 10**9, 16 * 2**30, False, torch.bfloat16),  # 38 GB > 11.2 GB
    ("cpu", 5 * 10**8, 16 * 2**30, True, torch.bfloat16),   # AVX512_BF16 / AMX
])
def test_pick_compute_dtype_table(device, n, ram, native, expected):
    assert loader.pick_compute_dtype(device, n, ram, native) == expected


def test_auto_dtype_loads_float32_on_cpu_without_native_bf16(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(loader, "cpu_has_native_bf16", lambda: False)
    monkeypatch.setattr(loader, "available_ram_bytes", lambda: 1 << 40)
    lm = loader.load_model(tiny_model_dir, device="cpu")
    assert next(lm.model.parameters()).dtype == torch.float32 and lm.device_note is None


def test_auto_dtype_bf16_with_note_when_ram_tight(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(loader, "cpu_has_native_bf16", lambda: False)
    monkeypatch.setattr(loader, "available_ram_bytes", lambda: 1)
    lm = loader.load_model(tiny_model_dir, device="cpu")
    assert next(lm.model.parameters()).dtype == torch.bfloat16
    assert loader.SLOW_BF16_NOTE in lm.device_note


def test_explicit_dtype_is_respected(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(loader, "cpu_has_native_bf16", lambda: False)
    lm = loader.load_model(tiny_model_dir, device="cpu", dtype=torch.bfloat16)
    assert next(lm.model.parameters()).dtype == torch.bfloat16


def test_native_bf16_cpu_keeps_bf16_without_note(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(loader, "cpu_has_native_bf16", lambda: True)
    lm = loader.load_model(tiny_model_dir, device="cpu")
    assert next(lm.model.parameters()).dtype == torch.bfloat16 and lm.device_note is None


def test_fp32_policy_loads_without_bf16_rounding(tmp_path, tiny_model_dir, monkeypatch):
    # a float32 checkpoint must not be rounded through bf16 on its way to float32
    shutil.copytree(tiny_model_dir, tmp_path / "m")
    f = tmp_path / "m" / "model.safetensors"
    sd = load_file(f)
    key = next(k for k, v in sd.items() if v.ndim == 2)
    sd[key] = torch.full_like(sd[key], 1 + 2**-12, dtype=torch.float32)   # not bf16-representable
    save_file(sd, f, metadata={"format": "pt"})
    monkeypatch.setattr(loader, "cpu_has_native_bf16", lambda: False)
    monkeypatch.setattr(loader, "available_ram_bytes", lambda: 1 << 40)
    lm = loader.load_model(tmp_path / "m", device="cpu")
    assert torch.equal(dict(lm.model.named_parameters())[key], sd[key])


def test_hardware_probes_return_plain_values():
    from neshter import hardware

    assert isinstance(hardware.cpu_has_native_bf16(), bool)
    assert hardware.available_ram_bytes() > 0


def test_auto_device_stays_on_cpu_when_vram_too_small(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1000, 8 * 1024**3))
    lm = load_model(tiny_model_dir, device="auto", dtype=torch.float32)
    assert next(lm.model.parameters()).device.type == "cpu"
    assert lm.device_note == loader.GPU_FALLBACK_NOTE


def test_auto_device_picks_cuda_when_it_fits(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (8 * 1024**3, 8 * 1024**3))
    seen = []
    monkeypatch.setattr(loader, "_pick_device", lambda m, d, _orig=loader._pick_device: seen.append(_orig(m, d)) or ("cpu", None))
    load_model(tiny_model_dir, device="auto", dtype=torch.float32)
    assert seen == [("cuda", None)]


def test_folder_without_safetensors_is_rejected(tmp_path, tiny_model_dir):
    ref = load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    m = tmp_path / "m"
    m.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if (tiny_model_dir / name).exists():
            shutil.copy(tiny_model_dir / name, m)
    torch.save(ref.model.state_dict(), m / "pytorch_model.bin")
    with pytest.raises(UnsupportedArchitecture, match="safetensors"):
        load_model(m, device="cpu", dtype=torch.float32)


def test_read_license_is_public():
    assert loader.read_license is loader._read_license
