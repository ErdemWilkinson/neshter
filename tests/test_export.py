import json
import sys
from pathlib import Path

import pytest
import torch
from conftest import logits, recipe_for
from transformers import AutoModelForCausalLM

import neshter.export as export
from neshter.compare import Summary
from neshter.export import (
    bake,
    find_llama_cpp,
    model_card,
    ollama_create,
    save_safetensors,
    to_gguf,
    verify_source_untouched,
)
from neshter.interventions import DirectionAblation, HeadAblation, apply


def _state(lm):
    return {k: v.detach().clone() for k, v in lm.model.state_dict().items()}


def test_bake_matches_hook(lm):
    iv = [HeadAblation(1, 2)]
    with apply(lm, iv):
        hooked = logits(lm, "deneme metni")
    undo = bake(lm, iv)
    baked = logits(lm, "deneme metni")
    undo()
    assert torch.allclose(hooked, baked, atol=1e-5)


def test_bake_direction_matches_hook(lm):
    g = torch.Generator().manual_seed(0)
    iv = [
        DirectionAblation(torch.randn(lm.arch.d_model, generator=g)),
        DirectionAblation(torch.randn(lm.arch.d_model, generator=g)),
        HeadAblation(0, 3),
    ]
    with apply(lm, iv):
        hooked = logits(lm, "deneme metni")
    base = logits(lm, "deneme metni")
    undo = bake(lm, iv)
    baked = logits(lm, "deneme metni")
    undo()
    assert not torch.allclose(hooked, base, atol=1e-3)  # müdahale gerçekten etkili
    assert torch.allclose(hooked, baked, atol=1e-4)


def test_undo_restores(lm):
    base = logits(lm, "x")
    undo = bake(lm, [HeadAblation(0, 0)])
    undo()
    assert torch.equal(logits(lm, "x"), base)


def test_undo_restores_direction_exactly(lm):
    before = _state(lm)
    undo = bake(lm, [DirectionAblation(torch.ones(lm.arch.d_model))])
    undo()
    after = lm.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_bake_direction_gemma2_not_implemented(lm, monkeypatch):
    monkeypatch.setattr(lm, "arch", lm.arch.__class__(**{**lm.arch.__dict__, "family": "gemma2"}))
    before = _state(lm)
    with pytest.raises(NotImplementedError, match="gemma2"):
        bake(lm, [DirectionAblation(torch.ones(lm.arch.d_model))])
    assert all(torch.equal(before[k], v) for k, v in lm.model.state_dict().items())


@pytest.fixture(scope="module")
def tied_lm(tmp_path_factory):
    """Conftest'teki minik Qwen2 yapılandırmasının bağlı gömmeli (tie_word_embeddings=True) eşi."""
    import conftest
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from neshter.loader import load_model

    path = tmp_path_factory.mktemp("tiny_qwen2_tied")
    tokenizer = conftest._build_tokenizer()
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=conftest.VOCAB_SIZE,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    Qwen2ForCausalLM(config).to(torch.float32).save_pretrained(path)
    tokenizer.save_pretrained(path)
    m = load_model(path, device="cpu", dtype=torch.float32)
    assert m.model.get_output_embeddings().weight is m.model.get_input_embeddings().weight
    return m


def test_tied_embeddings_bake_save_undo(tied_lm, tmp_path):
    lm = tied_lm
    iv = [DirectionAblation(torch.randn(lm.arch.d_model, generator=torch.Generator().manual_seed(5)))]
    base = logits(lm, "deneme metni")
    with apply(lm, iv):
        hooked = logits(lm, "deneme metni")
    assert not torch.allclose(hooked, base, atol=1e-3)

    undo = bake(lm, iv)
    assert lm.model.get_output_embeddings().weight is not lm.model.get_input_embeddings().weight
    assert torch.allclose(logits(lm, "deneme metni"), hooked, atol=1e-4)
    undo()
    assert lm.model.get_output_embeddings().weight is lm.model.get_input_embeddings().weight
    assert lm.model.config.tie_word_embeddings is True
    assert torch.equal(logits(lm, "deneme metni"), base)

    out = save_safetensors(lm, recipe_for(lm, iv), tmp_path / "out", dtype=torch.float32)
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["tie_word_embeddings"] is False
    assert lm.model.get_output_embeddings().weight is lm.model.get_input_embeddings().weight
    assert lm.model.config.tie_word_embeddings is True
    assert torch.equal(logits(lm, "deneme metni"), base)
    assert verify_source_untouched(lm)
    m2 = AutoModelForCausalLM.from_pretrained(out)
    with torch.no_grad():
        got = m2(**lm.tokenizer("deneme metni", return_tensors="pt")).logits
    assert torch.allclose(got, hooked, atol=1e-4)


def test_save_reload_equivalent(lm, tmp_path):
    out = save_safetensors(lm, recipe_for(lm, [HeadAblation(1, 1)]), tmp_path / "out", dtype=torch.float32)
    m2 = AutoModelForCausalLM.from_pretrained(out)
    with apply(lm, [HeadAblation(1, 1)]):
        ref = logits(lm, "abc")
    with torch.no_grad():
        got = m2(**lm.tokenizer("abc", return_tensors="pt")).logits
    assert torch.allclose(got, ref, atol=1e-4)
    assert (out / "recipe.nester.json").exists() and "neshter" in (out / "README.md").read_text(
        encoding="utf-8"
    )
    assert verify_source_untouched(lm)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "abliterated" not in cfg["neshter"] and "version" in cfg["neshter"]
    assert not (tmp_path / "out.tmp").exists()


def test_save_reload_direction_equivalent(lm, tmp_path):
    iv = [DirectionAblation(torch.randn(lm.arch.d_model, generator=torch.Generator().manual_seed(3)))]
    base = logits(lm, "abc")
    out = save_safetensors(lm, recipe_for(lm, iv), tmp_path / "out", dtype=torch.float32)
    assert torch.equal(logits(lm, "abc"), base)
    m2 = AutoModelForCausalLM.from_pretrained(out)
    with apply(lm, iv):
        ref = logits(lm, "abc")
    with torch.no_grad():
        got = m2(**lm.tokenizer("abc", return_tensors="pt")).logits
    assert torch.allclose(got, ref, atol=1e-4)


def test_failed_save_leaves_nothing(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(
        lm.model, "save_pretrained", lambda *a, **k: (_ for _ in ()).throw(OSError("disk dolu"))
    )
    base = logits(lm, "abc")
    with pytest.raises(OSError):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.tmp").exists()
    assert torch.equal(logits(lm, "abc"), base) and verify_source_untouched(lm)


def test_failed_late_step_cleans_tmp(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "save_recipe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    base = logits(lm, "abc")
    with pytest.raises(RuntimeError):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.tmp").exists()
    assert torch.equal(logits(lm, "abc"), base)


def test_refusal_card_tag(lm, tmp_path):
    out = save_safetensors(
        lm, recipe_for(lm, [HeadAblation(0, 0)], concept="refusal"), tmp_path / "o"
    )
    assert "safety-ablated" in (out / "README.md").read_text(encoding="utf-8")
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["neshter"]["abliterated"] == "refusal"


def test_existing_out_dir_not_overwritten(lm, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 0)]), out)
    assert (out / "keep.txt").read_text() == "x"
    assert not (tmp_path / "out.tmp").exists()


def test_disk_space_precheck(lm, tmp_path, monkeypatch):
    class Usage:
        total, used, free = 10**12, 10**12, 10

    monkeypatch.setattr(export.shutil, "disk_usage", lambda p: Usage())
    base = logits(lm, "abc")
    with pytest.raises(OSError, match="GB"):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 0)]), tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.tmp").exists()
    assert torch.equal(logits(lm, "abc"), base)


def test_refuses_to_write_inside_source(lm):
    with pytest.raises(ValueError):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 0)]), Path(lm.source_path) / "out")
    assert verify_source_untouched(lm)


def test_model_card_contents(lm):
    r = recipe_for(lm, [HeadAblation(1, 2), DirectionAblation(torch.ones(lm.arch.d_model))], "refusal")
    card = model_card(lm, r, Summary("good", "Türkçe özet.", "English summary."), safety_ablated=True)
    assert card.startswith("---\n")
    assert "license: other" in card and "safety-ablated" in card and "neshter" in card
    assert "Türkçe özet." in card and "English summary." in card
    assert lm.source_sha256 in card and "direction (64)" in card
    assert "Llama" not in card
    plain = model_card(lm, recipe_for(lm, [HeadAblation(0, 0)]), None, safety_ablated=False)
    assert "safety-ablated" not in plain


def test_model_card_llama_notice(lm, monkeypatch):
    monkeypatch.setattr(lm, "arch", lm.arch.__class__(**{**lm.arch.__dict__, "family": "llama"}))
    card = model_card(lm, recipe_for(lm, [HeadAblation(0, 0)]), None, safety_ablated=False)
    assert "Llama" in card


def test_find_llama_cpp_env(tmp_path, monkeypatch):
    exe = "llama-quantize.exe" if sys.platform == "win32" else "llama-quantize"
    (tmp_path / exe).write_bytes(b"")
    monkeypatch.setenv("NESHTER_LLAMA_CPP", str(tmp_path))
    assert find_llama_cpp() == tmp_path


def test_find_llama_cpp_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("NESHTER_LLAMA_CPP", raising=False)
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    assert find_llama_cpp() is None


def test_to_gguf_commands(tmp_path, monkeypatch):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    (llama / export._quantize_name()).write_text("")
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    out = tmp_path / "m.gguf"
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        Path(cmd[-1] if len(calls) == 1 else cmd[-2]).write_bytes(b"x")  # 1: f16 tmp, 2: q tmp

    monkeypatch.setattr(export.subprocess, "run", fake_run)
    assert to_gguf(hf, out, "Q4_K_M", llama) == out
    f16 = Path(str(out) + ".f16.tmp")
    qtmp = Path(str(out) + ".tmp")
    assert calls[0] == [
        sys.executable, str(llama / "convert_hf_to_gguf.py"), str(hf),
        "--outtype", "f16", "--outfile", str(f16),
    ]
    assert Path(calls[1][0]).name.startswith("llama-quantize") and Path(calls[1][0]).parent == llama
    assert calls[1][1:] == [str(f16), str(qtmp), "Q4_K_M"]
    assert out.exists() and not f16.exists() and not qtmp.exists()


def test_to_gguf_missing_converter(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
        to_gguf(tmp_path / "hf", tmp_path / "m.gguf", "Q8_0", tmp_path)


def test_ollama_create(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(export.shutil, "which", lambda name: r"C:\Araclar\ollama.exe")
    monkeypatch.setattr(export.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))
    ollama_create("benim-model", gguf)
    modelfile = tmp_path / "Modelfile"
    assert modelfile.read_text(encoding="utf-8").strip() == f"FROM {gguf.resolve()}"
    assert calls == [([r"C:\Araclar\ollama.exe", "create", "benim-model", "-f", str(modelfile)], True)]


def test_ollama_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="Ollama"):
        ollama_create("x", tmp_path / "m.gguf")


# --- Fix wave: I6(b) CPU anlık görüntü + bloklu dikleştirme, I7 kayıt dtype'ı, Minor 10/19 ---------


def test_snapshot_is_independent_cpu_copy():
    w = torch.randn(4, 3)
    snap = export._snapshot(w)
    assert snap.device.type == "cpu" and torch.equal(snap, w)
    assert snap.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()


def test_bake_snapshots_go_through_cpu_helper(lm, monkeypatch):
    seen = []
    real = export._snapshot
    monkeypatch.setattr(export, "_snapshot", lambda p: seen.append(real(p)) or seen[-1])
    undo = bake(lm, [DirectionAblation(torch.ones(lm.arch.d_model)), HeadAblation(0, 1)])
    undo()
    # embed + 2 katman × (o_proj, down_proj) = 5 parametre (head o_proj'u yönle aynı tensör)
    assert len(seen) == 5 and all(t.device.type == "cpu" for t in seen)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_orthogonalization_matches_reference(dtype):
    g = torch.Generator().manual_seed(0)
    r = torch.randn(8, generator=g)
    r = r / r.norm()
    rows = torch.randn(37, 8, generator=g).to(dtype)
    ref_rows = (rows.double() - torch.outer(rows.double() @ r.double(), r.double())).to(dtype)
    export._orth_rows_(rows, r, chunk=5)  # birden çok blok
    cols = torch.randn(8, 29, generator=g).to(dtype)
    ref_cols = (cols.double() - torch.outer(r.double(), r.double() @ cols.double())).to(dtype)
    export._orth_out_(cols, r, chunk=4)
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.allclose(rows.float(), ref_rows.float(), atol=tol)
    assert torch.allclose(cols.float(), ref_cols.float(), atol=tol)
    assert rows.dtype == dtype and cols.dtype == dtype


def test_float32_model_saves_bf16_by_default(lm, tmp_path):
    from safetensors import safe_open

    out = save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "out")
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        dtypes = {f.get_slice(k).get_dtype() for k in f.keys()}
    assert dtypes == {"BF16"}
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["dtype"] == "bfloat16"
    assert next(lm.model.parameters()).dtype == torch.float32  # bellekteki model değişmedi
    assert verify_source_untouched(lm)


def test_save_dtype_keeps_float16_sources(lm, monkeypatch):
    monkeypatch.setattr(export, "_source_config", lambda _lm: {"torch_dtype": "float16"})
    assert export.save_dtype(lm) == torch.float16
    monkeypatch.setattr(export, "_source_config", lambda _lm: {"dtype": "float32"})
    assert export.save_dtype(lm) == torch.bfloat16


def test_safety_flag_cannot_be_removed_and_can_be_forced(lm, tmp_path):
    out = save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)], concept=None), tmp_path / "o", safety_ablated=True)
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["neshter"]["abliterated"] == "refusal"
    assert "safety-ablated" in (out / "README.md").read_text(encoding="utf-8")
    out2 = save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)], concept="refusal"), tmp_path / "o2", safety_ablated=False)
    assert json.loads((out2 / "config.json").read_text(encoding="utf-8"))["neshter"]["abliterated"] == "refusal"


def test_license_files_are_copied(tmp_path, tiny_model_dir):
    import shutil as sh

    from neshter.loader import load_model

    src = tmp_path / "src"
    sh.copytree(tiny_model_dir, src)
    (src / "LICENSE").write_text("Llama lisans metni", encoding="utf-8")
    (src / "USE_POLICY.md").write_text("Kullanım politikası", encoding="utf-8")
    m = load_model(src, device="cpu", dtype=torch.float32)
    out = save_safetensors(m, recipe_for(m, [HeadAblation(0, 1)]), tmp_path / "out")
    assert (out / "LICENSE").read_text(encoding="utf-8") == "Llama lisans metni"
    assert (out / "USE_POLICY.md").is_file()


def test_to_gguf_checks_quantize_and_config(tmp_path):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    hf = tmp_path / "hf"
    hf.mkdir()
    with pytest.raises(FileNotFoundError, match="llama-quantize"):
        to_gguf(hf, tmp_path / "m.gguf", "Q8_0", llama)
    (llama / export._quantize_name()).write_text("")
    with pytest.raises(FileNotFoundError, match="config.json"):
        to_gguf(hf, tmp_path / "m.gguf", "Q8_0", llama)


@pytest.mark.parametrize("name", ["-rm", "../x", "a b", "", "x" * 65])
def test_ollama_create_rejects_bad_names(tmp_path, monkeypatch, name):
    monkeypatch.setattr(export.shutil, "which", lambda n: "ollama")
    monkeypatch.setattr(export.subprocess, "run", lambda *a, **k: pytest.fail("çalıştırılmamalı"))
    with pytest.raises(ValueError, match="Ollama model adı"):
        ollama_create(name, tmp_path / "m.gguf")
