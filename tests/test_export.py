import dataclasses
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from conftest import logits, recipe_for
from transformers import AutoModelForCausalLM

import neshter.export as export
from neshter.compare import Summary
from neshter.export import (
    GEMMA_NOTICE,
    QUANT_BYTES_PER_PARAM,
    DiskFull,
    LlamaCppMissing,
    bake,
    estimate_export_bytes,
    export_gguf,
    gguf_metadata,
    modelfile_license,
    write_modelfile,
    is_llama3,
    license_notices,
    find_llama_cpp,
    model_card,
    ollama_create,
    save_safetensors,
    to_gguf,
    verify_source_untouched,
)
from neshter.interventions import DirectionAblation, HeadAblation, apply
from neshter.progress import Cancelled


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


def _gemma_ivs(d_model: int):
    """Birbirine dik olmayan üç yön + iki head (γ-ölçekli tabanlar gerçekten sınansın)."""
    g = torch.Generator().manual_seed(7)
    a, b = torch.randn(d_model, generator=g), torch.randn(d_model, generator=g)
    return [
        DirectionAblation(a),
        DirectionAblation(a + 0.4 * b),
        DirectionAblation(b),
        HeadAblation(0, 2),
        HeadAblation(1, 0),
    ]


GEMMA_TEXT = "merhaba dünya, bugün hava çok güzel"


def test_gemma2_bake_matches_hook(gemma_lm):
    lm = gemma_lm
    iv = _gemma_ivs(lm.arch.d_model)
    before = _state(lm)
    base = logits(lm, GEMMA_TEXT)
    with apply(lm, iv):
        hooked = logits(lm, GEMMA_TEXT)
    undo = bake(lm, iv)
    assert lm.model.get_output_embeddings().weight is not lm.model.get_input_embeddings().weight
    baked = logits(lm, GEMMA_TEXT)
    undo()
    assert not torch.allclose(hooked, base, atol=1e-3)  # müdahale gerçekten etkili
    assert torch.allclose(hooked, baked, atol=1e-4)
    after = lm.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert lm.model.get_output_embeddings().weight is lm.model.get_input_embeddings().weight
    assert lm.model.config.tie_word_embeddings is True
    assert torch.equal(logits(lm, GEMMA_TEXT), base)


def test_gemma2_save_reload_direction_equivalent(gemma_lm, tmp_path):
    from neshter.loader import load_model

    lm = gemma_lm
    iv = _gemma_ivs(lm.arch.d_model)
    with apply(lm, iv):
        hooked = logits(lm, GEMMA_TEXT)
    out = save_safetensors(lm, recipe_for(lm, iv), tmp_path / "out", dtype=torch.float32)
    assert verify_source_untouched(lm)
    assert lm.model.get_output_embeddings().weight is lm.model.get_input_embeddings().weight
    m2 = load_model(out, device="cpu", dtype=torch.float32)
    assert m2.arch.family == "gemma2"
    assert torch.allclose(logits(m2, GEMMA_TEXT), hooked, atol=1e-4)


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
    card = model_card(lm, r, Summary("good", "Türkçe özet.", "English summary.", 24), safety_ablated=True)
    assert card.startswith("---\n")
    assert "license: other" in card and "safety-ablated" in card and "neshter" in card
    assert "Türkçe özet." in card and "English summary." in card
    assert lm.source_sha256 in card and "direction (64)" in card
    assert "Llama" not in card
    plain = model_card(lm, recipe_for(lm, [HeadAblation(0, 0)]), None, safety_ablated=False)
    assert "safety-ablated" not in plain


def _with(lm, license=None, family=None, name=None, license_text=None):
    arch = dataclasses.replace(lm.arch, family=family) if family else lm.arch
    kw = {"arch": arch, "license": license}
    if name:
        kw["name"] = name
    if license_text is not None:
        kw["license_text"] = license_text
    return dataclasses.replace(lm, **kw)


def _card(m, ivs=None):
    return model_card(m, recipe_for(m, ivs or []), None, False)


def test_llama3_notice_only_for_llama3(lm):
    l3 = _with(lm, "llama3.1", "llama")
    smol = dataclasses.replace(l3, license="apache-2.0", name="SmolLM2-135M-Instruct")
    assert "Built with Llama" in _card(l3)
    assert 'start with "Llama"' in _card(l3) and "başlamalı" in _card(l3)
    assert "Built with Llama" not in _card(smol)
    assert "Llama lisansı, türetilmiş model adlarının" not in _card(smol)  # eski her-Llama cümlesi yok


def test_is_llama3_rules(lm):
    assert is_llama3(_with(lm, "llama3.2", "llama"))
    assert is_llama3(_with(lm, "other", "llama", name="Meta-Llama-3-8B"))
    assert is_llama3(_with(lm, None, "llama", name="llama_3.1-8b"))
    assert not is_llama3(_with(lm, "llama3.1", "qwen2"))  # aile llama değil
    assert not is_llama3(_with(lm, "mit", "llama", name="TinyLlama-1.1B"))
    big = _with(lm, None, "llama")
    big.model.config.vocab_size = 128256
    try:
        assert is_llama3(big)
    finally:
        big.model.config.vocab_size = 512


def test_gemma_notice_and_notice_file(gemma_lm, tmp_path):
    g = dataclasses.replace(gemma_lm, license="gemma")
    notices = "\n".join(license_notices(g))
    assert GEMMA_NOTICE in notices and "ai.google.dev/gemma/prohibited_use_policy" in notices
    assert "ai.google.dev/gemma/terms" in _card(g)
    out = save_safetensors(g, recipe_for(g, [HeadAblation(0, 1)]), tmp_path / "out")
    assert "ai.google.dev/gemma/terms" in (out / "NOTICE").read_text(encoding="utf-8")
    assert "ai.google.dev/gemma/terms" in (out / "README.md").read_text(encoding="utf-8")


def test_existing_notice_file_is_kept(gemma_lm, tmp_path):
    src = tmp_path / "src"
    shutil.copytree(gemma_lm.source_path, src)
    (src / "NOTICE").write_text("kaynağın kendi NOTICE'i", encoding="utf-8")
    from neshter.loader import load_model

    m = load_model(src, device="cpu", dtype=torch.float32)
    out = save_safetensors(m, recipe_for(m, [HeadAblation(0, 1)]), tmp_path / "out")
    assert (out / "NOTICE").read_text(encoding="utf-8") == "kaynağın kendi NOTICE'i"


def test_qwen_non_apache_warning(lm):
    warn = "Apache-2.0 değil"
    assert warn in "\n".join(license_notices(_with(lm, "qwen-research")))
    assert "qwen-research" in _card(_with(lm, "qwen-research"))
    assert "is not Apache-2.0" in "\n".join(license_notices(_with(lm, "qwen")))
    assert warn not in "\n".join(license_notices(_with(lm, "apache-2.0")))
    assert license_notices(_with(lm, "apache-2.0")) == []
    assert "bilinmiyor" in "\n".join(license_notices(_with(lm, None)))


def test_unknown_license_card(lm):
    card = _card(_with(lm, None))
    assert "license: other" in card and "license_name: unknown" in card and "lisansı bulunamadı" in card
    assert "not found" in card
    known = _card(_with(lm, "apache-2.0"))
    assert "license: apache-2.0" in known and "license_name" not in known and "bulunamadı" not in known


def test_untrusted_license_id_is_limited_in_card(lm):
    evil = "x]\n# SAHTE BAŞLIK <script>alert(1)</script> [a](http://e.vil) `" + "A" * 500
    card = _card(_with(lm, evil))
    assert "<script>" not in card and "[a](" not in card and "\n# SAHTE" not in card
    assert "A" * 100 not in card


def test_license_text_written_when_no_license_file(lm, tmp_path):
    m = _with(lm, "mit", license_text="X")
    out = save_safetensors(m, recipe_for(m, [HeadAblation(0, 1)]), tmp_path / "out")
    assert (out / "LICENSE").read_text(encoding="utf-8") == "X"
    assert not (out / "NOTICE").exists()  # qwen2: NOTICE yazılmaz


def test_source_license_file_not_overwritten_by_license_text(tmp_path, tiny_model_dir):
    from neshter.loader import load_model

    src = tmp_path / "src"
    shutil.copytree(tiny_model_dir, src)
    (src / "LICENSE.txt").write_text("kaynağın lisansı", encoding="utf-8")
    m = load_model(src, device="cpu", dtype=torch.float32)
    assert m.license_text == "kaynağın lisansı"
    out = save_safetensors(m, recipe_for(m, [HeadAblation(0, 1)]), tmp_path / "out")
    assert (out / "LICENSE.txt").read_text(encoding="utf-8") == "kaynağın lisansı"
    assert not (out / "LICENSE").exists()


def test_card_has_modification_line(lm):
    card = _card(lm, [HeadAblation(0, 1), DirectionAblation(torch.randn(64))])
    assert re.search(r"Modified with Neşter \S+ on \d{4}-\d{2}-\d{2}: 1 heads silenced, 1 directions removed\.", card)


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
    ollama_create("benim-model", gguf, license_text="Apache-2.0 metni")
    modelfile = tmp_path / "Modelfile"
    text = modelfile.read_text(encoding="utf-8")
    assert text.startswith(f"FROM {gguf.resolve()}\n")
    assert 'LICENSE """' in text and "Apache-2.0 metni" in text
    assert calls == [([r"C:\Araclar\ollama.exe", "create", "benim-model", "-f", str(modelfile)], True)]


def test_ollama_create_without_license_keeps_bare_modelfile(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(export.shutil, "which", lambda name: "ollama")
    monkeypatch.setattr(export.subprocess, "run", lambda cmd, check: None)
    ollama_create("benim-model", gguf)
    assert (tmp_path / "Modelfile").read_text(encoding="utf-8") == f"FROM {gguf.resolve()}\n"


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
    basis = r.unsqueeze(1)  # [d, k'] taban
    rows = torch.randn(37, 8, generator=g).to(dtype)
    ref_rows = (rows.double() - torch.outer(rows.double() @ r.double(), r.double())).to(dtype)
    export._orth_rows_(rows, basis, chunk=5)  # birden çok blok
    cols = torch.randn(8, 29, generator=g).to(dtype)
    ref_cols = (cols.double() - torch.outer(r.double(), r.double() @ cols.double())).to(dtype)
    export._orth_out_(cols, basis, chunk=4)
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


def test_disk_estimate_uses_save_dtype_not_memory_dtype(lm):  # lm is float32 in memory
    n = sum(p.numel() for p in lm.model.parameters())
    assert export._estimate_bytes(lm, directions=False) == n * 2


# --------------------------------------------------------------- span projection (çoklu yön)


def test_bake_two_correlated_directions_matches_hook(lm):
    g = torch.Generator().manual_seed(10)
    r1 = torch.randn(lm.arch.d_model, generator=g)
    r2 = 0.8 * r1 / r1.norm() + 0.6 * torch.randn(lm.arch.d_model, generator=g)
    ivs = [DirectionAblation(r1), DirectionAblation(r2)]
    before = {k: v.clone() for k, v in lm.model.state_dict().items()}
    with apply(lm, ivs):
        hooked = logits(lm, "merhaba dünya")
    undo = export.bake(lm, ivs)
    baked = logits(lm, "merhaba dünya")
    undo()
    assert torch.allclose(hooked, baked, atol=1e-4)
    assert all(torch.equal(v, lm.model.state_dict()[k]) for k, v in before.items())


def test_bake_duplicate_and_dependent_directions_with_heads_match_hook(lm):
    g = torch.Generator().manual_seed(11)
    r, u = torch.randn(lm.arch.d_model, generator=g), torch.randn(lm.arch.d_model, generator=g)
    ivs = [
        DirectionAblation(r),
        HeadAblation(1, 0),
        DirectionAblation(r),  # kopya
        DirectionAblation(-2 * r),  # bağımlı
        DirectionAblation(0.3 * r + u),
        HeadAblation(0, 2),
    ]
    base = logits(lm, "deneme metni")
    with apply(lm, ivs):
        hooked = logits(lm, "deneme metni")
    undo = export.bake(lm, ivs)
    baked = logits(lm, "deneme metni")
    undo()
    assert torch.isfinite(baked).all() and not torch.allclose(hooked, base, atol=1e-3)
    assert torch.allclose(hooked, baked, atol=1e-4)
    assert torch.equal(logits(lm, "deneme metni"), base)


def test_bake_round_to_equals_saved_weights(lm, tmp_path):
    ivs = [DirectionAblation(torch.randn(lm.arch.d_model, generator=torch.Generator().manual_seed(12)))]
    undo = export.bake(lm, ivs, round_to=torch.bfloat16)
    touched = {k: v.clone() for k, v in lm.model.state_dict().items()}
    undo()
    export.save_safetensors(lm, recipe_for(lm, ivs), tmp_path / "m")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "m", dtype=torch.float32).state_dict()
    writers = [k for k in touched if k.endswith(("o_proj.weight", "down_proj.weight", "embed_tokens.weight"))]
    assert writers  # yalnızca dokunulan parametreler karşılaştırılır (round_to sınırlaması)
    for k in writers:
        assert torch.equal(touched[k], reloaded[k]), k


def test_bake_round_to_rounds_untied_lm_head_copy(tied_lm):
    """Bağlı gömmede bake lm_head'e ayrı kopya verir; kayıtta o da bf16'ya yuvarlanır, bellekte de öyle olmalı."""
    lm = tied_lm
    emb = lm.model.get_input_embeddings().weight
    orig = emb.detach().clone()
    iv = [DirectionAblation(torch.randn(lm.arch.d_model, generator=torch.Generator().manual_seed(8)))]
    undo = export.bake(lm, iv, round_to=torch.bfloat16)
    try:
        head = lm.model.get_output_embeddings().weight
        assert head is not lm.model.get_input_embeddings().weight
        assert torch.equal(head, orig.to(torch.bfloat16).to(orig.dtype))
        assert not torch.equal(head, orig)  # gerçekten yuvarlanmış (bf16'da tam gösterilemeyen değerler)
    finally:
        undo()
    assert lm.model.get_output_embeddings().weight is lm.model.get_input_embeddings().weight
    assert torch.equal(lm.model.get_input_embeddings().weight, orig)


# --- Task 9 fix round 1 ---------------------------------------------------------------------------


def test_is_llama3_regex_is_anchored(lm):
    assert not is_llama3(_with(lm, None, "llama", name="llama-30b"))
    assert not is_llama3(_with(lm, None, "llama", name="open-llama-33B"))
    assert is_llama3(_with(lm, None, "llama", name="Llama-3-8B"))
    assert is_llama3(_with(lm, None, "llama", name="Llama-3.1-8B-Instruct"))


def test_license_ids_compare_case_insensitively(lm):
    assert is_llama3(_with(lm, "Llama3.1", "llama"))
    assert license_notices(_with(lm, "Apache-2.0")) == []
    assert license_notices(_with(lm, "APACHE-2.0")) == []


def test_qwen_unknown_license_gets_one_combined_notice(lm):
    notes = license_notices(_with(lm, None))
    assert len(notes) == 1
    assert "lisansı bulunamadı" in notes[0] and "Qwen" in notes[0] and "not found" in notes[0]
    assert len(license_notices(_with(lm, "qwen-research"))) == 1
    non_qwen = _with(lm, None, "llama", name="SmolLM2")
    assert len(license_notices(non_qwen)) == 1 and "Qwen" not in license_notices(non_qwen)[0]


def test_card_heading_escapes_model_name(lm):
    evil = _with(lm, "mit", name="x\n## SAHTE <img src=x onerror=1> [a](http://e.vil) `" + "N" * 150)
    card = _card(evil)
    heading = next(line for line in card.splitlines() if line.startswith("# "))
    assert "<img" not in heading and "[a](" not in heading and "`" not in heading
    assert len(heading) < 140
    assert "\n## SAHTE" not in card
    assert _card(_with(lm, "mit", name="qwen2.5:7b")).count("# qwen2.5:7b (neshter)") == 1


def test_copy_license_files_skips_files_resolving_outside(tmp_path, tiny_model_dir, monkeypatch):
    from neshter.loader import load_model

    src = tmp_path / "src"
    shutil.copytree(tiny_model_dir, src)
    (src / "LICENSE").write_text("kaynağın lisansı", encoding="utf-8")
    (src / "NOTICE").write_text("kaynağın notu", encoding="utf-8")
    m = load_model(src, device="cpu", dtype=torch.float32)
    real = export.inside_model
    # Yalnızca NOTICE'in klasör dışına çözüldüğü varsayılır (Windows'ta sembolik bağ kurulamayabilir).
    monkeypatch.setattr(export, "inside_model", lambda folder, f: False if f.name == "NOTICE" else real(folder, f))
    out = save_safetensors(m, recipe_for(m, [HeadAblation(0, 1)]), tmp_path / "out")
    assert (out / "LICENSE").read_text(encoding="utf-8") == "kaynağın lisansı"
    assert not (out / "NOTICE").exists()


def test_copy_license_files_skips_real_symlink_outside(tmp_path, tiny_model_dir):
    from neshter.loader import load_model

    src = tmp_path / "src"
    shutil.copytree(tiny_model_dir, src)
    secret = tmp_path / "gizli.txt"
    secret.write_text("DIŞARIDAKİ SIR", encoding="utf-8")
    try:
        (src / "USE_POLICY.md").symlink_to(secret)
    except OSError:
        pytest.skip("symlink oluşturulamadı")
    m = load_model(src, device="cpu", dtype=torch.float32)
    out = save_safetensors(m, recipe_for(m, [HeadAblation(0, 1)]), tmp_path / "out")
    assert not (out / "USE_POLICY.md").exists()


# --- Task 10: GGUF/Ollama dışa aktarımı kartı, etiketi, tarifi ve lisansı korur ------------------


def _fake_llama(tmp_path):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    (llama / export._quantize_name()).write_text("")
    return llama


class _Tools:
    """subprocess.run yerine geçer: dönüştürücü --outfile'ı, quantize çıktıyı yazar; argv'ler kaydedilir."""

    def __init__(self, monkeypatch, on_convert=None):
        self.calls: list[list[str]] = []
        self.metadata: dict | None = None
        self.on_convert = on_convert
        monkeypatch.setattr(export.subprocess, "run", self)

    def __call__(self, cmd, check=False, **kw):
        assert isinstance(cmd, list) and check is True and not kw.get("shell")
        self.calls.append(list(cmd))
        if cmd[1].endswith("convert_hf_to_gguf.py"):
            if "--metadata" in cmd:
                self.metadata = json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text(encoding="utf-8"))
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF-f16")
            if self.on_convert:
                self.on_convert()
        else:
            Path(cmd[2]).write_bytes(b"GGUF-q")


def _no_space(monkeypatch, free=1):
    monkeypatch.setattr(export.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - free, free))


def test_export_gguf_keeps_card_recipe_license_next_to_gguf(lm, tmp_path, monkeypatch):
    tools = _Tools(monkeypatch)
    out_dir = tmp_path / "o"
    src_license = lm.license_text
    lm.license_text = "Apache License text"
    try:
        out = export_gguf(
            lm, recipe_for(lm, [HeadAblation(0, 1)], concept="refusal"), out_dir,
            "m", "Q8_0", _fake_llama(tmp_path), safety=True,
        )
    finally:
        lm.license_text = src_license
    assert out == out_dir / "m.gguf"
    names = {p.name for p in out_dir.iterdir()}
    assert {"m.gguf", "README.md", "recipe.nester.json", "LICENSE"} <= names
    assert "_hf" not in names and not any(n.endswith(".tmp") for n in names)
    assert "safety-ablated" in (out_dir / "README.md").read_text(encoding="utf-8")
    assert (out_dir / "LICENSE").read_text(encoding="utf-8") == "Apache License text"
    assert len(tools.calls) == 2  # dönüştür + quantize
    assert tools.metadata["general.tags"] == ["neshter", "safety-ablated", "abliterated"]


def test_export_gguf_failure_removes_out_dir(lm, tmp_path, monkeypatch):
    def boom(cmd, check=False, **kw):
        raise subprocess.CalledProcessError(3, cmd)

    monkeypatch.setattr(export.subprocess, "run", boom)
    out_dir = tmp_path / "o"
    with pytest.raises(RuntimeError, match="3"):
        export_gguf(lm, recipe_for(lm, [HeadAblation(0, 1)]), out_dir, "m", "Q8_0", _fake_llama(tmp_path), safety=False)
    assert not out_dir.exists()


def test_export_gguf_refuses_existing_out_dir_and_bad_name(lm, tmp_path, monkeypatch):
    _Tools(monkeypatch)
    llama = _fake_llama(tmp_path)
    existing = tmp_path / "var"
    existing.mkdir()
    (existing / "keep.txt").write_text("x")
    recipe = recipe_for(lm, [HeadAblation(0, 1)])
    with pytest.raises(FileExistsError):
        export_gguf(lm, recipe, existing, "m", "Q8_0", llama, safety=False)
    assert (existing / "keep.txt").exists()  # var olan klasöre dokunulmaz
    with pytest.raises(ValueError):
        export_gguf(lm, recipe, tmp_path / "o2", "../kacis", "Q8_0", llama, safety=False)
    assert not (tmp_path / "o2").exists()


def test_to_gguf_passes_metadata_file(tmp_path, monkeypatch):
    llama = _fake_llama(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    out = tmp_path / "m.gguf"
    tools = _Tools(monkeypatch)
    meta = {"general.license": "apache-2.0", "general.tags": ["neshter", "safety-ablated"]}
    to_gguf(hf, out, "Q4_K_M", llama, metadata=meta)
    meta_file = Path(str(out) + ".metadata.json.tmp")
    assert tools.calls[0] == [
        sys.executable, str(llama / "convert_hf_to_gguf.py"), str(hf),
        "--outtype", "f16", "--outfile", str(out) + ".f16.tmp", "--metadata", str(meta_file),
    ]
    assert tools.metadata["general.license"] == "apache-2.0"
    assert "safety-ablated" in tools.metadata["general.tags"]
    assert not meta_file.exists() and out.exists()  # çağrıdan sonra silinir


def test_to_gguf_without_metadata_has_no_flag(tmp_path, monkeypatch):
    llama = _fake_llama(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    tools = _Tools(monkeypatch)
    to_gguf(hf, tmp_path / "m.gguf", "Q8_0", llama)
    assert "--metadata" not in tools.calls[0]


def test_to_gguf_cancel_between_convert_and_quantize(tmp_path, monkeypatch):
    llama = _fake_llama(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    ev = threading.Event()
    tools = _Tools(monkeypatch, on_convert=ev.set)
    with pytest.raises(Cancelled):
        to_gguf(hf, tmp_path / "m.gguf", "Q8_0", llama, metadata={"general.license": "mit"}, cancel=ev)
    assert len(tools.calls) == 1  # quantize çalışmadı
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hf", "llama"]  # f16, metadata, çıktı kalmadı


def test_to_gguf_missing_quantize_is_llama_cpp_missing(tmp_path):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    with pytest.raises(LlamaCppMissing) as ei:
        to_gguf(hf, tmp_path / "m.gguf", "Q8_0", llama)
    assert isinstance(ei.value, FileNotFoundError)


def test_gguf_metadata_fields(lm):
    safe = gguf_metadata(lm, recipe_for(lm, [HeadAblation(0, 1), DirectionAblation(torch.randn(64))]), safety=True)
    assert safe["general.license"] == (lm.license or "other")
    assert safe["general.tags"] == ["neshter", "safety-ablated", "abliterated"]
    assert "1 heads" in safe["general.description"] and "1 directions" in safe["general.description"]
    plain = gguf_metadata(lm, recipe_for(lm, [HeadAblation(0, 1)]), safety=False)
    assert plain["general.tags"] == ["neshter"]


def test_gguf_metadata_license_other_when_unknown(lm):
    old = lm.license
    lm.license = None
    try:
        assert gguf_metadata(lm, recipe_for(lm, [HeadAblation(0, 1)]), safety=False)["general.license"] == "other"
    finally:
        lm.license = old


def test_modelfile_has_license(tmp_path):
    mf = write_modelfile(tmp_path / "m.gguf", 'Apache """x"""').read_text(encoding="utf-8")
    assert 'LICENSE """' in mf and mf.count('"""') == 2
    assert "'''x'''" in mf


def test_modelfile_without_license(tmp_path):
    gguf = tmp_path / "m.gguf"
    path = write_modelfile(gguf, None)
    assert path == tmp_path / "Modelfile"
    assert path.read_text(encoding="utf-8") == f"FROM {gguf.resolve()}\n"


def test_modelfile_license_prefers_file_then_lm_then_notices(lm, tmp_path):
    old = lm.license_text
    try:
        (tmp_path / "LICENSE").write_text("dosyadaki", encoding="utf-8")
        lm.license_text = "bellekteki"
        assert modelfile_license(lm, tmp_path) == "dosyadaki"
        (tmp_path / "LICENSE").unlink()
        assert modelfile_license(lm, tmp_path) == "bellekteki"
        lm.license_text = None
        assert modelfile_license(lm, tmp_path) == "\n\n".join(export.license_notices(lm))
    finally:
        lm.license_text = old


def test_export_gguf_cancel_cleans_up(lm, tmp_path, monkeypatch):
    tools = _Tools(monkeypatch)
    before = _state(lm)
    ev = threading.Event()
    ev.set()
    out_dir = tmp_path / "o"
    with pytest.raises(Cancelled):
        export_gguf(lm, recipe_for(lm, [HeadAblation(0, 1)]), out_dir, "m", "Q8_0", _fake_llama(tmp_path),
                    safety=False, cancel=ev)
    assert not out_dir.exists() and tools.calls == []
    after = _state(lm)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_export_gguf_cancel_mid_run_leaves_nothing(lm, tmp_path, monkeypatch):
    ev = threading.Event()
    tools = _Tools(monkeypatch, on_convert=ev.set)  # dönüştürücü bitince iptal gelir
    before = _state(lm)
    out_dir = tmp_path / "o"
    with pytest.raises(Cancelled):
        export_gguf(lm, recipe_for(lm, [HeadAblation(0, 1)]), out_dir, "m", "Q8_0", _fake_llama(tmp_path),
                    safety=False, cancel=ev)
    assert len(tools.calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["llama"]  # ne klasör ne .tmp
    after = _state(lm)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_export_gguf_reports_monotonic_progress(lm, tmp_path, monkeypatch):
    _Tools(monkeypatch)
    seen = []
    export_gguf(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o", "m", "Q8_0", _fake_llama(tmp_path),
                safety=False, progress=lambda p, msg: seen.append(p))
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0) and len(seen) >= 4


def test_save_safetensors_cancel_restores(lm, tmp_path, monkeypatch):
    before = _state(lm)
    ev = threading.Event()
    real = type(lm.model).save_pretrained

    def save_then_cancel(self, *a, **k):
        real(self, *a, **k)
        ev.set()  # kayıt bittikten hemen sonra iptal

    monkeypatch.setattr(type(lm.model), "save_pretrained", save_then_cancel)
    with pytest.raises(Cancelled):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o", cancel=ev)
    assert list(tmp_path.iterdir()) == []  # ne çıktı ne <out_dir>.tmp
    after = _state(lm)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_save_safetensors_preset_cancel_does_nothing(lm, tmp_path):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o", cancel=ev)
    assert list(tmp_path.iterdir()) == []


def test_save_safetensors_reports_progress(lm, tmp_path):
    seen = []
    save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o", progress=lambda p, m: seen.append(p))
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_disk_full_raises_diskfull(lm, tmp_path, monkeypatch):
    _no_space(monkeypatch)
    with pytest.raises(DiskFull) as ei:
        save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o")
    assert isinstance(ei.value, OSError)
    assert not (tmp_path / "o").exists()


def test_export_gguf_disk_full_before_any_write(lm, tmp_path, monkeypatch):
    tools = _Tools(monkeypatch)
    _no_space(monkeypatch)
    with pytest.raises(DiskFull):
        export_gguf(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o", "m", "Q8_0", _fake_llama(tmp_path),
                    safety=False)
    assert not (tmp_path / "o").exists() and tools.calls == []


def test_disk_check_uses_gguf_peak(lm, tmp_path, monkeypatch):
    # safetensors için yeter (1.5x), ama GGUF tepe kullanımı (st + 2n + q·n) sığmaz → DiskFull
    _Tools(monkeypatch)
    llama = _fake_llama(tmp_path)
    _no_space(monkeypatch, free=int(export._estimate_bytes(lm, False) * 1.5))
    with pytest.raises(DiskFull):
        export_gguf(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "o", "m", "Q8_0", llama, safety=False)


def test_estimate_export_bytes_gguf(lm):
    n = sum(p.numel() for p in lm.model.parameters())
    st = export._estimate_bytes(lm, False)
    assert estimate_export_bytes(lm, "safetensors", False) == st
    assert estimate_export_bytes(lm, "gguf_q8_0", False) == int(st + 2 * n + 1.07 * n)
    assert estimate_export_bytes(lm, "gguf_q4_k_m", False) == int(st + 2 * n + 0.6 * n)
    assert QUANT_BYTES_PER_PARAM == {"Q4_K_M": 0.6, "Q8_0": 1.07}
