"""Gerçek model duman testleri: Llama ailesi (SmolLM2-135M-Instruct), gerçek GGUF yükleme, isteğe bağlı Gemma2.

Varsayılan çalıştırmada atlanır: `uv run pytest -m integration -v -s`.
Ortam: HF_HOME (model önbelleği), NESHTER_SMOKE_DIR, NESHTER_SMOKE_RESULTS, NESHTER_SMOKE_THREADS.

- Llama zinciri: baş susturma (kancalı = kayıtlı, greedy token kimlikleri tam aynı) ve yön silme
  (kancalı ~ gömülü ~ kayıtlı-yeniden yüklenen); kayıt dosyası bf16, hesaplama politika dtype'ıdır.
- GGUF: küçük gerçek bir GGUF yüklenir; uzantısız (Ollama blob'u gibi) kopya da sihirli baytla tanınır.
  GGUF *dışa aktarma* bu dosyada yok: llama.cpp klasöründe `convert_hf_to_gguf.py` bulunmuyor (test_real_model.py).
- Gemma2: gated; yalnızca önbellekte varsa çalışır, asla indirilmez.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from smoke_support import (  # noqa: F401  (results ve work_dir fikstürleri)
    CODE_PROMPT,
    N_TOKENS,
    Phase,
    greedy_ids,
    kl_logits,
    logits_of,
    results,
    set_smoke_threads,
    work_dir,
)

from neshter.compare import generate_pair
from neshter.concepts.hunt import hunt
from neshter.concepts.model import load_preset
from neshter.export import bake, model_card, save_dtype, save_safetensors, verify_source_untouched
from neshter.hardware import cpu_has_native_bf16
from neshter.interventions import DirectionAblation, HeadAblation
from neshter.loader import is_gguf, load_model
from neshter.metrics import load_neutral, real_effect
from neshter.recipe import from_interventions

pytestmark = pytest.mark.integration

LLAMA_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"  # llama mimarisi, Apache-2.0, ~270 MB
LLAMA_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
# Küçük gerçek GGUF (<= 200 MB). Qwen/Qwen2.5-0.5B-Instruct-GGUF Q8_0 ~676 MB olduğundan SmolLM2 Q8_0 (145 MB) seçildi.
GGUF_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
GGUF_REV = "09816acd5d99df7be770d85ea30822623dab342c"
GGUF_FILE = "SmolLM2-135M-Instruct-Q8_0.gguf"
GEMMA_ID = "google/gemma-2-2b-it"  # gated, ~5 GB: asla indirilmez
TOP_HEADS = 6
LLAMA_FILES = ["*.json", "*.txt", "*.safetensors", "README.md", "LICENSE*"]  # onnx/ klasörü hariç


# --------------------------------------------------------------------------- Llama (SmolLM2)


@pytest.fixture(scope="module")
def llama_dir() -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(LLAMA_ID, revision=LLAMA_REV, allow_patterns=LLAMA_FILES, local_files_only=True))
    except Exception:
        return Path(snapshot_download(LLAMA_ID, revision=LLAMA_REV, allow_patterns=LLAMA_FILES))


@pytest.fixture(scope="module")
def llama(llama_dir, results):
    set_smoke_threads()
    with Phase(results["timings"], "llama_load"):
        lm = load_model(llama_dir, device="cpu")
    assert lm.arch.family == "llama"
    policy = torch.bfloat16 if cpu_has_native_bf16() else torch.float32
    assert next(lm.model.parameters()).dtype == policy
    assert lm.license == "apache-2.0"
    results["checks"]["llama_license"] = lm.license
    results["checks"]["llama_arch"] = {
        "layers": lm.arch.n_layers, "heads": lm.arch.n_heads, "kv_heads": lm.arch.n_kv_heads,
        "d_model": lm.arch.d_model, "tied": bool(getattr(lm.model.config, "tie_word_embeddings", False)),
    }
    return lm


def test_llama_smollm2_chain(llama, results, work_dir):
    lm = llama
    # av (activation_diff, CPU varsayılanı): ölçülmüş baş listesi dolu
    with Phase(results["timings"], "llama_hunt_activation_diff"):
        res = hunt(lm, load_preset("code"), "activation_diff")
    assert res.measured
    top = [(l, h) for l, h, _ in res.heads[:TOP_HEADS]]
    heads = [HeadAblation(l, h) for l, h in top]
    results["checks"]["llama_top_heads"] = top
    results["checks"]["llama_val_accuracy"] = res.val_accuracy
    results["checks"]["llama_val_ci"] = list(res.val_ci)

    # gerçek etki alanları (p < 0,05 şartı yok: yalnızca kaydedilir)
    with Phase(results["timings"], "llama_effect"):
        eff = real_effect(lm, heads, load_neutral(), seed=0)
    assert eff.n_random == 30 and eff.n_prompts > 0
    assert 0.0 <= eff.percentile <= 100.0 and 0.0 < eff.p_value <= 1.0
    results["checks"]["llama_effect"] = {
        "kl": eff.kl, "random_kl": eff.random_kl, "percentile": eff.percentile, "p_value": eff.p_value,
        "layer_matched": eff.layer_matched, "n_random": eff.n_random,
    }

    # kart: Llama 3.x bildirimi yok (SmolLM2 Llama 3 değil)
    recipe = from_interventions(heads, lm, "code", "activation_diff", 0)
    card = model_card(lm, recipe, None, False)
    assert "Built with Llama" not in card
    results["checks"]["llama_card_has_built_with_llama"] = "Built with Llama" in card

    # kayıt -> yeniden yükle -> 40 greedy token kancalıyla aynı
    hooked_ids = greedy_ids(lm, heads)
    out = work_dir / "llama_head_cut"
    with Phase(results["timings"], "llama_save_heads"):
        save_safetensors(lm, recipe, out)
    assert verify_source_untouched(lm)
    re_lm = load_model(out, device="cpu")
    try:
        saved_ids = greedy_ids(re_lm, [])
        first_diff = next((i for i, (a, b) in enumerate(zip(hooked_ids, saved_ids)) if a != b), None)
        results["checks"]["llama_head_reload_token_match"] = saved_ids == hooked_ids
        results["checks"]["llama_head_reload_first_diff"] = first_diff
        assert len(hooked_ids) == N_TOKENS
        assert saved_ids == hooked_ids, f"ilk fark token {first_diff}"
    finally:
        del re_lm


def test_llama_smollm2_direction_chain(llama, results, work_dir):
    lm = llama
    res = hunt(lm, load_preset("code"), "direction")
    iv = [DirectionAblation(res.direction)]
    results["checks"]["llama_direction_layer"] = res.direction_layer

    # yön varsa gömülü kip otomatik seçilir
    with Phase(results["timings"], "llama_generate_pair_baked"):
        orig, baked_text, mode = generate_pair(lm, CODE_PROMPT, iv, max_new_tokens=N_TOKENS)
    assert mode == "baked"
    results["checks"]["llama_pair_mode"] = mode
    results["checks"]["llama_pair_original"] = orig[:200]
    results["checks"]["llama_pair_baked"] = baked_text[:200]

    texts = load_neutral()[:4]
    base = [logits_of(lm, [], t) for t in texts]
    hooked = [logits_of(lm, iv, t) for t in texts]

    # kanca == gömme (yuvarlamasız): izdüşüm ağırlıklara taşınınca aynı hesap
    undo = bake(lm, iv)
    try:
        exact = [logits_of(lm, [], t) for t in texts]
    finally:
        undo()
    # kayıt dtype'ına yuvarlanmış gömme
    undo = bake(lm, iv, round_to=save_dtype(lm))
    try:
        baked = [logits_of(lm, [], t) for t in texts]
    finally:
        undo()
    assert all(torch.equal(a, logits_of(lm, [], t)) for a, t in zip(base, texts))  # undo orijinali tam geri getirdi

    recipe = from_interventions(iv, lm, "code", "direction", 0)
    out = work_dir / "llama_dir_cut"
    with Phase(results["timings"], "llama_save_direction"):
        save_safetensors(lm, recipe, out)
    assert verify_source_untouched(lm)
    re_lm = load_model(out, device="cpu")
    try:
        reloaded = [logits_of(re_lm, [], t) for t in texts]
    finally:
        del re_lm

    kl_hook_exact = [kl_logits(h, e) for h, e in zip(hooked, exact)]
    kl_baked_reloaded = [kl_logits(b, r) for b, r in zip(baked, reloaded)]
    kl_hook_reloaded = [kl_logits(h, r) for h, r in zip(hooked, reloaded)]
    kl_cut = [kl_logits(b, h) for b, h in zip(base, hooked)]
    results["checks"]["llama_direction"] = {
        "kl_orig_vs_hooked": kl_cut,
        "kl_hooked_vs_bake_exact": kl_hook_exact,
        "kl_baked_vs_reloaded": kl_baked_reloaded,
        "kl_hooked_vs_reloaded": kl_hook_reloaded,
    }
    print(f"[smoke] llama yön: {results['checks']['llama_direction']}", flush=True)
    assert max(kl_hook_exact) < 1e-4  # kanca == gömme
    assert max(kl_baked_reloaded) < 1e-3  # bellekteki gömülü == diskten yeniden yüklenen


# --------------------------------------------------------------------------- gerçek GGUF


@pytest.fixture(scope="module")
def gguf_path() -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REV, local_files_only=True))
    except Exception:
        return Path(hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REV))


def test_gguf_real_load_and_suffixless_blob(gguf_path, results, tmp_path):
    set_smoke_threads()
    size_mb = gguf_path.stat().st_size / 2**20
    assert size_mb <= 200
    assert is_gguf(gguf_path)
    with Phase(results["timings"], "gguf_load"):
        lm = load_model(gguf_path, device="cpu")
    assert lm.arch.family == "llama"
    ids = greedy_ids(lm, [], n_tokens=10)
    text = lm.tokenizer.decode(ids, skip_special_tokens=True)
    assert text.strip(), f"boş GGUF cevabı: {ids}"
    results["checks"]["gguf"] = {
        "file": GGUF_FILE, "size_mb": round(size_mb, 1), "family": lm.arch.family, "license": lm.license,
        "dtype": str(next(lm.model.parameters()).dtype), "greedy10": text[:80],
    }
    del lm

    # Ollama blob'u gibi uzantısız kopya: sihirli baytla tanınır, name görünen ad olur
    blob = tmp_path / "blobs" / ("sha256-" + "0" * 64)
    blob.parent.mkdir()
    shutil.copyfile(gguf_path, blob)
    assert blob.suffix == "" and is_gguf(blob)
    with Phase(results["timings"], "gguf_blob_load"):
        lm2 = load_model(blob, name="smollm2:135m", device="cpu")
    assert lm2.name == "smollm2:135m" and lm2.arch.family == "llama"
    ids2 = greedy_ids(lm2, [], n_tokens=10)
    results["checks"]["gguf_blob"] = {"load_ok": True, "same_greedy_as_file": ids2 == ids, "license": lm2.license}
    assert lm2.tokenizer.decode(ids2, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- Gemma2 (isteğe bağlı)


def test_gemma2_real_optional(results, work_dir):
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(GEMMA_ID, local_files_only=True)
    except Exception:
        results["checks"]["gemma2"] = "SKIPPED"
        pytest.skip("Gemma2 gerçek modeli önbellekte yok ya da erişim izni yok (gated); atlandı.")
    set_smoke_threads()
    lm = load_model(Path(path), device="cpu")
    assert lm.arch.family == "gemma2"
    res = hunt(lm, load_preset("tone"), "direction")
    iv = [DirectionAblation(res.direction)]
    texts = load_neutral()[:4]
    hooked = [logits_of(lm, iv, t) for t in texts]
    undo = bake(lm, iv, round_to=save_dtype(lm))
    try:
        baked = [logits_of(lm, [], t) for t in texts]
    finally:
        undo()
    recipe = from_interventions(iv, lm, "tone", "direction", 0)
    out = work_dir / "gemma_dir_cut"
    save_safetensors(lm, recipe, out)
    re_lm = load_model(out, device="cpu")
    try:
        reloaded = [logits_of(re_lm, [], t) for t in texts]
    finally:
        del re_lm
    kl_hb = [kl_logits(h, b) for h, b in zip(hooked, baked)]
    kl_br = [kl_logits(b, r) for b, r in zip(baked, reloaded)]
    results["checks"]["gemma2"] = {"kl_hooked_vs_baked": kl_hb, "kl_baked_vs_reloaded": kl_br}
    assert max(kl_hb) < 1e-3  # kanca == gömme (bf16 yuvarlaması dahil)
    assert max(kl_br) < 1e-3
