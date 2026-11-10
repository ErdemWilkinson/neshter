"""Gerçek model duman testi (Qwen2.5-0.5B-Instruct, CPU, politika dtype'ı).

Varsayılan çalıştırmada atlanır: `uv run pytest -m integration -v -s`.
Ortam: HF_HOME (model önbelleği), NESHTER_SMOKE_DIR (kayıt klasörü; C: yerine D: için),
NESHTER_SMOKE_RESULTS (süre/bellek/sayı sonuçlarının yazılacağı JSON).

Zincir: yükle -> hunt(code, attribution) -> ilk 6 head gerçek etkisi (30 katman eşleşmeli kontrol; p ve
yüzdelik dilim yalnızca kaydedilir) -> orijinal kod isteğinde detect_code -> save_safetensors -> yeniden
yükle -> greedy cevap kancalı cevapla aynı. Ek: fp32 süre ölçümü (v0.1 bf16 ile karşılaştırma).

dtype politikası: yerel bf16'sı olmayan CPU'da RAM yeterse float32, aksi halde bf16 (`load_model`).
Kayıt dosyası her durumda bf16'dır.

bf16 notu: baş susturma kayıtta o_proj sütunlarını sıfırlamaktır; kancadaki sıfırlamayla bit düzeyinde
aynı hesabı verir. Bu yüzden kayıt/yükleme turu greedy token kimliklerini TAM korumalıdır. Buna karşılık
yön silme ağırlıkları bf16'ya yuvarlar (kancada ise aktivasyon bf16'da izdüşürülür); sonuç yalnızca
yaklaşık aynıdır, bu yüzden orada KL ile göreli tolerans denetlenir.
"""

from __future__ import annotations

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
from neshter.concepts.detectors import detect_code
from neshter.concepts.hunt import hunt
from neshter.concepts.model import load_preset
from neshter.export import find_llama_cpp, save_safetensors, to_gguf, verify_source_untouched
from neshter.hardware import cpu_has_native_bf16
from neshter.interventions import DirectionAblation, HeadAblation
from neshter.loader import is_gguf, load_model
from neshter.metrics import load_neutral, real_effect
from neshter.recipe import from_interventions
from neshter.scan import default_roots, scan_ollama

pytestmark = pytest.mark.integration

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_HEADS = 6
V01_HUNT_SECONDS = 684  # v0.1: Qwen2.5-0.5B, attribution av, bf16 CPU
V01_EFFECT_SECONDS = 240  # v0.1: gerçek etki (40 nötr istek, 3 rastgele çekiliş), bf16 CPU


# --------------------------------------------------------------------------- fikstürler


@pytest.fixture(scope="module")
def model_dir() -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(MODEL_ID, local_files_only=True))
    except Exception:
        return Path(snapshot_download(MODEL_ID))


@pytest.fixture(scope="module")
def lm(model_dir, results):
    set_smoke_threads()
    with Phase(results["timings"], "load"):
        loaded = load_model(model_dir, device="cpu")  # dtype politikası
    assert loaded.arch.family == "qwen2"
    # Yerel bf16'sı olmayan CPU'da float32 yüklenir (0,5B için RAM yeter); aksi halde bf16.
    policy = torch.bfloat16 if cpu_has_native_bf16() else torch.float32
    assert next(loaded.model.parameters()).dtype == policy
    results["checks"]["compute_dtype"] = str(policy)
    results["checks"]["native_bf16"] = cpu_has_native_bf16()
    results["checks"]["qwen_license"] = loaded.license
    assert loaded.license == "apache-2.0"  # Qwen2.5-0.5B-Instruct: README/LICENSE Apache-2.0
    assert loaded.license_text and "Apache License" in loaded.license_text
    return loaded


@pytest.fixture(scope="module")
def code_hunt(lm, results):
    with Phase(results["timings"], "hunt_attribution"):
        res = hunt(lm, load_preset("code"), "attribution")
    return res


# --------------------------------------------------------------------------- ana zincir


def test_head_chain(lm, code_hunt, results, work_dir, model_dir):
    top = [(l, h) for l, h, _ in code_hunt.heads[:TOP_HEADS]]
    heads = [HeadAblation(l, h) for l, h in top]
    results["checks"]["top_heads"] = top
    results["checks"]["layer_scores_argmax"] = max(range(len(code_hunt.layer_scores)), key=code_hunt.layer_scores.__getitem__)
    results["checks"]["val_accuracy"] = code_hunt.val_accuracy
    results["checks"]["val_ci"] = list(code_hunt.val_ci)
    results["checks"]["ref_layer"] = code_hunt.ref_layer
    results["checks"]["measured_top"] = [
        (m.layer, m.head, None if m.signal_drop is None else round(m.signal_drop, 4), round(m.neutral_kl, 5)) for m in code_hunt.measured[:6]
    ]
    print(f"[smoke] ilk {TOP_HEADS} head: {top}", flush=True)

    # 1) gerçek etki > rastgele kontrol
    prompts = load_neutral()
    with Phase(results["timings"], "effect"):
        eff = real_effect(lm, heads, prompts, seed=0)  # varsayılan: 30 katman eşleşmeli kontrol
    results["checks"]["effect"] = {
        "kl": eff.kl, "kl_std": eff.kl_std, "random_kl": eff.random_kl, "n_prompts": eff.n_prompts,
        "n_random": eff.n_random, "percentile": eff.percentile, "p_value": eff.p_value,
        "layer_matched": eff.layer_matched, "random_p05": eff.random_p05, "random_p95": eff.random_p95,
    }
    print(
        f"[smoke] KL={eff.kl:.5f} rastgele={eff.random_kl:.5f} n={eff.n_prompts} "
        f"yüzdelik={eff.percentile:.0f} p={eff.p_value:.3f} eşleşmeli={eff.layer_matched}",
        flush=True,
    )
    assert eff.n_random == 30  # p < 0,05 şartı yok: yalnızca kaydedilir
    assert eff.kl > eff.random_kl

    # 2) orijinal kod isteğinde kod çıkar; kancalı cevap da üretilir
    with Phase(results["timings"], "generate_pair"):
        orig, hooked, mode = generate_pair(lm, CODE_PROMPT, heads, max_new_tokens=N_TOKENS * 2)
    results["checks"]["original_answer"] = orig[:300]
    results["checks"]["hooked_answer"] = hooked[:300]
    results["checks"]["pair_mode"] = mode
    assert detect_code(orig), f"orijinal cevap kod değil: {orig!r}"

    # 3) kayıt -> yeniden yükle -> greedy cevap kancalıyla aynı
    hooked_ids = greedy_ids(lm, heads)
    recipe = from_interventions(heads, lm, "code", "attribution", 0)
    out = work_dir / "head_cut"
    with Phase(results["timings"], "save"):
        save_safetensors(lm, recipe, out)
    assert verify_source_untouched(lm)
    saved_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**30
    results["checks"]["saved_gb"] = round(saved_gb, 3)

    with Phase(results["timings"], "reload"):
        re_lm = load_model(out, device="cpu")  # aynı politika: aynı dtype
    try:
        assert next(re_lm.model.parameters()).dtype == next(lm.model.parameters()).dtype
        saved_ids = greedy_ids(re_lm, [])
        results["checks"]["head_reload_token_match"] = saved_ids == hooked_ids
        first_diff = next((i for i, (a, b) in enumerate(zip(hooked_ids, saved_ids)) if a != b), None)
        results["checks"]["head_reload_first_diff"] = first_diff
        results["checks"]["head_reload_text"] = lm.tokenizer.decode(saved_ids, skip_special_tokens=True)[:300]
        assert saved_ids == hooked_ids, f"ilk fark token {first_diff}"
        # Kullanıcıya görünen metin de aynı (pair cevabıyla, aynı uzunlukta kesilmiş hâliyle)
        assert lm.tokenizer.decode(saved_ids, skip_special_tokens=True) == lm.tokenizer.decode(
            hooked_ids, skip_special_tokens=True
        )
    finally:
        del re_lm


# --------------------------------------------------------------------------- fp32 süreleri (v0.1 ile)


def test_qwen_fp32_timings(lm, code_hunt, results):
    if not cpu_has_native_bf16():
        assert next(lm.model.parameters()).dtype == torch.float32
    # code_hunt hunt(code, "attribution") süresini zaten ölçtü: ikinci kez çalıştırılmaz.
    secs = results["timings"]["hunt_attribution"]["seconds"]
    results["checks"]["hunt_attribution_vs_v01"] = {
        "seconds": secs, "v01_seconds": V01_HUNT_SECONDS, "speedup": round(V01_HUNT_SECONDS / secs, 2)
    }
    with Phase(results["timings"], "hunt_activation_diff"):
        res = hunt(lm, load_preset("code"), "activation_diff")
    assert res.measured
    top = [HeadAblation(l, h) for l, h, _ in code_hunt.heads[:TOP_HEADS]]
    # v0.1 ile aynı iş: 40 nötr istek, 3 rastgele çekiliş (bire bir karşılaştırma)
    with Phase(results["timings"], "effect_n3"):
        eff3 = real_effect(lm, top, load_neutral(), n_random=3, seed=0)
    assert eff3.n_random == 3
    e = results["timings"]["effect_n3"]["seconds"]
    results["checks"]["effect_vs_v01"] = {
        "seconds": e, "v01_seconds": V01_EFFECT_SECONDS, "speedup": round(V01_EFFECT_SECONDS / e, 2)
    }
    # Sert eşik yok (V01_HUNT_SECONDS / 3 hedefi doğrulanmamış): sayılar raporlanır, karar denetleyicide.


# --------------------------------------------------------------------------- (a) yön silme


def test_direction_cut(lm, results, work_dir):
    """Refusal olmayan kavram (tone) ile yön silme: kayıtlı model ~ kancalı model (bf16 toleransı)."""
    concept = load_preset("tone")
    res = hunt(lm, concept, "direction")
    ablation = DirectionAblation(res.direction)
    results["checks"]["direction_layer"] = res.direction_layer

    recipe = from_interventions([ablation], lm, "tone", "direction", 0)
    out = work_dir / "dir_cut"
    with Phase(results["timings"], "save_direction"):
        save_safetensors(lm, recipe, out)
    assert verify_source_untouched(lm)

    texts = [
        "The capital of France is Paris, and it is known for",
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
        "Merhaba, bugün hava çok güzel ve ben yürüyüşe çıkmak istiyorum.",
    ]
    re_lm = load_model(out, device="cpu")
    try:
        stats = []
        for t in texts:
            base = logits_of(lm, [], t)
            hooked = logits_of(lm, [ablation], t)
            baked = logits_of(re_lm, [], t)
            kl_cut = kl_logits(base, hooked)
            kl_rt = kl_logits(hooked, baked)
            top1 = (hooked.argmax(-1) == baked.argmax(-1)).float().mean().item()
            stats.append({"kl_orig_vs_hooked": kl_cut, "kl_hooked_vs_baked": kl_rt, "top1_agree": top1,
                          "max_abs_logit_diff": (hooked - baked).abs().max().item()})
        results["checks"]["direction_roundtrip"] = stats
        print(f"[smoke] yön silme: {stats}", flush=True)
        for s in stats:
            # kayıt turu hatası (bf16 yuvarlama) kesmenin kendi etkisinden çok küçük ve token tercihleri
            # büyük ölçüde korunmuş olmalı
            assert s["kl_hooked_vs_baked"] < 0.05
            assert s["top1_agree"] > 0.8
    finally:
        del re_lm


# --------------------------------------------------------------------------- (b) GGUF, (c) Ollama


def test_gguf_converter_available(results, work_dir, lm):
    llama = find_llama_cpp()
    results["checks"]["llama_cpp_dir"] = str(llama) if llama else None
    if llama is None:
        pytest.skip("find_llama_cpp() bulunamadı")
    converter = llama / "convert_hf_to_gguf.py"
    if not converter.is_file():
        results["checks"]["gguf"] = "SKIPPED: convert_hf_to_gguf.py yok"
        # to_gguf, eksik dönüştürücüyü açık bir hatayla bildirmeli (denenebilir yan etki yok)
        with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
            to_gguf(work_dir / "head_cut", work_dir / "x.gguf", "Q8_0", llama)
        pytest.skip(f"{converter} yok; llama.cpp kaynağı indirilmez")
    out = to_gguf(work_dir / "head_cut", work_dir / "head_cut.gguf", "Q8_0", llama)
    assert is_gguf(out)
    results["checks"]["gguf"] = f"PASS: {out.stat().st_size} bayt"


def test_ollama_scan(results):
    """Ollama blob'larını diskten tarar (sunucu gerekmez); yalnızca okur."""
    entries = scan_ollama(default_roots()["ollama"][0])
    results["checks"]["ollama_entries"] = [(e.name, e.size_bytes, is_gguf(e.path)) for e in entries]
    if not entries:
        pytest.skip("Ollama manifestleriyle eşleşen blob yok")
    for e in entries:
        assert e.fmt == "gguf" and is_gguf(e.path), e.name
