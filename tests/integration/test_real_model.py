"""Gerçek model duman testi (Qwen2.5-0.5B-Instruct, CPU, bf16).

Varsayılan çalıştırmada atlanır: `uv run pytest -m integration -v -s`.
Ortam: HF_HOME (model önbelleği), NESHTER_SMOKE_DIR (kayıt klasörü; C: yerine D: için),
NESHTER_SMOKE_RESULTS (süre/bellek/sayı sonuçlarının yazılacağı JSON).

Zincir: yükle -> hunt(code, attribution) -> ilk 6 head gerçek etkisi > rastgele -> orijinal kod
isteğinde detect_code -> save_safetensors -> yeniden yükle -> greedy cevap kancalı cevapla aynı.

bf16 notu: baş susturma kayıtta o_proj sütunlarını sıfırlamaktır; kancadaki sıfırlamayla bit düzeyinde
aynı hesabı verir. Bu yüzden kayıt/yükleme turu greedy token kimliklerini TAM korumalıdır. Buna karşılık
yön silme ağırlıkları bf16'ya yuvarlar (kancada ise aktivasyon bf16'da izdüşürülür); sonuç yalnızca
yaklaşık aynıdır, bu yüzden orada KL ile göreli tolerans denetlenir.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import psutil
import pytest
import torch

from neshter.compare import _encode, generate_pair
from neshter.concepts.detectors import detect_code
from neshter.concepts.hunt import hunt
from neshter.concepts.model import load_preset
from neshter.export import find_llama_cpp, save_safetensors, to_gguf, verify_source_untouched
from neshter.interventions import DirectionAblation, HeadAblation, apply
from neshter.loader import is_gguf, load_model
from neshter.metrics import kl_effect, load_neutral, real_effect
from neshter.recipe import from_interventions
from neshter.scan import default_roots, scan_ollama

pytestmark = pytest.mark.integration

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CODE_PROMPT = "Write a Python function that reverses a string."
N_TOKENS = 40  # greedy karşılaştırma uzunluğu
TOP_HEADS = 6


# --------------------------------------------------------------------------- ölçüm yardımcıları


class Phase:
    """Bir aşamanın süresini ve tepe RSS'ini (örnekleyerek) ölçer."""

    def __init__(self, store: dict, name: str):
        self.store, self.name = store, name
        self._proc = psutil.Process()

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                self.peak = max(self.peak, self._proc.memory_info().rss)
                time.sleep(0.05)

        self._t = threading.Thread(target=sample, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.t0
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)
        self.store[self.name] = {"seconds": round(elapsed, 2), "peak_rss_mb": round(self.peak / 2**20)}
        print(f"[smoke] {self.name}: {elapsed:.1f} s, tepe RSS {self.peak / 2**20:.0f} MB", flush=True)


@pytest.fixture(scope="module")
def results():
    data: dict = {"timings": {}, "checks": {}}
    yield data
    out = os.environ.get("NESHTER_SMOKE_RESULTS")
    if out:
        proc = psutil.Process()
        peak = getattr(proc.memory_info(), "peak_wset", None)
        data["process_peak_wset_mb"] = round(peak / 2**20) if peak else None
        Path(out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    base = os.environ.get("NESHTER_SMOKE_DIR")
    if base:
        d = Path(base)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path_factory.mktemp("smoke")


@pytest.fixture(scope="module")
def model_dir() -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(MODEL_ID, local_files_only=True))
    except Exception:
        return Path(snapshot_download(MODEL_ID))


@pytest.fixture(scope="module")
def lm(model_dir, results):
    torch.set_num_threads(int(os.environ.get("NESHTER_SMOKE_THREADS", "4")))  # içe aktarmada değil: birim testleri etkilenmesin
    with Phase(results["timings"], "load"):
        loaded = load_model(model_dir, device="cpu")  # varsayılan bf16
    assert loaded.arch.family == "qwen2"
    assert next(loaded.model.parameters()).dtype == torch.bfloat16
    return loaded


@pytest.fixture(scope="module")
def code_hunt(lm, results):
    with Phase(results["timings"], "hunt_attribution"):
        res = hunt(lm, load_preset("code"), "attribution")
    return res


def _greedy_ids(lm, interventions, n_tokens: int = N_TOKENS) -> list[int]:
    """Tek satırlık (yığınsız) greedy üretim; kancalı ve kayıtlı modeller aynı biçimde karşılaştırılsın."""
    enc = _encode(lm, CODE_PROMPT, 1)
    with apply(lm, interventions):
        torch.manual_seed(0)
        with torch.no_grad():
            out = lm.model.generate(
                **enc, do_sample=False, max_new_tokens=n_tokens, pad_token_id=lm.tokenizer.pad_token_id
            )
    return out[0, enc["input_ids"].shape[1] :].tolist()


def _logits(lm, interventions, text: str) -> torch.Tensor:
    enc = lm.tokenizer(text, return_tensors="pt")
    with apply(lm, interventions), torch.no_grad():
        return lm.model(**enc, use_cache=False).logits[0].to(torch.float32)


# --------------------------------------------------------------------------- ana zincir


def test_head_chain(lm, code_hunt, results, work_dir, model_dir):
    top = [(l, h) for l, h, _ in code_hunt.heads[:TOP_HEADS]]
    heads = [HeadAblation(l, h) for l, h in top]
    results["checks"]["top_heads"] = top
    results["checks"]["layer_scores_argmax"] = max(range(len(code_hunt.layer_scores)), key=code_hunt.layer_scores.__getitem__)
    results["checks"]["val_accuracy"] = code_hunt.val_accuracy
    print(f"[smoke] ilk {TOP_HEADS} head: {top}", flush=True)

    # 1) gerçek etki > rastgele kontrol
    prompts = load_neutral()
    with Phase(results["timings"], "effect"):
        eff = real_effect(lm, heads, prompts, n_random=3, seed=0)
    results["checks"]["effect"] = {
        "kl": eff.kl, "kl_std": eff.kl_std, "random_kl": eff.random_kl, "n_prompts": eff.n_prompts
    }
    print(f"[smoke] KL={eff.kl:.5f} rastgele={eff.random_kl:.5f} n={eff.n_prompts}", flush=True)
    assert eff.kl > eff.random_kl

    # 2) orijinal kod isteğinde kod çıkar; kancalı cevap da üretilir
    with Phase(results["timings"], "generate_pair"):
        orig, hooked, mode = generate_pair(lm, CODE_PROMPT, heads, max_new_tokens=N_TOKENS * 2)
    results["checks"]["original_answer"] = orig[:300]
    results["checks"]["hooked_answer"] = hooked[:300]
    results["checks"]["pair_mode"] = mode
    assert detect_code(orig), f"orijinal cevap kod değil: {orig!r}"

    # 3) kayıt -> yeniden yükle -> greedy cevap kancalıyla aynı
    hooked_ids = _greedy_ids(lm, heads)
    recipe = from_interventions(heads, lm, "code", "attribution", 0)
    out = work_dir / "head_cut"
    with Phase(results["timings"], "save"):
        save_safetensors(lm, recipe, out)
    assert verify_source_untouched(lm)
    saved_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**30
    results["checks"]["saved_gb"] = round(saved_gb, 3)

    with Phase(results["timings"], "reload"):
        re_lm = load_model(out, device="cpu")  # aynı dtype: bf16
    try:
        assert next(re_lm.model.parameters()).dtype == next(lm.model.parameters()).dtype
        saved_ids = _greedy_ids(re_lm, [])
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
            base = _logits(lm, [], t)
            hooked = _logits(lm, [ablation], t)
            baked = _logits(re_lm, [], t)
            lp = lambda x: x.log_softmax(-1)
            kl_cut = (lp(base).exp() * (lp(base) - lp(hooked))).sum(-1).mean().item()
            kl_rt = (lp(hooked).exp() * (lp(hooked) - lp(baked))).sum(-1).mean().item()
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
