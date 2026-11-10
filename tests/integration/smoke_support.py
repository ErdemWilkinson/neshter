"""Gerçek model duman testlerinin ortak yardımcıları (süre/RSS ölçümü, sonuç JSON'u, greedy/KL)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import psutil
import pytest
import torch

from neshter.compare import _encode
from neshter.interventions import apply

CODE_PROMPT = "Write a Python function that reverses a string."
N_TOKENS = 40  # greedy karşılaştırma uzunluğu


def set_smoke_threads() -> None:
    """İçe aktarmada değil: birim testleri etkilenmesin."""
    torch.set_num_threads(int(os.environ.get("NESHTER_SMOKE_THREADS", "4")))


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


@pytest.fixture(scope="session")
def results():
    """Oturum boyunca tek sonuç sözlüğü; NESHTER_SMOKE_RESULTS verilmişse sonunda JSON yazılır."""
    data: dict = {"timings": {}, "checks": {}}
    yield data
    out = os.environ.get("NESHTER_SMOKE_RESULTS")
    if out:
        proc = psutil.Process()
        peak = getattr(proc.memory_info(), "peak_wset", None)
        data["process_peak_wset_mb"] = round(peak / 2**20) if peak else None
        Path(out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="session")
def work_dir(tmp_path_factory):
    base = os.environ.get("NESHTER_SMOKE_DIR")
    if base:
        d = Path(base)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path_factory.mktemp("smoke")


def greedy_ids(lm, interventions, n_tokens: int = N_TOKENS, prompt: str = CODE_PROMPT) -> list[int]:
    """Tek satırlık (yığınsız) greedy üretim; kancalı ve kayıtlı modeller aynı biçimde karşılaştırılsın."""
    enc = _encode(lm, prompt, 1)
    with apply(lm, interventions):
        torch.manual_seed(0)
        with torch.no_grad():
            out = lm.model.generate(
                **enc, do_sample=False, max_new_tokens=n_tokens, pad_token_id=lm.tokenizer.pad_token_id
            )
    return out[0, enc["input_ids"].shape[1] :].tolist()


def logits_of(lm, interventions, text: str) -> torch.Tensor:
    enc = lm.tokenizer(text, return_tensors="pt")
    with apply(lm, interventions), torch.no_grad():
        return lm.model(**enc, use_cache=False).logits[0].to(torch.float32)


def kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) konum ortalaması (logit'lerden)."""
    lp, lq = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()
