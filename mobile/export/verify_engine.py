"""Motorun (PyTorch ve isteğe bağlı ONNX) Neşter'in kendi ölçümleriyle aynı sayıları verdiğini doğrular.

  python mobile/export/verify_engine.py MODEL_KLASORU [ONNX_DOSYASI]
Karşılaştırılanlar: head gücü (head_output_power), dikkat deseni, logit lens, kesimli KL (kl_effect).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neshter.activity import head_output_power  # noqa: E402
from neshter.graph import attention_patterns, logit_lens  # noqa: E402
from neshter.interventions import HeadAblation  # noqa: E402
from neshter.loader import load_model  # noqa: E402
from neshter.metrics import kl_effect  # noqa: E402

from export_qwen2 import NAMES_IN, Qwen2Engine, empty_past  # noqa: E402

PROMPT = "İstanbul Türkiye'nin en kalabalık şehri ve"
CUT = [(0, 1), (2, 3), (10, 5)]


def run_torch(engine, ids, mask, keep, lens_hidden, target, past=None):
    pk, pv = past if past is not None else empty_past(engine)
    with torch.no_grad():
        return [t.numpy() for t in engine(torch.tensor([ids]), mask, torch.tensor(keep), lens_hidden, torch.tensor(target),
                                          torch.as_tensor(pk), torch.as_tensor(pv))]


def greedy(engine_fn, ids, mask, n, M, cached):
    """n token açgözlü üretim: önbellekli (her adım tek token) ya da her adımda baştan."""
    ids = list(ids)
    out = []
    past = None
    step = ids
    for _ in range(n):
        r = engine_fn(step, mask, 1, torch.zeros(0, M), 0, past) if cached else engine_fn(ids, mask, 1, torch.zeros(0, M), 0)
        nxt = int(np.argmax(r[0][-1]))
        out.append(nxt)
        ids.append(nxt)
        if cached:
            past = (r[8], r[9])
            step = [nxt]
    return out


def main() -> int:
    model_dir = Path(sys.argv[1])
    onnx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    lm = load_model(model_dir, device="cpu", dtype=torch.float32)
    engine = Qwen2Engine(lm.model).eval()
    ids = lm.tokenizer(PROMPT)["input_ids"]
    T = len(ids)
    ones = torch.ones(engine.L, engine.H)
    cut = ones.clone()
    for l, h in CUT:
        cut[l, h] = 0

    def runner(engine_fn):
        out = {}
        base = engine_fn(ids, ones, T, torch.zeros(0, engine.M), 0)
        out["power"] = base[1]
        out["attn3"] = base[2][3]
        resid = torch.tensor(base[3][:, -1])  # [L+1, d], son konum
        final = int(np.argmax(base[0][-1]))
        lens = engine_fn(ids[:1], ones, 1, resid, final)
        out["lens_id"], out["lens_p_target"], out["lens_rank"] = lens[5], lens[6], lens[7]
        cutrun = engine_fn(ids, cut, min(16, T), torch.zeros(0, engine.M), 0)
        logp, logq = base[0][-min(16, T):], cutrun[0]
        out["kl"] = float((np.exp(logp) * (logp - logq)).sum(-1).mean())
        return out

    def check(name, a, b, tol):
        err = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
        ok = err <= tol
        print(f"{'OK ' if ok else 'HATA'} {name}: en büyük fark {err:.3g} (sınır {tol})")
        return ok

    ref_power = head_output_power(lm, PROMPT).numpy()
    ref_attn = np.array(attention_patterns(lm, PROMPT, 3)["patterns"])
    ref_lens = logit_lens(lm, PROMPT, -1, top_k=3)
    ref_kl = kl_effect(lm, [PROMPT], [HeadAblation(l, h) for l, h in CUT])[0]

    fns = {"torch": lambda *a: run_torch(engine, *a)}
    results = {"torch": runner(fns["torch"])}
    if onnx_path is not None:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        def run_onnx(ids_, mask, keep, lens_hidden, target, past=None):
            pk, pv = past if past is not None else [t.numpy() for t in empty_past(engine)]
            feed = dict(zip(NAMES_IN, [np.array([ids_], dtype=np.int64), mask.numpy().astype(np.float32),
                                       np.array(keep, dtype=np.int64), lens_hidden.numpy().astype(np.float32),
                                       np.array(target, dtype=np.int64), np.asarray(pk, np.float32),
                                       np.asarray(pv, np.float32)]))
            return sess.run(None, feed)

        results["onnx"] = runner(run_onnx)
        fns["onnx"] = run_onnx

    ok = True
    for kind, r in results.items():
        # motorun gömme tablosu int8, ONNX'in matris çarpımları 8 bit: birebir değil, analizde fark etmeyecek
        # kadar yakınlık aranır. Isı haritası katman içinde 0-1'e ölçeklendiği için güç o ölçekte karşılaştırılır.
        k = 25 if kind == "onnx" else 8
        print(f"--- {kind}")
        norm = lambda p: (p - p.min(1, keepdims=True)) / np.maximum(p.max(1, keepdims=True) - p.min(1, keepdims=True), 1e-9)  # noqa: E731
        # ölçüt: masaüstü Neşter'in kendi bf16 hesabı fp32'den güçte ~0,08, dikkatte ~0,08 sapıyor (Qwen2.5-0.5B);
        # telefon motorundan bunun en fazla iki katı beklenir
        ok &= check("head gücü, katman içi 0-1", norm(r["power"].mean(-1)), norm(ref_power.mean(-1)), 0.16 if kind == "onnx" else 0.032)
        ok &= check("dikkat K3", r["attn3"], ref_attn, 0.16 if kind == "onnx" else 0.008)
        ok &= check("lens p_final", r["lens_p_target"], [x["p_final"] for x in ref_lens["layers"]], 2e-3 * k)
        same_top = sum(int(r["lens_id"][i][0]) == ref_lens["layers"][i]["top"][0]["token_id"] for i in range(len(ref_lens["layers"])))
        print(f"     lens ilk tahmin aynı: {same_top}/{len(ref_lens['layers'])}")
        ok &= check("kesimli KL", r["kl"], ref_kl, max(1e-4, 0.02 * ref_kl) * k)
    for kind, fn in fns.items():
        a = greedy(fn, ids, cut, 6, engine.M, cached=False)
        b = greedy(fn, ids, cut, 6, engine.M, cached=True)
        same = a == b
        ok &= same
        print(f"{'OK ' if same else 'HATA'} {kind} önbellekli üretim = baştan üretim: {b} / {a}")
    print("SONUÇ:", "geçti" if ok else "KALDI")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
