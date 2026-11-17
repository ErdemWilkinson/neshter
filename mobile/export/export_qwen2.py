"""Neşter mobil: bir Qwen2 modelini telefonda çalışacak tek bir ONNX motoruna çevirir.

Motor, Qwen2'nin ileri geçişini sade işlemlerle yeniden yazar (kanca yok), böylece Neşter'in ihtiyaç duyduğu
her şey grafiğin içinde olur:

  girdiler   input_ids [1, T] int64 · head_mask [L, H] float32 (1 açık, 0 kesik: o_proj girdisinde dilim sıfırlanır)
             keep [] int64 (log-olasılığı istenen son konum sayısı)
             lens_hidden [R, d] float32 · lens_target [] int64 (R = 0 ise lens hiç hesaplanmaz)
  çıktılar   logprobs [keep, V] · power [L, H, T] (‖W_h z_h‖, maskeden sonra) · attn [L, H, T, T]
             resid [L+1, T, d] (0 = gömme, ℓ+1 = katman ℓ çıkışı, normsuz) ·
             lens_p [R, 3] · lens_id [R, 3] · lens_p_target [R] · lens_rank_target [R]

Kullanım (proje dışındaki dönüştürme ortamıyla):
  python mobile/export/export_qwen2.py MODEL_KLASORU CIKIS_KLASORU [--no-quant]
Çıkış: engine.onnx (int8), tokenizer.json, meta.json, LICENSE. --verify sayısal karşılaştırmayı da koşar.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from torch import nn


class Qwen2Engine(nn.Module):
    def __init__(self, hf) -> None:
        super().__init__()
        cfg = hf.config
        self.L = cfg.num_hidden_layers
        self.H = cfg.num_attention_heads
        self.KV = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.M = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, self.D, 2, dtype=torch.float32) / self.D)))
        m = hf.model
        f = lambda t: t.detach().to(torch.float32).clone()  # noqa: E731
        # gömme tablosu satır başına ölçekli int8 (motorun içinde: int8 tabloda arama, sonra ölçekle çarpma)
        e = f(m.embed_tokens.weight)
        scale = (e.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        self.register_buffer("emb_q", torch.round(e / scale).to(torch.int8))
        self.register_buffer("emb_s", scale)
        # lm_head ayrı ve önceden çevrilmiş: MatMul ağırlığı sabit bir başlatıcı olsun (int8'e çevrilebilsin)
        self.lm_w = nn.Parameter(f(hf.lm_head.weight).T.contiguous(), requires_grad=False)
        self.final_norm = nn.Parameter(f(m.norm.weight), requires_grad=False)
        self.layers = nn.ModuleList()
        for layer in m.layers:
            a, p = layer.self_attn, layer.mlp
            mod = nn.Module()
            for name, t in {
                "ln1": layer.input_layernorm.weight, "ln2": layer.post_attention_layernorm.weight,
                "wq": a.q_proj.weight.T, "bq": a.q_proj.bias, "wk": a.k_proj.weight.T, "bk": a.k_proj.bias,
                "wv": a.v_proj.weight.T, "bv": a.v_proj.bias, "wo": a.o_proj.weight.T,
                "wg": p.gate_proj.weight.T, "wu": p.up_proj.weight.T, "wd": p.down_proj.weight.T,
            }.items():
                setattr(mod, name, nn.Parameter(f(t).contiguous(), requires_grad=False))
            # head gücü için Gram matrisleri G_h = W_hᵀ W_h, [H, D, D] (Neşter'in head_grams'ı; W_o'nun ikinci kopyası değil)
            w = f(a.o_proj.weight).reshape(self.M, self.H, self.D)
            mod.grams = nn.Parameter(torch.einsum("mhd,mhe->hde", w, w).contiguous(), requires_grad=False)
            self.layers.append(mod)

    def _norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, input_ids, head_mask, keep, lens_hidden, lens_target, past_k, past_v):
        """past_k/past_v [L, KV, P, D]: önceki token'ların (RoPE'lu) anahtar/değerleri; analizde P=0, metin üretirken
        her adım yalnız yeni token'ı işler. Dikkat çıktısı [L, H, T, P+T]."""
        ids = input_ids[0]
        T = ids.shape[0]
        P = past_k.shape[2]
        x = self.emb_q[ids].to(torch.float32) * self.emb_s[ids]  # [T, M]
        pos = torch.arange(T, dtype=torch.float32) + P
        freqs = pos[:, None] * self.inv_freq[None, :]
        ang = torch.cat([freqs, freqs], dim=-1)
        cos, sin = ang.cos()[None], ang.sin()[None]  # [1, T, D]
        causal = torch.triu(torch.full((T, T + P), -1e9), diagonal=P + 1)
        group = self.H // self.KV
        resid = [x]
        powers, attns, pres_k, pres_v = [], [], [], []
        for li, ly in enumerate(self.layers):
            h = self._norm(x, ly.ln1)
            q = (h @ ly.wq + ly.bq).reshape(T, self.H, self.D).transpose(0, 1)  # [H, T, D]
            k = (h @ ly.wk + ly.bk).reshape(T, self.KV, self.D).transpose(0, 1)
            v = (h @ ly.wv + ly.bv).reshape(T, self.KV, self.D).transpose(0, 1)
            q = q * cos + self._rot(q) * sin
            k = k * cos + self._rot(k) * sin
            k = torch.cat([past_k[li], k], dim=1)  # [KV, P+T, D]
            v = torch.cat([past_v[li], v], dim=1)
            pres_k.append(k)
            pres_v.append(v)
            k = k.repeat_interleave(group, dim=0)
            v = v.repeat_interleave(group, dim=0)
            att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.D) + causal, dim=-1)  # [H, T, T]
            attns.append(att)
            z = (att @ v).transpose(0, 1) * head_mask[li][None, :, None]  # [T, H, D], kesik head'ler sıfır
            sq = torch.einsum("thd,hde,the->th", z, ly.grams, z)
            powers.append(sq.clamp_min(0).sqrt().transpose(0, 1))  # [H, T]
            x = x + z.reshape(T, self.H * self.D) @ ly.wo
            h = self._norm(x, ly.ln2)
            x = x + (torch.nn.functional.silu(h @ ly.wg) * (h @ ly.wu)) @ ly.wd
            resid.append(x)
        last = x[T - keep :]
        logprobs = torch.log_softmax(self._norm(last, self.final_norm) @ self.lm_w, dim=-1)
        lens_logits = self._norm(lens_hidden, self.final_norm) @ self.lm_w
        lens_prob = torch.softmax(lens_logits, dim=-1)
        top_p, top_id = torch.topk(lens_prob, 3, dim=-1)
        p_target = lens_prob[:, lens_target]
        rank = (lens_prob > p_target[:, None]).sum(-1)
        return (logprobs, torch.stack(powers), torch.stack(attns), torch.stack(resid), top_p, top_id, p_target, rank,
                torch.stack(pres_k), torch.stack(pres_v))


NAMES_IN = ["input_ids", "head_mask", "keep", "lens_hidden", "lens_target", "past_k", "past_v"]
NAMES_OUT = ["logprobs", "power", "attn", "resid", "lens_p", "lens_id", "lens_p_target", "lens_rank_target",
             "present_k", "present_v"]


def empty_past(engine: "Qwen2Engine", P: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (engine.L, engine.KV, P, engine.D)
    return torch.zeros(shape), torch.zeros(shape)


def example(engine: Qwen2Engine, T: int = 6, R: int = 2):
    return (
        torch.randint(0, 1000, (1, T), dtype=torch.int64),
        torch.ones(engine.L, engine.H),
        torch.tensor(2, dtype=torch.int64),
        torch.randn(R, engine.M),
        torch.tensor(5, dtype=torch.int64),
        *empty_past(engine, 3),
    )


def export(engine: Qwen2Engine, path: Path) -> None:
    torch.onnx.export(
        engine,
        example(engine),
        str(path),
        input_names=NAMES_IN,
        output_names=NAMES_OUT,
        dynamic_axes={"input_ids": {1: "T"}, "lens_hidden": {0: "R"}, "logprobs": {0: "K"}, "power": {2: "T"},
                      "attn": {2: "T", 3: "S"}, "resid": {1: "T"}, "lens_p": {0: "R"}, "lens_id": {0: "R"},
                      "lens_p_target": {0: "R"}, "lens_rank_target": {0: "R"}, "past_k": {2: "P"}, "past_v": {2: "P"},
                      "present_k": {2: "S"}, "present_v": {2: "S"}},
        opset_version=20,
        dynamo=False,
        external_data=True,
    )


def quantize(src: Path, dst: Path) -> None:
    """Yalnızca ağırlıklar sıkıştırılır, aktivasyonlar fp32 kalır (Qwen'in artık akımındaki aykırı değerler
    dinamik int8 aktivasyon nicemlemesini bozuyor). Matris çarpımları 8 bit, 32'lik bloklarla, simetrik. Gömme tablosu
    zaten motorun içinde satır başına int8 (bkz. Qwen2Engine)."""
    import onnx
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    model = onnx.load(str(src))
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4,
                                       quant_format=QuantFormat.QOperator, op_types_to_quantize=("MatMul",),
                                       quant_axes=(("MatMul", 0),), bits=8)
    q = MatMulNBitsQuantizer(model, algo_config=cfg)
    q.process()
    onnx.save(q.model.model, str(dst))


PHONE_NEUTRAL = 8  # telefonda "gerçek etki" bu kadar nötr istemle ölçülür (40'ı telefonda dakikalar sürer)


def model_name(model_dir: Path) -> str:
    """HF önbellek klasöründe (models--Org--Ad/snapshots/<hash>) asıl ad: "Ad"; düz klasörde klasör adı."""
    for part in reversed(model_dir.resolve().parts):
        if part.startswith("models--"):
            return part.split("--")[-1]
    return model_dir.name


def neutral_prompts(k: int = PHONE_NEUTRAL) -> list[str]:
    import yaml

    path = Path(__file__).resolve().parents[2] / "neshter" / "concepts" / "neutral.yaml"
    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["prompts"])[:k]


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tokenizer_txt(out: Path) -> None:
    """tokenizer.json'dan telefonun JSON ayrıştırmadan okuyacağı üç düz dosya: vocab.txt (satır = id, bayt düzeyi
    işaretli token), merges.txt (sırayla "a b"), added.txt ("id<TAB>metin<TAB>special"). Bayt düzeyi token'larda ham
    satır sonu yoktur (Ċ olarak işaretlenir), dolayısıyla satır başına bir kayıt güvenlidir."""
    t = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = t["model"]["vocab"]
    rows = [""] * (max(vocab.values()) + 1)
    for piece, i in vocab.items():
        assert '\n' not in piece and '\r' not in piece, piece
        rows[i] = piece
    nl = chr(10)
    (out / 'vocab.txt').write_text(nl.join(rows) + nl, encoding='utf-8', newline=nl)
    merges = [m if isinstance(m, str) else ' '.join(m) for m in t['model']['merges']]
    (out / 'merges.txt').write_text(nl.join(merges) + nl, encoding='utf-8', newline=nl)
    added = [f"{a['id']}{chr(9)}{a['content']}{chr(9)}{int(a['special'])}" for a in t['added_tokens']]
    (out / 'added.txt').write_text(nl.join(added) + nl, encoding='utf-8', newline=nl)


def write_meta(model_dir: Path, out: Path, hf) -> dict:
    """meta.json: telefonun bilmesi gereken her şey. Sohbet şablonu Jinja olarak değil, kullanıcı mesajının önüne
    ve arkasına gelen iki düz metin olarak yazılır (telefonda şablon motoru yok)."""
    from transformers import AutoTokenizer

    cfg = hf.config
    tok = AutoTokenizer.from_pretrained(model_dir)
    mark = "\u0000"
    chat = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
    prefix, suffix = chat.split(mark)
    lic = None
    if (model_dir / "LICENSE").exists():
        first = (model_dir / "LICENSE").read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
        lic = "apache-2.0" if "Apache" in first else first[:60]
    eos = cfg.eos_token_id
    meta = {
        "name": model_name(model_dir),
        "family": "qwen2",
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "d_model": cfg.hidden_size,
        "d_mlp": cfg.intermediate_size,
        "n_params": int(sum(p.numel() for p in hf.parameters())),
        "vocab": int(cfg.vocab_size),
        "eos": [eos] if isinstance(eos, int) else list(eos),
        "license": lic,
        "chat_prefix": prefix,
        "chat_suffix": suffix,
        "neutral": neutral_prompts(),
        "size_bytes": (out / "engine.onnx").stat().st_size,
        "sha256": sha256_file(out / "engine.onnx"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--meta-only", action="store_true", help="motoru yeniden üretmeden yalnız meta.json")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    if hf.config.model_type != "qwen2":
        print(f"Yalnızca qwen2 destekleniyor: {hf.config.model_type}", file=sys.stderr)
        return 2
    final = a.out / "engine.onnx"
    if not a.meta_only:
        engine = Qwen2Engine(hf).eval()
        a.out.mkdir(parents=True, exist_ok=True)
        fp32 = a.out / "fp32" / "engine.onnx"
        fp32.parent.mkdir(exist_ok=True)
        with torch.no_grad():
            export(engine, fp32)
        if a.no_quant:
            shutil.copytree(fp32.parent, a.out, dirs_exist_ok=True)
        else:
            quantize(fp32, final)
    for name in ("tokenizer.json", "LICENSE"):
        if (a.model / name).exists():
            shutil.copy(a.model / name, a.out / name)
    write_tokenizer_txt(a.out)
    meta = write_meta(a.model, a.out, hf)
    print(f"tamam: {final} ({final.stat().st_size / 1e6:.0f} MB), {meta['name']}, sha {meta['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
