"""Plan 3 M2: logit lens, dikkat desenleri, tek cevap atıf grafiği ve bağ (yol) kesimi.

Tanımlar (spec 2026-11-10-neshter-3d-graph-design.md §2–§4):
  - M = logit_y − mean_v(logit_v), son pozisyonda. y varsayılan 1. tahmin.
  - Yazıcı katkısı a_A[p]: head için W_O[:, h]·z_h[p], nöron için W_down[:, n]·act_n[p], token için gömme
    (yalnızca kendi pozisyonunda).
  - Düğüm önemi I_A = Σ_p ∇_{x_A[p]} M · x_A[p] (gradyan × aktivasyon).
  - Kenar ağırlığı e(A→B, c) = Σ_p ⟨J_B[p]ᵀ u_B^c[p], a_A[p]⟩: u_B^c, M'nin B'nin c kanalının norm çıktısındaki
    okumasına göre gradyanı; J_B, B'nin RMSNorm'unun x'e göre Jacobian'ı (γ ve rms terimiyle). e, B'nin c
    kanalında yalnızca A'nın katkısı ε kadar ölçeklenince M'nin türevidir: yol kesiminde (ε = −1) ΔM ≈ −e.
  - Yol kesimi: B, c kanalında norm(x − a_A) okur; A'nın başka her etkisi aynı kalır. Ağırlığa gömülemez
    (yalnızca keşif).

v0.1 sınırı: yalnızca llama ve qwen2 (Gemma2'nin post-norm ve softcap'leri ayrı bir türev ister). K bileşimi yok.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

from neshter.arch import decoder_layers, embed
from neshter.loader import LoadedModel
from neshter.progress import ProgressFn, check_cancel

GRAPH_FAMILIES = ("llama", "qwen2")
MAX_GRAPH_TOKENS = 128
MAX_LENS_TOKENS = 512
GRAPHS_KEPT = 4
_ATTN_NAME = "neshter_eager"


class GraphUnsupported(ValueError):
    """Bu model ailesi için grafik çıkarılamaz (422)."""


# --- eager attention: desenler aynı geçişte modülde saklanır ---------------------------------------


def _stashing_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    out, weights = eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    module._neshter_pattern = weights  # [B, H, Tq, Tk]; o_proj ön-kancası aynı geçişte okur
    return out, weights


AttentionInterface.register(_ATTN_NAME, _stashing_eager)
AttentionMaskInterface.register(_ATTN_NAME, eager_mask)  # nedensel maske eager ile aynı biçimde kurulsun


@contextmanager
def eager_patterns(model):
    """Geçiş süresince dikkat desenleri hesaplanır ve her self_attn modülünde `_neshter_pattern` olur."""
    cfg = model.config
    old = cfg._attn_implementation
    cfg._attn_implementation = _ATTN_NAME
    try:
        yield
    finally:
        cfg._attn_implementation = old
        for layer in decoder_layers(model):
            if hasattr(layer.self_attn, "_neshter_pattern"):
                del layer.self_attn._neshter_pattern


def _encode(lm: LoadedModel, prompt: str):
    device = next(lm.model.parameters()).device
    return lm.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)


def _piece(lm: LoadedModel, token_id: int) -> str:
    return lm.tokenizer.convert_tokens_to_string([lm.tokenizer.convert_ids_to_tokens(int(token_id))])


def _token_pieces(lm: LoadedModel, ids: torch.Tensor) -> list[str]:
    return [_piece(lm, int(i)) for i in ids]


def _check_family(lm: LoadedModel) -> None:
    if lm.arch.family not in GRAPH_FAMILIES:
        raise GraphUnsupported(
            f"Bağ grafiği şimdilik yalnızca Llama ve Qwen2 modellerinde var (bu model: {lm.arch.family})."
        )


def _check_len(n: int, limit: int) -> None:
    if n == 0:
        raise ValueError("İstem boş: en az bir token gerekli.")
    if n > limit:
        raise ValueError(f"İstem çok uzun: en fazla {limit} token (şu an {n}).")


def _final_softcap(model, logits: torch.Tensor) -> torch.Tensor:
    cap = getattr(model.config, "final_logit_softcapping", None)
    return logits if not cap else torch.tanh(logits / cap) * cap


# --- logit lens --------------------------------------------------------------------------------------


def _residuals(lm: LoadedModel, enc) -> list[torch.Tensor]:
    """[L+1] artık akım, [T, d] float32: 0 = ilk katmanın girdisi (Gemma2'de ölçekli gömme), ℓ+1 = katman ℓ çıkışı."""
    layers = decoder_layers(lm.model)
    out: list[torch.Tensor | None] = [None] * (len(layers) + 1)
    handles = [layers[0].register_forward_pre_hook(lambda m, a, kw: out.__setitem__(0, a[0] if a else kw["hidden_states"]), with_kwargs=True)]

    def make(i):
        def hook(module, args, output):
            out[i + 1] = output[0] if isinstance(output, tuple) else output

        return hook

    handles += [layer.register_forward_hook(make(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            lm.model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return [x[0].detach().float() for x in out]  # type: ignore[union-attr]


def _lens_logits(lm: LoadedModel, resid_vec: torch.Tensor) -> torch.Tensor:
    model = lm.model
    norm = model.model.norm
    w = next(norm.parameters())
    with torch.no_grad():
        x = norm(resid_vec.to(w.dtype).unsqueeze(0))
        logits = model.lm_head(x).float()[0]
    return _final_softcap(model, logits)


def logit_lens(lm: LoadedModel, prompt: str, position: int = -1, top_k: int = 5) -> dict:
    """Her katmandan sonra artık akımı son norm + lm_head'den geçirir: model o katmanda ne tahmin ediyor.

    `closeness[ℓ]`: son tahminin (y) o katmandaki olasılığı; 3B gövde nehrinin genişliği.
    """
    enc = _encode(lm, prompt)
    ids = enc["input_ids"][0]
    _check_len(len(ids), MAX_LENS_TOKENS)
    if not -len(ids) <= position < len(ids):
        raise ValueError(f"position {position} aralık dışı (istem {len(ids)} token).")
    resid = _residuals(lm, enc)
    rows = [torch.softmax(_lens_logits(lm, r[position]), dim=-1) for r in resid]
    y = int(rows[-1].argmax())
    layers = []
    for i, p in enumerate(rows):
        top = torch.topk(p, top_k)
        layers.append(
            {
                "layer": i - 1,  # −1: gömme (ilk katmandan önce)
                "top": [{"token_id": int(t), "text": _piece(lm, int(t)), "p": float(v)} for v, t in zip(top.values, top.indices)],
                "p_final": float(p[y]),
                "rank_final": int((p > p[y]).sum()),
            }
        )
    return {
        "tokens": _token_pieces(lm, ids),
        "position": position % len(ids),
        "final": {"token_id": y, "text": _piece(lm, y)},
        "layers": layers,
    }


# --- dikkat desenleri -------------------------------------------------------------------------------


def attention_patterns(lm: LoadedModel, prompt: str, layer: int) -> dict:
    """Katman `layer`'daki her Q head'in deseni: [H][T_sorgu][T_anahtar], satırlar 1'e toplanır."""
    arch = lm.arch
    if not 0 <= layer < arch.n_layers:
        raise ValueError(f"Katman {layer} aralık dışı (0..{arch.n_layers - 1}).")
    enc = _encode(lm, prompt)
    ids = enc["input_ids"][0]
    _check_len(len(ids), MAX_GRAPH_TOKENS)
    attn = decoder_layers(lm.model)[layer].self_attn
    got: dict[str, torch.Tensor] = {}
    handle = attn.o_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("p", attn._neshter_pattern.detach()))
    try:
        with eager_patterns(lm.model), torch.no_grad():
            lm.model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    pat = got["p"][0].float()
    return {"tokens": _token_pieces(lm, ids), "layer": layer, "patterns": [[[round(float(v), 4) for v in row] for row in h] for h in pat]}


# --- yakalama ---------------------------------------------------------------------------------------


@dataclass
class _Capture:
    emb: torch.Tensor | None = None  # [1, T, d] ilk katmanın girdisi (grad'lı)
    q: dict[int, torch.Tensor] = field(default_factory=dict)  # q_proj çıkışı [1, T, H·hd]
    v: dict[int, torch.Tensor] = field(default_factory=dict)  # v_proj çıkışı [1, T, KV·hd]
    z: dict[int, torch.Tensor] = field(default_factory=dict)  # o_proj girdisi [1, T, H·hd]
    gate: dict[int, torch.Tensor] = field(default_factory=dict)
    up: dict[int, torch.Tensor] = field(default_factory=dict)
    act: dict[int, torch.Tensor] = field(default_factory=dict)  # down_proj girdisi [1, T, d_mlp]
    x_attn: dict[int, torch.Tensor] = field(default_factory=dict)  # input_layernorm girdisi
    x_mlp: dict[int, torch.Tensor] = field(default_factory=dict)  # post_attention_layernorm girdisi
    pattern: dict[int, torch.Tensor] = field(default_factory=dict)  # [H, T, T]
    x_final: torch.Tensor | None = None  # son norm girdisi [1, T, d]


def _capture_hooks(model, cap: _Capture, grad: bool) -> list:
    handles = []
    layers = decoder_layers(model)

    def emb_hook(module, args, output):
        if grad:
            output = output.detach().requires_grad_(True)
        cap.emb = output
        return output

    handles.append(embed(model).register_forward_hook(emb_hook))
    for i, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp

        def store(d, i=i):
            def hook(module, args, output):
                d[i] = output

            return hook

        def store_in(d, i=i):
            def hook(module, args):
                d[i] = args[0]

            return hook

        def z_hook(module, args, i=i, attn=attn):
            cap.z[i] = args[0]
            cap.pattern[i] = attn._neshter_pattern[0].detach()

        handles += [
            attn.q_proj.register_forward_hook(store(cap.q)),
            attn.v_proj.register_forward_hook(store(cap.v)),
            attn.o_proj.register_forward_pre_hook(z_hook),
            mlp.gate_proj.register_forward_hook(store(cap.gate)),
            mlp.up_proj.register_forward_hook(store(cap.up)),
            mlp.down_proj.register_forward_pre_hook(store_in(cap.act)),
            layer.input_layernorm.register_forward_pre_hook(store_in(cap.x_attn)),
            layer.post_attention_layernorm.register_forward_pre_hook(store_in(cap.x_mlp)),
        ]
    handles.append(model.model.norm.register_forward_pre_hook(lambda m, a: setattr(cap, "x_final", a[0])))
    return handles


def _metric(model, logits_last: torch.Tensor, y: int) -> torch.Tensor:
    """M = logit_y − mean_v logit_v (softcap öncesi; llama/qwen2'de softcap yok)."""
    return logits_last[y] - logits_last.mean()


def _norm_pullback(norm, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """RMSNorm n = γ⊙x/rms(x) için Jᵀu = γ⊙u/rms − x·⟨x, γ⊙u⟩/(d·rms³). x, u: [..., d], float."""
    gamma = norm.weight.detach().to(u.dtype)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    d = x.shape[-1]
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    gu = gamma * u
    return gu / rms - x * (x * gu).sum(-1, keepdim=True) / (d * rms**3)


# --- grafik ---------------------------------------------------------------------------------------


@dataclass
class _Node:
    id: str
    kind: str  # token | head | neuron | output
    layer: int  # token −1, output n_layers
    index: int  # pozisyon / head / nöron / y
    score: float = 0.0
    order: float = 0.0  # yazma/okuma sırası: token −1, head 2ℓ, nöron 2ℓ+1, output ∞


@dataclass
class Graph:
    id: str
    prompt: str
    y: int
    metric: float
    nodes: dict[str, _Node]
    edges: list[dict]
    json: dict


class GraphStore:
    """Oturumdaki son GRAPHS_KEPT grafik; model değişince temizlenir."""

    def __init__(self) -> None:
        self._g: OrderedDict[str, Graph] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, g: Graph) -> None:
        with self._lock:
            self._g[g.id] = g
            while len(self._g) > GRAPHS_KEPT:
                self._g.popitem(last=False)

    def get(self, graph_id: str) -> Graph:
        with self._lock:
            return self._g[graph_id]  # KeyError: eskidi

    def clear(self) -> None:
        with self._lock:
            self._g.clear()


def _forward_capture(lm: LoadedModel, enc, grad: bool, extra_hooks=()):
    cap = _Capture()
    model = lm.model
    handles = _capture_hooks(model, cap, grad)
    handles += [h() for h in extra_hooks]
    try:
        with eager_patterns(model), torch.set_grad_enabled(grad):
            out = model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return cap, out.logits[0, -1]


def answer_graph(
    lm: LoadedModel,
    prompt: str,
    target_token_id: int | None = None,
    heads_per_layer: int = 6,
    neurons_per_layer: int = 12,
    max_nodes: int = 300,
    max_edges: int = 500,
    edge_threshold_frac: float = 0.01,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> Graph:
    """Tek ileri + tek geri geçişle tahmini atıf grafiği (spec §3, §5 "Tek cevap")."""
    _check_family(lm)
    arch, model = lm.arch, lm.model
    enc = _encode(lm, prompt)
    ids = enc["input_ids"][0]
    T = len(ids)
    _check_len(T, MAX_GRAPH_TOKENS)
    check_cancel(cancel)
    if progress:
        progress(0.05, "İleri geçiş")
    L = arch.n_layers

    # Büyük bir model işlemcide (özellikle yerel bf16'sı olmayan CPU'da) ileri ve geri geçişte dakikalar harcar:
    # her katmanda ilerleme bildirilir ve iptal dinlenir, yoksa iş "takılmış" görünür ve iptal edilemez.
    def layer_hook(i):
        def hook(module, args, output):
            check_cancel(cancel)
            if progress:
                progress(0.05 + 0.25 * (i + 1) / L, f"İleri geçiş: katman {i + 1}/{L}")

        return lambda: decoder_layers(model)[i].register_forward_hook(hook)

    cap, logits = _forward_capture(lm, enc, grad=True, extra_hooks=[layer_hook(i) for i in range(L)])
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    top5 = torch.topk(logits, 5).indices.tolist()
    y = top5[0] if target_token_id is None else int(target_token_id)
    if not 0 <= y < logits.shape[0]:
        raise ValueError(f"target_token_id {y} sözlükte yok.")
    M = _metric(model, logits, y)
    H, KV, hd = arch.n_heads, arch.n_kv_heads, arch.head_dim
    group = H // KV
    order = [cap.emb] + [cap.q[i] for i in range(L)] + [cap.v[i] for i in range(L)] + [cap.z[i] for i in range(L)]
    order += [cap.gate[i] for i in range(L)] + [cap.up[i] for i in range(L)] + [cap.act[i] for i in range(L)]
    check_cancel(cancel)
    if progress:
        progress(0.3, "Geri geçiş")
    # Türev son katmandan ilke doğru akar: her katmanın girdisine türev ulaştığında bildir; iptal burada atılan
    # Cancelled ile geri geçişi yarıda keser.
    back_done = [0]

    def back_hook(grad):
        back_done[0] += 1
        check_cancel(cancel)
        if progress:
            progress(0.3 + 0.2 * back_done[0] / L, f"Geri geçiş: katman {L - back_done[0] + 1}/{L}")
        return grad

    back = [cap.x_attn[i].register_hook(back_hook) for i in range(L) if cap.x_attn[i].requires_grad]
    try:
        grads = torch.autograd.grad(M, order, allow_unused=True)
    finally:
        for hnd in back:
            hnd.remove()
    grads = [torch.zeros_like(t) if g is None else g for t, g in zip(order, grads)]
    f = torch.promote_types(cap.emb.dtype, torch.float32)
    g_emb = grads[0][0].to(f)
    g_q = [grads[1 + i][0].to(f) for i in range(L)]
    g_v = [grads[1 + L + i][0].to(f) for i in range(L)]  # yalnızca doğrulama (head başına V analitik kurulur)
    g_z = [grads[1 + 2 * L + i][0].to(f) for i in range(L)]
    g_gate = [grads[1 + 3 * L + i][0].to(f) for i in range(L)]
    g_up = [grads[1 + 4 * L + i][0].to(f) for i in range(L)]
    g_act = [grads[1 + 5 * L + i][0].to(f) for i in range(L)]
    del g_v
    det = lambda t: t[0].detach().to(f)  # noqa: E731
    emb = det(cap.emb)
    z = {i: det(cap.z[i]).view(T, H, hd) for i in range(L)}
    act = {i: det(cap.act[i]) for i in range(L)}

    # düğüm önemi
    nodes: dict[str, _Node] = {}
    for p in range(T):
        nodes[f"t:{p}"] = _Node(f"t:{p}", "token", -1, p, float((g_emb[p] * emb[p]).sum()), -1)
    head_scores = torch.stack([(g_z[i].view(T, H, hd) * z[i]).sum(dim=(0, 2)) for i in range(L)])  # [L, H]
    neuron_scores = [(g_act[i] * act[i]).sum(0) for i in range(L)]  # [d_mlp]
    cand: list[_Node] = []
    for i in range(L):
        for h in torch.topk(head_scores[i].abs(), min(heads_per_layer, H)).indices.tolist():
            cand.append(_Node(f"h:{i}:{h}", "head", i, h, float(head_scores[i, h]), 2 * i))
        if neurons_per_layer > 0:
            for n in torch.topk(neuron_scores[i].abs(), min(neurons_per_layer, arch.d_mlp)).indices.tolist():
                cand.append(_Node(f"n:{i}:{n}", "neuron", i, n, float(neuron_scores[i][n]), 2 * i + 1))
    scored = L * H + L * arch.d_mlp + T
    cand.sort(key=lambda n: abs(n.score), reverse=True)
    for n in cand[:max_nodes]:
        nodes[n.id] = n
    nodes["out"] = _Node("out", "output", L, y, float(M.detach()), float("inf"))
    check_cancel(cancel)
    if progress:
        progress(0.5, "Bağ ağırlıkları")

    layers = decoder_layers(model)
    W_U = model.lm_head.weight.detach().to(f)

    # yazıcılar [nW, T, d]
    writers, wnodes = [], []
    for n in nodes.values():
        if n.kind == "token":
            a = torch.zeros(T, arch.d_model, dtype=f, device=emb.device)
            a[n.index] = emb[n.index]
        elif n.kind == "head":
            W_O = layers[n.layer].self_attn.o_proj.weight.detach().to(f)[:, n.index * hd : (n.index + 1) * hd]
            a = z[n.layer][:, n.index] @ W_O.T
        elif n.kind == "neuron":
            a = act[n.layer][:, n.index : n.index + 1] * layers[n.layer].mlp.down_proj.weight.detach().to(f)[:, n.index]
        else:
            continue
        writers.append(a)
        wnodes.append(n)
    Wt = torch.stack(writers)

    # okuyucular ve kenarlar
    edges: list[dict] = []

    def add_edges(reader: torch.Tensor, dst: _Node, channel: str, read_order: float):
        e = torch.einsum("atd,td->a", Wt, reader)
        for k, src in enumerate(wnodes):
            if src.order < read_order:
                edges.append({"src": src.id, "dst": dst.id, "channel": channel, "weight": float(e[k])})

    readers = [n for n in nodes.values() if n.kind in ("head", "neuron")]
    for k, n in enumerate(readers):
        if progress and k % 8 == 0:
            progress(0.5 + 0.4 * k / max(1, len(readers)), f"Bağ ağırlıkları: {k}/{len(readers)} düğüm")
        if n.kind == "head":
            i, h = n.layer, n.index
            attn, norm = layers[i].self_attn, layers[i].input_layernorm
            x = det(cap.x_attn[i])
            W_q = attn.q_proj.weight.detach().to(f)[h * hd : (h + 1) * hd]
            u_q = g_q[i].view(T, H, hd)[:, h] @ W_q
            add_edges(_norm_pullback(norm, x, u_q), n, "q", 2 * i)
            A = cap.pattern[i][h].to(f)  # [Tq, Tk]
            dv = A.T @ g_z[i].view(T, H, hd)[:, h]  # head'e özel ∂M/∂v_h
            g = h // group
            W_v = attn.v_proj.weight.detach().to(f)[g * hd : (g + 1) * hd]
            add_edges(_norm_pullback(norm, x, dv @ W_v), n, "v", 2 * i)
        elif n.kind == "neuron":
            i, k = n.layer, n.index
            mlp, norm = layers[i].mlp, layers[i].post_attention_layernorm
            x = det(cap.x_mlp[i])
            u = g_gate[i][:, k : k + 1] * mlp.gate_proj.weight.detach().to(f)[k] + g_up[i][:, k : k + 1] * mlp.up_proj.weight.detach().to(f)[k]
            add_edges(_norm_pullback(norm, x, u), n, "resid", 2 * i + 1)
        check_cancel(cancel)
    x_f = det(cap.x_final)
    u_out = torch.zeros(T, arch.d_model, dtype=f, device=emb.device)
    u_out[-1] = W_U[y] - W_U.mean(0)
    add_edges(_norm_pullback(model.model.norm, x_f, u_out), nodes["out"], "logit", float("inf"))

    # budama
    candidate = len(edges)
    mx = max((abs(e["weight"]) for e in edges), default=0.0)
    thr = edge_threshold_frac * mx
    edges = sorted((e for e in edges if abs(e["weight"]) >= thr and abs(e["weight"]) > 0), key=lambda e: -abs(e["weight"]))[:max_edges]
    used = {e["src"] for e in edges} | {e["dst"] for e in edges}
    nodes = {k: n for k, n in nodes.items() if n.kind in ("token", "output") or k in used}
    for k, e in enumerate(edges):
        e["id"] = f"e{k}"
        e["badge"] = "yalnızca keşif"
    if progress:
        progress(0.9, "Nehir")

    # nehir: katman başına cevap yönüne izdüşüm (logit lens yakınlığı) ve norm
    with torch.no_grad():
        resid_last = [det(cap.x_attn[i])[-1] for i in range(L)] + [x_f[-1]]
        u_y = W_U[y] - W_U.mean(0)
        trunk = [float((model.model.norm(r.to(next(model.parameters()).dtype).unsqueeze(0))[0].to(f) * u_y).sum()) for r in resid_last]
        norms = [float(r.norm()) for r in resid_last]

    def node_json(n: _Node) -> dict:
        return {"id": n.id, "kind": n.kind, "layer": n.layer, "index": n.index, "score": n.score}

    gid = uuid.uuid4().hex[:12]
    j = {
        "graph_id": gid,
        "mode": "prompt",
        "estimated": True,
        "safety": False,
        "dtype": str(cap.emb.dtype).replace("torch.", ""),
        "warnings": ["K bileşimi gösterilmiyor."],
        "metric": {"kind": "logit_diff", "token_id": y, "text": _piece(lm, y), "value": float(M.detach())},
        "candidates": [{"token_id": t, "text": _piece(lm, t)} for t in top5],
        "tokens": [{"pos": p, "token_id": int(ids[p]), "text": _piece(lm, int(ids[p]))} for p in range(T)],
        "n_layers": L,
        "n_heads": H,
        "nodes": [node_json(n) for n in nodes.values()],
        "edges": edges,
        "rivers": {"trunk": trunk, "norms": norms},
        "pruning": {"scored_nodes": scored, "kept_nodes": len(nodes), "candidate_edges": candidate, "kept_edges": len(edges), "edge_threshold": thr},
    }
    if progress:
        progress(1.0, "Grafik hazır")
    return Graph(gid, prompt, y, float(M.detach()), nodes, edges, j)


# --- yol kesimi -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeCut:
    src: str  # t:p | h:ℓ:h | n:ℓ:k
    dst: str  # h:ℓ:h | n:ℓ:k | out
    channel: str  # q | v | resid | logit
    scale: float = 1.0  # 1 = tam yol kesimi; testlerde küçük ε türev denetimi için


def _parse(node_id: str) -> tuple[str, int, int]:
    parts = node_id.split(":")
    if parts[0] == "t":
        return "token", -1, int(parts[1])
    if parts[0] == "h":
        return "head", int(parts[1]), int(parts[2])
    if parts[0] == "n":
        return "neuron", int(parts[1]), int(parts[2])
    if node_id == "out":
        return "output", -1, -1
    raise ValueError(f"Bilinmeyen düğüm: {node_id!r}")


def _validate_cut(lm: LoadedModel, c: EdgeCut) -> None:
    arch = lm.arch
    sk, sl, si = _parse(c.src)
    dk, dl, di = _parse(c.dst)
    want = {"head": ("q", "v"), "neuron": ("resid",), "output": ("logit",)}
    if dk not in want or c.channel not in want[dk]:
        raise ValueError(f"{c.dst} için {c.channel!r} kanalı yok.")
    if sk not in ("token", "head", "neuron"):
        raise ValueError(f"Kaynak {c.src} bir yazıcı değil.")
    for k, layer, i in ((sk, sl, si), (dk, dl, di)):
        if k == "head" and not (0 <= layer < arch.n_layers and 0 <= i < arch.n_heads):
            raise ValueError(f"Head {layer}:{i} bu modelde yok.")
        if k == "neuron" and not (0 <= layer < arch.n_layers and 0 <= i < arch.d_mlp):
            raise ValueError(f"Nöron {layer}:{i} bu modelde yok.")
    src_order = -1 if sk == "token" else 2 * sl + (1 if sk == "neuron" else 0)
    dst_order = float("inf") if dk == "output" else 2 * dl + (1 if dk == "neuron" else 0)
    if not src_order < dst_order:
        raise ValueError(f"{c.src} → {c.dst}: kaynak hedeften sonra geliyor.")


def _rms_apply(norm, x: torch.Tensor) -> torch.Tensor:
    """Llama/Qwen2 RMSNorm'unu modülü ÇAĞIRMADAN uygular: modül çağrısı yakalama kancalarını yeniden tetikler
    ve saklanan girdinin üzerine yazardı (aynı katmandaki ikinci okuyucu kesimi iki kez görürdü)."""
    dt = x.dtype
    h = x.to(torch.float32)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return norm.weight * h.to(dt)


def run_with_cuts(lm: LoadedModel, prompt: str, cuts: list[EdgeCut], y: int) -> dict:
    """Bağ kesimleriyle tek ileri geçiş (gerçek, tahmini değil): M, ilk 5 tahmin. Kesim yoksa temiz geçiş."""
    _check_family(lm)
    for c in cuts:
        _validate_cut(lm, c)
    arch, model = lm.arch, lm.model
    enc = _encode(lm, prompt)
    ids = enc["input_ids"][0]
    _check_len(len(ids), MAX_GRAPH_TOKENS)
    hd, H = arch.head_dim, arch.n_heads
    group = H // arch.n_kv_heads
    layers = decoder_layers(model)
    live: dict = {"emb": None, "z": {}, "act": {}, "x_attn": {}, "x_mlp": {}, "x_final": None}

    def contrib(src: str) -> torch.Tensor:
        """Aynı geçişteki a_A [T, d] (modelin dtype'ında)."""
        k, layer, i = _parse(src)
        if k == "token":
            a = torch.zeros_like(live["emb"])
            a[i] = live["emb"][i]
            return a
        if k == "head":
            W_O = layers[layer].self_attn.o_proj.weight[:, i * hd : (i + 1) * hd]
            return live["z"][layer][:, i * hd : (i + 1) * hd] @ W_O.T
        return live["act"][layer][:, i : i + 1] * layers[layer].mlp.down_proj.weight[:, i]

    def removed(dst: str, channel: str) -> torch.Tensor | None:
        sel = [c for c in cuts if c.dst == dst and c.channel == channel]
        if not sel:
            return None
        return sum(c.scale * contrib(c.src) for c in sel)

    handles = []
    handles.append(embed(model).register_forward_hook(lambda m, a, o: live.__setitem__("emb", o[0])))
    for li, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp
        handles.append(layer.input_layernorm.register_forward_pre_hook(lambda m, a, li=li: live["x_attn"].__setitem__(li, a[0][0])))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a, li=li: live["x_mlp"].__setitem__(li, a[0][0])))

        def q_hook(module, args, output, li=li, layer=layer):
            out = output.clone()
            for h in range(H):
                rem = removed(f"h:{li}:{h}", "q")
                if rem is not None:
                    n = _rms_apply(layer.input_layernorm, live["x_attn"][li] - rem)
                    w = module.weight[h * hd : (h + 1) * hd]
                    b = None if module.bias is None else module.bias[h * hd : (h + 1) * hd]
                    out[0, :, h * hd : (h + 1) * hd] = torch.nn.functional.linear(n, w, b)
            return out

        def o_pre(module, args, li=li, layer=layer, attn=attn):
            zin = args[0]
            live["z"][li] = zin[0]
            changed = None
            for h in range(H):
                rem = removed(f"h:{li}:{h}", "v")
                if rem is None:
                    continue
                g = h // group
                vproj = attn.v_proj
                w = vproj.weight[g * hd : (g + 1) * hd]
                b = None if vproj.bias is None else vproj.bias[g * hd : (g + 1) * hd]
                x = live["x_attn"][li]
                v_old = torch.nn.functional.linear(_rms_apply(layer.input_layernorm, x), w, b)
                v_new = torch.nn.functional.linear(_rms_apply(layer.input_layernorm, x - rem), w, b)
                A = attn._neshter_pattern[0, h].to(v_new.dtype)
                if changed is None:
                    changed = zin.clone()
                changed[0, :, h * hd : (h + 1) * hd] = zin[0, :, h * hd : (h + 1) * hd] + A @ (v_new - v_old)
            if changed is not None:
                live["z"][li] = changed[0]
                return (changed,)
            return None

        def mlp_proj_hook(which, li=li, layer=layer):
            def hook(module, args, output):
                out = None
                for c in cuts:
                    if c.channel != "resid":
                        continue
                    k, dl, di = _parse(c.dst)
                    if dl != li:
                        continue
                    rem = removed(c.dst, "resid")
                    n = _rms_apply(layer.post_attention_layernorm, live["x_mlp"][li] - rem)
                    if out is None:
                        out = output.clone()
                    out[0, :, di] = n @ module.weight[di] + (0 if module.bias is None else module.bias[di])
                return out

            return hook

        handles += [
            attn.q_proj.register_forward_hook(q_hook),
            attn.o_proj.register_forward_pre_hook(o_pre),
            mlp.gate_proj.register_forward_hook(mlp_proj_hook("gate")),
            mlp.up_proj.register_forward_hook(mlp_proj_hook("up")),
            mlp.down_proj.register_forward_pre_hook(lambda m, a, li=li: live["act"].__setitem__(li, a[0][0])),
        ]
    handles.append(model.model.norm.register_forward_pre_hook(lambda m, a: live.__setitem__("x_final", a[0][0])))

    def head_hook(module, args, output):
        rem = removed("out", "logit")
        if rem is None:
            return None
        out = output.clone()
        # A'nın doğrudan (unembedding) etkisi: son pozisyonun tüm logit satırı norm(x − a_A)'dan yeniden hesaplanır.
        n = _rms_apply(model.model.norm, (live["x_final"] - rem)[-1])
        out[0, -1] = torch.nn.functional.linear(n, module.weight, module.bias)
        return out

    handles.append(model.lm_head.register_forward_hook(head_hook))
    try:
        with eager_patterns(model), torch.no_grad():
            logits = model(**enc, use_cache=False, logits_to_keep=1).logits[0, -1]
    finally:
        for h in handles:
            h.remove()
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    probs = torch.softmax(_final_softcap(model, logits).float(), -1)
    top = torch.topk(probs, 5)
    return {
        "metric": float(_metric(model, logits, y)),
        "p_target": float(probs[y]),
        "top": [{"token_id": int(t), "text": _piece(lm, int(t)), "p": float(v)} for v, t in zip(top.values, top.indices)],
        "log_probs": torch.log_softmax(logits.float(), -1),
    }


def cut_report(lm: LoadedModel, g: Graph, edge_ids: list[str]) -> dict:
    """Seçilen bağlar birlikte (yol kesimi) kesilince grafiğin cevabında ne değişti: gerçek ΔM, KL, ilk 5."""
    by_id = {e["id"]: e for e in g.edges}
    edge_ids = list(dict.fromkeys(edge_ids))  # aynı bağ iki kez yazılırsa katkısı iki kez çıkarılmasın
    missing = [e for e in edge_ids if e not in by_id]
    if missing:
        raise KeyError(f"Bu grafikte olmayan bağ: {missing[0]}")
    cuts = [EdgeCut(by_id[e]["src"], by_id[e]["dst"], by_id[e]["channel"]) for e in edge_ids]
    before = run_with_cuts(lm, g.prompt, [], g.y)
    after = run_with_cuts(lm, g.prompt, cuts, g.y)
    p, q = before["log_probs"], after["log_probs"]
    kl = max(0.0, float((p.exp() * (p - q)).sum()))  # bf16 yuvarlaması küçük negatifler üretebilir
    est = -sum(by_id[e]["weight"] for e in edge_ids)
    return {
        "edge_ids": edge_ids,
        "metric_before": before["metric"],
        "metric_after": after["metric"],
        "delta_metric": after["metric"] - before["metric"],
        "delta_estimated": est,
        "p_before": before["p_target"],
        "p_after": after["p_target"],
        "kl": kl,
        "top_before": before["top"],
        "top_after": after["top"],
    }
