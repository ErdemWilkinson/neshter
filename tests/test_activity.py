import torch

from neshter.activity import (
    head_output_power,
    normalize_global,
    normalize_per_layer,
    tokens,
    top_mlp_neurons,
)
from neshter.interventions import HeadAblation, apply, head_means

PROMPT = "bugün hava nasıl"


def test_shape_and_token_count(lm):
    p = head_output_power(lm, PROMPT)
    assert p.shape[:2] == (2, 4) and p.shape[2] == len(tokens(lm, PROMPT))
    assert (p >= 0).all()
    assert p.device.type == "cpu" and p.dtype == torch.float32


def test_ablated_head_has_zero_power(lm):
    with apply(lm, [HeadAblation(1, 3)]):
        p = head_output_power(lm, PROMPT)
    assert torch.all(p[1, 3] == 0)
    assert (p[1, 2] > 0).any()


def test_hooks_removed_after_call(lm):
    before = head_output_power(lm, PROMPT)
    top_mlp_neurons(lm, PROMPT, k=3)
    after = head_output_power(lm, PROMPT)
    assert torch.equal(before, after)
    for layer in lm.model.model.layers:
        assert not layer.self_attn.o_proj._forward_pre_hooks
        assert not layer.mlp.down_proj._forward_pre_hooks


def test_power_matches_manual_computation(lm):
    captured = {}
    o = lm.model.model.layers[0].self_attn.o_proj
    h = o.register_forward_pre_hook(lambda m, a: captured.setdefault("x", a[0].detach().clone()))
    try:
        with torch.no_grad():
            lm.model(**lm.tokenizer(PROMPT, return_tensors="pt"))
    finally:
        h.remove()
    z = captured["x"][0]  # [T, H*d]
    d = lm.arch.head_dim
    expected = torch.stack(
        [(z[:, i * d : (i + 1) * d] @ o.weight[:, i * d : (i + 1) * d].T).norm(dim=-1) for i in range(4)]
    )
    assert torch.allclose(head_output_power(lm, PROMPT)[0], expected, atol=1e-5)


def test_tokens_are_display_strings(lm):
    toks = tokens(lm, PROMPT)
    assert all(isinstance(t, str) for t in toks)
    assert not any("Ġ" in t for t in toks)


def test_top_mlp_neurons(lm):
    top = top_mlp_neurons(lm, PROMPT, k=5)
    assert len(top) == 5 and all(0 <= l < 2 and 0 <= n < 128 for l, n, _ in top)
    assert [v for *_, v in top] == sorted([v for *_, v in top], reverse=True)
    assert all(type(l) is int and type(n) is int and type(v) is float for l, n, v in top)


def test_top_mlp_neurons_token_index(lm):
    n_tok = len(tokens(lm, PROMPT))
    assert top_mlp_neurons(lm, PROMPT, token_index=-1, k=5) == top_mlp_neurons(
        lm, PROMPT, token_index=n_tok - 1, k=5
    )
    assert top_mlp_neurons(lm, PROMPT, token_index=0, k=300) != top_mlp_neurons(
        lm, PROMPT, token_index=-1, k=300
    )
    assert len(top_mlp_neurons(lm, PROMPT, k=10_000)) == 2 * 128


def test_normalize_per_layer():
    x = torch.tensor([[1., 3., 2.], [5., 5., 5.]])
    assert torch.allclose(normalize_per_layer(x), torch.tensor([[0., 1., .5], [0., 0., 0.]]))


def test_normalize_global():
    x = torch.tensor([[1., 3.], [5., 1.]])
    assert torch.allclose(normalize_global(x), torch.tensor([[0., .5], [1., 0.]]))
    assert torch.all(normalize_global(torch.full((2, 2), 3.0)) == 0)


# --- Fix wave: I6(a) Gram matrisleri, Minor 1 (çift BOS), Minor 2 (use_cache) ----------------------


def test_gram_cache_is_used_and_matches(lm, monkeypatch):
    import neshter.activity as act

    calls = []
    real = act.head_grams
    monkeypatch.setattr(act, "head_grams", lambda *a: calls.append(1) or real(*a))
    grams: dict = {}
    first = head_output_power(lm, PROMPT, grams=grams)
    assert len(calls) == 2 and set(grams) == {0, 1}  # katman başına bir kez
    assert all(g.shape == (4, 16, 16) and g.dtype == torch.float32 for g in grams.values())
    second = head_output_power(lm, "başka bir istek", grams=grams)
    assert len(calls) == 2  # önbellekten
    assert second.shape[:2] == (2, 4)
    assert torch.allclose(first, head_output_power(lm, PROMPT), atol=1e-5)


def test_gram_power_matches_explicit_norm_for_all_layers(lm):
    caps = {}
    hs = [
        layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: caps.setdefault(i, a[0].detach().clone()))
        for i, layer in enumerate(lm.model.model.layers)
    ]
    try:
        with torch.no_grad():
            lm.model(**lm.tokenizer(PROMPT, return_tensors="pt"))
    finally:
        for h in hs:
            h.remove()
    got = head_output_power(lm, PROMPT)
    d = lm.arch.head_dim
    for i, layer in enumerate(lm.model.model.layers):
        w = layer.self_attn.o_proj.weight
        exp = torch.stack([(caps[i][0, :, k * d : (k + 1) * d] @ w[:, k * d : (k + 1) * d].T).norm(dim=-1) for k in range(4)])
        assert torch.allclose(got[i], exp, atol=1e-4)


def test_add_special_tokens_is_forwarded(lm, monkeypatch):
    seen = []
    real = lm.tokenizer.__class__.__call__

    def spy(self, *a, **kw):
        seen.append(kw.get("add_special_tokens"))
        return real(self, *a, **kw)

    monkeypatch.setattr(lm.tokenizer.__class__, "__call__", spy)
    head_output_power(lm, PROMPT, add_special_tokens=False)
    tokens(lm, PROMPT, add_special_tokens=False)
    assert seen == [False, False]


def test_activity_forwards_disable_kv_cache(lm, monkeypatch):
    seen = []
    real = lm.model.model.forward

    def spy(*a, **kw):
        seen.append(kw.get("use_cache"))
        return real(*a, **kw)

    monkeypatch.setattr(lm.model.model, "forward", spy)
    head_output_power(lm, PROMPT)
    top_mlp_neurons(lm, PROMPT, k=3)
    assert seen == [False, False]


def test_activity_and_head_means_skip_lm_head(lm, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("lm_head çağrıldı")

    monkeypatch.setattr(lm.model.lm_head, "forward", boom)
    head_output_power(lm, "merhaba dünya")
    top_mlp_neurons(lm, "merhaba dünya")
    head_means(lm, ["merhaba dünya"])


def test_activation_diff_does_not_add_second_bos(lm, monkeypatch):
    import neshter.concepts.hunt as hunt_mod
    from neshter.concepts.model import load_preset

    seen = []
    real = hunt_mod.head_output_power

    def spy(lm_, prompt, add_special_tokens=True, grams=None):
        seen.append((add_special_tokens, grams is not None))
        return real(lm_, prompt, add_special_tokens, grams)

    monkeypatch.setattr(hunt_mod, "head_output_power", spy)
    c = load_preset("code")
    small = c.model_copy(update={"positive": c.positive[:2], "negative": c.negative[:2]})
    hunt_mod.activation_diff(lm, small)
    assert lm.tokenizer.chat_template  # tiny tokenizer'ın şablonu var → özel token eklenmez
    assert seen == [(False, True)] * 4
