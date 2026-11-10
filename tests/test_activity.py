import torch

from neshter.activity import (
    head_output_power,
    normalize_global,
    normalize_per_layer,
    tokens,
    top_mlp_neurons,
)
from neshter.interventions import HeadAblation, apply

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
