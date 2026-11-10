import pytest
import torch
from conftest import logits

from neshter import interventions
from neshter.arch import decoder_layers, embed, o_proj
from neshter.interventions import DirectionAblation, HeadAblation, apply, head_means


def _hook_count(lm):
    mods = [embed(lm.model)] + list(decoder_layers(lm.model))
    mods += [o_proj(lm.model, i) for i in range(lm.arch.n_layers)]
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in mods)


def _grab_first(store):
    def grab(module, args, output):
        store.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    return grab


def test_head_ablation_changes_then_restores(lm):
    base = logits(lm, "merhaba dünya")
    with apply(lm, [HeadAblation(1, 2)]):
        assert not torch.allclose(logits(lm, "merhaba dünya"), base)
    assert torch.allclose(logits(lm, "merhaba dünya"), base)


def test_hooks_removed_on_exception(lm):
    base = logits(lm, "a b c")
    with pytest.raises(RuntimeError):
        with apply(lm, [HeadAblation(0, 0)]):
            raise RuntimeError("boom")
    assert torch.equal(logits(lm, "a b c"), base)
    assert _hook_count(lm) == 0


def test_batch_index_only_affects_that_row(lm):
    enc = lm.tokenizer(["x y z", "x y z"], return_tensors="pt", padding=True)
    with torch.no_grad(), apply(lm, [HeadAblation(1, 0)], batch_index=1):
        out = lm.model(**enc).logits
    assert torch.allclose(out[0], logits(lm, "x y z")[0], atol=1e-5)
    assert not torch.allclose(out[1], out[0])


def test_direction_batch_index_only_affects_that_row(lm):
    torch.manual_seed(1)
    r = torch.randn(lm.arch.d_model)
    enc = lm.tokenizer(["x y z", "x y z"], return_tensors="pt", padding=True)
    with torch.no_grad(), apply(lm, [DirectionAblation(r)], batch_index=0):
        out = lm.model(**enc).logits
    ref = logits(lm, "x y z")[0]
    assert torch.allclose(out[1], ref, atol=1e-5)
    assert not torch.allclose(out[0], ref)


def test_batch_index_out_of_batch_raises(lm):
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(0, 0)], batch_index=3):
            logits(lm, "a b c")
    assert _hook_count(lm) == 0


def test_direction_ablation_removes_component(lm):
    torch.manual_seed(2)
    r = torch.randn(lm.arch.d_model)
    r_hat = r / r.norm()
    captured = []

    # Müdahalesiz: bileşen gerçekten var (test anlamlı mı?).
    handle = decoder_layers(lm.model)[0].register_forward_hook(_grab_first(captured))
    try:
        logits(lm, "merhaba dünya")
    finally:
        handle.remove()
    assert (captured[0] @ r_hat).abs().max() > 1e-2

    # apply() kancasından SONRA kaydedilen ikinci kanca, müdahale sonrası katman çıktısını görür.
    captured.clear()
    with apply(lm, [DirectionAblation(r)]):
        handle = decoder_layers(lm.model)[0].register_forward_hook(_grab_first(captured))
        try:
            logits(lm, "merhaba dünya")
        finally:
            handle.remove()
    assert (captured[0] @ r_hat).abs().max() < 1e-4


def test_direction_ablation_on_embedding_and_every_layer(lm):
    torch.manual_seed(3)
    r = torch.randn(lm.arch.d_model)
    r_hat = r / r.norm()
    seen = []
    with apply(lm, [DirectionAblation(r)]):
        mods = [embed(lm.model)] + list(decoder_layers(lm.model))
        handles = [m.register_forward_hook(_grab_first(seen)) for m in mods]
        try:
            logits(lm, "merhaba dünya")
        finally:
            for h in handles:
                h.remove()
    assert len(seen) == 1 + lm.arch.n_layers
    for h in seen:
        assert (h @ r_hat).abs().max() < 1e-4


def test_direction_constructor_normalizes():
    d = DirectionAblation(torch.tensor([3.0, 4.0]))
    assert torch.allclose(d.direction.norm(), torch.tensor(1.0))
    with pytest.raises(ValueError):
        DirectionAblation(torch.zeros(4))


def test_direction_hook_handles_tensor_and_tuple_output(lm):
    # transformers 5.x katman çıktısı Tensor ya da tuple olabilir; tip korunmalı.
    d = DirectionAblation(torch.randn(lm.arch.d_model))
    x = torch.randn(2, 5, lm.arch.d_model)
    hook = interventions._direction_hook(d.direction, None)
    out_t = hook(None, (), x)
    out_tuple = hook(None, (), (x, "extra"))
    assert isinstance(out_t, torch.Tensor)
    assert isinstance(out_tuple, tuple) and out_tuple[1] == "extra"
    assert torch.equal(out_t, out_tuple[0])
    assert torch.allclose(out_t, x - (x @ d.direction).unsqueeze(-1) * d.direction, atol=1e-6)


def test_out_of_range(lm):
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(9, 0)]):
            pass
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(0, 4)]):
            pass
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(-1, 0)]):
            pass
    with pytest.raises(ValueError):
        with apply(lm, [DirectionAblation(torch.ones(lm.arch.d_model + 1))]):
            pass


def test_validation_happens_before_any_hook_is_registered(lm):
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(0, 0), HeadAblation(0, 99)]):
            pass
    assert _hook_count(lm) == 0
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(0, 0)], head_mode="mean"):  # head_means yok
            pass
    assert _hook_count(lm) == 0
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(0, 0)], head_mode="bogus"):
            pass
    assert _hook_count(lm) == 0


def test_head_zero_ablation_equals_zeroed_o_proj_columns(lm):
    base = logits(lm, "merhaba dünya")
    w = o_proj(lm.model, 1).weight
    hd = lm.arch.head_dim
    saved = w.data.clone()
    with apply(lm, [HeadAblation(1, 2)]):
        hooked = logits(lm, "merhaba dünya")
    try:
        w.data[:, 2 * hd : 3 * hd] = 0
        zeroed = logits(lm, "merhaba dünya")
    finally:
        w.data.copy_(saved)
    assert torch.allclose(hooked, zeroed, atol=1e-5)
    assert torch.allclose(logits(lm, "merhaba dünya"), base)


def test_head_means_values_and_mean_mode(lm):
    prompts = ["merhaba dünya", "hello world, this is a test"]
    hd = lm.arch.head_dim
    means = head_means(lm, prompts)
    assert set(means) == {(l, h) for l in range(lm.arch.n_layers) for h in range(lm.arch.n_heads)}
    assert all(v.shape == (hd,) for v in means.values())

    # Elle hesap: o_proj girdisinin tüm token'lar üzerinden ortalaması.
    captured = []
    h = o_proj(lm.model, 0).register_forward_pre_hook(lambda m, a: captured.append(a[0].detach().clone()))
    try:
        for p in prompts:
            logits(lm, p)
    finally:
        h.remove()
    allx = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured])
    assert torch.allclose(means[(0, 1)], allx[:, hd : 2 * hd].mean(0), atol=1e-5)
    assert _hook_count(lm) == 0

    # mean modu: o_proj girdisindeki dilim tam olarak ortalamaya eşit olur ve sıfırlamadan farklı sonuç verir.
    seen = []
    with apply(lm, [HeadAblation(0, 1)], head_mode="mean", head_means=means):
        # apply'ın ön-kancasından sonra kaydedildiği için değişmiş girdiyi görür.
        h = o_proj(lm.model, 0).register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
        try:
            mean_out = logits(lm, "merhaba dünya")
        finally:
            h.remove()
    assert torch.allclose(seen[0][..., hd : 2 * hd], means[(0, 1)].expand(1, seen[0].shape[1], hd), atol=1e-6)
    with apply(lm, [HeadAblation(0, 1)]):
        zero_out = logits(lm, "merhaba dünya")
    assert not torch.allclose(mean_out, zero_out)


def test_head_means_rejects_empty(lm):
    with pytest.raises(ValueError):
        head_means(lm, [])
