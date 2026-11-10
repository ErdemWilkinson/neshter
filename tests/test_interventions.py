
import pytest
import torch
from conftest import logits

from neshter import interventions
from neshter.arch import decoder_layers, down_proj, embed, o_proj
from neshter.interventions import (
    DirectionAblation,
    HeadAblation,
    apply,
    direction_basis,
    head_means,
    writer_bases,
)


def _hook_count(lm):
    mods = [embed(lm.model)] + list(decoder_layers(lm.model))
    mods += [o_proj(lm.model, i) for i in range(lm.arch.n_layers)]
    mods += [down_proj(lm.model, i) for i in range(lm.arch.n_layers)]
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


def test_direction_ablation_mid_layer_residual_and_writers(lm):
    """Katman içi ara rezidüel (MLP'den önceki norm'un girdisi) ve her yazıcının çıktısı r'ye dik."""
    torch.manual_seed(4)
    r = torch.randn(lm.arch.d_model)
    r_hat = r / r.norm()
    mid, writers = [], []

    def grab_input(module, args):
        mid.append(args[0].detach().clone())

    with apply(lm, [DirectionAblation(r)]):
        handles = [
            layer.post_attention_layernorm.register_forward_pre_hook(grab_input)
            for layer in decoder_layers(lm.model)
        ]
        for i in range(lm.arch.n_layers):
            for m in (o_proj(lm.model, i), down_proj(lm.model, i)):
                handles.append(m.register_forward_hook(_grab_first(writers)))
        try:
            logits(lm, "merhaba dünya")
        finally:
            for h in handles:
                h.remove()
    assert len(mid) == lm.arch.n_layers and len(writers) == 2 * lm.arch.n_layers
    for h in mid + writers:
        assert (h @ r_hat).abs().max() < 1e-4
    assert _hook_count(lm) == 0


def test_residual_writers_are_projections_for_all_families(lm, gemma_lm):
    for m in (lm, gemma_lm):
        w = interventions.residual_writers(m.model, m.arch.family)
        expected = [embed(m.model)] + [
            x for i in range(m.arch.n_layers) for x in (o_proj(m.model, i), down_proj(m.model, i))
        ]
        assert [id(x) for x in w] == [id(x) for x in expected]


# --- Gemma2: γ-ölçekli tabanlar (post-norm öncesi izdüşüm) ----------------------------------------


def _gemma_dirs(d_model: int, seed: int = 11) -> list[torch.Tensor]:
    """Birbirine dik OLMAYAN üç yön (ikisi aynı eksene yakın)."""
    g = torch.Generator().manual_seed(seed)
    a, b = torch.randn(d_model, generator=g), torch.randn(d_model, generator=g)
    return [a, a + 0.3 * b, b - 0.5 * a]


def test_tiny_gemma2_loads(gemma_lm):
    assert gemma_lm.arch.family == "gemma2" and gemma_lm.model.config.tie_word_embeddings
    gammas = [
        1 + norm.weight
        for layer in decoder_layers(gemma_lm.model)
        for norm in (layer.post_attention_layernorm, layer.post_feedforward_layernorm)
    ]
    assert all(not torch.allclose(g, torch.ones_like(g)) for g in gammas)  # γ = 1 değil
    assert gammas[0][0] == 0  # katman 0 post-attn γ girdisi tam sıfır


def test_gemma2_hook_residual_orthogonal_to_r(gemma_lm):
    """Gömme çıktısı, her katmanın ara rezidüeli (MLP öncesi) ve çıktısı span(R)'ye dik (head ile birlikte)."""
    m = gemma_lm
    dirs = _gemma_dirs(m.arch.d_model)
    Q = direction_basis(dirs)
    assert Q.shape[1] == 2  # üç yön, iki boyutlu span
    ivs = [DirectionAblation(r) for r in dirs] + [HeadAblation(0, 1), HeadAblation(1, 3)]
    seen = []

    def grab_input(module, args):
        seen.append(args[0].detach().clone())

    with apply(m, ivs):
        handles = [embed(m.model).register_forward_hook(_grab_first(seen))]
        for layer in decoder_layers(m.model):
            handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(grab_input))
            handles.append(layer.register_forward_hook(_grab_first(seen)))
        try:
            logits(m, "merhaba dünya, bugün hava çok güzel")
        finally:
            for h in handles:
                h.remove()
    assert len(seen) == 1 + 2 * m.arch.n_layers
    for h in seen:
        h = h.reshape(-1, m.arch.d_model)
        for r in dirs:
            assert ((h @ (r / r.norm())).abs() <= 1e-4 * h.norm(dim=-1)).all()
        assert ((h @ Q).norm(dim=-1) <= 1e-4 * h.norm(dim=-1)).all()
    assert _hook_count(m) == 0


def test_gemma2_writer_bases_are_gamma_scaled(gemma_lm):
    m = gemma_lm
    dirs = _gemma_dirs(m.arch.d_model)
    bases = writer_bases(m.model, "gemma2", dirs)
    assert [id(x) for x, _ in bases] == [id(x) for x in interventions.residual_writers(m.model, "gemma2")]
    assert torch.allclose(bases[0][1] @ bases[0][1].T, direction_basis(dirs) @ direction_basis(dirs).T, atol=1e-5)
    for i, layer in enumerate(decoder_layers(m.model)):
        for module, norm in (
            (o_proj(m.model, i), layer.post_attention_layernorm),
            (down_proj(m.model, i), layer.post_feedforward_layernorm),
        ):
            (mod, S), = [(x, S) for x, S in bases if x is module]
            want = direction_basis([(1 + norm.weight.detach().float()) * r for r in dirs])
            assert torch.allclose(S @ S.T, want @ want.T, atol=1e-5)  # aynı izdüşüm


def test_gemma2_basis_handles_zero_gamma(gemma_lm):
    torch.manual_seed(0)
    bases = writer_bases(gemma_lm.model, "gemma2", [torch.randn(64)])
    assert len(bases) == 1 + 2 * gemma_lm.arch.n_layers
    for _, S in bases:
        assert torch.isfinite(S).all() and torch.allclose(S.T @ S, torch.eye(S.shape[1]), atol=1e-5)
    # Yalnızca γ'nın sıfır olduğu eksendeki yön: o yazıcı için taban boş (k' = 0), diğerleri değil.
    e0 = torch.zeros(64)
    e0[0] = 1.0
    bases = writer_bases(gemma_lm.model, "gemma2", [e0])
    assert bases[1][1].shape == (64, 0)
    assert all(S.shape == (64, 1) for i, (_, S) in enumerate(bases) if i != 1)


def test_direction_constructor_normalizes():
    d = DirectionAblation(torch.tensor([3.0, 4.0]))
    assert torch.allclose(d.direction.norm(), torch.tensor(1.0))
    with pytest.raises(ValueError):
        DirectionAblation(torch.zeros(4))


def test_direction_hook_handles_tensor_and_tuple_output(lm):
    # transformers 5.x katman çıktısı Tensor ya da tuple olabilir; tip korunmalı.
    d = DirectionAblation(torch.randn(lm.arch.d_model))
    x = torch.randn(2, 5, lm.arch.d_model)
    hook = interventions._direction_hook(d.direction.unsqueeze(1), None)  # [d, 1] taban
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


# --------------------------------------------------------------- span projection (çoklu yön)


def test_direction_basis_orthonormal_and_spans():
    g = torch.Generator().manual_seed(0)
    R = [torch.randn(64, generator=g) for _ in range(3)]
    Q = direction_basis(R)
    assert Q.dtype == torch.float32 and Q.shape == (64, 3)
    assert torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5)
    for r in R:
        r = r / r.norm()
        assert (r - Q @ (Q.T @ r)).norm() < 1e-5


def test_duplicate_directions_equal_single(lm):
    r = torch.randn(lm.arch.d_model, generator=torch.Generator().manual_seed(6))
    assert direction_basis([r, r, 2 * r]).shape[1] == 1
    with apply(lm, [DirectionAblation(r), DirectionAblation(r)]):
        two = logits(lm, "merhaba dünya")
    with apply(lm, [DirectionAblation(r)]):
        one = logits(lm, "merhaba dünya")
    assert torch.isfinite(two).all() and torch.allclose(two, one, atol=1e-6)


@pytest.mark.parametrize("order", ["r r u", "r u r", "u r 2r"])  # bağımlı sütun ortada / başta
def test_direction_basis_dependent_columns_keep_later_directions(order):
    g = torch.Generator().manual_seed(1)
    r, u = torch.randn(64, generator=g), torch.randn(64, generator=g)
    vecs = [{"r": r, "u": u, "2r": 2 * r}[t] for t in order.split()]
    Q = direction_basis(vecs)
    assert Q.shape[1] == 2 and torch.allclose(Q.T @ Q, torch.eye(2), atol=1e-5)
    for v in vecs:
        v = v / v.norm()
        assert (v - Q @ (Q.T @ v)).norm() < 1e-5  # her girdi vektörü tabanın gerdiği uzayda


def test_direction_basis_skips_zero_vectors_and_may_be_empty():
    assert direction_basis([torch.zeros(8)]).shape == (8, 0)
    assert direction_basis([torch.zeros(8), torch.eye(8)[0]]).shape == (8, 1)


def test_correlated_directions_both_removed(lm):
    g = torch.Generator().manual_seed(7)
    d = lm.arch.d_model
    r1 = torch.randn(d, generator=g)
    r1 = r1 / r1.norm()
    u = torch.randn(d, generator=g)
    u = u - (u @ r1) * r1
    u = u / u.norm()
    r2 = 0.8 * r1 + 0.6 * u
    seen = []
    with apply(lm, [DirectionAblation(r1), DirectionAblation(r2)]):
        # Gözlem kancaları apply()'dan SONRA kaydedilir: embed'de silme kancasından sonra çalışır.
        mods = [embed(lm.model), *(decoder_layers(lm.model)[i] for i in range(lm.arch.n_layers))]
        hs = [m.register_forward_hook(_grab_first(seen)) for m in mods]
        try:
            logits(lm, "merhaba dünya, bugün hava güzel")
        finally:
            for h in hs:
                h.remove()
    assert len(seen) == 1 + lm.arch.n_layers
    for h in seen:  # sıralı bileşimde başarısız olur
        h = h.float()
        assert (h @ r1).abs().max() <= 1e-4 * h.norm(dim=-1).max()
        assert (h @ r2).abs().max() <= 1e-4 * h.norm(dim=-1).max()


def test_apply_registers_one_hook_per_writer(lm):
    g = torch.Generator().manual_seed(8)
    ivs = [DirectionAblation(torch.randn(lm.arch.d_model, generator=g)) for _ in range(3)]
    writers = interventions.residual_writers(lm.model, lm.arch.family)
    with apply(lm, ivs):
        assert all(len(m._forward_hooks) == 1 for m in writers)
    assert all(len(m._forward_hooks) == 0 for m in writers)


def test_writer_bases_follow_residual_writers(lm):
    g = torch.Generator().manual_seed(9)
    dirs = [torch.randn(lm.arch.d_model, generator=g) for _ in range(2)]
    got = writer_bases(lm.model, lm.arch.family, dirs)
    writers = interventions.residual_writers(lm.model, lm.arch.family)
    assert [id(m) for m, _ in got] == [id(m) for m in writers]
    ref = direction_basis(dirs)
    for _, Q in got:
        assert Q.shape == (lm.arch.d_model, 2) and torch.allclose(Q @ Q.T, ref @ ref.T, atol=1e-6)
