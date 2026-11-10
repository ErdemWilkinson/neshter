import pytest
import torch
from pydantic import ValidationError

from neshter.concepts.detectors import detect_code, detect_refusal
from neshter.concepts.hunt import (
    activation_diff,
    directions,
    format_prompt,
    head_attribution,
    hunt,
    last_token_resid,
    layer_scores,
)
from neshter.concepts.model import PRESET_ORDER, Concept, load_preset, split


def _small(c: Concept, n: int = 8) -> Concept:
    """Hızlı testler için az örnekli kavram (doğrulayıcı atlanır)."""
    return c.model_copy(update={"positive": c.positive[:n], "negative": c.negative[:n]})


# --- Brief testleri ---------------------------------------------------------------------------


def test_presets_valid_and_ordered():
    for cid in PRESET_ORDER:
        c = load_preset(cid); assert len(c.positive) >= 40 and len(c.negative) >= 40
    assert load_preset("refusal").safety_ablation and PRESET_ORDER[0] == "code"


def test_too_few_examples_rejected():
    with pytest.raises(ValidationError): Concept(id="x", name_tr="x", name_en="x", positive=["a"]*5, negative=["b"]*40, detector=None)


def test_split_disjoint():
    tr, va = split(load_preset("code")); assert not set(tr.positive) & set(va.positive)
    assert len(va.positive) == round(len(load_preset("code").positive) * .2)


def test_detectors():
    assert detect_code("```python\nprint(1)\n```") and detect_code("def f():\n  pass")
    assert not detect_code("Python bir programlama dilidir.")
    assert detect_refusal("Üzgünüm, ama bu konuda yardımcı olamam.") and not detect_refusal("Tabii, işte adımlar:")


def test_hunt_shapes(lm):
    c = load_preset("code")
    for method in ("attribution", "activation_diff", "direction"):
        r = hunt(lm, c, method)
        assert len(r.layer_scores) == 3 and max(r.layer_scores) == pytest.approx(1)
        assert all(0 <= l < 2 and 0 <= h < 4 for l, h, _ in r.heads)
        assert (r.direction is not None) == (method == "direction")


# --- Ek testler -------------------------------------------------------------------------------


def test_presets_content_rules():
    detectors = {"code": "code", "refusal": "refusal", "math": None, "tone": None}
    for cid in PRESET_ORDER:
        c = load_preset(cid)
        assert c.id == cid and c.detector == detectors[cid]
        assert c.safety_ablation == (cid == "refusal")
        allp = c.positive + c.negative
        assert len(set(allp)) == len(allp), f"{cid}: tekrar eden istek var"
    with pytest.raises((KeyError, ValueError)):
        load_preset("../x")


def test_split_sizes_deterministic_and_complete():
    c = load_preset("tone")
    tr, va = split(c, seed=3)
    tr2, va2 = split(c, seed=3)
    assert tr.positive == tr2.positive and va.negative == va2.negative
    assert sorted(tr.positive + va.positive) == sorted(c.positive)
    assert sorted(tr.negative + va.negative) == sorted(c.negative)
    assert len(va.negative) == round(len(c.negative) * 0.2)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Concept(id="x", name_tr="x", name_en="x", positive=["a"] * 40, negative=["b"] * 40, detector=None, foo=1)


def test_detector_edge_cases():
    assert detect_code("Şöyle:\nfunction f() {}") and detect_code("#include <stdio.h>")
    assert detect_code("class A:\n  pass")
    assert not detect_code("Bu sınıf (class) kavramı nesne yönelimlidir.")
    assert detect_refusal("I'm sorry, but I can't help with that.") and detect_refusal("As an AI, I ...")
    assert detect_refusal("BUNU YAPAMAM.")
    assert not detect_refusal("Sure, here is the code.")


def test_last_token_resid_matches_hf_hidden_states_with_chat_template(tiny_model_dir):
    # Ayrı kopya: output_hidden_states=True, transformers 5'te modele kalıcı kancalar ekler;
    # paylaşılan `lm` fikstürünü kirletmesin.
    from neshter.loader import load_model

    ref = load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    prompts = ["merhaba dünya", "hello world"]
    r = last_token_resid(ref, prompts)
    assert r.shape == (3, 2, 64) and r.dtype == torch.float32
    text = format_prompt(ref, prompts[1])
    assert text == ref.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts[1]}], tokenize=False, add_generation_prompt=True
    ) and text != prompts[1]
    from neshter.arch import decoder_layers

    last = {}
    handle = decoder_layers(ref.model)[-1].register_forward_hook(
        lambda mod, args, output: last.__setitem__("x", output[0] if isinstance(output, tuple) else output)
    )
    try:
        with torch.no_grad():
            out = ref.model(**ref.tokenizer(text, return_tensors="pt"), output_hidden_states=True)
    finally:
        handle.remove()
    # 0..L−1: HF hidden_states ile aynı; L: son katman çıktısı (norm ÖNCESİ, HF'nin L'si norm sonrası).
    assert torch.allclose(r[:2, 1], torch.stack([h[0, -1] for h in out.hidden_states[:2]]), atol=1e-5)
    assert torch.allclose(r[2, 1], last["x"][0, -1], atol=1e-5)
    assert not torch.allclose(r[2, 1], out.hidden_states[2][0, -1], atol=1e-3)


def test_no_hooks_left_behind(lm):
    c = _small(load_preset("code"), n=4)
    for method in ("attribution", "activation_diff", "direction"):
        hunt(lm, c, method)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in lm.model.modules())


def test_directions_unit_norm(lm):
    c = _small(load_preset("math"))
    d = directions(lm, c)
    assert d.shape == (3, 64)
    norms = d.norm(dim=-1)
    # Katman 0 (gömme): son token tüm isteklerde aynı şablon token'ı -> fark sıfır -> sıfır yön.
    assert all(n == pytest.approx(1, abs=1e-5) or n == 0 for n in norms.tolist())
    assert norms[1:].min() == pytest.approx(1, abs=1e-5)


def test_layer_scores_separable_val_is_max_one(lm):
    c = _small(load_preset("code"))
    s = layer_scores(lm, c, c)  # train == val: train yönü kendi izdüşümlerini ayırır -> d > 0
    assert len(s) == 3 and max(s) == pytest.approx(1)
    assert s[0] == 0.0  # gömme katmanı ayırt edemez


def test_head_attribution_respects_causality(lm):
    c = _small(load_preset("code"))
    a1 = head_attribution(lm, c, ref_layer=1)  # katman 1 çıktısı sonrası değil: yalnızca katman 0 head'leri etkiler
    assert a1.shape == (2, 4)
    assert torch.all(a1[1] == 0) and a1[0].abs().max() == pytest.approx(1)
    assert torch.all(head_attribution(lm, c, ref_layer=0) == 0)
    a2 = head_attribution(lm, c, ref_layer=2)
    assert a2.abs().max() == pytest.approx(1) and a2[1].abs().sum() > 0


def test_head_attribution_matches_finite_difference(lm):
    """Doğrusal yaklaşım küçük yamada gerçek değişimle örtüşmeli (gradyan doğru alınıyor mu?)."""
    from neshter.arch import decoder_layers, o_proj

    c = _small(load_preset("code"), n=4)
    ref = 2
    r_ref = directions(lm, c)[ref]
    raw = head_attribution(lm, c, ref, normalize=False)
    # μ^neg ve m'yi elle hesapla; her head için z_h <- z_h + ε(μ_h − z_h) yaması ile sonlu fark.
    H, d = lm.arch.n_heads, lm.arch.head_dim

    def run(prompt, patch=None):
        zs = {}

        def make(layer):
            def hook(mod, args):
                x = args[0].clone()
                zs[layer] = x[0, -1].reshape(H, d).clone()
                if patch is not None and patch[0] == layer:
                    _, h, target, eps = patch
                    zl = x[0, -1].reshape(H, d)
                    zl[h] = zl[h] + eps * (target - zl[h])
                    x[0, -1] = zl.reshape(-1)
                return (x,)
            return hook

        last = {}
        hs = [o_proj(lm.model, l).register_forward_pre_hook(make(l)) for l in range(2)]
        # ref = 2 = n_layers -> son decoder katmanının çıktısı (norm öncesi)
        hs.append(decoder_layers(lm.model)[-1].register_forward_hook(
            lambda mod, args, output: last.__setitem__("x", output[0] if isinstance(output, tuple) else output)))
        try:
            with torch.no_grad():
                lm.model.model(**lm.tokenizer(format_prompt(lm, prompt), return_tensors="pt"), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return float(last["x"][0, -1] @ r_ref), zs

    mu = torch.stack([torch.stack([run(p)[1][l] for l in range(2)]) for p in c.negative]).mean(0)
    eps = 1e-3
    for layer, head in [(0, 1), (1, 2)]:
        delta = 0.0
        for p in c.positive:
            base, _ = run(p)
            patched, _ = run(p, (layer, head, mu[layer, head], eps))
            delta += (patched - base) / eps
        expected = -delta / len(c.positive)
        assert raw[layer, head].item() == pytest.approx(expected, rel=0.05, abs=1e-4)


def test_hunt_raises_when_concept_does_not_separate(lm):
    c = load_preset("math")
    same = c.model_copy(update={"positive": c.positive[:10], "negative": c.positive[:10]})
    with pytest.raises(ValueError, match="ayrışmıyor"):
        hunt(lm, same, "direction")


def test_activation_diff_normalized(lm):
    a = activation_diff(lm, _small(load_preset("tone")))
    assert a.shape == (2, 4) and (a >= 0).all() and a.max() == pytest.approx(1)


@pytest.mark.parametrize("cid", PRESET_ORDER)
def test_hunt_all_presets_all_methods(lm, cid):
    c = load_preset(cid)
    # Rastgele tiny modelde eşleştirilmiş refusal çiftleri seed 0'da val'de ayrışmıyor (hunt doğru şekilde
    # ValueError verir); seed 1 dört kavramın hepsinde ayrışıyor.
    for method in ("attribution", "activation_diff", "direction"):
        r = hunt(lm, c, method, seed=1)
        assert len(r.heads) == 8 and len({(l, h) for l, h, _ in r.heads}) == 8
        scores = [s for _, _, s in r.heads]
        assert scores == sorted(scores, reverse=True) and all(type(s) is float for s in scores)
        assert -1 <= r.val_consistency <= 1
        ref = max(range(3), key=lambda i: r.layer_scores[i])
        if method == "direction":
            assert r.direction_layer == ref and r.direction.shape == (64,)
            assert r.direction.dtype == torch.float32 and r.direction.device.type == "cpu"
        else:
            assert r.direction is None and r.direction_layer is None
