import threading

import pytest
import torch
from pydantic import ValidationError

import neshter.concepts.hunt as hunt_mod
from neshter.arch import decoder_layers, o_proj
from neshter.compare import load_suite
from neshter.concepts.detectors import (
    BROKEN_NLL,
    DETECTORS,
    detect_code,
    detect_refusal,
    is_broken,
    repetition,
)
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
from neshter.interventions import HeadAblation, apply
from neshter.metrics import kl_effect, load_neutral, wilson
from neshter.progress import Cancelled


def _small(c: Concept, n: int = 8) -> Concept:
    """Hızlı testler için az örnekli kavram (doğrulayıcı atlanır)."""
    return c.model_copy(update={"positive": c.positive[:n], "negative": c.negative[:n]})


# --- Brief testleri ---------------------------------------------------------------------------


def test_presets_valid_and_ordered():
    for cid in PRESET_ORDER:
        c = load_preset(cid); assert len(c.positive) >= 40 and len(c.negative) >= 40
    assert load_preset("refusal").safety_ablation and PRESET_ORDER[0] == "code"


def test_too_few_examples_rejected():
    with pytest.raises(ValidationError): Concept(id="x", name_tr="x", name_en="x", positive=["a"]*5, negative=["b"]*40, suite_target=None)


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
        assert r.ref_layer == 1 and r.layer_scores[r.ref_layer] == pytest.approx(1)
        assert 0 <= r.val_accuracy <= 1 and r.val_n > 0 and 0 <= r.val_correct <= r.val_n


# --- Ek testler -------------------------------------------------------------------------------


def test_presets_content_rules():
    targets = {"code": "code", "refusal": "harmful_refusal", "math": "math", "tone": None}
    suite_cats = {i.category for i in load_suite()}
    for cid in PRESET_ORDER:
        c = load_preset(cid)
        assert c.id == cid and c.suite_target == targets[cid]
        assert c.suite_target is None or c.suite_target in suite_cats
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
        Concept(id="x", name_tr="x", name_en="x", positive=["a"] * 40, negative=["b"] * 40, suite_target=None, foo=1)


def test_detector_edge_cases():
    assert detect_code("Şöyle:\nfunction f() {}") and detect_code("#include <stdio.h>")
    assert detect_code("class A:\n  pass")
    assert not detect_code("Bu sınıf (class) kavramı nesne yönelimlidir.")
    assert detect_refusal("I'm sorry, but I can't help with that.") and detect_refusal("As an AI, I ...")
    assert detect_refusal("BUNU YAPAMAM.")
    assert not detect_refusal("Sure, here is the code.")


@pytest.mark.parametrize(
    "text,code",
    [
        ("fn main() {", True),
        ("func add(a int) int {", True),
        ("SELECT * FROM t;", True),
        ("import os", True),
        ("#!/bin/bash", True),
        ("public class A {", True),
        ("x = foo(1);", True),
        ("Merhaba, nasılsın?", False),
        ("Önce eve gittim; sonra uyudum;", False),
        ("Bir şey daha: kediler uyur;", False),
    ],
)
def test_code_detector_wider(text, code):
    assert detect_code(text) is code


@pytest.mark.parametrize(
    "text,ref",
    [
        ("I'm unable to help with that.", True),
        ("I must decline.", True),
        ("Maalesef bunu yapamam.", True),
        ("Bunu yapamayacağım.", True),
        ("I cannot do that.", True),
        ("As a language model, I", True),
        ("I apologise, that is not possible.", True),
        ("I am not able to do this.", True),
        ("Size bu konuda yardımcı olamayacağım.", True),
        ("I'm sorry, but I can't help with that.", True),
        ("I won't help with that.", True),
        ("I am sorry, I cannot assist.", True),
        ("I'm unable to comply.", True),
        ("Maalesef bu konuda bilgi veremem.", True),
        ("Maalesef, buna yardımcı olamam.", True),
        ("Üzgünüm, bunu yapmayacağım.", True),
        ("Bunu yapamam!", True),
        ("I can't wait to help!", False),
        ("left him unable to work", False),
        ("made him sorry", False),
        ("This app has an AI assistant", False),
        ("Hawaii wont", False),
        ("I'm sorry to hear that! Here is the recipe:", False),
        ("I won't lie, it is easy", False),
        ("Maalesef yarın yağmur var, işte tarif:", False),
        ("yapamamak zor", False),
        ("Maalesef yağmur var. Tamam, işte tarif:", False),
        ("I can't help but smile.", False),
        ("Sure, here it is.", False),
    ],
)
def test_refusal_detector_wider(text, ref):
    assert detect_refusal(text) is ref


def test_is_broken():
    assert is_broken("") and is_broken("a b c d " * 10) and is_broken("ok fine text", nll=9.0)
    assert not is_broken("Paris is the capital of France.")


def test_is_broken_thresholds():
    assert BROKEN_NLL == 7.0
    assert is_broken("   \n ")
    assert not is_broken("ok fine text", nll=7.0) and is_broken("ok fine text", nll=7.0001)
    assert not is_broken("ok fine text", nll=None)
    loop7 = "a a a a a a a"  # 7 kelime: tekrar oranı 0.75 ama kural ≥ 8 kelimede uygulanır
    assert repetition(loop7.split()) > 0.5 and not is_broken(loop7)


def test_repetition():
    assert repetition([]) == 0.0 and repetition(["a", "b", "c"]) == 0.0
    assert repetition("a b c d".split()) == 0.0
    words = ("a b c d " * 10).split()  # 37 dörtlü, 4 farklı
    assert repetition(words) == pytest.approx(1 - 4 / 37)
    assert repetition("one two three four five six seven eight".split()) == 0.0


def test_detectors_registry():
    assert DETECTORS == {"code": detect_code, "refusal": detect_refusal}


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
    # r yığınlı (sola dolgulu) hesaplanır; HF değerleri tek istek: sayısal fark 1e-4'e kadar kabul.
    assert torch.allclose(r[:2, 1], torch.stack([h[0, -1] for h in out.hidden_states[:2]]), atol=1e-4)
    assert torch.allclose(r[2, 1], last["x"][0, -1], atol=1e-4)
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
        assert not hasattr(r, "val_consistency") and 0 <= r.val_accuracy <= 1
        ref = r.ref_layer  # pencere ref'i 1'e zorlar; en büyük d pencerenin dışında olabilir
        assert r.layer_scores[ref] == pytest.approx(1.0)
        if method == "direction":
            assert r.direction_layer == ref and r.direction.shape == (64,)
            assert r.direction.dtype == torch.float32 and r.direction.device.type == "cpu"
        else:
            assert r.direction is None and r.direction_layer is None


# --- v0.1.1: yığınlama, erken durdurma, z yeniden kullanımı, ilerleme/iptal --------------------


def _reference_last_token_z(lm, prompts):
    """Eski tek tek (yığınsız) `_last_token_z`: [N, n_layers, n_heads, head_dim] o_proj girdisi, son token."""
    arch = lm.arch
    out = []
    for p in prompts:
        zs = {}

        def make_hook(layer):
            def hook(module, args):
                zs[layer] = args[0].detach()[0, -1].reshape(arch.n_heads, arch.head_dim).to("cpu", torch.float32)

            return hook

        handles = [o_proj(lm.model, l).register_forward_pre_hook(make_hook(l)) for l in range(arch.n_layers)]
        try:
            with torch.no_grad():
                lm.model.model(**hunt_mod._encode(lm, p), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        out.append(torch.stack([zs[l] for l in range(arch.n_layers)]))
    return torch.stack(out)


def test_batched_resid_matches_unbatched(lm):
    prompts = ["a", "merhaba dünya, bugün hava çok güzel ve kediler çatıda uyuyor", "hello world"]
    batched, _ = hunt_mod._last_token_states(lm, prompts)
    single = torch.cat([hunt_mod._last_token_states(lm, [p])[0] for p in prompts], dim=1)
    assert torch.allclose(batched, single, atol=1e-4)


def test_hidden_states_stop_at_skips_later_layers(lm):
    calls = []
    h = decoder_layers(lm.model)[1].register_forward_hook(lambda *a: calls.append(1))
    try:
        states = hunt_mod._hidden_states(lm, hunt_mod._encode_batch(lm, ["merhaba"]), stop_at=1)
    finally:
        h.remove()
    assert len(states) == 2 and calls == []


def test_captured_z_matches_last_token_z(lm):
    ps = ["merhaba dünya", "hello world, this is a test"]
    _, z = hunt_mod._last_token_states(lm, ps, capture_z=True)
    assert z.shape == (2, 2, 4, 16) and z.dtype == torch.float32
    assert torch.allclose(z, _reference_last_token_z(lm, ps), atol=1e-5)


def test_attribution_hunt_reuses_negative_z(lm, monkeypatch):
    real, captured = hunt_mod._last_token_states, []

    def spy(lm_, prompts, capture_z=False, **k):
        captured.append(capture_z)
        return real(lm_, prompts, capture_z=capture_z, **k)

    monkeypatch.setattr(hunt_mod, "_last_token_states", spy)
    hunt_mod.hunt(lm, load_preset("code"), "attribution")
    assert captured.count(True) == 1  # yalnızca negatifler; z için ikinci geçiş yok


def test_hunt_reports_progress_and_cancels(lm):
    seen = []
    hunt_mod.hunt(lm, load_preset("code"), "activation_diff", progress=lambda p, m: seen.append(p))
    assert seen == sorted(seen) and seen[-1] >= 0.9
    ev = threading.Event()
    with pytest.raises(Cancelled):
        hunt_mod.hunt(lm, load_preset("code"), "activation_diff", progress=lambda p, m: ev.set(), cancel=ev)


def test_hunt_attribution_progress_monotonic_and_cancels(lm):
    seen = []
    hunt_mod.hunt(lm, load_preset("code"), "attribution", progress=lambda p, m: seen.append(p))
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    ev = threading.Event()
    with pytest.raises(Cancelled):
        hunt_mod.hunt(lm, load_preset("code"), "attribution", progress=lambda p, m: ev.set(), cancel=ev)


def test_default_method_cpu_is_activation_diff(lm):
    assert hunt_mod.default_method(lm) == "activation_diff"


def test_batched_activation_diff_matches_unbatched(lm, monkeypatch):
    c = _small(load_preset("tone"), n=10)  # 10 > HUNT_BATCH: iki yığın
    batched = activation_diff(lm, c)
    monkeypatch.setattr(hunt_mod, "HUNT_BATCH", 1)
    single = activation_diff(lm, c)
    assert torch.allclose(batched, single, atol=1e-4)


# --- v0.1.1 Task 3: katman penceresi, doğrulama doğruluğu (Wilson), ölçülmüş ilk 20 --------------


def test_candidate_layers():
    assert list(hunt_mod.candidate_layers(24)) == list(range(1, 20))
    assert list(hunt_mod.candidate_layers(2)) == [1]
    assert list(hunt_mod.candidate_layers(1)) == [1]


def test_hunt_ref_layer_stays_in_window(lm, monkeypatch):
    monkeypatch.setattr(hunt_mod, "_cohen_d", lambda *a: torch.tensor([0.0, 0.5, 9.0]))
    assert hunt_mod.hunt(lm, load_preset("code"), "activation_diff").ref_layer == 1


def test_hunt_raises_when_only_out_of_window_layers_separate(lm, monkeypatch):
    monkeypatch.setattr(hunt_mod, "_cohen_d", lambda *a: torch.tensor([0.0, -0.5, 9.0]))
    with pytest.raises(ValueError, match="ayrışmıyor"):
        hunt_mod.hunt(lm, load_preset("code"), "activation_diff")


def test_val_accuracy_counts():
    e = torch.eye(4)[0]
    tp, tn = torch.stack([e, e]), torch.stack([-e, -e])  # eşik 0
    vp, vn = torch.stack([0.5 * e, -0.2 * e]), torch.stack([-0.3 * e])
    assert hunt_mod.val_accuracy(e, tp, tn, vp, vn) == (2, 3)


def test_val_accuracy_threshold_is_train_midpoint():
    e = torch.eye(4)[0]
    tp, tn = torch.stack([3 * e]), torch.stack([1 * e])  # eşik 2
    vp, vn = torch.stack([2.5 * e, 1.5 * e]), torch.stack([2.0 * e, 2.5 * e])  # pos: 2.5 doğru; neg: 2.0 doğru (≤ t)
    assert hunt_mod.val_accuracy(e, tp, tn, vp, vn) == (2, 4)


def test_hunt_reports_val_accuracy_with_wilson(lm):
    r = hunt_mod.hunt(lm, load_preset("code"), "activation_diff")
    assert r.val_ci == wilson(r.val_correct, r.val_n)[1:] and 0 <= r.val_accuracy <= 1
    assert r.val_accuracy == pytest.approx(r.val_correct / r.val_n)
    assert not hasattr(r, "val_consistency")


def test_layer_scores_normalized_over_window(lm, monkeypatch):
    monkeypatch.setattr(hunt_mod, "_cohen_d", lambda *a: torch.tensor([0.0, 0.5, 9.0]))
    scores = hunt_mod.layer_scores(lm, load_preset("code"), load_preset("code"))
    assert scores[2] == 0.0 and scores[1] == 1.0  # tiny modelde pencere range(1, 2)


def _manual_signal_drop(lm, concept, ref, layer, head):
    """measure_top_heads'ten bağımsız: apply() altında tam ileri geçişle elle hesap."""
    train, val = split(concept, seed=0)
    pos, _ = hunt_mod._last_token_states(lm, train.positive)
    neg, _ = hunt_mod._last_token_states(lm, train.negative)
    r_ref = hunt_mod._directions_from(pos, neg)[ref]
    vp = hunt_mod._last_token_states(lm, val.positive)[0][ref] @ r_ref
    vn = hunt_mod._last_token_states(lm, val.negative)[0][ref] @ r_ref
    with apply(lm, [HeadAblation(layer, head)]):
        vp_h = hunt_mod._last_token_states(lm, val.positive)[0][ref] @ r_ref
    return float((vp.mean() - vp_h.mean()) / (vp.mean() - vn.mean()))


def test_measure_top_heads_matches_manual_zeroing(lm):
    c = load_preset("code")
    r = hunt_mod.hunt(lm, c, "attribution")
    assert len(r.measured) == min(20, lm.arch.n_layers * lm.arch.n_heads)
    drops = [m.signal_drop for m in r.measured if m.signal_drop is not None]
    assert drops == sorted(drops, reverse=True)
    n_measured = len(drops)
    assert all(m.signal_drop is None for m in r.measured[n_measured:])  # ölçülmeyenler sonda
    assert any(m.layer < r.ref_layer for m in r.measured)  # tiny modelde ref=1: katman 0 etkili
    est = {(l, h): s for l, h, s in r.heads}
    neutral = load_neutral()[:8]
    for m in r.measured:
        if m.layer < r.ref_layer:
            assert m.signal_drop == pytest.approx(_manual_signal_drop(lm, c, r.ref_layer, m.layer, m.head), abs=1e-5)
        else:
            assert m.signal_drop is None
        assert m.estimated == est[(m.layer, m.head)]
        kl = kl_effect(lm, neutral, [HeadAblation(m.layer, m.head)])
        assert m.neutral_kl == pytest.approx(sum(kl) / len(kl), abs=1e-5)
    assert any(m.signal_drop not in (None, 0.0) for m in r.measured)  # katman 0 head'i gerçekten bir şey değiştirir


def test_measured_heads_at_or_after_ref_layer_have_no_signal_drop(lm):
    # M5: ref katmanı ve sonrası m'yi yapısal olarak etkileyemez; 0.0 ("ölçüldü: etkisiz") değil None.
    r = hunt_mod.hunt(lm, load_preset("code"), "attribution")
    late = [m for m in r.measured if m.layer >= r.ref_layer]
    assert late and all(m.signal_drop is None for m in late)
    assert all(m.neutral_kl >= 0 for m in late)  # nötr KL yine ölçülür
    early = [m for m in r.measured if m.layer < r.ref_layer]
    assert r.measured == early + late  # None en sonda


def test_measure_applies_to_every_method(lm):
    for method in ("activation_diff", "direction"):
        r = hunt_mod.hunt(lm, load_preset("code"), method)
        assert len(r.measured) == 8 and all(m.neutral_kl >= 0 for m in r.measured)


def test_measure_top_heads_respects_k(lm, monkeypatch):
    monkeypatch.setattr(hunt_mod, "MEASURE_TOP_K", 3)
    r = hunt_mod.hunt(lm, load_preset("code"), "activation_diff")
    assert len(r.measured) == 3
    assert {(m.layer, m.head) for m in r.measured} == {(l, h) for l, h, _ in r.heads[:3]}


def test_measure_stage_progress_in_last_slice_and_cancels(lm):
    seen = []
    hunt_mod.hunt(lm, load_preset("code"), "activation_diff", progress=lambda p, m: seen.append(p))
    assert seen == sorted(seen) and all(0 <= p <= 1 for p in seen) and seen[-1] == pytest.approx(1.0)
    assert any(0.8 < p < 1.0 for p in seen)  # ölçüm dilimi (0.8–1.0) ara ilerleme bildirir
    ev = threading.Event()

    def cancel_in_measure(p, m):
        if p > 0.8:
            ev.set()

    with pytest.raises(Cancelled):
        hunt_mod.hunt(lm, load_preset("code"), "activation_diff", progress=cancel_in_measure, cancel=ev)


def test_encode_batch_pads_left_even_if_tokenizer_default_is_right(lm):
    tok = lm.tokenizer
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = hunt_mod._encode_batch(lm, ["kısa", "bu biraz daha uzun bir istek metni"])
    finally:
        tok.padding_side = old
    assert int(enc["attention_mask"][:, -1].min()) == 1  # son sütun gerçek token



# --- Final fix wave M6: activation_diff artık akım geçişlerinin z'sini yeniden kullanır ---------------------


@pytest.mark.parametrize("method", ["activation_diff", "direction"])
def test_hunt_activation_diff_reuses_residual_pass_z(lm, monkeypatch, method):
    c = load_preset("code")
    train, _ = split(c, seed=0)
    expected = activation_diff(lm, train)  # bağımsız: ayrı ileri geçişlerle
    monkeypatch.setattr(hunt_mod, "activation_diff", lambda *a, **k: pytest.fail("ek ileri geçiş yapıldı"))
    r = hunt_mod.hunt(lm, c, method)
    got = {(layer, head): s for layer, head, s in r.heads}
    for layer in range(lm.arch.n_layers):
        for head in range(lm.arch.n_heads):
            assert got[(layer, head)] == pytest.approx(float(expected[layer, head]), abs=1e-5)


def test_hunt_activation_diff_runs_fewer_forward_passes(lm):
    c = load_preset("code")
    train, val = split(c, seed=0)
    calls = []
    h = lm.model.model.register_forward_pre_hook(lambda mod, args: calls.append(1))
    try:
        hunt_mod.hunt(lm, c, "activation_diff")
    finally:
        h.remove()
    batches = lambda n: -(-n // hunt_mod.HUNT_BATCH)  # noqa: E731
    resid = sum(batches(len(x)) for x in (train.positive, train.negative, val.positive, val.negative))
    # artık akım geçişleri + ölçüm (head başına val_pos geçişleri); eskiden train pos/neg bir kez daha geçerdi
    measure_max = hunt_mod.MEASURE_TOP_K * batches(len(val.positive))
    assert len(calls) <= resid + measure_max



def test_sweep_directions_on_model_are_orthonormal_and_bounded(lm):
    from neshter.concepts.hunt import directions, split, sweep_directions

    c = load_preset("code")
    train, val = split(c)
    r1 = directions(lm, train)[1]
    dirs, steps = sweep_directions(lm, train, val, r1, max_k=3, stop_acc=0.0)  # 0: hiç durma, sınıra kadar
    assert 1 <= len(dirs) <= 3
    assert len(steps) == len(dirs)
    q = torch.stack(dirs)
    assert torch.allclose(q @ q.T, torch.eye(len(dirs)), atol=1e-4)
    assert all(0.0 <= s["residual_accuracy"] <= 1.0 for s in steps)


def test_sweep_stops_at_threshold(lm):
    from neshter.concepts.hunt import directions, split, sweep_directions

    train, val = split(load_preset("code"))
    dirs, steps = sweep_directions(lm, train, val, directions(lm, train)[1], max_k=5, stop_acc=1.01)
    assert len(dirs) == 1 and len(steps) == 1


def test_hunt_direction_with_sweeps_returns_extra(lm):
    r = hunt(lm, load_preset("code"), "direction", sweeps=3)
    assert len(r.sweep) >= 1
    assert len(r.extra_directions) == len(r.sweep) - 1 or len(r.extra_directions) == len(r.sweep)


def test_hunt_with_sweeps_reaches_full_progress_only_after_the_sweep(lm):
    seen: list[tuple[float, str]] = []
    hunt(lm, load_preset("code"), "direction", sweeps=3, progress=lambda p, m="": seen.append((p, m)))
    values = [p for p, _ in seen]
    assert values == sorted(values) and values[-1] == 1.0
    first_full = next(i for i, (p, _) in enumerate(seen) if p >= 1.0)
    assert any(m.startswith("Süpürme") for _, m in seen[:first_full + 1])
