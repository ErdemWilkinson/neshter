import math
import random
import threading
from fractions import Fraction

import pytest
import torch
from conftest import logits

from neshter.interventions import HeadAblation, apply
from neshter import metrics as m
from neshter.metrics import kl_effect, perplexity, real_effect, wilson
from neshter.progress import Cancelled


def test_wilson_known_value():
    p, lo, hi = wilson(50, 100)
    assert p == 0.5 and abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_wilson_edges():
    assert wilson(0, 0) == (0.0, 0.0, 1.0)
    p, lo, hi = wilson(0, 10)
    assert p == 0.0 and lo == 0.0 and 0.2 < hi < 0.31
    p, lo, hi = wilson(10, 10)
    assert p == 1.0 and hi == 1.0 and 0.69 < lo < 0.8
    with pytest.raises(ValueError):
        wilson(11, 10)
    with pytest.raises(ValueError):
        wilson(-1, 10)


def test_mcnemar_exact_values():
    assert m.mcnemar_exact(10, 0) == pytest.approx(0.001953125)
    assert m.mcnemar_exact(6, 1) == pytest.approx(0.125)
    assert m.mcnemar_exact(0, 0) == 1.0 and m.mcnemar_exact(3, 3) == 1.0


def _binom_two_sided(b, c):
    """Bağımsız hesap: Binom(n_d, 1/2) altında olasılığı gözlenenden büyük olmayan tüm sonuçların toplamı."""
    nd = b + c
    if nd == 0:
        return Fraction(1)
    probs = [Fraction(math.comb(nd, i), 2**nd) for i in range(nd + 1)]
    obs = probs[b]
    return min(Fraction(1), sum(p for p in probs if p <= obs))


def test_mcnemar_exact_matches_independent_binomial():
    for b in range(0, 16):
        for c in range(0, 16):
            assert m.mcnemar_exact(b, c) == pytest.approx(float(_binom_two_sided(b, c)), rel=1e-12), (b, c)


def test_mcnemar_exact_is_symmetric_and_rejects_negative():
    assert m.mcnemar_exact(7, 2) == m.mcnemar_exact(2, 7)
    with pytest.raises(ValueError):
        m.mcnemar_exact(-1, 2)


def test_paired_ci_regression_values():  # plan tanımlı skor aralığının regresyon değerleri; a=12 b=2 c=1 d=6
    lo, hi = m.paired_ci(12, 2, 1, 6)
    assert lo == pytest.approx(-0.1999, abs=1e-3) and hi == pytest.approx(0.1080, abs=1e-3)


def test_paired_ci_matches_formula_with_wilson():
    a, b, c, d = 9, 5, 2, 4
    n = a + b + c + d
    p1, l1, u1 = wilson(a + b, n)
    p2, l2, u2 = wilson(a + c, n)
    phi = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    delta = p2 - p1
    lo = delta - math.sqrt((p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2)
    hi = delta + math.sqrt((u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2)
    assert m.paired_ci(a, b, c, d) == pytest.approx((lo, hi))


def test_paired_ci_rejects_bad_counts():
    with pytest.raises(ValueError):
        m.paired_ci(0, 0, 0, 0)
    with pytest.raises(ValueError):
        m.paired_ci(1, -1, 0, 0)


def test_paired_diff_all_pass_has_width():
    d = m.paired_diff(25, 0, 0, 0)
    assert d.diff == 0 and d.lo == pytest.approx(-0.1332, abs=1e-3) and d.hi == pytest.approx(0.1332, abs=1e-3)
    assert d.uncertain and d.p_value == 1.0 and d.p_adj is None


def test_paired_diff_clear_drop():
    d = m.paired_diff(15, 10, 0, 0)
    assert d.diff == pytest.approx(-0.4) and d.lo == pytest.approx(-0.5926, abs=1e-3)
    assert d.hi == pytest.approx(-0.1872, abs=1e-3) and not d.uncertain
    assert d.p_value == pytest.approx(2 * 0.5**10)


def test_paired_diff_uncertain_iff_p_at_least_005():
    assert m.paired_diff(10, 6, 1, 3).uncertain  # p = 0.125
    assert not m.paired_diff(10, 6, 0, 3).uncertain  # p = 2/64 = 0.03125
    assert m.paired_diff(10, 5, 0, 3).uncertain  # p = 2/32 = 0.0625


def test_rate_diff_removed():
    assert not hasattr(m, "rate_diff")


def test_holm():
    assert m.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_edge_cases():
    assert m.holm([]) == []
    assert m.holm([0.2]) == [0.2]
    assert m.holm([0.5, 0.001, 0.4, 0.02]) == pytest.approx([0.8, 0.004, 0.8, 0.06])
    assert m.holm([0.9, 0.9]) == [1.0, 1.0]


def test_min_detectable_drop():
    assert m.min_detectable_drop(25) == pytest.approx(0.24)
    assert m.min_detectable_drop(25, m=4) == pytest.approx(0.32)
    assert m.min_detectable_drop(5) == 1.0


def test_min_detectable_drop_matches_brute_force():
    for mm in (1, 2, 3, 4, 7):
        k = next(k for k in range(1, 100) if 2 * Fraction(1, 2) ** k < Fraction(5, 100) / mm)
        assert m.mcnemar_exact(k, 0) < 0.05 / mm <= m.mcnemar_exact(k - 1, 0)
        assert m.min_detectable_drop(40, m=mm) == pytest.approx(min(1.0, k / 40))
    assert m.min_detectable_drop(25, m=0) == m.min_detectable_drop(25, m=1)  # m en az 1


def test_kl_zero_without_intervention(lm):
    assert max(kl_effect(lm, ["a b c d", "x y"], [])) < 1e-6


def test_kl_positive_with_ablation_and_matches_manual(lm):
    prompts = ["bir iki üç dört beş altı yedi", "hello world this is a test"]
    heads = [HeadAblation(0, 1), HeadAblation(1, 2)]
    got = kl_effect(lm, prompts, heads, last_k=3)
    assert len(got) == 2 and all(v > 0 for v in got)
    for prompt, value in zip(prompts, got):
        base = logits(lm, prompt)[0].float()
        with apply(lm, heads):
            cut = logits(lm, prompt)[0].float()
        lp, lq = base.log_softmax(-1)[-3:], cut.log_softmax(-1)[-3:]
        manual = (lp.exp() * (lp - lq)).sum(-1).mean().item()
        assert value == pytest.approx(manual, rel=1e-4, abs=1e-8)


def test_kl_last_k_longer_than_prompt_uses_all_positions(lm):
    a = kl_effect(lm, ["a b c"], [HeadAblation(0, 0)], last_k=1000)
    b = kl_effect(lm, ["a b c"], [HeadAblation(0, 0)], last_k=3)
    n = len(lm.tokenizer("a b c")["input_ids"])
    c = kl_effect(lm, ["a b c"], [HeadAblation(0, 0)], last_k=n)
    assert a == pytest.approx(c)
    assert len(b) == 1


def test_kl_restores_model(lm):
    before = logits(lm, "a b c d")
    kl_effect(lm, ["a b c d"], [HeadAblation(0, 0)])
    assert torch.equal(before, logits(lm, "a b c d"))


def test_kl_rejects_bad_input(lm):
    with pytest.raises(ValueError):
        kl_effect(lm, ["a b"], [], last_k=0)
    with pytest.raises(ValueError):
        kl_effect(lm, [""], [])


def test_real_effect_fields(lm):
    r = real_effect(lm, [HeadAblation(1, 0)], ["bir iki üç dört", "beş altı"], n_random=2)
    assert r.kl > 0 and r.random_kl >= 0 and r.n_prompts == 2


def test_real_effect_mean_std_and_seed(lm):
    prompts = ["bir iki üç dört", "beş altı", "hello world again"]
    heads = [HeadAblation(1, 0)]
    r = real_effect(lm, heads, prompts, n_random=2, seed=7)
    per = kl_effect(lm, prompts, heads)
    mean = sum(per) / len(per)
    assert r.kl == pytest.approx(mean)
    assert r.kl_std == pytest.approx(math.sqrt(sum((v - mean) ** 2 for v in per) / len(per)))
    assert real_effect(lm, heads, prompts, n_random=2, seed=7) == r


def test_real_effect_random_control_excludes_chosen_and_averages(lm, monkeypatch):
    import neshter.metrics as m

    seen: list[list[HeadAblation]] = []
    bases = []
    real = m._kl_from

    def spy(lm_, base, prompts, interventions, last_k):
        seen.append(list(interventions))
        bases.append(base)
        return real(lm_, base, prompts, interventions, last_k)

    monkeypatch.setattr(m, "_kl_from", spy)
    chosen = [HeadAblation(0, 0), HeadAblation(1, 3)]
    prompts = ["bir iki üç", "dört beş"]
    r = m.real_effect(lm, chosen, prompts, n_random=5, seed=1)
    assert len(seen) == 1 + 5
    assert seen[0] == chosen
    assert all(b is bases[0] for b in bases)  # orijinal log-olasılıklar bir kez hesaplanır
    for control in seen[1:]:
        assert len(control) == 2 and len(set(control)) == 2
        assert not set(control) & set(chosen)
        assert sorted(h.layer for h in control) == [0, 1]  # katman eşleşmeli
    means = [sum(vals) / len(vals) for vals in (kl_effect(lm, prompts, c) for c in seen[1:])]
    assert r.random_kl == pytest.approx(sum(means) / 5)
    assert r.random_kls == pytest.approx(means) and r.n_random == 5 and r.layer_matched


def test_real_effect_dedupes_heads(lm):
    prompts = ["bir iki üç dört"]
    a = real_effect(lm, [HeadAblation(1, 0), HeadAblation(1, 0)], prompts, n_random=2, seed=3)
    b = real_effect(lm, [HeadAblation(1, 0)], prompts, n_random=2, seed=3)
    assert a == b


def test_real_effect_small_pool_raises(lm):
    # 2 katman x 4 head = 8 çift; 7'si seçili, havuzda 1 çift kalır: "aynı sayıda" kontrol imkânsız.
    chosen = [HeadAblation(l, h) for l in range(2) for h in range(4)][:7]
    with pytest.raises(ValueError, match="yeterli seçilmemiş head"):
        real_effect(lm, chosen, ["bir iki üç"], n_random=2)


def test_real_effect_rejects_bad_input(lm):
    with pytest.raises(ValueError):
        real_effect(lm, [HeadAblation(0, 0)], [], n_random=2)
    with pytest.raises(ValueError):
        real_effect(lm, [HeadAblation(0, 0)], ["a b"], n_random=0)


def test_perplexity_positive(lm):
    assert perplexity(lm, ["merhaba dünya bugün"]) > 1


def test_perplexity_matches_manual_token_weighted(lm):
    texts = ["merhaba dünya bugün hava", "the quick brown fox jumps"]
    total, count = 0.0, 0
    for t in texts:
        ids = lm.tokenizer(t, return_tensors="pt")["input_ids"]
        lg = logits(lm, t)[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum")
        total += nll.item()
        count += ids.shape[1] - 1
    assert perplexity(lm, texts) == pytest.approx(math.exp(total / count), rel=1e-5)


def test_perplexity_changes_under_intervention(lm):
    texts = ["merhaba dünya bugün hava çok güzel"]
    base = perplexity(lm, texts)
    cut = perplexity(lm, texts, [HeadAblation(0, 0), HeadAblation(1, 1)])
    assert cut != base and perplexity(lm, texts) == pytest.approx(base)


def test_perplexity_rejects_no_predictable_tokens(lm):
    with pytest.raises(ValueError):
        perplexity(lm, [])
    with pytest.raises(ValueError):
        perplexity(lm, ["a"])  # tek token: tahmin edilecek hedef yok


# --- Fix wave: I3/I4 (yön kontrolü, nötr set), Minor 18 (logits_to_keep) ---------------------------


def test_recipe_effect_with_direction_has_random_unit_control(lm, monkeypatch):
    import neshter.metrics as m
    from neshter.interventions import DirectionAblation

    seen = []
    real = m._kl_from
    monkeypatch.setattr(m, "_kl_from", lambda lm_, b, p, iv, k: seen.append(list(iv)) or real(lm_, b, p, iv, k))
    d = DirectionAblation(torch.ones(lm.arch.d_model))
    r = m.recipe_effect(lm, [d, HeadAblation(0, 1)], ["bir iki üç dört beş"], n_random=2, seed=0)
    assert r.kl > 0 and r.random_kl >= 0 and len(r.random_kls) == 2
    for control in seen[1:]:
        dirs = [iv for iv in control if isinstance(iv, DirectionAblation)]
        heads = [iv for iv in control if isinstance(iv, HeadAblation)]
        assert len(dirs) == 1 and len(heads) == 1 and heads[0] != HeadAblation(0, 1)
        assert not torch.allclose(dirs[0].direction, d.direction)
    again = m.recipe_effect(lm, [d, HeadAblation(0, 1)], ["bir iki üç dört beş"], n_random=2, seed=0)
    assert again == r  # tohumlu
    assert r.direction_control == "isotropic" and r.layer_matched  # yön var: etiket dürüst, head'ler yine eşleşmiş
    ctrl, matched = m.random_control(lm, [d, HeadAblation(0, 1)], random.Random(0), torch.Generator().manual_seed(0))
    assert matched and len(ctrl) == 2  # artık (kontrol, eşleşti_mi) çifti döner


def test_effect_does_not_touch_global_seed(lm):
    torch.manual_seed(123)
    random.seed(456)
    t_before, r_before = torch.get_rng_state(), random.getstate()
    real_effect(lm, [HeadAblation(0, 0)], ["bir iki"], n_random=3, seed=5)
    assert torch.equal(t_before, torch.get_rng_state()) and r_before == random.getstate()


def test_neutral_prompts_are_reserved():
    from neshter.compare import load_suite
    from neshter.concepts.model import PRESET_ORDER, load_preset
    from neshter.metrics import load_neutral

    neutral = load_neutral()
    assert len(neutral) >= 32 and len(set(neutral)) == len(neutral)
    suite = {i.prompt for i in load_suite()}
    concepts = {p for cid in PRESET_ORDER for c in [load_preset(cid)] for p in c.positive + c.negative}
    assert not set(neutral) & suite and not set(neutral) & concepts
    assert all(len(p.split()) >= 15 for p in neutral)  # ~20+ token


def test_kl_uses_logits_to_keep(lm, monkeypatch):
    seen = []
    real = lm.model.forward

    def spy(*a, **kw):
        seen.append(kw.get("logits_to_keep"))
        return real(*a, **kw)

    monkeypatch.setattr(lm.model, "forward", spy)
    kl_effect(lm, ["bir iki üç dört beş altı"], [HeadAblation(0, 0)], last_k=3)
    assert seen == [3, 3]


# --- Task 6: katman eşleşmeli kontrol, yüzdelik ve permütasyon p ----------------------------------


def test_layer_matched_control_same_layers(lm):
    import neshter.metrics as m

    chosen = [HeadAblation(0, 0), HeadAblation(0, 1), HeadAblation(1, 3)]
    ctrl, matched = m.layer_matched_heads(lm, chosen, random.Random(0))
    assert matched and sorted(h.layer for h in ctrl) == [0, 0, 1] and not set(ctrl) & set(chosen)


def test_layer_matched_control_falls_back_to_nearest_layer(lm):  # tiny modelde H = 4
    import neshter.metrics as m

    chosen = [HeadAblation(0, 0), HeadAblation(0, 1), HeadAblation(0, 2)]
    ctrl, matched = m.layer_matched_heads(lm, chosen, random.Random(0))
    assert not matched and sorted(h.layer for h in ctrl) == [0, 1, 1]
    assert len(set(ctrl)) == 3 and not set(ctrl) & set(chosen)


def test_layer_matched_control_raises_when_model_too_small(lm):
    import neshter.metrics as m

    chosen = [HeadAblation(l, h) for l in range(2) for h in range(4)][:7]
    with pytest.raises(ValueError, match="yeterli seçilmemiş head"):
        m.layer_matched_heads(lm, chosen, random.Random(0))


def test_effect_percentile_and_p_value(lm, monkeypatch):
    import neshter.metrics as m

    values = iter([[0.5]] + [[0.1]] * 18 + [[0.6], [0.7]])
    monkeypatch.setattr(m, "_kl_from", lambda *a, **k: next(values))
    r = m.real_effect(lm, [HeadAblation(0, 0)], ["bir iki"], n_random=20)
    assert r.p_value == pytest.approx(3 / 21) and r.percentile == pytest.approx(90.0)
    assert r.random_p05 == pytest.approx(0.1) and r.random_p95 == pytest.approx(0.605)
    assert r.n_random == 20 and len(r.random_kls) == 20 and r.random_kl == pytest.approx(sum(r.random_kls) / 20)
    assert r.layer_matched and r.direction_control is None  # yalnız head tarifi
    assert r.kl_se == pytest.approx(r.kl_std / math.sqrt(r.n_prompts))


def test_effect_default_is_30_draws(lm):
    import neshter.metrics as m

    assert m.DEFAULT_N_RANDOM == 30
    r = m.real_effect(lm, [HeadAblation(0, 0)], ["bir iki"])
    assert r.n_random == 30 and len(r.random_kls) == 30


def test_effect_reports_progress_and_cancels(lm):
    import neshter.metrics as m

    seen = []
    m.real_effect(lm, [HeadAblation(0, 0)], ["bir iki"], n_random=3, progress=lambda p, msg: seen.append(p))
    assert len(seen) >= 3 and seen == sorted(seen) and seen[-1] >= 0.9
    assert all(0.0 <= p <= 1.0 for p in seen)
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        m.real_effect(lm, [HeadAblation(0, 0)], ["bir iki"], n_random=3, cancel=ev)


# --- Final fix wave I8: KL istekleri yığınlanır (sola dolgu + dikkat maskesi), değerler aynı kalır -----------

_MIXED_PROMPTS = [
    "a",  # last_k'dan kısa
    "merhaba dünya",
    "the quick brown fox jumps over the lazy dog " * 6,
    "dikkat kafaları",
    "kediler çatıda uyuyor ve köpekler bahçede oynuyor, " * 3,
    "x y",
    "she sells sea shells by the sea shore",
    "İstanbul'da yağmur yağıyor, şemsiyemi unuttum " * 4,
    "bir iki üç dört beş altı yedi sekiz dokuz on",  # 9. istek: ikinci yığın
    "hello",
]


def _unbatched_reference_kl(lm, prompts, interventions, last_k):
    """Bağımsız başvuru: istek başına tek, dolgusuz ileri geçiş (eski davranış)."""
    out = []
    for p in prompts:
        enc = lm.tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            lp = lm.model(**enc, use_cache=False).logits[0, -last_k:].float().log_softmax(-1)
            with apply(lm, interventions):
                lq = lm.model(**enc, use_cache=False).logits[0, -last_k:].float().log_softmax(-1)
        out.append((lp.exp() * (lp - lq)).sum(-1).mean().item())
    return out


@pytest.mark.parametrize("last_k", [16, 1, 4])
def test_kl_effect_batched_matches_unbatched_reference(lm, last_k):
    from neshter.interventions import DirectionAblation

    g = torch.Generator().manual_seed(3)
    for ivs in ([HeadAblation(0, 1)], [HeadAblation(1, 2), DirectionAblation(torch.randn(64, generator=g))]):
        got = kl_effect(lm, _MIXED_PROMPTS, ivs, last_k=last_k)
        ref = _unbatched_reference_kl(lm, _MIXED_PROMPTS, ivs, last_k)
        assert len(got) == len(ref) and all(r > 0 for r in ref)
        for a, b in zip(got, ref):
            assert a == pytest.approx(b, abs=1e-5, rel=1e-4)


def test_recipe_effect_batched_matches_unbatched_reference(lm):
    heads = [HeadAblation(0, 1)]
    r = m.recipe_effect(lm, heads, _MIXED_PROMPTS, n_random=3, seed=5)
    ref = _unbatched_reference_kl(lm, _MIXED_PROMPTS, heads, 16)
    assert r.kl == pytest.approx(sum(ref) / len(ref), abs=1e-5)
    rng, gen = random.Random(5), torch.Generator().manual_seed(5)
    for got in r.random_kls:
        control, _ = m.random_control(lm, heads, rng, gen)
        ref = _unbatched_reference_kl(lm, _MIXED_PROMPTS, control, 16)
        assert got == pytest.approx(sum(ref) / len(ref), abs=1e-5)


def test_kl_forward_passes_are_batched(lm):
    calls = []
    h = lm.model.register_forward_pre_hook(lambda mod, args, kwargs: calls.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
    try:
        kl_effect(lm, _MIXED_PROMPTS, [HeadAblation(0, 1)])
    finally:
        h.remove()
    # 10 istek, yığın 8: orijinal 2 + müdahaleli 2 geçiş (eskiden 10 + 10)
    assert calls == [8, 2, 8, 2]


def test_kl_effect_batched_matches_unbatched_reference_gemma2(gemma_lm):
    ivs = [HeadAblation(0, 1), HeadAblation(1, 3)]
    got = kl_effect(gemma_lm, _MIXED_PROMPTS, ivs)
    ref = _unbatched_reference_kl(gemma_lm, _MIXED_PROMPTS, ivs, 16)
    for a, b in zip(got, ref):
        assert a == pytest.approx(b, abs=1e-5, rel=1e-4)


# --- Plan 2 Task 4: sıralı Besag–Clifford permütasyon testi (/api/effect) ------------------------------

from dataclasses import asdict  # noqa: E402

from neshter.metrics import (  # noqa: E402
    CPU_N_MAX_CAP,
    DEFAULT_N_MAX,
    MIN_N_MAX,
    TAU,
    futility_k,
    recipe_effect,
    sequential_permutation,
    sequential_real_effect,
)

PROMPTS = ["bir iki üç dört", "beş altı yedi", "hello world again", "merhaba dünya"]


def test_sequential_constants():
    assert (DEFAULT_N_MAX, MIN_N_MAX, CPU_N_MAX_CAP, TAU) == (39, 19, 199, 0.05)
    assert m.DEFAULT_N_RANDOM == 30  # suite yolu sabit çekiş kalır


def test_futility_k():
    assert futility_k(39, 0.05) == 1 and futility_k(19, 0.05) == 0
    assert futility_k(199, 0.05) == 9 and futility_k(39, 0.025) == 0


def test_sequential_stops_at_second_exceedance():
    out, seen = sequential_permutation(1.0, iter([0.1, 2.0, 0.2, 3.0] + [0.0] * 100), 39)
    assert (out.n_drawn, out.k_exceed, out.p_value, out.p_kind) == (4, 2, 0.5, "besag_clifford")
    assert out.stopped_early and not out.significant and len(seen) == 4


def test_sequential_full_run_no_exceedance():
    out, _ = sequential_permutation(1.0, iter([0.0] * 39), 39)
    assert out.p_value == pytest.approx(1 / 40) and out.significant and out.p_kind == "full"
    assert out.n_drawn == 39 and out.k_exceed == 0 and not out.stopped_early


def test_one_exceedance_is_boundary_significant():
    out, _ = sequential_permutation(1.0, iter([2.0] + [0.0] * 38), 39)
    assert out.p_value == pytest.approx(0.05) and out.significant and not out.stopped_early


def test_ties_count_as_exceedance():
    out, _ = sequential_permutation(1.0, iter([1.0, 1.0] + [0.0] * 37), 39)
    assert out.stopped_early


def test_draws_not_consumed_after_stop():
    pulled = []
    gen = (pulled.append(v) or v for v in [5.0, 5.0, 5.0])
    sequential_permutation(1.0, gen, 39)
    assert len(pulled) == 2


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("n_max,tau", [(19, 0.05), (39, 0.05), (39, 0.025), (199, 0.05)])
def test_early_stop_decision_matches_full_run(seed, n_max, tau):
    """Erken durma kararı, tüm n_max çekişin (1+k)/(1+n_max) <= tau kararıyla aynıdır."""
    rng = random.Random(seed)
    target = rng.random()
    draws = [rng.random() ** 3 for _ in range(n_max)]  # çoğu hedefin altında: iki yön de görülsün
    out, seen = sequential_permutation(target, iter(draws), n_max, tau)
    full_k = sum(d >= target for d in draws)
    assert out.significant == ((1 + full_k) / (1 + n_max) <= tau)
    assert seen == draws[: out.n_drawn]
    if out.stopped_early:
        assert out.p_kind == "besag_clifford" and out.p_value == out.k_exceed / out.n_drawn
        assert out.k_exceed == futility_k(n_max, tau) + 1 and out.n_drawn < n_max
    else:  # tüm çekişler yapıldıysa tam p raporlanır (son çekişte aşım olsa bile)
        assert out.n_drawn == n_max and out.p_kind == "full" and out.p_value == (1 + full_k) / (1 + n_max)


def test_sequential_permutation_rejects_short_iterable_and_bad_tau():
    with pytest.raises(ValueError):
        sequential_permutation(1.0, iter([0.0] * 5), 39)
    with pytest.raises(ValueError):
        sequential_permutation(1.0, iter([0.0] * 39), 39, tau=0.01)  # 1/40 > 0.01: hiçbir zaman anlamlı olamaz


def test_sequential_real_effect_prefix_matches_recipe_effect(lm):
    s = sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:4], n_max=19, seed=3)
    r = recipe_effect(lm, [HeadAblation(0, 1)], PROMPTS[:4], n_random=19, seed=3)
    assert s.kl == pytest.approx(r.kl) and s.random_kls == pytest.approx(r.random_kls[: s.n_drawn])
    k = sum(v >= s.kl for v in s.random_kls)
    assert s.k_exceed == k and s.n_max == 19 and s.seed == 3 and s.tau == 0.05
    assert s.p_min == pytest.approx(1 / 20) and s.layer_matched


def test_sequential_real_effect_fields_and_band(lm, monkeypatch):
    """Bant ve yüzdelik yalnız çekilen değerler üzerinden; n_drawn < 2 ise None."""
    calls = iter([0.0] + [5.0] * 100)  # hedef 0.0; ilk rastgele 5.0 >= 0.0: n_max=19'da ilk aşımda durur
    real_kl_from = m._kl_from

    def fake(lm_, base, prompts, interventions, last_k, cancel=None):
        real_kl_from(lm_, base, prompts, interventions, last_k, cancel)
        return [next(calls)] * len(prompts)

    monkeypatch.setattr(m, "_kl_from", fake)
    s = sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:2], n_max=19)
    assert (s.n_drawn, s.k_exceed, s.p_kind, s.stopped_early, s.significant) == (1, 1, "besag_clifford", True, False)
    assert s.p_value == 1.0 and s.random_kls == [5.0]
    assert s.random_kl is None and s.random_p05 is None and s.random_p95 is None and s.percentile is None


def test_sequential_real_effect_same_seed_same_result(lm):
    a = sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:2], n_max=19, seed=5)
    b = sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:2], n_max=19, seed=5)
    assert asdict(a) == asdict(b)


def test_sequential_rejects_small_n_max(lm):
    with pytest.raises(ValueError):
        sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:2], n_max=18)


def test_sequential_cancel_raises(lm):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:2], n_max=19, cancel=ev)


def test_sequential_progress_message(lm):
    seen = []
    s = sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:2], n_max=19, progress=lambda p, msg: seen.append((p, msg)))
    msgs = [msg for _, msg in seen if msg.startswith("Rastgele kontrol")]
    assert msgs[0] == "Rastgele kontrol 1/19 (erken durabilir)" and len(msgs) == s.n_drawn
    assert [p for p, _ in seen] == sorted(p for p, _ in seen) and all(0 <= p <= 1 for p, _ in seen)
