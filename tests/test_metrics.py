import math
import random
import threading

import pytest
import torch
from conftest import logits

from neshter.interventions import HeadAblation, apply
from neshter.metrics import kl_effect, perplexity, rate_diff, real_effect, wilson
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


def test_rate_diff_uncertain_and_certain():
    assert rate_diff(78, 100, 77, 100).uncertain
    d = rate_diff(94, 100, 11, 100)
    assert not d.uncertain and d.hi < 0


def test_rate_diff_formula():
    d = rate_diff(60, 100, 40, 100)
    se = math.sqrt(0.6 * 0.4 / 100 + 0.4 * 0.6 / 100)
    assert d.diff == pytest.approx(-0.2)
    assert d.lo == pytest.approx(-0.2 - 1.96 * se)
    assert d.hi == pytest.approx(-0.2 + 1.96 * se)


def test_rate_diff_rejects_bad_counts():
    with pytest.raises(ValueError):
        rate_diff(1, 0, 1, 10)
    with pytest.raises(ValueError):
        rate_diff(11, 10, 1, 10)


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
