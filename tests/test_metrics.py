import math

import pytest
import torch
from conftest import logits

from neshter.interventions import HeadAblation, apply
from neshter.metrics import kl_effect, perplexity, rate_diff, real_effect, wilson


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
    real = m.kl_effect

    def spy(lm_, prompts, interventions, last_k=16):
        seen.append(list(interventions))
        return real(lm_, prompts, interventions, last_k)

    monkeypatch.setattr(m, "kl_effect", spy)
    chosen = [HeadAblation(0, 0), HeadAblation(1, 3)]
    r = m.real_effect(lm, chosen, ["bir iki üç", "dört beş"], n_random=5, seed=1)
    assert len(seen) == 1 + 5
    assert seen[0] == chosen
    for control in seen[1:]:
        assert len(control) == 2 and len(set(control)) == 2
        assert not set(control) & set(chosen)
    prompts = ["bir iki üç", "dört beş"]
    means = [sum(vals) / len(vals) for vals in (real(lm, prompts, c) for c in seen[1:])]
    expected = sum(means) / 5
    assert r.random_kl == pytest.approx(expected)


def test_real_effect_small_pool_samples_what_exists(lm):
    # 2 katman x 4 head = 8 çift; 7'si seçili, havuzda 1 çift kalır.
    chosen = [HeadAblation(l, h) for l in range(2) for h in range(4)][:7]
    r = real_effect(lm, chosen, ["bir iki üç"], n_random=2)
    assert r.n_prompts == 1 and r.random_kl >= 0


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
