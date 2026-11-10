import threading

import pytest
import torch

from neshter import compare
from neshter.compare import (
    CategoryResult,
    SuiteItem,
    generate_pair,
    load_suite,
    run_suite,
    summarize,
)
from neshter.interventions import HeadAblation
from neshter.metrics import rate_diff


def test_pair_identical_without_intervention(lm):
    a, b, mode = generate_pair(lm, "merhaba", [], max_new_tokens=8)
    assert a == b and mode == "batch"


def test_pair_differs_with_strong_intervention(lm):
    heads = [HeadAblation(layer, h) for layer in range(2) for h in range(4)]
    a, b, _ = generate_pair(lm, "merhaba", heads, max_new_tokens=8)
    assert a != b


def _make_failing_generate(lm, error):
    original = lm.model.generate

    def wrapper(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.shape[0] == 2:
            raise error
        return original(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize(
    "error",
    [RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), torch.cuda.OutOfMemoryError("oom")],
)
def test_oom_falls_back_to_sequential(lm, monkeypatch, error):
    interventions = [HeadAblation(0, 1)]
    ref = generate_pair(lm, "merhaba", interventions, max_new_tokens=8)
    assert ref[2] == "batch"
    monkeypatch.setattr(lm.model, "generate", _make_failing_generate(lm, error))
    a, b, mode = generate_pair(lm, "merhaba", interventions, max_new_tokens=8)
    assert mode == "sequential" and (a, b) == ref[:2]


def test_other_errors_propagate(lm, monkeypatch):
    monkeypatch.setattr(lm.model, "generate", _make_failing_generate(lm, RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        generate_pair(lm, "merhaba", [], max_new_tokens=4)


def test_hooks_removed_after_pair(lm):
    before = generate_pair(lm, "merhaba", [], max_new_tokens=6)
    generate_pair(lm, "merhaba", [HeadAblation(0, 0), HeadAblation(1, 2)], max_new_tokens=6)
    assert generate_pair(lm, "merhaba", [], max_new_tokens=6) == before


def test_load_suite_default():
    items = load_suite()
    assert len(items) == 125
    cats = list(dict.fromkeys(i.category for i in items))
    assert cats == ["code", "general", "math", "turkish", "harmful_refusal"]
    for cat in cats:
        assert sum(i.category == cat for i in items) == 25
    checks = {i.category: i.check for i in items}
    assert checks == {
        "code": "code",
        "general": "contains",
        "math": "contains",
        "turkish": "contains",
        "harmful_refusal": "refusal",
    }
    assert all(i.answer for i in items if i.check == "contains")


def test_load_suite_custom_path(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text('- {category: a, check: contains, prompt: "x", answer: "y"}\n', encoding="utf-8")
    assert load_suite(p) == [SuiteItem(category="a", prompt="x", check="contains", answer="y")]


def test_summarize_levels():
    good = [
        CategoryResult("code", 100, 94, 11, rate_diff(94, 100, 11, 100)),
        CategoryResult("general", 100, 78, 77, rate_diff(78, 100, 77, 100)),
    ]
    s = summarize(good, "code")
    assert s.level == "good"
    assert s.tr == "Hedef davranış belirgin şekilde azaldı, genel yetenekte anlamlı bir kayıp görünmüyor."
    assert s.en.startswith("The target behavior dropped clearly")
    bad = good + [CategoryResult("math", 100, 64, 30, rate_diff(64, 100, 30, 100))]
    s = summarize(bad, "code")
    assert s.level == "bad" and "math" in s.tr and "math" in s.en


def test_summarize_mixed_and_target_not_dropped():
    target = CategoryResult("code", 100, 94, 11, rate_diff(94, 100, 11, 100))
    slight = CategoryResult("general", 100, 78, 72, rate_diff(78, 100, 72, 100))
    assert slight.diff.uncertain and slight.diff.diff < -0.02
    s = summarize([target, slight], "code")
    assert s.level == "mixed" and "general" in s.tr and "general" in s.en

    flat = CategoryResult("code", 100, 90, 88, rate_diff(90, 100, 88, 100))
    s = summarize([flat, slight], "code")
    assert s.level == "bad"
    assert "azalmadı" in s.tr and "did not" in s.en


def test_summarize_boundary_two_points_is_good():
    target = CategoryResult("code", 100, 94, 11, rate_diff(94, 100, 11, 100))
    exact = CategoryResult("general", 100, 78, 76, rate_diff(78, 100, 76, 100))  # tam -2 puan
    assert summarize([target, exact], "code").level == "good"


def test_summarize_unknown_target():
    with pytest.raises(ValueError):
        summarize([CategoryResult("code", 10, 5, 5, rate_diff(5, 10, 5, 10))], "math")


def test_run_suite_cancel(lm):
    ev = threading.Event()
    ev.set()
    assert run_suite(lm, load_suite()[:3], [], cancel=ev) == []


def test_run_suite_cancel_between_items(lm):
    ev = threading.Event()
    calls = []

    def progress(done, total):
        calls.append((done, total))
        ev.set()

    items = load_suite()[:3]
    assert run_suite(lm, items, [], progress=progress, cancel=ev, max_new_tokens=4) == []
    assert calls[-1] == (1, 3)


def test_run_suite_counts_and_progress(lm, monkeypatch):
    outputs = {
        "p1": ("```py\nx\n```", "plain"),
        "p2": ("nope", "Ankara!"),
        "p3": ("I cannot help", "sure thing"),
    }
    prompts = []

    def fake_pair(lm_, prompt, interventions, max_new_tokens=200, seed=42):
        prompts.append((prompt, max_new_tokens))
        return (*outputs[prompt], "batch")

    monkeypatch.setattr(compare, "generate_pair", fake_pair)
    items = [
        SuiteItem(category="code", prompt="p1", check="code"),
        SuiteItem(category="general", prompt="p2", check="contains", answer="Ankara"),
        SuiteItem(category="harmful_refusal", prompt="p3", check="refusal"),
    ]
    seen = []
    res = run_suite(lm, items, [], progress=lambda d, t: seen.append((d, t)), max_new_tokens=5)
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert [p[1] for p in prompts] == [5, 5, 5]
    assert [(r.category, r.n, r.k_orig, r.k_int) for r in res] == [
        ("code", 1, 1, 0),
        ("general", 1, 0, 1),
        ("harmful_refusal", 1, 1, 0),
    ]
    assert res[0].diff == rate_diff(1, 1, 0, 1)


def test_run_suite_real_model_smoke(lm):
    items = load_suite()[:2]
    res = run_suite(lm, items, [HeadAblation(0, 0)], max_new_tokens=4)
    assert len(res) == 1 and res[0].n == 2
