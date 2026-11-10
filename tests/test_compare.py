import threading
from types import SimpleNamespace

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
from neshter.metrics import holm, mcnemar_exact, min_detectable_drop, paired_diff


def _cr(category, a, b, c, d, broken_orig=0, broken_int=0):
    return CategoryResult(
        category, a + b + c + d, a + b, a + c, a, b, c, d, broken_orig, broken_int, paired_diff(a, b, c, d)
    )


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
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
        torch.cuda.OutOfMemoryError("oom"),
        RuntimeError("[enforce fail at alloc_cpu.cpp:121] DefaultCPUAllocator: not enough memory: you tried to allocate 8 GB"),
        RuntimeError("[enforce fail] DefaultCPUAllocator: can't allocate memory: you tried to allocate 123 bytes"),
    ],
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
    assert all(i.accepted() for i in items if i.check == "contains")


def test_load_suite_custom_path(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text('- {category: a, check: contains, prompt: "x", answer: "y"}\n', encoding="utf-8")
    assert load_suite(p) == [SuiteItem(category="a", prompt="x", check="contains", answer="y")]


def test_summarize_levels():
    good = [_cr("code", 2, 20, 0, 3), _cr("general", 20, 1, 1, 3)]
    compare._apply_holm(good, "code")
    s = summarize(good, "code")
    assert s.level == "good" and s.mde_points == 24
    assert s.tr == (
        "Hedef davranış belirgin şekilde azaldı; diğer kategorilerde anlamlı bir düşüş görülmedi. "
        "Bu test yaklaşık 24 puandan küçük düşüşleri göremez."
    )
    assert s.en == (
        "The target behavior dropped clearly; no other category dropped significantly. "
        "This test cannot detect drops smaller than about 24 points."
    )
    assert "genel yetenekte" not in s.tr
    bad = good + [_cr("math", 5, 12, 0, 8)]
    compare._apply_holm(bad, "code")
    s = summarize(bad, "code")
    assert s.level == "bad" and "math" in s.tr and "math" in s.en


def test_summarize_mixed_and_target_not_dropped():
    target = _cr("code", 10, 83, 1, 6)
    slight = _cr("general", 70, 7, 1, 22)  # -6 puan, p = 18/256
    compare._apply_holm([target, slight], "code")
    assert slight.diff.uncertain and slight.diff.diff < -0.02
    s = summarize([target, slight], "code")
    assert s.level == "mixed" and "general" in s.tr and "general" in s.en
    assert s.tr.endswith("Bu test yaklaşık 6 puandan küçük düşüşleri göremez.")  # n = 100, m = 1
    assert s.en.endswith("This test cannot detect drops smaller than about 6 points.")

    flat = _cr("code", 86, 4, 2, 8)
    s = summarize([flat, slight], "code")
    assert s.level == "bad"
    assert "azalmadı" in s.tr and "did not" in s.en


def test_summarize_boundary_two_points_is_good():
    target = _cr("code", 10, 83, 1, 6)
    exact = _cr("general", 76, 2, 0, 22)  # tam -2 puan
    compare._apply_holm([target, exact], "code")
    assert summarize([target, exact], "code").level == "good"


def test_summarize_unknown_target():
    with pytest.raises(ValueError):
        summarize([_cr("code", 5, 0, 0, 5)], "math")


def test_summarize_uses_holm_and_states_mde():
    target = _cr("code", 2, 20, 0, 3)
    raw = _cr("general", 200, 6, 0, 94)  # -2 puan, ham p = 2/64 = 0.03125
    others = [_cr(name, 20, 1, 1, 3) for name in ("math", "turkish", "harmful_refusal")]
    results = [target, raw, *others]
    assert raw.diff.p_value == pytest.approx(0.03125) and compare._significant_decrease(raw)  # Holm'suz: anlamlı
    compare._apply_holm(results, "code")
    assert target.diff.p_adj is None
    assert raw.diff.p_adj == pytest.approx(4 * 0.03125) and raw.diff.uncertain
    s = summarize(results, "code")
    assert s.level == "good" and s.mde_points == 32
    assert "puandan küçük" in s.tr and "cannot detect" in s.en and "32" in s.tr


def test_summarize_mde_uses_min_n_over_non_target():
    results = [_cr("code", 2, 20, 0, 3), _cr("general", 50, 0, 0, 50), _cr("math", 20, 0, 0, 5)]
    compare._apply_holm(results, "code")
    s = summarize(results, "code")
    assert s.mde_points == round(100 * min_detectable_drop(25, m=2)) == 28


def test_summarize_reports_broken_target_answers():
    target = _cr("code", 2, 20, 0, 3, broken_orig=0, broken_int=3)
    results = [target, _cr("general", 20, 1, 1, 3)]
    compare._apply_holm(results, "code")
    s = summarize(results, "code")
    assert s.tr.endswith("Hedef kategoride 3 cevap bozuk çıktı (bozuk cevaplar başarı sayılmaz).")
    assert s.en.endswith("3 answers in the target category were broken (broken answers never count as success).")
    same = [_cr("code", 2, 20, 0, 3, broken_orig=2, broken_int=2), _cr("general", 20, 1, 1, 3)]
    compare._apply_holm(same, "code")
    assert "bozuk" not in summarize(same, "code").tr


def test_significance_rule_uses_p_not_interval():
    def r(diff, p, p_adj=None):
        return SimpleNamespace(diff=SimpleNamespace(diff=diff, p_value=p, p_adj=p_adj))

    assert not compare._significant_decrease(r(+0.4, 0.002))  # artış hiçbir zaman düşüş değildir
    assert compare._significant_decrease(r(-0.4, 0.002))  # hedef: kendi p'si
    assert not compare._significant_decrease(r(-0.4, 0.002, p_adj=0.06))  # hedef dışı: Holm düzeltmeli p
    assert compare._significant_decrease(r(-0.4, 0.02, p_adj=0.04))


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

    def fake_pair(lm_, prompt, interventions, max_new_tokens=200, seed=42, **kw):
        assert kw == {}  # baked yolu yok: generate_pair'e ek argüman gitmez
        prompts.append((prompt, max_new_tokens))
        return (*outputs[prompt], "batch")

    monkeypatch.setattr(compare, "generate_pair", fake_pair)
    monkeypatch.setattr(compare, "answer_nll", lambda lm_, prompt, answer: 1.0)
    items = [
        SuiteItem(category="code", prompt="p1", check="code"),
        SuiteItem(category="general", prompt="p2", check="contains", answer="Ankara"),
        SuiteItem(category="harmful_refusal", prompt="p3", check="refusal"),
    ]
    seen = []
    res = run_suite(lm, items, [], progress=lambda d, t: seen.append((d, t)), max_new_tokens=5)
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert [p[1] for p in prompts] == [5, 5, 5]
    assert [(r.category, r.n, r.k_orig, r.k_int, r.a, r.b, r.c, r.d) for r in res] == [
        ("code", 1, 1, 0, 0, 1, 0, 0),
        ("general", 1, 0, 1, 0, 0, 1, 0),
        ("harmful_refusal", 1, 1, 0, 0, 1, 0, 0),
    ]
    expected = paired_diff(0, 1, 0, 0)
    expected.p_adj = holm([1.0, 1.0, 1.0])[0]  # hedef yok: tüm kategoriler Holm'a girer
    assert res[0].diff == expected


def test_run_suite_pairs_and_broken(lm, monkeypatch):
    script = {  # prompt -> (orijinal, müdahaleli)
        "c1": ("```x```", ""),  # müdahaleli bozuk (boş) -> hedefte geçti sayılır: a
        "c2": ("```y```", "plain words here"),  # b
        "c3": ("just prose", "def f():\n    pass"),  # c
        "c4": ("prose", "more prose"),  # d
        "g1": ("Ankara", "Ankara " * 8),  # müdahaleli tekrar döngüsü -> bozuk -> hedef dışında kaldı: b
        "g2": ("", "Ankara"),  # orijinal bozuk -> kaldı: c
        "g3": ("Ankara", "Ankara xq"),  # müdahaleli NLL 9 -> bozuk -> kaldı: b
    }
    calls = []

    def fake_pair(lm_, prompt, interventions, max_new_tokens, seed=42, **kw):
        assert kw == {}
        return (*script[prompt], "batch")

    def fake_nll(lm_, prompt, answer):
        calls.append((prompt, answer))
        return 9.0 if answer == "Ankara xq" else 1.0

    monkeypatch.setattr(compare, "generate_pair", fake_pair)
    monkeypatch.setattr(compare, "answer_nll", fake_nll)
    items = [SuiteItem(category="code", prompt=p, check="code") for p in ("c1", "c2", "c3", "c4")]
    items += [SuiteItem(category="general", prompt=p, check="contains", answer="Ankara") for p in ("g1", "g2", "g3")]
    seen = []
    res = run_suite(lm, items, [HeadAblation(0, 0)], progress=lambda d, t: seen.append((d, t)), target="code")
    assert seen == [(i, 7) for i in range(1, 8)]
    code, general = res
    assert (code.a, code.b, code.c, code.d) == (1, 1, 1, 1)
    assert (code.k_orig, code.k_int, code.broken_orig, code.broken_int) == (2, 2, 0, 1)
    assert (general.a, general.b, general.c, general.d) == (0, 2, 1, 0)
    assert (general.k_orig, general.k_int, general.broken_orig, general.broken_int) == (2, 1, 1, 2)
    assert len(calls) == 14 and {p for p, _ in calls} == set(script)  # her iki cevap için NLL
    assert code.diff.p_adj is None and code.diff.p_value == mcnemar_exact(1, 1)
    assert general.diff.p_adj == pytest.approx(mcnemar_exact(2, 1))  # m = 1
    assert general.diff.uncertain


def test_run_suite_holm_over_non_target(lm, monkeypatch):
    script = {f"t{i}": ("```x```", "prose") for i in range(10)}
    script |= {f"g{i}": ("Ankara", "yok") for i in range(6)} | {"g6": ("Ankara", "Ankara")}
    script |= {f"m{i}": ("Ankara", "yok") for i in range(2)}
    monkeypatch.setattr(compare, "generate_pair", lambda lm_, p, iv, max_new_tokens, seed=42: (*script[p], "batch"))
    monkeypatch.setattr(compare, "answer_nll", lambda lm_, prompt, answer: 1.0)
    items = [SuiteItem(category="code", prompt=f"t{i}", check="code") for i in range(10)]
    items += [SuiteItem(category="general", prompt=f"g{i}", check="contains", answer="Ankara") for i in range(7)]
    items += [SuiteItem(category="math", prompt=f"m{i}", check="contains", answer="Ankara") for i in range(2)]
    code, general, math_ = run_suite(lm, items, [], target="code")
    raw = [mcnemar_exact(6, 0), mcnemar_exact(2, 0)]
    assert general.diff.p_value == pytest.approx(raw[0]) and general.diff.p_value < 0.05
    assert [general.diff.p_adj, math_.diff.p_adj] == pytest.approx(holm(raw))
    assert general.diff.p_adj == pytest.approx(0.0625) and general.diff.uncertain  # Holm sonrası belirsiz
    assert code.diff.p_adj is None and not code.diff.uncertain


def test_answer_nll_matches_manual(lm):
    tok = lm.tokenizer
    prompt, answer = "merhaba", "dünya bugün"
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(text, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p_ids + a_ids])
    with torch.no_grad():
        logp = lm.model(input_ids=ids).logits[0].float().log_softmax(-1)
    manual = -sum(logp[len(p_ids) - 1 + i, t].item() for i, t in enumerate(a_ids)) / len(a_ids)
    assert compare.answer_nll(lm, prompt, answer) == pytest.approx(manual, rel=1e-5)
    assert compare.answer_nll(lm, prompt, "") is None


def test_run_suite_nll_runs_with_weights_restored(lm, monkeypatch):
    real = compare.answer_nll

    def n_hooks():
        return sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in lm.model.modules())

    baseline = n_hooks()
    hooked = []

    def spy(lm_, prompt, answer):
        hooked.append(n_hooks() != baseline)  # müdahale kancası takılıyken çağrılmamalı
        return real(lm_, prompt, answer)

    monkeypatch.setattr(compare, "answer_nll", spy)
    items = load_suite()[:2]
    res = run_suite(lm, items, [HeadAblation(0, 0), HeadAblation(1, 1)], max_new_tokens=4, target="code")
    assert len(hooked) == 4 and not any(hooked)
    assert res[0].n == 2


def test_run_suite_real_model_smoke(lm):
    items = load_suite()[:2]
    res = run_suite(lm, items, [HeadAblation(0, 0)], max_new_tokens=4)
    assert len(res) == 1 and res[0].n == 2


# --- Fix wave: I3 (Wilson), Minor 4 (contains) ----------------------------------------------------


def test_category_result_has_wilson_bounds():
    import dataclasses

    from neshter.metrics import wilson

    r = CategoryResult("code", 20, 15, 4, 3, 12, 1, 4, 0, 2, paired_diff(3, 12, 1, 4))
    assert r.orig_ci == wilson(15, 20)[1:] and r.int_ci == wilson(4, 20)[1:]
    d = dataclasses.asdict(r)
    assert {"orig_ci", "int_ci", "a", "b", "c", "d", "broken_orig", "broken_int"} <= set(d)
    assert {"p_value", "p_adj"} <= set(d["diff"])


@pytest.mark.parametrize(
    "answer,text,ok",
    [
        ("istanbul", "İstanbul", True),
        ("İstanbul", "Cevap: İSTANBUL.", True),
        ("12", "120", False),
        ("12", "Bir yılda 12 ay vardır.", True),
        ("12", "3.12", False),
        ("0", "10 derece", False),
        ("0", "0 derece", True),
        ("h2o", "H₂O", True),
        ("doğu", "Güneş doğudan doğar.", True),
        ("bahar", "ilkbahar", False),
        ("stanbul", "İstanbul", False),
    ],
)
def test_answer_matching(answer, text, ok):
    from neshter.compare import answer_matches

    assert answer_matches(answer, text) is ok


def test_contains_accepts_any_of_answers():
    from neshter.compare import _passes

    item = SuiteItem(category="t", prompt="p", check="contains", answers=["ilkbahar", "bahar"])
    assert _passes(item, "Bahar gelir.") and _passes(item, "İlkbahar") and not _passes(item, "Yaz")


def test_suite_answers_not_in_prompt():
    from neshter.compare import answer_matches

    for item in load_suite():
        if item.check == "contains":
            assert not any(answer_matches(a, item.prompt) for a in item.accepted()), item.prompt
