import copy
import dataclasses

import pytest
import torch

from neshter.graph import (
    EdgeCut,
    GraphUnsupported,
    answer_graph,
    attention_patterns,
    cut_report,
    logit_lens,
    run_with_cuts,
)

PROMPT = "hello world, this is a tiny test"


@pytest.fixture(scope="module")
def lm64(lm):
    """Türev denetimi için float64 kopya (doğrusallaştırma hatası yuvarlamaya karışmasın)."""
    return dataclasses.replace(lm, model=copy.deepcopy(lm.model).to(torch.float64))


def test_logit_lens_shape_and_final_matches_model(lm):
    r = logit_lens(lm, PROMPT)
    assert len(r["layers"]) == lm.arch.n_layers + 1
    assert r["layers"][0]["layer"] == -1
    enc = lm.tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        y = int(lm.model(**enc).logits[0, -1].argmax())
    assert r["final"]["token_id"] == y
    assert r["layers"][-1]["rank_final"] == 0
    assert all(0 <= row["p_final"] <= 1 for row in r["layers"])


def test_attention_rows_sum_to_one_and_are_causal(lm):
    r = attention_patterns(lm, PROMPT, 1)
    pats = torch.tensor(r["patterns"])
    T = len(r["tokens"])
    assert pats.shape == (lm.arch.n_heads, T, T)
    assert torch.allclose(pats.sum(-1), torch.ones(lm.arch.n_heads, T), atol=1e-3)
    assert float(pats.triu(1).abs().max()) == 0.0


def test_attention_layer_out_of_range(lm):
    with pytest.raises(ValueError):
        attention_patterns(lm, PROMPT, lm.arch.n_layers)


def test_graph_shape_and_pruning(lm):
    g = answer_graph(lm, PROMPT, heads_per_layer=4, neurons_per_layer=6, max_nodes=50, max_edges=80)
    j = g.json
    ids = {n["id"] for n in j["nodes"]}
    assert "out" in ids
    assert {f"t:{t['pos']}" for t in j["tokens"]} <= ids
    assert 0 < len(j["edges"]) <= 80
    for e in j["edges"]:
        assert e["src"] in ids and e["dst"] in ids
        assert e["channel"] in ("q", "v", "resid", "logit")
    weights = [abs(e["weight"]) for e in j["edges"]]
    assert weights == sorted(weights, reverse=True)
    assert len(j["rivers"]["trunk"]) == lm.arch.n_layers + 1
    assert j["metric"]["token_id"] == j["candidates"][0]["token_id"]


def test_graph_target_token(lm):
    g = answer_graph(lm, PROMPT, target_token_id=5, max_edges=20)
    assert g.y == 5 and g.json["metric"]["token_id"] == 5


def test_no_cut_equals_clean_metric(lm):
    g = answer_graph(lm, PROMPT, max_edges=20)
    assert run_with_cuts(lm, PROMPT, [], g.y)["metric"] == pytest.approx(g.metric, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("channel", ["q", "v", "resid", "logit"])
def test_edge_weight_is_derivative_of_path_cut(lm64, channel):
    """e(A→B) = −dM/dε, B'nin c kanalı x − ε·a_A okurken (spec §3 doğrusallaştırması)."""
    g = answer_graph(lm64, PROMPT, heads_per_layer=4, neurons_per_layer=8, max_nodes=100, max_edges=5000, edge_threshold_frac=0.0)
    picked = [e for e in g.edges if e["channel"] == channel][:3]
    assert picked, f"{channel} kanalında kenar yok"
    eps = 1e-2  # RMSNorm float64 modelde de içeride float32 hesaplar: küçük adım gürültüde kaybolur
    for e in picked:
        plus = run_with_cuts(lm64, PROMPT, [EdgeCut(e["src"], e["dst"], channel, eps)], g.y)["metric"]
        minus = run_with_cuts(lm64, PROMPT, [EdgeCut(e["src"], e["dst"], channel, -eps)], g.y)["metric"]
        slope = (plus - minus) / (2 * eps)
        assert slope == pytest.approx(-e["weight"], rel=2e-2, abs=1e-5), e


def test_cut_report_changes_answer_and_kl_is_nonnegative(lm):
    g = answer_graph(lm, PROMPT, max_edges=40)
    top = [g.edges[0]["id"], g.edges[1]["id"]]
    r = cut_report(lm, g, top)
    assert r["kl"] >= -1e-6
    assert r["metric_before"] == pytest.approx(g.metric, rel=1e-4, abs=1e-5)
    assert r["delta_metric"] != 0
    with pytest.raises(KeyError):
        cut_report(lm, g, ["e99999"])


def test_cut_validation(lm):
    with pytest.raises(ValueError):
        run_with_cuts(lm, PROMPT, [EdgeCut("h:1:0", "h:0:0", "q")], 0)  # sonra gelen kaynak
    with pytest.raises(ValueError):
        run_with_cuts(lm, PROMPT, [EdgeCut("h:0:0", "h:1:0", "resid")], 0)  # head'in resid kanalı yok


def test_gemma2_is_rejected(gemma_lm):
    with pytest.raises(GraphUnsupported):
        answer_graph(gemma_lm, PROMPT)


def test_joint_cuts_in_one_layer_add_up(lm64):
    """Aynı katmandaki iki okuyucu birlikte kesilince türevler toplanır (kesim kodu saklı girdiyi bozmamalı)."""
    g = answer_graph(lm64, PROMPT, heads_per_layer=4, neurons_per_layer=8, max_nodes=100, max_edges=5000, edge_threshold_frac=0.0)
    for channel in ("resid", "v", "q"):
        es = [e for e in g.edges if e["channel"] == channel]
        first = es[0]
        layer = first["dst"].split(":")[1]
        other = next(e for e in es[1:] if e["dst"] != first["dst"] and e["dst"].split(":")[1] == layer)
        eps = 1e-2

        def m(s):
            cuts = [EdgeCut(e["src"], e["dst"], channel, s) for e in (first, other)]
            return run_with_cuts(lm64, PROMPT, cuts, g.y)["metric"]

        slope = (m(eps) - m(-eps)) / (2 * eps)
        assert slope == pytest.approx(-(first["weight"] + other["weight"]), rel=2e-2, abs=1e-5), channel


def test_graph_reports_every_layer_forward_and_backward(lm):
    seen: list[tuple[float, str]] = []
    answer_graph(lm, PROMPT, progress=lambda p, m: seen.append((p, m)))
    values = [p for p, _ in seen]
    assert values == sorted(values) and values[-1] == 1.0
    L = lm.arch.n_layers
    assert sum(m.startswith("İleri geçiş: katman") for _, m in seen) == L
    assert sum(m.startswith("Geri geçiş: katman") for _, m in seen) == L


def test_graph_cancel_during_backward_stops_the_job(lm):
    import threading

    from neshter.progress import Cancelled

    cancel = threading.Event()

    def report(p, m):
        if m.startswith("Geri geçiş: katman"):
            cancel.set()  # kullanıcı geri geçişin ortasında iptale bastı

    with pytest.raises(Cancelled):
        answer_graph(lm, PROMPT, progress=report, cancel=cancel)
    # kancalar sökülmüş olmalı: aynı model hemen yeniden grafik çizebilir
    assert answer_graph(lm, PROMPT).json["nodes"]


def test_cut_report_counts_a_repeated_edge_once(lm):
    g = answer_graph(lm, PROMPT, max_edges=40)
    eid = g.edges[0]["id"]
    once, twice = cut_report(lm, g, [eid]), cut_report(lm, g, [eid, eid])
    assert twice["edge_ids"] == [eid]
    assert twice["delta_metric"] == pytest.approx(once["delta_metric"], abs=1e-6)
    assert twice["delta_estimated"] == pytest.approx(once["delta_estimated"])
    assert twice["kl"] >= 0.0


def test_empty_prompt_is_a_clear_input_error(lm, monkeypatch):
    import neshter.graph as G

    # A tokenizer can give zero tokens (no chat template, empty text): a ValueError the server turns into 422.
    monkeypatch.setattr(G, "_encode", lambda lm_, p: {"input_ids": torch.zeros(1, 0, dtype=torch.long)})
    with pytest.raises(ValueError, match="boş"):
        answer_graph(lm, "")
    with pytest.raises(ValueError, match="boş"):
        attention_patterns(lm, "", 0)
