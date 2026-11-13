"""Plan 3 M2 uçları: /api/lens, /api/attention, /api/graph (iş), /api/graph/{id}/cut, /api/hunt sweeps."""

import pytest

from tests.test_server import TOKEN, _make, wait_done


@pytest.fixture(scope="module")
def ready(tiny_model_dir):
    c = _make()
    a = {"X-Neshter-Token": TOKEN}
    job = c.post("/api/session/load", json={"path": str(tiny_model_dir), "dtype": "float32"}, headers=a).json()["job_id"]
    wait_done(c, a, job)
    return c, a


def test_lens(ready):
    c, a = ready
    r = c.post("/api/lens", json={"prompt": "merhaba dünya"}, headers=a)
    assert r.status_code == 200
    j = r.json()
    assert len(j["layers"]) == 3 and j["layers"][-1]["rank_final"] == 0
    assert c.post("/api/lens", json={"prompt": "x", "position": 99}, headers=a).status_code == 422
    assert c.post("/api/lens", json={"prompt": "x", "ekstra": 1}, headers=a).status_code == 422


def test_attention(ready):
    c, a = ready
    j = c.post("/api/attention", json={"prompt": "merhaba dünya", "layer": 0}, headers=a).json()
    assert len(j["patterns"]) == 4 and len(j["patterns"][0]) == len(j["tokens"])
    assert c.post("/api/attention", json={"prompt": "x", "layer": 9}, headers=a).status_code == 422


def test_graph_job_then_cut(ready):
    c, a = ready
    job = c.post("/api/graph", json={"prompt": "hello world, this is", "budget": {"max_edges": 40}}, headers=a).json()["job_id"]
    st = wait_done(c, a, job)
    g = st["result"]
    assert g["estimated"] is True and 0 < len(g["edges"]) <= 40
    r = c.post(f"/api/graph/{g['graph_id']}/cut", json={"edge_ids": [g["edges"][0]["id"]]}, headers=a)
    assert r.status_code == 200
    body = r.json()
    assert {"metric_before", "metric_after", "delta_metric", "delta_estimated", "kl", "top_after"} <= set(body)
    assert "log_probs" not in body
    assert c.post(f"/api/graph/{g['graph_id']}/cut", json={"edge_ids": ["e9999"]}, headers=a).status_code == 422
    assert c.post("/api/graph/yok/cut", json={"edge_ids": ["e0"]}, headers=a).status_code == 404


def test_graph_budget_validation(ready):
    c, a = ready
    assert c.post("/api/graph", json={"prompt": "x", "budget": {"max_edges": 5}}, headers=a).status_code == 422


def test_hunt_sweeps(ready):
    c, a = ready
    body = {"concept_id": "code", "method": "direction", "sweeps": 3}
    st = wait_done(c, a, c.post("/api/hunt", json=body, headers=a).json()["job_id"], timeout=120)
    res = st["result"]
    assert len(res["sweep"]) >= 1 and isinstance(res["extra_directions"], list)
    assert c.post("/api/hunt", json={**body, "sweeps": 9}, headers=a).status_code == 422
