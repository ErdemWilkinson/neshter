"""Yerel API sunucusu: güvenlik katmanı, iş kuyruğu ve uçlar (TestClient; gerçek sunucu/ağ yok)."""

import base64
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import recipe_for
from neshter.server import app as app_module
from neshter.server.app import create_app
from neshter.server.jobs import JobManager

PORT = 47999
TOKEN = "test-token-123"


def _make(roots=None):
    app = create_app(TOKEN, PORT, roots={} if roots is None else roots)
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


@pytest.fixture
def token():
    return TOKEN


@pytest.fixture
def port():
    return PORT


@pytest.fixture
def auth():
    return {"X-Neshter-Token": TOKEN}


@pytest.fixture
def client():
    return _make()


def wait_done(client, auth, job_id, timeout=60.0, expect="done"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}", headers=auth).json()
        if st["state"] in ("done", "error", "cancelled"):
            assert st["state"] == expect, st
            return st
        time.sleep(0.05)
    raise AssertionError(f"iş zaman aşımına uğradı: {job_id}")


@pytest.fixture(scope="module")
def ready(tiny_model_dir):
    """Tiny model yüklenmiş sunucu istemcisi (modül boyunca paylaşılır)."""
    c = _make()
    a = {"X-Neshter-Token": TOKEN}
    body = {"path": str(tiny_model_dir), "dtype": "float32"}  # testler float32 (kural); varsayılan bf16
    job = c.post("/api/session/load", json=body, headers=a).json()["job_id"]
    wait_done(c, a, job)
    return c, a


HEAD_OPS = [{"type": "head", "layer": 0, "head": 1}]


# --- Güvenlik -----------------------------------------------------------------------------------


def test_rejects_missing_token(client):
    assert client.get("/api/system").status_code == 403


def test_rejects_wrong_token(client):
    assert client.get("/api/system", headers={"X-Neshter-Token": "yanlis"}).status_code == 403


def test_rejects_non_ascii_token_without_crashing(client):
    r = client.get("/api/system", headers={"X-Neshter-Token": "ş".encode()})
    assert r.status_code == 403


def test_rejects_foreign_origin(client, token):
    r = client.get("/api/system", headers={"X-Neshter-Token": token, "Origin": "https://kotu.example"})
    assert r.status_code == 403


def test_rejects_origin_with_other_port(client, token, port):
    r = client.get("/api/system", headers={"X-Neshter-Token": token, "Origin": f"http://127.0.0.1:{port + 1}"})
    assert r.status_code == 403


def test_accepts_local_origin(client, token, port):
    r = client.get("/api/system", headers={"X-Neshter-Token": token, "Origin": f"http://127.0.0.1:{port}"})
    assert r.status_code == 200
    r = client.get("/api/system", headers={"X-Neshter-Token": token, "Origin": f"http://localhost:{port}"})
    assert r.status_code == 200


def test_accepts_missing_origin(client, auth):
    assert client.get("/api/system", headers=auth).status_code == 200


def test_rejects_foreign_host_header(client, auth):
    r = client.get("/api/system", headers={**auth, "Host": "kotu.example"})
    assert r.status_code == 403
    r = client.get("/api/system", headers={**auth, "Host": f"127.0.0.1:{PORT + 1}"})
    assert r.status_code == 403


def test_accepts_localhost_host_header(client, auth):
    assert client.get("/api/system", headers={**auth, "Host": f"localhost:{PORT}"}).status_code == 200


def test_every_api_route_needs_token(client):
    for method, path in [
        ("get", "/api/models"),
        ("get", "/api/jobs/x"),
        ("get", "/api/jobs/x/events"),
        ("post", "/api/jobs/x/cancel"),
        ("post", "/api/activity"),
        ("get", "/api/yok-boyle-bir-uc"),
    ]:
        assert getattr(client, method)(path).status_code == 403, path


def test_no_cors_headers_and_no_docs(client, auth, port):
    r = client.get("/api/system", headers={**auth, "Origin": f"http://127.0.0.1:{port}"})
    assert "access-control-allow-origin" not in r.headers
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_empty_token_refused():
    with pytest.raises(ValueError):
        create_app("", PORT)


# --- Sistem ve model listesi --------------------------------------------------------------------


def test_system(client, auth):
    r = client.get("/api/system", headers=auth).json()
    assert set(r) == {"gpu_name", "vram_bytes", "ram_bytes"}
    assert r["ram_bytes"] > 0


def test_models_empty_roots(client, auth):
    assert client.get("/api/models", headers=auth).json() == []


def test_models_lists_entries_with_badge(tmp_path, auth):
    (tmp_path / "kucuk.gguf").write_bytes(b"x" * 1000)
    c = _make({"lmstudio": [tmp_path]})
    (entry,) = c.get("/api/models", headers=auth).json()
    assert entry["name"] == "kucuk" and entry["source"] == "lmstudio" and entry["fmt"] == "gguf"
    assert entry["size_bytes"] == 1000
    assert entry["memory_bytes"] > 0
    assert entry["badge"] in ("ok", "slow", "no")
    assert entry["license"] is None


# --- Oturum durumu ------------------------------------------------------------------------------


def test_endpoints_need_loaded_model(client, auth):
    for path, body in [
        ("/api/activity", {"prompt": "a", "token_index": -1, "scale": "layer"}),
        ("/api/mlp", {"prompt": "a", "token_index": -1}),
        ("/api/compare", {"prompt": "a", "recipe_ops": HEAD_OPS}),
        ("/api/effect", {"heads": [[0, 0]]}),
        ("/api/hunt", {"concept_id": "code", "method": "attribution"}),
        ("/api/suite", {"recipe_ops": HEAD_OPS, "target": "code"}),
        ("/api/export", {"recipe_ops": HEAD_OPS, "name": "x", "format": "safetensors", "ollama": False, "out_dir": "x"}),
    ]:
        r = client.post(path, json=body, headers=auth)
        assert r.status_code == 409, path
        assert "model" in r.json()["detail"].lower()


def test_load_then_activity(client, auth, tiny_model_dir):
    job = client.post("/api/session/load", json={"path": str(tiny_model_dir)}, headers=auth).json()["job_id"]
    st = wait_done(client, auth, job)
    assert st["result"]["arch"]["n_layers"] == 2 and st["result"]["name"]
    r = client.post("/api/activity", json={"prompt": "merhaba", "token_index": -1, "scale": "layer"}, headers=auth).json()
    assert len(r["power"]) == 2 and len(r["power"][0]) == 4
    assert len(r["tokens"]) >= 1
    assert all(0.0 <= v <= 1.0 for row in r["power"] for v in row)


def test_load_replaces_and_frees_old_model(client, auth, tiny_model_dir):
    import weakref

    job = client.post("/api/session/load", json={"path": str(tiny_model_dir)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job)
    old = weakref.ref(client.app.state.session.lm)
    job = client.post("/api/session/load", json={"path": str(tiny_model_dir)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job)
    assert client.app.state.session.lm is not None
    assert old() is None  # eski model serbest bırakıldı


def test_load_missing_path_is_422(client, auth, tmp_path):
    r = client.post("/api/session/load", json={"path": str(tmp_path / "yok")}, headers=auth)
    assert r.status_code == 422


def test_load_unsupported_dir_is_job_error(client, auth, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    job = client.post("/api/session/load", json={"path": str(tmp_path)}, headers=auth).json()["job_id"]
    st = wait_done(client, auth, job, expect="error")
    assert "desteklenmiyor" in st["error"]
    assert client.app.state.session.lm is None


def test_activity_global_scale_and_bad_token_index(ready):
    c, a = ready
    r = c.post("/api/activity", json={"prompt": "merhaba dünya", "token_index": 0, "scale": "global"}, headers=a)
    assert r.status_code == 200 and len(r.json()["power"]) == 2
    r = c.post("/api/activity", json={"prompt": "merhaba", "token_index": 999, "scale": "layer"}, headers=a)
    assert r.status_code == 422


def test_activity_rejects_bad_body(ready):
    c, a = ready
    assert c.post("/api/activity", json={"prompt": "x", "token_index": -1, "scale": "kotu"}, headers=a).status_code == 422
    body = {"prompt": "x", "token_index": -1, "scale": "layer", "ekstra": 1}
    assert c.post("/api/activity", json=body, headers=a).status_code == 422


def test_mlp(ready):
    c, a = ready
    r = c.post("/api/mlp", json={"prompt": "merhaba dünya", "token_index": -1}, headers=a).json()
    assert len(r) > 0 and set(r[0]) == {"layer", "neuron", "value"}
    assert r[0]["value"] >= r[-1]["value"]


# --- İşler --------------------------------------------------------------------------------------


def test_effect_job(ready):
    c, a = ready
    body = {"heads": [[0, 1]], "prompts": ["merhaba dünya", "hello world"]}
    job = c.post("/api/effect", json=body, headers=a).json()["job_id"]
    res = wait_done(c, a, job)["result"]
    assert set(res) == {"kl", "kl_std", "random_kl", "n_prompts"} and res["n_prompts"] == 2


def test_effect_rejects_out_of_range_head(ready):
    c, a = ready
    assert c.post("/api/effect", json={"heads": [[9, 0]]}, headers=a).status_code == 422
    assert c.post("/api/effect", json={"heads": []}, headers=a).status_code == 422


def test_hunt_job_serializes_direction_as_base64(ready):
    c, a = ready
    job = c.post("/api/hunt", json={"concept_id": "code", "method": "direction"}, headers=a).json()["job_id"]
    res = wait_done(c, a, job, timeout=120)["result"]
    assert res["direction_shape"] == [64]
    raw = base64.b64decode(res["direction"], validate=True)
    assert len(raw) == 4 * 64
    assert len(res["layer_scores"]) == 3  # gömme çıktısı + 2 katman
    assert len(res["heads"]) == 8 and set(res["heads"][0]) == {"layer", "head", "score"}
    assert res["direction_layer"] in (0, 1)


def test_hunt_without_direction_has_null_direction(ready):
    c, a = ready
    job = c.post("/api/hunt", json={"concept_id": "code", "method": "activation_diff"}, headers=a).json()["job_id"]
    res = wait_done(c, a, job, timeout=120)["result"]
    assert res["direction"] is None and res["direction_shape"] is None


def test_hunt_validation(ready):
    c, a = ready
    assert c.post("/api/hunt", json={"method": "attribution"}, headers=a).status_code == 422  # kavram yok
    assert c.post("/api/hunt", json={"concept_id": "yok", "method": "attribution"}, headers=a).status_code == 422
    both = {"concept_id": "code", "concept": {}, "method": "attribution"}
    assert c.post("/api/hunt", json=both, headers=a).status_code == 422
    assert c.post("/api/hunt", json={"concept_id": "code", "method": "kotu"}, headers=a).status_code == 422


def test_compare(ready):
    c, a = ready
    r = c.post("/api/compare", json={"prompt": "merhaba", "recipe_ops": HEAD_OPS, "max_new_tokens": 4}, headers=a)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"original", "modified", "mode"} and body["mode"] in ("batch", "sequential")


def test_compare_rejects_bad_ops(ready):
    c, a = ready
    bad_layer = [{"type": "head", "layer": 99, "head": 0}]
    assert c.post("/api/compare", json={"prompt": "a", "recipe_ops": bad_layer}, headers=a).status_code == 422
    extra = [{"type": "head", "layer": 0, "head": 0, "x": 1}]
    assert c.post("/api/compare", json={"prompt": "a", "recipe_ops": extra}, headers=a).status_code == 422
    vec = base64.b64encode(b"\x00\x00\x80\x3f" * 3).decode()
    wrong_dim = [{"type": "direction", "d_model": 3, "vector_b64": vec}]
    assert c.post("/api/compare", json={"prompt": "a", "recipe_ops": wrong_dim}, headers=a).status_code == 422


def test_suite_job_with_stub_runner(ready, monkeypatch):
    from neshter.compare import CategoryResult
    from neshter.metrics import rate_diff

    def fake(lm, items, interventions, progress=None, cancel=None, max_new_tokens=200):
        progress(1, 2)
        progress(2, 2)
        return [
            CategoryResult("code", 10, 10, 2, rate_diff(10, 10, 2, 10)),
            CategoryResult("general", 10, 9, 9, rate_diff(9, 10, 9, 10)),
        ]

    monkeypatch.setattr(app_module, "run_suite", fake)
    c, a = ready
    job = c.post("/api/suite", json={"recipe_ops": HEAD_OPS, "target": "code"}, headers=a).json()["job_id"]
    res = wait_done(c, a, job)["result"]
    assert [r["category"] for r in res["categories"]] == ["code", "general"]
    assert res["categories"][0]["diff"]["diff"] < 0
    assert res["summary"]["level"] in ("good", "mixed", "bad")


def test_suite_rejects_unknown_target(ready):
    c, a = ready
    assert c.post("/api/suite", json={"recipe_ops": HEAD_OPS, "target": "yok"}, headers=a).status_code == 422


def test_cancel_job(ready, monkeypatch):
    started = threading.Event()

    def slow(lm, items, interventions, progress=None, cancel=None, max_new_tokens=200):
        started.set()
        for _ in range(400):  # en çok ~20 sn; iptalde hemen çıkar
            if cancel is not None and cancel.is_set():
                return []
            time.sleep(0.05)
        return []

    monkeypatch.setattr(app_module, "run_suite", slow)
    c, a = ready
    job = c.post("/api/suite", json={"recipe_ops": HEAD_OPS, "target": "code"}, headers=a).json()["job_id"]
    assert started.wait(10)
    assert c.get(f"/api/jobs/{job}", headers=a).json()["state"] == "running"
    assert c.post(f"/api/jobs/{job}/cancel", headers=a).status_code == 200
    wait_done(c, a, job, timeout=10, expect="cancelled")


def test_unknown_job_is_404(client, auth):
    assert client.get("/api/jobs/yok", headers=auth).status_code == 404
    assert client.post("/api/jobs/yok/cancel", headers=auth).status_code == 404
    assert client.get("/api/jobs/yok/events", headers=auth).status_code == 404


def test_sse_events_until_done(ready):
    c, a = ready

    def fn(report, cancel):
        report(0.5, "yarı yolda")
        time.sleep(0.2)
        return {"ok": True}

    job = c.app.state.jobs.submit(fn)
    with c.stream("GET", f"/api/jobs/{job}/events", headers=a) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
    assert events[-1] == "done" and "progress" in events
    last = [line for line in text.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(last.removeprefix("data: "))["result"] == {"ok": True}


def test_sse_error_event(ready):
    c, a = ready

    def fn(report, cancel):
        raise RuntimeError("patladı")

    job = c.app.state.jobs.submit(fn)
    with c.stream("GET", f"/api/jobs/{job}/events", headers=a) as r:
        text = "".join(r.iter_text())
    assert "event: error" in text and "patladı" in text


# --- Dışa aktarım -------------------------------------------------------------------------------


def _export_body(out_dir, **kw):
    base = {"recipe_ops": HEAD_OPS, "name": "kesik", "format": "safetensors", "ollama": False, "out_dir": str(out_dir)}
    return {**base, **kw}


def test_export_safetensors(ready, tmp_path):
    c, a = ready
    out = tmp_path / "cikti"
    job = c.post("/api/export", json=_export_body(out), headers=a).json()["job_id"]
    res = wait_done(c, a, job, timeout=120)["result"]
    assert (out / "model.safetensors").is_file() and (out / "recipe.nester.json").is_file()
    assert res["path"] == str(out)


def test_export_refusal_concept_marks_metadata(ready, tmp_path):
    c, a = ready
    out = tmp_path / "ret"
    job = c.post("/api/export", json=_export_body(out, concept="refusal"), headers=a).json()["job_id"]
    wait_done(c, a, job, timeout=120)
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["neshter"]["abliterated"] == "refusal"


def test_export_existing_dir_is_job_error(ready, tmp_path):
    c, a = ready
    out = tmp_path / "var"
    out.mkdir()
    job = c.post("/api/export", json=_export_body(out), headers=a).json()["job_id"]
    st = wait_done(c, a, job, expect="error")
    assert "zaten var" in st["error"]


def test_export_inside_source_is_job_error(ready, tiny_model_dir):
    c, a = ready
    job = c.post("/api/export", json=_export_body(tiny_model_dir / "icerde"), headers=a).json()["job_id"]
    st = wait_done(c, a, job, expect="error")
    assert "kaynak" in st["error"]
    assert not (tiny_model_dir / "icerde").exists()


def test_export_validation(ready, tmp_path):
    c, a = ready
    x = tmp_path / "x"
    assert c.post("/api/export", json=_export_body(x, ollama=True), headers=a).status_code == 422  # GGUF değil
    assert c.post("/api/export", json=_export_body(x, name="../kotu"), headers=a).status_code == 422
    assert c.post("/api/export", json=_export_body(x, format="onnx"), headers=a).status_code == 422
    assert c.post("/api/export", json=_export_body(x, ekstra=1), headers=a).status_code == 422


def test_export_gguf_without_llama_cpp_is_job_error(ready, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "find_llama_cpp", lambda: None)
    c, a = ready
    out = tmp_path / "g"
    job = c.post("/api/export", json=_export_body(out, format="gguf_q8_0"), headers=a).json()["job_id"]
    st = wait_done(c, a, job, expect="error")
    assert "llama.cpp" in st["error"]
    assert not out.exists()


def test_export_gguf_pipeline_with_stubbed_tools(ready, tmp_path, monkeypatch):
    calls = {}

    def fake_to_gguf(hf_dir, out_file, quant, llama_cpp):
        calls.update(hf=(hf_dir / "model.safetensors").is_file(), out=out_file, quant=quant)
        out_file.write_bytes(b"GGUF")
        return out_file

    def fake_ollama(name, gguf):
        calls["ollama"] = (name, gguf)

    monkeypatch.setattr(app_module, "find_llama_cpp", lambda: tmp_path)
    monkeypatch.setattr(app_module, "to_gguf", fake_to_gguf)
    monkeypatch.setattr(app_module, "ollama_create", fake_ollama)
    c, a = ready
    out = tmp_path / "g2"
    body = _export_body(out, format="gguf_q4_k_m", ollama=True)
    res = wait_done(c, a, c.post("/api/export", json=body, headers=a).json()["job_id"], timeout=120)["result"]
    assert calls["hf"] and calls["quant"] == "Q4_K_M"
    assert calls["ollama"][0] == "kesik"
    assert (out / "kesik.gguf").is_file() and not (out / "_hf").exists()
    assert res["path"] == str(out / "kesik.gguf")


# --- Tarif --------------------------------------------------------------------------------------


def _recipe(lm):
    from neshter.interventions import HeadAblation

    return json.loads(recipe_for(lm, [HeadAblation(0, 1)]).model_dump_json())


def test_recipe_check_endpoint(ready, lm):
    c, a = ready
    res = c.post("/api/recipe/check", json={"recipe": _recipe(lm)}, headers=a)
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True and body["errors"] == [] and isinstance(body["sha_mismatch"], bool)


def test_recipe_check_reports_problems(ready, lm):
    c, a = ready
    r = _recipe(lm)
    r["interventions"] = [{"type": "head", "layer": 7, "head": 0}]
    body = c.post("/api/recipe/check", json={"recipe": r}, headers=a).json()
    assert body["ok"] is False and body["errors"]


def test_recipe_check_endpoint_rejects_extra_field(ready, lm):
    c, a = ready
    r = _recipe(lm)
    r["gizli"] = 1
    assert c.post("/api/recipe/check", json={"recipe": r}, headers=a).status_code == 422
    assert c.post("/api/recipe/check", json={"recipe": _recipe(lm), "x": 1}, headers=a).status_code == 422


# --- Örnek model indirme ------------------------------------------------------------------------


def test_sample_model_download_is_mocked(client, auth, tmp_path, monkeypatch):
    seen = {}

    def fake_download(repo_id, **kw):
        seen["repo"] = repo_id
        bar = kw["tqdm_class"](total=100, unit="B")
        bar.update(40)
        bar.update(60)
        return str(tmp_path)

    monkeypatch.setattr(app_module, "snapshot_download", fake_download)
    job = client.post("/api/models/sample", headers=auth).json()["job_id"]
    st = wait_done(client, auth, job)
    assert seen["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert st["result"] == {"path": str(tmp_path)}
    assert st["progress"] == 1.0


# --- Statik dosyalar ----------------------------------------------------------------------------


def test_static_dist_is_mounted_only_if_it_exists(tmp_path, monkeypatch, auth):
    assert _make().get("/").status_code == 404  # dist yok (varsayılan)
    (tmp_path / "index.html").write_text("<h1>ada</h1>", encoding="utf-8")
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path)
    c = _make()
    assert "ada" in c.get("/").text
    assert c.get("/api/system").status_code == 403  # API hâlâ korumalı ve önceliklidir
    assert c.get("/api/system", headers=auth).status_code == 200


# --- JobManager ---------------------------------------------------------------------------------


def _wait_state(jm, jid, states, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        st = jm.status(jid)
        if st.state in states:
            return st
        time.sleep(0.01)
    raise AssertionError(jm.status(jid))


def test_job_runs_and_reports_progress():
    jm = JobManager()
    gate = threading.Event()

    def fn(report, cancel):
        report(0.25, "başladı")
        gate.wait(5)
        return 42

    jid = jm.submit(fn)
    _wait_state(jm, jid, {"running"})
    time.sleep(0.05)
    st = jm.status(jid)
    assert st.progress == 0.25 and st.message == "başladı"
    gate.set()
    st = _wait_state(jm, jid, {"done"})
    assert st.result == 42 and st.error is None and st.progress == 1.0


def test_job_exception_sets_error():
    jm = JobManager()

    def fn(report, cancel):
        raise ValueError("kötü girdi")

    st = _wait_state(jm, jm.submit(fn), {"error"})
    assert st.error == "kötü girdi" and st.result is None


def test_jobs_run_one_at_a_time_in_order():
    jm = JobManager()
    order, active, peak = [], [0], [0]
    lock = threading.Lock()

    def make(i):
        def fn(report, cancel):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            order.append(i)
            with lock:
                active[0] -= 1

        return fn

    ids = [jm.submit(make(i)) for i in range(4)]
    for j in ids:
        _wait_state(jm, j, {"done"})
    assert order == [0, 1, 2, 3] and peak[0] == 1


def test_cancel_running_job_ends_cancelled():
    jm = JobManager()

    def fn(report, cancel):
        while not cancel.is_set():
            time.sleep(0.05)
        return "yarım"

    jid = jm.submit(fn)
    _wait_state(jm, jid, {"running"})
    jm.cancel(jid)
    st = _wait_state(jm, jid, {"cancelled"})
    assert st.result is None


def test_cancel_queued_job_never_runs():
    jm = JobManager()
    gate, ran = threading.Event(), []
    first = jm.submit(lambda report, cancel: gate.wait(5))
    second = jm.submit(lambda report, cancel: ran.append(1))
    _wait_state(jm, first, {"running"})
    jm.cancel(second)
    assert jm.status(second).state == "cancelled"
    gate.set()
    _wait_state(jm, first, {"done"})
    time.sleep(0.1)
    assert ran == [] and jm.status(second).state == "cancelled"


def test_cancel_finished_job_is_noop_and_unknown_raises():
    jm = JobManager()
    jid = jm.submit(lambda report, cancel: 1)
    _wait_state(jm, jid, {"done"})
    jm.cancel(jid)
    assert jm.status(jid).state == "done"
    with pytest.raises(KeyError):
        jm.status("yok")
    with pytest.raises(KeyError):
        jm.cancel("yok")


# --- Fix wave: yükleme (C1 ad, I7 dtype, Minor 6/12) ---------------------------------------------


def test_load_defaults_to_bf16_and_accepts_name(client, auth, tiny_model_dir):
    body = {"path": str(tiny_model_dir), "name": "qwen2.5:7b"}
    st = wait_done(client, auth, client.post("/api/session/load", json=body, headers=auth).json()["job_id"])
    assert st["result"]["name"] == "qwen2.5:7b"
    assert st["result"]["device"] == "cpu" and st["result"]["device_note"] is None
    import torch

    assert next(client.app.state.session.lm.model.parameters()).dtype == torch.bfloat16


def test_load_rejects_unknown_dtype(client, auth, tiny_model_dir):
    body = {"path": str(tiny_model_dir), "dtype": "int8"}
    assert client.post("/api/session/load", json=body, headers=auth).status_code == 422


def test_cancel_during_load_leaves_session_empty(client, auth, lm, monkeypatch, tiny_model_dir):
    entered, release = threading.Event(), threading.Event()

    def slow_load(path, dtype, name):
        entered.set()
        assert release.wait(10)
        return lm

    monkeypatch.setattr(app_module, "_load", slow_load)
    job = client.post("/api/session/load", json={"path": str(tiny_model_dir)}, headers=auth).json()["job_id"]
    assert entered.wait(10)
    client.post(f"/api/jobs/{job}/cancel", headers=auth)
    release.set()
    wait_done(client, auth, job, expect="cancelled")
    assert client.app.state.session.lm is None


def test_unexpected_load_error_is_wrapped_in_turkish(client, auth, monkeypatch, tiny_model_dir):
    def bad(*a, **kw):
        raise KeyError("model.layers.0.weight")

    monkeypatch.setattr(app_module, "load_model", bad)
    job = client.post("/api/session/load", json={"path": str(tiny_model_dir)}, headers=auth).json()["job_id"]
    st = wait_done(client, auth, job, expect="error")
    assert st["error"].startswith("Model yüklenemedi: KeyError")


# --- Fix wave: I3/I4 (tarif metrikleri, nötr set) ---------------------------------------------------


def _stub_suite(lm, items, interventions, progress=None, cancel=None, max_new_tokens=200):
    from neshter.compare import CategoryResult
    from neshter.metrics import rate_diff

    return [
        CategoryResult("code", 10, 10, 2, rate_diff(10, 10, 2, 10)),
        CategoryResult("general", 10, 9, 9, rate_diff(9, 10, 9, 10)),
    ]


def test_suite_job_returns_recipe_metrics_and_wilson(ready, monkeypatch):
    from neshter.metrics import load_neutral, wilson

    monkeypatch.setattr(app_module, "run_suite", _stub_suite)
    c, a = ready
    job = c.post("/api/suite", json={"recipe_ops": HEAD_OPS, "target": "code"}, headers=a).json()["job_id"]
    res = wait_done(c, a, job, timeout=120)["result"]
    m = res["metrics"]
    assert m["kl"] >= 0 and m["kl_random"] >= 0 and m["kl_random_note"] is None
    assert m["perplexity_orig"] > 1 and m["perplexity_int"] > 1
    assert m["n_prompts"] == len(load_neutral())
    assert res["categories"][0]["orig_ci"] == list(wilson(10, 10)[1:])


def test_effect_defaults_to_reserved_neutral_prompts(ready, monkeypatch):
    from neshter.metrics import load_neutral

    seen = {}

    def fake(lm, heads, prompts, **kw):
        from neshter.metrics import EffectResult

        seen["prompts"] = prompts
        return EffectResult(0.0, 0.0, 0.0, len(prompts))

    monkeypatch.setattr(app_module, "real_effect", fake)
    c, a = ready
    wait_done(c, a, c.post("/api/effect", json={"heads": [[0, 1]]}, headers=a).json()["job_id"])
    assert seen["prompts"] == load_neutral()
