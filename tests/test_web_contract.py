"""Arayüz sözleşmesi: `web/src/api/__fixtures__/*.json`, canlı sunucu yanıtlarının anahtar şekliyle aynı kalmalı.

Karşılaştırma yalnızca anahtar kümeleridir (iç içe sözlükler özyinelemeli, listeler ilk öğeleriyle); değerler
UI testlerinde elle yazılır ve burada denetlenmez. `NESHTER_UPDATE_FIXTURES=1` dosyaları yeniden yazar
(listeler 2 öğeye, metinler 200 karaktere kısaltılır, mutlak yollar `C:\\Users\\x\\model` olur).
"""

import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import recipe_for
from neshter.interventions import HeadAblation
from neshter.server.app import create_app

PORT = 47998
TOKEN = "contract-token"
AUTH = {"X-Neshter-Token": TOKEN}
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "web" / "src" / "api" / "__fixtures__"
UPDATE = os.environ.get("NESHTER_UPDATE_FIXTURES") == "1"
HEAD_OPS = [{"type": "head", "layer": 0, "head": 1}]
PLACEHOLDER_PATH = "C:\\Users\\x\\model"
_ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/)")

NAMES = [
    "system", "session_empty", "session_loaded", "models", "concepts", "activity", "mlp", "compare",
    "recipe_check", "job_effect", "job_hunt", "job_suite", "estimate", "job_export", "jobs", "drafts",
    "draft", "prefs", "error_busy",
]


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
        if st["state"] in ("done", "error", "cancelled"):
            assert st["state"] == "done", st
            return st
        time.sleep(0.05)
    raise AssertionError(f"iş zaman aşımına uğradı: {job_id}")


def _ok(response):
    assert response.status_code in (200, 202), response.text
    return response.json()


def _job(client: TestClient, route: str, body: dict) -> dict:
    return _wait(client, _ok(client.post(route, json=body, headers=AUTH))["job_id"])


@pytest.fixture(scope="module")
def live(tiny_model_dir, tmp_path_factory) -> dict:
    """Her fixture adı için canlı yanıt (tiny model yüklü; modül boyunca bir kez üretilir)."""
    work = tmp_path_factory.mktemp("contract")
    snap = work / "hub" / "models--tiny--qwen2" / "snapshots" / "abc123"
    shutil.copytree(tiny_model_dir, snap)
    app = create_app(TOKEN, PORT, roots={"huggingface": [work / "hub"]})
    c = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")

    def get(route):
        return _ok(c.get(route, headers=AUTH))

    def post(route, body):
        return _ok(c.post(route, json=body, headers=AUTH))

    out: dict = {"system": get("/api/system"), "session_empty": get("/api/session")}
    models = get("/api/models")
    assert models and models[0]["arch"] is not None, models
    out["models"] = models
    out["concepts"] = get("/api/concepts")
    _job(c, "/api/session/load", {"path": str(snap), "dtype": "float32"})
    out["session_loaded"] = session = get("/api/session")

    out["activity"] = post("/api/activity", {"prompt": "merhaba dünya", "token_index": 0})
    out["mlp"] = post("/api/mlp", {"prompt": "merhaba dünya", "token_index": 0})
    out["compare"] = post(
        "/api/compare",
        {"prompt": "merhaba", "recipe_ops": HEAD_OPS, "max_new_tokens": 4, "concept_id": "code"},
    )
    recipe = recipe_for(app.state.session.lm, [HeadAblation(0, 1)])
    out["recipe_check"] = post("/api/recipe/check", {"recipe": json.loads(recipe.model_dump_json())})

    out["job_effect"] = _job(c, "/api/effect", {"heads": [[0, 1]], "prompts": ["merhaba dünya", "hello world"]})
    out["job_hunt"] = _job(c, "/api/hunt", {"concept_id": "code", "method": "direction"})
    out["job_suite"] = _job(c, "/api/suite", {"recipe_ops": HEAD_OPS, "target": "code", "quick": True})
    out["estimate"] = post("/api/export/estimate", {"format": "safetensors", "recipe_ops": HEAD_OPS})
    export_body = {
        "recipe_ops": HEAD_OPS, "name": "kesik", "format": "safetensors", "ollama": False,
        "out_dir": str(work / "cikti"),
    }
    out["job_export"] = _job(c, "/api/export", export_body)
    out["jobs"] = get("/api/jobs")

    sha = session["source_sha256"]
    draft = {
        "schema_version": 1, "sha256": sha, "model_name": session["name"], "unsaved_count": 1,
        "basket": HEAD_OPS, "shelf": [], "saved_ops": [], "prompts": ["merhaba"],
    }
    assert c.put(f"/api/draft/{sha}", json=draft, headers=AUTH).status_code == 200
    out["draft"] = get(f"/api/draft/{sha}")
    out["drafts"] = get("/api/drafts")
    out["prefs"] = get("/api/prefs")

    assert app.state.session.lock.acquire(blocking=False)
    try:
        busy = c.post("/api/mlp", json={"prompt": "x", "token_index": 0}, headers=AUTH)
    finally:
        app.state.session.lock.release()
    assert busy.status_code == 409 and busy.json()["error_code"] == "busy"
    out["error_busy"] = busy.json()
    assert set(out) == set(NAMES)
    return out


def shape(value):
    """Anahtar şekli: sözlük -> {anahtar: şekil}; liste -> [ilk öğenin şekli] (boşsa []); diğerleri yaprak (None)."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return None


def _kind(shape_value) -> str:
    return "object" if isinstance(shape_value, dict) else "array" if isinstance(shape_value, list) else "value"


def diff_paths(expected, actual, path: str = "$") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = [f"{path}.{k}: fixture'da var, canlıda yok" for k in sorted(expected.keys() - actual.keys())]
        out += [f"{path}.{k}: canlıda var, fixture'da yok" for k in sorted(actual.keys() - expected.keys())]
        for k in sorted(expected.keys() & actual.keys()):
            out += diff_paths(expected[k], actual[k], f"{path}.{k}")
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if expected and actual:  # biri boşsa öğe şekli karşılaştırılamaz
            return diff_paths(expected[0], actual[0], f"{path}[0]")
        return []
    if _kind(expected) != _kind(actual):
        return [f"{path}: tür değişti ({_kind(expected)} -> {_kind(actual)})"]
    return []


def _fixture_form(value):
    """Yazılacak sözleşme verisi: listeler 2 öğe, metinler 200 karakter, mutlak yollar yer tutucu."""
    if isinstance(value, dict):
        return {k: _fixture_form(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fixture_form(v) for v in value[:2]]
    if isinstance(value, str):
        return PLACEHOLDER_PATH if _ABS_PATH.match(value) else value[:200]
    return value


def test_shape_helpers():
    assert shape({"a": [{"b": 1}, {"c": 2}], "d": None, "e": []}) == {"a": [{"b": None}], "d": None, "e": []}
    new_key = diff_paths(shape({"a": [{"b": 1}]}), shape({"a": [{"b": 1, "c": 2}]}))
    assert new_key == ["$.a[0].c: canlıda var, fixture'da yok"]
    assert diff_paths(shape({"a": {"b": 1}}), shape({"a": None})) == ["$.a: tür değişti (object -> value)"]
    assert diff_paths(shape({"a": []}), shape({"a": [1]})) == []


def test_fixture_form_truncates_and_scrubs():
    form = _fixture_form({"p": "C:\\Users\\erdem\\x", "q": "/tmp/x", "s": "a" * 300, "l": [1, 2, 3]})
    assert form == {"p": PLACEHOLDER_PATH, "q": PLACEHOLDER_PATH, "s": "a" * 200, "l": [1, 2]}


@pytest.mark.parametrize("name", NAMES)
def test_web_fixtures_match_live_shapes(name, live):
    path = FIXTURE_DIR / f"{name}.json"
    if UPDATE:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_fixture_form(live[name]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    assert path.is_file(), f"{path.name} yok: NESHTER_UPDATE_FIXTURES=1 ile üretin"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    problems = diff_paths(shape(fixture), shape(live[name]))
    assert not problems, f"{path.name} canlı şekille uyuşmuyor:\n" + "\n".join(problems)
