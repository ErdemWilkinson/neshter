"""Derlenmiş arayüzün sunumu: CSP, dist yolu ve wheel içeriği."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neshter.server import app as app_module
from neshter.server.app import create_app
from neshter.server.errors import SECURITY_HEADERS

ROOT = Path(__file__).resolve().parent.parent


def test_csp_states_script_worker_font_self():
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    for part in ("script-src 'self'", "worker-src 'self'", "font-src 'self'", "img-src 'self' data:", "connect-src 'self'"):
        assert part in csp
    assert "unsafe-eval" not in csp and "wasm-unsafe-eval" not in csp


def test_serves_built_index_with_csp(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<!doctype html><title>Neşter</title>", encoding="utf-8")
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path)
    r = TestClient(create_app("t", 47998, roots={}), base_url="http://127.0.0.1:47998").get("/")
    assert r.status_code == 200 and "Neşter" in r.text
    assert r.headers["content-security-policy"] == SECURITY_HEADERS["Content-Security-Policy"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git yok")
def test_built_ui_path_not_gitignored():
    assert subprocess.run(["git", "check-ignore", "-q", "neshter/web/dist/index.html"], cwd=ROOT).returncode == 1


def test_wheel_artifacts_include_dist():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "neshter/web/dist/**" in cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]
