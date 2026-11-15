"""Derlenmiş arayüzün sunumu: CSP, dist yolu ve wheel içeriği."""

from __future__ import annotations

import hashlib
import json
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


def test_index_revalidates_and_hashed_assets_cache(tmp_path, monkeypatch):
    """Yeni derleme bir sonraki açılışta görünsün: index.html her seferinde sorulur, içerik özetli /assets/ uzun süre saklanır."""
    (tmp_path / "index.html").write_text("<!doctype html><title>Neşter</title>", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-Ab12Cd34.js").write_text("export {};", encoding="utf-8")
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path)
    client = TestClient(create_app("t", 47998, roots={}), base_url="http://127.0.0.1:47998")
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/index.html").headers["cache-control"] == "no-cache"
    assert client.get("/assets/index-Ab12Cd34.js").headers["cache-control"] == "public, max-age=31536000, immutable"


@pytest.mark.skipif(shutil.which("git") is None, reason="git yok")
def test_built_ui_path_not_gitignored():
    assert subprocess.run(["git", "check-ignore", "-q", "neshter/web/dist/index.html"], cwd=ROOT).returncode == 1


def test_wheel_artifacts_include_dist():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "neshter/web/dist/**" in cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]


WEB = ROOT / "web"
_HASH_ROOT_FILES = ("index.html", "vite.config.ts", "tsconfig.json", "package-lock.json")
_HASH_ROOT_DIRS = ("src", "public")


def source_hash(web_dir: Path) -> str:
    """`web/scripts/source-hash.mjs` in Python: same file set, same EOL normalisation, same digest layout."""
    files: list[str] = []
    for name in _HASH_ROOT_DIRS:
        base = web_dir / name
        if base.is_dir():
            files.extend(p.relative_to(web_dir).as_posix() for p in base.rglob("*") if p.is_file())
    files.extend(name for name in _HASH_ROOT_FILES if (web_dir / name).is_file())
    files.sort(key=lambda rel: rel.encode("utf-16-be"))  # JS compares UTF-16 code units
    digest = hashlib.sha256()
    for rel in files:
        if "/__tests__/" in f"/{rel}" or "/__fixtures__/" in rel or ".test." in rel:
            continue
        content = (web_dir / rel).read_bytes().replace(b"\r\n", b"\n")
        digest.update(f"{rel}\0{hashlib.sha256(content).hexdigest()}\n".encode())
    return digest.hexdigest()


def _dist_tracked() -> bool:
    if shutil.which("git") is None:
        return False
    out = subprocess.run(
        ["git", "ls-files", "neshter/web/dist/index.html"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def test_python_source_hash_matches_golden_vector():
    fixture = WEB / "scripts" / "__tests__" / "__fixtures__" / "hash"
    expected = (fixture / "expected.txt").read_text(encoding="utf-8").strip()
    assert source_hash(fixture) == expected  # the Vitest suite asserts the same vector


@pytest.mark.skipif(shutil.which("git") is None, reason="git yok")
@pytest.mark.skipif(not (WEB / "package.json").exists(), reason="web/ kaynakları yok (paketlenmiş kurulum)")
def test_built_ui_is_committed():
    assert _dist_tracked(), "neshter/web/dist/index.html git'te değil: `npm --prefix web run build` sonrası dist'i ekleyin"


@pytest.mark.skipif(not (WEB / "package.json").exists(), reason="web/ kaynakları yok (paketlenmiş kurulum)")
def test_committed_dist_matches_web_sources():
    if not _dist_tracked():
        pytest.skip("dist git'te değil")
    info = json.loads((ROOT / "neshter" / "web" / "dist" / "build-info.json").read_text(encoding="utf-8"))
    assert info["source_hash"] == source_hash(WEB), "dist bayat: web/ kaynakları değişti, `npm --prefix web run build` çalıştırın"
