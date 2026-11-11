import base64
import json
import re
import socket
import threading
import tomllib
from pathlib import Path

import pytest
import torch
from conftest import recipe_for

from neshter import cli
from neshter import export as export_module
from neshter.cli import main
from neshter.interventions import HeadAblation
from neshter.interventions import DirectionAblation
from neshter.recipe import SourceRef, load_recipe, save_recipe
from neshter.server.app import create_app

ROOT = Path(__file__).resolve().parent.parent


def _recipe_file(tmp_path, lm, ops, concept="code") -> Path:
    path = tmp_path / "r.nester.json"
    save_recipe(recipe_for(lm, ops, concept=concept), path)
    return path


def _apply(recipe: Path, model: Path, out: Path, *extra: str) -> int:
    return main(["apply", str(recipe), str(model), "--out", str(out), *extra])


# --- apply ------------------------------------------------------------------------------------


def test_apply_roundtrip(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 0
    assert (tmp_path / "o" / "model.safetensors").exists()
    assert (tmp_path / "o" / "recipe.nester.json").exists()
    cfg = json.loads((tmp_path / "o" / "config.json").read_text(encoding="utf-8"))
    assert "abliterated" not in cfg["neshter"]


def test_apply_refusal_concept_sets_metadata(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)], concept="refusal")
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 0
    cfg = json.loads((tmp_path / "o" / "config.json").read_text(encoding="utf-8"))
    assert cfg["neshter"]["abliterated"] == "refusal"
    assert "safety-ablated" in (tmp_path / "o" / "README.md").read_text(encoding="utf-8")


def test_apply_rejects_out_of_range(tmp_path, tiny_model_dir, lm, capsys):
    r = _recipe_file(tmp_path, lm, [HeadAblation(7, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 2
    err = capsys.readouterr().err
    assert "katman 7" in err
    assert not (tmp_path / "o").exists()


def test_apply_rejects_invalid_recipe_file(tmp_path, tiny_model_dir, capsys):
    bad = tmp_path / "bad.nester.json"
    bad.write_text('{"schema_version": 1, "evil": true}', encoding="utf-8")
    assert _apply(bad, tiny_model_dir, tmp_path / "o") == 2
    assert capsys.readouterr().err.strip()
    assert not (tmp_path / "o").exists()


def test_apply_sha_mismatch_needs_force(tmp_path, tiny_model_dir, lm, capsys):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    recipe = load_recipe(r)
    recipe = recipe.model_copy(
        update={"source_model": SourceRef(name=recipe.source_model.name, sha256="0" * 64)}
    )
    save_recipe(recipe, r)

    assert _apply(r, tiny_model_dir, tmp_path / "o") == 3
    assert "SHA-256" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()

    assert _apply(r, tiny_model_dir, tmp_path / "o", "--force") == 0
    assert (tmp_path / "o" / "model.safetensors").exists()


def test_apply_refuses_existing_out_dir(tmp_path, tiny_model_dir, lm, capsys):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    (tmp_path / "o").mkdir()
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 1
    assert "zaten var" in capsys.readouterr().err


def test_apply_gguf_without_llama_cpp_exits_4(tmp_path, tiny_model_dir, lm, monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_llama_cpp", lambda: None)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "gguf_q4_k_m") == 4
    assert "llama.cpp gerekli" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("fmt, quant", [("gguf_q4_k_m", "Q4_K_M"), ("gguf_q8_0", "Q8_0")])
def test_apply_gguf_uses_safetensors_then_converter(
    tmp_path, tiny_model_dir, lm, monkeypatch, fmt, quant
):
    fake_cpp = tmp_path / "llama"
    calls = {}

    def fake_to_gguf(hf_dir, out_file, q, llama_cpp, metadata=None, cancel=None):
        calls.update(hf=Path(hf_dir), out=Path(out_file), quant=q, cpp=Path(llama_cpp), metadata=metadata)
        assert (Path(hf_dir) / "model.safetensors").exists()
        Path(out_file).write_bytes(b"GGUF")
        return Path(out_file)

    monkeypatch.setattr(cli, "find_llama_cpp", lambda: fake_cpp)
    monkeypatch.setattr(export_module, "to_gguf", fake_to_gguf)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)], concept="refusal")
    out = tmp_path / "o"
    assert _apply(r, tiny_model_dir, out, "--format", fmt) == 0
    assert calls["quant"] == quant
    assert calls["cpp"] == fake_cpp
    assert calls["out"].parent == out
    assert calls["out"].exists()
    assert not calls["hf"].exists()  # ara safetensors klasörü silinir
    assert "safety-ablated" in calls["metadata"]["general.tags"]  # tarifin kavramı refusal
    names = {p.name for p in out.iterdir()}
    assert {"o.gguf", "README.md", "recipe.nester.json"} <= names  # kart ve tarif GGUF'un yanında kalır
    assert "safety-ablated" in (out / "README.md").read_text(encoding="utf-8")


def test_apply_gguf_failure_removes_partial_out(tmp_path, tiny_model_dir, lm, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("dönüştürme patladı")

    monkeypatch.setattr(cli, "find_llama_cpp", lambda: tmp_path)
    monkeypatch.setattr(export_module, "to_gguf", boom)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "gguf_q8_0") == 1
    assert "dönüştürme patladı" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_apply_unsupported_format_is_usage_error(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "onnx") == 2


def test_apply_gemma2_direction_recipe_saves(tmp_path, tiny_gemma2_dir, gemma_lm):
    from neshter.loader import load_model

    r_vec = torch.randn(gemma_lm.arch.d_model, generator=torch.Generator().manual_seed(2))
    r = _recipe_file(tmp_path, gemma_lm, [DirectionAblation(r_vec), HeadAblation(0, 1)])
    out = tmp_path / "o"
    assert _apply(r, tiny_gemma2_dir, out) == 0
    assert (out / "model.safetensors").is_file()
    m = load_model(out, device="cpu", dtype=torch.float32)
    assert m.arch.family == "gemma2"
    w = m.model.get_input_embeddings().weight
    proj = (w @ (r_vec / r_vec.norm())).abs()
    assert (proj <= 2e-2 * w.norm(dim=-1)).all()  # bf16 kayıt: gömme r'ye (yuvarlama payıyla) dik


# --- serve ------------------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch):
    """create_app/_run_server/webbrowser.open sahtelenir; gerçek sunucu başlamaz."""
    seen = {"opened": threading.Event(), "urls": []}

    def fake_create_app(token, port, roots=None):
        seen.update(token=token, app_port=port)
        return "APP"

    def fake_run(app, **kw):
        seen.update(app=app, run_kw=kw)
        # Tarayıcı iş parçacığı sunucunun hazır olmasını bekler: portu dinleyen sahte bir sunucu.
        with socket.socket() as s:
            s.bind((kw["host"], kw["port"]))
            s.listen()
            if not seen.get("no_browser"):
                assert seen["opened"].wait(10)

    def fake_open(url, *a, **k):
        seen["urls"].append(url)
        seen["opened"].set()
        return True

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli, "_run_server", fake_run)
    monkeypatch.setattr(cli.webbrowser, "open", fake_open)
    return seen


def test_serve_binds_loopback_same_port_and_opens_browser(served, capsys):
    assert main([]) == 0
    kw = served["run_kw"]
    assert served["app"] == "APP"
    assert kw["host"] == "127.0.0.1"
    assert kw["port"] == served["app_port"] > 0
    token = served["token"]
    assert len(token) >= 32
    url = f"http://127.0.0.1:{kw['port']}/#token={token}"
    assert served["urls"] == [url]
    assert url in capsys.readouterr().out


def test_run_server_wires_request_exit(monkeypatch):
    class FakeServer:
        def __init__(self, config):
            self.should_exit = False
            FakeServer.inst = self

        def run(self):
            app.state.request_exit()

    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)
    app = create_app("t", 47997, roots={})
    cli._run_server(app, "127.0.0.1", 47997)
    assert FakeServer.inst.should_exit is True


def _hold_sse_then_shutdown(monkeypatch, tmp_path, hook):
    """Gerçek uvicorn.Server, çalışan iptal edilebilir bir işin açık SSE akışı; kapat isteği sonrası geçen süre."""
    import http.client
    import time

    monkeypatch.setenv("NESHTER_HOME", str(tmp_path))
    port = cli._free_port()
    token = "t" * 30
    app = create_app(token, port, roots={})
    started = threading.Event()

    def job(report, cancel):
        started.set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not cancel.is_set():
            report(0.1, "x")
            time.sleep(0.05)
        return {}

    job_id = app.state.jobs.submit(job, kind="x")
    result = {}

    def client():
        try:
            headers = {"X-Neshter-Token": token, "Host": f"127.0.0.1:{port}"}
            for _ in range(100):  # sunucu dinlemeye başlayana dek
                try:
                    sse = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    sse.request("GET", f"/api/jobs/{job_id}/events", headers=headers)
                    break
                except OSError:
                    time.sleep(0.1)
            result["sse"] = sse  # bağlantı sunucu kapanana dek açık kalmalı (çöp toplayıcı kapatmasın)
            resp = sse.getresponse()
            resp.fp.readline()  # akış gerçekten açık
            assert started.wait(10)
            result["t0"] = time.monotonic()
            result["status"] = hook(port, headers)
        except BaseException as e:  # istemci düşerse sunucu kapat isteği almaz ve test sonsuza dek takılırdı
            result["error"] = e
        finally:
            for _ in range(100):  # request_exit, _run_server içinde bağlanır
                if app.state.request_exit is not None:
                    app.state.request_exit()  # idempotent
                    break
                time.sleep(0.05)

    t = threading.Thread(target=client, daemon=True)
    t.start()
    cli._run_server(app, "127.0.0.1", port)
    t1 = time.monotonic()
    t.join(5)
    return result, t1


def test_shutdown_returns_promptly_with_open_sse_stream(monkeypatch, tmp_path):
    import http.client

    def post_shutdown(port, headers):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/api/shutdown", headers=headers)
        return c.getresponse().status

    result, t1 = _hold_sse_then_shutdown(monkeypatch, tmp_path, post_shutdown)
    assert "error" not in result, result.get("error")
    assert result["status"] == 202
    assert t1 - result["t0"] < 3  # iş 30 sn sürerdi; kapat isteği işi iptal edip akışı bitirir


def test_sse_shutdown_helper_does_not_hang_when_the_client_thread_fails(monkeypatch, tmp_path):
    def boom(port, headers):
        raise RuntimeError("istemci düştü")

    result, _ = _hold_sse_then_shutdown(monkeypatch, tmp_path, boom)  # eski hali burada sonsuza dek beklerdi
    assert isinstance(result["error"], RuntimeError)


def test_graceful_shutdown_has_an_upper_bound(monkeypatch):
    seen = {}

    class FakeServer:
        def __init__(self, config):
            seen["timeout"] = config.timeout_graceful_shutdown

        def run(self):
            pass

    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)
    cli._run_server(create_app("t", 47996, roots={}), "127.0.0.1", 47996)
    assert seen["timeout"] == cli.GRACEFUL_SHUTDOWN_SECONDS


def test_serve_no_browser_does_not_open(served, capsys):
    served["no_browser"] = True
    assert main(["serve", "--no-browser"]) == 0
    assert served["urls"] == []
    assert f"#token={served['token']}" in capsys.readouterr().out


def test_serve_default_command_accepts_flags(served):
    served["no_browser"] = True
    assert main(["--no-browser"]) == 0
    assert served["run_kw"]["host"] == "127.0.0.1"
    assert served["urls"] == []


def test_serve_explicit_port(served):
    served["no_browser"] = True
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert main(["serve", "--no-browser", "--port", str(port)]) == 0
    assert served["app_port"] == port == served["run_kw"]["port"]


@pytest.mark.parametrize("bad", ["0", "65536", "-1", "abc"])
def test_serve_rejects_invalid_port(served, bad):
    assert main(["serve", "--no-browser", "--port", bad]) == 2
    assert "run_kw" not in served


def test_serve_has_no_host_flag(served):
    assert main(["serve", "--no-browser", "--host", "0.0.0.0"]) == 2
    assert "run_kw" not in served


def test_serve_tokens_are_fresh(served):
    served["no_browser"] = True
    main(["--no-browser"])
    first = served["token"]
    main(["--no-browser"])
    assert served["token"] != first


# --- Fix wave: I9 (tedarik zinciri) ---------------------------------------------------------------


def test_launcher_uses_lockfile_and_asks_before_installing_uv():
    raw = (ROOT / "neshter.bat").read_bytes()
    bat = raw.decode("ascii")  # yalnızca ASCII (cmd kod sayfasından bağımsız)
    assert raw.count(b"\r\n") == raw.count(b"\n") > 0  # CRLF
    assert not [b for b in raw if b < 32 and b not in (10, 13)]  # kaçış dizisi yüzünden sızan denetim baytı yok
    lines = bat.split("\r\n")
    assert r'set "PATH=%USERPROFILE%\.local\bin;%PATH%"' in lines
    assert 'uv run --project "%~dp0." --frozen neshter %*' in bat and "uvx" not in bat
    # choice: beklenmeyen hata kodu (0, Ctrl+C) asla kuruma gitmez; yalnızca 1 (Y) kurar
    i = lines.index('choice /M "uv simdi kurulsun mu"')
    assert lines[i + 1 : i + 4] == ["if errorlevel 2 goto declined", "if errorlevel 1 goto install", "goto declined"]
    assert lines.index(":install") > i and "irm https://astral.sh/uv/" in lines[lines.index(":install") + 1]
    # uv run başarısızsa çift tıklayan hatayı görsün: pause, sonra çıkış
    j = lines.index('uv run --project "%~dp0." --frozen neshter %*')
    assert lines[j + 1 : j + 4] == ['set "rc=%errorlevel%"', 'if not "%rc%"=="0" pause', "exit /b %rc%"]
    assert "choice /M" in bat and re.search(r"https://astral\.sh/uv/\d+\.\d+\.\d+/install\.ps1", bat)
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert all("<" in d for d in deps) and "jinja2>=3.1.6,<4" in deps
    assert "transformers>=5.22,<6" in deps and "torch>=2.15,<3" in deps


def test_lockfile_matches_pyproject():
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv PATH'te yok")
    r = subprocess.run([uv, "lock", "--check", "--offline"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


# --- Güvenlik etiketi (Task 11) ---------------------------------------------------------------


def test_apply_prints_intervention_summary(tmp_path, tiny_model_dir, lm, capsys):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 0
    assert "Tarif: 1 head, 0 yön, kavram=code" in capsys.readouterr().out


def _dir_recipe(tmp_path, lm, concept):
    v = torch.randn(lm.arch.d_model, generator=torch.Generator().manual_seed(0))
    return _recipe_file(tmp_path, lm, [DirectionAblation(v)], concept=concept)


def test_apply_warns_on_unlabelled_direction_recipe(tmp_path, tiny_model_dir, lm, capsys):
    r = _dir_recipe(tmp_path, lm, "code")
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 0
    cap = capsys.readouterr()
    assert "Tarif: 0 head, 1 yön, kavram=code" in cap.out
    assert "yön silme" in cap.err and "--label-safety" in cap.err
    cfg = json.loads((tmp_path / "o" / "config.json").read_text(encoding="utf-8"))
    assert "abliterated" not in cfg["neshter"]  # uyarı etiketlemez


def test_apply_no_warning_for_refusal_or_head_only_or_flag(tmp_path, tiny_model_dir, lm, capsys):
    assert _apply(_dir_recipe(tmp_path, lm, "refusal"), tiny_model_dir, tmp_path / "o1") == 0
    assert "yön silme" not in capsys.readouterr().err
    assert _apply(_recipe_file(tmp_path, lm, [HeadAblation(1, 0)]), tiny_model_dir, tmp_path / "o2") == 0
    assert "yön silme" not in capsys.readouterr().err
    assert _apply(_dir_recipe(tmp_path, lm, "code"), tiny_model_dir, tmp_path / "o3", "--label-safety") == 0
    assert "yön silme" not in capsys.readouterr().err


def test_apply_label_safety_sets_metadata(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--label-safety") == 0
    cfg = json.loads((tmp_path / "o" / "config.json").read_text(encoding="utf-8"))
    assert cfg["neshter"]["abliterated"] == "refusal"
    assert "safety-ablated" in (tmp_path / "o" / "README.md").read_text(encoding="utf-8")


def test_apply_label_safety_reaches_gguf_metadata(tmp_path, tiny_model_dir, lm, monkeypatch):
    calls = {}

    def fake_to_gguf(hf_dir, out_file, q, llama_cpp, metadata=None, cancel=None):
        calls["metadata"] = metadata
        Path(out_file).write_bytes(b"GGUF")
        return Path(out_file)

    monkeypatch.setattr(cli, "find_llama_cpp", lambda: tmp_path / "llama")
    monkeypatch.setattr(export_module, "to_gguf", fake_to_gguf)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "gguf_q8_0", "--label-safety") == 0
    assert "safety-ablated" in calls["metadata"]["general.tags"]


def test_label_safety_recipe_labels_itself(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)], concept="code")
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--label-safety") == 0
    assert load_recipe(tmp_path / "o" / "recipe.nester.json").concept == "refusal"
    assert load_recipe(r).concept == "code"  # girdi tarifi değişmez


def test_label_safety_recipe_labels_itself_in_gguf_folder(tmp_path, tiny_model_dir, lm, monkeypatch):
    def fake_to_gguf(hf_dir, out_file, q, llama_cpp, metadata=None, cancel=None):
        Path(out_file).write_bytes(b"GGUF")
        return Path(out_file)

    monkeypatch.setattr(cli, "find_llama_cpp", lambda: tmp_path / "llama")
    monkeypatch.setattr(export_module, "to_gguf", fake_to_gguf)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)], concept="code")
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "gguf_q8_0", "--label-safety") == 0
    assert load_recipe(tmp_path / "o" / "recipe.nester.json").concept == "refusal"


def test_unlabelled_recipe_keeps_its_concept(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)], concept="code")
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 0
    assert load_recipe(tmp_path / "o" / "recipe.nester.json").concept == "code"


def test_wheel_build_contains_web_dist(tmp_path):
    """Gerçek wheel: `dist/` .gitignore'da olsa bile (hatch süzgeci) `artifacts` onu wheel'e sokar."""
    import shutil
    import subprocess
    import zipfile

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv yok: yalnızca pyproject denetimi çalışır")
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(ROOT / "pyproject.toml", proj)
    shutil.copytree(ROOT / "neshter", proj / "neshter", ignore=shutil.ignore_patterns("__pycache__", "dist"))
    (proj / ".gitignore").write_text("dist/\n", encoding="utf-8")  # hatch .gitignore'u uygular
    (proj / "neshter" / "web" / "dist").mkdir(parents=True)
    (proj / "neshter" / "web" / "dist" / "index.html").write_text("<!doctype html>", encoding="utf-8")
    out = tmp_path / "out"
    r = subprocess.run([uv, "build", "--wheel", "--out-dir", str(out), str(proj)], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr
    (wheel,) = out.glob("*.whl")
    assert "neshter/web/dist/index.html" in zipfile.ZipFile(wheel).namelist()


def test_wheel_ships_web_dist_artifacts():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "neshter/web/dist/**" in cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]


# --- Final fix wave: M1 (kaynak değişti: çıktı silinir), M8 (Refusal büyük harf) ----------------------------


def test_apply_source_modified_removes_output(tmp_path, tiny_model_dir, lm, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "verify_source_untouched", lambda lm_, progress=None, cancel=None: False)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o") != 0
    assert not (tmp_path / "o").exists()
    assert "SHA-256" in capsys.readouterr().err


def test_apply_capitalised_refusal_concept_is_labelled_without_warning(tmp_path, tiny_model_dir, lm, capsys):
    r = _dir_recipe(tmp_path, lm, "Refusal")
    assert _apply(r, tiny_model_dir, tmp_path / "o") == 0
    assert "yön silme" not in capsys.readouterr().err
    cfg = json.loads((tmp_path / "o" / "config.json").read_text(encoding="utf-8"))
    assert cfg["neshter"]["abliterated"] == "refusal"
