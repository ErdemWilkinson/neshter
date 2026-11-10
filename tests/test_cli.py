import json
import socket
import threading
from pathlib import Path

import pytest
from conftest import recipe_for

from neshter import cli
from neshter.cli import main
from neshter.interventions import HeadAblation
from neshter.recipe import SourceRef, load_recipe, save_recipe


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

    def fake_to_gguf(hf_dir, out_file, q, llama_cpp):
        calls.update(hf=Path(hf_dir), out=Path(out_file), quant=q, cpp=Path(llama_cpp))
        assert (Path(hf_dir) / "model.safetensors").exists()
        Path(out_file).write_bytes(b"GGUF")
        return Path(out_file)

    monkeypatch.setattr(cli, "find_llama_cpp", lambda: fake_cpp)
    monkeypatch.setattr(cli, "to_gguf", fake_to_gguf)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    out = tmp_path / "o"
    assert _apply(r, tiny_model_dir, out, "--format", fmt) == 0
    assert calls["quant"] == quant
    assert calls["cpp"] == fake_cpp
    assert calls["out"].parent == out
    assert calls["out"].exists()
    assert not calls["hf"].exists()  # ara safetensors klasörü silinir


def test_apply_gguf_failure_removes_partial_out(tmp_path, tiny_model_dir, lm, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("dönüştürme patladı")

    monkeypatch.setattr(cli, "find_llama_cpp", lambda: tmp_path)
    monkeypatch.setattr(cli, "to_gguf", boom)
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "gguf_q8_0") == 1
    assert "dönüştürme patladı" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_apply_unsupported_format_is_usage_error(tmp_path, tiny_model_dir, lm):
    r = _recipe_file(tmp_path, lm, [HeadAblation(1, 0)])
    assert _apply(r, tiny_model_dir, tmp_path / "o", "--format", "onnx") == 2


# --- serve ------------------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch):
    """create_app/uvicorn.run/webbrowser.open sahtelenir; gerçek sunucu başlamaz."""
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
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
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


def test_serve_has_no_host_flag(served):
    assert main(["serve", "--no-browser", "--host", "0.0.0.0"]) == 2
    assert "run_kw" not in served


def test_serve_tokens_are_fresh(served):
    served["no_browser"] = True
    main(["--no-browser"])
    first = served["token"]
    main(["--no-browser"])
    assert served["token"] != first
