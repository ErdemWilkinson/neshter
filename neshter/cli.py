"""Komut satırı: `neshter` (yerel sunucu + tarayıcı) ve `neshter apply` (tarifi başsız uygula).

Güvenlik: sunucu yalnızca 127.0.0.1'e bağlanır (host bayrağı yoktur); her oturum rastgele bir
belirteç alır ve belirteç URL'nin `#` parçasında taşınır (sunucuya hiç gönderilmez).
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from neshter.export import find_llama_cpp, save_safetensors, to_gguf, verify_source_untouched
from neshter.loader import load_model
from neshter.recipe import check_against, load_recipe, to_interventions
from neshter.server.app import create_app

HOST = "127.0.0.1"
_QUANT = {"gguf_q4_k_m": "Q4_K_M", "gguf_q8_0": "Q8_0"}
_FORMATS = ("safetensors", *_QUANT)

EXIT_OK, EXIT_ERROR, EXIT_RECIPE, EXIT_SHA, EXIT_NO_LLAMA_CPP = 0, 1, 2, 3, 4


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _open_when_ready(url: str, port: int, timeout: float = 30.0) -> None:
    """Sunucu portu dinlemeye başlayınca tarayıcıyı açar (uvicorn.run bloklar)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    webbrowser.open(url)


def _serve(args: argparse.Namespace) -> int:
    port = args.port if args.port else _free_port()
    token = secrets.token_urlsafe(24)
    app = create_app(token, port)
    url = f"http://{HOST}:{port}/#token={token}"
    print(f"Neşter çalışıyor: {url}")
    print("Durdurmak için Ctrl+C. Adres yalnızca bu bilgisayardan erişilebilir.")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url, port), daemon=True).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
    return EXIT_OK


def _apply(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
    except Exception as e:  # geçersiz şema, büyük dosya, okunamayan dosya
        _err(f"Tarif okunamadı: {e}")
        return EXIT_RECIPE

    quant = _QUANT.get(args.format)
    llama_cpp = None
    if quant:
        llama_cpp = find_llama_cpp()
        if llama_cpp is None:
            _err(
                "llama.cpp gerekli: GGUF çıktısı için NESHTER_LLAMA_CPP ortam değişkenini "
                "llama.cpp klasörüne ayarlayın ya da llama-quantize'ı PATH'e ekleyin."
            )
            return EXIT_NO_LLAMA_CPP

    out = Path(args.out)
    if out.exists():
        _err(f"'{out}' zaten var; üzerine yazılmaz. Başka bir klasör adı seçin.")
        return EXIT_ERROR

    try:
        lm = load_model(Path(args.model), device="cpu")
    except Exception as e:
        _err(f"Model yüklenemedi: {e}")
        return EXIT_ERROR

    check = check_against(recipe, lm.arch, lm.source_sha256)
    if not check.ok:
        _err("Tarif bu modele uymuyor:")
        for e in check.errors:
            _err(f"  - {e}")
        return EXIT_RECIPE
    if check.sha_mismatch:
        _err(
            "Uyarı: tarifin yazıldığı modelin SHA-256 değeri bu modelinkiyle uyuşmuyor; "
            "tarif başka bir model sürümü için hazırlanmış olabilir."
        )
        if not args.force:
            _err("Yine de uygulamak için --force kullanın.")
            return EXIT_SHA

    try:
        count = len(to_interventions(recipe))
        # Kayıt tarifin kendisinden yapılır: recipe.concept == "refusal" ise metadata ve model kartı işlenir.
        if quant is None:
            result = save_safetensors(lm, recipe, out)
        else:
            result = _save_gguf(lm, recipe, out, quant, llama_cpp)
        if not verify_source_untouched(lm):
            _err("Hata: kaynak model dosyaları işlem sırasında değişmiş görünüyor (SHA-256 uyuşmuyor).")
            return EXIT_ERROR
    except Exception as e:
        _err(f"Kayıt başarısız: {e}")
        return EXIT_ERROR

    print(f"{count} müdahale uygulandı, kaydedildi: {result}")
    return EXIT_OK


def _save_gguf(lm, recipe, out: Path, quant: str, llama_cpp: Path) -> Path:
    """safetensors ara klasörü → GGUF; out bu işte oluşur, hata olursa silinir."""
    try:
        hf_dir = out / "_hf"
        save_safetensors(lm, recipe, hf_dir)
        gguf = to_gguf(hf_dir, out / f"{out.name}.gguf", quant, llama_cpp)
        shutil.rmtree(hf_dir)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return gguf


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neshter",
        description="Neşter: yerel dil modellerinin dikkat head'lerini inceler ve ablasyon yapar.",
    )
    sub = p.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="yerel sunucuyu başlat ve tarayıcıyı aç (varsayılan)")
    serve.add_argument("--no-browser", action="store_true", help="tarayıcıyı açma")
    serve.add_argument("--port", type=int, default=0, help="sabit port (varsayılan: boş bir port)")
    serve.set_defaults(func=_serve)

    apply = sub.add_parser("apply", help="bir .nester.json tarifini modele uygula ve kaydet")
    apply.add_argument("recipe", type=Path, help="tarif dosyası (.nester.json)")
    apply.add_argument("model", type=Path, help="model klasörü, .gguf veya .pt dosyası")
    apply.add_argument("--out", type=Path, required=True, help="çıktı klasörü (var olmamalı)")
    apply.add_argument("--format", choices=_FORMATS, default="safetensors")
    apply.add_argument("--force", action="store_true", help="SHA-256 uyuşmazlığına rağmen uygula")
    apply.set_defaults(func=_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if not args_in or (args_in[0].startswith("-") and args_in[0] not in ("-h", "--help")):
        args_in.insert(0, "serve")  # `neshter` ve `neshter --no-browser` = serve
    parser = _parser()
    try:
        args = parser.parse_args(args_in)
    except SystemExit as e:  # argparse hata/--help: kodu döndür
        return e.code if isinstance(e.code, int) else EXIT_RECIPE
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
