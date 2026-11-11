"""Komut satırı: `neshter` (yerel sunucu + tarayıcı) ve `neshter apply` (tarifi başsız uygula).

Güvenlik: sunucu yalnızca 127.0.0.1'e bağlanır (host bayrağı yoktur); her oturum rastgele bir
belirteç alır ve belirteç URL'nin `#` parçasında taşınır (sunucuya hiç gönderilmez).
"""

from __future__ import annotations

import argparse
import secrets
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from neshter.export import (
    SourceModified,
    ensure_source_untouched,
    export_gguf,
    find_llama_cpp,
    is_refusal,
    save_safetensors,
)
from neshter.loader import load_model
from neshter.recipe import HeadOp, check_against, load_recipe, to_interventions
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


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"geçersiz port: {value!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port 1 ile 65535 arasında olmalı: {port}")
    return port


GRACEFUL_SHUTDOWN_SECONDS = 3  # açık SSE akışı (Ctrl+C) kapanışı sonsuza dek bekletmesin


def _run_server(app, host: str, port: int) -> None:
    """uvicorn.Server ile çalıştırır; `/api/shutdown` sunucuyu `should_exit` ile zarifçe durdurur.

    uvicorn, lifespan kapanışından (işleri iptal eden `jobs.shutdown()`) ÖNCE açık bağlantıların bitmesini
    bekler; çalışan işin açık SSE akışı varsa iş bitene dek kapanmazdı. Bu yüzden kapat isteği önce işleri
    iptal eder (akış `cancelled` ile biter), Ctrl+C için de `timeout_graceful_shutdown` üst sınırı vardır."""
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS
    )
    server = uvicorn.Server(config)
    if hasattr(app, "state"):  # testlerde sahte uygulama (str) olabilir

        def request_exit() -> None:
            app.state.jobs.shutdown()  # idempotent
            server.should_exit = True

        app.state.request_exit = request_exit
    server.run()


def _serve(args: argparse.Namespace) -> int:
    port = args.port if args.port else _free_port()
    token = secrets.token_urlsafe(24)
    app = create_app(token, port)
    url = f"http://{HOST}:{port}/#token={token}"
    print(f"Neşter çalışıyor: {url}")
    print("Durdurmak için Ctrl+C. Adres yalnızca bu bilgisayardan erişilebilir.")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url, port), daemon=True).start()
    _run_server(app, host=HOST, port=port)
    return EXIT_OK


def _apply(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
    except Exception as e:  # geçersiz şema, büyük dosya, okunamayan dosya
        _err(f"Tarif okunamadı: {e}")
        return EXIT_RECIPE

    heads = sum(isinstance(op, HeadOp) for op in recipe.interventions)
    dirs = len(recipe.interventions) - heads
    print(f"Tarif: {heads} head, {dirs} yön, kavram={recipe.concept or '-'}")
    if dirs and not is_refusal(recipe.concept) and not args.label_safety:
        _err(
            "Uyarı: bu tarif yön silme içeriyor; Neşter tarifin iddia ettiği kavramı doğrulayamaz. "
            "Bir reddetme yönüyse --label-safety ile etiketleyin."
        )

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
        lm = load_model(Path(args.model), device="auto")  # sunucuyla aynı politika: sığarsa GPU, dtype pick_compute_dtype
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
        # Etiket: tarifin kavramı "refusal" ya da --label-safety (kayıt katmanı ikisini de zorlar).
        safety = args.label_safety or is_refusal(recipe.concept)
        if safety:  # kaydedilen recipe.nester.json kendi kendini etiketlesin
            recipe = recipe.model_copy(update={"concept": "refusal"})
        if quant is None:
            result = save_safetensors(lm, recipe, out, safety_ablated=safety)
        else:
            result = export_gguf(lm, recipe, out, out.name, quant, llama_cpp, safety=safety)
        ensure_source_untouched(lm, out)  # uyuşmazlıkta `out` silinir
    except SourceModified:
        _err("Hata: kaynak model dosyaları işlem sırasında değişmiş görünüyor (SHA-256 uyuşmuyor); çıktı silindi.")
        return EXIT_ERROR
    except Exception as e:
        _err(f"Kayıt başarısız: {e}")
        return EXIT_ERROR

    print(f"{count} müdahale uygulandı, kaydedildi: {result}")
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neshter",
        description="Neşter: yerel dil modellerinin dikkat head'lerini inceler ve ablasyon yapar.",
    )
    sub = p.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="yerel sunucuyu başlat ve tarayıcıyı aç (varsayılan)")
    serve.add_argument("--no-browser", action="store_true", help="tarayıcıyı açma")
    serve.add_argument("--port", type=_port, default=0, help="sabit port (varsayılan: boş bir port)")
    serve.set_defaults(func=_serve)

    apply = sub.add_parser("apply", help="bir .nester.json tarifini modele uygula ve kaydet")
    apply.add_argument("recipe", type=Path, help="tarif dosyası (.nester.json)")
    apply.add_argument("model", type=Path, help="model klasörü, .gguf veya .pt dosyası")
    apply.add_argument("--out", type=Path, required=True, help="çıktı klasörü (var olmamalı)")
    apply.add_argument("--format", choices=_FORMATS, default="safetensors")
    apply.add_argument("--force", action="store_true", help="SHA-256 uyuşmazlığına rağmen uygula")
    apply.add_argument(
        "--label-safety", action="store_true",
        help="çıktıyı güvenlik-ablasyonu olarak etiketle (tarifteki yön bir reddetme yönüyse)",
    )
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
