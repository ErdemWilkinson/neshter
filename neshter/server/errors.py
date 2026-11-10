"""Hata kodları: iş ve HTTP hataları arayüze sabit, İngilizce `snake_case` bir `error_code` ile gider.

Kullanıcıya gösterilen metin (`detail` / `error`) Türkçe kalır; arayüz davranışı koda göre seçer.
Bu modül `app`'i içe aktarmaz (jobs.py ve app.py ikisi de buradan alır).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from neshter.compare import _is_oom
from neshter.export import DiskFull, LlamaCppMissing, SourceModified
from neshter.loader import UnsafeModelFile, UnsupportedArchitecture
from neshter.progress import Cancelled


log = logging.getLogger("neshter.server")

# Derlenmiş yerel arayüz için yeterli en dar CSP: satır içi betik yok; stil için 'unsafe-inline' (React stil öznitelikleri).
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class NoModel(Exception):
    """Oturumda yüklü model yok (HTTP 409 `no_model`)."""


def _code_of(exc: BaseException) -> str | None:
    if isinstance(exc, Cancelled):
        return "cancelled"
    if isinstance(exc, NoModel):
        return "no_model"
    if isinstance(exc, UnsupportedArchitecture):
        return "unsupported_arch"
    if isinstance(exc, UnsafeModelFile):
        return "unsafe_file"
    # LlamaCppMissing bir FileNotFoundError'dır: "not_found"tan önce denetlenmeli.
    if isinstance(exc, LlamaCppMissing):
        return "llama_cpp_missing"
    if isinstance(exc, DiskFull):
        return "disk_full"
    if isinstance(exc, SourceModified):
        return "source_modified"
    if isinstance(exc, FileExistsError):
        return "exists"
    if isinstance(exc, MemoryError) or _is_oom(exc):
        return "oom"
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, ValueError):
        return "invalid_input"
    return None


def error_code_for(exc: BaseException) -> str:
    """`exc`, sonra `__cause__`/`__context__` zinciri; ilk eşleşen kod kazanır, yoksa `internal`."""
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        e = stack.pop(0)
        if id(e) in seen:
            continue
        seen.add(id(e))
        if (code := _code_of(e)) is not None:
            return code
        stack.extend(x for x in (e.__cause__, e.__context__) if x is not None)
    return "internal"


class ApiError(HTTPException):
    """Kendi `error_code`unu taşıyan HTTP hatası."""

    def __init__(self, status_code: int, detail: str, code: str) -> None:
        super().__init__(status_code, detail)
        self.code = code


_STATUS_CODES = {403: "forbidden", 404: "not_found", 409: "busy", 422: "invalid_input"}


def _http_code(exc: StarletteHTTPException) -> str:
    if isinstance(exc, ApiError):
        return exc.code
    if exc.status_code in _STATUS_CODES:
        return _STATUS_CODES[exc.status_code]
    return "invalid_input" if 400 <= exc.status_code < 500 else "internal"


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return JSONResponse(
            {"detail": exc.detail, "error_code": _http_code(exc)}, status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(NoModel)
    async def _no_model(_: Request, exc: NoModel):
        return JSONResponse({"detail": str(exc), "error_code": "no_model"}, status_code=409)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(
            {"detail": jsonable_encoder(exc.errors()), "error_code": "invalid_input"}, status_code=422
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        # Yığın izi ve istisna metni istemciye gitmez (yol/bilgi sızıntısı); sunucu günlüğüne yazılır.
        log.error("Beklenmeyen hata: %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(
            {"detail": "Beklenmeyen hata.", "error_code": "internal"}, status_code=500, headers=SECURITY_HEADERS
        )
