"""Yerel FastAPI sunucusu: belirteç/Origin/Host koruması, oturum modeli ve arka plan işleri.

Güvenlik (Global Constraints): sunucu yalnızca 127.0.0.1'e bağlanır (bağlama uvicorn tarafında);
her /api isteği (SSE dahil) `X-Neshter-Token` ister, Origin yoksa kabul edilir, varsa yalnızca
yerel adres olabilir, Host da yerel adres olmalıdır (DNS-rebinding). CORS ara katmanı yoktur.
"""

from __future__ import annotations

import base64
import gc
import hmac
import json
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm
from pydantic import BaseModel, Field, model_validator

import neshter
from neshter.activity import head_output_power, normalize_global, normalize_per_layer, tokens, top_mlp_neurons
from neshter.arch import estimate_memory_bytes, fit_badge
from neshter.compare import generate_pair, load_suite, run_suite, summarize
from neshter.concepts.hunt import hunt
from neshter.concepts.model import Concept, load_preset
from neshter.export import find_llama_cpp, ollama_create, save_safetensors, to_gguf
from neshter.hardware import system_info
from neshter.interventions import HeadAblation
from neshter.loader import LoadedModel, UnsafeModelFile, UnsupportedArchitecture, load_model, read_license
from neshter.metrics import real_effect
from neshter.recipe import DirectionOp, HeadOp, Recipe, SourceRef, check_against, to_interventions
from neshter.scan import scan_all
from neshter.server.jobs import TERMINAL, JobManager

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
SAMPLE_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
_QUANT = {"gguf_q4_k_m": "Q4_K_M", "gguf_q8_0": "Q8_0"}

Op = Annotated[HeadOp | DirectionOp, Field(discriminator="type")]


# --- İstek şemaları (bilinmeyen alan yasak) -------------------------------------------------------


class _Req(BaseModel, extra="forbid"):
    pass


class ActivityReq(_Req):
    prompt: str = Field(min_length=1)
    token_index: int
    scale: Literal["layer", "global"]


class MlpReq(_Req):
    prompt: str = Field(min_length=1)
    token_index: int


class LoadReq(_Req):
    path: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1, max_length=256)  # görünen ad (ör. "qwen2.5:7b")
    dtype: Literal["auto", "bfloat16", "float32"] = "auto"  # auto = bf16 (CPU ve GPU'da aynı)


class EffectReq(_Req):
    heads: list[tuple[int, int]] = Field(min_length=1)
    prompts: list[str] | None = Field(default=None, min_length=1)


class HuntReq(_Req):
    concept_id: str | None = None
    concept: Concept | None = None
    method: Literal["attribution", "activation_diff", "direction"]
    seed: int = 0

    @model_validator(mode="after")
    def _one_concept(self) -> HuntReq:
        if (self.concept_id is None) == (self.concept is None):
            raise ValueError("concept_id ya da concept'ten tam biri verilmeli.")
        return self


class CompareReq(_Req):
    prompt: str = Field(min_length=1)
    recipe_ops: list[Op] = Field(min_length=1)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)


class SuiteReq(_Req):
    recipe_ops: list[Op] = Field(min_length=1)
    target: str
    max_new_tokens: int = Field(default=200, ge=1, le=1024)


class ExportReq(_Req):
    recipe_ops: list[Op] = Field(min_length=1)
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    format: Literal["safetensors", "gguf_q4_k_m", "gguf_q8_0"]
    ollama: bool
    out_dir: str = Field(min_length=1)
    concept: str | None = Field(default=None, max_length=64)  # "refusal" ise güvenlik-ablasyonu işareti

    @model_validator(mode="after")
    def _ollama_needs_gguf(self) -> ExportReq:
        if self.ollama and self.format == "safetensors":
            raise ValueError("Ollama'ya eklemek için GGUF biçimi seçilmeli.")
        return self


class RecipeCheckReq(_Req):
    recipe: Recipe


# --- Oturum ---------------------------------------------------------------------------------------


class NoModel(Exception):
    pass


class Session:
    """Tek yüklü model. `lock`, modeli kullanan her işlemi (kanca kuran işler dahil) sıraya sokar."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lm: LoadedModel | None = None

    def require(self) -> LoadedModel:
        if self.lm is None:
            raise NoModel("Henüz model yüklenmedi. Önce bir model yükleyin.")
        return self.lm

    def free(self) -> None:
        self.lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class InvalidRecipe(ValueError):
    pass


def _recipe_from_ops(lm: LoadedModel, ops: list, concept: str | None = None) -> Recipe:
    """İstekteki işlemleri tarif modeline sokar ve yüklü modelle denetler (uymazsa InvalidRecipe)."""
    recipe = Recipe(
        schema_version=1,
        neshter_version=neshter.__version__,
        source_model=SourceRef(name=lm.name, sha256=lm.source_sha256),
        concept=concept,
        method="api",
        seed=0,
        dataset_sha256=None,
        head_mode="zero",
        interventions=ops,
        created_at=datetime.now(timezone.utc),
    )
    check = check_against(recipe, lm.arch, lm.source_sha256)
    if not check.ok:
        raise InvalidRecipe(" ".join(check.errors))
    return recipe


@contextmanager
def _bad_input():
    try:
        yield
    except (ValueError, IndexError) as e:
        raise HTTPException(422, str(e)) from e


# --- Güvenlik ara katmanı (saf ASGI; akışlı yanıtları bozmaz) ------------------------------------


class _Guard:
    def __init__(self, app, token: str, port: int) -> None:
        self.app = app
        self._token = token.encode("utf-8")
        self._hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self._origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def _deny(self, path: str, headers: dict[str, str]) -> str | None:
        if headers.get("host", "").lower() not in self._hosts:
            return "Geçersiz Host başlığı."
        origin = headers.get("origin")
        if origin is not None and origin.lower() not in self._origins:
            return "Geçersiz Origin."
        if path == "/api" or path.startswith("/api/"):
            given = headers.get("x-neshter-token")
            if given is None or not hmac.compare_digest(given.encode("utf-8"), self._token):
                return "Geçersiz ya da eksik X-Neshter-Token."
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        reason = self._deny(scope["path"], headers)
        if reason is None:
            await self.app(scope, receive, send)
        elif scope["type"] == "http":
            await JSONResponse({"detail": reason}, status_code=403)(scope, receive, send)
        else:
            await send({"type": "websocket.close", "code": 1008})


# --- Yardımcılar ----------------------------------------------------------------------------------


def _b64_f32(t: torch.Tensor) -> str:
    return base64.b64encode(np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype="<f4").tobytes()).decode("ascii")


_DTYPES = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _memory_bytes(fmt: str, size_bytes: int) -> int:
    # GGUF yüklenirken bf16'ya açılır; ~Q4 varsayımıyla parametre sayısı tahmin edilir (muhafazakâr).
    # Yükleme politikası CPU'da da GPU'da da bf16'dır → parametre başına 2 bayt.
    n_params = size_bytes / 0.6 if fmt == "gguf" else size_bytes / 2
    return estimate_memory_bytes(int(n_params), 2)


def _load(path: Path, dtype: torch.dtype, name: str | None) -> LoadedModel:
    """load_model; beklenmeyen (çoğu İngilizce) hatalar Türkçe bir öneke sarılır."""
    try:
        return load_model(path, device="auto", dtype=dtype, name=name)
    except (UnsupportedArchitecture, UnsafeModelFile):
        raise
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {type(e).__name__}: {e}") from e


def _default_effect_prompts() -> list[str]:
    items = load_suite()
    return [i.prompt for i in items[:: max(1, len(items) // 8)][:8]]


def _sse(event: str, status) -> str:
    return f"event: {event}\ndata: {json.dumps(asdict(status), ensure_ascii=False)}\n\n"


class _Cancelled(Exception):
    pass


# --- Uygulama -------------------------------------------------------------------------------------


def create_app(token: str, port: int, roots: dict | None = None) -> FastAPI:
    if not token:
        raise ValueError("Boş belirteçle sunucu kurulamaz.")
    session = Session()
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="Neşter", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session = session
    app.state.jobs = jobs
    app.add_middleware(_Guard, token=token, port=port)

    @app.exception_handler(NoModel)
    async def _no_model(_, exc: NoModel):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def require_model() -> LoadedModel:
        return session.require()

    def start(fn) -> dict:
        return {"job_id": jobs.submit(fn)}

    # -- sistem / modeller --

    @app.get("/api/system")
    def get_system():
        return asdict(system_info())

    @app.get("/api/models")
    def get_models():
        info = system_info()
        out = []
        for e in scan_all(roots):
            mem = _memory_bytes(e.fmt, e.size_bytes)
            out.append(
                {
                    "name": e.name,
                    "source": e.source,
                    "path": str(e.path),
                    "fmt": e.fmt,
                    "size_bytes": e.size_bytes,
                    "memory_bytes": mem,
                    "badge": fit_badge(mem, info.vram_bytes, info.ram_bytes),
                    "license": read_license(e.path) if e.path.is_dir() else None,
                }
            )
        return out

    @app.post("/api/models/sample")
    def download_sample():
        def fn(report, cancel):
            class Bar(hf_tqdm):
                """İndirme çubuğunu konsola yazmadan bayt ilerlemesine çevirir; iptalde indirmeyi keser."""

                def __init__(self, *a, **kw):
                    kw["disable"] = True
                    super().__init__(*a, **kw)
                    self._seen = kw.get("initial", 0) or 0

                def update(self, n=1):
                    if cancel.is_set():
                        raise _Cancelled()
                    self._seen += n or 0
                    if self.total:
                        report(
                            self._seen / self.total,
                            f"{self._seen / 1e6:.0f} / {self.total / 1e6:.0f} MB indirildi",
                        )
                    return True

            report(0.0, "İndirme başlıyor")
            return {"path": str(snapshot_download(SAMPLE_REPO, tqdm_class=Bar))}

        return start(fn)

    # -- oturum --

    @app.post("/api/session/load")
    def load(body: LoadReq):
        path = Path(body.path)
        if not path.exists():
            raise HTTPException(422, f"'{path}' bulunamadı.")

        def fn(report, cancel):
            with session.lock:
                report(0.05, "Önceki model bırakılıyor")
                session.free()
                if cancel.is_set():
                    return None
                report(0.1, "Model yükleniyor")
                lm = _load(path, _DTYPES[body.dtype], body.name)
                if cancel.is_set():  # yükleme sürerken iptal edildi: durum "cancelled", oturum boş kalsın
                    del lm
                    session.free()
                    return None
                session.lm = lm
            return {
                "arch": asdict(lm.arch),
                "name": lm.name,
                "license": lm.license,
                "device": next(lm.model.parameters()).device.type,
                "device_note": lm.device_note,
            }

        return start(fn)

    # -- anlık (eşzamanlı) uçlar --

    @app.post("/api/activity")
    def activity(body: ActivityReq):
        with session.lock, _bad_input():
            lm = require_model()
            power = head_output_power(lm, body.prompt)  # [L, H, T]
            n_tokens = power.shape[-1]
            if not -n_tokens <= body.token_index < n_tokens:
                raise ValueError(f"token_index {body.token_index} aralık dışı (prompt {n_tokens} token).")
            at = power[:, :, body.token_index]
            norm = normalize_per_layer(at) if body.scale == "layer" else normalize_global(at)
            return {"tokens": tokens(lm, body.prompt), "power": norm.tolist()}

    @app.post("/api/mlp")
    def mlp(body: MlpReq):
        with session.lock, _bad_input():
            top = top_mlp_neurons(require_model(), body.prompt, body.token_index)
        return [{"layer": layer, "neuron": neuron, "value": value} for layer, neuron, value in top]

    @app.post("/api/compare")
    def compare(body: CompareReq):
        with session.lock, _bad_input():
            lm = require_model()
            interventions = to_interventions(_recipe_from_ops(lm, body.recipe_ops))
            original, modified, mode = generate_pair(lm, body.prompt, interventions, max_new_tokens=body.max_new_tokens)
        return {"original": original, "modified": modified, "mode": mode}

    @app.post("/api/recipe/check")
    def recipe_check(body: RecipeCheckReq):
        lm = require_model()
        return asdict(check_against(body.recipe, lm.arch, lm.source_sha256))

    # -- işler --

    @app.post("/api/effect")
    def effect(body: EffectReq):
        arch = require_model().arch
        for layer, head in body.heads:
            if not (0 <= layer < arch.n_layers and 0 <= head < arch.n_heads):
                raise HTTPException(
                    422, f"Head ({layer}, {head}) bu modelde yok (katman 0..{arch.n_layers - 1}, head 0..{arch.n_heads - 1})."
                )
        heads = [HeadAblation(layer, head) for layer, head in body.heads]
        prompts = body.prompts or _default_effect_prompts()

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                if cancel.is_set():
                    return None
                report(0.1, "Gerçek etki hesaplanıyor")
                return asdict(real_effect(lm, heads, prompts))

        return start(fn)

    @app.post("/api/hunt")
    def hunt_endpoint(body: HuntReq):
        require_model()
        if body.concept is not None:
            concept = body.concept
        else:
            try:
                concept = load_preset(body.concept_id)
            except KeyError as e:
                raise HTTPException(422, f"Bilinmeyen hazır kavram: {body.concept_id!r}.") from e

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                if cancel.is_set():
                    return None
                report(0.1, "Kavram aranıyor")
                r = hunt(lm, concept, body.method, seed=body.seed)
            return {
                "layer_scores": r.layer_scores,
                "heads": [{"layer": layer, "head": head, "score": score} for layer, head, score in r.heads],
                "direction": None if r.direction is None else _b64_f32(r.direction),
                "direction_shape": None if r.direction is None else list(r.direction.shape),
                "direction_layer": r.direction_layer,
                "val_consistency": r.val_consistency,
            }

        return start(fn)

    @app.post("/api/suite")
    def suite(body: SuiteReq):
        lm = require_model()
        items = load_suite()
        if body.target not in {i.category for i in items}:
            raise HTTPException(422, f"Hedef kategori '{body.target}' test setinde yok.")
        with _bad_input():
            _recipe_from_ops(lm, body.recipe_ops)

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                interventions = to_interventions(_recipe_from_ops(lm, body.recipe_ops))
                results = run_suite(
                    lm,
                    items,
                    interventions,
                    progress=lambda done, total: report(done / total, f"{done}/{total} test"),
                    cancel=cancel,
                    max_new_tokens=body.max_new_tokens,
                )
            if cancel.is_set():
                return None
            return {"categories": [asdict(r) for r in results], "summary": asdict(summarize(results, body.target))}

        return start(fn)

    @app.post("/api/export")
    def export(body: ExportReq):
        lm = require_model()
        with _bad_input():
            _recipe_from_ops(lm, body.recipe_ops, body.concept)

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                recipe = _recipe_from_ops(lm, body.recipe_ops, body.concept)
                out_dir = Path(body.out_dir)
                if body.format == "safetensors":
                    report(0.1, "Ağırlıklara gömülüyor ve kaydediliyor")
                    return {"path": str(save_safetensors(lm, recipe, out_dir))}
                return {"path": str(_export_gguf(lm, recipe, out_dir, body, report))}

        return start(fn)

    # -- iş durumu --

    def job_status(job_id: str):
        try:
            return jobs.status(job_id)
        except KeyError as e:
            raise HTTPException(404, "İş bulunamadı.") from e

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return asdict(job_status(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job_status(job_id)
        jobs.cancel(job_id)
        return asdict(jobs.status(job_id))

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        job_status(job_id)

        def stream():
            last = None
            while True:
                st = jobs.status(job_id)
                key = (st.state, st.progress, st.message)
                if key != last:
                    last = key
                    if st.state == "error":
                        yield _sse("error", st)
                        return
                    if st.state in TERMINAL:  # done ya da cancelled
                        yield _sse("done", st)
                        return
                    yield _sse("progress", st)
                time.sleep(0.1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # -- statik arayüz (yalnızca derlenmişse; API rotaları önceliklidir) --

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def _export_gguf(lm: LoadedModel, recipe: Recipe, out_dir: Path, body: ExportReq, report) -> Path:
    """safetensors ara klasörü → GGUF (→ Ollama). out_dir yeni olmalı; ara klasör işin sonunda silinir."""
    llama_cpp = find_llama_cpp()
    if llama_cpp is None:
        raise FileNotFoundError(
            "llama.cpp bulunamadı. GGUF için NESHTER_LLAMA_CPP ortam değişkenini llama.cpp klasörüne "
            "ayarlayın ya da llama-quantize'ı PATH'e ekleyin."
        )
    if out_dir.exists():
        raise FileExistsError(f"'{out_dir}' zaten var; üzerine yazılmaz. Başka bir klasör adı seçin.")
    try:
        hf_dir = out_dir / "_hf"
        report(0.1, "Ağırlıklara gömülüyor ve kaydediliyor")
        save_safetensors(lm, recipe, hf_dir)
        report(0.5, "GGUF'a dönüştürülüyor")
        try:
            gguf = to_gguf(hf_dir, out_dir / f"{body.name}.gguf", _QUANT[body.format], llama_cpp)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF dönüştürme başarısız oldu (llama.cpp {e.returncode} koduyla çıktı).") from e
        shutil.rmtree(hf_dir)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)  # out_dir bu işte oluştu (başta yoktu)
        raise
    if body.ollama:
        report(0.9, "Ollama'ya ekleniyor")
        try:
            ollama_create(body.name, gguf)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama'ya ekleme başarısız oldu ('ollama create' {e.returncode} koduyla çıktı).") from e
    return gguf
