"""Yerel FastAPI sunucusu: belirteç/Origin/Host koruması, oturum modeli ve arka plan işleri.

Güvenlik (Global Constraints): sunucu yalnızca 127.0.0.1'e bağlanır (bağlama uvicorn tarafında);
her /api isteği (SSE dahil) `X-Neshter-Token` ister, Origin yoksa kabul edilir, varsa yalnızca
yerel adres olabilir, Host da yerel adres olmalıdır (DNS-rebinding). CORS ara katmanı yoktur.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import hmac
import json
import re
import shutil
import subprocess
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm
from pydantic import BaseModel, Field, model_validator

import neshter
from neshter.activity import head_output_power, normalize_global, normalize_per_layer, tokens, top_mlp_neurons
from neshter.arch import estimate_memory_bytes, fit_badge
from neshter.compare import generate_pair, load_suite, run_suite, summarize
from neshter.concepts.hunt import MEASURE_TOP_K, default_method, hunt
from neshter.concepts.model import PRESET_ORDER, Concept, load_preset
from neshter.export import (
    LlamaCppMissing,
    export_gguf,
    find_llama_cpp,
    modelfile_license,
    ollama_create,
    save_safetensors,
    suggest_export_name,
    verify_source_untouched,
)
from neshter.hardware import system_info
from neshter.interventions import HeadAblation
from neshter.loader import (
    SUPPORTED_FAMILIES,
    LoadedModel,
    UnsafeModelFile,
    UnsupportedArchitecture,
    load_model,
)
from neshter.metrics import DEFAULT_N_RANDOM, kl_effect, load_neutral, perplexity, real_effect, recipe_effect
from neshter.progress import Cancelled, ProgressFn, check_cancel, sub_progress
from neshter.recipe import RECIPE_TEXT, DirectionOp, HeadOp, Recipe, SourceRef, check_against, to_interventions
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
    dtype: Literal["auto", "bfloat16", "float32"] = "auto"  # auto = politika (loader.pick_compute_dtype)


class EffectReq(_Req):
    heads: list[tuple[int, int]] = Field(min_length=1)
    prompts: list[str] | None = Field(default=None, min_length=1)
    n_random: int = Field(DEFAULT_N_RANDOM, ge=20, le=50)  # kullanıcıya dönük alt sınır 20; kütüphane en az 1


class HuntReq(_Req):
    concept_id: str | None = None
    concept: Concept | None = None
    method: Literal["attribution", "activation_diff", "direction"] | None = None  # None: default_method(lm)
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
    target: str | None = None  # None: özet yok (ör. "tone"un test kategorisi yok)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)


class ExportReq(_Req):
    recipe_ops: list[Op] = Field(min_length=1)
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # None: suggest_export_name
    format: Literal["safetensors", "gguf_q4_k_m", "gguf_q8_0"]
    ollama: bool
    out_dir: str = Field(min_length=1)
    concept: str | None = Field(default=None, pattern=RECIPE_TEXT)  # "refusal" ise güvenlik-ablasyonu işareti

    @model_validator(mode="after")
    def _ollama_needs_gguf(self) -> ExportReq:
        if self.ollama and self.format == "safetensors":
            raise ValueError("Ollama'ya eklemek için GGUF biçimi seçilmeli.")
        if self.out_dir.startswith(("\\\\", "//")):
            # UNC (\\sunucu\paylaşım): makinenin uzak bir paylaşıma kimlik doğrulaması yapmasını önler.
            raise ValueError("Ağ (UNC) yolları desteklenmiyor; yerel bir klasör seçin.")
        if not Path(self.out_dir).is_absolute():
            raise ValueError("out_dir mutlak bir yol olmalı (ör. C:\\Modeller\\yeni-model).")
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
        # Bu modelde safety_ablation'lı kavramlardan (ör. refusal) bulunan birim yönler (CPU float32).
        self.safety_dirs: list[torch.Tensor] = []
        # Aynı kavramlardan bulunmuş (katman, head) çiftleri (avın ölçülen ilk MEASURE_TOP_K head'i).
        self.safety_heads: set[tuple[int, int]] = set()

    def require(self) -> LoadedModel:
        if self.lm is None:
            raise NoModel("Henüz model yüklenmedi. Önce bir model yükleyin.")
        return self.lm

    def free(self) -> None:
        self.lm = None
        self.safety_dirs = []
        self.safety_heads = set()
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


SAFETY_COS = 0.99
REFUSAL_SUITE_CATEGORY = "harmful_refusal"  # reddetme dedektörünün (check: refusal) test kategorisi


def _is_safety_concept(c: Concept) -> bool:
    """Av bir güvenlik avı mı? Bayrak, "refusal" kimliği ya da reddetme dedektörünün kategorisi (özel kavramlar
    bayrağı kapalı bırakıp yine de reddetmeyi arıyor olabilir; etiket bayrağa bağlı kalmaz)."""
    return c.safety_ablation or c.id == "refusal" or c.suite_target == REFUSAL_SUITE_CATEGORY


def _is_safety(session: Session, ops: list, concept: str | None) -> bool:
    """Kayıt güvenlik-ablasyonu sayılır mı? İstemcinin "refusal" demesi YA DA sunucu tarafı eşleşme:
    tarifteki herhangi bir head, bu oturumda güvenlik kavramından bulunmuş bir head'dir YA DA herhangi bir
    yön, bulunmuş bir yöne |cos| ≥ 0.99. Tek bir eşleşme yeter; istemci etiketi kaldıramaz."""
    if concept == "refusal":
        return True
    if any(isinstance(op, HeadOp) and (op.layer, op.head) in session.safety_heads for op in ops):
        return True
    if not session.safety_dirs:
        return False
    dirs = [op for op in ops if isinstance(op, DirectionOp)]
    if not dirs:
        return False
    for iv in to_interventions(Recipe.model_construct(interventions=dirs)):
        v = iv.direction.detach().to("cpu", torch.float32)
        v = v / v.norm()
        for s in session.safety_dirs:
            if v.shape == s.shape and abs(float(v @ s)) >= SAFETY_COS:
                return True
    return False


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


_DTYPES = {"auto": None, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _memory_bytes(fmt: str, size_bytes: int) -> int:
    # GGUF yüklenirken bf16'ya açılır; ~Q4 varsayımıyla parametre sayısı tahmin edilir (muhafazakâr).
    # Yükleme politikası CPU'da da GPU'da da bf16'dır → parametre başına 2 bayt.
    n_params = size_bytes / 0.6 if fmt == "gguf" else size_bytes / 2
    return estimate_memory_bytes(int(n_params), 2)


def _load(
    path: Path,
    dtype: torch.dtype | None,
    name: str | None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> LoadedModel:
    """load_model; beklenmeyen (çoğu İngilizce) hatalar Türkçe bir öneke sarılır."""
    try:
        return load_model(path, device="auto", dtype=dtype, name=name, progress=progress, cancel=cancel)
    except (UnsupportedArchitecture, UnsafeModelFile, Cancelled):
        raise
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {type(e).__name__}: {e}") from e


def _default_effect_prompts() -> list[str]:
    """Ayrılmış nötr istekler (asla test setinden değil; spec §3)."""
    return load_neutral()


def _recipe_metrics(
    lm: LoadedModel,
    interventions: list,
    n_random: int = DEFAULT_N_RANDOM,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    """Tarif düzeyi metrikler (nötr set): KL + aynı biçimde rastgele kontrol, perplexity önce/sonra.

    Hedef davranış ORANI için rastgele kontrol v0.2'de (test setini ikinci kez koşturmak gerekir).
    İlerleme 0..1 aralığındadır (çağıran sub_progress ile eşler); iptal çekilişler arasında kontrol edilir.
    """
    neutral = load_neutral()
    out: dict = {"n_prompts": len(neutral), "kl_random_note": None}
    try:
        eff = recipe_effect(
            lm, interventions, neutral, n_random=n_random, progress=sub_progress(progress, 0.0, 0.9), cancel=cancel
        )
        out.update(
            kl=eff.kl, kl_std=eff.kl_std, kl_random=eff.random_kl, kl_random_p=eff.p_value, kl_percentile=eff.percentile,
            direction_control=eff.direction_control,
        )
    except ValueError as e:  # ör. neredeyse tüm head'ler seçili: aynı sayıda rastgele kontrol yok
        values = kl_effect(lm, neutral, interventions)
        out.update(
            kl=sum(values) / len(values), kl_std=None, kl_random=None, kl_random_p=None, kl_percentile=None,
            direction_control=None, kl_random_note=str(e),
        )
    check_cancel(cancel)
    if progress is not None:
        progress(0.9, "Perplexity ölçülüyor")
    out["perplexity_orig"] = perplexity(lm, neutral)
    check_cancel(cancel)
    out["perplexity_int"] = perplexity(lm, neutral, interventions)
    if progress is not None:
        progress(1.0, "Metrikler hazır")
    return out


def _sse(event: str, status) -> str:
    return f"event: {event}\ndata: {json.dumps(asdict(status), ensure_ascii=False)}\n\n"


class _Cancelled(Exception):
    pass


async def job_stream(jobs: JobManager, job_id: str, request, poll: float = 0.1, keepalive: float = 15.0):
    """SSE akışı (async: threadpool işçisi tutmaz). İstemci koparsa hemen biter; olay yokken
    `keepalive` saniyede bir yorum satırı gönderir (proxy/tarayıcı bağlantıyı kapatmasın)."""
    last = None
    idle = 0.0
    while True:
        if await request.is_disconnected():
            return
        st = jobs.status(job_id)
        key = (st.state, st.progress, st.message)
        if key != last:
            last = key
            idle = 0.0
            if st.state == "error":
                yield _sse("error", st)
                return
            if st.state in TERMINAL:  # done ya da cancelled
                yield _sse("done", st)
                return
            yield _sse("progress", st)
        await asyncio.sleep(poll)
        idle += poll
        if idle >= keepalive:
            idle = 0.0
            yield ": keepalive\n\n"


_PLACEHOLDER_HTML = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Neşter</title></head>
<body style="font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem">
<h1>Neşter</h1>
<p>Arayüz henüz derlenmedi; yerel API çalışıyor.</p>
<p lang="en">The web UI has not been built yet; the local API is running.</p>
</body></html>
"""


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
                    "license": e.license,
                    "arch": (arch := e.arch),
                    "supported": arch is not None and arch["family"] in SUPPORTED_FAMILIES,
                }
            )
        return out

    @app.get("/api/concepts")
    def get_concepts():
        out = []
        for cid in PRESET_ORDER:
            c = load_preset(cid)
            out.append(
                {
                    "id": c.id,
                    "name_tr": c.name_tr,
                    "name_en": c.name_en,
                    "safety_ablation": c.safety_ablation,
                    "suite_target": c.suite_target,
                    "positive": c.positive,
                    "negative": c.negative,
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
                lm = _load(path, _DTYPES[body.dtype], body.name, report, cancel)
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
                "dtype": str(next(lm.model.parameters()).dtype).removeprefix("torch."),
            }

        return start(fn)

    # -- anlık (eşzamanlı) uçlar --

    @contextmanager
    def _model_busy_guard():
        """Eşzamanlı uçlar kilidi beklemez: model bir işte meşgulse hemen 409 (threadpool tükenmesin)."""
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "Model şu an bir işte meşgul; iş bitince tekrar deneyin.")
        try:
            yield
        finally:
            session.lock.release()

    @app.post("/api/activity")
    def activity(body: ActivityReq):
        with _model_busy_guard(), _bad_input():
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
        with _model_busy_guard(), _bad_input():
            top = top_mlp_neurons(require_model(), body.prompt, body.token_index)
        return [{"layer": layer, "neuron": neuron, "value": value} for layer, neuron, value in top]

    @app.post("/api/compare")
    def compare(body: CompareReq):
        with _model_busy_guard(), _bad_input():
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
                return asdict(
                    real_effect(
                        lm, heads, prompts, n_random=body.n_random,
                        progress=sub_progress(report, 0.1, 1.0), cancel=cancel,
                    )
                )

        return start(fn)

    @app.post("/api/hunt")
    def hunt_endpoint(body: HuntReq):
        require_model()
        if body.concept is not None:
            concept = body.concept
            if not re.fullmatch(RECIPE_TEXT, concept.id):
                # Tarife ve dışa aktarıma girecek kimlik; avlanabilir ama kaydedilemez bir kavram olmasın.
                raise HTTPException(
                    422,
                    "Kavram kimliği yalnızca harf, rakam, '.', '_', '-' içerebilir (en fazla 64 karakter): "
                    f"{concept.id[:64]!r}.",
                )
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
                method = body.method or default_method(lm)
                report(0.0, "Kavram aranıyor")
                r = hunt(lm, concept, method, seed=body.seed, progress=report, cancel=cancel)
                # Etiketler yalnızca av tamamen bittiyse ve iş iptal edilmediyse oturuma girer (geç iptal
                # edilen av iz bırakmaz). Bu yön/head'ler kaydedilirse etiket istemciye bağlı kalmaz.
                if _is_safety_concept(concept) and not cancel.is_set():
                    if r.direction is not None:
                        d = r.direction.detach().to("cpu", torch.float32)
                        session.safety_dirs.append(d / d.norm())
                    session.safety_heads.update((layer, head) for layer, head, _ in r.heads[:MEASURE_TOP_K])
            measured = {(m.layer, m.head): m for m in r.measured}

            def _measured(layer: int, head: int):
                m = measured.get((layer, head))
                return None if m is None else {"signal_drop": m.signal_drop, "neutral_kl": m.neutral_kl}

            return {
                "method": method,
                "layer_scores": r.layer_scores,
                "heads": [
                    {"layer": layer, "head": head, "score": score, "measured": _measured(layer, head)}
                    for layer, head, score in r.heads
                ],
                "direction": None if r.direction is None else _b64_f32(r.direction),
                "direction_shape": None if r.direction is None else list(r.direction.shape),
                "direction_layer": r.direction_layer,
                "ref_layer": r.ref_layer,
                "val_accuracy": r.val_accuracy,
                "val_ci": list(r.val_ci),
                "val_n": r.val_n,
            }

        return start(fn)

    @app.post("/api/suite")
    def suite(body: SuiteReq):
        lm = require_model()
        items = load_suite()
        if body.target is not None and body.target not in {i.category for i in items}:
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
                    progress=lambda done, total: report(0.95 * done / total, f"{done}/{total} adım"),
                    cancel=cancel,
                    max_new_tokens=body.max_new_tokens,
                    target=body.target,
                )
                if cancel.is_set():
                    return None
                report(0.95, "KL ve perplexity ölçülüyor")
                metrics = _recipe_metrics(lm, interventions, progress=sub_progress(report, 0.95, 1.0), cancel=cancel)
            if cancel.is_set():
                return None
            return {
                "categories": [asdict(r) for r in results],
                "summary": None if body.target is None else asdict(summarize(results, body.target)),
                "metrics": metrics,
            }

        return start(fn)

    @app.post("/api/export")
    def export(body: ExportReq):
        lm = require_model()
        with _bad_input():
            _recipe_from_ops(lm, body.recipe_ops, body.concept)

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                safety = _is_safety(session, body.recipe_ops, body.concept)
                concept = "refusal" if safety else body.concept  # istemci etiketi kaldıramaz
                recipe = _recipe_from_ops(lm, body.recipe_ops, concept)
                name = body.name or suggest_export_name(lm, safety)
                out_dir = Path(body.out_dir)
                if body.format == "safetensors":
                    path = save_safetensors(
                        lm, recipe, out_dir, safety_ablated=safety,
                        progress=sub_progress(report, 0.0, 0.9), cancel=cancel,
                    )
                else:
                    llama_cpp = find_llama_cpp()
                    if llama_cpp is None:
                        raise LlamaCppMissing(
                            "llama.cpp bulunamadı. GGUF için NESHTER_LLAMA_CPP ortam değişkenini llama.cpp "
                            "klasörüne ayarlayın ya da llama-quantize'ı PATH'e ekleyin."
                        )
                    path = export_gguf(
                        lm, recipe, out_dir, name, _QUANT[body.format], llama_cpp, safety,
                        progress=sub_progress(report, 0.0, 0.9), cancel=cancel,
                    )
                report(0.9, "Kaynak model doğrulanıyor")
                if not verify_source_untouched(lm):
                    raise RuntimeError(
                        "Kaynak model dosyaları kayıt sırasında değişmiş görünüyor (SHA-256 uyuşmuyor)."
                    )
                if cancel.is_set():  # son adımda gelen iptal de çıktıyı bırakmaz (out_dir bu işte oluştu)
                    shutil.rmtree(out_dir, ignore_errors=True)
                    return None
                if body.ollama:
                    report(0.95, "Ollama'ya ekleniyor")
                    try:
                        ollama_create(name, path, license_text=modelfile_license(lm, out_dir))
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(
                            f"Ollama'ya ekleme başarısız oldu ('ollama create' {e.returncode} koduyla çıktı)."
                        ) from e
            return {
                "path": str(path),
                "source_sha256": lm.source_sha256,
                "source_untouched": True,
                "safety_ablated": safety,
                "name": name,
            }

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
    def job_events(job_id: str, request: Request):
        job_status(job_id)
        return StreamingResponse(
            job_stream(jobs, job_id, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- statik arayüz (yalnızca derlenmişse; API rotaları önceliklidir) --

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return _PLACEHOLDER_HTML

    return app
