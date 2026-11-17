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
from fastapi import FastAPI, HTTPException, Path as PathParam, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm
from pydantic import BaseModel, Field, model_validator

import neshter
from neshter.activity import head_output_power, normalize_global, normalize_per_layer, tokens, top_mlp_neurons
from neshter.arch import estimate_memory_bytes, fit_badge
from neshter.compare import generate_pair, load_suite, run_suite, summarize
from neshter.concepts.detectors import DETECTORS
from neshter.concepts.hunt import MEASURE_TOP_K, default_method, hunt
from neshter.concepts.model import PRESET_ORDER, Concept, load_preset
from neshter.graph import GraphStore, answer_graph, attention_patterns, cut_report, logit_lens
from neshter.export import (
    LlamaCppMissing,
    estimate_export_bytes,
    export_gguf,
    find_llama_cpp,
    modelfile_license,
    ollama_create,
    save_safetensors,
    suggest_export_name,
    ensure_source_untouched,
    is_bad_file_name,
    is_refusal,
)
from neshter.hardware import available_ram_bytes, cpu_has_native_bf16, system_info
from neshter.interventions import HeadAblation
from neshter.loader import (
    SUPPORTED_FAMILIES,
    LoadedModel,
    UnsafeModelFile,
    UnsupportedArchitecture,
    load_model,
    pick_compute_dtype,
    planned_device,
)
from neshter.metrics import (
    CPU_N_MAX_CAP,
    DEFAULT_N_MAX,
    DEFAULT_N_RANDOM,
    MIN_N_MAX,
    kl_effect,
    load_neutral,
    perplexity,
    probe_answer,
    recipe_effect,
    sequential_real_effect,
)
from neshter.progress import Cancelled, ProgressFn, check_cancel, sub_progress
from neshter.recipe import RECIPE_TEXT, DirectionOp, HeadOp, Recipe, SourceRef, check_against, to_interventions
from neshter.scan import scan_all
from neshter.server.errors import SECURITY_HEADERS, ApiError, NoModel, install_handlers
from neshter.server.jobs import TERMINAL, JobManager
from neshter.server.state import (
    DRAFT_MAX_BYTES,
    SHA_RE,
    Draft,
    DraftIn,
    DraftSummary,
    Prefs,
    StateStore,
    neshter_home,
)

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
SAMPLE_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
# Duman testinde doğrulanan anlık görüntü; "main" sessizce değişmesin. README.md, read_license'ın
# okuduğu lisans ön bilgisini taşır (yoksa Apache örnek "Qwen Apache değil" uyarısı alırdı).
SAMPLE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SAMPLE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "LICENSE*", "README.md"]
_QUANT = {"gguf_q4_k_m": "Q4_K_M", "gguf_q8_0": "Q8_0"}

Op = Annotated[HeadOp | DirectionOp, Field(discriminator="type")]

QUICK_GUARD = "general"  # hızlı kip: hedef kategori + bu "kimseyi bozmadım" kontrol kategorisi
QUICK_MAX_NEW_TOKENS = 64
QUICK_N_RANDOM = 20  # hızlı kipte tarif KL kontrolünün sabit çekiş sayısı (/api/effect sıralı testi ayrı: EffectReq.n_max)
MAX_ACTIVITY_TOKENS = 512  # /api/activity yanıtı L×H×T'ye büyür: token sayısı sınırlanır
MAX_ACTIVITY_CHARS = 8000  # tokenlaştırmadan önce ucuz üst sınır


# --- İstek şemaları (bilinmeyen alan yasak) -------------------------------------------------------


class _Req(BaseModel, extra="forbid"):
    pass


class ActivityReq(_Req):
    prompt: str = Field(min_length=1, max_length=MAX_ACTIVITY_CHARS)
    token_index: int | None = None  # None: yalnızca ham güç (tüm token'lar tek çağrıda)
    scale: Literal["layer", "global"] = "layer"


class LensReq(_Req):
    prompt: str = Field(min_length=1, max_length=MAX_ACTIVITY_CHARS)
    position: int = -1


class AttentionReq(_Req):
    prompt: str = Field(min_length=1, max_length=MAX_ACTIVITY_CHARS)
    layer: int = Field(ge=0)


class GraphBudget(_Req):
    heads_per_layer: int = Field(6, ge=1, le=32)
    neurons_per_layer: int = Field(12, ge=0, le=64)
    max_nodes: int = Field(300, ge=10, le=600)
    max_edges: int = Field(500, ge=20, le=2000)


class GraphReq(_Req):
    prompt: str = Field(min_length=1, max_length=MAX_ACTIVITY_CHARS)
    target_token_id: int | None = Field(None, ge=0)
    budget: GraphBudget = Field(default_factory=GraphBudget)


class GraphCutReq(_Req):
    edge_ids: list[str] = Field(min_length=1, max_length=64)


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
    n_max: int = Field(DEFAULT_N_MAX, ge=MIN_N_MAX, le=999)  # sıralı testin en çok çekişi; CPU'da en çok CPU_N_MAX_CAP
    seed: int = Field(0, ge=0, le=2**32 - 1)  # sabit tohum: aynı istek aynı çekişleri verir
    # Keşif ekranındaki istem: verilirse sonuç, bu istemde cevabın olasılığını head'ler açık/kapalı da taşır (yüz nokta).
    probe: str | None = Field(default=None, min_length=1, max_length=8000)


class HuntReq(_Req):
    concept_id: str | None = None
    concept: Concept | None = None
    lang: Literal["tr", "en"] = "tr"  # hazır kavramın örnek dili (en: yalnız İngilizce örnekler)
    method: Literal["attribution", "activation_diff", "direction"] | None = None  # None: default_method(lm)
    seed: int = 0
    sweeps: int = Field(1, ge=1, le=8)  # >1: "direction" yönteminde model düzeyinde tekrarlı süpürme

    @model_validator(mode="after")
    def _one_concept(self) -> HuntReq:
        if (self.concept_id is None) == (self.concept is None):
            raise ValueError("concept_id ya da concept'ten tam biri verilmeli.")
        return self


class CompareReq(_Req):
    prompt: str = Field(min_length=1)
    recipe_ops: list[Op] = Field(min_length=1)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    concept_id: str | None = None  # DETECTORS'ta varsa yanıt "target_hit" taşır


class SuiteReq(_Req):
    recipe_ops: list[Op] = Field(min_length=1)
    target: str | None = None  # None: özet yok (ör. "tone"un test kategorisi yok)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    quick: bool = False  # yalnızca hedef + QUICK_GUARD kategorisi, ≤ 64 token, 20 rastgele kontrol


def _check_out_dir(out_dir: str) -> None:
    if re.search(r"[\x00-\x1f]", out_dir):  # ör. satır sonu Modelfile'ın FROM satırına ulaşabilirdi
        raise ValueError("out_dir denetim karakteri içeremez.")
    if out_dir.startswith(("\\\\", "//")):
        # UNC (\\sunucu\paylaşım): makinenin uzak bir paylaşıma kimlik doğrulaması yapmasını önler.
        raise ValueError("Ağ (UNC) yolları desteklenmiyor; yerel bir klasör seçin.")
    if not Path(out_dir).is_absolute():
        raise ValueError("out_dir mutlak bir yol olmalı (ör. C:\\Modeller\\yeni-model).")


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
        if self.name is not None and is_bad_file_name(self.name):
            raise ValueError("Bu ad Windows'ta ayrılmış (CON, NUL, COM1...) ya da nokta/boşlukla bitiyor.")
        _check_out_dir(self.out_dir)
        return self


class EstimateReq(_Req):
    """Dışa aktarım disk tahmini. `recipe_ops` verilirse yön var mı ve güvenlik etiketi ondan türetilir
    (dışa aktarma işiyle aynı); verilmezse açık `has_directions` ve yalnızca `concept` kullanılır, ad geçicidir."""

    format: Literal["safetensors", "gguf_q4_k_m", "gguf_q8_0"]
    recipe_ops: list[Op] | None = Field(default=None, min_length=1)
    has_directions: bool = False
    out_dir: str | None = Field(default=None, min_length=1)
    concept: str | None = Field(default=None, pattern=RECIPE_TEXT)

    @model_validator(mode="after")
    def _local_out_dir(self) -> EstimateReq:
        if self.out_dir is not None:
            _check_out_dir(self.out_dir)
        return self


class RecipeCheckReq(_Req):
    recipe: Recipe


# --- Oturum ---------------------------------------------------------------------------------------


class Session:
    """Tek yüklü model. `lock`, modeli kullanan her işlemi (kanca kuran işler dahil) sıraya sokar."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lm: LoadedModel | None = None
        # Bu modelde safety_ablation'lı kavramlardan (ör. refusal) bulunan birim yönler (CPU float32).
        self.safety_dirs: list[torch.Tensor] = []
        # Aynı kavramlardan bulunmuş (katman, head) çiftleri (avın ölçülen ilk MEASURE_TOP_K head'i).
        self.safety_heads: set[tuple[int, int]] = set()
        # Bu modelle tamamlanan (iptal edilmemiş) /api/effect ölçümleri: seçilerek bulunan sonuç hipotezdir.
        self.measure_count = 0

    def require(self) -> LoadedModel:
        if self.lm is None:
            raise NoModel("Henüz model yüklenmedi. Önce bir model yükleyin.")
        return self.lm

    def free(self) -> None:
        self.lm = None
        self.safety_dirs = []
        self.safety_heads = set()
        self.measure_count = 0
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
    if is_refusal(concept):
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
    except MemoryError as e:  # ör. gömme öncesi RAM denetimi (export.ensure_bake_ram)
        raise ApiError(503, str(e) or "Bellek yetmedi.", "oom") from e


# --- Güvenlik ara katmanı (saf ASGI; akışlı yanıtları bozmaz) ------------------------------------


_SECURITY_HEADER_PAIRS = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in SECURITY_HEADERS.items()]


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

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                names = {k for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in _SECURITY_HEADER_PAIRS if k not in names]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        if reason is None:
            await self.app(scope, receive, send_with_security_headers)
        elif scope["type"] == "http":
            await JSONResponse({"detail": reason, "error_code": "forbidden"}, status_code=403)(
                scope, receive, send_with_security_headers
            )
        else:
            await send({"type": "websocket.close", "code": 1008})


class _WebFiles(StaticFiles):
    """Derlenmiş arayüz. index.html her açılışta sunucuya sorulur (yeni derleme önbellekte takılmasın);
    /assets/ altındaki adlar içerik özetli olduğundan bir yıl saklanabilir."""

    def file_response(self, full_path, stat_result, scope, status_code: int = 200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        immutable = scope["path"].startswith("/assets/")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return response


# --- Yardımcılar ----------------------------------------------------------------------------------


def _b64_f32(t: torch.Tensor) -> str:
    return base64.b64encode(np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype="<f4").tobytes()).decode("ascii")


_DTYPES = {"auto": None, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _ollama_available() -> bool:
    return shutil.which("ollama") is not None


def _check_export_tools(fmt: str, ollama: bool) -> None:
    """GGUF/Ollama dışa aktarımı için araçlar: yoksa iş başlamadan 422 (`llama_cpp_missing` / `ollama_missing`)."""
    if fmt != "safetensors":
        llama = find_llama_cpp()
        if llama is None or not (llama / "convert_hf_to_gguf.py").is_file():
            raise ApiError(
                422,
                "llama.cpp bulunamadı ya da convert_hf_to_gguf.py eksik. GGUF için NESHTER_LLAMA_CPP ortam "
                "değişkenini llama.cpp klasörüne ayarlayın (llama-quantize ve convert_hf_to_gguf.py orada olmalı).",
                "llama_cpp_missing",
            )
    if ollama and not _ollama_available():
        raise ApiError(
            422, "Ollama bulunamadı (PATH'te 'ollama' yok). Ollama'yı kurun ya da bu adımı kapatın.", "ollama_missing"
        )


def _memory_bytes(fmt: str, size_bytes: int, available_ram: int, native_bf16: bool) -> int:
    """Açınca gereken bellek: parametre sayısı tahmini × yükleme politikasının dtype'ı.

    GGUF: ~Q4 varsayımıyla (0.6 bayt/parametre, muhafazakâr); safetensors: bf16 dosya (2 bayt/parametre).
    `load_model(dtype=None)` ile aynı karar: GPU'ya sığarsa bf16; yerel bf16'sı olmayan CPU'da RAM yeterse
    float32 (parametre başına 4 bayt), değilse bf16 (2 bayt).
    """
    n_params = int(size_bytes / 0.6 if fmt == "gguf" else size_bytes / 2)
    dtype = pick_compute_dtype(planned_device("auto", n_params), n_params, available_ram, native_bf16)
    return estimate_memory_bytes(n_params, 4 if dtype == torch.float32 else 2)


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


def _unsupported_reason(arch: dict | None) -> str | None:
    if arch is None:
        return "unknown_arch"
    return None if arch["family"] in SUPPORTED_FAMILIES else "unsupported_family"


def _free_bytes(out_dir: str) -> int:
    """Boş disk: `out_dir`in (henüz yoksa) var olan en yakın atasında ölçülür; klasör yaratılmaz."""
    p = Path(out_dir)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return int(shutil.disk_usage(p).free)
    except OSError as e:  # var olmayan sürücü (ör. Q:\x), erişilemeyen birim
        raise HTTPException(422, "Hedef klasörün sürücüsü bulunamadı.") from e


def _check_cpu_cap(lm: LoadedModel, n_max: int) -> None:
    """CPU'da sıralı testin çekiş üst sınırı (ValueError -> 422 / invalid_input)."""
    if next(lm.model.parameters()).device.type == "cpu" and n_max > CPU_N_MAX_CAP:
        raise ValueError(f"CPU'da en çok {CPU_N_MAX_CAP} çekiş yapılabilir (istenen: {n_max}).")


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
        line = store.exit_line()  # Ctrl+C / /api/shutdown: taslağa yazılan kesimler için tek satır
        if line is not None:
            print(line, flush=True)

    app = FastAPI(title="Neşter", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session = session
    app.state.jobs = jobs
    app.state.request_exit = None  # cli._run_server bağlar; yoksa /api/shutdown 501 döner
    store = app.state.store = StateStore(neshter_home() / ".state")
    app.add_middleware(_Guard, token=token, port=port)

    install_handlers(app)

    def require_model() -> LoadedModel:
        return session.require()

    graphs = app.state.graphs = GraphStore()

    def start(fn, kind: str) -> dict:
        return {"job_id": jobs.submit(fn, kind=kind)}

    # -- sistem / modeller --

    @app.get("/api/system")
    def get_system():
        llama = find_llama_cpp()
        return {
            **asdict(system_info()),
            "default_out_dir": str(neshter_home()),
            "llama_cpp": {
                "found": llama is not None,
                "path": None if llama is None else str(llama),
                "converter": llama is not None and (llama / "convert_hf_to_gguf.py").is_file(),
            },
            "ollama": shutil.which("ollama") is not None,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "native_bf16": cpu_has_native_bf16(),
            "cpu_threads": torch.get_num_threads(),
        }

    @app.get("/api/models")
    def get_models():
        info = system_info()
        ram, native = available_ram_bytes(), cpu_has_native_bf16()
        out = []
        for e in scan_all(roots):
            mem = _memory_bytes(e.fmt, e.size_bytes, ram, native)
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
                    "unsupported_reason": _unsupported_reason(arch),
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
                    "positive_en": c.positive_en,
                    "negative_en": c.negative_en,
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
                        raise Cancelled()
                    self._seen += n or 0
                    if self.total:
                        report(
                            self._seen / self.total,
                            f"{self._seen / 1e6:.0f} / {self.total / 1e6:.0f} MB indirildi",
                        )
                    return True

            report(0.0, "İndirme başlıyor")
            return {"path": str(snapshot_download(
                    SAMPLE_REPO, revision=SAMPLE_REVISION, allow_patterns=SAMPLE_PATTERNS, tqdm_class=Bar
                ))}

        return start(fn, "download")

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
                "source_sha256": lm.source_sha256,
            }

        return start(fn, "load")

    @app.get("/api/session")
    def get_session():
        """Kilit almaz: yükleme/av sürerken de anlık durumu verir. `source_path` yerel arayüz için açıktır;
        belirteç hiçbir yanıta girmez."""
        lm = session.lm
        if lm is None:
            return {"loaded": False}
        param = next(lm.model.parameters())
        return {
            "loaded": True,
            "name": lm.name,
            "arch": asdict(lm.arch),
            "device": param.device.type,
            "dtype": str(param.dtype).removeprefix("torch."),
            "license": lm.license,
            "memory_bytes": sum(p.numel() * p.element_size() for p in lm.model.parameters()),
            "device_note": lm.device_note,
            "source_path": str(lm.source_path),
            "source_sha256": lm.source_sha256,
            "default_method": default_method(lm),
            "safety_dirs": len(session.safety_dirs),
            "safety_heads": len(session.safety_heads),
            "measure_count": session.measure_count,
        }

    # -- sunucu tarafı taslak / tercih (model SHA-256'sına göre; sepetin aynası, doğruluk kaynağı değil) --

    @app.get("/api/drafts")
    def list_drafts() -> list[DraftSummary]:
        return store.list_drafts()

    @app.get("/api/draft/{sha}")
    def get_draft(sha: Annotated[str, PathParam(pattern=SHA_RE)]) -> Draft:
        d = store.get_draft(sha)
        if d is None:
            raise HTTPException(404, "Bu model için taslak yok.")
        return d

    @app.put("/api/draft/{sha}")
    async def put_draft(sha: Annotated[str, PathParam(pattern=SHA_RE)], request: Request) -> Draft:
        # Ayrıştırmadan ÖNCE boyut sınırı: önce content-length, sonra gerçek gövde.
        too_large = ApiError(413, "Taslak çok büyük (en fazla 8 MB).", "too_large")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > DRAFT_MAX_BYTES:
            raise too_large
        raw = await request.body()
        if len(raw) > DRAFT_MAX_BYTES:
            raise too_large
        with _bad_input():  # ValidationError ve SHA uyuşmazlığı birer ValueError: 422 invalid_input
            return await run_in_threadpool(lambda: store.put_draft(sha, DraftIn.model_validate_json(raw)))

    @app.delete("/api/draft/{sha}", status_code=204)
    def delete_draft(sha: Annotated[str, PathParam(pattern=SHA_RE)]) -> Response:
        store.delete_draft(sha)
        return Response(status_code=204)

    @app.get("/api/prefs")
    def get_prefs() -> Prefs:
        return store.get_prefs()

    @app.put("/api/prefs")
    def put_prefs(body: Prefs) -> Prefs:
        return store.put_prefs(body)

    @app.post("/api/shutdown", status_code=202)
    def shutdown() -> dict:
        request_exit = app.state.request_exit
        if request_exit is None:
            raise ApiError(501, "Bu çalıştırma biçiminde sunucu arayüzden kapatılamaz.", "unsupported")
        request_exit()
        return {"ok": True}

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
            toks = tokens(lm, body.prompt)
            if len(toks) > MAX_ACTIVITY_TOKENS:
                raise ValueError(f"İstem çok uzun: en fazla {MAX_ACTIVITY_TOKENS} token (şu an {len(toks)}).")
            power = head_output_power(lm, body.prompt)  # [L, H, T]; tek ileri geçiş, tüm token'lar
            out = {"tokens": toks, "power_raw": power.tolist()}
            if body.token_index is not None:
                n_tokens = power.shape[-1]
                if not -n_tokens <= body.token_index < n_tokens:
                    raise ValueError(f"token_index {body.token_index} aralık dışı (prompt {n_tokens} token).")
                at = power[:, :, body.token_index]
                out["power"] = (normalize_per_layer(at) if body.scale == "layer" else normalize_global(at)).tolist()
            return out

    @app.post("/api/lens")
    def lens(body: LensReq):
        with _model_busy_guard(), _bad_input():
            return logit_lens(require_model(), body.prompt, body.position)

    @app.post("/api/attention")
    def attention(body: AttentionReq):
        with _model_busy_guard(), _bad_input():
            return attention_patterns(require_model(), body.prompt, body.layer)

    @app.post("/api/graph")
    def graph(body: GraphReq):
        lm0 = require_model()
        from neshter.graph import GraphUnsupported, _check_family

        try:
            _check_family(lm0)
        except GraphUnsupported as e:
            raise HTTPException(422, str(e)) from e

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                if cancel.is_set():
                    return None
                b = body.budget
                g = answer_graph(
                    lm,
                    body.prompt,
                    body.target_token_id,
                    heads_per_layer=b.heads_per_layer,
                    neurons_per_layer=b.neurons_per_layer,
                    max_nodes=b.max_nodes,
                    max_edges=b.max_edges,
                    progress=report,
                    cancel=cancel,
                )
                g.json["model_sha256"] = lm.source_sha256
                graphs.put(g)
                return g.json

        return start(fn, "graph")

    @app.post("/api/graph/{graph_id}/cut")
    def graph_cut(graph_id: str, body: GraphCutReq):
        try:
            g = graphs.get(graph_id)
        except KeyError as e:
            raise HTTPException(404, "Grafik eskidi; yeniden çizin.") from e
        with _model_busy_guard(), _bad_input():
            lm = require_model()
            if g.json.get("model_sha256") != lm.source_sha256:
                raise HTTPException(404, "Grafik başka bir model için çizildi; yeniden çizin.")
            try:
                r = cut_report(lm, g, body.edge_ids)
            except KeyError as e:
                raise HTTPException(422, str(e).strip("'")) from e
        return r

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
        detector = DETECTORS.get(body.concept_id) if body.concept_id is not None else None
        target_hit = None if detector is None else {"original": detector(original), "modified": detector(modified)}
        return {"original": original, "modified": modified, "mode": mode, "target_hit": target_hit}

    @app.post("/api/recipe/check")
    def recipe_check(body: RecipeCheckReq):
        lm = require_model()
        return asdict(check_against(body.recipe, lm.arch, lm.source_sha256))

    # -- işler --

    @app.post("/api/effect")
    def effect(body: EffectReq):
        lm0 = require_model()
        arch = lm0.arch
        with _bad_input():
            _check_cpu_cap(lm0, body.n_max)
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
                _check_cpu_cap(lm, body.n_max)  # iş kuyruktayken model değişmiş olabilir
                report(0.1, "Gerçek etki hesaplanıyor")
                eff = sequential_real_effect(
                    lm, heads, prompts, n_max=body.n_max, seed=body.seed,
                    progress=sub_progress(report, 0.1, 1.0), cancel=cancel,
                )
                if cancel.is_set():  # iptal edilen ölçüm sayılmaz
                    return None
                probe = {"probe": probe_answer(lm, heads, body.probe)} if body.probe is not None else {}
                session.measure_count += 1
                return asdict(eff) | {"session_measure_index": session.measure_count} | probe

        return start(fn, "effect")

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
                concept = load_preset(body.concept_id, body.lang)
            except KeyError as e:
                raise HTTPException(422, f"Bilinmeyen hazır kavram: {body.concept_id!r}.") from e

        def fn(report, cancel):
            with session.lock:
                lm = require_model()
                if cancel.is_set():
                    return None
                method = body.method or default_method(lm)
                report(0.0, "Kavram aranıyor")
                extra = {"sweeps": body.sweeps} if body.sweeps > 1 else {}
                r = hunt(lm, concept, method, seed=body.seed, progress=report, cancel=cancel, **extra)
                # Etiketler yalnızca av tamamen bittiyse ve iş iptal edilmediyse oturuma girer (geç iptal
                # edilen av iz bırakmaz). Bu yön/head'ler kaydedilirse etiket istemciye bağlı kalmaz.
                if _is_safety_concept(concept) and not cancel.is_set():
                    if r.direction is not None:
                        d = r.direction.detach().to("cpu", torch.float32)
                        session.safety_dirs.append(d / d.norm())
                    for extra in r.extra_directions:  # süpürmenin ek yönleri de güvenlik etiketini tetikler
                        e = extra.detach().to("cpu", torch.float32)
                        session.safety_dirs.append(e / e.norm())
                    session.safety_heads.update((layer, head) for layer, head, _ in r.heads[:MEASURE_TOP_K])
            measured = {(m.layer, m.head): m for m in r.measured}

            def _measured(layer: int, head: int):
                # signal_drop null: head ref_layer'da ya da sonrasında, sinyali yapısal olarak etkileyemez
                # (ölçülmedi; "etkisiz" ölçümü değil). neutral_kl her ölçülen head için doludur.
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
                "extra_directions": [_b64_f32(d) for d in r.extra_directions],
                "sweep": r.sweep,
                "ref_layer": r.ref_layer,
                "val_accuracy": r.val_accuracy,
                "val_ci": list(r.val_ci),
                "val_n": r.val_n,
            }

        return start(fn, "hunt")

    @app.post("/api/suite")
    def suite(body: SuiteReq):
        lm = require_model()
        items = load_suite()
        if body.target is not None and body.target not in {i.category for i in items}:
            raise HTTPException(422, f"Hedef kategori '{body.target}' test setinde yok.")
        with _bad_input():
            _recipe_from_ops(lm, body.recipe_ops)
        if body.quick:
            items = [i for i in items if i.category in {body.target, QUICK_GUARD}]
        max_new_tokens = min(body.max_new_tokens, QUICK_MAX_NEW_TOKENS) if body.quick else body.max_new_tokens
        extra = {"n_random": QUICK_N_RANDOM} if body.quick else {}

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
                    max_new_tokens=max_new_tokens,
                    target=body.target,
                )
                if cancel.is_set():
                    return None
                report(0.95, "KL ve perplexity ölçülüyor")
                metrics = _recipe_metrics(
                    lm, interventions, progress=sub_progress(report, 0.95, 1.0), cancel=cancel, **extra
                )
            if cancel.is_set():
                return None
            return {
                "categories": [asdict(r) for r in results],
                "summary": None if body.target is None else asdict(summarize(results, body.target)),
                "metrics": metrics,
                "quick": body.quick,
            }

        return start(fn, "suite")

    @app.post("/api/export/estimate")
    def export_estimate(body: EstimateReq):
        """Disk tahmini + önerilen ad. Model kullanmaz (yalnızca parametre boyutları): kilit gerekmez."""
        lm = require_model()
        if body.recipe_ops is not None:
            with _bad_input():
                _recipe_from_ops(lm, body.recipe_ops, body.concept)
            has_directions = any(isinstance(op, DirectionOp) for op in body.recipe_ops)
            safety = _is_safety(session, body.recipe_ops, body.concept)
        else:  # tarif yok: ad geçicidir (head/yön eşleşmesi bilinmez); yalnızca concept == "refusal"
            has_directions = body.has_directions
            safety = _is_safety(session, [], body.concept)
        needed = int(estimate_export_bytes(lm, body.format, has_directions) * 1.1)
        free = _free_bytes(body.out_dir or str(neshter_home()))
        return {
            "bytes_needed": needed,
            "free_bytes": free,
            "enough": free >= needed,
            "suggested_name": suggest_export_name(lm, safety),
            "safety_ablated": safety,
        }

    @app.post("/api/export")
    def export(body: ExportReq):
        lm = require_model()
        with _bad_input():
            _recipe_from_ops(lm, body.recipe_ops, body.concept)
        _free_bytes(body.out_dir)  # sürücü yoksa iş başlamadan 422 (iş hatası yerine)
        _check_export_tools(body.format, body.ollama)  # eksik araç uzun kayıttan SONRA değil, önce anlaşılır

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
                # Uyuşmazlık: out_dir silinir, SourceModified (source_modified); iptal edilebilir, ilerleme bildirir.
                ensure_source_untouched(lm, out_dir, sub_progress(report, 0.9, 0.95), cancel)
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

        return start(fn, "export")

    # -- iş durumu --

    def job_status(job_id: str):
        try:
            return jobs.status(job_id)
        except KeyError as e:
            raise HTTPException(404, "İş bulunamadı.") from e

    @app.get("/api/jobs")
    def list_jobs():
        return [{k: v for k, v in asdict(st).items() if k != "result"} for st in jobs.list()]

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
        app.mount("/", _WebFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return _PLACEHOLDER_HTML

    return app
