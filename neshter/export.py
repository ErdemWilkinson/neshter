"""Müdahaleleri ağırlıklara gömme, atomik kayıt, model kartı, GGUF/Ollama dışa aktarımı.

Gömülen model, kancalı modelle (neshter.interventions.apply) birebir aynı davranmalıdır:
  - HeadAblation: o_proj.weight[:, h·d:(h+1)·d] = 0 (o_proj girdisinin head dilimini sıfırlamaya denk).
  - DirectionAblation: tüm yönler tek izdüşümle silinir; Q = direction_basis(r_1..r_k) yönlerin
    gerdiği uzayın dik birim tabanıdır ([d, k'], SVD). embed.weight ← W − (W Q) Qᵀ; her katmanda
    rezidüele yazan o_proj ve down_proj için W ← W − Q (Qᵀ W) (varsa bias da: b ← b − Q (Qᵀ b)).
    Yazıcı/taban listesi kancayla aynıdır (interventions.writer_bases). Rezidüel girdisi zaten
    span(R)'ye dik olduğundan katman çıktısı da diktir → kanca ile aynı; yön sırası önemsiz.
    gemma2: o_proj/down_proj için taban S = direction_basis([γ ⊙ r_i]) (γ = 1 + post-norm ağırlığı);
    W ← W − S (Sᵀ W) post-norm çıktısını her r_i'ye dik yapar (rms skaler) → yine kanca ile aynı.
Kaynak modelin dosyalarına asla yazılmaz.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import date
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch import nn

import neshter
from neshter.arch import embed, o_proj
from neshter.compare import Summary
from neshter.interventions import (
    DirectionAblation,
    HeadAblation,
    Intervention,
    _validate,
    writer_bases,
)
from neshter.licenses import clean_id, inside_model
from neshter.loader import LoadedModel, sha256_path
from neshter.progress import ProgressFn, check_cancel, sub_progress
from neshter.recipe import DirectionOp, HeadOp, Recipe, save_recipe

_QUANTS = ("Q4_K_M", "Q8_0")
_GB = 1024**3
QUANT_BYTES_PER_PARAM = {"Q4_K_M": 0.6, "Q8_0": 1.07}


class DiskFull(OSError):
    """Diskte dışa aktarım için yeterli yer yok (yazmaya başlamadan önce tespit edilir)."""


class LlamaCppMissing(FileNotFoundError):
    """llama.cpp (dönüştürücü ya da llama-quantize) bulunamadı."""


def _quantize_name() -> str:
    return "llama-quantize.exe" if os.name == "nt" else "llama-quantize"


# --------------------------------------------------------------------------- bake


ROW_CHUNK = 8192  # embed satır bloğu
COL_CHUNK = 4096  # o_proj/down_proj sütun bloğu


def _orth_rows_(w: torch.Tensor, basis: torch.Tensor, chunk: int = ROW_CHUNK) -> None:
    """embed, yerinde: W[vocab, d] ← W − (W Q) Qᵀ (Q = basis [d, k']).

    float32'de, satır bloklarıyla (tam float32 kopya yok).
    """
    q = basis.to(device=w.device, dtype=torch.float32)
    for i in range(0, w.shape[0], chunk):
        blk = w[i : i + chunk]
        wf = blk.to(torch.float32)
        blk.copy_(wf - (wf @ q) @ q.T)


def _orth_out_(w: torch.Tensor, basis: torch.Tensor, chunk: int = COL_CHUNK) -> None:
    """Rezidüele yazan katman, yerinde: W[d, in] ← W − Q (Qᵀ W); bias[d] ← b − Q (Qᵀ b).

    Q = basis [d, k']. Sütunlar bağımsızdır; float32'de sütun bloklarıyla işlenir.
    """
    q = basis.to(device=w.device, dtype=torch.float32)
    if w.ndim == 1:
        wf = w.to(torch.float32)
        w.copy_(wf - q @ (q.T @ wf))
        return
    for j in range(0, w.shape[1], chunk):
        blk = w[:, j : j + chunk]
        wf = blk.to(torch.float32)
        blk.copy_(wf - q @ (q.T @ wf))


def _snapshot(p: torch.Tensor) -> torch.Tensor:
    """Undo için CPU kopyası: GPU'da ikinci bir ağırlık kopyası tutulmaz."""
    return p.detach().to("cpu", copy=True)


def _tied_lm_head(model) -> nn.Module | None:
    head = model.get_output_embeddings()
    if head is not None and head.weight is embed(model).weight:
        return head
    return None


def bake(
    lm: LoadedModel, interventions: Sequence[Intervention], round_to: torch.dtype | None = None
) -> Callable[[], None]:
    """Müdahaleleri ağırlıklara yerinde işler; orijinal tensörleri birebir geri yükleyen undo döner.

    round_to verilirse dokunulan her parametre p, işlendikten sonra p.to(round_to).to(p.dtype) olur;
    böylece bellekteki model kaydedilen (ör. bf16) dosyayla aynı değerleri taşır.
    Bağlı gömmede lm_head'e verilen ayrı kopya da yuvarlanır (kayıtta o da round_to'ya çevrilir).
    Sınırlama: yalnızca DOKUNULAN parametreler (ve bu lm_head kopyası) yuvarlanır. Bütün modelin kayıtlı dosyayla eşitliği,
    ancak dokunulmayan kaynak ağırlıklar zaten round_to ile tam gösterilebiliyorsa (bf16 kaynak) geçerlidir.
    """
    _validate(lm, interventions, None, "zero", None)
    model, arch = lm.model, lm.arch
    directions = [iv.direction for iv in interventions if isinstance(iv, DirectionAblation)]

    originals: dict[int, tuple[nn.Parameter, torch.Tensor]] = {}

    def touch(p: nn.Parameter) -> None:
        if id(p) not in originals:
            originals[id(p)] = (p, _snapshot(p))

    # Bağlı (tied) gömme: lm_head embed ile aynı tensörü paylaşıyorsa, kanca lm_head'e dokunmadığı
    # için yön silmeden önce lm_head'e ayrı bir kopya verilir; undo yeniden bağlar.
    tied_head = _tied_lm_head(model) if directions else None
    config_tie = getattr(model.config, "tie_word_embeddings", None)

    def undo() -> None:
        with torch.no_grad():
            for p, orig in originals.values():
                p.copy_(orig.to(p.device))
        if tied_head is not None:
            tied_head.weight = embed(model).weight
            model.config.tie_word_embeddings = config_tie

    try:
        with torch.no_grad():
            if tied_head is not None:
                tied_head.weight = nn.Parameter(
                    embed(model).weight.detach().clone(), requires_grad=False
                )
                model.config.tie_word_embeddings = False

            for iv in interventions:
                if isinstance(iv, HeadAblation):
                    w = o_proj(model, iv.layer).weight
                    touch(w)
                    d = arch.head_dim
                    w[:, iv.head * d : (iv.head + 1) * d] = 0

            # Kancayla aynı yazıcı/taban listesi; tüm yönler tek izdüşümle (span) silinir.
            emb = embed(model)
            for module, basis in writer_bases(model, arch.family, directions) if directions else []:
                if basis.shape[1] == 0:
                    continue
                if module is emb:
                    touch(module.weight)
                    _orth_rows_(module.weight, basis)
                    continue
                for p in (module.weight, getattr(module, "bias", None)):
                    if p is not None:
                        touch(p)
                        _orth_out_(p, basis)

            if round_to is not None:
                for p, _ in originals.values():
                    p.copy_(p.to(round_to).to(p.dtype))
                if tied_head is not None:  # ayrılan lm_head kopyası da kayıtta round_to'ya çevrilir
                    h = tied_head.weight
                    h.copy_(h.to(round_to).to(h.dtype))
    except BaseException:
        undo()
        raise
    return undo


# --------------------------------------------------------------------------- model card


def _describe(op: HeadOp | DirectionOp) -> str:
    if isinstance(op, HeadOp):
        return f"head: layer {op.layer}, head {op.head}"
    return f"direction ({op.d_model})"


GEMMA_NOTICE = "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms"
_GEMMA_POLICY = "ai.google.dev/gemma/prohibited_use_policy"
_SAFE_TEXT = re.compile(r"[^\w.:/+\- ]")


def _md_safe(text: str, limit: int) -> str:
    """Modelden gelen güvenilmez görüntü metni (lisans kimliği, model adı): denetim karakterleri atılır,
    markdown/HTML işaretleri '?' olur, uzunluk sınırlanır."""
    return _SAFE_TEXT.sub("?", clean_id(text))[:limit].strip()


def _display_license(lm: LoadedModel) -> str | None:
    """Karta yazılacak lisans kimliği."""
    return _md_safe(lm.license, 64) or None if lm.license else None


def is_llama3(lm: LoadedModel) -> bool:
    if lm.arch.family != "llama":
        return False
    return (
        (lm.license or "").lower().startswith("llama3")
        or re.search(r"llama[-_ ]?3(?!\d)", lm.name, re.IGNORECASE) is not None
        or getattr(lm.model.config, "vocab_size", None) == 128256
    )


def license_notices(lm: LoadedModel) -> list[str]:
    """Aileye göre çift dilli lisans/kullanım notları (model kartı satırları)."""
    lid = _display_license(lm)
    out: list[str] = []
    if is_llama3(lm):
        out.append(
            "**Lisans / License:** Built with Llama. Llama 3.x topluluk lisansı gereği türetilmiş "
            "modelin adı \"Llama\" ile başlamalıdır. Under the Llama 3.x Community License, the "
            "derivative model name must start with \"Llama\"."
        )
    if lm.arch.family == "gemma2":
        out.append(
            f"**Lisans / License:** {GEMMA_NOTICE}. Kullanım kısıtları / Prohibited Use Policy: {_GEMMA_POLICY}"
        )
    qwen = lm.arch.family == "qwen2"
    if qwen and lid is not None and lid.lower() != "apache-2.0":
        out.append(
            f"**Uyarı / Warning:** Bu Qwen modelinin lisansı Apache-2.0 değil ({lid}); "
            "dağıtmadan önce lisans koşullarını kontrol edin. "
            f"This Qwen model's license is not Apache-2.0 ({lid}); "
            "check the license terms before redistributing."
        )
    elif qwen and lid is None:  # tek, birleşik not (Qwen uyarısı + bulunamadı notu iki kez yazılmaz)
        out.append(
            "**Uyarı / Warning:** Kaynak modelin lisansı bulunamadı; dağıtmadan önce kontrol edin. "
            "Qwen modellerinin bazı boyutları Apache-2.0 değildir (lisans bilinmiyor). "
            "The source model's license was not found; check it before redistributing. "
            "Some Qwen sizes are not Apache-2.0 (license unknown)."
        )
    elif lid is None:
        out.append(
            "**Uyarı / Warning:** Kaynak modelin lisansı bulunamadı; dağıtmadan önce kontrol edin. "
            "The source model's license was not found; check it before redistributing."
        )
    return out


def model_card(
    lm: LoadedModel, recipe: Recipe, summary: Summary | None, safety_ablated: bool
) -> str:
    tags = ["neshter"] + (["safety-ablated"] if safety_ablated else [])
    lid = _display_license(lm)
    meta: dict = {"license": lid or "other"}
    if lid is None:
        meta["license_name"] = "unknown"
    meta |= {"base_model": lm.name, "tags": tags}
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
    n_heads = sum(isinstance(op, HeadOp) for op in recipe.interventions)
    n_dirs = sum(isinstance(op, DirectionOp) for op in recipe.interventions)
    lines = [
        "---",
        front.rstrip("\n"),
        "---",
        "",
        f"# {_md_safe(lm.name, 120) or 'model'} (neshter)",
        "",
        "Bu model, Neşter ile seçili dikkat head'leri susturularak ve/veya rezidüel akımdan "
        "yönler silinerek kaynak modelden türetilmiştir. Değişiklikler ağırlıklara gömülmüştür.",
        "",
        "This model was derived from the base model with Neşter (neshter) by silencing selected "
        "attention heads and/or removing directions from the residual stream. The changes are baked "
        "into the weights.",
        "",
        f"Modified with Neşter {neshter.__version__} on {date.today().isoformat()}: "
        f"{n_heads} heads silenced, {n_dirs} directions removed.",
        "",
    ]
    if safety_ablated:
        lines += [
            "**Uyarı / Warning:** Reddetme (refusal) davranışı zayıflatılmıştır; güvenlik "
            "korumaları azalmış olabilir. Refusal behaviour has been ablated (safety-ablated); "
            "safety guardrails may be reduced.",
            "",
        ]
    for notice in license_notices(lm):
        lines += [notice, ""]
    lines += ["## Tarif / Recipe", ""]
    lines += [f"- concept: {recipe.concept or '-'}"]
    lines += [f"- method: {recipe.method}"]
    lines += [f"- seed: {recipe.seed}"]
    lines += [f"- source sha256: `{recipe.source_model.sha256}`"]
    lines += [f"- neshter version: {neshter.__version__}"]
    lines += ["", "### Müdahaleler / Interventions", ""]
    lines += [f"- {_describe(op)}" for op in recipe.interventions] or ["- (yok / none)"]
    if summary is not None:
        lines += ["", "## Özet / Summary", "", summary.tr, "", summary.en]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- save


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _estimate_bytes(lm: LoadedModel, directions: bool) -> int:
    model = lm.model
    unit = torch.empty((), dtype=save_dtype(lm)).element_size()  # diske yazılan dtype, bellektekiyle değil
    total = sum(p.numel() * unit for p in model.parameters())
    if directions and _tied_lm_head(model) is not None:
        total += embed(model).weight.numel() * unit  # bağ çözülünce lm_head ayrıca yazılır
    return total


def estimate_export_bytes(
    lm: LoadedModel, fmt: Literal["safetensors", "gguf_q4_k_m", "gguf_q8_0"], has_directions: bool
) -> int:
    """Dışa aktarımın disk tepe kullanımı. GGUF: _hf klasörü, f16 dosyası ve nicelenmiş dosya birlikte durur."""
    st = _estimate_bytes(lm, has_directions)
    if fmt == "safetensors":
        return st
    quant = fmt.removeprefix("gguf_").upper()
    if quant not in QUANT_BYTES_PER_PARAM:
        raise ValueError(f"Geçersiz dışa aktarım biçimi {fmt!r}.")
    n = sum(p.numel() for p in lm.model.parameters())
    return int(st + 2 * n + QUANT_BYTES_PER_PARAM[quant] * n)


def _check_disk(parent: Path, estimate: int) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < estimate * 1.1:
        raise DiskFull(
            f"Diskte yeterli yer yok: gereken yaklaşık {estimate * 1.1 / _GB:.2f} GB, "
            f"boş {free / _GB:.2f} GB."
        )


_LICENSE_GLOBS = ("LICENSE*", "USE_POLICY*", "NOTICE*")


def _copy_license_files(source: Path, dest: Path) -> None:
    """Kaynak klasördeki lisans/kullanım politikası dosyalarını kopyalar ("model kartı + lisans her zaman")."""
    if not source.is_dir():
        return
    for pattern in _LICENSE_GLOBS:
        for f in sorted(source.glob(pattern)):
            # Sembolik bağ modelin klasörü (ya da HF blobs/) dışına çözülüyorsa kopyalanmaz.
            if inside_model(source, f) and not (dest / f.name).exists():
                shutil.copy2(f, dest / f.name)


def _source_config(lm: LoadedModel) -> dict:
    src = Path(lm.source_path)
    folder = src if src.is_dir() else src.parent if src.suffix.lower() == ".pt" else None
    if folder is None:
        return {}
    try:
        return json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_dtype(lm: LoadedModel) -> torch.dtype:
    """Kayıt dtype'ı: kaynak float16 ise float16, aksi halde bf16 (spec: "safetensors bf16").

    Model bellekte float32 olsa da (ör. testler) dosya bf16 yazılır; sunucu ve CLI aynı dosyayı üretir.
    """
    cfg = _source_config(lm)
    src = cfg.get("dtype") or cfg.get("torch_dtype")
    return torch.float16 if src == "float16" else torch.bfloat16


def _cast_state_dict(model, dtype: torch.dtype) -> dict[str, torch.Tensor] | None:
    """Kayan noktalı tensörleri dtype'a çevirir (paylaşılan tensörler paylaşılmış kalır).

    Zaten hepsi dtype'taysa None döner: save_pretrained modeli olduğu gibi yazar, ek kopya olmaz.
    """
    sd = model.state_dict()
    if all(not v.is_floating_point() or v.dtype == dtype for v in sd.values()):
        return None
    cache: dict[tuple, torch.Tensor] = {}
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not v.is_floating_point() or v.dtype == dtype:
            out[k] = v
            continue
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))
        if key not in cache:
            cache[key] = v.to(dtype)
        out[k] = cache[key]
    return out


def save_safetensors(
    lm: LoadedModel,
    recipe: Recipe,
    out_dir: Path,
    summary: Summary | None = None,
    safety_ablated: bool = False,
    dtype: torch.dtype | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Müdahaleleri gömüp safetensors klasörü olarak atomik kaydeder; modeli eski haline döndürür.

    Güvenlik işareti (`abliterated: refusal` + "safety-ablated") tarifin kavramı "refusal" ise YA DA
    safety_ablated=True ise konur; çağıran bu işareti kaldıramaz. `dtype` verilmezse save_dtype(lm)
    (bf16; kaynak float16 ise float16); float32 yalnızca açıkça istenirse (testler).
    """
    out_dir = Path(out_dir)
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    source = Path(lm.source_path)
    if _is_within(out_dir, source) or (source.is_file() and out_dir.resolve() == source.resolve()):
        raise ValueError("Kayıt klasörü kaynak modelin içinde olamaz; kaynak dosyalara asla yazılmaz.")
    if out_dir.exists():
        raise FileExistsError(f"'{out_dir}' zaten var; üzerine yazılmaz. Başka bir klasör adı seçin.")
    if tmp.exists():
        raise FileExistsError(
            f"Geçici klasör '{tmp}' zaten var (yarım kalmış bir kayıt olabilir); silip yeniden deneyin."
        )

    from neshter.recipe import to_interventions

    interventions = to_interventions(recipe)
    has_dir = any(isinstance(iv, DirectionAblation) for iv in interventions)
    _check_disk(out_dir.parent, _estimate_bytes(lm, has_dir))

    safety = safety_ablated or recipe.concept == "refusal"  # tek doğruluk kaynağı
    target = save_dtype(lm) if dtype is None else dtype
    check_cancel(cancel)
    if progress is not None:
        progress(0.05, "Ağırlıklara gömülüyor")
    undo = bake(lm, interventions)
    try:
        try:
            if progress is not None:
                progress(0.3, "Ağırlıklar yazılıyor")
            lm.model.save_pretrained(
                tmp, safe_serialization=True, state_dict=_cast_state_dict(lm.model, target)
            )
            check_cancel(cancel)  # Cancelled: aşağıdaki except tmp'yi siler, finally ağırlıkları geri alır
            lm.tokenizer.save_pretrained(tmp)
            cfg_path = tmp / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            dtype_name = str(target).removeprefix("torch.")
            cfg["dtype"] = dtype_name
            if "torch_dtype" in cfg:
                cfg["torch_dtype"] = dtype_name
            meta: dict[str, str] = {"version": neshter.__version__}
            if safety:
                meta["abliterated"] = "refusal"
            cfg["neshter"] = meta
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            (tmp / "README.md").write_text(
                model_card(lm, recipe, summary, safety_ablated=safety), encoding="utf-8"
            )
            save_recipe(recipe, tmp / "recipe.nester.json")
            _copy_license_files(Path(lm.source_path), tmp)
            if lm.license_text and not any(tmp.glob("LICENSE*")):
                (tmp / "LICENSE").write_text(lm.license_text, encoding="utf-8")
            if lm.arch.family == "gemma2" and not any(tmp.glob("NOTICE*")):
                (tmp / "NOTICE").write_text(GEMMA_NOTICE + "\n", encoding="utf-8")
            if progress is not None:
                progress(0.95, "Kayıt tamamlanıyor")
            os.replace(tmp, out_dir)
            if progress is not None:
                progress(1.0, "Kaydedildi")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    finally:
        undo()
    return out_dir


def verify_source_untouched(lm: LoadedModel) -> bool:
    return sha256_path(lm.source_path) == lm.source_sha256


# --------------------------------------------------------------------------- gguf / ollama


def find_llama_cpp() -> Path | None:
    env = os.environ.get("NESHTER_LLAMA_CPP")
    if env:
        folder = Path(env)
        if (folder / _quantize_name()).is_file() or (folder / "llama-quantize").is_file():
            return folder
    found = shutil.which("llama-quantize")
    return Path(found).parent if found else None


def to_gguf(
    hf_dir: Path,
    out_file: Path,
    quant: Literal["Q4_K_M", "Q8_0"],
    llama_cpp: Path,
    metadata: dict | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """safetensors klasörü → f16 GGUF → nicelenmiş GGUF. `metadata` verilirse geçici bir JSON'a yazılır ve
    dönüştürücüye `--metadata <dosya>` ile geçer (general.license, general.tags, ...); dosya her durumda silinir.
    İptal yalnızca dönüştürme ile quantize arasında denetlenir (alt süreç ortasında kesilmez)."""
    if quant not in _QUANTS:
        raise ValueError(f"Geçersiz niceleme '{quant}'; {', '.join(_QUANTS)} olmalı.")
    llama_cpp, out_file = Path(llama_cpp), Path(out_file)
    converter = llama_cpp / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise LlamaCppMissing(
            f"'{llama_cpp}' klasöründe convert_hf_to_gguf.py bulunamadı. GGUF'a dönüştürmek için "
            "llama.cpp kaynak deposundaki convert_hf_to_gguf.py (ve gguf-py) bu klasörde olmalı."
        )
    quantize = llama_cpp / _quantize_name()
    if not quantize.is_file():
        raise LlamaCppMissing(f"'{llama_cpp}' klasöründe {_quantize_name()} bulunamadı.")
    if not (Path(hf_dir) / "config.json").is_file():
        raise FileNotFoundError(f"'{hf_dir}' içinde config.json yok; dönüştürülecek safetensors klasörü eksik.")
    if out_file.exists():
        raise FileExistsError(f"'{out_file}' zaten var; üzerine yazılmaz.")
    f16 = out_file.with_name(out_file.name + ".f16.tmp")
    qtmp = out_file.with_name(out_file.name + ".tmp")
    meta_file = out_file.with_name(out_file.name + ".metadata.json.tmp")
    try:
        cmd = [sys.executable, str(converter), str(hf_dir), "--outtype", "f16", "--outfile", str(f16)]
        try:
            if metadata is not None:
                meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                cmd += ["--metadata", str(meta_file)]
            subprocess.run(cmd, check=True)
        finally:
            meta_file.unlink(missing_ok=True)
        check_cancel(cancel)
        try:
            subprocess.run([str(quantize), str(f16), str(qtmp), quant], check=True)
            os.replace(qtmp, out_file)
        except BaseException:
            qtmp.unlink(missing_ok=True)
            raise
    finally:
        f16.unlink(missing_ok=True)
    return out_file


def gguf_metadata(lm: LoadedModel, recipe: Recipe, safety: bool) -> dict:
    """Dönüştürücünün `--metadata` dosyası: lisans, etiketler, tek satırlık İngilizce açıklama."""
    n_heads = sum(isinstance(op, HeadOp) for op in recipe.interventions)
    n_dirs = sum(isinstance(op, DirectionOp) for op in recipe.interventions)
    return {
        "general.license": _display_license(lm) or "other",
        "general.tags": ["neshter"] + (["safety-ablated", "abliterated"] if safety else []),
        "general.description": (
            f"Derived from {_md_safe(lm.name, 120) or 'a base model'} with Neshter {neshter.__version__}: "
            f"{n_heads} heads silenced, {n_dirs} directions removed."
        ),
    }


_KEEP_GLOBS = ("README.md", "recipe.nester.json", *_LICENSE_GLOBS)
_GGUF_NAME = re.compile(r'[^/\\:*?"<>|\x00-\x1f]+')


def export_gguf(
    lm: LoadedModel,
    recipe: Recipe,
    out_dir: Path,
    name: str,
    quant: str,
    llama_cpp: Path,
    safety: bool,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """safetensors ara klasörü → GGUF. Son düzen: out_dir/{name}.gguf + README.md + recipe.nester.json +
    lisans dosyaları (ara `_hf` klasörü silinir). out_dir yeni olmalı; hata ya da iptalde tamamen silinir."""
    out_dir = Path(out_dir)
    if quant not in _QUANTS:
        raise ValueError(f"Geçersiz niceleme '{quant}'; {', '.join(_QUANTS)} olmalı.")
    if name in ("", ".", "..") or not _GGUF_NAME.fullmatch(name):
        raise ValueError(f"Geçersiz dosya adı {name!r}.")
    source = Path(lm.source_path)
    if _is_within(out_dir, source) or (source.is_file() and out_dir.resolve() == source.resolve()):
        raise ValueError("Kayıt klasörü kaynak modelin içinde olamaz; kaynak dosyalara asla yazılmaz.")
    if out_dir.exists():
        raise FileExistsError(f"'{out_dir}' zaten var; üzerine yazılmaz. Başka bir klasör adı seçin.")
    check_cancel(cancel)
    has_dir = any(isinstance(op, DirectionOp) for op in recipe.interventions)
    _check_disk(out_dir.parent, estimate_export_bytes(lm, f"gguf_{quant.lower()}", has_dir))
    safety = safety or recipe.concept == "refusal"

    out_dir.mkdir()  # bundan sonra out_dir bu işe aittir
    try:
        hf_dir = out_dir / "_hf"
        save_safetensors(
            lm, recipe, hf_dir, safety_ablated=safety, progress=sub_progress(progress, 0.02, 0.6), cancel=cancel
        )
        if progress is not None:
            progress(0.6, "GGUF'a dönüştürülüyor ve nicelendiriliyor")
        try:
            gguf = to_gguf(
                hf_dir, out_dir / f"{name}.gguf", quant, llama_cpp,
                metadata=gguf_metadata(lm, recipe, safety), cancel=cancel,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF dönüştürme başarısız oldu (llama.cpp {e.returncode} koduyla çıktı).") from e
        check_cancel(cancel)
        if progress is not None:
            progress(0.95, "Kart, tarif ve lisans GGUF'un yanına taşınıyor")
        for pattern in _KEEP_GLOBS:
            for f in sorted(hf_dir.glob(pattern)):
                if f.is_file() and not (out_dir / f.name).exists():
                    os.replace(f, out_dir / f.name)
        shutil.rmtree(hf_dir)
        if progress is not None:
            progress(1.0, "GGUF hazır")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return gguf


def modelfile_license(lm: LoadedModel, out_dir: Path) -> str | None:
    """Ollama Modelfile'a yazılacak lisans metni: out_dir/LICENSE, yoksa modelin lisans metni, yoksa bildirimler."""
    path = Path(out_dir) / "LICENSE"
    if path.is_file():
        with open(path, "rb") as f:
            text = f.read(1024 * 1024).decode("utf-8", errors="replace")
        if text.strip():
            return text
    if lm.license_text and lm.license_text.strip():
        return lm.license_text
    return "\n\n".join(license_notices(lm)) or None


OLLAMA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


_TR_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def suggest_export_name(lm: LoadedModel, safety: bool) -> str:
    """Varsayılan dışa aktarım adı: `<ad>-neshter[-abliterated]`; her zaman OLLAMA_NAME'e (en çok 64) uyar.

    Llama 3.x lisansı türev modelin adının "Llama" ile başlamasını ister; ad zaten öyle değilse önek eklenir.
    `base` önek dahil en çok 40 karakterdir; en uzun sonuç 40 + len("-neshter-abliterated") = 60'tır.
    """
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", lm.name.translate(_TR_ASCII)).strip("-._")
    if safety:  # ad zaten "-abliterated" ile bitiyorsa sonek üst üste binmesin
        clean = re.sub(r"(?:-abliterated)+$", "", clean, flags=re.IGNORECASE).strip("-._")
    prefix = "Llama-" if is_llama3(lm) and not clean.lower().startswith("llama") else ""
    base = prefix + (clean[: 40 - len(prefix)].strip("-._") or "model")
    return f"{base}-neshter" + ("-abliterated" if safety else "")


def write_modelfile(gguf: Path, license_text: str | None) -> Path:
    """gguf'un yanına Modelfile yazar: `FROM <mutlak yol>` ve lisans metni varsa üç tırnaklı LICENSE bloğu."""
    gguf = Path(gguf)
    content = f"FROM {gguf.resolve()}\n"
    if license_text and license_text.strip():
        safe = license_text.replace('"""', "'''")  # üç tırnak Modelfile'ı erken kapatır
        content += f'LICENSE """\n{safe}\n"""\n'
    path = gguf.parent / "Modelfile"
    path.write_text(content, encoding="utf-8")
    return path


def ollama_create(name: str, gguf: Path, license_text: str | None = None) -> None:
    if not OLLAMA_NAME.fullmatch(name):
        raise ValueError(
            f"Geçersiz Ollama model adı {name!r}: harf/rakamla başlamalı; yalnızca harf, rakam, '.', '_', '-' (en çok 64)."
        )
    exe = shutil.which("ollama")
    if exe is None:
        raise FileNotFoundError(
            "Ollama bulunamadı (PATH'te 'ollama' yok). Ollama'yı kurun ya da bu adımı atlayın."
        )
    modelfile = write_modelfile(Path(gguf), license_text)
    # Tam yol: Windows'ta CreateProcess çıplak adı önce çalışma klasöründe arar.
    subprocess.run([exe, "create", name, "-f", str(modelfile)], check=True)
