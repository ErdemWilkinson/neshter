"""Müdahaleleri ağırlıklara gömme, atomik kayıt, model kartı, GGUF/Ollama dışa aktarımı.

Gömülen model, kancalı modelle (neshter.interventions.apply) birebir aynı davranmalıdır:
  - HeadAblation: o_proj.weight[:, h·d:(h+1)·d] = 0 (o_proj girdisinin head dilimini sıfırlamaya denk).
  - DirectionAblation (r birim): embed.weight ← W − (W r) rᵀ; her katmanda rezidüele yazan
    o_proj ve down_proj için W ← W − r (rᵀ W) (varsa bias da: b ← b − (b·r) r).
    Rezidüel girdisi zaten r'ye dik olduğundan katman çıktısı da diktir → kanca ile aynı.
Kaynak modelin dosyalarına asla yazılmaz.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch import nn

import neshter
from neshter.arch import decoder_layers, down_proj, embed, o_proj
from neshter.compare import Summary
from neshter.interventions import DirectionAblation, HeadAblation, Intervention, _validate
from neshter.loader import LoadedModel, sha256_path
from neshter.recipe import DirectionOp, HeadOp, Recipe, save_recipe

_QUANTS = ("Q4_K_M", "Q8_0")
_GB = 1024**3


def _quantize_name() -> str:
    return "llama-quantize.exe" if os.name == "nt" else "llama-quantize"


# --------------------------------------------------------------------------- bake


def _orth_rows(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """embed: W[vocab, d] ← W − (W r) rᵀ."""
    wf = w.to(torch.float32)
    return (wf - torch.outer(wf @ r, r)).to(w.dtype)


def _orth_out(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rezidüele yazan katman: W[d, in] ← W − r (rᵀ W); bias[d] ← b − (b·r) r."""
    wf = w.to(torch.float32)
    if wf.ndim == 1:
        return (wf - (wf @ r) * r).to(w.dtype)
    return (wf - torch.outer(r, r @ wf)).to(w.dtype)


def _tied_lm_head(model) -> nn.Module | None:
    head = model.get_output_embeddings()
    if head is not None and head.weight is embed(model).weight:
        return head
    return None


def bake(lm: LoadedModel, interventions: Sequence[Intervention]) -> Callable[[], None]:
    """Müdahaleleri ağırlıklara yerinde işler; orijinal tensörleri birebir geri yükleyen undo döner."""
    _validate(lm, interventions, None, "zero", None)
    model, arch = lm.model, lm.arch
    directions = [iv.direction for iv in interventions if isinstance(iv, DirectionAblation)]
    if directions and arch.family == "gemma2":
        raise NotImplementedError(
            "gemma2 için yön silme ağırlıklara gömülemiyor: o_proj/down_proj ile rezidüel toplama "
            "arasındaki ek normlar (post-attention/post-feedforward) yüzünden ağırlık dikleştirmesi "
            "kanca ile aynı sonucu vermez. Head susturma desteklenir."
        )

    originals: dict[int, tuple[nn.Parameter, torch.Tensor]] = {}

    def touch(p: nn.Parameter) -> None:
        if id(p) not in originals:
            originals[id(p)] = (p, p.detach().clone())

    # Bağlı (tied) gömme: lm_head embed ile aynı tensörü paylaşıyorsa, kanca lm_head'e dokunmadığı
    # için yön silmeden önce lm_head'e ayrı bir kopya verilir; undo yeniden bağlar.
    tied_head = _tied_lm_head(model) if directions else None
    config_tie = getattr(model.config, "tie_word_embeddings", None)

    def undo() -> None:
        with torch.no_grad():
            for p, orig in originals.values():
                p.copy_(orig)
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

            for direction in directions:  # sıralı, liste düzeninde (kanca ile aynı)
                e = embed(model).weight
                r = direction.to(device=e.device, dtype=torch.float32)
                touch(e)
                e.copy_(_orth_rows(e, r))
                for layer in range(len(decoder_layers(model))):
                    for mod in (o_proj(model, layer), down_proj(model, layer)):
                        for p in (mod.weight, mod.bias):
                            if p is not None:
                                touch(p)
                                p.copy_(_orth_out(p, r))
    except BaseException:
        undo()
        raise
    return undo


# --------------------------------------------------------------------------- model card


def _describe(op: HeadOp | DirectionOp) -> str:
    if isinstance(op, HeadOp):
        return f"head: layer {op.layer}, head {op.head}"
    return f"direction ({op.d_model})"


def model_card(
    lm: LoadedModel, recipe: Recipe, summary: Summary | None, safety_ablated: bool
) -> str:
    tags = ["neshter"] + (["safety-ablated"] if safety_ablated else [])
    front = yaml.safe_dump(
        {"license": lm.license or "other", "base_model": lm.name, "tags": tags},
        allow_unicode=True,
        sort_keys=False,
    )
    lines = [
        "---",
        front.rstrip("\n"),
        "---",
        "",
        f"# {lm.name} (neshter)",
        "",
        "Bu model, Neşter ile seçili dikkat head'leri susturularak ve/veya rezidüel akımdan "
        "yönler silinerek kaynak modelden türetilmiştir. Değişiklikler ağırlıklara gömülmüştür.",
        "",
        "This model was derived from the base model with Neşter (neshter) by silencing selected "
        "attention heads and/or removing directions from the residual stream. The changes are baked "
        "into the weights.",
        "",
    ]
    if safety_ablated:
        lines += [
            "**Uyarı / Warning:** Reddetme (refusal) davranışı zayıflatılmıştır; güvenlik "
            "korumaları azalmış olabilir. Refusal behaviour has been ablated (safety-ablated); "
            "safety guardrails may be reduced.",
            "",
        ]
    if lm.arch.family == "llama":
        lines += [
            "**Lisans / License:** Llama lisansı, türetilmiş model adlarının \"Llama\" ile "
            "başlamasını gerektirir. The Llama license requires derivative model names to start "
            "with \"Llama\".",
            "",
        ]
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
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    if directions and _tied_lm_head(model) is not None:
        w = embed(model).weight
        total += w.numel() * w.element_size()  # bağ çözülünce lm_head ayrıca yazılır
    return total


def save_safetensors(
    lm: LoadedModel, recipe: Recipe, out_dir: Path, summary: Summary | None = None
) -> Path:
    """Müdahaleleri gömüp safetensors klasörü olarak atomik kaydeder; modeli eski haline döndürür."""
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
    estimate = _estimate_bytes(lm, has_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir.parent).free
    if free < estimate * 1.1:
        raise OSError(
            f"Diskte yeterli yer yok: gereken yaklaşık {estimate * 1.1 / _GB:.2f} GB, "
            f"boş {free / _GB:.2f} GB."
        )

    undo = bake(lm, interventions)
    try:
        try:
            lm.model.save_pretrained(tmp, safe_serialization=True)
            lm.tokenizer.save_pretrained(tmp)
            cfg_path = tmp / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            meta: dict[str, str] = {"version": neshter.__version__}
            if recipe.concept == "refusal":
                meta["abliterated"] = "refusal"
            cfg["neshter"] = meta
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            safety = recipe.concept == "refusal"
            (tmp / "README.md").write_text(
                model_card(lm, recipe, summary, safety_ablated=safety), encoding="utf-8"
            )
            save_recipe(recipe, tmp / "recipe.nester.json")
            os.replace(tmp, out_dir)
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
    hf_dir: Path, out_file: Path, quant: Literal["Q4_K_M", "Q8_0"], llama_cpp: Path
) -> Path:
    if quant not in _QUANTS:
        raise ValueError(f"Geçersiz niceleme '{quant}'; {', '.join(_QUANTS)} olmalı.")
    llama_cpp, out_file = Path(llama_cpp), Path(out_file)
    converter = llama_cpp / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(
            f"'{llama_cpp}' klasöründe convert_hf_to_gguf.py bulunamadı. GGUF'a dönüştürmek için "
            "llama.cpp kaynak deposundaki convert_hf_to_gguf.py (ve gguf-py) bu klasörde olmalı."
        )
    if out_file.exists():
        raise FileExistsError(f"'{out_file}' zaten var; üzerine yazılmaz.")
    quantize = llama_cpp / _quantize_name()
    f16 = out_file.with_name(out_file.name + ".f16.tmp")
    qtmp = out_file.with_name(out_file.name + ".tmp")
    try:
        subprocess.run(
            [sys.executable, str(converter), str(hf_dir), "--outtype", "f16", "--outfile", str(f16)],
            check=True,
        )
        try:
            subprocess.run([str(quantize), str(f16), str(qtmp), quant], check=True)
            os.replace(qtmp, out_file)
        except BaseException:
            qtmp.unlink(missing_ok=True)
            raise
    finally:
        f16.unlink(missing_ok=True)
    return out_file


def ollama_create(name: str, gguf: Path) -> None:
    if shutil.which("ollama") is None:
        raise FileNotFoundError(
            "Ollama bulunamadı (PATH'te 'ollama' yok). Ollama'yı kurun ya da bu adımı atlayın."
        )
    gguf = Path(gguf)
    modelfile = gguf.parent / "Modelfile"
    modelfile.write_text(f"FROM {gguf.resolve()}\n", encoding="utf-8")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
