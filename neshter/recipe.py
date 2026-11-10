"""Paylaşılabilir kesim tarifi (.nester.json).

Tarifler başkalarından gelir, yani güvenilmeyen girdidir: şema katıdır (bilinmeyen alan yasak),
içinde kod ya da dosya yolu taşınmaz, dosya boyutu sınırlıdır.
"""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator

import neshter
from neshter.arch import ArchInfo
from neshter.interventions import DirectionAblation, HeadAblation, Intervention
from neshter.loader import LoadedModel

MAX_RECIPE_BYTES = 5 * 1024 * 1024


class HeadOp(BaseModel, extra="forbid"):
    type: Literal["head"]
    layer: NonNegativeInt
    head: NonNegativeInt


class DirectionOp(BaseModel, extra="forbid"):
    type: Literal["direction"]
    d_model: PositiveInt
    vector_b64: str  # float32, little-endian, base64

    @model_validator(mode="after")
    def _check_vector(self) -> DirectionOp:
        try:
            raw = base64.b64decode(self.vector_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"vector_b64 geçerli base64 değil: {e}") from None
        if len(raw) != 4 * self.d_model:
            raise ValueError(
                f"vector_b64 {len(raw)} bayt çözüldü; d_model={self.d_model} için "
                f"{4 * self.d_model} bayt (float32) gerekir."
            )
        v = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
        # DirectionAblation ile aynı ölçüt: float32 normu sonlu ve > 0 olmalı.
        norm = v.norm()
        if not torch.isfinite(v).all() or not torch.isfinite(norm) or norm == 0:
            raise ValueError("vector_b64 sonlu, sıfırdan farklı ve normalize edilebilir bir vektör olmalı.")
        return self


class SourceRef(BaseModel, extra="forbid"):
    name: str
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


class Recipe(BaseModel, extra="forbid"):
    schema_version: Literal[1]
    neshter_version: str
    source_model: SourceRef
    concept: str | None
    method: str
    seed: int
    dataset_sha256: str | None
    head_mode: Literal["zero"]
    interventions: list[Annotated[HeadOp | DirectionOp, Field(discriminator="type")]]
    created_at: datetime


@dataclass
class RecipeCheck:
    ok: bool
    errors: list[str]
    sha_mismatch: bool


def check_against(r: Recipe, arch: ArchInfo, sha256: str) -> RecipeCheck:
    errors: list[str] = []
    for op in r.interventions:
        if isinstance(op, HeadOp):
            if op.layer >= arch.n_layers:
                errors.append(
                    f"Tarifteki katman {op.layer} bu modelde yok (0..{arch.n_layers - 1})."
                )
            if op.head >= arch.n_heads:
                errors.append(f"Tarifteki head {op.head} bu modelde yok (0..{arch.n_heads - 1}).")
        elif op.d_model != arch.d_model:
            errors.append(
                f"Tarifteki yön vektörü {op.d_model} boyutlu; bu modelin boyutu {arch.d_model}."
            )
    return RecipeCheck(ok=not errors, errors=errors, sha_mismatch=r.source_model.sha256 != sha256)


def to_interventions(r: Recipe) -> list[Intervention]:
    out: list[Intervention] = []
    for op in r.interventions:
        if isinstance(op, HeadOp):
            out.append(HeadAblation(op.layer, op.head))
        else:
            v = np.frombuffer(base64.b64decode(op.vector_b64), dtype="<f4").copy()
            out.append(DirectionAblation(torch.from_numpy(v)))
    return out


def from_interventions(
    interventions: Sequence[Intervention],
    lm: LoadedModel,
    concept: str | None,
    method: str,
    seed: int,
    dataset_sha256: str | None = None,
) -> Recipe:
    ops: list[HeadOp | DirectionOp] = []
    for iv in interventions:
        if isinstance(iv, HeadAblation):
            ops.append(HeadOp(type="head", layer=iv.layer, head=iv.head))
        elif isinstance(iv, DirectionAblation):
            v = iv.direction.detach().cpu().to(torch.float32).numpy()
            ops.append(
                DirectionOp(
                    type="direction",
                    d_model=v.shape[0],
                    vector_b64=base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode("ascii"),
                )
            )
        else:
            raise ValueError(f"Bilinmeyen müdahale türü: {type(iv).__name__}.")
    return Recipe(
        schema_version=1,
        neshter_version=neshter.__version__,
        source_model=SourceRef(name=lm.name, sha256=lm.source_sha256),
        concept=concept,
        method=method,
        seed=seed,
        dataset_sha256=dataset_sha256,
        head_mode="zero",
        interventions=ops,
        created_at=datetime.now(timezone.utc),
    )


def load_recipe(path: Path) -> Recipe:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_RECIPE_BYTES:
        raise ValueError(
            f"Tarif dosyası çok büyük ({size} bayt); sınır {MAX_RECIPE_BYTES} bayt."
        )
    return Recipe.model_validate_json(path.read_bytes())


def save_recipe(r: Recipe, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(r.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
