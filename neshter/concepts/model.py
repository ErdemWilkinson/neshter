"""Kavram modeli, hazır kavramların yüklenmesi ve train/val bölmesi."""

from __future__ import annotations

import random
from importlib.resources import files

import yaml
from pydantic import BaseModel, field_validator

PRESET_ORDER = ["code", "refusal", "math", "tone"]
MIN_EXAMPLES = 32


class Concept(BaseModel, extra="forbid"):
    id: str
    name_tr: str
    name_en: str
    positive: list[str]
    negative: list[str]
    suite_target: str | None = None  # test setindeki hedef kategori (ör. refusal → harmful_refusal)
    safety_ablation: bool = False

    @field_validator("positive", "negative")
    @classmethod
    def _enough_examples(cls, v: list[str]) -> list[str]:
        if len(v) < MIN_EXAMPLES:
            raise ValueError(f"En az {MIN_EXAMPLES} örnek gerekli, {len(v)} verildi.")
        return v


def load_preset(concept_id: str) -> Concept:
    """Paketle gelen hazır kavramı (neshter/concepts/presets/<id>.yaml) yükler ve doğrular."""
    if concept_id not in PRESET_ORDER:
        raise KeyError(f"Bilinmeyen hazır kavram: {concept_id!r}")
    text = files("neshter.concepts").joinpath("presets", f"{concept_id}.yaml").read_text(encoding="utf-8")
    return Concept.model_validate(yaml.safe_load(text))


def _split_list(items: list[str], val_frac: float, rng: random.Random) -> tuple[list[str], list[str]]:
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = round(len(items) * val_frac)
    val_idx = set(idx[:n_val])
    train = [x for i, x in enumerate(items) if i not in val_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    return train, val


def split(c: Concept, val_frac: float = 0.2, seed: int = 0) -> tuple[Concept, Concept]:
    """(train, val). Her listeden round(len·val_frac) öğe val'e; sıra korunur, tohum sabittir.

    Alt kümeler doğrulanmaz (model_copy): val küçüktür, train de MIN_EXAMPLES altına düşebilir.
    """
    rng = random.Random(seed)
    pos_tr, pos_va = _split_list(c.positive, val_frac, rng)
    neg_tr, neg_va = _split_list(c.negative, val_frac, rng)
    train = c.model_copy(update={"positive": pos_tr, "negative": neg_tr})
    val = c.model_copy(update={"positive": pos_va, "negative": neg_va})
    return train, val
