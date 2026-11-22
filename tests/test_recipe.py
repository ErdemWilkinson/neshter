import base64
import json

import pytest
import torch
from pydantic import ValidationError

from conftest import recipe_for
from neshter.interventions import DirectionAblation, HeadAblation
from neshter.recipe import (
    HeadOp,
    Recipe,
    check_against,
    from_interventions,
    load_recipe,
    save_recipe,
    to_interventions,
)

SHA = "a" * 64


def _raw(**over):
    d = {
        "schema_version": 1,
        "neshter_version": "0.1.0",
        "source_model": {"name": "m", "sha256": SHA},
        "concept": "code",
        "method": "manual",
        "seed": 0,
        "dataset_sha256": None,
        "head_mode": "zero",
        "interventions": [{"type": "head", "layer": 0, "head": 1}],
        "created_at": "2026-11-10T00:00:00Z",
    }
    d.update(over)
    return d


def _b64(n):
    return base64.b64encode(torch.ones(n).numpy().astype("<f4").tobytes()).decode()


def test_roundtrip(tmp_path, lm):
    r = from_interventions(
        [HeadAblation(1, 2), DirectionAblation(torch.ones(64))],
        lm, concept="code", method="attribution", seed=0,
    )
    save_recipe(r, tmp_path / "t.nester.json")
    back = to_interventions(load_recipe(tmp_path / "t.nester.json"))
    assert back[0] == HeadAblation(1, 2) and torch.allclose(back[1].direction, torch.ones(64) / 8)
    assert r.source_model.name == lm.name and r.source_model.sha256 == lm.source_sha256
    assert r.created_at.tzinfo is not None and r.created_at.utcoffset().total_seconds() == 0
    assert not (tmp_path / "t.nester.json.tmp").exists()


def test_recipe_for_helper(lm):
    r = recipe_for(lm, [HeadAblation(0, 0)])
    assert r.concept == "code" and r.method == "manual" and r.seed == 0


def test_unknown_field_rejected(tmp_path):
    p = tmp_path / "x.nester.json"
    p.write_text(json.dumps(_raw(exec="rm -rf")), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_recipe(p)


def test_unknown_field_in_op_rejected():
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(interventions=[{"type": "head", "layer": 0, "head": 0, "x": 1}]))


def test_bad_sha_and_negative_index_rejected():
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(source_model={"name": "m", "sha256": "xyz"}))
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(interventions=[{"type": "head", "layer": -1, "head": 0}]))


def test_direction_length_mismatch_rejected():
    ok = {"type": "direction", "d_model": 8, "vector_b64": _b64(8)}
    Recipe.model_validate(_raw(interventions=[ok]))
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(interventions=[{**ok, "d_model": 9}]))
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(interventions=[{**ok, "vector_b64": "!!not base64!!"}]))


def test_other_model_rejected(lm):
    r = recipe_for(lm, [HeadAblation(0, 0)])
    r = r.model_copy(update={"interventions": [HeadOp(type="head", layer=40, head=0)]})
    c = check_against(r, lm.arch, lm.source_sha256)
    assert not c.ok and "40" in c.errors[0]


def test_direction_d_model_mismatch(lm):
    r = Recipe.model_validate(
        _raw(
            source_model={"name": "m", "sha256": lm.source_sha256},
            interventions=[{"type": "direction", "d_model": 8, "vector_b64": _b64(8)}],
        )
    )
    c = check_against(r, lm.arch, lm.source_sha256)
    assert not c.ok and "8" in c.errors[0] and "64" in c.errors[0]


def test_sha_mismatch_is_warning(lm):
    r = Recipe.model_validate(_raw(source_model={"name": "m", "sha256": "0" * 64}))
    c = check_against(r, lm.arch, lm.source_sha256)
    assert c.ok and c.sha_mismatch and c.errors == []


def test_matching_sha(lm):
    r = recipe_for(lm, [HeadAblation(1, 3)])
    c = check_against(r, lm.arch, lm.source_sha256)
    assert c.ok and not c.sha_mismatch


def test_oversize_file(tmp_path):
    (tmp_path / "b.json").write_bytes(b" " * (5 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        load_recipe(tmp_path / "b.json")


@pytest.mark.parametrize("value", [1e30, 1e-42], ids=["norm-overflow", "norm-underflow"])
def test_direction_norm_overflow_underflow_rejected(value):
    raw = base64.b64encode(torch.full((8,), value).numpy().astype("<f4").tobytes()).decode()
    op = {"type": "direction", "d_model": 8, "vector_b64": raw}
    with pytest.raises(ValidationError):
        Recipe.model_validate_json(json.dumps(_raw(interventions=[op])))


# --- Fix wave: Minor 9 (tarif sağlamlaştırma) ------------------------------------------------------


def test_nested_unknown_field_and_unknown_type_rejected():
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(source_model={"name": "m", "sha256": SHA, "path": "C:/x"}))
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(interventions=[{"type": "neuron", "layer": 0, "head": 1}]))


@pytest.mark.parametrize("field, limit", [("concept", 64), ("method", 64), ("neshter_version", 256)])
def test_long_display_strings_rejected(field, limit):
    Recipe.model_validate(_raw(**{field: "x" * limit}))
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(**{field: "x" * (limit + 1)}))
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(source_model={"name": "x" * 257, "sha256": SHA}))


@pytest.mark.parametrize("bad", ["a b", "x" * 65, "<b>", "refusal\n", ""])
@pytest.mark.parametrize("field", ["concept", "method"])
def test_recipe_concept_and_method_pattern(field, bad):
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(**{field: bad}))


@pytest.mark.parametrize("good", ["refusal", "My_Concept-2.1", "x" * 64])
def test_recipe_concept_and_method_accept_safe_text(good):
    r = Recipe.model_validate(_raw(concept=good, method=good))
    assert r.concept == good and r.method == good
    assert Recipe.model_validate(_raw(concept=None)).concept is None


def test_naive_created_at_rejected():
    with pytest.raises(ValidationError):
        Recipe.model_validate(_raw(created_at="2026-11-10T00:00:00"))


def test_exact_size_boundary(tmp_path):
    body = json.dumps(_raw()).encode()
    limit = 5 * 1024 * 1024
    ok = body + b" " * (limit - len(body))
    (tmp_path / "ok.json").write_bytes(ok)
    assert load_recipe(tmp_path / "ok.json").concept == "code"
    (tmp_path / "big.json").write_bytes(ok + b" ")
    with pytest.raises(ValueError, match="çok büyük"):
        load_recipe(tmp_path / "big.json")


def test_save_recipe_removes_tmp_on_failure(tmp_path, monkeypatch):
    import neshter.recipe as rmod

    r = Recipe.model_validate(_raw())
    monkeypatch.setattr(rmod.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("kilitli")))
    with pytest.raises(OSError):
        save_recipe(r, tmp_path / "r.nester.json")
    assert list(tmp_path.iterdir()) == []
