"""Sunucu tarafı taslak/tercih deposu (tmp_path; gerçek ~/Neshter'e asla yazmaz)."""

from datetime import datetime, timedelta, timezone

import pytest

from neshter.server.state import DraftIn, Prefs, StateStore, neshter_home

SHA = "a" * 64


def draft_in(sha=SHA, n=1):
    return DraftIn(
        schema_version=1, sha256=sha, model_name="m", unsaved_count=n,
        basket=[{"type": "head", "layer": 0, "head": 1}], shelf=[], saved_ops=[], prompts=["merhaba"],
    )


def test_lazy_no_dir_until_write(tmp_path):
    StateStore(tmp_path / "s").list_drafts()
    assert not (tmp_path / "s").exists()


def test_put_get_roundtrip(tmp_path):
    s = StateStore(tmp_path)
    d = s.put_draft(SHA, draft_in())
    assert s.get_draft(SHA).basket == d.basket and d.updated_at.tzinfo is not None


def test_put_rejects_sha_mismatch(tmp_path):
    with pytest.raises(ValueError):
        StateStore(tmp_path).put_draft("b" * 64, draft_in())


def test_atomic_write_leaves_no_tmp(tmp_path):
    s = StateStore(tmp_path)
    s.put_draft(SHA, draft_in())
    s.put_prefs(Prefs(lang="tr"))
    assert not list(tmp_path.rglob("*.tmp"))


def test_keeps_newest_ten(tmp_path):
    s = StateStore(tmp_path)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    shas = [f"{i:064x}" for i in range(12)]
    for i, sha in enumerate(shas):
        s.put_draft(sha, draft_in(sha), now=t0 + timedelta(minutes=i))
    assert [x.sha256 for x in s.list_drafts(now=t0 + timedelta(minutes=20))] == shas[::-1][:10]
    assert s.get_draft(shas[0]) is None and s.get_draft(shas[1]) is None


def test_prunes_older_than_60_days(tmp_path):
    s = StateStore(tmp_path)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    s.put_draft(SHA, draft_in(), now=t0)
    assert s.list_drafts(now=t0 + timedelta(days=61)) == []
    assert s.get_draft(SHA) is None


def test_keeps_draft_within_60_days(tmp_path):
    s = StateStore(tmp_path)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    s.put_draft(SHA, draft_in(), now=t0)
    assert [x.sha256 for x in s.list_drafts(now=t0 + timedelta(days=59))] == [SHA]


def test_corrupt_file_is_ignored(tmp_path):
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / f"{SHA}.json").write_text("{bozuk")
    assert StateStore(tmp_path).get_draft(SHA) is None and StateStore(tmp_path).list_drafts() == []


def test_schema_invalid_file_is_ignored(tmp_path):
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / f"{SHA}.json").write_text('{"schema_version": 2}')
    (tmp_path / "prefs.json").write_text('{"lang": "de"}')
    s = StateStore(tmp_path)
    assert s.get_draft(SHA) is None and s.list_drafts() == []
    assert s.get_prefs() == Prefs()


def test_non_sha_path_is_refused(tmp_path):
    with pytest.raises(ValueError):
        StateStore(tmp_path).get_draft("../prefs")


def test_prefs_defaults_and_roundtrip(tmp_path):
    s = StateStore(tmp_path)
    assert s.get_prefs() == Prefs()
    s.put_prefs(Prefs(lang="en", refusal_card_seen=True))
    assert StateStore(tmp_path).get_prefs().lang == "en"


def test_unsaved_written_latest_per_sha_and_delete(tmp_path):
    s = StateStore(tmp_path)
    s.put_draft(SHA, draft_in(n=3))
    s.put_draft(SHA, draft_in(n=2))
    s.put_draft("b" * 64, draft_in("b" * 64, 1))
    assert s.unsaved_written() == 3
    s.delete_draft("b" * 64)
    assert s.unsaved_written() == 2


def test_exit_line(tmp_path):
    s = StateStore(tmp_path)
    assert s.exit_line() is None
    s.put_draft(SHA, draft_in(n=2))
    assert s.exit_line() == "2 kaydedilmemiş kesim taslağa yazıldı; sonraki açılışta geri yüklenir."


def test_neshter_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NESHTER_HOME", str(tmp_path))
    assert neshter_home() == tmp_path


def test_neshter_home_empty_env_falls_back(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("NESHTER_HOME", "")
    assert neshter_home() == Path.home() / "Neshter"
