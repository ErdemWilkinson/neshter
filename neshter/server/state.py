"""Sunucu tarafı taslak ve tercih deposu: `<NESHTER_HOME>/.state/` altında, model SHA-256'sına göre anahtarlı.

Değişmez: kesimlerin tek doğruluk kaynağı istemci sepetidir; buradaki taslak yalnızca geri yükleme için
aynadır. İşlemler (ops) sunucu için opaktır; taslak `source_path` ya da model çıktısı metni tutmaz.
Depo tembeldir: ilk yazmaya kadar hiçbir klasör oluşmaz.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, ValidationError

log = logging.getLogger("neshter.server")

SHA_RE = r"^[0-9a-f]{64}$"
DRAFT_MAX_BYTES = 8 * 1024 * 1024
DRAFT_KEEP = 10
DRAFT_MAX_AGE_DAYS = 60


def neshter_home() -> Path:
    """`NESHTER_HOME` (doluysa), yoksa `~/Neshter`. Çağrı anında okunur."""
    env = os.environ.get("NESHTER_HOME")
    return Path(env) if env else Path.home() / "Neshter"


class DraftIn(BaseModel, extra="forbid"):
    schema_version: Literal[1]
    sha256: str = Field(pattern=SHA_RE)
    model_name: str = Field(max_length=256)
    unsaved_count: int = Field(ge=0, le=10_000)
    basket: list[dict[str, Any]] = Field(max_length=1000)
    shelf: list[dict[str, Any]] = Field(max_length=1000)
    saved_ops: list[str] = Field(max_length=1000)
    prompts: list[Annotated[str, Field(max_length=8000)]] = Field(max_length=20)


class Draft(DraftIn):
    updated_at: AwareDatetime


class DraftSummary(BaseModel):
    sha256: str
    model_name: str
    updated_at: AwareDatetime
    unsaved_count: int


class Prefs(BaseModel, extra="forbid"):
    lang: Literal["tr", "en"] | None = None
    quality: Literal["high", "balanced", "low"] = "balanced"
    reduced_motion: Literal["system", "on", "off"] = "system"
    refusal_card_seen: bool = False


_SHA = re.compile(SHA_RE)


def _check_sha(sha: str) -> str:
    if not _SHA.fullmatch(sha):
        raise ValueError("Geçersiz model SHA-256 değeri.")
    return sha


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._written: dict[str, int] = {}  # bu süreçte yazılan SHA -> son unsaved_count

    # -- dosya yardımcıları --

    @property
    def _drafts_dir(self) -> Path:
        return self.root / "drafts"

    @property
    def _prefs_path(self) -> Path:
        return self.root / "prefs.json"

    def _draft_path(self, sha: str) -> Path:
        return self._drafts_dir / f"{_check_sha(sha)}.json"

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    @staticmethod
    def _read(path: Path, model: type[BaseModel]):
        """Bozuk ya da şeması geçersiz dosya günlüğe yazılır ve yok sayılır."""
        try:
            return model.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValidationError, ValueError) as e:
            log.warning("Durum dosyası okunamadı, yok sayıldı: %s (%s)", path.name, type(e).__name__)
            return None

    def _scan(self) -> list[Draft]:
        if not self._drafts_dir.is_dir():
            return []
        out = []
        for p in self._drafts_dir.glob("*.json"):
            if not _SHA.fullmatch(p.stem):
                continue
            d = self._read(p, Draft)
            if d is not None and d.sha256 == p.stem:
                out.append(d)
        return sorted(out, key=lambda d: d.updated_at, reverse=True)

    def _remove(self, sha: str) -> bool:
        try:
            self._draft_path(sha).unlink()
        except FileNotFoundError:
            return False
        finally:
            self._written.pop(sha, None)
        return True

    # -- taslaklar --

    def list_drafts(self, now: datetime | None = None) -> list[DraftSummary]:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=DRAFT_MAX_AGE_DAYS)
        with self._lock:
            keep = []
            for d in self._scan():
                if d.updated_at < cutoff:
                    self._remove(d.sha256)
                else:
                    keep.append(d)
        return [
            DraftSummary(sha256=d.sha256, model_name=d.model_name, updated_at=d.updated_at, unsaved_count=d.unsaved_count)
            for d in keep
        ]

    def get_draft(self, sha: str) -> Draft | None:
        with self._lock:
            return self._read(self._draft_path(sha), Draft)

    def put_draft(self, sha: str, d: DraftIn, now: datetime | None = None) -> Draft:
        _check_sha(sha)
        if d.sha256 != sha:
            raise ValueError("Taslaktaki SHA-256 yol ile uyuşmuyor.")
        draft = Draft(**d.model_dump(), updated_at=now or datetime.now(timezone.utc))
        with self._lock:
            self._atomic_write(self._draft_path(sha), draft.model_dump_json().encode("utf-8"))
            self._written[sha] = draft.unsaved_count
            for old in self._scan()[DRAFT_KEEP:]:
                self._remove(old.sha256)
        return draft

    def delete_draft(self, sha: str) -> bool:
        with self._lock:
            return self._remove(sha)

    # -- tercihler --

    def get_prefs(self) -> Prefs:
        with self._lock:
            return self._read(self._prefs_path, Prefs) or Prefs()

    def put_prefs(self, p: Prefs) -> Prefs:
        with self._lock:
            self._atomic_write(self._prefs_path, p.model_dump_json().encode("utf-8"))
        return p

    # -- çıkış satırı --

    def unsaved_written(self) -> int:
        with self._lock:
            return sum(self._written.values())

    def exit_line(self) -> str | None:
        n = self.unsaved_written()
        if n == 0:
            return None
        return f"{n} kaydedilmemiş kesim taslağa yazıldı; sonraki açılışta geri yüklenir."
