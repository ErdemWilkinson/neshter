"""İlerleme ve iptal tesisatı: uzun hesaplar (av, etki) iş kuyruğundan gelen `report`/`cancel` ile konuşur."""

from __future__ import annotations

import threading
from collections.abc import Callable

ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    """Kullanıcı işi iptal etti; iş kuyruğu bunu "cancelled" durumuna çevirir."""

    def __init__(self, message: str = "İşlem iptal edildi.") -> None:
        super().__init__(message)


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def sub_progress(progress: ProgressFn | None, lo: float, hi: float) -> ProgressFn | None:
    """0..1 aralığını lo..hi aralığına eşleyen ilerleme geri çağrısı (progress None ise None)."""
    if progress is None:
        return None

    def report(p: float, message: str = "") -> None:
        progress(lo + (hi - lo) * min(1.0, max(0.0, p)), message)

    return report
