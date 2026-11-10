"""Arka plan iş kuyruğu: tek işçi iş parçacığı (aynı anda tek ağır iş), iptal ve ilerleme."""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

JobFn = Callable[[Callable[[float, str], None], threading.Event], Any]
State = Literal["queued", "running", "done", "error", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"done", "error", "cancelled"})


@dataclass
class JobStatus:
    id: str
    state: State
    progress: float = 0.0
    message: str = ""
    result: Any | None = None
    error: str | None = None


@dataclass
class _Job:
    status: JobStatus
    fn: JobFn
    cancel: threading.Event


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="neshter-jobs", daemon=True)
        self._worker.start()

    def submit(self, fn: JobFn) -> str:
        job_id = uuid.uuid4().hex
        job = _Job(JobStatus(id=job_id, state="queued"), fn, threading.Event())
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job_id

    def status(self, job_id: str) -> JobStatus:
        with self._lock:
            return replace(self._jobs[job_id].status)  # KeyError: bilinmeyen iş

    def cancel(self, job_id: str) -> None:
        """Kuyruktaki iş hemen "cancelled" olur; çalışan işin olayı kurulur ve iş döndüğünde "cancelled" olur."""
        with self._lock:
            job = self._jobs[job_id]
            if job.status.state in TERMINAL:
                return
            job.cancel.set()
            if job.status.state == "queued":
                job.status.state = "cancelled"

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel.set()
        self._queue.put(None)

    # --- işçi ---

    def _run(self) -> None:
        while (job_id := self._queue.get()) is not None:
            with self._lock:
                job = self._jobs[job_id]
                if job.status.state != "queued":  # kuyrukta iptal edildi
                    continue
                job.status.state = "running"
            self._execute(job)

    def _execute(self, job: _Job) -> None:
        def report(progress: float, message: str = "") -> None:
            with self._lock:
                if job.status.state == "running":
                    job.status.progress = min(1.0, max(0.0, float(progress)))
                    job.status.message = message

        result: Any = None
        error: str | None = None
        try:
            result = job.fn(report, job.cancel)
        except BaseException as e:  # işçi iş parçacığı hiçbir iş yüzünden ölmemeli
            error = str(e) or type(e).__name__
        with self._lock:
            st = job.status
            if job.cancel.is_set():
                st.state, st.result, st.error = "cancelled", None, None
            elif error is not None:
                st.state, st.error = "error", error
            else:
                st.state, st.result, st.progress = "done", result, 1.0
