"""A transport that records its submissions, so a restarted driver adopts them.

A campaign driver that dies (killed, crashed, the laptop slept or quit)
leaves its queued and running jobs behind. They finish on the compute host,
but nothing fetches them, and a restarted driver that reaches the same
point submits the same calculation again beside the orphan. On 2026-10-27
that cost 21 duplicate submissions and 9 SiC points computed three times.

:class:`LedgerTransport` wraps any :class:`~vibe_basis.transports.base.Transport`
and appends every submission to a JSONL ledger before returning its handle.
A submission is identified by what it would run, not by its label: the
resolved work directory, the command, the resources and the content of every
input file in the directory. When the same submission is asked for again and
the ledger holds it unfetched, the recorded job is adopted instead of
submitted, unless the queue reports that it ended without completing.

An adopted job is trusted once per process. If the caller asks for the same
submission again, the adopted job did not deliver (it vanished from the
host, or its fetch failed), and a fresh job is submitted.

The job id is only known once the queue has accepted the job, so a driver
killed between that and the ledger line leaves one job the ledger never
saw. A restart computes that point again; the orphan is visible in the
queue's own listing under its label.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from .base import TERMINAL_STATES, SUCCESS_STATES, JobHandle, Transport, TransportError

#: Ledger states after which a submission is no longer waiting to be fetched.
CLOSED_STATES = frozenset({"fetched", "abandoned"})


def _input_digest(work_dir: Path) -> dict[str, str]:
    """sha256 of every regular file directly in *work_dir*, by name."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(work_dir.iterdir())
        if p.is_file()
    }


def submission_key(
    work_dir: Path | str,
    command: Sequence[str],
    cpus: Optional[int],
    wall_time_s: Optional[int],
) -> str:
    """What a submission would run, as a stable short hash."""
    blob = json.dumps(
        [str(Path(work_dir).resolve()), list(command), cpus, wall_time_s,
         _input_digest(Path(work_dir))],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class LedgerTransport(Transport):
    """Wrap *inner* and journal its submissions to *ledger* (JSONL).

    The wrapped transport does the work; this class only decides whether a
    submission is new. :meth:`open_submissions` lists jobs submitted or
    adopted but never fetched: after a clean stop they are the jobs still
    on the queue, which a later run adopts or an operator cancels.
    """

    def __init__(self, inner: Transport, ledger: Path | str) -> None:
        self.inner = inner
        self.ledger = Path(ledger)
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {}
        self._by_job: dict[str, str] = {}
        self._adopted: set[str] = set()
        self.adoptions = 0
        self.submissions = 0
        if self.ledger.is_file():
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a kill
                self._note(record)

    def _note(self, record: dict[str, Any]) -> None:
        self._latest[record["key"]] = record
        self._by_job[record["job_id"]] = record["key"]

    def _record(self, key: str, state: str, handle: JobHandle, **extra: Any) -> None:
        record = {
            "key": key, "state": state, "job_id": handle.job_id,
            "work_dir": str(handle.work_dir), "command": list(handle.command),
            "label": handle.label, "t": time.time(), **extra,
        }
        with self._lock:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            self._note(record)

    def open_submissions(self) -> list[dict[str, Any]]:
        """Ledger records of jobs submitted or adopted and not yet fetched."""
        with self._lock:
            return [r for r in self._latest.values() if r["state"] not in CLOSED_STATES]

    def submit(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        *,
        cpus: Optional[int] = None,
        wall_time_s: Optional[int] = None,
        label: Optional[str] = None,
    ) -> JobHandle:
        key = submission_key(work_dir, command, cpus, wall_time_s)
        with self._lock:
            known = self._latest.get(key)
            retry = key in self._adopted
            if known is not None and known["state"] not in CLOSED_STATES and not retry:
                self._adopted.add(key)
            else:
                known = None
        if known is not None:
            handle = JobHandle(
                job_id=known["job_id"], work_dir=Path(work_dir),
                command=tuple(command), label=known.get("label"),
            )
            try:
                state = self.inner.poll(handle)
            except TransportError:
                # Says nothing about the job; wait() will keep asking.
                state = "unknown"
            if state in TERMINAL_STATES and state not in SUCCESS_STATES:
                # Ended without completing (cancelled, crashed, timed out):
                # its output is not a result of this submission.
                self._record(key, "abandoned", handle, queue_state=state)
            else:
                self._record(key, "adopted", handle, queue_state=state)
                self.adoptions += 1
                return handle
        handle = self.inner.submit(
            work_dir, command, cpus=cpus, wall_time_s=wall_time_s, label=label,
        )
        self._record(key, "submitted", handle)
        self.submissions += 1
        return handle

    def poll(self, handle: JobHandle) -> str:
        return self.inner.poll(handle)

    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        out = self.inner.fetch(handle, dest)
        with self._lock:
            key = self._by_job.get(handle.job_id)
        if key is not None:
            self._record(key, "fetched", handle)
        return out


__all__ = ["CLOSED_STATES", "LedgerTransport", "submission_key"]
