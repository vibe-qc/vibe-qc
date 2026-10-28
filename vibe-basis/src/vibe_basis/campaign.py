"""A campaign directory: one resumable optimisation run (roadmap L4, M4).

A topology scan or a basis refit is hundreds of SCF evaluations over days,
and its driver will be killed before it finishes: the laptop sleeps or
quits, a host stops answering, an operator stops it to land a fix. M4's
gate asks that such a campaign resume *identically* and write nothing into
a managed checkout. :class:`Campaign` owns the directory that makes both
true::

    <workdir>/
      campaign.json      settings binding and one record per session
      campaign.lock      the live driver; a dead one's lock is taken over
      energies.jsonl     the EnergyCache journal: every finished energy
      submissions.jsonl  the LedgerTransport ledger: every queued job
      points/<key>/      one directory per evaluation, named by its key
      <records>          the recipe's results, written atomically

**Identical resumption** rests on three things. Every finished energy is
journalled under a key derived from what was computed (:mod:`.cache`), so a
replay hands the optimiser the same numbers in the same order. Point
directories are named by that key, not by a counter, so a restart cannot
overwrite an earlier point's outputs. And the settings that shape the
trajectory are bound on the first session: a later session with different
ones is refused (:class:`CampaignMismatch`) rather than silently mixing two
campaigns in one journal. Settings that do not change the trajectory, such
as parallelism or a round cap being raised, are declared ``volatile``.

**No writes into a checkout.** :meth:`Campaign.open` refuses a directory
inside a git working tree before it creates anything there.

A session that ends without :meth:`Campaign.close` (``kill -9``, a crash)
leaves its record without an end time and its lock behind; the next
session on the same host takes the lock over when its process is gone.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .cache import EnergyCache

FORMAT = 1


class ManagedCheckoutError(RuntimeError):
    """The campaign directory is inside a git working tree."""


class CampaignLocked(RuntimeError):
    """Another live driver holds the campaign."""


class CampaignMismatch(ValueError):
    """A session's settings differ from the ones the campaign was bound to."""


class BudgetExhausted(RuntimeError):
    """The campaign's evaluation budget is spent."""


class CampaignHalted(RuntimeError):
    """The campaign was halted; no further evaluation starts."""


def checkout_root(path: Path | str) -> Optional[Path]:
    """The git working tree that contains *path*, or ``None``.

    A ``.git`` file counts as well as a directory: worktrees and submodules
    keep a file there. *path* need not exist yet.
    """
    p = Path(path).expanduser().resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return d
    return None


def source_identity(path: Path | str) -> dict[str, Any]:
    """The git revision of the checkout at *path*, and whether it is dirty.

    For a campaign's session record: the code that computed a session's
    points. Returns ``{"path": ...}`` alone when *path* is not a checkout.
    """
    path = Path(path).resolve()
    out: dict[str, Any] = {"path": str(path)}

    def git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if sha is not None:
        out["git_sha"] = sha
        out["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _plain(value: Any) -> Any:
    """*value* as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def write_json_atomic(path: Path, data: Any) -> Path:
    """Write *data* to *path* so that a kill leaves the old file or the new one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)
    return path


class Campaign:
    """One campaign directory; open it with :meth:`open`.

    Use it as a context manager: leaving the block closes the session and
    records why it stopped (``"finished"``, ``"budget"`` or the exception).
    """

    RECORD = "campaign.json"
    LOCK = "campaign.lock"
    JOURNAL = "energies.jsonl"
    LEDGER = "submissions.jsonl"

    def __init__(self, workdir: Path, budget: Optional[int], sources: Mapping[str, Any]):
        self.workdir = workdir
        self.budget = budget
        self._guard = threading.Lock()
        self._charged = 0
        self._halted: Optional[str] = None
        self._closed = False
        self._ledgers: list[Any] = []
        self._took_over: Optional[dict[str, Any]] = None
        self._acquire()
        try:
            self.cache = EnergyCache(workdir / self.JOURNAL)
            path = workdir / self.RECORD
            if path.is_file():
                self.record = json.loads(path.read_text())
            else:
                self.record = {"format": FORMAT, "created_unix": time.time(),
                               "binding": None, "sessions": []}
            self.session: dict[str, Any] = {
                "started_unix": time.time(),
                "pid": os.getpid(),
                "sources": _plain(dict(sources)),
                "budget": budget,
                "journal_entries_at_start": len(self.cache),
                "ended_unix": None,
                "stopped_because": None,
            }
            if self._took_over is not None:
                self.session["took_over_lock"] = self._took_over
            self.record["sessions"].append(self.session)
            self._save()
        except BaseException:
            self._release()
            raise

    @classmethod
    def open(
        cls,
        workdir: Path | str,
        *,
        budget: Optional[int] = None,
        sources: Optional[Mapping[str, Any]] = None,
    ) -> "Campaign":
        """Open *workdir* as a campaign, creating it if needed.

        ``budget`` caps the evaluations the campaign computes over all its
        sessions, counted from the journal; ``None`` is no cap. ``sources``
        is recorded with the session (see :func:`source_identity`).
        """
        workdir = Path(workdir).expanduser().resolve()
        root = checkout_root(workdir)
        if root is not None:
            raise ManagedCheckoutError(
                f"campaign directory {workdir} is inside the git checkout {root}; "
                "campaign files belong outside every checkout"
            )
        workdir.mkdir(parents=True, exist_ok=True)
        return cls(workdir, budget, sources or {})

    # -- lock ----------------------------------------------------------

    def _acquire(self) -> None:
        lock = self.workdir / self.LOCK
        me = {"pid": os.getpid(), "host": socket.gethostname(), "started_unix": time.time()}
        for _ in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    held = json.loads(lock.read_text())
                except (OSError, json.JSONDecodeError):
                    held = {}
                same_host = held.get("host") == me["host"]
                pid = held.get("pid")
                if same_host and isinstance(pid, int) and not _alive(pid):
                    self._took_over = held
                    lock.unlink(missing_ok=True)
                    continue
                where = "this host" if same_host else f"host {held.get('host')!r}"
                raise CampaignLocked(
                    f"{self.workdir} is held by pid {pid} on {where}; if that driver "
                    f"is gone, remove {lock}"
                ) from None
            with os.fdopen(fd, "w") as fh:
                json.dump(me, fh)
            return
        raise CampaignLocked(f"could not take the lock {lock}")

    def _release(self) -> None:
        (self.workdir / self.LOCK).unlink(missing_ok=True)

    # -- record --------------------------------------------------------

    def _save(self) -> None:
        write_json_atomic(self.workdir / self.RECORD, self.record)

    @property
    def sessions(self) -> list[dict[str, Any]]:
        return self.record["sessions"]

    def bind(self, settings: Mapping[str, Any], *, volatile: Iterable[str] = ()) -> None:
        """Bind the campaign to *settings*, or check a resumed session against them.

        Keys in ``volatile`` are recorded with the session but not compared.
        Raises :class:`CampaignMismatch` naming the keys that differ.
        """
        volatile = set(volatile)
        identity = _plain({k: v for k, v in settings.items() if k not in volatile})
        bound = self.record.get("binding")
        if bound is None:
            self.record["binding"] = identity
        elif bound != identity:
            keys = sorted(k for k in set(bound) | set(identity) if bound.get(k) != identity.get(k))
            detail = ", ".join(f"{k}: {bound.get(k)!r} -> {identity.get(k)!r}" for k in keys)
            raise CampaignMismatch(
                f"{self.workdir} was started with other settings ({detail}); "
                "resume it with its own settings or start a new campaign directory"
            )
        self.session["settings"] = _plain(dict(settings))
        self._save()

    # -- evaluations ---------------------------------------------------

    def point_dir(self, key: str) -> Path:
        """The directory for the evaluation with cache key *key*."""
        return self.workdir / "points" / key.replace(":", "-")

    def charge(self) -> None:
        """Account for one new evaluation before it starts.

        Raises :class:`BudgetExhausted` when the budget is spent and
        :class:`CampaignHalted` after :meth:`halt`.
        """
        with self._guard:
            if self._halted is not None:
                raise CampaignHalted(self._halted)
            done = self.session["journal_entries_at_start"] + self._charged
            if self.budget is not None and done >= self.budget:
                self._halted = f"budget of {self.budget} evaluations spent"
                raise BudgetExhausted(self._halted)
            self._charged += 1

    def halt(self, reason: str) -> None:
        """Start no further evaluation: the next :meth:`charge` raises."""
        with self._guard:
            if self._halted is None:
                self._halted = reason

    def ledger(self, inner: Any) -> Any:
        """*inner* wrapped in a :class:`LedgerTransport` journalling to this campaign."""
        from .transports.ledger import LedgerTransport

        wrapped = LedgerTransport(inner, self.workdir / self.LEDGER)
        self._ledgers.append(wrapped)
        return wrapped

    def write_record(self, name: str, data: Any) -> Path:
        """Write a result file into the campaign directory, atomically."""
        return write_json_atomic(self.workdir / name, data)

    def summary(self) -> dict[str, Any]:
        """The campaign's sessions in brief, for a recipe's record."""
        return {
            "workdir": str(self.workdir),
            "sessions": [
                {k: s.get(k) for k in ("started_unix", "ended_unix", "sources",
                                         "journal_entries_at_start", "evaluations",
                                         "stopped_because")}
                for s in self.sessions
            ],
            "journal_entries": len(self.cache),
        }

    def close(self, stopped_because: str = "finished") -> None:
        """End the session: record why it stopped, release the lock."""
        if self._closed:
            return
        self._closed = True
        try:
            self.session.update({
                "ended_unix": time.time(),
                "stopped_because": stopped_because,
                "evaluations": self._charged,
                "cache": self.cache.stats,
            })
            if self._ledgers:
                self.session["submissions"] = sum(t.submissions for t in self._ledgers)
                self.session["adoptions"] = sum(t.adoptions for t in self._ledgers)
                self.session["open_submissions"] = [
                    r["job_id"] for t in self._ledgers for r in t.open_submissions()
                ]
            self._save()
        finally:
            self._release()

    def __enter__(self) -> "Campaign":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.close("finished")
        elif isinstance(exc, BudgetExhausted):
            self.close("budget")
        else:
            self.close(f"{exc_type.__name__}: {exc}")


__all__ = [
    "BudgetExhausted",
    "Campaign",
    "CampaignHalted",
    "CampaignLocked",
    "CampaignMismatch",
    "ManagedCheckoutError",
    "checkout_root",
    "source_identity",
    "write_json_atomic",
]
