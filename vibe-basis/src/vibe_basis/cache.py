"""Content-addressed energy cache with an append-only resume journal.

A cohesive-energy campaign is thousands of SCF evaluations over days on a
remote queue. Two properties decide whether that is affordable:

**Isolated atoms are shared.** Every system in a joint objective needs
the free-atom energies of its elements, the element set across a test
set is small, and atoms are the *expensive* part -- in the GPAW
PW-limit work a single free F atom cost 100-240 s while the solid was
quick (``handovers/HANDOVER_GPAW_PW_REFERENCE.md``). Recomputing O for
MgO, CaO, BaO and Li2O separately would dominate the campaign. Keying
atom entries without a system identifier makes the sharing automatic.

**A killed campaign resumes.** Every entry is appended to a JSONL
journal as it is computed, so re-running picks up where it stopped
rather than re-spending the queue time. The journal is the cache: there
is no separate index that could disagree with it.

Keys are derived from *declared configuration*, never from observed
output. In particular the engine's reported version is deliberately not
part of the key -- ``Crystal23Engine.version()`` returns ``None`` until
its first evaluation, so keying on it would make the key change halfway
through a run and silently split the cache. The engine *name* is in the
key (and it already encodes the targeted major version), while the
observed version is recorded in the value for audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .engine import EngineEnergy


def digest(*parts: Any) -> str:
    """Stable short hash of *parts*.

    ``json.dumps(..., sort_keys=True)`` rather than ``hash()``: the
    built-in hash is salted per process, so a journal written today
    would miss every entry tomorrow.
    """
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def crystal_key(
    engine: str, system: str, basis_text: str, method: str, **settings: Any
) -> str:
    """Cache key for one crystal evaluation."""
    return "c:" + digest("crystal", engine, system, digest(basis_text), method, settings)


def atom_key(
    engine: str,
    Z: int,
    basis_text: str,
    method: str,
    *,
    host: Optional[str] = None,
    **settings: Any,
) -> str:
    """Cache key for one free-atom evaluation.

    **No system identifier when *host* is None**, which is what makes a bare atom computed for
    MgO reusable for CaO. The whole *basis_text* is hashed rather than
    just element ``Z``'s block: over-conservative, because two systems
    with different basis blobs will not share their atoms even where
    the relevant element block is identical. In practice that costs
    nothing -- a joint objective evaluates every system at the *same*
    candidate basis, so sharing is complete within the evaluation that
    matters, and across optimizer iterations the basis genuinely
    changed and the atoms genuinely must be recomputed. Extracting the
    per-element block would buy a rarer case at the price of parsing
    the basis text here, which this module deliberately does not do.

    *host* is the counterpoise host system's name, and passing it
    **disables that sharing on purpose**. A counterpoise atom is
    computed in the ghost basis of its own crystal, so the oxygen in
    MgO and the oxygen in CaO are genuinely different calculations and
    reusing one for the other would be wrong. Bare atoms keep sharing.
    """
    return "a:" + digest(
        "atom", engine, Z, digest(basis_text), method, host, settings
    )


@dataclass
class CacheEntry:
    """One cached evaluation: the key, the result, and how it was made."""

    key: str
    energy: Optional[float]
    ok: bool
    engine: str
    engine_version: Optional[str] = None
    failure_mode: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_energy(cls, key: str, result: EngineEnergy) -> "CacheEntry":
        return cls(
            key=key,
            energy=result.energy,
            ok=result.ok,
            engine=result.engine,
            engine_version=result.engine_version,
            failure_mode=result.failure_mode,
            detail=dict(result.detail),
        )

    def to_energy(self) -> EngineEnergy:
        return EngineEnergy(
            energy=self.energy,
            ok=self.ok,
            engine=self.engine,
            engine_version=self.engine_version,
            failure_mode=self.failure_mode,
            detail=dict(self.detail),
        )


class EnergyCache:
    """In-memory cache over an append-only JSONL journal.

    Parameters
    ----------
    journal
        Path to the JSONL journal. ``None`` gives a memory-only cache
        (tests, one-shot runs).
    cache_failures
        Whether a failed evaluation is remembered. Default ``True``:
        during a topology search the optimizer probes infeasible basis
        regions repeatedly, and re-submitting a job that is known to
        diverge wastes exactly the queue time this class exists to
        save. Set ``False`` when failures are expected to be transient
        (a flaky transport rather than bad physics), where remembering
        them would poison the run.
    """

    def __init__(
        self,
        journal: Optional[Path | str] = None,
        *,
        cache_failures: bool = True,
    ) -> None:
        self.journal = Path(journal) if journal is not None else None
        self.cache_failures = cache_failures
        self._entries: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        if self.journal is not None and self.journal.is_file():
            self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        """Replay the journal. Later entries win, so a re-run overwrites."""
        assert self.journal is not None
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A journal truncated mid-write by a kill -9 ends in a
                # partial line. Losing that one entry is correct and
                # cheap; refusing to load the other thousands is not.
                continue
            self._entries[record["key"]] = CacheEntry(**record)

    def _append(self, entry: CacheEntry) -> None:
        if self.journal is None:
            return
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), sort_keys=True, default=str) + "\n")

    # -- lookup --------------------------------------------------------

    def get(self, key: str) -> Optional[EngineEnergy]:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry.to_energy()

    def put(self, key: str, result: EngineEnergy) -> None:
        if not result.ok and not self.cache_failures:
            return
        entry = CacheEntry.from_energy(key, result)
        self._entries[key] = entry
        self._append(entry)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CacheEntry]:
        return iter(self._entries.values())

    @property
    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}
