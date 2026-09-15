"""On-disk JSON cache for external structure fetches.

Cache root: ``$XDG_CACHE_HOME/vibeqc/fetch/`` (fall back to
``~/.cache/vibeqc/fetch/`` if ``$XDG_CACHE_HOME`` unset).

One JSON file per ``(source_db, source_id)``:
    ``<cache_root>/<source_db>/<sanitised_id>.json``

TTL: 30 days for OPTIMADE / MP / NOMAD; **infinite** for COD (CIFs
there don't change once published).

Cache schema (one entry):
    {
      "fetched_at": ISO8601,
      "raw":         <verbatim API response, JSON-serialisable>,
      "test_system": <serialised TestSystem dict, post-heuristics>
    }

Storing the raw API response lets us re-derive the TestSystem when the
heuristic table or Provenance schema changes without re-fetching.

See ``docs/tutorial/external_data_fetcher.md`` Sec. 7.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# 30 days = 30 * 24 * 3600 s. Infinite TTL signalled by None.
DEFAULT_TTL_SECONDS = 30 * 24 * 3600
COD_TTL_SECONDS: Optional[int] = None  # COD CIFs are effectively immutable.

# Source-DB string constants. Single point of truth for the ``source_db``
# field on ``Provenance`` and the cache directory layout. Phase-1
# clients (OPTIMADE/COD/MP/NOMAD) historically hard-coded their own
# values; the phase-2 reference fetcher uses these named constants from
# day one to avoid drift between the cache key and the Provenance field.
SOURCE_DB_CCCBDB = "CCCBDB"
"""NIST Computational Chemistry Comparison and Benchmark Database (SRD 101).
Citation DOI: ``doi:10.18434/T47C7Z`` (Release 22, 2022)."""
SOURCE_DB_WEBBOOK = "WebBook"
"""NIST Chemistry WebBook (SRD 69). Used as a fallback for molecules
CCCBDB doesn't cover; see
``docs/tutorial/external_data_fetcher.md`` Sec. 4.2."""

# Sanitiser for filesystem-safe ids: anything outside [A-Za-z0-9._-]
# becomes '_'. Provider-prefixed OPTIMADE ids ("mp/mp-1265") get
# rewritten to "mp_mp-1265" -- readable round-trip, no surprises on
# case-sensitive vs case-insensitive filesystems.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitise(s: str) -> str:
    return _SAFE.sub("_", s)


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_root() -> Path:
    """Resolve the cache root, honouring ``$VIBEQC_FETCH_CACHE_ROOT``
    (override for tests / CI), then ``$XDG_CACHE_HOME``, then
    ``~/.cache``."""
    if (override := os.environ.get("VIBEQC_FETCH_CACHE_ROOT")):
        return Path(override).expanduser()
    if (xdg := os.environ.get("XDG_CACHE_HOME")):
        return Path(xdg).expanduser() / "vibeqc" / "fetch"
    return Path("~/.cache/vibeqc/fetch").expanduser()


class FetchCacheMiss(KeyError):
    """Raised by ``--cache-only`` callers when the entry is missing."""


class FetchCache:
    """File-backed JSON cache.

    Thread-unsafe by design -- the fetcher is a single-process CLI
    tool; concurrent invocations are accepted to clobber each other's
    in-flight writes (last write wins, which is correct for an
    idempotent cache).
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.root = Path(root).expanduser() if root else cache_root()
        self.ttl_seconds = ttl_seconds

    # ---- Path helpers ----------------------------------------------------

    def path_for(self, source_db: str, source_id: str) -> Path:
        return self.root / _sanitise(source_db) / f"{_sanitise(source_id)}.json"

    # ---- Read / write ----------------------------------------------------

    def get(
        self,
        source_db: str,
        source_id: str,
        *,
        ttl_seconds: Optional[int] = ...,    # type: ignore[assignment]
    ) -> Optional[dict[str, Any]]:
        """Return the cached entry or ``None`` if missing / expired.

        The per-call ``ttl_seconds`` overrides the instance default.
        Pass ``None`` for infinite (COD).
        """
        if ttl_seconds is ...:
            ttl_seconds = self.ttl_seconds
        path = self.path_for(source_db, source_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt cache entry: drop it and force refetch.
            return None
        if ttl_seconds is None:
            return data
        ts = data.get("fetched_at")
        if not isinstance(ts, str):
            return None
        try:
            fetched_at = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None
        age = datetime.now(tz=timezone.utc) - fetched_at
        if age > timedelta(seconds=ttl_seconds):
            return None
        return data

    def put(
        self,
        source_db: str,
        source_id: str,
        *,
        raw: Any,
        test_system: Any,
    ) -> Path:
        """Write an entry. ``test_system`` may be a dataclass or dict."""
        path = self.path_for(source_db, source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(test_system, "__dataclass_fields__"):
            ts_dict = asdict(test_system)
        else:
            ts_dict = test_system
        payload = {
            "fetched_at": _utcnow_iso(),
            "raw": raw,
            "test_system": ts_dict,
        }
        # Atomic write: write to temp then rename. On macOS / Linux
        # rename within the same directory is atomic.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    def has(self, source_db: str, source_id: str) -> bool:
        return self.path_for(source_db, source_id).is_file()
