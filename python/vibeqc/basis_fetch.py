"""Materialise a basis set from the Basis Set Exchange, on demand.

vibe-qc bundles 255 `.g94` files. The Basis Set Exchange catalogues 776, so
roughly 595 sets are simply unreachable from a stock install. This module
makes one of them reachable without adding it to the bundle: it renders the
BSE record to the `.g94` (plus `.ecp` sidecar) layout libint and libecpint
already read, into a per-user cache, with a provenance record beside it.

**This is not a network fetch.** The ``basis-set-exchange`` distribution
ships the whole catalogue as package data (~334 MB), so a lookup is a local
read and works on an air-gapped host. Two consequences worth knowing:

* What you get is pinned by the *package version*, not by what a server
  happened to serve today, so a run is reproducible as long as the
  environment is. The version is recorded in every provenance file.
* 334 MB is far too much to require of every install, so it is the optional
  ``[bse]`` extra and is imported lazily. Without it, nothing here runs and
  the bundled library behaves exactly as before.

Nothing in this module changes how a basis resolves. It writes files and
returns paths; making libint look at them is the caller's decision, which is
deliberate -- vibe-qc already resolves through two layers, and that
precedence has silently shadowed newer data once already (see
``_warn_if_basis_overlay_is_stale``). A third layer is opt-in and logged, not
automatic.

Licensing: the same terms as the bundled BSE-derived files, recorded in
``docs/license.md`` § 3a. Basis-set parameters are numerical data rather than
creative expression; the per-record originating publications travel in the
file header, and :func:`fetch_basis_from_bse` refuses to write a record whose
header it could not obtain.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = [
    "BseUnavailable",
    "FetchedBasis",
    "bse_cache_dir",
    "bse_catalogue_version",
    "compose_library",
    "ensure_bases_available",
    "fetch_basis_from_bse",
    "fetch_requested_by_env",
    "library_root",
]

# ``<Sym>-ECP <lmax> <ncore>``. Deliberately a second copy of the pattern in
# ``scripts/basisset_dev/split_ecp_g94.py``: that script runs during
# ``setup_basis_library.sh``, before vibe-qc is importable, so it is
# stdlib-only by design and cannot import this module. Both are pinned
# against the same shipped sidecars by tests, and the field order (lmax
# first, then ncore) is the one ``ecp_metadata`` parses.
_ECP_HEADER_RE = re.compile(r"^\s*([A-Z][A-Za-z]?)-ECP\s+\d+\s+\d+\s*$")


class BseUnavailable(RuntimeError):
    """The ``basis-set-exchange`` distribution is not importable."""


def _require_bse():
    """Import ``basis_set_exchange`` or explain how to get it."""
    try:
        import basis_set_exchange  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise BseUnavailable(
            "fetching a basis from the Basis Set Exchange needs the "
            "'basis-set-exchange' distribution, which vibe-qc does not "
            "install by default because it carries ~334 MB of catalogue "
            "data.\n"
            "  pip install 'vibe-qc[bse]'      (or: pip install basis-set-exchange)\n"
            "Without it the bundled library is unchanged and unbundled "
            "basis names keep refusing, which is the default behaviour."
        ) from exc
    return basis_set_exchange


def bse_catalogue_version() -> str:
    """Version of the installed BSE catalogue, e.g. ``"0.12"``."""
    return str(_require_bse().version())


def bse_cache_dir(cache_dir: Optional[Path | str] = None) -> Path:
    """Where fetched basis data is written.

    ``$VIBEQC_BSE_CACHE`` first (tests and CI set it), then
    ``$XDG_CACHE_HOME/vibeqc/basis``, then ``~/.cache/vibeqc/basis``. Mirrors
    :mod:`vibeqc.fetch.cache`, which does the same for structures.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if override := os.environ.get("VIBEQC_BSE_CACHE"):
        return Path(override).expanduser()
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg).expanduser() / "vibeqc" / "basis"
    return Path("~/.cache/vibeqc/basis").expanduser()


@dataclass(frozen=True)
class FetchedBasis:
    """One materialised basis, and enough provenance to cite and reproduce it.

    ``ecp_path`` is ``None`` for an all-electron set. ``elements`` are the
    atomic numbers the written record actually carries, read back from the
    file rather than taken from the request.
    """

    name: str
    bse_name: str
    g94_path: Path
    ecp_path: Optional[Path]
    elements: tuple[int, ...]
    bse_version: str
    fetched_at: str
    provenance_path: Path

    @property
    def has_ecp(self) -> bool:
        return self.ecp_path is not None

    def manifest_row(self) -> dict[str, object]:
        """Flat scalar row for the ``.system`` manifest's ``[basis_library]``.

        Scalars only: the manifest's TOML emitter has no list support, so a
        sequence would be stringified. The full element list stays in the
        provenance JSON, which this row points at by path.
        """
        return {
            "name": self.name,
            "bse_name": self.bse_name,
            "bse_version": self.bse_version,
            "element_count": len(self.elements),
            "has_ecp": self.has_ecp,
            "fetched_at": self.fetched_at,
            "g94_path": str(self.g94_path),
            "ecp_path": str(self.ecp_path) if self.ecp_path else "",
            "provenance_path": str(self.provenance_path),
        }


def _split_ecp(text: str) -> tuple[str, str]:
    """Split a BSE ``gaussian94`` payload into orbital and ECP halves.

    libint2's parser rejects ``<Sym>-ECP`` header lines outright, so a file
    carrying both never loads. Everything from the first ECP header to the
    end is the ECP half; the orbital half keeps the comment header, so the
    originating publications travel with it.
    """
    lines = text.splitlines(keepends=True)
    first = next(
        (i for i, line in enumerate(lines) if _ECP_HEADER_RE.match(line)), None
    )
    if first is None:
        return text, ""
    # A '****' terminator immediately before the first ECP block closes the
    # orbital section and belongs with it.
    return "".join(lines[:first]), "".join(lines[first:])


_ELEMENT_HEADER_RE = re.compile(r"^\s*([A-Z][a-z]?)\s+0\s*$", re.MULTILINE)


def _elements_in(text: str) -> tuple[int, ...]:
    """Atomic numbers of the element blocks in a ``.g94`` payload."""
    from .basis_registry import element_number

    out: set[int] = set()
    for symbol in _ELEMENT_HEADER_RE.findall(text):
        z = element_number(symbol)
        if z:
            out.add(z)
    return tuple(sorted(out))


def _resolve_bse_name(bse, name: str) -> str:
    """Map a vibe-qc basis name onto the BSE catalogue's own spelling."""
    metadata = bse.get_metadata()
    wanted = name.strip().lower()
    if wanted in metadata:
        return wanted
    raise LookupError(
        f"the Basis Set Exchange catalogue (version {bse.version()}) has no "
        f"basis named {name!r}. Names are matched exactly, lowercased; "
        f"the catalogue holds {len(metadata)} sets."
    )


def fetch_basis_from_bse(
    name: str,
    *,
    elements: Optional[Iterable[int]] = None,
    cache_dir: Optional[Path | str] = None,
    refresh: bool = False,
) -> FetchedBasis:
    """Render one BSE basis into the cache and return where it landed.

    Parameters
    ----------
    name
        Basis-set name, matched against the BSE catalogue case-insensitively.
    elements
        Atomic numbers the caller needs. When given, a record that does not
        cover **all** of them is refused rather than written: a basis missing
        an element is what the coverage guard exists to catch, and a
        half-covering cache entry would defeat it. When ``None`` the whole
        record is written as-is.
    cache_dir
        Override the cache root; see :func:`bse_cache_dir`.
    refresh
        Rewrite even when a cache entry from the same catalogue version is
        already present.

    Returns
    -------
    FetchedBasis

    Raises
    ------
    BseUnavailable
        The optional distribution is not installed.
    LookupError
        The catalogue has no such basis, or it does not cover ``elements``.
    """
    bse = _require_bse()
    bse_name = _resolve_bse_name(bse, name)
    version = str(bse.version())
    stem = name.strip().lower()

    root = bse_cache_dir(cache_dir) / "basis"
    g94_path = root / f"{stem}.g94"
    ecp_path = root / f"{stem}.ecp"
    prov_path = root / f"{stem}.provenance.json"

    if not refresh and g94_path.is_file() and prov_path.is_file():
        try:
            cached = json.loads(prov_path.read_text())
        except (OSError, ValueError):
            cached = {}
        if cached.get("bse_version") == version:
            return FetchedBasis(
                name=stem,
                bse_name=cached.get("bse_name", bse_name),
                g94_path=g94_path,
                ecp_path=ecp_path if ecp_path.is_file() else None,
                elements=tuple(cached.get("elements", ())),
                bse_version=version,
                fetched_at=cached.get("fetched_at", ""),
                provenance_path=prov_path,
            )

    text = bse.get_basis(name=bse_name, fmt="gaussian94", header=True)
    if not text or not text.strip():
        raise LookupError(f"{name}: the catalogue returned an empty record")

    orbital, ecp = _split_ecp(text)
    covered = _elements_in(orbital)
    if not covered:
        raise LookupError(
            f"{name}: the rendered record has no element blocks, so nothing "
            "would load. Refusing to write it."
        )

    if elements is not None:
        wanted = {int(z) for z in elements}
        missing = sorted(wanted - set(covered))
        if missing:
            from .basis_registry import element_symbol

            names = ", ".join(f"{element_symbol(z)} (Z={z})" for z in missing)
            raise LookupError(
                f"{name}: the Basis Set Exchange record does not cover "
                f"{names}. Nothing was written to the cache. Fetching "
                "supplies whole basis sets; it deliberately does not graft "
                "BSE blocks onto a bundled record to fill a gap, because the "
                "join between two sources under one basis name would have no "
                "reviewed provenance."
            )

    header = (
        f"! vibeqc: rendered from the Basis Set Exchange, catalogue version "
        f"{version}.\n"
        f"! BSE name: {bse_name}\n"
        f"! Not a bundled vibe-qc basis. See docs/license.md section 3a for "
        f"terms;\n"
        f"! the originating publications are in the BSE header below.\n!\n"
    )

    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(g94_path, header + orbital)
    if ecp.strip():
        _write_atomic(ecp_path, header + ecp)
    elif ecp_path.exists():
        ecp_path.unlink()

    fetched_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "name": stem,
        "bse_name": bse_name,
        "bse_version": version,
        "fetched_at": fetched_at,
        "elements": list(covered),
        "has_ecp": bool(ecp.strip()),
        "citation_keys": ["pritchard_bse_2019", "_bse_bundled_provenance"],
        "note": (
            "Rendered locally from the basis-set-exchange distribution's "
            "bundled catalogue; no network access was involved."
        ),
    }
    _write_atomic(prov_path, json.dumps(provenance, indent=2) + "\n")

    return FetchedBasis(
        name=stem,
        bse_name=bse_name,
        g94_path=g94_path,
        ecp_path=ecp_path if ecp.strip() else None,
        elements=covered,
        bse_version=version,
        fetched_at=fetched_at,
        provenance_path=prov_path,
    )


def _write_atomic(path: Path, text: str) -> None:
    """Write via a sibling temp file and rename.

    A half-written `.g94` in the cache would be indistinguishable from a
    short basis, and the next run would load it.
    """
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Making a rendered basis resolvable
# ---------------------------------------------------------------------------
#
# libint reads ``$LIBINT_DATA_PATH/basis/<name>.g94`` and honours exactly one
# root, so pointing it at the cache would *hide* the bundled library: a run
# using a fetched orbital basis with a bundled auxiliary would then fail on
# the auxiliary. The composed directory below is the fix -- symlinks to every
# file of the current root plus the rendered ones, so it is a strict superset
# and nothing that resolved before stops resolving.


def library_root() -> Optional[Path]:
    """The directory libint is reading right now, if there is one.

    The environment wins, because that is what libint itself consults at
    BasisSet construction; the package resolution is the fallback for a
    process that has not set it. Recorded per run in the ``.system``
    manifest so a result says which library layer answered.
    """
    if root := os.environ.get("LIBINT_DATA_PATH"):
        return Path(root)
    from . import _resolve_basis_library

    return _resolve_basis_library()


def compose_library(
    fetched: Sequence[FetchedBasis], *, cache_dir: Optional[Path | str] = None
) -> Path:
    """Build a library root carrying the current one **plus** ``fetched``.

    Content-addressed on (source root, rendered stems), so repeated runs
    reuse one directory instead of rebuilding it, and two processes racing
    to build the same one converge rather than clobbering each other.
    """
    source = library_root()
    if source is None:
        raise LookupError(
            "no basis library is currently resolved, so there is nothing to "
            "compose a fetched basis on top of"
        )
    stems = sorted(f.name for f in fetched)
    key = hashlib.sha256(
        "\0".join([str(Path(source).resolve())] + stems).encode()
    ).hexdigest()[:16]
    root = bse_cache_dir(cache_dir) / "composed" / key
    if (root / "basis").is_dir():
        return root

    tmp = root.with_name(f"{root.name}.partial-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "basis").mkdir(parents=True)
    for entry in (Path(source) / "basis").iterdir():
        if entry.is_file():
            (tmp / "basis" / entry.name).symlink_to(entry.resolve())
    for item in fetched:
        for path in (item.g94_path, item.ecp_path):
            if path is None:
                continue
            link = tmp / "basis" / path.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(path).resolve())
    try:
        tmp.replace(root)
    except OSError:
        # Another process finished the same composition first. Its content is
        # ours by construction, so drop the duplicate and use theirs.
        shutil.rmtree(tmp, ignore_errors=True)
    return root


def fetch_requested_by_env() -> bool:
    """Whether ``$VIBEQC_FETCH_BSE`` asks for on-demand fetching."""
    return os.environ.get("VIBEQC_FETCH_BSE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def ensure_bases_available(
    names: Iterable[Optional[str]],
    *,
    elements: Optional[Iterable[int]] = None,
    cache_dir: Optional[Path | str] = None,
) -> tuple[FetchedBasis, ...]:
    """Render any of ``names`` that does not resolve, and make it resolvable.

    A name the bundled library already answers is left completely alone: no
    lookup, no cache entry, no change to resolution. Only when at least one
    name is unresolvable is a composed root built and ``$LIBINT_DATA_PATH``
    repointed at it.

    A basis that resolves but lacks one of ``elements`` is **not** a fetch
    trigger. That case belongs to the coverage guard, which refuses naming
    the element; filling the gap from BSE would put two sources under one
    basis name with no reviewed provenance for the join.

    Returns what was rendered, empty when nothing needed to be.
    """
    from .basis_registry import basis_file_path

    missing: list[str] = []
    for name in names:
        text = str(name or "").strip()
        if not text or basis_file_path(text) is not None:
            continue
        if text not in missing:
            missing.append(text)
    if not missing:
        return ()

    wanted = tuple(int(z) for z in elements) if elements is not None else None
    fetched = tuple(
        fetch_basis_from_bse(name, elements=wanted, cache_dir=cache_dir)
        for name in missing
    )
    os.environ["LIBINT_DATA_PATH"] = str(compose_library(fetched, cache_dir=cache_dir))
    return fetched
