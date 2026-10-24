"""Materialise a basis set from the Basis Set Exchange, on demand.

vibe-qc bundles 255 basis-set records. The Basis Set Exchange catalogues
776, so roughly 520 sets are simply unreachable from a stock install. This
module makes one of them reachable without adding it to the bundle: it
turns the BSE record into a QVF-Basis record with ``kind = bse`` in its
provenance, stores it by content hash in the per-user fetch cache
(:mod:`vibeqc.basis_store`, layer 4 of the resolution order) and renders
the ``.g94`` (plus ``.ecp`` sidecar) libint and libecpint read.

**This is not a network fetch.** The ``basis-set-exchange`` distribution
ships the whole catalogue as package data (~334 MB), so a lookup is a local
read and works on an air-gapped host. Two consequences worth knowing:

* What you get is pinned by the *package version*, not by what a server
  happened to serve today, so a run is reproducible as long as the
  environment is. The version is recorded in every record's provenance and
  in the cache index.
* 334 MB is far too much to require of every install, so it is the optional
  ``[bse]`` extra and is imported lazily. Without it, nothing here runs and
  the bundled library behaves exactly as before.

Nothing in this module changes how a basis resolves by itself.
:func:`fetch_basis_from_bse` writes the cache and returns paths; making
libint look at them is :func:`ensure_bases_available`'s job, which hands
the names to :func:`vibeqc.basis_store.resolve`. A third layer is opt-in and
logged, not automatic.

Licensing: the same terms as the bundled BSE-derived records, recorded in
``docs/license.md`` § 3a. Basis-set parameters are numerical data rather than
creative expression; the per-record originating publications travel in the
record's description and in the rendered header, and
:func:`fetch_basis_from_bse` refuses to write a record whose header it could
not obtain.
"""

from __future__ import annotations

import os
import re
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

# ``<Sym>-ECP <lmax> <ncore>``, the header ``ecp_metadata`` parses (lmax
# first, then ncore). A BSE gaussian94 record carries its ECP blocks in the
# same text as the shells; libint rejects them, so they are split off here.
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
    """Where fetched basis records are cached.

    ``$VIBEQC_BSE_CACHE`` first (tests and CI set it), then
    ``$XDG_CACHE_HOME/vibeqc/basis``, then ``~/.cache/vibeqc/basis``. Mirrors
    :mod:`vibeqc.fetch.cache`, which does the same for structures. The
    layout under it is :mod:`vibeqc.basis_store`'s: ``records/<sha256>.qvf.json``,
    ``render/<sha256>/basis/``, ``index.json`` and ``composed/``.
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
    atomic numbers the record carries. ``provenance_path`` is the record
    itself (``records/<content_sha256>.qvf.json``); its ``provenance`` block
    names the catalogue version and the BSE name.
    """

    name: str
    bse_name: str
    g94_path: Path
    ecp_path: Optional[Path]
    elements: tuple[int, ...]
    bse_version: str
    fetched_at: str
    provenance_path: Path
    content_sha256: str = ""

    @property
    def has_ecp(self) -> bool:
        return self.ecp_path is not None

    def manifest_row(self) -> dict[str, object]:
        """Flat scalar row for the ``.system`` manifest's ``[basis_library]``.

        Scalars only: the manifest's TOML emitter has no list support, so a
        sequence would be stringified. The full element list stays in the
        record, which this row points at by path.
        """
        return {
            "name": self.name,
            "bse_name": self.bse_name,
            "bse_version": self.bse_version,
            "element_count": len(self.elements),
            "has_ecp": self.has_ecp,
            "fetched_at": self.fetched_at,
            "content_sha256": self.content_sha256,
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


def _fetched_from_stored(stored, *, bse_name: str, bse_version: str) -> FetchedBasis:
    from .basis_toolkit import library
    from .basis_toolkit._element_data import ATOMIC_NUMBER

    record = library.load_record(stored.path)
    return FetchedBasis(
        name=stored.name,
        bse_name=bse_name,
        g94_path=stored.g94_path,
        ecp_path=stored.ecp_path,
        elements=tuple(sorted(ATOMIC_NUMBER[s] for s in record.elements if s in ATOMIC_NUMBER)),
        bse_version=bse_version,
        fetched_at=stored.recorded_at,
        provenance_path=stored.path,
        content_sha256=stored.content_sha256,
    )


def fetch_basis_from_bse(
    name: str,
    *,
    elements: Optional[Iterable[int]] = None,
    cache_dir: Optional[Path | str] = None,
    refresh: bool = False,
) -> FetchedBasis:
    """Render one BSE basis into the content-addressed cache and return where it landed.

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
        Re-render even when the cache holds this name from the same
        catalogue version. Identical content lands on the same record file.

    Returns
    -------
    FetchedBasis

    Raises
    ------
    BseUnavailable
        The optional distribution is not installed.
    LookupError
        The catalogue has no such basis, or it does not cover ``elements``.
    vibeqc.basis_store.BasisStoreIntegrityError
        A cached record or rendering for ``name`` no longer matches its hash.
    """
    from . import basis_store
    from .basis_toolkit import library
    from .basis_toolkit.model import Upstream

    bse = _require_bse()
    bse_name = _resolve_bse_name(bse, name)
    version = str(bse.version())
    stem = name.strip().lower()

    if not refresh:
        hit = basis_store.cache_get(stem, root=cache_dir, require={"bse_version": version})
        if hit is not None:
            return _fetched_from_stored(hit, bse_name=str(hit.extra.get("bse_name", bse_name)),
                                        bse_version=version)

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

    header = [line[1:].strip() for line in orbital.splitlines() if line.startswith("!")]
    description = "\n".join(
        [
            "Not a bundled vibe-qc basis: rendered from the Basis Set Exchange.",
            f"BSE catalogue version {version}, BSE name {bse_name}.",
            "See docs/license.md section 3a for terms; the originating publications "
            "are in BSE's own header below.",
            "",
        ]
        + header
    )
    record = library.record_from_text(
        stem, orbital, ecp if ecp.strip() else None,
        kind="bse",
        upstream=[Upstream(name="basis-set-exchange", version=version,
                           url="https://www.basissetexchange.org", identifier=bse_name)],
        citation_keys=["pritchard_bse_2019"],
        description=description,
        notes="Rendered locally from the basis-set-exchange distribution's bundled catalogue; no network access was involved.",
    )
    stored = basis_store.cache_put(record, name=stem, root=cache_dir, bse_name=bse_name, bse_version=version)
    return _fetched_from_stored(stored, bse_name=bse_name, bse_version=version)


# ---------------------------------------------------------------------------
# Making a rendered basis resolvable
# ---------------------------------------------------------------------------


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

    Thin wrapper over :func:`vibeqc.basis_store.compose`: content-addressed
    on the source root and the records, so repeated runs reuse one
    directory and two processes racing to build the same one converge.
    """
    from . import basis_store

    source = library_root()
    if source is None:
        raise LookupError(
            "no basis library is currently resolved, so there is nothing to "
            "compose a fetched basis on top of"
        )
    records = []
    for item in fetched:
        stored = basis_store.cache_get(item.name, root=cache_dir)
        if stored is None:
            raise LookupError(f"{item.name}: not in the fetch cache")
        records.append(stored)
    return basis_store.compose(source, records, root=bse_cache_dir(cache_dir))


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

    A name the bundled library (or the user store) already answers is left
    completely alone: no lookup, no cache entry, no change to resolution.
    Only when at least one name is unresolvable is the fetch cache consulted,
    the catalogue rendered, a composed root built and ``$LIBINT_DATA_PATH``
    repointed at it (:func:`vibeqc.basis_store.resolve`).

    A basis that resolves but lacks one of ``elements`` is **not** a fetch
    trigger. That case belongs to the coverage guard, which refuses naming
    the element; filling the gap from BSE would put two sources under one
    basis name with no reviewed provenance for the join.

    Returns what was rendered, empty when nothing needed to be.
    """
    from . import basis_store

    resolution = basis_store.resolve(
        names, elements=elements, fetch=True, cache_root=cache_dir,
    )
    return tuple(resolution.fetched)
