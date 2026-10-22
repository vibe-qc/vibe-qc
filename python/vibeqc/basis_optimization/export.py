"""Save an optimised basis as a basis-store record, with its provenance.

Every in-process recipe ends with a ``{symbol: CrystalAtomBasis}`` dict: the
starting basis with its free exponents and coefficients moved to the optimum.
This module turns that dict into a QVF-Basis record of kind ``vibe-basis``
and hands it to the user's basis store (:mod:`vibeqc.basis_store`, #410), so
a later calculation loads it by name like any bundled set::

    from vibeqc.basis_optimization.export import save_optimised_basis

    save_optimised_basis(res.optimized_atoms, name="pob-tzvp-h2o-pbe",
                         starting_basis="pob-TZVP",
                         method_citations=res.citations,
                         objective={"reference": "RKS/PBE", "systems": ["H2O"]})
    vq.BasisSet(mol, "pob-tzvp-h2o-pbe")       # resolves from the store

The result objects of the recipes wrap this as ``res.save_to_store(name)``.

What the record says about itself
---------------------------------
* ``kind = "vibe-basis"`` and ``origin = "generated"``.
* ``upstream``: this optimiser with the running vibe-qc version and the run
  identity, and the starting basis it was optimised from.
* ``citations``: the optimiser's method papers (BDIIS, Py-BOBYQA, CMA-ES,
  ...), plus the starting basis's own citation routes restricted to the
  elements the record carries. The citation registry cites every key a store
  record names, without an element filter, so copying the starting record's
  routes unrestricted would make an optimised hydrogen basis cite the
  fifth-period pob papers.
* ``notes``: the objective summary as JSON (reference, systems, final value).

Exactness
---------
The record stores the optimiser's doubles, and the candidate files the
optimiser evaluated are written with shortest round-trip precision
(:func:`vibeqc.basis_crystal.emit_g94`), so ``BasisSet(mol, name)`` on the
stored record rebuilds bit for bit the basis the optimiser scored. A shell
with a CRYSTAL scale factor other than 1 is refused: the record stores the
optimiser's exponents as they are, while the emitters apply the factor
(exponent * SCAL**2, :func:`vibeqc.basis_crystal.crystal_exponent_factor`).
A parsed basis always has 1, because the parser applies the factor on the
way in, so only a hand-built shell can trip this.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from ..basis_crystal import CrystalAtomBasis
    from ..basis_store import StoredRecord
    from ..basis_toolkit.model import BasisSetData, CitationRoute

__all__ = [
    "OPTIMISER_UPSTREAM",
    "optimised_basis_record",
    "save_optimised_basis",
    "starting_basis_citations",
    "write_optimised_basis",
]

OPTIMISER_UPSTREAM = "vibe-qc basis_optimization"
"""``Upstream.name`` of the in-process optimiser in a record's provenance."""


def _bundled_stem(name: str) -> str:
    """The record stem a bundled basis name resolves to."""
    try:
        from ..basis_registry import canonical_basis_name

        stem = canonical_basis_name(name)
        if stem:
            return stem
    except Exception:  # noqa: BLE001 - the native resolver is optional here
        pass
    return str(name).strip().lower()


def starting_basis_citations(
    starting_basis: str, elements: Iterable[str]
) -> list["CitationRoute"]:
    """Citation routes of the bundled record ``starting_basis`` that cover ``elements``.

    A route without an element list covers every element and is kept as is;
    a route with one is kept restricted to the intersection, and dropped when
    the intersection is empty. Raises ``ValueError`` when ``starting_basis``
    is not a bundled record, because a record that cannot say what it was
    optimised from should not claim a source.
    """
    from ..basis_toolkit import CitationRoute, library

    stem = _bundled_stem(starting_basis)
    if stem not in set(library.stems()):
        raise ValueError(
            f"starting basis {starting_basis!r} is not a bundled record (looked for "
            f"{stem!r}); pass its publications with citation_keys= instead"
        )
    record = library.load_record(stem)
    wanted = set(elements)
    routes: list[CitationRoute] = []
    for route in (record.provenance.citations if record.provenance else []):
        if route.elements is None:
            routes.append(CitationRoute(keys=list(route.keys)))
            continue
        covered = sorted(wanted.intersection(route.elements))
        if covered:
            routes.append(CitationRoute(keys=list(route.keys), elements=covered))
    return routes


def _check_unit_scale(atoms: Mapping[str, "CrystalAtomBasis"]) -> None:
    for symbol, atom in atoms.items():
        for idx, shell in enumerate(atom.shells):
            if float(shell.scale_factor) != 1.0:
                raise ValueError(
                    f"{symbol} shell {idx} has CRYSTAL scale factor "
                    f"{shell.scale_factor!r}; an optimised basis is exported only "
                    "with unit scale factors (fold the factor into the exponents "
                    "before optimising)"
                )


def _atomic_number(atom: "CrystalAtomBasis") -> int:
    return int(getattr(atom, "Z", 0) or 0)


def optimised_basis_record(
    atoms: Mapping[str, "CrystalAtomBasis"],
    *,
    name: str,
    starting_basis: Optional[str] = None,
    method_citations: Sequence[str] = (),
    citation_keys: Sequence[str] = (),
    run_id: Optional[str] = None,
    description: str = "",
    objective: Optional[Mapping[str, Any]] = None,
) -> "BasisSetData":
    """A ``vibe-basis`` QVF-Basis record of the optimised ``atoms``.

    Parameters
    ----------
    atoms
        ``{symbol: CrystalAtomBasis}`` at the optimum, as the recipes return
        it (``optimized_atoms``) or as ``parametrisation.unpack(res.x)``.
    name
        The record's name, the name ``BasisSet`` will resolve.
    starting_basis
        The bundled basis the optimisation started from (``"pob-TZVP"``).
        Its citation routes for these elements are carried over.
    method_citations
        Citation-database keys of the optimiser (``res.citations``,
        ``BDIIS_CITATION_KEYS``, ...).
    citation_keys
        Further keys, e.g. the publications of a starting basis that is not
        bundled.
    run_id, description
        Identity and one-line description of the run.
    objective
        JSON-serialisable summary of what was optimised; stored as notes.
    """
    from .. import __version__ as vibeqc_version
    from ..basis_toolkit import (
        CitationRoute,
        Provenance,
        ProvenanceKind,
        Upstream,
        from_crystal_atom,
    )

    if not atoms:
        raise ValueError("no atoms to export")
    _check_unit_scale(atoms)

    record = None
    for symbol in sorted(atoms, key=lambda s: (_atomic_number(atoms[s]), s)):
        part = from_crystal_atom(atoms[symbol], name=name)
        if record is None:
            record = part
            continue
        record.elements.update(part.elements)
        if part.ecps:
            merged = dict(record.ecps or {})
            merged.update(part.ecps)
            record.ecps = merged
    assert record is not None
    record.name = str(name).strip().lower()
    record.references = []
    record.description = description or (
        f"Optimised in process by vibeqc.basis_optimization"
        + (f" from {starting_basis}" if starting_basis else "")
        + (f", run {run_id}" if run_id else "")
        + "."
    )

    upstream = [
        Upstream(name=OPTIMISER_UPSTREAM, version=str(vibeqc_version),
                 identifier=run_id, detail="in-process basis optimiser"),
    ]
    routes: list[CitationRoute] = []
    method_keys = [k for k in dict.fromkeys(method_citations) if k]
    if method_keys:
        routes.append(CitationRoute(keys=method_keys))
    if starting_basis:
        upstream.append(Upstream(name=str(starting_basis), detail="starting basis"))
        routes.extend(starting_basis_citations(starting_basis, record.elements))
    extra = [k for k in dict.fromkeys(citation_keys) if k]
    if extra:
        routes.append(CitationRoute(keys=extra))

    record.provenance = Provenance(
        origin="generated",
        kind=ProvenanceKind.VIBE_BASIS,
        retrieval_date=_dt.date.today().isoformat(),
        upstream=upstream,
        citations=routes,
        run_id=run_id,
        notes=(json.dumps(dict(objective), sort_keys=True, default=str)
               if objective is not None else None),
    )
    return record


def _default_run_id(name: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}-{stamp}"


def save_optimised_basis(
    atoms: Mapping[str, "CrystalAtomBasis"],
    *,
    name: str,
    starting_basis: Optional[str] = None,
    method_citations: Sequence[str] = (),
    citation_keys: Sequence[str] = (),
    run_id: Optional[str] = None,
    description: str = "",
    objective: Optional[Mapping[str, Any]] = None,
    root: Optional[Path | str] = None,
    replace: bool = False,
    activate: bool = True,
) -> "StoredRecord":
    """Import the optimised ``atoms`` into the user basis store as ``name``.

    Builds the record with :func:`optimised_basis_record` and imports it with
    :func:`vibeqc.basis_store.import_record` (kind ``vibe-basis``). ``root``
    selects a store other than the user's default (``$VIBEQC_BASIS_STORE``,
    then the XDG data directory); ``replace`` overwrites a record of the same
    name. The store refuses a bundled name.

    Every new vibe-qc process resolves the record by name. With ``activate``
    (default) the current process is pointed at the updated store as well,
    unless ``LIBINT_DATA_PATH`` is pinned by the user (layer 1 of the
    resolution order) or currently points at a temporary candidate library,
    in which case activation is skipped and the record resolves from the
    next process on.
    """
    from .. import basis_store

    run_id = run_id or _default_run_id(name)
    record = optimised_basis_record(
        atoms, name=name, starting_basis=starting_basis,
        method_citations=method_citations, citation_keys=citation_keys,
        run_id=run_id, description=description, objective=objective,
    )
    stored = basis_store.import_record(
        record, name=name, kind="vibe-basis", replace=replace, run_id=run_id,
        root=root,
    )
    if activate and _may_activate():
        basis_store.activate_user_store(root=root)
    return stored


def _may_activate() -> bool:
    """False while ``LIBINT_DATA_PATH`` points at a partial (candidate) library.

    A :class:`~vibeqc.basis_optimization.io.TempBasisLibrary` holds only the
    candidate files; recomposing the store over the bundle there would take
    the candidates out of libint's view for the rest of that context.
    """
    from .. import _basis_library_has_standard_payload

    current = os.environ.get("LIBINT_DATA_PATH")
    return not current or _basis_library_has_standard_payload(current)


def write_optimised_basis(
    atoms: Mapping[str, "CrystalAtomBasis"],
    directory: Path | str,
    *,
    name: str,
    starting_basis: Optional[str] = None,
    method_citations: Sequence[str] = (),
    citation_keys: Sequence[str] = (),
    run_id: Optional[str] = None,
    description: str = "",
    objective: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the optimised ``atoms`` as ``<directory>/<name>.qvf.json``.

    The file carries the same provenance :func:`save_optimised_basis` stores,
    so it can be shared and imported elsewhere with
    ``vibe-qc-basis import <name>.qvf.json``.
    """
    from ..basis_toolkit import library

    record = optimised_basis_record(
        atoms, name=name, starting_basis=starting_basis,
        method_citations=method_citations, citation_keys=citation_keys,
        run_id=run_id or _default_run_id(name), description=description,
        objective=objective,
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return library.save_record(record, directory / f"{record.name}{library.RECORD_SUFFIX}")
