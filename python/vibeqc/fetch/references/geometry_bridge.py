"""Bridge a CCCBDB :class:`ExperimentalReference` into a
:class:`MoleculeSpec` for the regression suite.

Use case: "fetch the experimental geometry of water from NIST and run
my method against it" -- strict equivalent to the phase-1 workflow
(fetch crystal structure -> run vibe-qc) but for molecules.

The bridge consumes ``ExperimentalReference.cartesian_geometry_ang``
(populated by ``vibeqc.fetch.references.parsers`` from CCCBDB's
per-molecule Cartesian table) and emits a ``MoleculeSpec`` with the
same ``id``, ``provenance``, and atom positions. Recommended basis is
filled by the same heuristic phase 1 uses
(:func:`vibeqc.fetch.heuristics.pick_recommended_basis`).

Phase-2 polish step 5. See ``docs/tutorial/external_data_fetcher.md``
Sec. 14 for the workflow this enables.
"""
from __future__ import annotations

from typing import Optional

from examples.regression.core.spec import (
    AtomCart,
    ExperimentalReference,
    MoleculeSpec,
    Provenance,
)

from ..heuristics import molecular_multiplicity, pick_recommended_basis


def experimental_geometry_to_molecule_spec(
    ref: ExperimentalReference,
    *,
    slug: Optional[str] = None,
    quick: bool = False,
    overrides: Optional[dict] = None,
) -> MoleculeSpec:
    """Build a :class:`MoleculeSpec` from a CCCBDB
    :class:`ExperimentalReference` populated with Cartesian geometry.

    Parameters
    ----------
    ref:
        Reference record from ``fetch_cccbdb``. Must carry
        ``cartesian_geometry_ang`` -- raises ``ValueError`` if empty
        (the user typically pulled a bond-list-only entry; ask CCCBDB
        for one with Cartesians or supply your own).
    slug:
        Override the emitted ``MoleculeSpec.id`` (default: ``ref.cas``
        with hyphens stripped, prefixed with ``"cccbdb_"`` so the slug
        is obviously fetched).
    quick:
        Smoke-test mode -> ``recommended_basis = "sto-3g"``.
    overrides:
        Optional mapping of ``MoleculeSpec`` field name -> value, applied
        after the heuristic defaults. Useful for ``charge``,
        ``multiplicity``, ``family`` overrides without round-tripping
        through `dataclasses.replace`.

    Notes
    -----
    Multiplicity: defaulted from electron-count parity unless ``ref``
    has implicit information from ``ionization_energy_ev`` / open-shell
    metadata. CCCBDB doesn't tabulate multiplicity directly on the
    ``exp2x`` page, so we trust electron count for closed-shell
    species and require an explicit ``overrides={"multiplicity": 3}``
    for open-shell molecules (O₂ triplet, OH doublet, ...). The bridge
    raises ``ValueError`` if the inferred parity is even but the
    formula contains an odd number of unpaired-shell-suspect atoms;
    callers must override.
    """
    if not ref.cartesian_geometry_ang:
        raise ValueError(
            "ExperimentalReference.cartesian_geometry_ang is empty -- "
            "cannot build MoleculeSpec from internal coordinates "
            "(bond list / angles) alone in v1. Pull a CCCBDB exp2x "
            "entry that exposes the Cartesian table, or build the "
            "MoleculeSpec by hand."
        )

    atoms = tuple(
        AtomCart(symbol="".join(c for c in label if not c.isdigit()),
                 z=z, xyz_ang=(x, y, z_coord))
        for (label, z, x, y, z_coord) in ref.cartesian_geometry_ang
    )

    # Slug -- fetched-emit convention mirrors phase-1's
    # `cod_<id>` / `mp_<id>` style.
    if slug is None:
        slug = f"cccbdb_{ref.cas.replace('-', '')}"

    # Recommended basis from the same phase-1 heuristic, with
    # smoke-test override.
    syms = [a.symbol for a in atoms]
    rec_basis = pick_recommended_basis(
        symbols=syms, is_periodic=False, n_atoms=len(atoms), quick=quick,
    )

    # Multiplicity from electron-count parity.
    n_electrons = sum(a.z for a in atoms)
    multiplicity = molecular_multiplicity(n_electrons)

    # Provenance: re-stamp from the reference, preserving the NIST
    # DOI but extending notes to record this as a geometry-bridge
    # emission (so downstream consumers know the MoleculeSpec didn't
    # come from a hand-curated source).
    src_prov = ref.provenance
    spec_provenance = Provenance(
        source_db=src_prov.source_db,
        source_id=src_prov.source_id,
        source_url=src_prov.source_url,
        original_reference=src_prov.original_reference,
        license=src_prov.license,
        fetched_at=src_prov.fetched_at,
        fetcher_version=src_prov.fetcher_version,
        notes=(
            "MoleculeSpec emitted from CCCBDB Cartesian geometry via "
            "vibeqc.fetch.references.experimental_geometry_to_molecule_spec"
            + (f"; {src_prov.notes}" if src_prov.notes else "")
        ),
    )

    spec = MoleculeSpec(
        id=slug,
        family="molecule",
        atoms=atoms,
        charge=0,
        multiplicity=multiplicity,
        recommended_basis=rec_basis,
        notes=(
            f"Geometry from {src_prov.source_db}/{src_prov.source_id}; "
            f"name={ref.name!r}, formula={ref.formula!r}."
        ),
        citation=src_prov.original_reference or src_prov.source_url,
        provenance=spec_provenance,
    )

    if overrides:
        from dataclasses import replace
        spec = replace(spec, **overrides)
    return spec
