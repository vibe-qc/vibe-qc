"""Source identity of the neutral fitted-torus Hamiltonian (#519, RG-2).

The two producers of the neutral fitted torus, the real-Gamma supercell SCF
(:mod:`vibeqc.periodic.ccm.direct`) and the neutral-Bloch multi-k SCF
(:mod:`vibeqc.periodic.ccm.ri`), evaluate one specified Hamiltonian in two
representations (Paper 1, Theorem 1). Until now a Theorem-1 pair was audited
field by field: the exchange-``q=0`` label, the RSGDF tail, the fit source.
Each of those was a wrong-number class on its own (IID 307, #378, #466), and
the list grows with every new control (the cell-list cutoff, the XC image
reach of #465/#512, the auxiliary basis).

This module records the whole operator identity once, as the
:class:`~vibeqc.fock_state.SourceProvenance` of #472, on every result of
both producers: the geometry and torus, the orbital and auxiliary bases, the
fit (engine, base reciprocal cutoff, tail, cell-list cutoff, screening), the
exchange seam convention and, for KS, the functional and the executed XC
image reach. The record is **representation-independent**: storage mode,
builder batching, solver settings and the SCF state stay on ``cderi_info``,
the result backend and the convention records. Two results with equal
``neutral_source.identity`` evaluated one Hamiltonian; a comparison across
unequal identities is refused by :func:`assert_matched_neutral_source`, which
names the differing fields instead of averaging them.

A caller-supplied cderi (``fit_source="injected"``) carries no fit record, so
its identity never equals a built run's: the builder, not the array, knows
what it fitted. The fit-source boundary of the production track (RG-7) gives
the array a record of its own.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ...fock_state import SourceProvenance

__all__ = [
    "NEUTRAL_TORUS_PROVIDER",
    "NEUTRAL_TORUS_SCHEMA",
    "assert_matched_neutral_source",
    "has_recordable_system",
    "neutral_source_of",
    "neutral_torus_source",
]

#: ``SourceProvenance.provider`` of every neutral fitted-torus record.
NEUTRAL_TORUS_PROVIDER = "ccm-neutral-fitted-torus"
#: Schema tag inside the details; bump when a field is added or renamed, so
#: identities from different schema versions never compare equal by accident.
NEUTRAL_TORUS_SCHEMA = "vibeqc.ccm-neutral-fitted-torus.v1"


def _executed_tail(tail_ke_cutoff, ke_cutoff):
    """The RSGDF tail a builder executes, as the identity records it: none
    unless it exceeds the base cutoff (the builders' rule), so an explicit
    ``0`` and an absent tail name one Hamiltonian (#519)."""
    tail = _finite(tail_ke_cutoff, "tail_ke_cutoff")
    base = _finite(ke_cutoff, "ke_cutoff")
    if tail is None or (base is not None and tail <= base) or tail <= 0.0:
        return None
    return tail


def _finite(value, name):
    if value is None:
        return None
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"neutral_torus_source: {name} must be finite, got {value!r}")
    return scalar


def has_recordable_system(ccm) -> bool:
    """Whether ``ccm`` carries a real periodic system the record can read.

    The direct drivers are exercised on synthetic stand-in systems in their
    unit tests (a namespace with ``dim`` and ``charge`` only); those runs
    carry no source record rather than failing the driver.
    """
    unit = getattr(ccm, "unit_system", None)
    return (all(hasattr(unit, name) for name in
                ("unit_cell_molecule", "lattice", "dim", "charge"))
            and all(hasattr(ccm, name) for name in
                    ("basis_name", "nrep", "n_cells")))


def _geometry(ccm):
    if not has_recordable_system(ccm):
        raise TypeError(
            "neutral_torus_source: the cluster's unit_system must be a "
            "PeriodicSystem (unit_cell_molecule, lattice, dim, charge)")
    unit = ccm.unit_system
    mol = unit.unit_cell_molecule()
    lattice = np.asarray(unit.lattice, dtype=float)
    return {
        "lattice": [[round(float(x), 12) for x in row] for row in lattice],
        "dimensionality": int(unit.dim),
        "nuclei": [[int(a.Z), [round(float(x), 12) for x in a.xyz]]
                   for a in mol.atoms],
        "charge": int(unit.charge),
    }


def neutral_torus_source(
    ccm, *, fit_engine, fit_source, aux_basis, ke_cutoff, tail_ke_cutoff,
    lattice_cutoff_bohr, exchange_q0, exchange_q0_applicability,
    omega_screen=0.0, screened_exchange=None, functional=None,
    xc_lattice_cutoff_bohr=None, grid_image_radius_bohr=None,
    caller_shape=None, one_electron=None, xc_source=None,
    xc_torus_complete=False,
) -> SourceProvenance:
    """The :class:`~vibeqc.fock_state.SourceProvenance` of one neutral torus.

    Parameters name what each producer reads off its own executed record:
    ``fit_engine`` the builder that fitted (``"native-classic-fft-shared-q"``
    for the fold and the multi-k classic engine, ``"supercell-fft"`` for the
    supercell fit, ``"caller"`` for an injected array), ``fit_source`` the
    ``"classic"`` / ``"sr"`` / ``"injected"`` label the results already
    carry, ``aux_basis`` the **resolved** auxiliary basis name, ``ke_cutoff``
    the base reciprocal cutoff (Ha), ``tail_ke_cutoff`` the executed tail
    (Ha, or ``None``), ``lattice_cutoff_bohr`` the fit's real-space cell-list
    cutoff. ``exchange_q0`` is recorded only when the seam is
    ``"active"``: a pure functional carries no seam, and both producers must
    say so the same way. ``functional`` and the XC reach enter for KS only.
    ``caller_shape`` distinguishes injected arrays from each other.
    ``one_electron`` is the record of a non-default one-electron source
    (RG-8b); the default classic fold adds no entry, so every identity
    recorded before the option existed is unchanged. ``xc_torus_complete``
    (RG-1) marks the complete torus XC density, the domain-converged XC term,
    in place of the pair graph truncated at the XC domain; it is recorded only
    when set, for the same reason.
    """
    applicability = str(exchange_q0_applicability or "")
    if applicability not in ("active", "inactive"):
        raise ValueError(
            "neutral_torus_source: exchange_q0_applicability must be 'active' "
            f"or 'inactive', got {exchange_q0_applicability!r}")
    details = {
        "schema": NEUTRAL_TORUS_SCHEMA,
        "geometry": _geometry(ccm),
        "torus": {"nrep": [int(n) for n in ccm.nrep],
                  "n_cells": int(ccm.n_cells)},
        "orbital_basis": str(ccm.basis_name),
        "fit": {
            "engine": str(fit_engine or "caller"),
            "source": str(fit_source or ""),
            "auxiliary_basis": None if aux_basis is None else str(aux_basis),
            "base_reciprocal_cutoff_ha": _finite(ke_cutoff, "ke_cutoff"),
            "tail_reciprocal_cutoff_ha": _executed_tail(tail_ke_cutoff, ke_cutoff),
            "lattice_cutoff_bohr": _finite(lattice_cutoff_bohr, "lattice_cutoff_bohr"),
            "omega_screen": _finite(omega_screen, "omega_screen") or 0.0,
            "caller_shape": (None if caller_shape is None
                             else [int(n) for n in caller_shape]),
        },
        "exchange": {
            "q0": str(exchange_q0 or "") if applicability == "active" else "",
            "applicability": applicability,
            "screened": (None if screened_exchange is None
                         else [_finite(x, "screened_exchange")
                               for x in screened_exchange]),
        },
        "xc": (None if functional is None else {
            "functional": str(functional).strip().lower(),
            "image_cutoff_bohr": _finite(xc_lattice_cutoff_bohr,
                                         "xc_lattice_cutoff_bohr"),
            "grid_image_radius_bohr": _finite(grid_image_radius_bohr,
                                              "grid_image_radius_bohr"),
        }),
    }
    if one_electron is not None:
        details["one_electron"] = dict(one_electron)
    if xc_source is not None:
        if details.get("xc") is None:
            raise ValueError(
                "neutral_torus_source: an XC source record needs a functional")
        details["xc"]["source"] = dict(xc_source)
    if xc_torus_complete:
        if details.get("xc") is None:
            raise ValueError(
                "neutral_torus_source: a complete torus XC density needs a "
                "functional")
        details["xc"]["density"] = "complete-torus"
    return SourceProvenance.capture(NEUTRAL_TORUS_PROVIDER, **details)


def neutral_source_of(result):
    """The neutral-source record of a producer result, or ``None``."""
    if isinstance(result, SourceProvenance):
        return result
    record = getattr(result, "neutral_source", None)
    if record is None:
        convention = (getattr(result, "real_gamma", None)
                      or getattr(result, "neutral_bloch", None))
        payload = getattr(convention, "neutral_source", None)
        if isinstance(payload, dict) and payload.get("provider"):
            return SourceProvenance(payload["provider"],
                                    json.dumps(payload.get("details", {})))
    return record


def _flatten(details, prefix=""):
    out = {}
    for key, value in details.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def assert_matched_neutral_source(items, *, context: str = "comparison") -> str:
    """Refuse a comparison of results that did not evaluate one Hamiltonian.

    ``items`` are producer results (direct, Bloch, or the runner-contract
    views carrying their convention record) or
    :class:`~vibeqc.fock_state.SourceProvenance` objects. Raises
    :class:`ValueError` when any item carries no record or when the
    identities differ, naming the differing fields. Returns the shared
    identity.
    """
    records = []
    for index, item in enumerate(items):
        record = neutral_source_of(item)
        if record is None:
            raise ValueError(
                f"{context}: item {index} carries no neutral-source record "
                "(neutral_source); it cannot be compared by identity")
        records.append(record)
    if not records:
        raise ValueError(f"{context}: nothing to compare")
    first = records[0]
    identities = {r.identity for r in records}
    if len(identities) == 1:
        return first.identity
    flat_first = _flatten(json.loads(first.details_json))
    differing = []
    for index, record in enumerate(records[1:], start=1):
        flat = _flatten(json.loads(record.details_json))
        for key in sorted(set(flat_first) | set(flat)):
            a, b = flat_first.get(key), flat.get(key)
            if a != b:
                differing.append(f"{key}: item 0 {a!r} vs item {index} {b!r}")
    raise ValueError(
        f"{context}: the results did not evaluate one neutral fitted-torus "
        "Hamiltonian; these are different operators, not a tolerance "
        "question. Differing fields: " + "; ".join(differing))
