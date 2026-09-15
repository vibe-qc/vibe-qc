"""Published conventions shared by molecular correlated methods.

The frozen-core policy is the element table published as ORCA 6.1
Table 2.69, converted from frozen electrons to doubly occupied spatial
orbitals.  vibe-qc applies the resulting *count* to the lowest-energy
occupied orbitals.  ORCA also uses atomic-orbital character to repair unusual
canonical orderings; that character-based reorder is not implemented here.
"""

from __future__ import annotations

from numbers import Integral

from ._vibeqc_core import (
    _published_frozen_core_orbitals_for_atomic_number,
)

PUBLISHED_FROZEN_CORE_CONVENTION = "orca-6.1-table-2.69-count-only"
_PUBLISHED_FROZEN_CORE_CITATION_KEY = "orca_manual_frozen_core_6_1_1"

_PUBLISHED_SELECTORS = frozenset(
    {
        "chemical",
        "chemical-core",
        "default",
        "published",
        PUBLISHED_FROZEN_CORE_CONVENTION,
    }
)
_ALL_ELECTRON_SELECTORS = frozenset(
    {
        "all-electron",
        "all-electrons",
        "all",
        "none",
        "off",
    }
)


def _normalise_frozen_core_selector(requested: str) -> str:
    normalized = requested.strip().lower().replace("_", "-")
    return "-".join(normalized.split())


def _uses_published_frozen_core_selector(requested=None) -> bool:
    """Whether ``requested`` explicitly or implicitly selects the table.

    This predicate is intentionally selector-based rather than count-based:
    an explicit integer that happens to equal the published count remains an
    explicit user recipe and must not be cited or labelled as the ORCA-table
    convention.
    """

    if requested is None or requested is True:
        return True
    if isinstance(requested, str):
        return _normalise_frozen_core_selector(requested) in _PUBLISHED_SELECTORS
    return False


def published_frozen_core_orbital_count(molecule, reference=None) -> int:
    """Return the published count of frozen spatial orbitals for ``molecule``.

    The sum is element based.  A conventional ``Z=0`` ghost centre
    contributes zero.  With ``reference`` (an SCF result or options struct
    carrying an ECP on either route), the electrons the ECP already removed
    on each atom are subtracted from that atom's published core first:
    ``max(0, fc_A - n_core,A) / 2`` orbitals per atom, so an ECP that
    replaces the whole published core freezes nothing and a small-core ECP
    leaves the semicore shells frozen.  The count does not inspect orbital
    character, charge, or canonical orbital ordering; correlated routes
    validate that the resolved count fits their occupied space.
    """

    try:
        atoms = list(molecule.atoms)
    except AttributeError as exc:
        raise TypeError("molecule must expose an atoms collection") from exc
    if reference is None:
        return sum(
            int(_published_frozen_core_orbitals_for_atomic_number(int(atom.Z)))
            for atom in atoms
        )
    from .ecp_metadata import ecp_core_electrons_per_atom

    cores = ecp_core_electrons_per_atom(molecule, reference)
    total = 0
    for atom, core in zip(atoms, cores):
        fc_electrons = 2 * int(
            _published_frozen_core_orbitals_for_atomic_number(int(atom.Z))
        )
        total += max(0, fc_electrons - int(core)) // 2
    return total


def resolve_frozen_core_count(molecule, requested=None, *, reference=None) -> int:
    """Resolve a public frozen-core selector to a spatial-orbital count.

    ``None``, ``True``, ``"published"``, and ``"chemical"`` select the
    published element table. ``False`` and ``"all-electron"`` select zero.
    A non-negative integer is an explicit count.  Other values fail closed.
    ``reference`` is the SCF result (or options) the correlated method will
    consume; it makes the published count ECP-aware
    (:func:`published_frozen_core_orbital_count`).
    """

    if requested is None or requested is True:
        return published_frozen_core_orbital_count(molecule, reference)
    if requested is False:
        return 0
    if isinstance(requested, Integral):
        count = int(requested)
        if count < 0:
            raise ValueError("frozen-core orbital count must be non-negative")
        return count
    if isinstance(requested, str):
        normalized = _normalise_frozen_core_selector(requested)
        if normalized in _PUBLISHED_SELECTORS:
            return published_frozen_core_orbital_count(molecule, reference)
        if normalized in _ALL_ELECTRON_SELECTORS:
            return 0
    raise ValueError(
        "frozen_core must be None, a non-negative integer, a boolean, "
        "'published'/'chemical', or 'all-electron'"
    )


def resolve_frozen_core(molecule, requested=None, *, reference=None) -> int:
    """Compatibility alias for :func:`resolve_frozen_core_count`."""

    return resolve_frozen_core_count(molecule, requested, reference=reference)


def effective_electron_count(molecule, reference) -> int:
    """Return the variational electron count represented by ``reference``.

    Molecular ECP SCF results carry the number of replaced physical core
    electrons as authoritative provenance. Consumers that derive occupations
    from :meth:`Molecule.n_electrons` must subtract that count; otherwise they
    label core-removed virtual orbitals as occupied. Invalid or inconsistent
    provenance fails closed instead of producing a plausible-looking count.
    """
    physical = int(molecule.n_electrons())
    try:
        ncore = int(getattr(reference, "ecp_total_ncore", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid SCF ecp_total_ncore provenance") from exc
    if ncore < 0:
        raise ValueError("invalid negative ecp_total_ncore provenance")
    if ncore > physical:
        raise ValueError(
            "SCF ecp_total_ncore exceeds the molecule's physical electron "
            f"count ({ncore} > {physical})"
        )
    return physical - ncore


__all__ = [
    "PUBLISHED_FROZEN_CORE_CONVENTION",
    "effective_electron_count",
    "published_frozen_core_orbital_count",
    "resolve_frozen_core",
    "resolve_frozen_core_count",
]
