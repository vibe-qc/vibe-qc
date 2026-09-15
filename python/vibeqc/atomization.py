"""General atomization energies for mean-field methods.

The atomization energy is ``S_A E(atom A) - E(molecule)`` (positive = bound),
with the free-atom references computed at the *same* level of theory as the
molecule.  This is method-agnostic infrastructure: any mean-field reference
(RHF / UHF / RKS / UKS) gets atomization for free once the free-atom ground
states are known.  Free-atom energies are cached per ``(Z, method, basis,
functional, grid)`` so a job pays at most one SCF per distinct element.

Free-atom ground-state spin multiplicities (2S+1, neutral atoms, Hund's rules)
are tabulated for the main-group elements H-Kr where the ground state is
unambiguous and a plain UHF SCF converges to it.  The 3d transition metals
(Sc-Zn) are intentionally omitted: their near-degenerate ground configurations
make a black-box atomic UHF unreliable, so atomization for systems containing
them raises rather than returning a wrong reference.

Semiempirical engines (MSINDO) carry their own atomic reference (``ateng``) and
report a binding energy directly; this module is for the Gaussian-basis
mean-field methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Neutral-atom ground-state spin multiplicity (2S+1), Hund's rules.
# H-Kr main group (+ K, Ca); 3d transition metals deliberately excluded.
_GROUND_STATE_MULTIPLICITY: Dict[int, int] = {
    1: 2, 2: 1,                                              # H He
    3: 2, 4: 1, 5: 2, 6: 3, 7: 4, 8: 3, 9: 2, 10: 1,        # Li-Ne
    11: 2, 12: 1, 13: 2, 14: 3, 15: 4, 16: 3, 17: 2, 18: 1, # Na-Ar
    19: 2, 20: 1,                                            # K Ca
    31: 2, 32: 3, 33: 4, 34: 3, 35: 2, 36: 1,               # Ga-Kr
}

# kcal/mol per Hartree (CODATA-consistent with the rest of vibe-qc).
_HARTREE_TO_KCAL = 627.509474063


@dataclass(frozen=True)
class AtomizationResult:
    """Atomization energy of a molecule (energies in Hartree)."""

    e_molecule: float
    e_atoms_sum: float
    atomization: float           # S E_atom - E_mol  (> 0 for a bound molecule)
    atomization_per_atom: float
    n_atoms: int
    atomic_energies: Dict[int, float]   # Z -> free-atom ground-state energy
    method: str
    basis: str
    functional: Optional[str] = None

    @property
    def atomization_kcal(self) -> float:
        return self.atomization * _HARTREE_TO_KCAL


# Module-level free-atom energy cache, keyed by (Z, method, basis, functional, grid).
_ATOM_ENERGY_CACHE: Dict[tuple, float] = {}


def supported_elements() -> frozenset:
    """Atomic numbers with a tabulated free-atom ground state."""
    return frozenset(_GROUND_STATE_MULTIPLICITY)


def atomic_ground_state_energy(
    Z: int, method: str, basis: str, *, functional: Optional[str] = None,
    grid_level: str = "orca-defgrid3", grid_options=None,
) -> float:
    """Free-atom ground-state energy (Hartree) at the given level of theory.

    ``method`` in {rhf, uhf, rks, uks, rohf, roks}; closed-shell atoms
    (singlet) use the restricted solver, open-shell atoms the unrestricted
    one.  ``rohf`` / ``roks`` use the spin-restricted open-shell solver for
    *both* (it reduces to RHF/RKS at a closed shell), giving spin-pure
    atomic references.  Cached per ``(Z, method, basis, functional, grid)``.
    """
    method = method.lower()
    if method not in ("rhf", "uhf", "rks", "uks", "rohf", "roks"):
        raise NotImplementedError(
            f"atomization references are defined for mean-field methods "
            f"(rhf/uhf/rks/uks/rohf/roks); got {method!r}.")
    if Z not in _GROUND_STATE_MULTIPLICITY:
        raise NotImplementedError(
            f"no tabulated free-atom ground state for Z={Z} "
            f"(supported: {sorted(_GROUND_STATE_MULTIPLICITY)}).")
    grid_key = None
    if method in ("rks", "uks", "roks"):
        from ._vibeqc_core import GridOptions
        from .runner import _apply_grid_level, _GRID_OPTION_FIELDS

        # An explicit grid is the molecule's operative grid, even when it
        # happens to equal legacy construction defaults.
        if grid_options is None:
            grid_options = GridOptions()
            _apply_grid_level(grid_options, grid_level)
        values = (getattr(grid_options, f) for f in _GRID_OPTION_FIELDS)
        grid_key = tuple(tuple(v) if isinstance(v, list) else v for v in values)
    key = (Z, method, basis.lower(), (functional or "").lower(), grid_key)
    cached = _ATOM_ENERGY_CACHE.get(key)
    if cached is not None:
        return cached

    from ._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions, RKSOptions,
                               UHFOptions, UKSOptions, run_rhf, run_rks,
                               run_uhf, run_uks)

    mult = _GROUND_STATE_MULTIPLICITY[Z]
    atom = Molecule([Atom(Z, [0.0, 0.0, 0.0])], 0, mult)
    bas = BasisSet(atom, basis)
    # Closed-shell singlet -> restricted; everything else -> unrestricted.
    restricted = mult == 1
    if method == "rohf":
        # Spin-pure restricted-open-shell reference (reduces to RHF at a
        # closed shell), so no restricted/unrestricted branch is needed.
        from .rohf import ROHFOptions, run_rohf

        res = run_rohf(atom, bas, ROHFOptions())
    elif method == "roks":
        from .roks import ROKSOptions, run_roks

        res = run_roks(
            atom, bas, ROKSOptions(grid=grid_options), functional=functional or "lda"
        )
    elif method in ("rhf", "uhf"):
        if restricted:
            res = run_rhf(atom, bas, RHFOptions())
        else:
            res = run_uhf(atom, bas, UHFOptions())
    else:  # rks / uks
        if restricted:
            opts = RKSOptions()
            opts.functional = functional or "lda"
            opts.grid = grid_options
            res = run_rks(atom, bas, opts)
        else:
            opts = UKSOptions()
            opts.functional = functional or "lda"
            opts.grid = grid_options
            res = run_uks(atom, bas, opts)
    if not bool(getattr(res, "converged", False)):
        raise RuntimeError(
            f"free-atom SCF did not converge for Z={Z} (mult={mult}, "
            f"{method}/{basis}); atomization reference unavailable.")
    e = float(res.energy)
    _ATOM_ENERGY_CACHE[key] = e
    return e


def atomization_energy(
    molecule, e_molecule: float, method: str, basis: str, *,
    functional: Optional[str] = None,
    grid_level: str = "orca-defgrid3", grid_options=None,
) -> AtomizationResult:
    """Atomization energy ``S E(atom) - E(molecule)`` at the molecule's level.

    Raises :class:`NotImplementedError` if any element lacks a tabulated
    free-atom ground state (so a system is never given a silently-wrong
    reference).  Free-atom SCFs are cached across calls. For DFT, pass the
    molecule's ``grid_options`` or ``grid_level`` (default ``orca-defgrid3``)
    so the molecule and its atomic references use the same numerical grid.
    """
    zs = [int(a.Z) for a in molecule.atoms]
    unsupported = sorted({z for z in zs if z not in _GROUND_STATE_MULTIPLICITY})
    if unsupported:
        raise NotImplementedError(
            f"atomization needs free-atom ground states for all elements; "
            f"missing Z={unsupported} (3d transition metals are excluded -- "
            f"their atomic ground state is not a black-box UHF target).")
    atomic_energies: Dict[int, float] = {}
    for z in set(zs):
        atomic_energies[z] = atomic_ground_state_energy(
            z, method, basis, functional=functional,
            grid_level=grid_level, grid_options=grid_options)
    e_atoms_sum = sum(atomic_energies[z] for z in zs)
    atomization = e_atoms_sum - e_molecule
    n = len(zs)
    return AtomizationResult(
        e_molecule=float(e_molecule),
        e_atoms_sum=float(e_atoms_sum),
        atomization=float(atomization),
        atomization_per_atom=float(atomization / n) if n else 0.0,
        n_atoms=n,
        atomic_energies=atomic_energies,
        method=method.lower(),
        basis=basis,
        functional=functional,
    )


__all__ = [
    "AtomizationResult",
    "atomization_energy",
    "atomic_ground_state_energy",
    "supported_elements",
]
