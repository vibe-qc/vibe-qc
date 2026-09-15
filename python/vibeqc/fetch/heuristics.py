"""Heuristics for SCF defaults and basis recommendations.

Pure functions: input is a list of element symbols (and the atom count
or system type where relevant), output is the picked default. The
fetcher uses these to fill ``default_initial_guess`` / ``default_damping``
/ ``default_kmesh`` / ``recommended_basis`` / ``is_open_shell`` on the
emitted SPEC.

Source for the (initial-guess, damping) decision table:
``examples/regression/systems/periodic/*.py`` -- every ionic / deep-core
spec sets SAD + 0.85, every wide-gap covalent uses HCORE + 0.5. We
copy that pattern rather than reinvent it.

See ``docs/tutorial/external_data_fetcher.md`` Sec. 6 for rationale.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

# Elements that need SAD + heavier damping to dodge the
# HCORE-divergence diagnostic the v0.7 SCF driver raises. Any of:
#   alkali / alkaline-earth s-block (deep core),
#   group-13 (Al),
#   halogens with heavy core (Cl onwards -- F is light enough for HCORE),
#   3d / 4d / 5d transition metals (every Z 21-30, 39-48, 57+).
# F kept out deliberately: the LiH / NeF / HF regression specs run fine
# on HCORE+0.5. Mg / Na / Cl / Li / Al observed to need SAD+0.85.
_SAD_TRIGGER_ELEMENTS = frozenset({
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Tl",
    "Cl", "Br", "I",
    # 3d transition metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d (omitting La/Ac series -- out of v1 ECP coverage anyway)
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})


def _has_sad_trigger(symbols: Iterable[str]) -> bool:
    return any(s in _SAD_TRIGGER_ELEMENTS for s in symbols)


def pick_initial_guess(symbols: Iterable[str]) -> str:
    """``"SAD"`` for ionic / deep-core, ``"HCORE"`` otherwise.

    See Sec. 6.1 of the structure-fetcher handover.
    """
    return "SAD" if _has_sad_trigger(symbols) else "HCORE"


def pick_damping(symbols: Iterable[str]) -> float:
    """``0.85`` for ionic / deep-core, ``0.5`` otherwise.

    See Sec. 6.1 of the structure-fetcher handover.
    """
    return 0.85 if _has_sad_trigger(symbols) else 0.5


def pick_kmesh(n_atoms: int) -> Tuple[int, int, int]:
    """k-mesh seed by cell size.

    A *seed* for a convergence study, not a converged value.
    See Sec. 6.2 of the structure-fetcher handover; emit a
    ``# TODO: k-mesh convergence`` comment in the generated input
    referencing ``examples/input-k-mesh-convergence.py``.
    """
    if n_atoms <= 4:
        return (4, 4, 4)
    if n_atoms <= 20:
        return (2, 2, 2)
    return (1, 1, 1)


def pick_recommended_basis(
    *,
    symbols: Iterable[str],
    is_periodic: bool,
    n_atoms: int,
    quick: bool = False,
) -> str:
    """Recommended basis given system type. See Sec. 6.3.

    ``quick=True`` (smoke-test mode) returns ``"sto-3g"`` regardless.
    """
    if quick:
        return "sto-3g"
    if is_periodic:
        # All three periodic rows in Sec. 6.3 land on pob-tzvp. We don't
        # try to gate transition-metal ECPs here -- emit pob-tzvp and
        # rely on the user-visible "verify ECP coverage" note in the
        # generated input script.
        return "pob-tzvp"
    # Molecular: def2-svp regardless of size; the "bump to def2-tzvp"
    # advice for >= 10 atoms is a comment in the generated script, not
    # an automatic choice (we don't second-guess affordability).
    return "def2-svp"


def open_shell_default(
    *,
    is_periodic: bool,
    mp_is_magnetic: Optional[bool] = None,
    mp_total_magnetization: Optional[float] = None,
    n_electrons: Optional[int] = None,
) -> Tuple[bool, Optional[Tuple[float, ...]]]:
    """Open-shell heuristic -- see Sec. 6.4.

    Periodic systems: open-shell only when MP confirms magnetic AND
    ``total_magnetization > 0.1`` mu_B/cell. Otherwise default to closed
    shell. (Wrong-side default is a UKS guess for a diamagnetic
    insulator; better to err conservative.)

    Molecules: spin parity is computed elsewhere from electron count;
    this function returns ``(False, None)`` for the periodic-default
    path and is a no-op when called for a molecule (callers should use
    ``MoleculeSpec.charge`` + ``multiplicity`` directly). Returned tuple
    shape: ``(is_open_shell, magnetic_moments_or_None)``.
    """
    if not is_periodic:
        return (False, None)
    if mp_is_magnetic is True and (mp_total_magnetization or 0.0) > 0.1:
        # Without per-site spin info, return open_shell=True with no
        # initial moments -- caller may layer on a per-site guess from
        # MP if it has the data.
        return (True, None)
    return (False, None)


def molecular_multiplicity(n_electrons: int) -> int:
    """Default multiplicity for a molecule by electron parity.

    Even electron count -> singlet (multiplicity 1); odd -> doublet
    (multiplicity 2). Override at the CLI level when the ground state
    is genuinely high-spin (e.g. O₂ triplet).
    """
    return 1 if n_electrons % 2 == 0 else 2
