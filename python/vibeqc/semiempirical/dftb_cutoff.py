"""Default pair cutoff for the full k-point DFTB0/SCC-DFTB lattice sums.

The full k-point drivers truncate the Slater-Koster H0 and the AO overlap
S at one pair distance. The overlap is the analytic overlap of the minimal
STO-6G valence basis (#373), so it has no finite table range; dropping
pairs whose overlap is still sizeable leaves S(k) indefinite and the
drivers refuse the mesh (#565).

The default is derived from the basis itself: for every pair of elements
in the cell, the largest separation at which some AO overlap between the
two atoms still reaches ``DFTB_OVERLAP_TOLERANCE``, measured on a
two-atom molecule with the native DFTB basis. The cutoff is the largest
such reach, but never below the historical 15 bohr, so a cell that ran
before keeps its exact lattice domain. Measured at 1e-6: H-H 14, C-C 13.5,
Mg-O 20.5, H-Li 25, Li-Li 33, Ga-Ga 39 bohr.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np

# Historical fixed cutoff and the floor of the derived default.
DFTB_KPOINT_CUTOFF_FLOOR_BOHR = 15.0
# Inter-atomic AO overlap that a pair cutoff may drop (maintainer decision,
# 2026-10-24): about 8 bohr of margin over every measured S(k) refusal.
DFTB_OVERLAP_TOLERANCE = 1.0e-6

_SCAN_STEP_BOHR = 0.5
_SCAN_LIMIT_BOHR = 200.0
# The tail must stay below the tolerance this far past the last crossing.
_SCAN_CONFIRM_BOHR = 5.0


def _ao_count(params, z: int) -> int:
    return sum(2 * l + 1 for l in range(4) if params.sto_exponent(z, l) > 0.0)


def _max_pair_overlap(params, za: int, zb: int, distance: float) -> float:
    import vibeqc as vq
    from vibeqc._vibeqc_core import semiempirical as _se

    electrons = int(params.valence_electrons(za)) + int(params.valence_electrons(zb))
    # The overlap does not depend on the charge; keep the pair closed shell.
    charge = electrons % 2
    molecule = vq.Molecule(
        [vq.Atom(int(za), [0.0, 0.0, 0.0]), vq.Atom(int(zb), [0.0, 0.0, float(distance)])],
        charge,
        1,
    )
    overlap = np.asarray(_se.run_dftb0(molecule, params).overlap)
    n_a = _ao_count(params, za)
    if n_a + _ao_count(params, zb) != overlap.shape[0]:
        raise RuntimeError(
            f"DFTB overlap reach: basis of Z={za}, Z={zb} has "
            f"{overlap.shape[0]} AOs, not the expected {n_a + _ao_count(params, zb)}"
        )
    return float(np.abs(overlap[:n_a, n_a:]).max())


@lru_cache(maxsize=None)
def _pair_reach_cached(sha256: str, za: int, zb: int, tolerance: float) -> float:
    from vibeqc.semiempirical.parameters import default_parameters

    params = default_parameters()
    if str(params.content_sha256()) != sha256:
        raise RuntimeError("DFTB overlap reach: parameter set changed during the run")
    return _pair_reach(params, za, zb, tolerance)


def _pair_reach(params, za: int, zb: int, tolerance: float) -> float:
    last_above = 0.0
    below_since = None
    distance = _SCAN_STEP_BOHR
    while distance <= _SCAN_LIMIT_BOHR:
        if _max_pair_overlap(params, za, zb, distance) >= tolerance:
            last_above = distance
            below_since = None
        elif below_since is None:
            below_since = distance
        elif distance - below_since >= _SCAN_CONFIRM_BOHR:
            break
        distance += _SCAN_STEP_BOHR
    else:
        raise RuntimeError(
            f"DFTB overlap reach: Z={za}-Z={zb} overlap still reaches "
            f"{tolerance:g} at {_SCAN_LIMIT_BOHR} bohr"
        )
    lo, hi = last_above, last_above + _SCAN_STEP_BOHR
    while hi - lo > 1.0e-3:
        mid = 0.5 * (lo + hi)
        if _max_pair_overlap(params, za, zb, mid) >= tolerance:
            lo = mid
        else:
            hi = mid
    return float(np.ceil(hi * 10.0) / 10.0)


def dftb_pair_overlap_reach_bohr(
    za: int, zb: int, *, tolerance: float = DFTB_OVERLAP_TOLERANCE,
) -> float:
    """Largest Z_a-Z_b separation (bohr) at which an AO overlap reaches *tolerance*.

    Measured with the default DFTB parameter set's native basis and cached
    per parameter content and tolerance; rounded up to 0.1 bohr.
    """
    from vibeqc.semiempirical.parameters import default_parameters

    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must lie in (0, 1); got {tolerance!r}")
    params = default_parameters()
    za, zb = sorted((int(za), int(zb)))
    for z in (za, zb):
        if not params.has_element(z):
            raise ValueError(f"the DFTB parameter set has no element Z={z}")
    return _pair_reach_cached(str(params.content_sha256()), za, zb, float(tolerance))


def default_dftb_kpoint_cutoff_bohr(
    system, *, tolerance: float = DFTB_OVERLAP_TOLERANCE,
) -> float:
    """Default full k-point DFTB pair cutoff (bohr) for *system*.

    ``max(15, reach)`` over every unordered element pair present, including
    an element with its own periodic images.
    """
    elements = sorted({int(atom.Z) for atom in system.unit_cell})
    reach = max(
        dftb_pair_overlap_reach_bohr(za, zb, tolerance=tolerance)
        for za, zb in combinations_with_replacement(elements, 2)
    )
    return max(DFTB_KPOINT_CUTOFF_FLOOR_BOHR, reach)
