"""Cross-process SCC branch determinism on degenerate homonuclear chains (#424).

The rp246 wave (build 127682e39) produced 9 case-groups -- all homonuclear B
(Z5) or Ga (Z31) zigzag chains, all dim=1 / cutoff=12 / T=0 -- that flipped
between converged=True at n_iter=1 and converged=False at n_iter=500 across
fleet members running identical inputs on one shared binary.  Root cause
(issue #424 diagnosis, 2026-08-27):

* Both chain sites are glide-equivalent, so the SCC seed dq=0 is an exact
  fixed point of the charge map, and hard T=0 aufbau meets a frontier pair
  that glide symmetry forces degenerate at the zone edge.  The pre-#316
  asymmetric lattice-image selection split that pair by ~3.2e-7 Ha -- far
  above the roundoff scale (256*eps) at which the zero-temperature ensemble
  in compute_closed_shell_kpoint_occupations equalizes occupations, so the
  boundary cut a near-degenerate manifold instead of sharing it.
* Near-degeneracy makes the frontier eigenvectors ill-conditioned: an
  ulp-scale Hamiltonian difference eps rotates them by ~eps/3.2e-7, which
  amplified per-host last-ulp differences (one binary, runtime-dispatched
  libm/SIMD kernel paths across the different compute-node families)
  into iteration-1 Mulliken residuals of 1e-13..>1e-10.
* run_scc_dftb_kpoints declares convergence when that first residual is
  below conv_tol_charge (1e-10 in the wave).  The symmetric fixed point is
  dynamically UNSTABLE under the T=0 occupancy discontinuity: any host whose
  residual landed above tolerance stepped off and entered a permanent
  occupancy-swap limit cycle (n_iter=500, |dq| ~ 7e-6..1.5e-3), never
  converging.  Hence the observed binary n_iter=1 vs n_iter=500 branch,
  selected per (case, host) by a rounding race.

The #316 pair-distance image selection (660f72862) restored the symmetric
image sets: the frontier pairs are exactly degenerate again, the T->0
ensemble handles any cut manifold, and the iteration-1 residual of every
wave-flip case drops to the 1e-17..1e-15 rounding floor.

This regression pins that margin.  It runs every wave-flip case at
conv_tol_charge=1e-12 -- 100x TIGHTER than the wave setting -- and requires
the immediate symmetric fixed point.  On the wave build these assertions
fail (at tol<=1e-11 every case below degrades to the 500-iteration limit
cycle); after #316 they hold with >=3 orders of cross-host headroom.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.parameters import default_parameters

# The 9 rp246 flip groups span these 6 distinct (Z, a, mesh) inputs; all are
# dim=1, cutoff=12, T=0.  Geometry is the rp246 two-site zigzag cell:
# sites (0, 0, 0) and (a/2, 0.35, 0) bohr, vacuum 30 bohr, which makes the
# two homonuclear sites glide-equivalent (equal bond lengths both ways).
_WAVE_FLIP_CASES = (
    (31, 3.8, 6),
    (31, 5.2, 2),
    (31, 6.0, 32),
    (5, 4.2, 2),
    (5, 4.6, 48),
    (5, 7.0, 48),
)
_CUTOFF = 12.0


def _homonuclear_chain(z: int, a: float) -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([a, 30.0, 30.0]),
        [
            Atom(z, [0.0, 0.0, 0.0]),
            Atom(z, [a / 2.0, 0.35, 0.0]),
        ],
        0,
        1,
    )


def _run_case(z: int, a: float, mesh: int, conv_tol_charge: float):
    system = _homonuclear_chain(z, a)
    kmesh = monkhorst_pack(system, (mesh, 1, 1))
    scc_options = _se.SCCOptions()
    scc_options.max_iter = 500
    scc_options.conv_tol_charge = conv_tol_charge
    occupation_options = _se.KPointOccupationOptions()
    occupation_options.smearing_temperature = 0.0
    return _se.run_scc_dftb_kpoints(
        system,
        default_parameters(),
        kmesh,
        scc_options,
        _CUTOFF,
        occupation_options,
    )


@pytest.mark.parametrize(("z", "a", "mesh"), _WAVE_FLIP_CASES)
def test_wave_flip_case_holds_symmetric_fixed_point_with_margin(z, a, mesh):
    # 100x tighter than the wave's 1e-10: the branch selection was a race
    # between a host-dependent rounding residual and the tolerance, so the
    # regression demands the fixed point with margin, not at the old edge.
    result = _run_case(z, a, mesh, conv_tol_charge=1.0e-12)
    assert result.converged
    # The glide-symmetric dq=0 seed must be recognized as the fixed point
    # immediately; reaching iteration 2 means the first Mulliken residual
    # was above 1e-12, i.e. back within one ulp-lottery of the wave flips.
    assert result.n_iter == 1
    charges = np.asarray(result.charges)
    assert charges.shape == (2,)
    # Rounding floor, not physics: post-#316 both sites are symmetric to
    # ~1e-15; 1e-12 leaves the same >=3 orders of slack that separates the
    # fixed branch from the wave build (residuals 6e-12..>1e-10 there).
    assert float(np.abs(charges).max()) < 1.0e-12


def test_flagship_case_is_process_deterministic():
    # Same input, N fresh interpreters: byte-identical branch and energy.
    # Guards the per-process half of #424 (state leakage between cases in a
    # worker was ruled out; keep it ruled out).  The cross-host half is
    # covered by the margin test above, which any single host can falsify.
    z, a, mesh = _WAVE_FLIP_CASES[0]
    script = (
        "import numpy as np\n"
        "from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack\n"
        "from vibeqc._vibeqc_core import semiempirical as _se\n"
        "from vibeqc.semiempirical.parameters import default_parameters\n"
        f"system = PeriodicSystem(1, np.diag([{a!r}, 30.0, 30.0]),"
        f" [Atom({z}, [0.0, 0.0, 0.0]), Atom({z}, [{a!r} / 2.0, 0.35, 0.0])],"
        " 0, 1)\n"
        f"kmesh = monkhorst_pack(system, ({mesh}, 1, 1))\n"
        "opts = _se.SCCOptions()\n"
        "opts.max_iter = 500\n"
        "opts.conv_tol_charge = 1.0e-10\n"
        "occ = _se.KPointOccupationOptions()\n"
        "occ.smearing_temperature = 0.0\n"
        "res = _se.run_scc_dftb_kpoints(system, default_parameters(), kmesh,"
        f" opts, {_CUTOFF!r}, occ)\n"
        "print(res.converged, res.n_iter, repr(res.energy),"
        " [repr(float(c)) for c in res.charges])\n"
    )
    outcomes = set()
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        outcomes.add(proc.stdout)
    assert len(outcomes) == 1, outcomes
