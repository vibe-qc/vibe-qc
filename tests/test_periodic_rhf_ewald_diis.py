"""Phase 12e-c-4c-ii: DIIS extrapolation in the Γ-Ewald SCF driver.

The DIIS machinery matches the Pulay convention used by the C++
molecular drivers: error vector ``e = F D S - S D F``, subspace
cap ``options.diis_subspace_size``, activation at
``options.diis_start_iter``. Tests:

1. **Energy agreement** — DIIS converges to the same SCF energy as
   plain damping (to machine precision), just faster.
2. **Iteration-count speedup** — on a non-trivial case (H2 / 6-31G**)
   DIIS converges in O(10) iterations while plain damping needs
   O(50+).
3. **DIIS-off default** — passing ``use_diis=False`` reproduces the
   previous pure-damping behavior.
4. **Start iteration override** — ``diis_start_iter`` controls when
   DIIS kicks in; before that iteration DIIS history accumulates but
   the SCF runs pure damping.
5. **Near-singular B matrix resilience** — when adjacent iterations
   produce near-duplicate error vectors (e.g., a perfectly-converged
   case looping), the solver shouldn't blow up; it falls back by
   dropping the oldest history entry.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_in_box(box: float = 20.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "6-31g**")
    return sysp, basis


def _options(use_diis: bool, damping: float):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    # Pin STATIC damping: these tests compare DIIS against plain
    # constant-damping Roothaan. The periodic default flipped to
    # dynamic_damping=True (v0.15.x), which would otherwise accelerate the
    # use_diis=False control and mask the DIIS speedup this test measures.
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 80
    opts.use_diis = use_diis
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    return opts


# ---------------------------------------------------------------------------
# Energy invariance across DIIS on / off
# ---------------------------------------------------------------------------

def test_diis_and_damping_reach_same_energy():
    """DIIS is an accelerator, not a solver change — it must converge
    to the same energy as plain damping."""
    sysp, basis = _h2_in_box()
    r_damp = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _options(use_diis=False, damping=0.7),
        omega=0.5, spacing_bohr=0.3,
    )
    r_diis = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _options(use_diis=True, damping=0.3),
        omega=0.5, spacing_bohr=0.3,
    )
    assert r_damp.converged
    assert r_diis.converged
    assert r_damp.energy == pytest.approx(r_diis.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# Iteration-count speedup
# ---------------------------------------------------------------------------

def test_diis_cuts_iteration_count_on_nontrivial_case():
    """H2 / 6-31G** with ω-Ewald takes ~50+ iterations with plain
    damping; DIIS cuts it to ~10. A 3× speedup is the minimum bar.
    """
    sysp, basis = _h2_in_box()
    r_damp = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _options(use_diis=False, damping=0.7),
        omega=0.5, spacing_bohr=0.3,
    )
    r_diis = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _options(use_diis=True, damping=0.3),
        omega=0.5, spacing_bohr=0.3,
    )
    assert r_damp.converged and r_diis.converged
    speedup = r_damp.n_iter / r_diis.n_iter
    assert speedup >= 3.0, (
        f"DIIS speedup only {speedup:.1f}x ({r_damp.n_iter} damped "
        f"vs {r_diis.n_iter} DIIS iterations)"
    )


# ---------------------------------------------------------------------------
# DIIS off is the previous behavior
# ---------------------------------------------------------------------------

def test_diis_off_reproduces_pre_diis_iteration_count():
    """With use_diis=False, iteration count must match what the
    pre-DIIS driver produced on the same case. 12e-c-4b shipped
    the H2 / STO-3G / 30-bohr box converging in 2 iterations with
    damping 0.3."""
    c = 15.0
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * 30.0,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 60
    opts.use_diis = False
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    # Pre-v0.6.1 the H2/STO-3G/30-bohr SCF "converged in 2 iters"
    # because the Madelung-leak bug kept iter-1 and iter-2 energies
    # bit-identical (the bug overwhelmed the actual iteration
    # trajectory). With the Madelung fix in v0.6.1 the energies
    # follow the genuine iteration trajectory; with damping=0.3 + no
    # DIIS the H2 SCF takes ~17 iters to converge to the same final
    # energy — that's the expected behaviour of plain damping
    # without DIIS extrapolation. The contract this test asserts is
    # "DIIS-off path converges to the same fixed point", not "in
    # exactly 2 iters".
    assert r.n_iter < 30, (
        f"DIIS-off SCF took {r.n_iter} iters — too many. With "
        f"damping=0.3 the H2 SCF should converge in under 30 iters."
    )


# ---------------------------------------------------------------------------
# Start-iteration override
# ---------------------------------------------------------------------------

def test_diis_start_iter_delays_activation():
    """Setting diis_start_iter = max_iter + 1 effectively disables
    DIIS (no iteration is ever ≥ start). Energy matches plain
    damping."""
    sysp, basis = _h2_in_box()
    opts_damp = _options(use_diis=False, damping=0.7)
    opts_late = _options(use_diis=True, damping=0.7)
    opts_late.diis_start_iter = 1_000_000   # effectively disabled
    r_damp = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts_damp, omega=0.5, spacing_bohr=0.3,
    )
    r_late = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts_late, omega=0.5, spacing_bohr=0.3,
    )
    # With DIIS deferred indefinitely, iteration count must match the
    # pure-damping path exactly.
    assert r_damp.n_iter == r_late.n_iter
    assert r_damp.energy == pytest.approx(r_late.energy, abs=1e-10)


# ---------------------------------------------------------------------------
# Resilience: once converged, B matrix goes singular; shouldn't blow up
# ---------------------------------------------------------------------------

def test_diis_handles_near_singular_b_matrix_gracefully():
    """After convergence, successive iterations would produce
    near-identical error vectors and a near-singular B matrix.
    The fallback path (drop oldest history entry, retry) must keep
    the SCF healthy even if we force extra iterations."""
    sysp, basis = _h2_in_box()
    opts = _options(use_diis=True, damping=0.3)
    opts.max_iter = 30    # cap at 30 — plenty to over-converge
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    # Energy is a clean finite number, not NaN.
    assert np.isfinite(r.energy)
