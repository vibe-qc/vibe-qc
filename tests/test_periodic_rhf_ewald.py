"""Phase 12e-c-4b: Γ-point periodic RHF SCF driver using EWALD_3D.

The correctness contract is the ω-invariance of the Ewald identity:
at SCF convergence, the total energy from
``run_rhf_periodic_gamma_ewald3d`` must be independent of the
splitting parameter ω, up to the finite-box / finite-grid residual
measured in 12e-c-4a (about 0.3 % on H2 / STO-3G / 30-bohr / 0.3-bohr
grid).

The driver's energy differs from the existing
``run_rhf_periodic_gamma`` (DIRECT_TRUNCATED) reference by the
Makov–Payne constant shift α · S integrated against the density —
well-defined physics, and the offset matches the prediction
``α ≈ -α_M · Q / L`` (simple-cubic Madelung constant) to within a
percent on suitable test systems.

Systems used
------------

Every test uses H₂ / STO-3G — no tight cores, cleanly resolved on a
0.3-bohr grid. H₂O / STO-3G fails because the O 1s core is too tight
for the default grid; that's the known-xfail case from 12e-c-4a,
documented in :func:`vibeqc.build_j_ewald_3d`.

Tests
-----

1. **Convergence on H₂** — SCF terminates successfully.
2. **ω-invariance** — total energy stable across a spread of ω.
3. **Makov–Payne consistency** — ``E_ewald - E_direct`` matches
   ``0.5 · α_M · n_elec² / L`` for a cubic box.
4. **Result shape correctness** — mo_coeffs / mo_energies / density /
   fock / overlap come out as expected-shape numpy arrays.
5. **Rejection of multi-atom multiplicity mismatches** — open-shell
   structures can't go through the closed-shell RHF driver.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq


def _run_ewald3d(*args, **kwargs):
    """``run_rhf_periodic_gamma_ewald3d`` with the truncation
    auto-optimiser pinned OFF (defensive — on the dilute H₂ boxes here
    it is a no-op since the overlap is already PSD; kept off so the one
    tight LiH-primitive gauge test never pays a cutoff-tuning re-run).

    The actual speed-up for this file is ``use_diis=True`` in
    ``_default_options``: these Γ-Ewald SCFs converge in ~3 iterations
    instead of ~17, and the dominant cost is the per-iteration analytic-FT
    Hartree-J rebuild (PBC-audit E2), so fewer iterations ⇒ proportionally
    fewer rebuilds (≈508→231 s CPU, all assertions unchanged). Note:
    auto_optimize_truncation / grid-spacing were measured *not* to matter
    here (45.8 vs 46.2 s; 100³ vs 38³ grid identical) — the J is analytic.
    """
    kwargs.setdefault("auto_optimize_truncation", False)
    return _run_rhf_periodic_gamma_ewald3d(*args, **kwargs)


_run_rhf_periodic_gamma_ewald3d = vq.run_rhf_periodic_gamma_ewald3d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_in_box(box: float = 30.0):
    """H₂ centered in a cubic box of given side (bohr). Returns
    ``(system, basis)``. Default 30 bohr matches 12e-c-4a tests."""
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options(iter_limit: int = 60, damping: float = 0.3):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    opts.max_iter = iter_limit
    # DIIS converges these H₂/Ewald SCFs in ~3 iterations instead of ~17
    # (identical fixed point), so the per-iteration analytic-FT J rebuild
    # is paid 3× not 17× — the dominant cost here. The old "driver doesn't
    # implement DIIS yet" comment was stale; the Γ-Ewald DIIS path is the
    # subject of test_periodic_rhf_ewald_diis.py.
    opts.use_diis = True
    return opts


# ---------------------------------------------------------------------------
# Convergence on H2
# ---------------------------------------------------------------------------


def test_h2_converges():
    """The SCF terminates successfully on a clean test case."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options()
    r = _run_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged, f"did not converge after {r.n_iter} iterations"
    assert r.n_iter <= opts.max_iter
    # Energy is finite and sensible (H2 RHF is ~ -1 Ha).
    assert -2.0 < r.energy < 0.0


# ---------------------------------------------------------------------------
# ω-invariance (the headline correctness witness)
# ---------------------------------------------------------------------------


def test_energy_is_omega_invariant_on_h2():
    """SCF total energy must be independent of ω to ~0.3 % — this is
    the deep contract of the Ewald decomposition. Any mismatch at the
    mHa scale on a 30-bohr H₂ box indicates a bug in the composed-J
    pipeline (which was validated separately in 12e-c-4a at the matrix
    level)."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options()

    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = _run_ewald3d(
            sysp,
            basis,
            opts,
            omega=omega,
            spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)

    spread = max(energies) - min(energies)
    # Spread < 0.5 % of |E| (matching the 12e-c-4a finite-box
    # Makov-Payne bound).
    assert spread < 0.005 * abs(min(energies)), (
        f"ω-invariance violated: {energies}, spread = {spread:.3e}"
    )


# ---------------------------------------------------------------------------
# Makov–Payne consistency with DIRECT_TRUNCATED
# ---------------------------------------------------------------------------


def test_energy_matches_direct_truncated_in_atomic_limit():
    """v0.6.1 Madelung-cancellation fix: after the SCF energy applies
    +α_M(Q_n²+Q_e²)/(2L), the Ewald-3D total energy in a big-enough
    box matches DIRECT_TRUNCATED to within the 12e-c-4a finite-box
    accuracy. Pre-fix, Ewald was off by ~ 0.5·α·n_electrons (the
    Makov-Payne shift). Post-fix that shift is removed by the
    Madelung correction; Ewald and DIRECT both converge to the
    molecular-limit total energy at large L.
    """
    box = 30.0
    sysp, basis = _h2_in_box(box=box)
    opts = _default_options()

    r_direct = vq.run_rhf_periodic_gamma(sysp, basis, opts)
    r_ewald = _run_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r_direct.converged
    assert r_ewald.converged

    diff = r_ewald.energy - r_direct.energy
    # 10 mHa tolerance — at L=30 the residual is ~3 mHa from the
    # spread-density vs point-charge Madelung approximation.
    assert abs(diff) < 1e-2, (
        f"Ewald - Direct mismatch in atomic limit: "
        f"E_ewald = {r_ewald.energy:.6f}, E_direct = {r_direct.energy:.6f}, "
        f"diff = {diff:.6f} Ha"
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_has_expected_shapes():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    r = _run_ewald3d(sysp, basis, opts, omega=0.5)
    n_bf = basis.nbasis
    assert r.density.shape == (n_bf, n_bf)
    assert r.fock.shape == (n_bf, n_bf)
    assert r.overlap.shape == (n_bf, n_bf)
    # Canonical orth may shrink MO space; columns ≤ n_bf.
    assert r.mo_coeffs.shape[0] == n_bf
    assert r.mo_coeffs.shape[1] <= n_bf
    assert r.mo_energies.shape == (r.mo_coeffs.shape[1],)
    # scf_trace entries are SCFIteration dataclass instances (the same
    # type the C++ DIRECT_TRUNCATED driver emits — see scf_trace
    # type unification commit).
    assert len(r.scf_trace) >= 1
    first = r.scf_trace[0]
    assert isinstance(first, vq.SCFIteration)
    assert first.iter >= 1
    assert isinstance(first.energy, float)
    assert isinstance(first.delta_e, float)
    assert isinstance(first.grad_norm, float)
    assert r.grid_shape == (100, 100, 100)  # auto_grid(30 bohr, 0.3 spacing)
    # ω = 0.5 passed explicitly; not auto-derived.
    assert r.omega == pytest.approx(0.5)


def test_result_matrices_are_symmetric():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    r = _run_ewald3d(sysp, basis, opts, omega=0.5)
    assert np.allclose(r.overlap, r.overlap.T, atol=1e-12)
    assert np.allclose(r.fock, r.fock.T, atol=1e-8)
    assert np.allclose(r.density, r.density.T, atol=1e-8)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_accepts_non_orthorhombic_lattice_smoke():
    """Skew cells run through the native FFT metric path."""
    lat = np.array(
        [
            [30.0, 1.5, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )
    c = 15.0
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options(iter_limit=1)
    r = _run_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)


def test_rejects_open_shell_system():
    """Closed-shell RHF driver must reject odd-electron systems."""
    # H atom (1 electron, multiplicity 2) in a box
    c = 15.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * 30.0,
        [vq.Atom(1, [c, c, c])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options()
    with pytest.raises(ValueError, match="closed-shell"):
        _run_ewald3d(sysp, basis, opts)


def test_auto_enforces_ewald3d_gauge():
    """A1 regression: EWALD_3D driver auto-sets coulomb_method=EWALD_3D."""
    a = 4.084 / 0.529177210903
    system = vq.PeriodicSystem(
        3,
        np.diag([a, a, a]),
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 10
    opts.conv_tol_energy = 1e-7
    opts.use_diis = True  # fewer iterations on the tight LiH-primitive cell
    result = _run_ewald3d(system, basis, opts, progress=False)
    assert opts.lattice_opts.coulomb_method == vq.CoulombMethod.EWALD_3D
    assert -15.0 < result.energy < -1.0, (
        f"EWALD_3D energy out of physical range: {result.energy:.3f} Ha"
    )
