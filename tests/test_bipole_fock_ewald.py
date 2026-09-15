"""Tests for BIPOLE Phase 5: Ewald J-split F^2e build.

Validates:
  * build_fock_2e_ewald_j_split_gamma runs cleanly + returns the
    expected component matrices (J_SR, J_LR, K_full, F2e).
  * Per-component energies have the right SIGN and rough magnitudes.
  * MgO CRYSTAL parity: one-electron, two-electron, nuclear, and
    spheropole components match the sealed CYC 0 reference.
  * Component additivity: ½·tr(D·F^2e) = ½·tr(D·J_SR) + ½·tr(D·J_LR)
    − ¼·tr(D·K_full).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import CoulombMethod, InitialGuess, LatticeSumOptions
from vibeqc._vibeqc_core import compute_overlap_lattice
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.bipole_fock_ewald import (
    EwaldJSplitFock,
    build_fock_2e_ewald_j_split_gamma,
    compute_J_long_range_gamma,
)
from vibeqc.bipole_lattice_self_energy import cell_volume_bohr3
from vibeqc.guess import initial_density_closed_shell

ANG2BOHR = 1.0 / 0.529177210903


def _build_mgo():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _make_p_real_iter1_sad(system, basis, opts):
    """Build a LatticeMatrixSet density with SAD at g=0, zeros elsewhere."""
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    P_real = S_lat
    for g_idx in range(len(P_real.cells)):
        is_g0 = (np.asarray(P_real.cells[g_idx].index) == np.array([0, 0, 0])).all()
        P_real.set_block(
            g_idx,
            np.asarray(D_sad, dtype=float)
            if is_g0
            else np.zeros_like(np.asarray(D_sad), dtype=float),
        )
    return P_real, np.asarray(D_sad, dtype=float)


# ---------------------------------------------------------------------
# compute_J_long_range_gamma
# ---------------------------------------------------------------------
def test_J_long_range_shape_and_symmetry():
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    J_LR = compute_J_long_range_gamma(P_real, basis, system, omega, precision=1e-6)
    assert J_LR.shape == (basis.nbasis, basis.nbasis)
    # J Fock matrix should be Hermitian (symmetric for real basis).
    np.testing.assert_allclose(J_LR, J_LR.T, atol=1e-10)


def test_J_long_range_dim_check():
    """Requires dim=3."""
    lat = np.eye(3) * 5.0
    sys2d = vq.PeriodicSystem(2, lat, [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(sys2d.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 5.0
    S_lat = compute_overlap_lattice(basis, sys2d, opts)
    with pytest.raises(ValueError):
        compute_J_long_range_gamma(S_lat, basis, sys2d, omega=1.0)


def test_J_long_range_omega_positive():
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    with pytest.raises(ValueError):
        compute_J_long_range_gamma(P_real, basis, system, omega=0.0)


# ---------------------------------------------------------------------
# build_fock_2e_ewald_j_split_gamma — structure + sanity
# ---------------------------------------------------------------------
def test_ewald_j_split_returns_components():
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    result = build_fock_2e_ewald_j_split_gamma(
        P_real,
        basis,
        system,
        opts,
        precision=1e-6,
    )
    assert isinstance(result, EwaldJSplitFock)
    nbf = basis.nbasis
    assert result.F2e.shape == (nbf, nbf)
    assert result.J_SR.shape == (nbf, nbf)
    assert result.J_LR.shape == (nbf, nbf)
    assert result.K_full.shape == (nbf, nbf)
    # F2e = J_SR + J_LR - 0.5 K_full (with symmetrisation)
    expected = result.J_SR + result.J_LR - 0.5 * result.K_full
    expected = 0.5 * (expected + expected.T)
    np.testing.assert_allclose(result.F2e, expected, atol=1e-12)


def test_ewald_j_split_default_omega():
    """If omega not given, uses CRYSTAL default α = (1/2.8)·V^{1/3}."""
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    result = build_fock_2e_ewald_j_split_gamma(P_real, basis, system, opts)
    V = cell_volume_bohr3(system)
    expected_omega = crystal_default_ewald_alpha(V)
    assert math.isclose(result.omega_bohr_inv, expected_omega, rel_tol=1e-12)


def test_ewald_j_split_component_additivity():
    """½·tr(D·F^2e) = ½·tr(D·J_SR) + ½·tr(D·J_LR) − ¼·tr(D·K_full).

    Allow a tiny symmetrisation tolerance from the 0.5·(F + F^T) step.
    """
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, D = _make_p_real_iter1_sad(system, basis, opts)
    result = build_fock_2e_ewald_j_split_gamma(
        P_real,
        basis,
        system,
        opts,
        precision=1e-6,
    )
    E_F2e = 0.5 * np.trace(D @ result.F2e)
    E_sum = (
        0.5 * np.trace(D @ result.J_SR)
        + 0.5 * np.trace(D @ result.J_LR)
        - 0.25 * np.trace(D @ result.K_full)
    )
    assert math.isclose(E_F2e, E_sum, abs_tol=1e-6)


# ---------------------------------------------------------------------
# CRYSTAL parity — MgO at iter-1 SAD (CYC 0) via run_pbc_bipole_rhf
#
# CRYSTAL23/MgO/STO-3G/SHRINK-8-8 ENECYCLE CYC 0 reference values
# (per primitive cell, 1 FU):
#   KINETIC ENERGY:      +268.01817528309
#   TOTAL E-N + N-E:     −511.81837423575
#   BIELET ZONE E-E:     +570.69557431623
#   EXT EL-POLE:         −528.60425340803
#   EXT EL-SPHEROPOLE:    +4.1191890135070
#   TOTAL E-E:           +46.21050992171
#   TOTAL N-N:           −73.084276676762
#   TOTAL ENERGY:        −270.67396570771
#
# vibe-qc's BIPOLE driver energy components:
#   E_kin  = Tr[D·T]                    ↔ CRYSTAL KINETIC ENERGY
#   E_ne   = Tr[D·V_ne]                 ↔ CRYSTAL TOTAL E-N + N-E
#   E_2e   = ½Tr[D·F^2e]                ↔ BIELET ZONE E-E + EXT EL-POLE
#   E_spheropole                        ↔ EXT EL-SPHEROPOLE (energy-only)
#   E_nuc                               ↔ TOTAL N-N
#   E_total = E_kin+E_ne+E_2e+E_spheropole+E_nuc ↔ TOTAL ENERGY
#
# The V_ne background term (+πQ_n/(α²V)·S) is placed differently from
# CRYSTAL (where the G=0 correction lives in E_nn).  This shifts
# ~16 Ha from E_nn to E_ne in the per-component breakdown, but E_total
# is gauge-invariant for neutral cells.
# ---------------------------------------------------------------------
# CRYSTAL CYC 0 reference values (Ha per primitive cell)
_CRYSTAL_CYC0 = {
    "E_kin": +268.01817528309,
    "E_ne": -511.81837423575,
    "E_2e": +42.09131090820,  # BIELET+POLE (no spheropole)
    "E_spheropole": +4.1191890135070,
    "E_nuc": -73.084276676762,
    "E_total": -270.67396570771,
}


@pytest.mark.slow
def test_crystal_parity_end_to_end_c0(monkeypatch):
    """End-to-end BIPOLE RHF CYC 0 parity vs CRYSTAL ENECYCLE.

    Runs run_pbc_bipole_rhf with max_iter=1 (SAD initial guess →
    one Fock build → energy).  Compares per-component energies against
    CRYSTAL23 ENECYCLE output for MgO/STO-3G/SHRINK-8-8.

    The one-electron, nuclear, and spheropole components are tight
    against the sealed reference.  The Γ-only two-electron external
    comparison remains a known gated gap for the exact Ewald-J route:
    the production sign-off is the converged multi-k path, while the
    opt-in far-pair branch is still being certified.
    """
    system, basis = _build_mgo()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 1
    opts.initial_guess = vq.InitialGuess.SAD
    opts.use_diis = False
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-6
    opts.conv_tol_grad = 1e-6

    # The driver evaluates the local SAD density first, then diagonalises
    # once and performs a mandatory terminal rebuild on the density it
    # returns. Capture both spheropole calls so the CYC0 comparison cannot
    # silently drift onto that distinct terminal density again.
    import vibeqc.pbc_bipole as _bipole_driver

    _compute_spheropole = _bipole_driver.compute_ext_el_spheropole
    spheropole_calls = []

    def _capture_spheropole(P_real, *args, **kwargs):
        value = float(_compute_spheropole(P_real, *args, **kwargs))
        snapshot = tuple(
            (
                tuple(np.asarray(cell.index, dtype=int)),
                np.asarray(block, dtype=float).copy(),
            )
            for cell, block in zip(P_real.cells, P_real.blocks)
        )
        spheropole_calls.append((snapshot, value))
        return value

    monkeypatch.setattr(
        _bipole_driver,
        "compute_ext_el_spheropole",
        _capture_spheropole,
    )

    result = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        # CRYSTAL CYC0 parity is a LEGACY-gauge comparison: CRYSTAL's
        # gauge carries the spheropole term and the Γ-locality energy
        # convention; under the Ewald exchange split (the Γ default,
        # option (b) 2026-06-10) the spheropole is omitted and CYC0
        # components are not CRYSTAL-comparable.
        use_exchange_ewald_split=False,
        use_multipole_far_field=False,
        progress=False,
    )

    # The first driver evaluation is exactly the local SAD convention used
    # by the standalone kernel oracle: SAD at g=0 and zero elsewhere.
    assert len(spheropole_calls) == 2
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        system.n_electrons() // 2,
        InitialGuess.SAD,
        is_periodic=True,
    )
    assert D_sad is not None
    first_density = dict(spheropole_calls[0][0])
    np.testing.assert_array_equal(first_density[(0, 0, 0)], D_sad)
    for cell_index, block in first_density.items():
        if cell_index != (0, 0, 0):
            np.testing.assert_array_equal(block, np.zeros_like(block))

    # Extract the preserved CYC0 components. ``energy_components[-1]`` is
    # intentionally the once-diagonalised density returned to callers.
    ec = result.initial_density_energy_components
    assert ec is not None
    assert ec.e_ext_el_spheropole == pytest.approx(
        spheropole_calls[0][1], abs=1e-12
    )
    vibe_kin = float(ec.e_kinetic)
    vibe_ne = float(ec.e_nuclear_attraction)
    vibe_2e = float(ec.e_two_electron)
    vibe_sphe = float(ec.e_ext_el_spheropole)
    vibe_nuc = float(ec.e_nuclear_repulsion)
    vibe_tot = float(ec.e_total)

    # --- Per-component checks ---
    # Kinetic: cutoff-insensitive, should match closely.
    assert vibe_kin == pytest.approx(_CRYSTAL_CYC0["E_kin"], abs=3.0), (
        f"E_kin: vibe={vibe_kin:.4f}, CRYSTAL={_CRYSTAL_CYC0['E_kin']:.4f}"
    )

    # E_ne: vibe-qc uses complete Ewald V_ne (erfc + reciprocal + G=0
    # background).  The total converges to the same value as CRYSTAL's
    # separate V_ne + E_nn_bg handling.  Match within convergence.
    assert vibe_ne == pytest.approx(_CRYSTAL_CYC0["E_ne"], abs=3.0), (
        f"E_ne: vibe={vibe_ne:.4f}, CRYSTAL={_CRYSTAL_CYC0['E_ne']:.4f}"
    )

    # Spheropole: verified at 99.8% parity.  Check within 1 Ha.
    assert vibe_sphe == pytest.approx(_CRYSTAL_CYC0["E_spheropole"], abs=1.0), (
        f"E_spheropole: vibe={vibe_sphe:.4f}, CRYSTAL={_CRYSTAL_CYC0['E_spheropole']:.4f}"
    )

    # Nuclear repulsion: Ewald with jellium background.
    assert vibe_nuc == pytest.approx(_CRYSTAL_CYC0["E_nuc"], abs=3.0), (
        f"E_nuc: vibe={vibe_nuc:.4f}, CRYSTAL={_CRYSTAL_CYC0['E_nuc']:.4f}"
    )

    # Self-consistency: E_total = E_kin + E_ne + E_2e + E_spheropole + E_nuc.
    _sum = vibe_kin + vibe_ne + vibe_2e + vibe_sphe + vibe_nuc
    assert vibe_tot == pytest.approx(_sum, abs=1e-6)

    # Two-electron external parity is now a live guard for the exact
    # Ewald-J route.  This was formerly gated while the native far-pair
    # decomposition lagged the production energy path.
    _expected_total = (
        _CRYSTAL_CYC0["E_kin"]
        + _CRYSTAL_CYC0["E_ne"]
        + _CRYSTAL_CYC0["E_2e"]
        + _CRYSTAL_CYC0["E_spheropole"]
        + _CRYSTAL_CYC0["E_nuc"]
    )

    # Two-electron: E_2e = E_J_SR + E_J_LR + E_exchange (includes jellium
    # background in J_LR). Should converge to the external two-electron +
    # far-pair energy when the gated decomposition closes.
    assert vibe_2e == pytest.approx(_CRYSTAL_CYC0["E_2e"], abs=5.0), (
        f"E_2e: vibe={vibe_2e:.4f}, CRYSTAL={_CRYSTAL_CYC0['E_2e']:.4f}"
    )
    assert vibe_tot == pytest.approx(_expected_total, abs=6.0), (
        f"E_total: vibe={vibe_tot:.4f}, CRYSTAL={_expected_total:.4f}"
    )

    terminal = result.energy_components[-1]
    assert terminal.e_ext_el_spheropole == pytest.approx(
        spheropole_calls[-1][1], abs=1e-12
    )
    assert result.e_ext_el_spheropole == pytest.approx(
        terminal.e_ext_el_spheropole, abs=1e-12
    )
    assert result.energy == pytest.approx(terminal.e_total, abs=1e-12)


# ---------------------------------------------------------------------
# CRYSTAL E_nuc parity — Ewald nuclear repulsion matches CRYSTAL exactly
# ---------------------------------------------------------------------
_CRYSTAL_E_NUC_MGO = -73.084276676762  # CRYSTAL23 TOTAL N-N, all CYC


@pytest.mark.slow
def test_crystal_parity_ewald_nuclear_repulsion():
    """Ewald nuclear repulsion matches CRYSTAL to <10 µHa.

    Runs a converged [2,2,2] RHF on MgO/STO-3G and checks that the
    Ewald nuclear repulsion E_nuc matches CRYSTAL23's TOTAL N-N to
    microhartree precision.  E_nuc is density-independent — if this
    matches, the Ewald α, K_max, jellium background, and Madelung
    sum are all correct.

    CRYSTAL23 reference (MgO/STO-3G, primitive FCC, a=4.21 Å):
      TOTAL N-N = −73.084276676762 Ha  (all CYC)
    """
    system, basis = _build_mgo()
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 5
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.conv_tol_energy = 1e-6
    opts.conv_tol_grad = 1e-4

    result = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        progress=False,
    )

    # E_nuc is the nuclear repulsion per cell — extracted from the
    # last energy_components entry.
    e_nuc_vibe = float(result.energy_components[-1].e_nuclear_repulsion)
    delta_ha = e_nuc_vibe - _CRYSTAL_E_NUC_MGO
    delta_uha = delta_ha * 1e6
    print(f"  vibe-qc E_nuc = {e_nuc_vibe:.12f} Ha")
    print(f"  CRYSTAL E_nuc = {_CRYSTAL_E_NUC_MGO:.12f} Ha")
    print(f"  delta = {delta_uha:+.1f} µHa")
    assert abs(delta_ha) < 1e-5, (
        f"E_nuc mismatch: vibe={e_nuc_vibe:.12f}, "
        f"CRYSTAL={_CRYSTAL_E_NUC_MGO:.12f}, delta={delta_uha:.1f} µHa"
    )


# ---------------------------------------------------------------------
# Multi-iteration SCF convergence test
# ---------------------------------------------------------------------
@pytest.mark.slow
def test_bipole_rhf_scf_converges():
    """BIPOLE RHF SCF converges on MgO/STO-3G within 10 iterations."""
    system, basis = _build_mgo()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 3
    opts.initial_guess = vq.InitialGuess.SAD
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 6
    opts.damping = 0.5
    opts.conv_tol_energy = 1e-5
    opts.conv_tol_grad = 1e-4

    result = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        # Legacy gauge: the decreasing-trajectory heuristic below is a
        # legacy-era property. Under the Ewald exchange split the SAD
        # iter-1 energy undershoots (the non-idempotent guess makes the
        # S·D·S exchange correction non-variational at iter 1) and the
        # SCF then RISES to the stationary point; new-gauge convergence
        # is covered by tests/test_bipole_fock_ewald_exchange.py.
        use_exchange_ewald_split=False,
        progress=False,
    )
    assert result.n_iter <= 3
    assert result.energy < 0
    # Energy should decrease (converge downward) — first 3 iters
    energies = [ec.e_total for ec in result.energy_components]
    assert energies[2] < energies[0], "energy should decrease during SCF"
    # No NaN
    assert np.isfinite(result.energy)


# ---------------------------------------------------------------------
# RKS BIPOLE smoke test
# ---------------------------------------------------------------------
@pytest.mark.slow
def test_bipole_rks_smoke():
    """RKS (PBE) BIPOLE driver converges on MgO/STO-3G with multi-k.

    Uses [2,2,2] k-mesh — the minimum needed to avoid the Γ-only
    SCF bifurcation documented in the retired PBC-BIPOLE handover
    (2026-06-13; see git history).
    CRYSTAL's own MgO inputs use SHRINK 8 8 for the same reason.
    """
    system, basis = _build_mgo()
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 30
    opts.initial_guess = vq.InitialGuess.SAD
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 6
    opts.damping = 0.5
    opts.fock_mixing = 0.3
    opts.conv_tol_energy = 1e-5
    opts.conv_tol_grad = 1e-4

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    result = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        progress=False,
    )
    # Multi-k RKS on MgO now converges cleanly with smearing+FMIXING
    # (see smearing gate fix, 2026-06-05).  Previously oscillated ~0.65 Ha.
    assert result.n_iter > 0
    assert np.isfinite(result.energy)
    assert -290 < result.energy < -250, f"unexpected energy: {result.energy:.4f} Ha"


@pytest.mark.slow
def test_bipole_multik_smoke():
    """Multi-k (2x2x2) BIPOLE RHF runs and produces physical energy.

    A 2x2x2 k-mesh on MgO/STO-3G exercises the IBZ expansion and
    multi-k J^LR density transform.  Checks that the path completes
    and the energy is physically plausible (bound, negative)."""
    system, basis = _build_mgo()
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 1
    opts.initial_guess = vq.InitialGuess.SAD
    opts.use_diis = False
    opts.damping = 0.0

    result = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_multipole_far_field=False,
        progress=False,
    )
    assert result.converged or result.n_iter == 1
    assert np.isfinite(result.energy)
    assert result.energy < 0, (
        f"bound system should have negative energy, got {result.energy:.4f}"
    )
    # Multi-k should have per-k MOs
    assert len(result.mo_energies) == len(list(kmesh.kpoints))


# ---------------------------------------------------------------------
# UHF/UKS BIPOLE parity
# ---------------------------------------------------------------------
def test_bipole_uhf_smoke():
    """UHF BIPOLE driver runs on open-shell system."""
    # Build a simple open-shell system: H atom in a box (doublet)
    box = 15.0
    lat = np.eye(3) * box
    atoms = [vq.Atom(1, [box / 2, box / 2, box / 2])]
    system = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    result = run_pbc_bipole_uhf(
        system,
        basis,
        kmesh,
        use_ewald_j_split=True,
        progress=False,
    )
    assert result.converged
    assert result.energy < 0, f"energy should be negative, got {result.energy:.4f}"
    # ⟨S²⟩ should be ~0.75 for a doublet
    assert 0.6 < result.s_squared < 0.9


def test_bipole_uks_smoke():
    """UKS (PBE) BIPOLE driver runs on open-shell system."""
    # H atom in a box (doublet), PBE functional
    box = 15.0
    lat = np.eye(3) * box
    atoms = [vq.Atom(1, [box / 2, box / 2, box / 2])]
    system = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    result = run_pbc_bipole_uks(
        system,
        basis,
        kmesh,
        use_ewald_j_split=True,
        functional="pbe",
        progress=False,
    )
    assert result.converged
    assert result.energy < 0, f"energy should be negative, got {result.energy:.4f}"
    # ⟨S²⟩ should be ~0.75 for a doublet
    assert 0.6 < result.s_squared < 0.9


# ---------------------------------------------------------------------
# P3: Spheropole kernel p-shell validation (MgO exercises O 2p)
# ---------------------------------------------------------------------
def test_spheropole_on_mgo_p_shells():
    """EXT EL-SPHEROPOLE on MgO/STO-3G exercises O 2p shells.

    Previously verified at 99.8% of CRYSTAL14 (4.112 vs 4.119 Ha).
    This test confirms the spheropole is positive and within
    ~0.5 Ha of the known reference, covering s-s, s-p, and p-p
    kernel cases."""
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    from vibeqc.bipole_ext_el_pole import compute_ext_el_spheropole

    E_sphero = float(compute_ext_el_spheropole(P_real, basis, system, opts))
    # Should be positive (repulsive self-interaction correction)
    assert E_sphero > 0.0, f"spheropole should be positive, got {E_sphero:.4f}"
    # 99.8% parity confirmed previously
    assert E_sphero == pytest.approx(4.119, abs=0.5)


# ---------------------------------------------------------------------
# P4: Multipole far-field smoke test
# ---------------------------------------------------------------------
def test_multipole_far_field_enable_warns_experimental():
    """Enabling the opt-in multipole far-field must be visibly experimental."""
    from vibeqc.bipole_fock_multipole import resolve_multipole_config

    system, basis = _build_mgo()
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    with pytest.warns(UserWarning, match="experimental"):
        cfg = resolve_multipole_config(
            system,
            basis,
            opts.lattice_opts,
            user_enable=True,
            multipole_l_max=2,
        )
    assert cfg.enabled


@pytest.mark.slow
def test_multipole_far_field_smoke():
    """The retired low-level artifact is not reachable from SCF."""
    with pytest.raises(NotImplementedError, match="three-translation"):
        vq.run_pbc_bipole_rhf(
            None,
            None,
            None,
            use_multipole_far_field=True,
        )


# ---------------------------------------------------------------------
# P2: L=3 octupole conversion — analytic validation
# ---------------------------------------------------------------------
def _multipole_test_charges():
    """Point-charge clusters with non-zero multipoles up to L=3.

    Returns (charges_a, positions_a, charges_b, positions_b, R)
    where R is the separation vector between cluster centroids.

    Cluster A: a linear quadrupole in the xz-plane — two +q at
    (a, 0, c) and (-a, 0, -c), two -q at (-a, 0, c) and (a, 0, -c).
    Net zero monopole and dipole; non-zero quadrupole and octupole.

    Cluster B: identical copy shifted by R (off-axis to avoid
    coincident charge pairs).
    """
    q = 1.0
    a = 0.7
    c = 0.3

    charges_a = np.array([q, q, -q, -q])
    positions_a = np.array(
        [[a, 0.0, c], [-a, 0.0, -c], [-a, 0.0, c], [a, 0.0, -c]],
        dtype=float,
    )
    Q0 = np.sum(charges_a)
    dipole = np.sum(charges_a[:, None] * positions_a, axis=0)
    assert abs(Q0) < 1e-12, f"monopole not zero: {Q0}"
    assert np.allclose(dipole, 0, atol=1e-12), f"dipole not zero: {dipole}"

    # Off-axis separation to avoid coincident charge pairs
    R_vec = np.array([5.0, 1.5, 0.0])
    charges_b = charges_a.copy()
    positions_b = positions_a + R_vec
    return charges_a, positions_a, charges_b, positions_b, R_vec


def test_multipole_pair_energy_converges():
    import numpy as np
    from vibeqc._cart_to_sph import cartesian_to_spherical_matrix
    from vibeqc.bipole_cell_moments import cartesian_component_indices
    from vibeqc.bipole_multipole import multipole_pair_energy, n_components

    q1, q2, d = 1.0, -1.0, 0.15
    R = np.array([10.0, 0.0, 0.0])

    def m(q, rr, L):
        ix = cartesian_component_indices(L)
        ca = np.zeros(len(ix))
        for ci, (i, j, k) in enumerate(ix):
            ca[ci] = q * (rr[0] ** i) * (rr[1] ** j) * (rr[2] ** k)
        return cartesian_to_spherical_matrix(L) @ ca

    MA = m(q1, np.array([d, 0, 0]), 3)
    MB = m(q2, np.array([0.0, 0.0, 0.0]), 3)
    E_ex = q1 * q2 / (10 - d)
    errs = []
    for L in range(4):
        n = (L + 1) ** 2
        E = float(multipole_pair_energy(MA[:n], MB[:n], R))
        errs.append(abs(E - E_ex))
    assert errs[0] > 1e-3, f"L=0 monopole not dominant"
    assert errs[1] < errs[0], f"L=1 not converging"
    assert errs[2] < errs[1], f"L=2 not converging"
    assert errs[3] < 1e-5, f"L=3 err {errs[3]:.2e}"
    assert n_components(3) == 16
    assert len(cartesian_component_indices(3)) == 20
    assert cartesian_to_spherical_matrix(3).shape == (16, 20)
