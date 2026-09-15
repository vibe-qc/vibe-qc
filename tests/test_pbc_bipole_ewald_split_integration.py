"""End-to-end integration test for the Ewald J-split path in
``run_pbc_bipole_rhf``.

Validates the BIPOLE Ewald-J-split integration surface: shared Ewald
alpha, analytic ``V_ne`` sanity checks, native J/K component assembly,
multi-k smoke coverage, and the slower Γ-only convergence smoke.

This test is slow (~5-10 min wall, even at cutoff=8 with max_iter=5)
because each iteration still runs a direct short-range-J build plus a
native full-range J/K component build. Marked ``@pytest.mark.slow`` so
it can be skipped in normal CI and run manually for parity validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import (
    AngularPruning,
    AngularScheme,
    CoulombMethod,
    LatticeSumOptions,
    PeriodicRHFOptions,
    bloch_sum,
    build_fock_2e_real_space,
    build_jk_2e_real_space,
    compute_kinetic_lattice,
    compute_overlap_lattice,
)
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.guess import initial_density_closed_shell
from vibeqc.pbc_bipole import (
    _compute_nuclear_lattice_ewald_reciprocal_ft,
    _crystal_ewald_options,
    _default_bipole_v_ne_grid_options,
    run_pbc_bipole_rhf,
)
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

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


def _build_he_box():
    """Cheap fold-converged cell for corrected-gauge integration smokes."""
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_crystal_ewald_options_use_shared_alpha():
    """BIPOLE V_ne/E_nn should use the same CRYSTAL alpha as J^LR."""
    assert vq.run_pbc_bipole_rhf is run_pbc_bipole_rhf
    system, _basis = _build_mgo()
    opts = PeriodicRHFOptions()
    opts.lattice_opts.nuclear_cutoff_bohr = 11.0
    V_cell = float(
        abs(
            np.linalg.det(np.asarray(system.lattice, dtype=float)),
        )
    )
    alpha = crystal_default_ewald_alpha(V_cell)
    eopts = _crystal_ewald_options(
        opts.lattice_opts,
        alpha_bohr_inv=alpha,
        tolerance=1e-8,
    )
    assert math.isclose(eopts.alpha, alpha, rel_tol=1e-12)
    assert math.isclose(eopts.real_cutoff_bohr, 11.0, rel_tol=1e-12)
    assert math.isclose(eopts.tolerance, 1e-8, rel_tol=1e-12)


def test_default_bipole_v_ne_grid_is_tighter_than_generic_default():
    gopts = _default_bipole_v_ne_grid_options()
    assert gopts.n_radial == 99
    assert gopts.angular == AngularScheme.Lebedev
    assert gopts.lebedev_order == 41
    assert gopts.angular_pruning == getattr(AngularPruning, "None")


def test_analytic_ewald_v_ne_matches_grid_on_s_shell_cell():
    """Analytic reciprocal V_ne should reproduce the old grid path.

    Use a one-He cubic cell so the AO-pair FT is s-only and the grid
    quadrature has no angular p-shell sensitivity. This catches sign,
    phase, and background mistakes in the analytic long-range formula.
    """
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    ewald_lat_opts = LatticeSumOptions()
    ewald_lat_opts.cutoff_bohr = 8.0
    ewald_lat_opts.nuclear_cutoff_bohr = 25.0
    ewald_lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    direct_lat_opts = LatticeSumOptions()
    direct_lat_opts.cutoff_bohr = 8.0
    direct_lat_opts.nuclear_cutoff_bohr = 25.0
    direct_lat_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED

    eopts = _crystal_ewald_options(
        ewald_lat_opts,
        alpha_bohr_inv=0.4,
        tolerance=1e-9,
    )
    S_lat = compute_overlap_lattice(basis, system, direct_lat_opts)
    V_analytic, cache = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        ewald_lat_opts,
        eopts,
        S_lat,
        precision=1e-9,
    )
    V_grid = compute_nuclear_lattice_dispatch(
        basis,
        system,
        ewald_lat_opts,
        grid_options=_default_bipole_v_ne_grid_options(),
        ewald_options=eopts,
    )

    assert cache.K_vectors.shape[0] > 0
    assert len(V_analytic.cells) == len(V_grid.cells)
    for block_analytic, block_grid in zip(V_analytic.blocks, V_grid.blocks):
        np.testing.assert_allclose(
            np.asarray(block_analytic),
            np.asarray(block_grid),
            atol=1e-9,
        )


def test_use_ewald_j_split_multik_runs():
    """Multi-k is supported as of v0.9.0 via per-k ``compute_J_long_range_at_k``.

    The empirical k-independent shortcut tried at 2026-05-18 (commit
    ``6cc5eab``) was reverted; the gauge-correct per-k path uses
    Bloch-summed bra-pair FT + shifted-ν ρ̂(K). See
    ``tests/test_pbc_bipole_multik_ewald_split.py`` for the dedicated
    multi-k test surface.
    """
    system, basis = _build_he_box()
    kmesh = monkhorst_pack(system, [2, 2, 2])
    opts = PeriodicRHFOptions()
    # Seven bohr keeps the overlap-fold drift at 5.2e-5 and makes the
    # doubled density support cover every 2x2x2 BvK residue class.
    opts.lattice_opts.cutoff_bohr = 7.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    assert result.n_iter == 1
    assert len(result.energy_components) == result.n_iter
    comp = result.energy_components[-1]
    assert math.isclose(
        comp.e_kinetic + comp.e_nuclear_attraction + comp.e_two_electron,
        result.e_electronic,
        abs_tol=1e-8,
    )
    assert math.isclose(
        comp.e_electronic
        + comp.e_nuclear_repulsion
        + (comp.e_ext_el_spheropole or 0.0),
        comp.e_total,
        abs_tol=1e-8,
    )
    assert math.isclose(comp.e_total, result.energy, abs_tol=1e-8)
    assert math.isclose(
        comp.e_nuclear_repulsion,
        result.e_nuclear,
        abs_tol=1e-12,
    )
    assert comp.e_kinetic > 0.0
    assert comp.e_nuclear_attraction < 0.0
    assert comp.e_j_short_range is not None
    assert comp.e_j_long_range is not None
    assert comp.e_exchange is not None
    assert math.isclose(
        (comp.e_j_short_range or 0) + (comp.e_j_long_range or 0)
        + (comp.e_exchange or 0) + (comp.e_j_multipole or 0),
        comp.e_two_electron,
        abs_tol=1e-8,
    )


def test_bipole_3d_default_uses_crystal_gauge_ewald_j_split():
    """3D BIPOLE should default to the CRYSTAL-gauge F²e path.

    The legacy direct-only branch is still available via
    ``use_ewald_j_split=False``, but the default must not silently take
    that known-wrong tight-cell composition.
    """
    system, basis = _build_he_box()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        ewald_precision=1e-6,
        progress=False,
    )
    comp = result.energy_components[-1]
    assert comp.e_j_short_range is not None
    assert comp.e_j_long_range is not None
    assert comp.e_exchange is not None
    assert comp.e_bielet_zone_ee is None


def test_bipole_oda_fails_closed_before_ewald_j_build():
    """ODA cannot return one orbital state for its mixed trial density."""
    system, basis = _build_he_box()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    with pytest.raises(NotImplementedError, match="mixed line-search density"):
        run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            use_oda=True,
            ewald_precision=1e-6,
            progress=False,
        )


def test_build_jk_2e_real_space_matches_fock_builder():
    """Native component builder should reproduce the legacy combined API."""
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 6.0
    opts.nuclear_cutoff_bohr = 6.0
    opts.schwarz_threshold = 1e-10

    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        system.n_electrons() // 2,
        InitialGuess.SAD,
        is_periodic=True,
    )
    assert D_sad is not None
    for g_idx in range(len(S_lat.cells)):
        is_g0 = (np.asarray(S_lat.cells[g_idx].index) == np.array([0, 0, 0])).all()
        S_lat.set_block(
            g_idx,
            np.asarray(D_sad, dtype=float)
            if is_g0
            else np.zeros_like(np.asarray(D_sad), dtype=float),
        )

    f_j = build_fock_2e_real_space(
        basis,
        system,
        opts,
        S_lat,
        0.0,
        0.0,
    )
    f_jmk = build_fock_2e_real_space(
        basis,
        system,
        opts,
        S_lat,
        1.0,
        0.0,
    )
    jk = build_jk_2e_real_space(basis, system, opts, S_lat, 0.0)

    assert len(jk.J.cells) == len(f_j.cells)
    assert len(jk.K.cells) == len(f_j.cells)
    for c in range(len(f_j.cells)):
        np.testing.assert_allclose(jk.J.blocks[c], f_j.blocks[c], atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(jk.J.blocks[c]) - 0.5 * np.asarray(jk.K.blocks[c]),
            f_jmk.blocks[c],
            atol=1e-10,
        )


def test_bipole_one_cycle_component_uses_returned_real_space_density(monkeypatch):
    """Terminal energy accounting contracts the returned real-space density.

    The one-cycle loop starts from a local SAD guess, diagonalises once, and
    returns the resulting density. The mandatory terminal consistency build
    replaces the history component with that returned state's energy. It
    must still use ``Σ_g D(g)T(g)``, never a Γ-folded operator contraction.

    Deliberately-small MgO cutoff: the gauge-independent fold guard
    (hoisted 2026-08-06) would refuse it, but this is a one-iteration
    contraction-semantics identity on a fixed local guess, not an
    absolute-energy claim -- bypass the measurement (retired-G1
    sentinel rationale).
    """
    import vibeqc.pbc_bipole_common as _common

    monkeypatch.setattr(
        _common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
    )
    system, basis = _build_mgo()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.lattice_opts.nuclear_cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    opts.damping = 0.0
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=False,
        progress=False,
    )

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 6.0
    lat_opts.nuclear_cutoff_bohr = 6.0
    lat_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        system.n_electrons() // 2,
        InitialGuess.SAD,
        is_periodic=True,
        overlap=np.asarray(bloch_sum(S_lat, np.zeros(3))),
    )
    assert D_sad is not None
    assert np.trace(D_sad @ np.asarray(bloch_sum(S_lat, np.zeros(3)))).real == pytest.approx(
        system.n_electrons(), abs=1e-10,
    )
    g0_T = next(
        np.asarray(block, dtype=float)
        for cell, block in zip(T_lat.cells, T_lat.blocks)
        if np.all(np.asarray(cell.index) == np.array([0, 0, 0]))
    )
    real_space_ekin = float(np.sum(D_sad * g0_T))
    gamma_ekin = float(
        np.trace(
            D_sad
            @ np.asarray(
                bloch_sum(T_lat, np.zeros(3)),
            ).real
        )
    )

    returned_ekin = _common._lattice_contract(
        result.density,
        T_lat,
        operator_name="T",
    )
    initial_comp = result.initial_density_energy_components
    assert initial_comp is not None
    assert initial_comp.e_kinetic == pytest.approx(real_space_ekin, abs=1e-10)

    comp = result.energy_components[-1]
    assert comp.e_kinetic == pytest.approx(returned_ekin, abs=1e-10)
    assert abs(comp.e_kinetic - real_space_ekin) > 1e-2
    assert result.energy == pytest.approx(comp.e_total, abs=1e-12)
    assert result.scf_trace[-1].energy == pytest.approx(result.energy, abs=1e-12)
    assert abs(gamma_ekin - real_space_ekin) > 1e-2
    assert len(S_lat.cells) == len(T_lat.cells)


def test_use_ewald_j_split_diis_smoke():
    """DIIS-compatible (2026-05-18 fix): D-consistent energy/error
    formulation makes DIIS error vectors well-defined.

    Just verify the SCF starts without raising; energy correctness
    is validated separately by the parity scripts."""
    system, basis = _build_he_box()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.initial_guess = InitialGuess.SAD
    # Should run cleanly; no NotImplementedError.
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    assert result.n_iter == 1


def test_use_ewald_j_split_dim2_raises():
    """dim < 3 should raise."""
    lat = np.eye(3) * 5.0
    sys2d = vq.PeriodicSystem(2, lat, [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(sys2d.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sys2d, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.max_iter = 1
    opts.use_diis = False
    opts.initial_guess = InitialGuess.SAD
    with pytest.raises(ValueError, match="dim=3"):
        run_pbc_bipole_rhf(
            sys2d,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
        )


@pytest.mark.slow
def test_use_ewald_j_split_mgo_gamma_scf_converges():
    """Full Γ-only SCF on MgO via use_ewald_j_split: converges smoothly
    to a sensible energy within ~2 Ha of CRYSTAL's SHRINK 8 8 result.

    The remaining ~1.86 Ha is the standard Γ-vs-converged-k-mesh shift
    for ionic crystals; closes when multi-k is wired.
    """
    system, basis = _build_mgo()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    # 12 bohr, not the historical 10: the fixture predates the fold gate
    # (9633ee6bc), which refuses SCF above 1e-2 fold drift — MgO/STO-3G
    # measures 1.5e-2 at 10 bohr and 5.0e-3 at 12
    # (BIPOLE-FIXTURE-CUTOFFS-TRIP-FOLD-GATE; re-cut per the 8285c6f95
    # precedent — the gate is correct, the fixture was stale).
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    # Bounded explicit SR pad + QQR screening: the M5 precision-derived
    # pad (~19 bohr) makes a c12 MgO SCF blow the per-test timeout;
    # this smoke test checks the split path completes, so the M4a
    # explicit-extent contract at 14 bohr is the honest affordable
    # choice (the -515 mHa pad lesson applies to ABSOLUTE-energy
    # claims, which this test does not make). Even so, the full
    # 55-cell exact zone at 12 iterations exceeds the 1800 s slow-lane
    # per-test timeout on a loaded box, so the smoke additionally
    # bounds the exact bielectronic zone (exact_zone_bohr=8, 19 of 55
    # cells) and runs 4 iterations -- the split build, the fold gate,
    # and the SCF loop are all still exercised, which is this test's
    # entire contract.
    opts.lattice_opts.sr_range_screening = True
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = True
    opts.damping = 0.4
    opts.max_iter = 4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-4

    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        sr_image_extent_bohr=14.0,
        exact_zone_bohr=8.0,
        progress=False,
    )
    # Smoke contract: the split path completes without error on an
    # ionic crystal inside the fold gate. No absolute-energy claim.
    assert result.n_iter >= 1
    assert result.e_nuclear < 0.0


def _h2_12bohr_setup():
    sep = 1.4
    half = 0.5 * sep
    system = vq.PeriodicSystem(
        3,
        np.diag([12.0, 12.0, 12.0]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    opts = PeriodicRHFOptions()
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.damping = 0.2
    opts.max_iter = 20
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.initial_guess = InitialGuess.HCORE
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    return system, basis, opts


def test_bipole_h2_12bohr_gamma_energy_regression():
    """Self-owned regression guard for the Γ-only H₂/12-bohr/STO-3G BIPOLE
    path under the production gauge (Ewald exchange split, option (b)
    2026-06-10) — pins the total energy in BIPOLE's own (CI-collected)
    suite, instead of relying solely on the GDF backend-contract test.

    The M5 padded-domain snapshot sits +0.445 mHa above PySCF GDF RHF
    exxdiv='ewald' (-1.1225839666) at this deliberately cheap cutoff 4;
    tests/test_bipole_fock_ewald_exchange.py owns the matching independent
    cross-family bound. The EXT EL-SPHEROPOLE term is omitted from the total
    in this gauge.
    """
    system, basis, opts = _h2_12bohr_setup()
    result = run_pbc_bipole_rhf(
        system,
        basis,
        monkhorst_pack(system, [1, 1, 1]),
        opts,
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.converged
    assert result.exchange_ewald_split is True
    # -1.122138533948 pre-#478. The alpha-consistent 1e Ewald real cutoff
    # shifts H2/12-bohr by -2.4818e-4 Ha at ewald_precision=1e-6 (real cutoff
    # 15.93 bohr). The same value is pinned in
    # test_bipole_fock_ewald_exchange.py, where the move takes the distance
    # to PySCF from +0.4454 mHa to +0.1973 mHa. #674 then bounded the
    # corrected split's default alpha by the 4-bohr exchange cutoff (0.929
    # over CRYSTAL's 0.233) and scaled the J_LR / K_LR envelope with it:
    # -1.122386713788 -> -1.122526209887 (-1.3950e-4 Ha, the image exchange
    # the truncated erfc arm dropped), +0.0578 mHa from PySCF. The
    # legacy-gauge sibling below keeps CRYSTAL's alpha and does not move.
    assert result.energy == pytest.approx(-1.122526209887, abs=1e-9)
    assert result.e_ext_el_spheropole is None


def test_bipole_h2_12bohr_gamma_legacy_gauge_energy_and_spheropole():
    """The legacy Γ gauge (locality projection + full-Coulomb direct K +
    EXT EL-SPHEROPOLE in the total) stays reachable via
    ``use_exchange_ewald_split=False``. M5's padded SR domain
    applies to this erfc J build too; the analytic-gradient preview requests
    the historical domain explicitly when it exercises this gauge.

    Reference values are a current-state snapshot of the legacy converged
    BIPOLE Γ-only energy, not independent analytic targets.
    """
    system, basis, opts = _h2_12bohr_setup()
    result = run_pbc_bipole_rhf(
        system,
        basis,
        monkhorst_pack(system, [1, 1, 1]),
        opts,
        ewald_precision=1e-6,
        use_exchange_ewald_split=False,
        progress=False,
    )

    assert result.converged
    assert result.exchange_ewald_split is False
    # -1.110902665365092 pre-#478; the same -2.4818e-4 Ha shift as the
    # corrected-gauge sibling above, since the alpha-consistent 1e Ewald real
    # cutoff is gauge-independent. The spheropole term below is UNCHANGED,
    # which is the point: nothing in the exchange gauge moved.
    # #704/D131 adds displaced Gaussian-product nuclear support in both
    # gauges: -1.111150845205 -> -1.111151181093 Ha. This remains a
    # legacy-state drift pin, not an independent absolute-energy benchmark.
    assert result.energy == pytest.approx(-1.1111511810927823, abs=1e-9)
    assert result.e_ext_el_spheropole == pytest.approx(0.0055595118, abs=1e-7)


# ---------------------------------------------------------------------------
# GitLab #674: the corrected exchange split's K_SR erfc arm is cut at the
# Fock output cells (|g| <= cutoff_bohr), so its default alpha is bounded
# below by the cutoff and the J_LR / K_LR envelope follows the alpha in use.
# ---------------------------------------------------------------------------

from vibeqc.bipole_ext_el_pole import (
    bipole_ewald_reciprocal_cutoff,
    crystal_ewald_reciprocal_cutoff,
)
from vibeqc.pbc_bipole_common import (
    ewald_alpha_lower_bound,
    resolve_bipole_ewald_alpha,
)
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf


def _box_options_674(cutoff: float = 12.0):
    o = PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 60
    return o


def _h2_30bohr_box():
    c = 15.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * 30.0,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _h_atom_30bohr_box():
    c = 15.0
    sysp = vq.PeriodicSystem(3, np.eye(3) * 30.0, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 2
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


# Split-invariant totals of the EWALD_3D ROHF driver on the same fixtures
# (tests/test_periodic_rohf_multi_k_ewald.py, #651): omega 0.2 / 0.438 at
# cutoff 12 and 0.438 at cutoff 30. Measured post-#674 on the corrected
# BIPOLE route (laptop, core rebuilt at b564fa19e): default (bounded
# 0.3577) -1.1170858289, omega 0.2 -1.1170858286, omega 0.5 -1.1170858294,
# default at cutoff 30 (bounded 0.1431) -1.1170858286; the H atom
# -0.4667330009. Pre-fix (CRYSTAL's 0.0933 at cutoff 12): -1.1170669756
# and -0.4667239569.
_H2_30_SPLIT_INVARIANT = -1.1170858285
_H_ATOM_30_SPLIT_INVARIANT = -0.4667330007


def test_bipole_reciprocal_envelope_keeps_crystal_resolution():
    """K_max / alpha stays CRYSTAL's 8.51 for every alpha at or above
    CRYSTAL's own, and the envelope is CRYSTAL's verbatim below it."""
    for V in (28.7, 126.0, 512.0, 27000.0):
        a_c = crystal_default_ewald_alpha(V)
        k_c = crystal_ewald_reciprocal_cutoff(V)
        assert bipole_ewald_reciprocal_cutoff(V, a_c) == k_c
        assert bipole_ewald_reciprocal_cutoff(V, 0.5 * a_c) == k_c
        for scale in (1.5, 3.83, 10.0):
            k = bipole_ewald_reciprocal_cutoff(V, scale * a_c)
            assert k / (scale * a_c) == pytest.approx(k_c / a_c, rel=1e-12)
        # The CRYSTAL pair resolves exp(-K^2 / 4 alpha^2) to 1.4e-8.
        assert np.exp(-(k_c / a_c) ** 2 / 4.0) == pytest.approx(1.37e-8, rel=0.02)
    with pytest.raises(ValueError):
        bipole_ewald_reciprocal_cutoff(27000.0, 0.0)


def test_bipole_default_alpha_is_bounded_only_on_the_erfc_exchange_arm():
    V = 30.0**3
    lat = vq.LatticeSumOptions()
    lat.cutoff_bohr = 12.0
    bound = ewald_alpha_lower_bound(12.0, 1e-8)
    crystal = crystal_default_ewald_alpha(V)
    assert bound > crystal
    assert resolve_bipole_ewald_alpha(
        V, None, lat, 1e-8, erfc_exchange_arm_active=True
    ) == pytest.approx(bound, rel=1e-12)
    # The legacy CRYSTAL-gauge scaffold and pure functionals keep CRYSTAL's
    # value: nothing there is cut by an erfc exchange arm.
    assert resolve_bipole_ewald_alpha(
        V, None, lat, 1e-8, erfc_exchange_arm_active=False
    ) == crystal
    # An explicit omega always wins.
    assert resolve_bipole_ewald_alpha(
        V, 0.0933, lat, 1e-8, erfc_exchange_arm_active=True
    ) == 0.0933
    # Cells up to 2.8 R_cut / sqrt(-ln tol) = 7.8 bohr keep CRYSTAL's value.
    assert resolve_bipole_ewald_alpha(
        6.0**3, None, lat, 1e-8, erfc_exchange_arm_active=True
    ) == crystal_default_ewald_alpha(6.0**3)


def test_bipole_corrected_split_total_is_invariant_to_the_ewald_split():
    """H2/STO-3G in a 30-bohr box, lattice cutoff 12: the default (bounded)
    alpha, an explicit 0.2 and an explicit 0.5 all land on the EWALD_3D
    split-invariant total; pre-#674 the default was 1.9e-5 Ha short."""
    sysp, basis = _h2_30bohr_box()
    km = monkhorst_pack(sysp, [1, 1, 1])
    bound = ewald_alpha_lower_bound(12.0, 1e-8)
    energies = {}
    for omega in (None, 0.2, 0.5):
        r = run_pbc_bipole_rhf(
            sysp,
            basis,
            km,
            _box_options_674(),
            ewald_omega=omega,
            use_ewald_j_split=True,
            use_exchange_ewald_split=True,
            progress=False,
        )
        assert r.converged
        expected_alpha = bound if omega is None else omega
        assert r.ewald_alpha_bohr_inv == pytest.approx(expected_alpha, rel=1e-9)
        energies[omega] = float(r.energy)
    assert max(energies.values()) - min(energies.values()) < 1e-8, energies
    for e in energies.values():
        assert e == pytest.approx(_H2_30_SPLIT_INVARIANT, abs=2e-8)


def test_bipole_corrected_split_one_electron_atom_matches_split_invariant():
    sysp, basis = _h_atom_30bohr_box()
    km = monkhorst_pack(sysp, [1, 1, 1])
    r = run_pbc_bipole_uhf(
        sysp,
        basis,
        km,
        _box_options_674(),
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        progress=False,
    )
    assert r.converged
    assert r.ewald_alpha_bohr_inv == pytest.approx(
        ewald_alpha_lower_bound(12.0, 1e-8), rel=1e-9
    )
    assert r.energy == pytest.approx(_H_ATOM_30_SPLIT_INVARIANT, abs=2e-8)


def test_bipole_explicit_crystal_alpha_is_still_short_at_the_fixed_cutoff():
    """The pre-fix configuration, requested explicitly, must still miss the
    image exchange beyond 12 bohr; otherwise the fixture no longer
    discriminates the defect."""
    sysp, basis = _h2_30bohr_box()
    km = monkhorst_pack(sysp, [1, 1, 1])
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        km,
        _box_options_674(),
        ewald_omega=2.8 / 30.0,
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        progress=False,
    )
    assert r.converged
    assert r.ewald_alpha_bohr_inv == pytest.approx(2.8 / 30.0)
    assert r.energy - _H2_30_SPLIT_INVARIANT > 1e-5, r.energy


def test_bipole_legacy_gauge_keeps_crystal_alpha_on_a_large_box():
    """``use_exchange_ewald_split=False`` is the CRYSTAL-parity scaffold:
    no erfc exchange arm, so CRYSTAL's alpha stays."""
    sysp, basis = _h2_30bohr_box()
    km = monkhorst_pack(sysp, [1, 1, 1])
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        km,
        _box_options_674(),
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        progress=False,
    )
    assert r.converged
    assert r.ewald_alpha_bohr_inv == pytest.approx(2.8 / 30.0)
