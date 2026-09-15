"""Phase 15c-2: multi-k periodic RKS SCF driver using EWALD_3D.

DFT counterpart of test_periodic_rhf_multi_k_ewald.py with the same
contracts and one extra (hybrid functional). Pin:

  1. Convergence on H₂ / PBE / STO-3G at [1,1,1] mesh.
  2. The [1,1,1] multi-k driver matches the Γ-only RKS Ewald driver
     to ~µHa — they're the same SCF in different containers.
  3. Hybrid-functional path (B3LYP) is wired through the new
     ``exchange_scale`` argument on
     :func:`build_periodic_fock_ewald3d_k`.
  4. ω-invariance at [1,1,1] mesh.
  5. Result-shape sanity (per-k arrays, scf_trace, KS energy
     decomposition).
  6. Skew-cell FFT metric support and open-shell rejection.
  7. Dispatcher routing: ``run_rks_periodic_scf`` with EWALD_3D and
     a [2,1,1] mesh now lands here (was rejected before 15c-2).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_rks_multi_k_ewald import (
    _estimate_legacy_periodic_xc_cache_bytes,
    _guard_legacy_periodic_xc_memory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_in_box(box: float = 30.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options(
    functional: str = "PBE", iter_limit: int = 60, damping: float = 0.3
):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    opts.max_iter = iter_limit
    opts.use_diis = True
    return opts


def test_legacy_periodic_xc_memory_guard_estimates_gga_cache(monkeypatch):
    """MgO/pob-TZVP-scale GGA input should fail before the OS OOM-kills it."""
    estimate = _estimate_legacy_periodic_xc_cache_bytes(
        n_grid=367_200,
        nbf=148,
        n_cells=19,
        is_gga=True,
        n_threads=16,
    )
    assert estimate > 75 * 1024**3

    monkeypatch.setenv("VIBEQC_PERIODIC_XC_MAX_ESTIMATED_GB", "1")
    monkeypatch.delenv("VIBEQC_ALLOW_LEGACY_PERIODIC_XC_OOM", raising=False)
    with pytest.raises(MemoryError, match="legacy periodic-XC"):
        _guard_legacy_periodic_xc_memory(
            n_grid=367_200,
            nbf=148,
            n_cells=19,
            is_gga=True,
            functional="pbe",
        )


def test_legacy_periodic_xc_memory_guard_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VIBEQC_PERIODIC_XC_MAX_ESTIMATED_GB", "1")
    monkeypatch.setenv("VIBEQC_ALLOW_LEGACY_PERIODIC_XC_OOM", "1")
    _guard_legacy_periodic_xc_memory(
        n_grid=367_200,
        nbf=148,
        n_cells=19,
        is_gga=True,
        functional="pbe",
    )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_h2_pbe_converges_at_111_mesh():
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("PBE")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts, omega=0.5)
    assert r.converged
    assert r.n_iter <= opts.max_iter
    # H₂ / PBE / STO-3G in 30-bohr box ≈ -1.34 Ha (after MP shift on
    # the standard truncated-nuclear convention).
    assert -2.0 < r.energy < -0.8


# ---------------------------------------------------------------------------
# Multi-k vs Γ-only equivalence at [1,1,1] mesh
# ---------------------------------------------------------------------------


def test_multi_k_at_gamma_mesh_matches_gamma_driver():
    """[1,1,1] mesh through the multi-k driver should be bit-equivalent
    to the Γ-only driver — same SCF, different machinery."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("PBE")

    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_multi = vq.run_rks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
    )
    r_gamma = vq.run_rks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
    )
    assert r_multi.converged and r_gamma.converged
    # Both drivers now force EWALD_3D (handover F4, 2026-06-01), so V_ne /
    # e_nuclear are gauge-consistent with the Hartree J and the v0.6.1 Madelung
    # correction is disabled on both paths (E_madelung_fix=0.0 for EWALD_3D).
    # They are therefore bit-equivalent at [1,1,1] — this was a ~0.7 mHa
    # Madelung gap before the multi-k driver was migrated to force the gauge.
    assert abs(r_multi.energy - r_gamma.energy) < 1e-10
    assert abs(r_multi.e_xc - r_gamma.e_xc) < 1e-10


# ---------------------------------------------------------------------------
# Hybrid path (alpha != 0)
# ---------------------------------------------------------------------------


def test_b3lyp_path_uses_hf_exchange():
    """B3LYP carries 20 % HF exchange, so ``e_hf_exchange`` must be live."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("B3LYP")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts, omega=0.5)
    assert r.converged
    assert abs(r.e_hf_exchange) > 1e-3, (
        f"e_hf_exchange ≈ 0 ({r.e_hf_exchange:.3e}) but B3LYP α = 0.2 — "
        "the full-range exchange arm may not be wired."
    )


def test_hybrid_multi_k_uses_the_ewald_exxdiv_gauge_not_the_gamma_one():
    """The full-range arm rides the corrected Ewald split, so multi-k no
    longer reproduces the Γ driver bit-for-bit.

    The Γ driver (``run_rks_periodic_gamma_ewald3d``) sums the *bare*
    ``1/r`` exchange over its image ball. That is legitimate at Γ, where
    the molecular-limit density decays with image distance, and it lands
    on a truncated gauge carrying no q -> 0 correction. On a finite k
    mesh the density is Born-von-Kármán-torus periodic, that same sum
    diverges, and the multi-k driver must instead use the Ewald split
    with the probe-charge ``q + G = 0`` term — the ``exxdiv='ewald'``
    convention.

    The two are therefore different gauges by construction and differ by
    a Madelung-scale amount. This test pins that they differ *and* that
    the multi-k side is the one agreeing with the external reference; it
    replaces an older assertion that pinned them equal to 1e-10, which
    encoded the pre-fix bare-sum behaviour.
    """
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("B3LYP")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts, omega=0.5)
    r_gamma = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged and r_gamma.converged
    # Not zero, and specifically not float noise. The threshold is well
    # below the measured ~74 uHa shift on this fixture: the shift is small
    # here only because a 30-bohr box at a 12-bohr cutoff holds a single
    # lattice cell, which is exactly why this fixture could never have
    # caught the divergence itself. Do not raise it to 1e-4.
    assert abs(r.energy - r_gamma.energy) > 1e-5
    # A pure functional builds no full-range arm, so the two gauges
    # coincide there — that parity is pinned by
    # test_multi_k_at_gamma_mesh_matches_gamma_driver and must not move.


def test_hybrid_exchange_is_cutoff_independent():
    """The full-range exchange sum must converge in the lattice cutoff.

    This is the regression for the divergent bare-``1/r`` image sum: with
    a Born-von-Kármán-torus-periodic density the old sum fell 177 mHa
    going from a 12- to a 20-bohr cutoff on this cell. The corrected
    split's short-range arm is ``erfc(alpha r)/r``, so it converges, and
    the reciprocal / ``q + G = 0`` arms are cutoff-independent by
    construction.

    The cell is deliberately small enough that ``direct_lattice_cells``
    returns more than one cell at every cutoff below — a 30-bohr box at
    cutoff 12 yields a single cell, which makes any image-sum assertion
    structurally blind.
    """
    box = 12.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    energies = []
    for cutoff in (12.0, 20.0):
        opts = vq.PeriodicKSOptions()
        opts.functional = "PBE0"
        opts.lattice_opts.cutoff_bohr = cutoff
        opts.lattice_opts.nuclear_cutoff_bohr = cutoff
        opts.damping = 0.3
        opts.max_iter = 200
        opts.conv_tol_energy = 1e-10
        opts.conv_tol_grad = 1e-7
        r = vq.run_rks_periodic_multi_k_ewald3d(
            sysp, basis, vq.monkhorst_pack(sysp, [2, 1, 1]), opts,
            auto_optimize_truncation=False,
        )
        assert r.converged
        assert len(r.density.cells) > 1, (
            f"cutoff {cutoff} gave a single-cell image ball; this test "
            "cannot see an image-sum divergence with one cell"
        )
        energies.append(r.energy)
    spread = abs(energies[1] - energies[0])
    assert spread < 1e-5, (
        f"full-range exchange is not cutoff-converged: {energies}, "
        f"spread = {spread:.3e} Ha"
    )

    # Published target: PySCF 2.14.0 KRKS.density_fit(), xc='pbe0',
    # exxdiv='ewald', H2 in a 12-bohr cube (atoms at z = c ± 0.7 bohr),
    # sto-3g, Γ-centred (2,1,1) mesh: -1.155219229638 Ha. The same run
    # with exxdiv=None gives -1.117597525253 Ha, so this also pins that
    # the q -> 0 correction is applied rather than omitted.
    E_PYSCF_KRKS_PBE0_211 = -1.155219229638
    assert energies[0] == pytest.approx(E_PYSCF_KRKS_PBE0_211, abs=2e-4)


def test_hybrid_multi_k_matches_the_bipole_corrected_route():
    """Independent cross-check: the BIPOLE route implements the same split."""
    box = 12.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])

    def _opts():
        o = vq.PeriodicKSOptions()
        o.functional = "PBE0"
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 12.0
        o.damping = 0.3
        o.max_iter = 200
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-7
        return o

    r = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(), auto_optimize_truncation=False
    )
    r_bipole = vq.run_pbc_bipole_rks(
        sysp, basis, kmesh, _opts(),
        functional="PBE0",
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        exchange_exxdiv="ewald",
        progress=False,
    )
    assert r.converged and r_bipole.converged
    assert r.energy == pytest.approx(r_bipole.energy, abs=5e-5)


def test_hybrid_fails_closed_on_a_symmetry_reduced_mesh():
    """No silent fallback to the divergent sum when the split cannot serve.

    The Born-von-Kármán torus density needs a full uniform-weight
    Monkhorst-Pack mesh; an explicit / symmetry-reduced k list does not
    qualify, and a hybrid must refuse rather than sum a divergent series.
    """
    box = 12.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    full = vq.monkhorst_pack(sysp, [2, 2, 1])
    # An explicit k list: carries no mesh dimensions, so prod(mesh) = 1
    # while n_k = 2 -- the shape a symmetry-reduced (IBZ) mesh also has.
    ks = np.asarray([np.asarray(k, dtype=float) for k in list(full.kpoints)[:2]])
    reduced = vq.as_bloch_kmesh(
        vq.KPoints(
            kpoints_cart=ks,
            kpoints_frac=np.zeros_like(ks),
            weights=np.array([0.5, 0.5]),
        )
    )
    opts = _default_options("PBE0")
    with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
        vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, reduced, opts)


# ---------------------------------------------------------------------------
# ω-invariance
# ---------------------------------------------------------------------------


def test_omega_invariance_at_111_mesh():
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("PBE")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = vq.run_rks_periodic_multi_k_ewald3d(
            sysp,
            basis,
            kmesh,
            opts,
            omega=omega,
            spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)
    spread = max(energies) - min(energies)
    assert spread < 0.005 * abs(min(energies)), (
        f"ω-invariance violated at multi-k=[1,1,1]: {energies}, spread = {spread:.3e}"
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_has_expected_per_k_shapes():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts, omega=0.5)
    n_k = len(kmesh.kpoints)
    n_bf = basis.nbasis
    assert len(r.fock) == n_k
    assert len(r.overlap) == n_k
    assert len(r.hcore) == n_k
    assert len(r.mo_energies) == n_k
    assert len(r.mo_coeffs) == n_k
    for k_idx in range(n_k):
        assert r.fock[k_idx].shape == (n_bf, n_bf)
        assert r.overlap[k_idx].shape == (n_bf, n_bf)
        assert r.hcore[k_idx].shape == (n_bf, n_bf)
    assert len(r.scf_trace) >= 1
    first = r.scf_trace[0]
    assert isinstance(first, vq.SCFIteration)
    assert isinstance(first.energy, float)
    assert r.functional.upper().startswith("PBE")
    # ω is auto-derived from nuclear_cutoff_bohr to match the C++ Ewald α
    # (see periodic_rks_ewald.py / 49f8ae91); the driver ``omega`` kwarg
    # is overridden so the jellium background cancels exactly.
    assert r.omega == pytest.approx(0.5)


def test_energy_decomposition_consistent():
    """E_total = e_electronic + e_nuclear for the EWALD_3D gauge.

    The multi-k RKS Ewald driver now forces EWALD_3D (handover F4,
    2026-06-01), so V_ne / e_nuclear are Ewald-gauge-consistent with the
    Hartree J and the v0.6.1 Madelung leak correction is disabled
    (``madelung_energy_correction_for_lat`` returns 0.0 for EWALD_3D).
    Mirrors the RKS Γ decomposition test (test_periodic_rks_ewald.py).
    """
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts, omega=0.5)
    assert r.converged
    # For EWALD_3D: E_total = E_elec + E_nuc (no Madelung fix).
    assert abs(r.energy - (r.e_electronic + r.e_nuclear)) < 1e-10
    # PBE α = 0; HF exchange must vanish.
    assert abs(r.e_hf_exchange) < 1e-12


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_accepts_non_orthorhombic_lattice_smoke():
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
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
        grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)


def test_rejects_open_shell_system():
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
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    with pytest.raises(ValueError, match="closed-shell"):
        vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts)


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatcher_multi_k_ewald_routes_to_new_driver():
    """Dense k-mesh through ``run_rks_periodic_scf`` with EWALD_3D
    must dispatch to the new multi-k driver — was a
    NotImplementedError before 15c-2."""
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rks_periodic_scf(sysp, basis, kmesh, opts, omega=0.5)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSMultiKEwaldResult)


def test_dispatcher_gamma_mesh_still_routes_to_gamma_driver():
    """[1,1,1] through the multi-k entry must keep delegating to the
    cheaper Γ-only driver (PeriodicRKSEwaldResult, not the multi-k
    result type)."""
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_scf(sysp, basis, kmesh, opts, omega=0.5)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSEwaldResult)


def test_band_overlap_t0_mesh_converges_with_global_fill_and_mixer():
    """#509: a band-overlap T = 0 mesh used to fail closed on this route
    (the fixed C[:, :n_occ] slice cannot express the global fill's per-k
    varying counts). The occupation-driven builder plus the metal density
    mixer must converge it instead of refusing."""
    from vibeqc.smearing import (
        occupations_are_per_k_integer_aufbau as _occupations_are_per_k_integer_aufbau,
    )

    sysp = vq.PeriodicSystem(3, 5.0 * np.eye(3), [vq.Atom(12, [0, 0, 0])], 0, 1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _default_options("PBE")
    opts.max_iter = 60
    opts.conv_tol_grad = 1e-5
    r = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, density_mixer="anderson", progress=False)
    assert r.converged
    assert np.isfinite(r.energy)
    res = vq.apply_smearing(
        r.mo_energies,
        weights=np.asarray(km.weights, dtype=float),
        n_electrons_per_cell=12.0,
        n_occ_each=6,
        smearing=None,
    )
    assert not _occupations_are_per_k_integer_aufbau(
        res.occupations_per_k, 6
    ), "the converged occupations collapsed to the per-k integer pattern; " \
       "the band-overlap fill was not exercised"
