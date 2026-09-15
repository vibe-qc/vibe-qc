"""Phase 12e-c-4c-iii-b tests: multi-k Ewald SCF driver.

Contracts exercised:

1. **Basic convergence** — on an isolated H2 in a big box the driver
   converges and the result structures have the right shapes
   (n_k per-k MOs, density as a LatticeMatrixSet).

2. **[1,1,1] mesh equals Γ-only driver** — at a single k-point at Γ,
   the inverse-Bloch fold replicates D(0) into every cell which is the
   molecular-limit convention used by
   ``run_rhf_periodic_gamma_ewald3d``; the two energies must agree to
   ~µHa.

3. **MO orthonormality** — ``C(k)^† S(k) C(k) = I`` at every k (per-k
   canonical-orth SCF contract).

4. **Hermiticity of converged F(k)** — each F(k) in the result is
   Hermitian.

5. **ω-invariance** — total SCF energy is ω-independent to the
   12e-c-4a/4c-iii-a ~0.3–1 % bound.

6. **[2,2,2] mesh converges** — exercises the non-trivial Bloch-phase
   bookkeeping with multiple k-points and a multi-cell D_real.

7. **Result trace and counters** — ``scf_trace`` has one entry per
   iteration and ``n_iter > 0`` when converged.

8. **Multi-k DIIS** — Pulay extrapolation on the per-k Fock list with
   k-weighted error inner products. Activated by ``use_diis=True``;
   reaches the same energy as plain damping and accelerates convergence.

Metallic-Fermi occupation and cross-driver validation against a
published bulk-HF reference are deferred to follow-up commits.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq

_mk_driver = vq.run_rhf_periodic_multi_k_ewald3d
_gamma_driver = vq.run_rhf_periodic_gamma_ewald3d


def _run_mk(*args, **kwargs):
    """``run_rhf_periodic_multi_k_ewald3d`` with the truncation
    auto-optimiser pinned OFF (defensive no-op on these dilute boxes).

    Every test here asserts a cutoff-independent contract (per-k
    S-orthonormality, F(k) hermiticity, ω-invariance, DIIS-vs-damping
    energy agreement, [1,1,1]==Γ, trace bookkeeping). The file was
    de-slowed (≈596→208 s CPU) by two contract-preserving levers:
    ``use_diis`` (already the default — few iterations, so few
    per-iteration analytic-FT J rebuilds, the dominant cost / PBC-audit
    E2), and dropping the gratuitous ``cutoff=35`` on the [2,2,2]
    algebraic tests to the default 12 (orthonormality/hermiticity/
    finiteness are cutoff-independent; cutoff 12 = home cell only on the
    30-bohr box, ~27× fewer real-space ERI cells than cutoff 35).
    auto_optimize_truncation was measured *not* to matter here.
    """
    kwargs.setdefault("auto_optimize_truncation", False)
    return _mk_driver(*args, **kwargs)


def _run_g(*args, **kwargs):
    """``run_rhf_periodic_gamma_ewald3d`` (Γ reference for the
    [1,1,1]==Γ contracts) with the auto-optimiser pinned OFF."""
    kwargs.setdefault("auto_optimize_truncation", False)
    return _gamma_driver(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _opts(cutoff: float = 12.0, damping: float = 0.3, max_iter: int = 40):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = max(cutoff, 15.0)
    opts.damping = damping
    opts.max_iter = max_iter
    # DIIS converges these multi-k Ewald SCFs in a few iterations instead
    # of dozens (same fixed point), so the per-iteration analytic-FT J
    # rebuild — the dominant cost — is paid only a few times.
    opts.use_diis = True
    return opts


# ---------------------------------------------------------------------------
# Basic convergence + structural shape contracts
# ---------------------------------------------------------------------------

def test_driver_converges_and_result_shapes_are_consistent():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = _run_mk(
        sysp, basis, km, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert r.n_iter > 0
    n_k = len(km.kpoints)
    nbf = basis.nbasis
    assert len(r.mo_energies) == n_k
    assert len(r.mo_coeffs) == n_k
    assert len(r.fock) == n_k
    assert len(r.overlap) == n_k
    assert len(r.hcore) == n_k
    for idx in range(n_k):
        assert r.mo_coeffs[idx].shape[0] == nbf
        assert r.fock[idx].shape == (nbf, nbf)
        assert r.overlap[idx].shape == (nbf, nbf)
        assert r.hcore[idx].shape == (nbf, nbf)
        assert np.isfinite(r.mo_energies[idx]).all()
    assert r.density is not None
    assert len(r.density.cells) >= 1


# ---------------------------------------------------------------------------
# [1,1,1] mesh reduces to the Γ-only Ewald driver
# ---------------------------------------------------------------------------

def test_single_k_mesh_matches_gamma_ewald_driver():
    """With a [1,1,1] mesh the inverse-Bloch fold puts D(0) into every
    cell — the same molecular-limit convention used by
    ``run_rhf_periodic_gamma_ewald3d``. Both drivers solve the same
    SCF and must reach the same energy."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_mk = _run_mk(
        sysp, basis, km, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    r_g = _run_g(
        sysp, basis, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    assert r_mk.converged and r_g.converged
    assert r_mk.energy == pytest.approx(r_g.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# MO orthonormality per k
# ---------------------------------------------------------------------------

def test_mo_coeffs_are_s_k_orthonormal_at_every_k():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    r = _run_mk(
        sysp, basis, km, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    for idx in range(len(km.kpoints)):
        C = r.mo_coeffs[idx]
        S = r.overlap[idx]
        M = C.conj().T @ S @ C
        err = np.linalg.norm(M - np.eye(M.shape[0]))
        assert err < 1e-10, (
            f"C(k[{idx}]) not S(k)-orthonormal: ‖C†SC − I‖ = {err:.3e}"
        )


# ---------------------------------------------------------------------------
# F(k) Hermitian on the converged result
# ---------------------------------------------------------------------------

def test_converged_fock_is_hermitian_at_every_k():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    r = _run_mk(
        sysp, basis, km, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    for idx, F_k in enumerate(r.fock):
        herm = np.abs(F_k - F_k.conj().T).max()
        assert herm < 1e-9, (
            f"F(k[{idx}]) not Hermitian at convergence: "
            f"‖F − F†‖_∞ = {herm:.3e}"
        )


# ---------------------------------------------------------------------------
# ω-invariance of the total SCF energy
# ---------------------------------------------------------------------------

def test_total_energy_is_omega_invariant():
    """Total SCF energy is ω-independent up to the 12e-c-4a Makov-Payne
    residual (~0.3 %) plus the multi-cell propagation cushion
    exercised in 12e-c-4c-iii-a (~1 %)."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    energies = []
    for omega in (0.3, 0.5, 1.0):
        r = _run_mk(
            sysp, basis, km, _opts(), omega=omega, spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)
    spread = max(energies) - min(energies)
    ref = abs(energies[0])
    rel = spread / ref
    assert rel < 1e-2, (
        f"ω-spread of total energy too large: {spread:.3e} Ha "
        f"(rel = {rel:.3e})"
    )


# ---------------------------------------------------------------------------
# [2,2,2] mesh: multi-k SCF actually iterates and closes
# ---------------------------------------------------------------------------

def test_222_mesh_converges_and_produces_real_energy():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    r = _run_mk(
        sysp, basis, km, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    # Energy is a real finite number.
    assert np.isfinite(r.energy)
    assert isinstance(r.energy, float)
    # Per-k ε come out real.
    for eps_k in r.mo_energies:
        assert eps_k.dtype.kind == "f"   # real
        assert np.isfinite(eps_k).all()


# ---------------------------------------------------------------------------
# Scf trace bookkeeping
# ---------------------------------------------------------------------------

def test_scf_trace_has_one_entry_per_iteration():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = _run_mk(
        sysp, basis, km, _opts(), omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert len(r.scf_trace) == r.n_iter
    # First entry has iter = 1.
    assert r.scf_trace[0].iter == 1
    # Last entry's iter == n_iter.
    assert r.scf_trace[-1].iter == r.n_iter
    # All entries are SCFIteration instances (unified across Ewald +
    # DIRECT_TRUNCATED backends).
    assert all(isinstance(it, vq.SCFIteration) for it in r.scf_trace)


# ---------------------------------------------------------------------------
# Multi-k DIIS
# ---------------------------------------------------------------------------

def _h2_chain_tight(a: float = 10.0):
    """Periodic H₂ chain in a cubic box. Tight enough that multi-k SCF
    benefits from DIIS; loose enough that both damping and DIIS
    converge under reasonable tolerances."""
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _diis_options(use_diis: bool):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 15
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.damping = 0.5
    opts.max_iter = 60
    opts.use_diis = use_diis
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    # Multi-k SCF plateaus its commutator gradient at the
    # k-degenerate-orbital level (≈ 1e-6 on a single-cell H₂ chain
    # at [2,2,2] mesh) — slightly looser than the molecular default
    # so the "energy converged" SCFs aren't reported as oscillating.
    opts.conv_tol_grad = 1e-5
    return opts


def test_diis_and_damping_reach_same_energy_multi_k():
    """DIIS is an accelerator, not a solver change — multi-k DIIS
    must converge to the same energy as plain damping."""
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    r_damp = _run_mk(
        sysp, basis, km, _diis_options(use_diis=False),
        omega=0.5, spacing_bohr=0.3,
    )
    r_diis = _run_mk(
        sysp, basis, km, _diis_options(use_diis=True),
        omega=0.5, spacing_bohr=0.3,
    )
    assert r_damp.converged and r_diis.converged
    # Both paths run to ``conv_tol_grad = 1e-5`` (cf. ``_diis_options``);
    # at that gradient tolerance the energy is converged to ≈ 1e-6 Ha.
    # With Schwarz screening (default 1e-12 Ha per integral) the per-
    # iteration noise floor is well below that, but DIIS and damping
    # stop at slightly different SCF iterates so the two energies can
    # legitimately disagree by ~1e-6. 1e-7 was the pre-screening lucky
    # bit-reproducibility — the contract here is "same energy at
    # convergence", not "bit-identical SCF trajectory".
    assert r_damp.energy == pytest.approx(r_diis.energy, abs=1e-6)


def test_multi_k_diis_does_not_increase_iteration_count():
    """Multi-k DIIS should not need *more* iterations than plain
    damping on a tractable case. Equal-or-fewer is the contract;
    on tighter cells DIIS gives a meaningful speedup, but we only
    require non-regression here so the test is robust."""
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    r_damp = _run_mk(
        sysp, basis, km, _diis_options(use_diis=False),
        omega=0.5, spacing_bohr=0.3,
    )
    r_diis = _run_mk(
        sysp, basis, km, _diis_options(use_diis=True),
        omega=0.5, spacing_bohr=0.3,
    )
    assert r_damp.converged and r_diis.converged
    assert r_diis.n_iter <= r_damp.n_iter, (
        f"DIIS regressed: {r_diis.n_iter} iters vs {r_damp.n_iter} damped"
    )


def test_multi_k_diis_off_disables_extrapolation():
    """``use_diis=False`` must reproduce the pure-damping path —
    iteration count and energy match a baseline run with the same
    damping."""
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts_off = _diis_options(use_diis=False)
    opts_late = _diis_options(use_diis=True)
    opts_late.diis_start_iter = 1_000_000   # effectively disabled
    r_off = _run_mk(
        sysp, basis, km, opts_off, omega=0.5, spacing_bohr=0.3,
    )
    r_late = _run_mk(
        sysp, basis, km, opts_late, omega=0.5, spacing_bohr=0.3,
    )
    assert r_off.converged and r_late.converged
    assert r_off.n_iter == r_late.n_iter
    assert r_off.energy == pytest.approx(r_late.energy, abs=1e-10)


def test_single_gamma_multi_k_matches_gamma_driver_on_tight_cell():
    """The [1,1,1] multi-k path must use the Γ molecular-limit Fock.

    A previous regression fed the one-point inverse-Bloch density into
    the generic multi-cell Fock builder, producing a homogeneous-density
    fixed point near +8 Ha. The dedicated Γ driver is the reference for
    the single-k convention.
    """
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts_mk = _diis_options(use_diis=False)
    opts_g = _diis_options(use_diis=False)
    r_mk = _run_mk(
        sysp, basis, km, opts_mk, omega=0.5, spacing_bohr=0.3,
    )
    r_g = _run_g(
        sysp, basis, opts_g, omega=0.5, spacing_bohr=0.3,
    )
    assert r_mk.converged and r_g.converged
    assert r_mk.scf_trace[-1].grad_norm < opts_mk.conv_tol_grad
    assert r_mk.energy == pytest.approx(r_g.energy, abs=1e-9)


def test_multi_k_diis_handles_near_singular_b_matrix_gracefully():
    """After convergence, successive iterations produce near-identical
    error vectors and the Pulay B matrix goes near-singular. The
    fallback (drop oldest history entry, retry) must keep the SCF
    finite even if we force iterations beyond convergence."""
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _diis_options(use_diis=True)
    opts.max_iter = 30   # cap; plenty to over-converge
    r = _run_mk(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert np.isfinite(r.energy)


def test_band_overlap_t0_mesh_converges_with_the_global_fill():
    """#509: a T = 0 mesh whose bands straddle the Fermi level used to fail
    closed on this route (the fixed C[:, :n_occ] density slice cannot
    represent the global fill's per-k varying occupied counts). The
    occupation-driven builder must let the route converge, and the converged
    occupations must be the global fill, not the legacy per-k pattern."""
    from vibeqc.smearing import (
        occupations_are_per_k_integer_aufbau as _occupations_are_per_k_integer_aufbau,
    )

    sysp = vq.PeriodicSystem(3, 5.0 * np.eye(3), [vq.Atom(12, [0, 0, 0])], 0, 1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _diis_options(use_diis=True)
    opts.max_iter = 60
    r = _run_mk(sysp, basis, km, opts)
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


# ---------------------------------------------------------------------------
# The default Ewald alpha and the real-space image cutoff are one contract
# (GitLab #651)
# ---------------------------------------------------------------------------
#
# The EWALD_3D drivers default omega to CRYSTAL's 2.8 / V^(1/3), which falls
# with the cell volume, while their erfc-screened real-space sums (the K_SR of
# the corrected exchange split, the V_ne short range) are cut at the fixed
# LatticeSumOptions.cutoff_bohr = 15. Past an 8-bohr cube erfc(omega r)/r has
# not decayed by the cutoff and the truncated sum is silently short. Measured
# here before the fix, H2/STO-3G in a 16-bohr cube at a (2,1,1) mesh:
#
#     omega 0.175 (CRYSTAL) / cutoff 15   -1.1180942761 Ha
#     omega 0.175           / cutoff 30   -1.1181372061 Ha   <- converged
#     omega 0.350           / cutoff 15   -1.1181371957 Ha
#     omega 0.500           / cutoff 15   -1.1181371958 Ha
#
# i.e. 4.29e-5 Ha short at the default, 1.05e-8 Ha with alpha bounded below
# at the same cutoff. A lone Gamma point on its default molecular-limit
# exchange gauge is alpha-invariant; with exchange_exxdiv='ewald' it takes
# the same erfc arm and moves too (tests/test_ccm_periodic_3d.py, 6.1e-5
# Ha/cell on the 16-bohr (2,2,2) supercell). J is analytic-FT and V_ne /
# E_nn size their own alpha, so only the exchange arm is exposed.

from vibeqc.pbc_bipole_common import (  # noqa: E402
    default_ewald_alpha,
    ewald_alpha_lower_bound,
    ewald_real_cutoff_for_alpha,
)

_A16 = 16.0
_TOL = 1e-12


def _h2_cube(a: float) -> "vq.PeriodicSystem":
    return vq.PeriodicSystem(
        3, np.diag([a, a, a]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )


_E16_CACHE: dict = {}


def test_default_ewald_alpha_cites_eq_32_at_printed_page_273():
    """The primary Ewald paper prints Eq. (32) and its split discussion
    on p. 273; the PDF OCR misreads that page number as 215 (GitLab #651)."""
    citation = default_ewald_alpha.__doc__ or ""
    assert "Eq. (32), p. 273" in citation
    assert "pp. 214-215" not in citation


def _e16_211(omega: float, cutoff: float) -> tuple[float, float]:
    """(energy, omega used) for the 16-bohr cube at (2,1,1); cached."""
    key = (omega, cutoff)
    if key not in _E16_CACHE:
        unit = _h2_cube(_A16)
        basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = cutoff
        res = _mk_driver(
            unit, basis, vq.monkhorst_pack(unit, [2, 1, 1]),
            omega=omega, options=opts,
        )
        assert res.converged
        _E16_CACHE[key] = (float(res.energy), float(res.omega))
    return _E16_CACHE[key]


def test_default_ewald_alpha_is_crystal_until_the_cutoff_bound_binds():
    """Helper contract: CRYSTAL's value wherever erfc(alpha R_cut) <= tol
    already holds, the pointwise bound sqrt(-ln tol)/R_cut otherwise; the
    bound is the exact inverse of BIPOLE's ewald_real_cutoff_for_alpha."""
    bound = ewald_alpha_lower_bound(15.0, _TOL)
    assert bound == pytest.approx(np.sqrt(-np.log(_TOL)) / 15.0)
    assert ewald_real_cutoff_for_alpha(bound, _TOL) == pytest.approx(15.0)
    # MgO-sized primitive cell: CRYSTAL's 0.5585 already satisfies the bound.
    assert default_ewald_alpha(126.0, real_cutoff_bohr=15.0, tolerance=_TOL) == (
        pytest.approx(2.8 / 126.0 ** (1.0 / 3.0)))
    # 8-bohr cube: CRYSTAL's 0.350 sits 1.2e-3 below the bound, bound wins.
    assert default_ewald_alpha(512.0, real_cutoff_bohr=15.0, tolerance=_TOL) == (
        pytest.approx(bound))
    # 16-bohr cube: CRYSTAL's 0.175 is half the bound.
    assert default_ewald_alpha(4096.0, real_cutoff_bohr=15.0, tolerance=_TOL) == (
        pytest.approx(bound))
    # A wide enough cutoff lets CRYSTAL's value stand again (at 30 bohr the
    # bound is 0.1752, still a hair above 0.175; at 40 bohr it is 0.131).
    assert default_ewald_alpha(4096.0, real_cutoff_bohr=40.0, tolerance=_TOL) == (
        pytest.approx(0.175))
    assert default_ewald_alpha(4096.0, real_cutoff_bohr=30.0, tolerance=_TOL) == (
        pytest.approx(ewald_alpha_lower_bound(30.0, _TOL)))
    with pytest.raises(ValueError, match="real_cutoff_bohr"):
        ewald_alpha_lower_bound(0.0, _TOL)
    with pytest.raises(ValueError, match="tolerance"):
        ewald_alpha_lower_bound(15.0, 1.0)


def test_h2_16bohr_211_default_alpha_matches_the_wide_cutoff_reference():
    """#651 regression: with no omega given, the 16-bohr (2,1,1) run uses the
    bounded alpha and lands within 5e-8 Ha of the 30-bohr-cutoff reference
    (pre-fix: 4.29e-5 Ha short). The driver records the alpha it used."""
    e_ref, _ = _e16_211(0.0, 30.0)
    e_def, omega_used = _e16_211(0.0, 15.0)
    assert omega_used == pytest.approx(ewald_alpha_lower_bound(15.0, _TOL), rel=1e-9)
    assert abs(e_def - e_ref) < 5.0e-8, (e_def, e_ref, e_def - e_ref)
    assert e_ref == pytest.approx(-1.1181372, abs=2e-7)


def test_h2_16bohr_211_unbounded_crystal_alpha_is_short_at_the_default_cutoff():
    """The pre-fix configuration, requested explicitly: CRYSTAL's alpha
    0.175 at the 15-bohr cutoff must still be measurably short of the
    reference, otherwise this fixture no longer discriminates the defect."""
    e_ref, _ = _e16_211(0.0, 30.0)
    e_unbounded, omega_used = _e16_211(0.175, 15.0)
    assert omega_used == pytest.approx(0.175)
    assert e_unbounded - e_ref > 1.0e-5, (e_unbounded, e_ref)


# ---------------------------------------------------------------------------
# #725: MOM declares its fixed occupied subspace to the T = 0 fill
# ---------------------------------------------------------------------------


def test_mom_declares_a_fixed_occupied_subspace_to_the_t0_fill(monkeypatch):
    """The maximum overlap method permutes ``C(k)`` / ``eps(k)`` so its
    selected occupied states lead (Gilbert, Besley, Gill 2008); the ``T = 0``
    fill must occupy that block positionally instead of re-picking the lowest
    states across the mesh, which since #85 undid the selection and made the
    fixed-slice guard refuse every multi-k MOM run (#725). The driver
    therefore passes ``fixed_occupied_subspace=use_mom`` to the shared
    occupation wrapper. Pinned at the first occupation call (right after the
    initial-guess diagonalisation) so the test costs no SCF iteration; the
    semantics of the flag are covered in ``tests/test_smearing_package.py``.
    Pre-fix the wrapper received no such keyword at all."""
    import vibeqc.periodic_rhf_multi_k_ewald as drv

    class _Stop(RuntimeError):
        pass

    calls: list = []

    def _recording(*args, **kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(drv, "_closed_shell_periodic_occupations", _recording)
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    for use_mom in (False, True):
        with pytest.raises(_Stop):
            _run_mk(sysp, basis, km, _opts(), use_mom=use_mom)
        assert calls[-1]["fixed_occupied_subspace"] is use_mom
    assert len(calls) == 2
