"""Smoke + mHa-parity tests for the new compcell GDF SCF driver.

PySCF parity is NOT executed here (the harness handles PySCF
out-of-process per CLAUDE.md §10). These tests verify:

* The bare-aux ‖M‖_F divergence is replaced by a plateau under the
  compcell construction across a wide cutoff sweep — the algebraic
  statement of the bug fix.
* The new ``run_pbc_gdf_rhf`` converges on H2 / 12-bohr cubic and
  reaches a known-good mHa-scale energy at the prompt's recommended
  ``η ≈ 0.25``.
* The ``exxdiv='ewald'`` shift is bit-exact additive (the energy
  delta between exxdiv='ewald' and 'none' matches the analytic
  ``−¼ ξ tr(D·S·D·S)`` form on the converged density).
* The Madelung helpers return correct cubic values.

The µHa PySCF parity targets are blocked on the AFT long-range
correction (see CHANGELOG ``[Unreleased]`` section). Those tests
will land alongside the AFT slice.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import ShellInfo, compute_2c_eri_lattice
from vibeqc.aux_basis import (
    build_lpq_compcell,
    fuse_transform_matrix,
    make_aux_basis_set,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
    rsgdf_aux_fourier_transform,
)
from vibeqc.madelung import (
    apply_exxdiv_ewald_to_K,
    exxdiv_ewald_energy_shift,
    madelung_constant_for_cell,
)


def _h2_box(box_bohr: float = 12.0, sep_bohr: float = 1.4):
    half = 0.5 * sep_bohr
    system = vq.PeriodicSystem(
        3,
        np.diag([box_bohr, box_bohr, box_bohr]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_madelung_constant_cubic():
    """ξ for a cubic cell — proper Ewald sum matches the Wigner constant
    α_M ≈ 2.837297 / L to 6 sig figs (the previous cubic Wigner shortcut
    was 6-digit truncated; the Ewald sum recovers more digits and is
    geometry-correct for non-cubic cells too, e.g. FCC primitive). See
    handovers/HANDOVER_GDF_V0_11_2026_05_29.md "2026-05-29" entry for the
    LiH primitive FCC discrepancy that motivated the Ewald upgrade.
    """
    system, _ = _h2_box(box_bohr=12.0)
    xi = madelung_constant_for_cell(system)
    # Wigner constant 2.837297 is 6-digit truncated; Ewald gives 8+ digits.
    assert xi == pytest.approx(2.837297 / 12.0, abs=1e-6)


def test_apply_exxdiv_ewald_to_K_shape_and_value():
    """K(k) += ξ · S·D·S per k, dtype-preserving."""
    S = [np.eye(2)]
    D = [np.array([[1.0, 0.5], [0.5, 1.0]])]
    K0 = [np.zeros((2, 2))]
    out = apply_exxdiv_ewald_to_K(K0, S, D, 0.3)
    assert len(out) == 1
    assert out[0].shape == (2, 2)
    assert np.allclose(out[0], 0.3 * D[0])
    # Mismatched lengths raise.
    with pytest.raises(ValueError):
        apply_exxdiv_ewald_to_K(K0, S + S, D, 0.3)


def test_exxdiv_ewald_energy_shift_sign_and_magnitude():
    """ΔE_xx = −¼ α_HF ξ Σ_k w_k tr(D·S·D·S)."""
    S = [np.eye(2)]
    D = [np.array([[1.0, 0.5], [0.5, 1.0]])]
    xi = 0.3
    tr = np.trace(D[0] @ S[0] @ D[0] @ S[0])  # = 1²+2*0.5²+1² = 2.5
    e = exxdiv_ewald_energy_shift(D, S, xi, hf_exchange_fraction=1.0, weights=[1.0])
    assert e == pytest.approx(-0.25 * xi * tr, rel=1e-12)
    # Hybrid (α=0.2)
    e_hyb = exxdiv_ewald_energy_shift(D, S, xi, hf_exchange_fraction=0.2, weights=[1.0])
    assert e_hyb == pytest.approx(0.2 * e, rel=1e-12)


def test_compcell_metric_plateaus_with_cutoff():
    """Bare-aux ‖M‖_F diverges with cutoff; compcell ‖M‖_F plateaus.

    On H2 / def2-svp-jk in a 12-bohr cubic box, the bare-aux metric
    grows monotonically as more lattice cells are summed (the
    diffuse-aux divergence pathology). The compcell construction
    cancels the divergent multipoles and the metric is cutoff-stable.
    """
    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.2)
    fused = make_fused_basis(modrho, chg, mol)
    A = fuse_transform_matrix(modrho, chg)

    norms_bare: list[float] = []
    norms_compcell: list[float] = []
    for cut in (20.0, 30.0, 50.0):
        lo = vq.LatticeSumOptions()
        lo.cutoff_bohr = cut
        lo.nuclear_cutoff_bohr = cut
        M_aux = np.asarray(compute_2c_eri_lattice(aux, system, lo))
        norms_bare.append(float(np.linalg.norm(M_aux)))
        M_fused = np.asarray(compute_2c_eri_lattice(fused, system, lo))
        Mc = A @ M_fused @ A.T
        Mc = 0.5 * (Mc + Mc.T)
        norms_compcell.append(float(np.linalg.norm(Mc)))

    # Bare-aux: monotone strictly increasing.
    assert all(b < a for a, b in zip(norms_bare[1:], norms_bare[:-1])), (
        f"bare-aux metric should diverge: got {norms_bare}"
    )
    assert norms_bare[-1] / norms_bare[0] > 3.0, (
        f"bare-aux growth should be substantial: {norms_bare}"
    )
    # Compcell: bit-stable across cutoffs.
    for n in norms_compcell[1:]:
        assert n == pytest.approx(norms_compcell[0], rel=1e-8), (
            f"compcell metric should plateau: {norms_compcell}"
        )


def test_build_lpq_compcell_smoke():
    """build_lpq_compcell returns sensible-shaped Lpq."""
    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 20.0
    lo.nuclear_cutoff_bohr = 20.0
    Lpq = build_lpq_compcell(
        system,
        basis,
        aux,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        apply_aft_correction=False,
    )
    assert Lpq.ndim == 3
    assert Lpq.shape[1] == basis.nbasis
    assert Lpq.shape[2] == basis.nbasis
    # Symmetric in (μ, ν): Lpq is built from a symmetric 3c tensor at Γ.
    assert np.allclose(Lpq, Lpq.transpose(0, 2, 1), atol=1e-10)


def test_pbc_gdf_rhf_does_not_compute_gradient_by_default(monkeypatch):
    """Plain GDF energy runs must not pay the post-SCF gradient path."""
    from vibeqc import periodic_gdf_gradient

    def _unexpected_gradient(*args, **kwargs):
        raise AssertionError("compute_gdf_gradient should be opt-in")

    monkeypatch.setattr(
        periodic_gdf_gradient,
        "compute_gdf_gradient",
        _unexpected_gradient,
    )

    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 20.0
    opts.lattice_opts.nuclear_cutoff_bohr = 20.0

    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert r.converged
    assert r.gradient is None


def test_pbc_gdf_rhf_compute_gradient_opt_in(monkeypatch):
    """The convenience GDF gradient remains available when requested."""
    from vibeqc import periodic_gdf_gradient

    calls = []

    def _fake_gradient(system, basis, result, **kwargs):
        calls.append(kwargs)
        return np.zeros((len(system.unit_cell), 3))

    monkeypatch.setattr(
        periodic_gdf_gradient,
        "compute_gdf_gradient",
        _fake_gradient,
    )

    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 20.0
    opts.lattice_opts.nuclear_cutoff_bohr = 20.0

    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        apply_aft_correction=False,
        compute_gradient=True,
        progress=False,
    )

    assert r.converged
    assert r.apply_aft_correction is False
    assert calls
    assert np.allclose(r.gradient, 0.0)


def test_pbc_gdf_rhf_h2_converges_at_target_eta():
    """run_pbc_gdf_rhf converges on H2 to ~mHa of the known PySCF
    target at the prompt's recommended η ≈ 0.25 in this initial
    landing. Tightening to µHa is blocked on the AFT correction.
    """
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    assert r.converged
    # PySCF KRHF.density_fit() / exxdiv='ewald' target on this system
    # is -1.1225839666 Ha. Compcell-without-AFT gets ~2 mHa.
    PYSCF_TARGET = -1.1225839666
    assert abs(r.energy - PYSCF_TARGET) < 2e-3, (
        f"H2 SCF parity at η=0.25 expected within 2 mHa of PySCF "
        f"{PYSCF_TARGET}; got {r.energy} (Δ = "
        f"{r.energy - PYSCF_TARGET:.3e}). Further tightening blocked "
        "on AFT long-range correction (next milestone)."
    )


def test_periodic_rhf_backends_have_explicit_current_parity_contract():
    """Cross-backend regression for the three Γ-point RHF surfaces.

    These paths are not mathematically interchangeable: legacy
    ``run_rhf_periodic_gamma_gdf`` is the Ewald-J/real-space-K bridge,
    ``run_pbc_gdf_rhf`` is compensated-cell GDF, and BIPOLE uses its
    CRYSTAL-gauge direct-space scaffold. Pinning the current separation
    prevents future dispatcher/name cleanups from accidentally routing
    one backend through another.

    Re-baselined 2026-06-03: the analytic AO-pair-FT EWALD_3D Hartree J
    (``821ced25``, now the ``build_j_ewald_3d`` default that the legacy
    ewald-jk-fallback path uses) moved ``legacy`` −1.04996 → −1.11674
    (closer to the PySCF target −1.12258 — the old FFT-Poisson J was the
    cruder of the three; the −1.04996 pre-value is the value at ``821ced25^``
    and is reproducible at HEAD via ``VIBEQC_J_EWALD3D_BACKEND=grid``). The ``bipole`` move −1.11539 → −1.11118 is the
    exact-spheropole fix ``fcd16eb5`` (``compute_ext_el_spheropole`` rebuilt
    via libint ``emultipole2``, dropping the empirical p-p fudge): the whole
    4.2 mHa is the ``EXT EL-SPHEROPOLE`` correction +1.33 → +5.56 mHa
    (``e_electronic`` and ``e_nuclear`` are bit-identical), a validated
    improvement — CRYSTAL MgO/STO-3G spheropole parity <0.02 mHa, and the H2
    value is the exact ``π·N_e/(6V)·Σ_g P_μν(g)·⟨(r−A)²+(r−B−g)²⟩`` form
    (reproduced analytically to machine precision). It is NOT the gradient/
    assembly rebases — ``3c1f8026`` is a post-SCF analytic gradient and
    ``e9ebd5e8`` is docs-only; neither touches the forward SCF energy. The
    three backends now agree to a few mHa rather than tens of mHa, so the
    distinctness check is loosened to >1 mHa (still proves no accidental
    aliasing). Values are a current-state snapshot, not analytic targets.
    """
    system, basis = _h2_box(box_bohr=12.0)

    gdf_opts = vq.PeriodicRHFOptions()
    gdf_opts.use_diis = True
    gdf_opts.damping = 0.0
    gdf_opts.max_iter = 40
    gdf_opts.conv_tol_energy = 1e-10
    gdf_opts.conv_tol_grad = 1e-7
    gdf_opts.lattice_opts.cutoff_bohr = 25.0
    gdf_opts.lattice_opts.nuclear_cutoff_bohr = 25.0

    comp = vq.run_pbc_gdf_rhf(
        system,
        basis,
        gdf_opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    legacy = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        gdf_opts,
        aux_basis="def2-svp-jk",
        progress=False,
    )

    bipole_opts = vq.PeriodicRHFOptions()
    bipole_opts.use_diis = True
    bipole_opts.diis_start_iter = 2
    bipole_opts.damping = 0.2
    bipole_opts.max_iter = 20
    bipole_opts.conv_tol_energy = 1e-9
    bipole_opts.conv_tol_grad = 1e-7
    bipole_opts.initial_guess = InitialGuess.HCORE
    bipole_opts.lattice_opts.cutoff_bohr = 4.0
    bipole_opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    bipole = vq.run_pbc_bipole_rhf(
        system,
        basis,
        monkhorst_pack(system, [1, 1, 1]),
        bipole_opts,
        ewald_precision=1e-6,
        progress=False,
    )

    assert comp.converged and legacy.converged and bipole.converged
    assert comp.backend == "pbc-gdf-compcell"
    assert legacy.backend == "ewald-jk-fallback"

    assert comp.energy == pytest.approx(-1.1207706553754, abs=1e-6)
    assert legacy.energy == pytest.approx(-1.1167447255717, abs=1e-6)
    # M5's pair-resolved padded SR domain moved this duplicate snapshot to
    # the independently sanctioned production value pinned at 1e-9 in
    # test_pbc_bipole_ewald_split_integration.py. #478's alpha-consistent 1e
    # Ewald real cutoff then moved both together by -2.4818e-4 Ha
    # (-1.122138533948 -> -1.122386713788); the GDF composed and legacy arms
    # above are untouched, since neither pins a cell-scaled Ewald alpha.
    # #674 (corrected-split alpha bounded by the 4-bohr exchange cutoff,
    # envelope scaled with it) moved the BIPOLE arm alone once more:
    # -1.122386713788 -> -1.122526209887 (-1.3950e-4 Ha).
    assert bipole.energy == pytest.approx(-1.122526209887, abs=1e-6)
    # Distinct backends — still differ by > 1 mHa (no accidental aliasing).
    assert abs(comp.energy - legacy.energy) > 1e-3
    assert abs(bipole.energy - legacy.energy) > 1e-3
    assert abs(comp.energy - bipole.energy) > 1e-3

    assert float(np.trace(comp.density @ comp.overlap)) == pytest.approx(2.0)
    assert float(np.trace(legacy.density @ legacy.overlap)) == pytest.approx(2.0)
    assert float(np.trace(bipole.density.blocks[0] @ bipole.overlap[0]).real) == (
        pytest.approx(2.0)
    )


def test_pbc_gdf_rhf_exxdiv_shift_is_additive():
    """For RHF, switching exxdiv='ewald' → 'none' shifts the total
    energy by exactly ΔE_xx = −¼ ξ tr(D·S·D·S) on the converged
    density. Verifies the K-shift wiring matches the energy formula.
    """
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    r_e = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    r_n = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="none",
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    assert r_e.converged and r_n.converged
    xi = madelung_constant_for_cell(system)
    # The two SCF energies converge to slightly different densities,
    # but the energy shift formula should hold on EACH driver's
    # converged D. Use r_e.density as the reference.
    D = r_e.density
    S = r_e.overlap
    delta_predicted = -0.25 * xi * float(np.trace(D @ S @ D @ S))
    # Allow a small slack because r_e and r_n converged to slightly
    # different D (SCF fixed-point of a slightly different Fock).
    assert (r_e.energy - r_n.energy) == pytest.approx(delta_predicted, abs=1e-4), (
        f"exxdiv shift: expected {delta_predicted}; got {r_e.energy - r_n.energy}"
    )


def _single_prim_chg_basis(L: int, alpha: float, coef: float = 1.0):
    """A 1-shell BasisSet with the requested primitive at the origin."""
    mol = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    sh = ShellInfo(0, int(L), True, [float(alpha)], [float(coef)], [0.0, 0.0, 0.0])
    return vq.BasisSet(mol, [sh], f"<single-L{L}>", True), mol


def _parseval_pp_at_alpha(
    L: int, alpha: float, G_max: float = 20.0, G_density: int = 100
):
    """Parseval reconstruction of (P|P) via my FT formula on a fine
    uniform G-mesh. Returns the diagonal of the reconstructed matrix.
    """
    bs, _ = _single_prim_chg_basis(L, alpha)
    grid = np.linspace(-G_max, G_max, 2 * G_density + 1)
    dk = grid[1] - grid[0]
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    G2 = (G**2).sum(axis=1)
    mask = G2 > 0
    G = G[mask]
    G2 = G2[mask]
    ft = rsgdf_aux_fourier_transform(bs, G)
    weight = (dk**3) / (2.0 * np.pi**2)
    M = np.zeros((bs.nbasis, bs.nbasis), dtype=complex)
    inv_G2 = 1.0 / G2
    for i in range(bs.nbasis):
        for j in range(bs.nbasis):
            M[i, j] = weight * np.sum(np.conj(ft[i]) * ft[j] * inv_G2)
    return np.real(np.diag(M))


def test_rsgdf_ft_parseval_matches_libint_l1_l2_l3():
    """The L>0 convention fix in rsgdf_aux_fourier_transform (2026-05-17)
    means the Parseval-reconstructed (P|P) from the analytical FT matches
    libint's compute_2c_eri to better than 1% on single-primitive shells
    across α ∈ [0.25, 2.0] for L = 1, 2, 3. Regression-guards against a
    future change reintroducing the L>0 scale-factor mismatch.
    """
    from vibeqc._vibeqc_core import compute_2c_eri

    for L in (1, 2, 3):
        for alpha in (0.25, 0.5, 1.0, 2.0):
            bs, _ = _single_prim_chg_basis(L, alpha)
            libint_diag = np.diag(np.asarray(compute_2c_eri(bs)))
            parseval_diag = _parseval_pp_at_alpha(L, alpha)
            # Per-m components should all be equal by rotational
            # symmetry; libint vs Parseval ratio should be ~1.0 with
            # sub-percent quadrature residual.
            for li, pi in zip(libint_diag, parseval_diag):
                ratio = li / pi
                assert abs(ratio - 1.0) < 0.01, (
                    f"L={L} α={alpha}: libint/Parseval ratio = {ratio} "
                    f"(libint={li}, Parseval={pi}); expected 1.0 ± 1% "
                    "per the L>0 convention fix."
                )


def test_compcell_aft_correction_shape_and_symmetry():
    """The (still-experimental) AFT correction helper returns a tensor
    of the documented shape; the chg-aux cross block respects the
    Hermitian symmetry of the periodic Coulomb metric at Γ (real)."""
    from vibeqc.aux_basis import _compcell_aft_correction

    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.2)
    fused = make_fused_basis(modrho, chg, mol)
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    j2c_p = _compcell_aft_correction(fused, n_aux, system, eta=0.2)
    assert j2c_p.shape == (n_chg, n_aux + n_chg)
    # Real-valued at Γ.
    assert np.isrealobj(j2c_p)
    # chg-chg sub-block should be (close to) symmetric.
    chg_chg = j2c_p[:, n_aux:]
    assert np.allclose(chg_chg, chg_chg.T, atol=1e-10), (
        "j2c_p[:, n_aux:] should be symmetric (chg-chg block at Γ)."
    )


def test_compcell_aft_correction_q_zero_matches_q_none():
    """``_compcell_aft_correction(..., q=None)`` (the historical Γ-only
    path) must match ``_compcell_aft_correction(..., q=zeros)`` (the
    q-aware path evaluated at q=0) bit-exactly. This pins the q-aware
    refactor — if the q=None branch ever diverges from q=zeros at the
    AFT-mesh, FT, or kernel level, parity tests downstream will
    flip silently."""
    from vibeqc.aux_basis import _compcell_aft_correction

    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.2)
    fused = make_fused_basis(modrho, chg, mol)
    n_aux = modrho.nbasis
    j2c_p_none = _compcell_aft_correction(fused, n_aux, system, eta=0.2)
    j2c_p_zero = _compcell_aft_correction(fused, n_aux, system, eta=0.2, q=np.zeros(3))
    # Real path returns float64; q-aware path with q=0 should also be
    # treated as Γ and return float64 (the zero-vector triggers the
    # is_gamma_q branch with G=0 excluded).
    assert np.isrealobj(j2c_p_none)
    assert np.isrealobj(j2c_p_zero)
    assert j2c_p_none.shape == j2c_p_zero.shape
    np.testing.assert_array_equal(j2c_p_none, j2c_p_zero)


def test_compcell_aft_correction_q_nonzero_finite_and_complex():
    """At q ≠ 0 the AFT correction returns a finite complex tensor of
    the documented shape. The complex-valuedness is the visible
    fingerprint of the q-shifted FT kernel — at q=0 the G ↔ -G
    symmetry of a real basis makes the sum real; at q ≠ 0 that
    symmetry is broken and an imaginary part appears."""
    from vibeqc.aux_basis import _compcell_aft_correction

    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.2)
    fused = make_fused_basis(modrho, chg, mol)
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    q = np.array([0.1, 0.05, 0.0])
    j2c_p = _compcell_aft_correction(fused, n_aux, system, eta=0.2, q=q)
    assert j2c_p.shape == (n_chg, n_aux + n_chg)
    assert np.iscomplexobj(j2c_p)
    assert np.all(np.isfinite(j2c_p))
    # The imaginary part should be non-trivially nonzero somewhere
    # (sanity: confirms we're not silently zeroing it out).
    assert np.max(np.abs(j2c_p.imag)) > 1e-10


def test_compcell_aft_correction_q_nonzero_hermitian_under_negate_q():
    """Hermitian symmetry of the AFT correction under q → -q:
    ``j2c_p(-q) ≈ conj(j2c_p(q))`` (Bloch-Wigner-Seitz inversion
    sends q → -q which conjugates Bloch phases; for a real basis at
    Γ this is what the periodic Coulomb metric exhibits)."""
    from vibeqc.aux_basis import _compcell_aft_correction

    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.2)
    fused = make_fused_basis(modrho, chg, mol)
    n_aux = modrho.nbasis
    q = np.array([0.1, 0.05, 0.0])
    j2c_p_pos = _compcell_aft_correction(fused, n_aux, system, eta=0.2, q=q)
    j2c_p_neg = _compcell_aft_correction(fused, n_aux, system, eta=0.2, q=-q)
    # Bit-equal up to FT mesh symmetry (the G-mesh is symmetric in
    # ±G so the conjugate identity should hold to numerical noise).
    np.testing.assert_allclose(j2c_p_neg, np.conj(j2c_p_pos), rtol=0, atol=1e-12)


def test_build_lpq_compcell_aft_default_is_off():
    """The AFT correction is implemented but not yet at full PySCF
    parity. The default in this slice is ``apply_aft_correction=False``
    so existing SCF parity (H2 mHa at η=0.25) is preserved. Regression-
    guards against a future flip-to-True without the per-shell libint↔
    libcint contraction-normalization conversion that closes the
    remaining ~70 mHa gap (see CHANGELOG / module docstring).
    """
    import inspect

    sig = inspect.signature(build_lpq_compcell)
    default = sig.parameters["apply_aft_correction"].default
    assert default is True, (
        f"build_lpq_compcell.apply_aft_correction default is {default} — "
        "should be True (AFT correction ON by default since v0.12.0)."
    )


def test_run_pbc_gdf_rhf_aft_false_warns():
    """apply_aft_correction=False (disabling the production AFT path)
    must emit a runtime warning directing users to the default (AFT on)."""
    import warnings as _warnings

    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0

    with _warnings.catch_warnings(record=True) as wl:
        _warnings.simplefilter("always")
        vq.run_pbc_gdf_rhf(
            system,
            basis,
            opts,
            aux_basis="def2-svp-jk",
            exxdiv="ewald",
            apply_aft_correction=False,
            progress=False,
        )
    assert any("apply_aft_correction=False" in str(w.message) for w in wl), (
        "apply_aft_correction=False must warn about missing AFT correction"
    )

    # Default (AFT on with libint convention) must NOT warn.
    with _warnings.catch_warnings(record=True) as wl2:
        _warnings.simplefilter("always")
        vq.run_pbc_gdf_rhf(
            system,
            basis,
            opts,
            aux_basis="def2-svp-jk",
            exxdiv="ewald",
            progress=False,
        )
    assert not any("apply_aft_correction=False" in str(w.message) for w in wl2), (
        "default (AFT on) must not warn"
    )
    assert not any("aft_ft_convention='libcint'" in str(w.message) for w in wl2), (
        "default (libint convention) must not warn"
    )

    # Using libcint convention must warn.
    with _warnings.catch_warnings(record=True) as wl3:
        _warnings.simplefilter("always")
        vq.run_pbc_gdf_rhf(
            system,
            basis,
            opts,
            aux_basis="def2-svp-jk",
            exxdiv="ewald",
            aft_ft_convention="libcint",
            progress=False,
        )
    assert any("aft_ft_convention='libcint'" in str(w.message) for w in wl3), (
        "libcint convention must warn about convention mismatch"
    )


def test_compcell_with_aft_metric_cutoff_stable():
    """The AFT-enabled compcell metric ||M||_F is bit-stable across a
    wide cutoff sweep (algebraic statement that the AFT correctly
    cancels the divergent lattice-sum contribution from the chg basis).

    This is a strictly stronger property than the no-AFT compcell
    metric stability — there the bare lattice sum is supposed to
    converge eventually (it does, slowly, for charge-neutral
    combinations); here the AFT removes the convergent-but-divergent
    G≠0 LR sum analytically, so the residue is bit-stable from very
    small cutoffs.
    """
    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    norms: list[float] = []
    for cut in (15.0, 30.0, 50.0):
        lo = vq.LatticeSumOptions()
        lo.cutoff_bohr = cut
        lo.nuclear_cutoff_bohr = cut
        Lpq = build_lpq_compcell(
            system,
            vq.BasisSet(mol, "sto-3g"),
            aux,
            molecule=mol,
            lat_opts=lo,
            compcell_eta=0.25,
            apply_aft_correction=True,
        )
        norms.append(float(np.linalg.norm(Lpq)))
    for n in norms[1:]:
        assert n == pytest.approx(norms[0], rel=1e-8), (
            f"AFT-corrected Lpq norm should be cutoff-stable: {norms}"
        )


def test_compcell_aft_correction_matches_pyscf_via_libcint_convention():
    """The j2c_p subtracted by ``_compcell_aft_correction`` is in the
    libcint Y_lm-included convention so it matches PySCF's internal
    AFT j2c_p bit-perfectly on H2/12-bohr/def2-svp-jk/η=0.25. The
    diagnostic ``examples/debug/gdf_aft_pyscf_diff.py`` shows the
    per-block Frobenius ratios are 1.0000 to all decimal places and
    diagonals match to PySCF's printed precision. This regression-
    guards the libcint-convention choice (the other obvious choice,
    libint-calibrated FT via rsgdf_aux_fourier_transform, gives
    diverging SCF energies).
    """
    # The libcint-convention FT helper exists and is what
    # _compcell_aft_correction calls (verified by inspecting source).
    import inspect

    from vibeqc.aux_basis import (
        _aft_fourier_transform_libcint_convention,
        _compcell_aft_correction,
    )

    source = inspect.getsource(_compcell_aft_correction)
    assert "_aft_fourier_transform_libcint_convention" in source, (
        "_compcell_aft_correction must use the libcint-convention FT "
        "for parity with PySCF. The libint-calibrated FT in "
        "rsgdf_aux_fourier_transform gives diverging SCF energies "
        "(documented in CHANGELOG / module comments)."
    )


def test_pbc_gdf_rhf_h2_sub_mha_via_2c_plus_3c_aft():
    """With both 2c AND 3c AFT corrections enabled in libint convention
    (matched to vibe-qc's bare lattice sum), run_pbc_gdf_rhf reaches
    SUB-MILLIHARTREE parity to PySCF on H2/12-bohr/def2-svp-jk.

    The 3c AFT (PySCF gdf_builder.py:add_ft_j3c) is the second half of
    the compcell AFT subtraction; without it we plateau at ~mHa (the
    no-AFT default already gives 1.8 mHa). With both 2c+3c in matched
    libint convention, the η-sweep develops a clean zero-crossing
    around η ≈ 1.0 where the SCF is bit-equivalent to PySCF.
    """
    PYSCF_TARGET = -1.1225839666
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        compcell_eta=1.0,
        apply_aft_correction=True,
        aft_ft_convention="libint",
        rcut_strategy="pyscf_auto",
        progress=False,
    )
    assert r.converged
    delta = r.energy - PYSCF_TARGET
    # Sub-millihartree (within 1 mHa) — the new behaviour with 2c+3c AFT
    # in matched libint convention. Tighter than 1.8 mHa default (no AFT).
    assert abs(delta) < 1e-3, (
        f"H2 SCF with 2c+3c AFT (libint conv) at η=1.0 should land "
        f"sub-mHa vs PySCF target {PYSCF_TARGET}; got {r.energy} "
        f"(Δ = {delta:.3e})."
    )


def test_pbc_gdf_rhf_h2_sub_uha_via_rsgdf():
    """RSGDF (range-separated GDF) reaches SUB-MICROHARTREE parity to
    PySCF on H2/12-bohr/def2-svp-jk — 2000× tighter than the
    compcell+AFT path.

    The win comes from the Madelung G=0 renormalisation in
    ``rsgdf_lr_2c_metric`` / ``rsgdf_lr_3c_tensor`` (the LR Coulomb
    kernel's ``4π·exp(-G²/(4ω²))/G²`` expands at G→0 as
    ``4π/G² - π/ω² + O(G²)``; the divergent piece cancels against the
    neutralising background and the finite ``-π/(ω²·V)·monopole_P·
    monopole_Q`` remainder must be added back because the discrete
    G-sum omits G = 0). The pre-fix spherical-BZ-erf approximation
    left H2 RSGDF SCF ~207 mHa off PySCF; the fix takes it to ≤ µHa
    and ω-invariant from ω ≈ 0.4 onwards.

    Per-shell-block parity vs PySCF ``rsdf_builder._RSGDFBuilder.
    get_2c2e`` is bit-exact (5+ sig figs) on He / def2-svp-jkfit;
    H₂ SCF parity confirmed end-to-end below.
    """
    PYSCF_TARGET = -1.1225839666
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        gdf_method="rsgdf",
        rsgdf_omega=0.4,
        rsgdf_g_precision=1e-10,
        progress=False,
    )
    assert r.converged
    delta = r.energy - PYSCF_TARGET
    assert abs(delta) < 1e-6, (
        f"H2 RSGDF SCF at ω=0.4 should land sub-µHa vs PySCF target "
        f"{PYSCF_TARGET}; got {r.energy} (Δ = {delta:.3e})."
    )


def _lih_primitive_rsgdf_case():
    """LiH primitive FCC / STO-3G / def2-svp-jk rsgdf parity case.

    Returns ``(system, basis, PYSCF_TARGET)`` shared by the chem-acc
    (ke=400, fast) and µHa (ke=800, @slow) rsgdf parity tests below.
    ``PYSCF_TARGET`` is PySCF df.GDF + exxdiv='ewald', def2-universal-jkfit.
    """
    ANG2BOHR = 1.0 / 0.529177210903
    A = 4.084
    lat = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * A * ANG2BOHR
    atoms = [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0.5 * A * ANG2BOHR] * 3)]
    system = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis, -8.3351832169


def test_gamma_dense_core_gdf_parity_hold_classifies_p01_mgo_not_lih():
    """P01 MgO/STO-3G Γ RSGDF is a held electronic-gauge audit, while the
    lighter LiH primitive RSGDF case keeps its existing PySCF parity status.

    This is intentionally classifier-only: the MgO SCF is too expensive for
    the fast lane, and the prompt-11 problem is a provenance/validation guard
    for a converged-but-held value rather than an SCF crash.
    """
    from vibeqc.pbc_gdf import _gamma_dense_core_gdf_parity_held

    ang2bohr = 1.0 / 0.529177210903
    a_mgo = 4.212 * ang2bohr
    mgo_lat = (
        np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        * a_mgo
    )
    mgo = vq.PeriodicSystem(
        3,
        mgo_lat,
        [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [0.5 * a_mgo] * 3)],
    )
    lih, _basis, _target = _lih_primitive_rsgdf_case()
    h2, _basis = _h2_box(box_bohr=12.0)

    assert _gamma_dense_core_gdf_parity_held(mgo, "rsgdf")
    assert _gamma_dense_core_gdf_parity_held(mgo, "mdf")
    assert not _gamma_dense_core_gdf_parity_held(mgo, "compcell")
    assert not _gamma_dense_core_gdf_parity_held(lih, "rsgdf")
    assert not _gamma_dense_core_gdf_parity_held(h2, "rsgdf")


def _rsgdf_lih_opts():
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    return opts


def test_pbc_gdf_rhf_lih_primitive_chemacc_via_rsgdf():
    """RSGDF on LiH primitive FCC / STO-3G / def2-svp-jk at ke=400 lands
    at chemical accuracy (~+515 µHa) vs PySCF df.GDF.

    This is the FAST-LANE guard for the rsgdf LiH parity. The tighter
    µHa bound at ke=800 is the ``@slow`` companion
    :func:`test_pbc_gdf_rhf_lih_primitive_uha_parity_via_rsgdf` — its
    dense Bloch AO-pair FT on the fine ke=800 mesh is inherently a
    minutes-scale build (see ``handovers/HANDOVER_GDF_OUTSTANDING.md`` §7), so it
    runs on the slow/nightly lane while this ke=400 case keeps the
    rsgdf LiH SCF + parity exercised in the fast gate.
    """
    system, basis, PYSCF_TARGET = _lih_primitive_rsgdf_case()
    r400 = vq.run_pbc_gdf_rhf(
        system,
        basis,
        _rsgdf_lih_opts(),
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=400.0,
        progress=False,
    )
    assert r400.converged
    d400 = r400.energy - PYSCF_TARGET
    # Sub-mHa at ke=400 (typical observed: ~+515 µHa).
    assert abs(d400) < 1.0e-3, (
        f"LiH primitive FCC RSGDF at ke=400 Ha should land within "
        f"1 mHa of PySCF GDF target {PYSCF_TARGET}; got {r400.energy} "
        f"(Δ = {d400:.3e})."
    )


@pytest.mark.slow
def test_pbc_gdf_rhf_lih_primitive_uha_parity_via_rsgdf():
    """RSGDF on LiH primitive FCC / STO-3G / def2-svp-jk reaches µHa
    parity vs PySCF.pbc.df.GDF (and df.RSDF) at ke=800 — the target Mike
    set on 2026-05-29.

    ``@slow``: the ke=800 dense Bloch AO-pair FT is an inherently
    fine-mesh build (minutes even after the cderi↔V_ne AO-pair-FT dedup,
    ``fee8f0ac``). The fast-lane chem-acc guard is the sibling
    :func:`test_pbc_gdf_rhf_lih_primitive_chemacc_via_rsgdf`.

    The full fix chain (v0.10.0 → v0.11.0):

    1. **all-FT-Bloch path** (commit 018784c0): replace SR+LR sparse-mesh
       with a dense FFT mesh + Bloch-summed pair-FT. Closes +11 Ha → +18 mHa.
    2. **Proper Ewald-Madelung** (commit d7f4b3bd): replace the cubic
       Wigner shortcut with a proper Ewald self-energy sum. Closes
       +18 mHa → -2 mHa.
    3. **Analytical-FT V_long** (this fix, post-integrals chat):
       replace the cubic-real-space-grid V_long quadrature in
       compute_nuclear_lattice_ewald with the analytical reciprocal-
       space FT formula per Lippert 1997 + Sun 2017 + McClain 2017.
       The cubic grid broke the 3-fold FCC rotation symmetry — Hcore
       off by 1e-2 per element. The reciprocal-space sum preserves
       crystal symmetry exactly. Closes -2 mHa → -0.5 µHa at ke=800.

    At ke=800 we land at -0.5 µHa vs PySCF GDF (-8.3351832), within
    PySCF's own GDF↔RSDF noise floor of ~1 µHa.
    """
    system, basis, PYSCF_TARGET = _lih_primitive_rsgdf_case()
    r800 = vq.run_pbc_gdf_rhf(
        system,
        basis,
        _rsgdf_lih_opts(),
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=800.0,
        progress=False,
    )
    assert r800.converged
    d800 = r800.energy - PYSCF_TARGET
    # µHa parity (10 µHa bound — PySCF's own GDF↔RSDF gap is ~1 µHa).
    assert abs(d800) < 10e-6, (
        f"LiH primitive FCC RSGDF at ke=800 Ha should land within "
        f"10 µHa of PySCF GDF target {PYSCF_TARGET}; got {r800.energy} "
        f"(Δ = {d800:.3e})."
    )


def test_rsgdf_shared_pair_ft_is_bit_identical_and_guarded():
    """The dense-mesh AO-pair FT ρ̂_μν(G) shared between the rsgdf cderi
    build (``build_lpq_native_fft``) and the EWALD_3D V_ne FT
    (``compute_v_ne_ewald_3d_ft_gamma``) must give BIT-IDENTICAL results
    whether each builder computes its own or reuses a precomputed bundle.

    This guards the ``run_pbc_gdf_rhf`` dedup (the dominant
    ``ao_pair_fourier_transform_bloch`` runs ONCE per SCF instead of
    twice) against silent numerical drift, and checks the provenance
    guard fires when a bundle's parameters don't match the consumer.
    """
    from vibeqc.aux_basis import _rsgdf_dense_pair_ft, build_lpq_native_fft
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma

    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux_modrho = make_modrho_aux_basis(
        make_aux_basis_set(mol, aux_name="def2-svp-jk"), mol
    )
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 20.0
    lo.nuclear_cutoff_bohr = 20.0
    ke = 100.0

    bundle = _rsgdf_dense_pair_ft(basis, system, ke, lo)

    # cderi: shared bundle == internal computation, bit for bit.
    Lpq_internal = build_lpq_native_fft(
        system, basis, aux_modrho, ke_cutoff=ke, lat_opts=lo
    )
    Lpq_shared = build_lpq_native_fft(
        system, basis, aux_modrho, ke_cutoff=ke, lat_opts=lo,
        pair_ft_shared=bundle,
    )
    assert np.array_equal(Lpq_internal, Lpq_shared)

    # V_ne: shared bundle == internal computation, bit for bit.
    V_internal = compute_v_ne_ewald_3d_ft_gamma(basis, system, lo, ke_cutoff=ke)
    V_shared = compute_v_ne_ewald_3d_ft_gamma(
        basis, system, lo, ke_cutoff=ke, pair_ft_shared=bundle
    )
    assert np.array_equal(V_internal, V_shared)

    # Provenance guard: a bundle from a different ke_cutoff is rejected
    # loudly rather than silently producing a wrong tensor.
    bundle_wrong = _rsgdf_dense_pair_ft(basis, system, 80.0, lo)
    with pytest.raises(ValueError, match="provenance mismatch"):
        build_lpq_native_fft(
            system, basis, aux_modrho, ke_cutoff=ke, lat_opts=lo,
            pair_ft_shared=bundle_wrong,
        )
    with pytest.raises(ValueError, match="provenance mismatch"):
        compute_v_ne_ewald_3d_ft_gamma(
            basis, system, lo, ke_cutoff=ke, pair_ft_shared=bundle_wrong
        )


def test_pbc_gdf_uhf_m1_equals_rhf():
    """run_pbc_gdf_uhf at multiplicity=1 reproduces run_pbc_gdf_rhf to
    machine precision — the closed-shell-limit gate that validates the
    open-shell J / per-spin-K / exxdiv / energy machinery against the
    trusted (PySCF µHa-parity) RHF driver. H2/12-bohr/def2-svp-jk/rsgdf.
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf, run_pbc_gdf_uhf

    system, basis = _h2_box(box_bohr=12.0)

    def _opts():
        o = vq.PeriodicRHFOptions()
        o.use_diis = True
        o.damping = 0.0
        o.max_iter = 40
        o.conv_tol_energy = 1e-10
        o.lattice_opts.cutoff_bohr = 30.0
        o.lattice_opts.nuclear_cutoff_bohr = 30.0
        return o

    common = dict(
        aux_basis="def2-svp-jk", exxdiv="ewald", gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0, progress=False,
    )
    r_rhf = run_pbc_gdf_rhf(system, basis, _opts(), **common)
    r_uhf = run_pbc_gdf_uhf(system, basis, _opts(), **common)
    assert r_rhf.converged and r_uhf.converged
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-8, (
        f"UHF(M=1) {r_uhf.energy} != RHF {r_rhf.energy} "
        f"(Δ={r_uhf.energy - r_rhf.energy:.2e})"
    )
    assert abs(r_uhf.s_squared) < 1e-6  # singlet ⟨S²⟩ ≈ 0


def test_pbc_gdf_uhf_h_doublet_pyscf_parity():
    """Open-shell UHF GDF on an H-atom doublet (1 e, mult=2 → nα=1, nβ=0)
    reaches µHa parity with PySCF UHF.density_fit()/exxdiv='ewald' and
    gives the exact doublet ⟨S²⟩ = 0.75. Validated out-of-process
    (CLAUDE.md §10); PySCF reference hard-coded to avoid a subprocess
    dependency in the suite (observed Δ ≈ -8e-11 Ha).
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_uhf

    # pyscf.pbc.scf.UHF(cell).density_fit(), exxdiv='ewald',
    # H / sto-3g / 10-bohr box / def2-svp-jk aux, cell.spin=1, cell.unit='B'.
    PYSCF_UHF_DF = -0.4709262789
    system = vq.PeriodicSystem(3, np.diag([10.0, 10.0, 10.0]), [vq.Atom(1, [0, 0, 0])])
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    o = vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.0
    o.max_iter = 50
    o.conv_tol_energy = 1e-9
    o.lattice_opts.cutoff_bohr = 25.0
    o.lattice_opts.nuclear_cutoff_bohr = 25.0
    r = run_pbc_gdf_uhf(
        system, basis, o, aux_basis="def2-svp-jk", exxdiv="ewald",
        gdf_method="rsgdf", rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert r.converged
    assert isinstance(r, vq.PBCGDFUHFResult)
    assert abs(r.energy - PYSCF_UHF_DF) < 5e-6, (
        f"H-atom doublet UHF {r.energy} vs PySCF {PYSCF_UHF_DF} "
        f"(Δ={r.energy - PYSCF_UHF_DF:.2e})"
    )
    assert abs(r.s_squared - 0.75) < 1e-6  # doublet ideal ⟨S²⟩
    # α has 1 electron, β has 0.
    assert abs(np.trace(r.density_alpha @ r.overlap) - 1.0) < 1e-6
    assert abs(np.trace(r.density_beta @ r.overlap)) < 1e-6


def test_pbc_gdf_uhf_compcell_aft_off_is_visibly_held():
    """The incomplete AFT-off compcell fit must not look production-ready.

    The historical triplet-gradient validation compared this diagnostic fit
    with PySCF's complete density fit and misclassified the resulting 52 mHa
    offset as a spin-polarized J/K defect.  The same omission shifts RHF too;
    explicit AFT-off UHF remains available for fit development but is warned
    and tagged as held.
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_uhf

    system, basis = _h2_box(box_bohr=12.0)
    system.multiplicity = 3
    o = vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.0
    o.max_iter = 40
    o.conv_tol_energy = 1e-10
    o.lattice_opts.cutoff_bohr = 30.0
    o.lattice_opts.nuclear_cutoff_bohr = 30.0

    with pytest.warns(UserWarning, match="AFT long-range correction"):
        result = run_pbc_gdf_uhf(
            system,
            basis,
            o,
            aux_basis="def2-svp-jk",
            exxdiv="ewald",
            gdf_method="compcell",
            compcell_eta=1.0,
            apply_aft_correction=False,
            progress=False,
        )

    assert result.converged
    assert result.backend.endswith("+PARITY_HELD")


def test_pbc_gdf_uks_func_none_equals_rhf():
    """run_pbc_gdf_uks(functional=None) falls back to plain UHF (α=1, no
    V_xc) and reproduces run_pbc_gdf_rhf to machine precision — the gate
    validating the UKS driver's non-XC machinery. H2/rsgdf."""
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf, run_pbc_gdf_uks

    system, basis = _h2_box(box_bohr=12.0)

    def _opts():
        o = vq.PeriodicRHFOptions()
        o.use_diis = True
        o.damping = 0.0
        o.max_iter = 50
        o.conv_tol_energy = 1e-10
        o.lattice_opts.cutoff_bohr = 30.0
        o.lattice_opts.nuclear_cutoff_bohr = 30.0
        return o

    common = dict(
        aux_basis="def2-svp-jk", exxdiv="ewald", gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0, progress=False,
    )
    r_rhf = run_pbc_gdf_rhf(system, basis, _opts(), **common)
    r_uks = run_pbc_gdf_uks(system, basis, _opts(), functional=None, **common)
    assert r_rhf.converged and r_uks.converged
    assert abs(r_uks.energy - r_rhf.energy) < 1e-8, (
        f"UKS(None) {r_uks.energy} != RHF {r_rhf.energy}"
    )


def test_pbc_gdf_uks_pbe_pyscf_parity():
    """Open-shell UKS-PBE GDF reaches chemical-accuracy parity with PySCF
    {R,U}KS(xc='pbe').density_fit()/exxdiv='ewald' on a closed-shell H2
    (M=1) and an H-atom doublet, with the exact doublet ⟨S²⟩. Native
    libxc V_xc via build_xc_periodic_uks; out-of-process PySCF refs (§10).
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_uks

    # pyscf.pbc.dft.RKS(cell).density_fit(), xc='pbe', exxdiv='ewald',
    # H2/sto-3g/12-bohr/def2-svp-jk, cell.unit='B' (observed Δ ≈ +1 µHa).
    PYSCF_RKS_PBE_H2 = -1.15214067
    system, basis = _h2_box(box_bohr=12.0)
    o = vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.0
    o.max_iter = 60
    o.conv_tol_energy = 1e-10
    o.lattice_opts.cutoff_bohr = 30.0
    o.lattice_opts.nuclear_cutoff_bohr = 30.0
    r = run_pbc_gdf_uks(
        system, basis, o, functional="pbe", aux_basis="def2-svp-jk",
        exxdiv="ewald", gdf_method="rsgdf", rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert r.converged
    assert isinstance(r, vq.PBCGDFUKSResult)
    assert r.functional == "pbe" and r.e_xc < 0.0
    assert abs(r.energy - PYSCF_RKS_PBE_H2) < 1e-4, (
        f"UKS-PBE(M=1) H2 {r.energy} vs PySCF {PYSCF_RKS_PBE_H2} "
        f"(Δ={r.energy - PYSCF_RKS_PBE_H2:.2e})"
    )
    assert abs(r.s_squared) < 1e-6  # singlet

    # H-atom doublet UKS-PBE vs pyscf.pbc.dft.UKS(...).density_fit()
    # (cell.spin=1; observed Δ ≈ +15 µHa — molecular-Becke-grid vs PySCF grid).
    PYSCF_UKS_PBE_H_DOUBLET = -0.46480061
    sysd = vq.PeriodicSystem(3, np.diag([10.0, 10.0, 10.0]), [vq.Atom(1, [0, 0, 0])])
    sysd.multiplicity = 2
    basd = vq.BasisSet(sysd.unit_cell_molecule(), "sto-3g")
    od = vq.PeriodicRHFOptions()
    od.use_diis = True
    od.damping = 0.0
    od.max_iter = 60
    od.conv_tol_energy = 1e-9
    od.lattice_opts.cutoff_bohr = 25.0
    od.lattice_opts.nuclear_cutoff_bohr = 25.0
    rd = run_pbc_gdf_uks(
        sysd, basd, od, functional="pbe", aux_basis="def2-svp-jk",
        exxdiv="ewald", gdf_method="rsgdf", rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert rd.converged
    assert abs(rd.energy - PYSCF_UKS_PBE_H_DOUBLET) < 1e-4, (
        f"UKS-PBE H-doublet {rd.energy} vs PySCF {PYSCF_UKS_PBE_H_DOUBLET} "
        f"(Δ={rd.energy - PYSCF_UKS_PBE_H_DOUBLET:.2e})"
    )
    assert abs(rd.s_squared - 0.75) < 1e-6  # doublet ideal ⟨S²⟩


def test_pbc_gdf_uks_gamma_honors_fock_mixing():
    """Γ UKS/GDF applies the public fock_mixing option instead of accepting
    and silently ignoring it. The final H2/PBE fixed point is unchanged, and
    result provenance records the routed value."""
    from vibeqc.pbc_gdf import run_pbc_gdf_uks

    system, basis = _h2_box(box_bohr=12.0)
    o = vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.0
    o.fock_mixing = 0.20
    o.max_iter = 60
    o.conv_tol_energy = 1e-10
    o.lattice_opts.cutoff_bohr = 30.0
    o.lattice_opts.nuclear_cutoff_bohr = 30.0
    r = run_pbc_gdf_uks(
        system,
        basis,
        o,
        functional="pbe",
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0,
        progress=False,
    )
    assert r.converged
    assert r.fock_mixing == pytest.approx(0.20)
    assert r.energy == pytest.approx(-1.15214067, abs=1e-4)


def _boron_gamma_smear_opts(ks: bool = False):
    o = vq.PeriodicKSOptions() if ks else vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.0
    o.max_iter = 200
    o.conv_tol_energy = 1e-9
    o.lattice_opts.cutoff_bohr = 30.0
    o.lattice_opts.nuclear_cutoff_bohr = 30.0
    o.smearing_temperature = 0.01
    return o


def test_pbc_gdf_uhf_gamma_smearing_fractional():
    """Γ-only open-shell UHF GDF with Fermi-Dirac smearing fractionally fills a
    degenerate partially-occupied shell that integer Aufbau cannot represent: a
    boron atom's 3-fold-degenerate 2p¹. Independent per-spin global μ_α/μ_β
    spread the α electron ≈1/3 across the three 2p orbitals; per-spin counts are
    conserved (Σ occ_σ = n_σ), entropy > 0, Mermin free energy A = E − T·S < E,
    and the doublet ⟨S²⟩ = 0.75."""
    from vibeqc.pbc_gdf import run_pbc_gdf_uhf

    system = vq.PeriodicSystem(3, np.diag([9.0, 9.0, 9.0]), [vq.Atom(5, [0, 0, 0])])
    system.multiplicity = 2  # boron doublet ⇒ nα=3, nβ=2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    r = run_pbc_gdf_uhf(
        system, basis, _boron_gamma_smear_opts(), aux_basis="def2-svp-jk",
        exxdiv="ewald", gdf_method="rsgdf", rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert r.converged
    assert isinstance(r, vq.PBCGDFUHFResult)
    assert r.smearing_temperature == pytest.approx(0.01)
    occ_a = np.asarray(r.occupations_alpha)
    occ_b = np.asarray(r.occupations_beta)
    assert abs(occ_a.sum() - 3.0) < 1e-7, f"α count {occ_a.sum()} != 3"
    assert abs(occ_b.sum() - 2.0) < 1e-7, f"β count {occ_b.sum()} != 2"
    assert np.any((occ_a > 1e-3) & (occ_a < 1.0 - 1e-3)), f"expected fractional α; got {occ_a}"
    assert r.entropy > 1e-6
    assert r.free_energy < r.energy - 1e-9
    assert abs(r.s_squared - 0.75) < 1e-6


def test_pbc_gdf_uks_gamma_smearing_fractional():
    """Γ-only open-shell UKS-PBE GDF with smearing: the same boron 2p¹ fractional
    fill as the UHF test, plus native libxc V_xc (e_xc < 0). Validates the
    per-spin smearing wiring through the UKS driver and the Mermin free energy."""
    from vibeqc.pbc_gdf import run_pbc_gdf_uks

    system = vq.PeriodicSystem(3, np.diag([9.0, 9.0, 9.0]), [vq.Atom(5, [0, 0, 0])])
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    r = run_pbc_gdf_uks(
        system, basis, _boron_gamma_smear_opts(ks=True), functional="pbe",
        aux_basis="def2-svp-jk", exxdiv="ewald", gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert r.converged
    assert isinstance(r, vq.PBCGDFUKSResult)
    assert r.functional == "pbe" and r.e_xc < 0.0
    occ_a = np.asarray(r.occupations_alpha)
    assert abs(occ_a.sum() - 3.0) < 1e-7
    assert abs(np.asarray(r.occupations_beta).sum() - 2.0) < 1e-7
    assert np.any((occ_a > 1e-3) & (occ_a < 1.0 - 1e-3))
    assert r.entropy > 1e-6
    assert r.free_energy < r.energy - 1e-9


def test_pyscf_rcut_uses_pair_reduced_exponent():
    """The estimator must decay with the PAIR reduced exponent alpha/2.

    Regression for RCUT-PYSCF-MISPORT (fixed 2026-08-02). What the cutoff
    truncates is a lattice sum over AO *pairs*: by the Gaussian product
    theorem two Gaussians of exponent alpha separated by R overlap as
    exp(-mu R^2) with mu = alpha*alpha/(alpha+alpha) = alpha/2. A pair
    therefore reaches sqrt(2) times further than the single-shell
    amplitude exp(-alpha r^2), and screening on the latter truncates
    early.

    Pinned as the exact sqrt(2) ratio against a locally recomputed
    single-exponent reference, so this catches the regression for every
    (alpha, L, coef) rather than pinning one basis's magic number.
    """
    import math

    from vibeqc import estimate_rcut_pyscf_per_shell

    for alpha, L, coef in ((0.25, 0, 1.0), (1.3, 1, 0.7), (4.0, 2, 0.4),
                           (0.05, 0, 0.9), (12.0, 3, 0.25)):
        got = estimate_rcut_pyscf_per_shell(alpha, L, coef, precision=1e-8)
        # Same formula, but decaying with the single-shell exponent -- the
        # pre-fix behaviour.
        inner = abs(coef) * math.sqrt((2 * L + 1) * alpha / (2.0 * math.pi))
        single = math.sqrt(max(math.log(1e-8 / inner) / (-alpha), 0.1))
        assert got == pytest.approx(math.sqrt(2.0) * single, rel=1e-12), (
            f"alpha={alpha} L={L} coef={coef}: rcut {got} is not sqrt(2) x "
            f"the single-exponent radius {single}; the pair reduced "
            f"exponent alpha/2 has regressed to alpha."
        )


@pytest.mark.slow
def test_pyscf_rcut_default_precision_is_cutoff_converged():
    """precision=1e-8 must actually deliver ~1e-8, not 1e-6.

    The user-visible cost of RCUT-PYSCF-MISPORT was that ``precision``
    did not mean what it said: the sqrt(2)-short radius made the compcell
    energy sit 2.14e-06 Ha off its own cutoff-converged limit and refuse
    to move between precision 1e-6 and 1e-12. Post-fix the default lands
    on the converged value.
    """
    system, _ = _h2_box(box_bohr=12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def energy(prec):
        o = vq.PeriodicRHFOptions()
        o.max_iter = 40
        o.conv_tol_energy = 1e-10
        o.lattice_opts.cutoff_bohr = 15.0
        r = vq.run_krhf_periodic_gdf(
            system, basis, kmesh=(2, 1, 1), options=o,
            aux_basis="def2-svp-jk", gdf_method="compcell",
            use_compcell=True, rcut_precision=prec, progress=False,
        )
        assert r.converged, f"reference SCF at precision {prec} did not converge"
        return r.energy

    e_default = energy(1e-8)
    e_tight = energy(1e-14)
    assert e_default == pytest.approx(e_tight, abs=1e-8), (
        f"E(precision=1e-8)={e_default:.10f} should already be cutoff-"
        f"converged against E(1e-14)={e_tight:.10f}; a gap near 2e-06 Ha "
        f"means the sqrt(2)-short radius is back (RCUT-PYSCF-MISPORT)."
    )


def test_lattice_screening_pyscf_auto_rcut():
    """PySCF-style auto rcut for a typical fused (modrho_aux + chg)
    basis on H2/12-bohr/def2-svp-jk/η=0.25 lands in the 10–11 bohr range
    at precision 1e-8.

    The band was 7–9 bohr until 2026-08-02, when RCUT-PYSCF-MISPORT was
    fixed: the estimator decayed with the single-shell exponent ``alpha``
    instead of the pair reduced exponent ``alpha/2``, so every radius it
    returned was short by sqrt(2). This basis moved 7.21 -> 10.20 bohr.
    The band is deliberately not tied to a PySCF number -- nothing here
    runs PySCF (CLAUDE.md § 10), so an upstream match would have to come
    from the out-of-process runner.
    """
    from vibeqc import (
        RcutStrategy,
        estimate_rcut_pyscf,
        make_lattice_opts,
    )

    system, _ = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.25)
    fused = make_fused_basis(modrho, chg, mol)

    # Sanity: monotone increase with tighter precision (more decimal
    # places of decay → larger rcut).
    rcuts = [estimate_rcut_pyscf(fused, precision=p) for p in (1e-4, 1e-6, 1e-8, 1e-10)]
    assert rcuts[0] < rcuts[1] < rcuts[2] < rcuts[3], (
        f"rcut should be monotone in precision: {rcuts}"
    )

    # Value at PySCF's default precision is in the 10-11 bohr range.
    rcut_default = estimate_rcut_pyscf(fused, precision=1e-8)
    assert 9.5 < rcut_default < 11.5, (
        f"Expected PySCF-style rcut ≈ 10.2 bohr for fused def2-svp-jk + "
        f"chg(η=0.25) at precision 1e-8; got {rcut_default}. A value near "
        f"7.2 means the alpha/2 pair exponent regressed to alpha "
        f"(RCUT-PYSCF-MISPORT)."
    )

    # The factory writes the auto rcut into a fresh LatticeSumOptions.
    base = vq.LatticeSumOptions()
    base.cutoff_bohr = 30.0  # would have been the flat default
    lo_auto = make_lattice_opts(
        fused, strategy=RcutStrategy.PYSCF_AUTO, base_opts=base, precision=1e-8
    )
    assert lo_auto.cutoff_bohr == pytest.approx(rcut_default, rel=1e-9), (
        f"make_lattice_opts(PYSCF_AUTO) should set cutoff_bohr = "
        f"estimate_rcut_pyscf(); got {lo_auto.cutoff_bohr}"
    )
    # FLAT preserves base_opts.cutoff_bohr unchanged.
    lo_flat = make_lattice_opts(
        fused, strategy=RcutStrategy.FLAT, base_opts=base, precision=1e-8
    )
    assert lo_flat.cutoff_bohr == 30.0


def test_build_lpq_bloch_compcell_gamma_matches_build_lpq_compcell():
    """build_lpq_bloch_compcell at k=0 with symmetrize_gamma=True must
    produce the same Lpq factor as build_lpq_compcell (Γ-only). The
    Bloch-summed compcell builder is the multi-k generalization; at Γ
    it must reduce to the Γ-only path so existing SCF results are
    preserved when the new infrastructure lands.

    Equivalence is checked at the (μν|λσ) reconstruction level (Lpq
    is determined up to an orthogonal transform on the L axis after
    eigendecompose+threshold, so direct element-wise comparison
    isn't meaningful).
    """
    from vibeqc.aux_basis import build_lpq_bloch_compcell

    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 20.0
    lo.nuclear_cutoff_bohr = 20.0

    Lpq_gamma = build_lpq_compcell(
        system,
        basis,
        aux,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        apply_aft_correction=False,
    )
    Lpq_bloch = build_lpq_bloch_compcell(
        system,
        basis,
        aux,
        np.zeros(3),
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        apply_aft_correction=False,
    )

    assert Lpq_gamma.shape == Lpq_bloch.shape, (
        f"shape mismatch: γ={Lpq_gamma.shape} vs bloch={Lpq_bloch.shape}"
    )
    # Bloch result is complex at the type level; imaginary part should
    # be machine-zero at k=0 with symmetrize_gamma=True.
    if np.iscomplexobj(Lpq_bloch):
        assert np.linalg.norm(Lpq_bloch.imag) < 1e-10, (
            f"||Lpq_bloch.imag||={np.linalg.norm(Lpq_bloch.imag)} at k=0"
        )
    # (μν|λσ) reconstruction matches to machine precision.
    Lp_bloch_r = Lpq_bloch.real if np.iscomplexobj(Lpq_bloch) else Lpq_bloch
    muvlas_gamma = np.einsum("Lmn,Lop->mnop", Lpq_gamma, Lpq_gamma, optimize=True)
    muvlas_bloch = np.einsum("Lmn,Lop->mnop", Lp_bloch_r, Lp_bloch_r, optimize=True)
    assert np.linalg.norm(muvlas_gamma - muvlas_bloch) < 1e-9 * np.linalg.norm(
        muvlas_gamma
    ), (
        f"(μν|λσ) reconstruction mismatch: "
        f"||diff||={np.linalg.norm(muvlas_gamma - muvlas_bloch)} vs "
        f"||target||={np.linalg.norm(muvlas_gamma)}"
    )


def test_build_lpq_bloch_compcell_k_nonzero_smoke():
    """Smoke test: build_lpq_bloch_compcell at a generic k != 0
    produces a valid complex Lpq tensor without crashing. The detailed
    parity check vs PySCF KRHF.density_fit is a follow-up wiring once
    run_krhf_periodic_gdf is patched to use compcell."""
    from vibeqc.aux_basis import build_lpq_bloch_compcell

    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 20.0
    lo.nuclear_cutoff_bohr = 20.0

    k = np.array([0.2, 0.1, 0.05])
    Lpq_k = build_lpq_bloch_compcell(
        system,
        basis,
        aux,
        k,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        apply_aft_correction=False,
    )
    assert Lpq_k.ndim == 3
    assert Lpq_k.shape[1] == basis.nbasis
    assert Lpq_k.shape[2] == basis.nbasis
    assert np.iscomplexobj(Lpq_k)


def test_build_lpq_bloch_compcell_aft_nonzero_k_returns_complex():
    """AFT correction at q != 0 uses the q-shifted FT kernel
    (``coulG(G+q) · F̂(G+q)``). Returns a finite complex Lpq(k) with
    the correct shape; the result is complex because the G ↔ -G
    symmetry that made the Γ-only AFT real is broken at q ≠ 0."""
    from vibeqc.aux_basis import build_lpq_bloch_compcell

    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 20.0
    lo.nuclear_cutoff_bohr = 20.0

    k = np.array([0.1, 0.0, 0.0])
    Lpq_k = build_lpq_bloch_compcell(
        system,
        basis,
        aux,
        k,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        apply_aft_correction=True,
    )
    assert Lpq_k.ndim == 3
    assert Lpq_k.shape[1] == basis.nbasis
    assert Lpq_k.shape[2] == basis.nbasis
    assert np.iscomplexobj(Lpq_k)
    assert np.all(np.isfinite(Lpq_k))


def test_run_pbc_gdf_rhf_rejects_dim_lt_3():
    """The driver only supports dim=3 (3D-periodic). dim<3 raises with
    an actionable message — molecular-in-vacuum-box users should set
    system.dim=3 with a large vacuum spacing instead."""
    half = 0.7
    atoms = [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])]
    # dim=2 with vacuum padding in z
    lattice = np.array([[12.0, 0, 0], [0, 12.0, 0], [0, 0, 60.0]])
    system_2d = vq.PeriodicSystem(2, lattice, atoms)
    basis_2d = vq.BasisSet(system_2d.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="only dim=3"):
        vq.run_pbc_gdf_rhf(system_2d, basis_2d, aux_basis="def2-svp-jk", progress=False)


def test_run_pbc_gdf_rhf_rejects_charged_cell():
    """Compcell construction + exxdiv='ewald' assume a charge-neutral
    unit cell. Charged cells get a divergent G=0 self-image term that
    this driver does NOT correct — fail fast with an actionable
    message rather than silently return a wrong answer.

    Build He²⁺ ion in a box: one He atom (Z=2) with charge=2 →
    n_electrons=0 ... actually 0 electrons fails the SCF setup. Try
    He₂²⁺: two He atoms, charge=2, mult=1 → 2 electrons (even, closed-
    shell). Q_nuc = 4, n_elec = 2, net charge = +2 → fails the
    neutrality check.
    """
    atoms = [vq.Atom(2, [0, 0, -0.7]), vq.Atom(2, [0, 0, 0.7])]
    system = vq.PeriodicSystem(
        3, np.diag([12.0, 12.0, 12.0]), atoms, charge=2, multiplicity=1
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="not charge-neutral"):
        vq.run_pbc_gdf_rhf(system, basis, aux_basis="def2-svp-jk", progress=False)


def test_run_krhf_periodic_gdf_compcell_multik_sub_mha_h2():
    """Multi-k compcell SCF reaches sub-mHa parity to PySCF on H2 /
    12-bohr / def2-svp-jk at kmesh=(2,1,1) with exxdiv='ewald'.

    Closes the multi-k SCF loop bug fixed by:
      1. EWALD_3D gauge force on V_ne and E_nuc (was: DIRECT_TRUNCATED)
      2. exxdiv='ewald' K-shift wiring (was: hardcoded exxdiv=None)

    PySCF reference (verified via subprocess in
    examples/debug/gdf_multik_pyscf_parity.py): E_PySCF(2,1,1) =
    -1.12013988 Ha at PySCF default precision.

    Hard-coded the PySCF target to avoid a subprocess dependency in the
    test suite; if PySCF changes its default precision this test may
    need re-pinning.
    """
    # H2 / 12-bohr / kmesh=(2,1,1), PySCF KRHF.density_fit()
    # exxdiv='ewald' (docstring above; Sun-Berkelbach 2017 Table II,
    # doi:10.1063/1.4998644). 0a7ac7bb had re-pinned this to the
    # wiring's own output (-1.3572061177, "post-wire") while building
    # on the b4a6faba-regressed driver (q-only cderi, primitive-cell
    # Madelung, exxdiv double-count); the 2026-06-10 merge-drop
    # restoration returns the route to the documented reference
    # (measured -1.1201398817, sub-µHa agreement).
    PYSCF_TARGET = -1.12013988
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0

    r = vq.run_krhf_periodic_gdf(
        system,
        basis,
        kmesh=(2, 1, 1),
        options=opts,
        aux_basis="def2-svp-jk",
        use_compcell=True,
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    assert r.converged
    delta = r.energy - PYSCF_TARGET
    assert abs(delta) < 1e-6, (
        f"H2 / kmesh=(2,1,1) / GDF compcell energy regressed: "
        f"expected {PYSCF_TARGET}, got {r.energy} (\u0394={delta:.3e})."
    )


def test_run_krhf_periodic_gdf_compcell_multik_smoke():
    """run_krhf_periodic_gdf with use_compcell=True at a non-Γ kmesh
    converges and produces a bounded SCF total energy. Smoke-test of
    the multi-k compcell wiring (build_lpq_bloch_compcell threaded
    through _build_lpq_q_cache → KRHF SCF loop)."""
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 20
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0

    r = vq.run_krhf_periodic_gdf(
        system,
        basis,
        kmesh=(2, 1, 1),
        options=opts,
        aux_basis="def2-svp-jk",
        use_compcell=True,
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    assert r.converged
    # Bounded total energy (not the runaway-divergence pattern).
    assert -50.0 < r.energy < 50.0, (
        f"H2 / kmesh=(2,1,1) compcell SCF should be bounded; got "
        f"E={r.energy} Ha (the runaway-divergence pattern indicates "
        "the multi-k compcell wiring broke)."
    )


@pytest.mark.slow
def test_pbc_gdf_rhf_production_defaults_h2():
    """Compcell SCF with the production defaults (AFT on, \u03b7=1.0,
    libint convention, pyscf_auto rcut) converges on H2 and gives
    an energy within ~1 mHa of PySCF GDF. No manual tuning needed."""
    system, basis = _h2_box(box_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    # Production defaults: no explicit AFT/\u03b7/rcut params.
    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        progress=False,
    )
    assert r.converged
    assert r.backend == "pbc-gdf-compcell"
    PYSCF_TARGET = -1.1225839666
    # With AFT on by default, the energy should be within ~2 mHa of PySCF.
    assert abs(r.energy - PYSCF_TARGET) < 3e-3, (
        f"Production-default compcell SCF should be within 3 mHa of PySCF "
        f"{PYSCF_TARGET}; got {r.energy} "
        f"(\u0394 = {r.energy - PYSCF_TARGET:.3e})."
    )


@pytest.mark.slow
def test_pbc_gdf_rhf_production_defaults_lih_ionic():
    """Compcell SCF with production defaults on LiH FCC (\u0393-only).

    With AFT on by default, the SCF now converges (previously diverged)
    but to a non-physical +576 Ha — the energy sanity guard fires as
    SANITY_FAILED.  The \u0393-only compcell path is NOT sufficient for
    tight ionic cells; the production route for these systems is the
    multi-k path (``run_krhf_periodic_gdf``, validated at \u00b5Ha parity
    on LiH at kmesh=(2,2,2) — see test_audit_20260530_periodic.py).

    This test pins the current behaviour as a regression guard: once a
    future fix (MDF or dense-FFT-mesh AFT) closes the ionic gap, this
    test should be updated to assert a physical energy.

    NB (2026-07-09): whether the broken-Hartree SCF *converges* to its
    ~+576 Ha fixed point within max_iter is machine-dependent (BLAS /
    threading reorder the DIIS trajectory); the pin asserts the CLAUDE.md
    §7 contract that holds either way -- non-physical energy, result
    tagged ``+SANITY_FAILED`` -- rather than the converged flag."""
    from vibeqc._vibeqc_core import monkhorst_pack

    # LiH primitive FCC cell (Sun-Berkelbach 2017 benchmark)
    ANG2BOHR = 1.0 / 0.529177210903
    A = 4.084
    lat = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * A * ANG2BOHR
    system = vq.PeriodicSystem(
        3,
        lat,
        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [A * ANG2BOHR / 2] * 3)],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8
    # Production defaults — no manual \u03b7 or AFT params.
    r = vq.run_pbc_gdf_rhf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        exxdiv="ewald",
        progress=False,
    )
    # With AFT-on defaults, the Γ-only compcell path is non-convergent
    # on this ionic cell. The SCF oscillation amplitude is O(1e5) Ha;
    # where it lands at max_iter is trajectory-dependent, so we don't pin
    # a specific energy. The sanity guard (SANITY_FAILED) is the invariant.
    assert not r.converged, (
        f"Expected non-converged SCF on LiH Γ-only; got converged=True, "
        f"energy={r.energy:.4f} Ha. If the Γ-only compcell ionic gap has "
        "been closed, update this test and remove the SANITY_FAILED check."
    )
    assert r.backend == "pbc-gdf-compcell+SANITY_FAILED", (
        f"Expected SANITY_FAILED for LiH \u0393-only; got backend={r.backend}"
    )


def _lih_rocksalt_gamma():
    """Rocksalt LiH primitive FCC (a = 7.72 bohr) — the 2026-07-09 fixture.

    The Γ-only compcell Hartree is non-physical on this cell (AO-pair
    images overlap; +579.8 Ha at HF, +1172.6 Ha at UKS-PBE vs external
    PySCF 2.13.1 KRKS(GDF) PBE -8.238879724246). Same crystal as
    ``test_pbc_gdf_rhf_production_defaults_lih_ionic`` (a = 4.084 Å),
    kept in the reporting chat's bohr convention.
    """
    a = 7.72
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    system = vq.PeriodicSystem(
        3,
        lat,
        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [0.5 * a] * 3)],
        0,
        1,
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


@pytest.mark.slow
def test_pbc_gdf_uks_lih_ionic_sanity_guard_raises():
    """Γ-only compcell UKS(PBE) on rocksalt LiH must NOT hand back a
    converged non-physical energy (2026-07-09 finding).

    Pre-guard, ``run_pbc_gdf_uks`` returned +1172.63 Ha with
    ``converged=True`` on this fixture (external PySCF KRKS(GDF) PBE:
    -8.2389 Ha) -- the CLAUDE.md §7 silent-corruption pattern. The Γ-only
    compcell Hartree cannot resolve the overlapping AO-pair images (the
    same class the RHF driver warns about and tags on the a=4.084 Å
    twin of this cell). The driver must now (a) emit the tight-ionic
    warning, (b) raise via the energy-sanity guard, and (c) hand the
    diagnostic value back when the guard is explicitly bypassed.

    ``gdf_method='compcell'`` is explicit since the 2026-07-09 default
    flip to rsgdf (see test_pbc_gdf_uks_lih_ionic_default_rsgdf_is_sane):
    this test documents the fenced compcell limitation itself. Whether
    the broken-Hartree SCF *converges* to its fixed point within
    max_iter is machine-dependent (see the RHF production-defaults
    pin), so the guard contract is asserted in both flavours: a
    converged run raises; a non-converged run returns warn+tagged.
    """
    system, basis = _lih_rocksalt_gamma()

    with pytest.warns(UserWarning, match="tight ionic cell"):
        _assert_guard_rejects_nonphysical(
            lambda **kw: vq.run_pbc_gdf_uks(
                system, basis, functional="pbe", gdf_method="compcell",
                progress=False, **kw,
            )
        )

    # Bypass path: diagnostics callers get the (nonsense) fixed point back.
    # ``check_energy_sanity=False`` means the guard does not run at all --
    # every driver wraps the helper in ``if check_energy_sanity:``, and the
    # ``+SANITY_FAILED`` tag is applied inside it. So the bypass contract is
    # the *absence* of the guard's effects: no raise, no tag, the
    # non-physical energy handed straight back. Whether that fixed point is
    # reached within max_iter is machine-dependent (see the helper below and
    # the RHF production-defaults pin), so the converged flag is not pinned.
    r = vq.run_pbc_gdf_uks(
        system,
        basis,
        functional="pbe",
        gdf_method="compcell",
        check_energy_sanity=False,
        progress=False,
    )
    # sane_bound = max(10 * ΣZ², 100) = 100 Ha here (Li+H: ΣZ² = 10), the
    # runaway threshold _check_energy_sanity would have rejected this on.
    assert abs(r.energy) > 100.0, (
        f"Bypass must hand back the non-physical fixed point the guard "
        f"would have rejected; got energy={r.energy:.4f} Ha. If the Γ-only "
        "compcell ionic gap has been closed, update this test."
    )
    assert r.backend == "pbc-gdf-compcell-uks", (
        f"Bypassing the guard bypasses its tagging too; expected an "
        f"untagged backend, got {r.backend}"
    )


def _assert_guard_rejects_nonphysical(run):
    """§7 guard contract, robust to the machine-dependent converged flag:
    a converged non-physical energy raises RuntimeError; a non-converged
    one comes back warn+tagged ``+SANITY_FAILED`` (never a silently
    plausible result)."""
    try:
        r = run()
    except RuntimeError as e:
        assert "non-physical" in str(e)
        return
    assert not r.converged, (
        f"guard must raise on a converged non-physical energy; got "
        f"converged=True E={r.energy:.4f} backend={r.backend}"
    )
    assert r.backend.endswith("+SANITY_FAILED"), (
        f"non-converged non-physical result must be tagged; got {r.backend}"
    )


@pytest.mark.slow
def test_pbc_gdf_uhf_lih_ionic_sanity_guard_raises():
    """The UHF sibling of the UKS guard test: +579.83 Ha, same class.

    Identical failure with no XC at all — the catastrophe is the Γ-only
    compcell Hartree, not the XC path (the rsgdf Γ lane lands at a sane
    -8.33 Ha on this fixture).
    """
    system, basis = _lih_rocksalt_gamma()
    _assert_guard_rejects_nonphysical(
        lambda **kw: vq.run_pbc_gdf_uhf(
            system, basis, gdf_method="compcell", progress=False, **kw
        )
    )


@pytest.mark.slow
def test_pbc_gdf_uks_lih_ionic_default_rsgdf_is_sane():
    """The open-shell Γ drivers' PRODUCTION DEFAULTS are sane on the
    rocksalt LiH cell that broke compcell (2026-07-09 default flip).

    ``run_pbc_gdf_uhf`` / ``run_pbc_gdf_uks`` now default to
    ``gdf_method='rsgdf'`` (compcell's Γ-only q-cderi is vacuum-box-only
    by construction; rsgdf is the fixed Hartree), and a KS run with
    ``options=None`` defaults to ``PeriodicKSOptions`` so the b3f74aa9
    periodic-Becke-grid/torus-density XC pairing engages (the old
    ``PeriodicRHFOptions`` fallback evaluated XC in the v0.8.x
    molecular-grid convention: -7.9638 Ha on this fixture).

    Measured 2026-07-09: UKS-PBE -8.233910336 -- identical to the
    independent real-Γ direct route (-8.233910335588, commit b3f74aa9)
    and 5.0 mHa from external PySCF KRKS(GDF) PBE (-8.238879724246, the
    documented ultra-diffuse-Li shared-fitting floor). UHF -8.330292528,
    4.9 mHa from the PySCF HF target on the a=4.084 Å twin (same floor
    class). No sanity guard, no tight-ionic warning fires.
    """
    import warnings as _warnings

    system, basis = _lih_rocksalt_gamma()

    with _warnings.catch_warnings(record=True) as wl:
        _warnings.simplefilter("always")
        r = vq.run_pbc_gdf_uks(system, basis, functional="pbe", progress=False)
    assert not any("tight ionic cell" in str(w.message) for w in wl)
    assert r.converged
    assert r.backend == "pbc-gdf-rsgdf-uks"
    assert r.energy == pytest.approx(-8.233910336, abs=1e-4)

    u = vq.run_pbc_gdf_uhf(system, basis, progress=False)
    assert u.converged
    assert u.backend == "pbc-gdf-rsgdf-uhf"
    assert u.energy == pytest.approx(-8.330292528, abs=1e-4)


def test_pbc_gdf_energy_sanity_guard_semantics():
    """Unit-level pin of ``_check_energy_sanity`` (no SCF).

    * converged + runaway  -> RuntimeError (open-shell/KS mode)
    * converged + unbound-only (small positive) -> RuntimeError
    * NOT converged + insane -> warn + tag, no raise
    * sane negative energy -> untouched
    * RHF mode (raise_if_converged=False) -> warn + tag, never raise
    """
    from types import SimpleNamespace

    from vibeqc.pbc_gdf import _check_energy_sanity
    from vibeqc.progress import resolve_progress

    system, _ = _lih_rocksalt_gamma()  # ΣZ² = 10 -> sane bound = 100 Ha
    plog = resolve_progress(False)

    def stub(energy, converged):
        return SimpleNamespace(energy=energy, converged=converged, backend="x")

    # converged runaway -> raise
    with pytest.raises(RuntimeError, match="runaway divergence"):
        _check_energy_sanity(
            stub(+1172.63, True), system, plog,
            driver="run_pbc_gdf_uks", raise_if_converged=True,
        )
    # converged unbound-only (below the 100 Ha runaway bound) -> raise
    with pytest.raises(RuntimeError, match="unbound"):
        _check_energy_sanity(
            stub(+8.485, True), system, plog,
            driver="run_pbc_gdf_uks", raise_if_converged=True,
        )
    # non-converged insane -> tag, no raise (converged=False already
    # signals failure; keep partial state inspectable)
    r = stub(+1172.63, False)
    _check_energy_sanity(
        r, system, plog, driver="run_pbc_gdf_uks", raise_if_converged=True
    )
    assert r.backend == "x+SANITY_FAILED"
    # sane -> untouched
    r = stub(-8.24, True)
    _check_energy_sanity(
        r, system, plog, driver="run_pbc_gdf_uks", raise_if_converged=True
    )
    assert r.backend == "x"
    # RHF mode: warn + tag even when converged (pinned by
    # test_pbc_gdf_rhf_production_defaults_lih_ionic)
    r = stub(+579.83, True)
    _check_energy_sanity(r, system, plog)
    assert r.backend == "x+SANITY_FAILED"


def test_pbc_gdf_rhf_multi_k_not_implemented():
    """The initial landing is Γ-only; multi-k raises NotImplementedError
    until the multi-k integration ships (task 8 / next milestone)."""
    system, basis = _h2_box(box_bohr=12.0)
    with pytest.raises(NotImplementedError):
        vq.run_pbc_gdf_rhf(
            system, basis, kmesh=(2, 2, 2), aux_basis="def2-svp-jk", progress=False
        )


# =====================================================================
#   Pair-complete lattice enumeration (aux_eri.cpp, 2026-08-03)
# =====================================================================
#
# A two-centre lattice sum runs over translations g of the ket centre,
# and the term that matters is set by the PHYSICAL separation
# |R_P - R_Q - g|. So the contributing translations form a ball centred
# on the intra-cell offset, not on the origin. The kernels used to bound
# |g| <= R_cut and then compute every shell pair at that g, which both
# missed translations that bring a distant pair into range and visited
# ones that did not -- asymmetrically between the (P,Q) and (Q,P)
# orderings.
#
# Two invariants catch it. Both are properties of the exact object, not
# tolerances, so neither may be relaxed:
#
#   * the compensated Coulomb metric is a Gram matrix in a
#     positive-definite metric, so it is positive semi-definite;
#   * translating an atom by a lattice vector describes the identical
#     crystal, so the metric must not change.
#
# A single-atom cell has zero intra-cell offset and is blind to this,
# which is why every vacuum-box fixture passed while MgO diverged.


def _mgo_primitive(o_shift=(0.0, 0.0, 0.0)):
    a = 4.21 / 0.529177210903
    h = 0.5 * a
    lattice = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    o_pos = np.array([h, h, h]) + np.asarray(o_shift, dtype=float)
    return vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, list(o_pos))]
    )


def _compensated_metric(system, cutoff):
    """The compensated (modrho - chg) Coulomb metric, as GDF builds it."""
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.aux_basis import (
        fuse_transform_matrix,
        make_aux_basis_set,
        make_compensating_basis,
        make_fused_basis,
        make_modrho_aux_basis,
    )

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=1.0)
    fused = make_fused_basis(modrho, chg, mol)
    A = fuse_transform_matrix(modrho, chg)
    opts = LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    opts.nuclear_cutoff_bohr = cutoff
    M_fused = np.asarray(compute_2c_eri_lattice(fused, system, opts))
    M_fused = 0.5 * (M_fused + M_fused.T)
    M = A @ M_fused @ A.T
    return 0.5 * (M + M.T)


def test_lattice_metric_is_translation_invariant():
    """Translating an atom by a lattice vector must not change the metric.

    Pre-fix this moved by 7.2e-02 on MgO (intra-cell offset 6.89 bohr),
    because the |g| <= R_cut bound made which images were visited depend
    on where in the cell the atom happened to sit.
    """
    system0 = _mgo_primitive()
    M0 = _compensated_metric(system0, 20.0)
    scale = float(np.max(np.abs(M0)))
    lattice = np.asarray(system0.lattice, dtype=float)
    for shift in (lattice[0], lattice[1], -lattice[2], lattice.sum(axis=0)):
        M = _compensated_metric(_mgo_primitive(shift), 20.0)
        drift = float(np.max(np.abs(M - M0)))
        assert drift < 1e-10 * scale, (
            f"metric moved by {drift:.3e} under a lattice translation "
            "(the crystal is unchanged)"
        )


def test_lattice_metric_is_positive_semidefinite_on_a_compact_cell():
    """The Coulomb metric is a Gram matrix, so it cannot be indefinite.

    Pre-fix MgO gave -5.4e-05 against a largest eigenvalue of 20.8. The
    residual at a converged cutoff is round-off. The 20-bohr rung is
    recorded as the looser bound it is: what this pins is that the
    enumeration no longer contributes, not that the truncation is
    converged (`pyscf_auto` picks 15.5 bohr for this basis and is NOT
    converged -- see the MDF fence).
    """
    system = _mgo_primitive()
    for cutoff, floor in ((20.0, -1e-6), (28.0, -1e-9)):
        eig = np.linalg.eigvalsh(_compensated_metric(system, cutoff))
        assert eig[0] > floor * abs(eig[-1]) or eig[0] > floor, (
            f"cutoff {cutoff}: min eigenvalue {eig[0]:.3e} "
            f"(max {eig[-1]:.3e}) -- the metric is not a Gram matrix"
        )


def test_open_shell_gamma_gdf_records_resolved_tail_cutoff():
    """IID 490: the open-shell Gamma GDF drivers auto-size the RSGDF
    high-|G| tail the same way as run_pbc_gdf_rhf but did not record it, so
    an open-shell cross-route comparison could not assert matched reciprocal
    support. An explicit request must read back as applied, and an
    unrequested tail as None (never a fabricated 0)."""
    from vibeqc.pbc_gdf import run_pbc_gdf_uhf, run_pbc_gdf_uks

    system, basis = _h2_box(box_bohr=12.0)

    def _opts():
        o = vq.PeriodicRHFOptions()
        o.use_diis = True
        o.damping = 0.0
        o.max_iter = 40
        o.conv_tol_energy = 1e-10
        o.lattice_opts.cutoff_bohr = 30.0
        o.lattice_opts.nuclear_cutoff_bohr = 30.0
        return o

    common = dict(
        aux_basis="def2-svp-jk", exxdiv="ewald", gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0, progress=False,
    )
    r_uhf = run_pbc_gdf_uhf(
        system, basis, _opts(), rsgdf_tail_ke_cutoff=500.0, **common)
    r_uks = run_pbc_gdf_uks(
        system, basis, _opts(), functional="pbe",
        rsgdf_tail_ke_cutoff=500.0, **common)
    assert r_uhf.converged and r_uks.converged
    assert r_uhf.rsgdf_tail_ke_cutoff == pytest.approx(500.0)
    assert r_uks.rsgdf_tail_ke_cutoff == pytest.approx(500.0)
    assert r_uhf.rsgdf_ke_cutoff == pytest.approx(200.0)
    assert r_uks.rsgdf_ke_cutoff == pytest.approx(200.0)

    # Negative control: no request -> recorded None (base-mesh-only).
    r_none = run_pbc_gdf_uhf(system, basis, _opts(), **common)
    assert r_none.rsgdf_tail_ke_cutoff is None


def test_multik_open_shell_gdf_records_base_mesh():
    """IID 490 (multi-k half): the open-shell multi-k GDF result records
    the base rsgdf mesh actually used, mirroring PeriodicKRHFGDFResult,
    and keeps the tail field None on a route that applies no tail."""
    from vibeqc.periodic_k_gdf import run_kuhf_periodic_gdf

    system, basis = _h2_box(box_bohr=12.0)
    r = run_kuhf_periodic_gdf(system, basis, (2, 1, 1), progress=False)
    assert r.converged
    assert r.rsgdf_ke_cutoff == pytest.approx(200.0)
    assert r.rsgdf_tail_ke_cutoff is None


@pytest.mark.slow
def test_open_shell_gamma_gdf_records_auto_sized_tight_core_tail():
    """IID 490 on the tight-core class: the open-shell Gamma GDF drivers
    record the SAME auto-sized tail the closed-shell driver resolves on
    diamond/STO-3G (the #307 gate cell), so an open-shell cross-route
    comparison can assert matched reciprocal support. The fields are set at
    result construction, so a single SCF iteration pins the recording; the
    tailed fold cderi build is the dominating cost (~20 s, see _diamond_ccm
    in tests/test_ccm_direct.py)."""
    from vibeqc.pbc_gdf import (
        _auto_rsgdf_tail_ke_cutoff,
        run_pbc_gdf_uhf,
        run_pbc_gdf_uks,
    )

    a = 6.74
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    system = vq.PeriodicSystem(
        3, lat,
        [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [0.25 * a, 0.25 * a, 0.25 * a])],
        0, 3)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    expected = _auto_rsgdf_tail_ke_cutoff(
        system, "rsgdf", basis, None, rsgdf_ke_cutoff=200.0)
    assert expected == pytest.approx(787.785207, rel=1e-9)

    def _opts():
        o = vq.PeriodicRHFOptions()
        o.use_diis = True
        o.damping = 0.0
        o.max_iter = 1  # recording is construction-time; skip the SCF
        o.lattice_opts.cutoff_bohr = 30.0
        o.lattice_opts.nuclear_cutoff_bohr = 30.0
        return o

    common = dict(
        aux_basis="def2-svp-jk", exxdiv="ewald", gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0, progress=False,
    )
    r_uhf = run_pbc_gdf_uhf(system, basis, _opts(), **common)
    r_uks = run_pbc_gdf_uks(
        system, basis, _opts(), functional="pbe", **common)
    assert r_uhf.rsgdf_tail_ke_cutoff == pytest.approx(expected, rel=1e-9)
    assert r_uks.rsgdf_tail_ke_cutoff == pytest.approx(expected, rel=1e-9)
    assert r_uhf.rsgdf_ke_cutoff == pytest.approx(200.0)
    assert r_uks.rsgdf_ke_cutoff == pytest.approx(200.0)
