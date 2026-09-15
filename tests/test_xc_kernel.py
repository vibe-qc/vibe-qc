"""Phase D2c-KS-LDA — XC kernel matvec validation.

The closed-shell unpolarised LDA XC kernel ships first; the GGA path
(5 extra terms, Pople-Gill-Johnson CPL 199, 557 1992) and the
polarized LDA / GGA paths follow in subsequent commits.

The unambiguous correctness witness is the finite-difference of the
static XC potential V_xc(D):

    W^XC[δD] = lim_{ε→0} (V_xc(D + ε δD) − V_xc(D − ε δD)) / (2ε)

That's what the analytic implementation in
``cpp/src/xc_kernel.cpp::UnpolarisedLDAXCKernelBuilder::apply`` must
reproduce. The test below builds V_xc entirely in NumPy from the
existing ``Functional.eval_unpolarised`` libxc binding (no dependence
on the C++ XC kernel), perturbs D in both directions, and checks the
central-difference result against the analytic matvec.

Coverage:
  1. LDA functional ("LDA", which routes through libxc's SVWN5 default)
     on H₂O / sto-3g: FD vs analytic to ~1e-6.
  2. GGA functional ("PBE") raises a clear NotImplementedError-style
     RuntimeError pointing at the follow-up commit.
  3. Builder API: a fresh ``apply`` returns symmetric output and
     scales linearly with the perturbation magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    Atom,
    BasisSet,
    Functional,
    GridOptions,
    Molecule,
    build_grid,
    compute_overlap,
    evaluate_ao_with_gradient,
    make_batched_polarised_xc_kernel_builder,
    make_polarised_gga_xc_kernel_builder,
    make_polarised_lda_xc_kernel_builder,
    make_polarised_xc_kernel_builder,
    make_unpolarised_xc_kernel_builder,
    sad_density,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

def _h2o():
    return Molecule([
        Atom(8, [0.0, 0.0,  0.117 * 1.8897259886]),
        Atom(1, [0.0,  0.755 * 1.8897259886, -0.471 * 1.8897259886]),
        Atom(1, [0.0, -0.755 * 1.8897259886, -0.471 * 1.8897259886]),
    ])


@pytest.fixture
def h2o_setup():
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    # A reasonable density to linearise around — SAD guess is enough.
    D = sad_density(mol, basis)
    # Smaller-than-default grid for test speed; the FD comparison is
    # exact in grid weights (both sides use the same grid).
    grid_opts = GridOptions()
    grid_opts.n_radial = 30
    grid_opts.n_theta = 14
    grid_opts.n_phi = 28
    grid = build_grid(mol, grid_opts)
    chi, gx, gy, gz = evaluate_ao_with_gradient(basis, grid.points)
    return {
        "mol": mol, "basis": basis, "D": np.asarray(D),
        "grid": grid, "chi": np.asarray(chi),
        "dchi_x": np.asarray(gx), "dchi_y": np.asarray(gy),
        "dchi_z": np.asarray(gz),
    }


def _python_vxc_lda(func, grid, chi, D):
    """V_xc(D) for an unpolarised LDA functional, built from the
    libxc Python binding (independent of the C++ XC code).

    V_xc_{μν} = Σ_g w_g · v_ρ(g) · χ_μ(g) χ_ν(g)
              = χᵀ · diag(w · v_ρ) · χ
    """
    chiD = chi @ D
    rho = np.einsum("gp,gp->g", chiD, chi)
    sigma = np.zeros(0)
    exc, v_rho, v_sigma = func.eval_unpolarised(rho, sigma)
    w_vrho = np.asarray(grid.weights) * v_rho
    return chi.T @ (w_vrho[:, None] * chi)


def _python_vxc_gga(func, grid, chi, dchi, D):
    """V_xc(D) for an unpolarised GGA functional, built from libxc.

    Mirrors cpp/src/rks.cpp::build_xc — independent of the C++ XC
    kernel code (which is what the FD test exercises against).
    """
    chiD = chi @ D
    rho = np.einsum("gp,gp->g", chiD, chi)
    # ∇ρ_c(g) = 2 · Σ_μ (χD)[g,μ] · ∂_c χ_μ(g)   (D symmetric)
    grho = [2.0 * np.einsum("gp,gp->g", chiD, dchi[c]) for c in range(3)]
    sigma = grho[0] ** 2 + grho[1] ** 2 + grho[2] ** 2

    _exc, v_rho, v_sigma = func.eval_unpolarised(rho, sigma)
    w = np.asarray(grid.weights)
    V = chi.T @ ((w * v_rho)[:, None] * chi)
    # GGA piece: V_{μν} += Σ_g 2 w v_σ · (F_μ χ_ν + χ_μ F_ν)
    # with F[g,μ] = Σ_c ∇ρ_c(g) ∂_c χ_μ(g).
    F = (grho[0][:, None] * dchi[0]
         + grho[1][:, None] * dchi[1]
         + grho[2][:, None] * dchi[2])
    u = 2.0 * w * v_sigma
    Fu = F.T @ (u[:, None] * chi)
    V = V + Fu + Fu.T
    return V


def _random_symmetric(n, rng):
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


# ---------------------------------------------------------------------------
# 1. FD vs analytic — the headline correctness witness.
# ---------------------------------------------------------------------------

def test_lda_kernel_matches_finite_difference_vxc(h2o_setup):
    """Central-difference of V_xc(D) along δD must match W^XC[δD] to ~1e-6.

    This is the unambiguous CPKS/CPHF Hessian-vector witness: the second
    derivative of E_xc[ρ[D]] = first derivative of V_xc(D). Any prefactor
    bug or grid-weight slip in the analytic implementation fails here.
    """
    setup = h2o_setup
    func = Functional("LDA")  # libxc SVWN5 default
    assert func.kind == vq.XCKind.LDA

    rng = np.random.default_rng(seed=0xD2C)
    delta_D = _random_symmetric(setup["basis"].nbasis, rng)
    # Scale δD so a 1e-5 step is well-resolved relative to ‖D‖_F.
    delta_D *= np.linalg.norm(setup["D"]) / max(
        np.linalg.norm(delta_D), 1e-12)

    kernel = make_unpolarised_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"],
        setup["D"],
    )
    W_analytic = np.asarray(kernel.apply(delta_D))

    eps = 1e-5
    V_plus = _python_vxc_lda(
        func, setup["grid"], setup["chi"], setup["D"] + eps * delta_D)
    V_minus = _python_vxc_lda(
        func, setup["grid"], setup["chi"], setup["D"] - eps * delta_D)
    W_fd = (V_plus - V_minus) / (2.0 * eps)

    # Central difference is O(eps²); with eps=1e-5 we expect ~1e-10
    # truncation error, but the LDA kernel evaluation has ~1e-13 noise
    # per grid point times ~10k points → ~1e-9 floor. Allow 1e-6.
    rel_err = np.linalg.norm(W_analytic - W_fd) / np.linalg.norm(W_fd)
    assert rel_err < 1e-6, (
        f"W^XC analytic vs FD: relative error {rel_err:.3e} "
        f"(‖W_an‖={np.linalg.norm(W_analytic):.3e}, "
        f"‖W_fd‖={np.linalg.norm(W_fd):.3e})"
    )


# ---------------------------------------------------------------------------
# 2. GGA FD vs analytic — same witness as LDA but with the 5-term
#    Pople-Gill-Johnson kernel (v2rho2 + v2rhosigma + v2sigma2 + v_σ·∇δρ
#    + cross terms grouped into 3 contributions).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["PBE", "BLYP"])
def test_gga_kernel_matches_finite_difference_vxc(h2o_setup, name):
    """Central-difference of V_xc(D) for a GGA functional must match
    the analytic W^XC[δD] from the C++ kernel to ~1e-5.

    More forgiving tolerance than LDA because the GGA kernel
    involves more arithmetic per grid point (three matrix accumulations
    + the δσ chain) — round-off floor is a few × 1e-9 on H₂O / sto-3g.
    """
    setup = h2o_setup
    func = Functional(name)
    assert func.kind == vq.XCKind.GGA

    rng = np.random.default_rng(seed=0xD2C + hash(name) % 100)
    delta_D = _random_symmetric(setup["basis"].nbasis, rng)
    delta_D *= np.linalg.norm(setup["D"]) / max(
        np.linalg.norm(delta_D), 1e-12)

    kernel = make_unpolarised_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"],
        setup["D"],
    )
    W_analytic = np.asarray(kernel.apply(delta_D))

    eps = 1e-5
    dchi = (setup["dchi_x"], setup["dchi_y"], setup["dchi_z"])
    V_plus = _python_vxc_gga(
        func, setup["grid"], setup["chi"], dchi, setup["D"] + eps * delta_D)
    V_minus = _python_vxc_gga(
        func, setup["grid"], setup["chi"], dchi, setup["D"] - eps * delta_D)
    W_fd = (V_plus - V_minus) / (2.0 * eps)

    rel_err = np.linalg.norm(W_analytic - W_fd) / np.linalg.norm(W_fd)
    assert rel_err < 1e-5, (
        f"{name}: W^XC analytic vs FD rel err {rel_err:.3e} "
        f"(‖W_an‖={np.linalg.norm(W_analytic):.3e}, "
        f"‖W_fd‖={np.linalg.norm(W_fd):.3e})"
    )


def test_metagga_kernel_raises_with_roadmap_pointer(h2o_setup):
    """Meta-GGA (tau-dependent) fxc is not implemented; the factory
    should not silently fall through. Today's `Functional` only
    classifies LDA/GGA, so meta-GGA functionals route to the GGA
    path and either work or produce wrong fxc — the kernel does NOT
    silently lie. This guard test is a placeholder: when the
    `Functional` class grows a META_GGA kind, the kernel factory
    should raise with the same pointer as polarised GGA does."""
    pytest.skip(
        "Meta-GGA kind not yet distinguished in Functional (Phase 17e+); "
        "kernel factory test deferred until then.")


# ---------------------------------------------------------------------------
# 3. Builder invariants.
# ---------------------------------------------------------------------------

def test_lda_kernel_output_is_symmetric(h2o_setup):
    """W^XC[δD] is symmetric in μ↔ν by construction (the AO product
    χ_μ χ_ν is symmetric and v2rho2 is a scalar at each grid point)."""
    setup = h2o_setup
    func = Functional("LDA")
    rng = np.random.default_rng(1)
    delta_D = _random_symmetric(setup["basis"].nbasis, rng)
    kernel = make_unpolarised_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"],
        setup["D"],
    )
    W = np.asarray(kernel.apply(delta_D))
    assert np.allclose(W, W.T, atol=1e-12)


def test_lda_kernel_is_linear_in_perturbation(h2o_setup):
    """W^XC is linear in δD by construction. W^XC[2·δD] == 2·W^XC[δD]."""
    setup = h2o_setup
    func = Functional("LDA")
    rng = np.random.default_rng(2)
    delta_D = _random_symmetric(setup["basis"].nbasis, rng)
    kernel = make_unpolarised_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"],
        setup["D"],
    )
    W1 = np.asarray(kernel.apply(delta_D))
    W2 = np.asarray(kernel.apply(2.0 * delta_D))
    assert np.allclose(W2, 2.0 * W1, atol=1e-10)


def test_lda_kernel_apply_does_not_mutate_input(h2o_setup):
    """``apply`` is const — passing the same δD repeatedly must give
    identical output and not mutate the input array."""
    setup = h2o_setup
    func = Functional("LDA")
    rng = np.random.default_rng(3)
    delta_D = _random_symmetric(setup["basis"].nbasis, rng)
    delta_D_original = delta_D.copy()
    kernel = make_unpolarised_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"],
        setup["D"],
    )
    W1 = np.asarray(kernel.apply(delta_D))
    W2 = np.asarray(kernel.apply(delta_D))
    assert np.array_equal(delta_D, delta_D_original)
    assert np.array_equal(W1, W2)


# ---------------------------------------------------------------------------
# 4. Phase 17e — open-shell polarised GGA XC kernel.
#
# Same FD witness as the closed-shell paths, now per-spin: the analytic
# W^XC_σ[δD_α, δD_β] must reproduce the central difference of the static
# UKS GGA potential V_xc,σ(D_α, D_β) when *both* spin densities are
# perturbed together. The αβ coupling (v2rho2_ab, v2rhosigma_*,
# v2sigma2_*, the σ_ab cross term) is what makes W_α depend on δD_β.
# ---------------------------------------------------------------------------

def _python_vxc_polarised_gga(func, grid, chi, dchi, Da, Db):
    """Per-spin V_xc(D_α, D_β) for an open-shell GGA functional, built
    from the libxc Python binding. Mirrors cpp/src/uks.cpp::build_xc —
    independent of the C++ XC-kernel code under test."""
    w = np.asarray(grid.weights)
    chiDa = chi @ Da
    chiDb = chi @ Db
    rho_a = np.einsum("gp,gp->g", chiDa, chi)
    rho_b = np.einsum("gp,gp->g", chiDb, chi)
    grho_a = [2.0 * np.einsum("gp,gp->g", chiDa, dchi[c]) for c in range(3)]
    grho_b = [2.0 * np.einsum("gp,gp->g", chiDb, dchi[c]) for c in range(3)]
    s_aa = sum(grho_a[c] ** 2 for c in range(3))
    s_bb = sum(grho_b[c] ** 2 for c in range(3))
    s_ab = sum(grho_a[c] * grho_b[c] for c in range(3))

    out = func.eval_polarised(rho_a, rho_b, s_aa, s_ab, s_bb)
    _exc, v_ra, v_rb, v_saa, v_sab, v_sbb = (np.asarray(x) for x in out)

    V_a = chi.T @ ((w * v_ra)[:, None] * chi)
    V_b = chi.T @ ((w * v_rb)[:, None] * chi)
    # Per-spin flow vector f_σ = 2 v_σσσ ∇ρ_σ + v_σαβ ∇ρ_σ'.
    fa = [2.0 * v_saa * grho_a[c] + v_sab * grho_b[c] for c in range(3)]
    fb = [2.0 * v_sbb * grho_b[c] + v_sab * grho_a[c] for c in range(3)]
    Fa = sum(fa[c][:, None] * dchi[c] for c in range(3))
    Fb = sum(fb[c][:, None] * dchi[c] for c in range(3))
    Ma = Fa.T @ (w[:, None] * chi)
    Mb = Fb.T @ (w[:, None] * chi)
    return V_a + Ma + Ma.T, V_b + Mb + Mb.T


@pytest.mark.parametrize("name", ["PBE", "BLYP", "B3LYP"])
def test_polarised_gga_kernel_matches_finite_difference_vxc(h2o_setup, name):
    """Central-difference of the per-spin UKS GGA potential along
    (δD_α, δD_β) must match the analytic W^XC_σ from the C++ kernel.

    Pure GGA (PBE, BLYP) and hybrid GGA (B3LYP — exercises the LDA
    VWN5 component inside the polarised fxc accumulation)."""
    setup = h2o_setup
    func = Functional(name, spin=2)
    assert func.kind == vq.XCKind.GGA

    # Spin-split the SAD density so ρ_α ≠ ρ_β (αβ coupling is live).
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]

    rng = np.random.default_rng(0x17E + abs(hash(name)) % 1000)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)
    scale = np.linalg.norm(setup["D"])
    dDa *= scale / max(np.linalg.norm(dDa), 1e-12)
    dDb *= scale / max(np.linalg.norm(dDb), 1e-12)

    kernel = make_polarised_gga_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db,
    )
    Wa, Wb = kernel.apply(dDa, dDb)
    Wa, Wb = np.asarray(Wa), np.asarray(Wb)

    eps = 1e-5
    dchi = (setup["dchi_x"], setup["dchi_y"], setup["dchi_z"])
    Vap, Vbp = _python_vxc_polarised_gga(
        func, setup["grid"], setup["chi"], dchi,
        Da + eps * dDa, Db + eps * dDb)
    Vam, Vbm = _python_vxc_polarised_gga(
        func, setup["grid"], setup["chi"], dchi,
        Da - eps * dDa, Db - eps * dDb)
    Wa_fd = (Vap - Vam) / (2.0 * eps)
    Wb_fd = (Vbp - Vbm) / (2.0 * eps)

    err_a = np.linalg.norm(Wa - Wa_fd) / np.linalg.norm(Wa_fd)
    err_b = np.linalg.norm(Wb - Wb_fd) / np.linalg.norm(Wb_fd)
    assert err_a < 1e-5, f"{name}: W^XC_α rel err {err_a:.3e}"
    assert err_b < 1e-5, f"{name}: W^XC_β rel err {err_b:.3e}"


def test_polarised_gga_kernel_couples_the_two_spins(h2o_setup):
    """W^XC_α must genuinely depend on δD_β — that dependence is the
    αβ block of the fxc kernel. apply(δD_α, 0) ≠ apply(δD_α, δD_β)."""
    setup = h2o_setup
    func = Functional("PBE", spin=2)
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]
    rng = np.random.default_rng(0xC0FFEE)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)
    kernel = make_polarised_gga_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db,
    )
    Wa_only_a, _ = kernel.apply(dDa, np.zeros_like(dDb))
    Wa_both, _ = kernel.apply(dDa, dDb)
    assert not np.allclose(np.asarray(Wa_only_a), np.asarray(Wa_both))


def test_polarised_gga_kernel_is_linear_and_symmetric(h2o_setup):
    """W^XC is linear in (δD_α, δD_β) jointly, and each spin block is
    symmetric in μ↔ν."""
    setup = h2o_setup
    func = Functional("PBE", spin=2)
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]
    rng = np.random.default_rng(7)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)
    kernel = make_polarised_gga_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db,
    )
    Wa1, Wb1 = (np.asarray(x) for x in kernel.apply(dDa, dDb))
    Wa2, Wb2 = (np.asarray(x) for x in kernel.apply(2.0 * dDa, 2.0 * dDb))
    assert np.allclose(Wa2, 2.0 * Wa1, atol=1e-9)
    assert np.allclose(Wb2, 2.0 * Wb1, atol=1e-9)
    assert np.allclose(Wa1, Wa1.T, atol=1e-12)
    assert np.allclose(Wb1, Wb1.T, atol=1e-12)


def test_polarised_gga_kernel_factory_rejects_lda(h2o_setup):
    """make_polarised_gga_xc_kernel_builder on an LDA functional must
    raise — LDA has no gradient terms; the LDA builder is the right
    (and cheaper) call."""
    setup = h2o_setup
    func = Functional("LDA", spin=2)
    with pytest.raises(RuntimeError, match="LDA"):
        make_polarised_gga_xc_kernel_builder(
            func, setup["grid"], setup["chi"],
            setup["dchi_x"], setup["dchi_y"], setup["dchi_z"],
            0.5 * setup["D"], 0.5 * setup["D"],
        )


def test_polarised_xc_kernel_dispatcher(h2o_setup):
    """The unified factory dispatches by functional kind: LDA and GGA
    both build a working kernel; meta-GGA raises with a roadmap
    pointer. The LDA path must agree with the dedicated LDA builder."""
    setup = h2o_setup
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]
    rng = np.random.default_rng(11)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)

    # LDA → must match the dedicated LDA builder bit-for-bit.
    lda = Functional("LDA", spin=2)
    k_disp = make_polarised_xc_kernel_builder(
        lda, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db)
    k_lda = make_polarised_lda_xc_kernel_builder(
        lda, setup["grid"], setup["chi"], Da, Db)
    a_disp, b_disp = (np.asarray(x) for x in k_disp.apply(dDa, dDb))
    a_lda, b_lda = (np.asarray(x) for x in k_lda.apply(dDa, dDb))
    assert np.allclose(a_disp, a_lda) and np.allclose(b_disp, b_lda)

    # GGA → builds and produces a non-trivial matvec.
    gga = Functional("PBE", spin=2)
    k_gga = make_polarised_xc_kernel_builder(
        gga, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db)
    a_gga, _ = k_gga.apply(dDa, dDb)
    assert np.any(np.asarray(a_gga) != 0.0)

    # Meta-GGA → raises.
    mgga = Functional("tpssh", spin=2)
    with pytest.raises(RuntimeError, match="meta-GGA"):
        make_polarised_xc_kernel_builder(
            mgga, setup["grid"], setup["chi"],
            setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db)


# ---------------------------------------------------------------------------
# 5. Batched (memory-bounded) polarised builder — the default-on UKS
#    post-convergence stability analysis path. The dense builders cache
#    whole-grid (n_pts, n_bf) AO tables (~15 concurrent extents across
#    setup + apply); the batched builder streams
#    kMolecularXcGridBatchSize-point slices and must agree with the
#    dense reference to numerical noise (same per-point math, different
#    summation grouping).
# ---------------------------------------------------------------------------

def test_batched_polarised_kernel_matches_dense_gga(h2o_setup):
    """Batched GGA fxc matvec == dense PolarisedGGAXCKernelBuilder."""
    setup = h2o_setup
    # The fixture grid has ~3.5e4 points, so the 4096-point slicing is
    # genuinely exercised (multiple slices, ragged final slice).
    from vibeqc._vibeqc_core import MOLECULAR_XC_GRID_BATCH_SIZE
    assert np.asarray(setup["grid"].points).shape[0] \
        > MOLECULAR_XC_GRID_BATCH_SIZE
    func = Functional("PBE", spin=2)
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]
    rng = np.random.default_rng(0xBA7C4)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)

    dense = make_polarised_gga_xc_kernel_builder(
        func, setup["grid"], setup["chi"],
        setup["dchi_x"], setup["dchi_y"], setup["dchi_z"], Da, Db)
    batched = make_batched_polarised_xc_kernel_builder(
        func, setup["grid"], setup["basis"], Da, Db)

    Wa_d, Wb_d = (np.asarray(x) for x in dense.apply(dDa, dDb))
    Wa_b, Wb_b = (np.asarray(x) for x in batched.apply(dDa, dDb))
    err_a = np.linalg.norm(Wa_b - Wa_d) / max(np.linalg.norm(Wa_d), 1e-300)
    err_b = np.linalg.norm(Wb_b - Wb_d) / max(np.linalg.norm(Wb_d), 1e-300)
    assert err_a < 1e-10, f"W^XC_α batched vs dense rel err {err_a:.3e}"
    assert err_b < 1e-10, f"W^XC_β batched vs dense rel err {err_b:.3e}"
    assert np.allclose(Wa_b, Wa_b.T, atol=1e-12)
    assert np.allclose(Wb_b, Wb_b.T, atol=1e-12)


def test_batched_polarised_kernel_parallel_waves_are_bitwise_ordered(
    h2o_setup,
):
    """Parallel stability batches preserve the serial grid reduction."""
    setup = h2o_setup
    func = Functional("PBE", spin=2)
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]
    rng = np.random.default_rng(0x204)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)
    initial = vq.get_num_threads()
    try:
        vq.set_num_threads(4)
        serial = make_batched_polarised_xc_kernel_builder(
            func, setup["grid"], setup["basis"], Da, Db,
            max_batch_workers=1,
        )
        parallel = make_batched_polarised_xc_kernel_builder(
            func, setup["grid"], setup["basis"], Da, Db,
            max_batch_workers=4,
        )
        Wa_s, Wb_s = (np.asarray(x) for x in serial.apply(dDa, dDb))
        Wa_p, Wb_p = (np.asarray(x) for x in parallel.apply(dDa, dDb))
    finally:
        vq.set_num_threads(initial)

    assert serial.batch_workers_used == 1
    assert parallel.batch_workers_used == 4
    assert np.array_equal(Wa_p, Wa_s)
    assert np.array_equal(Wb_p, Wb_s)


def test_batched_polarised_kernel_matches_dense_lda(h2o_setup):
    """Batched LDA fxc matvec == dense PolarisedLDAXCKernelBuilder."""
    setup = h2o_setup
    func = Functional("LDA", spin=2)
    Da = 0.52 * setup["D"]
    Db = 0.48 * setup["D"]
    rng = np.random.default_rng(0xBA7C5)
    n = setup["basis"].nbasis
    dDa = _random_symmetric(n, rng)
    dDb = _random_symmetric(n, rng)

    dense = make_polarised_lda_xc_kernel_builder(
        func, setup["grid"], setup["chi"], Da, Db)
    batched = make_batched_polarised_xc_kernel_builder(
        func, setup["grid"], setup["basis"], Da, Db)

    Wa_d, Wb_d = (np.asarray(x) for x in dense.apply(dDa, dDb))
    Wa_b, Wb_b = (np.asarray(x) for x in batched.apply(dDa, dDb))
    err_a = np.linalg.norm(Wa_b - Wa_d) / max(np.linalg.norm(Wa_d), 1e-300)
    err_b = np.linalg.norm(Wb_b - Wb_d) / max(np.linalg.norm(Wb_d), 1e-300)
    assert err_a < 1e-10, f"W^XC_α batched vs dense rel err {err_a:.3e}"
    assert err_b < 1e-10, f"W^XC_β batched vs dense rel err {err_b:.3e}"


def test_batched_polarised_kernel_rejects_metagga(h2o_setup):
    """Meta-GGA polarised fxc is not plumbed — the batched factory must
    fail closed exactly like the dense dispatcher."""
    setup = h2o_setup
    mgga = Functional("tpssh", spin=2)
    with pytest.raises(RuntimeError, match="meta-GGA"):
        make_batched_polarised_xc_kernel_builder(
            mgga, setup["grid"], setup["basis"],
            0.5 * setup["D"], 0.5 * setup["D"])
