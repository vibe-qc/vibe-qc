"""Density-fitting core: 2-centre + 3-centre integral kernels and the
DensityFitting object.

These tests pin matrix-level correctness for the DF infrastructure
without touching SCF. The reference is the existing direct 4-index
ERI tensor (vibeqc.compute_eri) — every assertion asks "does the RI
factorisation reproduce the direct integral, modulo the residual fit
error of the JKfit auxiliary basis?".

For the bundled def2-svp / def2-svp-jk pair on neutral organics, the
J fit error is well below 1 µHa per atom (Weigend, JCC 29, 167 (2008));
all tolerances below are sized accordingly with conservative slack.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    compute_2c_eri,
    compute_3c_eri,
    compute_eri,
)
from vibeqc.density_fitting import DensityFitting, default_aux_basis_for

from .conftest import GEOMETRIES


# -----------------------------------------------------------------------------
# Auxiliary-basis autodetection.
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "orbital,expected_jk",
    [
        ("def2-svp",   "def2-svp-jk"),
        ("def2-SVP",   "def2-svp-jk"),         # case-insensitive
        ("def2-sv(p)", "def2-svp-jk"),         # alias
        ("def2-svpd",  "def2-svp-jk"),         # diffuse alias
        ("def2-tzvp",  "def2-tzvp-jk"),
        ("def2-tzvpp", "def2-tzvpp-jk"),
        ("def2-qzvpp", "def2-qzvpp-jk"),
        ("cc-pvdz",    "cc-pvdz-jkfit"),
        ("cc-pvtz",    "cc-pvtz-jkfit"),
        ("cc-pv5z",    "cc-pv5z-jkfit"),
    ],
)
def test_default_aux_basis_jk(orbital, expected_jk):
    assert default_aux_basis_for(orbital, kind="jk") == expected_jk


@pytest.mark.parametrize(
    "orbital,expected_ri",
    [
        # Dunning correlation-consistent family.
        ("cc-pvdz",     "cc-pvdz-ri"),
        ("cc-pvtz",     "cc-pvtz-ri"),
        ("cc-pvqz",     "cc-pvqz-ri"),
        # def2 family — per-zeta + diffuse variants, all BSE-fetched.
        ("def2-svp",    "def2-svp-rifit"),
        ("def2-sv(p)",  "def2-sv(p)-rifit"),
        ("def2-svpd",   "def2-svpd-rifit"),
        ("def2-tzvp",   "def2-tzvp-rifit"),
        ("def2-tzvpd",  "def2-tzvpd-rifit"),
        ("def2-tzvpp",  "def2-tzvpp-rifit"),
        ("def2-tzvppd", "def2-tzvppd-rifit"),
        # def2-qzvp -> the QZVPP fit: BSE's def2-QZVP-RIFIT record is a
        # 42-element main-group subset of it (GitLab #483).
        ("def2-qzvp",   "def2-qzvpp-rifit"),
        ("def2-qzvpp",  "def2-qzvpp-rifit"),
        ("def2-qzvppd", "def2-qzvppd-rifit"),
    ],
)
def test_default_aux_basis_ri(orbital, expected_ri):
    assert default_aux_basis_for(orbital, kind="ri") == expected_ri


def test_bse_fetched_aux_bases_load_via_libint():
    """Sanity-check that every BSE-fetched aux basis we ship loads via
    libint at runtime. Catches a mismatch between the shipped .g94 and
    libint's parser, or a typo in the bundled filenames."""
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])])  # O atom
    bundled_aux = [
        "def2-universal-jkfit",
        "def2-universal-jfit",
        "def2-sv(p)-jkfit",
        "def2-sv(p)-rifit",
        "def2-svp-rifit",
        "def2-svpd-rifit",
        "def2-tzvp-rifit",
        "def2-tzvpd-rifit",
        "def2-tzvpp-rifit",
        "def2-tzvppd-rifit",
        "def2-qzvp-rifit",
        "def2-qzvpp-rifit",
        "def2-qzvppd-rifit",
    ]
    for name in bundled_aux:
        b = BasisSet(mol, name)
        assert b.nbasis > 0, f"{name}: BasisSet built but reports 0 basis functions"


def test_default_aux_basis_pob_raises():
    """pob-* orbital bases lack a published JK aux. Must raise clearly,
    not silently substitute a def2 family — see project memory entry
    "pob-TZVP aux basis"."""
    with pytest.raises(NotImplementedError, match="pob-"):
        default_aux_basis_for("pob-TZVP", kind="jk")
    with pytest.raises(NotImplementedError, match="pob-"):
        default_aux_basis_for("pob-tzvp-rev2", kind="jk")


def test_default_aux_basis_unknown_orbital():
    with pytest.raises(NotImplementedError, match="No default"):
        default_aux_basis_for("madeup-zeta-99", kind="jk")


def test_default_aux_basis_invalid_kind():
    with pytest.raises(ValueError, match="kind"):
        default_aux_basis_for("def2-svp", kind="foo")


# -----------------------------------------------------------------------------
# Helpers for the integral-level tests.
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2o_pair():
    """H2O / def2-svp orbital + def2-svp-jk auxiliary."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["H2O"]])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    return mol, basis, aux


@pytest.fixture(scope="module")
def ch4_pair():
    """CH4 / def2-svp orbital + def2-svp-jk auxiliary — extra coverage of
    the parallelised P-shell loop and pure-CCH topology."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["CH4"]])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    return mol, basis, aux


def _random_symmetric(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


# -----------------------------------------------------------------------------
# 2-centre kernel (P|Q): symmetry, SPD, agreement against known-good
# slow reference.
# -----------------------------------------------------------------------------

def test_2c_eri_symmetric(h2o_pair):
    _, _, aux = h2o_pair
    V = compute_2c_eri(aux)
    np.testing.assert_allclose(V, V.T, atol=1e-12)


def test_2c_eri_positive_definite(h2o_pair):
    """V_PQ = (P|Q) is the Coulomb Gram matrix of a linearly-independent
    auxiliary basis; smallest eigenvalue must be strictly positive.
    Cholesky factorisation depends on this."""
    _, _, aux = h2o_pair
    V = compute_2c_eri(aux)
    eigs = np.linalg.eigvalsh(V)
    assert eigs[0] > 0.0, f"V is not PD (min eig = {eigs[0]:.3e})"
    # def2-svp-jk on H2O is well-conditioned (cond ~ 1e3 in practice).
    cond = eigs[-1] / eigs[0]
    assert cond < 1e8, f"V condition number {cond:.2e} unexpectedly large"


def test_2c_eri_shape(h2o_pair):
    _, _, aux = h2o_pair
    V = compute_2c_eri(aux)
    assert V.shape == (aux.nbasis, aux.nbasis)


# -----------------------------------------------------------------------------
# 3-centre kernel (P|μν): shape, μν-symmetry.
# -----------------------------------------------------------------------------

def test_3c_eri_shape(h2o_pair):
    _, basis, aux = h2o_pair
    T = compute_3c_eri(basis, aux)
    assert T.shape == (aux.nbasis, basis.nbasis, basis.nbasis)


def test_3c_eri_symmetric_in_orbital_pair(h2o_pair):
    """(P|μν) = (P|νμ): the orbital pair density χ_μ χ_ν is symmetric in
    (μ, ν), so the integral inherits the symmetry. The kernel fills both
    off-diagonal positions explicitly; check that they actually agree."""
    _, basis, aux = h2o_pair
    T = compute_3c_eri(basis, aux)
    # Compare T[P, μ, ν] to T[P, ν, μ] over all (P, μ, ν).
    np.testing.assert_allclose(T, T.transpose(0, 2, 1), atol=1e-12)


# -----------------------------------------------------------------------------
# DF factorisation accuracy: J via RI ≈ J via direct 4-index ERI.
#
# These are the *physical* tests: do RI integrals reproduce the direct
# integrals at the matrix level, and is the residual within published
# fit-error bounds for the JKfit family?
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["h2o_pair", "ch4_pair"])
def test_J_matches_direct(fixture_name, request):
    """J^{RI}_μν vs J^{direct}_μν on a random symmetric density matrix.
    The Frobenius difference is the J-build fit error; def2-svp/def2-svp-jk
    on neutral organics gives < 1e-5 in absolute Frobenius units."""
    mol, basis, aux = request.getfixturevalue(fixture_name)
    n = basis.nbasis

    # Reference J via the dense 4-index path.
    eri = compute_eri(basis)
    D = _random_symmetric(n, seed=42)
    J_direct = np.einsum("mnls,ls->mn", eri, D, optimize=True)

    # RI J.
    df = DensityFitting(basis, aux, aux_basis_name="def2-svp-jk")
    J_ri = df.build_J(D)

    err = np.linalg.norm(J_ri - J_direct, ord="fro")
    rel = err / np.linalg.norm(J_direct, ord="fro")
    # def2-svp-jk fit accuracy: well below 1e-3 relative on neutral
    # organic molecules. Loosen by 10× for slack.
    assert rel < 1e-2, (
        f"{fixture_name}: relative J fit error {rel:.3e} > 1e-2 "
        f"(absolute Frobenius {err:.3e})"
    )
    # And the symmetry property: J^{RI} must itself be symmetric.
    np.testing.assert_allclose(J_ri, J_ri.T, atol=1e-10)


@pytest.mark.parametrize("fixture_name", ["h2o_pair", "ch4_pair"])
def test_K_matches_direct(fixture_name, request):
    """K^{RI}_μν vs K^{direct}_μν via build_K(C_occ).

    K^{direct}_μν = Σ_λσ D_λσ (μλ|νσ) for D = C_occ C_occ^T.
    Equivalently: K_μν = Σ_i (μi|νi).
    """
    mol, basis, aux = request.getfixturevalue(fixture_name)
    n = basis.nbasis

    # Build a "fake" set of occupied MOs from the n_occ smallest-eigenvalue
    # eigenvectors of a random symmetric matrix — gives orthonormal C_occ
    # without solving any SCF.
    n_occ = 4 if fixture_name == "h2o_pair" else 5  # H2O: 5e pairs, CH4: 5
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, n))
    A = 0.5 * (A + A.T)
    _, vecs = np.linalg.eigh(A)
    C_occ = vecs[:, :n_occ]
    D = C_occ @ C_occ.T

    eri = compute_eri(basis)
    # Convention: K_μν = Σ_λσ D_λσ (μλ|νσ).
    K_direct = np.einsum("mlns,ls->mn", eri, D, optimize=True)

    df = DensityFitting(basis, aux, aux_basis_name="def2-svp-jk")
    K_ri = df.build_K(C_occ)

    err = np.linalg.norm(K_ri - K_direct, ord="fro")
    rel = err / np.linalg.norm(K_direct, ord="fro")
    assert rel < 1e-2, (
        f"{fixture_name}: relative K fit error {rel:.3e} > 1e-2 "
        f"(absolute Frobenius {err:.3e})"
    )
    np.testing.assert_allclose(K_ri, K_ri.T, atol=1e-10)


def test_K_density_matches_K_from_C_occ(h2o_pair):
    """build_K_density(D = C C^T) must agree with build_K(C) up to
    floating-point: both express the same exchange operator."""
    _, basis, aux = h2o_pair
    n = basis.nbasis
    rng = np.random.default_rng(13)
    A = rng.standard_normal((n, n))
    A = 0.5 * (A + A.T)
    _, vecs = np.linalg.eigh(A)
    C_occ = vecs[:, :4]
    D = C_occ @ C_occ.T

    df = DensityFitting(basis, aux, aux_basis_name="def2-svp-jk")
    K_via_C = df.build_K(C_occ)
    K_via_D = df.build_K_density(D)
    np.testing.assert_allclose(K_via_C, K_via_D, atol=1e-10)


def test_B_tensor_factorises_metric(h2o_pair):
    """The half-transformed B-tensor satisfies V^{-1} = (L^{-T})(L^{-1}),
    so contracting B^P_μν against itself over P (and against an unrelated
    second pair) gives the RI approximation to (μν|λσ).

    Numerical check: pick a single (μν|λσ) pair, compute the RI value
    via Σ_P B^P_μν B^P_λσ, and compare to compute_eri's direct value
    within the def2-svp-jk fit accuracy.
    """
    _, basis, aux = h2o_pair
    df = DensityFitting(basis, aux, aux_basis_name="def2-svp-jk")
    eri = compute_eri(basis)
    n = basis.nbasis
    # Spot-check the (0,0|0,0) and (1,2|3,4) elements.
    for mu, nu, la, si in [(0, 0, 0, 0), (1, 2, 3, 4), (5, 5, 6, 6)]:
        if any(idx >= n for idx in (mu, nu, la, si)):
            continue
        ri_value = float(np.einsum("P,P->", df.B[:, mu, nu], df.B[:, la, si]))
        direct = float(eri[mu, nu, la, si])
        # Absolute tolerance: 1e-3 in Hartree-units of one ERI element is
        # well within JKfit error.
        assert abs(ri_value - direct) < 1e-3, (
            f"({mu}{nu}|{la}{si}): RI = {ri_value:.6e}, direct = {direct:.6e}"
        )


def test_metric_not_pd_diagnoses_clearly():
    """If V is somehow not PD (collapsed centres, duplicate basis, etc.),
    the constructor must raise with a diagnostic message — not silently
    produce garbage J/K matrices."""
    # Two H atoms exactly on top of each other => duplicated AOs => duplicate
    # aux centres => V has a zero eigenvalue.
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 0.0])])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    with pytest.raises(np.linalg.LinAlgError, match="not positive-definite"):
        DensityFitting(basis, aux, aux_basis_name="def2-svp-jk")


def test_mo_transform_reproduces_eri_in_mo_basis(h2o_pair):
    """Σ_P B^MO_pq B^MO_rs ≈ (pq|rs) in MO basis. Using random orthonormal
    MOs since the test cares about the math, not physical orbitals."""
    _, basis, aux = h2o_pair
    n = basis.nbasis
    rng = np.random.default_rng(21)
    A = rng.standard_normal((n, n))
    A = 0.5 * (A + A.T)
    _, C = np.linalg.eigh(A)  # orthonormal column basis

    df = DensityFitting(basis, aux, aux_basis_name="def2-svp-jk")
    eri = compute_eri(basis)
    # Direct MO-basis ERI.
    eri_mo = np.einsum("mp,nq,ls,rt,mnlr->pqst",
                       C, C, C, C, eri, optimize=True)

    # RI MO-basis B-tensor.
    B_mo = df.mo_transform(C, C)  # (n_aux, n, n)
    eri_mo_ri = np.einsum("Ppq,Pst->pqst", B_mo, B_mo, optimize=True)

    err = np.linalg.norm((eri_mo - eri_mo_ri).ravel())
    rel = err / np.linalg.norm(eri_mo.ravel())
    assert rel < 1e-2, f"MO-basis ERI relative fit error {rel:.3e} > 1e-2"
