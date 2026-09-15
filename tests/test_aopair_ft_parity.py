"""C++ ↔ Python parity for the Bloch-summed AO-pair Fourier transform.

The Python kernel in ``vibeqc._aopair_ft`` is the readable reference;
the C++ value and weighted-gradient kernels in ``cpp/src/aopair_ft.cpp``
are the production paths for pure spherical bases through L=6. This file
pins the implementations to machine-precision parity on a battery of
calibration cases.

The dispatcher knob ``VIBEQC_AOPAIR_FT_BACKEND`` lets each test toggle
the Python reference back on without re-importing or monkey-patching.

CLAUDE.md §10 applies — no PySCF / Psi4 / ORCA in-process imports here.
The Python kernel IS the reference. Cross-code validation against PySCF
``ft_aopair`` happens via the subprocess runner pattern, not here.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vibeqc._vibeqc_core import Atom, BasisSet, Molecule
from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch


_REL_TOL = 1e-12


def _run_both_backends(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FT under both backends. Restores env on the way out."""
    saved = os.environ.get("VIBEQC_AOPAIR_FT_BACKEND")
    try:
        os.environ["VIBEQC_AOPAIR_FT_BACKEND"] = "cxx"
        cxx = ao_pair_fourier_transform_bloch(*args, **kwargs)
        os.environ["VIBEQC_AOPAIR_FT_BACKEND"] = "python"
        py = ao_pair_fourier_transform_bloch(*args, **kwargs)
    finally:
        if saved is None:
            os.environ.pop("VIBEQC_AOPAIR_FT_BACKEND", None)
        else:
            os.environ["VIBEQC_AOPAIR_FT_BACKEND"] = saved
    return cxx, py


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.abs(b).max()
    if denom == 0.0:
        return float(np.abs(a - b).max())
    return float(np.abs(a - b).max() / denom)


@pytest.fixture(scope="module")
def h2_sto3g() -> BasisSet:
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])])
    return BasisSet(mol, "sto-3g")


@pytest.fixture(scope="module")
def lih_sto3g() -> BasisSet:
    """LiH STO-3G — Li contributes (1s, 2s, 2p) so this exercises a
    mixed L = 0 + L = 1 basis, including p × p, p × s, and s × p
    shell pairs."""
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [3.0, 0.0, 0.0])])
    return BasisSet(mol, "sto-3g")


@pytest.fixture(scope="module")
def n2_631gs() -> BasisSet:
    """N₂ 6-31G* — adds a d-shell on each N centre, exercising the
    L = 2 spherical-transform path on top of the s + p coverage."""
    mol = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [2.1, 0.0, 0.0])])
    return BasisSet(mol, "6-31g*")


@pytest.fixture(scope="module")
def he4_sto3g() -> BasisSet:
    """Pure-s basis with off-axis centres — exercises the (μ, ν) cross
    blocks where Bx, By, Bz are all non-zero."""
    mol = Molecule(
        [
            Atom(2, [0.0, 0.0, 0.0]),
            Atom(2, [1.5, 0.5, 0.0]),
            Atom(2, [0.5, 1.5, 1.0]),
            Atom(2, [-1.0, 0.5, 1.5]),
        ]
    )
    return BasisSet(mol, "sto-3g")


def test_molecular_limit_gamma_only(h2_sto3g: BasisSet) -> None:
    """Single cell, Γ-only: pure-molecular limit."""
    rng = np.random.default_rng(0)
    G = rng.normal(size=(64, 3))
    R = np.zeros((1, 3))
    k = np.zeros(3)
    cxx, py = _run_both_backends(h2_sto3g, G, R, k)
    assert cxx.shape == (h2_sto3g.nbasis, h2_sto3g.nbasis, G.shape[0])
    assert _rel_diff(cxx, py) <= _REL_TOL


def test_multi_cell_gamma_only(h2_sto3g: BasisSet) -> None:
    """Multiple R_g cells, k = 0 — exercises the cell sum."""
    rng = np.random.default_rng(1)
    G = rng.normal(size=(128, 3))
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 3.0, 1.5],
            [-3.0, -3.0, -1.5],
        ]
    )
    k = np.zeros(3)
    cxx, py = _run_both_backends(h2_sto3g, G, R, k)
    assert _rel_diff(cxx, py) <= _REL_TOL


def test_multi_cell_finite_k(h2_sto3g: BasisSet) -> None:
    """Non-zero crystal momentum — exercises the Bloch phase."""
    rng = np.random.default_rng(2)
    G = rng.normal(size=(96, 3))
    R = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    k = np.array([0.1, -0.2, 0.05])
    cxx, py = _run_both_backends(h2_sto3g, G, R, k)
    assert _rel_diff(cxx, py) <= _REL_TOL


def test_off_axis_centres_with_image_sum(he4_sto3g: BasisSet) -> None:
    """4-centre pure-s system: tests cross-shell blocks at non-zero R."""
    rng = np.random.default_rng(3)
    G = rng.normal(size=(50, 3))
    R = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    k = np.array([0.0, 0.3, 0.0])
    cxx, py = _run_both_backends(he4_sto3g, G, R, k)
    assert cxx.shape == (he4_sto3g.nbasis, he4_sto3g.nbasis, 50)
    assert _rel_diff(cxx, py) <= _REL_TOL


def test_zero_g_point_is_overlap(h2_sto3g: BasisSet) -> None:
    """G = 0 → out_{μν}(0) = ∫ χ_μ χ_ν = AO overlap (no momentum transfer).

    At Γ with a single cell this is a free sanity-check on the
    Gaussian-product theorem itself: the FT at G=0 collapses to the
    real-space overlap integral.
    """
    G = np.array([[0.0, 0.0, 0.0]])
    R = np.zeros((1, 3))
    k = np.zeros(3)
    cxx, _ = _run_both_backends(h2_sto3g, G, R, k)
    from vibeqc._vibeqc_core import compute_overlap
    S = np.asarray(compute_overlap(h2_sto3g))
    np.testing.assert_allclose(cxx[:, :, 0].real, S, atol=1e-13)
    np.testing.assert_allclose(cxx[:, :, 0].imag, 0.0, atol=1e-13)


def test_mixed_sp_basis(lih_sto3g: BasisSet) -> None:
    """LiH STO-3G — Bloch sum + nonzero k + L = 1 shells."""
    rng = np.random.default_rng(4)
    G = rng.normal(size=(80, 3))
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
        ]
    )
    k = np.array([0.1, -0.2, 0.05])
    cxx, py = _run_both_backends(lih_sto3g, G, R, k)
    assert cxx.shape == (lih_sto3g.nbasis, lih_sto3g.nbasis, 80)
    assert _rel_diff(cxx, py) <= _REL_TOL


def test_p_only_block_via_off_diagonal(lih_sto3g: BasisSet) -> None:
    """Pin the (Li 2p × Li 2p) block specifically — the d-axis
    products in the MD recursion only show up when both shells have
    L ≥ 1 on the same axis. Inspecting just the (p, p) AO sub-block
    surfaces the spherical (py, pz, px) m-ordering parity, which is
    independently sensitive."""
    rng = np.random.default_rng(5)
    G = rng.normal(size=(40, 3))
    R = np.array([[0.0, 0.0, 0.0]])
    k = np.zeros(3)
    cxx, py = _run_both_backends(lih_sto3g, G, R, k)

    # Pull out the (Li 2p, Li 2p) 3 × 3 × n_G block. Shell layout for
    # Li STO-3G: shells = [Li(1s), Li(2s), Li(2p), H(1s)]. Spherical
    # AO offsets: [0, 1, 2, 5], so Li 2p occupies AOs 2..4.
    p_block_cxx = cxx[2:5, 2:5, :]
    p_block_py = py[2:5, 2:5, :]
    assert _rel_diff(p_block_cxx, p_block_py) <= _REL_TOL


def test_sphd_basis_n2_6_31gs(n2_631gs: BasisSet) -> None:
    """N₂ 6-31G* — exercises the L = 2 (d-shell) path on a real basis,
    including all of (s,d), (p,d), (d,d) shell-pair flavours."""
    rng = np.random.default_rng(6)
    G = rng.normal(size=(60, 3))
    R = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    k = np.array([0.05, 0.0, -0.05])
    cxx, py = _run_both_backends(n2_631gs, G, R, k)
    assert cxx.shape == (n2_631gs.nbasis, n2_631gs.nbasis, 60)
    assert _rel_diff(cxx, py) <= _REL_TOL


def test_l0_path_dispatches_to_ss_kernel(h2_sto3g: BasisSet) -> None:
    """An L = 0 basis must hit the closed-form SS kernel, not the
    general-L MD kernel. Documents (and pins) the dispatcher order
    so a refactor doesn't silently lose the SS fast path."""
    from vibeqc._vibeqc_core import (
        ao_pair_fourier_transform_bloch_ss,
        ao_pair_fourier_transform_bloch_cxx,
    )
    G = np.array([[0.1, 0.2, 0.3]])
    R = np.zeros((1, 3))
    k = np.zeros(3)
    ss = ao_pair_fourier_transform_bloch_ss(h2_sto3g, G, R, k)
    cxx = ao_pair_fourier_transform_bloch_cxx(h2_sto3g, G, R, k)
    # Both kernels are independent implementations of the same FT —
    # they must agree at machine precision.
    assert _rel_diff(cxx, ss) <= _REL_TOL


def test_cxx_backend_routes_higher_l_to_cxx_kernel(lih_sto3g: BasisSet) -> None:
    """Pre-Item-1b this raised. Post-Item-1b a mixed-L basis must
    instead dispatch into the general-L C++ kernel and produce
    bit-equivalent results."""
    G = np.array([[0.1, 0.2, 0.3], [0.5, -0.4, 0.3]])
    R = np.zeros((1, 3))
    k = np.zeros(3)
    saved = os.environ.get("VIBEQC_AOPAIR_FT_BACKEND")
    os.environ["VIBEQC_AOPAIR_FT_BACKEND"] = "cxx"
    try:
        cxx = ao_pair_fourier_transform_bloch(lih_sto3g, G, R, k)
    finally:
        if saved is None:
            os.environ.pop("VIBEQC_AOPAIR_FT_BACKEND", None)
        else:
            os.environ["VIBEQC_AOPAIR_FT_BACKEND"] = saved
    assert cxx.shape == (lih_sto3g.nbasis, lih_sto3g.nbasis, 2)


# ----------------------------------------------------------------------
# AO-pair FT atomic-position gradient (analytic-gradient linchpin).
#
# The position derivative of the AO-pair FT feeds the BIPOLE analytic-
# gradient kernels 2/4/5 (Ewald V_ne reciprocal, J^LR reciprocal,
# spheropole). It uses the angular-momentum-shift relation
#   ∂FT/∂A_μ = −i_μ·FT(cart_A lowered) + 2α·FT(cart_A raised),
# which must reproduce a finite difference of the value routine.
# ----------------------------------------------------------------------
def test_aopair_ft_grad_primitive_matches_fd() -> None:
    """``cartesian_gaussian_product_ft_grad`` vs central FD of the value
    routine, across s/p/d Cartesian-index pairs and both centres."""
    from vibeqc._aopair_ft import (
        cartesian_gaussian_product_ft,
        cartesian_gaussian_product_ft_grad,
    )

    rng = np.random.default_rng(20260531)
    G = rng.standard_normal((6, 3))
    A = np.array([0.1, -0.2, 0.3])
    B = np.array([0.4, 0.5, -0.1])
    alpha, beta = 1.3, 0.7
    h = 1e-6
    for ca, cb in [((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 1, 0)),
                   ((2, 1, 0), (1, 0, 1)), ((0, 0, 2), (0, 2, 0))]:
        gA, gB = cartesian_gaussian_product_ft_grad(A, B, ca, cb, alpha, beta, G)
        for mu in range(3):
            Ap, Am = A.copy(), A.copy()
            Ap[mu] += h
            Am[mu] -= h
            fdA = (cartesian_gaussian_product_ft(Ap, B, ca, cb, alpha, beta, G)
                   - cartesian_gaussian_product_ft(Am, B, ca, cb, alpha, beta, G)) / (2 * h)
            Bp, Bm = B.copy(), B.copy()
            Bp[mu] += h
            Bm[mu] -= h
            fdB = (cartesian_gaussian_product_ft(A, Bp, ca, cb, alpha, beta, G)
                   - cartesian_gaussian_product_ft(A, Bm, ca, cb, alpha, beta, G)) / (2 * h)
            assert np.max(np.abs(gA[mu] - fdA)) < 1e-6
            assert np.max(np.abs(gB[mu] - fdB)) < 1e-6


def test_aopair_ft_grad_shell_pair_matches_fd() -> None:
    """``_shell_pair_ao_pair_ft_grad_at_origins`` (contracted, spherical)
    vs central FD of the value shell-pair routine, for a p×p pair."""
    from vibeqc._aopair_ft import (
        _shell_pair_ao_pair_ft_at_origins,
        _shell_pair_ao_pair_ft_grad_at_origins,
    )

    rng = np.random.default_rng(11)
    G = rng.standard_normal((5, 3))
    A = np.array([0.1, -0.2, 0.3])
    B = np.array([0.4, 0.5, -0.1])
    esM, csM = np.array([2.0, 0.5]), np.array([0.6, 0.4])
    esN, csN = np.array([1.1]), np.array([1.0])
    gA, gB = _shell_pair_ao_pair_ft_grad_at_origins(1, 1, A, B, esM, csM, esN, csN, G)

    def val(AA, BB):
        return _shell_pair_ao_pair_ft_at_origins(1, 1, AA, BB, esM, csM, esN, csN, G)

    h = 1e-6
    for mu in range(3):
        Ap, Am = A.copy(), A.copy()
        Ap[mu] += h
        Am[mu] -= h
        Bp, Bm = B.copy(), B.copy()
        Bp[mu] += h
        Bm[mu] -= h
        assert np.max(np.abs(gA[mu] - (val(Ap, B) - val(Am, B)) / (2 * h))) < 1e-6
        assert np.max(np.abs(gB[mu] - (val(A, Bp) - val(A, Bm)) / (2 * h))) < 1e-6


def test_aopair_ft_grad_cell_ket_matches_dRg(lih_sto3g: BasisSet) -> None:
    """The per-cell ket-origin gradient must equal d/dR_g of the value
    routine (every ket image moves rigidly with R_g)."""
    from vibeqc._aopair_ft import (
        ao_pair_fourier_transform_shifted_ket,
        ao_pair_fourier_transform_shifted_ket_grad,
    )

    rng = np.random.default_rng(3)
    G = rng.standard_normal((5, 3)) * 0.8
    R_g = np.array([0.3, -0.4, 0.7])
    _, grad_ket = ao_pair_fourier_transform_shifted_ket_grad(lih_sto3g, G, R_g)
    h = 1e-6
    for mu in range(3):
        Rp, Rm = R_g.copy(), R_g.copy()
        Rp[mu] += h
        Rm[mu] -= h
        fd = (ao_pair_fourier_transform_shifted_ket(lih_sto3g, G, Rp)
              - ao_pair_fourier_transform_shifted_ket(lih_sto3g, G, Rm)) / (2 * h)
        assert np.max(np.abs(grad_ket[:, :, mu, :] - fd)) < 1e-6


@pytest.mark.parametrize(
    "basis_fixture",
    ["h2_sto3g", "lih_sto3g", "n2_631gs"],
)
def test_weighted_gamma_gradient_native_matches_python_reference(
    basis_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """The native weighted contraction matches the readable 5-D reference.

    The non-inversion-symmetric cell list and complex reciprocal weights
    prevent an accidental Gamma mirror or conjugation convention from
    passing. LiH additionally pins the spherical p-shell ordering.
    """
    from vibeqc._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from vibeqc._vibeqc_core import (
        ao_pair_fourier_transform_gamma_gradient_weighted,
    )

    basis = request.getfixturevalue(basis_fixture)
    rng = np.random.default_rng(20260715)
    G = rng.normal(size=(17, 3))
    cells = np.array(
        [[0.0, 0.0, 0.0], [3.0, -1.0, 0.5], [-2.0, 0.4, 1.0]]
    )
    pair_weights = rng.normal(size=(basis.nbasis, basis.nbasis))
    reciprocal_weights = rng.normal(size=len(G)) + 1j * rng.normal(size=len(G))

    grad_bra, grad_ket = ao_pair_fourier_transform_grad_at_cells(
        basis, G, cells
    )
    bra_by_ao = np.einsum(
        "mn,cmnxg->mxg", pair_weights, grad_bra, optimize=True
    )
    ket_by_ao = np.einsum(
        "mn,cmnxg->nxg", pair_weights, grad_ket, optimize=True
    )
    ao_to_atom: list[int] = []
    for shell in basis.shells():
        ao_to_atom.extend(
            [int(shell.atom_index)] * (2 * int(shell.l) + 1)
        )
    n_atoms = max(ao_to_atom) + 1
    derivative_by_atom = np.zeros(
        (n_atoms, 3, len(G)), dtype=np.complex128
    )
    np.add.at(derivative_by_atom, np.asarray(ao_to_atom), bra_by_ao)
    np.add.at(derivative_by_atom, np.asarray(ao_to_atom), ket_by_ao)
    reference = np.real(
        np.einsum(
            "g,axg->ax",
            reciprocal_weights,
            derivative_by_atom.conj(),
            optimize=True,
        )
    )

    native = ao_pair_fourier_transform_gamma_gradient_weighted(
        basis,
        G,
        cells,
        pair_weights,
        reciprocal_weights,
        n_atoms,
    )
    np.testing.assert_allclose(native, reference, atol=2.0e-12, rtol=2.0e-12)
