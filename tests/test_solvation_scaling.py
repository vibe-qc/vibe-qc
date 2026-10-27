"""Memory scaling of the solvated SCF and gradient.

A 74-atom alkane in STO-3G has 15 500 cavity segments. Before these changes a
solvated SCF peaked at 9.9 GB and the gradient needed more: an
``(n_segments, n_basis, n_basis)`` stack of Coulomb operators (3.6 GB there, and
3 GB for a 30-atom molecule in def2-SVP), a second copy of the cavity matrix for
its LU factorization, and ``(n, n, 3)`` pair tensors for the gradient. Each now
stays at the size of the cavity matrix or below, and gives the same numbers.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.solvation.cavity import build_cavity
from vibeqc.solvation.cavity_derivative import cavity_A_adjoints
from vibeqc.solvation.cpcm import build_cavity_A_matrix
from vibeqc.solvation.driver import GaussianESPProvider
from vibeqc.solvation.engine import lu_cavity_solve

ANG = 1.8897261254578281

_WATER = [(8, (0.0, 0.0, 0.1193)), (1, (0.0, 0.7632, -0.477)), (1, (0.0, -0.7632, -0.477))]
_METHANOL = [
    (6, (-0.0471, 0.6644, 0.0)),
    (8, (-0.0471, -0.7586, 0.0)),
    (1, (-1.093, 0.9698, 0.0)),
    (1, (0.8785, -1.0485, 0.0)),
    (1, (0.4371, 1.0804, 0.8918)),
    (1, (0.4371, 1.0804, -0.8918)),
]


def _molecule(atoms):
    return vq.Molecule([vq.Atom(z, tuple(np.array(p) * ANG)) for z, p in atoms], 0, 1)


def _cavity(atoms, **kw):
    pos = np.array([p for _, p in atoms]) * ANG
    return build_cavity(atom_positions_bohr=pos, atom_numbers=[z for z, _ in atoms], **kw)


# --- the cavity matrix factorization ----------------------------------------


@pytest.mark.parametrize("order", ["C", "F"])
def test_the_in_place_factorization_solves_the_same_system(order):
    cav = _cavity(_METHANOL)
    A = np.array(build_cavity_A_matrix(cav), order=order)
    rhs = np.random.default_rng(0).normal(size=A.shape[0])
    expected = np.linalg.solve(A, rhs)
    kept = lu_cavity_solve(A.copy(order=order))(rhs)
    in_place = lu_cavity_solve(A, overwrite=True)(rhs)
    np.testing.assert_allclose(kept, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(in_place, kept, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("order", ["C", "F"])
def test_the_in_place_factorization_really_is_in_place(order):
    """Either memory layout, because the native builder returns column-major
    arrays and the Python builders row-major ones."""
    cav = _cavity(_WATER)
    A = np.array(build_cavity_A_matrix(cav), order=order)
    before = A.copy()
    lu_cavity_solve(A, overwrite=True)
    assert not np.array_equal(A, before)  # the buffer now holds the factors
    B = before.copy(order=order)
    lu_cavity_solve(B)
    assert np.array_equal(B, before)  # and the default leaves the input alone


def test_the_native_cavity_matrix_layout_is_known():
    """If this changes, lu_cavity_solve's in-place branch needs the other layout."""
    A = build_cavity_A_matrix(_cavity(_WATER))
    assert A.flags.f_contiguous or A.flags.c_contiguous


# --- the gradient's pair tensors --------------------------------------------


@pytest.mark.parametrize("kernel", ["point", "gaussian"])
@pytest.mark.parametrize("block_rows", [1, 7, 100000])
def test_blocked_adjoints_equal_the_unblocked_ones(kernel, block_rows):
    cav = _cavity(_METHANOL, charge_representation=kernel)
    rng = np.random.default_rng(1)
    u = rng.normal(size=cav.n_points)
    v = rng.normal(size=cav.n_points)
    kw = dict(
        representation=kernel,
        switching=cav.switching if kernel == "gaussian" else None,
        n_points_per_sphere=302,
        exponent_follows_area=kernel != "gaussian",
    )
    ref_p, ref_w = cavity_A_adjoints(cav.points, cav.weights, u, v, 0.37, block_rows=cav.n_points, **kw)
    p, w = cavity_A_adjoints(cav.points, cav.weights, u, v, 0.37, block_rows=block_rows, **kw)
    np.testing.assert_allclose(p, ref_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w, ref_w, rtol=1e-12, atol=1e-12)


def test_the_area_following_gaussian_channel_is_blocked_too():
    """The fine cavity's branch: its exponent follows the area, F == 1."""
    cav = _cavity(_WATER)
    rng = np.random.default_rng(2)
    q = rng.normal(size=cav.n_points)
    kw = dict(
        representation="gaussian",
        switching=np.ones(cav.n_points),
        n_points_per_sphere=302,
        exponent_follows_area=True,
    )
    ref = cavity_A_adjoints(cav.points, cav.weights, q, q, 0.5, block_rows=cav.n_points, **kw)
    got = cavity_A_adjoints(cav.points, cav.weights, q, q, 0.5, block_rows=5, **kw)
    for a, b in zip(got, ref):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * max(1.0, float(np.abs(b).max())))


# --- the operator stack ------------------------------------------------------


def test_the_streamed_provider_equals_the_cached_one():
    mol = _molecule(_METHANOL)
    basis = vq.BasisSet(mol, "def2-svp")
    pts = _cavity(_METHANOL).points
    cached = GaussianESPProvider(basis, pts)
    streamed = GaussianESPProvider(basis, pts, max_cache_bytes=0)
    assert cached.operators is not None
    assert streamed.operators is None
    rng = np.random.default_rng(3)
    nb = basis.nbasis
    D = rng.normal(size=(nb, nb))
    D = D + D.T
    q = rng.normal(size=pts.shape[0])
    esp_c, esp_s = cached.esp_at_cavity(D), streamed.esp_at_cavity(D)
    np.testing.assert_allclose(esp_s, esp_c, rtol=1e-11, atol=1e-11 * np.abs(esp_c).max())
    f_c, f_s = cached.fock_contribution(q), streamed.fock_contribution(q)
    np.testing.assert_allclose(f_s, f_c, rtol=1e-11, atol=1e-11 * np.abs(f_c).max())


def test_the_provider_caches_only_what_fits_its_budget():
    mol = _molecule(_WATER)
    basis = vq.BasisSet(mol, "sto-3g")
    pts = _cavity(_WATER).points
    stack = 8 * pts.shape[0] * basis.nbasis**2
    assert GaussianESPProvider(basis, pts, max_cache_bytes=stack).operators is not None
    assert GaussianESPProvider(basis, pts, max_cache_bytes=stack - 1).operators is None


def test_a_streamed_scf_and_gradient_equal_the_cached_ones():
    mol = _molecule(_METHANOL)
    basis = vq.BasisSet(mol, "def2-svp")
    out = {}
    for mb in (1024.0, 0.0):
        sm = vq.SolventModel(
            epsilon=78.39, name="water", operator_cache_mb=mb, max_macro_iter=60, tol_e_solv=1e-11
        )
        sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
        assert sol.converged
        grad = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"))
        out[mb] = (sol.energy, sol.e_solv, grad)
    assert out[0.0][0] == pytest.approx(out[1024.0][0], abs=1e-10)
    assert out[0.0][1] == pytest.approx(out[1024.0][1], abs=1e-10)
    np.testing.assert_allclose(out[0.0][2], out[1024.0][2], atol=1e-9)


def test_streaming_does_not_hold_the_operator_stack():
    """Python-side allocation peak of a solvated SCF, from tracemalloc.

    Water in def2-TZVP: the stack is 12 MB and the cavity matrix 5.5 MB. Cached,
    the peak is the two together (18 MB); streamed it is the matrix alone
    (5.8 MB), well under the stack. tracemalloc sees numpy buffers, which is
    where the stack lives.
    """
    mol = _molecule(_WATER)
    basis = vq.BasisSet(mol, "def2-tzvp")
    peaks = {}
    stack = None
    for mb in (1024.0, 0.0):
        sm = vq.SolventModel(epsilon=78.39, name="water", operator_cache_mb=mb)
        tracemalloc.start()
        try:
            sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
            _, peaks[mb] = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        stack = 8 * sol.cavity.n_points * basis.nbasis**2
    assert peaks[1024.0] > stack  # the control: the cached run does hold it
    assert peaks[0.0] < 0.7 * stack


def test_a_negative_cache_budget_is_refused():
    with pytest.raises(ValueError, match="operator_cache_mb"):
        vq.SolventModel(epsilon=78.39, operator_cache_mb=-1.0)
    vq.SolventModel(epsilon=78.39, operator_cache_mb=0.0)
