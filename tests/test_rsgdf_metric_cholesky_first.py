"""SR/LR (rsgdf) fit metric: Cholesky first, eigenvalue cut as fallback.

vibe-qc#179, maintainer decision 2026-10-27 (option 2 of issue note 40366).
The SR/LR fit keeps every auxiliary function when the Coulomb metric of a
momentum transfer Cholesky-factors and is not near singular, as PySCF's
GDF/RSGDF builders do (``pyscf/pbc/df/rsdf_builder.py``,
``cholesky_decomposed_metric``; Sun et al., J. Chem. Phys. 147, 164119
(2017); Ye & Berkelbach, J. Chem. Phys. 154, 131104 (2021)). Otherwise it
falls back to the absolute ``linear_dep_thr`` cut. compcell and MDF keep
their relative floors.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.aux_basis import (
    _build_lpq_range_separated_shared_q,
    _rsgdf_metric_keep,
    _rsgdf_metric_rounding_floor,
    _rsgdf_select_metric_decomposition,
)


def _hermitian(eigenvalues, seed=7):
    rng = np.random.default_rng(seed)
    n = len(eigenvalues)
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    unitary, _ = np.linalg.qr(raw)
    w = np.asarray(eigenvalues, dtype=float)
    return (unitary * w) @ unitary.conj().T, np.sort(w)


def test_selection_keeps_full_rank_for_a_positive_definite_metric():
    # Smallest eigenvalue below the 1e-9 cut, far above the rounding floor:
    # the LiH/STO-3G def2-svp-jk pattern (lambda_min 3.8e-10, max 21.2).
    metric, w = _hermitian([3.8e-10, 2e-9, 0.3, 4.0, 21.2])
    assert w[0] > _rsgdf_metric_rounding_floor(w, len(w))
    assert _rsgdf_select_metric_decomposition(metric, w, "cholesky") == "cholesky"
    np.testing.assert_array_equal(_rsgdf_metric_keep(w, 1e-9, "cholesky"), [True] * 5)
    # The opt-out and the fallback both apply the old absolute cut.
    assert _rsgdf_select_metric_decomposition(metric, w, "eigen") == "eigen"
    np.testing.assert_array_equal(
        _rsgdf_metric_keep(w, 1e-9, "eigen"), [False, True, True, True, True])


def test_selection_falls_back_for_an_indefinite_metric():
    # The MgO/STO-3G pattern: a rounding-level negative mode.
    metric, w = _hermitian([-1.3e-13, 5e-12, 0.2, 26.5])
    assert _rsgdf_select_metric_decomposition(metric, w, "cholesky") == "eigen"


def test_selection_guards_a_positive_definite_but_near_singular_metric():
    # potrf factors this matrix, but lambda_min sits below the eigensolver
    # rounding floor 64 eps n max|lambda|, so its 1/sqrt(lambda) is noise.
    w = np.array([1e-14, 0.5, 2.0, 30.0])
    metric = np.diag(w).astype(complex)
    assert w[0] < _rsgdf_metric_rounding_floor(w, len(w))
    from scipy.linalg import cholesky
    cholesky(metric, lower=True)  # positive definite to LAPACK
    assert _rsgdf_select_metric_decomposition(metric, w, "cholesky") == "eigen"


def test_unknown_decomposition_is_refused():
    w = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="cholesky"):
        _rsgdf_select_metric_decomposition(np.diag(w).astype(complex), w, "svd")
    with pytest.raises(ValueError, match="decomposition"):
        _rsgdf_metric_keep(w, 1e-9, "svd")


def _fit_fixture():
    from vibeqc import _vibeqc_core as core

    centers = [[.2, .3, .4], [1.6, .3, .4]]
    system = vq.PeriodicSystem(3, np.eye(3) * 5, [vq.Atom(1, r) for r in centers])
    mol = system.unit_cell_molecule()

    def make(exponents):
        return core.BasisSet(mol, [
            core.ShellInfo(i, 0, True, [a], [1.0], r)
            for i, r in enumerate(centers) for a in exponents
        ], "rsgdf-cholesky-test", coefficients_pre_normalized=False)
    # Two near-duplicate auxiliary exponents give a positive definite metric
    # with a small but well resolved eigenvalue.
    return system, make([.7]), make([.4, .41, 1.2, 3.0])


_OPTIONS = dict(
    omega=.5, pair_cutoff=12.0, auxiliary_cutoff=18.0, ke_cutoff=20.0,
    memory_byte_cap=8 * 2**20, native_workspace_byte_cap=2**20,
    image_candidate_cap=1000000, reciprocal_candidate_cap=1000000,
)


@pytest.mark.parametrize("q", [np.zeros(3), np.array([.13, -.07, .09])])
def test_leaf_cholesky_first_reproduces_the_full_metric_fit(q):
    system, basis, auxiliary = _fit_fixture()
    bras = np.array([[.03, -.02, .01], [.11, .02, -.04]]) if q.any() else np.zeros((1, 3))
    probe = _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, bras, q, linear_dep_thr=0.0,
        metric_decomposition="eigen", retain_fit_state=True, **_OPTIONS)
    w = probe.metric_eigenvalues
    assert w[0] > _rsgdf_metric_rounding_floor(w, len(w))
    # A cut between the two smallest modes: the old policy drops one mode.
    threshold = float(np.sqrt(w[0] * w[1]))
    assert w[0] < threshold < w[1]

    cut = _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, bras, q, linear_dep_thr=threshold,
        metric_decomposition="eigen", retain_fit_state=True, **_OPTIONS)
    full = _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, bras, q, linear_dep_thr=threshold,
        retain_fit_state=True, **_OPTIONS)
    n_aux = auxiliary.nbasis
    assert cut.factors.shape[1] == n_aux - 1
    assert cut.fit_state.metric_decomposition == "eigen"
    assert full.factors.shape[1] == n_aux
    assert full.fit_state.metric_decomposition == "cholesky"
    assert full.fit_state.keep.all()

    # Reference: T^dagger M^-1 T by a Cholesky solve of the raw metric,
    # rebuilt from the retained eigensystem of the unthresholded probe.
    U = probe.fit_state.eigenvectors
    metric = (U * w) @ U.conj().T
    from scipy.linalg import cho_factor, cho_solve
    factor = cho_factor(metric, lower=True)
    for k in range(len(bras)):
        raw = probe.fit_state.three_center[k].reshape(n_aux, -1)
        reference = raw.conj().T @ cho_solve(factor, raw)
        fitted = full.factors[k].reshape(n_aux, -1)
        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(fitted.conj().T @ fitted, reference,
                                   atol=1e-10 * scale, rtol=0)
        truncated = cut.factors[k].reshape(n_aux - 1, -1)
        # The cut loses the small mode's (positive) Coulomb-metric content.
        assert np.max(np.abs(truncated.conj().T @ truncated - reference)) > 1e-8 * scale


def test_leaf_falls_back_on_an_exactly_repeated_auxiliary():
    from vibeqc import _vibeqc_core as core

    system, basis, auxiliary = _fit_fixture()
    shells = list(auxiliary.shells()) + [list(auxiliary.shells())[0]]
    repeated = core.BasisSet(system.unit_cell_molecule(), shells,
                             "repeated-auxiliary", coefficients_pre_normalized=True)
    state = {}
    batch = _build_lpq_range_separated_shared_q(
        system, basis, repeated, np.zeros((1, 3)), np.zeros(3),
        linear_dep_thr=1e-9, retain_fit_state=True, _metric_state=state, **_OPTIONS)
    assert batch.fit_state.metric_decomposition == "eigen"
    assert batch.factors.shape[1] < repeated.nbasis
    assert state["metric_decomposition"] == "eigen"
    np.testing.assert_array_equal(batch.fit_state.keep, batch.metric_eigenvalues > 1e-9)


def _lih_fcc():
    a = 4.084 / 0.529177210903
    lattice = (a / 2) * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a / 2] * 3)])
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


# PySCF 2.14.0 on the identical cell: unit='B', precision=1e-10,
# GDF(auxbasis='def2-svp-jkfit') (= vibe-qc's 69-function def2-svp-jk),
# exxdiv='ewald', conv_tol=1e-11. Its Cholesky succeeds at every transfer.
_PYSCF_LIH_GAMMA = -8.335183216884866
_PYSCF_LIH_K222 = -7.922003260057798


@pytest.mark.slow
def test_lih_gamma_default_matches_pyscf_gdf():
    """#179: the default rsgdf fit keeps all 69 modes and matches PySCF.

    At the former 1e-9 cut (65 modes) this was -2.59 uHa below PySCF.
    """
    system, basis = _lih_fcc()
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.max_iter = 80
    result = vq.run_krhf_periodic_gdf(system, basis, (1, 1, 1), opts,
                                      aux_basis="def2-svp-jk")
    assert result.converged
    assert result.linear_dependence.n_fit_kept == 69
    assert abs(result.energy - _PYSCF_LIH_GAMMA) < 1e-7


@pytest.mark.slow
def test_lih_k222_default_matches_pyscf_gdf():
    """#179 headline limb: LiH/STO-3G KRHF (2,2,2) within 1e-7 Ha of PySCF.

    At the former 1e-9 cut it sat -31.9 uHa below PySCF GDF.
    """
    system, basis = _lih_fcc()
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.max_iter = 80
    result = vq.run_krhf_periodic_gdf(system, basis, (2, 2, 2), opts,
                                      aux_basis="def2-svp-jk")
    assert result.converged
    assert abs(result.energy - _PYSCF_LIH_K222) < 1e-7


def test_diffuse_zero_transfer_metric_block_is_the_reciprocal_coulomb_sum():
    """#179: diffuse auxiliary rows of the q=0 metric avoid the SR p=0
    cancellation (Ye & Berkelbach 2021, Eq. 15).

    The replaced block equals the native SR + LR - pi/(V omega^2) q_P q_Q
    assembly to that assembly's rounding, is Hermitian and converged in
    its reciprocal cutoff. Non-centrosymmetric skew cell, so the complex
    phase convention is exercised.
    """
    import inspect
    from vibeqc import _vibeqc_core as core
    from vibeqc import aux_basis as ab
    from vibeqc.aux_basis import make_aux_basis_set
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    lattice = np.array([[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]]).T
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(3, [0.2, 0.3, 0.4]),
                                            vq.Atom(1, [3.2, 0.9, 1.4])])
    mol = system.unit_cell_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    captured = {}
    native = core.compute_gdf_range_separated_integrals

    def recording(*args, **kwargs):
        out = native(*args, **kwargs)
        captured["metric"] = np.array(out[0])
        return out

    workspace = max(64 * 1024**2, core.gdf_short_range_workspace_bytes(
        basis, aux, core.get_num_threads()))
    core.compute_gdf_range_separated_integrals = recording
    try:
        _build_range_separated_lpq_cache(
            system, basis, aux, np.zeros((1, 3)), True, omega=0.4,
            raw_integral_error=1e-10, linear_dep_thr=1e-9,
            memory_byte_cap=4 * 1024**3, native_workspace_byte_cap=workspace,
            image_candidate_cap=10_000_000, reciprocal_candidate_cap=10_000_000,
            metric_decomposition="eigen")  # the unpatched native metric
    finally:
        core.compute_gdf_range_separated_integrals = native
    metric = captured["metric"]
    patched = metric.copy()
    diffuse = ab._rsgdf_diffuse_metric_zero_transfer(system, aux, 0.4, patched)
    assert diffuse.size == 8  # Li s(0.136, 0.069) and p(0.149, 0.075)
    np.testing.assert_allclose(patched, metric, atol=2e-12, rtol=0)
    np.testing.assert_allclose(patched, patched.conj().T, atol=1e-14, rtol=0)
    rest = np.setdiff1d(np.arange(aux.nbasis), diffuse)
    np.testing.assert_array_equal(patched[np.ix_(rest, rest)], metric[np.ix_(rest, rest)])
    # Doubling the reciprocal exponent window changes nothing beyond rounding.
    source = inspect.getsource(ab._rsgdf_diffuse_metric_zero_transfer)
    namespace = dict(ab.__dict__)
    exec(source.replace("* 48.0", "* 96.0").replace(
        "def _rsgdf_diffuse_metric_zero_transfer", "def wider"), namespace)
    wider = metric.copy()
    namespace["wider"](system, aux, 0.4, wider)
    np.testing.assert_allclose(wider, patched, atol=1e-14, rtol=0)
    # No diffuse auxiliary function: nothing is touched.
    untouched = metric.copy()
    assert ab._rsgdf_diffuse_metric_zero_transfer(system, aux, 0.2, untouched).size == 0
    np.testing.assert_array_equal(untouched, metric)
