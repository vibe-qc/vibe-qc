"""Opt-in failure observation must preserve the native UKS contract."""
import numpy as np
import pytest
from vibeqc import _vibeqc_core as core


def witness():
    mol = core.Molecule([core.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = core.BasisSet(mol, "sto-3g")
    opts = core.UKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 2
    opts.stability_check = False
    return mol, basis, opts


def test_operator_capture_success_preserves_result():
    mol, basis, opts = witness()
    expected = core.run_uks(mol, basis, opts)
    calls = []
    actual = core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert calls == []
    assert actual.energy == expected.energy
    assert actual.n_iter == expected.n_iter
    assert actual.converged == expected.converged
    for name in ("density_alpha", "density_beta", "fock_alpha", "fock_beta"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_operator_capture_preserves_early_failure_and_restores_scope():
    mol, basis, opts = witness()
    opts.functional = "not-a-functional-for-capture-test"
    with pytest.raises(Exception) as original:
        core.run_uks(mol, basis, opts)
    calls = []
    with pytest.raises(type(original.value)) as observed:
        core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert str(observed.value) == str(original.value)
    assert len(calls) == 1
    assert calls[0]["error"] == str(original.value)
    assert not calls[0]["failing_solve_captured"]
    assert calls[0]["matrices"] == {}
    # A failed observer scope cannot poison a subsequent calculation.
    opts.functional = "pbe"
    core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert len(calls) == 1


def test_operator_capture_callback_failure_cannot_replace_native_error(monkeypatch):
    import sys

    mol, basis, opts = witness()
    opts.functional = "not-a-functional-for-capture-test"
    with pytest.raises(Exception) as original:
        core.run_uks(mol, basis, opts)
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def broken_observer(record):
        raise RuntimeError("observer deliberately failed")

    with pytest.raises(type(original.value)) as observed:
        core._run_uks_with_operator_capture(mol, basis, opts, broken_observer)
    assert str(observed.value) == str(original.value)
    assert len(unraisable) == 1
    assert str(unraisable[0].exc_value) == "observer deliberately failed"


# The shared Fock solve projects the Hermitian part of an admitted AO operator
# (#672). The AO-frame guard (relative anti-Hermitian defect <= 1e-10) still
# refuses a non-Hermitian builder; an operator that passes it is no longer
# refused because an ill-conditioned X amplifies its roundoff in X^H F X.
def _ill_conditioned_pair(rel_defect):
    rng = np.random.default_rng(672)
    n = 30
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.geomspace(1e-4, 2.0, n)  # sigma_max(X)^2 = 1e4, like a diffuse basis
    overlap = (q * s) @ q.T
    overlap = 0.5 * (overlap + overlap.T)
    g = rng.normal(size=(n, n))
    fock = 0.5 * (g + g.T)
    # Anti-Hermitian part aligned with the smallest-overlap directions: the
    # case the X projection amplifies most.
    anti = np.outer(q[:, 0], q[:, 1]) - np.outer(q[:, 1], q[:, 0])
    anti *= rel_defect * np.linalg.norm(fock) / np.linalg.norm(anti)
    return fock, fock + anti, overlap, q / np.sqrt(s)


def _rel_defect(a):
    return np.linalg.norm(a - a.T) / max(1.0, np.linalg.norm(a))


def test_admitted_roundoff_is_not_refused_after_projection():
    hermitian, fock, overlap, x = _ill_conditioned_pair(1e-11)
    assert _rel_defect(fock) < 1e-10          # passes the AO guard
    assert _rel_defect(x.T @ fock @ x) > 1e-10  # the former projected refusal
    out = core.diagonalize_bloch(fock.astype(complex), overlap.astype(complex), 1e-9)
    # Reference: the generalized problem of the Hermitian part, in X.
    ref = np.linalg.eigvalsh(x.T @ hermitian @ x)
    np.testing.assert_allclose(np.asarray(out.energies), ref, rtol=1e-11, atol=1e-9)
    c = np.asarray(out.coefficients)
    np.testing.assert_allclose(c.conj().T @ overlap @ c, np.eye(len(ref)), atol=1e-8)


def test_non_hermitian_builder_is_still_refused():
    _, fock, overlap, _ = _ill_conditioned_pair(1e-8)
    assert _rel_defect(fock) > 1e-10
    with pytest.raises(ValueError, match="non-Hermitian component"):
        core.diagonalize_bloch(fock.astype(complex), overlap.astype(complex), 1e-9)


def test_ch_doublet_hybrid_aug_cc_pvtz_converges():
    # The #672 reproducer: refused at SCF iteration 16 before the fix.
    from vibeqc.dispersion_d4_reference_systems import build_molecule, reference_systems_for
    from vibeqc.runner import apply_ks_grid_default

    rs = next(r for r in reference_systems_for(6) if r.multiplicity == 2)
    mol = build_molecule(rs)
    basis = core.BasisSet(mol, "aug-cc-pvtz")
    opts = core.UKSOptions()
    apply_ks_grid_default(opts, "orca-defgrid3")
    opts.functional = "pbe0"
    calls = []
    result = core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert calls == []
    assert result.converged
    # PySCF 2.14 UKS-PBE0, same geometry/basis, grid level 9, after its
    # internal-stability follow: -38.438478727 Ha (<S^2> 0.88995). The 1e-6
    # allowance covers orca-defgrid3 against the converged grid (1.4e-7).
    assert result.energy == pytest.approx(-38.438478727, abs=1e-6)
