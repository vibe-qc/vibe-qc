"""Independent fixed-N checks of the HF/CC zero-mode seam (#530).

These are source checks, not periodic scientific acceptance. Xing--Lin,
doi:10.1103/PhysRevX.14.011059, Sec. II B, Eqs. (7)-(11).
"""
import numpy as np
import pytest

from vibeqc.periodic.chi.cc_zero_mode import CONVENTION, reference_madelung
from vibeqc.periodic.chi.canonical_cc import FactorIntegrals, evaluate_same_source_cc, ChiCCOptions
from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter, _spin_state
from tests.test_chi_cc_equations import model, determinant_reference


@pytest.mark.parametrize('nk', [1, 3])
@pytest.mark.parametrize('unrestricted,frozen', [(False, (0, 0)), (True, (1, 0))])
def test_diagonal_eri_has_exact_spin_character_and_frozen_normalization(monkeypatch, nk, unrestricted, frozen):
    from tests.test_chi_canonical_mp2 import fixture
    ref, channels = fixture(monkeypatch, nk=nk, unrestricted=unrestricted, source='classic', nao=4)
    adapter = CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    args = (adapter, tuple(c for c, e, n in channels), tuple(n for c, e, n in channels), frozen, 1)
    raw = FactorIntegrals(*args)
    corrected = FactorIntegrals(*args, madelung=.17)
    full = (slice(None),)*4
    delta = corrected[full]-raw[full]
    # Independent spin-orbital Kronecker deltas, with no production panels,
    # momentum grouping or factor contraction in the expected correction.
    labels = corrected.labels
    expected = np.zeros_like(delta)
    for p, q, r, s in np.ndindex(expected.shape):
        expected[p, q, r, s] = .17*(
            int(labels[p] == labels[r] and labels[q] == labels[s]) -
            int(labels[p] == labels[s] and labels[q] == labels[r]))
    np.testing.assert_allclose(delta, expected, atol=2e-15, rtol=0)
    no = corrected.no
    np.testing.assert_array_equal(delta[:no, :no, no:, no:], 0.)
    # Virtual ladders stream external slices; their diagonal terms must also
    # be present, and the repeated access must not accumulate the seam twice.
    sl = (slice(no, no+1), slice(no, None), slice(no, None), slice(no, None))
    np.testing.assert_allclose(corrected[sl]-raw[sl], expected[sl], atol=2e-15, rtol=0)
    np.testing.assert_array_equal(corrected[full], corrected[full])


@pytest.mark.parametrize('no', [2, 3])
def test_constant_kernel_cancels_in_independent_fixed_electron_hamiltonian(no):
    from vibeqc.periodic.chi._cc_equations import residual, _ccsd_energy_so
    f, g, t1, t2 = model(n=6, no=no)
    xi = .17
    identity = np.eye(len(f))
    correction = xi*(np.einsum('pr,qs->pqrs', identity, identity)-
                     np.einsum('ps,qr->pqrs', identity, identity))
    shifted_f = f.copy()
    shifted_f[:no, :no] -= xi*np.eye(no)
    before = determinant_reference(f, g, t1, t2)[:3]
    after = determinant_reference(shifted_f, g+correction, t1, t2)[:3]
    for original, corrected in zip(before, after):
        np.testing.assert_allclose(corrected, original, atol=3e-13, rtol=0)
    actual = (_ccsd_energy_so(t1, t2, shifted_f, g+correction, no),
              *residual(t1, t2, shifted_f, g+correction, no))
    for oracle, measured in zip(before, actual):
        np.testing.assert_allclose(measured, oracle, atol=3e-13, rtol=0)
    # The historical energy-only approximation defines a different finite
    # Hamiltonian and does not have the constant-kernel invariance.
    energy_only = determinant_reference(shifted_f, g, t1, t2)
    assert max(np.max(abs(energy_only[i]-before[i])) for i in (1, 2)) > 1e-3


@pytest.mark.parametrize('method', ['RHF', 'UHF'])
@pytest.mark.parametrize('nk', [1, 3])
def test_recover_executed_seam_and_reject_non_scalar_or_spin_mismatch(monkeypatch, method, nk):
    from tests.test_chi_hf_checkpoint import fixture
    ref, _, _ = fixture(monkeypatch, method, nk)
    hf = ref.hf_result
    channels = _spin_state(hf, method, 1e-9)
    assert reference_madelung(hf, channels) == pytest.approx(.125, abs=2e-15)
    field = 'fock' if method == 'RHF' else 'fock_beta'
    original = getattr(hf, field)
    corrupted = original.copy()
    corrupted[0, -1, -1] += .01
    setattr(hf, field, corrupted)
    with pytest.raises(ValueError, match='scalar zero mode'):
        reference_madelung(hf, channels)


def test_public_result_and_checkpoint_bind_corrected_equations(monkeypatch, tmp_path):
    from tests.test_chi_hf_checkpoint import fixture
    from vibeqc.output.chi_correlation import chi_correlation_record, write_chi_correlation
    from vibeqc.output.checkpoint import read_restart_state, write_restart_state
    ref, _, _ = fixture(monkeypatch, 'UHF', 1)
    path = tmp_path/'corrected.vqr'
    options = ChiCCOptions(compute_triples=False, coulomb_zero_mode='matched-eri')
    result = evaluate_same_source_cc(ref, options=options, checkpoint_path=path)
    assert result.madelung_hartree == pytest.approx(.125)
    record = chi_correlation_record(result)
    assert record['coulomb_zero_mode'] == CONVENTION
    assert record['madelung_hartree'] == pytest.approx(.125)
    lines = []
    write_chi_correlation(lines.append, result)
    assert CONVENTION in ''.join(lines)
    metadata, arrays = read_restart_state(path)
    assert metadata['schema'] == 'vibeqc.chi.canonical-cc/v2'
    assert metadata['binding']['coulomb_zero_mode'] == CONVENTION
    for change in ('old-schema', 'old-equations'):
        altered = dict(metadata)
        if change == 'old-schema':
            altered['schema'] = 'vibeqc.chi.canonical-cc/v1'
        else:
            altered['binding'] = dict(metadata['binding'], coulomb_zero_mode='energy-only')
        write_restart_state(path, altered, arrays)
        with pytest.raises(ValueError, match='schema|incompatible'):
            evaluate_same_source_cc(ref, options=options, resume_from=path)


def test_historical_default_and_old_archive_only_reuse_energy_only(monkeypatch, tmp_path):
    from vibeqc.periodic.chi.cc_zero_mode import ENERGY_ONLY
    from tests.test_chi_hf_checkpoint import fixture
    from vibeqc.output.checkpoint import read_restart_state, write_restart_state
    ref, _, _ = fixture(monkeypatch, 'RHF', 1)
    path = tmp_path/'legacy.vqr'
    result = evaluate_same_source_cc(ref, compute_triples=False, checkpoint_path=path)
    assert result.coulomb_zero_mode == ENERGY_ONLY
    metadata, arrays = read_restart_state(path)
    metadata['schema'] = 'vibeqc.chi.canonical-cc/v1'
    metadata['binding'].pop('coulomb_zero_mode')
    write_restart_state(path, metadata, arrays)
    with pytest.raises(ValueError, match='schema'):
        evaluate_same_source_cc(ref, options=ChiCCOptions(compute_triples=False,
            coulomb_zero_mode='matched-eri'), resume_from=path)
    reused = evaluate_same_source_cc(ref, compute_triples=False, resume_from=path)
    assert reused.restart_mode == 'restored-completed-result'
    assert reused.e_ccsd_per_cell == result.e_ccsd_per_cell
    np.testing.assert_array_equal(reused.t2, result.t2)
    with pytest.raises(ValueError, match='incompatible'):
        evaluate_same_source_cc(ref, options=ChiCCOptions(compute_triples=False,
            coulomb_zero_mode='matched-eri'), resume_from=path)
    for value in ('auto', None, True):
        with pytest.raises(ValueError, match='coulomb_zero_mode'):
            ChiCCOptions(coulomb_zero_mode=value).validate()


def test_nonzero_public_modes_match_distinct_independent_hamiltonians(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture, reseal
    from vibeqc.periodic.chi import canonical_cc
    from vibeqc.periodic.chi.cc_zero_mode import ENERGY_ONLY
    ref, channels = fixture(monkeypatch, nk=1, source='classic', nao=3)
    hf = ref.hf_result
    xi = .17
    d = hf.density/2
    # Construct a common physical one-body operator for this prescribed
    # stationary RHF determinant. Restore the real seam verifier, not the
    # arbitrary-Fock algebra fixture's stub.
    hf.hcore = hf.fock-hf.fitted_operator.coulomb(2*d)+hf.fitted_operator.exchange(d)+xi*d
    monkeypatch.setattr(canonical_cc, 'reference_madelung', reference_madelung)
    ref = reseal(ref)
    raw = FactorIntegrals(CanonicalFactorAdapter(hf.fitted_operator), channels[0][0], 1, 0, 1)
    g = raw[(slice(None),)*4]
    eps = np.array([channels[spin][1][k, p] for k, p, spin in raw.labels])
    eye = np.eye(len(eps))
    extra = xi*(np.einsum('pr,qs->pqrs', eye, eye)-np.einsum('ps,qr->pqrs', eye, eye))
    energies = []
    for mode, convention, tensor in [('hf-energy-only', ENERGY_ONLY, g), ('matched-eri', CONVENTION, g+extra)]:
        result = evaluate_same_source_cc(ref, options=ChiCCOptions(compute_triples=False,
            coulomb_zero_mode=mode, residual_tolerance=1e-11, energy_tolerance=1e-13))
        energy, r1, r2, _ = determinant_reference(np.diag(eps), tensor, result.t1, result.t2)
        assert result.e_ccsd_per_cell == pytest.approx(energy, abs=2e-13)
        assert max(abs(r1).max(), abs(r2).max()) < 1e-11
        assert result.madelung_hartree == pytest.approx(xi, abs=1e-14)
        assert result.coulomb_zero_mode == convention
        energies.append(result.e_ccsd_per_cell)
    assert abs(energies[0]-energies[1]) > 1e-7


def test_finite_size_hessian_has_cubic_value_volume_scaling_and_tensor_covariance():
    from benchmarks.chi_finite_size import ewald_hessian
    cubic = np.eye(3)*12
    np.testing.assert_allclose(ewald_hessian(cubic), np.eye(3)*4*np.pi/(3*12**3), atol=2e-14, rtol=0)
    skew = np.array([[12., 1.3, .2], [0., 14., .7], [0., 0., 16.]])
    h = ewald_hessian(skew)
    np.testing.assert_allclose(ewald_hessian(2*skew), h/8, atol=2e-14, rtol=0)
    rotation = np.linalg.qr(np.random.default_rng(12).normal(size=(3, 3)))[0]
    np.testing.assert_allclose(ewald_hessian(rotation@skew), rotation@h@rotation.T, atol=2e-14, rtol=0)
