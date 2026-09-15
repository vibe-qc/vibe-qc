from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.bands import _HARTREE_TO_EV
from vibeqc.periodic_gdf_properties import gdf_properties_from_result


def _state(*, spin=False, scalar_gamma=False):
    system = SimpleNamespace(unit_cell=[
        SimpleNamespace(Z=1, xyz=[0., 0., 0.]),
        SimpleNamespace(Z=1, xyz=[1.4, 0., 0.]),
    ])
    basis = SimpleNamespace(nbasis=2, shells=lambda: [
        SimpleNamespace(l=0, atom_index=0), SimpleNamespace(l=0, atom_index=1),
    ])
    c = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
    e = np.array([-1.2, -.8])
    fock = c @ np.diag(e) @ c.T
    density = 2 * np.outer(c[:, 0], c[:, 0])
    wrap = (lambda a: a) if scalar_gamma else (lambda a: [a])
    result = SimpleNamespace(converged=True, overlap=wrap(np.eye(2)),
                             hcore=wrap(-np.eye(2)))
    if not scalar_gamma:
        result.kpoint_weights = np.array([1.])
        result.kpoints_cart = np.zeros((1, 3))
    for suffix in (('_alpha', '_beta') if spin else ('',)):
        for name, value in (
            ('mo_energies', e), ('mo_coeffs', c), ('fock', fock),
            ('density', density / 2 if spin else density),
        ):
            setattr(result, name + suffix, wrap(value))
        setattr(result, 'fermi_level' + suffix, -1.)
    return result, system, basis


@pytest.mark.parametrize('scalar_gamma', [False, True])
def test_spectra_project_actual_hamiltonian_and_count_valence_density(scalar_gamma):
    result, system, basis = _state(scalar_gamma=scalar_gamma)
    # Bare atomic numbers deliberately disagree with the accepted electron
    # count, as for an ECP valence basis. Properties must follow the density.
    system.unit_cell[0].Z = 11
    dos, pdos, coop, cohp, mayer, _ = gdf_properties_from_result(
        result, system, basis, coop_cohp=True,
    )
    energies = dos['energies']
    assert dos['n_electrons'] == pytest.approx(2.)
    assert np.trapezoid(dos['dos'], energies) == pytest.approx(2., abs=1e-11)
    np.testing.assert_allclose(pdos['projections'].sum(axis=0), dos['dos'], atol=1e-12)
    assert cohp['integrated'][0] == pytest.approx(.2 * _HARTREE_TO_EV, abs=1e-12)
    occupied = energies < 0.
    assert np.trapezoid(cohp['projections'][0, occupied], energies[occupied]) == pytest.approx(
        cohp['integrated'][0], abs=1e-11,
    )
    np.testing.assert_allclose(coop['projections'], 0., atol=0)
    assert mayer[0, 1] == pytest.approx(1., abs=1e-12)
    # Hcore gives zero off-site COHP for this two-level system. Altering the
    # stored bare operator cannot change the accepted-Hamiltonian projection.
    result.hcore = None
    again = gdf_properties_from_result(result, system, basis, coop_cohp=True)
    np.testing.assert_array_equal(again[3]['projections'], cohp['projections'])


def test_spin_singlet_reduces_to_restricted_bond_populations():
    restricted = gdf_properties_from_result(*_state(), coop_cohp=True)
    unrestricted = gdf_properties_from_result(*_state(spin=True), coop_cohp=True)
    for index in (2, 3):
        for field in ('projections', 'integrated'):
            np.testing.assert_allclose(unrestricted[index][field].sum(axis=0),
                                       restricted[index][field], atol=1e-12, rtol=0)
    np.testing.assert_allclose(unrestricted[4], restricted[4], atol=1e-12, rtol=0)


def test_rohf_does_not_mislabel_an_effective_operator_as_spin_cohp():
    result, system, basis = _state(spin=True)
    result.mo_energies = result.mo_energies_alpha
    result.mo_coeffs = result.mo_coeffs_alpha
    del result.mo_energies_alpha
    del result.mo_energies_beta
    with pytest.raises(NotImplementedError, match='effective orbital operator'):
        gdf_properties_from_result(result, system, basis, coop_cohp=True)


def _rohf_state(*, multi_k=False):
    result, system, basis = _state(spin=True)
    result.mo_energies = result.mo_energies_alpha
    result.mo_coeffs = result.mo_coeffs_alpha
    result.fock = result.fock_alpha
    c = result.mo_coeffs[0].astype(complex)
    result.fock_alpha = [np.array([[-1.4, .3+.1j], [.3-.1j, -.7]])]
    result.fock_beta = [np.array([[-1.1, -.1+.05j], [-.1-.05j, -.4]])]
    # Non-idempotent accepted densities, including orbital coherence:
    # integrated populations cannot be recovered from integer refills.
    result.density_alpha = [c @ np.diag([.8, .2]) @ c.conj().T]
    result.density_beta = [c @ np.array([[.4, .08j], [-.08j, .1]]) @ c.conj().T]
    for suffix in ('_alpha', '_beta'):
        delattr(result, 'mo_energies'+suffix)
        delattr(result, 'mo_coeffs'+suffix)
    result.mo_occupations = [np.array([2., 1.])]
    if multi_k:
        result.kpoint_weights = np.array([.3, .7])
        result.kpoints_cart = np.array([[0., 0., 0.], [.13, -.07, .02]])
        for name in ('overlap', 'mo_coeffs', 'mo_energies', 'mo_occupations', 'fock',
                     'fock_alpha', 'fock_beta', 'density_alpha', 'density_beta'):
            values = getattr(result, name)
            values.append(values[0].copy())
        result.fock_alpha[1] *= .6
        result.fock_beta[1] *= 1.3
        result.density_beta[1] *= .7
    return result, system, basis


@pytest.mark.parametrize('multi_k', [False, True])
def test_private_rohf_projects_spin_focks_on_effective_energy_axis(multi_k):
    result, system, basis = _rohf_state(multi_k=multi_k)
    dos, pdos, coop, cohp, _, _ = gdf_properties_from_result(
        result, system, basis, coop_cohp=True, _rohf_effective_spectrum=True,
    )
    for payload in (dos, pdos, coop, cohp):
        assert payload['energy_operator'] == 'roothaan-effective-fock'
        assert payload['orbital_basis'] == 'shared-rohf'
        assert payload['population_density'] == 'accepted-spin-density'
        assert payload['validation_status'] == 'private-unvalidated'
    assert cohp['projection_operator'] == 'physical-spin-fock'
    assert coop['projection_operator'] == dos['projection_operator'] == 'overlap'
    np.testing.assert_allclose(dos['dos'][0], dos['dos'][1], atol=0., rtol=0.)
    np.testing.assert_allclose(pdos['projections'].sum(axis=1), dos['dos'], atol=1e-12)
    axis = dos['energies'] + dos['fermi_energy_ev']
    expected_electrons = 0.
    for spin, suffix in enumerate(('_alpha', '_beta')):
        expected_curve = np.zeros_like(axis)
        expected_integral = 0.
        for wk, e, c, f, d, s in zip(
            result.kpoint_weights, result.mo_energies, result.mo_coeffs,
            getattr(result, 'fock'+suffix), getattr(result, 'density'+suffix), result.overlap,
        ):
            assert np.linalg.norm(c.conj().T @ f @ c - np.diag(e)) > .1
            expected_integral -= wk * (d[1, 0]*f[0, 1]).real * _HARTREE_TO_EV
            expected_electrons += wk * np.trace(d@s).real
            for band, energy in enumerate(e):
                weight = (c[0, band].conj()*f[0, 1]*c[1, band]).real
                gaussian = np.exp(-.5*((axis-energy*_HARTREE_TO_EV)/.05)**2)/(.05*np.sqrt(2*np.pi))
                expected_curve -= wk * weight * _HARTREE_TO_EV * gaussian
        np.testing.assert_allclose(cohp['projections'][spin, 0], expected_curve, atol=2e-11, rtol=1e-12)
        assert cohp['integrated'][spin, 0] == pytest.approx(expected_integral, abs=1e-12)
    assert dos['n_electrons'] == pytest.approx(expected_electrons, abs=1e-12)
    # Changing the refill labels must not change the accepted populations.
    result.mo_occupations = None
    again = gdf_properties_from_result(
        result, system, basis, coop_cohp=True, _rohf_effective_spectrum=True,
    )
    np.testing.assert_array_equal(again[3]['integrated'], cohp['integrated'])


@pytest.mark.parametrize('invalid,match', [
    ('effective', 'do not diagonalize'), ('spin', 'Hermitian'), ('mesh', 'inconsistent meshes'),
])
def test_private_rohf_rejects_invalid_operator_channels(invalid, match):
    result, system, basis = _rohf_state()
    if invalid == 'effective':
        result.fock = result.fock_alpha
    elif invalid == 'spin':
        result.fock_beta[0][0, 1] += .3j
    else:
        result.fock_alpha = []
    with pytest.raises(ValueError, match=match):
        gdf_properties_from_result(result, system, basis, _rohf_effective_spectrum=True)


def test_private_rohf_closed_shell_limit_matches_restricted_populations():
    result, system, basis = _state(spin=True)
    result.mo_energies = result.mo_energies_alpha
    result.mo_coeffs = result.mo_coeffs_alpha
    result.fock = result.fock_alpha
    del result.mo_energies_alpha
    del result.mo_energies_beta
    got = gdf_properties_from_result(
        result, system, basis, coop_cohp=True, _rohf_effective_spectrum=True,
    )
    reference = gdf_properties_from_result(*_state(), coop_cohp=True)
    for index in (2, 3):
        for field in ('projections', 'integrated'):
            np.testing.assert_allclose(got[index][field].sum(axis=0),
                                       reference[index][field], atol=1e-12, rtol=0.)
    np.testing.assert_allclose(got[4], reference[4], atol=1e-12, rtol=0.)


def test_private_rohf_admission_precedes_spectrum_allocation(monkeypatch):
    import vibeqc.periodic_gdf_properties as properties
    state = _rohf_state()
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted ROHF spectrum allocated an energy grid')
    monkeypatch.setattr(properties.np, 'linspace', forbidden)
    with pytest.raises(MemoryError, match='spectral properties require'):
        gdf_properties_from_result(*state, _rohf_effective_spectrum=True, memory_byte_cap=1)


def test_fractional_accepted_density_controls_integrated_bond_population():
    result, system, basis = _state()
    result.density[0] *= .6
    dos, _, _, cohp, mayer, _ = gdf_properties_from_result(
        result, system, basis, coop_cohp=True,
    )
    assert dos['n_electrons'] == pytest.approx(1.2)
    assert cohp['integrated'][0] == pytest.approx(.12 * _HARTREE_TO_EV)
    assert mayer[0, 1] == pytest.approx(.36)


def test_property_rejects_unmatched_returned_hamiltonian():
    result, system, basis = _state()
    result.fock = result.hcore
    with pytest.raises(ValueError, match='do not diagonalize'):
        gdf_properties_from_result(result, system, basis)


def test_spectrum_admission_precedes_allocation():
    with pytest.raises(MemoryError, match='spectral properties require'):
        gdf_properties_from_result(*_state(), coop_cohp=True, memory_byte_cap=1)
