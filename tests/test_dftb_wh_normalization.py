"""Wolfsberg-Helmholz normalization against independent two-orbital oracles.

Wolfsberg and Helmholz, J. Chem. Phys. 20, 837 (1952), p. 838,
DOI 10.1063/1.1700580: H_AB = kappa*S_AB*(h_A+h_B)/2.
These are internal model/algebra checks, not physical geometry benchmarks.
"""
from __future__ import annotations

import numpy as np
import pytest

# Stewart, J. Chem. Phys. 52, 431 (1970), normalized 1s STO-6G expansion.
_ALPHA = np.array([0.06510953954, 0.1580884151, 0.4070988982,
                   1.185056519, 4.235915534, 23.10303149]) * 1.24**2
_COEFF = np.array([0.1303340841, 0.4164915298, 0.3705627997,
                   0.1685383049, 0.04936149294, 0.00916359628])
_EPSILON = -0.2066  # Ha; unchanged default hydrogen on-site parameter.
_KAPPA = 1.75
_REP_A = 5.0  # Ha bohr^12; unchanged placeholder, no fitting claim.


def _overlap(distance):
    """Contract normalized primitive s-Gaussian products and their derivative."""
    a, b = _ALPHA[:, None], _ALPHA[None, :]
    weight = _COEFF[:, None] * _COEFF[None, :] * (2*np.sqrt(a*b)/(a+b))**1.5
    weight /= weight.sum()
    reduced = a*b/(a+b)
    terms = weight * np.exp(-reduced*distance**2)
    return terms.sum(), (-2*reduced*distance*terms).sum()


def _molecular_oracle(distance):
    s, ds = _overlap(distance)
    bonding = _EPSILON * (1 + _KAPPA*s) / (1+s)
    antibonding = _EPSILON * (1 - _KAPPA*s) / (1-s)
    energy = 2*bonding + _REP_A/distance**12
    derivative = 2*_EPSILON*(_KAPPA-1)*ds/(1+s)**2 - 12*_REP_A/distance**13
    return s, np.array([bonding, antibonding]), energy, derivative


def _periodic_oracle(distance=2.0, length=6.0, fractional_k=0.0):
    """Two H per 1-D cell, pair cutoff 9 bohr: R, L-R, L, L+R.

    H(k)=kappa*epsilon*S(k)+(1-kappa)*epsilon*I. The occupied
    eigenvalue is associated with the larger overlap eigenvalue, which
    follows from kappa>1 and epsilon<0, without a native diagonalizer.
    """
    s0, d0 = _overlap(distance)
    sm, dm = _overlap(length-distance)
    sp, dp = _overlap(length+distance)
    sl, dl = _overlap(length)
    phase = np.exp(2j*np.pi*fractional_k)
    off_diagonal = s0 + sm/phase + sp*phase
    diagonal = 1 + 2*sl*np.cos(2*np.pi*fractional_k)
    overlap_eigenvalues = np.array([diagonal+abs(off_diagonal),
                                    diagonal-abs(off_diagonal)])
    orbitals = _KAPPA*_EPSILON + (1-_KAPPA)*_EPSILON/overlap_eigenvalues
    repulsive = _REP_A*(distance**-12 + (length-distance)**-12
                       + (length+distance)**-12 + 2*length**-12)
    energy = 2*orbitals[0] + repulsive
    # Derivatives here are for Gamma only.
    lam = 1 + 2*sl + s0 + sm + sp
    factor = -2*(1-_KAPPA)*_EPSILON/lam**2
    dr = factor*(d0-dm+dp) + 12*_REP_A*(-distance**-13
                  + (length-distance)**-13 - (length+distance)**-13)
    dl_energy = factor*(2*dl+dm+dp) - 12*_REP_A*(
        2*length**-13 + (length-distance)**-13 + (length+distance)**-13)
    strain = length*dl_energy + distance*dr
    return orbitals, energy, dr, strain


@pytest.fixture(scope="module")
def native():
    from vibeqc import Atom, Molecule
    from vibeqc._vibeqc_core import PeriodicSystem, monkhorst_pack, semiempirical
    return Atom, Molecule, PeriodicSystem, monkhorst_pack, semiempirical


@pytest.mark.parametrize("method", ["dftb0", "udftb0", "scc_dftb", "uscc_dftb"])
@pytest.mark.parametrize("distance", [1.4, 2.0])
def test_molecular_hydrogen_normalization_and_gradient(native, method, distance):
    Atom, Molecule, _, _, se = native
    params = se.SemiempiricalParameters.dftb0_default()
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [distance, 0, 0])], 0, 1)
    result = getattr(se, "run_"+method)(mol, params)
    if "scc" in method:
        assert result.converged
    s, orbitals, energy, derivative = _molecular_oracle(distance)
    # Absolute roundoff budgets for analytic Gaussian/algebra comparisons,
    # not external-method gradient or paper-acceptance tolerances.
    np.testing.assert_allclose(result.overlap, [[1, s], [s, 1]], rtol=0, atol=2e-12)
    assert np.asarray(result.hamiltonian)[0, 1] == pytest.approx(
        _KAPPA*_EPSILON*s, abs=2e-12)
    assert orbitals[0] < orbitals[1]
    np.testing.assert_allclose(result.mo_energies, orbitals, rtol=0, atol=2e-12)
    assert result.energy == pytest.approx(energy, abs=2e-12)
    np.testing.assert_allclose(result.charges, 0, rtol=0, atol=2e-12)
    gradient = getattr(se, "compute_"+method+"_gradient")(mol, result, params)
    np.testing.assert_allclose(gradient, [[-derivative, 0, 0], [derivative, 0, 0]],
                               rtol=0, atol=2e-11)


def test_heteronuclear_hopping_preserves_atom_average(native):
    Atom, Molecule, _, _, se = native
    params = se.SemiempiricalParameters.dftb0_default()
    mol = Molecule([Atom(8, [0, 0, 0]), Atom(1, [1.5, 0.8, 0.2]),
                    Atom(1, [-1.3, 1.0, -0.1])], 0, 1)
    result = se.run_dftb0(mol, params)
    atom_of_ao = [0, 0, 0, 0, 1, 2]
    elements = [8, 1, 1]
    H, S = np.asarray(result.hamiltonian), np.asarray(result.overlap)
    for mu, a in enumerate(atom_of_ao):
        for nu, b in enumerate(atom_of_ao):
            if a != b:
                expected = 0.5*params.kappa*S[mu, nu]*(
                    params.average_on_site(elements[a])+params.average_on_site(elements[b]))
                assert H[mu, nu] == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("method", ["dftb0", "udftb0", "scc_dftb", "uscc_dftb"])
def test_periodic_gamma_hydrogen_normalization_and_derivatives(native, method):
    Atom, _, PeriodicSystem, _, se = native
    params = se.SemiempiricalParameters.dftb0_default()
    system = PeriodicSystem(1, np.diag([6., 30., 30.]),
        [Atom(1, [0, 0, 0]), Atom(1, [2., 0, 0])], 0, 1)
    options = se.PeriodicSCCOptions() if "scc" in method else se.PeriodicDFTB0Options()
    options.cutoff_bohr = 9.
    result = getattr(se, "run_"+method+"_gamma")(system, params, options)
    if "scc" in method:
        assert result.converged
    orbitals, energy, derivative, strain = _periodic_oracle()
    np.testing.assert_allclose(result.mo_energies, orbitals, rtol=0, atol=2e-12)
    assert result.energy == pytest.approx(energy, abs=2e-12)
    # Noniterative Gamma results expose densities, not a charges field.
    # Each H has one AO and one neutral valence electron, so reconstruct
    # the Mulliken net charges from the actual returned density and overlap.
    density = (np.asarray(result.density_alpha) + np.asarray(result.density_beta)
               if method.startswith("u") else np.asarray(result.density))
    charges = 1. - np.diag(density @ np.asarray(result.overlap_gamma))
    np.testing.assert_allclose(charges, 0, rtol=0, atol=2e-12)
    if "scc" in method:
        np.testing.assert_allclose(result.charges, charges, rtol=0, atol=2e-12)
    gradient = getattr(se, "compute_periodic_"+method+"_gradient")(system, result, params)
    np.testing.assert_allclose(gradient, [[-derivative, 0, 0], [derivative, 0, 0]],
                               rtol=0, atol=2e-11)
    if method in ("dftb0", "scc_dftb"):
        stress = getattr(se, "compute_periodic_"+method+"_stress")(system, result, params)
        expected = np.zeros((3, 3)); expected[0, 0] = strain/(6*30*30)
        np.testing.assert_allclose(stress, expected, rtol=0, atol=2e-13)


@pytest.mark.parametrize("method", ["dftb0", "scc_dftb"])
def test_complex_bloch_hydrogen_normalization(native, method):
    Atom, _, PeriodicSystem, monkhorst_pack, se = native
    params = se.SemiempiricalParameters.dftb0_default()
    system = PeriodicSystem(1, np.diag([6., 30., 30.]),
        [Atom(1, [0, 0, 0]), Atom(1, [2., 0, 0])], 0, 1)
    mesh = monkhorst_pack(system, (5, 1, 1))
    if method == "dftb0":
        result = se.run_dftb0_kpoints(system, params, mesh, 9.)
    else:
        result = se.run_scc_dftb_kpoints(system, params, mesh, se.SCCOptions(), 9.)
        assert result.converged
    # Sorted spectra avoid depending on mesh ordering or time-reversal pairing.
    expected = [_periodic_oracle(fractional_k=k/5) for k in (-2, -1, 0, 1, 2)]
    spectra = np.array([row[0] for row in expected])
    assert spectra[:, 1].min() > spectra[:, 0].max()  # common gapped filling
    np.testing.assert_allclose(np.sort(np.asarray(result.eps_per_k), axis=0),
                               np.sort(spectra, axis=0), rtol=0, atol=2e-12)
    assert result.energy == pytest.approx(np.mean([row[1] for row in expected]), abs=2e-12)
