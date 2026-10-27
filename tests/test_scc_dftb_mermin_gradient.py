"""Molecular SCC-DFTB analytic gradient at finite electronic temperature (#563).

The Mermin free energy F = tr(D H0) - T S + E_scc + E_rep is stationary in the
Fermi-Dirac occupations, so its Hellmann-Feynman force carries the
energy-weighted density W = sum_i f_i eps_i c_i c_i^T with the occupations
that built D. The gradient used to rebuild W from the Aufbau 2/0 of the first
n_occ MOs, which is a different state whenever an occupation is fractional:
on square cyclobutadiene, whose half-filled frontier pair is exactly
degenerate, the force on a carbon was -0.0143 Ha/bohr against a central
difference of -0.0364.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import SCCDFTBModel

_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def _square_cyclobutadiene() -> Molecule:
    """D4h C4H4, C-C 1.44 A, C-H 1.089 A: an exactly degenerate pair holds
    two electrons, so at T > 0 each member carries occupation 1."""
    c, h = 0.72, 0.77
    rows = [(6, c, c), (6, -c, c), (6, -c, -c), (6, c, -c),
            (1, c + h, c + h), (1, -c - h, c + h), (1, -c - h, -c - h), (1, c + h, -c - h)]
    return Molecule(
        [Atom(z, [x * _BOHR_PER_ANGSTROM, y * _BOHR_PER_ANGSTROM, 0.0]) for z, x, y in rows],
        0, 1,
    )


def _displaced_furan() -> Molecule:
    """The Thiel-set furan (bohr) with its oxygen moved 0.15 bohr in x: a
    gapped molecule whose T = 0.05 Ha occupations are fractional on many
    levels."""
    rows = [
        (8, 0.15, 0.0, 2.19775132), (6, 2.07869859, 0.0, 0.67085273),
        (6, -2.07869859, 0.0, 0.67085273), (6, 1.34737463, 0.0, -1.78012188),
        (6, -1.34737463, 0.0, -1.78012188), (1, 3.92307115, 0.0, 1.54768558),
        (1, -3.92307115, 0.0, 1.54768558), (1, 2.63994721, 0.0, -3.37883007),
        (1, -2.63994721, 0.0, -3.37883007),
    ]
    return Molecule([Atom(z, [x, y, w]) for z, x, y, w in rows], 0, 1)


def _displaced(molecule: Molecule, atom: int, axis: int, step: float) -> Molecule:
    return Molecule(
        [
            Atom(a.Z, [c + (step if (i == atom and k == axis) else 0.0) for k, c in enumerate(a.xyz)])
            for i, a in enumerate(molecule.atoms)
        ],
        molecule.charge, molecule.multiplicity,
    )


def _model(molecule: Molecule, temperature: float) -> SCCDFTBModel:
    # The runner's Mermin retry recipe (no DIIS, low mixing), tightly converged.
    return SCCDFTBModel(molecule, max_iter=1500, charge_mixing=0.05,
                        electronic_temperature=temperature, use_diis=False)


@pytest.mark.parametrize(
    "molecule,temperature,components",
    [
        (_square_cyclobutadiene(), 0.001, [(0, 0), (0, 1), (4, 0)]),
        (_displaced_furan(), 0.05, [(0, 0), (1, 2), (5, 0)]),
    ],
    ids=["degenerate_cyclobutadiene", "smeared_furan"],
)
def test_mermin_gradient_matches_central_differences(molecule, temperature, components):
    model = _model(molecule, temperature)
    model.energy()
    assert model.converged
    occupations = np.asarray(model._last_result.occupations)
    fractional = (occupations > 1e-6) & (occupations < 2.0 - 1e-6)
    assert np.count_nonzero(fractional) >= 2
    gradient = np.asarray(model.gradient())
    step = 1.0e-4
    for atom, axis in components:
        energies = []
        for sign in (-1.0, 1.0):
            displaced = _model(_displaced(molecule, atom, axis, sign * step), temperature)
            energies.append(displaced.energy())
            assert displaced.converged
        central = (energies[1] - energies[0]) / (2.0 * step)
        # Measured 7e-8 (cyclobutadiene) and 2e-8 (furan); the former force
        # was 2.2e-2 off on the cyclobutadiene carbon.
        assert gradient[atom, axis] == pytest.approx(central, abs=2.0e-7)


def test_degenerate_pair_shares_the_occupation_that_built_the_density():
    model = _model(_square_cyclobutadiene(), 0.001)
    model.energy()
    result = model._last_result
    occupations = np.asarray(result.occupations)
    coefficients = np.asarray(result.mo_coeffs)
    # The Fermi level is solved to ~1e-12 in the electron count.
    assert occupations.sum() == pytest.approx(2.0 * result.n_occ, abs=1e-10)
    np.testing.assert_allclose(occupations[result.n_occ - 1:result.n_occ + 1], [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        coefficients @ np.diag(occupations) @ coefficients.T, np.asarray(result.density),
        atol=1e-12,
    )


def test_zero_temperature_occupations_are_exact_aufbau():
    """At T = 0 the stored occupations are exactly 2 and 0, so the gradient's
    energy-weighted density repeats the former Aufbau sum term by term."""
    model = SCCDFTBModel(_displaced_furan(), max_iter=500)
    model.energy()
    assert model.converged
    result = model._last_result
    occupations = np.asarray(result.occupations)
    expected = np.zeros_like(occupations)
    expected[: result.n_occ] = 2.0
    np.testing.assert_array_equal(occupations, expected)
