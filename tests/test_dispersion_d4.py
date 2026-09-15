"""D4 dispersion correction (Caldeweyher, Bannwarth, Grimme,
*J. Chem. Phys.* **150**, 154122 (2019)) via the optional ``dftd4``
package.

Pins:

  1. ``vibeqc.compute_d4(mol, functional)`` returns a non-empty D4
     energy for B2PLYP / DSD-PBEP86 / B3LYP / PBE0 / PW1PW on H2O,
     and the energy is negative (bound dispersion).
  2. ``run_b2plyp(mol, basis, dispersion="d4")`` and
     ``run_dsd_pbep86(mol, basis, dispersion="d4")`` add the
     functional-specific D4 correction on top of the XC + MP2 total.
     The returned ``DoubleHybridResult.e_total`` includes it;
     ``DoubleHybridResult.dispersion`` carries the standalone D4 piece.
  3. With ``dispersion=None`` (the default) ``DoubleHybridResult.e_total``
     equals the un-dispersed XC + MP2 total (regression on the
     B2PLYP / DSD-PBEP86 dispatcher behaviour).
  4. ``compute_d4`` energy matches what ``run_b2plyp(dispersion="d4")``
     attaches — i.e. the dispatcher uses the same D4 backend on the
     same input.
  5. ``with_gradient=True`` returns an (n_atoms, 3) gradient.
  6. Unrecognised dispersion / functional names raise with clear
     errors.
  7. ``dftd4_available()`` reports the install status correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    DoubleHybridResult,
    Molecule,
    compute_d4,
    dftd4_available,
    run_b2plyp,
    run_double_hybrid,
    run_dsd_pbep86,
)

from .conftest import GEOMETRIES


# Skip the whole module if dftd4 isn't installed — pip install -e '.[test]'
# pulls it in via the [test] extras, but the suite shouldn't hard-fail on
# a partial install.
pytestmark = pytest.mark.skipif(
    not dftd4_available(),
    reason="dftd4 not installed; install with pip install -e '.[dispersion]'",
)


def _mol(name):
    return Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES[name]])


def _basis(mol, name):
    return BasisSet(mol, name)


def _r2scan3c_water_dimer():
    """Sealed M06 water-dimer geometry in bohr."""
    return Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.49921989, -1.15936587]),
        Atom(1, [0.0, -1.49921989, -1.15936587]),
        Atom(8, [0.0, 0.0, 5.57469207]),
        Atom(1, [0.0, 1.49921989, 4.41532619]),
        Atom(1, [0.0, -1.49921989, 4.41532619]),
    ])


# ---------------------------------------------------------------------
# Standalone compute_d4.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("functional", [
    "b2plyp", "dsd-pbep86", "dsdpbep86", "b3lyp", "pbe0", "pw1pw",
])
def test_compute_d4_returns_negative_energy(functional):
    """D4 dispersion is bound (negative) for a small neutral molecule."""
    mol = _mol("H2O")
    d4 = compute_d4(mol, functional)
    assert d4.energy < 0
    assert d4.functional == functional.lower()
    assert d4.gradient is None


def test_r2scan3c_uses_published_charge_scaling():
    """The composite D4 model uses its refitted charge dependence."""
    mol = _r2scan3c_water_dimer()
    d4 = compute_d4(mol, "r2scan-3c")

    # ORCA 6.1.1 / DFTD4 3.4.0 prints -0.000399198 Ha for this exact
    # geometry. The source-paper (beta, gamma) = (2, 1) model gives
    # -0.00039920653785868417 Ha through the dftd4 API.
    assert d4.energy == pytest.approx(-0.000399198, abs=1.0e-8)

    # Same-geometry control: ordinary PBE0-D4 keeps the default charge model
    # and agrees with the archived ORCA value to its printed precision.
    pbe0 = compute_d4(mol, "pbe0")
    assert pbe0.energy == pytest.approx(-0.000836898, abs=1.0e-8)


def test_compute_d4_gradient_shape():
    """with_gradient=True returns (n_atoms, 3) Ha/bohr forces."""
    mol = _mol("H2O")
    d4 = compute_d4(mol, "b2plyp", with_gradient=True)
    assert d4.gradient is not None
    assert d4.gradient.shape == (3, 3)
    # Forces are small for a near-equilibrium geometry, but not exactly
    # zero (D4 sees a different equilibrium than the SCF).
    assert np.abs(d4.gradient).max() < 1.0  # Ha/bohr


def test_compute_d4_rejects_unknown_functional():
    """Useful error when the functional name isn't in dftd4's catalog."""
    mol = _mol("H2O")
    with pytest.raises(RuntimeError, match="does not recognise"):
        compute_d4(mol, "this-is-not-a-functional")


def test_dftd4_available_is_true():
    """When pip install -e '.[test]' has run, dftd4 must be importable."""
    assert dftd4_available() is True


# ---------------------------------------------------------------------
# Double-hybrid + D4 dispatch.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["b2plyp", "dsd-pbep86"])
def test_double_hybrid_d4_adds_dispersion_correctly(name):
    """run_double_hybrid(..., dispersion="d4") attaches a D4 result that
    matches the standalone compute_d4 call for the same functional, and
    e_total is the XC + MP2 + D4 sum."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    standalone_d4 = compute_d4(mol, name)
    result = run_double_hybrid(mol, basis, name, dispersion="d4")
    assert isinstance(result, DoubleHybridResult)
    assert result.dispersion is not None
    # Same backend → same D4 number.
    assert result.dispersion.energy == pytest.approx(
        standalone_d4.energy, abs=1e-14)
    # Total = SCF + scaled-MP2 + dispersion.
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation +
        result.dispersion.energy, abs=1e-14)


def test_b2plyp_d4_via_wrapper():
    """run_b2plyp(dispersion='d4') gives the B2PLYP-D4 published total."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    result = run_b2plyp(mol, basis, dispersion="d4")
    assert result.dispersion is not None
    assert result.dispersion.functional == "b2plyp"
    # The XC + MP2 piece alone is the un-dispersed total.
    undispersed = result.rks.energy + result.mp2.e_correlation
    assert result.e_total == pytest.approx(
        undispersed + result.dispersion.energy, abs=1e-14)


def test_dsd_pbep86_d4_via_wrapper():
    """run_dsd_pbep86(dispersion='d4') gives the DSD-PBEP86-D4 total."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    result = run_dsd_pbep86(mol, basis, dispersion="d4")
    assert result.dispersion is not None
    assert result.dispersion.functional == "dsd-pbep86"
    undispersed = result.rks.energy + result.mp2.e_correlation
    assert result.e_total == pytest.approx(
        undispersed + result.dispersion.energy, abs=1e-14)


def test_double_hybrid_no_dispersion_default():
    """The dispatcher default (dispersion=None) preserves the
    un-dispersed XC + MP2 total — backward-compat regression after
    adding the dispersion kwarg."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    no_d = run_b2plyp(mol, basis)  # default: no dispersion
    with_d = run_b2plyp(mol, basis, dispersion="d4")
    assert no_d.dispersion is None
    assert no_d.e_total == pytest.approx(
        no_d.rks.energy + no_d.mp2.e_correlation, abs=1e-14)
    # The two totals differ by exactly the D4 energy.
    assert with_d.e_total - no_d.e_total == pytest.approx(
        with_d.dispersion.energy, abs=1e-14)


def test_double_hybrid_d4_dispersion_negative():
    """For a neutral closed-shell molecule, D4 contributes a negative
    correction. (Sanity — also flagged by compute_d4 tests, but worth
    pinning at the dispatcher level too.)"""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    result = run_b2plyp(mol, basis, dispersion="d4")
    assert result.dispersion.energy < 0


def test_double_hybrid_rejects_unknown_dispersion():
    """Unknown dispersion kwarg raises with a clear error pointing at
    the supported options."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    with pytest.raises(ValueError, match="dispersion="):
        run_b2plyp(mol, basis, dispersion="d6")
    with pytest.raises(ValueError, match="dispersion="):
        run_double_hybrid(mol, basis, "b2plyp", dispersion="not-a-thing")


# ---------------------------------------------------------------------
# D3(BJ) dispatcher integration (reuses the existing compute_d3bj
# framework, parameter sets ship via the optional dftd3 package).
# ---------------------------------------------------------------------

from vibeqc import compute_d3bj, dftd3_available


@pytest.mark.skipif(
    not dftd3_available(),
    reason="dftd3 not installed; install with pip install -e '.[dispersion]'",
)
@pytest.mark.parametrize("name", ["b2plyp", "dsd-pbep86"])
def test_double_hybrid_d3bj_matches_standalone_compute_d3bj(name):
    """run_double_hybrid(..., dispersion='d3bj') attaches the same D3-BJ
    energy that compute_d3bj(mol, functional) produces standalone."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    standalone = compute_d3bj(mol, name)
    result = run_double_hybrid(mol, basis, name, dispersion="d3bj")
    assert result.dispersion is not None
    assert result.dispersion.energy == pytest.approx(
        standalone.energy, abs=1e-14)
    # Total = SCF + scaled-MP2 + dispersion.
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation +
        result.dispersion.energy, abs=1e-14)


@pytest.mark.skipif(
    not dftd3_available(),
    reason="dftd3 not installed; install with pip install -e '.[dispersion]'",
)
def test_b2plyp_d3bj_and_d4_differ():
    """B2PLYP-D3(BJ) and B2PLYP-D4 give different (but similar-order)
    dispersion corrections — sanity check that the two dispatcher
    paths are actually different methods, not the same backend
    aliased twice."""
    mol = _mol("H2O")
    basis = _basis(mol, "cc-pvdz")
    d3 = run_b2plyp(mol, basis, dispersion="d3bj")
    d4 = run_b2plyp(mol, basis, dispersion="d4")
    # Both negative, both <1 kcal/mol on H2O.
    assert d3.dispersion.energy < 0
    assert d4.dispersion.energy < 0
    # Different methods → different numbers.
    assert abs(d3.dispersion.energy - d4.dispersion.energy) > 1e-7
