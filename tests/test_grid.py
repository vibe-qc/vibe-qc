"""Molecular integration grid: Treutler + Becke quadrature accuracy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import Atom, Molecule, GridOptions, build_grid


# --- single-atom radial accuracy tests ------------------------------------

def test_grid_integrates_hydrogen_1s_density_to_one():
    """∫ |ψ_1s|^2 d^3r = 1 for the exponential density ρ(r) = e^{-2r} / π
    (Z=1, exact 1s STO with ζ=1). Default 75-point radial + 17×36 angular
    should hit < 1e-10."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    g = build_grid(mol)
    r = np.linalg.norm(g.points, axis=1)
    rho = np.exp(-2.0 * r) / math.pi
    integral = (rho * g.weights).sum()
    assert integral == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n_radial,tol", [(50, 1e-9), (75, 1e-10), (120, 1e-12)])
def test_grid_radial_convergence(n_radial, tol):
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    opts = GridOptions()
    opts.n_radial = n_radial
    g = build_grid(mol, opts)
    r = np.linalg.norm(g.points, axis=1)
    rho = np.exp(-2.0 * r) / math.pi
    integral = (rho * g.weights).sum()
    assert abs(integral - 1.0) < tol


def test_grid_integrates_normalised_gaussian():
    """∫ (a/π)^{3/2} exp(-a r²) d^3r = 1 for any a > 0."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    g = build_grid(mol)
    r2 = (g.points ** 2).sum(axis=1)
    for a in (0.25, 1.0, 4.0):
        norm = (a / math.pi) ** 1.5
        rho = norm * np.exp(-a * r2)
        integral = (rho * g.weights).sum()
        assert abs(integral - 1.0) < 1e-8, (
            f"Gaussian a={a}: ∫ = {integral}, diff = {integral - 1.0}"
        )


# --- molecular (multi-atom, Becke partitioning) ---------------------------

def test_grid_becke_weights_sum_to_one_per_point():
    """For every grid point in a molecule, summing the atom-owner weights
    across all atoms (P_A(r) before we pick one) should give 1. We can't
    access that directly, but total molecular integral of ρ should equal
    the sum of per-atom integrals of ρ · P_A. A simpler proxy: ∫ ρ should
    still give the correct answer when ρ is the superposition of atomic
    densities on each atom."""
    # H2 molecule at R = 1.4 bohr; ρ = sum of two 1s H densities.
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 0.0, 1.4])])
    g = build_grid(mol)
    # ρ_total = (exp(-2|r|) + exp(-2|r - R|)) / π  — two normalized 1s clouds.
    r0 = np.linalg.norm(g.points - np.array([0, 0, 0.0]), axis=1)
    r1 = np.linalg.norm(g.points - np.array([0, 0, 1.4]), axis=1)
    rho = (np.exp(-2.0 * r0) + np.exp(-2.0 * r1)) / math.pi
    integral = (rho * g.weights).sum()
    # Ideal integral = 2 (one electron per H). Cross-penetration terms
    # don't change the integral because each normalized density alone
    # integrates to 1 over all space; our grid covers enough of each
    # atomic region to reproduce that to ~1e-6.
    assert abs(integral - 2.0) < 1e-5


def test_grid_options_reproducible():
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.5, -1.16]),
                    Atom(1, [0.0, -1.5, -1.16])])
    g1 = build_grid(mol)
    g2 = build_grid(mol)
    np.testing.assert_allclose(g1.points, g2.points, rtol=0, atol=0)
    np.testing.assert_allclose(g1.weights, g2.weights, rtol=0, atol=0)


def test_grid_atom_of_point_assigns_full_coverage():
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.5, -1.16]),
                    Atom(1, [0.0, -1.5, -1.16])])
    g = build_grid(mol)
    counts = np.bincount(np.asarray(g.atom_of_point, dtype=int))
    # Every atom owns the same number of points (radial × angular × n_atoms
    # structure), even if Becke weights reduce some of them to ~0.
    assert counts.size == 3
    assert counts.min() == counts.max()
