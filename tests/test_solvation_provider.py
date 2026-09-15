"""The SolutePotentialProvider seam (vibeqc.solvation.provider).

CPCM/COSMO is reference-independent except two arrows (density→surface ESP,
charges→Fock operator).  These tests pin the seam: the Gaussian provider
satisfies the Protocol and reproduces the exact coupling run_cpcm_scf has always
used, and the Protocol is implementable by an arbitrary (non-Gaussian) reference
— the property MSINDO's multipole provider will rely on.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions, run_rhf)
from vibeqc.solvation.cavity import build_cavity
from vibeqc.solvation.driver import (GaussianESPProvider,
                                     _build_point_charge_operators,
                                     _density_potential_at_cavity,
                                     _fock_solvent_contribution)
from vibeqc.solvation.provider import SolutePotentialProvider

_A2B = 1.8897259886


def _h2o_density_and_cavity():
    mol = Molecule([Atom(8, [0, 0, 0]),
                    Atom(1, [0, 0.7572 * _A2B, 0.5865 * _A2B]),
                    Atom(1, [0, -0.7572 * _A2B, 0.5865 * _A2B])], 0, 1)
    bas = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, bas, RHFOptions())
    D = np.asarray(rhf.density)
    pos = np.array([[0, 0, 0],
                    [0, 0.7572 * _A2B, 0.5865 * _A2B],
                    [0, -0.7572 * _A2B, 0.5865 * _A2B]])
    cav = build_cavity(pos, [8, 1, 1], n_points_per_sphere=110)
    return mol, bas, D, cav


def test_gaussian_provider_satisfies_protocol():
    """GaussianESPProvider is a structural SolutePotentialProvider."""
    _, bas, _, cav = _h2o_density_and_cavity()
    prov = GaussianESPProvider(bas, cav.points)
    assert isinstance(prov, SolutePotentialProvider)


def test_gaussian_provider_reproduces_free_function_coupling():
    """The provider's two arrows equal the free-function coupling run_cpcm_scf
    used directly before the seam (so the refactor is behavior-preserving)."""
    _, bas, D, cav = _h2o_density_and_cavity()
    M = _build_point_charge_operators(bas, cav.points)
    prov = GaussianESPProvider(bas, cav.points)

    # Arrow 1: density -> ESP at the surface.
    np.testing.assert_allclose(prov.esp_at_cavity(D),
                               _density_potential_at_cavity(D, M), rtol=0, atol=0)
    # Arrow 2: apparent charges -> Fock operator.
    q = np.linspace(-0.01, 0.01, cav.points.shape[0])
    np.testing.assert_allclose(prov.fock_contribution(q),
                               _fock_solvent_contribution(q, M), rtol=0, atol=0)
    # The cached operator stack has the expected shape.
    assert prov.operators.shape == (cav.points.shape[0], bas.nbasis, bas.nbasis)


def test_seam_is_implementable_by_a_nongaussian_reference():
    """Any reference can implement the seam — a minimal monopole-style provider
    (atom-centred point charges, no Gaussian integrals) satisfies the Protocol
    and returns correctly-shaped arrows.  This is the contract MSINDO's
    multipole→segment provider will fulfil."""
    _, _, _, cav = _h2o_density_and_cavity()
    n_pts = cav.points.shape[0]

    class _MonopoleProvider:
        """Toy: atom charges -> ESP via 1/r; charges -> per-atom potential."""
        def __init__(self, atom_pos, points):
            self.atom_pos = np.asarray(atom_pos)
            self.points = np.asarray(points)
            self._d = np.linalg.norm(
                self.atom_pos[:, None, :] - self.points[None, :, :], axis=2)

        def esp_at_cavity(self, charges):           # charges: per atom
            return -(np.asarray(charges)[:, None] / self._d).sum(axis=0)

        def fock_contribution(self, q):             # -> per-atom potential
            return -(np.asarray(q)[None, :] / self._d).sum(axis=1)

    prov = _MonopoleProvider(cav.atom_positions, cav.points)
    assert isinstance(prov, SolutePotentialProvider)
    esp = prov.esp_at_cavity(np.array([6.0, 1.0, 1.0]))
    assert esp.shape == (n_pts,)
    assert np.all(esp < 0.0)                         # positive charges → ESP<0 (sign conv.)
    pot = prov.fock_contribution(np.full(n_pts, 0.001))
    assert pot.shape == (3,)                          # one per atom
