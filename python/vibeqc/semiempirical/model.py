"""Abstract base class for semiempirical models.

Defines the contract that all semiempirical methods (DFTB0, SCC-DFTB,
GFN2-xTB, ...) must satisfy. Each concrete model provides ``energy()``,
``gradient()``, and optional ``stress()``.

Stage 1: DFTB0 implementation; gradient is numerical finite-difference.
Stage 2: analytic gradients replace finite-difference.
Stage 6: stress tensor added for periodic cell optimization.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from vibeqc._vibeqc_core import Atom as _Atom
from vibeqc._vibeqc_core import Molecule as Molecule


class SemiempiricalModel(abc.ABC):
    """Abstract base for a semiempirical energy/gradient model.

    Each concrete subclass must implement ``_compute_energy()``.
    The ``gradient()`` default uses central finite differences.
    Analytic gradients can override ``gradient()`` directly.

    Parameters
    ----------
    mol : Molecule
        The molecular system (positions in bohr).
    solver : str
        Eigensolver for the SCF step (``"dense"``, ``"davidson"``,
        ``"lanczos"``).  Default ``"dense"``.  Currently a reserved
        keyword: the semiempirical C++ backends use dense
        ``GeneralizedSelfAdjointEigenSolver`` unconditionally;
        iterative solvers require a future Löwdin-orthogonalisation
        bridge in the C++ layer.
    """

    def __init__(self, mol: Molecule, *, solver: str = "dense"):
        self._mol = mol
        self._solver = solver

    @property
    def molecule(self) -> Molecule:
        """The current geometry (bohr)."""
        return self._mol

    @property
    def solver(self) -> str:
        """Eigensolver selection (``"dense"``, ``"davidson"``, ``"lanczos"``)."""
        return self._solver

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def energy(self) -> float:
        """Return the total energy (Hartree) at the current geometry."""

    def gradient(self) -> np.ndarray:
        """Nuclear gradient dE/dR (Hartree/bohr), shape (n_atoms, 3).

        Default implementation: central finite differences with
        h = 0.001 bohr. Override for analytic gradients.
        """
        h = 0.001
        e0 = self.energy()
        grad = np.zeros((len(self._mol.atoms), 3))
        atoms = list(self._mol.atoms)

        for i in range(len(atoms)):
            for c in range(3):
                # +h
                xyz_plus = list(atoms[i].xyz)
                xyz_plus[c] += h
                atoms_plus = list(atoms)
                atoms_plus[i] = _Atom(atoms[i].Z, xyz_plus)
                mol_plus = Molecule(
                    atoms_plus, self._mol.charge, self._mol.multiplicity
                )
                e_plus = self._energy_at(mol_plus)

                # -h
                xyz_minus = list(atoms[i].xyz)
                xyz_minus[c] -= h
                atoms_minus = list(atoms)
                atoms_minus[i] = _Atom(atoms[i].Z, xyz_minus)
                mol_minus = Molecule(
                    atoms_minus, self._mol.charge, self._mol.multiplicity
                )
                e_minus = self._energy_at(mol_minus)

                grad[i, c] = (e_plus - e_minus) / (2.0 * h)

        return grad

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _energy_at(self, mol: Molecule) -> float:
        """Compute energy for a geometry specified by ``mol``.

        Subclasses implement this to decouple the finite-difference
        displacement from the energy evaluation.
        """
