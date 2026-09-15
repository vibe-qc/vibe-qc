"""Coordinate representations for the uniform geometry optimization framework.

Every optimizer operates on a flat parameter vector in "optimizer
space".  The :class:`CoordinateRepresentation` translates between
that vector and Cartesian geometry, handling whatever internal
coordinate transforms are needed.

Phase 1 delivers :class:`CartesianCoordinates` -- the identity
transform.  Phase 4 will add :class:`DelocalizedInternalCoordinates`
(DLC) and :class:`RedundantInternals` behind the same abstraction
so optimizers never see the difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from .._vibeqc_core import Atom, Molecule

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CoordinateRepresentation(ABC):
    """Translates between Cartesian positions and an optimizer-space vector.

    Subclasses implement the forward (Cartesian -> optimiser) and backward
    (optimiser -> Cartesian) maps, plus optional projection of gradients
    and Hessians into the optimiser subspace.
    """

    @abstractmethod
    def x0(self, molecule: Molecule) -> np.ndarray:
        """Starting parameter vector from *molecule*."""
        ...

    @abstractmethod
    def to_cartesian(self, template: Molecule, x: np.ndarray) -> Molecule:
        """Rebuild a Molecule from the optimiser vector *x*.

        The *template* supplies atom identities, charge, and multiplicity.
        """
        ...

    def project_gradient(
        self, molecule: Molecule, cartesian_gradient: np.ndarray
    ) -> np.ndarray:
        """Project a Cartesian gradient into optimiser space.

        Default: identity (Cartesian <- Cartesian).
        """
        return np.asarray(cartesian_gradient, dtype=float).ravel()

    def project_hessian(
        self, molecule: Molecule, cartesian_hessian: np.ndarray
    ) -> np.ndarray:
        """Project a Cartesian Hessian into optimiser space.

        Default: identity (Cartesian <- Cartesian).
        """
        return np.asarray(cartesian_hessian, dtype=float)

    @property
    def n_params(self) -> int:
        """Number of optimiser parameters."""
        raise NotImplementedError

    def apply_frozen(
        self, x: np.ndarray, gradient: np.ndarray, frozen_set: set[int]
    ) -> np.ndarray:
        """Zero the gradient components that correspond to frozen atoms.

        The default implementation assumes 3 Cartesian components per
        atom.  Subclasses with different parameter topologies override
        this.
        """
        if not frozen_set:
            return gradient
        g = gradient.reshape(-1, 3).copy()
        for a in frozen_set:
            g[a, :] = 0.0
        return g.ravel()


# ---------------------------------------------------------------------------
# Cartesian (Phase 1 -- identity transform)
# ---------------------------------------------------------------------------


class CartesianCoordinates(CoordinateRepresentation):
    """Identity representation: optimiser space == Cartesian positions.

    This is the simplest coordinate system and the Phase 1 default.
    It has no internal-coordinate transforms, no B-matrix, and no
    back-transform failure modes -- making it the most robust choice
    for initial implementation and the fallback when internal
    coordinates fail.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    freeze_indices : sequence of int, optional
        Atom indices to hold fixed.
    """

    def __init__(
        self,
        n_atoms: int,
        freeze_indices: Optional[Sequence[int]] = None,
    ):
        self._n_atoms = n_atoms
        self._frozen_set: set[int] = (
            {int(i) for i in freeze_indices} if freeze_indices else set()
        )

    @property
    def n_params(self) -> int:
        return 3 * self._n_atoms

    def x0(self, molecule: Molecule) -> np.ndarray:
        flat: list[float] = []
        for atom in molecule.atoms:
            flat.extend(atom.xyz)
        return np.array(flat, dtype=float)

    def to_cartesian(self, template: Molecule, x: np.ndarray) -> Molecule:
        x = np.asarray(x, dtype=float).ravel()
        new_atoms: list[Atom] = []
        for i in range(self._n_atoms):
            xyz = [float(x[3 * i + c]) for c in range(3)]
            new_atoms.append(Atom(int(template.atoms[i].Z), xyz))
        return Molecule(new_atoms, template.charge, template.multiplicity)

    @property
    def frozen_set(self) -> set[int]:
        return self._frozen_set

    def apply_frozen(
        self, x: np.ndarray, gradient: np.ndarray, frozen_set: set[int]
    ) -> np.ndarray:
        """Zero gradient components on frozen atoms."""
        return super().apply_frozen(x, gradient, frozen_set)
