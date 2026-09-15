"""The solute<->cavity coupling seam for implicit solvation (CPCM / COSMO).

Everything in an apparent-surface-charge (ASC) solvation model is
reference-independent *except* two arrows between the solute and the cavity:

1. **solute density -> electrostatic potential at the surface points**
   ``V_elec(s_i)`` -- what polarizes the dielectric, and
2. **surface apparent charges -> the operator added to the Fock matrix**
   ``V_q`` -- how the reaction field acts back on the electrons.

The cavity tessellation (:mod:`vibeqc.solvation.cavity`), the segment
interaction A-matrix + dielectric screening + ASC solve
(:mod:`vibeqc.solvation.cpcm`), the polarization energy, and the SCF
macro-iteration all sit *outside* those two arrows.  Factoring the arrows
behind :class:`SolutePotentialProvider` lets one reaction-field engine serve
every reference:

* **Gaussian-basis HF / DFT** -- :class:`~vibeqc.solvation.driver.GaussianESPProvider`,
  the ESP-on-grid coupling (<mu|1/|r-s_i||ν> one-electron operators); and
* **semiempirical engines (MSINDO)** -- an atom-centred multipole->segment
  coupling (a follow-up; see ``docs/user_guide/msindo.md``).

This is the solvation analogue of the :class:`vibeqc.correlation.ERIProvider`
seam: build it once, and CPCM/COSMO works for any method.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SolutePotentialProvider(Protocol):
    """Method-specific coupling between a solute density and the cavity.

    A provider is constructed for a fixed geometry + cavity (it caches whatever
    integrals / geometry it needs) and exposes the two arrows the reaction-field
    engine needs.  Sign convention (shared with the Gaussian implementation):
    ``esp_at_cavity`` returns the *electronic* potential
    ``V_elec(s_i) = -∫ r(r)/|r-s_i| dr`` (typically negative), and
    ``fock_contribution`` returns the operator to **add** to the core
    Hamiltonian for apparent charges ``q`` (i.e. ``-S_i q_i <.|1/|r-s_i||.>``).
    """

    def esp_at_cavity(self, density: np.ndarray) -> np.ndarray:
        """Electronic ESP at each cavity surface point from the density."""
        ...

    def fock_contribution(self, charges: np.ndarray) -> np.ndarray:
        """Operator added to the core Hamiltonian for the given ASC vector."""
        ...


__all__ = ["SolutePotentialProvider"]
