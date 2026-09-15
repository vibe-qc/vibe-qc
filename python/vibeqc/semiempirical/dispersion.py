"""Dispersion corrections for semiempirical models.

Stage 9: Adds D3(BJ) and D4 dispersion as post-SCF corrections
to DFTB0 and SCC-DFTB energies and gradients.

Uses vibe-qc's existing D3(BJ) and D4 infrastructure from
``vibeqc.dispersion`` and ``vibeqc.dispersion_d4``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from vibeqc import Molecule
from vibeqc.dispersion import D3BJParams, compute_d3bj

# Default D3(BJ) parameters for DFTB-like methods.
# These are approximate -- DFTB-optimized D3 parameters would require
# fitting to reference data. For screening/preoptimization, PBE-like
# parameters provide a reasonable ballpark.
DFTB_D3_DEFAULTS = D3BJParams(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)
"""Default D3(BJ) parameters for DFTB/SCC-DFTB (PBE-like)."""


def d3bj_energy(mol: Molecule, params: Optional[D3BJParams] = None) -> float:
    """D3(BJ) dispersion energy for a molecule (Hartree)."""
    if params is None:
        params = DFTB_D3_DEFAULTS
    result = compute_d3bj(mol, params, with_gradient=False)
    return float(result.energy)


def d3bj_energy_gradient(
    mol: Molecule, params: Optional[D3BJParams] = None
) -> tuple[float, np.ndarray]:
    """D3(BJ) dispersion energy and gradient.

    Returns (energy_Hartree, gradient_Hartree_per_bohr).
    """
    if params is None:
        params = DFTB_D3_DEFAULTS
    result = compute_d3bj(mol, params, with_gradient=True)
    return float(result.energy), np.asarray(result.gradient)


class DispersionCorrectedModel:
    """Mixin that adds D3(BJ) dispersion to any SemiempiricalModel.

    Usage::

        model = DFTB0Model(mol)
        disp_model = DispersionCorrectedModel(model, d3_params)

    The dispersion energy and gradient are added post-SCF.
    """

    def __init__(
        self,
        base_model,
        d3_params: Optional[D3BJParams] = None,
    ):
        self._base = base_model
        self._d3_params = d3_params if d3_params is not None else DFTB_D3_DEFAULTS

    @property
    def molecule(self):
        return self._base.molecule

    def energy(self) -> float:
        e_dftb = self._base.energy()
        e_disp = d3bj_energy(self._base._mol, self._d3_params)
        return e_dftb + e_disp

    def gradient(self) -> np.ndarray:
        g_dftb = self._base.gradient()
        _, g_disp = d3bj_energy_gradient(self._base._mol, self._d3_params)
        return g_dftb + g_disp

    def _energy_at(self, mol: Molecule) -> float:
        # Build a temporary model for the displaced geometry
        # (needed for FD gradient validation)
        from vibeqc.semiempirical.dftb0 import DFTB0Model

        e_base = self._base._energy_at(mol)
        e_disp = d3bj_energy(mol, self._d3_params)
        return e_base + e_disp
