"""Implicit solvation (CPCM / COSMO) for molecular SCF.

vibe-qc v0.9.0 ships a polarisable-continuum solvation model in the
conductor variant (Cossi-Rega-Scalmani-Barone 2003; "C-PCM"). The
solute is enclosed in a molecular-shaped cavity whose boundary carries
an apparent surface charge ``q`` that screens the gas-phase
electrostatic potential by the dielectric factor
``f(e) = (e - 1)/e``. At convergence the surface-charge distribution
``q`` and the SCF density ``D`` solve

    A q = - f(e) . V(D, R_nuc)
    F(D) = H_core + 2J(D) - K(D) + V_q(D)

simultaneously, where ``A`` is the cavity self-interaction matrix,
``V(D, R_nuc)`` is the molecular electrostatic potential at the cavity
tessellation points, and ``V_q`` is the one-electron operator the
surface charges induce on the AO basis. The total solvation energy is

    E_solv = (1/2) S_i q_i V(R_nuc, D; s_i).

The v0.9.0 driver couples the two equations via the macro-iteration
loop standard in legacy PCM codes -- converged density -> ASC ->
modified Fock -> re-converged density -> next ASC. Three to five outer
cycles are typical for ΔE_solv < 1e-6 Hartree.

Public entry points
-------------------
``SolventModel``        Top-level solvation spec (cavity + dielectric).
``SOLVENT_PRESETS``     Dictionary of solvent name -> ``SolventModel``.
``build_cavity``        S1a -- Bondi / Lebedev tessellation.
``run_cpcm_scf``        S1b/d -- macro-iteration SCF in solvent.
``cpcm_gradient_fd``    S1c -- finite-difference solvation gradient
                        (analytic gradient is on the v0.9.1 roadmap).

For most users the right entry point is ``vibeqc.run_job(...,
solvent="water")``, which wires the driver through the standard SCF
output writer.

References
----------
* Klamt, A. & Schüürmann, G. *J. Chem. Soc. Perkin Trans. 2*, 799
  (1993) -- COSMO conductor model.
* Cossi, M., Rega, N., Scalmani, G. & Barone, V. *J. Comp. Chem.*
  24, 669 (2003) -- CPCM as the practical numerical variant.
* Scalmani, G. & Frisch, M. J. *J. Chem. Phys.* 132, 114110 (2010)
  -- continuous switching function (CSC) for smooth-cavity PCM.
* Tomasi, J., Mennucci, B. & Cammi, R. *Chem. Rev.* 105, 2999 (2005)
  -- review of the PCM family.
"""

from __future__ import annotations

from .cavity import BONDI_RADII_ANG, CavityTessellation, build_cavity
from .cpcm import (
    CPCMResult,
    build_A_matrix,
    build_cavity_A_matrix,
    solve_apparent_charges,
)
from .driver import (
    SolventModel,
    SolventResult,
    resolve_solvent,
    run_cpcm_scf,
)
from .gradient import cpcm_gradient, cpcm_gradient_fd
from .fine_cavity import FineCavity, build_fine_cavity
from .presets import SOLVENT_PRESETS
from .screening import ScreeningModel, dielectric_factor

__all__ = [
    "build_fine_cavity",
    "FineCavity",
    "ScreeningModel",
    "dielectric_factor",
    "BONDI_RADII_ANG",
    "CavityTessellation",
    "CPCMResult",
    "SOLVENT_PRESETS",
    "SolventModel",
    "SolventResult",
    "build_A_matrix",
    "build_cavity_A_matrix",
    "build_cavity",
    "cpcm_gradient",
    "cpcm_gradient_fd",
    "resolve_solvent",
    "run_cpcm_scf",
    "solve_apparent_charges",
]
