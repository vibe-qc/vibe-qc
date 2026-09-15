"""Production acceptance matrix for semiempirical methods.

Defines the minimum validation bar each method must meet before graduating
from "experimental" to "production-validated".  This is a living document;
update it as tests are added and validation gaps are closed.

Current status (2026-08-21)
----------------------------
- DFTB0/SCC-DFTB: screening/preopt only (in-house parameters, no fitted repulsives)
- GFN2-xTB: gated experimental (molecular AES + GAM3 + post-SCF native
  D4-style dispersion landed for the native D4 reference-data set;
  Gamma-periodic derivatives are FD-validated, while faithful periodic AES
  electrostatics, broader molecular-limit coverage, mixer tuning, and external
  parity gates remain)
- PM6/UPM6: native molecular development route; H-only and H-heavy s/p
  interactions are source-correct, but the full heavy-heavy two-center tensor
  and exact MOPAC heat-of-formation convention remain open
- PM7/UPM7: gated; published feathered electrostatics is not implemented
- OM2/OM3: validated molecular prescreening with published relative-energy fixtures;
  OM1 remains experimental because its analytic core-valence ECP is missing;
  Bloch-periodic OMx is gated

Reference oracles (out-of-process, CLAUDE.md §10)
--------------------------------------------------
- ``examples/regression/core/runner_xtb.py`` — xTB GFN2-xTB
- ``examples/regression/core/runner_mopac.py`` — MOPAC PM6
- ``examples/regression/core/runner_dftbp.py`` — DFTB+ (mio-1-1)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# Acceptance criteria per method
# ═══════════════════════════════════════════════════════════════════════════

ACCEPTANCE = {
    "dftb0": {
        "status": "screening_preopt",
        "validated": False,
        "criteria": {
            "small_molecules": ["H2", "H2O"],
            "atoms": [],
            "periodic": ["H chain", "C chain"],
            "gradients_fd": True,
            "size_consistency": False,
            "rotation_invariance": False,
            "external_parity": "DFTB+ (mio-1-1)",
        },
        "gating_issues": [
            "In-house parameters, no DFT-fitted repulsives",
            "Positive energies for CH4, NH3 with default R⁻¹² repulsives",
            "No external validation against DFTB+",
        ],
    },
    "scc_dftb": {
        "status": "screening_preopt",
        "validated": False,
        "criteria": {
            "small_molecules": ["H2", "H2O"],
            "atoms": [],
            "periodic": ["H chain", "C chain"],
            "gradients_fd": True,
            "size_consistency": False,
            "rotation_invariance": False,
            "external_parity": "DFTB+ (mio-1-1)",
        },
        "gating_issues": [
            "In-house parameters, no DFT-fitted repulsives",
            "SCC gradients are fixed-charge approximation (~0.4 Ha/bohr FD gap)",
        ],
    },
    "gfn2_xtb": {
        "status": "experimental_gated",
        "validated": False,
        "criteria": {
            "small_molecules": ["H2", "H2O", "CH4", "NH3", "CO2", "C2H4", "benzene"],
            "atoms": ["H", "He", "C", "N", "O", "F", "Ne", "Ar"],
            "ions_radicals": ["OH", "CH3"],
            "periodic": ["H chain", "C chain", "graphene_2c", "MgO rocksalt"],
            "molecular_limit": True,
            "gradients_fd": True,
            "size_consistency": True,
            "rotation_invariance": False,
            "charge_signs": True,
            "symmetry_equivalent_charges": True,
            "convergence_behavior": True,
            "external_parity": "xTB v6.7.1 GFN2",
        },
        "gating_issues": [
            (
                "Native post-SCF D4 is limited to H, He, B, C, N, O, F, Ne "
                "and returns zero with GFN2D4UnsupportedWarning outside that set"
            ),
            "Periodic AES image-cell multipole Ewald still pending",
            (
                "Periodic molecular-limit parity is strict only for the pinned "
                "H2O fixture so far"
            ),
            "DIIS/Broyden mixer convergence still needs polar-molecule tuning",
            "External xTB parity matrix not closed across molecules, ions, radicals, and periodic cells",
            "C2H4 / polar SCC convergence remains a tuning gate for non-simple mixers",
        ],
    },
    "pm6": {
        "status": "development_heavy_heavy_tensor_open",
        "validated": False,
        "criteria": {
            "small_molecules": ["H2", "H2O", "CH4", "NH3", "HF"],
            "atoms": ["H", "C", "N", "O", "F"],
            "ions_radicals": ["OH"],
            "periodic": ["H chain", "C chain", "MgO rocksalt"],
            "molecular_limit": True,
            "gradients_fd": True,
            "size_consistency": False,
            "rotation_invariance": False,
            "heat_of_formation_convention": False,
            "external_parity": "MOPAC 2016 PM6 heat-of-formation convention, open",
        },
        "gating_issues": [
            (
                "Reports a PM6-like total energy, not a MOPAC heat of "
                "formation; exact MOPAC convention parity remains open"
            ),
            (
                "Falls back to spherical Klopman-Ohno gamma/core-core terms "
                "where the bundled MOPAC diatomic multipole data is incomplete"
            ),
            (
                "The molecular heavy-heavy two-center block does not yet "
                "carry the full PM6 NDDO multipole tensor"
            ),
            (
                "Periodic PM6 uses native batched finite differences, but "
                "nonzero-image H-X exchange and full heavy-heavy multipoles "
                "remain gated"
            ),
        ],
    },
    "pm7": {
        "status": "gated_incomplete_hamiltonian",
        "validated": False,
        "criteria": {
            "parameter_registry": "75 bundled chemical-element records",
            "energy": False,
            "gradients_fd": False,
            "periodic": [],
            "external_parity": "Stewart 2013 PM7, not yet executable",
        },
        "gating_issues": [
            (
                "Stewart's smooth exact-electrostatic/point-charge feathering "
                "is not implemented across electron-electron, electron-core, "
                "and nuclear-nuclear channels"
            ),
            "PM7/UPM7 and Bloch-periodic PM7 fail closed before parameters",
        ],
    },
    "omx": {
        "status": (
            "OM2/OM3 validated molecular prescreening; OM1 experimental warning; "
            "Bloch-periodic gated"
        ),
        "validated": False,
        "criteria": {
            "small_molecules": ["H2", "H2O", "CH4", "NH3", "H3-", "C2H6"],
            "atoms": [],
            "periodic": [],
            "gradients_fd": True,
            "published_relative_energies": ["H3- bend", "ethane barrier"],
            "external_parity": "Published OM1/OM2/OM3 references (Dral 2016)",
        },
        "gating_issues": [
            "OM1 analytic core-valence ECP from Kolb-Thiel 1993 is missing",
            "OMx bond minima remain about 0.1-0.3 A from published references",
            (
                "Bloch-periodic OMx requires a full image-resolved published "
                "ORT/ECP/penetration Hamiltonian"
            ),
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Test cases defined but NOT yet implemented
# ═══════════════════════════════════════════════════════════════════════════

PENDING_TESTS = {
    "small_molecules": {
        "H2": "bond energy, charge=0, gradient vs FD",
        "H2O": "total energy, charge signs, dipole, gradient vs FD",
        "CH4": "total energy, symmetry, gradient vs FD",
        "NH3": "total energy, inversion barrier, gradient vs FD",
        "CO2": "total energy, linearity, gradient vs FD",
        "C2H4": "total energy, planarity, gradient vs FD",
        "benzene": "total energy, D6h symmetry, gradient vs FD",
    },
    "atoms": {
        "H": "total energy (unrestricted, doublet)",
        "He": "total energy (closed shell)",
        "C": "total energy (unrestricted, triplet)",
        "N": "total energy (unrestricted, quartet)",
        "O": "total energy (unrestricted, triplet)",
        "F": "total energy (unrestricted, doublet)",
        "Ne": "total energy (closed shell, valence only)",
        "Ar": "total energy (closed shell, valence only)",
        "Fe": "total energy (unrestricted, transition metal smoke test)",
        "Cu": "total energy (unrestricted, transition metal smoke test)",
    },
    "ions_radicals": {
        "OH": "doublet, charge=0, gradient vs FD",
        "CH3": "doublet, charge=0, gradient vs FD",
        "H3O+": "cation, charge=+1",
        "OH-": "anion, charge=-1",
    },
    "periodic": {
        "H_chain": "1D, a=3.0 bohr, 1 atom/cell, energy and gradient",
        "C_chain": "1D, a=2.5 bohr, 1 atom/cell, energy",
        "graphene_2c": "2D, 2 atoms/cell, energy",
        "MgO_rocksalt": "3D, 2 atoms/cell, ionic limit",
        "molecular_limit": "large cell, energy matches isolated molecule",
    },
    "invariants": {
        "size_consistency": "2×E(far molecules) = E(dimer at ∞)",
        "rotation_invariance": "rotated molecule gives same energy",
        "translation_invariance": "translated molecule gives same energy",
        "charge_conservation": "Σ q_i = total charge",
        "symmetry": "symmetry-equivalent atoms have equal charges",
        "newton3": "Σ F_i = 0 (or antisymmetric for 2-atom cells)",
    },
}
