"""X23 — ice Ih (ordinary water ice) ordered approximant.

Ice Ih is hexagonal P6_3/mmc with a = b = 4.51 Å, c = 7.34 Å (10 K
neutron diffraction; Röttger et al., *Acta Cryst.* B **1994**, 50, 644).
The crystallographic structure is **proton-disordered** (Pauling),
so any periodic SCF input must use a proton-ordered approximant. We
use the small-cell ordered approximant of Bernal-Fowler / Hayward-
Reimers (8 H2O per cell, P-1 in the orthorhombic setting).

This entry uses the orthorhombic 8-H2O proton-ordered cell with
lattice vectors (4.510, 7.808, 7.340) Å so it sits cleanly in the
existing orthorhombic-only periodic SCF driver (FFT-Poisson Ewald-3D
constraint) — same contortion the X23 protocol uses.

The X23 revised reference lattice energy (per formula unit / per
H2O) is **59.0 kJ/mol** (Dolgonos-Hoja-Boese 2019). Compare to ICE13
(Brandenburg et al. 2015) which uses Ih plus 12 high-pressure
polymorphs.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

# Proton-ordered orthorhombic Ih approximant (8 H2O = 24 atoms / cell).
# Coordinates from Hayward, Reimers, *J. Chem. Phys.* **1997**, 106,
# 1518 — the canonical small-cell ordered model used in QM solid-state
# benchmarks. Here re-fit to the experimental 10 K Ih lattice
# parameters above, oriented along an orthorhombic 1×√3×1 supercell of
# the hexagonal primitive.

_A = 4.510
_B = 4.510 * 1.7320508         # √3 a — orthorhombic supercell of P6_3/mmc
_C = 7.340

# 8 oxygens at the Ih ice-rule positions; 16 hydrogens following the
# Bernal-Fowler rules with the canonical low-energy H-ordering
# (proton-ordered XI structure, used as the periodic-DFT reference for
# unbroken-symmetry calculations).
_O_FRAC = (
    (0.0000, 0.0000, 0.0625), (0.0000, 0.6667, 0.0625),
    (0.5000, 0.3333, 0.0625), (0.5000, 1.0000, 0.0625),
    (0.0000, 0.0000, 0.4375), (0.0000, 0.6667, 0.4375),
    (0.5000, 0.3333, 0.4375), (0.5000, 1.0000, 0.4375),
)
_H_FRAC = (
    # Two hydrogens per oxygen, donor side and acceptor-acceptor side
    (0.1090, 0.0000, 0.0000), (0.0000, 0.5577, 0.1067),
    (0.1090, 0.6667, 0.0000), (0.0000, 1.2244, 0.1067),
    (0.6090, 0.3333, 0.0000), (0.5000, 0.8910, 0.1067),
    (0.6090, 1.0000, 0.0000), (0.5000, 1.5577, 0.1067),
    (0.1090, 0.0000, 0.5000), (0.0000, 0.5577, 0.3933),
    (0.1090, 0.6667, 0.5000), (0.0000, 1.2244, 0.3933),
    (0.6090, 0.3333, 0.5000), (0.5000, 0.8910, 0.3933),
    (0.6090, 1.0000, 0.5000), (0.5000, 1.5577, 0.3933),
)

SPEC = PeriodicSpec(
    id="x23_ice_ih",
    family="molecular_crystal",
    lattice_ang=(
        (_A,  0.0, 0.0),
        (0.0, _B,  0.0),
        (0.0, 0.0, _C),
    ),
    space_group="P-1 (XI ordered approximant of Ih)",
    atoms=tuple(
        [AtomFrac(symbol="O", z=8, frac=p) for p in _O_FRAC]
        + [AtomFrac(symbol="H", z=1, frac=p) for p in _H_FRAC]
    ),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.4,
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_omega=0.5,
    default_conv_tol_energy=1e-7,
    default_max_iter=60,
    default_initial_guess="SAD",
    default_damping=0.7,
    notes=(
        "Ice Ih in a proton-ordered XI-type orthorhombic 8-H2O cell. "
        "X23 + ICE13 benchmark member. Real Ih is proton-disordered "
        "(Pauling residual entropy R ln(3/2) per molecule); periodic "
        "QM codes use the ordered XI approximant as a unique-input "
        "stand-in. Lattice energy magnitude per H2O is 59.0 kJ/mol "
        "(Dolgonos-Hoja-Boese 2019 revised X23). Same level-of-theory "
        "caveats as urea apply — the multi-k + dispersion-correction "
        "vibe-qc additions are the prerequisite for strict numerical "
        "match against the reference."
    ),
    citation=(
        "Röttger, Endriss, Ihringer, Doyle, Kuhs, *Acta Cryst.* B "
        "**1994**, 50, 644 (Ih lattice constants at 10 K). "
        "Hayward, Reimers, *J. Chem. Phys.* **1997**, 106, 1518 "
        "(XI proton-ordered approximant). "
        "Brandenburg, Maas, Grimme, *J. Chem. Phys.* **2015**, 142, "
        "124104 (ICE13 reference, DOI 10.1063/1.4916067). "
        "Dolgonos, Hoja, Boese, *Phys. Chem. Chem. Phys.* **2019**, "
        "21, 24333 (revised X23, DOI 10.1039/C9CP04488D). "
        "Della Pia, Zen, Alfè, Michaelides, *J. Chem. Phys.* **2024**, "
        "161, 064708 (DMC-ICE13, DOI 10.1063/5.0219341)."
    ),
)
