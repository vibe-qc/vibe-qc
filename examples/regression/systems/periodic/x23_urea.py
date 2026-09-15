"""X23 — urea crystal, P-421m tetragonal (member of the X23 set).

Geometry: Swaminathan, Craven & McMullan, *Acta Cryst.* B **1984**, 40,
300 — neutron diffraction at 12 K (CCDC URECRA08 / ICSD #29585).
Tetragonal P-421m with a = b = 5.565 Å, c = 4.684 Å, two formula units
per cell. Atoms placed at the standard X23 reference positions.

The X23 lattice-energy benchmark uses experimental sublimation
enthalpy back-corrected to 0 K with phonon ZPE; the **revised** X23
reference for urea (Dolgonos, Hoja, Boese 2019) is **102.5 kJ/mol per
formula unit** (cohesive energy magnitude).
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

# Urea has 2 formula units per tetragonal cell (Z = 2). Z=8 atoms per
# molecule × 2 = 16 atoms total. Fractional coords from CCDC URECRA08
# normalised to the conventional setting.
SPEC = PeriodicSpec(
    id="x23_urea",
    family="molecular_crystal",
    lattice_ang=(
        (5.565, 0.000, 0.000),
        (0.000, 5.565, 0.000),
        (0.000, 0.000, 4.684),
    ),
    space_group="P-421m",
    atoms=(
        # Molecule 1
        AtomFrac(symbol="O", z=8, frac=(0.000, 0.500, 0.5953)),
        AtomFrac(symbol="C", z=6, frac=(0.000, 0.500, 0.3289)),
        AtomFrac(symbol="N", z=7, frac=(0.1429, 0.6429, 0.1825)),
        AtomFrac(symbol="N", z=7, frac=(0.8571, 0.3571, 0.1825)),
        AtomFrac(symbol="H", z=1, frac=(0.2570, 0.7570, 0.2840)),
        AtomFrac(symbol="H", z=1, frac=(0.7430, 0.2430, 0.2840)),
        AtomFrac(symbol="H", z=1, frac=(0.1430, 0.6430, -0.0287)),
        AtomFrac(symbol="H", z=1, frac=(0.8570, 0.3570, -0.0287)),
        # Molecule 2 (rotated 90°, related by P-421m symmetry)
        AtomFrac(symbol="O", z=8, frac=(0.500, 0.000, 0.4047)),
        AtomFrac(symbol="C", z=6, frac=(0.500, 0.000, 0.6711)),
        AtomFrac(symbol="N", z=7, frac=(0.6429, 0.1429, 0.8175)),
        AtomFrac(symbol="N", z=7, frac=(0.3571, 0.8571, 0.8175)),
        AtomFrac(symbol="H", z=1, frac=(0.7570, 0.2570, 0.7160)),
        AtomFrac(symbol="H", z=1, frac=(0.2430, 0.7430, 0.7160)),
        AtomFrac(symbol="H", z=1, frac=(0.6430, 0.1430, 1.0287)),
        AtomFrac(symbol="H", z=1, frac=(0.3570, 0.8570, 1.0287)),
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
        "Urea molecular crystal, X23 member. Tetragonal P-421m, Z=2 "
        "(16 atoms / cell). Strong N-H···O=C hydrogen-bond network "
        "binds the layers — the canonical X23 'molecular crystal "
        "with strong directional H-bonds' test case. Lattice energy "
        "magnitude 102.5 kJ/mol per formula unit (Dolgonos-Hoja-Boese "
        "2019 revised reference). vibe-qc periodic Γ-only RKS-LDA "
        "result is advisory until multi-k + dispersion correction "
        "land — the bare LDA underbinds molecular crystals by "
        "20-50 %, so the parity test is vs PySCF.pbc at matching "
        "settings, not vs the published lattice energy."
    ),
    citation=(
        "Swaminathan, Craven, McMullan, *Acta Cryst.* B **1984**, 40, "
        "300 (urea structure, CCDC URECRA08 / ICSD 29585). "
        "Reilly, Tkatchenko, *J. Chem. Phys.* **2013**, 139, 024705 "
        "(X23 origin, DOI 10.1063/1.4812819). "
        "Dolgonos, Hoja, Boese, *Phys. Chem. Chem. Phys.* **2019**, "
        "21, 24333 (revised X23, DOI 10.1039/C9CP04488D)."
    ),
)
