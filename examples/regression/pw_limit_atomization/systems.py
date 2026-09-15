"""Solid + free-atom definitions for the GPAW plane-wave-limit atomization
reference set.

Geometries are the r2SCAN-optimised lattice constants taken from the CRYSTAL
``.d12`` inputs of the pob benchmark (``references-cohesive/<compound>/``); the
VASP@900 eV r2SCAN atomization energies are the paper's current PW reference
(``references-cohesive/r2scan.dat``, kJ/mol per formula unit). Atomization
energy is flat in the lattice constant near the minimum, so these fixed
geometries are faithful for the energy comparison.

Free-atom spin multiplicities follow the Hund ground state; ``magmom`` is the
number of unpaired electrons (= 2S = multiplicity - 1), mirroring
``vibeqc.atomization._GROUND_STATE_MULTIPLICITY`` so the periodic reference uses
the same atomic reference states as vibe-qc's molecular atomization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Unpaired electrons in the Hund ground state, per element symbol (H–Kr main
# group). magmom = multiplicity - 1; kept in lock-step with
# vibeqc.atomization._GROUND_STATE_MULTIPLICITY.
HUND_MAGMOM: Dict[str, int] = {
    "H": 1,
    "He": 0,
    "Li": 1,
    "Be": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 2,
    "F": 1,
    "Ne": 0,
    "Na": 1,
    "Mg": 0,
    "Al": 1,
    "Si": 2,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Ar": 0,
    "K": 1,
    "Ca": 0,
    "Ga": 1,
    "Ge": 2,
    "As": 3,
    "Se": 2,
    "Br": 1,
    "Kr": 0,
    # d10 closed-shell (non-magnetic ground state, one 5s/4p unpaired):
    "Ag": 1,
    "Cd": 0,
    "Zn": 0,
    # heavy main-group:
    "In": 1,
    "Cs": 1,
    "Ba": 0,
    # 3d transition metals (experimental — NOT in vibe-qc's _GROUND_STATE_MULTIPLICITY):
    "Ti": 2,  # [Ar] 3d2 4s2, S=1 → mult=3 → magmom=2
}


@dataclass(frozen=True)
class PwSystem:
    """A solid plus the free atoms needed for its atomization energy.

    ``n_fu_per_cell`` is the number of formula units in the ASE ``bulk(...)``
    cell (1 for a binary rocksalt/zincblende primitive cell; 2 for an elemental
    diamond cell, where the benchmark "per formula unit" means "per atom").

    ``c`` is the out-of-plane lattice constant for non-cubic structures (rutile,
    wurtzite, etc.); ``None`` for cubic.  When ``c`` is not None, it is passed
    to ``ase.build.bulk(..., c=c)`` along with ``a``.
    """

    id: str
    structure: str  # ASE crystalstructure for ase.build.bulk
    a: float  # lattice constant a, Å (r2SCAN-optimised)
    formula: str  # ASE bulk formula, e.g. "LiF", "C", "BN"
    composition: Tuple[Tuple[str, int], ...]  # atoms per formula unit
    n_fu_per_cell: int
    vasp900_kjmol: float  # VASP@900eV r2SCAN atomization (kJ/mol / f.u.)
    note: str = ""
    c: Optional[float] = None  # lattice constant c (non-cubic); use keyword
    custom_kwargs: Optional[Dict] = (
        None  # for structure="custom": symbols, basis, spacegroup, cellpar
    )

    @property
    def lattice_kwargs(self) -> Dict[str, float]:
        """Keyword arguments for ``ase.build.bulk(..., **lattice_kwargs)``."""
        kw: Dict[str, float] = {"a": self.a}
        if self.c is not None:
            kw["c"] = self.c
        return kw

    def atom_symbols(self) -> Tuple[str, ...]:
        return tuple(sym for sym, _ in self.composition)

    def magmom(self, sym: str) -> int:
        return HUND_MAGMOM[sym]


# Simple, non-magnetic, light validation set — closed-shell insulating solids
# with well-defined r2SCAN minima and geometries available in the benchmark.
VALIDATION_SET: Tuple[PwSystem, ...] = (
    PwSystem(
        "lif_rocksalt",
        "rocksalt",
        4.045,
        "LiF",
        (("Li", 1), ("F", 1)),
        1,
        866.6,
        "light ionic, easy reference",
    ),
    PwSystem(
        "nacl_rocksalt", "rocksalt", 5.569, "NaCl", (("Na", 1), ("Cl", 1)), 1, 627.3
    ),
    PwSystem("naf_rocksalt", "rocksalt", 4.633, "NaF", (("Na", 1), ("F", 1)), 1, 749.6),
    PwSystem(
        "licl_rocksalt", "rocksalt", 5.150, "LiCl", (("Li", 1), ("Cl", 1)), 1, 696.7
    ),
    PwSystem("lih_rocksalt", "rocksalt", 3.979, "LiH", (("Li", 1), ("H", 1)), 1, 468.3),
    PwSystem(
        "mgo_rocksalt", "rocksalt", 4.224, "MgO", (("Mg", 1), ("O", 1)), 1, 1012.0
    ),
    PwSystem("mgs_rocksalt", "rocksalt", 5.260, "MgS", (("Mg", 1), ("S", 1)), 1, 775.0),
    PwSystem(
        "c_diamond",
        "diamond",
        3.553,
        "C",
        (("C", 1),),
        2,
        726.2,
        "elemental: per-f.u. == per-atom",
    ),
    PwSystem(
        "bn_zincblende", "zincblende", 3.592, "BN", (("B", 1), ("N", 1)), 1, 1311.4
    ),
)

# Extended main-group non-magnetic benchmark set — cubic binaries and
# insulators from the 52-compound benchmark (references-cohesive/r2scan.dat).
# Geometries: r2SCAN-optimised lattice constants from the CRYSTAL .d12 inputs.
MAIN_GROUP_SET: Tuple[PwSystem, ...] = (
    # --- rocksalt (sg 225, 2 atoms/cell → 1 f.u.) ---
    PwSystem(
        "agcl_rocksalt",
        "rocksalt",
        5.528,
        "AgCl",
        (("Ag", 1), ("Cl", 1)),
        1,
        526.0,
        "Ag d10 closed-shell",
    ),
    PwSystem(
        "bao_rocksalt", "rocksalt", 5.515, "BaO", (("Ba", 1), ("O", 1)), 1, 1001.5
    ),
    PwSystem("bas_rocksalt", "rocksalt", 6.364, "BaS", (("Ba", 1), ("S", 1)), 1, 950.4),
    PwSystem(
        "cao_rocksalt", "rocksalt", 4.796, "CaO", (("Ca", 1), ("O", 1)), 1, 1096.4
    ),
    PwSystem(
        "kbr_rocksalt", "rocksalt", 6.5888, "KBr", (("K", 1), ("Br", 1)), 1, 620.8
    ),
    PwSystem(
        "kcl_rocksalt", "rocksalt", 6.3257, "KCl", (("K", 1), ("Cl", 1)), 1, 640.0
    ),
    # --- zincblende (sg 216, 2 atoms/cell → 1 f.u.) ---
    PwSystem(
        "alas_zincblende", "zincblende", 5.649, "AlAs", (("Al", 1), ("As", 1)), 1, 742.6
    ),
    PwSystem(
        "aln_zincblende", "zincblende", 4.371, "AlN", (("Al", 1), ("N", 1)), 1, 1119.1
    ),
    PwSystem(
        "alp_zincblende", "zincblende", 5.450, "AlP", (("Al", 1), ("P", 1)), 1, 820.2
    ),
    PwSystem(
        "cdse_zincblende",
        "zincblende",
        6.042,
        "CdSe",
        (("Cd", 1), ("Se", 1)),
        1,
        489.8,
        "Cd d10 closed-shell",
    ),
    PwSystem(
        "gaas_zincblende", "zincblende", 5.640, "GaAs", (("Ga", 1), ("As", 1)), 1, 650.0
    ),
    PwSystem(
        "gan_beta_zincblende",
        "zincblende",
        4.52,
        "GaN",
        (("Ga", 1), ("N", 1)),
        1,
        873.0,
    ),
    PwSystem(
        "gap_zincblende", "zincblende", 5.46, "GaP", (("Ga", 1), ("P", 1)), 1, 717.9
    ),
    PwSystem(
        "inas_zincblende", "zincblende", 6.047, "InAs", (("In", 1), ("As", 1)), 1, 579.9
    ),
    PwSystem(
        "zns_zincblende",
        "zincblende",
        5.45,
        "ZnS",
        (("Zn", 1), ("S", 1)),
        1,
        615.1,
        "Zn d10 closed-shell",
    ),
    # --- caesium chloride (sg 221, 2 atoms/cell → 1 f.u.) ---
    PwSystem(
        "cscl_cscl", "cesiumchloride", 4.1024, "CsCl", (("Cs", 1), ("Cl", 1)), 1, 625.7
    ),
    # --- fluorite (sg 225, 3 atoms/cell = 1 CaF2 f.u.) ---
    PwSystem(
        "caf2_fluorite", "fluorite", 5.49, "CaF2", (("Ca", 1), ("F", 2)), 1, 1597.8
    ),
    # --- antifluorite = fluorite with swapped ions.  ASE assigns elements
    #     by formula order to Wyckoff positions; 'OLi2' puts O at the
    #     cation (0,0,0) site and Li at the anion sites — correct antifluorite.
    PwSystem(
        "li2o_antifluorite",
        "fluorite",
        4.621,
        "OLi2",
        (("Li", 2), ("O", 1)),
        1,
        1149.7,
    ),
    # --- zincblende (SiC is sg 216 zincblende, not diamond) ---
    PwSystem(
        "sic_zincblende", "zincblende", 4.35, "SiC", (("Si", 1), ("C", 1)), 1, 1242.5
    ),
)

# All non-magnetic systems — validation + main-group.
NONMAGNETIC_SET: Tuple[PwSystem, ...] = VALIDATION_SET + MAIN_GROUP_SET

# ---- Non-cubic main-group compounds ----
# These use wurtzite (built via ase.build.bulk with a + c) or corundum
# (built via ase.spacegroup.crystal, dispatched through custom_solid worker).
NONCUBIC_SET: Tuple[PwSystem, ...] = (
    # Wurtzite: 4 atoms/cell = 2 f.u. per primitive
    PwSystem(
        "beo_wurtzite",
        "wurtzite",
        2.714,
        "BeO",
        (("Be", 1), ("O", 1)),
        2,
        1210.2,
        c=4.387,
        note="wurtzite, hexagonal P6_3mc",
    ),
    PwSystem(
        "zno_wurtzite",
        "wurtzite",
        3.250,
        "ZnO",
        (("Zn", 1), ("O", 1)),
        2,
        728.2,
        c=5.200,
        note="wurtzite, hexagonal P6_3mc",
    ),
    # Corundum: built via spacegroup crystal (sg 167, R-3c).
    # Conventional hexagonal cell: 30 atoms = 6 Al2O3 f.u.
    # Al at (0,0,0.3475), O at (0.3060,0,0.2500).
    PwSystem(
        "al2o3_corundum",
        "custom",
        4.7718,
        "Al2O3",
        (("Al", 2), ("O", 3)),
        6,
        3108.9,
        c=12.9732,
        custom_kwargs=dict(
            symbols=["Al", "O"],
            basis=[(0.0, 0.0, 0.3475), (0.3060, 0.0, 0.2500)],
            spacegroup=167,
            cellpar=[4.7718, 4.7718, 12.9732, 90.0, 90.0, 120.0],
        ),
        note="corundum, R-3c, 30 atoms/cell = 6 Al2O3",
    ),
)

# ---- Transition-metal prototypes (experimental) ----
# These need spin-polarised solids, magnetic ordering, possibly DFT+U.
# TiO₂-r (rutile): Ti⁴⁺(d⁰) → solid is non-magnetic; simplest TM entry.
# Free Ti atom is spin-polarised (magmom=2 per Hund).
# Rutile primitive cell: 2 Ti + 4 O = 6 atoms = 2 TiO₂ f.u. (n_fu_per_cell=2).
# Built via ase.spacegroup.crystal (sg 136, P4₂/mnm).
TM_SYSTEMS: Tuple[PwSystem, ...] = (
    PwSystem(
        "tio2_rutile",
        "custom",
        4.578,
        "TiO2",
        (("Ti", 1), ("O", 2)),
        2,
        1986.1,
        c=2.942,
        custom_kwargs=dict(
            symbols=["Ti", "O"],
            basis=[(0.0, 0.0, 0.0), (0.3047, 0.3047, 0.0)],
            spacegroup=136,
            cellpar=[4.578, 4.578, 2.942, 90.0, 90.0, 90.0],
        ),
        note="rutile, P4_2/mnm, 6 atoms/cell = 2 TiO2, Ti4+(d0) non-magnetic",
    ),
)

BY_ID: Dict[str, PwSystem] = {
    s.id: s for s in NONMAGNETIC_SET + NONCUBIC_SET + TM_SYSTEMS
}
