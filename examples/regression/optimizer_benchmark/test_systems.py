"""Curated test systems for the optimizer benchmark.

Each system provides a *perturbed* starting geometry (not the minimum)
so every optimizer must descend. Geometries are stored as code (not
external files) for full reproducibility. All coordinates in bohr
(vibe-qc's native unit).

Categories mirror the paper's proposed structure:
  - Small covalent (H₂O, CH₄, C₂H₄, benzene)
  - H-bonded / flat PES (water dimer)
  - Dispersion-bound (stacked benzene)
  - Flexible torsions (glycine)
  - Medium organic (aspirin)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from vibeqc import Atom, Molecule


@dataclass
class TestSystem:
    """A benchmark test system with a perturbed starting geometry.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"h2o"``).
    category : str
        One of ``"covalent"``, ``"hbonded"``, ``"dispersion"``,
        ``"torsion"``, ``"medium_organic"``, ``"periodic"``.
    molecule : Molecule
        Starting geometry in bohr, perturbed from equilibrium.
    description : str
        Human-readable description for tables.
    n_atoms : int
        Convenience field (derived).
    charge : int
        Net charge.
    multiplicity : int
        Spin multiplicity.
    """

    name: str
    category: str
    molecule: Molecule
    description: str = ""
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_atoms(self) -> int:
        return len(list(self.molecule.atoms))


# ---------------------------------------------------------------------------
# Helper: build a perturbed geometry by scaling + adding noise
# ---------------------------------------------------------------------------


def _perturb(
    zs: list[int],
    coords: list[tuple[float, float, float]],
    scale: float = 1.0,
    noise: float = 0.0,
) -> list[Atom]:
    """Build atoms from Zs and coords, then scale and optionally add noise.

    ``scale`` stretches all bond lengths uniformly. ``noise`` adds
    random displacement (max amplitude in bohr) to each Cartesian
    component — use a fixed seed so results are deterministic.
    """
    import random

    random.seed(42)
    atoms = []
    for z, (x, y, zz) in zip(zs, coords):
        nx = x * scale
        ny = y * scale
        nz = zz * scale
        if noise > 0:
            nx += random.uniform(-noise, noise)
            ny += random.uniform(-noise, noise)
            nz += random.uniform(-noise, noise)
        atoms.append(Atom(z, [nx, ny, nz]))
    random.seed()  # reset
    return atoms


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def _h2o() -> TestSystem:
    """Water — the smallest covalent benchmark.
    Perturbed: O-H bonds stretched ~10%, angle opened ~5°.
    """
    coords = [
        (0.000000, 0.000000, 0.117311),  # O
        (0.000000, 1.447909, -0.939245),  # H1
        (0.000000, -1.447909, -0.939245),  # H2
    ]
    zs = [8, 1, 1]
    mol = Molecule(_perturb(zs, coords, scale=1.10))
    return TestSystem("h2o", "covalent", mol, "H₂O — bent triatomic")


def _ch4() -> TestSystem:
    """Methane — tetrahedral, stiff C-H bonds.
    Perturbed: one C-H bond elongated.
    """
    coords = [
        (0.000000, 0.000000, 0.000000),  # C
        (1.195000, 1.195000, 1.195000),  # H1
        (-1.195000, -1.195000, 1.195000),  # H2
        (-1.195000, 1.195000, -1.195000),  # H3
        (1.195000, -1.195000, -1.195000),  # H4
    ]
    zs = [6, 1, 1, 1, 1]
    mol = Molecule(_perturb(zs, coords, scale=1.08))
    return TestSystem("ch4", "covalent", mol, "CH₄ — tetrahedral")


def _c2h4() -> TestSystem:
    """Ethylene — flat π-system, C=C double bond.
    Perturbed: torsion ~15° out of plane, C=C stretched.
    """
    coords = [
        (0.000000, 0.000000, 1.261419),  # C1
        (0.000000, 0.000000, -1.261419),  # C2
        (0.000000, 1.748349, 2.326628),  # H1
        (0.000000, -1.748349, 2.326628),  # H2
        (0.000000, 1.748349, -2.326628),  # H3
        (0.000000, -1.748349, -2.326628),  # H4
    ]
    zs = [6, 6, 1, 1, 1, 1]
    mol = Molecule(_perturb(zs, coords, scale=1.12))
    return TestSystem("c2h4", "covalent", mol, "C₂H₄ — planar alkene")


def _benzene() -> TestSystem:
    """Benzene — aromatic ring, delocalized π.
    Perturbed: slight breathing-mode expansion.
    """
    r_cc = 2.639  # bohr, ~1.397 Å
    r_ch = 2.048  # bohr, ~1.084 Å
    angles = [i * math.pi / 3 for i in range(6)]
    coords = []
    zs = []
    for a in angles:
        coords.append((r_cc * math.cos(a), r_cc * math.sin(a), 0.0))
        zs.append(6)
    for a in angles:
        coords.append(((r_cc + r_ch) * math.cos(a), (r_cc + r_ch) * math.sin(a), 0.0))
        zs.append(1)
    mol = Molecule(_perturb(zs, coords, scale=1.06, noise=0.05))
    return TestSystem("benzene", "covalent", mol, "C₆H₆ — aromatic ring")


def _h2o_dimer() -> TestSystem:
    """Water dimer — H-bonded complex, very flat PES.
    Perturbed: O-O distance shifted.
    """
    coords = [
        # Donor water
        (0.000000, 0.000000, 0.000000),  # O_d
        (0.000000, 1.447909, -0.939245),  # H_d1
        (0.000000, -1.447909, -0.939245),  # H_d2
        # Acceptor water (H-bonded to H_d1)
        (0.000000, 5.500000, 0.000000),  # O_a
        (-1.447909, 5.939245, 0.000000),  # H_a1
        (1.447909, 5.939245, 0.000000),  # H_a2
    ]
    zs = [8, 1, 1, 8, 1, 1]
    mol = Molecule(_perturb(zs, coords, noise=0.10))
    return TestSystem("h2o_dimer", "hbonded", mol, "(H₂O)₂ — H-bonded dimer, flat PES")


def _stacked_benzene() -> TestSystem:
    """π-stacked benzene dimer — dispersion-bound, very flat PES.
    Perturbed: interplanar separation shifted.
    """
    r_cc = 2.639
    r_ch = 2.048
    angles = [i * math.pi / 3 for i in range(6)]
    separation = 7.0  # bohr, ~3.7 Å (stretched from ~3.4 Å equilibrium)
    coords = []
    zs = []
    # Lower benzene
    for a in angles:
        coords.append((r_cc * math.cos(a), r_cc * math.sin(a), -separation / 2))
        zs.append(6)
    for a in angles:
        coords.append(
            ((r_cc + r_ch) * math.cos(a), (r_cc + r_ch) * math.sin(a), -separation / 2)
        )
        zs.append(1)
    # Upper benzene
    for a in angles:
        coords.append((r_cc * math.cos(a), r_cc * math.sin(a), separation / 2))
        zs.append(6)
    for a in angles:
        coords.append(
            ((r_cc + r_ch) * math.cos(a), (r_cc + r_ch) * math.sin(a), separation / 2)
        )
        zs.append(1)
    mol = Molecule(_perturb(zs, coords, noise=0.08))
    return TestSystem(
        "stacked_benzene",
        "dispersion",
        mol,
        "(C₆H₆)₂ — π-stacked dimer, dispersion-bound PES",
    )


def _glycine() -> TestSystem:
    """Glycine — flexible amino acid, multiple soft torsions.
    Perturbed: backbone dihedral rotated.
    """
    # Approximate equilibrium geometry (bohr), then perturbed dihedral
    coords = [
        (2.265120, 0.290908, 0.036746),  # N
        (0.887161, -0.303088, -0.084162),  # CA
        (0.790868, -1.706152, 0.449410),  # C
        (0.010611, -0.268228, -1.680127),  # H1 (on CA)
        (0.138331, -0.373842, 1.090793),  # H2 (on CA)
        (2.344743, 1.215755, -0.415392),  # H3 (on N)
        (2.514815, 0.469434, 1.022829),  # H4 (on N)
        (1.725718, -2.201235, 0.486972),  # O1 (carbonyl)
        (-0.154605, -2.321903, 0.719370),  # O2 (hydroxyl)
        (1.089111, -2.616213, 1.313724),  # H5 (hydroxyl)
    ]
    zs = [7, 6, 6, 1, 1, 1, 1, 8, 8, 1]
    mol = Molecule(_perturb(zs, coords, noise=0.15))
    return TestSystem(
        "glycine", "torsion", mol, "Glycine — flexible amino acid, soft torsions"
    )


def _aspirin() -> TestSystem:
    """Aspirin (C₉H₈O₄) — medium-sized organic, ester + aromatic ring.
    Perturbed: moderate random displacement.
    """
    # Approximate starting geometry (bohr).  Aspirin = C₆H₄(COOH)(O-CO-CH₃).
    coords = [
        # Carboxyl group — COOH
        (2.925580, 0.486796, -0.040368),  # C1 (carboxyl C)
        (2.080440, -0.800598, -0.086666),  # C2 (ring C-COOH)
        (3.679552, 0.588622, -1.371786),  # O1 (C=O, carbonyl)
        (3.880788, 0.547560, 1.107714),  # O2 (OH, hydroxyl)
        (4.720000, 0.850000, 0.750000),  # H_carboxyl (on O2)
        # Aromatic ring — C₆H₄
        (0.683570, -0.774184, -0.039862),  # C3
        (-0.184920, 0.328144, 0.006436),  # C4
        (-1.581790, 0.189648, 0.037394),  # C5
        (-2.258780, -1.052100, 0.024952),  # C6
        (-1.399930, -2.155384, -0.027594),  # C7
        (-0.003062, -2.015304, -0.065942),  # C8
        # Ring hydrogens
        (0.275348, 1.336708, -0.000434),  # H_ring1
        (-1.991040, 1.201220, 0.063612),  # H_ring2
        (-3.341060, -1.149200, 0.047374),  # H_ring3
        (-1.823720, -3.157080, -0.044768),  # H_ring4
        # Ester — O-CO-CH₃ (O_bridge to ring, C=O carbonyl, CH₃)
        (2.800000, 1.500000, 1.800000),  # O3 (ester bridge O, ring–CO)
        (3.398232, 1.222838, 2.237784),  # O4 (ester C=O carbonyl)
        (5.304682, 0.642032, 0.939084),  # C9 (methyl)
        # Methyl hydrogens
        (5.732756, -0.278660, 0.546572),  # H_me1
        (5.658208, 0.743524, 1.964042),  # H_me2
        (5.692680, 1.489212, 0.346308),  # H_me3
    ]
    # C₉H₈O₄: 9×C + 8×H + 4×O = 21 atoms, 94 electrons
    zs = [
        6,
        6,
        8,
        8,
        1,  # carboxyl: C, C_ring, O=C, OH, H(carboxyl)
        6,
        6,
        6,
        6,
        6,
        6,  # ring carbons
        1,
        1,
        1,
        1,  # ring hydrogens
        8,  # ester bridge O (ring–CO)
        8,  # ester carbonyl O (C=O)
        6,  # methyl C
        1,
        1,
        1,  # methyl hydrogens
    ]
    mol = Molecule(_perturb(zs, coords, noise=0.12))
    return TestSystem(
        "aspirin",
        "medium_organic",
        mol,
        "Aspirin C₉H₈O₄ — ester + aromatic ring, 21 atoms",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SYSTEM_BUILDERS: dict[str, callable] = {
    "h2o": _h2o,
    "ch4": _ch4,
    "c2h4": _c2h4,
    "benzene": _benzene,
    "h2o_dimer": _h2o_dimer,
    "stacked_benzene": _stacked_benzene,
    "glycine": _glycine,
    "aspirin": _aspirin,
}


def get_system(name: str) -> TestSystem:
    """Return a TestSystem by name."""
    if name not in _SYSTEM_BUILDERS:
        available = ", ".join(sorted(_SYSTEM_BUILDERS))
        raise ValueError(f"Unknown system {name!r}. Available: {available}")
    return _SYSTEM_BUILDERS[name]()


def list_systems() -> list[str]:
    """Return all available system names."""
    return sorted(_SYSTEM_BUILDERS)


def get_all_systems() -> list[TestSystem]:
    """Return all test systems."""
    return [builder() for builder in _SYSTEM_BUILDERS.values()]
