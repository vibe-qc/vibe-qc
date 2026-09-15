"""Crystal-structure database for the pob-paper test sets.

Single source of truth for the calibration compounds behind
pob-TZVP (Peintinger-Vilela Oliveira-Bredow, *J. Comput. Chem.*
**34**, 451 (2013)), pob-TZVP-rev2 + pob-DZVP-rev2 (Vilela
Oliveira-Laun-Peintinger-Bredow, *J. Comput. Chem.* **40**, 2364
(2019)), and the upcoming mpei-TZVP HF-optimized sibling
(``GOAL8_MPEI_TZVP.md`` in vibe-qc's ``docs/basisset_dev/``).

Each :class:`Structure` carries:

* the chemical formula (display name),
* lattice parameters (a, b, c, α, β, γ) in **Ångström / degrees**,
* the **fully-expanded** unit-cell atom list (Z, fractional xyz) —
  pre-expanded from the asymmetric unit at database-build time so
  consumers never need spglib at runtime,
* the **asymmetric unit + space-group number** (CRYSTAL14 form) so
  the .d12 emitter in :mod:`vibe_basis.backends.crystal` doesn't
  re-derive symmetry,
* the source citation for the experimental geometry,
* which paper table the compound appears in,
* the spin multiplicity (closed-shell vs. open-shell for AFM TM
  oxides),
* the ECP requirement (None for all-electron, otherwise the
  libecpint XML library name and a per-element ncore map).

The pob papers used CRYSTAL on relaxed geometries; the lattice
constants below are the **experimental** references from PT2013
Tables 4–14 — what the SCF will be benchmarked against by Stage 0
of Goal 8 once the recipe driver lands.

Bohr conversion is done at input-emit time, not here. Keeping the
database in Å keeps the numbers human-readable and matches the
way the papers list them.

Phases shipped (per Goal 3 of basissetdev):

* **Phase 1** — 13 cubic ionic compounds (PT2013 T4 + T14):
  rocksalt + fluorite + antifluorite.
* **Phase 2** — 24 cubic semiconductors / carbides / nitrides /
  TM oxides (PT2013 T8 + T10). 6 AFM TM oxides have
  ``crystal_asymm_unit`` set but emit no .d12 until R3 lands
  (broken-symmetry guess for the magnetic ordering).
* **Phase 3** — 2 oxide-with-metal-in-low-coordination (Cu₂O
  cuprite, Cu₃N anti-ReO₃).

Hexagonal / tetragonal compounds (PT2013 T5 / T9 / T11) are
blocked on REQUIREMENTS-PERIODIC R1 (non-orthorhombic Ewald) and
will land in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StructureAtom:
    """One atom in the unit cell, expressed in fractional coordinates."""
    Z: int
    fxyz: tuple[float, float, float]


@dataclass(frozen=True)
class Structure:
    """Crystal structure + metadata for a pob-paper compound.

    Two atom representations co-exist:

    * ``unit_cell`` — the **fully expanded** unit cell, used by the
      vibe-qc input generator since vibe-qc's ``PeriodicSystem`` takes
      explicit Cartesian positions.
    * ``crystal_asymm_unit`` + ``crystal_spacegroup`` — the
      **asymmetric unit + space-group number**, used by the
      CRYSTAL14 .d12 generator (Phase 14h, 2026-05-10) since CRYSTAL
      expands from the asymmetric unit via the space group at
      runtime. This matches every published CRYSTAL benchmark (e.g.
      PT2013 SI Table 2's MgO is two lines ``12 0 0 0`` / ``8 0.5
      0.5 0.5`` plus the ``225`` space-group line).

    Both views must describe the same physical crystal; they're
    cross-checked in tests. CRYSTAL is treated strictly as an
    **external code** per CLAUDE.md § 10 — vibe-qc generates the
    .d12 input, ``vq submit`` ships it to compute-host, CRYSTAL runs
    out-of-process. No imports.
    """

    name: str                        # canonical compound name (filename slug)
    formula: str                     # display formula, e.g. "α-Al₂O₃"
    spacegroup: str                  # short symbol, e.g. "Fm-3m" or "P6_3/mmc"
    crystal_system: str              # cubic / hexagonal / tetragonal / orthorhombic / monoclinic
    a: float                         # Å
    b: float                         # Å
    c: float                         # Å
    alpha: float = 90.0              # degrees
    beta: float = 90.0
    gamma: float = 90.0
    unit_cell: tuple[StructureAtom, ...] = ()
    multiplicity: int = 1            # spin multiplicity of the unit cell
    afm_pattern: Optional[str] = None  # e.g. "AF2 (111)" for AFM TM oxides
    pob_tables: tuple[str, ...] = ()   # which paper-tables list the compound
    structure_source: str = ""       # where the experimental geometry comes from
    ecp_library: Optional[str] = None  # libecpint XML name, None = all-electron
    notes: str = ""                  # human-readable comment
    # ---- CRYSTAL14 .d12 parity-input fields ----------------------------
    crystal_spacegroup: int = 0      # ITC space-group number 1-230; 0 = unset (no .d12 emitted)
    crystal_asymm_unit: tuple[StructureAtom, ...] = ()

    def lattice_matrix_angstrom(self) -> list[list[float]]:
        """Return a 3×3 lattice matrix (rows = a, b, c vectors) in Å.

        For non-orthogonal cells (hexagonal etc.) computes the
        Cartesian lattice from (a, b, c, α, β, γ). For the Phase 1
        cubic-only compounds α=β=γ=90° and the matrix is simply
        a·diag(1, b/a, c/a).
        """
        import math
        ca = math.cos(math.radians(self.alpha))
        cb = math.cos(math.radians(self.beta))
        cg = math.cos(math.radians(self.gamma))
        sg = math.sin(math.radians(self.gamma))
        ax = self.a
        bx = self.b * cg
        by = self.b * sg
        cx = self.c * cb
        cy = self.c * (ca - cb * cg) / sg if sg else 0.0
        cz = math.sqrt(max(self.c * self.c - cx * cx - cy * cy, 0.0))
        # Round near-zero entries to zero for cosmetic cleanliness.
        def _z(x: float) -> float:
            return 0.0 if abs(x) < 1e-12 else x
        return [
            [_z(ax), 0.0, 0.0],
            [_z(bx), _z(by), 0.0],
            [_z(cx), _z(cy), _z(cz)],
        ]


# ---------- Symmetry expansion helpers --------------------------------------
#
# Pre-expand the asymmetric unit to a full conventional cell at
# database-build time so the generated input files never depend on
# spglib at runtime.

def _rocksalt(a: float, Z_cation: int, Z_anion: int) -> tuple[StructureAtom, ...]:
    """8-atom rocksalt cell (Fm-3m): 4 cations + 4 anions on FCC sublattices.

    Conventional cubic cell is orthorhombic (a=b=c, α=β=γ=90°), so it
    fits vibe-qc's FFT-Poisson constraint.
    """
    cation_sites = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    anion_sites = [(0.5, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0)]
    return tuple(
        StructureAtom(Z=Z_cation, fxyz=tuple(s))
        for s in cation_sites
    ) + tuple(
        StructureAtom(Z=Z_anion, fxyz=tuple(s))
        for s in anion_sites
    )


# ---- CRYSTAL14 asymmetric-unit helpers --------------------------------
#
# Each ``_asymm_*`` helper returns the 1-2 atom asymmetric unit that
# CRYSTAL re-expands via the space group at parse time. Paired with
# ``crystal_spacegroup`` (ITC number) on the Structure record.

def _asymm_rocksalt(Z_cation: int, Z_anion: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of rocksalt (Fm-3m, sg 225): cation at 4a (0,0,0),
    anion at 4b (½,½,½). CRYSTAL emits the 8-atom conventional cell."""
    return (
        StructureAtom(Z=Z_cation, fxyz=(0.0, 0.0, 0.0)),
        StructureAtom(Z=Z_anion,  fxyz=(0.5, 0.5, 0.5)),
    )


def _asymm_fluorite(Z_cation: int, Z_anion: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of fluorite (Fm-3m, sg 225): cation at 4a (0,0,0),
    anion at 8c (¼,¼,¼). Used for CaF₂."""
    return (
        StructureAtom(Z=Z_cation, fxyz=(0.0, 0.0, 0.0)),
        StructureAtom(Z=Z_anion,  fxyz=(0.25, 0.25, 0.25)),
    )


def _asymm_antifluorite(Z_cation: int, Z_anion: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of antifluorite (Fm-3m, sg 225): anion at 4a (0,0,0),
    cation at 8c (¼,¼,¼). Used for K₂O, Na₂Se, K₂S."""
    return (
        StructureAtom(Z=Z_anion,  fxyz=(0.0, 0.0, 0.0)),
        StructureAtom(Z=Z_cation, fxyz=(0.25, 0.25, 0.25)),
    )


def _fluorite(a: float, Z_cation: int, Z_anion: int) -> tuple[StructureAtom, ...]:
    """12-atom fluorite cell (Fm-3m, AB₂): 4 cations on FCC + 8 anions
    on tetrahedral sites.
    """
    cation_sites = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    anion_sites = [
        (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
        (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
        (0.75, 0.75, 0.75), (0.25, 0.25, 0.75),
        (0.25, 0.75, 0.25), (0.75, 0.25, 0.25),
    ]
    return tuple(
        StructureAtom(Z=Z_cation, fxyz=tuple(s))
        for s in cation_sites
    ) + tuple(
        StructureAtom(Z=Z_anion, fxyz=tuple(s))
        for s in anion_sites
    )


def _antifluorite(a: float, Z_cation: int, Z_anion: int) -> tuple[StructureAtom, ...]:
    """12-atom antifluorite cell (Fm-3m, A₂B): 8 cations + 4 anions —
    fluorite with cation/anion roles swapped (e.g. K₂O).

    Structurally identical to fluorite with the (cation, anion) labels
    flipped: anions occupy the FCC sublattice, cations occupy both
    tetrahedral sites. Reuse :func:`_fluorite` with swapped Z's so
    there's exactly one expansion routine per Bravais layout.
    """
    return _fluorite(a, Z_anion, Z_cation)


def _zincblende(a: float, Z_a: int, Z_b: int) -> tuple[StructureAtom, ...]:
    """8-atom zincblende cell (F-43m, AB-type binary covalent compound).

    Conventional cubic cell, orthorhombic-friendly. A on FCC at the
    origin, B on FCC offset by (¼, ¼, ¼). Used for AlP, AlN, GaAs,
    GaP, β-BN, β-SiC, β-ZnS, ZnSe, MnSe, etc. Diamond is the
    A=B special case.
    """
    a_sites = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    b_sites = [
        (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
        (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
    ]
    return tuple(
        StructureAtom(Z=Z_a, fxyz=tuple(s)) for s in a_sites
    ) + tuple(
        StructureAtom(Z=Z_b, fxyz=tuple(s)) for s in b_sites
    )


def _asymm_zincblende(Z_a: int, Z_b: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of zincblende (F-43m, sg 216): A at 4a (0,0,0),
    B at 4c (¼,¼,¼). Used for AlP, AlN, GaAs, GaP, β-BN, β-SiC,
    β-ZnS, ZnSe."""
    return (
        StructureAtom(Z=Z_a, fxyz=(0.0, 0.0, 0.0)),
        StructureAtom(Z=Z_b, fxyz=(0.25, 0.25, 0.25)),
    )


def _diamond(a: float, Z: int) -> tuple[StructureAtom, ...]:
    """8-atom diamond cell (Fd-3m): zincblende with both sublattices the
    same element. Used for C(diamond), Si, Ge, α-Sn (group-14
    elemental semiconductors).
    """
    return _zincblende(a, Z, Z)


def _asymm_diamond(Z: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of diamond (Fd-3m, sg 227): single atom at 8a
    (⅛,⅛,⅛) in origin choice 2 (CRYSTAL's default convention)."""
    return (
        StructureAtom(Z=Z, fxyz=(0.125, 0.125, 0.125)),
    )


def _anti_reo3(a: float, Z_a: int, Z_b: int) -> tuple[StructureAtom, ...]:
    """4-atom anti-ReO₃ cell (Pm-3m, A₃B): Cu₃N is the canonical case.

    A on the three face-centers (½, ½, 0), (½, 0, ½), (0, ½, ½)
    (Wyckoff 3c), B at the cube corner (0, 0, 0) (Wyckoff 1a).
    Inverted from ReO₃ where the metal is at the corner and oxygens
    on the faces.
    """
    a_sites = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    b_sites = [(0.0, 0.0, 0.0)]
    return tuple(
        StructureAtom(Z=Z_a, fxyz=tuple(s)) for s in a_sites
    ) + tuple(
        StructureAtom(Z=Z_b, fxyz=tuple(s)) for s in b_sites
    )


def _asymm_anti_reo3(Z_a: int, Z_b: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of anti-ReO₃ (Pm-3m, sg 221): B at 1a (0,0,0),
    A at 3c (½,½,0). CRYSTAL re-expands to 4 atoms/cell."""
    return (
        StructureAtom(Z=Z_b, fxyz=(0.0, 0.0, 0.0)),
        StructureAtom(Z=Z_a, fxyz=(0.5, 0.5, 0.0)),
    )


def _cuprite(a: float, Z_metal: int, Z_oxygen: int) -> tuple[StructureAtom, ...]:
    """6-atom cuprite cell (Pn-3m, M₂O), the canonical case Cu₂O.

    Metal on Wyckoff 4b at tetrahedral sites (¼, ¼, ¼) and the three
    distinct (¾, ¾, ¼)-style positions; oxygen on Wyckoff 2a at the
    BCC sublattice (0, 0, 0) and (½, ½, ½). Each O is linearly
    coordinated by two metal atoms; each metal is tetrahedrally
    coordinated by O.
    """
    metal_sites = [
        (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75), (0.75, 0.75, 0.25),
    ]
    oxygen_sites = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    return tuple(
        StructureAtom(Z=Z_metal, fxyz=tuple(s)) for s in metal_sites
    ) + tuple(
        StructureAtom(Z=Z_oxygen, fxyz=tuple(s)) for s in oxygen_sites
    )


def _asymm_cuprite(Z_metal: int, Z_oxygen: int) -> tuple[StructureAtom, ...]:
    """Asymm unit of cuprite (Pn-3m, sg 224): O at 2a (0,0,0),
    metal at 4b (¼,¼,¼). CRYSTAL re-expands to 6 atoms/cell."""
    return (
        StructureAtom(Z=Z_oxygen, fxyz=(0.0, 0.0, 0.0)),
        StructureAtom(Z=Z_metal,  fxyz=(0.25, 0.25, 0.25)),
    )


# ---------- The Phase 1 database --------------------------------------------
#
# Every compound is fully-determined by space-group, element pair,
# and the experimental lattice constant. Lattice constants come
# from Table 4 of PT2013 (and Table 2 of VO2019 for KBr / SiF₄,
# which are listed but not all in T4); citations point at the
# specific references the pob paper uses.

# Element atomic numbers — keep close at hand for the entries below.
H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca = (
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20
)
Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn = 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
Ga, Ge, As, Se, Br, Sr = 31, 32, 33, 34, 35, 38


STRUCTURES: dict[str, Structure] = {

    # ---- Rocksalt-type alkali halides (Fm-3m, Z=4) -------------------------

    "LiCl": Structure(
        name="LiCl", formula="LiCl",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.130, b=5.130, c=5.130,
        unit_cell=_rocksalt(5.130, Li, Cl),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Li, Cl),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Aguayo, Mazin, Singh, PRL 92, 147201 (2004) — Ref. 17 of PT2013",
    ),

    "NaCl": Structure(
        name="NaCl", formula="NaCl",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.640, b=5.640, c=5.640,
        unit_cell=_rocksalt(5.640, Na, Cl),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Na, Cl),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "PT2013-T7", "VO2019-T2"),
        structure_source="Walker, Verma, Cranswick, J. Solid State Chem. 88, 204 (1990) — Ref. 49 of PT2013",
    ),

    "LiF": Structure(
        name="LiF", formula="LiF",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.027, b=4.027, c=4.027,
        unit_cell=_rocksalt(4.027, Li, F),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Li, F),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Dupre, Recker, Wallrafen, Mater. Res. Bull. 27, 311 (1992) — Ref. 50 of PT2013",
    ),

    "NaF": Structure(
        name="NaF", formula="NaF",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.632, b=4.632, c=4.632,
        unit_cell=_rocksalt(4.632, Na, F),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Na, F),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Rao, Sanyal, Phys. Rev. B 42, 1810 (1990) — Ref. 51 of PT2013",
    ),

    "KF": Structure(
        name="KF", formula="KF",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.347, b=5.347, c=5.347,
        unit_cell=_rocksalt(5.347, K, F),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(K, F),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Chichagov, Mater. Sci. Forum 166, 193 (1994) — Ref. 52 of PT2013",
    ),

    "KBr": Structure(
        name="KBr", formula="KBr",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=6.570, b=6.570, c=6.570,
        unit_cell=_rocksalt(6.570, K, Br),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(K, Br),
        pob_tables=("VO2019-T2",),
        structure_source="VO2019 Ref. 24",
    ),

    # ---- Alkali / alkaline-earth oxides + alkaline-earth fluoride ---------

    "CaF2": Structure(
        name="CaF2", formula="CaF₂",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.463, b=5.463, c=5.463,
        unit_cell=_fluorite(5.463, Ca, F),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_fluorite(Ca, F),
        pob_tables=("PT2013-T4", "PT2013-T7", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Yang, Andersson, Acta Crystallogr. B 43, 1 (1987) — Ref. 53 of PT2013",
    ),

    "K2O": Structure(
        name="K2O", formula="K₂O",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=6.436, b=6.436, c=6.436,
        unit_cell=_antifluorite(6.436, K, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_antifluorite(K, O),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Straumanis, Ievins, Z. Anorg. Allg. Chem. 241, 281 (1939) — Ref. 54 of PT2013",
    ),

    "MgO": Structure(
        name="MgO", formula="MgO",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.217, b=4.217, c=4.217,
        unit_cell=_rocksalt(4.217, Mg, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Mg, O),
        pob_tables=("PT2013-T4", "PT2013-T7", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Walker, Verma, Cranswick, J. Solid State Chem. 88, 204 (1990) — same as NaCl",
    ),

    "CaO": Structure(
        name="CaO", formula="CaO",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.811, b=4.811, c=4.811,
        unit_cell=_rocksalt(4.811, Ca, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Ca, O),
        pob_tables=("PT2013-T4", "PT2013-T7", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Gajbhiye, Ningthoujam, Mater. Res. Bull. 41, 1612 (2006) — Ref. 55 of PT2013",
    ),

    # ---- Alkali / alkaline-earth hydrides ---------------------------------

    "LiH": Structure(
        name="LiH", formula="LiH",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.083, b=4.083, c=4.083,
        unit_cell=_rocksalt(4.083, Li, H),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Li, H),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Hasegawa, Yagi, J. Alloys Compd. 403, 131 (2005) — Ref. 56 of PT2013",
    ),

    "NaH": Structure(
        name="NaH", formula="NaH",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.890, b=4.890, c=4.890,
        unit_cell=_rocksalt(4.890, Na, H),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Na, H),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Sweeney, Heinz, Phys. Chem. Miner. 20, 63 (1993) — Ref. 57 of PT2013",
    ),

    "KH": Structure(
        name="KH", formula="KH",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.704, b=5.704, c=5.704,
        unit_cell=_rocksalt(5.704, K, H),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(K, H),
        pob_tables=("PT2013-T4", "PT2013-T13", "PT2013-T14", "VO2019-T2"),
        structure_source="Guerin, Guivarch, J. Appl. Phys. 66, 2122 (1989) — Ref. 58 of PT2013",
    ),

    # ---- Phase 2 — diamond-type group-14 semiconductors (Fd-3m, Z=8) ------

    "C-diamond": Structure(
        name="C-diamond", formula="C (diamond)",
        spacegroup="Fd-3m", crystal_system="cubic",
        a=3.567, b=3.567, c=3.567,
        unit_cell=_diamond(3.567, C),
        crystal_spacegroup=227,
        crystal_asymm_unit=_asymm_diamond(C),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 24 (BSE-recommended experimental value)",
    ),

    "Si": Structure(
        name="Si", formula="Si",
        spacegroup="Fd-3m", crystal_system="cubic",
        a=5.431, b=5.431, c=5.431,
        unit_cell=_diamond(5.431, Si),
        crystal_spacegroup=227,
        crystal_asymm_unit=_asymm_diamond(Si),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 25",
    ),

    "Ge": Structure(
        name="Ge", formula="Ge",
        spacegroup="Fd-3m", crystal_system="cubic",
        a=5.621, b=5.621, c=5.621,
        unit_cell=_diamond(5.621, Ge),
        crystal_spacegroup=227,
        crystal_asymm_unit=_asymm_diamond(Ge),
        pob_tables=("PT2013-T8",),
        structure_source="PT2013 Ref. 26",
    ),

    # ---- Zincblende-type binary semiconductors (F-43m, Z=4 fu) ------------

    "AlP": Structure(
        name="AlP", formula="AlP",
        spacegroup="F-43m", crystal_system="cubic",
        a=5.421, b=5.421, c=5.421,
        unit_cell=_zincblende(5.421, Al, P),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Al, P),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 76",
    ),

    "AlN": Structure(
        name="AlN", formula="AlN",
        spacegroup="F-43m", crystal_system="cubic",
        a=4.365, b=4.365, c=4.365,
        unit_cell=_zincblende(4.365, Al, N),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Al, N),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 45",
    ),

    "GaAs": Structure(
        name="GaAs", formula="GaAs",
        spacegroup="F-43m", crystal_system="cubic",
        a=5.653, b=5.653, c=5.653,
        unit_cell=_zincblende(5.653, Ga, As),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Ga, As),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 78",
    ),

    "GaP": Structure(
        name="GaP", formula="GaP",
        spacegroup="F-43m", crystal_system="cubic",
        a=5.448, b=5.448, c=5.448,
        unit_cell=_zincblende(5.448, Ga, P),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Ga, P),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 79",
    ),

    "ZnS-beta": Structure(
        name="ZnS-beta", formula="β-ZnS (zincblende)",
        spacegroup="F-43m", crystal_system="cubic",
        a=5.400, b=5.400, c=5.400,
        unit_cell=_zincblende(5.400, Zn, S),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Zn, S),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 30",
    ),

    "ZnSe": Structure(
        name="ZnSe", formula="ZnSe",
        spacegroup="F-43m", crystal_system="cubic",
        a=5.674, b=5.674, c=5.674,
        unit_cell=_zincblende(5.674, Zn, Se),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Zn, Se),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 80",
    ),

    "BN-beta": Structure(
        name="BN-beta", formula="β-BN (zincblende)",
        spacegroup="F-43m", crystal_system="cubic",
        a=3.625, b=3.625, c=3.625,
        unit_cell=_zincblende(3.625, B, N),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(B, N),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 27",
    ),

    "SiC-beta": Structure(
        name="SiC-beta", formula="β-SiC (zincblende)",
        spacegroup="F-43m", crystal_system="cubic",
        a=4.358, b=4.358, c=4.358,
        unit_cell=_zincblende(4.358, Si, C),
        crystal_spacegroup=216,
        crystal_asymm_unit=_asymm_zincblende(Si, C),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 52",
    ),

    # ---- Rocksalt-type carbides and nitrides (Fm-3m, Z=4 fu) --------------

    "TiC": Structure(
        name="TiC", formula="TiC",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.328, b=4.328, c=4.328,
        unit_cell=_rocksalt(4.328, Ti, C),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Ti, C),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 52",
    ),

    "VC": Structure(
        name="VC", formula="VC",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.163, b=4.163, c=4.163,
        unit_cell=_rocksalt(4.163, V, C),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(V, C),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 86",
    ),

    "TiN": Structure(
        name="TiN", formula="TiN",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.235, b=4.235, c=4.235,
        unit_cell=_rocksalt(4.235, Ti, N),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Ti, N),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 87",
    ),

    "VN": Structure(
        name="VN", formula="VN",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.137, b=4.137, c=4.137,
        unit_cell=_rocksalt(4.137, V, N),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(V, N),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 27",
    ),

    "CrN": Structure(
        name="CrN", formula="CrN",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.135, b=4.135, c=4.135,
        unit_cell=_rocksalt(4.135, Cr, N),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Cr, N),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 27",
        # Note: CrN is paramagnetic above ~280 K but AFM at low T.
        # PT2013 treats it spin-restricted; T8 lists a paramagnetic
        # rocksalt structure. R3 not strictly required.
    ),

    # ---- Antifluorite-type alkali chalcogenides (Fm-3m, Z=4 fu) -----------

    "Na2Se": Structure(
        name="Na2Se", formula="Na₂Se",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=6.825, b=6.825, c=6.825,
        unit_cell=_antifluorite(6.825, Na, Se),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_antifluorite(Na, Se),
        pob_tables=("PT2013-T8",),
        structure_source="PT2013 Ref. 77",
    ),

    "K2S": Structure(
        name="K2S", formula="K₂S",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=7.407, b=7.407, c=7.407,
        unit_cell=_antifluorite(7.407, K, S),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_antifluorite(K, S),
        pob_tables=("PT2013-T8", "PT2013-T7"),
        structure_source="PT2013 Ref. 72",
    ),

    # ---- AFM rocksalt transition-metal monoxides (PT2013 T10) -------------
    #
    # All four are paramagnetic at room temperature but order
    # antiferromagnetically below T_N (MnO ~120 K, FeO ~198 K,
    # CoO ~291 K, NiO ~523 K). PT2013 uses the AFM-II ordering
    # (magnetic moments parallel within {111} planes, alternating
    # plane-to-plane). The conventional cubic cell has total spin = 0
    # but the SCF needs broken-symmetry initialization. Tag with
    # afm_pattern; the generator emits a # BLOCKED ON: R3 stub.

    "MnO": Structure(
        name="MnO", formula="MnO",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.445, b=4.445, c=4.445,
        unit_cell=_rocksalt(4.445, Mn, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Mn, O),
        multiplicity=1,
        afm_pattern="AFM-II ⟨111⟩",
        pob_tables=("PT2013-T10", "VO2019-T9"),
        structure_source="PT2013 Ref. 19",
    ),

    "FeO": Structure(
        name="FeO", formula="FeO (wüstite)",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.326, b=4.326, c=4.326,
        unit_cell=_rocksalt(4.326, Fe, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Fe, O),
        multiplicity=1,
        afm_pattern="AFM-II ⟨111⟩",
        pob_tables=("PT2013-T10", "VO2019-T9"),
        structure_source="PT2013 Ref. 19",
    ),

    "CoO": Structure(
        name="CoO", formula="CoO",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.250, b=4.250, c=4.250,
        unit_cell=_rocksalt(4.250, Co, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Co, O),
        multiplicity=1,
        afm_pattern="AFM-II ⟨111⟩",
        pob_tables=("PT2013-T10", "VO2019-T9"),
        structure_source="PT2013 Ref. 19",
    ),

    "NiO": Structure(
        name="NiO", formula="NiO",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=4.195, b=4.195, c=4.195,
        unit_cell=_rocksalt(4.195, Ni, O),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Ni, O),
        multiplicity=1,
        afm_pattern="AFM-II ⟨111⟩",
        pob_tables=("PT2013-T10", "VO2019-T9"),
        structure_source="PT2013 Ref. 19",
    ),

    # AFM rocksalt sulfides — same R3 caveat.

    "MnS": Structure(
        name="MnS", formula="MnS (α, alabandite)",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.220, b=5.220, c=5.220,
        unit_cell=_rocksalt(5.220, Mn, S),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Mn, S),
        multiplicity=1,
        afm_pattern="AFM-II ⟨111⟩",
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 88",
    ),

    "MnSe": Structure(
        name="MnSe", formula="MnSe (α-rocksalt)",
        spacegroup="Fm-3m", crystal_system="cubic",
        a=5.460, b=5.460, c=5.460,
        unit_cell=_rocksalt(5.460, Mn, Se),
        crystal_spacegroup=225,
        crystal_asymm_unit=_asymm_rocksalt(Mn, Se),
        multiplicity=1,
        afm_pattern="AFM-II ⟨111⟩",
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 81",
    ),

    # ---- Phase 3 — anti-ReO₃ + cuprite (PT2013 T8 / T10 cubic TM oxide) ---

    "Cu3N": Structure(
        name="Cu3N", formula="Cu₃N",
        spacegroup="Pm-3m", crystal_system="cubic",
        a=3.817, b=3.817, c=3.817,
        unit_cell=_anti_reo3(3.817, Cu, N),
        crystal_spacegroup=221,
        crystal_asymm_unit=_asymm_anti_reo3(Cu, N),
        pob_tables=("PT2013-T8", "VO2019-T6"),
        structure_source="PT2013 Ref. 84",
    ),

    "Cu2O": Structure(
        name="Cu2O", formula="Cu₂O (cuprite)",
        spacegroup="Pn-3m", crystal_system="cubic",
        a=4.269, b=4.269, c=4.269,
        unit_cell=_cuprite(4.269, Cu, O),
        crystal_spacegroup=224,
        crystal_asymm_unit=_asymm_cuprite(Cu, O),
        pob_tables=("PT2013-T10", "VO2019-T9"),
        structure_source="Kirfel & Eichhorn, Acta Crystallogr. A 46, 271 (1990) — Ref. 68 of PT2013",
    ),
}


# ---------- Deferred Phase-3 entries — large cells / Wyckoff parameters ---
#
# Three cubic TM-oxide-row compounds and one cubic intermetallic from
# the pob test set are not in STRUCTURES yet. They each need a
# bespoke handler (Wyckoff free parameters, special-position
# expansion, or — for the metals — Fermi-Dirac smearing that vibe-qc
# doesn't yet have). All four have substantial unit cells too large
# for productive laptop testing; add them once the prerequisites are
# in place.
#
# Sc₂O₃ — bixbyite, Ia-3, a=9.846 Å. 32 Sc + 48 O = 80 atoms in the
#   conventional cubic cell. Wyckoff sites 8b (¼,¼,¼) and 24d (x,0,¼)
#   for Sc; 48e (x,y,z) for O — all three have free parameters from
#   Levy et al., Am. Mineral. 90, 1157 (2005). Implementing requires
#   either spglib symmetry expansion at runtime (unwanted dependency
#   for example inputs) or hand-baking the 80 fractional positions
#   from the published u/v/w values. **Defer pending Goal-3 cell-size
#   policy decision.**
#
# ZnCr₂O₄ — spinel, Fd-3m, a=8.329 Å. 8 Zn + 16 Cr + 32 O = 56 atoms.
#   Zn at 8a (1/8, 1/8, 1/8), Cr at 16d (½, ½, ½), O at 32e (u, u, u)
#   with u ≈ 0.385 from Levy et al. (Ref. 69 of PT2013). Same defer
#   reason as Sc₂O₃ — the 32e u-parameter needs hand-baking.
#
# Ni₃Al — Cu₃Au-type, Pm-3m, a=3.550 Å. 3 Ni + 1 Al = 4 atoms in the
#   primitive cubic cell. Structure trivial (= _anti_reo3(a, Ni, Al))
#   but Ni₃Al is a metal — needs metallic k-mesh + Fermi-Dirac
#   smearing (REQUIREMENTS-PERIODIC R2). **Defer until R2 lands.**
#
# Add these in a Phase 4 turn alongside the R3 unblocking work.


def all_structures() -> list[Structure]:
    """Return every Structure registered in the database, sorted by name."""
    return sorted(STRUCTURES.values(), key=lambda s: s.name)


def in_table(table_id: str) -> list[Structure]:
    """Return every structure that appears in the given paper table.

    ``table_id`` matches the ``pob_tables`` field — e.g. ``"PT2013-T4"``,
    ``"VO2019-T2"``.
    """
    return [s for s in STRUCTURES.values() if table_id in s.pob_tables]
