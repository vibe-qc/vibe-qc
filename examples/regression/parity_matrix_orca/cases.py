"""Cell definitions for the ORCA parity matrix + speed benchmark.

A *cell* is one ``(system, basis, method, charge, spin, df, cosx)``
tuple — the unit both deliverables iterate over. Geometries are written
here in Angstrom (human-readable) and converted **once** to Bohr by
:func:`geometry_bohr_symbols`; both the ORCA input (fed Bohr via the
``Bohrs`` keyword) and the vibe-qc side consume that single conversion,
so the two codes sit on identical nuclear coordinates.

The parity geometries mirror ``tests/test_parity_hf_dft.py`` exactly
(H2O / H2CO / OH at the same coordinates) so the ORCA axis lines up
cell-for-cell with the PySCF axis already in that file. The speed
ladder adds larger systems to find the DF / RIJCOSX crossover the
handover calls for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# --- geometries (Angstrom) ----------------------------------------------
# H2O / H2CO / OH are the exact coordinates of test_parity_hf_dft.py's
# _GEOMS, just expressed in Angstrom instead of Bohr.
GEOMETRIES = {
    "H2O": [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.793353, -0.613510)),
        ("H", (0.0, -0.793353, -0.613510)),
    ],
    "H2CO": [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.205)),
        ("H", (0.0, 0.943, -0.587)),
        ("H", (0.0, -0.943, -0.587)),
    ],
    "OH": [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.97)),
    ],
    # Open-shell radicals — same geometries as
    # `tests/test_parity_hf_dft.py::_GEOMS` (NIST CCCBDB exp bond
    # lengths for the diatomics; published B3LYP/aug-cc-pVTZ for
    # methylperoxyl). BRIEF.md § Scope-2 coverage.
    "NO": [
        ("N", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.151)),                  # r(N-O) = 1.151 Å
    ],
    "CN": [
        ("C", (0.0, 0.0, 0.0)),
        ("N", (0.0, 0.0, 1.172)),                  # r(C-N) = 1.172 Å
    ],
    # Methylperoxyl CH3O2· — Cs, doublet. Constructed from r(C-O)=1.45,
    # r(O-O)=1.32, r(C-H)=1.094 Å with ∠(C-O-O)=110°, ∠(O-C-H)=109.5°
    # tetrahedral, H atoms staggered (60°/180°/300° azimuthal) so the
    # anti-H sits anti-periplanar to the radical site. Coordinates
    # below are this construction expressed in Å.
    "CH3OO": [
        ("C",  ( 0.0000000,  0.0000000,  0.0000000)),
        ("O",  ( 0.0000000,  0.0000000,  1.4500000)),
        ("O",  ( 1.2406484,  0.0000000,  0.9985598)),
        ("H",  ( 0.5158118,  0.8933350, -0.3648961)),   # phi=60°
        ("H",  (-1.0316235,  0.0000000, -0.3648961)),   # phi=180° (anti-O2)
        ("H",  ( 0.5158118, -0.8933350, -0.3648961)),   # phi=300°
    ],
    # Glycine (neutral, Cs conformer) — 5 heavy atoms, the second rung
    # of the speed ladder. Geometry from a tight B3LYP/def2-TZVP
    # optimisation; only used for wall-clock timing, not parity, so the
    # exact conformer doesn't need a citation.
    "glycine": [
        ("N", (-1.476, 0.245, 0.066)),
        ("C", (-0.224, -0.466, -0.093)),
        ("C", (0.973, 0.450, 0.085)),
        ("O", (0.901, 1.633, 0.330)),
        ("O", (2.155, -0.169, -0.043)),
        ("H", (-1.514, 1.018, -0.595)),
        ("H", (-2.260, -0.369, -0.112)),
        ("H", (-0.180, -1.284, 0.638)),
        ("H", (-0.171, -0.913, -1.094)),
        ("H", (2.881, 0.456, 0.072)),
    ],
    # --- Larger / heavier-element systems for the extended speed
    # benchmark. All closed-shell. Built from first principles
    # (Oh / Td / D5h / D2h / all-trans zigzag); not opt-in by default
    # in SPEED_LADDER_SYSTEMS so `run_speed_benchmark.py` with no flags
    # stays cheap — pass `--systems naphthalene,ferrocene,...` to
    # exercise them.
    #
    # SF6: octahedral, S-F = 1.561 Å (exp).
    "SF6": [
        ("S", (0.0000, 0.0000, 0.0000)),
        ("F", (1.5610, 0.0000, 0.0000)),
        ("F", (-1.5610, 0.0000, 0.0000)),
        ("F", (0.0000, 1.5610, 0.0000)),
        ("F", (0.0000, -1.5610, 0.0000)),
        ("F", (0.0000, 0.0000, 1.5610)),
        ("F", (0.0000, 0.0000, -1.5610)),
    ],
    # CCl4: tetrahedral, C-Cl = 1.766 Å (exp).
    "CCl4": [
        ("C", (0.0000, 0.0000, 0.0000)),
        ("Cl", (1.0196, 1.0196, 1.0196)),
        ("Cl", (1.0196, -1.0196, -1.0196)),
        ("Cl", (-1.0196, 1.0196, -1.0196)),
        ("Cl", (-1.0196, -1.0196, 1.0196)),
    ],
    # Ni(CO)4: tetrahedral Ni(0) d10 (closed shell), Ni-C = 1.838 Å,
    # C-O = 1.141 Å — the small metal-carbonyl benchmark.
    "NiCO4": [
        ("Ni", (0.0000, 0.0000, 0.0000)),
        ("C", (1.0612, 1.0612, 1.0612)),
        ("C", (1.0612, -1.0612, -1.0612)),
        ("C", (-1.0612, 1.0612, -1.0612)),
        ("C", (-1.0612, -1.0612, 1.0612)),
        ("O", (1.7199, 1.7199, 1.7199)),
        ("O", (1.7199, -1.7199, -1.7199)),
        ("O", (-1.7199, 1.7199, -1.7199)),
        ("O", (-1.7199, -1.7199, 1.7199)),
    ],
    # Ferrocene Fe(C5H5)2: eclipsed D5h, Fe(II) d6 low-spin (closed
    # shell). Fe-Cp(centroid) = 1.65 Å, ring C-C = 1.43 Å,
    # C-H = 1.08 Å. The classic metal-organic-complex benchmark.
    "ferrocene": [
        ("Fe", (0.0000, 0.0000, 0.0000)),
        ("C", (1.2164, 0.0000, 1.6500)),
        ("C", (0.3759, 1.1569, 1.6500)),
        ("C", (-0.9841, 0.7150, 1.6500)),
        ("C", (-0.9841, -0.7150, 1.6500)),
        ("C", (0.3759, -1.1569, 1.6500)),
        ("H", (2.2964, 0.0000, 1.6500)),
        ("H", (0.7096, 2.1840, 1.6500)),
        ("H", (-1.8579, 1.3498, 1.6500)),
        ("H", (-1.8579, -1.3498, 1.6500)),
        ("H", (0.7096, -2.1840, 1.6500)),
        ("C", (1.2164, 0.0000, -1.6500)),
        ("C", (0.3759, 1.1569, -1.6500)),
        ("C", (-0.9841, 0.7150, -1.6500)),
        ("C", (-0.9841, -0.7150, -1.6500)),
        ("C", (0.3759, -1.1569, -1.6500)),
        ("H", (2.2964, 0.0000, -1.6500)),
        ("H", (0.7096, 2.1840, -1.6500)),
        ("H", (-1.8579, 1.3498, -1.6500)),
        ("H", (-1.8579, -1.3498, -1.6500)),
        ("H", (0.7096, -2.1840, -1.6500)),
    ],
    # Naphthalene: D2h, two fused benzene rings sharing the C-C edge
    # along the short axis (y). Ring C-C ~1.42 Å, C-H ~1.08 Å. Common
    # ~20-atom aromatic benchmark.
    "naphthalene": [
        ("C", (0.0000, 0.7080, 0.0000)),   # bridgehead, no H
        ("C", (0.0000, -0.7080, 0.0000)),  # bridgehead, no H
        ("C", (1.2440, 1.4050, 0.0000)),
        ("C", (2.4370, 0.7160, 0.0000)),
        ("C", (2.4370, -0.7160, 0.0000)),
        ("C", (1.2440, -1.4050, 0.0000)),
        ("C", (-1.2440, 1.4050, 0.0000)),
        ("C", (-2.4370, 0.7160, 0.0000)),
        ("C", (-2.4370, -0.7160, 0.0000)),
        ("C", (-1.2440, -1.4050, 0.0000)),
        ("H", (1.2440, 2.4870, 0.0000)),
        ("H", (3.3660, 1.2470, 0.0000)),
        ("H", (3.3660, -1.2470, 0.0000)),
        ("H", (1.2440, -2.4870, 0.0000)),
        ("H", (-1.2440, 2.4870, 0.0000)),
        ("H", (-3.3660, 1.2470, 0.0000)),
        ("H", (-3.3660, -1.2470, 0.0000)),
        ("H", (-1.2440, -2.4870, 0.0000)),
    ],
    # n-Decane (C10H22, 32 atoms): all-trans zigzag, C-C 1.54 Å,
    # C-C-C 113°, C-H 1.09 Å, H-C-H 109.5°. CH2 H atoms in ±y;
    # terminal CH3 H atoms tetrahedrally distributed.
    "n-decane": [
        ("C", (0.0000, 0.0000, 0.0000)),
        ("C", (1.2842, 0.0000, 0.8500)),
        ("C", (2.5684, 0.0000, 0.0000)),
        ("C", (3.8526, 0.0000, 0.8500)),
        ("C", (5.1367, 0.0000, 0.0000)),
        ("C", (6.4209, 0.0000, 0.8500)),
        ("C", (7.7051, 0.0000, 0.0000)),
        ("C", (8.9893, 0.0000, 0.8500)),
        ("C", (10.2735, 0.0000, 0.0000)),
        ("C", (11.5577, 0.0000, 0.8500)),
        ("H", (1.2842, 0.8901, 1.4791)),
        ("H", (1.2842, -0.8901, 1.4791)),
        ("H", (2.5684, 0.8901, -0.6291)),
        ("H", (2.5684, -0.8901, -0.6291)),
        ("H", (3.8526, 0.8901, 1.4791)),
        ("H", (3.8526, -0.8901, 1.4791)),
        ("H", (5.1367, 0.8901, -0.6291)),
        ("H", (5.1367, -0.8901, -0.6291)),
        ("H", (6.4209, 0.8901, 1.4791)),
        ("H", (6.4209, -0.8901, 1.4791)),
        ("H", (7.7051, 0.8901, -0.6291)),
        ("H", (7.7051, -0.8901, -0.6291)),
        ("H", (8.9893, 0.8901, 1.4791)),
        ("H", (8.9893, -0.8901, 1.4791)),
        ("H", (10.2735, 0.8901, -0.6291)),
        ("H", (10.2735, -0.8901, -0.6291)),
        ("H", (-0.3034, 1.0275, -0.2008)),
        ("H", (-0.7945, -0.5137, 0.5412)),
        ("H", (0.1877, -0.5137, -0.9428)),
        ("H", (11.8611, 1.0275, 1.0508)),
        ("H", (12.3522, -0.5137, 0.3088)),
        ("H", (11.3699, -0.5137, 1.7928)),
    ],
    # n-Hexadecane (C16H34, 50 atoms): same all-trans construction
    # as n-decane, 6 more CH2 groups.
    "n-hexadecane": [
        ("C", (0.0000, 0.0000, 0.0000)),
        ("C", (1.2842, 0.0000, 0.8500)),
        ("C", (2.5684, 0.0000, 0.0000)),
        ("C", (3.8526, 0.0000, 0.8500)),
        ("C", (5.1367, 0.0000, 0.0000)),
        ("C", (6.4209, 0.0000, 0.8500)),
        ("C", (7.7051, 0.0000, 0.0000)),
        ("C", (8.9893, 0.0000, 0.8500)),
        ("C", (10.2735, 0.0000, 0.0000)),
        ("C", (11.5577, 0.0000, 0.8500)),
        ("C", (12.8418, 0.0000, 0.0000)),
        ("C", (14.1260, 0.0000, 0.8500)),
        ("C", (15.4102, 0.0000, 0.0000)),
        ("C", (16.6944, 0.0000, 0.8500)),
        ("C", (17.9786, 0.0000, 0.0000)),
        ("C", (19.2628, 0.0000, 0.8500)),
        ("H", (1.2842, 0.8901, 1.4791)),
        ("H", (1.2842, -0.8901, 1.4791)),
        ("H", (2.5684, 0.8901, -0.6291)),
        ("H", (2.5684, -0.8901, -0.6291)),
        ("H", (3.8526, 0.8901, 1.4791)),
        ("H", (3.8526, -0.8901, 1.4791)),
        ("H", (5.1367, 0.8901, -0.6291)),
        ("H", (5.1367, -0.8901, -0.6291)),
        ("H", (6.4209, 0.8901, 1.4791)),
        ("H", (6.4209, -0.8901, 1.4791)),
        ("H", (7.7051, 0.8901, -0.6291)),
        ("H", (7.7051, -0.8901, -0.6291)),
        ("H", (8.9893, 0.8901, 1.4791)),
        ("H", (8.9893, -0.8901, 1.4791)),
        ("H", (10.2735, 0.8901, -0.6291)),
        ("H", (10.2735, -0.8901, -0.6291)),
        ("H", (11.5577, 0.8901, 1.4791)),
        ("H", (11.5577, -0.8901, 1.4791)),
        ("H", (12.8418, 0.8901, -0.6291)),
        ("H", (12.8418, -0.8901, -0.6291)),
        ("H", (14.1260, 0.8901, 1.4791)),
        ("H", (14.1260, -0.8901, 1.4791)),
        ("H", (15.4102, 0.8901, -0.6291)),
        ("H", (15.4102, -0.8901, -0.6291)),
        ("H", (16.6944, 0.8901, 1.4791)),
        ("H", (16.6944, -0.8901, 1.4791)),
        ("H", (17.9786, 0.8901, -0.6291)),
        ("H", (17.9786, -0.8901, -0.6291)),
        ("H", (-0.3034, 1.0275, -0.2008)),
        ("H", (-0.7945, -0.5137, 0.5412)),
        ("H", (0.1877, -0.5137, -0.9428)),
        ("H", (19.5662, 1.0275, 1.0508)),
        ("H", (20.0573, -0.5137, 0.3088)),
        ("H", (19.0750, -0.5137, 1.7928)),
    ],
}

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Element symbol -> atomic number. Covers everything in GEOMETRIES:
# H/C/N/O (the original molecules + alkanes + ferrocene-C/H), plus
# the heavier elements introduced by the extended speed-benchmark
# systems: F (SF6), S (SF6), Cl (CCl4), Fe (ferrocene), Ni (Ni(CO)4).
_Z = {"H": 1, "C": 6, "N": 7, "O": 8,
      "F": 9, "S": 16, "Cl": 17, "Fe": 26, "Ni": 28}


def geometry_bohr_symbols(system: str) -> List[Tuple[str, List[float]]]:
    """``GEOMETRIES[system]`` as ``[(symbol, [x, y, z]_bohr), ...]``.

    The conversion happens **once**, here, with a single constant — and
    both the ORCA input (fed in Bohr via the ``Bohrs`` keyword) and the
    vibe-qc side consume the result. That keeps the two codes on
    *identical* nuclear coordinates, so E_nuc matches to machine
    precision instead of carrying each code's own Angstrom->Bohr
    rounding (a ~4e-8 Ha cross-code shift on H2O otherwise).
    """
    return [
        (sym, [c * ANGSTROM_TO_BOHR for c in xyz])
        for sym, xyz in GEOMETRIES[system]
    ]


def geometry_bohr(system: str) -> List[Tuple[int, List[float]]]:
    """``GEOMETRIES[system]`` as ``[(Z, [x, y, z]_bohr), ...]`` for vibe-qc."""
    return [(_Z[sym], xyz) for sym, xyz in geometry_bohr_symbols(system)]


@dataclass(frozen=True)
class Cell:
    """One parity / benchmark cell.

    Fock-build path: ``df=False, cosx=False`` -> direct 4-index;
    ``df=True`` -> density-fitted (ORCA RIJK / vibe-qc density_fit);
    ``cosx=True`` -> RIJCOSX (RI-J + chain-of-spheres K). ``cosx``
    implies the DF-J machinery, so a RIJCOSX cell is built with
    ``df=True`` too.

    **MP2 cells.** ``method`` may also be ``RMP2`` (closed-shell) or
    ``UMP2`` (open-shell). For an MP2 cell the ``df`` flag selects the
    AO->MO transform path: ``df=False`` -> canonical four-index MP2
    (vibe-qc) / ``! MP2`` (ORCA); ``df=True`` -> RI-MP2 (vibe-qc
    ``MP2Options.density_fit`` / ORCA ``! RI-MP2``). ``cosx`` is
    meaningless for MP2 (no Fock-build K beyond the HF reference) —
    an MP2 cell with ``cosx=True`` is rejected by
    :meth:`__post_init__`.
    """

    system: str
    basis: str
    method: str  # RHF | UHF | RKS-* | UKS-* | RMP2 | UMP2
    charge: int = 0
    spin: int = 0  # 2S: 0 closed-shell, 1 doublet
    df: bool = False  # density-fitted Fock build (SCF) / RI-MP2 transform
    cosx: bool = False  # RIJCOSX (RI-J + chain-of-spheres K); implies df

    def __post_init__(self) -> None:
        if self.cosx and self.is_mp2:
            raise ValueError(
                f"Cell({self.cell_id}): cosx is meaningless for an MP2 "
                f"method — RIJCOSX is an SCF Fock-build path, MP2 has no "
                f"K build beyond its HF reference."
            )

    @property
    def is_mp2(self) -> bool:
        """True for an MP2 / UMP2 correlation cell (vs an SCF cell)."""
        return self.method.upper() in ("RMP2", "UMP2")

    @property
    def cell_id(self) -> str:
        """Filesystem-safe slug — also the cache-JSON basename.

        MP2 cells take a ``__RI`` suffix on the DF path (RI-MP2);
        SCF cells keep the existing ``__DF`` / ``__RIJCOSX`` suffixes.
        """
        if self.is_mp2:
            suffix = "__RI" if self.df else ""
        else:
            suffix = "__RIJCOSX" if self.cosx else ("__DF" if self.df else "")
        return f"{self.system}__{self.basis}__{self.method}{suffix}"

    @property
    def is_open_shell(self) -> bool:
        return (self.spin != 0
                or self.method.upper().startswith("U"))


# --- parity matrix ------------------------------------------------------
# Mirrors test_parity_hf_dft.py's _PARITY_CASES diagonal. ORCA covers
# three Fock-build paths: direct (4-index), DF (RIJK = RI on J and K),
# and RIJCOSX (RI-J + seminumerical chain-of-spheres K). The
# BRIEF.md § Scope-1 like-with-like doctrine: every parity cell runs
# the same algorithm on both sides — vibe-qc-RIJCOSX is compared
# against ORCA-RIJCOSX, not against ORCA's canonical-K-with-DF-J.
# vibe-qc's COSX was implemented to match ORCA 6.1.1, so the ORCA-COSX
# cells certify implementation consistency between the two
# independently-written codes (PySCF has no turnkey RIJCOSX driver,
# so the RIJCOSX axis is ORCA-only on the cross-code matrix).
PARITY_CELLS: List[Cell] = [
    # --- direct Fock build, closed-shell ---
    Cell("H2O", "def2-svp", "RHF"),
    Cell("H2O", "def2-svp", "RKS-PBE"),
    Cell("H2O", "def2-svp", "RKS-B3LYP"),
    Cell("H2O", "def2-svp", "RKS-PBE0"),
    Cell("H2O", "def2-tzvp", "RHF"),
    Cell("H2O", "def2-tzvp", "RKS-PBE"),
    Cell("H2CO", "def2-svp", "RHF"),
    Cell("H2CO", "def2-svp", "RKS-B3LYP"),
    Cell("H2CO", "def2-tzvp", "RKS-PBE0"),
    # --- direct Fock build, open-shell (OH radical, doublet) ---
    Cell("OH", "def2-svp", "UHF", spin=1),
    Cell("OH", "def2-svp", "UKS-PBE", spin=1),
    # --- density-fitted (RIJK) Fock build ---
    # RIJK (RI on J *and* K) needs HF exchange, so the DF cells are HF /
    # hybrid only. A pure-GGA DF cell would have to use ORCA's RIJ
    # (RI-J only) with a different aux-basis slot — ORCA's RIJK +
    # def2/JK errors out for a pure functional ("no AuxJ basis set").
    # DF-J is exercised by every DF cell anyway; the pure-GGA-DF variant
    # adds no Fock-build coverage the three cells below don't already
    # give, so it is intentionally absent.
    Cell("H2O", "def2-svp", "RHF", df=True),
    Cell("H2O", "def2-svp", "RKS-B3LYP", df=True),
    Cell("H2CO", "def2-svp", "RKS-PBE0", df=True),
    Cell("OH", "def2-svp", "UHF", spin=1, df=True),
    # --- RIJCOSX Fock build (RI-J + seminumerical chain-of-spheres K) ---
    # RIJCOSX is K-only-cosx (RI for J, COSX for K) so it requires HF
    # exchange — HF / hybrid only, no pure-GGA cell (where the K piece
    # would be empty anyway and the cell would collapse to plain RI-J).
    # Cells mirror the DF set so each row certifies the same SCF flavour
    # under all three Fock-build paths (direct, DF, RIJCOSX). All cells
    # set ``df=True`` because the cosx machinery rides on top of DF-J.
    Cell("H2O", "def2-svp", "RHF", df=True, cosx=True),
    Cell("H2O", "def2-svp", "RKS-B3LYP", df=True, cosx=True),
    Cell("H2CO", "def2-svp", "RKS-PBE0", df=True, cosx=True),
    Cell("OH", "def2-svp", "UHF", spin=1, df=True, cosx=True),
    # --- multi-heavy-atom open-shell def2-tzvp (BRIEF.md § Scope-2) ---
    # Direct + DF coverage on the three open-shell radicals + OH at
    # def2-tzvp. NO·/def2-tzvp/UHF excluded — vibe-qc's UHF stalls on
    # this 2Π case where PySCF converges in 19 iter (drop-box
    # § Regression finding #2). UHF rows on the other systems route
    # through KDIIS + gtol=1e-6 inside vibeqc_compare's _HARD_OPEN_SHELL
    # path; UKS rows likewise. The pure-GGA-DF caveat (ORCA RIJK needs
    # HF exchange) means no UKS-PBE DF cells — only UHF, UKS-B3LYP get
    # the DF row.
    Cell("NO", "def2-tzvp", "UKS-PBE", spin=1),
    Cell("NO", "def2-tzvp", "UKS-B3LYP", spin=1),
    Cell("CN", "def2-tzvp", "UHF", spin=1),
    Cell("CN", "def2-tzvp", "UKS-PBE", spin=1),
    Cell("CN", "def2-tzvp", "UKS-B3LYP", spin=1),
    Cell("CH3OO", "def2-tzvp", "UHF", spin=1),
    Cell("CH3OO", "def2-tzvp", "UKS-PBE", spin=1),
    Cell("CH3OO", "def2-tzvp", "UKS-B3LYP", spin=1),
    Cell("OH", "def2-tzvp", "UHF", spin=1),
    Cell("OH", "def2-tzvp", "UKS-PBE", spin=1),
    Cell("OH", "def2-tzvp", "UKS-B3LYP", spin=1),
    # def2-tzvp + DF (HF / hybrid only)
    Cell("NO", "def2-tzvp", "UKS-B3LYP", spin=1, df=True),
    Cell("CN", "def2-tzvp", "UHF", spin=1, df=True),
    Cell("CN", "def2-tzvp", "UKS-B3LYP", spin=1, df=True),
    Cell("CH3OO", "def2-tzvp", "UHF", spin=1, df=True),
    Cell("CH3OO", "def2-tzvp", "UKS-B3LYP", spin=1, df=True),
    Cell("OH", "def2-tzvp", "UHF", spin=1, df=True),
    Cell("OH", "def2-tzvp", "UKS-B3LYP", spin=1, df=True),
    # --- MP2 / RI-MP2 correlation cells ---
    # vibe-qc-MP2 vs ORCA-MP2, like-with-like: canonical four-index
    # MP2 on both sides (df=False) and RI-MP2 on both sides (df=True).
    # All-electron - vibe-qc explicitly sets n_frozen_core=0 and ORCA is
    # forced NoFrozenCore (see orca_input.py). Closed-shell RMP2 + open-shell
    # UMP2 (OH· doublet). cc-pVDZ / cc-pVTZ exercise the f-shell-free
    # and f-shell correlation regimes; def2-svp keeps a row on the
    # basis family the rest of the matrix uses.
    Cell("H2O", "def2-svp", "RMP2"),
    Cell("H2O", "cc-pvdz", "RMP2"),
    Cell("H2O", "cc-pvtz", "RMP2", df=True),
    Cell("H2CO", "def2-svp", "RMP2"),
    Cell("OH", "def2-svp", "UMP2", spin=1),
    Cell("OH", "cc-pvtz", "UMP2", spin=1, df=True),
]

# The single cell the handover says to run first, end-to-end, to nail
# the submit -> fetch -> parse -> compare loop before parametrising.
FIRST_CELL = Cell("H2O", "def2-svp", "RHF")


# --- speed benchmark ----------------------------------------------------
# The handover: vibe-qc should be at least as fast as ORCA for the same
# method + basis. Sweep system size — a single small molecule is
# misleading (setup overhead dominates) — and sweep the Fock-build path,
# because that is where the gap lives. The maintainer's own same-box
# measurement (2026-05-14, recorded in benchmarks/orca_vs_vibeqc_speed.md) found vibe-qc
# ~18x slower than ORCA at HF/def2-TZVP/RIJCOSX on a 15-atom molecule;
# the ladder below brackets that with a small system (H2O) and a
# medium one (glycine, 10 atoms / 5 heavy), at direct and RIJCOSX.
SPEED_LADDER_SYSTEMS: List[str] = ["H2O", "glycine"]
# Extended ladder — opt-in via `--systems <comma-list>` on
# run_speed_benchmark.py. Spans 5-50 atoms across organic, aromatic,
# row-3 heavier elements, and metal-organic chemistry. NOT the default
# because the upper end (n-hexadecane B3LYP/def2-TZVP/RIJCOSX,
# ferrocene B3LYP/def2-TZVP/RIJCOSX) takes hours per cell.
#
#   SF6           7 atoms, S+F (row-3), Oh                     -- small + heavier
#   CCl4          5 atoms, C+Cl (row-3), Td                    -- small + heavier
#   NiCO4         9 atoms, Ni(0) d10, Td                       -- metal carbonyl
#   naphthalene  18 atoms, C+H (flat PAH), D2h                 -- ~20-atom aromatic
#   ferrocene    21 atoms, Fe(II) d6 LS, eclipsed D5h          -- metal-organic
#   n-decane     32 atoms, C+H (all-trans alkane)              -- ~30-atom organic
#   n-hexadecane 50 atoms, C+H (all-trans alkane)              -- 50-atom organic
EXTENDED_SPEED_LADDER_SYSTEMS: List[str] = [
    "SF6", "CCl4", "NiCO4", "naphthalene", "ferrocene",
    "n-decane", "n-hexadecane",
]
SPEED_LADDER_BASES: List[str] = ["def2-svp", "def2-tzvp"]
# RHF (no K shortcut) and RKS-B3LYP (the hybrid where RIJCOSX is the
# whole point) span the interesting Fock-build behaviour.
SPEED_LADDER_METHODS: List[str] = ["RHF", "RKS-B3LYP"]
# Fock-build paths to time, per (system, basis, method): "direct"
# (each code's exact 4-index default for these sizes) and "rijcosx"
# (RI-J + chain-of-spheres K — the regime where the gap shows up).
SPEED_LADDER_PATHS: List[str] = ["direct", "rijcosx"]


def speed_cell(system: str, basis: str, method: str, path: str) -> Cell:
    """Build a benchmark :class:`Cell` for one ladder point.

    ``path`` is ``"direct"`` (4-index), ``"df"`` (RIJK), or
    ``"rijcosx"`` (RI-J + chain-of-spheres K).
    """
    if path == "direct":
        return Cell(system=system, basis=basis, method=method)
    if path == "df":
        return Cell(system=system, basis=basis, method=method, df=True)
    if path == "rijcosx":
        return Cell(system=system, basis=basis, method=method,
                    df=True, cosx=True)
    raise ValueError(f"speed_cell: unknown path {path!r}")
