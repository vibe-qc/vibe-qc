"""Geometry definitions for the canonical periodic test systems.

Each entry returns a tuple ``(label, atoms_ase)`` where ``atoms_ase``
is an ``ase.Atoms`` object. The vibe-qc and PySCF input scripts then
convert as needed.

Lattice parameters and Wyckoff positions from Springer Materials.
"""
from __future__ import annotations

from ase import Atoms
from ase.spacegroup import crystal as ase_crystal


def MgO_rocksalt(a_ang: float = 4.211) -> Atoms:
    """MgO conventional cubic rocksalt (Fm-3m, #225). a = 4.211 Å."""
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    syms = ["Mg"] * 4 + ["O"] * 4
    cell = [[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]]
    pos = [(fx*a_ang, fy*a_ang, fz*a_ang) for fx, fy, fz in mg_frac + o_frac]
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


def LiH_rocksalt(a_ang: float = 4.084) -> Atoms:
    """LiH conventional cubic rocksalt (Fm-3m, #225). a = 4.084 Å."""
    li_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    h_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    syms = ["Li"] * 4 + ["H"] * 4
    cell = [[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]]
    pos = [(fx*a_ang, fy*a_ang, fz*a_ang) for fx, fy, fz in li_frac + h_frac]
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


def NaCl_rocksalt(a_ang: float = 5.640) -> Atoms:
    """NaCl conventional cubic rocksalt (Fm-3m, #225). a = 5.640 Å."""
    na_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    cl_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    syms = ["Na"] * 4 + ["Cl"] * 4
    cell = [[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]]
    pos = [(fx*a_ang, fy*a_ang, fz*a_ang) for fx, fy, fz in na_frac + cl_frac]
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


def Ne_fcc(a_ang: float = 4.43) -> Atoms:
    """Ne conventional cubic FCC (Fm-3m, #225). a = 4.43 Å."""
    fcc_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    cell = [[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]]
    pos = [(fx*a_ang, fy*a_ang, fz*a_ang) for fx, fy, fz in fcc_frac]
    return Atoms(symbols=["Ne"]*4, positions=pos, cell=cell, pbc=True)


def Si_diamond(a_ang: float = 5.431) -> Atoms:
    """Si conventional cubic diamond (Fd-3m, #227). a = 5.431 Å."""
    return ase_crystal(
        ["Si"], basis=[(0, 0, 0)],
        spacegroup=227, cellpar=[a_ang, a_ang, a_ang, 90, 90, 90],
        setting=2,
    )


def C_diamond(a_ang: float = 3.567) -> Atoms:
    """C conventional cubic diamond (Fd-3m, #227). a = 3.567 Å."""
    return ase_crystal(
        ["C"], basis=[(0, 0, 0)],
        spacegroup=227, cellpar=[a_ang, a_ang, a_ang, 90, 90, 90],
        setting=2,
    )


def ZnO_wurtzite(a_ang: float = 3.2495, c_ang: float = 5.2069,
                 u: float = 0.382) -> Atoms:
    """ZnO wurtzite (P6_3 m c, #186). a = 3.2495 Å, c = 5.2069 Å, u ≈ 0.382."""
    return ase_crystal(
        ["Zn", "O"],
        basis=[(1/3, 2/3, 0.0), (1/3, 2/3, u)],
        spacegroup=186,
        cellpar=[a_ang, a_ang, c_ang, 90, 90, 120],
    )


def TiO2_rutile(a_ang: float = 4.5937, c_ang: float = 2.9587,
                x: float = 0.3053) -> Atoms:
    """TiO2 rutile (P4_2/mnm, #136). a = 4.5937 Å, c = 2.9587 Å."""
    return ase_crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (x, x, 0)],
        spacegroup=136,
        cellpar=[a_ang, a_ang, c_ang, 90, 90, 90],
    )


def alpha_SiO2_quartz(a_ang: float = 4.9134, c_ang: float = 5.4052,
                      u_si: float = 0.4697,
                      x_o: float = 0.4135, y_o: float = 0.2669,
                      z_o: float = 0.1191) -> Atoms:
    """α-quartz (P3_2 21, #154). a = 4.9134 Å, c = 5.4052 Å.
    Si at 3a (u, 0, 1/3); O at 6c (x, y, z).
    """
    return ase_crystal(
        ["Si", "O"],
        basis=[(u_si, 0, 1/3), (x_o, y_o, z_o)],
        spacegroup=154,
        cellpar=[a_ang, a_ang, c_ang, 90, 90, 120],
    )


def alpha_Al2O3_corundum(a_ang: float = 4.7589, c_ang: float = 12.991,
                         al_z: float = 0.35216,
                         o_x: float = 0.30624) -> Atoms:
    """α-Al2O3 corundum (R-3c, #167)."""
    return ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, al_z), (o_x, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_ang, a_ang, c_ang, 90, 90, 120],
    )


# Registry: (slug, builder) — slug is also the parent directory name
# under examples/periodic/<slug>/ and examples/periodic_pyscf/<slug>/.
# Order matters for sequential orchestration: smallest first so a
# slow heavy system at the end doesn't block reporting for the rest.
SYSTEMS = [
    ("LiH-rocksalt",       LiH_rocksalt),       # 8 atoms, ~52 AOs
    ("Ne-fcc",             Ne_fcc),             # 4 atoms, ~56 AOs
    ("ZnO-wurtzite",       ZnO_wurtzite),       # 4 atoms, ~116 AOs
    ("NaCl-rocksalt",      NaCl_rocksalt),      # 8 atoms, ~140 AOs
    ("TiO2-rutile",        TiO2_rutile),        # 6 atoms, ~152 AOs
    ("SiO2-alpha-quartz",  alpha_SiO2_quartz),  # 9 atoms, ~174 AOs
    ("C-diamond",          C_diamond),          # 8 atoms, ~120 AOs
    ("Si-diamond",         Si_diamond),         # 8 atoms, ~176 AOs
]
