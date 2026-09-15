"""Generate XSF test files for moltui using vibe-qc's native writers.

Creates files in moltui/tests/test_data/ that mirror what real
vibe-qc periodic calculations produce:
  - NaCl (3D bulk) — structure-only + structure+density
  - SiO2 alpha-quartz (3D bulk, hexagonal) — structure+density
  - Graphene (2D) — structure only
  - 1D H-chain — structure only (vacuum test)
  - Multi-grid file — structure+density+homo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.xsf import write_xsf_structure, write_xsf_volume

HERE = Path(__file__).resolve().parent
MOLTUI_TEST_DATA = HERE.parent / "moltui" / "tests" / "test_data"
MOLTUI_TEST_DATA.mkdir(parents=True, exist_ok=True)

ANG2BOHR = 1.0 / 0.529177210903
BOHR2ANG = 0.529177210903


def _make_grid(shape: tuple[int, int, int], *, kind: str = "gaussian") -> np.ndarray:
    """Synthetic volumetric data with a recognisable feature."""
    n1, n2, n3 = shape
    if kind == "gaussian":
        # Gaussian blob at centre
        xi = np.linspace(-1, 1, n1)
        yi = np.linspace(-1, 1, n2)
        zi = np.linspace(-1, 1, n3)
        xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
        r2 = xx**2 + yy**2 + zz**2
        return np.exp(-3.0 * r2)
    elif kind == "alternating":
        # Alternating pattern for MO-like data
        data = np.zeros((n1, n2, n3), dtype=np.float64)
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    data[i, j, k] = (-1.0) ** (i + j + k)
        return data
    else:
        return np.ones(shape, dtype=np.float64)


# ── 1. NaCl 3D bulk, structure-only ─────────────────────────────────
def nacl_structure_only():
    a = 5.640 * ANG2BOHR  # lattice parameter in bohr
    atoms = [
        Atom(11, [0.0, 0.0, 0.0]),
        Atom(17, [a / 2, a / 2, a / 2]),
    ]
    system = PeriodicSystem(3, np.diag([a, a, a]), atoms)
    write_xsf_structure(MOLTUI_TEST_DATA / "nacl_structure_vibeqc.xsf", system)
    print("✓ NaCl structure-only")


# ── 2. NaCl 3D bulk, structure + density ────────────────────────────
def nacl_with_density():
    a = 5.640 * ANG2BOHR
    atoms = [
        Atom(11, [0.0, 0.0, 0.0]),
        Atom(17, [a / 2, a / 2, a / 2]),
    ]
    system = PeriodicSystem(3, np.diag([a, a, a]), atoms)
    shape = (20, 20, 20)
    data = _make_grid(shape, kind="gaussian")
    write_xsf_volume(
        MOLTUI_TEST_DATA / "nacl_density_vibeqc.xsf",
        system,
        data=data,
        name="density",
        origin=np.zeros(3),
        span=np.diag([a, a, a]),
    )
    print("✓ NaCl with density")


# ── 3. SiO2 alpha-quartz (hexagonal, non-orthogonal) ─────────────────
def sio2_quartz():
    a = 4.913 * ANG2BOHR
    c = 5.405 * ANG2BOHR
    # Hexagonal lattice: a1 = (a, 0, 0), a2 = (-a/2, a*sqrt(3)/2, 0), a3 = (0, 0, c)
    lattice = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2, a * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, c],
        ]
    )
    # SiO2: 3 Si + 6 O in primitive cell (simplified positions)
    atoms = [
        # Si at u=0.47
        Atom(14, [0.47 * a, 0.0, 0.0]),
        Atom(14, [-0.47 * a / 2, 0.47 * a * np.sqrt(3) / 2, c / 3]),
        Atom(14, [-0.47 * a / 2, -0.47 * a * np.sqrt(3) / 2, 2 * c / 3]),
        # O near Si
        Atom(8, [0.41 * a, 0.27 * a, 0.12 * c]),
        Atom(8, [0.27 * a, 0.41 * a, -0.12 * c]),
        Atom(
            8,
            [
                -0.41 * a / 2 - 0.27 * a,
                0.41 * a * np.sqrt(3) / 2 - 0.27 * a * np.sqrt(3) / 2,
                c / 3,
            ],
        ),
        Atom(
            8,
            [
                -0.27 * a / 2 - 0.41 * a,
                0.27 * a * np.sqrt(3) / 2 - 0.41 * a * np.sqrt(3) / 2,
                c / 3,
            ],
        ),
        Atom(
            8,
            [
                -0.41 * a / 2 + 0.16 * a,
                0.41 * a * np.sqrt(3) / 2 + 0.16 * a * np.sqrt(3) / 2,
                2 * c / 3,
            ],
        ),
        Atom(
            8,
            [
                -0.27 * a / 2 + 0.16 * a,
                0.27 * a * np.sqrt(3) / 2 + 0.16 * a * np.sqrt(3) / 2,
                2 * c / 3,
            ],
        ),
    ]
    system = PeriodicSystem(3, lattice.T, atoms)
    shape = (16, 16, 12)
    data = _make_grid(shape, kind="gaussian")
    write_xsf_volume(
        MOLTUI_TEST_DATA / "sio2_quartz_vibeqc.xsf",
        system,
        data=data,
        name="density",
        origin=np.zeros(3),
        span=lattice.T,
    )
    print("✓ SiO2 quartz (hexagonal) with density")


# ── 4. Graphene (2D) — structure only ────────────────────────────────
def graphene():
    a = 2.46 * ANG2BOHR
    vacuum = 20.0 * ANG2BOHR
    lattice = np.array(
        [
            [a, 0.0, 0.0],
            [a / 2, a * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, vacuum],
        ]
    )
    atoms = [
        Atom(6, [0.0, 0.0, vacuum / 2]),
        Atom(6, [a / 2, a * np.sqrt(3) / 6, vacuum / 2]),
    ]
    system = PeriodicSystem(2, lattice.T, atoms)
    write_xsf_structure(MOLTUI_TEST_DATA / "graphene_vibeqc.xsf", system)
    print("✓ Graphene 2D structure")


# ── 5. 1D H-chain (vacuum test) ─────────────────────────────────────
def h_chain():
    a = 4.0 * ANG2BOHR
    vacuum = 30.0 * ANG2BOHR
    lattice = np.diag([a, vacuum, vacuum])
    atoms = [
        Atom(1, [0.0, vacuum / 2, vacuum / 2]),
        Atom(1, [a / 2, vacuum / 2, vacuum / 2]),
    ]
    system = PeriodicSystem(1, lattice, atoms)
    write_xsf_structure(MOLTUI_TEST_DATA / "h_chain_vibeqc.xsf", system)
    print("✓ 1D H-chain structure")


# ── 6. Multi-grid file (density + homo + lumo) ─────────────────────
def multi_grid():
    a = 5.640 * ANG2BOHR
    atoms = [
        Atom(11, [0.0, 0.0, 0.0]),
        Atom(17, [a / 2, a / 2, a / 2]),
    ]
    system = PeriodicSystem(3, np.diag([a, a, a]), atoms)
    shape = (12, 12, 12)

    # Write a multi-grid XSF manually (vibe-qc's write_xsf_volume only does one)
    origin_ang = np.zeros(3)
    span_ang = np.diag([a, a, a]) * BOHR2ANG

    from vibeqc.xsf import _atoms_block, _datagrid_block, _lattice_block

    with open(MOLTUI_TEST_DATA / "nacl_multi_vibeqc.xsf", "w") as f:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        f.write(_lattice_block(system) + "\n")
        f.write("PRIMCOORD\n")
        f.write(f"{len(system.unit_cell):5d} 1\n")
        f.write(_atoms_block(system) + "\n")
        f.write("\n")
        f.write("BEGIN_BLOCK_DATAGRID_3D\n")

        for name, kind in [
            ("density", "gaussian"),
            ("homo", "alternating"),
            ("lumo", "gaussian"),
        ]:
            f.write(f"  {name}\n")
            data = _make_grid(shape, kind=kind)
            if name == "lumo":
                data = 1.0 - data  # invert for lumo
            f.write(
                _datagrid_block(
                    name, data, origin_ang, span_ang[0], span_ang[1], span_ang[2]
                )
            )

        f.write("END_BLOCK_DATAGRID_3D\n")
    print("✓ NaCl multi-grid (density + homo + lumo)")


if __name__ == "__main__":
    nacl_structure_only()
    nacl_with_density()
    sio2_quartz()
    graphene()
    h_chain()
    multi_grid()
    print(f"\nAll files written to {MOLTUI_TEST_DATA}")
