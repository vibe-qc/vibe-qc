"""Ewald-summed Madelung constants for classic ionic crystals.

Run:
    .venv/bin/python input-madelung-constants.py

Produces:
    output-madelung-constants.out   — table of extracted Madelung
                                        constants vs. textbook values

The Madelung constant captures the electrostatic cohesion of an ionic
lattice of point charges — a foundational number in solid-state
physics. Its value is independent of the lattice constant: it's a
pure geometric sum over charge-weighted distances, and the direct
real-space series is only conditionally convergent.

vibe-qc ships a production Ewald summer that computes the absolutely-
convergent result to eight-digit agreement with textbook values. This
script wraps ``ewald_point_charge_energy`` (3D) and extracts M from

    E_per_ion-pair = - M / r_nn

for three archetypal crystals:

    NaCl (rocksalt)         M = 1.7475645946
    CsCl                    M = 1.7626747730
    ZnS (zincblende)        M = 1.6380550533

It then does the same in **two dimensions** with the rigorous 2D (slab)
Ewald summer ``ewald_2d_point_charge_energy`` (Parry 1975; de Leeuw &
Perram 1979) — the *true* 2D lattice sum, no vacuum-gap parameter — for
the alternating ±1 square ionic lattice:

    2D square lattice       M = 1.6155426267

Good first demo for the periodic machinery: no SCF, no basis sets,
just the Madelung constant landing on the textbook value. (The 2D
``ewald_2d_point_charge_energy`` primitive is the foundation for the
``CoulombMethod.SLAB_EWALD_2D`` SCF path — see
``handovers/HANDOVER_SLAB_EWALD_2D.md``.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vibeqc import (
    EwaldOptions,
    banner,
    ewald_2d_point_charge_energy,
    ewald_point_charge_energy,
)

HERE = Path(__file__).parent
OUT = HERE / "output-madelung-constants.out"

BOHR_PER_A = 1.0 / 0.529177210903


def nacl(a_angstrom: float = 5.6):
    """Rocksalt: FCC Bravais, Na at origin, Cl at (a/2, 0, 0)."""
    a = a_angstrom * BOHR_PER_A
    lattice = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 0, 0])])
    charges = np.array([+1.0, -1.0])
    r_nn = 0.5 * a                       # nearest-neighbor distance
    return lattice, positions, charges, r_nn, "NaCl", 1.7475645946, 3


def cscl(a_angstrom: float = 4.1):
    """Simple cubic lattice with 2-atom basis, Cs at corner, Cl at body center."""
    a = a_angstrom * BOHR_PER_A
    lattice = a * np.eye(3)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 1, 1])])
    charges = np.array([+1.0, -1.0])
    r_nn = 0.5 * a * np.sqrt(3)          # body-diagonal half-length
    return lattice, positions, charges, r_nn, "CsCl", 1.7626747730, 3


def zns(a_angstrom: float = 5.4):
    """Zincblende (sphalerite): FCC Zn, FCC S displaced by (1/4, 1/4, 1/4) a."""
    a = a_angstrom * BOHR_PER_A
    lattice = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.25 * a * np.array([1, 1, 1])])
    charges = np.array([+1.0, -1.0])
    r_nn = 0.25 * a * np.sqrt(3)
    return lattice, positions, charges, r_nn, "ZnS", 1.6380550533, 3


def square_2d(d_angstrom: float = 1.0):
    """Alternating ±1 square ionic lattice (2D / slab geometry).

    The checkerboard's like-charge sublattice has primitive vectors
    (d, d) and (d, −d); the basis is +1 at (0,0) and −1 at (d, 0), so the
    nearest-neighbour distance is d. The third lattice column is a vacuum
    axis (ignored by the 2D sum). M_2D = −E · d, like the 3D cases.
    """
    d = d_angstrom * BOHR_PER_A
    lattice = np.column_stack([[d, d, 0.0], [d, -d, 0.0], [0.0, 0.0, 60.0]])
    positions = np.column_stack([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    charges = np.array([+1.0, -1.0])
    return lattice, positions, charges, d, "2D square", 1.6155426267, 2


opts = EwaldOptions()
opts.real_cutoff_bohr = 40.0             # large enough for 8-digit accuracy
# alpha and reciprocal cutoff auto-selected when left at defaults

rows = []
for builder in (nacl, cscl, zns, square_2d):
    lattice, positions, charges, r_nn, name, reference, dim = builder()
    # 3D crystals use the standard Ewald sum; the 2D slab lattice uses the
    # rigorous 2D (Parry / de Leeuw–Perram) sum — same per-cell convention.
    ewald = ewald_point_charge_energy if dim == 3 else ewald_2d_point_charge_energy
    energy = ewald(lattice, positions, charges, opts)
    madelung = -energy * r_nn            # energy per ion-pair
    rows.append((name, madelung, reference, madelung - reference))


with open(OUT, "w", encoding="utf-8") as f:
    f.write(banner() + "\n\n")
    f.write("  Ewald-summed Madelung constants for classic ionic crystals\n")
    f.write(f"  real_cutoff = {opts.real_cutoff_bohr} bohr; alpha and "
            f"reciprocal cutoff auto-selected\n\n")
    f.write("  " + "-" * 58 + "\n")
    f.write(f"  {'crystal':<10s}  {'vibe-qc M':>16s}  {'reference M':>14s}  "
            f"{'|diff|':>10s}\n")
    f.write("  " + "-" * 58 + "\n")
    for name, m_vibeqc, m_ref, diff in rows:
        f.write(f"  {name:<10s}  {m_vibeqc:16.10f}  {m_ref:14.10f}  "
                f"{abs(diff):10.2e}\n")
    f.write("  " + "-" * 58 + "\n")
    f.write("\n  Reference values:\n")
    f.write("    NaCl      : 1.7475645946  (Crandall, J. Phys. A 20 (1987) 2123)\n")
    f.write("    CsCl      : 1.7626747730  (Madelung via Ewald, e.g. Tosi 1964)\n")
    f.write("    ZnS       : 1.6380550533  (Sakamoto, J. Sci. Hiroshima Univ. 1955)\n")
    f.write("    2D square : 1.6155426267  (2D Ewald; Parry 1975, de Leeuw-Perram 1979)\n")

print(f"Done. See {OUT.name}.")
