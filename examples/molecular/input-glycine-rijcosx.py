"""Glycine B3LYP/def2-SVP: conventional four-index SCF versus RIJCOSX.

System
    Glycine (C2H5NO2, 10 atoms, closed shell) at the B3LYP/6-31G* optimised
    geometry used by the RIJCOSX tutorial (docs/tutorial/rijcosx_glycine.md).
    The coordinates in input-glycine-rhf-direct.py are not reused: taken at
    face value they do not have glycine's bonding pattern (C-C 1.61
    angstrom, O-H 1.59 angstrom, N-H 1.24 angstrom), and a B3LYP SCF on them
    had not converged after 74 iterations.

Method and basis
    RKS with the B3LYP hybrid functional in def2-SVP (95 basis functions),
    once with the conventional in-core four-index Fock build
    (`density_fit=False`) and once as RIJCOSX (`density_fit=True, cosx=True`):
    RI-J for the Coulomb matrix with the auxiliary basis autodetected from
    the orbital basis name (def2-SVP -> def2-SVP-JK, printed below) and the
    chain-of-spheres seminumerical exchange on the default COSX grid.
    Both are production molecular routes.

Files (next to this script)
    input-glycine-rijcosx-conventional.* and input-glycine-rijcosx-rijcosx.*,
    one family per SCF, written by `run_job`: .out, .system, .qvf, .molden,
    .xyz, .population.txt / .population.json, .bibtex and .references.

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    Both SCFs converge and agree to well under a millihartree:
    E(conventional) = -284.058380 Ha (13 iterations) and E(RIJCOSX) =
    -284.058438 Ha (17 iterations), a difference of -0.058 mHa, which is the
    RI-J plus COSX grid error at this basis. The script asserts |dE| < 1 mHa.
    Wall times are printed for the two builds. Measured on a shared 18-core
    host with a load average above 200 they were 215 s (conventional) and
    170 s (RIJCOSX), so the printed ratio of 1.27 is not a clean speedup and
    no idle-machine timing was taken. def2-SVP on ten atoms is small enough
    that the conventional build is not yet the bottleneck RIJCOSX is meant to
    remove; the tutorial shows the larger def2-TZVP case.

Run:
    .venv/bin/python examples/molecular/input-glycine-rijcosx.py
"""

from __future__ import annotations

import time
from pathlib import Path

import vibeqc as vq
from vibeqc.molecule import ANGSTROM_TO_BOHR

HERE = Path(__file__).parent
STEM = Path(__file__).stem

# Glycine, B3LYP/6-31G* optimised geometry in angstrom, the geometry of the
# tutorial docs/tutorial/rijcosx_glycine.md. vq.Atom stores bohr.
GLYCINE_ANGSTROM = [
    (7, [-1.5012, -0.1265, 0.0000]),
    (6, [-0.4080, 0.7959, 0.0000]),
    (6, [0.9189, 0.0670, 0.0000]),
    (8, [1.0150, -1.1438, 0.0000]),
    (8, [1.9583, 0.7993, 0.0000]),
    (1, [-2.3543, 0.4169, 0.0000]),
    (1, [-1.5012, -0.7196, 0.8189]),
    (1, [-0.4773, 1.4360, 0.8843]),
    (1, [-0.4773, 1.4360, -0.8843]),
    (1, [2.7892, 0.2748, 0.0000]),
]
glycine = vq.Molecule(
    [vq.Atom(z, [x * ANGSTROM_TO_BOHR for x in xyz]) for z, xyz in GLYCINE_ANGSTROM]
)

BASIS = "def2-svp"
print(f"glycine  RKS/B3LYP  basis={BASIS}  "
      f"JK auxiliary autodetected as {vq.default_aux_basis_for(BASIS, kind='jk')!r}")

runs = {}
for label, kwargs in (
    ("conventional", dict(density_fit=False)),
    ("rijcosx", dict(density_fit=True, cosx=True)),
):
    t0 = time.perf_counter()
    result = vq.run_job(
        glycine,
        basis=BASIS,
        method="rks",
        functional="b3lyp",
        output=str(HERE / f"{STEM}-{label}"),
        **kwargs,
    )
    wall = time.perf_counter() - t0
    runs[label] = (result, wall)
    print(f"{label:>12}: E = {result.energy:16.8f} Ha   "
          f"{result.n_iter:2d} iterations   converged={result.converged}   "
          f"wall {wall:7.1f} s")

e_conv, w_conv = runs["conventional"][0].energy, runs["conventional"][1]
e_cosx, w_cosx = runs["rijcosx"][0].energy, runs["rijcosx"][1]
delta_mha = 1000.0 * (e_cosx - e_conv)
print()
print(f"E(RIJCOSX) - E(conventional) = {delta_mha:+.3f} mHa   "
      f"(reference build: -0.058 mHa)")
print(f"wall conventional / RIJCOSX  = {w_conv / w_cosx:.2f}x")

assert runs["conventional"][0].converged and runs["rijcosx"][0].converged
assert abs(delta_mha) < 1.0, "RIJCOSX deviates from the four-index energy by more than 1 mHa"
print("OK")
