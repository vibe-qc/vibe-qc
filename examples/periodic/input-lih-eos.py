"""LiH rocksalt equation of state: five lattice constants and a BM3 fit.

System
    LiH in the rocksalt structure, FCC primitive cell (one Li at the origin,
    one H at the cell-body centre), scanned over the conventional lattice
    constant a = 2.8, 3.0, 3.2, 3.4, 3.6 angstrom (primitive volume a^3/4).

Method and basis
    RHF/STO-3G at the Gamma point on the production Gaussian density-fitting
    route (`jk_method="gdf"`, default rsgdf backend, JK auxiliary basis
    autodetected). Gamma-only sampling of a two-atom cell and a minimal basis
    are not converged settings; they keep every point to one SCF of 6 to 7
    iterations, and the fitted numbers are a property of this model
    chemistry, not a prediction for LiH. The five energies are fitted with
    `vibeqc.eos.fit_birch_murnaghan`, a third-order Birch-Murnaghan fit whose
    status turns to `ill_conditioned_fit` when the energy span is tiny or the
    residual is a large fraction of it, and which reports separately whether
    the fitted minimum lies inside the scanned volumes (`minimum_bracketed`).

    Why the scan sits at 2.8 to 3.6 angstrom and not around the experimental
    4.08: for Gamma-only RHF/STO-3G the energy falls monotonically as the
    cell is compressed from 4.6 to 3.4 angstrom (E = -8.2572, -8.2875,
    -8.3178, -8.3475, -8.3753, -8.3997 and -8.4179 Ha at 4.6, 4.4, 4.2, 4.0,
    3.8, 3.6 and 3.4 angstrom). The rise of about 0.03 Ha per 0.2
    angstrom for a of 4.0 and above is of the order of the Madelung
    attraction of a +1/-1 point-charge lattice (about 0.02 Ha per step), and
    the curve turns over only below about 3.4 angstrom, which describes this
    model chemistry rather than LiH. A scan around 4.0 has no minimum, and
    `fit_birch_murnaghan` says so: it reports `minimum_bracketed = False`
    (its fit of the 3.6 to 4.4 points extrapolates to a minimum near 3.5
    angstrom that lies outside the data). The script therefore checks
    `minimum_bracketed` and refuses to report V0 otherwise.

Files (next to this script)
    input-lih-eos-a2.80.* ... input-lih-eos-a3.60.*, one family per lattice
    constant (.out, .system, .molden, .population.txt / .population.json,
    .bibtex, .references); the QVF archive and structure sidecars are
    switched off for the scan points. The fit summary goes to stdout.

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    The BM3 fit reports `status == "fit_succeeded"` with the minimum
    bracketed by the scan. All five SCFs converge in 6 or 7 iterations, and
    the lowest energy is at 3.2 angstrom (-8.426084 Ha). Obtained:
    E0 = -8.426176 Ha, V0 = 54.388 bohr^3 (a0 = 3.182 angstrom),
    B0 = 253 GPa, B0' = 3.50, RMSE = 1.95e-5 Ha over an energy span of
    4.155e-2 Ha (rmse/span = 4.7e-4). These describe Gamma-only RHF/STO-3G:
    a0 is far below the experimental 4.08 angstrom and B0 far above the
    experimental bulk modulus. The script asserts the fit status, the
    bracketing, and |a0 - 3.18| < 0.05 angstrom.
    Cost: the five points took 2313 s (2.8), 1471 s (3.0), 981 s (3.2),
    653 s (3.4) and 431 s (3.6 angstrom) of wall time, 5853 s in total
    (11594 s of user CPU with 6 threads), on a shared host whose load average
    was between about 70 and 200 during the run. The cost falls as the cell
    grows (in a separate set of parallel runs on the same host the user CPU
    fell from 2567 s at 2.8 angstrom to 815 s at 3.6), and no idle-machine
    timing was taken.

Run:
    .venv/bin/python examples/periodic/input-lih-eos.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.eos import FIT_SUCCEEDED, fit_birch_murnaghan

HERE = Path(__file__).parent
STEM = Path(__file__).stem

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
GPA_PER_HARTREE_PER_BOHR3 = 29421.02648438959  # CODATA 2018 atomic unit of pressure

A_SCAN = (2.8, 3.0, 3.2, 3.4, 3.6)  # conventional lattice constant, angstrom


def lih_primitive(a_ang: float) -> vq.PeriodicSystem:
    """Rocksalt LiH, FCC primitive cell for conventional lattice constant a."""
    a = a_ang * BOHR_PER_ANGSTROM
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    body_centre = 0.5 * (lattice[0] + lattice[1] + lattice[2])
    return vq.PeriodicSystem(
        dim=3,
        lattice=lattice,
        unit_cell=[vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, body_centre.tolist())],
    )


volumes, energies = [], []
print(f"{'a (A)':>6}  {'V (bohr^3)':>11}  {'E (Ha)':>16}  iters  wall")
for a_ang in A_SCAN:
    system = lih_primitive(a_ang)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    t0 = time.perf_counter()
    res = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gdf",
        output=str(HERE / f"{STEM}-a{a_ang:.2f}"),
        output_qvf=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )
    wall = time.perf_counter() - t0
    volume = abs(float(np.linalg.det(np.asarray(system.lattice))))
    volumes.append(volume)
    energies.append(float(res.energy))
    print(f"{a_ang:6.2f}  {volume:11.4f}  {res.energy:16.8f}  {res.n_iter:5d}  {wall:5.1f} s")
    assert res.converged, f"SCF did not converge at a = {a_ang}"

fit = fit_birch_murnaghan(volumes, energies)
print()
print(f"BM3 fit status: {fit['status']}  (model {fit['model']})")
if fit["status"] != FIT_SUCCEEDED:
    raise SystemExit(f"EOS fit not usable: {fit.get('reason')}")

v0 = fit["v0_bohr3"]
a0_ang = (4.0 * v0) ** (1.0 / 3.0) / BOHR_PER_ANGSTROM
if not fit["minimum_bracketed"]:
    raise SystemExit(
        f"the fitted minimum (V0 = {v0:.2f} bohr^3, a0 = {a0_ang:.2f} angstrom) "
        f"lies outside the scanned volumes {min(volumes):.1f} to "
        f"{max(volumes):.1f} bohr^3; move A_SCAN before trusting any EOS parameter"
    )

b0_gpa = fit["b0_hartree_per_bohr3"] * GPA_PER_HARTREE_PER_BOHR3
print(f"  E0  = {fit['e0_hartree']:.8f} Ha")
print(f"  V0  = {v0:.4f} bohr^3  ->  a0 = {a0_ang:.4f} angstrom "
      "(RHF/STO-3G at Gamma; experiment is 4.08 angstrom)")
print(f"  B0  = {b0_gpa:.1f} GPa   B0' = {fit['b0_prime']:.3f}")
print(f"  fit quality: RMSE = {fit['rmse_hartree']:.2e} Ha over a span of "
      f"{fit['energy_span_hartree']:.3e} Ha (rmse/span = {fit['rmse_to_span']:.3e}); "
      f"minimum bracketed = {fit['minimum_bracketed']}")

assert abs(a0_ang - 3.18) < 0.05, "fitted lattice constant moved"
print("OK")
