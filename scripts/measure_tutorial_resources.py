"""Measure peak RAM + wall time for the canonical calculation in
each tutorial that doesn't yet have a Resources callout.

Used by the doc-lead chat to fill out the standardized
``**Resources.** ...`` line in tutorials 01-17, 19. Not part of the
tutorial flow itself.

Run:
    python3 scripts/measure_tutorial_resources.py
"""

import gc
import resource
import time
from typing import Callable

import numpy as np
import vibeqc as vq


def _peak_mb() -> float:
    """Resident-set size peak in MB. Linux returns kB; macOS returns
    bytes — dispatch on platform."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: bytes; Linux: kB. Use a heuristic.
    if rss > 1e9:           # > 1 GB ⇒ bytes ⇒ macOS
        return rss / (1024 * 1024)
    return rss / 1024        # kB ⇒ MB


def _measure(label: str, fn: Callable[[], None]) -> None:
    gc.collect()
    rss_pre = _peak_mb()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    rss_post = _peak_mb()
    delta = max(0.0, rss_post - rss_pre)
    print(f"  {label:<48}  Δpeak RAM = {delta:6.1f} MB  "
          f"wall = {dt:7.2f} s   peak RSS = {rss_post:7.1f} MB")


# --- Molecular SCFs --------------------------------------------------------
H2O = [
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
]


def case_h2o_hf_sto3g():
    mol = vq.Molecule(H2O)
    basis = vq.BasisSet(mol, "sto-3g")
    vq.run_rhf(mol, basis)


def case_h2o_hf_6_31g_star():
    mol = vq.Molecule(H2O)
    basis = vq.BasisSet(mol, "6-31g*")
    vq.run_rhf(mol, basis)


def case_h2o_dft_pbe_6_31g_star():
    mol = vq.Molecule(H2O)
    basis = vq.BasisSet(mol, "6-31g*")
    opts = vq.RKSOptions(); opts.functional = "PBE"
    vq.run_rks(mol, basis, opts)


def case_oh_uhf_6_31g_star():
    oh = vq.Molecule(
        [vq.Atom(8, [0,0,0]), vq.Atom(1, [0,0,1.832])],
        multiplicity=2,
    )
    basis = vq.BasisSet(oh, "6-31g*")
    vq.run_uhf(oh, basis)


def case_h2o_cube_density():
    mol = vq.Molecule(H2O)
    basis = vq.BasisSet(mol, "6-31g*")
    r = vq.run_rhf(mol, basis)
    grid = vq.make_uniform_grid(mol, spacing_bohr=0.2, padding_bohr=3.0)
    vq.write_cube_density("/tmp/h2o_density.cube", mol, basis, r.density, grid)


def case_h2o_basis_convergence():
    mol = vq.Molecule(H2O)
    for b in ("sto-3g", "6-31g", "6-31g*", "6-31g**"):
        basis = vq.BasisSet(mol, b)
        vq.run_rhf(mol, basis)


def case_h2_chain_uniform():
    sysp = vq.PeriodicSystem(
        1, [[15.0, 0, 0], [0, 30, 0], [0, 0, 30]],
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [4, 1, 1])
    vq.run_rhf_periodic_scf(sysp, basis, km)


def case_h2_chain_dft():
    sysp = vq.PeriodicSystem(
        1, [[15.0, 0, 0], [0, 30, 0], [0, 0, 30]],
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions(); opts.functional = "PBE"
    km = vq.monkhorst_pack(sysp, [4, 1, 1])
    vq.run_rks_periodic_scf(sysp, basis, km, opts)


def case_madelung_nacl():
    a = 5.6 / 0.529177210903
    lattice = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float).T
    positions = np.column_stack([[0, 0, 0], [0.5 * a, 0, 0]])
    charges = np.array([+1.0, -1.0])
    eopts = vq.EwaldOptions(); eopts.real_cutoff_bohr = 30.0
    vq.ewald_point_charge_energy(lattice, positions, charges, eopts)


def case_h_chain_bands_dos():
    sysp = vq.PeriodicSystem(
        1, [[6.0, 0, 0], [0, 30, 0], [0, 0, 30]],
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kpath = vq.kpath_from_segments(
        sysp, [((0, 0, 0), "Γ", (0.5, 0, 0), "X")], points_per_segment=120,
    )
    vq.band_structure_hcore(sysp, basis, kpath, n_electrons_per_cell=2)
    vq.density_of_states_hcore(sysp, basis, [200, 1, 1],
                               sigma=0.01, n_electrons_per_cell=2)


CASES = [
    ("01 H2O / HF / sto-3g",                case_h2o_hf_sto3g),
    ("01-15 H2O / HF / 6-31g*",             case_h2o_hf_6_31g_star),
    ("02 H2O / RKS-PBE / 6-31g*",           case_h2o_dft_pbe_6_31g_star),
    ("03 OH radical / UHF / 6-31g*",        case_oh_uhf_6_31g_star),
    ("11 H2O cube density (0.2 bohr grid)", case_h2o_cube_density),
    ("13 H2O basis-convergence STO-3G→6-31G**", case_h2o_basis_convergence),
    ("04 H2-chain RHF, k=[4,1,1] sto-3g",   case_h2_chain_uniform),
    ("05 H2-chain RKS-PBE, k=[4,1,1] sto-3g", case_h2_chain_dft),
    ("06 NaCl Madelung Ewald",               case_madelung_nacl),
    ("12 H-chain bands+DOS (Hcore)",         case_h_chain_bands_dos),
]


def main() -> None:
    print(f"{'Case':<48}  {'Δpeak RAM':>12}  {'Wall':>10}  {'Peak RSS':>11}")
    print("-" * 88)
    for label, fn in CASES:
        try:
            _measure(label, fn)
        except Exception as e:
            print(f"  {label:<48}  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
