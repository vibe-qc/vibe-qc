"""vibe-qc quickstart — first 30 minutes after installation.

Walks through the four canonical first-time-user steps:

  1. Molecular SCF      — closed-shell RHF on H2O / 6-31G*.
  2. Open-shell SCF     — UHF on the OH radical to show how spin lives
                          on the Molecule (multiplicity), not on the
                          driver options.
  3. Periodic SCF       — bulk LiH via the EWALD_3D dispatcher
                          (run_rhf_periodic_gamma_scf).
  4. Orbital cube file  — write the H2O HOMO to a .cube readable by
                          VMD / PyMOL / Avogadro.

Output files are written to /tmp/vibeqc_quickstart/. Everything in
this script "just works" out of a fresh `pip install -e .` from the
vibe-qc checkout — no LIBINT_DATA_PATH to set, no separate basis-set
fetch.

Run:
    .venv/bin/python examples/quickstart.py

Paired with the user-facing walkthrough in docs/quickstart.md.
"""

import os
from pathlib import Path
import time

import numpy as np
import vibeqc as vq

OUTDIR = Path("/tmp/vibeqc_quickstart")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _bundled_basis_count() -> int:
    """How many .g94 files are reachable via the libint data path
    vibe-qc set up at import time. Works equally well for the bundled
    package layout, the legacy editable-install layout, or a user
    override via $LIBINT_DATA_PATH."""
    root = os.environ.get("LIBINT_DATA_PATH")
    if not root:
        return 0
    g94 = list(Path(root, "basis").glob("*.g94"))
    return len(g94)


def step_1_molecular_hf() -> None:
    """RHF on H2O — the smallest non-trivial molecular calculation."""
    print("\n  [1/4] H2O / RHF / 6-31G* — closed-shell molecular SCF")
    mol = vq.Molecule([
        vq.Atom(8, [ 0.0,  0.00,  0.00]),
        vq.Atom(1, [ 0.0,  1.43, -0.98]),
        vq.Atom(1, [ 0.0, -1.43, -0.98]),
    ])
    basis = vq.BasisSet(mol, "6-31g*")
    t0 = time.perf_counter()
    result = vq.run_rhf(mol, basis)
    dt = time.perf_counter() - t0
    eps = np.asarray(result.mo_energies)
    homo = mol.n_electrons() // 2 - 1
    print(f"     E(SCF)    = {result.energy:.6f} Ha   "
          f"({result.n_iter} iters, {dt*1000:.0f} ms)")
    print(f"     ε(HOMO)   = {eps[homo]*27.211386:7.2f} eV")
    print(f"     ε(LUMO)   = {eps[homo+1]*27.211386:7.2f} eV")
    print(f"     gap       = {(eps[homo+1] - eps[homo])*27.211386:5.2f} eV")
    return mol, basis, result


def step_2_open_shell() -> None:
    """UHF on OH — spin lives on Molecule.multiplicity, not on UHFOptions."""
    print("\n  [2/4] OH• / UHF / 6-31G* — open-shell radical")
    oh = vq.Molecule(
        atoms=[vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.832])],
        multiplicity=2,                          # doublet, one unpaired e-
    )
    basis = vq.BasisSet(oh, "6-31g*")
    t0 = time.perf_counter()
    result = vq.run_uhf(oh, basis)
    dt = time.perf_counter() - t0
    print(f"     E(UHF)    = {result.energy:.6f} Ha   "
          f"({result.n_iter} iters, {dt*1000:.0f} ms)")
    print(f"     ⟨S²⟩       ≈ {0.75:.3f}  (exact for a pure doublet; UHF "
          f"contamination shows up here)")


def step_3_periodic_scf() -> None:
    """Bulk LiH at the Γ point via the EWALD_3D dispatcher.

    Uses the 2-atom CsCl-type cubic cell (one Li at the origin, one H
    at the cube center) — not the actual LiH rocksalt structure, but
    a simple 2-atom periodic cell that runs in seconds. The point of
    this step is to demonstrate the dispatcher API and Ewald summation,
    not LiH thermochemistry. For real LiH workflows, use the rocksalt
    primitive (FCC + 2-atom basis) — see tutorial 4.
    """
    print("\n  [3/4] 2-atom LiH cubic / RHF / sto-3g via EWALD_3D dispatcher")
    a = 4.5                                      # bohr, tight cell
    shift = 0.05                                 # off-FFT-grid shift
    sysp = vq.PeriodicSystem(
        dim=3, lattice=np.eye(3) * a,
        unit_cell=[
            vq.Atom(3, [shift,         shift,         shift        ]),
            vq.Atom(1, [a/2 + shift,   a/2 + shift,   a/2 + shift  ]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0         # was 12 — quickstart cuts cost
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.max_iter = 30

    t0 = time.perf_counter()
    # progress=True streams a flushed per-iteration trace (banner +
    # one line per SCF step) to stdout. The cost is essentially zero;
    # leave it on by default for any user-facing run so the SCF is
    # never silently grinding.
    result = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5, progress=True,
    )
    dt = time.perf_counter() - t0
    print(f"     E(RHF)/cell = {result.energy:.6f} Ha   "
          f"(2 atoms, {result.n_iter} iters, {dt:.1f} s)")
    print(f"     converged   = {result.converged}")
    return sysp, basis, result


def step_4_orbital_cube(mol, basis, result) -> None:
    """Write the H2O HOMO as a Gaussian cube file."""
    print("\n  [4/4] Write H2O HOMO to a .cube file")
    homo_idx = mol.n_electrons() // 2 - 1
    out = OUTDIR / "h2o_homo.cube"
    t0 = time.perf_counter()
    vq.write_cube_mo(
        str(out),
        np.asarray(result.mo_coeffs), homo_idx, basis, mol,
        spacing=0.2, padding=3.0,
    )
    dt = time.perf_counter() - t0
    print(f"     wrote {out}  ({out.stat().st_size // 1024} KB, {dt*1000:.0f} ms)")
    print(f"     Open with:   open -a Avogadro2 {out}")
    print(f"                  vmd {out}")


def main() -> None:
    print("=" * 64)
    print(" vibe-qc quickstart  —  pip install → 4 calculations in <1 min")
    print("=" * 64)
    print(f" Output files: {OUTDIR}")
    print(f" vibe-qc version: {vq.VIBEQC_VERSION}")
    print(f" Basis sets available out of the box: {_bundled_basis_count()}")
    t_total = time.perf_counter()

    mol, basis, h2o_result = step_1_molecular_hf()
    step_2_open_shell()
    step_3_periodic_scf()
    step_4_orbital_cube(mol, basis, h2o_result)

    print(f"\n  Total wall: {time.perf_counter() - t_total:.1f} s")
    print(f"  Open the cube file in your viewer of choice and you're "
          f"officially using vibe-qc.")


if __name__ == "__main__":
    main()
