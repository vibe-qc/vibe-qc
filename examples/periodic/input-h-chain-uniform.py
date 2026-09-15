"""1D H2 molecular crystal — periodic RHF, k-mesh convergence.

Run:
    .venv/bin/python input-h-chain-uniform.py

Produces:
    output-h-chain-uniform.out   — banner + SCF trace for each k-mesh
                                    + table of energies vs k-mesh size

The unit cell holds a single H2 molecule (bond length 1.4 bohr)
separated from its periodic images by 4.6 bohr. Lattice vector
``a = 6 bohr`` along x; 30 bohr of vacuum in y and z turns the
periodic 1D chain into an isolated wire with no spurious inter-wire
interactions. Multi-k RHF (``run_rhf_periodic_scf``, the new
Coulomb-method dispatcher) gives the total energy per unit cell.

Each H2 is a closed-shell bonded pair, so this system is a trivial
band insulator — SCF converges in a handful of iterations. It's the
right first periodic system to look at: the main thing to see is
that multi-k energies stop changing once the mesh is fine enough.

The uniform 1D H-chain (equal H-H spacing) is *metallic* at almost
any density and famously Peierls-unstable. That's the subject of
``input-h-chain-peierls.py`` — here we avoid it deliberately by
starting from molecular H2 units.
"""

from pathlib import Path

import numpy as np

from vibeqc import (
    Atom,
    BasisSet,
    PeriodicSCFOptions,
    PeriodicSystem,
    banner,
    format_scf_trace,
    monkhorst_pack,
    run_rhf_periodic_scf,
)

HERE = Path(__file__).parent
OUT = HERE / "output-h-chain-uniform.out"

# --- system setup ---------------------------------------------------------
# H2 molecular crystal: one H2 per cell (bonded pair, 1.4 bohr apart),
# cells 15 bohr apart along the chain axis. Closed shell, insulating, and
# comfortably in vibe-qc's molecular-limit regime — each H2 barely sees its
# periodic neighbors, so SCF converges in ~10 iterations like a normal
# molecule. Shrink A to ~6 bohr to push into the real-periodic regime once
# you want to watch SCF convergence get harder.
#
# For dim=1 the *first* lattice vector is the periodic direction; the
# rest are implicit vacuum.
A = 15.0                                   # lattice parameter, bohr
R_HH = 1.4                                 # H2 intra-molecular bond, bohr
VACUUM = 30.0                              # yz vacuum (bohr)
lattice = np.diag([A, VACUUM, VACUUM])
unit_cell = [
    Atom(1, [0.0,   0.0, 0.0]),
    Atom(1, [R_HH, 0.0, 0.0]),
]

sysp = PeriodicSystem(
    dim=1,
    lattice=lattice,
    unit_cell=unit_cell,
    charge=0,
    multiplicity=1,
)

basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")

# Conservative lattice-sum cutoffs; large enough for STO-3G-level accuracy
# and small enough that the run finishes in a reasonable time per k-mesh.
opts = PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr = 15.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.conv_tol_energy = 1e-10
opts.conv_tol_grad = 1e-8
opts.max_iter = 100

# --- run ------------------------------------------------------------------
with open(OUT, "w", encoding="utf-8") as f:
    f.write(banner() + "\n\n")
    f.write("  1D H2 molecular crystal, multi-k RHF\n")
    f.write(f"  lattice a = {A} bohr (H2 bond {R_HH} bohr, "
            f"inter-molecular gap {A - R_HH} bohr)\n")
    f.write(f"  basis = pob-tzvp   vacuum (yz) = {VACUUM} bohr\n")
    f.write(f"  n_electrons/cell = {sysp.n_electrons()}\n\n")

    energies = []
    for nk in (1, 2, 4, 8, 16):
        km = monkhorst_pack(sysp, [nk, 1, 1])
        # CoulombMethod.DIRECT_TRUNCATED (the default on opts.lattice_opts)
        # routes through the legacy run_rhf_periodic backend; switch to
        # CoulombMethod.EWALD_3D for any quantitative 3D-bulk run.
        result = run_rhf_periodic_scf(sysp, basis, km, opts)

        f.write(f"  --- k-mesh = [{nk}, 1, 1]  "
                f"({len(km.kpoints)} k-points) ---\n")
        f.write(format_scf_trace(result, include_banner=False) + "\n\n")
        energies.append((nk, len(km.kpoints), result.energy, result.converged))

    f.write("  k-mesh convergence:\n")
    f.write("  " + "-" * 46 + "\n")
    f.write(f"  {'nk':>3}  {'n_kpts':>6}  {'E per cell (Ha)':>20}  conv\n")
    f.write("  " + "-" * 46 + "\n")
    for nk, n_kpts, e, conv in energies:
        tag = "yes" if conv else "NO"
        f.write(f"  {nk:3d}  {n_kpts:6d}  {e:20.10f}  {tag}\n")

    # Energy difference between the two densest meshes — the
    # extrapolated thermodynamic-limit residual.
    if len(energies) >= 2:
        de = energies[-1][2] - energies[-2][2]
        f.write(f"\n  E[{energies[-1][0]}] - E[{energies[-2][0]}] = "
                f"{de:+.3e} Ha (Hartree-Fock extrapolation residual)\n")

print(f"Done. See {OUT.name}.")
