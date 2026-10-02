"""1D H2 molecular crystal — periodic RHF, k-mesh convergence.

Run:
    .venv/bin/python input-h-chain-uniform.py

Produces:
    output-h-chain-uniform.out   — banner + SCF trace for each k-mesh
                                    + table of energies vs k-mesh size

The unit cell holds a single H2 molecule (bond length 1.4 bohr)
separated from its periodic images by 13.6 bohr. Lattice vector
``a = 15 bohr`` along x. The chain is a ``dim=3`` cell with 15 bohr of
vacuum in y and z, run with k-point GDF (``run_krhf_periodic_gdf``):
vibe-qc has no route with a consistent 1-D Coulomb gauge for a
``PeriodicSystem(dim=1, ...)``, so the vacuum box is the supported way
to sample k along a wire. The vacuum is part of the model; rerun with
a larger ``VACUUM`` to see how much the energy per cell moves.

Each H2 is a closed-shell bonded pair, so this system is a trivial
band insulator — SCF converges in a handful of iterations. It's the
right first periodic system to look at: the main thing to see is how
the energy per cell converges with the mesh. It converges as 1/nk:
each doubling of nk halves the change (2.3e-4 Ha between 8 and 16
points on vibe-qc 0.17.14). That tail is the expected exchange
finite-size term of the BvK-ewald gauge (#456), so 2 E(2n) - E(n)
removes it; the script prints that extrapolation. The extrapolated
value is the limit of this box, which still differs from an isolated
H2 by the interaction with its transverse images.

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
    PeriodicRHFOptions,
    PeriodicSystem,
    banner,
    format_scf_trace,
    run_krhf_periodic_gdf,
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
# The chain runs along the first lattice vector; the k mesh samples only
# that axis, and the other two lattice vectors are vacuum.
A = 15.0                                   # lattice parameter, bohr
R_HH = 1.4                                 # H2 intra-molecular bond, bohr
VACUUM = 15.0                              # yz vacuum (bohr)
lattice = np.diag([A, VACUUM, VACUUM])
unit_cell = [
    Atom(1, [0.0,   0.0, 0.0]),
    Atom(1, [R_HH, 0.0, 0.0]),
]

sysp = PeriodicSystem(
    dim=3,
    lattice=lattice,
    unit_cell=unit_cell,
    charge=0,
    multiplicity=1,
)

basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")

opts = PeriodicRHFOptions()
opts.conv_tol_energy = 1e-10
opts.conv_tol_grad = 1e-8
opts.max_iter = 100

# --- run ------------------------------------------------------------------
with open(OUT, "w", encoding="utf-8") as f:
    f.write(banner() + "\n\n")
    f.write("  1D H2 molecular crystal, multi-k RHF (k-point GDF)\n")
    f.write(f"  lattice a = {A} bohr (H2 bond {R_HH} bohr, "
            f"inter-molecular gap {A - R_HH} bohr)\n")
    f.write(f"  basis = pob-tzvp   vacuum (yz) = {VACUUM} bohr\n")
    f.write(f"  n_electrons/cell = {sysp.n_electrons()}\n\n")

    energies = []
    for nk in (1, 2, 4, 8, 16):
        result = run_krhf_periodic_gdf(sysp, basis, (nk, 1, 1), opts,
                                       progress=False)

        f.write(f"  --- k-mesh = [{nk}, 1, 1]  ({nk} k-points) ---\n")
        f.write(format_scf_trace(result, include_banner=False) + "\n\n")
        energies.append((nk, nk, result.energy, result.converged))

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
        # The exchange finite-size tail falls as 1/nk, so doubling the mesh
        # halves it and 2 E(2n) - E(n) cancels it.
        e_lim = 2.0 * energies[-1][2] - energies[-2][2]
        f.write(f"  2 E[{energies[-1][0]}] - E[{energies[-2][0]}] = "
                f"{e_lim:.10f} Ha (1/nk extrapolation, this box)\n")

print(f"Done. See {OUT.name}.")
