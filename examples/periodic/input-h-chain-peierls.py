"""1D H-chain: Peierls dimerisation scan at RHF / pob-TZVP.

Run:
    .venv/bin/python input-h-chain-peierls.py

Produces:
    output-h-chain-peierls.out   — banner + SCF summary for each delta
                                    + table of E(delta)

Scans the dimerisation coordinate ``delta`` on a two-H-per-cell chain
with fixed lattice parameter ``a = 5 bohr``. At ``delta = 0`` the two
atoms are evenly spaced (a/2 = 2.5 bohr apart, a half-filled band:
metallic in principle). As ``delta`` grows, one H-H distance shrinks
toward the H2 equilibrium (1.4 bohr) and the other stretches; the
energy drops and the chain becomes an insulator.

This is the classic Peierls instability: a half-filled 1D band is
unstable to a periodic distortion that opens a gap, lowering the
occupied-state energies. The output table makes the effect visible
as a monotonic drop in E(delta) with increasing delta up to the
H2-molecular-crystal endpoint.

The chain is modelled as a ``dim=3`` cell with vacuum on the two
transverse axes. vibe-qc has no route with a consistent 1-D Coulomb
gauge, so a ``PeriodicSystem(dim=1, ...)`` cannot be run with GDF; the
vacuum box is the supported way to sample k along a wire. The vacuum
``PAD`` is part of the model. Rerun with a larger ``PAD`` to check how
much the energies move.

This is a manual scan at fixed lattice parameter, not a geometry
optimisation.
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
OUT = HERE / "output-h-chain-peierls.out"

A = 5.0                                    # lattice parameter, bohr
PAD = 15.0                                 # transverse vacuum (y, z), bohr
KMESH = (8, 1, 1)                          # k sampling along the chain only


def build_system(delta: float) -> PeriodicSystem:
    """Two-H-per-cell chain with dimerisation offset ``delta``.

    Uniform: delta = 0 (H-H = a/2 both intra- and inter-cell).
    Dimerised: delta > 0 (short bond = a/2 - delta, long bond = a/2 + delta).
    """
    lattice = np.diag([A, PAD, PAD])
    unit_cell = [
        Atom(1, [0.0,               0.0, 0.0]),
        Atom(1, [A / 2.0 - delta,   0.0, 0.0]),
    ]
    return PeriodicSystem(dim=3, lattice=lattice, unit_cell=unit_cell)


def scf_opts() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    # Relax conv_tol_grad beyond the default: near the metallic end the HF
    # gradient settles slowly, but the energy stabilises to 1e-8 long
    # before that matters for the dimerisation physics we want to show.
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-3
    opts.max_iter = 80
    return opts


with open(OUT, "w", encoding="utf-8") as f:
    f.write(banner() + "\n\n")
    f.write("  1D H-chain Peierls-dimerisation scan (HF / pob-tzvp, k-point GDF)\n")
    f.write(f"  lattice a = {A} bohr    k-mesh = {list(KMESH)}    "
            f"transverse vacuum = {PAD} bohr\n")
    f.write("  For each delta: short = a/2 - delta, long = a/2 + delta (bohr)\n\n")

    results = []
    for delta in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1):
        sysp = build_system(delta)
        basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
        # Pass ``progress=True`` to get a flushed per-iteration trace
        # (useful for slow runs); skipped here because the loop runs seven
        # SCFs and would be noisy.
        result = run_krhf_periodic_gdf(sysp, basis, KMESH, scf_opts(),
                                       progress=False)

        r_short = A / 2.0 - delta
        r_long = A / 2.0 + delta
        results.append((delta, r_short, r_long, result.energy, result.converged,
                        result.n_iter))

        f.write(f"  --- delta = {delta:.2f}   short = {r_short:.3f}   "
                f"long = {r_long:.3f} ---\n")
        f.write(format_scf_trace(result, include_banner=False) + "\n\n")

    f.write("  Dimerisation energy scan:\n")
    f.write("  " + "-" * 64 + "\n")
    f.write(f"  {'delta':>6}  {'R_short':>7}  {'R_long':>7}  "
            f"{'E (Ha)':>18}  {'n_iter':>6}  conv\n")
    f.write("  " + "-" * 64 + "\n")
    for delta, rs, rl, e, conv, n_iter in results:
        tag = "yes" if conv else "NO"
        f.write(f"  {delta:6.2f}  {rs:7.3f}  {rl:7.3f}  "
                f"{e:18.10f}  {n_iter:6d}  {tag}\n")

    # Highlight the lowest-energy point so the Peierls minimum pops out.
    converged = [r for r in results if r[4]]
    if converged:
        best = min(converged, key=lambda r: r[3])
        f.write(f"\n  Lowest converged energy at delta = {best[0]:.2f} bohr "
                f"(short = {best[1]:.3f}, long = {best[2]:.3f})\n")
        f.write(f"  E_min = {best[3]:.6f} Ha per cell\n")

print(f"Done. See {OUT.name}.")
