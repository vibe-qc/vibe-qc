#!/usr/bin/env python3
"""COOP/COHP bonding analysis for a 1-D H₂ chain.

Demonstrates::

    compute_coop_cohp()         -- energy-resolved bonding analysis
    periodic_mayer_bond_orders  -- k-space Mayer bond orders
    coop_figure / cohp_figure   -- matplotlib plotters

Requires matplotlib for the plot output.
"""

from __future__ import annotations

import vibeqc as vq

# ---------------------------------------------------------------------------
# 1. Build a 1-D H₂ chain (6 bohr cell, 1.4 bohr bond, 30 bohr vacuum)
# ---------------------------------------------------------------------------
a = 6.0
vac = 30.0
system = vq.PeriodicSystem(
    1,
    [[a, 0.0, 0.0], [0.0, vac, 0.0], [0.0, 0.0, vac]],
    [vq.Atom(1, [0.0, vac / 2, vac / 2]), vq.Atom(1, [1.4, vac / 2, vac / 2])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# ---------------------------------------------------------------------------
# 2. Run periodic SCF with COOP/COHP
# ---------------------------------------------------------------------------
result = vq.run_periodic_job(
    system,
    basis,
    method="RKS",
    functional="lda",
    kpoints=[8, 1, 1],  # 1-D k-mesh
    output="h2_chain_coop",
    output_qvf=True,
    coop_cohp=True,
    dos_kmesh=[32, 1, 1],  # finer mesh for smooth curves
    verbose=1,
)

# ---------------------------------------------------------------------------
# 3. Parse results
# ---------------------------------------------------------------------------
print(f"\nSCF energy: {result.energy:.6f} Ha  converged={result.converged}")
print(f"\nQVF archive written: h2_chain_coop.qvf")
print("Inspect with: vibeqc coop h2_chain_coop.qvf")
print("              vibeqc mayer h2_chain_coop.qvf")

# ---------------------------------------------------------------------------
# 4. Optional: plot COOP/COHP (requires matplotlib)
# ---------------------------------------------------------------------------
try:
    # The QVF already has a post-SCF analysis. This separate calculation shows
    # the explicit-block API with the teaching-only Hcore operator.
    import matplotlib.pyplot as plt
    from vibeqc.plot import cohp_figure, coop_figure

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # We can't easily get the Fock terms from the runner result here,
    # so we demonstrate with a Hcore computation instead.
    opts = vq.LatticeSumOptions()
    S = vq.compute_overlap_lattice(basis, system, opts)
    T = vq.compute_kinetic_lattice(basis, system, opts)
    V = vq.compute_nuclear_lattice(basis, system, opts)
    km = vq.monkhorst_pack(system, [32, 1, 1])

    cc = vq.compute_coop_cohp(
        [T, V],
        S,
        system,
        basis,
        km,
        H_terms=[T, V],
        sigma=0.05,
        n_grid=301,
        n_electrons_per_cell=2,
    )

    coop_figure(cc, ax=ax1, title="COOP (Hcore)")
    cohp_figure(cc, ax=ax2, title="-COHP (Hcore)")

    fig.tight_layout()
    fig.savefig("h2_chain_coop_cohp.png", dpi=150)
    print("Plot saved: h2_chain_coop_cohp.png")
    plt.close()

except ImportError:
    print("(matplotlib not available; skipping plot)")
