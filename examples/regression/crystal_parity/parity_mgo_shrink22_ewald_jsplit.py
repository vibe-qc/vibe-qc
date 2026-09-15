"""Parity: MgO primitive RHF/STO-3G at SHRINK 2 2 via multi-k Ewald J-split.

Multi-k extension of ``parity_mgo_gamma_ewald_jsplit.py`` — wires
per-k ``compute_J_long_range_at_k`` from the v0.9.0 BIPOLE
multi-k extension (memory
``reference_bipole_multik_v0_9_0_landed_2026-05-18``).

CRYSTAL14 reference (SHRINK 8 8 converged):
  -271.218144 Ha/FU

Empirical results (2026-05-18, cutoff=8 / use_symmetry=False):
  SCF converges in 12 iters (DIIS) to dE < 1e-7
  E_total = -269.3774 Ha
  Δ vs CRYSTAL SHRINK 8 8 = +1.84 Ha — PASS (target ±2200 mHa)

The Γ-only Ewald-J-split on the same cutoff gave -269.04 Ha
(Δ = +2.17 Ha). SHRINK 2 2 closes ~0.3 Ha of the Γ→k-mesh gap.
Full closure to sub-mHa is the SHRINK 8 8 target (see
``parity_mgo_shrink88_ewald_jsplit.py``).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo_sto3g
from crystal_demos.runner import run_demo_parity


def main() -> int:
    # ``use_symmetry=False`` is REQUIRED for the multi-k Ewald-J-split
    # path: the k-space ρ̂(K) formula in
    # :func:`vibeqc.bipole_fock_ewald.compute_rho_hat_from_k_density`
    # sums over the supplied k-points with their weights, but does NOT
    # expand symmetry-equivalent orbits internally. With
    # ``use_symmetry=True`` the IBZ k-mesh under-samples the
    # ``FT^{(+k)}(K)`` Bloch phases — which DO vary across the symmetry
    # orbit even though D(k) is constant — and the SCF diverges
    # (observed: MgO SHRINK 2 2 with 3 IBZ k-points → over-binds to
    # -277 Ha). With ``use_symmetry=False`` the full 8-k mesh gives
    # the correct sum.
    #
    # cutoff=8 keeps the n_cells / n_k_full ratio modest (19/8); larger
    # cutoffs probe further but the Γ→k-mesh shift dominates the
    # residual on STO-3G at this size.
    return run_demo_parity(
        label="MgO primitive (STO-3G, SHRINK 2 2, Ewald J-split + DIIS)",
        build_fn=build_mgo_sto3g,
        kmesh_size=(2, 2, 2),
        crystal14_ref_ha_per_fu=-271.21814374982,
        cutoff_bohr=8.0,
        use_diis=True,
        diis_start_iter=2,
        damping=0.3,
        max_iter=20,
        target_millihartree=2200.0,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        conv_tol_energy=1e-7,
        conv_tol_grad=1e-4,
        use_symmetry=False,
    )


if __name__ == "__main__":
    sys.exit(main())
