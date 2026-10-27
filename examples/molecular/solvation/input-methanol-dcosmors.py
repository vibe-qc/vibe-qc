"""COSMO-RS on both densities, and an infinite-dilution activity coefficient.

Methanol in water at the recommended protocol (BPW91/def2-TZVP, fine cavity):

1. ``density="conductor"`` (the default) evaluates the solvation free energy on
   the ideally screened conductor density (Klamt 1998) and, with
   ``activity_coefficient=True``, also reports ln gamma-infinity of methanol
   in water against pure methanol.
2. ``density="direct"`` runs methanol in Direct COSMO-RS with the same water
   potential and evaluates the free energy on that self-consistent density
   (Klamt & Diedenhofen 2015 eq. 9). The direct density evaluates the
   potential on raw segment sigma, which converges with the cavity grid only
   near 0.25 A, so that run passes the conductor model at that spacing; the
   water surface it needs is then computed on the same grid.

The water surface is computed once at this protocol and cached under
``~/.cache/vibeqc/cosmors``; the second job reads it back.

Run from the repository root::

    .venv/bin/python examples/molecular/solvation/input-methanol-dcosmors.py
"""

import math

import vibeqc as vq
from vibeqc.solvation.cosmors import KLAMT_1998

ANG = 1.8897261254578281
methanol = vq.Molecule([
    vq.Atom(6, [-0.0471 * ANG, 0.6644 * ANG, 0.0]),
    vq.Atom(8, [-0.0471 * ANG, -0.7586 * ANG, 0.0]),
    vq.Atom(1, [-1.0930 * ANG, 0.9698 * ANG, 0.0]),
    vq.Atom(1, [0.8785 * ANG, -1.0485 * ANG, 0.0]),
    vq.Atom(1, [0.4371 * ANG, 1.0804 * ANG, 0.8918 * ANG]),
    vq.Atom(1, [0.4371 * ANG, 1.0804 * ANG, -0.8918 * ANG]),
], 0, 1)

common = dict(basis="def2-tzvp", method="rks", functional="bpw91")

conductor = vq.run_job(
    methanol,
    cosmors=vq.CosmoRSOptions(solvent="water", activity_coefficient=True),
    output="methanol_cosmors_conductor",
    **common,
).cosmors

converged_grid = vq.SolventModel(
    epsilon=math.inf, variant="cosmo", cavity="fine",
    radii=dict(KLAMT_1998.cavity_radii), radii_scale=1.0,
    solvent_probe_radius_ang=0.0, fine_grid_spacing_ang=0.25,
)
direct = vq.run_job(
    methanol,
    cosmors=vq.CosmoRSOptions(solvent="water", density="direct"),
    solvent=converged_grid,
    output="methanol_cosmors_direct",
    **common,
).cosmors

print("reference states: ideal gas at 1 bar -> 1 mol/mol in water")
print(f"dG_solv, conductor density = {conductor.dg_solv:8.3f} kcal/mol")
print(f"dG_solv, direct density    = {direct.dg_solv:8.3f} kcal/mol "
      f"({direct.n_macro_iter} macro-iterations)")
print(f"ln gamma_inf (methanol in water, vs pure methanol) = "
      f"{conductor.ln_gamma_inf:6.3f} "
      f"(residual {conductor.ln_gamma_inf_residual:.3f}, "
      f"combinatorial {conductor.ln_gamma_inf_combinatorial:.3f})")
