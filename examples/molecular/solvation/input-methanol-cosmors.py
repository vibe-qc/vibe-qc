"""COSMO-RS solvation free energy of methanol in water, solvent by name.

The solvent's conductor surface is computed once at the solute's protocol
(BPW91/def2-TZVP, the recommended one) and cached under
``~/.cache/vibeqc/cosmors``; a second run reads it back. Ring atoms are
perceived from covalent-radius bonds (none here).

Run from the repository root::

    .venv/bin/python examples/molecular/solvation/input-methanol-cosmors.py
"""

import vibeqc as vq

ANG = 1.8897261254578281
methanol = vq.Molecule([
    vq.Atom(6, [-0.0471 * ANG, 0.6644 * ANG, 0.0]),
    vq.Atom(8, [-0.0471 * ANG, -0.7586 * ANG, 0.0]),
    vq.Atom(1, [-1.0930 * ANG, 0.9698 * ANG, 0.0]),
    vq.Atom(1, [0.8785 * ANG, -1.0485 * ANG, 0.0]),
    vq.Atom(1, [0.4371 * ANG, 1.0804 * ANG, 0.8918 * ANG]),
    vq.Atom(1, [0.4371 * ANG, 1.0804 * ANG, -0.8918 * ANG]),
], 0, 1)

result = vq.run_job(
    methanol,
    basis="def2-tzvp",
    method="rks",
    functional="bpw91",
    cosmors=vq.CosmoRSOptions(solvent="water", n_ring_atoms="auto"),
    output="methanol_cosmors",
)

cr = result.cosmors
print(f"solvent surface      : {cr.solvent_name} ({cr.solvent_source})")
print(f"mu_S  (in water)     = {cr.mu_solvent.total:9.3f} kcal/mol")
print(f"mu_gas               = {cr.mu_gas:9.3f} kcal/mol")
print(f"dG_solv (1 bar -> 1 mol/mol) = {cr.dg_solv:9.3f} kcal/mol")
print(f"surface beyond fitted sigma range: {100 * cr.fraction_outside_grid:.1f} %")
