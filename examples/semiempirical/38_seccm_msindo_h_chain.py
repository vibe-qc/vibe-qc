"""SECCM (semiempirical cyclic cluster model) on a 1-D LiH chain with MSINDO.

Status: experimental. SECCM as a whole is an experimental line in the
project's status vocabulary; the MSINDO route used here is its most mature
member, and its closed-shell energies are validated against reference MSINDO
within the documented scope. The label comes from that vocabulary, not from a
runtime signal: this run raises no experimental warning, the route plan that
`run_job` attaches to the result reports `maturity = "production"` for the
MSINDO SECCM energy route, and the `.system` manifest records no status tag.
Treat the numbers as research output rather than as a release-gated result.

System
    An infinite, uniform ...Li-H-Li-H... chain with 1.6 angstrom Li-H spacing,
    represented as a cyclic cluster of n LiH units (n = 2, 3, 4, 5) closed on
    itself by one translation vector of length 2 n x 1.6 angstrom (CCM1D). A
    heteronuclear chain is used instead of a uniform H chain because the
    energy per atom of a uniform H chain does not converge monotonically with
    cluster size in this model: at 1.0 angstrom spacing it is -0.507, -0.548
    and -0.529 Ha for 4, 6 and 8 atoms. The ionic LiH chain converges with n
    and exercises the Madelung embedding.

Method and basis
    `run_job(mol, method="seccm", ccm_options=CCMOptions(...))`: closed-shell
    MSINDO in the cyclic cluster model with Wigner-Seitz image weighting and,
    for `madelung=True`, the 1-D finite Madelung sum. MSINDO carries its own
    minimal valence basis; no `basis=` is passed. The topology diagnostics are
    read from `vibeqc.semiempirical.seccm.build_wigner_seitz`, and the n = 5
    cluster is repeated with `madelung=False` through the library entry point
    `run_ccm` to isolate the embedding energy.

Files (next to this script)
    38_seccm_msindo_h_chain-lih<n>.out / .system / .qvf / .xyz / .bibtex /
    .references for each cyclic cluster, written by `run_job`.

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    Every cluster is a valid cyclic cluster (sum of Wigner-Seitz weights
    equals N_atoms - 1 for every atom). The energy per LiH unit converges
    with n: -0.833121, -0.849955, -0.849155, -0.849038 Ha for n = 2, 3, 4, 5,
    so |E/unit(5) - E/unit(4)| = 1.2e-4 Ha; the script asserts that this
    difference is below 5e-4 Ha and that the n = 5 value lies within 1e-5 Ha
    of -0.849038 Ha. The Madelung embedding lowers the n = 5 cluster by
    1.45e-3 Ha relative to the truncated (`madelung=False`) run. Each run_job
    call reported a job total of 1.5 to 3.1 s.

Run:
    .venv/bin/python examples/semiempirical/38_seccm_msindo_h_chain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.molecule import ANGSTROM_TO_BOHR
from vibeqc.semiempirical.methods.msindo_ccm import CCMOptions, run_ccm
from vibeqc.semiempirical.seccm import build_wigner_seitz

HERE = Path(__file__).parent
STEM = Path(__file__).stem

D_LIH = 1.6  # Li-H spacing along the chain, angstrom


def lih_chain(n_units: int):
    """Cyclic cluster of ``n_units`` LiH units and its single translation.

    Coordinates and the translation vector are in angstrom, the convention of
    ``CCMOptions.translations`` and ``run_ccm``.
    """
    Z, coords = [], []
    for i in range(n_units):
        Z += [3, 1]
        coords += [[2 * i * D_LIH, 0.0, 0.0], [(2 * i + 1) * D_LIH, 0.0, 0.0]]
    translations = [[2 * n_units * D_LIH, 0.0, 0.0]]
    return Z, coords, translations


def topology_report(Z, coords, translations) -> str:
    """One-line Wigner-Seitz topology summary from the public SECCM helper."""
    ws = build_wigner_seitz(
        np.asarray(coords) * ANGSTROM_TO_BOHR,
        [np.asarray(t) * ANGSTROM_TO_BOHR for t in translations],
    )
    n_atoms = len(Z)
    codes = sorted({d.code.value for d in ws.diagnostics(n_atoms)})
    return (
        f"dim={ws.dimensionality}  valid={ws.is_valid(n_atoms)}  "
        f"sum(w)/atom={ws.total_weight(0):.3f} (expect {n_atoms - 1})  "
        f"switching margin={ws.switching_margin:.3f}  "
        f"diagnostics={codes or ['none']}"
    )


print("SECCM / MSINDO on the uniform LiH chain (CCM1D, Madelung on)")
print("=" * 72)
per_unit = {}
for n in (2, 3, 4, 5):
    Z, coords, translations = lih_chain(n)
    mol = vq.Molecule(
        [vq.Atom(z, [x * ANGSTROM_TO_BOHR for x in xyz]) for z, xyz in zip(Z, coords)]
    )
    res = vq.run_job(
        mol,
        method="seccm",
        ccm_options=CCMOptions(translations=translations, madelung=True),
        output=str(HERE / f"{STEM}-lih{n}"),
    )
    per_unit[n] = res.energy / n
    plan = res.route_plan
    print(
        f"n={n}  atoms={2 * n:2d}  T={translations[0][0]:5.1f} A  "
        f"E={res.energy:14.8f} Ha  E/LiH={per_unit[n]:12.6f} Ha  "
        f"iters={res.n_iter:2d}  converged={res.converged}"
    )
    print(f"      route: dim={plan.periodic_dimension} "
          f"electrostatics={plan.electrostatics_kernel}")
    print(f"      topology: {topology_report(Z, coords, translations)}")

# Madelung embedding energy at n = 5 via the library entry point, which also
# exposes the Wigner-Seitz weighted nuclear (Madelung) term.
Z, coords, translations = lih_chain(5)
on = run_ccm(Z, coords, translations, madelung=True)
off = run_ccm(Z, coords, translations, madelung=False)
print()
print("Madelung embedding, n = 5:")
print(f"  E(madelung=True)  = {on.total_energy:14.8f} Ha")
print(f"  E(madelung=False) = {off.total_energy:14.8f} Ha")
print(f"  embedding energy  = {on.total_energy - off.total_energy:+.3e} Ha")
print(f"  WS-weighted nuclear (Madelung) term = {on.madelung_nuclear_energy:.6f} Ha")

delta = abs(per_unit[5] - per_unit[4])
print()
print(f"|E/LiH(5) - E/LiH(4)| = {delta:.2e} Ha  (reference build: 1.2e-4)")
assert delta < 5e-4, "per-unit energy did not converge with cluster size"
assert abs(per_unit[5] - (-0.849038)) < 1e-5, "n=5 energy per LiH unit moved"
assert abs(on.total_energy - 5 * per_unit[5]) < 1e-10, "run_job and run_ccm disagree"
print("OK")
