"""Water — DLPNO-CCSD(T): local solver + DLPNO-(T1).

`method="dlpno-ccsd(t)"` runs the reduced-scaling local DLPNO-CCSD solver
(pair-PNO amplitudes with residuals contracted in the atom-based extended PAO
domain) and then **DLPNO-(T1)** on the converged amplitudes
(``vibeqc.dlpno.triples_local``): the perturbative
triples are evaluated per occupied triple in a TNO domain with the
off-diagonal localised Fock coupling restored iteratively (Guo, Riplinger
et al., J. Chem. Phys. 148, 011101 (2018)). This removes the (T0)
semicanonical approximation while retaining the PNO/TNO-domain truncation;
it is not the canonical (T) at finite domains.

Validation limit: when the pair, PNO, and TNO domains are all complete,
(T1) reproduces canonical CCSD(T) to machine precision (the full-domain
parity ratchet). Finite NormalPNO domains remain a local-correlation
approximation. The (T) vanishes (to floating-point noise) for two-electron
systems. Set ``triples_mode="local"`` for the older DLPNO-(T0) (diagonal
localised Fock, ~0.1 kcal/mol looser).

Cost honesty: the (T) uses a spatial closed-shell kernel (no spin-orbital
redundancy), validated to machine precision against the spin-orbital
reference. The default extended-domain CCSD residual and the (T) currently
use NumPy; the compiled target-pair residual is confined to the explicit
legacy ``residual_domain="pair"`` mode.
For an independent exact-(T) oracle on validation-scale molecules, pass a
``DLPNOCCSDPilotOptions`` to fall back to the O(N⁶) pilot (64-bf capped).
See ``docs/user_guide/dlpno_mp2.md`` § DLPNO-CCSD and
``docs/tutorial/dlpno_local_correlation.md``.

Run:
    .venv/bin/python examples/molecular/input-h2o-dlpno-ccsd-t.py

Outputs (next to this script):
    input-h2o-dlpno-ccsd-t.out     — banner + RHF trace + DLPNO-CCSD(T) block
    input-h2o-dlpno-ccsd-t.molden  — orbitals at the RHF solution
    input-h2o-dlpno-ccsd-t.system  — provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-dlpno-ccsd-t"

mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# One-call DLPNO-CCSD(T): the local solver + local DLPNO-(T) on the
# converged amplitudes (default; no pilot cap). Spell out the two published
# post-#140/#448 defaults so the example's own artifacts state the recipe.
result = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="dlpno-ccsd(t)",
    frozen_core="published",
    dlpno_thresholds="normal",
    output=HERE / STEM,
)

cc = result.dlpno_ccsd
print(f"  E(RHF)              = {cc.e_hf:14.8f} Ha")
print(f"  E_corr(DLPNO-CCSD)  = {cc.e_corr:14.8f} Ha")
print(f"  E((T) correction)   = {cc.e_t:14.8f} Ha")
print(f"  E(DLPNO-CCSD(T))    = {result.energy_total:14.8f} Ha")
print(
    f"  pairs / avg PNOs    = {cc.n_pairs} / "
    f"{sum(cc.pno_per_pair.values()) / max(cc.n_pairs, 1):.1f}"
)
