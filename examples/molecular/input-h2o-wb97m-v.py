"""Water — ωB97M-V, a complete range-separated meta-GGA hybrid.

ωB97M-V (Mardirossian & Head-Gordon, *J. Chem. Phys.* **144**, 214110
(2016)) composes three advanced pieces in one functional:

    erf-attenuated long-range exact exchange   (range separation, ω = 0.3)
  + a tau-dependent semilocal meta-GGA
  + VV10 nonlocal correlation                  (b = 6.0, C = 0.01)

The VV10 term is what makes this the *complete* functional rather than a
truncated one; vibe-qc evaluates it self-consistently on the SCF grid.
It reproduces PySCF's ``dft.RKS(xc="wb97m-v", nlc="VV10")`` to within
5e-5 Ha at the same grid (validated in ``tests/test_wb97m_v.py``).

Range-separated functionals run through direct SCF only (no 3-centre
density-fit path for the erf-attenuated exchange yet); the molecular
driver selects the direct builder automatically. Molecular RKS / UKS
only; analytic gradients are queued, finite-difference works.

Run:
    .venv/bin/python examples/molecular/input-h2o-wb97m-v.py

Outputs (next to this script):
    input-h2o-wb97m-v.out      — banner + SCF trace + functional block
    input-h2o-wb97m-v.molden   — KS orbitals at the converged solution
    input-h2o-wb97m-v.system   — provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-wb97m-v"

# Water at the same experimental geometry used by the B2PLYP and
# revDSD-PBEP86 examples next door (coordinates in bohr).
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. Standard output files via run_job (banner, MO table, .molden,
#    .system manifest). Leave density_fit at its default: the driver
#    forces the direct Fock build for a range-separated functional.
vq.run_job(
    mol,
    basis="cc-pvdz",
    method="rks",
    functional="wb97m-v",
    output=HERE / STEM,
)

# 2. The energy, plus a look at the recipe the resolver carries.
basis = vq.BasisSet(mol, "cc-pvdz")
opts = vq.RKSOptions()
opts.functional = "wb97m-v"
result = vq.run_rks(mol, basis, opts)

fn = vq.Functional("wb97m-v")

out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== ωB97M-V recipe (range-separated meta-GGA + VV10) ===\n")
    fh.write(f"  kind                  = {fn.kind}\n")
    fh.write(f"  is_range_separated    = {fn.is_range_separated}\n")
    fh.write(f"  rsh_omega             = {fn.rsh_omega:.4f}  (bohr^-1)\n")
    fh.write(f"  needs_vv10            = {fn.needs_vv10}\n")
    fh.write(f"  vv10_b, vv10_C        = {fn.vv10_b:.4f}, {fn.vv10_C:.4f}\n")
    fh.write(f"  E(ωB97M-V / cc-pVDZ)  = {result.energy:14.8f} Ha\n")
    fh.write("  (matches PySCF dft.RKS(xc='wb97m-v', nlc='VV10') to 5e-5 Ha)\n")

print(f"  E(ωB97M-V / cc-pVDZ) = {result.energy:14.8f} Ha")
