---
myst:
  html_meta:
    "description": "AICCM quickstart: run minimal Γ-CCM and 3-D χ-CCM calculations with the two distinct construction identities explicit."
    "og:title": "vibe-qc - AICCM quickstart"
---

# AICCM quickstart

Two ways to run an experimental ab-initio Cyclic Cluster Model calculation:
Γ-CCM (`aiccm2026dev-a`, union-and-weight/Wigner--Seitz integral weighting)
and χ-CCM[^xccm-convention] (`aiccm2026dev-b`, finite-translation-group
characters). These are distinct approaches compared at a declared common
exchange-q=0 convention.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

## Γ-CCM / aiccm2026dev-a (union and weights)

```bash
cd studies/aiccm-2026
python run_case.py h-chain aiccm-hf
```

Expected output (`<system>__aiccm-hf.json`):

```json
{"system": "h-chain", "route": "aiccm-hf", "nrep": [8,1,1],
 "energy": -1.116713, "energy_per_atom": -0.558357, "converged": true}
```

Or from Python:

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf

BOHR = 1.0 / 0.529177210903
unit = PeriodicSystem(1, np.diag([4*BOHR, 40, 40]),
                      [Atom(1, [0,0,0]), Atom(1, [2*BOHR,0,0])])
ccm = CCMSystem(unit, (8,1,1), "sto-3g")
rhf = run_ccm_rhf(ccm, method="aiccm2026dev-a")
print(f"Γ-CCM RHF E/atom = {rhf.energy_per_atom:.6f} Ha")
```

## χ-CCM / aiccm2026dev-b (finite-character Γ-centred character mesh)

```bash
cd studies/aiccm-2026
python run_case_b.py c-diamond rhf-ri --mesh 1 1 1
```

The result JSON records the exact construction identity alongside the energy:

```json
{"system": "c-diamond", "route": "rhf-ri", "mesh": [1,1,1],
 "converged": true,
 "finite_torus_convention": {
   "ccm_approach": "chi-ccm",
   "ccm_construction": "finite-translation-group-character",
   "evaluation_representation": "gamma-centred-character-mesh",
   "coulomb_kernel": "3d-periodic-g0",
   "exchange_q0": "bvk-ewald"}}
```

The actual JSON also carries the computed energy and full producer fingerprint.

Or from Python:

```python
import numpy as np
import vibeqc as vq

system = vq.PeriodicSystem(3, np.diag([8, 20, 20]),
                           [vq.Atom(1, [0,0,0]), vq.Atom(1, [1.4,0,0])],
                           multiplicity=1)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
r = vq.run_periodic_job(system, basis, method="aiccm", variant="chi",
                        aiccm_lattice_extension=(1,1,1),
                        aiccm_backend="ri",
                        max_iter=40, progress=False,
                        citations=False, write_xyz_file=False,
                        output_qvf=False)
print(f"χ-CCM RHF/RI E/cell = {r.energy:.8f} Ha")
```

χ-CCM-B absolute-energy execution is currently 3D only. Every 1D/2D backend
fails closed pending one shared wire/slab Coulomb convention.

## What next?

| you want to… | go here |
|---|---|
| Walk through step by step | [Γ-CCM tutorial](aiccm2026dev_a.md) |
| See every route and backend | [Γ-CCM reference](../aiccm2026dev_a.md) or [χ-CCM guide](../user_guide/aiccm2026dev_b.md) |
| Run open-shell | [Open-shell AICCM](../experimental/aiccm2026dev_open_shell.md) |
| Debug a problem | [Troubleshooting](../experimental/aiccm2026dev_troubleshooting.md) |
| Run the full 28-system benchmark | [AICCM-2026 benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README.md) |
