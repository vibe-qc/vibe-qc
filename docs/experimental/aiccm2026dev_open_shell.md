---
myst:
  html_meta:
    "description": "Open-shell AICCM: union-and-weight Γ-CCM methods, A-namespace neutral fitted-torus controls, and χ-CCM methods. Spin contamination, the closed-shell-collapse gate, and current limits."
    "og:title": "vibe-qc - open-shell AICCM (Γ-CCM and χ-CCM)"
---

# Open-shell AICCM

Both Γ-CCM (`aiccm2026dev-a`) and χ-CCM[^xccm-convention]
(`aiccm2026dev-b`) ship unrestricted SCF (UHF, UKS), but their
open-shell correlation and gradient surfaces differ. Γ-CCM uses the
union-and-weight/Wigner--Seitz integral-weighting construction, while χ-CCM
uses the finite-translation-group character construction. The A module also
contains neutral fitted-torus correlation controls whose namespace does not
assign them either construction identity. This page maps what is available,
what is validated, and what is still closed.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

```{warning}
**Experimental.** Open-shell AICCM numbers are research-grade. Every
invocation emits an experimental warning. Validate against a
closed-shell consistency check before trusting a result.
```

## Γ-CCM / aiccm2026dev-a

Γ-CCM ships unrestricted SCF, UMP2, and analytic SCF gradients through
the CCM integral weighting. Some older APIs in the same namespace build a
neutral fitted-torus correlation control instead; that control is useful, but
it is not a Γ-CCM result.

| level | function | notes |
|---|---|---|
| UHF | `run_ccm_uhf` | WSSC-weighted UHF; spin-resolved SCF |
| UKS | `run_ccm_uks` | any libxc functional |
| UMP2 | `run_ccm_ump2` | on the CCM MO integrals |
| neutral-control UCCSD(T) | `run_ccm_uccsd` | neutral fitted-torus operator; not Γ-CCM |

The neutral-control consistency gate: on a closed-shell cluster
`run_ccm_uccsd` reproduces `run_ccm_ccsd` on the same neutral fitted-torus
operator to ≤ 1e-8 (both the CCSD and the (T) triples correction,
`tests/test_ccm_uccsd.py`). This gate does not validate the union-and-weight
Γ-CCM construction.

### Neutral-control UHF + UCCSD(T) on a doublet chain

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.uhf import run_ccm_uhf
from vibeqc.periodic.ccm.uccsd import run_ccm_uccsd
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral

BOHR = 1.0 / 0.529177210903
a = 3.2 * BOHR
system = PeriodicSystem(
    1, np.diag([a, 40.0, 40.0]),
    [Atom(3, [0, 0, 0]), Atom(1, [a/2, 0, 0])],
    charge=0, multiplicity=2,    # doublet → open-shell UHF
)
ccm = CCMSystem(system, (4, 1, 1), "sto-3g")

# Neutral fitted-torus control. Do not label this result Γ-CCM.
g = ccm_eri_neutral(ccm)

# UHF on the neutral control operator
uhf = run_ccm_uhf(ccm, eri=g)
print(f"UHF E/cell = {uhf.energy:.8f} Ha")
print(f"⟨S²⟩ = {uhf.s2_expectation:.4f}")

# UCCSD(T) on that same neutral control operator
ucc = run_ccm_uccsd(ccm, uhf, eri=g)
print(f"UCCSD(T) Ecorr = {ucc.e_correlation:.8f} Ha, (T) = {ucc.e_t:.2e}")
print(f"n_alpha = {ucc.n_alpha}, n_beta = {ucc.n_beta}")
```

### NiO AFM - antiferromagnetic rocksalt

NiO is the open-shell benchmark in the 28-system test set. The AFM
ordering doubles the primitive cell to 4 atoms (2 Ni↑, 2 Ni↓, 4 O):

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.uhf import run_ccm_uhf
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

BOHR = 1.0 / 0.529177210903
a = 4.162 * BOHR
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])

# AFM-II: double the cell along [001] to alternate spin layers.
# Primitive cell 2 atoms; AFM cell 4 atoms: Ni↑, Ni↓, O, O.
# (Simplified here - the test-set NiO entry uses the full CRYSTAL .d12
# geometry; see studies/aiccm-2026/testset.py for the exact builder.)
system = PeriodicSystem(
    3, np.diag([a, a, 2*a]),
    [Atom(28, [0, 0, 0]), Atom(28, [0, 0, a]),
     Atom(8,  [a/2, a/2, a/2]),
     Atom(8,  [a/2, a/2, 3*a/2])],
    charge=0, multiplicity=3,    # two unpaired spins → triplet
)
ccm = CCMSystem(system, (2, 2, 1), "sto-3g")

# Γ-CCM UHF through the union-and-weight construction.
uhf = run_ccm_uhf(ccm, method="aiccm2026dev-a")
print(f"NiO AFM UHF E/atom = {uhf.energy_per_atom:.8f} Ha")
print(f"⟨S²⟩ = {uhf.s2_expectation:.4f}")
```

The test-set runner keeps the union-and-weight Γ route and the neutral fitted-
torus GDF control separate:

```bash
cd studies/aiccm-2026
python run_case.py nio-afm aiccm-hf     # Γ-CCM union-and-weight UHF
python run_case.py nio-afm aiccm-ri     # neutral fitted-torus GDF control
```

## χ-CCM / aiccm2026dev-b

χ-CCM ships **unrestricted SCF** (UHF, UKS) and **unrestricted
correlation** (UMP2, UCCSD(T), DLPNO-UMP2, DLPNO-UCCSD(T)) through the
direct APIs:

```python
from vibeqc.periodic.chi.scf import run_aiccm2026dev_b_uhf, run_aiccm2026dev_b_uks
from vibeqc.periodic.chi.posthf import (
    run_aiccm2026dev_b_ump2,
    run_aiccm2026dev_b_uccsd_t,
    run_aiccm2026dev_b_dlpno_ump2,
    run_aiccm2026dev_b_dlpno_uccsd_t,
)
```

All χ-CCM SCF and post-HF routes are 3-D only: every 1-D/2-D backend fails
closed until a shared lower-dimensional Coulomb convention is derived. Alpha
and beta occupied projectors are localized
independently. The full-domain UCCSD(T) implementation is the explicitly
cost-capped O(N⁶) correctness oracle from the DLPNO stack, not a claim
of production reduced scaling.

The `run_periodic_job` entry point with `jk_method="aiccm2026dev-b"` is
**closed-shell only** at present - use the direct APIs for open-shell
χ-CCM work.

### UMP2 on a 3-D LiH doublet

```python
import numpy as np
import vibeqc as vq
from vibeqc.periodic.chi.posthf import run_aiccm2026dev_b_ump2

a = 4.0840
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(3, [0, 0, 0]),
     vq.Atom(1, (lat @ np.array([0.5, 0.5, 0.5])).tolist())],
    charge=1, multiplicity=2,     # doublet
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

ump2 = run_aiccm2026dev_b_ump2(system, basis, lattice_extension=(2, 1, 1))
print(f"UMP2 Ecorr = {ump2.e_correlation:.8f} Ha")
```

## Spin-contamination diagnostics

Both lines report `⟨S²⟩` on unrestricted SCF results. For a pure spin
state the expectation should be close to `S(S+1)`:

```python
# Γ-CCM
s2 = uhf.s2_expectation
s2_ideal = 0.5 * (0.5 + 1)   # doublet → 0.75
print(f"⟨S²⟩ = {s2:.4f}  (ideal = {s2_ideal})")

# χ-CCM - available on the direct API result
# result.s2_expectation, result.spin_contamination
```

## Current limits

| feature | union-and-weight Γ-CCM | A-namespace neutral fitted-torus control | χ-CCM |
|---|---|---|---|
| UHF | ✅ | ✅ | ✅ (3-D, direct API) |
| UKS | ✅ | ✅ | ✅ (3-D, direct API) |
| UMP2 | ✅ (`run_ccm_ump2`) | ✅ (`run_ccm_ump2(cderi=...)`) | ✅ (3-D, direct API) |
| UCCSD(T) | ❌ | ✅ (`run_ccm_uccsd`) | ✅ (3-D, direct API) |
| DLPNO-UMP2 | ❌ | ✅ (`ccm_dlpno_ump2`) | ✅ (3-D, direct API) |
| DLPNO-UCCSD(T) | ❌ | ✅ (`ccm_dlpno_uccsd`) | ✅ (3-D, direct API) |
| analytic SCF total gradients | ✅ (construction-specific) | ❌ (no neutral-control total-gradient contract) | ❌ (fails closed) |
| `run_periodic_job` entry | N/A | N/A | ✅ (3-D RHF/RKS/UHF/UKS) |
| Spin properties | `ccm_homo_lumo_gap` (spin-resolved) | same A-namespace analysis helpers | `derive_aiccm2026dev_b_scf_properties` (spin populations) |

- **Open-shell DLPNO in the A namespace** is implemented only for the neutral
  fitted-torus control. No union-and-weight Γ-CCM DLPNO implementation is
  currently assigned.
- **χ-CCM open-shell** is 3-D only at both SCF and post-HF levels; the shared
  low-dimensional RI/GDF mesh is not a wire/slab Coulomb kernel.
- **Analytic gradients** are construction-specific. Γ-CCM has analytic
  RHF/UHF/RKS/UKS gradients. χ-CCM total gradients fail closed; never
  substitute a Γ-CCM gradient for a χ-CCM energy.

## See also

- [Γ-CCM reference](../aiccm2026dev_a.md) - the union-and-weight method
  stack and the separate A-namespace neutral-control APIs
- [χ-CCM user guide](../user_guide/aiccm2026dev_b.md) - the χ-CCM API and
  backend matrix
- [Γ-CCM tutorial](../tutorial/aiccm2026dev_a.md) - step-by-step
  walkthrough
- [Molecular open-shell tutorial](../tutorial/open_shell.md) - UHF/UKS
  concepts that apply to the CCM as well
