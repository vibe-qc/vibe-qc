---
myst:
  html_meta:
    "description": "AICCM quickstart: run a minimal Gamma-CCM four-centre cyclic cluster and a minimal 3-D chi-CCM calculation, with the two construction identities explicit, measured numbers, the theory in brief, variations and pitfalls."
    "og:title": "vibe-qc - AICCM quickstart"
---

# AICCM quickstart

```{admonition} Status: experimental
:class: warning

Every AICCM variant (`run_periodic_job(method="aiccm", variant=...)` and the library drivers under `vibeqc.periodic.ccm` and `vibeqc.periodic.chi`) is an experimental route. Every Γ-line SCF driver emits `vibeqc.AICCM2026DevAExperimentalWarning` and every χ-CCM driver emits `vibeqc.AICCM2026DevBExperimentalWarning`; the `.system` manifest records `method_status = "experimental"` for every AICCM job run through `run_periodic_job(method="aiccm", variant=...)`. Its numbers are research output, not validated results, and it is not part of the release gate. The label vocabulary is defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**Goal.** Run the two smallest ab initio cyclic cluster model (AICCM)
calculations vibe-qc offers, one on each development line, and read the
construction identity each result carries.

**Prerequisites.** A built vibe-qc with its native core (no optional extras),
and the [cyclic cluster model tutorial](cyclic_cluster_model.md) for the
Wigner-Seitz weighting these runs rest on. Both scripts finish in well under a
minute on a laptop.

## The two lines

The ab initio CCM ships as two independent development lines that are
compared, not identified, with each other:

| line | selector | construction |
|---|---|---|
| Γ-CCM, `aiccm2026dev-a` | `run_periodic_job(method="aiccm", variant="real-gamma" \| "neutral-bloch" \| "four-center")` | the neutral finite-torus construction (`real-gamma`, `neutral-bloch`, one Hamiltonian in two representations) and, kept in the same line, the 2014 union-and-weight / Wigner-Seitz four-centre lineage (`four-center`) |
| χ-CCM, `aiccm2026dev-b` | `run_periodic_job(method="aiccm", variant="chi")` | the finite-translation-group character construction on a Γ-centred character mesh |

The two are compared at a declared common exchange-q=0
convention[^xccm-convention]. A difference between variants is a construction
difference, never a convergence artefact; see the
[AICCM user guide](../user_guide/aiccm.md) for the selector rules.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    Γ-CCM is the neutral finite-torus construction with two producers,
    `variant="real-gamma"` and `variant="neutral-bloch"` (ruling R1, 2026-08-21),
    while the `four-center` variant is the 2014 union-and-weight/Wigner-Seitz
    lineage kept in the same `aiccm2026dev-a` development line; χ-CCM is the
    finite-translation-group character construction. A declared common
    exchange-q=0 convention is a comparison constraint, not an identification of
    the constructions. Their distinction is not a choice of Coulomb kernels, and
    equality for a specified operator and route must be established rather than
    inferred from the labels.

## Γ-CCM line: the four-centre union-and-weight lineage

The library driver builds a cyclic cluster of eight unit cells of an
equidistant hydrogen chain (a 4 Å cell with two H atoms 2 Å apart) and runs
closed-shell HF on the symmetric Wigner-Seitz four-centre weighting:

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf

BOHR = 1.0 / 0.529177210903          # bohr per angstrom
unit = PeriodicSystem(3, np.diag([4 * BOHR, 40.0, 40.0]),   # chain along x,
                      [Atom(1, [0, 0, 0]), Atom(1, [2 * BOHR, 0, 0])])  # 40 bohr vacuum
ccm = CCMSystem(unit, (8, 1, 1), "sto-3g")                   # 8 cells = 16 H
rhf = run_ccm_rhf(ccm, method="aiccm2026dev-a")
print(f"Γ-CCM RHF E/atom = {rhf.energy_per_atom:.6f} Ha  ({rhf.n_iter} iterations)")
```

Measured with vibe-qc 0.17.15 development build (main, 2026-10-11), on a
loaded laptop:

```text
Γ-CCM RHF E/atom = -0.395941 Ha  (14 iterations)
```

The run took 0.4 s of wall time. The chain is declared as a `dim=3` cell with
40 bohr of transverse vacuum, the form vibe-qc uses for wires everywhere
else in the manual; `CCMSystem` with the `(8, 1, 1)` extension builds the
same one-dimensional cyclic cluster from a `PeriodicSystem(1, ...)`
declaration, and the energy is identical to the printed digits in both
forms. The result's `backend` field reads
`ccm-fourcenter-dense-aiccm2026dev-a-rhf`: the dense Python four-centre
tensor, which is a small-cluster validation path. `run_ccm_rhf_scalable`
runs the same weighting through the integral-direct C++ kernel and reaches
real three-dimensional cells.

The same lineage through the study harness, with the test set's own H-chain
geometry (a different cell from the snippet above):

```bash
cd studies/aiccm-2026
python run_case.py h-chain aiccm-hf
```

Recorded harness output (`<system>__aiccm-hf.json`; not re-measured in this
documentation pass):

```json
{"system": "h-chain", "route": "aiccm-hf", "nrep": [8,1,1],
 "energy": -1.116713, "energy_per_atom": -0.558357, "converged": true}
```

## χ-CCM line: the finite-character construction

The same thing through the front door, on a 3-D H2 cell with the RI backend
and a one-point character mesh:

```python
import numpy as np
import vibeqc as vq

system = vq.PeriodicSystem(3, np.diag([8, 20, 20]),
                           [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
                           multiplicity=1)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
r = vq.run_periodic_job(system, basis, method="aiccm", variant="chi",
                        aiccm_lattice_extension=(1, 1, 1),
                        aiccm_backend="ri",
                        max_iter=40, progress=False,
                        citations=False, write_xyz_file=False,
                        output_qvf=False, output="h2-chi-ri")
print(f"χ-CCM RHF/RI E/cell = {r.energy:.8f} Ha")
print(r.aiccm2026dev_b.finite_torus_convention)
```

Measured with vibe-qc 0.17.15 development build (main, 2026-10-11), on a
loaded laptop (22.5 s of wall time, most of it the fit build):

```text
χ-CCM RHF/RI E/cell = -1.12662561 Ha
```

The convention record printed on the second line names the construction
beside the number: `ccm_approach='chi-ccm'`,
`ccm_construction='finite-translation-group-character'`,
`evaluation_representation='gamma-centred-character-mesh'`,
`coulomb_kernel='3d-periodic-g0'` and `exchange_q0='bvk-ewald'`. The study
harness writes the same record into its result JSON
(`python run_case_b.py c-diamond rhf-ri --mesh 1 1 1`). This is a
route-and-invariant check; the fitted χ-CCM energies stay
`not-qualified` in v1 (decision D93), so do not quote the total as a
validated number.

χ-CCM absolute-energy execution is 3-D only. Every 1-D and 2-D backend fails
closed before SCF pending one shared wire/slab Coulomb convention.

## What the output means

* `energy_per_atom` on the Γ-CCM result is the total cyclic-cluster energy
  divided by the number of atoms in the cluster; `energy` is the whole
  cluster, divide by `ccm.n_cells` for a per-cell value.
* `r.energy` on the front-door result is per unit cell. The `.out` "J/K
  method" line reads `aiccm (chi) ...; experimental` and the `.system`
  manifest records `method_status = "experimental"`, `aiccm_variant = "chi"`
  and the convention record above.
* Neither number is converged with respect to the cluster size. The cyclic
  cluster (or character mesh) is the convergence parameter, the real-space
  dual of a k mesh; grow it and watch the energy per atom settle.

## Theory in brief

A cyclic cluster is a supercell of $N = N_1 N_2 N_3$ unit cells closed into a
Born-von Karman torus. Each atom interacts with its surroundings out to the
boundary of its own Wigner-Seitz supercell (WSSC), and an atom that sits on
that boundary is shared between equidistant images with the weight

$$
\omega_{\mu\nu'} = \frac{1}{n_{\nu'}},
\qquad
S^{\mathrm{CCM}}_{\mu\nu} = \sum_{\nu'} \omega_{\mu\nu'}\,\langle\mu|\nu'\rangle ,
$$

where $n_{\nu'}$ counts the translationally equivalent images of $\nu$ inside
the WSSC of $\mu$ ({vibeqc-cite}`bredow_geudtner_jug_ccm_2001`,
{vibeqc-cite}`peintinger_ccm_2014`). The three- and four-centre terms of the
ab initio Fock operator extend this rule with averaged two-centre weights
over the union of the participating WSSCs; the resulting Fock matrix is
formally molecular, which is what lets the molecular SCF and post-HF kernels
run on it unchanged. The finite-character χ-CCM line reaches the same torus
from the other side: it minimises the energy over translation-commuting
densities on the complete Γ-centred character mesh of the finite translation
group. The [cyclic cluster model tutorial](cyclic_cluster_model.md#how-the-ab-initio-ccm-works)
derives the weights with their equations and the Madelung embedding of the
semiempirical lineage ({vibeqc-cite}`noga_ccm_1999`,
{vibeqc-cite}`janetzko_ccm_long_range_2002`, {vibeqc-cite}`janetzko_ccm_2008`).

## Variations

* **Pick a formulation through the front door.** Replace `variant="chi"`
  with `"real-gamma"`, `"neutral-bloch"` or `"four-center"`; omitting
  `variant` fails closed listing the four values. `kpoints=` stays an alias
  for the torus mesh but never both.
* **Choose the SCF reference explicitly.** `scf_reference="rhf"` or
  `"uhf"` selects Hartree-Fock on every variant (and rejects a
  `functional=`); `"uhf"` on a closed-shell singlet is the case the parity
  rule cannot express. Kohn-Sham is selected by `functional=`.
* **Add correlation.** `correlation="mp2"` or `"ccsd"` on `four-center`
  and the two neutral variants, `"dlpno-mp2"` / `"dlpno-ccsd"` on the neutral
  variants, and `"mp2"` on `chi` under its explicit fitted-source contract.
  The [AICCM user guide](../user_guide/aiccm.md#correlation) lists what each
  arm cites.
* **Reproduce the 2012 program.** `run_ccm_rhf(ccm, method="p2014")`
  and `run_ccm_rhf_scalable(ccm, method="p2014")` run the four-centre
  construction as the 2014 paper's program actually ran it, with its own
  nuclear attraction (`nuclear_attraction="program-union"`, the default, or
  `"printed-eq13"`). It reproduces Table 2 of the paper to the printed digits
  and is diagnosis lineage, not a production route; the runner does not
  expose it.
* **Size the cluster by a radius.** `CCMSystem.from_interaction_range(unit,
  R_c, "sto-3g")` derives the smallest `nrep` whose WSSC encloses a sphere of
  radius `R_c` around every atom; `ccm_interaction_range_scan` is the CCM
  analogue of a k-point convergence study.

## Pitfalls

* The library drivers default to `method="union12"`, the historical product
  weight whose Coulomb supermatrix carries a negative subspace on any basis
  with more than one function per centre (issue #242). Pass
  `method="aiccm2026dev-a"` explicitly, as above; the front door always does.
* `four-center`, `real-gamma` and `neutral-bloch` are not one method: the
  bare four-centre operator and the neutral torus differ by a Madelung-scale
  background on ionic cells, so never compare a `four-center` energy with a
  neutral one as if it were a fitting error.
* The two neutral producers refuse a cell declared 3-D whose transverse
  directions measure as pure vacuum (a gap above 25 bohr and above four times
  the shortest lattice vector), because the neutral construction then corrects
  for images that do not exist. The H-chain cell above passes through the
  dense four-centre driver, which carries no such guard; the neutral
  variants need `aiccm_allow_vacuum_padding=True`, which warns and records the
  padded axes.
* χ-CCM fitted results stay `not-qualified` (D93) whatever the SCF
  convergence; the convention record and the `.system` manifest are the
  evidence to keep, not the bare total.

## What next?

| you want to... | go here |
|---|---|
| Walk through step by step | [Γ-CCM tutorial](aiccm2026dev_a.md) |
| See every route and backend | [Γ-CCM reference](../aiccm2026dev_a.md) or [χ-CCM guide](../user_guide/aiccm2026dev_b.md) |
| Select by variant, reference and correlation | [AICCM user guide](../user_guide/aiccm.md) |
| Run open-shell | [Open-shell AICCM](../experimental/aiccm2026dev_open_shell.md) |
| Debug a problem | [Troubleshooting](../experimental/aiccm2026dev_troubleshooting.md) |
| Run the full 28-system benchmark | [AICCM-2026 benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README.md) |
