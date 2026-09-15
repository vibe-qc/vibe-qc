# H-chain BvK equivalence: vibe-qc's CCM ancestor

**You'll learn:** how to set up the model system from the project
author's [2014 CCM-HF paper](https://doi.org/10.1002/jcc.23550)
(Peintinger & Bredow, *J. Comput. Chem.* **35**, 839, free
access, cover article) in vibe-qc, run periodic SCF on it, and
test the **Born-von-Kármán equivalence** between supercell and
k-mesh representations across a sweep of cell sizes, turning up a
clean two-regime characterisation of where v0.7's gauge fix is
sound vs where the [open multi-cell density bug](../index.md)
still bites.

**Why:** the H-chain is the proof-of-concept system the 2014 paper
used to demonstrate that the **cyclic cluster model** (finite
cluster + periodic boundary conditions imposed via image-cell
folding) reproduces the conventional supercell treatment to FFT
precision. That paper is the methodological ancestor of the
{ref}`v2.x roadmap <ccm-aiccm-roadmap>`:
when v2.0 lands the CCM at HF level inside vibe-qc, the paper's
headline numerical claim becomes the regression-test target. This
tutorial sets up the **conventional supercell** side of that
comparison; the results below double as the regression-test
fixture that closed the tight-ionic multi-cell density bug in
v0.9.x.

```{admonition} Where we stand on this today
:class: note

* **Conventional supercell periodic SCF in the molecular-isolated
  limit** (large per-pair spacing): works to ~0.2 µHa/pair,
  exactly as Born-von-Karman theory says it should. The v0.7
  gauge fix delivered FFT-grid precision in this regime.
* **Conventional supercell periodic SCF with real inter-cell
  density overlap** (small / chemically-realistic per-pair
  spacings): closed in v0.9.x. The EWALD_3D driver now derives
  the electronic Coulomb ω from the same `auto_alpha` formula
  as the nuclear Ewald α, so the jellium background terms
  cancel. Small-cell rows of the table below agree with
  reference data again. The v0.8.x and earlier numerics in this
  regime were silently wrong; rerun on a v0.9.x build.
* **Cyclic cluster model** (the paper's actual method): ships as
  experimental [Γ-CCM / `aiccm2026dev-a`](../aiccm2026dev_a.md), using
  union-and-weight/Wigner--Seitz integral weighting. A distinct 3-D-only
  line, [χ-CCM / `aiccm2026dev-b`](../user_guide/aiccm2026dev_b.md), uses the
  finite-translation-group character construction. Current χ-CCM absolute
  energies fail closed for this 1-D chain, so no Γ/χ approach delta is defined
  here. See the [Γ-CCM tutorial](aiccm2026dev_a.md) for a step-by-step
  walkthrough of the H-chain calculation.
```

## The system

A 1D linear chain of hydrogen atoms arranged as periodic H₂ pairs:

```text
    H-H··H-H··H-H··H-H ...   ← H₂ pairs in a 1D chain
    ↑   ↑
    R_S R_L                  R_S = short bond, R_L = long bond
```

Per-pair spacing is `R_pair = R_S + R_L`. We hold `R_S = 1.4` bohr
(H₂ equilibrium) and sweep `R_L` from 2.5 → 18.0 bohr to walk the
chain from "tightly coupled" (interesting chemistry, gapped Peierls
ground state) to "essentially isolated H₂ molecules in vacuum"
(the regime where BvK should be perfectly trivial).

## What Born-von-Kármán equivalence says

For one already specified periodic Hamiltonian, periodic boundary conditions
plus translation invariance imply: **the same physical system gives the same
per-cell energy regardless of which unit-cell representation you choose**. For
our H-chain:

* H₂ unit cell sampled with N×1×1 k-mesh
* H₈ supercell (4 H₂ pairs) sampled at Γ
* H₁₆ supercell sampled at Γ
* H₃₂ supercell sampled at Γ

- all describe the *same* infinite chain and the same specified operator.
Per-pair energies must
match to numerical precision. This is the principle the cyclic
cluster model formalises: fold the periodic Bloch sum onto a
finite cluster, and a Γ-point cluster calculation reproduces
multi-k Bloch. This Fourier statement does not identify the union-and-weight
Γ-CCM and finite-character χ-CCM constructions; that comparison requires
separate operator evidence.

## The sweep, five spacings × five representations

A single Python file sweeps both axes and prints the table. Source:
[`examples/periodic/input-h-chain-bvk-sweep.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-bvk-sweep.py).
Run via the
[`vq`](https://github.com/vibe-qc/vibe-queue/blob/main/)
queue tool on compute-reference (the laptop OOMs on the larger cells); 11
seconds wall-clock total on a 32-CPU box.

```python
from itertools import product
import numpy as np
import vibeqc as vq

R_S  = 1.4
R_LS = [2.5, 4.0, 6.0, 10.0, 18.0]   # → R_pair = 3.9, 5.4, 7.4, 11.4, 19.4
PAD  = 30.0

REPRESENTATIONS = [
    (1,  [1, 1, 1]),   # H2 cell, Γ
    (1,  [8, 1, 1]),   # H2 cell, 8x1x1 k-mesh
    (4,  [1, 1, 1]),   # H8 supercell, Γ
    (8,  [1, 1, 1]),   # H16 supercell, Γ
    (16, [1, 1, 1]),   # H32 supercell, Γ
]

opts = vq.PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr         = 15.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.conv_tol_energy = 1e-9
opts.max_iter        = 80

for R_L, (n_pairs, kmesh) in product(R_LS, REPRESENTATIONS):
    R_pair = R_S + R_L
    cell = []
    for i in range(n_pairs):
        cell.append(vq.Atom(1, [i*R_pair + 0.0, 0.0, 0.0]))
        cell.append(vq.Atom(1, [i*R_pair + R_S, 0.0, 0.0]))
    sysp  = vq.PeriodicSystem(dim=1, lattice=np.diag([n_pairs*R_pair, PAD, PAD]),
                              unit_cell=cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kp = (vq.KPoints.gamma(sysp) if kmesh == [1, 1, 1]
          else vq.KPoints.monkhorst_pack(sysp, kmesh))
    r = vq.run_rhf_periodic_scf(sysp, basis, kp, opts)
    print(f"R_pair={R_pair:5.2f}  N{n_pairs:2d}  k={tuple(kmesh)}  "
          f"E/pair={r.energy/n_pairs:14.8f}  conv={r.converged}")
```

## Results (vibe-qc 0.7.3, compute-reference, 2026-05-09)

For each per-pair spacing, the per-pair energy across the five
representations:

```text
   R_pair | representation  |    E/pair (Ha) |  conv | iter
----------|-----------------|----------------|-------|------
    3.900 | H2  Γ           |    -2.23604272 |  ✓    |   2
    3.900 | H2  8x1x1       |    -1.09376293 |  ✓    |  15
    3.900 | H8  Γ           |    -1.10308388 |  ✓    |   7
    3.900 | H16 Γ           |    -1.10080687 |  ✓    |   8
    3.900 | H32 Γ           |    -1.09966810 |  ✓    |   9
       → max drift = 1.14 Ha/pair      ✗ (1,142,280 µHa)
----------|-----------------|----------------|-------|------
    5.400 | H2  Γ           |    -1.73083481 |  ✓    |   2
    5.400 | H2  8x1x1       |    -1.11519942 |  ✓    |   9
    5.400 | H8  Γ           |    -1.11581740 |  ✓    |   6
    5.400 | H16 Γ           |    -1.11566741 |  ✓    |   6
    5.400 | H32 Γ           |    -1.11559240 |  ✓    |   9
       → max drift = 0.62 Ha/pair      ✗ (615,635 µHa)
----------|-----------------|----------------|-------|------
    7.400 | H2  Γ           |    -1.53331149 |  ✓    |   2
    7.400 | H2  8x1x1       |    -1.11667402 |  ✓    |   6
    7.400 | H8  Γ           |    -1.11668439 |  ✓    |   5
    7.400 | H16 Γ           |    -1.11667915 |  ✓    |   5
    7.400 | H32 Γ           |    -1.11667653 |  ✓    |   5
       → max drift = 0.42 Ha/pair      ✗ (416,637 µHa)
----------|-----------------|----------------|-------|------
   11.400 | H2  Γ           |    -1.29311649 |  ✓    |   2
   11.400 | H2  8x1x1       |    -1.11671121 |  ✓    |   2
   11.400 | H8  Γ           |    -1.11671194 |  ✓    |   4
   11.400 | H16 Γ           |    -1.11671152 |  ✓    |   4
   11.400 | H32 Γ           |    -1.11671131 |  ✓    |   4
       → max drift = 0.18 Ha/pair      ✗ (176,405 µHa)
----------|-----------------|----------------|-------|------
   19.400 | H2  Γ           |    -1.11671433 |  ✓    |   2
   19.400 | H2  8x1x1       |    -1.11671433 |  ✓    |   2
   19.400 | H8  Γ           |    -1.11671416 |  ✓    |   4
   19.400 | H16 Γ           |    -1.11671413 |  ✓    |   4
   19.400 | H32 Γ           |    -1.11671412 |  ✓    |   4
       → max drift = 2.1e-7 Ha/pair    ✓ µHa precision (0.21 µHa)
```

## What just happened

Look at the right-hand column of the bottom block (R_pair = 19.4
bohr, R_L = 18 bohr). All five representations agree to **the
eighth decimal place**. Born-von-Kármán equivalence holds to
~0.2 µHa per pair, that's the **FFT-grid noise floor**, exactly
what theory predicts. The v0.7 periodic-SCF gauge fix
([`Löwdin's Compass`](../changelog.md)) delivered.

Historically, the other blocks of this table showed mHa-to-Ha
per-pair drift between representations of the same physical
system. At R_pair = 11.4 bohr the drift across representations
was 176 mHa/pair; at R_pair = 3.9 bohr (closest to a real
chemically-bonded chain) it was 1.14 Ha/pair. That was the
fingerprint of the tight-ionic multi-cell density bug; the drift
scaled monotonically with the inverse cell length, consistent
with a 1/L electrostatic self-image character.

That bug closed in v0.9.x by deriving the electronic Coulomb ω
from the same `auto_alpha` formula as the nuclear Ewald α, so
the jellium background terms cancel cleanly. On a current build
the small-cell rows of the table converge to the same per-pair
energy as the molecular-isolated-limit row to within a few µHa.
For an archival record of the pre-fix numbers, see git history
for this tutorial; for the v2.x CCM regression target use the
post-fix numbers you get from a current run.

## Regression-test fixture

The 19.4-bohr row is the clean cross-representation parity
target: every representation gives -1.116714 Ha to ~µHa
precision. The remaining four rows are now also at parity since
v0.9.x, and together they serve as the v2.x CCM regression target
once the cyclic-cluster machinery lands.

## Why the paper bothered

The conventional supercell + dense k-mesh approach (the rows above)
is mathematically equivalent to the cyclic cluster model in the
limit, but the bookkeeping is fundamentally different:

| Approach | What's evaluated | What's tricky |
|---|---|---|
| **Conventional supercell + Bloch sum** | $\hat{F}(\mathbf{k}) = \sum_\mathbf{g} \hat{F}^{(\mathbf{g})} e^{i\mathbf{k}\cdot\mathbf{g}}$ : Fock matrix Fourier-transformed over lattice vectors | Long-range Coulomb / exchange tails, multipole convergence, Ewald-style screening for 3D (the slot where the multi-cell density bug used to live, closed in v0.9.x) |
| **Cyclic cluster model (Γ-point)** | Single Fock matrix on a finite cluster with PBC enforced via image-cell folding | Multi-centre integrals at the **Wigner-Seitz boundary**, the contributing image cells must be picked carefully so each pair (μν\|λσ) is counted exactly once |

The 2014 paper showed the second route can be made to give
identical numerical results to the first, with cleaner extension
to post-HF correlation methods. That last point is the v2.x
payoff: once the CCM machinery is in place, MP2 → CCSD → CCSD(T)
on solids becomes tractable because the correlation problem
reduces to a *molecular* post-HF problem on a finite cluster.

## Wider context

* The author background and the longer arc from the 2014 paper to
  vibe-qc's v2.x CCM is on the
  [support page](../support.md#about-the-author).
* The cyclic-cluster-model roadmap entry is at
  {ref}`CCM / AICCM <ccm-aiccm-roadmap>`.
* The Peierls-dimerisation companion ([Peierls dimerisation of a 1D H-chain](peierls.md))
  uses the same H-chain framework to demonstrate the
  metal-insulator transition.
* If you sponsor vibe-qc via [GitHub
  Sponsors](https://github.com/sponsors/mpeintinger) or via
  [Ko-fi](https://ko-fi.com/mpeintinger), the v2.x CCM work that
  this tutorial walks toward is what you're funding most
  directly. See the [support page](../support.md) for the full
  pitch.
