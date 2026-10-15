# H-chain BvK equivalence: vibe-qc's CCM ancestor

```{admonition} Status: experimental
:class: warning

The cyclic cluster model this page traces (every AICCM variant, `run_periodic_job(method="aiccm", variant=...)`) is an experimental route. Every Γ-line SCF driver emits `vibeqc.AICCM2026DevAExperimentalWarning` and every χ-CCM driver emits `vibeqc.AICCM2026DevBExperimentalWarning`; the `.system` manifest records `method_status = "experimental"` for every AICCM job run through `run_periodic_job(method="aiccm", variant=...)`. Its numbers are research output, not validated results, and it is not part of the release gate. The conventional supercell periodic SCF the sweep below runs on is the shipped production route; only the CCM side is experimental. The label vocabulary is defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**You'll learn:** how to set up the model system from the project
author's [2014 CCM-HF paper](https://doi.org/10.1002/jcc.23550)
(Peintinger & Bredow, *J. Comput. Chem.* **35**, 839, free
access, cover article; {vibeqc-cite}`peintinger_ccm_2014`) in vibe-qc, run periodic SCF on it, and
test the **Born-von-Kármán equivalence** between supercell and
k-mesh representations across a sweep of cell sizes.

**Why:** the H-chain is the proof-of-concept system the 2014 paper
used to demonstrate that the **cyclic cluster model** (finite
cluster + periodic boundary conditions imposed via image-cell
folding) reproduces the conventional supercell treatment to FFT
precision. That paper is the methodological ancestor of the
{ref}`v2.x roadmap <ccm-aiccm-roadmap>`:
the CCM at HF level now ships in vibe-qc as the experimental
`four-center` AICCM variant (see the box below), and the paper's
headline numerical claim is the natural regression target for it. This
tutorial sets up the **conventional supercell** side of that
comparison; the results below double as the regression-test
fixture that closed the tight-ionic multi-cell density bug in
v0.9.x.

```{admonition} Where we stand on this today
:class: note

* **Conventional supercell periodic SCF.** The sweep below models the
  chain as a `dim=3` cell with transverse vacuum and runs k-point GDF
  ({vibeqc-cite}`sun_berkelbach_gdf_2017`),
  because vibe-qc has no route with a consistent 1-D Coulomb gauge for
  a `PeriodicSystem(dim=1, ...)`. A Γ supercell and the k mesh that
  describe the same torus agree within 1.3e-8 Ha per pair for the
  4-cell torus at every spacing. The 8-cell torus agrees from
  R_pair = 7.4 bohr on, but differs by 135 µHa per pair at 3.9 bohr and
  7.6 µHa at 5.4 bohr. That residual is a nuclear-attraction cutoff
  defect in the long H16 supercell, not physics (#457); see
  [What just happened](#what-just-happened). Earlier
  versions of this page showed the legacy direct-truncated route of
  v0.7.3, whose wire energies depend on the lattice-sum cutoff; they
  are in git history.
* **Cyclic cluster model** (the paper's actual method): the 2014
  union-and-weight/Wigner-Seitz construction ships as the experimental
  `four-center` variant of AICCM, kept in the
  [`aiccm2026dev-a`](../aiccm2026dev_a.md) development line beside Γ-CCM,
  the neutral finite-torus construction (variants `real-gamma` and
  `neutral-bloch`). A distinct 3-D-only
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
A supercell of N cells at Γ and the unit cell on an N×1×1 mesh describe
the same periodic torus, so their per-pair energies must match to
numerical precision; tori of different size differ by a finite-size error
that vanishes as N grows. This is the principle the cyclic
cluster model formalises: fold the periodic Bloch sum onto a
finite cluster, and a Γ-point cluster calculation reproduces
multi-k Bloch. This Fourier statement does not identify the neutral Γ-CCM,
four-centre union-and-weight, and finite-character χ-CCM constructions with
one another; that comparison requires separate operator evidence.

## The sweep, five spacings × five representations

A single Python file sweeps both axes and prints the table. Source:
[`examples/periodic/input-h-chain-bvk-sweep.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-bvk-sweep.py).
The H16 supercell dominates the cost: the whole sweep took 26 minutes of
wall time with 4 OpenMP threads on a shared, heavily loaded laptop, about
2 minutes of it for each H16 SCF at the tightest spacing. The H32
supercell is opt-in (`--with-h32`), because one H32 Γ SCF ran for more
than 25 minutes there before it was stopped.

```python
import numpy as np
import vibeqc as vq

R_S  = 1.4
R_LS = [2.5, 4.0, 6.0, 10.0, 18.0]   # → R_pair = 3.9, 5.4, 7.4, 11.4, 19.4
PAD  = 15.0                          # transverse vacuum, bohr

REPRESENTATIONS = [
    (1, (1, 1, 1)),    # H2 cell, Γ
    (1, (4, 1, 1)),    # H2 cell, 4x1x1 k-mesh
    (1, (8, 1, 1)),    # H2 cell, 8x1x1 k-mesh
    (4, (1, 1, 1)),    # H8 supercell, Γ: same torus as 4x1x1
    (8, (1, 1, 1)),    # H16 supercell, Γ: same torus as 8x1x1
]

opts = vq.PeriodicRHFOptions()
opts.conv_tol_energy = 1e-9
opts.max_iter        = 80

for R_L in R_LS:
    R_pair = R_S + R_L
    for n_pairs, kmesh in REPRESENTATIONS:
        cell = []
        for i in range(n_pairs):
            cell.append(vq.Atom(1, [i*R_pair + 0.0, 0.0, 0.0]))
            cell.append(vq.Atom(1, [i*R_pair + R_S, 0.0, 0.0]))
        sysp  = vq.PeriodicSystem(dim=3,
                                  lattice=np.diag([n_pairs*R_pair, PAD, PAD]),
                                  unit_cell=cell)
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        r = vq.run_krhf_periodic_gdf(sysp, basis, kmesh, opts, progress=False)
        print(f"R_pair={R_pair:5.2f}  N{n_pairs:2d}  k={kmesh}  "
              f"E/pair={r.energy/n_pairs:14.8f}  conv={r.converged}")
```

## Results (vibe-qc 0.17.14, 2026-10-01)

For each per-pair spacing, the per-pair energy across the five
representations, and the two same-torus differences:

```text
   R_pair | representation  |    E/pair (Ha) | conv | iter
----------|-----------------|----------------|------|------
    3.900 | H2  Γ           |    -1.30857526 |  ✓   |   3
    3.900 | H2  4x1x1       |    -1.09951454 |  ✓   |   5
    3.900 | H2  8x1x1       |    -1.09933739 |  ✓   |   7
    3.900 | H8  Γ           |    -1.09951454 |  ✓   |   6
    3.900 | H16 Γ           |    -1.09947260 |  ✓   |   7
       → H8 Γ - H2 4x1x1 = -0.00 µHa;  H16 Γ - H2 8x1x1 = -135.21 µHa
----------|-----------------|----------------|------|------
    5.400 | H2  Γ           |    -1.18454714 |  ✓   |   3
    5.400 | H2  4x1x1       |    -1.11728562 |  ✓   |   5
    5.400 | H2  8x1x1       |    -1.11594898 |  ✓   |   7
    5.400 | H8  Γ           |    -1.11728562 |  ✓   |   5
    5.400 | H16 Γ           |    -1.11595659 |  ✓   |   7
       → H8 Γ - H2 4x1x1 = +0.00 µHa;  H16 Γ - H2 8x1x1 = -7.60 µHa
----------|-----------------|----------------|------|------
    7.400 | H2  Γ           |    -1.13352515 |  ✓   |   3
    7.400 | H2  4x1x1       |    -1.11772971 |  ✓   |   5
    7.400 | H2  8x1x1       |    -1.11676567 |  ✓   |   7
    7.400 | H8  Γ           |    -1.11772970 |  ✓   |   5
    7.400 | H16 Γ           |    -1.11676567 |  ✓   |   6
       → H8 Γ - H2 4x1x1 = +0.01 µHa;  H16 Γ - H2 8x1x1 = +0.00 µHa
----------|-----------------|----------------|------|------
   11.400 | H2  Γ           |    -1.12123221 |  ✓   |   3
   11.400 | H2  4x1x1       |    -1.11709168 |  ✓   |   4
   11.400 | H2  8x1x1       |    -1.11646638 |  ✓   |   5
   11.400 | H8  Γ           |    -1.11709168 |  ✓   |   4
   11.400 | H16 Γ           |    -1.11646638 |  ✓   |   5
       → H8 Γ - H2 4x1x1 = +0.00 µHa;  H16 Γ - H2 8x1x1 = -0.00 µHa
----------|-----------------|----------------|------|------
   19.400 | H2  Γ           |    -1.11879323 |  ✓   |   3
   19.400 | H2  4x1x1       |    -1.11657870 |  ✓   |   3
   19.400 | H2  8x1x1       |    -1.11621110 |  ✓   |   3
   19.400 | H8  Γ           |    -1.11657870 |  ✓   |   3
   19.400 | H16 Γ           |    -1.11621110 |  ✓   |   3
       → H8 Γ - H2 4x1x1 = -0.00 µHa;  H16 Γ - H2 8x1x1 = -0.00 µHa
```

## What just happened

Read the same-torus lines first. For the 4-cell torus the H8 supercell
at Γ and the H2 cell on a 4×1×1 mesh agree within 1.3e-8 Ha per pair
at every spacing: Born-von-Kármán equivalence holds, as it must
for one specified operator. The 8-cell torus does the same from
R_pair = 7.4 bohr on, but at the two tightest spacings the H16
supercell sits 135 µHa (3.9 bohr) and 7.6 µHa (5.4 bohr) per pair below
the 8×1×1 mesh. That gap is a defect in the supercell, not physics
(#457). The short-range half of the Ewald ({vibeqc-cite}`ewald_lattice_sum_1921`)
nuclear attraction only takes
nuclei from cells whose lattice translation lies within
`nuclear_cutoff_bohr` (25 bohr by default) of the origin. The H16 cell
is 31.2 bohr long at 3.9 bohr and 43.2 bohr at 5.4 bohr, so its outer
atoms lose the nuclei just across the cell face, while the H2 cell and
the H8 cell (15.6 and 21.6 bohr) keep them. At 3.9 bohr, evaluated at
one fixed density, the whole difference sits in the one-electron
energy; the fitted Coulomb and exchange agree to 1e-14 Ha. The missing
attraction falls off with the gap between neighbouring pairs, which is
why it vanishes from 7.4 bohr on. Until #457 is fixed, set
`opts.lattice_opts.nuclear_cutoff_bohr` above the supercell length
(40 bohr at 3.9, 50 bohr at 5.4): the H16 Γ energy then matches the
8×1×1 mesh to 1e-14 Ha per pair, and the 8×1×1 energy itself does not
change (bit-identical at 25 and 40 bohr for 3.9). The table above keeps
the default cutoff, so its H16 rows at the two tightest spacings carry
the defect.

The other rows compare tori of different size, so they carry genuine
finite-size error. The H2 cell at Γ is the crudest representation
(-1.309 Ha per pair at 3.9 bohr against about -1.099 Ha on the meshes),
and the gap between the 4- and 8-cell tori is still 0.37 mHa per pair at
19.4 bohr. In this vacuum box the energy per pair converges only as
1/N with the torus size, even for well-separated molecules: that tail is
the expected exchange finite-size term of the BvK-ewald gauge (#456), so
extrapolate it (2 E(2N) - E(N)) or use a dense mesh, and converge the
vacuum too, before quoting an absolute number.

## Regression-test fixture

The same-torus pairs are the parity target: at every spacing the
supercell at Γ must reproduce the unit cell on the matching k mesh.
The 4-cell torus meets that within 1.3e-8 Ha per pair today; the
8-cell torus does from 7.4 bohr on. Together they serve as the
regression target for the experimental cyclic-cluster variants.

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
to post-HF correlation methods. That last point is the payoff:
on a cyclic cluster, MP2 → CCSD → CCSD(T) on solids becomes
tractable because the correlation problem reduces to a *molecular*
post-HF problem on a finite cluster. The experimental `four-center`
and `real-gamma` AICCM variants already accept `correlation="mp2"` or
`"ccsd"` on `run_periodic_job(method="aiccm", ...)`.

## Theory

Born-von Kármán boundary conditions on a ring of $N$ primitive cells of
length $a$ admit only the wavevectors

$$
k_m = \frac{2\pi m}{N a}, \qquad m = 0, 1, \dots, N-1 ,
$$

the Γ-centred $N\times1\times1$ mesh. Bloch sums over the $N$ cells
block-diagonalise the Γ-point supercell Fock matrix into the $N$
primitive-cell problems
$\mathbf F(k_m)\mathbf C = \mathbf S(k_m)\mathbf C\boldsymbol\varepsilon$, and the
supercell density is the mesh average of the cell densities,

$$
\mathbf D(T) = \frac{1}{N}\sum_{m=0}^{N-1} e^{-i k_m T}\,\mathbf D(k_m),
\qquad
E_{\text{cell}}\big[N\times1\times1\big] = \frac{1}{N}\,E^{\Gamma}_{\text{supercell}(N)} ,
$$

for one specified operator. That identity is the H8 / 4×1×1 and
H16 / 8×1×1 test in the table. The cyclic cluster model works on the
supercell side of it: one real Γ problem on the finite cluster, with
Wigner-Seitz weights choosing which image of each partner enters every
integral so that each pair is counted once
({vibeqc-cite}`bredow_geudtner_jug_ccm_2001`, {vibeqc-cite}`peintinger_ccm_2014`).
Tori of different $N$ differ by a finite-size error; for exact exchange its
leading part comes from the integrable $q \to 0$ Coulomb singularity
({vibeqc-cite}`gygi_baldereschi_exx_1986`, {vibeqc-cite}`sundararaman_arias_exx_2013`),
the origin of the slow drift with torus size discussed above.

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
