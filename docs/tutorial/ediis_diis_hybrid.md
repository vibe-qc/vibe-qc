# Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS

Vanilla Pulay DIIS is the workhorse SCF accelerator for organic-
chemistry-style problems with a clear HOMO-LUMO gap. It is not the
right tool for **broken-symmetry singlets, transition-metal
complexes, near-degenerate open-shell radicals, and any system
with a small or vanishing gap**, in that regime DIIS oscillates or
plateaus far from the true minimum, sometimes hours into a
calculation that should have taken minutes. The default molecular
accelerator in vibe-qc since v0.8.0, `SCFAccelerator.EDIIS_DIIS`, 
switches the SCF history-mixing rule based on how close the iterates are
to the fixed point. Every accelerator on this page is a production route
(label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)).

This tutorial walks through

- why DIIS fails on stiff cases (the [Theory](#theory) section: the DIIS
  error vector and linear system, the EDIIS energy model, and the switch
  between them),
- the Garza-Scuseria 2012 EDIIS+DIIS hybrid recipe,
- a stretched-water comparison of the two accelerators, including what it
  does and does not show on a current build, and
- the dial-by-dial recommendations for what to change when the
  hybrid alone isn't enough.

It's the molecular companion to the periodic
[SCF-convergence tutorial](periodic_scf_convergence.md).

## Theory

Both accelerators are extrapolations over a short history of Fock matrices
$\mathbf F_i$ and densities $\mathbf D_i$ from earlier iterations. They
differ in which quantity they make small.

**DIIS** (Pulay 1980, 1982; {vibeqc-cite}`pulay_diis_1980`,
{vibeqc-cite}`pulay_diis_1982`) measures how far each iterate is from
self-consistency with the commutator error vector, which vanishes exactly
when $\mathbf F$ and $\mathbf D$ share an eigenbasis,

$$
\mathbf e_i = \mathbf F_i\mathbf D_i\mathbf S - \mathbf S\mathbf D_i\mathbf F_i ,
\qquad
\begin{pmatrix} \mathbf B & \mathbf 1 \\ \mathbf 1^{\mathsf T} & 0 \end{pmatrix}
\begin{pmatrix} \mathbf c \\ \lambda \end{pmatrix}
=
\begin{pmatrix} \mathbf 0 \\ 1 \end{pmatrix},
\qquad
B_{ij} = \operatorname{tr}\bigl(\mathbf e_i^{\mathsf T}\mathbf e_j\bigr),
$$

where $\mathbf S$ is the AO overlap matrix, $\mathbf 1$ a vector of ones and
$\lambda$ the Lagrange multiplier of the constraint $\sum_i c_i = 1$. Solving
the bordered linear system gives the coefficients that minimise
$\lVert\sum_i c_i\mathbf e_i\rVert$, and the next Fock matrix is
$\mathbf F = \sum_i c_i\mathbf F_i$. This works beautifully once the
iterates already sit in the asymptotic regime (roughly, an error norm
below the switching threshold defined below), where the error is nearly
linear in the parameters. Outside it, with a small HOMO-LUMO gap, a broken-symmetry start
or ligand-field-degenerate transition-metal orbitals, the commutator is
dominated by modes that are irrelevant to convergence, the linear model is
poor, and the extrapolation can oscillate between near-degenerate minima.

**EDIIS** (Energy-DIIS, {vibeqc-cite}`kudin_ediis_2002`) replaces the
commutator objective with an energy model. It interpolates the stored
densities and minimises the energy of the interpolated density over the
convex set $c_i \ge 0$, $\sum_i c_i = 1$ (a simplex),

$$
E_\text{EDIIS}(\mathbf c) = \sum_i c_i\,E_i
 - \tfrac12 \sum_{ij} c_i c_j\,
   \operatorname{tr}\bigl[(\mathbf D_i - \mathbf D_j)(\mathbf F_i - \mathbf F_j)\bigr] ,
$$

with $E_i$ the energy of iterate $i$. For Hartree-Fock this is the exact
energy of $\mathbf D(\mathbf c)=\sum_i c_i\mathbf D_i$, because the energy is
quadratic in the density; for Kohn-Sham DFT it is a quadratic model, not a
guaranteed upper bound. Because the coefficients are non-negative, EDIIS
never extrapolates: far from convergence it lands on a physically meaningful
descent direction even when the commutator is noise. Close to the fixed
point, energy differences become less sensitive than the commutator and
positive-simplex interpolation cannot extrapolate, so EDIIS can plateau
before tight convergence.

**EDIIS+DIIS** ({vibeqc-cite}`garza_scuseria_2012`) combines them with a
switching criterion. The rule implemented here is a one-way, hard switch on
a size-intensive form of the commutator norm, the RMS matrix element

$$
\epsilon = \frac{\lVert\mathbf e\rVert_F}{\sqrt{n_\text{blocks}}\,N_\text{bf}} ,
$$

with $N_\text{bf}$ the number of basis functions and $n_\text{blocks}$ equal
to 1 for restricted and 2 for unrestricted runs (one matrix per spin). Use the
EDIIS coefficients while $\epsilon$ is above `ediis_diis_switch_threshold`
(default $10^{-1}$; open-shell UHF uses a tighter
`ediis_diis_switch_threshold_open_shell = 1e-3` when
$n_\alpha \ne n_\beta$), and the classic Pulay coefficients from then on.
The result is robust early convergence plus the fast asymptotic behaviour
of DIIS. Because the criterion is per matrix element, EDIIS engages only
when $\lVert\mathbf e\rVert_F$ exceeds $0.1\,N_\text{bf}$, about 2.4 for
water in cc-pVDZ; a small molecule with a decent starting density can sit
below that from the first iteration, in which case the hybrid runs as plain
DIIS throughout. The remaining failure modes are the ones no acceleration rule can
fix: the iteration converges to a stationary point that is not the state you
wanted (fix it with the initial guess or MOM), or the Hessian at the
solution has negative curvature (a stability question, not a convergence
one).

## The recipe

For molecular SCF in v0.8.0, EDIIS+DIIS is **already the default**, 
every `RHFOptions`, `UHFOptions`, `RKSOptions`, `UKSOptions`
instance starts with:

```python
opts.scf_accelerator             = vq.SCFAccelerator.EDIIS_DIIS
opts.ediis_diis_switch_threshold = 1e-1     # intensive ‖e‖ below this → DIIS (open-shell UHF: ediis_diis_switch_threshold_open_shell = 1e-3, #377)
opts.diis_subspace_size          = 8         # rolling history cap
```

EDIIS, ADIIS, and their DIIS hybrids use an exact active-face search and
accept history caps from 2 through 12. Plain DIIS, KDIIS, R-CDIIS, and
AD-CDIIS do not use that exponential search and may retain deeper histories.

If you've been writing scripts that explicitly set
`scf_accelerator = SCFAccelerator.DIIS`, drop the line, the default
is now the production hybrid. For PySCF / vibe-qc-pre-v0.8 parity
runs that need vanilla DIIS:

```python
opts.scf_accelerator = vq.SCFAccelerator.DIIS
```

## A stretched-water comparison

Stretched H₂O at HF/cc-pVDZ is the classic small example of a collapsing
gap. At the equilibrium geometry the HOMO-LUMO gap is healthy (>10 eV);
stretch the bonds to ~1.6 Å and the gap shrinks as the closed-shell
wavefunction becomes a poor reference for the multi-reference dissociation
limit. The script below runs the two accelerators side by side.

Working directory:

```
~/vibeqc-runs/stretched-h2o/
    input-stretched.py
    h2o-stretched.xyz
```

`h2o-stretched.xyz`:

```text
3
H2O, both bonds stretched to 1.60 Angstrom, HOH angle 104.5 deg
O   0.00000   0.00000   0.00000
H   0.00000   1.26824  -0.97920
H   0.00000  -1.26824  -0.97920
```

`input-stretched.py`:

```python
from pathlib import Path
import vibeqc as vq

HERE = Path(__file__).parent

mol = vq.Molecule.from_xyz(HERE / "h2o-stretched.xyz")
basis = vq.BasisSet(mol, "cc-pvdz")

# Baseline: vanilla DIIS, an explicit override of the EDIIS_DIIS default.
opts_diis = vq.RHFOptions()
opts_diis.scf_accelerator = vq.SCFAccelerator.DIIS
opts_diis.max_iter        = 60
opts_diis.conv_tol_energy = 1e-8

# Default: the EDIIS+DIIS hybrid.
opts_hybrid = vq.RHFOptions()
opts_hybrid.max_iter        = 60
opts_hybrid.conv_tol_energy = 1e-8

res_diis   = vq.run_rhf(mol, basis, opts_diis)
res_hybrid = vq.run_rhf(mol, basis, opts_hybrid)

print(f"DIIS         : converged = {res_diis.converged}, "
      f"iters = {res_diis.n_iter}, E = {res_diis.energy:.8f}")
print(f"EDIIS+DIIS   : converged = {res_hybrid.converged}, "
      f"iters = {res_hybrid.n_iter}, E = {res_hybrid.energy:.8f}")
```

On a vibe-qc 0.17.10 development build (Apple M-series laptop) both runs
take about a second and print:

```text
DIIS         : converged = True, iters = 11, E = -75.73637199
EDIIS+DIIS   : converged = True, iters = 11, E = -75.73637199
```

The two accelerators tie here, and the reason is visible in the SCF trace
below: the first commutator norm is 1.33, and the switch to EDIIS needs
$\lVert\mathbf e\rVert_F > 0.1\,N_\text{bf} = 2.4$ for this 24-function
basis, so the "hybrid" ran as plain DIIS from iteration 1. That is an honest
result, not a failed demonstration: the default initial guess (PATOM for a
closed-shell light molecule, see the
[initial-guess walkthrough](initial_guess_walkthrough.md)) already puts this
small molecule close enough to the solution. The accelerator matters when the
start is poor. Repeating the
comparison with the core-Hamiltonian guess (`opts.initial_guess =
vq.InitialGuess.HCORE`) at longer bond lengths gives, for the same H-O-H
angle and basis:

| O-H / Å | guess | plain DIIS | EDIIS+DIIS |
|---|---|---|---|
| 1.6 | default | 11 iterations, -75.73751113 | 11 iterations, same energy |
| 2.5 | default | 14 iterations, -75.42794787 | 13 iterations, same energy |
| 2.5 | HCORE | 30 iterations, -75.46295494 | 14 iterations, -75.42794787 |
| 3.0 | default | 12 iterations, -75.42522732 | 13 iterations, same energy |
| 3.0 | HCORE | 31 iterations, -75.43565952 | 20 iterations, -75.43053824 |

From the poor start the hybrid needs about half the iterations, which is
what it is for. But read the energy column before celebrating: the
accelerator also chooses *which* restricted Hartree-Fock stationary point is
reached. At 2.5 Å plain DIIS from HCORE ends on a solution 35 mHa lower than
the other three runs, and at 3.0 Å the two HCORE runs end on different
states. A fast, converged SCF is not evidence that it converged to the
state you wanted. Compare final energies across settings, and check the
result with the [orbital-stability tools](opentrustregion_stability.md)
before trusting a stretched-geometry reference.

## What changes inside the iteration

Both runs walk through the same SCF inner loop
([scf_convergence § Application order](../user_guide/scf_convergence.md)):

```
for iter in 1..max_iter:
    1. D_used  = damping(D_prev, D)
    2. F       = Hcore + G[D_used]
    3. F       = level_shift(F, S, D_used)
    4. F_extra = accelerator.extrapolate(F, ...)   # ← what differs
    5. C, eps  = step(F_extra, ...)
    6. D       = build_density(C)
    7. check convergence
```

The only thing EDIIS+DIIS swaps out is step 4. While
the switching norm $\epsilon$ of the Theory section exceeds
$10^{-1}$ the extrapolator solves the simplex
$$
c^\star = \arg\min_{c \in \Delta_n}\;
   \sum_i c_i \,E[\mathbf{F}_i,\mathbf{D}_i]
 - \tfrac{1}{2}\sum_{i,j} c_i c_j \,\mathrm{tr}\!\left[(\mathbf{F}_j - \mathbf{F}_i)(\mathbf{D}_j - \mathbf{D}_i)\right]
$$
(Kudin et al. 2002, eq. 7) where $\Delta_n$ is the unit simplex.
Once $\epsilon$ drops below the threshold the same
slot fills with the classic Pulay least-squares fit on the AO
commutator. The switch is one-way: once DIIS takes over, the
hybrid does not flip back.

In the 1.6 Å run above there is no EDIIS phase at all: the commutator
norm $\lVert\mathbf e\rVert_F$ is 1.33 and 1.46 at the first two iterations
($\epsilon = 0.056$ and 0.061 with $N_\text{bf} = 24$, both under the
threshold), falls to 0.097 at the third, and the DIIS steps take it down
through 2e-2, 3e-3, 7e-4, 3e-4, 2e-4, 4e-5, 2e-6 to 2e-7 at iteration 11. The
`.scf.jsonl` structured log records the per-iteration norm (`grad_norm`,
the same quantity as the `||[F,DS]||` column of the `.out`), so you can plot
where a run crosses the switch yourself. The snippet below uses the poorer
HCORE start, for which the first two norms are 4.35 and 4.56, above the
threshold of 2.4, so EDIIS coefficients are used for those two iterations;
the norm then reads 1.86, 1.63, 1.31, 0.15 and falls to 7.9e-8 at iteration
15, all with DIIS:

```python
import json
import matplotlib.pyplot as plt
import vibeqc as vq

mol = vq.Molecule.from_xyz("h2o-stretched.xyz")
nbf = vq.BasisSet(mol, "cc-pvdz").nbasis
opts = vq.RHFOptions()
opts.max_iter, opts.conv_tol_energy = 60, 1e-8
opts.initial_guess = vq.InitialGuess.HCORE          # a poor start engages EDIIS
vq.run_job(mol, basis="cc-pvdz", method="rhf", rhf_options=opts,
           structured_log=True, output="stretched")   # writes stretched.scf.jsonl

rows = [json.loads(line) for line in open("stretched.scf.jsonl")]
scf = [r for r in rows if r["event"] == "scf_iter"]
plt.semilogy([r["iter"] for r in scf], [r["grad_norm"] for r in scf], marker="o")
plt.axhline(0.1 * nbf, color="gray", ls="--",
            label="EDIIS to DIIS switch (0.1 x N_bf)")
plt.xlabel("SCF iteration")
plt.ylabel("‖F D S − S D F‖_F")
plt.legend()
plt.savefig("ediis-diis-switch.png", dpi=150)
```

(`structured_log=True` on `run_job` writes the `.scf.jsonl`, 
see [output_files](../user_guide/output_files.md) § Structured
log.)

## The full converger menu

EDIIS+DIIS is only the default. It is one of eight accelerators, and the
convergence aids and the second-order finalizers compose with whichever
accelerator you pick. The complete set, all on `RHFOptions` /
`UHFOptions` / `RKSOptions` / `UKSOptions`:

**Accelerator** (`opts.scf_accelerator`, pick one):

| Value | Algorithm | Use |
|-------|-----------|-----|
| `SCFAccelerator.DIIS` | Pulay 1980/1982 (commutator error; molecular UKS uses the orthonormal `Xᵀ(FDS−SDF)X` form) | Standard workhorse |
| `SCFAccelerator.KDIIS` | Kollmar 1997 (orbital-gradient error; ORCA's opt-in `!KDIIS`) | Often more robust where plain DIIS oscillates on transition-metal / open-shell cases |
| `SCFAccelerator.EDIIS` | Energy-DIIS (Kudin/Scuseria/Cancès 2002) | Diagnostic on its own; pure EDIIS plateaus near convergence |
| `SCFAccelerator.ADIIS` | Augmented Roothaan-Hall (Hu/Yang 2010) | Diagnostic; Garza/Scuseria found it near-identical to EDIIS |
| `SCFAccelerator.EDIIS_DIIS` | Garza/Scuseria 2012 hybrid | **Default**: EDIIS while far out, plain DIIS near convergence |
| `SCFAccelerator.ADIIS_DIIS` | ADIIS then DIIS, the same switch as the hybrid | Alternative to the default hybrid |
| `SCFAccelerator.R_CDIIS`, `SCFAccelerator.AD_CDIIS` | Chupin et al. 2021 adaptive-depth CDIIS | Removes the fixed history size; see [Letting DIIS choose its own depth](adaptive_diis.md) |

The common Pulay kernel scale-normalizes the Gram block before solving its
bordered system. Uniformly scaling every residual therefore leaves the fitted
Fock unchanged in floating-point arithmetic, as the exact constrained
minimization requires.

**Convergence aids** (compose freely with the accelerator above):

* `opts.damping` (static) plus dynamic damping, for early-iteration oscillation;
* `opts.fock_mixing` (CRYSTAL-style FMIXING, static Fock-matrix mixing);
* `opts.level_shift` (Saunders-Hillier), to hold the occupied and virtual blocks apart;
* `opts.mom` (maximum overlap), to hold a chosen occupation pattern across iterations;
* `opts.smearing_temperature` (Fermi-Dirac, periodic only).

**Second-order finalizer** (`opts.newton_threshold` / `opts.trah_threshold` / `opts.soscf_threshold`, pick zero or one): switch on Newton, TRAH, or SOSCF once the error-vector norm drops below the threshold, finishing the last few orders of magnitude in quadratic time (see the next section).

Per-converger parameters, the periodic `convergence="auto"` strategy, and
the unified inner-loop application order are in
[user_guide/scf_convergence](../user_guide/scf_convergence.md).

## When the hybrid alone isn't enough

EDIIS+DIIS does most of the work for routine stiff systems. For
the really hostile cases (transition-metal complexes with
ligand-field-degenerate d-shells, large-active-space broken-
symmetry singlets) a few more dials matter:

| Symptom | Fix |
|---|---|
| EDIIS+DIIS plateaus near $10^{-3}$ Ha and won't go further | Add a **second-order finalizer**, `opts.newton_threshold = 1.0` activates the Phase D2c Newton step when $\lVert\mathbf{e}\rVert_F < 1$. Newton finishes the last 3-5 orders of magnitude in quadratic time. |
| The hybrid oscillates from iter 1 (degenerate occupations) | Add a **level shift**, `opts.level_shift = 0.2` keeps occupied / virtual blocks apart while the density settles. Compose freely with EDIIS+DIIS. |
| Wrong SCF stationary point (e.g. an excited determinant) | **Pin the initial guess** to a better starting density, `opts.initial_guess = vq.InitialGuess.SAD` instead of the AUTO default, or use the [MOM tutorial](initial_guess_walkthrough.md) for excited-state targeting. |
| Newton's per-iteration cost is too high (each Newton step does multiple Fock builds) | Use **TRAH** (`opts.trah_threshold = 1.0`), same Hessian, adaptive trust radius, more robust to overshooting, or **SOSCF** (`opts.soscf_threshold = 1.0`) for the cheap-per-step alternative on hybrid functionals. |
| You want vanilla DIIS for parity with a pre-v0.8.0 reference | `opts.scf_accelerator = vq.SCFAccelerator.DIIS` |

The full decision tree, including which finalizer plays well with
KS-DFT and what `newton_opts.cg_tol` / `trust_radius` to start
from, is in
[user_guide/scf_convergence](../user_guide/scf_convergence.md).

## Resources

H₂O / cc-pVDZ at the stretched geometry: about 1 s for either
accelerator (11 iterations each) on a modern laptop, the run's own
memory estimate is 0.15 GB.

## References

- **DIIS**, P. Pulay, "Convergence acceleration of iterative
  sequences. The case of SCF iteration," *Chem. Phys. Lett.* **73**,
  393 (1980); and "Improved SCF convergence acceleration,"
  *J. Comput. Chem.* **3**, 556 (1982).
- **EDIIS**, K. N. Kudin, G. E. Scuseria, E. Cancès, "A black-box
  self-consistent field convergence algorithm: One step closer,"
  *J. Chem. Phys.* **116**, 8255 (2002).
  [doi:10.1063/1.1470195](https://doi.org/10.1063/1.1470195)
- **EDIIS+DIIS hybrid**, A. J. Garza, G. E. Scuseria, "Comparison
  of self-consistent field convergence acceleration techniques,"
  *J. Chem. Phys.* **137**, 054110 (2012).
  [doi:10.1063/1.4740249](https://doi.org/10.1063/1.4740249)

Both EDIIS papers ship in the bundled
[citation database](../user_guide/citations.md); any `run_job` that
hits the `EDIIS` or `EDIIS_DIIS` route will cite them automatically
in the run's `.bibtex` and `.references` siblings.

## Next

- [`scf_convergence`](../user_guide/scf_convergence.md), full SCF-
  convergence reference: KDIIS, level shifts, FMIXING, the four
  second-order finalizers (Newton / TRAH / SOSCF / quadratic
  fallback), and the unified iteration order.
- [Initial-guess walkthrough](initial_guess_walkthrough.md), the
  *other* lever for stiff SCFs: start from a better density.
- [Periodic SCF convergence](periodic_scf_convergence.md), the
  periodic-side companion. Damping, DIIS, level shifts for
  tight-cell crystals.
