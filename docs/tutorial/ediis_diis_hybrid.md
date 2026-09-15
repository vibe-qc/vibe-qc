# Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS

Vanilla Pulay DIIS is the workhorse SCF accelerator for organic-
chemistry-style problems with a clear HOMO-LUMO gap. It is not the
right tool for **broken-symmetry singlets, transition-metal
complexes, near-degenerate open-shell radicals, and any system
with a small or vanishing gap**, in that regime DIIS oscillates or
plateaus far from the true minimum, sometimes hours into a
calculation that should have taken minutes. The default molecular
accelerator in vibe-qc v0.8.0, `SCFAccelerator.EDIIS_DIIS`, 
rescues those cases by switching the SCF history-mixing rule based
on how close the iterates are to the fixed point.

This tutorial walks through

- why DIIS fails on stiff cases (one-paragraph theory),
- the Garza-Scuseria 2012 EDIIS+DIIS hybrid recipe,
- a clean reproducer where vanilla DIIS gets stuck and EDIIS+DIIS
  doesn't, and
- the dial-by-dial recommendations for what to change when the
  hybrid alone isn't enough.

It's the molecular companion to the periodic
[SCF-convergence tutorial](periodic_scf_convergence.md).

## One-paragraph theory

DIIS (Pulay 1980) extrapolates to the SCF fixed point by minimising
the norm of the **commutator error vector**
$\mathbf{e} = \mathbf{F}\mathbf{D}\mathbf{S} - \mathbf{S}\mathbf{D}\mathbf{F}$
over a sliding window of past Fock + density pairs. It works
beautifully when the iterates already sit in the asymptotic regime
(roughly $\lVert\mathbf{e}\rVert_F < 10^{-1}$). Outside that regime, 
small HOMO-LUMO gap, broken-symmetry start, transition-metal
ligand-field-degenerate orbitals, the commutator is dominated by
modes that are physically irrelevant to convergence, and the DIIS
extrapolation oscillates between near-degenerate minima.

**EDIIS** (Energy-DIIS, Kudin, Scuseria, Cancès 2002) replaces the
commutator objective with a quadratic energy model on the convex hull
of the stored (F, D) pairs and minimises *that* over a simplex. For HF,
the model is the exact energy of the hull density; for KS-DFT it is a
quadratic interpolation, not a guaranteed upper bound. Far from
convergence, EDIIS lands on a physically meaningful descent direction
even when the commutator is dominated by noise. Close to the fixed
point, energy differences become less sensitive than the commutator
residual and positive-simplex interpolation is less effective than
Pulay extrapolation, so EDIIS can plateau before tight convergence.

**EDIIS+DIIS** (Garza, Scuseria 2012) takes the obvious step: use
EDIIS while the commutator is large, switch to plain DIIS once
$\lVert\mathbf{e}\rVert_F$ drops below a threshold (default
`1e-1`, matching PySCF and ORCA). The result is robust convergence
on the stiff cases plus the asymptotic quadratic-like behaviour DIIS
delivers near the minimum.

## The recipe

For molecular SCF in v0.8.0, EDIIS+DIIS is **already the default**, 
every `RHFOptions`, `UHFOptions`, `RKSOptions`, `UKSOptions`
instance starts with:

```python
opts.scf_accelerator             = vq.SCFAccelerator.EDIIS_DIIS
opts.ediis_diis_switch_threshold = 1e-1     # ‖e‖_F below this → DIIS
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

## A reproducer where vanilla DIIS struggles

Stretched H₂O at HF/cc-pVDZ is the classic textbook example of a
near-singlet-triplet near-degeneracy that breaks vanilla DIIS.
At the equilibrium geometry the HOMO-LUMO gap is healthy (>10 eV);
stretch the bonds to ~1.6 Å and the gap collapses as the
closed-shell wavefunction becomes a poor reference for the
multi-reference dissociation limit. DIIS responds by oscillating
between two near-degenerate density solutions.

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

# Baseline: vanilla DIIS — note the explicit override of the
# v0.8.0 EDIIS_DIIS default.
opts_diis = vq.RHFOptions()
opts_diis.scf_accelerator = vq.SCFAccelerator.DIIS
opts_diis.max_iter        = 60
opts_diis.conv_tol_energy = 1e-8

# Default (v0.8.0+) — EDIIS+DIIS hybrid.
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

What you should see (numbers fluctuate ±1 iter run-to-run depending
on tie-breaks in the simplex solver):

```text
DIIS         : converged = False, iters = 60, E = -75.94...
EDIIS+DIIS   : converged = True,  iters = 17, E = -75.95...
```

The hybrid not only converges where vanilla DIIS doesn't, it
converges to a slightly lower (correct) energy because vanilla
DIIS's oscillating iterates never sit at the minimum.

```{tip}
For a sharper demonstration, stretch the bonds further (to 2.0 Å or
beyond). Both methods will struggle, but EDIIS+DIIS still has a
much better chance, and the gap between them widens.
```

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
$\lVert\mathbf{e}\rVert_F > 10^{-1}$ the extrapolator solves the
simplex
$$
c^\star = \arg\min_{c \in \Delta_n}\;
   \sum_i c_i \,E[\mathbf{F}_i,\mathbf{D}_i]
 - \tfrac{1}{2}\sum_{i,j} c_i c_j \,\mathrm{tr}\!\left[(\mathbf{F}_j - \mathbf{F}_i)(\mathbf{D}_j - \mathbf{D}_i)\right]
$$
(Kudin et al. 2002, eq. 7) where $\Delta_n$ is the unit simplex.
Once $\lVert\mathbf{e}\rVert_F$ drops below the threshold the same
slot fills with the classic Pulay least-squares fit on the AO
commutator. The switch is one-way: once DIIS takes over, the
hybrid does not flip back.

In the run above, the EDIIS phase typically runs for the first 6-10
iterations until $\lVert\mathbf{e}\rVert_F$ falls below $10^{-1}$;
the asymptotic DIIS phase then nails the residual ~7 orders of
magnitude in 5-7 iterations. The `.scf.jsonl` structured log
records the per-iteration norm so you can plot the crossover
yourself:

```python
import json, matplotlib.pyplot as plt
rows = [json.loads(l) for l in open("output-stretched.scf.jsonl")
        if '"event": "iter"' in l]
plt.semilogy([r["iter"] for r in rows],
             [r["commutator_norm"] for r in rows],
             marker="o")
plt.axhline(0.1, color="gray", ls="--",
            label="EDIIS → DIIS switch threshold")
plt.xlabel("SCF iter")
plt.ylabel("‖F D S − S D F‖_F")
plt.legend()
plt.savefig("ediis-diis-switch.png", dpi=150)
```

(`structured_log=True` on `run_job` writes the `.scf.jsonl`, 
see [output_files](../user_guide/output_files.md) § Structured
log.)

## The full converger menu

EDIIS+DIIS is only the default. It is one of five accelerators, and the
convergence aids and the second-order finalizers compose with whichever
accelerator you pick. The complete set, all on `RHFOptions` /
`UHFOptions` / `RKSOptions` / `UKSOptions`:

**Accelerator** (`opts.scf_accelerator`, pick one):

| Value | Algorithm | Use |
|-------|-----------|-----|
| `SCFAccelerator.DIIS` | Pulay 1980/1982 (commutator error; molecular UKS uses the orthonormal `Xᵀ(FDS−SDF)X` form) | Standard workhorse |
| `SCFAccelerator.KDIIS` | Kollmar 1997 (orbital-gradient error; ORCA's opt-in `!KDIIS`) | Often more robust where plain DIIS oscillates on transition-metal / open-shell cases |

The common Pulay kernel scale-normalizes the Gram block before solving its
bordered system. Uniformly scaling every residual therefore leaves the fitted
Fock unchanged in floating-point arithmetic, as the exact constrained
minimization requires.
| `SCFAccelerator.EDIIS` | Energy-DIIS (Kudin/Scuseria/Cancès 2002) | Diagnostic on its own; pure EDIIS plateaus near convergence |
| `SCFAccelerator.ADIIS` | Augmented Roothaan-Hall (Hu/Yang 2010) | Diagnostic; Garza/Scuseria found it near-identical to EDIIS |
| `SCFAccelerator.EDIIS_DIIS` | Garza/Scuseria 2012 hybrid | **Default**: EDIIS while far out, plain DIIS near convergence |

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

H₂O / cc-pVDZ at the stretched geometry: ~50 ms per SCF iteration
on a modern laptop. The full DIIS run (60 max iters, no
convergence) finishes in ~3 s; EDIIS+DIIS (~17 iters, converged)
in ~1 s. Peak memory is well under 100 MB.

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
