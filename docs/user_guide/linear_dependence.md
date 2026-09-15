---
myst:
  html_meta:
    "description": "Linear-dependence and screening in periodic SCF, diagnosing critical-severity overlap matrices, and the choice between basis-set redesign and screening tightening. CRYSTAL EIGS / TOLINTEG equivalents in vibe-qc."
    "og:title": "Linear dependence in periodic SCF, vibe-qc user guide"
    "og:description": "When the AO overlap matrix loses positive-definiteness in a solid-state SCF: diagnostic stack, fix stack, and recommended workflows by system class."
---

# Linear dependence and screening in periodic SCF

Your solid-state SCF gives an energy 30× too negative, or it
doesn't converge at all, and the error message says something
about a *critical* overlap matrix. This chapter walks you from
that error to the right fix.

## What's a near-singular overlap matrix?

For an AO basis $\{\chi_\mu\}$ the overlap matrix is

$$
S_{\mu\nu} = \langle \chi_\mu \mid \chi_\nu \rangle.
$$

$S$ is a Gram matrix and so is **positive semi-definite by
construction**: every eigenvalue $\lambda_i \ge 0$. Numerically,
finite precision and finite cutoffs can drive $\lambda_\text{min}$
below zero, and once $S$ has a negative eigenvalue the canonical
$S^{-1/2}$ used to orthogonalise the basis blows up, the SCF
either fails outright or converges to a physically meaningless
single-determinant.

The **condition number**

$$
\kappa(S) \;=\; \lambda_\text{max} / \lambda_\text{min}
$$

quantifies how close to singular $S$ is. In molecular calculations
$\kappa \lesssim 10^4$ is typical; in periodic calculations on
densely packed crystals with molecularly-designed basis sets,
$\kappa$ can blow up to $10^{12}$ or beyond and $\lambda_\text{min}$
can go *negative*. The dense packing of atoms drives diffuse
Gaussian primitives toward overlap with their neighbours, and
because the SCF needs the inverse-square-root of $S$, even a
mildly under-conditioned $S$ propagates into garbage densities.

The CRYSTAL solid-state code historically refused to silently
truncate near-singular $S$ ([CRYSTAL23 manual p. 103,
398](https://www.crystal.unito.it/Manuals/crystal23.pdf)), its
EIGS keyword aborts before SCF if the overlap is bad, and the
recommended fix is **basis-set redesign**, not in-SCF truncation.
vibe-qc takes the same stance, with the same diagnostic surfaces
the CRYSTAL user is used to.

## The two failure modes

A "critical" overlap can come from one of two distinct causes,
and the right fix is different in each:

1. **Genuine basis-set linear dependence.** Too many diffuse
   Gaussian primitives for the lattice geometry; primitives on
   neighbouring atoms cover almost the same spatial region and
   their AOs become near-identical linear combinations. Fix at
   basis-construction time: use a basis designed for solids
   (pob-TZVP), or filter primitives below an exponent threshold
   (`vq.make_basis(..., exp_to_discard=0.1)`).

2. **Under-converged exchange screening.** The negative eigenvalues
   are an artefact of too-loose lattice-sum truncation: contributions
   that should have been kept got Schwarz-screened away, leaving
   the overlap matrix incomplete. Fix at SCF-options time: tighten
   `lat_opts.cutoff_bohr` and `lat_opts.schwarz_threshold`. CRYSTAL
   surfaces this as the ITOL4/ITOL5 distinction in the TOLINTEG
   keyword ([manual p. 130, 398](https://www.crystal.unito.it/Manuals/crystal23.pdf));
   the manual explicitly notes that what looks like linear
   dependence at runtime is *often* actually screening
   under-convergence.

The empirical finding from the v0.7 work in vibe-qc is concrete:
**LiH conventional rocksalt with STO-3G**, the textbook
"non-PSD overlap" case people reach for to demonstrate linear
dependence, diagnoses cleanly as `screening_undertight`, not
as a basis-set problem. Tightening `cutoff_bohr` by a factor of
2 takes the worst $\lambda_\text{min}$ from -0.16 to +0.07.

The screening fix is a strict subset of the basis-set fix
(tightening cutoffs always converges to the correct answer; a
new basis changes what "correct" means), so when the
disambiguation is inconclusive vibe-qc recommends the screening
fix first.

## The diagnostic stack

vibe-qc surfaces three diagnostics. They differ in *when* they
run and *what* they cost.

### `vq.eigs_preflight`, standalone, no SCF

CRYSTAL's `EIGS` keyword in vibe-qc form. Builds $S(\mathbf{k})$
at every requested k-point, diagonalises, returns a structured
report. **No SCF cycles run.** Use it before committing to an
expensive multi-iteration calculation:

```python
import vibeqc as vq

# Build a system + basis as you would for SCF
sysp, basis = ...   # see the `lih_pob_tzvp_solid_state` tutorial for a worked LiH setup
kpts = vq.KPoints.monkhorst_pack(sysp, [4, 4, 4])

report = vq.eigs_preflight(sysp, basis, kpts)
print(vq.format_eigs_report(report, basis_label="LiH/STO-3G"))
```

The report carries one entry per k-point with $\lambda_\text{min}$,
$\kappa$, and a per-k severity (`ok` / `warn` / `error` /
`critical`). The aggregate severity is the worst across all
k-points.

The k-point dimension matters: a basis can be perfectly fine at
$\Gamma$ and lose positive-definiteness at the zone boundary
(see Searle, Bernasconi, Harrison, [ARCHER eCSE04-16,
2017](https://www.archer.ac.uk/community/eCSE/eCSE04-16/)).
`eigs_preflight` catches that; a $\Gamma$-only check doesn't.

### `vq.scf_preflight_overlap_check`, runs at SCF startup

Every periodic SCF driver in v0.7+ runs this check before the
first Fock build. If the overlap is `critical` (non-PSD), the
driver raises `vq.LinearDependenceError` with a citation-rich
error message recommending the appropriate fix. You don't have
to call this yourself; the SCF entry points wire it in.

To opt out (run SCF anyway despite a critical-severity overlap)
pass `allow_critical=True` to the SCF driver. Read the §
[Last resort](#last-resort-explicit-opt-in-orthogonalisation)
section before doing this.

### `vq.disambiguate_critical_overlap`, basis vs screening

When `eigs_preflight` or the in-SCF preflight fires `critical`,
this re-runs at tightened cutoffs and classifies the cause:

```python
report = vq.eigs_preflight(sysp, basis, kpts)
if report.severity == "critical":
    diag = vq.disambiguate_critical_overlap(sysp, basis, kpts)
    print(vq.format_disambiguation_report(diag))
```

Output is one of three classifications:

| Classification | Meaning | Recommended fix |
|---|---|---|
| `basis_set_problem` | Tightening cutoffs doesn't help | Use a basis designed for solids (pob-TZVP), or `vq.make_basis(..., exp_to_discard=0.1)` |
| `screening_undertight` | Tightening cutoffs restores PSD | Tighten `cutoff_bohr` / `schwarz_threshold`; or just let `auto_optimize_truncation=True` (the default) handle it |
| `inconclusive` | Mixed evidence | Try the screening fix first (strict subset of the basis fix) |

## The fix stack

In order of preference, from "least invasive" to "last resort":

### 1. Use a basis designed for solids

The `pob-*` family ships in vibe-qc and is what CRYSTAL users
have used for two decades:

```python
basis = vq.BasisSet(mol, "pob-tzvp-rev2")   # Vilela Oliveira 2019
# or
basis = vq.BasisSet(mol, "pob-tzvp")        # Peintinger 2013
# or
basis = vq.BasisSet(mol, "pob-dzvp-rev2")   # Vilela Oliveira 2019
```

These are **not** simply re-tabulated molecular basis sets, the
diffuse primitives have been re-optimised against a condition-number
penalty so that the overlap matrix stays well-conditioned even
in tight ionic crystals. See [Peintinger, Vilela Oliveira, Bredow,
*J. Comput. Chem.* **34**, 451 (2013)](https://doi.org/10.1002/jcc.23153)
and [Vilela Oliveira, Laun, Peintinger, Bredow, *J. Comput. Chem.*
**40**, 2364 (2019)](https://doi.org/10.1002/jcc.26013) for the
design philosophy and per-element parametrisation.

The [user guide on basis sets](basis_sets.md) and
[Why solid-state calculations use `pob-TZVP`](../tutorial/pob_tzvp.md) walk through using
the pob family in practice.

### 2. Tighten screening (when the disambiguator says so)

If `disambiguate_critical_overlap` returns `screening_undertight`,
the loose cutoffs are at fault, not the basis. Tighten manually:

```python
opts = vq.PeriodicKSOptions()
opts.lattice_opts.cutoff_bohr        = 20.0     # default ~10
opts.lattice_opts.schwarz_threshold  = 1e-14    # default 1e-12
```

Or, in v0.7+ this happens automatically by default, let
`auto_optimize_truncation=True` bisect both knobs jointly to
find the loosest combination that keeps $S$ PSD at every
k-point:

```python
result = vq.run_rks_periodic_scf(
    sysp, basis, kpts, opts,
    auto_optimize_truncation=True,    # default ON in v0.7+
)
```

The optimiser short-circuits at one evaluation when the starting
settings are already PSD, and runs at most 8 evaluations
(typical: 1-3). The two knobs move proportionally so neither
runs far from default while the other stays put.

#### Sizing the cutoff analytically instead of searching for it

`vibeqc.lattice_screening.bloch_overlap_cutoff_bohr(basis, system)`
returns a cutoff derived from the basis in closed form, by bounding
the largest primitive-pair overlap the truncation would drop. It
needs no search and no repeated overlap builds, and it is what the
periodic-LCAO literature has always used: CRYSTAL's ITOL1 thresholds
the overlap of the two shells' adjoined Gaussians rather than a
radius. On the primitive rocksalt and diamond cells it puts
`pob-TZVP-rev2` at 25-29 bohr and def2-SVP at 33-52 bohr, which is a
fair measure of what a molecular basis costs on a dense lattice.

```{note}
The multi-k **GPW** drivers (RKS and UKS) use this criterion for every
real-space lattice-summed operator of the generalized eigenproblem —
overlap, kinetic, and the Ewald V_ne share one R-set. The uniformity
matters as much as the reach: completing only the overlap sum against a
15-bohr V_ne over-bound LiF/def2-SVP by 18 Ha (see the note in
[`gapw.md`](gapw.md)). An S(k) left indefinite by an insufficient sum
now fails closed with a diagnostic.
```
`vq.format_truncation_optimization_report(...)` prints the
trajectory.

### 3. Filter primitives at run-time

For a basis that *is* the problem (the disambiguator says
`basis_set_problem`, or you've decided you want this specific
basis even if it needs help), drop the most-diffuse primitives:

```python
basis = vq.make_basis(mol, "def2-tzvp", exp_to_discard=0.1)
```

This mirrors PySCF's `Cell.exp_to_discard`. Every Gaussian
primitive with exponent $\alpha < 0.1$ is removed; the
synthesised basis is cached as a `.g94` file for re-load
speed.

vibe-qc prints **every dropped primitive verbatim** in the
SCF output:

```text
basis filter — dropped primitives (exp_to_discard = 0.10):
  Element  shell  l   exponent (α)  libint coeff
  C        2sp    s    0.087        +0.023
  C        2sp    p    0.087        +0.171
  ...
```

Per the [CRYSTAL solid-state convention](https://www.crystal.unito.it/Manuals/crystal23.pdf):
**this modifies the basis content and must be cited** in any
publication using the results. The drop log ends with a citation
reminder (`vq.format_basis_filter_report`).

### Last resort: explicit opt-in orthogonalisation

If you've decided the SCF must run despite a `critical` preflight
- for example, you're reproducing an old calculation that was
done with silent truncation, three explicit orthogonalisation
methods are available:

```python
X, info = vq.orthogonalise_overlap(S, method="canonical")     # Löwdin 1970
X, info = vq.orthogonalise_overlap(S, method="pivoted_cholesky")  # safer for very ill-conditioned
X, info = vq.orthogonalise_overlap(S, method="symmetric")     # standard, fails on negative λ
```

The SCF driver opts in via `allow_critical=True` plus
`orth_method="canonical"` (or another method). **This is the last
resort** because silent basis truncation can give a converged-but-
physically-wrong total energy: the orthogonaliser drops the
near-singular subspace, the SCF converges in the smaller space,
and the answer is a projection of the true wavefunction onto
that subspace, not the true wavefunction. Without explicit
opt-in, vibe-qc refuses to do this on your behalf.

## Defaults users get without lifting a finger

The right behaviour is the default. Specifically, in v0.7+ all
8 periodic SCF drivers (RHF / UHF / RKS / UKS × Γ-only / multi-k):

* Run `scf_preflight_overlap_check` automatically at startup;
  abort with `LinearDependenceError` on `critical` severity.
* Run `optimize_truncation` automatically (`auto_optimize_truncation=True`)
  to find the loosest cutoffs that keep $S$ PSD; this typically
  costs 1-3 extra preflight evaluations and saves the user from
  having to tune cutoffs by hand.
* Print the active settings (`dump_active_settings`) to the SCF
  log so that an output is fully self-documenting and reproducible.

You only need to read this chapter when (a) a default fires and
you want to understand *why*, or (b) you want to override
defaults intentionally.

## Settings transparency

Every overridable parameter has a meaningful default AND can be
set explicitly AND is printed at SCF startup. The output of any
periodic SCF run includes a settings dump like:

```text
=== active settings (dump_active_settings) ===
  lattice_opts.cutoff_bohr         = 14.0  (auto-optimised from 10.0)
  lattice_opts.nuclear_cutoff_bohr = 18.0
  lattice_opts.schwarz_threshold   = 1e-13 (auto-optimised from 1e-12)
  preflight.severity               = warn
  preflight.lambda_min             = 8.4e-08
  preflight.kappa                  = 1.2e+09
  basis                            = sto-3g (no exp_to_discard filter)
  ...
```

A reproducer of any historical calculation is one settings-block
copy away, set the same explicit values, get the same numbers.

## Recommended workflows by system class

A short table for picking the right combination by system type:

| System | Basis | Cutoffs | Auto-truncation | Notes |
|---|---|---|---|---|
| **Closed-shell molecule (no PBC)** | def2-TZVP / cc-pVTZ | n/a | n/a | Linear dependence essentially never bites |
| **Open-shell molecule** | def2-TZVP / cc-pVTZ | n/a | n/a | UHF/UKS, no different from closed-shell |
| **Tight insulator** (LiH, MgO, NaCl) | **pob-TZVP-rev2** | default + auto-opt | **ON (default)** | Designed for exactly this; expect `ok` preflight |
| **Ionic crystal at experimental geometry** | pob-TZVP-rev2 | default + auto-opt | **ON (default)** | If preflight warns, try pob-TZVP (smaller); else `exp_to_discard=0.1` |
| **Molecular crystal** | pob-DZVP-rev2 + dispersion | default + auto-opt | **ON (default)** | Loose packing → easier on the overlap |
| **Metal (1D / 2D)** | STO-3G or pob-DZVP-rev2 | tighter (`cutoff_bohr=20`) | **ON (default)** | SCF convergence is the harder problem; see [`scf_convergence`](scf_convergence.md) |
| **Slab** (2D periodic) | pob-TZVP-rev2 | default + auto-opt | **ON (default)** | Vacuum padding ≥ 30 bohr in the non-periodic direction |
| **Reproducing an old molecular-basis calc** | def2-TZVP / cc-pVTZ + `exp_to_discard=0.1` | default + auto-opt | **ON (default)** | Print the drop log and cite it |

## References

* Lykos, P. G., & Schmeising, H. N. (1961). *Linear dependence in
  Gaussian basis sets and the catastrophe of orthonormalization.*
  J. Chem. Phys., **35**, 288.
* Löwdin, P.-O. (1970). *On the nonorthogonality problem.* Adv.
  Quantum Chem., **5**, 185, the canonical-orthogonalisation paper
  that gives the v0.7 codename *Löwdin's Compass*.
* Peintinger, M. F., Vilela Oliveira, D., & Bredow, T. (2013).
  Consistent Gaussian basis sets of triple-zeta valence with
  polarization quality for solid-state calculations. *J. Comput.
  Chem.*, **34**, 451.
  [doi:10.1002/jcc.23153](https://doi.org/10.1002/jcc.23153),
  pob-TZVP design paper (project author).
* Vilela Oliveira, D., Laun, J., Peintinger, M. F., & Bredow, T.
  (2019). BSSE-correction scheme for consistent Gaussian basis
  sets of double- and triple-zeta valence with polarization
  quality for solid-state calculations. *J. Comput. Chem.*,
  **40**, 2364.
  [doi:10.1002/jcc.26013](https://doi.org/10.1002/jcc.26013),
  pob-DZVP-rev2 / pob-TZVP-rev2 design paper.
* Searle, B., Bernasconi, L., & Harrison, N. (2017). *Linear
  dependence in periodic systems.* ARCHER eCSE04-16., k-point-
  dependent failure modes; the H atom basis at the BZ boundary
  example.
* Dovesi, R., Erba, A., Orlando, R., Zicovich-Wilson, C. M.,
  Civalleri, B., Maschio, L., Rérat, M., Casassa, S., Baima, J.,
  Salustro, S., & Kirtman, B. (2018). *Quantum-mechanical
  condensed matter simulations with CRYSTAL.* WIREs Comput. Mol.
  Sci., **8**, e1360., CRYSTAL TOLINTEG / EIGS / OPTBASIS
  reference manual.

## Where to next

* [Why solid-state calculations use `pob-TZVP`](../tutorial/pob_tzvp.md):
  worked example of the recommended basis for solids.
* [Solid-state walkthrough: LiH rocksalt with `pob-TZVP`](../tutorial/lih_pob_tzvp_solid_state.md):
  end-to-end periodic SCF + bands + DOS on a real ionic crystal.
* [User guide: basis sets](basis_sets.md) and
  [user guide: SCF convergence](scf_convergence.md): the broader
  context this chapter sits in.
* [Roadmap § linear-dependence + screening
  program](../roadmap.md#cross-code-feature-gaps-orca-611--crystal23-manual-sweep):
  what's planned beyond v0.7. The v0.8 OPTBASIS-style internal
  basis-set optimiser (CRYSTAL's `OPTBASIS`-with-condition-number-penalty
  equivalent) is the natural next step.
* The automated parity test suite at
  [`examples/regression/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression)
  benchmarks vibe-qc's periodic SCF against PySCF.pbc on a
  growing set of reference systems; consult it when you want to
  confirm an unfamiliar system is well-tested.
