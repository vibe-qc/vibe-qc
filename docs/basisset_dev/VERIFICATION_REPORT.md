# pob-* basis-set verification report

**Date:** 2026-05-08 (work conducted on `basissetdev`)
**Scope:** verify the three pob-* basis files bundled in vibe-qc
against the upstream Bredow-group sources and the supporting
information of the cited papers.
**Tool:** `scripts/basisset_dev/pob_basis_verify.py`

## Authoritative sources

* **pob-TZVP**, `python/vibeqc/basis_library/sources/pob-TZVP/`, 
  32 per-element CRYSTAL-format files distributed by the Bredow
  group as `pob-tzvp-tar.gz`
  (`https://www.chemie.uni-bonn.de/bredow/de/software/pob-tzvp-tar.gz`,
  retrieved 2026-05-07). Spot-checked element-by-element against
  Section 2 of the supporting information of M. F. Peintinger,
  D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **34**, 451
  (2013) [DOI 10.1002/jcc.23153] for H, Li, Be, B, C, N, O, F, Na,
  Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, all exponents and
  coefficients agree to printed precision.
* **pob-TZVP-rev2**, `sources/pob-TZVP-rev2/`, 32 files from
  `pob-tzvp-rev2-tar.gz`, retrieved 2026-05-07.
* **pob-DZVP-rev2**, `sources/pob-DZVP-rev2/`, 19 per-element files
  scraped from the per-element HTML pages at
  `https://www.chemie.uni-bonn.de/bredow/de/software/pob-dzvp-rev2/<NN_x>`
  (the Bredow group serves dzvp-rev2 as a Plone web folder rather
  than a tarball). Retrieved 2026-05-07.

## Findings

### Bug: sulfur d-polarisation column-swap in shipped pob-TZVP and pob-TZVP-rev2

Both shipped `pob-tzvp.g94` and `pob-tzvp-rev2.g94` had the d-shell
of sulfur in the wrong columns:

```
D    1   1.00
        0.0000000000        0.5207010100   ← bug: exponent zero
```

vs. authoritative source:

```
D    1   1.00
        0.5207010100        1.0000000000   ← correct (pob-TZVP)
        0.4107010100        1.0000000000   ← correct (pob-TZVP-rev2)
```

**Root cause.** The Bredow source files for sulfur in both bases
contain a typo, the d-shell scale `1.0` is mistyped as two
whitespace-separated tokens `1 0`:

```
0 3 1 0.0 1 0
  0.5207010100     1.00000000000000
```

CRYSTAL itself parses this correctly (it's whitespace-tolerant
across newlines). The previous CRYSTAL-to-`.g94` conversion that
seeded vibe-qc's bundled file was *not* tolerant: it took the `0`
on the header line as the start of the next primitive's exponent,
shifting all subsequent fields by one slot. Result: exponent column
ended up holding `0` and the coefficient column held the d-exponent
`0.5207010100`. The actual `1.0000000000000` coefficient was orphaned
on the next line and silently truncated.

**Chemistry impact.** A d-polarisation function with exponent zero
contributes nothing to the orbital expansion. Sulfur calculations
using the previously bundled pob-TZVP or pob-TZVP-rev2 silently ran
with **no d-polarisation on S**, equivalent to running pob-TZ rather
than pob-TZVP for the sulfur centres. Quantitative impact on
sulfur-containing systems (S, B₄C with no S, no, but ZnS, MnS, K₂S,
CoS, CuS, β-ZnS in the paper test set) is yet to be measured, 
expect lattice-constant errors O(0.01-0.05 Å) and total-energy errors
several kJ/mol per S centre, since the d-function is structurally
important on second-row p-block elements.

**Fix.** `pob_basis_verify.py` parser is line-aware and detects the
"`1 0`" typo, reconstructing the scale as `1.0`. `--regenerate`
re-emits the affected `.g94` files. Both `pob-tzvp.g94` and
`pob-tzvp-rev2.g94` now carry the correct d-polarisation on sulfur.

### Precision of all numbers

Beyond the sulfur fix, the main change is preserving full source
precision. The previous shipped files rounded contraction coefficients
to 10 decimal places; the Bredow CRYSTAL files publish 11 dp for
pob-TZVP and pob-TZVP-rev2 and 8 dp for pob-DZVP-rev2. The regenerated
files preserve every digit the source publishes.

### Element coverage

| Basis | Source files | Shipped `.g94` blocks | Match |
|-------|--------------|-----------------------|-------|
| pob-TZVP | 32 (H-Br excluding noble gases) | 32 | ✅ |
| pob-TZVP-rev2 | 32 (same range) | 32 | ✅ |
| pob-DZVP-rev2 | 19 (H-V excluding Si and noble gases) | 19 | ✅ |

## Verification command

```bash
$ python3 scripts/basisset_dev/pob_basis_verify.py --atol 1e-15
=== pob-TZVP ===
  source elements:  32
  in shipped g94:   32
  parity:           CLEAN within tolerance

=== pob-TZVP-rev2 ===
  source elements:  32
  in shipped g94:   32
  parity:           CLEAN within tolerance

=== pob-DZVP-rev2 ===
  source elements:  19
  in shipped g94:   19
  parity:           CLEAN within tolerance
```

All three basis sets are byte-clean against their authoritative
upstream sources to 1e-15 absolute tolerance, i.e. exact agreement
to all printed digits.

## Follow-ups

* Tell Bredow group about the `16_S` "`1 0`" typo so it can be fixed
  upstream (CRYSTAL parses it but other consumers may not).
* Add a regression test that runs `pob_basis_verify.py --atol 1e-15`
  in CI.
* Once sulfur-relevant features land in vibe-qc periodic
  (multi-k + open-shell + hexagonal), regenerate the pob-paper
  benchmark numbers for sulfides (ZnS, MnS, K₂S, CoS, CuS, β-ZnS)
  with the corrected basis and compare to SI Tables 1 / 8 / 9. The
  delta between previously shipped (broken d on S) and corrected
  basis is the chemistry impact of this bug.

## Driver comparison on the Stage-1 single-atom problem

**Date:** 2026-09-29, vibe-qc 0.17.8 development tree.
**Scope:** every optimiser driver in `vibeqc.basis_optimization` on the same
starting point and bounds (roadmap Phase 3), as a side-by-side of evaluation
count, wall-clock, final objective and final exponent, and as regression
protection against a driver silently degrading.
**Tool:** `scripts/basisset_dev/driver_benchmark_stage1.py` (the tables
below are its output, `--penalty` for the second one).

### Problem

The Stage-1 smoke of `recipes/single_atom.py`: pob-TZVP H, the diffuse-most
s exponent (shell 2, primitive 0) free in log space with the physical box
(0.05, 1.0), started at 1.5 times the published value, 0.269267 instead of
0.179511. The objective is the UHF energy of the isolated H atom (doublet),
built with `make_uhf_native_objective_gradient`, so the BDIIS rows can use the
analytic exponent gradient. The second table adds the OPTBASIS conditioning
term `gamma ln kappa(S)` with `gamma = 1e-3` Ha, the objective shape of
`docs/basisset_dev/ENERGY_GRADIENT_DESIGN.md`.

Columns: objective evaluations are SCF energies; gradient evaluations are
analytic-gradient calls, each one SCF plus the exponent-derivative integrals.
Wall-clock is one laptop run with the optional backends already imported; an
H-atom SCF takes about ten milliseconds, so the differences are the drivers'
own overheads. Tolerances: scipy `tol=1e-8`, BDIIS `tol_energy=1e-8` and
`tol_grad=1e-5`, Py-BOBYQA `rhoend=1e-6`, CMA-ES `tol_fun=1e-9` with
`seed=1`, the chained recipe a 60-evaluation CMA-ES budget with
`cmaes_tol_fun=1e-6` before the same BDIIS settings.

### Energy only

Start objective -0.489254681 Ha.

| driver | objective evals | gradient evals | final objective (Ha) | final exponent | converged | wall (s) |
|---|---:|---:|---:|---:|:---:|---:|
| scipy L-BFGS-B (FD) | 16 | 0 | -0.499400431 | 0.130788 | yes | 0.17 |
| minuit MIGRAD (FD) | 16 | 0 | -0.499373108 | 0.126225 | yes | 0.16 |
| bdiis (FD gradient) | 24 | 0 | -0.499400431 | 0.130785 | yes | 0.16 |
| bdiis (analytic gradient) | 11 | 9 | -0.499400431 | 0.130791 | yes | 0.20 |
| bobyqa (Py-BOBYQA) | 18 | 0 | -0.499400431 | 0.130789 | yes | 0.15 |
| cmaes (pycma) | 224 | 0 | -0.499400431 | 0.130789 | yes | 2.04 |
| cmaes + bdiis (analytic) | 68 | 4 | -0.499400431 | 0.130789 | yes | 0.51 |

### Energy plus `gamma ln kappa(S)`, `gamma = 1e-3` Ha

Start objective -0.484096518 Ha.

| driver | objective evals | gradient evals | final objective (Ha) | final exponent | converged | wall (s) |
|---|---:|---:|---:|---:|:---:|---:|
| scipy L-BFGS-B (FD) | 100 | 0 | -0.496125687 | 0.126684 | no | 0.99 |
| minuit MIGRAD (FD) | 14 | 0 | -0.496113779 | 0.123810 | yes | 0.18 |
| bdiis (FD gradient) | 21 | 0 | -0.496125687 | 0.126693 | yes | 0.30 |
| bdiis (analytic gradient) | 8 | 8 | -0.496125687 | 0.126681 | yes | 0.21 |
| bobyqa (Py-BOBYQA) | 22 | 0 | -0.496125687 | 0.126689 | yes | 0.26 |
| cmaes (pycma) | 212 | 0 | -0.496125687 | 0.126689 | yes | 2.27 |
| cmaes + bdiis (analytic) | 68 | 4 | -0.496125687 | 0.126689 | yes | 0.64 |

### Reading the tables

* **Same minimum.** Six of the seven rows agree on the final objective to
  1e-9 Ha and on the exponent to 1e-5, on both objectives, so no driver
  silently uses a wrong sign or scale. The atomic optimum, 0.1308, sits
  below the published 0.1795: pob-TZVP's diffuse H exponent was fit to
  solids, not to the free atom, and this is the smoke test's point, the
  machinery, not the number.
* **The exception is MIGRAD.** It stops 2.7e-5 Ha (energy) and 1.2e-5 Ha
  (penalty) short, at an exponent about 3 percent away. MIGRAD's stopping
  rule is an estimated distance to minimum scaled by `errordef`, and
  `errordef = 1` is a chi-square convention that means nothing for an
  objective in Hartree. Use `optimize_minuit` for its HESSE uncertainties at a
  converged point, as the production recipe does, not for the convergence
  itself.
* **scipy L-BFGS-B fails on the penalty objective.** It exhausts 100
  evaluations with `ABNORMAL` (a failed line search). Its forward-difference
  gradient with the default 1e-8 step is too noisy for the stiff `ln kappa`
  term; the central-difference BDIIS with a 1e-5 step and every
  derivative-free driver are unaffected. This is the "energy-gradient
  mismatch" risk from the roadmap made concrete: the penalty term is the
  part that finite differences handle worst, which is why its gradient is
  analytic.
* **Evaluation counts.** With one parameter the drivers are close: the
  analytic-gradient BDIIS needs 20 SCFs (11 energies plus 9 gradients),
  Py-BOBYQA 18 to 22 with no gradient at all, scipy 16, finite-difference
  BDIIS 21 to 24. The gap that matters opens with the parameter count: a
  central-difference gradient costs `2 n_params` SCFs per step, a BOBYQA
  step about one, an analytic BDIIS gradient one SCF plus derivative
  integrals. The 3-D mock in `test_basis_opt_dfo_drivers.py` pins
  `optimize_bobyqa` below finite-difference BDIIS in evaluation count.
* **CMA-ES alone is ten times dearer** (212 to 224 evaluations) and reaches
  the same point; that is the price of a global method on a unimodal
  problem. The chained recipe caps it at 60 evaluations and lets BDIIS
  finish in 4 gradient calls, 72 SCF-equivalents in total, which is the
  intended use: a pre-pass that only has to land in the right basin.
* **Wall-clock** tracks evaluation count here because every SCF is cheap;
  the derivative-free drivers add no measurable overhead of their own once
  their packages are imported (the first import of Py-BOBYQA or DFO-LS
  costs about a second, which the script excludes).

### What this does not show

One parameter cannot exercise the anisotropic step scaling of CMA-ES
(`CMA_stds` from the per-parameter box widths), the DIIS subspace of BDIIS,
or the multi-parameter evaluation-count advantage of the model-based
drivers; those are covered by the 3-D mock tests. The Stage-2 (LiH periodic
refit) and Stage-3 (multi-compound) comparisons wait on the periodic
blockers R1, R2 and R4 in `REQUIREMENTS-PERIODIC.md` and on the Goal 3 test
set.
