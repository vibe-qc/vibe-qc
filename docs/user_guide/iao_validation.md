# IAO numerical comparisons and reproducible examples

This page separates three kinds of evidence: published atomic charges,
analytical bond-index normalization, and independent numerical comparisons.
The definitions and supported inputs are in the [IAO analysis guide](iao_population.md).
All computed values below use the molecular determinant implementation
introduced in commit `366b20d`, including all occupied core electrons.

## Published charges: match the reference basis

[Knizia (2013), Table 1](https://doi.org/10.1021/ct400687b) reports several
orbital-basis choices and two different minimal references. Its main rows
(footnote a) use MINAO. The **cc-pVTZ row with footnote c uses Huzinaga MINI**,
matching vibe-qc's reference choice. Compare against that row first:

| Molecule / atom | Paper MINI/cc-pVTZ (e) | vibe-qc MINI/cc-pVTZ (e) | Computed minus printed (e) |
|---|---:|---:|---:|
| CH4 / C | -0.49 | -0.500597 | -0.010597 |
| CH4 / each H | +0.12 | +0.125149 | +0.005149 |
| HCN / H | +0.22 | +0.217465 | -0.002535 |
| HCN / C | -0.03 | -0.026190 | +0.003810 |
| HCN / N | -0.19 | -0.191274 | -0.001274 |

The largest absolute difference is **0.01060 e for CH4** and **0.00381 e for
HCN**. HCN reproduces the printed two-decimal charges. Methane does not
reproduce every printed digit. The paper gives only two decimal places; its
rounded methane entries themselves sum to -0.01 e. Computed charges are
never adjusted to reproduce such a rounded sum.

This is a comparison with published numbers, not an exact same-input
reproduction. The paper's Appendix E describes optimized geometries and
says that coordinates are available on request. The supplied article does
not include those coordinates. This calculation uses the fixed geometries
listed below. The reference family and orbital basis match footnote c,
but geometry and the precise reference contractions used in the original
calculation have not been independently matched to the author's input.
Do not assign the entire residual difference to geometry without testing it.

The commonly quoted MINAO rows are useful context, with an additional
reference-basis difference:

| Orbital basis | CH4 C, paper MINAO (e) | CH4 C, vibe-qc MINI (e) | HCN N, paper MINAO (e) | HCN N, vibe-qc MINI (e) |
|---|---:|---:|---:|---:|
| def2-SVP | -0.49 | -0.472559 | -0.20 | -0.182528 |
| def2-TZVPP | -0.52 | -0.499520 | -0.21 | -0.192610 |

For the two larger orbital bases, vibe-qc's methane carbon charge changes
by 0.00108 e between def2-TZVPP and cc-pVTZ. This illustrates the observed
basis stability for these inputs; it is not a universal error estimate.
The existing `test_iao_ibo.py` literature regressions use a 0.03 e window
against MINAO rows to accommodate reference/geometry differences. Passing
that window alone is weaker evidence than a comparison with matched input.

## Calculation protocol

`examples/iao_population_comparison.py` is the complete reproducible input.
It runs nine calculations through the public `run_job` API:

| System | Fixed geometry | Charge / multiplicity | Method / orbital bases |
|---|---|---|---|
| CH4 | Tetrahedral, C-H = 1.087 angstrom | 0 / 1 | RHF; def2-SVP, def2-TZVPP, cc-pVTZ |
| HCN | Linear H-C-N, H-C = 1.0655 angstrom, C-N = 1.153 angstrom | 0 / 1 | RHF; def2-SVP, def2-TZVPP, cc-pVTZ |
| H2 | H-H = 1.4 bohr | 0 / 1 | RHF/def2-SVP |
| H2+ | H-H = 2.0 bohr | +1 / 2 | UHF/def2-SVP |
| OH | O-H = 1.83 bohr | 0 / 2 | UHF/def2-SVP |

The script uses `1 angstrom = 1.8897261254578281 bohr`, conventional
four-index SCF (`density_fit=False`), an energy convergence threshold of
`1e-10`, a gradient threshold of `1e-8`, and at most 100 iterations.
Analysis uses Huzinaga MINI, symmetric orthogonalization, and separate
occupied spin spaces for UHF. Localization is disabled for every example.
The MINI data are the bundled Gaussian-format coefficients identified as
Basis Set Exchange MINI version 1, with values from GAMESS.

Run from the checkout with generated files outside it:

```sh
IAO_RESULTS=$(mktemp -d)
cp examples/iao_population_comparison.py "$IAO_RESULTS/input.py"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python "$IAO_RESULTS/input.py" --output-dir "$IAO_RESULTS" \
  > "$IAO_RESULTS/comparison.log" 2>&1
```

The log prints every atom's charge, the published target where applicable,
the difference, selected bond orders, spins and conservation residuals.
Each job also produces its ordinary `.out`, `.system`, population text/JSON,
structured log and citation files. Add `--qvf` to exercise the QVF charge
member and full vendor payload. The default run requests neither QVF nor
localization. Keep the copied input with these outputs.

## Bond orders and spin populations

For the chosen convention, an equal-weight doubly occupied bonding orbital
has $B=1$; a single alpha electron has $B=0.5$; doubly occupied bonding and
antibonding orbitals together give $B=0$. These are **analytical determinant
checks**, obtained from the Wiberg square and number-covariance identity,
not transcribed molecular entries from Knizia's charge table. The public
H2 and H2+ calculations above give 1.00000000 and 0.50000000, respectively.

The following are computed example values, not published targets:

| Calculation | Selected IAO-Wiberg bond | Computed value |
|---|---|---:|
| CH4 RHF/cc-pVTZ | C-H | 0.98187870 |
| HCN RHF/cc-pVTZ | H-C | 0.93368414 |
| HCN RHF/cc-pVTZ | C-N | 2.98070067 |
| OH UHF/def2-SVP | O-H | 0.87893501 |

For OH the charges are `[-0.34508291, +0.34508291] e` and the spin
populations are `[+1.04452837, -0.04452837]` alpha-minus-beta electrons,
in O,H order. The negative hydrogen spin population reflects polarization;
the spin sum remains one. Across the nine examples, total-charge residuals
were below `7e-14 e`, spin-sum residuals below `1e-14`, and occupied-space
reconstruction residuals below `6e-15` in the AO metric.

Wiberg's [1968 paper](https://doi.org/10.1016/0040-4020(68)88057-3) defines
the original bond-index construction, not these MINI/SCF examples. Likewise,
the [Molpro IBBA manual](https://www.molpro.net/manual/doku.php?id=intrinsic_basis_bonding_analysis_iao_ibo)
distinguishes IAO-Wiberg from an unpublished renormalized index. Its reference
and orthogonalization options must be matched before comparing numerical
results. No published same-input open-shell spin-population or bond-order
table is claimed here. Such a benchmark remains a separate validation task.

## Independent numerical checks

The optional `--pyscf` flag reruns each SCF, overlap integral and IAO
construction in PySCF. It reads the same orbital and MINI Gaussian basis
files, uses the same geometry, charge, multiplicity and spherical basis
convention, then symmetrically orthonormalizes PySCF's raw IAOs. Alpha/beta
occupied spaces are treated separately. The output reports maximum absolute
charge, spin and bond-matrix differences plus the total-energy difference;
full PySCF logs are preserved beside the native outputs.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python "$IAO_RESULTS/input.py" --output-dir "$IAO_RESULTS/oracle" \
  --pyscf --qvf > "$IAO_RESULTS/oracle.log" 2>&1
```

PySCF is optional and is used only by the example/test oracle. It is never
called by the runtime implementation. Its SCF thresholds are `1e-12` for
energy and `1e-9` for the orbital gradient. Matching the basis data avoids
confounding PySCF's default MINAO reference with MINI. Agreement tests code
and numerical consistency; it does not remove the unmatched inputs in the
published comparison above.

The focused regression commands are:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_iao_population.py tests/test_iao_ibo.py -q
```

They also cover complex coefficients, occupied rotations, atom permutation,
rotations within each atom's IAO block, an independently enumerated Slater
determinant number covariance, empty spin, diffuse and rank-deficient spaces,
unsupported input, and output/citation compatibility. The matched-overlap
PySCF tests isolate the IAO algebra; the example's `--pyscf` comparison also
checks independently computed SCF orbitals and integrals.

For the nine runs above, the largest absolute differences against the full
independent PySCF 2.14.0 calculation were:

| Quantity | Maximum difference | Case |
|---|---:|---|
| Total energy | 3.411e-13 hartree | CH4/def2-TZVPP |
| Atomic charge | 6.809e-10 e | HCN/def2-TZVPP |
| Atomic spin population | 9.190e-10 electrons | OH/def2-SVP |
| IAO-Wiberg matrix entry | 9.261e-11 | OH/def2-SVP |

At the matched MINI/cc-pVTZ input specifically, the largest charge differences
were `1.176e-11 e` for CH4 and `3.751e-10 e` for HCN. Both programs therefore
give essentially the same charges at these inputs, including the difference
from the printed methane row. The cause of that literature difference remains
unresolved until the original geometry and reference contractions are matched.
For all nine QVF-enabled examples, the complete `analysis/iao.json` payload
was identical to the population sidecar's `iao` object.
