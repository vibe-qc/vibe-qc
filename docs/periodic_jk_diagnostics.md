# Periodic J/K contraction records

`_vibeqc_core.trace_jk_2e_real_space(basis, system, options, density,
max_records, omega=0.0)` observes the ordinary shell-pair traversal. Use it
for bounded diagnosis of quartet membership, integral values and accumulation
rounding. It uses the same supplied density, lattice support, screening and
sorted contraction as `build_jk_2e_real_space`; it does not change their
thresholds or compute a density derivative.

The result contains `matrices` (the ordinary J/K result) and two array
properties. Each property returns a copy. Native record storage is bounded
by the explicit positive record cap and an absolute 128 MiB limit; account
for the returned arrays and ordinary builder storage separately. Unsupported
traversals, a shell-pair fallback or record-cap exhaustion raise an exception.
A partial trace is never returned. Ordinary builds allocate no trace buffer.

`identities` has uint64 columns:

| Columns | Meaning |
|---|---|
| 0-3 | Output cell, lambda cell, sigma cell, density cell |
| 4-7 | Output shells s1/s2 and contracting shells s3/s4 |
| 8-11 | Output AOs mu/nu and contracting AOs rho/tau |
| 12 | Channel: 0 for J, 1 for unscaled K |
| 13 | Actual sequential record position within the output shell/channel |

The first three cell positions index `matrices.J.cells`; the density cell
indexes the input density. Every record belongs to an admitted, evaluated
quartet. A null integral return uses `UINT64_MAX` in all four AO columns and
zero values, so it remains distinguishable from an absent quartet. Several
AO records can belong to one quartet. Distinct quartet keys (columns 0-7
and 12) should equal `matrices.integral_evaluations`.

`values` has float64 columns `[integral, density, rounded_product,
sum_before, sum_after]`. The product is separately rounded; the actual
published accumulator values include the Neumaier rounding correction.
The separately rounded product is added without fused multiply-add or reordered
quartets. Before/after values remain a chain that replays the returned blocks;
ordinary recursive addition of products no longer reproduces that chain. Records are sorted by output cell, output shell pair,
channel and sequence, independently of worker completion order.

Check trace fidelity before drawing conclusions: compare the ordinary and
observed blocks, replay each recorded accumulator chain exactly, and verify
quartet coverage. Compare full quartet identities across density perturbations;
equal counters do not establish equal membership. Exact arithmetic over the
recorded operands can then distinguish membership, integral, product and
accumulation effects. These diagnostics do not themselves establish SCF
convergence, a derivative criterion or external reference agreement.

## Compensated shell-pair contraction

Each output shell-pair task retains a separate two-double accumulator for each
AO pair in its J or K channel. A companion density has its own accumulators.
Scratch is 16 bytes per output AO pair per active task, or 32 bytes with a
companion, independent of the number of lattice images and admitted quartets.
Each product takes constant extra arithmetic; no term list or sorting is added.
The correction survives scalar and general-shell updates in the same task.
Completed matrix elements are published once per task, rather than after each
product. Trace records read the corrected accumulators directly before and
after each contribution, retaining the same intermediate values and final
matrices. Bounded compensated fallback builds likewise publish once per output.
Fallback discards partial outputs and its local accumulators together.

Ordinary cell-list and exhaustive contractions use the same accumulator,
including restarts after an incomplete shell-pair traversal. Their scratch is
bounded to 64 MiB per active output across all requested J/K channels and
companion densities. If the AO block exceeds that budget, those fallback paths
retain ordinary summation. Density-derivative and Gamma-adjoint paths also keep
their existing contraction. Integral evaluation, screening, physical support,
cell order, channel scaling and transpose/adjoint conventions are retained.
This reduces accumulation roundoff; it
does not remove screening error or guarantee correctly rounded results for all
signed inputs. See Higham, SIAM J. Sci. Comput. 14, 783-799 (1993), Sec. 3, pp. 790-791,
[doi:10.1137/0914050](https://doi.org/10.1137/0914050).
