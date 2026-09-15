# SCF Fock-build modes

vibe-qc offers two **Fock-build modes** for molecular SCF, the
strategy that constructs the J (Coulomb) and K (exchange) matrices each
iteration. Density fitting is a separate route: when `density_fit=True`, its
RI-J or RIJCOSX kernels supersede the four-index path and the mode is ignored.
Within that route, `cosx=True` selects COSX-K. The high-level
`run_job(cosx=True)` convenience also enables density fitting automatically.

For a worked resource comparison, including the 170-basis-function
H2O2/cc-pVQZ timing and memory measurements, see
[Direct SCF or in-core integrals?](../tutorial/direct_scf_memory_tradeoff.md).

## Three modes

| Mode | Enum value | Memory | Best for |
|---|---|---|---|
| `AUTO` | `SCFMode.AUTO` | - | Default, picks based on system size |
| `CONVENTIONAL` | `SCFMode.CONVENTIONAL` | O(n_bf⁴) | Small systems whose dense tensor fits comfortably |
| `DIRECT` | `SCFMode.DIRECT` | O(n_shells² + n_bf²) | Large systems, bounded integral memory |

All molecular SCF drivers (RHF, UHF, RKS, UKS) support the mode via
identical `scf_mode` / `scf_mode_auto_threshold` fields on their
options struct. Range-separated RKS and UKS are a deliberate exception to
the requested mode: their attenuated-exchange kernel requires the direct
builder, as described below.

## AUTO, the default

```python
from vibeqc import RHFOptions, SCFMode

opts = RHFOptions()
# opts.scf_mode defaults to SCFMode.AUTO
# opts.scf_mode_auto_threshold defaults to 140 BF
```

When `scf_mode = AUTO` (the default), vibe-qc counts the number of
basis functions after the basis set is constructed and decides:

* **\(n_{bf} \le 140\)** -> `CONVENTIONAL`, the in-core 4-index ERI
  tensor is built once and the per-iteration cost is a tensor contraction.

* **\(n_{bf} > 140\)** -> `DIRECT`, the in-core tensor would be
  multi-GB; switch to on-the-fly Schwarz-screened libint quartet
  evaluation that stays O(n_shells² + n_bf²) in memory.

The threshold is a resource policy, not a physical boundary. At the default
cutoff, a dense 140-function tensor is already 2.862 GiB before any other
process memory. To cap that tensor below 1 GiB, use 107, the largest basis
count satisfying that limit:

```python
from vibeqc import RHFOptions

opts = RHFOptions()
opts.scf_mode_auto_threshold = 107  # switch to DIRECT above about 1 GiB
```

## CONVENTIONAL, the in-core path

```python
from vibeqc import RHFOptions, SCFMode

opts = RHFOptions()
opts.scf_mode = SCFMode.CONVENTIONAL
```

The full 4-index electron-repulsion integral tensor (μν|λσ) is
computed once at the start and reused across SCF iterations. Fast
for small systems, but the tensor grows as n_bf⁴:

| n_bf | Full tensor size |
|---:|---:|
| 24 | 2.53 MiB |
| 95 | 621.42 MiB |
| 107 | 1000.06 MiB |
| 108 | 1.014 GiB |
| 170 | 6.223 GiB |
| 200 | 11.921 GiB |
| 250 | 29.104 GiB |
| 394 | 179.546 GiB |

These are $n_\mathrm{bf}^4\times8$ bytes for the dense tensor alone. Add
Fock, density, DIIS, eigensolver, runtime, and requested-output workspaces
when sizing a job. Lower the AUTO cutoff when the tensor exceeds the
workflow's budget, or set DIRECT explicitly.

## DIRECT, the integral-driven path

```python
from vibeqc import RHFOptions, SCFMode

opts = RHFOptions()
opts.scf_mode = SCFMode.DIRECT
```

Instead of storing the ERI tensor, the direct Fock build considers the unique
shell quartets on each SCF iteration and evaluates only those that survive
screening. An 8-fold-symmetric loop skips the vast majority of quartets using
the strict Cauchy-Schwarz bound:

\[
|\langle \mu\nu | \lambda\sigma \rangle| \;\le\;
Q(s_\mu, s_\nu) \cdot Q(s_\lambda, s_\sigma)
\]

where \(Q(s_a, s_b)\) is precomputed from the diagonal shell-pair
integrals at construction time and reused across iterations.

### Direct SCF options

All fields live on the per-method options struct (`RHFOptions`,
`UHFOptions`, `RKSOptions`, `UKSOptions`):

| Field | Default | Description |
|---|---|---|
| `schwarz_threshold` | 1e-10 | Per-quartet skip bound; lower = fewer skipped quartets |
| `schwarz_threshold_loose` | 1e-7 | Loose threshold for early SCF iterations (two-phase Schwarz screening) |
| `schwarz_threshold_tighten_at` | 1e-3 | Gradient-norm cutoff, switch from loose to tight below this |
| `incremental_fock` | `True` | Use a difference-density ΔP Fock build on the direct path |
| `incremental_fock_reset_freq` | 8 | Full-density rebuild every N iterations to limit recurrence error |

### Two-phase Schwarz tightening

By default, the direct Fock build starts with the looser
`schwarz_threshold_loose` (1e-7) so the per-shell density envelope
catches mid-SCF quartets aggressively. Once the SCF gradient norm
drops below `schwarz_threshold_tighten_at`, the builder switches to
the tight `schwarz_threshold` and discards the incremental cache. An energy
difference across that threshold and cache change compares two different
numerical Fock maps, so it is excluded from convergence testing.

Häser and Ahlrichs analyse iteration-dependent screening and show that the
error in a screened difference-Fock update can inherit errors from earlier
increments (Eqs. 30-35). That analysis motivates the coarse-to-tight
refinement; it does not prescribe vibe-qc's thresholds or transition policy.
Set
`schwarz_threshold_loose <= schwarz_threshold` to disable the
coarsening and use a uniform tight threshold throughout.

### Incremental ΔP Fock build

When `incremental_fock = True`, the builder caches the previous
density D_prev and two-electron Fock matrix G_2e_prev. Each iteration
computes ΔD = D − D_prev and returns:

\[
G_{2e}[D] = G_{2e}[D_\text{prev}] + G_{2e}[\Delta D]
\]

This recurrence is Eq. 20 of Almlöf, Fægri, and Korsell. Their discussion on
p. 389 also recommends a final energy-oriented iteration after a non-energy
monitor is satisfied. The exact certification policy below is vibe-qc's
implementation of that principle, not a prescription quoted from the paper.

The direct kernel applies the published density-weighted Schwarz test to
the difference density. For an incremental update with
\(s = \max |\Delta D| < 1\), vibe-qc first screens
\(\Delta D / s\), builds that normalized contribution, and multiplies the
result by \(s\). This is a vibe-qc relative-error policy: the omitted part
of an update scales to zero with its amplitude, instead of each small update
being allowed the same absolute omission and accumulating a persistent drift
floor. It is derived from the linearity of \(G_{2e}\); it is not a scaling
formula prescribed by Almlöf et al. or Häser and Ahlrichs.

Amplitude normalization deliberately prevents a uniform rescaling of
\(\Delta D\) from making the retained quartet set artificially sparser.
Incremental builds still benefit when the change density becomes spatially
localized, while accuracy no longer depends on how one density change is
split across SCF iterations.

A full rebuild happens every `incremental_fock_reset_freq` iterations
(default 8) to limit error inherited from screened increments as well as
ordinary arithmetic accumulation. The reset interval is a vibe-qc
implementation policy, not a reset rule attributed to Almlöf et al. When the
two-phase screen changes to its tight threshold, the driver also discards the
incremental cache. Set
`incremental_fock=False` for a controlled full-build comparison.

```python
from vibeqc import RHFOptions

opts = RHFOptions()
opts.incremental_fock = True
opts.incremental_fock_reset_freq = 8
```

### Final full-Fock certification

Loose-screened and incremental builds are acceleration phases. A molecular
direct SCF calculation enters a certified full-Fock phase when either:

* the orbital gradient first satisfies `conv_tol_grad`, making the row
  eligible for ordinary convergence; or

* the restart monitor detects a high-gradient stall while a coarse Fock map
  is still active.

The driver then forces the configured tight Schwarz threshold, disables
incremental updates, resets the Fock builder, and clears extrapolation and
mixing state. The first tight, nonincremental full-density row has an energy
difference relative to the preceding coarse-map row. That cross-map value is
reported but cannot certify convergence. The following row provides an energy
difference between two comparable tight, nonincremental full-density builds.
Only then can the calculation converge, and both the original
`conv_tol_energy` and `conv_tol_grad` must pass unchanged.

Consequently, certification requires two comparable same-map rows. It does
not relax either tolerance or infer an empirical energy-noise floor. A
standard diagnostic identifies a gradient-triggered or stall-triggered move
into this phase.

### When density_fit or cosx is active

At the options-struct level, `density_fit=True` selects the DF J/K
infrastructure and causes it to ignore `scf_mode`. Inside that route,
`cosx=True` replaces fitted exchange with COSX-K. A low-level options object
with `cosx=True` but `density_fit=False` does not activate COSX; set both and
provide `aux_basis`. The high-level `run_job(cosx=True)` argument performs
those two steps and auto-selects an auxiliary basis when possible.

### Range-separated hybrids force DIRECT

RKS and UKS range-separated hybrids need an erf-attenuated exchange build.
Only the direct builder currently implements that kernel, so these jobs force
`DIRECT` when `density_fit=False`, even if `AUTO` would choose conventional or
the options request `CONVENTIONAL`. Combining a range-separated hybrid with
`density_fit=True` currently raises an explicit unsupported-route error because
the required attenuated three-center integrals are not implemented.

## Performance

On compute-small (i9-10885H, 4 threads), RHF direct / def2-SVP, same-box ORCA
comparison:

This sweep uses def2-SVP throughout, so its H2O row has 24 basis functions.
The H2O/6-31G* examples use 18 basis functions instead.

| System | n_bf | vibe-qc (s) | ORCA (s) | Ratio |
|---|---|---|---|---|
| H₂O | 24 | 2.0 | 15.0 | 0.13 |
| glycine | 95 | 5.0 | 13.0 | 0.39 |
| naphthalene | 180 | 31.1 | 45.1 | 0.69 |
| n-decane | 250 | 226.3 | 67.2 | 3.37 |
| n-hexadecane | 394 | 639.2 | 185.3 | 3.45 |

This historical sweep placed the speed crossover against ORCA near 200 BF on
that host. The current AUTO policy switches at 140 BF because the CPU-time
crossover is not a safe memory cutoff: the dense tensor is already 2.862 GiB
at 140 BF. Energy agreement against ORCA was ≤ 2.4 µHa across all converged
cells.

That wall-time crossover does not define a safe memory crossover. A matched
one-core H2O2/cc-pVQZ case at 170 basis functions used 45.550 s wall time,
45.032 s active CPU time, and 6611.05 MiB peak RSS with vibe-qc's
conventional path. ORCA 6.1.1 used 52.669 s wall time, 51.232 s active CPU
time, and 126.25 MiB peak RSS in the same allocation. The energy difference
was $1.53\times10^{-10}$ Ha. The
[worked tutorial](../tutorial/direct_scf_memory_tradeoff.md) derives the
$170^4$ allocation, separates the different-host direct validation, and
defines a safe crossover sweep.

## References

* Almlöf, Fægri, Korsell, *J. Comput. Chem.* **3**, 385 (1982),
  [doi:10.1002/jcc.540030314](https://doi.org/10.1002/jcc.540030314).
  Difference-Fock recurrence (Eq. 20) and the final energy-oriented iteration
  recommendation (p. 389).
* Häser & Ahlrichs, *J. Comput. Chem.* **10**, 104 (1989),
  [doi:10.1002/jcc.540100111](https://doi.org/10.1002/jcc.540100111).
  Cauchy-Schwarz screening and inherited screened-increment errors
  (Eqs. 30-35).

These references are cited automatically in the `.bibtex` and `.references`
output files when the requested mode is `DIRECT` or when `AUTO` resolves to
`DIRECT`.
