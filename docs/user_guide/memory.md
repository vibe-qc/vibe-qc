# Memory budget

Every Gaussian-basis and semiempirical `run_job` route covered by the memory
estimator runs through a pre-flight check that (a) reports the estimated peak
memory in the text output and (b) aborts with an explanation if that estimate
exceeds the memory available to the process. Pretrained MLIP routes are a
single forward pass and currently skip this estimator. On Linux the probe
includes finite scheduler or container cgroup limits, even when the host
itself has substantially more free RAM. This is a guardrail against the most
common catastrophic failure mode of a QC code, calculations that silently
thrash to disk and freeze the host.

The estimate is a planning model, while peak resident set size (RSS) is a
measurement. For the practical tradeoff between retaining
$N_\mathrm{bf}^4$ integrals and recomputing them with direct SCF, see
[Direct SCF or in-core integrals?](../tutorial/direct_scf_memory_tradeoff.md).

## What you'll see in the `.out` file

```
Job: RKS / PBE  basis=cc-pvdz
Atoms (bohr)
------------------------------------------------------
   1  Z=  8       0.00000000      0.00000000      0.00000000
   2  Z=  1       0.00000000      1.43000000     -0.98000000
   3  Z=  1       0.00000000     -1.43000000     -0.98000000
charge=0  multiplicity=1  n_electrons=10

vibe-qc estimates this calculation will require ~0.17 GB of memory:
    ERI tensor                      2.5 MB
    Fock + density + 1e            0.04 MB
    DIIS history                   0.07 MB
    MO workspace                   0.02 MB
    Python runtime + NumPy overhead 100.0 MB
    DFT grid + chi                  13.4 MB
Available on this machine: 24.0 GB. Proceeding.

  iter     energy (Ha)            dE          ||[F,DS]||   DIIS
  ...
```

The headline figure already carries a **1.5x default safety headroom**
over the sum of the per-category numbers. Set
`VIBEQC_MEMORY_HEADROOM=2.0` (or another value >= 1.0) to tune the
factor for a site or scheduler wrapper.

Molecular density-fitting reports show the three-index formula directly,
including the parsed auxiliary-basis dimension. For example:

```
DF three-index tensors  28.86 GB (n_orb^2=1134^2 x n_aux=753 x n_blocks=2 x 8 bytes x 2x safety)
```

The tensor extent is `n_orb^2 * n_aux`, not an orbital-basis-only estimate.
At every basis size, preflight charges the two resident three-index blocks
and a separate 2x safety factor for overlapping raw and transformed
construction buffers: the native `DensityFitting` constructor holds four
`n_aux * n_orb^2` extents simultaneously while it builds the fitted
tensors, independent of `n_orb`. (An earlier revision applied the
construction factor only above 500 orbital functions; production
measurements on the BH9 wave showed the same ~2x construction peak on
small bases, so the size gate was removed.)

When `density_fit=True` is requested without an explicit `aux_basis`,
the preflight sizes `n_aux` from the same default JK-fitting auxiliary
basis the driver auto-resolves, so the estimate and the execution agree
on the auxiliary dimension. Only when no default is registered does it
fall back to the `3 * n_orb` design bound (Eichkorn et al., 1995). The
`density_fit=True` / `cosx=True` keyword arguments to `run_job` are
applied to the SCF options before the pre-flight, so a kwarg-configured
DF job is estimated on the DF branch, never as a direct-SCF job.

## Peak RSS is a high-water mark

Peak RSS records the largest resident memory observed at any point in the
process. It does not fall when an array is freed. If conventional RHF builds
and later releases a dense AO ERI tensor, the final peak still includes that
tensor. This is expected high-water-mark behavior and does not by itself
indicate that the tensor remained live.

`run_job(..., perf_log=True)` writes a `.perf` sibling with component wall
and CPU times plus RSS snapshots. Combine it with a whole-process
high-water-mark measurement when comparing algorithms. Record CPU time, wall
time, peak RSS, basis count, thread count, host, version, convergence
settings, and output plan together.

## When the estimate exceeds available RAM

```
vibe-qc estimates this calculation will require ~218.4 GB of memory:
    ERI tensor      186.0 GB
    ...
Available on this machine: 7.2 GB. ABORTING.

InsufficientMemoryError: Pass `memory_override=True` to `run_job` to proceed
anyway. In a low-level workflow, call `check_memory(..., allow_exceed=True)`.
To reduce the estimate, use a smaller basis, density fitting with an auxiliary
basis, or integral-direct SCF.
```

## Overriding the check

Pass `memory_override=True` to `run_job`:

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("large.xyz")
run_job(
    mol, basis="def2-tzvp", method="rhf",
    output="huge",
    memory_override=True,    # accept the risk of swap / freeze
)
```

When the estimate exceeds the available-memory probe, the output then reads
`Proceeding (override)` instead of `Proceeding` so anyone reading the log later
knows what happened. If the estimate already fits, the status remains
`Proceeding` because no override was needed.

## Estimators covered

| Method  | Dominant cost | Notes |
|---------|---------------|-------|
| RHF / UHF | Dense ERI tensor or direct-SCF shell-pair scratch | Direct SCF charges about 1 KB per shell pair |
| RKS / UKS | HF baseline + DFT grid, weights, coordinates, libxc scratch | meta-GGA adds tau buffers |
| MP2 / UMP2 | in-core OVOV or budgeted direct/disk panels plus DF factors | selected mode and panel dimensions are reported |
| CCSD / CCSD(T) | T1/T2 amplitudes, D1/D2 intermediates, budgeted triples workspace | phase peak, not a sum of released construction buffers |
| DLPNO-MP2 / DLPNO-CCSD(T) | PAO/PNO domains, pair lists, auxiliary pair workspaces, triples domains | conservative composition-level bound; geometry-dependent pair/PNO locality is not modeled |
| CAS/CI/FCI | CI or determinant vectors, RDMs, MO integral transforms | exponential methods report their vector storage |
| Selected CI | Dense Hamiltonian over the selected space, the selection-step candidate buffer, MO integral transform | sized by `SelectedCIOptions.target_size` and its growth factor, capped by the active space's determinant count, not by the full CI space |
| Periodic GDF / GPW / GAPW | GDF Lpq factors, bounded Ewald-J FT cache/batches, or FFT-grid collocation/cache arrays | Lpq and grid estimates use dry-run/live preflight; exact Ewald-J uses explicit cache and batch targets |

NEB uses a separate per-image-worker model and is intentionally not
folded into the single-point estimator.

The molecular DLPNO estimate is a conservative **composition-level bound**.
It sizes the pair and PNO workspaces from atom, basis, electron, threshold,
and thread counts, but it does not build the geometry-dependent pair domains
or model the locality of the requested molecule during pre-flight. Two
isomers with the same composition and basis can therefore receive the same
estimate even when their eventual strong-pair sets differ. On an extended or
dilute system, a refusal can be an overestimate of the local working set;
review the reported categories and use `memory_override=True` only when you
have independent peak-RSS evidence for that regime. The override changes only
admission and does not alter the DLPNO calculation.

## Correlated-method execution budgets

For MP2 and CCSD(T), preflight and execution share the same requested-memory
decision. `run_job(memory_budget_bytes=...)` caps the process allowance
further against `VIBEQC_MEMORY_LIMIT_BYTES`, `VQ_MEM_MB`, cgroup availability,
and host availability. It then reserves the estimator safety headroom and the
live SCF result matrices before passing the smaller method-owned allowance to
the native kernel. This distinction prevents an automatic slab or tile from
consuming the safety margin and then failing its own preflight check.
`memory_override=True` only bypasses admission; it does not disable a
requested native execution bound.

MP2 accepts `MP2Options.memory_mode` (or `UMP2Options.memory_mode`) with
`auto`, `incore`, `direct`, and `disk`. CCSD(T) accepts
`CCSDOptions.triples_memory_mode` with `auto`, `fast`, `blocked`, `direct`,
and `disk`. The corresponding result objects record the resolved strategy and
workspace. This telemetry is the scaling-regression contract.

Correlated estimates are phase peaks. SCF, DF construction, the CC iteration,
and triples do not all coexist, so the estimator takes their maximum while
retaining category detail for the controlling phase. Runtime baselines are
likewise maxima, not synthetic allocations added to every scientific tensor.
This is why the O2/def2-SVP estimate tracks the observed roughly 220 MB peak
without deleting the safety headroom.

The budget cannot erase irreducible state. A dense CCSD amplitude/integral
set or the open-shell spin-orbital four-index tensor can exceed the request
before triples begin; that route fails preflight rather than claiming the
triples tile made the full calculation bounded. Production coverage for the
large BUG 63 systems requires archived end-to-end peak-RSS runs.

The bounded kernel is not universal across every correlated variant. The
A-CCSD(T) Lambda correction, Brueckner orbital optimization, and FNO setup
still contain dense preparation phases. Preflight reports those phase peaks;
A-CCSD(T) rejects native blocked/direct/disk triples controls, while standard
CCSD(T) is the requested-memory route.

For molecular grid consumers, including RKS/UKS integration and COSX
workspace, the estimate follows the active angular scheme. Product grids use
`n_theta * n_phi`; unpruned Lebedev grids use the bundled point count for
`lebedev_order`. Angularly pruned and ORCA-style five-region grids
conservatively charge the densest active angular tier, so preflight remains an
upper bound even though some radial shells use fewer points.

Native molecular RKS and UKS evaluate AO values, AO derivatives, density
contractions, libxc vectors, and Fock projections in contiguous batches of at
most 4,096 points. Analytic molecular XC gradients use the same bound,
including their AO-Hessian tables. This does not coarsen or prune the grid:
every original Becke point and weight is included in the same order, and only
the association of the accumulated floating-point sums changes. Grid
coordinates, weights, and atom-ownership indices remain full-grid arrays.

VV10-paired functionals are the nonlocal exception. Their density and gradient
invariants are collected over the full grid, the VV10 double integral is
evaluated once with all cross-grid pairs present, and only the AO projection of
the resulting potential is batched. ROKS, TDDFT response setup, and explicitly
enabled molecular Newton/TRAH XC kernels still retain whole-grid AO tables and
are charged as dense routes. If the automatic large-RKS tail recovery decides
to enable TRAH after a non-converged first attempt, it performs a second dense
memory check before starting that retry.

The post-convergence UKS internal stability analysis uses the same 4,096-point
batched contract as the SCF loop: its f_xc matvec keeps only per-point vectors
across Davidson iterations and re-evaluates AO slices inside each matvec. The
response builder is released before any complete-energy line search or restart
SCF, so those bounded phases do not overlap in the preflight peak. Unsupported
response routes (meta-GGA, range-separated, VV10, DFT+U, active hybrid
RIJCOSX, and coupled solvent) skip the implicit check and allocate no stability
phase. Only the explicitly enabled Newton/TRAH second-order kernels retain the
dense whole-grid tables described above.

The 3D true-multi-k pure-DFT GDF route retains its analytic-FT Ewald J cache
only when the full tensor plus one similarly sized construction temporary
fits `VIBEQC_J_EWALD3D_CACHE_MIB` (4096 MiB by default). Larger cases contract
reciprocal vectors and output cells in batches, so they never retain an
`(n_cells, n_AO, n_AO, n_G)` complex tensor. Those batches use the separate
`VIBEQC_J_EWALD3D_FT_CELL_CHUNK_MIB` target (512 MiB by default). Lowering
either positive value trades reuse/batch size for memory; neither changes the
`VIBEQC_J_EWALD3D_KE` reciprocal cutoff or the energy convention.

## Reading the probe yourself

The cross-platform "how much memory is available right now" probe is
exposed for scripting:

```python
import vibeqc

available_gib = vibeqc.available_memory_bytes() / 1024**3
print(f"{available_gib:.1f} GiB")
```

An explicit `VIBEQC_MEMORY_LIMIT_BYTES` or `VQ_MEM_MB` value takes priority;
if both are set, the tighter limit wins. Without an explicit limit, the probe
first obtains a host-wide value from
[`psutil.virtual_memory().available`](https://psutil.readthedocs.io/) (install
with `pip install psutil` to get the highest-quality number). It falls back to
`/proc/meminfo` on Linux, `vm_stat` on macOS, and finally the generic
`os.sysconf` total-memory value when no available-memory probe succeeds. On
Linux, finite cgroup v2 (`memory.max` and `memory.current`) or cgroup v1
memory-controller limits bound the explicit or host value. Parent cgroup
limits are included, which is important when a scheduler gives each task an
unlimited leaf inside a limited job cgroup. The returned value is therefore no
greater than either the declared allocation or applicable host/cgroup limit.

The probe returns `0` if every applicable source fails. `check_memory` treats
that as "unknown" and silently proceeds rather than false-aborting on an
unsupported platform. A known exhausted cgroup is not unknown and fails the
preflight check.

## Writing your own estimator

If you call the low-level SCF drivers (`run_rhf`, `run_rks`, ...)
directly instead of through `run_job`, you can invoke the estimator
yourself:

```python
from vibeqc import estimate_memory, check_memory


def preflight_rhf(mol, basis, rhf_options):
    est = estimate_memory(mol, basis, method="rhf", options=rhf_options)
    print(est)                    # human-readable block
    check_memory(est)             # raises if the estimate exceeds the budget
    return est
```

The estimator returns a
[`MemoryEstimate`](../api/index.md) dataclass with a
`by_category` dict, so you can inspect exactly where the memory goes.

For conventional molecular SCF, the dense AO contribution is
$N_\mathrm{bf}^4\times8$ bytes. For direct SCF, the estimator replaces
that category with `Direct-SCF shell-pair scratch`. Switching the reference
SCF route does not remove method-specific MP2, CCSD, CCSD(T), response, or
property intermediates, so estimate the complete requested method.
