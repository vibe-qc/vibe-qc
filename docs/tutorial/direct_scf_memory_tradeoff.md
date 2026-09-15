# Direct SCF or in-core integrals?

Molecular Hartree-Fock and Kohn-Sham calculations repeatedly build Coulomb
and exchange matrices from electron-repulsion integrals (ERIs). There are two
valid ways to supply those integrals:

- **Conventional SCF** computes the full four-index AO ERI tensor once, keeps
  it in memory, and reuses it in every SCF iteration.
- **Direct SCF** computes screened shell quartets when each Fock matrix is
  built instead of retaining the full tensor.

This is a resource tradeoff, not a difference in the electronic-structure
model. Conventional SCF spends memory to save repeated integral work. Direct
SCF is willing to repeat integral evaluation to keep memory bounded. That
often costs more CPU time, but screening and cache behavior can reverse the
ordering on a particular system. Measure the crossover instead of assuming
one route is always faster.

| Route | Integral work | Memory behavior | CPU-time tendency |
|---|---|---|---|
| Conventional | form the AO tensor once, then contract it each iteration | dense $8N_\mathrm{bf}^4$ tensor remains available through SCF | larger startup, cheaper reuse when the tensor fits cache and RAM |
| Direct | evaluate screened shell quartets during each Fock build | shell-pair bounds and matrix-sized scratch, no dense AO tensor | repeated work, offset by Schwarz and density screening |

The best choice depends on the number and contraction pattern of the basis
functions, the number of SCF iterations, the available RAM, the memory
bandwidth and cache hierarchy, and the CPU.

## Why the basis-function count matters

For $N_\mathrm{bf}$ AO basis functions, vibe-qc's conventional molecular
path stores a dense double-precision tensor with

$$
M_\mathrm{ERI} = N_\mathrm{bf}^4 \times 8\ \mathrm{bytes}.
$$

The fourth power is unforgiving:

| Basis functions | Dense AO ERI tensor |
|---:|---:|
| 24 | 2.53 MiB |
| 95 | 621.42 MiB |
| 107 | 1000.06 MiB |
| 108 | 1.014 GiB |
| 170 | 6.223 GiB |
| 200 | 11.921 GiB |
| 250 | 29.104 GiB |
| 394 | 179.546 GiB |

These figures cover only the ERI tensor. The process also needs the basis,
one-electron, density, Fock, DIIS, eigensolver, Python, and requested-output
workspaces. Compare the complete estimate with the memory available to the
job, not with the machine's total installed RAM.

You can calculate the dense-tensor size before running:

```python
def dense_eri_gib(n_basis: int) -> float:
    return n_basis**4 * 8 / 1024**3


print(f"{dense_eri_gib(170):.3f} GiB")
# 6.223 GiB
```

## Select a mode explicitly

All four molecular SCF drivers, RHF, UHF, RKS, and UKS, use the same
`SCFMode` setting:

```python
from vibeqc import RHFOptions, SCFMode

opts = RHFOptions()

opts.scf_mode = SCFMode.AUTO
# Let vibe-qc select a route from the basis size.

opts.scf_mode = SCFMode.CONVENTIONAL
# Compute the dense AO ERI tensor once and retain it through SCF.

opts.scf_mode = SCFMode.DIRECT
# Recompute screened shell quartets for each Fock build.
```

`AUTO` is the normal choice. An explicit mode is useful for a controlled
benchmark or when a workflow must obey a known memory ceiling. Check the
installed version's `opts.scf_mode_auto_threshold` if the calculation is
close to the crossover. The current default is 140 basis functions. Releases
before vibe-qc v0.15.126 used 200, which is why the v0.15.65 case below
selected conventional SCF at 170 functions. A stricter memory-capped policy
can select direct SCF above 107 basis functions, the largest dense tensor
below 1 GiB:

```python
from vibeqc import RHFOptions, SCFMode

opts = RHFOptions()
opts.scf_mode = SCFMode.AUTO
opts.scf_mode_auto_threshold = 107
```

Explicit `DIRECT` does not depend on the installed AUTO threshold. Explicit
`CONVENTIONAL` remains available when you have budgeted the dense tensor and
all other process memory. The exception is an RKS or UKS range-separated
hybrid: its attenuated-exchange kernel is implemented only by the direct
builder, so vibe-qc forces `DIRECT` when density fitting is off. Density-fitted
range-separated exchange is not yet supported.

## Worked case: H2O2/RHF/cc-pVQZ

The performance-campaign case `qcil_00611_h2o2_rhf_cc_pvqz` contains 170
basis functions. It ran on one CPU on compute-cluster, with both programs in the same
scheduler allocation and on the same node. The vibe-qc run was compute-cluster job
`cc69b45aec9a`, using v0.15.65 at commit `1bd3ab31681e`:

| Program and route | Active CPU time | Process wall time | Peak RSS |
|---|---:|---:|---:|
| vibe-qc v0.15.65, conventional SCF | 45.032 s | 45.550 s | 6611.05 MiB |
| ORCA 6.1.1 reference | 51.232 s | 52.669 s | 126.25 MiB |

vibe-qc used 12.1 percent less CPU time and 13.5 percent less wall time in
this matched one-core run. Its peak RSS was 52.365 times the ORCA value. The
conventional vibe-qc allocation is explained almost completely by

$$
170^4 \times 8 = 6{,}681{,}680{,}000\ \mathrm{bytes}
                 = 6.223\ \mathrm{GiB}.
$$

The RHF energies agreed to $1.53\times10^{-10}$ Ha. The calculation was
scientifically correct and faster in this comparison; it simply occupied
much more memory because it chose to retain the full AO tensor.

A local direct-SCF validation of the same input reduced whole-process peak
RSS to 914.92 MiB. The maximum observed by the end of SCF was 270.4 MiB, and
the energy moved by only $-1.22\times10^{-12}$ Ha. Its process time was
23.51 CPU-s and 23.56 wall-s, but that run used a different host. Those times
confirm that the route is practical; they are not a same-host speed
comparison with the compute-cluster measurements.

This result also illustrates an important measurement detail. Peak RSS is a
process high-water mark. Freeing the conventional tensor after RHF does not
lower the recorded peak, and a later output step can raise it again. A high
end-of-run peak therefore does not by itself prove that the tensor remained
live.

## Preview memory before running

The high-level `run_job` entry point performs a memory preflight. Low-level
driver users can call the same estimator directly:

```python
from vibeqc import (
    BasisSet,
    Molecule,
    RHFOptions,
    SCFMode,
    check_memory,
    estimate_memory,
)

mol = Molecule.from_xyz("h2o2.xyz")
basis = BasisSet(mol, "cc-pvqz")

for mode in (SCFMode.CONVENTIONAL, SCFMode.DIRECT):
    opts = RHFOptions()
    opts.scf_mode = mode
    estimate = estimate_memory(
        mol,
        basis,
        method="rhf",
        options=opts,
    )
    print(mode, estimate)

# Apply the guard to the mode you intend to run.
opts = RHFOptions()
opts.scf_mode = SCFMode.DIRECT
estimate = estimate_memory(mol, basis, method="rhf", options=opts)
check_memory(estimate)
```

The conventional estimate contains an `ERI tensor` category. The direct
estimate contains `Direct-SCF shell-pair scratch` instead. Treat estimates as
planning values and peak RSS as the measured result.

## Measure CPU time and memory together

Wall time alone can hide the cost of extra threads, and CPU time alone does
not tell you whether the job fits in memory. Record at least:

- basis-function count and shell count;
- SCF mode and screening thresholds;
- SCF iteration count and final energy;
- active CPU time and process wall time;
- whole-process peak RSS;
- host CPU model, thread count, program version, and commit;
- requested outputs, because cube and population analyses can change the
  process peak.

Ask `run_job` for a component-level performance log:

```python
from vibeqc import Molecule, RHFOptions, SCFMode, run_job

mol = Molecule.from_xyz("h2o2.xyz")
opts = RHFOptions()
opts.scf_mode = SCFMode.DIRECT

run_job(
    mol,
    basis="cc-pvqz",
    method="rhf",
    rhf_options=opts,
    output="h2o2-direct",
    perf_log=True,
)
```

The resulting `h2o2-direct.perf` reports SCF wall and CPU time plus RSS
snapshots. For a whole-process high-water mark on Linux, wrap the command
with `/usr/bin/time -v`; on macOS use `/usr/bin/time -l`. Keep the two modes
on the same host, with the same thread count, geometry, basis, convergence
tolerances, initial guess, and output plan.

## Run a crossover sweep safely

A useful sweep varies basis size first and SCF mode second. Start with
`STO-3G`, `6-31G*`, `cc-pVDZ`, `cc-pVTZ`, and `cc-pVQZ` on one molecule.
For every row:

1. Construct the basis and record `basis.nbasis`.
2. Calculate $N_\mathrm{bf}^4\times8$ and run `estimate_memory`.
3. Skip a conventional row unless the complete estimate fits comfortably in
   the job's memory allocation.
4. Run conventional and direct modes from the same initial conditions.
5. Compare energy, iterations, CPU time, wall time, and peak RSS.
6. Repeat enough times to distinguish a stable timing from filesystem,
   scheduler, or cold-cache noise.

Do not expect CPU time to follow $N_\mathrm{bf}^4$ exactly. Direct-SCF cost
also depends on shell contractions, molecular locality, Schwarz screening,
density screening, and how quickly the SCF converges. This is why the
crossover should be measured on the systems and hardware that matter to the
workflow.

## Post-SCF calculations need a separate budget

Choosing direct SCF bounds the memory of the reference Fock build. It does not
guarantee that MP2, CCSD, CCSD(T), response, or property code will avoid its
own four-index transforms and intermediates. Estimate and measure the complete
requested method. A low-memory RHF reference can still be followed by a
high-memory correlation step.

For the option details, screening controls, and citations, see
[SCF Fock-build modes](../user_guide/scf_modes.md). For the preflight model
and override behavior, see [Memory budget](../user_guide/memory.md).
