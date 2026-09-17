---
myst:
  html_meta:
    "description": "Build the optional OpenTrustRegion backend and compare molecular RHF, UHF, RKS and UKS calculations with native SCF, including physical convergence and work-count diagnostics."
---

# Molecular SCF with OpenTrustRegion

This tutorial compares two ways of optimizing the same electronic state:
native SCF and the optional OpenTrustRegion orbital optimizer. Start with
neutral water at RHF/STO-3G, then repeat with PBE and the open-shell water
cation. The nuclei stay fixed throughout.

SCF stationarity means the occupied orbitals no longer mix with virtual
orbitals to first order in the energy. OpenTrustRegion uses orbital gradients
and Hessian-vector products to propose a step within a trust radius, evaluates
the trial energy, and accepts or rejects that step. vibe-qc supplies the
energies and physical derivatives; the pinned upstream library performs the
optimization. The electronic method, basis and DFT grid still determine the
energy being optimized.

For option definitions and supported combinations, keep the
[OpenTrustRegion reference](../user_guide/opentrustregion.md) nearby. For
native Newton, SOSCF and TRAH, see [second-order SCF](second_order_scf.md).
The [theory chapter](../user_guide/opentrustregion_theory.md) explains the
rotation coordinates, physical derivatives, trust-radius updates and
difference between stationarity and stability.

## 1. Build and check the optional component

Begin with a working native installation from the
[contributor setup](../contributor_setup.md). Add a Fortran compiler and LP64
BLAS/LAPACK, then rebuild in that installation's environment:

```sh
CMAKE_ARGS="-DVIBEQC_ENABLE_OPENTRUSTREGION=ON" \
  .venv/bin/pip install -e . --no-build-isolation
.venv/bin/python -c "import vibeqc; print(vibeqc.has_opentrustregion())"
```

The probe must print `True`. Default builds have the component disabled.
CMake fetches version 2.0.0 at the exact revision documented in the reference;
the build uses 32-bit BLAS integers. Installing a separate Python optimizer
package does not enable the native backend.

Use the same Python executable to run the examples. A missing capability
raises an error before calculation; explicitly selecting this backend never
silently chooses a native optimizer instead.

## 2. Compare water RHF from the same starting guess

From the repository root, create an output directory outside the checkout:

```sh
otr_runs=$(mktemp -d)
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method rhf --output-dir "$otr_runs/rhf"
```

Keep `otr_runs` in this shell for the commands below. The example writes the
usual output families under that directory using `run_job`; the final console
summary gives energies, density agreement, convergence residuals and work
counts. The full input is available as
{download}`01_compare_scf.py <../../examples/opentrustregion/01_compare_scf.py>`.

The molecule has oxygen at the origin and hydrogens at `(0, +/-1.43, 1.1)`
**bohr**. Both runs use STO-3G, ten electrons, an HCORE guess, an energy
tolerance of `1e-9` Ha and an AO commutator tolerance of `1e-7`. The
OpenTrustRegion orbital-gradient RMS tolerance is `1e-8`.

The central API selection is:

```python
import vibeqc as vq

molecule = vq.Molecule([
    vq.Atom(8, [0, 0, 0]),
    vq.Atom(1, [0, 1.43, 1.1]),
    vq.Atom(1, [0, -1.43, 1.1]),
])
options = vq.RHFOptions()
options.initial_guess = vq.InitialGuess.HCORE
options.max_iter = 100
options.conv_tol_energy = 1e-9
options.conv_tol_grad = 1e-7
result = vq.run_job(
    molecule, basis="sto-3g", method="rhf", rhf_options=options,
    orbital_optimizer="opentrustregion", output="water-otr",
    name_molecule=False, num_threads=1,
)
```

Run this standalone snippet in a calculation directory, since `output=` is
relative to the current directory. The downloadable script additionally runs
the native comparison and checks the results.

Expect an RHF energy near **-74.9624416490 Ha** for this geometry. The script
requires the two energies to agree within `2e-8` Ha and every AO density
element within `2e-6`. It also requires physical convergence and a converged,
stable verdict for real restricted orbital rotations. These checks establish
that the two routes reach the same state on this example. They do not certify
the basis-set limit or a global electronic minimum.

## 3. Repeat for the four supported SCF methods

```sh
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method rks --output-dir "$otr_runs/rks"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method uhf --output-dir "$otr_runs/uhf"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method uks --output-dir "$otr_runs/uks"
```

| Method | System | Electrons | Energy functional |
|---|---|---|---|
| RHF | neutral water, singlet | 5 alpha, 5 beta | Hartree-Fock |
| RKS | neutral water, singlet | 5 alpha, 5 beta | PBE |
| UHF | water cation, doublet | 5 alpha, 4 beta | Hartree-Fock |
| UKS | water cation, doublet | 5 alpha, 4 beta | PBE |

The unrestricted examples use bent water cation so that a freely orientable
hole in a linear molecule does not obscure the density comparison. Both spin
densities must agree independently. The two optimizers always receive the
same charge, multiplicity, basis and starting guess within each comparison.

The PBE examples explicitly set `n_radial=25`, `n_theta=12`, `n_phi=24`.
This small grid keeps the tutorial quick. It is a numerical teaching setup,
not a grid-converged prediction. When changing grid quality, change it for
both optimizers before comparing energies. Retain the usual production grid
defaults, or perform a grid-convergence study, for a scientific calculation.

A validation run of these exact inputs gave the following energies. Final
digits can vary by platform; the scripts check agreement at the tolerances
above rather than requiring identical printed strings.

| Method | Native energy, Ha | OpenTrustRegion energy, Ha | Maximum density-element difference |
|---|---:|---:|---:|
| RHF | -74.962441649032 | -74.962441649032 | 6.2e-9 |
| PBE RKS | -75.223866053105 | -75.223866053105 | 4.7e-12 |
| UHF | -74.654926151890 | -74.654926151890 | 1.1e-9 |
| PBE UKS | -74.867944410060 | -74.867944410060 | 2.3e-8 |

## 4. Read convergence and stability separately

The standard `.out` file records the backend, pinned version, subsolver,
rotation manifold, termination, work counts and stability verdict. It also
includes the OpenTrustRegion method citation. The same fields are available
on the result:

```python
report = result.opentrustregion
print(result.converged, report.termination, report.error_code)
print(report.energy_converged, report.gradient_converged)
print(report.final_residual, report.gradient_rms)
print(report.stability_checked, report.stability_converged, report.stable)
print(report.manifold)
```

`result.converged` requires a successful solve, small physical energy change,
small AO commutator residual and small orbital-gradient RMS. A small step or
an upstream machine-precision message alone does not meet this contract.

Stability asks a different question: is there downhill curvature at the
stationary point? Read `stable` only together with `stability_checked` and
`stability_converged`. The verdict applies to the reported real restricted
or unrestricted orbital manifold. In particular, an internally stable RHF
state can still lower its energy if alpha and beta orbitals are allowed to
separate. The [restart and stability tutorial](opentrustregion_stability.md)
demonstrates that distinction on stretched H2.

## 5. Compare work, then tune one option at a time

An orbital step may require many Fock builds and Hessian-vector products.
`result.n_iter` counts accepted model callbacks, including initialization;
it is not directly comparable to a native SCF iteration count. The upstream
C API does not expose macro/micro totals, so those fields are `-1` and output
labels them unavailable.

Use `accepted_evaluations`, `trial_evaluations`, `fock_evaluations` and
`response_evaluations` to understand the OpenTrustRegion work. For a deeper
comparison, the repository benchmark counts actual J and K calls, including
response builds:

```sh
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_opentrustregion.py \
  --case water --basis sto-3g
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_opentrustregion.py \
  --case o2-quintet --basis cc-pvdz
```

The second case is a harder open-shell example. Timings include the Python
counting wrapper and are descriptive, not portable performance targets.
Fewer accepted steps can still mean more response work and longer runtime.
The benchmark uses its own stated DFT grid, so do not compare its absolute
PBE energy with a different-grid tutorial run.

To try another upstream subsolver, pass a separate options object:

```python
otr = vq.OpenTrustRegionOptions()
otr.subsystem_solver = "jacobi-davidson"  # also "davidson" or "tcg"
otr.initial_trust_radius = 0.4
otr.stability = "check"
# Add opentrustregion_options=otr to the run_job call above.
```

Repeat the energy/density checks after each change. Keep the native
Newton/SOSCF/TRAH thresholds at zero for an OpenTrustRegion calculation;
explicit conflicting convergers are rejected. A line search is available
but can add many energy evaluations.

## Next steps

Continue with [READ restarts and saddle following](opentrustregion_stability.md).
The [reference](../user_guide/opentrustregion.md) lists unsupported functionals
and workflows, explains failure diagnostics, and gives the derivative and
accepted-state contract. The method citation is Greiner et al., *JCTC* **22**,
881-895 (2026), [doi:10.1021/acs.jctc.5c01576](https://doi.org/10.1021/acs.jctc.5c01576).
