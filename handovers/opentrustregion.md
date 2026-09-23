# OpenTrustRegion molecular adapter

Implemented against vibe-qc base `5a059a037d0781de7b930d79fc094395eeac0fe3`
and upstream revision `8fa7769ae66233a566868a6bf03cdcbdb1ee69d0` (2.0.0).
The upstream defining paper, SI Table S1, C header, Fortran callbacks and C
tests were checked against this revision. This is an optional molecular
SCF implementation; localization, CASSCF and periodic adapters remain gated.

## Implementation boundaries

- `cpp/src/opentrustregion.cpp`: chemistry-independent C ABI adapter, owned
  callback context, process-wide exclusion, nested-call refusal, error and
  logging translation. Upstream owns step acceptance and line search.
- `cpp/src/orbital_objective.cpp`: molecular occupied-virtual coordinates,
  physical gradients and matrix-free coupled-spin Hessian actions, isolated
  trials and final physical state reconstruction.
- `cpp/include/vibeqc/orbital_scf.hpp`: molecular convergence and result
  mapping. Four existing SCF drivers supply their JK and XC response paths.
- `cpp/cmake/OpenTrustRegion.cmake`: optional pinned source build, int32/LP64
  C/Fortran ABI, portable instructions, license installation. The default
  build does not enable Fortran.
- `python/vibeqc/runner.py` and `vibeqc.output`: explicit public selection,
  no automatic native retries, capability/unsupported-mode diagnostics,
  actual-use citation and structured result/report fields.

The user guide documents the initial-saddle behavior of the pinned solver,
the difference between its fixed negative-mode threshold and the requested
final stability threshold, and the limits of a real restricted stability
verdict. No eigenvalue or iteration total is inferred from log text.

## Validation commands

Use a dedicated clone and its own editable environment. Rebuild after native
changes and check that the loaded extension is newer than every C++ source.

```sh
CMAKE_ARGS="-DVIBEQC_ENABLE_OPENTRUSTREGION=ON" \
  VIBEQC_REQUIRE_VENDORED=ON .venv/bin/pip install -e . --no-build-isolation

OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_opentrustregion.py -q

OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/test_gate/run_full_suite.py \
  --wt "$PWD" --py "$PWD/.venv/bin/python" --jobs 4 \
  --lane smoke,molecular-scf-dft,output-docs --out /tmp/otr-lanes.jsonl

OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_uhf_newton.py tests/test_trah.py tests/test_uhf_stability.py \
  tests/test_rks_second_order.py tests/test_uks_second_order.py \
  tests/test_memory.py -m 'not slow' -q

.venv/bin/sphinx-build -b html docs/ docs/_build/html

OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/bench_opentrustregion.py --case water
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/bench_opentrustregion.py --case o2-quintet --basis cc-pvdz
```

The local build used AppleClang, GNU Fortran, Apple Accelerate LP64 and
Python 3.14 on arm64 macOS. All four upstream C ABI/settings smoke tests
passed; an `INTEGER_SIZE=8` configure request was rejected with the int32/LP64
diagnostic. The enabled install contains the upstream MPL license notice.
A separate adapter executable compiled without the optional library
reported capability false and the expected rebuild diagnostic. A minimal
CMake project including the optional module with it disabled configured with
an invalid Fortran-compiler path, confirming that the off path does not
enable Fortran. These standalone checks are distinct from the enabled full
extension test run.

The final focused module contains 39 passing tests. The supplemental native
Newton/TRAH/stability, second-order KS and memory selection passed 302 tests
with two explicitly slow tests deselected. After the last Python multi-guess
guard change, the focused module plus native UHF stability checks passed
51 tests, again excluding those two slow cases. The expected no-verdict
warning is exercised by a native stability regression. Sphinx HTML completed
without warnings, and the privacy hook, diff check and 190-section released
changelog guard passed.

The three broader lanes selected 122 file targets: 118 passed, two optional
TREXIO modules skipped because TREXIO was absent, and two periodic tests hit
the 300-second per-test timeout. Completed targets reported 5,002 passing
tests. The timed-out targets were `test_feature_citation_end_to_end.py`
(the KPPRA periodic job) and `test_periodic_molden_gamma_export.py`.
`gate_verdict.py --reverify` accepts only T0/T1 records and rejected the mixed
T2 lane input, so equivalent serial reruns used:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/test_gate/run_full_suite.py \
  --wt "$PWD" --py "$PWD/.venv/bin/python" --lane output-docs --jobs 1 \
  --only test_feature_citation_end_to_end.py,test_periodic_molden_gamma_export.py \
  --test-timeout 600 --file-timeout 1800 --out /tmp/otr-isolated.jsonl
```

The periodic Molden target passed on the isolated rerun (542.6 seconds for
the file). The citation target repeated its timeout at 600 seconds per test. Stack
sampling placed the work in `compute_gdf_range_separated_integrals` and
`ao_pair_gaussian_fourier_panel`, before SCF, with the native optimizer.
Those numerical paths are unchanged by this implementation. This is an
unverified periodic check, not evidence of a numerical mismatch or a claimed
green full lane. No periodic numerical changes or baseline exemptions were
made to obtain a pass.

The focused tests cover all three subsystem solvers, indefinite objectives,
zero rotations, callback exceptions, nonfinite trials, iteration exhaustion,
inconclusive stability, nested/concurrent calls, molecular failure state,
off-stationary noncanonical HF/LDA/PBE/PBE0 derivatives, coupled-spin response,
the restricted/unrestricted limit, direct-cache isolation, RIJK, all four SCF
routes, restart, output/citation and unsupported-mode diagnostics. Stretched
H2 tests unrestricted saddle escape. Quintet O2/cc-pVDZ from HCORE tests a
harder open-shell start; native stability correction is enabled when comparing
the final physical solution.

## Main integration and runnable documentation

After rebasing onto main revision `28d2604`, rebuilt the enabled native
extension and verified that its timestamp was newer than every file in
`cpp/`. The package loaded from the dedicated checkout and the capability
probe returned true. Ran:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_opentrustregion.py tests/test_uhf_newton.py tests/test_trah.py \
  tests/test_uhf_stability.py tests/test_rks_second_order.py \
  tests/test_uks_second_order.py tests/test_memory.py \
  tests/test_test_gate_lanes.py -m 'not slow' -q
```

Result: 412 passed, 2 deselected, one expected no-verdict warning, and one
unrelated failure in
`test_periodic_ccm_test_files_carry_experimental_pytestmark`: `test_kpoints.py`
imports experimental CCM without a module-level experimental marker. Both
the policy test and offending file are byte-identical to `origin/main`;
executing that baseline policy function on the baseline offending source
reproduced the same assertion. No marker or gate exemption was changed.

All six commands in `examples/opentrustregion/README.md` passed: four native
comparisons, the public UKS READ restart, and stretched-H2 saddle following.
Checked the normal output files for backend provenance and the Greiner
citation. The tutorials record the measured energies and density agreement;
the UKS restart retained five alpha and four beta electrons, and H2 lowered
its energy by 0.174760 Ha while retaining one electron per spin. Generated
job files and validation logs stayed outside the checkout.

Final integration is based on main revision `d660fae`, including the Python
Euler-extraction symmetry fix. The OpenTrustRegion implementation stayed
identical. Rebuilt, verified native freshness and capability, and ran
`tests/test_opentrustregion.py` with `tests/test_symmetry_core.py`: all 208
passed. The three HTML input downloads were checked byte for byte against
the tested scripts. The final standard Sphinx warnings-as-errors HTML build
passed, including both tutorials and the reference page.

## Theory and citation verification

The theory chapter explains the energy functional, nonredundant rotations,
physical derivatives, quadratic model, subsolver metrics, step acceptance and
stability. Its library citation is rendered from the runtime database,
including the verified journal issue number. The pinned TCG implementation
uses a preconditioner-weighted radius, while Davidson uses a Euclidean one.

An executed OpenTrustRegion result now selects its own converger citation
route. Previously, returning no accelerator selected the registry's default
DIIS references. The citation CLI now preserves the recorded runtime keys,
order and visibility, using current database metadata; its old plan-only
reconstruction omitted runtime choices such as this optimizer. Older
manifests without recorded entries retain the plan fallback. Invalid recorded
entries refuse regeneration before any bibliography file is rewritten.

Validation: 1,890 tests passed across `test_opentrustregion.py`,
`test_vibeqc_cite_cli.py`, `test_citations.py`,
`test_citation_database_integrity.py`, `test_citations_printable.py` and
`test_scf_accelerator_citation_coverage.py`. Four additional citation
reachability checks passed. The eight real-run citation cases cover native
and OpenTrustRegion RHF/UHF/RKS/UKS: correct DOI presence or absence, complete
BibTeX metadata, no default DIIS credit for OpenTrustRegion, and byte-identical
CLI bibliography regeneration. The runnable RHF comparison also passed,
including a command-line citation reprint. The warnings-as-errors Sphinx
build passed; the generated theory HTML contains the equations and the
database-rendered journal reference. All 191 released changelog pins match.

## Work-count observations

The benchmark uses the same molecular integrals, HCORE guess, tolerances and
grid for both backends. J/K counts include response and stability work.
OpenTrustRegion Fock counts include initial setup and final verification.
Wall times include SCF and stability but exclude common integral setup and
the later density-alignment diagnostic. Both routes use the same Python JK
counting wrapper, whose callback overhead is included and whose J/K calls
do not use a fused Fock-build shortcut. These are observations under
concurrent host load, not CI assertions or a speed claim.

| Case | Method | Native J/K | OTR J/K | OTR Fock/response | Native/OTR seconds | Energy difference, Ha |
|---|---|---:|---:|---:|---:|---:|
| Water/STO-3G | RHF | 28/47 | 119/119 | 63/56 | 0.0010/0.0037 | -4.3e-14 |
| Water/STO-3G | PBE RKS | 18/0 | 71/0 | 13/58 | 0.814/0.746 | 1.4e-14 |
| Quintet O2/cc-pVDZ | UHF | 110/220 | 523/1046 | 27/496 | 0.306/2.747 | -5.7e-14 |
| Quintet O2/cc-pVDZ | PBE UKS | 21/0 | 894/0 | 55/839 | 3.342/33.906 | 2.2e-11 |

Water densities agree within 1.4e-8 in Frobenius norm. O2 initially differs
by 0.272 (UHF) and 0.627 (UKS). The script fits the AO representation of a
rigid rotation about the molecular axis and applies the *same* rotation to
both spins; residual differences become 6.7e-8 and 3.4e-8. This identifies
rotationally equivalent densities instead of silently comparing different
states. Neither a smaller accepted-step count nor equal energy alone is used
as evidence of a speedup or an identical density.

## Next adapters and acceptance gates

1. **Molecular localization.** Inspect `python/vibeqc/iao.py` and the native
   localization objectives. Begin with Boys localization in an occupied
   subspace; minimize the negative of a maximized measure. Enumerate unique
   rotations inside that subspace, then add an independently selected virtual
   subspace. Implement an `OrbitalObjective` whose trials freeze both the
   reference orbitals and property-integral transforms. Require off-stationary
   energy/gradient finite differences, Hessian symmetry, arbitrary trial-order
   invariance, preserved subspace projectors and agreement of objective values
   with the existing localizer. Only then expose a backend option. Cite
   Hoyvik, Jansik and Jorgensen, JCTC 8, 3137 (2012).
2. **Molecular CASSCF.** Start at `python/vibeqc/solvers/_casscf.py`, especially
   `_orbital_gradient_and_fock`, rotations and Newton-CG. Enumerate nonredundant
   inactive-active, inactive-virtual and active-virtual rotations. Specify
   whether CI relaxation is eliminated through a response solve or retained
   as coupled variables; include that coupling in the Hessian. Freeze accepted
   orbitals, CI roots, RDMs and integral caches during trials, with explicit
   root-overlap tracking and state-average weights. Gate public selection on
   finite differences of the fully relaxed objective, Hessian symmetry,
   rejected-trial rollback and matched small active-space energies/root
   identities against the existing solver. A fixed-CI Hessian alone is not
   sufficient. Cite Helmich-Paris, JCP 156, 204104 (2022).
3. **Periodic localization/SCF.** Treat this as a separate parameterization,
   even at Gamma. Specify complex anti-Hermitian independent variables,
   conjugation, k weights, time-reversal/sewing relations, gauge and occupied
   projectors before wiring callbacks. Test rephasing, k-star permutations,
   weighted energy derivatives, Hessian Hermiticity in real coordinates and
   electron-count conservation on multiple meshes. Keep periodic capability
   unavailable until those derivative and state tests pass.
