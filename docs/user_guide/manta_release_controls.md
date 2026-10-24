# MANTA controls for the bounded 0.18 release

The approved 0.18 scope qualifies existing MANTA controls; it does not require
completion of the full MANTA design. MANTA remains optional and experimental.
The examples here use neutral, all-electron three-dimensional cells, fixed
occupations, unreduced regular meshes, RI-J and exact full-Coulomb K with the
explicit BvK Ewald convention. Physical supports remain fixed during SCF.
The public entry point is `vibeqc.run_periodic_job` with the existing typed
MANTA options. No default, symmetry permission or Fock convention changes.

## Frozen examples

The source-tree qualification driver is
[`benchmarks/manta_release_controls.py`](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.py).
It reuses the accepted fixtures rather than introducing another physical
producer. Run it from a matching development environment with NumPy, SciPy
and pytest installed, through the calculation queue for scientific acceptance:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python benchmarks/manta_release_controls.py --output /tmp/manta-controls
```

Use `--profile` to select only a case whose acceptance is missing. The driver
runs each enabled case and its fixed-source control, checks convergence and
QVF validity, and verifies strict-target refusal before a computed archive is
written. It prints source/native identity, full basis and physical supports,
certificate targets/results, modeled memory and measured CPU/wall cost.
Managed calculation output and QVF preserve the evaluated state. Keep this
record outside the checkout and associate it with the queue job and runtime.

| Profile | Method and mesh | Controls and requested targets |
| --- | --- | --- |
| `image-tight` | Shifted two-point Be RHF | S, T and density-family kinetic-energy image targets 1e-6; physical pair radius 10 bohr and auxiliary radius 12.1 bohr |
| `ordinary-rhf` | Shifted two-point Be RHF | Joint image, full-Coulomb exchange and nuclear admission; loose physical targets exercise the control path |
| `global-pbe0` | Shifted two-point Be PBE0 RKS | The same joint controls with the method's exact-exchange fraction and a fixed XC quadrature |
| `compact-tight` | Gamma He RHF, compact custom basis | Image, exchange and nuclear component targets 1e-8; pair radius 1 bohr, nuclear radius 18 bohr, reciprocal energy cutoff 512 Eh |

The ordinary RHF/PBE0 cases request image targets 1e6, exchange targets 1e12
and nuclear targets 1e6. These intentionally loose targets are not production
accuracy recommendations. Their RI coupling J/energy targets are 1e-6 over
the automatically admitted canonical density family. The compact case uses
exact finite coupling; it does not request a multipole truncation target.
The explicit resource caps are 4 GiB memory, 1e12 shell calls and 1e12 primitive
products. Scalar/kernel allowances are 7e18 for `image-tight` and 2e18 for the
other profiles, including current arithmetic reservations. These are worst-case
admission allowances, not measured operations or runtime estimates. A resource
refusal remains a failure to admit the requested case, not a scientific PASS.

The driver prints the precise contracted basis coefficients, geometry,
mesh shift and domains. These custom small fixtures demonstrate the controls;
they are not converged material predictions for a paper. A paper's actual
materials still need their own basis, k mesh, XC grid and physical-domain
convergence evidence.

## What the bounds mean

| Control | Included comparison | Exclusions relevant to release claims |
| --- | --- | --- |
| `one_electron_accuracy` | Omitted AO-image overlap and kinetic matrix contributions, plus kinetic energy over the admitted canonical occupied-space family | Arithmetic, overlap-rank stability, other energy components and density relaxation |
| `exchange_accuracy` | Omitted full-Coulomb exchange domains, with the method scale and fixed Madelung correction accounted for | Madelung numerics, arithmetic, screened/RI exchange, finite fitting error, relaxation and k-mesh error |
| `nuclear_accuracy` | Omitted nuclear potential and repulsion contributions, plus attraction energy over the same density family | Arithmetic, metric solve, other energy components and density relaxation |
| RI coupling `accuracy` | Approximate coupling action relative to its declared finite source, propagated to fixed-family J/Coulomb energy | Finite fitting-basis error, omitted physical source domains, arithmetic and relaxed SCF energy |

These certificates are separate ledgers with different reference objects.
Do not sum them and call the result a total relaxed SCF energy tolerance.
Inspect every requested certificate's `qualified` flag and measured bounds,
not just a successful process exit. Do not report an iteration-capped state as
converged. The existing full-Coulomb fitting and stored-metric controls retain
their separately documented scope in the [MANTA manual](manta.md).

The frozen release examples exclude RI-K exchange certificates, screened
exchange certificates, source-specific symmetry reduction, low-dimensional
boundaries, gradients/stress, correlated production claims and speedup claims.
Pair-tree metric-action bounds are a separate MANTA extension outside these
frozen examples. Actual integral/Fock contraction roundoff remains later
MANTA work; it does not hold this approved bounded release.

## Acceptance reuse and candidate handoff

The accepted component and complete-method evidence is recorded in the
[image](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_image_accuracy.md),
[exchange](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_exchange_accuracy.md)
and [nuclear](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_nuclear_accuracy.md)
benchmarks. Their independent Gaussian/Boys/Fourier/libint and NumPy/SciPy
references remain valid for unchanged components. No matched ORCA/CRYSTAL
reference is available for these custom finite domains; KS also shares libxc.
Fixed-source comparisons alone do not establish independent bulk parity.

For a containing candidate, retain the accepted source/native identities and
verify the relevant implementation and fixture files. Reuse valid evidence;
run only changed behavior or missing combinations. The compact simultaneous
image/exchange/nuclear configuration is the added integration check here.
Keep failed comparisons and unsupported combinations explicit. Exact candidate
CI, publication checks and the immutable release tag remain separate release
requirements. This page qualifies neither the complete 0.18 release nor all
MANTA capabilities by itself.


## Recorded qualification

The [bounded control benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md)
records eight converged complete calculations, ten strict-target refusals and
bitwise reuse of saved accepted states on source `7917b560ffd3d48fbbd1bf4f0ecb66da898e7b4d`.
The compact simultaneous-control example passes all selected 1e-8 component
targets. This closes the missing integration check for this frozen MANTA
subset. The later shared continuous-grid integration adds two PBE0
calculations, three strict refusals and an independent NumPy/SciPy comparison;
the unaffected HF evidence is reused. Its updated source/native identities
and measured errors are in the same benchmark. MANTA admission includes the
shared partition switching band. A containing 0.18 candidate must retain this
scope and evidence and pass the release gates; the overall release is not
declared ready here.


The [corrected-candidate linkage](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md#corrected-018-candidate-linkage)
also qualifies these physical profiles on source
`2d0fdb8ca06b5e37c1f6721667800e9954defdfe`: candidate `95c18c7e` plus
the converter-header repair `a92b4880` and the explicit `image-tight` work
allowance update after RI preflight reservations changed. Eight complete
calculations and ten strict refusals pass
with the documented caps; the old work-cap refusal is retained, and saved
physical states and component certificates are bitwise unchanged. The benchmark retains the actual local native identity,
cost and independent-reference reuse. This establishes the bounded MANTA
contribution for that corrected source. The release candidate must include
the repair and allowance update and reconcile its final identity. Its other
gates and the paper's material-specific convergence evidence remain separate.
