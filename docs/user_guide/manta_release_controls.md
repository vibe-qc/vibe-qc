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
The requested budgets for the frozen profiles are:

| Profile | S / T / kinetic-energy image targets | Exchange matrix / energy targets | Nuclear matrix / attraction-energy / repulsion-energy targets | Finite-source coupling J / Coulomb-energy targets |
| --- | --- | --- | --- | --- |
| `image-tight` | 1e-6 / 1e-6 / 1e-6 | Not requested | Not requested | 1e-6 / 1e-6 |
| `ordinary-rhf` | 1e6 / 1e6 / 1e6 | 1e12 / 1e12 | 1e6 / 1e6 / 1e6 | 1e-6 / 1e-6 |
| `global-pbe0` | 1e6 / 1e6 / 1e6 | 1e12 / 1e12 | 1e6 / 1e6 / 1e6 | 1e-6 / 1e-6 |
| `compact-tight` | 1e-8 / 1e-8 / 1e-8 | 1e-8 / 1e-8 | 1e-8 / 1e-8 / 1e-8 | Not requested; exact finite coupling |

S is dimensionless; the other matrix and energy bounds are in atomic units
of energy. Energy bounds apply to the declared canonical density family
and component, not to the relaxed total energy. A nuclear certificate's
`total_energy_error_bound` combines nuclear attraction and repulsion only.
Zero coupling-truncation error for an exact finite contraction does not
bound fitting-basis incompleteness, omitted physical domains or arithmetic.
An unrequested ledger is not a zero-error certificate. The ordinary profiles'
large physical budgets establish admission behavior, not material accuracy.

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

## Composition of the certificates

The provenance of a periodic RI HF job carries a `component_ledger` that
composes the separate certificates without double counting. Each requested
certificate is mapped to the disjoint energy component it bounds at the
canonical density and to its own reference object: `one_electron_accuracy`
to the kinetic energy (omitted overlap and kinetic images),
`nuclear_accuracy` to nuclear attraction plus repulsion (omitted real and
reciprocal images), `exchange_accuracy` to exact exchange (omitted
full-Coulomb domains with the fixed Madelung correction and method scale) and,
when a coupling target was requested, `fitting_accuracy` to the Coulomb energy
through the RI coupling action against its declared finite source. Because
the components are disjoint, `fixed_density_energy_error_bound` is their
outward-rounded sum. Each positive binary64 addition advances toward positive
infinity; zero terms and a single stored bound remain exact. The receipt
records `composition_arithmetic` as
`outward-binary64-sum-of-stored-nonnegative-bounds`. This covers only summation
of the supplied component certificates, not arithmetic in their construction.
Requested energy bounds and supplied matrix bounds must be finite and
nonnegative. Missing energy bounds, invalid bounds and sum overflow refuse
qualification, return no composed bound and suppress the relaxation delimiter.
Invalid energy values are represented by `null` in the failed ledger so it
remains serializable with strict JSON. No scientific target is relaxed.
The ledger refuses (`qualified` false with `reasons`) when a requested
certificate is unqualified, when two certificates claim one component, or
when the one-electron and nuclear certificates were evaluated over different
canonical density envelopes. Its `excludes` list is the union of the
certificates' exclusions, with a component another certificate claims
removed, plus density relaxation, k-mesh error, basis incompleteness,
integral roundoff, overlap-rank stability, Madelung numerics, XC and the
omitted physical Hartree domains for these frozen profiles, whose fitting
certificate covers only the coupling action relative to its declared finite
source. The separately supported full-Coulomb RI-BUPO certificate can replace
that coupling contribution when its requested targets, selected Gamma source
and canonical density identity qualify; it is never added to an overlapping
coupling bound. That extension does not change the frozen profile inputs. The sum is a fixed-density
component bound over the claimed components, not a relaxed total-energy
tolerance.

`relaxation_delimiter` delimits, and does not bound, the scale of density
relaxation for closed-shell results: from the certificates' maximum-entry
Fock error bounds and the converged canonical spectrum it reports the
uncoupled closed-shell second-order estimate `2 ||dF_mo||_F^2 / gap` with
`||dF_mo||_F <= n_basis * (sum of entry bounds) / s_min`, `s_min` the smallest
retained overlap eigenvalue and `gap` the smallest occupied-virtual gap over
the mesh. It is marked `is_certificate: false` and `applies` only with a
positive gap at every k point and `||dF_mo||_F < gap/2`; overlap perturbations
are excluded because they change the metric, and the coupled orbital
response is not included (it needs the stability eigenvalue, #625). On the
`compact-tight` profile the delimiter is far below the component sum; on the
ordinary profiles with loose targets it does not apply. Neither number is a
relaxed total-energy tolerance.

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

The [candidate inclusion check](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md#candidate-inclusion-check)
records that candidate `2b2ed883` includes the converter repair but still
lacks the accepted work allowance, available on main as `ebbf5957`.
Its numerical sources match the qualified integration. Include that allowance
before treating the candidate's `image-tight` example as qualified, then
reconcile the final source and native receipt while reusing the accepted
calculations for unchanged sources and inputs.

The [prepared-successor verdict](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md#prepared-successor-source-verdict)
closes the source linkage for `7151963`: it contains the accepted allowance
and matches all 2015 source/input hashes. The bounded scientific contribution
is accepted on the recorded local native build. Final candidate binary
identity and publication remain with the release owner; no new calculation
is required solely because the prepared source has a different commit ID.

The subsequent [selected-candidate linkage](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md#selected-candidate-linkage)
confirms both repairs in `5dfc97f0`. Its three changed source pins concern
the ECP UKS derivative state, outside these all-electron RHF/PBE0 energy
profiles. Their accepted scientific evidence remains applicable. The
benchmark distinguishes the actual local calculation core, the completed
predecessor's Linux core and the selected candidate's pending binary receipt;
none is treated as a deployed runtime. The bounded MANTA source contribution
is complete, while final candidate qualification stays with the release owner.
