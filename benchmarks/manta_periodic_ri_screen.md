# Fixed-metric periodic RI coupling screening

The bounded screening baseline reuses the existing GDF Gaussian-envelope
selector and producer-owned paired actions. It changes only short-range
three-center terms; the metric, reciprocal domains, finite-zero convention
and native factor remain fixed. It supports Coulomb J with independent exact
exchange, and pure KS. Fitted exchange rejects positive coupling-only
screening. It supplies no symmetry certificate or BIPOLE parity claim.

## Scientific acceptance

The implementing runner submitted, retrieved and assessed an actual local
`vq` calculation at source `9c396ff65590ca112b862dd4ce17319264339fe0` and
native SHA256 `dd6bc58536f42a7a3d7fd63133533629728008c86c1c14c8c432b9b2fbec9422`.
Before/after identity pins cover 1592 source/reference files.
Job, host and operational outputs are retained in the private evidence
record outside the product checkout. No separate reviewer or full CI was
required for this development increment.

Seven complete public states pass: shifted Be RHF, H UHF with empty beta,
Li UHF converged/capped, Be PBE RKS, and Li PBE UKS converged/capped. Four raw
complex-action cases cover Gamma/nonzero q and two screening settings. The
metric, native factor and retained projector are unchanged in paired
zero-screen controls. Public methods retain the shared Fock state, full
evaluated-density energies and exact QVF spin-density agreement. Capped
states remain nonconverged and unconfirmed.

References use libint erfc integrals, independent polynomial Gaussian Fourier
integration, NumPy retained factors and independent SciPy SCF. The screened
libint reference shares the existing physical image selector; unscreened
sources independently check the omission error. KS uses independent grid
contractions of the same libxc functional. A separate NumPy-only assessment
checks fetched arrays and 63 managed output hashes.
No matched ORCA/CRYSTAL reference is available for these custom finite domains.

| Evaluated-state quantity | Maximum absolute difference | Gate |
| --- | ---: | ---: |
| Fock entries | 5.095924e-14 | 8e-11 |
| Energy / Eh | 9.237056e-14 | 8e-11 |
| Independently relaxed energy / Eh | 9.592327e-14 | 8e-11 |
| Density entries | 2.790752e-10 | 5e-9 |
| Raw forward action | 2.842171e-13 | 4e-12 absolute + 3e-13 relative |
| Raw adjoint action | 4.582863e-13 | 4e-12 absolute + 3e-13 relative |
| Complex adjoint pairing | 5.084230e-13 | 4e-12 |

The KS calculations exercise nonzero omission: at the same evaluated
density, maximum Fock change from zero-screen RI is
1.232021e-07 and maximum energy change is
5.152294e-08 Eh. Both lie inside the propagated bounds.
The smaller HF fixture omits no resolvable contribution at the selected
screen, so its role is method/state compatibility rather than a savings claim.

## Focused checks

There are 291 distinct passing checks: 29 new screening checks, 25 existing
paired-RI checks, 34 producer-action checks, and 203 range-kernel, raw/tiled
fitting, selected-target, inventory and changelog checks. They include
complex indefinite spin densities, signed angular contractions, nonuniform
and zero weights, retained ranks zero through six, zero density, energy
derivatives, image transport, exact metric/factor identity, immutable source
controls, forbidden exchange/common-screen combinations and all five joint
resource caps. The initial test assumed every positive screen saves work;
a tight-screen case correctly omitted nothing. Correcting that work-count
expectation changed no scientific gate or numerical code. Source privacy,
syntax, diff and documentation fence/link checks also pass.

## Cost and conservatism

`python benchmarks/manta_periodic_ri_screen.py` runs three isolated processes
with three warm calls each and one numerical thread. The small fixture has
two k points, two AOs, six auxiliary functions and retained rank six. The
script records commit/native/source hashes, exact factor identity, samples,
actual calls and peak storage. Timing is on a shared host and is not a
scaling or general speedup claim.

| Coupling epsilon | Median / ms | Shell calls | Measured J error | J bound | Measured energy error / Eh | Energy bound / Eh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0e+00 | 16.075 | 19,500 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1e-06 | 14.345 | 17,772 | 5.058e-13 | 1.689e-02 | 4.496e-14 | 1.377e-05 |
| 1e-03 | 11.491 | 13,972 | 5.238e-10 | 1.689e+01 | 1.366e-10 | 2.124e-02 |
| 1e-01 | 8.701 | 10,492 | 9.596e-08 | 1.716e+03 | 3.156e-08 | 7.624e+01 |

Each build uses two coupling passes. Modeled peak storage remains
25,478,844 bytes. At epsilon 0.1 the selected
image work decreases from 19,500 to 10,492 calls, without a dense T/B tensor
or second metric. The bounds are **very conservative**: global candidate
censuses, uniform element bounds and the factor Frobenius norm amplify them.
They do not justify an automatic whole-method accuracy target. Tighter local
omission accounting and conditioning-aware propagation remain useful work.

Bounds compare the same evaluated density and finite retained metric.
Domain, RI-basis, density-relaxation and floating-point errors are separate.
This is an experimental bounded screening baseline, not hierarchical
RI-BUPO or a production accuracy profile. The full
[0.18.x capability matrix](../docs/design/manta_validation_plan.md) remains open.
