# Periodic BUPO Hartree-Fock benchmark

This increment connects native matched BUPO electrostatics to bounded RHF
and UHF, with native kinetic and explicitly finite reciprocal exchange.
It checks complete SCF energies and Fock states against independent Gaussian
and reciprocal-integral references. Electrostatic targets are enforced for
every evaluated density; they are not whole-HF error bounds.

## Reproduction and timing

Run `.venv/bin/python benchmarks/manta_periodic_bupo_hf.py --repeats 5`.
Three isolated Darwin arm64 workers use Python 3.14.8, NumPy 2.5.3 and one
numerical thread each. Each case receives a warm-up and five repetitions.
Times are medians. Complete SCF includes matched nuclear setup and terminal
confirmation; initial source construction is reported separately. The
numerical memory model excludes interpreter/process overhead. No scaling or
speedup claim follows from these small cases.

| Case | AO/k/spin | Iterations/builds | Source setup ms | SCF ms | Model MB | Process RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| rhf-gamma | 2/1/1 | 6/7 | 2.480 | 51.704 | 171.64 | 169.70 |
| uhf-open-shifted | 3/2/2 | 12/13 | 2.483 | 314.600 | 172.44 | 168.72 |
| mixed-bvk-shifted | 4/2/1 | 8/9 | 10.146 | 279.284 | 179.79 | 169.08 |

All cells are cubic with side 12 bohr. The He RHF case uses two normalized
s primitives, exponents 4 and 1.2. The Li doublet UHF case uses 8, 4 and 1.2,
with two alpha and one beta electron. The mixed case has two He nuclei
separated by 3 bohr, each with exponents 3200 and 960. Its deliberately
compact basis is a finite-method probe, not a chemically useful energy.
Shifted cases use a (2,1,1) mesh with shift (1,0,0).

All use physical eta 0.2, real Ewald cube 4 and reciprocal cube 8. Exchange
uses reciprocal cube 5 plus each k transfer, excluding exact zero. The
original image cube is 1. Ordinary cases use pair cutoff 3 bohr and exact
BUPO fallback. The mixed case uses cutoff 0.2, order 8, bubble radius 0.16
and integral admission 1e-9, retaining 17 direct and four approximate pairs.
It includes the original-mesh full-Coulomb BvK exchange correction.

Maximum observed electrostatic energy/matrix bounds are respectively
2.20e-14/1.94e-14 for He, 5.28e-13/1.79e-13 for Li and
3.03e-9/2.58e-9 for mixed BUPO. All meet the unchanged 1e-8 targets.
These analytical bounds exclude roundoff, finite S/T and exchange domains,
basis incompleteness and error from the relaxed SCF state.

## Numerical qualification

The focused source checks cover RHF, paired UHF, both populated open-shell
UHF, empty beta, Gamma and shifted meshes, admitted near/far BUPO,
BvK correction, rank reduction, capped evaluated-state semantics and
individual basis-image covariance. Shared `FockBuild` exports retain the
complete method energy, spin order and unreduced physical frame. All five
exact resource caps pass and one-short limits refuse before exchange
payload scans. Charge and electrostatic target refusals are explicit.

Independent references use analytic s overlap/kinetic, Gaussian
Laplace/characteristic integrals for Ewald Coulomb and nuclear components,
ordered point Ewald sums, independent Gaussian Fourier quadrature for
finite exchange, and a separate generalized-eigenproblem fixed-point SCF.
They do not call native integral or SCF routines. Previously qualified
ordered-image and angular component evidence is reused for the underlying
source. Local queued acceptance measurements follow below.

No suitable matched ORCA/CRYSTAL finite-domain BUPO SCF profile is available
for these cases. This is independent validation of the declared finite
Hamiltonian, not external production parity. Automatic whole-method
profiles, public job routing, q-aware RI-BUPO, diffuse/angular SCF expansion,
source-qualified symmetry, gradients and stress remain open.


## Queued scientific acceptance

Sixteen method calculations were assessed: thirteen converged SCFs and
three deliberately capped evaluated states. They cover He RHF and paired
UHF, H empty-beta UHF, Li doublet UHF, Gamma/shifted meshes, mixed near/far
RHF/UHF with BvK correction, overlap rank reduction, complex basis-image
covariance and an eta refinement. The selected wrapper and shared Fock
export were exercised as part of the calculation path.

The first batch completed fifteen cases, then refused eta 0.3 with
reciprocal cube 8 because its conservative electrostatic bound exceeded
the unchanged 1e-8 target. Those completed outputs and per-case assertions
were retained. A second queued run reused them, enlarged only the eta case
to reciprocal cube 12, and passed that calculation and the aggregate checks.
No implementation fix or tolerance relaxation was needed. Detailed receipt
JSON from the interrupted batch was not recovered; its fetched density,
Fock, components and reference arrays were independently reassessed.

A separate NumPy-only assessment verified fetched hashes, reconstructed
total energies from weighted density/Fock traces, recomputed all reference
components and reconstructed the Fock from H, J, K and the declared
correction. Maximum absolute discrepancies were:

| Quantity | Maximum discrepancy |
|---|---:|
| Native total energy vs density/Fock trace | 3.55e-15 hartree |
| Native vs independent reference energy | 9.10e-13 hartree |
| Native vs independent reference Fock | 9.10e-13 hartree |
| Independently recomputed component energy | 8.88e-16 hartree |
| Evaluated vs independent SCF or capped density | 1.97e-10 |
| Basis-image density/Fock covariance | 4.45e-16 / 8.89e-16 |
| Eta 0.2 vs refined eta 0.3 energy | 1.12e-15 hartree |

The comparison allowance is 3e-11 hartree, with 3e-9 for densities and
2e-8 for centered weighted energy derivatives. These are numerical
comparison allowances, separate from analytical error bounds. Derivative
checks use two steps, 1e-4 and 5e-5, at fixed spin charges.

The original-image improvement leaves finite J and energy bitwise unchanged
for a cutoff source while reducing its omitted mass bound from 5.96e-5 to
2.94e-18 and its electronic energy error enclosure from 0.2567 to 1.27e-14
hartree. The undersized neutral source and an impossibly small declared
accuracy target both refuse explicitly.

The measured MANTA source is equivalent to commit `91d8ebe3` across 266
implementation/reference pins. Native SHA-256:
`58219d8072a611805b13d0811a80472f190e9a3e01307539e194ee2d2a40b324`.
Rebasing preserved the incoming BIPOLE subtree-screening change; that
separate path was not compiled into this measured runtime or accepted by
these calculations. The 22 new focused checks and 190 related regressions
pass, as do manifest, changelog, citation, Sphinx and privacy checks.
