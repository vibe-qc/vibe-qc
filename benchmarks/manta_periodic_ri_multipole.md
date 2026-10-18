# Ordered periodic RI multipoles

This experimental shell-leaf approximation operates inside the existing
ordered GDF producer. The exact finite metric, reciprocal contribution,
finite-zero convention and paired forward/adjoint maps retain their owners.
Native Gaussian envelopes qualify individual erfc Taylor replacements and
direct MD evaluates the remainder. The source counts its actual finite
sphere/capsule support before apportioning per-image error. Executed error
bounds are summed within each tile and propagated through the unchanged
finite RI factor. No canonical-image substitution or new symmetry authority
is introduced.

## Scientific acceptance

An actual local `vq` calculation at source `f823d61431f323e16a469446721c2733f8d934ff` and
native SHA256 `b6ba38bbd057398f407f13c1c0cc9679cdf847f692ce3d60195392f0204f6a55` passed.
Before/after pins cover 1599 source/reference files. The implementing
runner submitted, fetched and separately assessed the results; job/host and
operational evidence are stored outside the product checkout. The runtime
includes the synchronized 0.17.18 main changes.

Seven complete public states exercise nonzero far replacement: shifted Be
RHF, H UHF with empty beta, Li UHF converged/capped, Be PBE RKS, and Li PBE
UKS converged/capped. They execute 36 to 156 far-image replacements per
evaluated paired Fock build. Four arbitrary complex action cases cover
Gamma/nonzero q and orders zero/two. The original metric, finite factor and
retained projector are unchanged. Public states retain shared Fock actions,
method-owned exchange/XC, evaluated energy and exact QVF spin-density agreement.
Capped states remain nonconverged and unconfirmed.

Exact finite references use independent Gaussian product/Fourier sums and
libint erfc integrals. NumPy factors and independent SciPy SCF validate the
paired contraction and complete method state using the separately exported
approximate coupling. Its difference from the exact finite coupling is
checked against the native envelope; this does not independently implement
the approximate partition. KS uses independent grid contractions with the
same libxc functional. A separate NumPy-only assessment checks fetched arrays
and 63 managed output hashes. No matched ORCA/CRYSTAL reference is
available for these custom finite-domain fixtures.

| Evaluated-state quantity | Maximum absolute difference | Gate |
| --- | ---: | ---: |
| Fock entries versus reference contraction | 5.051515e-15 | 8e-11 |
| Energy / Eh | 1.190159e-13 | 8e-11 |
| Independently relaxed energy / Eh | 1.119105e-13 | 8e-11 |
| Density entries | 1.518352e-09 | 5e-9 |
| Raw forward action | 8.881784e-16 | 4e-12 absolute + 3e-13 relative |
| Raw adjoint action | 2.220446e-16 | 4e-12 absolute + 3e-13 relative |
| Complex adjoint pairing | 4.440892e-16 | 4e-12 |

At the same evaluated density, the largest change from exact finite RI is
4.339208e-08 in Fock entries and
4.807159e-07 Eh in energy; both lie inside their propagated
bounds. Those bounds are very loose and do not establish whole-method accuracy.

## Focused checks and integration

There are 470 distinct passing checks: 48 new erfc/periodic
multipole cases, 166 existing finite-Ewald cases (one unavailable optional
reference skipped), 104 existing screened/raw/tiled GDF cases, 151 shared
range/action/paired/binding/changelog cases, and one exact MANTA-to-CCM Fock
comparison after synchronization. They cover signed Cartesian and pure
shells, angular quadrature, small erfc values without gauge cancellation,
nonharmonic trace terms, duality, image gauges, reciprocal relabelling,
rank-zero, zero density, weighted energy derivatives and every resource cap.

The new neutral CCM fit exporter rejects J-only approximation controls before
exporting exchange/correlated fits. Exact export remains unchanged. An output
inspection also found a citation namespace mismatch; a numerics route and
focused HF/KS output checks fix attribution without altering accepted science.
Source privacy, syntax, diff and documentation checks pass. No ordinary CI or
separate review chain is required.

## Cost and remaining work

`python benchmarks/manta_periodic_ri_multipole.py` measures three isolated
processes with three warm calls per setting and one numerical thread. The
compact fixture has two k points, two AOs, six auxiliary functions and rank
six. Every build uses two coupling passes. Source/native identities and all
samples are recorded. Shared-host timing is not a scaling claim.

| Requested epsilon | Order | Median / ms | Shell calls | Far images | Measured J error | J bound | Measured energy error / Eh | Energy bound / Eh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 21.162 | 19,500 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1 | 0 | 86.908 | 6,184 | 13,716 | 1.786e-06 | 4.715e+02 | 1.128e-06 | 4.985e+01 |
| 100 | 2 | 108.183 | 6,252 | 13,656 | 3.684e-08 | 1.739e+05 | 1.745e-08 | 3.664e+05 |
| 100 | 4 | 215.210 | 6,676 | 13,218 | 2.050e-12 | 2.483e+05 | 1.504e-12 | 5.452e+05 |

Modeled peak storage ranges from 25,478,848 to
26,115,736 bytes. Far replacement reduces direct
integral calls but the leaf traversal is roughly four to ten times slower
than exact evaluation on this small fixture. It recomputes extents, moments
and error queries during both passes and does not aggregate shell leaves.
The derivative envelope and global factor norm amplify the error bounds;
an order increase need not improve admission. These results do not support
a production speed or automatic-accuracy claim.

Hierarchy construction, reusable upward/downward moments, tighter error
allocation and large-cell scaling remain required for periodic RI-BUPO.
Bounds exclude finite-domain, RI-basis, floating-point and density-relaxation
errors. The full [0.18.x capability matrix](../docs/design/manta_validation_plan.md)
and persistent MANTA implementation goal remain open.
