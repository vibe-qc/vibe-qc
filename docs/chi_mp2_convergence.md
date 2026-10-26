# Classic chi RHF and MP2 convergence study

This is subsequent experimental chi development, outside the QC 0.18 and
Sketch 0.6 release gates. It does not extend CCSD, molecular DLPNO or D93
qualification. An accepted finite retained Hamiltonian and a physically
converged periodic calculation are different claims.

## Frozen systems and gates

Use the accepted #591 H2 and LiH archives. H2 has lattice rows
`(8,.3,.1)`, `(0,8.5,.2)`, `(0,0,9)` and H positions `(1,1,1)`, `(1,1,2.4)`.
LiH has a 6-bohr cubic lattice and Li/H at `(3,3,1.5)` / `(3,3,4.5)`.
All coordinates are bohr. Both systems are neutral, all-electron restricted
singlets, initially STO-3G with def2-universal-jkfit. No ECP is introduced.

The archived H2 controls are one-electron/pair radii 6.2/6.2 bohr, reciprocal
cutoff 18 Ha, nuclear real radius 8 bohr, reciprocal radius 5 inverse bohr and
Ewald alpha .6 inverse bohr. LiH differs in one-electron radius 12 bohr and
reciprocal cutoff 10 Ha. Metric threshold is 1e-9. Screening remains zero;
physical-pair support and the SCF Fock including Ewald exchange zero mode are
unchanged. LiH active/frozen partitions zero and one are independent.

Targets are fixed before collection:

- Existing accepted-reference stationarity, metric and occupation checks stay
  mandatory. Request SCF energy 1e-12 Ha and gradient 1e-10; inspect the actual
  final-rebuild residuals, overlap spectrum, retained fit spectrum and gap.
- Full-domain/zero-PNO raw local minus same-source canonical MP2 must be at most
  2e-9 Ha/cell, with residual at most 1e-11 and energy change at most 1e-13.
- An individual support plateau requires both last two successive changes
  below 1e-6 Ha/cell in HF and canonical MP2 at fixed other controls. This is
  an observed plateau, not a certified error bound or joint convergence.
- A local approximation target is 1e-5 Ha/cell raw error against the matching
  canonical baseline. Report PNO estimates separately; they do not satisfy
  the raw gate. Domain corrections are not computed.
- Independent-reference agreement at 1e-6 Ha/cell is conditional on matched
  and sufficiently converged approximations. Record unmatched terms instead
  of increasing the tolerance. Basis and mesh sensitivity alone cannot
  establish the complete-basis or thermodynamic limits.

## Reproduction and interpretation

[The payload](../examples/periodic/chi_mp2_convergence.py) consumes an accepted
HF archive and a JSON list of named cases. Each `changes` mapping contains at
most one control, `mesh`, `basis` or `aux_basis`. Empty mappings reuse the
archive. Numerical outputs and new archives must be outside the checkout.
Run the payload through vq with an identified Python/native runtime and its
required native SHA-256. Record the complete package manifest as well as
payload, archive and input hashes. A later executable does not inherit an
older runtime's scientific verdict.

One-electron AO tails, AO-pair fitting tails, fitted reciprocal resolution,
nuclear real/reciprocal sums, auxiliary basis and metric threshold are separate
axes. Classic fitting has no SR auxiliary-radius axis. Cell geometry is fixed
in this increment; increasing character counts in one direction probes that
finite mesh, not three-dimensional convergence or a molecular dilution limit.
Orbital-basis changes require their own support convergence. Failed
stationarity, occupation or metric checks are preserved, not repaired with
clipping, extra damping or relaxed admission.

The formulation follows Nejad et al., DOI 10.1063/5.0290816, Sec. II Eqs.
(1)-(8): BvK orbitals, unitary character transform, HF equations and MP2 energy
per primitive cell. Sun et al., DOI 10.1063/1.4998644, Sec. II and III, explains
periodic Coulomb fitting and the independent linear-dependence/reciprocal axes.
The classic finite reciprocal source here is not identical to compensated
GDF/MDF, and no equality is inferred from a common auxiliary-basis label.
CRYSTAL23 manual pages 120, 124 and 130 separately define local MP2, SHRINK and
TOLINTEG. Those thresholds do not equal chi physical radii. ORCA molecular
MP2 is not a direct reference for these finite periodic cells. A matched
periodic canonical implementation retaining SCF Ewald orbital energies is a
more direct independent MP2 control; differences in its fitting and supports
must still be quantified.

## Measured bounded increment

The qualification matrix attached to #572 and #591 records all inputs, basis
shells, archive and factor identities, measured conditioning, errors and refusals.
Raw logs, archives and failed carrier outputs are retained outside the product
checkout. There are 31 canonical cases: 29 completed and two pre-integration
memory refusals. Seven of ten new prepared local solves completed; three were
refused before iteration. Process exit zero is not a per-case PASS.

The canonical and new prepared-local runs used source
`ba3551069827ccac7afeb9b319e44e9a74aa3743`, native SHA-256
`d34edec5f0108f4bb8fbb15cff7f369c6d6fd1b6db3d1dc7a74a4e4ea3e641df`.
The classic HF/canonical/support Python code and relevant native scientific
kernels are unchanged through the previously accepted `de9a5d74` snapshot.
This bridge does not qualify later local factor, selection or transport changes.
The separate #591 note 52727 / #585 note 52751 bounded-backend PASS remains
bound to `de9a5d74` and native
`3f52e7e9e8b0d38783cd113b7132add6dae10d2329b441ddf60b2a723c9a9011`.
The #580 compact source-integrity evidence is reused within its own pin.
No changed scientific implementation is qualified with an older executable.

All completed chi SCF cases satisfy their original admission checks. Among
the initial ladders, maximum orbital residuals are 1.25e-11 (H2) and 2.10e-11
(LiH); minimum overlap eigenvalues are 0.0964 and 0.233, and minimum gaps are
0.857 and 1.442 Ha. Restored archives independently rebuild stationarity with
orbital/energy tolerances 1e-9/1e-11; these are not the requested SCF stopping
tolerances. Accepted reloads renew admission/process metadata and therefore
state identities; retained factor identities and canonical energies agree.
LiH's smallest retained metric eigenvalue is 1.03e-9. Its fit ranks change
from `[67,67,68,67]` to `[67,68,68,68]` on reciprocal refinement; that change
is an explicit approximation effect, not hidden by a fixed-rank claim.

### Physical-support results

All energies below are Ha per primitive cell. Each row changes only the named
axis from the frozen baseline, not from the preceding row of another axis.
The last-two-delta gate applies to both HF and each frozen-partition MP2 energy.

| System and axis | Tested values | Largest of last two HF / MP2 changes (Ha) | Conditional verdict |
|---|---|---|---|
| H2 one-electron radius (bohr) | 6.2, 10, 14, 18, 22 | 7.72e-8 / 4.59e-10 | PASS plateau |
| H2 AO-pair radius (bohr) | 6.2, 8, 10, 12, 14, 16 | 2.24e-7 / 5.48e-9 | PASS plateau |
| H2 reciprocal cutoff (Ha) | 18, 24, 30, 40, 50 | 1.94e-6 / 1.30e-6 | FAIL plateau |
| H2 nuclear real radius (bohr) | 8, 12, 16 | 4.61e-7 / 5.95e-9 | PASS plateau |
| H2 nuclear reciprocal radius (inverse bohr) | 5, 7, 9 | 2.63e-9 / 1.44e-12 | PASS plateau |
| H2 metric threshold | 1e-9, 1e-10, 1e-11 | 4.44e-16 / 1.22e-17 | PASS plateau |
| LiH one-electron radius (bohr) | 12, 16, 20; 24 refused | 0.3664 / 0.004206 | FAIL plateau |
| LiH reciprocal cutoff (Ha) | 10, 20, 40 | 0.07258 / 0.0002231 | FAIL plateau |

H2 baseline HF/MP2 is -1.132616778663140 / -0.012268098263550.
At one-electron radius 22 it is -1.137755593593475 / -0.012170040209412;
at pair radius 16, -1.132417719904343 / -0.012280774679151;
at reciprocal cutoff 50, -1.132564398130348 / -0.012288132492225.
These different conditional limits must not be combined into a claimed
jointly converged energy without new calculations.

LiH baseline HF is -9.224544983357706, with MP2 frozen 0/1
-0.006881336742081 / -0.005784454860660. At one-electron radius 20 these
become -9.619161072997088 and -0.006075159550529 / -0.001377924462656.
At reciprocal cutoff 40 they are -9.128699126182655 and
-0.006985492143957 / -0.005479099158137. This is a substantial physical-support
failure, despite positive overlap, a gap and stationary SCF. It is not yet a
localized implementation error.

### Separate approximation and reference limits

H2 character meshes `(1,1,1)`, `(2,1,1)`, the reused accepted `(3,1,1)`, and
`(4,1,1)` give HF -1.132616779, -1.125866149, -1.123515590, -1.122340390;
MP2 -0.0122680983, -0.0122193617, -0.0122416009, -0.0122530190.
This is UNQUALIFIED finite-size sensitivity, not a three-dimensional k limit.
Changing only the auxiliary basis to cc-pVDZ-JKFIT gives H2 HF/MP2
-1.132598555239006 / -0.012277079469607. Changing only the orbital basis to
6-31G gives -1.144261966478398 / -0.016240904957639. Neither establishes
auxiliary- or orbital-basis convergence. H2 cc-pVDZ and LiH radius 24 were
resource refusals at the original 2/4 GiB numerical caps; their bounded replay
is recorded below and does not qualify either convergence limit.

PySCF 2.6.2 periodic GDF KRHF/KMP2 uses identical cells, STO-3G,
def2-universal-jkfit, charge/spin and character meshes, with `exxdiv='ewald'`.
Its KMP2 kernel consumes the SCF orbital energies unchanged, checked explicitly;
no exchange-zero-mode subtraction is used. H2 Gamma precision 1e-8, 1e-10,
1e-12 gives a stable reference (HF last changes 2.78e-11 and 4.45e-12 Ha).
At 1e-12 its HF/MP2 is -1.137506045512623 / -0.012200632617494.
The archived chi differences are +0.004889266849483 / -0.000067465646056.
LiH `(4,1,1)` at reference precision 1e-10 gives HF -7.984593087472446 and
MP2 frozen 0/1 -0.021004129636132 / -0.020096649324724. Chi differences are
-1.239951895885260 and +0.014122792894051 / +0.014312194464064.
These exceed the desired comparison target. Physical parity is UNQUALIFIED:
classic finite reciprocal fitting and compensated GDF/supports are unmatched,
and chi support ladders have not converged. A shared basis name is insufficient.

CRYSTAL23 public 1.0.1 was also run for the same H2 geometry, STO-3G, P1,
`SHRINK 1 1`, `TOLDEE 12`. `TOLINTEG` ladders `(8,8,8,8,16)`,
`(10,10,10,10,20)`, `(12,12,12,12,24)` produced HF -3.1711402959794,
-3.4931705278661, -3.8631727592575. SCF converged, but this reference itself
fails a threshold plateau and is UNQUALIFIED as a finite-BvK comparison.
The drift is preserved, without attributing an unproven cause to either code.
The periodic-reference conventions need reconciliation before a parity claim.

### Local approximation and remaining implementation

New full-space prepared local solves agree with same-source canonical MP2 for
H2 one-electron radius 18, pair radius 12, reciprocal cutoff 30 and 6-31G,
and for LiH radius 20 with frozen core 1. Maximum raw error is 2.57e-15 Ha,
residual 2.09e-13 and absolute energy change 2.21e-14. PNO estimates remain
separate (zero for these H2 cases; 1.76e-18 Ha for LiH). H2 6-31G cutoffs
`TCutMKN=.01` and `TCutPNO=1e-6` do not change the energy; this does not
establish a useful truncation error bound.

The earlier containing-runtime LiH data remain the nontrivial truncation
control. At domain .01, raw errors for frozen 0/1 are 38.844/10.548 microhartree;
at PNO 1e-6 they are 46.916/11.171 microhartree, with separate estimates
-39.220/-10.155 microhartree. All four raw errors FAIL the 10-microhartree target.
Corrections do not turn those raw tests into PASS. Domain corrections are absent.

The new LiH radius-20 all-electron local solve refuses the older AO seed frame;
replay with the delivered explicit cosine option is still needed for this new
source. At reciprocal cutoff 40, both partitions refuse the retained-factor
recovery audit: round-trip defect 2.53e-10 (imaginary 6.42e-11). No tolerance
was changed. Replay and diagnosis on the current containing factor runtime
are required before calling this a current implementation defect.

No scientific equation repair is justified by this bounded study. It identified
the one-electron nuclear Fourier cache as the next workspace barrier: its
`32*(oneel_cells+1)*nao**2*nG` reserve refused the larger LiH support and H2 basis
rungs before integration. The bounded implementation below removes that
intermediate without establishing physical convergence. Next finish reciprocal and
joint-support ladders, new-source local admission, matched references and
multidimensional mesh/basis convergence. Production chi and D93 fitted/post-HF
remain not-qualified; #572/#591 stay open and #586 stays outside this increment.

### Bounded nuclear Fourier workspace

The classic chi nuclear attraction now uses the existing native weighted
per-cell AO-pair Fourier contraction, with one cell and budget-derived
reciprocal tiles (at most 256 vectors). It evaluates the complete finite
sum `Re(sum_G w(G) conj(FT_mn(G,R)))` before the original real assembly,
Y_lm correction, physical-pair mask and background overlap term. The native
kernel contracts within shell-pair Hermite tasks. Reciprocal-tail truncation,
shell screening and optional radial/power caches are disabled. No physical
radius, reciprocal vector, nuclear source, Ewald convention or precision gate
changes. The shared complete-cache function remains the explicit small-system
oracle and the default for other consumers.

Preflight now reserves complete lattice outputs, block snapshots, Bloch
matrices, support records and nuclear supports, plus the tiled phases,
weights, returned contraction block and assembly scratch. Native Hermite
powers, per-worker recurrences and accumulators are reserved under the
existing native-workspace cap. An impossible one-vector minimum refuses
before nuclear integration. The removed intermediate scaled as
`32*(oneel_cells+1)*nao**2*nG`; the reciprocal contraction workspace now scales
with the selected tile and angular momentum, without an AO-pair Fourier
cache. Complete lattice operators and reciprocal-vector/support lists remain
resident. The numerical cap excludes allocator overhead, Python object
metadata, external library state and downstream arrays; it is not a total
process-memory guarantee.

The returned reciprocal receipt and standard output report tile dimensions,
accounted peak, native reserve, removed-cache estimate and actual native
call/vector counts. These execution details do not enter the immutable
physical-support record. Tile accumulation changes floating-point association;
complete matrix and downstream checks retain the original tolerances.

Focused source checks pass 82 tests across `test_chi_fitted_controls.py`,
`test_periodic_v_ne_ewald_ft.py` and `test_chi_ecp_hamiltonian.py`. They compare
complete matrices and complex Bloch sums on a skew cell for STO-3G, 6-31G and
cc-pVDZ at tile widths 1, 7 and the planned width, with a 3e-12 absolute gate.
Gamma and three-character HF/canonical MP2 agree with the full-cache oracle
within the unchanged 1e-11/2e-12 energy gates. Support records are identical;
guards prohibit the full-cache builder, and exact/below-minimum budgets are
checked. Shared weighted-kernel and ECP assembly checks also pass.

For the archived refused inputs, the conservative preflight at the measured
18-worker setting gives the following numerical payloads (bytes):

| Input | Original cap | Removed Fourier reserve | New accounted peak | Reciprocal tile |
| --- | ---: | ---: | ---: | ---: |
| H2 cc-pVDZ | 2147483648 | 2213971200 | 613361664 | 256 |
| LiH one-electron radius 24 | 4294967296 | 5563014912 | 626432512 | 256 |

These are allocation bounds, not measured resident peaks or scientific
acceptance. Each cell/tile call repeats primitive geometry and Hermite setup;
no runtime speedup is claimed. Larger complete outputs, support records and
later fitting/correlation arrays can still impose independent limits.

The exact formerly refused manifests now complete through vq on the verified
containing snapshot, at their original 2/4 GiB numerical caps. Source/native
pins are checked before and after execution. The native SHA-256 is
`e1a582efa866faf0f5a4fd356a0a0b5b4fa12f85bdb8b2b20ef00efa72e22c7b`;
the containing source is `c2fa5152`. Two initial checkout-bound launches
refused before chemistry when an unrelated source file advanced; those
operational failures are preserved separately from the successful snapshot
runs. The independently owned H2 combined-support job remains untouched.

| Replay | HF (Ha/cell) | Canonical MP2 (Ha/cell) | Orbital residual |
| --- | ---: | ---: | ---: |
| H2 cc-pVDZ, frozen 0 | -1.145851721969042 | -0.024550872269520 | 2.94e-11 |
| LiH radius 24, frozen 0 | -9.620470608358930 | -0.006072028898480 | 1.64e-11 |
| LiH radius 24, frozen 1 | -9.620470608358930 | -0.001367774907557 | 1.64e-11 |

Both sources satisfy the original HF admission, with minimum overlap
eigenvalues 0.043747/0.256404 and gaps 0.812453/1.715460 Ha (H2/LiH).
The complete commands took 1.90/2.38 seconds; POSIX `wait4` measured peak RSS
208.58/211.19 MiB, the largest single process, not a sum across processes.
These are new-path observations, not matched speedup measurements against the
formerly refused full cache. Calculation-only times were 0.22/0.72 seconds.

LiH radius 20 to 24 changes HF by -1.309535e-3 Ha and MP2 frozen 0/1 by
+3.130652e-6/+1.014956e-5 Ha. All exceed the predeclared 1e-6 one-step target,
so the required two-step support plateau still FAILS. H2 cc-pVDZ is one
orbital-basis sensitivity point, not basis convergence. These exact manifests
contain canonical calculations only; the subsequent full-space local replay
is recorded below. Nonzero domain/PNO accuracy and matched independent-reference
limits remain UNQUALIFIED. No
correction or relaxed gate is used. The next chi-specific gap is physical
support convergence and nonzero local approximation accuracy, not another V_ne full-cache
workspace repair. Joint-support, mesh and reference limits above remain open;
chi is experimental and D93 fitted/post-HF remains not-qualified.

### Local admission on the newly admitted sources

The accepted H2/cc-pVDZ Gamma and LiH radius-24 `(4,1,1)` archives were reused
without rerunning SCF, refitting factors or changing supports. Six full-domain/zero-PNO solves
pass: bounded and prepared transport for H2 frozen 0, and for LiH frozen 0
and 1 independently. H2 uses the AO seed frame; LiH explicitly uses the
existing cosine frame. Streamed factors and compact preparation are retained,
with selection/factor/transport caps 64/64/256 MiB, 500 iterations maximum,
energy/residual stopping tolerances 1e-13/1e-11 and damping 1. No admission
threshold or canonical comparison gate changes.

| Source / frozen core | Raw local MP2 (Ha/cell) | Absolute canonical error | Residual | Separate PNO estimate (Ha/cell) |
| --- | ---: | ---: | ---: | ---: |
| H2 cc-pVDZ / 0 | -0.024550872269520 | 3.47e-18 | 1.39e-17 | 1.04e-17 |
| LiH radius 24 / 0 | -0.006072028898473 | 7.55e-15 | 3.61e-13 | 2.48e-18 |
| LiH radius 24 / 1 | -0.001367774907554 | 2.71e-15 | 2.19e-13 | 7.05e-19 |

The two backends give identical raw energies, residuals, energy changes and
PNO estimates, plus identical solver input snapshots, occupied selections,
domains and PAO/PNO ranks. Iteration counts are 2, 13 and 11 respectively;
maximum absolute energy change is 4.77e-14 Ha. H2 retains PNO rank 9; every LiH
pair retains PNO rank 16. Complete factor recovery passes the original 1e-10
gate (largest defect 9.79e-12); no component or check is screened. Canonical
energies reproduce the preceding accepted values within 2e-12 Ha. Corrections
are reported separately and do not enter any raw gate.

The calculations use the same immutable `c2fa5152` source/native snapshot
identified above, with hashes checked before and after execution. Relevant
chi/local/native code is unchanged through this evidence landing. The issue
matrix records exact archive/input/output hashes, source identities, controls
and all six results; raw output and reproducible payloads are retained.
Nejad Sec. V Eqs. (42)-(43) supplies the coupled residual and per-cell energy
conventions, with Riplinger and Neese Sec. II B3/C supplying PNO construction
and two-sided pair transport. This is a new-source replay of the delivered
method, not a new equation set or independent external-reference result.

No bounded implementation defect was demonstrated, so no method fix or native
rebuild was warranted. Accepted scope is full-space local admission for these
finite Hamiltonians only. LiH physical-support convergence still fails, H2
basis/mesh convergence remains unqualified, and nonzero domain/PNO errors on
these sources were unmeasured at that checkpoint. The bounded follow-up below
quantifies selected local truncation errors while keeping physical-support
and independent-reference work separate.

### LiH support extension and local truncation probes

The next bounded study starts from the accepted radius-24 LiH archive and
changes only the one-electron radius. Geometry, STO-3G/def2-universal-jkfit,
`(4,1,1)` mesh, reciprocal cutoff 10 Ha, nuclear/Ewald controls, metric cutoff,
SCF Fock and independent frozen 0/1 partitions stay fixed. The declared
sequence was 28, 32, 36, 40 bohr, stopping at the first two consecutive changes
below 1e-6 Ha/cell for HF and both MP2 partitions, a refusal, or radius 40.
Physical envelope estimates were 587/847/1189/1551 cells and numerical reserves
618/618/623/630 MB at two workers, within the original 4 GiB cap. No cap changed.

| One-electron step at reciprocal 10 Ha | HF change | MP2 frozen 0 change | MP2 frozen 1 change |
| --- | ---: | ---: | ---: |
| 24 to 28 | -1.348349e-4 | +1.489560e-7 | +1.129148e-6 |
| 28 to 32 | -6.254695e-7 | +1.471430e-9 | +6.390402e-9 |
| 32 to 36 | -8.367252e-9 | +7.403091e-12 | +8.872868e-11 |

All changes are Ha/cell. The last two steps pass the unchanged gate: a
**conditional one-electron plateau**, not joint convergence. Radius 40 at
reciprocal 10 was correctly not run after the stopping rule was satisfied.
The radius-36 energies are HF -9.620606077109800 and MP2 frozen 0/1
-0.006071878463645 / -0.001366639280146 Ha/cell.

That exact accepted radius-36/reciprocal-10 archive became the parent for
separately declared reciprocal probes 20, 40, 60 Ha. Only reciprocal support
changed, with the same stopping rule and an upper bound of 60 Ha.

| Reciprocal step at one-electron radius 36 | HF change | MP2 frozen 0 change | MP2 frozen 1 change |
| --- | ---: | ---: | ---: |
| 10 to 20 | +7.218317e-2 | -8.899838e-5 | +2.259265e-4 |
| 20 to 40 | +2.318867e-2 | -3.298356e-5 | +3.631151e-5 |
| 40 to 60 | +4.258098e-3 | -5.879945e-6 | +5.566866e-6 |

The reciprocal gate **FAILS through 60 Ha**. At the accepted reciprocal-60
parent, one-electron radii 40 and 44 were then tested independently. The HF
changes are -3.229417e-12 and zero at reported precision; maximum MP2 changes
are 2.55e-14 and 2.13e-16 Ha/cell. This confirms a conditional one-electron
plateau at the combined point. It does not repair the failed reciprocal gate,
so **joint convergence is not established**. The radius-44/reciprocal-60
energies are HF -9.520976139476302 and MP2 frozen 0/1
-0.006199740344550 / -0.001098834366835 Ha/cell.

All eight new HF sources and sixteen independent canonical MP2 evaluations
complete without a numerical-cap refusal. Maximum HF orbital residual is
1.65e-11 and minimum overlap eigenvalue is 0.2564029. Full-space local controls
pass for seven of the eight sources: 28 prepared/bounded solves satisfy the
original raw/residual/energy-change gates, with maxima 7.03e-14 Ha/cell,
2.29e-12 and 9.33e-14 Ha respectively. Compatible-gauge amplitudes and residual
arrays agree exactly between backends. All LiH controls use the explicit
cosine frame; no frame fallback occurs.

The radius-36/reciprocal-40 source passes HF/canonical admission but both frozen
partitions refuse local factor recovery in both transport backends. The
retained factors violate the character-conjugate reality relation by
4.046396e-10, implying a minimum real-projection loss of 2.023198e-10, above
the original 1e-10 gate. The worst retained component has magnitude 8.60e-11;
it must not be screened or dropped. The dense factor oracle also refuses.
An independent long-double complete Fourier calculation gives complex
round-trip error 5.61e-16 but real/symmetric recovery error 2.023198e-10.
Thus this is an incompatibility of the retained source with current local
real-representation admission, not a tiled accumulation or transport defect.
The 8 MiB and 64 MiB factor budgets reproduce the refusal. A separate 1 MiB
probe refuses its 1049856-byte minimum before audit; that resource refusal is
not a scientific failure. No source was projected, metric clipped or gate
loosened. Canonical validity does not establish local representability.

The prior H2/cc-pVDZ and LiH radius-24 full-space controls were reused, including
byte-identical amplitude/residual hashes in identical gauges. Only then were
independent domain and PNO probes run on those original sources. All twelve
truncated solves converge at the original residual/energy-change gates.
H2 retains its complete rank-9 space at every tested cutoff, so its four raw
errors of 3.47e-18 Ha/cell do not demonstrate a nontrivial truncation bound.
LiH results below are microhartree/cell; the raw target is 10 microhartree.

| LiH radius-24 approximation | Raw error frozen 0 | Raw error frozen 1 | Separate PNO estimate frozen 0 / 1 |
| --- | ---: | ---: | ---: |
| Domain .01, PNO zero | 8.6674 | 0.12135 | below 3e-12 / below 3e-12 |
| Domain .001, PNO zero | 4.8999 | below 3e-9 | below 3e-12 / below 3e-12 |
| Full domain, PNO 1e-6 | 25.5163 (FAIL) | 9.8546 | -17.6928 / -7.7243 |
| Full domain, PNO 1e-7 | 10.3898 (FAIL) | 1.6048 | -4.3909 / -1.2762 |

Corrections do not rescue a failed raw gate. These source-specific tests do
not establish production cutoffs or transfer their errors to the new support
points. No bounded method fix was demonstrated. The next implementation gap
is diagnosing conjugate-character fidelity in classic retained-factor
construction near small components, while preserving every retained mode and
support identity. Reciprocal convergence beyond 60 Ha, repeated joint checks,
matched independent references, auxiliary/basis/mesh limits and local admission
of the refused source remain missing. Existing references have unmatched
fitting/support limits and add no new external accuracy PASS here.

All stages ran through vq on immutable `c2fa5152` with the native SHA-256 above,
verified before and after execution. Exact accepted-parent archives, controls,
source identities, deltas, raw refusals and compatible-gauge comparisons are
in the issue qualification matrix. Primary conventions remain Nejad Sec. V
Eqs. (42)-(43), Riplinger and Neese Sec. II B3/C, and Sun Sec. II-III; no
scientific formulation changed. The separately owned H2 combined-support
campaign was untouched. Chi remains experimental and D93 fitted/post-HF
not-qualified; neither broad issue is closed.
