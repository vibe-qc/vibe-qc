# Cutoff-selected periodic BUPO HF

`manta_periodic_bupo_domains.py` measures count-only planning, reciprocal
domain generation (including opposite-support audits), and complete bounded
HF over an existing immutable BUPO source. Each case runs three times in a
separate process with one numerical thread. The source is constructed before
timing; complete HF timing includes generation and all evaluated-density
accuracy checks. Modeled memory is distinct from process peak RSS.

Cases use a six-bohr cubic cell and normalized one-primitive s shells.
He RHF has exponents 4 and 6; H UHF has the same basis with empty beta;
Li UHF has exponents 4, 6 and 9 and a doublet. UHF uses a shifted two-point
mesh. The Li timing deliberately stops after one evaluated iteration;
converged Li is checked separately for scientific acceptance. Nuclear and
electronic Ewald share real cube 4 and reciprocal cube 10, with eta 0.6
(He/H) or 0.5 (Li). AO images use cube 1 and physical cutoff 1 bohr (He/H),
or cube 2 and 6.1 bohr (Li). BvK exchange correction is enabled. Exchange
kinetic-energy cutoffs are 500 hartree (He/H) and 900 hartree (Li).

These small cases establish correctness and expose setup/SCF cost. Their
cutoffs are explicit fixtures, not production accuracy profiles. They do not
establish scaling or a speedup over another algorithm. Fixed-density
integral bounds exclude mesh and Madelung numerical convergence, basis,
relaxed-state/rank sensitivity and roundoff. Complete public BUPO source/job
selection, q-aware RI and operation-qualified symmetry remain open.

Run from the repository with its matching native runtime:

```console
.venv/bin/python benchmarks/manta_periodic_bupo_domains.py --repeats 3
```

Measured with source `e436193b`, Python 3.14.8, NumPy 2.5.3 and a Release
Clang FP64 build on macOS arm64. The native module SHA-256 was
`06855e53704eec55e8512efa2b4f41565dd6d0f7d0029c4e3f5e944a5874fc23`.
Each row reports the median of three runs. The benchmark workers were
separate processes; the local reference-calculation batch also used one
thread while these measurements ran. No machine-idle timing is claimed.

| Case | Candidate labels / executed Q | Completion radius | Plan, us | Generation, ms | Complete HF, s | Planned peak, MB | Actual-support modeled peak, MB | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| He RHF Gamma | 226981 / 115312 | 30 | 44.08 | 18.28 | 1.210 | 304.34 | 263.24 | 179.30 |
| H UHF shifted, empty beta | 900482 / 461416 | 30 | 46.08 | 71.96 | 4.089 | 619.65 | 455.39 | 204.09 |
| Li UHF shifted, one iteration | 2138886 / 1114900 | 41 | 31.75 | 177.88 | 8.002 | 1200.92 | 819.51 | 248.83 |

The complete runs evaluated 9, 8 and 1 SCF iterations respectively; the
last is intentionally nonconverged. Opposite-support audit counts equal
executed Q counts in every case. Reserved domain memory is 21.83, 86.49
and 205.37 MB; it remains charged during HF. Candidate bounds explain why
pre-generation admission is larger than the modeled peak using the actual
support. Neither number is a measured allocation or RSS promise.

Maximum observed matrix/energy exchange bounds were 1.052e-12/4.131e-12
for RHF, 1.607e-12/4.521e-12 for empty-beta UHF, and
5.736e-19/8.005e-17 for the capped doublet, below unchanged 1e-8 targets.
Atomic units and the method's restricted/unrestricted exchange scaling
apply. These analytic bounds exclude floating-point roundoff; independent
matrix/energy comparison tolerances are separate.

Scientific acceptance used five complete calculation scenarios through the
queue interface: Gamma and skew-cell He RHF, shifted H UHF with empty beta,
and shifted Li UHF in converged and deliberately capped states. Each of the
ten accepted native runs (factory plus explicit-vector control per case)
preserved D/F/H/S and energy bitwise. Independent ordered Gaussian/Fourier
integration and a separate generalized-eigenproblem SCF gave maximum Fock,
same-density energy and relaxed-energy discrepancies of 2.621e-13,
1.765e-11 and 1.892e-11 in atomic units. Density error was at most
9.750e-10. Comparisons retained the established 3e-11 energy/Fock and
3e-9 density tolerances, separate from analytic bounds excluding roundoff.

An independent fetched-array assessment reconstructed electron counts,
energy traces and all five exchange spheres by integer enumeration. Maximum
trace discrepancy was 5.152e-12 hartree; reciprocal geometry differed by at
most 7.106e-15 inverse bohr. Two insufficient cutoffs refused the unchanged
exchange targets. Stopped states remained unconfirmed/nonconverged and
retained their evaluated density, Fock and energy through the shared Fock
interface. The calculation/reference batch used 390.47 seconds elapsed,
384.94 CPU seconds and 637.11 MiB peak RSS under a one-thread allocation.

Twelve focused tests and 38 related regressions cover exact Gamma boundary
membership, shifted odd meshes, skew/left-handed cells, empty spheres,
conjugacy, owned exports, immutable receipts, original-source checks,
all five exact/one-short caps and output provenance. Native/reference
identities were pinned throughout calculation and fetched assessment.
No suitable ORCA/CRYSTAL reference matches these custom finite-support
certificates; this evidence qualifies the stated numerical route, without
claiming external production parity.
