# Public BUPO HF job cost

Run `python benchmarks/manta_periodic_bupo_job.py` in the built checkout.
Each case uses one numerical thread, a separate process and three repeated
complete public jobs. Every job constructs a fresh admitted source, runs
HF and writes managed text, geometry, citations, structured SCF records
and QVF. Output discovery is timed separately and constructs no scientific
source. Its memory reservation is the declared complete-job model cap,
not a calculation of source contents or a proof of feasibility.

The compact synthetic helium RHF and shifted empty-beta hydrogen UHF cases
converge. Shifted lithium UHF is capped after one evaluated state. These
are cost probes, not accuracy profiles. Reservations are conservative
ownership/work models and are distinct from RSS. RSS includes Python,
libraries and fixture data. There is no scaling or speedup claim.

Scientific acceptance requires public/native state parity, independently
accepted Gaussian/Fourier/eigensolver references for the identical physical
source, managed source identities, QVF evaluated-density round trips and
all five complete-job resource caps. No matched ORCA/CRYSTAL custom finite
support reference is available. Evaluated-density integral targets exclude
basis, mesh, relaxed-state/rank, Madelung numerical and roundoff errors.

## Measured public job cost

Three fresh jobs per case, one process per case, one numerical thread,
ARM64 macOS and Python 3.14.8. Other work was active on the machine, so
these are not idle-machine or comparative performance measurements.
Every row uses BvK Ewald exchange correction and writes nine managed
artifacts. Reciprocal cutoffs are 500, 500 and 900 hartree, respectively.
Dry-run timing includes the count-only census and output manifest.

| Case | Complete job (s) | Dry run (ms) | Job storage (MB) | Preparation admission (MB) | Actual-support model (MB) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| rhf-gamma | 0.715 | 32.24 | 1.81 | 498.87 | 457.77 | 260.06 |
| uhf-shifted-empty-beta | 2.172 | 31.35 | 1.84 | 814.21 | 649.95 | 275.70 |
| uhf-shifted-capped | 4.039 | 32.67 | 1.31 | 1399.68 | 1018.27 | 305.70 |

The preparation model reserves candidate exchange support as well as the
complete source and job owners. The actual-support model uses the generated
spheres. Their conservative bounds should not be equated with RSS. The
helium and empty-beta hydrogen runs converge in nine and eight iterations;
lithium retains its evaluated one-iteration nonconverged state. The dry-run
memory estimate is the configured 2 GiB cap in all three probes.

## Scientific acceptance

Five complete public calculations ran through `vq`: Gamma and skew-cell
helium RHF with strict zero-mode exchange, shifted empty-beta hydrogen UHF,
shifted lithium UHF, and a capped lithium state. The UHF cases use the BvK
Ewald convention. Public D/F/H/S arrays, energies and physical source/state
identities are bitwise equal to the previously accepted prepared-source
states. Previously accepted independent ordered Gaussian/Fourier,
electrostatic and generalized-eigenproblem artifacts were reused only after
checking their hashes, identical physical settings and unchanged reference
implementations. A separate NumPy-only assessment of fetched arrays passed.

Maximum differences from those independent references are 2.621e-13 in F,
1.765e-11 hartree in same-density energy, 1.892e-11 hartree in converged
energy, and 9.750e-10 in D. The energy-trace discrepancy is at most
5.152e-12 hartree. Existing tolerances remain 3e-11 for energy/F and 3e-9
for D, with the original relative allowances. All integral targets pass.
QVF round trips reproduce the actual spin densities exactly in these
five cases, including empty beta and the unconfirmed capped state.
All 45 managed output files were retrieved and hash-checked.

The calculation command used 61.09 s wall time, 59.59 CPU seconds and
348.16 MiB peak RSS. Focused source checks cover 22 new cases, 173 existing
direct/RI/KS public job cases, and seven additional inventory and hybrid
lifecycle cases. These include independent AO-image gauge transport,
insufficient support refusal, source-free dry runs and exact/one-short
admission for all five resource caps.

The measured source commit is
`06731fc71787419e412ac967732089ea07c3a183`; native SHA256 is
`123f7e1b8ed9ab2fd691466316ce6f4c0e60982115be849cbebcf6c47294ae0a`.
The benchmark uses this same native runtime. Implementation commit
`5fe0c0010c02f924f5049801199cbe38e202609b` retains identical numerical MANTA source.
Rebase-only integration and measured documentation preserve this evidence.
No matched ORCA/CRYSTAL reference is available for this custom finite-support
fixture. No production parity, mesh convergence, relaxed-state guarantee
or whole-method accuracy profile is inferred.
