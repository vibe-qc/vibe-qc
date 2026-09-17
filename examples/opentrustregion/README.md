# OpenTrustRegion molecular SCF examples

These self-contained inputs use vibe-qc's optional native OpenTrustRegion
backend. Build it with `-DVIBEQC_ENABLE_OPENTRUSTREGION=ON` and verify
`vibeqc.has_opentrustregion()` first. See the
[installation and option reference](../../docs/user_guide/opentrustregion.md).

Run from the repository root with its virtual environment. All scripts require
an explicit output directory; choose one outside the checkout. Standard job
files are written through `run_job`. A failed scientific check exits nonzero.

```sh
otr_runs=$(mktemp -d)
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method rhf --output-dir "$otr_runs/rhf"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method rks --output-dir "$otr_runs/rks"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method uhf --output-dir "$otr_runs/uhf"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/01_compare_scf.py \
  --method uks --output-dir "$otr_runs/uks"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/02_restart_uks.py \
  --output-dir "$otr_runs/restart"
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/03_stretched_h2.py \
  --output-dir "$otr_runs/h2-follow"
```

| Input | System and check | Tutorial |
|---|---|---|
| [01_compare_scf.py](01_compare_scf.py) | Native versus OpenTrustRegion RHF/RKS water or UHF/UKS water cation; energy, both spin densities where applicable, physical convergence and internal stability | [Molecular SCF](../../docs/tutorial/opentrustregion.md) |
| [02_restart_uks.py](02_restart_uks.py) | Public in-memory READ restart of PBE water cation; energy, density and five-alpha/four-beta electron counts | [Restarts and orbital stability](../../docs/tutorial/opentrustregion_stability.md) |
| [03_stretched_h2.py](03_stretched_h2.py) | Fixed-geometry H2 at four bohr; lower the restricted saddle through UHF rotations while retaining one electron per spin | [Restarts and orbital stability](../../docs/tutorial/opentrustregion_stability.md) |

All examples use STO-3G. PBE runs use an explicitly small teaching grid;
converge basis and grid separately for scientific predictions. An SCF
convergence flag is distinct from the reported manifold-specific stability
verdict. In particular, the stretched-H2 UHF determinant has broken spin
symmetry and is not a spin-pure singlet.

The upstream revision always follows an initial stationary saddle, even with
the `none` or `check` stability policy. Read the tutorial before using these
options to study multiple electronic states. The adapter currently supports
real molecular integer-occupation RHF/UHF and selected RKS/UKS functionals;
periodic, localization and CASSCF adapters are not enabled.
