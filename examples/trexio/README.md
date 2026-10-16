# TREXIO interchange examples

Install vibe-qc with its optional `[trexio]` extra into the active Python
environment. From this checkout, use `python -m pip install -e '.[trexio]'`.
QVF remains enabled in every calculation example; TREXIO is requested as an
additional wavefunction output.

Run from the repository root. Calculation scripts require `--output-dir`;
choose a directory outside the checkout. HDF5 is the default backend and
`--backend text` writes a directory that must be copied as a whole.

```sh
export OMP_NUM_THREADS=1
python examples/trexio/molecular_restart.py --case water --output-dir ~/vibeqc-runs/trexio/water
python examples/trexio/molecular_restart.py --case oh --output-dir ~/vibeqc-runs/trexio/oh
python examples/trexio/molecular_restart.py --case nah --output-dir ~/vibeqc-runs/trexio/nah
python examples/trexio/correlated_wavefunction.py --method casscf --output-dir ~/vibeqc-runs/trexio/lih
python examples/trexio/periodic_restart.py --output-dir ~/vibeqc-runs/trexio/helium
```

| Script | Model and checks |
|---|---|
| [`molecular_restart.py`](molecular_restart.py) | Water RHF and OH UHF/def2-SVP; NaH RHF/LANL2DZ ECP; orbital overlap, electron counts, reconstructed ECP and READ energy |
| [`correlated_wavefunction.py`](correlated_wavefunction.py) | LiH/STO-3G CASCI or CASSCF(2,2) with one frozen core orbital, or FCI; CI normalization, core bits, spin RDMs, energy and QVF |
| [`periodic_restart.py`](periodic_restart.py) | He cubic cell, RHF/STO-3G with two k points; sampling, complex-safe coefficients, weighted electrons and READ energy |
| [`convert_backend.py`](convert_backend.py) | Transport all fields supported by the installed TREXIO library; no native basis reconstruction; refuses existing destinations |

All geometries in the scripts are in bohr and printed energies are in
Hartree. These small models demonstrate file interchange, not converged
chemical or solid-state predictions. Successful calculation examples print
`PASS`; a failed assertion exits nonzero. Rerunning a calculation replaces
its output files, so use a fresh directory to preserve earlier results.

```sh
python examples/trexio/convert_backend.py \
    ~/vibeqc-runs/trexio/water/water.trexio.h5 \
    ~/vibeqc-runs/trexio/water-copy.trexio --backend text
```

For an independent molecular HF or CI energy reconstruction, install PySCF
and TREXIO into a reference environment and run the existing checker in that
interpreter. It does not import vibe-qc:

```sh
python examples/regression/runner_trexio_pyscf.py ~/vibeqc-runs/trexio/water/water.trexio.h5
python examples/regression/runner_trexio_pyscf.py ~/vibeqc-runs/trexio/nah/nah.trexio.h5 --tol-energy 1e-6
python examples/regression/runner_trexio_pyscf.py ~/vibeqc-runs/trexio/lih/lih-casscf.trexio.h5
```

The ECP comparison uses a microhartree tolerance for different integral
quadratures. This checker supports real molecular Gaussian wavefunctions;
it rejects periodic input. It evaluates a stored state rather than
performing an independent SCF or CASSCF optimization.

Read the [molecular tutorial](../../docs/tutorial/trexio_exchange.md),
[CI and periodic tutorial](../../docs/tutorial/trexio_correlated_periodic.md),
and [format reference and literature comparison](../../docs/user_guide/trexio.md).
