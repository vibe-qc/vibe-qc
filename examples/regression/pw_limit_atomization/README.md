# GPAW plane-wave-limit atomization references

vibe-qc's **out-of-process** generator of plane-wave-limit atomization
energies — the open-source **VASP replacement** for the PW reference data the
revised `pob` Gaussian basis sets are optimised against
(Peintinger & Bredow, *Approaching the plane wave limit with Gaussian basis
sets*, J. Comput. Chem.).

## License boundary (CLAUDE.md §1 / §10) — read first

GPAW is **GPLv3+**. vibe-qc is MPL-2.0. Nothing under `python/vibeqc/` — and
nothing in this tool's own process — imports GPAW or calls it in-process; that
would create a GPL derivative. GPAW runs only in a **separate interpreter**
launched by `subprocess` (`pw_reference.py::_run_external`). We reproduce
GPAW's *numbers*, never its code. The JSON sidecar records the external program
+ version as provenance. This mirrors the sanctioned `core/runner_gpaw.py`
boundary.

## What it computes

For each solid: the per-formula-unit atomization energy

```
E_at = Σ E(free atom) − E(solid) / n_formula_units
```

GPAW's total is PAW-referenced, so absolute totals are not cross-code
comparable — but `E_at` is a *difference* in which the per-atom PAW reference
cancels, so it is directly comparable to VASP and experiment.

Key method points (see `pw_reference.py` docstring for citations):

* **Matched cutoffs.** Solid and free atoms are evaluated at the *same* PW
  cutoff so the PAW reference cancels exactly. r2SCAN converges *slowly* with
  cutoff (its τ has high Fourier components) — for both the solid **and** the
  vacuum atoms — so a single under-converged cutoff is misleading; the tool
  sweeps cutoffs and extrapolates to the PW limit (`E_cut^(-3/2)`).
* **Free-atom references** use Hund ground-state spin, converged
  spin-polarised with a robust Davidson + small-Mixer setup (plain meta-GGA
  atom SCF does not converge); closed-shell atoms run spin-paired. An
  asymmetric box lifts open p-shell degeneracy (aspherical atom).
* **Geometries** are the r2SCAN lattice constants from the benchmark CRYSTAL
  `.d12` inputs; atomization is flat in `a` near the minimum.

## Usage

```sh
# one system, explicit cutoff sweep
python -m examples.regression.pw_limit_atomization.run_pw_reference \
    --systems lif_rocksalt --functional r2scan --cutoffs 800,1000,1200

# the whole light validation set -> Markdown + JSON sidecar
python -m examples.regression.pw_limit_atomization.run_pw_reference \
    --systems validation --out pwref_r2scan.md
```

Set `VIBEQC_GPAW_PYTHON=/path/to/python` (or `--gpaw-python`) to choose the
interpreter that runs GPAW.

## Validation status (LiF, r2SCAN, 2026-06-17)

Converged GPAW r2SCAN PW-limit LiF atomization = **840.5 kJ/mol** (matched
cutoff, flat 1000→1200 eV; k-mesh and atom box converged). Benchmark row:

| | GPAW (PW limit) | CRYSTAL/pob | VASP@900 | exp |
|---|---:|---:|---:|---:|
| LiF | **840.5** | 843.7 | 866.6 | 874.2 |

GPAW agrees with CRYSTAL/pob to ~3 kJ/mol; both sit ~25 kJ/mol below VASP.
**Open question for the paper:** the `E_pob − E_VASP` gap the paper attributes
to pob basis-set incompleteness is not reproduced by an independent PW/PAW code
(GPAW) — pointing at the atomic-reference / PAW-setup convention in VASP rather
than the Gaussian basis. Under active verification (PBE cross-check + more
systems). See `handovers/HANDOVER_GPAW_PW_REFERENCE.md`.

## Free-atom recipe

Free atoms use GPAW's own blessed cohesive-energy recipe (direct minimisation
`etdm-fdpw`, no density mixing, fixed-uniform occupations, symmetry off; its
`doc/.../energetics/cohesive_energy/pt.py`), with **conditional Hund**:
`hund=True` (max spin) for open-shell atoms, `hund=False` for closed-shell atoms
(magmom 0, e.g. Mg 3s²) — `hund=True` stalls the closed-shell meta-GGA direct
minimisation. It reproduces the prior SCF-based Li/F references to
**<0.02 kJ/mol**, and the closed-shell path is validated on MgO/r2SCAN
(GPAW 1009.5 kJ/mol vs VASP@900 1012.0, literature SCAN ~1011). Atom energies
are cached per (element, functional, cutoff, box) for the lifetime of the
process, so a full-set run computes each element once (O recurs across MgO,
Al₂O₃, BeO, …).

## Known limitations

* **Magnetic transition-metal systems** in the full benchmark (Sc₂O₃, Fe₃O₄,
  Cr₂O₃, …) needed spin-polarised solids, magnetic ordering and possibly
  DFT+U, beyond the closed-shell-insulator + main-group scope this doc was
  originally written against. TiO₂ has since been added to `systems.py`'s
  `TM_SYSTEMS` (marked experimental) with GPAW-vs-VASP reference deltas
  computed (2026-07-18) — check `systems.py` for which TM systems are
  wired in today rather than assuming the full list above is still excluded.
