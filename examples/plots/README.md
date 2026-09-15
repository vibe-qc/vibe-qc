# Plot-generation scripts

Each script in this directory reads the output of a parent example
(``../input-*.py``) and produces a figure for embedding in the
tutorial that walks through it. Outputs land at
``../../docs/_static/plots/<name>.png`` and are referenced from
``docs/tutorial/*.md`` via standard MyST image syntax:

```markdown
![Caption](../_static/plots/<name>.png)
```

## Convention

* One script per figure, named to match the figure file.
* Scripts are idempotent — re-running regenerates the PNG bit-for-bit
  (modulo non-determinism in the underlying calculation, which we
  pin where possible).
* Figures are 300 DPI PNG, ~5–6 inches wide, sized to read cleanly
  at the docs page width.
* Each script prints what it wrote and any sanity-check numbers
  (barrier height, peak position, fit residual, …) so you see at a
  glance whether the underlying calculation looks right.

## Running

If the parent example's output file is missing, the script raises
with a directive message pointing at it. Typical workflow:

```sh
python3 examples/workflows/input-nh3-umbrella-neb.py        # produces .traj
python3 examples/plots/nh3-umbrella-neb-mep.py    # reads .traj, writes .png
```

## Catalog

| Script | Reads | Writes | Embedded in |
|---|---|---|---|
| `nacl-ewald-alpha-invariance.py` | (re-runs Ewald sweep; <1 s) | `nacl-ewald-alpha-invariance.png` | tutorial 06 |
| `water-vibrational-frequencies.py` | (re-runs SCF + ASE Hessian; ~30 s) | `water-vibrational-frequencies.png` | tutorial 09 |
| `water-frontier-orbitals.py` | (re-runs the SCF; ~1 s) | `water-frontier-orbitals.png` | tutorial 11 |
| `h-chain-bands-dos.py` | (re-runs the calculation; ~1 s) | `h-chain-bands-dos.png` | tutorial 12 |
| `h-chain-crystalline-orbitals.py` | (re-runs Hcore + Bloch eigvecs; ~2 s) | `h-chain-crystalline-orbitals.png` | tutorial 12 |
| `water-basis-convergence.py` | (re-runs HF on cc-pVXZ series; ~30 s) | `water-basis-convergence.png` | tutorial 13 |
| `water-dimer-d3-curve.py` | (re-runs SCF scan; ~2 min) | `water-dimer-d3-curve.png` | tutorial 14 |
| `water-functional-comparison.py` | (re-runs six SCFs; ~5 s) | `water-functional-comparison.png` | tutorial 15 |
| `h-chain-peierls-energy.py` | (re-runs the SCF scan; ~30 s) | `h-chain-peierls-energy.png` | tutorial 17 |
| `h-chain-peierls-bands.py` | (re-runs Hcore bands; ~1 s) | `h-chain-peierls-bands.png` | tutorial 17 |
| `nh3-umbrella-neb-mep.py` | `output-nh3-umbrella-neb.traj` | `nh3-umbrella-neb-mep.png` | tutorial 19 |
