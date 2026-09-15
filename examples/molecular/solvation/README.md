# Implicit solvation (CPCM) — example scripts

Runnable companions to the [solvation
tutorial](../../../docs/tutorial/solvation_water.md). Each script
is a complete, copy-pasteable workflow you can drop into a fresh
Python file.

| Script | What it does |
|---|---|
| [`input-ch2o-water-single-point.py`](input-ch2o-water-single-point.py) | Single-point hydration energy of formaldehyde — gas vs CPCM water; reports `E_solv = ½ q·V` and `ΔG_solv` totals. |
| [`input-ch2o-water-opt.py`](input-ch2o-water-opt.py) | Geometry optimisation in gas and water via the ASE calculator + analytic CPCM gradient. Reports the solvent-induced C=O bond elongation (~+0.7 pm). |
| [`input-ch2o-dielectric-scan.py`](input-ch2o-dielectric-scan.py) | Sweeps every preset solvent (n-hexane → water) and plots the conductor-screening `(ε − 1)/ε` saturation curve. |

Run any of them from the repo root:

```sh
.venv/bin/python examples/molecular/solvation/input-ch2o-water-single-point.py
```

Each script writes its `.out` / `.molden` / `.traj` outputs next to
itself.

## Picking a different solvent

```python
result = vq.run_job(mol, basis="def2-svp", method="rks",
                    functional="b3lyp", solvent="dmso")
```

See `vibeqc.SOLVENT_PRESETS` for the bundled list (16 solvents +
aliases like `h2o` / `MeOH` / `DCM`), or pass a numeric ε for a
custom dielectric:

```python
result = vq.run_job(..., solvent=25.0)        # custom ε
result = vq.run_job(..., solvent={"epsilon": 25.0, "variant": "cosmo"})
```

## See also

- **User-guide chapter**: [`docs/user_guide/solvation.md`](../../../docs/user_guide/solvation.md)
- **Tutorial**: [`docs/tutorial/solvation_water.md`](../../../docs/tutorial/solvation_water.md)
- **API surface**: `vibeqc.run_cpcm_scf`, `vibeqc.SolventModel`,
  `vibeqc.cpcm_gradient`, `vibeqc.build_cavity` (lower-level entry
  points used internally by `run_job(solvent=...)`).
