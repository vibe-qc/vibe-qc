# Molecular basis-set optimisation

vibe-qc can re-optimise the free parameters of a Gaussian basis set,
its exponents and contraction coefficients, against a molecular energy,
using the **analytic** basis-parameter gradient and a robust BDIIS loop.
It is the basis-set analogue of CRYSTAL's `OPTBASIS`, but it drives the
optimiser with the analytic gradient instead of per-parameter
finite-difference re-SCFs, and it works for any of RHF / UHF / RKS / UKS.
The one-call entry point is `optimize_molecular_basis`.

```{note}
This is the in-process **molecular** path. The separate multi-crystal,
derivative-free production recipe
(`vibeqc.basis_optimization.recipes.production`, NLopt over an
[`EnergyEngine`](cohesive_energies.md)) is a different workflow aimed at
solid-state basis libraries. It needs an extra install step, see
[The external-program path](#the-external-program-path) below.
```

## The external-program path

The molecular recipe below runs entirely inside vibe-qc and needs
nothing beyond a normal install. The solid-state recipes are different:
they drive an **external** SCF program (CRYSTAL today) out of process
through the co-located `vibe-basis` driver, so they need that driver
installed.

These six modules import `vibe_basis` at module load:

| Module | Role |
|---|---|
| `basis_optimization.calculators` | `VibeQcEngine`, the in-process vibe-qc engine (imports the shared `EnergyEngine` protocol from the driver) |
| `basis_optimization.recipes.crystal_objective` | multi-crystal objective |
| `basis_optimization.recipes.crystal_stage1` | single-atom stage |
| `basis_optimization.recipes.crystal_stage2` | bulk-solid stage |
| `basis_optimization.recipes.crystal_stage3` | full test-set stage |
| `basis_optimization.recipes.production` | production refit |

Without the driver they raise `ModuleNotFoundError: No module named
'vibe_basis'` at import. For a standalone environment that does not build
vibe-qc, use the component lifecycle:

```sh
./vibe-basis/scripts/install.sh
```

That creates `vibe-basis/.venv`; matching `update.sh`, `reinstall.sh`, and
`uninstall.sh` commands manage it safely. To add the driver to an existing
vibe-qc environment instead, use vibe-qc's `basisopt` extra:

```sh
uv pip install -e '.[basisopt]'
```

pip ignores `[tool.uv.sources]`, so under pip install it in two steps:

```sh
pip install -e . && pip install -e vibe-basis/
```

Check an environment before starting a campaign:

```sh
.venv/bin/python examples/basisset_dev/check_basisopt_install.py
```

That script reports both paths separately and exits nonzero when the
external path is unusable, so it works as a guard at the top of a
campaign script. See [Installation](../installation.md) for the full
notes, including the `--no-build-isolation` caveat.

```{note}
`vibe-basis` is a *driver*, not a solver. It emits input decks, submits
them (locally or through the `vq` queue), and parses what comes back;
the energies are computed by the external program. It is versioned
independently of vibe-qc, see `vibe-basis/VERSIONING.md`.
```

## The one-call recipe

You declare which parameters are free, supply the molecule, pick the
reference, and call the optimiser:

```python
import vibeqc as vq
from vibeqc.basis_crystal import parse_crystal_atom_basis_file, emit_g94
from vibeqc.basis_optimization.parametrise import (
    BasisParametrisation, FreeSpec, Transform,
)
from vibeqc.basis_optimization.recipes.molecular import optimize_molecular_basis

# 1. The starting basis and which parameters are free. FreeSpec picks one
#    parameter by (symbol, shell index, primitive index, kind); LOG transform
#    plus a lower bound guard against collapse toward linear dependence.
atoms = {"O": parse_crystal_atom_basis_file(...), "H": parse_crystal_atom_basis_file(...)}
parametrisation = BasisParametrisation(
    atoms=atoms,
    free=[
        FreeSpec("O", 7, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
        FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
    ],
)

# 2. A factory that returns the molecule to optimise against (re-evaluated
#    each step). Geometry in bohr.
def molecule_factory():
    return vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                        vq.Atom(1, [0.0, 1.4304, 1.1071]),
                        vq.Atom(1, [0.0, -1.4304, 1.1071])], multiplicity=1)

# 3. Optimise (closed-shell RKS/PBE here; functional=None gives HF).
res = optimize_molecular_basis(parametrisation, molecule_factory, functional="PBE")
print(res.summary())

optimised = res.optimized_atoms                # {symbol: CrystalAtomBasis} at the optimum
print(emit_g94([optimised["O"]]))              # write the improved O basis as a Gaussian-94 deck
```

## Choosing the reference

The `functional` and `open_shell` arguments select the SCF reference the
basis is optimised against, exactly as for an ordinary single point:

| call | reference |
|---|---|
| `functional=None` | RHF (closed shell) or UHF (`open_shell=True`) |
| `functional="PBE"` (or any libxc name) | RKS (closed shell) or UKS (`open_shell=True`) |

## What you get back

`optimize_molecular_basis` returns a `MolecularOptResult`:

- `optimized_atoms`, the `{symbol: CrystalAtomBasis}` dict at the optimum,
  ready to write out with `emit_g94(...)` or `emit_crystal(...)`.
- `summary()`, a formatted report of the objective, the improvement in
  mHa, and the per-parameter start and end values.
- `improvement_mha`, `starting_objective` / `optimal_objective` (Ha),
  `n_evaluations`, `wall_seconds`, `converged`, `message`.
- `x_optimal` and `history` for the optimiser trajectory.
- `citations`, the optimiser-method citation keys to cite when you publish an
  optimised basis (BDIIS, the conditioning penalty, the Pulay-DIIS papers);
  `summary()` prints them. Also cite the starting basis set and the functional.

## Worked example

The bundled script `examples/molecular/optimize_basis_h2o_pbe.py`
re-optimises three pob-TZVP valence and polarisation exponents (O
diffuse-s, O d-polarisation, H p-polarisation) for an isolated water
molecule at RKS/PBE. Its measured run:

```text
molecular basis optimisation (RKS/PBE)
  objective: -76.36859599 -> -76.37085698 Ha
  improvement: +2.2610 mHa  (lower is better)
  evals: 14   wall: ~55 s
  converged: True
    O.shell3.prim0.exponent      0.356587 ->     0.346200   (diffuse s)
    O.shell7.prim0.exponent      0.253462 ->     0.513548   (d polarisation)
    H.shell3.prim0.exponent      0.800000 ->     1.194617   (p polarisation)
```

pob-TZVP is calibrated for solids, so the molecular optimum tightens the
polarisation functions and the landing point differs from the published
pob-TZVP values: the point is the machinery, not the numbers.

## Options and limits

The defaults are tuned for a well-conditioned exponent optimisation;
the knobs worth knowing:

- `analytic` (default `True`), drive BDIIS with the analytic gradient.
  Set `False` to fall back to the driver's finite-difference gradient,
  needed for SP-coupled `coeff_s` / `coeff_p` contraction parameters,
  which have no closed-form derivative. Single-primitive exponents and
  ordinary contraction coefficients all have analytic gradients.
- `use_cond_penalty` / `cond_gamma` and `use_ld_penalty` / `ld_lambda`,
  add a conditioning or linear-dependence penalty to keep the optimiser
  away from near-degenerate exponent sets (combine with `gated_symbols`
  to apply it per element).
- `max_iter`, `tol_energy`, `tol_grad`, `diis_size`, `trust_radius`, the
  BDIIS controls. `tol_grad` defaults to 1e-5 for HF and 1e-4 for KS (the
  grid-based KS gradient is accurate to about 1e-5).

The recipe optimises a *fixed contraction pattern* (it moves exponents
and coefficients, it does not add or remove primitives), and it is a
molecular path: solid-state basis libraries go through the multi-crystal
production recipe noted above.

## See also

- [Basis sets](basis_sets.md), the bundled basis library and how custom
  decks are loaded.
- `examples/molecular/optimize_basis_h2o_pbe.py`, the runnable script the
  measured numbers above come from.
- `examples/basisset_dev/check_basisopt_install.py`, the preflight guard
  for the external-program path.
- [Installation](../installation.md) § "Optional basis-set optimization
  driver (vibe-basis)", how to install the driver and what it pulls in.
- `vibe-basis/TUTORIAL.md`, the end-to-end walkthrough for a solid-state
  refit, from smoke test to publication-grade run.
