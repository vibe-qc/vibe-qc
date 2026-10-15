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

The molecular recipe is a production route; the label vocabulary is
defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## Theory

**The variational objective.** A Gaussian basis is a set of
parameters, the exponents $\zeta$ and contraction coefficients $c$ of
each primitive, and for a fixed molecule the SCF energy is a function
of them, $E(\zeta, c)$. Because the SCF energy is variational in the
orbital coefficients, lowering it by moving the basis parameters
improves the basis in the same sense that a larger basis would: the
best basis of a given size for that molecule is the minimiser of $E$.
The optimiser works in a transformed space, typically
$x = \ln \zeta$ (`Transform.LOG`), so that exponents stay positive and
steps are relative, and it minimises the composite objective

$$
\Omega(x) = E_{\rm SCF}(x) + \gamma\, \ln \kappa\!\bigl(S(x)\bigr) + \lambda\, P_{\rm LD}(x),
$$

in which $\kappa(S)$ is the condition number of the atomic overlap
matrix and $P_{\rm LD}$ a linear-dependence penalty. The penalties are
what keeps the minimiser away from the trivial solution: two exponents
drifting together, or a diffuse function collapsing toward a neighbour,
lower $E$ by a negligible amount while making the basis numerically
singular, and VandeVondele and Hutter
({vibeqc-cite}`vandevondele_basisopt_2007`) introduced the
$\gamma \ln \kappa$ term for exactly that reason when fitting their
molecularly optimised sets.

**The analytic gradient.** The derivative of the SCF energy with
respect to a basis parameter has the same structure as a nuclear
gradient: with the converged density $D$ and energy-weighted density $W$,

$$
\frac{\partial E}{\partial \zeta}
  = \sum_{\mu\nu} D_{\mu\nu} \frac{\partial h_{\mu\nu}}{\partial \zeta}
  + \frac12 \sum_{\mu\nu\lambda\sigma} \Gamma_{\mu\nu\lambda\sigma}
      \frac{\partial (\mu\nu|\lambda\sigma)}{\partial \zeta}
  + \int v_{xc}\, \frac{\partial \rho}{\partial \zeta}\, d\mathbf r
  - \sum_{\mu\nu} W_{\mu\nu} \frac{\partial S_{\mu\nu}}{\partial \zeta},
$$

where the derivative of each integral with respect to an exponent is
again a Gaussian integral (differentiating $e^{-\zeta r^2}$ brings down
$-r^2$, a shell of angular momentum raised by two) and the overlap term
accounts for the orbitals' response. That is what `analytic=True`
evaluates; SP-shared contraction coefficients have no closed form and
fall back to finite differences.

**The BDIIS update.** CRYSTAL's `OPTBASIS` and its BDIIS optimiser
({vibeqc-cite}`daga_optbasis_2020`) apply Pulay's direct inversion in
the iterative subspace ({vibeqc-cite}`pulay_diis_1980`;
{vibeqc-cite}`pulay_diis_1982`) to the basis parameters: the last
`diis_size` iterates $x_k$ and their error vectors $e_k$ are combined
with coefficients that minimise $\|\sum_k c_k e_k\|$ subject to
$\sum_k c_k = 1$, and the extrapolated point is the next iterate.
vibe-qc's implementation uses a BFGS inverse Hessian for the error
vectors, $e_k = -H_k^{-1} g_k$, so that the extrapolation carries
curvature information, and guards each step with a trust radius and
Armijo backtracking so that $\Omega$ never increases on an accepted
step; an ill-conditioned DIIS system falls back to the quasi-Newton
step. Convergence is declared when $|\Delta\Omega|$ and
$\|g\|_\infty$ fall below `tol_energy` and `tol_grad`.

**The cohesive objective.** A basis for solids is not optimised on one
molecule. The multi-crystal recipes (pob-TZVP was made this way)
minimise a weighted sum over a training set of structures,
$\Omega = \sum_s w_s E_s(x)$ with isolated atoms and bulk crystals in
the set, so that the same exponents serve the free atom and the solid;
a cohesive-energy objective $E_{\rm bulk} - \sum E_{\rm atom}$ balances
the two explicitly. That recipe drives an external periodic program
through the `vibe-basis` driver and is described below.

**Where it helps and where it fails.** Re-optimising a handful of
valence and polarisation exponents recovers a few mHa on a small
molecule and adapts a solid-state basis to a molecular environment (or
the reverse). The optimum is specific to the training set: a basis
fitted to one molecule at one functional is not transferable in the
way a published family is, and pushing exponents diffuse to chase the
energy produces near-linear-dependent sets that other geometries cannot
converge, which is what the penalties and bounds guard against.

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
  The other drivers export their keys the same way: `DFO_CITATION_KEYS` for
  Py-BOBYQA and DFO-LS, `CMAES_CITATION_KEYS` for CMA-ES, and
  `recipes.global_then_polish.method_citations()` for the chained recipe.

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

## Choosing the optimiser

`optimize_molecular_basis` always drives BDIIS. Below that one-call
recipe, `vibeqc.basis_optimization` exposes six drivers with one shared
return type, `OptResult`, so a script can swap them without changing
anything else. All of them take the objective, the starting vector and
the `bounds` from `parametrisation.optim_bounds()`; the objective and
its analytic gradient come from
`recipes.objective.make_native_objective_gradient` (or its
`make_rhf_`, `make_uhf_`, `make_rks_`, `make_uks_` wrappers).

| driver | needs | best for |
|---|---|---|
| `optimize_bdiis` | nothing (scipy-free, in-house) | the default: BDIIS with a BFGS inverse Hessian and a trust region, cheapest by far with the analytic gradient |
| `optimize_scipy` | scipy | L-BFGS-B baseline with finite-difference gradients |
| `optimize_minuit` | iminuit | Minuit2 MIGRAD plus HESSE parameter uncertainties for a publication |
| `optimize_bobyqa` | Py-BOBYQA | a scalar objective *without* an analytic gradient: external-program energies, SP-shell coefficients. A model-based step costs about one SCF, a finite-difference gradient costs two per parameter |
| `optimize_dfols` | DFO-LS | a least-squares fit of energies to reference values; needs the residual vector (`residuals=` or an objective with a `residuals()` method) and refuses a plain energy |
| `optimize_cmaes` | cma | a global pre-pass from a poor starting basis: population-based, derivative-free, tolerant of failed SCFs, not a precise finisher |

The last three are optional backends. Install them with the
`basisopt-drivers` extra:

```sh
pip install -e '.[basisopt-drivers]'
```

Each driver imports its backend lazily and raises a `RuntimeError` naming
the missing package, so a plain `import vibeqc.basis_optimization` never
needs any of them. Py-BOBYQA and DFO-LS are GPL-3.0 licensed; they are
opt-in for that reason and are never imported by the core package.

A starting basis that may sit in the wrong basin (a far-off guess, an
exponent deliberately perturbed to test robustness) is the case for the
CMA-ES pre-pass. The recipe chains it with a BDIIS polish:

```python
from vibeqc.basis_optimization.io import TempBasisLibrary
from vibeqc.basis_optimization.recipes.objective import make_uhf_native_objective_gradient
from vibeqc.basis_optimization.recipes.global_then_polish import optimize_global_then_polish

with TempBasisLibrary() as lib:
    objective, gradient = make_uhf_native_objective_gradient(parametrisation, molecule_factory, lib)
    res = optimize_global_then_polish(
        objective, parametrisation.pack(), bounds=parametrisation.optim_bounds(),
        grad=gradient, cmaes_max_evaluations=100, cmaes_seed=1)

res.driver              # "cmaes+bdiis"
res.raw["cmaes"].fun    # where the pre-pass got to
optimised = parametrisation.unpack(res.x)
```

CMA-ES sizes its first step from the *optimiser-space* box, a quarter of
the narrowest log-exponent range, so the initial population stays inside
a sensible exponent window. Pass `cmaes_sigma0` to override it and
`cmaes_seed` for a reproducible run.

`scripts/basisset_dev/driver_benchmark_stage1.py` runs every driver on the
single-atom Stage-1 problem and prints the evaluation counts side by side;
the measured table is in
[the verification report](../basisset_dev/VERIFICATION_REPORT.md).

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
