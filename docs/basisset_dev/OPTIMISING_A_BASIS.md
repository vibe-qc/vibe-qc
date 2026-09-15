# Optimising a basis set with vibe-qc (molecular)

vibe-qc has a native, in-process **molecular** basis-set optimiser: it adjusts
Gaussian exponents and contraction coefficients to minimise the SCF energy of
one or more reference systems, driven by a robust BDIIS loop with the **analytic**
energy gradient (no per-parameter re-SCF finite differences). It is the
basis-set analogue of CRYSTAL23's OPTBASIS, the same κ·ln-condition objective
option (VandeVondele 2007) and BDIIS driver (Daga 2020), but with analytic
gradients and for any of RHF / UHF / RKS / UKS.

This page is the usage guide. For the math see
[`ENERGY_GRADIENT_DESIGN.md`](ENERGY_GRADIENT_DESIGN.md); for the periodic
extension (work in progress) see
[`PERIODIC_GRADIENT_DESIGN.md`](PERIODIC_GRADIENT_DESIGN.md).

## The three steps

1. **Parametrise**, declare which numbers are free.
2. **Optimise**, one call.
3. **Emit**, write the improved basis back out.

```python
import vibeqc as vq
from vibeqc.basis_crystal import emit_g94, parse_crystal_atom_basis_file
from vibeqc.basis_optimization.parametrise import (
    BasisParametrisation, FreeSpec, Transform)
from vibeqc.basis_optimization.recipes.molecular import optimize_molecular_basis

src = ...  # python/vibeqc/basis_library/sources/pob-TZVP/06_C
par = BasisParametrisation(
    atoms={"C": parse_crystal_atom_basis_file(src)},
    free=[
        # vary C's d-polarisation exponent in log-space, with a lower bound
        FreeSpec("C", 7, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
    ],
)

res = optimize_molecular_basis(
    par,
    lambda: vq.Molecule([vq.Atom(6, [0, 0, 0])], multiplicity=3),
    functional="PBE", open_shell=True,        # UKS; see "reference" below
)
print(res.summary())
improved = emit_g94([res.optimized_atoms["C"]])   # the optimised basis as a .g94 deck
```

A complete runnable example is
[`examples/molecular/optimize_basis_h2o_pbe.py`](../../examples/molecular/optimize_basis_h2o_pbe.py).

## Choosing the reference (`functional` / `open_shell`)

| reference | call |
|-----------|------|
| RHF (closed-shell HF)  | `optimize_molecular_basis(par, mol)` |
| UHF (open-shell HF)    | `optimize_molecular_basis(par, mol, open_shell=True)` |
| RKS (closed-shell DFT) | `optimize_molecular_basis(par, mol, functional="PBE")` |
| UKS (open-shell DFT)   | `optimize_molecular_basis(par, mol, functional="PBE", open_shell=True)` |

LDA, GGA, and global hybrids are supported for the KS paths; meta-GGA,
range-separated, and double-hybrids raise `NotImplementedError` (use the FD
fallback, below).

## Free parameters

`FreeSpec(symbol, shell_idx, prim_idx, field, transform=..., bounds=...)`:

* `field="exponent"`, a primitive Gaussian exponent. Use `Transform.LOG`
  (exponents are positive and span orders of magnitude) with a lower `bounds`
  guard (e.g. `(0.05, None)`) against collapse toward linear dependence.
* `field="coeff"`, a (non-SP) contraction coefficient. Use `Transform.LINEAR`.
* SP shells (`coeff_s`/`coeff_p`) have no analytic derivative yet, see the FD
  fallback.

Both exponents and coefficients are fully analytic for HF and KS.

## Multiple reference systems (calibration set)

Real basis sets are fit to a *set* of systems, not one. Pass several molecule
factories and (optionally) weights:

```python
from vibeqc.basis_optimization.recipes.molecular import optimize_molecular_basis_multi

res = optimize_molecular_basis_multi(
    par, [mol_a, mol_b, mol_c], weights=[1.0, 1.0, 0.5], functional="PBE")
```

The objective is `Σ_s w_s · E(system_s)` with the analytic gradient
`Σ_s w_s · ∇E_s`; all systems share the one basis and the same reference.

## The conditioning penalty (OPTBASIS-style)

To defend against (near-)linear dependence as exponents move together, add the
VandeVondele condition-number penalty `γ·ln κ(S)` (and/or the λ_min hinge):

```python
res = optimize_molecular_basis(par, mol, use_cond_penalty=True, cond_gamma=1e-3)
```

The penalty is built from the per-element atomic overlap and has its own
analytic gradient, so it composes with the energy gradient at no extra SCF cost.

## Finite-difference fallback

`analytic=False` drives BDIIS with the driver's finite-difference gradient
instead of the analytic one, needed for SP `coeff_s`/`coeff_p` parameters, and
a general escape hatch:

```python
res = optimize_molecular_basis(par, mol, analytic=False)
```

## Result

`MolecularOptResult` carries:

* `optimized_atoms`, `{symbol: CrystalAtomBasis}` at the optimum (feed to
  `emit_g94` / `emit_crystal`),
* `starting_objective` / `optimal_objective` / `improvement_mha`,
* `converged` / `message` / `n_evaluations` / `wall_seconds`,
* `citations`, the method-citation keys (see below),
* `summary()`, a human-readable report (per-parameter start→optimum).

## Citations (mandatory, CLAUDE.md § 8)

When you publish an optimised basis, cite:

* the **optimiser method**, `res.citations` lists the keys
  (`daga_optbasis_2020` for BDIIS, `vandevondele_basisopt_2007` for the
  condition-number penalty, the Pulay-DIIS papers), resolvable in
  [`python/vibeqc/output/citations/database.toml`](../../python/vibeqc/output/citations/database.toml);
* the **starting basis set** (e.g. pob-TZVP: `peintinger_pob_tzvp_2013`) and the
  **functional**, these surface in the references block of any SCF `.out` the
  optimiser drives.
