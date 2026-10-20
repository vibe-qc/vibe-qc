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
3. **Emit**, write the improved basis back out, or save it to the basis
   store with `res.save_to_store(name, starting_basis="pob-TZVP")` so a
   later run loads it by name (user guide, "Saving the optimised basis").

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

For the record, CRYSTAL23's `OPTBASIS` (manual section 3, keyword
`OPTBASIS`) minimises exactly this `E_tot + gamma ln kappa` with BDIIS,
converging on an energy threshold (`TOLOPTE`) and a basis-gradient threshold
(`TOLOPTG`), and documents no diagnostics beyond the two overlap eigenvalues
that make up `kappa`; its only extra outputs are the per-cycle SCF log and the
per-cycle basis deck. vibe-qc's `LDDiagnostics.shell_contributions` (the
near-null eigenvector resolved per shell) therefore already goes beyond what
OPTBASIS exposes, and there is nothing further to port from it.

## Other drivers

`optimize_molecular_basis` is BDIIS only. For a script that builds its own
`(objective, gradient)` pair, `vibeqc.basis_optimization` also ships
`optimize_scipy`, `optimize_minuit`, `optimize_bobyqa` (model-based
derivative-free, the choice without an analytic gradient), `optimize_dfols`
(least-squares residuals) and `optimize_cmaes` (global pre-pass), plus the
`recipes.global_then_polish` CMA-ES-then-BDIIS chain. The user guide's
"Choosing the optimiser" section
([`docs/user_guide/basis_optimization.md`](../user_guide/basis_optimization.md))
has the table, and `scripts/basisset_dev/driver_benchmark_stage1.py` the
side-by-side evaluation counts.

## Periodic one-compound smoke (Stage 2)

`recipes.one_compound.run_lih_smoke` is the periodic sibling of the Stage-1
single-atom smoke: pob-TZVP H's diffuse-most s exponent refit on the two-atom
primitive cell of rocksalt LiH, the periodic RHF energy per cell from
`run_periodic_job` at a chosen k-mesh (BIPOLE Coulomb and exchange, since a
candidate basis has no fitting set), started at 1.5 times the published
value. Any driver works (`driver="bobyqa"`, `"bdiis"`, `"scipy"`, `"cmaes"`,
`"cmaes+bdiis"`); BDIIS runs on central differences here because the
periodic analytic gradient is not production yet. The design expected the
optimum between the free-atom value (about 0.131) and the published 0.1795;
the first live run (RHF, 2x2x2, Py-BOBYQA, 13 evaluations, vibe-qc 0.17.14)
landed at 0.189, just above the published value, with the cell energy
3.8 mHa above the PT2013 Hartree-Fock reference. The report prints both the
design window and the PT2013 reference for orientation; neither is asserted.
The measured run and its evaluation history are in
[the verification report](VERIFICATION_REPORT.md). Budget one periodic SCF
of about ten minutes per evaluation at 2x2x2 on a laptop.

```python
from vibeqc.basis_optimization.recipes.one_compound import run_lih_smoke
print(run_lih_smoke(kpoints=(4, 4, 4), driver="bobyqa").summary())
```

`run_one_compound` is the general form: any one-parameter parametrisation on
any `PeriodicSystem` factory, with `use_cond_penalty=True` for the OPTBASIS
objective shape.

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
