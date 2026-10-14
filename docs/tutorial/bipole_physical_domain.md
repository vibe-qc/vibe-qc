# BIPOLE: choosing and replaying a physical domain

This tutorial explains the finite operator used by ab initio BIPOLE RHF, UHF,
RKS and UKS. You will select the domain, inspect its recorded settings and keep
the same choices for forces. It uses LiH in a primitive rocksalt cell. The
example is a runnable input recipe; no numerical reference or runtime is
claimed here. Independent acceptance of the default migration is pending.

## 1. Understand what changes when an atom is relabelled

A periodic atom may be represented in any translated cell. Moving its stored
position by a lattice vector, and translating its attached basis, represents
the same crystal. An index ball truncates integer translations around zero;
it can therefore retain different AO products after that relabelling.
Physical support instead tests the actual separation of the AO centers.
Short-range interactions also use a resolved product-midpoint reach. Nuclear
and reciprocal partners, Coulomb and exchange use the same resolved controls.

At Gamma, overlap must be unchanged by this relabelling to the established
1e-12 absolute tolerance. Away from Gamma, matrix elements acquire the expected
Bloch phases: compare the phase-transformed matrices rather than equal raw
entries. This covariance identity is different from convergence with respect
to cutoff, basis and k-point sampling. Passing one does not prove the other.

## 2. Select the domain through the runner

Download or inspect
[`input-bipole-domain.py`](../../examples/periodic/input-bipole-domain.py).
Its default invocation prints help and performs no calculation:

```sh
python examples/periodic/input-bipole-domain.py
```

After numerical execution is authorized, use a matching native build and run
one calculation at a time. Keep outputs outside the product checkout:

```sh
python examples/periodic/input-bipole-domain.py --run --output /tmp/lih-physical
python examples/periodic/input-bipole-domain.py --run --historical --output /tmp/lih-historical
```

The relevant runner input is:

```python
result = run_periodic_job(
    system, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
    bipole_historical_domain=False,  # Default: physical pair support.
    output=output_stem,
)
```

`True` is an explicit historical reproduction request. It does not repair
historical relabelling dependence. The runner rejects that request when the
selected method is not one of the four ab initio BIPOLE drivers.

No extra cutoff, damping or energy-reference adjustment is needed to select
physical support. Cutoff and k-point convergence still need their own studies.
The example's Gamma calculation is not a converged Brillouin-zone result.

## 3. Inspect the resolved operator

The `.out` file contains a **BIPOLE resolved domain** block with the policy,
one-electron and two-electron settings, nuclear erfc controls, Ewald alpha,
real/reciprocal reaches and applicable grid settings. The completed `.system`
manifest carries `bipole_resolved_domain` and its settings digest; dry-run
output records only the planned policy, not an executed operator.

With a direct driver, inspect the same record without writing your own output
file:

```python
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

options = vq.PeriodicRHFOptions()
kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
scf_controls = dict(historical_domain=False, ewald_precision=1e-8)
scf = run_pbc_bipole_rhf(system, basis, kmesh, options, **scf_controls)
assert scf.converged
assert scf.resolved_domain.policy == "physical_pairs"
settings = scf.resolved_domain.as_dict()
print(scf.resolved_domain.settings_sha256)
```

The digest identifies settings, not geometry, basis, library build or accuracy.
The record also retains the requested lattice options. Drivers resolve private
copies, so the original `options` object can be reused without inheriting a
previous run's nuclear cutoff or screening mutations.

The named driver keyword takes precedence over
`options.lattice_opts.pair_complete_1e`, including an explicit False. To replay
an older BIPOLE index-ball result, use `historical_domain=True`. The generic
native option still defaults to False for other method families.

## 4. Carry the same controls into forces

Finite differences repeat SCF calculations. Reuse the options and driver
keywords from the reference calculation:

```python
from vibeqc.bipole_gradient import compute_bipole_gradient_fd

gradient = compute_bipole_gradient_fd(
    system, basis, kmesh, options, method="RHF", **scf_controls,
)
```

Do not execute this step as part of a source-only review: it requires multiple
numerical calculations. Each displaced SCF must converge. Check finite-domain
boundary crossings and finite-difference step convergence separately.
Direct BIPOLE optimizers accept the same `historical_domain` keyword; the runner
forwards `bipole_historical_domain` to optimization automatically.

For RKS/UKS, also retain the functional, all grid/profile/pruning controls,
periodic Becke selection and image radius. The analytic gradient remains a
restricted research preview: omitted controls restore the result's recorded
operator; incompatible explicit controls raise. Unsupported non-AUTO XC domains,
explicit nuclear quadratures and exact-FT J diagnostics must not silently fall
back to another analytic operator. SAP's all-space initial-guess quadrature is
separate from per-cell XC integration.

## 5. Assess a changed result

First require successful SCF termination and inspect its exact-operator residual.
Compare kinetic, nuclear attraction, Coulomb, exchange, XC and nuclear repulsion
at consistent densities and controls. Physical and historical support can retain
different terms at the same nominal radius, so equal total energies are not an
acceptance requirement. Neither a smaller residual nor agreement after loosening
a tolerance justifies changing a reference.

The [issue-to-code acceptance matrix](../bipole_domain_contract.md) lists the
prepared compact ionic/diffuse, non-Gamma, component-energy, nondefault-grid and
exact/incremental checks with their unchanged thresholds. The twelve-crystal
production matrix, full-stage performance profiles and slow-baseline reconciliation
remain separate evidence requirements. Wrong-answer issues stay open until
independent numerical verification, even when all source checks pass.
