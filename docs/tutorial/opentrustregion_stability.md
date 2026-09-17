---
myst:
  html_meta:
    "description": "Restart molecular UKS with OpenTrustRegion and follow a spin-breaking orbital instability of stretched H2 while checking electron counts, physical convergence and manifold-specific stability."
---

# OpenTrustRegion restarts and orbital stability

This tutorial uses two small calculations to distinguish restarting a
converged density from escaping an electronic saddle. First, restart a native
PBE water-cation calculation through OpenTrustRegion. Then start stretched H2
from a restricted determinant and let unrestricted orbital rotations lower
its energy. Both examples keep the nuclear geometry fixed.

Complete the capability check in
[Molecular SCF with OpenTrustRegion](opentrustregion.md) first. All coordinates
below are in bohr, all energies in Hartree, and all outputs go through
`run_job` into the chosen directory.

## 1. Restart an open-shell calculation

From the repository root:

```sh
otr_runs=$(mktemp -d)
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/02_restart_uks.py \
  --output-dir "$otr_runs/restart"
```

Download the complete input:
{download}`02_restart_uks.py <../../examples/opentrustregion/02_restart_uks.py>`.
It runs PBE/STO-3G for water cation with five alpha and four beta electrons,
then hands that result to the public READ interface:

```python
# seed is the converged UKS result; molecule and options() are defined
# in the downloadable input. Preserve the same functional and grid.
restarted = vq.run_job(
    molecule, basis="sto-3g", method="uks", functional="PBE",
    uks_options=options(), initial_guess="read", read_from=seed,
    orbital_optimizer="opentrustregion",
    output="uks-restarted", name_molecule=False, num_threads=1,
)
```

READ transfers the density; it does not resume the previous optimizer's
trust radius, history or iteration counters. The new backend builds a fresh
physical model. An idempotent integer-occupation density preserves its
occupied subspace at initialization, while a non-idempotent atomic guess is
projected to the requested integer ranks.

The script checks energy agreement within `2e-8` Ha and both spin densities
within `2e-6` elementwise. It also verifies electron counts with the AO
overlap matrix `S`:

```python
overlap = np.asarray(vq.compute_overlap(basis))
print(np.trace(restarted.density_alpha @ overlap))  # 5
print(np.trace(restarted.density_beta @ overlap))   # 4
```

The overlap is essential because the AO basis is nonorthogonal. Simply
summing the diagonal of a density matrix is not an electron-count check.
Expect only a small amount of accepted-state work from an already stationary
seed, but use the physical residuals and agreement checks as the success
criterion instead of a fixed iteration count.

The example uses an in-memory result to keep the electronic comparison
separate from file transport. For persistent QVF or TREXIO READ sources,
use the [initial-guess reference](../user_guide/initial_guess.md) and retain
the appropriate basis and geometry metadata.

## 2. Follow a spin-breaking saddle in stretched H2

At a four-bohr H-H separation, a restricted Hartree-Fock determinant is
stationary but can lower its energy when alpha and beta occupied orbitals
are allowed to differ. A vanishing orbital gradient alone cannot reveal
that downhill direction; the orbital Hessian has to be examined in the
larger unrestricted space.

```sh
OMP_NUM_THREADS=1 .venv/bin/python examples/opentrustregion/03_stretched_h2.py \
  --output-dir "$otr_runs/h2-follow"
```

Download the input:
{download}`03_stretched_h2.py <../../examples/opentrustregion/03_stretched_h2.py>`.
It first converges native RHF/STO-3G. It then builds a UHF READ guess with
one electron in each spin channel:

```python
uhf_options = vq.UHFOptions()
uhf_options.initial_guess = vq.InitialGuess.READ
uhf_options.read_density_alpha = np.asarray(rhf.density) / 2
uhf_options.read_density_beta = np.asarray(rhf.density) / 2
otr_options = vq.OpenTrustRegionOptions()
otr_options.stability = "follow"
```

The factor of two matters: the RHF density already includes both spins.
The script passes these objects as `uhf_options=` and
`opentrustregion_options=` to a UHF `run_job` call, with
`orbital_optimizer="opentrustregion"`.

Success means a converged UHF result, a lower energy by more than `0.05` Ha,
a converged stable verdict in the real unrestricted manifold, and
`Tr(D_alpha S) = Tr(D_beta S) = 1`. The input prints the actual energies,
energy lowering and spin expectation value. Swapping the alpha and beta
spatial densities produces an equivalent broken-symmetry solution.

For this exact input, a validation run produced:

```text
Restricted energy:    -0.761082247024 Ha
Unrestricted energy:  -0.935842328320 Ha
Energy lowering:      0.174760 Ha
Unrestricted <S^2>:   0.963992
Electron counts:      alpha=1, beta=1 (verified)
Stable manifold:      real unrestricted occupied-virtual
```

The result retains `M_S=0`, but the broken-symmetry UHF determinant is not
an eigenfunction of total spin. Its nonzero `<S^2>` is a physical limitation
of this approximation. An internally stable determinant is also not a
certificate of the global electronic minimum. For spin-pure dissociation,
compare against an appropriate multiconfigurational treatment separately;
OpenTrustRegion is not yet connected to CASSCF.

## 3. Understand the pinned library's stability policies

| Policy | Requested behavior |
|---|---|
| `none` | No additional final stability check |
| `check` | Check the final state without escaping a newly detected final mode |
| `follow` | Ask upstream to escape saddles during the solve, then check the final state |

There is an upstream exception to all three: the pinned version **always
checks and follows an initial stationary saddle**. Repeating the H2 example
with `check` or `none` can therefore still escape the starting restricted
state. Those options cannot preserve a chosen excited stationary state.

During saddle following, upstream uses its fixed `-0.01` curvature threshold.
The separate final check uses `-stability_tolerance`, `-1e-4` by default.
Consequently, a small negative mode can survive `follow` and still be
reported as unstable by the final check. An eigensolver that exhausts its
iteration cap produces an inconclusive verdict, not a stable one.

Read all three report fields together:

```python
report = result.opentrustregion
if not report.stability_checked:
    print("Final stability was not checked")
elif not report.stability_converged:
    print("Final stability is inconclusive")
elif report.stable:
    print("Stable within", report.manifold)
else:
    print("Negative curvature within", report.manifold)
```

The upstream C interface provides no minimum eigenvalue. Use these fields
instead of reading an eigenvalue from the legacy native-stability result.
Restricted checks do not include unrestricted spin breaking or complex
rotations; unrestricted checks here still cover real collinear orbitals only.

## 4. Choose follow-up checks

Try a shorter H-H distance and repeat both RHF and UHF. Look at the energy
lowering, spin expectation and stability verdict together. Keep the same
basis and tolerances, and preserve one alpha and one beta electron. Do not
interpret a failure to find a lower state as proof that no lower state exists.

For a difficult molecule, use separately chosen physical initial densities
and compare the converged states. Native `multi_guess_seeds` schedules are
explicitly rejected for this backend; no hidden native retry follows an
OpenTrustRegion failure. Diagnose the termination and physical residuals
before increasing iteration caps. See the
[reference troubleshooting table](../user_guide/opentrustregion.md#troubleshooting)
for unsupported response models and failure codes.
