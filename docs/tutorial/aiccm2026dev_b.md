# χ-CCM / aiccm2026dev-b: 3D backends and dimensional guards

See also the [user-guide reference](../user_guide/aiccm2026dev_b.md) for
the full keyword surface, backend selection, and scientific caveats.

This tutorial exercises only χ-CCM[^xccm-convention], the
finite-translation-group character CCM implemented under the
`aiccm2026dev-b` selector. Γ-CCM, the union-and-weight/Wigner--Seitz
integral-weighting sibling selected as `aiccm2026dev-a`, has its own
[page](../aiccm2026dev_a.md) and example. The inputs stay explicit so a future
construction-level comparison will not share either implementation's core;
the current reportable route map is empty.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

## Build a small neutral 3D cell

```python
import numpy as np
import vibeqc as vq

def h2_cell():
    lattice = np.diag([8.0, 12.0, 12.0])
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis
```

Lengths are in bohr. Current χ-CCM-B SCF is enabled only for 3D periodicity.
True 1D wire and 2D slab absolute energies fail closed until one shared
mixed-boundary Coulomb gauge is implemented. A chain-like visualization test
can still be built as a 3D vacuum-padded torus; that is not a 1D Coulomb
model.

## Check the fitted-route baseline

D99 pins three inherited behaviors at the B selector. Since `ddbe859d1`, all
supported 3D Gamma-only fitted paths and all multi-k RI and RIJCOSX restricted
or open-shell routes use `ewald_nuclear_repulsion` instead of
`nuclear_repulsion_per_cell`. This establishes routing only. D102 records the
historical shifted-pair cutoff defect in that shared helper. D104 binds the
pair-complete repair at `e578b86c0` to a producer-side ancestry check, a
same-process MgO basis-shift canary, and finalized D77 identity. Old D102 v1
records remain permanently revision-bound; a fresh exact-v2 record may clear
only this shared-Ewald gate. Gamma-only here is the one-character χ evaluation
path, not Γ-CCM construction identity.

Compact dense-core MDF calculations in the MgO/STO-3G class fail closed since
`6ce142339`; the validated vacuum-padded MDF envelope is unchanged. Every
fitted B row nevertheless remains D93 `not-qualified`. The bounded shared-q
auxiliary Fourier-transform cache inherited from `ad96a2f63` is bit-identical
and changes memory/performance only. Every fitted B dispatch also pins
`ibz_native=False` and `compute_gradient=False`, preserving the full unreduced
character mesh and total-gradient fail-close independently of generic driver
defaults. These changes do not open 1D/2D SCF, total gradients, or alter D83,
D88, or D98.

D103 separately gates direct overlap-fold support. Corrected-gauge
four-center execution stops before the first Fock build above `1e-2` drift;
moderate `1e-4 < drift <= 1e-2` still runs but is not quantitative. Fleet
qualification requires the exact recorded maximum drift to be at or below
`1e-4`, the diagnostic to compare the base-cutoff Bloch overlap fold with its
`1.5`-times-cutoff reference over the executed character mesh, and exact
cutoff and mesh matches. Fitted and post-HF routes record this field as exactly
`not-applicable`. This is independent of D93. D104 does not satisfy either
D93 or D103. A fresh direct row needs exact-v2 shared-Ewald evidence plus its
route-specific D93, D98, and D103 evidence; every old D102 v1 row remains
quarantined. Fitted and post-HF rows remain D93 `not-qualified` even when
D104 passes.

## Run 3D RHF and Kohn--Sham DFT

```python
system, basis = h2_cell()
for backend in ("four_center", "ri", "rijcosx"):
    rhf = vq.run_aiccm2026dev_b_rhf(
        system, basis, lattice_extension=(2, 1, 1),
        backend=backend, progress=False
    )
    rks = vq.run_aiccm2026dev_b_rks(
        system, basis, "pbe0", lattice_extension=(2, 1, 1),
        backend=backend, progress=False
    )
    print(
        backend,
        rhf.converged,
        rks.converged,
        rhf.aiccm2026dev_b.density_idempotency_error,
        rks.aiccm2026dev_b.density_idempotency_error,
    )
```

This checks route execution and finite-character invariants. This interactive
snippet does not create the finalized D77 and exact support record required
for an external absolute-energy claim, so do not quote its totals. Historical
D102 v1 totals remain permanently invalid; fresh fleet records are evaluated
under D104, D93, D98, and D103 independently.

`rijcosx` means RI-J plus COSX-K. It therefore exercises COSX in RHF and in a
hybrid such as PBE0. For a pure functional such as PBE, RI and RIJCOSX have the
same exchange content because there is no exact-exchange K term.

## Inspect the invariants

```python
d = rhf.aiccm2026dev_b
assert d.wigner_seitz_partition_error < 1e-12
assert d.density_idempotency_error < 1e-7
assert d.electron_count_error < 1e-7
print(d.mesh, d.backend, d.inverse_bloch_imaginary_residual)
print(d.coulomb_kernel, d.exchange_q0, d.boundary_model)
```

`d.mesh` is retained for compatibility; `d.lattice_extension` is the primary
name. To ask for interactions through two primitive translations on both
sides of each atom, use `wigner_seitz_shells=2`. This selects the odd
extension `(5, 5, 5)` in 3D and derives the equivalent reciprocal character
net automatically. Do not set a separate k mesh.

Do not validate the method by convergence alone. Repeat with increasing cyclic
meshes and a matched reciprocal-space reference. No 1D/2D χ-CCM-B absolute
energy is currently defined: every backend fails closed until one shared
wire/slab Coulomb Hamiltonian is derived.

## Confirm the lower-dimensional fail-close

Lower-dimensional inputs are guard coverage, not energy calculations. This
check must print six expected failures and no energy:

```python
for dim, lattice in (
    (1, np.diag([8.0, 20.0, 20.0])),
    (2, np.diag([8.0, 8.0, 20.0])),
):
    lowd_system = vq.PeriodicSystem(
        dim,
        lattice,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
    )
    lowd_basis = vq.BasisSet(lowd_system.unit_cell_molecule(), "sto-3g")
    for backend in ("four_center", "ri", "rijcosx"):
        try:
            vq.run_aiccm2026dev_b_rhf(
                lowd_system,
                lowd_basis,
                lattice_extension=(2, 1, 1),
                backend=backend,
                progress=False,
            )
        except NotImplementedError as exc:
            print(f"{dim}D {backend} expected fail-close: {exc}")
        else:
            raise AssertionError(
                f"{dim}D χ-CCM-B {backend} unexpectedly returned an energy"
            )
```

The convention fields are part of the numerical result, not prose decoration.
For the current production route they read `coulomb_kernel="3d-periodic-g0"`
and `exchange_q0="bvk-ewald"`. A strict-zero-mode exchange reference changes
the finite-N HF eigenvalues and therefore the MP2 or CC denominators, even
though both choices target the same infinite-crystal limit.

## Open the validated QVF fixtures

The docs include two small `aiccm2026dev-b` QVF archives that were rendered
with vibe-view as periodic visualization checks:

They were generated at pre-D89 source `d746d238` (shown as `d746d23` in the
archived output). They are visualization-only fixtures and lack the normative
D89 approach identity fields. Do not use them as construction-comparison or
numerical-validation evidence.

```bash
vibe-view open docs/_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.qvf
vibe-view open docs/_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf
```

The first is a 3D vacuum-padded H-chain with `aiccm_lattice_extension=(4,1,1)`; the
second is a 3D H2-pair with `aiccm_lattice_extension=(2,1,1)`. Each archive
contains the full BvK display cell, the density grid, HOMO/LUMO volume
sections, DOS/PDOS, the finite-torus convention vendor section, and an
`x_ccm.wannier_centers` overlay. The published inputs, verbose `.out` logs,
sanitized `.system` manifests, QVFs, and PNG captures are summarized in the
{download}`chi-CCM-B fixture README <../_static/examples/chi-ccm-b-qvf/README.md>`.

## Add canonical RI-MP2 in 3D

```python
system, basis = h2_cell()
mp2 = vq.run_aiccm2026dev_b_mp2(
    system,
    basis,
    (2, 1, 1),
    progress=False,
)
print(mp2.e_hf_per_cell)
print(mp2.e_corr_ss_per_cell, mp2.e_corr_os_per_cell)
print(mp2.e_total_per_cell)
assert mp2.momentum_conservation_error < 1e-12
assert mp2.max_energy_imaginary_residual < 1e-12
```

The three-center factors are expressed in the original auxiliary basis before
opposite momentum transfers are contracted. This removes arbitrary
eigenvector phases between the (q) and (-q) metric factorizations. MP2 is
not variational; its correctness gates are the finite-group momentum rule,
real energy, size normalization, and reciprocal-space KMP2 parity.
Check `mp2.finite_torus_convention == mp2.hf_result.aiccm2026dev_b.finite_torus_convention`
before comparing against an external KMP2 or CC calculation.

## Add local-PNO MP2 and CCSD(T)

```python
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
from vibeqc.dlpno.mp2 import DLPNOMP2Options

local_mp2 = vq.run_aiccm2026dev_b_dlpno_mp2(
    system,
    basis,
    (2, 1, 1),
    dlpno_options=DLPNOMP2Options(
        localise="wannier",
        tcut_pno=0.0,
        tcut_pno_weak=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        tcut_pairs_weak=0.0,
    ),
    progress=False,
)

local_cc = vq.run_aiccm2026dev_b_dlpno_ccsd_t(
    system,
    basis,
    (2, 1, 1),
    cc_options=LocalCCSDOptions(
        localise="pipek-mezey",
        tcut_pairs=0.0,
        coupling_radius=0.0,
        compute_triples=True,
        triples_mode="t1",
    ),
    progress=False,
)
canonical_cc = vq.run_aiccm2026dev_b_ccsd_t(
    system,
    basis,
    lattice_extension=(2, 1, 1),
    progress=False,
)
print(local_mp2.e_corr_per_cell)
print(local_mp2.raw_local_e_corr_per_cell)
print(local_mp2.complete_space_correction_per_cell)
print(local_mp2.local_correlation_space)
print(local_cc.e_corr_per_cell, local_cc.e_t_per_cell)
print(canonical_cc.e_corr_per_cell, canonical_cc.e_t_per_cell)
```

The real-torus transform is exact. `localise="wannier"` uses the B-only
finite-torus occupied gauge described in the
[localization tutorial](aiccm2026dev_b_localization.md). The returned local
space reports the exact-limit PAO rank and candidate translation pair orbits.
It does not skip equivalent pairs yet. IAO pair propagation is gated because
its explicit-supercell cross overlap is not exactly translation covariant.

At complete domains and zero PNO thresholds, the MP2 wrapper independently
evaluates a full-space, rotation-invariant DF-MP2 contraction. The raw local
solver value and `complete_space_correction_per_cell` remain visible; the
reported total recovers the canonical finite-torus limit. Truncated PNO
calculations receive no correction. Pipek--Mezey remains available, while
molecular Boys localization is rejected.

For the exact-limit diagnostic, use `localise="none"`, set `tcut_pno` and
`tcut_mkn` to zero, keep `tcut_pairs=0` and `coupling_radius=0`, and select
`triples_mode="exact"`. This is a validation calculation, not a scalable
production setting. The LiH example has a nonzero triples term and is pinned
against a separately contracted canonical finite-torus DF-CCSD(T) oracle.
The localized-gauge CC exact-limit audit is still open, so the CC example
continues to use Pipek--Mezey rather than claiming Wannier-gauge parity.

## Open-shell HF, KS, and correlation

```python
li = vq.PeriodicSystem(
    3,
    np.eye(3) * 15.0,
    [vq.Atom(3, [0.0, 0.0, 0.0])],
    multiplicity=2,
)
li_basis = vq.BasisSet(li.unit_cell_molecule(), "sto-3g")

uhf = vq.run_aiccm2026dev_b_uhf(
    li, li_basis, lattice_extension=(1, 1, 1),
    backend="ri", progress=False,
)
uks = vq.run_aiccm2026dev_b_uks(
    li, li_basis, "lda", lattice_extension=(1, 1, 1),
    backend="ri", progress=False,
)
ump2 = vq.run_aiccm2026dev_b_ump2(
    li, li_basis, lattice_extension=(1, 1, 1), progress=False,
)
uccsdt = vq.run_aiccm2026dev_b_uccsd_t(
    li, li_basis, lattice_extension=(1, 1, 1), progress=False,
)
local_uccsdt = vq.run_aiccm2026dev_b_dlpno_uccsd_t(
    li, li_basis, lattice_extension=(1, 1, 1), progress=False,
)

properties = vq.derive_aiccm2026dev_b_scf_properties(uhf, li, li_basis)
print(properties.n_alpha, properties.n_beta, properties.s_squared)
print(properties.mulliken_spin_populations)
# Diagnostic only: post-HF support remains D93 not-qualified.
print(ump2.energy, uccsdt.energy, local_uccsdt.energy)
```

The one-cell Li example is a code-path check, not a converged crystal. Increase
the real-space extension and inspect the localization aliasing flags before
using local domains. Canonical UMP2 uses zero PNO truncation. Canonical UCCSD
and UCCSD(T) use the full-domain O(N^6) pilot and are deliberately capped by
`max_nbf`; the current local sibling is the PNO projection oracle, not yet the
representative-pair reduced-scaling production algorithm.

## Run the maintained example

```bash
python examples/periodic/aiccm2026dev_b_demo.py --dim 3 --backend four_center
python examples/periodic/aiccm2026dev_b_demo.py --dim 3 --backend ri
python examples/periodic/aiccm2026dev_b_demo.py --dim 3 --backend rijcosx
python examples/periodic/aiccm2026dev_b_mp2.py
python examples/periodic/aiccm2026dev_b_local_correlation.py

# Expected fail-closed guard checks; these print no energy.
python examples/periodic/aiccm2026dev_b_demo.py --dim 1 --backend ri
python examples/periodic/aiccm2026dev_b_demo.py --dim 2 --backend rijcosx
```

All 1D/2D B SCF backends intentionally raise before SCF. Do not use a
lower-dimensional demo invocation as an energy calculation until one shared
wire/slab Coulomb convention is derived.

For a historical H4 side-by-side route diagnostic, run
`examples/regression/benchmark_aiccm2026dev_b.py`. It explicitly reports the
Γ-CCM/χ-CCM approach comparison as `not-defined` and emits no approach delta.

For the B/CRYSTAL fleet and future cross-approach study, generate the
χ-CCM-owned inputs from the shared system registry. The current
Γ-CCM/χ-CCM approach-comparison status remains `not-defined`:

```bash
python studies/aiccm-2026/make_jobs_b.py --profile coverage
```

This command is a preview; do not submit the coverage profile as a validation
campaign. Its ten runnable jobs are fitted RHF/RKS or fitted post-HF routes
and remain D93 `not-qualified`, even though fresh producers can now pass the
D104 shared-Ewald check. The profile also records 18 1D/2D rows as expected
fail-closed coverage and blocks three four-center LiH/STO-3G rows at their
measured 15-bohr overlap-fold drift of `7.0985e-2`. The larger `scf`,
`posthf`, `paper`, and `full` profiles are documented in
[`studies/aiccm-2026/README_B.md`](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README_B.md).
