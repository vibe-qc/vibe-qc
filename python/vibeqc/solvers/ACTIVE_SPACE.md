# Active-space convention for non-HF solvers

The `active_space=(n_active, n_elec)` kwarg in `run_job` restricts a
solver to a **complete active space** `CAS(n_elec, n_active)`: `n_elec`
electrons distributed over `n_active` spatial orbitals, with the
remaining electrons frozen in a doubly-occupied inactive core.  This is
the standard CAS partition used by `mcscf.CASCI` and is shared by
**every** non-HF solver in this package.

## Semantics

Given the full MO-basis Hamiltonian (`norb` orbitals, `nelec`
electrons), `active_space=(n_active, n_elec)` defines three orbital
groups, ordered by MO energy:

- **Inactive core** — the lowest `n_core = (nelec − n_elec) // 2`
  orbitals, kept doubly occupied. `nelec − n_elec` must be even and
  non-negative.
- **Active** — the next `n_active` orbitals, over which the solver does
  its full many-determinant treatment with `n_elec` electrons.
- **Virtual** — the remaining `norb − n_core − n_active` orbitals,
  unoccupied and dropped from the correlation treatment.

Spin: the active electrons carry the molecule's `ms2 = 2·S_z` (the
doubly-occupied core is spin-neutral). For a closed-shell singlet
`ms2 = 0`; for an open-shell reference the unpaired electrons live in
the active window.

## Implementation

The truncation is a single call on the full MO-basis `Hamiltonian`:

```python
H_full = build_hamiltonian_mo(molecule, basis, C)   # all MOs
H_cas  = H_full.active_space(n_active, n_elec)       # CAS(n_elec, n_active)
```

`Hamiltonian.active_space` (in `_common.py`) returns a Hamiltonian over
the active orbital block that folds the inactive electrons back in
**exactly** (it reuses the validated `_casci._frozen_core_dressing`):

- `h1e` — the active block dressed by the inactive mean field
  (physicist's `g_{pqrs} = <pq|rs>`):

  ```
  h̃_pq = h_pq + Σ_c (2 g_{pcqc} − g_{pccq})        (p, q active)
  ```

- `h2e` — the active–active–active–active sub-tensor.
- `nuclear_repulsion` — the original `E_nuc` **plus** the constant
  inactive energy

  ```
  E_core = 2 Σ_c h_cc + Σ_cd (2 g_{cdcd} − g_{cddc})
  ```

so that `E_total = (active CI eigenvalue) + nuclear_repulsion`
reproduces the untruncated CASCI total energy.

## Correctness

The frozen-core dressing is validated two ways:

- **Δ = 0 against an explicit CAS-determinant FCI over the full orbital
  space** — building the same CAS as full-orbital determinants (core
  always doubly occupied) and diagonalising with the *full* integrals
  gives the identical energy to the dressed active-space CI. Covered by
  `tests/test_solvers_active_space_api.py` via the variational sandwich
  and the `fci`/`casci` cross-method agreement.
- **`active_space=(norb, nelec)` is the exact identity** — no inactive
  core (`E_core = 0`), integrals unchanged. Covered by
  `tests/test_solvers_integration.py`.

The variational sandwich `E_FCI(full) ≤ E_CAS(n_active, n_elec) ≤ E_HF`
holds for any geometry/basis: the HF determinant is a member of any CAS
that includes the frontier orbitals, and full FCI is the global minimum.
Energy *differences* at fixed `n_active` (geometries, methods on the same
window) were always meaningful; absolute energies now are too.

> **History.** Before the dressing landed, the runner sliced the MO
> coefficient matrix to a bare orbital block and dropped both the `h1e`
> dressing and the `E_core` offset, so `active_space=(n, k)` with
> `n < norb` reported only the active-only contribution (e.g. ≈ −15.5 Ha
> for CAS(4,4)/H₂O instead of below the −74.97 Ha HF total). That gap was
> pinned by strict `xfail` in `test_solvers_active_space_api.py`; the
> xfails are now positive assertions.

## Supported methods

All determinant / RDM solvers consume the dressed CAS Hamiltonian through
`Hamiltonian.active_space`:

- `method="fci"`
- `method="selected_ci"`
- `method="cisd"`
- `method="dmrg"` (subject to ≤6 spatial orbitals)
- `method="v2rdm"`
- `method="transcorrelated_ci"`

`method="casci"`, `"casscf"`, `"nevpt2"`, `"caspt2"` use the **same** CAS
partition and the **same** `_frozen_core_dressing`, reached through
`casci()`'s own frozen-core path (which additionally supports an explicit
`active_orbitals` window and the C++ direct CI backend). Because the
dressing is shared, `run_job(method="fci", active_space=(n, k))` and
`run_job(method="casci", active_space=(n, k))` agree to numerical
precision on the same reference orbitals. `"casscf"` additionally
optimises the orbitals; `"caspt2"` is the internally-contracted variant,
validated against OpenMolcas `&CASPT2` to ≤2 µHa and **un-gated**, including
the canonical eight-class single-state IPEA contraction. The
strongly-contracted variant `caspt2(variant="sc")` stays gated behind
`VIBEQC_EXPERIMENTAL_MULTIREF=1`. See
[`handovers/HANDOVER_MULTIREF.md`](../../../handovers/HANDOVER_MULTIREF.md).

## Example

```python
run_job(mol, basis="sto-3g", method="selected_ci",
        active_space=(4, 4))  # 4 active orbitals, 4 active electrons
```
