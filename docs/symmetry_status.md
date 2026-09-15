# Symmetry Reduction Implementation, Status & Handover

Last updated: 2026-05-23 | Phase: complete, all milestones shipped

This is the historical status of the original reduction milestones. The
separate [shared symmetry infrastructure](developer_shared_symmetry.md)
workstream adds common group, space and diagnostic contracts. Its production
operator qualification, local-correlation reduction and derivative gates
remain open; the completion statement above does not cover that workstream.

## Milestone Status

| # | Milestone | Status |
|---|---|---|
| M1 | `symmetry=True` primitive reduction | ✅ |
| M2 | 1e compute reduction (S, T) | ✅ |
| M2a | Wire M2 into 5 drivers | ✅ |
| M3 | `build_jk_gamma_molecular_limit_explicit` | ✅ |
| M3b | `build_jk_2e_real_space_explicit` | ✅ |
| M3b | `build_jk_pair_contributions` (C++) | ✅ |
| M3b | `compute_jk_gamma_reduced` (Python) | ✅, origin-fixed cells only |
| M4 | IBZ-native SCF (4 drivers) | ✅ |
| M5 | `symmetry_scf.py` | ✅ |
| M5a | `symmetry_stabilize` in GDF | ✅ |
| SYM5 | `symmetrize_forces()` | ✅ |
| - | Non-symmorphic support | ✅, diamond Fd-3m validated |
| SYM6 | Character tables (Oh, Td, D4h, C6v) | ✅ |

## Known Limitations

1. **M3b: origin-fixed cells only.** `compute_jk_gamma_reduced` works for single-atom
   primitive cells (Mg, He) but not multi-atom cells (NaCl), per-atom lattice
   shifts break cell-pair Wigner-D reconstruction. Verified: Mg J err 2.7e-15.

2. **M2a: S/T template workaround.** The GDF driver computes full S/T as a template
   when using the symmetry-reduced path, because pybind11's LatticeMatrixSet has
   read-only nbf/cells/blocks. The ERI compute reduction is unaffected.

## Module Map

| Module | Purpose |
|---|---|
| `symmetry_core.py` | Wigner D-matrices (SYM1) |
| `symmetry_ao.py` | AO permutation matrices (SYM2a) |
| `symmetry_lattice.py` | Lattice-cell orbits (SYM2b) |
| `symmetry_lattice_c.py` | Atom-pair orbits (SYM2c) |
| `symmetry_integrals.py` | Storage-reduced 1e integrals (SYM3a) |
| `symmetry_integrals_reduced.py` | Compute-reduced 1e integrals (SYM3b) |
| `symmetry_fock_reduced.py` | Compute-reduced 2e Fock (M3b) |
| `symmetry_scf.py` | Group-averaging + force symmetrization (M5, SYM5) |
| `symmetry_salc.py` | Character tables + irrep projection (SYM6) |
| `periodic_runner.py` | `symmetry`/`reduce_to_primitive`/`symmetry_reduce_fock` kwargs |
| `periodic_rhf_gdf.py` | GDF driver with M2a/M5a/M3b integration |
| `pbc_bipole*.py` | BIPOLE drivers with M2a/M4 integration |

## C++ Changes

| File | Addition |
|---|---|
| `lattice_integrals.cpp` | `compute_1e_lattice_matrix_explicit` + wrappers |
| `periodic_fock.cpp` | `build_jk_gamma_molecular_limit_explicit`, `build_jk_2e_real_space_explicit`, `build_jk_pair_contributions` |
| `bindings.cpp` | pybind11 for all above |
