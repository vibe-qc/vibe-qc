# Symmetry capabilities and production-reduction status

The shared mathematical infrastructure is implemented and experimental.
Production reduction is qualified separately for each numerical method,
finite support, electronic state and retained space. There is no blanket
completion or production-authorization claim for symmetry reduction.

## Shared infrastructure

| Capability | Implementation and boundary |
| --- | --- |
| Finite groups and lattice image cocycles | Native exact algebra checks, scalar antiunitary parity and point-group conjugacy classes are available. A valid table does not establish geometric completeness. |
| Compact space actions and orbit operations | Budgeted block actions, typed density/operator transport, stabilizers and adjoint scatter/gather are available without dense all-space matrices. |
| Molecular and periodic AO transport | Molecular compact actions support mixed pure/Cartesian shells through `l=6`, sharing the periodic polynomial kernel. Periodic AO/Bloch transport retains atom images, fractional translations, mesh wraps and scalar time reversal. The legacy dense molecular API requires pure shells beyond s. |
| State and retained-space diagnostics | Metric, group-law, operator and leakage checks can bind a native periodic state and selected spaces. A successful numerical witness does not certify a producer or authorize pair skipping. |
| Production method integration | An experimental SR/LR GDF retained-auxiliary tensor diagnostic compares independently built batches. Production GDF scheduling, PAO/PNO/TNO transport, reduced correlation execution and analytic derivatives retain separate implementation and acceptance gates. |
| Multi-k GDF exchange wedge | `symmetry_reduce_k` (driver `ibz_native`) builds exchange at wedge bras only and admits each transporting operation against the executed exchange source before any integral: every space-group operation, glides and screws included, on the pair-centred SR/LR fit; only equal-lattice-shift operations (the image shift computed with the fractional translation) on the origin-centred classic fit; none on other sources. A fractional translation is carried by the atom-image shift of the Bloch transport, with no separate `exp(-i k.tau)` factor in the AO-centred convention. Transported and directly built exchange agree to roundoff on nonsymmetric complex trial densities for diamond Fd-3m and hcp P6_3/mmc. Occupations are filled once per star on the wedge representatives, with levels split within the measured star spectral spread sharing their electrons, so a star-degenerate zero-temperature frontier receives the symmetric ensemble fill instead of a noise-selected one (#428). Cartesian shells beyond s, a proof of electronic-state invariance (the SCF gate is a calibrated threshold) and floating-point membership of pairs lying exactly on a cutoff surface remain outside this admission. |

The [developer reference](developer_shared_symmetry.md) defines the implemented
APIs and their resource contracts. The [architecture page](design_shared_symmetry.md)
describes the remaining production gates. All shared-layer evidence retains
`production_reduction_authorized == False`; a method must establish its own
admission policy against an unreduced reference. Independent acceptance of a
particular change belongs to its tracker record, not to a general capability
checkmark.

## Historical reduction milestones

The following inventory records the original implementation milestones as of
2026-05-23. It is retained to explain module names and legacy APIs. It is not a
current all-method validation matrix, and its checks do not override the
limitations below. In particular, force symmetrization is not an analytic
gradient implementation, and non-symmorphic transport support does not qualify
every finite integral domain or reconstruction path.

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

## Legacy reduction limitations

1. **M3b: origin-fixed cells only.** `compute_jk_gamma_reduced` works for single-atom
   primitive cells (Mg, He) but not multi-atom cells (NaCl), per-atom lattice
   shifts break cell-pair Wigner-D reconstruction. The historical single-atom
   checks do not establish a general multi-atom or multi-k reduction.

2. **M2a: S/T template workaround.** The GDF driver computes full S/T as a template
   when using the symmetry-reduced path, because pybind11's LatticeMatrixSet has
   read-only nbf/cells/blocks. The ERI compute reduction is unaffected.

## Module map

| Module | Purpose |
|---|---|
| `symmetry_shared.py`, native `symmetry_shared` | Shared group, space, orbit and qualification contracts |
| `symmetry_core.py` | Real Wigner rotations, including stable near-pole extraction |
| `symmetry_ao.py` | Compact pure/Cartesian molecular actions and legacy dense pure-shell matrices |
| native `periodic_ao_bloch_transport`, `periodic_orbital_sewing` | Periodic AO/Bloch transport and state/retained-space audits |
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

## Native legacy entry points

| File | Addition |
|---|---|
| `lattice_integrals.cpp` | `compute_1e_lattice_matrix_explicit` + wrappers |
| `periodic_fock.cpp` | `build_jk_gamma_molecular_limit_explicit`, `build_jk_2e_real_space_explicit`, `build_jk_pair_contributions` |
| `bindings.cpp` | pybind11 for all above |
