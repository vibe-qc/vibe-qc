# Integrals first: the public integral primitives and their backends

Build the integrals and other low-level primitives before the methods
that depend on them. A method cannot run at production scale on a
primitive that has no efficient backend, so the primitive comes first.
The maintainer set this as project policy on 2026-05-29:

> *"Always integrals and these low-level things first, otherwise how
> shall the method ever work?"*

The policy exists because the opposite order kept costing releases. The
failure recurred in the same shape: a method chat lands a Python-level
driver that exercises an integral primitive at production scale before
that primitive has a C++ backend; the run exhausts memory or slows to a
crawl; the release slips; an integrals chat is then spawned to backfill
the C++. The GDF density-fitting work slipped four cycles this way, from
v0.8.0 through v0.11.0, before the missing primitive was finally ported.

This page is the catalogue that breaks the pattern. It lists every
public Python integral primitive vibe-qc exposes, records whether its
kernel is C++, Python, or hybrid, and flags whether real workloads push
it at scale. Before landing scale-sensitive code, a method chat can
check here whether its driver already has a C++ backend underneath it,
instead of finding out when a production run falls over.

**Ownership.** The release chat refreshes the catalogue at every tag
(see [`release_process.md`](release_process.md)). The integrals chat
owns the implementation column: whether each primitive is C++, Python,
or hybrid.

## How to read the table

* **Primitive**: the name a Python caller sees. Most are re-exported
  from `vibeqc/_vibeqc_core` (the pybind11 module) via
  `python/vibeqc/__init__.py`; some are pure-Python wrappers around
  lower-level primitives.
* **Impl**: *C++* if the kernel work is C++ (with pybind11 glue);
  *Python* if the kernel work is NumPy / Python; *hybrid* if the Python
  side performs non-trivial assembly on top of C++ kernels (typical for
  the density-fitting builders).
* **Hot in production?**: whether typical user workloads exercise this
  at scale (i.e. with `n_orb` × `n_aux` × `n_G` or `n_cells` large
  enough that a Python loop on top would dominate wall-clock or memory).
  A *yes* here paired with *Python* in **Impl** is the "land a C++ port
  before the next method chat lands a driver on top" warning the
  catalogue exists to surface.
* **Status / SHA**: the commit that landed the listed implementation,
  or the open follow-up task if a primitive is Python-only and hot.

## Molecular integrals

| Primitive | Impl | Hot in production? | Status / SHA |
|---|---|---|---|
| `compute_overlap(basis)` | C++ | yes, every SCF | shipped; libint `Operator::overlap` |
| `compute_kinetic(basis)` | C++ | yes, every SCF | shipped; libint `Operator::kinetic` |
| `compute_nuclear(basis, molecule)` | C++ | yes, every SCF | shipped; libint `Operator::nuclear` |
| `compute_nuclear_with_charges(basis, positions, charges)` | C++ | yes, solvation and ECP | shipped in commit `39bf8629` (ECP+CPCM) |
| `compute_dipole(basis, origin)` | C++ | yes, dipole moments and property analysis | shipped; libint `Operator::emultipole1` |
| `compute_eri(basis)` | C++ | yes, direct-SCF backends | shipped; dense 4-index tensor builder |
| `compute_2c_eri(aux)` | C++ | yes, DF builders | shipped; libint `BraKet::xs_xs` |
| `compute_3c_eri(orbital, aux)` | C++ | yes, DF builders | shipped; libint `BraKet::xs_xx` |
| `compute_ecp_matrix(basis, ecp_centers, ...)` | C++ | yes, ECP-aware SCF | shipped; libecpint wrapper |

## Periodic integrals: real-space lattice sums

| Primitive | Impl | Hot in production? | Status / SHA |
|---|---|---|---|
| `compute_overlap_lattice(basis, system, opts)` | C++ | yes, every periodic SCF | shipped |
| `compute_kinetic_lattice(basis, system, opts)` | C++ | yes, every periodic SCF | shipped |
| `compute_nuclear_lattice(basis, system, opts)` | C++ | yes, every periodic SCF | shipped |
| `compute_nuclear_lattice_ewald(basis, system, opts, ...)` | C++ | yes, Ewald V_ne | shipped |
| `compute_nuclear_erfc_lattice(basis, system, opts, omega)` | C++ | yes, Ewald V_ne short-range half | shipped |
| `compute_2c_eri_lattice(aux, system, opts)` | C++ | yes, periodic DF | shipped (see `cpp/include/vibeqc/aux_eri.hpp`) |
| `compute_2c_eri_lattice_blocks(aux, system, opts)` | C++ | yes, multi-k DF storage | shipped |
| `compute_3c_eri_lattice(orbital, aux, system, opts)` | C++ | yes, periodic DF | shipped |
| `compute_3c_eri_lattice_blocks(orbital, aux, system, opts)` | C++ | yes, multi-k DF storage | shipped |
| `compute_2c_eri_lattice_sr(aux, system, opts, omega)` | C++ | yes, RSGDF SR half | shipped (Ye & Berkelbach 2021) |
| `compute_3c_eri_lattice_sr(orbital, aux, system, opts, omega)` | C++ | yes, RSGDF SR half | shipped |
| `direct_lattice_cells(system, cutoff_bohr)` | C++ | yes, every lattice sum | shipped |
| `ewald_nuclear_potential(...)`, `ewald_nuclear_repulsion(...)` | C++ | yes, every periodic SCF | shipped |
| `multipole_moments_lattice(...)` | C++ | yes, periodic multipoles | shipped |

## Periodic integrals: reciprocal-space (G-mesh) primitives

This is the column that was Python-only and hot at the start of the
v0.11.0 cycle: the out-of-memory problem that motivated the catalogue.

| Primitive | Impl | Hot in production? | Status / SHA |
|---|---|---|---|
| `ao_pair_fourier_transform(basis, G)` | hybrid | yes, RSGDF dense-mesh 3c | Python loop (`_aopair_ft.py`) calls C++ Bloch wrapper at k = 0 and squeezes; for L > 0 walks the MD recursion in Python only when basis is Cartesian / L > 6 |
| `ao_pair_fourier_transform_at_cells(basis, G, R_g_list)` | Python | **no longer hot**, bypassed by the C++ Bloch wrapper | retained as the reference path for parity testing |
| `ao_pair_fourier_transform_shifted_ket(basis, G, R_g)` | Python | low | reference / debug path |
| `ao_pair_fourier_transform_bloch(basis, G, R_g_list, k_cart)` | **C++** (since `74f3eb4a` + `590d022d`) | yes, RSGDF dense-mesh 3c, every k-point | C++ s-only fast path + general-L MD kernel; streams Bloch sum over R_g so the per-cell intermediate never materialises. **This was the OOM driver pre-v0.11.0.** |
| `rsgdf_aux_fourier_transform(aux, G)` | Python | yes, RSGDF 2c metric assembly | single-AO FT (one shell axis); the loop is over n_aux × n_G, which is modest and currently fits inside SCF budgets. Track for promotion if larger aux bases land |
| `rsgdf_dense_g_mesh(system, ke_cutoff)` | C++ (via `FFTW3` grid + filter) | yes | shipped |

## Density-fitting assembly: hybrids that compose the above

These are not "integrals" per se, but they exercise the integral
primitives and decide whether a Python loop sits between the C++ kernels
and the user driver. Worth tracking because the loop is where scaling
problems live.

| Primitive | Impl | Hot in production? | Status / SHA |
|---|---|---|---|
| `aux_basis.build_lpq_native_fft(system, ao_basis, aux_basis, ...)` | hybrid | yes, RSGDF dense-mesh path | dispatches to the C++ pair-FT kernel via `_aopair_ft.ao_pair_fourier_transform_bloch`; the remaining work is small (auxiliary FT contraction + eigendecompose-threshold) |
| `aux_basis.build_lpq_native(system, ao_basis, aux_basis, ...)` | hybrid | yes, compcell GDF path | C++ `compute_2c_eri_lattice` + `compute_3c_eri_lattice` + Python fitting / threshold |
| `aux_basis.build_lpq_bloch_native(system, ...)` | hybrid | yes, multi-k GDF | C++ cell-resolved 2c/3c blocks + Python Bloch phase assembly |
| `aux_basis.rsgdf_lr_2c_metric(...)` / `rsgdf_lr_3c_tensor(...)` | hybrid | yes, RSGDF LR halves | analytic FTs in Python; modest scale today |

## Functional / DFT primitives

Out of scope for the integrals catalogue; covered by libxc.

## How to use this catalogue

1. **Before landing a periodic driver**, find the primitives it calls.
   If any are Python-only and the driver pushes them at production scale
   (large n_orb, dense G-meshes, many k-points, large cell lists), pause
   and open an integrals task to port the primitive *before* shipping
   the driver.
2. **Per-release sweep.** The release chat re-validates this table at
   every tag. Add rows for any new primitive landed in the cycle; update
   implementation status where it changed; flag anything newly hot in
   production.
3. **CI-side guard.** See `tests/test_binding_sanity.py` for the
   mechanical check that the `.so` actually exposes everything
   `python/vibeqc/__init__.py` claims it imports.

## Footnote: v0.11.0 cycle close

The Python `ao_pair_fourier_transform_bloch` row was Python-only and hot
at the start of the v0.11.0 cycle; the C++ port landed in commits
`74f3eb4a` (s-shell tier) and `590d022d` (general-L). The table flip
from *Python* to *C++* in the corresponding row is the canonical "fix
the catalogue before the next defer cycle starts" example. The catalogue
itself lands in this cycle to make the pattern repeatable.
