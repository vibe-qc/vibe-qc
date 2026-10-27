---
myst:
  html_meta:
    "description": "Multi-k periodic SCF in vibe-qc, native RHF/RKS Ewald drivers today, native FFTDF/GDF under development, PySCF and CRYSTAL as external references only."
    "og:title": "vibe-qc, multi-k periodic SCF"
    "og:description": "Current periodic multi-k SCF status, smearing for metals, and the native FFTDF/GDF roadmap."
---

# Multi-k periodic SCF

vibe-qc treats multi-k periodic SCF as a native feature, not as a
Python import wrapper around another quantum-chemistry code. PySCF and
CRYSTAL are reference programs: run them out-of-process, parse their
outputs, and compare energies, occupations, bands, and convergence
traces. They are not vibe-qc backends.

## Theory

Bloch's theorem turns the crystal into one Roothaan-Hall problem per crystal
momentum. From the Gaussian atomic orbitals $\chi_\mu$ of the home cell one
forms Bloch sums over the lattice translations $\mathbf T$,

$$
\phi_\mu^{\mathbf k}(\mathbf r) = \frac{1}{\sqrt{N_{\mathrm{cell}}}}\sum_{\mathbf T}
e^{i\mathbf k\cdot\mathbf T}\,\chi_\mu(\mathbf r - \mathbf T),
\qquad
\mathbf F(\mathbf k) = \sum_{g} e^{i\mathbf k\cdot\mathbf R_g}\,\mathbf F^{(g)},
\quad
\mathbf S(\mathbf k) = \sum_{g} e^{i\mathbf k\cdot\mathbf R_g}\,\mathbf S^{(g)},
$$

where $\mathbf F^{(g)}$ and $\mathbf S^{(g)}$ are the real-space lattice
blocks between the home cell and cell $g$ (`bloch_sum` applies exactly this
phase convention). At every sampled $\mathbf k$ the k-dependent Roothaan-Hall
equation $\mathbf F(\mathbf k)\mathbf C(\mathbf k) = \mathbf S(\mathbf k)\mathbf
C(\mathbf k)\boldsymbol\varepsilon(\mathbf k)$ is Hermitian and complex, and
the k points talk to each other only through the density,

$$
\mathbf D^{(g)} = \sum_{\mathbf k} w_{\mathbf k}\, e^{-i\mathbf k\cdot\mathbf R_g}
\sum_n f_{n\mathbf k}\,\mathbf C_n(\mathbf k)\,\mathbf C_n(\mathbf k)^\dagger,
\qquad \sum_{\mathbf k} w_{\mathbf k} = 1,
$$

with occupations $f_{n\mathbf k}$ (integer Aufbau, Fermi-Dirac at finite
$T$, or the Gilat-Raubenheimer fractions). The Brillouin-zone integral has
become a weighted sum over a Monkhorst-Pack mesh
({vibeqc-cite}`monkhorst_pack_1976`); on a symmetry-reduced mesh the weight of
an irreducible point is the size of its star divided by the full-mesh count,
$w_{\mathbf k} = |\mathrm{star}(\mathbf k)|/N_k$, and time reversal pairs
$\mathbf k$ with $-\mathbf k$. The mesh is the one approximation this section
introduces: an insulator's density is smooth in $\mathbf k$ and converges
exponentially, a metal's has a Fermi-surface discontinuity and needs dense
meshes plus [smearing](smearing.md) or the
[Gilat-Raubenheimer net](metallic_bz_integration.md).

On the GDF routes the two-electron terms are fitted in an auxiliary Gaussian
basis whose fitting quantities are themselves Bloch-summed
({vibeqc-cite}`sun_berkelbach_gdf_2017`). An orbital pair with bra momentum
$\mathbf k_i$ and ket momentum $\mathbf k_j$ carries the momentum transfer
$\mathbf q = \mathbf k_j - \mathbf k_i$, and the fitted three-index tensor is

$$
L_{P,\mu\nu}(\mathbf k_i, \mathbf k_j) = \sum_Q \big[\mathbf J(\mathbf q)^{-1/2}\big]_{PQ}\,
\big(Q_{\mathbf q}\,\big|\,\mu_{\mathbf k_i}\nu_{\mathbf k_j}\big),
\qquad
J_{\mu\nu}(\mathbf k) = \sum_P L_{P,\mu\nu}(\mathbf k,\mathbf k)\sum_{\mathbf k'} w_{\mathbf k'}
\sum_{\lambda\sigma} L^{*}_{P,\lambda\sigma}(\mathbf k',\mathbf k')\,D_{\lambda\sigma}(\mathbf k'),
$$

$$
K_{\mu\nu}(\mathbf k_i) = \sum_{\mathbf k_j} w_{\mathbf k_j}\sum_P\sum_{\lambda\sigma}
L_{P,\mu\lambda}(\mathbf k_i,\mathbf k_j)\,D_{\lambda\sigma}(\mathbf k_j)\,
L^{*}_{P,\nu\sigma}(\mathbf k_i,\mathbf k_j).
$$

The Coulomb matrix needs only $\mathbf q = 0$, one tensor for the whole mesh.
Exchange couples every $\mathbf k_i$ to every $\mathbf k_j$, so the cached
tensor is $n_k^2 \times n_{\mathrm{aux}} \times n_{\mathrm{ao}}^2$ complex
numbers: doubling a three-dimensional mesh along each axis multiplies the
$L_{pq}$ memory by 64, which is why multi-k GDF memory grows with the mesh
where the Gamma-only cost does not (`periodic_k_gdf` keeps it dense today, and
contracts exchange in the occupied-index form, $2 n_{\mathrm{aux}}
n_{\mathrm{ao}}^2 n_{\mathrm{occ}}$ per pair). Range-separated GDF
({vibeqc-cite}`ye_berkelbach_rsgdf_2021`, `gdf_method="rsgdf"`, the default)
splits the Coulomb kernel into an erfc part fitted analytically in real space
and an erf part evaluated on a reciprocal mesh, which removes the lattice-sum
truncation from the fit; `"compcell"` and `"mdf"` are the compensated-cell
and mixed Gaussian-plus-plane-wave fits described in
[periodic methods](periodic_methods.md#34-disadvantages).

Exact exchange on a finite mesh has one more term. The Coulomb kernel
$4\pi/|\mathbf q + \mathbf G|^2$ is integrable but singular at $\mathbf q +
\mathbf G = 0$, and a discrete mesh simply omits the singular point, so
$K(\mathbf k)$ carries an $\mathcal O(1/N_k)$ bias that does not vanish as the
mesh is refined at the rate one expects. The probe-charge correction of Gygi
and Baldereschi ({vibeqc-cite}`gygi_baldereschi_exx_1986`) adds back the
integral of the singularity over the missing volume, which in the Gaussian
basis is a Madelung shift of the exchange matrix,

$$
\mathbf K(\mathbf k) \;\to\; \mathbf K(\mathbf k) + \xi\,\mathbf S(\mathbf k)\,\mathbf D(\mathbf k)\,\mathbf S(\mathbf k),
$$

with $\xi$ the Ewald-Madelung constant of the Born-von Karman supercell
spanned by the mesh (`madelung_constant_for_cell`, applied per k by
`apply_exxdiv_ewald_to_K`; PySCF calls the same convention
`exxdiv='ewald'`). The Hartree term needs no such shift: its $\mathbf G = 0$
component cancels against the nuclear background in a neutral cell. Where
this machinery helps is every insulating or semiconducting crystal, where a
$4^3$ to $8^3$ mesh converges the energy to sub-millihartree; where it fails
is a dense metallic mesh on a large basis, where the $n_k^2$ exchange cache
exhausts memory before the Fermi surface is resolved, and that is the case
for smearing plus the cheapest route that still converges.

## Which mesh does `kpoints=(N,N,N)` mean?

:::{warning}
`kpoints=(N, N, N)` does **not** name the same set of k-points in every
code, and this is the single most expensive trap in cross-code periodic
comparison.

A plain mesh tuple, and {py:func}`vibeqc.monkhorst_pack`, build the
**Γ-centred** mesh `{0, 1/N, ..., (N-1)/N}` along each axis: Γ is always
sampled. `ase.dft.kpoints.monkhorst_pack` and GPAW build the **classical
shifted** mesh, offset by half a step.

{py:meth}`vibeqc.KPoints.monkhorst_pack` applies the classical auto-shift,
so it lands on the *other* convention at even N, agreeing with ASE where
the tuple does not. See [`k_points.md`](k_points.md) for that table. This
is why the `.out` reports the convention it actually resolved rather than
one inferred from the mesh you asked for.

The two **agree exactly for odd N** and are **disjoint for even N**
(measured 2026-08-02):

| mesh | k-points | shared with ASE |
|---|---|---|
| (2,2,2) | 8 | 0 |
| (3,3,3) | 27 | **27, identical** |
| (4,4,4) | 64 | 0 |
| (5,5,5) | 125 | **125, identical** |
| (6,6,6) | 216 | 0 |

For odd N the half-step offset wraps onto the same lattice; for even N it
lands exactly between vibe-qc's points. So a cross-code check validated at
(3,3,3) passes cleanly and tells you *nothing* about the (4,4,4)
production run beside it.

Every periodic run prints the convention and the resolved k-point count:

```text
    kpoints            = (2, 2, 2)
    k-mesh convention  = gamma-centred
                         Gamma is sampled; ASE/GPAW's mesh of
                         the same name is disjoint for even N
    k-points resolved  = 8
```

Set `VIBEQC_OUTPUT_LEVEL=verbose` to get the full fractional k-list, so
the sampling is reconstructable from the `.out` alone. To build the
shifted mesh instead, pass `is_shift` to
{py:func}`vibeqc.monkhorst_pack`.
:::

## Current Status

Native, usable today:

* Γ-only native GDF RHF, RKS and hybrids on a bulk cell through
  `run_periodic_job(jk_method="gdf", kpoints=(1, 1, 1))`, which runs the
  k-mesh engine at one point on the default SR/LR fit (or hands a
  `gdf_method="compcell"` / `"mdf"` request to the exact-Γ
  `run_pbc_gdf_rhf` when that driver implements every requested control;
  since #514 the delegation keeps the cutoff and AFT settings you passed).
  The legacy `run_rhf_periodic_gamma_gdf` serves only `dim=2` cells and
  3-D cells whose integral cutoff holds the home cell alone; a compact
  3-D cell is refused before any grid or integral work (#468):

  ```text
  NotImplementedError: run_rhf_periodic_gamma_gdf: compact 3D cells with
  image cells in the integral cutoff are unsupported by this public driver.
  Its Ewald-J fallback uses molecular-limit exchange, not periodic GDF
  exchange. Use run_periodic_job(jk_method='gdf', kpoints=(1, 1, 1)) for
  RHF/RKS, or run_pbc_gdf_rhf for a supported exact-Gamma RHF request.
  ```

  The same refusal is why `KPoints.recommend(classifier="pre-scf")`, which
  runs that legacy driver for a few cycles, cannot obtain a gap on a compact
  cell: it falls back to the SAFE metallic recommendation with a warning that
  names an unavailable classifier result, so pass `band_gap=` or `is_metal=`
  there. A `dim=1` wire raises `NotImplementedError` on every GDF entry
  point; model it as a `dim=3` cell with transverse vacuum.
* `run_krhf_periodic_gdf(..., kmesh, ...)` and
  `run_krks_periodic_gdf(..., kmesh, ...)` as the public k-point API
  for native GDF in 2D (closed-shell slabs) and 3D. A Γ-only mesh `(1, 1, 1)`
  delegates to the Γ-GDF implementation; **any other Monkhorst-Pack
  mesh runs the full native multi-k GDF loop**, k-dependent 3c/2c
  blocks, Bloch-phase assembly, multi-k DIIS. A `dim=1` wire with any other
  mesh raises `NotImplementedError`, because the route has no consistent
  1-D Coulomb gauge; run a wire that needs k sampling as a `dim=3` cell with
  vacuum on the two transverse axes. Every multi-k run builds
  the cached fit of its `gdf_method`: the `use_compcell=False` default
  is promoted to `use_compcell=True` on every method (for `"mdf"` and
  `"compcell"` with a pure functional only since 2026-09-29; before
  that those two kept the unfitted Ewald-gauge J and ignored the
  requested method). With `use_compcell=True` the multi-k GDF `Lpq` is
  built by the symmetry-preserving all-FT path resolved per
  `(k_i, k_j)` pair (`build_lpq_bloch_native_fft`) with a per-k Hartree
  J, per-pair exchange K, and a k-mesh-aware Madelung exchange shift,
  reaching **PySCF µHa parity** (LiH primitive FCC / sto-3g, kmesh
  (2,2,2): −1.4 µHa vs `pyscf.pbc.scf.KRHF.density_fit()` at a converged
  lattice cutoff; +1.6 mHa at the cheaper default; re-measured
  2026-08-05 against PySCF 2.13.1, run out of process). The AFT long-range
  correction is ON by default since v0.12.0 (η=1.0, libint convention,
  pyscf_auto rcut). HF/hybrid functionals are auto-routed to the
  exxdiv-corrected compcell path. Also reachable
  through `run_periodic_job(..., gdf_kmesh=(n, n, n))`.
* `compute_2c_eri_lattice_blocks(...)` and
  `compute_3c_eri_lattice_blocks(...)` expose translation-resolved
  periodic DF blocks in 1D, 2D, and 3D. Summing those blocks reproduces
  the Γ metric/tensor; the native multi-k GDF loop will apply Bloch
  phases to these blocks instead of rebuilding libint shell pairs.
  `bloch_sum_2c_eri_blocks(...)` and `bloch_sum_3c_eri_blocks(...)`
  pin that phase convention, and `build_lpq_bloch_native(...)` fits the
  phase-assembled tensor in the phase-assembled auxiliary metric.
* `run_rhf_periodic_scf(..., kmesh, opts)` with
  `opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D`.
* `run_rks_periodic_scf(..., kmesh, opts)` with the same EWALD_3D
  route.
* General lattice vectors: the code works with arbitrary 3D Bravais
  cells and uses the supplied lattice matrix to build reciprocal-space
  k-points and Bloch sums. Tests include skew/hexagonal cells so the
  implementation does not silently assume cubic axes.
* Closed-shell and per-spin finite-temperature Fermi-Dirac occupations for
  periodic multi-k Ewald RHF/RKS and UHF/UKS via
  `opts.smearing_temperature` (`k_B T` in Hartree). The standalone Ewald
  drivers also accept `bz_integration="gilat"` for parameter-free T=0
  occupations on gapped full or symmetry-reduced meshes. The native
  Γ-only GDF RHF/RKS driver and
  high-level `run_periodic_job(smearing_temperature=...)` support the
  same occupation model for Γ-point metal/debug work.
* Multi-k Ewald UHF/UKS report the spin diagnostic using all k points,
  their weights, complex alpha/beta orbital overlaps and the executed
  occupations. Arbitrary orbital phases therefore leave it unchanged.
  With smearing this is the occupation-weighted diagnostic shared by the
  periodic drivers, not a spin-projected state or a measure of pure-state
  spin contamination. Zero-iteration results use the supplied guess density.

Not native yet:

* Production-scale all-electron MgO/pob-TZVP at dense k-meshes needs
  the native FFTDF/GDF path. The legacy Ewald RKS path is useful for
  debug and correctness work, but its periodic XC memory footprint is
  intentionally guarded.

## Minimal RHF Example (Ewald-3D path)

```python
import numpy as np
import vibeqc as vq

a = 10.0
system = vq.PeriodicSystem(
    3,
    np.eye(3) * a,
    [vq.Atom(1, [0.0, 0.0, 0.0]),
     vq.Atom(1, [0.0, 0.0, 1.4])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(system, [2, 2, 2])

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.damping = 0.3
opts.use_diis = True

result = vq.run_rhf_periodic_scf(
    system, basis, kmesh, opts,
    omega=0.5,
    spacing_bohr=0.3,
)
print(result.energy, result.converged)
```

## Native GDF multi-k example

For production accuracy on ionic crystals, use the compcell GDF path:

```python
import numpy as np
import vibeqc as vq

# LiH primitive FCC (Sun-Berkelbach 2017 benchmark)
ANG = 1.0 / 0.529177210903
A_lat = 4.084
lat = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * A_lat * ANG
system = vq.PeriodicSystem(3, lat,
    [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [A_lat*ANG/2]*3)])
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

opts = vq.PeriodicRHFOptions()
opts.use_diis = True
opts.max_iter = 60
opts.conv_tol_energy = 1e-8
# For convergence testing, raise the lattice cutoff:
opts.lattice_opts.cutoff_bohr = 30.0
opts.lattice_opts.nuclear_cutoff_bohr = 30.0

# use_compcell=True enables the AFT-corrected compcell GDF path
# (AFT ON by default since v0.12.0; exxdiv auto-routed for HF/hybrids)
result = vq.run_krhf_periodic_gdf(
    system, basis, (2, 2, 2), opts,
    aux_basis="def2-svp-jk",
    use_compcell=True,
)
print(f"E = {result.energy:.8f} Ha, converged = {result.converged}")
# PySCF GDF reference: -7.92200323 Ha
# vibe-qc: -7.92039505 Ha at the default cutoff (+1.6 mHa),
#          -7.92200467 Ha at cutoff 30 (-1.4 uHa). Measured 2026-08-05.
```

`result.gdf_cderi_pairs` records the size of the density fit the SCF
contracted, counted on the fit the driver built: `n_k` `(k_i, k_j)` pairs
for a Coulomb-only fit (a pure functional), up to `n_k**2` with fitted
exact exchange (64 on the `(2, 2, 2)` mesh above), and `0` when no fit was
built. Read it, rather than the routing options, to
learn whether a run used a fit.

## Minimal RKS Example

```python
opts = vq.PeriodicKSOptions()
opts.functional = "pbe"
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.damping = 0.3
opts.use_diis = True

result = vq.run_rks_periodic_scf(
    system, basis, kmesh, opts,
    omega=0.5,
    spacing_bohr=0.3,
)
print(result.energy, result.e_xc, result.converged)
```

For MgO/pob-TZVP-class RKS, prefer this route only for small debug
meshes until native FFTDF/GDF lands; the full periodic XC grid can be
too memory-heavy for the legacy Ewald driver.

## Smearing For Metals

Closed-shell periodic RHF/RKS support Fermi-Dirac smearing on the
multi-k Ewald path and on the native Γ-only GDF path:

```python
opts.smearing_temperature = vq.resolve_smearing_temperature(
    1000.0,
    unit="kelvin",
).temperature
```

The option stores `k_B T` in Hartree. You can set it directly, convert
from Kelvin/eV/Ry, use a preset, or ask for a conservative automatic
guess:

```python
opts.smearing_temperature = vq.resolve_smearing_temperature("metal").temperature
opts.smearing_temperature = vq.resolve_smearing_temperature("0.1 eV").temperature
opts.smearing_temperature = vq.resolve_smearing_temperature(0.1, unit="eV").temperature
opts.smearing_temperature = vq.resolve_smearing_temperature(
    "auto",
    metallic=True,
).temperature
```

Numeric strings may carry their own unit (`"1000 K"`, `"0.1 eV"`,
`"0.01 Ha"`, `"0.02 Ry"`), which is often clearer in input files than a
bare Hartree number. `"auto"` is intentionally cautious: if no metallic
hint or prior band gap is supplied, it returns zero smearing rather than
silently changing an insulating calculation. With smearing enabled, the
drivers bisect the chemical potential so that the weighted k-mesh
occupation satisfies the per-cell electron count:

```text
sum_k w_k sum_i n_i(k) = N_electrons
n_i(k) = 2 / (1 + exp((eps_i(k) - mu) / T))
```

The result object reports:

* `result.energy`: internal total energy.
* `result.free_energy`: variational free energy `E - T S`.

* `result.entropy`: dimensionless electronic entropy `S/k_B`.
* `result.fermi_level`: chemical potential in Hartree.
* `result.occupations`: per-k occupation arrays in `[0, 2]`.

For finite-temperature runs, the iteration table reports `F` and `dF`.
The convergence summary labels both free energy `F` and internal energy `E`;
`result.energy` remains the internal energy. Their difference is the entropy
contribution `-T S`, not an SCF convergence error. At zero temperature the
ordinary energy and `dE` labels are retained.

`DIRECT_TRUNCATED` currently rejects positive
`smearing_temperature`; that backend would otherwise ignore the field
and run a hard-Aufbau calculation.

For the high-level native-GDF job runner, pass the same value as a
keyword:

```python
result = vq.run_periodic_job(
    system,
    basis,
    method="RKS",
    functional="pbe",
    smearing_temperature="auto",
    smearing_metallic=True,
)
```

## k-point mesh convergence for solids

Γ-only calculations are **not appropriate for real solids**. A single
k-point sees only the Γ-periodic component of the density; the full
Brillouin-zone integral must be sampled with a Monkhorst-Pack mesh.

The required mesh density depends on the crystal system:

### Mesh density by crystal type

| Crystal type | Typical primitive cell | Minimum mesh | Converged mesh |
|---|---:|---:|---:|
| Simple cubic (sc) | 1 atom, ∼3-5 Å | 4×4×4 | 8×8×8 |
| FCC (Cu, Al, Ag) | 1 atom, ∼3-4 Å | 6×6×6 | 12×12×12 |
| BCC (Fe, W, Na) | 1 atom, ∼3-4 Å | 6×6×6 | 12×12×12 |
| Rocksalt (MgO, NaCl) | 2 atoms, ∼4-6 Å | 4×4×4 | 8×8×8 |
| Diamond/Zincblende (Si, GaAs) | 2 atoms, ∼5-6 Å | 4×4×4 | 8×8×8 |
| Perovskite (BaTiO₃) | 5 atoms, ∼4 Å | 4×4×4 | 6×6×6 |
| Layered / 2D slab | variable | 6×6×1 | 12×12×1 |

**Rule of thumb**: the linear k-point density × lattice constant should be
≥ 20-30 Å for insulating systems. For metals, double it.

### Symmetry reduction

vibe-qc's `monkhorst_pack` applies time-reversal symmetry automatically.
Space-group symmetry reduction (IBZ) is available through `spglib`:

```python
vq.attach_symmetry(system)  # populates system.symmetry via spglib
kpts = vq.KPoints.monkhorst_pack(system, [8, 8, 8])                    # full mesh
kpts = vq.KPoints.monkhorst_pack(system, [8, 8, 8], symmetry=True)     # IBZ
```

Symmetry reduction can reduce the number of irreducible k-points by
factors of 2 to 48 depending on the space group, cutting SCF cost
proportionally. Examples on an 8×8×8 mesh:

| System | Space group | Full kpts | IBZ kpts | Factor |
|---|---:|---:|---:|
| MgO rocksalt (primitive cell) | Fm-3m (#225) | 512 | 60 | 8.5× |
| Si diamond (primitive cell) | Fd-3m (#227) | 512 | 60 | 8.5× |
| H₂ along z in a 10 bohr cubic box | P4/mmm (#123) | 512 | 40 | 12.8× |

These are the counts of the shifted Monkhorst-Pack mesh that
`KPoints.monkhorst_pack` builds (vibe-qc 0.17.17 development build); the
Γ-centred 8×8×8 mesh of either FCC crystal reduces to 29 points.

The IBZ weights are automatically normalized to sum to 1. A reduced
`KPoints` can be passed directly to `run_krhf_periodic_gdf`, but on the
GDF drivers it is a correctness bridge, not a saving: the driver checks
the reduction's representatives, orbit weights and `ir_mapping` against
the current system and declared mesh (#484), then expands it to the full
parent mesh and builds and diagonalises every point. A reduction whose
orbits are all singletons is accepted and normalised to native full-grid
order (#486). The wedge saving on the GDF route is the separate
`symmetry_reduce_k` flag of the next section, which reduces the exchange
build and leaves the k-point loop whole.

### Lattice cutoff convergence

The real-space lattice cutoff (`lattice_opts.cutoff_bohr`, default 15)
controls the number of periodic image cells included in the 2c/3c ERI
lattice sums. For multi-k GDF this affects the Bloch-pair cell list
used in the FT-based cderi construction.

Default cutoff (15 bohr) reaches chemical accuracy (+1.6 mHa vs PySCF)
on LiH FCC at kmesh=(2,2,2). Raising to 30 bohr reaches µHa parity
(−1.4 µHa). The internal FT kinetic-energy cutoff (`ke_cutoff`,
default 200 Ha) is sufficient for light-atom systems and does not need
tuning for LiH: the lattice cutoff is the primary accuracy knob.

**`rcut_strategy` and `rcut_precision` do not size the fit sums on this
route**, which is worth stating because their names suggest otherwise.
On the default `gdf_method="rsgdf"` route the per-pair builder
`build_lpq_bloch_native_fft` takes `lat_opts` directly and does not even
accept those two arguments; only the `gdf_method="compcell"` and
`"mdf"` builders forward them to the rcut estimator. So for the fit
sums, `cutoff_bohr` is the whole knob.

`rcut_strategy` does still reach the route, in a different role:
`_oneel_lattice_opts` reads it as a boolean gate for the one-electron
(overlap, kinetic, nuclear attraction and ECP) domain. With the default
`"pyscf_auto"` on a bulk (`dim = 3`) cell that domain is **pair
complete**: it keeps every AO pair whose centres lie within a cutoff of
each other, together with the nuclear images around that pair, so the
result does not depend on which lattice image of an atom the input names
(#438). The cutoff is measured, not estimated: the overlap is built
pair-complete out to an a-priori image-tail bound, and the cutoff is the
largest pair separation whose overlap element still exceeds 1e-16 of the
largest element (the same relative screen the nuclear attraction used
before). It never goes below your `cutoff_bohr`. On H2/STO-3G in a
6-bohr cube it is 20.4 bohr, on LiH/STO-3G 40.1 and on MgO/STO-3G
22.5. `"flat"` opts out and keeps your `cutoff_bohr` as a plain ball of
cells around the origin.
`rcut_precision` reaches nothing on this route at all. The
`gdf_method="compcell"` and `"mdf"` fits Fourier transform exactly the
AO pairs that the Gaussian three-centre tensor they correct keeps, every
pair within the lattice cutoff of each other, instead of the pairs of an
origin ball (#438, v0.17.15); on H2/STO-3G in a 6-bohr cube that moved the
Γ compcell energy by 2.3e-4 Ha and the `(2, 1, 1)` energy by 6.8e-4 Ha,
and the multi-k MDF loop at a single Γ point now agrees with the exact-Γ
MDF driver to 1e-15 Ha (#416).

The same `"pyscf_auto"` default also widens the image domain of the
exchange-correlation density on Kohn-Sham runs that use the periodic
Becke grid (the default). The XC cutoff grows to `2 r_ao + span`, where
`r_ao` is the largest radius at which a basis function still has an
amplitude of 1e-4 and `span` is the largest distance between two
basis-function centres of the cell, so every pair of basis functions
that overlap at one grid point is included. At 15 bohr the density was
truncated on compact cells with diffuse valence shells: the cubic Al2O3
model with STO-3G and PBE sat 1.55 mHa too high at Γ (#465). Its XC
cutoff is now about 23 bohr, about 18.5 bohr for diamond with def2-SVP
and about 31 bohr for rocksalt LiH with STO-3G, whose diffuse Li 2sp
shell reaches far. The AO-reach screening limits the extra work to the
basis functions that reach each grid batch, but compact cells with
diffuse shells still pay for it: one XC build takes about 2.5 times as
long as at 15 bohr on that Al2O3 cell and about twelve times as long on
LiH. The Γ-only GDF drivers
(`run_pbc_gdf_rks`, `run_pbc_gdf_uks`) use the same domain, and
`"flat"` keeps `cutoff_bohr` for XC as well.

**Convergence order**: converge the k-mesh first, THEN the cutoff.
A converged k-mesh with an under-converged cutoff can show spurious
mesh-convergence plateaus.
The [lattice-cutoff tutorial](../tutorial/periodic_lattice_cutoff_convergence.md)
measures the default and flat strategies on the cubic Al2O3 model.

### Convergence check

Always converge the k-mesh:

```python
for n in [2, 4, 6, 8, 10, 12]:
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 40
    result = vq.run_krhf_periodic_gdf(
        system, basis, (n, n, n), opts,
        aux_basis="def2-svp-jk", use_compcell=True,
    )
    print(f"{n}\u00d7{n}\u00d7{n}: {result.energy:.8f} Ha")
```

A converged mesh shows < 0.1 mHa/atom change between successive
densities.  For production work, **converge the k-mesh before converging
any other parameter** (cutoff, basis, functional).

### Space-group reduction of the exchange build

```{admonition} Status: preview
:class: warning

`symmetry_reduce_k=True` on `run_periodic_job` (driver keyword
`ibz_native=True`) runs closed- and open-shell multi-k GDF on 3-D cells
with exact exchange. It is exact for the finite fit it reduces: every
admitted operation is checked against the executed exchange source, and
the run fails closed wherever that check, the mesh or the electronic state
does not permit the reduction. The non-symmorphic admission (#452) and the
per-star occupation fill (#428) shipped in v0.17.15 without independent
fleet acceptance, and the shared symmetry layer keeps
`production_reduction_authorized` false, so the route is labelled preview.
```

Exact exchange is the one $n_k^2$ term of a multi-k GDF SCF. With
`symmetry_reduce_k=True`, $K(\mathbf k)$ is built only at the
representatives of the irreducible wedge and symmetry-transported to the
other members of each star, so the `Lpq` cache holds
$n_{\mathrm{IBZ}} \times n_k$ pairs instead of $n_k^2$; diagonalisation,
occupations and the energy expression still run on the full mesh, so the
total energy equals the unreduced one to roundoff. The request needs a
symmetry model:

```python
r = vq.run_periodic_job(
    system, basis, method="RHF", jk_method="gdf",
    kpoints=(3, 3, 3),
    symmetry="attach",          # spglib operations on the system
    symmetry_reduce_k=True,
    aux_basis="def2-svp-jk",
)
```

The `.out` states the reduction above its `[memory]` block
(`symmetry_reduce_k = exchange bras 27 -> 4 (irreducible wedge)` and
the `Lpq cderi pairs 729 -> 108 (6.8x fewer)` line that follows), and
the first iteration logs `IBZ-native occupations: filled once per star
on 4 representatives`. Occupations are filled once per star on the wedge
representatives with the star weight and copied to the members, after
levels split by no more than the measured star spectral spread (or by
less than one nanohartree) are grouped as degenerate (#428), so a
symmetry-degenerate frontier receives the symmetric ensemble occupation
instead of an integer split driven by lattice-sum noise. The state gate
then compares, in every iteration, each star member with its transported
representative and each representative with its own image under every
admitted operation of its little group, time reversal included (#470),
against a floor calibrated on the overlap; a density that breaks the
wedge's symmetry is refused with the little group named rather than
being symmetrised silently.

Every precondition fails closed. The messages you can meet, on a
rock-salt LiH/STO-3G cell (vibe-qc 0.17.15 development build, main,
2026-10-11):

```text
ValueError: run_periodic_job: symmetry_reduce_k requires an attached
symmetry model -- there is no irreducible wedge without one. Pass
symmetry='attach' (or attach validated operations to the PeriodicSystem)
alongside symmetry_reduce_k=True.

NotImplementedError: run_periodic_job: symmetry_reduce_k needs a true
multi-k Monkhorst-Pack mesh to reduce; this job samples 1 k-point(s). At
Gamma the wedge is the whole mesh and there is nothing to save.

NotImplementedError: ibz_native=True has no effect without exact exchange
(alpha == 0): the n_k^2 term it reduces is the HF exchange. Drop the flag
for a pure functional.

NotImplementedError: ibz_native=True cannot be combined with
bz_integration='gilat': the Gilat-Raubenheimer fill interpolates over the
full mesh, while the wedge-native state takes its occupations once per
star. Use the zero-temperature or smearing fill, or run without ibz_native.
```

A non-GDF `jk_method`, a `dim < 3` cell, a method other than RHF / RKS /
UHF / UKS and `optimize=True` are refused with their own messages, and a
state that breaks the wedge's symmetry is refused during the SCF
(`ibz_native=True: the ... does not respect the irreducible wedge's
symmetry ...`). Reduced meshes handed to the driver must be the declared
full mesh in native order modulo reciprocal lattice vectors with uniform
weights; reordered, duplicated or displaced points are refused before
integral work (#474). Glides and screws are admitted on the default SR/LR
fit and on the physical classic fit (#452, #466), so rock-salt LiH and
diamond reduce. The admission table, the transport convention and the
measured residuals are in
[periodic methods, section 3.2.3](periodic_methods.md#32-what-ships-today);
the [symmetry-reduced k mesh tutorial](../tutorial/symmetry_reduced_k_mesh.md)
measures the wedge, the star weights and the wall time on LiH.

### Guards every GDF run applies

* **Energy sanity.** A converged total energy outside the physical range
  (positive beyond a small slack, or `|E| > max(10 sum_A Z_A^2, 100)` Ha)
  raises `RuntimeError` on k meshes and, since v0.17.15, on a single Γ
  point handed to the exact-Γ driver as well. The message begins
  `run_krhf_periodic_gdf: SCF total energy ... Ha is non-physical` and ends
  with `Pass check_energy_sanity=False to bypass this guard (diagnostics
  only).` An unconverged run only warns. A cell of four H2 pairs had come
  back "converged" at -161.5 Ha with `gdf_method="compcell"` before the
  guard reached that route (#458).
* **Options object.** A grid-based KS request through
  `run_krhf_periodic_gdf` / `run_krks_periodic_gdf` with explicitly supplied
  HF options is refused before numerical setup (#505):
  `TypeError: run_krhf_periodic_gdf: grid-based XC requires
  PeriodicKSOptions; use PeriodicKSOptions() or omit options for the
  periodic XC grid defaults.`
* **Real-only solver admission.** The Γ GDF RHF/RKS and UHF/UKS drivers,
  the Γ Ewald RHF/RKS/ROHF drivers and the slab routes apply the real-only
  eigensolver contract to their orbital seeds and every later solve; an
  active complex Davidson guess or a complex solver frame is rejected
  (`NotImplementedError: Real-only SCF does not support complex Davidson
  guesses`) before it can reach the density construction (#494).
* **XC grid electron count.** Every Kohn-Sham run on these drivers records
  the grid integral of the final XC density as `result.xc_grid_electrons`
  and its relative error as `result.xc_grid_electron_error`, and emits
  `PeriodicXCGridChargeWarning` when the count misses the electrons per
  cell by more than 1e-3 relative (#385). A shortfall means the density's
  image reach is truncated; raise `cutoff_bohr` or let the default
  `rcut_strategy` size the XC domain (below). The check never raises and
  changes no energy.
* **Fit controls on a Γ point.** Smearing, Fock mixing and a level shift
  on a default-Γ closed-shell job move the job from the exact-Γ driver to
  the k-mesh engine at `(1, 1, 1)`, which implements them; the strategy
  block in the `.out` says so.

### The classic fit and its AO-pair support

`fit_source="classic"` selects the shared-q FFT fit on a 3-D RSGDF cell.
Since #466 it sums every AO pair whose centres lie within
`lattice_opts.cutoff_bohr` of each other, the term set of the native
three-centre integrals (`classic_support_policy="physical_pairs"`, the
default), so a Γ supercell and the k mesh it folds to build one fitted
operator: a 15.6-bohr H8 chain cell at Γ sat 9.79 mHa per H2 below its H2
cell on the `(4, 1, 1)` mesh and now agrees with it to 3e-13 Ha.
`classic_support_policy="historical_index_ball"` replays the former ball
of cells around the origin, and the executed fit source records which
support ran (`classic-physical` or `classic`). The physical fit is
unscreened, so a positive `fit_screen_threshold` is refused without the
replay:

```text
NotImplementedError: the physical classic fit is unscreened:
fit_screen_threshold > 0 requires classic_support_policy='historical_index_ball'
(whose pairs depend on the cell, #466) or the default SR/LR fit source
```

The analytic gradient of an explicitly selected classic fit runs on the
historical replay (#404); the physical-pair classic gradient is still
refused with a message naming the supported support. The experimental
AICCM/CCM and chi producers pin the historical ball so their own folds keep
one operator.

### Driver keywords and defaults

The multi-k GDF drivers (`run_krhf_periodic_gdf`, `run_krks_periodic_gdf`,
`run_kuhf_periodic_gdf`, `run_kuks_periodic_gdf`) take the lattice and
convergence settings from the options object and the fit settings as
keywords. The ones this page discusses:

| Keyword | Default | Meaning |
|---|---|---|
| `options.lattice_opts.cutoff_bohr` | `15.0` | Real-space pair cutoff of the fit, the one-electron terms and (with `rcut_strategy="flat"`) the XC density. |
| `rcut_strategy` | `"pyscf_auto"` | `"pyscf_auto"` grows the one-electron domain to a measured pair-complete bound and the XC image domain to the AO-pair overlap bound; `"flat"` keeps `cutoff_bohr` for both. |
| `gdf_method` | `"rsgdf"` | Fit builder: `"rsgdf"` (SR/LR source), `"compcell"`, `"mdf"`. |
| `fit_source` | `"auto"` | `"classic"` or `"sr"` on a 3-D RSGDF cell; `"auto"` keeps the default selection. |
| `classic_support_policy` | `None` (`"physical_pairs"`) | AO-pair support of the classic fit; `"historical_index_ball"` replays the origin ball. |
| `fit_screen_threshold` | `0.0` | Cauchy-Schwarz pre-screen of the RSGDF three-centre fit; positive values need the historical classic support or the SR/LR source. |
| `ibz_native` | `False` | Wedge-native exchange build; `run_periodic_job(symmetry_reduce_k=True)`. |
| `bz_integration` | `None` | `None` / `"smearing"` or `"gilat"`; `"gilat"` cannot be combined with smearing or `ibz_native`. |
| `check_energy_sanity` | `True` | Raise on a converged non-physical total energy. |
| `compute_gradient` | `False` | Analytic nuclear gradient on the SCF's own fit. |
| `rsgdf_ke_cutoff` | `None` | Lower bound on the SR/LR planner's reciprocal cutoff (Ha). |
| `options.smearing_temperature` | `0.0` | Fermi-Dirac $k_B T$ in Hartree. |

`run_periodic_job` exposes `kpoints`, `gdf_method`, `rsgdf_ke_cutoff`,
`mdf_ke_cutoff`, `symmetry_reduce_k`, `bz_integration` and the SCF
controls; `rcut_strategy`, `classic_support_policy`, `fit_screen_threshold`
and `check_energy_sanity` are driver keywords.

### Exchange finite-size error of the BvK-ewald gauge

HF and hybrid exchange on this route runs the `BvK-ewald` gauge
(`result.exchange_q0`): every `K(k)` receives the probe-charge shift
`ξ·S(k)·D(k)·S(k)`, where `ξ` is the Madelung constant of the
Born-von Kármán supercell `A·diag(n1, n2, n3)` (PySCF `exxdiv='ewald'`).
The shift removes the point-charge part of the `q → 0` exchange
singularity exactly. It does not remove the next term, which depends on
the shape of each occupied orbital's exchange charge. For a closed-shell
system whose occupied orbitals do not overlap between cells, the energy
per cell on a mesh is

$$E(\text{mesh}) - E_\infty = -\sum_{ab} M_{ab}\, C_{ab},$$

where $E_\infty$ is the infinite-mesh limit (for a molecular crystal, the
isolated molecule plus its small lattice interactions), $C$ is the spread
tensor of the occupied space (for one doubly occupied orbital, the
covariance $\langle r_a r_b\rangle - \langle r_a\rangle\langle r_b\rangle$),
and $M$ is the curvature at the origin of the supercell's Ewald potential of
a unit point charge with its own $1/r$ removed. Because
$\mathrm{Tr}\,M = 4\pi/V_\mathrm{BvK}$, the error falls as
$1/V_\mathrm{BvK} = 1/(N_k\Omega)$. That is the known inverse-volume rate
of Madelung-corrected exchange (Xing, Li and Lin, Math. Comput. 93, 679
(2024); Sundararaman and Arias, Phys. Rev. B 87, 165122 (2013)). Faster,
exponential convergence would need a truncated exchange kernel (Wigner-Seitz
truncation), which the `dim=3` route does not offer. This error does not
come from the SCF or the fit, and PySCF gives the same numbers.

The mesh shape decides how fast that is:

* **Cubic mesh `(n, n, n)` on a cubic cell.** $M$ is isotropic, so
  $E \approx E_\infty - 4\pi\,\mathrm{Tr}\,C/(3 n^3 \Omega)$, an $n^{-3}$
  approach. At `(1, 1, 1)` this is the $1/L^3$ molecule-in-a-box error
  quoted under
  [Where each method plugs in](periodic_methods.md#where-each-method-plugs-in).
* **One-axis mesh `(nk, 1, 1)`**, for example a wire in a `dim=3` vacuum
  box. The supercell grows along one axis only, so the energy per cell
  approaches its limit as $1/n_k$. For $n_k \ge 2$, with a square
  transverse cross-section of side $a_\perp$,

  $$E(n_k) = E_\infty + \frac{S_3}{a_\perp^3}\,(C_\parallel - C_\perp)
  - \frac{4\pi\,C_\parallel}{n_k\,\Omega},$$

  up to terms of order $e^{-2\pi n_k a/a_\perp}$. $C_\parallel$ is the
  spread along the sampled axis, $C_\perp$ across it, and
  $S_3 = \sum'_{\mathbf n \in \mathbb Z^2} |\mathbf n|^{-3} = 9.0336$. The
  middle term remains at any $n_k$, because the transverse images stay at
  Γ. It vanishes for an isotropic spread and falls as $a_\perp^{-3}$.

Measured 2026-10-01 on H₂ (bond 1.4 bohr along x) in a 15 bohr cube with
pob-TZVP and default `run_krhf_periodic_gdf` options. PySCF ran out of
process with the same orbital and auxiliary basis. The model column has no
fitted parameter: it uses only the isolated molecule's HF energy
(-1.1312165 Ha) and spread ($C_{xx} = 1.0072$, $C_{yy} = C_{zz} = 0.7392$
bohr²) from a molecular PySCF calculation in the same basis.

| mesh | vibe-qc (Ha) | PySCF `exxdiv='ewald'` (Ha) | model (Ha) | `ξ` |
|---|---:|---:|---:|---:|
| (1,1,1) | -1.1343212771 | -1.1343212771 | -1.1343015 | 0.18915317 |
| (2,1,1) | -1.1323881247 | -1.1323881247 | -1.1323744 | 0.12038945 |
| (4,1,1) | -1.1314484162 | -1.1314484162 | -1.1314367 | -0.01923502 |
| (8,1,1) | -1.1309788015 | -1.1309788015 | -1.1309679 | -0.29848770 |
| (16,1,1) | -1.1307440719 | -1.1307440719 | -1.1307335 | -0.85699306 |
| (2,2,2) | -1.1316120338 | -1.1316120338 | -1.1316022 | 0.09457658 |
| (3,3,3) | | -1.1313400482 | -1.1313308 | 0.06305106 |

vibe-qc and PySCF agree to 3e-12 Ha and use the same `ξ` on every mesh
both ran. The successive one-axis differences (1.933, 0.940, 0.470,
0.235 mHa) match the model to 0.3 %; from $n_k \ge 2$ each doubling
removes $2\pi C_{xx}/(n_k\Omega)$. The computed energies sit a
near-constant 10 to 20 µHa below the model, which is the density-fitting
error of the def2-TZVP-JK fit (17 µHa on the isolated molecule). Two controls on H₂/STO-3G in the same box confirm the
mechanism. Laying the molecule across the sampled axis shrinks the tail by
$C_\parallel/C_\perp$ (1.485 measured, 1.486 predicted). Widening the
transverse vacuum from 15 to 20 bohr shrinks it by $(15/20)^2$.

What to do with this:

* **Extrapolate a one-axis mesh instead of reading one difference.** The
  leading error is exactly $b/n_k$, so $2E(2n) - E(n)$ removes it (take
  $n \ge 2$; the Γ point is not on that line). Here
  $2E(8) - E(4) = -1.1305092$ and $2E(16) - E(8) = -1.1305093$ Ha, which is
  0.23 mHa above the `(16, 1, 1)` energy.
* **Converge the transverse vacuum separately.** The extrapolated value is
  the `(∞, 1, 1)` limit of this box, not the isolated molecule or wire. For
  this H₂ it lies 0.72 mHa above the isolated molecule (the $S_3$ term),
  0.30 mHa at $a_\perp$ = 20 bohr and 0.09 mHa at 30 bohr.
* **Do not drop the shift instead.** On a one-axis mesh `ξ` falls linearly,
  by $\pi a/(3 a_\perp^2)$ per added k point (0.0698 here), because the
  neutralising background fills an ever longer supercell. Without the
  shift, PySCF's `exxdiv=None` on the same cell gives -0.945, -1.012,
  -1.151 and -1.429 Ha at $n_k$ = 1, 2, 4 and 8.

## CRYSTAL And PySCF Parity

CRYSTAL and PySCF remain essential references, but the validation
boundary is external:

1. Write a CRYSTAL or PySCF input next to the vibe-qc input.
2. Run the external program as a separate executable or script.
3. Parse its output into a comparison record.
4. Store method, basis, lattice, k-mesh, energy, occupation, and
   convergence metadata in the benchmark manifest.

This keeps licensing, dependency, and numerical-responsibility
boundaries clean. It also makes it obvious whether a discrepancy is in
vibe-qc or in an adapter/parser.

## Native FFTDF/GDF Roadmap

The production target is a native periodic stack:

* FFTDF first for speed and plane-wave-style Coulomb handling.
* GDF for all-electron Gaussian density fitting and CRYSTAL/PySCF
  parity benchmarks.
* Range-separated exchange screening for hybrids and HF in large
  cells.
* Disk-backed Cholesky/DF tensors for correlated methods later.

The public `run_krhf_periodic_gdf` / `run_krks_periodic_gdf` names are
the stable API. A Γ-only mesh runs through the native Γ-GDF bridge;
any other Monkhorst-Pack mesh runs the native multi-k GDF loop. The
remaining roadmap work is performance and dense-solid scaling
(disk-backed tensors; the exchange build already reduces to the
irreducible wedge under `symmetry_reduce_k`), not correctness of the
multi-k path itself.

## See Also

* [`periodic_methods.md`](periodic_methods.md): comparative tour of
  vibe-qc's periodic-SCF kernels (BIPOLE, native GDF, GPW/GAPW),
  bonding-organised example gallery, and the surface-reactions
  workflow.
* [`k_points.md`](k_points.md), Monkhorst-Pack meshes, IBZ reduction,
  and band paths.
* [`ewald.md`](ewald.md), Ewald nuclear attraction, Madelung terms,
  and the memory guard on legacy periodic XC.
* [`band_structure.md`](band_structure.md), band structures, DOS, and
  Fermi-level reporting.
* [`../roadmap.md`](../roadmap.md), FFTDF/GDF and metallic-smearing
  roadmap items.
