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

* `run_rhf_periodic_gamma_gdf(...)` for Γ-only native Gaussian density
  fitting RHF, RKS, and hybrids across `PeriodicSystem.dim = 1, 2, 3`.
* `run_krhf_periodic_gdf(..., kmesh, ...)` and
  `run_krks_periodic_gdf(..., kmesh, ...)` as the public k-point API
  for native GDF in 1D, 2D, and 3D. A Γ-only mesh `(1, 1, 1)`
  delegates to the Γ-GDF implementation; **any other Monkhorst-Pack
  mesh runs the full native multi-k GDF loop**, k-dependent 3c/2c
  blocks, Bloch-phase assembly, multi-k DIIS. The `use_compcell=False`
  default uses the Ewald-gauge Fock path (within **0.000 mHa of the
  \u0393-point energy in the molecular limit**). **For tight ionic cells use
  `use_compcell=True`**: the multi-k GDF `Lpq` is then
  built by the symmetry-preserving all-FT path resolved per
  `(k_i, k_j)` pair (`build_lpq_bloch_native_fft`) with a per-k Hartree
  J, per-pair exchange K, and a k-mesh-aware Madelung exchange shift \u2014
  reaching **PySCF \u00b5Ha parity** (LiH primitive FCC / sto-3g, kmesh
  (2,2,2): \u22121.4 \u00b5Ha vs `pyscf.pbc.scf.KRHF.density_fit()` at a converged
  lattice cutoff; +1.6 mHa at the cheaper default \u2014 re-measured
  2026-08-05 against PySCF 2.13.1, run out of process). The AFT long-range
  correction is ON by default since v0.12.0 (\u03b7=1.0, libint convention,
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

\u0393-only calculations are **not appropriate for real solids**. A single
k-point sees only the \u0393-periodic component of the density; the full
Brillouin-zone integral must be sampled with a Monkhorst-Pack mesh.

The required mesh density depends on the crystal system:

### Mesh density by crystal type

| Crystal type | Typical primitive cell | Minimum mesh | Converged mesh |
|---|---:|---:|---:|
| Simple cubic (sc) | 1 atom, \u223c3-5 \u00c5 | 4\u00d74\u00d74 | 8\u00d78\u00d78 |
| FCC (Cu, Al, Ag) | 1 atom, \u223c3-4 \u00c5 | 6\u00d76\u00d76 | 12\u00d712\u00d712 |
| BCC (Fe, W, Na) | 1 atom, \u223c3-4 \u00c5 | 6\u00d76\u00d76 | 12\u00d712\u00d712 |
| Rocksalt (MgO, NaCl) | 2 atoms, \u223c4-6 \u00c5 | 4\u00d74\u00d74 | 8\u00d78\u00d78 |
| Diamond/Zincblende (Si, GaAs) | 2 atoms, \u223c5-6 \u00c5 | 4\u00d74\u00d74 | 8\u00d78\u00d78 |
| Perovskite (BaTiO\u2083) | 5 atoms, \u223c4 \u00c5 | 4\u00d74\u00d74 | 6\u00d76\u00d76 |
| Layered / 2D slab | variable | 6\u00d76\u00d71 | 12\u00d712\u00d71 |

**Rule of thumb**: the linear k-point density \u00d7 lattice constant should be
\u2265 20-30 \u00c5 for insulating systems. For metals, double it.

### Symmetry reduction

vibe-qc's `monkhorst_pack` applies time-reversal symmetry automatically.
Space-group symmetry reduction (IBZ) is available through `spglib`:

```python
vq.attach_symmetry(system)  # populates system.symmetry via spglib
kpts = vq.KPoints.monkhorst_pack(system, [8, 8, 8])                    # full mesh
kpts = vq.KPoints.monkhorst_pack(system, [8, 8, 8], symmetry=True)     # IBZ
```

Symmetry reduction can reduce the number of irreducible k-points by
factors of 2\u201348 depending on the space group, cutting SCF cost
proportionally. Examples on an 8\u00d78\u00d78 mesh:

| System | Space group | Full kpts | IBZ kpts | Factor |
|---|---:|---:|---:|
| MgO rocksalt | Fm\u20103m (#225) | 512 | 20 | 25.6\u00d7 |
| H\u2082 in cubic box | (trivial) | 512 | 40 | 12.8\u00d7 |
| Si diamond | Fd\u20103m (#227) | 512 | 29 | 17.7\u00d7 |

The IBZ weights are automatically normalized to sum to 1. The `KPoints`
object can be passed directly to `run_krhf_periodic_gdf` \u2014 the driver
handles the normalization internally.

### Lattice cutoff convergence

The real-space lattice cutoff (`lattice_opts.cutoff_bohr`, default 15)
controls the number of periodic image cells included in the 2c/3c ERI
lattice sums. For multi-k GDF this affects the Bloch-pair cell list
used in the FT-based cderi construction.

Default cutoff (15 bohr) reaches chemical accuracy (+1.6 mHa vs PySCF)
on LiH FCC at kmesh=(2,2,2). Raising to 30 bohr reaches \u00b5Ha parity
(\u22121.4 \u00b5Ha). The internal FT kinetic-energy cutoff (`ke_cutoff`,
default 200 Ha) is sufficient for light-atom systems and does not need
tuning for LiH \u2014 the lattice cutoff is the primary accuracy knob.

**`rcut_strategy` and `rcut_precision` do not size the fit sums on this
route**, which is worth stating because their names suggest otherwise.
On the default `gdf_method="rsgdf"` route the per-pair builder
`build_lpq_bloch_native_fft` takes `lat_opts` directly and does not even
accept those two arguments; only the `gdf_method="compcell"` and
`"mdf"` builders forward them to the rcut estimator. So for the fit
sums, `cutoff_bohr` is the whole knob.

`rcut_strategy` does still reach the route, in a different role:
`_oneel_lattice_opts` reads it as a boolean gate. The default
`"pyscf_auto"` enables a **measured** widening of the one-electron S/T
cutoff (it grows the cutoff until the Bloch overlap `S(k)` leaves the
non-PSD tier, rather than trusting any a-priori estimate), and
`"flat"` opts out. This only ever *raises* the cutoff, and for tight
bases (sto-3g, pob-TZVP-REV2) it returns it unchanged: on the LiH case
above, `"flat"` and `"pyscf_auto"` give bit-identical energies. Expect
it to matter for diffuse bases on dense lattices, not here.
`rcut_precision` reaches nothing on this route at all.

**Convergence order**: converge the k-mesh first, THEN the cutoff.
A converged k-mesh with an under-converged cutoff can show spurious
mesh-convergence plateaus.

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
(disk-backed tensors, IBZ-weighted J/K), not correctness of the
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
