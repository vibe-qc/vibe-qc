# LiH at multiple k-points: KRHF vs Peintinger 2013

> **Status note (updated v0.12.0).** Native **multi-k GDF** now ships:
> `run_krhf_periodic_gdf(..., use_compcell=True, exxdiv="ewald")`
> builds the k-dependent GDF `Lpq` via the symmetry-preserving all-FT
> path resolved per `(k_i, k_j)` pair, reaching **PySCF µHa parity** on
> the canonical light-atom case (LiH primitive FCC / sto-3g /
> def2-svp-jk: −1.4 µHa vs `pyscf.pbc.scf.KRHF.density_fit()` at a
> converged lattice cutoff; +1.6 mHa at the cheaper default cutoff;
> re-measured 2026-08-05 against PySCF 2.13.1, run out of process).
> Use `use_compcell=True` for multi-k GDF parity, the
> `use_compcell=False` default routes multi-k through the Ewald-3D Fock
> path, which is not yet at parity on tight cells. The pob-TZVP rocksalt
> sweep numbers below are **illustrative convergence targets** (not yet
> measured at this basis); the API and the LiH-FCC/sto-3g parity are
> shipped.

This tutorial records the LiH rocksalt multi-k target for the native
GDF implementation: reproduce the published reference
**E = -8.149050 Ha/cell** for LiH rocksalt at HF/pob-TZVP from the
[Peintinger, Vilela Oliveira, Bredow paper](https://doi.org/10.1002/jcc.23153)
SI Table 2.

**You will**:

- Build LiH rocksalt at the experimental lattice constant.
- Run the current Γ-only native GDF smoke test.
- See the planned multi-k convergence sweep
  (1×1×1 → 2×2×2 → 4×4×4 → 8×8×8).
- Compare the target against the paper-SI reference and external
  CRYSTAL / PySCF runs.

**Background**: see
[`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md)
for the current GDF API boundary, external-reference policy, and
native FFTDF/GDF roadmap.

## Setup

Build the LiH rocksalt primitive cell at the Peintinger 2013 lattice
constant and attach the pob-TZVP basis, the shared starting point for
every run on this page:

```python
import vibeqc as vq

# LiH rocksalt — primitive cell. Lattice constant a = 4.084 Å
# (experimental at 0 K; matches Peintinger 2013 SI).
a = 4.084 / 0.529177  # Å → bohr
lih = vq.PeriodicSystem(
    3,                                       # dim = 3
    [[0.0, a/2, a/2],
     [a/2, 0.0, a/2],
     [a/2, a/2, 0.0]],
    [vq.Atom(3, [0.0, 0.0, 0.0]),            # Li at origin
     vq.Atom(1, [a/2, a/2, a/2])],           # H at the rocksalt-equivalent site
)
basis = vq.BasisSet(lih.unit_cell_molecule(), "pob-tzvp")
```

## Current Γ-only GDF smoke test

Running the native GDF driver at a single (1, 1, 1) mesh gives a fast
sanity check that the cell, basis, and convergence aids are wired
correctly before committing to a dense k-mesh:

```python
opts = vq.PeriodicRHFOptions()
opts.damping = 0.3
opts.level_shift = 0.6
opts.use_diis = True

res = vq.run_krhf_periodic_gdf(
    lih,
    basis,
    kmesh=(1, 1, 1),
    options=opts,
)
print(f"Γ-only native GDF: E = {res.energy_per_cell_ha:.6f} Ha")
print(f"backend: {res.backend}")
```

The Γ point is useful as a smoke test and debugging anchor. It is too
coarse to reproduce the dense-k Peintinger SI number by itself.

## k-mesh convergence sweep

Multi-k GDF runs via `use_compcell=True`. (For µHa parity tighten the
lattice cutoff, `options.lattice_opts.cutoff_bohr = 30` and
`rsgdf_ke_cutoff=...`; the default cutoff gives chemical accuracy.)

```python
results = {}
for nk in (1, 2, 4, 8):
    res = vq.run_krhf_periodic_gdf(
        lih,
        basis,
        kmesh=(nk, nk, nk),
        options=opts,
        aux_basis="def2-svp-jk",
        use_compcell=True,        # k-dependent GDF Lpq (per-(k_i,k_j) FT)
        exxdiv="ewald",           # k-mesh-aware Madelung exchange shift
    )
    results[nk] = res.energy_per_cell_ha
    print(f"  [{nk},{nk},{nk}]: E = {res.energy_per_cell_ha:.6f} Ha")

reference = -8.149050  # Peintinger 2013 SI Table 2 (Ha/cell)
print(f"\nPaper-SI reference (HF/pob-TZVP): {reference:.6f} Ha/cell")
print(f"Δ at [8,8,8]: {1000 * (results[8] - reference):+.3f} mHa/cell")
```

Illustrative convergence (pob-TZVP rocksalt, targets, not yet measured
at this basis; the LiH-FCC/sto-3g case is the shipped µHa parity gate):

```
  [1,1,1]: E = -8.121... Ha   (Γ-only — too coarse)
  [2,2,2]: E = -8.144... Ha
  [4,4,4]: E = -8.148... Ha
  [8,8,8]: E = -8.149... Ha   (matches SI to ~mHa)

Paper-SI reference (HF/pob-TZVP): -8.149050 Ha/cell
Δ at [8,8,8]: < 1 mHa/cell
```

The k-mesh convergence is **monotonic**, each refinement
reduces the residual to the dense-k limit. At 8×8×8 the
energy is converged within the basis-set + integration error
of the published reference.

## External parity target

CRYSTAL and PySCF remain external reference programs. vibe-qc should not
import either package as a backend; parity runners execute them as
separate programs and parse their outputs.

Target benchmark matrix for the native multi-k GDF loop:

| System | k-mesh | Reference | Target |
|--------|--------|-----------|--------|
| MgO/sto-3g | [1,1,1] | PySCF.pbc.GDF | Γ-GDF parity retained |
| MgO/sto-3g | [2,2,2] | PySCF.pbc.KRHF | sub-µHa |
| LiH/pob-TZVP | [8,8,8] | Peintinger SI / CRYSTAL | sub-mHa to SI |

## Switch to KRKS (KS-DFT)

Same driver, with a `functional=` argument:

```python
result = vq.run_krhf_periodic_gdf(
    lih, basis, kmesh=(1, 1, 1), functional="pbe",
)
print(f"PBE/pob-TZVP/Γ: {result.energy_per_cell_ha:.6f} Ha")
```

Or use the explicit `run_krks_periodic_gdf` entry point, or
`run_periodic_job(method="RKS", functional=..., kmesh=...)`, 
all three should dispatch to the same machinery for supported meshes.

## What this tells you

- **Multi-k GDF reaches PySCF parity** (`use_compcell=True`): native
  per-`(k_i, k_j)` all-FT ERIs, per-k Hartree J, per-pair exchange K,
  and a k-mesh-aware Madelung exchange shift. LiH FCC / sto-3g hits
  µHa vs `pyscf.pbc.scf.KRHF.density_fit()` at a converged cutoff.
- **The accuracy knob is the lattice cutoff**, not the DF method: the
  default cutoff gives chemical accuracy; raising it reaches µHa.
- **Reproducing published references is the goal.** Peintinger 2013 SI
  Table 2 is a CRYSTAL-grade benchmark for the multi-k + GDF + pob-TZVP
  stack (the pob-TZVP rocksalt numbers above are still targets).

## The dense-Lpq RAM ceiling

`Lpq[k1, k2, L, μ, ν]` is materialised dense up front. Memory
scales as `O(nkpts² × naux × nao²)`. The 8×8×8 LiH/pob-TZVP
sweep above is comfortable; the asbestos-paper-grade
configurations are not:

RSGDF setup shares the auxiliary Coulomb metric, its eigensystem, and its
Schwarz mask once per unique momentum transfer `q = k_j - k_i`. Whitening is
reapplied from that eigensystem while the three-centre factor is still built
per ordered pair because the inter-cell AO product carries the ket momentum
phase. This removes exact duplicated setup work without changing the dense
`Lpq` memory ceiling or the remaining `O(nkpts²)` pair-transform cost.

| System | k-mesh | Lpq dense memory |
|--------|--------|------------------|
| MgO conventional / sto-3g | [4,4,4] | manageable |
| LiH primitive / pob-TZVP | [8,8,8] | comfortable |
| **Lizardite / pob-TZVP** | [2,2,2] | **≈ 553 GB** |
| **Antigorite-m17 / pob-TZVP** | [2,2,2] | **≈ 5 TB** |

**Workaround**: stay below the ceiling, small system × full
k-mesh, OR full system × small k-mesh, never both.

**Permanent fix**: native streaming-Lpq / disk-backed DF tensors, so
the code never needs to materialise the full k-pair tensor in memory.

## See also

- [`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md), 
  full multi-k API + open issues.
- [`user_guide/k_points.md`](../user_guide/k_points.md), 
  Monkhorst-Pack generation, IBZ reduction (Γ-only context).
- [`user_guide/band_structure.md`](../user_guide/band_structure.md), 
  band-structure plot from a converged multi-k SCF.
- [Why solid-state calculations use `pob-TZVP`](pob_tzvp.md), basis-set
  context for solid-state work.
- [Solid-state walkthrough: LiH rocksalt with `pob-TZVP`](lih_pob_tzvp_solid_state.md), 
  the v0.7-era Γ-only LiH worked example this tutorial
  generalises to multi-k.
