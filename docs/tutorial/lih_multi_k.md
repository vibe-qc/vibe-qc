# LiH at multiple k-points: KRHF vs Peintinger 2013

> **Status note (updated v0.12.0).** Native **multi-k GDF** now ships:
> `run_krhf_periodic_gdf(..., use_compcell=True)`
> builds the k-dependent GDF `Lpq` via the symmetry-preserving all-FT
> path resolved per `(k_i, k_j)` pair, reaching **PySCF µHa parity** on
> the canonical light-atom case (LiH primitive FCC / sto-3g /
> def2-svp-jk: −1.4 µHa vs `pyscf.pbc.scf.KRHF.density_fit()` at a
> converged lattice cutoff; +1.6 mHa at the cheaper default cutoff;
> re-measured 2026-08-05 against PySCF 2.13.1, run out of process).
> Multi-k GDF now always builds this fit: the `use_compcell=False`
> default is promoted to `use_compcell=True` on every `gdf_method`
> (since 2026-09-29 also for `"mdf"` and `"compcell"` with a pure
> functional), so it no longer reaches the Ewald-3D Fock path. The pob-TZVP rocksalt
> study below remains a **reference-reproduction target**, with no measured
> pob-TZVP sweep supplied; the API and the LiH-FCC/sto-3g parity are shipped.

This tutorial records the LiH rocksalt multi-k target for the native
GDF implementation: reproduce the published reference
**E = -8.149050 Ha/cell** for LiH rocksalt at HF/pob-TZVP from the
[Peintinger, Vilela Oliveira, Bredow paper](https://doi.org/10.1002/jcc.23153)
SI Table 2.

**You will**:

- Build LiH rocksalt at the experimental lattice constant.
- Run the current Γ-only native GDF smoke test.
- Run a bounded multi-k pilot (1×1×1 → 2×2×2), then extend it only
  after checking resources and numerical convergence.
- Compare the target against the paper-SI reference and external
  CRYSTAL / PySCF runs.

**Background**: see
[`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md)
for the current GDF API boundary, external-reference policy, and
native FFTDF/GDF roadmap.

## Background: Bloch sums and k convergence

The [k-point tutorial](kpoints_brillouin_bloch.md) explains what a k-point
is; this paragraph says why a rocksalt insulator converges the way the sweep
below shows. In a Gaussian-basis periodic SCF every atom-centred function
$\chi_\mu$ of the reference cell is replaced by its Bloch sum
$\phi_\mu^{\mathbf k}(\mathbf r) = \sum_{\mathbf T} e^{i\mathbf k\cdot\mathbf T}\chi_\mu(\mathbf r - \mathbf T)$,
and the Fock, overlap and density matrices become independent problems at
each $\mathbf k$ coupled only through the density, which is the
Brillouin-zone average

$$
\mathbf D(\mathbf T) = \sum_{\mathbf k} w_{\mathbf k}\, e^{-i\mathbf k\cdot\mathbf T}\,\mathbf D(\mathbf k),
\qquad \sum_{\mathbf k} w_{\mathbf k} = 1 .
$$

A Monkhorst-Pack mesh ({vibeqc-cite}`monkhorst_pack_1976`) replaces the
integral by a uniform $N\times N\times N$ sum, and an $N^3$ mesh is exactly
equivalent to Born-von Karman periodicity on an $N\times N\times N$ supercell:
the mesh cannot resolve real-space correlations longer than $N$ cells. For an
insulator the density matrix $\mathbf D(\mathbf T)$ decays exponentially with
$|\mathbf T|$ (the decay length is set by the gap), so the error of the
$N$-point sum also falls off exponentially and a modest $N$ suffices; LiH,
with its wide gap, is a benign case, which is what makes the sweep from
$1^3$ to $8^3$ such a clean convergence test. A metal has no gap, the density
matrix decays only algebraically, and the same sweep would converge as a power
law and need [smearing](fermi_dirac_smearing.md). The Coulomb term needs
one extra ingredient at $\mathbf k \neq 0$: the exact-exchange
$\mathbf q = 0$ singularity is removed with the Ewald probe-charge
correction ({vibeqc-cite}`gygi_baldereschi_exx_1986`), which the driver
applies automatically to HF and hybrid exchange (the `exxdiv='ewald'`
convention of PySCF; the keyword is no longer an argument of
`run_krhf_periodic_gdf`). Multi-k GDF is a production route
(see [Methods: theory and status](../methods.md#how-to-read-the-status-labels)).

## Setup

Build the LiH rocksalt primitive cell at the Peintinger 2013 lattice
constant and attach the pob-TZVP basis, the shared starting point for
every run on this page. Save the blocks in one script and run them in order
with the installed engine's Python, outside the checkout. These low-level
calls do not automatically write the full `run_periodic_job` output family;
retain the input, console log and runtime identity with your measurements:

```python
import vibeqc as vq

# LiH rocksalt, primitive cell. Lattice constant a = 4.084 Å
# (experimental at 0 K; matches Peintinger 2013 SI).
a = 4.084 / 0.529177210903  # Å → bohr
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
# doc-audit: skip - continues with vq, lih and basis from Setup
opts = vq.PeriodicRHFOptions()
aux_basis = "def2-svp-jk"

res = vq.run_krhf_periodic_gdf(
    lih,
    basis,
    kmesh=(1, 1, 1),
    options=opts,
    aux_basis=aux_basis,
)
if not res.converged:
    raise RuntimeError("Gamma-point GDF did not converge; inspect the original trace")
gamma_result = res
print(f"Γ-only native GDF: E = {res.energy_per_cell_ha:.6f} Ha")
print(f"backend: {res.backend}")
```

The Γ point is useful as a smoke test and debugging anchor. It is too
coarse to reproduce the dense-k Peintinger SI number by itself.

## k-mesh convergence sweep

Begin with two meshes and reuse the Gamma result from the smoke test. Keep
the same auxiliary basis and options at every point. A passed Gamma test does
not predict the cost or accuracy of a dense mesh. The
[lattice-cutoff study](periodic_lattice_cutoff_convergence.md) explains why
mesh refinement alone is insufficient; support settings that worked for the
STO-3G reference do not qualify pob-TZVP automatically.

```python
# doc-audit: skip - continues with the converged gamma_result and setup above
results = {1: gamma_result.energy_per_cell_ha}
mesh_sizes = (2,)  # extend to (2, 4, 8) only after checking memory and pilot cost
for nk in mesh_sizes:
    res = vq.run_krhf_periodic_gdf(
        lih,
        basis,
        kmesh=(nk, nk, nk),
        options=opts,
        aux_basis=aux_basis,
        use_compcell=True,        # k-dependent GDF Lpq (per-(k_i,k_j) FT); the
                                  # k-mesh-aware Madelung exchange shift is automatic
    )
    if not res.converged:
        raise RuntimeError(f"GDF did not converge on mesh {nk}x{nk}x{nk}")
    previous_nk = max(results)
    delta_mha = 1000 * (res.energy_per_cell_ha - results[previous_nk])
    results[nk] = res.energy_per_cell_ha
    print(f"  Change from {previous_nk}^3: {delta_mha:+.3f} mHa/cell")
    print(f"  [{nk},{nk},{nk}]: E = {res.energy_per_cell_ha:.6f} Ha")

reference = -8.149050  # Peintinger 2013 SI Table 2 (Ha/cell)
print(f"\nPaper-SI reference (HF/pob-TZVP): {reference:.6f} Ha/cell")
last_nk = max(results)
print(f"Difference at {last_nk}^3: {1000 * (results[last_nk] - reference):+.3f} mHa/cell")
```

No measured pob-TZVP sweep is supplied here. The script prints your actual
values; neither reaching the last requested mesh nor a small difference from
the SI number establishes acceptance. Check geometry, basis contractions,
auxiliary fit, lattice support, exchange convention and energy normalization
before interpreting that difference. The primitive cell here contains one LiH
formula unit; do not compare its energy directly with a conventional-cell total.

Do not assume monotonic convergence or sub-millihartree accuracy at 8×8×8.
Record successive mesh increments and independently test the remaining
numerical controls at the accuracy required by your observable. A failed or
unfinished point remains missing, not an interpolated convergence target.

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
# doc-audit: skip - uses lih, basis, aux_basis and vq from the setup above
result = vq.run_krhf_periodic_gdf(
    lih, basis, kmesh=(1, 1, 1), functional="pbe", aux_basis=aux_basis,
)
if not result.converged:
    raise RuntimeError("Gamma-point PBE did not converge")
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
- **Converge more than the mesh.** Lattice support and fitting controls
  must be checked at the chosen basis and cell. A measured tolerance on the
  STO-3G case is not a general accuracy guarantee.
- **Reproducing published references is the goal.** Peintinger 2013 SI
  Table 2 is a CRYSTAL-grade benchmark for the multi-k + GDF + pob-TZVP
  stack (the pob-TZVP comparison above remains a target).

## The dense-Lpq RAM ceiling

`Lpq[k1, k2, L, μ, ν]` is materialised dense up front. Memory
scales as `O(nkpts² × naux × nao²)`. The 8×8×8 LiH/pob-TZVP
mesh is a resource-planning decision, not a default tutorial workload.
Use the actual orbital and auxiliary dimensions and the available memory
before extending the pilot. The number of k points is 512 at 8×8×8, so the
number of ordered k-point pairs is 262,144 before other tensor dimensions:

RSGDF setup shares the auxiliary Coulomb metric, its eigensystem, and its
Schwarz mask once per unique momentum transfer `q = k_j - k_i`. Whitening is
reapplied from that eigensystem while the three-centre factor is still built
per ordered pair because the inter-cell AO product carries the ket momentum
phase. This removes exact duplicated setup work without changing the dense
`Lpq` memory ceiling or the remaining `O(nkpts²)` pair-transform cost.

| System | k-mesh | Lpq dense memory |
|--------|--------|------------------|
| MgO conventional / sto-3g | [4,4,4] | manageable |
| LiH primitive / pob-TZVP | [8,8,8] | estimate with actual AO/auxiliary dimensions before running |
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
