# k-point meshes

Periodic SCF replaces the molecular sum-over-occupied-orbitals with
an integral over the first Brillouin zone (BZ). Practically that
means picking a finite mesh of $\mathbf{k}$-points, solving the SCF
self-consistently at each, and integrating to get the total energy.
This page is the reference for **how to choose, build, and interpret
that mesh**, Monkhorst-Pack vs Γ-centered, IBZ reduction, mesh
convergence, band paths.

For the Bravais-lattice + reciprocal-lattice background, what the
BZ actually *looks like* for FCC, BCC, hexagonal, etc., and where
Γ, X, K, L, M sit, see [crystal lattices](crystal_lattices.md).

Mesh sizes passed to `KPoints` constructors or `monkhorst_pack` must be
positive integers. Python and NumPy integer entries are accepted; floats
(including `2.0`), booleans and strings are rejected before native grid
construction. For lower-dimensional systems, active-axis vectors are padded
and explicit inactive integer entries retain the existing pinning to one
point at Gamma.

The runner, ASE periodic-force/GPW and dimer raw mesh adapters, GDF tuple
and IBZ metadata readers, the
Madelung supercell and BvK density helpers, and four-center CCM use the same
exact integer rule. The runner retains its three-axis scalar repetition;
GDF slab meshes retain their explicit inactive-axis requirement of one.
This count rule does not restrict physical fractional k-point coordinates.

## Why we sample the BZ

For a periodic crystal the Bloch states $\psi_{n\mathbf{k}}(\mathbf{r})$
are labeled by a band index $n$ and a continuous wavevector
$\mathbf{k}$ in the first BZ. The total energy per unit cell is

$$
E = \frac{1}{V_\text{BZ}} \int_\text{BZ} \sum_{n \text{ occ}}
    \varepsilon_{n\mathbf{k}} \, d^3\mathbf{k}
$$

(plus exchange-correlation, electron-electron, nuclear, etc. terms).
We approximate this integral by a finite weighted sum over a mesh
of points:

$$
E \approx \sum_i w_i \sum_{n \text{ occ}} \varepsilon_{n,\mathbf{k}_i}
\quad \text{with} \quad \sum_i w_i = 1.
$$

The choice of mesh, its size, its centring, whether it exploits
crystal symmetry, controls how accurate the integral is. **All
properties** (energy, forces, band gap, density of states) inherit
that accuracy.

## The Monkhorst-Pack mesh

The standard choice. Monkhorst and Pack (1976) proposed sampling on
a uniform grid in *fractional* reciprocal-lattice coordinates:

$$
\mathbf{k}_{p,q,r} = \frac{2p - n_1 - 1}{2 n_1}\,\mathbf{b}_1
                  + \frac{2q - n_2 - 1}{2 n_2}\,\mathbf{b}_2
                  + \frac{2r - n_3 - 1}{2 n_3}\,\mathbf{b}_3
$$

for $p = 1\ldots n_1$, $q = 1\ldots n_2$, $r = 1\ldots n_3$. The
mesh has $n_1 n_2 n_3$ points, each with weight $1/(n_1 n_2 n_3)$
before any symmetry reduction.

Two common conventions for the *origin* of the mesh:

* **Centered** (above), point $(p,q,r) = ((n_1+1)/2, \ldots)$ sits
  at $\mathbf{k} = 0$ only when all three $n_i$ are odd. This is the
  original 1976 Monkhorst-Pack specification.
* **Γ-centered**, shifted so $\mathbf{k} = 0$ is *always* on the
  mesh. Useful for hexagonal lattices where the symmetry of the
  Γ point matters; some VASP-style workflows default to this.

:::{important}
**vibe-qc reaches both, through the same `kpoints=` argument.** Which
one you get depends on how you build the mesh, and the two differ for
even $n_i$ (measured 2026-08-02, cubic cell):

| built with | (2,2,2) | (3,3,3) | (4,4,4) |
|---|---|---|---|
| {py:func}`vibeqc.monkhorst_pack`, or a plain mesh tuple | Γ-centered | Γ-centered | Γ-centered |
| {py:meth}`vibeqc.KPoints.monkhorst_pack` | centered (`is_shift=(1,1,1)`) | Γ-centered | centered (`is_shift=(1,1,1)`) |

`KPoints.monkhorst_pack` applies the classical auto-shift, so it agrees
with ASE/GPAW at even $n_i$; the mesh-tuple form does not. For odd $n_i$
the half-step offset wraps onto the same lattice and all three agree.

An earlier revision of this page called the classical centered
convention "vibe-qc's default". That was true of `KPoints` and false of
the tuple form, which is the more common entry point. Every periodic run
now prints the convention it actually resolved; see
[`multi_k_scf.md`](multi_k_scf.md), so read it from the `.out` rather
than inferring it from the mesh you asked for.
:::

In vibe-qc, the user-facing entry point is the
:class:`vibeqc.KPoints` class (Phase K1, first-class k-point
ergonomics matching what users coming from VASP / CRYSTAL / Quantum
ESPRESSO expect):

```python
import numpy as np
import vibeqc as vq

# 8×8×8 Monkhorst-Pack mesh on a cubic crystal.
sysp = vq.PeriodicSystem(
    dim=3,
    lattice=8.0 * np.eye(3),
    unit_cell=[vq.Atom(11, [0, 0, 0])],
)
kp = vq.KPoints.monkhorst_pack(sysp, mesh=[8, 8, 8])

print(kp.n_kpoints)        # 512 (full mesh)
print(kp.kind)             # "monkhorst_pack"
print(kp.shift)            # (1, 1, 1)  --  classical-MP auto-shift
                           # for even meshes; (0, 0, 0) for odd
print(kp.is_symmetry_reduced)   # False  --  full mesh, no IBZ folding yet
```

Other common builders:

```python
# Γ-centred mesh (k=0 is always on the grid):
kp = vq.KPoints.gamma_centred(sysp, [8, 8, 8])

# Γ-only (single k-point at the BZ origin):
kp = vq.KPoints.gamma(sysp)

# Custom shift on top of the MP base:
kp = vq.KPoints.monkhorst_pack(sysp, [8, 8, 8], shift=(0, 0, 1))

# From an explicit list of fractional k-points (Phase K4):
import numpy as np
kp = vq.KPoints.from_list(
    sysp,
    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]),
    weights=np.array([1, 8, 6]),   # auto-normalised to sum to 1
)
```

For 1D and 2D systems you may pass only the active axes, or the full
three-axis form. vibe-qc pins inactive reciprocal axes to Γ either way:

```python
# 1D polymer: [12] is equivalent to [12, 1, 1].
kp = vq.KPoints.monkhorst_pack(sysp_1d, [12])
print(kp.n_kpoints)      # 12, not 192

# 2D slab: [8, 8] is equivalent to [8, 8, 1].
kp = vq.KPoints.gamma_centred(sysp_2d, [8, 8])
```

The same active-axis shorthand works for explicit fractional k-points:

```python
kp = vq.KPoints.from_list(sysp_2d, [[0.0, 0.0], [0.5, 0.25]])
```

If you provide a full length-3 explicit k-vector for a low-dimensional
system, inactive components must be zero. This catches accidental
sampling along a vacuum axis.

```{tip}
**Back-compat**. The old bare-function ``vq.monkhorst_pack(sysp,
mesh)`` still ships and returns a :class:`vibeqc.BlochKMesh`
directly (the lower-level type that periodic SCF drivers consume).
It accepts the same low-dimensional active-axis shorthand (``[n]`` for
1D, ``[n1, n2]`` for 2D), but preserves its historical gamma-centred
default shift. ``KPoints.monkhorst_pack`` uses the richer Python API
and the classical Monkhorst-Pack auto-shift convention unless you call
``KPoints.gamma_centred``. ``KPoints.to_bloch_kmesh()`` adapts
new-style ``KPoints`` to the legacy format for any driver that hasn't
been updated to take ``KPoints`` natively.
```

### Density-based auto-meshes

For convenience, pick a mesh size from a single physically-motivated
parameter rather than guessing a 3-integer mesh by hand. Three
conventions ship (Phase K5):

```python
# AFLOW convention: ~1000 k-points per reciprocal atom (Curtarolo 2012)
kp = vq.KPoints.from_kppra(sysp, n_kpts_per_atom=1000)

# Materials Project / ASE convention: target k-spacing in 2π/Å
kp = vq.KPoints.from_kspacing(sysp, kspacing=0.20)

# VASP "Auto" convention: a single length parameter (Å)
kp = vq.KPoints.auto(sysp, length=30.0)
```

The KPPRA builder (``from_kppra``) takes an optional ``metallic=True``
flag that bumps the density 4× and warns if smearing isn't enabled,
metals need both fine k-meshes and Fermi-Dirac smearing to converge.
For ``from_kspacing`` / ``auto``, request a denser mesh directly with a
smaller ``kspacing`` (or larger ``length``).

### AUTO mode, let vibe-qc choose (Phase K8)

The density builders above still ask *you* for the spacing. ``KPoints.recommend``
goes one step further: it classifies the system as metal / small-gap /
insulator, picks the target k-spacing Δk for that character, builds a
symmetry-reduced mesh, and hands back a ready-to-use spec bundled with a
recommended smearing and a plain-English rationale. Treat it as a **good
starting point, not a guarantee**, the ``verify=True`` path (below) confirms it.

```python
# Insulator (gap known) → coarser mesh, no smearing.
spec = vq.KPoints.recommend(sysp, band_gap=1.1)      # eV
print(spec.rationale)
# insulator, Δk=0.3 Å⁻¹, Γ-centred, 5×5×5, 10 irreducible k-points

# Metal → denser mesh + a recommended smearing.
# run_periodic_job and run_*_periodic_scf now auto-read .smearing /
# .bz_integration from a recommended KPoints  --  no manual wiring needed.
spec = vq.KPoints.recommend(sysp, is_metal=True)
result = vq.run_periodic_job(sysp, basis, method="RKS", functional="lda",
                             kpoints=spec)  # smearing=spec.smearing is automatic
# For the low-level dispatch: smearing is carried via options.smearing_temperature;
# bz_integration is auto-read from the KPoints as well.

# Metal via the parameter-free Gilat--Raubenheimer net (tetrahedron family)
# instead of smearing. AUTO drops the smearing recommendation and records
# the backend. Pass to run_*_periodic_scf via the bz_integration= kwarg, or
# let run_periodic_job auto-read it from the KPoints metadata.
spec = vq.KPoints.recommend(sysp, is_metal=True, bz_integration="gilat")
# spec.bz_integration == "gilat"; spec.smearing is None.
result = vq.run_rhf_periodic_scf(sysp, basis, kmesh=spec, opts=opts,
                                 bz_integration=spec.bz_integration)
```

The character signal is resolved in priority order: ``band_gap`` (eV) →
``is_metal`` → an optional ``classifier(system)`` hook → otherwise the
**SAFE default** (treat as metal, the dense over-converged choice) with a
warning that character was assumed. Δk targets are
insulator ≈ 0.30, small-gap ≈ 0.18, metal ≈ 0.12 Å⁻¹ (Materials-Project
KSPACING units, i.e. 2π/Å); all thresholds are named constants at the top
of ``vibeqc/kpoints.py`` so they are easy to tune. Hexagonal/trigonal
lattices get a Γ-centred grid automatically; very large supercells collapse
to Γ-only; slab/wire vacuum axes stay at ``N = 1``.

For metals you can pick how the Brillouin zone is integrated via
``bz_integration``: the default ``None`` / ``"smearing"`` recommends
temperature broadening (a :class:`SmearingOptions`), while ``"gilat"`` selects
the parameter-free **Gilat-Raubenheimer net** (the tetrahedron-family, CRYSTAL
``SHRINK IS ISP`` analogue), a T = 0 integrator with no smearing width. AUTO
then drops the smearing recommendation (``spec.smearing is None``) and records
``spec.bz_integration == "gilat"`` to pass to the multi-k RHF driver's
``bz_integration=`` argument. The net runs on the efficient symmetry-reduced
(IBZ) mesh, with the occupation layer expanding eigenvalues to the full BZ
internally, so it composes with the default IBZ reduction. The public BIPOLE
route supports RKS plus per-spin UHF/UKS Gilat occupations; BIPOLE RHF, ROHF,
and ROKS remain gated. The standalone multi-k Ewald RHF, RKS, UHF, and UKS
drivers also expose the selector.

For a gapped system Gilat occupations reduce to the same integer fixed point
as Aufbau occupations. A sharp T = 0 occupation map is discontinuous when
bands cross the Fermi level, so Gilat-driven metallic SCF is not a supported
convergence route. Converge metallic densities with smearing, then apply the
Gilat occupations and DOS post-SCF.

Pass ``verify=True`` with an ``scf_energy_fn`` to run a convergence ladder
(Δk·√2, Δk, Δk/√2, …) that refines until the total energy/atom changes by
less than ``tolerance_meV_per_atom`` between rungs, returning the converged
mesh with the ladder attached as ``spec.verification``
(a :class:`~vibeqc.KPointConvergence`):

```python
spec = vq.KPoints.recommend(
    sysp, band_gap=1.1, verify=True,
    scf_energy_fn=lambda kp: vq.run_rks_periodic_scf(
        sysp, opts, kpoints=kp).energy,
)
print(spec.verification.converged, spec.verification.chosen_mesh)
```

An advanced ``predictor=`` hook accepts a machine-learned Δk model
(``predictor(features) -> (Δk, σ)``); AUTO uses the conservative end of the
predicted interval.  A **bundled model** ships with vibe-qc  --  a
scikit-learn random-forest regressor trained on synthetic data that
replicates the scaling laws observed in the Choudhary--Tavazza
k-point-convergence predictor (Choudhary & Tavazza, *npj Comput. Mater.*
**6**, 39, 2020), using ~600 synthetic training points that encode the
metal/insulator character, cell volume, atom count, and dimensionality.

**Quick start (bundled model):**

```python
# 1. Install the optional scikit-learn dependency
#    pip install -e '.[ml]'

# 2. Enable the model (never auto-loaded silently)
import os
os.environ["VIBEQC_ML_KPOINTS"] = "1"

# 3. Use it
spec = vq.KPoints.recommend(sysp, band_gap=1.1, predictor="ml")
print(spec.rationale)
# insulator, Δk=0.26 Å⁻¹ (ML: μ=0.29, σ=0.03), Γ-centred, 5×5×5, …
```

**Accuracy estimates:**

- Training MAE: ~0.016 Å⁻¹  --  accurate to about one mesh increment on a
  typical 5 Å cell.
- Ensemble standard deviation (σ): ~0.025 Å⁻¹, subtracted from the mean
  for the conservative-end logic.  The conservative mesh (Δk − σ) is
  ~0.015−0.05 Å⁻¹ denser than the mean  --  about one extra k-point per
  reciprocal direction  --  and thus rarely under-samples.
- The model is a *starter predictor* trained on synthetic data derived
  from JARVIS-DFT scaling heuristics (CC-BY-4.0, Choudhary et al.,
  *npj Comput. Mater.* **6**, 173, 2020).  When production JARVIS-DFT-
  trained weights are available, the same ``.pkl`` swap upgrades it
  with no code change.

**Gate env var:** ``VIBEQC_ML_KPOINTS=1``.  The bundled model is never
loaded silently  --  without this variable, ``predictor="ml"`` raises
``NotImplementedError``, and the ``ml_predictor=<callable>`` user-
supplied path is unaffected.

**Bring your own model:** pass ``ml_predictor=<callable>`` returning
``(Δk, σ)`` (no env var needed); see Choudhary & Tavazza (2020) and
conformal-quantile-regression k-spacing models for training guidance.

### Generalized regular grids

Generalized regular (GR) grids keep the uniform-grid structure but
allow sheared integer generators instead of only axis-aligned
Monkhorst-Pack subdivisions. They are useful for skewed or anisotropic
3D cells where a rectangular MP mesh wastes points.

```python
# On-the-fly HNF search: exactly 96 k-points.
kp = vq.KPoints.optimal(sysp, target_n_kpts=96)
print(kp.kind)          # "generalized-regular"
print(kp.grid_matrix)   # 3x3 integer HNF; det == 96

# Build a specific generalized grid explicitly.
kp = vq.KPoints.generalized_regular(
    sysp,
    [[2, 1, 0],
     [0, 2, 0],
     [0, 0, 2]],
)

# Remote GMU/NRL pre-generated weighted table lookup. No table data is
# bundled; this fetches the table on demand and returns a weighted
# explicit mesh.
kp = vq.KPoints.from_database(sysp, lattice="fcc", order=6, special=True)
kp = vq.KPoints.from_database(sysp, lattice="hex", order=(6, 3))
```

`KPoints.optimal(..., metallic=True)` uses a 4x effective target count,
matching the density-helper convention for Fermi-surface integration.
The on-the-fly GR search is 3D-only and does not require a network
database. `KPoints.from_database(...)` is deliberately explicit about
network access: it supports the public GMU/NRL `sc`, `bcc`, `fcc`, and
`hex` tables, normalizes the returned symmetry weights, and converts
the lattice-coordinate k-points through the system's reciprocal
lattice.

## Equivalent k-points and IBZ reduction

The crystal's space-group symmetry maps points in the BZ onto each
other: $\mathbf{k}$ and $R\mathbf{k}$ (for $R$ a point-group rotation)
give *identical* energies and properties. Including both is
redundant. The **irreducible Brillouin zone (IBZ)** is the smallest
piece of the BZ such that no two points within it are
symmetry-related; the rest is recoverable by applying the symmetry
operations.

For an FCC lattice with cubic point-group $O_h$ (48 operations), an
8×8×8 mesh has 512 points but only **35 in the IBZ**, a 14× speedup
on the SCF if you exploit it.

In vibe-qc you opt in via the ``KPoints.symmetry_reduce()``
builder method (Phase K2, IBZ reduction via spglib):

```python
from vibeqc import attach_symmetry

attach_symmetry(sysp)             # spglib detects the space group
kp = vq.KPoints.monkhorst_pack(sysp, [8, 8, 8])
kp_irr = kp.symmetry_reduce()
print(kp_irr.n_kpoints)           # ~35 for FCC
print(kp_irr.is_symmetry_reduced) # True
print(sum(kp_irr.weights))        # still 1.0  --  symmetry-equivalent
                                   # points fold into boosted weights
```

The returned ``KPoints`` instance carries the multiplicity per
point through the ``weights`` array; downstream periodic-SCF
drivers see a smaller k-list with the right BZ-integration
weighting.

```{warning}
**Hex / trigonal cells (SG 143--194) refuse non-zero MP shifts**  --
the classical (½,½,½) offset breaks the three-fold symmetry, so
``symmetry_reduce()`` raises an actionable error pointing at
``KPoints.gamma_centred(...)`` (the safe choice for those
spacegroups).
```

```{note}
**How the SCF drivers treat a reduced mesh** (as of v0.15.85+):

* The multi-k **EWALD_3D** drivers (``run_rhf_periodic_scf``,
  ``run_rks_periodic_scf``, ``run_uks_periodic_multi_k_ewald3d``)
  run **wedge-native** for HF and global hybrids: diagonalisation
  happens only at the irreducible points, and the density is
  symmetry-unfolded to the full BZ for the exchange sum. This is
  the genuine O(N_IBZ) saving.
* The multi-k **GDF** drivers (``run_krhf_periodic_gdf`` /
  ``run_krks_periodic_gdf`` and the open-shell twins) accept a
  reduced mesh by **expanding it up front to its full parent
  mesh** (same dims, same shift). Correct by construction, but
  every full-mesh point is still built and diagonalised: no
  wedge saving on this route yet.

See the ``jk_method`` route table in
[periodic methods](periodic_methods.md) for per-route status.
```

## Choosing the mesh size

The minimum mesh that gives a converged total energy depends on the
system. The right approach is always: **start with a small mesh,
double, recompute, stop when the energy stops moving**.

Rough starting points:

| System type | Suggested starting mesh |
|---|---|
| Finite cluster in periodic box (molecular limit) | $[1, 1, 1]$ |
| 1D polymer | $[6, 1, 1]$ |
| 2D slab / surface | $[6, 6, 1]$ |
| 3D wide-gap insulator (NaCl, MgO) | $[4, 4, 4]$ |
| 3D narrow-gap semiconductor (Si, GaAs) | $[8, 8, 8]$ |
| 3D metal | $[12, 12, 12]$ + smearing (see roadmap) |

**Anisotropic cells** want anisotropic meshes. A surface slab with
$c \gg a$ doesn't need many k-points perpendicular to the surface
(the bands there are flat). HCP with $c/a \approx 1.6$ wants
roughly equal sampling in-plane and along $c$.

```python
# Slab: lots of in-plane k, very few perpendicular
kp = vq.KPoints.monkhorst_pack(sysp_slab, [8, 8, 1])

# HCP: mesh density along c roughly matches in-plane
kp = vq.KPoints.monkhorst_pack(sysp_hcp, [6, 6, 4])
```

The general rule of thumb is **roughly equal density** along each
direction in *reciprocal space*: $n_i \cdot |\mathbf{a}_i|$ should
be roughly equal across $i = 1, 2, 3$. So a long real-space
direction wants fewer k-points along that direction.

### Convergence study workflow

```python
import vibeqc as vq

results = {}
for n in [2, 4, 6, 8, 10, 12]:
    kp = vq.KPoints.monkhorst_pack(sysp, [n, n, n])
    result = vq.run_rhf_periodic_scf(sysp, basis, opts,
                                      kmesh=kp.to_bloch_kmesh())
    results[n] = result.energy
    print(f"  n={n}:  E = {result.energy:.6f} Ha   |Δ| from n-2: "
          f"{abs(result.energy - results.get(n-2, result.energy)) * 1e3:.3f} mHa")

# Stop doubling when |Δ| drops below your target  --  typically 1 mHa
# for energies, 1 µHa for forces (forces converge faster than
# energy in the k-mesh limit).
```

[Band structure and density of states](../tutorial/band_structure.md) walks through a
convergence study on a 1D H-chain. For **GDF-specific multi-k**
convergence (including crystal-type mesh recommendations and IBZ
symmetry reduction), see [multi-k SCF](multi_k_scf.md).

```{tip}
**For metals** (or near-metals), the total energy is **discontinuous**
in the k-mesh  --  the Fermi level can drop below or rise above
discrete eigenvalues as the mesh changes. The fix is **smearing**
(Fermi-Dirac, Gaussian, or Methfessel-Paxton): occupations become
fractional near the Fermi level, smoothing the integral.
Vibe-qc's Fermi-Dirac smearing landed in
[Phase C1b](../roadmap.md); without it, k-mesh convergence on
metals is essentially impossible. **Insulators don't need
smearing**  --  the gap means the occupations are 0 or 2 by
definition.
```

## Bands: k-paths through the BZ

For band-structure plots you want a **path** of k-points along the
edges of the IBZ, not a uniform mesh. The standard convention
(Setyawan-Curtarolo 2010) defines a path per Bravais lattice:

| Lattice | Standard path |
|---|---|
| Cubic (cP) | Γ → X → M → Γ → R → X → M → R |
| FCC (cF) | Γ → X → W → K → Γ → L → U → W → L → K |
| BCC (cI) | Γ → H → N → Γ → P → H → P → N |
| Hexagonal (hP) | Γ → M → K → Γ → A → L → H → A → L → M → K → H |
| Tetragonal (tP) | Γ → X → M → Γ → Z → R → A → Z → X → R → M → A |

Each segment is sampled with N points (typically 30-60 per leg);
the SCF *result* is post-processed to give $\varepsilon_n(\mathbf{k})$
along the path. This is **non-self-consistent**, the density from
the converged uniform-mesh SCF is fixed; only the band energies are
re-evaluated at the path k-points.

In vibe-qc, the lattice-aware path is auto-detected via
[seekpath](https://github.com/giovannipizzi/seekpath) using the
HPKOT 2017 convention (Phase K3). One call:

```python
# Auto-detected path for the system's Bravais lattice (HPKOT/Hinuma):
kp_path = vq.KPoints.band_path(sysp, scheme="auto",
                                points_per_segment=30)
print(kp_path.kind)        # "band_path"
print(kp_path.n_kpoints)   # ~210 for FCC's standard 7-leg path

# Explicit override  --  pass your own segments:
kp_path = vq.KPoints.band_path(
    sysp,
    scheme="manual",
    segments=[("Γ", "X"), ("X", "W"), ("W", "K"), ("K", "Γ")],
    points_per_segment=40,
)

# Convert to the band-energy evaluation format:
bands = vq.compute_bands(sysp, basis, scf_result,
                          kp_path.to_kpath())
# bands.shape = (n_kpath_points, n_bands)
```

The auto-detection step calls spglib + seekpath under the hood:
spglib identifies the Bravais lattice, seekpath returns the
canonical path for that lattice in the HPKOT convention. Manual
override is available for cases where you want a non-standard
path or a specific high-symmetry segment.

See [band structure of an H-chain](../tutorial/band_structure.md)
for the complete plotting workflow, and
[`examples/periodic/input-h-chain-bands.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-bands.py)
for the runnable script.

## Density of states

The density of states (DOS) is

$$
g(\varepsilon) = \sum_n \int_\text{BZ} \delta(\varepsilon -
\varepsilon_{n\mathbf{k}}) \, \frac{d^3\mathbf{k}}{V_\text{BZ}}.
$$

Like the energy, this needs a finite k-mesh, but the right
mesh for DOS is typically *finer* than for energy. Energy
integrates the *occupied* states, where $\varepsilon < \varepsilon_F$
is well-defined; DOS evaluates the integrand at every $\varepsilon$,
including in regions where the bands are flat (high DOS), the mesh
needs to resolve those features.

Practical rule: **double the mesh size** you used for energy
convergence to get a smooth DOS. A 16×16×16 mesh on Si gives a
visually clean DOS where 8×8×8 was sufficient for the total energy.

vibe-qc's DOS API:

```python
from vibeqc import compute_dos

kp_dense = vq.KPoints.monkhorst_pack(sysp, [16, 16, 16])
dos = vq.compute_dos(sysp, basis, scf_result,
                     kmesh=kp_dense.to_bloch_kmesh(),
                     sigma=0.05)  # eV broadening
# dos.energies, dos.dos
```

See [the band-structure user guide](band_structure.md) for the
full DOS + projected DOS (PDOS) workflow.

## Smearing for metals

When the occupation function $f(\varepsilon)$ is sharp (a step at
$\varepsilon_F$), the BZ integral has discontinuities every time a
band crosses the Fermi level, a finite mesh can't resolve them
cleanly. The fix is to *broaden* the step:

* **Fermi-Dirac**: $f = (1 + e^{(\varepsilon - \mu)/k_B T})^{-1}$.
  Has a thermodynamic interpretation (electronic temperature $T$);
  use this for finite-T studies.
* **Gaussian**: $f$ ≈ erf-shaped. No physical interpretation, but
  converges faster than Fermi-Dirac at the same broadening width.
* **Methfessel-Paxton**: a hierarchy of corrections that approach
  the zero-T limit as the order increases. Common in plane-wave DFT.

Vibe-qc currently exposes Fermi-Dirac smearing through
``smearing_temperature`` on the periodic SCF options (Phase C1b).
This is ``k_B T`` in Hartree, not Kelvin; use
``vq.kelvin_to_hartree_temperature(T_K)`` or
``vq.resolve_smearing_temperature("1000 K")`` when you want to specify
a physical electronic temperature. Numeric strings can carry units such
as ``"0.1 eV"``, ``"0.01 Ha"``, and ``"0.02 Ry"``. Gaussian and
Methfessel-Paxton are tracked on the roadmap.

```python
opts = vq.PeriodicSCFOptions()
opts.smearing_temperature = vq.kelvin_to_hartree_temperature(300.0)
kp = vq.KPoints.monkhorst_pack(sysp, [12, 12, 12])
result = vq.run_rks_periodic_scf(
    sysp, basis, kp.to_bloch_kmesh(), opts,
)
# Reported finite-T fields: result.free_energy, result.entropy,
# result.fermi_level, result.occupations
```

## Worked example: NaCl rocksalt

Putting it together, a converged SCF on rocksalt NaCl with a
properly-sized k-mesh:

```python
import numpy as np
import vibeqc as vq
from vibeqc import attach_symmetry

# 1. Build the lattice + basis (FCC primitive + 2-atom basis).
a = 10.65   # bohr (NaCl: a = 5.64 Å)
sysp = vq.PeriodicSystem(
    dim=3,
    lattice=(a / 2) * np.array([
        [0, 1, 1], [1, 0, 1], [1, 1, 0],
    ]),
    unit_cell=[
        vq.Atom(11, [0.0, 0.0, 0.0]),
        vq.Atom(17, [0.5, 0.5, 0.5]),
    ],
)

# 2. Symmetry analysis  --  informs the k-mesh size.
attach_symmetry(sysp)

# 3. Convergence-screened k-mesh; 6×6×6 is fine for NaCl
#    (wide-gap insulator). symmetry_reduce() builds the IBZ via
#    spglib (Phase K2).
kp = vq.KPoints.monkhorst_pack(sysp, [6, 6, 6]).symmetry_reduce()
print(f"Full mesh: {6**3} points; IBZ: {kp.n_kpoints} points "
      f"({6**3 / kp.n_kpoints:.1f}× speedup if SCF supports it)")

# 4. Converged SCF.
basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
result = vq.run_rks_periodic_scf(sysp, basis, opts,
                                 kmesh=kp.to_bloch_kmesh(),
                                 omega=0.5, spacing_bohr=0.5,
                                 functional="PBE")
print(f"E(NaCl) = {result.energy:.6f} Ha/cell  "
      f"({result.n_iter} iters, converged={result.converged})")
```

For more end-to-end periodic examples, including the Madelung-
constant cross-validation that exercises the k=0 limit of this
machinery, see the
[example scripts and generated-output guide](../example_outputs.md).

## References

* H. J. Monkhorst and J. D. Pack, "Special points for Brillouin-zone
  integrations," *Phys. Rev. B* **13**, 5188 (1976), defines the
  uniform-grid scheme that bears their name.
* W. Setyawan and S. Curtarolo, "High-throughput electronic band
  structure calculations: Challenges and tools," *Comput. Mater.
  Sci.* **49**, 299 (2010), defines the standard band-path
  conventions per Bravais lattice.
* M. Methfessel and A. T. Paxton, "High-precision sampling for
  Brillouin-zone integration in metals," *Phys. Rev. B* **40**,
  3616 (1989), the corrected-Gaussian smearing scheme.
* N. Marzari, D. Vanderbilt, and others, "Thermal contraction and
  disordering of the Al(110) surface," *Phys. Rev. Lett.* **82**,
  3296 (1999), discussion of finite-T smearing for metallic
  surfaces.
* The [VASP wiki KPOINTS page](https://www.vasp.at/wiki/index.php/KPOINTS)
  is the de-facto reference for plane-wave DFT k-mesh conventions;
  most of what's there carries over to Gaussian-basis periodic
  SCF, modulo "smooth-cutoff" considerations that are PW-specific.

## See also

* [`periodic_methods.md`](periodic_methods.md): comparative tour of
  the periodic-SCF kernels (BIPOLE, GDF, GPW/GAPW) the k-mesh
  feeds, recommended `(N, N, 1)` slab defaults, and the
  surface-reactions workflow.
