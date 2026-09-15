---
myst:
  html_meta:
    "description": "Comparative tour of vibe-qc's periodic-SCF strategies: BIPOLE (an Ewald J-split), native GDF (Gaussian density fitting), and the GPW / GAPW (Lippert-Hutter) plane-wave route. Theory, vibe-qc implementation, advantages, disadvantages, and a method-selection chart for 1D / 2D / 3D systems."
    "og:title": "vibe-qc - periodic-SCF methods: BIPOLE, GDF, GPW / GAPW"
    "og:description": "Side-by-side comparison of the three periodic Coulomb-build routes in vibe-qc, with theory, dimensionality coverage, pros/cons, references and examples."
---

# Periodic-SCF methods: BIPOLE, GDF, GPW / GAPW

This page is the comparative tour of the periodic Hartree-Fock and
Kohn-Sham machinery in vibe-qc. It is meant to be the page you read
*first* when you need to pick a route for a new crystalline (or
periodic slab / wire) calculation. The per-method deep dives are
in the specialised pages linked from each section.

Periodic boundary conditions are vibe-qc's headline focus, and
**surface chemistry (slab single-points, adsorption energies,
reaction-path search, transition-state characterisation) is the
flagship workflow.** Section 7 walks the surface-reactions stack
end-to-end; sections 2 to 5 are the per-kernel theory plus
implementation reference that workflow sits on top of.

```{note}
**The cyclic cluster model (CCM): a real-space alternative.** Beyond the
k-point routes on this page, vibe-qc's signature periodic approach is the
[cyclic cluster model](cyclic_cluster_model.md), which treats a crystal as a
finite cyclic cluster rather than sampling the Brillouin zone. Two independent
experimental ab-initio CCM lines ship in the current release:
[Γ-CCM / `aiccm2026dev-a`](../aiccm2026dev_a.md)
(union-and-weight/Wigner--Seitz integral weighting) and
[χ-CCM / `aiccm2026dev-b`](aiccm2026dev_b.md)
(finite-translation-group characters; 3-D SCF and correlation only). They are
distinct approaches; a matched exchange-q=0 convention is necessary for a
comparison but does not define an approach delta. Both are selected through
one front door, `run_periodic_job(method="aiccm", variant=...)`, documented
on the [AICCM page](aiccm.md). The semi-empirical (MSINDO)
CCM is also available for bulk, surface, and adsorption calculations.
```

```{seealso}
* [`periodic_systems.md`](periodic_systems.md): the `PeriodicSystem`
  container, Born-von Karman PBC contract, and basic 1D / 2D / 3D
  setup.
* [`crystal_lattices.md`](crystal_lattices.md): the 14 Bravais
  lattices, with visualisations and worked examples for the common
  cases (rocksalt, diamond, perovskite, HCP, wurtzite, rutile,
  corundum, alpha-quartz, graphene).
* [`k_points.md`](k_points.md): Monkhorst-Pack mesh generation and
  Brillouin-zone sampling.
* [`bipole.md`](bipole.md): full BIPOLE driver reference.
* [`cyclic_cluster_model.md`](cyclic_cluster_model.md): the cyclic cluster
  model, vibe-qc's real-space (Born-von Karman) alternative to the k-point
  routes on this page.
* [`density_fitting.md`](density_fitting.md): RIJ / RIJK / RIJCOSX
  for molecules, plus the GDF kernel reuse for periodic Gamma-only.
* [`multi_k_scf.md`](multi_k_scf.md): the multi-k SCF surface,
  smearing for metals, and the multi-k GDF roadmap.
* [`ewald.md`](ewald.md): the Ewald summation primitives that the
  one-electron and the long-range J builds share.
```

## 1. Dimensionality model and the periodic Coulomb problem

Every periodic calculation in vibe-qc lives in a single container,
`PeriodicSystem(dim, lattice, unit_cell)`:

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem

# 1D atomic wire (H-H along x). Columns 1 and 2 are vacuum.
sys_1d = PeriodicSystem(
    dim=1,
    lattice=np.diag([4.0, 30.0, 30.0]),
    unit_cell=[Atom(1, [0, 0, 0]),
               Atom(1, [0, 0, 1.4])],
)

# 2D graphene-like slab. Prefer vq.slab_2d: it takes only the two
# in-plane vectors and synthesizes the (non-physical) third column for
# you. The SCF is invariant to that column; it is not a vacuum gap.
sys_2d = vq.slab_2d(
    [4.6487, 0.0000, 0.0],       # a1 (bohr)
    [2.3244, 4.0259, 0.0],       # a2 (bohr)
    [Atom(6, [0.0000, 0.0000, 0.0]),
     Atom(6, [2.3244, 1.3420, 0.0])],
)

# 3D crystalline bulk (silicon, conventional cubic cell, bohr).
sys_3d = PeriodicSystem(
    dim=3,
    lattice=np.diag([5.4, 5.4, 5.4]),
    unit_cell=[Atom(14, [0, 0, 0]),
               Atom(14, [0.25, 0.25, 0.25])],
)
```

`dim` selects how many of the three lattice columns are physical.
The lattice matrix is always 3x3 and may be triclinic. Reciprocal
vectors are generated automatically as $2\pi A^{-T}$ so that
$a_i \cdot b_j = 2\pi \delta_{ij}$ holds for any cell metric. There
are no cubic-only or orthorhombic-only fast paths in the public
periodic API: methods accept the full lattice matrix and the
test suite pins triclinic, monoclinic, orthorhombic, tetragonal,
trigonal, hexagonal, and cubic metrics on every periodic route.

### Why a naive lattice sum diverges

The fundamental difficulty is that the periodic two-electron
operator,

$$
J_{\mu\nu}[D]
  = \sum_{\mathbf{T}}\sum_{\lambda\sigma}
    D_{\lambda\sigma}\,
    (\mu_{\mathbf{0}}\nu_{\mathbf{0}} | \lambda_{\mathbf{T}}\sigma_{\mathbf{T}}),
$$

contains a tail that decays like $1/|\mathbf{T}|$. In 3D the sum
of $1/|\mathbf{T}|$ over a Bravais lattice is conditionally
convergent: it depends on the order of summation, and a brute
real-space truncation can be off by tens or hundreds of Hartree
on ionic crystals like MgO. The same holds for the
electron-nucleus and nucleus-nucleus interactions, where it is
just the classical Madelung problem. Periodic codes therefore
always do one of three things:

1. **Split the Coulomb interaction.** Write
   $1/r = \mathrm{erfc}(\omega r)/r + \mathrm{erf}(\omega r)/r$.
   The short-range piece is summed in real space; the long-range
   smooth piece is summed in reciprocal space. This is the Ewald
   construction. BIPOLE and the GDF Ewald-3D path do this; so does
   the existing FFT-Poisson driver.
2. **Density-fit on a finite auxiliary basis.** Project the AO
   pair density onto a charge-compensated auxiliary basis so that
   the divergent monopole contribution is cancelled at the source.
   The resulting fit Coulomb metric is finite. This is the GDF
   approach (Gaussian density fitting).
3. **Carry the density on a uniform grid and solve the Poisson
   equation with FFT.** Pin $V(\mathbf{G} = 0) = 0$ for the
   neutral cell. The smooth-grid Hartree-J is then an
   $\mathcal{O}(N_g \log N_g)$ operation, and a Gaussian
   augmentation around each nucleus restores all-electron
   accuracy. This is the GPW / GAPW approach.

vibe-qc ships an implementation of each of these three families.
They coexist, and the active method (with all of its parameters)
is recorded in the `.system` manifest so reproducibility never
relies on the AUTO heuristic.

### Where each method plugs in

The user-facing dispatch lives in
[`PeriodicJKMethod`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic_jk_method.py):

```python
import vibeqc as vq
from vibeqc import PeriodicJKMethod

# Three ways to be explicit about the Coulomb route:
result = vq.run_periodic_job(sysp, basis, method="RHF",
                             jk_method="bipole")
result = vq.run_periodic_job(sysp, basis, method="RHF",
                             jk_method="gdf")
result = vq.run_periodic_job(sysp, basis, method="RHF",
                             jk_method=PeriodicJKMethod.GDF)

# The ab initio cyclic cluster model (experimental): method="aiccm" plus a
# variant; the SCF reference is inferred (functional given -> KS, singlet -> RKS).
result = vq.run_periodic_job(sysp, basis, method="aiccm", variant="chi",
                             functional="pbe0",
                             aiccm_lattice_extension=(4, 4, 4),
                             aiccm_backend="rijcosx")
```

`jk_method="auto"` (the default) branches on the system's
dimensionality first:

* **`dim=2` (a slab)**: always **SLAB_EWALD_2D**, the vacuum-free 2D
  Ewald gauge, for RHF / RKS / UKS. This AUTO choice remains the general
  route. Closed-shell RHF/RKS may explicitly select the bounded slab-GDF
  route; unsupported bulk builders still raise rather than returning an
  `a3`-dependent energy.
* **`dim=1`**: **GDF** for RHF / RKS / UHF / UKS. There is no maintained
  public dim=1 ROHF route.
* **`dim=3`**: **GDF** for closed-shell RHF / RKS and **BIPOLE** for
  open-shell UHF / UKS.

The resolved method (never the literal `"auto"`) is logged in the banner
so a re-run does not depend on the AUTO heuristic version.

A method-by-method status table, kept in sync with
`python/vibeqc/periodic_jk_method.py`:

For the experimental CCM selectors, Γ-CCM and χ-CCM[^xccm-convention] are
distinct construction approaches: union-and-weight/Wigner-Seitz integral
weighting and the
finite-translation-group character construction, respectively. Comparisons
hold the declared exchange-q=0 convention fixed as a comparison constraint;
that convention does not identify the constructions.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

| `jk_method=` (AICCM rows: `method="aiccm", variant=`) | What it dispatches to | Status |
|---|---|---|
| `"slab_ewald_2d"` | Rigorous vacuum-free 2D (slab) Coulomb, the Parry / de Leeuw-Perram gauge. RHF / RKS / UKS at Gamma and multi-k. **This is what AUTO selects for `dim=2`**, so you rarely name it. `dim=2` only; raises on bulk or polymer cells. Total energy is invariant to the synthesized `a3`; `E_nuclear` is k-mesh independent. No analytic gradients on this direct route (use explicit `"gdf"` for closed-shell slab forces), no smearing, no `UHF` yet. Pure and screened-exchange envelopes are supported; full-range multi-k RKS fails closed, while the legacy RHF/UKS full-range exchange paths warn and remain provisional. Aliases: `"slab"`, `"ewald_2d"`. | Production for pure/screened envelopes; full-range RHF/UKS provisional |
| `"gdf"` | Native Gaussian density fitting. Closed-shell RHF / RKS / full-range hybrids and maintained-preview ROHF are public at Gamma and multi-k; ROHF uses `run_krohf_periodic_gdf` and integer 2/1/0 occupations. On `dim=2`, explicit GDF uses the dedicated signed slab-truncated metric and rigorous Parry-gauge one-electron channels on a full Gamma-centered `(n1,n2,1)` tuple mesh. Symmetry-reduced (IBZ) meshes from `KPoints.monkhorst_pack(..., symmetry=True)` are accepted: the driver expands them to the full BZ before the HF exchange sum (correctness, not a speed win: every full-mesh point is still built and diagonalised); a mesh that cannot account for its own full-mesh parent (e.g. an explicit `KPoints` list without that provenance) still fails closed rather than being expanded against fabricated metadata. Slab analytic gradients ship (2026-07-30): `compute_gradient=True` / `optimize=True` on the closed-shell slab route. Slab open shell, ROKS/GDF, smearing, restarts, and DFT+U fail closed. In 1D/3D, open-shell UHF / UKS remains available at Gamma and multi-k. AUTO selects GDF for 1D closed- and open-shell work, `slab_ewald_2d` for every `dim=2` job, and BIPOLE for 3D open shell. Full-grid external XC is experimental and zero-temperature only on the complete periodic AO-density route. | Production in 1D/3D; ROHF maintained preview; verified closed-shell energy + analytic-gradient route in 2D; external XC experimental |
| `method="aiccm", variant="chi"` (legacy `jk_method="aiccm2026dev-b"`, `"chi"`, `"chi-ccm"` resolve with a `DeprecationWarning`) | χ-CCM, the finite-translation-group character construction evaluated on a Γ-centred character mesh. `aiccm_lattice_extension=(N1,N2,N3)` defines the real cyclic dimensions and derives the character net; `aiccm_wigner_seitz_shells=s` requests odd dimensions `2s+1`. `aiccm_backend=` selects `"four_center"`, `"ri"`, or `"rijcosx"`. All backends are 3-D only; every 1-D/2-D request fails closed. Pure full-grid external XC is accepted for zero-temperature RKS/UKS only with `four_center`; fitted backends and external-provider hybrids remain gated. See [the derivation](../design_aiccm2026dev_b.md) and the [AICCM page](aiccm.md). | Experimental, opt-in; 3-D only |
| `method="aiccm", variant="real-gamma"` (legacy `jk_method="real-gamma"`, `"real_gamma"` warn) | The neutral Γ-CCM construction in its k-free real-Γ supercell representation, through the per-unit-cell adapter `periodic/ccm/real_gamma_runner.py`. RHF / RKS / UHF / UKS including screened hybrids; `aiccm_lattice_extension` (or the legacy Γ-centred `kpoints` alias, not both) is the BvK torus; fixed-cell `optimize=True` works for ordinary functionals. Zero-temperature pure full-grid external XC is accepted for RKS/UKS, but its gradients and optimization remain gated. No DFT+U, smearing, Hessian, or cell relaxation; an explicit `damping`, `fock_mixing`, `fmixing_percent`, `density_mixer` or `level_shift` fails closed because the direct-torus loop would drop it. | Experimental, opt-in; 3-D only |
| `method="aiccm", variant="neutral-bloch"` | The same neutral construction in the Bloch representation, through the producer `run_ccm_*_gdf` (the unit-cell GDF drivers on the torus mesh) and the adapter `periodic/ccm/neutral_bloch_runner.py`. RHF / RKS / UHF / UKS; the Gamma-centred mesh is the BvK torus; the producer refuses a vacuum-padded cluster and a positive converged neutral energy, and the arm forwards no mixing control. Full-grid external XC fails closed pending proof that its atom-block periodic partition is representation invariant. No gradients, DFT+U or restart. The former `jk_method` aliases `"neutral-bloch"`, `"bloch-control"`, `"gdf-control"`, `"aiccm-ri"` (plain unit-cell GDF) are retired and fail closed. | Experimental, opt-in; 3-D only |
| `method="aiccm", variant="four-center"` (legacy `jk_method="aiccm2026dev-a"` warns) | The union-and-weight / Wigner-Seitz four-centre Γ-CCM lineage through the per-unit-cell adapter. RHF, RKS, UHF, and UKS run in 3-D; HF uses the scalable lattice-sum builder, while the front door always selects the symmetric `aiccm2026dev-a` weighting. Zero-temperature pure full-grid external providers are accepted for RKS/UKS. External-provider hybrids and ECP-bearing external-XC jobs remain gated, as do DFT+U, smearing, gradients and optimization, Hessians, and explicit mixing controls. The bare `"gamma"`, `"gamma-ccm"`, `"gamma_ccm"` spellings fail closed with the ruling-R1 message on both surfaces. | Experimental, opt-in; 3-D only |
| `"bipole"` | an exact Ewald J-split; multi-k RHF / UHF / RKS / UKS, plus maintained-preview ROHF and ROKS (Gamma + full Monkhorst-Pack meshes, energies only; ROHF and nondegenerate ROKS use integer 2/1/0 occupations, while an exactly degenerate one-electron ROKS shell uses a fixed equal-occupation ensemble; ROKS covers pure functionals and global hybrids) on the corrected-Ewald-exchange EWALD_3D engine; production FD forces plus fixed-cell atomic optimisation; variable-cell work fails closed and analytic gradients remain a gated preview. The public route is 3D-only and rejects `dim=1` and `dim=2`; use GDF for wires, `"auto"` / `"slab_ewald_2d"` for surfaces, or a physical `dim=3` model when BIPOLE forces are required. Full-grid external XC is experimental, zero-temperature only, and requires the corrected gauge plus a complete Monkhorst-Pack mesh or valid expandable IBZ. | Production (3D exact route); external XC experimental |
| `"direct"` | 4-centre real-space lattice sum with Schwarz screening. Valid only for vacuum-padded molecular-limit cells. | Diagnostic |
| `"fft_poisson"` | **Retired (v0.13.0).** The Γ-only EWALD_3D path returned a ~2 Ha-wrong energy on cells where periodic AO images overlap (its molecular-limit density convention `D(g≠0)=0` breaks down, e.g. LiH rocksalt / STO-3G, whose Li 2s/2p are very diffuse). Requesting it now raises; use `"gdf"` (default) / `"bipole"` / `"gpw"`. The internal Γ-only EWALD drivers remain for dilute periodic systems and fail closed on the image-overlap regime. | Retired |
| `"gpw"` | Gaussian-on-grid collocation + FFT-Poisson Hartree-J on a smooth real-space grid. The dispatcher runs a full SCF for Gamma RHF / maintained-preview ROHF / RKS / UHF / UKS and multi-k RKS, with forces, FD Hessian, +U for RHF/RKS/UHF/UKS at Gamma and RKS at multi-k, smearing, and D3(BJ). Multi-k open-shell +U is not wired. ROHF is 3D Gamma-only with integer occupations and fails closed for gradients, smearing, and the other unsupported envelopes. The standalone `run_periodic_rohf_gpw` entry point remains available. Full-grid external XC has a narrower contract: zero-temperature 3D RKS at Gamma or on a complete unreduced multi-k mesh. It bypasses pointwise FFT XC but retains FFT-Poisson Hartree-J; positive smearing, ECP-bearing jobs, and symmetry-reduced representative meshes fail closed. Emits `GAPWExperimentalWarning`. | Production (opt-in warning); ROHF maintained preview; external XC experimental |
| `"gapw"` | GPW plus per-atom Gaussian augmentation on a fine radial grid for all-electron accuracy. The standard drivers cover Gamma RHF / RKS / UHF / UKS and multi-k RKS, with +U for RHF/RKS/UHF/UKS at Gamma and RKS at multi-k, smearing, and D3(BJ). Multi-k open-shell +U is not wired. At single Gamma, the method-aware `one_centre="auto"` policy selects the fit-free analytic-ERI augmentation for RHF/UHF only when the caller declares `molecular_limit=True` (`gapw_molecular_limit=True` on `run_periodic_job`); otherwise HF fails closed instead of guessing from geometry or falling back to the known-wrong block functional. RKS/UKS retain block. Analytic HF is validated only for vacuum-padded molecular-limit cells, not compact crystals or multi-k GAPW; use GDF or BIPOLE there. The high-level GAPW HF route rejects geometry/cell optimization and Hessians; `VibeqcGAPW` supplies fixed-cell forces, using full-SCF finite differences in analytic mode. HF phonons require explicit block mode, while DFT phonons default to block. Full-grid external XC is narrower still: zero-temperature 3D Gamma-only RKS, with the split smooth/hard/soft XC construction bypassed in favor of one complete AO-density evaluation. Positive smearing, multi-k, open-shell, and ECP-bearing external-XC jobs fail closed. Emits `GAPWExperimentalWarning`. | Production (opt-in warning; analytic HF molecular-limit only); external XC experimental |
| `"rijcosx"` | RI-J plus seminumerical chain-of-spheres exchange K (Neese 2009) on a periodic Becke grid. Gamma RHF uses the dedicated periodic RIJCOSX driver. True multi-k RHF / RKS / UHF / UKS route through the native GDF loop with `k_exchange="cosx"`; RKS/UKS hybrids exercise COSX exchange, while pure functionals have zero exact-exchange weight and therefore validate the RI-J plumbing. One-cell RKS/UHF/UKS requests fail closed rather than being mislabeled as RIJCOSX. Full-grid external XC is experimental and zero-temperature only on an unreduced complete multi-k mesh. | Implemented, experimental periodic COSX |
| `"rsgdf"` | **Top-level route not implemented** (enum-reserved): range-separated GDF as a standalone `jk_method` (Ye and Berkelbach, 2021). Distinct from the *internal* `gdf_method="rsgdf"` backend, which is the production default the GDF route uses for tight ionic cells (see § 3.2.1). | Roadmap |
| `"cfmm"` | Continuous fast multipole. Enum-reserved; not implemented. | Roadmap |

Two periodic method families also expose dedicated entry points in addition
to the public runner where noted:

* **RSGAPW (range-separated GAPW)** for range-separated hybrids
  (HSE06, ωB97X, CAM-B3LYP): `run_periodic_rhf_rsgapw` and
  `run_periodic_rks_rsgapw` in `periodic_gapw_range_sep.py`. It splits the
  Coulomb operator with the error function and carries the long-range
  piece on the plane-wave grid. See § 4.7.
* **Periodic ROHF / ROKS** (restricted open-shell HF and KS):
  `run_rohf_periodic_gamma_ewald3d` (Gamma) and
  `run_rohf_periodic_multi_k_ewald3d` (multi-k) in
  `periodic_rohf_ewald.py` / `periodic_rohf_multi_k_ewald.py`,
  `run_roks_periodic_multi_k_ewald3d` (Gamma + multi-k) in
  `periodic_roks_multi_k_ewald.py`,
  `run_krohf_periodic_gdf` (Gamma + multi-k, native GDF) in
  `periodic_rohf_gdf.py`, plus the
  standalone 3D Gamma GPW routes `run_periodic_rohf_gpw` /
  `run_periodic_roks_gpw` in
  `periodic_gapw_open_shell.py`. See § 3.8.

The method-aware one-centre promotion applies only to the standard Gamma
`run_periodic_rhf_gapw` / `run_periodic_uhf_gapw` drivers and their public
`run_periodic_job(jk_method="gapw")` dispatch. The dedicated RSGAPW and OT
research entry points use separate experimental builders and do not inherit
that default or its molecular-limit validation claim.

**Screened hybrids (HSE06) per route.** Every periodic driver resolves
its functional through the shared CAM assembly
(`vibeqc.periodic_screened_exchange`):
`K_HF = c_full·K_full + c_sr·K_erfc(ω_screen)`. Global hybrids (PBE0)
keep the full-range fraction unchanged. Screened hybrids (HSE06,
`c_full = 0`) build the genuine erfc short-range exchange (a pure
direct lattice sum with no Madelung/exxdiv seam and no bipolar
far-field, the CRYSTAL treatment) on the routes that have that
kernel: **BIPOLE**, the **Ewald drivers** (Γ + multi-k, RKS + UKS),
and the **slab route** (`slab_ewald_2d` / AUTO on dim=2). Routes whose
exact exchange is full-range only (**GDF** Γ + multi-k, **GPW/GAPW**,
**COSX**) fail closed on any range-separated functional instead of
silently running HSE06 as PBE0 (the pre-2026-07-12 behaviour, on every
route). Long-range-corrected RSH (ωB97X, CAM-B3LYP, LC-ωPBE) fail
closed everywhere periodically: their `K_full` arm needs an exxdiv
treatment that is not validated. The experimental RSGAPW entry points
above are unchanged by this and remain the GAPW-side option.

**The full-range (`c_full`) arm on a multi-k mesh uses the corrected
Ewald split.** On a finite k mesh the SCF density is
Born-von-Kármán-torus periodic, so `D(g)` does not decay with image
distance and a bare `1/r` image sum does not converge: it drifts
without limit as the lattice cutoff grows. The multi-k EWALD_3D drivers
(RHF, RKS, UKS) therefore build that arm as `erfc(α r)/r` in real space
plus a reciprocal channel per momentum transfer `q = k − k'` plus the
probe-charge `q + G = 0` term: the `exxdiv='ewald'` convention, shared
with BIPOLE and implemented once in
`vibeqc.periodic_corrected_exchange`. Consequences worth knowing:

* A hybrid or HF multi-k job no longer matches the **Γ-only** drivers
  bit-for-bit. Those keep their own truncated gauge, which is legitimate
  at Γ where the molecular-limit density does decay. Pure functionals
  build no full-range arm and are unaffected, as are screened hybrids
  (`c_full = 0`).
* Multi-k **RHF at a `(1,1,1)` mesh** keeps its molecular-limit branch
  by default (it zeroes every image block first, so its sum has one term)
  and stays in bit-parity with the Γ driver. Pass
  `exchange_exxdiv='ewald'` to `run_rhf_periodic_multi_k_ewald3d` to put a
  single-Γ mesh on the corrected split instead; `exchange_exxdiv='none'`
  names the molecular-limit default and is refused on any real mesh, where
  that sum does not converge. **The two are different finite-size
  conventions with the same infinite-k limit** (Carrier, Rohra & Görling,
  Phys. Rev. B 75, 205126 (2007), Sec. IV), 21.7 mHa apart at Γ on
  H₂/STO-3G in an 8-bohr cube. A Γ-supercell energy therefore band-folds
  onto a multi-k energy only when both sides name the same gauge; a
  k-convergence sequence taken with the defaults spans two of them, so
  fix `exchange_exxdiv` before comparing meshes. See
  `tests/test_periodic_exchange_gauge_band_folding.py`.
* A **symmetry-reduced (IBZ) Monkhorst-Pack mesh works, wedge-native**:
  the SCF diagonalises only at the irreducible points and unfolds the
  density to the full Brillouin zone per iteration with the Pisani/Dovesi
  star transport (`D(Vk) = W D(k) W^H` plus time reversal). This covers
  3D full-range and screened hybrids and 2D slab screened hybrids, including
  oblique lattices. The exchange quadrature is exact. K-dependent
  diagonalisation and Bloch work shrink with the wedge; 3D exchange can also
  use orbit-reduced real-space builds. Slabs retain the raw finite-cutoff
  exchange operator used by their full mesh, so their dominant real-space
  exchange build does not shrink by the star multiplicity; only the density
  is reconstructed.
  Caveat: at loose lattice cutoffs the truncated operators themselves
  break the point symmetry, so wedge and full-mesh answers can still
  differ by the truncation asymmetry; both coincide once the sums are
  fold-converged (the default `auto_optimize_truncation` regime). An
  **explicit k list** carries no star metadata: a full-range arm fails
  closed, while a screened-only arm treats its supplied weights as the
  user's quadrature and makes no symmetry-unfolding claim.
* **2D slab and `dim != 3`** cells cannot use the corrected full-range
  split (the reciprocal q-channel cache is 3D-only). Full-range behavior is
  route-specific: RKS fails closed, legacy RHF/UKS paths warn and remain
  provisional, and long-range-corrected functionals are rejected by the
  periodic exchange resolver. HSE-type screened hybrids have `c_full = 0`,
  need no reciprocal q channel, and are supported by the direct erfc
  exchange route, including the 2D IBZ unfolding above.

The rest of this page is the side-by-side treatment of the three
families that carry most day-to-day work: **BIPOLE**, **GDF**, and the
**GPW / GAPW** plane-wave route, with RIJCOSX and RSGAPW noted alongside.

> **Validating a periodic route?** Each JK route is its own implementation
> (never an external QC program at runtime). GDF and GPW/GAPW have
> same-family out-of-process references in PySCF and CP2K/GPAW. CRYSTAL is a
> component and convergence reference for BIPOLE, but the supported vibe-qc
> exact Ewald-J route does not execute CRYSTAL's quartet replacement. The
> [periodic JK routes & parity policy](../periodic_jk_routes.md) page is
> the map of which reference belongs to which route (and why cross-family
> comparison is a trap). It is the periodic backbone reference for the
> method paper.

## 2. BIPOLE: an Ewald J-split

### 2.1 Theory

BIPOLE is vibe-qc's own direct-space Gaussian periodic
SCF route. CRYSTAL famously splits the Fock build into two halves
that live in the **same** electrostatic gauge:

* The one-electron Coulomb $V_{ne}$ and the nuclear-nuclear energy
  $E_{nn}$ are evaluated with 3D Ewald. The same Ewald parameter
  $\omega$ is shared by both (a single shared Ewald state).
  Direct truncation diverges on charged-nucleus
  crystals because the point-charge / Gaussian-pair interaction
  on the nucleus side does not decay.
* The two-electron Fock matrix $F^{2e} = J - \tfrac{1}{2}K$ is
  assembled in CRYSTAL by direct BIPOLE screening and multipole
  far-pair replacement. The supported vibe-qc route instead uses an exact
  Ewald-3D J-split in the same electrostatic gauge,

  $$
  J(\omega) = J^{SR}(\omega) + J^{LR}(\omega) + V_{bg}\cdot S,
  \qquad
  V_{bg} = -\frac{\pi N_e}{\omega^2 V},
  $$

  where $\omega$ is the *same* Ewald parameter used by $V_{ne}$
  and $E_{nn}$. $J^{SR}$ uses erfc-screened real-space ERIs;
  $J^{LR}$ is built from a small reciprocal-space sum over
  AO-pair Fourier transforms; the uniform background restores
  charge neutrality. Exchange $K$ is assembled fully in direct
  space via shell-pair Schwarz screening (the same machinery that
  the molecular four-index path uses).

The periodic quartet expansion is given in Chapter II.4c of
**Pisani, Dovesi, and Roetti (1988)**, where Eq. II.4.10 retains the
common lattice-translation sum. **Saunders et al. (1992)** develops the
related Gaussian-density electrostatic potential, shell-penetration, and
spheropole treatment; it does not derive vibe-qc's quartet dispatcher.
**Dovesi et al. (2014)** and **Erba et al. (2023)** provide modern
CRYSTAL context for the EXT EL-POLE / EXT EL-SPHEROPOLE conventions.
The BIPOLE name in vibe-qc explicitly nods to that lineage.

### 2.2 What ships today

The BIPOLE driver is split across four method-flavour entry points
in [`pbc_bipole.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/pbc_bipole.py)
and friends:

```python
from vibeqc.pbc_bipole     import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks
```

All four support:

* Full real-space one-electron pipeline at Ewald gauge: $S(g)$,
  $T(g)$, $V_{ne}(g)$ at `opts.lattice_opts.cutoff_bohr`,
  Bloch-summed to per-k $S(k)$ and $H_{core}(k)$, canonical
  orthogonalisation $X(k)$. The historical enumeration keeps the
  translations with $|g| \le$ `cutoff_bohr`; the physically right
  criterion is the pair separation $|O_\mu - O_\nu - g|$ (Sharma and
  Beylkin, *JCTC* 17, 3916 (2021), Eqs. 17-18), and the two differ on any
  cell with an intra-cell offset: LiH rocksalt / def2-SVP moves by 0.72 in
  an overlap element at the shipped cutoff when the same crystal is
  described with an atom one lattice vector away. Since GitLab #429
  stage 2, `opts.lattice_opts.pair_complete_1e = True` enumerates the
  pair-complete term set for $S$, $T$, the direct $V_{ne}$ and their
  gradient partners (invariant to round-off; the cell list grows, MgO at
  the default from 135 to 369 cells, and the padded cells that no pair
  reaches stay exactly zero). The option remains off by default pending
  cutoff convergence and performance review.
  `vibeqc.pair_complete_lattice_cells` returns the one-electron enclosure.
  The two-electron enclosure also includes the interaction reach between
  physical product midpoints: J restricts the AB/CD products, while K
  restricts AC/BD. Exchange output and density cells can extend beyond
  the one-electron support. Consumers align these lists by integer cell
  key and extend one-electron matrices with zeros where appropriate.
  `eri_interaction_cutoff_bohr` sets the midpoint reach; zero uses
  `cutoff_bohr`, and the BIPOLE SR extent overrides that reach when supplied.
  The **Ewald-split nuclear family** (analytical erfc, grid and FT
  $V_{ne}$, SAP and their gradient partners) uses the same AO-pair filter
  in each complementary term. Nuclear source images are centered on
  each physical AO-product midpoint. The reciprocal cache and derivatives
  preserve the same pair support.
  Physical mode uses the full Fock build. The legacy symmetry projector,
  periodic COSX and diagnostic grid Hartree backend refuse this option
  because they do not implement its exchange/domain contract. Image
  relabelling preserves a fixed physical domain; convergence of the
  periodic operator additionally requires converging the pair cutoff,
  interaction reach and reciprocal resolution.
* The shared-gauge Ewald J-split two-electron build by default
  on 3D systems (`use_ewald_j_split=None` -> `True`): short-range
  $J$ from direct erfc-screened ERIs, long-range $J$ from
  reciprocal AO-pair Fourier transforms, matching electron
  background potential, full direct-space exchange.
* SCF inner loop with Bloch-sum -> direct real-space
  energy evaluation ($\sum_g D(g) M(g)$) -> optional multi-k
  Pulay DIIS -> optional level-shift -> diagonalisation ->
  optional MOM occupied-subspace reorder -> density rebuild.
* Multi-k acceleration in the dense-k regime. Symmetry-reduced
  Monkhorst-Pack inputs are accepted and internally expanded back
  to the full mesh until true IBZ orbit expansion lands.
* Parameter-free `bz_integration="gilat"` occupations on the public RKS,
  UHF, and UKS routes. UHF/UKS integrate the fixed alpha and beta populations
  independently with single-spin degeneracy and zero entropy. This is a
  supported self-consistent path for gapped systems; metallic calculations
  use smearing for SCF and Gilat post-SCF because the sharp T = 0 occupation
  map is discontinuous at band crossings. RHF/ROHF/ROKS remain gated here.
* Periodic DFT quadrature uses an atom-image neighborhood centered on each
  integration point. `becke_image_radius_bohr` sets its minimum physical
  radius; it expands to twice the nearest-image distance where needed to
  keep vacuum regions covered. It does not truncate the atomic radial grid.
  Converge this radius together with radial and angular grid resolution,
  especially for diffuse densities and large vacuum regions. A radius of
  zero explicitly requests molecular partitioning. Moving an atom to an
  equivalent image preserves the physical partition; raw atomic weights
  and atom ownership retain the same layout.
* Production finite-difference atomic forces for all four flavours and
  fixed-cell atomic relaxation. Variable-cell and coupled atom/cell
  optimization fail closed; `compute_stress_tensor` is only a force-virial
  diagnostic. See
  [`bipole.md`](bipole.md) for the gradient and optimisation
  surface.

Dense-k STO-3G sign-off vs CRYSTAL14 at `cutoff = 14 bohr`:

| Demo | vibe-qc $E_{\text{total}}$ (Ha / cell) | $\Delta$ vs CRYSTAL14 |
|---|---:|---:|
| MgO SHRINK 8 | $-271.2177748509$ | $+0.369\;\text{mHa}$ |
| Diamond SHRINK 8 | $-74.8771393842$ | $-0.145\;\text{mHa}$ |
| Silicon SHRINK 8 | $-571.3214715798$ | $-0.659\;\text{mHa}$ |

These recorded sign-off numbers predate the repository split. The old
`tests/demos/` path is no longer available in this tree; current diagnostics
are in [`examples/regression/bipole_parity/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/bipole_parity).
Do not treat the historical figures as a fresh validation of the current build.

### 2.3 Advantages

* **Multi-k production today, including multi-k hybrids.** Multi-k SCF on
  UHF / UKS as well as RHF / RKS, and the route that carries exact-exchange
  K at $\mathbf{k}\neq 0$ for hybrid functionals. (Open-shell GDF also runs
  multi-k UHF / UKS via `run_kuhf_periodic_gdf` / `run_kuks_periodic_gdf`;
  In 3D, BIPOLE remains the AUTO open-shell default and the multi-k hybrid
  route; 1D AUTO uses GDF and 2D AUTO uses the slab route.)
* **Global and screened hybrid DFT work on the supported envelope.** Global
  hybrids and the validated HSE screened-exchange branch share BIPOLE's
  direct-K machinery. Long-range-corrected range-separated hybrids fail
  closed, as do VV10/nonlocal-correlation and double-hybrid functionals whose
  additional energy terms are absent from the BIPOLE driver.
* **Production FD forces and fixed-cell optimisation.** Atom optimisation
  runs in fractional coordinates. Variable-cell optimisation fails closed
  pending a certified stress and coupled convergence implementation.
* **External convergence reference.** Matched basis and k-mesh calculations
  provide useful component and convergence checks, but the supported exact
  Ewald-J route is not the same truncation algorithm as a bipolar-expansion
  calculation and no universal per-mHa cross-code guarantee is claimed.
* **Diagnostic decomposition.** Each iteration logs
  $E_{kinetic}$, $E_{nuclear\_attract}$,
  $E_{J^{SR}}$, $E_{J^{LR}}$, $E_{exchange}$, $E_{nn}$
  separately so a parity mismatch can be localised to a single
  Coulomb contribution.

### 2.4 Disadvantages

* **3D only through the public BIPOLE route.** A `dim=1` wire routes to
  GDF under AUTO, and a `dim=2` slab routes to `SLAB_EWALD_2D`. An explicit
  public `jk_method="bipole"` request rejects both dimensions before setup.
  The low-dimensional direct-only BIPOLE branch is a direct-API diagnostic.
* **Cost scales with the real-space cutoff.** Direct $K$ at large
  cutoff is the dominant per-iteration cost. Corrected-gauge direct
  RHF/RKS/UHF/UKS runs therefore stop before the first Fock build when the
  cheap overlap-fold preflight is in the critically truncated
  ``drift > 1e-2`` regime; increase ``bipole_cutoff_bohr`` until the drift is
  below ``1e-4`` for quantitative work. If that converged cutoff is too
  expensive, use a supported GDF or GPW route. The experimental
  multipole-far-pair branch is a retired research artifact, not a supported
  shortcut for this cost.
* **No true IBZ acceleration yet.** Symmetry-reduced Monkhorst-
  Pack inputs are accepted, but they are internally expanded to
  the full mesh before the Ewald-J build, so the cost is the same
  as a full-mesh run.

### 2.5 When to use BIPOLE

* You need an open-shell (UHF / UKS) periodic SCF.
* You need production FD forces or fixed-cell atom optimisation.
* You are doing CRYSTAL parity work.
* You need multi-k DFT today, including hybrid DFT, on dense
  meshes (2x2x2 through ~8x8x8) and modest basis sizes
  (STO-3G through pob-TZVP / def2-SVP class).

### 2.6 Quick start

```python
import numpy as np
import vibeqc as vq
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

ANG2BOHR = 1.0 / 0.529177210903
a = 4.21 * ANG2BOHR
lattice = (a / 2.0) * np.array(
    [[0.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0]]
)
sysp = vq.PeriodicSystem(
    3, lattice,
    [vq.Atom(12, [0.0, 0.0, 0.0]),
     vq.Atom(8,  [a / 2.0, a / 2.0, a / 2.0])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

kmesh = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 14.0
opts.lattice_opts.nuclear_cutoff_bohr = 14.0
opts.initial_guess = vq.InitialGuess.SAD
opts.use_diis = True
opts.max_iter = 30

result = run_pbc_bipole_rhf(
    sysp, basis, kmesh, opts,
    use_ewald_j_split=True,
    ewald_precision=1e-8,
)
print(f"E_total = {result.energy:+.6f} Ha")
```

End-to-end MgO RKS / PBE input lives at
[`examples/periodic/input-bipole-mgo-rks-pbe.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-bipole-mgo-rks-pbe.py),
open-shell Li UHF at
[`examples/periodic/input-bipole-li-uhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-bipole-li-uhf.py),
and the unified-runner workflow at
[`examples/periodic/input-bipole-workflow-mgo.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-bipole-workflow-mgo.py).

### Periodic atomic grids

For routes using an atom-centered periodic grid, each quadrature point
normalizes its Becke or Stratmann weight over the physical atom images
within `max(becke_image_radius_bohr, 2*d_nearest)`. Here `d_nearest` is the
distance from that point to its nearest atom image. This makes the image
neighborhood independent of which lattice image labels a home atom, and
retains coverage in vacuum for diffuse densities. The raw atomic points,
weights and ownership keep the selected grid profile.

The default minimum radius remains 10 bohr. It is a truncation control,
not a total-energy tolerance: converge it together with radial and angular
resolution. For forces, compare more than one displacement step as well;
a finite image neighborhood can introduce small boundary steps. A zero
radius explicitly requests the molecular partition. These grid checks do
not qualify an otherwise unsupported periodic method or gradient route.

### 2.7 Citations

* C. Pisani, R. Dovesi, C. Roetti, *Hartree-Fock Ab Initio Treatment
  of Crystalline Systems*, Lecture Notes in Chemistry 48, Springer,
  1988. [DOI 10.1007/978-3-642-93385-1](https://doi.org/10.1007/978-3-642-93385-1).
  Chapter II.4c gives the periodic quartet bipolar expansion.
* V. R. Saunders, C. Freyria-Fava, R. Dovesi, L. Salasco, C. Roetti,
  "On the electrostatic potential in crystalline systems where the
  charge density is expanded in Gaussian functions," *Molecular Physics*
  **77**, 629 (1992).
  [DOI 10.1080/00268979200102671](https://doi.org/10.1080/00268979200102671).
  Source for the related shell-penetration and spheropole construction.
* R. Dovesi, V. R. Saunders, C. Roetti, R. Orlando,
  C. M. Zicovich-Wilson, F. Pascale, B. Civalleri, K. Doll,
  N. M. Harrison, I. J. Bush, P. D'Arco, M. Llunell, M. Causa,
  Y. Noel, "CRYSTAL14: A program for the *ab initio* investigation
  of crystalline solids", *Int. J. Quantum Chem.* **114**, 1287
  (2014). [DOI 10.1002/qua.24658](https://doi.org/10.1002/qua.24658).
* A. Erba et al., "CRYSTAL23: a program for computational solid
  state physics and chemistry", *J. Chem. Theory Comput.* **19**,
  6891 (2023).
  [DOI 10.1021/acs.jctc.2c00958](https://doi.org/10.1021/acs.jctc.2c00958).

## 3. GDF: native Gaussian density fitting

### 3.1 Theory

Gaussian density fitting (Whitten 1973; Dunlap 1979; Eichkorn et
al. 1995; Vahtras, Almlof, Feyereisen 1993) projects the AO pair
density onto a finite auxiliary basis $\{\chi_P\}$:

$$
\rho_{\mu\nu}(\mathbf{r})
  = \phi_\mu(\mathbf{r})\phi_\nu(\mathbf{r})
  \approx \sum_P d^{\mu\nu}_P\,\chi_P(\mathbf{r}).
$$

The fit coefficients minimise the Coulomb error and require only
the two-centre auxiliary metric $M_{PQ} = (P|Q)$ and the
three-centre integral $(\mu\nu|P)$. The factorisation

$$
L_{P,\mu\nu} = \sum_Q (\mu\nu|Q) (M^{-1/2})_{QP}
$$

reduces the four-centre ERI to a single tensor, and the closed-shell
Fock pieces collapse to

$$
J_{\mu\nu} = \sum_L L_{L,\mu\nu}\;\rho_L,
\qquad
\rho_L = \sum_{\lambda\sigma} L_{L,\lambda\sigma}\,D_{\lambda\sigma},
$$

$$
K_{\mu\nu} = \sum_L \left(L_L\,D\,L_L^\mathrm{T}\right)_{\mu\nu}.
$$

This is the standard molecular RIJK kernel that vibe-qc's
`DFJKBuilder` implements; the *periodic* version requires three
extra ingredients on top:

1. **Periodic auxiliary metric** $M^{per}_{PQ} = \sum_T (P_0 | Q_T)$.
   Naively, the 3D lattice sum of $1/r$ across a charge-neutral
   set of aux primitives is conditionally convergent. The fix is
   PySCF-style **modrho** charge compensation: each contracted aux
   shell is renormalised so its monopole equals
   $\sqrt{1/(4\pi)}$, and the AO-pair side is charge-balanced by
   smooth compensating Gaussians. The compensated integral
   converges in real space; the missing
   $(\text{aux}|\text{comp})$ piece is added back analytically in
   reciprocal space.
2. **Periodic three-centre tensor** $T^{per}_{\mu\nu,P} = \sum_T
   (\mu_0 \nu_0 | P_T)$. Implemented as a libint lattice sum with
   Schwarz screening, mirroring the molecular `compute_3c_eri`
   layout. For each k-point the Bloch-summed
   $T^{per}_{\mu\nu,P}(\mathbf{k})$ is the same contraction.
3. **Cholesky factor** $L^{per}_{P,\mu\nu} = \sum_Q
   (\mu_0\nu_0|Q)\,(M^{per})^{-1/2}_{QP}$, computed once per SCF.

vibe-qc's native GDF stack lives in
[`pbc_gdf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/pbc_gdf.py),
[`periodic_rhf_gdf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic_rhf_gdf.py),
[`periodic_gdf_blocks.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic_gdf_blocks.py),
and [`periodic_k_gdf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic_k_gdf.py).
The full design walkthrough, including the modrho compensation
empirical record, is in
[`design_native_gdf.md`](../design_native_gdf.md).

### 3.2 What ships today

Native GDF in vibe-qc covers:

* **Gamma-only RHF, RKS, and hybrids in 1D / 3D** (a `dim=2` slab raises) via
  `run_rhf_periodic_gamma_gdf` (and the runner-level
  `run_periodic_job(..., jk_method="gdf")`).
* **Multi-k RHF / RKS in 1D / 3D** (a `dim=2` slab raises) via
  `run_krhf_periodic_gdf(..., kmesh, ...)` and
  `run_krks_periodic_gdf(..., kmesh, ...)`. A `(1, 1, 1)` mesh
  delegates to the Gamma-GDF path; any other Monkhorst-Pack mesh
  runs the full multi-k loop with k-dependent 3c / 2c blocks,
  Bloch-phase assembly, and multi-k DIIS. The multi-k Ewald-gauge
  fix brings a multi-k mesh to within 0.000 mHa of the Gamma
  energy in the molecular limit. On the 3D true-multi-k RKS route
  with a pure functional and `use_compcell=False`, the analytic-FT
  Ewald Hartree J cache is retained only when it fits the bounded
  dense-cache target. Larger cases contract J directly from
  $D(\mathbf{k})$ in bounded reciprocal-vector and output-cell
  batches, then Bloch-fold it; they do not retain the former
  `(n_cells, n_AO, n_AO, n_G)` complex tensor. Small jobs keep their
  pre-SCF cache reuse. The `G = 0` omission, Ewald gauge,
  `VIBEQC_J_EWALD3D_KE` cutoff, and SCF defaults are unchanged. Gamma,
  HF/hybrid cached-Lpq, `use_compcell=True`, lower-dimensional, and
  diagnostic grid-backend routes do not use this new contraction.
* **Translation-resolved DF block accessors** via
  `compute_2c_eri_lattice_blocks` and
  `compute_3c_eri_lattice_blocks`, plus the Bloch-sum helpers
  `bloch_sum_2c_eri_blocks` / `bloch_sum_3c_eri_blocks` and the
  fit helper `build_lpq_bloch_native`. Summing blocks reproduces
  the Gamma metric exactly.
* **Native gauge sharing.** $V_{ne}$ and $E_{nn}$ are forced to
  Ewald-3D on 3D systems regardless of the user's J/K route choice
  so the total energy is gauge-consistent (the
  `_gauge_lat_opts_for_v_ne_and_e_nuc` helper in
  `periodic_rhf_gdf.py`).
* **Fermi-Dirac smearing** at the Gamma point and on multi-k
  meshes via `opts.smearing_temperature` ($k_B T$ in Hartree).

The driver picks one of two backends per call and records it on
the `PeriodicRHFGDFResult.backend` field:

* `"native-gamma-gdf"`: true density fitting via `Lpq`. Used for
  1D wires and 3D molecular-limit cells whose GDF cutoff includes only
  the home cell.
* `"ewald-jk-fallback"`: Ewald-3D $J$ + real-space $K$. Used for
  3D systems with image cells inside the GDF cutoff. See
  `vibeqc.pbc_gdf` for the compensated-cell GDF path that will
  eventually subsume this branch.

#### 3.2.1 RSGDF all-FT-Bloch path, chemical-accuracy milestone (v0.9)

For tight ionic 3D crystals (LiH, MgO and similar), the
compensated-cell path in `vibeqc.pbc_gdf` now runs an
**all-FT-Bloch** route in its `gdf_method="rsgdf"` mode: the
3c tensor is assembled on a dense FFT mesh from Bloch-summed
AO pair-FTs, rather than the older sparse-mesh SR + LR split
that conflated single-orbital lattice sums (libint short
range) with molecular pair-FT long range. The unified all-FT
mesh handles both the vacuum-box limit and the tight ionic
limit with the same algorithm.

What landed across commit chain `018784c0..d7f4b3bd` brings
the PBC GDF route to within chemical accuracy on LiH:

| System | Pre-session | Post-session |
|---|---:|---:|
| H₂ / 12-bohr / def2-svp-jk | 0.6 mHa | 0.04 µHa @ ke=400 |
| LiH primitive FCC / sto-3g / def2-svp-jk | +11 Ha (broken) | −2.3 mHa @ ke=200 |

The three contributing fixes:

1. **All-FT-Bloch path** (`018784c0`), dense FFT mesh +
   Bloch-summed pair-FT for the 3c side, replacing the
   sparse SR+LR split inside the `rsgdf` route.
2. **modrho-direct fix** (`c949d748`), `build_lpq_native_fft`
   now takes the modrho aux basis directly, so the per-L
   `1/√(4π/(2L+1))` factor baked into the modrho coefficients
   is not double-applied. Pre-fix Lpq was 330× too large on
   LiH.
3. **Proper Ewald-Madelung** (`d7f4b3bd`), replaced the cubic
   Wigner shortcut `α_M / L` (exact only for cubic Bravais
   cells, 1.77 % off for primitive FCC) with a proper Ewald
   self-energy sum over real-space images + reciprocal G-vectors.
   Bit-exact to PySCF's `pbc.tools.madelung` on cubic, FCC, and
   intermediate cells. See [`ewald.md`](ewald.md) for the user-
   facing surface and `python/vibeqc/madelung.py` for the
   implementation.

The remaining −2.3 mHa LiH residue is **not a GDF bug**,
it is located in the V_ne (nuclear-electron) one-electron
Hcore: `compute_nuclear_lattice_ewald` builds V_long via grid
quadrature whose Becke partitioning breaks the (1,1,1)
3-fold rotational symmetry of the FCC primitive cell
(Li 2p_x ≠ 2p_y ≠ 2p_z by ~1.9 mHa per AO). A prototype
analytic FT replacement preserves the symmetry to machine
precision; productionising it is the periodic-1e chat's job,
not GDF. See
[Archived `HANDOVER_GDF_V0_11_2026_05_29.md`](https://vibe-qc.com/docs/)
§ 2026-05-29 for the diagnostic chain.

#### 3.2.2 Cauchy-Schwarz screening of the three-centre fit

The multi-k drivers accept `fit_screen_threshold` (default `0.0` = off,
exact) on the `gdf_method="rsgdf"` path:

```python
r = vq.run_krhf_periodic_gdf(
    system, basis, (2, 2, 2),
    fit_screen_threshold=1e-10,
)
```

AO pairs $\mu\nu$ whose Schwarz bound
$\max_P \sqrt{(P|P)}\,\sqrt{(\mu\nu|\mu\nu)}$ on the fit mesh stays
below the threshold are dropped from every per-$(k_i,k_j)$ three-centre
fit tensor (Neese, Wennmohs, Hansen and Becker, *Chem. Phys.* **356**,
98 (2009), § 3.1: the RI analogue of Häser-Ahlrichs direct-SCF
screening). The pair-norm factor is an analytic upper bound, so the
screen never underestimates a pair; `1e-10`-class thresholds reproduce
the unscreened energy to well below SCF accuracy. The kept/dropped
pair counts are reported in the SCF log (no silent truncation). The
CCM RI routes (`run_ccm_rhf_gdf` / `run_ccm_rks_gdf`) forward driver
kwargs, so they inherit the option unchanged. Screened fits compose
with `compute_gradient=True` (Γ and multi-k, since 2026-07-30): the
analytic gradient differentiates the screened objective at the SCF's
fixed pair mask. Non-`rsgdf` fit methods
raise `NotImplementedError` rather than silently ignoring the
threshold.

The Γ-point drivers (`run_pbc_gdf_rhf` / `run_pbc_gdf_uhf` /
`run_pbc_gdf_uks`, and the Γ `(1,1,1)` fast path of the multi-k
drivers) accept the same option. There it additionally selects the
**memory-lean build**: the fit sweeps its reciprocal mesh in G-chunks
and the Ewald-3D $V_{ne}$ Fourier transform streams its own chunked
AO-pair FT, so the dense $(n_{\mathrm{AO}}, n_{\mathrm{AO}}, n_G)$
pair-FT tensor (the dominant memory of a Γ-point GDF build on
production-basis cells) is never materialised (at the cost of running
the pair FT once per consumer instead of once shared). Γ k-meshes that
fall back to the legacy molecular-limit GDF driver (open options such
as `level_shift` / `fock_mixing` / smearing) raise
`NotImplementedError` rather than silently dropping the threshold.

Since 2026-07-16 only an **explicit** `fock_mixing` / `fmixing_percent`
/ `level_shift` request sends a default-Γ closed-shell RHF job to that
legacy fallback. The `convergence="auto"` profile knobs (e.g. the
ionic-insulator FMIXING 30% that MgO-class cells resolve) are
capability-filtered to zero instead: the run stays on the
PySCF-parity `run_pbc_gdf_rhf` driver, whose DIIS/accelerator stack
and auto-sized high-|G| tail carry the convergence, because the
legacy fallback's dense-core absolute energies are parity-held
(G-GDF-001 in `handovers/HANDOVER_GATED_ITEMS.md`). The strategy
block in the `.out` states the filtered knob and the reason.

#### 3.2.3 Space-group reduction of the exchange build

Exact exchange is the one $n_k^2$ term in a multi-k GDF SCF: $K(k_i)$
needs a sum over the **whole** Brillouin zone for every bra $k_i$. The
space-group saving is therefore on the *bra* index alone: build
$K(k_i)$ only at the irreducible-wedge representatives and
symmetry-transport it across each star. Reducing the ket sum as well
is not the same operation and is wrong (measured $+1.389$ Ha on MgO).

Request it from the production entry point with `symmetry_reduce_k`:

```python
r = vq.run_periodic_job(
    system, basis,
    method="RHF",
    jk_method="gdf",
    kpoints=(4, 4, 4),
    symmetry="attach",          # the wedge needs a symmetry model
    symmetry_reduce_k=True,
    aux_basis="def2-svp-jk",
    output="nacl_reduced",
)
```

The `Lpq` cderi cache then holds $n_{\mathrm{IBZ}} \times n_k$ blocks
instead of $n_k^2$, and the memory preflight is charged at that
reduced count. Measured 2026-08-14 on NaCl primitive / def2-SVP with
`aux_basis="def2-svp-jk"` (33 AOs, 240 aux) at $(4,4,4)$,
$n_{\mathrm{IBZ}} = 8$ of 64: the `Lpq` term drops from 15.95 GB to
1.99 GB (8x) and the total preflight from 25.08 GB to 4.14 GB (the
AO-pair FT bundle and the per-$k$ buffers do not reduce). The `.out`
states the reduction explicitly
(`symmetry_reduce_k = exchange bras 64 -> 8 (irreducible wedge)`)
directly above the `[memory]` block, so the estimate can be accounted
for.

Everything else is untouched (diagonalisation at every $k$,
occupations, the energy expression, and the result shape), so the
reduction is **exact**, not an approximation, and bands / DOS / COOP /
QVF consumers see no change. Accuracy is limited only by the cell
list the transport rides on: on LiH FCC $(2,2,2)$ the $K$ transport
residual is $1.5\times10^{-3}$ at `cutoff_bohr = 15`,
$8.2\times10^{-6}$ at 20 and $1.1\times10^{-8}$ at 26, and the driver
warns below 20 bohr rather than degrading quietly.

Every precondition **fails closed** rather than silently running
full-BZ, because a performance flag that looks effective and is not
is worse than a refusal:

* `jk_method="gdf"`, `dim=3`, a true multi-k Monkhorst-Pack mesh, and
  `method` in RHF / RKS / UHF / UKS (spin-restricted open-shell GDF
  has no wedge surface);
* an attached symmetry model: pass `symmetry="attach"`, or attach
  validated operations to the `PeriodicSystem` yourself;
* a **symmorphic** space group. A glide or screw carries a fractional
  translation $\tau$, contributing an $e^{-i k\cdot\tau}$ factor the
  per-atom lattice-shift Bloch transport does not build, so the
  refusal names the group (rocksalt $Fm\bar{3}m$ works; diamond
  $Fd\bar{3}m$ is rejected up front);
* exact exchange, because on a pure functional there is no $n_k^2$ term to
  reduce;
* no geometry optimization: the wedge-native SCF converges the
  k-transported exchange, whose gradient is not the captured
  single-point objective.

The underlying driver keyword is `ibz_native=True` on
`run_krhf_periodic_gdf` / `run_krks_periodic_gdf` /
`run_kuhf_periodic_gdf` / `run_kuks_periodic_gdf`, for callers driving
the drivers directly.

### 3.3 Advantages

* **Best closed-shell scaling.** With aux size $M \approx 3N$,
  per-iteration cost is $\mathcal{O}(N^2 M) = \mathcal{O}(N^3)$,
  compared to $\mathcal{O}(N^4)$ for the direct-K piece in BIPOLE.
* **Production multi-k for closed-shell RHF / RKS** in 1D and 3D, plus a
  verified closed-shell energy + analytic-gradient route in 2D. Slab GDF uses a separate signed
  truncated metric and rigorous `V_ne(k)` / nuclear gauge; it is not the old
  bulk builder with a normal-axis vacuum cell. `jk_method="auto"` still picks
  the direct `SLAB_EWALD_2D` route for `dim=2`; request `"gdf"` explicitly.
* **Hybrid DFT included.** Exact-exchange coefficients are
  contracted out of the same `Lpq` factor that builds $J$.
* **Same kernel as molecular RIJK.** Auxiliary-basis selection,
  Cholesky robustness, and linear-dependence handling are all
  shared with the molecular `DFJKBuilder`.
* **Dimensionality clean in 1D.** For a wire there is no 3D Madelung
  problem to invent gauge for; the compensated lattice sum converges
  in real space and the route is genuinely single-shot.

### 3.4 Disadvantages

* **AUTO routing is dimensional.** In 3D, AUTO keeps BIPOLE as the UHF/UKS
  default and explicit `jk_method="gdf"` opts into native open-shell GDF. In
  1D, where the public BIPOLE route is unavailable, AUTO selects GDF for
  RHF/RKS/UHF/UKS. Native open-shell GDF ships at Gamma
  (`run_pbc_gdf_uhf` / `run_pbc_gdf_uks`, including per-spin Fermi-Dirac
  smearing) and multi-k (`run_kuhf_periodic_gdf` /
  `run_kuks_periodic_gdf`).
* **The analytic periodic GDF gradient has route-specific envelopes.**
  The original `compute_gdf_gradient` implementation covers 3D Gamma
  `gdf_method="compcell"` for
  closed-shell **RHF** and **RKS** (`run_pbc_gdf_rhf(functional=...,
  compute_gradient=True)`, milestone 2: general-$a_x$ J/K machinery plus
  the lattice-summed XC Pulay primitive on the SCF's own quadrature) and,
  since milestone 3b (2026-07-29), **UHF**
  (`run_pbc_gdf_uhf(compute_gradient=True)`), on **both** AFT settings:
  the 2c/3c AFT reciprocal-space corrections are differentiated
  analytically, so the production `apply_aft_correction=True` fit no
  longer needs to be disabled for forces. Its Ewald $V_{ne}$ derivative is
  fully analytic, it reuses the SCF's resolved 2c/3c cutoffs, and it
  differentiates the threshold-truncated fit eigenspace. The same
  milestone identified and fixed the long-standing flat
  $3.4\times10^{-3}$ Ha/bohr FD residual (a truncated-direct-sum nuclear
  gradient paired with the Ewald-gauge $E_{nn}$): the end-to-end H2/
  STO-3G FD regressions now hold every route (RHF, LDA/PBE/PBE0, and the
  UHF triplet) to $10^{-6}$ Ha/bohr (observed $\sim 10^{-8}$). **UKS**
  followed as milestone 4 (`run_pbc_gdf_uks(functional=...,
  compute_gradient=True)`): the per-spin assembly takes the functional's
  exact-exchange fraction and adds the spin-polarised XC Pulay;
  UKS(M=1) collapses onto the RKS gradient at machine precision and the
  triplet passes the same AFT-on FD gate. The **production rsgdf fit**
  gained the same full Gamma ladder as milestone 6
  (`gdf_method="rsgdf"`, `compute_gradient=True`, RHF/RKS/UHF/UKS):
  its reciprocal-space metric and 3c tensor carry atom positions only
  in `exp(-iG.R)` phases, so their weighted centre derivatives are
  exact, and the full-SCF FD sits at ~4e-9 even at the ke=200
  production default. Since the open-shell drivers default to rsgdf,
  `run_pbc_gdf_uhf/uks(compute_gradient=True)` needs no extra knobs.
  Tailed dense-core rsgdf fits are differentiated too: the long-standing
  dense-core analytic-vs-FD mismatch (~2.5e-5 Ha/bohr on MgO) was an
  e_nuc real-space truncation artefact in the *energy*, fixed 2026-07-29
  by switching the 3D GDF drivers to the converged Ewald nuclear term.
  **Multi-k GDF gradients landed 2026-07-30** (G-PBC-002 Item 4):
  `run_krhf_periodic_gdf` / `run_krks_periodic_gdf` /
  `run_kuhf_periodic_gdf` / `run_kuks_periodic_gdf` accept
  `compute_gradient=True` and return the analytic nuclear gradient on
  `result.gradient`, differentiating the shared-q rsgdf fit
  k-resolvedly (per-q metric/3c centre derivatives, DF-K over every
  momentum-transfer group, the exxdiv W-shift, and the XC response on the
  SCF's own inverse-Bloch folded density). The KS response combines the
  analytic fixed-grid AO/GGA derivative with a fixed-density central
  difference of the moving atom-centred quadrature points and Becke weights;
  this is an XC-only correction, not a 6N-SCF gradient. Full-SCF FD gates on H2
  (2,1,1) at h=2e-4: KRHF $1.1\times10^{-8}$, KUHF triplet
  $1.6\times10^{-8}$, KRKS LDA $4.1\times10^{-10}$ / PBE0
  $1.5\times10^{-8}$ Ha/bohr. A genuine-spin LiH triplet/PBE KUKS gate
  matches FD to about $7\times10^{-9}$ Ha/bohr and conserves net force,
  with the KUKS(M=1) collapse onto KRKS retained at machine precision.
  Supported multi-k envelope: 3D cells,
  `gdf_method="rsgdf"` on the cached-Lpq SCF (`use_compcell=True`;
  HF/hybrids promote automatically), Γ-centered uniform full-BZ
  meshes, gapped integer T = 0 Aufbau occupations, pure and global-hybrid
  functionals. A band-overlap T = 0 global-Aufbau ensemble is supported for
  energies but fails closed for analytic gradients until its sharp-Fermi
  response is validated. The ASE `backend="gdf"` calculator routes
  `kpts != (1,1,1)` through these drivers (rsgdf only; the tuple
  `kpts` is Γ-centered there, unlike the `ewald` backend's shifted
  Monkhorst-Pack on even meshes). **Schwarz-screened fits are
  differentiated too since 2026-07-30** (Γ and multi-k): the gradient
  cache re-derives the SCF's per-q pair mask from the same mask
  builder and differentiates the screened objective at that FIXED
  mask (masked pairs are hard zeros of the fit, so their derivative
  is exactly zero; the mask itself is a discrete quantity with no
  derivative). Screened full-SCF FD gates: Γ H2 $2.7\times10^{-9}$,
  KRHF (2,1,1) $1.2\times10^{-8}$ Ha/bohr. **Finite-temperature
  (Fermi-Dirac) smearing is differentiated too since 2026-07-30** (Γ
  open-shell + multi-k): the analytic gradient is $dA/dR$ of the
  driver's reported Mermin free energy $A = E - T S$: by Mermin's
  variational principle (Phys. Rev. 137, A1441 (1965)) the
  occupation- and $\mu$-response terms vanish at self-consistency, so
  the $T = 0$ assemblers run on the fractional-occupation density and
  energy-weighted density (the standard smeared-force statement,
  Marzari & Vanderbilt, PRL 82, 3296 (1999)). Smeared full-SCF FD
  gates against the free energy: Γ UHF $3.8\times10^{-10}$
  (compcell) / $1.9\times10^{-10}$ (rsgdf), KRHF (2,1,1)
  $3.5\times10^{-10}$ Ha/bohr at genuinely fractional occupations
  ($f \approx 1.86/0.15$), plus a $T \to 0$ consistency pin at
  $\le 10^{-15}$. FD-differencing `result.energy` instead of
  `result.free_energy` disagrees by the $T\,dS/dR$ term (~0.1
  Ha/bohr on the gate fixture); the free energy is the force
  surface. `bz_integration="smearing"` is accepted (it is literally
  the same occupation path). Restriction: Fermi-Dirac only for now:
  the MP/MV generalized entropies are conjugate to their occupations,
  but their GDF force paths do not yet have dedicated full-SCF gates,
  and MP occupations can be negative while the present fractional-block
  assembly takes square roots. Fixed-geometry smeared GDF gradients remain
  available and differentiate the free energy. High-level smeared GDF
  optimization now fails closed: although the force is the matching $dA/dR$,
  the optimization, trajectory, and QVF schemas do not yet distinguish the
  Mermin objective $A$ from internal energy $E$. **Slab (dim=2) GDF
  gradients landed 2026-07-30** (G-PBC-002 § 6): on a `dim=2` system,
  `run_krhf_periodic_gdf` / `run_krks_periodic_gdf` with
  `compute_gradient=True` assemble the analytic gradient of the slab
  route's two-gauge composition: the bare Parry / de Leeuw-Perram
  2D-Ewald nuclear term, the slab $V_{ne}$ derivative (z-resolved
  reciprocal blocks included), the S/T lattice folds, the signed
  Sundararaman-Arias truncated-metric DF-J/K fit derivative (the
  indefinite metric's Fréchet response, finite negative $K(0)$ zero
  mode and all), and the BvK probe-charge exxdiv overlap response,
  plus the periodic-Becke XC Pulay for RKS. Full-SCF FD on the
  compact H2 (2,2,1) slab fixture at h=2e-4: KRHF
  $\le 2.2\times10^{-10}$ Ha/bohr per component (gate 1e-6,
  slab normal included; Γ (1,1,1) reduction $2.1\times10^{-10}$).
  KRKS is gated at $2\times10^{-4}$: the XC term is exact at fixed
  grid ($9\times10^{-12}$, asserted), and the full-SCF residual
  (PBE/PBE0 $\sim 10^{-5}$ to $8\times10^{-5}$ on that deliberately
  compact cell) is the moving periodic-Becke partition's
  `becke_image_radius_bohr` truncation derivative; it falls to
  $1.4\times10^{-6}$ at a 20-bohr image radius and is the same
  quadrature-limited class as the documented Γ KS gates.
  `run_periodic_job(optimize=True, jk_method="gdf")` on a slab
  relaxes on this objective through the same optimizer capture as
  the bulk routes; `optimize_cell` stays fail-closed (no slab GDF
  analytic stress). Slab open shell, smearing, custom meshes, and
  restarts remain fail-closed on the slab route itself; the ASE
  calculator still accepts only fully periodic (`pbc=(True, True,
  True)`) cells, so ASE slab forces remain closed at the boundary.
  Not yet differentiated
  (pinned pre-dry-run errors naming the reason): MDF/compcell multi-k
  fits, non-Fermi-Dirac smearing flavors, `bz_integration="gilat"`,
  custom-weight k-point sets, pure-functional bulk multi-k RKS without
  the cached-Lpq fit, range-separated functionals,
  `k_exchange="cosx"`, DFT+U, wire (dim=1) meshes. **Optimizer wiring landed 2026-07-30:**
  `run_periodic_job(optimize=True)` with a GDF-routed SCF (Γ and
  multi-k) relaxes on this GDF analytic-gradient objective -- the
  exact driver call the SCF dispatched, re-run with
  `compute_gradient=True` at each candidate geometry -- rather than
  switching to the BIPOLE force surface, so the relaxed geometry is a
  stationary point of the energy the job reports. The `.out`
  optimization block states the active force objective; unsupported
  gradient envelopes fail during runner preflight before dry-run/SCF, and
  periodic dispersion and positive-temperature requests fail before SCF
  because the optimizer does not add D3 per geometry and cannot yet label
  a Mermin free-energy objective distinctly;
  `optimize_cell=True` fails closed on the GDF and BIPOLE routes because
  neither has a certified production stress and coupled cell optimizer.
  Periodic GDF KS also fails closed on VV10/nonlocal-correlation and
  double-hybrid functionals because those drivers do not add the nonlocal
  VV10 potential or the perturbative MP2 correlation term.
  Molecular RIJK has separate analytic-gradient coverage; see
  [`density_fitting.md`](density_fitting.md#df-analytic-gradient-status).
* **Gamma-only `gdf_method="compcell"` cannot do tight ionic cells.**
  On compact cells with Z > 1 atoms (e.g. rocksalt LiH primitive FCC,
  115 bohr³) the Gamma-only compcell Hartree cannot resolve AO-pair
  overlap between periodic images and the SCF converges to a
  non-physical fixed point (LiH/STO-3G: +579.8 Ha at HF, +1172.6 Ha at
  UKS-PBE, vs PySCF's -8.24 Ha). The drivers warn on entry and the
  energy-sanity guard rejects the converged garbage:
  `run_pbc_gdf_uhf` / `run_pbc_gdf_uks` raise a `RuntimeError`
  (bypass with `check_energy_sanity=False` for diagnostics);
  `run_pbc_gdf_rhf` warns and tags the result backend
  `+SANITY_FAILED`. Because of this, the **open-shell Gamma drivers
  default to `gdf_method="rsgdf"` since 2026-07-09** (on the broken
  LiH fixture the rsgdf defaults land at -8.33 Ha UHF / -8.23 Ha
  UKS-PBE, at the documented few-mHa diffuse-Li fitting floor from
  PySCF); `compcell` stays selectable for development and keeps the
  warning + guard. A KS run through `run_pbc_gdf_uks` with
  `options=None` also defaults to `PeriodicKSOptions` now, so the
  periodic-Becke-grid/torus-density XC pairing engages by default
  (the old `PeriodicRHFOptions` fallback silently evaluated XC in the
  v0.8.x molecular-grid convention: -7.9638 Ha on the same fixture).
  `run_pbc_gdf_rhf` keeps its `compcell` default (its warn-and-tag
  contract is pinned); use the multi-k route (validated at µHa parity
  on LiH FCC at `kmesh=(2,2,2)`) or `gdf_method="rsgdf"` at Gamma for
  such cells. Note the production runner already dispatches Gamma GDF
  jobs with `gdf_method="rsgdf"`; the compcell default only concerns
  direct API calls to the closed-shell driver.
* **`gdf_method="mdf"` fails closed on compact dense-core cells.**
  MDF's real-space Hartree inherits the same Gamma-only tight-ionic
  compcell instability: measured on MgO primitive FCC/STO-3G
  (G-GDF-001 MDF sub-gate, 2026-07-29), Gamma MDF is non-convergent
  and trial-dependent at `mdf_ke_cutoff=40/60` (+1703/+2143/+418 Ha)
  and falsely reports `converged=True` at `ke=80` at -1651 Ha (PySCF
  MDF: -271.0499 Ha); the multi-k `(1,1,2)` route lands -11741 Ha
  non-converged. All five GDF drivers raise `NotImplementedError` on
  this class (Z >= 8 in < 250 bohr³/atom) instead of returning the
  value behind a warning. Use the default `gdf_method="rsgdf"` there
  (the Gamma driver auto-sizes its high-|G| tail to PySCF parity;
  multi-k accepts `rsgdf_tail_ke_cutoff` on every method -- a pure-DFT
  dim=3 multi-k run given an explicit tail routes its Hartree through
  the tail-consuming cached-Lpq GDF J instead of the default EWALD_3D
  J, which builds no cderi and cannot consume one; pre-fix the knob
  was accepted, echoed, and silently ignored on that route, IID 146).
  **Since IID 518 that pure-DFT dim=3 multi-k route also auto-sizes the
  tail on the dense-core class**, exactly as the Gamma and CCM routes
  always have, so a plain `run_periodic_job` on such a cell no longer
  needs the explicit knob and is no longer tagged `+PARITY_HELD`.
  Sparse cells are unaffected: the classifier is basis- and cell-driven,
  resolves no tail there, and leaves the EWALD_3D default bit-identical.
  The other multi-k routes (KRHF, hybrids, and the open-shell KUHF/KUKS
  drivers) are already on the cached-Lpq J, where a tail is genuine
  added cderi cost at n_k^2 pair builds; they still require the explicit
  knob and still tag `+PARITY_HELD` without it. One visible consequence
  on a dense-core cell: a `KUKS(M=1)` run does not reproduce its `KRKS`
  twin, and carries the tag where the KRKS run no longer does.
  MDF's validated envelope is
  unaffected: the steep-core atom in a vacuum box, where it closes the
  all-electron floor at a modest mesh (Ne/STO-3G/10-bohr: 0.14 mHa vs
  PySCF MDF). Since the same closure, that class is no longer
  mis-tagged `+PARITY_HELD` (the tight-basis reciprocal-mesh hold is an
  rsgdf concept; MDF resolves the steep core in real space).
* **Auxiliary basis coverage.** Standard `def2-*-jkfit` aux bases
  work cleanly. `pob-tzvp` does not yet have a matching aux:
  designing a `pob-tzvp-jk` is a paper-worthy item on its own; the
  pragmatic workaround is `def2-tzvp-jk` (aux size scales with
  orbital size, not orbital identity) or falling back to BIPOLE on
  pob-TZVP work.
* **Aux conditioning is real.** Diffuse aux primitives produce a
  near-singular metric; vibe-qc uses pivoted Cholesky with a
  `linear_dep_thr` knob (default `1e-9`) and the modrho
  renormalisation, but you can still see warnings on very diffuse
  composite bases. Cut diffuse exponents with `drop_eta` if needed.

### 3.5 When to use GDF

* You are doing closed-shell RHF / RKS on a `dim=1` wire, a `dim=3` bulk
  crystal, or a closed-shell `dim=2` slab on a full Gamma-centered tuple mesh
  (energies and analytic forces).
* You want the cheapest production route on bulk for hybrid DFT
  and large basis (def2-SVP through def2-TZVP class).
* You want an open-shell UHF / UKS single-point on the density-fitting
  metric (Gamma or multi-k). In 3D, pass `jk_method="gdf"` explicitly
  because AUTO routes open-shell to BIPOLE; in 1D, AUTO already selects GDF.
* You want analytic forces on the periodic path: the GDF gradient
  ladder covers Gamma, multi-k, and closed-shell `dim=2` slabs.

### 3.6 Quick start

```python
import vibeqc as vq

# 2D graphene sheet, RKS / PBE, Gamma-only.
# AUTO resolves a dim=2 slab to the direct SLAB_EWALD_2D route. Select GDF
# explicitly only for the bounded closed-shell fitted energy route.
sysp  = build_graphene()                       # dim=2 PeriodicSystem
basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")

result = vq.run_periodic_job(
    sysp, basis,
    method="RKS", functional="pbe",
    jk_method="gdf",                           # signed slab-truncated fit
    kpoints=(2, 2, 1),
    output="graphene-pbe",
)
print(f"E_total = {result.energy:+.6f} Ha / cell")
```

Multi-k closed-shell example:

```python
import vibeqc as vq
from vibeqc import monkhorst_pack
from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

sysp  = build_mgo()
basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")
kmesh = monkhorst_pack(sysp, [4, 4, 4])

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 12.0
opts.use_diis = True

result = run_krhf_periodic_gdf(sysp, basis, kmesh, opts,
                               aux_basis="def2-svp-jk")
print(f"E_total = {result.energy:+.6f} Ha   k-mesh = 4x4x4")
```

The full runner-level workflow is at
[`examples/periodic/input-mgo-rocksalt-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-mgo-rocksalt-rhf.py)
and the unified-runner variants under
`examples/periodic/input-mgo-*`.

### 3.7 Citations

* J. L. Whitten, "Coulombic potential energy integrals and
  approximations", *J. Chem. Phys.* **58**, 4496 (1973).
* B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, "On some
  approximations in applications of X-alpha theory",
  *J. Chem. Phys.* **71**, 3396 (1979).
* O. Vahtras, J. Almlof, M. W. Feyereisen, "Integral
  approximations for LCAO-SCF calculations",
  *Chem. Phys. Lett.* **213**, 514 (1993).
* K. Eichkorn, O. Treutler, H. Ohm, M. Haser, R. Ahlrichs,
  "Auxiliary basis sets to approximate Coulomb potentials",
  *Chem. Phys. Lett.* **240**, 283 (1995).
* F. Weigend, "Accurate Coulomb-fitting basis sets for H to Rn
  (def2-jkfit family)", *Phys. Chem. Chem. Phys.* **8**, 1057
  (2006). [DOI 10.1039/b515623h](https://doi.org/10.1039/b515623h).
* Q. Sun, T. C. Berkelbach, J. D. McClain, G. K.-L. Chan, "Gaussian
  and plane-wave mixed density fitting for periodic systems",
  *J. Chem. Phys.* **147**, 164119 (2017). The reference algorithm
  for the modrho-compensated periodic GDF that vibe-qc reproduces.

### 3.8 Periodic ROHF and ROKS: restricted open-shell SCF

For open-shell systems that need a single set of doubly-occupied spatial
orbitals plus a defined set of singly-occupied orbitals (rather than the
independent α / β orbitals of UHF), vibe-qc ships periodic restricted
open-shell Hartree-Fock drivers on the Ewald-3D gauge and Gamma-point GPW.
The Gamma-point GPW route also provides the corresponding spin-pure ROKS
method with periodic XC:

```python
import vibeqc as vq
from vibeqc import run_periodic_rohf_gpw, run_periodic_roks_gpw
from vibeqc.periodic_rohf_ewald import run_rohf_periodic_gamma_ewald3d
from vibeqc.periodic_rohf_multi_k_ewald import run_rohf_periodic_multi_k_ewald3d
```

`run_rohf_periodic_gamma_ewald3d` runs the Gamma-point ROHF SCF;
`run_rohf_periodic_multi_k_ewald3d` runs the multi-k variant over a
Monkhorst-Pack mesh. Both share the Ewald-3D one-electron gauge and J / K
build with the UHF Ewald path, so the total energy is gauge-consistent
with the other 3D routes.

`run_periodic_rohf_gpw` is the 3D Gamma-point HF alternative. It combines
the smooth-grid GPW Hartree J with per-spin exact K and the periodic Ewald
one-electron/nuclear gauge, then uses the shared Roothaan loop to produce one
restricted spatial-orbital set with integer closed/open/virtual occupations
of 2/1/0. The same maintained-preview route is available through
`vq.run_periodic_job(system, basis, method="ROHF", jk_method="gpw")`.

The corrected-Ewald-exchange multi-k engine is also wired into the public
runner on the BIPOLE route:
`vq.run_periodic_job(system, basis, method="ROHF", jk_method="bipole")`
(the AUTO default when only `method="ROHF"` is given) runs
`run_rohf_periodic_multi_k_ewald3d` at Gamma (a `(1,1,1)` mesh) or on a
full Monkhorst-Pack mesh via `kpoints=[n1, n2, n3]`, with the ordinary
`.out` / `.system` / QVF / citation output. The LiH+/STO-3G `(3,1,1)`
anchor of that engine is validated out of process against PySCF KROHF/GDF
to 0.062 mHa. BIPOLE-specific knobs the engine does not honour (MOM, ODA,
multipole far-field, `ewald_omega` overrides, `sr_image_precision`,
Fock mixing) fail closed rather than being silently ignored.

`run_periodic_roks_gpw(..., functional="pbe")` uses the same restricted
orbital and Roothaan contracts while adding the established spin-polarised
GPW XC potential and energy. It is also available through
`vq.run_periodic_job(system, basis, method="ROKS", functional="pbe",
jk_method="gpw")`. LDA, GGA, meta-GGA, and global-hybrid functionals share
the Gamma GPW UKS envelope. Range-separated and direct double-hybrid ROKS are
not implemented on this backend.

ROKS also runs on the BIPOLE route, on the same corrected-Ewald-exchange
multi-k engine as ROHF above:
`vq.run_periodic_job(system, basis, method="ROKS", functional="pbe",
jk_method="bipole")` reaches `run_roks_periodic_multi_k_ewald3d` at Gamma
(a `(1,1,1)` mesh) or on a full Monkhorst-Pack mesh via
`kpoints=[n1, n2, n3]`. Each per-spin Fock is
$F_\sigma(k) = H_\mathrm{core}(k) + J(k) - c_\mathrm{full} K_\sigma(k) +
V_{xc}^\sigma(k)$, and the two are combined by the same Roothaan coupling
ROHF uses. Supported: LDA, GGA, meta-GGA and **global** hybrids. A pure
functional builds no exchange at all: neither the exchange caches nor the
$O(n_k^2)$ long-range $K$ loop, which is what makes dense meshes
affordable there. Screened hybrids (`hse06` and friends) need an
erfc-attenuated exchange arm this engine does not carry and fail closed, as
do long-range-corrected functionals and double hybrids. The Li/PBE/STO-3G
`(1,1,2)` anchor agrees with an out-of-process PySCF `KROKS().density_fit()`
reference to 0.11 mHa, and the closed-shell PBE0 limit reproduces the
BIPOLE RKS route to 0.02 mHa.

```{note}
Like the ROHF route, this driver keeps the cell's alpha and beta counts at
every k point. Its ROKS-specific occupation rule also detects when the one
open electron selects an exactly degenerate orbital set. It locks that set as
one restricted-open shell and shares the electron equally over its members;
for example, a twofold shell has occupations `0.5/0.5`. This is the fixed,
zero-temperature ensemble for a symmetry-degenerate shell, not electronic
smearing. The Li/PBE/STO-3G 6-bohr-box `(2,2,1)` case now converges in 6
iterations instead of 87-200 cycles with an arbitrary integer SOMO.

General metallic filling, finite-temperature open-shell smearing, and
multiple partially occupied open shells are not inferred by this path. Use a
supported unrestricted route when the system needs those occupation models.
```

**ROHF on the native GDF route.** `run_periodic_job(method="ROHF",
jk_method="gdf")` is the normal output-producing entry point;
`run_krohf_periodic_gdf`
(`periodic_rohf_gdf.py`) is the density-fitted alternative to the Ewald-3D
engine above. Gamma is the `(1,1,1)` mesh of the same code path, so there
is only one driver to reason about:

```python
from vibeqc import run_periodic_job

result = run_periodic_job(
    system, basis, method="ROHF", jk_method="gdf",
    kpoints=(3, 1, 1),                    # omit for Gamma
    aux_basis="def2-svp-jk", gdf_method="rsgdf",
)
result.energy, result.s_squared, result.mo_occupations[0]
```

It reuses the multi-k GDF machinery verbatim -- the per-$(k_i, k_j)$
cderi cache, Hartree $J$ from the BZ-summed total density, the
`exxdiv='ewald'` BvK-supercell Madelung shift applied once per spin, and
the converged Ewald nuclear repulsion -- and differs from
`run_kuhf_periodic_gdf` only in the orbital step, where the two spin
Focks are combined by Roothaan's effective Fock. At multiplicity 1 it
reproduces `run_krhf_periodic_gdf` to machine precision; Li/STO-3G in a
10-bohr box at Gamma sits $+2.86\times10^{-5}$ Ha from out-of-process
PySCF 2.13.1 `KROHF/GDF`, and that residual collapses to
$+2.2\times10^{-7}$ Ha as the rsgdf mesh is refined -- the shared
fit-truncation ladder, not a convention difference.

Exchange is the only $O(n_k^2)$ term, so it is built by contracting the
cderi cache against the **occupied MO blocks** rather than the
$n_\mathrm{bf}\times n_\mathrm{bf}$ densities: the same operator at
$n_\mathrm{occ}/n_\mathrm{bf}$ of the cost, as two GEMM calls per k-pair
(so the work stays inside threaded BLAS). Since 2026-08-02 every multi-k
GDF driver builds exchange that way, not just this one. The rsgdf cderi
setup rides the existing MPI k-point farming described in § 6.

```{note}
That contraction is an exact reassociation, not an approximation, and it
does not change what any run reports: converged energies move by at most
$2.5\times10^{-11}$ Ha, with identical iteration counts. It also does not
speed up a *small* job noticeably, because on test-scale cells the
exchange build is a negligible fraction of the wall time (the cderi build
dominates); the saving is aimed at production meshes and realistic bases,
where exchange is what dominates.
```

ROKS on GDF is a separate increment and is **not** implemented: a
functional raises rather than silently running something else. Smearing
and 2D slabs raise as well. Analytic gradients, COSX exchange and
symmetry-reduced (`ibz_native`) exchange are not part of this driver's
argument surface at all, so there is no way to ask for them. The public route
adds `.out`, `.system`, Molden/QVF sidecars when applicable, and the ROHF plus
GDF references; it does not change the standalone driver's energy or gauge.

For both methods, GAPW, 1D/2D cells, multi-k GPW, gradients,
restart densities, and electronic smearing fail closed. The standalone
drivers'
`smearing_alpha` argument controls nuclear-charge smoothing in the
one-electron potential, not electronic Fermi smearing.

ROHF avoids the spin contamination of UHF at the cost of the more involved
coupling operator; reach for it when a clean $\langle S^2\rangle$ matters more
than variational freedom.

## 4. GPW and GAPW: the plane-wave / augmented route

### 4.1 Theory

The Gaussian + plane-wave (GPW) and Gaussian-augmented plane-wave
(GAPW) methods, due to Lippert and Hutter and implemented as the
core CP2K periodic-DFT engine, replace the four-centre Coulomb
build with a Poisson solve on a uniform real-space grid:

1. **Collocate** the Gaussian-built electron density onto a uniform
   3D grid sampled in fractional cell coordinates,
   $\rho(\mathbf{r}) = \sum_{\mu\nu} D_{\mu\nu}
   \chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r})$.
2. **Solve Poisson** in reciprocal space:
   $V_H(\mathbf{G}) = 4\pi\,\rho(\mathbf{G})/|\mathbf{G}|^2$, with
   $V_H(\mathbf{G}=0)$ pinned to zero. A neutral cell handles the
   long-range tail through this gauge; an unbalanced charge gets
   the standard uniform-background self-image shift, which is the
   same Madelung convention as the rest of vibe-qc.
3. **Project back to the AO basis**:
   $J_{\mu\nu} = \int \chi_\mu(\mathbf{r}) \chi_\nu(\mathbf{r})
   V_H(\mathbf{r})\,\mathrm{d}\mathbf{r}
   \approx \Delta V \sum_g \chi_\mu(\mathbf{r}_g)
   \chi_\nu(\mathbf{r}_g) V_H(\mathbf{r}_g)$.

The cost of the FFT is $\mathcal{O}(N_g \log N_g)$ where $N_g =
N_x N_y N_z$ is set by the plane-wave cutoff
$|\mathbf{G}_{\max}|^2 = 2 E_{\text{cut}}$. For a smooth pseudo-
density that is much cheaper than building the full four-centre
ERI tensor.

The catch is that core densities are sharply peaked, and a grid
fine enough to resolve them is enormous. The two responses to
this are:

* **GPW**: use a pseudopotential to remove the core electrons.
  The remaining valence pseudo-density is smooth and a moderate
  $E_{\text{cut}}$ (tens to a few hundred Ry) suffices.
* **GAPW** (Lippert and Hutter, 1999): keep the all-electron
  problem, but split the density into a smooth part $\tilde\rho$
  that lives on the coarse plane-wave grid plus per-atom
  hard / soft corrections $\rho_a - \tilde\rho_a$ that live on a
  fine *radial* grid around each nucleus. The Hartree potential
  decomposes as

  $$
  V_H[\rho]
    = V_H[\tilde\rho]
    + \sum_a \Big( V_H[\rho_a] - V_H[\tilde\rho_a] \Big),
  $$

  where the smooth piece reuses the GPW Poisson solve and the
  atomic pieces are local 1D radial Poisson solves with no FFT.
  The augmentation restores all-electron accuracy without paying
  the cost of resolving the core on the smooth grid. The PAW
  generalisation (Blochl 1994) takes the same dual-grid idea
  further by transforming all-electron orbitals through a
  per-atom projector basis.

### 4.2 Status in vibe-qc

The GPW / GAPW track grew out of the v0.10.x acceleration program and now
**drives a full periodic SCF**. The design doc with the full milestone
history is at
[`design_periodic_gapw.md`](../design_periodic_gapw.md).

Both routes are reached with `jk_method="gpw"` or `jk_method="gapw"` on
`run_periodic_job`, and both emit a `GAPWExperimentalWarning` when you opt
in. GAPW RHF/UHF additionally requires
`gapw_molecular_limit=True`; without that explicit isolated-cell declaration
it fails closed because its promoted analytic one-centre correction is not a
dense-crystal image-summed functional. What runs today:

* **Gamma-point SCF for RHF, RKS, UHF, and UKS** via
  `run_periodic_rhf_gpw` / `run_periodic_uhf_gpw` / `run_periodic_uks_gpw`
  (GPW) and `run_periodic_rhf_gapw` / `run_periodic_uhf_gapw` /
  `run_periodic_uks_gapw` (GAPW). RKS / UKS require a `functional=`.
* **Maintained-preview 3D Gamma-point ROHF and ROKS** via
  `run_periodic_rohf_gpw` / `run_periodic_roks_gpw`, or the corresponding
  `run_periodic_job(method="ROHF", jk_method="gpw")` or
  `run_periodic_job(method="ROKS", functional=..., jk_method="gpw")` route.
  Both use
  GPW J and the shared Roothaan loop with integer 2/1/0 occupations; ROHF adds
  per-spin exact K and ROKS adds spin-polarised XC plus functional-scaled K.
  They are not GAPW, multi-k, or electronic-smearing routes.
* **Multi-k RKS (pure DFT)** via `run_periodic_rks_gpw_multi_k` and
  `run_periodic_rks_gapw_multi_k`, including LDA/GGA and self-regularising
  meta-GGAs such as r2SCAN. Hybrid / HF multi-k is not yet wired on this
  route and raises.
* **Forces, +U, Fermi-Dirac smearing, and D3(BJ) dispersion.** GPW/GAPW
  cover RHF/RKS/UHF/UKS +U at Gamma and RKS +U at multi-k; multi-k
  open-shell +U is not wired. GPW retains the high-level finite-difference
  optimization and Hessian plumbing. GAPW RHF/UHF instead rejects high-level
  `optimize`, `optimize_cell`, and `hessian` requests so a follow-on workflow
  cannot silently change the analytic one-centre energy surface. Fixed-cell
  GAPW forces remain available through `VibeqcGAPW`; block-mode DFT retains
  the existing derivative plumbing.

The milestone trail that landed this:

| Milestone | Status | Deliverable |
|---|---|---|
| M1 | landed | `PlaneWaveGrid` infrastructure + Gaussian-on-grid collocation |
| M2 | landed | GPW Hartree-J via FFT-Poisson + full Gamma-point SCF |
| M3 | landed | GAPW dual-grid augmentation; multi-k RKS; open-shell UHF / UKS |
| M4 | in progress | Metallic GAPW refinements with smearing |
| M5 (stretch) | scheduled | ACE-decorated K for hybrids + on-demand PAW dataset fetcher |

The grid kernels live in
[`periodic_gapw_j.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic_gapw_j.py)
(GPW) and
[`periodic_gapw_augment.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic_gapw_augment.py)
(GAPW augmentation), with the runner dispatch in `periodic_runner.py`. The
`GpwJBuilder` orchestrates *density collocation* -> *FFT-Poisson* ->
*AO-basis projection* into the Hartree-J matrix; the GAPW driver adds the
per-atom augmentation on top.

```{note}
GPW and GAPW carry an opt-in `GAPWExperimentalWarning` because the route is
younger than GDF / BIPOLE and its parity sweep against CP2K is still
broadening, not because it is unimplemented. For tight-cell hybrid work the
mature choice is still GDF or BIPOLE.
```

### 4.3 Advantages

* **Linear-log scaling Hartree-J.** FFT-Poisson is
  $\mathcal{O}(N_g \log N_g)$, independent of basis size. For
  large supercells with modest valence densities this is the
  decisive scaling win.
* **Natural fit for large cells.** Slabs with hundreds of atoms
  and metallic systems where smearing matters benefit most.
* **GAPW gives all-electron accuracy** without bundled
  pseudopotentials, by carrying the core augmentation on per-atom
  radial grids.
* **Pluggable exchange.** GAPW replaces the J build only; exchange
  goes through whichever K-builder the existing periodic SCF
  would use. ACE decoration (Lin Lin 2016) is the planned
  hybrid-DFT accelerator and is being designed so it can be shared
  with the locality-exploitation track (ADMM, OT).

### 4.4 Disadvantages (intrinsic and current)

* **Younger than GDF / BIPOLE.** The route runs a full SCF but its
  CP2K parity sweep is still broadening, so it carries the opt-in
  `GAPWExperimentalWarning`. For a production tight-cell hybrid, prefer
  GDF or BIPOLE today.
* **Multi-k is pure-DFT RKS only.** Hybrid and HF multi-k are not yet
  wired on the GPW / GAPW route; multi-k hybrids stay on BIPOLE.
* **Plane-wave cutoff is an extra knob.** Sensible defaults are
  derived from the tightest primitive exponent and a CP2K-style
  `REL_CUTOFF` safety factor (calibrated against `REL_CUTOFF = 60 Ry`).
* **Pseudopotential licensing is non-trivial.** The v0.10.x design
  decision (see `design_periodic_gapw.md` section 9) is to bundle
  *no* PAW dataset and instead build a vqfetch-style on-demand
  fetcher that filters out GPL / proprietary records. GBRV (CC-BY-SA),
  ABINIT-JTH (mixed), GPAW datasets (GPL), PSlibrary (mixed), and
  VASP POTCARs (proprietary) all fail the bundle test under
  MPL-2.0 (see [`docs/license.md`](../license.md) for the full
  inventory).
* **3D-focused.** The route is exercised mainly on 3D periodic cells;
  2D slab and 1D wire coverage is thinner than on GDF, which remains the
  go-to for low-dimensional systems.
* **FFTW3 dependency is GPL-v2.** The combined binary linking
  FFTW3 is effectively GPL-v2 (`CLAUDE.md` section 1). A future
  FFT-backend abstraction
  (`VIBEQC_FFT_BACKEND={fftw3, pocketfft, kissfft}`) is on the
  roadmap to allow commercial-friendly binaries.

### 4.5 When to use GPW / GAPW

Reach for GPW / GAPW when you want plane-wave Hartree-J on a large
smooth-density cell: large-cell periodic DFT (slabs and bulk supercells
with hundreds of atoms) where the BIPOLE direct-K cost and the GDF
aux-basis cost both become uncomfortable, and where the
$\mathcal{O}(N_g \log N_g)$ FFT-Poisson build pays off as $N$ grows. GAPW
adds all-electron accuracy without pseudopotential files. Both run a full
$\Gamma$-point SCF (RHF / RKS / UHF / UKS) plus multi-k RKS today. For
tight-cell hybrids and for production CRYSTAL parity, stay on GDF / BIPOLE
until the GPW / GAPW parity sweep and hybrid multi-k land.

### 4.6 Citations

* G. Lippert, J. Hutter, M. Parrinello, "A hybrid Gaussian and
  plane-wave density functional scheme", *Mol. Phys.* **92**,
  477 (1997).
* G. Lippert, J. Hutter, M. Parrinello, "The Gaussian and
  augmented-plane-wave density functional method for *ab initio*
  molecular dynamics simulations", *Theor. Chem. Acc.* **103**,
  124 (1999). The GAPW formulation.
* M. Krack, M. Parrinello, "All-electron *ab initio* molecular
  dynamics", *Phys. Chem. Chem. Phys.* **2**, 2105 (2000).
* J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello,
  T. Chassaing, J. Hutter, "Quickstep: fast and accurate density
  functional calculations using a mixed Gaussian and plane waves
  approach", *Comput. Phys. Commun.* **167**, 103 (2005). The
  CP2K Quickstep reference.
* P. E. Blochl, "Projector augmented-wave method",
  *Phys. Rev. B* **50**, 17953 (1994). The PAW formulation that
  the M5 stretch goal targets.
* L. Lin, "Adaptively compressed exchange operator",
  *J. Chem. Theory Comput.* **12**, 2242 (2016). The ACE
  acceleration of exact exchange that M5 wraps around GAPW for
  hybrid-DFT scaling.

### 4.7 RSGAPW: range-separated GAPW for screened hybrids

Range-separated hybrids (HSE06, ωB97X, ωB97M, CAM-B3LYP) split the
Coulomb operator with the error function,
$1/r = \mathrm{erf}(\omega r)/r + \mathrm{erfc}(\omega r)/r$, and apply a
different fraction of exact exchange to each piece. vibe-qc carries these
on the plane-wave grid through a dedicated range-separated GAPW path:

```python
from vibeqc.periodic_gapw_range_sep import (
    run_periodic_rhf_rsgapw,   # range-separated HF exchange
    run_periodic_rks_rsgapw,   # range-separated hybrid KS
)
```

The long-range (smooth) Coulomb piece is solved by FFT-Poisson on the
plane-wave grid; the short-range piece uses libint erfc-screened
integrals. The GAPW per-atom augmentation sees only the total (unscreened)
Coulomb potential, so the $\omega$-split applies to the smooth-grid J and
the K builders alone. `omega_for_functional(name)` returns the
$(\omega, \text{HF}_{SR}, \text{HF}_{LR})$ triple for the supported
functionals.

This is a **dedicated entry point, not a `jk_method` route**: you call
`run_periodic_rhf_rsgapw` / `run_periodic_rks_rsgapw` directly rather than
selecting it through `run_periodic_job(jk_method=...)`. Citations: Heyd,
Scuseria, Ernzerhof, *J. Chem. Phys.* **118**, 8207 (2003) (HSE); Chai and
Head-Gordon, *J. Chem. Phys.* **128**, 084106 (2008) (ωB97X); Yanai, Tew,
Handy, *Chem. Phys. Lett.* **393**, 51 (2004) (CAM-B3LYP).

### 4.8 RIJCOSX: density-fitted J with chain-of-spheres K

RIJCOSX pairs a GDF-fitted Coulomb build with a seminumerical
chain-of-spheres (COSX) exchange build on a periodic Becke grid. It is the
periodic analogue of the molecular RIJCOSX accelerator (Neese 2009) and is
selected with `jk_method="rijcosx"`:

```python
result = vq.run_periodic_job(sysp, basis, method="RHF",
                             jk_method="rijcosx")
```

What runs today:

* **Gamma RHF.** The dedicated `run_periodic_rijcosx_rhf` driver keeps the
  original periodic RIJCOSX path for one-point closed-shell HF. Its two
  Gamma JK builders iterate on the smooth seminumerical exchange surface,
  then apply the analytic one-center replacement to the final Fock and
  orbitals after convergence. The reported energy remains on the iterated
  seminumerical surface, matching the molecular RIJCOSX contract.
* **True multi-k RHF / RKS / UHF / UKS.** `run_periodic_job(...,
  jk_method="rijcosx", kpoints=(...))` uses the native multi-k GDF loop for
  RI-J and switches exact exchange to the composed periodic COSX backend
  (`k_exchange="cosx"`). Closed-shell density mixers, closed-shell READ
  restart, closed-shell +U, and closed-shell `rsgdf_tail_ke_cutoff` follow
  the same support envelope as the GDF multi-k route.
  After convergence the multi-k drivers apply the analytic one-center
  replacement to the returned Fock matrices and orbitals, evaluated with
  the same short-range kernel as the bridge's real-space exchange; the
  reported energy remains on the iterated seminumerical surface, matching
  the molecular RIJCOSX and Gamma-builder contract.
* **One-cell RKS on vacuum-padded cells.** `run_periodic_job(...,
  method="RKS", jk_method="rijcosx", kpoints=(1, 1, 1))` runs the Gamma
  RIJCOSX RKS driver when every active lattice length is at least
  12 bohr and the cell is not tight: molecular DF J plus periodic COSX
  K through the dedicated Gamma builder, XC integrated on the molecular
  Becke grid (exact in that envelope), and the post-convergence
  one-center replacement applied to the returned Kohn-Sham matrix.
  Hybrids route their exact-exchange fraction through COSX; pure
  functionals never build K.
* **One-cell UHF/UKS on vacuum-padded cells.** The open-shell Gamma
  RIJCOSX drivers follow the same envelope rule as RKS: molecular DF J
  from the total density, per-spin periodic COSX K, per-spin
  post-convergence one-center replacement, and (UKS) XC on the
  molecular Becke grid. Spin counts follow the system multiplicity.
* **Home-cell electrostatics on the vacuum-padded envelope.** Because
  the molecular DF J of these one-cell drivers contains no image-cell
  electrons, the nuclear attraction, the nuclear repulsion and the COSX
  exchange are evaluated over the home cell as well, regardless of
  `cutoff_bohr` and `nuclear_cutoff_bohr`. In periodic boundary
  conditions only the total electrostatic energy of a neutral cell is
  well defined (Makov and Payne 1995); keeping image nuclei or the
  image-ket exchange class without the matching image electrons leaves
  the bare monopole sums in the energy. Before this was enforced (issue
  #707), a one-electron H atom in a 12-bohr box converged to -1.638 Ha
  instead of -0.467 Ha, and H2 RHF in a 24-bohr box was 0.5 Ha too low.
  Every box inside the envelope now reproduces the molecular RIJCOSX
  result to the COSX quadrature floor; the neglected image terms are the
  neutral-cell multipole tail that the molecular DF J already neglects.
* **One-cell tight-cell RKS/UHF/UKS fail closed.** Tight or short-axis
  cells need the periodic-density XC grid and the Gamma-folded exchange
  classes, so vibe-qc refuses to label those one-cell calculations as
  RIJCOSX.

When to use it: explore the periodic COSX approximation for HF and hybrid
DFT on compact periodic cells, with GDF as the independent reference route.
For pure functionals such as PBE, RIJCOSX and RI share the same zero-exact-
exchange KS Fock build; use a hybrid such as PBE0 when you specifically want
to exercise COSX exchange. Citation: F. Neese, F. Wennmohs,
A. Hansen, U. Becker, "Efficient, approximate and parallel Hartree-Fock and
hybrid DFT calculations. A 'chain-of-spheres' algorithm for the
Hartree-Fock exchange", *Chem. Phys.* **356**, 98 (2009).

## 5. Side-by-side comparison

### 5.1 Dimensionality and capability matrix

|  | 1D wires | 2D slabs | 3D bulk | RHF | UHF | RKS pure | RKS hybrid | UKS | Multi-k | Analytic gradient | Cell opt |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **SLAB_EWALD_2D** | no | **yes** | no | yes | no | yes | yes | yes | yes | no | no |
| **BIPOLE** | no (direct-API diagnostic only) | raises | yes | yes | yes | yes | yes | yes | yes | preview | no |
| **GDF** (native) | yes | closed-shell | yes | yes | yes (Γ + multi-k) | yes | yes | yes (Γ + multi-k) | yes | yes (Γ / multi-k / closed-shell slab) | no |
| **GPW** | limited | raises | yes | yes (Γ) | yes (Γ) | yes (Γ) | yes (Γ) | yes (Γ) | yes (RKS) | FD | no |
| **GAPW** | limited | raises | yes | yes (Γ) | yes (Γ) | yes (Γ) | yes (Γ) | yes (Γ) | yes (RKS) | block only | no |
| **RIJCOSX** | yes (multi-k) | raises | yes | yes | yes (multi-k) | yes (multi-k) | yes (multi-k) | yes (multi-k) | yes | no | no |
| **RSGAPW** | limited | raises | yes | yes (Γ) | no | yes (Γ) | yes (Γ, RS) | no | no | FD | FD |
| **ROHF / ROKS** | no | no | yes | ROHF (Ewald; GDF Γ + multi-k; GPW Γ) | n/a | ROKS (Ewald; GPW Γ) | ROKS (Ewald global hybrids; GPW Γ) | no | yes (ROHF + ROKS Ewald) | no | no |

**Slabs have two routes.** `SLAB_EWALD_2D` is what `jk_method="auto"`
selects for `dim=2`, and the explicit slab-GDF route is the closed-shell
alternative that also carries analytic gradients (`optimize=True`,
2026-07-30). Every bulk builder raises on `dim=2` rather than returning
an `a3`-dependent energy. Fermi-Dirac smearing and open-shell HF still
need a `dim=3` cell (`vq.slab(..., periodic_z=True)`); so do
freeze-mask relaxation and NEB, which run on the BIPOLE force stack.

The direct `run_pbc_bipole_*` APIs retain a 1D `DIRECT_TRUNCATED`
molecular-limit diagnostic. `run_periodic_job` does not expose it: explicit
BIPOLE rejects dim=1, and AUTO selects the maintained GDF wire route.

Notes on the matrix: GDF open-shell (UHF / UKS) ships at Gamma and multi-k.
In 3D, AUTO routes open shell to BIPOLE by design, so you opt into GDF with an
explicit `jk_method="gdf"`; in 1D, AUTO selects GDF. GPW / GAPW multi-k is
pure-DFT RKS only (hybrid / HF multi-k raise on that route). GPW forces and
fixed-cell gradients use the finite-difference path the runner shares; cell
optimization fails closed. GAPW direct
analytic gradients cover block mode, while high-level HF optimization and
Hessian requests fail closed; block-mode DFT retains its existing derivative
plumbing, but cell optimization also fails closed. `VibeqcGAPW` uses analytic block-mode forces or full-SCF finite
differences for the molecular-limit analytic mode.
RIJCOSX is Gamma RHF plus true multi-k RHF/RKS/UHF/UKS through the GDF/COSX
backend; one-cell RKS/UHF/UKS requests fail closed so they are not mislabeled.
RSGAPW (range-separated GAPW) remains a dedicated entry point. ROHF has
dedicated Ewald entry points, public and standalone native-GDF entry points
(`run_periodic_job(method="ROHF", jk_method="gdf")` and
`run_krohf_periodic_gdf`, Gamma + multi-k), and a public
`jk_method="gpw"` route at 3D Gamma
(the ROHF "RHF" column reads as restricted open-shell HF on the Ewald-3D
gauge at Gamma and multi-k, on the GDF route at Gamma and multi-k, plus
GPW at Gamma). RSGAPW handles screened
hybrids such as HSE06 / ωB97X / CAM-B3LYP.

### 5.2 Scaling and cost intuition

| | Per-iter J | Per-iter K | Memory | Notes |
|---|---|---|---|---|
| BIPOLE | $\mathcal{O}(N^2)$ per cell + reciprocal $J^{LR}$ | $\mathcal{O}(N^4)$ direct with Schwarz | $\mathcal{O}(N^2)$ | $K$ dominates at large cutoff. The quartet multipole prototype is fail-closed and is not a supported shortcut. |
| GDF | $\mathcal{O}(N^2 M)$ via $L_{P,\mu\nu}$ on fitted branches; $\mathcal{O}(N_k N^2 N_g N_c)$ batched AFT on the 3D pure-DFT Ewald-J fallback | $\mathcal{O}(N^2 M)$ via $L_{P,\mu\nu}$; absent for pure DFT | $\mathcal{O}(N^2 M)$ for fitted $L$; bounded FT batch plus $\mathcal{O}(N_c N^2)$ J blocks on the pure-DFT fallback | Aux size $M \approx 3N$; the pure-DFT fallback never retains the all-cell/all-$G$ FT tensor. |
| GPW / GAPW | $\mathcal{O}(N_g \log N_g)$ FFT + $\mathcal{O}(N N_g)$ collocation + projection | inherits whichever K builder is composed with it (target: ACE) | $\mathcal{O}(N_g)$ for the grid | $N_g$ depends on the plane-wave cutoff, not on $N$; the win shows up as $N$ grows. |

`N` is the AO basis size per unit cell, `M` is the auxiliary
basis size, `N_k` is the number of k-points, `N_g` is the reciprocal
or plane-wave grid count, and `N_c` is the number of real-space lattice
cells.

### 5.3 Method-selection chart

```
Periodic SCF
  |
  +-- dim=2 slab
  |     +-- AUTO: SLAB_EWALD_2D (energies)
  |     +-- explicit closed-shell GDF (energies + forces)
  |
  +-- dim=1 wire
  |     +-- GDF for RHF/RKS/UHF/UKS
  |     +-- ROHF unavailable through AUTO
  |
  +-- dim=3
        +-- UHF/UKS: AUTO selects BIPOLE
        +-- closed shell: GDF by default
              +-- BIPOLE for exact-route FD forces and fixed-cell optimization
              +-- GPW/GAPW for large smooth-density systems
```

The slab branch comes first because it is exclusive: no bulk builder
accepts `dim=2`. Explicit closed-shell slab GDF supplies analytic forces.
For smearing, open shell, freeze masks, or periodic Gaussian NEB, build the
cell as `dim=3` with `vq.slab(..., periodic_z=True)` instead. These are
separate capability envelopes: finite-temperature BIPOLE single points are
supported, but BIPOLE atom optimization and periodic Gaussian NEB require
zero-temperature occupations.

The open-shell branch points at BIPOLE because that is the AUTO default;
open-shell GDF (UHF / UKS, Gamma + multi-k) also ships and is reachable
with an explicit `jk_method="gdf"`.

### 5.4 Coexistence

The three routes share one infrastructure stack and may be mixed
in the same session without re-installation:

* `S`, `T`, `V_{ne}` lattice integrals live in C++ behind
  `compute_overlap_lattice`, `compute_kinetic_lattice`, and the
  shared Ewald-gauged $V_{ne}$ helper. All three drivers consume
  them.
* `nuclear_repulsion_per_cell` is the same Ewald-3D nuclear sum
  for everyone on a 3D system.
* `BasisSet` and `Molecule` constructors are shared with the
  molecular SCF; the same auxiliary-basis manager
  (`vibeqc.aux_basis.make_aux_basis_set`) backs both molecular
  RIJK and periodic GDF.
* The `.system` manifest records which J/K route was active, plus
  its parameters (Ewald omega for BIPOLE, aux basis name +
  `linear_dep_thr` for GDF, plane-wave cutoff and grid shape for
  GPW / GAPW). A future re-run reproduces the exact route.

## 6. Worked examples by bonding situation and dimensionality

The examples below intentionally cover different bonding regimes
because the right route is rarely a function of dimensionality
alone. Diffuseness of the valence density, charge transfer,
band-gap, and the strength of the long-range Coulomb tail all
push you toward different J/K kernels.

All systems use the canonical lattice parameters from
[`examples/periodic/_systems.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/_systems.py)
(Springer Materials).

### 6.1 Ionic, closed-shell 3D: MgO and NaCl (rocksalt)

Ionic rocksalt crystals are the textbook stress test for the
periodic Coulomb gauge: charge transfer of order one electron
between cation and anion, Madelung sums that diverge under naive
truncation, and a wide gap so the SCF itself is well-behaved.
Both BIPOLE and GDF are production routes here.

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

# MgO FCC primitive (1 Mg + 1 O), a = 4.211 A.
a = 4.211 * ANG2BOHR
lattice = (a / 2.0) * np.array(
    [[0.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0]]
)
mgo = vq.PeriodicSystem(
    3, lattice,
    [vq.Atom(12, [0.0, 0.0, 0.0]),
     vq.Atom(8,  [a / 2.0, a / 2.0, a / 2.0])],
)
basis = vq.BasisSet(mgo.unit_cell_molecule(), "sto-3g")

# Route A: BIPOLE, multi-k 2x2x2 (matches CRYSTAL14 SHRINK 2 baseline).
res_bipole = vq.run_periodic_job(
    mgo, basis, method="RHF",
    jk_method="bipole",
    kpoints=(2, 2, 2),
    output="mgo-bipole",
)

# Route B: GDF Gamma-only with the universal JK aux.
res_gdf = vq.run_periodic_job(
    mgo, basis, method="RHF",
    jk_method="gdf",
    aux_basis="def2-universal-jkfit",
    output="mgo-gdf",
)

print(f"BIPOLE 2x2x2 : {res_bipole.energy:+.6f} Ha")
print(f"GDF    Gamma : {res_gdf.energy:+.6f} Ha")
```

NaCl (a = 5.64 A) has a much more diffuse valence density on Cl
and so probes aux conditioning more aggressively. Swap the
geometry helpers and rerun; the GDF route may need `drop_eta` on
the auxiliary basis if the metric is near-singular.

Pick BIPOLE for fixed-cell finite-difference forces, or GDF where its
analytic gradient envelope applies. Use neither for variable-cell work.

### 6.2 Covalent network 3D: diamond and silicon (Fd-3m)

Diamond C and silicon are the canonical covalent network
crystals: every atom sp3, narrow density between bonded atoms,
no charge transfer, indirect band gap of order eV. The lattice
vectors are FCC primitive (60-deg angles between the column
vectors, *not* orthogonal axes).

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

# Si diamond primitive (2 Si in the rhombohedral cell), a = 5.431 A.
a = 5.431 * ANG2BOHR
lattice = (a / 2.0) * np.array(
    [[0.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0]]
)
si = vq.PeriodicSystem(
    3, lattice,
    [vq.Atom(14, [0.0, 0.0, 0.0]),
     vq.Atom(14, [a / 4.0, a / 4.0, a / 4.0])],
)
basis = vq.BasisSet(si.unit_cell_molecule(), "sto-3g")

# GDF, multi-k 4x4x4 (closed-shell RHF or RKS / PBE).
res_rks = vq.run_periodic_job(
    si, basis,
    method="RKS", functional="pbe",
    jk_method="gdf",
    aux_basis="def2-universal-jkfit",
    kpoints=(4, 4, 4),
    output="si-pbe",
)
print(f"Si bulk PBE : {res_rks.energy:+.6f} Ha / cell")
```

For diamond C, swap to `Atom(6, ...)` and `a = 3.567 A`. Diamond's
parity reference at STO-3G with BIPOLE on SHRINK 8 is
$-74.8771393842\,\text{Ha}$ (delta vs CRYSTAL14: $-0.145\,\text{mHa}$).

Covalent crystals are also a clean target for the GPW / GAPW route, which
runs a full SCF today: the smooth pseudo-density on the FFT grid
represents the bonding region efficiently, and this is where the
plane-wave scaling win shows up first as the cell grows.

### 6.3 Metallic 3D: fcc aluminium with Fermi-Dirac smearing

Metals are the second classical stress test for periodic SCF:
partial occupations at the Fermi level, no gap to anchor DIIS,
and integer-occupation aufbau diverges. vibe-qc handles this with
finite-temperature Fermi-Dirac smearing on the multi-k BIPOLE
RKS / RHF path. The driver auto-routes through BIPOLE when
`smearing_metallic=True`.

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

# Al fcc primitive (1 Al), a = 4.05 A.
a = 4.05 * ANG2BOHR
lattice = (a / 2.0) * np.array(
    [[0.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0]]
)
al = vq.PeriodicSystem(
    3, lattice, [vq.Atom(13, [0.0, 0.0, 0.0])],
)
basis = vq.BasisSet(al.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    al, basis,
    method="RKS", functional="pbe",
    jk_method="bipole",
    kpoints=(8, 8, 8),
    smearing_temperature="auto",   # ~0.01 Ha for metallic systems
    smearing_metallic=True,
    output="al-fcc-pbe",
)
print(f"Al fcc PBE   : E   = {result.energy:+.6f} Ha")
print(f"               F   = {result.free_energy:+.6f} Ha (= E - T*S)")
print(f"               E_f = {result.fermi_level:+.6f} Ha")
```

Two things to note. First, the meaningful thermodynamic quantity
on a smeared metal is the free energy $F = E - T S$, not the
total energy; both are reported. Second, `smearing_metallic=True`
relaxes the gap-closure safety guards that the standard SCF
inserts; on a wide-gap insulator it should stay `False` (the
default) so an accidental run on the wrong template fails loudly.
For UKS open-shell metals, use the BIPOLE UKS route when you need per-spin
Fermi-Dirac smearing today.  The non-BIPOLE multi-k Ewald-UKS path still
raises on positive `smearing_temperature` so spin-polarized metallic inputs do
not run with silently integer occupations.

### 6.4 Open-shell metallic 3D: bcc lithium (BIPOLE UHF)

Light alkali metals are the simplest place to exercise the
open-shell periodic SCF. BIPOLE is the AUTO open-shell route and the one
that carries open-shell Fermi-Dirac smearing on the multi-k UKS / RKS
path; open-shell GDF and GPW / GAPW also run UHF / UKS (GDF at Gamma and
multi-k, GPW / GAPW at Gamma) when selected explicitly.

`PeriodicSystem` defaults to multiplicity 1. When an odd-electron primitive
cell reaches open-shell SCF with that default, `run_periodic_job` selects the
lowest parity-compatible multiplicity (normally a doublet); the same rule is
applied after conventional-to-primitive reduction. An explicit multiplicity
is preserved. Restricted RHF and RKS instead reject an odd-electron final cell
and direct the calculation to UHF or UKS.

The `run_periodic_job(method="ROHF", jk_method="gpw")` route and its
standalone `run_periodic_rohf_gpw` driver are available for a 3D Gamma-point
spin-pure HF reference, but they keep integer 2/1/0 occupations and are not
electronic-smearing routes. The standalone driver's `smearing_alpha` option
is nuclear smoothing, not a metal occupation temperature.

```python
import numpy as np
import vibeqc as vq
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

a = 6.6   # bohr, bcc Li
sysp = vq.PeriodicSystem(
    3, np.eye(3) * a,
    [vq.Atom(3, [0.0, 0.0, 0.0])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 14.0
opts.use_diis = True
opts.max_iter = 40
opts.initial_guess = vq.InitialGuess.SAD

result = run_pbc_bipole_uhf(sysp, basis, kmesh, opts,
                            ewald_precision=1e-6)
print(f"E_total = {result.energy:+.6f} Ha   <S^2> = {result.s_squared:.4f}")
```

The full open-shell input lives at
[`examples/periodic/input-bipole-li-uhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-bipole-li-uhf.py).

### 6.5 Mixed ionic-covalent 3D: ZnO wurtzite (hexagonal cell)

ZnO wurtzite mixes polar ionic character (Zn-O charge transfer)
with covalent sp3 bonds in a non-orthogonal hexagonal cell
(`gamma = 120 deg`). It is the canonical test that the periodic
machinery does *not* secretly assume cubic axes.

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

# ZnO wurtzite (P6_3 m c, #186). a = 3.2495 A, c = 5.2069 A, u = 0.382.
a, c, u = 3.2495 * ANG2BOHR, 5.2069 * ANG2BOHR, 0.382
lattice = np.array([
    [a,            0.0, 0.0],
    [-a / 2.0,     a * np.sqrt(3.0) / 2.0, 0.0],
    [0.0,          0.0, c],
]).T

# Two formula units in the conventional hex cell.
unit_cell = [
    vq.Atom(30, lattice @ np.array([1/3, 2/3, 0.0])),     # Zn
    vq.Atom(30, lattice @ np.array([2/3, 1/3, 0.5])),
    vq.Atom( 8, lattice @ np.array([1/3, 2/3, u])),       # O
    vq.Atom( 8, lattice @ np.array([2/3, 1/3, 0.5 + u])),
]
zno = vq.PeriodicSystem(3, lattice, unit_cell)
basis = vq.BasisSet(zno.unit_cell_molecule(), "pob-tzvp")

# GDF would need a matching pob-tzvp-jk aux which is not yet shipped;
# BIPOLE handles pob-TZVP natively without an aux basis.
result = vq.run_periodic_job(
    zno, basis,
    method="RKS", functional="pbe",
    jk_method="bipole",
    kpoints=(4, 4, 4),
    output="zno-wurtzite-pbe",
)
print(f"ZnO wurtzite PBE : {result.energy:+.6f} Ha")
```

For wurtzite-class systems on pob-TZVP, BIPOLE is the operational
default until the pob-TZVP JK aux basis is published (see
[`density_fitting.md`](density_fitting.md#auto-picking-an-aux-basis)).

### 6.6 Van der Waals 3D: neon fcc (RKS / PBE + D3)

Closed-shell rare-gas solids are dominated by dispersion. The
HF / pure-GGA part is mostly a sanity check on the periodic SCF;
the cohesion comes from the D3 / D4 correction added on top.
Both BIPOLE and GDF work here; GDF is cheaper since the system
is closed-shell.

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903
a = 4.43 * ANG2BOHR   # Ne fcc conventional cubic
lattice = (a / 2.0) * np.array(
    [[0.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0]]
)
ne = vq.PeriodicSystem(3, lattice, [vq.Atom(10, [0.0, 0.0, 0.0])])
basis = vq.BasisSet(ne.unit_cell_molecule(), "def2-svp")

result = vq.run_periodic_job(
    ne, basis,
    method="RKS",
    functional="pbe-d3bj",          # PBE + D3(BJ) dispersion
    jk_method="gdf",
    aux_basis="def2-svp-jk",
    kpoints=(2, 2, 2),
    output="ne-fcc-pbe-d3bj",
)
print(f"Ne fcc PBE+D3 : {result.energy:+.6f} Ha")
```

vdW solids are also where the *opt-in* nature of GAPW's all-
electron augmentation matters most: relative cohesive energies
on the meV scale are easy to get wrong with too-coarse a smooth
grid, so a CP2K parity sweep on Ne / Ar is part of the M3
acceptance set in the [GAPW design doc](../design_periodic_gapw.md).

### 6.7 1D: trans-polyene chain (Peierls distortion)

A 1D chain is the cleanest demonstration of why dimensionality
matters: the 3D Madelung problem vanishes, GDF runs natively
without any compensating-charge machinery, and Peierls distortion
opens a gap in the otherwise metallic uniform chain. Both the
uniform and the distorted geometries should run on the same code
path.

```python
import numpy as np
import vibeqc as vq

# Trans-polyacetylene-like H-H chain. Column 0 is the periodic
# axis; columns 1 and 2 are vacuum (>= 25 bohr decouples images).
a       = 4.0    # bohr, lattice period
delta   = 0.20   # bohr, Peierls displacement (=0 for the uniform chain)

sysp = vq.PeriodicSystem(
    dim=1,
    lattice=np.diag([a, 30.0, 30.0]),
    unit_cell=[vq.Atom(1, [0.0,           0.0, 0.0]),
               vq.Atom(1, [1.4 + delta,   0.0, 0.0])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    sysp, basis,
    method="RHF",
    jk_method="gdf",
    kpoints=(8, 1, 1),
    output="h-chain-peierls",
)
print(f"H-chain (delta = {delta:+.2f} bohr) : {result.energy:+.6f} Ha")
```

The matching bandstructure / DOS workflow lives at
[`examples/periodic/input-h-chain-bands.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-bands.py);
the uniform vs Peierls comparison at
[`examples/periodic/input-h-chain-peierls.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-peierls.py).

### 6.8 2D: graphene sheet (semi-metal) and a graphene nanoribbon

Graphene at half-filling is the textbook 2D Dirac semimetal: the
Fermi level sits exactly at the K-point band crossing, so any
finite-temperature smearing > 0 introduces partial occupation in
the pi* band. A Gamma-only single-point will still run cleanly
because the cone is not at Gamma.

```python
import numpy as np
import vibeqc as vq

# Graphene sheet, primitive hexagonal cell, a = 2.46 A in-plane.
# slab_2d takes only the two in-plane vectors: no vacuum, real-z atoms.
ANG2BOHR = 1.0 / 0.529177210903
a = 2.46 * ANG2BOHR

graphene = vq.slab_2d(
    [a,       0.0,                    0.0],
    [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
    [vq.Atom(6, [0.0,     0.0,                    0.0]),
     vq.Atom(6, [a / 2.0, a * np.sqrt(3.0) / 6.0, 0.0])],
)
basis = vq.BasisSet(graphene.unit_cell_molecule(), "def2-svp")

result = vq.run_periodic_job(
    graphene, basis,
    method="RKS", functional="pbe",
    jk_method="auto",                 # dim=2 -> SLAB_EWALD_2D
    kpoints=(6, 6, 1),                # in-plane mesh; 1 along the normal
    output="graphene-pbe",
)
print(f"Graphene PBE : {result.energy:+.6f} Ha / cell")
```

For an armchair graphene *nanoribbon* (effectively a 1D system
embedded in a 2D sheet of vacuum), the same code path holds with
`dim=1` and only the ribbon-axis lattice column physical.

```{admonition} Low-dimensional cells get no AUTO convergence profile
:class: warning

The AUTO convergence profiler's covalent / ionic / metallic signals are
**3D-only**: they are gated on a tight-cell test that requires `dim == 3`.
Every `dim=1` and `dim=2` cell therefore reports

    profile: unknown
      - no low-dimensional branch (dim=2, ...) -- the covalent/ionic/metallic
        signals are 3D-only, so this cell is unclassified by construction,
        not judged inconclusive; a semimetallic or small-gap sheet needs an
        explicit smearing choice

and resolves `smearing_temperature = 0`. That default is *wrong for a
semimetal*: graphene at half-filling cannot converge on integer
occupations, and a small-gap sheet may not either. Pass `smearing_method`
and `smearing_temperature` explicitly for such systems on routes that
support them. Note that the explicit closed-shell **slab GDF** route
currently rejects smearing outright (`NotImplementedError`), so a 2D
semimetal has no smeared path there yet.

Band occupations on the multi-k slab GDF route are filled under **one
global Fermi level** across the whole mesh, not per k point. When the
bands overlap, that fill is fractional at the Fermi-degenerate group and
the `.out` reports `occupation class = fractionally occupied / smeared`
rather than a zero-temperature band gap.
```

```{admonition} A negative band gap is printed, not clamped
:class: note

`gap (indirect)` is reported as `CBM - VBM` with **no floor applied**. If
the value comes out negative, the band ordering across the mesh is
inverted and the line carries a `negative: band ordering inverted across
the mesh` annotation. Earlier releases clamped this at `0.000000 Ha`,
which made an inverted ordering indistinguishable from a genuine zero
gap. Treat a negative gap as a diagnostic to investigate (CLAUDE.md § 7),
not as a physical result.
```

### 6.9 Slab: MgO(001) 5-layer surface

A slab is a genuinely 2D periodic system: the in-plane lattice vectors
are the surface mesh, and there is nothing periodic along the normal.
Give `vq.slab_2d` those two vectors and the atoms at their real `z`.
No vacuum is needed, and none is used: the third lattice column is
synthesized for AO and spglib bookkeeping, and the SCF energy does not
depend on it.

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903
a = 4.211 * ANG2BOHR

# MgO(001) 5-layer slab, primitive 1x1 surface cell. No vacuum: the two
# in-plane vectors are the whole periodicity, and the layers keep their
# real z.
in_plane = a / np.sqrt(2.0)

layers = []
for k, layer_z in enumerate(np.linspace(-2.0 * a, 2.0 * a, 5)):
    if k % 2 == 0:
        layers.append(vq.Atom(12, [0.0,      0.0,      layer_z]))
        layers.append(vq.Atom( 8, [in_plane / 2.0, in_plane / 2.0, layer_z]))
    else:
        layers.append(vq.Atom( 8, [0.0,      0.0,      layer_z]))
        layers.append(vq.Atom(12, [in_plane / 2.0, in_plane / 2.0, layer_z]))

slab = vq.slab_2d([in_plane, 0.0, 0.0], [0.0, in_plane, 0.0], layers)
basis = vq.BasisSet(slab.unit_cell_molecule(), "pob-tzvp")

result = vq.run_periodic_job(
    slab, basis,
    method="RKS", functional="pbe",
    jk_method="auto",                 # dim=2 -> SLAB_EWALD_2D
    kpoints=(4, 4, 1),                # in-plane mesh; 1 along the normal
    output="mgo-001-slab",
)
print(f"MgO(001) 5-layer : {result.energy:+.6f} Ha / cell")
```

```{warning}
The pre-v0.16 bulk GDF path was wrong on `dim=2`: it summed the Coulomb
interaction as if the layers were stacked `a3` apart, so the nuclear
repulsion depended on the k-mesh and the total energy could be off by
thousands of Hartree. The current explicit slab-GDF route is a different
implementation with a finite 2D Coulomb metric and rigorous Parry-gauge
one-electron terms. It is closed-shell only; analytic gradients ship
since 2026-07-30.
```

Surface slabs are the use case where GPW / GAPW's scaling tells: the
smooth-grid Hartree-J cost is set by the cell volume, which scales
linearly with slab thickness, while the BIPOLE direct-K cost grows
quadratically with the number of in-plane atoms. GPW / GAPW run a full SCF
today (Gamma RHF / ROHF / ROKS / RKS / UHF / UKS, multi-k pure-DFT
RKS / ROKS / UKS), but
GPW/GAPW do not accept
a `dim=2` slab. The general slab route is `SLAB_EWALD_2D`; closed-shell
energy and relaxation work may explicitly use the slab-truncated GDF
route (analytic gradients since 2026-07-30).

### 6.10 Method-choice summary by bonding regime

| System type | Recommended first choice | Notes |
|---|---|---|
| Ionic 3D bulk (MgO, NaCl, LiH) | GDF (closed-shell) or BIPOLE (forces / opt) | Both cleanly handle the Madelung gauge. |
| Covalent 3D bulk (diamond, Si) | GDF for cost, BIPOLE for CRYSTAL parity | GPW / GAPW is a natural smooth-density fit and runs full SCF today. |
| Metallic 3D bulk (Al, Cu) | BIPOLE + smearing (`smearing_metallic=True`) | RKS/UHF/UKS BIPOLE support Fermi-Dirac; BIPOLE RHF remains gated. |
| Open-shell 3D (Li bcc, magnetic oxides) | BIPOLE UHF / UKS (AUTO) | GDF UHF / UKS (Gamma + multi-k) and GPW / GAPW (Gamma) also run open-shell; ROHF and ROKS are available on Ewald-3D (Gamma + multi-k), while maintained-preview GPW provides Gamma ROHF and Gamma or multi-k pure-DFT ROKS with integer occupations. |
| Mixed ionic-covalent (ZnO, TiO2, Al2O3) | BIPOLE on pob-TZVP, GDF on def2-*-jk | pob-TZVP-jk aux is on the roadmap. |
| Van der Waals 3D (Ne, Ar, molecular crystals) | GDF + dispersion correction | GPW / GAPW carry D3(BJ) too; their meV-cohesion CP2K sign-off is still broadening. |
| 1D wires (H chain, polyacetylene) | GDF (any zeta) | AUTO uses GDF for closed and open shell; public BIPOLE rejects dim=1. |
| 2D sheets (graphene, h-BN) | SLAB_EWALD_2D (AUTO); explicit GDF for closed-shell energy scaling | Both are vacuum-free and a3-invariant. Slab GDF is full-tuple and zero-temperature; closed-shell analytic forces ship (`optimize=True`). |
| 2D surface slabs | SLAB_EWALD_2D (AUTO); explicit GDF for closed-shell energies + relaxation | No vacuum needed. Plain closed-shell relaxation: explicit GDF with `optimize=True`. Freeze-mask/NEB/smearing workflows: `vq.slab(..., periodic_z=True)` and run dim=3. |

## 7. Surface reactions: adsorption, TS search, vibrational analysis

Surface chemistry (adsorption energies, reaction barriers,
transition states, intermediates, reaction-path scans) is the
**flagship workflow** vibe-qc is built around. Bulk single-points
are infrastructure for it; molecular SCF is infrastructure for it.
This section walks the end-to-end stack: slab setup, adsorbate
placement, geometry relaxation, NEB reaction path, vibrational
analysis at the transition state.

### 7.1 Slab setup: dim = 2 is the physical model

There are two ways to make a slab look "non-periodic in the surface
normal direction". They are not equally correct.

* **dim = 2 (the physical model, and the default).** Give the two
  in-plane lattice vectors and put the atoms at their real Cartesian
  `z`. There is **no vacuum gap**. Build it with `vq.slab_2d(a1, a2,
  atoms)` or `vq.slab(...)`, which returns `dim=2`. `jk_method="auto"`
  resolves to `CoulombMethod.SLAB_EWALD_2D`, the rigorous vacuum-free
  2D Ewald gauge (Parry 1975; de Leeuw and Perram 1979). The total
  energy is provably invariant to the synthesized third lattice column
  and the nuclear repulsion is k-mesh independent. Available for
  RHF / RKS / UKS at Gamma and multi-k.
  **Analytic gradients in dim = 2** ship on the closed-shell
  slab-GDF route (2026-07-30): `jk_method="gdf"` with
  `optimize=True` (or the drivers' `compute_gradient=True`) relaxes
  atoms on the slab analytic-gradient objective. **Still not
  available in dim = 2:** gradients on the direct `slab_ewald_2d`
  route itself, NEB / FD Hessians / freeze masks, Fermi-Dirac
  smearing, `UHF`, and the bulk J/K routes (`bipole`, `gpw`, `gapw`,
  `rijcosx`), all of which raise rather than return a wrong number.
* **dim = 3 with thick vacuum** (the plane-wave-code convention).
  Make the c-axis large enough that the slab images do not see each
  other (20 to 30 bohr of vacuum on each side), and run the whole
  calculation in 3D. Build it with `vq.slab(..., periodic_z=True)`.
  **Production finite-difference forces work** through BIPOLE; analytic
  gradients remain a gated preview. Smearing is available, and the periodic
  ASE adapter (`vibeqc.ase_periodic`)
  accepts the geometry without special-casing. The cost is real: you
  pay for one extra Bloch sum in the normal direction, and a residual
  slab-slab dipole-image interaction shifts the absolute energy by an
  amount that grows with the surface dipole and shrinks only as the
  vacuum grows. It cancels in *differences* such as binding energies
  only to the extent that the dipole is the same on both sides.

```{warning}
Do **not** build a `dim=2` cell and then hand it to a bulk J/K builder.
A slab run through a 3D Coulomb gauge is treated as a crystal of sheets
stacked `a3` apart, so its energy depends on `a3` **and** on the k-mesh.
vibe-qc now raises rather than letting that happen. Before v0.15.32 it
did not, and `jk_method="auto"` silently chose GDF.
```

**Which to use.** For single points and reference energies, use `dim = 2`:
it is the correct physics and costs less. Explicit closed-shell slab GDF also
supports fixed-cell relaxation. For freeze masks, periodic Gaussian NEB,
open shell, or metallic-smearing **single points**, use `dim = 3 + thick
vacuum` today and accept the image error. Do not combine positive smearing
with BIPOLE optimization or periodic Gaussian NEB; both routes fail closed
until their result schemas represent the Mermin free-energy objective.
Periodic BIPOLE Hessians fail closed in every dimension.

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

def build_pt111_slab(n_layers=4, n_xy=2, vacuum_bohr=20.0):
    """Pt(111) n_xy x n_xy x n_layers slab. dim = 3 with thick vacuum."""
    a_pt = 3.92 * ANG2BOHR / np.sqrt(2.0)        # nearest-neighbour, bohr
    in_plane = a_pt * n_xy
    layer_spacing = a_pt * np.sqrt(2.0 / 3.0)    # FCC (111) interlayer

    # Hex 2D mesh in-plane, c-axis = slab + vacuum.
    lattice = np.array([
        [in_plane,            0.0,                      0.0],
        [in_plane / 2.0,      in_plane * np.sqrt(3.0) / 2.0, 0.0],
        [0.0,                 0.0,                      n_layers * layer_spacing + 2 * vacuum_bohr],
    ]).T

    atoms = []
    for layer in range(n_layers):
        z = layer * layer_spacing + vacuum_bohr
        # ABCABC stacking offset per layer.
        shift = (layer % 3) * np.array([in_plane / 3.0,
                                        in_plane * np.sqrt(3.0) / 6.0, 0.0])
        for i in range(n_xy):
            for j in range(n_xy):
                pos = (i * np.array([a_pt, 0, 0])
                       + j * np.array([a_pt / 2.0,
                                       a_pt * np.sqrt(3.0) / 2.0, 0])
                       + shift
                       + np.array([0, 0, z]))
                atoms.append(vq.Atom(78, list(pos)))

    return vq.PeriodicSystem(dim=3, lattice=lattice, unit_cell=atoms)

slab = build_pt111_slab(n_layers=4, n_xy=2, vacuum_bohr=20.0)
print(f"{len(slab.unit_cell)} Pt atoms in 2x2x4 slab")
```

For convenience you can also build a slab through ASE and
convert:

```python
from ase.build import fcc111
from vibeqc.ase_periodic import atoms_to_periodic_system

ase_slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0, a=3.92)
ase_slab.pbc = (True, True, True)   # the dim = 3 + vacuum convention
slab = atoms_to_periodic_system(ase_slab)
```

`atoms_to_periodic_system` currently *requires* `pbc.sum() == 3`,
which is exactly the convention above. Slabs declared with
`pbc = (True, True, False)` still fail closed at the boundary even
though the dim = 2 slab-GDF gradient now exists (2026-07-30):
mapping ASE's 2D `pbc` onto the slab route's synthesized-normal
convention is a boundary-convention decision that has not been
made, so the constraint stays until it is. Native dim = 2
relaxation goes through `run_periodic_job(jk_method="gdf",
optimize=True)` instead.

### 7.2 Adsorbate placement

vibe-qc ships a native, pure-Python slab + adsorbate builder
(``vibeqc.slab`` + ``vibeqc.place_adsorbate``) that does not
depend on ASE at runtime. See
[`slabs_and_adsorbates.md`](slabs_and_adsorbates.md) for the full
reference. Quick form:

```python
import vibeqc as vq

slab, info = vq.slab("Pt", "fcc", (1, 1, 1), n_layers=4, vacuum=10.0,
                     supercell=(2, 2), periodic_z=True)
# periodic_z=True -> dim=3 with a real vacuum gap. Freeze masks, NEB,
# open shell, and smeared single points use the dim=3 BIPOLE stack, but
# positive smearing cannot be combined with BIPOLE optimization or NEB.
# dim=2 analytic forces exist only on the closed-shell slab-GDF route
# (run_periodic_job(jk_method="gdf", optimize=True)).
slab_with_h2 = vq.place_adsorbate(slab, "H2", site="top")
```

Built-in adsorbate library: H, H2, N2, O2, CO, OH, NH, NH3, H2O,
CH4. Site names: `"top"` / `"bridge"` / `"hollow"` /
`"fcc-hollow"` / `"hcp-hollow"` / `"long-bridge"` /
`"short-bridge"`, plus explicit `(x, y)` coordinates.
``info.bottom_layer_indices(n)`` is the convenience accessor
that pairs with the freeze mask in § 7.3.

The ASE bridge is still available if you prefer ASE builders:

```python
from ase.build import add_adsorbate, fcc111, molecule

ase_slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0, a=3.92)
add_adsorbate(ase_slab, molecule("H2"), height=2.0, position="ontop")
ase_slab.pbc = (True, True, True)
slab_with_h2 = vq.ase_periodic.atoms_to_periodic_system(ase_slab)
```

For charged adsorbates, set `charge=` and `multiplicity=` on
`atoms_to_periodic_system` (or, with the native builder, on the
`PeriodicSystem` constructor directly).

### 7.3 Geometry relaxation

Fixed-cell BIPOLE atom relaxation goes through
`vibeqc.bipole_optimize.relax_atoms`, using finite-difference forces by
default. A `freeze_indices=` mask is available for fully 3D periodic systems,
and `VibeQCPeriodic` supplies the ASE `Calculator` surface for the same
fully periodic boundary. Explicit closed-shell slab GDF instead supports
unmasked `dim=2` analytic-gradient optimization through
`run_periodic_job(jk_method="gdf", optimize=True)`.

Do not use the former Pt/POB example as a runnable recipe. The bundled POB
coverage for Pt is ECP-paired and the public BIPOLE all-electron guard rejects
that combination. Also, `relax_atoms` has no dispersion objective, so a
compound functional spelling such as `"pbe-d3bj"` is not a substitute for a
separately implemented dispersion force. Choose a basis with a complete
all-electron record for every element and a BIPOLE-supported functional.
Long-range-corrected range-separated hybrids, VV10/nonlocal correlation, and
double hybrids fail closed. Converge the k-mesh and lattice cutoffs for the
actual system.

### 7.4 Reaction-path search via climbing-image NEB

vibe-qc ships a native Nudged Elastic Band driver, `vibeqc.run_neb`, that
handles molecular and bounded periodic systems through one entry point. See
[`neb.md`](neb.md) for the executable light-element periodic example.

Periodic Gaussian NEB fails closed before image evaluation unless the system
is `dim=3`, every element has a supported all-electron basis, and no periodic
dispersion correction is requested. POB basis names, ECP metadata or paired
bases, `dim=1`/`dim=2`, and `dispersion_params` are rejected. These guards mean
the Pt/POB/D3 sketch formerly shown here was not a runnable production route.

Two flavours of NEB matter for surface chemistry:

* **Plain NEB** (`climbing_image=False`). Relaxes the chain onto
  the MEP but does not pin the highest-energy image to the
  saddle; the TS is interpolated from the band, which is
  approximate.
* **Climbing-image NEB** (`climbing_image=True`, Henkelman,
  Uberuaga, Jonsson 2000). The highest-energy image is pushed
  along the tangent to climb up to the saddle point exactly.
  **Use this for any production barrier number.**

The ASE NEB on top of the molecular `VibeQC` calculator is still
the right pattern for molecule-in-vacuum reaction paths
(precedent: [`examples/workflows/input-nh3-umbrella-neb.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/workflows/input-nh3-umbrella-neb.py)
for NH3 umbrella inversion). For periodic systems, use native `run_neb` only
inside its declared all-electron, no-dispersion, 3D Gaussian envelope.

### 7.5 Transition-state characterisation: imaginary modes

A true first-order saddle is normally certified by one imaginary normal mode
along the reaction coordinate. Periodic BIPOLE Hessian requests now fail
closed before SCF because the former construction differentiated an isolated
unit-cell Hamiltonian rather than the executed periodic energy. Do not use a
`dim=3` vacuum cell or a frozen subset as a workaround. At present, obtain a
periodic Hessian from a separately validated route before attaching ZPE or
thermal corrections to a BIPOLE NEB barrier.

### 7.6 Pt(111) workflow status

The Pt(111) geometry builders remain useful for constructing and inspecting
candidate structures. There is currently no supported end-to-end native
Pt/POB/D3 recipe: the basis/ECP guard rejects that Gaussian combination,
periodic Gaussian NEB rejects dispersion, and periodic BIPOLE Hessians fail
closed. Treat these as explicit method boundaries, not knobs to bypass.

### 7.7 Recommended defaults for surface-reactions runs

| Knob | Default | Reason |
|---|---|---|
| `jk_method` | `"bipole"` (dim=3 force workflows) / `"auto"` (dim=2 single points) | BIPOLE is the production FD force route for closed- *and* open-shell periodic systems, and it needs dim=3. On a dim=2 slab, AUTO selects SLAB_EWALD_2D. |
| `dim` | 2 for physical slab single points and supported closed-shell GDF relaxation; 3 with converged vacuum for smearing, freeze masks, and periodic Gaussian NEB | Slab GDF analytic gradients ship. Periodic Gaussian NEB remains 3D-only. |
| `basis` | A basis with a complete all-electron record for every element | POB and ECP-paired bases are rejected by periodic Gaussian NEB; converge the chosen basis for the target chemistry. |
| `functional` and dispersion | A BIPOLE-supported semilocal, global-hybrid, or validated screened-hybrid functional for relaxation/NEB; dispersion only on routes that document matching forces | Long-range-corrected range-separated hybrids, VV10/nonlocal correlation, and double hybrids fail closed. `functional="pbe-d3bj"` is not a libxc functional, and periodic Gaussian NEB rejects `dispersion_params` before image evaluation. |
| `kpoints` | `(N, N, 1)` with `N` chosen so $\lvert\mathbf{k}\rvert\cdot \lvert\mathbf{a}_\parallel\rvert \approx 2\pi / (10\,\text{bohr})$ | Standard slab Brillouin-zone sampling; one k-point in the normal direction. |
| `cutoff_bohr` | Converge for the chosen basis and system | The required real-space extent is basis and geometry dependent; do not inherit the former Pt/POB sketch's cutoff. |
| `smearing_temperature` | `0` for BIPOLE relaxation and periodic Gaussian NEB; `"auto"` only for fixed-geometry metallic single points | The SCF drivers support smeared metals, but the BIPOLE optimizer and periodic Gaussian NEB reject positive temperature until their outputs distinguish the Mermin objective from internal energy. |
| `relax_atoms.conv_tol_grad` | $5 \cdot 10^{-4}$ Ha/bohr (production: $1 \cdot 10^{-4}$) | Standard tightness; ASE's BFGS `fmax = 0.05` eV/A equivalent. |
| NEB images | 7 (5 intermediate) | Enough resolution to resolve a single saddle; expand to 9-11 for multi-step paths. |
| NEB optimiser | quick-min (default in `vibeqc.run_neb`) + `climbing_image=True` | Quick-min is the velocity-Verlet-style integrator the native NEB driver uses; robust on noisy SCF gradients and parallelises trivially per image. |
| Reaction-coord scan (cheap first pass before NEB) | `vibeqc.relaxed_scan(system, basis, ("bond", i, j), values, freeze_indices=...)` | Single-coordinate sweep with per-step warm-start; resolves the rough barrier shape in O(N_points) SCFs without an NEB band. Use it to choose `n_images` and identify which atoms move. |

### 7.8 Current caveats and gaps

The historical work list was recorded in
[Archived `handovers/HANDOVER_SURFACE_REACTIONS.md`](https://vibe-qc.com/docs/);
items marked LANDED below reflect that snapshot. New work belongs in the
current core repository, not the archived handover.

1. **`VibeQCPeriodic` Calculator wrapper.** **LANDED.** Use
   `atoms.calc = VibeQCPeriodic(...)` for fully periodic ASE `Atoms`.
   Direct `pbc.sum() != 3` ASE conversion remains outside this wrapper.
2. **dim = 2 production forces.** **LANDED for explicit closed-shell GDF.**
   `run_periodic_job(jk_method="gdf", optimize=True)` uses the analytic slab
   gradient. The direct `SLAB_EWALD_2D` route itself, slab open shell,
   freeze masks, and NEB remain unsupported.
3. **dim = 2 Fermi-Dirac smearing.** Open. `smearing_temperature`
   is not supported by the slab-GDF envelope, while GPW / GAPW /
   RIJCOSX do not accept a slab. A metallic `dim = 2` slab therefore cannot
   be smeared today. Use dim = 3 + thick vacuum for metal surfaces that need
   it.
4. **Atom-freeze mask in `relax_atoms`.** **LANDED 2026-05-25**
   in commit ``eb83c6c7``. Use
   ``relax_atoms(..., freeze_indices=[...])`` (see § 7.3);
   ``SlabInfo.bottom_layer_indices(n)`` is the convenience
   accessor.
5. **Periodic NEB native to vibe-qc.** **LANDED 2026-05-25** in
   commits ``d47a3b55`` ... ``0395159f`` (NEB increments 1-5).
   Public API: ``vibeqc.run_neb(reactant, product, basis, ...,
   kpoints=..., freeze_indices=..., climbing_image=True)``
   returning ``NEBResult`` with ``.write_qvf(path)`` for
   vibe-view rendering. Reference: [`neb.md`](neb.md).
6. **Periodic BIPOLE Hessian.** Open. Requests fail closed because the former
   construction did not differentiate the executed periodic Hamiltonian;
   a frozen-index option cannot repair that method boundary.
7. **Surface-dipole correction.** Open (lowest priority). Polar
   slabs (terminations with net dipole, e.g. polar ZnO surfaces)
   need a dipole correction to cancel the spurious slab-slab
   image interaction in the dim = 3 + vacuum convention. The
   workaround is symmetric (non-polar) slab termination.

The remaining open gaps include slab smearing/open shell, broader NEB basis
and dispersion support, a correct periodic Hessian, and polar-slab dipole
corrections. The
[`design_periodic_gapw.md`](../design_periodic_gapw.md) plane-wave
route in v0.10.x is the other half of the long-term answer because
slab supercells with hundreds of atoms are where GPW / GAPW's
$\mathcal{O}(N_g \log N_g)$ Hartree-J scaling overtakes both
BIPOLE and GDF.

## 8. Reproducibility and provenance

vibe-qc records every routing choice in the run output so a paper
or a later re-run can reconstruct the calculation without relying
on AUTO heuristic versions:

* The `.out` banner names the resolved method (never the literal
  `"auto"`).
* The `.system` manifest records Ewald omega (BIPOLE), aux basis
  name + linear-dependence threshold (GDF), and the plane-wave cutoff
  plus grid shape (GPW / GAPW).
* The `.references` and `.bibtex` siblings carry the per-method
  citations through the [citation database](citations.md).
  Pisani-Dovesi-Roetti 1988 / Saunders 1992 / Dovesi 2014 / Erba 2023
  fire on the registered BIPOLE route;
  Whitten 1973 / Dunlap 1979 / Eichkorn 1995 / Weigend 2006 /
  Sun 2017 fire on the GDF route; Lippert-Hutter 1999 /
  VandeVondele 2005 / Krack-Parrinello 2000 / Blochl 1994 /
  Lin 2016 fire on the GPW / GAPW route.

If you publish work that used vibe-qc, the references block in
the `.out` file (plus the BibTeX sibling) is the copy-paste
surface; cite the method paper for the route you actually ran,
not just the catch-all CRYSTAL or CP2K citation.

## 9. Roadmap pointers

* **BIPOLE Phase 6a / 6b / 7**: redesign the quartet far-pair branch
  around the exact three-translation Fock domain before any activation,
  complete the higher-l `compute_ext_el_spheropole`, and pin the
  numerical parity sign-off on the 15-system demo suite. Status in
  [`bipole.md`](bipole.md#bipole-phase-status).
* **GDF open-shell**: native UHF / UKS drivers have **landed** at Gamma
  (`run_pbc_gdf_uhf` / `run_pbc_gdf_uks`) and multi-k
  (`run_kuhf_periodic_gdf` / `run_kuks_periodic_gdf`). Analytic gradients
  cover the production rsgdf Γ ladder (RHF/RKS/UHF/UKS), the multi-k
  KRHF/KRKS/KUHF/KUKS drivers, and `run_periodic_job(optimize=True)`
  relaxes GDF-routed jobs on that same GDF objective; remaining gradient
  work is stress/variable-cell plus the guarded envelopes (screened
  fits, smearing, IBZ, range separation).
* **GPW / GAPW**: M2 (Gamma SCF) and M3 (GAPW dual grid, multi-k RKS,
  open-shell) have **landed**; M4 metallic refinements with smearing and
  M5 ACE-decorated K + on-demand PAW dataset fetcher remain. Track in
  [`design_periodic_gapw.md`](../design_periodic_gapw.md).
* **RIJCOSX**: Gamma RHF, vacuum-padded Gamma RKS/UHF/UKS, and true
  multi-k RHF / RKS / UHF / UKS ship, including the post-convergence
  one-center replacement on every route, weighted full-BZ quadratures,
  and HSE-type screened hybrids (hse06-class, closed-shell) on the
  multi-k backend. The screened exchange uses the physical erfc kernel
  with its finite G=0 mode included (the VASP/CRYSTAL convention shared
  with the BIPOLE screened route); no exxdiv shift applies. Remaining
  work includes tight-cell parity for the dedicated Gamma K, open-shell
  multi-k screened hybrids, and the periodic analytic gradient.
* **Top-level RSGDF and CFMM**: enum reserved; the standalone
  `jk_method="rsgdf"` / `"cfmm"` routes are not implemented. (The
  *internal* `gdf_method="rsgdf"` backend that the GDF route uses for
  tight ionic cells is shipped and is the production default, see § 3.2.1.)

## See also

* [`bipole.md`](bipole.md): BIPOLE driver reference (phases,
  options, gradients, optimisation).
* [`density_fitting.md`](density_fitting.md): molecular RIJ / RIJK
  / RIJCOSX and the JKBuilder dispatch that GDF reuses.
* [`multi_k_scf.md`](multi_k_scf.md): multi-k SCF surface
  (BIPOLE and GDF), smearing for metals, and the multi-k GDF
  roadmap.
* [`ewald.md`](ewald.md): the Ewald primitives that the one-electron
  and the long-range J builds share.
* [`k_points.md`](k_points.md): Monkhorst-Pack mesh generation
  and BZ sampling.
* [`crystal_lattices.md`](crystal_lattices.md): the 14 Bravais
  lattices with worked examples.
* [`citations.md`](citations.md): how method routing produces the
  per-run references block.
* [`design_native_gdf.md`](../design_native_gdf.md): design
  walkthrough for the native periodic GDF Lpq construction
  (modrho compensation, aux conditioning, Cholesky robustness).
* [`design_periodic_gapw.md`](../design_periodic_gapw.md): GAPW
  design doc, milestone breakdown, PAW dataset license analysis.
