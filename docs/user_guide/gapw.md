# Plane-wave Hartree-J via GPW / GAPW

```{admonition} Status: GPW production within its shipped envelope; GAPW experimental
:class: note

**GPW** (`jk_method="gpw"`) is production within the envelope listed below:
Gamma RHF / UHF / RKS / UKS, multi-k pure-DFT RKS / ROKS / UKS, Gamma analytic
gradients, DFT+U and D3-BJ. The `.out` route description carries an
experimental tag (`J/K method: GPW -- ...; experimental`) while the GPW / GAPW
family is completed. `run_periodic_job` calls the GPW drivers with
`quiet=True`, so their own opt-in warning is silent, but the grid and
J-builder infrastructure still emit `GAPWExperimentalWarning` once per build
(measured 2026-09-29 on a Gamma RHF run: `PlaneWaveGrid built at
cutoff=...` and `GPW J build: GpwJBuilder ...`); read it as a scope reminder,
not as a defect flag. **GAPW** (`jk_method="gapw"`) is experimental: its
numbers are research output, not validated results, and it is not part of the
release gate. **GPW ROHF and ROKS** are a maintained preview. The **MPI
z-slab grid overlay** is experimental. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

**GPW production envelope, shipped since v0.12:**

- RHF / RKS / UHF / UKS at Γ via ``run_periodic_job(jk_method="gpw")``
  or the standalone ``run_periodic_*_gpw`` entries.
- Multi-k pure-DFT via ``run_periodic_rks_gpw_multi_k`` (closed shell;
  LDA/GGA + meta-GGA) and ``run_periodic_uks_gpw_multi_k`` (spin-polarised;
  LDA/GGA + meta-GGA), both also reachable through ``run_periodic_job`` with
  ``kpoints=``. Both handle
  vacuum-padded / molecular-limit cells (Γ-folded density) and compact
  crystals (Bloch real-space density + per-k projection). Multi-k
  UHF / hybrids need per-k exact exchange and remain Γ-only.
- DIIS, density damping, analytic forces, finite-difference Hessians,
  DOS / band-path helpers, ``.npz`` restart I/O, and the ``VibeqcGPW``
  ASE calculator.
- Both V_ne conventions (``"ewald"`` default / ``"smeared_erfc"``).
- Reproduces the molecular RHF limit to **sub-µHa** (0.9 µHa vs
  ``pyscf`` RHF on H₂/STO-3G, run out of process).

Gamma GPW collocation uses separable Gaussian factors on axis-aligned
plane-wave grids, retaining the full grid and periodic image list. Skew cells
and custom coordinate generators use the general point evaluator. This changes
only evaluation work and floating-point rounding, not the cutoff, basis
normalization or image support. Each displaced geometry builds its own AO
cache; caches must not be shared across finite-difference displacements.

**All-electron GAPW augmentation (experimental; the two Γ correctness bugs
are fixed as of 2026-07-21, accuracy preliminary):**

- **GAPW augmentation** - all-electron per-atom radial × Lebedev
  Hartree/XC correction (``\`jk_method="gapw"\```, ``periodic_gapw_augment``).
  The two open **correctness** bugs are fixed, so the Γ-point closed-shell
  **RHF / RKS (LDA, GGA)** and open-shell **UHF** augmentation SCF no longer
  emits ``GAPWExperimentalWarning``:

  - **Analytic exact Hartree Fock.** ``GapwJBuilder.build_J`` now returns the
    exact ``J = ∂E_H/∂D`` (``½ tr(D·J) = E_H`` to ~1e-13). Previously it was
    not the energy derivative, so a partially-occupied core collapsed (O fell
    to ~−35 Ha with an empty 1s); the SCF now recovers the core (O −73.56,
    e_kin 73.4, ε₁ₛ ≈ −20).
  - **Overlapping-augmentation double-count fixed** (own-atom compensator +
    partition-of-unity across spheres). Molecular GAPW-DFT now matches the
    molecular all-electron reference: **H₂/STO-3G LDA −3.5 mHa, PBE −1.7 mHa**
    (was ~−101 mHa). Single-atom results are unchanged (the partition reduces
    to the bare window for an isolated atom).
  - **SCF energy consistency.** Gamma RKS, Gamma UKS, and closed-shell multi-k
    now include the atomic XC augmentation in the convergence energy and SCF
    trace, matching the augmented Fock and final energy breakdown. Previously,
    He/STO-3G LDA returned ``-2.7737713066 Ha`` while its last trace entry
    recorded ``-2.4735585858 Ha``, a 300.2 mHa omission. The implementation
    reuses the energy density from the same libxc evaluation that builds the
    atomic XC Fock correction.
  - **Hydrogen keeps its complete valence contraction.** The soft-basis split
    no longer treats hydrogen's 3.425 bohr⁻² STO-3G primitive as a core
    function. H has no core, so its sole contracted 1s shell remains intact;
    core-bearing atoms in mixed systems are still softened. The H₂ GAPW-HF
    residual drops from about −108 mHa to +0.732 mHa on a 24³ grid and to
    −0.0304 mHa on a 48³ grid against out-of-process PySCF 2.13.1 molecular
    RHF with the same geometry and basis.

  **Accuracy caveats (preliminary - not correctness bugs):**

  - The per-atom augmentation has an absolute residual on second-row atoms.
    The historical "Ne +339 mHa / O +215 mHa" figures were measured against
    the legacy Γ GDF fallback, which is itself ~0.38 Ha below the molecular
    limit on dense-core cells (re-baselined 2026-07-29); against honest
    oracles the legacy block route sits at Ne +53 mHa / O −1.7 / He +1.2
    at 48³, and the fit-free ``one_centre="analytic"`` mode at Ne +18 / O
    +12 / He +0.4 (see the one-centre modes section below). The method-aware
    ``"auto"`` policy selects analytic for RHF/UHF inside a declared
    molecular-limit cell and block for RKS/UKS.
    For atomization energies the per-atom residual largely cancels between
    atom and molecule, but treat **absolute** GAPW totals as preliminary.
  - **meta-GGA GAPW** is supported for **self-regularising** functionals
    (SCAN / rSCAN / r2SCAN): closed- and open-shell at Γ
    (``run_periodic_rks_gapw`` / ``run_periodic_uks_gapw`` with e.g.
    ``functional="r2scan"``), plus closed-shell pure-DFT multi-k RKS via
    ``run_periodic_rks_gapw_multi_k``. The per-atom augmentation telescopes
    the kinetic-energy-density τ term (τ from AO gradients on the atomic grid,
    routed through libxc's ``eval_*_mgga`` with the smooth grid's von
    Weizsäcker floor); compact multi-k cells additionally use Bloch τ
    collocation and per-k ``v_tau`` projection. He/STO-3G r2SCAN matches the
    molecular all-electron reference to ~0.8 mHa at Γ, the same augmentation
    floor as LDA/PBE.
    **Non-self-regularising** meta-GGAs (TPSS, M06-L, …) still fail closed
    with ``NotImplementedError`` via the smooth-grid stiffness guard (which
    runs before the augmentation) - use r2SCAN or a non-GPW/GAPW route such as
    GDF.
    Open-shell GAPW DFT
    (``run_periodic_uks_gapw``) additionally required the per-spin
    soft-generation telescoping fix (the UKS smooth XC previously collocated
    the hard density, over-binding ~300 mHa for *all* functionals).

**Experimental extensions (opt-in):**

- **Multi-k GAPW** (``run_periodic_rks_gapw_multi_k``) handles compact
  crystals as of 2026-08-01. Its Hartree term is split by locality: the
  smooth density comes from the Bloch sum with its potential projected
  per k, while the one-centre augmentation stays k-independent on the
  on-site density block (the atomic regions are non-overlapping by
  construction). Previously the whole Hartree term consumed the Γ-folded
  density, which double-counts inter-cell density on a compact cell (LiH
  rocksalt (2,2,2): 9.82 electrons for a 4-electron cell; the SCF did not
  converge). Measured accuracy on LiH rocksalt/LDA against the
  PySCF-validated multi-k GDF route: **+17.7 mHa at (2,2,2)**, +25.4 at
  (3,3,3), grid-converged - the ordinary GAPW augmentation-accuracy
  class, **not** sub-mHa parity. **Validated only near the equilibrium
  lattice constant:** scaling LiH's lattice by 1.6x / 2.2x (with the
  structure held) leaves the SCF unconverged, 206 / 271 mHa from GDF,
  which converges on all three. Cohesive energies do **not** inherit
  the error cancellation that makes molecular atomization work either
  - the free atoms agree to sub-mHa between routes, so the solid's
  residual passes straight through (E_coh differs from GDF by
  -17.2 mHa on LiH). Tracked as
  MULTIK-GAPW-BLOCH-FAILS-ON-EXPANDED-CELLS. A fold-charge guard still
  covers the paths the port does not reach. For quantitative
  compact-crystal work use ``jk_method="gdf"``.

  .. note::

     The expanded-cell measurements in this bullet (206 / 271 mHa,
     SCF unconverged) are **pre-fix**. A convergence fix for
     GitLab **#65** has since landed (``4494885ef``, ``b4f545b52``,
     both ancestors of the v0.15.138 release SHA), and the
     verification runs that would replace these numbers are in
     flight as of 2026-08-22. Until that evidence is fetched and
     independently verified, treat the figures above as describing
     the pre-fix behaviour rather than current behaviour, and do
     not quote them as the post-fix result. The ``jk_method="gdf"``
     recommendation stands either way.
- **Range-separated hybrids** - HSE06 / ωB97X / CAM-B3LYP on the
  GPW/GAPW path (``periodic_gapw_range_sep``).
- **Orbital Transformation (OT) solver** - direct energy minimisation
  avoiding diagonalisation (``periodic_gapw_ot``).
- **MPI z-slab grid overlay** - distributed-memory Hartree-J build via
  ``GpwJBuilder(mpi_aware=True)`` (z-slab domain decomposition over
  ``mpi4py``). The multi-rank build was broken for every world size > 1
  through v0.12.0 (the v0.12 post-release audit caught it; fixed and
  pinned by a simulated-rank parity suite,
  ``tests/test_mpi_gpw_parity.py``). Remains experimental until
  validated on a real ``mpirun`` multi-rank run. The serial GPW route
  is unaffected - world size 1 always degenerated to the correct serial
  build.

For tight all-electron crystals today, use the
[GDF](density_fitting.md) or [BIPOLE](bipole.md) routes.

**Known limitation - boundary-atom density quadrature (∫ρ ≠ N):**

The periodic-AO image sum in ``collocate_density_on_grid`` sums
nearest-neighbour images (image_radius=1, 27 cells) to fold escaping
Gaussian tails back into the cell. This is exact for compact (valence)
bases in cells with side ≥ ~6 bohr, but **diffuse bases in very tight
cells may still leak** - the quadrature error shows as a small
under-integration of the density (∫ρ < N). GPW cubes, ELF maps, and
post-SCF grid-based properties derived from the collocated density are
affected. The SCF energy and wavefunction are **not** affected (they
use the rigorous lattice-summed AO integrals). Users with diffuse
bases in tight cells can increase ``image_radius`` via the
``_AO_IMAGE_RADIUS`` module constant (not yet a public option).
```

## Fock diagonalization

Gamma GPW/GAPW RHF/RKS/UHF/UKS and range-separated GAPW route their SCF
Fock solves through the [shared operator contract](solver_framework.md).
Each unrestricted spin has its own operator metadata and diagnostics. The
solver checks Hermiticity, eigenpair residuals and metric orthogonality before
returning coefficients; it rejects inconsistent input or failed solves.
Overlap construction, occupations, smearing, DIIS and physical support gates
remain owned by the grid method.

Closed-shell GPW retains `use_davidson` and `davidson_min_dim`. Davidson
recycles the preceding subspace without mutating its input options. Other
Gamma grid routes retain their dense solver. Multi-k GPW RKS/UKS/ROKS,
GAPW RKS and the fixed-density band helper also use shared dense solves.
AO matrices carry X/S into the solve; projected DIIS matrices remain in their
orthogonal frame, with residuals labelled accordingly. The driver maps
coefficients back to AO coordinates and assigns global occupations. Orbital
normalization never includes k-point weights.

Gamma GPW ROHF/ROKS now shares the molecular Roothaan solver, including
Guest-Saunders block canonicalization. Its occupation and level-shift
policies remain in that loop. Initial-guess diagonalizations and orbital
transformation remain separate migration work. The
shared boundary does not extend physical support or certify experimental
calculations; compact GAPW's density-fold refusal remains in force.

## What GPW is

The Gaussian + plane-wave (GPW) method (Lippert & Hutter 1997,
*Mol. Phys.* 92, 477) replaces the Gaussian-Gaussian four-index
Coulomb integral that dominates the periodic Fock build's J
piece with three smaller steps on a smooth real-space grid:

1. **Density collocation.** Evaluate
   `ρ(r) = Σ_μν D_μν · χ_μ(r) · χ_ν(r)` on a uniform 3D grid
   spanning the unit cell, pairs the Gaussian basis with a
   plane-wave-style sampling.
2. **FFT-Poisson.** Solve `V(r) = ∫ 1/|r − r'| ρ(r') dr'` on
   the same grid via a forward / multiply-by-`4π/G²` /
   backward FFT cycle. The G=0 component is pinned to zero
   (the cell carries an implicit uniform neutralising
   background).
3. **AO-basis projection.** Compute
   `J_μν = ∫ χ_μ(r) χ_ν(r) V(r) dr` by quadrature on the
   same grid.

The cost is `O(n_basis² · N_grid)` for the collocation +
projection plus `O(N_grid · log N_grid)` for the FFT. For a
large enough basis the FFT becomes irrelevant and only the AO
work matters; that's the regime where GPW wins against direct-
space periodic Coulomb. CP2K is the canonical implementation
the v0.10.x track is reproducing.

GPW assumes **pseudopotentials** (the density is "smooth", no
sharp 1s cores). For all-electron periodic systems, the GAPW
route adds a per-atom radial-grid augmentation (the "GA" in GAPW)
that restores the hard cores. Multi-k and the other extensions listed
above remain experimental.

```{warning}
**GPW's Gamma-only (`n_k==1`) branch has an open, unfixed convergence
bug** (`GPW-GAMMA-SINGLEK-NONCONVERGENT`, `agentic-loop/bug-claims.md`):
on a batch of compact-basis systems with comfortably positive-definite
overlap metrics, all 8 Gamma-only cases failed to converge in 120
iterations, landing 3.7-19 eV above the matched-mesh reference -- while
the *same* systems on the multi-k branch converge correctly. The
Gamma-only path uses a floored $S^{-1/2}$ orthogonaliser instead of the
canonical multi-k one, a different code path now shown broken
independently of overlap-truncation/R-set issues. Do not treat a
Gamma-point smoke test as evidence a GPW route "works" until this
lands on the canonical orthogonaliser or the floored branch is fixed.
Measured on the 2026-09-29 development build: H₂/STO-3G in an 8-bohr cube
converges at Gamma through `run_periodic_job(jk_method="gpw")` to
-1.123531 Ha; the compact-basis batch above has not been re-measured here.
```

## Theory

GPW ({vibeqc-cite}`lippert_hutter_gpw_1997`; the Quickstep implementation is
{vibeqc-cite}`vandevondele_hutter_quickstep_2005`) keeps the Gaussian
orbitals for the wavefunction and moves the density onto a uniform grid
spanning the cell, $N_g = N_xN_yN_z$ points $\mathbf r_g$ with volume element
$\Delta V = \Omega/N_g$. From the density matrix $\mathbf D$,

$$
\rho(\mathbf r_g) = \sum_{\mu\nu} D_{\mu\nu}\,\chi_\mu(\mathbf r_g)\,\chi_\nu(\mathbf r_g),
\qquad
\tilde\rho(\mathbf G) = \frac{1}{N_g}\sum_g \rho(\mathbf r_g)\,e^{-i\mathbf G\cdot\mathbf r_g},
$$

where the AOs are summed over the neighbouring image cells so that a tail
leaving the cell re-enters on the other side. The Hartree potential is one
Poisson solve in reciprocal space, $\nabla^2 V_H = -4\pi\rho$,

$$
\tilde V_H(\mathbf G) = \frac{4\pi}{|\mathbf G|^2}\,\tilde\rho(\mathbf G)
\quad(\mathbf G\neq 0),\qquad \tilde V_H(0) = 0,
\qquad
E_H = 2\pi\Omega\sum_{\mathbf G\neq 0}\frac{|\tilde\rho(\mathbf G)|^2}{|\mathbf G|^2},
$$

with the $\mathbf G = 0$ term dropped, which is the uniform neutralising
background of a periodic cell (the same Madelung convention as the rest of
vibe-qc; see the section on the Madelung shift below). The Coulomb matrix and
the exchange-correlation terms are then grid quadratures,

$$
J_{\mu\nu} = \Delta V\sum_g \chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g)\,V_H(\mathbf r_g),
\qquad
E_{xc} = \Delta V\sum_g \rho(\mathbf r_g)\,\varepsilon_{xc}\big(\rho,\nabla\rho\big)(\mathbf r_g),
\qquad
V^{xc}_{\mu\nu} = \Delta V\sum_g \chi_\mu\chi_\nu\,v_{xc}(\mathbf r_g)
+ \text{(gradient terms for GGA / meta-GGA)},
$$

and the grid is sized by the plane-wave cutoff, $|\mathbf G|^2/2 \le
E_{\mathrm{cut}}$. The cost is $\mathcal O(N_g\log N_g)$ for the FFTs plus
$\mathcal O(N^2 N_g)$ for collocation and projection, so the method wins as
the cell grows and the basis stays smooth. The approximation that enters is
the grid: the density must be resolvable at $E_{\mathrm{cut}}$, which is why
GPW is normally paired with pseudopotentials and why an all-electron core on
the grid is prohibitively expensive.

GAPW ({vibeqc-cite}`lippert_hutter_parrinello_gapw_1999`,
{vibeqc-cite}`krack_parrinello_allelectron_2000`) keeps the all-electron
problem by splitting the density into a smooth part that lives on the grid
and one-centre corrections that live on atom-centred radial grids, the same
dual-grid construction that Blochl's projector augmented wave method uses
({vibeqc-cite}`blochl_paw_1994`):

$$
\rho(\mathbf r) = \tilde\rho(\mathbf r) + \sum_A\big[\rho_A(\mathbf r) - \tilde\rho_A(\mathbf r)\big],
\qquad
\rho_A = \sum_{\mu\nu\in A} D_{\mu\nu}\,\chi_\mu\chi_\nu,
\qquad
\tilde\rho_A = \sum_{\mu\nu\in A} D_{\mu\nu}\,\tilde\chi_\mu\tilde\chi_\nu .
$$

Here $\tilde\rho$ is the density built from the soft (small-exponent) part of
every basis function and collocated on the plane-wave grid; $\rho_A$ is the
hard one-centre density of atom $A$ from all its primitives, expanded on a
radial times Lebedev grid inside the atomic sphere; and $\tilde\rho_A$ is the
soft one-centre density of the same atom, so that $\rho_A - \tilde\rho_A$
vanishes outside the sphere and the three pieces sum to the all-electron
density. Because the Hartree and exchange-correlation functionals are
evaluated piecewise,

$$
V_H[\rho] = V_H[\tilde\rho] + \sum_A\big(V_H[\rho_A] - V_H[\tilde\rho_A]\big),
\qquad
E_{xc}[\rho] = E_{xc}[\tilde\rho] + \sum_A\big(E_{xc}[\rho_A] - E_{xc}[\tilde\rho_A]\big),
$$

the smooth term reuses the GPW Poisson solve and the atomic terms are local
one-dimensional radial Poisson solves and atomic-grid quadratures with no
FFT; the atomic spheres must not overlap for the split to be exact, which
vibe-qc enforces with a partition of unity across spheres and an own-atom
compensating charge whose multipoles match those of $\rho_A - \tilde\rho_A$
(the `one_centre=` modes below choose between the fitted block construction
and the fit-free analytic one-centre ERIs). The approximations are the soft
/ hard split of the basis, the atomic-grid quadrature, and the compensation
charge; the method helps wherever an all-electron answer is wanted on a large
smooth cell without pseudopotential files, and it fails, today, on compact
crystals and multi-k cells, where the augmentation residual is tens of
millihartree and GDF is the route to use.

## What today's implementation does

The M2-full stack ships these surfaces, all on the public
`vibeqc.*` namespace:

* **`vibeqc.PlaneWaveGrid`**, frozen dataclass holding
  a uniform 3D grid (`nx`, `ny`, `nz`, lattice, cutoff). Lazy
  fractional + Cartesian coordinate generators; reciprocal-
  lattice vectors with FFT wrap-around convention.
* **`vibeqc.make_grid`**, construct a grid sized from a
  plane-wave cutoff (Hartree). Rounds each extent up to a
  `{2ᵃ · 3ᵇ · 5ᶜ}` value so FFTW3 hits its fast paths.
* **`vibeqc.collocate_density_on_grid`**, Gaussian
  density on the grid via the existing C++ `evaluate_ao`
  accessor. Integrates to `tr(D · S)` (the total number of
  electrons for a closed-shell density) to finite-grid
  quadrature precision.
* **`vibeqc.collocate_point_charges_on_grid`**, point
  charges as normalised Gaussians of width σ on the grid,
  summing periodic images out to `image_shells` neighbours.
  Used by M1e's Madelung-constant Hartree test.
* **`vibeqc.GpwJBuilder`**, caches the AO-on-grid
  values across SCF iterations; `build_J(D)` returns the
  Hartree-J matrix.
* **`vibeqc.evaluate_gpw_energy`**, single-point
  periodic-energy breakdown at a given density. Returns
  `vibeqc.GpwEnergyBreakdown` with the per-term
  decomposition (`e_kinetic`, `e_nuclear_attraction`,
  `e_hartree`, `e_hf_exchange`, `e_nuclear_repulsion`,
  `e_total`).
* **`vibeqc.run_periodic_rhf_gpw`**, minimal iterative
  closed-shell RHF SCF using GPW for J and the existing
  molecular kernel for K. Returns `vibeqc.GpwScfResult`
  with energy, density, MO coefficients + energies, and the
  per-term breakdown.

### Minimal worked example, He STO-3G

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

# He atom at the centre of a 16-bohr cubic cell.
L = 16.0
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
basis = vq.BasisSet(
    vq.Molecule(list(system.unit_cell), 0, 1),
    "sto-3g",
)

# Suppress the experimental warning during opt-in use.
warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_rhf_gpw(
    system, basis,
    cutoff_ha=300.0,        # ~600 Ry; CP2K's production default
    max_iter=50,
    conv_tol_energy=1e-9,
    conv_tol_density=1e-7,
)

print(f"converged: {result.converged}")
print(f"n_iter: {result.n_iter}")
print(f"E_total: {result.energy:.10f} Ha")
print()
print("breakdown:")
for field in ("e_kinetic", "e_nuclear_attraction",
              "e_hartree", "e_hf_exchange",
              "e_nuclear_repulsion"):
    print(f"  {field}: {getattr(result.breakdown, field):+.6f}")
```

The standalone diagnostic `evaluate_gpw_energy(system, basis,
D, grid=grid)` does the same per-term decomposition at a
density `D` you supply, useful when you want to evaluate the
GPW energy at the converged molecular density for comparison.

### V_ne convention switch (M3a vs. M3b)

Both periodic V_ne conventions ship and are physically equivalent
on neutral cells. The default `"ewald"` is the M3a path that uses
`compute_nuclear_lattice_ewald` (G = 0 dropped, v_bg jellium shift
applied), production-tested, no extra parameter. The opt-in
`"smeared_erfc"` is the CP2K-style decomposition

```
V_ne = V_ne_long  +  V_ne_short  +  c_α · S
     = (FFT-Poisson on Σ_a Z_a · (α/√π)³ exp(-α²|r-R_a|²))
     + ⟨χ_μ | -Σ_a Z_a · erfc(α · r_a) / r_a | χ_ν⟩
     + π · (Σ_a Z_a) / (α² · V_cell) · S    # PW-DFT α-correction
```

routing every Hartree-class quantity through the same Poisson
solver, the convention the GAPW augmentation work plugs into.

```python
# M3a default (Ewald):
r_ewald = vq.run_periodic_rhf_gpw(system, basis, cutoff_ha=300.0)

# M3b CP2K-style erfc/erf split:
r_smear = vq.run_periodic_rhf_gpw(
    system, basis, cutoff_ha=300.0,
    v_ne_convention="smeared_erfc",
    smearing_alpha=2.0,             # bohr⁻¹; None ⇒ auto from grid
)

# Both converge to the same SCF total within < 1 mHa on neutral
# cells (parity tests pin this at < 50 µHa on He STO-3G across
# α ∈ {1.0, 1.5, 2.0} and at < 1 µHa on H2 STO-3G at α = 2.0).
```

### Atomic radial × Lebedev grids

The per-atom quadrature grids for the GAPW hard/soft Hartree
decomposition are available in ``vibeqc.periodic_gapw_atomic_grid``.
These grids are now **wired into the GAPW SCF**, the augmentation
uses radial Poisson solves + Lebedev integration inside each atomic
sphere, with a multipole-compensating ρ₀ on the smooth FFT grid.

What the module provides:

* `AtomicRadialGrid`, a frozen dataclass that owns a 1D
  Mura-Knowles radial mesh (`r`, `w_r`, with the `r²` Jacobian
  pre-folded into `w_r`) and a Lebedev-Laikov angular mesh
  (`angular_xyz`, `w_a`, with `Σ_l w_l = 4π`). Helpers:
  `cartesian_points()` returns the product grid as
  `(n_radial, n_angular, 3)`; `combined_weights()` returns the
  `(n_radial, n_angular)` `w_k · w_l`; `integrate(values)`
  evaluates `Σ_kl values[k,l] · w_k · w_l` directly.
* `AtomicRadialGrid.build(centre_bohr, n_radial=..., alpha=...,
  lebedev_order=...)` and `AtomicRadialGrid.from_element(
  centre_bohr, Z, ...)` are the two constructors. `from_element`
  pulls `α` from `default_alpha_for_element(Z)` and emits a
  `GAPWExperimentalWarning` (suppressible via `quiet=True`).
* `default_alpha_for_element(Z)`, per-element radial scale
  picked to hug the nucleus and the diffuse tail equally; the
  CP2K `grid_atom` family's choice.
* `lebedev_supported_orders()`, the Lebedev orders SciPy
  exposes (`scipy.integrate.lebedev_rule`); the grid constructor
  validates against this list.

```python
import numpy as np
import vibeqc as vq

grid_O = vq.periodic_gapw_atomic_grid.AtomicRadialGrid.from_element(
    centre_bohr=np.array([0.0, 0.0, 0.0]),
    Z=8,
    n_radial=80,
    lebedev_order=29,
    quiet=True,
)
# grid_O.cartesian_points() has shape (n_radial, n_angular, 3)
# grid_O.combined_weights() has shape (n_radial, n_angular)
# Σ_k w_r[k] = ∫_0^∞ r² · 1 dr  truncated to the mesh range
# Σ_l w_a[l] = 4π
```

The radial rule is Mura-Knowles (1996), `r(x) = -α · ln(1 - x³)`
on uniform `x ∈ (0, 1)`; numerically equivalent to the
log-transformed Gauss-Chebyshev rule the CP2K source uses for
the smooth integrands GAPW sees. The implementation is
paraphrased from the published paper (no GPL source reproduced).

```{note}
The grids are **unpartitioned** - no Becke weighting. GAPW only
needs them inside an augmentation sphere around each nucleus,
so the molecular Becke grid (`vibeqc._vibeqc_core.build_grid`)
is the wrong tool here. The augmentation work in M3b-aug-B
applies the soft-cut at the augmentation radius on top of
these raw atomic shells.
```

See `docs/design_periodic_gapw.md` § Q2 / D1 / D2 for the
audit reference against CP2K's `grid_atom.F`.

## The Madelung-shift convention

GPW's FFT-Poisson kernel runs on a periodically-repeated cell.
Even a single neutral atom in a finite box (He, H₂, …) is
treated as one cell of an infinite periodic lattice, the
electrostatic energy you get out is the lattice's energy, not
the isolated-molecule energy. The difference is the **cubic-
cell Madelung shift**: each point charge interacts with its own
periodic images plus the uniform neutralising background. The
shift per unit charge in a cubic cell of side `L` is

```
ΔE_Madelung = q² · ξ_NaCl / (2 · L)
```

with `ξ_NaCl = 2.837297…` the Madelung constant for the simple
cubic lattice.

At the **M2 J-only level**, this shift falls between the
periodic GPW Hartree and the molecular Hartree as a clean
single-term correction. The `vibeqc.evaluate_gpw_energy`
result's `e_hartree` field carries the electronic-side
Madelung shift on a charged cell.

At the **M3a SCF level**, `V_ne` is built in the same Ewald
gauge as the Hartree-J (`compute_nuclear_lattice_dispatch` with
`CoulombMethod.EWALD_3D`, G = 0 reciprocal mode dropped, v_bg
jellium shift applied), so the three pieces all sit in the
same gauge. On a **neutral cell** the cross-term cancels both
the electronic and nuclear self-image shifts exactly:

```
E_periodic_M3a ≈ E_molecular  (vacuum-padded neutral cell;
                              < 1 mHa residual from finite
                              cell + finite grid)
```

For He STO-3G in a 16-bohr cube, periodic E_HF and molecular
E_HF now agree to better than 2·10⁻⁵ Ha (verified by
`tests/test_periodic_gapw_j.py::
test_rhf_gpw_he_recovers_molecular_in_vacuum_padded_cell`).

Before the M3a lift, `V_ne` was at the molecular limit while
the Hartree-J and nuclear-repulsion both lived in the Ewald
gauge, leaving a known "2× per-charge Madelung shift" in the
SCF total energy (0.71 Ha for He STO-3G at L = 16). That is
gone: any residual cell-size dependence is now the physical
finite-cell / finite-grid contribution, not a gauge artefact.

At the **M2 J-only level** (`evaluate_gpw_energy(...).e_hartree`
in isolation), the periodic-vs-molecular Hartree difference
still carries the electronic self-image shift `q² · ξ_NaCl /
(2L)`. That is the FFT-Poisson convention of the J build
itself and is the M2 internal-validation anchor that pairs
with M1e's neutral-cell sub-µHa NaCl test. The `e_hartree`
field thus continues to read out a Madelung-shifted Hartree
energy on charged cells; the total-energy cancellation kicks
in only when V_ne and E_nn are summed in.

## What the runner reports if you ask for it via `run_periodic_job`

As of the M3b dispatch wiring, the runner serialises GPW SCF
output through the standard `.out` / `.system` / `.molden`
artefacts. Usage:

```python
result = vq.run_periodic_job(
    system, basis,
    method="RHF",
    jk_method="gpw",          # M3b: now reachable
    output="my_gpw_calc",
)
print(f"E_total = {result.energy:.10f} Ha")
```

The runner wraps the standalone ``GpwScfResult`` with the
:class:`vibeqc.periodic_gapw_runner_adapter.GpwRunnerResult`
adapter so the downstream output-rendering / manifest plumbing
sees the same shape the GDF / BIPOLE routes produce
(``energy`` / ``e_electronic`` / ``e_nuclear`` / ``e_coulomb`` /
``e_hf_exchange`` / ``n_iter`` / ``converged`` / ``mo_coeffs`` /
``mo_energies`` / ``density`` / ``fock`` / ``overlap`` /
``occupations`` / ``scf_trace``).

Constraints (current): **RHF / RKS / UHF / UKS**. Γ-only for
RHF and UHF through the runner; multi-k RKS and UKS (pure DFT) available
via ``kpoints``. Neutral cells (``RKS`` / ``UKS`` require a
``functional=``).
``jk_method="auto"`` continues to pick GDF (closed-shell) /
BIPOLE (open-shell); GPW/GAPW are opt-in experimental routes.

## Tutorials

The walk-throughs below mirror GPAW's tutorial style: a short
motivation, a single self-contained Python block, and a pointer
at the API reference earlier on this page so you can jump back
to the underlying knob when you want to tweak it. Every example
uses H₂ in a vacuum-padded cube, small enough to converge in
seconds on a laptop, big enough that the periodic answer matches
the molecular reference. Drop each block into a `.py` file next
to your checkout and run with the project's `.venv/bin/python`.

### Tutorial 1: Your first GPW calculation

The fastest way to convince yourself the plane-wave-on-Gaussian
machinery is wired up is to run a single GPW SCF and compare
its total energy against a textbook number. H₂ at 1.4 bohr in a
15-bohr cube is a good first system: it is closed-shell, the
vacuum-padded cell carries no Madelung shift (see *The
Madelung-shift convention*), and the molecular RHF/STO-3G
energy is the well-known −1.1167 Ha.

What you're seeing in the block below: we build a 3D
`PeriodicSystem` with a cubic lattice, attach an STO-3G basis
to its two H atoms, and call
[`vibeqc.run_periodic_rhf_gpw`](#what-todays-implementation-does).
The function returns a `GpwScfResult` carrying the converged
energy, iteration count, and the per-term breakdown documented
in the API section above. The `e_hartree` and
`e_nuclear_attraction` fields both live in the Ewald gauge at
this milestone, so the total is directly comparable against the
molecular RHF number.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L = 15.0
d = 1.4  # H-H bond, bohr
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_rhf_gpw(
    system, basis,
    cutoff_ha=300.0,
    max_iter=50,
    conv_tol_energy=1e-9,
)
print(f"E_HF (periodic GPW) = {result.energy:.6f} Ha")
print(f"molecular reference  ≈ -1.1167 Ha (H2 / STO-3G)")
```

**Next:** [Tutorial 2: Periodic DFT on a hydrogen molecule](#tutorial-2-periodic-dft-on-a-hydrogen-molecule).

### Tutorial 2: Periodic DFT on a hydrogen molecule

GPW's J kernel is method-agnostic: once the Hartree-J is in the
Ewald gauge, swapping the molecular RHF on top for a pure-DFT
RKS only changes the K piece. The M3d milestone wired libxc
into the GPW SCF, so you reach RKS-LDA on the periodic side
with a single keyword swap.

Why does the periodic LDA number land on the molecular LDA
number? Because the H₂ cell is **vacuum-padded and neutral**.
With 12+ bohr of empty space between periodic images, the
nearest H₂ replica sits beyond the practical range of the
Hartree kernel; on a neutral cell the Ewald V_ne / E_Madelung
cross-term cancels the electronic self-image shift (see *The
Madelung-shift convention*); and LDA's exchange-correlation is
fully local so it can't pick up any inter-image contribution at
all. The takeaway: if you want a molecular DFT answer out of
the periodic stack, pad the cell.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_rks_gpw(
    system, basis,
    functional="lda",
    cutoff_ha=300.0,
    conv_tol_energy=1e-8,
)
print(f"E_LDA (periodic GPW) = {result.energy:.6f} Ha")
print("(matches molecular RKS-LDA on a vacuum-padded cell)")
```

**Next:** [Tutorial 3: Multi-k Brillouin-zone sampling](#tutorial-3-multi-k-brillouin-zone-sampling).

### Tutorial 3: Multi-k Brillouin-zone sampling

For metallic and dispersive systems the SCF total energy
depends on a Brillouin-zone integral over the occupied bands,
and Γ-only sampling is not enough. The M3e milestone added
[`vibeqc.run_periodic_rks_gpw_multi_k`](#what-todays-implementation-does)
- a closed-shell, pure-DFT route that Bloch-sums the core
operators (T, S, V_ne) on a Monkhorst-Pack mesh while sharing a
single FFT-grid density (and therefore a single Hartree-J and
V_xc) across all k-points.

On a compact cell, the per-k Bloch AO values used to construct that density
and project the local potential use a fixed memory policy. Small jobs retain
all k-point tables only while their total fits under 1 GiB; larger jobs use
bounded 64 MiB point batches. This avoids an unbounded `n_k` multiplier while
preserving the iteration-saving cache for modest grids. Meta-GGA spectral AO
derivatives couple the whole FFT grid, so a streamed job keeps one k-point's
table at a time. The memory preflight uses the same limits. Batching changes
only contraction order: the calculated density, projected potential, and total
energy use the same GPW equations. **Both spin branches follow this policy.**
When the all-k cache fits, its construction shares real Gaussian image values
across k-points in small point and shell blocks. This also covers skew cells
and arbitrary point coordinates. The Bloch phases, image list and its summation
order are unchanged, and every geometry constructs its own tables. Larger jobs
keep the existing streaming path.
The open-shell driver (`run_periodic_uks_gpw_multi_k`, and the ROKS route that
shares it) kept the dense all-k cache unconditionally until issue #89 and was
estimated at 702.7 GiB on the P16 NiO / DFT+U cell (232 AOs, 128^3 grid, 64 k)
against 1.19 GiB for the same cell closed shell; it now reads the same cache
target and streams above it, for 1.55 GiB.

On the H₂ / STO-3G vacuum-padded cube the answer is independent
of `kmesh`, the bands are flat (no dispersion) because each
H₂ image is electronically isolated, so every k-point sees the
same Fock spectrum. That's the diagnostic: if your kmesh sweep
shows < µHa drift, you're correctly in the molecular limit and
Γ is sufficient. The moment you shrink the cell or pack atoms
densely enough that bands disperse, the kmesh result will
diverge from Γ and you'll need to converge it (start at
(2,2,2), double until ΔE < your threshold).

The block below runs the same H₂ cell at a (2,2,2) Monkhorst-
Pack mesh and prints the energy alongside Γ-only for
comparison.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

r_gamma = vq.run_periodic_rks_gpw(
    system, basis, functional="lda", cutoff_ha=300.0,
)
kmesh = vq.monkhorst_pack(system, [2, 2, 2])
r_mk = vq.run_periodic_rks_gpw_multi_k(
    system, basis, kmesh,
    functional="lda",
    cutoff_ha=300.0,
)
print(f"E_LDA (Γ-only)    = {r_gamma.energy:.8f} Ha")
print(f"E_LDA (2x2x2 MP)  = {r_mk.energy:.8f} Ha")
print(f"|ΔE|              = {abs(r_gamma.energy - r_mk.energy):.2e} Ha")
```

```{note}
**Lattice-sum truncation on compact crystals.** At every k the driver
needs the Bloch overlap

```
S(k) = sum_R exp(i k . R) S(R),   S(R)_mn = <chi_m(0) | chi_n(R)>
```

which is the Gram matrix of the Bloch orbitals and is therefore positive
definite only when the sum over lattice vectors R is complete. Earlier
releases cut that sum at a flat 15 bohr -- far inside the tail of a
diffuse molecular basis on a dense lattice -- which left S(k) indefinite
(measured min eigenvalue over a Gamma-centred (4,4,4) mesh at def2-SVP:
LiF -1.51, MgO -0.20, Si -0.16) and quietly destroyed multi-k band
structures (negative MgO/LiF gaps and a 0.0017 eV silicon gap in the
2026-08-02 validation wave).

The multi-k GPW drivers now size the sum from the basis with
`vibeqc.lattice_screening.bloch_overlap_cutoff_bohr` and apply the SAME
R-set to every real-space lattice-summed operator entering the
eigenproblem (overlap, kinetic, and the Ewald V_ne). The uniformity is
essential, not cosmetic: completing the overlap sum while V_ne kept its
own truncation over-bound LiF/def2-SVP by 18 Ha while the band structure
looked perfect, because the neglected far-field one-electron terms are
proportional to the overlap tail and cancel only when H(k) and S(k) drop
the same tail (Pisani & Dovesi 1980, Sec. 4). An indefinite metric now
fails closed with a diagnostic instead of being absorbed by the
linear-dependence projection.

The cost tracks the basis diffuseness: def2-SVP on rocksalt LiF needs a
52-bohr sum, while a solid-state basis (`pob-tzvp-rev2`) stays near the
25-bohr floor. `vq.eigs_preflight` diagonalises S(k) across your mesh
before you commit to an SCF. See
[`linear_dependence.md`](linear_dependence.md).
```

**Next:** [Tutorial 4: Choosing the V_ne convention](#tutorial-4-choosing-the-v_ne-convention-ewald-vs-cp2k-style).

### Tutorial 4: Choosing the V_ne convention (Ewald vs CP2K-style)

GPW's nuclear-attraction operator V_ne is gauge-invariant on a
neutral cell, two conventions ship and both give the same SCF
total to better than 1 mHa. The default
`v_ne_convention="ewald"` builds V_ne in the same Ewald gauge as
the Hartree-J directly. The opt-in
`v_ne_convention="smeared_erfc"` is the CP2K-style decomposition
that splits V_ne into a smooth long-range piece (routed through
the FFT-Poisson solver as a Gaussian-smeared nuclear charge
density) plus a short-range erfc/r correction in AO basis (see
*V_ne convention switch (M3a vs. M3b)* on this page).

For routine vacuum-padded RHF / RKS jobs the default Ewald
convention is the right pick: one less parameter, no
`smearing_alpha`, and no spurious Madelung shift to track. The
smeared-erfc convention earns its keep when you head toward
**GAPW augmentation**: it routes every Hartree-class quantity
through the same FFT-Poisson solver, which is the convention the
per-atom radial-grid hard/soft cancellation will exploit. If
you want to start instrumenting against the augmentation work,
develop with `smeared_erfc`; if you just want an energy out,
stick with Ewald.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

r_ewald = vq.run_periodic_rhf_gpw(
    system, basis, cutoff_ha=300.0, v_ne_convention="ewald",
)
r_smear = vq.run_periodic_rhf_gpw(
    system, basis, cutoff_ha=300.0,
    v_ne_convention="smeared_erfc", smearing_alpha=2.0,
)
print(f"E (Ewald V_ne)        = {r_ewald.energy:.8f} Ha")
print(f"E (smeared-erfc V_ne) = {r_smear.energy:.8f} Ha")
print(f"|ΔE|                  = {abs(r_ewald.energy - r_smear.energy):.2e} Ha")
```

**Next:** [Tutorial 5: Visualising the density](#tutorial-5-visualising-the-density).

### Tutorial 5: Visualising the density

A periodic GPW job collocates the density on the FFT grid as
its very first SCF step, so dropping it to disk is essentially
free. Driving the calculation through the runner, via
[`run_periodic_job`](#what-the-runner-reports-if-you-ask-for-it-via-run_periodic_job)
with `write_density=True`, emits the standard `.out` /
`.system` / `.molden` triple alongside an `.xsf` volumetric
file. Open the `.xsf` in
[VESTA](https://jp-minerals.org/vesta/en/) or any volumetric-
file viewer (vibe-view, VMD, Avogadro) to inspect the bonding
density.

The block below runs the same H₂ vacuum-padded cube under
RKS-LDA and asks for the density dump. The runner picks the GPW
J/K backend (`jk_method="gpw"`) and routes the rest of the
periodic-SCF plumbing through the normal `run_periodic_job`
flow, so the output layout matches every other periodic job's.
Future milestones may add a dedicated CUBE writer (e.g. a
`write_cube_density_periodic` helper) and a QVF
`volume.density` section for the GAPW runner; for now the XSF
file is the one to open.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_job(
    system, basis,
    method="RKS",
    functional="lda",
    jk_method="gpw",
    write_density=True,
    output="h2_gpw_density",
)
print(f"E_LDA = {result.energy:.8f} Ha")
print("open h2_gpw_density.xsf in VESTA or vibe-view")
```

**Next:** [Tutorial 6: Saving and resuming a GPW calculation](#tutorial-6-saving-and-resuming-a-gpw-calculation).

### Tutorial 6: Saving and resuming a GPW calculation

A converged GPW SCF carries everything a downstream post-SCF
analysis needs, density, MO coefficients + energies, the
per-term breakdown, and the lattice / basis / atoms that built
it. The v0.12-prep restart format ships three helpers that
round-trip the lot through a single compressed `.npz` archive:
`vibeqc.save_gpw_result(path, result, basis, system)` writes
the archive, `vibeqc.load_gpw_result(path)` reads it back into
a plain `dict`, and `vibeqc.describe_gpw_result(path)` returns
a short human-readable summary string for quick inspection.

This is useful any time you'd like to **avoid re-running the
SCF**, running a band path against a converged density, running
a DOS sweep over different smearing widths, sharing a result
with a collaborator, or just resuming an analysis after a long
SCF without having to wait the SCF out again. Both
`GpwScfResult` (Γ-only) and `GpwMultiKScfResult` (multi-k)
serialise through the same pair of calls.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_rks_gpw(
    system, basis, functional="lda", cutoff_ha=300.0,
)
written = vq.save_gpw_result("h2_gpw.npz", result, basis, system)
print(f"wrote {written} ({written.stat().st_size} bytes)")
print(vq.describe_gpw_result(written))

# Resume later in a fresh process:
data = vq.load_gpw_result(written)
print(f"resumed E = {data['energy']:.8f} Ha")
print(f"resumed density shape = {data['density'].shape}")
```

**Next:** [Tutorial 7: Computing a density of states](#tutorial-7-computing-a-density-of-states).

### Tutorial 7: Computing a density of states

The Γ-only spectrum is one diagonalisation; a multi-k spectrum
is a stack of diagonalisations weighted by the Monkhorst-Pack
weights. Either way, what you want next is usually a
**Gaussian-broadened density of states**, and a quick HOMO /
LUMO read-out to anchor the energy axis. The v0.12-prep
post-SCF helpers in `vibeqc.periodic_gapw_postscf` (re-exported
on the top-level `vibeqc.*` namespace) cover both:
`gaussian_dos(mo_energies, ...)` is the raw broadening kernel,
`compute_dos_from_result(result, sigma=..., ...)` pulls
eigenvalues straight off a `GpwScfResult` or
`GpwMultiKScfResult` and weights k-points for you, and
`compute_homo_lumo(result)` returns the `(e_homo, e_lumo, gap)`
triple from the converged spectrum.

If you also need a band path between high-symmetry k-points,
`band_path_eigenvalues(result, basis, system, k_path, ...)`
diagonalises `F(k) C = ε S(k) C` at each k on a user-supplied
path, useful for plotting band structures once you have a
converged density.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_rks_gpw(
    system, basis, functional="lda", cutoff_ha=300.0,
)
e_homo, e_lumo, gap = vq.compute_homo_lumo(result, system=system)
print(f"HOMO = {e_homo:+.4f} Ha   LUMO = {e_lumo:+.4f} Ha   gap = {gap:.4f} Ha")

e_grid, dos = vq.compute_dos_from_result(
    result, system=system, sigma=0.01, e_range=(e_homo - 0.5, e_lumo + 0.5),
)
print(f"DOS array shape = {dos.shape};  max DOS = {dos.max():.2f} states/Ha")
```

**Next:** [Tutorial 8: Geometry optimisation with ASE](#tutorial-8-geometry-optimisation-with-ase).

### Tutorial 8: Geometry optimisation with ASE

The v0.12 R1 cycle added ASE-compatible numerical forces to
the [`VibeqcGPW`](#what-todays-implementation-does) Calculator
- central-difference around each Cartesian coordinate, exposed
through the standard `atoms.get_forces()` interface. That is
the only piece ASE's geometry-optimisation drivers need: drop
the calculator on an `ase.Atoms` object and any optimiser in
`ase.optimize` will relax it.

H₂ at HF/STO-3G has a well-known equilibrium bond length near
1.346 bohr (~0.712 Å). The block below starts from a stretched
1.6-bohr bond and runs BFGS for a handful of steps; the bond
contracts toward the optimum and the final length lands inside
the noise floor of the numerical-force step size. The same
recipe drives any closed-shell GPW geometry: pick a basis +
cutoff that converge your system's total energy, attach the
calculator, and hand the `Atoms` off to BFGS / LBFGS / FIRE.

```python
import warnings
import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.optimize import BFGS
import vibeqc as vq

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

L_bohr = 12.0
L_ang = L_bohr * Bohr
r0_bohr = 1.6  # stretched start
p1 = np.array([L_bohr / 2 - r0_bohr / 2, L_bohr / 2, L_bohr / 2]) * Bohr
p2 = np.array([L_bohr / 2 + r0_bohr / 2, L_bohr / 2, L_bohr / 2]) * Bohr
atoms = Atoms("H2", positions=[p1, p2], pbc=True)
atoms.set_cell(np.eye(3) * L_ang)
atoms.calc = vq.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)

r_initial = atoms.get_distance(0, 1)
BFGS(atoms, logfile=None).run(fmax=0.05, steps=10)
r_final = atoms.get_distance(0, 1)
print(f"initial bond = {r_initial / Bohr:.3f} bohr")
print(f"final   bond = {r_final / Bohr:.3f} bohr (HF/STO-3G opt ≈ 1.346 bohr)")
```

**Next:** [Tutorial 9: Metallic systems with smearing](#tutorial-9-metallic-systems-with-smearing).

### Tutorial 9: Metallic systems with smearing

Metallic and small-gap systems oscillate under sharp Aufbau
occupations because the Fermi level passes through a band of
states the SCF can't decide between. The fix is **Fermi-Dirac
smearing**: spread the occupations smoothly across an energy
window of width kT, converge the resulting variationally
correct free energy `F = E − T·S`, then extrapolate to T=0 if
needed. The v0.12 R1 cycle wires this onto the multi-k GPW
entry [`run_periodic_rks_gpw_multi_k`](#what-todays-implementation-does)
via the `smearing_temperature` kwarg (in Hartree).

The block below runs the H₂ vacuum-padded cube at a tiny
finite kT just to show what the surface looks like, on a
gapped molecular reference the entropy stays small and `F` sits
just below `E` by `T·S`. On a real metal you'd typically pick
kT in the 0.001-0.01 Ha range (≈ 300-3000 K), monitor the
`smearing_entropy` term to make sure it isn't ballooning, and
report `free_energy` (not `energy`) as the converged total.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L, d = 15.0, 1.4
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2 - d / 2, L / 2, L / 2]),
    core.Atom(1, [L / 2 + d / 2, L / 2, L / 2]),
]
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

kmesh = vq.monkhorst_pack(system, [2, 2, 2])
result = vq.run_periodic_rks_gpw_multi_k(
    system, basis, kmesh,
    functional="lda", cutoff_ha=300.0,
    smearing_temperature=0.005,
)
print(f"E (internal)   = {result.energy:.8f} Ha")
print(f"F = E - T·S    = {result.free_energy:.8f} Ha")
print(f"smearing T     = {result.smearing_temperature:.4f} Ha")
print(f"entropy S      = {result.smearing_entropy:.4e}")
```

**Next:** [Tutorial 10: Periodic UHF on a hydrogen atom](#tutorial-10-periodic-uhf-on-a-hydrogen-atom).

### Tutorial 10: Periodic UHF on a hydrogen atom

The v0.12 R2 cycle wires the open-shell GPW route:
`run_periodic_uhf_gpw` (pure HF) and `run_periodic_uks_gpw`
(DFT with a libxc functional) both take per-spin occupations
`n_alpha` / `n_beta`, run a Γ-only joint α + β DIIS-accelerated
SCF, and return a `GpwUhfScfResult` / `GpwUksScfResult` with
per-spin densities, MO coefficients, and MO energies. They
reduce numerically to the closed-shell `run_periodic_rhf_gpw` /
`run_periodic_rks_gpw` results when `n_alpha == n_beta`, that
parity is the regression test pin.

The block below runs a single H atom in a vacuum-padded box at
`(n_alpha, n_beta) = (1, 0)`, the spin-polarised reference for
the 1s electron. The reported energy is the standard UHF total;
`spin_squared` (`<S²>`) tracks the spin contamination and lands
near 0.75 (the doublet exact value) when the open-shell SCF
converges cleanly.

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

L = 12.0
system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
# Single H atom = 1 electron → multiplicity 2 (doublet); mult 1 would raise
# "n_electrons and multiplicity are inconsistent".
basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 2), "sto-3g")

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

result = vq.run_periodic_uhf_gpw(
    system, basis,
    n_alpha=1, n_beta=0,
    cutoff_ha=200.0,
)
print(f"E (UHF/STO-3G) = {result.energy:.8f} Ha")
print(f"n_alpha        = {result.n_alpha}")
print(f"n_beta         = {result.n_beta}")
print(f"converged      = {result.converged} ({result.n_iter} iter)")
```

**Next:** [Tutorial 11: Geometry optimisation with analytic forces](#tutorial-11-geometry-optimisation-with-analytic-forces).

### Tutorial 11: Geometry optimisation with analytic forces

The v0.12 R2 cycle replaces Tutorial 8's central-difference
numerical-force loop with an **analytic-where-possible** gradient
driver (`vibeqc.periodic_gapw_gradient.compute_gradient_gpw`):
the Pulay overlap-Lagrangian uses the C++ periodic gradient
primitive, and the combined Hellmann-Feynman piece (T + V_ne +
Hartree-on-grid + XC-on-grid + E_nn at fixed density) is a
central-difference on the *energy evaluator only*, no extra SCF
iteration per displacement. Cost per atom per axis: two
fixed-density energy evaluations. On STO-3G H₂ this is ~6× faster
than the 6N-SCF path from Tutorial 8, and the dominant bond-axis force
agrees with the full-SCF finite-difference reference to better than 1%
relative error.

`VibeqcGPW.calculate(properties=("energy", "forces"))` now
defaults to the analytic path on the Γ-only route, so ASE's
optimisers pick it up automatically. The block below repeats
the Tutorial 8 H₂ BFGS relaxation with the analytic gradient
selected by default; pass `use_numerical_forces=True` to fall
back to the legacy 6N-SCF path for cross-checks.

```python
import warnings
import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.optimize import BFGS
import vibeqc as vq

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

L_bohr = 12.0
L_ang = L_bohr * Bohr
r0_bohr = 1.6  # stretched start
p1 = np.array([L_bohr / 2 - r0_bohr / 2, L_bohr / 2, L_bohr / 2]) * Bohr
p2 = np.array([L_bohr / 2 + r0_bohr / 2, L_bohr / 2, L_bohr / 2]) * Bohr
atoms = Atoms("H2", positions=[p1, p2], pbc=True)
atoms.set_cell(np.eye(3) * L_ang)
atoms.calc = vq.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)

r_initial = atoms.get_distance(0, 1)
BFGS(atoms, logfile=None).run(fmax=0.05, steps=10)
r_final = atoms.get_distance(0, 1)
print(f"initial bond = {r_initial / Bohr:.3f} bohr")
print(f"final   bond = {r_final / Bohr:.3f} bohr (HF/STO-3G opt ≈ 1.346 bohr)")
```

## One-centre augmentation modes (`one_centre=`)

The Γ GAPW drivers accept four `one_centre=` selections. The public default,
**`"auto"`**, is method-aware and uses an explicit physical-regime
declaration:

* RHF and UHF select **`"analytic"`** when the standalone driver receives
  `molecular_limit=True` (or `run_periodic_job` receives
  `gapw_molecular_limit=True`). Without that declaration they fail closed
  before allocating the FFT grid or dense ERIs.
* RKS and UKS select **`"block"`**. Their multi-atom XC augmentation has no
  validated variationally safe analytic replacement yet.

`"auto"` is a dispatch policy, not a claim that one augmentation is valid in
every physical regime. The declaration is intentional: volume, vacuum-axis,
and cross-cell AO-overlap heuristics cannot distinguish every isolated
molecule from an arbitrary dense supercell. The three concrete constructions
are:

* **`"block"` (explicit legacy mode)**: one-centre densities from the atom's
  own AO block of D, windowed radial integrals, and monopole compensation.
  This remains the automatic DFT construction and is retained for
  reproducibility. It is reasonable for single atoms, H2, and some
  near-spherical environments, but is **Ha-scale wrong for HF on a
  core-bearing atom with multiple bonded neighbours**: H2O/STO-3G is
  `-8.859694 Ha` below the molecular reference. The block restriction drops
  off-centre AO tails and bond-density terms from the atom's one-centre
  density, contrary to the total-density partition required by the GAPW
  formulation.
* **`"analytic"`**: the fit-free analytic-ERI augmentation: the
  exact hard/soft ERI difference plus charge-conserving
  s-compensators; every term is an exact contraction or a
  bounded-kernel quadrature, and the Fock is the exact energy
  derivative (`1/2 tr(D J) = E_H` to 1e-12). **HF: unrestricted
  within the molecular-limit Γ envelope** (isolated cluster in a
  box) and basis-robust: H2O +11.499 mHa (STO-3G) / +7.4 (6-31G) /
  +26.0 (def2-SVP) vs molecular references; LiH -7; atoms
  +0.4..+24. **DFT: single-atom cells only** (exact on-centre XC
  densities; Ne LDA +15..+23 mHa across bases); multi-atom DFT fails
  closed because fitted off-centre XC densities soften with basis
  richness (LiH LDA -10.6 -> -105 mHa from STO-3G to def2-SVP).
* **`"projector"`**: the fitted projector construction
  (research/diagnostic): single-atom cells, HF only.

The analytic HF validation envelope is an isolated molecule or atom in a
vacuum-padded box at a single Γ point. It is **not** a dense-crystal or
multi-k validation claim. For molecular DFT, compact all-electron crystals,
and anything else outside that molecular-limit envelope, use
`jk_method="gdf"` or `"bipole"`. Explicit `one_centre="analytic"` remains
available as the expert opt-in spelling of the same molecular-limit
assertion; it is never selected automatically without the declaration.

Direct GAPW gradient and stress entry points fail closed when the converged
result resolved to `one_centre="analytic"`; differentiating the old block
functional would not be the derivative of the promoted energy. For RHF/UHF,
the high-level `run_periodic_job(jk_method="gapw")` route also rejects
`optimize=True`, `optimize_cell=True`, and `hessian=True`, so those workflows
cannot silently switch away from the analytic one-centre energy surface. When
forces are requested through `VibeqcGAPW`, ASE uses numerical central
differences of the full SCF energy in analytic mode and the established
analytic gradient in block mode. Block-mode RKS/UKS retains its existing
derivative plumbing.

For a multi-k `VibeqcGAPW` calculation, `gapw_kwargs` may still set the radial
and angular grid controls, but `one_centre` and `molecular_limit` are rejected
rather than silently ignored. The supported multi-k pure-DFT RKS builder uses
its fixed block construction. GAPW HF phonons likewise require an explicit
`gapw_kwargs={"one_centre": "block"}` declaration; DFT phonons default to
block. Analytic and projector one-centre phonons fail closed because no
matching derivative is implemented. These derivative rules do not change the
established block-mode RKS/UKS derivative path.

## What's coming

The implemented route includes RHF / RKS / UHF / UKS at Γ, multi-k pure-DFT,
DIIS, DOS / band helpers, restart I/O, the ``VibeqcGPW`` and ``VibeqcGAPW``
ASE calculators, the C++ SCF host (``run_rhf_scf_gpw_cpp``), and the
``PeriodicJKMethod.GPW`` runner wiring. GPW retains its finite-difference
optimization and Hessian workflows; GAPW HF derivatives follow the
fail-closed and ASE rules above. The open roadmap items are the all-electron
and hybrid extensions:

1. **GAPW augmentation hardening.** The per-atom augmentation ships with
   method-aware defaults (fit-free analytic RHF/UHF in declared
   molecular-limit cells, and block RKS/UKS at Γ).
   Open pieces are a variationally safe XC formulation for fitted off-centre
   content, analytic derivatives, compact-crystal validation, and multi-k
   GAPW. The H2O-class HF Hartree default defect and open-shell analytic
   wiring are complete.
2. **CP2K parity sweep** via the
   ``examples/regression/core/runner_cp2k.py`` subprocess runner, gated on
   the augmentation above.
3. **Metallic GAPW**, Fermi-Dirac smearing already ships on the multi-k
   GPW route (``smearing_temperature=``); pairing it with the augmentation
   for genuine all-electron metals is the open piece.
4. **ACE-decorated K** for hybrids, and **PAW pseudopotentials** via an
   on-demand ``paw_fetcher.py``.

The full milestone map is in
[`docs/design_periodic_gapw.md`](../design_periodic_gapw.md).

## See also

* [`docs/design_periodic_gapw.md`](../design_periodic_gapw.md)
  - the M0 design doc with the full milestone list, the
  five resolved asks, and the architecture sketch.
* `vibeqc.periodic_gapw_grid`, the
  `PlaneWaveGrid` dataclass + collocation helpers.
* `vibeqc.periodic_gapw_j`, the J builder, the energy
  evaluator, and the SCF entry.
* [`density_fitting.md`](density_fitting.md), the GDF route
  (Γ-only Gaussian J via density fitting), the closest
  Gaussian-only sibling.
* [`bipole.md`](bipole.md), the BIPOLE route (multi-k
  Gaussian-only CRYSTAL-gauge Ewald J-split).
* Lippert, Hutter, Parrinello, *Mol. Phys.* 92, 477 (1997)
  - the foundational GPW paper.
* Hutter, Iannuzzi, Schiffmann, VandeVondele, *WIREs Comp.
  Mol. Sci.* 4, 15 (2014), CP2K + GPW from the implementers.

## Evaluated Gamma HF and KS states

The Python Gamma GPW RHF and libxc RKS drivers, including their Dudarev
DFT+U paths, expose `result.fock_build`.
It owns the physical Fock, evaluated density, energy components, metric frame
and numerical-source identity. DIIS operators remain separate solver states.
The final energy uses the requested nuclear-potential convention, including
`smeared_erfc`, and any density-independent dispersion correction.

At an iteration limit, the returned density and physical Fock still belong to
one evaluation. `next_density` is a separate look-ahead SCF step and
`density_fixed_point_residual` measures its Frobenius distance. Returned
orbitals and occupations reconstruct the evaluated density, including mixed
densities from damping; orbital energies are physical-Fock Rayleigh quotients.
They need not be canonical eigenpairs before convergence. Finite-temperature
entropy uses those evaluated occupations. QVF labels these density orbitals
`natural`, with the energy interpretation recorded in the run fields. The runner preserves this state and
its identity in output. External XC, UKS+U and augmented GAPW paths retain
their existing result conventions pending their own migrations.

The retained home-cell exchange approximation is qualified in its padded
molecular limit. A compact-cell energy-derivative inconsistency in that
existing kernel is tracked in issue #534; this migration does not establish
compact-cell variational or bulk reference parity.

For libxc RKS, the same evaluated build owns both the explicit XC energy and
its AO potential. LDA, GGA, admitted meta-GGA and global-hybrid functionals
retain their existing grid, density-gradient, kinetic-energy-density and
stiffness policies. Pure functionals skip libint exchange; PATOM still uses a
separate full-HF seed. The source receipt includes the functional components,
exchange fraction, collocation table, reciprocal grid and meta-GGA policy.
Changing a captured numerical source requires a fresh provider and frame.
Range-separated GPW functionals retain their existing refusal; external XC
state migration is deferred.

For RHF and libxc RKS with DFT+U, the same build includes the explicit spin-summed
Dudarev energy as a separate `dft_plus_u` correction and its potential in the
physical Fock or KS matrix. It retains the existing AO projectors: each per-spin
occupation is the active block of `S (D/2) S`, and the AO potential is
`S V_U S`. The source identity includes the active AO groups, effective
Hubbard parameters and metric frame. PATOM remains a full-HF seed without U.
The result and runner retain `e_dft_plus_u`; it is already included in the
total energy and must not be added again. This changes state ownership, not
the projector definition or its physical validation envelope.

Gamma GPW UHF, including UHF+U, also returns one evaluated `fock_build`, with ordered
alpha/beta density and physical-Fock channels. Their shared Coulomb term uses
the total density; libint exchange retains its per-spin convention. PATOM
and coupled DIIS use the common spin-labelled solver boundary. PATOM retains
its canonical overlap cutoff of `1e-7` in a separate seed frame. Returned
`occupations_alpha` and `occupations_beta` reconstruct each evaluated density,
including damping and zero-iteration restarts. The physical Rayleigh energies,
source/frame identities and coupled fixed-point residual survive the runner
and QVF output. The numerical source keeps the existing grid, nuclear gauge,
image sum and zero mode; this adds no MANTA, BIPOLE or symmetry admission.

For UHF+U, each Hubbard occupation uses its own `S D_spin S` active AO
block. The two Dudarev energies enter one explicit `dft_plus_u` correction;
the separate `S V_spin S` potentials enter the physical alpha/beta Focks.
The Hubbard source binds the metric, ordered active AO groups and effective
U parameters. PATOM remains bare UHF without U, with its existing retained
space. Zero U recovers plain UHF numerically; an empty spin contributes zero
Hubbard energy but may retain a nonzero Hubbard potential.

Plain Gamma GPW UKS with libxc also returns one evaluated `fock_build`.
Both physical spin Focks share the total-density Hartree potential and a
joint spin XC evaluation. The XC energy is an explicit electronic component,
not half a trace of its potential. GGA cross-spin gradients and meta-GGA
kinetic-energy terms retain the existing grid projection and regularisation.
Pure functionals skip exact exchange during SCF; global hybrids retain
libint per-spin exchange. PATOM still uses a separate full-HF seed.

Converged, capped, damped and zero-iteration UKS results preserve their
actual spin densities, density-defining occupations and physical Rayleigh
energies through runner and QVF output. A stopped state's orbitals need not
diagonalise its physical Focks. Functional, grid, collocation and meta-GGA
policy changes require a new provider. This migration does not add UKS+U,
range-separated, augmented GAPW, multi-k or symmetry capabilities.
