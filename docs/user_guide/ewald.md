# Ewald summation in periodic systems

In 3D periodic solids the Coulomb series $\sum_{g,A,B} Z_A Z_B / |R_{AB} + g|$
is **conditionally convergent**: different truncation shapes give
different answers. The standard fix is Ewald summation, which splits
each $1/r$ term into a short-range part (evaluated in real space, where
it converges exponentially) plus a long-range part (evaluated in
reciprocal space, where the Gaussian damping makes the sum converge
fast too). vibe-qc's full 3D Ewald infrastructure, nuclear repulsion,
nuclear attraction $V(g)$, and the electron-repulsion Coulomb
$J(\mu\nu)$, is shipped through a single Coulomb-method dispatcher;
this page tells you how to use it.

## The user-facing recommendation

For any 3D-periodic SCF, **set the Coulomb method to ``EWALD_3D`` and
call the dispatcher**:

```python
import vibeqc as vq

opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D

# Multi-k:
result = vq.run_rhf_periodic_scf(system, basis, kmesh, opts,
                                 omega=0.5, spacing_bohr=0.3)
# Γ-only:
result = vq.run_rhf_periodic_gamma_scf(system, basis, opts,
                                       omega=0.5, spacing_bohr=0.3)
```

The dispatcher routes on ``opts.lattice_opts.coulomb_method`` to the
appropriate backend:

| ``CoulombMethod`` | Routed backend (multi-k) | Routed backend (Γ-only) | Status |
|---|---|---|---|
| ``DIRECT_TRUNCATED`` | ``run_rhf_periodic`` | ``run_rhf_periodic_gamma`` | Default. Half-summed direct truncation, exact in the molecular-limit regime (large unit cell, cutoff excludes all non-zero cells). Conditionally convergent in 3D bulk. |
| ``EWALD_3D`` | ``run_rhf_periodic_multi_k_ewald3d`` | ``run_rhf_periodic_gamma_ewald3d`` | Full Gaussian-charge Ewald, unconditionally convergent. The native FFT long-range solver now supports skew cells via a full reciprocal-lattice metric; the legacy EWALD_3D SCF route remains a parity/debug path while FFTDF/GDF production work is in progress. |
| ``SLAB_EWALD_2D`` | ``run_rhf_periodic_multi_k_ewald3d`` / ``run_rks_periodic_multi_k_ewald3d`` / ``run_uks_periodic_multi_k_ewald3d`` with slab options | ``run_rhf_periodic_gamma_ewald2d`` / ``run_rks_periodic_gamma_ewald2d`` / ``run_uks_periodic_gamma_ewald2d`` | Rigorous 2D slab Ewald for RHF/RKS/UKS SCF. Analytic gradients still raise cleanly. |
| ``NEUTRALIZED_1D`` | - | - | Not yet wired into SCF dispatch. |

Nuclear repulsion routes through the same enum: when ``EWALD_3D`` or
``SLAB_EWALD_2D`` is selected, ``nuclear_repulsion_per_cell`` automatically
uses the matching Ewald block, so the electronic and nuclear contributions
stay self-consistent.

```{note}
``DIRECT_TRUNCATED`` is the default ``CoulombMethod`` on a
``LatticeSumOptions`` you construct yourself, but you rarely set this enum
by hand. Through the runner, ``jk_method="auto"`` selects the Coulomb gauge
for you, and on a ``dim == 2`` slab it always selects ``SLAB_EWALD_2D``.
Every bulk J/K route raises on a slab.
```

### 2D slabs (``SLAB_EWALD_2D``)

For a system periodic in two dimensions and finite along the third (a
slab / layer, ``dim == 2`` with lattice columns 0,1 in-plane and column 2
an auto-synthesized, non-physical normal vector), the correct Coulomb sum
is the **rigorous 2D Ewald** of
Parry (Surf. Sci. **49**, 433 (1975)) and de Leeuw & Perram (Mol. Phys.
**37**, 1313 (1979)): the *true* 2D lattice sum, with no vacuum-gap
parameter and no spurious inter-image dipole field along the normal. It
adds a reciprocal-space term with an ``erfc(α z)`` structure and a special
``g = 0`` term that has no 3D analogue.

The point-charge energy primitive is exposed as
``vibeqc.ewald_2d_point_charge_energy(lattice, positions_cart, charges,
options)``. It requires a **charge-neutral** cell: a net-charged
2D-periodic plane has a divergent Madelung energy with no finite jellium
regularisation, so a non-neutral cell raises ``ValueError``.

The self-consistent **RHF/RKS/UKS** paths use the same slab electrostatics.
Gamma-only RHF is wired through ``run_rhf_periodic_gamma_scf``:

```python
opts = vq.PeriodicRHFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
opts.lattice_opts.slab_ewald_alpha = 0.4

result = vq.run_rhf_periodic_gamma_scf(system_2d, basis, opts)
```

Closed-shell DFT is wired through the KS dispatcher:

```python
opts = vq.PeriodicKSOptions()
opts.functional = "PBE"
opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
opts.lattice_opts.slab_ewald_alpha = 0.4

result = vq.run_rks_periodic_gamma_scf(system_2d, basis, opts)
```

Open-shell DFT uses the direct Gamma slab driver:

```python
opts = vq.PeriodicKSOptions()
opts.functional = "PBE"
opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
opts.lattice_opts.slab_ewald_alpha = 0.4

result = vq.run_uks_periodic_gamma_ewald2d(system_2d, basis, opts)
```

The split parameter ``slab_ewald_alpha`` is shared by the bare nuclear block,
the electron-nuclear attraction, and the slab Hartree ``J``. A positive
dispatcher ``omega`` argument is accepted as an alias for this Gamma slab
split on the RHF/RKS dispatchers and on the direct UKS slab driver. Dense
k-mesh RHF and RKS slabs route through ``run_rhf_periodic_scf`` and
``run_rks_periodic_scf`` with a ``SLAB_EWALD_2D`` lattice option; dense
k-mesh UKS slabs use ``run_uks_periodic_multi_k_ewald3d`` with the same
option object. Slab analytic gradients remain a separate workstream. For 1D
wire systems, ``NEUTRALIZED_1D`` is not implemented yet.

### KS and UKS routes

The pattern above is the **HF** dispatcher. Closed-shell KS uses the same
dispatcher shape. Open-shell UKS exposes the matching Gamma and multi-k Ewald
drivers directly:

| Driver family | Dispatcher (multi-k) | Dispatcher (Γ-only) | Phase |
|---|---|---|---|
| Closed-shell HF | ``run_rhf_periodic_scf`` | ``run_rhf_periodic_gamma_scf`` | 12e-c-4 |
| Closed-shell KS | ``run_rks_periodic_scf`` | ``run_rks_periodic_gamma_scf`` | 15c-1, 15c-2 |
| Open-shell UKS | ``run_uks_periodic_multi_k_ewald3d`` | ``run_uks_periodic_gamma_ewald3d`` / ``run_uks_periodic_gamma_ewald2d`` | 15c-3 |

Each accepts the same Ewald keyword arguments (``omega``, ``grid_shape``,
``origin``, ``spacing_bohr``, ``linear_dep_threshold``). Pick the options class
that matches your method (``PeriodicSCFOptions`` / ``PeriodicRHFOptions`` for
HF, ``PeriodicKSOptions`` for RKS/UKS), set the Coulomb method, and call the
matching route above.

The Ewald-specific keyword arguments to the dispatcher (``omega``,
``grid_shape``, ``origin``, ``spacing_bohr``, ``linear_dep_threshold``)
are forwarded to the ``EWALD_3D`` backend and ignored on the
``DIRECT_TRUNCATED`` path. For ``SLAB_EWALD_2D`` Gamma RHF/RKS/UKS, a positive
``omega`` is treated as the slab split parameter alias described above; the
FFT grid arguments are unused. Defaults: for 3D Ewald ``omega`` is
CRYSTAL's ``2.8 / V^(1/3)`` bounded below by ``sqrt(-ln tol) / cutoff_bohr``
with ``tol = opts.ewald_tolerance`` (1e-12), see "Choosing cutoffs" below;
``slab_ewald_alpha=0.4`` for 2D slab Ewald; and ``spacing_bohr=0.3`` for the
3D reciprocal-space FFT grid.

## A complete worked example

```python
import vibeqc as vq

# Conventional cubic LiH rocksalt - 8 atoms per cell, lattice 4.084 Å.
a_ang = 4.084
a = a_ang / 0.529177210903
import numpy as np
lat = a * np.eye(3)

# Off-FFT-grid shift: atoms exactly at the corner of the box inflate
# the long-range residual ~100x; a small (0.05 bohr) shift fixes it.
shift = 0.05
unit_cell = []
for z in (3, 1):                                       # 3 = Li, 1 = H
    base = 0.5 if z == 1 else 0.0
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        x = (fx + base) * a + shift
        y = (fy + base) * a + shift
        z_ = (fz + base) * a + shift
        unit_cell.append(vq.Atom(z, [x, y, z_]))

system = vq.PeriodicSystem(dim=3, lattice=lat, unit_cell=unit_cell)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# Γ-only SCF with Ewald
opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr = 12.0          # short-range AO cutoff
opts.lattice_opts.nuclear_cutoff_bohr = 25.0   # nuclear-Ewald real cutoff

result = vq.run_rhf_periodic_gamma_scf(
    system, basis, opts,
    omega=0.5,
    spacing_bohr=0.4,
)

print(f"Converged:    {result.converged}")
print(f"Total energy: {result.energy:.8f} Ha per cell")
print(f"SCF iters:    {result.n_iter}")
```

For multi-k, build a :class:`vibeqc.KPoints` (or, for back-compat,
a raw ``BlochKMesh``) and call
``run_rhf_periodic_scf(system, basis, kpoints, opts)``. The result
mirrors the Γ-only structure plus per-k orbital coefficients.
``KPoints`` exposes Monkhorst-Pack, Γ-centered, IBZ-reduced (via
spglib), HPKOT band paths (via seekpath), explicit user lists, and
density-based auto-meshes (KPPRA / kspacing / VASP ``Auto``), see
[user_guide/k_points.md](k_points.md) for the full reference.

## What "ω-invariance" means and why we test it

Ewald has one free parameter, the splitting $\omega$ that decides
how much of $1/r$ is sent to real-space vs reciprocal-space. The
*total* energy must be **independent of $\omega$** to whatever the
real- and reciprocal-space cutoffs allow. This is the primary
correctness test of any Ewald implementation. vibe-qc's bulk-crystal
benchmark suite verifies it explicitly across H₂ / LiH / MgO / Ne
fixtures: scanning $\omega \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ recovers
the same total energy to within $10^{-8}$ Ha when the cutoffs are
adequate. If your run is $\omega$-sensitive at the $10^{-4}$ level,
the cutoffs are too tight, increase ``cutoff_bohr`` and
``nuclear_cutoff_bohr`` and re-test.

## Choosing cutoffs and the FT mesh

The ``EWALD_3D`` path has three convergence knobs:

- **``opts.lattice_opts.cutoff_bohr``**, real-space cutoff for AO
  pair / ERI integrals and the AO-pair-FT Bloch cell list. Default
  15 bohr; can usually drop to 10-12 for valence-only bases like
  sto-3g, pob-DZVP.
- **``opts.lattice_opts.nuclear_cutoff_bohr``**, real-space cutoff
  for the nuclear-Ewald real-space term. Default 25 bohr; auto-derived
  from ``omega`` if you pass ``omega`` explicitly.
- **``VIBEQC_J_EWALD3D_KE``**, kinetic-energy cutoff (Hartree) for
  the analytical AO-pair-FT Hartree J mesh. Default 200 Ha. It applies
  to the opt-in ``use_exact_ft_j=True`` cross-check oracle on
  ``run_pbc_bipole_rks`` and to the bounded analytic-FT J contraction
  on 3D true-multi-k pure-RKS GDF jobs with ``use_compcell=False``.
  The production BIPOLE pure-RKS route uses the padded SR+LR erfc
  composition and does not read it. Increase the cutoff for
  benchmark-quality all-electron tight-core cells; the BIPOLE oracle
  warns with the required estimate when under-covered.

The nuclear cutoff is applied to the complete pair distance
$|R_A-R_B+g|$. The translation search therefore includes every lattice
vector that a basis displacement can move inside that physical sphere. This
keeps dense cells cutoff-converged and makes the result invariant when an atom
is represented in an equivalent neighboring unit cell; users do not need to
inflate the cutoff merely to cover the unit-cell basis span.

The historical FFT-Poisson collocation backend is still available for
diagnostics with ``VIBEQC_J_EWALD3D_BACKEND=grid``. In that mode only,
the dispatcher ``spacing_bohr=`` kwarg controls the real-space FFT grid.

The splitting parameter and the real-space cutoff are two halves of one
truncation contract: the real-space sum is cut at ``cutoff_bohr``, so
erfc$(\omega \cdot R_{\text{cut}})$ must already be negligible there
(Ewald 1921 trades the two series against each other through exactly this
parameter). The nuclear Ewald derives its $\alpha$ from its own cutoff so
that erfc$(\alpha \cdot R_{\text{cut}}) \approx 10^{-12}$. The
``EWALD_3D`` SCF drivers default ``omega`` to CRYSTAL's volume rule
``2.8 / V^(1/3)`` for term-by-term parity, but that value falls with the
cell volume while ``cutoff_bohr`` stays put, so since GitLab #651 it is
bounded below by ``sqrt(-ln tol) / cutoff_bohr`` (0.35 bohr⁻¹ at the 15-bohr
default and ``tol = opts.ewald_tolerance = 1e-12``). Cells up to an 8-bohr
cube keep the CRYSTAL value; larger cells get the bound. The exposed sum is
the erfc arm of the corrected exchange split (``exchange_exxdiv='ewald'``);
the analytic-FT Hartree ``J`` is $\omega$-invariant and the nuclear terms
size their own $\alpha$. Measured on H₂ in a 16-bohr cube at a (2,1,1) mesh,
the unbounded ``omega = 0.175`` at 15 bohr was 4.3e-5 Ha short of the 30-bohr
value, the bounded 0.35 within 1.1e-8 Ha. The reciprocal envelope of the
long-range exchange follows the same $\omega$ in every ``EWALD_3D`` driver
(the ROHF and ROKS multi-k drivers used CRYSTAL's volume-only $K_{\max}$
before #651); a total that changes when ``omega`` changes at adequate
cutoffs is a bug to report, not a parameter to tune.
Pass ``omega`` explicitly (or ``opts.ewald_omega``) to override; the total
energy must not depend on the choice once both cutoffs are adequate.

The BIPOLE route carries the same contract on its corrected exchange split
(``use_exchange_ewald_split``, the default gauge on every 3D BIPOLE run
with exact exchange). Its erfc-screened exchange ``K_SR`` is summed over
the Fock output cells, $|g| \le$ ``cutoff_bohr``, an AO-overlap length
that does not grow with the cell, while CRYSTAL's ``2.8 / V^(1/3)`` shrinks
with it, so past a 7.8-bohr cube at the 12-bohr cutoff the image exchange
beyond the cutoff was silently dropped (GitLab #674): H₂/STO-3G in a
30-bohr box at (1,1,1) read -1.1170669756 Ha against the split-invariant
-1.1170858285 Ha, 1.9e-5 Ha short, and the one-electron H atom 9.0e-6 Ha
short. Since #674 the BIPOLE drivers bound that default below by
``sqrt(-ln tol) / cutoff_bohr`` with ``tol = ewald_precision`` (1e-8, the
tolerance BIPOLE already uses for its erfc image balls: 0.358 bohr⁻¹ at a
12-bohr cutoff, 0.286 at the 15-bohr default), and scale CRYSTAL's
reciprocal envelope with the alpha in use so that $K_{\max}/\alpha$ keeps
CRYSTAL's value of 8.51 (the reciprocal Gaussian resolved to 1.4e-8) in the
energy and in every analytic-gradient term alike. A run whose alpha is
CRYSTAL's is unchanged bit for bit; the legacy CRYSTAL-gauge scaffold
(``use_exchange_ewald_split=False``) and pure functionals keep CRYSTAL's
alpha, since they have no erfc exchange arm to converge. On the 30-bohr
box the corrected BIPOLE RHF now gives -1.1170858289 Ha at the default, at
``ewald_omega=0.2`` and at 0.5 alike, within 1e-9 Ha of the EWALD_3D value;
an explicit ``ewald_omega=0.0933`` (CRYSTAL's value there) reproduces the
pre-fix shortfall, which is what the regression test pins. Ewald's
potential is invariant to the screening parameter only when both series
are summed to convergence, and CRYSTAL's ``2.8 / V^(1/3)`` is the cost
optimum for fully summed series (Saunders, Freyria-Fava, Dovesi and
Roetti, Mol. Phys. 77, 629 (1992), Property 11 and Eq. (55)), not an
accuracy requirement: the bound restores the first condition where the
BIPOLE real-space sum is not carried to convergence.

The SCF result records `ewald_precision`. BIPOLE analytic gradients carry this
value through the nuclear image selection, reciprocal terms and orbital
response, so nondefault precision differentiates the same finite sums as
the SCF. Older results without the field use the historical `1e-8` default.
The precision is an image-domain control, not a guaranteed absolute force
error. The reciprocal envelope still follows the resolved Ewald alpha;
independent cutoff and force-convergence checks remain necessary.

```{warning}
The legacy multi-k RKS `EWALD_3D` path stores periodic AO values for
every real-space density cell, and GGA functionals store three AO
gradient arrays as well. Conventional-cell, pob-TZVP-scale ionic
crystals can therefore hit a tens-of-GiB memory cliff before the first
SCF iteration. The driver now raises `MemoryError` with an allocation
estimate instead of letting the operating system kill the process.
Use the GDF periodic route (`run_krhf_periodic_gdf` /
`run_krks_periodic_gdf`) where available, or coarsen the DFT grid only
for small debug runs. The current multi-k GDF spike uses PySCF's
k-point-symmetry classes for weighted IBZ meshes when available, with a
full-mesh fallback for older PySCF / shifted meshes. A fully native GDF
loop remains the long-term performance follow-up.
```

## Low-level Ewald primitives

The dispatcher is the right entry point for SCF work. For
quasi-classical lattice-electrostatics calculations (Madelung
constants, point-charge potentials, charge-only models), use the
primitives directly:

```python
from vibeqc import EwaldOptions, ewald_point_charge_energy
import numpy as np

# NaCl-like rocksalt with ±1 charges
a = 5.6 / 0.529177210903                       # 5.6 Å in bohr
lattice = 0.5 * a * np.array([[0, 1, 1],
                              [1, 0, 1],
                              [1, 1, 0]], dtype=float).T
positions = np.column_stack([[0, 0, 0],
                             [0.5 * a, 0, 0]])
charges = np.array([+1.0, -1.0])

opts = EwaldOptions()
opts.real_cutoff_bohr = 30.0          # auto-derive α and recip cutoff
energy = ewald_point_charge_energy(lattice, positions, charges, opts)
print(f"NaCl Madelung / pair: {energy:.8f} Ha   (expected -0.330275485)")
```

Other low-level entry points:
- ``vq.ewald_point_charge_potential``, Madelung *potential* at given probe points.
- ``vq.ewald_nuclear_potential`` / ``vq.ewald_nuclear_repulsion``, nuclei-only versions.
- ``vq.compute_nuclear_lattice_ewald``, nuclear-attraction matrix elements via Ewald.
- ``vq.build_j_ewald_3d``, direct J(μν) Coulomb build at a single density (used inside the SCF dispatcher).

These all operate on the same ``EwaldOptions`` / ``LatticeSumOptions``
machinery; the dispatcher just wires them together at SCF time.

### Madelung constant for arbitrary Bravais cells

`vibeqc.madelung.madelung_constant_for_cell(system, *, precision=1e-12,
eta=None)` returns the Ewald-Madelung constant ξ for any Bravais
lattice via a proper real-space + reciprocal-space Ewald self-energy
sum with a neutralising background. It is η-invariant at convergence
and bit-exact to PySCF's `pbc.tools.madelung` to 6+ significant
figures on cubic, primitive FCC, and intermediate cells.

```python
import vibeqc as vq
from vibeqc.madelung import madelung_constant_for_cell

xi = madelung_constant_for_cell(system)
```

Prior to commit `d7f4b3bd`, this routine used the cubic Wigner
shortcut ``α_M ≈ 2.837297 / V^(1/3)``, which is correct for
conventional cubic cells but off by ~1.77 % for primitive FCC
cells. The 1.77 % gap propagated through the `exxdiv='ewald'`
K-shift to ~16 mHa of SCF bias on LiH primitive FCC. The proper
Ewald sum closes that gauge defect and was a key contributor to
the v0.9 PBC GDF chemical-accuracy milestone on LiH (see
[`periodic_methods.md`](periodic_methods.md) § 3.2.1).

## Theory

### Splitting the Coulomb sum

For a 3D-periodic distribution of point charges $\{Z_A\}$ at positions
$\{R_A\}$ in a unit cell with translations $\{g\}$, the energy

$$
E_{\text{nn}} = \tfrac{1}{2} \sum_{A, B} {\sum_{g}}' \frac{Z_A Z_B}{|R_A - R_B + g|}
$$

(primed sum excludes the $A = B, g = 0$ term) is conditionally
convergent. Ewald's 1921 trick is to split each $1/r$ as

$$
\frac{1}{r} = \frac{\operatorname{erfc}(\alpha r)}{r} + \frac{\operatorname{erf}(\alpha r)}{r}
$$

The first term decays exponentially in real space; the second is smooth
enough to Fourier-transform and sum in reciprocal space. The total
energy under the **tin-foil + jellium** convention becomes

$$
E_{\text{nn}} = E_{\text{real}} + E_{\text{recip}} + E_{\text{self}} + E_{\text{bg}}
$$

with

- $E_{\text{real}} = \tfrac{1}{2} \sum_{A,B} {\sum_{g}}' Z_A Z_B \cdot \operatorname{erfc}(\alpha \, r) / r$, exponentially convergent real-space sum.
- $E_{\text{recip}} = \tfrac{2\pi}{V} \sum_{G \neq 0} |S(G)|^2 \, e^{-|G|^2/(4\alpha^2)} / |G|^2$, k-space sum with Gaussian damping. $S(G) = \sum_A Z_A \, e^{iG \cdot R_A}$ is the nuclear structure factor.
- $E_{\text{self}} = -(\alpha/\sqrt{\pi}) \sum_A Z_A^2$, removes the spurious self-interaction from each nucleus's Gaussian-smeared image.
- $E_{\text{bg}} = -(\pi/(2 \alpha^2 V)) (\sum_A Z_A)^2$, background correction for a non-neutral cell (the "jellium" prescription).

### From point charges to AO densities

For an AO basis the analogous construction extends the same split to
**Gaussian charge distributions**. CRYSTAL's reference method can classify
shell-pair products into an exact penetration zone and a multipole-treated
outer zone. The periodic bipolar quartet expansion itself is given in
Chapter II.4c, especially Eqs. II.4.7-II.4.10, of the 1988 monograph by
Pisani, Dovesi, and Roetti. Saunders et al. 1992 develops the related
electrostatic-potential shell model, penetration terms, and spheropole.

The supported vibe-qc BIPOLE route does not currently replace distant
quartets by multipoles. It evaluates the real-space $J^{SR}$ contribution
with exact erfc-screened Gaussian ERIs and evaluates $J^{LR}$ from analytic
AO-pair Fourier transforms in reciprocal space. The low-level quartet
multipole implementation is unavailable from the SCF drivers because its
two-translation dispatch does not yet preserve the exact contraction's
three-translation periodic domain. Explicit
`use_multipole_far_field=True` therefore raises before setup.

### Nuclear-attraction $V(g)$

The same erfc / erf splitting applies to the nuclei-electron
attraction matrix elements

$$
V_{\mu\nu}(g) = \sum_{A, h} -Z_A \, \langle \chi_\mu | 1 / r_{A,h} | \chi_\nu^g \rangle
$$

The short-range piece is computed by ``compute_nuclear_erfc_lattice``
(real-space, exponentially convergent in $h$). The long-range piece
adds the smooth erf$(\omega r)/r$ kernel, which integrates exactly
over Gaussian AO pairs in reciprocal space. The dispatcher wires both
together inside the SCF Fock build.

## References

- **Ewald, P. P.** "Die Berechnung optischer und elektrostatischer
  Gitterpotentiale," *Ann. Phys.* **64**, 253 (1921). The original
  Ewald paper. DOI: ``10.1002/andp.19213690304``.
- **Saunders, V. R.; Freyria-Fava, C.; Dovesi, R.; Salasco, L.;
  Roetti, C.** "On the electrostatic potential in crystalline systems
  where the charge density is expanded in Gaussian functions,"
  *Mol. Phys.* **77**, 629 (1992). Gaussian-density electrostatics,
  shell penetration, and spheropole treatment in CRYSTAL.
- **Toukmaji, A. Y.; Board Jr., J. A.** "Ewald summation techniques
  in perspective: a survey," *Comput. Phys. Commun.* **95**, 73 (1996).
  Modern Ewald review covering the tin-foil convention, $\alpha$
  optimization, and the various flavors used in MD codes.
- **Allen, M. P.; Tildesley, D. J.** *Computer Simulation of Liquids*,
  2nd ed., Oxford University Press (2017). Clear textbook treatment
  with the $\alpha$ auto-tuning formula vibe-qc uses.
- **Pisani, C.; Dovesi, R.; Roetti, C.** *Hartree-Fock Ab Initio
  Treatment of Crystalline Systems*. Lecture Notes in Chemistry
  vol. 48, Springer (1988).
  [DOI 10.1007/978-3-642-93385-1](https://doi.org/10.1007/978-3-642-93385-1).
  Chapter II.4c gives the periodic quartet bipolar expansion and its
  common lattice-translation sum.

## See also

* [`periodic_methods.md`](periodic_methods.md): comparative tour of
  the periodic-SCF kernels (BIPOLE, GDF, GPW/GAPW) that consume
  these Ewald primitives, plus the surface-reactions workflow.
