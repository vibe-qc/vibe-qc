# Implicit solvation (CPCM / COSMO)

vibe-qc ships polarisable-continuum implicit solvation in the
conductor variant (Cossi-Rega-Scalmani-Barone 2003; "C-PCM"). Aqueous
chemistry is most users' default reaction medium, `solvent="water"`
is now a one-keyword drop-in for `run_job`.

## Quick-start

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("ethanol.xyz")
result = vq.run_job(
    mol,
    basis="def2-svp",
    method="rks",
    functional="pbe0",
    solvent="water",          # ← the only new ingredient
    output="ethanol_aq",
)

sol = result.solvent_result   # SolventResult bundle
print(f"E_total (in water)   = {sol.energy:.6f} Ha")
print(f"E_total (gas-phase)  = {sol.e_gas:.6f} Ha")
print(f"ΔG_solv              = {sol.e_solv * 627.509:+.3f} kcal/mol")
print(f"Macro-iters          = {sol.n_macro_iter}")
```

`solvent=` accepts:

| Input form                                  | Meaning                                                |
|---------------------------------------------|--------------------------------------------------------|
| `None` (default)                            | gas-phase SCF, same as omitting the kwarg.            |
| `"water"`, `"dmso"`, `"acetonitrile"`, …    | preset lookup → `vibeqc.SOLVENT_PRESETS`.              |
| `"H2O"`, `"MeOH"`, `"DCM"`                  | preset aliases, case-insensitive.                     |
| `78.39`                                     | numeric ε, custom dielectric.                         |
| `{"epsilon": 25.0, "variant": "cosmo"}`     | dict → forwarded to :class:`vibeqc.SolventModel`.      |
| `vq.SolventModel(epsilon=25, …)`            | direct construction for full cavity control.           |
| `"vacuum"` / `"none"` / `"gas"`             | ε = 1, short-circuits to gas-phase (uniform return).  |

## Supported methods

Implicit solvation composes with the **mean-field SCFs** and with
**MSINDO**:

| `method=`                    | Solvation route                          |
|------------------------------|------------------------------------------|
| `rhf`, `uhf`, `rks`, `uks`   | CPCM macro-iteration (`run_cpcm_scf`)    |
| `msindo`                     | dedicated COSMO driver (see [MSINDO](msindo.md)) |

Every other method refuses a non-`None` `solvent=` with a clear
`ValueError`: there is no solvation composition for them yet, and
silently returning a gas-phase energy would misrepresent the result.

* **CAS family** (`casscf`, `caspt2`, `nevpt2`, `casci`, `mrci`) and
  the other wavefunction solvers (`cisd`, `selected_ci`, `fci`,
  `dmrg`, `v2rdm`, `transcorrelated_ci`): no CPCM coupling of the
  correlated density. This includes geometry optimisations: the
  finite-difference fallback path refuses too, rather than walking the
  gas-phase surface while claiming solvation (see
  [geometry optimisation](geometry_optimization.md)).
* **Post-SCF correlation** (`mp2` and spin-scaled variants, `ccsd`,
  `ccsd(t)`, coupled-pair variants, `dlpno-*`, `ovgf`): the
  mean-field reference *could* run in solvent, but the correlation
  step's reported total is not composed against the in-solvent
  reference, so the combination is gated until a validated
  post-SCF-in-solvent scheme exists.
* **Semi-empirical** (`gfn2_xtb`, `pm6`, `om1`/`om2`/`om3`, `dftb0`,
  `scc_dftb`, `ccm`): only MSINDO has a COSMO route.
* **MLIPs** (`mace`): a pre-trained gas-phase surface; no continuum
  coupling.
* `rohf` / `roks` raise `NotImplementedError` (the open-shell Roothaan
  drivers are not yet coupled to CPCM; see the ROHF handover).

`method="auto"` with `solvent=` works whenever the automatic selection
lands on a mean-field method; if it resolves to FCI / selected-CI
(very small systems), the error names the resolution; pass
`method="rhf"` (or `rks` + functional) explicitly.

## Solvent preset table

Static dielectric constants ε at 298 K, matching the Gaussian 16
`SCRF=Solvent=…` table to ±1 %. The full list of preset keys lives in
`python/vibeqc/solvation/presets.py`.

| Solvent              | ε     | Aliases                          |
|----------------------|-------|----------------------------------|
| water                | 78.39 | `h2o`                            |
| methanol             | 32.61 | `meoh`                           |
| ethanol              | 24.85 | `etoh`                           |
| dmso                 | 46.83 |                                  |
| dmf                  | 37.22 |                                  |
| acetonitrile         | 35.94 | `mecn`, `acn`, `ch3cn`           |
| acetone              | 20.49 |                                  |
| thf                  | 7.43  |                                  |
| dichloromethane      | 8.93  | `dcm`, `ch2cl2`                  |
| chloroform           | 4.71  | `chcl3`                          |
| benzene              | 2.27  |                                  |
| toluene              | 2.37  |                                  |
| n-hexane             | 1.88  | `hexane`                         |
| vacuum               | 1.0   | `none`, `gas`                    |

For a solvent outside this table, pass a numeric ε or build a custom
:class:`vibeqc.SolventModel`:

```python
ionic_liquid = vq.SolventModel(
    epsilon=15.6,
    name="1-butyl-3-methylimidazolium-BF4",
    radii_scale=1.25,           # tighter sphere scaling
    n_points_per_sphere=302,    # tighter Lebedev
    variant="cpcm",             # "cpcm" (default) or "cosmo"
)
```

## What CPCM actually computes

The solute sits in a molecular-shaped cavity built as the union of
atom-centred spheres (Bondi vdW radii ×1.20). Outside the cavity, the
solvent is treated as a homogeneous dielectric. The boundary carries
an apparent surface charge $q$ that satisfies

$$
A \mathbf{q} = -f(\varepsilon)\, \mathbf{V}(R_\text{nuc}, D),
\qquad
f(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon}
$$

where $A$ is the discrete CPCM matrix (Scalmani-Frisch 2010 diagonal +
$1/|s_i - s_j|$ off-diagonal) and $\mathbf{V}$ is the molecular
electrostatic potential at the cavity tessellation points. The
solvation energy is

$$
E_\text{solv} = \tfrac{1}{2} \sum_i q_i V_i
$$

and the in-solvent total energy is $E_\text{tot}^\text{solv} = E_\text{tot}^\text{gas} + E_\text{solv}$.

The full coupled problem $\{q, D\}$ is solved by a macro-iteration loop
(Cossi-Scalmani 2003 § II.B): converge SCF for the current $V_q$, build a
new $V_q$ from the new density, repeat until $|\Delta E_\text{solv}| <$
`tol_e_solv` (default 1 µHa). Three to five outer cycles is typical for
small organic molecules in water.

## Cavity tessellation

Each atomic sphere is discretised by a Lebedev-Laikov quadrature; points
sitting inside a neighbouring atomic sphere are smoothly removed via the
Scalmani-Frisch continuous switching function (CSC; $\sigma = 0.5\,a_0$).
This makes the cavity surface, and hence $E_\text{solv}(R)$, $C^1$
continuous, which is essential for clean geometry optimisation.

**Defaults that match Gaussian's `SCRF=PCM` cavity:**

| Parameter                   | Default | Notes                                |
|-----------------------------|---------|--------------------------------------|
| vdW radii                   | Bondi 1964 | per-element override via `radii=`  |
| `radii_scale`               | 1.20    | PCM / GEPOL convention               |
| `solvent_probe_radius_ang`  | 0.0     | 0 = vdW-SES; set 1.385 Å for water SAS |
| `n_points_per_sphere`       | 302     | Lebedev order 29 (Gaussian default)  |
| `switching_sigma_bohr`      | 0.5     | Scalmani-Frisch CSC width            |
| `switching_drop_threshold`  | 1e-8    | Below this switched weight, drop the point |

Supported Lebedev point counts: 6, 14, 26, 38, 50, 74, 86, 110, 146,
170, 194, 230, 266, 302, 350, 434, 590, 770, 974. The PySCF / ORCA
default is 110; the Gaussian default is 302; Q-Chem defaults to 194.

## Inspecting the cavity

```python
from vibeqc.solvation import build_cavity

cav = build_cavity(
    atom_positions_bohr=...,    # (n_atoms, 3) bohr
    atom_numbers=...,           # iterable of Z
    radii_scale=1.20,
    n_points_per_sphere=302,
)

print(f"{cav.n_points} surface points after switching")
print(f"Total area: {cav.total_surface_area_bohr2:.2f} bohr²")

# Dump for visualisation
import numpy as np
np.savetxt("cavity.xyz", np.hstack([cav.points, cav.weights[:, None]]))
```

## Geometry optimisation in solvent

vibe-qc ships a **closed-form analytic CPCM solvation gradient**,
:func:`vibeqc.cpcm_gradient`. The ASE calculator picks it up
automatically when `solvent=` is set on the calculator:

```python
from ase.optimize import BFGSLineSearch
from vibeqc.ase import VibeQC

atoms.calc = VibeQC(basis="def2-svp", functional="b3lyp",
                    solvent="water")
BFGSLineSearch(atoms).run(fmax=0.05)
```

This covers `variant="cpcm"` and `variant="cosmo"`. A Direct COSMO-RS
(`variant="dcosmo-rs"`) energy is not smooth in the geometry, so it has no
gradient and `cpcm_gradient` refuses it; see
[Direct COSMO-RS](#when-it-will-not-converge).

The gradient of the in-solvent total energy
$E_\text{tot} = E_\text{HF}^\text{gas}[D^\text{solv}] + \tfrac{1}{2}q^\mathsf{T}V$
decomposes (envelope theorem at SCF + CPCM convergence) into four
pieces, each evaluated in closed form:

1. the standard gas-phase analytic SCF gradient at the in-solvent
   density,
2. $\tfrac{1}{2f}\,q^\mathsf{T}(\partial A/\partial R)\,q$, the
   cavity self-interaction-matrix derivative, where $f$ is the screening
   factor of the run's own `variant`, the same one that built $q$,
3. $q^\mathsf{T}(\partial V^\text{nuc}/\partial R)$, the nuclear-ESP
   derivative,
4. $q^\mathsf{T}(\partial V^\text{elec}/\partial R)$, the
   electronic-ESP derivative, evaluated with a single libint
   nuclear-attraction-gradient pass over the cavity points
   (`compute_external_charge_density_gradient`).

No geometry rebuilds, no finite-difference step error. A
finite-difference reference gradient,
:func:`vibeqc.cpcm_gradient_fd` (full SCF + CPCM re-converge at
$6\cdot n_\text{atoms}$ displaced geometries), is retained as a
cross-check oracle; `cpcm_gradient(..., use_fd_electronic=True)`
likewise swaps just the electronic-ESP piece back to a
finite-difference path for verification.

The assembled analytic gradient agrees with the full-energy
finite-difference oracle to FD-truncation level, on water/6-31G the
maximum deviation scales as $O(h^2)$ down to ~6e-9 Ha/bohr at
$h = 2\cdot10^{-4}$ bohr. (Releases before v0.12.0 carried a
factor-of-2 error in the off-diagonal $\partial A/\partial R$ piece, 
a systematic ~3e-4 Ha/bohr on water-sized solutes, below typical
`fmax` optimisation thresholds but large enough to shift tightly
converged minima; fixed and FD-pinned per component since.)

(Releases before v0.15.157 evaluated piece 2 with the **CPCM** screening
factor regardless of `variant`, so a `variant="cosmo"` run differentiated
a different energy than it reported. Because the two factors diverge as
$\varepsilon$ falls, the error grew in nonpolar solvents: the cavity
contribution was 18 % low at benzene, 0.6 % low at water, and the
analytic-vs-FD residual plateaued at ~5e-5 Ha/bohr instead of converging.
CPCM runs were unaffected. Fixed and pinned in
`tests/test_solvation_variant_gradient.py`; issue #546.)

## Choosing the cavity

`SolventModel(cavity=...)` selects the construction:

* `"lebedev"` (default): each atomic sphere is paved with a Lebedev grid and
  segments are switched off where spheres overlap. This describes the *convex*
  solute surface and leaves the concave crevices between atoms unpaved.
* `"fine"`: the Klamt & Diedenhofen 2018 COSMO FINE Cavity, a pseudo-density
  iso-surface triangulated by marching tetrahedra
  (:mod:`vibeqc.solvation.fine_cavity`). One closed smooth surface, so the
  concave regions pave automatically.

The default is deliberately unchanged: the two give different segment sets, so
switching moves every solvated energy. On water/STO-3G at $\varepsilon = 78.39$
the measured difference is $E_\text{solv} = -3.85$ against $-3.35$ kcal/mol.

The FINE cavity matters most for COSMO-RS, where the screening charge
*density* is the central descriptor: polarization charge accumulating at the
edges of paved regions distorts a sigma profile directly.

The analytic gradient **does** support the FINE cavity. It does not reuse the
Lebedev chain rule -- a FINE segment neither rides rigidly on its parent atom
nor carries a switching function -- but differentiates the construction itself:
the pseudo-density iso-surface points, the triangle and basis-point areas, the
Becke coarsening onto the segment grid, and the marching box that is anchored
to the molecule (:mod:`vibeqc.solvation.fine_cavity_gradient`, issues #729 and
#746). On water/STO-3G it agrees with `cpcm_gradient_fd` to better than
5e-6 Ha/bohr, which is that reference's own noise floor.

The derivatives hold at *fixed segment topology*, the same caveat the Lebedev
cavity carries through its `drop_threshold`.
:func:`vibeqc.cpcm_gradient_fd` re-converges displaced geometries and so makes
no assumption about the construction at all.

### The molecule-fixed frame

A solvated energy must not depend on how the solute happens to be oriented in
the laboratory -- the cavity is a property of the molecule. Until #769 it did,
because the marching lattice of step 2 and the geodesic segment directions of
step 6 were laid out along the laboratory axes. Measured on water at
$\delta = 0.40$ Å, RHF/STO-3G, $\varepsilon = 78.39$: rotating the solute
about a generic axis moved the energy by 1.0e-05 Ha and changed the segment
count between 235 and 239.

The CFC is now built in the solute's own frame, which is the paper's workflow
step 1 -- origin at the centre of the atom positions, $x$ toward the atom
farthest from it, $y$ from the largest orthogonal component. The cavity rotates
rigidly with the solute: the same measurement gives 5.7e-14 Ha, and the segment
set is reproduced to 1.3e-14 bohr. Records come back in laboratory coordinates
either way; the frame is an internal device.

`SolventModel(cavity="fine", fine_frame="lab")` restores the old behaviour. It
is worth knowing about for two reasons: it is the right choice for a **linear**
solute, whose molecule-fixed frame has no nuclear derivative (below), and the
two frames give different segment sets, so a CFC number from before #769 is
reproduced only with `"lab"`.

The analytic gradient supports it. A frame that turns with the solute gives the
marching lattice and every segment direction a derivative of their own, and that
contribution is not small -- 2.3e+01 of a cavity gradient whose largest
component is 2.9e+01. Rather than rewrite the chain rule for a turning lattice,
:class:`vibeqc.solvation.cavity_derivative.FrameCavityDerivative` reverses the
cavity into the frame it was built in, where the existing chain is exactly
valid, and composes it with the frame's own derivative.

That buys an exact check no finite difference is needed for: **a molecule in
free space feels no torque**. On the asymmetric solute above, the lab frame's
analytic gradient has a net torque of 2.9e-05 Ha -- the rotational counterpart
of its orientation-dependent energy -- and the molecule-fixed frame's has
1.0e-13. Agreement with `cpcm_gradient_fd` improves too, from 9.0e-05 to
1.4e-05 Ha/bohr.

Two cases are weaker than the rest, both local and both measured. A **linear**
solute determines no y-axis, so its azimuth is a discontinuous lab-derived
choice with no nuclear derivative at all -- it is refused; use `"lab"` there,
and note `cpcm_gradient_fd` cannot rescue it either, since its displacements
cross the same jump. And the frame's two `argmax` selections **tie** at
symmetric geometries, water's equilibrium structure included. The *energy* is
safe there: the two candidate frames are related by a symmetry operation of the
solute, so they produce the same cavity relabeled and the energy is smooth
across the crossing. The **gradient** is not, because at a tie the frame jumps
rather than varies: the analytic gradient carries a spurious 1.9e-05 Ha/bohr
force along the tie-breaking direction, in a component symmetry requires to
vanish, where a converged finite difference gives 1e-11. Break the tie by any
amount and agreement returns to 4.7e-07 Ha/bohr. In practice this means a
symmetric structure is not exactly a stationary point of the analytic gradient,
at a level well below the thresholds an optimizer converges to. An *accidental*
tie, with no symmetry relating the two atoms, costs a 5.8e-06 Ha step in the
energy and remains open.

### The tetragon split

Step 3 cuts four edges when a tetrahedron has two corners inside the surface
and two outside, and splits the resulting tetragon into two triangles on its
shorter diagonal. That comparison flips where the two diagonals are equal.
The quad's *total* area survives the flip, but step 4's "a third of each
triangle to each of its three corners" does not: the same total lands on the
four corners differently, and each corner's basis area jumps by about
0.023 bohr². About 40 % of a typical triangulation is tetragon halves.

Step 4 now gives a tetragon's four corners a quarter of the pair's area each,
which is the paper's own principle -- the area is "equally assigned to the
corner points" -- applied to the figure step 3 actually produced. No diagonal
can change it. Scanning a coordinate through a flip, the second difference of
the solvated energy was 125 % away from its trend and is now 0.074 %; agreement
between the analytic gradient and `cpcm_gradient_fd` on an asymmetric solute
improves from 7.1e-06 to 1.2e-06 Ha/bohr. Energies move by under 1e-06 Ha
(issue #779).

What remains at a flip is the tetragon's own non-planarity: two flat triangles
spanning four points that are not coplanar have slightly different total areas
depending on the diagonal, worth 7e-06 bohr² of surface area and 2e-04 bohr³ of
enclosed volume. That is inherent to representing a curved patch with flat
triangles, and it shrinks with the grid.

### Smooth atomic cells

Step 5 assigns each basis point to "the atom of smallest relative distance",
$\tau_a = |r - R_a| / \rho_a$. That `argmin` is discrete, so a point on a
boundary used to flip at an infinitesimal displacement and its whole area moved
to the other atom's segment grid: 0.19 bohr² at once, a 6.9 % error in the local
slope of the energy, a 2.9e-08 Ha step. Twelve coordinates scanned over
±6e-03 bohr found thirty crossings.

The cells are now smooth (`vibeqc.solvation.atom_partition`), in the same
spirit as the segment partition one level down. The width of the sharing band is
the **marching spacing**: whether a basis point falls on one side of a boundary
is only meaningful to within the spacing that placed it, so the smoothing tends
to the hard rule as the grid refines. The per-atom area budget moves 0.9 % from
the hard assignment at $\delta = 0.40$ Å and 0.33 % at 0.20 Å, and a basis
point feeds 1.8 atoms on average at 0.40 Å and 1.4 at 0.20 Å. The energy's slope
across a former crossing is smooth to 1.4e-03 instead of 6.9e-02 (issue #771).

This costs roughly a factor of two in cavity construction, and it moved every
`cavity="fine"` energy by about 7e-06 Ha.

### The outward projection

Step 6 pushes a segment centre onto its atom's sphere when the area-weighted
mean falls inside -- which on a curved surface it always does. The paper states
that as a hard rule, and a hard maximum is C0 but not C1: the segment position's
slope jumps where a centre crosses the sphere. The energy is untouched; it is
the **gradient** that steps, so what this cost was finite-difference second
derivatives. An area-weighted functional's slope jumped 0.07 % across one
crossing, and crossings arrive about every 0.025 bohr of displacement per
coordinate.

The projection is now eased over a band 1e-03 of the sphere radius wide
(0.003 bohr) with a softplus, so it is smooth and never places a segment inside
the sphere it screens. The band is narrow against what it eases -- the
projection depth is 0.019 bohr median -- and wide enough for a finite difference
to resolve, which is the consumer that needed it. It recovers the paper's rule
as the band vanishes: the solvated energy moves 6.6e-07 Ha from the hard rule at
this width, 2.1e-09 at a tenth of it and nothing at a hundredth (issue #770).

One side effect is worth knowing: the resolution guard on
`fine_grid_spacing_ang` fires far less often. Laying the segment directions out
in the laboratory means the surface meets them at an arbitrary angle, so a Becke
cell can be left holding a single basis point and two segments then coincide. On
water it first fired at 0.44 Å; after #769 fixed the direction grid's
orientation and #771 let a basis point feed more than one atom's grid, the
first refusal is at 0.70 Å in both frames.

## Moving the switching width

`switching_sigma_bohr` widens the Scalmani-Frisch switch, which its own
docstring recommends for close atom pairs where the default 0.5 bohr clip is
too aggressive. Two things had to be fixed before that was safe to do.

The gradient did not know the width. `SolventModel.switching_sigma_bohr`
reached `build_cavity`, but the tessellation did not record it, so
`cpcm_gradient` read a hard-coded 0.5 and differentiated a cavity the run never
built -- wrong by 0.35 % of the largest component at $\sigma = 0.35$, against a
converged finite difference. The cavity now carries its own width and the
gradient reads it; a cavity that switches and cannot say how widely is refused
rather than assumed (issue #745).

What made that survive for so long is worth knowing if you are ever debugging a
gradient: a gradient taken with the wrong width is still **self-consistent**.
It satisfies translational invariance to 1e-15 and every other exact identity.
Only `cpcm_gradient_fd`, which rebuilds the energy, can see it.

The second is on the energy side. `switching_drop_threshold` (default 1e-8)
drops points whose switched weight falls below that fraction of their raw
Lebedev weight. The switching function is smooth across the cutoff but the
*point set* is not, so a point crossing it steps the energy. At the default
width nothing sits near the cutoff and this is invisible. A wide switch puts
many points there: at $\sigma = 0.8$ a 2.5e-04 bohr displacement moves the
count from 904 to 905, the local slope reads -0.027 against a trend of +0.016,
and no gradient can match a finite difference across that step.

So if you widen the switch, **set `switching_drop_threshold=0.0`**. It costs
zeros in the $A$ matrix and buys a smooth energy: at $\sigma = 0.8$ the
analytic gradient goes from 2.1e-02 to 1.2e-07 Ha/bohr against FD.

## Direct COSMO-RS

`variant="dcosmo-rs"` feeds the solvent's sigma potential back into the
electronic Hamiltonian, so the solute's wavefunction responds to a *real*
solvent rather than to a scaled conductor. The method is Sinnecker, Rajendran,
Klamt, Diedenhofen & Neese, *J. Phys. Chem. A* **110**, 2235 (2006),
[doi:10.1021/jp056016z](https://doi.org/10.1021/jp056016z), and it is three
equations:

$$q = -A^{-1}\phi, \qquad
  \phi_t^{\Delta RS} = \mu_S'(\sigma_t), \qquad
  V^{RS} = -\sum_t \frac{q_t + q_t^{\Delta RS}}{|r - r_t|}$$

with $q^{\Delta RS}$ solved from $\phi^{\Delta RS}$ through the same first
equation, iterated to self-consistency.

```python
from vibeqc.solvation import cosmors as rs

potential = rs.sigma_potential_segments(solvent_segments, mole_fractions, params)
sm = vq.SolventModel(
    variant="dcosmo-rs",
    sigma_potential=potential,
    epsilon=float("inf"),      # required; see below
    cavity="fine",             # required; see below
)
```

The energy is the conductor COSMO energy plus the solute's own COSMO-RS
chemical potential, $G = E_{\rm gas}[D] + \tfrac12 q\cdot V_{\rm tot} +
\sum_t a_t\,\mu_S(\sigma_t)$. The paper states the operator but not the
energy; the two are the same statement, because differentiating that third term
with respect to the density is exactly what produces the extra charges
$q^{\Delta RS}$.

### Getting a solvent potential

The sigma potential is the solvent, so it has to come from somewhere. Compute
it from the solvent's own conductor COSMO run:

```python
import math
from vibeqc.solvation import cosmors as rs

K = rs.KLAMT_1998
solvent = vq.run_cpcm_scf(
    water, basis, method="rhf",
    solvent=vq.SolventModel(
        epsilon=math.inf, variant="cosmo", cavity="fine",
        radii=dict(K.cavity_radii), radii_scale=1.0,   # the set's own radii
        solvent_probe_radius_ang=0.0,
    ),
)
potential = rs.solvent_sigma_potential(solvent, K, method="rhf", basis="sto-3g")
```

`solvent_sigma_potential` refuses two mistakes that produce a perfectly
ordinary-looking potential and a wrong number.

A **screened run**: `build_conductor_surface` takes the charges as stored, and
those are the ideal $q^*$ only at $\varepsilon=\infty$; at finite dielectric
vibe-qc stores $q = f q^*$, so the profile is scaled by $f$ and nothing
downstream can detect it.

**Foreign cavity radii**: a parameter set is fitted against a radius set --
`KLAMT_1998` against H 1.30, C 2.00, N 1.83, O 1.72, Cl 2.05 Å -- and since
$\sigma = q/a$, the default scaled-Bondi cavity changes every $\sigma$ the
constants were fitted to reproduce.

What it cannot refuse is the **QC protocol**. `KLAMT_1998` was fitted on
`dmol/bpw91/dnp/cosmo-inf/nspa92`, and whatever you compute carries its own
method and basis. That is a real accuracy limit, not a bug, and no code can
repair it -- so it is recorded: `potential.protocol` says what the surface was
and `params.fitted_protocol` says what the constants expect. Compare them
before believing a number.

A computed water potential has the shape water should have: **positive** near
$\sigma = 0$, which is the cost of putting a nonpolar surface in water, and
**negative at both wings**, where a polarized surface donates or accepts a
hydrogen bond. One consequence looks like a bug and is not: the COSMO-RS term
for a small solute in water comes out *unfavourable* by around half a kcal/mol
on top of the conductor energy, because most of a small solute's surface sits
at moderate $\sigma$ where water's potential is positive. That is the
hydrophobic penalty.

### Two requirements, both measured rather than stylistic

**`epsilon=inf`.** The sigma potential *is* the real-solvent physics, so
scaling by $f(\varepsilon)$ as well would count the solvent twice. The paper is
explicit that the correction acts on "the ideal screening charges appearing in
a conductor". Direct COSMO-RS replaces the dielectric scaling rather than
composing with it, and the solvent is chosen by which sigma potential you pass.

**`cavity="fine"`.** The feedback is evaluated at $\sigma_t = q_t/a_t$, and on
a switched Lebedev cavity that is ill-conditioned: the switching function
shrinks weights continuously, so the smallest segment sits 2.9e+07 below the
median and $|\sigma|$ reaches 640 e/Å² against a potential parameterized on
±0.025. Dropping the tail does not help -- there is no clean cut, and a
vanishing segment's energy does not vanish with it, since
$a\,\mu(\sigma) \sim a\sigma^2 \sim 1/a$. The COSMO FINE Cavity's segments
all carry real area and put the 99th percentile of $|\sigma|$ at 0.028, inside
the parameterization.

### When it will not converge

Klamt's hydrogen-bond term is piecewise linear in each density, so
$\mu_S'$ **jumps** where a segment's $\sigma$ crosses $\pm\sigma_{hb}$, and
the Boltzmann average over solvent partners does not damp it: every partner
steps at the same $\sigma$. A segment parked on that corner sees a
discontinuous potential from one macro-iteration to the next.

`result.direct_feedback` reports both failure modes, and they are the first
things to read when the outer loop will not settle:

| | meaning |
|---|---|
| `fraction_near_hb_corner` | segments sitting on the discontinuity (about a quarter is normal) |
| `fraction_outside_grid` | area whose sigma is extrapolated beyond the fitted range |

The extrapolation is through the misfit term, which is quadratic and defined
everywhere, so it is smooth rather than a clamp -- but it is still
extrapolation, and a large share means the surface and the parameterization
disagree about what a polar segment looks like.

Because the feedback is discontinuous in $\sigma$, a Direct COSMO-RS energy is
**not smooth in the geometry**. There is no analytic nuclear gradient for it,
and a finite-difference one will step wherever a segment crosses a threshold.

`cpcm_gradient` therefore refuses a Direct COSMO-RS result with
`NotImplementedError`. v0.17.1, where Direct COSMO-RS first shipped, did not
refuse it: it returned
the conductor COSMO gradient, without the COSMO-RS energy term or the
$q^{\Delta RS}$ operator. That gradient passes translational invariance to
1e-15 and is wrong by up to 1.4e-03 Ha/bohr on water. For an analytic gradient
use `variant="cosmo"`, which is a different energy. `cpcm_gradient_fd`
differentiates the Direct COSMO-RS energy itself, subject to the steps above.

## Comparing a solvated run against another code

Two quantities of a solvated run can be compared with another program, and
they behave differently.

* **The in-solvent total** (`result.energy`, printed as `In-solvent total`)
  compares only against the other code's in-solvent total, and carries every
  gas-phase convention difference on top of the solvation ones.
* **The solvent-response component** (`sol.e_solv`, printed as
  `Solvation energy (1/2 q.V_tot)`) is the more robust one: it is a
  difference of two energies on the same functional and grid, so gas-phase
  conventions largely cancel. ORCA prints the same quantity as
  `CPCM Dielectric` in its `TOTAL SCF ENERGY` block.

Do not compare a solvated total against a *gas-phase* total, and do not
compare either against an inner-SCF energy. Before v0.15.138 vibe-qc's banner
printed the inner SCF's $E_\text{SCF}^{w/V_q}$ rather than the in-solvent
total (archived tracker `vibeqc#148`) -- 23-42 mHa away from it. A harvester that read the banner
reported a 54.6 mHa cross-code disagreement on the case below that does not
exist; the run's own solvation block was right all along.

`Total solvation shift` (in-solvent minus gas-phase) is **not** `e_solv`: the
solute density also relaxes in the field, and the block prints that
polarisation cost separately. Compare like with like.

### Measured against ORCA

CH$_3$COOH / def2-TZVP / B3LYP, water, on one geometry, against an archived
ORCA 6.1.1 `! B3LYP def2-TZVP TightSCF CPCM(water)` reference
(in-solvent total $-229.0790600869$, `CPCM Dielectric` $-0.0126977537$ Ha):

| vibe-qc setting | in-solvent total | response component | component vs ORCA |
|---|---|---|---|
| defaults (`solvent="water"`) | $-229.0799875548$ | $-0.0139868457$ | $-1.29$ mHa |
| ORCA's $\varepsilon$ and radii | $-229.0821294531$ | $-0.0164592598$ | $-3.76$ mHa |

The in-solvent totals agree to **0.93 mHa**, so raw solvated totals are
comparable once both sides quote the same quantity.

The second row is the surprise, and it is why this section exists: making the
*named* parameters agree makes the answer **worse**. ORCA's hydrogen radius is
1.32 A against vibe-qc's 1.44 A, and the tighter cavity deepens the response
past ORCA's own value. What remains is the surface itself -- vibe-qc switches a
Lebedev grid per sphere (1895 points here), ORCA lays down a Gaussian-charge
surface at constant charge density (830 points). Two cavities that carry the
same radii are still not the same cavity, and that sets the floor on any
cross-code component comparison. Matching $\varepsilon$ and radii is
therefore not a recipe for agreement; it is only a way to remove those two
variables when attributing a difference.

**The geometry has to be identical, not similar.** Re-running the same case on
a geometry whose C=O bonds differ by at most 0.07 A moves the component by
2.5 mHa -- twice the entire cross-code gap. A component comparison across codes
is meaningless unless both sides ran the same coordinates.

For reference, ORCA 6.1's `CPCM(water)` defaults against vibe-qc's:

| knob | vibe-qc default | ORCA 6.1 `CPCM(water)` |
|---|---|---|
| dielectric | 78.39 (`water` preset) | 80.1510 |
| C / O radius | 2.040 / 1.824 A (Bondi x 1.20) | 2.0400 / 1.8240 A |
| H radius | 1.44 A | 1.3200 A |
| probe | `solvent_probe_radius_ang=0.0` | VDW surface |
| discretisation | switched Lebedev, per sphere | Gaussian VDW, constant charge density |
| screening | `variant="cpcm"`, $f=(\varepsilon-1)/\varepsilon$ | `Epsilon function type ... CPCM` |

```python
orca_like = vq.SolventModel(
    epsilon=80.1510,                      # ORCA's water, not 78.39
    variant="cpcm",
    radii={1: 1.10, 6: 1.70, 8: 1.52},    # pre-scaling, Angstrom
    radii_scale=1.2,                      # -> H 1.32, C 2.04, O 1.824 A
)
```

## Choosing CPCM vs COSMO

The two are not separate formulas. Klamt & Schuurmann 1993 p. 800 gives the
family both belong to,

$$f(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon + x},
\qquad x \in [0, 2],$$

and the `variant=` field just selects $x$:

* `"cpcm"` (default): $x = 0$, so
  $f(\varepsilon) = (\varepsilon - 1)/\varepsilon$.
  Converges smoothly to the conductor limit at $\varepsilon \to \infty$.
* `"cosmo"`: $x = 1/2$, so
  $f(\varepsilon) = (\varepsilon - 1)/(\varepsilon + 0.5)$
  (Klamt 1993 p. 801, which extends the exact conductor result to finite
  $\varepsilon$ with a relative error below $\varepsilon^{-1}/2$).

`vibeqc.solvation.ScreeningModel` carries the resolved factor, and
`SolventResult.screening` is the one that actually built the apparent charges.
Read `f` from there rather than recomputing it from `(epsilon, variant)`: that
recomputation is what made COSMO gradients differentiate the wrong energy
(issue #546). A gas-phase result carries `screening = None`, which is how
"no reaction field" is expressed; `epsilon = inf` gives exactly $f = 1$.

The two agree to better than 1 % above $\varepsilon \approx 30$, for
water and most polar solvents the choice is inconsequential
($\Delta E_\text{solv} \lesssim 0.1$ kcal/mol). For very low-dielectric
solvents (e.g. n-hexane, $\varepsilon \approx 1.9$) the choice can
shift $E_\text{solv}$ by ~5 %.

## Density fitting and RIJCOSX

The CPCM macro-iteration drives the inner SCF through whatever
Fock-build path the method options select, there is nothing
solvation-specific about the two-electron build. Set `density_fit`
(and optionally `cosx`) on the method options and the CPCM SCF picks
up RI-JK / RIJCOSX automatically:

```python
import vibeqc as vq

opts = vq.RHFOptions()
opts.density_fit = True            # RI-JK
# opts.aux_basis = "def2-svp-jk"   # optional — autodetected if omitted
# opts.cosx = True                 # RIJCOSX (RI-J + chain-of-spheres K)

result = vq.run_job(mol, basis="def2-tzvp", method="rhf",
                    solvent="water", rhf_options=opts)
```

When `aux_basis` is left empty, it is autodetected from the orbital
basis name via `vibeqc.default_aux_basis_for(orbital, kind="jk")` and
filled in on the options struct (so the gas-phase bootstrap SCF, the
macro-iteration inner SCFs, and the JKBuilder all share one aux
basis). Density fitting changes the in-solvent total energy by the
usual fitting error, ~1e-4 Ha for `def2-svp-jk`, and leaves
`E_solv` essentially unchanged (~1e-6 Ha).

The CPCM analytic gradient currently uses conventional (non-DF)
two-electron integral derivatives for the gas-phase SCF piece even
when the SCF itself was density-fitted; the resulting DF/conventional
mismatch is within the DF fitting error and below typical
optimisation tolerances. A DF-consistent CPCM gradient is a future
refinement.

## Known limitations

* **Molecules only**, periodic CPCM is out of scope until v1.x; the
  conductor model needs adaptation to handle the periodic image
  problem cleanly.
* **No non-electrostatic terms**, vibe-qc's CPCM is currently
  electrostatic-only. The cavitation, dispersion, and repulsion (CDR)
  terms that a full SMD-style model adds are on the future roadmap.

## Theoretical references

The implementation follows the standard CPCM literature:

* **Klamt, A. & Schüürmann, G.** *J. Chem. Soc. Perkin Trans. 2*, 799
  (1993), COSMO conductor model.
* **Cossi, M., Rega, N., Scalmani, G. & Barone, V.** *J. Comp. Chem.*
  24, 669 (2003), CPCM as the practical numerical variant; cavity +
  matrix layout used here.
* **Scalmani, G. & Frisch, M. J.** *J. Chem. Phys.* 132, 114110
  (2010), continuous-surface-charge (CSC) formulation; the
  Scalmani-Frisch switching function and the
  $A_{ii} = 1.0694\sqrt{4\pi / w_i}$ diagonal.
* **Tomasi, J., Mennucci, B. & Cammi, R.** *Chem. Rev.* 105, 2999
  (2005), comprehensive PCM-family review.
* **Lange, A. W. & Herbert, J. M.** *J. Chem. Phys.* 133, 244111
  (2010), modern Scalmani-style analytic CPCM gradient.
* **Sinnecker, S., Rajendran, A., Klamt, A., Diedenhofen, M. & Neese, F.**
  *J. Phys. Chem. A* 110, 2235 (2006), Direct COSMO-RS. A
  `variant="dcosmo-rs"` run cites it together with the COSMO-RS
  (Klamt 1995, 1998) and COSMO FINE Cavity (Klamt & Diedenhofen 2018)
  papers. Scalmani and Frisch are left out, because their diagonal is the
  Lebedev cavity's.

For citation in published work, see
[citation discipline](https://github.com/vibe-qc/vibe-qc/blob/main/CONTRIBUTING.md#citation-discipline).
