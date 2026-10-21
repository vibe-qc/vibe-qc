---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for the intrinsic reaction coordinate: locate the planar transition state of the ammonia umbrella inversion with the dimer method at RHF/STO-3G, confirm it with one imaginary frequency, then follow the IRC in both directions with run_irc down to the two pyramidal minima. Includes the theory of mass-weighted steepest descent, the Ishida-Morokuma-Komornicki stabilised integrator, and what an IRC does and does not prove about a mechanism."
    "og:title": "vibe-qc tutorial - following the IRC from a transition state"
---

# Following the IRC from a transition state

A transition-state search ends at a first-order saddle point: a geometry
with zero gradient and exactly one negative Hessian eigenvalue. That
proves the structure is a saddle, not that it is *the* saddle you wanted.
A saddle connects two minima, and nothing in the search tells you which
two; a dimer or climbing-image NEB run started from a poor guess can
converge to a rotational barrier, an inversion or a conformational flip
instead of the bond-breaking step you were after. The **intrinsic
reaction coordinate** (IRC, {vibeqc-cite}`fukui_irc_1981`) settles the
question: it is the steepest-descent path from the saddle in
mass-weighted coordinates, followed in both directions until the
gradient vanishes, and its two ends are the minima the transition state
actually links.

vibe-qc's `run_irc` traces that path for molecules on any SCF reference
with a gradient (RHF, UHF, ROHF, RKS, UKS, ROKS) and on the MACE
potential; it is production for molecules and fails closed for periodic
systems (see [Methods: theory and status](../methods.md#how-to-read-the-status-labels)).
This page runs the whole workflow on the ammonia umbrella inversion at
RHF/STO-3G: locate the minimum, find the planar transition state with the
dimer method, check it has one imaginary frequency, and follow the IRC to
both pyramidal minima. The same molecule is treated with climbing-image
NEB in [Nudged elastic band](neb_reaction_path.md), so you can compare
the two ways of finding the saddle.

## Setup

```python
import numpy as np
import vibeqc as vq

ANG = 1.8897259886
KCAL = 627.5094740631
MASS = {7: 14.003074, 1: 1.007825}            # amu, for the arc length below

def build_nh3(theta_deg=106.67, r_ang=1.012, z_sign=-1):
    """Pyramidal NH3: N at the origin, three H below (z_sign=-1) or above it."""
    theta, r = np.deg2rad(theta_deg), r_ang * ANG
    sa = np.sqrt(2.0 * (1.0 - np.cos(theta)) / 3.0)
    ca = np.sqrt(1.0 - sa ** 2)
    atoms = [vq.Atom(7, [0.0, 0.0, 0.0])]
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        atoms.append(vq.Atom(1, [r * sa * np.cos(phi), r * sa * np.sin(phi), z_sign * r * ca]))
    return vq.Molecule(atoms, 0, 1)

def height(system):
    """Mean height of the hydrogens above the nitrogen (bohr): 0 means planar."""
    z = [a.xyz[2] for a in system.atoms]
    return float(np.mean(z[1:]) - z[0])

# 1. The minimum, for reference.
minimum = vq.optimize_molecule(build_nh3(), "sto-3g", method="rhf")
print(f"minimum: E = {minimum.energy:.8f} Ha, height = {height(minimum.system):.3f} bohr")

# 2. The saddle: a dimer search from a flattened pyramid (HNH = 118 deg).
ts = vq.run_dimer(build_nh3(theta_deg=118.0), "sto-3g", method="RHF", conv_tol_force=1e-3)
print(f"dimer: converged={ts.converged} is_saddle={ts.is_saddle} "
      f"E = {ts.energy:.8f} Ha, curvature = {ts.curvature:.4f} Ha/bohr^2, "
      f"height = {height(ts.system):.3f} bohr, {ts.n_force_evals} force evaluations")

# 3. Confirm one imaginary frequency at the saddle.
hess = vq.compute_hessian_fd(ts.system, "sto-3g", method="RHF")
print("TS frequencies / cm^-1:", np.round(hess.frequencies_cm1, 1))

# 4. The IRC, both directions, starting along the dimer's lowest mode.
irc = vq.run_irc(ts.system, "sto-3g", method="RHF", transition_mode=ts.mode,
                 step_size=0.1, max_points=100, conv_tol_grad=1e-3)
print(f"forward: {len(irc.forward_path)} points, converged={irc.forward_converged}, "
      f"barrier {irc.barrier_forward * KCAL:.2f} kcal/mol")
print(f"reverse: {len(irc.reverse_path)} points, converged={irc.reverse_converged}, "
      f"barrier {irc.barrier_reverse * KCAL:.2f} kcal/mol")
print(f"minima reached: {irc.forward_energies[-1]:.8f} and {irc.reverse_energies[-1]:.8f} Ha, "
      f"heights {height(irc.forward_minimum):+.3f} and {height(irc.reverse_minimum):+.3f} bohr")

# 5. Energy profile against the mass-weighted arc length s.
def arc_length(path, start):
    m = np.repeat([MASS[int(a.Z)] for a in start.atoms], 3)
    xs = [np.asarray([c for a in start.atoms for c in a.xyz])]
    xs += [np.asarray([c for a in geom.atoms for c in a.xyz]) for geom in path]
    return np.cumsum([np.linalg.norm(np.sqrt(m) * (xs[k + 1] - xs[k]))
                      for k in range(len(xs) - 1)])

s_fwd = arc_length(irc.forward_path, irc.transition_state)
s_rev = arc_length(irc.reverse_path, irc.transition_state)
profile = ([(-s, e, g) for s, e, g in zip(s_rev, irc.reverse_energies, irc.reverse_path)][::-1]
           + [(0.0, irc.ts_energy, irc.transition_state)]
           + list(zip(s_fwd, irc.forward_energies, irc.forward_path)))
print(f"{'s / bohr amu^1/2':>16} {'E - E_TS / kcal/mol':>20} {'height / bohr':>14}")
for s, e, geom in profile[::6]:
    print(f"{s:16.3f} {(e - irc.ts_energy) * KCAL:20.3f} {height(geom):14.3f}")

irc.write_citations("nh3_irc")               # nh3_irc.bibtex + nh3_irc.references
```

On a vibe-qc 0.17.10 development build (Apple M5 Max): the optimisation
takes well under a second, the dimer search about 15 seconds, the
finite-difference Hessian one second and the IRC seven seconds.

```
minimum: E = -55.45541963 Ha, height = -0.806 bohr
dimer: converged=True is_saddle=True E = -55.43766745 Ha, curvature = -0.1406 Ha/bohr^2, height = -0.006 bohr, 144 force evaluations
TS frequencies / cm^-1: [-1080.9     0.      0.      0.      0.      0.      0.   1866.3  1866.5
  4022.9  4362.6  4362.8]
forward: 29 points, converged=True, barrier 11.14 kcal/mol
reverse: 30 points, converged=True, barrier 11.14 kcal/mol
minima reached: -55.45541748 and -55.45541748 Ha, heights -0.800 and +0.800 bohr
s / bohr amu^1/2  E - E_TS / kcal/mol  height / bohr
          -2.742              -11.138          0.800
          -2.622              -10.597          0.775
          -2.022               -9.805          0.678
          -1.423               -5.999          0.469
          -0.823               -1.873          0.261
           0.000                0.000         -0.006
           0.691               -2.431         -0.293
           1.291               -6.802         -0.508
           1.891              -10.065         -0.703
           2.441              -11.092         -0.778
```

(The table prints every sixth point of the 60; the full profile is in
`irc.forward_energies` and `irc.reverse_energies`.)

## What just happened

The dimer search started from a pyramid flattened to an HNH angle of
118 degrees, about halfway between the minimum (106.7) and the planar
saddle (120), rotated its two-image dimer onto the lowest-curvature
direction and walked uphill along it while relaxing everything else. It
converged in 47 outer iterations and 144 energy-and-force evaluations to
a structure that is planar to 0.006 bohr with a negative curvature along
the dimer axis: `is_saddle=True` is the combination of
`converged` and `curvature < 0`. The finite-difference Hessian confirms
the character independently: one imaginary frequency, $1081i$ cm$^{-1}$,
the umbrella mode, six zeros for translation and rotation, and five real
vibrations. Anything other than exactly one imaginary frequency means
the IRC that follows would not be a reaction path.

`run_irc` then took one step of 0.1 bohr amu$^{1/2}$ off the saddle along
$+\mathbf v$ and one along $-\mathbf v$, where $\mathbf v$ is the dimer
mode passed as `transition_mode`, and followed the mass-weighted
gradient downhill from each until the largest Cartesian force component
fell below `conv_tol_grad` = $10^{-3}$ Ha/bohr. Both branches converged,
in 29 and 30 points, to pyramids of height $\pm 0.80$ bohr with the
hydrogens on opposite sides of the nitrogen: the two enantiomeric
minima of the inversion. Their energies agree to $10^{-8}$ Ha with each
other and to $2\times10^{-6}$ Ha with the independently optimised minimum
(the IRC stops at a looser force threshold than the optimiser). The
barrier is $E_\text{TS} - E_\text{min} = 0.01775$ Ha $= 11.14$ kcal/mol
on each side, symmetric as it must be for an inversion.

The profile table is the physical content of the run. $s$ is the
cumulative mass-weighted arc length from the saddle, negative on the
reverse branch by convention; the energy decreases monotonically along
each branch (the integrator guarantees this, see the theory) and the
height column shows the hydrogens moving smoothly from one side of the
plane to the other. The path is short, about 2.6 bohr amu$^{1/2}$ per
side, because the inversion moves only light hydrogens; a path that
breaks a bond between heavy atoms runs to 5 to 10 in the same units. The
`.references` file written at the end cites Fukui 1981 and
Ishida-Morokuma-Komornicki 1977 for the path, beside the basis and SCF
references.

The number itself, 11.1 kcal/mol, is about twice the experimental
inversion barrier of 5.8 kcal/mol (2020 cm$^{-1}$). That is the
level of theory, not the path: HF has no correlation and STO-3G has no
polarisation functions on nitrogen, and both stiffen the planar
structure. [Nudged elastic band](neb_reaction_path.md) reaches the same
11 kcal/mol from the climbing image at the same level; a hybrid
functional or MP2 with a triple-zeta basis brings it to within a
kilocalorie of experiment. The IRC tells you *which* minima a saddle
connects and *how* the geometry evolves between them; the height of the
barrier is a property of the electronic-structure method.

## Variations

**Let the Hessian find the mode.** Omit `transition_mode` and `run_irc`
computes a finite-difference Hessian at the start geometry itself and
takes its lowest normal mode, warning if that mode is not clearly
imaginary ("is this really a transition state?"). This costs $6N$
gradient evaluations up front, 24 for ammonia, and reproduces the run
above point for point (barrier 11.14 kcal/mol, 31 forward points):

```python
irc_auto = vq.run_irc(ts.system, "sto-3g", method="RHF",
                      step_size=0.1, max_points=100, conv_tol_grad=1e-3)
print(irc_auto.forward_converged, irc_auto.barrier_forward * KCAL)
```

Use the explicit mode when you have one (a `DimerResult.mode`, or the
imaginary-mode vector of a Hessian you already computed), and the
automatic route when you arrive with only a geometry. `method="mace"`
and `method="ROKS"` require the explicit mode.

**One direction, finer steps.** `direction="forward"` or `"reverse"`
traces one branch; `step_size` (bohr amu$^{1/2}$, default 0.1) sets the
resolution of the profile and `max_points` caps each branch. Halving
the step doubles the cost and gives a smoother curve; it does not change
where the branch ends. A branch that hits `max_points` without
converging returns `forward_converged=False` and its last geometry is
not a minimum; raise the cap or finish with `optimize_molecule` from
`irc.forward_minimum`.

**Other references.** `method="RKS"` with `functional=` follows the IRC
on a DFT surface (the default `functional="pbe"` applies only when you
select RKS/UKS/ROKS); `method="UHF"` and `"UKS"` serve radicals. All of
them reuse the per-geometry gradient evaluators of `run_neb`, so the
Hamiltonian the IRC differentiates is the one the saddle search used.

## Theory

The reaction path is defined in **mass-weighted Cartesian coordinates**
$q_i = \sqrt{m_i}\,x_i$, in which the classical kinetic energy is
$\tfrac12\sum_i \dot q_i^2$ and the nuclear equations of motion take
their simplest form. Fukui's intrinsic reaction coordinate is the curve
$\mathbf q(s)$ that leaves the saddle point along the eigenvector of the
negative Hessian eigenvalue and thereafter follows the steepest descent,

$$
\frac{d\mathbf q}{ds} = -\frac{\mathbf g(\mathbf q)}{|\mathbf g(\mathbf q)|},
\qquad
g_i = \frac{\partial E}{\partial q_i} = \frac{1}{\sqrt{m_i}}\frac{\partial E}{\partial x_i},
$$

with $s$ the arc length in bohr amu$^{1/2}$. It is the path a classical
system would follow if it started at the saddle with an infinitesimal
push and its kinetic energy were removed at every instant: the
minimum-energy path with the physically correct weighting of light and
heavy atoms. Without the mass weighting the same steepest descent in
plain Cartesian coordinates gives a different curve, and a Cartesian
coordinate system tied to a different origin or orientation would give
yet another; the mass weighting is what makes the IRC *intrinsic*, that
is, invariant to translations, rotations and the choice of internal
coordinates.

The direct integration of the steepest-descent equation is unstable:
away from the saddle the gradient is dominated by the stiff directions
perpendicular to the path, an Euler step overshoots across the valley,
and the next step comes back, so a naive integrator zig-zags and can
climb. Ishida, Morokuma and Komornicki
({vibeqc-cite}`ishida_morokuma_komornicki_irc_1977`) stabilised the
first practical ab initio IRC by rejecting any step that raises the
energy and reducing the step length. `vibeqc.irc._irc_branch` uses that
rule: from the current point it proposes
$\Delta\mathbf q = -s\,\mathbf g/|\mathbf g|$, evaluates the energy
there, and if it has risen halves $s$ and tries again; after an accepted
step $s$ grows back by 20 percent toward its nominal value. The energy
is therefore non-increasing along every returned branch, and a branch
converges when the largest Cartesian force component drops below
`conv_tol_grad` or the step has shrunk to $10^{-5}$ of its nominal
value. The price is a factor of roughly two in gradient evaluations near
the minimum, where many proposals are rejected. The Gonzalez-Schlegel
constrained second-order integrator (C. Gonzalez and H. B. Schlegel,
*J. Chem. Phys.* **90**, 2154 (1989)) takes larger, more accurate steps
by optimising each point on a hypersphere around a pivot; it is the
standard in most production codes and is listed in the module as a
future refinement.

What an IRC proves is precise and limited. It proves that the saddle
you found connects the two specific minima at its ends, so a "one-step
mechanism" claim built on that saddle is consistent, and it gives the
minimum-energy path along which the geometry, and any property you
evaluate along it, changes between reactant and product. It does not
give dynamics: real trajectories carry kinetic energy, cut corners and
can bypass the IRC entirely when the path curves sharply or the valley
bifurcates after the saddle. It does not prove that no lower saddle
exists, and it says nothing about the barrier height beyond what the
electronic-structure method already said at the two stationary points.
A converged IRC whose end is not the minimum you expected is the single
most useful negative result a transition-state search can return.

## Resources

- [IRC and the dimer method](../user_guide/irc_and_dimer.md): keyword
  reference for `run_dimer` and `run_irc`, the result fields and the
  supported references.
- [Nudged elastic band](../user_guide/neb.md) and its
  [tutorial](neb_reaction_path.md): the chain-of-states alternative for
  finding the same saddle, with the ammonia example animated.
- [Vibrational frequencies](vibrational_frequencies.md): the Hessian
  and normal modes used above to confirm the saddle.
- [Geometry optimization](geometry_optimization.md): `optimize_molecule`
  and the convergence thresholds that define a minimum.
- `examples/workflows/input-nh3-umbrella-neb.py`: the same molecule
  through NEB, whose geometry builder this page reuses.
- [Feature matrix](../features.md) and
  [Methods: theory and status](../methods.md#geometry-optimisation-relaxed-scans-and-intrinsic-reaction-coordinates).

Cost: every IRC point is one gradient evaluation, plus one for each
rejected step; 60 points on ammonia at RHF/STO-3G are seven seconds,
and the cost scales with the gradient of the chosen method. The dimer
search costs about three gradients per iteration.

## References

- **The IRC.** K. Fukui, "The path of chemical reactions: the IRC
  approach," *Acc. Chem. Res.* **14**, 363 (1981),
  {vibeqc-cite}`fukui_irc_1981`. Defines the intrinsic reaction
  coordinate as the mass-weighted steepest-descent path.
- **Stabilised steepest descent.** K. Ishida, K. Morokuma and
  A. Komornicki, "The intrinsic reaction coordinate. An ab initio
  calculation for HNC to HCN and H$^-$ + CH$_4$ to CH$_4$ + H$^-$,"
  *J. Chem. Phys.* **66**, 2153 (1977),
  {vibeqc-cite}`ishida_morokuma_komornicki_irc_1977`. The
  reject-and-halve integrator used by `run_irc`.
- **Second-order path following.** C. Gonzalez and H. B. Schlegel,
  "An improved algorithm for reaction path following," *J. Chem. Phys.*
  **90**, 2154 (1989). The constrained integrator most codes use; a
  planned refinement here.
- **The dimer method.** G. Henkelman and H. Jonsson, "A dimer method for
  finding saddle points on high dimensional potential surfaces using
  only first derivatives," *J. Chem. Phys.* **111**, 7010 (1999),
  {vibeqc-cite}`henkelman_jonsson_dimer_1999`. The saddle search that
  produced the transition state and its mode.
- **Ammonia inversion barrier.** J. D. Swalen and J. A. Ibers,
  "Potential function for the inversion of ammonia," *J. Chem. Phys.*
  **36**, 1914 (1962). The experimental 2020 cm$^{-1}$ barrier quoted
  above.

## Next

- [Nudged elastic band](neb_reaction_path.md) finds the same saddle
  from the two minima instead of from a guess near the top.
- [Vibrational frequencies](vibrational_frequencies.md) and
  [Thermodynamics](thermodynamics.md) turn the stationary points at the
  ends of the IRC into free-energy barriers.
- [Relaxed PES scans](relaxed_pes_scan.md) give a one-coordinate
  approximation to the path when a full IRC is too expensive.
