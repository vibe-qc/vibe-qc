---
myst:
  html_meta:
    "description": "Reference for transition-state searches and reaction paths in vibe-qc: the dimer method (run_dimer, geom_opt='dimer'), eigenvector following and partitioned RFO (geom_opt='ef' or 'prfo'), Bofill Hessian updates, and the intrinsic reaction coordinate (run_irc), with theory, options, results, verified examples and every gate."
    "og:title": "vibe-qc user guide - transition-state searches and intrinsic reaction coordinates"
---

# Transition-state searches: the dimer method, eigenvector following and the IRC

A first-order saddle point is a geometry with zero gradient and exactly one
negative Hessian eigenvalue. vibe-qc finds one from a single starting geometry
with the dimer method (`run_dimer`, or `geom_opt="dimer"` in `run_job`), which
uses forces only, or with eigenvector following and partitioned rational
function optimisation (`geom_opt="ef"` or `"prfo"`), which build an approximate
Hessian. Once you have a saddle, `run_irc` follows the intrinsic reaction
coordinate downhill in both directions to the two minima it connects. All of
these are molecular routes on RHF, UHF, ROHF, RKS, UKS and ROKS references and
the MACE potential. The dimer and the IRC are production for molecules and the
periodic IRC fails closed. The dimer also runs on periodic systems (BIPOLE
with finite-difference forces). Eigenvector following and P-RFO are selectable
but were not reliable in the trials described under
[Choosing a search](#choosing-a-search). The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
Chain-of-states searches are in the [NEB guide](neb.md).

The tutorial [Following the IRC from a transition state](../tutorial/irc_from_transition_state.md)
works the ammonia inversion end to end; this page is the reference.

## Quick start

The HCN to HNC isomerisation at RHF/STO-3G: dimer search from a guess with the
hydrogen bridging carbon and nitrogen, a frequency check, then the IRC.

```python
import time
import numpy as np
import vibeqc as vq

ANG = 1.8897259886
KCAL = 627.5094740631

# A guess for the transition state with the H atom bridging C and N.
guess = vq.Molecule([
    vq.Atom(6, [0.0, 0.0, 0.0]),
    vq.Atom(7, [1.16 * ANG, 0.0, 0.0]),
    vq.Atom(1, [0.346 * ANG, 1.139 * ANG, 0.0]),
], 0, 1)
direction = np.zeros((3, 3))
direction[2, 0] = 1.0                      # the H atom migrates along x, from C toward N

def bonds(system):
    p = np.array([a.xyz for a in system.atoms]) / ANG          # Angstrom
    d = lambda i, j: np.linalg.norm(p[i] - p[j])
    return f"C-N {d(0, 1):.3f}  C-H {d(0, 2):.3f}  N-H {d(1, 2):.3f} A"

t0 = time.perf_counter()
ts = vq.run_dimer(guess, "sto-3g", method="RHF", initial_direction=direction,
                  conv_tol_force=1e-3)
print(f"dimer: converged={ts.converged} is_saddle={ts.is_saddle} E = {ts.energy:.8f} Ha, "
      f"curvature = {ts.curvature:.4f} Ha/bohr^2, {ts.n_iter} iterations, "
      f"{ts.n_force_evals} force evaluations, {time.perf_counter() - t0:.0f} s")
print("saddle:", bonds(ts.system))

hess = vq.compute_hessian_fd(ts.system, "sto-3g", method="RHF")
print("lowest frequencies / cm^-1:", np.round(hess.frequencies_cm1[:3], 1))

t0 = time.perf_counter()
irc = vq.run_irc(ts.system, "sto-3g", method="RHF", transition_mode=ts.mode,
                 step_size=0.1, max_points=150, conv_tol_grad=1e-3)
print(f"IRC: {time.perf_counter() - t0:.0f} s")
for name, path, energies, conv, barrier, end in (
        ("forward", irc.forward_path, irc.forward_energies, irc.forward_converged,
         irc.barrier_forward, irc.forward_minimum),
        ("reverse", irc.reverse_path, irc.reverse_energies, irc.reverse_converged,
         irc.barrier_reverse, irc.reverse_minimum)):
    print(f"{name}: {len(path)} points, converged={conv}, E_end = {energies[-1]:.8f} Ha, "
          f"barrier {barrier * KCAL:.2f} kcal/mol, end: {bonds(end)}")
irc.write_citations("hcn_irc")             # hcn_irc.bibtex + hcn_irc.references
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) on an idle machine the dimer takes about 25 s,
the Hessian a second and the IRC about a minute (longer when the machine is busy):

```text
dimer: converged=True is_saddle=True E = -91.56485757 Ha, curvature = -0.1529 Ha/bohr^2, 29 iterations, 90 force evaluations, 25 s
saddle: C-N 1.222  C-H 1.204  N-H 1.432 A
lowest frequencies / cm^-1: [-1243.4     0.      0.]
IRC: 60 s
forward: 89 points, converged=True, E_end = -91.64443121 Ha, barrier 49.93 kcal/mol, end: C-N 1.170  C-H 2.182  N-H 1.011 A
reverse: 43 points, converged=True, E_end = -91.67520654 Ha, barrier 69.25 kcal/mol, end: C-N 1.153  C-H 1.070  N-H 2.223 A
```

Reading it. `is_saddle` is `converged and curvature < 0`: the dimer axis lies along a
direction of negative curvature and the maximum force is below 1e-3 Ha/bohr. The
finite-difference Hessian confirms exactly one imaginary mode ($1243i$
cm$^{-1}$), the requirement for a first-order saddle. The IRC's two ends are HNC
(N-H 1.011 Å) and HCN (C-H 1.070 Å, C-N 1.153 Å). Which branch is `forward`
depends only on the sign of the mode vector: `forward` follows $+$`transition_mode`,
and the dimer's mode has an arbitrary overall sign, so here `forward` is HNC.
Identify the ends by geometry, not by name. HNC lies 19.3 kcal/mol above HCN at
this level and the barrier is 49.9 kcal/mol from HNC and 69.3 from HCN.
Hartree-Fock with a minimal basis overestimates the barriers; the point is the
connectivity of the path, which the run proves.

## Choosing a search

| Route | Needs | Use it when |
|---|---|---|
| `run_dimer` / `geom_opt="dimer"` | forces only; an initial direction | You have a rough geometry and a guess of the reaction direction. The one route that converged in the trials here, provided the initial direction is sensible (without one, the default direction failed on HCN to HNC: 200 iterations, positive curvature, `is_saddle=False`). |
| `geom_opt="ef"`, `"prfo"` | gradient; ideally an exact initial Hessian | You are already close to the saddle and a good Hessian is at hand. Not reliable from a rough start (below). |
| `run_neb` with a climbing image | both endpoints | You know reactant and product and want the path as well as the saddle. See [NEB](neb.md). |
| `run_irc` | a converged saddle | You have a saddle and want to know which minima it links. |

```{admonition} Status: preview
:class: note

Eigenvector following and P-RFO (`geom_opt="ef"` and `geom_opt="prfo"`) are maintained but not reliable from an
approximate start in the current build. Through `run_job` with the default diagonal
initial Hessian and `geom_hessian_update="bofill"`, both searches failed on the two
test systems tried. On HCN to HNC, EF from a geometry 0.08 Å from the converged
dimer saddle ended after 200 iterations with 191 rejected trust-radius steps and a
maximum gradient of 4.9e-3 Ha/bohr; the `.out` step table shows the `[rejected]`
markers but no message says the search did not converge, and the final SCF is
printed as usual. On the ammonia inversion from an HNH angle of 118 degrees, EF and
P-RFO both stepped to a geometry where the SCF did not converge and raised
`RuntimeError` ("SCF did not converge after 100 iterations during geometry
optimization"). With an exact initial Hessian passed as `hessian_provider` to
`vibeqc.geomopt.run_geomopt`, EF and P-RFO on ammonia still did not converge in 60
iterations and returned `is_saddle=None`. The dimer converged on ammonia through
`run_job` (energy -55.43766582 Ha, 29 s). Check the step table and
`GeomOptResult.converged` and `is_saddle` before you trust an EF or P-RFO result.
See [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

The same searches through `run_job`, where the `vibeqc.geomopt` framework wraps
the optimiser and prints its step table into the `.out`:

```python
import numpy as np
from vibeqc import Atom, Molecule, run_job

ANG = 1.8897259886
# NH3 slightly flattened from the pyramid (HNH 118 degrees); N at the origin, H below it.
theta, r = np.deg2rad(118.0), 1.012 * ANG
sa = np.sqrt(2.0 * (1.0 - np.cos(theta)) / 3.0)
ca = np.sqrt(1.0 - sa ** 2)
atoms = [Atom(7, [0.0, 0.0, 0.0])] + [
    Atom(1, [r * sa * np.cos(2 * np.pi * k / 3), r * sa * np.sin(2 * np.pi * k / 3), -r * ca])
    for k in range(3)
]
nh3 = Molecule(atoms, 0, 1)

direction = np.zeros((4, 3))
direction[1:, 2] = 1.0                       # the three H atoms move together along z

result = run_job(
    nh3, basis="sto-3g", method="rhf", optimize=True,
    geom_opt="dimer", geom_target="transition_state", fmax=0.02,
    geom_opt_options={"initial_direction": direction},
    output="nh3_dimer",
)
print(result.energy)        # -55.43766582 Ha, the planar saddle
```

The planar saddle comes out at $-55.43766582$ Ha in about 30 s; the tutorial's
tighter `run_dimer` search gives $-55.43766745$ Ha. `geom_target="transition_state"`
selects the saddle mode; `geom_opt="dimer"` requires it and raises `ValueError`
under `geom_target="minimum"`. With a transition-state target and any other
optimiser name the framework substitutes `"ef"`.

## Theory

**The dimer method** ({vibeqc-cite}`henkelman_jonsson_dimer_1999`) finds a saddle
from forces alone. A pair of images, one at the centre $\mathbf R$ and one at
$\mathbf R+\Delta\,\mathbf N$ with $|\mathbf N|=1$ and $\Delta=$`dimer_separation`
(0.01 bohr), samples the local curvature along $\mathbf N$ without a Hessian,

$$
C(\mathbf N)\approx\frac{(\mathbf F_2-\mathbf F_1)\cdot\mathbf N}{2\Delta},
$$

where $\mathbf F_1$ is the force at the displaced image and $\mathbf F_2=2\mathbf F_0-\mathbf F_1$ the
force at its mirror image. Each outer iteration has two phases. In the
**rotation** phase the axis is turned to minimise $C$, that is, to align it with the
lowest-curvature eigenvector of the Hessian: the rotational force
$\mathbf F_\perp=(\mathbf F_1-\mathbf F_2)_\perp$ gives the direction and the optimal angle is found
analytically from one trial rotation, on the curvature model
$C(\phi)=C_0-a_1+a_1\cos2\phi+b_1\sin2\phi$. In the **translation** phase the dimer
centre moves on the effective force

$$
\mathbf F_\text{eff}=\begin{cases}
\mathbf F_0-2\,\mathbf F_\parallel & C<0\quad\text{(climb along }\mathbf N\text{, relax in the rest)}\\[2pt]
-\mathbf F_\parallel & C>0\quad\text{(still convex: drive toward negative curvature)}
\end{cases}
\qquad \mathbf F_\parallel=(\mathbf F_0\cdot\mathbf N)\,\mathbf N,
$$

which inverts the force along the axis so that the saddle becomes an attractor. The
step follows a quick-min velocity update capped at `max_step`; translations of the
centre of mass are projected out unless atoms are frozen. The search stops when
$\max|\mathbf F_0|<$`conv_tol_force`. Each iteration costs about three force
evaluations (ammonia: 47 iterations, 144 evaluations; HCN to HNC: 29 and 90). If the
curvature stays positive the run ended in a minimum or has not found the mode:
`is_saddle=False`, so try another `initial_direction`. The geomopt wrapper can
also accelerate the translation with an L-BFGS two-loop recursion
(`lbfgs_acceleration=True`; Kästner and Sherwood, *J. Chem. Phys.* **128**, 014106,
2008).

**Eigenvector following and P-RFO** need a Hessian model $\mathbf B$ (initially
diagonal, or exact from a `hessian_provider`). Diagonalise $\mathbf B$; for the
followed eigenpair $(\lambda_i,\mathbf v_i)$ solve the $2\times2$ rational-function
(augmented Hessian) problem
$\begin{pmatrix}\lambda_i & g_i\\ g_i & 0\end{pmatrix}$, $g_i=\mathbf g\cdot\mathbf v_i$
({vibeqc-cite}`banerjee_rfo_1985`), take a Newton step $-g_j/\lambda_j$ in every other
eigendirection, and scale the total step to the trust radius. For a transition state
the followed mode is the lowest one, which is *maximised*, while the remaining modes are
*minimised* ({vibeqc-cite}`baker_ts_1986`). A step is accepted by Powell's ratio of
actual to predicted energy change, and the trust radius shrinks or grows accordingly.
After every accepted step the model is updated with the secant pair
$\mathbf s=\Delta\mathbf x$, $\mathbf y=\Delta\mathbf g$ by the **Bofill** formula (J. M.
Bofill, *J. Comput. Chem.* **15**, 1 (1994)), a weighted mixture of the Powell
symmetric-Broyden (PSB) and symmetric rank-one (SR1) updates that tolerates the
indefinite Hessian of a saddle search:

$$
\mathbf B'=\varphi\,\mathbf B_\text{PSB}+(1-\varphi)\,\mathbf B_\text{SR1},\qquad
\varphi=\frac{(\mathbf r\cdot\mathbf s)^2}{|\mathbf r|^2|\mathbf s|^2},\quad
\mathbf r=\mathbf y-\mathbf B\mathbf s.
$$

`geom_hessian_update` accepts `"none"`, `"bfgs"`, `"sr1"`, `"powell"`, `"bofill"` and
`"murtagh_sargent"`; `run_job` defaults it to `"none"`, so set `"bofill"` for a saddle
search.

**The intrinsic reaction coordinate** ({vibeqc-cite}`fukui_irc_1981`) is the
steepest-descent path from the saddle in mass-weighted coordinates
$\mathbf q=\mathbf M^{1/2}\mathbf x$,

$$
\frac{d\mathbf q}{ds}=-\frac{\mathbf g_q}{|\mathbf g_q|},\qquad
\mathbf g_q=\mathbf M^{-1/2}\,\frac{\partial E}{\partial\mathbf x},
$$

with $s$ the arc length in bohr amu$^{1/2}$. It starts from the saddle along
$\pm\mathbf v$, the Cartesian transition vector, and ends at the minima the saddle
connects. A plain Euler step of this equation zig-zags across the valley, so
`run_irc` uses the damped, stabilised steepest descent of Ishida, Morokuma and
Komornicki ({vibeqc-cite}`ishida_morokuma_komornicki_irc_1977`): every step moves
$s$ along the unit mass-weighted gradient; a step that would raise the energy is
rejected and $s$ halved; an accepted step grows $s$ by 20 percent back to its nominal
value. The energy is therefore non-increasing along a returned branch. A branch
converges when the largest Cartesian force component is below `conv_tol_grad` or the
step has shrunk to $10^{-5}$ of its nominal size. The constrained second-order
integrator of Gonzalez and Schlegel (*J. Chem. Phys.* **90**, 2154 (1989)) is not
implemented.

An IRC proves which two minima a saddle links and gives the minimum-energy path
between them. It does not give dynamics, it does not prove there is no lower saddle,
and it says nothing about the barrier height beyond what the electronic-structure
method says at the stationary points.

## Options

`run_dimer(initial, basis=None, *, method="RKS", functional="pbe", initial_direction=None, dimer_separation=0.01, max_iter=200, conv_tol_force=1e-3, max_step=0.1, initial_step=0.05, n_rotations=4, rotation_force_tol=1e-3, freeze_indices=None, ...)`

| Argument | Meaning |
|---|---|
| `initial` | A `Molecule` or a `PeriodicSystem`. |
| `method` | `"RHF"`, `"UHF"`, `"ROHF"`, `"RKS"`, `"UKS"`, `"ROKS"` or `"MACE"` (case-insensitive). `functional` applies to the KS methods only. |
| `initial_direction` | `(n_atoms, 3)` guess of the reaction mode; the dimer rotates it to the nearest lowest-curvature mode. `None` uses a fixed pseudo-random seed, which is reproducible but rarely finds the mode you want. Pass a guess. |
| `dimer_separation` | Half-distance of the two images, bohr. |
| `conv_tol_force` | Maximum true force component, Ha/bohr. |
| `max_step`, `initial_step` | Translation step controls, bohr. |
| `n_rotations`, `rotation_force_tol` | Maximum rotations per translation and the rotational-force tolerance. |
| `freeze_indices` | Atoms held fixed (for example the substrate of a slab). |
| `kpoints` | Monkhorst-Pack mesh for a periodic system. |
| `rhf_options`, `uhf_options`, `rks_options`, `uks_options`, `rohf_options`, `roks_options`, `gradient_options`, `grid_options`, `grid_level`, `dispersion_params`, `mlip_options`, `fd_step_bohr`, `dft_plus_u` | Forwarded to the per-geometry evaluators shared with `run_neb`. |
| `progress` | `True` prints one line per iteration. |

`DimerResult` has `system`, `energy`, `max_force`, `curvature` (Ha/bohr$^2$; negative at a
saddle), `mode` (unit vector), `converged`, `is_saddle`, `n_iter`, `n_force_evals`, and
`write_citations(stem)`.

`run_irc(transition_state, basis=None, *, method="RKS", functional="pbe", transition_mode=None, direction="both", step_size=0.1, max_points=100, conv_tol_grad=1e-3, ..., hessian_step_bohr=0.005, fd_step_bohr=1e-3)`

| Argument | Meaning |
|---|---|
| `transition_mode` | `(n_atoms, 3)` transition vector (a `DimerResult.mode`, or an imaginary-mode vector from your own Hessian). `None` computes an FD Hessian at the start geometry and takes its lowest mode; a `RuntimeWarning` is issued if that mode is not clearly imaginary. Required for `method="MACE"` and `"ROKS"`. |
| `direction` | `"both"`, `"forward"` ($+$mode) or `"reverse"` ($-$mode). |
| `step_size` | Mass-weighted step, bohr amu$^{1/2}$. Halving it doubles the cost and smooths the profile; it does not change where the branch ends. |
| `max_points`, `conv_tol_grad` | Points per branch and the force threshold. A branch that reaches `max_points` unconverged returns `*_converged=False`; its last geometry is not a minimum, so raise the cap or finish with a geometry optimisation. |

`IRCResult` has `transition_state`, `ts_energy`, `forward_path` and `reverse_path`
(lists of `Molecule`, ordered from the first point off the saddle to the end),
`forward_energies`, `reverse_energies`, `forward_converged`, `reverse_converged`,
`forward_minimum`, `reverse_minimum`, `barrier_forward`, `barrier_reverse` (Hartree, from the
saddle energy to the branch end), and `write_citations(stem)`.

In `run_job` the keywords are `optimize=True`, `geom_opt` (`"dimer"`, `"ef"`, `"prfo"`,
`"trust"` are the saddle-capable optimisers), `geom_target="transition_state"`,
`geom_hessian_init`, `geom_hessian_update`, `geom_freeze` and `geom_opt_options`. The
dimer accepts `initial_direction`, `dimer_separation` (0.01), `max_step` (0.1),
`translation_step` or `initial_step` (0.05), `rotation_max_iter` or `n_rotations` (4),
`rotation_force_tol` (1e-3), `lbfgs_acceleration` (`False`) and `lbfgs_memory` (10);
its convergence threshold is the gradient threshold of the run (`fmax`). Without
`initial_direction` the `run_job` wrapper starts along the negative gradient at the
start geometry.

## Output

`run_dimer` and `run_irc` return result objects and write nothing by themselves except
through `write_citations`. That method writes `<stem>.bibtex` and `<stem>.references`
with Henkelman and Jónsson 1999 (dimer) or Fukui 1981 and Ishida, Morokuma and
Komornicki 1977 (IRC), next to the method, basis and functional entries (or the MACE
model papers). `run_job(geom_opt=...)` writes the usual `.out`, `.system`, `.xyz` and,
with `output_qvf=True`, a QVF trajectory; the `.out` carries the step table and the
route line `Geometry optimization (geomopt/<name>)`. To draw an IRC profile, use
`forward_energies`, `reverse_energies` and the cumulative mass-weighted distance
between successive geometries, as the tutorial does.

## Limits and gates

- **Periodic**: `run_irc` accepts only a `Molecule` and raises `NotImplementedError`
  for a `PeriodicSystem` (periodic IRC is not implemented). `run_dimer` runs on a
  `PeriodicSystem` for RHF, UHF, RKS, UKS and MACE with the NEB periodic evaluator
  (BIPOLE with finite-difference gradients); periodic ROHF and ROKS raise
  `NotImplementedError`.
- **Methods**: anything outside `RHF`, `UHF`, `ROHF`, `RKS`, `UKS`, `ROKS`, `MACE`
  raises `ValueError`. `run_irc` with `ROKS` and no `transition_mode` raises
  `NotImplementedError` (the FD Hessian needs an analytic ROKS gradient that does not
  exist). A basis name is required except for MACE. MACE with `dft_plus_u` raises
  `ValueError`; ROHF or ROKS with `dft_plus_u` raises `NotImplementedError`.
- **ECPs**: ROHF and ROKS dimer and IRC gradients refuse molecular ECP operators
  (`NotImplementedError`); RHF, UHF, RKS and UKS carry the ECP along the path.
- **Inputs**: `direction` must be `"both"`, `"forward"` or `"reverse"`.
  `transition_mode` must be a finite real array of the right shape with nonzero norm
  (`ValueError`). `freeze_indices` out of range raises `ValueError`. A zero
  `initial_direction` raises `ValueError`.
- **Reference**: a search finds the saddle of the surface it differentiates. Dispersion
  and solvent enter only where the evaluator accepts them (`dispersion_params`); check
  the surface against the one used for the minima.
- **Confirm the character**: a converged dimer run with `is_saddle=True` and one
  imaginary frequency from `compute_hessian_fd` is a first-order saddle; anything else
  is not. Run the IRC before you claim a mechanism.
- **Geometry-optimiser gates** apply to the `run_job` route: `geom_opt=` is refused for
  semiempirical and MLIP methods (use the standalone functions), and
  `geom_target` must be `"minimum"` or `"transition_state"`.

## See also

- [Following the IRC from a transition state](../tutorial/irc_from_transition_state.md):
  the ammonia inversion, with the arc-length profile.
- [Nudged elastic band](neb.md) and the
  [NEB tutorial](../tutorial/neb_reaction_path.md): chain-of-states searches from two
  endpoints.
- [Geometry optimization](geometry_optimization.md): the optimiser family, coordinate
  systems and convergence policy.
- [Vibrational analysis](vibrational_analysis.md): the Hessian that confirms one
  imaginary mode, and thermochemistry for free-energy barriers.
- [Relaxed scans](relaxed_scan.md): a one-coordinate approximation to the path.
- [Citations](citations.md), and [Methods: theory and status](../methods.md).
