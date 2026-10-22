# Solvation in water (CPCM)

For most "real chemistry", drug binding, organic reactions, redox
potentials, p$K_a$, the gas-phase electronic structure is the easy
part. Putting the molecule in solvent moves energies by tens of
kcal/mol and routinely flips conformer rankings. vibe-qc v0.9.0
ships CPCM (the conductor-screening Polarisable Continuum Model;
Cossi-Rega-Scalmani-Barone 2003) wired through `run_job` with one
keyword:

```python
result = vq.run_job(mol, basis="def2-svp", method="rks",
                    functional="b3lyp", solvent="water")
```

This tutorial walks through three workflows you'll actually use:

1. **Single-point hydration energy**, how much does dissolving a
   neutral molecule in water lower its energy?
2. **Geometry optimisation in solvent**, what does the structure
   relax to with the surrounding water turned on?
3. **Dielectric scan**, how much of the answer is "water"
   specifically vs "polar medium" generally?

We use formaldehyde (H₂C=O) throughout, small enough to converge
in seconds, polar enough that solvation is a non-trivial perturbation.

CPCM and COSMO energies and analytic gradients are **production** for RHF, UHF,
RKS and UKS (label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)).
Two families of combinations are refused rather than run, and the
[Theory](#theory) section at the end says why:

```{admonition} Status: gated
:class: caution

Implicit solvation composes with the mean-field SCF reference only. Post-SCF
correlation (`mp2`, `ccsd`, DLPNO and the coupled-pair variants) fails closed
with `ValueError: Implicit solvation is not supported for method='mp2':
CPCM composes with the mean-field SCF reference only; the post-SCF mp2
treatment in solvent is not implemented`, because the correlation step's
reported total would not be composed against the in-solvent reference.
Restricted open-shell methods fail closed too: `method="rohf"` and
`method="roks"` with a `solvent=` raise `NotImplementedError: Implicit
solvation (CPCM) with method='rohf' is not yet supported`. Run these in gas
phase, or use `rhf`, `uhf`, `rks` or `uks`. See
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## 1. Single-point hydration energy

The simplest CPCM calculation: gas-phase SCF, then in-solvent SCF,
take the difference.

```python
import vibeqc as vq

# Formaldehyde, Å → bohr (vibe-qc internal units).
ANG = 1.8897261339213
mol = vq.Molecule([
    vq.Atom(6, [ 0.0,  0.000,  0.0000 * ANG]),   # C
    vq.Atom(8, [ 0.0,  0.000,  1.2055 * ANG]),   # O
    vq.Atom(1, [ 0.0,  0.940, -0.5870 * ANG]),   # H
    vq.Atom(1, [ 0.0, -0.940, -0.5870 * ANG]),   # H
], 0, 1)

# Gas-phase reference.
gas = vq.run_job(mol, basis="def2-svp", method="rks", functional="pbe0",
                 output="ch2o_gas", progress=False)

# In water.
aqueous = vq.run_job(mol, basis="def2-svp", method="rks", functional="pbe0",
                     output="ch2o_water", solvent="water", progress=False)

sol = aqueous.solvent_result
delta_g_solv_au = sol.energy - gas.energy
delta_g_solv_kcal = delta_g_solv_au * 627.5094740631

print(f"E(gas)            = {gas.energy:12.6f} Ha")
print(f"E(water, total)   = {sol.energy:12.6f} Ha")
print(f"E_solv (½ q · V)  = {sol.e_solv:10.6f} Ha = {sol.e_solv * 627.5094740631:+8.3f} kcal/mol")
print(f"ΔG_solv (total)   = {delta_g_solv_au:10.6f} Ha = {delta_g_solv_kcal:+8.3f} kcal/mol")
print(f"CPCM macro-iters  = {sol.n_macro_iter}")
print(f"Cavity points     = {sol.cavity.n_points}")
print(f"Solvent           = {sol.solvent_name}  (ε = {sol.epsilon:.3f})")
```

Output (PBE0/def2-SVP, vibe-qc 0.17.10 development build):

```
E(gas)            =  -114.283604 Ha
E(water, total)   =  -114.290172 Ha
E_solv (½ q · V)  =  -0.007985 Ha =   -5.011 kcal/mol
ΔG_solv (total)   =  -0.006568 Ha =   -4.122 kcal/mol
CPCM macro-iters  = 7
Cavity points     = 1047
Solvent           = water  (ε = 78.390)
```

A few notes on what you're looking at:

* **`E_solv = ½ q · V`** is the *polarisation energy*, the
  stabilisation from the surrounding dielectric pulling charge
  toward the cavity surface. For a neutral polar molecule like
  formaldehyde it's a few kcal/mol; for a charged species (e.g. an
  acetate anion) it's tens to hundreds of kcal/mol.
* **`ΔG_solv` < `E_solv` by ~0.9 kcal/mol.** ``E_solv`` is the
  polarisation contribution evaluated at the in-solvent density;
  ``ΔG_solv`` is the *total* in-solvent energy minus the gas-phase
  reference. The ~20 % gap is the **electronic-reorganisation
  cost**, the gas-phase HF/KS energy at the in-solvent density is
  slightly higher than at the gas-phase density, which cancels
  part of the ½ q·V stabilisation. The two are only equal in the
  limit of vanishing density response; for most polar solutes the
  SCF density does shift visibly in solvent (it's the whole point
  of running the macro-iteration), so the two quantities should
  differ by 10-30 %.
* **CPCM macro-iters** is the outer self-consistency cycle (apparent
  surface charge ↔ SCF density). Three to ten iterations is
  typical; if it goes above ~15, either the cavity is pathologic
  (atoms inside each other) or the molecule is so polarisable that
  the linear-response model breaks down.

The literature value for formaldehyde's hydration free energy
(MNSol database) is **−1.7 kcal/mol**, our pure-electrostatic CPCM
gives −4.1 kcal/mol. The ~2.4 kcal/mol gap is the *non-electrostatic*
contribution (cavitation + dispersion + repulsion), which a full
SMD model adds on top. SMD-style non-electrostatic terms are
typically positive for small neutral solutes (cavity-formation cost
outpaces solute-solvent dispersion gain), shifting the CPCM-only
overshoot back toward the experimental value. vibe-qc does not yet include
those terms; pure CPCM is the electrostatic backbone everyone
builds on.

## 2. Geometry optimisation in solvent

Solvent doesn't just shift energies, it changes the equilibrium
geometry. For formaldehyde the C=O bond lengthens in water (the
solvent stabilises the polarised C^δ⁺=O^δ⁻ resonance form). Let's
see by how much.

```python
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.optimize import BFGSLineSearch
from ase.units import Bohr

from vibeqc.ase import VibeQC

def build_atoms():
    return Atoms(
        numbers=[6, 8, 1, 1],
        positions=[
            [ 0.0,  0.000,  0.0000],
            [ 0.0,  0.000,  1.2055],
            [ 0.0,  0.940, -0.5870],
            [ 0.0, -0.940, -0.5870],
        ],
    )

# Gas-phase optimisation.
atoms_gas = build_atoms()
atoms_gas.calc = VibeQC(basis="def2-svp", functional="pbe0")
with Trajectory("ch2o_gas.traj", "w", atoms_gas) as traj:
    opt = BFGSLineSearch(atoms_gas, logfile=None)
    opt.attach(traj)
    opt.run(fmax=0.01)

# CPCM-water optimisation. The ``solvent="water"`` keyword routes
# the SCF + forces through vibeqc.solvation; the analytic CPCM
# gradient is used automatically.
atoms_aq = build_atoms()
atoms_aq.calc = VibeQC(basis="def2-svp", functional="pbe0",
                       solvent="water")
with Trajectory("ch2o_water.traj", "w", atoms_aq) as traj:
    opt = BFGSLineSearch(atoms_aq, logfile=None)
    opt.attach(traj)
    opt.run(fmax=0.01)

def r_co(atoms):
    return float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))

print(f"r(C=O) gas    : {r_co(atoms_gas):.4f} Å")
print(f"r(C=O) water  : {r_co(atoms_aq):.4f} Å")
print(f"Δr(C=O)       : {r_co(atoms_aq) - r_co(atoms_gas):+.4f} Å")
```

Result (PBE0/def2-SVP, BFGSLineSearch fmax=0.01). Re-run on the 0.17.10
development build (the two optimisations together took over an hour on a
heavily loaded shared machine):

```
r(C=O) gas    : 1.1961 Å
r(C=O) water  : 1.2025 Å
Δr(C=O)       : +0.0065 Å
```

(The v0.9.0 run gave 1.2027 Å in water, a difference of 0.2 mÅ.) A ~0.65 pm elongation, small in absolute terms, but the right sign
and a reproducible solvent effect that matches the literature
(experimental gas-phase: 1.2078 Å; aqueous IR/Raman: ~1.21 Å). The
solvent stabilises the carbonyl's polarisation, weakening the π
bond slightly.

### A note on optimisation cost

The CPCM-water optimisation runs **a few× slower per step** than the
gas-phase one, almost entirely because of the SCF: each macro-
iteration of CPCM runs a full SCF, and 3-7 macro-iterations are
needed per energy evaluation.

The gradient itself is cheap and fully analytic. On top of the
standard analytic SCF gradient at the in-solvent density, the CPCM
solvation gradient adds three closed-form pieces, the cavity
A-matrix derivative, the nuclear-ESP derivative, and the
electronic-ESP derivative (one libint nuclear-attraction-gradient
pass over the cavity points). No geometry rebuilds, no
finite-difference step error.

For routine work with `n_atoms < 50` and a coarser tessellation than the
default (`n_points_per_sphere = 110` instead of the default 302),
CPCM-aware optimisations are tractable on a laptop.

## 3. Dielectric scan, water vs solvents you've heard of

How much of solvent stabilisation is "water" and how much is "polar
dielectric"? Scan `ε` from gas (1.0) through hexane (1.9) up to water
(78.4):

```python
import vibeqc as vq

mol = ...  # formaldehyde, as above

solvents = [
    ("vacuum",       1.0),
    ("n-hexane",     1.88),
    ("toluene",      2.37),
    ("chloroform",   4.71),
    ("thf",          7.43),
    ("dichloromethane", 8.93),
    ("acetone",     20.49),
    ("ethanol",     24.85),
    ("methanol",    32.61),
    ("acetonitrile", 35.94),
    ("dmso",        46.83),
    ("water",       78.39),
]

print(f"{'solvent':<20} {'ε':>8} {'E_solv (kcal/mol)':>18}")
print("-" * 48)
for name, _eps in solvents:
    result = vq.run_job(mol, basis="def2-svp", method="rks",
                        functional="pbe0", solvent=name,
                        output=f"ch2o_{name}", progress=False)
    sol = result.solvent_result
    e_kcal = sol.e_solv * 627.5094740631
    print(f"{name:<20} {sol.epsilon:>8.3f} {e_kcal:>18.3f}")
```

Scan (PBE0/def2-SVP, formaldehyde). A re-run on the 0.17.10 development
build reproduced every row below to the last digit:

```
solvent                     ε  E_solv (kcal/mol)
------------------------------------------------
vacuum                  1.000              0.000
n-hexane                1.880             -1.916
toluene                 2.370             -2.471
chloroform              4.710             -3.669
thf                     7.430             -4.166
dichloromethane         8.930             -4.316
acetone                20.490             -4.752
ethanol                24.850             -4.813
methanol               32.610             -4.881
acetonitrile           35.940             -4.902
dmso                   46.830             -4.948
water                  78.390             -5.011
```

A textbook **`f(ε) = (ε − 1)/ε`** curve, the conductor screening
factor. By ε ≈ 20 (acetone) the solvation energy is already at
~95 % of its conductor-limit value; water adds only ~0.25 kcal/mol
beyond that. This is the famous CPCM observation that "any
reasonably polar solvent looks the same as water" for purely
electrostatic stabilisation. Where solvents actually differ is in
hydrogen-bonding, explicit-water structure, and dispersion, all
beyond pure CPCM.

## What "solvent='water'" expands to

The single keyword `solvent="water"` is shorthand for:

```python
sm = vq.SolventModel(
    epsilon=78.39,                  # IUPAC static ε of liquid H₂O at 298 K
    name="water",
    variant="cpcm",                 # (ε − 1)/ε factor (vs "cosmo": (ε − 1)/(ε + 0.5))
    radii_scale=1.20,               # Bondi vdW × 1.20 (PCM/GEPOL convention)
    solvent_probe_radius_ang=0.0,   # scaled-vdW cavity (not SAS)
    cavity="lebedev",               # union of switched Lebedev spheres (default)
    n_points_per_sphere=302,        # Lebedev order 29 (Gaussian default)
    switching_sigma_bohr=0.5,       # Scalmani-Frisch CSC smoothing
    max_macro_iter=30,
    tol_e_solv=1e-6,
)
result = vq.run_job(mol, ..., solvent=sm)
```

Pass a numeric ε for a custom solvent, or override any field via the
dict form `solvent={"epsilon": 25.0, "radii_scale": 1.25}`. See the
[user-guide chapter](../user_guide/solvation.md) for the full
parameter reference.

## Inspecting the cavity

For a sanity check or a figure, dump the cavity surface points to
XYZ:

```python
import numpy as np
from vibeqc.solvation import build_cavity

pos = np.array([list(a.xyz) for a in mol.atoms])
Zs  = [a.Z for a in mol.atoms]
cav = build_cavity(pos, Zs, n_points_per_sphere=302)

# Dump as XYZ with the apparent charge in column 5 (visualise in PyMOL
# / VMD by colour-mapping the 5th column to a charge gradient).
with open("ch2o_cavity.xyz", "w") as f:
    f.write(f"{cav.n_points}\nFormaldehyde CPCM cavity, scaled vdW\n")
    for i in range(cav.n_points):
        x, y, z = cav.points[i] / ANG    # Å for PyMOL / VMD
        f.write(f"X  {x:12.6f}  {y:12.6f}  {z:12.6f}  {cav.weights[i]:12.6f}\n")
```

You'll see ~900-1100 points draped over the four atomic spheres with
the heavily-buried inter-atomic regions smoothly cut by the
switching function.

## Theory

**Conductor-like screening.** A continuum model replaces the solvent by a
dielectric of constant $\varepsilon$ outside a cavity that holds the solute.
The conductor-like idea ({vibeqc-cite}`klamt_schuurmann_cosmo_1993`) is to
solve the electrostatics for a perfect conductor ($\varepsilon\to\infty$), where
the boundary condition is simple: the total potential vanishes on the cavity
surface. The surface is cut into segments $i$, each carrying an apparent
charge $q_i$, and the condition becomes a linear system,

$$
\mathbf A\,\mathbf q = -f(\varepsilon)\,\mathbf V ,
\qquad
f(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon + x} ,
\qquad
E_\text{solv} = \tfrac12\,\mathbf q^{\mathsf T}\mathbf V ,
$$

where $V_i$ is the electrostatic potential of the solute (nuclei and electrons)
at segment $i$, $A_{ij} = 1/|\mathbf r_i - \mathbf r_j|$ for $i \ne j$ with the
diagonal $A_{ii} = 1.0694\sqrt{4\pi/a_i}$ set by the segment area $a_i$
({vibeqc-cite}`scalmani_frisch_csc_2010`), and $f(\varepsilon)$ scales the
conductor charges to the finite dielectric. The CPCM of
{vibeqc-cite}`cossi_cpcm_2003` has $x = 0$, $f = (\varepsilon-1)/\varepsilon$; the
original COSMO has $x = \tfrac12$, $f = (\varepsilon-1)/(\varepsilon+\tfrac12)$
(`variant="cpcm"` or `"cosmo"`). For water the two factors are 0.9872 and
0.9810, so the solvation energies differ by about 0.6 percent (the STO-3G
formaldehyde runs give -1.609 and -1.597 kcal/mol). $f$ saturates quickly, which
is why the dielectric scan above is nearly flat above $\varepsilon\approx20$.

**The macro-iteration.** The potential $\mathbf V$ depends on the solute density
$\mathbf D$, and the charges $\mathbf q$ act back on the electrons as a set of
point charges, an extra one-electron term in the Fock matrix, so density and
charges must be made consistent. vibe-qc solves a gas-phase SCF first, builds
$\mathbf q$, runs an SCF with the charges held fixed, rebuilds $\mathbf V$ and
$\mathbf q$, and repeats until the change in $E_\text{solv}$ falls below
`tol_e_solv` (1e-6 Ha; at most `max_macro_iter` = 30 cycles). Seven cycles
were needed for formaldehyde in water; `q_diis=True` extrapolates the charges
if a charged solute needs many more.

**The cavity.** By default the cavity is a union of atomic spheres
(`cavity="lebedev"`): Bondi van der Waals radii scaled by `radii_scale = 1.20`,
each sphere paved with a Lebedev grid of `n_points_per_sphere = 302`
points, and points buried inside other spheres switched off by a smooth
function of width `switching_sigma_bohr = 0.5` so that the energy and its
gradient are continuous. This describes the convex outer surface and leaves the
concave crevices between atoms unpaved. The alternative,
`cavity="fine"`, is the COSMO FINE cavity of
{vibeqc-cite}`klamt_fine_cavity_2018`: an iso-surface of a pseudo-density that
also paves the concave regions. When you choose it, the construction runs by
default in the molecule-fixed frame (`fine_frame="molecular"`), so that the
cavity, the energy and the gradient rotate rigidly with the solute instead of
depending on its orientation in the laboratory frame; `fine_frame="lab"`
restores the older behaviour and is the right choice for a linear solute, whose
molecule-fixed frame has no nuclear derivative. The two cavities give
different segment sets and so different energies: for formaldehyde in water
(STO-3G, PBE0) the union of spheres gave $E_\text{solv} = -1.609$ kcal/mol on
1047 segments and the FINE cavity -1.292 kcal/mol on 392, so compare solvation
energies only within one cavity choice.

**Gradients and what is left out.** The nuclear gradient is analytic for both
cavities. On top of the SCF gradient at the in-solvent density it adds three
closed-form pieces, the derivative of the cavity matrix $\mathbf A$, the
nuclear-potential derivative and the electronic-potential derivative (one
nuclear-attraction gradient pass over the cavity points), which is why
solvated optimisations converge like gas-phase ones. The model is purely
electrostatic and linear-response: it has no cavitation, dispersion or
repulsion terms, no specific hydrogen bonding, and it composes with the
mean-field density only, which is the reason for the gated combinations at
the top (post-SCF correlation and ROHF/ROKS). COSMO-RS, which turns the
surface charge density into a statistical-thermodynamic model of the liquid,
is a separate experimental route in `vibeqc.solvation.cosmors`.

## Next

* **[User-guide chapter on solvation](../user_guide/solvation.md)**, 
  the full parameter reference, cavity construction details, CPCM
  vs COSMO, theory references.
* **Example scripts under
  [`examples/molecular/solvation/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/molecular/solvation)**
  - single-point, geometry-opt, dielectric scan in runnable form
  (the same scripts that produced the numbers above).
* **Density fitting / RIJCOSX**, set `density_fit` (and optionally
  `cosx`) on the method options and the CPCM SCF picks up RI-JK /
  RIJCOSX automatically; see the
  [user-guide § Density fitting and RIJCOSX](../user_guide/solvation.md).
* **Not yet available**: non-electrostatic terms (SMD-style cavitation
  + dispersion + repulsion) and CPCM in periodic systems.
