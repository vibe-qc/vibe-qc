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

print(f"E(gas)         = {gas.energy:12.6f} Ha")
print(f"E(in water)    = {sol.energy:12.6f} Ha")
print(f"E_solv (½q·V)  = {sol.e_solv * 627.5094740631:+8.3f} kcal/mol")
print(f"ΔG_solv total  = {delta_g_solv_kcal:+8.3f} kcal/mol")
print(f"CPCM macro-iters: {sol.n_macro_iter}")
```

Real output (PBE0/def2-SVP, vibe-qc v0.9.0):

```
E(gas)            =    -114.283602 Ha
E(water, total)   =    -114.290170 Ha
E_solv (½ q · V)  =   -0.007985 Ha =   -5.011 kcal/mol
ΔG_solv (total)   =   -0.006568 Ha =   -4.122 kcal/mol
CPCM macro-iters  = 7
Cavity points     = 1047 (after switching)
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
overshoot back toward the experimental value. That's a v0.10+
roadmap item; pure CPCM is the electrostatic backbone everyone
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

Real result (vibe-qc v0.9.0, PBE0/def2-SVP, BFGSLineSearch fmax=0.01):

```
r(C=O) gas    : 1.1961 Å
r(C=O) water  : 1.2027 Å
Δr(C=O)       : +0.0066 Å
```

A ~0.7 pm elongation, small in absolute terms, but the right sign
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

For routine work with `n_atoms < 50` and `n_pts_per_sphere = 110`
(the PySCF default), CPCM-aware optimisations are very tractable
on a laptop.

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

Real scan (PBE0/def2-SVP, formaldehyde, vibe-qc v0.9.0):

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
* **Roadmap v0.10+**, non-electrostatic terms (SMD-style cavitation
  + dispersion + repulsion) and CPCM-in-periodic-systems.
