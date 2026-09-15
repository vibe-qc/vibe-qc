# MSINDO: semiempirical INDO

This tutorial walks through a complete MSINDO workflow inside vibe-qc, 
from a single-point energy through geometry optimisation, solvation, periodic
surface calculations, and excited states.  MSINDO uses **no external basis set**
(the Slater-type orbital basis is built in), so every example is a self-contained
script.

The MSINDO engine is validated against a reference MSINDO build to
**≤ 1 µHa** on the full H-Br element set.

> **What you need.** A working vibe-qc install (``pip install -e .`` from a
> clone); see {doc}`../quickstart`.  The examples in this tutorial can be
> copied straight into a Python interpreter or script.

---

## 1. Single-point energy, your first MSINDO calculation

The shortest MSINDO run: build a water molecule and hand it to
``run_job`` with ``method="msindo"``, which returns the total energy and
the binding energy relative to the isolated-atom reference.

```python
from vibeqc import Atom, Molecule, run_job

# Water molecule (bohr)
h2o = Molecule([
    Atom(8,  [0.00,  0.00,  0.00]),
    Atom(1,  [0.00,  1.43,  1.11]),
    Atom(1,  [0.00, -1.43,  1.11]),
])

result = run_job(h2o, method="msindo")
print(f"Total energy:  {result.energy:.10f} Ha")
print(f"Binding energy: {result.binding_energy:.6f} Ha")
```

The ``binding_energy`` is ``E_total − Σ ATENG(Z)``, the energy needed to
separate the molecule into its constituent atoms at the INDO reference level.
For H₂O the MSINDO total is −17.01820877 Ha.

### Using the direct API

When you need the density matrix, MO energies, or tight control over the SCF,
call ``run_msindo`` directly with an Angstrom geometry:

```python
from vibeqc.semiempirical.methods import msindo

r = msindo.run_msindo(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]],  # Å
)
print(f"E_elec = {r.electronic_energy:.10f} Ha")
print(f"E_tot  = {r.total_energy:.10f} Ha")
print(f"n_iter = {r.n_iter}, converged = {r.converged}")
print(f"HOMO energy = {r.mo_energies[r.mo_energies.size // 2 - 1]:.6f} Ha")
```

The direct API returns a ``MsindoResult`` with the full density matrix,
MO eigenvalue vector, and SCF diagnostics, useful for post-processing
or feeding into a subsequent post-SCF method.

---

## 2. Geometry optimisation

``msindo_optimize`` wraps SciPy's L-BFGS-B with a finite-difference gradient:

```python
from vibeqc.semiempirical.methods.msindo import msindo_optimize

# Starting guess for water (Å)
Z = [8, 1, 1]
xyz_start = [[0.0, 0.0, 0.0], [0.0, 0.8, 0.6], [0.0, -0.8, 0.6]]

result = msindo_optimize(Z, xyz_start, fmax=4.5e-4)
print(f"Optimised in {result.n_iter} steps")
print(f"Final energy:    {result.energy:.10f} Ha")
print(f"Max gradient:    {result.fmax:.6f} Ha/bohr")
print("Geometry (Å):")
for row in result.coords_angstrom:
    print(f"  {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
```

The ``history`` field in the result records the energy and max gradient at
every optimisation step, so you can plot the convergence.

---

## 3. The NDDO mode, MSINDO's default parametrisation

The reference MSINDO program uses the NDDO (Neglect of Diatomic Differential
Overlap) parametrisation by default.  vibe-qc reproduces closed-shell NDDO to
≤ 1 µHa for H, Li-F, and Na-Cl, including the SPDD terms used by Al-Cl:

```python
from vibeqc.semiempirical.methods import msindo

# Compare INDO vs NDDO for HF
indo = msindo.run_msindo([9, 1], [[0, 0, 0], [0, 0, 0.917]])
nddo = msindo.run_msindo([9, 1], [[0, 0, 0], [0, 0, 0.917]], nddo=True)

print(f"INDO total  = {indo.total_energy:.10f} Ha")
print(f"NDDO total  = {nddo.total_energy:.10f} Ha")
print(f"Δ(NDDO−INDO) = {nddo.total_energy - indo.total_energy:.4f} Ha")
# Δ ≈ +1.2654 Ha — NDDO significantly shifts the absolute energy
```

The difference between INDO and NDDO is large (up to ~1.3 Ha per molecule)
and molecule-dependent.  NDDO is the parametrisation you should use for
comparison with reference MSINDO calculations.

```python
# Multi-molecule NDDO validation
targets = {
    "HF":  ([9, 1],        [[0, 0, 0], [0, 0, 0.917]]),
    "H2O": ([8, 1, 1],     [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]),
    "CH4": ([6, 1, 1, 1, 1], [[0, 0, 0], [0.629, 0.629, 0.629],
           [-0.629, -0.629, 0.629], [0.629, -0.629, -0.629], [-0.629, 0.629, -0.629]]),
}
for name, (Z, xyz) in targets.items():
    r = msindo.run_msindo(Z, xyz, nddo=True)
    print(f"{name:4s} NDDO: {r.total_energy:.10f} Ha")
```

---

## 4. Open-shell calculations (UHF)

MSINDO UHF is available for s/p elements (H-F).  Pass ``multiplicity > 1``
to ``run_msindo`` or build a ``Molecule`` with the desired multiplicity:

```python
from vibeqc import Atom, Molecule, run_job
from vibeqc.semiempirical.methods import msindo

# O₂ triplet — through run_job
o2 = Molecule([Atom(8, [0, 0, 0]), Atom(8, [0, 0, 2.283])], multiplicity=3)
print(f"O₂ triplet: {run_job(o2, method='msindo').energy:.10f} Ha")

# OH radical — through the direct API
oh = msindo.run_msindo([8, 1], [[0, 0, 0], [0, 0, 1.833]], multiplicity=2)
print(f"OH radical: {oh.total_energy:.10f} Ha  ({oh.n_iter} iterations)")

# Charged open-shell (H₂O⁺ cation)
h2o_plus = Molecule([
    Atom(8, [0, 0, 0]), Atom(1, [0, 1.43, 1.11]), Atom(1, [0, -1.43, 1.11])
], charge=+1, multiplicity=2)
print(f"H₂O⁺: {run_job(h2o_plus, method='msindo').energy:.10f} Ha")
```

---

## 5. Periodic systems, the Cyclic Cluster Model (CCM)

For solids and surfaces, MSINDO uses the **Cyclic Cluster Model**, 
a finite supercell with Wigner-Seitz periodic environment and optional
Ewald Madelung embedding.

### 5a. Bulk MgO (3-D periodic)

A rocksalt Mg₄O₄ cell driven through ``run_ccm`` with ``madelung=True``
gives the bulk MgO energy under full 3-D periodicity with Ewald
embedding.

```python
from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

a = 4.21  # MgO lattice constant (Å)

# Rocksalt Mg₄O₄ cell
fcc_frac = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
Z = [12] * 4 + [8] * 4
coords = ([[f * a for f in p] for p in fcc_frac]           # Mg
          + [[(p[0] + 0.5) * a, p[1] * a, p[2] * a]         # O
             for p in fcc_frac])
cell_3d = [[a, 0, 0], [0, a, 0], [0, 0, a]]

bulk = run_ccm(Z, coords, cell_3d, madelung=True)
print(f"Bulk MgO (Madelung):  {bulk.total_energy:.10f} Ha  ({bulk.n_iter} it)")
```

### 5b. MgO(100) surface (2-D periodic)

Dropping to a 2-D cell (two lattice vectors, finite in z) turns the same
machinery into a slab calculation: an MgO(100) bilayer that is periodic
in-plane and terminated along the surface normal.

```python
# MgO(100) bilayer slab: periodic in-plane, finite in z
a_half = a / 2
slab_Z = [12, 8, 8, 12, 8, 12, 12, 8]
slab_C = [[0, 0, 0], [a, 0, 0], [0, a, 0], [a, a, 0],
          [0, 0, a], [a, 0, a], [0, a, a], [a, a, a]]
cell_2d = [[2*a, 0, 0], [0, 2*a, 0]]

slab = run_ccm(slab_Z, slab_C, cell_2d, madelung=True)
print(f"MgO(100) slab:        {slab.total_energy:.10f} Ha")
```

### 5c. Water adsorption on MgO(100)

Placing an H₂O molecule above the slab and subtracting the clean-slab
and gas-phase-water energies gives the adsorption energy, the flagship
surface-chemistry quantity this CCM workflow is built for.

```python
# Add an H₂O adsorbate above the surface
water_Z = slab_Z + [8, 1, 1]
water_C = slab_C + [
    [a, 0, a + 2.2],                 # O (adsorbate)
    [a + 0.76, 0, a + 2.76],          # H
    [a - 0.76, 0, a + 2.76],          # H
]
sys = run_ccm(water_Z, water_C, cell_2d, madelung=True)
print(f"Slab + H₂O:           {sys.total_energy:.10f} Ha")

# Adsorption energy (gas-phase water from the molecular engine)
gas = msindo.run_msindo(
    [8, 1, 1],
    [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
)
e_ads = sys.total_energy - slab.total_energy - gas.total_energy
KCAL_PER_HA = 627.509474
print(f"E_ads = {e_ads:.6f} Ha  =  {e_ads * KCAL_PER_HA:.2f} kcal/mol")
```

### 5d. Adsorbate relaxation

Relax the water molecule on the frozen MgO slab:

```python
from vibeqc.semiempirical.methods.msindo_ccm import ccm_optimize

# Freeze the slab atoms (indices 0–7), relax the adsorbate (8,9,10)
relaxed_C, relaxed_res = ccm_optimize(
    water_Z, water_C, cell_2d, madelung=True,
    frozen=list(range(8)),           # hold the slab fixed
    fmax=2e-3,
)
e_ads_relaxed = relaxed_res.total_energy - slab.total_energy - gas.total_energy
print(f"Relaxed E_ads = {e_ads_relaxed:.6f} Ha"
      f"  =  {e_ads_relaxed * KCAL_PER_HA:.2f} kcal/mol")

# Force on the adsorbate atoms
from vibeqc.semiempirical.methods.msindo_ccm import ccm_gradient_fd
grad = ccm_gradient_fd(water_Z, relaxed_C, cell_2d, madelung=True)
adsorbate_forces = grad[len(slab_Z):]
import numpy as np
print(f"Max adsorbate force: {np.max(np.abs(adsorbate_forces)):.2e} Ha/bohr")
```

> **Valid clusters** must satisfy the CCM validity rule: the summed
> Wigner-Seitz weight for every atom must round to ``NATOMS − 1``.
> An invalid cluster raises ``ValueError``.  Odd-*N* evenly-spaced
> chains/cells are the most robust shapes.

---

## 6. Implicit solvation (COSMO)

MSINDO COSMO puts the molecule in a dielectric continuum.  The default
``cavity="gepol"`` reproduces reference MSINDO to ~2 × 10⁻⁹ Ha:

```python
from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

# Water in water (ε = 78.39)
Z = [8, 1, 1]
xyz = [(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)]  # Å

r = msindo_cosmo(Z, xyz, epsilon=78.39)
print(f"In-water total:   {r.total_energy:.10f} Ha")
print(f"Solvation energy: {r.e_solv * 1000:.3f} mHa")

# Through run_job (bohr coords, solvent="water")
from vibeqc import Atom, Molecule, run_job
h2o = Molecule([Atom(8, [0, 0, 0]),
                Atom(1, [0, 1.43, 1.11]),
                Atom(1, [0, -1.43, 1.11])])
r2 = run_job(h2o, method="msindo", solvent="water")
print(f"run_job COSMO:    {r2.energy:.10f} Ha  (E_solv = {r2.e_solv*1000:.3f} mHa)")
```

Several small molecules in water (default GEPOL cavity):

```python
cases = {
    "H₂O": ([8, 1, 1],
            [(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)]),
    "HF":  ([9, 1], [(0, 0, 0), (0, 0, 0.917)]),
    "CH₄": ([6, 1, 1, 1, 1],
            [(0, 0, 0), (0.629, 0.629, 0.629), (-0.629, -0.629, 0.629),
             (0.629, -0.629, -0.629), (-0.629, 0.629, -0.629)]),
}
for name, (Z, xyz) in cases.items():
    r = msindo_cosmo(Z, xyz, epsilon=78.39)
    print(f"{name:4s} E_solv = {r.e_solv * 1000:+.3f} mHa  "
          f"total = {r.total_energy:.10f} Ha")
```

---

## 7. Excited states (CIS)

MSINDO CIS gives vertical singlet and triplet excitation energies:

```python
from vibeqc.semiempirical.methods.msindo import msindo_cis

# H₂O — first 5 singlet states
s = msindo_cis([8, 1, 1],
               [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]],
               spin="singlet", n_states=5)
for i, ev in enumerate(s.excitation_energies_ev, 1):
    print(f"  S{i} = {ev:.3f} eV")

# First few triplet states
t = msindo_cis([8, 1, 1],
               [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]],
               spin="triplet", n_states=3)
print(f"  T1 = {t.excitation_energies_ev[0]:.3f} eV")
```

> vibe-qc ships the **rigorous** INDO CIS.  Reference MSINDO applies
> empirical scaling (``SCALEDCIS``, ``CISCORR``) for spectroscopic
> fitting, so its excitation energies differ from the unscaled values.

---

## 8. Correlation and quasiparticle methods

Two post-SCF layers sit on top of the MSINDO reference: MP2 (and its
spin-component-scaled variants) for the correlation energy, and Green's
function methods for ionisation potentials.

### MP2

MSINDO MP2 adds the second-order correlation energy on top of the SCF
total, with ``variant="scs"`` and ``variant="sos"`` selecting the
spin-component-scaled and same-spin-only flavours.

```python
from vibeqc.semiempirical.methods.msindo import msindo_mp2

r = msindo_mp2([8, 1, 1],
               [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
print(f"E_SCF  = {r.e_scf:.10f} Ha")
print(f"E_corr = {r.e_corr:.10f} Ha  ({r.e_corr*1000:.3f} mHa)")
print(f"E_MP2  = {r.e_total:.10f} Ha")

# SCS-MP2 and SOS-MP2 scaling
from vibeqc.semiempirical.methods.msindo import msindo_mp2 as mp2
scs = mp2([8, 1, 1],
          [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]],
          variant="scs")
sos = mp2([8, 1, 1],
          [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]],
          variant="sos")
print(f"SCS-MP2 corr = {scs.e_corr:.10f} Ha")
print(f"SOS-MP2 corr = {sos.e_corr:.10f} Ha")
```

### Green's function (GF2 / OVGF), ionisation potentials

``msindo_ovgf`` returns ionisation potentials beyond Koopmans' theorem:
the GF2 self-energy correction to each orbital, with ``renormalize=True``
switching to the renormalised variant for improved IPs.

```python
from vibeqc.semiempirical.methods.msindo import msindo_ovgf

r = msindo_ovgf([8, 1, 1],
                [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
print(f"HOMO IP (Koopmans) = {r.koopmans_ip_ev:.2f} eV")
print(f"HOMO IP (GF2)      = {r.ip_homo_ev:.2f} eV")

# Renormalised GF2 for improved IPs
r_ren = msindo_ovgf([8, 1, 1],
                    [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]],
                    renormalize=True)
print(f"HOMO IP (renorm)   = {r_ren.ip_homo_ev:.2f} eV")
```

---

## 9. NEB, reaction paths

Find the minimum-energy path for the NH₃ umbrella inversion:

```python
import numpy as np
from vibeqc import Atom, Molecule, run_neb

def nh3(flip: bool = False):
    """Pyramidal NH₃; ``flip=True`` gives the mirror image."""
    r, angle = 1.012, np.deg2rad(106.67)
    sin_a = np.sqrt(2 * (1 - np.cos(angle)) / 3)
    cos_a = np.sqrt(1 - sin_a**2)
    z_sign = +1 if flip else -1
    coords = [[0, 0, 0]]
    for k in range(3):
        phi = k * 2 * np.pi / 3
        coords.append([r * sin_a * np.cos(phi),
                       r * sin_a * np.sin(phi),
                       z_sign * r * cos_a])
    return Molecule([Atom(7, c) for c in coords[:1]]
                  + [Atom(1, c) for c in coords[1:]])

result = run_neb(nh3(False), nh3(True), method="msindo",
                 n_images=7, climbing_image=True)
ts = result.transition_state_index
barrier_kcal = (result.energies[ts] - result.energies[0]) * 627.509474
print(f"Barrier height: {barrier_kcal:.2f} kcal/mol")
print(f"TS energy:      {result.energies[ts]:.10f} Ha")
```

> Each NEB image costs 6N SCF calls per outer iteration (FD gradient).
> Parallelised across images.  Practical for up to ~10-15 atoms.

---

## 10. Post-processing, charges, properties

Mulliken/Löwdin charges coincide in MSINDO's orthogonal basis:

```python
from vibeqc.semiempirical.methods import msindo

Z = [16, 1, 1]
xyz = [[0, 0, 0], [0.9627, 0, 0.9264], [-0.9627, 0, 0.9264]]
res = msindo.run_msindo(Z, xyz)

blocks, _ = msindo._atom_blocks(Z)
for i_atom, (lo, hi) in enumerate(blocks):
    pop = float(res.density.diagonal()[lo:hi].sum())
    q = msindo.eff_core_charge(Z[i_atom]) - pop
    sym = {16: "S", 1: "H"}[Z[i_atom]]
    print(f"  {sym}  pop = {pop:.4f}  charge = {q:+.4f}")
# H₂S: S ≈ −0.25, H ≈ +0.12 — the expected slight polarity
```

---

## 11. Citing

MSINDO results from vibe-qc should cite the method's lineage.  When you
use ``run_job``, citations are emitted automatically into ``.out``,
``.bibtex``, and ``.references``.  The papers that should appear:

- B. Ahlswede, K. Jug, *J. Comput. Chem.* **20**, 563 (1999), INDO method (Part I)
- B. Ahlswede, K. Jug, *J. Comput. Chem.* **20**, 572 (1999), INDO method (Part II)
- B. Voigt, *Theor. Chim. Acta* **31**, 289 (1973), if using **NDDO**
- M. J. S. Dewar, W. Thiel, *Theor. Chim. Acta* **46**, 89 (1977), if using **NDDO**
- T. Bredow, G. Geudtner, K. Jug, *J. Comput. Chem.* **22**, 89 (2001), if using **CCM**
- M. F. Peintinger, T. Bredow, *J. Comput. Chem.* **35**, 839 (2014), if using **CCM**
- A. Klamt, G. Schüürmann, *J. Chem. Soc. Perkin Trans.* **2**, 799 (1993), if using **COSMO**

---

## Next

- The {doc}`../user_guide/msindo` user guide has the full API reference,
  feature matrix, and a keyword-to-Python mapping table for users
  familiar with the Fortran MSINDO program. It also carries the full
  worked narrative (not just the example scripts) for four topics this
  tutorial doesn't walk through directly: [molecular
  dynamics](../user_guide/msindo.md#molecular-dynamics), [well-tempered
  metadynamics](../user_guide/msindo.md#well-tempered-metadynamics),
  [conical intersections
  (MECI)](../user_guide/msindo.md#conical-intersections-meci), and
  [variational CISD](../user_guide/msindo.md#configuration-interaction-cisd).
- [`examples/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/semiempirical/)
  has the runnable scripts behind every topic above (24, 25, 26, 27) plus
  the periodic-semiempirical and cross-method-validation scripts the
  [periodic semiempirical & validation
  tutorial](semiempirical_periodic_and_validation.md) covers.
