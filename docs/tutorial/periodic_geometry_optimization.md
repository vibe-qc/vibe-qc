# Periodic geometry optimization

Relaxing a periodic structure is the crystalline analogue of the
molecular [geometry optimization](geometry_optimization.md): drive the
atoms downhill on the Born-Oppenheimer surface until the forces vanish.
vibe-qc provides a one-line ASE calculator (`VibeQCPeriodic`) and the native
fixed-cell `relax_atoms` entry point. Variable-cell BIPOLE optimization is
not currently certified and fails closed. This page runs a real fixed-cell
periodic relaxation through the ASE route, then describes the force surface
and the current cell-optimization boundary.

The worked example uses pure-DFT (LDA) forces on purpose. vibe-qc's
periodic gradient is **exact for pure DFT** (LDA / GGA) at the Γ point in
the regime where the lattice sum is well behaved; the Hartree-Fock and
hybrid periodic force still carries a small true-periodic
exchange-derivative residual (tracked as G1a-2), so production HF /
hybrid relaxation falls back to the finite-difference force. Pick a pure
functional and the analytic path is tight.

## Setup

A periodic SCF whose images sit far apart, a molecule in a large
periodic box, is the cleanest place to watch a relaxation: it is a
genuine `PeriodicSystem` driven through the periodic SCF and periodic
gradient, but the cells do not overlap, so the analytic Γ force is exact
(this is the **molecular limit** the gradient module validates to
Newton's-third-law precision). Build H₂ in an 18 Å cubic box with a
deliberately stretched bond, using ASE's `Atoms` with periodic boundary
conditions on all three axes:

```python
import numpy as np
from ase import Atoms
from vibeqc.ase_periodic import VibeQCPeriodic

# H2 in a large periodic box, bond stretched to 0.90 Å (equilibrium is
# shorter) so the relaxation has work to do. pbc=True on all axes makes
# this a fully 3D-periodic Atoms, exactly what VibeQCPeriodic expects.
atoms = Atoms("H2",
              positions=[[0, 0, 0], [0, 0, 0.90]],
              cell=[18.0, 18.0, 18.0],
              pbc=True)
```

The 18 Å box keeps periodic images ~17 Å apart, far enough that the
lattice sum reduces to the single home cell, so the energy and force are
the periodic code's molecular limit.

## Relaxing the atoms with ASE BFGS

Attach `VibeQCPeriodic` as the calculator and hand the `Atoms` to any ASE
optimizer. The calculator runs one periodic SCF per geometry and returns
the energy plus analytic forces; ASE's BFGS uses those forces to pull the
bond back to its minimum:

```python
from ase.optimize import BFGS

atoms.calc = VibeQCPeriodic(
    basis="sto-3g",
    functional="lda",      # pure DFT → exact analytic periodic force
    kpts=(1, 1, 1),        # Γ-only
    conv_tol_energy=1e-8,
    max_iter=60,
)

# Energy + forces from a single SCF, before any optimization:
print(f"start energy = {atoms.get_potential_energy():.6f} eV")
print(f"start force  = {atoms.get_forces()[:, 2]} eV/Å")   # along the bond

dyn = BFGS(atoms, trajectory="h2_box_relax.traj")
dyn.run(fmax=0.02, steps=20)            # converged when |F|_max < 0.02 eV/Å

bond = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
print(f"relaxed bond = {bond:.4f} Å")
print(f"relaxed E    = {atoms.get_potential_energy():.6f} eV")
```

This is the actual `VibeQCPeriodic` + `ase.optimize` path, energy and
forces come from one SCF per geometry, and ASE caches them until
`atoms.positions` change. The run writes `h2_box_relax.traj` (one frame
per BFGS step, positions + energy + forces), viewable with `ase gui
h2_box_relax.traj`.

## Reading the trace

The optimization converges in five BFGS steps. The verified output (LDA /
STO-3G, Apple-silicon laptop):

```
start energy = -30.060097 eV
start force  = [ 4.71211242 -4.71211242] eV/Å

      Step     Time          Energy          fmax
BFGS:    0 22:59:11      -30.060097        4.712112
BFGS:    1 22:59:11      -30.491865        1.213484
BFGS:    2 22:59:11      -30.502128        0.875419
BFGS:    3 22:59:11      -30.509774        0.074684
BFGS:    4 22:59:12      -30.509835        0.004132

relaxed bond = 0.7367 Å
relaxed E    = -30.509835 eV   (= -1.12121575 Ha)
```

Reading the trace: the starting force is ±4.71 eV/Å along the bond, the
two H atoms pull toward each other because the 0.90 Å bond is stretched
past the LDA/STO-3G minimum. BFGS walks downhill, the energy lowers
monotonically by ~0.45 eV, and `fmax` collapses from 4.7 to 0.004 eV/Å in
five steps. The relaxed bond, **0.7367 Å**, is the LDA/STO-3G H₂
equilibrium, and the periodic total energy at the minimum is **−1.12121575
Ha**. That `fmax` reaches the floor is the physical check: the analytic
periodic force is consistent with the energy it differentiates, the
optimizer stops exactly where the gradient vanishes.

The whole relaxation runs in about a second, each step is one periodic SCF
plus one analytic Γ gradient on a 2-basis-function cell. A 3D crystal with
a real basis is heavier per step but the workflow is identical.

## The native route (no ASE)

If you would rather not depend on ASE, the same relaxation runs through
vibe-qc's native periodic optimizer. `relax_atoms` wraps the periodic
forces under scipy's L-BFGS-B while preserving the lattice:

```python
import vibeqc as vq
from vibeqc.bipole_optimize import relax_atoms

# Same H2-in-a-box as a native PeriodicSystem (positions in bohr).
box = 18.0 / 0.529177210903
system = vq.PeriodicSystem(
    3, [[box, 0, 0], [0, box, 0], [0, 0, box]],
    [vq.Atom(1, [0, 0, 0]),
     vq.Atom(1, [0, 0, 0.90 / 0.529177210903])])
kmesh = vq.monkhorst_pack(system, [1, 1, 1])

result = relax_atoms(
    system, basis_name="sto-3g", kmesh=kmesh,
    method="RKS", functional="lda",
    conv_tol_grad=5e-4, max_iter=40,
)
print(f"converged: {result.converged}   E = {result.energy:.8f} Ha")
```

The native path defaults to **finite-difference forces**
(`force_mode="fd"`), which differentiate the real total energy and are
correct by construction, at the cost of ~6N SCFs per gradient. For
production surface-catalysis relaxations the native `relax_atoms(...,
freeze_indices=[...])` is the canonical form because it freezes the bottom
slab layers natively; see
[`periodic_methods.md` § 7.3](../user_guide/periodic_methods.md).

## On to a real crystal

The molecular-limit box exercises every part of the periodic relaxation
pipeline, but it is not yet a crystal. Moving to a tight 3D cell, LiH
rocksalt at its 4.084 Å lattice constant, changes the SCF substantially:
the images now overlap, the Coulomb lattice sum is only conditionally
convergent (it needs Ewald, see
[Madelung constants via Ewald summation](madelung_with_ewald.md)), and the
energy must be k-sampled.

Build the primitive FCC cell, the same one the
[LiH multi-k tutorial](lih_multi_k.md) uses, and run a single point with
the **density-fitted (GDF) driver**, which handles the tight-cell Coulomb
problem cleanly:

```python
import vibeqc as vq

BOHR = 0.529177210903
a = 4.084 / BOHR
lih = vq.PeriodicSystem(
    3, [[0.0, a/2, a/2], [a/2, 0.0, a/2], [a/2, a/2, 0.0]],
    [vq.Atom(3, [0.0, 0.0, 0.0]),
     vq.Atom(1, [a/2, a/2, a/2])])
basis = vq.BasisSet(lih.unit_cell_molecule(), "sto-3g")

opts = vq.PeriodicRHFOptions()
opts.damping = 0.3
opts.level_shift = 0.6
opts.use_diis = True

res = vq.run_krhf_periodic_gdf(
    lih, basis, kmesh=(1, 1, 1), functional="lda", options=opts)
print(f"Γ-only GDF LDA: E = {res.energy_per_cell_ha:.8f} Ha/cell")
# verified: E = -7.82612822 Ha/cell
```

This converges in ~30 s and lands at a sane **−7.82612822 Ha/cell**. To
see where the lattice constant wants to go you can scan the energy versus
`a`, the periodic equation of state, each point a single GDF single-point:

```python
for a_ang in [3.6, 3.7, 3.8, 3.9, 4.0, 4.084, 4.2]:
    a = a_ang / BOHR
    sys_ = vq.PeriodicSystem(
        3, [[0.0, a/2, a/2], [a/2, 0.0, a/2], [a/2, a/2, 0.0]],
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a/2, a/2, a/2])])
    b = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
    e = vq.run_krhf_periodic_gdf(
        sys_, b, kmesh=(1, 1, 1), functional="lda", options=opts
    ).energy_per_cell_ha
    print(f"a = {a_ang:.3f} Å   E = {e:.8f} Ha")
```

The verified Γ-only curve:

```
a = 3.600 Å   E = -8.22436938 Ha
a = 3.700 Å   E = -8.03873186 Ha
a = 3.800 Å   E = -8.01018356 Ha
a = 3.900 Å   E = -7.91276101 Ha
a = 4.000 Å   E = -7.85879979 Ha
a = 4.084 Å   E = -7.82612822 Ha
a = 4.200 Å   E = -7.79556302 Ha
```

```{warning}
**A single Γ point is too coarse to give a relaxed lattice constant.**
This curve has no minimum inside the scanned window, it rises
monotonically as the cell expands, so any apparent stationary point is a
k-sampling artefact, not the equilibrium geometry. A converged equation
of state needs a dense k-mesh, exactly as the
[LiH multi-k tutorial](lih_multi_k.md) flags `[1,1,1]` as "too coarse"
for the energy itself. **Do not quote a relaxed `a` from this scan.**
```

```{admonition} Status: tight-cell Γ relaxation
:class: important
The atomic-relaxation drivers that work cleanly in the molecular-limit
box above (`VibeQCPeriodic` + `ase.optimize`, native `relax_atoms`) route
the **Γ-point force** through the direct-lattice-sum (`DIRECT_TRUNCATED`)
Kohn-Sham path, which is the wrong Coulomb treatment for a tight 3D ionic
cell and on LiH/STO-3G is currently unreliable, it does not converge to a
trustworthy energy at this cutoff, and the analytic Γ gradient stalls on
the overlapping-cell lattice sum. This is a backlog item
([Archived CLAUDE.md § 7](https://vibe-qc.com/docs/):
a periodic SCF landing at an impossible energy is a bug, not a convergence
aid problem), not something to paper over with damping or level shifts.
For tight crystals, determine the lattice with a converged equation-of-state
scan (next section) or another validated code, and take periodic single-point
energies from the GDF driver shown above. The variable-cell BIPOLE entry
points fail closed.
```

## Lattice relaxation status

Atomic relaxation moves atoms inside a fixed box. Relaxing the **cell**
itself, the lattice constant of a rocksalt crystal, is a separate problem:
by symmetry both ions sit on special positions, so the atomic forces are
identically zero at every lattice constant and only the cell-stress drives
the optimization.

The BIPOLE `relax_cell`, `relax_cell_gradient`, and `relax_full` names remain
importable for compatibility but raise `NotImplementedError`. The previous
implementation applied strain with the wrong matrix convention for a general
oblique lattice, and its alternating optimizer could declare atom and cell
convergence from gradients evaluated at different geometries. A production
replacement must evaluate both forces and stress on the same terminal
geometry. Until then, scan a physically motivated lattice coordinate with a
converged k-mesh and fit an equation of state, or use another validated code.

## Theory

The sections below build up the periodic optimization machinery: where the
per-cell forces come from, what the cell-stress adds on top, and how the
ASE bridge ties it to ASE's optimizers.

### Forces in a periodic Gaussian basis

Fix the nuclei of the reference cell at positions
$\mathbf{R} = \{\mathbf{R}_A\}$ and let the lattice translations
$\mathbf{T}$ tile them through all space. The per-cell electronic energy
$E(\mathbf{R})$ is the Born-Oppenheimer potential for the crystal, and the
force on atom $A$ in the reference cell is

$$
\mathbf{F}_A \;=\; -\frac{\partial E}{\partial \mathbf{R}_A}.
$$

As in the molecular case, the variational
[Hellmann-Feynman theorem](geometry_optimization.md#the-hellmann-feynman-theorem-and-analytic-forces)
reduces the electronic part to expectation values of explicit Hamiltonian
derivatives, and because the Gaussian AOs are pinned to the moving nuclei,
**Pulay forces** from $\partial \chi / \partial \mathbf{R}$ appear on top.
The lattice sum threads through every term. vibe-qc assembles the per-cell
gradient as four contributions (Phase G1a):

$$
\mathbf{F}_A = \frac{\partial E^{\text{pc}}_\text{nn}}{\partial \mathbf{R}_A}
  + \sum_{\mathbf{g}} \operatorname{tr}\!\Big(\mathbf{D}(\mathbf{g})\,
      \frac{\partial \mathbf{h}(\mathbf{g})}{\partial \mathbf{R}_A}\Big)
  - \sum_{\mathbf{g}} \operatorname{tr}\!\Big(\mathbf{W}(\mathbf{g})\,
      \frac{\partial \mathbf{S}(\mathbf{g})}{\partial \mathbf{R}_A}\Big)
  + \sum_{\mathbf{g},\mathbf{h},\mathbf{h}'} \Gamma(\mathbf{g},\mathbf{h},\mathbf{h}')\,
      \frac{\partial (\mu\nu \mathbf{0} \mid \lambda\mathbf{h}\,\sigma\mathbf{h}')}{\partial \mathbf{R}_A}.
$$

Reading left to right: the per-cell **nuclear-repulsion gradient**
(lattice-summed, Ewald for a 3D bulk); the **one-electron Pulay** term
(density contracted with the kinetic + nuclear-attraction derivatives);
the **overlap-Lagrangian** (Pulay) term, the only piece that consumes
orbital energies through the energy-weighted density $\mathbf{W}$; and the
**two-electron Pulay** term over lattice-summed ERI derivatives. For
KS-DFT a fifth piece, the **XC Pulay** term (libxc on the periodic Becke
grid with analytic $\partial_c \chi$), is added; it is exact for LDA and
GGA. For pure DFT the two-electron term is the Coulomb (J) derivative only
($\alpha_{\text{HF}} = 0$), which matches finite difference to
Newton's-third-law precision in the molecular limit, this is why the
worked relaxation above uses LDA, and why its `fmax` floors out cleanly.

One subtlety vibe-qc handles for you: the Ewald exchange driver reports
orbital energies on a shifted gauge (a constant exact-exchange self-image
offset), which would inject a spurious
$-\Delta \cdot \operatorname{tr}(\mathbf{D}\,\partial \mathbf{S}/\partial \mathbf{R})$
force through $\mathbf{W}$. The gradient rebuilds the energy-weighted
density from the gauge-free variational Fock so the overlap term stays
consistent with the SCF energy. Hybrids' true-periodic exchange (K)
derivative is the one piece not yet finite-difference-tight (G1a-2); pure
DFT is unaffected.

### Stress and cell relaxation

Relaxing the lattice needs the derivative of the energy with respect to
the cell, not the atoms. The natural object is the **stress tensor**

$$
\sigma_{ij} \;=\; \frac{1}{\Omega}\,\frac{\partial E}{\partial \varepsilon_{ij}},
$$

the energy response to a strain $\varepsilon_{ij}$ of the lattice vectors,
with $\Omega$ the cell volume. A virial-style closed form exists, but it
assumes the atoms scale rigidly with the strain and misses the explicit
lattice / Ewald and Gaussian-Pulay stress, on a Gaussian basis the naive
virial comes out with the **wrong sign** for the strain-energy objective.
The standalone force-virial helper is diagnostic only. The BIPOLE
variable-cell entry points fail closed until a correct column-vector strain
map, full energy derivative, and same-geometry coupled convergence test are
implemented. A converged k-mesh remains essential for any manual equation of
state scan.

### The ASE bridge

`vibeqc.ase_periodic.VibeQCPeriodic` is an
[ASE Calculator](../user_guide/ase_integration.md): hand it a
fully-periodic `Atoms` and it converts the cell + positions to a vibe-qc
`PeriodicSystem` (transposing lattice vectors at the boundary, ASE stores
them as rows, vibe-qc as columns), routes the energy + analytic forces
through the periodic SCF and gradient drivers, and returns them in ASE
units (eV, eV/Å). Because it implements ASE's `energy` and `forces`
properties, every force-driven ASE workflow, `ase.optimize` (BFGS / FIRE),
`ase.mep.NEB`, `ase.md`, drives a vibe-qc periodic system by changing one
line. ASE caches the single-SCF result until `atoms.positions` change, so
an optimizer step costs exactly one SCF + one gradient. The calculator
covers closed-shell HF and RKS / UKS at Γ and multi-k; open-shell HF (UHF)
periodic gradients are not available yet and raise. **Stress is not exposed
on the calculator**. ASE and native BIPOLE workflows therefore support
fixed-cell atomic relaxation only.

## References

- **Hellmann-Feynman theorem.** H. Hellmann, *Einführung in die
  Quantenchemie*, Deuticke (1937); R. P. Feynman, "Forces in molecules,"
  *Phys. Rev.* **56**, 340 (1939).
- **Pulay forces.** P. Pulay, "Ab initio calculation of force constants and
  equilibrium geometries in polyatomic molecules. I. Theory," *Mol. Phys.*
  **17**, 197 (1969).
- **BFGS update.** C. G. Broyden, "The convergence of a class of
  double-rank minimization algorithms," *IMA J. Appl. Math.* **6**, 76
  (1970), and the parallel Fletcher / Goldfarb / Shanno papers (1970).
  Textbook: J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
  Springer (2006).
- **Gaussian-density 3D Ewald (periodic nuclear-repulsion + Coulomb).** V.
  R. Saunders et al., "On the electrostatic potential in crystalline
  systems where the charge density is expanded in Gaussian functions,"
  *Mol. Phys.* **77**, 629 (1992). This is electrostatic-potential and
  penetration context; the supported BIPOLE energy route uses exact
  erfc-screened ERIs plus reciprocal long-range Coulomb.
- **Solid-state basis sets (pob-TZVP).** M. F. Peintinger, D. Vilela
  Oliveira, and T. Bredow, "Consistent Gaussian basis sets of triple-zeta
  valence with polarization quality for solid-state calculations," *J.
  Comput. Chem.* **34**, 451 (2013).
- **ASE (the optimizer driver).** A. H. Larsen et al., "The atomic
  simulation environment, a Python library for working with atoms," *J.
  Phys. Condens. Matter* **29**, 273002 (2017).

## Next

- [Geometry optimization](geometry_optimization.md), the molecular
  counterpart, BFGS line search, convergence criteria, and the full force
  theory this page builds on.
- [Nudged elastic band reaction paths](neb_reaction_path.md), once you can
  relax periodic endpoints, find the transition state between them.
- [LiH rocksalt at multiple k-points](lih_multi_k.md), the converged
  multi-k energy a tight-crystal equation of state needs.
- [ASE integration](../user_guide/ase_integration.md), the full ASE
  calculator surface, constraints, and the `VibeQCPeriodic` reference.
