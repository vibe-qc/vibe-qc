# Slabs and adsorption: the ab-initio surface workflow

Surface chemistry is where vibe-qc points: a slab of solid with a
molecule sitting on it, and the energy released when the two meet. This
tutorial runs that workflow end to end with an honest ab-initio
Hartree-Fock / DFT periodic SCF, no machine-learned shortcut. We build a
small MgO(100) slab, drop a water molecule on it, and compute the
adsorption energy from three single-point periodic calculations. The
[machine-learning potentials with MACE](mace_mlip.md) tutorial walks the
same physical system on a learned potential; this is its first-principles
counterpart, slower but a genuine total-energy method with a
wavefunction.

This page is deliberately concrete about what converges and what costs.
A converged all-electron periodic slab is one of the harder things a
quantum-chemistry code does, and the workflow below is the one that
actually settles, with the cost and accuracy caveats stated plainly.

## What you will build

The deliverable is one adsorption difference assembled from three periodic
single points that share a basis, a functional, a Coulomb gauge, and an
occupation objective. At zero temperature that difference is the adsorption
energy,

```
E_ads = E[slab + adsorbate] - E[slab] - E[adsorbate, gas]
```

while this smeared tutorial must instead use the electronic Mermin free energy
in every term,

```
F_ads(electronic) = F[slab + adsorbate] - F[slab] - F[adsorbate, gas].
```

A negative value means binding on the corresponding objective. Every term
here is a periodic SCF on a three-dimensional cell with a vacuum gap, run
through vibe-qc's BIPOLE Ewald path.

## Setup: this tutorial uses a 3D cell with a vacuum gap, on purpose

vibe-qc's default and physically correct model of a surface is a genuine
**`dim=2`** slab: two in-plane lattice vectors, atoms at their real `z`,
and no vacuum at all. `vibeqc.build.slab` returns exactly that, and
`jk_method="auto"` routes it to the rigorous vacuum-free 2D Ewald gauge.
See [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md).

This tutorial deliberately does **not** use it. A thin oxide slab has
surface states crowding the gap, so the SCF oscillates without
Fermi-Dirac smearing, and `smearing_temperature` is not wired through the
2D route yet. So we build the legacy **`dim=3` cell with a vacuum gap**,
which does support smearing, and we pay the price: a residual slab-slab
image interaction that shifts the absolute energy. Some of that shift may
cancel in a difference, but the adsorption result must still be converged
with respect to vacuum thickness.

Stack enough vacuum (10 to 15 Angstrom) above the atoms and the periodic
images stop seeing each other along the surface normal, so the cell
behaves like an isolated surface even though the Coulomb sum is fully
three-dimensional. Build such a cell either by hand, as below, or with
`vibeqc.build.slab(..., periodic_z=True)`.

`vibeqc.build` ships closed-form surface cells for fcc/bcc/hcp **metals**
(see [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md)). MgO
is ionic rocksalt, not in that table, so we lay the cell out by hand,
which is the general recipe for any non-metal surface. The MgO(100) face
is a square net of alternating Mg and O; one atomic layer holds one Mg
and one O, and successive layers shift by half the cell diagonal:

```python
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

a = 4.21          # MgO lattice constant (Angstrom)
d = a / 2.0       # interlayer spacing along [100]
n_layers = 2      # a 2-layer slab is the minimum sensible thickness
vacuum = 12.0     # Angstrom of vacuum above the slab

# Third lattice vector = slab thickness + vacuum; centre the slab in z.
slab_thick = (n_layers - 1) * d
c_len = slab_thick + vacuum
z0 = (c_len - slab_thick) / 2.0

# Square in-plane cell of side a; one Mg + one O per layer, checkerboarded,
# with successive layers swapping the Mg/O sublattices.
lattice = np.zeros((3, 3))
lattice[:, 0] = [a, 0.0, 0.0]
lattice[:, 1] = [0.0, a, 0.0]
lattice[2, 2] = c_len

def to_atoms(coords):                      # coords: (Z, x, y, z) in Angstrom
    return [vq.Atom(Z, [x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR])
            for (Z, x, y, z) in coords]

slab_coords = []
for L in range(n_layers):
    z = z0 + L * d
    if L % 2 == 0:
        slab_coords += [(12, 0.0, 0.0, z), (8, a / 2, a / 2, z)]   # Mg, O
    else:
        slab_coords += [(8, 0.0, 0.0, z), (12, a / 2, a / 2, z)]   # O, Mg

slab = vq.PeriodicSystem(3, lattice * ANG2BOHR, to_atoms(slab_coords))
print(f"slab: dim={slab.dim}, {len(slab.unit_cell)} atoms, "
      f"{slab.n_electrons()} electrons")
```

The slab is `dim=3` with a 12 Angstrom vacuum gap, as this tutorial
intends. A water molecule goes
on top of a surface Mg, oxygen down, about 2.2 Angstrom above the top
layer; the gas-phase reference is the same molecule in the same cell so
the Coulomb gauge matches:

```python
# top-layer Mg site (the slab's top layer swaps sublattices each layer)
z_top = max(c[3] for c in slab_coords)
mg_xy = (0.0, 0.0) if (n_layers - 1) % 2 == 0 else (a / 2, a / 2)

h = 2.2  # Angstrom, O above the surface
water_on = [(8, mg_xy[0], mg_xy[1], z_top + h),
            (1, mg_xy[0] + 0.7572, mg_xy[1], z_top + h + 0.586),
            (1, mg_xy[0] - 0.7572, mg_xy[1], z_top + h + 0.586)]

zc = c_len / 2.0  # gas-phase water, same cell, centred
water_gas = [(8, 0.0, 0.0, zc),
             (1, 0.7572, 0.0, zc + 0.586),
             (1, -0.7572, 0.0, zc + 0.586)]

slab_with_water = vq.PeriodicSystem(3, lattice * ANG2BOHR,
                                    to_atoms(slab_coords + water_on))
water_box = vq.PeriodicSystem(3, lattice * ANG2BOHR, to_atoms(water_gas))
```

## Run the three single points

With the geometry in hand, the calculation is three calls to the same
periodic SCF driver. We use closed-shell RKS with the PBE functional and
the minimal STO-3G basis, the cheapest combination that still produces a
real ionic-surface energy; `jk_method="bipole"` selects the CRYSTAL-gauge
Ewald J/K build that converges on tight ionic cells. A thin oxide slab
has surface states that crowd the gap, so the plain SCF oscillates;
**Fermi-Dirac smearing** (`smearing_temperature`) lets the occupations go
fractional near the Fermi level and settles it. The value matters: keep
it *below* the slab's gap for the physical state (see the warning at the
end of this section). The 0.003 Ha used here is the sub-gap value for this
small model. The same three calls must share basis, functional, cell,
gauge, and electronic temperature so they evaluate one consistent
finite-temperature objective:

```python
def slab_energy(system, label):
    return vq.run_periodic_job(
        system,
        vq.BasisSet(system.unit_cell_molecule(), "sto-3g"),
        method="RKS",
        functional="pbe",
        jk_method="bipole",          # CRYSTAL-gauge Ewald J/K
        kpoints=(1, 1, 1),           # Gamma-only here; (4, 4, 1) for production
        smearing_temperature=0.003,  # Hartree; sub-gap Fermi-Dirac smearing
        conv_tol_energy=1e-7,
        max_iter=120,
        output=label,                # writes <label>.out with the full trace
    )

r_slab  = slab_energy(slab,            "mgo_slab")
r_water = slab_energy(water_box,       "water_gas")
r_sys   = slab_energy(slab_with_water, "mgo_slab_water")
```

Each call writes a `<label>.out` with the banner, the SCF iteration
trace, the internal total energy, and a finite-temperature block containing
`free_energy`. At nonzero smearing the variational thermodynamic quantity is
the Mermin (electronic Helmholtz) free energy $F=E-TS$, not the internal
`result.energy` value labelled `Total energy` in the output. Use the returned
result fields directly and take a difference of the three free energies:

```python
f_slab  = float(r_slab.free_energy)
f_water = float(r_water.free_energy)
f_sys   = float(r_sys.free_energy)

f_ads = f_sys - f_slab - f_water
print(f"F_ads(electronic) = {f_ads:+.6f} Ha = {f_ads * 27.2114:+.3f} eV")
```

Each `<label>.out` ends with both an energy summary and a
`Finite-temperature (smearing)` block. Record the three `free_energy` values,
their common `kBT_smearing`, and the convergence status with any reported
adsorption result. A representative run has this shape:

```text
mgo_slab        free_energy = <F_slab>  Ha   (converged)
water_gas       free_energy = <F_water> Ha   (converged)
mgo_slab_water  free_energy = <F_sys>   Ha   (converged)

F_ads(electronic) = F_sys - F_slab - F_water
```

For this deliberately cheap model, even a converged negative
`F_ads(electronic)` is only a starting point. The minimal STO-3G basis has
large basis-set superposition error, and the unrelaxed thin slab at full
monolayer coverage is not a publication model. The
[MACE tutorial](mace_mlip.md) gives about -0.8 eV for its separately defined
model; do not treat that as a parity target for this electronic-structure
setup.

The slab energy is also sensitive to the smearing, which is the whole
point of the warning below. The same two-layer slab can converge to
different internal energies with 0.003 Ha (the sub-gap value requested by
the code) and 0.01 Ha (large enough to populate a near-metallic basin).
Using one electronic temperature makes the three terms comparable, but the
entropy and basin shifts do **not** generally cancel. Converge the reported
free-energy difference with respect to smearing and, when a zero-temperature
adsorption energy is wanted, reduce the smearing toward zero while following
the same electronic state.

The cross-checks that say the machinery itself is sound sit right next to
the result: the same BIPOLE path converges bulk rocksalt MgO (the FCC
primitive, RHF/STO-3G) to -271.752 Ha per formula unit, and the
gas-phase water term, -75.224 Ha, matches a molecular PBE/STO-3G water.
So the recipe is trustworthy; it is the *model* (basis, slab, coverage,
geometry) that is deliberately cheap here. Tighten every knob in the
[Next](#next) section before quoting an adsorption energy in a paper, and
for publication-grade screening across many sites and surfaces in
seconds, let the [MACE tutorial](mace_mlip.md) run this very system on a
learned potential.

```{warning}
**This slab is a small-gap, near-metallic case, and vibe-qc says so.**
A two-layer MgO/STO-3G slab has a HOMO-LUMO gap of only ~0.5 eV at the
Gamma point. If the Fermi-Dirac smearing temperature is set comparable to
that gap, vibe-qc warns that the SCF has settled a *near-metallic basin*
with several fractional occupations, which can sit hundreds of mHa from
the physical gapped solution. The fix it points you to is to drop the
smearing well below the gap (here ~0.003 Ha, ~0.08 eV) so the SCF lands
the gapped state. Heed that warning: a slab free energy from the wrong basin
makes the adsorption result meaningless. This is the
[periodic-SCF discipline](periodic_scf_convergence.md) in action, an
oscillating or wrong-basin ionic slab is a signal to diagnose, not to
crank the smearing until the number stops moving.
```

```{note}
**Cost.** An all-electron, exact-ERI periodic slab is genuinely
expensive: each of these single points sums two-electron integrals over
the lattice images of a large, anisotropic vacuum cell, so a 2-layer
STO-3G slab single point is minutes of wall time, not seconds, and a
production slab (thicker, larger basis, a surface k-mesh) is
substantially more. This is the price of a true total-energy method with
a wavefunction; the [MACE tutorial](mace_mlip.md) trades it for a learned
potential when you need thousands of geometries fast.
```

## Theory: the slab model

The sections below unpack the three ideas the script leans on: how a
surface becomes a tractable periodic cell, what the adsorption energy
actually measures, and the pitfalls that separate an illustrative number
from a publishable one.

### Two-dimensional periodicity, and why this cell is three-dimensional

A clean crystal surface is periodic in the two directions parallel to the
face and aperiodic along the normal: it simply ends, with vacuum above
and bulk below. A finite calculation cannot carry semi-infinite bulk, so
the standard construction is the **slab supercell**. Take a few atomic
layers thick enough to develop a bulk-like interior, place them in a cell
whose third lattice vector is much longer than the slab, and fill the
remainder with vacuum. Repeated periodically, this tiles space with
slabs separated by vacuum gaps.

Two convergence parameters control the fidelity of the model. The
**vacuum gap** must be wide enough that the electron density of one slab
has decayed to zero before it reaches the next image; 10 to 15 Angstrom
is typical for Gaussian basis sets, whose orbitals are spatially compact.
The **slab thickness** must be large enough that the middle layers
recover the bulk electronic structure and both surfaces are well
separated; properties converge from below as layers are added, and a
two- or three-layer slab is a starting point, not a converged answer.

Because the cell above is declared `dim=3`, vibe-qc sums the Coulomb
interaction in all three dimensions with Ewald (see
[Madelung constants via Ewald summation](madelung_with_ewald.md) for why
the three-dimensional electrostatic sum is only conditionally convergent
and needs Ewald rather than naive truncation). A symmetric, charge-
neutral slab carries no net dipole across the cell, so no dipole
correction is required; an asymmetric slab or a polar face would.

The better model, when you can use it, is `dim=2`. A `PeriodicSystem`
built with two in-plane lattice vectors (`vq.slab_2d`) is periodic in
only two directions, and vibe-qc evaluates its electrostatics with the
rigorous vacuum-free 2D Ewald sum of Parry and of de Leeuw and Perram.
That total energy is provably independent of the synthesized third
lattice column, and its nuclear repulsion does not depend on the k-mesh.
There is no vacuum to converge and no slab-slab image error to cancel.

Use `dim=2` for single-point surface energies. Explicit closed-shell slab
GDF also supports fixed-cell analytic-gradient relaxation. Stay with the
`dim=3` slab-plus-vacuum construction when you need **Fermi-Dirac smearing**
(this tutorial), freeze masks, or periodic Gaussian NEB. Periodic BIPOLE
Hessians fail closed.

```{warning}
Before v0.15.32 a `dim=2` cell was silently routed to a bulk Coulomb
builder, which treated the slab as a crystal of sheets stacked `a3`
apart. The nuclear repulsion then depended on the k-mesh and the total
energy was wrong by thousands of Hartree. Bulk routes now raise on
`dim=2` instead. If you have older `dim=2` slab energies, discard them.
```

### What the adsorption energy measures

The adsorption energy is a reaction energy for

```
slab + molecule(gas)  ->  slab-with-molecule
```

so

```
E_ads = E[slab + adsorbate] - E[slab] - E[adsorbate, gas].
```

A negative value means the bound state is lower in energy than the
separated fragments: the molecule binds. The magnitude sorts
physisorption (tenths of an eV, dispersion and weak electrostatics) from
chemisorption (an eV or more, a real bond). Because the same per-atom and
per-cell systematic errors appear in all three terms with the same basis,
functional, and cell, they cancel to a large degree in the difference,
which is why a modest basis can give a meaningful `E_ads` even when no
single total energy is near the basis-set limit.

That equation is the zero-temperature limit. At nonzero electronic smearing,
replace every $E$ by the result's Mermin free energy $F=E-TS$ and report
`F_ads(electronic)` plus the common smearing temperature. Do not mix internal
energies and free energies in one difference, and do not assume that the
entropy contribution cancels.

The cancellation is only as good as the consistency. All three single
points must use the **same basis set, the same functional, the same
k-mesh, and the same Coulomb gauge**. Mixing a molecular gas-phase code
with a periodic slab code, or changing the cell between terms, breaks the
cancellation and corrupts `E_ads`.

### Pitfalls: BSSE, coverage, relaxation

Three systematic effects separate the number above from a converged
adsorption energy.

**Basis-set superposition error (BSSE).** With atom-centred Gaussian
basis functions, the slab-plus-adsorbate complex borrows basis functions
from its partner and artificially lowers its energy, over-binding the
adsorbate. The counterpoise correction recomputes each fragment in the
presence of the other's (ghost) basis functions and subtracts the
difference. BSSE shrinks as the basis grows; with STO-3G it is large, so the
raw adsorption difference here is an over-bound estimate.

**Coverage.** A `p(1x1)` surface cell places one adsorbate per surface
metal site, the dense full-monolayer limit, and the periodic images of
the adsorbate interact laterally at the in-plane lattice spacing. The
adsorption energy you compute is therefore the high-coverage value,
lateral interactions included. To approach the dilute, single-molecule
limit, enlarge the lateral cell to a `(2x2)` or `(3x3)` supercell so the
adsorbates are far enough apart not to interact, and watch `E_ads`
converge with cell size.

**Relaxation.** The energies above are single points at a fixed,
hand-placed geometry. A real adsorption energy uses **relaxed**
structures: the clean slab relaxed, the gas molecule relaxed, and the
complex relaxed with the adsorbate and the top surface layers free while
the bottom layers stay fixed at bulk positions. vibe-qc drives this
frozen-substrate relaxation through `relax_atoms` with a `freeze_indices`
list (see [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md) and
[Periodic geometry optimization](periodic_geometry_optimization.md)).
Recompute all three terms after relaxation; their separate changes determine
the adsorption difference, so the sign of that change is not guaranteed. On
strongly basic surfaces, also check whether the molecule stayed intact or
dissociated, the two are different physical states.

## References

- **Solid-state basis sets (pob-TZVP).** M. F. Peintinger, D. Vilela
  Oliveira, and T. Bredow, "Consistent Gaussian basis sets of triple-zeta
  valence with polarization quality for solid-state calculations,"
  *J. Comput. Chem.* **34**, 451 (2013). The production basis to step up
  to from STO-3G; see [Why solid-state calculations use `pob-TZVP`](pob_tzvp.md).
- **PBE exchange-correlation functional.** J. P. Perdew, K. Burke, and M.
  Ernzerhof, "Generalized gradient approximation made simple,"
  *Phys. Rev. Lett.* **77**, 3865 (1996).
- **Ewald summation for crystalline Gaussian densities.** V. R. Saunders,
  C. Freyria-Fava, R. Dovesi, L. Salasco, and C. Roetti, "On the
  electrostatic potential in crystalline systems where the charge density
  is expanded in Gaussian functions," *Mol. Phys.* **77**, 629 (1992).
  Context for Gaussian-density electrostatics and penetration terms. The
  supported vibe-qc BIPOLE path uses exact erfc-screened ERIs plus reciprocal
  long-range Coulomb; its quartet multipole prototype is fail-closed.
- **Counterpoise correction for BSSE.** S. F. Boys and F. Bernardi, "The
  calculation of small molecular interactions by the differences of
  separate total energies. Some procedures with reduced errors,"
  *Mol. Phys.* **19**, 553 (1970).
- **CRYSTAL reference.** A. Erba, J. K. Desmarais, S. Casassa, et al.,
  "CRYSTAL23: a program for computational solid state physics and
  chemistry," *J. Chem. Theory Comput.* **19**, 6891 (2023). The
  Gaussian-basis periodic code whose gauge vibe-qc's BIPOLE path follows.

## Next

- **Step up the basis and k-mesh.** Choose a basis with a supported record for
  every element and add a surface-plane k-mesh (`kmesh=(4, 4, 1)`); see
  [LiH rocksalt with pob-TZVP](lih_pob_tzvp_solid_state.md) and
  [LiH at multiple k-points](lih_multi_k.md) for the multi-k machinery.
- **Relax the structure.** Run the frozen-substrate relaxation from
  [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md) and watch
  the consistently recomputed adsorption difference converge.
- **Add dispersion.** For physisorption, the D3-BJ correction matters;
  pass `dispersion="pbe"` (see
  [Dispersion corrections](dispersion.md) and the periodic D3 section of
  [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md)).
- **Screen fast first, confirm with ab initio.** The
  [MACE tutorial](mace_mlip.md) screens hundreds of adsorption sites in
  seconds on a learned potential; use it to find candidate geometries,
  then confirm the winners with the first-principles workflow here.

## See also

- [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md), the
  surface-builder + adsorbate-placement + frozen-relaxation reference.
- [Machine-learning interatomic potentials with MACE](mace_mlip.md), the
  same MgO+water adsorption on a learned potential.
- [Periodic KS-DFT](periodic_dft.md) and [Periodic HF](periodic_hf.md),
  the SCF machinery underneath.
- [Periodic SCF convergence](periodic_scf_convergence.md), what to do
  when an ionic slab oscillates.
- [Madelung constants via Ewald summation](madelung_with_ewald.md), the
  3D electrostatic sum the slab energy relies on.
