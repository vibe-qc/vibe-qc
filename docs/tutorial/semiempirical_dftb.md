(semiempirical_dftb)=
# Semiempirical DFTB: fast preoptimization

vibe-qc ships a self-contained DFTB (density-functional tight-binding)
implementation for rapid structure preoptimization and large-system
screening.  DFTB is 10²-10⁴× faster than full DFT while capturing
qualitative bonding, making it ideal for:

- geometry preoptimization before DFT refinement
- screening hundreds of conformers
- quick energy estimates for large systems
- periodic Γ-point and k-point DFTB screening

## Methods available

vibe-qc ships two DFTB levels, each with a closed- and open-shell model
class, plus an optional D3(BJ) dispersion wrapper:

| Method | Description | Closed‑shell | Open‑shell |
|--------|-------------|-------------|------------|
| DFTB0 | Non‑self‑consistent tight‑binding | `DFTB0Model` | `UDFTB0Model` |
| SCC‑DFTB | Self‑consistent‑charge DFTB | `SCCDFTBModel` | `USCCDFTBModel` |
| +D3(BJ) | Dispersion‑corrected | `DispersionCorrectedModel` | `DispersionCorrectedModel` |

All molecular DFTB variants expose gradients. DFTB0 and UDFTB0 match
finite differences tightly; SCC-DFTB uses a CP-SCC charge-response
correction, while USCC-DFTB remains an approximate fixed-charge gradient.

## How DFTB works

DFTB (density-functional tight-binding) sits between classical force
fields and full Kohn-Sham DFT on the accuracy-versus-cost spectrum. The
idea is to approximate the DFT total energy by expanding it around a
reference density, truncating aggressively, and folding the expensive
integrals into a small set of pre-tabulated parameters, so that runtime
cost scales like tight-binding.

### Derivation sketch

Start from the Kohn-Sham total energy and split the density into a
neutral-atom reference $n_0$ (a superposition of free-atom densities)
plus a fluctuation $\delta n$:

$$
E[n] = T_s[n] + E_{\rm ext}[n] + E_{H}[n] + E_{xc}[n] + E_{nn},
\qquad n = n_0 + \delta n .
$$

Expanding $E[n]$ about $n_0$ and keeping terms order by order in
$\delta n$ gives vibe-qc's two DFTB levels.

**DFTB0 (`DFTB0Model`, non-self-consistent).** The first-order term
vanishes at the reference, leaving a band-structure sum over occupied
orbitals plus a repulsive pair potential:

$$
E_{\rm DFTB0} = \sum_i f_i\, \langle \psi_i | \hat{H}^0 | \psi_i \rangle
              + E_{\rm rep}\big(\{R_{AB}\}\big),
$$

with occupations $f_i$. $\hat{H}^0$ is the reference Hamiltonian and
$E_{\rm rep}$ absorbs the double-counting and ion-ion repulsion as a
short-range pair term. No SCF loop is needed, which is why DFTB0 is the
fastest option and the default for preoptimization.

**SCC-DFTB (`SCCDFTBModel`, second order).** The second-order term in
$\delta n$ becomes a self-consistent-charge correction built from
atomic charge fluctuations $\Delta q_A$:

$$
E_2 = \tfrac{1}{2} \sum_{A,B} \Delta q_A\, \gamma_{AB}\, \Delta q_B .
$$

$\gamma_{AB}$ is an effective Coulomb kernel that runs from the on-site
chemical hardness (the Hubbard $U$) at short range to $1/R$ at long
range. Because the charges depend on the orbitals and the orbitals
depend on the charge-dependent potential, this needs a self-consistent
loop over charges (hence "SCC"). It markedly improves polar bonds,
charged species, and anything with real charge transfer.

vibe-qc stops at second order. Third-order DFTB3 is not implemented; the
more broadly parametrized GFN2-xTB, a relative in the same
expand-and-tabulate family, ships separately as an experimental method
(see the [semiempirical guide](../user_guide/semiempirical.md)).

### How vibe-qc builds the Hamiltonian, the charges, and γ

Canonical DFTB codes such as DFTB+ store $H^0_{\mu\nu}$ and $S_{\mu\nu}$
as Slater-Koster tables, tabulated per element pair from confined-atom
DFT and rotated to the bond axis at runtime. **vibe-qc takes a lighter,
parametrized route** over a minimal valence basis of Slater-type
orbitals (STO-6G expansions):

* **Overlap** $S_{\mu\nu}$ is evaluated directly with libint from the
  minimal basis, not tabulated.
* **Off-diagonal $H^0$** uses a Wolfsberg-Helmholz (extended-Hückel)
  form. With $\bar{h}_{AB} = \tfrac{1}{2}(h_A + h_B)$ the mean on-site
  energy of the two atoms,
  $$
  H^0_{\mu\nu} = \tfrac{1}{2}\,\kappa\, S_{\mu\nu}\, \bar{h}_{AB},
  \qquad \mu \in A,\ \nu \in B,
  $$
  where $\kappa$ is the Wolfsberg-Helmholz constant. Diagonal elements
  are the tabulated on-site orbital energies.
* **Charges** are Mulliken populations,
  $\Delta q_A = n_A^{\rm val} - \sum_{\mu \in A} (DS)_{\mu\mu}$.
* **$\gamma_{AB}$** uses an Ohno-Klopman damped Coulomb,
  $\gamma_{AB} = 1/\sqrt{R_{AB}^2 + \eta^2}$ with
  $\eta = \tfrac{1}{2}(1/U_A + 1/U_B)$, and on-site $\gamma_{AA} = U_A$.
* The SCC step shifts the Hamiltonian to
  $H_{\mu\nu} = H^0_{\mu\nu} + \tfrac{1}{2} S_{\mu\nu}(V_A + V_B)$, with
  the on-atom potential $V_A = \sum_C \gamma_{AC}\,\Delta q_C$, iterated
  to charge self-consistency.

The on-site energies, Hubbard $U$ values, $\kappa$, and repulsive pair
potentials are vibe-qc's own in-house parameter set (91 elements; see
the Element coverage section below). They are **not** the dftb.org mio
or 3ob Slater-Koster sets and are not tuned for DFTB+ parity, so
absolute energies differ from a DFTB+ run on the same system. The
repulsive term uses in-house pair parameters, falling back to a default
$R^{-12}$ estimator for elements beyond the core set (H, C, N, O, F, P,
S, Cl).

### What DFTB gets right, and where it breaks

Reliable: geometries of organic and main-group molecules, relative
conformer energies, vibrational trends, and qualitative bonding, which
is what makes it a good screening and preoptimization tool at a small
fraction of a DFT step.

Shaky: strongly ionic solids, f-electron systems, and transition-metal
compounds with strong multicenter bonding, where the two-center,
minimal-basis picture is too coarse; London dispersion unless you add a
D3 or D4 correction explicitly (vibe-qc exposes
`DispersionCorrectedModel`, shown above); and any system far from the
neutral-atom references the parameters were built around.

The deeper point: DFTB accuracy is fixed by the parametrization and the
minimal basis, so you cannot systematically converge it the way you
converge Kohn-Sham DFT with a larger basis and a better functional. Its
error bars are empirical rather than controlled. Treat it as a fast
first pass, then refine the survivors with full DFT (the Preoptimization
to DFT handoff section below). vibe-qc-specific caveats (the SCC
gradient approximation, periodic gradients, repulsive coverage) are in
Known limitations at the end.

Method references: non-SCC DFTB, Porezag, Frauenheim, Köhler, Seifert,
Kaschner, *Phys. Rev. B* **51**, 12947 (1995); SCC-DFTB, Elstner,
Porezag, Jungnickel, Elsner, Haugk, Frauenheim, Suhai, Seifert, *Phys.
Rev. B* **58**, 7260 (1998); the Wolfsberg-Helmholz $H^0$ form,
Wolfsberg, Helmholz, *J. Chem. Phys.* **20**, 837 (1952).


## Quick start, H₂O single‑point

A single point is just constructing a model on a `Molecule` and asking for
its energy (and gradient). This builds water at its experimental geometry
and evaluates both the DFTB0 and SCC-DFTB energies:

```python
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import DFTB0Model, SCCDFTBModel

# H₂O at experimental geometry (bohr)
import numpy as np
theta = np.deg2rad(104.5 / 2)
mol = Molecule(
    [
        Atom(8,  [0.0, 0.0, 0.0]),
        Atom(1,  [1.81 * np.sin(theta), 1.81 * np.cos(theta), 0.0]),
        Atom(1,  [-1.81 * np.sin(theta), 1.81 * np.cos(theta), 0.0]),
    ],
    charge=0,
    multiplicity=1,
)

# Non‑self‑consistent (fastest)
dftb0 = DFTB0Model(mol)
print(f"DFTB0:  {dftb0.energy():12.6f} Ha")
print(f"gradient:\n{dftb0.gradient()}")

# Self‑consistent‑charge (more accurate charges)
scc = SCCDFTBModel(mol)
print(f"SCC:    {scc.energy():12.6f} Ha")
```


## Geometry optimization

Use the ASE calculator interface for geometry optimization:

```python
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import DFTB0Model, SemiempiricalParameters
from ase import Atoms, units
from ase.optimize import BFGSLineSearch
from ase.calculators.calculator import Calculator
import numpy as np

class DFTB0Calculator(Calculator):
    """ASE Calculator wrapping DFTB0."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, params=None):
        super().__init__()
        self.params = params or SemiempiricalParameters.dftb0_default()

    def calculate(self, atoms, properties, system_changes):
        pos_bohr = atoms.positions / units.Bohr
        mol = Molecule(
            [Atom(int(z), list(xyz))
             for z, xyz in zip(atoms.numbers, pos_bohr)],
            charge=0, multiplicity=1,
        )
        model = DFTB0Model(mol, params=self.params)
        self.results["energy"] = model.energy() * units.Hartree
        self.results["forces"] = -model.gradient() * (
            units.Hartree / units.Bohr
        )

# Build water molecule
theta = np.deg2rad(104.5 / 2)
mol = Molecule(
    [Atom(8, [0, 0, 0]),
     Atom(1, [1.81 * np.sin(theta), 1.81 * np.cos(theta), 0]),
     Atom(1, [-1.81 * np.sin(theta), 1.81 * np.cos(theta), 0])],
    charge=0, multiplicity=1,
)

atoms = Atoms(
    numbers=[8, 1, 1],
    positions=np.array([a.xyz for a in mol.atoms]) * units.Bohr,
)
atoms.calc = DFTB0Calculator()
opt = BFGSLineSearch(atoms)
opt.run(fmax=0.05, steps=50)

final = atoms.positions / units.Bohr
for i in range(3):
    print(f"Atom {i}: {final[i]}")
```


## Open‑shell, NO₂ radical

Open-shell systems use the unrestricted model classes. This runs the NO₂
doublet radical through `UDFTB0Model` and `USCCDFTBModel`:

```python
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import UDFTB0Model, USCCDFTBModel

no2 = Molecule(
    [
        Atom(7, [0.0, 0.0, 0.0]),
        Atom(8, [2.2, 0.0, 0.0]),
        Atom(8, [-1.1, 1.9, 0.0]),
    ],
    charge=0,
    multiplicity=2,  # doublet
)

udftb0 = UDFTB0Model(no2)
print(f"UDFTB0: {udftb0.energy():.6f} Ha")

uscc = USCCDFTBModel(no2)
print(f"USCC:   {uscc.energy():.6f} Ha")
```


## Periodic systems

DFTB also runs under periodic boundary conditions via `PeriodicSystem`.
This sets up a 1D carbon chain and screens it with the Γ-point DFTB0 and
SCC drivers:

```python
from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
import numpy as np

# 1D carbon chain at 3.0 bohr spacing
atoms = [Atom(6, [0.0, 0.0, 0.0])]
cell = np.diag([3.0, 30.0, 30.0])
system = PeriodicSystem(1, cell, atoms, charge=0, multiplicity=1)

params = _se.SemiempiricalParameters.dftb0_default()

# Gamma‑point DFTB0
result = _se.run_dftb0_gamma(system, params)
print(f"Periodic DFTB0: {result.energy:.6f} Ha/cell")

# SCC variant
opts = _se.PeriodicSCCOptions()
opts.max_iter = 50
scc_result = _se.run_scc_dftb_gamma(system, params, opts)
print(f"Periodic SCC:   {scc_result.energy:.6f} Ha/cell")
print(f"  converged: {scc_result.converged}")
```


## Dispersion corrections

To add London dispersion, wrap any DFTB model in `DispersionCorrectedModel`
with a D3(BJ) parameter set; the energy then includes the dispersion term:

```python
from vibeqc.semiempirical import (
    SCCDFTBModel, DispersionCorrectedModel,
    DFTB_D3_DEFAULTS, d3bj_energy,
)

# Build a dispersion‑corrected model
scc = SCCDFTBModel(mol)
disp_model = DispersionCorrectedModel(scc, DFTB_D3_DEFAULTS)
print(f"SCC + D3(BJ): {disp_model.energy():.6f} Ha")
```


## Preoptimization → DFT handoff

The primary use case: preoptimize with DFTB, then refine with DFT.

```python
from vibeqc.semiempirical.preoptimize import preoptimize_molecule
from vibeqc import Molecule, Atom

# Starting geometry (rough guess)
mol = Molecule(
    [Atom(8, [0, 0, 0]),
     Atom(1, [1.5, 0, 0]),
     Atom(1, [-1.0, 1.2, 0])],
    charge=0, multiplicity=1,
)

# Preoptimize with DFTB0
opt_mol = preoptimize_molecule(mol, method="dftb0")
print("Preoptimized geometry:")
for a in opt_mol.atoms:
    print(f"  Z={a.Z}: {a.xyz}")

# Now feed into run_job for DFT refinement:
# from vibeqc import run_job
# run_job(opt_mol, basis="6-31G*", method="rks", functional="pbe")
```


## Parameter customisation

The parameter set is fully inspectable and editable. This queries the
built-in sets for per-element data and then builds a custom set from
scratch with explicit on-site energies, Hubbard $U$, and repulsive pairs:

```python
from vibeqc.semiempirical import SemiempiricalParameters

# Built‑in parameter sets
default = SemiempiricalParameters.dftb0_default()
production = SemiempiricalParameters.dftb0_production()

# Query element data
print(f"O on‑site (s): {default.on_site_energy(8, 0):.4f} Ha")
print(f"O Hubbard U:   {default.hubbard_u(8):.4f} Ha")
print(f"kappa:          {default.kappa}")

# Build custom parameters
custom = SemiempiricalParameters()
custom.add_element(
    Z=1, on_site=[-0.21], zeta=[1.24],
    hubbard_u=0.42, valence_electrons=1,
)
custom.add_element(
    Z=8, on_site=[-0.89, -0.33], zeta=[2.25, 2.25],
    hubbard_u=0.45, valence_electrons=6,
)
# Set repulsive pair (R⁻¹² form)
custom.set_repulsive_pair_analytic(1, 1, A=5.0)
custom.set_repulsive_pair_analytic(1, 8, A=15.0)
custom.set_repulsive_pair_analytic(8, 8, A=40.0)
```


## Element coverage

The built-in DFTB parameter sets cover **91 elements** (H-U except
Po/Z=84), including the 3d/4d/5d transition metals, lanthanides
(La-Lu), and early actinides (Ac-U). All parameters are in-house
estimates; DFT-fitted production repulsive potentials are deferred.

Check coverage:

```python
params = SemiempiricalParameters.dftb0_default()
elements = [Z for Z in range(1, 93) if params.has_element(Z)]
print(f"{len(elements)} elements: {elements[:10]}...")
```


## Performance tips

- **DFTB0** is 3-5× faster than SCC‑DFTB (no SCF loop).  Use it
  for preoptimization where the SCC charge correction is less
  important.
- **Gradients** are available for all molecular DFTB variants. DFTB0
  and UDFTB0 gradients match finite differences tightly; SCC gradients
  are approximate.
- **Periodic systems** support Gamma-point and DFTB k-point paths with
  lattice sums. For very tight cells, increase `cutoff_bohr` and verify
  against a higher-level calculation.
- **Memory** is negligible, the basis is minimal (one function
  per valence shell).


## Known limitations

- SCC and USCC gradients use an approximate fixed‑charge formula
  (see [roadmap](../roadmap.md)).  For quantitative geometry
  optimisation, prefer DFTB0.
- DFTB0 and SCC-DFTB have periodic k-point paths, but periodic
  gradients/stress remain Gamma-point oriented. Treat tight-cell
  production use as experimental until validated for the target system.
- Repulsive pair potentials for elements beyond the core set
  (H, C, N, O, F, P, S, Cl) use a default R⁻¹² estimator.
