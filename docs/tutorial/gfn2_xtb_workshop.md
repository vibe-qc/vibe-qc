---
myst:
  html_meta:
    "description": "GFN2-xTB workshop -- a complete walkthrough of vibe-qc's extended tight-binding method, mirroring the Grimme lab xtb workshop. The xTB Hamiltonian (extended-Hueckel H0, shell-resolved second-order electrostatics, anisotropic multipoles, third-order on-site term, D4, the SCC iteration), its experimental status in vibe-qc, single-point energy, geometry optimization, spin polarization, frequencies, PES scans, molecular dynamics, and periodic calculations."
    "og:title": "vibe-qc tutorial -- GFN2-xTB workshop"
---

(gfn2_xtb_workshop)=
# GFN2-xTB Workshop

This tutorial is a hands-on walkthrough of vibe-qc's GFN2-xTB
implementation, structured to mirror the
[Grimme lab xtb workshop](https://grimme-lab.github.io/workshops/page/xtb).
vibe-qc ships its own native GFN2-xTB engine -- not a wrapper -- with
molecular and Gamma-point periodic single-point energies, analytic gradients,
post-SCF D4 dispersion, and spin polarization.

```{admonition} Status: experimental
:class: warning

GFN2-xTB (`method="gfn2_xtb"`) is an experimental route. It emits
`GFN2ExperimentalWarning` on every evaluation through the public API
(`GFN2Model(..., warn=False)` silences it deliberately), and the route is
closed-shell only at every public boundary. The H{sup}`0` shape, shell-resolved
SCC, third-order term, molecular AES, analytic gradients and post-SCF native D4
are implemented and finite-difference validated; the remaining production
gates are faithful periodic AES image-cell multipole terms, periodic
molecular-limit parity, and the full external-parity matrix against the `xtb`
reference. Its numbers are research output, not validated results, and it is
not part of the release gate. Use it for screening and preoptimisation. The
label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels);
the per-route table is at {ref}`semiempirical-status`.
```

All numbers quoted on this page come from a vibe-qc 0.17.x development
build on Apple M2; expect the last digits to move between builds.

All runnable examples for this workshop live in
[`examples/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/semiempirical/)
as `30_gfn2_*.py` through `37_gfn2_*.py`.

---

## Theory

GFN2-xTB ({vibeqc-cite}`bannwarth_gfn2_2019`) is the second member of the
*geometry, frequency, non-covalent* extended tight-binding family that
began with GFN1-xTB ({vibeqc-cite}`grimme_gfn1_2017`). Like DFTB it is a
second-order expansion of the DFT energy around a reference density of
superposed atoms, but it keeps more of the physics: shell-resolved rather
than atomic charge fluctuations, anisotropic (dipole and quadrupole)
electrostatics, a third-order on-site term and a built-in dispersion
correction, parametrised for the whole periodic table up to radon against
reference data from higher-level calculations (geometries, frequencies and
non-covalent interaction energies) rather than against experiment. The total energy is a
sum of five pieces,

$$
E_{\text{GFN2}} = E_{\text{rep}} + E_{\text{EHT}} + E_{\text{IES+IXC}} + E_{\text{AES+AXC}} + E_{\text{disp}} ,
$$

which map onto the components the code reports: `e_repulsive` is
$E_{\text{rep}}$, `e_electronic` is the band-structure energy from the
zeroth-order Hamiltonian, `e_scc` collects the second- and third-order
charge terms together with the anisotropic electrostatics, and the D4
dispersion is added after the SCC.

**The zeroth-order Hamiltonian.** The basis is a minimal valence set of
Slater functions (expanded in Gaussians, with a polarisation shell on some
elements), and the reference Hamiltonian has the extended-Hueckel form

$$
H^0_{\mu\nu} = \tfrac12\, K_{ll'}\, S_{\mu\nu}\,\bigl(h_l^A + h_{l'}^B\bigr)\,
X(\mathrm{EN}_A,\mathrm{EN}_B)\,\Pi(R_{AB}; l, l') ,
\qquad
h_l^A = h_l^{A,0} - k_{\mathrm{CN}}^{A,l}\,\mathrm{CN}_A ,
$$

where $\mu$ belongs to shell $l$ of atom $A$ and $\nu$ to shell $l'$ of atom
$B$, $S_{\mu\nu}$ is the overlap, $K_{ll'}$ a shell-pair scaling constant,
$h_l^A$ the shell self-energy shifted linearly by the fractional
coordination number $\mathrm{CN}_A$ (this is what "CN-dependent shell
shifts" means in the components table below), $X$ an electronegativity
correction $1 + k_{\mathrm{EN}}(\mathrm{EN}_A - \mathrm{EN}_B)^2$ that
lowers the coupling between dissimilar atoms, and $\Pi$ a distance-dependent
polynomial that sharpens the bonding at short range. Diagonal elements are
the self-energies themselves. The band-structure energy is
$\sum_i n_i \langle\psi_i|\hat H^0|\psi_i\rangle$ over the occupied
orbitals with occupations $n_i$, the analogue of the DFTB0 energy.
$E_{\text{rep}}$ is a pairwise, exponentially decaying atom-atom repulsion
with element-pair parameters.

**Second-order isotropic electrostatics and the gamma function.** Charge
fluctuations are resolved per shell: $q_A^l$ is the Mulliken charge of
shell $l$ on atom $A$ relative to its neutral-atom occupation. They
interact through

$$
E_{\text{IES+IXC}} = \tfrac12 \sum_{A,B}\sum_{l\in A,\,l'\in B}
q_A^l\, \gamma_{AB}^{ll'}\, q_B^{l'}
+ \tfrac13 \sum_A \Gamma_{A}\,q_A^3 ,
\qquad
\gamma_{AB}^{ll'} = \frac{1}{\sqrt{R_{AB}^2 + \eta_{AB,ll'}^{-2}}} ,
$$

where $\gamma$ is the Mataga-Nishimoto-Ohno-Klopman kernel that tends to
$1/R_{AB}$ at long range and to the averaged shell hardness $\eta$ (the
Hubbard parameter, scaled per shell) at $R_{AB}=0$, so the on-site term
$\gamma_{AA}^{ll'}$ is finite and carries the exchange-correlation part of
the second-order response ("IXC"). The cubic term with the on-site
derivative $\Gamma_A$ of the hardness, acting on the atomic charge
$q_A = \sum_{l\in A} q_A^l$, is the **third-order** contribution
(`GAM3` in the components table); it is what lets the method describe
strongly charged fragments and very polar bonds without the charges
running away, the same physics that DFTB3 adds to SCC-DFTB.

**Anisotropic multipole electrostatics.** The charge distribution on an
atom is not spherical, so GFN2 also carries cumulative atomic dipoles
$\boldsymbol\mu_A$ and traceless quadrupoles $\boldsymbol\Theta_A$, obtained
from the density matrix and the dipole and quadrupole integrals over the
basis. They interact with the monopoles and with each other through the
classical charge-dipole ($R^{-2}$), dipole-dipole and charge-quadrupole
($R^{-3}$) kernels, each multiplied by a short-range damping function
$f_n(R_{AB}) = \bigl[1 + 6\,(R_0^{AB}/R_{AB})^{a_n}\bigr]^{-1}$ that switches
the interaction off inside the covalent radius, plus on-site anisotropic
exchange-correlation terms
$\sum_A \bigl(f^{\mu}_{\mathrm{XC},A}\,|\boldsymbol\mu_A|^2 + f^{\Theta}_{\mathrm{XC},A}\,\|\boldsymbol\Theta_A\|^2\bigr)$
that penalise unphysical polarisation. This AES block is what distinguishes
GFN2 from GFN1 and from every DFTB flavour; it is the main reason GFN2
handles hydrogen bonds, halogen bonds and metal coordination without
element-pair-specific corrections. `opts.aes_faithful` selects the published
atom-resolved or legacy shell-resolved model (see "The AES faithfulness flag");
the periodic image-cell multipole sums are the open production gate.

**Dispersion.** London dispersion enters through the D4 model: pairwise
$C_6$ and $C_8$ coefficients interpolated from reference polarisabilities as
a function of coordination number *and* atomic charge, damped with the
Becke-Johnson function, plus an Axilrod-Teller-Muto three-body term. In the
reference `xtb` code the charge dependence makes D4 part of the SCC; vibe-qc
evaluates it once after the SCC with the converged charges (the
"post-SCF native D4" of the element-coverage section), which is exact for the
energy and differs from the self-consistent form only through the small
charge response.

**The SCC iteration.** Every term beyond $H^0$ depends on the charges and
multipoles, which depend on the orbitals. The code therefore builds
$H = H^0 + H^1(q, \boldsymbol\mu, \boldsymbol\Theta)$, where $H^1_{\mu\nu}$
is the overlap-weighted average of the shell potentials
$\partial E/\partial q_A^l$ and their multipole analogues, solves the
generalised eigenproblem $H C = S C \varepsilon$, recomputes Mulliken
shell charges and cumulative multipoles from the new density, mixes them
with the previous iterate, and repeats until the charges stop changing. The
mixer matters more than in an ab initio SCF because the multipoles make the
fixed-point map stiff; section 9 lists the simple, DIIS and Broyden options
and why the periodic driver defaults to a joint charge-and-moment Broyden
mixer. Fractional occupations at a finite electronic temperature (Fermi
smearing) smooth the map further and are on by default in the periodic
driver.

**Where it helps and where it fails.** GFN2 is built for what its name
says: geometries, vibrational frequencies and non-covalent interaction
energies of molecules of any composition, at a cost that allows thousands
of conformers or long dynamics. It is not built for reaction energies or
barriers to chemical accuracy, its absolute energies live on the method's
own scale (section 2), open-shell states are approximate at best (and
closed-shell only in vibe-qc), metals and strongly correlated systems are
outside the model, and anything the parametrisation never saw, such as
exotic oxidation states, can fail without warning. Treat it as the fast
first pass whose survivors go to DFT.

---

## 1. Installation -- parameters fetch

vibe-qc ships the GFN2-xTB engine code but **not** the 86-element
parameter set (LGPL-3.0 licensed). On first use, parameters are fetched
automatically from the upstream Grimme-group repository and cached under
`$VIBEQC_GFN2_CACHE_DIR`, else `$XDG_CACHE_HOME/vibeqc/`, else
`~/.cache/vibeqc/`. No separate download step is needed.

On a host with no outbound network the fetch cannot happen and GFN2 fails
closed rather than guessing. Seed the cache from a networked host and point
the jobs at it, as described under "Where the parameter cache lives, and
seeding it for offline hosts" in
[the user guide](../user_guide/semiempirical.md).

```python
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()  # auto-fetches and caches
print(f"Loaded {params.n_elements()} elements")
print(f"Has carbon: {params.has_element(6)}")
```

```text
Loaded 86 elements
Has carbon: True
```

The `GFN2ParameterSet` carries per-element data (zeta exponents, Hubbard U,
on-site energies), per-shell CN-dependent self-energy and hardness
polynomials (internal C++), element-pair repulsive potentials, and D4
damping parameters. You can inspect individual elements:

```python
ed = params.element(6)   # carbon (CoreElementData)
print(f"C: Z={ed.Z}")
print(f"   zeta:         {list(ed.zeta)}")
print(f"   Hubbard U:     {ed.hubbard_u:.4f}")
print(f"   on-site en:    {[f'{v:.4f}' for v in ed.on_site]}")
print(f"   valence e⁻:    {ed.valence_electrons}")

print(f"\nD4 damping: s8={params.d4_s8:.3f}  a1={params.d4_a1:.3f}  a2={params.d4_a2:.3f}")
```

---

## 2. Single-point energy

The simplest entry point is `run_job`:

```python
from vibeqc import Molecule, Atom, run_job

mol = Molecule([
    Atom(8, [ 0.00,  0.00,  0.00]),
    Atom(1, [ 1.43,  0.98,  0.00]),
    Atom(1, [-1.43,  0.98,  0.00]),
])

run_job(mol, method="gfn2_xtb", output="h2o_gfn2")
```

The console log carries the total:

```text
SCF converged in 47 iterations; E = -5.0650921073 Ha
```

For programmatic use, the `GFN2Model` class gives direct access to
energy, gradient, charges, MO energies, and convergence diagnostics:

```python
from vibeqc.semiempirical.methods.gfn2 import GFN2Model

model = GFN2Model(mol, params=params, warn=False)
energy = model.energy()
print(f"GFN2-xTB energy: {energy:.8f} Ha")
print(f"Converged: {model.converged} in {model.n_iter} iterations")
```

```text
GFN2-xTB energy: -5.06509211 Ha
Converged: True in 47 iterations
```

### Energy components

The GFN2-xTB total energy decomposes into four physical terms:

```python
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb

opts = _xtb.XTBSccOptions()
result = _xtb.run_gfn2_xtb(mol, params, opts)
print(f"Electronic (H0 + H1): {result.e_electronic:12.6f} Ha")
print(f"Repulsive:            {result.e_repulsive:12.6f} Ha")
print(f"SCC (H1 + GAM3):      {result.e_scc:12.6f} Ha")
print(f"Total (incl D4):      {model.energy():12.6f} Ha")
print(f"Mulliken charges:     {[f'{q:+.4f}' for q in result.charges]}")
```

| Term | Meaning |
|------|---------|
| `e_electronic` | Band-structure energy from H{sup}`0` with CN-dependent shell shifts (water: -5.1155 Ha) |
| `e_repulsive` | Short-range pairwise atom-atom repulsion (+0.0506 Ha) |
| `e_scc` | Second- and third-order (GAM3) charge-fluctuation correction plus the anisotropic electrostatics (+0.0247 Ha) |
| D4 | Post-SCF dispersion (added by `GFN2Model.energy()`) |

The Mulliken charges for this geometry are O -0.568 and H +0.284.

### Understanding the energy scale

GFN2-xTB total energies live on the method's own parametrised reference
scale. Water at GFN2-xTB is approximately −5.07 Ha; the same water at
HF/6-31G\* is −76.01 Ha. The numbers are not comparable across methods.
What *is* meaningful within GFN2-xTB:

- conformer energy differences
- reaction energies (isogyric/isodesmic)
- barrier heights
- relative stability ordering

---

## 3. Experimental open-shell occupations

The low-level `run_ugfn2_xtb` driver supports different alpha and beta
occupations of a common Hamiltonian. It counts the parameterized valence
electrons minus the molecular charge. Incompatible multiplicities,
negative electron counts and occupations exceeding the basis capacity
raise `ValueError`; occupations are never truncated or clamped.

This legacy driver is experimental. Its atom-charge SCC loop does not
implement the closed-shell driver's shell SCC, third-order and AES terms,
or the Python model's D4 correction. Electron-count validation does not
establish quantitative open-shell GFN2 accuracy. Only the native low-level
entry point is available; `run_job` remains closed-shell for GFN2.
Its configurable iteration controls are `max_iter`, `conv_tol_charge`, and
`charge_mixing`. The shared options must select `scc_mixer=Simple`,
`electronic_temperature=0`, `gamma_form=KlopmanOhno`, and
`aes_faithful=False`; other values raise `ValueError`. The restricted
driver's `auto_stabilize` ladder, accelerator history, and AES damping
controls do not apply to this partial model.

```python
from vibeqc._vibeqc_core import Atom as CAtom, Molecule as CMolecule
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb

# OH radical (doublet)
oh = CMolecule([CAtom(8, [0, 0, 0]), CAtom(1, [0, 0, 1.81])], 0, 2)

opts = _xtb.XTBSccOptions()
result = _xtb.run_ugfn2_xtb(oh, params, opts)
if not result.converged:
    raise RuntimeError("Experimental U-GFN2 SCC did not converge")
print(f"U-GFN2-xTB OH: {result.energy:.8f} Ha")
print(f"n_alpha={result.n_alpha}  n_beta={result.n_beta}")
print(f"converged={result.converged}  n_iter={result.n_iter}")
```

```text
n_alpha=4  n_beta=3
```

OH has seven valence electrons, with oxygen's 1s pair frozen. The fuller
native `run_gfn2_xtb` accepts open shells only with `aes_faithful=True`.
It retains the shell SCC, third-order and faithful AES functional, and can
consume explicit SCC D4 operands. Its common orbitals carry separate alpha
and beta occupations, with separate Fermi levels at positive temperature.
Inspect `occupations_alpha`, `occupations_beta`, `n_alpha`, and `n_beta`, and
check `converged` before using a result. This remains an experimental native
route; `run_job` is still closed-shell, and quantitative open-shell accuracy
requires further validation. The legacy example above uses the partial
atom-charge model and is not a substitute for this fuller functional.
For elements with an f-in-core model, check the valence spin count rather
than inferring it from the sum of atomic numbers alone. The `Molecule`
constructor also requires compatible all-electron spin parity. Cases such
as an isolated neutral Ce atom cannot currently satisfy both constraints;
they are rejected instead of silently producing a different charge state.

Triplet O{sub}`2`:

```python
# O2 triplet, R ≈ 2.3 bohr
o2 = CMolecule([CAtom(8, [0, 0, 0]), CAtom(8, [0, 0, 2.3])], 0, 3)
result = _xtb.run_ugfn2_xtb(o2, params, opts)
print(f"U-GFN2-xTB O2 (triplet): {result.energy:.8f} Ha")
```

```text
U-GFN2-xTB O2 (triplet): -7.90043985 Ha
```

---

## 4. Geometry optimization

### Via `run_job`

```python
run_job(
    mol, method="gfn2_xtb",
    output="h2o_gfn2_opt",
    optimize=True, fmax=0.02,  # eV/Å
)
```

This produces `h2o_gfn2_opt.out` (optimization log), `h2o_gfn2_opt.traj`
(per-step trajectory, visualisable with `ase gui`), and the final
`.qvf` archive.

### Via ASE

For fine-grained control over the optimizer and convergence criteria,
wire up the ASE Calculator:

```python
import warnings
from ase import Atoms
from ase.optimize import BFGS
from ase.units import Bohr
from ase.calculators.calculator import Calculator
import numpy as np
from vibeqc import Molecule as VMol, Atom as VAtom
from vibeqc.semiempirical.methods.gfn2 import GFN2Model

class GFN2Calculator(Calculator):
    """Minimal ASE Calculator backed by GFN2Model."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, params):
        super().__init__()
        self._params = params

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        mol = VMol(
            [VAtom(int(z), (pos / Bohr).tolist())
             for z, pos in zip(atoms.numbers, atoms.positions)],
            charge=0, multiplicity=1,
        )
        m = GFN2Model(mol, params=self._params, warn=False)
        energy_ha = m.energy()
        gradient_ha_bohr = np.asarray(m.gradient())
        from ase.units import Hartree
        self.results["energy"] = energy_ha * Hartree
        self.results["forces"] = -gradient_ha_bohr * (Hartree / Bohr)

# Build ASE Atoms (angstrom)
atoms = Atoms('OH2', positions=[
    [0.0, 0.0, 0.0],
    [0.756, 0.519, 0.0],
    [-0.756, 0.519, 0.0],
])

atoms.calc = GFN2Calculator(params)

opt = BFGS(atoms)
opt.run(fmax=0.005)  # eV/angstrom; tight enough for the frequencies of section 5
print(f"Optimised energy: {atoms.get_potential_energy():.8f} eV")
```

```text
Optimised energy: -137.96245362 eV
```

That is -5.07002670 Ha after seven BFGS steps, with r(OH) = 0.967 Å and
∠HOH = 104.3°.

---

## 5. Vibrational frequencies

Compute the Hessian by finite-differencing the analytic gradients at the
geometry optimised in section 4 (harmonic frequencies are only defined at a
stationary point, and a loose optimiser tolerance shows up directly as large
rotational "frequencies"), mass-weight it with ASE's atomic-mass table, and
diagonalise:

```python
import numpy as np
from ase.data import atomic_masses
from ase.units import Bohr
from vibeqc import Molecule, Atom
from vibeqc.semiempirical.methods.gfn2 import GFN2Model

def hessian_fd(model, h=0.005):
    """Central-difference Hessian from GFN2Model analytic gradients."""
    mol = model.molecule
    n = len(mol.atoms)
    H = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for c in range(3):
            xyz_p = list(mol.atoms[a].xyz)
            xyz_p[c] += h
            mol_p = Molecule(
                [Atom(at.Z, xyz_p if i == a else list(at.xyz))
                 for i, at in enumerate(mol.atoms)], mol.charge, mol.multiplicity)
            g_p = np.asarray(GFN2Model(mol_p, params=model.params, warn=False).gradient()).flatten()

            xyz_m = list(mol.atoms[a].xyz)
            xyz_m[c] -= h
            mol_m = Molecule(
                [Atom(at.Z, xyz_m if i == a else list(at.xyz))
                 for i, at in enumerate(mol.atoms)], mol.charge, mol.multiplicity)
            g_m = np.asarray(GFN2Model(mol_m, params=model.params, warn=False).gradient()).flatten()

            H[3 * a + c, :] = (g_p - g_m) / (2.0 * h)
    return (H + H.T) / 2.0

# Water at the geometry optimised in section 4 (ASE `atoms`, angstrom -> bohr)
mol = Molecule([
    Atom(int(z), (pos / Bohr).tolist())
    for z, pos in zip(atoms.numbers, atoms.positions)
])
model = GFN2Model(mol, params=params, warn=False)
H = hessian_fd(model)

# Mass-weight (amu) and diagonalise
import math
HA_TO_J = 4.359744722e-18
BOHR_TO_M = 5.291772109e-11
AMU_KG = 1.66053906660e-27
C_CM = 2.99792458e10
_HESS_TO_CM1 = math.sqrt(HA_TO_J / (BOHR_TO_M**2 * AMU_KG)) / (2 * math.pi * C_CM)
masses = np.array([atomic_masses[at.Z] for at in mol.atoms])
M_inv_sqrt = np.diag(1.0 / np.sqrt(np.repeat(masses, 3)))
H_mw = M_inv_sqrt @ H @ M_inv_sqrt
eigvals = np.linalg.eigvalsh(H_mw)

# Signed square root: the six translations/rotations come out near zero
freqs_cm1 = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * _HESS_TO_CM1
print("Harmonic frequencies (cm⁻¹):")
for i, f in enumerate(np.sort(freqs_cm1)):
    print(f"  mode {i+1:2d}: {f:8.1f} cm⁻¹")
```

```text
Harmonic frequencies (cm⁻¹):
  mode  1:    -16.6 cm⁻¹
  mode  2:     -9.8 cm⁻¹
  mode  3:     -0.3 cm⁻¹
  mode  4:     -0.0 cm⁻¹
  mode  5:      0.0 cm⁻¹
  mode  6:      4.7 cm⁻¹
  mode  7:   1528.1 cm⁻¹
  mode  8:   3484.9 cm⁻¹
  mode  9:   3546.4 cm⁻¹
```

The last three are the harmonic GFN2-xTB frequencies of water at its own
minimum: the bend, the symmetric stretch and the asymmetric stretch
(experiment: 1595, 3657 and 3756 cm⁻¹; the stretches come out about 5 percent
low, which is typical for GFN2 without scaling). The first six are the
translations and rotations, which should vanish: their residual of a few
wavenumbers measures the finite-difference step and the optimiser tolerance,
not physics (with the looser `fmax=0.02` the same code gives three rotational
values near -45 cm⁻¹), so tighten both if you need them closer to zero.

---

## 6. Molecular dynamics

Wire GFN2-xTB as the force engine for ASE molecular dynamics:

```python
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# NVE (microcanonical) -- constant energy
dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

energies = []
for step in range(200):
    dyn.run(1)
    e_pot = atoms.get_potential_energy()
    e_kin = atoms.get_kinetic_energy()
    energies.append(e_pot + e_kin)

print(f"Total-energy drift: {np.std(energies):.2e} eV")
print(f"Mean total energy:  {np.mean(energies):.4f} eV")

# NVT (canonical) -- temperature control
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn_nvt = NVTBerendsen(atoms, timestep=1.0 * units.fs,
                        temperature_K=300, taut=100 * units.fs)
```

---

## 7. PES scans

### Bond-length scan

```python
import numpy as np

def scan_bond(mol_template, bond_idx, distances):
    """Return GFN2-xTB energies for a bond-length scan."""
    model = GFN2Model(mol_template, params=params, warn=False)
    result = []
    for r in distances:
        atoms = list(mol_template.atoms)
        a1, a2 = bond_idx
        # displace atom a2 along the bond vector
        vec = np.array(atoms[a2].xyz) - np.array(atoms[a1].xyz)
        vec = vec / np.linalg.norm(vec) * r
        atoms[a2] = Atom(atoms[a2].Z, (np.array(atoms[a1].xyz) + vec).tolist())
        mol_r = Molecule(atoms, mol_template.charge, mol_template.multiplicity)
        m = GFN2Model(mol_r, params=model.params, warn=False)
        result.append(m.energy())
    return np.array(result)

# OH bond scan in water
r_vals = np.linspace(1.2, 3.0, 19)
energies = scan_bond(mol, (0, 1), r_vals)
e_min = energies.min()
for r, e in zip(r_vals, energies):
    print(f"  r(OH) = {r:.2f} bohr  E = {e:.8f} Ha  ΔE = {(e - e_min)*627.509:.2f} kcal/mol")
```

### Angle scan

```python
def scan_angle(mol_template, angle_deg_vals):
    """Return GFN2-xTB energies for a bond-angle scan, keeping both OH bonds equal."""
    result = []
    for angle_deg in angle_deg_vals:
        theta = np.deg2rad(angle_deg / 2.0)
        r_oh = 1.81  # fixed bond length
        atoms = [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
            Atom(1, [-r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
        ]
        mol_a = Molecule(atoms)
        m = GFN2Model(mol_a, params=params, warn=False)
        result.append(m.energy())
    return np.array(result)

angles = np.linspace(70, 140, 15)
e_angles = scan_angle(mol, angles)
e_amin = e_angles.min()
print("\nAngle scan:")
for ang, e in zip(angles, e_angles):
    print(f"  ∠HOH = {ang:5.1f}°  E = {e:.8f} Ha  ΔE = {(e - e_amin)*627.509:.2f} kcal/mol")
```

---

## 8. Periodic calculations

GFN2-xTB supports Gamma-point periodic calculations through the native C++
driver. General k-point meshes and band paths fail closed until the complete
complex Bloch, per-k energy, and AES model is implemented.

### Γ-point: molecular crystal

```python
from vibeqc._vibeqc_core import PeriodicSystem, Atom as PAtom
import numpy as np

# H2O in a 10-bohr cubic box (molecular crystal)
lattice = np.eye(3) * 10.0
atoms = [
    PAtom(8, [0.0, 0.0, 0.0]),
    PAtom(1, [1.43, 0.98, 0.0]),
    PAtom(1, [-1.43, 0.98, 0.0]),
]
system = PeriodicSystem(3, lattice, atoms, charge=0, multiplicity=1)

opts = _xtb.XTBSccOptions()
opts.max_iter = 400
result = _xtb.run_gfn2_xtb_gamma(system, params, opts)
print(f"Periodic GFN2-xTB: E = {result.energy:.8f} Ha")
print(f"  free energy A = {result.free_energy:.8f} Ha")
print(f"  n_cells={result.n_cells}  n_iter={result.n_iter}  converged={result.converged}")
```

```text
Periodic GFN2-xTB: E = -5.07186955 Ha
  free energy A = -5.07186955 Ha
  n_cells=19  n_iter=10  converged=True
```

The free energy equals the energy here because the box is a gapped
molecular crystal: at 0.001 Ha the default frontier smearing produces no
fractional occupation, so the entropy term is zero.

Periodic GFN2-xTB smears the frontier by default (0.001 Ha, about 316 K),
so `result.smearing_temperature`, `result.entropy`, and the Mermin free
energy `result.free_energy = energy - T*S` are populated on every periodic
run. Pass `opts.electronic_temperature = 0.0` explicitly for the exact
zero-temperature Aufbau occupations.

### Graphene: 2D with Γ-point

```python
# Graphene 2-atom unit cell (a = 2.46 Å ≈ 4.65 bohr)
a = 4.65
c = 20.0
lattice_gr = np.array([
    [a, 0, 0],
    [a / 2, a * np.sqrt(3) / 2, 0],
    [0, 0, c],
])
atoms_gr = [
    PAtom(6, [0.0, 0.0, 0.0]),
    PAtom(6, [a / 2, a * np.sqrt(3) / 6, 0.0]),
]
system_gr = PeriodicSystem(2, lattice_gr, atoms_gr, charge=0, multiplicity=1)

opts.max_iter = 400
result_gr = _xtb.run_gfn2_xtb_gamma(system_gr, params, opts)
print(f"Graphene GFN2-xTB: E = {result_gr.energy:.8f} Ha/cell")
print(f"  n_cells={result_gr.n_cells}  converged={result_gr.converged}")
```

```text
Graphene GFN2-xTB: E = -3.78116272 Ha/cell
  n_cells=47  converged=True
```

### k-point sampling

General k-point GFN2 and GFN2 band paths are intentionally unavailable. The
experimental implementation dropped complex Bloch phases, contracted its
bare-band energy with the Gamma Hamiltonian, and omitted AES. Both public
entry points now fail explicitly instead of returning a plausible wrong
answer. Use the Gamma driver above, or use DFTB0/SCC-DFTB when Brillouin-zone
sampling is required.

### Periodic gradients and stress

```python
from vibeqc._vibeqc_core import semiempirical as _se

# Native gradient (including periodic AES response)
grad = _se.compute_periodic_gfn2_gradient(system_gr, result_gr, params)
print(f"Gradient shape: {grad.shape}")
print(f"|grad|_max:     {np.abs(grad).max():.6f} Ha/bohr")

# Stress tensor
stress = _se.compute_periodic_gfn2_stress(system_gr, result_gr, params)
print(f"Stress (Ha/bohr³):\n{np.array2string(stress, precision=6)}")
```

```text
Gradient shape: (2, 3)
|grad|_max:     0.058603 Ha/bohr
Stress (Ha/bohr³):
[[-4.893920e-04  4.556563e-04  0.000000e+00]
 [ 4.556563e-04 -2.868883e-03  0.000000e+00]
 [ 3.354844e-20 -1.501837e-19  0.000000e+00]]
```

The in-plane stress is non-zero because 4.65 bohr is the experimental
lattice constant, not the GFN2-xTB minimum. The third column and row (the
vacuum direction of a `dim=2` system) vanish to numerical precision, and
the finite gradient on the two carbons says the atoms are not at their own
minimum in this idealised, unrelaxed cell either.

---

## 9. Convergence tuning

The default *molecular* GFN2-xTB SCC uses a simple damping mixer with charge
mixing 0.1 and up to 3600 iterations. The *periodic Gamma* driver overrides
that default (issue #409): leaving `scc_mixer` at `Simple` there selects an
Eyert Broyden mixer acting on the joint charge-and-moment state, because
mixing shell charges alone cannot converge a polarisable cell. On MgO the
joint mixer reaches the cubic-symmetric state in 87 iterations where damped
simple mixing does not converge in 3000. Setting `scc_mixer` explicitly opts
back out. For difficult cases, tune the mixer:

```python
opts = _xtb.XTBSccOptions()

# Simple mixer -- robust, slow
opts.scc_mixer = _se.SCCMixer.Simple
opts.charge_mixing = 0.15
opts.max_iter = 1000

# DIIS -- faster for well-behaved systems
opts.scc_mixer = _se.SCCMixer.DIIS
opts.mixer_memory = 8       # DIIS subspace size
opts.mixer_damping = 0.3

# Broyden -- best for periodic / polar / transition metals
opts.scc_mixer = _se.SCCMixer.Broyden
opts.mixer_memory = 12      # Broyden history length
opts.mixer_damping = 0.2
opts.max_iter = 2000

# Electronic temperature (molecular: default 0 = exact Aufbau; periodic
# GFN2-xTB: default 0.001 Ha frontier smearing, explicit 0 restores Aufbau)
opts.electronic_temperature = 0.001  # Ha (~315 K)

# Automatic stabilization -- when the primary solve fails, retry with
# a more conservative recipe. Molecular: unless a temperature was set
# explicitly, the last retry runs at 0.005 Ha; result.smearing_temperature
# records it, and run_job then reports the Mermin free energy and writes
# gfn2_executed_electronic_temperature to the .system record
opts.auto_stabilize = True

result = _xtb.run_gfn2_xtb(mol, params, opts)
print(f"converged={result.converged}  n_iter={result.n_iter}")

# Identical options are forwarded through GFN2Model:
model = GFN2Model(
    mol, params=params, warn=False,
    scc_mixer="broyden",
    charge_mixing=0.2,
    mixer_memory=12,
    mixer_damping=0.1,
    max_iter=500,
)
print(f"energy={model.energy():.8f}  n_iter={model.n_iter}")
```

### Mixer selection guide

| Mixer | `scc_mixer=` | Best for |
|-------|-------------|----------|
| Simple damping | `"simple"` | The default: a two-phase polyalgorithm (damped simple mixing with stall-adaptive step halving, then a guarded DIIS handoff) |
| DIIS | `"diis"` | Explicit unguarded accelerator (bypasses the polyalgorithm) |
| Broyden | `"broyden"` | Explicit unguarded accelerator (bypasses the polyalgorithm) |

`SCCMixer.Newton` is reserved for the nontrivial GFN2-SECCM engine. The
direct molecular native boundary rejects it instead of substituting the
default molecular polyalgorithm.

### The AES faithfulness flag

`opts.aes_faithful` defaults to `True`: atom-resolved AES and on-site
anisotropic XC follow Bannwarth et al. (2019), Eqs. 25 and 39-44.
`False` explicitly selects the historical shell-resolved multipole model;
it is not a dipole-only speed approximation. `GFN2Model` and the runner's
geometry provider share the faithful default. The separate post-SCF D4
approximation is unchanged, so AES agreement alone is not full-energy parity.

---

## Element coverage

GFN2-xTB is parametrised for elements up to radon (Z = 86). The
post-SCF native D4 dispersion is parity-validated for H, He, B, C,
N, O, F, and Ne. For elements outside that set, D4 returns zero and
emits a `GFN2D4UnsupportedWarning`.

```python
from vibeqc.semiempirical.methods.gfn2 import GFN2D4UnsupportedWarning

# Molecule with chlorine (Z=17, outside native D4 set)
import warnings
ccl4 = Molecule([
    Atom(6, [0, 0, 0]),
    Atom(17, [1.5, 1.5, 1.5]),
    Atom(17, [-1.5, -1.5, 1.5]),
    Atom(17, [1.5, -1.5, -1.5]),
    Atom(17, [-1.5, 1.5, -1.5]),
])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model_cl = GFN2Model(ccl4, params=params, warn=False)
    e = model_cl.energy()
    d4_warnings = [x for x in w if issubclass(x.category, GFN2D4UnsupportedWarning)]
    if d4_warnings:
        print(f"D4 unsupported: {d4_warnings[0].message}")
    print(f"CCl4 GFN2-xTB: {e:.8f} Ha (D4=0 for Cl)")
```

---

## Summary -- what vibe-qc's GFN2-xTB can and cannot do

| Workshop topic | vibe-qc GFN2-xTB | Notes |
|---|---|---|
| Single-point | ✓ | Molecular + periodic Gamma-point |
| Open-shell occupations | Experimental partial model | low-level `run_ugfn2_xtb`; no quantitative acceptance |
| Geometry optimization | ✓ | Via `run_job` or ASE |
| Frequencies | ✓ | FD Hessian from analytic gradients |
| Molecular dynamics | ✓ | ASE NVE / NVT with the `GFN2Calculator` defined in section 4 |
| PES scans | ✓ | Programmatic bond/angle scans |
| Periodic | ✓ | 1D/2D/3D Gamma-point; general k-points fail closed |
| Solvation | ✗ | Not implemented |
| ONIOM | ✗ | Not implemented |
| Docking | ✗ | Not implemented |
| Thermo submodule | ✗ | Use ASE thermochemistry or `vibeqc.thermo` |
| GFN1-xTB | ✗ | Not implemented |
| GFN-FF | ✗ | Not implemented |

---

## See also

- {doc}`pm6_and_gfn2` -- the shorter introduction to PM6 and GFN2-xTB
- {doc}`semiempirical_dftb` -- the DFTB stack (DFTB0, SCC-DFTB)
- {doc}`semiempirical_periodic_and_validation` -- periodic semiempirical methods and cross-validation
- {doc}`../user_guide/semiempirical` -- the full user guide
- [`examples/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/semiempirical/) -- all runnable examples
- [Grimme lab xtb workshop](https://grimme-lab.github.io/workshops/page/xtb) -- the upstream workshop this tutorial mirrors
