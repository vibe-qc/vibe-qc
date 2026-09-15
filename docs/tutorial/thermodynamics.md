# Thermodynamics at temperature

Once you have the electronic energy (SCF) and the vibrational
frequencies (Hessian), ASE's ``IdealGasThermo`` does the standard
statistical-mechanics arithmetic to produce **enthalpy**, **entropy**,
and **Gibbs free energy** at any (T, P). That's everything you need
for reaction thermochemistry in the ideal-gas approximation.

This tutorial chains off of the
[vibrational frequencies](vibrational_frequencies.md) tutorial, 
run that first to see how the Hessian is built.

## Chaining the pieces

This script runs the full chain on a water molecule: a `BFGSLineSearch`
relaxation to a stationary point, an ASE `Vibrations` finite-difference
Hessian, then `IdealGasThermo` to print H, S, and G at 298 and 373 K
under the `VibeQC` calculator at HF/6-31G\*.

```python
from ase import Atoms
from ase.optimize import BFGSLineSearch
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from vibeqc.ase import VibeQC

atoms = Atoms(
    symbols=["O", "H", "H"],
    positions=[[0.0, 0.0, 0.0],
               [0.0, 0.76, 0.59],
               [0.0, -0.76, 0.59]],
)
atoms.calc = VibeQC(basis="6-31g*")

# 1. Relax to a stationary point
BFGSLineSearch(atoms, logfile=None).run(fmax=0.01)
e_elec = atoms.get_potential_energy()               # eV

# 2. Hessian -> vibrational energies
vib = Vibrations(atoms, name="water_thermo")
vib.run()
vib_energies = vib.get_energies()                   # complex ndarray (eV)

# 3. Statistical mechanics
thermo = IdealGasThermo(
    vib_energies=vib_energies,
    potentialenergy=e_elec,
    atoms=atoms,
    geometry="nonlinear",      # "linear" / "nonlinear" / "monatomic"
    symmetrynumber=2,          # C2v for water
    spin=0,                    # 2S, not multiplicity
)

for T in (298.15, 373.15):
    H = thermo.get_enthalpy(temperature=T)
    S = thermo.get_entropy(temperature=T, pressure=101325.0)   # 1 atm
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325.0)
    print(f"T={T:6.2f} K   H={H:.4f} eV   "
          f"S={S * 1000:.2f} meV/K   G={G:.4f} eV")
```

Output (HF/6-31G\*, water):

```
T=298.15 K   H=-2067.5934 eV   S=1.95 meV/K   G=-2068.1750 eV
T=373.15 K   H=-2067.5674 eV   S=2.03 meV/K   G=-2068.3244 eV
```

## What `IdealGasThermo` accounts for

``IdealGasThermo`` prints a full breakdown when you call
``thermo.get_enthalpy(T, verbose=True)``. For water at 298 K:

```
E_pot              -2068.320 eV    electronic (SCF) energy
E_ZPE                  0.623 eV    zero-point vibrational energy
Cv_trans (0->T)        0.039 eV    3/2 kT
Cv_rot (0->T)          0.039 eV    3/2 kT (non-linear molecule)
Cv_vib (0->T)          0.000 eV    essentially all modes above kT
-------------------------------
U                  -2067.619 eV    internal energy
(C_v -> C_p)           0.026 eV    ideal-gas pV = kT
H                  -2067.593 eV    enthalpy
```

- ``E_ZPE`` = ½ Σᵢ ℏωᵢ, the vibrational ground state doesn't sit at
  the potential minimum, it sits ZPE above it. vibe-qc's HF/6-31G\*
  value for water is 60.1 kJ/mol; the experimental ZPE is 55.4
  kJ/mol.
- ``Cv_trans`` / ``Cv_rot`` are the classical equipartition
  contributions.
- ``Cv_vib`` is essentially zero for water because the lowest bend is
  at 2000 cm⁻¹ (far above ``kT ≈ 207 cm⁻¹`` at 298 K).

And entropy:

```
S_trans (1 bar)    0.00150 eV/K        0.448 eV·K⁻¹ at 298 K
S_rot              0.00045 eV/K        0.134 eV·K⁻¹
S_elec             0.00000 eV/K        (singlet → zero)
S_vib              0.00000 eV/K        (stiff modes → zero)
```

The translational entropy dominates, which is why small molecules
have large ``-TS`` and binding equilibria favor one big fragment
over two small ones.

## Reaction thermodynamics

To get ΔH, ΔS, ΔG for a reaction, do the thermo calculation on every
species and subtract. A toy example: dimerisation of water at 298 K.

```python
def thermo_for(xyz_path, symmetrynumber, geometry="nonlinear"):
    atoms = ...  # load / build
    atoms.calc = VibeQC(basis="6-31g*")
    BFGSLineSearch(atoms, logfile=None).run(fmax=0.01)
    e_elec = atoms.get_potential_energy()
    vib = Vibrations(atoms, name=f"vib_{xyz_path.stem}")
    vib.run()
    return IdealGasThermo(
        vib_energies=vib.get_energies(),
        potentialenergy=e_elec,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetrynumber,
        spin=0,
    )

monomer = thermo_for("h2o.xyz", symmetrynumber=2)
dimer   = thermo_for("h2o_dimer.xyz", symmetrynumber=1)  # Cs → σ=1

T = 298.15
dH = dimer.get_enthalpy(T) - 2 * monomer.get_enthalpy(T)
dG = dimer.get_gibbs_energy(T, 101325) - 2 * monomer.get_gibbs_energy(T, 101325)

print(f"ΔH(dim) = {dH * 96.485:.2f} kJ/mol")
print(f"ΔG(dim) = {dG * 96.485:.2f} kJ/mol")
```

You'll see ΔH is negative (H-bond is favourable) but ΔG is less
negative because dimerisation cuts the translational entropy in half.
At STP the Gibbs free energy of dimerisation is slightly positive, 
which is why water-dimer isn't the dominant species in the gas phase.

## Caveats

- **Ideal-gas approximation.** ``IdealGasThermo`` treats every mode as
  decoupled and every rotation as classical. For floppy molecules
  near symmetry-related conformers, or near-free internal rotations,
  use the ``HinderedThermo`` class instead.
- **Low-frequency modes are noisy.** The finite-difference Hessian
  has a floor around 50 cm⁻¹ at default SCF tolerance. If you have
  real modes in that region, tighten ``conv_tol_grad`` on the
  ``RHFOptions``.
- **No scaling factor applied.** The raw harmonic frequencies over-
  estimate fundamentals by ~10-15% at HF/6-31G\*; multiply by ~0.89
  (or 0.96 for DFT) if comparing to IR data.
- **Missing dispersion at HF.** Weakly-bound species (van der Waals,
  H-bonded dimers) under-bind at HF. Dispersion corrections (D3/D4)
  are on the wishlist.

## Theory

The sections below build the result from the ground up: the ideal-gas
partition function and its four factors, the thermodynamic functions
derived from it, the zero-point and enthalpy corrections, and the
regimes where the ideal-gas treatment breaks down.

### The ideal-gas partition function

Classical statistical mechanics gives every thermodynamic quantity
from the canonical partition function $Z(T, V, N)$. For an ideal
gas of non-interacting molecules, $Z$ factorises into translational,
rotational, vibrational, and electronic contributions:

$$
Z = Z_\text{trans} \, Z_\text{rot} \, Z_\text{vib} \, Z_\text{elec}.
$$

Each factor treats one degree of freedom as decoupled from the
others, which is exact for a truly non-interacting molecule and a
good approximation for gas-phase chemistry at room T and modest
pressure.

- **Translational:**
  $Z_\text{trans} = \bigl( \tfrac{2\pi M k_B T}{h^2} \bigr)^{3/2} V$.
  Dominates the entropy.
- **Rotational** (non-linear rigid rotor):
  $Z_\text{rot} = \tfrac{\sqrt{\pi}}{\sigma} \prod_{i} \sqrt{T / \theta_{\text{rot},i}}$
  where $\sigma$ is the symmetry number (2 for H₂O, 6 for NH₃, 12
  for CH₄) and $\theta_{\text{rot},i} = \hbar^2 / (2 I_i k_B)$ are
  the principal-axis rotational temperatures.
- **Vibrational** (harmonic): a product of independent
  one-dimensional oscillators:
  $Z_\text{vib} = \prod_k \frac{e^{-\hbar \omega_k / 2 k_B T}}{1 - e^{-\hbar \omega_k / k_B T}}$.
  The numerator is the zero-point contribution, the denominator is
  the thermal occupation.
- **Electronic:**
  $Z_\text{elec} \approx g_0$ for molecules with a large gap to the
  first excited state; typically $g_0 = 1$ for closed-shell singlets.

ASE's ``IdealGasThermo`` assembles all four, with mass and moment
of inertia computed from the atomic positions.

### The thermodynamic functions

Given $Z$, textbook relations give the functions of state:

$$
U = k_B T^2 \frac{\partial \ln Z}{\partial T},
\quad
H = U + pV = U + N k_B T \text{ (ideal gas)},
$$

$$
S = k_B \ln Z + k_B T \frac{\partial \ln Z}{\partial T},
\quad
G = H - TS.
$$

The decomposition $G = G_\text{elec} + G_\text{trans} + G_\text{rot} + G_\text{vib}$
(where $G_\text{elec} = E_\text{SCF} + E_\text{ZPE}$) is what ASE
prints. The translational and rotational contributions are what
push H-bonded complexes toward dissociation: losing three
translational degrees of freedom costs $T \Delta S \approx 0.4$ eV
at 298 K per dissociated fragment.

### Zero-point energy

The ZPE is the $T \to 0$ limit of the vibrational contribution:

$$
E_\text{ZPE} = \tfrac{1}{2} \sum_k \hbar \omega_k.
$$

ZPE is the vibrational ground-state energy above the classical PES
minimum. For 3-atom molecules it's typically 50-80 kJ/mol, bigger
than most interaction energies, so never forget to subtract two
monomer ZPEs from a dimer ZPE in binding-energy calculations.

### The enthalpy correction: $U \to H$ and $C_v \to C_p$

For an ideal gas, $H = U + nRT$ adds the $pV$ work of pushing the
gas aside. The molar heat capacities differ by the same constant:
$C_p - C_v = R$. At 298 K, $RT \approx 2.5$ kJ/mol, small relative
to ZPE and $T \Delta S$, but non-zero.

### Where ``IdealGasThermo`` fails

The assumptions break when:

1. **Low-frequency modes** ($\omega < \sim 100$ cm⁻¹), the harmonic
   approximation gives huge entropy contributions to floppy modes
   that aren't really harmonic. ``HinderedThermo`` in ASE handles
   hindered rotors; manual damping or mode-by-mode replacement of
   the partition function is the alternative.
2. **Condensed phase / solution**, translational / rotational
   partition functions need heavy modification. For aqueous
   chemistry you want an implicit-solvation method (roadmap item)
   and a careful treatment of the standard-state change.
3. **Transition metals with dense excited-state manifolds**, $Z_\text{elec}$
   stops being ~1 when there are excitations within $k_B T$.

## Resources

Negligible, partition-function arithmetic on the vibrational-frequency
output is sub-second on any laptop. The expensive step is the
underlying frequency calculation (see [Vibrational frequencies](vibrational_frequencies.md))
which dominates the total wall time.

## References

- **Ideal-gas partition-function textbook.** D. A. McQuarrie,
  *Statistical Mechanics*, University Science Books (2000).
  Chapters 6-8 are the clearest derivation of $Z_\text{trans}$,
  $Z_\text{rot}$, $Z_\text{vib}$ at the level needed for
  thermochemistry.
- **Physical-chemistry textbook.** P. Atkins and J. de Paula,
  *Atkins' Physical Chemistry*, 11th ed., Oxford University Press
  (2018). Chapter 13 covers the partition-function decomposition
  at an undergraduate level.
- **ASE.** A. H. Larsen et al., "The atomic simulation environment
  - a Python library for working with atoms," *J. Phys. Condens.
  Matter* **29**, 273002 (2017).
- **Frequency scaling for thermochemistry.** A. P. Scott and L.
  Radom, "Harmonic vibrational frequencies: an evaluation of
  Hartree-Fock, Møller-Plesset, quadratic configuration
  interaction, density functional theory, and semiempirical scale
  factors," *J. Phys. Chem.* **100**, 16502 (1996).
- **Standard thermochemical data (experimental).** NIST Chemistry
  WebBook, <https://webbook.nist.gov/chemistry/>. Experimental
  Δ*H*, Δ*S*, Δ*G* for comparison / benchmarking.

## Next

Transition-state search via NEB, reaction thermochemistry across a
series, and activation-energy calculations are all reachable with the
same ASE-driven pattern used here, write-ups will land as those
examples stabilise.
