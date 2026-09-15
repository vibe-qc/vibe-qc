# Initial guesses: a hands-on walkthrough

A practical tour of the initial-guess methods you will reach for most
often (HCORE, SAD, SAP, AUTO). Copy each block into a Python interpreter
to follow along. By the end you'll have seen:

1. The three methods producing the **same converged energy** on a
   well-behaved system.
2. SAD **rescuing convergence** on a system where HCore lands on the
   wrong minimum (OH/6-31G\*).
3. SAD **rescuing a periodic SCF** that HCore drives into the
   bombing failure mode (NaCl/STO-3G).
4. **AUTO** picking the right method for each case automatically.

Background theory + math is in
[`docs/user_guide/initial_guess.md`](../user_guide/initial_guess.md);
this tutorial is the running tour.

## Why an SCF needs a guess

Hartree-Fock and Kohn-Sham SCF solve a chicken-and-egg problem. The
Fock matrix $\mathbf{F}$ that defines the orbitals is built from the
electron density $\mathbf{D}$, but $\mathbf{D}$ is built from those
same orbitals. Neither exists without the other, so the iteration has
to be *primed* with a starting guess and then refined until the input
and output densities agree (self-consistency).

The guess does not change where a well-behaved SCF ends up: every
method below reaches the same converged energy, to better than
$10^{-10}$ Ha. What it changes is the *path*, namely how many
iterations that takes, and occasionally *which* solution you land on
when a system has more than one stationary point (Step 2 is exactly
that trap).

## The ready methods at a glance

This tour walks the four you will reach for most: `HCORE`, `SAD`,
`SAP`, and `AUTO`. vibe-qc also ships `PATOM`, `HUECKEL`, `MINAO`, and
`READ`; see the note below.

| Guess | Builds | Idea in one line | Cost and character |
|-------|--------|------------------|--------------------|
| `HCORE` | a Fock matrix | Drop all electron-electron repulsion and start from $\mathbf{F} = \mathbf{T} + \mathbf{V}_{\rm ne}$ (kinetic plus nuclear attraction only). | Free, but the shell structure is wrong; it can drive ionic solids to nonsense energies before recovering. |
| `SAD` | a density | Stack pre-converged *atomic* densities, one block per atom, into a molecular density. | One tiny atomic SCF per element (cached); robust on every system class vibe-qc supports. |
| `SAP` | a Fock matrix | Replace the bare nuclei with *screened atomic potentials* fit to exact atomic SCFs, so the first orbitals already have the right shell structure, with no atomic SCF to run. | A handful of integrals per atom; the cheap shell-correct choice, with its biggest iteration savings on heavy atoms (Lehtola 2020). |
| `AUTO` | picks one | Inspect the system and choose: `SAP` for closed-shell light molecules, `SAD` for periodic, open-shell, or transition-metal systems. | The default in every SCF options struct. |

In one sentence: `HCORE` is the crude baseline, `SAD` is the robust
workhorse, `SAP` is the cheap shell-correct option for closed-shell
molecules, and `AUTO` chooses between them so you usually do not have
to. The full math for each method, the per-spin UHF / UKS packaging,
and the SAP citation requirement live in
[`initial_guess.md`](../user_guide/initial_guess.md).

### Is there a Hückel guess? (and PATOM, MINAO, READ)

Yes. As of the v0.9.x increment that landed them, `PATOM`, `HUECKEL`,
and `MINAO` are implemented molecular guesses you can select directly:

* `PATOM`: SAD followed by one in-field re-polarisation step, so the
  atomic densities relax in the molecular field before the SCF proper.
  Good for transition-metal d-shell ordering. Builds on SAD (Van
  Lenthe 2006); the ORCA PAtom idea. It is also wired on the closed-shell
  periodic RHF/RKS routes.
* `HUECKEL`: a parameter-free generalised Wolfsberg-Helmholz
  (extended-Hückel) guess over computed atomic-orbital energies
  (Wolfsberg-Helmholz 1952, Hoffmann 1963, Lehtola 2019). Both
  `initial_guess="huckel"` and `"hueckel"` are accepted. It is also wired as
  a lattice Fock-mode guess for closed-shell periodic RHF/RKS, including the
  closed-shell Python periodic density path.
* `MINAO`: free-atom minimal-basis (ANO-RCC) reference densities
  projected onto your basis (Knizia 2013). The closed-shell periodic RHF/RKS
  drivers use a lattice-summed projection variant, including the closed-shell
  Python periodic density path.

`PATOM`, along with `SAP`, `HUECKEL`, and `MINAO`, now also reaches the
open-shell Python periodic UHF/UKS drivers (Gamma-Ewald, multi-k Ewald,
GDF, and BIPOLE), not just closed-shell Gamma/multi-k RHF/RKS, since
v0.15.26 (2026-07-05). `SAD` remains the periodic `AUTO` default.
`READ` (restart from a prior calculation, via
`read_from=` or `opts.read_path`) is now implemented too, including supported
periodic restart routes; see
[`initial_guess.md`](../user_guide/initial_guess.md) for its API. Step 6
shows when overriding `AUTO` is worth it.

## Setup

Every step below shares these imports and the Angstrom-to-bohr factor,
so run this block once before working through the tour:

```python
import vibeqc as vq
from vibeqc import (
    Atom, Molecule, BasisSet,
    RHFOptions, UHFOptions, PeriodicRHFOptions,
    PeriodicSystem,
    run_rhf, run_uhf, run_rhf_periodic_gamma,
)
import numpy as np

A2B = 1.0 / 0.529177210903    # Angstrom → bohr conversion
```

## Step 1, Parity on H₂O (the well-behaved case)

When SCF behaves itself, every guess converges to the same energy.
The point of running this is to see that the choice is about
*efficiency*, not *correctness*.

```python
mol = Molecule([
    Atom(8, [0.0,  0.0,  0.0]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "6-31g*")

def run_with_guess(kind):
    opts = RHFOptions()
    opts.initial_guess = kind
    opts.conv_tol_energy = 1e-10
    return run_rhf(mol, basis, opts)

for kind in [vq.InitialGuess.HCORE,
             vq.InitialGuess.SAD,
             vq.InitialGuess.SAP,
             vq.InitialGuess.AUTO]:
    r = run_with_guess(kind)
    print(f"{kind.name:<6}  E = {r.energy:.10f}  n_iter = {r.n_iter}")
```

You'll see something like:

```
HCORE   E = -76.0107424867  n_iter = 12
SAD     E = -76.0107424867  n_iter = 10
SAP     E = -76.0107424867  n_iter = 11
AUTO    E = -76.0107424867  n_iter = 11
```

All four reach the same energy to better than $10^{-10}$ Ha.
Iteration counts vary by ±1-2, well within the noise that DIIS
introduces on a small system. AUTO resolved to SAP for this
closed-shell light-atom case.

**Why SAP didn't win here:** STO-3G / 6-31G\* are minimal basis sets
where SAP's shell-structure-correct initial orbitals are nearly
identical to what HCore-plus-DIIS produces after one extra
iteration. SAP's advantage grows with basis size and atomic number.

## Step 2, The OH false-minimum trap

This is the textbook example where the initial guess **changes which
minimum the SCF finds**. With HCore, OH/6-31G\* UHF converges to a
local minimum ~0.16 Ha above the true UHF minimum.

```python
mol = Molecule(
    [Atom(8, [0.0, 0.0, 0.0]),
     Atom(1, [0.97 * A2B, 0.0, 0.0])],   # 0.97 Å O-H bond
    multiplicity=2,                       # doublet radical
)
basis = BasisSet(mol, "6-31g*")

def run_uhf_with_guess(kind):
    opts = UHFOptions()
    opts.initial_guess = kind
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 300                   # HCore needs many iters
    return run_uhf(mol, basis, opts)

# The PySCF / textbook reference UHF energy is -75.3809309907.
TRUE_E = -75.3809309907

for kind in [vq.InitialGuess.HCORE, vq.InitialGuess.SAD,
             vq.InitialGuess.AUTO]:
    r = run_uhf_with_guess(kind)
    print(f"{kind.name:<6}  E = {r.energy:.6f}  "
          f"S² = {r.s_squared:.3f}  "
          f"Δ from true min = {r.energy - TRUE_E:+.6f}")
```

Typical output:

```
HCORE   E = -75.221182  S² = 0.764  Δ from true min = +0.159749   ← false minimum
SAD     E = -75.380931  S² = 0.755  Δ from true min = +0.000000   ← correct
AUTO    E = -75.380931  S² = 0.755  Δ from true min = +0.000000   ← AUTO → SAD
```

AUTO sees `is_open_shell=True` (because `n_alpha != n_beta`) and
picks SAD, sidestepping the trap. This is one reason the default
flipped from a hardcoded SAD to AUTO in v0.9, the engine now picks
the right method automatically without you having to remember.

## Step 3, NaCl, the periodic bombing case

This is the failure mode that drove the v0.5.6 → v0.9 default flip.
HCore on an ionic insulator can push the first-iteration energy into
$\pm 10^4$ Ha territory, and DIIS may never recover. SAD lives at
the unit-cell g=0 cell and gives a sane starting density.

```python
# NaCl rocksalt; STO-3G is too small for production but enough to
# expose the failure mode.
a = 10.7    # bohr (close to 5.64 Å)
lat = a * np.eye(3)
sysp = PeriodicSystem(
    3, lat,
    [Atom(11, [0.0, 0.0, 0.0]),         # Na
     Atom(17, [a/2, a/2, a/2])],        # Cl
    charge=0, multiplicity=1,
)
basis = BasisSet(sysp.unit_cell_molecule(), "sto-3g")

# AUTO → SAD for any periodic system. This is the default now.
opts = PeriodicRHFOptions()
opts.conv_tol_energy = 1e-8
opts.max_iter = 50
# opts.initial_guess defaults to InitialGuess.AUTO

result = run_rhf_periodic_gamma(sysp, basis, opts)
print(f"AUTO → SAD:  E/cell = {result.energy:.6f}  "
      f"n_iter = {result.n_iter}  converged = {result.converged}")

# Re-run with explicit HCORE to see the failure (or the slow recovery):
opts.initial_guess = vq.InitialGuess.HCORE
opts.max_iter = 100
try:
    result = run_rhf_periodic_gamma(sysp, basis, opts)
    print(f"HCORE:       E/cell = {result.energy:.6f}  "
          f"n_iter = {result.n_iter}  converged = {result.converged}")
    print("(HCore *may* recover on this small basis; on larger "
          "basis sets it bombs.)")
except RuntimeError as e:
    print(f"HCORE bombed: {e}")
```

The SAD/AUTO run converges in 5-15 iterations. The HCORE run either
recovers slowly or bombs, depending on basis size and damping. **On
real-research basis sets like pob-TZVP, HCore on NaCl never recovers
without level-shift intervention**, which is precisely why AUTO
defaults to SAD for periodic systems.

## Step 4, Inspecting AUTO's choice

The resolved kind is logged on every SCF run. To see it, look at
`result.scf_trace` (or just look at the live SCF log if you have
`progress=True`):

```python
mol = Molecule([
    Atom(6, [0.0, 0.0, 0.0]),                            # C
    Atom(8, [0.0, 0.0, 2.20]),                           # O - CO molecule
])
basis = BasisSet(mol, "cc-pvdz")
r = run_rhf(mol, basis)
# Look at the log output during the run; you'll see:
#   [scf]  initial guess: SAP (Lehtola/Visscher/Engel 2020, sap_helfem_large)
```

For programmatic inspection of what AUTO would pick for a given
system, you can call the engine directly:

```python
from vibeqc._vibeqc_core import _guess_closed_shell_density

# AUTO returns a density (for SAD-/PATOM-like resolutions) or None
# (for HCORE / Fock-mode kinds where the SCF core fills in). When
# AUTO resolves to SAP, the engine internally diagonalises F_SAP
# and returns the resulting density - non-None.
D = _guess_closed_shell_density(
    mol, basis, mol.n_electrons() // 2,
    vq.InitialGuess.AUTO, is_periodic=False,
)
print("AUTO yielded a starting density:", D is not None)
```

## Step 5, The transition-metal nuance

Closed-shell SAP can land the wrong d-shell occupation for a TM atom
because the underlying atomic potential is averaged over an open d
shell. AUTO detects TM atoms by Z range and resolves to SAD:

```python
# Iron pentacarbonyl Fe(CO)5 - closed-shell singlet but with iron Z=26.
# AUTO will route this to SAD even though it's closed-shell.
fe = [Atom(26, [0.0, 0.0, 0.0])]                # just iron, for illustration
mol = Molecule(fe, charge=2, multiplicity=1)    # Fe²⁺ d⁶ low-spin
basis = BasisSet(mol, "cc-pvdz")

opts = RHFOptions()
opts.initial_guess = vq.InitialGuess.AUTO

# In the log you'll see:
#   [scf]  initial guess: SAD (superposition of atomic densities)
# because has_transition_metal triggered.
```

You can select `PATOM` explicitly for these molecules when you want the
one-step re-polarised SAD start; AUTO remains conservative and keeps SAD as
the transition-metal default.

## Step 6, Picking a non-default deliberately

Most code shouldn't override AUTO. Cases where you might:

* **Unit tests / reference outputs**, pin `HCORE` for
  implementation-independent comparisons.
* **Benchmarking SAP vs SAD on heavy atoms**, pin `SAP` to measure
  its iteration-count win.
* **Diagnosing a hard convergence case**, switch back to `HCORE` to
  see if the issue is the guess or the SCF body.

```python
# Pinning HCORE for a deterministic regression test:
opts = RHFOptions()
opts.initial_guess = vq.InitialGuess.HCORE
opts.conv_tol_energy = 1e-12
r_ref = run_rhf(mol, basis, opts)
```

## Next

* **Theory and math** for each method: [`initial_guess.md`](../user_guide/initial_guess.md).
* **Tuning SCF convergence** (damping, DIIS, EDIIS+DIIS hybrid):
  [`scf_convergence.md`](../user_guide/scf_convergence.md).
* **`READ`** (restart a molecular SCF from a prior result, `.qvf`, or
  `.molden`) now ships, alongside `PATOM`, `HUECKEL`, and `MINAO`. The
  full initial-guess menu is in
  [`initial_guess.md`](../user_guide/initial_guess.md).

## See also

* Runnable example:
  [`examples/molecular/input-h2o-initial-guess-comparison.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-initial-guess-comparison.py)
  - a longer self-contained version of Steps 1-2.
