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

The [Theory](#theory) section below gives the equations of each guess in
brief, and the full treatment (per-spin packaging, datasets, periodic
variants) is in
[`docs/user_guide/initial_guess.md`](../user_guide/initial_guess.md); this
tutorial is the running tour. Numbers on this page are from a vibe-qc 0.17.10
development build on an Apple M-series laptop. The initial guesses are
production routes (label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)).

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
| `SAP` | a Fock matrix | Replace the bare nuclei with *screened atomic potentials* fit to exact atomic SCFs, so the first orbitals already have the right shell structure, with no atomic SCF to run. | A handful of integrals per atom; the cheap shell-correct choice, with its biggest iteration savings on heavy atoms (Lehtola 2019). |
| `AUTO` | picks one | Select the supported system and route policy: usually `PATOM` for closed-shell light molecules, `SAD` for ordinary periodic routes. See the exceptions below. | The default in every SCF options struct. |

In one sentence: `HCORE` is the crude baseline, `SAD` is the robust
workhorse, `SAP` is the cheap shell-correct option for closed-shell
molecules, and `AUTO` selects a supported construction, including `PATOM`,
for the requested system and route. The full math for each method,
the per-spin UHF / UKS packaging,
and the SAP citation requirement live in
[`initial_guess.md`](../user_guide/initial_guess.md).

Molecular open-shell d/f-block systems and isolated spin-polarized atoms
usually resolve to PATOM. Other molecular open-shell routes and restricted
closed-shell transition-metal systems usually resolve to SAD. Explicit
`atomic_spins` keeps AUTO on SAD where supported. Routes exposing only HCORE
resolve AUTO to HCORE; consult the [route table](../user_guide/initial_guess.md#route-coverage).

### A degenerate starting frontier

For molecular RHF `HCORE` and closed-shell molecular `PATOM`, a degenerate
block crossing the occupied frontier receives equal fractional occupations
when its total energy width is at most `1e-8` hartree. This preserves the
requested electron count and avoids selecting an arbitrary subset of the
frontier eigenvectors. `AUTO` inherits this behavior when it selects the
affected construction.

This is a starting-density rule, not electronic-temperature smearing or a
change to final SCF occupations. Gapped seeds, compatible `READ` states, and
periodic/open-shell seed policies retain their existing behavior. It does
not guarantee the global ground state or rotation invariance for every
calculation. See the [HCORE reference](../user_guide/initial_guess.md#hcore-core-hamiltonian-guess)
for the construction and its partial-spectrum guard.

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

## Theory

The guesses differ in what they build and how the first density arises. Write
$\mathbf F[\mathbf D] = \mathbf h + \mathbf G[\mathbf D]$ for the Fock matrix
of density $\mathbf D$, with $\mathbf h = \mathbf T + \mathbf V_\text{ne}$ the
core Hamiltonian ($\mathbf T$ kinetic, $\mathbf V_\text{ne}$ nuclear attraction)
and $\mathbf G$ the two-electron part.

**HCORE** drops $\mathbf G$ altogether. It diagonalises
$\mathbf F^{(0)} = \mathbf h$ in the generalised eigenproblem
$\mathbf F^{(0)}\mathbf C = \mathbf S\mathbf C\boldsymbol\varepsilon$ and occupies
the lowest $n_\text{occ}$ orbitals, $\mathbf D^{(0)} = 2\,\mathbf C_\text{occ}\mathbf C_\text{occ}^{\mathsf T}$.
Without electron screening every electron feels the full nuclear charge, so
the orbital ordering is hydrogenic and the shell structure of a heavy or ionic
system is wrong. The density is idempotent and has the right electron count,
but it is not close to the converged one.

**SAD** builds the starting density from atoms instead of orbitals
({vibeqc-cite}`vanlenthe_sad_2006`). Each unique element gets a
spherically averaged atomic SCF with fractional occupations $f_i$ per
angular-momentum channel, and the molecular density is the block-diagonal sum

$$
\mathbf D_\text{SAD} = \bigoplus_{A}\mathbf D_A,
\qquad
\mathbf D_A = \sum_i f_i\,\mathbf C^{(A)}_i\mathbf C^{(A)\mathsf T}_i .
$$

The result has the right electron count, $\operatorname{tr}(\mathbf D_\text{SAD}\mathbf S) = N$,
but it is not idempotent: the SCF starts from a Fock build on this density
rather than from a diagonalisation. The spherical average is the weakness for
open d shells, where the true atomic density is not spherical.

**SAP** ({vibeqc-cite}`lehtola_sap_2019`) works on the Fock side and needs no
atomic SCF. It replaces the bare nuclear attraction by a superposition of
screened atomic potentials, each a sum of error-function-screened Coulomb
terms fitted to a numerically exact atomic SCF,

$$
V_\text{SAP}(\mathbf r) = -\sum_A \sum_i c_i^{(A)}\,
  \frac{\operatorname{erf}\!\bigl(\sqrt{\alpha_i^{(A)}}\,|\mathbf r - \mathbf R_A|\bigr)}{|\mathbf r - \mathbf R_A|},
\qquad
\mathbf F_\text{SAP} = \mathbf T + \mathbf V_\text{SAP} ,
$$

with tabulated exponents $\alpha_i^{(A)}$ and coefficients $c_i^{(A)}$ and
the atom positions $\mathbf R_A$. The potential already contains the nuclear
attraction (it tends to $-Z_A/r$ at short range), so it *replaces*
$\mathbf V_\text{ne}$. One diagonalisation gives orbitals with the correct
shell structure and the right atomic screening at the cost of a few integrals
per atom.

**PATOM** starts from $\mathbf D_\text{SAD}$ and applies one in-field
mean-field re-polarisation step, so the superposed atomic densities relax in
the molecular field before the SCF proper. It is the density-mode upgrade of
SAD for cases where spherical atoms are too rigid, notably transition-metal
d-shell ordering, and it is what `AUTO` picks for closed-shell light-atom
molecules. **AUTO** is a policy, not a fifth construction: it chooses among
these according to the system class and the route.

**Where the guess matters.** All of them, given enough iterations, land on
the same solution for a well-behaved system, so the guess is mostly a matter
of iteration count. It becomes a matter of *which* solution when the energy
surface has more than one stationary point (open shells, stretched bonds,
broken-symmetry states): a poor guess can converge to a saddle or a higher
minimum, and a good one only makes that less likely. No guess certifies the
global minimum; check the energy across guesses and, if it matters, the
orbital stability.

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

Compare whether the guesses reach the same solution for this small
closed-shell example. Check both convergence and the final energy before
comparing iteration counts.

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

A run of the loop gave:

```
HCORE   E = -76.0066778869  n_iter = 12
SAD     E = -76.0066778869  n_iter = 10
SAP     E = -76.0066778869  n_iter = 11
AUTO    E = -76.0066778869  n_iter = 9
```

All four reached the same energy. `AUTO` resolves to PATOM for this
closed-shell light-atom case (`r.guess_selection.effective` says so) and
needs the fewest iterations. Compare energies and convergence from
your own run rather than expecting the iteration counts to match exactly.

The relative cost and convergence depend on the molecule, basis and SCF
settings. This small example does not rank the guesses for other systems.

## Step 2, The OH radical: the guess changes the path

OH is the textbook example where the initial guess can **change which
minimum the SCF finds**. Earlier builds of vibe-qc converged OH/6-31G\* UHF
from the core-Hamiltonian guess to a local minimum about 0.16 Ha above the
true UHF minimum. On the current build that trap does not reproduce: all
guesses reach the true minimum, and the guess only changes how long it takes.

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
          f"Δ from true min = {r.energy - TRUE_E:+.6f}  "
          f"n_iter = {r.n_iter}")
```

Output from the current build:

```
HCORE   E = -75.380931  S² = 0.755  Δ from true min = +0.000000  n_iter = 23
SAD     E = -75.380931  S² = 0.755  Δ from true min = +0.000000  n_iter = 12
AUTO    E = -75.380931  S² = 0.755  Δ from true min = +0.000000  n_iter = 12
```

AUTO sees `is_open_shell=True` (because `n_alpha != n_beta`) and picks SAD,
which halves the iteration count against HCORE here. The false minimum is a
reminder of what a bad start can do rather than a current failure: the
same test on a different molecule or basis can still reach a stationary
point other than the one you meant, which is why comparing final energies
across guesses is a cheap check on any open-shell system. This is one reason
the default flipped from a hardcoded SAD to AUTO in v0.9.

## Step 3, NaCl, the periodic ionic case

This is the system class that drove the v0.5.6 → v0.9 default flip. Earlier
builds sent the first-iteration energy of an ionic insulator into
$\pm 10^4$ Ha territory from the core-Hamiltonian guess, and DIIS might never
recover; SAD lives at the unit-cell g=0 cell and gives a sane starting
density. The run below shows how the two starts behave on the current build.

```python
# NaCl rocksalt; STO-3G is too small for production but enough to
# compare the two starts.
a = 10.7    # bohr (close to 5.64 Å)
lat = a * np.eye(3)
sysp = PeriodicSystem(
    3, lat,
    [Atom(11, [0.0, 0.0, 0.0]),         # Na
     Atom(17, [a/2, a/2, a/2])],        # Cl
    charge=0, multiplicity=1,
)
basis = BasisSet(sysp.unit_cell_molecule(), "sto-3g")

# AUTO resolves to SAD on this ordinary periodic RHF route.
opts = PeriodicRHFOptions()
opts.conv_tol_energy = 1e-8
opts.max_iter = 100
# opts.initial_guess defaults to InitialGuess.AUTO

result = run_rhf_periodic_gamma(sysp, basis, opts)
print(f"AUTO → SAD:  E/cell = {result.energy:.6f}  "
      f"n_iter = {result.n_iter}  converged = {result.converged}")

# Re-run with explicit HCORE:
opts.initial_guess = vq.InitialGuess.HCORE
opts.max_iter = 100
try:
    result = run_rhf_periodic_gamma(sysp, basis, opts)
    print(f"HCORE:       E/cell = {result.energy:.6f}  "
          f"n_iter = {result.n_iter}  converged = {result.converged}")
    print("Compare this result with the AUTO run above.")
except RuntimeError as e:
    print(f"HCORE bombed: {e}")
```

On a vibe-qc 0.17.10 development build (Gamma-only, one 2-atom cell, each
run taking a few minutes on a heavily loaded machine) both starts ended at
the same energy but neither reported convergence within its iteration cap:

```
AUTO → SAD:  E/cell = -1597.827513  n_iter = 100  converged = False
HCORE:       E/cell = -1597.827516  n_iter = 100  converged = False
```

The energies agree to 3 microhartree, so on this
build the core-Hamiltonian guess does not lead the SCF to a different state
here (the first-iteration energies were not recorded). The run also shows
that the convergence flag, the energy and the iteration count must be read
together: a
Gamma-only cell this small is a stiff test of the SCF convergence criteria,
independent of the guess, and it is a poor model of a converged crystal
calculation (use a k mesh and a larger basis for that, see the
[periodic tutorials](periodic_scf_convergence.md)). AUTO uses SAD on this
route as a conservative starting policy; other periodic routes can have
different capability limits.

## Step 4, Inspecting AUTO's choice

Inspect `result.guess_selection` to distinguish the requested selector from
the effective construction and density transport:

```python
mol = Molecule([
    Atom(6, [0.0, 0.0, 0.0]),                            # C
    Atom(8, [0.0, 0.0, 2.20]),                           # O - CO molecule
])
basis = BasisSet(mol, "cc-pvdz")
r = run_rhf(mol, basis)
# AUTO resolves to PATOM for this molecule.
print("requested:", r.guess_selection.requested)
print("effective:", r.guess_selection.effective)
print("transport:", r.guess_selection.transport)
```

A density being present does not by itself identify the chosen method. Use
the recorded selection rather than a private density-construction helper
when inspecting what the runner actually executed.

## Step 5, The transition-metal nuance

Closed-shell SAP can land the wrong d-shell occupation for a TM atom
because the underlying atomic potential is averaged over an open d
shell. For a restricted closed-shell transition-metal system, AUTO resolves
to SAD. For an open-shell d/f-block system it usually resolves to PATOM to
re-polarize the atomic seed in the molecular field:

```python
# A closed-shell singlet with iron (Z=26). AUTO routes it to SAD even
# though it is closed-shell.
fe = [Atom(26, [0.0, 0.0, 0.0])]                # just iron, for illustration
mol = Molecule(fe, charge=2, multiplicity=1)    # Fe²⁺ d⁶ low-spin
basis = BasisSet(mol, "cc-pvdz")

opts = RHFOptions()
opts.initial_guess = vq.InitialGuess.AUTO
opts.max_iter = 100

result = run_rhf(mol, basis, opts)
print(result.converged, result.n_iter, result.guess_selection.effective)
# True 22 InitialGuess.SAD
```

The run converged in 22 iterations to `E = -1261.474540` Ha, and the recorded
selection is SAD because the transition-metal rule applied.

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
