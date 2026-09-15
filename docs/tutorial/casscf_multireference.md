---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for multireference quantum chemistry: CASCI, orbital-optimized CASSCF, and the CASPT2 and NEVPT2 dynamic-correlation corrections, run through run_job with an active_space. Real energies on H2, N2, and H2O validated against OpenMolcas, plus the theory of static versus dynamic correlation, the active space, CASSCF orbital optimization, and the CASPT2-vs-NEVPT2 intruder-state difference."
    "og:title": "vibe-qc tutorial - multireference: CASSCF, CASPT2, NEVPT2"
---

# Multireference: CASSCF, CASPT2, NEVPT2

Some molecules are not honestly described by a single electron
configuration. Stretch a bond toward dissociation, open a diradical,
expose a near-degenerate transition-metal d-shell, or ask for an
excited state, and the wavefunction becomes an unavoidable mixture of
several determinants of comparable weight. Hartree-Fock, built on one
determinant, gets these *qualitatively* wrong, and patching it with a
single-reference correlation method on top (MP2, CCSD) inherits the
error. The cure is a wavefunction that holds the competing
configurations on an equal footing from the start: the complete active
space (CAS) family. This page walks the standard ladder, RHF, then
CASCI, then orbital-optimized CASSCF, then the CASPT2 and NEVPT2
dynamic-correlation corrections, on real systems with real numbers, and
the Theory section explains what each rung buys you and why.

Every method here runs through the same `run_job` entry point you
already know, selected with `method=` and an `active_space=(n_orb,
n_elec)` tuple. The lower-level `vibeqc.solvers.{casci,casscf,nevpt2,
caspt2}` functions are available when you need finer control (state
averaging, custom references); the [non-mean-field solvers
tutorial](non_hf_solvers.md) covers that API and the active-space
contract in depth.

## Setup

The CAS methods need nothing beyond a molecule and a basis. A CAS
calculation is specified by two numbers: how many electrons and how
many spatial orbitals you place in the active space. vibe-qc takes
them as `active_space=(n_orb, n_elec)` and partitions the RHF orbitals
for you: the lowest `(n_total_elec - n_elec) / 2` orbitals become a
doubly-occupied frozen core, the next `n_orb` are active, and the rest
are dropped. The frozen core is *dressed exactly* (its mean field
enters the active one-electron integrals and a constant `E_core` is
folded into the energy offset), so the number `run_job` reports is a
real total energy you can put on a curve, not an active-space-only
fragment. The [active-space section of the solvers
tutorial](non_hf_solvers.md#active-spaces) spells out that dressing.

```python
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

# H2 at equilibrium (bond length 1.4 bohr)
h2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])

result = run_job(h2, basis="sto-3g", method="casscf", active_space=(2, 2))
print(f"E(CASSCF(2,2)) = {result.energy:.8f} Ha")
```

```
E(CASSCF(2,2)) = -1.13727594 Ha
```

For H2 in a minimal basis the full valence space *is* CAS(2,2), the
bonding $\sigma_g$ and antibonding $\sigma_u^*$ orbitals with both
electrons, so this CASSCF is the exact (full-CI) STO-3G answer. Every
result object exposes `.energy` (the total energy), `.method` (a label
recording the active space and reference), and `.converged`; the PT2
and state-averaged variants add `.root_energies` for the per-state
energies. We will read those off as we go.

When comparing dense and selected CI at the exact-selection limit, use
identical orbitals, the same active-space Hamiltonian and the same target
state. CASSCF also optimizes the orbitals: independent runs can reach
different stationary solutions with different energies or iteration histories.
Exact CI agreement at fixed orbitals does not establish a common
orbital-optimization minimum or the global CASSCF minimum. The regression
comparison corrected in [#56](https://github.com/vibe-qc/vibe-qc/issues)
uses this fixed-orbital boundary.

## The full ladder on N2

N2 at its equilibrium bond length is the cleanest illustration of the
ladder, because vibe-qc's CASSCF and CASPT2 energies for it are pinned
to an external OpenMolcas reference (recorded in
`tests/test_solvers_mrpt_parity.py`), so you can trust every digit. The
honest active space for the nitrogen triple bond is **CAS(6,6)**: the
six electrons of the $\sigma$ and two $\pi$ bonds in the six valence
orbitals $\sigma_{2p}$, $\pi_{2p}\times 2$, $\pi^*_{2p}\times 2$,
$\sigma^*_{2p}$. The 1s cores and the 2s-derived $\sigma/\sigma^*$ pair
are the inactive frozen core.

This script climbs the whole ladder. It runs RHF, CASCI (CI in the
frozen HF orbitals), CASSCF (the same CI but with the orbitals relaxed
to minimize the CAS energy), then NEVPT2 and CASPT2 on the
orbital-optimized CASSCF reference, which is the standard production
composition.

```python
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job
from vibeqc.solvers import CASSCFOptions

# N2 at equilibrium, R = 2.074 bohr (1.098 Angstrom)
n2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])

def energy(method, **kw):
    return run_job(n2, basis="sto-3g", method=method, **kw).energy

e_rhf    = energy("rhf")
e_casci  = energy("casci",  active_space=(6, 6))
e_casscf = energy("casscf", active_space=(6, 6))

# PT2 on the orbital-optimized CASSCF reference (pass casscf_options to
# switch the reference from CASCI-on-HF-orbitals to a converged CASSCF).
e_nevpt2 = energy("nevpt2", active_space=(6, 6), casscf_options=CASSCFOptions())
e_caspt2 = energy("caspt2", active_space=(6, 6), casscf_options=CASSCFOptions())

print(f"  RHF              {e_rhf:.8f}")
print(f"  CASCI(6,6)       {e_casci:.8f}   (static, frozen HF orbitals)")
print(f"  CASSCF(6,6)      {e_casscf:.8f}   (static, optimized orbitals)")
print(f"  NEVPT2 / CASSCF  {e_nevpt2:.8f}   (+ dynamic, intruder-free)")
print(f"  CASPT2 / CASSCF  {e_caspt2:.8f}   (+ dynamic)")
print()
print(f"  static  correlation (RHF -> CASSCF): {(e_rhf - e_casscf) * 627.509:7.2f} kcal/mol")
print(f"  dynamic correlation (CASSCF -> CASPT2): {(e_casscf - e_caspt2) * 627.509:7.2f} kcal/mol")
```

This is exactly what you get:

```
  RHF              -107.49584213
  CASCI(6,6)       -107.62174545   (static, frozen HF orbitals)
  CASSCF(6,6)      -107.63683517   (static, optimized orbitals)
  NEVPT2 / CASSCF  -107.64546059   (+ dynamic, intruder-free)
  CASPT2 / CASSCF  -107.64834004   (+ dynamic)

  static  correlation (RHF -> CASSCF):  88.46 kcal/mol
  dynamic correlation (CASSCF -> CASPT2):  7.22 kcal/mol
```

Read the ladder top to bottom. Each rung lowers the energy, and each
lowering has a distinct physical meaning:

- **RHF to CASCI** ($-107.4958 \to -107.6217$, about 79 kcal/mol):
  letting the six active electrons distribute over the bonding *and*
  antibonding valence orbitals, instead of forcing them all into the
  bonds. This is the bulk of the **static** correlation, the part a
  single determinant structurally cannot hold.
- **CASCI to CASSCF** ($-107.6217 \to -107.6368$, about 9 kcal/mol):
  relaxing the orbitals themselves. The HF orbitals were optimal for
  the single HF determinant, not for the multi-determinant CAS
  wavefunction. Re-optimizing them under the CAS density recovers the
  rest of the static correlation.
- **CASSCF to NEVPT2 / CASPT2** (about 5 to 7 kcal/mol): the **dynamic**
  correlation, the short-range, many-small-contributions avoidance from
  all the excitations *outside* the active space. This is what a
  perturbation theory on top of a good reference is for.

The two external checks: vibe-qc's CASSCF(6,6) is $-107.63683517$ Ha
against the OpenMolcas RASSCF value of $-107.63683522$ Ha (agreement to
0.05 microhartree), and the CASPT2-on-CASSCF total is $-107.64834004$
Ha against OpenMolcas $-107.64834018$ Ha (0.14 microhartree). Those
OpenMolcas numbers were generated out-of-process and recorded as
constants in the test suite (CLAUDE.md disallows importing another
quantum-chemistry program into vibe-qc), and the parity is what lets
this tutorial state real digits rather than hand-wave at a trend.

```{admonition} Two references, two PT2 totals
:class: note
By default `method="caspt2"` and `method="nevpt2"` build their CAS
reference as **CASCI on the HF orbitals** (matching OpenMolcas
`RASSCF ... CIonly`). Passing `casscf_options=CASSCFOptions()`
switches the reference to an **orbital-optimized CASSCF**, the standard
workflow and the one pinned to OpenMolcas above. The energies differ
because the reference differs: on this N2 case the CASCI-referenced
CASPT2 is $-107.64577302$ Ha, the CASSCF-referenced one
$-107.64834004$ Ha. Pick the CASSCF reference unless you have a
specific reason to perturb the frozen-orbital CASCI.
```

## Why a single determinant fails: H2 dissociation

The ladder above was run at equilibrium, where the static correlation
is real but moderate. To see *why* the multireference machinery is not
optional, pull a bond apart and watch RHF break. H2 in a 6-31G basis is
the smallest honest example: the full valence active space is again
CAS(2,2), so CASSCF(2,2) is the exact-in-basis curve, and RHF is the
single-determinant approximation to it. This script scans the bond from
equilibrium out toward dissociation and prints the gap.

```python
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

def energy(mol, method, **kw):
    return run_job(mol, basis="6-31g", method=method, **kw).energy

print(f" {'R/bohr':>7s} {'E(RHF)':>14s} {'E(CASSCF)':>14s} {'gap/kcal':>9s}")
for R in [1.4, 2.0, 3.0, 4.0, 5.0, 6.0]:
    h2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, R])])
    e_rhf = energy(h2, "rhf")
    e_cas = energy(h2, "casscf", active_space=(2, 2))
    print(f" {R:>7.2f} {e_rhf:>14.8f} {e_cas:>14.8f} {(e_rhf - e_cas) * 627.509:>9.1f}")
```

```
  R/bohr         E(RHF)      E(CASSCF)  gap/kcal
    1.40    -1.12674270    -1.14624789      12.2
    2.00    -1.08381072    -1.11498960      19.6
    3.00    -0.98167091    -1.04404698      39.1
    4.00    -0.90055091    -1.00937107      68.3
    5.00    -0.84318557    -0.99927739      97.9
    6.00    -0.80456891    -0.99705370     120.8
```

Two things to read off. First, the **CASSCF curve dissociates
correctly**: it flattens toward $\approx -0.997$ Ha, which is twice the
energy of a single hydrogen atom in this basis, the physically correct
limit of two neutral H atoms. Second, the **RHF curve does not**: it
keeps climbing, and at 6 bohr it sits 120.8 kcal/mol above the true
limit. The error is not a small quantitative miss, it is the wrong
shape of the curve, the **RHF dissociation catastrophe**.

The reason is structural. As the bond stretches, the exact ground state
becomes an equal mixture of the closed-shell determinant
($\sigma_g^2$) and the doubly-excited one ($\sigma_u^{*2}$). RHF is
locked into keeping $\sigma_g$ doubly occupied, which forces in
high-energy ionic terms (H$^+\cdots$H$^-$) that should vanish at
dissociation. No single determinant can represent the
correctly-mixed state, and so no amount of single-reference correlation
on top of RHF can fix it (MP2 diverges here; CCSD develops a spurious
bump). CAS(2,2) holds both determinants from the start, which is why
its curve is right at every distance. The gap between the two curves
*is* the static correlation, and the fact that it grows without bound as
you stretch the bond is the fingerprint that tells you a single-reference
method has no business being used.

At equilibrium ($R = 1.4$) the CASSCF(2,2) value $-1.14624789$ Ha
matches the OpenMolcas reference of $-1.14624788$ Ha in the test suite,
the same external pinning as the N2 case.

## A water example with CASPT2 parity

For a polyatomic that is not bond-breaking, H2O at equilibrium in
STO-3G with CAS(4,4) gives a compact ladder whose CASPT2 total is again
OpenMolcas-pinned. Here the static correlation is small (water is a
well-behaved closed shell), so the rungs are close together and most of
the recovered correlation is dynamic, the regime where a single
CASPT2/NEVPT2 step does the heavy lifting.

```python
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

h2o = Molecule([
    Atom(8, [0.0,  0.0,   0.0]),
    Atom(1, [0.0,  1.43, -0.93]),
    Atom(1, [0.0, -1.43, -0.93]),
])

def energy(method, **kw):
    return run_job(h2o, basis="sto-3g", method=method, **kw).energy

print(f"  RHF          {energy('rhf'):.8f}")
print(f"  CASCI(4,4)   {energy('casci',  active_space=(4, 4)):.8f}")
print(f"  CASSCF(4,4)  {energy('casscf', active_space=(4, 4)):.8f}")
print(f"  NEVPT2(4,4)  {energy('nevpt2', active_space=(4, 4)):.8f}")
print(f"  CASPT2(4,4)  {energy('caspt2', active_space=(4, 4)):.8f}")
```

```
  RHF          -74.94142025
  CASCI(4,4)   -74.94699163
  CASSCF(4,4)  -74.95274478
  NEVPT2(4,4)  -74.97146663
  CASPT2(4,4)  -74.97302709
```

The NEVPT2 and CASPT2 lines here use the **default CASCI-on-HF
reference** (no `casscf_options`), which is what OpenMolcas reproduces
with its `CIonly` keyword. vibe-qc's CASPT2 total $-74.97302709$ Ha
matches the recorded OpenMolcas `&CASPT2` value of $-74.97302712$ Ha to
0.03 microhartree. Note that the dynamic-correlation step (CASSCF to
CASPT2, about 13 kcal/mol) is much larger than the static step (RHF to
CASSCF, about 7 kcal/mol) here: at a closed-shell equilibrium geometry,
most of the missing correlation is dynamic, and the active space is
doing comparatively little. That is the opposite balance from
stretched N2, and recognizing which regime you are in is half the
skill of using these methods.

## Theory

The sections below build the picture from the ground: what an active
space is, the difference between static and dynamic correlation, why
one determinant is not enough, what CASSCF's orbital optimization adds
over CASCI, and how the two PT2 corrections recover the rest, including
the intruder-state difference that distinguishes CASPT2 from NEVPT2.

### Static versus dynamic correlation

Electron correlation, the energy a single Slater determinant misses,
comes in two physically distinct flavors, and the whole CAS workflow is
organized around treating them separately.

**Static** (or *strong*, or *non-dynamic*) correlation appears when two
or more electron configurations have comparable weight in the true
wavefunction. It is a near-degeneracy effect: bond-breaking (the
bonding and antibonding configurations cross), diradicals (two
near-degenerate singly-occupied orbitals), transition-metal complexes
(partially-filled d-shells), and many excited states. Static
correlation is large, it is *qualitative*, and no single determinant
can capture it, because the very thing that makes it static is that
several determinants matter at once.

**Dynamic** correlation is the short-range, instantaneous avoidance of
electrons, the Coulomb hole. It is the sum of an enormous number of
small contributions from excitations into the full virtual space. It is
present in *every* system, including closed-shell molecules at
equilibrium where there is no static correlation at all. Dynamic
correlation is what MP2 and coupled cluster recover on top of a single
good reference.

The division of labor in the CAS family follows this split exactly: a
CAS wavefunction handles static correlation rigorously (by including
all configurations within a chosen active space), and a perturbation
theory on top of it (CASPT2, NEVPT2) recovers the dynamic correlation
from outside the active space. The art is choosing an active space
large enough to capture the static correlation but small enough to be
affordable.

### The active space

A complete-active-space wavefunction partitions the molecular orbitals
into three sets:

- **Inactive** (core): doubly occupied in *every* determinant. These
  orbitals are frozen, they contribute a constant energy and a mean
  field, but no correlation.
- **Active**: the orbitals over which the wavefunction is allowed to
  correlate freely. The CAS ansatz includes *all* Slater determinants
  obtained by distributing the active electrons among the active
  orbitals in all symmetry-allowed ways, a full CI within the active
  space.
- **Virtual** (secondary): empty in every determinant.

A CAS($n$, $m$) calculation places $m$ electrons in $n$ active spatial
orbitals. The number of active determinants grows combinatorially, so
active spaces are kept small (typically up to CAS(14,14) or so for
conventional determinant CI; larger needs DMRG). Choosing the active
space is the one genuinely non-black-box decision in a multireference
calculation: it must contain every orbital involved in the
near-degeneracy you care about. For a breaking bond, that is the
bonding/antibonding pair; for the N2 triple bond, all three bonding
orbitals and their three antibonds; for a $\pi$ system, the full set of
$\pi$ and $\pi^*$ orbitals.

### CASCI: full CI in a chosen window

The cheapest CAS method is **CASCI**: take a fixed set of orbitals
(here, the converged RHF orbitals), partition them into
inactive/active/virtual, and diagonalize the Hamiltonian in the full
determinant space of the active electrons. The inactive orbitals are
frozen but *dressed*: they contribute an effective one-electron
potential to the active orbitals plus a constant $E_\text{core}$, so
the energy CASCI reports is a real total energy. CASCI is variational
($E_\text{CASCI} \le E_\text{HF}$ when the HF determinant lies inside
the active space, and $E_\text{CASCI} \ge E_\text{FCI}$ always), and it
captures the static correlation that lives in the active space. What it
does *not* do is improve the orbitals, it inherits whatever orbitals
you hand it.

### CASSCF: optimizing the orbitals too

The HF orbitals are optimal for the single HF determinant. They are
*not* optimal for a multi-determinant CAS wavefunction, the density has
changed, so the best orbitals have changed. **CASSCF** (complete active
space self-consistent field) closes that loop: it optimizes the CI
coefficients *and* the orbital shapes simultaneously, minimizing the
CAS energy with respect to both. Formally it iterates two coupled
steps, a CI solve in the current orbitals and an orbital rotation
$\mathbf{C} \to \mathbf{C}\,e^{\boldsymbol\kappa}$ driven by the orbital
gradient, until the gradient vanishes.

The payoff is that CASSCF gives the *best possible* CAS-quality
wavefunction for a given active space, and a reference whose orbitals
are self-consistent with the correlated density, which matters when you
then put perturbation theory on top. The cost is that the orbital
optimization is non-linear and can land in different stationary points
from different starting orbitals (the N2 and H2 cases here have unique
minima; water's CAS(4,4) is famously basin-rich, which is why the test
suite asserts only a variational bound for it rather than exact
OpenMolcas equality). In the ladders above, the CASCI-to-CASSCF drop is
precisely the static correlation that frozen HF orbitals leave on the
table.

### CASPT2 and NEVPT2: the dynamic-correlation correction

A CASSCF wavefunction is an excellent reference but still misses
dynamic correlation, the excitations into the virtual space it never
touches. Both **CASPT2** and **NEVPT2** recover that dynamic
correlation by second-order perturbation theory built on the CAS
reference, in direct analogy to how MP2 corrects a single HF
determinant. The first-order wavefunction spans the excitations out of
the active space (internal-to-virtual, internal-to-active, and so on),
and the second-order energy is the resulting correction. Both reduce to
MP2 exactly when the active space is empty, and both give zero
correction when the active space is the full space (nothing left
outside to correlate), limits the vibe-qc test suite checks explicitly.

The two methods differ in their choice of **zeroth-order Hamiltonian**
$\hat H_0$, the operator whose eigenvalues become the energy
denominators of the perturbation series, and that difference has a
practical consequence:

- **CASPT2** (Andersson, Malmqvist, Roos) uses a generalized Fock
  operator as $\hat H_0$. It is the established workhorse, but its $\hat
  H_0$ is not guaranteed to keep the perturber states above the
  reference in energy. When a perturber accidentally falls *near* the
  reference energy, its denominator goes to zero and the second-order
  energy diverges, the **intruder-state problem**. The standard
  remedies are a real or imaginary *level shift* that pushes the
  offending denominator away from zero; vibe-qc exposes an imaginary
  shift through `caspt2_options=CASPT2Options(imaginary=...)`.
- **NEVPT2** (Angeli, Cimiraglia, Malrieu) uses Dyall's $\hat H_0$,
  which retains the two-electron interaction *within* the active space.
  This is more expensive per term but **structurally intruder-free**:
  the perturber denominators are bounded away from zero by
  construction, so NEVPT2 needs no level shift and behaves smoothly even
  on difficult references. That robustness is its main selling point
  over CASPT2.

On well-behaved references the two agree closely, in the N2 ladder
above, NEVPT2/CASSCF and CASPT2/CASSCF differ by under 2 kcal/mol, a
typical spread. NEVPT2 is the safer default when you suspect intruder
trouble; CASPT2 remains a common, widely benchmarked choice for excited
states and spectroscopy. vibe-qc ships internally-contracted CASPT2 un-gated
and validated against OpenMolcas to within a few microhartree, including the
canonical eight-class single-state IPEA contraction. The strongly-contracted
variant remains gated behind `VIBEQC_EXPERIMENTAL_MULTIREF=1`.

### Choosing a method

A short decision guide, since the ladder offers several stopping
points:

- If you are at or near a **closed-shell equilibrium** and just want
  the correlation energy, you do not need a multireference method at
  all, MP2 or CCSD on the RHF reference is cheaper and entirely
  appropriate (see the [non-HF solvers tutorial](non_hf_solvers.md)).
- If more than one configuration matters, **bond-breaking, diradicals,
  transition-metal d-shells, excited states**, reach for the CAS
  family. Use **CASCI** for a quick look in fixed orbitals, **CASSCF**
  for a proper variational reference, and add **NEVPT2 or CASPT2** when
  you need the dynamic correlation on top.
- Between the two PT2s: prefer **NEVPT2** when intruder states are a
  risk (stretched geometries, dense excited-state manifolds) because it
  needs no shift; prefer **CASPT2** when you want to match the large
  body of CASPT2 benchmark literature, and reach for its imaginary
  shift if a denominator misbehaves.

## Nuclear gradients and geometry optimization

A converged CASSCF wavefunction has well-defined nuclear gradients, but
vibe-qc's current analytic implementation is incomplete. It is not tight
against a central finite difference of the full CASSCF energy, so do not use
`result.gradient` as a production force or correctness reference yet. The
geometry optimizer deliberately differentiates the full energy instead.

`run_job` attaches the gradient to the result whenever a CASSCF single
point converges:

```python
from vibeqc.solvers import CASSCFOptions

result = run_job(mol, basis="cc-pvdz", method="casscf", active_space=(6, 6))
forces = -result.gradient            # (n_atoms, 3), Hartree/bohr
```

The single-state gradient is the **complete analytic derivative** of the
CASSCF energy: a converged state-specific CASSCF is stationary in every
orbital rotation and CI coefficient, so no response ("z-vector") term
exists, and the analytic vector matches a central finite difference of the
energy to ~2e-7 Ha/bohr (H2/6-31G and H2/cc-pVDZ CAS(2,2)) and to 1.1e-6
on N2/cc-pVDZ CAS(6,6), where the residual is the finite difference's own
truncation error. A finite-difference cross-check is available as
`CASSCFOptions(compute_wz="numerical")` (expensive: the CASSCF is
re-converged at every displaced geometry).

`CASSCFOptions(compute_wz=True)`, which once selected an experimental
"W^z" CP-MCSCF correction, is now an accepted no-op alias of the analytic
gradient and emits a `FutureWarning`: that correction was a phantom term
for a variational energy and produced spurious gradient components on
linear molecules with p/d functions (GitLab #516). State-averaged runs
(`nroots > 1` with `weights`) expose the state-averaged analytic
approximation, which is checked for finiteness and translational
invariance but not against finite differences.

For geometry optimization, call `optimize_molecule`. It uses central finite
differences of the full CASSCF energy and returns the trajectory of frames:

```python
from vibeqc import Atom, Molecule
from vibeqc.molecular_optimize import optimize_molecule
from vibeqc.solvers import CASSCFOptions

h2 = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.5])])   # stretched, bohr
opt = optimize_molecule(
    h2, "6-31g", method="casscf", active_space=(2, 2),
    casscf_options=CASSCFOptions(), conv_tol_grad=1e-2, max_iter=20,
)
print(opt.energy, opt.trajectory_frames[-1])     # R = 1.5 relaxes toward ~1.4 bohr
```

`run_job(..., optimize=True)` uses the same full-energy finite-difference
path. The known-failing analytic-gradient reproducer lives at
`examples/regression/casscf_gradient_fd_reproducer.py` and will flip to PASS
when the single-point gradient is safe to use.

## Natural orbitals and QVF wavefunction export

A converged CASSCF/CASCI result has no ordinary canonical mean-field MOs to
export, but it does have **natural orbitals**, the active-space 1-RDM's
eigenvectors, with fractional occupation numbers that are exactly the
multireference character the calculation was run to find. `run_job` computes
these automatically and, with the default `output_qvf=True`, embeds them as
the QVF archive's wavefunction section, so vibe-view can resample any natural
orbital on demand just as it would a mean-field MO:

```python
result = run_job(
    mol, basis="cc-pvdz", method="casscf", active_space=(6, 6),
    output="n2_casscf",   # -> n2_casscf.out, n2_casscf.qvf
)
```

```sh
vibe-view open n2_casscf.qvf   # browse the natural orbitals + occupations
```

No extra flag is needed: `output_qvf` already defaults to `True`. This is
the same packaging path mean-field routes use for their MOs; CASSCF/CASCI
results are detected because they populate `natural_orbitals` /
`natural_occupations` on the returned `SolverResult` where a mean-field
result would populate `mo_energies` instead.

For a second worked example (H2/6-31G/CAS(2,2), the textbook
two-configuration case), see
[`examples/vibe_view/showcase_natural_orbitals.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/vibe_view/showcase_natural_orbitals.py),
which regenerates the fixture pinning vibe-view's HONO/LUNO path against
computed (not invented) data.

## References

The papers below define the methods used on this page. vibe-qc emits
the matching citations automatically in the references block of any run
that uses them (the routing lives in
`python/vibeqc/output/citations/database.toml`).

- **CASCI / CASSCF.** B. O. Roos, P. R. Taylor, and P. E. M. Siegbahn,
  "A complete active space SCF method (CASSCF) using a density matrix
  formulated super-CI approach," *Chem. Phys.* **48**, 157 (1980),
  doi:10.1016/0301-0104(80)80045-0. The original density-matrix
  super-CI CASSCF; vibe-qc cites it for both `method="casci"` and
  `method="casscf"` and as the CAS reference for the PT2 methods.
- **CASPT2.** K. Andersson, P.-Å. Malmqvist, and B. O. Roos,
  "Second-order perturbation theory with a complete active space
  self-consistent field reference function," *J. Chem. Phys.* **96**,
  1218 (1992), doi:10.1063/1.462209. The internally-contracted CASPT2
  that is vibe-qc's default `method="caspt2"`. The original
  single-state formulation is K. Andersson, P.-Å. Malmqvist, B. O.
  Roos, A. J. Sadlej, and K. Wolinski, *J. Phys. Chem.* **94**, 5483
  (1990), doi:10.1021/j100377a012 (vibe-qc's gated `variant="sc"`).
- **NEVPT2.** C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger,
  and J.-P. Malrieu, "Introduction of n-electron valence states for
  multireference perturbation theory," *J. Chem. Phys.* **114**, 10252
  (2001), doi:10.1063/1.1361246. Defines NEVPT2 with Dyall's
  intruder-free zeroth-order Hamiltonian; vibe-qc implements the
  strongly-contracted variant as `method="nevpt2"`.
- **IPEA shift (CASPT2).** G. Ghigo, B. O. Roos, and P.-Å. Malmqvist,
  "A modified definition of the zeroth-order Hamiltonian in
  multiconfigurational perturbation theory (CASPT2)," *Chem. Phys.
  Lett.* **396**, 142 (2004). The empirical level shift available via
  `caspt2_options=CASPT2Options(ipea=...)`. The canonical internally
  contracted treatment follows Y. Nishimoto, "Analytic First-Order
  Derivatives of CASPT2 with IPEA Shift," *J. Chem. Phys.* **158**,
  174112 (2023), doi:10.1063/5.0147611.
- **Textbook.** A. Szabo and N. S. Ostlund, *Modern Quantum
  Chemistry*, Dover (1996), §3.8 walks the H2 dissociation failure of
  RHF that motivates the whole multireference picture. For CASSCF and
  multireference perturbation theory specifically, T. Helgaker, P.
  Jorgensen, and J. Olsen, *Molecular Electronic-Structure Theory*,
  Wiley (2000), chapters 12 and 14, give the complete derivations.

## Next

Once the ladder makes sense, these are the natural follow-ups,
deeper API control, runnable scripts, the single-reference methods for
the easy regime, and convergence help when the orbital optimization
misbehaves.

- The [non-mean-field solvers tutorial](non_hf_solvers.md) covers the
  lower-level `vibeqc.solvers` API, the active-space contract in full,
  state-averaged CASSCF, and the other determinant-based solvers
  (Selected-CI, DMRG, v2RDM) you can use as CAS references.
- Runnable scripts for every method live in
  [`examples/wavefunction/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/wavefunction);
  `01_casscf_h2o.py` is the CASCI / CASSCF / PT2 / state-averaged
  walkthrough this page distills, and
  `n2_dissociation_active_space.py` scans the breaking N2 triple bond
  with RHF, CASCI(6,6), and CASSCF(6,6); `02_casscf_gradient.py` and
  `03_casscf_geometry_optimization.py` cover the analytic nuclear gradient
  and geometry optimization.
- For the single-reference correlation methods appropriate near
  equilibrium (MP2, CCSD), see the [molecular HF
  tutorial](molecular_hf.md) and the post-HF entries in the
  [API reference](../api/index.md).
- If your CASSCF will not converge or lands in the wrong basin, the
  [SCF convergence guide](../user_guide/scf_convergence.md) covers the
  orbital-optimization knobs.
