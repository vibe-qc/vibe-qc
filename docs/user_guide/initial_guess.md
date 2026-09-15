# Initial guesses

Every SCF needs a starting point. The *initial guess* supplies the
first density matrix (or, equivalently, the first set of orbital
coefficients) that the SCF iteration refines. A good guess can cut
iteration counts in half and steer SCF away from spurious local
minima; a bad one can stall the iteration entirely or, in periodic
calculations, drive the energy off to nonsense values
(O(10⁴) Ha) before it ever recovers.

vibe-qc uses one **unified initial-guess framework**. The native
[`vibeqc::GuessEngine`](../api/index.md) owns molecular construction and the
system-aware AUTO policy. The shared `vibeqc.guess` adapter adds the lattice
context needed by periodic density and Fock builders. Public runners use the
same canonical selector and native AUTO resolver before dispatch, so the
method that executes, the output provenance, and the citations agree.
[`InitialGuess`](#initialguess-enum) selects the method; the
[`AUTO`](#auto-pick-the-right-guess-for-me) selector picks one for you based
on the system and route.

## Quick start

```python
import vibeqc as vq
from vibeqc import Atom, Molecule, BasisSet, RHFOptions, run_rhf

mol = Molecule([
    Atom(8, [0.0,  0.0,  0.0]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "6-31g*")

opts = RHFOptions()
opts.initial_guess = vq.InitialGuess.AUTO     # (default) picks PATOM for this case

result = run_rhf(mol, basis, opts)
```

The nine available kinds:

```python
vq.InitialGuess.AUTO       # engine picks per system and route
vq.InitialGuess.HCORE      # diagonalise the core Hamiltonian
vq.InitialGuess.SAD        # superposition of atomic densities
vq.InitialGuess.SAP        # superposition of atomic potentials (Lehtola 2020)
vq.InitialGuess.PATOM      # SAD + one in-field re-polarisation step
vq.InitialGuess.HUECKEL    # parameter-free generalised Wolfsberg-Helmholz
vq.InitialGuess.MINAO      # minimal-AO (ANO-RCC) reference projection
vq.InitialGuess.READ       # restart from a prior result / .qvf / Molden file
vq.InitialGuess.FRAGMO     # superposition of converged fragment densities
```

All nine are implemented today. `SAP` works through the public molecular
RHF/UHF/RKS/UKS API and the supported three-dimensional periodic backends;
the precise periodic route matrix is listed in the SAP section below.
`PATOM`, `HUECKEL`, and `MINAO` also have periodic implementations on the
route envelopes documented in their sections.
`READ` works for molecular and Gamma periodic routes, and supported closed-
and open-shell multi-k routes from in-memory results. Closed-shell multi-k
QVF archives also carry restart payloads (see
the **READ** section). Periodic `FRAGMO` accepts finite fragments with explicit
atom/image ownership and embeds their densities with Bloch translation phases.

## Route coverage

The table applies to SCF methods the route itself implements. A missing
whole method (for example ROKS/GDF or restricted-open GAPW) is rejected
independently of the guess. `A` means all six atomic constructions: HCORE,
SAD, SAP, PATOM, HUECKEL and MINAO. SAP requires three-dimensional periodicity
and a complete fitted or numerical atomic reference. AUTO is accepted on every listed route.

| Execution route | SCF families | Atomic guesses | READ | FRAGMO |
|---|---|---|---|---|
| Molecular native and Python | RHF, UHF, ROHF, RKS, UKS, ROKS | A | Result, QVF, Molden, prepared densities | Fragment sources or prepared densities |
| Native periodic Gamma / multi-k | RHF, RKS | A | Result, QVF, prepared per-k densities | Fragment sources through Python wrappers |
| Python Ewald Gamma / multi-k | All six | A | Gamma source or complete per-k payload | Via `run_periodic_job` |
| GDF Gamma / multi-k | RHF, UHF, ROHF, RKS, UKS | A | Gamma source or complete per-k payload | Via `run_periodic_job` |
| RIJCOSX Gamma / multi-k | RHF, UHF, RKS, UKS | A | Gamma source or complete per-k payload | Via `run_periodic_job` |
| BIPOLE Gamma / multi-k | All six; RO uses corrected Ewald | A | Gamma source or complete per-k payload | Via `run_periodic_job` |
| GPW Gamma | All six | A | Gamma source / density | Via `run_periodic_job` |
| GAPW Gamma | RHF, UHF, RKS, UKS | A | Gamma source / density | Via `run_periodic_job` |
| GPW true multi-k | Pure RKS, UKS, ROKS | A | Complete per-k payload | Via `run_periodic_job` |
| GAPW true multi-k | Pure RKS | A | Complete per-k payload | Via `run_periodic_job` |
| Native GPW host | RHF Gamma | A | Prepared density | Via `run_periodic_job` |
| Explicit / RI cyclic SCF and CCM correlation helpers | RHF, UHF, RKS, UKS | HCORE | Unsupported | Unsupported |
| AICCM real-Gamma / four-center | Adapter-supported families | HCORE | Unsupported | Unsupported |
| AICCM neutral-Bloch / chi | Producer-supported families | Underlying GDF / BIPOLE contract | Producer payload guards apply | Unsupported |

Standalone CCM SCF, MP2, CC, double-hybrid and direct gradient/optimization
helpers accept the same selector and retain `guess_selection` on their results.
Their existing cyclic loops construct HCORE; AUTO resolves to HCORE, while
explicit atomic or restart requests fail before integrals. These loops do not
expose an atomic-density or restart projection in their cyclic overlap metric.
The CCM GDF wrappers retain the underlying GDF capability set. A supplied SCF
reference retains its original provenance and cannot be relabeled by a new
selector. Direct gradient controls inherit the effective HCORE construction.
The correlation `reference="direct"` option chooses the reference Hamiltonian,
independently of the initial guess.

True multi-k PATOM needs full exact exchange even for a pure density
functional: its seed is one HF step, not one Hartree/XC step. The multi-k
GPW/GAPW drivers use the shared analytic Ewald Hartree and corrected full
exchange operators for this one seed step, with the target route's H(k).
Their subsequent pure-DFT iterations use the normal grid Hartree/XC operators.
This seed requires a supported full Monkhorst-Pack mesh; it does not enable
hybrid GPW/GAPW SCF. Native periodic READ accepts `read_from=result_or_qvf`
or target-ordered `options.read_density_k` (Gamma RHF: `read_density`).
Native result `density` remains a home-cell block; `mo_coeffs_k` and
`occupations_k` carry the complete Bloch restart orbitals. Real-lattice restarts require a complete,
time-reversal-compatible mesh. Ewald RHF/RKS/UHF/UKS retain the physical
per-k state independently of the lattice cutoff. Lattice-only consumers,
including restricted-open Ewald and BIPOLE, also require an exact inverse
Bloch round trip. Reduced or incompatible quadratures fail before SCF.

`CORE` aliases HCORE, `HUCKEL` aliases HUECKEL, and `MOREAD` / `COREAD` alias
READ. Case-insensitive enum names and strings use the canonical coercer.
ATOMSPIN is the `atomic_spins` input to SAD, not a separate enum or a fallback
selector. It requires an applicable open-shell driver, one tag in {-1,0,1}
per atom, and a physically compatible target population. AUTO may carry tags
only if its effective construction is SAD; READ transport cannot carry them.

## Effective core potentials

On routes that apply ECPs, SAD and its ATOMSPIN seed, PATOM, and HUECKEL
construct each atomic state with the actual atom's basis and selected ECP.
The one-center Hamiltonian contains kinetic energy, nuclear attraction with
`Z_eff = Z - n_core`, and the semilocal ECP operator. Chemical element identity
remains `Z`. Closed-core occupations are removed by angular momentum before
forming the atomic density. Molecular and periodic FRAGMO fragments preserve
the parent's actual AO shells and inherit their subset of its ECP operator.
Periodic images translate the owned AOs and ECP centers together.
The 28-electron core is `[Ar]3d10`, so neutral Ag
retains 19 electrons with s/p/d populations 3/6/10. Equal elements with different
operators or basis shells have separate atomic states. Population normalization
in the route's overlap metric follows this physical construction.

This follows Van Lenthe et al. (2006), Procedure p. 927, including its
one-center pseudopotential correction, and Andrae et al. (1990), Secs. 1-2,
for the valence Hamiltonian and 28/60-electron cores. Unknown closed-core
configurations and bases unable to represent their valence angular channels
fail explicitly. MINAO solves valence-only atomic HF in the ANO-RCC minimal
reference basis with the actual ECP before projection. ECP SAP constructs a
neutral atomic local potential from the spherical valence HF density:
`-Z_eff/r + J[rho_atom] + v_x_LDA[rho_atom]`, then adds the actual nonlocal
ECP matrix. Molecular and periodic adapters use the same numerical radial
reference; periodic potentials sum the owned atoms and their lattice images.
The radial charge integral is checked and normalized to atomic neutrality,
so numerical integration cannot introduce a residual Coulomb tail. This is
a numerical LDA-on-HF reference, following the construction described by
[Lehtola, Visscher and Engel (2020)](https://arxiv.org/abs/2002.02587), rather
than reusing the all-electron fits for a reduced nuclear charge. It is more
expensive than the fitted all-electron SAP path and depends on the working
atomic basis and the periodic image cutoff. HCORE and compatible READ remain available.

Periodic ECP construction is available in the 3D GDF RHF/UHF/RKS/UKS routes
and the native restricted Gamma/multi-k drivers. Other routes retain their
whole-method ECP restrictions; selecting HCORE does not bypass those limits.
All-electron POB metadata may carry the physical nuclear charges without an
ECP operator. Public preflight and direct Ewald, BIPOLE, multi-k GDF and native
periodic entry points recognize this charge-only record against
the actual atoms. Reduced charges, missing operators, and inconsistent core
metadata still fail validation before SCF. The all-electron Gamma GDF/RIJCOSX,
ROHF/GDF and private slab GDF guards use the same identity rule and reject
unsupported ECP metadata before setup.
AUTO remains the documented system/route resolver. There is no literature-backed
metal-versus-oxide classifier encoded in this release.

## What an initial guess actually does

The SCF iteration solves

$$
\mathbf{F}(\mathbf{D})\,\mathbf{C} = \mathbf{S}\,\mathbf{C}\,\boldsymbol{\varepsilon}, \qquad
\mathbf{D} = 2\,\mathbf{C}_{\rm occ}\,\mathbf{C}_{\rm occ}^{\!T},
$$

a fixed-point problem where the Fock matrix
$\mathbf{F}$ depends on the density $\mathbf{D}$, which is built from
the occupied orbitals $\mathbf{C}_{\rm occ}$, which come from
diagonalising $\mathbf{F}$. To start the loop you need *either* an
initial $\mathbf{D}$ (a density-mode guess), *or* an initial
$\mathbf{F}_{\rm guess}$ that the driver can diagonalise once to get
$\mathbf{C}$ and then $\mathbf{D}$ (a Fock-mode guess), *or* an
initial $\mathbf{C}$ directly (an MO-mode guess from a prior
calculation).

The literature classifies guess methods by which of those three
artifacts they produce:

| Construction mode | Methods | Input to the first SCF cycle |
|-------------------|---------|------------------------------|
| Density | `SAD`, `PATOM`, `MINAO` | A constructed `D` |
| One-particle Hamiltonian | `HCORE`, `SAP`, `HUECKEL` | The driver diagonalises `F_guess`, then builds `D` with its normal occupations |

`READ` and `FRAGMO` sit outside this table: both are resolved in the
`run_*` wrapper rather than the engine, injecting a density as
`opts.read_density` (closed shell) or `opts.read_density_alpha` /
`opts.read_density_beta` (open shell). `READ` injects a *prior* density;
`FRAGMO` injects a *freshly assembled* block-diagonal superposition of
converged fragment densities. The **READ** and **FRAGMO** sections below
cover them.

For molecular calculations the `GuessEngine` returns the constructed density
after any required one-particle diagonalisation. Closed-shell drivers see
[`GuessClosedShellResult`](#guess-result-types); open-shell drivers see
`GuessOpenShellResult`, which carries per-spin matrices. The periodic adapter
likewise returns a Gamma density for density-seed routes, while true multi-k
drivers receive `F_guess(k)` for SAP and HUECKEL and apply their established
global Aufbau, fixed-per-k, or smearing occupation policy.

Different guesses can converge to different stationary points. For a system
with a unique accessible minimum they should agree to the SCF tolerance;
iteration counts and intermediate energies can differ. Open-shell and
fractionally occupied states can have distinct physical basins, so a basin
comparison must also inspect occupations and spin densities. The OH/6-31G*
UHF false-minimum trap is one example: HCORE can reach a higher stationary
point than SAD.

### Density normalization and spin seeds

A density-mode guess is generally non-idempotent. Its population is measured
in the same AO overlap metric used by the calculation, not by the ordinary
matrix trace. Molecular and Gamma starts satisfy `Tr(D S) = N`. A common
home-cell density used at every k-point satisfies

$$
S_{\mathrm{eff}} = \sum_k w_k S(k), \qquad
\mathrm{Tr}(D S_{\mathrm{eff}}) = N, \qquad \sum_k w_k = 1.
$$

The shared constructor Hermitian-symmetrizes the atomic superposition and
rescales its total density in this metric when the count differs by more than
1e-12 electrons. Already-normalized seeds retain their bits, and identical
singlet channels remain exactly equal; roundoff rescaling must not perturb a
symmetry-sensitive initial state. Open-shell starts additionally
satisfy `Tr(D_alpha S_eff) = N_alpha` and
`Tr(D_beta S_eff) = N_beta`. Molecular overlap normalization does not replace
periodic normalization: lattice-image overlap can change the count even at
Gamma. Complex periodic matrices retain their phases.

For an unseeded open shell, spin normalization preserves the total density
and reduces the Hund spin difference by convex alpha/beta mixing when
possible. If the requested charge or spin needs a larger population transfer,
it transfers a positive fraction of one channel to the other. For ATOMSPIN,
the starting atomic blocks and their requested spin signs are explicit
constraints: only the overrepresented sign group is attenuated, and tag-zero
atoms remain unpolarized. An incompatible multiplicity or charge is rejected
rather than reversing a requested sign or dropping the seed.

ROHF/GDF accepts the density constructors SAD and MINAO and the Fock
constructors SAP and HUECKEL at Gamma and multi-k. PATOM performs one
spin-resolved in-field HF step before ROHF imposes common orbitals. Density
and occupied-factor exchange representations are replaced together. A
one-cycle density-mode result has no orbital payload until an actual ROHF
orbital update has occurred; its seed density is not mislabeled with HCORE
orbitals. READ replaces both spin densities and exchange factors; periodic FRAGMO enters through the same prepared-density transport;
ROKS/GDF requires an XC driver that this adapter does not provide.

## Multiplicity and the initial guess

Multiplicity and the initial guess do different jobs. The multiplicity
fixes *how many* electrons of each spin the SCF must converge to; the
guess supplies a starting density that respects those counts but does
not, on its own, decide *where* the spin sits.

**Multiplicity sets the spin counts.** From the multiplicity $2S+1$
and the total electron count $n_e$ (set by the nuclear charges and the
molecular or cell charge), each driver derives the alpha and beta
occupations with the usual convention

$$
n_\alpha = \frac{n_e + (\text{mult}-1)}{2}, \qquad
n_\beta  = \frac{n_e - (\text{mult}-1)}{2},
$$

so $n_\alpha - n_\beta = \text{mult}-1 = 2S$ is the number of net
unpaired electrons. That is the only spin quantity you set directly.
vibe-qc's open-shell SCF is *unrestricted* (UHF / UKS) with these two
counts held fixed; $\langle S^2\rangle$ is not constrained, so the
converged value is whatever the unrestricted solution yields (hence
the spin-contamination check in the [worked examples](#worked-examples)).

**The guess builds to those counts; its spin symmetry follows a
published, deliberate policy.** Given $n_\alpha$ and $n_\beta$, the
open-shell guess produces a starting point with the right counts and
a spin symmetry decided by the *electron counts and the requested
seed*, not by atomic Hund rules alone:

* $n_\alpha = n_\beta$ (spin-balanced): **spin-averaged atomic
  occupations** -- each atom contributes $\tfrac12\,f_i$ to both spins
  from the spherically-averaged atomic SCF, so $\mathbf{D}_\alpha =
  \mathbf{D}_\beta$ exactly. This is the deterministic symmetric start
  (the Van Lenthe 2006 standard: the UHF start comes from the
  closed-shell-type density), and it keeps a genuine closed-shell
  singlet bit-symmetric through the whole SCF.
* $n_\alpha \ne n_\beta$: **Hund-split atomic densities** -- each atom
  contributes its natural majority-alpha configuration
  ($f_i^\alpha$, $f_i^\beta$ from `spin_resolve_occupations`), so the
  starting spin density is per-atom polarised. This is what lands
  FeCl$_3$-class open shells in the right basin (BUG 88): a spherical
  spin density there converges to an excited solution.
* The UKS packaging diagonalises $\mathbf{F}_{\rm SAD}$ once from the
  total density and keeps the first $n_\alpha$ and $n_\beta$ columns.

The physical spin density develops *during* SCF. For a single radical
(one unpaired electron) there is only one way to place the spin and
the SCF finds it. For a system with several competing spin solutions
the default symmetric start biases toward the most symmetric one; the
**default-on internal stability analysis** (Seeger-Pople 1977;
Lehtola 2020 section 10) then detects a saddle and attempts to descend
toward a broken-symmetry minimum -- verified on H2 past the Coulson-Fischer
point, ozone, twisted ethylene, p-benzyne, and Cr2.
The escape searches both signs of the unstable orbital rotation and
rechecks the final Hessian before accepting the result. The first four
systems reach identical solutions before and after the spin-averaging fix
(`handovers/HANDOVER_GUESS_SPIN_SYMMETRY.md`). A *deliberate*
antiferromagnetic / ferrimagnetic pattern is requested explicitly via
`atomic_spins` (below), never imposed by the default guess.

**Seeding a broken-symmetry spin pattern (the CRYSTAL `ATOMSPIN`
analogue).** Molecular UHF and UKS accept `atomic_spins`, a per-atom
list of `+1` / `-1` / `0` in atom order (`+1` = majority alpha, `-1` =
majority beta, `0` = unpolarised). It seeds a broken-symmetry SAD start
in which each tagged atom contributes a Hund's-rule spin-resolved atomic
density, assembled block-diagonally into $\mathbf{D}_\alpha$ /
$\mathbf{D}_\beta$:

```python
opts = UHFOptions()
opts.atomic_spins = [+1, -1]   # atom 0 spin-up, atom 1 spin-down (antiparallel)
```

This is what lets you initialise an antiferromagnetic or ferrimagnetic
state that the spin-symmetric SAD split cannot reach (think alternating
metal-site spins in an oxide). The SCF then refines the pattern while
the total $n_\alpha - n_\beta$ stays fixed by the multiplicity.
`atomic_spins` is consulted only when the guess resolves to SAD
(including `AUTO` on an open shell); pairing it with another guess, or
giving a wrong-length list, raises.

`atomic_spins` works for **periodic** UHF/UKS too: the per-atom tags seed
the g=0 cell block (in unit-cell atom order), which Bloch-sums to a
broken-symmetry $\mathbf{D}(\mathbf{k})$, the route to an AFM/ferrimagnetic
periodic start. It is wired on the UHF/UKS Ewald, GDF and BIPOLE drivers
at Gamma and multi-k, plus the supported GPW/GAPW open-shell routes. Pass it
through `vq.run_periodic_job(..., method="UHF"/"UKS", atomic_spins=[...])`,
or set `opts.atomic_spins` on a driver directly.

**Holding the pattern through early iterations (`SPINLOCK`).** A
broken-symmetry start can still wash out to the symmetric solution in
the first few iterations on a hard system. `SPINLOCK` (the CRYSTAL
analogue) holds it, in either of two modes on molecular UHF / UKS, set
through `opts.spinlock_mode`, `opts.spinlock_iterations`, and (for the
schedule) `opts.spinlock_value`:

* `SpinlockMode.PATTERN_HOLD` keeps the seeded broken-symmetry occupied
  set in place by maximum-overlap selection (MOM, Gilbert/Besley/Gill
  2008) instead of plain Aufbau for the first `spinlock_iterations`
  cycles, then releases. Pair it with `atomic_spins` to stop an
  antiferromagnetic seed from collapsing early.
* `SpinlockMode.SPIN_SCHEDULE` runs a two-phase SCF: it converges at a
  locked $n_\alpha - n_\beta = $ `spinlock_value` for the first
  `spinlock_iterations` cycles, then restarts from that density at the
  multiplicity target. This is CRYSTAL's `SPINLOCK n nstep`; use it to
  drive the SCF through a high-spin intermediate into the state you want.

Direct native `run_uhf_scf_with_jk` and `run_uks_scf_with_jk` do not implement
the two population phases and reject active SPIN_SCHEDULE. Use the molecular
`run_uhf` / `run_uks` wrappers for that schedule.

```python
opts = UHFOptions()
opts.atomic_spins = [+1, -1]                      # seed the broken-symmetry start
opts.spinlock_mode = vq.SpinlockMode.PATTERN_HOLD
opts.spinlock_iterations = 8                      # hold 8 cycles, then release
```

**Periodic SPINLOCK.** Both modes also work for periodic open-shell SCF:
`PATTERN_HOLD` on the Γ UHF/UKS Ewald, GDF and BIPOLE drivers and the multi-k
UKS Ewald driver (per-k per-spin MOM hold, the path that protects an
`atomic_spins` AFM seed against collapse on multi-k); `SPIN_SCHEDULE` on the
Γ UHF/UKS Ewald, Γ GDF and Γ BIPOLE drivers (a two-phase SCF that locks a high-spin
state, then restarts at the target multiplicity from that density via the READ
machinery). Both are reachable from the high-level runner:
`vq.run_periodic_job(..., method="UHF"/"UKS", spinlock="pattern_hold"` /
`"spin_schedule", spinlock_iterations=N[, spinlock_value=M])`. Periodic
SPINLOCK on a driver that does not implement the requested mode (the multi-k
UHF Ewald path for any mode, or `SPIN_SCHEDULE` on the multi-k UKS Ewald and
BIPOLE paths)
fails closed with a clear error rather than silently ignoring the request.

On every driver that implements it (molecular UHF/UKS, Γ UHF/UKS Ewald, GDF,
BIPOLE, and multi-k UKS Ewald), `PATTERN_HOLD` also suspends the SCF
accelerator (plain DIIS as well as the EDIIS / ADIIS / KDIIS variants) for
the hold window (density damping stays live) and starts the extrapolation
history fresh at release. MOM only selects *which* orbitals are occupied; it
cannot stop Fock extrapolation from rotating the occupied orbitals themselves
toward the spin-symmetric solution, so letting the accelerator extrapolate
across held iterations would quietly collapse the seed inside the hold
window: exactly the failure the hold exists to prevent.

Still pending: `BETALOCK` (locking only the beta count), `SPIN_SCHEDULE` on
the multi-k routes, and per-element starting occupations; those are on the
[roadmap](../roadmap.md).

Two further occupation-related controls ship, though neither seeds a
spin pattern the way `atomic_spins` does:

| Control | Where | What it does | What it is *not* |
|---------|-------|--------------|------------------|
| `use_mom=True` (maximum overlap method) | periodic multi-k RHF / RKS | Holds the occupied set fixed by tracking maximum orbital overlap across iterations instead of refilling by energy (Gilbert, Besley and Gill 2008), so a chosen non-Aufbau occupation survives the SCF. On a k-mesh the zero-temperature fill then occupies exactly the tracked block at every k point instead of re-selecting the lowest states across the mesh; MOM is a zero-temperature contract and is refused together with finite smearing. | A way to *specify* a per-site spin pattern from scratch; it follows the orbitals you start from. |
| `smearing_temperature = T` (Fermi-Dirac) | periodic RHF / RKS | Replaces hard Aufbau with fractional occupations near $E_F$, easing convergence of metals and near-degenerate states. | A lock; it smooths occupations rather than fixing a pattern. |

The underlying methods are MOM (Gilbert, Besley, Gill 2008) and the
Mermin finite-temperature functional (Mermin 1965); both are in the
[references](#references).

## The methods, with math

### HCORE, core-Hamiltonian guess

The cheapest possible guess. Set the initial Fock to the
**one-electron core Hamiltonian** and ignore electron-electron
repulsion entirely:

$$
\mathbf{F}_{\rm HCORE} \;=\; \mathbf{T} + \mathbf{V}_{\rm ne}
$$

where $\mathbf{T}$ is the kinetic-energy matrix and
$\mathbf{V}_{\rm ne}$ is the nuclear-attraction matrix. Diagonalise
once, build $\mathbf{D}$ from the occupied orbitals, hand off to
the SCF loop.

**Cost:** trivial, no extra integrals beyond those the driver
needs anyway.

**Pathology:** the shell structure is wrong. Without screening from
the inner electrons, all the orbitals localise on the heaviest atom
and the outer-valence orbitals have the wrong sign of energy. On
molecular systems with light atoms this is a mild slowdown
(2-3 extra iterations); on ionic insulators with deep cores the
first Fock build from $\mathbf{D}_{\rm HCORE}$ can push the energy
into the $\pm 10^4$ Ha range, which DIIS then has to climb back out
of, the **NaCl-bombing** failure mode.

Use HCORE explicitly when you need a deterministic, system-
independent reference (e.g. for unit tests that compare
implementations).

### SAD, superposition of atomic densities

Build the molecular density as a block-diagonal sum of converged
atomic densities. For each unique element $A$ in the molecule, run a
small **spherically-averaged atomic SCF** with fractional occupations:

$$
\mathbf{D}_{A,\rm atom} \;=\; \sum_{i} f_i \, \mathbf{C}_i^{(A)} \mathbf{C}_i^{(A)\,T}
$$

where $f_i \in [0, 2]$ are the orbital occupations. Because the
spherically-averaged atomic Fock is block-diagonal in the angular
momentum $\ell$, every atomic MO is a pure-$\ell$ function, and the
occupations are filled **per $\ell$-channel** up to the element's
ground-state electron count for that channel (Aufbau *within* each
channel, with degenerate orbitals sharing equal fractional occupation so
the result stays spherical). The per-$\ell$ counts come from the
neutral-atom ground-state configuration, the Madelung order plus the
standard anomalies (Cr, Cu, Pd, the lanthanide/actinide irregulars, …),
the same data PySCF, Psi4, and ORCA pin for their atomic guesses. This
matters for the transition metals: a single global eigenvalue sort over
*all* orbitals mis-occupies the 3d row, because the 3d/4s/4p
near-degeneracy makes the bare-atom orbital order disagree with the
physical configuration (it would seat Ni at 3d¹⁰ 4s⁰ rather than the
physical 3d⁸ 4s²). Pinning per channel reproduces the experimental
d-shell filling for all of Sc-Zn. Then assemble the molecular density by
placing each atomic block on the diagonal:

$$
\mathbf{D}_{\rm SAD} \;=\; \bigoplus_{A \in \text{atoms}} \mathbf{D}_{A,\rm atom}.
$$

For periodic systems the engine places $\mathbf{D}_{\rm SAD}$ at
the $g=0$ cell and leaves other cells zero; the SCF builds the
rest by Bloch summation on the first iteration.

**Per-spin packaging.** Open-shell drivers see two flavours:

* **UHF (proportional split):**
  $\mathbf{D}_{\alpha} = (n_\alpha / n_e)\,\mathbf{D}_{\rm SAD}$,
  $\mathbf{D}_{\beta} = (n_\beta / n_e)\,\mathbf{D}_{\rm SAD}$.
  Cheap; spin polarisation develops on the first iteration via
  $\mathbf{F}_\alpha \ne \mathbf{F}_\beta$ when $n_\alpha \ne n_\beta$.
* **UKS (Fock-diagonalise split):** build the closed-shell-style
  $\mathbf{F}_{\rm SAD} = \mathbf{H}_{\rm core} + \mathbf{J}(\mathbf{D}_{\rm SAD}) -
  \tfrac{1}{2}\,\mathbf{K}(\mathbf{D}_{\rm SAD})$, diagonalise it, and
  split the occupied columns into per-spin densities. Costs one
  extra Fock build at SCF start but lands closer to the
  Kohn-Sham minimum.

The engine selects the packaging automatically based on whether a
`JKBuilder*` is supplied (UKS supplies one; UHF does not).

**Cost:** one atomic SCF per unique element (cached). For typical
molecules this is microseconds; even on a heavy atom (e.g. Z=86 Rn)
the atomic SCF is fast because it's spherically symmetric and a
single S-shell expansion.

**Strength:** robust on every system class vibe-qc supports. SAD is
the right default for ionic insulators, transition-metal complexes,
and any system where the Hcore failure mode would bite.

### SAP, superposition of atomic potentials (Lehtola 2020)

A Fock-mode method designed by Lehtola, Visscher, and Engel to give
**shell-structure-correct** starting orbitals without SAD's atomic-
SCF cost. Each atom $A$ contributes an **erf-screened Coulomb
potential** fit to a numerically-exact atomic SCF:

$$
V_{\rm SAP}^{(A)}(r) \;=\; -\sum_{i} c_i^{(A)}\,\frac{\operatorname{erf}\!\left(\sqrt{\alpha_i^{(A)}}\,r\right)}{r}
$$

where $(\alpha_i^{(A)}, c_i^{(A)})$ are tabulated for $Z = 1$ to $86$
(see [bundled datasets](#sap-datasets) below). The total SAP
potential is

$$
V_{\rm SAP}(\mathbf{r}) \;=\; \sum_{A} V_{\rm SAP}^{(A)}(|\mathbf{r} - \mathbf{R}_A|)
$$

and the initial Fock matrix is

$$
\mathbf{F}_{\rm SAP} \;=\; \mathbf{T} + \mathbf{V}_{\rm SAP}.
$$

For molecular UHF and UKS, vibe-qc diagonalises this spin-independent
Hamiltonian once and occupies the common orbital set separately through
$n_\alpha$ and $n_\beta$. Consequently
$\operatorname{tr}(\mathbf{D}_\alpha\mathbf{S})=n_\alpha$ and
$\operatorname{tr}(\mathbf{D}_\beta\mathbf{S})=n_\beta$ exactly; the guess
does not manufacture or discard electrons when the spin populations differ.

Note that $\mathbf{V}_{\rm SAP}$ *replaces* $\mathbf{V}_{\rm ne}$
entirely, the screened potential already includes the bare nuclear
attraction in its $r \to 0$ limit, and adding $\mathbf{V}_{\rm ne}$
would double-count it.

The matrix element of one $(\alpha_i, R_A)$ primitive is exactly
libint2's `Operator::erf_nuclear` with attenuation
$\omega = \sqrt{\alpha_i}$ and a "point charge" $-c_i$ at $R_A$:

$$
\langle\chi_\mu \mid V_{\rm Gaussian}(\alpha, R_A) \mid \chi_\nu\rangle
\;=\; \langle\chi_\mu \mid \frac{\operatorname{erf}(\sqrt{\alpha}\,|\mathbf{r}-\mathbf{R}_A|)}{|\mathbf{r}-\mathbf{R}_A|} \mid \chi_\nu\rangle
$$

So the engine loops over (atom $A$, primitive $i$) pairs, calls libint
once per pair, and accumulates into $\mathbf{V}_{\rm SAP}$.

**Cost:** one libint `erf_nuclear` engine call per (atom, primitive)
pair, typically 5-30 primitives per atom × $n_{\rm atoms}$, then one
diagonalisation. No atomic SCF; no Fock build. Cheaper than SAD on
heavy elements where the atomic SCF dominates.

**Strength:** unlike HCore, the shell structure is correct, the
first iteration sees orbitals that look like Hartree-Fock orbitals
rather than artefacts of nuclear attraction alone. Iteration counts
on light-atom molecules typically match SAD within ±1; on heavy
atoms SAP wins by 30-50% per [Lehtola 2020] Tables I-II.

**Limitations:**

* **Periodic SAP uses the actual Bloch potential.** The lattice-summed SAP
  potential (`compute_vsap_lattice`, built by a three-dimensional Ewald split
  and validated against the molecular SAP matrix) gives
  `F_SAP(k) = T(k) + V_SAP(k)`. It is diagonalised independently at every
  sampled k-point. True multi-k routes then retain their route's normal
  occupation policy (global Aufbau, fixed-per-k filling, or smearing where
  supported); Gamma density-seed routes use their normal integer target
  occupations, just as their Hcore starts do. The public GDF and RIJCOSX
  routes support RHF/RKS/UHF/UKS; BIPOLE supports
  RHF/ROHF/ROKS/RKS/UHF/UKS; GPW supports
  those six methods at Gamma and RKS/ROKS/UKS at true multi-k; GAPW supports
  RHF/RKS/UHF/UKS at Gamma and RKS at true multi-k. The exported legacy
  three-dimensional Ewald entry points support all six SCF families at Gamma
  or on their full-mesh routes, and the native periodic RHF/RKS entry points
  support SAP as well. GDF ROHF supports the atomic density and Fock guesses.
  AICCM real-Gamma/four-center supercell adapters accept AUTO or HCORE and
  fail before SCF for other requests; the option-aware
  neutral-Bloch and chi variants retain their own route capability checks.
  One- and two-dimensional SAP are likewise not yet defined. AUTO still
  resolves to SAD on routes that support it; HCORE-only adapters resolve
  AUTO to HCORE and record that choice.
* **Element coverage.** The bundled tables cover Z = 1-86. Every SAP route
  requires complete table coverage and raises an error naming a missing
  atomic number. Substituting an unscreened bare nuclear potential would
  change the selected physical construction.

(sap-datasets)=
#### SAP datasets

Two atomic-potential libraries ship with vibe-qc in
`python/vibeqc/basis_library/basis/`:

| File | Reference | Default for |
|------|-----------|-------------|
| `sap_helfem_large.g94` | All-electron, Helfem fully-numerical atomic SCF | Non-relativistic calculations |
| `sap_grasp_large.g94`  | Relativistic Dirac-Hartree-Fock, GRASP | x2c / DKH calculations |

The engine selects `sap_helfem_large` by default; the GRASP table
becomes the AUTO choice once relativistic-Hamiltonian detection
lands (the data is already in the wheel; only the selection
heuristic is missing).

**Where the tables are looked up.** SAP tables resolve at run time,
with the same precedence libint uses for orbital basis sets:
`$LIBINT_DATA_PATH/basis/<name>.g94` first, then the data directory
recorded when the binary was built. `import vibeqc` already points
`$LIBINT_DATA_PATH` at the bundled library, so a wheel installed
anywhere -- or a relocatable bundle whose build tree is long gone --
finds its tables without you setting anything. Set
`LIBINT_DATA_PATH` yourself only to point SAP (and the orbital
bases) at a different library. If a table is genuinely missing, the
error lists every path that was searched.

**Citation requirement.** Any published result that uses the SAP initial
guess must cite both the assessment introducing the transferable atomic
potential approach and the Gaussian-basis implementation/table paper:

> S. Lehtola, *Assessment of Initial Guesses for Self-Consistent Field
> Calculations. Superposition of Atomic Potentials: Simple yet Efficient*,
> *J. Chem. Theory Comput.* **15**, 1593-1604 (2019).

> S. Lehtola, L. Visscher, E. Engel, *Efficient Implementation of the
> Superposition of Atomic Potentials Initial Guess for Electronic
> Structure Calculations in Gaussian Basis Sets*,
> *J. Chem. Phys.* **152**, 144105 (2020).

See [docs/license.md §3a](../license.md) for the full bundled-data
inventory.

### AUTO, pick the right guess for me

The default for all six SCF options structs. AUTO inspects the
molecule and chosen system flags through `GuessEngine::resolve_auto`.
The shared resolver also accepts the route's capability set: it retains the
system-preferred construction when supported, otherwise selects its supported SAD parent, then HCORE when
that route supports HCORE. Explicit selectors never undergo this substitution.
`run_periodic_job` defaults to the literal `"AUTO"`, preserving the
distinction between an omitted keyword and an explicit `"SAD"`.

| System | AUTO resolves to | Reason |
|--------|------------------|--------|
| Periodic, atomic-guess-capable routes | **SAD** | Preserves the established atomic-density start; no evidence-backed metal/oxide classification is currently available |
| AICCM real-Gamma and four-center adapters | **HCORE** | These supercell SCF loops currently expose only their core-Hamiltonian construction; explicit SAD and other unsupported guesses fail |
| Molecular, isolated open-shell atom | **PATOM** | A lone atom has no molecular field to break the SAD basin, so one in-field re-polarisation step selects the requested atomic term |
| Molecular, other open-shell system | **SAD** | Spin polarisation develops via per-spin Fock asymmetry |
| Molecular, containing transition or f-block atoms (except the isolated open-shell case above) | **SAD** | SAP closed-shell averaging can land the wrong d-shell occupation; select `PATOM` (now shipped, molecular) for a re-polarised SAD start |
| Molecular, closed-shell, light atoms | **PATOM** | Preserves the established molecular default density and energy basins |

The periodic choice is a conservative compatibility policy, not a claim that
SAD is universally optimal for solids. Van Lenthe et al. (2006) and Lehtola
(2019, 2020) support the atomic density/potential constructions; their molecular
comparisons do not establish a metal-versus-oxide selector for our periodic
backends. No such classifier is encoded. An `atomic_spins` seed requires the
resolved kind to be SAD; AUTO resolving to HCORE rejects the seed.

The hints (`is_open_shell`, `has_transition_metal`, `is_periodic`)
are filled in automatically by each SCF wrapper:

* `is_open_shell` from `n_alpha != n_beta`
* `has_transition_metal` from a Z-range scan over the molecule
  (Sc-Zn, Y-Cd, La/Hf-Hg, Ce-Lu, Ac/Rf-Cn, Th-Lr)
* `is_periodic` from which entry point the caller used

The resolved kind is recorded on the result so the choice is visible
in the SCF log:

```python
result = run_rhf(mol, basis, opts)
# (look in the SCF log; "initial guess: SAP (Lehtola/Visscher/Engel 2020 …)"
```

### PATOM, HUECKEL, and MINAO

Three further guesses landed together. `PATOM` and molecular `MINAO` are
density-mode guesses that return a starting density the SCF consumes directly;
`HUECKEL` is Fock-mode and is diagonalised once to build the starting density.
Periodic availability is route-specific. The shared adapter supplies the
HUECKEL lattice Fock and MINAO projection wherever the selected driver can
consume them. PATOM additionally requires a route-local periodic J/K seam for
its in-field step. A route without that seam rejects PATOM before SCF rather
than changing it to Hcore. Open-shell Python SAP/HUECKEL build per-spin
densities from the shared helpers; open-shell MINAO uses the projected total
density with the proportional spin split; supported open-shell PATOM routes
start from SAD and apply one HF-like periodic in-field re-polarisation step per
spin.

* **PATOM** takes the SAD density and applies one in-field mean-field
  re-polarisation step, letting the superposed free-atom densities
  relax in the molecular field before the SCF proper. It is the
  density-mode upgrade of SAD for cases where SAD's spherical atoms are
  too rigid, notably transition-metal d-shell ordering. `run_uhf`
  forwards its `JKBuilder` to the engine only for PATOM, so no other
  guess's packaging changes. The periodic closed-shell drivers run the same
  one-step re-polarisation: Gamma forwards the periodic `JKBuilder`, while
  multi-k builds the in-field Fock in real space and diagonalises it at each
  k-point. The Python Gamma/multi-k Ewald, GDF, and BIPOLE UHF/UKS drivers use
  their own periodic J/K builders for the open-shell per-spin in-field step.
  (ORCA PAtom; builds on Van Lenthe SAD.)
* **HUECKEL** is a parameter-free generalised Wolfsberg-Helmholz
  (extended-Hückel) guess. The off-diagonal elements are
  $H_{\mu\nu} = \tfrac{1}{2} K\, S_{\mu\nu}\,(H_{\mu\mu} + H_{\nu\nu})$
  with $K = 1.75$, built over *computed* atomic-orbital energies rather
  than tabulated parameters, then diagonalised to a density. (Wolfsberg
  and Helmholz 1952; Hoffmann 1963; the parameter-free variant of
  Lehtola 2019.) For periodic closed-shell RHF/RKS the same GWH form is
  built as a lattice Fock, $H_{\mu\nu}(\mathbf{g})$, from the lattice overlap
  $S_{\mu\nu}(\mathbf{g})$ and diagonalised at Gamma or each k-point. The
  closed-shell Python periodic drivers Gamma-fold that lattice Fock and
  inject the resulting g=0 density. Accepts `initial_guess="huckel"` or
  `"hueckel"`.
* **MINAO** builds free-atom reference densities in the ANO-RCC minimal
  contraction and projects them onto the target basis (Knizia 2013
  minimal-AO concept; ANO-RCC reference data). For periodic systems the
  projection is lattice-summed, `P = S(Γ)^{-1} S_tm(Γ)` with the cross-basis
  overlap `S_tm(Γ) = Σ_g <χ(0)|χ_ref(g)>`; it is wired into the closed-shell
  Γ and multi-k RHF/RKS drivers and the closed-shell Python periodic density
  seam, seeding the g=0 density cell (like SAD).

### FRAGMO, superposition of converged fragment densities

Where SAD superposes converged *atomic* densities, FRAGMO superposes
converged *fragment* densities. You partition the molecule into fragments;
each fragment is converged with its own SCF, and the fragments' density
matrices are assembled **block-diagonally** into the supersystem guess:

$$
\mathbf{D}_{\rm FRAGMO} \;=\; \bigoplus_{F \in \text{fragments}} \mathbf{D}_{F},
$$

each fragment block $\mathbf{D}_F$ placed in the supersystem AO rows/columns
of that fragment's atoms (intra-fragment cross terms preserved;
inter-fragment blocks left at zero). FRAGMO is resolved in the `run_*`
wrapper and injected via `opts.read_density` exactly like READ, so one path
covers RHF / UHF / RKS / UKS.

```python
import vibeqc as vq
from vibeqc import Fragment, run_rhf

opts = vq.RHFOptions()
opts.initial_guess = vq.InitialGuess.FRAGMO
# Two fragments: atoms 0-2 and atoms 3-5 (each neutral closed-shell).
result = run_rhf(mol, basis, opts, fragments=[[0, 1, 2], [3, 4, 5]])

# A charged / open-shell fragment uses an explicit Fragment spec:
result = run_rhf(mol, basis, opts,
                 fragments=[Fragment(atoms=[0, 1, 2], charge=+1),
                            Fragment(atoms=[3, 4, 5], charge=-1)])
```

Each fragment is converged with the supersystem's method and functional (a
closed-shell fragment with RHF/RKS, an odd-electron or `multiplicity > 1`
fragment with UHF/UKS); for an open-shell supersystem each fragment's α/β
blocks feed the matching supersystem spin.

**Strength.** For weakly-interacting assemblies (hydrogen-bonded dimers,
van-der-Waals complexes, host-guest systems) the fragments already carry
almost all of the electronic structure, so the guess starts much closer to
the supersystem solution than SAD and converges in noticeably fewer SCF
iterations. For a strongly-coupled / covalently-bonded partition the
inter-fragment blocks matter and SAD or SAP is usually better.

**Honest errors.** The fragments must be a disjoint partition covering every
atom exactly once, and their charges must sum to the molecule charge
(electron conservation); each violation raises a clear `ValueError`. A
fragment SCF that fails to converge raises; passing `fragments=` without
`initial_guess=FRAGMO` raises. FRAGMO currently requires a *named* basis set
(it rebuilds each fragment's basis by name). Automatic `fragments=` assembly
is available in molecular wrappers, native periodic Python wrappers, and
`run_periodic_job`. `run_neb` has no
`fragments=` keyword; molecular NEB accepts FRAGMO only when the caller has
already populated complete `read_density` matrices in the selected SCF
options.

(Roadmap D2h, the fragment-MO assembly guess; assembling the fragment
*densities* is equivalent to a block-diagonal MO assembly for seeding and
needs no occupation bookkeeping.)

For periodic systems, `PeriodicFragment` gives each listed atom an integer
lattice image in the same order as `atoms`. Lattice vectors are the columns
of `system.lattice`. For a two-atom fragment crossing the third cell boundary:

```python
fragment = vq.PeriodicFragment(
    atoms=[0, 1], images=[[0, 0, 0], [0, 0, -1]],
    charge=0, multiplicity=1,
)
result = vq.run_periodic_job(
    system, basis, method="RHF", jk_method="gdf", kpoints=[1, 1, 3],
    initial_guess="FRAGMO", fragments=[fragment],
)
```

The fragments must partition every input-cell atom exactly once. Home-cell
fragments may use ordinary `Fragment` objects or atom-index lists.
`spin_orientation=-1` reverses a periodic fragment's spin for an AFM seed.
Fragment charges must sum to the cell charge. Primitive reduction must be
disabled to preserve atom ownership; a shared translation of an entire
fragment leaves its Bloch density unchanged. For AO-owned image translations
$T_\mu$, the embedding is
$D_{\mu\nu}(k)=e^{-ik\cdot(T_\mu-T_\nu)}D^F_{\mu\nu}$.
The target driver then normalizes the complete density in its weighted
Bloch overlap. This constructs finite isolated fragment references; it does
not solve a periodic fragment SCF. Results record physical construction
`FRAGMO` and density transport `READ`.

### READ: restart from a prior calculation

`READ` skips a from-scratch guess and starts an SCF from a density you already
have. It is useful for an explicitly managed geometry sequence, a supported
first-band NEB seed, or continuing a job that ran out of iterations. The
accepted source depends on the entry point; READ is not a universal warm-start
switch for every optimizer and reaction-path driver. Select it with
`opts.initial_guess = vq.InitialGuess.READ` and use one of the sources that
the chosen driver documents.

**In-memory molecular single point**, from a prior result, via the
`read_from=` keyword on the RHF/UHF/RKS/UKS wrappers:

```python
r1 = vq.run_rhf(mol, basis, opts)                     # first point
opts2 = vq.RHFOptions()
opts2.initial_guess = vq.InitialGuess.READ
r2 = vq.run_rhf(mol2, basis, opts2, read_from=r1)     # restart from r1
```

`read_from` accepts prior results from all six molecular SCF families.
Results retain their source basis, so changed geometries and bases use the
same least-squares density projection as QVF and Molden sources. The two
projected spin densities use the shared READ normalization in the target
overlap metric. A changed multiplicity can populate an empty source spin channel by
transferring density from the other channel; a triplet with no beta electrons
can therefore seed a singlet. Populated source channels retain their separate
spatial patterns, including local broken symmetry after geometry projection;
each is rescaled to its target population. This differs from the mixing
policy for an unseeded atomic superposition.
NEB and geometry optimizers retain those snapshots across their internal
warm starts. An internal READ transport retains the requested and physical
construction in result metadata and citations.

**From a `.qvf` or Molden file**, via `opts.read_path`:

```python
opts.read_path = "prev.qvf"      # reads the QVF restart section
# or
opts.read_path = "prev.molden"
# ORCA's default exporter name is accepted directly too:
opts.read_path = "prev.molden.input"
```

Write the source with QVF output (`run_job(..., output_qvf=True)`), which
embeds `wavefunction.gto` for molecular jobs and
`x_vibeqc.bloch_wavefunction` for periodic jobs, or write a
Molden file for molecular / Gamma-style orbital restarts. READ accepts both
the raw Gaussian contraction coefficients written by vibe-qc and the
primitive-normalized convention written by ORCA's `orca_2mkl`. It selects the
convention from the MO orthonormality condition and refuses the file if neither
interpretation is valid, if an MO omits its occupation or has a non-finite
coefficient, or if any occupied MO has an incomplete or duplicate
AO-coefficient block, rather than silently changing the electron count.
If both contraction interpretations pass, the smaller orthonormality residual
wins; an exact tie retains the historical raw-coefficient interpretation.
Coefficient tables rounded too coarsely to establish the invariant are
rejected rather than guessed.
This restart reader currently supports pure-spherical, separate-shell Molden
bases; combined `sp` shells and Cartesian `[6D]` / `[10F]` variants are not a
supported restart surface.

Closed-shell drivers use the prior total density, open-shell drivers the
per-spin pair (`opts.read_density` / `opts.read_density_alpha` /
`opts.read_density_beta`).

**Projection across a changed basis or geometry.** A file source carries
its own basis and geometry, so the prior density is projected onto the
*current* basis automatically:

$$
\mathbf{D}_{\rm cur} = \mathbf{P}\,\mathbf{D}_{\rm prior}\,\mathbf{P}^{T},
\qquad \mathbf{P} = \mathbf{S}_{tt}^{-1}\,\mathbf{S}_{tm},
$$

with $\mathbf{S}_{tt}$ the current-basis overlap and $\mathbf{S}_{tm}$
the cross-basis overlap $\langle \chi^{\rm cur} | \chi^{\rm prior} \rangle$.
The projection is exact when the source and target bases coincide at the same
geometry. It prepares a prior density for the **current call's** basis and
geometry, so an explicitly managed scan can restart each point from a file.
For molecular Gaussian NEB, `options.read_path` is resolved once against the
reactant endpoint to seed the initial band; later outer iterations use NEB's
own `warm_start` cache. Modern in-memory results carry an AO-basis snapshot and use the same
projection as file sources. Explicit preloaded matrices are already in the
current basis; a matrix alone cannot describe a geometry change. Legacy
result objects without a basis snapshot are rejected even when their AO
dimension matches.

**Aliases.** In the periodic guess-string interface, `"moread"` and
`"coread"` are accepted ORCA-style spellings of `READ`; the molecular
`run_*` drivers take the enum directly.

**Periodic restart.** `READ` also restarts periodic SCF jobs, including
explicit geometry-scan and extrapolation sequences on slabs and bulk.
`vq.run_periodic_job(..., initial_guess="read", read_from=prior_or_path)`
takes a prior periodic result (in-memory) or a `.qvf` / `.molden` path at
Gamma. The prior g=0 cell density is projected onto the current cell basis
(the same MINAO-style projector, evaluated with the cell overlaps) and
injected at g=0, where it Bloch-sums to a k-independent
$\mathbf{D}(\mathbf{k})$ for the first Fock build. Wired on the Ewald, GDF,
and BIPOLE Gamma drivers.

Closed-shell multi-k restarts are supported from an **in-memory** prior result
or a current vibe-qc **`.qvf`** archive on BIPOLE, GDF, RIJCOSX, GPW, and
GAPW routes. In
this case vibe-qc uses the physical SCF density when present in memory;
otherwise it rebuilds every complex Bloch density block from the prior
coefficients and occupations,
$\mathbf{D}(\mathbf{k}) = \mathbf{C}(\mathbf{k}) f(\mathbf{k})
\mathbf{C}^{\dagger}(\mathbf{k})$, and starts the next multi-k SCF from that
per-k density list. Current QVF archives carry the all-k restart data in the
first-party vendor section `x_vibeqc.bloch_wavefunction`, including source
basis, lattice, k coordinates and quadrature weights. Exact mesh permutations
preserve complex densities and occupations. A Gamma source can seed a new mesh
through an explicit on-site density projection followed by weighted overlap
normalization. Different non-Gamma quadratures or changed non-Gamma source
bases fail closed: they require a Bloch cross-basis or finite-real-space
interpolation contract. A single selected non-Gamma visualization block is
insufficient. A mesh containing one non-Gamma point still requires Bloch
transport and a matching target quadrature; its density may be complex even
though it has only one block. Readers use source coordinates, rather than
the block count, to identify Gamma. An all-k QVF payload takes precedence
over selected visualization orbitals. Gamma-only readers reject a non-Gamma
source even if its density happens to be real. Older all-k archives without
weights must be re-exported.
Open-shell in-memory all-k restarts preserve separate alpha/beta densities
on BIPOLE, GDF, RIJCOSX and GPW routes, including restricted-open references
where the underlying SCF method exists. The real-lattice BIPOLE adapter
checks the inverse Bloch round trip and refuses states it cannot represent
without discarding complex information.
When the target multiplicity changes, spin normalization uses the global
populations `sum_k w_k Tr(D_spin(k) S(k))`. Each populated channel is rescaled
uniformly across the mesh; an empty channel is seeded from the other spin.
This preserves complex phases and relative k-point occupations.
It does not impose integer populations separately at each k point. This
contract is shared by Ewald, GDF/RIJCOSX, BIPOLE and GPW restart adapters.
READ can switch between restricted and spin-resolved references on these
routes. A restricted source supplies half its total density to each spin
before target population normalization. A spin-resolved source supplies the
sum of its alpha and beta densities to a restricted target. Each spin channel
is validated before summing; incomplete spin data cannot fall back to a
redundant total density. Both conversions preserve the full k-point mapping. Gamma GDF retains its
selected Hamiltonian route when consuming a prepared READ density.
A current restricted all-k QVF archive can therefore seed an open-shell job.
Current open-shell QVF archives store both physical spin densities in the
same vendor section using its version 1.1 density payload. Each k point has
separate complex alpha and beta AO matrices, with source basis, lattice and
quadrature metadata. This preserves the actual SCF state, including local
magnetic order, without reconstructing it from later canonical orbitals.
READ verifies the relevant member checksums and validates both channels
before projection, normalization or conversion to a restricted total.
Older vibe-qc readers reject this spin payload; the new reader continues to
accept version 1.0 restricted orbital payloads. An older open-shell archive
containing only selected visualization orbitals is insufficient for an
all-k spin restart.

This `run_periodic_job` restart surface does not automatically extend to
every geometry driver. Fixed-cell BIPOLE optimization rejects READ because it
cannot preserve the restart across displaced geometries; run the restart
single point first, then optimize from a geometry-defined SAD/HCORE/ATOMSPIN
guess. Periodic Gaussian BIPOLE NEB likewise rejects READ and FRAGMO before
dry-run: its image and finite-difference displacement loop has neither a
geometry-projected full-lattice restart seam nor a fragment-assembly seam,
and `run_neb` has no `read_from=` or `fragments=` keyword.
All four BIPOLE NEB SCF families support PATOM; unrestricted paths may also
use ATOMSPIN. Run any requested
restart or fragment guess as a supported single point first, then begin the
NEB band from SAD/HCORE (or an unrestricted magnetic seed).

Like `HCORE`, `READ` carries no citation: it is mechanical, not a published
method.

## When to use which, decision table

| Situation | Recommended | Why |
|-----------|-------------|-----|
| Default, any molecular system | **AUTO** | Engine selects the documented system and route policy |
| Default, any periodic system | **AUTO** | Selects the supported conservative route default |
| Difficult open-shell convergence | **SAD** | Robust; OH/6-31G* false-minimum example |
| Heavy atoms (Z > 36), closed shell | **SAP** | Cheaper than SAD's atomic-SCF cost |
| Transition-metal complex | **SAD** (AUTO default) or **PATOM** | PATOM re-polarises the SAD densities, helping the d-shell ordering that SAP's closed-shell averaging mis-orders |
| Unit tests, deterministic reference | **HCORE** | System-independent, no implementation-dependent atomic SCF |
| Explicit molecular/periodic scan sequence | **READ** | Restart each supported single-point call from the previous result or a projected file source |
| Molecular Gaussian NEB first-band seed | **READ** via `options.read_path` or complete preloaded densities | `run_neb` has no `read_from=` keyword; its internal `warm_start` handles later outer iterations |
| BIPOLE fixed-cell optimization | **SAD/HCORE/ATOMSPIN**, not READ | The optimizer rejects READ across displaced geometries; run a READ single point first if needed |
| Weakly-interacting dimer / vdW complex / cluster | **FRAGMO** | Superpose converged fragment densities; starts close to the supersystem solution, fewer iterations than SAD |
| Broken-symmetry singlet | **SAD** + `guessmix_angle_deg` (when it ships) | α HOMO/LUMO rotation to break spatial symmetry |

The remaining method marked "when it ships" (broken-symmetry
`guessmix_angle_deg`) is still scaffolded in the engine; its
concrete implementation is tracked in
[`docs/roadmap.md`](../roadmap.md) under the initial-guess track
(§G2) and the DIIS-family section (§D1).

### Molecular ROHF and ROKS

ROHF and ROKS use the same native guess constructions as UHF and UKS.
The Roothaan adapter supplies the molecular overlap, Hcore and J/K builder,
so its selector set includes the overlap-dependent constructions.

| Value | Effect |
|-------|--------|
| `SAD` | Atomic density start |
| `HCORE` / `CORE` | Diagonalise the core Hamiltonian |
| `AUTO` | Shared molecular policy: SAD for molecular open shells and PATOM for isolated spin-polarised atoms |
| `SAP`, `PATOM`, `HUECKEL`, `MINAO` | Execute the corresponding native molecular builder |
| `READ` | Load a compatible source through `run_job(read_from=...)`, `read_path`, or prepared spin densities |
| `FRAGMO` | Prepare fragment densities through `run_job(fragments=...)` |

Enum and string spellings are accepted. `run_job(initial_guess=...)` also
preserves the selector for ROHF/ROKS references. Builder errors propagate;
an accepted selector never substitutes Hcore after a construction failure.
Malformed selectors fail even when `initial_density` is supplied.

```python
from vibeqc.rohf import ROHFOptions

opts = ROHFOptions(initial_guess="sap")
```

ROHF/ROKS results expose `guess_selection.requested`, `.effective`, and
`.transport`. The transport can be READ when a prepared density is supplied;
this is separate from resolving an AUTO construction. Guess citations use the
same molecule-aware resolver as execution.

**SAD populations are normalized before the first Fock build.** Free-atom
Hund populations need not match the molecular multiplicity: for example,
the neutral atomic superposition for OH carries 6 alpha and 3 beta electrons,
whereas the doublet requires 5 and 4. The shared density constructor now
enforces the target counts for unrestricted and restricted-open-shell
routes alike. This is particularly important for ROHF/ROKS, whose closed,
open and virtual projectors are derived from the seed density itself.
Issue #119 established the consequence: an uncorrected NH2/STO-3G default
start could converge to an excited state 95 mHa above the ground state.
The normalization contract above preserves the intended Hund pattern while
satisfying the target populations; it does not postpone correction until an
SCF occupation update.

## Worked examples

### Comparing iteration counts across guesses

```python
import vibeqc as vq
from vibeqc import Atom, Molecule, BasisSet, RHFOptions, run_rhf

mol = Molecule([
    Atom(8, [0.0,  0.0,  0.0]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])

for basis_name in ["sto-3g", "6-31g*", "cc-pvdz"]:
    basis = BasisSet(mol, basis_name)
    for kind in [vq.InitialGuess.HCORE,
                 vq.InitialGuess.SAD,
                 vq.InitialGuess.SAP]:
        opts = RHFOptions()
        opts.initial_guess = kind
        opts.conv_tol_energy = 1e-10
        r = run_rhf(mol, basis, opts)
        print(f"{basis_name:>10}  {kind.name:<6}  "
              f"n_iter={r.n_iter:2d}  E={r.energy:.6f}")
```

Output on H₂O (typical):

```
    sto-3g  HCORE   n_iter= 8  E=-74.964156
    sto-3g  SAD     n_iter= 8  E=-74.964156
    sto-3g  SAP     n_iter= 8  E=-74.964156
    6-31g*  HCORE   n_iter=12  E=-76.006529
    6-31g*  SAD     n_iter=10  E=-76.006529
    6-31g*  SAP     n_iter=11  E=-76.006529
   cc-pvdz  HCORE   n_iter=12  E=-76.023841
   cc-pvdz  SAD     n_iter=11  E=-76.023841
   cc-pvdz  SAP     n_iter=11  E=-76.023841
```

All three converge to the same energy to machine precision; iteration
counts shift by ±1-2 depending on basis. SAP's advantage grows with
atom heaviness, Lehtola's Tables I-II show 40-60% iteration-count
reductions on Au / Ag / I-containing systems.

### Open-shell radical (OH)

The textbook example where HCore gets the wrong minimum:

```python
from vibeqc import Atom, Molecule, BasisSet, UHFOptions, run_uhf
import vibeqc as vq

A2B = 1.0 / 0.529177210903    # Angstrom → bohr
mol = Molecule(
    [Atom(8, [0.0, 0.0, 0.0]),
     Atom(1, [0.97 * A2B, 0.0, 0.0])],
    multiplicity=2,
)
basis = BasisSet(mol, "6-31g*")

# HCore lands on a false minimum 0.16 Ha above the true one:
opts = UHFOptions()
opts.initial_guess = vq.InitialGuess.HCORE
r_hcore = run_uhf(mol, basis, opts)
print(f"HCORE: E = {r_hcore.energy:.6f}  S² = {r_hcore.s_squared:.3f}")
# HCORE: E = -75.221182  S² = ~0.764 (or worse)

# SAD finds the right minimum, S² near 0.755 (doublet):
opts.initial_guess = vq.InitialGuess.SAD
r_sad = run_uhf(mol, basis, opts)
print(f"SAD:   E = {r_sad.energy:.6f}  S² = {r_sad.s_squared:.3f}")
# SAD:   E = -75.380931  S² = 0.755
```

`run_uhf` defaults to `AUTO`, which resolves to SAD for molecular
open shells and PATOM for isolated open-shell atoms, so omitting `opts.initial_guess` entirely also
reaches the correct minimum.

### Periodic ionic insulator (NaCl)

```python
import vibeqc as vq
import numpy as np
from vibeqc import Atom, BasisSet, PeriodicSystem, PeriodicRHFOptions

a = 10.7      # NaCl rocksalt lattice parameter, bohr (close enough for sto-3g)
lat = a * np.eye(3)
atoms = [
    Atom(11, [0.0, 0.0, 0.0]),         # Na
    Atom(17, [a/2, a/2, a/2]),         # Cl
]
sysp = PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
basis = BasisSet(sysp.unit_cell_molecule(), "sto-3g")

# AUTO → SAD for periodic. The default works.
opts = PeriodicRHFOptions()
opts.conv_tol_energy = 1e-8
result = vq.run_rhf_periodic_gamma(sysp, basis, opts)
print(f"NaCl Γ-only RHF: E = {result.energy:.6f}, "
      f"n_iter = {result.n_iter}")

# If you explicitly request HCORE, you may hit the bombing failure
# mode (first-iter energy ~+30 000 Ha, recovers slowly or not at all):
opts.initial_guess = vq.InitialGuess.HCORE
# ... (don't do this for ionic insulators)
```

The SAD choice closes the v0.5.6 NaCl-bombing failure mode that
prompted this whole refactor.

## Inspecting what AUTO chose

Every SCF result exposes `guess_selection.requested`, `.effective`, and
`.transport`. For an ordinary periodic AUTO run they are AUTO, SAD, SAD.
A convergence retry can retain its original physical construction while
transporting the current density through READ. The output prints, for example,
`initial_guess = AUTO -> SAD`; the manifest and citation assembly use the
effective construction. An explicit READ restart has READ as its physical
construction and needs no initial-guess paper citation.

In-memory periodic restarts prefer the returned physical density blocks over
MO coefficients from a later diagonalization. Closed-shell GPW/GAPW results
include the actual per-k overlap matrices so weighted populations can be
checked against the metric used in SCF.

```python
selection = result.guess_selection
print(selection.requested.name, selection.effective.name, selection.transport.name)
```

For preflight, `vibeqc.guess.select_initial_guess` accepts a system, selector,
and concrete capability set. `periodic_guess_capabilities` is the shared
registry used by public runners and direct Python adapters. Native entry
points use the same selector domain and AUTO resolver and validate their
matrix/context capabilities before construction.

## Diagnosing convergence problems

When SCF won't converge or lands on the wrong state, the initial
guess is one of three places to look (the others are damping/DIIS
and the system geometry). Quick checklist:

1. **Is a periodic energy diverging or oscillating?** Preserve the input and
   iteration trace and report a reproducible numerical defect. A changed
   guess does not establish that the Hamiltonian, Coulomb gauge or density
   propagation is correct. Do not mask the behavior by tuning damping.
2. **Is it converging but to the wrong S²?** Open-shell trap. Try
   `SAD` (the ordinary open-shell AUTO choice) and check `result.s_squared` against the
   expected $S(S+1)$ from the multiplicity. For broken-symmetry
   singlets, use `SAD` and (once it ships) `guess_request.guessmix_angle_deg
   = 45`.
3. **Iteration count creeping up over a geometry scan?** Use `READ` to
   restart from the previous step's density (`read_from=` the prior
   result, or `read_path=` a `.qvf`); it projects across the geometry
   change. See the READ section above.
4. **Periodic system with a charge density that looks "ionic" but
   the metal sites are wrong?** SAD is doing its job, but the AUTO-
   selected even-spin split may not bias correctly for AFM / FM
   states. Use SAD explicitly + (future) `atomic_occupations` to set
   per-element starting occupations.

## Extending the engine

Adding a new method is a small follow-up commit on the existing
scaffold. The contract:

1. **Declare it.** Add the enum value in `cpp/include/vibeqc/guess.hpp`
   (the nine current kinds are all implemented; this step is for a
   future tenth).
2. **Implement it.** Add a free function in `cpp/src/guess.cpp`
   (or a new file `cpp/src/guess_<name>.cpp` and add it to the
   CMakeLists) that builds whichever artifact the method produces
   (density, Fock, or MOs).
3. **Wire it.** Add a `case InitialGuess::<NAME>:` arm in
   `GuessEngine::build_closed_shell` (and `build_open_shell` if it
   handles spin); populate the matching field on `GuessResult` and
   write a `provenance` string.
4. **Test it.** Add at least one parity test
   (`tests/test_guess.py`): the new method must converge to the
   same total energy as HCORE / SAD on a small system to machine
   precision.

See the existing SAP implementation
([Archived commit `d2fbfe9`](https://vibe-qc.com/docs/))
for a worked example. The whole landing was ~500 lines including
tests, citation entry in `docs/license.md`, and CHANGELOG.

(guess-result-types)=
## Result types, closed-shell vs open-shell

The engine returns one of two tagged structs depending on which entry point
you call. Current molecular constructors populate the density field after any
required one-particle diagonalisation. An empty density is the explicit HCORE construction contract; it is never
used to conceal another builder's failure. The Fock and MO fields remain
reserved for native artifact paths; periodic Fock artifacts use the Python
adapter.

### `GuessClosedShellResult`

```cpp
struct GuessClosedShellResult {
    InitialGuess resolved_kind;  // what AUTO actually picked
    std::string provenance;      // human-readable trace
    Eigen::MatrixXd D;           // constructed starting density
    Eigen::MatrixXd F_guess;     // reserved; periodic Fock artifacts use Python adapter
    Eigen::MatrixXd C_guess;     // reserved MO-mode artifact
    Eigen::VectorXd eps_guess;
};
```

### `GuessOpenShellResult`

```cpp
struct GuessOpenShellResult {
    InitialGuess resolved_kind;
    std::string provenance;
    Eigen::MatrixXd D_alpha, D_beta;               // constructed densities
    Eigen::MatrixXd F_guess_alpha, F_guess_beta;   // reserved
    Eigen::MatrixXd C_guess_alpha, C_guess_beta;   // reserved
    Eigen::VectorXd eps_guess_alpha, eps_guess_beta;
};
```

The Python helpers in `vibeqc.guess`
(`initial_density_closed_shell`, `initial_densities_open_shell`)
already unpack the result for the Python periodic SCF drivers; most
users won't touch the result type directly.

### `InitialGuess` enum

```cpp
enum class InitialGuess {
    AUTO,      // default: engine picks per system
    HCORE,     // F = T + V_ne
    SAD,       // superposition of atomic densities
    SAP,       // superposition of atomic potentials (Lehtola 2020)
    PATOM,     // SAD + one in-field re-polarisation step
    HUECKEL,   // parameter-free generalised Wolfsberg-Helmholz
    MINAO,     // minimal-AO (ANO-RCC) reference projection
    READ,      // restart from a prior result / .qvf / .molden
    FRAGMO,    // converged fragments; periodic images embedded in Bloch space
};
```

Bound to Python as `vibeqc.InitialGuess`.

## References

* S. Lehtola, L. Visscher, E. Engel, *Efficient Implementation of the
  Superposition of Atomic Potentials Initial Guess for Electronic
  Structure Calculations in Gaussian Basis Sets*,
  *J. Chem. Phys.* **152**, 144105 (2020).
  [DOI:10.1063/5.0004046](https://doi.org/10.1063/5.0004046)
* S. Lehtola, *Assessment of Initial Guesses for Self-Consistent Field
  Calculations*, *J. Chem. Theory Comput.* **15**, 1593 (2019).
  [DOI:10.1021/acs.jctc.8b01089](https://doi.org/10.1021/acs.jctc.8b01089)
* J. H. Van Lenthe, R. Zwaans, H. J. J. Van Dam, M. F. Guest,
  *Starting SCF Calculations by Superposition of Atomic Densities*,
  *J. Comput. Chem.* **27**, 926 (2006).
  [DOI:10.1002/jcc.20393](https://doi.org/10.1002/jcc.20393)
* Q. Sun et al., *PySCF: the Python-based simulations of chemistry
  framework*, *WIREs Comput. Mol. Sci.* **8**, e1340 (2018), 
  description of the `minao` projection used in PySCF.

## See also

* [SCF convergence](scf_convergence.md), damping, DIIS, EDIIS, and
  level-shift options that interact with the initial guess.
* [Linear dependence](linear_dependence.md), the canonical
  orthogonalisation step that SAP and the F-diagonalise SAD
  packaging both rely on.
* [Initial guesses: a hands-on walkthrough](../tutorial/initial_guess_walkthrough.md)
  - hands-on comparison of guesses on H₂O, OH, and NaCl.
* [CHANGELOG](../changelog.md), v0.9.x initial-guess track entries.

### Direct periodic option adapters

The `run_rhf_periodic_scf` and `run_rks_periodic_scf` dispatchers preserve
restart densities, source paths, atomic spin controls and ECP data when
converting option types. Their Ewald multi-k routes accept
`initial_density_k` explicitly; a Gamma option matrix is not substituted for
a complete per-k payload. Native DIRECT_TRUNCATED routes accept target-ordered
`initial_density_k` blocks and the public wrappers accept `read_from` sources.
Their real lattice representation requires a complete time-reversal-compatible
state. Python Ewald drivers
reject ECP Hamiltonians, including direct calls, because those drivers do not
assemble the periodic pseudopotential operator. Use a supported GDF or native
DIRECT_TRUNCATED route for ECP calculations.

Ewald RHF/RKS/UHF/UKS results retain physical per-k density blocks separately
from their cutoff-dependent lattice view. READ validates time reversal before
folding those blocks to the real lattice representation, while retaining the
full complex D(k) for the SCF. Restricted-open lattice routes require a full
set of BvK representatives because they reconstruct D(k) from lattice storage;
a missing representative fails explicitly. Ewald ROHF/ROKS do not implement
SPINLOCK schedules or pattern hold and reject active requests.
