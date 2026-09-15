# DFT+U for correlated transition-metal oxides

Plain LDA and GGA functionals badly mis-describe the localized `3d`
and `4f` electrons of transition-metal oxides: the residual
self-interaction error smears a manifold that should be integer-filled
and atomic-like into a partially-filled, over-delocalized band, which
in turn collapses the fundamental gap. Rock-salt NiO is the textbook
casualty. Experiment puts it at a wide ~4.3 eV charge-transfer
insulator; plain LSDA gives a small fraction of that, sometimes a
metal. This tutorial adds a Hubbard `U` correction on the Ni `3d`
channel with vibe-qc's `vq.dft_plus_u` / `vq.HubbardSite` surface,
runs the open-shell periodic SCF, and shows the effect of `U` on the
total energy and the spin-resolved gap.

**Wall time:** this is a slow calculation. The open-shell BIPOLE
periodic SCF on a tight 3D rock-salt cell builds a full lattice-summed
Fock matrix every iteration; a single SCF can take many minutes, and
the worked sweep below runs more than one. Stream the per-iteration
SCF lines with `progress=True` so the run does not look frozen, and
keep the geometry small (STO-3G, the primitive 2-atom cell) for an
interactive session. The `## Convergence and cost` section near the
end is required reading before you scale this up.

**Companion script:**
[`examples/periodic/input-bipole-nio-uks-plus-u.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-bipole-nio-uks-plus-u.py)
sweeps `U_eff in {0, 4, 7}` eV on the same NiO cell and is the
canonical recipe this page is built around.

## Why NiO, and which periodic route

NiO is the canonical DFT+U benchmark for one reason: it is the
simplest system where a pure functional is qualitatively, not just
quantitatively, wrong. The chemistry and the route both deserve a
sentence before any code.

* **The system.** Rock-salt NiO, conventional edge `a = 4.164` Å,
  taken here as the 2-atom FCC primitive cell (one Ni, one O). High-
  spin Ni²⁺ is `d⁸` (`t₂g⁶ eg²`), giving `S = 1` and `2S+1 = 3`. The
  physical ground state is antiferromagnetic (AFM-II, alternating Ni
  spin along [111]), which needs a *doubled* cell; the primitive cell
  here is ferromagnetically ordered, which is unphysical for NiO but
  keeps the SCF small enough to teach with. Treat the numbers below as
  a *recipe demonstration*, not a publication benchmark, exactly as
  the companion script's own docstring warns.

* **The route.** Open-shell periodic SCF goes through vibe-qc's
  **BIPOLE** driver (`jk_method="bipole"`), the CRYSTAL-gauge Ewald
  `J`-split path that handles RHF/UHF/RKS/UKS at multiple k-points.
  This remains the recommended route for the tight-cell magnetic-oxide
  walkthrough here. The smoother GPW route also supports Gamma-point
  open-shell UHF/UKS +U for route-audit and grid-based validation jobs.

```{warning}
**Do not reach for the other periodic paths on this system.** Two
look tempting and both are wrong for a tight 3D oxide cell:

* `run_rks_periodic` with the default `DIRECT_TRUNCATED` Coulomb sum
  can stall on a tight 3D cell.
* multi-k Gaussian density fitting with a compensating-cell
  (`use_compcell=True`) returns energies that scale with the number
  of k-points, which is a bug, not a physical trend.

The open-shell BIPOLE path in the companion script is the route that
is validated for NiO+U. Stay on it.
```

## Step 1: Build the cell and the basis

Lay out the rock-salt primitive cell as an FCC lattice with Ni at the
origin and O at the body-centre, then attach a (deliberately small)
STO-3G basis and declare the open-shell multiplicity:

```python
import numpy as np
import vibeqc as vq
from vibeqc.periodic_runner import run_periodic_job

ANG2BOHR = 1.0 / 0.529177210903

# Rock-salt NiO conventional edge a = 4.164 Å → FCC primitive cell.
a = 4.164 * ANG2BOHR
lattice = (a / 2.0) * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
atoms = [
    vq.Atom(28, [0.0, 0.0, 0.0]),                # Ni at origin
    vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),     # O at body-centre
]
# High-spin Ni²⁺ d⁸: t₂g⁶ eg² → S = 1, 2S+1 = 3 (ferromagnetic
# ordering in this primitive cell; AFM-II would need a doubled cell).
system = vq.PeriodicSystem(3, lattice, atoms, charge=0, multiplicity=3)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
```

STO-3G is far too small for transition-metal `d` orbitals; a real
study uses pob-TZVP, def2-TZVP, or cc-pVDZ-DK. It is used here only so
the SCF is tractable in a tutorial. The cell carries 36 electrons and
23 basis functions, which already takes minutes per SCF on the BIPOLE
path.

```{note}
**`U` is per `(atom, l)`, not per orbital, and the projector must be
spherical.** The Mulliken projector vibe-qc uses groups *all* AOs of a
given `l` on the chosen atom into one block. A basis whose `d` space
has two shells (a TZ basis with an outer-core `d` in addition to the
`3d`) puts both into the same projector. The projector also requires
pure spherical harmonics; a Cartesian `d`/`f` basis raises an error,
because the Cartesian `d` trace mixes in `s`-like character and breaks
the rotational invariance of the Dudarev energy.
```

## Step 2: Define the Hubbard site

A Hubbard site is an `(atom_index, l, U_eff)` triple. Construct it with
`U_ev` in electron-volts; vibe-qc converts to Hartree at the SCF
boundary for you:

```python
# U = 7 eV on Ni's 3d channel. atom_index=0 is Ni (first atom),
# l=2 is the d-channel (the 2l+1 = 5 d-AOs form the projector block).
hubbard = [vq.HubbardSite(atom_index=0, l=2, U_ev=7.0)]
```

The Dudarev energy depends only on the combination `U_eff = U - J`, so
if you have a separate Hund `J` you pass it as `J_ev=` and only the
difference matters:

```python
# Equivalent effective correction U_eff = 7.5 - 0.5 = 7.0 eV.
hubbard = [vq.HubbardSite(atom_index=0, l=2, U_ev=7.5, J_ev=0.5)]
```

`U = 7` eV is in the range usually quoted for NiO `3d`. The
`HubbardSite` exposes `U_eff_ev` and `U_eff_hartree` if you want to see
the converted value (`7.0` eV is `0.2572` Hartree).

## Step 3: Run the open-shell SCF, with and without U

Drive the SCF through `run_periodic_job` with `method="UKS"`,
`jk_method="bipole"`, and the Hubbard site passed as `dft_plus_u=`.
Running once with `dft_plus_u=None` and once with the site is the whole
experiment, the difference between the two runs *is* the effect of `U`:

```python
def nio_scf(hubbard):
    return run_periodic_job(
        system, basis,
        method="UKS", functional="lda", jk_method="bipole",
        kpoints=(1, 1, 1),               # Γ-only; (2,2,2) for a real mesh
        initial_guess="SAD",             # atomic-density guess; HCORE is far off
        max_iter=80, conv_tol_energy=1e-6,
        dft_plus_u=hubbard,              # None ⇒ plain LDA
        progress=True,                   # stream SCF lines; see "cost" below
        output="nio-uks-lda",
    )

plain = nio_scf(None)
plus_u = nio_scf([vq.HubbardSite(atom_index=0, l=2, U_ev=7.0)])

HA2EV = 27.211386245988
for tag, r in (("LDA", plain), ("LDA+U(7eV)", plus_u)):
    eps_a = np.asarray(r.mo_energies_alpha[0])   # α MOs at Γ
    eps_b = np.asarray(r.mo_energies_beta[0])    # β MOs at Γ
    na, nb = int(r.n_alpha), int(r.n_beta)
    gap_a = (eps_a[na] - eps_a[na - 1]) * HA2EV
    gap_b = (eps_b[nb] - eps_b[nb - 1]) * HA2EV
    e_u = float(getattr(r, "e_dft_plus_u", 0.0))
    print(f"{tag:>12}: E = {r.energy:.6f} Ha  "
          f"E_U = {e_u:.6f} Ha  "
          f"α-gap = {gap_a:.3f} eV  β-gap = {gap_b:.3f} eV  "
          f"(conv={r.converged}, {r.n_iter} iters)")
```

`result.e_dft_plus_u` is the Dudarev contribution that was added to
`result.energy`; it is `0` when `U` is off. The α- and β-spin Γ-point
HOMO-LUMO gaps are read straight off the per-spin MO energies as a
quick spectroscopic proxy. Smaller-spin (β) electrons see the larger
exchange splitting, so the β gap is the meaningful one to watch as `U`
opens the manifold.

On this hardware the NiO+U SCF does not finish within an interactive
session: at the production 15-bohr lattice-sum cutoff each BIPOLE
iteration costs about ten minutes, and an ionic insulator needs tens of
iterations, so a converged `U` sweep is an overnight batch job rather
than a live run. What the run shows immediately is a clean start. The
SCF banner reports `n_alpha = 19, n_beta = 17`, that is `Ms = 2` net
unpaired electrons, the high-spin Ni²⁺ `d⁸` configuration declared by
`multiplicity=3`; the driver auto-selects the ionic-insulator profile
(SAD guess, Fermi-Dirac smearing, Fock mixing); the linear-dependence
preflight passes; and iteration 1 lands at a sane `E = -1561.94 Ha`.

When the batch completes, two quantities carry the physics.
`result.e_dft_plus_u` is the Dudarev contribution added to
`result.energy`: zero with `U` off, a small positive value with `U` on.
The per-spin Gamma-point HOMO-LUMO gaps come straight off the MO
energies; the smaller-spin (β) electrons see the larger exchange
splitting, so the β gap is the one to watch as `U` opens the `d`
manifold while the total energy barely moves.

```{note}
A production NiO+U benchmark wants the doubled antiferromagnetic (AFM-II)
cell, a pob-TZVP basis, and a (2,2,2) k-mesh, run as a queued batch job
(see [the queue tutorial](vq_queue_remote_job.md)). The STO-3G
primitive-cell setup here is the smallest illustration of the API and
the cost reality, not a converged result.
```

```{warning}
**This SCF is slow enough that it may not finish in an interactive
session.** On the open-shell BIPOLE path the first SCF *iteration* of
this NiO cell costs roughly ten minutes at the driver's default
real-space cutoff (the one-time lattice-integral build alone is ~2.5
minutes), and an ionic insulator typically needs tens of iterations to
converge. A `U=0` then `U=7` sweep is therefore an hours-scale job, not
a minutes-scale one. Run it on a real machine under
`nohup python script.py > log 2>&1 &` with `progress=True` and
`tail -f log`; do not expect it to return promptly at a prompt. The
``## Convergence and cost`` section below documents what was actually
observed and the cutoff trade-off that governs it.
```

## What `U` is doing, physically

The Hubbard term is a penalty on fractional occupation of the chosen
projector. Its job is to push the `3d` occupation matrix back toward an
integer-filled, atomic-like state and, in doing so, to drive the
occupied `d` states down and the empty `d` states up, which is what
opens the gap. A few things to expect from the converged comparison:

* **`E_U` is small and positive at the converged density.** The
  Dudarev energy `(U_eff/2)(tr n - tr n²)` vanishes for integer
  eigenvalues of `n` (`0` or `1`) and is maximal at half-filling. A
  well-localized, nearly-integer `3d⁸` therefore contributes only a
  small residual penalty at convergence; the correction does most of
  its work through the *Fock potential* `V_U`, not through a large
  energy shift.

* **The gap responds more than the total energy.** `U` reorganizes the
  one-electron spectrum (`V_U` shifts occupied `d` down, empty `d` up)
  far more than it moves the total energy. This is the entire point of
  `+U`: it is a spectroscopic correction.

* **STO-3G undershoots the real gap badly.** Do not read the absolute
  gap here as physical. A minimal basis has no room for the Ni `3d`
  radial flexibility or the O `2p` covalency that set the real
  charge-transfer gap; the *trend* with `U` is the lesson, not the
  number.

## Theory

The sections below unpack why a pure functional fails on localized `d`
electrons, the rotationally-invariant Dudarev correction vibe-qc
implements, and the occupation-matrix projector it is built on.

### Self-interaction and the localized-electron problem

In exact Kohn-Sham theory the self-interaction of every electron in the
Hartree term is cancelled by the exchange-correlation functional. LDA
and GGA approximate that cancellation, and the residual self-repulsion
that survives is largest exactly where the density is most localized,
the atomic-like `3d`/`4f` manifolds. The spurious leftover repulsion
makes it energetically favorable to *spread* a localized electron over
several sites or bands, because smearing the density lowers the
self-repulsion the functional failed to cancel. The visible symptoms
are a `3d` manifold that is partially filled when it should be integer-
filled, an over-delocalized band, and a fundamental gap that is far too
small, the metal-or-tiny-gap picture LSDA gives for NiO.

### The Dudarev rotationally-invariant +U correction

DFT+U restores the missing physics with an explicit on-site Coulomb
penalty on the correlated subspace. vibe-qc implements the
**rotationally-invariant Dudarev formulation** (Dudarev *et al.* 1998),
which depends only on the single combined parameter `U_eff = U - J`:

$$
E_U = \tfrac{1}{2} \sum_{A,l} U_\text{eff}^{A,l}\,
      \operatorname{Tr}\!\Big[\, n^{A,l}\big(\mathbf{1} - n^{A,l}\big) \Big]
    = \tfrac{1}{2} \sum_{A,l} U_\text{eff}^{A,l}
      \Big(\operatorname{Tr}\, n^{A,l} - \operatorname{Tr}\,(n^{A,l})^2\Big).
$$

The bracket is a projector-like penalty: it is zero when every
eigenvalue (occupation) of `n^{A,l}` is `0` or `1`, and maximal at
half-filling. Adding it to the energy therefore rewards integer
filling of the correlated manifold and discourages the fractional,
over-delocalized occupations a pure functional drifts into. Only the
difference `U - J` enters, which is why a separate Hund `J` is optional
in the API.

The corresponding contribution to the Fock/Kohn-Sham matrix is the
derivative of `E_U` with respect to the (spin-resolved) density,

$$
V_U^{A,l} = U_\text{eff}^{A,l}\Big(\tfrac{1}{2}\mathbf{1} - n^{A,l}\Big),
$$

which pushes a less-than-half-filled state down (`n < ½`, potential
negative, stabilizing) and a more-than-half-filled state up. That
spectral re-ordering is what opens the gap.

### The occupation-matrix projector, and the overlap sandwich

The correlated subspace is defined by a Mulliken-style AO projector. For
a chosen `(atom A, angular channel l)` the `(2l+1)` AOs of that shell
form a block, and the occupation matrix is that block of the
density-overlap product:

$$
n^{A,l}_{mm'} = (S\,P\,S)_{mm'}, \qquad m, m' \in (A, l),
$$

evaluated per spin in the open-shell case. For a multi-k periodic
system the block is averaged over the mesh,
`n = Σ_k w_k Re[(S(k) P(k) S(k))_{(A,l)}]`.

In a non-orthogonal Gaussian AO basis the Fock contribution is not just
`V_U` scattered back into the AO basis; it must be **sandwiched with the
overlap on both sides**,

$$
F^\sigma_\text{AO} \mathrel{+}= S\, V_U^\sigma\, S,
$$

so that the SCF is variational with respect to the total energy. (Plane-
wave/PAW codes work in an orthonormal projector basis and show `V_U`
un-sandwiched; for Gaussian AOs the sandwich is load-bearing, without it
the analytic gradient mismatches finite difference by tens of mHa/bohr
and the SCF settles on a non-variational stationary point.) vibe-qc's
kernel computes `S P S` once, slices the per-channel block, forms `E_U`
from the trace identity, and adds the sandwiched `V_U` to the AO Fock
(or, on the multi-k path, to each `F(k)` via a real-space convolution
so DIIS, FDS, and diagonalization all see `+U` consistently).

## Convergence and cost

The honest caveats around running this come last, because they decide
whether the calculation finishes at all. Read them before scaling up.

Concretely, what this page's author measured on a multi-core box at the
driver's default real-space cutoff (15 bohr): the one-time `S`/`T`/`V`
lattice-integral build took ~2.5 minutes, and the *first* SCF iteration
took ~10 minutes, with later iterations no cheaper. At that rate the
tens of iterations an ionic insulator needs put a single `U` value into
the hour-plus range and the `U=0` then `U=7` comparison well beyond an
interactive session. Budget for an overnight batch job on a real
machine, not a number at the prompt.

* **Cost is dominated by the lattice sums, per iteration.** On the
  BIPOLE path the one-time `S`/`T`/`V` lattice-integral build at the
  default real-space cutoff is already minutes for this tight 3D cell,
  and *every* SCF iteration rebuilds a full lattice-summed Fock matrix.
  A single NiO/STO-3G UKS SCF runs many minutes on a laptop; the
  `U` sweep in the companion script is correspondingly longer. Always
  pass `progress=True` so per-iteration lines stream to stdout (set
  `VIBEQC_LIVE_LOGGING=0` to silence it for batch jobs), and thread-cap
  on a shared machine.

* **An ionic insulator is a hard SCF.** The driver auto-selects an
  ionic-insulator profile for this cell: `SAD` initial guess, integer
  occupations, and CRYSTAL-style 30% Fock mixing. Smearing is deliberately
  excluded because it can metallize a gapped ionic system. Keep the
  `SAD` guess, `HCORE` is far from the rock-salt density and converges
  poorly. If the SCF oscillates or lands at an impossible energy, that
  is a bug signal, not a tuning problem: reproduce it, check the gauge
  and the image sums, and file a regression test rather than papering
  over it with damping.

* **Do not over-tighten the real-space cutoff to buy speed.** It is the
  obvious knob to cut per-iteration cost, but it trades accuracy for it
  fast and then breaks. Measured on this cell: shrinking the cutoff from
  15 to 12 bohr drops the integral build to ~15 s and the first
  iteration to ~40 s, but the driver then prints
  `S(Γ) fold truncation 3.1e-02 ... absolute energies UNRELIABLE`,
  the lattice overlap sum is too truncated to trust the total energy.
  Shrinking further to 8 bohr makes `S(k)` lose positive-definiteness
  outright (`min eig = -5.3e-02`), and vibe-qc's linear-dependence
  preflight aborts with a `LinearDependenceError`. An AO Gram matrix is
  PSD by construction; a negative eigenvalue means the cutoff is too
  small, not that the matrix is ill-conditioned. The lesson: keep the
  cutoff at the well-converged default and pay the cost, rather than
  loosening it to finish faster.

* **For a real study, change three things.** Use a real basis (pob-TZVP
  or def2-TZVP), a doubled cell with AFM-II spin ordering, and a proper
  k-mesh (`kpoints=(2,2,2)` or denser). Each of those multiplies the
  cost; budget accordingly and run on a real machine, not in an
  interactive session.

## Next

* **A real NiO benchmark.** Swap STO-3G for pob-TZVP, double the cell
  for AFM-II ordering, and push the k-mesh to `(2,2,2)`+. That is the
  configuration that can be compared to the experimental ~4.3 eV gap
  and to published Dudarev/Cococcioni numbers.
* **Pick a better functional.** `functional="pbe"` (GGA) is a strict
  improvement over LDA here; a hybrid (`pbe0`, `b3lyp`) opens the gap
  further before any `U` and is the natural next comparison. See
  [Periodic KS-DFT](periodic_dft.md) and
  [Functional comparison](functional_comparison.md).
* **Geometry optimization with +U.** `run_periodic_job(..., optimize=True,
  dft_plus_u=[...])` relaxes the structure on the `+U` surface for the
  open-shell BIPOLE methods. See
  [Periodic geometry optimization](periodic_geometry_optimization.md).
* **Other correlated oxides.** MnO, FeO, CoO follow the same recipe
  with a different `Z`, multiplicity, and `U`; lanthanide/actinide `4f`
  systems use `l=3`.
* **Calibrating `U`.** vibe-qc takes `U` as user input; the
  linear-response prescription of Cococcioni & de Gironcoli (2005) is
  the standard way to derive it from first principles, and is the paper
  to cite alongside Dudarev when you report a `+U` number.

## References

* **Dudarev rotationally-invariant DFT+U.** S. L. Dudarev, G. A. Botton,
  S. Y. Savrasov, C. J. Humphreys, A. P. Sutton, "Electron-energy-loss
  spectra and the structural stability of nickel oxide: An LSDA+U
  study," *Phys. Rev. B* **57**, 1505 (1998),
  [doi:10.1103/PhysRevB.57.1505](https://doi.org/10.1103/PhysRevB.57.1505).
  The `U_eff = U - J` energy and `V_U` potential vibe-qc implements come
  from this paper. It fires automatically into the run's
  `.bibtex` / `.references` whenever `dft_plus_u=` is set.
* **The original LDA+U idea.** V. I. Anisimov, J. Zaanen, O. K. Andersen,
  "Band theory and Mott insulators: Hubbard U instead of Stoner I,"
  *Phys. Rev. B* **44**, 943 (1991),
  [doi:10.1103/PhysRevB.44.943](https://doi.org/10.1103/PhysRevB.44.943).
  The foundational rotationally-*non*-invariant formulation that Dudarev
  later simplified to a single `U_eff`.
* **Linear-response calculation of `U`.** M. Cococcioni, S. de Gironcoli,
  "Linear response approach to the calculation of the effective
  interaction parameters in the LDA+U method," *Phys. Rev. B* **71**,
  035105 (2005),
  [doi:10.1103/PhysRevB.71.035105](https://doi.org/10.1103/PhysRevB.71.035105).
  The standard first-principles route to the `U` value; cite it
  alongside Dudarev when reporting `+U` results. It also fires
  automatically into the citation block.

## See also

* [DFT+U (Hubbard correction)](../user_guide/dft_plus_u.md), the user-
  guide reference for the `HubbardSite` / `dft_plus_u=` surface across
  all drivers.
* [Solid-state walkthrough: LiH rocksalt with `pob-TZVP`](lih_pob_tzvp_solid_state.md),
  the periodic-SCF spine (cells, k-meshes, gaps) this page builds on.
* [Periodic KS-DFT](periodic_dft.md), choosing the underlying
  functional for a periodic run.
* [Open-shell systems](open_shell.md), the UHF/UKS spin conventions.
* [Periodic geometry optimization](periodic_geometry_optimization.md),
  `optimize=True` with `+U`.
* `python/vibeqc/dft_plus_u.py`, the Python AO-projection helpers;
  `cpp/include/vibeqc/dft_plus_u.hpp`, the C++ kernel API.
