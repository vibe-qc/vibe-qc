(semiempirical)=
# Semiempirical Methods

vibe-qc ships a self-contained semiempirical platform covering four
method families, **DFTB**, **GFN-xTB**, **NDDO**, and **INDO**, for
molecules and periodic solids. The platform is vibe-qc's own
implementation, not a wrapper around external programs
({doc}`external_codes`). The MACE machine-learning interatomic potential
is documented separately because it is an external pre-trained model, not
a semiempirical electronic-structure method ({doc}`mlip`).

```{warning}
**Production readiness varies by method.** DFTB is intended for
screening/preoptimization; GFN2-xTB is gated experimental; PM6 is available
for molecular work with its documented total-energy convention; OM2 and OM3
are validated development/prescreening routes, not external-parity production
claims; OM1 is
experimental until its analytic core-valence ECP lands; MSINDO is
reference-parity validated for its current molecular scope. See the
{ref}`status table <semiempirical-status>` and the comparative
production brief in {doc}`semiempirical_mlip_comparison`.
```

With `citations=True`, the high-level molecular and periodic runners take
their method identity, integral and SCF flags, boundary context, and runtime
electronic temperature from the concrete semiempirical route plan. The
generated reference files therefore describe the engine and boundary that
actually ran; direct model APIs return results without coordinating job
output files.

## Method families

| Family | Methods | Best for | Cost |
|--------|---------|----------|------|
| DFTB | DFTB0, SCC-DFTB, UDFTB0, USCC-DFTB | Screening, preoptimization | Fastest |
| GFN-xTB | GFN2-xTB | Organic / main-group | Fast |
| NDDO | PM6, OM1, OM2, OM3 | Development benchmarking, pre-screening | Fast |
| INDO | MSINDO | Reference-parity molecular semiempirical runs inside the supported element/spin scope | Fast |

## Quick start

### DFTB0, non-self-consistent tight-binding

```python
from vibeqc.semiempirical import DFTB0Model
from vibeqc import Molecule, Atom

mol = Molecule([
    Atom(8, [0.00, 0.00, 0.00]),
    Atom(1, [1.55, 0.90, 0.00]),
    Atom(1, [-1.55, 0.90, 0.00]),
])

model = DFTB0Model(mol)
print(f"Energy: {model.energy():.6f} Ha")
print(f"Gradient shape: {model.gradient().shape}")  # (3, 3)
```

### SCC-DFTB, self-consistent charges

```python
from vibeqc.semiempirical import SCCDFTBModel

model = SCCDFTBModel(mol, charge_mixing=0.2)
print(f"Energy: {model.energy():.6f} Ha")
```

`charge_mixing` is the maximum fraction used by the molecular SCC charge
update. At zero electronic temperature, bounded vector Aitken relaxation may
reduce and subsequently recover that fraction when consecutive Mulliken
residuals reveal charge sloshing; it never raises the fraction above the value
requested by the caller. Molecular geometry optimization also seeds each new
geometry from the preceding converged Mulliken charges, keeping the optimizer
on a continuous SCC branch.

**Finite-temperature retry in the default runner.** In pi-conjugated systems
with N or O heteroatoms (furan, pyridine, imidazole, triazine, cytosine,
pyrimidine, H2CO, HCOOH, and the S22 dimers), the frontier pi/pi* orbitals are
nearly degenerate. At zero electronic temperature the integer occupation
numbers switch discontinuously when a frontier orbital crosses the Fermi
level between SCC iterations, which makes the Mulliken charge response a
non-smooth fixed point that no mixer (Aitken, DIIS, or Broyden) can stabilize.
The standard remedy in DFTB+, tblite, and xtb is Fermi-Dirac occupation
smearing.

`run_job(method="scc_dftb")` therefore applies a bounded Mermin free-energy
retry ladder after a zero-temperature failure:

| Rung | electronic temperature | DIIS | charge mixing |
|------|------------------------|------|---------------|
| 1 | 0.001 Ha (~316 K) | off | 0.05 |
| 2 | 0.0012 Ha (~379 K) | off | 0.05 |
| 3 | 0.0015 Ha (~474 K) | off | 0.05 |

The ladder is only reached when the zero-T SCC fails, so systems whose
integer-occupation fixed point converges keep their exact zero-T energies
(bit-identical to the direct `SCCDFTBModel` result). Each rung restarts from
zero charges with a 500-iteration budget, and the reported iteration count is
the cumulative total across all attempts. Because the SCC fixed point does
not depend on the mixing fraction, the retried energies are unchanged from
the same-temperature model-level run; only the iteration path differs.

At finite temperature the reported energy is the **Mermin free energy**
`E_free = E_internal - T*S`, which is the stationary potential whose
derivative is the reported gradient. The `.out` file labels both quantities
explicitly (a "Mermin free energy (finite electronic temperature)" block with
the temperature, `E_free`, `E_internal`, and `-T*S`), and the result carries
`electronic_temperature` (0.0 for a zero-T result), the cumulative `n_iter`,
`e_internal`, and `entropy`. The direct model-level API
(`SCCDFTBModel(mol, electronic_temperature=...)`) performs no retries and
reports whatever the single requested calculation produced.

### GFN2-xTB, published parameters (experimental)

```python
from vibeqc.semiempirical import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()  # auto-fetches 86-element Grimme parameter set
model = GFN2Model(mol, params=params, warn=False)  # warn=False silences experimental gate
print(f"Energy: {model.energy():.6f} Ha")
```

```{warning}
GFN2-xTB is **gated experimental**. The H{sup}`0`/overlap deep-state
bug was fixed 2026-06-01, and molecular AES (dipole + quadrupole), the
GAM3 third-order term, and post-SCF native D4-style dispersion are now
implemented.
The classical repulsion term E_rep was restored to the total energy on
2026-08-12 (Bannwarth, Ehlert & Grimme, JCTC 2019, Eqs. 8-9): it is a
separate zeroth-order pairwise term, not something H0 parameterization
folds in, and without it the water O-H potential energy surface has no
short-range wall (the minimum collapses below 1.7 bohr). With E_rep the
water O-H minimum sits at 1.81 bohr, matching xtb 6.7.1 on the identical
geometry, and the repulsive energy itself matches xtb to ~1e-11 Ha.
A residual electronic-term gap against xtb remains (water ~15 mHa; tracked
as the GFN2 external-parity matrix workstream), so GFN2-xTB stays gated
experimental.
The remaining production gates are periodic AES image-cell multipole Ewald
terms, stricter periodic molecular-limit parity, difficult periodic/polar
fixtures, and a full external-parity matrix against `xtb`. See
{ref}`semiempirical-status`.
The bounded h-BN expansion sweep at lattice scales 1.02 through 1.06 converges
on a smooth branch when given an explicit 1000-iteration total budget; its
scale-1.06 point correctly fails under a 500-iteration cap. The seven-point
bulk-Si sweep converges within 200 iterations. Periodic GFN2 honors
`max_iter` as a hard total across its ordinary and automatic-stabilization
attempts, and the public runner forwards both that budget and `conv_tol` to
the native SCC loop. These bounded results do not relax the experimental GFN2
gate.

**SCC iteration default (2026-08-14).** The molecular GFN2 driver's default
charge iteration is a two-phase polyalgorithm: damped simple mixing tracks
the physical fixed point (the third-order term admits a spurious
over-polarized basin that history-based accelerators overshoot from the
neutral guess), halves its step when the residual stalls, and hands off to
pulay-DIIS once the simple phase is genuinely contracting. A DIIS phase that
stops improving is abandoned and the driver finishes from its best charge
state under simple mixing. Every branch converges to the same fixed point,
so energies are unchanged; on the five-system wall matrix (H2O, CH4, NH3,
adenine, naphthalene) the iteration counts dropped 2.3-3.9x (adenine
1984 -> 514), with the converged energies pinned to the previous values.
Explicitly requesting `scc_mixer="diis"` or `"broyden"` bypasses the
polyalgorithm and uses the unguarded accelerator (the historical behaviour).
Newton mixing is implemented only by the nontrivial GFN2-SECCM engine. The
direct molecular native boundary rejects `SCCMixer.Newton` rather than
silently executing and relabeling the default polyalgorithm.
Separately, the graphene periodic-PM6 sweep now has finite image exchange and
converges. Diamond-Si periodic PM6 now uses one shared gamma kernel for electronic and
core image monopoles and crosses its 15-bohr shell boundary smoothly. The
h-BN PM6 physics gap remains open; Bloch-periodic OMx is gated as described
below.
The post-SCF native D4 term is included for H, He, B, C, N, O, F, and Ne;
outside that reference-data set GFN2 returns zero D4 and emits
`GFN2D4UnsupportedWarning`.

For a hands-on walkthrough covering single-point energies, spin
polarization, geometry optimisation, frequencies, PES scans, molecular
dynamics, and periodic calculations, see the
{doc}`GFN2-xTB workshop tutorial <../tutorial/gfn2_xtb_workshop>`.
```

**What the reported GFN2 energy contains, and how to compare it to `xtb`.**
`GFN2Model.energy()`, `run_semiempirical("gfn2_xtb", ...)` and
`run_job(method="gfn2_xtb")` all return the **native SCC energy plus the
post-SCF D4 dispersion correction**. The native binding
`vibeqc._vibeqc_core.semiempirical.xtb.run_gfn2_xtb(...).energy` returns the
SCC energy **alone**: D4 is an a-posteriori term in GFN2 (Bannwarth, Ehlert
& Grimme, *J. Chem. Theory Comput.* **15**, 1652-1671 (2019),
[doi:10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)),
so the two differ by exactly that correction.

`xtb`'s own total energy already includes dispersion (its output prints it as
a sub-term of the SCC energy, `-> dispersion`), so **an `xtb` parity
comparison is only like-for-like against the D4-corrected value.** The gap is
not academic. On the archived THIEL adenine frame vibe-qc's SCC energy is
−27.508108468 Ha and the reported total is −27.521386606 Ha, 13.278 mHa
apart, against `xtb` 6.7.0's −27.499575435 Eh (whose own `-> dispersion` is
−13.538 mHa). Quoting the SCC energy where the total belongs turns a
−21.811 mHa residual into a −8.5 mHa one, 133x the 0.1 mHa parity gate.
Guard: `tests/test_gfn2_xtb.py::TestGFN2PublicEnergyIsDispersionCorrected`.

The `.system` manifest records the parameter provenance a GFN2 number
depends on, including `gfn2_cache_sha256` (the digest of the fetched
parameter cache, `gfn2_xtb_params.toml`) and `gfn2_d4_refdata_sha256`, so a
reported energy can be tied to the exact parameters that produced it.

#### Where the parameter cache lives, and seeding it for offline hosts

The cache root follows the same precedence as every other vibe-qc cache:

1. `$VIBEQC_GFN2_CACHE_DIR`, if set (used verbatim);
2. `$XDG_CACHE_HOME/vibeqc/`, if set;
3. `~/.cache/vibeqc/` otherwise.

The parameters are fetched on first use (they are LGPL-3.0 and are not
bundled, see [`license.md`](../license.md)), so a **compute node with no
outbound network fails closed** with `GFN2ParameterUnavailableError` unless a
cache is already present. On a cluster, populate one cache from a host that
does have network access and point the jobs at it:

```sh
# once, on a host with network access
python -c "from vibeqc.semiempirical.methods.gfn2_params import \
load_gfn2_params; load_gfn2_params()"

# then make it visible to the compute nodes
rsync -a ~/.cache/vibeqc/ <shared-path>/vibeqc/
export VIBEQC_GFN2_CACHE_DIR=<shared-path>/vibeqc
```

The cache is SHA-pinned and lineage-checked on load, so a truncated or
substituted file fails closed rather than producing silently different
numbers.

### PM6, NDDO with published parameters

```python
from vibeqc.semiempirical import PM6Model

model = PM6Model(mol)
print(f"Energy: {model.energy():.6f} Ha")
```

The high-level closed-shell `run_job(method="pm6")` route first attempts the
ordinary zero-temperature SCF. If hard occupations cycle across
near-degenerate fragment orbitals, it uses a bounded finite-temperature
occupation homotopy and then cools back to an idempotent zero-temperature
density. Only the cooled PM6 energy is accepted; the finite-temperature
intermediate is never reported as a successful result.

Gamma-periodic PM6 instead uses Pulay extrapolation of the physical
commutator residual. Convergence requires both the density change and
`[F, D]` residual to meet `conv_tol`; this prevents a repeated extrapolated
Fock matrix from being accepted when its density is not stationary under the
physical Fock operator. It uses the shared guarded Pulay history, which
shortens a linearly dependent history before extrapolating and falls back to
the current physical Fock matrix when no stable history remains. This avoids
the cancellation-dominated extrapolation that previously made the archived
Pa-3 dry-ice scale-0.96 point fail after 1200 iterations and made its
scale-0.98 neighbor unusually slow. The archived proton-ordered Ice-Ih and
bounded dry-ice expansion slices now converge on smooth solver branches.

Production Gamma-periodic PM6 lattice sums carry the exact s/p-s/p NDDO
multipole tensor of the molecular driver in every image cell for the Coulomb
and core-attraction blocks and in the zero cell for the interatomic exchange
block; non-zero images keep the overlap-damped monopole exchange model,
because a Gamma-only density has no cell resolution and the undamped 1/R
exchange tail of the tensor diverges with the cutoff (issue #419).
Electrostatic lattice sums are truncated per image cell, so an admitted image
always enters as a neutral group; the pair-distance inventory decides which
cells enter. A neutral molecule in a wide box therefore approaches the
molecular PM6 energy monotonically (CO: 179, 96, 28, 17 microhartree at
12, 14, 16, 20 bohr). The SCF seeds with the spherically averaged
neutral-atom valence density rather than a zero density, which selects the
pi-occupied branch on the expanded graphene cell.

Periodic PM6 remains experimental until its energies receive independent
external-reference validation. In particular, dry-ice energies remain
sensitive to direct-space image-shell cutoff; solver convergence does not
establish a physical molecular-crystal equation of state. The periodic
diatomic core branch also cancels the neutral-pair monopole retained by the
electronic decomposition. A closed-shell fcc-Ar regression pins that
cancellation and the repulsive compressed branch. Bare PM6 has no validated
rare-gas dispersion model here, however, so its dissociative Ar curve is not
a physical rare-gas equation of state.

For elements such as silicon, whose electronic block uses the multi-term NDDO
gamma expansion, the neutral core baseline uses that identical kernel rather
than the separate PM6 short-range core kernel. Damping and Gaussian pair terms
remain corrections on top. This keeps a neutral image shell neutral before
truncation: the diamond-Si 15-bohr sphere changes from 55 to 43 cells between
lattice scales 1.02 and 1.04, but the total-energy step is about `0.00195 Ha`
rather than the former `1.01917 Ha`. This is a bounded implementation
regression, not independent MOPAC validation.

### OMx, orthogonalization-corrected NDDO

```python
from vibeqc.semiempirical import OMxModel

model = OMxModel(mol, variant="om2")  # "om1", "om2", or "om3"
print(f"Energy: {model.energy():.6f} Ha")
```

### Through `run_job`

All seven methods are available via {func}`vibeqc.runner.run_job`:

```python
from vibeqc import run_job

run_job(mol, method="dftb0", optimize=True, output="h2o_dftb0")
run_job(mol, method="pm6", output="h2o_pm6")
run_job(mol, method="gfn2_xtb", output="h2o_gfn2")   # emits experimental warning
```

`run_job` treats Molden and population sidecars independently. DFTB0,
SCC-DFTB, and GFN2-xTB emit `.population.txt` and `.population.json` by
default from the engine's native net atomic Mulliken charges. DFTB0 forms
these charges from its final one-shot density and overlap; UDFTB0 uses the
sum of its alpha and beta densities. Analyses that require the Gaussian-AO
property stack are present as explicit `unsupported:` entries. Molden remains
unavailable for every semiempirical route because the current writer
serializes Gaussian GTO shells, not the methods' minimal Slater-orbital bases.
PM6/OMx and MSINDO population sidecars remain unavailable until their native
result adapters expose a validated atomic-population contract. An explicit
unsupported `write_molden_file=True` or
`write_population_file=True` fails before calculation; `None` (the default)
selects only the sidecars the chosen route can produce truthfully.

Every molecular semiempirical route refuses to report a total energy of
exactly `0.0` Ha, or a non-finite one, raising `SemiempiricalEnergyError`
(a `RuntimeError` subclass). A semiempirical total is an electronic term plus
a core-core repulsion or repulsive term, so for a system carrying electrons
neither value is reachable: both mean the calculation stopped before it
produced a result. The check is bit-exact rather than a tolerance band, so a
legitimately small total energy is still reported, and it is independent of
`converged`: an unconverged run carrying a real energy is still returned, and
callers keep gating on `result.converged` as before.

### DFTB0-SECCM (experimental)

The first non-MSINDO SECCM adapter is available through the direct
`run_dftb0_seccm` API. It consumes a separately built and finite-group-bound
SECCM topology; it does not reinterpret the ordinary Gamma-periodic DFTB0
cutoff domain. The gate accepts neutral, closed-shell, insulating cyclic clusters in one,
two, or three dimensions. Element scope is the explicit repulsive-pair table
of the built-in in-house screening set rather than a fixed element list, so it
covers H, C, N, O, F, P, S and Cl and every one of their pairs; a pair without
an explicit repulsive is refused by name. DFTB0 carries no SCC, so neither the
even-replica charge-map pathology nor the Klopman-Ohno thermodynamic-limit
defect reaches this route, and both replica parities are supported:

```python
import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_dftb0_seccm
from vibeqc.semiempirical.seccm import (
    bind_finite_group,
    build_seccm_topology,
)

a = np.array([8.0, 0.0, 0.0])
coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
topology = build_seccm_topology(
    coords, [a], length_unit="bohr", geometry_quantum=1.0e-10
)
topology = bind_finite_group(
    topology,
    primitive_vectors=[a],
    replicas=(1, 1, 1),
    geometry_tolerance=1.0e-9,
    length_unit="bohr",
)
mol = Molecule([Atom(1, xyz.tolist()) for xyz in coords])
result = run_dftb0_seccm(mol, topology, compute_gradient=True)
```

The reported `energy` is per primitive cell. Its complete T3a closure is

```text
energy = electronic_energy + repulsive_energy
       + long_range_energy + dispersion_energy + specific_energy
```

where the last three terms are explicitly zero. The adapter evaluates the
full finite cyclic cluster, so a valid group-only defect topology is allowed;
translation-orbit reduction is not used. With `compute_gradient=True`,
`result.gradient` is the analytic derivative of the same per-cell electronic
and repulsive closure over the frozen record set. It does not differentiate a
topology switch or the cyclic translations.

Periodic reaction paths can select this adapter explicitly with
`run_neb(..., method="dftb0", seccm_topology=topology)`. The endpoint cell must
match the topology translations. A topology containing exact ownership ties
also requires a positive `seccm_max_tie_score_excursion` trust bound. SCC
charges, open shells, charged cells, stress, smearing, broader parameter sets,
and two- or three-dimensional public calls fail closed.

### SCC-DFTB-SECCM (experimental)

SCC-DFTB-SECCM runs the self-consistent-charge tight-binding Hamiltonian
inside the same frozen Wigner-Seitz supercell construction. One Wigner-Seitz
weighted S and H0 are assembled for the full cyclic cluster, the supercell
Mulliken charge fluctuations iterate to self-consistency, and the total
energy is reported per primitive cell:

```text
A = (tr(D H0) - T*S + 1/2 dq . gamma . dq + E_mad + E_rep)
    / group_order
```

The public `energy` and `free_energy` fields both report this Mermin free
energy. `E_mad` is zero unless the opt-in embedding below is active, and the
entropy term is zero on the default hard-Aufbau path.

Long-range image electrostatics can join through an opt-in Madelung/Ewald
embedding (`madelung=True`): 1-D background-corrected wire Ewald and 2-D
Parry/Heyes Ewald, both sharing the CCM SMADEL short-range subtraction
convention. The Madelung potential joins the ordinary SCC potential in the
same overlap-weighted variational Hamiltonian. Neutral ionic cells can need
this embedding even though their net cell charge is zero. The current 3-D
Ohno correction has no thermodynamic limit, so 3-D embedded requests fail
closed. The unembedded 3-D route currently requires an odd replica count on
all three axes. Noga et al. (1999) adopt odd counts as a natural central-cell
convention, not a mathematical requirement, while Peintinger and Bredow
(2014) define fractional WSSC boundary ownership. In vibe-qc, the observed
even branch diverges instead of approaching the dense-k comparator, and no
route-specific Nyquist ownership rule has been validated. Uniform and
mixed-parity 3-D requests therefore fail before the native calculation starts.

Neutral MgO(100) slab thickness is a guarded model-domain limitation, not a
supported production study. The regression uses a genuine neutral Tasker-I B1
control: every (100) plane has equal Mg/O counts, adjacent planes swap
sublattices, and its two-plane 3-D extension has six unlike nearest neighbours.
It does not use the single-species polar-plane stress fixture tracked by issue
#471.

At the fixed 4.212 Angstrom geometry, the public two-layer unembedded route
converges with a 0.00428 Ha finite-torus frontier gap, while the identical
route with 2-D Madelung embedding fails closed even with DIIS and a 0.005 Ha
electronic temperature. Increasing thickness also shrinks the unembedded
frontier; the current genuine-B1 L3 and L4 controls also fail to converge
within the tested iteration budgets and mixers. These observations do not
establish that the gap causes the failure, and the finite-torus value is not a
physical MgO band gap.

The fixed-geometry wall is separate from the parameter-set limitation reported
by ``DFTB0RepulsivePlaceholderWarning``. O-O, Mg-O and Mg-Mg all carry the
A/R{sup}`−12` placeholder rather than a fitted repulsive, and those
placeholders are added only after the SCC iteration, so they cannot change its
charges, gap, iterations, or convergence. It is the same fixed-geometry
additive scalar in both arms, so the
comparison is only an internal operator diagnostic. The built-in electronic
and repulsive parameters are in-house screening values, not a
literature-validated MgO DFTB set; no energy, structure, surface energy, EOS
result, charge, or band gap from this control is validated MgO chemistry.

Equation review found no sign, factor, or double-counting defect in the finite
2-D Madelung construction. The separate 3-D gamma thermodynamic-limit defect
tracked by issues #211, #425, and #444 remains outside this result. Damping,
smearing, disabled embedding, fallback SCF, or an invented repulsive spline
must not be used to relabel a refused MgO slab as a converged result.

```python
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_scc_dftb_seccm
from vibeqc.semiempirical.seccm import (
    bind_finite_group,
    build_seccm_topology,
)

# One-dimensional H-Li chain, three primitive cells (neutral).
a = 4.1  # bohr
sites = [(0.17, 0.31, 0.0), (1.39, -0.22, 0.0)]
coords = []
for cell in range(3):
    for site in sites:
        coords.append((site[0] + cell * a, site[1], site[2]))
topology = build_seccm_topology(
    coords, [(3 * a, 0.0, 0.0)], length_unit="bohr",
    geometry_quantum=1.0e-10,
)
topology = bind_finite_group(
    topology,
    primitive_vectors=[(a, 0.0, 0.0)],
    replicas=(3, 1, 1),
    geometry_tolerance=1.0e-9,
    length_unit="bohr",
)
mol = Molecule(
    [Atom(1 if k % 2 == 0 else 3, c) for k, c in enumerate(coords)]
)
result = run_scc_dftb_seccm(mol, topology)
print(result.energy)  # per primitive cell
```

The molecular limit (one replica, large cyclic translations) reproduces the
molecular SCC-DFTB driver bit-for-bit. A one-dimensional H-Li chain
converges towards the independent k-point dense limit, reaching the mHa
range at 32 replicas. Two- and three-dimensional cells are finite and
lattice-symmetry invariant.

**Charged cells.** Pass `charge` on the `Molecule` and set `madelung=True`.
The embedding couples the Mulliken fluctuations through the overlap-weighted
H^SCC operator, adds their classical self-energy, and conserves total charge
to the convergence tolerance. Charged one-dimensional polar chains
converge for even replica counts at T = 0 under DIIS. Odd replica counts sit
at a degenerate frontier (gap ~1e-15) whose hard-Aufbau response no mixer
contracts; pass a small `electronic_temperature` (for example 0.005 Ha) to
smooth the occupation and converge. The reported energy is then the Mermin
free energy A = E - T*S (`free_energy` is an alias, `entropy` and
`smearing_temperature` are recorded). Such a run is admitted with
`gap_guard_waived = True` and the applied epsilon in
`finite_torus_gap_tolerance` (1e-8 Ha by default): finite temperature
resolves the occupation, but the frontier is still numerically degenerate,
so the row must be screened on `gap_guard_waived` rather than on
`homo_lumo_gap > 0`. At T = 0 the same frontier is rejected outright.

**Screening a smeared row: `gap_guard_waived` is not sufficient on its own.**
That flag compares the gap to an *absolute* epsilon in Hartree, while the
scale that resolves an occupation under smearing is kT. A frontier can
therefore sit far above the epsilon and still be completely unresolved: the
Mg/O rocksalt(100) three-layer slab at `electronic_temperature = 0.005` has a
gap of 2.4e-8 Ha -- 2.4x the epsilon, but 5e-6 of the smearing width -- and
its HOMO and LUMO each hold 1.337 electrons. So `run_scc_dftb_seccm` also
records the occupation it actually applied:

| field | meaning |
|---|---|
| `occupations` | the per-MO occupation the accepted density was built from |
| `aufbau_occupation_deviation` | `max_i \|f_i - f_i^Aufbau\|`; exactly `0.0` at T = 0 |
| `aufbau_occupation` | the deviation is within `aufbau_occupation_tolerance` |
| `aufbau_occupation_tolerance` | the roundoff epsilon actually applied (1e-8) |

A row with `aufbau_occupation = False` is a stationary point of the
fractional-occupation functional, which differs from the Aufbau one by exactly
the `-T*S` term the route subtracts (Weinert and Davenport, *Phys. Rev. B*
**45**, 13709 (1992), Eqs. 8 and 10'). Screen on `aufbau_occupation` before
comparing a smeared energy against a T = 0 reference. A deviation at or above
`1.0` is stronger still: the chemical potential has reached the Aufbau LUMO,
so the row is metallic rather than merely thermally broadened.

**Mulliken charges past the formal valence count are not a failure signal.**
`charges` are gross Mulliken charges, `dq_A = n_val(A) - sum_{mu in A}(DS)_mu,mu`.
Gross populations are not confined to `[0, 2 n_shell]`: Mulliken already
recorded (*J. Chem. Phys.* **23**, 1833 (1955), Sec. 3, p. 1835) that small
negative values and slight excesses over 2.00 per closed sub-shell occur, and
that the invariant is the total, "necessarily an integer". vibe-qc therefore
does not gate on a raw charge range. When a small excursion appears, check
`aufbau_occupation` first: on the slab above it is a symptom of the
unresolved frontier, and it disappears when the occupation changes.

**Charge iteration.** The default is the same vector-Aitken relaxation as the
molecular driver. `use_diis=True` switches to damped Pulay-DIIS on the charge
vector, and `use_broyden=True` selects the modified Broyden quasi-Newton
mixer. The two accelerators are mutually exclusive and both reject
unphysical extrapolations back onto a conservative damped step.

**Gradients.** `compute_gradient=True` returns the analytic fixed-topology
nuclear gradient of the converged per-cell energy:

```text
dE/dR = (M . dS/dR + repulsive derivative + 1/2 dq . dgamma/dR . dq)
      / group_order,  M = -W + D . 1/2 (kappa hbar - V_A - V_B)
```

The image record set stays frozen while its displacements track the geometry
(the `topology.rebuild_displacements` convention). The molecular limit is
bit-identical to the molecular SCC-DFTB gradient, and one-dimensional chains
match fixed-topology finite differences to 2e-9 Ha/bohr.

Embedded cells (`madelung=True`) also report
`gradient_method="analytic"`. The Madelung potential is part of the same
variational Mulliken-charge operator as the ordinary SCC potential, so the
self-consistent charge response cancels by stationarity. The force adds the
explicit Madelung derivative 1/2 dq . dM/dR . dq (the one-dimensional wire
kernel derivative or the validated 2-D CCM deposit). The N=3 smeared chain
and the N=2 zero-temperature chain match re-converged finite differences of
the energy to the microhartree/bohr scale.

### PM6-SECCM, OMx-SECCM, and GFN2-SECCM (experimental)

All direct SECCM adapters accept a valid one-atom, one-replica topology even
though its directed pair-record set is empty; the one-center Hamiltonian terms
remain well-defined and reach the method's ordinary acceptance gates.

PM6-SECCM assembles the supercell NDDO Fock with Wigner-Seitz weighted
two-center terms and matches the molecular PM6 driver bit-for-bit in the
molecular limit for its implemented s/p element envelope. Full
heavy-heavy two-electron tensor parity with the defining PM6 implementation
remains open, so the adapter is not a quantitative PM6 reference. Bundled PM6
records that require d orbitals fail closed: the complete PM6 spd Hamiltonian
has not yet landed in the SECCM adapter, and treating those five d functions
as p channels would return a different model. Madelung embedding is also not
implemented, so neutral ionic crystals requiring long-range image
electrostatics are outside the current PM6-SECCM scope. Its gradients
(`compute_gradient=True`) use central
differences of the converged per-cell energy over the frozen topology
displacements (`gradient_method="finite_difference"`, `gradient_fd_step`
controls the displacement). No analytic derivative of the current PM6
Hamiltonian is exposed. OMx-SECCM builds the same WS-weighted supercell Fock for
OM2/OM3: molecular one-center blocks plus directed two-center terms
(penetration-corrected core attraction, resonance, two-electron
Coulomb/exchange, core-core repulsion, and the one-center VORT
corrections) accumulated through the WS records with their fractional
weights, matching `run_omx_v2` bit-for-bit in the molecular limit. Its
three-center terms (the G1/G2 VORT corrections and the OM2/OM3 ECP)
follow the Peintinger-Bredow 2014 image weighting: the production eq-13
scheme over the union WSSC(MN) by default
(`three_center_weighting="peintinger_eq13"`), with Janetzko's original
eq-10 scheme available for comparison
(`three_center_weighting="janetzko_eq10"`). The selected weighting convention
is reported on the result and routed to its primary citation. OM1-SECCM fails
closed because its defining analytic core-valence ECP is absent.

OM2/OM3-SECCM does not yet contain the self-consistent Madelung/Ewald terms
that the primary CCM papers require for ionic crystals. Because partial
ionicity is an SCF result rather than a sound input-only classification,
every OMx-SECCM topology with multiple replicas, wrapped image records, or
fractional image ownership now fails closed by default. The exact
one-replica, zero-image, full-weight molecular limit remains available.
Developers testing image weighting or finite-group mechanics may pass
`allow_truncated_electrostatics=True`; this is an explicit acknowledgement
that the long-range contribution is omitted, and the resulting number is not
a quantitative solid-state prediction. Both
`result.truncated_electrostatics_acknowledged` and the same immutable field on
`result.route_plan` record that condition while
`result.route_plan.electrostatics_kernel` remains `"none"`.

GFN2-SECCM assembles the
WS-weighted supercell GFN2-xTB SCC the same way: one-center S/H0 blocks
are the molecular ones, every directed two-center coupling (the H0 image
block, the shell-resolved gamma, the shell-resolved multipole AES
potential, the pair repulsion) is accumulated through the WS records with
their fractional weights, and the per-cell energy is the finite-cluster
total divided by the group order (`total_cyclic_energy` and the `cyclic_*`
decomposition are reported). The SCC loop mirrors the molecular driver
(damped simple mixing plus the stabilization retries); D4 dispersion and
the experimental faithful AES are not part of the adapter. At T = 0, the
molecular limit (1x1x1 cyclic cluster) still reproduces the validated
molecular `run_gfn2_xtb` driver bit-for-bit. At finite temperature it reports
the identical molecular state and internal components, but maps the SECCM
`energy` field to the variational Mermin free energy rather than the molecular
driver's historical internal-energy field. All three are
neutral and closed-shell, with the same `run_<method>_seccm` direct-API
shape.

The coordination-dependent H0 self-energy uses one atom-local
Bannwarth-Ehlert-Grimme Eq. 18 coordination vector assembled from the directed
WS records at their physical image distances with their ownership weights, the
same inventory the cyclic H0 assembly uses, so retyping a torus site by a
lattice translation cannot move H0 (issue 354, fixed).

**The default multipole AES moment integrals still depend on the coordinate
representative typed for a torus site (issue 348), and `aes_faithful=True` is
the fix.** With the default kernel the AES is the only remaining
representative-dependent term in the route: on a 4-unit polar HF chain three
representatives of one crystal give -5.230514880891, -5.230364176724 and
-5.230359290006 Ha, a 1.6e-4 Ha spread, while `include_aes=False` makes all
three agree bit-for-bit. `aes_faithful=True` builds the multipole integrals
over the same Wigner-Seitz records as the overlap and H0 and evaluates the
Bannwarth 2019 Eq. 25 energy and Eqs. 39-44 Fock on the resulting
atom-resolved cumulative moments; all three representatives then give
-5.233747472334 Ha. The two paths cannot be blended, because the ad-hoc
kernel consumes shell-resolved moments and a pair inventory produces
atom-resolved ones. The flag defaults off only so that molecular-limit bit
parity with the molecular driver, whose own `aes_faithful` default is off,
keeps its meaning; the two defaults move together.
`include_aes=False` is an explicit channel-isolation control for validation,
not a promoted quantitative Hamiltonian. Periodic *Gamma* GFN2 no longer
shares this defect: its pair-complete one-electron support landed under issue
316, and issues 296/338 replaced the home-cell shell gamma and the ad-hoc
damped multipole kernel with a translation-covariant Ewald-split lattice sum
and the Bannwarth 2019 model on image-resolved moments (see "Periodic
systems" below). It stays experimental for the separate reason that external
xtb parity is not closed. The multi-k functional keeps its complex-phase and
AES defects, tracked as issue 351.

GFN2-SECCM returns a schema-versioned `result.hamiltonian_identity`; the same
record is available as `result.route_plan.gfn2_hamiltonian_identity`. It
records the requested electrostatics family and electronic temperature
separately from the kernel and temperature that actually executed, plus
`include_aes`, the two Madelung operator modifiers, the Ewald molecular
on-site selector, and molecular delegation. This distinction matters when a
one-cell Ewald request delegates to the exact molecular Hamiltonian or a
T = 0 request converges on the bounded finite-T retry.
`result.hamiltonian_identity.to_dict()` is the JSON-safe provenance payload
for this standalone direct API, and `from_dict()` restores it exactly.

Solver and acceptance provenance is separate from Hamiltonian identity.
`result.run_controls` (also
`result.route_plan.gfn2_run_controls`) records the parameter set, total SCC
budget, charge threshold and mixing control, requested mixer, the fixed
finite-torus gap tolerance, the requested and resolved Parry K=0 policy, and
an immutable count plus SHA-256 identity for any supplied restart. Automatic
stabilization applies a supplied restart only to the primary attempt; every
fallback begins from neutral charges with the Simple mixer. The ordered
`result.attempts` records the executed solver, rung controls, restart source,
exit reason, and immutable residual trace for each attempt.
`result.n_iter`, `result.scc_max_change_trace`, and the per-attempt counts and
traces reconcile exactly. `result.selected_attempt_index` identifies the
accepted final attempt. A bounded failure raises
`GFN2SECCMConvergenceError`, a `RuntimeError` subclass whose `result` retains
the same complete frozen ledger for diagnosis.

**Comparing SCC states (issue 421).** `converged` and `physical_basin`
do not establish that two scan points reached the same electronic branch.
The magnitude-based basin checks cannot identify every redistribution of
charge between equivalent sites. `translation_symmetry_charge_spread`
reports the largest charge spread between same-species sites related by a
primitive translation. It retains nonperiodic coordinates, so separate slab
layers or parallel chains are separate orbits. It does not test equivalence
under point-group operations. Absent or nonfinite charges give NaN; finite
partial charges can be diagnosed even when the run did not converge.

For a caller-ordered size or geometry scan with the same composition per
primitive cell, use
`vibeqc.semiempirical.compare_gfn2_seccm_states(previous, current)`.
The immutable result reports signed changes (current minus previous) in
atomic charge RMS, frontier gap, SCC energy, free energy, translation-orbit
spread, and electronic temperature. Charge measures are in electrons;
energies and temperatures are in Hartree. Energies are already per primitive
cell and receive no second division by cluster size. `same_hamiltonian`
compares parameter and Hamiltonian-option provenance, including resolved
temperature, so a finite-temperature retry or a kernel change is visible.
It does not establish identical geometry or composition. Failed or nonfinite
states cannot supply a valid comparison and raise `ValueError`.

These diagnostics apply no new threshold, alter no acceptance verdict, and
do not select a preferred branch. A small scalar difference does not prove
state continuity, and a nonzero difference does not prove an unphysical
state. The remaining AES and 3-D gamma defects (issues 348 and 444) still
limit quantitative interpretation of the affected routes.

**Metallic supercells.** GFN2-SECCM exposes an opt-in
`electronic_temperature` (Hartree, default 0 = hard Aufbau). Metallic
supercells (bulk Cu/Pd/Ag, doped semiconductors) have a degenerate
frontier whose T = 0 occupation can switch branches between neighbouring
geometries, making the energy surface discontinuous; a small temperature
(for example 0.005 Ha) converges branch-free and the reported energy is
the Mermin free energy A = E - T*S (`free_energy` is an alias, `entropy`
and `smearing_temperature` are recorded). A smeared run still measures and
records its frontier gap in `homo_lumo_gap`; what finite temperature
changes is only the *verdict*. At T = 0 a frontier at or below the guard
epsilon (`run_controls.finite_torus_gap_tolerance`, 1e-8 Ha by default) is
rejected, because its Aufbau occupation is ambiguous. At T > 0 the
Fermi-Dirac occupation resolves that ambiguity, so the state is admitted
with `gap_guard_waived = True` recorded on the result. A waived row is
admissible but is **not** a gapped row -- its energy is a free energy on a
numerically degenerate frontier -- so screen on `gap_guard_waived`, never on
`homo_lumo_gap > 0` alone. Bulk-metal supercell stability is tracked in the bug
tracker (compressed-volume branch switching in fcc Cu is issue 130); run
metals at even-atom-count supercells (Cu and Ag have 11 valence electrons,
so odd cells fail the closed-shell electron-count check) and compare
neighbouring supercell sizes before trusting a lattice constant.

**Neutral ionic crystals.** GFN2-SECCM exposes an opt-in
`madelung=True` embedding that restores the Coulombic image-sum tail
beyond the Wigner-Seitz cell (MSINDO CCM convention, SMADEL short-range
subtraction; 1-D background-corrected wire Ewald, 2-D Parry/Heyes Ewald,
3-D Ewald). The WS-truncated shell gamma carries the damped short-range
Klopman-Ohno part, and without the tail the unembedded curve overbinds
ionic crystals and loses its lattice minimum (MgO 2x2x2 decreases
monotonically to below 3.9 A unembedded). With the embedding the 2x2x2
MgO curve has a minimum near 4.30 A (+2% vs experiment), the Mulliken
charges relax to the xtb scale, and the self-energy is reported as
`e_madelung` / `cyclic_madelung_energy`. The embedding is intended for
neutral ionic cells; metallic supercells do not benefit from it (their
over-polarization is intra-atomic shell transfer, invisible to atomic
charges). Strongly ionic cells like corundum sit beyond the embedding's
stable regime (the ionic-mode feedback of the diagonal-deposit
convention exceeds the shell hardness, so the SCC runs away and fails
closed).

**Self-consistent periodic shell gamma (`ewald_gamma`).**
`ewald_gamma=True` supports 1-D wires and 2-D slabs, and is mutually
exclusive with `madelung`. It combines the bare-Coulomb Ewald kernel with
the WS-folded Klopman-Ohno-minus-Coulomb correction and on-site hardness.
**Non-molecular 3-D cyclic topologies fail closed (issue #444)** before
the divergent kernel is assembled. A finite WS sum is not a thermodynamic
limit: the pair-dependent remainder decays as `R^-3`, whose neutral-cell
coefficient need not cancel in three dimensions. Exact zero-image molecular
delegation is preserved, including 3-D topologies; group order one alone
is not a molecular test. No Elstner mapping, changed finite-distance GFN2
energetics, or replacement scientific default is supplied by this containment.
See [the derivation and fixed-charge controls](../design_seccm_gfn2_long_range_gamma.md).

Historical MgO/corundum bulk energies and the named #130 Cu energy/size
assertions remain evidence, not reachable 3-D `ewald_gamma` production
expectations. The corresponding Cu recipes now pin the explicit rejection;
small charges or a converged SCC solve do not establish a thermodynamic
limit. Molecular, 1-D/2-D, and non-`ewald_gamma` paths are unchanged.
In 2-D
the two-layer synthetic aligned-plane stress cell converges and matches the
`madelung=True` construction on the same cell. That fixture is an artificial
two-coordinate P4/mmm stack, not B1 rocksalt(100), so its former physical
surface conclusions are withdrawn. Its four-layer version cycles under the
tested T = 0 simple-mixing route and fails closed; this measured
nonconvergence is not a proof that no exact fixed point exists. Earlier
trees could land fcc Cu in a charge-density-wave basin associated with the
d-first transition-metal parameter rotation fixed with issue 43; those
historical solver results do not override #444's route rejection.
`include_aes=False` is an
experimental validation knob that removes the WS-folded anisotropic
second-order channel for SCC instability isolation; its energy misses a
physical term and must not be used for production numbers.

`scc_mixer` selects `"simple"` (default), `"broyden"`, `"diis"`,
`"broyden_eyert"` (the tblite `broyden.f90` scheme the xtb binary
uses; `charge_mixing` defaults to xtb's bromix 0.4 for it), or
`"newton"`. The last option is a line-searched frozen-lagged-multipole
chord/quasi-Newton step, not a full Newton solve; combined with finite-T
occupation smoothing it converges the four-layer synthetic stress map used
for numerical testing. `ewald_gamma_k0_global` is retained only as a
compatibility no-op: both values use the complete pairwise Parry/de Leeuw
K=0 kernel. The result carries `scc_max_change_trace` (per-iteration max
|dq_new - dq|), which exposes sustained limit cycles without interpreting
them as a proof of fixed-point nonexistence.

`initial_shell_charges` optionally supplies the GFN2-SECCM SCC starting
state. A nonempty value must be a one-dimensional, finite vector with exactly
`n_shells` entries. It is rejected on the molecular-delegation route because
the molecular GFN2 driver has no shell-charge restart field. Invalid restart
shape and finiteness are rejected before topology processing; exact length
and route compatibility are rejected at the native boundary before topology
SVD or any SCC/eigensolver work begins.

## Periodic systems

Periodic support is route-specific. DFTB0, SCC-DFTB, GFN2-xTB, and PM6 have
Gamma-point periodic energy routes; their periodic gradients and stress remain
experimental and route-labeled as analytic or finite-difference stopgaps in
the status table below. The analytic DFTB0, UDFTB0, SCC-DFTB, USCC-DFTB, and
GFN2 gradient and stress routes reuse the effective lattice cutoff (and
DFTB0's `gamma_only_0` mode) recorded on their SCF result, so a nondefault
cutoff differentiates the same truncated lattice sum the energy used; results
without that provenance fail closed.

The **analytic Gamma-point GFN2-xTB gradient and stress differentiate the
shipped energy** (issue #338, 2026-09-06). Both were previously incomplete --
the stress was gated off as 5.3x too large on `xx` and sign-inverted on `yy`
and `zz`, and the gradient carried a 2.4e-3 Ha/bohr residual -- because they
differentiated an energy the driver did not actually evaluate. The periodic
kernel is now a translation-covariant Ewald-split lattice sum of the shell
gamma with the Bannwarth 2019 anisotropic electrostatics on image-resolved
cumulative atomic multipole moments (issue #296), and the derivatives
differentiate that same expression term by term.

Because the anisotropic-electrostatics Fock matrix is the exact `dE/dP` of the
energy, the reduced SCC state is variational and **no response (Z-vector) term
is needed**. Measured against central finite differences of the same energy:

| fixture | max gradient residual | max stress residual |
|---|---|---|
| polar HF cell, 3-D, 14 bohr | 3.9e-11 Ha/bohr | 2.7e-13 Ha/bohr^3 |
| h-BN monolayer, 2-D, smeared | 2.2e-09 Ha/bohr | 1.2e-10 Ha/bohr^3 |

Both are at the finite-difference noise floor rather than at a tolerance.
`vibeqc.semiempirical.periodic.evaluate_periodic_energy_gradient` returns the
analytic gradient for `gfn2_xtb` (it no longer central-differences), and
`vibeqc._vibeqc_core.semiempirical.compute_periodic_gfn2_stress` returns the
analytic stress. `finite_difference_stress` remains available and is still
what the *full-k* cell optimizer uses; multi-k GFN2 itself stays gated on
issue #351.

Bloch-periodic OMx is gated. Full k-point public semiempirical routes
are currently limited to closed-shell DFTB0 and SCC-DFTB energy/band jobs plus
the experimental lower-level DFTB derivative and NEB facades. MSINDO periodic
work uses the SECCM cyclic-cluster route, not the Gamma/k-point runner.
K-point weight sums and band capacity use compensated accumulation, and the
shared electron-capacity guard derives its rounding bound from the accumulated
mesh size. Fully occupied dense meshes therefore remain valid without relaxing
rejection of a genuinely overfilled minimal basis.

At zero electronic temperature, full-k DFTB0 and SCC-DFTB use one weighted
electron-count constraint over the complete mesh. The native multi-k GFN2
route fails closed until its complex Bloch, per-k energy, and AES model is
complete.
The globally lowest states are filled under one Fermi level, and a
roundoff-degenerate group at that level shares one fractional occupation to
preserve symmetry. Degeneracy is decided by a roundoff test, not a physical
broadening: sorted states within 256 ulps of the spectrum's scale of their
*neighbour* belong to one group (a transitive chain, so two adjacent groups
are always further apart than that tolerance and a roundoff-degenerate
manifold is never cut between two indistinguishable members, GitLab #544),
and a group's span is capped at 16 times the tolerance so an ulp-spaced
ladder cannot merge into one arbitrarily wide level. A boundary that lands on
that cap is recorded as an unresolved frontier cut (GitLab #434) rather than
integer-filled silently. This remains correct when valence and conduction bands
overlap between different k-points, where a fixed occupied-band count at every
k-point is not an Aufbau state. A single-spectrum Gamma calculation follows the
same convention: it uses integer hard Aufbau whenever the frontier has an open
gap, and equal fractional occupations over the whole roundoff-degenerate
manifold when the electron count cuts one. Equal occupation makes the density
an invariant of that eigenspace, so the energy, the energy-weighted density,
and every analytic derivative are unique instead of depending on which
eigenvectors the solver happened to return -- ideal cubic Si is the reference
case, where the arbitrary cut produced 2.4e-3 Ha/bohr of spurious force
against a finite-difference value of ~4e-12. Unrestricted Gamma jobs apply the
convention per spin channel (full occupancy 1.0 each), and the occupations
actually used are recorded on the result as `occupations` (restricted) or
`occupations_alpha` / `occupations_beta` (unrestricted). Band-path points are not a Brillouin-zone
quadrature; the DFTB0 band-path helper therefore retains independent integer
occupations at each plotted point, and its aggregate energy is not a mesh
integration result.

The analytic derivatives of such an ensemble result are the derivatives of the
executed objective *at fixed ensemble occupations*: for a Gamma DFTB0 or
SCC-DFTB result whose frontier cuts a degenerate manifold,
`compute_periodic_dftb0_stress` / `compute_periodic_scc_dftb_stress` return
`d/d(eps)` of `sum_i f_i eps_i(eps) + E_rep(eps)` with the manifold entering as a
block trace, and a central difference of that objective reproduces all nine
components to `1e-9` (ideal Si, issue #529). Along a symmetry-preserving
strain -- the isotropic strain a cubic lattice optimizer applies -- the
re-filled hard-Aufbau energy coincides with the ensemble energy on both sides
of the step, so its derivative is the analytic trace. Along a
symmetry-breaking strain the split manifold is re-filled differently on each
side and the T=0 hard-Aufbau energy has a cusp: its one-sided slopes differ
from the ensemble slope by first-order occupation changes, and a central
difference of the executed energy converges to `sigma_xx V + (d - s)/3`
(`s`, `d` the strain slopes of the split-off singlet and doublet), 10.9 % from
the analytic value on primitive Si and 2.4 % on conventional diamond C. That
number is not a derivative of anything; do not read a component-wise finite
difference of a T=0 energy at a degenerate frontier as a stress. A route that
must differentiate re-filled energies at strained cells needs finite
electronic temperature, where the occupations are smooth in the strain.

Every full-k result (DFTB0, SCC-DFTB, and the GFN2 one-point-Gamma
spelling) also measures its band edges from the occupations actually used
(issue #426): `valence_band_max` / `conduction_band_min` (with defining
k-indices and `valence_max_per_k` / `conduction_min_per_k`),
`indirect_gap`, and `direct_gap` (the minimum per-k gap). A state counts
as valence when its occupation exceeds `1e-8` and as conduction when it
falls below `2 - 1e-8` -- the same masks the ab-initio multi-k band
summary uses -- so a fractionally occupied Fermi group (degenerate
zero-temperature manifold or smeared frontier) belongs to both classes
and a converged metal reports an explicit unclamped `<= 0` gap. NaN marks
only an edge that does not exist in the model space (no electrons, or a
completely filled valence-minimal basis) or an unconverged diagnostics
record; it never stands in for a zero gap. `run_periodic_job` records the
four scalar edge fields in the `.system` manifest `[run]` section
(non-finite values as `"not-measured"`).

At a fractionally occupied *degenerate* edge, where the Fermi level is
pinned inside a partially filled manifold, the gap is physically exactly
zero, and `indirect_gap` reports that: every manifold member counts as both
valence and conduction, so the value is the negative manifold spread, which
is `<= 0` and zero to machine precision for a degenerate manifold. This
matches the Gamma route's pooled-aufbau convention
(`eps[n_fill] - eps[n_fill-1]`) to machine precision, so Gamma and k-route
gaps are directly comparable.

Two neighbouring quantities are easy to confuse with it, so both are
reported under their own names:

* `gap_above_fermi_manifold` is the *occupancy partition*,
  `min(eps | n <= 1e-8) - max(eps | n > 1e-8)` -- the distance to the next
  entirely empty state. At a fractional edge it steps over the whole
  partially filled manifold and is strictly positive. It is a legitimate
  quantity but it is **not** the band gap.
* `is_metallic` is the structural gapless flag: true when some occupation
  is fractional or the measured `indirect_gap` is `<= 0`. Screen on this
  rather than on the sign of a float -- under a zero-temperature global
  aufbau fill the pooled gap can never be negative, so a "negative gap"
  test alone is vacuous on mesh rows.

`gap <= 0` (equivalently `is_metallic`) means "metallic or gapless as
computed", not a defect flag.

The same SECCM boundary also carries the experimental DFTB0, SCC-DFTB, PM6,
OM2/OM3, and GFN2-xTB adapters at the levels described above. OM1-SECCM is
gated with its missing analytic core-valence ECP.

### DFTB periodic

```{note}
**Periodic SCC-DFTB uses the Elstner gamma with Ewald summation (issue #425,
fixed 2026-09-07).** The periodic `gamma` matrix used to be built by
truncating the Klopman-Ohno form `1/sqrt(R^2 + eta^2)` in real space. That is
a sum of `1/R`, divergent in three dimensions, and a pair-distance cutoff
gives different pairs different image counts, so the cancellation that charge
neutrality should provide was incomplete. The error scaled as the square of
the charge transfer, leaving covalent systems unaffected in practice and
strongly ionic ones unusable: LiH at a = 3.8 bohr swung over ~10 Ha between
cutoffs, every rung reporting `converged=True`.

Elstner et al., *Phys. Rev. B* **58**, 7260 (1998) p. 7263 names this -- the
Ohno/Klopman gamma forms "can therefore not be used" for periodic systems --
and its Eq. 17 `gamma = 1/R - S(R)` exists precisely so the long-range part
can be evaluated by Ewald summation while the exponentially decaying
short-range part is summed over a few cells. Periodic and k-point SCC-DFTB
now do both, and `PeriodicSCCOptions.gamma_form` selects the form. On the
same LiH fixture the Elstner energy is flat at -0.450188859 Ha from a 40-bohr
cutoff out to 200; Klopman-Ohno still drifts from -0.419 to -0.872 Ha over
that range even with the Coulomb tail Ewald-summed, so the functional form,
not the Ewald split alone, is what supplies the thermodynamic limit.

**Molecular SCC-DFTB still defaults to Klopman-Ohno.** Per the maintainer's
staged decision the flag lands first and the molecular cutover follows a
published repin table, so that one method name keeps meaning one gamma. The
molecular move is large on polar systems -- 13 kcal/mol on water, 28 on the
water dimer -- and is tracked in
`handovers/HANDOVER_GFN2_PERIODIC_ELECTROSTATICS.md`.
```

**Full-k geometry and cell optimization at finite electronic temperature.**
`run_periodic_job(..., method="dftb0" | "scc_dftb", kpoints=..., optimize=True,
smearing_temperature=kT)` minimises the Mermin free energy `F = E - TS`
(Mermin 1965), not the internal energy: every optimizer callback runs at the
same resolved temperature, the gradient and stress are derivatives of `F`
(Weinert and Davenport 1992: with fractional occupations the variational
functional is `F` and the force is `-dF/dR` with no occupation-number term),
and the optimized-geometry block reports `F_final` (minimised), `E_final`
(internal energy) and `-TS` side by side. The returned result carries
`minimised_potential="free_energy"`, `free_energy`, `internal_energy` and
`smearing_temperature`. At zero temperature the optimizer minimises `E` as
before. This also closes the loop with the zero-temperature frontier guard:
when a k-mesh cuts a frontier the guard cannot resolve, supplying the
smearing temperature it recommends is accepted on the optimization route
(issue #545; the route used to refuse finite temperature outright).


```python
from vibeqc._vibeqc_core import PeriodicSystem, Atom
from vibeqc._vibeqc_core import semiempirical as _se
import numpy as np

# 1D carbon chain
atoms = [Atom(6, [0.0, 0.0, 0.0])]
cell = np.diag([2.5, 30.0, 30.0])
system = PeriodicSystem(1, cell, atoms)

params = _se.SemiempiricalParameters.dftb0_default()
result = _se.run_dftb0_gamma(system, params)
print(f"Energy: {result.energy:.6f} Ha")
```

### GFN2 periodic

```python
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()
result = _xtb.run_gfn2_xtb_gamma(system, params)
```

Periodic GFN2-xTB smears the frontier by default: the supported Gamma driver
applies a small Fermi-Dirac temperature of 0.001 Ha (about 316 K) unless the
caller sets one explicitly. This keeps the SCC on one occupation branch when
a near-degenerate frontier crosses on a lattice sweep. The molecular GFN2
path is unchanged and remains exact zero-temperature Aufbau.

General k-point GFN2 and GFN2 band paths currently fail closed. The former
experimental route used real `cos(k dot g)` factors instead of full complex
Bloch phases, evaluated the bare-band energy against the Gamma Hamiltonian,
and omitted the AES term. A one-point Gamma mesh remains accepted for API
compatibility and delegates directly to `run_gfn2_xtb_gamma`; any non-Gamma
mesh or band path raises an explicit error. Use DFTB0/SCC-DFTB for supported
k-point semiempirical calculations until the complete GFN2 functional is
implemented.

The result reports the GFN2 internal energy as `energy` and the conjugate
Mermin/Helmholtz potential as `free_energy = energy - smearing_temperature *
entropy`, where `smearing_temperature` is electronic `k_B*T` in Hartree and
`entropy` is dimensionless `S/k_B`; it also exposes `fermi_level` and the
converged `occupations`. Finite-temperature gradients and stress (the native
fixed-charge analytic functions and the finite-difference facades) follow the
free-energy surface; optimizers driven by the periodic semiempirical energy
closures minimize the free energy of a smeared run.

At Gamma, an atom position and that position plus any integer lattice vector
are the same site. The driver normalizes those input representatives before it
constructs home-cell shell-gamma and atom-centred multipole objects. The shared
issue #316 inventory independently enumerates a pair-complete image superset
for finite-range overlap, H0, repulsion, and AES terms and screens on physical
atom or shell separation. Together, these keep the energy unchanged when a
different lattice image of the same site is typed. The periodic result exposes
`e_electronic`,
`e_repulsive`, `e_scc`, `e_band0`, `e_aes`, and `e_3rd`; their sum identities
are pinned for both the direct Gamma and accepted one-point Gamma interfaces.
The ordinary periodic route also requires a noncontractible cycle in the
image-labelled pair graph. This keeps a finite molecular cluster fail-closed
independently of whether that cluster crosses a cell face, without rejecting a
wide supercell whose physical pair graph spans the lattice.

To request the exact zero-temperature Aufbau occupations, set the electronic
temperature explicitly:

```python
opts = _xtb.XTBSccOptions()
opts.electronic_temperature = 0.0  # exact Aufbau (T = 0)
result = _xtb.run_gfn2_xtb_gamma(system, params, opts)
```

An explicit positive temperature is honoured as-is:

```python
opts = _xtb.XTBSccOptions()
opts.electronic_temperature = 0.005  # electronic k_B T in Hartree
result = _xtb.run_gfn2_xtb_gamma(system, params, opts)
```

At zero temperature, `entropy == 0` and `free_energy == energy` exactly. The
public Gamma `run_periodic_job` route applies the same default-on policy:
leave `smearing_temperature` unset for the default width, pass `0` (or
`None`) for exact Aufbau, or pass a positive width explicitly. The route
writes the finite-temperature block (`k_B T`, entropy, free energy, Fermi
level) to the `.out` file of a smeared run.

### PM6 periodic and the PM7/OMx gates

```python
from vibeqc.semiempirical import PeriodicPM6Model

pm6 = PeriodicPM6Model(system)
print(f"PM6: {pm6.energy():.6f} Ha")
```

Bloch-periodic OM1/OM2/OM3 currently fail closed. The retired prototype omitted
the defining ORT, ECP, and penetration terms and therefore could return finite
numbers under an OMx label without evaluating the published Hamiltonian. Use
the explicitly topology-bound OM2/OM3-SECCM routes for experimental cyclic-cluster
work; a future Bloch-periodic implementation must port the full image-resolved
Hamiltonian before this gate is removed.

Molecular and periodic PM7 also fail closed. Stewart's PM7 Hamiltonian applies
a smooth transition from NDDO electrostatics to exact point charges and changes
the coupled electron-electron, electron-core, and core-core terms together. The
retired prototype only loaded PM7 parameters into the PM6-family kernel, so it
could not evaluate the published method. The bundled PM7 parameter registry is
retained for completing and validating that implementation.

## Preoptimization workflows

Use semiempirical methods for fast structure preoptimization before
an expensive DFT calculation:

```python
from vibeqc.semiempirical import preoptimize_periodic

# Preoptimize a periodic system with DFTB0, then run DFT
preoptimize_periodic(
    system,
    method="dftb0",
    fmax=0.01,
)
```

For molecular systems, use `optimize=True` with `run_job`:

```python
from vibeqc import run_job

# Preoptimize with DFTB0, then refine with DFT
run_job(mol, method="dftb0", optimize=True)
run_job(mol, method="rks", functional="PBE", basis="def2-svp", optimize=True)
```

(semiempirical-status)=
## Method status

| Method | Status | Energy accuracy | Gradient | Periodic | Open-shell | Elements |
|--------|--------|-----------------|----------|----------|------------|----------|
| DFTB0 / UDFTB0 | Screening/preopt | In-house parameters, not DFTB+ parity | Analytic | Gamma; full-k DFTB0 closed-shell | molecular/Gamma yes; full-k closed-shell only | 91 in-house |
| DFTB0-SECCM | Gated experimental | In-house screening parameters, explicit repulsive-pair scope; finite-cluster closure | Analytic, fixed topology | 1-D / 2-D / 3-D direct finite torus | no | H, C, N, O, F, P, S, Cl |
| SCC-DFTB-SECCM | Gated experimental | Molecular-limit bit parity with SCC-DFTB; 1-D chain within 60 microhartree of the independent dense-k limit at 32 cells; charged cells through opt-in Madelung/Ewald embedding | Analytic, fixed topology | 1-D/2-D embedded; odd-replica 3-D embedded with `gamma_form="elstner"` (#211); 1-D/2-D and odd-replica 3-D unembedded finite torus | no | in-house DFTB screening set |
| PM6-SECCM | Gated experimental | Molecular-limit bit parity with PM6 inside the current s/p-only envelope; full heavy-heavy tensor parity remains open and d-orbital records fail closed | FD, fixed topology | 1-D/2-D/3-D finite torus | no | bundled MOPAC, s/p records only |
| OM2/OM3-SECCM / GFN2-SECCM | Gated experimental | OMx-SECCM: molecular-limit bit parity with `run_omx_v2`; WS-weighted supercell Fock with cited eq-13 (default) or eq-10 three-center image weighting. GFN2-SECCM: molecular-limit bit parity with `run_gfn2_xtb`; WS-weighted supercell shell-resolved SCC (image-block H0, shell gamma, multipole AES, GAM3) | not implemented | OMx: exact zero-image molecular limit by default; 1-D/2-D/3-D cyclic-image topologies only with the explicit non-quantitative truncated-electrostatics acknowledgement. GFN2: 1-D/2-D/3-D finite torus | no | published OMx / fetched GFN2 sets |
| SCC-DFTB / USCC | Screening/preopt | In-house parameters, not DFTB+ parity | Analytic | Gamma; full-k SCC-DFTB closed-shell | molecular/Gamma yes; full-k closed-shell only | 91 in-house |
| GFN2-xTB | Gated experimental | External xTB parity matrix still open | Analytic molecular and Gamma-periodic derivatives. The periodic gradient and all-nine stress match gapped finite differences; public reaction-path forces remain FD while broader parity is open. | Gamma experimental; full-k gated | no | 86 fetched (LGPL) |
| PM6 / UPM6 | Molecular development | H-only and H-heavy s/p interactions are source-correct; full heavy-heavy two-center tensor parity remains open. Spherical Klopman-Ohno fallback applies where MOPAC diatomic data is absent | FD | Gamma experimental, closed-shell | molecular yes; periodic no | 75 chemical-element records from the bundled MOPAC cache |
| PM7 / UPM7 | Gated | Published feathered electrostatics and coupled NDDO terms are not implemented | unavailable | unavailable | no | 75 chemical-element parameter records retained for implementation |
| OM2 / OM3 | Validated molecular prescreening | Published relative energetics match H3- and ethane fixtures, but integral stand-ins leave bond minima ~0.1-0.2 Å long and external implementation parity is open | FD | Bloch-periodic route gated; topology-bound OMx-SECCM experimental | molecular yes; periodic no | 5 published (H,C,N,O,F) |
| OM1 | Experimental (warns) | Analytic core-valence ECP (Kolb-Thiel 1993) not implemented; X-H bonds ~0.3 A short, close contacts can collapse | FD | Bloch-periodic and topology-bound OM1-SECCM routes gated | molecular yes; periodic no | 5 published (H,C,N,O,F) |
| MSINDO | Production within scope | Reference MSINDO parity <=1 uHa for INDO and closed-shell NDDO fixtures, including Al-Cl SPDD | FD plus analytic INDO molecular/CCM; analytic closed-shell NDDO molecular | SECCM 1-D/2-D/3-D + Ewald | molecular INDO UHF s/p/d within validated fixtures; NDDO and SECCM closed-shell | H-Xe INDO; NDDO energy/gradient H,Li-F,Na-Cl |

The same implementation labels are available from Python through
`vibeqc.semiempirical.semiempirical_route_status(route)`. Routes such as
`msindo-cis`, `msindo-cis-gradient`, `msindo-ovgf`, `msindo-md`, and
`msindo-metadynamics` are explicitly marked `python-reference` until their hot
loops move to native kernels or are declared intentionally orchestration-only.
`periodic-pm6` is marked `mixed-native`: Gamma energy uses native NDDO kernels,
while gradient, stress, and cell-optimization helpers use one native batched
finite-difference workspace. Molecular and periodic PM7 are
gated because the shared prototype omitted the defining electrostatic terms.
`periodic-omx` is gated because the former prototype was not the published
OMx Hamiltonian. The lookup also
labels molecular `pm6-gradient-fd` and `omx-gradient-fd` as `native-fd`: their
displacement loops are C++-backed, but they remain finite-difference stopgaps
for those methods. The lookup accepts public method
aliases such as `dftb0`, `scc-dftb`, `gfn2xtb`, `om2`, `om2-gradient-fd`,
`gfn2`, `msindo`, and `ccm`.

Direct `run_semiempirical(...)` results compute gradients lazily when
`result.gradient()` is called. DFTB0/SCC-DFTB, GFN2-xTB, PM6/UPM6, and OMx
expose their existing gradient surfaces through that adapter, while
closed-shell MSINDO INDO uses the native analytic-gradient route and validated
closed-shell NDDO H, Li-F, and Na-Cl use the mixed
native-energy/Python-analytic route. Open-shell NDDO keeps `gradient()`
unavailable and fails closed when an analytic derivative is requested.
Geometry-optimizer
`SemiempiricalProvider` calls the same unified runner, preserving the documented
MSINDO finite-difference force fallback when a MSINDO result is energy-only.

The lower-level `run_native_energy(...)` facade accepts the same bohr-valued
`Molecule` geometry as the rest of vibe-qc. For MSINDO INDO and NDDO it uses
the exact inverse of the engine's pinned 1986-CODATA Angstrom-to-bohr
conversion. The direct `run_msindo(...)` API remains Angstrom-valued, as
documented in {doc}`msindo`.

```{seealso}
{doc}`semiempirical_mlip_comparison` for production guidance and
`../semiempirical_acceptance_matrix.py` for the living validation-gate
matrix.
```

## Element coverage

**DFTB**, 91 elements (H-U except Po/Z=84), including 3d/4d/5d
transition metals, lanthanides (La-Lu), and early actinides
(Ac-U). All parameters are in-house estimates; DFT-fitted
production repulsive potentials are deferred.

**GFN2-xTB**, 86 elements from the published Grimme-group parameter
set.  Parameters are fetched on demand at first use (LGPL-3.0
licensed, not bundled, see ADR-002).

**PM6**, 75 chemical-element records from the bundled MOPAC PM6 parameter cache
(Apache-2.0 provenance in the TOML header), with the Stewart 2007
H/C/N/O/F subset still available. The public wrapper auto-selects the
MOPAC-derived cache for elements outside H/C/N/O/F.

**PM7**, 75 chemical-element parameter records from the bundled MOPAC cache.
The records are available for implementation work, but all PM7 energy and
derivative routes are gated until the full published Hamiltonian is present.

**OMx**, 5 elements (H, C, N, O, F) from Dral 2016 Tables 1-3.

```python
# Check element coverage
from vibeqc.semiempirical import SemiempiricalParameters

params = SemiempiricalParameters.dftb0_default()
elements = [Z for Z in range(1, 93) if params.has_element(Z)]
print(f"DFTB covers {len(elements)} elements")
```

## Parameter customisation

### DFTB custom parameters

```python
from vibeqc.semiempirical import SemiempiricalParameters

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

model = DFTB0Model(mol, params=custom)
```

### GFN2 parameters

GFN2-xTB parameters are fetched automatically from the Grimme group's
GitHub repository.  To force a refresh:

```python
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params(force_refetch=True)
```

For offline batch planning, probe the local cache without opening the network:

```python
from vibeqc.semiempirical import semiempirical_route_runtime_available

if not semiempirical_route_runtime_available("gfn2_xtb"):
    # Mark GFN2-xTB rows unavailable before submitting the batch.
    ...
```

### Published-parameter identity

vibe-qc does not transcribe the GFN2-xTB parameter values into its own
source tree. They are fetched at runtime from the Grimme group's
`param_gfn2-xtb.txt` (LGPL-3.0) and parsed into a parameter object whose
canonical content hash is pinned in the native core. A load that
reproduces the pinned hash reports the identity
`published:gfn2-xtb-2019-v1`; any other load is labelled `custom:<hash>`,
so a tampered or hand-edited set cannot present itself as the published
one.

```python
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params(allow_fetch=False)
print(params.parameter_identity())  # published:gfn2-xtb-2019-v1
print(params.content_sha256())      # 0b3c70a5...
```

The hash attests the *parsed parameter object*, not the upstream file
bytes; the file's own digest is carried separately in the cache header as
`source_sha256`.

**Re-attestation of 2026-08-27 (issues #43 and #446).** The pinned digest
changed from `71b83ab4...` to `0b3c70a5...` because a transcription error
was corrected. The upstream file lists each element's shells in
occupation order, which is d-first for the 41 transition-metal elements
(`ao=3d4s4p`, `4d5s5p`, `5d6s6p`; Z 21-29, 39-47, 57-79), while the
`KCNS`/`KCNP`/`KCND` and `POLYS`/`POLYP`/`POLYD` records are named per
angular momentum. The converter indexed those named records by shell
position, so on exactly those 41 elements the s parameter was applied to
the d shell, the p parameter to the s shell, and the d parameter to the p
shell. It surfaced as an H0 error that grew with distance: Cu2 deviated
from `xtb` by +35.8/+56.8/+139.5 mHa at 2.2/2.5/3.0 Angstrom, free
Cu8/Cu16 clusters never converged, and fcc Cu cyclic-cluster cells
settled into a spurious charge-density-wave basin. Elements whose shells
are already listed in angular-momentum order were unaffected, and their
parameters are bit-identical across the change.

The corrected values are the published ones. They were checked against
Bannwarth, Ehlert and Grimme, *J. Chem. Theory Comput.* **15**, 1652-1671
(2019),
[doi:10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176),
Supporting Information Table S52, "Element-specific shell parameters
employed in GFN2-xTB", which tabulates the polynomial scaling parameters,
the coordination-number-dependent level enhancement, the level constants
and the Slater exponents *per shell label*. Per-angular-momentum keying is
the paper's own convention, fixed by its eq 17 and eq 19. Taking Cu as the
worked example, Table S52 lists polynomial scaling parameters 0.177983,
0.149778 and -0.265089 for the 4s, 4p and 3d shells, and level constants
-6.922958, -2.267723 and -9.506548 eV; the loaded set carries exactly
those values on l = 0, 1 and 2.

```{note}
The defect was originally located by an element-by-element H0 bisection
against `tblite`. That is an independent implementation, not the
publication, so it corroborates the fix but does not establish published
provenance. The identity claim above rests on SI Table S52.
```

To re-run the check rather than trusting the pinned constant, use the
regression oracle
`tests/test_gfn2_xtb.py::TestGFN2DFirstParameterProjection`, which pins
the per-l Cu values and the Cu2 parity bounds, together with
`tests/test_gfn2_parameter_identity.py`, whose header records the full
derivation. Reading a shell parameter back and comparing it against
Table S52 is the direct check:

```python
from vibeqc.semiempirical.methods.gfn2_params import _read_cached_toml

cu = next(e for e in _read_cached_toml()["element"] if int(e["Z"]) == 29)
print({int(s["l"]): s["poly"] for s in cu["shells"]})
# {2: -0.265089, 0: 0.177983, 1: 0.149778}
```

**How the converter holds that mapping (issue #467).** It does not rely
on the upstream file's line order. Each `KCN*`/`POLY*` record is mapped
onto an angular momentum by its own shell letter as it is read (`S` to
l = 0, `P` to 1, `D` to 2), and a duplicated or unrecognised letter
aborts the refresh rather than being absorbed, so no cache is written
from a source this converter cannot map. Collecting the records in line
order would have stayed correct only for as long as every upstream block
happened to list them in s, p, d order, which all 86 blocks of the
revision current at 2026-08-28 do, but which nothing upstream
guarantees. A re-ordered future revision would have re-created the
rotation described above with no test failing: the rotated cache is
still structurally valid, and its only visible symptom would be the
content digest moving off the pin, which reads as tampering rather than
as a mis-read source. The guard is
`tests/test_gfn2_xtb.py::TestGFN2ShellRecordKeying`, which renders a
synthetic element block with those records deliberately out of order.

Any future change to the parameter tables or to the converter must run
`tests/test_gfn2_parameter_identity.py`, because that lane is what holds
the published-identity claim honest.

### PM6 parameters

```python
from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

params = load_pm6_params()
model = PM6Model(mol, params=params)
```

## Comparing against external programs

Reference energies from external programs can be obtained via
out-of-process subprocess runners ({doc}`external_codes`):

```python
from examples.regression.core.runner_xtb import energy as xtb_energy
from examples.regression.core.runner_mopac import energy as mopac_energy
from examples.regression.core.runner_dftbp import energy as dftbp_energy

print(f"xTB GFN2 H2O:  {xtb_energy('H2O'):.6f} Eh")
print(f"MOPAC PM6 H2O: {mopac_energy('H2O'):.6f} Ha")
print(f"DFTB+ H2O:     {dftbp_energy('H2O'):.6f} Ha")
```

These runners require the external program to be installed on `$PATH`
(see each runner's docstring for install instructions).

## Performance tips

- **DFTB0** is 3-5× faster than SCC‑DFTB (no SCF loop).  Use it
  for preoptimization where charge self-consistency is less
  important.
- **DFTB gradients** (DFTB0 and SCC-DFTB) are analytic and match
  finite differences tightly; the SCC energy is variational in
  the density, so its fixed-charge gradient is exact at SCC
  convergence.
- **Periodic systems** support Gamma-point energy routes for DFTB0/SCC-DFTB,
  GFN2-xTB, PM6, and OMx. Public full-k semiempirical support is DFTB0 and
  SCC-DFTB only, closed-shell only. Increase the lattice cutoff (`cutoff_bohr`)
  for tight cells.
- **Memory** is negligible, the basis is minimal (one function
  per valence shell).

## Known limitations

- DFTB repulsive potentials are in-house R{sup}`−12` estimates;
  DFT-fitted production repulsives are deferred. This holds for **every**
  element pair, including the ones the built-in table lists explicitly: a
  table row supplies a hand-rounded prefactor `A`, not a fitted potential, and
  is evaluated by the same `A/R`{sup}`−12` expression as the combining-rule
  fallback. `DFTB0RepulsivePlaceholderWarning` therefore fires for any pair
  whose repulsive is on that branch, tabulated or not, and names the pairs.
  Absolute energies, equilibrium geometries and EOS fits from those runs are
  placeholder physics; fixed-geometry differences (k-mesh or supercell
  convergence, Madelung on/off, Gamma vs multi-k) remain exactly valid.
- GFN2-xTB uses a finite, physical-distance-screened AES image inventory but
  still lacks a production periodic multipole Ewald construction and a closed
  external `xtb` parity matrix ({ref}`semiempirical-status`).
- PM6 reports a PM6-like total, not a MOPAC heat of formation; use MOPAC
  out-of-process when exact MOPAC convention parity is required.
- Periodic PM6 cancels neutral-pair lattice monopoles, but bare PM6 has no
  validated rare-gas dispersion model; do not use its dissociative fcc-Ar
  curve as a physical equation of state.
- OM2/OM3 are validated molecular development/prescreening paths within their
  documented H/C/N/O/F scope; they are not external-parity production claims.
  OM1 remains experimental until the analytic core-valence ECP lands.
- Periodic GFN2/NDDO gradients are finite-difference only;
  analytic periodic NDDO gradients are deferred.
- MSINDO molecular closed-shell analytic gradients and closed-shell CCM
  analytic / finite-difference gradients are available through the native route
  inside their documented scopes. Molecular closed-shell NDDO H, Li-F, and
  Na-Cl also have an analytic route; periodic NDDO, odd-electron analytic gradients, and
  excited-state/root-tracking gradients remain on their documented fallback,
  gated, or reference paths. See {doc}`msindo`.
