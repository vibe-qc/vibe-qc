# Periodic semiempirical methods and cross-method validation

**You'll learn:** how vibe-qc's semiempirical methods (DFTB0, SCC-DFTB,
GFN2-xTB, PM6) run under periodic boundary conditions, how to check a
periodic result against its own molecular limit, and how the project's
internal cross-method and cross-reference comparisons are structured,
including the one rule that matters most when reading them: **never
compare absolute energies across method families.**

This page assumes the molecular-level methods already covered in
[Semiempirical DFTB](semiempirical_dftb.md) (which also has the basic
Gamma-only periodic DFTB0/SCC example this page builds on),
[PM6 and GFN2-xTB](pm6_and_gfn2.md), and [MSINDO](msindo.md). It does not
repeat NEB, COSMO, molecular dynamics, metadynamics, conical
intersections, or CISD; those are MSINDO-specific and already covered
end to end in the [MSINDO tutorial](msindo.md) and the
[MSINDO user guide](../user_guide/msindo.md).

Runnable scripts for everything below live in
[`examples/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/semiempirical/).

## Periodic DFTB0 beyond Gamma: k-meshes and band structure

[Semiempirical DFTB](semiempirical_dftb.md#periodic-systems) already
shows the Gamma-point-only case. The same 1D carbon-chain setup extends
to a Monkhorst-Pack mesh and a band path with two more calls:

```python
from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import semiempirical as _se
import numpy as np

atoms = [Atom(6, [0.0, 0.0, 0.0])]
cell = np.diag([2.5, 30.0, 30.0])          # 2.5 bohr C-C spacing, 30 bohr vacuum
system = PeriodicSystem(1, cell, atoms, charge=0, multiplicity=1)
params = _se.SemiempiricalParameters.dftb0_default()

# Gamma point (as in the DFTB tutorial)
gamma = _se.run_dftb0_gamma(system, params)

# 4x1x1 Monkhorst-Pack mesh
kmesh = monkhorst_pack(system, (4, 1, 1))
mesh_result = _se.run_dftb0_kpoints(system, params, kmesh)
print(f"Gamma:    {gamma.energy:.6f} Ha/cell")
print(f"4x1x1 MP: {mesh_result.energy:.6f} Ha/cell")

# Band path: eps_per_k is one eigenvalue array per k-point on the path
kpath = monkhorst_pack(system, (8, 1, 1))   # dense sampling stands in for a real path
band = _se.run_dftb0_bandpath(system, params, kpath)
for k, eps in enumerate(band.eps_per_k):
    print(f"  k[{k}]: {eps[:3]}")           # lowest three bands at this k-point
```

`run_dftb0_kpoints` and `run_dftb0_bandpath` are the periodic siblings of
`run_dftb0_gamma`, taking the same `(system, params)` pair plus a
k-point set from `monkhorst_pack`.

The self-consistent k-routes (`run_scc_dftb_kpoints`, and
`run_gfn2_xtb_kpoints` on its supported one-point Gamma mesh) fail loud:
when the SCC loop exhausts `max_iter` they raise
`SCFNonConvergenceError` (a `RuntimeError` subclass, also exported as
`vibeqc.semiempirical.SCFNonConvergenceError`) instead of returning an
unconverged record. Pass `allow_unconverged=True` to receive the record
(still flagged `converged=False`) for convergence diagnostics.
DFTB0 is non-self-consistent, so its k-route has no such failure mode.

## Periodic PM6: cutoff convergence and the molecular limit

Periodic PM6 lives in `vibeqc.semiempirical.methods.periodic_pm6`. The
production entry point is `run_pm6_gamma`, a real-space sum truncated at
`cutoff_bohr`:

```python
from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc.semiempirical.methods.pm6_params import load_pm6_params
from vibeqc.semiempirical.methods.periodic_pm6 import run_pm6_gamma

params = load_pm6_params()   # Stewart 2007, H/C/N/O/F
he_chain = PeriodicSystem(
    1, np.diag([3.0, 30.0, 30.0]), [Atom(2, [0.0, 0.0, 0.0])], 0, 1,
)

# Real-space cutoff convergence
for cutoff in (3.0, 5.0, 10.0, 15.0):
    r = run_pm6_gamma(he_chain, params, cutoff_bohr=cutoff)
    print(f"cutoff={cutoff:>5.1f} bohr: E={r.energy:.8f} Ha  n_cells={r.n_cells}")
```

`PeriodicPM6Result` also carries `.e_electronic`, `.e_core`, `.converged`,
`.n_iter`, `.n_basis`, and `.n_occ` if you need the SCF breakdown rather
than just the total.

`cutoff_bohr` bounds the **pair distance** `|R_A - R_B - g|`, not the
lattice-translation length (issue #316): every periodic semiempirical
route sums exactly the image pairs within the cutoff, so a Gamma
supercell wider than the cutoff keeps its image interactions instead of
silently degrading to a free-boundary cluster, and the result is
invariant under relabeling an atom by a lattice vector.  `n_cells`
reports the translations that carry at least one in-range pair
(`vibeqc._vibeqc_core.semiempirical.atom_pair_interaction_cells` exposes
the same list).  A genuinely isolated system - no image pair within the
cutoff - still fails closed unless the explicit `gamma_only_0=True`
molecular mode is selected.  For a single-atom cell the pair rule and
the translation rule coincide, so the He-chain sweep above is unchanged.

**Checking against the molecular limit.** A periodic PM6 cell large
enough to isolate one molecule and evaluated only at the true Gamma point
(`gamma_only_0=True`, which skips the periodic image sum entirely rather
than approximating it) should reproduce the plain molecular PM6 energy:

```python
from vibeqc import Molecule, Atom as MolAtom
from vibeqc._vibeqc_core.semiempirical.nddo import (
    run_pm6 as run_pm6_mol, PeriodicPM6Options, run_pm6_gamma as run_pm6_gamma_raw,
)

h2_mol = Molecule([MolAtom(1, [0, 0, 0]), MolAtom(1, [0, 0, 1.4])])
e_mol = run_pm6_mol(h2_mol, params).energy

h2_cell = PeriodicSystem(
    3, np.diag([10.0, 10.0, 10.0]),
    [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1,
)
opts = PeriodicPM6Options()
opts.gamma_only_0 = True
e_per = run_pm6_gamma_raw(h2_cell, params, opts).energy

print(f"molecular: {e_mol:.10f} Ha")
print(f"periodic:  {e_per:.10f} Ha")
print(f"match:     {abs(e_mol - e_per) < 1e-10}")   # True
```

This is the same parity check `examples/semiempirical/08_periodic_pm6.py`
and `09_periodic_pm6_tutorial.py` print for an isolated H₂ in a 10-bohr
cubic cell: a useful sanity check any time you're unsure whether a
periodic-limit setup is wired correctly, on any system.

For geometry relaxation, `vibeqc.semiempirical.periodic.optimize_pm6_cell(system,
fmax=0.01, max_steps=100, cutoff_bohr=15.0)` wraps `run_pm6_gamma` with FD
gradients and stresses (`compute_pm6_gamma_gradient_fd`,
`compute_pm6_gamma_stress_fd`) behind ASE's cell optimizer and returns
the relaxed `PeriodicSystem`.

**Elements outside PM6's core set.** `load_pm6_params()` only covers
H/C/N/O/F (Stewart 2007). For anything else,
`load_pm6_params_auto(atomic_numbers)` auto-selects the broader MOPAC
parameter set instead, and `run_pm6_gamma(system)` called with no
`params` argument does the same element-based auto-selection internally,
which is how a periodic system containing e.g. Mg or O falls back to
MOPAC parameters without you having to check element coverage by hand.

## All four periodic families on one system: graphene

`examples/semiempirical/11_periodic_graphene.py` is the one place all
four periodic semiempirical method families run on the same genuinely
2D system: a 2-atom hexagonal graphene cell, `a = 4.65` bohr (≈2.46 Å),
20 bohr of vacuum along z, built with `PeriodicSystem(2, lattice, atoms,
0, 1)` (`dim=2`, unlike the 1D chains above):

```python
import numpy as np
from vibeqc import Atom
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical import SemiempiricalParameters

a, c = 4.65, 20.0   # lattice constant, vacuum layer (bohr)
lattice = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
atoms = [Atom(6, [0.0, 0.0, 0.0]), Atom(6, [a / 2, a * np.sqrt(3) / 6, 0.0])]
graphene = PeriodicSystem(2, lattice, atoms, charge=0, multiplicity=1)

params = SemiempiricalParameters.dftb0_default()
dftb0 = _se.run_dftb0_gamma(graphene, params)

scc_opts = _se.PeriodicSCCOptions()
scc_opts.max_iter = 80
scc = _se.run_scc_dftb_gamma(graphene, params, scc_opts)

from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

gfn2_params = load_gfn2_params()
xtb_opts = _xtb.XTBSccOptions()
xtb_opts.max_iter = 400
gfn2 = _xtb.run_gfn2_xtb_gamma(graphene, gfn2_params, xtb_opts)

from vibeqc.semiempirical import PeriodicPM6Model

pm6_energy = PeriodicPM6Model(graphene).energy()
```

(`graphene` here is the `PeriodicSystem` built as described above; see
the example script for the exact lattice/atom construction.) Each method
family has its own options class (`PeriodicSCCOptions`, `XTBSccOptions`,
or none at all for DFTB0/PM6), worth knowing before you reach for one on
a system of your own, since the option surface isn't uniform across
families.

## Cross-method and cross-reference validation

Four example scripts exist purely to check vibe-qc's semiempirical
methods against each other or against pinned external references. None
of them are "run this to get a scientific result" scripts; they're the
project's own validation harness, worth understanding so you can read
their output correctly (or write a similar check for your own system).

**Internal consistency** (`07_cross_validate.py`, despite the name, does
not call an external code). It runs the same six-molecule set (H₂, H₂O,
CH₄, NH₃, CO₂, C₂H₄) through `run_job` with `method` in `"dftb0"`,
`"scc_dftb"`, `"pm6"`, `"gfn2_xtb"`, `"om1"`, `"om2"`, `"om3"`, and checks
one weak, physically-motivated ordering: `E(scc_dftb) <= E(dftb0) +
1e-8` for every molecule (self-consistent charges should not raise the
energy above the uncorrected model). The script is explicit that this is
the *only* safe cross-method claim: **do not read a universal energy
ordering across DFTB, GFN2-xTB, PM6, and OMx from this table**: the
methods use different reference states and parametrizations.

**PM6 parameter-set comparison** (`10_pm6_vs_mopac.py`) runs the same
six molecules through `PM6Model` with `load_pm6_params()` (Stewart 2007)
and `load_mopac_pm6_params()` (parsed from MOPAC's own open-source
parameter file) side by side, and compares both against a small set of
pinned reference energies. The gaps it reports (~0.5 Ha for Stewart,
~3 Ha for MOPAC-parameter vibe-qc vs. the pinned MOPAC references) are
*expected*: they come from differences in the core-core repulsion
formula, not a bug. This script demonstrates known, characterized
disagreement, not parity.

**The benchmark suite** (`12_benchmark_suite.py`) runs DFTB0,
SCC-DFTB, GFN2-xTB, and PM6 on the six-molecule set and times each,
catching exceptions per (molecule, method) so one failure doesn't stop
the sweep. Unlike the other three scripts, it writes a report file,
`examples/semiempirical/12_benchmark_suite.md`, with a convergence/
timing table rather than only printing to the console.

**The three-tier comparison** (`18_comparison_all.py`) runs a single
water molecule through `run_job` across three tiers: ab initio (`hf`,
`pbe`, `pbe0`, `b3lyp`), semiempirical (`dftb0`, `scc_dftb`, `gfn2_xtb`,
`pm6`, `om1`, `om2`, `om3`, `msindo`, the full eight-method family), and
MLIP (`method="mace"`). It performs no comparison itself; every call
just writes its own `.out`. The point is the caveat stated in both the
script's docstring and its closing print: **absolute total energies are
not comparable across tiers.** Compare geometries, forces, frequencies,
or relative energies within one tier, never raw totals across them.

## DFT-fitted repulsive potentials

`examples/semiempirical/fit_repulsive.py` is a standalone tool with **no
vibe-qc imports at all**: it's offline curve-fitting math (NumPy/SciPy),
not something that calls into the C++ core. Given `(R, E_DFT, E_DFTB0)`
triples (or synthetic demo data if you don't supply any), it fits a
cubic-spline correction `V_rep(R) = E_DFT(R) - E_DFTB0(R) - baseline` and
writes the knots to a TOML file.

That output is meant to feed
`vibeqc.semiempirical.io.load_parameters`'s spline-repulsive loader,
but not automatically: the
loader expects an array-of-tables `[[repulsive]]` block with `Z1`, `Z2`,
`R_bohr`, and `V_Ha` keys per pair, while `fit_repulsive.py` writes those
same keys at the TOML root. Wrap its output in a `[[repulsive]]` header
before loading it, or call `params.set_repulsive_pair_spline(Z1, Z2, R,
V)` directly with the fitted arrays instead of going through the file at
all. This is exactly the "DFT-fitted production repulsive potentials"
that [Semiempirical DFTB's parameter-customisation
section](semiempirical_dftb.md#parameter-customisation) notes are
deferred in the built-in sets; `fit_repulsive.py` is the tool that
would produce them for a pair you've fitted yourself.

## Next

- [Semiempirical DFTB](semiempirical_dftb.md), [PM6 and
  GFN2-xTB](pm6_and_gfn2.md), and [MSINDO](msindo.md) for the molecular-level
  methods this page assumes.
- The [MSINDO user guide](../user_guide/msindo.md) for molecular dynamics,
  metadynamics, conical intersections, and CISD, MSINDO-specific
  capabilities not covered on this page.
- [`examples/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/semiempirical/)
  for every script referenced above, plus the DFTB/PM6/GFN2/MSINDO
  gradient-validation and atomization/frequency scripts (19-21) not
  detailed here.
