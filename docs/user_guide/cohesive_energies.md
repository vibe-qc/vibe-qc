# Cohesive energies and the energy-engine abstraction

The **cohesive** (atomization) energy of a solid is the energy needed to
pull it apart into free atoms. It is the quantity the solid-state
basis-optimisation work optimises against, and computing one is not a
single calculation but a sequence: relax the cell, take a single point,
correct for zero-point motion, compute the free atoms, and assemble.

`vibe_basis.pipeline` runs that sequence. `vibe_basis.engine` is the
seam that decides *which program* runs each step.

```{note}
These live in the co-located `vibe-basis` driver, so they need vibe-qc's
`basisopt` extra. See
[The external-program path](basis_optimization.md#the-external-program-path).
The plan they implement is `vibe-basis/ROADMAP.md`.
```

## The engine abstraction

Every engine implements one protocol, `EnergyEngine`, so which program
computes a number is configuration rather than structure:

| Role | Engine | Where it lives |
|---|---|---|
| Primary, Gaussian | `VibeQcEngine` | `vibeqc.basis_optimization.calculators` (phonons in `...basis_optimization.phonons`) |
| Primary, plane wave | vibe-qc GPW / GAPW | not wired yet |
| Fallback, Gaussian | `Crystal23Engine` | `vibe_basis.engines.crystal23` |
| Fallback, plane wave | `GpawEngine` | `vibe_basis.engines.gpaw` (declared, not built) |

**vibe-qc is the engine. The external ones are fallbacks and
cross-checks.** They exist to unblock work whose vibe-qc capability has
not landed, and to give an *independent* second opinion: agreement
between two implementations is evidence in a way that one
implementation agreeing with itself is not.

The primary engine lives outside the driver package while the fallbacks
live inside it. That inversion is deliberate. `VibeQcEngine` needs
vibe-qc's compiled core; `Crystal23Engine` needs only subprocess and
file I/O. Keeping the external engines dependency-free is what makes
the standalone `./vibe-basis/scripts/install.sh` route lightweight for a
collaborator who has their own CRYSTAL and no interest in building vibe-qc.
vibe-basis is not on PyPI yet, so the managed source installer is the supported
onboarding path.

### Every number carries its provenance

An engine returns an `EngineEnergy`, not a bare float:

```python
from vibe_basis.engines import Crystal23Engine
from vibe_basis.transports.local import LocalTransport
from vibe_basis.io.structures import STRUCTURES

engine = Crystal23Engine(transport=LocalTransport())
result = engine.crystal_energy("<inline basis text>", STRUCTURES["MgO"], method="pbe")

print(result.ok)              # did it converge and complete?
print(result.energy)          # Hartree per unit cell
print(result.engine)          # "crystal23"
print(result.engine_version)  # what the output banner actually said
print(result.failure_mode)    # None on success
```

"MgO is -275.4776 Ha" is not a result. "CRYSTAL23 computed -275.4776 Ha
for MgO at RHF/pob-TZVP" is. A campaign that swaps engines as easily as
this one does has to record which one produced each point, or it cannot
be audited afterwards.

If you only want a number, `evaluate_crystal` and `evaluate_atom` return
`float | None`. They are *derived* from the rich methods, not a parallel
implementation, so the two views cannot disagree.

### `VibeQcEngine` defaults are wiring defaults, not converged settings

Three of them will bite a first campaign if taken at face value:

- **Gamma point only.** The periodic single point runs at a single
  k-point unless you say otherwise. That is enough to check the wiring
  and far too coarse for a production number. Pass a converged mesh with
  `VibeQcEngine(kpoints=...)`.
- **BIPOLE, not GDF.** GDF needs an auxiliary fitting basis, and a
  freshly emitted candidate basis has none registered. Borrowing another
  basis's JKFIT set would lay a fitting error on top of the
  basis-incompleteness signal the objective exists to measure, so the
  route that needs no auxiliary basis is the default. Override through
  `periodic_kwargs` if you have a fitting set you trust for the basis in
  hand.
- **Static lattice unless another engine relaxes it.** `VibeQcEngine` does
  not advertise `relax`: BIPOLE variable-cell optimization is fail-closed
  pending a certified stress and coupled atom/cell optimizer. Use
  `CohesivePipeline(..., relax=False)` for a static-lattice calculation or
  select an engine with a certified cell optimizer.

### Zero-point energy: `VibeQcEngine` has Gamma-point phonons

`VibeQcEngine` computes the ZPE stage itself, on the same BIPOLE route,
k-mesh and functional as the single point. `Crystal23Engine` still has no
phonon path and records a skipped stage.

Three things to know before quoting a number from it.

**It is a Gamma-only ZPE, which is an approximation for a solid.** It
samples one q-point and therefore misses the dispersion of the acoustic
branches away from Gamma entirely. It is the standard cheap estimate and
it is the right size; it is not a converged thermodynamic quantity.
`EngineEnergy.detail["approximation"]` says `"gamma_point_only"`.
Phonon-dispersion ZPE (supercell or q-mesh) is not implemented.

**The Hessian is built from total energies, not from forces.** On the
GAPW route, `vibeqc.periodic_gapw_phonon` differences the analytic GAPW
force, which is production there. On the BIPOLE route the analytic
gradient is a research-preview surface (see `vibeqc.bipole_optimize`) and
the production force is *itself* a finite difference, so differencing it
again would stack two FD layers. Central differences of the energy are
one layer resting on a quantity correct by construction. The price is
that it costs `18 N^2 + 1` SCFs and grows quadratically with cell size:

```python
# doc-audit: skip
from vibeqc.basis_optimization.phonons import estimate_scf_count
estimate_scf_count(2)   # 73  -- a 2-atom primitive cell
estimate_scf_count(8)   # 1153
```

Budget it before starting a campaign. `detail["n_scf_estimated"]` carries
the same number into the run record.

**No acoustic sum rule is imposed by default.** The three Gamma acoustic
modes are zero in exact arithmetic, so their residual is a direct measure
of the calculation's noise floor; zeroing it silently would make that
unmeasurable. They are identified by projection onto the mass-weighted
uniform-translation subspace (not by "the three lowest frequencies",
which picks the wrong modes as soon as an optical mode goes imaginary),
excluded from the ZPE sum, and reported:

```python
# doc-audit: skip
zpe = engine.zero_point_energy(basis_text, structure, method="pbe")
zpe.detail["acoustic_residual_cm1"]        # the noise floor, uncorrected
zpe.detail["n_imaginary_modes_excluded"]   # geometry is not a minimum
zpe.detail["asr"]                          # 'none' unless you asked
```

Pass `VibeQcEngine(phonon_kwargs={"asr": "rowsum", "step_bohr": 0.03})`
to change either. The finite-difference step is a real choice: an energy
Hessian divides by `h^2`, so SCF noise enters as `eps/h^2` while
anharmonicity enters as `O(h^2)`. The default is 0.02 bohr, twice the
GAPW module's, for exactly that reason. Scan it on a new system class
rather than assuming it transfers --
`examples/basisset_dev/gamma_phonon_gate.py` does the scan and the
cross-route comparison.

### The free-atom reference: aspherical and spin-polarised

Half of a cohesive energy is the free atoms, and an atom reference is
only comparable to an external one computed under the **same**
convention. `Crystal23Engine` emits `SYMMREMO`, `UHF` (or `SPIN` in the
DFT block) and `SPINLOCK` for an open-shell atom, so the solution is
aspherical and spin-polarised; `VibeQcEngine` runs the spin-polarised
ground-state multiplicity. Both record which convention they used in
`EngineEnergy.detail["atom_reference"]`.

This is worth checking rather than assuming, because the failure is
silent: a spin-restricted atom SCF converges perfectly well and simply
returns the wrong energy. In the plane-wave reference work,
spin-restricting the bromine reference alone moved KBr's atomization
energy by 37 kJ/mol, which is far larger than the basis-set effects a
campaign is trying to resolve.

### Counterpoise: matching the published pob protocol

Every free-atom deck in the pob cohesive-energy reference set uses
CRYSTAL's `ATOMBSSE`, meaning the atom is computed **in the ghost basis
of its own crystal** rather than bare. The published numbers are
therefore counterpoise corrected, and a cohesive energy assembled from
bare atoms is not the same quantity.

```python
# doc-audit: skip
pipeline = CohesivePipeline(engine, method="r2scan", counterpoise=True)
result = pipeline.run(basis_text, structure)
assert result.counterpoise_applied
```

The difference between the two is the basis-set superposition error,
which for an incomplete basis is exactly what a basis-optimisation
campaign is trying to reduce. Leaving it out does not cancel when you
compare against a corrected reference.

Two consequences:

- **It is off by default**, and `CohesiveResult.counterpoise_applied`
  records which you got. An engine that cannot do it (currently
  `VibeQcEngine`) produces a recorded skipped stage rather than quietly
  falling back to bare atoms.
- **Counterpoise atoms are not shared across systems.** A bare oxygen is
  the same calculation for MgO and CaO, so the cache computes it once. A
  counterpoise oxygen is computed in *its own host's* ghost basis, so
  the two are genuinely different runs and the cache keys them by host.
  That is a real cost: it removes the saving that makes a campaign
  affordable, so use it for reference comparison rather than for every
  point of an optimisation.

`nstar` and `rmax` control how far the ghost shell extends. The
reference set uses per-system converged values (LiF `30 / 10.0`, MgO
`10 / 5.0`), so treat `Crystal23Engine`'s defaults as a starting point
to converge rather than a setting to trust.

### Failures are returned, not raised

A failed evaluation comes back with `ok=False` and a `failure_mode`. It
does not raise. Basis optimisation walks into infeasible regions
routinely: an exponent collapses, the overlap goes singular, the queue
drops a job. A derivative-free optimiser handles that by scoring the
point `inf` and moving on, so an exception would abort a campaign that
is minutes per evaluation and days long.

One consequence worth internalising: `energy_if_ok()` returns `None` for
a failed evaluation **even when `.energy` holds a number**. A
non-converged SCF's last-cycle energy is kept for diagnosis, and
silently consuming it as if it were converged is exactly the failure
that method exists to prevent.

## The pipeline

```python
from vibe_basis.cache import EnergyCache
from vibe_basis.pipeline import CohesivePipeline
from vibe_basis.engines import Crystal23Engine
from vibe_basis.transports.local import LocalTransport
from vibe_basis.io.structures import STRUCTURES

pipeline = CohesivePipeline(
    Crystal23Engine(transport=LocalTransport()),
    method="pbe",            # bulk single point and the free atoms
    relax_method="pbe",      # geometry relaxation, if the engine can
    relax=True,
    zero_point=True,
    cache=EnergyCache("runs/campaign.jsonl"),
)

result = pipeline.run("<inline basis text>", STRUCTURES["MgO"])
print(result.summary())
```

Stages, in order:

1. **Relax** the cell and internal coordinates at `relax_method`, so the
   basis is judged at *its own* equilibrium geometry rather than at one
   some other basis produced.
2. **Single point** at `method`, on the relaxed geometry. The usual
   pattern is a cheap GGA relaxation followed by a hybrid single point.
3. **Zero-point energy** of the solid.
4. **Isolated atoms**, one per distinct element.
5. **Assemble.**

With `VibeQcEngine`, request `relax=False`: its capability set deliberately
causes a requested relaxation stage to be recorded as skipped rather than
silently returning the input geometry as if it had been optimized.

The arithmetic, per formula unit:

```text
E_coh  =  SUM_atoms E_atom  -  E_bulk_per_formula_unit  -  ZPE_per_fu
```

positive for a bound solid, matching the sign convention of
`vibeqc.atomization.AtomizationResult`. Zero-point motion
*reduces* the cohesive energy: the solid's true ground state sits ZPE
above its static-lattice minimum, while free atoms have no vibrations to
correct.

Formula units are counted from the cell's stoichiometry, as the greatest
common divisor of the per-element counts. Rocksalt MgO with 4 Mg and 4 O
is 4 formula units; fluorite CaF2 with 4 Ca and 8 F is also 4. Pass
`n_formula_units=` explicitly for a cell whose formula is not the reduced
one, such as a defect supercell.

## Read `zero_point_applied` before comparing to anything

```python
# doc-audit: skip
if not result.zero_point_applied:
    ...  # this is a STATIC-LATTICE cohesive energy
```

A static-lattice cohesive energy and a ZPE-corrected one are different
quantities, and for these solids they differ by tens of kJ/mol, which is
the same size as the basis-set effects being measured. Comparing one to
a reference computed as the other produces a discrepancy that looks like
physics and is not.

So `CohesiveResult` states which it is, and the pipeline never quietly
substitutes a default. If an engine cannot relax, or cannot do phonons,
that stage is recorded as **skipped, with the engine named in the
reason**:

```python
# doc-audit: skip
for stage in result.stages:
    print(stage.name, stage.status, stage.reason)
# relax        skipped  engine 'crystal23' has no relaxer
# single_point ok       None
# zero_point   skipped  engine 'crystal23' has no phonons
# atom_Z8      ok       None
# atom_Z12     cached   None
```

Returning the input geometry for a missing relaxation would look like a
converged relaxation; returning zero for a missing ZPE would look like a
solid with no vibrations. Both would produce a number that looks
finished and is not.

A **failed** zero-point stage is survivable, and downgrades the result to
static-lattice with `zero_point_applied=False`. A **failed relaxation is
not**: an unconverged geometry is not a geometry, so the run stops
before spending an SCF on it.

## Caching and resume

`EnergyCache` is content-addressed over an append-only JSONL journal.
Pass one cache across every system in a campaign:

```python
# doc-audit: skip
cache = EnergyCache("runs/campaign.jsonl")
for name in ("MgO", "CaO", "BaO"):
    CohesivePipeline(engine, method="pbe", cache=cache).run(basis_text, STRUCTURES[name])
```

Free-atom entries are keyed **without a system identifier**, so the
oxygen computed for MgO is reused for CaO and BaO. That is the win that
matters: atoms are the expensive part. In the plane-wave reference work
a single free fluorine atom cost 100 to 240 seconds while the solid was
quick, so recomputing oxygen once per compound would dominate a
campaign.

Because the journal is append-only and *is* the cache, a campaign killed
mid-run resumes rather than re-spending the queue time. A journal
truncated by `kill -9` costs the one torn entry, not the file.

Keys derive from declared configuration only, never from observed
output. In particular the engine's reported version is deliberately not
part of the key: `Crystal23Engine` cannot report one until its first
evaluation, so keying on it would change the key halfway through a run
and silently split the cache. The observed version is recorded in the
value instead, for audit.

## Which CRYSTAL produced this?

`Crystal23Engine` targets CRYSTAL23, and the parsed result records the
major version read off the output banner:

```python
from vibe_basis.backends.crystal import parse_output_file

parsed = parse_output_file("mgo.out")
print(parsed.crystal_version)        # 23, or None if no banner in the text
print(parsed.version_matches(23))    # True / False / None
```

`version_matches` is three-valued on purpose. `None` means the banner
was not in the text, which is normal for a truncated output or a tail
slice, and is **not** a failure: an output with no visible banner but a
converged total energy is still a usable energy. `False` means the
banner said something else, and the energy is still parsed, but
attributing it to CRYSTAL23 would be false.

Enforcement is a policy decision, so it is opt-in. Set
`Crystal23Engine(..., strict_version=True)` for a production campaign,
where a stray binary on one host quietly contributing a few points is
the real hazard. Leave it off for a parity study that legitimately wants
CRYSTAL14 output.

## See also

- [Molecular basis-set optimisation](basis_optimization.md), the
  in-process analytic-gradient path.
- [Basis sets](basis_sets.md), the bundled library.
- `vibe-basis/ROADMAP.md`, the milestones this implements.
- `vibe-basis/TUTORIAL.md`, a worked walkthrough.
- `examples/basisset_dev/cohesive_pipeline_demo.py`, runnable with no
  external program and no native build.
