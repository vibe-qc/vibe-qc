# Post-SCF properties

Every `run_job` calculation now reports a properties block at the end
of the `.out` file with **atomic charges**, **bond orders**, and the
**dipole moment**, the standard sanity-check output every QC program
emits.

## What you'll see in the `.out`

```
Atomic charges
------------------------------------------------------------------
   #  elem    Mulliken      Löwdin   Hirshfeld
------------------------------------------------------------------
   1     O   -0.910152   -0.646928   -0.365368
   2     H    0.455076    0.323464    0.182684
   3     H    0.455076    0.323464    0.182684
------------------------------------------------------------------
        sum  -0.000000   -0.000000    0.000000

Bond orders (Mayer)
----------------------------------------
        atoms    bond order
----------------------------------------
O1   — H2          0.7628
O1   — H3          0.7628

Dipole moment (origin: centre of mass)
----------------------------------------------------
   component     a.u. (e·bohr)       Debye
----------------------------------------------------
           x       -0.00000000     -0.0000
           y       -0.00000000     -0.0000
           z       -0.81069504     -2.0606
         |μ|        0.81069504      2.0606
```

The Hirshfeld column appears whenever the grid build + SAD
promolecule succeed (effectively always for standard bases). If
either step fails the table falls back to the two-column
Mulliken / Löwdin layout, the properties block is best-effort
and never crashes the run.

Bond orders below 0.10 are suppressed from the table to keep the
output compact; the full matrix is available programmatically.

## Programmatic access

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("h2o.xyz")
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)

# Per-atom charges (numpy array, length = n_atoms)
mul = vq.mulliken_charges(result, basis, mol)
low = vq.loewdin_charges(result, basis, mol)

# Hirshfeld charges — real-space partition on the Becke-Lebedev grid,
# far less basis-sensitive than Mulliken/Löwdin. Returns a rich
# HirshfeldResult dataclass; `.charges` is the bare-array view.
h = vq.hirshfeld_charges(result, basis, mol)
print("Hirshfeld q(O):", h.charges[0])
print("Grid integration sanity:",
      f"∫ρ_mol = {h.molecule_norm:.4f}  vs n_e expected.")

# Bond-order matrix (n_atoms × n_atoms numpy)
B = vq.mayer_bond_orders(result, basis, mol)
print("O–H bond order:", B[0, 1])

# Dipole moment (with conversions)
dip = vq.dipole_moment(result, basis, mol)
print(f"|μ| = {dip.total_debye:.3f} Debye  (origin: {dip.origin})")
print(f"μ = ({dip.x:.4f}, {dip.y:.4f}, {dip.z:.4f}) a.u.")
```

All four methods accept the result from any of the four molecular SCF
drivers (`run_rhf`, `run_uhf`, `run_rks`, `run_uks`). UHF / UKS results
are handled correctly, the total density `D_α + D_β` is used for
populations and dipole, and Mayer bond orders use the spin-resolved
formula `2·[(P_α S)(P_α S) + (P_β S)(P_β S)]` which reduces to the
closed-shell expression when `D_α = D_β`.

Beyond these built-ins, the [bond analysis guide](bond_analysis.md)
covers Wiberg bond indices, the NPA implementation gate and provisional
NBO search, energy decomposition analysis, and orbital-entanglement
measures. Wiberg data land automatically in the
`.population.{txt,json}` sidecars. The reserved NPA field is empty with a
structured error until full occupancy-weighted NAOs are available.

### `dipole_moment` vs `compute_dipole`

These are two different things and the names are close enough to trip
over, so, explicitly:

| Call | Returns | Use it for |
|---|---|---|
| `vq.dipole_moment(result, basis, mol, *, origin=None)` | `DipoleMoment`: `.x/.y/.z`, `.total`, `.total_debye`, `.origin` | the **physical dipole moment** of a converged SCF |
| `vq.compute_dipole(basis, origin=(0,0,0))` | `DipoleIntegrals`: `.x/.y/.z`, each an `(n_bf, n_bf)` matrix | the raw **AO dipole-integral matrices** `⟨μ|r − O|ν⟩`, when you are building your own response or localisation code |

`compute_dipole` is an integral builder: it knows nothing about a
molecule or a density, takes no `Molecule` and no density matrix, and
its signature has been `(basis, origin)` since the binding was
introduced. Passing `(molecule, basis, density)`, a plausible-looking
guess, raises

```text
TypeError: compute_dipole(): incompatible function arguments.
```

If you want the dipole moment, call `dipole_moment`; `run_job` also
writes it into the `.out` properties block and the `.qvf`
`dipole_moment` metadata with no extra call at all.

### Static polarizability sidecar

`run_job` has no polarizability option, so the supported route for a
polarizability is a sidecar next to the standard artifact set:

```python
import json
import numpy as np
import vibeqc as vq

options = vq.RHFOptions()
vq.run_job(mol, basis="def2-svp", method="rhf", rhf_options=options,
           output="water")                      # full artifact set + dipole

basis = vq.BasisSet(mol, "def2-svp")
result = vq.run_rhf(mol, basis=basis, options=options)
alpha = np.asarray(vq.dipole_polarizability_rhf(result, basis, mol))
mu = vq.dipole_moment(result, basis, mol)       # NOT compute_dipole

json.dump({
    "polarizability_au": alpha.tolist(),
    "polarizability_isotropic_au": float(np.trace(alpha) / 3.0),
    "dipole_au": [mu.x, mu.y, mu.z],
    "dipole_total_debye": mu.total_debye,
}, open("water.properties.json", "w"), indent=1)
```

`dipole_polarizability_rhf` returns the symmetric `(3, 3)` static
tensor in atomic units; pass a precomputed `eri=` if you are also
running `dynamic_polarizability_rhf` on the same system, so the ERI
tensor is built once. Both live in `vibeqc.cphf` and are re-exported at
the top level. This recipe is pinned end to end in
`tests/test_properties.py`
(`test_property_sidecar_dipole_and_polarizability`).

## Validation

vibe-qc's Mulliken charges and dipole-moment components match PySCF's to
better than `1e-6` on H₂O / 6-31G\* (verified in
`tests/test_properties.py`). Hirshfeld charges are validated in
`tests/test_hirshfeld_charges.py`: the weight-normalisation identity
`Σ q_A = molecule.charge` holds to < `1e-5` e on the default grid,
the grid sweep is block-size-independent to floating-point round-off,
and the open-shell (UHF) path is exercised on the OH radical. The Mayer bond order on H₂ at the
equilibrium bond length comes out near 1.0 as expected for a single
covalent bond; in H₂O at HF/6-31G\* the O-H bond orders are about 0.76
and the H…H "bond order" is below `0.005`.

## Choosing a charge model

| Model | Cost | Basis sensitivity | Best for |
|---|---|---|---|
| Mulliken  | trivial | high (diffuse-fn sensitive) | trend across a series at fixed basis |
| Löwdin    | one symmetric orth | medium | same, less diffuse-sensitive |
| Hirshfeld | one grid build + AO eval + SAD promol | low (real-space) | quantitative-ish per-atom charges; charge-dependent dispersion methods (D4) |

Hirshfeld is the recommended default when you want a per-atom charge
number that doesn't shift wildly across reasonable basis choices.
It's also the natural input to charge-dependent dispersion methods
- vibe-qc's v0.10.0 D2b refinement plumbs ``hirshfeld_charges`` into
the D4 C6 scaling step (see [`docs/roadmap.md` § v0.10.0](../roadmap.md)).

## Caveats

- **Mulliken charges depend strongly on the basis set**, especially on
  diffuse functions. Use them for trends along a series, not for
  quantitative charge assignment. Löwdin charges are usually less
  basis-sensitive.
- **Hirshfeld charges in vibe-qc are ~0.04-0.06 e more negative on
  heavy atoms than ORCA-reported values** because the promolecule
  is constructed in the *molecular* basis (via the SAD initial guess)
  rather than from tabulated Slater-type atomic densities. Same
  qualitative trend, same response to chemistry, but expect a
  small offset when cross-comparing to ORCA Hirshfeld tables.
- **The default origin for the dipole is the center of mass**, computed
  from a built-in atomic-mass table covering H-Kr. For heavier elements
  pass an explicit `origin=` (a 3-vector in bohr).
- **Dipole values for charged species are origin-dependent**; callers
  should choose an origin that matches the convention they want to
  report (center of charge, nuclear centroid, etc.).
- **Periodic systems are not yet covered** by the properties block.
  Results from the periodic SCF drivers (`run_rhf_periodic_scf` for
  HF, `run_rks_periodic_scf` for KS) don't currently flow through the
  `.out` output infrastructure; periodic dipole moments themselves
  are subtle anyway (require a Berry-phase polarization treatment
  for unambiguous values), and arrive in a later phase.
