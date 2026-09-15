(relaxed_scan)=
# Relaxed coordinate scans

`vq.relaxed_scan(...)` drives one internal coordinate, a bond, an
angle, or a dihedral, across a sequence of target values and relaxes
all other degrees of freedom at each step. This is the canonical
"scan the reaction coordinate" workflow used to locate transition-state
guesses and map reaction energy profiles. It works for both molecules
and periodic systems with the same API.

## Quick start, molecular O-H stretch

```python
import numpy as np
import vibeqc as vq

h2o = vq.Molecule(
    [
        vq.Atom(8, [0.00,  0.00, 0.0]),
        vq.Atom(1, [1.81,  0.00, 0.0]),
        vq.Atom(1, [-0.46, 1.75, 0.0]),
    ],
    charge=0, multiplicity=1,
)

result = vq.relaxed_scan(
    h2o, "sto-3g",
    coordinate=("bond", 0, 1),          # O (atom 0) → H (atom 1)
    values=np.linspace(1.6, 2.4, 9),    # bohr
    method="RHF",
    output="oh_stretch",                # writes oh_stretch.qvf
)

print(result.values)        # target bond lengths
print(result.energies)      # relaxed energies (Ha)
print(result.converged_flags)  # bool per scan point
print(result.geometries[0]) # vq.Molecule at the first scan point
```

The optimizer freezes the two constraint atoms (so the bond length
stays exactly at the target) and lets every other atom relax. Each
scan point warm-starts from the previous relaxed geometry, which gives
smoother paths.

## Quick start, periodic bond scan

The same call signature works for periodic systems; pass a
`PeriodicSystem` and a `kpoints` mesh.

```python
import numpy as np
import vibeqc as vq

L = np.eye(3) * 12.0
h2_box = vq.PeriodicSystem(
    3, L,
    [vq.Atom(1, [6.0, 6.0, 6.0]),
     vq.Atom(1, [7.4, 6.0, 6.0])],
    charge=0, multiplicity=1,
)

result = vq.relaxed_scan(
    h2_box, "sto-3g",
    coordinate=("bond", 0, 1),
    values=np.linspace(1.2, 2.4, 7),
    method="RHF",
    kpoints=(1, 1, 1),     # Γ-only
    output="h2_dissociation",
)
```

## Coordinate types

Three coordinate kinds are supported. Bond targets are in **bohr**;
angle and dihedral targets are in **radians**.

| Coordinate                 | Form                              | Unit    |
| -------------------------- | --------------------------------- | ------- |
| Bond length                | `("bond", i, j)`                  | bohr    |
| Bond angle (vertex `j`)    | `("angle", i, j, k)`              | radians |
| Dihedral (i-j-k-l torsion) | `("dihedral", i, j, k, l)`        | radians |

Atom indices are 0-based and refer to the order of atoms in
`molecule.atoms` (or `system.unit_cell` for periodic systems).

## Freezing additional atoms, the slab pattern

For surface catalysis, freeze the slab bottom while the adsorbate
moves. The constraint atoms are always frozen; pass extra indices via
`freeze_indices`:

A relaxed scan needs periodic forces, and slab analytic gradients are not
shipped yet, so build the cell as `dim=3` with `periodic_z=True`:

```python
# periodic_z=True -> dim=3 with a real vacuum gap (forces need it).
slab_sys, slab_info = vq.slab(..., periodic_z=True)
n_layers_frozen = 2

result = vq.relaxed_scan(
    slab_sys, "def2-svp",
    coordinate=("bond", n2_idx_a, n2_idx_b),  # N2 bond stretch
    values=np.linspace(2.1, 4.0, 12),
    method="RKS", functional="PBE",
    kpoints=(2, 2, 1),
    freeze_indices=slab_info.bottom_layer_indices(n_layers_frozen),
    output="n2_dissociation_fe100",
)
```

The frozen-atom forwarding mirrors {func}`vibeqc.relax_atoms` (see
{doc}`bipole`).

## Approach: rigid step with frozen endpoints

At each scan point:

1. The starting geometry is mutated so the constrained coordinate hits
   the target value exactly. For a bond, atom `j` is moved along the
   `i → j` direction. For an angle, atom `k` is rotated about vertex
   `j`. For a dihedral, atom `l` is rotated about the `j → k` axis.
2. The atoms participating in the constraint are added to the
   optimizer's `freeze_indices`. The remaining atoms relax around the
   constrained geometry using the existing analytic-gradient L-BFGS-B
   driver.
3. The resulting geometry, energy, and convergence flag are recorded.

This is the same "rigid step" approach used by most QC packages for
relaxed scans. The slightly more elegant alternative, projecting the
gradient orthogonal to the constraint direction, is a planned v1.0
enhancement.

## Output: `ScanResult` and the QVF reaction path

`relaxed_scan` returns a {class}`vibeqc.ScanResult`:

```python
result.values             # np.ndarray of target coordinate values
result.energies           # np.ndarray of relaxed energies (Ha)
result.geometries         # list of Molecule / PeriodicSystem
result.converged_flags    # list of bool
result.coordinate         # the constraint spec passed in
result.output_path        # Path to the QVF, if output= was set
```

When `output="..."` is set (or you call `result.write_qvf(path)`),
a vibe-view-compatible `reaction.path` QVF is emitted. Frames become
the scan animation; the reaction coordinate axis is the constraint
value; endpoints are tagged as reactant / product waypoints, and the
highest-energy frame is tagged as a generic `point` waypoint. (It is
*not* tagged as a transition state, that requires a Hessian check;
see {doc}`../tutorial/vibrational_frequencies`.)

The energy-plot x-axis is labelled from the scan coordinate, a bond
scan reads `bond 0–1 (bohr)`, an angle/dihedral scan reads radians, 
so the plot is self-describing without you annotating it.

**Marking a verified transition state.** If you have *independently*
confirmed (e.g. via a frequency calculation with a single imaginary
mode) that a particular scan frame is a transition state, pass its
index to `write_qvf` to tag it with the red TS marker:

```python
result.write_qvf("scan.qvf", transition_state_frames=[3])
```

This suppresses the generic `point` waypoint at that frame. We never
auto-promote the energy maximum to a transition state, a scan maximum
is not a verified saddle point.

Open the resulting `.qvf` with vibe-view to scrub the scan
interactively.

**Periodic scans.** When `geometries` is a list of
`PeriodicSystem` (periodic scan), the archive attaches the per-frame
lattice + dim to the `reaction.path` section, staying
`qvf_version: 1`: the `lattice` member's presence is what marks the
path periodic. vibe-view's renderer (QVF Inc B) draws
the unit-cell wireframe and wraps atom positions across in-plane
periodic boundaries, slab scans animate with the cell visible
and adsorbates don't jump across the cell boundary mid-scan.

**Per-frame densities (animated isosurface).** Pass
`emit_volumes_every=N` to attach the electron density, evaluated on a
shared real-space grid, for every `N`-th scan frame, so vibe-view can
morph the density isosurface as you scrub the scan:

```python
result.write_qvf("scan.qvf", emit_volumes_every=1)   # every frame
result.write_qvf("scan.qvf", emit_volumes_every=5)   # every 5th (decimated)
```

Each emitted frame costs one extra single-point SCF, so cubes are
opt-in and decimated; the grid box auto-sizes to the union of all
emitted geometries (tune with `volume_spacing` / `volume_padding`).
The density rides on the `reaction.path` section as an optional
`frame_volumes` 4-D member + a shared `volume_grid`, additive
metadata that older viewers ignore. **Molecular scans only** for now;
periodic per-frame densities (GPW collocation) raise
`NotImplementedError` and remain a future enhancement.

**Citation surface.** `write_qvf` emits a `citations` section
containing the BibTeX for every per-frame SCF (method / basis /
functional). The relaxed-scan driver itself doesn't fire a
driver-specific paper (relaxed coordinate scans are textbook;
there is no single "scan paper" to cite). NEB callers needing
the Henkelman+Jónsson 2000 / Henkelman+Uberuaga+Jónsson 2000 /
Smidstrup 2014 citations should use
{meth}`vibeqc.NEBResult.write_qvf` instead, it fires the
`routes.drivers.neb` (and optionally `ci_neb`) routes
automatically.

## 2D scans, energy surfaces

To sweep **two** coordinates at once, use
{func}`vibeqc.relaxed_scan_2d`. It drives a grid of
`(coordinate_a, coordinate_b)` values, relaxing everything else at each
node, and returns a {class}`vibeqc.ScanResult2D` with a `[nA, nB]`
energy grid:

```python
import vibeqc as vq

res = vq.relaxed_scan_2d(
    mol, "sto-3g",
    coordinate_a=("bond", 0, 1), values_a=np.linspace(1.4, 2.0, 7),
    coordinate_b=("angle", 1, 0, 2), values_b=np.linspace(1.7, 2.3, 7),
    method="RHF",
    output="surface",   # writes surface.qvf
)
res.write_qvf("surface.qvf")   # or call explicitly
```

The two coordinates should be independent (disjoint moved atoms). The
`.write_qvf` archive is a `scan.surface` section, vibe-view renders it
as a filled-contour energy map with the minimum starred and both axes
labelled from the coordinate names + units. The relaxed geometry at
every node is stored too (pass `include_geometries=False` to omit it).

## Limits and out-of-scope features

* **Constrained-gradient projection** is not implemented; the rigid-
  step approach above is what ships today.
* **Synchronous-transit / NEB** path methods live in a separate
  workstream, `relaxed_scan` is the simpler reaction-coordinate
  driver.
