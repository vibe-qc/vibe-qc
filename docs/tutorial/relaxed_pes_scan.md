---
myst:
  html_meta:
    "description": "Relaxed potential-energy surface scans in vibe-qc: drive one bond, angle, or dihedral while every other coordinate relaxes at each step. The relaxed_scan API, a worked H2O bond scan recovering the equilibrium geometry, and the ScanResult object."
    "og:title": "vibe-qc tutorial - relaxed potential-energy surface scans"
---

# Relaxed potential-energy surface scans

A **relaxed scan** maps how the energy changes along one chosen coordinate (a
bond, angle, or dihedral) while every *other* degree of freedom relaxes at
each step. Unlike a rigid scan, which freezes the whole geometry, a relaxed
scan traces the true minimum-energy path along that coordinate: the right
tool for a dissociation curve, a torsional profile, or a one-dimensional
reaction coordinate.

## Running a scan

`relaxed_scan` takes the molecule, a basis, the coordinate to drive, and the
values to step through. The remaining coordinates are optimised at each
point:

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0,  1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])

scan = vq.relaxed_scan(
    mol, "sto-3g",
    coordinate=("bond", 0, 1),           # the O-H(1) bond
    values=[1.4, 1.6, 1.8, 2.0, 2.2],    # bond length in bohr
    method="RHF",
)
for r, e in zip(scan.values, scan.energies):
    print(f"{r:.2f} bohr   {e:.6f} Ha")
```

```text
1.40 bohr   -74.845508 Ha
1.60 bohr   -74.934691 Ha
1.80 bohr   -74.964215 Ha
2.00 bohr   -74.961050 Ha
2.20 bohr   -74.939884 Ha
```

The minimum sits near 1.8 bohr (about 0.95 Angstrom), the equilibrium O-H
distance. The relaxed scan recovers it because the other coordinates (the
second O-H and the H-O-H angle) relax at every step; a rigid scan of a
distorted starting geometry would not.

## The result object

`relaxed_scan` returns a `ScanResult` with:

- `values` and `energies`: the scan coordinate and the relaxed energy at each
  point.
- `geometries`: the optimised structure at each point, for export or restart.
- `converged_flags`: whether each constrained optimisation converged.

The `coordinate` argument also takes `("angle", i, j, k)` and
`("dihedral", i, j, k, l)`, and a periodic system can be scanned the same way
by passing `kpoints=`.

## See also

- [Geometry optimization](geometry_optimization.md): the unconstrained
  relaxation that each scan point performs.
- [Nudged elastic band](neb_reaction_path.md): for a full reaction path
  between two minima, rather than a single driven coordinate.
- [The relaxed-scan reference](../user_guide/relaxed_scan.md) for the
  complete option set.
