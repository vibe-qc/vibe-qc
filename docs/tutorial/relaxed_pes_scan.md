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

The minimum sits between 1.8 and 2.0 bohr: a parabola through the three
lowest points puts it near 1.88 bohr (about 0.995 Angstrom), the HF/STO-3G
equilibrium O-H distance (experiment: 0.96 Angstrom). The relaxed scan recovers it because the other coordinates (the
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

## Theory

A relaxed scan evaluates the constrained minimum of the Born-Oppenheimer
energy along one internal coordinate $q(\mathbf R)$:

$$
E_\text{relaxed}(q_0) = \min_{\mathbf R:\; q(\mathbf R) = q_0} E(\mathbf R).
$$

At such a point the only gradient left is along the constraint. This is the
stationarity condition of the Lagrangian
$L(\mathbf R, \lambda) = E(\mathbf R) - \lambda\,[q(\mathbf R) - q_0]$,

$$
\nabla_{\mathbf R} E = \lambda\, \nabla_{\mathbf R}\, q,
\qquad q(\mathbf R) = q_0,
\qquad \lambda = \frac{\mathrm{d}E_\text{relaxed}}{\mathrm{d}q_0},
$$

so the multiplier is the generalised force needed to hold the coordinate
(J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer
(2006), chapter 12). vibe-qc imposes the constraint by a rigid step: the
atoms that define $q$ are moved so that $q = q_0$ exactly and then frozen,
and the remaining atoms relax on the analytic gradient with the L-BFGS-B
optimizer behind `optimize_molecule` (R. H. Byrd, P. Lu, J. Nocedal and
C. Zhu, *SIAM J. Sci. Comput.* **16**, 1190 (1995), doi:10.1137/0916069),
or with `relax_atoms` for a periodic system. Each point starts from the
previous relaxed geometry. For angles and dihedrals, freezing whole atoms is
stricter than the Lagrangian condition; the
[reference page](../user_guide/relaxed_scan.md) explains the consequence.
The highest point of a profile is only a transition-state candidate: confirm
it with a [frequency calculation](vibrational_frequencies.md), which must
show exactly one imaginary mode.

## See also

- [Geometry optimization](geometry_optimization.md): the unconstrained
  relaxation that each scan point performs.
- [Nudged elastic band](neb_reaction_path.md): for a full reaction path
  between two minima, rather than a single driven coordinate.
- [The relaxed-scan reference](../user_guide/relaxed_scan.md) for the
  complete option set.
