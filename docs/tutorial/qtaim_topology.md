---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for QTAIM, Bader's quantum theory of atoms in molecules: locate the critical points of the electron density of water at RHF/6-31G*, read the density and Laplacian at the bond critical points, classify covalent, polar and ionic bonds by the sign of the Laplacian, and find the topology.qtaim section in the QVF archive. Includes the theory of zero-flux surfaces and critical-point classification."
    "og:title": "vibe-qc tutorial - QTAIM: critical points and bond paths"
---

# QTAIM: critical points and bond paths

A converged wavefunction gives you orbitals, but orbitals are not
observables and their shapes depend on the basis and on unitary rotations
that leave the energy unchanged. The electron density $\rho(\mathbf r)$ is
different: it is an observable, and its *topology*, the pattern of maxima,
saddle points and minima, is the same whichever orbitals produced it.
Bader's quantum theory of atoms in molecules (QTAIM,
{vibeqc-cite}`bader_qtaim_1985`) reads chemistry from that topology. Every
nucleus sits at a maximum of $\rho$; two atoms are bonded when a ridge of
density, the *bond path*, connects their maxima through a saddle point,
the *bond critical point* (BCP); and the density and its Laplacian at the
BCP tell a covalent bond from an ionic one or a hydrogen bond.

vibe-qc ships the critical-point search and the bond-path tracer in
`vibeqc.qtaim`, on the analytic AO Hessian of the native core, and writes
the result into the QVF archive as a `topology.qtaim` section. The
analysis is production on every molecular SCF reference (RHF, UHF, RKS,
UKS; see [Methods: theory and status](../methods.md#how-to-read-the-status-labels)).
This page runs it on water, reads the numbers, repeats it on four bonds of
different character, and derives the classification rules in the theory
section.

## Setup

Two routes lead to the same analysis. `run_job(..., qtaim=True)` runs the
topology after the SCF and stores it in the QVF; the library function
`qtaim_analysis` runs it on any converged result and returns Python
objects. The script does both, so you can see where each number ends up.

```python
import json
import zipfile

import numpy as np
import vibeqc as vq
from vibeqc.qtaim import qtaim_analysis

ANG = 1.8897259886           # Angstrom -> bohr (vibe-qc coordinates are bohr)
BOHR = 0.529177210903        # bohr -> Angstrom

# Water, experimental-like geometry: r(OH) = 0.958 A, HOH = 104.5 deg.
mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0,  0.7572 * ANG, 0.5865 * ANG]),
    vq.Atom(1, [0.0, -0.7572 * ANG, 0.5865 * ANG]),
], 0, 1)

# Route 1: the runner. qtaim=True adds a topology.qtaim section to the QVF.
result = vq.run_job(mol, basis="6-31g*", method="rhf", qtaim=True,
                    output="h2o_qtaim")
print(f"E(RHF/6-31G*) = {result.energy:.8f} Ha")

with zipfile.ZipFile("h2o_qtaim.qvf") as qvf:
    manifest = json.loads(qvf.read("manifest.json"))
    section = next(s for s in manifest["sections"] if s["kind"] == "topology.qtaim")
    topology = json.loads(qvf.read(section["members"]["critical_points"]["path"]))
print("QVF critical points:", [cp["type"] for cp in topology["points"]])

# Route 2: the library function on the same converged result.
basis = vq.BasisSet(mol, "6-31g*")
top = qtaim_analysis(result, basis, mol)      # grid_spacing=0.15 bohr by default

print(f"{'type':>4} {'atoms':>7} {'x/A':>7} {'y/A':>7} {'z/A':>7} "
      f"{'rho/au':>8} {'lap/au':>9} {'eps':>6}")
for cp in top.critical_points:
    x, y, z = cp.position                      # Angstrom
    rho = cp.rho * BOHR**3                     # e/A^3 -> e/bohr^3
    lap = cp.laplacian * BOHR**5               # e/A^5 -> e/bohr^5
    eps = "" if cp.ellipticity is None else f"{cp.ellipticity:6.3f}"
    pair = "" if cp.atom_pair is None else f"{cp.atom_pair[0]}-{cp.atom_pair[1]}"
    print(f"{cp.type:>4} {pair:>7} {x:7.3f} {y:7.3f} {z:7.3f} {rho:8.4f} {lap:9.4f} {eps:>6}")
print("bond paths:", [bp.atoms for bp in top.bond_paths])
for bp in top.bond_paths:
    pts = np.asarray(bp.path)                  # Angstrom; runs atoms[0] -> BCP -> atoms[1]
    length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    print(f"bond path {bp.atoms}: {len(pts)} points, length {length:.4f} A")
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) the whole script
takes about six seconds: the SCF converges in nine iterations to
$E = -76.00910804$ Ha, the runner's topology adds about one second, and the
second, explicit analysis another second.

```
E(RHF/6-31G*) = -76.00910804 Ha
QVF critical points: ['bcp', 'bcp', 'ncp', 'ncp', 'ncp']
type   atoms     x/A     y/A     z/A   rho/au    lap/au    eps
 bcp     2-0  -0.000  -0.605   0.474   0.3555   -1.9395  0.024
 bcp     1-0  -0.000   0.605   0.474   0.3555   -1.9395  0.024
 ncp           0.000   0.000   0.000 291.1051 -1926203.7956
 ncp           0.000   0.757   0.586   0.3899  -16.8004
 ncp           0.000  -0.757   0.586   0.3899  -16.8004
bond paths: [(2, 0), (1, 0)]
bond path (2, 0): 22 points, length 0.9579 A
bond path (1, 0): 22 points, length 0.9579 A
```

## What just happened

`qtaim_analysis` lays a uniform grid over the molecule (0.15 bohr spacing,
2.5 bohr of padding), evaluates $\rho$, $\nabla\rho$ and the $3\times3$
Hessian $\nabla\nabla\rho$ at every point with the analytic AO second
derivatives, picks the voxels where $|\nabla\rho|$ is a local minimum as
seeds, and polishes each seed with Newton-Raphson steps until
$|\nabla\rho| < 10^{-7}$. The sign pattern of the Hessian eigenvalues at
the converged point classifies it. Water has the textbook set: three
nuclear critical points (`ncp`) and two bond critical points (`bcp`), no
ring, no cage. The Poincare-Hopf relation $n - b + r - c = 1$ (nuclei
minus bonds plus rings minus cages) gives $3 - 2 + 0 - 0 = 1$, the first
sanity check on any critical-point search: a missing BCP or a spurious
ring shows up as a violation.

The two BCPs are what you look at. Each sits on the O-H axis, 0.769 A from
the oxygen and 0.189 A from the hydrogen. That asymmetry is the first
qualitative signal: the BCP lies close to the less electronegative partner,
and its distance to each nucleus defines the *bonded radius* of that atom
in this bond. The density at the BCP, $\rho_b = 0.356$ e/bohr$^3$, is high
(a hydrogen bond has $\rho_b \approx 0.01$ to $0.04$, a C-C single bond
about $0.25$), and its Laplacian $\nabla^2\rho_b = -1.94$ e/bohr$^5$ is
strongly negative: density is locally concentrated at the BCP, which is
the QTAIM signature of a shared-shell, covalent interaction. The
ellipticity $\varepsilon = 0.024$ is almost zero, so the density is
cylindrically symmetric around the bond axis, as it should be for a
$\sigma$ bond with no $\pi$ character.

The nuclear critical points are (3,-3) maxima. The density at the oxygen
nucleus, 291 e/bohr$^3$, and its enormous negative Laplacian are dominated
by the 1s core and carry no bonding information; the values at the
hydrogens (0.39 e/bohr$^3$) are also cusp values and not comparable with
the BCP numbers. Strictly a Gaussian basis has no cusp, so the nuclear
maxima are found at the nuclear positions by construction rather than by
the search (`qtaim_analysis` inserts an NCP at every nucleus it did not
find within 0.5 bohr).

The unit conventions are worth keeping straight. `CriticalPoint.position`
is in Angstrom, `rho` in e/A$^3$ and `laplacian` in e/A$^5$; the script
multiplies by $a_0^3$ and $a_0^5$ to quote atomic units, which is what the
QTAIM literature uses. The QVF section
(`topology/critical_points.json` inside the archive) stores the same
fields in the same units, plus the `bond_paths` list; vibe-view's
terminal mode lists it among the sections it reads (see
[vibe-view terminal mode](../user_guide/vibe_view_terminal_mode.md)).

Each `BondPath` holds the two atoms it connects and the traced `path`
polyline in Angstrom. The tracer leaves the BCP along plus and minus the
eigenvector of the one positive Hessian eigenvalue and follows steepest
ascent on $\rho$ to the two nuclei, so the polyline starts on
`atoms[0]`, passes through the BCP and ends on `atoms[1]`. On water each
bond path has 22 points and a length of 0.9579 A, equal to the 0.9578 A
nuclear separation to 1e-4 A: the O-H bonds are straight. A strained or
bent bond shows up as a bond-path length longer than the internuclear
distance.

## Variation: covalent, polar, ionic and pi bonds

The value of the Laplacian at the BCP is the classic QTAIM discriminator.
The loop below runs the same analysis on four diatomics and one alkene,
all at RHF/6-31G*, and prints only the bond critical points.

```python
import numpy as np
import vibeqc as vq
from vibeqc.qtaim import qtaim_analysis

ANG = 1.8897259886
BOHR = 0.529177210903

def bcps(mol, label):
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    top = qtaim_analysis(res, basis, mol)
    for cp in top.critical_points:
        if cp.type == "bcp":
            print(f"{label:6s} {str(cp.atom_pair):8s} rho_b={cp.rho*BOHR**3:.4f}"
                  f"  lap_b={cp.laplacian*BOHR**5:+.4f}  eps={cp.ellipticity:.3f}")

bcps(vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5957 * ANG])]), "LiH")
bcps(vq.Molecule([vq.Atom(9, [0, 0, 0]), vq.Atom(1, [0, 0, 0.917 * ANG])]), "HF")
bcps(vq.Molecule([vq.Atom(7, [0, 0, 0]), vq.Atom(7, [0, 0, 1.098 * ANG])]), "N2")

# Ethylene, D2h: r(CC) = 1.339 A, r(CH) = 1.087 A, HCH = 117.4 deg.
half_cc, r_ch, half = 1.339 * ANG / 2, 1.087 * ANG, np.deg2rad(117.4 / 2)
sy, cz = r_ch * np.sin(half), r_ch * np.cos(half)
c2h4 = vq.Molecule([
    vq.Atom(6, [0, 0, -half_cc]), vq.Atom(6, [0, 0, half_cc]),
    vq.Atom(1, [0,  sy, -half_cc - cz]), vq.Atom(1, [0, -sy, -half_cc - cz]),
    vq.Atom(1, [0,  sy,  half_cc + cz]), vq.Atom(1, [0, -sy,  half_cc + cz]),
])
bcps(c2h4, "C2H4")
```

```
LiH    (0, 1)   rho_b=0.0368  lap_b=+0.1562  eps=0.000
HF     (1, 0)   rho_b=0.3512  lap_b=-2.6609  eps=0.000
N2     (0, 1)   rho_b=0.6806  lap_b=-2.4681  eps=0.000
C2H4   (3, 0)   rho_b=0.2792  lap_b=-1.0137  eps=0.013
C2H4   (5, 1)   rho_b=0.2792  lap_b=-1.0137  eps=0.013
C2H4   (0, 1)   rho_b=0.3483  lap_b=-1.0911  eps=0.445
C2H4   (2, 0)   rho_b=0.2792  lap_b=-1.0137  eps=0.013
C2H4   (4, 1)   rho_b=0.2792  lap_b=-1.0137  eps=0.013
```

Read the table row by row. LiH is the ionic case: a small $\rho_b$ of
0.037 and a *positive* Laplacian, density depleted at the BCP and
concentrated in the two closed shells on either side, the closed-shell
interaction of a Li$^+$ and an H$^-$. HF and N$_2$ are shared-shell
interactions with large $\rho_b$ and negative Laplacians; the triple bond
of N$_2$ carries almost twice the density of the single O-H bond of water.
Ethylene shows what the ellipticity is for: the four C-H bonds have
$\varepsilon \approx 0.01$, but the C=C bond has $\varepsilon = 0.445$,
because the $\pi$ density flattens the bond cross-section into an ellipse
whose major axis lies in the $\pi$ plane. The Poincare-Hopf count for
ethylene is again $6 - 5 + 0 - 0 = 1$. The five analyses together take
about seven seconds; the ethylene grid (36 basis functions, a bigger box)
accounts for half of that.

Two knobs matter when a search misses a critical point. `grid_spacing`
(bohr) sets the seed density; halving it to 0.1 costs eight times the
grid work (water goes from one to four seconds) and on water finds the
same two BCPs to four digits. `gradient_tol` (default 0.2) is the
$|\nabla\rho|$ ceiling for a voxel to count as a seed; raise it for
diffuse, weak interactions such as hydrogen bonds, where the gradient
minimum is shallow, and lower `cp_rho_floor` if a BCP with
$\rho_b < 10^{-4}$ is being discarded as vacuum.

## Theory

QTAIM starts from a partition of real space rather than of the
wavefunction. The gradient vector field $\nabla\rho(\mathbf r)$ is
integrated from every point; almost every trajectory ends at a nucleus,
and the set of points whose trajectories end at a given nucleus is that
atom's *basin*. Basins are separated by surfaces that no gradient path
crosses, the **zero-flux surfaces**,

$$
\nabla\rho(\mathbf r)\cdot\mathbf n(\mathbf r) = 0
\quad\text{for every }\mathbf r\text{ on the surface},
$$

where $\mathbf n$ is the surface normal. Bader showed that a region
bounded by a zero-flux surface satisfies its own virial theorem, which is
what makes the basin a physically meaningful "atom in the molecule" and
its integrated charge and energy well defined. vibe-qc does not yet
integrate basins; this page is about the critical points that organise
them.

A **critical point** is a point where the gradient vanishes,
$\nabla\rho(\mathbf r_c) = 0$. It is characterised by the eigenvalues
$\lambda_1 \le \lambda_2 \le \lambda_3$ of the Hessian
$H_{ij} = \partial^2\rho/\partial x_i\partial x_j$ at $\mathbf r_c$,
written as a pair (rank, signature), the rank being the number of
non-zero eigenvalues and the signature the sum of their signs:

| Signature | Name | Meaning |
|---|---|---|
| $(3,-3)$ | nuclear critical point, `ncp` | local maximum; a nucleus (in rare cases a non-nuclear attractor) |
| $(3,-1)$ | bond critical point, `bcp` | maximum in two directions, minimum along the third: the saddle between two bonded atoms |
| $(3,+1)$ | ring critical point, `rcp` | minimum in the ring plane, maximum perpendicular to it |
| $(3,+3)$ | cage critical point, `ccp` | local minimum, enclosed by rings |

`_classify_cp` in `vibeqc.qtaim` does exactly this count of negative
eigenvalues. For a molecule the counts obey the Poincare-Hopf relation
$n - b + r - c = 1$; for a periodic system the right-hand side is 0.

The eigenvector of the single positive eigenvalue $\lambda_3$ at a BCP
points along the bond. Following $\nabla\rho$ uphill from the BCP in the
$\pm\mathbf v_3$ directions traces the two halves of the **bond path**,
the line of maximum density linking the two nuclei; a bond path exists
between two atoms exactly when a BCP exists between them, which is
QTAIM's operational definition of a chemical bond. The two negative
eigenvalues measure how fast the density falls off perpendicular to the
bond, and their ratio defines the **ellipticity**

$$
\varepsilon = \frac{\lambda_1}{\lambda_2} - 1 \;\ge 0 ,
$$

which is zero for a cylindrically symmetric $\sigma$ bond and grows with
$\pi$ character (0.45 for the C=C bond above, and it would be zero again
for a triple bond, whose two $\pi$ systems restore the symmetry).

The **Laplacian** $\nabla^2\rho = \lambda_1 + \lambda_2 + \lambda_3$ says
whether density is locally concentrated ($\nabla^2\rho < 0$) or depleted
($\nabla^2\rho > 0$) relative to its surroundings. Its physical content
comes from the local form of the virial theorem,

$$
\tfrac{1}{4}\nabla^2\rho(\mathbf r) = 2\,G(\mathbf r) + V(\mathbf r),
$$

with $G \ge 0$ the kinetic-energy density and $V \le 0$ the potential-energy
density (atomic units). Where $\nabla^2\rho_b < 0$ the potential energy
dominates and electrons are shared between the nuclei: covalent and polar
bonds (water, HF, N$_2$, ethylene above). Where $\nabla^2\rho_b > 0$ the
kinetic energy dominates, the density is squeezed out of the
internuclear region into the atomic shells, and the interaction is
closed-shell: ionic bonds (LiH), hydrogen bonds and van der Waals
contacts. The magnitude of $\rho_b$ then sorts the closed-shell cases by
strength, and for bonds within one family $\rho_b$ correlates well with
bond order and inversely with bond length.

Two limitations frame the interpretation. First, the topology is a
property of $\rho$ alone, so it inherits nothing from the orbital picture
but also cannot separate a bond's $\sigma$ and $\pi$ components beyond
what the ellipticity encodes. Second, the numbers depend on the quality
of $\rho$: a small basis or a poor functional shifts $\rho_b$ and
$\nabla^2\rho_b$ by amounts that matter near the covalent/closed-shell
boundary (the Laplacian sign of polar bonds such as C-O or Si-O is known
to be delicate), so compare values within one level of theory and one
basis family.

## Resources

- [QTAIM user guide](../user_guide/qtaim.md): the data types, algorithm
  sketch and the location of every piece in the code.
- [Bond analysis](../user_guide/bond_analysis.md) and
  [post-SCF properties](post_scf_properties.md): Mulliken, Lowdin and
  Mayer descriptors that partition the wavefunction rather than real
  space, for comparison with the density-based picture here.
- [Volumetric data](../user_guide/volumetric_data.md): export $\rho$ as a
  cube file to look at the surfaces the critical points organise.
- [QVF file format](qvf_file_format.md): where `topology.qtaim` sits among
  the section kinds.
- [Feature matrix](../features.md) and
  [Methods: theory and status](../methods.md#electronic-structure-of-solids-bands-dos-coopcohp-bond-orders-qtaim):
  the status row and the one-paragraph summary.
- Tests: `tests/test_qtaim.py` pins the water and ethylene topologies used
  on this page.

Cost: the grid evaluation dominates and scales with the number of voxels
times the number of basis functions; water at 0.15 bohr is about one
second, ethylene in 6-31G* about four. Memory is the grid arrays plus the
$3\times3$ Hessian per voxel, evaluated in chunks of $10^5$ points.

## References

- **QTAIM.** R. F. W. Bader, "Atoms in molecules," *Acc. Chem. Res.*
  **18**, 9 (1985), {vibeqc-cite}`bader_qtaim_1985`; and R. F. W. Bader,
  *Atoms in Molecules: A Quantum Theory* (Oxford University Press, 1990),
  the full account of zero-flux surfaces, critical points and the atomic
  virial theorem.
- **Classification of interactions.** R. F. W. Bader and H. Essen,
  "The characterization of atomic interactions," *J. Chem. Phys.* **80**,
  1943 (1984). Shared-shell versus closed-shell interactions from the sign
  of the Laplacian; the local virial relation used above.
- **Bond ellipticity and pi character.** R. F. W. Bader, T. S. Slee,
  D. Cremer and E. Kraka, "Description of conjugation and hyperconjugation
  in terms of electron distributions," *J. Am. Chem. Soc.* **105**, 5061
  (1983).
- **Reference implementation.** T. Lu and F. Chen, "Multiwfn: A
  multifunctional wavefunction analyzer," *J. Comput. Chem.* **33**, 580
  (2012). The seed-and-refine critical-point search that
  `vibeqc.qtaim` follows.

## Next

- [Post-SCF properties](post_scf_properties.md) computes the
  orbital-space bond orders and charges you will want to place beside
  the BCP descriptors.
- [Natural orbitals](natural_orbitals.md) shows the other basis-independent
  view of a correlated density, its natural occupation numbers.
- [Orbital visualization](orbital_visualization.md) exports the density
  and orbitals to cube files for vibe-view, where the bond critical points
  can be placed on the isosurfaces.
