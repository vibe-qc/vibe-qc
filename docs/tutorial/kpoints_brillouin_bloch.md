# k-points, the Brillouin zone, and Bloch phase

A periodic calculation never builds the infinite crystal. It solves a
single unit cell once for each point $\mathbf{k}$ in a small region of
*reciprocal* space, and a band index together with $\mathbf{k}$ labels
every electronic state. This tutorial is the conceptual companion to the
practical [k-point mesh guide](../user_guide/k_points.md) and the
[band-structure tutorial](band_structure.md): it explains what a
k-point *is*, why states carry a wavevector at all, and how to choose a
mesh. There is no code to run here, but one interactive figure to drag.

## Real space and reciprocal space

A crystal is defined by its lattice vectors $\mathbf{a}_1, \mathbf{a}_2,
\mathbf{a}_3$: translating the whole structure by any
$\mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3$
(integer $n_i$) maps it onto itself. The **reciprocal lattice**
$\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ is the dual basis defined by

$$
\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\,\delta_{ij}.
$$

Reciprocal space has units of inverse length and is the Fourier dual of
the real lattice. That duality has a consequence worth stating plainly: a
single k-point is **not** a place in the unit cell. A plane wave
$e^{i\mathbf{k}\cdot\mathbf{r}}$ is spread over *all* of real space;
fixing $\mathbf{k}$ picks one such wave, not one atom or one position.
k-points are geometric points in reciprocal space, and nothing else. The
common mental slip of imagining a k-point somewhere "inside the cell" is
the single most useful misconception to drop before reading on.

## Bloch's theorem

Because the Hamiltonian commutes with every lattice translation, its
eigenstates can be chosen to be simultaneous eigenstates of translation.
That is Bloch's theorem:

$$
\psi_{n,\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}}\,
u_{n,\mathbf{k}}(\mathbf{r}),
\qquad u_{n,\mathbf{k}}(\mathbf{r}+\mathbf{R}) = u_{n,\mathbf{k}}(\mathbf{r}),
$$

a plane wave $e^{i\mathbf{k}\cdot\mathbf{r}}$ times a cell-periodic part
$u_{n,\mathbf{k}}$. Every state carries two labels: a band index $n$
(which orbital manifold) and a wavevector $\mathbf{k}$ (the plane-wave
phase). This is what reduces an infinite crystal to a finite problem: you
solve the cell once per $\mathbf{k}$ instead of diagonalising an infinite
matrix. vibe-qc constructs these Bloch states by lattice (Bloch)
summation of the unit-cell integrals; see
[periodic methods](../user_guide/periodic_methods.md).

## The Brillouin zone and the Γ point

$\mathbf{k}$ is unique only up to a reciprocal lattice vector
$\mathbf{G}$: $\mathbf{k}$ and $\mathbf{k}+\mathbf{G}$ label the *same*
state, because $e^{i\mathbf{G}\cdot\mathbf{R}} = 1$ for every lattice
vector $\mathbf{R}$. The set of inequivalent $\mathbf{k}$ is the **first
Brillouin zone (BZ)**, the reciprocal-space analogue of the unit cell
(formally, the Wigner-Seitz cell of the reciprocal lattice). The zone
centre, $\mathbf{k} = (0,0,0)$, is **Γ**: the fully in-phase state, the
one a Γ-only calculation solves.

## k-points as a phase label

The single most useful thing to fix in your head is what a wavevector
actually labels: not the contents of a cell, but the phase relating one
cell to the next.

```{important}
Every unit cell is identical and holds the same orbital with the same
amplitude, so the density $|\psi|^2$ is periodic. A k-point does not
change *what* sits in a cell, only the **phase** carried from one cell to
the next. That phase, and nothing else, is what a wavevector encodes.
```

Concretely, the cell-to-cell relation is

$$
\psi(\mathbf{r}+\mathbf{R}) = e^{i\mathbf{k}\cdot\mathbf{R}}\,\psi(\mathbf{r}),
$$

so stepping one cell along $\mathbf{a}$ multiplies the wavefunction by a
fixed factor $e^{i\mathbf{k}\cdot\mathbf{a}}$. The amplitude repeats; the
phase *winds* by $\mathbf{k}\cdot\mathbf{a}$ per cell. The mechanical
analogue is a line of identical coupled pendulums, or the atoms in a
phonon mode: every pendulum is the same and swings with the same
amplitude, and a travelling wave lives entirely in the steady phase lag
from one pendulum to the next.

```{raw} html
<iframe src="../_static/widgets/bloch_phase_winding.html"
        title="Bloch phase winding across the unit cells"
        style="width:100%; height:600px; border:1px solid #e2e5ea; border-radius:12px;"
        loading="lazy"></iframe>
```

*Drag $k$ from Γ (the left preset) to the zone boundary and watch two
things. The dashed amplitude envelope stays identical in every cell,
while only the phasor angle winds. At Γ ($k = 0$) all cells are in phase,
the all-bonding combination, wavelength $\lambda \to \infty$. At the
boundary ($k = \pi/a$) the phase flips every cell, the all-antibonding
combination, $\lambda = 2$ cells.*

How far you walk before the pattern repeats is set by
$\mathbf{k}\cdot\mathbf{a}$. Write $k\,a / 2\pi = p/q$ in lowest terms:
the pattern repeats every $q$ cells. If $k\,a / 2\pi$ is irrational it
*never* repeats, an **incommensurate** state. In a finite crystal of $N$
cells with Born-von Karman boundary conditions (wrap the chain into a
ring), the allowed wavevectors are quantised, $k = 2\pi m/(Na)$, which
are always rational and repeat every $M = N/\gcd(m, N)$ cells.

## Why the phase matters

The phase winding is not bookkeeping. It is what turns a set of standing
atomic states into propagating waves:

* **Crystal momentum and group velocity.** A Bloch state carries crystal
  momentum $\hbar\mathbf{k}$ and moves with group velocity
  $\mathbf{v} = \tfrac{1}{\hbar}\,\nabla_{\mathbf{k}} E(\mathbf{k})$. Flat
  bands are slow and heavy; steep bands are fast and light.
* **It generates the band.** For one orbital per cell with
  nearest-neighbour coupling $\beta$, the energy is
  $E(k) = \alpha + 2\beta\cos(ka)$: bonding at Γ ($\cos 0 = +1$), rising
  to antibonding at the zone edge ($\cos\pi = -1$). The dispersion *is*
  the phase running through the cosine. The
  [band-structure tutorial](band_structure.md) plots this exact
  H-chain band.
* **Selection rules and diffraction.** Conservation of crystal momentum
  up to a $\mathbf{G}$ is the periodic version of a selection rule, and
  the Laue / Bragg diffraction condition is $\Delta\mathbf{k} =
  \mathbf{G}$.
* **Geometric phase.** How the phase of $u_{n,\mathbf{k}}$ twists as
  $\mathbf{k}$ traverses the BZ is a Berry phase, the origin of the
  modern theories of electric polarization and of band topology.

## High-symmetry points and their names

The labels Γ, X, L, K, W, and the rest are not arbitrary. They come from
the group theory of the crystal (the Bouckaert-Smoluchowski-Wigner naming
of 1936). A bare letter marks a $\mathbf{k}$ whose **little group**, the
symmetry operations that leave $\mathbf{k}$ fixed (modulo a $\mathbf{G}$),
is unusually large. A subscript, as in $\Gamma_{25}'$ or $X_1$, names an
**irreducible representation** of that little group: it says how the
bands at that point transform under the symmetry, not just where they
sit. By convention, Greek letters label points in the *interior* of the
zone and Roman letters label points on its *surface*.

This has a direct, checkable consequence. A band's degeneracy at a
$\mathbf{k}$-point cannot exceed the dimension of the largest irrep of
its little group. Move off a high-symmetry point along a path and the
little group shrinks, so a multiplet must split into the irreps the
smaller group allows. The textbook example: a triply degenerate
$\Gamma_{25}'$ level splits into $\Delta_5 + \Delta_2'$ as you walk along
the $[100]$ direction. vibe-qc's crystal-symmetry layer (see
[symmetry storage](symmetry_storage.md)) is what detects the little
group and reduces the sampling to the irreducible $\mathbf{k}$-set.

## Sampling the Brillouin zone: Monkhorst-Pack

Every observable, the total energy, the density, the density of states,
is an *integral* over the BZ. In practice that integral is approximated
by a weighted sum over a finite mesh of $\mathbf{k}$-points. The standard
construction is the **Monkhorst-Pack** grid: a uniform
$N_1 \times N_2 \times N_3$ mesh in fractional reciprocal coordinates.

There is an exact correspondence worth internalising: an $N \times N
\times N$ k-mesh is *identical* to imposing Born-von Karman periodicity
on an $N \times N \times N$ **supercell** in real space. Mesh density and
real-space cell size therefore trade off inversely,

$$
(\text{k-mesh density}) \times (\text{real-space cell size}) \approx \text{const},
$$

so a large supercell has a small BZ, and for a large enough cell a single
$\Gamma$ point suffices. This is exactly why a molecule in a big box, or
vibe-qc's cyclic-cluster cells, can run Γ-only, while a one-atom
primitive metal needs a dense mesh.

## Choosing the mesh in practice

There is **no closed-form formula** that gives you the mesh from the
space group and atomic positions alone. The density you need is set by
the *electronic structure*, mainly whether the system is a metal or an
insulator: a metal has a Fermi surface, which makes the BZ integrand
discontinuous and demands a fine mesh, plus smearing (see
[Fermi-Dirac smearing](fermi_dirac_smearing.md)).

The workable recipe:

1. Pick a target k-spacing $\Delta k$ and set
   $N_i = \lceil\, |\mathbf{b}_i| / \Delta k\,\rceil$ along each
   reciprocal axis. Anisotropic cells get anisotropic meshes for free
   this way.
2. Confirm with a convergence ladder, $2^3 \to 4^3 \to 6^3 \to 8^3$, and
   take the coarsest mesh whose total energy sits within tolerance of the
   next finer one (a common target is 1 meV/atom).

Rough starting $\Delta k$ targets:

| System | $\Delta k$ ($\text{Å}^{-1}$) |
|--------|------------------------------|
| Insulator / wide-gap semiconductor | 0.25 to 0.35 |
| Small-gap semiconductor | 0.15 to 0.20 |
| Metal | 0.10 to 0.15 (plus smearing) |

Use a $\Gamma$-centred grid for hexagonal cells, since an off-Γ mesh
breaks the hexagonal symmetry. vibe-qc's **AUTO** k-point mode applies
exactly these criteria from a requested spacing; the practical knobs,
including an explicit k-mesh and the AUTO selector, are in the
[k-point mesh guide](../user_guide/k_points.md).

## References

- **Bloch's theorem.** F. Bloch, "Über die Quantenmechanik der Elektronen
  in Kristallgittern," *Z. Phys.* **52**, 555 (1929).
- **Cyclic boundary conditions.** M. Born and Th. von Kármán, "Über
  Schwingungen in Raumgittern," *Phys. Z.* **13**, 297 (1912): the
  finite-crystal periodicity that quantises the allowed wavevectors into a
  discrete mesh, with as many k-points as there are cells.
- **High-symmetry point names.** L. P. Bouckaert, R. Smoluchowski, and E.
  Wigner, "Theory of Brillouin Zones and Symmetry Properties of Wave
  Functions in Crystals," *Phys. Rev.* **50**, 58 (1936).
- **k-mesh sampling.** H. J. Monkhorst and J. D. Pack, "Special points for
  Brillouin-zone integrations," *Phys. Rev. B* **13**, 5188 (1976).
- **Textbook.** N. W. Ashcroft and N. D. Mermin, *Solid State Physics*:
  chapters 8 to 9 (Bloch states, the BZ) and 22 (group theory of
  crystals).
- **Modern reference.** R. M. Martin, *Electronic Structure: Basic Theory
  and Practical Methods*, 2nd ed. (2020): chapter 4 (periodic systems),
  chapter 10 (Bloch representation), chapter 14 (k-space integration).

## Next

With the concept in hand, the practical tutorials are
[band structure and DOS](band_structure.md) (sampling a k-path and a
k-mesh), [multi-k LiH](lih_multi_k.md) (a full multi-k SCF), and
[periodic SCF convergence](periodic_scf_convergence.md) (converging the
self-consistent solution at each k). For the underlying machinery, the
[k-point mesh guide](../user_guide/k_points.md),
[multi-k SCF](../user_guide/multi_k_scf.md), and
[periodic methods](../user_guide/periodic_methods.md) cover how vibe-qc
builds and samples Bloch states.
