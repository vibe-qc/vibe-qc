---
myst:
  html_meta:
    "description": "Tutorial: space-group reduction of the exact-exchange build on a multi-k GDF SCF with symmetry_reduce_k=True. Rock-salt LiH on 2x2x2 and 3x3x3 meshes: stars and weights, the irreducible wedge, energies equal to the full mesh, the cost saved, the state gate and the refusals, with the group theory behind it."
    "og:title": "vibe-qc tutorial: symmetry-reduced k meshes for exact exchange"
---

# Symmetry-reduced k meshes for exact exchange

A Hartree-Fock or hybrid SCF on a k mesh spends most of its time on exact
exchange, the one term whose cost grows with the square of the number of
k points. This tutorial switches on the space-group reduction of that term,
`symmetry_reduce_k=True`, on rock-salt LiH. You will list the stars of the
mesh and their weights, run the same SCF with and without the reduction,
check that the energies agree to roundoff, read what the `.out` reports, and
meet the requests the route refuses. The
[multi-k SCF guide](../user_guide/multi_k_scf.md#space-group-reduction-of-the-exchange-build)
is the reference for every option.

```{admonition} Status: preview
:class: warning

`symmetry_reduce_k=True` (driver keyword `ibz_native=True`) is reachable,
maintained and exact for the finite fit it reduces: each admitted operation
is checked against the executed exchange source, and the run fails closed
whenever the mesh, the source or the electronic state does not permit the
reduction. The non-symmorphic admission and the per-star occupation fill
shipped in v0.17.15 without independent fleet acceptance, so the route is
labelled preview here. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**Prerequisites.** The [k-point tutorial](kpoints_brillouin_bloch.md) for
meshes and Bloch sums, and the [LiH multi-k tutorial](lih_multi_k.md) for a
first periodic GDF run. Run the scripts from an empty scratch directory.

The numbers on this page were measured with the vibe-qc 0.17.17
development build (main, 2026-10-14) on a laptop shared with other jobs
(load average about 50 on 18 cores), so the wall times show the ratio
between the runs, not what your machine will take.

## The crystal and its stars

Rock-salt LiH has space group $Fm\bar 3m$ (number 225), with 48 point
operations. Attach the operations with spglib and ask the mesh for its
stars:

```python
import numpy as np
import vibeqc as vq

ANG = 1.0 / 0.529177210903
a = 4.084 * ANG                                   # bohr
lattice = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a
lih = vq.PeriodicSystem(3, lattice,
                        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [a / 2] * 3)])

vq.attach_symmetry(lih)
print(lih.symmetry)
for mesh in [(2, 2, 2), (3, 3, 3)]:
    kp = vq.KPoints.gamma_centred(lih, list(mesh)).symmetry_reduce()
    print(mesh, kp.n_kpoints, "irreducible of", kp.full_mesh_size)
    for f, w in zip(kp.kpoints_frac, kp.weights):
        print(f"   k = ({f[0]:+.4f}, {f[1]:+.4f}, {f[2]:+.4f})"
              f"   w = {round(w * kp.full_mesh_size)}/{kp.full_mesh_size}")
```

```text
SpaceGroup(number=225, symbol='Fm-3m', order=48)
(2, 2, 2) 3 irreducible of 8
   k = (+0.0000, +0.0000, +0.0000)   w = 1/8
   k = (+0.5000, -0.0000, -0.0000)   w = 4/8
   k = (+0.5000, +0.5000, -0.0000)   w = 3/8
(3, 3, 3) 4 irreducible of 27
   k = (+0.0000, +0.0000, +0.0000)   w = 1/27
   k = (+0.3333, -0.0000, -0.0000)   w = 8/27
   k = (+0.3333, +0.3333, -0.0000)   w = 6/27
   k = (-0.3333, +0.3333, +0.0000)   w = 12/27
```

Every mesh point belongs to exactly one star, the set of points that the
group's operations map onto each other. A star is represented by one point,
and its weight is the fraction of the mesh it covers. On (2, 2, 2), 8 points
fall into 3 stars. On (3, 3, 3), 27 points fall into 4 stars, one of them
holding 12 points. The weights in each list sum to one.

## Run the SCF twice

The reduction is a flag on `run_periodic_job`. It needs a symmetry model,
which `symmetry="attach"` builds on the system:

```python
import time

basis = vq.BasisSet(lih.unit_cell_molecule(), "sto-3g")
common = dict(method="RHF", jk_method="gdf", kpoints=(2, 2, 2),
              aux_basis="def2-svp-jk", conv_tol_energy=1e-8,
              output_qvf=False)

t0 = time.perf_counter()
full = vq.run_periodic_job(lih, basis, output="lih_full", **common)
t_full = time.perf_counter() - t0

t0 = time.perf_counter()
ibz = vq.run_periodic_job(lih, basis, output="lih_ibz",
                          symmetry="attach", symmetry_reduce_k=True, **common)
t_ibz = time.perf_counter() - t0

print(f"full  E = {full.energy:.10f} Ha  {full.n_iter} it  {t_full:.0f} s")
print(f"wedge E = {ibz.energy:.10f} Ha  {ibz.n_iter} it  {t_ibz:.0f} s")
print(f"dE = {ibz.energy - full.energy:+.1e} Ha")
```

| Mesh | Run | E per cell, Ha | Iterations | Exchange bras | `Lpq` pairs | Wall, s |
|---|---|---:|---:|---:|---:|---:|
| (2, 2, 2) | full mesh | -7.9220351208 | 7 | 8 | 64 | 1462 |
| (2, 2, 2) | wedge | -7.9220351208 | 7 | 3 | 24 | 967 |
| (3, 3, 3) | full mesh | -7.9484183256 | 7 | 27 | 729 | |
| (3, 3, 3) | wedge | -7.9484183256 | 7 | 4 | 108 | 4462 |

The two (2, 2, 2) energies agree to all printed digits (6.7e-13 Ha in an
earlier measurement on the 0.17.15 build), which is roundoff, and the SCF
takes the same seven iterations. The wedge run was 1.5 times faster here and
3.1 times faster in that earlier measurement on a more heavily loaded
machine: on a shared laptop the wall time is a noisy measure. The robust
measure is the exchange bookkeeping: the bra count goes 8 to 3, and the
density-fitted three-centre cache that holds $K$'s pairs 64 to 24. On the
(3, 3, 3) mesh it goes 27 to 4 bras and 729 to 108 pairs, a factor of 6.75
in the dominant term, so the saving grows with the mesh. The (3, 3, 3) wedge
energy equals the full-mesh value to every printed digit; the full-mesh
row's energy comes from the 0.17.15 build, and its wall time on this machine
was not measured.

The `.out` of the wedge run states the reduction above the memory block:

```text
    symmetry_reduce_k   = exchange bras 8 -> 3 (irreducible wedge)
                          Lpq cderi pairs 64 -> 24 (2.7x fewer)
                          exact: K is symmetry-transported over each star
```

## What is reduced, and what is not

Only the exchange build is reduced. For each star the exchange matrix
$\mathbf K(\mathbf k)$ is built at the representative and carried to every
other member by the operation that maps one onto the other. The Fock matrix
is then assembled, diagonalised and occupied at every point of the full
mesh, and the energy is summed over the full mesh. That is why the energy
equals the unreduced one to roundoff: nothing is approximated, the same
numbers are computed fewer times.

Two checks make this safe:

- **Admission of each operation.** An operation is used only if the
  executed exchange source transforms correctly under it. On the default
  range-separated fit every operation of the group is admitted, glides and
  screws included. On the classic fit only operations that move every atom
  by the same lattice vector are admitted. Other sources are refused.
- **The state gate.** In every iteration each star member is compared with
  its transported representative, and each representative with its own
  image under every operation that fixes it (its little group), time
  reversal included. A density that breaks the symmetry it is supposed to
  carry is refused with the little group named, instead of being
  symmetrised silently.

Occupations are filled once per star, on the representatives with the
star's weight, and copied to the members. Levels that differ by less than
the measured spread within a star, or by less than a nanohartree, count as
degenerate. A symmetry-degenerate frontier therefore receives a symmetric
fractional occupation rather than an integer split decided by lattice-sum
noise.

## What the route refuses

Each precondition fails closed with a message that names the remedy:

```text
ValueError: run_periodic_job: symmetry_reduce_k requires an attached
symmetry model -- there is no irreducible wedge without one. Pass
symmetry='attach' (or attach validated operations to the PeriodicSystem)
alongside symmetry_reduce_k=True.

NotImplementedError: run_periodic_job: symmetry_reduce_k needs a true
multi-k Monkhorst-Pack mesh to reduce; this job samples 1 k-point(s). At
Gamma the wedge is the whole mesh and there is nothing to save.

NotImplementedError: ibz_native=True has no effect without exact exchange
(alpha == 0): the n_k^2 term it reduces is the HF exchange. Drop the flag
for a pure functional.

NotImplementedError: ibz_native=True cannot be combined with
bz_integration='gilat': the Gilat-Raubenheimer fill interpolates over the
full mesh, while the wedge-native state takes its occupations once per
star. Use the zero-temperature or smearing fill, or run without ibz_native.
```

The third one is worth reading twice. A pure functional such as PBE has no
exact exchange, so there is no $n_k^2$ term to reduce and the flag would do
nothing; vibe-qc says so instead of accepting it silently. Hybrids such as
PBE0 are reduced.

## Theory

**Stars and the irreducible wedge.** A space-group operation
$\{\mathbf R|\boldsymbol\tau\}$ maps a crystal momentum $\mathbf k$ onto
$\mathbf R\mathbf k$ modulo a reciprocal-lattice vector, and Bloch states at
the two points are related by a unitary transformation of the basis
functions. The star of $\mathbf k$ is the set $\{\mathbf R\mathbf k\}$ over
the point group, and the little group of $\mathbf k$ is the subgroup that
leaves $\mathbf k$ fixed. A mesh sum of a symmetric quantity then reduces to
a weighted sum over one representative per star,

$$
\frac{1}{N_k}\sum_{\mathbf k} f(\mathbf k)
= \sum_{\mathbf k \in \mathrm{IBZ}} w_{\mathbf k}\, f(\mathbf k),
\qquad
w_{\mathbf k} = \frac{|\mathrm{star}(\mathbf k)|}{N_k}
= \frac{|G|}{N_k\,|G_{\mathbf k}|},
$$

with $|G|$ the order of the point group and $|G_{\mathbf k}|$ that of the
little group, which is the counting the weights above follow (Gamma has the
whole group as its little group and weight $1/N_k$). The meshes and their
reduction follow {vibeqc-cite}`monkhorst_pack_1976`; the operations come
from spglib ({vibeqc-cite}`togo_shinohara_tanaka_2024`).

**Transporting exchange.** In an atom-centred Gaussian basis an operation
maps each basis function onto a function on the image atom, possibly in a
neighbouring cell, combined with the other components of its shell through
a rotation matrix $\mathbf D(\mathbf R)$. For the exchange matrix this gives

$$
\mathbf K(\mathbf R\mathbf k) = \mathbf U_{\mathbf R}(\mathbf k)\,
\mathbf K(\mathbf k)\, \mathbf U_{\mathbf R}(\mathbf k)^{\dagger},
$$

where $\mathbf U_{\mathbf R}(\mathbf k)$ collects the shell rotations, the
atom permutation and the Bloch phase of the lattice vector each atom image
is shifted by. In vibe-qc's atom-centred Bloch convention the fractional
translation of a glide or screw enters only through those image shifts, so
no separate $e^{-i\mathbf k\cdot\boldsymbol\tau}$ factor appears. The same
construction underlies the symmetry-adapted crystalline orbitals of
CRYSTAL ({vibeqc-cite}`dovesi_symmetry_lcao_1986`,
{vibeqc-cite}`zicovich_wilson_crystalline_orbitals_1998`).

**Why exchange.** In a multi-k GDF SCF the Coulomb matrix needs only the
density summed over the mesh, but exchange couples every pair
$(\mathbf k, \mathbf k')$ through the fitted three-centre integrals at the
momentum transfer $\mathbf k - \mathbf k'$. Its cost and the size of the
`Lpq` cache are proportional to $N_k^2$; building it only for the
representatives makes them proportional to $N_{\mathrm{IBZ}} N_k$.

## Variations

- **A hybrid functional.** `method="RKS", functional="pbe0"` is reduced in
  the same way; the exact-exchange fraction is what the wedge saves.
- **Open shell.** `method="UHF"` with a triplet cell reduces both spin
  exchange builds, with the per-star occupation fill in each channel.
- **Larger meshes.** The saving grows with the mesh: on (4, 4, 4) LiH has 8
  stars out of 64 points.

## Pitfalls

- **Do not expect a saving for pure functionals.** The flag is refused; use
  an ordinary mesh.
- **Low-symmetry cells save little.** A triclinic cell has only inversion
  (from time reversal), which roughly halves the mesh.
- **A symmetry-broken state is refused, not fixed.** If your crystal really
  distorts, an antiferromagnet or a Peierls cell, run without the reduction,
  or describe the lower symmetry in the cell itself.
- **Meshes passed as reduced `KPoints`** to the GDF drivers are expanded to
  the full mesh again: that path checks the reduction but saves nothing. The
  saving comes only from `symmetry_reduce_k`.

## See also

- [Multi-k SCF](../user_guide/multi_k_scf.md#space-group-reduction-of-the-exchange-build),
  the option reference and every refusal.
- [Periodic methods](../user_guide/periodic_methods.md), the transport
  convention and the measured residuals.
- [Symmetry status](../symmetry_status.md), what the shared symmetry layer
  implements and what stays gated.
- [k-points and Bloch functions](kpoints_brillouin_bloch.md).
