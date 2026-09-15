# Natural orbitals and the idempotency diagnostic

Canonical molecular orbitals (MOs) are the eigenvectors of the Fock
matrix $\mathbf{F}$. They diagonalize the *one-electron Hamiltonian*,
which makes them the right basis for orbital energies, Koopmans'
ionisation potentials, and SCF convergence.

**Natural orbitals (NOs)** are eigenvectors of the *one-particle
reduced density matrix* $\hat\gamma$. They diagonalize the
*observable*, the electron density, which makes them the right
basis for population analysis, multireference diagnostics, and any
question of the form "where do the electrons actually live?"

For a single Slater determinant the two coincide up to within-subspace
rotations. They diverge, informatively, for open-shell, correlated,
or symmetry-broken states. This tutorial shows how to compute and read
NOs in vibe-qc, and how to interpret the **idempotency deviation**
$\Delta$ that quantifies the gap between the actual SCF density and a
clean single-determinant one.

## The basic call

``vq.natural_orbitals(result, basis)`` returns a ``NaturalOrbitals``
dataclass holding occupations and AO-basis NO coefficients, sorted by
descending occupation:

```python
import vibeqc as vq
import numpy as np

mol = vq.Molecule([
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "6-31g*")

result = vq.run_rhf(mol, basis)
no = vq.natural_orbitals(result, basis)

print(no.occupations[:8])
# [2.0  2.0  2.0  2.0  2.0  ~0  ~0  ~0]   <- exactly five at 2, rest at 0
print(vq.idempotency_deviation(no))
# ~1e-14  <- numerically zero: this is a single Slater determinant
```

For a closed-shell H₂O / 6-31G\* single-determinant SCF the result is
unambiguous: five orbitals at occupation exactly 2.0, the rest at
0.0, and $\Delta < 10^{-13}$. The NO coefficient matrix is
S-normalized (``no.coefficients.T @ S @ no.coefficients == I``) and
drop-in compatible with the existing ``write_cube_mo`` / Molden
writers, natural orbitals *are* just another AO-basis MO matrix.

## Open-shell: spin natural orbitals

For UHF or UKS, the total NOs (default ``kind="uhf-total"``) come from
diagonalizing $\mathbf{D}_\alpha + \mathbf{D}_\beta$. The
*spin* NOs (``kind="uhf-spin"``) come from $\mathbf{D}_\alpha -
\mathbf{D}_\beta$ and have occupations in $[-1, 1]$. The two spin-NOs
with the largest $|n_i|$ pick out exactly where the unpaired
electrons live.

```python
o2 = vq.Molecule(
    [vq.Atom(8, [0, 0, +1.14]), vq.Atom(8, [0, 0, -1.14])],
    multiplicity=3,                              # ground-state triplet
)
basis_o2 = vq.BasisSet(o2, "6-31g*")
res_o2   = vq.run_uhf(o2, basis_o2)

no_total = vq.natural_orbitals(res_o2, basis_o2, kind="uhf-total")
no_spin  = vq.natural_orbitals(res_o2, basis_o2, kind="uhf-spin")

print(no_total.occupations[:10])
# [2.000 2.000 1.9999 1.9991 ... 1.000 1.000 0.0067]
#                                ^^^^^^^^^^^ two singly-occupied πg
print(no_spin.occupations[:4])
# [+1.000 +1.000 +0.115 +0.115]
#  ^^^^^^^^^^^^ the two unpaired α electrons in πg*
```

The leading two spin-NOs have occupations exactly +1: those are the
two unpaired α electrons sitting in the $\pi_g^*$ pair. Render either
of them with ``vq.evaluate_ao(basis, points) @ no_spin.coefficients[:, 0]``
and you get the textbook $\pi^*$ shape, the same orbitals chemistry
draws on the back of an envelope, derived directly from the SCF
density.

![Natural orbital occupations for closed-shell H₂O and triplet O₂. Panel (a) H₂O / 6-31G\* RHF, exactly 5 NOs at occupation 2, rest at 0, idempotency deviation ~1e-14. Panel (b) O₂ triplet UHF total NOs, 7 NOs at 2, two at exactly 1 (the unpaired electrons). Panel (c) O₂ triplet UHF spin NOs, two at +1 (the unpaired α electrons in πg\*), the rest near zero with small spin-polarization tail. Panel (d) 2D contour slice of the leading O₂ spin natural orbital showing the textbook πg\* shape with lobes above and below each O atom in the (x, z) plane.](../_static/plots/natural-orbital-occupations.png)

A four-panel anatomy of natural orbitals across the closed-shell /
open-shell boundary. **Panel (a)** confirms a closed-shell HF state is
genuinely a single determinant, every NO occupation is integer to
machine precision. **Panel (b)** shows that even for an open-shell
UHF, $D_\alpha + D_\beta$ has natural occupations clustered at
$\{0, 1, 2\}$: the two NOs at exactly 1 *are* the unpaired electrons.
**Panel (c)** is the same data viewed through the spin density
$D_\alpha - D_\beta$, only the two unpaired electrons have $|n_i|
\approx 1$, the rest are spin-polarization tail. **Panel (d)** plots
the leading spin NO directly: the oxygen $\pi_g^*$ orbital with a
nodal plane at $z = 0$ and lobes pointing $\pm x$ above and below
each O nucleus. Reproduce with
``python3 examples/plots/natural-orbital-occupations.py``.

## Render archived natural orbitals in vibe-view

When a QVF contains a `wavefunction.gto` section whose `orbital_kind` is
`natural`, vibe-view can sample every coefficient row directly. No separate
cube is needed:

```sh
vibe-view tui correlated.qvf
```

Press <kbd>Tab</kbd> until the natural-orbital section is active. The terminal UI
renders its default orbital and opens a surface table on the right. Use
<kbd>Up</kbd>/<kbd>Down</kbd> and <kbd>Enter</kbd> to render a selected natural
orbital, or <kbd>n</kbd>/<kbd>p</kbd> to step through the set. The rows
emphasize occupation rather than a canonical MO energy. When the section
declares `occupation_semantics: electron_occupation`, <kbd>D</kbd> renders the
occupation-weighted total density; <kbd>i</kbd>/<kbd>I</kbd> adjust the contour.

Natural transition orbitals can use the same archive marker, but their
numbers are transition weights rather than electron occupations. The
orbital isosurface remains valid, but a weighted sum is not the ground-state
electron density. The TUI therefore omits **Total density** and refuses
<kbd>D</kbd> for that section. It also refuses density for an older natural
section with no semantic marker: individual orbital surfaces remain valid,
but the values cannot safely be classified by their sum alone. See
[Orbital and density visualization](orbital_visualization.md) for the full
terminal workflow.

## When NOs differ from canonical MOs

For a **single Slater determinant** the natural orbitals span the
same occupied / virtual subspaces as the canonical MOs, the two
sets are related by a within-subspace unitary rotation that the
density doesn't see. NO occupations come out at exactly $\{0, 2\}$
(or $\{0, 1\}$ for per-spin UHF), and $\Delta = 0$ to numerical
precision.

For a **correlated wavefunction** (CASSCF, MP2 natural orbitals,
DLPNO-CCSD natural orbitals, …) the eigenvalues of $\hat\gamma$
spread between 0 and 2. Strongly correlated orbitals have occupations
near 1 (most informative), weakly correlated ones near 0 or 2. NO
occupations are the standard diagnostic for "how much
multi-reference character is there?":

| Largest NO occupation $\ne 2$ | Interpretation |
|---|---|
| $> 1.95$ | Single-reference: HF / DFT / single-reference CC are reliable |
| $1.8 \dots 1.95$ | Mild correlation: MP2 / CCSD usually adequate |
| $1.4 \dots 1.8$ | Strong correlation: need CASSCF / NEVPT2 / DMRG |
| $\to 1.0$ | Open-shell singlet / diradical: CASSCF mandatory |

For a single-reference SCF the occupation tail is always at 2 (or 0)
to numerical precision; this tutorial's H₂O example sits at $2.0$ to
14 digits because the Hartree-Fock density is exactly idempotent. As
soon as a correlated method enters, including DLPNO-CCSD(T), which is
shipped and covered in
[DLPNO local correlation](dlpno_local_correlation.md), this becomes
load-bearing and the spread starts to matter.

## Reading $\Delta$ correctly

The ``idempotency_deviation`` formula depends on which density was
diagonalized:

| ``kind`` | $\Delta$ formula | What "small $\Delta$" means |
|---|---|---|
| ``rhf`` / ``uhf-total`` | $\tfrac{1}{2}\sum_i n_i (2 - n_i)$ | Density is close to a single closed-shell determinant |
| ``uhf-alpha`` / ``uhf-beta`` | $\sum_i n_i (1 - n_i)$ | Density is close to an idempotent per-spin density (always true for UHF) |
| ``uhf-spin`` | $\tfrac{1}{2}\sum_i (1 - n_i^2)$ | **Yamaguchi-style spin contamination across the AO basis**, *not* $N_{\text{unpaired}}$ |

The triplet-O₂ example has $\Delta = 1.04$ for ``uhf-total`` and
$\Delta \approx 13$ for ``uhf-spin``. **Neither indicates a problem.**
$\Delta_{\text{total}} \approx 1$ comes from the two singly-occupied
NOs each contributing $\tfrac{1}{2} \cdot 1 \cdot (2 - 1) = 0.5$;
$\Delta_{\text{spin}} \approx 13$ measures spin density spread
across the AO basis (~26 spin-polarized orbitals each contributing
$\sim 0.5$). To recover the correct $N_{\text{unpaired}} = 2$ from
the spin NOs, **count eigenvalues with $|n_i| > 0.95$**, don't sum the
Yamaguchi $\Delta$. The proper Head-Gordon $N_{\text{unpaired}}$ is
$\sum_i \min(n_i, 2 - n_i)$ for the *total* NOs, equal to 2 for our
O₂.

## Theory

The one-particle reduced density matrix is

$$
\gamma(\mathbf{r}, \mathbf{r}') = N \int \Psi^*(\mathbf{r}',
\mathbf{r}_2, \dots, \mathbf{r}_N) \, \Psi(\mathbf{r}, \mathbf{r}_2,
\dots, \mathbf{r}_N) \, d\mathbf{r}_2 \cdots d\mathbf{r}_N
$$

The **electron density** is its diagonal: $\rho(\mathbf{r}) =
\gamma(\mathbf{r}, \mathbf{r})$. Löwdin's 1955 theorem says the
operator $\hat\gamma$ has a complete set of eigenfunctions, 
the natural orbitals $\phi_i^{\text{NO}}$, whose eigenvalues
$n_i \in [0, 2]$ (in [0, 1] for spin-orbitals) are the *natural
occupation numbers*:

$$
\hat\gamma \, \phi_i^{\text{NO}} = n_i \, \phi_i^{\text{NO}}, \qquad
\sum_i n_i = N_{\text{electrons}}
$$

In a finite AO basis the operator equation becomes the generalised
eigenproblem $\mathbf{D} \mathbf{S} \mathbf{c}_i = n_i \mathbf{c}_i$,
which vibe-qc solves via Löwdin's symmetric route: form
$\tilde{\mathbf{D}} = \mathbf{S}^{1/2} \mathbf{D} \mathbf{S}^{1/2}$,
diagonalize, and back-transform. This avoids the non-symmetric
eigenproblem $\mathbf{D} \mathbf{S}$ (whose eigenvalues are real but
whose eigenvectors require complex arithmetic).

A Hartree-Fock density is **idempotent** in the orthonormal basis:

$$
\mathbf{D}_{\text{ortho}}^2 = \mathbf{D}_{\text{ortho}}
\quad\Longleftrightarrow\quad
\mathbf{D} \mathbf{S} \mathbf{D} \mathbf{S} = \mathbf{D} \mathbf{S}
$$

That algebraic statement *is* the statement that $n_i \in \{0, 2\}$
exactly, every eigenvalue of $\hat\gamma$ is either fully occupied
or fully unoccupied. The deviation $\Delta = \tfrac{1}{2}\sum_i n_i
(2 - n_i)$ is zero iff every $n_i \in \{0, 2\}$, and grows as
occupations drift into the (0, 2) interior. For correlated densities
$\Delta$ is the standard quantitative measure of how much
multi-reference character is present.

For UHF, $\mathbf{D}_\alpha$ and $\mathbf{D}_\beta$ are each
idempotent (in their respective spin spaces) but their *sum* is not, 
hence the small "spin polarization" tail in panel (b) above. The
**spin density** $\mathbf{D}_\alpha - \mathbf{D}_\beta$ has
eigenvalues in $[-1, 1]$, and its leading eigenvectors localise the
unpaired-electron density. This is the most direct route to "where
are the radical centers" without going through the canonical-MO
labeling.

## Resources

- H₂O / 6-31G* RHF + NO diagonalisation: ~25 MB peak RAM, <0.05 s on
  one core (Apple M2).
- O₂ triplet / 6-31G* UHF + total + spin NO + 2D render: ~50 MB peak
  RAM, <0.5 s on one core.
- The full figure (script `[`examples/plots/natural-orbital-occupations.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/plots/natural-orbital-occupations.py)`)
  runs end-to-end in under 1 s. NO diagonalisation cost is
  $\mathcal{O}(N_{\text{bf}}^3)$, negligible compared to any SCF.

## Caveats

- **NO sign is arbitrary.** Like canonical MOs, NOs are eigenvectors, 
  defined only up to sign. Render the *density* $|\phi_i^{\text{NO}}|^2$
  if you want a sign-invariant picture.
- **Within-degenerate-subspace rotations.** When two NOs share an
  occupation (the two $\pi_g^*$ in O₂ triplet, both at $n = +1$),
  any orthogonal mixture is also a valid pair of NOs. Renderings can
  legitimately appear "rotated" between runs / packages.
- **Basis dependence.** Like Mulliken / Löwdin charges, NO occupations
  drift with basis-set size, diffuse functions populate weakly,
  giving long tails of small-$n$ NOs. Compare across basis sets only
  for the *leading* NOs.
- **Active-space picking.** When NO occupations are used to define a
  CASSCF active space, the standard rule is "include any NO with
  $0.02 < n < 1.98$." vibe-qc ships CASSCF as
  `run_job(method="casscf", active_space=(...))`, and the NO output
  here is exactly the right input for choosing that active space.

## References

- **Löwdin, P.-O.** "Quantum theory of many-particle systems. I.
  Physical interpretations by means of density matrices, natural
  spin-orbitals, and convergence problems in the method of
  configurational interaction," *Phys. Rev.* **97**, 1474 (1955).
  The original natural-orbital paper. Defines $\hat\gamma$, proves
  the eigenvalue inclusion $n_i \in [0, 2]$, and introduces the
  occupation-number diagnostic.
- **Davidson, E. R.** *Reduced Density Matrices in Quantum
  Chemistry*. Academic Press (1976). The standard monograph treatment
  of $\hat\gamma$ and natural orbitals; the source for most modern
  multi-reference diagnostics.
- **McWeeny, R.** *Methods of Molecular Quantum Mechanics*, 2nd ed.
  Academic Press (1989), Chapter 5. Clearest derivation of the
  idempotency relation $\mathbf{D}\mathbf{S}\mathbf{D}\mathbf{S} =
  \mathbf{D}\mathbf{S}$ and its physical content.
- **Yamaguchi, K.; Jensen, F.; Dorigo, A.; Houk, K. N.** "A spin
  correction procedure for unrestricted Hartree-Fock and Møller-Plesset
  wavefunctions for singlet diradicals and polyradicals," *Chem. Phys.
  Lett.* **149**, 537 (1988). Origin of the spin-NO contamination
  diagnostic the ``uhf-spin`` formula approximates.
- **Head-Gordon, M.** "Characterizing unpaired electrons from the
  one-particle density matrix," *Chem. Phys. Lett.* **372**, 508
  (2003). The modern definition of $N_{\text{unpaired}} = \sum_i
  \min(n_i, 2 - n_i)$ used in panel-(b)'s "two NOs at $n = 1$"
  reading.

## Next

- [Orbital and density visualization](orbital_visualization.md)
  shows how to write the NO coefficients to ``.molden`` / ``.cube``
  for external viewers, the NO matrix is drop-in compatible with the
  same writers.
- [Projected DOS](pdos.md) is the periodic-system
  analogue, energy-resolved decomposition of the density of states
  onto per-atom orbital projections.
