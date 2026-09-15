# Experimental χ-CCM occupied localization

`aiccm2026dev-b` is the code selector for χ-CCM[^xccm-convention], the
finite-translation-group character CCM line. It can turn a converged,
gapped RHF or RKS occupied space into localized orbitals on its finite
Born-von Karman torus. This API is experimental and belongs only to χ-CCM. It
does not change Γ-CCM, the union-and-weight sibling approach, or the SCF
orbitals stored on the original result.

The prerequisite χ-CCM SCF result must currently be 3D. Every 1D/2D backend
fails closed before SCF, so no lower-dimensional result may be supplied to the
localizer until the shared wire/slab Coulomb convention is derived.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

```python
import vibeqc as vq

scf = vq.run_aiccm2026dev_b_rhf(
    system,
    basis,
    mesh=(4, 1, 1),
    backend="four_center",
    progress=False,
)
wannier = vq.localize_aiccm2026dev_b_occupied(
    scf,
    system,
    basis,
    method="wannier",
)

assert wannier.density_invariance_error < 1.0e-8
assert wannier.one_particle_energy_error < 1.0e-8
```

The coefficient row order is `(cyclic cell, AO)`. The cell order is recorded
in `wannier.translations`. Localization may reorder the columns, so downstream
code must use `wannier.translation_permutations` instead of assuming a
`(cell, band)` column order. The table maps each column to its image under the
three primitive cyclic translations.

## What is minimized

For a mesh with (N=N_1N_2N_3) characters, the unlocalized real-torus
orbitals are

\[
 w_{n\mathbf R}(\mathbf r)
 = \frac{1}{\sqrt N}\sum_{\mathbf k}
 e^{-2\pi i\mathbf k\cdot\mathbf R}\psi_{n\mathbf k}(\mathbf r).
\]

The implementation applies one unitary matrix (U) to the complete
(Nn_{\mathrm{occ}})-dimensional occupied space. Therefore

\[
 C' = C U,\qquad U^\dagger U=I,\qquad
 C'C'^\dagger=CC^\dagger.
\]

The last identity is the hard density and SCF-energy invariance gate.

The `wannier` path jointly diagonalizes sine and cosine operators formed from
AO centers on the cyclic supercell. If

\[
 z_{n\alpha}=\langle w_n|
 e^{2\pi i r_\alpha/L_\alpha}|w_n\rangle,
\]

the reported diagonal spread contribution is

\[
 \Omega_{n\alpha}=-\left(\frac{L_\alpha}{2\pi}\right)^2
 \log |z_{n\alpha}|^2.
\]

This has the periodic topology of the Marzari-Vanderbilt functional. It is
not yet the exact continuum functional because the present integral API does
not expose cross-k matrix elements of the exponential position operator. The
result labels this explicitly as a `projected circular AO-centre
approximation`. Spreads and centers use bohr, the native vibe-qc geometry
unit, and bohr squared.

## IAO population gauge

The alternative

```python
iao = vq.localize_aiccm2026dev_b_occupied(
    scf,
    system,
    basis,
    method="iao",
    iao_reference_basis="ano-rcc-mb",
)
```

builds intrinsic atomic orbitals from the ANO-RCC-MB minimal reference. It
then maximizes the sum of squared IAO atomic populations. This follows the
projection construction of Knizia rather than Mulliken partitioning. The
current cross-basis overlap is evaluated in the explicit finite supercell;
cross-boundary reference overlaps are not yet periodized. Converge the cyclic
cluster and inspect the wrap diagnostic before using IAO domains.

## Wrap-around and metallic guards

`aliasing_fraction[n]` measures orbital weight in the antipodal shell of the
cyclic cluster. `aliasing_detected[n]` also fires when the inferred
localization length exceeds one quarter of the shortest cluster vector. A
warning means that a local-correlation domain cannot distinguish a physical
tail from its wrapped image. Increase the cluster; do not silence the warning
and tighten a domain cutoff.

The function rejects a zero or negative indirect band gap. Disentanglement of
metallic or entangled bands is not implemented. A Wannier90 comparison path
is also still open because no Wannier90 file exporter is wired to this B-only
experimental route.

The implementation cites Marzari and Vanderbilt, Phys. Rev. B 56, 12847
(1997), DOI `10.1103/PhysRevB.56.12847`, and Knizia, J. Chem. Theory Comput. 9,
4834 (2013), DOI `10.1021/ct400687b`. Both references already exist in the
vibe-qc citation database under the `wannier` and `ibo` property routes.
