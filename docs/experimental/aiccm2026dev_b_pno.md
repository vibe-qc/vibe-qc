# χ-CCM PAO and PNO pipeline

Status: experimental and χ-CCM only. It does not import or modify the
`aiccm2026dev-a` implementation.

## Public primitives

`projected_atomic_orbitals` projects a supplied AO domain out of the occupied
space and removes linear dependencies by canonical orthogonalization.
`build_pair_natural_orbitals` diagonalizes a Hermitian pair density and
retains occupations above a threshold. `enumerate_pair_orbits` partitions
unordered occupied pairs under measured finite-torus translation
permutations.

```python
paos = vq.projected_atomic_orbitals(C_occ, S, F)
pnos = vq.build_pair_natural_orbitals(
    T_ij,
    paos.coefficients,
    diagonal_pair=(i == j),
    occupation_threshold=1.0e-7,
    fock=F,
)

print(paos.rank, paos.occupied_leakage_error)
print(pnos.retained_rank, pnos.discarded_occupation)
```

The projector is (Q=I-C_oC_o^\dagger S). If the selected raw PAOs are
(\widetilde V=QE_D), their Gram eigenvectors give

\[
 V=\widetilde V X_+g_+^{-1/2},
 \qquad V^\dagger S V=I,
 \qquad C_o^\dagger S V=0.
\]

For pair amplitudes (T), the implemented pair density is

\[
 D=\frac{TT^\dagger+T^\dagger T}{1+\delta_{ij}}.
\]

It is positive semidefinite. Tightening the occupation threshold therefore
gives nested PNO ranks, and a zero threshold retains the complete supplied
virtual span. This does not make the MP2 energy variational.

## Integrated finite-torus route

Use the B-specific Wannier gauge with the 3D local-MP2 wrapper:

```python
from vibeqc.dlpno.mp2 import DLPNOMP2Options

result = vq.run_aiccm2026dev_b_dlpno_mp2(
    system,
    basis,
    mesh=(4, 1, 1),
    dlpno_options=DLPNOMP2Options(
        localise="wannier",
        tcut_pno=0.0,
        tcut_pno_weak=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        tcut_pairs_weak=0.0,
    ),
)

print(result.local_correlation_space)
print(result.raw_local_e_corr_per_cell)
print(result.complete_space_correction_per_cell)
```

At complete domains and zero PNO thresholds, a separate full-space DF-MP2
contraction enforces the canonical finite-torus limit. The raw local-pair
energy and the correction remain visible. A truncated calculation receives
no correction.

Translation pair orbits are currently diagnostics. No representative is
skipped, so `translation_reduction_factor` is not a measured speedup. The
`translation_orbits_validated` flag requires a covariance residual below
(10^{-8}). The current finite-supercell IAO cross overlap does not satisfy
that gate; use Wannier orbit mappings for diagnostics and keep all pairs.

Minimum-image pair screening, local auxiliary fitting, representative-only
amplitude propagation, and a localized-gauge CC exact-limit audit remain
open. The code fails closed where an existing molecular distance definition
would break cyclic translation symmetry.
