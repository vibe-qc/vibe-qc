# Saved BIPOLE state and unified operator compatibility

This offline diagnostic consumes an existing Si PBE/POB-TZVP-rev2 BIPOLE
result on the full 2x2x2 mesh. It exercises the draft unified operator API
without importing a native extension or rerunning SCF. The operator modules
are loaded from an explicit source directory, without modifying that checkout.
This permits testing the draft while its merge request is still open.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python studies/pbc-operator-compatibility/audit_saved_si.py \
  --operator-source "$OPERATOR_SOURCE/python/vibeqc/solvers" \
  --state-dir "$SAVED_STATE" --output "$EVIDENCE/result.json"
```

NumPy and SciPy are required. Keep downloaded states and generated evidence
outside the product checkout. The state directory contains `state.npz`,
`execution-input.json` and `manifest.json` from the retained Si fixed-density
study. The manifest seals the state and input. The execution input supplies
the lattice (column vectors), Cartesian k points and normalized weights.
The archive supplies Fock, overlap, real-space density and integer cell
indices. This is a fixture-specific diagnostic, not a general restart reader.

The script reconstructs D(k) from one minimum-radius representative of each
BvK residue class and checks every retained real-space density block. It
then checks:

- immutable F/S snapshots and block actions at all eight k points;
- full spectra and occupied projectors against SciPy generalized LAPACK;
- complex diagonal AO coordinate changes of F, S and D, including electron
  count and one-particle traces;
- rejection of reused subspaces from a different AO, spin, k-point or
  symmetry frame before any native solver can run;
- preservation of the input archive, source bytes and evaluated density.

The complex coordinate changes test representation covariance. They do not
establish crystal symmetry of the finite integral source. In particular, the
script never enables irreducible-mesh or quartet reduction. Occupations remain
the fixture's 14 doubly occupied orbitals per k point; weights enter cell
traces, not orbital normalization. The newly solved orbital projector is
reported separately from the density at which the stored Fock was evaluated.

## Witness and scope

The initial author run on 2026-10-09 used operator draft `b0047d47`
(merge request !203, issue #472) and the saved Si state produced at
`c60cc53a`. Its state SHA-256 is
`085cf9c5abb55c70bae213978a151464491580ab367c95ecf7c387cfd50d33fa`.
The machine-readable report records full source, input and script hashes.

Sixteen dense solves and 32 incompatible-frame refusals passed. Maximum
spectrum difference from generalized LAPACK was 9.85e-13 Ha; maximum occupied
projector difference was 2.19e-13. Complex-frame projector covariance differed
by at most 2.34e-13. The weighted overlap population was 28.000000000000007.
The evaluated-to-orbital density difference reached 6.81e-9 and was retained
as a diagnostic, not overwritten or interpreted as a new SCF acceptance test.

This witness validates an exported-state crossing into the Python dense API.
It does not qualify a native rebuild, integration into the BIPOLE SCF loop,
iterative solver convergence, HF/hybrid/UKS/SKALA coverage, MANTA source
equivalence, symmetry reduction, CRYSTAL23 accuracy or end-to-end performance.
The operator draft remains a separate dependency; this study does not merge
or enable it in production.
