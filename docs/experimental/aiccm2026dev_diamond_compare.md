# Diamond Γ-CCM / χ-CCM bond and band comparison

```{admonition} Status: experimental
:class: warning

Both routes this input runs (the four-centre Γ-line drivers `run_ccm_rhf_scalable` / `run_ccm_rks` and the χ-CCM drivers) is an experimental route. Every Γ-line SCF driver emits `vibeqc.AICCM2026DevAExperimentalWarning` and every χ-CCM driver emits `vibeqc.AICCM2026DevBExperimentalWarning`; the `.system` manifest records `method_status = "experimental"` for every AICCM job run through `run_periodic_job(method="aiccm", variant=...)`. Its numbers are research output, not validated results, and it is not part of the release gate. The label vocabulary is defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

This experimental input compares two distinct CCM development approaches on
primitive diamond:

- `aiccm2026dev-a`, the Γ-CCM development line in `vibeqc.periodic.ccm`:
  Γ-CCM is the neutral finite-torus construction (variants `real-gamma` and
  `neutral-bloch`); the script's Γ side runs the `four-center` variant, the
  2014 union-and-weight/Wigner-Seitz lineage kept in the same line.
- `aiccm2026dev-b`, χ-CCM[^xccm-convention], the finite-translation-group
  character stream in `vibeqc.periodic.chi`.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM is
    the neutral finite-torus construction with two producers, `variant="real-gamma"`
    and `variant="neutral-bloch"` (ruling R1, 2026-08-21); the `four-center` variant
    is the 2014 union-and-weight/Wigner-Seitz lineage, a different construction
    kept in the same `aiccm2026dev-a` development line. χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

The script writes the two property bundles side by side and explicitly marks
the Γ-CCM/χ-CCM approach comparison `not-defined`. It does not form energy,
band, charge, or bond-order deltas. A matched exchange-q=0 label is necessary
but not sufficient to prove a common finite Hamiltonian.

The input is:

```bash
python examples/periodic/aiccm2026dev_diamond_bonds_bands_compare.py \
  --route both \
  --basis sto-3g \
  --functional pbe \
  --backend four_center \
  --extension 2 2 2
```

It writes all files below `output/aiccm-diamond-compare/` by default:

- `diamond-aiccm2026dev-a-vs-b-summary.json`, with energies per primitive
  cell, finite-net gaps, Mulliken charges, C-C Mayer bond orders, band
  metadata, and occupied-localization diagnostics.
- `diamond-aiccm2026dev-a-rhf.qvf` and `diamond-aiccm2026dev-b-rhf.qvf`,
  containing the primitive structure, the HF band path, and QVF bond
  connectivity.
- `diamond-aiccm2026dev-a-rks.qvf` and `diamond-aiccm2026dev-b-rks.qvf`,
  the matching KS-DFT band and bond archives.
- `diamond-aiccm2026dev-a-rhf-localized.qvf` and
  `diamond-aiccm2026dev-b-rhf-localized.qvf`, finite-supercell localized
  occupied orbitals for visualization.
- PNG band plots when matplotlib is installed.

The default is a route smoke calculation, not a publication benchmark or an
approach comparison. Do not quote cross-approach differences from this output.
The band path is obtained from each route's converged finite-torus Fock blocks:
it is exact at the folded character net and an interpolation between those
points. Converge it with respect to `--extension`.

## Optional local-PNO summary

Add `--with-b-pno-mp2` to run the χ-CCM local-PNO MP2 pilot after the SCF
steps:

```bash
python examples/periodic/aiccm2026dev_diamond_bonds_bands_compare.py \
  --route b \
  --backend ri \
  --extension 2 2 2 \
  --with-b-pno-mp2
```

This writes `diamond-aiccm2026dev-b-dlpno-mp2.json` with PAO rank,
translation-pair reduction, and local-PNO MP2 energy data. Full correlated
natural orbitals from a relaxed correlated one-particle density matrix are not
implemented in this input and should not be inferred from the PNO summary.
