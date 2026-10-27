### Fixed: frequencies and thermochemistry in solvent were the gas phase's

`run_job(hessian=True, solvent=...)` ran a solvated SCF and then printed the
gas-phase Hessian under it, with a `.out` that said nothing about the
difference: every number smooth and plausible, on the wrong surface (a solvated
optimization had the same defect before it was fixed). `compute_hessian_fd` now
takes `solvent=` and finite-differences the in-solvent analytic gradient
(`cpcm_gradient`), re-converging the apparent charges at every displacement, so
the frequencies are those of the equilibrium-solvation surface. The frequency
block names the medium (`Surface: RHF/sto-3g in CPCM water (eps = 78.39)` and a
`Medium:` note) and the `.system` manifest records it in `[hessian]`.

Supported for RHF, UHF, RKS and UKS with `variant="cpcm"` or `"cosmo"`, on
either cavity; functionals with an incomplete analytic gradient take the
full-energy route in solvent too. Refused, with the reason: ROHF and ROKS,
selected integral engines, ECPs, and Direct COSMO-RS (its energy has a cusp at a
symmetric geometry that gradient differences would straddle). A warning fires
for the fine cavity in the molecule-fixed frame at a symmetric geometry, where
the frame tie makes the modes that break the symmetry wrong (measured on water:
the asymmetric stretch 45 cm^-1 low; use the default cavity or
`fine_frame="lab"`).

Validated against the second derivative of the energy the solvated job reports:
the solvent changes the curvature along the leading directions by 7.5e-3 and
3.9e-3 Ha/bohr^2 and the Hessian reproduces the energy's curvature to 1.1e-5
and 5.0e-5. Kohn-Sham (analytic and full-energy routes) and open shell are
checked through the solvent's change in curvature, which cancels the gas-phase
grid error of the analytic KS gradient: 8e-8 apart for PBE and CAM-B3LYP.
