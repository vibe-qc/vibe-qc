### Added: geometry optimization, forces and frequencies for ECP solutes in solvent

The solvated gradient refused every molecule with a pseudopotential, and with it
the optimizer, ASE forces and the finite-difference Hessian, so no heavy-element
solute (iodine, Pd, Pt, Zn ...) could be relaxed or have its frequencies computed
in solution. `cpcm_gradient`, `optimize_molecule`, `run_job(optimize=True)` and
`run_job(hessian=True)` now accept them (RHF, UHF, RKS, UKS; CPCM and COSMO): the
reaction field never touches the ECP, so the gradient is the ECP-aware gas-phase
one plus the usual cavity terms. With no `options` it repeats the ECP provenance
the SCF result recorded, and it refuses a result whose cavity and SCF disagree on
whether there is an ECP.

Against central differences of the reported solvated energy: [ZnH]+ (ecp10mdf)
9e-9 Ha/bohr, vDZP methane (inline primitives) 1e-8, HI in def2-SVP 2e-8; the
solvated HI Hessian reproduces the energy's curvature to 3e-8 Ha/bohr^2 with the
solvent contributing 9.4e-4.

### Fixed: frequencies, thermochemistry and MD refused every atom heavier than krypton

The atomic mass table stopped at Kr, so any job with a heavy atom died on "no
built-in mass" before computing anything, which is exactly the set of elements an
ECP basis exists for. It now reaches plutonium (IUPAC conventional values; the
mass number of the longest-lived isotope where no stable one exists).
`center_of_mass` and the thermochemistry origin accept those atoms too.
