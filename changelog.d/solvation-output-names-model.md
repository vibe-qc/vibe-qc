### Changed: the solvation block of the `.out` names the model that produced the energy

The **Implicit solvation** block printed the energies and nothing about the model
behind them. It now opens with the solvent and its dielectric constant, the
screening family and its `f(eps)`, the cavity construction (switched Lebedev
spheres with the grid, radii scale and switching width, or the COSMO FINE
cavity), the number of segments and the cavity area, the segment-charge kernel
(point charges or Gaussian, which move the energy by percent, #681) and the
number of reaction-field cycles. Everything is read off the result. MSINDO COSMO
jobs, whose result carries none of this, print the block as before.

The solvation user guide's quick-start also labelled `1/2 q.V` as the solvation
free energy; the electrostatic free energy is the in-solvent total minus the
gas-phase total (4.08 against 4.30 kcal/mol for water in STO-3G), and the guide
now says so.
