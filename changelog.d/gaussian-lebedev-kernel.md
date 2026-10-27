### Added: opt-in Gaussian segment charges on the Lebedev cavity (#681)

`SolventModel(charge_representation="gaussian")` keeps the switched Lebedev
cavity and replaces the point-charge cavity matrix, which is indefinite (its
smallest eigenvalue runs -6 to -28 on distorted methanol, so the energy has
poles and the analytic gradient spikes near them), with York and Karplus's
Gaussian charges, whose matrix is positive definite at every geometry. The
analytic gradient is exact for the construction: on 60 distorted methanols it
is within 1.6e-8 Ha/bohr of a finite difference everywhere, where the default
kernel disagrees by more than 1e-5 at 20% of them. The citation report names the
kernel that ran (`cpcm_gaussian` / `cosmo_gaussian`).

The default is unchanged, because the kernel moves every solvated energy: the
point kernel overshoots the sharp union-of-spheres answer by 17% (302 points)
to 27% (974 points) at the default switching width and drifts further as the
grid is refined, where the Gaussian kernel is 2% to 4% off. Whether it becomes
the default is the maintainer's decision, tracked in #681; the numbers are in
the user guide under "Choosing the number of cavity points".
