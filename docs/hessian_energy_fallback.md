# Full-energy molecular Hessian fallback

For RKS and UKS functionals with missing analytic nuclear-gradient terms,
`compute_hessian_fd` evaluates the complete SCF energy at both sides of every
inner Cartesian displacement. The inner gradient step is 0.001 bohr; the outer
Hessian step remains `HessianFDOptions.step_bohr`. The output announces the
functional, omitted terms and inner step. Every displaced SCF must converge,
and nonfinite energies are refused. Cost is quadratic in Cartesian degrees of
freedom; this route does not implement analytic long-range exchange or VV10
response derivatives.

The same displacement wrapper preserves explicit SCF/grid options, moves ECP
centres with their atoms and projects READ densities without accumulating
changes to caller options. Optional dispersion is differentiated once through
its existing complete gradient. Dipoles use the converged outer displaced
states. Analytic RKS/UKS Hessian entry points retain their explicit refusal;
the ASE calculator selects the FD route for these functionals.

The outer convention is H = d(grad E)/dR = -dF/dR, as in Larsen et al.,
*J. Phys.: Condens. Matter* **29**, 273002 (2017), section 8.1,
[doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).
Numerical step convergence and tight SCF convergence remain necessary for
scientific frequency acceptance; source checks alone do not establish accuracy.
