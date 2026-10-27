# Fixed-cell charged BIPOLE UHF checks

This scoped check qualifies the six configurations below against independent
PySCF 2.14.0 analytic-Fourier UHF calculations. It does not establish CRYSTAL23
parity, bulk magnetic-state accuracy, charged-defect corrections, or performance.
The comparison suite still refuses charged requests without an implemented
background-convention contract. Completed 0.18 release qualification is unchanged.

## Physical and numerical inputs

All cases use a three-dimensional 8-bohr cubic cell, STO-3G without ECP,
zero-temperature UHF, and uniform k weights. He and He+ have one atom at the
origin. He2+ has atoms at (0, 0, -1) and (0, 0, 1) bohr. Charges and spin counts
in the table are per primitive cell. The Gamma point is (0, 0, 0); the full
(2, 1, 1) mesh has Cartesian points (0, 0, 0) and (pi/8, 0, 0) bohr^-1,
each with weight 1/2. No symmetry reduction or local magnetic seed is requested.

Both codes use uniform neutralizing-background Ewald electrostatics and the
probe-charge Ewald correction to exact exchange. No isolated-ion or
Makov-Payne correction is applied. The fixed cell and common convention are
essential: these absolute energies are not isolated-ion energies or comparisons
between different cell sizes. See Makov and Payne, Phys. Rev. B 51, 4014 (1995),
[Sec. III.B, Eqs. (11)-(13)](https://doi.org/10.1103/PhysRevB.51.4014).

BIPOLE uses physical AO-pair support of 12 bohr, requested nuclear support of
16 bohr, Ewald precision 1e-10, and default SR-image precision 1e-8. Resolved
alpha is 0.39987715934900675 bohr^-1 and reciprocal cutoff is
3.8376418216567427 bohr^-1. Resolved nuclear Ewald real support is 12 bohr,
the electron-nuclear erfc image reach is 18.810420605408556 bohr, and the
SR interaction reach is 24.451721203846606 bohr in all six rows.
Direct erfc J/K and analytic reciprocal sums are
used; no fitting or XC quadrature is involved. The initial guess is HCORE;
energy and residual thresholds are 1e-10 Ha and 1e-8, with at most 40 iterations.

PySCF uses KUHF with `exxdiv="ewald"`, AFTDF, cell precision 1e-11, HCORE,
energy and gradient thresholds of 1e-11 and 1e-8, and at most 50 iterations.
The Cartesian k coordinates are passed explicitly. Its `mf.nelec` is explicitly
set to `N_k * (N_alpha, N_beta)`: setting only `cell.charge` and `cell.spin`
does not define the same per-cell state on a multi-k mesh. In particular,
the default He+ (2, 1, 1) reference fills (2, 1) electrons in the BvK cell;
the requested per-cell doublet requires (2, 0). Electron counts are independently
checked by tracing each returned spin density against the overlap.
The analytic Gaussian-product Fourier formulation follows Sun et al.,
J. Chem. Phys. 147, 164119 (2017),
[Sec. II.A, Eq. (16)](https://doi.org/10.1063/1.4998644).

## Results and limits

Energies are Hartree per primitive cell. References use a 101 x 101 x 101
Fourier mesh; repeating all reference SCFs at 81 x 81 x 81 changes energies
by at most 1.01e-11 Ha, below the predeclared 1e-10 Ha reference-convergence
gate. All SCFs converge. Observed spin counts pass the 1e-10-electron gate.
The total-energy comparison gate is 1e-8 Ha; no existing gate was relaxed.

| Cell contents | Charge | Alpha/beta | Mesh | BIPOLE | PySCF AFTDF | Difference |
|---|---:|---:|---|---:|---:|---:|
| He | 0 | 1/1 | 111 | -2.816593507255126 | -2.816593507241620 | -1.35e-11 |
| He | 0 | 1/1 | 211 | -2.812226541445677 | -2.812226541423381 | -2.23e-11 |
| He+ | +1 | 1/0 | 111 | -2.117738604303766 | -2.117738604293459 | -1.03e-11 |
| He+ | +1 | 1/0 | 211 | -2.115567414906857 | -2.115567414892238 | -1.46e-11 |
| He2+ | +1 | 2/1 | 111 | -5.014444221385117 | -5.014444221374941 | -1.02e-11 |
| He2+ | +1 | 2/1 | 211 | -5.008072978235878 | -5.008072978203993 | -3.19e-11 |

The two ionic species are doublets; neutral He supplies the closed-shell
control. They do not qualify magnetic bulk materials. There is no matched
ORCA/CRYSTAL result for these rows, and no timing ratio is claimed: the two
programs used different implementations and setup timing boundaries.

Source/native identity and complete inputs, resolved supports, component
energies and independent reference outputs are retained with tracker #689.
Python source was based on `a3242f5c`; the unchanged native module came from
`855e57b48ae42c4bc58af4af44159fd5c629aaed`, SHA-256
`7b88b6f1b6b7280199ea52bb3334744e0d27289518fe3b7a14b53f1c84f54629`.
This is not a qualification of every native change on current main.

The regression pins are in
`tests/test_bipole_fock_ewald_exchange.py`, selected with
`-k fixed_cell_charged`. Real-checkpoint SKALA, the general omitted-Coulomb
bound, broader method/basis coverage and end-to-end CRYSTAL23 speed remain open.
