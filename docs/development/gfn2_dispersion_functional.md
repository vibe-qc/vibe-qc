# Molecular GFN2 dispersion integration

`vibeqc.semiempirical.methods.gfn2_dispersion` provides internal
energy, charge-potential and geometry-derivative operands. An immutable
native snapshot can participate in molecular SCC. This is an internal
integration path, not a selectable high-level GFN2 model. Current molecular
`GFN2Model` still uses its documented post-SCF EEQ correction.

For references indexed by `i` and `j`, the symmetric matrix `K` contains
reference C6 coefficients times negative Becke-Johnson damping, with zero
same-atom blocks. The reference weights `W_i(q_A)` combine normalized CN
interpolation and charge scaling. Then

```text
E2(q) = 1/2 W(q)^T K W(q)
dE2/dq_A = sum_{i on A} (dW_i/dq_A) (K W)_i
```

The factor `1/2` counts each atom pair once. For positive Mulliken partial
charges `q_A = Zval_A - population_A`, the corresponding AO Hamiltonian term
is `-1/2 S_mu,nu (dE/dq_A + dE/dq_B)`. Using the opposite sign or dropping
the pair factor breaks the density derivative. The ATM contribution uses
neutral-charge C6 coefficients and therefore contributes no charge potential.
This follows Bannwarth, Ehlert and Grimme, *JCTC* **15**, 1652 (2019),
doi:10.1021/acs.jctc.8b01176, and the D4 interpolation of Caldeweyher et al.,
*JCP* **150**, 154122 (2019), doi:10.1063/1.5090222.

Construction freezes the reference data, CN weights and geometric matrix.
Evaluation accepts explicit partial charges and performs no EEQ charge solve.
It returns energy in Hartree and the charge derivative in Hartree per electron.
`energy_potential_gradient` also returns the explicit gradient in Hartree per
bohr at fixed partial charges, including CN response and neutral ATM.
The current builder uses the native independently generated references and
EEQ coordination numbers. These conventions still differ from published
GFN2 dispersion; changing the charge model alone does not establish parity.

`tests/test_gfn2_dispersion_functional.py` compares matrix energies with an
independent pair sum, charge potentials with two finite-difference steps,
and the AO term with electron-number-preserving density perturbations.
It also checks neutral ATM, atom permutations, immutable operands, the
isolated-atom limit and rejection of malformed or missing reference data.
Native integration checks compare full coupled SCC free-energy derivatives
at two displacement sizes, including fractional occupations and empty
valence spaces.

The internal native call accepts `kernel.to_native()` as its fourth argument.
It requires faithful AES and exactly matching elements and coordinates.
The kernel is included in every Fock build, the final stationarity check,
and each automatic retry. Its energy is counted once in the reported internal
and free energies. The analytic gradient combines its explicit geometry
derivative with the Mulliken-overlap term and the energy-weighted density
of the coupled Fock. Do not add a post-SCF D4 correction to this result.

The native result reports `e_dispersion` and `dispersion_sha256`. The latter
is a canonical hash of all consumed kernel and derivative operands, computed
in the native constructor from detached copies. It is retained even on a
failed attempt. It identifies operands, not an independently validated
published reference dataset.

Remaining requirements are high-level model and output integration,
independent numerical review, and reference-convention acceptance.
No public default or production-readiness claim follows from these operand
checks. The original energy and force acceptance requirements remain open.
