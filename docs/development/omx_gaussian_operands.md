# Internal OMx Gaussian pair operands

The private native `semiempirical.nddo._evaluate_omx_gaussian_pair` accessor
accepts two `_OMxGaussianContraction` records with explicit exponents, s
coefficients, optional p coefficients and positive zeta. Coefficients multiply
normalized Cartesian primitives. Exponents in bohr^-2 are multiplied by
zeta squared and contractions are explicitly normalized. `raw_norms_a/b`
record squared norms before this normalization. No published basis tables are
bundled by this interface.

Atom-local AO order is s, px, py, pz (or s only), with positive Cartesian p
phases. Origins are bohr. Overlap is dimensionless. ERIs, attractions and core
energy are Hartree. `eri_unscaled` and `eri_scaled` are pair matrices with
row `mu*n_a+nu` and column `lambda*n_b+sigma`, representing
`(mu_A nu_A | lambda_B sigma_B)`. All density-product channels are retained.
The supplied `gamma_ss` sets `f_ko = gamma_ss / eri_unscaled[0,0]`.
Attractions include the negative sign and the other core's supplied charge;
core energy is `f_ko * core_charge_a * core_charge_b / R`.

`coulomb_a(P_b)` and `coulomb_b(P_a)` contract the complete scaled tensor
with a symmetric spin-summed atom-local density, without an extra spin factor.
They return Coulomb contributions only. `_evaluate_omx_legacy_pair_operands`
provides the existing two-atom overlap and `F(P)-F(0)` for algebraic baseline
comparisons without running SCF.

These are internal development operands, following Dral et al., JCTC 12,
1082 (2016), DOI 10.1021/acs.jctc.5b01046, section 2.3 and equations 18-20.
The current OM2/OM3 driver and parameter tables are unchanged. The pair API is
not a complete published OMx Hamiltonian: consistent overlap, penetration,
valence-core, orthogonalization and derivative integration still require
implementation and independent numerical validation. Operand agreement does
not establish an SCF state, minimum, external gradient tolerance or molecular
energy agreement. Refs #496.
