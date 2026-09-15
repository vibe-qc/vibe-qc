// Open-shell UCCSD and perturbative (T) on a UHF reference, density-fitted
// by default or with exact four-index integrals on the canonical
// conventional route (CCSDOptions.density_fit = false).
//
// Equation provenance (the unrestricted sibling of cpp/src/ccsd.cpp):
//
//   The residuals are the spin-orbital CCSD working equations of
//   J. F. Stanton, J. Gauss, J. D. Watts, R. J. Bartlett, J. Chem. Phys.
//   94, 4334 (1991), doi:10.1063/1.460620 (SGWB), Eqs. (1)-(13), evaluated
//   directly over spin orbitals (no spin adaptation). The spin-orbital form
//   is reference-agnostic, so the same equations that the closed-shell
//   kernel spin-integrates here run unmodified on a UHF reference -- the
//   only difference is that the alpha and beta spin orbitals carry distinct
//   spatial MOs. The reference spin-orbital implementation lives in
//   python/vibeqc/dlpno/_ccsd_ref.py (run_ref_uccsd); this C++ kernel was
//   validated against it to machine precision and against PySCF
//   cc.UCCSD / UCCSD(T) to < 1 uHa (tests/test_uccsd_anchor.py,
//   tests/test_uccsd.py).
//
//   Original CCSD: G. D. Purvis III, R. J. Bartlett, J. Chem. Phys. 76,
//     1910 (1982), doi:10.1063/1.443164.
//   Density fitting of all two-electron integrals follows A. E. DePrince
//     III, C. D. Sherrill, J. Chem. Theory Comput. 9, 2687 (2013),
//     doi:10.1021/ct400250u: every integral block is assembled from the
//     MO-basis three-index DF tensor, (pq|rs) = sum_P B^P_pq B^P_rs, built
//     here as a block-diagonal *spin-orbital* B-tensor (alpha pairs use the
//     alpha MO transform, beta pairs the beta transform, cross-spin pairs
//     vanish).
//   The (T) correction is K. Raghavachari, G. W. Trucks, J. A. Pople,
//     M. Head-Gordon, Chem. Phys. Lett. 157, 479 (1989),
//     doi:10.1016/0009-2614(89)87395-6, in its spin-orbital connected /
//     disconnected form, evaluated triple-by-triple (the t3 tensor is never
//     materialised). (T) assumes canonical orbitals (f_ov = 0 in the
//     triples denominators); run_uccsd hands it the converged UHF orbital
//     energies.
//
// Scope: dense O(N^6) small-molecule pilot, matching cpp/src/ccsd.cpp. The
// antisymmetrised virtual-virtual block is materialised (nv^4 over spin
// orbitals); fine for small open-shell molecules (radicals, O2), not tuned
// for production scale. Frozen occupied cores are supported (the same
// chemical-core count is frozen in each spin channel); frozen virtuals are
// not.

#pragma once

#include "basis.hpp"
#include "ccsd.hpp"   // CCSDOptions, CCSDResult (shared with the RHF kernel)
#include "molecule.hpp"
#include "uhf.hpp"

namespace vibeqc {

// Run open-shell UCCSD (and optionally UCCSD(T)) on a converged UHF
// reference.  `uhf` supplies the converged alpha/beta MO coefficients and
// orbital energies; `mol` and `basis` define the AO space for the integral
// build (DF by default, exact four-index when opts.density_fit is false).
//
// Throws std::runtime_error if `uhf.converged` is false, if density fitting
// is requested without an auxiliary basis, or if the frozen-core window is
// invalid.
//
// When `opts.compute_triples` is true, the (T) correction is appended to
// `result.e_t` and `result.e_ccsd_t`.  The returned CCSDResult uses the same
// fields as the closed-shell kernel.
CCSDResult run_uccsd(const Molecule& mol,
                     const BasisSet& basis,
                     const UHFResult& uhf,
                     const CCSDOptions& opts = {});

// Run open-shell UCCSD(T) from explicit arrays -- the entry point for
// ROHF-reference CCSD and any other reference whose MO coefficients and
// per-spin Fock matrices are available but that does not carry a UHFResult.
//
// C_alpha / C_beta  — AO-to-MO coefficient matrices (n_ao × n_orb).  For an
//   ROHF reference these are identical (a single set of spatial orbitals),
//   with the occupied/virtual partition differing per spin via
//   n_alpha / n_beta derived from mol.multiplicity().
// F_alpha / F_beta  — per-spin AO Fock matrices.  The C++ kernel dot-products
//   C^T F C over the active window to build the spin-orbital Fock.
// e_hf              — SCF reference energy, attached to result.e_hf.
// ecp_total_ncore   — mandatory provenance for the otherwise opaque arrays;
//                     any nonzero replaced-core count is rejected.
//
// The same frozen-core, DIIS, and convergence controls apply.  All other
// parameters (molecule, basis, options) are identical to run_uccsd.
CCSDResult run_uccsd_from_mos(const Molecule& mol,
                              const BasisSet& basis,
                              const Eigen::MatrixXd& C_alpha,
                              const Eigen::MatrixXd& C_beta,
                              const Eigen::MatrixXd& F_alpha,
                              const Eigen::MatrixXd& F_beta,
                              double e_hf,
                              int ecp_total_ncore,
                              const CCSDOptions& opts = {});

// A single spin-orbital CCSD residual evaluation for a caller-supplied local
// DLPNO domain. The local basis is ordered [occupied | virtual]; B_so stores
// the corresponding spin-orbital DF factors with flattened pair columns
// p*n+q. T2_flat and R2_flat use rows i*no+j and columns a*nv+b.
//
// This is a numerical primitive only: it performs no amplitude iteration,
// domain construction, screening, or public-method dispatch.
void dlpno_uccsd_pair_residual(
    const Eigen::MatrixXd& T1,
    const Eigen::MatrixXd& T2_flat,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_so,
    const Eigen::MatrixXd& f_so,
    Eigen::MatrixXd& R1,
    Eigen::MatrixXd& R2_flat);

// Energy contribution for one distinct occupied spin-orbital triple
// (i < j < k) in a caller-supplied virtual domain. T2_flat uses rows
// p*n_occ+q and columns a*n_vir+b. The antisymmetrised integral blocks use
// flattened row-major pair indices:
//   eri_vovv[e*n_occ+i, b*n_vir+c] = <e i||b c>
//   eri_ovoo[m*n_vir+a, j*n_occ+k] = <m a||j k>
//   eri_oovv[j*n_occ+k, b*n_vir+c] = <j k||b c>
// This numerical primitive performs no triple screening, TNO construction,
// amplitude iteration, or public-method dispatch.
double dlpno_spin_orbital_triple_energy(
    Eigen::Index i,
    Eigen::Index j,
    Eigen::Index k,
    const Eigen::MatrixXd& T1,
    const Eigen::MatrixXd& T2_flat,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& eri_vovv,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& eri_ovoo,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& eri_oovv,
    const Eigen::VectorXd& eps_o,
    const Eigen::VectorXd& eps_v);

}  // namespace vibeqc
