#pragma once

/// Open-shell (UHF-based) DLPNO-CCSD amplitude solver contracted in PNO spaces.
///
/// The Python local solver (python/vibeqc/dlpno/uccsd_local_solver.py) keeps
/// every spin-orbital pair's doubles in that pair's PNO space but, on its
/// "grouped" route, evaluates each residual in the union of its coupling
/// set's pair spaces with the dense spin-orbital kernel
/// dlpno_uccsd_pair_residual, which forms every antisymmetrised integral of
/// that union: O((n_occ + n_union)^4) memory and O(n^6) work per group.
///
/// This solver evaluates the spin-orbital CCSD residual of Stanton, Gauss,
/// Watts and Bartlett (J. Chem. Phys. 94, 4334 (1991)), density-fitted, with
/// every term contracted in the PNO space of the amplitude that carries the
/// contracted virtual index (Riplinger and Neese, J. Chem. Phys. 138, 034106
/// (2013), Sec. II C):
///
///   * a doubles amplitude t_kl (k < l, antisymmetric in its virtuals) lives
///     in [kl]; a contracted virtual index of it runs over [kl];
///   * a free virtual index of the target pair ij is projected into [ij]
///     through S^{ij,kl} = U_ij^T U_kl, an ordinary matrix product because
///     every PNO set is expressed in the orthonormal canonical spin-virtual
///     basis;
///   * singles live in the full canonical virtual space of their spin (or in
///     a caller-supplied singles basis); their products with integrals are
///     folded into the density-fitting factors, so no singles term is
///     truncated by a pair space.
///
/// The terms are the transcription of the density-fitted rewrite checked
/// against python/vibeqc/dlpno/_ccsd_ref.py::so_residuals in
/// tests/test_dlpno_uccsd_pno.py. Because every contracted virtual index is
/// carried by an amplitude, the pair residual equals U_ij^T R U_ij of the
/// canonical spin-orbital residual R evaluated with the pair amplitudes
/// embedded in the full virtual space, at any PNO truncation, whenever each
/// coupling set is the whole occupied space; the tests pin that identity.
///
/// Spin orbitals: occupied index m in [0, n_occ_a + n_occ_b), alpha first;
/// virtual coordinates [0, n_vir_a + n_vir_b), alpha first. The DF factors
/// are passed per spin (they are block-diagonal in spin); a pair's PNO
/// coefficient matrix U has n_vir_a + n_vir_b rows.

#include <Eigen/Core>

#include <cstddef>
#include <vector>

#include "vibeqc/dlpno_ccsd.hpp"

namespace vibeqc {

/// One kept spin-orbital pair (i < j) of the open-shell local solver.
struct DLPNOUPairInput {
    int i = 0;
    int j = 0;
    /// (n_vir_a + n_vir_b, n_pno) PNO coefficients in the canonical
    /// spin-virtual basis, orthonormal columns, Fock-diagonal in the pair.
    Eigen::MatrixXd U;
    /// (n_pno) semicanonical PNO energies.
    Eigen::VectorXd eps;
    /// (n_pno, n_pno) antisymmetrised <ij||ab> in the PNO basis.
    Eigen::MatrixXd K;
    /// (n_pno, n_pno) initial antisymmetric doubles t_ij^{ab}.
    Eigen::MatrixXd T2;
    /// Sorted occupied coupling set; contains i and j.
    std::vector<int> L;
};

/// Singles of one occupied spin orbital.
struct DLPNOUSinglesInput {
    /// Sorted occupied coupling set of the singles residual; contains the
    /// occupied itself.
    std::vector<int> L;
    /// Optional (n_vir_s, n_s) singles basis within the occupied's own spin
    /// block, orthonormal and Fock-diagonal; empty keeps the full canonical
    /// virtual space of that spin.
    Eigen::MatrixXd U;
    /// (n_s) energies of the singles basis; ignored when U is empty.
    Eigen::VectorXd eps;
};

struct DLPNOUCCSDPNOOptions {
    int max_iter = 100;
    double conv_tol_energy = 1e-9;
    double conv_tol_residual = 1e-7;
    int diis_size = 6;
};

struct DLPNOUCCSDPNOResult {
    /// Converged doubles per pair, input order.
    std::vector<Eigen::MatrixXd> T2;
    /// (n_occ, n_vir_a + n_vir_b) singles in the canonical spin-virtual
    /// basis (zero outside each occupied's own spin block).
    Eigen::MatrixXd T1;
    /// Iterated correlation energy of the kept pairs.
    double e_corr = 0.0;
    int n_iter = 0;
    bool converged = false;
    std::vector<double> trace_e_corr;
    std::vector<double> trace_r_norm;
    /// (pair, occupied) factor rows stored outside the pair's own coupling
    /// set because another target's contraction needs them.
    std::size_t n_offlist_rows = 0;
    std::size_t static_factor_bytes = 0;
    double setup_seconds = 0.0;
    double iteration_seconds = 0.0;
};

/// Solve the open-shell DLPNO-CCSD amplitude equations in PNO spaces.
///
/// @param B_ov_a (n_aux, n_occ_a*n_vir_a) row-major B^P_{i a}, alpha.
/// @param B_ov_b (n_aux, n_occ_b*n_vir_b) row-major, beta.
/// @param B_oo_a (n_aux, n_occ_a*n_occ_a), B_oo_b likewise.
/// @param B_vv_a (n_aux, n_vir_a*n_vir_a), B_vv_b likewise.
/// @param f_oo   (n_occ, n_occ) localised spin-orbital occupied Fock.
/// @param f_ov   (n_occ, n_vir) spin-orbital occupied-virtual Fock.
/// @param f_vv_a (n_vir_a, n_vir_a) and f_vv_b: canonical virtual Fock.
DLPNOUCCSDPNOResult dlpno_uccsd_pno_solve(
    const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
    const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
    const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& f_vv_a, const Eigen::MatrixXd& f_vv_b,
    std::vector<DLPNOUPairInput> pairs,
    std::vector<DLPNOUSinglesInput> singles,
    const DLPNOUCCSDPNOOptions& opts);

/// Evaluate the residuals once for the given amplitudes (testing).
///
/// T1 is (n_occ, n_vir) in the canonical spin-virtual basis; the doubles are
/// the pairs' T2. Returns the pair residuals in their PNO spaces and the
/// singles residual in the full canonical spin-virtual basis (before any
/// singles-basis projection).
void dlpno_uccsd_pno_residuals(
    const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
    const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
    const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& f_vv_a, const Eigen::MatrixXd& f_vv_b,
    std::vector<DLPNOUPairInput> pairs,
    std::vector<DLPNOUSinglesInput> singles,
    const Eigen::MatrixXd& T1,
    std::vector<Eigen::MatrixXd>& R2, Eigen::MatrixXd& R1);

/// Result of the local open-shell (T0) correction.
struct DLPNOUTriplesResult {
    /// Sum of the per-triple contributions.
    double e_t = 0.0;
    /// TNO count and energy per evaluated triple, input order.
    std::vector<int> n_tno;
    std::vector<double> e_triple;
    double seconds = 0.0;
};

/// Local semicanonical (T0) correction on converged open-shell PNO
/// amplitudes, spin-orbital (Raghavachari et al. 1989 in the connected /
/// disconnected form of python/vibeqc/dlpno/_ccsd_ref.py::so_triple_energy).
///
/// Each triple i < j < k gets a TNO space: the orthonormalised union of its
/// three pair spaces (singular values below ``lindep`` times the largest are
/// dropped), optionally truncated to the eigenvectors of the three projected
/// pair densities with occupation above ``tcut_tno`` (0 keeps the union),
/// and semicanonicalised. Every virtual index of the triple lives in that
/// space; the occupied sum of the connected term runs over every m whose
/// pair with the triple's occupied is kept. Only those pairs are projected
/// and only the triple's density-fitting blocks are formed. The occupied
/// energies are the diagonal of the localised Fock (the T0 approximation).
DLPNOUTriplesResult dlpno_uccsd_local_triples(
    const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
    const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
    const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
    const Eigen::VectorXd& eps_o, const Eigen::MatrixXd& f_vv_a,
    const Eigen::MatrixXd& f_vv_b, const std::vector<int>& pair_i,
    const std::vector<int>& pair_j, const std::vector<Eigen::MatrixXd>& U,
    const std::vector<Eigen::MatrixXd>& T2, const Eigen::MatrixXd& T1,
    const std::vector<std::vector<int>>& triples, double tcut_tno,
    double lindep);

}  // namespace vibeqc
