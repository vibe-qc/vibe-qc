#pragma once
/// Closed-shell DLPNO-CCSD amplitude solver contracted in PNO spaces.
///
/// The Python local solver (python/vibeqc/dlpno/ccsd_local_solver.py) stores
/// every pair's doubles in that pair's PNO space but evaluates the residual
/// once per (coupling set, extended atom domain) group in the group's
/// extended PAO basis, a canonical-shaped closed-shell CCSD residual of
/// dimension nL x n_ext. That is exact with respect to the coupling set, but
/// its cost per group is O(nL^2 n_ext^2 n_aux + nL^3 n_ext^3), and past the
/// size where the extended domain becomes a proper subset of the virtual
/// space the number of groups approaches the number of pairs, so the
/// per-iteration cost is far above canonical CCSD (issues #66, #190).
///
/// This solver evaluates the same closed-shell residual (Stanton, Gauss,
/// Watts, Bartlett 1991 spin-integrated as in DePrince and Sherrill, J. Chem.
/// Theory Comput. 9, 2687 (2013), doi:10.1021/ct400250u; density-fitted) with every term contracted in
/// the PNO space that carries the contracted index, the scheme of Riplinger
/// and Neese, J. Chem. Phys. 138, 034106 (2013), Sec. II C:
///
///   * a doubles amplitude t_kl lives in [kl]; whenever one of its virtual
///     indices is contracted, the contraction runs over [kl];
///   * a free virtual index of the target pair ij is projected into [ij]
///     through the overlap S^{ij,kl} = U_ij^T U_kl (Eq. 30), which is an
///     ordinary matrix product here because every PNO set is expressed in
///     the orthonormal canonical virtual basis;
///   * singles t_i live in [ii]; their products with integrals are folded
///     into the density-fitting factors over the FULL virtual space (the
///     u/w vectors of the Python pair-space ladder), so no t1 x t1 product is
///     ever truncated.
///
/// Because every contracted virtual index is carried by some amplitude, the
/// result is exact with respect to the occupied coupling set: it reproduces
/// the extended-domain group residual to working precision at any PNO
/// truncation, and canonical CCSD when every pair's PNO space is the whole
/// virtual space (the parity ratchet of tests/test_dlpno_ccsd_solver.py).
///
/// Cost per iteration is O(n_pairs nL^2 n_aux n_pno^2) with global RI
/// factors, independent of the extended-domain dimension.

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <functional>

namespace vibeqc {

using DLPNORowMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/// Read-only view of a row-major factor. The solver maps the global DF
/// factors in place instead of copying them (the virtual-virtual factor is
/// n_aux * n_vir^2 doubles, about 18 GB at C40H82/cc-pVDZ); rows must be
/// contiguous, i.e. outerStride() == cols().
using DLPNORowMatCRef = Eigen::Ref<const DLPNORowMat>;

/// One kept pair (i <= j) of the local solver, in its own PNO space.
struct DLPNOPairInput {
    int i = 0;
    int j = 0;
    /// (n_vir, n_pno) PNO coefficients in the canonical virtual basis;
    /// orthonormal columns, Fock-diagonal within the pair space.
    Eigen::MatrixXd U;
    /// (n_pno) semicanonical PNO energies, diag(U^T f_vv U).
    Eigen::VectorXd eps;
    /// (n_pno, n_pno) exchange block K_ab = (ia|jb) in the PNO basis.
    Eigen::MatrixXd K;
    /// (n_pno, n_pno) initial doubles t_ij^{ab} (semicanonical MP2).
    Eigen::MatrixXd T2;
    /// Sorted occupied coupling set; contains i and j.
    std::vector<int> L;
};

struct DLPNOCCSDPNOOptions {
    int max_iter = 100;
    double conv_tol_energy = 1e-9;
    double conv_tol_residual = 1e-7;
    int diis_size = 6;
};

struct DLPNOCCSDPNOResult {
    /// Converged doubles per pair, same order as the input.
    std::vector<Eigen::MatrixXd> T2;
    /// Converged singles per active occupied, in the diagonal pair's PNO
    /// space; an empty vector where no diagonal pair was kept.
    std::vector<Eigen::VectorXd> t1;
    /// Iterated correlation energy of the kept pairs (no screened-pair or
    /// PNO-truncation corrections; the caller adds those).
    double e_corr = 0.0;
    int n_iter = 0;
    bool converged = false;
    /// Per-iteration energy and max |R/D| (the DIIS error vector).
    std::vector<double> trace_e_corr;
    std::vector<double> trace_r_norm;
    /// Number of (pair, occupied) factor rows built outside the pair's own
    /// coupling set because some target pair's contraction needed them.
    std::size_t n_offlist_rows = 0;
    /// Resident bytes of the static per-pair density-fitting factors.
    std::size_t static_factor_bytes = 0;
    /// Wall-clock seconds spent building the static factors, and in the
    /// amplitude iterations (all of them together).
    double setup_seconds = 0.0;
    double iteration_seconds = 0.0;
};

/// Internal arbitrary-amplitude residual oracle. No iteration or corrections.
struct DLPNOCCSDResidualResult {
    std::vector<Eigen::VectorXd> R1;
    std::vector<Eigen::MatrixXd> R2;
    std::vector<std::map<std::string,Eigen::MatrixXd>> singles_terms,doubles_terms;
    double energy=0.;
};
DLPNOCCSDResidualResult dlpno_ccsd_pno_evaluate(
    const DLPNORowMatCRef&,const DLPNORowMatCRef&,const DLPNORowMatCRef&,
    const Eigen::MatrixXd&,const Eigen::MatrixXd&,const Eigen::MatrixXd&,
    std::vector<DLPNOPairInput>,const std::vector<Eigen::VectorXd>&,bool);

using DLPNOCCOperand=std::function<Eigen::MatrixXd(const std::string&,const std::vector<int>&)>;
std::uint64_t translation_ccsd_workspace(int,int,int,int,bool);
DLPNOCCSDResidualResult translation_ccsd_evaluate(
    const Eigen::MatrixXd&,int,int,int,const std::vector<int>&,const std::vector<int>&,
    const std::vector<double>&,const DLPNOCCOperand&,std::uint64_t,bool);

/// Solve the closed-shell DLPNO-CCSD amplitude equations in PNO spaces.
///
/// @param B_ov  (n_aux, n_act*n_vir) row-major DF factor B^P_{m v} over the
///              localised active occupieds m and canonical virtuals v.
/// @param B_oo  (n_aux, n_act*n_act) row-major B^P_{m n}.
/// @param B_vv  (n_aux, n_vir*n_vir) row-major B^P_{v w}.
/// @param f_oo  (n_act, n_act) localised occupied Fock block.
/// @param f_ov  (n_act, n_vir) occupied-virtual Fock block (zero for a
///              converged reference; kept general).
/// @param f_vv  (n_vir, n_vir) virtual Fock block.
/// @param pairs kept pairs with their PNO data; taken by value.
DLPNOCCSDPNOResult dlpno_ccsd_pno_solve(
    const DLPNORowMatCRef& B_ov,
    const DLPNORowMatCRef& B_oo,
    const DLPNORowMatCRef& B_vv,
    const Eigen::MatrixXd& f_oo,
    const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& f_vv,
    std::vector<DLPNOPairInput> pairs,
    const DLPNOCCSDPNOOptions& opts);

/// Result of the compiled local closed-shell (T0) correction.
struct DLPNOTriplesResult {
    /// Sum of the per-key contributions.
    double e_t = 0.0;
    /// TNO count and energy per occupied key, input order.
    std::vector<int> n_tno;
    std::vector<double> e_key;
    double seconds = 0.0;
};

/// Semicanonical MP2 energy of every active occupied pair over the whole
/// canonical virtual space, the pair-screening estimate of the local CCSD
/// route (python/vibeqc/dlpno/ccsd_local_solver.py):
///
///   e_ij = (2 - delta_ij) sum_ab K_ab (2 K_ab - K_ba) / (f_ii + f_jj - e_a - e_b),
///   K_ab = sum_P B^P_{ia} B^P_{jb}.
///
/// ``B_ov`` is (n_aux, n_act*n_vir) row-major over the localised active
/// occupieds; ``f_dd`` their diagonal Fock elements and ``eps_v`` the
/// canonical virtual energies. Returns the symmetric (n_act, n_act) matrix
/// of pair energies. These are the sums of the NumPy loop it replaces,
/// evaluated one GEMM per pair in parallel; that loop is O(n_act^2 n_vir^2
/// n_aux) and ran on one BLAS thread.
Eigen::MatrixXd dlpno_sc_mp2_pair_energies(const DLPNORowMatCRef& B_ov,
                                           const Eigen::VectorXd& f_dd,
                                           const Eigen::VectorXd& eps_v);

/// Local semicanonical closed-shell (T0) on converged PNO amplitudes.
///
/// ``keys`` are the distinct occupied sets (one to three sorted occupieds)
/// of python/vibeqc/dlpno/triples_local.py::_local_distinct_triple_keys.
/// Each key's TNO space is the orthonormalised union of the PNO spaces of
/// its kept pairs (diagonal pairs included; singular values below ``lindep``
/// times the largest dropped), optionally truncated to the eigenvectors of
/// the projected pair densities with occupation above ``tcut_tno``, and
/// semicanonicalised against ``f_vv``. The key's ordered triples are then
/// evaluated with the closed-shell W/V expression of Rendell, Lee and
/// Komornicki (Chem. Phys. Lett. 178, 462 (1991)); the occupied sum of its
/// connected term runs over every l whose pair with the triple's occupied
/// is kept. Only those pairs are projected and only the key's
/// density-fitting blocks are formed. ``eps_o`` is the diagonal of the
/// localised occupied Fock (the T0 approximation); ``T1`` holds each
/// occupied's singles in the full canonical virtual space.
DLPNOTriplesResult dlpno_ccsd_local_triples(
    const DLPNORowMatCRef& B_ov, const DLPNORowMatCRef& B_oo,
    const DLPNORowMatCRef& B_vv, const Eigen::VectorXd& eps_o,
    const Eigen::MatrixXd& f_vv, const std::vector<int>& pair_i,
    const std::vector<int>& pair_j, const std::vector<Eigen::MatrixXd>& U,
    const std::vector<Eigen::MatrixXd>& T2, const Eigen::MatrixXd& T1,
    const std::vector<std::vector<int>>& keys, double tcut_tno, double lindep);

}  // namespace vibeqc
