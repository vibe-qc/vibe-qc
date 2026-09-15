// Closed-shell CCSD and perturbative (T) on an RHF reference,
// density-fitted by default or with exact four-index integrals on the
// canonical conventional route (CCSDOptions.density_fit = false).
//
// Equations: the closed-shell spin integration of the spin-orbital CCSD
// working equations of J. F. Stanton, J. Gauss, J. D. Watts, R. J.
// Bartlett, J. Chem. Phys. 94, 4334 (1991), doi:10.1063/1.460620,
// Eqs. (1)-(13).  The spin-orbital form is implemented verbatim in
// python/vibeqc/dlpno/_ccsd_ref.py (FCI-anchored); the spatial equations
// implemented in cpp/src/ccsd.cpp were validated against that kernel to
// machine precision at arbitrary amplitudes before transcription, and the
// converged correlation energies are pinned to it by
// tests/test_ccsd_anchor.py (H2O/STO-3G and H2O/def2-SVP, < 5e-6 Ha).
// The full spatial equation set, in the exact form implemented, is
// documented above compute_residuals in cpp/src/ccsd.cpp.
//
// Further references:
//   G. D. Purvis III, R. J. Bartlett, J. Chem. Phys. 76, 1910 (1982),
//     doi:10.1063/1.443164.  Original CCSD.
//   A. E. DePrince III, C. D. Sherrill, J. Chem. Theory Comput. 9, 2687
//     (2013), doi:10.1021/ct400250u.  DF-CCSD: all two-electron integrals
//     here are assembled from MO-basis three-index DF tensors,
//     (pq|rs) = sum_P B^P_pq B^P_rs.
//   K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon,
//     Chem. Phys. Lett. 157, 479 (1989), doi:10.1016/0009-2614(89)87395-6.
//     The (T) correction; evaluated classwise over closed-shell spin
//     blocks (see compute_triples in ccsd.cpp).
//   P. Pulay, Chem. Phys. Lett. 73, 393 (1980).  DIIS, applied to the
//     joint (T1, T2) amplitude/residual vectors.
//
// Conventions: i..n = occupied, a..f = virtual spatial MOs; chemists'
// integrals (pq|rs); T2 stored flat as T2(i*no+j, a*nv+b) = t_ij^ab in
// the alpha-beta convention (t_ij^ab = t_ji^ba).  Frozen-core: the
// lowest n_frozen_core occupied orbitals are excluded from the
// correlated window and enter only through the converged Fock operator.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <string>
#include <vector>

#include "basis.hpp"
#include "molecule.hpp"
#include "rhf.hpp"

namespace vibeqc {

// ==========================================================================
// Options
// ==========================================================================

struct CCSDOptions {
    // Density fitting.  When true, the two-electron integrals are built
    // from the DF three-index B-tensor; when false, the canonical
    // four-index ERI path is used (much slower for CC).  Default true:
    // DF is the standard approach for CC.
    bool density_fit = true;
    std::string aux_basis;

    // Iteration control.
    int max_iter = 100;
    double conv_tol_energy = 1e-8;      // Ha — energy change threshold
    double conv_tol_residual = 1e-7;    // ‖R1‖ + ‖R2‖ convergence

    // DIIS subspace size (Pulay 1980).  Set to 0 to disable DIIS.
    std::size_t diis_subspace_size = 6;

    // Frozen-core: exclude the n_frozen_core lowest-energy occupied MOs
    // from the correlated window.  -1 (the default) resolves the published
    // ORCA 6.1 Table 2.69 count; 0 explicitly requests all-electron
    // correlation; positive values are explicit counts.  The convention is
    // count-only and does not reorder orbitals by atomic character.  Frozen
    // virtuals are not supported.
    int n_frozen_core = -1;

    // (T) triples correction.
    bool compute_triples = true;

    // Which perturbative-triples correction lands in result.e_t when
    // compute_triples is true (closed-shell kernel only; run_uccsd
    // rejects anything but "(t)"):
    //   "(t)" (default)  standard Raghavachari CCSD(T) = E[T] + E_ST
    //   "[t]"            bracket correction CCSD[T] = E[T] only
    //                    (identically Urban's CCSD+T(CCSD): Urban, Noga,
    //                    Cole, Bartlett, J. Chem. Phys. 83, 4041 (1985),
    //                    doi:10.1063/1.449067).
    // Both components are always reported on the result (e_t4 / e_t5_st).
    // e_t5_st is the full disconnected fifth-order contribution: T1*ERI
    // plus the Fov*T2 completion required for semicanonical orbitals.
    std::string triples_variant = "(t)";

    // Perturbative-triples memory strategy:
    //   "auto"    choose the fastest strategy that fits requested_memory_bytes;
    //   "fast"    dense per-thread virtual cubes (the historical GEMM path);
    //   "blocked" bounded virtual-a tiles, evaluated directly from DF factors;
    //   "direct"  scalar W/Wd evaluation with no virtual-cube workspace;
    //   "disk"    spill DF B_ov/B_vv factors to scratch and stream bounded
    //              auxiliary rows while retaining tiled virtual work arrays.
    // The legacy spelling "low" is accepted as an alias for "blocked".
    std::string triples_memory_mode = "auto";

    // Requested native-memory budget for the triples phase, in bytes.  The
    // planner counts the retained amplitudes/integral inputs and the aggregate
    // workspace of every active OpenMP thread.  Zero means no explicit cap.
    std::size_t requested_memory_bytes = 0;

    // Optional planner controls. Zero selects an automatic tile/thread count.
    // triples_tile_size is an upper bound on the blocked virtual-a tile.
    int triples_tile_size = 0;
    int triples_max_threads = 0;

    // Optional scratch directory used by the disk factor-spill backend. Empty
    // selects the platform temporary directory. Scratch files are unique and
    // removed by RAII on both successful return and exception unwinding.
    std::string triples_scratch_directory;

    // Approximate coupled-cluster / coupled-pair variant of the amplitude
    // equations (closed-shell
    // kernel only; run_uccsd rejects anything but "ccsd"):
    //   "ccsd"    (default) full CCSD.
    //   "cc2"     second-order approximate CC singles and doubles:
    //             full singles projection and first-order doubles equation
    //             (Christiansen, Koch & Jorgensen, CPL 243, 409 (1995)).
    //   "ccd"     CCD: T1 frozen at zero (Cizek 1966).
    //   "bccd"    Brueckner CCD: T1 frozen at zero after the caller has
    //             supplied Brueckner orbitals.
    //   "lccd"    linearized CCD (doubles residual truncated to the
    //             terms linear in T2; T1 frozen at zero).
    //   "lccsd"   linearized CCSD == CEPA(0) ("cepa(0)" is an alias).
    //   "cepa(1)", "cepa(2)", "cepa(3)"
    //             Meyer's coupled-electron-pair approximations: the
    //             linearized residual with pair-specific EPV shifts
    //             -Delta_ij t_ij^ab (Meyer, J. Chem. Phys. 58, 1017
    //             (1973); shift table as in Wennmohs & Neese, Chem.
    //             Phys. 343, 217 (2008)).
    // The linear variants use the linear energy functional (no tau
    // t1*t1 term).  (T) is available for CCSD, BCCD, and QCISD; the
    // remaining coupled-pair variants reject compute_triples=true.
    std::string cc_variant = "ccsd";
};

// ==========================================================================
// Iteration trace record
// ==========================================================================

struct CCSDIteration {
    int iter = 0;
    double energy = 0.0;
    double delta_e = 0.0;
    double r1_norm = 0.0;
    double r2_norm = 0.0;
    int diis_subspace = 0;
};

// ==========================================================================
// Result struct — return value of run_ccsd
// ==========================================================================

struct CCSDResult {
    double e_hf = 0.0;
    double e_ccsd_correlation = 0.0;
    double e_ccsd = 0.0;          // e_hf + e_ccsd_correlation
    double e_t = 0.0;             // selected triples correction (0 if not
                                  // computed): E[T]+E_ST for "(t)", E[T]
                                  // for "[t]"
    double e_t4 = 0.0;            // fourth-order piece E[T] (CCSD[T])
    double e_t5_st = 0.0;         // fifth-order singles-triples piece E_ST
    double e_ccsd_t = 0.0;        // e_ccsd + e_t
    double e_total = 0.0;         // alias for e_ccsd_t

    int n_iter = 0;
    bool converged = false;

    double t1_norm = 0.0;         // Frobenius norm of final T1
    double t2_norm = 0.0;         // Frobenius norm of final T2
    Eigen::MatrixXd t1_amplitudes; // final T1 amplitudes (n_occ_active x n_vir)

    // Realised triples memory plan. Empty/zero when triples were not run.
    // triples_workspace_bytes is the full modeled triples peak (retained
    // inputs plus aggregate active-thread scratch), so it is directly
    // comparable to requested_memory_bytes.
    std::string triples_memory_mode_used;
    int triples_tile_size_used = 0;
    int triples_threads_used = 0;
    std::size_t triples_workspace_bytes = 0;
    std::size_t triples_disk_bytes = 0;

    std::vector<CCSDIteration> cc_trace;
};

// ==========================================================================
// Main entry point
// ==========================================================================

// Run closed-shell CCSD (and optionally CCSD(T)) on a converged RHF
// reference.  `rhf` supplies the converged MO coefficients and orbital
// energies.  `mol` and `basis` define the AO space for the integral build
// (DF by default, exact four-index when opts.density_fit is false).
//
// Throws std::runtime_error if `rhf.converged` is false, if the system
// is not closed-shell, or if the DF build fails.
//
// When `opts.compute_triples` is true, the (T) correction is appended
// to `result.e_t` and `result.e_ccsd_t`.
CCSDResult run_ccsd(const Molecule& mol,
                    const BasisSet& basis,
                    const RHFResult& rhf,
                    const CCSDOptions& opts = {});

// Run closed-shell DF-CCSD(T) from explicit MO coefficients and an AO
// Fock matrix, rather than an RHFResult.  Low-level entry point for
// references that do not produce a converged RHFResult -- in particular
// frozen-natural-orbital (FNO) CCSD(T), where the virtual space has been
// truncated to the dominant MP2 natural orbitals and semicanonicalized
// (DePrince-Sherrill, J. Chem. Theory Comput. 9, 2687 (2013)).
//
// `C` is (n_ao, n_occ_full + n_vir_kept): the occupied columns followed by
// the retained (semicanonical) virtual columns.  `F` is the AO Fock matrix
// (rotation-invariant, so the converged RHF AO Fock is passed unchanged).
// Orbital energies are taken as diag(C^T F C): canonical for the occupied
// block, semicanonical for the truncated virtuals (f_vv diagonal), which
// keeps the (T) canonical-orbital denominators valid.  `e_hf` is the SCF
// reference energy, attached as result.e_hf. `ecp_total_ncore` is mandatory
// provenance because these raw arrays cannot reveal whether their SCF
// Hamiltonian removed core electrons; the occupied space is partitioned
// from mol.n_electrons() - ecp_total_ncore, and the published frozen-core
// default (-1) is refused for a nonzero count (the caller resolves the
// ECP-aware count in Python and passes it explicitly).
//
// Throws std::invalid_argument if the molecule is not closed-shell or the
// matrix dimensions are inconsistent.
CCSDResult run_ccsd_from_mos(const Molecule& mol,
                             const BasisSet& basis,
                             const Eigen::MatrixXd& C,
                             const Eigen::MatrixXd& F,
                             double e_hf,
                             int ecp_total_ncore,
                             const CCSDOptions& opts = {});

// ==========================================================================
// DLPNO per-pair residual kernel
// ==========================================================================
// A single closed-shell DF-CCSD residual evaluation (no SCF loop): the
// per-pair kernel the DLPNO local solver
// (python/vibeqc/dlpno/ccsd_local_solver.py) calls in place of the numpy
// `cs_ccsd_residual`. Row-major DF B-tensors B_ov (n_aux x no*nv),
// B_oo (n_aux x no*no), B_vv (n_aux x nv*nv); amplitudes T1 (no x nv),
// T2_flat (no*no x nv*nv). Builds the chemist-notation integral blocks and
// applies the validated `compute_residuals`; returns the SGWB residuals
// R1 (no x nv) and R2_flat (no*no x nv*nv).
//
// `include_ladder=false` omits the particle-particle (W_abef) term and the
// O(nv^4) (ae|bf) build that exists only for it. The DLPNO extended-domain
// path (issue #700) contracts that term in each pair's own PNO space instead,
// where it is exact and costs n_pno^4 rather than n_ext^4; a caller that does
// NOT contract it elsewhere would silently get a wrong residual.
void dlpno_pair_residual(
    const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_ov,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_oo,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_vv,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
    const Eigen::MatrixXd& f_ov,
    Eigen::MatrixXd& R1, Eigen::MatrixXd& R2_flat,
    bool include_ladder = true, bool include_ring = true,
    Eigen::MatrixXd* W1_out = nullptr, Eigen::MatrixXd* W2_out = nullptr,
    Eigen::MatrixXd* WX_out = nullptr);

// Target-selective counterpart used by the local DLPNO solver. The coupled
// amplitudes and intermediates are unchanged, but only residual row target_i
// and pair block (target_i,target_j) are assembled. This avoids computing and
// discarding every other occupied-pair residual for each PNO-domain call.
void dlpno_target_pair_residual(
    const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_ov,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_oo,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_vv,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
    const Eigen::MatrixXd& f_ov,
    Eigen::Index target_i, Eigen::Index target_j,
    Eigen::MatrixXd& R1_target, Eigen::MatrixXd& R2_target,
    bool include_ladder = true, bool include_ring = true,
    Eigen::MatrixXd* W1_out = nullptr, Eigen::MatrixXd* W2_out = nullptr,
    Eigen::MatrixXd* WX_out = nullptr);

// Closed-shell spatial CCSD(T) triples contraction over caller-supplied DF
// blocks. This is the native counterpart of
// python/vibeqc/dlpno/_ccsd_cs.py::cs_triples_correction and lets the DLPNO
// local triples path reuse the validated OpenMP C++ kernel without rebuilding
// integrals from a molecule/basis object.
double dlpno_spatial_triples_correction(
    const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_ov,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_oo,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& B_vv,
    const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v);

// Energy contribution for one ordered occupied triple (i,j,k), using local
// chemist-notation integral blocks already assembled over the caller's virtual
// domain. This is the native counterpart of
// python/vibeqc/dlpno/_ccsd_cs.py::cs_triple_energy for the screened/TNO
// DLPNO triples loops.
double dlpno_spatial_triple_energy(
    Eigen::Index i, Eigen::Index j, Eigen::Index k,
    const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& ov_vv,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& oo_ov,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& ov_ov,
    const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v);

// ==========================================================================
// Spatial (Python-bindable) CCSD residual and solver.
// These accept the same flat-matrix (no*nv, no*nv) integral-block layout
// used by the internal kernel, enabling Python callers to pass pre-built
// numpy arrays without conversion.  The caller reshapes its 4D tensors to
// 2D before calling (no copy needed for C-contiguous arrays with RowMajor
// Eigen maps).
// ==========================================================================

void spatial_ccsd_residual(
    const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
    const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& ovov_flat, const Eigen::MatrixXd& oooo_flat,
    const Eigen::MatrixXd& ooov_flat, const Eigen::MatrixXd& oovv_flat,
    const Eigen::MatrixXd& ovvv_flat, const Eigen::MatrixXd& vvvv_flat,
    Eigen::MatrixXd& R1, Eigen::MatrixXd& R2_flat);

struct SpatialCCSDResult {
    Eigen::MatrixXd T1;
    Eigen::MatrixXd T2_flat;
    double e_corr = 0.0;
    int n_iter = 0;
    bool converged = false;
};

SpatialCCSDResult spatial_ccsd_solve(
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
    const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& ovov_flat, const Eigen::MatrixXd& oooo_flat,
    const Eigen::MatrixXd& ooov_flat, const Eigen::MatrixXd& oovv_flat,
    const Eigen::MatrixXd& ovvv_flat, const Eigen::MatrixXd& vvvv_flat,
    double e_hf,
    int max_iter = 100,
    double conv_tol_energy = 1e-8,
    double conv_tol_residual = 1e-7,
    int diis_size = 6);

// Closed-shell A-CCSD(T) Lambda triples correction.  The right-hand
// triples moment is built from the converged CCSD amplitudes (T1,T2);
// the left-hand moment uses the Lambda multipliers (L1,L2).
double spatial_lambda_triples(
    const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
    const Eigen::MatrixXd& L1, const Eigen::MatrixXd& L2_flat,
    const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& ov_vv, const Eigen::MatrixXd& oo_ov,
    const Eigen::MatrixXd& ov_ov,
    const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v);

// Project DLPNO pair amplitudes from their per-pair PNO bases into a common
// TNO (triples natural orbital) basis.  For each pair (i,j) the PNO
// coefficients U_pno_k (nv x n_pno_k) and amplitudes T2_pno_k
// (n_pno_k x n_pno_k) are projected through the TNO basis V_T (nv x n_T):
//
//   S = V_T^T * U_pno_k              (n_T x n_pno_k)
//   T2_TNO[i,j] = S * T2_pno_k * S^T  (n_T x n_T)
//   t1_TNO[i]   = V_T^T * t1[i]       (n_T,)
//
// The output t1_out is (n_act, n_T) row-major; T2_out is
// (n_act*n_act, n_T*n_T) row-major.  Pairs not in the pair lists leave
// their block zero.
void dlpno_project_tno_amplitudes(
    const Eigen::MatrixXd& V_T,
    int n_act,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& U_pno,
    const std::vector<Eigen::MatrixXd>& T2_pno,
    const std::vector<int>& t1_idx,
    const std::vector<Eigen::VectorXd>& t1_vec,
    Eigen::MatrixXd& t1_out,
    Eigen::MatrixXd& T2_out);

// Build the TNO occupation density D = sum_{p,q in distinct} T_{pq} T_{pq}^T
// + (T_{pq} T_{pq}^T)^T, diagonalise it, and truncate V_T to the eigenvectors
// whose occupation exceeds tcut_tno.  Returns the truncated V_T (may be
// smaller than the input).  When tcut_tno <= 0 or V_T has <= 1 column, the
// input V_T is returned unchanged.  This replaces the Python double-loops in
// triples_local.py build_tno (lines 475-487) when TNO truncation is active.
Eigen::MatrixXd dlpno_build_tno_density(
    const Eigen::MatrixXd& V_T,
    const std::vector<int>& distinct,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& U_pno,
    const std::vector<Eigen::MatrixXd>& T2_pno,
    double tcut_tno);

}  // namespace vibeqc
