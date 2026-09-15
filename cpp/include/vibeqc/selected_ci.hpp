// Selected CI (CIPSI / heat-bath style) over arbitrary SpinDet lists.
//
// The variational engine behind vibeqc.solvers.selected_casci's C++
// backend (roadmap 25i): grow a selected subset of an active space's
// determinants by perturbative importance, diagonalize H exactly in the
// selected space (sparse build + block Davidson), and provide spin-summed
// 1-/2-RDMs of the truncated wavefunction.
//
// Conventions (matching the Python engine, solvers/_selected_ci.py and
// solvers/_slater_condon.py):
//  * A determinant is an (alpha_mask, beta_mask) pair of uint64 occupation
//    bitmasks over n_act <= 64 active spatial orbitals.
//  * Jordan-Wigner phases are PER SPIN SECTOR (popcount of set bits below
//    the target orbital within that sector's mask); cross-sector phases
//    cancel pairwise for the particle-conserving operators used here.
//  * ``eri_chem`` is the chemist-ordered active ERI block (pq|rs),
//    row-major n_act^4 (same layout casci_direct_solve consumes).
//  * Selection: candidates D outside the space are scored by the coherent
//    multi-root CIPSI estimate
//        w_D = sum_k |<D|H|Psi_k>|^2 / |E_k - H_DD|
//    accumulated over every significant variational determinant
//    (|c_I| > significant_coeff), Epstein-Nesbet denominators: the same
//    criterion as the Python kernel.  ``select_eps`` optionally pre-filters
//    contributions with |H_DI c_I| below the bound (heat-bath style) for
//    large runs; 0 disables (exact CIPSI scoring).
//  * RDMs are spin-summed in the PySCF convention
//    (dm1[p,q] = <a^+_p a_q>, dm2[p,q,r,s] = <a^+_p a^+_r a_s a_q>),
//    matching solvers._rdm.make_rdm12 on the same determinant list,
//    exact for truncated lists (direct two-body Slater-Condon between
//    list determinants; no intermediate resolution outside the space).

#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <vector>

namespace vibeqc {

struct SelectedCIOptionsCpp {
    int nroots = 1;
    int max_cycles = 30;
    int target_size = 200000;
    int max_new_per_cycle = 20000;
    double conv_tol_energy = 1e-9;
    double pt2_threshold = 1e-9;
    double significant_coeff = 0.005;
    double select_eps = 0.0;     // heat-bath prefilter on |H_DI c_I|; 0 = off
    // Walk presorted |H|-ordered double-excitation lists instead of
    // enumerating every virtual pair when select_eps > 0 (Holmes-Tubman-
    // Umrigar 2016); identical accepted-candidate set, enumeration stops
    // at the first sub-threshold entry.  false = brute-force prefilter
    // at the same eps (the equivalence-test hook).
    bool use_heat_bath_walk = true;
    bool spin_complete = true;   // close M_s = 0 spaces under alpha<->beta
    int davidson_max_iter = 300;
    double davidson_tol = 1e-10;
};

struct SelectedCIResultCpp {
    std::vector<std::uint64_t> dets_a;  // selected determinants (insertion
    std::vector<std::uint64_t> dets_b;  // order; index-aligned pairs)
    Eigen::VectorXd eigenvalues;        // lowest nroots, active-space only
    Eigen::MatrixXd ci;                 // (ndet, nroots)
    int n_cycles = 0;
    bool converged = false;
};

// Grow-and-diagonalize selected CI in the active space.  ``guess_a/b``
// (optional, index-aligned) seed the variational space, e.g. the previous
// CASSCF macro-iteration's selected set; empty selects the aufbau
// reference.  ``h1`` is the (core-dressed) active one-electron matrix.
SelectedCIResultCpp selected_ci_solve(
    const Eigen::MatrixXd& h1,
    const std::vector<double>& eri_chem,
    int n_act,
    int n_alpha,
    int n_beta,
    const SelectedCIOptionsCpp& opts = {},
    const std::vector<std::uint64_t>& guess_a = {},
    const std::vector<std::uint64_t>& guess_b = {});

// Spin-summed (rdm1, rdm2) of ``ci`` over an arbitrary determinant list
// (PySCF convention; see header notes).  rdm1 is (n_act, n_act); rdm2 is
// row-major n_act^4.
void selected_ci_rdm12(const std::vector<std::uint64_t>& dets_a,
                       const std::vector<std::uint64_t>& dets_b,
                       const Eigen::VectorXd& ci,
                       int n_act,
                       Eigen::MatrixXd& rdm1,
                       std::vector<double>& rdm2);

// Epstein-Nesbet PT2 on a selected wavefunction (Sharma, Holmes,
// Jeanmairet, Alavi & Umrigar, JCTC 13, 1595 (2017)).  ``e0`` is the
// root's variational ACTIVE-SPACE eigenvalue (same frame as the dressed
// ``h1``); ``ci`` is that root's coefficients over (dets_a, dets_b).
//
// Deterministic: Eq. 5 -- sum_a (sum_i^{(eps2)} H_ai c_i)^2/(E0 - H_aa),
// keeping contributions with |H_ai c_i| >= eps2 (eps2 = 0 keeps all);
// the heat-bath walk prunes the double-excitation enumeration under the
// same predicate.  Perturbers with |E0 - H_aa| < 1e-10 are skipped.
double selected_ci_en_pt2_deterministic(
    const Eigen::MatrixXd& h1, const std::vector<double>& eri_chem,
    int n_act, const std::vector<std::uint64_t>& dets_a,
    const std::vector<std::uint64_t>& dets_b, const Eigen::VectorXd& ci,
    double e0, double eps2, bool use_heat_bath_walk,
    std::int64_t* n_perturbers_out = nullptr);

// Stochastic difference term of the semistochastic scheme (Eq. 11):
// mean +/- stderr over ``n_samples`` independent batches of the unbiased
// estimator (Eq. 10) of  E2[eps2] - E2[eps2_loose],  each batch sampling
// ``sample_size`` determinants with replacement from p_i = |c_i|/sum|c|
// (Eq. 7).  Total:  E2 = deterministic(eps2_loose) + mean.  With
// eps2_loose == eps2 the difference is identically zero; with
// eps2_loose = +inf this is the fully stochastic estimate of E2[eps2].
void selected_ci_en_pt2_stochastic(
    const Eigen::MatrixXd& h1, const std::vector<double>& eri_chem,
    int n_act, const std::vector<std::uint64_t>& dets_a,
    const std::vector<std::uint64_t>& dets_b, const Eigen::VectorXd& ci,
    double e0, double eps2, double eps2_loose, int n_samples,
    int sample_size, std::uint64_t seed, bool use_heat_bath_walk,
    double* mean_out, double* stderr_out);

}  // namespace vibeqc
