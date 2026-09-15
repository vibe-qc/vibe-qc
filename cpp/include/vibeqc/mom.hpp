// Maximum-overlap-method (MOM) occupied-orbital selection for the molecular
// UHF / UKS SCF loops.
//
// This is the C++-runtime mirror of python/vibeqc/mom.py
// (``select_occupied_by_max_overlap``), kept algorithmically identical to the
// Python version the periodic multi-k drivers use. The molecular SCF runs in
// C++ and the periodic multi-k SCF in Python, so the kernel is mirrored across
// the two runtimes the same way the Saunders-Hillier level shift already is
// (uhf.cpp vs periodic_uhf_ewald.py); keep the two copies in lock-step.
//
// Method: Gilbert, Besley & Gill, J. Phys. Chem. A 112, 13164 (2008).
//
// Used by the SPINLOCK pattern-hold: for the first ``spinlock_iterations``
// SCF cycles the occupied orbitals are chosen by maximum overlap with the
// previous cycle rather than by pure aufbau, so a broken-symmetry seed
// (ATOMSPIN / atomic_spins) is held against collapse to the symmetric
// solution, then released.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <numeric>
#include <vector>

namespace vibeqc {

// Indices of the ``n_occ`` columns of ``C_new`` whose summed
// |<C_prev_occ | S | C_new>|^2 is largest (the new orbitals most overlapping
// the previous occupied subspace). Mirror of mom.py select_occupied_by_max_overlap.
inline std::vector<int> mom_select_occupied(
    const Eigen::MatrixXd& C_new, const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& C_prev_occ, int n_occ) {
    // O(i,j) = <prev_occ_i | S | C_new_j>; proj_j = sum_i O(i,j)^2.
    const Eigen::MatrixXd O = C_prev_occ.transpose() * S * C_new;
    const Eigen::VectorXd proj = O.colwise().squaredNorm();
    std::vector<int> idx(static_cast<std::size_t>(C_new.cols()));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(), idx.begin() + n_occ, idx.end(),
        [&](int a, int b) { return proj(a) > proj(b); });
    idx.resize(static_cast<std::size_t>(n_occ));
    return idx;
}

// Permute ``C_new`` / ``eps_new`` in place so the ``n_occ`` MOM-selected
// occupied columns come first (sorted by energy among themselves, preserving
// the canonical occupied-first-by-energy convention), then the remaining
// columns in their original order. After this, ``leftCols(n_occ)`` is the held
// occupied set, so the standard aufbau density build picks up the MOM pattern
// with no further change. No-op on degenerate inputs.
inline void mom_reorder_occupied(
    Eigen::MatrixXd& C_new, Eigen::VectorXd& eps_new, const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& C_prev_occ, int n_occ) {
    if (n_occ <= 0 || C_prev_occ.cols() != n_occ || C_new.cols() < n_occ) {
        return;
    }
    std::vector<int> occ = mom_select_occupied(C_new, S, C_prev_occ, n_occ);
    std::sort(occ.begin(), occ.end(),
              [&](int a, int b) { return eps_new(a) < eps_new(b); });
    std::vector<char> is_occ(static_cast<std::size_t>(C_new.cols()), 0);
    for (int j : occ) is_occ[static_cast<std::size_t>(j)] = 1;
    std::vector<int> order = occ;
    for (int j = 0; j < static_cast<int>(C_new.cols()); ++j) {
        if (!is_occ[static_cast<std::size_t>(j)]) order.push_back(j);
    }
    Eigen::MatrixXd C_perm(C_new.rows(), C_new.cols());
    Eigen::VectorXd eps_perm(eps_new.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
        C_perm.col(static_cast<Eigen::Index>(k)) = C_new.col(order[k]);
        eps_perm(static_cast<Eigen::Index>(k)) = eps_new(order[k]);
    }
    C_new = std::move(C_perm);
    eps_new = std::move(eps_perm);
}

}  // namespace vibeqc
