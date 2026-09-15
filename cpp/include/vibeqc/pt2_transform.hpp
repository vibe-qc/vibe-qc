#pragma once
/// C++ 4-index MO integral transform kernel.
///
/// Replaces the Python einsum bottleneck in
/// _compute_caspt2_zvector_correction and _compute_nevpt2_zvector_correction.

#include <Eigen/Dense>

namespace vibeqc {

/// Transform 4-index MO integrals: g_new[p,r,q,s] = sum_abcd U[a,p] U[b,r] U[c,q] U[d,s] g[a,b,c,d].
/// g_phys is in physicist's notation (pr|qs) flat row-major: idx(p,r,q,s) = ((p*norb+r)*norb+q)*norb+s.
/// U is (norb, norb).  Returns flat-packed (norb^4) vector in the same ordering.
/// Uses OpenMP collapse(2) over the 4-index output loops when available.
Eigen::VectorXd transform_4index_mo(
    const Eigen::VectorXd& g_phys_flat,
    const Eigen::MatrixXd& U,
    int norb);

}  // namespace vibeqc
