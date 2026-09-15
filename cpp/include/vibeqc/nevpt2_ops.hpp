#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vibeqc {

// Spin-orbital second-quantization helpers (match Python _mrpt.py conventions).
// Spin-orbital index of spatial orbital p with spin s (0=alpha, 1=beta)
// is p + s * norb.

inline int _so(int p, int s, int norb) { return p + s * norb; }

// Phase-aware annihilation: returns (sign, new_mask) or (0, 0) if not occupied.
inline std::pair<int, uint64_t> _ann(uint64_t mask, int x) {
    if (!((mask >> x) & 1)) return {0, 0};
    int sign = (__builtin_popcountll(mask & ((1ULL << x) - 1)) & 1) ? -1 : 1;
    return {sign, mask & ~(1ULL << x)};
}

// Phase-aware creation: returns (sign, new_mask) or (0, 0) if already occupied.
inline std::pair<int, uint64_t> _cre(uint64_t mask, int x) {
    if ((mask >> x) & 1) return {0, 0};
    int sign = (__builtin_popcountll(mask & ((1ULL << x) - 1)) & 1) ? -1 : 1;
    return {sign, mask | (1ULL << x)};
}

// Apply sum_{pq} h1[p,q] * sum_s a^dagger_{ps} a_{qs} to a sparse determinant state.
std::unordered_map<uint64_t, double> apply_1body_cpp(
    const std::unordered_map<uint64_t, double>& state,
    const Eigen::MatrixXd& h1,
    int norb,
    const std::vector<int>* idx = nullptr);

// Apply 1/2 sum_{pqrs} eri(p,q,r,s) * sum_{st} a^dagger_{ps} a^dagger_{rt} a_{st} a_{qs}
// to a sparse determinant state.  eri is chemist's (pq|rs) stored flat row-major:
// index(p,q,r,s) = ((p*norb + q)*norb + r)*norb + s.
std::unordered_map<uint64_t, double> apply_2body_cpp(
    const std::unordered_map<uint64_t, double>& state,
    const std::vector<double>& eri,
    int norb,
    const std::vector<int>* idx = nullptr);

// Dot product of two sparse determinant vectors.
double dot_cpp(const std::unordered_map<uint64_t, double>& a,
               const std::unordered_map<uint64_t, double>& b);

// Add two sparse determinant vectors (a + b).
std::unordered_map<uint64_t, double> add_cpp(
    const std::unordered_map<uint64_t, double>& a,
    const std::unordered_map<uint64_t, double>& b);

// Compute 1-RDM and 2-RDM (PySCF convention) from a sparse determinant state.
// rdm1[p,q] = <psi|E_pq|psi>, rdm2[p,q,r,s] = <psi|a+_p a+_r a_s a_q|psi>.
// Both are row-major: rdm1.data()[p + q*norb], rdm2.data()[((p*norb+q)*norb+r)*norb+s].
void state_rdm12_cpp(
    const std::unordered_map<uint64_t, double>& state,
    int norb,
    Eigen::MatrixXd& rdm1,
    Eigen::MatrixXd& rdm2);  // rdm2 is (norb^2, norb^2) flat

// Compute 3-RDM (E-operator convention, OpenMolcas G3) from a sparse determinant state.
// G3[t,u,v,x,y,z] = <psi| E_tuvxyz |psi>  (E-operator convention).
// Stored flat row-major: index = ((((t*n+v)*n+x)*n+y)*n+z)*n+u where n = n_act.
// All indices are active-relative (0..n_act-1).
//
// Algorithm: mkfg3.F90 E-operator triple-product with delta-function subtractions.
void state_rdm3_cpp(
    const std::unordered_map<uint64_t, double>& state,
    int norb,
    int n_act,
    std::vector<double>& rdm3);

}  // namespace vibeqc
