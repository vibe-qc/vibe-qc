// C++ OpenMP far-field gradient contractor.
//
// Port of Python's build_bipolar_far_field_gradient_contribution.
// OpenMP-parallelised over quartets with per-thread tensor caching
// and force accumulation.

#include "vibeqc/bipole_far_field_gradient.hpp"
#include "vibeqc/bipole_contractor.hpp"
#include "vibeqc/bipole_multipole.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <tuple>

namespace vibeqc {

namespace {

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

// Key for per-thread interaction-tensor gradient cache:
// (rx_bin, ry_bin, rz_bin, L, mu_bin).
using GradCacheKey = std::tuple<int, int, int, int, int>;

struct GradCacheKeyHash {
    std::size_t operator()(const GradCacheKey& k) const noexcept {
        const std::uint64_t a = static_cast<std::uint64_t>(std::get<0>(k) + 524288);
        const std::uint64_t b = static_cast<std::uint64_t>(std::get<1>(k) + 524288);
        const std::uint64_t c = static_cast<std::uint64_t>(std::get<2>(k) + 524288);
        const std::uint64_t d = static_cast<std::uint64_t>(std::get<3>(k));
        const std::uint64_t e = static_cast<std::uint64_t>(std::get<4>(k) + 524288);
        return static_cast<std::size_t>(
            (a * 73856093ULL) ^ (b * 19349663ULL) ^ (c * 83492791ULL)
            ^ (d * 222334565ULL) ^ e);
    }
};

// Round to 1e-5 bins.
inline int bin_coord(double x) {
    return static_cast<int>(std::round(x * 100000.0));
}

// Prototype screening composition. Saunders (1992), Eq. (91), is a radial
// derivative of 1/r and does not derive this expression:
//   1/mu_eff = 1/gamma_bra + 1/gamma_ket + 1/omega^2
inline double effective_screening(double gb, double gk, double omega) {
    if (omega <= 0.0) return 0.0;
    if (gb <= 1e-30 || gk <= 1e-30) return 0.0;
    double inv = 1.0 / gb + 1.0 / gk + 1.0 / (omega * omega);
    if (inv <= 1e-30) return 0.0;
    return 1.0 / inv;
}

// ---------------------------------------------------------------------------
// Per-thread accumulator
// ---------------------------------------------------------------------------

struct ThreadGradientAccum {
    // Interaction-tensor gradient cache: (rx, ry, rz, L, mu) → grad_T.
    std::unordered_map<GradCacheKey,
        std::vector<Eigen::MatrixXd>, GradCacheKeyHash> tensor_cache;
    // Per-atom force contribution (n_atoms, 3).
    Eigen::MatrixXd forces;
};

}  // namespace

// ---------------------------------------------------------------------------
// Main gradient entry point
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_bipolar_far_field_gradient_cpp(
    const BipoleMomentBufferCpp& moment_buffer,
    const std::vector<BipoleQuartetEntryCpp>& dispatch,
    const std::vector<DensityBlockCpp>& density_blocks,
    double ewald_omega,
    int n_atoms,
    const std::vector<std::vector<int>>& shell_to_atom,
    bool include_moment_derivative) {

    const int n_quartets = static_cast<int>(dispatch.size());
    const int n_sph = moment_buffer.n_sph;

    // Build an index: (s1, s2, cell) → entry index for fast lookup.
    // Also build shell-slice lookup.
    using MomentKey = std::tuple<int, int, int>;
    struct MomentKeyHash {
        std::size_t operator()(const MomentKey& k) const noexcept {
            return static_cast<std::size_t>(
                static_cast<std::uint64_t>(std::get<0>(k)) * 73856093ULL
                ^ static_cast<std::uint64_t>(std::get<1>(k)) * 19349663ULL
                ^ static_cast<std::uint64_t>(std::get<2>(k)) * 83492791ULL);
        }
    };
    std::unordered_map<MomentKey, int, MomentKeyHash> moment_index;
    moment_index.reserve(moment_buffer.entries.size());
    for (int i = 0; i < static_cast<int>(moment_buffer.entries.size()); ++i) {
        const auto& e = moment_buffer.entries[i];
        moment_index[{e.shell_index_1, e.shell_index_2, e.cell_index}] = i;
    }

    // Build density map: (ix, iy, iz) → Eigen::MatrixXd*.
    using CellTriple = std::tuple<int, int, int>;
    struct CellTripleHash {
        std::size_t operator()(const CellTriple& k) const noexcept {
            return static_cast<std::size_t>(
                static_cast<std::uint64_t>(std::get<0>(k) + 524288) * 73856093ULL
                ^ static_cast<std::uint64_t>(std::get<1>(k) + 524288) * 19349663ULL
                ^ static_cast<std::uint64_t>(std::get<2>(k) + 524288) * 83492791ULL);
        }
    };
    std::unordered_map<CellTriple, const Eigen::MatrixXd*, CellTripleHash> density_map;
    density_map.reserve(density_blocks.size());
    for (const auto& db : density_blocks) {
        density_map[{db.cell_ix, db.cell_iy, db.cell_iz}] = &db.density;
    }

    // Per-thread accumulators.
    std::vector<ThreadGradientAccum> thread_accums;
    #pragma omp parallel
    {
        #pragma omp single
        {
            thread_accums.resize(omp_get_num_threads());
        }
    }
    for (auto& t : thread_accums) {
        t.forces = Eigen::MatrixXd::Zero(n_atoms, 3);
        t.tensor_cache.reserve(256);
    }

    // -----------------------------------------------------------------------
    // Main quartet loop — OpenMP parallel
    // -----------------------------------------------------------------------
    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < n_quartets; ++q) {
        const int tid = omp_get_thread_num();
        auto& local = thread_accums[tid];

        const auto& dq = dispatch[q];
        const int L_order = dq.truncation_order;
        if (L_order <= 0) continue;

        // --- Look up moment entries ----------------------------------------
        MomentKey bra_key{dq.shell_1, dq.shell_2, dq.cell_bra};
        MomentKey ket_key{dq.shell_3, dq.shell_4, dq.cell_ket};
        auto it_bra = moment_index.find(bra_key);
        auto it_ket = moment_index.find(ket_key);
        if (it_bra == moment_index.end() || it_ket == moment_index.end()) continue;

        const auto& bra_entry = moment_buffer.entries[it_bra->second];
        const auto& ket_entry = moment_buffer.entries[it_ket->second];

        const int n1 = bra_entry.bf_count_1;
        const int n2 = bra_entry.bf_count_2;
        const int n3 = ket_entry.bf_count_1;
        const int n4 = ket_entry.bf_count_2;

        const int n_keep = (L_order + 1) * (L_order + 1);
        if (n_keep > n_sph) continue;

        // --- Separation vector ---------------------------------------------
        const double rx = ket_entry.centre_x - bra_entry.centre_x;
        const double ry = ket_entry.centre_y - bra_entry.centre_y;
        const double rz = ket_entry.centre_z - bra_entry.centre_z;
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (r2 < 1e-30) continue;

        // --- Effective screening -------------------------------------------
        const double mu_eff = effective_screening(
            dq.bra_width, dq.ket_width, ewald_omega);

        // --- Interaction tensor gradient -----------------------------------
        // Check per-thread cache.
        GradCacheKey cache_key{
            bin_coord(rx), bin_coord(ry), bin_coord(rz),
            L_order, bin_coord(mu_eff)};
        std::vector<Eigen::MatrixXd> grad_T;  // 3 matrices of (n_keep, n_keep)
        auto cache_it = local.tensor_cache.find(cache_key);
        if (cache_it != local.tensor_cache.end()) {
            grad_T = cache_it->second;
        } else {
            if (ewald_omega > 0.0 && mu_eff > 0.0) {
                grad_T = multipole_erfc_interaction_tensor_gradient(
                    L_order, L_order, rx, ry, rz, mu_eff);
            } else {
                grad_T = multipole_interaction_tensor_gradient(
                    L_order, L_order, rx, ry, rz);
            }
            local.tensor_cache[cache_key] = grad_T;
        }

        // --- Density lookup ------------------------------------------------
        CellTriple ket_cell_key{
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_ket][0] : 0,
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_ket][1] : 0,
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_ket][2] : 0};
        CellTriple bra_cell_key{
            dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_bra][0] : 0,
            dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_bra][1] : 0,
            dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_bra][2] : 0};

        auto d_ket_it = density_map.find(ket_cell_key);
        if (d_ket_it == density_map.end()) continue;
        const Eigen::MatrixXd& D_ket_full = *d_ket_it->second;

        auto d_bra_it = density_map.find(bra_cell_key);
        if (d_bra_it == density_map.end()) continue;
        const Eigen::MatrixXd& D_bra_full = *d_bra_it->second;

        // Extract density sub-blocks.
        const int b1 = bra_entry.bf_offset_1;
        const int b2 = bra_entry.bf_offset_2;
        const int b3 = ket_entry.bf_offset_1;
        const int b4 = ket_entry.bf_offset_2;

        if (b1 + n1 > D_bra_full.rows() || b2 + n2 > D_bra_full.cols()) continue;
        if (b3 + n3 > D_ket_full.rows() || b4 + n4 > D_ket_full.cols()) continue;

        Eigen::MatrixXd D_bra_sub = D_bra_full.block(b1, b2, n1, n2);
        Eigen::MatrixXd D_ket_sub = D_ket_full.block(b3, b4, n3, n4);

        // Flatten density row-major.
        Eigen::VectorXd D_bra_flat(n1 * n2);
        Eigen::VectorXd D_ket_flat(n3 * n4);
        for (int i = 0; i < n1; ++i)
            for (int j = 0; j < n2; ++j)
                D_bra_flat(i * n2 + j) = D_bra_sub(i, j);
        for (int i = 0; i < n3; ++i)
            for (int j = 0; j < n4; ++j)
                D_ket_flat(i * n4 + j) = D_ket_sub(i, j);

        // --- Moments -------------------------------------------------------
        // Slice to truncation order.
        Eigen::Map<const Eigen::MatrixXd> mom_bra_full(
            bra_entry.moments_flat.data(), n1 * n2, n_sph);
        Eigen::Map<const Eigen::MatrixXd> mom_ket_full(
            ket_entry.moments_flat.data(), n3 * n4, n_sph);

        // --- dT/dR contraction: f_R[a] = D_bra @ M_bra @ grad_T[a] @ M_ket^T @ D_ket
        Eigen::Vector3d f_R = Eigen::Vector3d::Zero();
        for (int a = 0; a < 3; ++a) {
            // T_ket = grad_T[a] @ M_ket^T  : (n_keep, n_keep) @ (n_keep, n3*n4)
            Eigen::MatrixXd T_ket = grad_T[a]
                * mom_ket_full.leftCols(n_keep).transpose();
            // E_pair = M_bra @ T_ket  : (n1*n2, n_keep) @ (n_keep, n3*n4)
            Eigen::MatrixXd E_pair = mom_bra_full.leftCols(n_keep) * T_ket;
            // f_R[a] = D_bra @ E_pair @ D_ket
            f_R(a) = D_bra_flat.dot(E_pair * D_ket_flat);
        }

        // --- Atom identification -------------------------------------------
        std::vector<int> bra_atoms, ket_atoms;
        if (!shell_to_atom.empty()) {
            for (int sh = 0; sh < static_cast<int>(shell_to_atom.size()); ++sh) {
                if (sh == dq.shell_1 || sh == dq.shell_2) {
                    for (int a : shell_to_atom[sh])
                        bra_atoms.push_back(a);
                }
                if (sh == dq.shell_3 || sh == dq.shell_4) {
                    for (int a : shell_to_atom[sh])
                        ket_atoms.push_back(a);
                }
            }
        }

        // Deduplicate.
        std::sort(bra_atoms.begin(), bra_atoms.end());
        bra_atoms.erase(std::unique(bra_atoms.begin(), bra_atoms.end()), bra_atoms.end());
        std::sort(ket_atoms.begin(), ket_atoms.end());
        ket_atoms.erase(std::unique(ket_atoms.begin(), ket_atoms.end()), ket_atoms.end());

        // --- Distribute dT/dR forces: bra gets -f_R, ket gets +f_R ---------
        if (!bra_atoms.empty()) {
            Eigen::Vector3d f_per = -f_R / static_cast<double>(bra_atoms.size());
            for (int a : bra_atoms) {
                if (a >= 0 && a < n_atoms)
                    local.forces.row(a) += f_per;
            }
        }
        if (!ket_atoms.empty()) {
            Eigen::Vector3d f_per = f_R / static_cast<double>(ket_atoms.size());
            for (int a : ket_atoms) {
                if (a >= 0 && a < n_atoms)
                    local.forces.row(a) += f_per;
            }
        }

        // --- dM/dA contribution (sub-dominant, O(1/R^{L+2})) ----------------
        // NOTE: moment_grads are NOT currently stored in BipoleMomentBufferCpp.
        // The dM/dA terms use the separate MomentDerivativeBufferCpp added
        // via bipole_moment_derivatives.hpp.  This section is a placeholder
        // and will be wired once the moment_grads are passed through.
        // For now, include_moment_derivative is a no-op at the C++ level;
        // the Python wrapper calls this function without dM/dA support
        // and then adds dM/dA in Python as before.
    }

    // -----------------------------------------------------------------------
    // Merge thread results
    // -----------------------------------------------------------------------
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_atoms, 3);
    for (const auto& t : thread_accums) {
        result += t.forces;
    }
    return result;
}

}  // namespace vibeqc
