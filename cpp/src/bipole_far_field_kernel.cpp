// Dormant prototype far-field Fock-kernel apply.
//
// The pre-computed K_matrix and per-iteration density contraction are
// implementation cache choices, not an algorithm derived in Saunders (1992).
//
// OpenMP-parallelised.

#include "vibeqc/bipole_far_field_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <omp.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vibeqc {

namespace {

using CellTuple = std::tuple<int, int, int>;

struct CellTupleHash {
    std::size_t operator()(const CellTuple& k) const noexcept {
        const std::uint64_t a = static_cast<std::uint64_t>(std::get<0>(k) + 1024);
        const std::uint64_t b = static_cast<std::uint64_t>(std::get<1>(k) + 1024);
        const std::uint64_t c = static_cast<std::uint64_t>(std::get<2>(k) + 1024);
        return static_cast<std::size_t>(
            (a * 2654435761ULL) ^ (b * 40503ULL) ^ c);
    }
};

struct ThreadLocalFock {
    std::unordered_map<CellTuple, Eigen::MatrixXd, CellTupleHash> blocks;
    double coulomb_energy = 0.0;
    int quartets = 0;
};

}  // namespace

FarFieldFockResult apply_far_field_fock_kernel(
    const std::vector<FarFieldFockKernelEntry>& entries,
    const std::vector<int>& density_cell_keys_flat,
    const std::vector<Eigen::MatrixXd>& density_blocks,
    int nbf) {

    // Build density lookup: cell tuple → density matrix pointer.
    std::map<CellTuple, const Eigen::MatrixXd*> density_map;
    const int n_density = static_cast<int>(density_blocks.size());
    for (int i = 0; i < n_density; ++i) {
        int ix = density_cell_keys_flat[3 * i];
        int iy = density_cell_keys_flat[3 * i + 1];
        int iz = density_cell_keys_flat[3 * i + 2];
        density_map[CellTuple{ix, iy, iz}] = &density_blocks[i];
    }

    const int n_threads = omp_get_max_threads();
    std::vector<ThreadLocalFock> thread_results(n_threads);

    const int n_entries = static_cast<int>(entries.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_entries; ++i) {
        const int tid = omp_get_thread_num();
        auto& local = thread_results[tid];
        const auto& e = entries[i];

        // Look up ket density.
        CellTuple ket_key{e.ket_cell_ix, e.ket_cell_iy, e.ket_cell_iz};
        auto it = density_map.find(ket_key);
        if (it == density_map.end()) continue;

        const Eigen::MatrixXd& D_ket = *it->second;

        int n3 = e.ket_b3e - e.ket_b3;
        int n4 = e.ket_b4e - e.ket_b4;
        int n1 = e.bra_b1e - e.bra_b1;
        int n2 = e.bra_b2e - e.bra_b2;
        if (n1 <= 0 || n2 <= 0 || n3 <= 0 || n4 <= 0) continue;

        // Extract ket density sub-block and flatten.
        Eigen::MatrixXd D_sub = D_ket.block(e.ket_b3, e.ket_b4, n3, n4);
        Eigen::Map<const Eigen::VectorXd> D_flat(
            D_sub.data(), static_cast<Eigen::Index>(n3 * n4));

        // Contract: dF = K_matrix @ D_flat
        Eigen::VectorXd dF_flat = e.K_matrix * D_flat;

        // Reshape and accumulate into thread-local Fock block.
        Eigen::Map<Eigen::MatrixXd> dF_bra(
            dF_flat.data(),
            static_cast<Eigen::Index>(n1),
            static_cast<Eigen::Index>(n2));

        CellTuple bra_key{e.bra_cell_ix, e.bra_cell_iy, e.bra_cell_iz};
        auto& F_block = local.blocks[bra_key];
        if (F_block.size() == 0) {
            F_block = Eigen::MatrixXd::Zero(nbf, nbf);
        }
        F_block.block(e.bra_b1, e.bra_b2, n1, n2) += dF_bra;

        // Accumulate Coulomb far-field energy:
        // E = 0.5 * sum_ij D_bra[i,j] * dF_bra[i,j]
        // (Frobenius inner product; D_bra is symmetric).
        auto bra_it = density_map.find(bra_key);
        if (bra_it != density_map.end()) {
            const Eigen::MatrixXd& D_bra = *bra_it->second;
            Eigen::MatrixXd D_bra_sub = D_bra.block(
                e.bra_b1, e.bra_b2, n1, n2);
            local.coulomb_energy +=
                0.5 * D_bra_sub.cwiseProduct(dF_bra).sum();
        }

        local.quartets++;
    }

    // Merge thread-local results.
    std::unordered_map<CellTuple, Eigen::MatrixXd, CellTupleHash> merged;
    double total_coulomb = 0.0;
    int total_quartets = 0;

    for (const auto& local : thread_results) {
        total_coulomb += local.coulomb_energy;
        total_quartets += local.quartets;
        for (const auto& [key, block] : local.blocks) {
            auto& target = merged[key];
            if (target.size() == 0) {
                target = block;
            } else {
                target += block;
            }
        }
    }

    // Convert to output struct.
    FarFieldFockResult result;
    result.fock_matrices.reserve(merged.size());
    result.cell_ix.reserve(merged.size());
    result.cell_iy.reserve(merged.size());
    result.cell_iz.reserve(merged.size());

    for (auto& [key, block] : merged) {
        result.cell_ix.push_back(std::get<0>(key));
        result.cell_iy.push_back(std::get<1>(key));
        result.cell_iz.push_back(std::get<2>(key));
        result.fock_matrices.push_back(std::move(block));
    }
    result.coulomb_energy_far = total_coulomb;
    result.n_quartets_applied = total_quartets;

    return result;
}

// ============================================================================
// Batched K-matrix builder
// ============================================================================

std::vector<Eigen::MatrixXd> build_far_field_k_matrices_batch(
    const std::vector<Eigen::MatrixXd>& bra_moments_2d,
    const std::vector<Eigen::MatrixXd>& ket_moments_2d,
    const Eigen::MatrixXd& T_matrix) {

    const int n_quartets = static_cast<int>(bra_moments_2d.size());
    std::vector<Eigen::MatrixXd> K_matrices(n_quartets);

    if (n_quartets != static_cast<int>(ket_moments_2d.size())) {
        throw std::runtime_error(
            "build_far_field_k_matrices_batch: bra and ket moment lists "
            "must have the same length");
    }

    // Pre-compute T @ ket_moments^T for all quartets? No — each ket
    // moment matrix is different, so we compute per quartet.
    //
    // K_i = bra_i @ T @ ket_i^T
    // Efficient path: (bra_i @ T) @ ket_i^T
    // — bra_i @ T is (n_bra_ao × n_sph) @ (n_sph × n_sph) = (n_bra_ao × n_sph)
    // — result @ ket_i^T is (n_bra_ao × n_sph) @ (n_sph × n_ket_ao)
    //   = (n_bra_ao × n_ket_ao)

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_quartets; ++i) {
        // Compute intermediate: M_temp = bra_i @ T
        Eigen::MatrixXd M_temp = bra_moments_2d[i] * T_matrix;
        // Compute K_i = M_temp @ ket_i^T
        K_matrices[i] = M_temp * ket_moments_2d[i].transpose();
    }

    return K_matrices;
}

}  // namespace vibeqc
