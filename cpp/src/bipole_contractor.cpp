// Direct contractor for the dormant quartet-level bipolar prototype.
// Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the expansion source; this
// contractor and its screening composition are implementation-specific.

#include "vibeqc/bipole_contractor.hpp"
#include "vibeqc/bipole_multipole.hpp"

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

// Key for per-thread interaction-tensor cache:
// (rx_bin, ry_bin, rz_bin, L, mu_bin).
// Separations are binned at 1e-5 bohr to merge nearly-identical vectors.
// mu_bin is the binned effective screening parameter (zero for bare Coulomb).
using TensorCacheKey = std::tuple<int, int, int, int, int>;

struct TensorCacheKeyHash {
    std::size_t operator()(const TensorCacheKey& k) const noexcept {
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

// Round a double to integer bins of width 1e-5 for tensor caching.
inline int bin_coordinate(double x) {
    return static_cast<int>(std::round(x * 100000.0));
}

// Standard Gaussian overlap-decay parameter: gamma = a1*a2/(a1+a2).
inline double product_distribution_width(double a1, double a2) {
    if (a1 <= 0.0 || a2 <= 0.0) return 0.0;
    return (a1 * a2) / (a1 + a2);
}

// Prototype quartet screening composition; Saunders (1992), Eq. (91), is
// instead a radial derivative of 1/r and does not derive this expression:
//   1/mu_eff = 1/gamma_bra + 1/gamma_ket + 1/omega^2
inline double quartet_effective_screening_cpp(
    double gamma_bra, double gamma_ket, double omega) {
    if (omega <= 0.0) return 0.0;
    const double inv_gamma_bra = (gamma_bra > 1e-30) ? 1.0 / gamma_bra : 0.0;
    const double inv_gamma_ket = (gamma_ket > 1e-30) ? 1.0 / gamma_ket : 0.0;
    const double inv_omega2 = 1.0 / (omega * omega);
    const double inv_mu_eff = inv_gamma_bra + inv_gamma_ket + inv_omega2;
    if (inv_mu_eff <= 1e-30) return 0.0;
    return 1.0 / inv_mu_eff;
}

// Cell tuple for density-map lookup.
using CellTuple = std::tuple<int, int, int>;

struct CellTupleHash {
    std::size_t operator()(const CellTuple& k) const noexcept {
        const std::uint64_t a = static_cast<std::uint64_t>(std::get<0>(k) + 1024);
        const std::uint64_t b = static_cast<std::uint64_t>(std::get<1>(k) + 1024);
        const std::uint64_t c = static_cast<std::uint64_t>(std::get<2>(k) + 1024);
        return static_cast<std::size_t>((a * 2654435761ULL) ^ (b * 40503ULL) ^ c);
    }
};

// Key for moment-buffer lookup: (shell_1, shell_2, cell_index).
using MomentKey = std::tuple<int, int, int>;

struct MomentKeyHash {
    std::size_t operator()(const MomentKey& k) const noexcept {
        return static_cast<std::size_t>(
            (static_cast<std::uint64_t>(std::get<0>(k)) << 40) ^
            (static_cast<std::uint64_t>(std::get<1>(k)) << 20) ^
            static_cast<std::uint64_t>(std::get<2>(k)));
    }
};

// Thread-local Fock accumulator.
struct ThreadLocalFockContract {
    std::unordered_map<CellTuple, Eigen::MatrixXd, CellTupleHash> fock_blocks;
    double coulomb_energy = 0.0;
    int quartets = 0;
    // Per-thread tensor cache.
    std::unordered_map<TensorCacheKey, Eigen::MatrixXd, TensorCacheKeyHash> tensor_cache;
};

}  // namespace

BipoleFarFieldResultCpp compute_bipolar_coulomb_far_field_cpp(
    const BipoleMomentBufferCpp& moment_buffer,
    const std::vector<BipoleQuartetEntryCpp>& dispatch,
    const std::vector<DensityBlockCpp>& density_blocks,
    double ewald_omega,
    int nbf_total) {

    // -----------------------------------------------------------------------
    // Build density lookup: cell tuple → density matrix pointer.
    // -----------------------------------------------------------------------
    std::map<CellTuple, const Eigen::MatrixXd*> density_map;
    for (const auto& db : density_blocks) {
        density_map[CellTuple{db.cell_ix, db.cell_iy, db.cell_iz}] = &db.density;
    }

    // -----------------------------------------------------------------------
    // Build moment buffer lookup: (s1, s2, cell_idx) → entry index.
    // -----------------------------------------------------------------------
    std::unordered_map<MomentKey, int, MomentKeyHash> moment_index;
    for (int i = 0; i < static_cast<int>(moment_buffer.entries.size()); ++i) {
        const auto& e = moment_buffer.entries[i];
        moment_index[MomentKey{e.shell_index_1, e.shell_index_2, e.cell_index}] = i;
    }

    // -----------------------------------------------------------------------
    // Thread-local storage.
    // -----------------------------------------------------------------------
    const int n_threads = omp_get_max_threads();
    std::vector<ThreadLocalFockContract> thread_results(n_threads);

    // -----------------------------------------------------------------------
    // Per-quartet contraction loop — OpenMP-parallel.
    // -----------------------------------------------------------------------
    const int n_quartets = static_cast<int>(dispatch.size());

    #pragma omp parallel for schedule(dynamic, 64)
    for (int q = 0; q < n_quartets; ++q) {
        const int tid = omp_get_thread_num();
        auto& local = thread_results[tid];

        const auto& dq = dispatch[q];
        const int L_order = dq.truncation_order;
        if (L_order <= 0) continue;

        // --- Retrieve bra and ket spherical moments ------------------------
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

        const int n_sph = moment_buffer.n_sph;
        const int n_keep = multipole_n_components(L_order);  // (L_order+1)^2
        if (n_keep > n_sph) continue;

        // Slice moments to truncation order: first n_keep columns.
        // The moments_flat data has already been converted from numpy row-major
        // to Eigen column-major by pybind11, so the default Map is correct.
        Eigen::Map<const Eigen::MatrixXd> mom_bra_full(
            bra_entry.moments_flat.data(), n1 * n2, n_sph);
        Eigen::Map<const Eigen::MatrixXd> mom_ket_full(
            ket_entry.moments_flat.data(), n3 * n4, n_sph);

        // --- Interaction tensor --------------------------------------------
        const double rx = ket_entry.centre_x - bra_entry.centre_x;
        const double ry = ket_entry.centre_y - bra_entry.centre_y;
        const double rz = ket_entry.centre_z - bra_entry.centre_z;
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (r2 < 1e-30) continue;  // overlapping centres → multipole diverges

        // Check per-thread cache.  Include L_order (not mu_eff) in the
        // key when ewald_omega==0 (bare Coulomb: tensor depends only on
        // R and L).  For erfc-screened paths, include the bin of mu_eff
        // because the effective screening parameter varies per-quartet
        // (depends on bra/ket product widths).
        double mu_eff = 0.0;
        if (ewald_omega > 0.0) {
            const double gamma_bra = dq.bra_width;
            const double gamma_ket = dq.ket_width;
            mu_eff = quartet_effective_screening_cpp(
                gamma_bra, gamma_ket, ewald_omega);
        }
        TensorCacheKey cache_key{
            bin_coordinate(rx), bin_coordinate(ry), bin_coordinate(rz),
            L_order, bin_coordinate(mu_eff)};
        Eigen::MatrixXd T_mat;
        auto cache_it = local.tensor_cache.find(cache_key);
        if (cache_it != local.tensor_cache.end()) {
            T_mat = cache_it->second;
        } else {
            if (ewald_omega > 0.0 && mu_eff > 0.0) {
                T_mat = multipole_erfc_interaction_tensor(
                    L_order, L_order, rx, ry, rz, mu_eff);
            } else {
                T_mat = multipole_interaction_tensor(
                    L_order, L_order, rx, ry, rz);
            }
            local.tensor_cache[cache_key] = T_mat;
        }

        // --- Density lookup ------------------------------------------------
        CellTuple ket_cell{dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
            ? moment_buffer.cell_indices[dq.cell_ket][0]
            : 0,
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
            ? moment_buffer.cell_indices[dq.cell_ket][1]
            : 0,
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
            ? moment_buffer.cell_indices[dq.cell_ket][2]
            : 0};

        // The ket cell in the dispatch uses the buffer's cell list; map to
        // density cell keys.  Actually, the dispatch stores cell indices into
        // the buffer's cell list, which may differ from the density cell keys.
        // We use the moment buffer cell positions to match density cells.
        // Simplification: look up density by the ket entry's centre-based key
        // which is the cell_index into the buffer's cell list.
        // The density blocks are indexed by their own cell (ix, iy, iz).
        // We need to map buffer cell index → cell (ix, iy, iz).
        CellTuple ket_density_key{
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_ket][0]
                : 0,
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_ket][1]
                : 0,
            dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_ket][2]
                : 0};

        CellTuple bra_density_key{
            dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_bra][0]
                : 0,
            dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_bra][1]
                : 0,
            dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())
                ? moment_buffer.cell_indices[dq.cell_bra][2]
                : 0};

        auto d_it = density_map.find(ket_density_key);
        if (d_it == density_map.end()) continue;
        const Eigen::MatrixXd& D_ket_full = *d_it->second;

        // Extract ket sub-block and flatten row-major (matching numpy).
        const int b3 = ket_entry.bf_offset_1;
        const int b4 = ket_entry.bf_offset_2;
        Eigen::MatrixXd D_sub = D_ket_full.block(b3, b4, n3, n4);
        // Check for negligible density.
        if (D_sub.squaredNorm() < 1e-60) continue;

        // Flatten row-major: D_flat[i*n4 + j] = D_sub(i,j)
        Eigen::VectorXd D_flat_row(n3 * n4);
        for (int i = 0; i < n3; ++i)
            for (int j = 0; j < n4; ++j)
                D_flat_row(i * n4 + j) = D_sub(i, j);

        // --- Contraction: F_bra += M_bra @ T @ M_ket^T @ D_flat ------------
        // Step 1: T_ket = T @ M_ket^T  (n_keep × n_keep) @ (n_keep × n3*n4)
        //   → (n_keep × n3*n4)
        Eigen::MatrixXd T_ket = T_mat * mom_ket_full.leftCols(n_keep).transpose();

        // Step 2: M_T_ket = M_bra @ T_ket  (n1*n2 × n_keep) @ (n_keep × n3*n4)
        //   → (n1*n2 × n3*n4)
        Eigen::MatrixXd E_pair = mom_bra_full.leftCols(n_keep) * T_ket;

        // Step 3: dF = E_pair @ D_flat  (n1*n2 × n3*n4) @ (n3*n4) → (n1*n2)
        Eigen::VectorXd dF_flat_vec = E_pair * D_flat_row;

        // Reshape and accumulate into thread-local Fock block.
        // Use RowMajor to match numpy's default reshape (row-major).
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>> dF_bra(
            dF_flat_vec.data(), n1, n2);

        // Bra Fock block.
        const int b1 = bra_entry.bf_offset_1;
        const int b2 = bra_entry.bf_offset_2;

        auto& F_block = local.fock_blocks[bra_density_key];
        if (F_block.size() == 0) {
            F_block = Eigen::MatrixXd::Zero(nbf_total, nbf_total);
        }
        F_block.block(b1, b2, n1, n2) += dF_bra;

        // --- Coulomb energy: 0.5 * Tr(D_bra_sub @ dF_bra) -----------------
        // Note: skipped if bra density not found (cell not in density map).
        auto d_bra_it = density_map.find(bra_density_key);
        if (d_bra_it != density_map.end()) {
            const Eigen::MatrixXd& D_bra_full = *d_bra_it->second;
            if (b1 + n1 <= D_bra_full.rows() && b2 + n2 <= D_bra_full.cols()) {
                Eigen::MatrixXd D_bra_sub = D_bra_full.block(b1, b2, n1, n2);
                // Element-wise product sum: sum(D_bra_sub .* dF_bra)
                double contrib = 0.0;
                for (int i = 0; i < n1; ++i) {
                    for (int j = 0; j < n2; ++j) {
                        contrib += D_bra_sub(i, j) * dF_bra(i, j);
                    }
                }
                local.coulomb_energy += 0.5 * contrib;
            }
        }

        local.quartets++;
    }

    // -----------------------------------------------------------------------
    // Merge thread-local results.
    // -----------------------------------------------------------------------
    std::unordered_map<CellTuple, Eigen::MatrixXd, CellTupleHash> merged_fock;
    double total_energy = 0.0;
    int total_quartets = 0;

    for (const auto& local : thread_results) {
        total_energy += local.coulomb_energy;
        total_quartets += local.quartets;
        for (const auto& [key, block] : local.fock_blocks) {
            auto& target = merged_fock[key];
            if (target.size() == 0) {
                target = block;
            } else {
                target += block;
            }
        }
    }

    // Convert to output struct.
    BipoleFarFieldResultCpp result;
    result.fock_blocks.reserve(merged_fock.size());
    result.fock_cell_ix.reserve(merged_fock.size());
    result.fock_cell_iy.reserve(merged_fock.size());
    result.fock_cell_iz.reserve(merged_fock.size());

    for (auto& [key, block] : merged_fock) {
        result.fock_cell_ix.push_back(std::get<0>(key));
        result.fock_cell_iy.push_back(std::get<1>(key));
        result.fock_cell_iz.push_back(std::get<2>(key));
        result.fock_blocks.push_back(std::move(block));
    }
    result.coulomb_energy_far = total_energy;
    result.quartets_processed = total_quartets;

    return result;
}

}  // namespace vibeqc
