// Pre-computed far-field Fock kernel builder (C++ impl).

#include "vibeqc/bipole_fock_kernel_builder.hpp"

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

using RKey = std::tuple<int, int, int, int>;  // (rx_bin, ry_bin, rz_bin, L)

struct RKeyHash {
    std::size_t operator()(const RKey& k) const noexcept {
        return static_cast<std::size_t>(
            (static_cast<std::uint64_t>(std::get<0>(k) + 524288) * 73856093ULL) ^
            (static_cast<std::uint64_t>(std::get<1>(k) + 524288) * 19349663ULL) ^
            (static_cast<std::uint64_t>(std::get<2>(k) + 524288) * 83492791ULL) ^
            static_cast<std::uint64_t>(std::get<3>(k)));
    }
};

int bin(double x) { return static_cast<int>(std::round(x * 100000.0)); }

double mu_effective(double ga, double gb, double omega) {
    if (omega <= 0.0) return 0.0;
    double inv = (ga > 1e-30 ? 1.0/ga : 0.0)
               + (gb > 1e-30 ? 1.0/gb : 0.0)
               + 1.0/(omega*omega);
    return (inv > 1e-30) ? 1.0/inv : 0.0;
}

// Moment key for buffer lookup.
using MKey = std::tuple<int, int, int>;
struct MKeyHash {
    std::size_t operator()(const MKey& k) const noexcept {
        return static_cast<std::size_t>(
            (static_cast<std::uint64_t>(std::get<0>(k)) << 40) ^
            (static_cast<std::uint64_t>(std::get<1>(k)) << 20) ^
            static_cast<std::uint64_t>(std::get<2>(k)));
    }
};

}  // namespace

std::vector<FarFieldFockKernelEntry> build_far_field_fock_kernel_cpp(
    const BipoleMomentBufferCpp& moment_buffer,
    const std::vector<BipoleQuartetEntryCpp>& dispatch,
    double ewald_omega) {

    // Build moment lookup: (s1, s2, cell_idx) → entry index.
    std::unordered_map<MKey, int, MKeyHash> moment_index;
    for (int i = 0; i < static_cast<int>(moment_buffer.entries.size()); ++i) {
        const auto& e = moment_buffer.entries[i];
        moment_index[MKey{e.shell_index_1, e.shell_index_2, e.cell_index}] = i;
    }

    // -------------------------------------------------------------------
    // Pass 1: group quartets by (rounded R_sep, L_order).
    // -------------------------------------------------------------------
    std::unordered_map<RKey, std::vector<int>, RKeyHash> groups;
    const int n_q = static_cast<int>(dispatch.size());

    for (int q = 0; q < n_q; ++q) {
        const auto& dq = dispatch[q];
        int L = dq.truncation_order;
        if (L <= 0) continue;

        // Look up bra and ket pair centres.
        MKey bk{dq.shell_1, dq.shell_2, dq.cell_bra};
        MKey kk{dq.shell_3, dq.shell_4, dq.cell_ket};
        auto it_b = moment_index.find(bk);
        auto it_k = moment_index.find(kk);
        if (it_b == moment_index.end() || it_k == moment_index.end()) continue;

        const auto& be = moment_buffer.entries[it_b->second];
        const auto& ke = moment_buffer.entries[it_k->second];
        double rx = ke.centre_x - be.centre_x;
        double ry = ke.centre_y - be.centre_y;
        double rz = ke.centre_z - be.centre_z;
        if (rx*rx + ry*ry + rz*rz < 1e-30) continue;

        groups[RKey{bin(rx), bin(ry), bin(rz), L}].push_back(q);
    }

    // -------------------------------------------------------------------
    // Pass 2: for each group, compute tensor + K matrices.
    // -------------------------------------------------------------------
    std::vector<FarFieldFockKernelEntry> result;

    for (const auto& [rkey, qlist] : groups) {
        int L_order = std::get<3>(rkey);
        int n_keep = multipole_n_components(L_order);

        // Use first quartet's centres for the tensor.
        int q0 = qlist[0];
        const auto& d0 = dispatch[q0];
        auto it_b0 = moment_index.find(MKey{d0.shell_1, d0.shell_2, d0.cell_bra});
        auto it_k0 = moment_index.find(MKey{d0.shell_3, d0.shell_4, d0.cell_ket});
        if (it_b0 == moment_index.end() || it_k0 == moment_index.end()) continue;

        const auto& be0 = moment_buffer.entries[it_b0->second];
        const auto& ke0 = moment_buffer.entries[it_k0->second];
        double rx = ke0.centre_x - be0.centre_x;
        double ry = ke0.centre_y - be0.centre_y;
        double rz = ke0.centre_z - be0.centre_z;

        // Compute interaction tensor.
        Eigen::MatrixXd T_mat;
        if (ewald_omega > 0.0) {
            double mu = mu_effective(d0.bra_width, d0.ket_width, ewald_omega);
            if (mu > 0.0)
                T_mat = multipole_erfc_interaction_tensor(L_order, L_order, rx, ry, rz, mu);
            else
                T_mat = multipole_interaction_tensor(L_order, L_order, rx, ry, rz);
        } else {
            T_mat = multipole_interaction_tensor(L_order, L_order, rx, ry, rz);
        }

        // Collect moment matrices for this group.
        std::vector<Eigen::MatrixXd> bra_moms, ket_moms;
        std::vector<FarFieldFockKernelEntry> meta;

        for (int q : qlist) {
            const auto& dq = dispatch[q];
            auto it_b = moment_index.find(MKey{dq.shell_1, dq.shell_2, dq.cell_bra});
            auto it_k = moment_index.find(MKey{dq.shell_3, dq.shell_4, dq.cell_ket});
            if (it_b == moment_index.end() || it_k == moment_index.end()) continue;

            const auto& be = moment_buffer.entries[it_b->second];
            const auto& ke = moment_buffer.entries[it_k->second];

            int n1 = be.bf_count_1, n2 = be.bf_count_2;
            int n3 = ke.bf_count_1, n4 = ke.bf_count_2;

            // Extract moment sub-blocks truncated to L_order.
            Eigen::Map<const Eigen::MatrixXd> mb_full(be.moments_flat.data(), n1*n2, moment_buffer.n_sph);
            Eigen::Map<const Eigen::MatrixXd> mk_full(ke.moments_flat.data(), n3*n4, moment_buffer.n_sph);
            Eigen::MatrixXd mb = mb_full.leftCols(n_keep);
            Eigen::MatrixXd mk = mk_full.leftCols(n_keep);

            bra_moms.push_back(mb);
            ket_moms.push_back(mk);

            // Cell keys.
            int bcix = 0, bciy = 0, bciz = 0;
            int kcix = 0, kciy = 0, kciz = 0;
            if (dq.cell_bra >= 0 && dq.cell_bra < static_cast<int>(moment_buffer.cell_indices.size())) {
                bcix = moment_buffer.cell_indices[dq.cell_bra][0];
                bciy = moment_buffer.cell_indices[dq.cell_bra][1];
                bciz = moment_buffer.cell_indices[dq.cell_bra][2];
            }
            if (dq.cell_ket >= 0 && dq.cell_ket < static_cast<int>(moment_buffer.cell_indices.size())) {
                kcix = moment_buffer.cell_indices[dq.cell_ket][0];
                kciy = moment_buffer.cell_indices[dq.cell_ket][1];
                kciz = moment_buffer.cell_indices[dq.cell_ket][2];
            }

            FarFieldFockKernelEntry entry;
            entry.bra_cell_ix = bcix;
            entry.bra_cell_iy = bciy;
            entry.bra_cell_iz = bciz;
            entry.bra_b1 = be.bf_offset_1;
            entry.bra_b1e = be.bf_offset_1 + n1;
            entry.bra_b2 = be.bf_offset_2;
            entry.bra_b2e = be.bf_offset_2 + n2;
            entry.ket_cell_ix = kcix;
            entry.ket_cell_iy = kciy;
            entry.ket_cell_iz = kciz;
            entry.ket_b3 = ke.bf_offset_1;
            entry.ket_b3e = ke.bf_offset_1 + n3;
            entry.ket_b4 = ke.bf_offset_2;
            entry.ket_b4e = ke.bf_offset_2 + n4;

            meta.push_back(entry);
        }

        // Batch matmul: K_i = M_bra_i @ T @ M_ket_i^T
        std::vector<Eigen::MatrixXd> K_mats = build_far_field_k_matrices_batch(
            bra_moms, ket_moms, T_mat);

        for (std::size_t i = 0; i < K_mats.size(); ++i) {
            meta[i].K_matrix = K_mats[i];
            result.push_back(std::move(meta[i]));
        }
    }

    return result;
}

}  // namespace vibeqc
