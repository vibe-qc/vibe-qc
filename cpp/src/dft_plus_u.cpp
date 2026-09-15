// DFT+U (Dudarev rotationally-invariant) — C++ kernel.
//
// See cpp/include/vibeqc/dft_plus_u.hpp for the math + spin convention.

#include "vibeqc/dft_plus_u.hpp"

#include <stdexcept>

namespace vibeqc {

DftPlusUResult compute_dft_plus_u(
    const std::vector<HubbardSiteCxx>& sites,
    const std::vector<std::vector<int>>& ao_groups,
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& S) {

    const Eigen::Index nbf = P.rows();
    DftPlusUResult out;
    out.V = Eigen::MatrixXd::Zero(nbf, nbf);
    out.energy = 0.0;

    if (sites.empty()) {
        return out;
    }

    if (sites.size() != ao_groups.size()) {
        throw std::invalid_argument(
            "dft_plus_u: sites and ao_groups must be parallel arrays of "
            "equal length");
    }
    if (P.rows() != P.cols() || S.rows() != S.cols()
        || P.rows() != S.rows()) {
        throw std::invalid_argument(
            "dft_plus_u: P and S must be square nbf x nbf matrices of "
            "equal shape");
    }

    // Pre-compute (S P S) once and slice per-site. For typical SCF use
    // the cost is two GEMMs on nbf x nbf and a handful of small block
    // ops on (2l+1) x (2l+1) per site, so this is cheap relative to
    // the surrounding Fock build.
    const Eigen::MatrixXd SPS = S * P * S;

    for (std::size_t s = 0; s < sites.size(); ++s) {
        const auto& site = sites[s];
        const auto& idx = ao_groups[s];
        if (idx.empty()) {
            continue;
        }
        const Eigen::Index k = static_cast<Eigen::Index>(idx.size());

        // n^A_l = (S P S)_block — gather rows then columns.
        Eigen::MatrixXd n(k, k);
        for (Eigen::Index a = 0; a < k; ++a) {
            for (Eigen::Index b = 0; b < k; ++b) {
                n(a, b) = SPS(idx[static_cast<std::size_t>(a)],
                              idx[static_cast<std::size_t>(b)]);
            }
        }

        // Energy contribution: (U_eff / 2) · (tr n − tr n²).
        const double tr_n = n.trace();
        const double tr_n_sq = (n * n).trace();
        out.energy += 0.5 * site.U_eff_au * (tr_n - tr_n_sq);

        // V_U^A_{mm'} = U_eff · (δ_{mm'} / 2 − n_{mm'}); scatter back
        // into the (A,l) block of the AO Fock contribution.
        Eigen::MatrixXd V_local = -site.U_eff_au * n;
        for (Eigen::Index a = 0; a < k; ++a) {
            V_local(a, a) += 0.5 * site.U_eff_au;
        }
        for (Eigen::Index a = 0; a < k; ++a) {
            for (Eigen::Index b = 0; b < k; ++b) {
                out.V(idx[static_cast<std::size_t>(a)],
                      idx[static_cast<std::size_t>(b)]) += V_local(a, b);
            }
        }
    }

    // Symmetrize defensively. (S P S) is symmetric in real basis and
    // V_local is built from a symmetric n, so this is a no-op modulo
    // floating-point rounding — but it costs nothing and protects
    // downstream code that assumes symmetric F.
    Eigen::MatrixXd V_AO = 0.5 * (out.V + out.V.transpose());

    // Variational form for non-orthogonal AO basis. The standard
    // Dudarev convention V_U^A_{mm'} = U_eff (½ δ - n^A_l_{mm'}) is
    // the functional derivative ∂E_U/∂n. To make the SCF variational
    // w.r.t. E_total = E_HF + E_U with n = (S P S)_(A,l), the Fock
    // contribution must be ∂E_U/∂P = S V_AO S (the chain rule
    // through n = SPS gives this S-sandwich). For orthonormal-
    // projector bases (plane-wave PAW) S = I and the two forms
    // coincide — that's why most published codes show V_U
    // unsandwiched. For Gaussian AO bases, the sandwich is
    // load-bearing: without it, the SCF converges to a non-variational
    // stationary point and the analytic Pulay gradient mismatches FD
    // by O(10 mHa/bohr) (see docs/user_guide/dft_plus_u.md; the original
    // debug note is retired in git history).
    out.V = S * V_AO * S;
    return out;
}

DftPlusUMultiK compute_dft_plus_u_multi_k_per_spin(
    const std::vector<HubbardSiteCxx>& sites,
    const std::vector<std::vector<int>>& ao_groups,
    const std::vector<Eigen::MatrixXcd>& S_k,
    const std::vector<Eigen::MatrixXcd>& P_sigma_k,
    const std::vector<double>& weights) {

    if (S_k.size() != P_sigma_k.size()
        || P_sigma_k.size() != weights.size()) {
        throw std::invalid_argument(
            "compute_dft_plus_u_multi_k_per_spin: S_k / P_σ_k / "
            "weights must have equal length");
    }
    DftPlusUMultiK out;
    const Eigen::Index nbf = S_k.empty() ? 0 : S_k[0].rows();
    out.V_AO_per_spin = Eigen::MatrixXd::Zero(nbf, nbf);
    out.energy_total = 0.0;
    if (sites.empty() || S_k.empty()) {
        return out;
    }
    if (sites.size() != ao_groups.size()) {
        throw std::invalid_argument(
            "compute_dft_plus_u_multi_k_per_spin: sites and "
            "ao_groups must be parallel arrays of equal length");
    }

    // Per-site computation. The (A,l) blocks are disjoint (each AO
    // belongs to at most one (A,l) channel), so we can accumulate
    // independently.
    for (std::size_t s = 0; s < sites.size(); ++s) {
        const auto& site = sites[s];
        const auto& idx = ao_groups[s];
        if (idx.empty()) {
            continue;
        }
        const Eigen::Index k = static_cast<Eigen::Index>(idx.size());

        // Per-spin AO occupation matrix on the (A,l) block:
        //   n_σ^A_l = Σ_k w_k Re[(S(k) P_σ(k) S(k))_{(A,l),mm'}]
        // (P_σ here is the per-spin density; no 1/2 halving inside.)
        Eigen::MatrixXd n_block = Eigen::MatrixXd::Zero(k, k);
        for (std::size_t ik = 0; ik < S_k.size(); ++ik) {
            const Eigen::MatrixXcd& Sk = S_k[ik];
            const Eigen::MatrixXcd& Pk = P_sigma_k[ik];
            const Eigen::MatrixXcd SPS_k = Sk * Pk * Sk;
            for (Eigen::Index a = 0; a < k; ++a) {
                for (Eigen::Index b = 0; b < k; ++b) {
                    n_block(a, b) += weights[ik]
                        * SPS_k(idx[static_cast<std::size_t>(a)],
                                idx[static_cast<std::size_t>(b)]).real();
                }
            }
        }
        // Force exact Hermiticity (the imaginary-part trim above is
        // physical, this guards against round-off asymmetry).
        n_block = 0.5 * (n_block + n_block.transpose().eval());

        // Per-spin Dudarev energy: (U_eff/2)(tr n_σ − tr n_σ²).
        const double tr_n = n_block.trace();
        const double tr_n_sq = (n_block * n_block).trace();
        out.energy_total += 0.5 * site.U_eff_au * (tr_n - tr_n_sq);

        // Per-spin V_AO on the (A,l) block: U_eff (½ δ − n_σ).
        Eigen::MatrixXd V_local = -site.U_eff_au * n_block;
        for (Eigen::Index a = 0; a < k; ++a) {
            V_local(a, a) += 0.5 * site.U_eff_au;
        }
        // Scatter into the full AO matrix.
        for (Eigen::Index a = 0; a < k; ++a) {
            for (Eigen::Index b = 0; b < k; ++b) {
                out.V_AO_per_spin(idx[static_cast<std::size_t>(a)],
                                   idx[static_cast<std::size_t>(b)])
                    += V_local(a, b);
            }
        }
    }
    out.V_AO_per_spin = 0.5 * (out.V_AO_per_spin
                                + out.V_AO_per_spin.transpose());
    return out;
}

DftPlusUMultiK compute_dft_plus_u_multi_k_closed_shell(
    const std::vector<HubbardSiteCxx>& sites,
    const std::vector<std::vector<int>>& ao_groups,
    const std::vector<Eigen::MatrixXcd>& S_k,
    const std::vector<Eigen::MatrixXcd>& P_k,
    const std::vector<double>& weights) {

    // Closed shell: per-spin density is half the total. Halve P(k)
    // and call the per-spin kernel; double the energy for the
    // spin sum.
    std::vector<Eigen::MatrixXcd> P_sigma_k;
    P_sigma_k.reserve(P_k.size());
    for (const auto& Pk : P_k) {
        P_sigma_k.emplace_back(0.5 * Pk);
    }
    auto out = compute_dft_plus_u_multi_k_per_spin(
        sites, ao_groups, S_k, P_sigma_k, weights);
    out.energy_total *= 2.0;
    return out;
}

}  // namespace vibeqc
