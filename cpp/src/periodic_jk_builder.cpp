#include "vibeqc/periodic_jk_builder.hpp"

#include "vibeqc/periodic_fock.hpp"

namespace vibeqc {

namespace {

class PeriodicGammaJKBuilder final : public JKBuilder {
public:
    PeriodicGammaJKBuilder(const BasisSet& basis,
                           const PeriodicSystem& system,
                           const LatticeSumOptions& opts,
                           double omega)
        : basis_(&basis),
          system_(&system),
          opts_(opts),
          omega_(omega) {}

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        // The lattice-sum kernel computes J and K simultaneously; for
        // build_J / build_K alone we discard the other half. RHF runs
        // through build_g_rhf below (one call, both halves used) so
        // the wasteful path is only hit for UHF / RKS / UKS — those
        // still get a correct result, just with an O(2x) cost on the
        // K piece. A future fused build with caching can land here
        // without changing the SCF drivers.
        return build_jk_gamma_molecular_limit(*basis_, *system_, opts_,
                                              D, omega_).J;
    }

    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        return build_jk_gamma_molecular_limit(*basis_, *system_, opts_,
                                              D, omega_).K;
    }

    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                double alpha_hf) const override {
        // Single lattice-sum loop covers both J and K — the RHF
        // closed-shell case computes G = J − ½·α·K from the same
        // shell-quartet pass.
        const auto jk = build_jk_gamma_molecular_limit(
            *basis_, *system_, opts_, D, omega_);
        Eigen::MatrixXd G = jk.J;
        if (alpha_hf != 0.0) {
            G.noalias() -= 0.5 * alpha_hf * jk.K;
        }
        return G;
    }

private:
    const BasisSet* basis_;
    const PeriodicSystem* system_;
    LatticeSumOptions opts_;
    std::string method_;
    double omega_;
};

}  // namespace

std::unique_ptr<JKBuilder> make_periodic_gamma_jk_builder(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    double omega) {
    return std::make_unique<PeriodicGammaJKBuilder>(basis, system, opts, omega);
}

// ---- CCM-weighted periodic-Γ JKBuilder -----------------------------------

namespace {

class CCMWeightedGammaJKBuilder final : public JKBuilder {
public:
    CCMWeightedGammaJKBuilder(const BasisSet& basis,
                             const PeriodicSystem& system,
                             const LatticeSumOptions& opts,
                             std::vector<Eigen::Vector3i> weight_cells,
                             std::vector<Eigen::MatrixXd> weight_matrices,
                             const std::string& method,
                             double omega)
        : basis_(&basis),
          system_(&system),
          opts_(opts),
          weight_cells_(std::move(weight_cells)),
          weight_matrices_(std::move(weight_matrices)),
          method_(method),
          omega_(omega) {}

    // The four-center sigma index sits at c_p + gd (both WSSC cells), so the
    // integral image range must reach the pairwise-sum (±2t) cells — twice the
    // first-shell radius that opts_.cutoff_bohr encodes. Enumerate a sphere wide
    // enough to contain every c_p + gd; cells with zero WSSC weight are skipped
    // in the kernel, so the extra cells cost nothing.
    std::vector<LatticeCell> integral_cells() const {
        const auto shell1 = direct_lattice_cells(*system_, opts_.cutoff_bohr);
        double max_norm = 0.0;
        for (const auto& c : shell1) max_norm = std::max(max_norm, c.r_cart.norm());
        const double cutoff = std::max(opts_.cutoff_bohr, 2.05 * max_norm);
        return direct_lattice_cells(*system_, cutoff);
    }

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        const auto cells = integral_cells();
        return build_jk_ccm_weighted(*basis_, *system_,
            cells, opts_, D,
            weight_cells_, weight_matrices_, method_, omega_).J;
    }

    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        const auto cells = integral_cells();
        return build_jk_ccm_weighted(*basis_, *system_,
            cells, opts_, D,
            weight_cells_, weight_matrices_, method_, omega_).K;
    }

    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                double alpha_hf) const override {
        const auto cells = integral_cells();
        const auto jk = build_jk_ccm_weighted(*basis_, *system_,
            cells, opts_, D,
            weight_cells_, weight_matrices_, method_, omega_);
        // J and K are already Hermitised inside build_jk_ccm_weighted.
        Eigen::MatrixXd G = jk.J;
        if (alpha_hf != 0.0) {
            G.noalias() -= 0.5 * alpha_hf * jk.K;
        }
        return G;
    }

private:
    const BasisSet* basis_;
    const PeriodicSystem* system_;
    LatticeSumOptions opts_;
    std::vector<Eigen::Vector3i> weight_cells_;
    std::vector<Eigen::MatrixXd> weight_matrices_;
    std::string method_;
    double omega_;
};

}  // namespace

std::unique_ptr<JKBuilder> make_periodic_gamma_ccm_jk_builder(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<Eigen::Vector3i>& weight_cells,
    const std::vector<Eigen::MatrixXd>& weight_matrices,
    const std::string& method,
    double omega) {
    return std::make_unique<CCMWeightedGammaJKBuilder>(
        basis, system, opts, weight_cells, weight_matrices, method, omega);
}

}  // namespace vibeqc
