#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

namespace vibeqc {

/// One evaluated representative-amplitude snapshot, before any next update.
struct TranslationMP2Progress {
    int iteration;
    double energy;
    double energy_change;
    double maximum_residual;
};

struct TranslationMP2Result {
    std::vector<Eigen::MatrixXd> amplitudes;
    std::vector<Eigen::MatrixXd> residuals;
    std::vector<TranslationMP2Progress> trace;
    bool converged = false;
};

/// Experimental real restricted finite-torus Jacobi solve. Only representative
/// tensors are retained; translated source bases are contracted on demand.
/// The raw source-space residual follows Nejad et al., doi:10.1063/5.0290816,
/// Eqs. (42)-(44). No occupied-Fock cutoff or denominator floor is applied.
TranslationMP2Result translation_mp2_solve(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& integrals,
    const std::vector<Eigen::MatrixXd>& virtual_fock,
    const std::vector<Eigen::MatrixXd>& seeds,
    const std::vector<Eigen::MatrixXd>& coefficients,
    const Eigen::MatrixXd& occupied_fock, const Eigen::MatrixXd& overlap,
    const std::vector<double>& energy_weights, int maximum_iterations,
    double energy_tolerance, double residual_tolerance, double damping,
    double symmetry_tolerance);

/// Native-only prepared payload. Python exposes no constructor or mutable data.
/// No AO matrices, AO coefficients, or placed-pair tensors survive preparation.
struct TranslationMP2Transport {
    struct Link {
        Eigen::MatrixXd overlap;
        std::size_t source;
    };
    struct Coupling {
        std::size_t link;
        double fock;
        bool transpose;
    };
    std::vector<Link> links;
    std::vector<Coupling> couplings;
    std::vector<Eigen::MatrixXd> virtual_fock, denominators, stabilizers;
    std::vector<double> energy_weights;
    int n_occupied = 0;
    double symmetry_tolerance = 0;
    bool compact = false;
    bool recipe_only = false;
    bool cyclic_metric = false;
    std::uint64_t stable_links = 0;
    std::uint64_t retained_bytes = 0, workspace_bytes = 0, required_bytes = 0;
};

TranslationMP2Transport prepare_translation_mp2_transport(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& virtual_fock,
    const std::vector<Eigen::MatrixXd>& coefficients,
    const Eigen::MatrixXd& occupied_fock, const Eigen::MatrixXd& overlap,
    const std::vector<double>& energy_weights, double symmetry_tolerance,
    std::uint64_t maximum_bytes,
    const std::vector<std::vector<int>>& domains = {},
    const std::vector<Eigen::MatrixXd>& local_coefficients = {},
    const std::vector<Eigen::MatrixXd>& occupied_coefficients = {},
    const Eigen::MatrixXd& all_occupied = Eigen::MatrixXd(),
    bool occupied_home = false);

/// The same prepared graph, with S[(R,a),(T,b)] = home_blocks[T-R](a,b).
/// All reciprocal mesh symbols are validated without a full AO matrix.
TranslationMP2Transport prepare_translation_mp2_transport_blocks(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& virtual_fock,
    const std::vector<Eigen::MatrixXd>& coefficients,
    const Eigen::MatrixXd& occupied_fock,
    const std::vector<Eigen::MatrixXd>& overlap_home_blocks,
    const std::vector<double>& energy_weights, double symmetry_tolerance,
    std::uint64_t maximum_bytes,
    const std::vector<std::vector<int>>& domains = {},
    const std::vector<Eigen::MatrixXd>& local_coefficients = {},
    const std::vector<Eigen::MatrixXd>& occupied_coefficients = {},
    const Eigen::MatrixXd& all_occupied = Eigen::MatrixXd(),
    bool occupied_home = false);

TranslationMP2Result translation_mp2_solve_prepared(
    const TranslationMP2Transport& transport,
    const std::vector<Eigen::MatrixXd>& integrals,
    const std::vector<Eigen::MatrixXd>& seeds, int maximum_iterations,
    double energy_tolerance, double residual_tolerance, double damping);

}  // namespace vibeqc
