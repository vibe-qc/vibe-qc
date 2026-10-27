#pragma once

// Internal MANTA provider/consumer boundary. The initial reference provider
// uses libint; a provider name never implies a Coulomb or exchange algorithm.
#include "vibeqc/basis.hpp"
#include "vibeqc/jk_builder.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace vibeqc::manta {
using Count = std::uint64_t;
struct KernelRequest {
    std::string kernel = "coulomb";
    double omega = 0.;
    std::string scalar = "real";
    int centers = 4;
    int derivative = 0;
    std::array<std::array<std::int64_t, 3>, 4> images{};
};
struct Capability {
    // Angular limits by axis role for the queried kernel, center count and
    // derivative order. Auxiliary axes are both axes of a two-center request
    // and the first axis of a three-center request; every other axis is an
    // orbital axis. Planners derive their admission from these limits.
    bool supported = false;
    int maximum_l = 0;
    int maximum_auxiliary_l = 0;
    std::string reason;
};
struct ResourceBudget {
    Count memory_bytes = Count{256} << 20;
    Count shell_calls = 10000000;
    Count primitive_quartets = 100000000000ULL;
    Count scalar_terms = 100000000000ULL;
    Count kernel_terms = 1000000000000ULL;
    Count other_live_bytes = 0;
    Count backend_margin_bytes = Count{8} << 20;
};
struct BuildPlan {
    std::string provider = "libint";
    std::string schedule = "ordered-jk";
    // Radial kernel the workspace engine is built for: coulomb, erf or erfc
    // with its range parameter; a workspace serves only a matching provider.
    std::string kernel = "coulomb";
    double omega = 0.;
    Count n_basis = 0, n_shells = 0, maximum_primitives = 0;
    int maximum_l = 0;
    Count basis_payload_bytes = 0, engine_bytes = 0, target_bytes = 0;
    Count tile_elements = 0, modeled_peak_bytes = 0;
    Count shell_calls = 0, primitive_quartets_upper_bound = 0, scalar_terms = 0;
    Count kernel_terms = 0;
    Count channel_batch = 1;
};
struct BuildReceipt {
    std::string provider, provider_version, source_identity, density_identity;
    std::string kernel = "coulomb";
    double omega = 0.;
    std::string normalization = "libint-shell-axial-v1";
    std::string algorithm = "ordered-direct";
    Count shell_calls = 0, null_tiles = 0, integral_elements_consumed = 0;
    Count primitive_quartets_requested = 0, density_terms_accumulated = 0;
    Count modeled_peak_bytes = 0;
    Count primitive_quartets_evaluated = 0;
    bool primitive_evaluations_measured = false;
    std::string contraction_strategy = "segmented";
    double primitive_precision = 0;
};
struct JKResult {
    Eigen::MatrixXd J, K;
    BuildReceipt receipt;
};
struct BatchPlan {
    Count n_densities=0, shell_calls=0, primitive_quartets_upper_bound=0;
    Count scalar_terms=0, kernel_terms=0, modeled_peak_bytes=0;
};
struct BatchResult {
    std::vector<Eigen::MatrixXd> J, K;
    std::vector<std::string> density_identities;
    BuildReceipt receipt;
};
struct TileView {
    const double* values = nullptr;
    std::array<std::size_t, 4> shape{};
    // Last index contiguous; no permutation multiplicities. A null buffer
    // denotes a screened/underflowed libint tile, not a certified exact zero.
};
struct TileBatch {
    std::vector<TileView> tiles;
    Count primitive_quartets_evaluated=0;
    bool primitive_evaluations_measured=false;
};

Capability libint_capability(const KernelRequest&);
BuildPlan plan_libint(const BasisSet&, const KernelRequest&, const ResourceBudget&);
BuildPlan plan_libint_tiles(const BasisSet&,const KernelRequest&,const ResourceBudget&);

class Workspace {
public:
    explicit Workspace(const BuildPlan&);
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string backend_;
    std::vector<double> values_;
    friend class LibintProvider;
    friend class MantaProvider;
};

class IntegralProvider {
public:
    virtual ~IntegralProvider() = default;
    virtual const BasisSet& basis() const = 0;
    virtual const BuildPlan& plan() const = 0;
    virtual const std::string& source_identity() const = 0;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual KernelRequest kernel_request() const { return {}; }
    // The view is borrowed from this workspace until its next evaluation.
    virtual TileView evaluate(const std::array<std::size_t, 4>&, Workspace&) const = 0;
    virtual TileBatch evaluate_batch(const std::vector<std::array<std::size_t,4>>&,Workspace&) const;
};

class LibintProvider final : public IntegralProvider {
public:
    LibintProvider(const BasisSet&, const KernelRequest&, const ResourceBudget&,bool tiles_only=false);
    const BasisSet& basis() const override { return basis_; }
    const BuildPlan& plan() const override { return plan_; }
    const std::string& source_identity() const override { return source_; }
    std::string name() const override { return "libint"; }
    std::string version() const override;
    KernelRequest kernel_request() const override { return request_; }
    TileView evaluate(const std::array<std::size_t, 4>&, Workspace&) const override;
private:
    // Declaration order matters: complete admission precedes the basis copy.
    BuildPlan plan_;
    BasisSet basis_;
    KernelRequest request_;
    std::string source_;
};

class OrderedJKBuilder : public JKBuilder {
public:
    const BuildPlan& plan() const { return provider_->plan(); }
    const IntegralProvider& provider() const { return *provider_; }
    JKResult build(const Eigen::MatrixXd&, bool make_j = true, bool make_k = true) const;
    BatchPlan plan_batch(Count n_densities, bool make_j=true, bool make_k=true) const;
    BatchResult build_batch(const std::vector<Eigen::MatrixXd>&, bool make_j=true, bool make_k=true) const;
    Eigen::MatrixXd build_J(const Eigen::MatrixXd&) const override;
    Eigen::MatrixXd build_K(const Eigen::MatrixXd&) const override;
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> build_K_pair(const Eigen::MatrixXd&,const Eigen::MatrixXd&) const override;
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd&, double alpha_hf = 1.) const override;
protected:
    void validate_density(const Eigen::MatrixXd&) const;
    const ResourceBudget& budget() const { return budget_; }
    OrderedJKBuilder(std::shared_ptr<const IntegralProvider> provider,const ResourceBudget& budget)
        : provider_(std::move(provider)),budget_(budget) {}
private:
    BatchResult contract(const std::vector<const Eigen::MatrixXd*>&,bool make_j,bool make_k,Count peak) const;
    std::shared_ptr<const IntegralProvider> provider_;
    ResourceBudget budget_;
};
class ReferenceJKBuilder final : public OrderedJKBuilder {
public:
    ReferenceJKBuilder(const BasisSet&, const KernelRequest&, const ResourceBudget&);
};
} // namespace vibeqc::manta
