#pragma once
#include "vibeqc/manta.hpp"

namespace vibeqc::manta {
Capability md_capability(const KernelRequest&);
BuildPlan plan_md(const BasisSet&,const KernelRequest&,const ResourceBudget&,Count channel_batch=1);
BuildPlan plan_md_tiles(const BasisSet&,const KernelRequest&,const ResourceBudget&);

class MantaProvider final : public IntegralProvider {
public:
    MantaProvider(const BasisSet&,const KernelRequest&,const ResourceBudget&,Count channel_batch=1,bool tiles_only=false);
    const BasisSet& basis() const override { return basis_; }
    const BuildPlan& plan() const override { return plan_; }
    const std::string& source_identity() const override { return source_; }
    std::string name() const override { return "manta"; }
    std::string version() const override { return "md-fp64-v1"; }
    KernelRequest kernel_request() const override { return request_; }
    TileView evaluate(const std::array<std::size_t,4>&,Workspace&) const override;
    TileBatch evaluate_batch(const std::vector<std::array<std::size_t,4>>&,Workspace&) const override;
private:
    BuildPlan plan_;
    BasisSet basis_;
    std::string source_;
    KernelRequest request_;
};
class DirectJKBuilder final : public OrderedJKBuilder {
public:
    DirectJKBuilder(const BasisSet& b,const KernelRequest& r,const ResourceBudget& budget,Count channel_batch=1)
        : OrderedJKBuilder(std::make_shared<MantaProvider>(b,r,budget,channel_batch),budget) {}
    Eigen::MatrixXd build_K_erf(const Eigen::MatrixXd&,double omega) const override;
};
} // namespace vibeqc::manta
