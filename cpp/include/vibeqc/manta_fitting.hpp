#pragma once

#include "vibeqc/manta.hpp"
#include "vibeqc/manta_md.hpp"

namespace vibeqc::manta {
struct FittingPlan {
    std::vector<Count> shape, shell_extents;
    Count shell_calls=0, primitive_products=0, kernel_terms=0;
    Count integral_elements=0, tile_elements=0, modeled_peak_bytes=0;
};
struct MetricConstructionPlan {
    Count modeled_peak_bytes=0,scalar_terms=0,kernel_terms=0,shell_calls=0,primitive_products=0;
};
struct MetricConstructionCertificate {
    md::Matrix evaluated,lower,upper,error_bounds;
};
MetricConstructionPlan plan_metric_construction(const BasisSet&,const BasisSet&,const KernelRequest&,const ResourceBudget&);
MetricConstructionCertificate certify_metric_construction(const BasisSet&,const BasisSet&,const KernelRequest&,const ResourceBudget&);
struct FittingTile {
    md::Matrix values;
    std::vector<Count> shape;
    Count primitive_products=0;
};
inline KernelRequest fitting_request(std::size_t arity) {
    KernelRequest r; r.centers=static_cast<int>(arity); return r;
}
FittingPlan plan_fitting(const std::vector<const BasisSet*>& axes,const KernelRequest&,const ResourceBudget&);
inline FittingPlan plan_fitting(const std::vector<const BasisSet*>& axes,const ResourceBudget& budget) {
    return plan_fitting(axes,fitting_request(axes.size()),budget);
}

// Two axes mean (P|Q); three axes mean (P|mu nu). Auxiliary axes admit
// l<=7 and orbital axes l<=6. Each evaluation returns one owned shell tile.
// No full three-index tensor or metric matrix is retained by this provider.
class FittingTiles {
public:
    FittingTiles(const std::vector<const BasisSet*>& axes,const ResourceBudget& budget)
        : FittingTiles(axes,fitting_request(axes.size()),budget) {}
    FittingTiles(const std::vector<const BasisSet*>& axes,const KernelRequest&,const ResourceBudget&);
    const KernelRequest& kernel_request() const { return request_; }
    const FittingPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    const BasisSet& basis(std::size_t axis) const { return bases_.at(axis); }
    FittingTile evaluate(const std::vector<std::size_t>& indices) const;
private:
    FittingPlan plan_;
    std::vector<BasisSet> bases_;
    std::string source_;
    KernelRequest request_;
};
} // namespace vibeqc::manta
