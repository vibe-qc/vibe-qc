#pragma once
#include "vibeqc/manta_fitting.hpp"
namespace vibeqc::manta {
struct ThreeCenterConstructionTile {
    FittingTile evaluated;
    std::vector<double> lower,upper,error_bounds;
};
MetricConstructionPlan plan_three_center_construction(const BasisSet&,const BasisSet&,const BasisSet&,
    const KernelRequest&,const ResourceBudget&);
// Streamed component proof on exact stored normalized basis data. Periodic
// image sums, coupling propagation and relaxed-energy accuracy are separate.
class ThreeCenterConstructionTiles {
public:
    ThreeCenterConstructionTiles(const BasisSet&,const BasisSet&,const BasisSet&,
        const KernelRequest&,const ResourceBudget&);
    const MetricConstructionPlan& plan() const {return plan_;}
    const std::string& source_identity() const {return source_.source_identity();}
    ThreeCenterConstructionTile evaluate(const std::vector<std::size_t>&) const;
private:
    MetricConstructionPlan plan_;
    FittingTiles source_;
};
} // namespace vibeqc::manta
