#pragma once

#include "vibeqc/manta_fitting.hpp"

namespace vibeqc::manta {
struct DerivativePlan {
    std::vector<Count> shape,shell_extents;
    Count components=0,shell_calls=0,primitive_products=0,primitive_evaluations_upper_bound=0;
    Count integral_elements=0,tile_elements=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct DerivativeTile : FittingTile {
    Count primitive_evaluations=0;
};
inline KernelRequest derivative_request(std::size_t arity) {
    auto request=fitting_request(arity); request.derivative=1; return request;
}
DerivativePlan plan_derivatives(const std::vector<const BasisSet*>&,const KernelRequest&,const ResourceBudget&);

// One Cartesian derivative of one physical shell center per evaluation.
// Nuclear assembly, coefficient/metric derivatives and stress are consumers.
class DerivativeTiles {
public:
    DerivativeTiles(const std::vector<const BasisSet*>&,const KernelRequest&,const ResourceBudget&);
    const DerivativePlan& plan() const { return plan_; }
    const KernelRequest& kernel_request() const { return request_; }
    const std::string& source_identity() const { return source_; }
    DerivativeTile evaluate(const std::vector<std::size_t>& indices,int center,int axis) const;
private:
    DerivativePlan plan_;
    KernelRequest request_;
    std::vector<BasisSet> bases_;
    std::string source_;
};
} // namespace vibeqc::manta
