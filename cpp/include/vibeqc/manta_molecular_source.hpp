#pragma once
#include "vibeqc/manta.hpp"

namespace vibeqc::manta {
struct MolecularSourceSupportPlan {
    Count operations=0, shell_checks=0, radial_values=0;
    Count additional_bytes=0, modeled_peak_bytes=0, scalar_terms=0;
};
MolecularSourceSupportPlan plan_molecular_source_support(
    const OrderedJKBuilder&,Count operations,const ResourceBudget&);

// Producer-owned finite source/support evidence, not an arithmetic or group
// representation certificate. Shared symmetry still audits every retained
// ordered tile and stabilizer before reconstructing angular components.
class MolecularSourceSupport {
public:
    MolecularSourceSupport(std::shared_ptr<OrderedJKBuilder>,Count operations,
        const double* rotations,const double* translations,const std::int64_t* destinations,
        double geometry_tolerance,const ResourceBudget&);
    void validate(const OrderedJKBuilder&) const;
    void validate_maps(Count operations,Count shells,const std::int64_t*) const;
    const std::vector<std::int64_t>& destinations() const { return destinations_; }
    const MolecularSourceSupportPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    const std::string& identity() const { return identity_; }
    double maximum_center_residual() const { return residual_; }
    double geometry_tolerance() const { return tolerance_; }
private:
    std::shared_ptr<OrderedJKBuilder> builder_;
    MolecularSourceSupportPlan plan_;
    std::string source_,identity_;
    std::vector<std::int64_t> destinations_;
    double residual_=0.,tolerance_=0.;
};
} // namespace vibeqc::manta
