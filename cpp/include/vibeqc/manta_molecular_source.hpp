#pragma once
#include "vibeqc/manta.hpp"

namespace vibeqc::manta {
struct MolecularQuartetConstructionPlan {
    Count shell_calls=1, primitive_products=0, scalar_terms=0, kernel_terms=0;
    Count additional_bytes=0, modeled_peak_bytes=0;
};
MolecularQuartetConstructionPlan plan_molecular_quartet_construction(
    const OrderedJKBuilder&,const std::array<std::size_t,4>&,const ResourceBudget&);

// Diagnostic for one actual native Coulomb tile. The rigorous reference uses
// exact stored normalized s-shell data; source equivariance is NOT implied.
class MolecularQuartetConstruction {
public:
    MolecularQuartetConstruction(std::shared_ptr<OrderedJKBuilder>,
        const std::array<std::size_t,4>&,const ResourceBudget&);
    void validate(const OrderedJKBuilder&) const;
    const MolecularQuartetConstructionPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    const std::array<std::size_t,4>& key() const { return key_; }
    double evaluated() const { return evaluated_; }
    double lower() const { return lower_; }
    double upper() const { return upper_; }
    double error_bound() const { return error_; }
    bool null_tile() const { return null_; }
private:
    std::shared_ptr<OrderedJKBuilder> builder_;
    MolecularQuartetConstructionPlan plan_;
    std::string source_;
    std::array<std::size_t,4> key_;
    double evaluated_=0.,lower_=0.,upper_=0.,error_=0.;
    bool null_=false;
};

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
