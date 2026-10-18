#pragma once
#include "vibeqc/manta_lattice_tail.hpp"

namespace vibeqc::manta {
struct GaussianMajorantTerm {
    md::Point center{};
    double exponent=0,charge_bound=0;
};
struct GaussianMajorantPlan {
    Count maximum_terms=0,shell_calls=1,primitive_pairs=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
class GaussianMajorant;
GaussianMajorantPlan plan_pair_gaussian_majorant(const BasisSet&,std::array<Count,2>,const ResourceBudget&);
GaussianMajorantPlan plan_auxiliary_gaussian_majorant(const BasisSet&,Count,const ResourceBudget&);
GaussianMajorant compute_pair_gaussian_majorant(const BasisSet&,std::array<Count,2>,const md::Point&,const ResourceBudget&);
GaussianMajorant compute_auxiliary_gaussian_majorant(const BasisSet&,Count,const ResourceBudget&);

// Native producer-only pointwise envelope, uniform over every AO component of
// one orbital pair or auxiliary shell. No user-defined numerical constructor.
class GaussianMajorant final {
public:
    const std::vector<GaussianMajorantTerm>& terms() const { return terms_; }
    const GaussianMajorantPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    const std::string& basis_identity() const { return basis_; }
    const std::string& kind() const { return kind_; }
    double charge_bound() const { return charge_; }
private:
    GaussianMajorant()=default;
    GaussianMajorantPlan plan_;
    std::vector<GaussianMajorantTerm> terms_;
    std::string source_,basis_,kind_;
    double charge_=0;
    friend GaussianMajorant compute_pair_gaussian_majorant(const BasisSet&,std::array<Count,2>,const md::Point&,const ResourceBudget&);
    friend GaussianMajorant compute_auxiliary_gaussian_majorant(const BasisSet&,Count,const ResourceBudget&);
};

struct LatticeGaussianDomainPlan {
    LatticeDomainPlan domain;
    Count target_terms=0,source_terms=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,majorant_bytes=0,modeled_peak_bytes=0;
};
struct LatticeGaussianDomainRemainder {
    LatticeGaussianDomainPlan plan;
    double physical_eta=0,minimum_effective_eta=std::numeric_limits<double>::infinity();
    double minimum_omitted_real_distance=std::numeric_limits<double>::infinity();
    double real_remainder=0,reciprocal_remainder=0,total=0;
    Count majorant_pairs_evaluated=0,real_cube_radius=0,reciprocal_cube_radius=0;
    std::string source_identity,kernel_identity,target_majorant_identity,source_majorant_identity;
    std::string real_domain_identity,reciprocal_domain_identity,near_domain_identity;
    std::string qualification="native-gaussian-infinite-ewald-domain-remainder";
    bool roundoff_included=false;
};
LatticeGaussianDomainPlan plan_lattice_gaussian_domain_remainder(Count target_terms,Count source_terms,
    Count real_cells,Count reciprocal_vectors,Count near_cells,const ResourceBudget&);
LatticeGaussianDomainRemainder lattice_gaussian_domain_remainder(const Eigen::Matrix3d& cell,double eta,
    const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,const std::vector<LatticeIndex>& near,
    const GaussianMajorant& target,const GaussianMajorant& source,const ResourceBudget&);
} // namespace vibeqc::manta
