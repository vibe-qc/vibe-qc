#pragma once
#include "vibeqc/manta.hpp"
#include "vibeqc/manta_md.hpp"

namespace vibeqc::manta {
struct MultipolePlan {
    int order=0;
    Count moments=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct MultipoleMatrix {
    Eigen::MatrixXd values;
    std::vector<md::Index> indices;
    std::string operation,source_identity,normalization="raw-cartesian-moments";
    Count coefficients_evaluated=0,modeled_peak_bytes=0;
};
MultipolePlan plan_multipoles(int order,const ResourceBudget&);
// M(target) = W(source_center-target_center) M(source).
MultipoleMatrix multipole_translation(const md::Point&,int order,const ResourceBudget&);
// E = M(target)^T T(target_center-source_center) M(source), total degree <= order.
MultipoleMatrix multipole_coulomb(const md::Point&,int order,const ResourceBudget&);
double compact_multipole_remainder(double distance,double target_radius,double source_radius,
    double target_norm,double source_norm,int order);
} // namespace vibeqc::manta
