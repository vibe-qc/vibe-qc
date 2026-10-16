#pragma once
#include "vibeqc/manta_gaussian_majorants.hpp"

namespace vibeqc::manta::gaussian_domain_detail {
// The same inverse-residual certificate as the Ewald-domain proof, without
// constructing an unrelated kernel or domain for a one-electron source.
double real_metric_lower(const Eigen::Matrix3d&);
// Prepared only after joint admission. A consuming source validates/sorts its
// immutable domains once, then streams component bounds without domain copies.
struct Geometry {
    Count real_cells=0,reciprocal_vectors=0,near_cells=0;
    Count real_cube_radius=0,reciprocal_cube_radius=0;
    double eta=0,volume=0,metric=0,reciprocal_metric=0;
    std::string kernel_identity,real_identity,reciprocal_identity,near_identity;
};
Geometry prepare(const Eigen::Matrix3d&,double,const std::vector<LatticeIndex>&,
    const std::vector<LatticeIndex>&,const std::vector<LatticeIndex>&);
LatticeGaussianDomainRemainder evaluate(const Geometry&,const GaussianMajorant&,
    const GaussianMajorant&,const ResourceBudget&);
struct PointRemainder {
    double real=0,reciprocal=0,total=0;
};
// Unit point charges are exact distributions, not fake finite-width Gaussian
// envelopes. These private consumers inherit the native source's admission.
PointRemainder point_pair(const Geometry&,const md::Point&,const md::Point&);
PointRemainder gaussian_point(const Geometry&,const GaussianMajorant&,const md::Point&);
} // namespace vibeqc::manta::gaussian_domain_detail
