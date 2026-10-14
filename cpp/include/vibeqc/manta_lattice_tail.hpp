#pragma once
#include "vibeqc/manta_multipole.hpp"
#include <limits>

namespace vibeqc::manta {
using LatticeIndex = std::array<std::int64_t,3>;
struct LatticeTailPlan {
    int order=0;
    Count moments=0,real_cells=0,reciprocal_vectors=0,near_cells=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,workspace_bytes=0;
    Count domain_bytes=0,modeled_peak_bytes=0;
};
struct LatticeTailMatrix {
    Eigen::MatrixXd values;
    std::vector<md::Index> indices;
    LatticeTailPlan plan;
    std::string source_identity,kernel_identity;
    std::string real_domain_identity,reciprocal_domain_identity,near_domain_identity;
    std::string normalization="raw-cartesian-moments";
    std::string boundary="3d-zero-average-conducting";
    std::string qualification="finite-domain-coefficients";
    double eta=0,cell_volume=0,gauge_constant=0;
    Count radial_derivatives_evaluated=0,reciprocal_coefficients_evaluated=0;
    Count matrix_coefficients_evaluated=0;
};
LatticeTailPlan plan_lattice_tail(int order,Count real_cells,Count reciprocal_vectors,
    Count near_cells,const ResourceBudget&);
// Cell vectors are columns, in bohr. Reciprocal vectors are 2*pi*A^(-T)*g.
// E = M(target)^T T(target-source) M(source), with total degree <= order.
// T differentiates the finite Ewald Green function minus the identical near
// bare-Coulomb images. This component alone certifies no infinite-sum error,
// Gaussian support, point self correction, symmetry reduction or SCF method.
LatticeTailMatrix multipole_lattice_tail(const Eigen::Matrix3d& cell,
    const md::Point& target_minus_source,double eta,
    const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,
    const std::vector<LatticeIndex>& near,int order,const ResourceBudget&);

struct LatticeTailRemainderPlan {
    int order=0;
    Count real_cells=0,reciprocal_vectors=0,near_cells=0;
    Count radial_integrals=0,scalar_terms=0,kernel_terms=0;
    Count output_bytes=0,domain_bytes=0,workspace_bytes=0,modeled_peak_bytes=0;
};
struct LatticeTailRemainder {
    LatticeTailRemainderPlan plan;
    double target_radius=0,source_radius=0,target_norm=0,source_norm=0;
    double near_remainder=0,real_remainder=0,reciprocal_remainder=0,total=0;
    double minimum_non_near_distance=std::numeric_limits<double>::infinity();
    std::string source_identity,expansion_identity,kernel_identity;
    std::string real_domain_identity,reciprocal_domain_identity,near_domain_identity;
    std::string qualification="finite-domain-compact-taylor";
    bool roundoff_included=false;
    Count radial_integrals_evaluated=0,reciprocal_terms_evaluated=0;
};
LatticeTailRemainderPlan plan_lattice_tail_remainder(int order,Count real_cells,
    Count reciprocal_vectors,Count near_cells,const ResourceBudget&);
// Absolute pair-energy Taylor remainder for compact signed distributions.
// Norms bound their integrals of absolute charge; radii enclose their supports.
// This is conditional on those caller envelopes. It excludes lattice-domain
// truncation, Gaussian tails and floating-point coefficient/contraction error.
LatticeTailRemainder lattice_tail_remainder(const Eigen::Matrix3d& cell,
    const md::Point& target_minus_source,double eta,
    const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,
    const std::vector<LatticeIndex>& near,int order,double target_radius,
    double source_radius,double target_norm,double source_norm,const ResourceBudget&);

struct LatticeDomainPlan {
    Count real_cells=0,reciprocal_vectors=0,near_cells=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,domain_bytes=0,workspace_bytes=0,modeled_peak_bytes=0;
};
struct LatticeDomainRemainder {
    LatticeDomainPlan plan;
    Count real_cube_radius=0,reciprocal_cube_radius=0;
    double target_radius=0,source_radius=0,target_norm=0,source_norm=0;
    double displacement_bound=0,real_metric_lower=0,reciprocal_metric_lower=0;
    double minimum_omitted_real_distance=0,real_remainder=0,reciprocal_remainder=0,total=0;
    std::string source_identity,kernel_identity,real_domain_identity,reciprocal_domain_identity,near_domain_identity;
    std::string qualification="compact-infinite-ewald-domain-remainder";
    bool roundoff_included=false;
};
LatticeDomainPlan plan_lattice_domain_remainder(Count real_cells,Count reciprocal_vectors,
    Count near_cells,const ResourceBudget&);
// Infinite-minus-finite Ewald error for conditional compact signed sources.
// Real and reciprocal domains must be complete centered integer cubes, with
// reciprocal zero excluded. The identical near subtraction cancels from this
// error. It is separate from Taylor/Gaussian/source-image/roundoff errors.
LatticeDomainRemainder lattice_domain_remainder(const Eigen::Matrix3d& cell,
    const md::Point& target_minus_source,double eta,
    const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,
    const std::vector<LatticeIndex>& near,double target_radius,double source_radius,
    double target_norm,double source_norm,const ResourceBudget&);
} // namespace vibeqc::manta
