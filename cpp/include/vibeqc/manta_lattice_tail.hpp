#pragma once
#include "vibeqc/manta_multipole.hpp"

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
} // namespace vibeqc::manta
