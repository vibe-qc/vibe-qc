#pragma once
#include "vibeqc/manta_periodic_nuclear.hpp"
#include "vibeqc/manta_lattice_tail.hpp"

namespace vibeqc::manta::nuclear_detail {
std::vector<LatticeIndex> closed_support(const md::Matrix&,const Eigen::Matrix3d&,bool exclude_zero);
Count candidate_bound(const PeriodicSystem&,double cutoff);
std::string source_identity(const BasisSet&,const PeriodicSystem&,const md::Matrix& g,
    const md::Matrix& cells,const md::Matrix& k,double alpha,double cutoff,std::optional<double> pair_cutoff);
} // namespace vibeqc::manta::nuclear_detail
