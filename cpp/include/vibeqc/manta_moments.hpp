#pragma once
#include "vibeqc/manta.hpp"
#include "vibeqc/manta_md.hpp"

namespace vibeqc::manta {
struct PairMomentPlan {
    std::array<Count,2> shape{};
    int order=0;
    Count moments=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PairMomentReceipt {
    std::string source_identity,normalization="raw-cartesian-moments;libint-shell-axial-v1";
    Count shell_calls=1,primitive_pairs=0,primitive_expansions=0,primitive_moment_products=0,modeled_peak_bytes=0;
};
struct PairMomentResult {
    md::Matrix values;
    std::vector<md::Index> indices;
    std::array<Count,2> ao_shape{};
    PairMomentReceipt receipt;
};
PairMomentPlan plan_pair_moments(const BasisSet&,const std::array<std::size_t,2>&,int order,const ResourceBudget&);
PairMomentResult compute_pair_moments(const BasisSet&,const std::array<std::size_t,2>&,
    const md::Point& origin,int order,const ResourceBudget&);
struct AuxiliaryMomentPlan {
    std::array<Count,2> shape{};
    int order=0;
    Count moments=0,primitives=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct AuxiliaryMomentReceipt {
    std::string source_identity,normalization="raw-cartesian-moments;libint-shell-axial-v1";
    Count shell_calls=1,primitives=0,primitive_expansions=0,primitive_moment_products=0,modeled_peak_bytes=0;
};
struct AuxiliaryMomentResult {
    md::Matrix values;
    std::vector<md::Index> indices;
    AuxiliaryMomentReceipt receipt;
};
AuxiliaryMomentPlan plan_auxiliary_moments(const BasisSet&,std::size_t shell,int order,const ResourceBudget&);
AuxiliaryMomentResult compute_auxiliary_moments(const BasisSet&,std::size_t shell,
    const md::Point& origin,int order,const ResourceBudget&);
} // namespace vibeqc::manta
