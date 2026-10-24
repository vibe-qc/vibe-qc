#pragma once
#include "vibeqc/manta.hpp"

namespace vibeqc::manta {
struct ExtentOptions {
    double pointwise_tolerance=1e-14,tail_mass_tolerance=1e-12;
    double minimum_radius=2.,maximum_radius=1e4;
    int moment_order=6;
    bool retain_full_moments=false;
};
struct ExtentPlan {
    Count shell_pairs=0,primitive_pairs=0,maximum_pair_primitives=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PairExtent {
    std::array<std::size_t,2> shells{};
    std::array<double,3> center{};
    double radius=0.,pointwise_bound=0.,tail_mass_bound=0.;
    double l1_norm_bound=0.,linfinity_bound=0.,potential_bound=0.;
    std::vector<double> tail_moment_bounds;
    std::vector<double> full_moment_bounds;
    bool always_near=false;
    std::string reason;
};
struct ExtentReceipt {
    std::string source_identity,center_policy="geometric-midpoint",envelope="absolute-cartesian-polynomial-v1";
    Count shell_pairs=0,primitive_pairs=0,envelope_evaluations=0,always_near_pairs=0,modeled_peak_bytes=0;
};
struct PairExtents {
    std::vector<PairExtent> pairs;
    ExtentReceipt receipt;
};
ExtentPlan plan_pair_extents(const BasisSet&,const ExtentOptions&,const ResourceBudget&);
PairExtents compute_pair_extents(const BasisSet&,const ExtentOptions&,const ResourceBudget&);
struct AuxiliaryExtent {
    std::size_t shell=0;
    std::array<double,3> center{};
    double radius=0.,pointwise_bound=0.,tail_mass_bound=0.;
    double l1_norm_bound=0.,linfinity_bound=0.,potential_bound=0.;
    std::vector<double> tail_moment_bounds,full_moment_bounds;
    bool always_near=false;
    std::string reason;
};
struct AuxiliaryExtentPlan {
    Count shells=0,primitives=0,maximum_shell_primitives=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct AuxiliaryExtentReceipt {
    std::string source_identity,center_policy="shell-center",envelope="absolute-cartesian-polynomial-v1";
    Count shells=0,primitives=0,envelope_evaluations=0,always_near_shells=0,modeled_peak_bytes=0;
};
struct AuxiliaryExtents {
    std::vector<AuxiliaryExtent> shells;
    AuxiliaryExtentReceipt receipt;
};
AuxiliaryExtentPlan plan_auxiliary_extents(const BasisSet&,const ExtentOptions&,const ResourceBudget&);
AuxiliaryExtents compute_auxiliary_extents(const BasisSet&,const ExtentOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
