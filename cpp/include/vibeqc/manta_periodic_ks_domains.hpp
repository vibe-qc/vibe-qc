#pragma once
#include "vibeqc/manta_periodic_ks.hpp"
namespace vibeqc::manta {
struct PeriodicKSDomainPlan {
    PeriodicKSPlan ks;
    double reciprocal_energy_cutoff=0.,pair_cutoff_bohr=0.,translation_radius_bohr=0.,xc_image_cutoff_bohr=0.;
    Count translation_candidates=0,hartree_candidates=0,xc_image_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_work=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicKSDomainReceipt {
    std::string identity;
    Count translations=0,hartree_vectors=0,xc_images=0,conjugacy_vectors=0;
};
struct PeriodicKSDomains {
    PeriodicKSDomainPlan plan;
    PeriodicKSDomainReceipt receipt;
    md::Matrix translations,reciprocal_vectors;
    Eigen::MatrixXd xc_images;
};
template<class Options> struct PeriodicHybridKSDomainPlan {
    PeriodicKSPlan ks;
    double reciprocal_energy_cutoff=0.,pair_cutoff_bohr=0.,translation_radius_bohr=0.,xc_image_cutoff_bohr=0.;
    Count momenta=0,translation_candidates=0,hartree_candidates=0,xc_image_candidates=0;
    Count maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,primitive_points=0,primitive_work=0,scalar_terms=0,kernel_terms=0;
};
template<class Options> struct PeriodicHybridKSDomainReceipt {
    std::string identity,pair_identity,xc_image_identity;
    Count translations=0,hartree_vectors=0,exchange_vectors=0,xc_images=0,conjugacy_vectors=0;
};
template<class Options> struct PeriodicHybridKSDomains {
    PeriodicHybridKSDomainPlan<Options> plan;
    PeriodicHybridKSDomainReceipt<Options> receipt;
    md::Matrix translations,reciprocal_vectors,exchange_vectors;
    std::vector<Count> transfer_offsets;
    Eigen::MatrixXd xc_images;
};
using PeriodicGlobalHybridKSDomainPlan=PeriodicHybridKSDomainPlan<PeriodicGlobalHybridKSOptions>;
using PeriodicGlobalHybridKSDomainReceipt=PeriodicHybridKSDomainReceipt<PeriodicGlobalHybridKSOptions>;
using PeriodicGlobalHybridKSDomains=PeriodicHybridKSDomains<PeriodicGlobalHybridKSOptions>;
using PeriodicRIGlobalHybridKSDomainPlan=PeriodicHybridKSDomainPlan<PeriodicRIGlobalHybridKSOptions>;
using PeriodicRIGlobalHybridKSDomainReceipt=PeriodicHybridKSDomainReceipt<PeriodicRIGlobalHybridKSOptions>;
using PeriodicRIGlobalHybridKSDomains=PeriodicHybridKSDomains<PeriodicRIGlobalHybridKSOptions>;
using PeriodicRIRangeSeparatedKSDomainPlan=PeriodicHybridKSDomainPlan<PeriodicRIRangeSeparatedKSOptions>;
using PeriodicRIRangeSeparatedKSDomainReceipt=PeriodicHybridKSDomainReceipt<PeriodicRIRangeSeparatedKSOptions>;
using PeriodicRIRangeSeparatedKSDomains=PeriodicHybridKSDomains<PeriodicRIRangeSeparatedKSOptions>;
using PeriodicRangeSeparatedKSDomainPlan=PeriodicHybridKSDomainPlan<PeriodicRangeSeparatedKSOptions>;
using PeriodicRangeSeparatedKSDomainReceipt=PeriodicHybridKSDomainReceipt<PeriodicRangeSeparatedKSOptions>;
using PeriodicRangeSeparatedKSDomains=PeriodicHybridKSDomains<PeriodicRangeSeparatedKSOptions>;
// Positive physical pair and reciprocal kinetic cutoffs, nonnegative canonical
// XC image radius. The quadrature remains caller supplied and independently
// refined. Count-only boxes admit every source traversal and worst-case SCF
// before enumeration. No exchange transfer vectors are generated for pure KS.
PeriodicKSDomainPlan plan_periodic_ks_from_cutoffs(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,Count grid_points,
    const PeriodicKSOptions&,const ResourceBudget&);
PeriodicKSDomains build_periodic_ks_domains(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,Count grid_points,
    const PeriodicKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ks_from_cutoffs(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,const PeriodicKSOptions&,const ResourceBudget&);
// Same certified G/Q spheres and physical pair source as HF, with independent
// canonical XC images and complete GLOBAL HYBRID admission, never an HF proxy.
PeriodicGlobalHybridKSDomainPlan plan_periodic_global_hybrid_ks_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicGlobalHybridKSDomains build_periodic_global_hybrid_ks_domains(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_global_hybrid_ks_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,
    const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
// Same nonzero base Q spheres; the complete range-separated method owns the
// finite erfc zero expansion, every component kernel and singular convention.
PeriodicRangeSeparatedKSDomainPlan plan_periodic_range_separated_ks_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicRangeSeparatedKSDomains build_periodic_range_separated_ks_domains(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_range_separated_ks_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,
    const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
// Fitted pure KS shares the pure-domain schema and geometry; its .ks plan
// admits the actual auxiliary/projector owners and functional solver. Hybrid
// K selects exact target-minus-source or fitted ket-minus-bra spheres.
PeriodicKSDomainPlan plan_periodic_ri_ks_from_cutoffs(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRIKSOptions&,const ResourceBudget&);
PeriodicKSDomains build_periodic_ri_ks_domains(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRIKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_ks_from_cutoffs(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,const PeriodicRIKSOptions&,const ResourceBudget&);
PeriodicRIGlobalHybridKSDomainPlan plan_periodic_ri_global_hybrid_ks_from_cutoffs(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicRIGlobalHybridKSDomains build_periodic_ri_global_hybrid_ks_domains(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_global_hybrid_ks_from_cutoffs(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
// Each resolved physical exchange kernel keeps its own fitting source; the
// generated geometry never supplies the finite screened Gamma zero.
PeriodicRIRangeSeparatedKSDomainPlan plan_periodic_ri_range_separated_ks_from_cutoffs(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicRIRangeSeparatedKSDomains build_periodic_ri_range_separated_ks_domains(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    Count grid_points,const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_range_separated_ks_from_cutoffs(const BasisSet&,const BasisSet& auxiliary,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
