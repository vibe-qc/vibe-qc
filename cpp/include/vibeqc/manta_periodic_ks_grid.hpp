#pragma once
#include "vibeqc/manta_periodic_ks_domains.hpp"
#include "vibeqc/manta_periodic_quadrature.hpp"
namespace vibeqc::manta {
struct PeriodicKSGridPlan {
    PeriodicQuadraturePlan quadrature;
    PeriodicKSDomainPlan domains;
    Count grid_live_bytes=0,modeled_peak_bytes=0,shell_calls=0,primitive_work=0,scalar_terms=0,kernel_terms=0;
};
template<class Options> struct PeriodicHybridKSGridPlan {
    PeriodicQuadraturePlan quadrature;
    PeriodicHybridKSDomainPlan<Options> domains;
    Count grid_live_bytes=0,modeled_peak_bytes=0,shell_calls=0,primitive_work=0,scalar_terms=0,kernel_terms=0;
};
using PeriodicGlobalHybridKSGridPlan=PeriodicHybridKSGridPlan<PeriodicGlobalHybridKSOptions>;
using PeriodicRIGlobalHybridKSGridPlan=PeriodicHybridKSGridPlan<PeriodicRIGlobalHybridKSOptions>;
using PeriodicRIRangeSeparatedKSGridPlan=PeriodicHybridKSGridPlan<PeriodicRIRangeSeparatedKSOptions>;
using PeriodicRangeSeparatedKSGridPlan=PeriodicHybridKSGridPlan<PeriodicRangeSeparatedKSOptions>;
// A single admission covers generation, retained grid payload, cutoff domains
// and the complete physical KS solver. Explicit grid controls remain finite
// source choices; this API does not assert automatic quadrature accuracy.
PeriodicKSGridPlan plan_periodic_ks_with_quadrature(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ks_with_quadrature(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicKSOptions&,const ResourceBudget&);
PeriodicGlobalHybridKSGridPlan plan_periodic_global_hybrid_ks_with_quadrature(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_global_hybrid_ks_with_quadrature(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicRangeSeparatedKSGridPlan plan_periodic_range_separated_ks_with_quadrature(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_range_separated_ks_with_quadrature(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSGridPlan plan_periodic_ri_ks_with_quadrature(const BasisSet&,const BasisSet& auxiliary,
    const PeriodicSystem&,const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRIKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_ks_with_quadrature(const BasisSet&,const BasisSet& auxiliary,
    const PeriodicSystem&,const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRIKSOptions&,const ResourceBudget&);
PeriodicRIGlobalHybridKSGridPlan plan_periodic_ri_global_hybrid_ks_with_quadrature(const BasisSet&,const BasisSet& auxiliary,
    const PeriodicSystem&,const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_global_hybrid_ks_with_quadrature(const BasisSet&,const BasisSet& auxiliary,
    const PeriodicSystem&,const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicRIRangeSeparatedKSGridPlan plan_periodic_ri_range_separated_ks_with_quadrature(const BasisSet&,const BasisSet& auxiliary,
    const PeriodicSystem&,const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_range_separated_ks_with_quadrature(const BasisSet&,const BasisSet& auxiliary,
    const PeriodicSystem&,const RegularKMesh&,const Functional&,double reciprocal_energy_cutoff,double xc_image_cutoff_bohr,
    const PeriodicQuadratureOptions&,const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
