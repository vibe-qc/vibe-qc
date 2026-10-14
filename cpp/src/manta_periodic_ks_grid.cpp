#include "vibeqc/manta_periodic_ks_grid.hpp"
#include "vibeqc/manta_admission.hpp"
namespace vibeqc::manta {
namespace {
using namespace admission;
template<class Plan> ResourceBudget remaining(const ResourceBudget& b,const Plan& p) {
    ResourceBudget out=b;out.other_live_bytes=add(b.other_live_bytes,p.grid_live_bytes);
    out.scalar_terms-=p.quadrature.scalar_terms;out.kernel_terms-=p.quadrature.kernel_terms;return out;
}
template<class Plan,class Options,class Planner>
Plan plan_with_grid(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const Options& options,const ResourceBudget& budget,Planner planner) {
    Plan p;p.quadrature=plan_periodic_quadrature(system,grid_options,budget);
    // output_bytes admits the complete shared Grid plus 32 bytes/point for
    // explicit KS source copies. Atom metadata and wrapper/receipt owners
    // remain live while the domains, solver and terminal builds execute.
    p.grid_live_bytes=admission::add(p.quadrature.output_bytes,admission::add(32768,admission::mul(256,p.quadrature.atoms)));
    p.domains=planner(basis,system,mesh,f,cutoff,xc_radius,p.quadrature.grid_points,options,remaining(budget,p));
    p.modeled_peak_bytes=std::max(p.quadrature.modeled_peak_bytes,p.domains.modeled_peak_bytes);
    p.shell_calls=p.domains.shell_calls;p.primitive_work=p.domains.primitive_work;
    p.scalar_terms=admission::add(p.quadrature.scalar_terms,p.domains.scalar_terms);
    p.kernel_terms=admission::add(p.quadrature.kernel_terms,p.domains.kernel_terms);
    return p;
}
template<class Plan,class Options,class Runner>
PeriodicKSResult run_with_grid(const Plan& p,const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const Options& options,const ResourceBudget& budget,Runner runner) {
    const auto grid=build_periodic_quadrature(system,grid_options,budget);
    const Eigen::MatrixXd points=grid.grid.points;
    const std::vector<double> weights(grid.grid.weights.data(),grid.grid.weights.data()+grid.grid.weights.size());
    auto result=runner(basis,system,mesh,f,cutoff,xc_radius,points,weights,options,remaining(budget,p));
    auto& plan=result.plan;plan.quadrature=grid.plan;plan.quadrature_live_bytes=p.grid_live_bytes;
    plan.scalar_terms=admission::add(plan.scalar_terms,p.quadrature.scalar_terms);
    plan.kernel_terms=admission::add(plan.kernel_terms,p.quadrature.kernel_terms);
    plan.modeled_peak_bytes=std::max(plan.modeled_peak_bytes,p.quadrature.modeled_peak_bytes);
    auto& receipt=result.receipt;receipt.quadrature=grid.receipt;
    receipt.grid_radial_points=grid_options.radial_points;receipt.grid_lebedev_order=grid_options.lebedev_order;
    receipt.grid_partition_radius_bohr=grid_options.partition_radius_bohr;receipt.grid_stratmann=grid_options.stratmann;
    receipt.modeled_peak_bytes=plan.modeled_peak_bytes;return result;
}
} // namespace
PeriodicKSGridPlan plan_periodic_ks_with_quadrature(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicKSOptions& options,const ResourceBudget& budget) {
    return plan_with_grid<PeriodicKSGridPlan>(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,plan_periodic_ks_from_cutoffs);
}
PeriodicKSResult run_periodic_ks_with_quadrature(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicKSOptions& options,const ResourceBudget& budget) {
    const auto p=plan_periodic_ks_with_quadrature(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget);
    return run_with_grid(p,basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,run_periodic_ks_from_cutoffs);
}
PeriodicGlobalHybridKSGridPlan plan_periodic_global_hybrid_ks_with_quadrature(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    return plan_with_grid<PeriodicGlobalHybridKSGridPlan>(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,
        plan_periodic_global_hybrid_ks_from_cutoffs);
}
PeriodicKSResult run_periodic_global_hybrid_ks_with_quadrature(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto p=plan_periodic_global_hybrid_ks_with_quadrature(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget);
    return run_with_grid(p,basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,run_periodic_global_hybrid_ks_from_cutoffs);
}
PeriodicRangeSeparatedKSGridPlan plan_periodic_range_separated_ks_with_quadrature(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    return plan_with_grid<PeriodicRangeSeparatedKSGridPlan>(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,
        plan_periodic_range_separated_ks_from_cutoffs);
}
PeriodicKSResult run_periodic_range_separated_ks_with_quadrature(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto p=plan_periodic_range_separated_ks_with_quadrature(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget);
    return run_with_grid(p,basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,run_periodic_range_separated_ks_from_cutoffs);
}
PeriodicKSGridPlan plan_periodic_ri_ks_with_quadrature(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRIKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_ks_from_cutoffs(b,auxiliary,args...);};
    return plan_with_grid<PeriodicKSGridPlan>(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,planner);
}
PeriodicKSResult run_periodic_ri_ks_with_quadrature(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRIKSOptions& options,const ResourceBudget& budget) {
    const auto p=plan_periodic_ri_ks_with_quadrature(basis,auxiliary,system,mesh,f,cutoff,xc_radius,grid_options,options,budget);
    const auto runner=[&](const BasisSet& b,auto&&... args){return run_periodic_ri_ks_from_cutoffs(b,auxiliary,args...);};
    return run_with_grid(p,basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,runner);
}
PeriodicRIGlobalHybridKSGridPlan plan_periodic_ri_global_hybrid_ks_with_quadrature(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRIGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_global_hybrid_ks_from_cutoffs(b,auxiliary,args...);};
    return plan_with_grid<PeriodicRIGlobalHybridKSGridPlan>(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,planner);
}
PeriodicKSResult run_periodic_ri_global_hybrid_ks_with_quadrature(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRIGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto p=plan_periodic_ri_global_hybrid_ks_with_quadrature(basis,auxiliary,system,mesh,f,cutoff,xc_radius,grid_options,options,budget);
    const auto runner=[&](const BasisSet& b,auto&&... args){return run_periodic_ri_global_hybrid_ks_from_cutoffs(b,auxiliary,args...);};
    return run_with_grid(p,basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,runner);
}
PeriodicRIRangeSeparatedKSGridPlan plan_periodic_ri_range_separated_ks_with_quadrature(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRIRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_range_separated_ks_from_cutoffs(b,auxiliary,args...);};
    return plan_with_grid<PeriodicRIRangeSeparatedKSGridPlan>(basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,planner);
}
PeriodicKSResult run_periodic_ri_range_separated_ks_with_quadrature(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const PeriodicQuadratureOptions& grid_options,const PeriodicRIRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto p=plan_periodic_ri_range_separated_ks_with_quadrature(basis,auxiliary,system,mesh,f,cutoff,xc_radius,grid_options,options,budget);
    const auto runner=[&](const BasisSet& b,auto&&... args){return run_periodic_ri_range_separated_ks_from_cutoffs(b,auxiliary,args...);};
    return run_with_grid(p,basis,system,mesh,f,cutoff,xc_radius,grid_options,options,budget,runner);
}
} // namespace vibeqc::manta
