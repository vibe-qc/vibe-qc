#include "vibeqc/manta_periodic_xc_refinement.hpp"
#include "manta_periodic_xc_source.hpp"
#include "manta_periodic_xc_component.hpp"

namespace vibeqc::manta {
namespace {
using namespace admission;
using Kind=periodic_xc_component::Kind;
Kind hybrid_kind(const Functional& f) {
    // Inspect the complete descriptor even if aggregate CAM metadata cancels.
    for(const auto& component:f.exchange_components())
        if(std::get<0>(component)!="coulomb") return Kind::RangeSeparated;
    return Kind::GlobalHybrid;
}
ResourceBudget live_budget(const ResourceBudget& b,Count live) {
    auto out=b;out.other_live_bytes=add(b.other_live_bytes,live);return out;
}
void caps(const PeriodicXCRefinementPlan& p,const ResourceBudget& b) {
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA XC refinement memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA XC refinement shell-call budget exceeded");
    cap(p.primitive_points,b.primitive_quartets,"MANTA XC refinement primitive-point budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA XC refinement scalar-work budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA XC refinement kernel-work budget exceeded");
}
PeriodicXCRefinementPlan plan_refinement(const BasisSet& basis,const PeriodicSystem& system,
    const Functional& f,Count images,Count momenta,const std::vector<PeriodicQuadratureOptions>& grids,
    Count reference,Count batch,const ResourceBudget& budget,Kind kind) {
    using namespace admission;
    if(grids.size()<2 || grids.size()>32 || reference>=grids.size())
        throw std::invalid_argument("MANTA XC refinement requires 2..32 grids and a valid reference index");
    PeriodicXCRefinementPlan p;p.spins=f.spin();p.momenta=momenta;p.n_basis=basis.nbasis();p.reference_index=reference;
    p.semilocal_component_only=kind!=Kind::Pure;
    const Count e=mul(mul(p.spins,momenta),square(p.n_basis));
    // Binding input copies, retained reference potential plus export/receipt, bounded
    // scalar exports and plan/receipt metadata, including up to two canonical
    // exchange descriptors per functional component. Candidate output and the live
    // shared Grid plus explicit points/weights are in their child plans.
    p.driver_live_bytes=add(65536,add(mul(64,e),add(mul(256,add(add(images,momenta),basis.nshells())),
        mul(grids.size(),add(16384,add(mul(32,mul(p.spins,momenta)),add(mul(4,f.name().size()),mul(4096,f.components().size()))))))));
    cap(add(budget.other_live_bytes,p.driver_live_bytes),budget.memory_bytes,"MANTA XC refinement metadata memory budget exceeded");
    auto unlimited=live_budget(budget,p.driver_live_bytes);
    unlimited.memory_bytes=unlimited.shell_calls=unlimited.primitive_quartets=unlimited.scalar_terms=unlimited.kernel_terms=std::numeric_limits<Count>::max();
    const Count comparisons=mul(grids.size(),add(mul(32,e),mul(128,add(square(images),basis.nshells()))));
    p.scalar_terms=comparisons;p.kernel_terms=mul(32,comparisons);
    for(const auto& grid:grids) {
        auto q=plan_periodic_quadrature(system,grid,unlimited);
        auto b=live_budget(unlimited,q.output_bytes);
        auto x=periodic_xc_component::plan(basis,f,images,momenta,q.grid_points,batch,b,kind);
        p.modeled_peak_bytes=std::max(p.modeled_peak_bytes,std::max(q.modeled_peak_bytes,x.modeled_peak_bytes));
        p.shell_calls=add(p.shell_calls,x.shell_calls);p.primitive_points=add(p.primitive_points,x.primitive_points);
        p.scalar_terms=add(p.scalar_terms,add(q.scalar_terms,x.scalar_terms));
        p.kernel_terms=add(p.kernel_terms,add(q.kernel_terms,x.kernel_terms));
        p.quadrature.push_back(q);p.xc.push_back(x);
    }
    caps(p,budget);return p;
}
PeriodicXCRefinementResult assess_refinement(const BasisSet& basis,const PeriodicSystem& system,
    const Functional& f,const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,const std::vector<double>& weights,
    const std::vector<std::complex<double>>& d,const std::vector<PeriodicQuadratureOptions>& grids,
    Count reference,Count batch,const ResourceBudget& budget,Kind kind) {
    using namespace periodic_xc_detail;
    PeriodicXCRefinementResult out;
    out.plan=plan_refinement(basis,system,f,images.rows(),k.rows(),grids,reference,batch,budget,kind);
    if(out.plan.semilocal_component_only)
        out.comparison="frozen-density;semilocal-component-only;explicit-reference-grid;not-an-error-bound";
    const auto& p=out.plan;const Count n=p.n_basis,blocks=mul(p.spins,p.momenta);
    // Validate the frozen numerical inputs before any grid generation. Empty
    // point arrays here identify the unchanged basis/image/spin/k state only.
    PeriodicLocalExchangePlan validation=p.xc.front();validation.grid_points=0;
    PeriodicLocalExchangeReceipt common;
    initialize(basis,system.lattice,images,k,weights,Eigen::MatrixXd(0,3),{},d,p.spins,validation,common,
               "manta.periodic-xc-refinement.frozen.v1");
    out.density_identity=common.density_identity;
    out.steps.resize(grids.size());
    auto evaluate=[&](Count index) {
        const auto q=build_periodic_quadrature(system,grids[index],live_budget(budget,p.driver_live_bytes));
        const Eigen::MatrixXd points=q.grid.points;
        const std::vector<double> qw(q.grid.weights.data(),q.grid.weights.data()+q.grid.weights.size());
        auto r=periodic_xc_component::compute(basis,f,system.lattice,images,k,weights,points,qw,d,batch,
            live_budget(budget,add(p.driver_live_bytes,p.quadrature[index].output_bytes)),kind);
        auto& step=out.steps[index];step.options=grids[index];step.quadrature=q.receipt;step.xc=r.receipt;
        step.potential_max_abs.assign(blocks,0.);step.potential_frobenius.assign(blocks,0.);
        step.population_differences.assign(p.spins,0.);
        out.shell_calls=add(out.shell_calls,r.receipt.shell_calls);
        out.primitive_points=add(out.primitive_points,r.receipt.primitive_points);
        return r;
    };
    auto target=evaluate(reference);
    Digest source;source.text("manta.periodic-xc-refinement.source.v1");source.text(common.source_identity);
    source.text(target.receipt.functional);source.text(target.receipt.backend_version);source.word(reference);
    for(Count index=0;index<grids.size();++index) {
        if(index!=reference) {
            const auto r=evaluate(index);auto& step=out.steps[index];
            step.energy_difference=r.receipt.energy-target.receipt.energy;require_finite(step.energy_difference);
            for(Count spin=0;spin<p.spins;++spin) {
                step.population_differences[spin]=r.receipt.electrons_grid[spin]-target.receipt.electrons_grid[spin];
                require_finite(step.population_differences[spin]);
            }
            for(Count block=0;block<blocks;++block) {
                double maximum=0.,norm=0.;
                for(Count i=0;i<square(n);++i) {
                    const auto delta=r.values[block*square(n)+i]-target.values[block*square(n)+i];finite(delta);
                    const double absolute=std::abs(delta);maximum=std::max(maximum,absolute);norm=std::hypot(norm,absolute);
                }
                require_finite(norm);step.potential_max_abs[block]=maximum;step.potential_frobenius[block]=norm;
            }
        }
        source.text(out.steps[index].quadrature.source_identity);source.text(out.steps[index].xc.source_identity);
    }
    out.source_identity=source.hash.finish_hex();out.reference_values=std::move(target.values);return out;
}
} // namespace

PeriodicXCRefinementPlan plan_periodic_xc_refinement(const BasisSet& basis,const PeriodicSystem& system,
    const Functional& f,Count images,Count momenta,const std::vector<PeriodicQuadratureOptions>& grids,
    Count reference,Count batch,const ResourceBudget& budget) {
    return plan_refinement(basis,system,f,images,momenta,grids,reference,batch,budget,Kind::Pure);
}
PeriodicXCRefinementResult assess_periodic_xc_refinement(const BasisSet& basis,const PeriodicSystem& system,
    const Functional& f,const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,const std::vector<double>& weights,
    const std::vector<std::complex<double>>& d,const std::vector<PeriodicQuadratureOptions>& grids,
    Count reference,Count batch,const ResourceBudget& budget) {
    return assess_refinement(basis,system,f,images,k,weights,d,grids,reference,batch,budget,Kind::Pure);
}
PeriodicXCRefinementPlan plan_periodic_hybrid_xc_refinement(const BasisSet& basis,const PeriodicSystem& system,
    const Functional& f,Count images,Count momenta,const std::vector<PeriodicQuadratureOptions>& grids,
    Count reference,Count batch,const ResourceBudget& budget) {
    return plan_refinement(basis,system,f,images,momenta,grids,reference,batch,budget,hybrid_kind(f));
}
PeriodicXCRefinementResult assess_periodic_hybrid_xc_refinement(const BasisSet& basis,const PeriodicSystem& system,
    const Functional& f,const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,const std::vector<double>& weights,
    const std::vector<std::complex<double>>& d,const std::vector<PeriodicQuadratureOptions>& grids,
    Count reference,Count batch,const ResourceBudget& budget) {
    return assess_refinement(basis,system,f,images,k,weights,d,grids,reference,batch,budget,hybrid_kind(f));
}
} // namespace vibeqc::manta
