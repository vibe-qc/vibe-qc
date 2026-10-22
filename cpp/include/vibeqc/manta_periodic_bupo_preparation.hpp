#pragma once
#include "vibeqc/manta_periodic_bupo_domains.hpp"
#include <mutex>

namespace vibeqc::manta {
struct PeriodicBupoSourceOptions {
    PeriodicPairOptions pairs;
    BubbleOptions bubbles;
    PeriodicBupoOptions bupo;
    Count pair_cube_radius=1,real_cube_radius=4,reciprocal_cube_radius=10,near_cube_radius=0;
    double eta=.6;
    PeriodicBupoSourceOptions() { bupo.bound_infinite_domain=true;bupo.bound_pair_images=true; }
};
using PeriodicBupoHFSourceOptions = PeriodicBupoSourceOptions;
struct PeriodicBupoSourcePreparationPlan {
    PeriodicPairPlan pairs;
    PeriodicBubblePlan bubbles;
    PeriodicEwaldPlan ewald;
    PeriodicBupoPlan bupo;
    Count generator_live_bytes=0,generator_scalar_terms=0,generator_kernel_terms=0;
    Count source_live_bytes=0,source_shell_calls=0,source_primitive_pairs=0,source_scalar_terms=0,source_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicBupoHFPreparationPlan : PeriodicBupoSourcePreparationPlan {
    PeriodicBupoHFDomainPlan exchange_domains;
};
// Shared electronic-source preparation, independent of nuclear, exchange or
// SCF ownership. Every original native producer is admitted cumulatively.
// The returned immutable source may be retained directly by an RI fitting
// certificate. remaining_budget includes all retained source reservations
// and subtracts preparation work; the consumer must plan its own workspace.
class PreparedPeriodicBupoSource {
public:
    PreparedPeriodicBupoSource(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
        const PeriodicBupoSourceOptions&,Count reciprocal_batch,const ResourceBudget&);
    const PeriodicBupoSourcePreparationPlan& plan() const { return plan_; }
    const PeriodicBupoSourcePreparationReceipt& receipt() const { return receipt_; }
    const PeriodicBupoSourceOptions& source_options() const { return source_options_; }
    std::shared_ptr<PeriodicBupoSource> source() const { return electronic_; }
    ResourceBudget remaining_budget() const { return runtime_budget_; }
private:
    PeriodicBupoSourcePreparationPlan plan_;
    PeriodicBupoSourcePreparationReceipt receipt_;
    PeriodicBupoSourceOptions source_options_;
    ResourceBudget runtime_budget_;
    std::shared_ptr<PeriodicBupoSource> electronic_;
};
// Preparation executes source construction with cumulative staged admission;
// it is not a count-only dry run. No electronic density or SCF state is held.
// All physical inputs and policies are copied. One run lease per handle
// prevents concurrent use from exceeding the admitted workspace ownership.
class PreparedPeriodicBupoHF {
public:
    PreparedPeriodicBupoHF(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,double reciprocal_energy_cutoff,
        const PeriodicBupoHFSourceOptions&,const PeriodicBupoHFOptions&,const ResourceBudget&);
    const PeriodicBupoHFPreparationPlan& plan() const { return plan_; }
    const PeriodicBupoHFPreparationReceipt& receipt() const { return receipt_; }
    const PeriodicBupoHFOptions& options() const { return options_; }
    const PeriodicBupoHFSourceOptions& source_options() const { return source_options_; }
    PeriodicBupoHFResult run() const;
private:
    PeriodicBupoHFPreparationPlan plan_;
    PeriodicBupoHFPreparationReceipt receipt_;
    PeriodicSystem system_;
    RegularKMesh mesh_;
    PeriodicBupoHFOptions options_;
    PeriodicBupoHFSourceOptions source_options_;
    ResourceBudget runtime_budget_;
    std::shared_ptr<const PeriodicBupoSource> electronic_;
    PeriodicBupoHFExchangeDomains domains_;
    mutable std::mutex run_mutex_;
};
} // namespace vibeqc::manta
