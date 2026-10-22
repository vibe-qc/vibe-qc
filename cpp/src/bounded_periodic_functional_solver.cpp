#include "vibeqc/bounded_periodic_functional_solver.hpp"
#include "bounded_periodic_scf_kernel.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a)
        throw std::overflow_error("bounded periodic functional resource count overflow");
    return a+b;
}
struct Joint {
    const BoundedPeriodicFunctionalProvider& provider;
    double energy=std::numeric_limits<double>::quiet_NaN();
    static void evaluate(const std::complex<double>* d,std::size_t dn,
        std::complex<double>* g,std::size_t gn,void* pointer) {
        auto& self=*static_cast<Joint*>(pointer);
        self.energy=std::numeric_limits<double>::quiet_NaN();
        self.provider.evaluate(d,dn,g,gn,&self.energy,self.provider.context);
        if(!std::isfinite(self.energy))
            throw std::invalid_argument("bounded periodic functional callback did not write a finite interaction energy");
    }
};
BoundedPeriodicRHFTwoElectronProvider adapter(const BoundedPeriodicFunctionalProvider& p,Joint* context=nullptr) {
    if(!p.evaluate || !p.maximum_work_units_per_call)
        throw std::invalid_argument("bounded periodic functional requires a bounded joint energy/potential callback");
    return {Joint::evaluate,context,add(p.retained_numerical_bytes,sizeof(double)),
        p.peak_workspace_numerical_bytes,add(p.maximum_work_units_per_call,32)};
}
} // namespace
BoundedPeriodicRHFMemoryPlan plan_bounded_periodic_rks_solver(
    std::uint64_t k,std::uint64_t n,std::uint64_t electrons,std::uint64_t rank,
    const BoundedPeriodicFunctionalProvider& p,const BoundedPeriodicRHFOptions& o,std::uint64_t other) {
    return plan_bounded_periodic_rhf_solver(k,n,electrons,rank,adapter(p),o,other);
}
BoundedPeriodicUHFMemoryPlan plan_bounded_periodic_uks_solver(
    std::uint64_t k,std::uint64_t n,std::uint64_t alpha,std::uint64_t beta,std::uint64_t rank,
    const BoundedPeriodicFunctionalProvider& p,const BoundedPeriodicUHFOptions& o,std::uint64_t other) {
    return plan_bounded_periodic_uhf_solver(k,n,alpha,beta,rank,adapter(p),o,other);
}
BoundedPeriodicRHFResult solve_bounded_periodic_rks(const BoundedPeriodicRHFInput& input,
    const BoundedPeriodicFunctionalProvider& p,const BoundedPeriodicRHFOptions& o,const BoundedPeriodicRHFCaps& caps,
    BoundedPeriodicRHFProgressCallback progress,void* context) {
    Joint joint{p}; const auto provider=adapter(p,&joint);
    return detail::BoundedPeriodicSCFKernel::rhf(input,provider,o,caps,progress,context,&joint.energy);
}
BoundedPeriodicUHFResult solve_bounded_periodic_uks(const BoundedPeriodicUHFInput& input,
    const BoundedPeriodicFunctionalProvider& p,const BoundedPeriodicUHFOptions& o,const BoundedPeriodicUHFCaps& caps,
    BoundedPeriodicUHFProgressCallback progress,void* context) {
    Joint joint{p}; const auto provider=adapter(p,&joint);
    return detail::BoundedPeriodicSCFKernel::uhf(input,provider,o,caps,progress,context,&joint.energy);
}
} // namespace vibeqc
