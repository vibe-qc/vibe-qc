#pragma once
#include "vibeqc/bounded_periodic_uhf_solver.hpp"

namespace vibeqc::detail {
// Private shared iteration kernels. A non-null scalar is overwritten by the
// joint provider on EVERY F[D] call and supplies E_interaction[D] per cell.
// It excludes Tr(D H), nuclear energy and any mesh replication factor.
struct BoundedPeriodicSCFKernel {
    static BoundedPeriodicRHFResult rhf(const BoundedPeriodicRHFInput&,
        const BoundedPeriodicRHFTwoElectronProvider&,const BoundedPeriodicRHFOptions&,
        const BoundedPeriodicRHFCaps&,BoundedPeriodicRHFProgressCallback,void*,const double*);
    static BoundedPeriodicUHFResult uhf(const BoundedPeriodicUHFInput&,
        const BoundedPeriodicUHFTwoElectronProvider&,const BoundedPeriodicUHFOptions&,
        const BoundedPeriodicUHFCaps&,BoundedPeriodicUHFProgressCallback,void*,const double*);
};
} // namespace vibeqc::detail
