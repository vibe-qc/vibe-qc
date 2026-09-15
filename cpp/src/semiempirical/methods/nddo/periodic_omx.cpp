// Fail-closed compatibility entry point for the retired periodic OMx prototype.

#include "vibeqc/semiempirical/methods/nddo/periodic_omx.hpp"

#include <stdexcept>

namespace vibeqc {
namespace semiempirical {
namespace nddo {

PeriodicOMxResult run_omx_gamma(
    const PeriodicSystem& system,
    const OMxParameterSet& params,
    const PeriodicOMxOptions& opts) {
    (void)system;
    (void)params;
    (void)opts;
    throw std::logic_error(
        "Bloch-periodic OMx is gated: the retired prototype did not "
        "implement the published image-resolved ORT, ECP, and penetration "
        "Hamiltonian; use the explicitly topology-bound OMx-SECCM route");
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
