// Unified molecular semiempirical dispatch over existing native kernels.

#pragma once

#include <cstddef>

#include "vibeqc/semiempirical/core/parameter_identity.hpp"

namespace vibeqc {

class Molecule;

namespace semiempirical {

class SemiempiricalParameters;

namespace xtb {
class GFN2ParameterSet;
}

namespace nddo {
class PM6ParameterSet;
class OMxParameterSet;
}

namespace indo {
struct MsindoParameterSet;
}

enum class SemiempiricalNativeMethod {
    DFTB0,
    SCCDFTB,
    GFN2XTB,
    PM6,
    OM1,
    OM2,
    OM3,
    MSINDO,
    MSINDONDDO,
};

enum class SemiempiricalNativeSpin {
    ClosedShell,
    Unrestricted,
};

struct SemiempiricalNativeRoute {
    SemiempiricalNativeMethod method = SemiempiricalNativeMethod::DFTB0;
    SemiempiricalNativeSpin spin = SemiempiricalNativeSpin::ClosedShell;
    int max_iter = 100;
    double conv_tol = 1.0e-7;
    double charge_mixing = 0.2;
};

struct SemiempiricalNativeResult : ParameterIdentifiedResult {
    SemiempiricalNativeRoute route;
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    double e_core = 0.0;
    double e_scc = 0.0;
    double binding_energy = 0.0;
    int n_basis = 0;
    int n_iter = 0;
    bool converged = false;
    std::size_t retained_result_bytes = 0;
    std::size_t workspace_bytes = 0;
    bool memory_counters_complete = false;
};

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const SemiempiricalParameters& parameters);

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const xtb::GFN2ParameterSet& parameters);

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const nddo::PM6ParameterSet& parameters);

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const nddo::OMxParameterSet& parameters);

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const indo::MsindoParameterSet& parameters);

}  // namespace semiempirical
}  // namespace vibeqc
