// Pre-computed far-field Fock kernel builder.
//
// Groups far-field quartets by (R_sep, L_order) to reuse interaction
// tensors, then builds the effective K matrices via batched matmul.
// This replaces the Python build_far_field_fock_kernel setup function
// which dominates geometry-setup time for large systems (200k+ quartets).
//
// The grouping and cache construction are implementation prototypes.
// Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the expansion source but does
// not derive this kernel builder.

#pragma once

#include "bipole_contractor.hpp"
#include "bipole_far_field_kernel.hpp"
#include "bipole_multipole.hpp"

#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace vibeqc {

// Build the far-field Fock kernel from the moment buffer and dispatch.
//
// Groups quartets by (rounded_R_sep, L_order), computes the interaction
// tensor once per group, and builds K = M_bra @ T @ M_ket^T matrices
// via the batched C++ builder.
//
// Returns a vector of FarFieldFockKernelEntry ready for
// apply_far_field_fock_kernel.
std::vector<FarFieldFockKernelEntry> build_far_field_fock_kernel_cpp(
    const BipoleMomentBufferCpp& moment_buffer,
    const std::vector<BipoleQuartetEntryCpp>& dispatch,
    double ewald_omega);

}  // namespace vibeqc
