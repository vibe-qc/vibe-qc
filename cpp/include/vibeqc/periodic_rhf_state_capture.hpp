#pragma once

/// \file periodic_rhf_state_capture.hpp
/// \brief Experimental, retained-payload-bounded physical multi-k RHF capture.
///
/// This request is deliberately separate from PeriodicSCFOptions.  Ordinary
/// periodic SCF neither allocates nor retains an all-k post-HF snapshot.  It is
/// also separate from PeriodicCorrelationResourcePlan: that planner describes
/// the future blocked/streamed correlation executor and must not be used to
/// authorize the current direct-truncated SCF implementation.
///
/// The producer first requires each raw direct-truncated F[D](k) to satisfy the
/// fixed mean-field-state Hermiticity tolerance.  The retained physical Fock is
/// then 0.5 * (F_raw + F_raw^H), the same small projection diagonalized by the
/// periodic SCF path.  A material finite-domain defect fails closed rather than
/// being hidden by that projection.

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vibeqc/periodic_mean_field_state.hpp"

namespace vibeqc {

/// Explicit request made only by an experimental correlation caller.
///
/// maximum_retained_numerical_payload_bytes bounds the exact numerical payload
/// owned by PeriodicRestrictedMeanFieldState.  It is not a total-RSS, SCF-peak,
/// scratch, or correlation-resource admission limit.
struct PeriodicRHFStateCaptureRequest {
    std::string calculation_identity;
    std::uint64_t maximum_retained_numerical_payload_bytes = 0;
    double minimum_band_gap_hartree = 0.0;
    /// One explicit binary mask for every full-mesh k point and every band.
    /// All-zero masks are valid but must still be supplied.
    std::vector<std::vector<std::uint8_t>> frozen_core_mask_per_k;
};

/// Result of the tensor-allocation-free request preflight.
struct PeriodicRHFStateCapturePreflight {
    std::uint64_t retained_numerical_payload_bytes = 0;
    std::uint64_t n_frozen_core = 0;
    std::uint64_t n_correlated_occupied = 0;
    std::uint64_t n_virtual = 0;
};

/// Validate the physical periodic dimension, exact regular mesh, explicit core
/// selection, and retained-state cap before periodic integral or all-k matrix
/// construction begins.
PeriodicRHFStateCapturePreflight preflight_periodic_rhf_state_capture(
    const PeriodicRHFStateCaptureRequest& request,
    int periodic_dimension,
    const std::array<int, 3>& mesh,
    const std::array<int, 3>& is_shift,
    const Eigen::Matrix3d& reciprocal_lattice,
    const std::vector<Eigen::Vector3d>& kpoints,
    const std::vector<double>& weights,
    bool symmetry_reduced_or_reconstructed,
    std::uint64_t n_basis,
    std::uint64_t electrons_per_cell);

/// Physical-density checks that cannot be reconstructed from the final state.
///
/// The density is spin summed, P = 2 C_occ C_occ^H.  Consequently its metric
/// idempotency relation is P S P = 2 P, not P S P = P.
struct PeriodicRHFPhysicalDensityDiagnostics {
    double commutator_frobenius = 0.0;
    double commutator_roundoff_allowance = 0.0;
    double metric_idempotency_frobenius = 0.0;
    double metric_idempotency_relative = 0.0;
};

PeriodicRHFPhysicalDensityDiagnostics periodic_rhf_physical_density_diagnostics(
    const PeriodicMeanFieldComplexMatrix& overlap,
    const PeriodicMeanFieldComplexMatrix& physical_fock,
    const PeriodicMeanFieldComplexMatrix& spin_summed_density);

struct PeriodicRHFDensityFixedPointDiagnostics {
    double frobenius = 0.0;
    double relative = 0.0;
};

/// P_physical versus 2 C_occ C_occ^H for an Aufbau RHF closure check.
PeriodicRHFDensityFixedPointDiagnostics
periodic_rhf_density_fixed_point_diagnostics(
    const PeriodicMeanFieldComplexMatrix& spin_summed_density,
    const PeriodicMeanFieldComplexMatrix& coefficients,
    std::uint64_t n_occupied);

}  // namespace vibeqc
