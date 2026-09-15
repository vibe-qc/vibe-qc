#pragma once

// Private numerical seam shared by state-origin v1 and density-independent
// sources. No ownership, provenance, resource admission or source claim is
// supplied by these plain structs. Only authenticated factories may publish
// a source. Implementations remain with the existing fixed-FMA/interval
// enumerator; do not create a second numerical predicate here.

#include "vibeqc/periodic_correlation_reciprocal_metric.hpp"

namespace vibeqc {
namespace detail {

struct PeriodicReciprocalNumericGeometry {
    Eigen::Matrix3d reciprocal_lattice = Eigen::Matrix3d::Zero();
    Eigen::Vector3d q_fractional = Eigen::Vector3d::Zero();
    std::array<std::int64_t, 3> lower_bounds = {0, 0, 0};
    std::array<std::int64_t, 3> upper_bounds = {0, 0, 0};
    double cutoff_squared = 0.0;
    double radial_limit = 0.0;
    double radial_limit_squared = 0.0;
    double cell_volume = 0.0;
    bool exact_gamma = false;
};

struct PeriodicReciprocalNumericPlan {
    PeriodicReciprocalNumericGeometry geometry;
    PeriodicCorrelationReciprocalMetricEnumerationPlan enumeration;
    Eigen::Vector3d q_cartesian = Eigen::Vector3d::Zero();
    double maximum_radius = 0.0;
    double boundary_tolerance = 0.0;
};

PeriodicReciprocalNumericPlan prepare_periodic_reciprocal_numeric_source(
    const Eigen::Matrix3d& reciprocal_lattice,
    const Eigen::Vector3d& q_fractional, double reciprocal_energy_cutoff,
    bool exact_gamma, std::uint64_t candidate_count_cap);

using PeriodicReciprocalRecordCallback = void (*)(
    const std::array<std::int64_t, 3>& integer_label,
    const std::array<double, 5>& lanes, void* user);

// No allocation or wrapper ownership. Caller must already have admitted the
// complete Cartesian traversal. References expire when the callback returns.
std::uint64_t visit_periodic_reciprocal_numeric_source(
    const PeriodicReciprocalNumericGeometry& geometry,
    PeriodicReciprocalRecordCallback callback, void* user);

// Caller proves opposite centered addresses, identical lattice/volume and
// equal accepted counts. This audits the exact injective label map against
// the opposite predicate and bitwise p/p2/w; equal counts then prove closure.
std::uint64_t require_periodic_reciprocal_numeric_conjugacy(
    const PeriodicReciprocalNumericGeometry& source,
    const PeriodicReciprocalNumericGeometry& opposite,
    const std::array<std::int64_t, 3>& partner_shift,
    std::uint64_t expected_accepted_count);

} // namespace detail
} // namespace vibeqc
