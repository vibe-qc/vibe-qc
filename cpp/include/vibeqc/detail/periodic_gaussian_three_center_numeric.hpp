#pragma once

// Private shared numerical contraction. Origin wrappers authenticate every
// input, census images and admit complete caller/borrowed lifetimes first.
#include "vibeqc/detail/periodic_gaussian_metric_numeric.hpp"
#include "vibeqc/periodic.hpp"

namespace vibeqc {
namespace detail {

struct ThreeCenterNumericalSelection {
    std::uint64_t ao_pair_begin = 0, ao_pair_count = 0;
    std::uint64_t auxiliary_begin = 0, auxiliary_count = 0;
};
struct ThreeCenterNumericalResult {
    std::vector<std::complex<double>> values;
    std::uint64_t visited_reciprocal_count = 0;
};
ThreeCenterNumericalResult contract_three_center_numeric(
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicSystem& system,
    const Eigen::Vector3d& canonical_ket_cartesian,
    const ThreeCenterNumericalSelection& selection,
    const std::vector<std::complex<double>>& whitener,
    std::uint64_t accepted_count, std::uint64_t reciprocal_capacity,
    double image_cutoff, std::uint64_t maximum_image_candidates,
    std::uint64_t expected_image_candidates, std::uint64_t expected_retained_images,
    std::uint64_t owned_numeric_byte_cap,
    MetricReciprocalVisitor visitor, const void* source_user);

} // namespace detail
} // namespace vibeqc
