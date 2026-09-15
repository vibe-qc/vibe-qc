#pragma once

// Private arithmetic/seal helpers for the real-local certificate/provider.
// Outward bounds apply to the represented finite native input payloads,
// not to errors in AO integration, SCF or the infinite-image tail.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "periodic_correlation_local_factors_internal.hpp"
#include "vibeqc/periodic_correlation_density_factors.hpp"

namespace vibeqc {
namespace periodic_correlation_real_local_detail {

using Complex = std::complex<double>;
using Digest = periodic_correlation_local_detail::Digest;
inline constexpr char kFloatPolicy[] =
    "binary64;round-to-nearest;gradual-underflow;no-fast-math;correctly-rounded-basic-and-sqrt;outward-nextafter-v1";

inline void float_environment() {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
                  && std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53,
                  "real-local certificates require IEEE binary64");
#if defined(__FAST_MATH__)
    throw std::invalid_argument("real-local certificates do not permit fast-math arithmetic");
#endif
    if (std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("real-local certificates require round-to-nearest arithmetic");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.0, zero = 0.0;
    if (!(tiny > 0.0) || tiny * one != tiny || tiny + zero != tiny || std::fma(tiny, one, zero) != tiny
        || std::nextafter(0.0, 1.0) != tiny)
        throw std::invalid_argument("real-local certificates require gradual underflow");
}
inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("real-local count product overflows uint64");
    return a * b;
}
inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("real-local count sum overflows uint64");
    return a + b;
}
inline double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("real-local arithmetic or bound is non-finite");
    return value;
}
inline Complex finite(Complex value) { finite(value.real()); finite(value.imag()); return value; }
inline double up(double value) { return finite(std::nextafter(finite(value), std::numeric_limits<double>::infinity())); }
inline double down(double value) { return finite(std::nextafter(finite(value), -std::numeric_limits<double>::infinity())); }
inline double add_up(double a, double b) { return a == 0.0 ? b : b == 0.0 ? a : up(a + b); }
inline double mul_up(double a, double b) { return a == 0.0 || b == 0.0 ? 0.0 : up(a * b); }
inline double add_down(double a, double b) { return a == 0.0 ? b : b == 0.0 ? a : std::max(0.0, down(a + b)); }
inline double mul_down(double a, double b) { return a == 0.0 || b == 0.0 ? 0.0 : std::max(0.0, down(a * b)); }
inline double sqrt_up(double value) { return value == 0.0 ? 0.0 : up(std::sqrt(value)); }
inline double difference_up(double a, double b) { return a == b ? 0.0 : up(std::abs(finite(a - b))); }
inline void norm_lane(double& squared, double magnitude_upper) {
    squared = add_up(squared, mul_up(magnitude_upper, magnitude_upper));
}
inline void norm_value(double& squared, Complex value) {
    norm_lane(squared, std::abs(finite(value.real()))); norm_lane(squared, std::abs(finite(value.imag())));
}
inline void norm_difference(double& squared, Complex a, Complex b) {
    norm_lane(squared, difference_up(a.real(), b.real())); norm_lane(squared, difference_up(a.imag(), b.imag()));
}
inline double norm_up(Complex value) { double squared = 0.0; norm_value(squared, value); return sqrt_up(squared); }
inline double defect_up(Complex a, Complex b) { double squared = 0.0; norm_difference(squared, a, b); return sqrt_up(squared); }
inline double norm_down(Complex value) {
    const auto re = std::abs(value.real()), im = std::abs(value.imag());
    const auto squared = add_down(mul_down(re, re), mul_down(im, im));
    return squared == 0.0 ? 0.0 : std::max(0.0, down(std::sqrt(squared)));
}
inline void tolerance_pair(double absolute, double relative) {
    if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0.0 || relative < 0.0
        || absolute >= 1.0 || relative >= 1.0 || (absolute == 0.0 && relative == 0.0))
        throw std::invalid_argument("real-local tolerance pairs must be finite in [0,1) and not both zero");
}
inline bool within(Complex a, Complex b, double absolute, double relative) {
    const auto threshold_lower = add_down(absolute, mul_down(relative, std::max(norm_down(a), norm_down(b))));
    return defect_up(a, b) <= threshold_lower;
}
inline std::uint64_t negative_cell(std::uint64_t cell, const std::array<int, 3>& mesh) {
    const auto z = cell % mesh[2], y = (cell / mesh[2]) % mesh[1], x = cell / (std::uint64_t(mesh[1]) * mesh[2]);
    return ((x ? mesh[0] - x : 0) * mesh[1] + (y ? mesh[1] - y : 0)) * mesh[2] + (z ? mesh[2] - z : 0);
}
inline void extent(std::uint64_t bytes) {
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uint64_t>::max() / 8 - 16384)
        throw std::length_error("real-local payload exceeds native or SHA extent");
}
inline void indices(const std::uint64_t* data, std::size_t accessible, std::size_t count,
                    const PeriodicRestrictedMeanFieldState& state) {
    const auto bytes = mul(16, count);
    if (!data || accessible != mul(2, count) || reinterpret_cast<std::uintptr_t>(data) % alignof(std::uint64_t))
        throw std::invalid_argument("real-local indices require an aligned uint64 [count,2] view");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(data))
        throw std::overflow_error("real-local occupied pointer extent overflows");
    for (std::size_t i = 0; i < count; ++i) {
        if (data[2*i] >= state.n_correlated_occupied() || data[2*i+1] >= state.n_kpoints())
            throw std::out_of_range("real-local occupied index or modular cell is out of range");
        for (std::size_t j = 0; j < i; ++j)
            if (data[2*i] == data[2*j] && data[2*i+1] == data[2*j+1])
                throw std::invalid_argument("real-local occupied list contains a duplicate");
    }
}
inline std::string local_basis_identity(const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationWannier& wannier, const std::string& gauge, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const std::uint64_t* labels, std::uint64_t occupied,
    const PeriodicCorrelationVirtualBlockSelection& v) {
    Digest selection("vibeqc.periodic.correlation.local-orbital-factors.selection");
    selection.u64(occupied);
    for (std::uint64_t i = 0; i < mul(2, occupied); ++i) selection.u64(labels[i]);
    selection.u64(v.begin); selection.u64(v.count); selection.u64(v.translation_cell);
    Digest basis("vibeqc.periodic.correlation.local-orbital-factors.basis");
    basis.string(reference.state().state_identity_sha256()); basis.string(reference.state().calculation_identity());
    basis.string(reference.dimensions().allocation_identity); basis.string(selection.finish());
    basis.string(wannier.wannier_identity_sha256()); basis.string(gauge);
    basis.string(domain.pao_domain_identity_sha256()); basis.string(space.pao_space_identity_sha256());
    return basis.finish();
}
inline void accumulate(Complex term, Complex& sum, Complex& correction) {
    finite(term); const auto next = finite(sum + term);
    const auto re = std::abs(sum.real()) >= std::abs(term.real())
        ? (sum.real() - next.real()) + term.real() : (term.real() - next.real()) + sum.real();
    const auto im = std::abs(sum.imag()) >= std::abs(term.imag())
        ? (sum.imag() - next.imag()) + term.imag() : (term.imag() - next.imag()) + sum.imag();
    correction = finite(correction + Complex(re, im)); sum = next;
}
struct Sum {
    Complex sum{}, correction{};
    void include(Complex value) { accumulate(value, sum, correction); }
    Complex value() const { return finite(sum + correction); }
};

}  // namespace periodic_correlation_real_local_detail
}  // namespace vibeqc
