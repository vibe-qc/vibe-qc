#include "vibeqc/periodic_correlation_reciprocal_metric.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/detail/periodic_gaussian_metric_numeric.hpp"
#include "vibeqc/kmesh_address.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"

namespace vibeqc {

namespace {

constexpr std::array<int, 3> kGammaCentered = {0, 0, 0};
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kFourPi = 4.0 * kPi;
constexpr std::uint64_t kMaximumExactBinary64IntegerCount =
    9007199254740992ULL;  // 2^53
constexpr double kMaximumExactBinary64Integer =
    static_cast<double>(kMaximumExactBinary64IntegerCount);
constexpr char kProducerIdentityDomain[] =
    "vibeqc.periodic.correlation.reciprocal-metric.producer";
constexpr char kSourceIdentityDomain[] =
    "vibeqc.periodic.correlation.reciprocal-metric.source";
constexpr char kPayloadIdentityDomain[] =
    "vibeqc.periodic.correlation.reciprocal-metric.payload";
constexpr std::uint64_t kSha256MaximumMessageBytes =
    std::numeric_limits<std::uint64_t>::max() / 8U;

static_assert(std::numeric_limits<double>::is_iec559,
              "reciprocal metric source requires IEEE-754 binary64");
static_assert(std::numeric_limits<double>::radix == 2,
              "reciprocal metric source requires a radix-two double");
static_assert(std::numeric_limits<double>::digits == 53,
              "reciprocal metric source requires 53-bit binary64 precision");
static_assert(std::numeric_limits<double>::max_exponent == 1024,
              "reciprocal metric source requires binary64 exponent range");
static_assert(
    std::numeric_limits<double>::has_denorm == std::denorm_present,
    "reciprocal metric source requires gradual binary64 underflow");
static_assert(
    std::numeric_limits<double>::round_style == std::round_to_nearest,
    "reciprocal metric source requires round-to-nearest binary64");
static_assert(sizeof(double) == sizeof(std::uint64_t),
              "reciprocal metric source requires IEEE-754 binary64");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "reciprocal metric source requires size_t to fit uint64");
static_assert(
    kPeriodicCorrelationReciprocalMetricSourceManifestBytes == 64U,
    "reciprocal metric source manifest is one lowercase SHA-256 string");
static_assert(
    sizeof(kPayloadIdentityDomain) - 1U == 53U,
    "reciprocal metric payload domain changed without a contract update");
static_assert(
    sizeof(std::complex<double>) == 16U,
    "reciprocal metric byte accounting requires 16-byte complex128");

#if defined(__FAST_MATH__)
#error "reciprocal metric source certificate forbids fast math"
#endif

#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0
#error "reciprocal metric source certificate requires IEEE infinities"
#endif

#if FLT_EVAL_METHOD != 0
#error "reciprocal metric source certificate requires binary64 evaluation"
#endif

class CanonicalSourceHasher {
public:
    CanonicalSourceHasher(const char* domain, std::uint32_t version) {
        add_string(domain);
        add_u32(version);
    }

    void add_u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (24U - 8U * static_cast<unsigned>(index)));
        }
        add_bytes(encoded.data(), encoded.size());
    }

    void add_i32(std::int32_t value) {
        add_u32(static_cast<std::uint32_t>(value));
    }

    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (56U - 8U * static_cast<unsigned>(index)));
        }
        add_bytes(encoded.data(), encoded.size());
    }

    void add_i64(std::int64_t value) {
        add_u64(static_cast<std::uint64_t>(value));
    }

    void add_double(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "reciprocal metric source identity cannot encode a "
                "non-finite value");
        }
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(bits));
        add_u64(bits);
    }

    void add_string(const std::string& value) {
        add_u64(static_cast<std::uint64_t>(value.size()));
        add_bytes(
            reinterpret_cast<const std::uint8_t*>(value.data()),
            value.size());
    }

    void add_string(const char* value) {
        const std::size_t size = std::strlen(value);
        add_u64(static_cast<std::uint64_t>(size));
        add_bytes(reinterpret_cast<const std::uint8_t*>(value), size);
    }

    std::string finish_hex() { return hasher_.finish_hex(); }

    std::uint64_t wire_bytes() const noexcept { return wire_bytes_; }

private:
    void add_bytes(const std::uint8_t* data, std::size_t size) {
        const std::uint64_t count = static_cast<std::uint64_t>(size);
        if (count > kSha256MaximumMessageBytes - wire_bytes_) {
            throw std::length_error(
                "reciprocal metric canonical source exceeds SHA-256's "
                "less-than-2^64-bit message-length domain");
        }
        hasher_.update(data, size);
        wire_bytes_ += count;
    }

    detail::Sha256 hasher_;
    std::uint64_t wire_bytes_ = 0U;
};

bool is_lower_sha256(const std::string& value) {
    return value.size() == 64U
        && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9')
                   || (character >= 'a' && character <= 'f');
           });
}

void require_lower_sha256(const std::string& value, const char* label) {
    if (!is_lower_sha256(value)) {
        throw std::invalid_argument(
            std::string(label) + " must be a lowercase SHA-256");
    }
}

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            std::string("reciprocal metric source arithmetic overflow: ")
            + label);
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* label) {
    if (left != 0U
        && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            std::string("reciprocal metric source arithmetic overflow: ")
            + label);
    }
    return left * right;
}

void require_supported_binary64_environment() {
    if (std::fegetround() != FE_TONEAREST) {
        throw std::runtime_error(
            "reciprocal metric unsupported_floating_point_environment: "
            "round-to-nearest is required");
    }
    volatile double lambda = std::numeric_limits<double>::denorm_min();
    volatile double zero = 0.0;
    volatile double one = 1.0;
    volatile double twice = lambda + lambda;
    volatile double expected_twice = std::scalbn(lambda, 1);
    volatile double fma_identity = std::fma(lambda, one, zero);
    const double next_positive = std::nextafter(
        0.0, std::numeric_limits<double>::infinity());
    if (!(lambda > 0.0) || twice != expected_twice
        || fma_identity != lambda || next_positive != lambda) {
        throw std::runtime_error(
            "reciprocal metric unsupported_floating_point_environment: "
            "gradual binary64 underflow is required");
    }
}

struct OutwardInterval {
    double lower = 0.0;
    double upper = 0.0;
};

bool is_zero_interval(const OutwardInterval& value) noexcept {
    return value.lower == 0.0 && value.upper == 0.0;
}

bool is_point_interval(const OutwardInterval& value, double point) noexcept {
    return value.lower == point && value.upper == point;
}

OutwardInterval point_interval(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "reciprocal metric non_finite_interval input");
    }
    if (value == 0.0) value = 0.0;
    return {value, value};
}

OutwardInterval interval_around_rounded(double value) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "reciprocal metric non_finite_interval arithmetic");
    }
    const OutwardInterval result = {
        std::nextafter(value, -std::numeric_limits<double>::infinity()),
        std::nextafter(value, std::numeric_limits<double>::infinity()),
    };
    if (!std::isfinite(result.lower) || !std::isfinite(result.upper)) {
        throw std::overflow_error(
            "reciprocal metric non_finite_interval endpoint");
    }
    return result;
}

OutwardInterval negate_interval(const OutwardInterval& value) {
    return {-value.upper, -value.lower};
}

OutwardInterval scalar_add_interval(double left, double right) {
    if (left == 0.0) return point_interval(right);
    if (right == 0.0) return point_interval(left);
    if (left == -right) return point_interval(0.0);
    return interval_around_rounded(std::fma(1.0, right, left));
}

OutwardInterval scalar_subtract_interval(double left, double right) {
    if (right == 0.0) return point_interval(left);
    if (left == 0.0) return point_interval(-right);
    if (left == right) return point_interval(0.0);
    return interval_around_rounded(std::fma(-1.0, right, left));
}

OutwardInterval scalar_multiply_interval(double left, double right) {
    if (left == 0.0 || right == 0.0) return point_interval(0.0);
    if (left == 1.0) return point_interval(right);
    if (right == 1.0) return point_interval(left);
    if (left == -1.0) return point_interval(-right);
    if (right == -1.0) return point_interval(-left);
    return interval_around_rounded(std::fma(left, right, 0.0));
}

OutwardInterval scalar_divide_interval(double numerator,
                                        double denominator) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator)
        || denominator == 0.0) {
        throw std::invalid_argument(
            "reciprocal metric non_finite_interval division input");
    }
    if (numerator == 0.0) return point_interval(0.0);
    if (denominator == 1.0) return point_interval(numerator);
    if (denominator == -1.0) return point_interval(-numerator);
    volatile double quotient = numerator / denominator;
    return interval_around_rounded(quotient);
}

OutwardInterval add_intervals(const OutwardInterval& left,
                              const OutwardInterval& right) {
    if (is_zero_interval(left)) return right;
    if (is_zero_interval(right)) return left;
    const auto lower = scalar_add_interval(left.lower, right.lower);
    const auto upper = scalar_add_interval(left.upper, right.upper);
    return {lower.lower, upper.upper};
}

OutwardInterval subtract_intervals(const OutwardInterval& left,
                                   const OutwardInterval& right) {
    if (is_zero_interval(right)) return left;
    if (is_zero_interval(left)) return negate_interval(right);
    if (left.lower == left.upper && right.lower == right.upper
        && left.lower == right.lower) {
        return point_interval(0.0);
    }
    const auto lower = scalar_subtract_interval(left.lower, right.upper);
    const auto upper = scalar_subtract_interval(left.upper, right.lower);
    return {lower.lower, upper.upper};
}

OutwardInterval multiply_intervals(const OutwardInterval& left,
                                   const OutwardInterval& right) {
    if (is_zero_interval(left) || is_zero_interval(right)) {
        return point_interval(0.0);
    }
    if (is_point_interval(left, 1.0)) return right;
    if (is_point_interval(right, 1.0)) return left;
    if (is_point_interval(left, -1.0)) return negate_interval(right);
    if (is_point_interval(right, -1.0)) return negate_interval(left);

    const std::array<OutwardInterval, 4> products = {
        scalar_multiply_interval(left.lower, right.lower),
        scalar_multiply_interval(left.lower, right.upper),
        scalar_multiply_interval(left.upper, right.lower),
        scalar_multiply_interval(left.upper, right.upper),
    };
    double lower = products[0].lower;
    double upper = products[0].upper;
    for (std::size_t index = 1; index < products.size(); ++index) {
        lower = std::min(lower, products[index].lower);
        upper = std::max(upper, products[index].upper);
    }
    return {lower, upper};
}

OutwardInterval intersect_intervals(const OutwardInterval& left,
                                    const OutwardInterval& right) {
    const OutwardInterval intersection = {
        std::max(left.lower, right.lower),
        std::min(left.upper, right.upper),
    };
    if (intersection.lower > intersection.upper) {
        throw std::logic_error(
            "reciprocal metric determinant interval expansions do not "
            "intersect");
    }
    return intersection;
}

double absolute_interval_upper(const OutwardInterval& value) noexcept {
    return std::max(std::abs(value.lower), std::abs(value.upper));
}

double add_up(double left, double right) {
    return scalar_add_interval(left, right).upper;
}

double subtract_down(double left, double right) {
    return scalar_subtract_interval(left, right).lower;
}

double subtract_up(double left, double right) {
    return scalar_subtract_interval(left, right).upper;
}

double multiply_up(double left, double right) {
    if (left < 0.0 || right < 0.0) {
        throw std::logic_error(
            "reciprocal metric certificate expected nonnegative factors");
    }
    return scalar_multiply_interval(left, right).upper;
}

double divide_up(double numerator, double positive_denominator) {
    if (numerator < 0.0 || !(positive_denominator > 0.0)) {
        throw std::logic_error(
            "reciprocal metric certificate expected nonnegative division");
    }
    return scalar_divide_interval(numerator, positive_denominator).upper;
}

double square_root_up(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::overflow_error(
            "reciprocal metric certificate has a non-finite squared bound");
    }
    if (value == 0.0) return 0.0;
    const double root = std::sqrt(value);
    if (!std::isfinite(root)) {
        throw std::overflow_error(
            "reciprocal metric certificate square root is non-finite");
    }
    return std::nextafter(root, std::numeric_limits<double>::infinity());
}

Eigen::Vector3d fixed_binary64_matvec(const Eigen::Matrix3d& matrix,
                                      const Eigen::Vector3d& vector) {
    Eigen::Vector3d result;
    for (Eigen::Index row = 0; row < 3; ++row) {
        double value = std::fma(matrix(row, 2), vector[2], 0.0);
        value = std::fma(matrix(row, 1), vector[1], value);
        result[row] = std::fma(matrix(row, 0), vector[0], value);
    }
    return result;
}

double fixed_binary64_squared_norm(const Eigen::Vector3d& vector) {
    double value = std::fma(vector[2], vector[2], 0.0);
    value = std::fma(vector[1], vector[1], value);
    return std::fma(vector[0], vector[0], value);
}

double fixed_binary64_normalized_determinant(
    const Eigen::Matrix3d& matrix,
    double normalization_scale) {
    Eigen::Matrix3d normalized;
    for (Eigen::Index row = 0; row < 3; ++row) {
        for (Eigen::Index column = 0; column < 3; ++column) {
            volatile double quotient =
                matrix(row, column) / normalization_scale;
            normalized(row, column) = quotient;
        }
    }
    const auto difference_of_products = [](double first_left,
                                           double first_right,
                                           double second_left,
                                           double second_right) {
        const double second =
            std::fma(second_left, second_right, 0.0);
        return std::fma(first_left, first_right, -second);
    };
    const double cofactor_00 = difference_of_products(
        normalized(1, 1),
        normalized(2, 2),
        normalized(1, 2),
        normalized(2, 1));
    const double cofactor_01 = difference_of_products(
        normalized(1, 2),
        normalized(2, 0),
        normalized(1, 0),
        normalized(2, 2));
    const double cofactor_02 = difference_of_products(
        normalized(1, 0),
        normalized(2, 1),
        normalized(1, 1),
        normalized(2, 0));
    double determinant =
        std::fma(normalized(0, 2), cofactor_02, 0.0);
    determinant =
        std::fma(normalized(0, 1), cofactor_01, determinant);
    determinant =
        std::fma(normalized(0, 0), cofactor_00, determinant);
    if (!std::isfinite(determinant) || determinant == 0.0) {
        throw std::runtime_error(
            "reciprocal metric fixed determinant evaluation is unusable");
    }
    return determinant;
}

double fixed_binary64_cell_volume(
    const Eigen::Matrix3d& reciprocal_lattice,
    double normalization_scale,
    double certified_determinant_lower,
    double certified_determinant_upper) {
    const double normalized_determinant =
        fixed_binary64_normalized_determinant(
            reciprocal_lattice, normalization_scale);
    if ((certified_determinant_lower > 0.0
         && normalized_determinant <= 0.0)
        || (certified_determinant_upper < 0.0
            && normalized_determinant >= 0.0)) {
        throw std::runtime_error(
            "reciprocal metric fixed determinant sign disagrees with its "
            "outward certificate");
    }
    volatile double scaled_two_pi = kTwoPi / normalization_scale;
    const double scaled_two_pi_squared =
        std::fma(scaled_two_pi, scaled_two_pi, 0.0);
    const double scaled_two_pi_cubed =
        std::fma(scaled_two_pi_squared, scaled_two_pi, 0.0);
    volatile double cell_volume =
        scaled_two_pi_cubed / std::abs(normalized_determinant);
    if (!std::isfinite(cell_volume) || cell_volume <= 0.0) {
        throw std::overflow_error(
            "reciprocal metric fixed cell-volume evaluation is not finite "
            "and positive");
    }
    return cell_volume;
}

void require_upstream_consistency(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule) {
    if (!reference.state_handle() || !schedule.state_handle()) {
        throw std::invalid_argument(
            "reciprocal metric source requires live admitted-reference "
            "and schedule state handles");
    }
    if (reference.state_handle().get() != schedule.state_handle().get()) {
        throw std::invalid_argument(
            "reciprocal metric source reference and schedule do not share "
            "the same mean-field state object");
    }

    const auto& dimensions = reference.dimensions();
    const auto& plan = reference.plan();
    const auto& state = reference.state();
    if (reference.contract_version()
            != kPeriodicCorrelationAdmittedReferenceContractVersion
        || schedule.contract_version()
            != kPeriodicCorrelationFactorStreamContractVersion
        || state.contract_version()
            != kPeriodicRestrictedMeanFieldStateContractVersion
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission
            != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || plan.allocation_contract_version
            != dimensions.allocation_contract_version
        || !plan.census_identity.empty()
        || plan.pair_domain_census_complete
        || plan.triple_domain_census_complete) {
        throw std::logic_error(
            "reciprocal metric source received an inconsistent upstream "
            "contract");
    }
    if (dimensions.symmetry_reduction_requested
        || !dimensions.symmetry_mapping_identity.empty()
        || dimensions.symmetry_representative_count
            != dimensions.n_kpoints
        || dimensions.symmetry_weight_sum != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 requires an unreduced "
            "full mesh");
    }
    if (state.periodic_dimension() != dimensions.periodic_dimension) {
        throw std::logic_error(
            "reciprocal metric source periodic dimensionality disagrees "
            "between the state and admitted dimensions");
    }
    if (state.periodic_dimension() != 3) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 supports only physical "
            "periodic dimension three");
    }
    if (dimensions.is_shift != kGammaCentered
        || schedule.is_shift() != kGammaCentered
        || state.is_shift() != kGammaCentered) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 requires a "
            "Gamma-centered mesh");
    }
    if (dimensions.factor_q_block != 1U
        || dimensions.factor_k_bra_block != 1U
        || dimensions.factor_k_ket_block != 1U
        || dimensions.n_spin_channels != 1U
        || reference.budget().mpi_ranks != 1U) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 requires unit q/k "
            "blocks, one spin channel, and one MPI rank");
    }
    if (plan.calculation_identity != dimensions.calculation_identity
        || plan.allocation_identity != dimensions.allocation_identity
        || state.calculation_identity() != dimensions.calculation_identity
        || schedule.calculation_identity()
            != dimensions.calculation_identity
        || schedule.allocation_identity()
            != dimensions.allocation_identity
        || schedule.state_identity_sha256()
            != state.state_identity_sha256()
        || schedule.mesh() != dimensions.mesh
        || state.mesh() != dimensions.mesh
        || static_cast<std::uint64_t>(state.n_kpoints())
            != dimensions.n_kpoints
        || schedule.shape().n_kpoints != dimensions.n_kpoints
        || schedule.shape().n_basis != dimensions.n_basis
        || schedule.shape().n_auxiliary != dimensions.n_auxiliary) {
        throw std::logic_error(
            "reciprocal metric source upstream identities or dimensions "
            "disagree");
    }
    require_lower_sha256(
        state.state_identity_sha256(), "mean-field state identity");
    require_lower_sha256(
        dimensions.calculation_identity, "calculation identity");
    require_lower_sha256(
        dimensions.allocation_identity, "allocation identity");
    require_lower_sha256(
        schedule.schedule_identity_sha256(), "factor-stream identity");
}

void require_source_configuration(
    const PeriodicCorrelationFactorBuildConfig& config) {
    if (config.producer_mode
            != PeriodicCorrelationFactorProducerMode::
                FullCoulombAllReciprocalReference
        || config.short_range_policy
            != PeriodicCorrelationShortRangePolicy::DisabledAllReciprocal
        || config.short_range_real_space_cutoff != 0.0
        || config.range_separation_omega != 0.0) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 supports only the "
            "full-Coulomb all-reciprocal producer with short range "
            "disabled and zero range-separation parameters");
    }
    if (config.transfer_convention
        != PeriodicCorrelationTransferRepresentativeConvention::
            CenteredHalfOpenNegativeNyquist) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 requires centred "
            "[-1/2,1/2) transfers with negative even-mesh Nyquist");
    }
    if (config.q_concurrency != 1U) {
        throw std::invalid_argument(
            "reciprocal metric source contract v1 requires q concurrency "
            "one");
    }
    require_lower_sha256(
        config.auxiliary_basis_identity_sha256,
        "auxiliary basis identity");
    require_lower_sha256(config.producer_identity_sha256,
                         "producer identity");
    const std::string compiled_identity =
        periodic_correlation_reciprocal_metric_producer_identity_sha256();
    if (config.producer_identity_sha256 != compiled_identity) {
        throw std::invalid_argument(
            "factor-build producer identity does not name the compiled "
            "full-Coulomb reciprocal metric producer");
    }
    if (!std::isfinite(config.reciprocal_energy_cutoff)
        || config.reciprocal_energy_cutoff <= 0.0
        || config.reciprocal_energy_cutoff
            > std::numeric_limits<double>::max() / 2.0) {
        throw std::invalid_argument(
            "reciprocal energy cutoff must be finite, positive, and have "
            "a finite doubled value");
    }
}

std::pair<std::array<int, 3>, std::array<int, 3>> centered_transfer(
    const RegularKMesh& addressing,
    std::uint64_t q_index) {
    if (q_index >= static_cast<std::uint64_t>(addressing.size())) {
        throw std::out_of_range(
            "reciprocal metric source q index is outside the full mesh");
    }
    const auto q = addressing.transfer_address(
        static_cast<std::size_t>(q_index));
    std::array<int, 3> numerator{};
    std::array<int, 3> wrap{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const int modular = q.doubled[axis];
        const int divisions = addressing.mesh()[axis];
        if (modular >= divisions) {
            numerator[axis] = modular - 2 * divisions;
            wrap[axis] = 1;
        } else {
            numerator[axis] = modular;
            wrap[axis] = 0;
        }
    }
    return {numerator, wrap};
}

std::uint64_t axis_width(std::int64_t lower, std::int64_t upper);

OutwardInterval normalized_entry_interval(double value, double scale) {
    if (value == 0.0) return point_interval(0.0);
    if (value == scale) return point_interval(1.0);
    if (value == -scale) return point_interval(-1.0);
    return scalar_divide_interval(value, scale);
}

using IntervalMatrix3 =
    std::array<std::array<OutwardInterval, 3>, 3>;

IntervalMatrix3 cofactor_intervals(const IntervalMatrix3& a) {
    IntervalMatrix3 c{};
    c[0][0] = subtract_intervals(
        multiply_intervals(a[1][1], a[2][2]),
        multiply_intervals(a[1][2], a[2][1]));
    c[0][1] = subtract_intervals(
        multiply_intervals(a[1][2], a[2][0]),
        multiply_intervals(a[1][0], a[2][2]));
    c[0][2] = subtract_intervals(
        multiply_intervals(a[1][0], a[2][1]),
        multiply_intervals(a[1][1], a[2][0]));
    c[1][0] = subtract_intervals(
        multiply_intervals(a[0][2], a[2][1]),
        multiply_intervals(a[0][1], a[2][2]));
    c[1][1] = subtract_intervals(
        multiply_intervals(a[0][0], a[2][2]),
        multiply_intervals(a[0][2], a[2][0]));
    c[1][2] = subtract_intervals(
        multiply_intervals(a[0][1], a[2][0]),
        multiply_intervals(a[0][0], a[2][1]));
    c[2][0] = subtract_intervals(
        multiply_intervals(a[0][1], a[1][2]),
        multiply_intervals(a[0][2], a[1][1]));
    c[2][1] = subtract_intervals(
        multiply_intervals(a[0][2], a[1][0]),
        multiply_intervals(a[0][0], a[1][2]));
    c[2][2] = subtract_intervals(
        multiply_intervals(a[0][0], a[1][1]),
        multiply_intervals(a[0][1], a[1][0]));
    return c;
}

OutwardInterval sum_three_intervals(const OutwardInterval& first,
                                    const OutwardInterval& second,
                                    const OutwardInterval& third,
                                    bool right_associated) {
    if (right_associated) {
        return add_intervals(first, add_intervals(second, third));
    }
    return add_intervals(add_intervals(first, second), third);
}

OutwardInterval certified_determinant_interval(
    const IntervalMatrix3& a,
    const IntervalMatrix3& cofactors) {
    bool initialized = false;
    OutwardInterval determinant{};
    const auto include_expansion =
        [&determinant, &initialized](const OutwardInterval& expansion) {
            if (!initialized) {
                determinant = expansion;
                initialized = true;
            } else {
                determinant = intersect_intervals(determinant, expansion);
            }
        };

    for (std::size_t row = 0; row < 3U; ++row) {
        std::array<OutwardInterval, 3> terms{};
        for (std::size_t column = 0; column < 3U; ++column) {
            terms[column] = multiply_intervals(
                a[row][column], cofactors[row][column]);
        }
        include_expansion(
            sum_three_intervals(terms[0], terms[1], terms[2], false));
        include_expansion(
            sum_three_intervals(terms[0], terms[1], terms[2], true));
    }
    for (std::size_t column = 0; column < 3U; ++column) {
        std::array<OutwardInterval, 3> terms{};
        for (std::size_t row = 0; row < 3U; ++row) {
            terms[row] = multiply_intervals(
                a[row][column], cofactors[row][column]);
        }
        include_expansion(
            sum_three_intervals(terms[0], terms[1], terms[2], false));
        include_expansion(
            sum_three_intervals(terms[0], terms[1], terms[2], true));
    }
    if (!initialized) {
        throw std::logic_error(
            "reciprocal metric determinant certificate is empty");
    }
    return determinant;
}

PeriodicCorrelationReciprocalMetricEnumerationPlan
make_certified_enumeration_plan(
    const Eigen::Matrix3d& reciprocal_lattice,
    const Eigen::Vector3d& centered_transfer_fractional,
    double squared_membership_limit) {
    require_supported_binary64_environment();
    if (!reciprocal_lattice.allFinite()) {
        throw std::invalid_argument(
            "reciprocal metric enumeration lattice must be finite");
    }
    if (!centered_transfer_fractional.allFinite()) {
        throw std::invalid_argument(
            "reciprocal metric enumeration transfer must be finite");
    }
    for (Eigen::Index axis = 0; axis < 3; ++axis) {
        const double component = centered_transfer_fractional[axis];
        if (component < -0.5 || !(component < 0.5)) {
            throw std::invalid_argument(
                "reciprocal metric enumeration transfer must use the "
                "centred [-1/2,1/2) convention");
        }
    }
    if (!std::isfinite(squared_membership_limit)
        || squared_membership_limit <= 0.0) {
        throw std::invalid_argument(
            "reciprocal metric enumeration squared membership limit must "
            "be finite and positive");
    }

    const double scale = reciprocal_lattice.cwiseAbs().maxCoeff();
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::invalid_argument(
            "reciprocal metric determinant_not_certified: zero or "
            "non-finite normalization scale");
    }

    IntervalMatrix3 normalized{};
    for (std::size_t row = 0; row < 3U; ++row) {
        for (std::size_t column = 0; column < 3U; ++column) {
            normalized[row][column] = normalized_entry_interval(
                reciprocal_lattice(
                    static_cast<Eigen::Index>(row),
                    static_cast<Eigen::Index>(column)),
                scale);
        }
    }
    const IntervalMatrix3 cofactors = cofactor_intervals(normalized);
    const OutwardInterval determinant =
        certified_determinant_interval(normalized, cofactors);
    if (determinant.lower <= 0.0 && determinant.upper >= 0.0) {
        throw std::runtime_error(
            "reciprocal metric determinant_not_certified: outward "
            "determinant interval contains zero");
    }
    const double determinant_magnitude_lower = determinant.lower > 0.0
        ? determinant.lower
        : -determinant.upper;
    if (!(determinant_magnitude_lower > 0.0)
        || !std::isfinite(determinant_magnitude_lower)) {
        throw std::runtime_error(
            "reciprocal metric determinant_not_certified: no positive "
            "determinant-magnitude lower bound");
    }

    std::array<double, 3> inverse_row_one{};
    std::array<double, 3> inverse_row_two{};
    double maximum_inverse_row_one = 0.0;
    double maximum_inverse_row_two = 0.0;
    for (std::size_t row = 0; row < 3U; ++row) {
        double row_one = 0.0;
        double row_two_squared = 0.0;
        for (std::size_t column = 0; column < 3U; ++column) {
            const double entry_upper = divide_up(
                absolute_interval_upper(cofactors[column][row]),
                determinant_magnitude_lower);
            row_one = add_up(row_one, entry_upper);
            row_two_squared = add_up(
                row_two_squared,
                multiply_up(entry_upper, entry_upper));
        }
        inverse_row_one[row] = row_one;
        inverse_row_two[row] = square_root_up(row_two_squared);
        maximum_inverse_row_one =
            std::max(maximum_inverse_row_one, inverse_row_one[row]);
        maximum_inverse_row_two =
            std::max(maximum_inverse_row_two, inverse_row_two[row]);
    }

    double normalized_matrix_infinity_norm = 0.0;
    for (std::size_t row = 0; row < 3U; ++row) {
        double row_sum = 0.0;
        for (std::size_t column = 0; column < 3U; ++column) {
            row_sum = add_up(
                row_sum, absolute_interval_upper(normalized[row][column]));
        }
        normalized_matrix_infinity_norm =
            std::max(normalized_matrix_infinity_norm, row_sum);
    }

    constexpr double unit_roundoff = 0x1p-53;
    constexpr double subnormal_spacing =
        std::numeric_limits<double>::denorm_min();
    constexpr double absolute_error_bound =
        4.0 * std::numeric_limits<double>::denorm_min();
    const double three_u = 3.0 * unit_roundoff;
    const double gamma_denominator = subtract_down(1.0, three_u);
    const double gamma3 = divide_up(three_u, gamma_denominator);
    const double norm_denominator = subtract_down(1.0, gamma3);
    if (!(gamma_denominator > 0.0) || !(norm_denominator > 0.0)) {
        throw std::runtime_error(
            "reciprocal metric unsupported_floating_point_environment: "
            "roundoff denominators are not positive");
    }

    const double accepted_radius_squared = divide_up(
        add_up(squared_membership_limit, absolute_error_bound),
        norm_denominator);
    const double accepted_radius =
        square_root_up(accepted_radius_squared);
    const double feedback = multiply_up(
        multiply_up(gamma3, normalized_matrix_infinity_norm),
        maximum_inverse_row_one);
    if (!std::isfinite(feedback) || !(feedback < 1.0)) {
        throw std::runtime_error(
            "reciprocal metric noncontractive_matvec_certificate: "
            "outward feedback bound is not below one");
    }
    const double feedback_denominator = subtract_down(1.0, feedback);
    if (!(feedback_denominator > 0.0)) {
        throw std::runtime_error(
            "reciprocal metric noncontractive_matvec_certificate: "
            "no positive feedback denominator");
    }

    const double fixed_point_numerator = add_up(
        multiply_up(maximum_inverse_row_two, accepted_radius),
        multiply_up(maximum_inverse_row_one, absolute_error_bound));
    const double shifted_infinity_bound = divide_up(
        divide_up(fixed_point_numerator, scale),
        feedback_denominator);
    const double matvec_error_bound = add_up(
        divide_up(
            multiply_up(
                multiply_up(gamma3, normalized_matrix_infinity_norm),
                fixed_point_numerator),
            feedback_denominator),
        absolute_error_bound);

    PeriodicCorrelationReciprocalMetricEnumerationPlan plan;
    plan.normalization_scale = scale;
    plan.determinant_lower_bound = determinant.lower;
    plan.determinant_upper_bound = determinant.upper;
    plan.normalized_matrix_infinity_norm_upper_bound =
        normalized_matrix_infinity_norm;
    plan.inverse_row_one_norm_upper_bounds = inverse_row_one;
    plan.inverse_row_two_norm_upper_bounds = inverse_row_two;
    plan.dot_relative_error_upper_bound = gamma3;
    plan.dot_absolute_error_upper_bound = absolute_error_bound;
    plan.accepted_vector_radius_upper_bound = accepted_radius;
    plan.matvec_feedback_upper_bound = feedback;
    plan.matvec_error_upper_bound = matvec_error_bound;
    plan.shifted_vector_infinity_norm_upper_bound = shifted_infinity_bound;

    bool empty = false;
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const double stored_shift_bound = divide_up(
            add_up(
                multiply_up(inverse_row_two[axis], accepted_radius),
                multiply_up(inverse_row_one[axis], matvec_error_bound)),
            scale);
        const double exact_shift_bound = divide_up(
            add_up(stored_shift_bound, subnormal_spacing),
            subtract_down(1.0, unit_roundoff));
        if (!std::isfinite(stored_shift_bound)
            || !std::isfinite(exact_shift_bound)) {
            throw std::overflow_error(
                "reciprocal metric non_finite_reach");
        }
        plan.stored_shift_component_upper_bounds[axis] = stored_shift_bound;
        plan.exact_shift_component_upper_bounds[axis] = exact_shift_bound;

        const double fractional = centered_transfer_fractional[
            static_cast<Eigen::Index>(axis)];
        const double real_lower = subtract_down(-exact_shift_bound, fractional);
        const double real_upper = subtract_up(exact_shift_bound, fractional);
        const double integer_lower = std::ceil(real_lower);
        const double integer_upper = std::floor(real_upper);
        if (!std::isfinite(integer_lower) || !std::isfinite(integer_upper)
            || integer_lower < -kMaximumExactBinary64Integer
            || integer_lower > kMaximumExactBinary64Integer
            || integer_upper < -kMaximumExactBinary64Integer
            || integer_upper > kMaximumExactBinary64Integer) {
            throw std::overflow_error(
                "reciprocal metric exact_integer_range_exceeded");
        }
        plan.lower_bounds[axis] =
            static_cast<std::int64_t>(integer_lower);
        plan.upper_bounds[axis] =
            static_cast<std::int64_t>(integer_upper);
        empty = empty || plan.lower_bounds[axis] > plan.upper_bounds[axis];
    }

    if (empty) {
        plan.candidate_count = 0U;
        return plan;
    }
    std::uint64_t candidate_count = 1U;
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        candidate_count = checked_multiply(
            candidate_count,
            axis_width(plan.lower_bounds[axis], plan.upper_bounds[axis]),
            "certified Cartesian candidate count");
    }
    plan.candidate_count = candidate_count;
    return plan;
}

std::uint64_t axis_width(std::int64_t lower, std::int64_t upper) {
    if (lower > upper) {
        throw std::logic_error(
            "reciprocal metric source axis bounds are reversed");
    }
    // The certified endpoints are within +/-2^53, so this signed difference is
    // exact and cannot approach int64 overflow.
    return static_cast<std::uint64_t>(upper - lower) + 1U;
}

using ReciprocalSourceGeometry = detail::PeriodicReciprocalNumericGeometry;

struct AcceptedReciprocalVector {
    std::array<std::int64_t, 3> integer_label = {0, 0, 0};
    /// px, py, pz, |p|^2, and 4*pi/(Omega |p|^2).
    std::array<double, 5> lanes = {0.0, 0.0, 0.0, 0.0, 0.0};
};

/// Allocation-free canonical traversal shared by the census/hash passes and
/// the later panelised numerical builder that will be appended to this TU.
template <typename AcceptedCallback>
std::uint64_t for_each_accepted_vector(
    const ReciprocalSourceGeometry& geometry,
    AcceptedCallback&& callback) {
    require_supported_binary64_environment();
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        if (geometry.lower_bounds[axis] > geometry.upper_bounds[axis]) {
            return 0U;
        }
    }
    std::uint64_t accepted_count = 0U;
    std::int64_t n0 = geometry.lower_bounds[0];
    for (;;) {
        std::int64_t n1 = geometry.lower_bounds[1];
        for (;;) {
            std::int64_t n2 = geometry.lower_bounds[2];
            for (;;) {
                const std::array<std::int64_t, 3> integer_label = {
                    n0, n1, n2};
                const bool exact_zero = geometry.exact_gamma
                    && n0 == 0 && n1 == 0 && n2 == 0;
                if (!exact_zero) {
                    Eigen::Vector3d shifted;
                    for (std::size_t axis = 0; axis < 3U; ++axis) {
                        shifted[static_cast<Eigen::Index>(axis)] =
                            std::fma(
                                1.0,
                                static_cast<double>(integer_label[axis]),
                                geometry.q_fractional[
                                    static_cast<Eigen::Index>(axis)]);
                    }
                    const Eigen::Vector3d p =
                        fixed_binary64_matvec(
                            geometry.reciprocal_lattice, shifted);
                    if (!p.allFinite()) {
                        throw std::overflow_error(
                            "reciprocal metric source produced a non-finite "
                            "shifted reciprocal vector");
                    }
                    const double p2 = fixed_binary64_squared_norm(p);
                    if (!std::isfinite(p2)) {
                        throw std::overflow_error(
                            "reciprocal metric source squared norm "
                            "overflowed binary64");
                    }
                    const double membership_limit = geometry.exact_gamma
                        ? geometry.cutoff_squared
                        : geometry.radial_limit_squared;
                    if (p2 <= membership_limit) {
                        if (p2 <= 0.0) {
                            throw std::logic_error(
                                "a nonzero integer-labelled reciprocal "
                                "vector has non-positive squared norm");
                        }
                        // Eq. (20) of Sun, Berkelbach, McClain & Chan,
                        // J. Chem. Phys. 147, 164119 (2017),
                        // doi:10.1063/1.4998644, contains the positive
                        // plane-wave factor 4*pi/(Omega*|G+k|^2). Here q has
                        // the paper's k role and p=G+q; this source records
                        // the factor, not the surrounding metric sign.
                        const double weight =
                            (kFourPi / geometry.cell_volume) / p2;
                        if (!std::isfinite(weight) || weight <= 0.0) {
                            throw std::overflow_error(
                                "reciprocal metric Coulomb weight is not "
                                "finite and positive");
                        }
                        const AcceptedReciprocalVector record = {
                            integer_label,
                            {p[0], p[1], p[2], p2, weight},
                        };
                        callback(record);
                        accepted_count = checked_add(
                            accepted_count,
                            1U,
                            "accepted reciprocal-vector count");
                    }
                }
                if (n2 == geometry.upper_bounds[2]) break;
                ++n2;
            }
            if (n1 == geometry.upper_bounds[1]) break;
            ++n1;
        }
        if (n0 == geometry.upper_bounds[0]) break;
        ++n0;
    }
    return accepted_count;
}

void add_source_prefix(
    CanonicalSourceHasher& digest,
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const std::string& producer_identity,
    const std::string& auxiliary_basis_identity,
    std::uint64_t q_index,
    const std::array<int, 3>& centered_doubled_numerator,
    const std::array<int, 3>& centered_reciprocal_wrap,
    const Eigen::Matrix3d& reciprocal_lattice,
    double reciprocal_energy_cutoff,
    double maximum_reciprocal_radius,
    double radial_boundary_tolerance,
    double cell_volume,
    const std::array<std::int64_t, 3>& lower_bounds,
    const std::array<std::int64_t, 3>& upper_bounds,
    std::uint64_t candidate_count,
    std::uint64_t accepted_count,
    std::uint64_t zero_mode_excluded_count) {
    digest.add_string(kPeriodicCorrelationReciprocalMetricProducerId);
    digest.add_u32(kPeriodicCorrelationReciprocalMetricProducerVersion);
    digest.add_string(producer_identity);
    digest.add_u32(reference.contract_version());
    digest.add_u32(schedule.contract_version());
    digest.add_u32(reference.state().digest_version());
    digest.add_u32(reference.dimensions().allocation_contract_version);
    digest.add_string(reference.state().state_identity_sha256());
    digest.add_string(reference.dimensions().calculation_identity);
    digest.add_string(reference.dimensions().allocation_identity);
    digest.add_string(schedule.schedule_identity_sha256());
    digest.add_u32(kAuxiliaryBasisContentDigestVersion);
    digest.add_string(auxiliary_basis_identity);
    digest.add_u32(static_cast<std::uint32_t>(
        reference.state().periodic_dimension()));
    for (const int component : reference.dimensions().mesh) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    for (const int component : reference.dimensions().is_shift) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    digest.add_u64(q_index);
    for (const int component : centered_doubled_numerator) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    for (const int component : centered_reciprocal_wrap) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    for (Eigen::Index row = 0; row < 3; ++row) {
        for (Eigen::Index column = 0; column < 3; ++column) {
            digest.add_double(reciprocal_lattice(row, column));
        }
    }
    digest.add_double(reciprocal_energy_cutoff);
    digest.add_double(maximum_reciprocal_radius);
    digest.add_double(radial_boundary_tolerance);
    digest.add_double(cell_volume);
    for (const auto component : lower_bounds) digest.add_i64(component);
    for (const auto component : upper_bounds) digest.add_i64(component);
    digest.add_u64(candidate_count);
    digest.add_u64(accepted_count);
    digest.add_u64(zero_mode_excluded_count);
}

}  // namespace

PeriodicCorrelationReciprocalMetricEnumerationPlan
plan_periodic_correlation_reciprocal_metric_enumeration(
    const Eigen::Matrix3d& reciprocal_lattice,
    const Eigen::Vector3d& centered_transfer_fractional,
    double squared_membership_limit) {
    return make_certified_enumeration_plan(
        reciprocal_lattice,
        centered_transfer_fractional,
        squared_membership_limit);
}

detail::PeriodicReciprocalNumericPlan
detail::prepare_periodic_reciprocal_numeric_source(
    const Eigen::Matrix3d& reciprocal_lattice,
    const Eigen::Vector3d& q_fractional, double reciprocal_energy_cutoff,
    bool exact_gamma, std::uint64_t candidate_count_cap) {
    if (candidate_count_cap == 0U) {
        throw std::invalid_argument("reciprocal numeric source requires a positive candidate cap");
    }
    // The arithmetic below is the v1 factory's original numerical block.
    // Shared extraction changes neither its evaluation order nor its wire.
    if (!reciprocal_lattice.allFinite()) {
        throw std::logic_error("admitted reciprocal lattice is no longer finite");
    }
    const double cutoff_squared = 2.0 * reciprocal_energy_cutoff;
    const double maximum_radius = std::sqrt(cutoff_squared);
    const Eigen::Vector3d q_cartesian =
        fixed_binary64_matvec(reciprocal_lattice, q_fractional);
    if (!q_cartesian.allFinite()) {
        throw std::overflow_error("reciprocal metric source q vector is non-finite");
    }
    const double q_norm = std::sqrt(fixed_binary64_squared_norm(q_cartesian));
    if (!std::isfinite(q_norm)) {
        throw std::overflow_error("reciprocal metric source q norm is non-finite");
    }
    double boundary_tolerance = 0.0;
    if (!exact_gamma) {
        const double tolerance_scale = std::max(maximum_radius + q_norm, 1.0);
        boundary_tolerance = 128.0 * std::numeric_limits<double>::epsilon() * tolerance_scale;
    }
    const double radial_limit = maximum_radius + boundary_tolerance;
    const double radial_limit_squared = radial_limit * radial_limit;
    if (!std::isfinite(maximum_radius) || !std::isfinite(boundary_tolerance)
        || !std::isfinite(radial_limit) || !std::isfinite(radial_limit_squared)) {
        throw std::overflow_error("reciprocal metric source radial cutoff is non-finite");
    }
    const double membership_limit = exact_gamma ? cutoff_squared : radial_limit_squared;
    const auto enumeration_plan = plan_periodic_correlation_reciprocal_metric_enumeration(
        reciprocal_lattice, q_fractional, membership_limit);
    if (enumeration_plan.candidate_count > candidate_count_cap) {
        throw std::length_error(
            "reciprocal metric source candidate_cap_exceeded: certified "
            "candidate count exceeds the caller cap");
    }
    const double cell_volume = fixed_binary64_cell_volume(
        reciprocal_lattice, enumeration_plan.normalization_scale,
        enumeration_plan.determinant_lower_bound, enumeration_plan.determinant_upper_bound);
    PeriodicReciprocalNumericPlan result;
    result.geometry.reciprocal_lattice = reciprocal_lattice;
    result.geometry.q_fractional = q_fractional;
    result.geometry.lower_bounds = enumeration_plan.lower_bounds;
    result.geometry.upper_bounds = enumeration_plan.upper_bounds;
    result.geometry.cutoff_squared = cutoff_squared;
    result.geometry.radial_limit = radial_limit;
    result.geometry.radial_limit_squared = radial_limit_squared;
    result.geometry.cell_volume = cell_volume;
    result.geometry.exact_gamma = exact_gamma;
    result.enumeration = enumeration_plan;
    result.q_cartesian = q_cartesian;
    result.maximum_radius = maximum_radius;
    result.boundary_tolerance = boundary_tolerance;
    return result;
}

std::uint64_t detail::visit_periodic_reciprocal_numeric_source(
    const PeriodicReciprocalNumericGeometry& geometry,
    PeriodicReciprocalRecordCallback callback, void* user) {
    if (callback == nullptr) throw std::invalid_argument("reciprocal numeric source requires a callback");
    return for_each_accepted_vector(geometry, [&](const AcceptedReciprocalVector& record) {
        callback(record.integer_label, record.lanes, user);
    });
}

std::string
periodic_correlation_reciprocal_metric_producer_identity_sha256() {
    CanonicalSourceHasher digest(
        kProducerIdentityDomain,
        kPeriodicCorrelationReciprocalMetricSourceContractVersion);
    digest.add_string(kPeriodicCorrelationReciprocalMetricProducerId);
    digest.add_u32(kPeriodicCorrelationReciprocalMetricProducerVersion);
    return digest.finish_hex();
}

std::uint64_t periodic_correlation_reciprocal_metric_source_wire_bytes(
    std::uint64_t accepted_vector_count) {
    static_assert(
        kPeriodicCorrelationReciprocalMetricAcceptedVectorWireBytes
            == 3U * sizeof(std::int64_t) + 5U * sizeof(double),
        "one reciprocal metric source record is three i64 labels and five "
        "binary64 lanes");
    static_assert(
        kPeriodicCorrelationReciprocalMetricSourceFixedPrefixWireBytes
            <= kSha256MaximumMessageBytes,
        "reciprocal metric source prefix must fit the SHA-256 message domain");
    constexpr std::uint64_t available =
        kSha256MaximumMessageBytes
        - kPeriodicCorrelationReciprocalMetricSourceFixedPrefixWireBytes;
    if (accepted_vector_count
        > available
            / kPeriodicCorrelationReciprocalMetricAcceptedVectorWireBytes) {
        throw std::length_error(
            "reciprocal metric canonical source exceeds SHA-256's "
            "less-than-2^64-bit message-length domain");
    }
    return kPeriodicCorrelationReciprocalMetricSourceFixedPrefixWireBytes
        + accepted_vector_count
            * kPeriodicCorrelationReciprocalMetricAcceptedVectorWireBytes;
}

std::uint64_t periodic_correlation_reciprocal_metric_payload_wire_bytes(
    std::uint64_t n_auxiliary) {
    static_assert(
        kPeriodicCorrelationReciprocalMetricPayloadFixedPrefixWireBytes
            <= kSha256MaximumMessageBytes,
        "reciprocal metric payload prefix must fit the SHA-256 message "
        "domain");
    constexpr std::uint64_t available =
        kSha256MaximumMessageBytes
        - kPeriodicCorrelationReciprocalMetricPayloadFixedPrefixWireBytes;
    constexpr std::uint64_t maximum_elements =
        available / static_cast<std::uint64_t>(sizeof(std::complex<double>));
    if (n_auxiliary != 0U
        && n_auxiliary > maximum_elements / n_auxiliary) {
        throw std::length_error(
            "reciprocal metric canonical payload exceeds SHA-256's "
            "less-than-2^64-bit message-length domain");
    }
    const std::uint64_t element_count = n_auxiliary * n_auxiliary;
    return kPeriodicCorrelationReciprocalMetricPayloadFixedPrefixWireBytes
        + static_cast<std::uint64_t>(sizeof(std::complex<double>))
            * element_count;
}

PeriodicCorrelationReciprocalMetricSourceManifest::
    PeriodicCorrelationReciprocalMetricSourceManifest(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::array<char, 64> source_identity_ascii,
        int periodic_dimension,
        std::array<int, 3> mesh,
        std::array<int, 3> is_shift,
        std::uint64_t q_index,
        std::array<int, 3> centered_doubled_numerator,
        std::array<int, 3> centered_reciprocal_wrap,
        Eigen::Vector3d q_fractional,
        Eigen::Vector3d q_cartesian,
        Eigen::Matrix3d reciprocal_lattice,
        double reciprocal_energy_cutoff,
        double maximum_reciprocal_radius,
        double radial_boundary_tolerance,
        double cell_volume_bohr3,
        std::array<std::int64_t, 3> lower_bounds,
        std::array<std::int64_t, 3> upper_bounds,
        std::uint64_t candidate_count,
        std::uint64_t accepted_vector_count,
        std::uint64_t zero_mode_excluded_count)
    : state_(std::move(state)),
      source_identity_ascii_(source_identity_ascii),
      periodic_dimension_(periodic_dimension),
      mesh_(mesh),
      is_shift_(is_shift),
      q_index_(q_index),
      centered_doubled_numerator_(centered_doubled_numerator),
      centered_reciprocal_wrap_(centered_reciprocal_wrap),
      q_fractional_(std::move(q_fractional)),
      q_cartesian_(std::move(q_cartesian)),
      reciprocal_lattice_(std::move(reciprocal_lattice)),
      reciprocal_energy_cutoff_(reciprocal_energy_cutoff),
      maximum_reciprocal_radius_(maximum_reciprocal_radius),
      radial_boundary_tolerance_(radial_boundary_tolerance),
      cell_volume_bohr3_(cell_volume_bohr3),
      lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      candidate_count_(candidate_count),
      accepted_vector_count_(accepted_vector_count),
      zero_mode_excluded_count_(zero_mode_excluded_count) {}

std::string PeriodicCorrelationReciprocalMetricSourceManifest::
    source_identity_sha256() const {
    return std::string(
        source_identity_ascii_.data(), source_identity_ascii_.size());
}

PeriodicCorrelationFactorBuildQRecord
PeriodicCorrelationReciprocalMetricSourceManifest::factor_build_q_record()
    const {
    if (!state_) {
        throw std::invalid_argument(
            "moved-from reciprocal metric source manifest has no "
            "mean-field state");
    }
    if (!std::all_of(
            source_identity_ascii_.begin(),
            source_identity_ascii_.end(),
            [](char character) {
                return (character >= '0' && character <= '9')
                    || (character >= 'a' && character <= 'f');
            })) {
        throw std::logic_error(
            "reciprocal metric source manifest has an invalid identity");
    }
    PeriodicCorrelationFactorBuildQRecord record;
    record.q_index = q_index_;
    record.centered_doubled_numerator = centered_doubled_numerator_;
    record.centered_reciprocal_wrap = centered_reciprocal_wrap_;
    record.base_reciprocal_vector_count = accepted_vector_count_;
    record.tail_reciprocal_vector_count = 0U;
    record.zero_mode_excluded_count = zero_mode_excluded_count_;
    record.short_range_metric_cell_count = 0U;
    record.short_range_metric_task_count = 0U;
    record.short_range_three_center_cell_pair_count = 0U;
    record.short_range_three_center_task_count = 0U;
    record.source_manifest_bytes =
        kPeriodicCorrelationReciprocalMetricSourceManifestBytes;
    return record;
}

PeriodicCorrelationReciprocalMetricSourceManifest
make_periodic_correlation_reciprocal_metric_source_manifest(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildConfig& config,
    const BasisSet& auxiliary_basis,
    std::uint64_t q_index,
    std::uint64_t candidate_count_cap) {
    require_supported_binary64_environment();
    require_upstream_consistency(reference, schedule);
    require_source_configuration(config);
    if (candidate_count_cap == 0U) {
        throw std::invalid_argument(
            "reciprocal metric source candidate-count cap must be "
            "positive");
    }
    if (static_cast<std::uint64_t>(auxiliary_basis.nbasis())
            != reference.dimensions().n_auxiliary
        || static_cast<std::uint64_t>(auxiliary_basis.nbasis())
            != schedule.shape().n_auxiliary) {
        throw std::invalid_argument(
            "auxiliary BasisSet size does not match the admitted factor "
            "row extent");
    }
    const std::string auxiliary_basis_identity =
        auxiliary_basis_content_identity_sha256(auxiliary_basis);
    if (auxiliary_basis_identity
        != config.auxiliary_basis_identity_sha256) {
        throw std::invalid_argument(
            "auxiliary BasisSet content identity does not match the "
            "factor-build configuration");
    }

    const auto& mesh = reference.dimensions().mesh;
    const RegularKMesh addressing(mesh, reference.dimensions().is_shift);
    const auto centered = centered_transfer(addressing, q_index);
    const auto& numerator = centered.first;
    const auto& wrap = centered.second;
    Eigen::Vector3d q_fractional;
    bool exact_gamma = true;
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const double denominator =
            2.0 * static_cast<double>(mesh[axis]);
        q_fractional[static_cast<Eigen::Index>(axis)] =
            static_cast<double>(numerator[axis]) / denominator;
        exact_gamma = exact_gamma && numerator[axis] == 0;
    }

    const Eigen::Matrix3d reciprocal_lattice =
        reference.state().reciprocal_lattice();
    const auto numeric = detail::prepare_periodic_reciprocal_numeric_source(
        reciprocal_lattice, q_fractional, config.reciprocal_energy_cutoff,
        exact_gamma, candidate_count_cap);
    const auto& geometry = numeric.geometry;
    const auto& q_cartesian = numeric.q_cartesian;
    const double maximum_radius = numeric.maximum_radius;
    const double boundary_tolerance = numeric.boundary_tolerance;
    const auto& lower_bounds = geometry.lower_bounds;
    const auto& upper_bounds = geometry.upper_bounds;
    const std::uint64_t candidate_count = numeric.enumeration.candidate_count;
    const double cell_volume = geometry.cell_volume;

    const std::uint64_t accepted_count = for_each_accepted_vector(
        geometry, [](const AcceptedReciprocalVector&) {});
    if (accepted_count == 0U) {
        throw std::invalid_argument(
            "reciprocal metric source has no usable vectors at this q and "
            "cutoff");
    }
    const std::uint64_t zero_mode_excluded_count = exact_gamma ? 1U : 0U;
    const std::uint64_t expected_source_wire_bytes =
        periodic_correlation_reciprocal_metric_source_wire_bytes(
            accepted_count);

    const std::string producer_identity =
        periodic_correlation_reciprocal_metric_producer_identity_sha256();
    CanonicalSourceHasher digest(
        kSourceIdentityDomain,
        kPeriodicCorrelationReciprocalMetricSourceContractVersion);
    add_source_prefix(
        digest,
        reference,
        schedule,
        producer_identity,
        auxiliary_basis_identity,
        q_index,
        numerator,
        wrap,
        reciprocal_lattice,
        config.reciprocal_energy_cutoff,
        maximum_radius,
        boundary_tolerance,
        cell_volume,
        lower_bounds,
        upper_bounds,
        candidate_count,
        accepted_count,
        zero_mode_excluded_count);
    if (digest.wire_bytes()
        != kPeriodicCorrelationReciprocalMetricSourceFixedPrefixWireBytes) {
        throw std::logic_error(
            "reciprocal metric source prefix wire extent changed without a "
            "contract-version update");
    }
    const std::uint64_t hashed_count = for_each_accepted_vector(
        geometry, [&digest](const AcceptedReciprocalVector& record) {
            for (const auto component : record.integer_label) {
                digest.add_i64(component);
            }
            for (const double lane : record.lanes) {
                digest.add_double(lane);
            }
        });
    if (hashed_count != accepted_count) {
        throw std::logic_error(
            "reciprocal metric source enumeration changed between census "
            "and identity passes");
    }
    if (digest.wire_bytes() != expected_source_wire_bytes) {
        throw std::logic_error(
            "reciprocal metric source record wire extent changed without a "
            "contract-version update");
    }
    std::string source_identity = digest.finish_hex();
    if (source_identity.size()
        != kPeriodicCorrelationReciprocalMetricSourceManifestBytes) {
        throw std::logic_error(
            "reciprocal metric source SHA-256 did not produce 64 ASCII "
            "bytes");
    }
    std::array<char, 64> source_identity_ascii{};
    std::copy(
        source_identity.begin(),
        source_identity.end(),
        source_identity_ascii.begin());

    return PeriodicCorrelationReciprocalMetricSourceManifest(
        reference.state_handle(),
        source_identity_ascii,
        reference.state().periodic_dimension(),
        mesh,
        reference.dimensions().is_shift,
        q_index,
        numerator,
        wrap,
        std::move(q_fractional),
        std::move(q_cartesian),
        std::move(reciprocal_lattice),
        config.reciprocal_energy_cutoff,
        maximum_radius,
        boundary_tolerance,
        cell_volume,
        lower_bounds,
        upper_bounds,
        candidate_count,
        accepted_count,
        zero_mode_excluded_count);
}

namespace {

std::array<char, 64> copy_sha256_ascii(const std::string& identity,
                                       const char* label) {
    require_lower_sha256(identity, label);
    std::array<char, 64> result{};
    std::copy(identity.begin(), identity.end(), result.begin());
    return result;
}

std::string sha256_ascii_string(const std::array<char, 64>& identity) {
    return std::string(identity.data(), identity.size());
}

bool same_q_record(const PeriodicCorrelationFactorBuildQRecord& left,
                   const PeriodicCorrelationFactorBuildQRecord& right) {
    return left.q_index == right.q_index
        && left.centered_doubled_numerator
            == right.centered_doubled_numerator
        && left.centered_reciprocal_wrap
            == right.centered_reciprocal_wrap
        && left.base_reciprocal_vector_count
            == right.base_reciprocal_vector_count
        && left.tail_reciprocal_vector_count
            == right.tail_reciprocal_vector_count
        && left.zero_mode_excluded_count
            == right.zero_mode_excluded_count
        && left.short_range_metric_cell_count
            == right.short_range_metric_cell_count
        && left.short_range_metric_task_count
            == right.short_range_metric_task_count
        && left.short_range_three_center_cell_pair_count
            == right.short_range_three_center_cell_pair_count
        && left.short_range_three_center_task_count
            == right.short_range_three_center_task_count
        && left.source_manifest_bytes == right.source_manifest_bytes;
}

bool exactly_equal(const Eigen::Vector3d& left,
                   const Eigen::Vector3d& right) noexcept {
    for (Eigen::Index index = 0; index < 3; ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

bool exactly_equal(const Eigen::Matrix3d& left,
                   const Eigen::Matrix3d& right) noexcept {
    for (Eigen::Index row = 0; row < 3; ++row) {
        for (Eigen::Index column = 0; column < 3; ++column) {
            if (left(row, column) != right(row, column)) return false;
        }
    }
    return true;
}

void require_same_source_manifest(
    const PeriodicCorrelationReciprocalMetricSourceManifest& supplied,
    const PeriodicCorrelationReciprocalMetricSourceManifest& replayed) {
    if (!supplied.state_handle()
        || supplied.state_handle().get() != replayed.state_handle().get()) {
        throw std::invalid_argument(
            "reciprocal metric source does not retain the live mean-field "
            "state");
    }
    if (supplied.contract_version() != replayed.contract_version()
        || supplied.source_identity_sha256()
            != replayed.source_identity_sha256()
        || supplied.periodic_dimension() != replayed.periodic_dimension()
        || supplied.mesh() != replayed.mesh()
        || supplied.is_shift() != replayed.is_shift()
        || supplied.q_index() != replayed.q_index()
        || supplied.centered_doubled_numerator()
            != replayed.centered_doubled_numerator()
        || supplied.centered_reciprocal_wrap()
            != replayed.centered_reciprocal_wrap()
        || !exactly_equal(
            supplied.q_fractional(), replayed.q_fractional())
        || !exactly_equal(
            supplied.q_cartesian(), replayed.q_cartesian())
        || !exactly_equal(
            supplied.reciprocal_lattice(),
            replayed.reciprocal_lattice())
        || supplied.reciprocal_energy_cutoff()
            != replayed.reciprocal_energy_cutoff()
        || supplied.maximum_reciprocal_radius()
            != replayed.maximum_reciprocal_radius()
        || supplied.radial_boundary_tolerance()
            != replayed.radial_boundary_tolerance()
        || supplied.cell_volume_bohr3()
            != replayed.cell_volume_bohr3()
        || supplied.lower_bounds() != replayed.lower_bounds()
        || supplied.upper_bounds() != replayed.upper_bounds()
        || supplied.candidate_count() != replayed.candidate_count()
        || supplied.accepted_vector_count()
            != replayed.accepted_vector_count()
        || supplied.zero_mode_excluded_count()
            != replayed.zero_mode_excluded_count()
        || !same_q_record(
            supplied.factor_build_q_record(),
            replayed.factor_build_q_record())) {
        throw std::logic_error(
            "reciprocal metric source replay does not reproduce the "
            "supplied manifest");
    }
}

ReciprocalSourceGeometry source_geometry(
    const PeriodicCorrelationReciprocalMetricSourceManifest& source) {
    ReciprocalSourceGeometry geometry;
    geometry.reciprocal_lattice = source.reciprocal_lattice();
    geometry.q_fractional = source.q_fractional();
    geometry.lower_bounds = source.lower_bounds();
    geometry.upper_bounds = source.upper_bounds();
    geometry.cutoff_squared = 2.0 * source.reciprocal_energy_cutoff();
    geometry.radial_limit = source.maximum_reciprocal_radius()
        + source.radial_boundary_tolerance();
    geometry.radial_limit_squared =
        geometry.radial_limit * geometry.radial_limit;
    geometry.cell_volume = source.cell_volume_bohr3();
    geometry.exact_gamma = source.zero_mode_excluded_count() == 1U;
    return geometry;
}

std::uint64_t normalized_binary64_bits(double value) noexcept {
    if (value == 0.0) value = 0.0;
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void require_self_conjugate_source_closure(
    const ReciprocalSourceGeometry& geometry,
    const PeriodicCorrelationReciprocalMetricSourceManifest& source) {
    std::array<std::int64_t, 3> partner_shift = {0, 0, 0};
    const auto& mesh = source.mesh();
    const auto& numerator = source.centered_doubled_numerator();
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        if (mesh[axis] <= 0 || numerator[axis] % mesh[axis] != 0) {
            throw std::logic_error(
                "self-conjugate reciprocal source has a non-integral "
                "twice-q component");
        }
        const int twice_q = numerator[axis] / mesh[axis];
        if (twice_q != 0 && twice_q != -1) {
            throw std::logic_error(
                "self-conjugate reciprocal source is outside the "
                "centered Gamma/Nyquist convention");
        }
        partner_shift[axis] = static_cast<std::int64_t>(-twice_q);
    }

    const double membership_limit = geometry.exact_gamma
        ? geometry.cutoff_squared
        : geometry.radial_limit_squared;
    const std::uint64_t audited_count = for_each_accepted_vector(
        geometry, [&](const AcceptedReciprocalVector& record) {
            std::array<std::int64_t, 3> partner_label = {0, 0, 0};
            Eigen::Vector3d shifted_partner;
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                if (record.integer_label[axis]
                    == std::numeric_limits<std::int64_t>::min()) {
                    throw std::overflow_error(
                        "self-conjugate reciprocal partner label "
                        "overflows int64");
                }
                std::int64_t partner = -record.integer_label[axis];
                if (partner_shift[axis] != 0) {
                    if (partner
                        == std::numeric_limits<std::int64_t>::max()) {
                        throw std::overflow_error(
                            "self-conjugate reciprocal partner label "
                            "overflows int64");
                    }
                    ++partner;
                }
                const auto exact_limit = static_cast<std::int64_t>(
                    kMaximumExactBinary64Integer);
                if (partner < -exact_limit || partner > exact_limit) {
                    throw std::overflow_error(
                        "self-conjugate reciprocal partner label is not "
                        "exactly representable in binary64");
                }
                if (partner < geometry.lower_bounds[axis]
                    || partner > geometry.upper_bounds[axis]) {
                    throw std::runtime_error(
                        "self-conjugate reciprocal source is not closed "
                        "under p -> -p");
                }
                partner_label[axis] = partner;
                shifted_partner[static_cast<Eigen::Index>(axis)] = std::fma(
                    1.0,
                    static_cast<double>(partner),
                    geometry.q_fractional[
                        static_cast<Eigen::Index>(axis)]);
            }

            const bool partner_is_zero = geometry.exact_gamma
                && partner_label[0] == 0 && partner_label[1] == 0
                && partner_label[2] == 0;
            if (partner_is_zero) {
                throw std::runtime_error(
                    "self-conjugate reciprocal source maps a retained "
                    "vector onto the excluded zero mode");
            }
            const Eigen::Vector3d partner = fixed_binary64_matvec(
                geometry.reciprocal_lattice, shifted_partner);
            if (!partner.allFinite()) {
                throw std::overflow_error(
                    "self-conjugate reciprocal partner is non-finite");
            }
            const double partner_p2 = fixed_binary64_squared_norm(partner);
            if (!std::isfinite(partner_p2) || partner_p2 <= 0.0
                || partner_p2 > membership_limit) {
                throw std::runtime_error(
                    "self-conjugate reciprocal partner is not accepted by "
                    "the sealed source predicate");
            }
            const double partner_weight =
                (kFourPi / geometry.cell_volume) / partner_p2;
            if (!std::isfinite(partner_weight)
                || partner_weight <= 0.0) {
                throw std::overflow_error(
                    "self-conjugate reciprocal partner weight is not "
                    "finite and positive");
            }
            for (Eigen::Index axis = 0; axis < 3; ++axis) {
                if (normalized_binary64_bits(partner[axis])
                    != normalized_binary64_bits(
                        -record.lanes[static_cast<std::size_t>(axis)])) {
                    throw std::runtime_error(
                        "self-conjugate reciprocal partner is not the "
                        "bitwise negative momentum");
                }
            }
            if (normalized_binary64_bits(partner_p2)
                    != normalized_binary64_bits(record.lanes[3])
                || normalized_binary64_bits(partner_weight)
                    != normalized_binary64_bits(record.lanes[4])) {
                throw std::runtime_error(
                    "self-conjugate reciprocal partner does not preserve "
                    "the sealed norm and Coulomb weight");
            }
        });
    if (audited_count != source.accepted_vector_count()) {
        throw std::logic_error(
            "self-conjugate reciprocal source closure audit changed the "
            "accepted-vector count");
    }
}

void require_size_t_extent(std::uint64_t extent, const char* label) {
    if (extent > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            std::string("reciprocal metric ") + label
            + " exceeds size_t");
    }
}

PeriodicCorrelationByteCount reciprocal_metric_phase_peak(
    const PeriodicCorrelationFactorBuildPlan& plan) {
    bool found = false;
    PeriodicCorrelationByteCount peak = 0U;
    for (const auto& phase : plan.phases) {
        if (phase.phase
            != PeriodicCorrelationFactorBuildPhase::ReciprocalMetric) {
            continue;
        }
        if (found) {
            throw std::logic_error(
                "factor-build plan contains duplicate reciprocal metric "
                "phases");
        }
        found = true;
        peak = phase.peak_memory_bytes;
    }
    if (!found) {
        throw std::logic_error(
            "factor-build plan has no reciprocal metric phase");
    }
    return peak;
}

void neumaier_add(double value, double& sum, double& correction) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "reciprocal metric accumulation received a non-finite term");
    }
    const double next = sum + value;
    if (!std::isfinite(next)) {
        throw std::overflow_error(
            "reciprocal metric accumulation overflowed binary64");
    }
    const double increment = std::abs(sum) >= std::abs(value)
        ? (sum - next) + value
        : (value - next) + sum;
    const double next_correction = correction + increment;
    if (!std::isfinite(next_correction)) {
        throw std::overflow_error(
            "reciprocal metric compensation overflowed binary64");
    }
    sum = next;
    correction = next_correction;
}

void accumulate_diagonal(double value, std::complex<double>& accumulator) {
    double sum = accumulator.real();
    double correction = accumulator.imag();
    neumaier_add(value, sum, correction);
    accumulator = {sum, correction};
}

void accumulate_off_diagonal(
    const std::complex<double>& value,
    std::complex<double>& accumulator,
    std::complex<double>& compensation) {
    double real_sum = accumulator.real();
    double real_correction = compensation.real();
    neumaier_add(value.real(), real_sum, real_correction);
    double imag_sum = accumulator.imag();
    double imag_correction = compensation.imag();
    neumaier_add(value.imag(), imag_sum, imag_correction);
    accumulator = {real_sum, imag_sum};
    compensation = {real_correction, imag_correction};
}

double normalized_zero(double value) noexcept {
    return value == 0.0 ? 0.0 : value;
}

std::complex<double> normalized_complex(double real,
                                        double imaginary) noexcept {
    return {normalized_zero(real), normalized_zero(imaginary)};
}

bool momentum_requires_conjugation(double x,
                                   double y,
                                   double z) noexcept {
    for (const double component : {x, y, z}) {
        if (component > 0.0) return false;
        if (component < 0.0) return true;
    }
    return false;
}

double self_conjugate_subnormal_floor(
    std::uint64_t accepted_vector_count) noexcept {
    double count = static_cast<double>(
        std::max<std::uint64_t>(accepted_vector_count, 1U));
    if (accepted_vector_count
        > kMaximumExactBinary64IntegerCount) {
        count = std::nextafter(
            count, std::numeric_limits<double>::infinity());
    }
    return 64.0 * std::numeric_limits<double>::denorm_min() * count;
}

double inflated_nonnegative_diagonal(double value,
                                     double subnormal_floor) {
    if (!std::isfinite(value) || value < 0.0
        || !std::isfinite(subnormal_floor) || subnormal_floor < 0.0) {
        throw std::logic_error(
            "reciprocal metric diagonal inflation inputs are invalid");
    }
    const double relative_allowance =
        64.0 * std::numeric_limits<double>::epsilon() * value;
    const double maximum = std::numeric_limits<double>::max();
    if (value > maximum - relative_allowance) return maximum;
    const double inflated = value + relative_allowance;
    if (inflated > maximum - subnormal_floor) return maximum;
    return inflated + subnormal_floor;
}

double stable_nonnegative_geometric_mean(double left, double right) {
    if (!std::isfinite(left) || !std::isfinite(right)
        || left < 0.0 || right < 0.0) {
        throw std::logic_error(
            "reciprocal metric geometric-mean inputs are invalid");
    }
    const double high = std::max(left, right);
    const double low = std::min(left, right);
    if (low == 0.0) return 0.0;
    const double result = std::sqrt(high) * std::sqrt(low);
    if (!std::isfinite(result)) {
        throw std::overflow_error(
            "reciprocal metric geometric-mean scale is non-finite");
    }
    return result;
}

double self_conjugate_imaginary_tolerance(
    double diagonal_left,
    double diagonal_right,
    std::uint64_t accepted_vector_count) {
    const double subnormal_floor = self_conjugate_subnormal_floor(
        accepted_vector_count);
    const double scale = stable_nonnegative_geometric_mean(
        inflated_nonnegative_diagonal(
            diagonal_left, subnormal_floor),
        inflated_nonnegative_diagonal(
            diagonal_right, subnormal_floor));
    const double tolerance = std::fma(
        256.0 * std::numeric_limits<double>::epsilon(),
        scale,
        subnormal_floor);
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::overflow_error(
            "self-conjugate reciprocal metric tolerance is invalid");
    }
    return tolerance;
}

}  // namespace

PeriodicCorrelationReciprocalMetricResult::
    PeriodicCorrelationReciprocalMetricResult(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::array<char, 64> source_identity_ascii,
        std::array<char, 64> census_identity_ascii,
        std::array<char, 64> plan_identity_ascii,
        std::array<char, 64> payload_identity_ascii,
        std::array<char, 64> auxiliary_basis_identity_ascii,
        std::uint64_t q_index,
        std::uint64_t conjugate_q_index,
        std::uint64_t n_auxiliary,
        std::uint64_t accepted_vector_count,
        std::uint64_t reciprocal_panel_capacity,
        std::uint64_t completed_panel_count,
        PeriodicCorrelationByteCount metric_bytes,
        PeriodicCorrelationByteCount reciprocal_panel_bytes,
        PeriodicCorrelationByteCount maximum_auxiliary_fourier_panel_bytes,
        PeriodicCorrelationByteCount maximum_weighted_fourier_panel_bytes,
        PeriodicCorrelationByteCount admitted_reciprocal_metric_peak_bytes,
        bool self_conjugate_transfer,
        double maximum_self_conjugate_imaginary_residual,
        std::vector<std::complex<double>> matrix_row_major)
    : state_(std::move(state)),
      source_identity_ascii_(source_identity_ascii),
      census_identity_ascii_(census_identity_ascii),
      plan_identity_ascii_(plan_identity_ascii),
      payload_identity_ascii_(payload_identity_ascii),
      auxiliary_basis_identity_ascii_(auxiliary_basis_identity_ascii),
      q_index_(q_index),
      conjugate_q_index_(conjugate_q_index),
      n_auxiliary_(n_auxiliary),
      accepted_vector_count_(accepted_vector_count),
      reciprocal_panel_capacity_(reciprocal_panel_capacity),
      completed_panel_count_(completed_panel_count),
      metric_bytes_(metric_bytes),
      reciprocal_panel_bytes_(reciprocal_panel_bytes),
      maximum_auxiliary_fourier_panel_bytes_(
          maximum_auxiliary_fourier_panel_bytes),
      maximum_weighted_fourier_panel_bytes_(
          maximum_weighted_fourier_panel_bytes),
      admitted_reciprocal_metric_peak_bytes_(
          admitted_reciprocal_metric_peak_bytes),
      self_conjugate_transfer_(self_conjugate_transfer),
      maximum_self_conjugate_imaginary_residual_(
          maximum_self_conjugate_imaginary_residual),
      matrix_row_major_(std::move(matrix_row_major)) {}

std::string PeriodicCorrelationReciprocalMetricResult::
    source_identity_sha256() const {
    return sha256_ascii_string(source_identity_ascii_);
}

std::string PeriodicCorrelationReciprocalMetricResult::
    census_identity_sha256() const {
    return sha256_ascii_string(census_identity_ascii_);
}

std::string PeriodicCorrelationReciprocalMetricResult::
    plan_identity_sha256() const {
    return sha256_ascii_string(plan_identity_ascii_);
}

std::string PeriodicCorrelationReciprocalMetricResult::
    payload_identity_sha256() const {
    return sha256_ascii_string(payload_identity_ascii_);
}

std::string PeriodicCorrelationReciprocalMetricResult::
    auxiliary_basis_identity_sha256() const {
    return sha256_ascii_string(auxiliary_basis_identity_ascii_);
}

detail::ReciprocalMetricNumericalResult detail::contract_reciprocal_metric_numeric(
    const BasisSet& auxiliary_basis, std::uint64_t n_auxiliary,
    std::uint64_t accepted_count, std::uint64_t panel_capacity,
    bool self_conjugate, std::uint64_t numerical_byte_cap,
    MetricReciprocalVisitor visitor, const void* source_user) {
    if (visitor == nullptr || n_auxiliary == 0U || accepted_count == 0U
        || panel_capacity == 0U || panel_capacity > accepted_count
        || auxiliary_basis.nbasis() != n_auxiliary || numerical_byte_cap == 0U) {
        throw std::invalid_argument("reciprocal metric numerical leaf has invalid shape, visitor or cap");
    }
    const auto metric_elements = checked_multiply(n_auxiliary, n_auxiliary, "metric leaf square");
    const auto metric_bytes = checked_multiply(16U, metric_elements, "metric leaf bytes");
    const auto reciprocal_panel_elements = checked_multiply(5U, panel_capacity, "metric leaf reciprocal elements");
    const auto reciprocal_bytes = checked_multiply(8U, reciprocal_panel_elements, "metric leaf reciprocal bytes");
    const auto fourier_panel_elements = checked_multiply(n_auxiliary, panel_capacity, "metric leaf Fourier elements");
    const auto one_fourier_panel_bytes = checked_multiply(16U, fourier_panel_elements, "metric leaf Fourier bytes");
    const auto required = checked_add(checked_add(metric_bytes, reciprocal_bytes, "metric leaf peak"),
        checked_multiply(2U, one_fourier_panel_bytes, "metric leaf double Fourier"), "metric leaf peak");
    if (required > numerical_byte_cap) throw std::length_error("reciprocal metric numerical leaf exceeds cap");
    require_size_t_extent(metric_elements, "metric leaf elements");
    require_size_t_extent(reciprocal_panel_elements, "metric leaf reciprocal elements");
    require_size_t_extent(fourier_panel_elements, "metric leaf Fourier elements");
    std::vector<std::complex<double>> metric(
        static_cast<std::size_t>(metric_elements), {0.0, 0.0});
    std::vector<double> reciprocal_panel(
        static_cast<std::size_t>(reciprocal_panel_elements), 0.0);
    std::vector<std::complex<double>> weighted_fourier_panel(
        static_cast<std::size_t>(fourier_panel_elements), {0.0, 0.0});

    std::uint64_t fill = 0U;
    std::uint64_t completed_panels = 0U;
    PeriodicCorrelationByteCount maximum_fourier_panel_bytes = 0U;
    const auto flush_panel = [&](std::uint64_t count) {
        if (count == 0U || count > panel_capacity) {
            throw std::logic_error(
                "reciprocal metric panel flush has an invalid extent");
        }
        if (self_conjugate) {
            // Pin Fourier parity rather than relying on platform sin/cos to
            // be bitwise odd/even.  Both members of a p/-p pair are evaluated
            // at the same canonical momentum; a temporarily negative p2 lane
            // marks records whose transform must be conjugated and restored.
            for (std::uint64_t vector = 0U; vector < count; ++vector) {
                const auto x = static_cast<std::size_t>(vector);
                const auto y = static_cast<std::size_t>(
                    panel_capacity + vector);
                const auto z = static_cast<std::size_t>(
                    2U * panel_capacity + vector);
                const auto p2 = static_cast<std::size_t>(
                    3U * panel_capacity + vector);
                if (momentum_requires_conjugation(
                        reciprocal_panel[x],
                        reciprocal_panel[y],
                        reciprocal_panel[z])) {
                    reciprocal_panel[x] = -reciprocal_panel[x];
                    reciprocal_panel[y] = -reciprocal_panel[y];
                    reciprocal_panel[z] = -reciprocal_panel[z];
                    reciprocal_panel[p2] = -reciprocal_panel[p2];
                }
            }
        }
        AuxiliaryFourierVectorView vectors;
        vectors.x = reciprocal_panel.data();
        vectors.y = reciprocal_panel.data()
            + static_cast<std::size_t>(panel_capacity);
        vectors.z = reciprocal_panel.data()
            + static_cast<std::size_t>(2U * panel_capacity);
        vectors.count = static_cast<std::size_t>(count);
        AuxiliaryFourierPanel fourier =
            auxiliary_gaussian_fourier_panel(
                auxiliary_basis, vectors, one_fourier_panel_bytes);
        const auto expected_fourier_bytes = checked_multiply(
            checked_multiply(
                n_auxiliary, count, "active Fourier panel elements"),
            static_cast<std::uint64_t>(sizeof(std::complex<double>)),
            "active Fourier panel bytes");
        if (fourier.n_auxiliary
                != static_cast<std::size_t>(n_auxiliary)
            || fourier.n_vectors != static_cast<std::size_t>(count)
            || fourier.output_bytes != expected_fourier_bytes) {
            throw std::logic_error(
                "auxiliary Fourier panel violates its admitted shape");
        }
        if (self_conjugate) {
            for (std::uint64_t vector = 0U; vector < count; ++vector) {
                const auto p2 = static_cast<std::size_t>(
                    3U * panel_capacity + vector);
                if (reciprocal_panel[p2] >= 0.0) continue;
                const auto x = static_cast<std::size_t>(vector);
                const auto y = static_cast<std::size_t>(
                    panel_capacity + vector);
                const auto z = static_cast<std::size_t>(
                    2U * panel_capacity + vector);
                reciprocal_panel[x] = -reciprocal_panel[x];
                reciprocal_panel[y] = -reciprocal_panel[y];
                reciprocal_panel[z] = -reciprocal_panel[z];
                reciprocal_panel[p2] = -reciprocal_panel[p2];
                for (std::uint64_t auxiliary = 0U;
                     auxiliary < n_auxiliary;
                     ++auxiliary) {
                    const auto row = static_cast<std::size_t>(auxiliary);
                    const auto column = static_cast<std::size_t>(vector);
                    fourier(row, column) =
                        std::conj(fourier(row, column));
                }
            }
        }
        maximum_fourier_panel_bytes = std::max(
            maximum_fourier_panel_bytes, fourier.output_bytes);

        for (std::uint64_t auxiliary = 0U;
             auxiliary < n_auxiliary;
             ++auxiliary) {
            for (std::uint64_t vector = 0U; vector < count; ++vector) {
                const std::complex<double> value = fourier(
                    static_cast<std::size_t>(auxiliary),
                    static_cast<std::size_t>(vector));
                const double weight = reciprocal_panel[
                    static_cast<std::size_t>(4U * panel_capacity + vector)];
                const std::complex<double> weighted(
                    weight * value.real(), -weight * value.imag());
                if (!std::isfinite(weighted.real())
                    || !std::isfinite(weighted.imag())) {
                    throw std::overflow_error(
                        "weighted auxiliary Fourier value is non-finite");
                }
                weighted_fourier_panel[
                    static_cast<std::size_t>(
                        auxiliary * panel_capacity + vector)] = weighted;
            }
        }

        for (std::uint64_t p = 0U; p < n_auxiliary; ++p) {
            for (std::uint64_t q = p; q < n_auxiliary; ++q) {
                const std::size_t upper = static_cast<std::size_t>(
                    p * n_auxiliary + q);
                const std::size_t lower = static_cast<std::size_t>(
                    q * n_auxiliary + p);
                for (std::uint64_t vector = 0U;
                     vector < count;
                     ++vector) {
                    const auto& left = weighted_fourier_panel[
                        static_cast<std::size_t>(
                            p * panel_capacity + vector)];
                    const std::complex<double> right = fourier(
                        static_cast<std::size_t>(q),
                        static_cast<std::size_t>(vector));
                    const double real = std::fma(
                        left.real(),
                        right.real(),
                        -(left.imag() * right.imag()));
                    if (p == q) {
                        accumulate_diagonal(real, metric[upper]);
                        continue;
                    }
                    const double imaginary = std::fma(
                        left.real(),
                        right.imag(),
                        left.imag() * right.real());
                    accumulate_off_diagonal(
                        {real, imaginary}, metric[upper], metric[lower]);
                }
            }
        }
        completed_panels = checked_add(
            completed_panels, 1U, "completed reciprocal panel count");
    };

    std::uint64_t received_count = 0U;
    auto consume_record = [&](const std::array<std::int64_t, 3>&,
                              const std::array<double, 5>& lanes) {
            if (received_count == accepted_count) {
                throw std::logic_error("reciprocal metric visitor exceeded its admitted source count");
            }
            ++received_count;
            for (std::uint64_t lane = 0U; lane < 5U; ++lane) {
                reciprocal_panel[static_cast<std::size_t>(
                    lane * panel_capacity + fill)] =
                    lanes[static_cast<std::size_t>(lane)];
            }
            ++fill;
            if (fill == panel_capacity) {
                flush_panel(fill);
                fill = 0U;
            }
    };
    const std::uint64_t streamed_count = visitor(
        [](const std::array<std::int64_t, 3>& label,
           const std::array<double, 5>& lanes, void* user) {
            (*static_cast<decltype(consume_record)*>(user))(label, lanes);
        }, &consume_record, source_user);
    if (received_count != streamed_count) {
        throw std::logic_error("reciprocal metric visitor count disagrees with callbacks");
    }
    if (fill != 0U) {
        flush_panel(fill);
        fill = 0U;
    }
    if (streamed_count != accepted_count) {
        throw std::logic_error(
            "reciprocal metric numerical replay changed the accepted "
            "source count");
    }
    const std::uint64_t expected_panel_count =
        accepted_count / panel_capacity
        + (accepted_count % panel_capacity == 0U ? 0U : 1U);
    if (completed_panels != expected_panel_count
        || maximum_fourier_panel_bytes != one_fourier_panel_bytes) {
        throw std::logic_error(
            "reciprocal metric panel execution does not match its "
            "admitted extent");
    }

    double maximum_self_conjugate_imaginary_residual = 0.0;
    for (std::uint64_t p = 0U; p < n_auxiliary; ++p) {
        const std::size_t diagonal = static_cast<std::size_t>(
            p * n_auxiliary + p);
        const double diagonal_value =
            metric[diagonal].real() + metric[diagonal].imag();
        if (!std::isfinite(diagonal_value) || diagonal_value < 0.0) {
            throw std::runtime_error(
                "reciprocal metric has a non-finite or negative diagonal");
        }
        metric[diagonal] = normalized_complex(diagonal_value, 0.0);
    }
    for (std::uint64_t p = 0U; p < n_auxiliary; ++p) {
        for (std::uint64_t q = p + 1U; q < n_auxiliary; ++q) {
            const std::size_t upper = static_cast<std::size_t>(
                p * n_auxiliary + q);
            const std::size_t lower = static_cast<std::size_t>(
                q * n_auxiliary + p);
            double real = metric[upper].real() + metric[lower].real();
            double imaginary =
                metric[upper].imag() + metric[lower].imag();
            if (!std::isfinite(real) || !std::isfinite(imaginary)) {
                throw std::overflow_error(
                    "reciprocal metric finalization produced a non-finite "
                    "element");
            }
            if (self_conjugate) {
                const double residual = std::abs(imaginary);
                maximum_self_conjugate_imaginary_residual = std::max(
                    maximum_self_conjugate_imaginary_residual, residual);
                const double tolerance =
                    self_conjugate_imaginary_tolerance(
                        metric[static_cast<std::size_t>(
                            p * n_auxiliary + p)].real(),
                        metric[static_cast<std::size_t>(
                            q * n_auxiliary + q)].real(),
                        accepted_count);
                if (residual > tolerance) {
                    throw std::runtime_error(
                        "self-conjugate reciprocal metric violates its "
                        "real-gauge residual bound");
                }
                imaginary = 0.0;
            }
            metric[upper] = normalized_complex(real, imaginary);
            metric[lower] = normalized_complex(real, -imaginary);
        }
    }

    ReciprocalMetricNumericalResult result;
    result.matrix = std::move(metric);
    result.completed_panel_count = completed_panels;
    result.maximum_fourier_panel_bytes = maximum_fourier_panel_bytes;
    result.maximum_self_conjugate_imaginary_residual = maximum_self_conjugate_imaginary_residual;
    return result;
}

PeriodicCorrelationReciprocalMetricResult
build_periodic_correlation_reciprocal_metric(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    const BasisSet& auxiliary_basis) {
    // Admission is deliberately first.  In particular, a bad source or basis
    // must not make an under-budget factor build walk reciprocal space.
    PeriodicCorrelationFactorBuildComponents admitted_components;
    PeriodicCorrelationByteCount admitted_metric_peak = 0U;
    std::array<char, 64> plan_identity_ascii{};
    {
        const PeriodicCorrelationFactorBuildPlan plan =
            plan_periodic_correlation_factor_build(
                reference, schedule, census);
        if (plan.admission
            != PeriodicCorrelationFactorBuildAdmissionCode::Admitted) {
            throw std::runtime_error(
                "reciprocal metric build requires an admitted factor-build "
                "plan");
        }
        require_lower_sha256(
            plan.plan_identity_sha256, "factor-build plan identity");
        admitted_components = plan.components;
        admitted_metric_peak = reciprocal_metric_phase_peak(plan);
        plan_identity_ascii = copy_sha256_ascii(
            plan.plan_identity_sha256, "factor-build plan identity");
    }

    if (!source.state_handle()
        || source.state_handle().get() != reference.state_handle().get()
        || source.state_handle().get() != schedule.state_handle().get()
        || source.state_handle().get() != census.state_handle().get()) {
        throw std::invalid_argument(
            "reciprocal metric source does not belong to the admitted "
            "reference, schedule, and census");
    }
    if (source.q_index()
        >= static_cast<std::uint64_t>(census.q_records().size())) {
        throw std::out_of_range(
            "reciprocal metric source q index is outside the factor-build "
            "census");
    }
    const auto& census_record = census.q_records()[
        static_cast<std::size_t>(source.q_index())];
    if (!same_q_record(source.factor_build_q_record(), census_record)) {
        throw std::invalid_argument(
            "reciprocal metric source record does not match the "
            "factor-build census");
    }

    // This is the allocation-free source preflight: reconstruct the source
    // from live immutable upstream objects, recount it, replay every canonical
    // record into SHA-256, and compare the entire resulting manifest.
    ReciprocalSourceGeometry geometry;
    std::array<char, 64> verified_source_identity_ascii{};
    std::uint64_t accepted_count = 0U;
    {
        const auto replayed_source =
            make_periodic_correlation_reciprocal_metric_source_manifest(
                reference,
                schedule,
                census.config(),
                auxiliary_basis,
                source.q_index(),
                source.candidate_count());
        require_same_source_manifest(source, replayed_source);
        geometry = source_geometry(replayed_source);
        verified_source_identity_ascii =
            replayed_source.source_identity_ascii();
        accepted_count = replayed_source.accepted_vector_count();
    }

    const std::uint64_t n_auxiliary = census.shape().n_auxiliary;
    if (n_auxiliary == 0U
        || n_auxiliary
            != static_cast<std::uint64_t>(auxiliary_basis.nbasis())) {
        throw std::logic_error(
            "reciprocal metric auxiliary extent changed after source "
            "preflight");
    }
    const std::uint64_t panel_capacity = std::min(
        census.config().reciprocal_block, accepted_count);
    if (panel_capacity == 0U) {
        throw std::logic_error(
            "reciprocal metric source or reciprocal panel is empty");
    }

    const std::uint64_t metric_elements = checked_multiply(
        n_auxiliary, n_auxiliary, "raw auxiliary metric elements");
    const PeriodicCorrelationByteCount metric_bytes = checked_multiply(
        metric_elements,
        static_cast<std::uint64_t>(sizeof(std::complex<double>)),
        "raw auxiliary metric bytes");
    // Reject payloads outside SHA-256's message domain before attempting any
    // matrix or panel allocation, even when a native caller supplies a very
    // large admitted node budget.
    const std::uint64_t expected_payload_wire_bytes =
        periodic_correlation_reciprocal_metric_payload_wire_bytes(
            n_auxiliary);
    const std::uint64_t reciprocal_panel_elements = checked_multiply(
        5U, panel_capacity, "reciprocal panel elements");
    const PeriodicCorrelationByteCount reciprocal_panel_bytes =
        checked_multiply(
            panel_capacity,
            kPeriodicCorrelationFactorBuildReciprocalVectorRecordBytes,
            "reciprocal panel bytes");
    const std::uint64_t fourier_panel_elements = checked_multiply(
        n_auxiliary, panel_capacity, "auxiliary Fourier panel elements");
    const PeriodicCorrelationByteCount one_fourier_panel_bytes =
        checked_multiply(
            fourier_panel_elements,
            static_cast<std::uint64_t>(sizeof(std::complex<double>)),
            "auxiliary Fourier panel bytes");

    const auto admitted_full_reciprocal_panel_bytes = checked_multiply(
        census.config().reciprocal_block,
        kPeriodicCorrelationFactorBuildReciprocalVectorRecordBytes,
        "admitted reciprocal panel bytes");
    const auto admitted_full_fourier_double_panel_bytes = checked_multiply(
        checked_multiply(
            n_auxiliary,
            census.config().reciprocal_block,
            "admitted auxiliary Fourier panel elements"),
        2U * static_cast<std::uint64_t>(sizeof(std::complex<double>)),
        "admitted double auxiliary Fourier panel bytes");
    if (admitted_components.metric_matrix_bytes != metric_bytes
        || admitted_components.reciprocal_g_panel_bytes
            != admitted_full_reciprocal_panel_bytes
        || admitted_components.auxiliary_fourier_double_panel_bytes
            != admitted_full_fourier_double_panel_bytes) {
        throw std::logic_error(
            "factor-build admission does not describe the reciprocal "
            "metric allocations");
    }
    require_size_t_extent(metric_elements, "matrix element count");
    require_size_t_extent(
        reciprocal_panel_elements, "reciprocal panel element count");
    require_size_t_extent(
        fourier_panel_elements, "Fourier panel element count");
    if (metric_elements
        > static_cast<std::uint64_t>(
            std::vector<std::complex<double>>().max_size())
        || fourier_panel_elements
            > static_cast<std::uint64_t>(
                std::vector<std::complex<double>>().max_size())
        || reciprocal_panel_elements
            > static_cast<std::uint64_t>(
                std::vector<double>().max_size())) {
        throw std::length_error(
            "reciprocal metric allocation exceeds vector max_size");
    }

    // Count-zero evaluation performs the complete basis-content and shell-
    // convention validation without allocating output or evaluating a
    // Gaussian.  Unsupported Cartesian/high-L auxiliaries therefore fail
    // before the admitted matrix and work panels are materialized.
    const AuxiliaryFourierPanel basis_preflight =
        auxiliary_gaussian_fourier_panel(
            auxiliary_basis, AuxiliaryFourierVectorView{}, 1U);
    if (basis_preflight.n_auxiliary
            != static_cast<std::size_t>(n_auxiliary)
        || basis_preflight.n_vectors != 0U
        || basis_preflight.output_bytes != 0U
        || !basis_preflight.data.empty()) {
        throw std::logic_error(
            "auxiliary Fourier basis preflight allocated or changed "
            "extent");
    }

    const RegularKMesh addressing(
        reference.dimensions().mesh, reference.dimensions().is_shift);
    const auto q_address = addressing.transfer_address(
        static_cast<std::size_t>(source.q_index()));
    const std::uint64_t conjugate_q_index = static_cast<std::uint64_t>(
        addressing.transfer_index(addressing.negate(q_address)));
    const bool self_conjugate = conjugate_q_index == source.q_index();
    if (self_conjugate) {
        require_self_conjugate_source_closure(geometry, source);
    }

    auto numerical = detail::contract_reciprocal_metric_numeric(
        auxiliary_basis, n_auxiliary, accepted_count, panel_capacity, self_conjugate,
        checked_add(checked_add(metric_bytes, reciprocal_panel_bytes, "metric numerical peak"),
            checked_multiply(2U, one_fourier_panel_bytes, "metric double Fourier"), "metric numerical peak"),
        [](detail::PeriodicReciprocalRecordCallback callback, void* user, const void* source_user) {
            return detail::visit_periodic_reciprocal_numeric_source(
                *static_cast<const ReciprocalSourceGeometry*>(source_user), callback, user);
        }, &geometry);
    auto& metric = numerical.matrix;
    const auto completed_panels = numerical.completed_panel_count;
    const auto maximum_fourier_panel_bytes = numerical.maximum_fourier_panel_bytes;
    const auto maximum_self_conjugate_imaginary_residual =
        numerical.maximum_self_conjugate_imaginary_residual;

    CanonicalSourceHasher payload(
        kPayloadIdentityDomain,
        kPeriodicCorrelationReciprocalMetricResultContractVersion);
    payload.add_string(
        sha256_ascii_string(verified_source_identity_ascii));
    payload.add_u64(source.q_index());
    payload.add_u64(n_auxiliary);
    payload.add_u64(metric_elements);
    payload.add_u64(metric_bytes);
    if (payload.wire_bytes()
        != kPeriodicCorrelationReciprocalMetricPayloadFixedPrefixWireBytes) {
        throw std::logic_error(
            "reciprocal metric payload prefix changed without a "
            "contract-version update");
    }
    for (const auto& value : metric) {
        payload.add_double(value.real());
        payload.add_double(value.imag());
    }
    if (payload.wire_bytes() != expected_payload_wire_bytes) {
        throw std::logic_error(
            "reciprocal metric payload extent changed without a "
            "contract-version update");
    }
    const std::string payload_identity = payload.finish_hex();

    return PeriodicCorrelationReciprocalMetricResult(
        reference.state_handle(),
        verified_source_identity_ascii,
        copy_sha256_ascii(
            census.census_identity_sha256(), "factor-build census identity"),
        plan_identity_ascii,
        copy_sha256_ascii(
            payload_identity, "reciprocal metric payload identity"),
        copy_sha256_ascii(
            census.config().auxiliary_basis_identity_sha256,
            "auxiliary basis identity"),
        source.q_index(),
        conjugate_q_index,
        n_auxiliary,
        accepted_count,
        panel_capacity,
        completed_panels,
        metric_bytes,
        reciprocal_panel_bytes,
        maximum_fourier_panel_bytes,
        one_fourier_panel_bytes,
        admitted_metric_peak,
        self_conjugate,
        maximum_self_conjugate_imaginary_residual,
        std::move(metric));
}

static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationReciprocalMetricSourceManifest>::value,
    "a reciprocal metric source manifest must never copy its state");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationReciprocalMetricSourceManifest>::value,
    "a reciprocal metric source manifest must be cheaply movable");
static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationReciprocalMetricResult>::value,
    "a reciprocal metric result must never copy its retained state");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationReciprocalMetricResult>::value,
    "a reciprocal metric result must be cheaply movable");

std::uint64_t visit_periodic_correlation_reciprocal_metric_source(
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    void (*callback)(const std::array<double, 5>&, void*), void* context) {
    if (!source.state_handle() || callback == nullptr) {
        throw std::invalid_argument("reciprocal metric source replay requires a live source and callback");
    }
    const auto count = for_each_accepted_vector(
        source_geometry(source), [&](const AcceptedReciprocalVector& record) {
            callback(record.lanes, context);
        });
    if (count != source.accepted_vector_count()) {
        throw std::logic_error("reciprocal metric replay changed the sealed source count");
    }
    return count;
}

std::uint64_t require_periodic_correlation_reciprocal_source_conjugacy(
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    const PeriodicCorrelationReciprocalMetricSourceManifest& conjugate_source,
    std::uint64_t maximum_candidates_per_source) {
    if (!source.state_handle() || !conjugate_source.state_handle()
        || source.state_handle() != conjugate_source.state_handle()
        || source.mesh() != conjugate_source.mesh()
        || source.is_shift() != conjugate_source.is_shift()
        || source.reciprocal_energy_cutoff() != conjugate_source.reciprocal_energy_cutoff()
        || source.cell_volume_bohr3() != conjugate_source.cell_volume_bohr3()
        || !exactly_equal(source.reciprocal_lattice(), conjugate_source.reciprocal_lattice())) {
        throw std::invalid_argument("reciprocal source conjugacy requires matching live state owners and source geometry");
    }
    if (maximum_candidates_per_source == 0U
        || source.candidate_count() > maximum_candidates_per_source
        || conjugate_source.candidate_count() > maximum_candidates_per_source) {
        throw std::length_error("reciprocal source conjugacy exceeds the explicit candidate cap");
    }
    const RegularKMesh addressing(source.mesh());
    if (addressing.negate_index(source.q_index()) != conjugate_source.q_index()) {
        throw std::invalid_argument("reciprocal source conjugacy requires opposite q addresses");
    }
    if (source.accepted_vector_count() != conjugate_source.accepted_vector_count()
        || source.zero_mode_excluded_count() != conjugate_source.zero_mode_excluded_count()) {
        throw std::runtime_error("reciprocal source conjugacy has unequal accepted-vector counts");
    }
    std::array<std::int64_t, 3> partner_shift{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const auto numerator = static_cast<std::int64_t>(source.centered_doubled_numerator()[axis])
            + conjugate_source.centered_doubled_numerator()[axis];
        const auto denominator = 2 * static_cast<std::int64_t>(source.mesh()[axis]);
        if (numerator % denominator != 0 || (numerator / denominator != 0 && numerator / denominator != -1)) {
            throw std::logic_error("reciprocal source conjugacy has inconsistent centered integer addresses");
        }
        partner_shift[axis] = -numerator / denominator;
    }
    const auto geometry = source_geometry(source);
    const auto opposite = source_geometry(conjugate_source);
    return detail::require_periodic_reciprocal_numeric_conjugacy(
        geometry, opposite, partner_shift, source.accepted_vector_count());
}

std::uint64_t detail::require_periodic_reciprocal_numeric_conjugacy(
    const PeriodicReciprocalNumericGeometry& geometry,
    const PeriodicReciprocalNumericGeometry& opposite,
    const std::array<std::int64_t, 3>& partner_shift,
    std::uint64_t expected_accepted_count) {
    const double membership_limit = opposite.exact_gamma
        ? opposite.cutoff_squared : opposite.radial_limit_squared;
    const auto count = for_each_accepted_vector(geometry,
        [&](const AcceptedReciprocalVector& record) {
            Eigen::Vector3d shifted;
            std::array<std::int64_t, 3> label{};
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                if (record.integer_label[axis] == std::numeric_limits<std::int64_t>::min()) {
                    throw std::overflow_error("reciprocal source conjugate label overflows");
                }
                auto value = -record.integer_label[axis];
                if (partner_shift[axis] != 0) {
                    if (value == std::numeric_limits<std::int64_t>::max()) {
                        throw std::overflow_error("reciprocal source conjugate label overflows");
                    }
                    ++value;
                }
                const auto exact_limit = static_cast<std::int64_t>(kMaximumExactBinary64Integer);
                if (value < -exact_limit || value > exact_limit) {
                    throw std::overflow_error("reciprocal source conjugate label is not exactly representable");
                }
                if (value < opposite.lower_bounds[axis] || value > opposite.upper_bounds[axis]) {
                    throw std::runtime_error("reciprocal source conjugate label is outside the opposite source bounds");
                }
                label[axis] = value;
                shifted[axis] = std::fma(1.0, static_cast<double>(value), opposite.q_fractional[axis]);
            }
            if (opposite.exact_gamma && label == std::array<std::int64_t, 3>{0, 0, 0}) {
                throw std::runtime_error("reciprocal source conjugacy maps a retained vector to the excluded zero mode");
            }
            const auto p = fixed_binary64_matvec(opposite.reciprocal_lattice, shifted);
            const auto p2 = fixed_binary64_squared_norm(p);
            if (!p.allFinite() || !std::isfinite(p2) || p2 <= 0.0 || p2 > membership_limit) {
                throw std::runtime_error("reciprocal source conjugate vector is not accepted by the opposite predicate");
            }
            const double weight = (kFourPi / opposite.cell_volume) / p2;
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                if (normalized_binary64_bits(p[axis]) != normalized_binary64_bits(-record.lanes[axis])) {
                    throw std::runtime_error("reciprocal source conjugacy lacks bitwise opposite momentum");
                }
            }
            if (!std::isfinite(weight) || weight <= 0.0
                || normalized_binary64_bits(p2) != normalized_binary64_bits(record.lanes[3])
                || normalized_binary64_bits(weight) != normalized_binary64_bits(record.lanes[4])) {
                throw std::runtime_error("reciprocal source conjugacy changes the sealed norm or Coulomb weight");
            }
        });
    if (count != expected_accepted_count) {
        throw std::logic_error("reciprocal source conjugacy replay changed the sealed count");
    }
    return count;
}

}  // namespace vibeqc
