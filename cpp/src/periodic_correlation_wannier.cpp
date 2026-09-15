#include "vibeqc/periodic_correlation_wannier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"
#include "vibeqc/periodic_correlation_diabatic_seed.hpp"
#include "vibeqc/periodic_correlation_iao_optimizer.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;
constexpr char kGaugeDomain[] = "vibeqc.periodic.correlation.wannier.gauge";
constexpr char kCoefficientDomain[] = "vibeqc.periodic.correlation.wannier.coefficients";
constexpr char kIdentityDomain[] = "vibeqc.periodic.correlation.wannier.identity";

std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("Wannier count or byte product overflows uint64");
    }
    return left * right;
}

std::uint64_t sum(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("Wannier memory inventory overflows uint64");
    }
    return left + right;
}

std::uint64_t required_node_memory(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationWannierMemoryPlan& memory,
    std::uint64_t extra_live_bytes = 0U) {
    const auto& dims = reference.dimensions();
    const auto& budget = reference.budget();
    const auto replicas = product(budget.mpi_ranks, budget.workers_per_rank);
    return sum(sum(dims.external_bytes, dims.shared_bytes),
        sum(product(budget.mpi_ranks,
                    sum(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            product(replicas, sum(sum(memory.peak_owned_numerical_bytes,
                                     memory.caller_gauge_bytes), extra_live_bytes))));
}

bool finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

// Fixed wire: length-prefixed ASCII domain, u32 version, then documented
// integer and complex lanes. Big-endian integers/binary64; signed zeros
// canonicalize to +0. The count-only planner reserves 4096 prefix bytes in
// SHA-256's 64-bit *bit* length domain, independently of native size_t.
class WannierDigest {
public:
    explicit WannierDigest(const char* domain) {
        string(domain);
        u32(kPeriodicCorrelationWannierContractVersion);
    }
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (unsigned i = 0; i < 4U; ++i) bytes[i] = value >> (24U - 8U * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned i = 0; i < 8U; ++i) bytes[i] = value >> (56U - 8U * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void real(double value) {
        if (!std::isfinite(value)) {
            throw std::overflow_error("Wannier content digest encountered a non-finite value");
        }
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void complex(Complex value) { real(value.real()); real(value.imag()); }
    void string(const std::string& value) {
        u64(value.size());
        hash_.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
};

struct ComplexSum {
    double real = 0.0, imag = 0.0, real_c = 0.0, imag_c = 0.0;
    static void add_lane(double term, double& value, double& correction) {
        const double next = value + term;
        correction += std::abs(value) >= std::abs(term)
            ? (value - next) + term : (term - next) + value;
        value = next;
    }
    void add(Complex term) {
        add_lane(term.real(), real, real_c);
        add_lane(term.imag(), imag, imag_c);
    }
    Complex value() const {
        const Complex result(real + real_c, imag + imag_c);
        if (!finite(result)) {
            throw std::overflow_error("Wannier compensated contraction is non-finite");
        }
        return result;
    }
};

void require_options(const PeriodicCorrelationWannierOptions& options) {
    const std::array<double, 5> controls{
        options.gauge_unitarity_tolerance,
        options.time_reversal_absolute_tolerance,
        options.time_reversal_relative_tolerance,
        options.real_absolute_tolerance,
        options.real_relative_tolerance};
    for (double value : controls) {
        if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
            throw std::invalid_argument("Wannier tolerances must be finite and in [0,1)");
        }
    }
    if (options.gauge_unitarity_tolerance == 0.0) {
        throw std::invalid_argument("Wannier gauge unitarity tolerance must be positive");
    }
}

void require_reference(const PeriodicCorrelationAdmittedReference& reference) {
    if (!reference.state_handle()) {
        throw std::invalid_argument("Wannier transform requires a live admitted state");
    }
    const auto& state = reference.state();
    const auto& dims = reference.dimensions();
    const auto& plan = reference.plan();
    if (state.is_shift() != std::array<int, 3>{0, 0, 0}) {
        throw std::invalid_argument("Wannier contract v1 requires the exact Gamma-centered mesh");
    }
    if (reference.contract_version() != kPeriodicCorrelationAdmittedReferenceContractVersion
        || state.contract_version() != kPeriodicRestrictedMeanFieldStateContractVersion
        || state.normalization() != PeriodicMeanFieldNormalizationConvention::
            UnnormalizedAoBlochSumsUniformFullBzWeights
        || state.reference_kind() != PeriodicMeanFieldReferenceKind::RestrictedHartreeFock
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || dims.symmetry_reduction_requested || !dims.symmetry_mapping_identity.empty()
        || dims.symmetry_representative_count != dims.n_kpoints
        || dims.symmetry_weight_sum != dims.n_kpoints
        || plan.allocation_contract_version != dims.allocation_contract_version
        || plan.calculation_identity != dims.calculation_identity
        || plan.allocation_identity != dims.allocation_identity
        || state.calculation_identity() != dims.calculation_identity
        || state.mesh() != dims.mesh || state.is_shift() != dims.is_shift
        || state.n_kpoints() != dims.n_kpoints || state.n_basis() != dims.n_basis
        || state.n_effective_orbitals() != dims.n_effective_orbitals
        || state.n_correlated_occupied() != dims.n_home_occupied
        || state.n_frozen_core() + state.n_correlated_occupied()
            != dims.n_home_total_occupied) {
        throw std::logic_error("Wannier transform received inconsistent admitted-reference seals");
    }
}

void require_mask_shapes(const PeriodicRestrictedMeanFieldState& state) {
    for (std::size_t k = 0; k < state.n_kpoints(); ++k) {
        const auto& active = state.correlated_occupied_mask(k);
        const auto& frozen = state.frozen_core_mask(k);
        const auto& virtuals = state.virtual_mask(k);
        if (active.size() != state.n_effective_orbitals()
            || frozen.size() != active.size() || virtuals.size() != active.size()) {
            throw std::logic_error("Wannier occupied mask shape is inconsistent");
        }
        std::uint64_t count = 0, frozen_count = 0;
        for (std::size_t band = 0; band < active.size(); ++band) {
            if (active[band] > 1U || frozen[band] > 1U || virtuals[band] > 1U
                || active[band] + frozen[band] + virtuals[band] != 1U) {
                throw std::logic_error("Wannier occupied masks are not a disjoint partition");
            }
            count += active[band];
            frozen_count += frozen[band];
        }
        if (count != state.n_correlated_occupied() || frozen_count != state.n_frozen_core()) {
            throw std::logic_error("Wannier occupied mask counts vary over the mesh");
        }
    }
}

bool compare_tr(Complex left, Complex right, double& maximum,
                const PeriodicCorrelationWannierOptions& options) {
    const double residual = std::abs(left - std::conj(right));
    const double scale = std::max(std::abs(left), std::abs(right));
    if (!std::isfinite(residual) || !std::isfinite(scale)) {
        throw std::overflow_error("Wannier time-reversal diagnostic overflowed");
    }
    maximum = std::max(maximum, residual);
    return residual <= options.time_reversal_absolute_tolerance
        + options.time_reversal_relative_tolerance * scale;
}

// Character of the exact integer torus, evaluated without an Nk^2 phase
// table. Evaluate one representative of k/-k and conjugate the other;
// self-inverse characters are integer parities, hence exactly +/-1.
Complex character(const RegularKMesh& mesh, std::size_t k, std::size_t cell) {
    const auto minus_k = mesh.negate_index(k);
    const auto canonical_k = std::min(k, minus_k);
    const auto m = mesh.address(canonical_k).doubled;
    const auto r = mesh.address(cell).doubled;
    if (k == minus_k) {
        std::int64_t parity = 0;
        for (std::size_t d = 0; d < 3U; ++d) {
            if (m[d] != 0) parity += r[d] / 2;
        }
        return Complex(parity % 2 == 0 ? 1.0 : -1.0, 0.0);
    }
    double turns = 0.0;
    for (std::size_t d = 0; d < 3U; ++d) {
        const auto modulus = static_cast<std::int64_t>(mesh.mesh()[d]);
        const auto residue = ((static_cast<std::int64_t>(m[d]) / 2)
            * (static_cast<std::int64_t>(r[d]) / 2)) % modulus;
        turns += static_cast<double>(residue) / static_cast<double>(modulus);
    }
    turns -= std::round(turns);
    const Complex phase(std::cos(kTwoPi * turns), std::sin(kTwoPi * turns));
    return k == canonical_k ? phase : std::conj(phase);
}

}  // namespace

PeriodicCorrelationWannierMemoryPlan plan_periodic_correlation_wannier(
    std::array<int, 3> mesh, std::uint64_t n_basis,
    std::uint64_t n_home_occupied) {
    const RegularKMesh addressing(mesh);
    if (n_home_occupied == 0U || n_basis == 0U || n_home_occupied > n_basis) {
        throw std::invalid_argument("Wannier dimensions require 0 < n_active <= n_basis");
    }
    // The consumer is an admitted correlation state; candidate-count overflow
    // must fail here too, without creating a state or any orbital array.
    (void) estimate_periodic_correlation_translation_pair_counts(mesh, n_home_occupied);
    PeriodicCorrelationWannierMemoryPlan result;
    result.n_cells = addressing.size();
    result.n_basis = n_basis;
    result.n_home_occupied = n_home_occupied;
    result.coefficient_count = product(product(result.n_cells, n_basis), n_home_occupied);
    result.gauge_element_count = product(product(result.n_cells, n_home_occupied), n_home_occupied);
    result.caller_gauge_bytes = product(result.gauge_element_count, sizeof(Complex));
    result.retained_coefficient_bytes = product(result.coefficient_count, sizeof(Complex));
    result.temporary_gauged_coefficient_bytes = result.retained_coefficient_bytes;
    result.peak_owned_numerical_bytes = product(result.retained_coefficient_bytes, 2U);
    constexpr auto maximum_hash_payload = std::numeric_limits<std::uint64_t>::max() / 8U - 4096U;
    if (result.retained_coefficient_bytes > maximum_hash_payload
        || result.caller_gauge_bytes > maximum_hash_payload) {
        throw std::overflow_error("Wannier payload exceeds the SHA-256 bit-length domain");
    }
    if (result.coefficient_count > std::vector<Complex>().max_size()
        || result.gauge_element_count > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
        throw std::length_error("Wannier payload exceeds native address or vector max_size");
    }
    return result;
}

const Complex* PeriodicCorrelationWannier::cell_coefficients(std::size_t cell) const {
    if (cell >= memory_.n_cells || coefficients_.size() != memory_.coefficient_count) {
        throw std::out_of_range("Wannier cell index is out of range or owner is consumed");
    }
    return coefficients_.data() + cell * memory_.n_basis * memory_.n_home_occupied;
}

Complex PeriodicCorrelationWannier::coefficient(
    std::size_t cell, std::size_t ao, std::size_t occupied) const {
    if (ao >= memory_.n_basis || occupied >= memory_.n_home_occupied) {
        throw std::out_of_range("Wannier AO or occupied index is out of range");
    }
    return cell_coefficients(cell)[ao * memory_.n_home_occupied + occupied];
}

Complex PeriodicCorrelationWannier::translated_coefficient(
    std::size_t cell, std::size_t orbital_cell, std::size_t ao,
    std::size_t occupied) const {
    if (cell >= memory_.n_cells || orbital_cell >= memory_.n_cells
        || !state_) {
        throw std::out_of_range("Wannier translated cell index is out of range");
    }
    const RegularKMesh addressing(state_->mesh());
    // transfer_index(a,b) is b-a on the same integer finite group. These
    // indices name REAL-space cells here, not momentum transfer values.
    return coefficient(addressing.transfer_index(orbital_cell, cell), ao, occupied);
}

PeriodicCorrelationWannier make_periodic_correlation_wannier(
    const PeriodicCorrelationAdmittedReference& reference,
    const Complex* gauges, std::size_t gauge_element_count,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationWannierOptions& options) {
    require_reference(reference);
    require_options(options);
    const auto& state = reference.state();
    const auto memory = plan_periodic_correlation_wannier(
        state.mesh(), state.n_basis(), state.n_correlated_occupied());
    if (owned_numerical_byte_cap == 0U
        || memory.peak_owned_numerical_bytes > owned_numerical_byte_cap) {
        throw std::length_error("Wannier numerical byte cap is missing or exceeded");
    }
    if (gauges == nullptr || gauge_element_count != memory.gauge_element_count
        || reinterpret_cast<std::uintptr_t>(gauges) % alignof(Complex) != 0U) {
        throw std::invalid_argument("Wannier gauge view has an invalid shape, pointer or alignment");
    }
    if (memory.caller_gauge_bytes > std::numeric_limits<std::uintptr_t>::max()
            - reinterpret_cast<std::uintptr_t>(gauges)) {
        throw std::overflow_error("Wannier gauge view exceeds the native pointer address range");
    }
    const auto& dims = reference.dimensions();
    const auto& budget = reference.budget();
    const auto required_node_bytes = required_node_memory(reference, memory);
    if (budget.memory_limit_bytes == 0U || required_node_bytes > budget.memory_limit_bytes) {
        throw std::length_error("Wannier peak plus caller inventory exceeds admitted node memory");
    }
    require_mask_shapes(state);
    const auto nk = static_cast<std::size_t>(memory.n_cells);
    const auto nao = static_cast<std::size_t>(memory.n_basis);
    const auto nocc = static_cast<std::size_t>(memory.n_home_occupied);
    const RegularKMesh addressing(state.mesh());
    PeriodicCorrelationWannierDiagnostics diagnostics;
    diagnostics.required_node_memory_bytes = required_node_bytes;
    // Validate all user-controlled lanes and both unitary products BEFORE
    // allocating either size-dependent payload. No Eigen expression/BLAS.
    for (std::size_t index = 0; index < gauge_element_count; ++index) {
        if (!finite(gauges[index])) {
            throw std::invalid_argument("Wannier gauge contains a non-finite value");
        }
    }
    for (std::size_t k = 0; k < nk; ++k) {
        const auto* u = gauges + k * nocc * nocc;
        for (std::size_t i = 0; i < nocc; ++i) {
            for (std::size_t j = 0; j < nocc; ++j) {
                ComplexSum left, right;
                for (std::size_t l = 0; l < nocc; ++l) {
                    left.add(std::conj(u[l * nocc + i]) * u[l * nocc + j]);
                    right.add(u[i * nocc + l] * std::conj(u[j * nocc + l]));
                }
                const double target = i == j ? 1.0 : 0.0;
                const double left_error = std::abs(left.value() - target);
                const double right_error = std::abs(right.value() - target);
                diagnostics.maximum_left_unitarity_residual = std::max(
                    diagnostics.maximum_left_unitarity_residual, left_error);
                diagnostics.maximum_right_unitarity_residual = std::max(
                    diagnostics.maximum_right_unitarity_residual, right_error);
                if (left_error > options.gauge_unitarity_tolerance
                    || right_error > options.gauge_unitarity_tolerance) {
                    throw std::invalid_argument("Wannier active occupied gauge is not unitary");
                }
            }
        }
        const auto minus_k = addressing.negate_index(k);
        for (std::size_t mu = 0; mu < nao; ++mu) {
            for (std::size_t nu = 0; nu < nao; ++nu) {
                const bool s_ok = compare_tr(state.overlap(k)(mu, nu),
                    state.overlap(minus_k)(mu, nu),
                    diagnostics.maximum_overlap_time_reversal_residual, options);
                const bool f_ok = compare_tr(state.fock(k)(mu, nu),
                    state.fock(minus_k)(mu, nu),
                    diagnostics.maximum_fock_time_reversal_residual, options);
                diagnostics.time_reversal_compatible &= s_ok && f_ok;
            }
        }
    }
    if (options.require_time_reversal && !diagnostics.time_reversal_compatible) {
        throw std::invalid_argument("Wannier reference overlap/Fock failed time reversal");
    }
    WannierDigest gauge_digest(kGaugeDomain);
    gauge_digest.u64(nk);
    gauge_digest.u64(nocc);
    for (std::size_t index = 0; index < gauge_element_count; ++index) {
        gauge_digest.complex(gauges[index]);
    }

    // The only temporary numerical buffer is one home-sized C_active U.
    std::vector<Complex> gauged(static_cast<std::size_t>(memory.coefficient_count));
    for (std::size_t k = 0; k < nk; ++k) {
        const auto& mask = state.correlated_occupied_mask(k);
        const auto& c = state.coefficients(k);
        const auto* u = gauges + k * nocc * nocc;
        for (std::size_t mu = 0; mu < nao; ++mu) {
            for (std::size_t i = 0; i < nocc; ++i) {
                ComplexSum contraction;
                std::size_t active_row = 0;
                for (std::size_t band = 0; band < mask.size(); ++band) {
                    if (mask[band] != 0U) {
                        contraction.add(c(mu, band) * u[active_row * nocc + i]);
                        ++active_row;
                    }
                }
                gauged[(k * nao + mu) * nocc + i] = contraction.value();
            }
        }
    }
    for (std::size_t k = 0; k < nk; ++k) {
        const auto minus_k = addressing.negate_index(k);
        for (std::size_t index = 0; index < nao * nocc; ++index) {
            const bool ok = compare_tr(gauged[k * nao * nocc + index],
                gauged[minus_k * nao * nocc + index],
                diagnostics.maximum_coefficient_time_reversal_residual, options);
            diagnostics.time_reversal_compatible &= ok;
        }
    }
    if (options.require_time_reversal && !diagnostics.time_reversal_compatible) {
        throw std::invalid_argument("Wannier gauged occupied coefficients failed time reversal");
    }

    PeriodicCorrelationWannier result;
    result.memory_ = memory;
    result.options_ = options;
    result.coefficients_.resize(static_cast<std::size_t>(memory.coefficient_count));
    const double inverse_nk = 1.0 / static_cast<double>(nk);
    for (std::size_t cell = 0; cell < nk; ++cell) {
        for (std::size_t mu = 0; mu < nao; ++mu) {
            for (std::size_t i = 0; i < nocc; ++i) {
                ComplexSum transform;
                for (std::size_t k = 0; k < nk; ++k) {
                    // Weight each term, avoiding an otherwise unnecessary
                    // Nk-fold intermediate growth for high-magnitude C.
                    transform.add(character(addressing, k, cell)
                        * (gauged[(k * nao + mu) * nocc + i] * inverse_nk));
                }
                const auto value = transform.value();
                const double magnitude = std::abs(value);
                if (!std::isfinite(magnitude)) {
                    throw std::overflow_error("Wannier coefficient magnitude overflowed");
                }
                const double imaginary = std::abs(value.imag());
                diagnostics.maximum_home_imaginary_magnitude = std::max(
                    diagnostics.maximum_home_imaginary_magnitude, imaginary);
                diagnostics.maximum_home_coefficient_magnitude = std::max(
                    diagnostics.maximum_home_coefficient_magnitude, magnitude);
                diagnostics.real_home_coefficients_compatible &= imaginary
                    <= options.real_absolute_tolerance + options.real_relative_tolerance * magnitude;
                result.coefficients_[(cell * nao + mu) * nocc + i] = value;
            }
        }
    }
    if (options.require_real_home_coefficients
        && !diagnostics.real_home_coefficients_compatible) {
        throw std::invalid_argument("Wannier home coefficients failed the real-gauge tolerance");
    }
    result.state_ = reference.state_handle();
    result.allocation_identity_ = dims.allocation_identity;
    result.diagnostics_ = diagnostics;
    result.gauge_digest_ = gauge_digest.finish();
    WannierDigest coefficient_digest(kCoefficientDomain);
    for (int component : state.mesh()) coefficient_digest.u32(component);
    coefficient_digest.u64(nao);
    coefficient_digest.u64(nocc);
    for (const auto value : result.coefficients_) coefficient_digest.complex(value);
    result.coefficient_digest_ = coefficient_digest.finish();
    WannierDigest identity(kIdentityDomain);
    identity.u32(state.digest_version());
    identity.string(state.state_identity_sha256());
    identity.string(dims.calculation_identity);
    identity.string(dims.allocation_identity);
    identity.string(result.gauge_digest_);
    identity.string(result.coefficient_digest_);
    identity.string("inverse-bloch-home-coefficients-1/Nk;active-mask-ascending;exact-gamma");
    identity.real(options.gauge_unitarity_tolerance);
    identity.real(options.time_reversal_absolute_tolerance);
    identity.real(options.time_reversal_relative_tolerance);
    identity.real(options.real_absolute_tolerance);
    identity.real(options.real_relative_tolerance);
    identity.u32(options.require_time_reversal ? 1U : 0U);
    identity.u32(options.require_real_home_coefficients ? 1U : 0U);
    result.identity_digest_ = identity.finish();
    return result;
}

PeriodicCorrelationWannier make_periodic_correlation_wannier_from_diabatic_seed(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationDiabaticSeed& seed,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationWannierOptions& options) {
    require_reference(reference);
    if (!seed.state_handle() || seed.state_handle().get() != reference.state_handle().get()
        || seed.allocation_identity() != reference.dimensions().allocation_identity) {
        throw std::invalid_argument("Wannier seed requires the exact admitted state allocation");
    }
    const auto& state = reference.state();
    const auto memory = plan_periodic_correlation_wannier(
        state.mesh(), state.n_basis(), state.n_correlated_occupied());
    const auto& seed_memory = seed.memory();
    if (seed_memory.n_cells != memory.n_cells || seed_memory.n_basis != memory.n_basis
        || seed_memory.n_active != memory.n_home_occupied
        || seed_memory.gauge_count != memory.gauge_element_count
        || seed_memory.retained_gauge_bytes != memory.caller_gauge_bytes) {
        throw std::invalid_argument("Wannier seed gauge dimensions do not match the reference");
    }
    // This gate precedes gauge reads and transform allocation. The raw-view
    // factory already charges the gauge payload, but cannot know that the
    // seed's active-band and pivot arrays also remain live throughout the call.
    const auto required = required_node_memory(reference, memory, seed_memory.retained_index_bytes);
    if (reference.budget().memory_limit_bytes == 0U
        || required > reference.budget().memory_limit_bytes) {
        throw std::length_error("Wannier peak plus live diabatic seed exceeds admitted node memory");
    }
    auto result = make_periodic_correlation_wannier(reference, seed.gauges_data(),
        static_cast<std::size_t>(seed_memory.gauge_count), owned_numerical_byte_cap, options);
    result.diagnostics_.required_node_memory_bytes = required;
    result.diagnostics_.live_diabatic_seed_index_bytes = seed_memory.retained_index_bytes;
    return result;
}

PeriodicCorrelationWannier make_periodic_correlation_wannier_from_iao_optimizer(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationIAOOptimizerResult& optimizer,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationWannierOptions& options) {
    require_reference(reference);
    if (!optimizer.state_handle() || optimizer.state_handle().get() != reference.state_handle().get()
        || optimizer.allocation_identity() != reference.dimensions().allocation_identity
        || optimizer.contract_version() != kPeriodicCorrelationIAOOptimizerContractVersion) {
        throw std::invalid_argument("Wannier optimizer requires the exact admitted state allocation");
    }
    if (!optimizer.converged())
        throw std::invalid_argument("Wannier optimizer bridge requires a converged native localization");
    if (!options.require_time_reversal || !options.require_real_home_coefficients)
        throw std::invalid_argument("Wannier optimizer bridge requires time-reversal and real-home audits");
    const auto& state = reference.state();
    const auto memory = plan_periodic_correlation_wannier(state.mesh(), state.n_basis(), state.n_correlated_occupied());
    const auto& om = optimizer.memory();
    const auto index_bytes = product(8, product(memory.n_cells, memory.n_home_occupied));
    if (om.n_points != memory.n_cells || om.n_basis != memory.n_basis || om.n_active != memory.n_home_occupied
        || om.output_numerical_bytes != sum(memory.caller_gauge_bytes, index_bytes)) {
        throw std::invalid_argument("Wannier optimizer output dimensions differ from the reference");
    }
    if (!owned_numerical_byte_cap || owned_numerical_byte_cap < memory.peak_owned_numerical_bytes)
        throw std::length_error("Wannier optimizer bridge owned byte cap is missing or exceeded");
    const auto required = required_node_memory(reference, memory, index_bytes);
    if (!reference.budget().memory_limit_bytes || required > reference.budget().memory_limit_bytes)
        throw std::length_error("Wannier peak plus live optimizer exceeds admitted node memory");
    for (std::size_t k = 0; k < memory.n_cells; ++k) {
        std::size_t active = 0;
        for (std::size_t band = 0; band < state.n_effective_orbitals(); ++band) {
            if (!state.correlated_occupied_mask(k)[band]) continue;
            if (active >= memory.n_home_occupied || optimizer.active_band(k, active) != band)
                throw std::invalid_argument("Wannier optimizer active-band order differs from the reference");
            ++active;
        }
        if (active != memory.n_home_occupied)
            throw std::invalid_argument("Wannier optimizer active-band count differs from the reference");
    }
    auto result = make_periodic_correlation_wannier(reference, optimizer.gauges_data(),
        static_cast<std::size_t>(memory.gauge_element_count), owned_numerical_byte_cap, options);
    result.diagnostics_.required_node_memory_bytes = required;
    result.diagnostics_.live_optimizer_index_bytes = index_bytes;
    result.localization_identity_ = optimizer.optimizer_identity_sha256();
    return result;
}

}  // namespace vibeqc
