#include "vibeqc/periodic_correlation_occupied_fock.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
constexpr double kOccupiedFockTwoPi = 6.283185307179586476925286766559005768;
static_assert(sizeof(double) == 8U && sizeof(Complex) == 16U,
              "occupied Fock accounting requires binary64/complex128");

std::uint64_t checked_product(std::uint64_t a, std::uint64_t b) {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("occupied Fock byte/count product overflows uint64");
    }
    return a * b;
}

std::uint64_t checked_sum(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("occupied Fock byte inventory overflows uint64");
    }
    return a + b;
}

bool is_finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

struct CompensatedComplex {
    double real = 0.0, imag = 0.0, real_c = 0.0, imag_c = 0.0;
    static void add_lane(double term, double& sum, double& correction) {
        const double next = sum + term;
        correction += std::abs(sum) >= std::abs(term)
            ? (sum - next) + term : (term - next) + sum;
        sum = next;
    }
    void add(Complex term) {
        add_lane(term.real(), real, real_c);
        add_lane(term.imag(), imag, imag_c);
    }
    Complex value() const {
        const Complex result(real + real_c, imag + imag_c);
        if (!is_finite(result)) {
            throw std::overflow_error("occupied Fock compensated contraction is non-finite");
        }
        return result;
    }
};

// Same canonical scalar wire as the state/Wannier owners: length-prefixed
// strings, big-endian integers and binary64, canonical +0 lanes. The planner
// verifies the SHA-256 bit-length domain with ample constant prefix room.
class OccupiedFockDigest {
public:
    OccupiedFockDigest(const char* domain, std::uint32_t version) {
        string(domain);
        u32(version);
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
            throw std::invalid_argument("occupied Fock digest has a non-finite lane");
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

void require_controls(const PeriodicCorrelationOccupiedFockOptions& options) {
    const std::array<double, 6> values{
        options.hermiticity_absolute_tolerance, options.hermiticity_relative_tolerance,
        options.time_reversal_absolute_tolerance, options.time_reversal_relative_tolerance,
        options.real_absolute_tolerance, options.real_relative_tolerance};
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
            throw std::invalid_argument("occupied Fock tolerances must be finite and in [0,1)");
        }
    }
    if (options.hermiticity_absolute_tolerance == 0.0
        && options.hermiticity_relative_tolerance == 0.0) {
        throw std::invalid_argument("occupied Fock requires an explicit positive Hermiticity tolerance");
    }
}

bool compare(Complex left, Complex right, double absolute, double relative,
             double& maximum) {
    const double residual = std::abs(left - right);
    const double scale = std::max(std::abs(left), std::abs(right));
    if (!std::isfinite(residual) || !std::isfinite(scale)) {
        throw std::overflow_error("occupied Fock matrix diagnostic overflowed");
    }
    maximum = std::max(maximum, residual);
    return residual <= absolute + relative * scale;
}

void require_reference(const PeriodicCorrelationAdmittedReference& reference,
                       const PeriodicCorrelationWannier& wannier) {
    if (!reference.state_handle() || !wannier.state_handle()
        || reference.state_handle() != wannier.state_handle()) {
        throw std::invalid_argument("occupied Fock requires the identical immutable Wannier state owner");
    }
    const auto& state = reference.state();
    const auto& dims = reference.dimensions();
    const auto& plan = reference.plan();
    if (reference.contract_version() != kPeriodicCorrelationAdmittedReferenceContractVersion
        || wannier.contract_version() != kPeriodicCorrelationWannierContractVersion
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || state.is_shift() != std::array<int, 3>{0, 0, 0}
        || dims.symmetry_reduction_requested
        || state.calculation_identity() != dims.calculation_identity
        || plan.calculation_identity != dims.calculation_identity
        || plan.allocation_identity != dims.allocation_identity
        || wannier.allocation_identity() != dims.allocation_identity
        || wannier.state().state_identity_sha256() != state.state_identity_sha256()
        || dims.mesh != state.mesh() || dims.is_shift != state.is_shift()
        || dims.n_kpoints != state.n_kpoints() || dims.n_basis != state.n_basis()
        || dims.n_home_occupied != state.n_correlated_occupied()
        || wannier.n_cells() != dims.n_kpoints
        || wannier.n_home_occupied() != dims.n_home_occupied) {
        throw std::invalid_argument("occupied Fock reference, gauge and allocation provenance disagree");
    }
    (void) wannier.cell_coefficients(0);  // reject a consumed numerical owner
}

Complex gauged_coefficient(const PeriodicRestrictedMeanFieldState& state,
                           std::size_t k, std::size_t ao, std::size_t i,
                           const Complex* u, std::size_t nocc) {
    const auto& mask = state.correlated_occupied_mask(k);
    const auto& c = state.coefficients(k);
    CompensatedComplex result;
    std::size_t active_row = 0;
    for (std::size_t band = 0; band < mask.size(); ++band) {
        if (mask[band] != 0U) {
            result.add(c(ao, band) * u[active_row * nocc + i]);
            ++active_row;
        }
    }
    if (active_row != nocc) {
        throw std::logic_error("occupied Fock active mask count is inconsistent");
    }
    return result.value();
}

Complex positive_character(std::size_t m, std::size_t r, std::size_t extent) {
    if (m == 0U) return Complex(1.0, 0.0);
    if (extent % 2U == 0U && m == extent / 2U) {
        return Complex(r % 2U == 0U ? 1.0 : -1.0, 0.0);
    }
    const auto canonical = std::min(m, extent - m);
    // Mesh divisions are bounded by RegularKMesh, so the product fits u64.
    const auto residue = (static_cast<std::uint64_t>(canonical) * r) % extent;
    double turns = static_cast<double>(residue) / static_cast<double>(extent);
    if (turns > 0.5) turns -= 1.0;
    const Complex phase(std::cos(kOccupiedFockTwoPi * turns),
                        std::sin(kOccupiedFockTwoPi * turns));
    return m == canonical ? phase : std::conj(phase);
}

void transform_in_place(std::vector<Complex>& blocks, std::array<int, 3> mesh,
                        std::size_t nk, std::size_t nocc) {
    // Projection scratch was freed by the caller before this allocation.
    std::vector<Complex> line(static_cast<std::size_t>(*std::max_element(mesh.begin(), mesh.end())));
    const auto square = nocc * nocc;
    std::size_t stride = 1;
    for (int axis = 2; axis >= 0; --axis) {
        const auto extent = static_cast<std::size_t>(mesh[axis]);
        const auto outer_count = nk / (extent * stride);
        const double weight = 1.0 / static_cast<double>(extent);
        if (extent != 1U) {
            for (std::size_t outer = 0; outer < outer_count; ++outer) {
                for (std::size_t inner = 0; inner < stride; ++inner) {
                    for (std::size_t ij = 0; ij < square; ++ij) {
                        for (std::size_t m = 0; m < extent; ++m) {
                            line[m] = blocks[((outer * extent + m) * stride + inner) * square + ij];
                        }
                        for (std::size_t r = 0; r < extent; ++r) {
                            CompensatedComplex value;
                            for (std::size_t m = 0; m < extent; ++m) {
                                value.add(positive_character(m, r, extent) * (line[m] * weight));
                            }
                            blocks[((outer * extent + r) * stride + inner) * square + ij] = value.value();
                        }
                    }
                }
            }
        }
        stride *= extent;
    }
}

}  // namespace

PeriodicCorrelationOccupiedFockMemoryPlan plan_periodic_correlation_occupied_fock(
    std::array<int, 3> mesh, std::uint64_t n_basis, std::uint64_t n_home_occupied) {
    const auto wannier = plan_periodic_correlation_wannier(mesh, n_basis, n_home_occupied);
    PeriodicCorrelationOccupiedFockMemoryPlan result;
    result.n_cells = wannier.n_cells;
    result.n_basis = n_basis;
    result.n_home_occupied = n_home_occupied;
    result.element_count = wannier.gauge_element_count;
    result.retained_block_bytes = wannier.caller_gauge_bytes;
    result.projection_workspace_bytes = checked_product(32U, n_basis);
    result.fourier_workspace_bytes = checked_product(16U,
        static_cast<std::uint64_t>(*std::max_element(mesh.begin(), mesh.end())));
    result.peak_owned_numerical_bytes = checked_sum(result.retained_block_bytes,
        std::max(result.projection_workspace_bytes, result.fourier_workspace_bytes));
    result.caller_gauge_bytes = wannier.caller_gauge_bytes;
    result.live_wannier_bytes = wannier.retained_coefficient_bytes;
    const auto maximum = std::vector<Complex>().max_size();
    if (result.element_count > maximum || checked_product(2U, n_basis) > maximum) {
        throw std::length_error("occupied Fock payload exceeds vector max_size");
    }
    return result;
}

const Complex* PeriodicCorrelationOccupiedFock::block(std::size_t translation) const {
    if (translation >= memory_.n_cells || blocks_.size() != memory_.element_count) {
        throw std::out_of_range("occupied Fock translation is out of range or owner is consumed");
    }
    return blocks_.data() + translation * memory_.n_home_occupied * memory_.n_home_occupied;
}

Complex PeriodicCorrelationOccupiedFock::element(
    std::size_t translation, std::size_t i, std::size_t j) const {
    if (i >= memory_.n_home_occupied || j >= memory_.n_home_occupied) {
        throw std::out_of_range("occupied Fock orbital index is out of range");
    }
    return block(translation)[i * memory_.n_home_occupied + j];
}

Complex PeriodicCorrelationOccupiedFock::placed_element(
    std::size_t bra_cell, std::size_t ket_cell, std::size_t i, std::size_t j) const {
    if (!state_ || bra_cell >= memory_.n_cells || ket_cell >= memory_.n_cells) {
        throw std::out_of_range("occupied Fock placed cell index is out of range");
    }
    const RegularKMesh addressing(state_->mesh());
    return element(addressing.transfer_index(ket_cell, bra_cell), i, j);
}

PeriodicCorrelationOccupiedFock make_periodic_correlation_occupied_fock(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t gauge_element_count,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationOccupiedFockOptions& options) {
    require_reference(reference, wannier);
    require_controls(options);
    const auto& state = reference.state();
    const auto memory = plan_periodic_correlation_occupied_fock(
        state.mesh(), state.n_basis(), state.n_correlated_occupied());
    if (owned_numerical_byte_cap == 0U || owned_numerical_byte_cap < memory.peak_owned_numerical_bytes) {
        throw std::length_error("occupied Fock numerical byte cap is missing or exceeded");
    }
    if (gauges == nullptr || gauge_element_count != memory.element_count
        || reinterpret_cast<std::uintptr_t>(gauges) % alignof(Complex) != 0U) {
        throw std::invalid_argument("occupied Fock gauge view has invalid shape, pointer or alignment");
    }
    if (memory.caller_gauge_bytes > std::numeric_limits<std::uintptr_t>::max()
            - reinterpret_cast<std::uintptr_t>(gauges)) {
        throw std::overflow_error("occupied Fock gauge view exceeds native pointer extent");
    }
    if (memory.live_wannier_bytes != wannier.memory().retained_coefficient_bytes) {
        throw std::logic_error("occupied Fock live Wannier inventory is inconsistent");
    }
    const auto& dims = reference.dimensions();
    const auto& budget = reference.budget();
    const auto replicas = checked_product(budget.mpi_ranks, budget.workers_per_rank);
    const auto worker = checked_sum(memory.peak_owned_numerical_bytes,
        checked_sum(memory.caller_gauge_bytes, memory.live_wannier_bytes));
    const auto required_node = checked_sum(checked_sum(dims.external_bytes, dims.shared_bytes),
        checked_sum(checked_product(budget.mpi_ranks,
                                    checked_sum(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
                    checked_product(replicas, worker)));
    if (budget.memory_limit_bytes == 0U || required_node > budget.memory_limit_bytes) {
        throw std::length_error("occupied Fock live inventory exceeds admitted node memory");
    }
    const auto nk = static_cast<std::size_t>(memory.n_cells);
    const auto nao = static_cast<std::size_t>(memory.n_basis);
    const auto nocc = static_cast<std::size_t>(memory.n_home_occupied);
    const auto square = nocc * nocc;
    // This is the exact Wannier gauge payload wire, not a new caller label.
    OccupiedFockDigest gauge_digest("vibeqc.periodic.correlation.wannier.gauge",
        kPeriodicCorrelationWannierContractVersion);
    gauge_digest.u64(nk);
    gauge_digest.u64(nocc);
    for (std::size_t index = 0; index < gauge_element_count; ++index) {
        gauge_digest.complex(gauges[index]);
    }
    const auto gauge_identity = gauge_digest.finish();
    if (gauge_identity != wannier.gauge_payload_sha256()) {
        throw std::invalid_argument("occupied Fock gauge content differs from the sealed Wannier gauge");
    }

    PeriodicCorrelationOccupiedFock result;
    result.memory_ = memory;
    result.options_ = options;
    result.diagnostics_.required_node_memory_bytes = required_node;
    result.blocks_.resize(static_cast<std::size_t>(memory.element_count));
    auto& diagnostics = result.diagnostics_;
    {
        // Reuse two AO columns across all occupied pairs. No full gauged C,
        // occupied-square temporary or eigensolver is created.
        std::vector<Complex> scratch(2U * nao);
        auto* cg_j = scratch.data();
        auto* fcg_j = scratch.data() + nao;
        for (std::size_t k = 0; k < nk; ++k) {
            const auto* u = gauges + k * square;
            const auto& fock = state.fock(k);
            const auto& mask = state.correlated_occupied_mask(k);
            const auto& eps = state.orbital_energies(k);
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t mu = 0; mu < nao; ++mu) {
                    cg_j[mu] = gauged_coefficient(state, k, mu, j, u, nocc);
                }
                for (std::size_t mu = 0; mu < nao; ++mu) {
                    CompensatedComplex value;
                    for (std::size_t nu = 0; nu < nao; ++nu) value.add(fock(mu, nu) * cg_j[nu]);
                    fcg_j[mu] = value.value();
                }
                for (std::size_t i = 0; i < nocc; ++i) {
                    CompensatedComplex projection;
                    for (std::size_t mu = 0; mu < nao; ++mu) {
                        projection.add(std::conj(gauged_coefficient(state, k, mu, i, u, nocc)) * fcg_j[mu]);
                    }
                    const auto value = projection.value();
                    result.blocks_[(k * nocc + i) * nocc + j] = value;
                    CompensatedComplex canonical;
                    std::size_t row = 0;
                    for (std::size_t band = 0; band < mask.size(); ++band) {
                        if (mask[band] != 0U) {
                            canonical.add(std::conj(u[row * nocc + i]) * (eps[band] * u[row * nocc + j]));
                            ++row;
                        }
                    }
                    const double discrepancy = std::abs(value - canonical.value());
                    diagnostics.maximum_canonical_projection_discrepancy = std::max(
                        diagnostics.maximum_canonical_projection_discrepancy, discrepancy);
                    diagnostics.canonical_projection_discrepancy_frobenius = std::hypot(
                        diagnostics.canonical_projection_discrepancy_frobenius, discrepancy);
                    if (!std::isfinite(diagnostics.canonical_projection_discrepancy_frobenius)) {
                        throw std::overflow_error("occupied Fock canonical diagnostic overflowed");
                    }
                }
            }
        }
    }  // projection scratch freed BEFORE the Fourier line is allocated
    const RegularKMesh addressing(state.mesh());
    for (std::size_t k = 0; k < nk; ++k) {
        const auto minus_k = addressing.negate_index(k);
        for (std::size_t i = 0; i < nocc; ++i) {
            for (std::size_t j = 0; j < nocc; ++j) {
                const auto value = result.blocks_[(k * nocc + i) * nocc + j];
                if (!compare(value, std::conj(result.blocks_[(k * nocc + j) * nocc + i]),
                        options.hermiticity_absolute_tolerance, options.hermiticity_relative_tolerance,
                        diagnostics.maximum_projected_hermiticity_residual)) {
                    throw std::invalid_argument("occupied Fock physical projection failed Hermiticity");
                }
                diagnostics.time_reversal_compatible &= compare(value,
                    std::conj(result.blocks_[(minus_k * nocc + i) * nocc + j]),
                    options.time_reversal_absolute_tolerance, options.time_reversal_relative_tolerance,
                    diagnostics.maximum_projected_time_reversal_residual);
            }
        }
    }
    if (options.require_time_reversal && !diagnostics.time_reversal_compatible) {
        throw std::invalid_argument("occupied Fock projected blocks failed time reversal");
    }
    transform_in_place(result.blocks_, state.mesh(), nk, nocc);
    for (std::size_t r = 0; r < nk; ++r) {
        const auto minus_r = addressing.negate_index(r);
        for (std::size_t i = 0; i < nocc; ++i) {
            for (std::size_t j = 0; j < nocc; ++j) {
                const auto value = result.blocks_[(r * nocc + i) * nocc + j];
                if (!compare(value, std::conj(result.blocks_[(minus_r * nocc + j) * nocc + i]),
                        options.hermiticity_absolute_tolerance, options.hermiticity_relative_tolerance,
                        diagnostics.maximum_translation_hermiticity_residual)) {
                    throw std::invalid_argument("occupied Fock translation blocks failed Hermiticity");
                }
                const double imaginary = std::abs(value.imag());
                const double magnitude = std::abs(value);
                if (!std::isfinite(magnitude)) throw std::overflow_error("occupied Fock magnitude overflowed");
                diagnostics.maximum_block_imaginary_magnitude = std::max(
                    diagnostics.maximum_block_imaginary_magnitude, imaginary);
                diagnostics.real_blocks_compatible &= imaginary <= options.real_absolute_tolerance
                    + options.real_relative_tolerance * magnitude;
            }
        }
    }
    if (options.require_real_blocks && !diagnostics.real_blocks_compatible) {
        throw std::invalid_argument("occupied Fock translation blocks failed the real-block gate");
    }
    result.state_ = reference.state_handle();
    result.wannier_identity_ = wannier.wannier_identity_sha256();
    result.gauge_identity_ = gauge_identity;
    result.allocation_identity_ = dims.allocation_identity;
    OccupiedFockDigest payload("vibeqc.periodic.correlation.occupied-fock.blocks",
        kPeriodicCorrelationOccupiedFockContractVersion);
    for (int component : state.mesh()) payload.u32(component);
    payload.u64(nocc);
    for (const auto value : result.blocks_) payload.complex(value);
    result.block_identity_ = payload.finish();
    OccupiedFockDigest identity("vibeqc.periodic.correlation.occupied-fock.identity",
        kPeriodicCorrelationOccupiedFockContractVersion);
    identity.string(state.state_identity_sha256());
    identity.string(dims.calculation_identity);
    identity.string(dims.allocation_identity);
    identity.string(result.wannier_identity_);
    identity.string(result.gauge_identity_);
    identity.string(result.block_identity_);
    identity.string("physical-AO-F-projection;inverse-bloch-1/Nk;bra-minus-ket;exact-gamma");
    identity.real(options.hermiticity_absolute_tolerance);
    identity.real(options.hermiticity_relative_tolerance);
    identity.real(options.time_reversal_absolute_tolerance);
    identity.real(options.time_reversal_relative_tolerance);
    identity.real(options.real_absolute_tolerance);
    identity.real(options.real_relative_tolerance);
    identity.u32(options.require_time_reversal ? 1U : 0U);
    identity.u32(options.require_real_blocks ? 1U : 0U);
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
