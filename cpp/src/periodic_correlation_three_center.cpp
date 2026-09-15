#include "vibeqc/periodic_correlation_three_center.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/detail/periodic_gaussian_three_center_numeric.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
constexpr double kTilePi = 3.14159265358979323846;

std::uint64_t multiply(std::uint64_t left, std::uint64_t right) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw std::length_error("three-center tile extent overflow");
    }
    return left * right;
}

std::uint64_t add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::length_error("three-center tile byte count overflow");
    }
    return left + right;
}

void finite(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("three-center tile numerical value is non-finite");
    }
}

void accumulate(double term, double& sum, double& correction) {
    finite(term);
    const double next = sum + term;
    correction += std::abs(sum) >= std::abs(term)
        ? (sum - next) + term : (term - next) + sum;
    sum = next;
    finite(sum);
    finite(correction);
}

class TileDigest {
public:
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (unsigned i = 0; i < 4; ++i) bytes[i] = value >> (24 - 8 * i);
        digest_.update(bytes.data(), bytes.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned i = 0; i < 8; ++i) bytes[i] = value >> (56 - 8 * i);
        digest_.update(bytes.data(), bytes.size());
    }
    void string(const std::string& value) {
        u64(value.size());
        digest_.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    }
    void real(double value) {
        finite(value);
        if (value == 0.0) value = 0.0;
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    std::string finish() { return digest_.finish_hex(); }
private:
    detail::Sha256 digest_;
};

std::array<char, 64> identity(const std::string& input) {
    if (input.size() != 64 || !std::all_of(input.begin(), input.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) {
        throw std::invalid_argument("three-center tile requires lowercase SHA-256 identities");
    }
    std::array<char, 64> result;
    std::copy(input.begin(), input.end(), result.begin());
    return result;
}

std::uint64_t phase_peak(const PeriodicCorrelationFactorBuildPlan& plan,
                        PeriodicCorrelationFactorBuildPhase phase) {
    for (const auto& item : plan.phases) {
        if (item.phase == phase) return item.peak_memory_bytes;
    }
    throw std::logic_error("three-center tile admission omits a required phase");
}

void digest_descriptor(TileDigest& hash, const PeriodicCorrelationFactorTileDescriptor& d) {
    hash.u64(d.sequence_index);
    hash.u64(d.q_index);
    hash.u64(d.k_bra_index);
    hash.u64(d.k_ket_index);
    for (int value : d.k_ket_reciprocal_wrap) hash.u32(static_cast<std::uint32_t>(value));
    hash.u64(d.ao_pair_begin);
    hash.u64(d.ao_pair_count);
    hash.u64(d.auxiliary_begin);
    hash.u64(d.auxiliary_count);
    hash.u64(d.element_count);
}

struct ReciprocalTileAssembly {
    const BasisSet& ao;
    const BasisSet& auxiliary;
    const PeriodicSystem& system;
    const Eigen::Vector3d& ket;
    const detail::ThreeCenterNumericalSelection& descriptor;
    double image_cutoff;
    std::uint64_t image_cap;
    std::uint64_t expected_image_candidates;
    std::uint64_t expected_retained_images;
    std::size_t capacity;
    std::size_t count = 0;
    std::uint64_t visited = 0;
    std::vector<Complex>& raw;
    std::vector<Complex>& correction;
    std::vector<double>& reciprocal;
    std::vector<Complex>& weighted;

    void flush() {
        if (count == 0) return;
        const auto a = auxiliary.nbasis();
        const auto pairs = static_cast<std::size_t>(descriptor.ao_pair_count);
        AuxiliaryFourierVectorView view;
        view.x = reciprocal.data();
        view.y = reciprocal.data() + capacity;
        view.z = reciprocal.data() + 2 * capacity;
        view.count = count;
        auto f = auxiliary_gaussian_fourier_panel(auxiliary, view,
                                                  multiply(multiply(a, count), 16));
        auto rho = ao_pair_gaussian_fourier_panel(
            ao, system, view, ket, descriptor.ao_pair_begin, descriptor.ao_pair_count,
            image_cutoff, image_cap, multiply(multiply(pairs, count), 16));
        if (rho.image_candidate_count != expected_image_candidates
            || rho.retained_pair_image_count != expected_retained_images
            || f.n_auxiliary != a || f.n_vectors != count
            || rho.n_pairs != pairs || rho.n_vectors != count) {
            throw std::logic_error("three-center Fourier panel changed its admitted shape or image policy");
        }
        for (std::size_t p = 0; p < a; ++p) {
            for (std::size_t g = 0; g < count; ++g) {
                const Complex value = std::conj(f(p, g)) * reciprocal[4 * capacity + g];
                finite(value.real());
                finite(value.imag());
                weighted[p * capacity + g] = value;
            }
        }
        // Every output entry follows the source's fixed reciprocal order,
        // independently of chunk size. No k weight or extra conjugation.
        for (std::size_t p = 0; p < a; ++p) {
            for (std::size_t pair = 0; pair < pairs; ++pair) {
                const auto index = p * pairs + pair;
                double real = raw[index].real(), imag = raw[index].imag();
                double real_c = correction[index].real(), imag_c = correction[index].imag();
                for (std::size_t g = 0; g < count; ++g) {
                    const Complex term = weighted[p * capacity + g]
                                         * rho.data[pair * count + g];
                    accumulate(term.real(), real, real_c);
                    accumulate(term.imag(), imag, imag_c);
                }
                raw[index] = {real, imag};
                correction[index] = {real_c, imag_c};
            }
        }
        count = 0;
    }

    static void receive(const std::array<double, 5>& lanes, void* pointer) {
        auto& self = *static_cast<ReciprocalTileAssembly*>(pointer);
        for (std::size_t d = 0; d < 5; ++d) {
            finite(lanes[d]);
            self.reciprocal[d * self.capacity + self.count] = lanes[d];
        }
        ++self.count;
        ++self.visited;
        if (self.count == self.capacity) self.flush();
    }
};

}  // namespace

#define VIBEQC_TILE_IDENTITY(name, field) \
std::string PeriodicCorrelationThreeCenterTile::name() const { \
    return std::string(field.begin(), field.end()); \
}
VIBEQC_TILE_IDENTITY(source_identity_sha256, source_identity_)
VIBEQC_TILE_IDENTITY(whitener_payload_identity_sha256, whitener_identity_)
VIBEQC_TILE_IDENTITY(ao_basis_identity_sha256, ao_identity_)
VIBEQC_TILE_IDENTITY(auxiliary_basis_identity_sha256, auxiliary_identity_)
VIBEQC_TILE_IDENTITY(census_identity_sha256, census_identity_)
VIBEQC_TILE_IDENTITY(plan_identity_sha256, plan_identity_)
VIBEQC_TILE_IDENTITY(payload_identity_sha256, payload_identity_)
#undef VIBEQC_TILE_IDENTITY

detail::ThreeCenterNumericalResult detail::contract_three_center_numeric(
    const BasisSet& ao_basis, const BasisSet& auxiliary_basis, const PeriodicSystem& system,
    const Eigen::Vector3d& ket, const ThreeCenterNumericalSelection& descriptor,
    const std::vector<Complex>& whitener,
    std::uint64_t accepted_count, std::uint64_t capacity64,
    double image_cutoff_bohr, std::uint64_t maximum_image_candidates,
    std::uint64_t expected_image_candidates, std::uint64_t expected_retained_images,
    std::uint64_t owned_numeric_byte_cap, MetricReciprocalVisitor visitor, const void* source_user) {
    const auto a = static_cast<std::uint64_t>(auxiliary_basis.nbasis());
    const auto n = static_cast<std::uint64_t>(ao_basis.nbasis());
    const auto pairs = descriptor.ao_pair_count, square = multiply(n, n);
    if (visitor == nullptr || a == 0 || pairs == 0 || accepted_count == 0
        || capacity64 == 0 || capacity64 > accepted_count
        || descriptor.ao_pair_begin > square || pairs > square - descriptor.ao_pair_begin
        || descriptor.auxiliary_count == 0 || descriptor.auxiliary_begin > a
        || descriptor.auxiliary_count > a - descriptor.auxiliary_begin
        || whitener.size() != multiply(a, a) || maximum_image_candidates == 0
        || expected_image_candidates > maximum_image_candidates
        || expected_retained_images > expected_image_candidates || owned_numeric_byte_cap == 0) {
        throw std::invalid_argument("three-center numerical leaf shape or admitted source mismatch");
    }
    const auto full_panel_elements = multiply(a, pairs);
    const auto reciprocal_elements = multiply(5U, capacity64);
    const auto auxiliary_elements = multiply(a, capacity64);
    const auto pair_elements = multiply(pairs, capacity64);
    const auto output_elements = multiply(descriptor.auxiliary_count, pairs);
    const auto assembly_bytes = add(add(multiply(32U, full_panel_elements), multiply(8U, reciprocal_elements)),
        add(multiply(32U, auxiliary_elements), add(multiply(16U, pair_elements),
            ao_pair_fourier_fixed_numeric_workspace_bytes())));
    const auto whitening_bytes = add(multiply(16U, full_panel_elements), multiply(16U, output_elements));
    if (std::max(assembly_bytes, whitening_bytes) > owned_numeric_byte_cap) {
        throw std::length_error("three-center numerical leaf exceeds owned memory cap");
    }
    for (auto elements : {full_panel_elements, auxiliary_elements, pair_elements, output_elements}) {
        if (elements > std::vector<Complex>().max_size()) throw std::length_error("three-center numerical leaf vector extent");
    }
    if (reciprocal_elements > std::vector<double>().max_size()) throw std::length_error("three-center numerical reciprocal extent");
    ThreeCenterNumericalResult result;
    std::vector<Complex> raw(static_cast<std::size_t>(full_panel_elements));
    {
        std::vector<Complex> correction(static_cast<std::size_t>(full_panel_elements));
        std::vector<double> reciprocal(static_cast<std::size_t>(reciprocal_elements));
        std::vector<Complex> weighted(static_cast<std::size_t>(auxiliary_elements));
        ReciprocalTileAssembly assembly{
            ao_basis, auxiliary_basis, system, ket, descriptor, image_cutoff_bohr,
            maximum_image_candidates, expected_image_candidates, expected_retained_images,
            static_cast<std::size_t>(capacity64), 0, 0, raw, correction, reciprocal, weighted};
        struct CallbackContext { ReciprocalTileAssembly* assembly; std::uint64_t expected; };
        CallbackContext callback{&assembly, accepted_count};
        const auto visited = visitor(
            [](const std::array<std::int64_t, 3>&, const std::array<double, 5>& lanes, void* pointer) {
                auto& c = *static_cast<CallbackContext*>(pointer);
                if (c.assembly->visited >= c.expected) {
                    throw std::logic_error("three-center numerical visitor exceeded admitted count");
                }
                ReciprocalTileAssembly::receive(lanes, c.assembly);
            }, &callback, source_user);
        assembly.flush();
        if (visited != accepted_count || assembly.visited != visited) {
            throw std::logic_error("three-center tile reciprocal traversal count mismatch");
        }
        result.visited_reciprocal_count = visited;
        for (std::size_t element = 0; element < raw.size(); ++element) {
            raw[element] += correction[element];
            finite(raw[element].real());
            finite(raw[element].imag());
        }
    }
    // All Fourier/reciprocal/compensation allocations are dead before this
    // whitening output is allocated. W rows remain original auxiliary AOs.
    result.values.resize(static_cast<std::size_t>(output_elements));
    const auto& w = whitener;
    for (std::uint64_t row = 0; row < descriptor.auxiliary_count; ++row) {
        for (std::uint64_t pair = 0; pair < pairs; ++pair) {
            double real = 0, imag = 0, real_c = 0, imag_c = 0;
            for (std::uint64_t p = 0; p < a; ++p) {
                const Complex term = w[(descriptor.auxiliary_begin + row) * a + p]
                                     * raw[p * pairs + pair];
                accumulate(term.real(), real, real_c);
                accumulate(term.imag(), imag, imag_c);
            }
            real += real_c;
            imag += imag_c;
            finite(real);
            finite(imag);
            result.values[row * pairs + pair] = {real == 0 ? 0.0 : real, imag == 0 ? 0.0 : imag};
        }
    }
    return result;
}

PeriodicCorrelationThreeCenterTile build_periodic_correlation_three_center_tile(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    const PeriodicCorrelationMetricFactorizationResult& whitener,
    const BasisSet& ao_basis, const BasisSet& auxiliary_basis,
    std::uint64_t sequence_index, double image_cutoff_bohr,
    std::uint64_t maximum_image_candidates) {
    const auto descriptor = schedule.descriptor(sequence_index);
    const auto& config = census.config();
    const auto a = census.shape().n_auxiliary;
    const auto pairs = descriptor.ao_pair_count;
    const auto full_panel_elements = multiply(a, pairs);
    const auto full_panel_bytes = multiply(full_panel_elements, 16);
    const auto planned_panel_bytes = multiply(multiply(a, census.shape().ao_pair_block), 16);
    const auto w_elements = multiply(a, a);
    const auto w_bytes = multiply(w_elements, 16);
    const auto output_bytes = multiply(descriptor.element_count, 16);
    const auto capacity64 = std::min(config.reciprocal_block, source.accepted_vector_count());
    const auto reciprocal_elements = multiply(capacity64, 5);
    const auto auxiliary_elements = multiply(capacity64, a);
    const auto pair_elements = multiply(capacity64, pairs);
    const auto scratch = ao_pair_fourier_fixed_numeric_workspace_bytes();
    if (full_panel_elements > std::vector<Complex>().max_size()
        || auxiliary_elements > std::vector<Complex>().max_size()
        || pair_elements > std::vector<Complex>().max_size()
        || reciprocal_elements > std::vector<double>().max_size()
        || descriptor.element_count > std::vector<Complex>().max_size()
        || output_bytes > std::numeric_limits<std::uint64_t>::max() / 8 - 4096) {
        throw std::length_error("three-center tile exceeds allocation or payload SHA extent");
    }
    if (!std::isfinite(image_cutoff_bohr) || image_cutoff_bohr < 0
        || maximum_image_candidates == 0 || capacity64 == 0) {
        throw std::invalid_argument("three-center tile requires finite image cutoff and positive source caps");
    }
    if (config.producer_mode != PeriodicCorrelationFactorProducerMode::FullCoulombAllReciprocalReference
        || config.short_range_policy != PeriodicCorrelationShortRangePolicy::DisabledAllReciprocal
        || config.backend.exact_extra_retained_bytes < planned_panel_bytes
        || config.backend.per_thread_fourier_transform_fixed_workspace_bytes < scratch) {
        throw std::invalid_argument("three-center tile requires admitted all-reciprocal mode, compensation and Fourier workspace");
    }
    PeriodicCorrelationThreeCenterTile result;
    result.descriptor_ = descriptor;
    result.image_cutoff_bohr_ = image_cutoff_bohr;
    result.output_bytes_ = output_bytes;
    {
        const auto plan = plan_periodic_correlation_factor_build(reference, schedule, census);
        if (plan.admission != PeriodicCorrelationFactorBuildAdmissionCode::Admitted) {
            throw std::runtime_error("three-center tile resource plan is not admitted");
        }
        if (!whitener.state_handle() || !source.state_handle()
            || whitener.state_handle().get() != reference.state_handle().get()
            || source.state_handle().get() != reference.state_handle().get()
            || whitener.census_identity_sha256() != census.census_identity_sha256()
            || whitener.plan_identity_sha256() != plan.plan_identity_sha256
            || whitener.source_identity_sha256() != source.source_identity_sha256()
            || whitener.q_index() != descriptor.q_index
            || source.q_index() != descriptor.q_index
            || whitener.diagnostics().n_auxiliary != a
            || whitener.matrix_row_major().size() != w_elements
            || whitener.diagnostics().retained_rank == 0
            || source.periodic_dimension() != 3) {
            throw std::invalid_argument("three-center tile source, whitener or plan provenance mismatch");
        }
        if (plan.components.raw_three_center_panel_bytes != planned_panel_bytes
            || plan.components.whitening_input_panel_bytes != planned_panel_bytes
            || plan.components.whitening_output_panel_bytes != planned_panel_bytes
            || plan.components.whitener_matrix_bytes != w_bytes
            || output_bytes > plan.components.whitening_output_panel_bytes) {
            throw std::logic_error("three-center tile shapes differ from admitted components");
        }
        result.admitted_assembly_peak_bytes_ = phase_peak(plan, PeriodicCorrelationFactorBuildPhase::ReciprocalThreeCenter);
        result.admitted_whitening_peak_bytes_ = phase_peak(plan, PeriodicCorrelationFactorBuildPhase::Whitening);
        result.plan_identity_ = identity(plan.plan_identity_sha256);
    }
    if (ao_basis.nbasis() != census.shape().n_basis || auxiliary_basis.nbasis() != a
        || auxiliary_basis_content_identity_sha256(ao_basis) != config.ao_basis_identity_sha256
        || auxiliary_basis_content_identity_sha256(auxiliary_basis) != config.auxiliary_basis_identity_sha256
        || whitener.auxiliary_basis_identity_sha256() != config.auxiliary_basis_identity_sha256) {
        throw std::invalid_argument("three-center tile AO or auxiliary basis content mismatch");
    }
    // Replay exact reciprocal support before allocating any numerical panel.
    {
        const auto verified = make_periodic_correlation_reciprocal_metric_source_manifest(
            reference, schedule, config, auxiliary_basis, source.q_index(), source.candidate_count());
        if (verified.source_identity_sha256() != source.source_identity_sha256()) {
            throw std::invalid_argument("three-center tile reciprocal source replay mismatch");
        }
    }
    const RegularKMesh addressing(schedule.mesh(), schedule.is_shift());
    const auto ket_address = addressing.add_transfer_with_wrap(
        addressing.address(descriptor.k_bra_index), addressing.transfer_address(descriptor.q_index));
    if (addressing.index(ket_address.address) != descriptor.k_ket_index) {
        throw std::logic_error("three-center tile canonical ket address mismatch");
    }
    for (int d = 0; d < 3; ++d) {
        if (ket_address.wrap[d] != descriptor.k_ket_reciprocal_wrap[d]) {
            throw std::logic_error("three-center tile ket reciprocal wrap mismatch");
        }
    }
    const Eigen::Vector3d ket = reference.state_handle()->kpoint_cartesian(descriptor.k_ket_index);
    PeriodicSystem system;
    system.dim = 3;
    const auto& reciprocal_lattice = reference.state_handle()->reciprocal_lattice();
    system.lattice = (2.0 * kTilePi) * reciprocal_lattice.inverse().transpose();
    const double dual_residual = (system.lattice.transpose() * reciprocal_lattice
        - (2.0 * kTilePi) * Eigen::Matrix3d::Identity()).norm();
    const double dual_tolerance = 256.0 * std::numeric_limits<double>::epsilon()
        * std::max(2.0 * kTilePi, system.lattice.norm() * reciprocal_lattice.norm());
    if (!system.lattice.allFinite() || !std::isfinite(dual_residual)
        || !std::isfinite(dual_tolerance) || dual_residual > dual_tolerance) {
        throw std::runtime_error("three-center tile reconstructed direct lattice failed duality check");
    }
    const auto ao_preflight = ao_pair_gaussian_fourier_panel(
        ao_basis, system, AuxiliaryFourierVectorView{}, ket,
        descriptor.ao_pair_begin, pairs, image_cutoff_bohr, maximum_image_candidates, 1);
    const auto aux_preflight = auxiliary_gaussian_fourier_panel(
        auxiliary_basis, AuxiliaryFourierVectorView{}, 1);
    if (!ao_preflight.data.empty() || !aux_preflight.data.empty()) {
        throw std::logic_error("three-center tile zero-count preflight allocated payload");
    }
    result.image_candidate_count_ = ao_preflight.image_candidate_count;
    result.retained_pair_image_count_ = ao_preflight.retained_pair_image_count;
    result.reciprocal_vector_count_ = source.accepted_vector_count();
    result.source_identity_ = identity(source.source_identity_sha256());
    result.whitener_identity_ = identity(whitener.payload_identity_sha256());
    result.ao_identity_ = identity(config.ao_basis_identity_sha256);
    result.auxiliary_identity_ = identity(config.auxiliary_basis_identity_sha256);
    result.census_identity_ = identity(census.census_identity_sha256());
    std::uint64_t assembly_bytes = add(w_bytes, multiply(2, full_panel_bytes));
    assembly_bytes = add(assembly_bytes, multiply(reciprocal_elements, 8));
    assembly_bytes = add(assembly_bytes, multiply(multiply(2, auxiliary_elements), 16));
    assembly_bytes = add(assembly_bytes, multiply(pair_elements, 16));
    assembly_bytes = add(assembly_bytes, scratch);
    result.numerical_peak_bytes_ = std::max(assembly_bytes,
        add(add(w_bytes, full_panel_bytes), output_bytes));
    if (assembly_bytes > result.admitted_assembly_peak_bytes_
        || add(add(w_bytes, full_panel_bytes), output_bytes)
            > result.admitted_whitening_peak_bytes_) {
        throw std::logic_error("three-center tile numerical lifetime exceeds admitted phase peak");
    }
    const detail::ThreeCenterNumericalSelection selection{
        descriptor.ao_pair_begin, descriptor.ao_pair_count,
        descriptor.auxiliary_begin, descriptor.auxiliary_count};
    auto numerical = detail::contract_three_center_numeric(
        ao_basis, auxiliary_basis, system, ket, selection, whitener.matrix_row_major(),
        source.accepted_vector_count(), capacity64, image_cutoff_bohr, maximum_image_candidates,
        result.image_candidate_count_, result.retained_pair_image_count_,
        result.numerical_peak_bytes_ - w_bytes,
        [](detail::PeriodicReciprocalRecordCallback callback, void* user, const void* pointer) {
            const auto& source = *static_cast<const PeriodicCorrelationReciprocalMetricSourceManifest*>(pointer);
            struct Adapter { detail::PeriodicReciprocalRecordCallback callback; void* user; };
            Adapter adapter{callback, user};
            return visit_periodic_correlation_reciprocal_metric_source(source,
                [](const std::array<double, 5>& lanes, void* p) {
                    auto& adapter = *static_cast<Adapter*>(p);
                    adapter.callback({{0, 0, 0}}, lanes, adapter.user);
                }, &adapter);
        }, &source);
    result.values_ = std::move(numerical.values);
    TileDigest payload;
    payload.string("vibeqc.periodic.correlation.three-center.payload");
    payload.u32(result.contract_version());
    payload.string(result.source_identity_sha256());
    payload.string(result.whitener_payload_identity_sha256());
    payload.string(result.ao_basis_identity_sha256());
    payload.string(result.auxiliary_basis_identity_sha256());
    payload.string(kPeriodicCorrelationThreeCenterImagePolicy);
    payload.string(kPeriodicCorrelationThreeCenterLatticePolicy);
    payload.real(image_cutoff_bohr);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) payload.real(system.lattice(i, j));
    for (int d = 0; d < 3; ++d) payload.real(ket[d]);
    digest_descriptor(payload, descriptor);
    payload.u64(result.image_candidate_count_);
    payload.u64(result.retained_pair_image_count_);
    payload.u64(result.reciprocal_vector_count_);
    payload.u64(result.output_bytes_);
    for (const auto value : result.values_) { payload.real(value.real()); payload.real(value.imag()); }
    result.payload_identity_ = identity(payload.finish());
    return result;
}

PeriodicCorrelationThreeCenterStreamReceipt stream_periodic_correlation_three_center_tiles(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    const BasisSet& ao_basis, const BasisSet& auxiliary_basis,
    double image_cutoff_bohr, double negative_tolerance,
    const PeriodicCorrelationThreeCenterStreamCaps& caps,
    void (*receive)(const PeriodicCorrelationThreeCenterTile&, void*), void* receiver,
    void (*progress)(const PeriodicCorrelationThreeCenterStreamProgress&, void*),
    void* progress_context) {
    const auto& shape = schedule.shape();
    const auto& config = census.config();
    if (receive == nullptr || caps.maximum_tile_count == 0
        || caps.maximum_logical_bytes == 0 || caps.maximum_tile_bytes == 0
        || caps.maximum_reciprocal_candidates_per_q == 0
        || caps.maximum_image_candidates_per_tile == 0) {
        throw std::invalid_argument("physical factor stream requires a receiver and explicit positive work caps");
    }
    if (shape.tile_count > caps.maximum_tile_count
        || shape.logical_bytes > caps.maximum_logical_bytes
        || shape.maximum_tile_bytes > caps.maximum_tile_bytes) {
        throw std::length_error("physical factor stream exceeds tile or logical work caps");
    }
    if (!std::isfinite(image_cutoff_bohr) || image_cutoff_bohr < 0.0
        || !std::isfinite(negative_tolerance) || negative_tolerance <= 0.0
        || negative_tolerance > config.metric_absolute_eigenvalue_threshold) {
        throw std::invalid_argument("physical factor stream has invalid image or negative-spectrum cutoff");
    }
    // Preflight the complete digest domain and receiver lifetime before any
    // callback or size-dependent source allocation. Each q contributes 152
    // bytes and each tile 80 bytes; 4096 safely bounds the fixed prefix.
    const auto wire_bytes = add(4096, add(multiply(shape.n_kpoints, 152),
                                         multiply(shape.tile_count, 80)));
    if (wire_bytes > std::numeric_limits<std::uint64_t>::max() / 8) {
        throw std::length_error("physical factor stream exceeds SHA-256 bit-length domain");
    }
    const auto compensation_bytes = multiply(
        multiply(shape.n_auxiliary, shape.ao_pair_block), 16);
    if (config.producer_mode != PeriodicCorrelationFactorProducerMode::FullCoulombAllReciprocalReference
        || config.short_range_policy != PeriodicCorrelationShortRangePolicy::DisabledAllReciprocal
        || config.backend_identity_sha256
            != periodic_correlation_metric_factorization_backend_identity_sha256()
        || config.backend.eigensolver_workspace_bytes != 0
        || config.backend.whitener_workspace_bytes != 0
        || config.backend.exact_extra_retained_bytes
            < add(compensation_bytes, caps.receiver_retained_numeric_bytes)
        || config.backend.exact_extra_control_bytes
            < sizeof(PeriodicCorrelationReciprocalMetricSourceManifest)
        || config.backend.per_thread_fourier_transform_fixed_workspace_bytes
            < ao_pair_fourier_fixed_numeric_workspace_bytes()) {
        throw std::invalid_argument("physical factor stream requires its compiled backend and receiver/Fourier workspace admission");
    }
    if (ao_basis.nbasis() != shape.n_basis || auxiliary_basis.nbasis() != shape.n_auxiliary
        || auxiliary_basis_content_identity_sha256(ao_basis) != config.ao_basis_identity_sha256
        || auxiliary_basis_content_identity_sha256(auxiliary_basis) != config.auxiliary_basis_identity_sha256) {
        throw std::invalid_argument("physical factor stream basis content differs from its census");
    }
    PeriodicCorrelationThreeCenterStreamReceipt result;
    {
        const auto plan = plan_periodic_correlation_factor_build(reference, schedule, census);
        if (plan.admission != PeriodicCorrelationFactorBuildAdmissionCode::Admitted) {
            throw std::runtime_error("physical factor stream resource plan is not admitted");
        }
        result.admitted_peak_memory_bytes = plan.required_memory_bytes;
        result.plan_identity_sha256 = plan.plan_identity_sha256;
    }
    result.schedule_identity_sha256 = schedule.schedule_identity_sha256();
    result.census_identity_sha256 = census.census_identity_sha256();
    TileDigest digest;
    digest.string("vibeqc.periodic.correlation.three-center.stream");
    digest.u32(result.contract_version);
    digest.string(result.schedule_identity_sha256);
    digest.string(result.census_identity_sha256);
    digest.string(result.plan_identity_sha256);
    digest.real(image_cutoff_bohr);
    digest.real(negative_tolerance);
    digest.u64(shape.n_kpoints);
    digest.u64(shape.tile_count);
    digest.u64(shape.logical_element_count);
    digest.u64(shape.logical_bytes);
    PeriodicCorrelationThreeCenterStreamProgress event;
    event.total_q_count = shape.n_kpoints;
    event.total_tile_count = shape.tile_count;
    event.total_logical_bytes = shape.logical_bytes;
    const auto notify = [&](PeriodicCorrelationThreeCenterStreamStage stage) {
        event.stage = stage;
        if (progress != nullptr) progress(event, progress_context);
    };
    for (std::uint64_t q = 0; q < shape.n_kpoints; ++q) {
        event.q_index = q;
        notify(PeriodicCorrelationThreeCenterStreamStage::Source);
        const auto source = make_periodic_correlation_reciprocal_metric_source_manifest(
            reference, schedule, config, auxiliary_basis, q,
            caps.maximum_reciprocal_candidates_per_q);
        const auto opposite_q = RegularKMesh(schedule.mesh()).negate_index(q);
        if (opposite_q == q) {
            require_periodic_correlation_reciprocal_source_conjugacy(
                source, source, caps.maximum_reciprocal_candidates_per_q);
        } else {
            // Only one additional constant-size manifest, charged to extra
            // control storage above; freed before raw-M or W allocations.
            const auto opposite = make_periodic_correlation_reciprocal_metric_source_manifest(
                reference, schedule, config, auxiliary_basis, opposite_q,
                caps.maximum_reciprocal_candidates_per_q);
            require_periodic_correlation_reciprocal_source_conjugacy(
                source, opposite, caps.maximum_reciprocal_candidates_per_q);
        }
        notify(PeriodicCorrelationThreeCenterStreamStage::Metric);
        auto raw = build_periodic_correlation_reciprocal_metric(
            reference, schedule, census, source, auxiliary_basis);
        notify(PeriodicCorrelationThreeCenterStreamStage::Factorization);
        const auto whitener = factorize_periodic_correlation_metric(
            reference, schedule, census, std::move(raw), negative_tolerance);
        digest.u64(q);
        digest.string(source.source_identity_sha256());
        digest.string(whitener.payload_identity_sha256());
        notify(PeriodicCorrelationThreeCenterStreamStage::Tiles);
        for (std::uint64_t in_q = 0; in_q < shape.tiles_per_q; ++in_q) {
            const auto tile = build_periodic_correlation_three_center_tile(
                reference, schedule, census, source, whitener, ao_basis, auxiliary_basis,
                result.completed_tile_count, image_cutoff_bohr,
                caps.maximum_image_candidates_per_tile);
            const auto& descriptor = tile.descriptor();
            if (descriptor.q_index != q || descriptor.sequence_index != result.completed_tile_count
                || tile.output_bytes() > caps.maximum_tile_bytes) {
                throw std::logic_error("physical factor stream violated its canonical sequence");
            }
            // Hash before delivery; the receiver only sees a const borrow.
            // Never publish completion counts until delivery has succeeded.
            digest.u64(descriptor.sequence_index);
            digest.string(tile.payload_identity_sha256());
            receive(tile, receiver);
            result.completed_tile_count = add(result.completed_tile_count, 1);
            result.completed_element_count = add(result.completed_element_count, descriptor.element_count);
            result.completed_logical_bytes = add(result.completed_logical_bytes, tile.output_bytes());
            result.maximum_tile_bytes = std::max(result.maximum_tile_bytes, tile.output_bytes());
            event.completed_tile_count = result.completed_tile_count;
            event.completed_logical_bytes = result.completed_logical_bytes;
            notify(PeriodicCorrelationThreeCenterStreamStage::Tiles);
        }
        ++result.completed_q_count;
        event.completed_q_count = result.completed_q_count;
        notify(PeriodicCorrelationThreeCenterStreamStage::QComplete);
    }
    if (result.completed_q_count != shape.n_kpoints
        || result.completed_tile_count != shape.tile_count
        || result.completed_element_count != shape.logical_element_count
        || result.completed_logical_bytes != shape.logical_bytes) {
        throw std::logic_error("physical factor stream ended without complete coverage");
    }
    result.payload_identity_sha256 = digest.finish();
    notify(PeriodicCorrelationThreeCenterStreamStage::Complete);
    return result;
}

}  // namespace vibeqc
