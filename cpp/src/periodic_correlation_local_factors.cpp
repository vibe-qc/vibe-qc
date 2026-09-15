#include "vibeqc/periodic_correlation_local_factors.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "periodic_correlation_local_factors_internal.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
using Selection = PeriodicCorrelationLocalOrbitalSelection;
using Memory = PeriodicCorrelationLocalFactorMemoryPlan;
using Orientation = PeriodicCorrelationLocalFactorOrientation;
using Digest = periodic_correlation_local_detail::Digest;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;
static_assert(sizeof(Complex) == 16 && sizeof(double) == 8,
              "local factor accounting requires binary64/complex128");

std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("local factor count or byte product overflows uint64");
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("local factor count or byte sum overflows uint64");
    return a + b;
}
void finite(Complex value) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::overflow_error("local factor numerical contraction is non-finite");
}
void accumulate(Complex term, Complex& sum, Complex& correction) {
    finite(term);
    const Complex next = sum + term;
    const double re = std::abs(sum.real()) >= std::abs(term.real())
        ? (sum.real() - next.real()) + term.real() : (term.real() - next.real()) + sum.real();
    const double im = std::abs(sum.imag()) >= std::abs(term.imag())
        ? (sum.imag() - next.imag()) + term.imag() : (term.imag() - next.imag()) + sum.imag();
    correction += Complex(re, im);
    sum = next;
    finite(sum);
    finite(correction);
}
struct Sum {
    Complex sum{}, correction{};
    void add(Complex value) { accumulate(value, sum, correction); }
    Complex value() const { const auto result = sum + correction; finite(result); return result; }
};

void extent(std::uint64_t bytes) {
    // Bound vector allocation, pointer arithmetic and canonical SHA bit count.
    if (bytes / 16 > std::vector<Complex>().max_size()
        || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uint64_t>::max() / 8 - 8192)
        throw std::length_error("local factor payload exceeds native or SHA extent");
}

void require_objects(const PeriodicCorrelationAdmittedReference& reference,
                     const PeriodicCorrelationWannier& wannier,
                     const PeriodicCorrelationPAODomain& domain,
                     const PeriodicCorrelationPAOSpace& space,
                     const Selection& selection) {
    periodic_correlation_local_detail::validate_wannier(reference, wannier);
    periodic_correlation_local_detail::validate_pao(reference, domain, space);
    const auto& state = reference.state();
    if (selection.occupied_index >= state.n_correlated_occupied()
        || selection.occupied_cell >= state.n_kpoints()
        || selection.virtual_translation_cell >= state.n_kpoints()
        || selection.virtual_count == 0 || selection.virtual_begin >= space.retained_dimension()
        || selection.virtual_count > space.retained_dimension() - selection.virtual_begin)
        throw std::out_of_range("local factor orbital selection or modular translation is out of range");
}

std::uint64_t coefficient_work(std::uint64_t nao, std::uint64_t neff,
                               std::uint64_t domain, std::uint64_t virtuals) {
    // Conservative loop units: occupied AO*band scan, and per virtual column
    // AO*domain phase scatter, S*t, virtual projections/expansion, zero/finalize.
    return add(product(nao, add(neff, 1)), product(virtuals,
        add(product(nao, domain), add(product(nao, nao),
            add(product(product(2, nao), neff), product(4, nao))))));
}

void node_inventory(const PeriodicCorrelationAdmittedReference& reference, Memory& memory) {
    const auto& dims = reference.dimensions();
    const auto& budget = reference.budget();
    const auto worker = add(memory.peak_owned_numerical_bytes,
        add(memory.caller_gauge_bytes, add(memory.live_wannier_bytes,
        add(memory.live_domain_bytes, add(memory.live_space_bytes,
        add(memory.live_reader_numeric_bytes, memory.live_reader_control_bytes))))));
    memory.required_node_memory_bytes = add(add(dims.external_bytes, dims.shared_bytes),
        add(product(budget.mpi_ranks, add(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            product(product(budget.mpi_ranks, budget.workers_per_rank), worker)));
}

void require_caps(const PeriodicCorrelationAdmittedReference& reference, const Memory& memory,
                  const PeriodicCorrelationLocalFactorCaps& caps, bool factors) {
    if (caps.maximum_owned_numerical_bytes == 0
        || memory.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("local factor owned numerical byte cap is missing or exceeded");
    if (caps.maximum_work_units == 0 || memory.work_units > caps.maximum_work_units)
        throw std::length_error("local factor work cap is missing or exceeded");
    if (factors && (caps.maximum_tile_visits == 0 || memory.tile_visits > caps.maximum_tile_visits
        || caps.maximum_reader_tile_bytes == 0
        || memory.maximum_reader_tile_bytes > caps.maximum_reader_tile_bytes))
        throw std::length_error("local factor reader tile or visit cap is missing or exceeded");
    if (reference.budget().memory_limit_bytes == 0
        || memory.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("local factor live inventory exceeds admitted node memory");
}

std::string require_gauges(const Memory& memory, const PeriodicCorrelationWannier& wannier,
                           const Complex* gauges, std::size_t count) {
    if (memory.caller_gauge_bytes != wannier.memory().caller_gauge_bytes)
        throw std::logic_error("local factor gauge inventory differs from the sealed Wannier shape");
    return periodic_correlation_local_detail::validate_gauges(wannier, gauges, count);
}

Complex negative_character(std::uint64_t k, std::uint64_t r, const std::array<int, 3>& mesh) {
    Complex result(1, 0);
    for (int axis = 2; axis >= 0; --axis) {
        const auto n = static_cast<std::uint64_t>(mesh[axis]);
        const auto m = k % n, cell = r % n;
        k /= n; r /= n;
        if (m == 0) continue;
        if (n % 2 == 0 && m == n / 2) {
            result *= cell % 2 == 0 ? 1.0 : -1.0;
            continue;
        }
        const auto canonical = std::min(m, n - m);
        const auto residue = (canonical * cell) % n;
        double turns = static_cast<double>(residue) / static_cast<double>(n);
        if (turns > 0.5) turns -= 1;
        const Complex phase(std::cos(kTwoPi * turns), -std::sin(kTwoPi * turns));
        result *= m == canonical ? phase : std::conj(phase);
    }
    finite(result);
    return result;
}

// Fill caller-owned admitted column-major storage. The two AO scratch columns
// are reused: t -> S*t, then t becomes the compensated virtual-output sum.
void fill_coefficients(const PeriodicRestrictedMeanFieldState& state,
                       const Complex* gauges, const PeriodicCorrelationPAODomain& domain,
                       const PeriodicCorrelationPAOSpace& space, const Selection& selection,
                       std::size_t ko, std::size_t kv, Complex* output, Complex* scratch) {
    periodic_correlation_local_detail::fill_occupied_column(state, gauges,
        selection.occupied_index, selection.occupied_cell, ko, output);
    periodic_correlation_local_detail::fill_virtual_columns(state, domain, space,
        selection.virtual_begin, selection.virtual_count, selection.virtual_translation_cell,
        kv, output + state.n_basis(), scratch);
}

void identity_inputs(Digest& identity, const PeriodicCorrelationAdmittedReference& reference,
                     const PeriodicCorrelationWannier& wannier,
                     const PeriodicCorrelationPAODomain& domain,
                     const PeriodicCorrelationPAOSpace& space, const Selection& selection) {
    identity.string(reference.state().state_identity_sha256());
    identity.string(reference.state().calculation_identity());
    identity.string(reference.dimensions().allocation_identity);
    identity.string(wannier.wannier_identity_sha256());
    identity.string(wannier.gauge_payload_sha256());
    identity.string(domain.pao_domain_identity_sha256());
    identity.string(space.pao_space_identity_sha256());
    identity.selection(selection);
}

bool same_descriptor(const PeriodicCorrelationFactorTileDescriptor& a,
                     const PeriodicCorrelationFactorTileDescriptor& b) {
    return a.sequence_index == b.sequence_index && a.q_index == b.q_index
        && a.k_bra_index == b.k_bra_index && a.k_ket_index == b.k_ket_index
        && a.k_ket_reciprocal_wrap == b.k_ket_reciprocal_wrap
        && a.ao_pair_begin == b.ao_pair_begin && a.ao_pair_count == b.ao_pair_count
        && a.auxiliary_begin == b.auxiliary_begin && a.auxiliary_count == b.auxiliary_count
        && a.element_count == b.element_count;
}

}  // namespace

namespace periodic_correlation_local_detail {

void validate_reference(const PeriodicCorrelationAdmittedReference& reference) {
    if (std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("local factors require round-to-nearest arithmetic");
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, zero = 0.0;
    if (!(tiny > 0.0) || std::fma(tiny, one, zero) != tiny)
        throw std::invalid_argument("local factors require gradual underflow");
    if (!reference.state_handle())
        throw std::invalid_argument("local factors require an immutable state owner");
    const auto& state = reference.state();
    const auto& dims = reference.dimensions();
    const auto& plan = reference.plan();
    if (reference.contract_version() != kPeriodicCorrelationAdmittedReferenceContractVersion
        || state.contract_version() != kPeriodicRestrictedMeanFieldStateContractVersion
        || state.reference_kind() != PeriodicMeanFieldReferenceKind::RestrictedHartreeFock
        || state.normalization() != PeriodicMeanFieldNormalizationConvention::UnnormalizedAoBlochSumsUniformFullBzWeights
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || state.is_shift() != std::array<int, 3>{0, 0, 0}
        || dims.symmetry_reduction_requested || !dims.symmetry_mapping_identity.empty()
        || dims.symmetry_representative_count != dims.n_kpoints || dims.symmetry_weight_sum != dims.n_kpoints
        || dims.mesh != state.mesh() || dims.is_shift != state.is_shift()
        || dims.n_kpoints != state.n_kpoints() || dims.n_basis != state.n_basis()
        || dims.n_effective_orbitals != state.n_effective_orbitals()
        || dims.n_home_occupied != state.n_correlated_occupied() || dims.n_home_virtual != state.n_virtual()
        || dims.n_home_total_occupied != state.n_frozen_core() + state.n_correlated_occupied()
        || dims.calculation_identity != state.calculation_identity()
        || plan.calculation_identity != dims.calculation_identity || plan.allocation_identity != dims.allocation_identity
        || plan.allocation_contract_version != dims.allocation_contract_version)
        throw std::invalid_argument("local factor state or allocation provenance disagrees");
}

void validate_wannier(const PeriodicCorrelationAdmittedReference& reference,
                      const PeriodicCorrelationWannier& wannier) {
    validate_reference(reference);
    if (reference.state_handle() != wannier.state_handle())
        throw std::invalid_argument("local factors require identical immutable state owners");
    const auto& state = reference.state();
    if (wannier.contract_version() != kPeriodicCorrelationWannierContractVersion
        || wannier.allocation_identity() != reference.dimensions().allocation_identity
        || wannier.n_cells() != state.n_kpoints() || wannier.n_basis() != state.n_basis()
        || wannier.n_home_occupied() != state.n_correlated_occupied())
        throw std::invalid_argument("local factor Wannier or allocation provenance disagrees");
    (void) wannier.cell_coefficients(0);
}

void validate_pao(const PeriodicCorrelationAdmittedReference& reference,
                  const PeriodicCorrelationPAODomain& domain,
                  const PeriodicCorrelationPAOSpace& space) {
    validate_reference(reference);
    if (reference.state_handle() != domain.state_handle() || reference.state_handle() != space.state_handle())
        throw std::invalid_argument("local factors require identical immutable state owners");
    if (domain.contract_version() != kPeriodicCorrelationPAODomainContractVersion
        || space.contract_version() != kPeriodicCorrelationPAOSpaceContractVersion
        || domain.allocation_identity() != reference.dimensions().allocation_identity
        || space.allocation_identity() != reference.dimensions().allocation_identity
        || space.state_identity_sha256() != reference.state().state_identity_sha256()
        || space.domain_dimension() != domain.domain_dimension()
        || space.pao_domain_identity_sha256() != domain.pao_domain_identity_sha256()
        || space.domain_index_sha256() != domain.domain_index_sha256())
        throw std::invalid_argument("local factor domain or PAO-space provenance disagrees");
    if (!space.usable() || space.retained_dimension() == 0 || domain.domain_dimension() == 0)
        throw std::invalid_argument("local factors require a usable nonzero retained PAO rank");
    (void) domain.column(0);
    (void) space.coefficients_data();
}

void validate_reader(const PeriodicCorrelationAdmittedReference& reference,
                     const PeriodicCorrelationFactorStreamSchedule& schedule,
                     const PeriodicCorrelationPrivateFactorReader& reader) {
    validate_reference(reference);
    const auto& shape = schedule.shape();
    if (schedule.state_handle() != reference.state_handle() || reader.state_handle() != reference.state_handle()
        || schedule.calculation_identity() != reference.state().calculation_identity()
        || schedule.allocation_identity() != reference.dimensions().allocation_identity
        || reader.calculation_identity() != schedule.calculation_identity()
        || reader.allocation_identity() != schedule.allocation_identity()
        || reader.schedule_identity_sha256() != schedule.schedule_identity_sha256()
        || reader.state_identity_sha256() != reference.state().state_identity_sha256()
        || reader.tile_count() != shape.tile_count || shape.n_kpoints != reference.state().n_kpoints()
        || shape.n_basis != reference.state().n_basis() || shape.n_auxiliary != reference.dimensions().n_auxiliary
        || !reader.finite_image_reference() || reader.ao_image_source_certified()
        || std::string(reader.image_policy()) != kPeriodicCorrelationThreeCenterImagePolicy)
        throw std::invalid_argument("local factor reader, schedule or finite-source provenance disagrees");
}

std::string validate_gauges(const PeriodicCorrelationWannier& wannier,
                            const Complex* gauges, std::size_t count) {
    const auto& memory = wannier.memory();
    if (gauges == nullptr || count != memory.caller_gauge_bytes / 16
        || reinterpret_cast<std::uintptr_t>(gauges) % alignof(Complex) != 0)
        throw std::invalid_argument("local factor gauge view has invalid shape, pointer or alignment");
    if (memory.caller_gauge_bytes > std::numeric_limits<std::uintptr_t>::max()
        - reinterpret_cast<std::uintptr_t>(gauges))
        throw std::overflow_error("local factor gauge view exceeds pointer extent");
    Digest digest("vibeqc.periodic.correlation.wannier.gauge");
    digest.u64(memory.n_cells); digest.u64(memory.n_home_occupied);
    for (std::size_t i = 0; i < count; ++i) digest.complex(gauges[i]);
    const auto result = digest.finish();
    if (result != wannier.gauge_payload_sha256())
        throw std::invalid_argument("local factor gauge content differs from the sealed Wannier gauge");
    return result;
}

void fill_occupied_column(const PeriodicRestrictedMeanFieldState& state,
                          const Complex* gauges, std::uint64_t occupied_index,
                          std::uint64_t cell, std::size_t ko, Complex* output) {
    const auto nao = static_cast<std::size_t>(state.n_basis());
    const auto no = static_cast<std::size_t>(state.n_correlated_occupied());
    const auto& occupied = state.correlated_occupied_mask(ko);
    const auto& co = state.coefficients(ko);
    const auto* u = gauges + ko * no * no;
    const auto phase_o = negative_character(ko, cell, state.mesh());
    for (std::size_t mu = 0; mu < nao; ++mu) {
        Sum value;
        std::size_t row = 0;
        for (std::size_t band = 0; band < occupied.size(); ++band) {
            if (occupied[band]) {
                if (row >= no) throw std::logic_error("local factor active mask exceeds its sealed shape");
                value.add(co(mu, band) * u[row * no + occupied_index]);
                ++row;
            }
        }
        if (row != no) throw std::logic_error("local factor active mask changed its sealed shape");
        output[mu] = value.value() * phase_o;
        finite(output[mu]);
    }
}

void fill_virtual_columns(const PeriodicRestrictedMeanFieldState& state,
                         const PeriodicCorrelationPAODomain& domain,
                         const PeriodicCorrelationPAOSpace& space,
                         std::uint64_t begin, std::uint64_t count,
                         std::uint64_t translation_cell, std::size_t kv,
                         Complex* output, Complex* scratch) {
    const auto nao = static_cast<std::size_t>(state.n_basis());
    const auto& cv = state.coefficients(kv);
    const auto& overlap = state.overlap(kv);
    const auto& virtuals = state.virtual_mask(kv);
    const auto phase_v = negative_character(kv, translation_cell, state.mesh());
    auto* t = scratch;
    auto* st = scratch + nao;
    for (std::size_t a = 0; a < count; ++a) {
        auto* column = output + a * nao;
        for (std::size_t mu = 0; mu < nao; ++mu) {
            Sum value;
            for (std::size_t d = 0; d < domain.domain_dimension(); ++d) {
                const auto label = domain.column(d);
                if (label.cell >= state.n_kpoints() || label.ao >= nao)
                    throw std::logic_error("local factor PAO domain label violates its sealed shape");
                if (label.ao == mu)
                    value.add(negative_character(kv, label.cell, state.mesh()) * phase_v
                              * space.coefficient(d, begin + a));
            }
            t[mu] = value.value();
        }
        for (std::size_t mu = 0; mu < nao; ++mu) {
            Sum value;
            for (std::size_t nu = 0; nu < nao; ++nu) value.add(overlap(mu, nu) * t[nu]);
            st[mu] = value.value();
        }
        std::fill_n(t, nao, Complex{});
        std::fill_n(column, nao, Complex{});
        for (std::size_t band = 0; band < virtuals.size(); ++band) {
            if (!virtuals[band]) continue;
            Sum projection;
            for (std::size_t mu = 0; mu < nao; ++mu) projection.add(std::conj(cv(mu, band)) * st[mu]);
            const auto amplitude = projection.value();
            for (std::size_t mu = 0; mu < nao; ++mu)
                accumulate(cv(mu, band) * amplitude, column[mu], t[mu]);
        }
        for (std::size_t mu = 0; mu < nao; ++mu) {
            column[mu] += t[mu];
            finite(column[mu]);
        }
    }
}

}  // namespace periodic_correlation_local_detail

Memory plan_periodic_correlation_local_coefficient_panel(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
    const Selection& selection) {
    require_objects(reference, wannier, domain, space, selection);
    const auto& state = reference.state();
    Memory m;
    m.n_cells = state.n_kpoints(); m.n_basis = state.n_basis();
    m.n_home_occupied = state.n_correlated_occupied();
    m.domain_dimension = domain.domain_dimension();
    m.retained_virtual_dimension = space.retained_dimension(); m.virtual_count = selection.virtual_count;
    m.coefficient_panel_bytes = product(16, product(m.n_basis, add(m.virtual_count, 1)));
    m.coefficient_scratch_bytes = product(32, m.n_basis);
    m.retained_output_bytes = m.coefficient_panel_bytes;
    m.peak_owned_numerical_bytes = add(m.coefficient_panel_bytes, m.coefficient_scratch_bytes);
    const auto w = plan_periodic_correlation_wannier(state.mesh(), m.n_basis, m.n_home_occupied);
    m.caller_gauge_bytes = w.caller_gauge_bytes; m.live_wannier_bytes = w.retained_coefficient_bytes;
    m.live_domain_bytes = add(product(16, m.domain_dimension), product(32, product(m.domain_dimension, m.domain_dimension)));
    m.live_space_bytes = add(product(16, product(m.domain_dimension, m.retained_virtual_dimension)),
                            product(8, add(m.domain_dimension, m.retained_virtual_dimension)));
    if (m.live_domain_bytes != add(domain.memory().retained_domain_index_bytes, domain.memory().retained_matrix_bytes)
        || m.live_space_bytes != space.memory().output_numerical_bytes
        || m.live_wannier_bytes != wannier.memory().retained_coefficient_bytes)
        throw std::logic_error("local factor borrowed numerical inventory is inconsistent");
    m.work_units = add(m.caller_gauge_bytes / 16, add(m.domain_dimension,
        add(coefficient_work(m.n_basis, state.n_effective_orbitals(), m.domain_dimension, m.virtual_count),
            m.retained_output_bytes / 16)));
    extent(m.coefficient_panel_bytes); extent(m.coefficient_scratch_bytes); extent(m.caller_gauge_bytes);
    node_inventory(reference, m);
    return m;
}

Memory plan_periodic_correlation_local_factor_block(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
    const Selection& selection, std::uint64_t q, std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count) {
    auto m = plan_periodic_correlation_local_coefficient_panel(reference, wannier, domain, space, selection);
    const auto& shape = schedule.shape();
    periodic_correlation_local_detail::validate_reader(reference, schedule, reader);
    if (q >= m.n_cells || auxiliary_count == 0 || auxiliary_begin >= shape.n_auxiliary
        || auxiliary_count > shape.n_auxiliary - auxiliary_begin)
        throw std::out_of_range("local factor q or auxiliary slice is out of range");
    const auto first = auxiliary_begin / shape.auxiliary_block;
    const auto last = (auxiliary_begin + auxiliary_count - 1) / shape.auxiliary_block;
    const auto tiles = add(last - first, 1);
    const auto covered_begin = product(first, shape.auxiliary_block);
    const auto covered_end = std::min(product(add(last, 1), shape.auxiliary_block), shape.n_auxiliary);
    const auto maximum_aux = std::min(shape.auxiliary_block, shape.n_auxiliary - covered_begin);
    m.auxiliary_count = auxiliary_count;
    m.retained_output_bytes = product(16, product(auxiliary_count, selection.virtual_count));
    m.compensation_bytes = m.retained_output_bytes;
    m.peak_owned_numerical_bytes = add(add(m.retained_output_bytes, m.compensation_bytes),
                                      add(m.coefficient_panel_bytes, m.coefficient_scratch_bytes));
    m.maximum_reader_tile_bytes = product(16, product(maximum_aux, std::min(shape.ao_pair_block, shape.n_ao_pairs)));
    m.live_reader_numeric_bytes = add(kPeriodicCorrelationPrivateFactorStoreCodecPageBytes, m.maximum_reader_tile_bytes);
    m.live_reader_control_bytes = periodic_correlation_private_factor_store_fixed_control_bytes();
    m.tile_visits = product(m.n_cells, product(shape.ao_pair_tile_count, tiles));
    m.reader_payload_bytes = product(16, product(m.n_cells, product(shape.n_ao_pairs, covered_end - covered_begin)));
    // The reader hashes numerical payload twice (independent record/physical
    // checks); 256 units/visit conservatively cover fixed metadata/digests.
    m.work_units = add(m.caller_gauge_bytes / 16, add(m.domain_dimension,
        add(product(m.n_cells, coefficient_work(m.n_basis, reference.state().n_effective_orbitals(), m.domain_dimension, m.virtual_count)),
        add(product(product(m.n_cells, shape.n_ao_pairs), product(auxiliary_count, m.virtual_count)),
        add(product(2, m.reader_payload_bytes / 16), add(product(256, m.tile_visits), m.retained_output_bytes / 16))))));
    extent(m.retained_output_bytes); extent(m.maximum_reader_tile_bytes);
    // Consumed-tile wire carries sequence plus one SHA string per tile.
    extent(product(80, m.tile_visits));
    node_inventory(reference, m);
    return m;
}

const Complex* PeriodicCorrelationLocalCoefficientPanel::data() const {
    if (!state_ || values_.size() != memory_.retained_output_bytes / 16)
        throw std::logic_error("local coefficient panel is consumed");
    return values_.data();
}
Complex PeriodicCorrelationLocalCoefficientPanel::coefficient(std::size_t ao, std::size_t column) const {
    if (ao >= memory_.n_basis || column > memory_.virtual_count)
        throw std::out_of_range("local coefficient panel index is out of range");
    return data()[column * memory_.n_basis + ao];
}
const Complex* PeriodicCorrelationLocalFactorBlock::data() const {
    if (!state_ || values_.size() != memory_.retained_output_bytes / 16)
        throw std::logic_error("local factor block is consumed");
    return values_.data();
}
Complex PeriodicCorrelationLocalFactorBlock::element(std::size_t auxiliary, std::size_t virtual_column) const {
    if (auxiliary >= memory_.auxiliary_count || virtual_column >= memory_.virtual_count)
        throw std::out_of_range("local factor block index is out of range");
    return data()[auxiliary * memory_.virtual_count + virtual_column];
}

PeriodicCorrelationLocalCoefficientPanel make_periodic_correlation_local_coefficient_panel(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t count, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const Selection& selection,
    std::uint64_t ko, std::uint64_t kv, const PeriodicCorrelationLocalFactorCaps& caps) {
    const auto memory = plan_periodic_correlation_local_coefficient_panel(reference, wannier, domain, space, selection);
    if (ko >= memory.n_cells || kv >= memory.n_cells)
        throw std::out_of_range("local coefficient occupied or virtual k index is out of range");
    require_caps(reference, memory, caps, false);
    (void) require_gauges(memory, wannier, gauges, count);
    PeriodicCorrelationLocalCoefficientPanel result;
    result.memory_ = memory; result.selection_ = selection; result.occupied_k_ = ko; result.virtual_k_ = kv;
    result.values_.resize(static_cast<std::size_t>(memory.retained_output_bytes / 16));
    {
        std::vector<Complex> scratch(static_cast<std::size_t>(memory.coefficient_scratch_bytes / 16));
        fill_coefficients(reference.state(), gauges, domain, space, selection, ko, kv,
                          result.values_.data(), scratch.data());
    }
    result.state_ = reference.state_handle();
    Digest payload("vibeqc.periodic.correlation.local-coefficients.payload");
    payload.u64(memory.n_basis); payload.u64(memory.virtual_count); payload.u64(ko); payload.u64(kv);
    for (const auto value : result.values_) payload.complex(value);
    result.payload_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.local-coefficients.identity");
    identity_inputs(identity, reference, wannier, domain, space, selection);
    identity.u64(ko); identity.u64(kv); identity.string(result.payload_);
    identity.string("Cactive-U;retained-virtual-projector;negative-cell-phase;column-major");
    result.identity_ = identity.finish();
    return result;
}

PeriodicCorrelationLocalFactorBlock build_periodic_correlation_local_factor_block(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t count, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const Selection& selection, std::uint64_t q,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count, Orientation orientation,
    const PeriodicCorrelationLocalFactorCaps& caps) {
    if (orientation != Orientation::OccupiedVirtual && orientation != Orientation::VirtualOccupied)
        throw std::invalid_argument("local factor orientation must be explicitly OV or VO");
    const auto memory = plan_periodic_correlation_local_factor_block(
        reference, schedule, reader, wannier, domain, space, selection, q, auxiliary_begin, auxiliary_count);
    require_caps(reference, memory, caps, true);
    (void) require_gauges(memory, wannier, gauges, count);
    PeriodicCorrelationLocalFactorBlock result;
    result.memory_ = memory; result.selection_ = selection; result.q_ = q;
    result.auxiliary_begin_ = auxiliary_begin; result.orientation_ = orientation;
    result.store_ = reader.storage_identity_sha256();
    result.values_.resize(static_cast<std::size_t>(memory.retained_output_bytes / 16));
    const auto& shape = schedule.shape();
    const auto first = auxiliary_begin / shape.auxiliary_block;
    const auto last = (auxiliary_begin + auxiliary_count - 1) / shape.auxiliary_block;
    Digest consumed("vibeqc.periodic.correlation.local-factors.consumed-tiles");
    consumed.string(result.store_); consumed.u64(memory.tile_visits);
    {
        std::vector<Complex> correction(result.values_.size());
        std::vector<Complex> coefficients(static_cast<std::size_t>(memory.coefficient_panel_bytes / 16));
        std::vector<Complex> scratch(static_cast<std::size_t>(memory.coefficient_scratch_bytes / 16));
        struct Sink {
            PeriodicCorrelationFactorTileDescriptor expected;
            Orientation orientation;
            std::uint64_t nao, virtuals, auxiliary_begin, auxiliary_count, visited = 0;
            const Complex* coefficients;
            Complex* output;
            Complex* correction;
            std::string* source;
            std::string* whitener;
            Digest* consumed;
            static void receive(const PeriodicCorrelationPrivateFactorTileView& tile, void* context) {
                auto& s = *static_cast<Sink*>(context);
                const auto& d = tile.descriptor;
                if (!same_descriptor(s.expected, d) || tile.element_count != d.element_count
                    || tile.data == nullptr || tile.payload_bytes != product(16, d.element_count))
                    throw std::logic_error("local factor verified reader tile changed its expected shape");
                if (tile.source_identity_sha256.size() != 64 || tile.whitener_payload_identity_sha256.size() != 64
                    || tile.payload_identity_sha256.size() != 64)
                    throw std::logic_error("local factor tile lacks sealed source identities");
                if (s.visited == 0) {
                    *s.source = tile.source_identity_sha256;
                    *s.whitener = tile.whitener_payload_identity_sha256;
                } else if (*s.source != tile.source_identity_sha256 || *s.whitener != tile.whitener_payload_identity_sha256) {
                    throw std::invalid_argument("local factor q source or whitener differs across consumed tiles");
                }
                const auto begin = std::max(d.auxiliary_begin, s.auxiliary_begin);
                const auto end = std::min(d.auxiliary_begin + d.auxiliary_count, s.auxiliary_begin + s.auxiliary_count);
                for (std::uint64_t pair = 0; pair < d.ao_pair_count; ++pair) {
                    const auto mu = (d.ao_pair_begin + pair) / s.nao;
                    const auto nu = (d.ao_pair_begin + pair) % s.nao;
                    for (std::uint64_t p = begin; p < end; ++p) {
                        const auto b = tile.data[(p - d.auxiliary_begin) * d.ao_pair_count + pair];
                        for (std::uint64_t a = 0; a < s.virtuals; ++a) {
                            const auto* cv = s.coefficients + (a + 1) * s.nao;
                            const auto product_c = s.orientation == Orientation::OccupiedVirtual
                                ? std::conj(s.coefficients[mu]) * cv[nu]
                                : std::conj(cv[mu]) * s.coefficients[nu];
                            const auto offset = (p - s.auxiliary_begin) * s.virtuals + a;
                            accumulate(product_c * b, s.output[offset], s.correction[offset]);
                        }
                    }
                }
                s.consumed->u64(d.sequence_index); s.consumed->string(tile.payload_identity_sha256);
                ++s.visited;
            }
        } sink{{}, orientation, memory.n_basis, memory.virtual_count, auxiliary_begin, auxiliary_count,
               0, coefficients.data(), result.values_.data(), correction.data(), &result.source_, &result.whitener_, &consumed};
        for (std::uint64_t k = 0; k < shape.n_kpoints; ++k) {
            const auto base = q * shape.tiles_per_q + k * shape.tiles_per_k_bra;
            const auto first_descriptor = schedule.descriptor(base + first);
            const auto ket = first_descriptor.k_ket_index;
            const auto ko = orientation == Orientation::OccupiedVirtual ? k : ket;
            const auto kv = orientation == Orientation::OccupiedVirtual ? ket : k;
            fill_coefficients(reference.state(), gauges, domain, space, selection, ko, kv,
                              coefficients.data(), scratch.data());
            for (std::uint64_t pair_tile = 0; pair_tile < shape.ao_pair_tile_count; ++pair_tile) {
                for (std::uint64_t aux_tile = first; aux_tile <= last; ++aux_tile) {
                    const auto sequence = base + pair_tile * shape.auxiliary_tile_count + aux_tile;
                    sink.expected = schedule.descriptor(sequence);
                    reader.visit_tile(sequence, caps.maximum_reader_tile_bytes, &Sink::receive, &sink);
                }
            }
        }
        if (sink.visited != memory.tile_visits)
            throw std::logic_error("local factor tile traversal did not complete its admitted census");
        const double n = static_cast<double>(shape.n_kpoints);
        const double normalization = (1.0 / n) / std::sqrt(n);
        if (!std::isfinite(normalization) || normalization <= 0)
            throw std::overflow_error("local factor inverse-BvK normalization is not positive finite");
        for (std::size_t i = 0; i < result.values_.size(); ++i) {
            result.values_[i] = (result.values_[i] + correction[i]) * normalization;
            finite(result.values_[i]);
        }
    }
    result.state_ = reference.state_handle(); result.consumed_ = consumed.finish();
    Digest payload("vibeqc.periodic.correlation.local-factors.payload");
    payload.u64(q); payload.u64(auxiliary_begin); payload.u64(auxiliary_count);
    payload.u64(memory.virtual_count); payload.u32(static_cast<std::uint32_t>(orientation));
    for (const auto value : result.values_) payload.complex(value);
    result.payload_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.local-factors.identity");
    identity_inputs(identity, reference, wannier, domain, space, selection);
    identity.u64(q); identity.u64(auxiliary_begin); identity.u64(auxiliary_count);
    identity.u32(static_cast<std::uint32_t>(orientation));
    identity.string(result.store_); identity.string(result.source_); identity.string(result.whitener_);
    identity.string(result.consumed_); identity.string(result.payload_);
    identity.string("global-auxiliary-RI;Nk^-3/2;coherent-k-sum;explicit-OV-or-VO;complex-retained");
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
