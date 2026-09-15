#include "vibeqc/periodic_correlation_density_factors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "periodic_correlation_local_factors_internal.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
using Memory = PeriodicCorrelationDensityFactorMemoryPlan;
using Kind = PeriodicCorrelationDensityFactorKind;
using Virtual = PeriodicCorrelationVirtualBlockSelection;
using Occupied = PeriodicCorrelationPlacedOccupied;
using Digest = periodic_correlation_local_detail::Digest;
namespace local = periodic_correlation_local_detail;
static_assert(sizeof(Complex) == 16 && sizeof(Occupied) == 16,
              "density factor accounting requires complex128 and two uint64 indices");

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("density factor count or byte product overflows uint64");
    return a * b;
}
std::uint64_t plus(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("density factor count or byte sum overflows uint64");
    return a + b;
}
void extent(std::uint64_t bytes) {
    if (bytes / 16 > std::vector<Complex>().max_size()
        || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uint64_t>::max() / 8 - 8192)
        throw std::length_error("density factor payload exceeds native or SHA extent");
}
void finite(Complex z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::overflow_error("density factor contraction is non-finite");
}
void accumulate(Complex value, Complex& sum, Complex& correction) {
    finite(value);
    const auto next = sum + value;
    const double re = std::abs(sum.real()) >= std::abs(value.real())
        ? (sum.real() - next.real()) + value.real() : (value.real() - next.real()) + sum.real();
    const double im = std::abs(sum.imag()) >= std::abs(value.imag())
        ? (sum.imag() - next.imag()) + value.imag() : (value.imag() - next.imag()) + sum.imag();
    correction += Complex(re, im);
    sum = next;
    finite(sum); finite(correction);
}

Memory common_plan(const PeriodicCorrelationAdmittedReference& reference,
                   const PeriodicCorrelationFactorStreamSchedule& schedule,
                   const PeriodicCorrelationPrivateFactorReader& reader,
                   std::uint64_t left, std::uint64_t right, std::uint64_t q,
                   std::uint64_t begin, std::uint64_t count) {
    local::validate_reference(reference);
    local::validate_reader(reference, schedule, reader);
    const auto& shape = schedule.shape();
    if (!left || !right || q >= shape.n_kpoints || !count || begin >= shape.n_auxiliary
        || count > shape.n_auxiliary - begin)
        throw std::out_of_range("density factor rank, q or auxiliary slice is out of range");
    Memory m;
    m.n_cells = shape.n_kpoints; m.n_basis = shape.n_basis;
    m.left_count = left; m.right_count = right; m.auxiliary_count = count;
    m.retained_output_bytes = mul(16, mul(count, mul(left, right)));
    m.compensation_bytes = m.retained_output_bytes;
    m.coefficient_panel_bytes = mul(16, mul(m.n_basis, plus(left, right)));
    const auto first = begin / shape.auxiliary_block;
    const auto last = (begin + count - 1) / shape.auxiliary_block;
    const auto covered_begin = mul(first, shape.auxiliary_block);
    const auto covered_end = std::min(mul(plus(last, 1), shape.auxiliary_block), shape.n_auxiliary);
    m.maximum_reader_tile_bytes = mul(16, mul(std::min(shape.auxiliary_block, shape.n_auxiliary - covered_begin),
                                             std::min(shape.ao_pair_block, shape.n_ao_pairs)));
    m.live_reader_numeric_bytes = plus(kPeriodicCorrelationPrivateFactorStoreCodecPageBytes, m.maximum_reader_tile_bytes);
    m.live_reader_control_bytes = periodic_correlation_private_factor_store_fixed_control_bytes();
    m.tile_visits = mul(m.n_cells, mul(shape.ao_pair_tile_count, plus(last - first, 1)));
    m.reader_payload_bytes = mul(16, mul(m.n_cells, mul(shape.n_ao_pairs, covered_end - covered_begin)));
    m.work_units = plus(mul(mul(m.n_cells, shape.n_ao_pairs), mul(count, mul(left, right))),
        plus(mul(2, m.reader_payload_bytes / 16), plus(mul(256, m.tile_visits), m.retained_output_bytes / 16)));
    extent(m.retained_output_bytes); extent(m.coefficient_panel_bytes); extent(m.maximum_reader_tile_bytes);
    extent(mul(80, m.tile_visits));
    return m;
}

void finish_plan(const PeriodicCorrelationAdmittedReference& reference, Memory& m) {
    m.peak_owned_numerical_bytes = plus(plus(m.retained_output_bytes, m.compensation_bytes),
        plus(m.retained_index_bytes, plus(m.coefficient_panel_bytes, m.coefficient_scratch_bytes)));
    const auto live = plus(m.peak_owned_numerical_bytes, plus(m.caller_index_bytes,
        plus(m.caller_gauge_bytes, plus(m.live_wannier_bytes, plus(m.live_domain_bytes,
        plus(m.live_space_bytes, plus(m.live_reader_numeric_bytes, m.live_reader_control_bytes)))))));
    const auto& d = reference.dimensions();
    const auto& b = reference.budget();
    m.required_node_memory_bytes = plus(plus(d.external_bytes, d.shared_bytes),
        plus(mul(b.mpi_ranks, plus(d.per_rank_bytes, d.localization_window_bytes_per_rank)),
             mul(mul(b.mpi_ranks, b.workers_per_rank), live)));
}
void admit(const PeriodicCorrelationAdmittedReference& reference, const Memory& m,
           const PeriodicCorrelationLocalFactorCaps& caps) {
    if (!caps.maximum_owned_numerical_bytes || m.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("density factor owned numerical/index byte cap is missing or exceeded");
    if (!caps.maximum_work_units || m.work_units > caps.maximum_work_units)
        throw std::length_error("density factor work cap is missing or exceeded");
    if (!caps.maximum_tile_visits || m.tile_visits > caps.maximum_tile_visits
        || !caps.maximum_reader_tile_bytes || m.maximum_reader_tile_bytes > caps.maximum_reader_tile_bytes)
        throw std::length_error("density factor reader tile or visit cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || m.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("density factor live inventory exceeds admitted node memory");
}
std::uint64_t pairs(std::uint64_t n) {
    return n % 2 == 0 ? mul(n / 2, n - 1) : mul(n, (n - 1) / 2);
}
void validate_indices(const std::uint64_t* values, std::size_t accessible, std::size_t count,
                      const PeriodicRestrictedMeanFieldState& state) {
    const auto needed = mul(count, 2);
    const auto bytes = mul(needed, sizeof(std::uint64_t));
    if (!values || accessible != needed || reinterpret_cast<std::uintptr_t>(values) % alignof(std::uint64_t))
        throw std::invalid_argument("density factor occupied list requires an aligned uint64 [count,2] view");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(values))
        throw std::overflow_error("density factor occupied list exceeds pointer extent");
    for (std::size_t i = 0; i < count; ++i) {
        if (values[2 * i] >= state.n_correlated_occupied() || values[2 * i + 1] >= state.n_kpoints())
            throw std::out_of_range("density factor occupied index or modular cell is out of range");
        for (std::size_t j = 0; j < i; ++j)
            if (values[2 * i] == values[2 * j] && values[2 * i + 1] == values[2 * j + 1])
                throw std::invalid_argument("density factor occupied list contains a duplicate orbital");
    }
}
void validate_virtual(const Virtual& v, const PeriodicRestrictedMeanFieldState& state,
                      const PeriodicCorrelationPAOSpace& space) {
    if (!v.count || v.begin >= space.retained_dimension() || v.count > space.retained_dimension() - v.begin
        || v.translation_cell >= state.n_kpoints())
        throw std::out_of_range("density factor virtual slice or modular translation is out of range");
}
void virtual_wire(Digest& digest, const Virtual& v) {
    digest.u64(v.begin); digest.u64(v.count); digest.u64(v.translation_cell);
}
void base_identity(Digest& digest, const PeriodicCorrelationAdmittedReference& reference,
                   const std::string& selection, const std::string& store,
                   const std::string& source, const std::string& whitener,
                   const std::string& consumed, const std::string& payload) {
    digest.string(reference.state().state_identity_sha256());
    digest.string(reference.state().calculation_identity());
    digest.string(reference.dimensions().allocation_identity);
    digest.string(selection); digest.string(store); digest.string(source); digest.string(whitener);
    digest.string(consumed); digest.string(payload);
    digest.string("global-auxiliary-RI;zero-mode-omitted;Nk^-3/2;coherent-k-sum;complex-retained");
}
std::string payload_digest(Kind kind, const Memory& m, std::uint64_t q, std::uint64_t begin,
                            const std::vector<Complex>& values) {
    Digest digest("vibeqc.periodic.correlation.density-factors.payload");
    digest.u32(static_cast<std::uint32_t>(kind)); digest.u64(q); digest.u64(begin);
    digest.u64(m.auxiliary_count); digest.u64(m.left_count); digest.u64(m.right_count);
    for (auto value : values) digest.complex(value);
    return digest.finish();
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

// One common coherent contraction. Coefficient callbacks are native stack
// contexts, not std::function or Python callbacks. No scratch survives return.
std::string consume(const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const Memory& m,
    std::uint64_t q, std::uint64_t begin, const PeriodicCorrelationLocalFactorCaps& caps,
    std::vector<Complex>& output, std::string& source, std::string& whitener,
    void (*fill)(std::size_t, std::size_t, Complex*, Complex*, void*), void* context) {
    const auto& shape = schedule.shape();
    const auto first = begin / shape.auxiliary_block;
    const auto last = (begin + m.auxiliary_count - 1) / shape.auxiliary_block;
    Digest consumed("vibeqc.periodic.correlation.density-factors.consumed-tiles");
    consumed.string(reader.storage_identity_sha256()); consumed.u64(m.tile_visits);
    std::vector<Complex> correction(output.size());
    std::vector<Complex> coefficients(static_cast<std::size_t>(m.coefficient_panel_bytes / 16));
    std::vector<Complex> scratch(static_cast<std::size_t>(m.coefficient_scratch_bytes / 16));
    struct Sink {
        PeriodicCorrelationFactorTileDescriptor expected;
        const Memory& m;
        std::uint64_t begin, visited = 0;
        const Complex* coefficients;
        Complex* output;
        Complex* correction;
        std::string& source;
        std::string& whitener;
        Digest& consumed;
        static void receive(const PeriodicCorrelationPrivateFactorTileView& tile, void* context) {
            auto& s = *static_cast<Sink*>(context);
            const auto& d = tile.descriptor;
            if (!same_descriptor(s.expected, d) || tile.element_count != d.element_count
                || !tile.data || tile.payload_bytes != mul(16, d.element_count))
                throw std::logic_error("density factor verified tile differs from its expected shape");
            if (tile.source_identity_sha256.size() != 64 || tile.whitener_payload_identity_sha256.size() != 64
                || tile.payload_identity_sha256.size() != 64)
                throw std::logic_error("density factor tile lacks sealed source identities");
            if (s.visited == 0) {
                s.source = tile.source_identity_sha256; s.whitener = tile.whitener_payload_identity_sha256;
            } else if (s.source != tile.source_identity_sha256 || s.whitener != tile.whitener_payload_identity_sha256) {
                throw std::invalid_argument("density factor q source or whitener changed within the block");
            }
            const auto lo = std::max(d.auxiliary_begin, s.begin);
            const auto hi = std::min(d.auxiliary_begin + d.auxiliary_count, s.begin + s.m.auxiliary_count);
            const auto* right = s.coefficients + s.m.left_count * s.m.n_basis;
            for (std::uint64_t pair = 0; pair < d.ao_pair_count; ++pair) {
                const auto mu = (d.ao_pair_begin + pair) / s.m.n_basis;
                const auto nu = (d.ao_pair_begin + pair) % s.m.n_basis;
                for (std::uint64_t p = lo; p < hi; ++p) {
                    const auto factor = tile.data[(p - d.auxiliary_begin) * d.ao_pair_count + pair];
                    for (std::uint64_t l = 0; l < s.m.left_count; ++l) {
                        const auto left = std::conj(s.coefficients[l * s.m.n_basis + mu]);
                        for (std::uint64_t r = 0; r < s.m.right_count; ++r) {
                            const auto offset = ((p - s.begin) * s.m.left_count + l) * s.m.right_count + r;
                            accumulate((left * right[r * s.m.n_basis + nu]) * factor,
                                       s.output[offset], s.correction[offset]);
                        }
                    }
                }
            }
            s.consumed.u64(d.sequence_index); s.consumed.string(tile.payload_identity_sha256);
            ++s.visited;
        }
    } sink{{}, m, begin, 0, coefficients.data(), output.data(), correction.data(), source, whitener, consumed};
    for (std::uint64_t k = 0; k < shape.n_kpoints; ++k) {
        const auto base = q * shape.tiles_per_q + k * shape.tiles_per_k_bra;
        const auto ket = schedule.descriptor(base + first).k_ket_index;
        fill(k, ket, coefficients.data(), scratch.data(), context);
        for (std::uint64_t pair_tile = 0; pair_tile < shape.ao_pair_tile_count; ++pair_tile) {
            for (std::uint64_t auxiliary_tile = first; auxiliary_tile <= last; ++auxiliary_tile) {
                const auto sequence = base + pair_tile * shape.auxiliary_tile_count + auxiliary_tile;
                sink.expected = schedule.descriptor(sequence);
                reader.visit_tile(sequence, caps.maximum_reader_tile_bytes, &Sink::receive, &sink);
            }
        }
    }
    if (sink.visited != m.tile_visits) throw std::logic_error("density factor traversal changed its admitted tile count");
    const double n = static_cast<double>(shape.n_kpoints);
    const double normalization = (1.0 / n) / std::sqrt(n);
    if (!std::isfinite(normalization) || normalization <= 0)
        throw std::overflow_error("density factor inverse-BvK normalization is not positive finite");
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = (output[i] + correction[i]) * normalization;
        finite(output[i]);
    }
    return consumed.finish();
}

}  // namespace

Memory plan_periodic_correlation_occupied_density_factor_block(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    std::uint64_t left, std::uint64_t right, std::uint64_t q, std::uint64_t begin, std::uint64_t count) {
    local::validate_wannier(reference, wannier);
    const auto maximum = mul(reference.state().n_kpoints(), reference.state().n_correlated_occupied());
    if (left > maximum || right > maximum)
        throw std::out_of_range("density factor occupied list exceeds the finite-torus active count");
    auto m = common_plan(reference, schedule, reader, left, right, q, begin, count);
    const auto rows = plus(left, right);
    m.retained_index_bytes = mul(16, rows); m.caller_index_bytes = m.retained_index_bytes;
    const auto w = plan_periodic_correlation_wannier(reference.state().mesh(), m.n_basis,
                                                   reference.state().n_correlated_occupied());
    m.caller_gauge_bytes = w.caller_gauge_bytes; m.live_wannier_bytes = w.retained_coefficient_bytes;
    if (m.live_wannier_bytes != wannier.memory().retained_coefficient_bytes)
        throw std::logic_error("density factor live Wannier inventory differs from its seal");
    m.work_units = plus(m.work_units, plus(m.caller_gauge_bytes / 16,
        plus(plus(pairs(left), pairs(right)), plus(mul(3, rows),
            mul(m.n_cells, mul(rows, mul(m.n_basis, plus(reference.state().n_effective_orbitals(), 1))))))));
    extent(m.retained_index_bytes); extent(m.caller_gauge_bytes);
    if (left > std::vector<Occupied>().max_size() || right > std::vector<Occupied>().max_size())
        throw std::length_error("density factor occupied lists exceed vector extent");
    finish_plan(reference, m);
    return m;
}

Memory plan_periodic_correlation_virtual_density_factor_block(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const Virtual& left, const Virtual& right,
    std::uint64_t q, std::uint64_t begin, std::uint64_t count) {
    local::validate_pao(reference, domain, space);
    validate_virtual(left, reference.state(), space); validate_virtual(right, reference.state(), space);
    auto m = common_plan(reference, schedule, reader, left.count, right.count, q, begin, count);
    m.coefficient_scratch_bytes = mul(32, m.n_basis);
    const auto d = domain.domain_dimension(), r = space.retained_dimension();
    m.live_domain_bytes = plus(mul(16, d), mul(32, mul(d, d)));
    m.live_space_bytes = plus(mul(16, mul(d, r)), mul(8, plus(d, r)));
    if (m.live_domain_bytes != plus(domain.memory().retained_domain_index_bytes, domain.memory().retained_matrix_bytes)
        || m.live_space_bytes != space.memory().output_numerical_bytes)
        throw std::logic_error("density factor borrowed PAO inventory differs from its seal");
    const auto column_work = plus(mul(m.n_basis, d), plus(mul(m.n_basis, m.n_basis),
        plus(mul(mul(2, m.n_basis), reference.state().n_effective_orbitals()), mul(4, m.n_basis))));
    m.work_units = plus(m.work_units, plus(d, mul(m.n_cells, mul(plus(left.count, right.count), column_work))));
    extent(m.coefficient_scratch_bytes);
    finish_plan(reference, m);
    return m;
}

const Complex* PeriodicCorrelationDensityFactorBlock::data() const {
    if (!state_ || values_.size() != memory_.retained_output_bytes / 16)
        throw std::logic_error("density factor block is consumed");
    return values_.data();
}
Complex PeriodicCorrelationDensityFactorBlock::element(std::size_t auxiliary, std::size_t left, std::size_t right) const {
    if (auxiliary >= memory_.auxiliary_count || left >= memory_.left_count || right >= memory_.right_count)
        throw std::out_of_range("density factor block index is out of range");
    return data()[(auxiliary * memory_.left_count + left) * memory_.right_count + right];
}
Occupied PeriodicCorrelationDensityFactorBlock::left_occupied(std::size_t index) const {
    (void) data();
    if (kind_ != Kind::OccupiedOccupied || index >= left_.size())
        throw std::out_of_range("density factor has no selected left occupied index");
    return left_[index];
}
Occupied PeriodicCorrelationDensityFactorBlock::right_occupied(std::size_t index) const {
    (void) data();
    if (kind_ != Kind::OccupiedOccupied || index >= right_.size())
        throw std::out_of_range("density factor has no selected right occupied index");
    return right_[index];
}
Virtual PeriodicCorrelationDensityFactorBlock::left_virtual() const {
    (void) data();
    if (kind_ != Kind::VirtualVirtual) throw std::logic_error("occupied density factor has no virtual block selection");
    return left_virtual_;
}
Virtual PeriodicCorrelationDensityFactorBlock::right_virtual() const {
    (void) data();
    if (kind_ != Kind::VirtualVirtual) throw std::logic_error("occupied density factor has no virtual block selection");
    return right_virtual_;
}

PeriodicCorrelationDensityFactorBlock build_periodic_correlation_occupied_density_factor_block(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t gauge_count,
    const std::uint64_t* left, std::size_t left_accessible, std::size_t left_count,
    const std::uint64_t* right, std::size_t right_accessible, std::size_t right_count,
    std::uint64_t q, std::uint64_t begin, std::uint64_t count, const PeriodicCorrelationLocalFactorCaps& caps) {
    const auto memory = plan_periodic_correlation_occupied_density_factor_block(
        reference, schedule, reader, wannier, left_count, right_count, q, begin, count);
    admit(reference, memory, caps);
    validate_indices(left, left_accessible, left_count, reference.state());
    validate_indices(right, right_accessible, right_count, reference.state());
    (void) local::validate_gauges(wannier, gauges, gauge_count);
    PeriodicCorrelationDensityFactorBlock result;
    result.kind_ = Kind::OccupiedOccupied; result.memory_ = memory; result.q_ = q; result.auxiliary_begin_ = begin;
    result.left_.resize(left_count); result.right_.resize(right_count);
    for (std::size_t i = 0; i < left_count; ++i) result.left_[i] = {left[2 * i], left[2 * i + 1]};
    for (std::size_t i = 0; i < right_count; ++i) result.right_[i] = {right[2 * i], right[2 * i + 1]};
    result.values_.resize(static_cast<std::size_t>(memory.retained_output_bytes / 16));
    Digest selected("vibeqc.periodic.correlation.density-factors.selection");
    selected.u32(static_cast<std::uint32_t>(result.kind_)); selected.u64(left_count); selected.u64(right_count);
    for (auto row : result.left_) { selected.u64(row.occupied_index); selected.u64(row.cell); }
    for (auto row : result.right_) { selected.u64(row.occupied_index); selected.u64(row.cell); }
    result.selection_ = selected.finish(); result.store_ = reader.storage_identity_sha256();
    struct Columns {
        const PeriodicRestrictedMeanFieldState& state;
        const Complex* gauges;
        const std::vector<Occupied>& left;
        const std::vector<Occupied>& right;
        static void fill(std::size_t bra, std::size_t ket, Complex* coefficients, Complex*, void* context) {
            const auto& s = *static_cast<Columns*>(context);
            const auto nao = s.state.n_basis();
            for (std::size_t i = 0; i < s.left.size(); ++i)
                local::fill_occupied_column(s.state, s.gauges, s.left[i].occupied_index, s.left[i].cell,
                                            bra, coefficients + i * nao);
            for (std::size_t i = 0; i < s.right.size(); ++i)
                local::fill_occupied_column(s.state, s.gauges, s.right[i].occupied_index, s.right[i].cell,
                                            ket, coefficients + (s.left.size() + i) * nao);
        }
    } columns{reference.state(), gauges, result.left_, result.right_};
    result.consumed_ = consume(schedule, reader, memory, q, begin, caps, result.values_, result.source_,
                              result.whitener_, &Columns::fill, &columns);
    result.state_ = reference.state_handle();
    result.payload_ = payload_digest(result.kind_, memory, q, begin, result.values_);
    Digest identity("vibeqc.periodic.correlation.density-factors.identity");
    base_identity(identity, reference, result.selection_, result.store_, result.source_, result.whitener_, result.consumed_, result.payload_);
    identity.string(wannier.wannier_identity_sha256()); identity.string(wannier.gauge_payload_sha256());
    result.identity_ = identity.finish();
    return result;
}

PeriodicCorrelationDensityFactorBlock build_periodic_correlation_virtual_density_factor_block(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const Virtual& left, const Virtual& right,
    std::uint64_t q, std::uint64_t begin, std::uint64_t count, const PeriodicCorrelationLocalFactorCaps& caps) {
    const auto memory = plan_periodic_correlation_virtual_density_factor_block(
        reference, schedule, reader, domain, space, left, right, q, begin, count);
    admit(reference, memory, caps);
    PeriodicCorrelationDensityFactorBlock result;
    result.kind_ = Kind::VirtualVirtual; result.memory_ = memory; result.q_ = q; result.auxiliary_begin_ = begin;
    result.left_virtual_ = left; result.right_virtual_ = right;
    result.values_.resize(static_cast<std::size_t>(memory.retained_output_bytes / 16));
    Digest selected("vibeqc.periodic.correlation.density-factors.selection");
    selected.u32(static_cast<std::uint32_t>(result.kind_)); virtual_wire(selected, left); virtual_wire(selected, right);
    result.selection_ = selected.finish(); result.store_ = reader.storage_identity_sha256();
    struct Columns {
        const PeriodicRestrictedMeanFieldState& state;
        const PeriodicCorrelationPAODomain& domain;
        const PeriodicCorrelationPAOSpace& space;
        Virtual left, right;
        static void fill(std::size_t bra, std::size_t ket, Complex* coefficients, Complex* scratch, void* context) {
            const auto& s = *static_cast<Columns*>(context);
            local::fill_virtual_columns(s.state, s.domain, s.space, s.left.begin, s.left.count,
                                       s.left.translation_cell, bra, coefficients, scratch);
            local::fill_virtual_columns(s.state, s.domain, s.space, s.right.begin, s.right.count,
                                       s.right.translation_cell, ket, coefficients + s.left.count * s.state.n_basis(), scratch);
        }
    } columns{reference.state(), domain, space, left, right};
    result.consumed_ = consume(schedule, reader, memory, q, begin, caps, result.values_, result.source_,
                              result.whitener_, &Columns::fill, &columns);
    result.state_ = reference.state_handle();
    result.payload_ = payload_digest(result.kind_, memory, q, begin, result.values_);
    Digest identity("vibeqc.periodic.correlation.density-factors.identity");
    base_identity(identity, reference, result.selection_, result.store_, result.source_, result.whitener_, result.consumed_, result.payload_);
    identity.string(domain.pao_domain_identity_sha256()); identity.string(space.pao_space_identity_sha256());
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
