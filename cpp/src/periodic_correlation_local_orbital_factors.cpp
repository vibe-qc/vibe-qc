#include "vibeqc/periodic_correlation_local_orbital_factors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "periodic_correlation_local_factors_internal.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
using Memory = PeriodicCorrelationLocalOrbitalFactorMemoryPlan;
using Occupied = PeriodicCorrelationPlacedOccupied;
using Virtual = PeriodicCorrelationVirtualBlockSelection;
using Caps = PeriodicCorrelationLocalFactorCaps;
using Digest = periodic_correlation_local_detail::Digest;
namespace local = periodic_correlation_local_detail;
static_assert(sizeof(Complex) == 16 && sizeof(Occupied) == 16,
              "local orbital panel requires complex128 and two uint64 labels");

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("local orbital factor count product overflows uint64");
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("local orbital factor count sum overflows uint64");
    return a + b;
}
void extent(std::uint64_t bytes) {
    if (bytes / 16 > std::vector<Complex>().max_size()
        || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uint64_t>::max() / 8 - 8192)
        throw std::length_error("local orbital factor payload exceeds native or SHA extent");
}
void finite(Complex z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::overflow_error("local orbital factor contraction is non-finite");
}
void accumulate(Complex value, Complex& sum, Complex& correction) {
    finite(value);
    const auto next = sum + value;
    const double re = std::abs(sum.real()) >= std::abs(value.real())
        ? (sum.real() - next.real()) + value.real() : (value.real() - next.real()) + sum.real();
    const double im = std::abs(sum.imag()) >= std::abs(value.imag())
        ? (sum.imag() - next.imag()) + value.imag() : (value.imag() - next.imag()) + sum.imag();
    correction += Complex(re, im); sum = next;
    finite(sum); finite(correction);
}
void admit(const PeriodicCorrelationAdmittedReference& reference, const Memory& m, const Caps& caps) {
    if (!caps.maximum_owned_numerical_bytes || m.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("local orbital factor owned numerical/index byte cap is missing or exceeded");
    if (!caps.maximum_work_units || m.work_units > caps.maximum_work_units)
        throw std::length_error("local orbital factor work cap is missing or exceeded");
    if (!caps.maximum_tile_visits || m.tile_visits > caps.maximum_tile_visits
        || !caps.maximum_reader_tile_bytes || m.maximum_reader_tile_bytes > caps.maximum_reader_tile_bytes)
        throw std::length_error("local orbital factor tile or visit cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || m.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("local orbital factor live inventory exceeds admitted node memory");
}
void validate_indices(const std::uint64_t* values, std::size_t accessible, std::size_t count,
                      const PeriodicRestrictedMeanFieldState& state) {
    const auto needed = mul(2, count), bytes = mul(8, needed);
    if (!values || accessible != needed || reinterpret_cast<std::uintptr_t>(values) % alignof(std::uint64_t))
        throw std::invalid_argument("local orbital factor labels require an aligned uint64 [count,2] view");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(values))
        throw std::overflow_error("local orbital factor label pointer extent overflows");
    for (std::size_t i = 0; i < count; ++i) {
        if (values[2 * i] >= state.n_correlated_occupied() || values[2 * i + 1] >= state.n_kpoints())
            throw std::out_of_range("local orbital factor occupied index or modular cell is out of range");
        for (std::size_t j = 0; j < i; ++j)
            if (values[2 * i] == values[2 * j] && values[2 * i + 1] == values[2 * j + 1])
                throw std::invalid_argument("local orbital factor occupied list contains a duplicate");
    }
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

Memory plan_periodic_correlation_local_orbital_factor_panel(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
    std::uint64_t occupied, const Virtual& selected, std::uint64_t q, std::uint64_t begin, std::uint64_t count) {
    local::validate_wannier(reference, wannier);
    local::validate_pao(reference, domain, space);
    local::validate_reader(reference, schedule, reader);
    const auto& state = reference.state();
    const auto& shape = schedule.shape();
    if (!occupied || occupied > mul(state.n_kpoints(), state.n_correlated_occupied()))
        throw std::out_of_range("local orbital factor occupied count exceeds the finite-torus active count");
    if (!selected.count || selected.begin >= space.retained_dimension()
        || selected.count > space.retained_dimension() - selected.begin || selected.translation_cell >= state.n_kpoints())
        throw std::out_of_range("local orbital factor virtual slice or modular translation is out of range");
    if (q >= shape.n_kpoints || !count || begin >= shape.n_auxiliary || count > shape.n_auxiliary - begin)
        throw std::out_of_range("local orbital factor q or auxiliary slice is out of range");
    Memory m;
    m.n_cells = shape.n_kpoints; m.n_basis = shape.n_basis;
    m.occupied_count = occupied; m.virtual_count = selected.count;
    m.orbital_count = add(occupied, selected.count); m.auxiliary_count = count;
    m.retained_output_bytes = mul(16, mul(count, mul(m.orbital_count, m.orbital_count)));
    m.compensation_bytes = m.retained_output_bytes;
    m.coefficient_panel_bytes = mul(32, mul(m.n_basis, m.orbital_count));
    m.coefficient_scratch_bytes = mul(32, m.n_basis);
    m.retained_index_bytes = mul(16, occupied); m.caller_index_bytes = m.retained_index_bytes;
    m.peak_owned_numerical_bytes = add(add(m.retained_output_bytes, m.compensation_bytes),
        add(m.retained_index_bytes, add(m.coefficient_panel_bytes, m.coefficient_scratch_bytes)));
    const auto w = plan_periodic_correlation_wannier(state.mesh(), m.n_basis, state.n_correlated_occupied());
    m.caller_gauge_bytes = w.caller_gauge_bytes; m.live_wannier_bytes = w.retained_coefficient_bytes;
    const auto d = domain.domain_dimension(), r = space.retained_dimension();
    m.live_domain_bytes = add(mul(16, d), mul(32, mul(d, d)));
    m.live_space_bytes = add(mul(16, mul(d, r)), mul(8, add(d, r)));
    if (m.live_wannier_bytes != wannier.memory().retained_coefficient_bytes
        || m.live_domain_bytes != add(domain.memory().retained_domain_index_bytes, domain.memory().retained_matrix_bytes)
        || m.live_space_bytes != space.memory().output_numerical_bytes)
        throw std::logic_error("local orbital factor live orbital inventory differs from its seal");
    const auto first = begin / shape.auxiliary_block, last = (begin + count - 1) / shape.auxiliary_block;
    const auto covered_begin = mul(first, shape.auxiliary_block);
    const auto covered_end = std::min(mul(add(last, 1), shape.auxiliary_block), shape.n_auxiliary);
    m.maximum_reader_tile_bytes = mul(16, mul(std::min(shape.auxiliary_block, shape.n_auxiliary - covered_begin),
                                             std::min(shape.ao_pair_block, shape.n_ao_pairs)));
    m.live_reader_numeric_bytes = add(kPeriodicCorrelationPrivateFactorStoreCodecPageBytes, m.maximum_reader_tile_bytes);
    m.live_reader_control_bytes = periodic_correlation_private_factor_store_fixed_control_bytes();
    m.tile_visits = mul(m.n_cells, mul(shape.ao_pair_tile_count, add(last - first, 1)));
    m.reader_payload_bytes = mul(16, mul(m.n_cells, mul(shape.n_ao_pairs, covered_end - covered_begin)));
    const auto virtual_work = add(mul(m.n_basis, d), add(mul(m.n_basis, m.n_basis),
        add(mul(mul(2, m.n_basis), state.n_effective_orbitals()), mul(4, m.n_basis))));
    const auto coefficient_work = mul(mul(2, m.n_cells), add(mul(selected.count, virtual_work),
        mul(occupied, mul(m.n_basis, add(state.n_effective_orbitals(), 1)))));
    const auto uniqueness_work = occupied % 2 == 0 ? mul(occupied / 2, occupied - 1)
                                                   : mul(occupied, (occupied - 1) / 2);
    const auto contraction_work = mul(mul(m.n_cells, shape.n_ao_pairs), mul(count, mul(m.orbital_count, m.orbital_count)));
    const auto reader_work = add(mul(2, m.reader_payload_bytes / 16), mul(256, m.tile_visits));
    const auto scan_work = add(m.caller_gauge_bytes / 16, add(uniqueness_work, add(mul(3, occupied), d)));
    m.work_units = add(add(contraction_work, reader_work),
        add(m.retained_output_bytes / 16, add(coefficient_work, scan_work)));
    for (auto bytes : {m.retained_output_bytes, m.coefficient_panel_bytes, m.coefficient_scratch_bytes,
                       m.retained_index_bytes, m.caller_gauge_bytes, m.maximum_reader_tile_bytes, mul(80, m.tile_visits)})
        extent(bytes);
    if (occupied > std::vector<Occupied>().max_size())
        throw std::length_error("local orbital factor occupied list exceeds vector extent");
    const auto live = add(m.peak_owned_numerical_bytes, add(m.caller_index_bytes,
        add(m.caller_gauge_bytes, add(m.live_wannier_bytes, add(m.live_domain_bytes,
        add(m.live_space_bytes, add(m.live_reader_numeric_bytes, m.live_reader_control_bytes)))))));
    const auto& dimensions = reference.dimensions();
    const auto& budget = reference.budget();
    m.required_node_memory_bytes = add(add(dimensions.external_bytes, dimensions.shared_bytes),
        add(mul(budget.mpi_ranks, add(dimensions.per_rank_bytes, dimensions.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), live)));
    return m;
}

const Complex* PeriodicCorrelationLocalOrbitalFactorPanel::data() const {
    if (!state_ || values_.size() != memory_.retained_output_bytes / 16 || occupied_.size() != memory_.occupied_count)
        throw std::logic_error("local orbital factor panel is consumed");
    return values_.data();
}
Occupied PeriodicCorrelationLocalOrbitalFactorPanel::occupied(std::size_t index) const {
    (void) data();
    if (index >= occupied_.size()) throw std::out_of_range("local orbital factor occupied label is out of range");
    return occupied_[index];
}
Complex PeriodicCorrelationLocalOrbitalFactorPanel::element(std::size_t auxiliary, std::size_t left, std::size_t right) const {
    if (auxiliary >= memory_.auxiliary_count || left >= memory_.orbital_count || right >= memory_.orbital_count)
        throw std::out_of_range("local orbital factor element is out of range");
    return data()[(auxiliary * memory_.orbital_count + left) * memory_.orbital_count + right];
}

PeriodicCorrelationLocalOrbitalFactorPanel build_periodic_correlation_local_orbital_factor_panel(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t gauge_count, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const std::uint64_t* indices, std::size_t accessible,
    std::size_t occupied_count, const Virtual& selected, std::uint64_t q, std::uint64_t begin,
    std::uint64_t count, const Caps& caps) {
    const auto memory = plan_periodic_correlation_local_orbital_factor_panel(
        reference, schedule, reader, wannier, domain, space, occupied_count, selected, q, begin, count);
    admit(reference, memory, caps);
    validate_indices(indices, accessible, occupied_count, reference.state());
    const auto gauge_sha = local::validate_gauges(wannier, gauges, gauge_count);
    PeriodicCorrelationLocalOrbitalFactorPanel result;
    result.memory_ = memory; result.q_ = q; result.auxiliary_begin_ = begin; result.virtual_ = selected;
    result.occupied_.resize(occupied_count);
    for (std::size_t i = 0; i < occupied_count; ++i) result.occupied_[i] = {indices[2 * i], indices[2 * i + 1]};
    result.values_.resize(static_cast<std::size_t>(memory.retained_output_bytes / 16));
    const auto& state = reference.state();
    const auto& shape = schedule.shape();
    result.store_ = reader.storage_identity_sha256();
    result.ao_ = reader.ao_basis_identity_sha256(); result.auxiliary_ = reader.auxiliary_basis_identity_sha256();
    Digest consumed("vibeqc.periodic.correlation.local-orbital-factors.consumed-tiles");
    consumed.string(result.store_); consumed.u64(memory.tile_visits);
    {
        std::vector<Complex> correction(result.values_.size());
        std::vector<Complex> coefficients(static_cast<std::size_t>(memory.coefficient_panel_bytes / 16));
        std::vector<Complex> scratch(static_cast<std::size_t>(memory.coefficient_scratch_bytes / 16));
        struct Sink {
            PeriodicCorrelationFactorTileDescriptor expected;
            const Memory& memory;
            std::uint64_t begin, visits = 0;
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
                    throw std::logic_error("local orbital factor verified tile differs from expected shape");
                if (tile.source_identity_sha256.size() != 64 || tile.whitener_payload_identity_sha256.size() != 64
                    || tile.payload_identity_sha256.size() != 64)
                    throw std::logic_error("local orbital factor tile lacks sealed source identities");
                if (!s.visits) {
                    s.source = tile.source_identity_sha256; s.whitener = tile.whitener_payload_identity_sha256;
                } else if (s.source != tile.source_identity_sha256 || s.whitener != tile.whitener_payload_identity_sha256) {
                    throw std::invalid_argument("local orbital factor q source or whitener changed within the panel");
                }
                const auto lo = std::max(d.auxiliary_begin, s.begin);
                const auto hi = std::min(d.auxiliary_begin + d.auxiliary_count, s.begin + s.memory.auxiliary_count);
                const auto nao = s.memory.n_basis, n = s.memory.orbital_count;
                const auto* right = s.coefficients + n * nao;
                for (std::uint64_t pair = 0; pair < d.ao_pair_count; ++pair) {
                    const auto mu = (d.ao_pair_begin + pair) / nao, nu = (d.ao_pair_begin + pair) % nao;
                    for (std::uint64_t p = lo; p < hi; ++p) {
                        const auto factor = tile.data[(p - d.auxiliary_begin) * d.ao_pair_count + pair];
                        for (std::uint64_t l = 0; l < n; ++l) {
                            const auto left = std::conj(s.coefficients[l * nao + mu]);
                            for (std::uint64_t r = 0; r < n; ++r) {
                                const auto offset = ((p - s.begin) * n + l) * n + r;
                                accumulate((left * right[r * nao + nu]) * factor, s.output[offset], s.correction[offset]);
                            }
                        }
                    }
                }
                s.consumed.u64(d.sequence_index); s.consumed.string(tile.payload_identity_sha256);
                ++s.visits;
            }
        } sink{{}, memory, begin, 0, coefficients.data(), result.values_.data(), correction.data(),
               result.source_, result.whitener_, consumed};
        const auto first = begin / shape.auxiliary_block, last = (begin + count - 1) / shape.auxiliary_block;
        for (std::uint64_t k = 0; k < shape.n_kpoints; ++k) {
            const auto base = q * shape.tiles_per_q + k * shape.tiles_per_k_bra;
            const auto ket = schedule.descriptor(base + first).k_ket_index;
            for (unsigned side = 0; side < 2; ++side) {
                auto* panel = coefficients.data() + side * memory.n_basis * memory.orbital_count;
                const auto momentum = side == 0 ? k : ket;
                for (std::size_t i = 0; i < occupied_count; ++i) {
                    const auto label = result.occupied_[i];
                    local::fill_occupied_column(state, gauges, label.occupied_index, label.cell,
                                                momentum, panel + i * memory.n_basis);
                }
                local::fill_virtual_columns(state, domain, space, selected.begin, selected.count,
                    selected.translation_cell, momentum, panel + occupied_count * memory.n_basis, scratch.data());
            }
            for (std::uint64_t pair_tile = 0; pair_tile < shape.ao_pair_tile_count; ++pair_tile) {
                for (std::uint64_t auxiliary_tile = first; auxiliary_tile <= last; ++auxiliary_tile) {
                    const auto sequence = base + pair_tile * shape.auxiliary_tile_count + auxiliary_tile;
                    sink.expected = schedule.descriptor(sequence);
                    reader.visit_tile(sequence, caps.maximum_reader_tile_bytes, &Sink::receive, &sink);
                }
            }
        }
        if (sink.visits != memory.tile_visits) throw std::logic_error("local orbital factor traversal changed its admitted count");
        const double nk = static_cast<double>(shape.n_kpoints);
        const double normalization = (1.0 / nk) / std::sqrt(nk);
        if (!std::isfinite(normalization) || normalization <= 0)
            throw std::overflow_error("local orbital factor inverse-BvK normalization is not positive finite");
        for (std::size_t i = 0; i < result.values_.size(); ++i) {
            result.values_[i] = (result.values_[i] + correction[i]) * normalization;
            finite(result.values_[i]);
        }
    }
    result.consumed_ = consumed.finish(); result.state_ = reference.state_handle();
    Digest selection("vibeqc.periodic.correlation.local-orbital-factors.selection");
    selection.u64(occupied_count);
    for (auto label : result.occupied_) { selection.u64(label.occupied_index); selection.u64(label.cell); }
    selection.u64(selected.begin); selection.u64(selected.count); selection.u64(selected.translation_cell);
    result.selection_ = selection.finish();
    Digest basis("vibeqc.periodic.correlation.local-orbital-factors.basis");
    basis.string(state.state_identity_sha256()); basis.string(state.calculation_identity());
    basis.string(reference.dimensions().allocation_identity); basis.string(result.selection_);
    basis.string(wannier.wannier_identity_sha256()); basis.string(gauge_sha);
    basis.string(domain.pao_domain_identity_sha256()); basis.string(space.pao_space_identity_sha256());
    result.basis_ = basis.finish();
    Digest payload("vibeqc.periodic.correlation.local-orbital-factors.payload");
    payload.u64(q); payload.u64(begin); payload.u64(count); payload.u64(memory.orbital_count);
    for (auto value : result.values_) payload.complex(value);
    result.payload_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.local-orbital-factors.identity");
    identity.string(result.basis_); identity.string(result.ao_); identity.string(result.auxiliary_);
    identity.string(result.store_); identity.string(result.source_); identity.string(result.whitener_);
    identity.string(result.consumed_); identity.string(result.payload_);
    identity.string("global-auxiliary-RI;zero-mode-omitted;Nk^-3/2;occupied-first;all-ordered-densities;complex-retained");
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
