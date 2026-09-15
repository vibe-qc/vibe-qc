#include "vibeqc/periodic_correlation_pao_overlap.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "periodic_correlation_local_factors_internal.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
using Memory = PeriodicCorrelationPAOOverlapMemoryPlan;
using Selection = PeriodicCorrelationVirtualBlockSelection;
using Digest = periodic_correlation_local_detail::Digest;
namespace local = periodic_correlation_local_detail;
static_assert(sizeof(Complex) == 16, "PAO overlap accounting requires complex128");

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("PAO overlap count or byte product overflows uint64");
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("PAO overlap count or byte sum overflows uint64");
    return a + b;
}
void extent(std::uint64_t bytes) {
    if (bytes / 16 > std::vector<Complex>().max_size()
        || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uint64_t>::max() / 8 - 8192)
        throw std::length_error("PAO overlap payload exceeds native or SHA extent");
}
void finite(Complex z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::overflow_error("PAO overlap contraction is non-finite");
}
void accumulate(Complex term, Complex& sum, Complex& correction) {
    finite(term);
    const auto next = sum + term;
    const double re = std::abs(sum.real()) >= std::abs(term.real())
        ? (sum.real() - next.real()) + term.real() : (term.real() - next.real()) + sum.real();
    const double im = std::abs(sum.imag()) >= std::abs(term.imag())
        ? (sum.imag() - next.imag()) + term.imag() : (term.imag() - next.imag()) + sum.imag();
    correction += Complex(re, im); sum = next;
    finite(sum); finite(correction);
}
struct Sum {
    Complex sum{}, correction{};
    void add(Complex z) { accumulate(z, sum, correction); }
    Complex value() const { const auto z = sum + correction; finite(z); return z; }
};
void validate_selection(const Selection& s, const PeriodicRestrictedMeanFieldState& state,
                         const PeriodicCorrelationPAOSpace& space) {
    if (!s.count || s.begin >= space.retained_dimension() || s.count > space.retained_dimension() - s.begin
        || s.translation_cell >= state.n_kpoints())
        throw std::out_of_range("PAO overlap selected rank or modular translation is out of range");
}
std::uint64_t domain_bytes(const PeriodicCorrelationPAODomain& domain) {
    const auto d = domain.domain_dimension();
    const auto bytes = add(mul(16, d), mul(32, mul(d, d)));
    if (bytes != add(domain.memory().retained_domain_index_bytes, domain.memory().retained_matrix_bytes))
        throw std::logic_error("PAO overlap live domain inventory differs from its seal");
    return bytes;
}
std::uint64_t space_bytes(const PeriodicCorrelationPAOSpace& space) {
    const auto d = space.domain_dimension(), r = space.retained_dimension();
    const auto bytes = add(mul(16, mul(d, r)), mul(8, add(d, r)));
    if (bytes != space.memory().output_numerical_bytes)
        throw std::logic_error("PAO overlap live space inventory differs from its seal");
    return bytes;
}
std::uint64_t column_work(std::uint64_t nao, std::uint64_t neff, std::uint64_t d) {
    return add(mul(nao, d), add(mul(nao, nao), add(mul(mul(2, nao), neff), mul(4, nao))));
}
void selection_wire(Digest& digest, const Selection& s) {
    digest.u64(s.begin); digest.u64(s.count); digest.u64(s.translation_cell);
}

}  // namespace

Memory plan_periodic_correlation_pao_space_overlap(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationPAODomain& left_domain,
    const PeriodicCorrelationPAOSpace& left_space, const Selection& left,
    const PeriodicCorrelationPAODomain& right_domain, const PeriodicCorrelationPAOSpace& right_space,
    const Selection& right) {
    local::validate_pao(reference, left_domain, left_space);
    local::validate_pao(reference, right_domain, right_space);
    const auto& state = reference.state();
    validate_selection(left, state, left_space); validate_selection(right, state, right_space);
    Memory m;
    m.n_cells = state.n_kpoints(); m.n_basis = state.n_basis();
    m.left_count = left.count; m.right_count = right.count;
    m.output_bytes = mul(16, mul(left.count, right.count)); m.compensation_bytes = m.output_bytes;
    m.coefficient_panel_bytes = mul(16, mul(m.n_basis, add(left.count, right.count)));
    m.scratch_bytes = mul(32, m.n_basis);
    m.peak_owned_numerical_bytes = add(add(m.output_bytes, m.compensation_bytes),
                                      add(m.coefficient_panel_bytes, m.scratch_bytes));
    m.unique_domain_owners = std::addressof(left_domain) == std::addressof(right_domain) ? 1 : 2;
    m.unique_space_owners = std::addressof(left_space) == std::addressof(right_space) ? 1 : 2;
    m.live_domain_bytes = domain_bytes(left_domain);
    const auto right_domain_bytes = domain_bytes(right_domain);
    if (m.unique_domain_owners == 2) m.live_domain_bytes = add(m.live_domain_bytes, right_domain_bytes);
    m.live_space_bytes = space_bytes(left_space);
    const auto right_space_bytes = space_bytes(right_space);
    if (m.unique_space_owners == 2) m.live_space_bytes = add(m.live_space_bytes, right_space_bytes);
    const auto coefficient_work = add(mul(left.count, column_work(m.n_basis, state.n_effective_orbitals(), left_domain.domain_dimension())),
        mul(right.count, column_work(m.n_basis, state.n_effective_orbitals(), right_domain.domain_dimension())));
    const auto projection_work = add(mul(right.count, mul(m.n_basis, m.n_basis)),
        mul(mul(left.count, right.count), add(m.n_basis, 1)));
    m.work_units = add(mul(m.n_cells, add(coefficient_work, projection_work)), m.output_bytes / 16);
    extent(m.output_bytes); extent(m.coefficient_panel_bytes); extent(m.scratch_bytes);
    const auto& d = reference.dimensions();
    const auto& b = reference.budget();
    m.required_node_memory_bytes = add(add(d.external_bytes, d.shared_bytes),
        add(mul(b.mpi_ranks, add(d.per_rank_bytes, d.localization_window_bytes_per_rank)),
            mul(mul(b.mpi_ranks, b.workers_per_rank), add(m.peak_owned_numerical_bytes,
                add(m.live_domain_bytes, m.live_space_bytes)))));
    return m;
}

const Complex* PeriodicCorrelationPAOSpaceOverlap::data() const {
    if (!state_ || values_.size() != memory_.output_bytes / 16)
        throw std::logic_error("PAO overlap owner is consumed");
    return values_.data();
}
Complex PeriodicCorrelationPAOSpaceOverlap::element(std::size_t left, std::size_t right) const {
    if (left >= memory_.left_count || right >= memory_.right_count)
        throw std::out_of_range("PAO overlap element is out of range");
    return data()[left * memory_.right_count + right];
}

PeriodicCorrelationPAOSpaceOverlap make_periodic_correlation_pao_space_overlap(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationPAODomain& left_domain,
    const PeriodicCorrelationPAOSpace& left_space, const Selection& left,
    const PeriodicCorrelationPAODomain& right_domain, const PeriodicCorrelationPAOSpace& right_space,
    const Selection& right, const PeriodicCorrelationPAOOverlapCaps& caps) {
    const auto memory = plan_periodic_correlation_pao_space_overlap(
        reference, left_domain, left_space, left, right_domain, right_space, right);
    if (!caps.maximum_owned_numerical_bytes || memory.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("PAO overlap owned numerical byte cap is missing or exceeded");
    if (!caps.maximum_work_units || memory.work_units > caps.maximum_work_units)
        throw std::length_error("PAO overlap work cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || memory.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("PAO overlap live inventory exceeds admitted node memory");
    PeriodicCorrelationPAOSpaceOverlap result;
    result.memory_ = memory; result.left_ = left; result.right_ = right;
    result.values_.resize(static_cast<std::size_t>(memory.output_bytes / 16));
    const auto& state = reference.state();
    const auto nao = static_cast<std::size_t>(memory.n_basis);
    {
        std::vector<Complex> correction(result.values_.size());
        std::vector<Complex> coefficients(static_cast<std::size_t>(memory.coefficient_panel_bytes / 16));
        std::vector<Complex> scratch(static_cast<std::size_t>(memory.scratch_bytes / 16));
        auto* ca = coefficients.data();
        auto* cb = ca + left.count * nao;
        for (std::size_t k = 0; k < memory.n_cells; ++k) {
            local::fill_virtual_columns(state, left_domain, left_space, left.begin, left.count,
                                        left.translation_cell, k, ca, scratch.data());
            local::fill_virtual_columns(state, right_domain, right_space, right.begin, right.count,
                                        right.translation_cell, k, cb, scratch.data());
            const auto& overlap = state.overlap(k);
            // Expansion scratch is now free for S times each right column.
            for (std::size_t r = 0; r < right.count; ++r) {
                for (std::size_t mu = 0; mu < nao; ++mu) {
                    Sum value;
                    for (std::size_t nu = 0; nu < nao; ++nu) value.add(overlap(mu, nu) * cb[r * nao + nu]);
                    scratch[mu] = value.value();
                }
                for (std::size_t l = 0; l < left.count; ++l) {
                    Sum value;
                    for (std::size_t mu = 0; mu < nao; ++mu) value.add(std::conj(ca[l * nao + mu]) * scratch[mu]);
                    const auto at = l * right.count + r;
                    accumulate(value.value(), result.values_[at], correction[at]);
                }
            }
        }
        const double weight = 1.0 / static_cast<double>(memory.n_cells);
        if (!std::isfinite(weight) || weight <= 0)
            throw std::overflow_error("PAO overlap full-BZ weight is not positive finite");
        for (std::size_t i = 0; i < result.values_.size(); ++i) {
            result.values_[i] = (result.values_[i] + correction[i]) * weight;
            finite(result.values_[i]);
        }
    }
    result.state_ = reference.state_handle();
    Digest payload("vibeqc.periodic.correlation.pao-overlap.payload");
    payload.u64(memory.n_cells); payload.u64(memory.left_count); payload.u64(memory.right_count);
    for (auto value : result.values_) payload.complex(value);
    result.payload_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.pao-overlap.identity");
    identity.string(state.state_identity_sha256()); identity.string(state.calculation_identity());
    identity.string(reference.dimensions().allocation_identity);
    identity.string(left_domain.pao_domain_identity_sha256()); identity.string(left_space.pao_space_identity_sha256());
    selection_wire(identity, left);
    identity.string(right_domain.pao_domain_identity_sha256()); identity.string(right_space.pao_space_identity_sha256());
    selection_wire(identity, right);
    identity.string(result.payload_);
    identity.string("retained-virtual-projector;direct-S-overlap;full-BZ-1/Nk;independent-negative-cell-phases;complex-retained");
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
