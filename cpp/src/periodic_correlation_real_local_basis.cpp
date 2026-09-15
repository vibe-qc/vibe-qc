#include "vibeqc/periodic_correlation_real_local_basis.hpp"

#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {

namespace local = periodic_correlation_local_detail;
namespace real = periodic_correlation_real_local_detail;
using Complex = std::complex<double>;
using Memory = PeriodicCorrelationRealLocalBasisMemoryPlan;
using Options = PeriodicCorrelationRealLocalBasisOptions;
using real::add;
using real::mul;

void options_valid(const Options& options) {
    real::tolerance_pair(options.coefficient_tr_absolute_tolerance, options.coefficient_tr_relative_tolerance);
    real::tolerance_pair(options.orthonormality_absolute_tolerance, options.orthonormality_relative_tolerance);
    real::tolerance_pair(options.fock_absolute_tolerance, options.fock_relative_tolerance);
    if (!std::isfinite(options.maximum_fock_projection_error) || options.maximum_fock_projection_error <= 0.0)
        throw std::invalid_argument("real-local maximum Fock projection error must be finite positive");
}
void option_wire(real::Digest& hash, const Options& o) {
    hash.real(o.coefficient_tr_absolute_tolerance); hash.real(o.coefficient_tr_relative_tolerance);
    hash.real(o.orthonormality_absolute_tolerance); hash.real(o.orthonormality_relative_tolerance);
    hash.real(o.fock_absolute_tolerance); hash.real(o.fock_relative_tolerance);
    hash.real(o.maximum_fock_projection_error);
}

}  // namespace

Memory plan_periodic_correlation_real_local_basis(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
    std::uint64_t occupied, const PeriodicCorrelationVirtualBlockSelection& selected) {
    real::float_environment();
    local::validate_wannier(reference, wannier); local::validate_pao(reference, domain, space);
    const auto& state = reference.state();
    if (!occupied || occupied > mul(state.n_kpoints(), state.n_correlated_occupied()))
        throw std::out_of_range("real-local occupied count exceeds finite-torus active count");
    if (!selected.count || selected.begin >= space.retained_dimension()
        || selected.count > space.retained_dimension() - selected.begin || selected.translation_cell >= state.n_kpoints())
        throw std::out_of_range("real-local virtual slice or modular translation is out of range");
    Memory m;
    m.n_cells = state.n_kpoints(); m.n_basis = state.n_basis();
    m.occupied_count = occupied; m.virtual_count = selected.count; m.orbital_count = add(occupied, selected.count);
    const auto square = mul(m.orbital_count, m.orbital_count);
    m.retained_index_bytes = mul(16, occupied); m.caller_index_bytes = m.retained_index_bytes;
    m.retained_fock_bytes = mul(8, add(add(mul(occupied, occupied), mul(selected.count, selected.count)), mul(occupied, selected.count)));
    m.retained_output_bytes = add(m.retained_index_bytes, m.retained_fock_bytes);
    m.projection_matrix_bytes = mul(64, square);
    m.coefficient_panel_bytes = mul(32, mul(m.n_basis, m.orbital_count));
    m.coefficient_scratch_bytes = mul(32, m.n_basis);
    m.peak_owned_numerical_bytes = add(m.retained_index_bytes,
        add(m.projection_matrix_bytes, add(m.coefficient_panel_bytes, m.coefficient_scratch_bytes)));
    const auto w = plan_periodic_correlation_wannier(state.mesh(), m.n_basis, state.n_correlated_occupied());
    m.caller_gauge_bytes = w.caller_gauge_bytes; m.live_wannier_bytes = w.retained_coefficient_bytes;
    const auto d = domain.domain_dimension(), r = space.retained_dimension();
    m.live_domain_bytes = add(mul(16, d), mul(32, mul(d, d)));
    m.live_space_bytes = add(mul(16, mul(d, r)), mul(8, add(d, r)));
    if (m.live_wannier_bytes != wannier.memory().retained_coefficient_bytes
        || m.live_domain_bytes != add(domain.memory().retained_domain_index_bytes, domain.memory().retained_matrix_bytes)
        || m.live_space_bytes != space.memory().output_numerical_bytes)
        throw std::logic_error("real-local borrowed orbital inventory differs from its seal");
    const auto vc = add(mul(m.n_basis, d), add(mul(m.n_basis, m.n_basis),
        add(mul(mul(2, m.n_basis), state.n_effective_orbitals()), mul(4, m.n_basis))));
    const auto coefficient_work = mul(mul(2, m.n_cells), add(mul(selected.count, vc),
        mul(occupied, mul(m.n_basis, add(state.n_effective_orbitals(), 1)))));
    const auto projection_work = mul(m.n_cells, add(mul(mul(2, m.orbital_count), mul(m.n_basis, m.n_basis)),
        add(mul(mul(2, square), m.n_basis), mul(m.n_basis, m.orbital_count))));
    const auto uniqueness = occupied % 2 == 0 ? mul(occupied / 2, occupied - 1) : mul(occupied, (occupied - 1) / 2);
    // Includes outward bounds, gates, hashes and scalar bookkeeping with a
    // deliberately conservative constant, not an operation-time prediction.
    m.work_units = mul(128, add(add(coefficient_work, projection_work),
        add(m.caller_gauge_bytes / 16, add(uniqueness, add(mul(8, square), mul(4, occupied))))));
    for (auto bytes : {m.retained_output_bytes, m.projection_matrix_bytes, m.coefficient_panel_bytes,
                       m.coefficient_scratch_bytes, m.caller_gauge_bytes}) real::extent(bytes);
    if (square > std::vector<Complex>().max_size() || mul(2, occupied) > std::vector<std::uint64_t>().max_size()
        || m.retained_fock_bytes / 8 > std::vector<double>().max_size())
        throw std::length_error("real-local basis exceeds vector extent");
    const auto live = add(m.peak_owned_numerical_bytes, add(m.caller_index_bytes,
        add(m.caller_gauge_bytes, add(m.live_wannier_bytes, add(m.live_domain_bytes, m.live_space_bytes)))));
    const auto& dimensions = reference.dimensions(); const auto& budget = reference.budget();
    m.required_node_memory_bytes = add(add(dimensions.external_bytes, dimensions.shared_bytes),
        add(mul(budget.mpi_ranks, add(dimensions.per_rank_bytes, dimensions.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), live)));
    return m;
}

void PeriodicCorrelationRealLocalBasis::require_live() const {
    if (!state_ || indices_.size() != memory_.retained_index_bytes / 8 || fock_.size() != memory_.retained_fock_bytes / 8)
        throw std::logic_error("real-local basis owner is consumed");
}
const std::uint64_t* PeriodicCorrelationRealLocalBasis::occupied_indices_data() const { require_live(); return indices_.data(); }
PeriodicCorrelationPlacedOccupied PeriodicCorrelationRealLocalBasis::occupied(std::size_t i) const {
    require_live();
    if (i >= memory_.occupied_count) throw std::out_of_range("real-local occupied label is out of range");
    return {indices_[2*i], indices_[2*i+1]};
}
const double* PeriodicCorrelationRealLocalBasis::f_oo_data() const { require_live(); return fock_.data(); }
const double* PeriodicCorrelationRealLocalBasis::f_vv_data() const {
    return f_oo_data() + memory_.occupied_count * memory_.occupied_count;
}
const double* PeriodicCorrelationRealLocalBasis::f_ov_data() const {
    return f_vv_data() + memory_.virtual_count * memory_.virtual_count;
}
double PeriodicCorrelationRealLocalBasis::fock(std::size_t p, std::size_t q) const {
    if (p >= memory_.orbital_count || q >= memory_.orbital_count)
        throw std::out_of_range("real-local Fock label is out of range");
    const auto o = memory_.occupied_count, v = memory_.virtual_count;
    if (p < o && q < o) return f_oo_data()[p*o+q];
    if (p >= o && q >= o) return f_vv_data()[(p-o)*v+q-o];
    return p < o ? f_ov_data()[p*v+q-o] : f_ov_data()[q*v+p-o];
}

PeriodicCorrelationRealLocalBasis make_periodic_correlation_real_local_basis(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t gauge_count, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const std::uint64_t* labels, std::size_t accessible,
    std::size_t occupied_count, const PeriodicCorrelationVirtualBlockSelection& selected,
    const Options& options, const PeriodicCorrelationRealLocalBasisCaps& caps) {
    const auto memory = plan_periodic_correlation_real_local_basis(reference, wannier, domain, space, occupied_count, selected);
    options_valid(options);
    if (!caps.maximum_owned_numerical_bytes || memory.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("real-local basis owned byte cap is missing or exceeded");
    if (!caps.maximum_work_units || memory.work_units > caps.maximum_work_units)
        throw std::length_error("real-local basis work cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || memory.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("real-local basis live inventory exceeds admitted node memory");
    real::indices(labels, accessible, occupied_count, reference.state());
    const auto gauge_sha = local::validate_gauges(wannier, gauges, gauge_count);
    PeriodicCorrelationRealLocalBasis result;
    result.memory_ = memory; result.options_ = options; result.virtual_ = selected;
    result.indices_.assign(labels, labels + 2*occupied_count);
    const auto& state = reference.state();
    const auto n = memory.orbital_count, nao = memory.n_basis, square = n*n;
    std::vector<Complex> f(static_cast<std::size_t>(square));
    {
        std::vector<Complex> s(static_cast<std::size_t>(square));
        std::vector<Complex> sc(static_cast<std::size_t>(square)), fc(static_cast<std::size_t>(square));
        std::vector<Complex> coefficients(static_cast<std::size_t>(memory.coefficient_panel_bytes / 16));
        std::vector<Complex> scratch(static_cast<std::size_t>(memory.coefficient_scratch_bytes / 16));
        double tr_squared = 0.0;
        for (std::uint64_t k = 0; k < memory.n_cells; ++k) {
            const auto partner = real::negative_cell(k, state.mesh());
            for (unsigned side = 0; side < 2; ++side) {
                auto* panel = coefficients.data() + side*nao*n;
                const auto momentum = side == 0 ? k : partner;
                for (std::size_t i = 0; i < occupied_count; ++i)
                    local::fill_occupied_column(state, gauges, labels[2*i], labels[2*i+1], momentum, panel+i*nao);
                local::fill_virtual_columns(state, domain, space, selected.begin, selected.count,
                    selected.translation_cell, momentum, panel+occupied_count*nao, scratch.data());
            }
            const auto* c = coefficients.data(); const auto* cb = c + nao*n;
            for (std::uint64_t at = 0; at < nao*n; ++at) {
                const auto target = std::conj(c[at]);
                result.diagnostics_.maximum_coefficient_tr_error = std::max(
                    result.diagnostics_.maximum_coefficient_tr_error, real::defect_up(cb[at], target));
                real::norm_difference(tr_squared, cb[at], target);
                if (!real::within(cb[at], target, options.coefficient_tr_absolute_tolerance, options.coefficient_tr_relative_tolerance))
                    throw std::invalid_argument("real-local actual orbital coefficients violate physical time reversal");
            }
            ++result.diagnostics_.inspected_kpoints;
            if (partner == k) ++result.diagnostics_.inspected_trim_points;
            const auto& sk = state.overlap(k); const auto& fk = state.fock(k);
            for (std::uint64_t q = 0; q < n; ++q) {
                for (std::uint64_t mu = 0; mu < nao; ++mu) {
                    real::Sum sv, fv;
                    for (std::uint64_t nu = 0; nu < nao; ++nu) {
                        sv.include(sk(mu,nu)*c[q*nao+nu]); fv.include(fk(mu,nu)*c[q*nao+nu]);
                    }
                    scratch[mu] = sv.value(); scratch[nao+mu] = fv.value();
                }
                for (std::uint64_t p = 0; p < n; ++p) {
                    real::Sum sv, fv;
                    for (std::uint64_t mu = 0; mu < nao; ++mu) {
                        sv.include(std::conj(c[p*nao+mu])*scratch[mu]);
                        fv.include(std::conj(c[p*nao+mu])*scratch[nao+mu]);
                    }
                    real::accumulate(sv.value(), s[p*n+q], sc[p*n+q]);
                    real::accumulate(fv.value(), f[p*n+q], fc[p*n+q]);
                }
            }
        }
        result.diagnostics_.coefficient_tr_frobenius_upper_bound = real::sqrt_up(tr_squared);
        double metric_squared = 0.0;
        const auto weight = 1.0 / static_cast<double>(memory.n_cells);
        for (std::uint64_t p = 0; p < n; ++p) for (std::uint64_t q = 0; q < n; ++q) {
            const auto at = p*n+q;
            const auto overlap = real::finite((s[at]+sc[at])*weight);
            f[at] = real::finite((f[at]+fc[at])*weight);
            const Complex target(p == q ? 1.0 : 0.0, 0.0);
            result.diagnostics_.maximum_orthonormality_error = std::max(
                result.diagnostics_.maximum_orthonormality_error, real::defect_up(overlap, target));
            real::norm_difference(metric_squared, overlap, target);
            if (!real::within(overlap, target, options.orthonormality_absolute_tolerance, options.orthonormality_relative_tolerance))
                throw std::invalid_argument("real-local finite-torus selected orbitals are not orthonormal");
        }
        result.diagnostics_.orthonormality_frobenius_upper_bound = real::sqrt_up(metric_squared);
    }
    // Only complex F remains at this phase, so the compact real output fits
    // below the admitted projection peak without comparable hidden copies.
    result.fock_.resize(static_cast<std::size_t>(memory.retained_fock_bytes / 8));
    double projection_squared = 0.0;
    const auto o = memory.occupied_count, v = memory.virtual_count;
    for (std::uint64_t p = 0; p < n; ++p) for (std::uint64_t q = p; q < n; ++q) {
        const auto a = f[p*n+q], b = f[q*n+p];
        const double projected = real::finite(0.5*a.real() + 0.5*b.real());
        for (const auto raw : {a, b}) {
            result.diagnostics_.maximum_raw_fock_imaginary_magnitude = std::max(
                result.diagnostics_.maximum_raw_fock_imaginary_magnitude, std::abs(raw.imag()));
            if (!real::within(raw, Complex(raw.real(),0.0), options.fock_absolute_tolerance, options.fock_relative_tolerance))
                throw std::invalid_argument("real-local directly projected Fock matrix is not real");
            const auto correction = real::defect_up(raw, Complex(projected,0.0));
            result.diagnostics_.maximum_fock_projection_error = std::max(result.diagnostics_.maximum_fock_projection_error, correction);
            if (correction > options.maximum_fock_projection_error)
                throw std::invalid_argument("real-local Fock projection exceeds explicit error budget");
        }
        result.diagnostics_.maximum_raw_fock_hermitian_error = std::max(
            result.diagnostics_.maximum_raw_fock_hermitian_error, real::defect_up(a, std::conj(b)));
        if (!real::within(a, std::conj(b), options.fock_absolute_tolerance, options.fock_relative_tolerance))
            throw std::invalid_argument("real-local directly projected Fock matrix is not Hermitian");
        real::norm_difference(projection_squared, a, Complex(projected,0.0));
        if (p != q) real::norm_difference(projection_squared, b, Complex(projected,0.0));
        if (q < o) {
            result.fock_[p*o+q] = projected; result.fock_[q*o+p] = projected;
        } else if (p >= o) {
            result.fock_[o*o+(p-o)*v+q-o] = projected; result.fock_[o*o+(q-o)*v+p-o] = projected;
        } else result.fock_[o*o+v*v+p*v+q-o] = projected;
    }
    result.diagnostics_.fock_projection_frobenius_upper_bound = real::sqrt_up(projection_squared);
    result.state_ = reference.state_handle(); result.allocation_ = reference.dimensions().allocation_identity;
    result.basis_ = real::local_basis_identity(reference, wannier, gauge_sha, domain, space, labels, occupied_count, selected);
    if (domain.pao_domain_identity_sha256().size()!=64 || space.pao_space_identity_sha256().size()!=64)
        throw std::logic_error("real-local virtual component identity extent is inconsistent");
    std::copy(domain.pao_domain_identity_sha256().begin(),domain.pao_domain_identity_sha256().end(),
              result.virtual_domain_.begin());
    std::copy(space.pao_space_identity_sha256().begin(),space.pao_space_identity_sha256().end(),
              result.virtual_space_.begin());
    real::Digest payload("vibeqc.periodic.correlation.real-local-basis.payload");
    payload.u64(o); payload.u64(v);
    for (auto label : result.indices_) payload.u64(label);
    for (auto value : result.fock_) payload.real(value);
    result.payload_ = payload.finish();
    real::Digest identity("vibeqc.periodic.correlation.real-local-basis.identity");
    identity.string(result.basis_); identity.string(result.payload_); identity.string(real::kFloatPolicy);
    option_wire(identity, options);
    const auto& d = result.diagnostics_;
    identity.u64(d.inspected_kpoints); identity.u64(d.inspected_trim_points);
    for (auto value : {d.maximum_coefficient_tr_error, d.coefficient_tr_frobenius_upper_bound,
         d.maximum_orthonormality_error, d.orthonormality_frobenius_upper_bound,
         d.maximum_raw_fock_imaginary_magnitude, d.maximum_raw_fock_hermitian_error,
         d.maximum_fock_projection_error, d.fock_projection_frobenius_upper_bound}) identity.real(value);
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
