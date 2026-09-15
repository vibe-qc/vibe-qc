#include "vibeqc/periodic_correlation_iao_pm.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Z = std::complex<double>;
static_assert(sizeof(Z) == 16 && sizeof(double) == 8 && sizeof(U) == 8
              && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
              "IAO PM requires IEEE binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "IAO PM forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "IAO PM requires binary64 evaluation"
#endif
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a) throw std::overflow_error("IAO PM count sum overflow");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a) throw std::overflow_error("IAO PM count product overflow");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("IAO PM arithmetic is not finite");
    return x;
}
void accumulate(Z x, Z& sum, Z& correction) {
    finite(x.real()); finite(x.imag());
    const Z next = sum + x;
    const double re = std::abs(sum.real()) >= std::abs(x.real())
        ? (sum.real() - next.real()) + x.real() : (x.real() - next.real()) + sum.real();
    const double im = std::abs(sum.imag()) >= std::abs(x.imag())
        ? (sum.imag() - next.imag()) + x.imag() : (x.imag() - next.imag()) + sum.imag();
    correction += Z(re, im); sum = next;
    finite(sum.real()); finite(sum.imag()); finite(correction.real()); finite(correction.imag());
}
Z total(Z sum, Z correction) {
    const Z result = sum + correction;
    finite(result.real()); finite(result.imag()); return result;
}
Z phase(const RegularKMesh& mesh, U k, U cell) {
    const auto a = mesh.address(k).doubled, b = mesh.address(cell).doubled;
    double turns = 0.0;
    for (unsigned d = 0; d < 3; ++d) {
        const U n = static_cast<U>(mesh.mesh()[d]);
        const U x = static_cast<U>(a[d] / 2), y = static_cast<U>(b[d] / 2);
        turns += static_cast<double>((x * y) % n) / static_cast<double>(n);
    }
    const double angle = 6.283185307179586476925286766559 * turns;
    return {std::cos(angle), std::sin(angle)};
}
void require_reference(const PeriodicCorrelationAdmittedReference& reference) {
    if (!reference.state_handle() || reference.state().periodic_dimension() != 3
        || reference.state().is_shift() != std::array<int, 3>{0, 0, 0}
        || reference.dimensions().symmetry_reduction_requested
        || reference.plan().admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus) {
        throw std::invalid_argument("IAO PM requires a live full Gamma-centered 3D reference");
    }
}
}  // namespace

PeriodicCorrelationIAOPMMemoryPlan plan_periodic_correlation_iao_pm(
    const PeriodicCorrelationAdmittedReference& reference, U r) {
    require_reference(reference);
    const auto& state = reference.state();
    const U k = state.n_kpoints(), o = state.n_correlated_occupied();
    const U p = add(o, state.n_frozen_core()), e = state.n_effective_orbitals();
    if (!o || p > r || r > e) throw std::invalid_argument("IAO PM minimal/occupied dimensions are invalid");
    PeriodicCorrelationIAOPMMemoryPlan m;
    m.n_points = k; m.n_active = o; m.n_occupied = p; m.n_minimal = r;
    const U koo = mul(k, mul(o, o)), ro = mul(r, o), ko = mul(k, o);
    m.gradient_bytes = m.gradient_compensation_bytes = m.borrowed_gauge_bytes = mul(16, koo);
    m.cell_workspace_bytes = add(mul(72, ro), mul(8, o));
    m.active_index_bytes = mul(8, ko);
    m.peak_owned_numerical_bytes = add(mul(2, m.gradient_bytes), add(m.cell_workspace_bytes, m.active_index_bytes));
    m.borrowed_owner_pointer_bytes = mul(sizeof(const PeriodicCorrelationBlochIAO*), k);
    const U iao_output = add(mul(16, add(add(mul(e, r), mul(r, r)), mul(2, mul(r, p)))), mul(8, add(r, p)));
    m.live_iao_numerical_bytes = mul(k, iao_output);
    const U live = add(m.peak_owned_numerical_bytes,
        add(m.borrowed_gauge_bytes, add(m.borrowed_owner_pointer_bytes, m.live_iao_numerical_bytes)));
    const auto& d = reference.dimensions(); const auto& budget = reference.budget();
    m.required_node_memory_bytes = add(add(d.external_bytes, d.shared_bytes),
        add(mul(budget.mpi_ranks, add(d.per_rank_bytes, d.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), live)));
    // Includes pointer/label/band validation, both unitary products, scalar
    // finite Fourier sums, group search, charge/gradient accumulation and norms.
    const auto preflight_work = mul(k, add(add(mul(r, p), mul(o, mul(o, o))), mul(o, e)));
    const auto cell_work = mul(k, add(mul(k, mul(r, mul(o, o))), mul(mul(r, r), o)));
    m.work_units = mul(256, add(add(preflight_work, cell_work), add(koo, 1)));
    const U address_limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.peak_owned_numerical_bytes > address_limit || m.borrowed_gauge_bytes > address_limit
        || m.borrowed_owner_pointer_bytes > address_limit || koo > std::vector<Z>().max_size()) {
        throw std::length_error("IAO PM numerical extent exceeds native address range");
    }
    return m;
}

PeriodicCorrelationIAOPMResult evaluate_periodic_correlation_iao_pm(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationBlochIAO* const* points, std::size_t point_count,
    const Z* gauges, std::size_t gauge_count,
    const PeriodicCorrelationIAOPMOptions& options, const PeriodicCorrelationIAOPMCaps& caps) {
    require_reference(reference);
    const auto pointer_address = reinterpret_cast<std::uintptr_t>(points);
    const U pointer_bytes = mul(sizeof(*points), point_count);
    if (!points || point_count != reference.state().n_kpoints()
        || pointer_address % alignof(const PeriodicCorrelationBlochIAO*)
        || pointer_bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || pointer_bytes > std::numeric_limits<std::uintptr_t>::max() - pointer_address)
        throw std::invalid_argument("IAO PM owner pointer array has an invalid shape, alignment or extent");
    if (!points[0])
        throw std::invalid_argument("IAO PM requires one native IAO owner per full-mesh point");
    const auto memory = plan_periodic_correlation_iao_pm(reference, points[0]->memory().n_minimal);
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < memory.peak_owned_numerical_bytes
        || !caps.maximum_work_units || caps.maximum_work_units < memory.work_units)
        throw std::length_error("IAO PM byte or work cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || reference.budget().memory_limit_bytes < memory.required_node_memory_bytes)
        throw std::length_error("IAO PM live native owners exceed admitted node memory");
    for (const double x : {options.gauge_unitarity_tolerance, options.charge_normalization_tolerance})
        if (!std::isfinite(x) || x <= 0.0 || x >= 1.0) throw std::invalid_argument("IAO PM audit tolerances must lie in (0,1)");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0.0) || std::fma(tiny, one, zero) != tiny)
        throw std::invalid_argument("IAO PM requires round-to-nearest and gradual underflow");
    const U kcount = memory.n_points, o = memory.n_active, r = memory.n_minimal, p = memory.n_occupied;
    const U koo = memory.gradient_bytes / 16, ro = r * o;
    const auto address = reinterpret_cast<std::uintptr_t>(gauges);
    if (!gauges || gauge_count != koo || address % alignof(Z)
        || memory.borrowed_gauge_bytes > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("IAO PM gauges require aligned immutable complex [Nk,active,active] storage");
    const auto& state = reference.state();
    for (U k = 0; k < kcount; ++k) {
        if (!points[k] || points[k]->state_handle().get() != reference.state_handle().get()
            || points[k]->allocation_identity() != reference.dimensions().allocation_identity
            || points[k]->point() != k || points[k]->memory().n_minimal != r
            || points[k]->memory().n_occupied != p
            || points[k]->memory().output_numerical_bytes != memory.live_iao_numerical_bytes / kcount
            || points[k]->declared_provenance().minimal_basis_identity != points[0]->declared_provenance().minimal_basis_identity)
            throw std::invalid_argument("IAO PM point owners, dimensions or minimal basis do not match");
        (void) points[k]->occupied_coefficients_data();
        for (U rho = 0; rho < r; ++rho)
            if (points[k]->atom_label(rho) != points[0]->atom_label(rho))
                throw std::invalid_argument("IAO PM atomic labels differ between k points");
    }
    for (U x = 0; x < koo; ++x) { finite(gauges[x].real()); finite(gauges[x].imag()); }
    PeriodicCorrelationIAOPMResult result; result.memory = memory;
    for (U k = 0; k < kcount; ++k) for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j) {
        Z left{}, lc{}, right{}, rc{};
        for (U l = 0; l < o; ++l) {
            accumulate(std::conj(gauges[(k * o + l) * o + i]) * gauges[(k * o + l) * o + j], left, lc);
            accumulate(gauges[(k * o + i) * o + l] * std::conj(gauges[(k * o + j) * o + l]), right, rc);
        }
        const double error = std::max(std::abs(total(left, lc) - double(i == j)), std::abs(total(right, rc) - double(i == j)));
        result.maximum_unitarity_residual = std::max(result.maximum_unitarity_residual, finite(error));
        if (error > options.gauge_unitarity_tolerance) throw std::invalid_argument("IAO PM active gauges are not unitary");
    }
    std::vector<U> active_columns(kcount * o);
    for (U k = 0; k < kcount; ++k) {
        U active = 0;
        for (U j = 0; j < p; ++j) {
            const U band = points[k]->occupied_band(j);
            if (band >= state.n_effective_orbitals()) throw std::logic_error("IAO PM occupied band is invalid");
            if (state.correlated_occupied_mask(k)[band]) {
                if (active >= o) throw std::logic_error("IAO PM has too many active occupied bands");
                active_columns[k * o + active++] = j;
            }
        }
        if (active != o) throw std::logic_error("IAO PM lost an explicitly active occupied band");
    }
    result.gradient.resize(koo);
    std::vector<Z> gradient_correction(koo), cell(4 * ro);
    std::vector<double> weights(ro), norms(o);
    const RegularKMesh mesh(state.mesh());
    Z objective{}, objective_correction{};
    const double inverse_k = 1.0 / static_cast<double>(kcount);
    for (U cell_index = 0; cell_index < kcount; ++cell_index) {
        std::fill(cell.begin(), cell.end(), Z{});
        auto* v = cell.data(); auto* w = v + ro; auto* vc = w + ro; auto* wc = vc + ro;
        for (U k = 0; k < kcount; ++k) {
            const auto* b = points[k]->occupied_coefficients_data();
            const auto* d = points[k]->occupied_covariant_data();
            const Z character = phase(mesh, k, cell_index) * inverse_k;
            for (U rho = 0; rho < r; ++rho) for (U n = 0; n < o; ++n) {
                Z bv{}, bvc{}, dw{}, dwc{};
                for (U j = 0; j < o; ++j) {
                    const U source = rho * p + active_columns[k * o + j];
                    const Z rotation = gauges[(k * o + j) * o + n];
                    accumulate(b[source] * rotation, bv, bvc); accumulate(d[source] * rotation, dw, dwc);
                }
                accumulate(character * total(bv, bvc), v[rho * o + n], vc[rho * o + n]);
                accumulate(character * total(dw, dwc), w[rho * o + n], wc[rho * o + n]);
            }
        }
        for (U x = 0; x < ro; ++x) { v[x] = total(v[x], vc[x]); w[x] = total(w[x], wc[x]); }
        for (U rho = 0; rho < r; ++rho) {
            U first = 0;
            while (first < rho && points[0]->atom_label(first) != points[0]->atom_label(rho)) ++first;
            if (first != rho) {
                std::copy_n(weights.data() + first * o, o, weights.data() + rho * o); continue;
            }
            for (U n = 0; n < o; ++n) {
                Z q{}, qc{};
                for (U sigma = rho; sigma < r; ++sigma)
                    if (points[0]->atom_label(sigma) == points[0]->atom_label(rho))
                        accumulate(std::conj(v[sigma * o + n]) * w[sigma * o + n], q, qc);
                const Z charge = total(q, qc); const double real = charge.real();
                result.maximum_charge_imaginary_magnitude = std::max(result.maximum_charge_imaginary_magnitude, std::abs(charge.imag()));
                norms[n] = finite(norms[n] + real);
                const double square = finite(real * real);
                accumulate(Z(finite(square * square), 0), objective, objective_correction);
                weights[rho * o + n] = finite(4.0 * finite(square * real));
            }
        }
        for (U k = 0; k < kcount; ++k) {
            const auto* b = points[k]->occupied_coefficients_data(); const auto* d = points[k]->occupied_covariant_data();
            const Z character = std::conj(phase(mesh, k, cell_index)) * inverse_k;
            for (U j = 0; j < o; ++j) for (U n = 0; n < o; ++n) {
                Z value{}, correction{};
                for (U rho = 0; rho < r; ++rho) {
                    const U source = rho * p + active_columns[k * o + j], target = rho * o + n;
                    accumulate((std::conj(b[source]) * w[target] + std::conj(d[source]) * v[target]) * weights[target], value, correction);
                }
                const U target = (k * o + j) * o + n;
                accumulate(character * total(value, correction), result.gradient[target], gradient_correction[target]);
            }
        }
    }
    result.objective = total(objective, objective_correction).real();
    for (const double value : norms) result.maximum_charge_normalization_residual = std::max(result.maximum_charge_normalization_residual, std::abs(value - 1.0));
    if (result.maximum_charge_normalization_residual > options.charge_normalization_tolerance)
        throw std::invalid_argument("IAO PM finite-torus charges do not normalize to one per active orbital");
    for (U x = 0; x < koo; ++x) {
        result.gradient[x] = total(result.gradient[x], gradient_correction[x]);
        result.gradient_frobenius_norm = finite(std::hypot(result.gradient_frobenius_norm, std::abs(result.gradient[x])));
    }
    return result;
}
}  // namespace vibeqc
