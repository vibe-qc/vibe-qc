#include "vibeqc/periodic_correlation_iao_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/hermitian_jacobi.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {
namespace {
using I = std::uint64_t;
using Z = std::complex<double>;
using Options = PeriodicCorrelationIAOOptimizerOptions;
using Diagnostics = PeriodicCorrelationIAOOptimizerDiagnostics;
using Status = PeriodicCorrelationIAOOptimizerStatus;
using Event = PeriodicCorrelationIAOOptimizerEvent;
static_assert(sizeof(double) == 8 && sizeof(Z) == 16
              && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
              "IAO optimizer requires IEEE binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "IAO optimizer forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "IAO optimizer requires binary64 evaluation"
#endif
I add(I a, I b) {
    if (b > std::numeric_limits<I>::max() - a) throw std::overflow_error("IAO optimizer count sum overflow");
    return a + b;
}
I mul(I a, I b) {
    if (a && b > std::numeric_limits<I>::max() / a) throw std::overflow_error("IAO optimizer count product overflow");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("IAO optimizer arithmetic is not finite");
    return x;
}
double magnitude(Z x) { return finite(std::abs(x)); }
struct Sum {
    double re = 0, im = 0, rc = 0, ic = 0;
    static void push_lane(double x, double& sum, double& correction) {
        finite(x); const double next = finite(sum + x);
        correction = finite(correction + (std::abs(sum) >= std::abs(x)
            ? (sum - next) + x : (x - next) + sum));
        sum = next;
    }
    void push(Z x) { push_lane(x.real(), re, rc); push_lane(x.imag(), im, ic); }
    Z value() const { return {finite(re + rc), finite(im + ic)}; }
};
void check(Z a, Z b, double& maximum, const Options& o, const char* message) {
    const double error = magnitude(a - b); maximum = std::max(maximum, error);
    if (error > o.absolute_tolerance + o.relative_tolerance * std::max(magnitude(a), magnitude(b)))
        throw std::invalid_argument(message);
}
double average(double a, double b) {
    if (a == b) return a;
    const double largest = std::max(std::abs(a), std::abs(b));
    int exponent = 0; if (largest) std::frexp(largest, &exponent);
    const double x = std::scalbn(a, -exponent), y = std::scalbn(b, -exponent);
    if (std::scalbn(x, exponent) != a || std::scalbn(y, exponent) != b)
        throw std::overflow_error("IAO optimizer tangent averaging loses input range");
    return finite(std::scalbn(x + y, exponent - 1));
}
Z average(Z a, Z b) { return {average(a.real(), b.real()), average(a.imag(), b.imag())}; }
void require_options(const Options& o) {
    for (const double x : {o.initial_step, o.minimum_step, o.riemannian_gradient_tolerance})
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("IAO optimizer requires positive finite steps and gradient tolerance");
    for (const double x : {o.backtracking_factor, o.armijo_fraction, o.jacobi_relative_tolerance})
        if (!std::isfinite(x) || x <= 0 || x >= 1) throw std::invalid_argument("IAO optimizer fractions must lie in (0,1)");
    if (!o.maximum_iterations || !o.maximum_line_search_trials || !o.jacobi_max_sweeps
        || o.minimum_step > o.initial_step || !std::isfinite(o.absolute_tolerance)
        || o.absolute_tolerance < 0 || o.absolute_tolerance >= 1 || !std::isfinite(o.relative_tolerance)
        || o.relative_tolerance < 0 || o.relative_tolerance >= 1
        || (o.absolute_tolerance == 0 && o.relative_tolerance == 0))
        throw std::invalid_argument("IAO optimizer requires explicit iteration and audit controls");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0) || std::fma(tiny, one, zero) != tiny)
        throw std::invalid_argument("IAO optimizer requires round-to-nearest and gradual underflow");
}
void require_seed(const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationDiabaticSeed& seed) {
    if (!reference.state_handle() || seed.state_handle().get() != reference.state_handle().get()
        || seed.contract_version() != kPeriodicCorrelationDiabaticSeedContractVersion
        || seed.allocation_identity() != reference.dimensions().allocation_identity)
        throw std::invalid_argument("IAO optimizer seed must share the exact admitted state owner and allocation");
    const auto& s = reference.state(); const auto& m = seed.memory();
    if (m.n_cells != s.n_kpoints() || m.n_active != s.n_correlated_occupied()
        || m.n_basis != s.n_basis() || m.n_effective_orbitals != s.n_effective_orbitals())
        throw std::invalid_argument("IAO optimizer seed dimensions differ from its reference");
    (void) seed.gauges_data();
}
void unitary(const Z* a, I n, const Options& o, double& maximum) {
    for (I i = 0; i < n; ++i) for (I j = 0; j < n; ++j) {
        Sum left, right;
        for (I l = 0; l < n; ++l) {
            left.push(std::conj(a[l * n + i]) * a[l * n + j]);
            right.push(a[i * n + l] * std::conj(a[j * n + l]));
        }
        check(left.value(), double(i == j), maximum, o, "IAO optimizer unitary audit failed");
        check(right.value(), double(i == j), maximum, o, "IAO optimizer unitary audit failed");
    }
}
I partner(const RegularKMesh& mesh, I k) { return mesh.index(mesh.negate(mesh.address(k))); }
void physical_audit(const PeriodicCorrelationDiabaticSeed& seed, const RegularKMesh& mesh,
                    const Z* raw, const Options& o, Diagnostics& d) {
    const auto& state = seed.state(); const I n = state.n_correlated_occupied();
    for (I k = 0; k < state.n_kpoints(); ++k) {
        const I l = partner(mesh, k);
        unitary(raw + k * n * n, n, o, d.maximum_raw_unitarity_residual);
        for (I mu = 0; mu < state.n_basis(); ++mu) for (I i = 0; i < n; ++i) {
            Sum a, b;
            for (I j = 0; j < n; ++j) {
                a.push(state.coefficients(k)(mu, seed.active_band(k, j)) * raw[(k * n + j) * n + i]);
                b.push(state.coefficients(l)(mu, seed.active_band(l, j)) * raw[(l * n + j) * n + i]);
            }
            const Z value = a.value();
            check(value, std::conj(b.value()), d.maximum_physical_time_reversal_residual, o,
                  "IAO optimizer physical C_active U violates time reversal");
            if (k == l) d.maximum_trim_physical_imaginary_magnitude = std::max(
                d.maximum_trim_physical_imaginary_magnitude, std::abs(value.imag()));
        }
    }
}
void charge_frame_audit(const PeriodicCorrelationDiabaticSeed& seed, const RegularKMesh& mesh,
                       const PeriodicCorrelationBlochIAO* const* points, const I* active,
                       I r, I p, I n, const Options& o, Diagnostics& d) {
    for (I k = 0; k < mesh.size(); ++k) {
        const I l = partner(mesh, k);
        for (unsigned covariant = 0; covariant < 2; ++covariant) {
            const auto* a = covariant ? points[k]->occupied_covariant_data() : points[k]->occupied_coefficients_data();
            const auto* b = covariant ? points[l]->occupied_covariant_data() : points[l]->occupied_coefficients_data();
            for (I rho = 0; rho < r; ++rho) for (I i = 0; i < n; ++i) {
                Sum left, right;
                for (I j = 0; j < n; ++j) {
                    left.push(a[rho * p + active[k * n + j]] * seed.gauge(k, j, i));
                    right.push(b[rho * p + active[l * n + j]] * seed.gauge(l, j, i));
                }
                check(left.value(), std::conj(right.value()), d.maximum_charge_frame_time_reversal_residual, o,
                      "IAO optimizer seed-transformed IAO charge frames violate time reversal");
            }
        }
    }
}

// Exact antisymmetric/skew-Hermitian assignments preserve the structural
// contract needed by the Hermitian exponential, including degenerate bands.
double tangent(const RegularKMesh& mesh, I n, const Z* raw, const Z* e,
               Z* h, Z* matrix, const Options& o, Diagnostics& d) {
    for (I k = 0; k < mesh.size(); ++k) {
        for (I i = 0; i < n; ++i) for (I j = 0; j < n; ++j) {
            Sum value; for (I l = 0; l < n; ++l) value.push(std::conj(raw[(k * n + l) * n + i]) * e[(k * n + l) * n + j]);
            matrix[i * n + j] = value.value();
        }
        for (I i = 0; i < n; ++i) for (I j = i; j < n; ++j) {
            const Z value = i == j ? Z(0, matrix[i * n + i].imag())
                : average(matrix[i * n + j], -std::conj(matrix[j * n + i]));
            h[(k * n + i) * n + j] = value;
            if (i != j) h[(k * n + j) * n + i] = -std::conj(value);
        }
    }
    for (I k = 0; k < mesh.size(); ++k) {
        const I l = partner(mesh, k); if (l < k) continue;
        for (I i = 0; i < n; ++i) for (I j = i; j < n; ++j) {
            const I a = (k * n + i) * n + j, b = (l * n + i) * n + j;
            const Z value = l == k ? Z(i == j ? 0.0 : h[a].real(), 0.0) : average(h[a], std::conj(h[b]));
            h[a] = value;
            if (i != j) h[(k * n + j) * n + i] = -std::conj(value);
            if (l != k) {
                h[b] = std::conj(value);
                if (i != j) h[(l * n + j) * n + i] = -value;
            }
        }
    }
    double norm = 0; Sum directional;
    for (I k = 0; k < mesh.size(); ++k) for (I i = 0; i < n; ++i) for (I j = 0; j < n; ++j) {
        norm = finite(std::hypot(norm, magnitude(h[(k * n + i) * n + j])));
        Sum du; for (I a = 0; a < n; ++a) du.push(raw[(k * n + i) * n + a] * h[(k * n + a) * n + j]);
        const Z contribution = std::conj(e[(k * n + i) * n + j]) * du.value();
        finite(contribution.real()); finite(contribution.imag());
        directional.push(Z(contribution.real(), 0));
    }
    const double slope = finite(norm * norm);
    check(directional.value(), slope, d.maximum_tangent_slope_residual, o,
          "IAO optimizer projected tangent slope differs from its full-mesh squared norm");
    d.riemannian_gradient_norm = norm;
    if (norm > o.riemannian_gradient_tolerance && slope == 0)
        throw std::overflow_error("IAO optimizer nonzero tangent slope underflows");
    return slope;
}

void retract(const PeriodicCorrelationDiabaticSeed& seed, const RegularKMesh& mesh, I n,
             const Z* v, const Z* h, double step, Z* trial_v, Z* trial_u,
             Z* matrix, Z* eigenvectors, double* eigenvalues, Z* row,
             const Options& o, Diagnostics& d) {
    for (I k = 0; k < mesh.size(); ++k) {
        const I l = partner(mesh, k); if (l < k) continue;
        const auto* hk = h + k * n * n;
        double scale = 0;
        for (I i = 0; i < n * n; ++i) {
            matrix[i] = Z(-hk[i].imag(), hk[i].real()); // iH is Hermitian
            scale = std::max(scale, magnitude(matrix[i]));
        }
        const auto solution = hermitian_jacobi_in_place(matrix, n * n, eigenvectors, n * n,
            eigenvalues, n, n, {o.jacobi_max_sweeps, o.jacobi_relative_tolerance});
        if (solution.status != HermitianJacobiStatus::Success || solution.scaling_underflow_components)
            throw std::runtime_error("IAO optimizer exponential Jacobi failed or lost input range");
        d.jacobi_sweeps = add(d.jacobi_sweeps, solution.sweeps);
        unitary(eigenvectors, n, o, d.maximum_exponential_eigenvector_unitarity_residual);
        for (I i = 0; i < n; ++i) for (I j = 0; j < n; ++j) {
            Sum av;
            for (I a = 0; a < n; ++a) av.push(Z(-hk[i * n + a].imag(), hk[i * n + a].real()) * eigenvectors[a * n + j]);
            const double error = magnitude(av.value() - eigenvectors[i * n + j] * eigenvalues[j]);
            const double relative = scale ? finite(error / scale) : error;
            d.maximum_exponential_eigen_relative_residual = std::max(d.maximum_exponential_eigen_relative_residual, relative);
            if (relative > o.absolute_tolerance + n * o.relative_tolerance)
                throw std::invalid_argument("IAO optimizer original exponential eigensystem audit failed");
        }
        for (I i = 0; i < n; ++i) {
            for (I b = 0; b < n; ++b) {
                Sum value;
                for (I a = 0; a < n; ++a) value.push(v[(k * n + i) * n + a] * eigenvectors[a * n + b]);
                row[b] = value.value();
            }
            for (I j = 0; j < n; ++j) {
                Sum value;
                for (I b = 0; b < n; ++b) {
                    const double angle = finite(-step * eigenvalues[b]);
                    value.push(row[b] * Z(std::cos(angle), std::sin(angle)) * std::conj(eigenvectors[j * n + b]));
                }
                Z coefficient = value.value();
                if (k == l) {
                    const double error = std::abs(coefficient.imag());
                    d.maximum_trim_relative_imaginary_correction = std::max(d.maximum_trim_relative_imaginary_correction, error);
                    if (error > o.absolute_tolerance + o.relative_tolerance * magnitude(coefficient))
                        throw std::invalid_argument("IAO optimizer TRIM exponential has a nonreal relative rotation");
                    coefficient = Z(coefficient.real(), 0);
                }
                trial_v[(k * n + i) * n + j] = coefficient;
                if (k != l) trial_v[(l * n + i) * n + j] = std::conj(coefficient);
            }
        }
    }
    for (I k = 0; k < mesh.size(); ++k) {
        unitary(trial_v + k * n * n, n, o, d.maximum_relative_unitarity_residual);
        for (I i = 0; i < n; ++i) for (I j = 0; j < n; ++j) {
            Sum value; for (I a = 0; a < n; ++a) value.push(seed.gauge(k, i, a) * trial_v[(k * n + a) * n + j]);
            trial_u[(k * n + i) * n + j] = value.value();
        }
    }
    physical_audit(seed, mesh, trial_u, o, d);
}

class Digest {
public:
    explicit Digest(const char* domain) { string(domain); u32(kPeriodicCorrelationIAOOptimizerContractVersion); }
    void u32(std::uint32_t x) { std::array<std::uint8_t, 4> b{}; for (unsigned i = 0; i < 4; ++i) b[i] = x >> (24 - 8 * i); h_.update(b.data(), b.size()); }
    void u64(I x) { std::array<std::uint8_t, 8> b{}; for (unsigned i = 0; i < 8; ++i) b[i] = x >> (56 - 8 * i); h_.update(b.data(), b.size()); }
    void real(double x) { finite(x); if (x == 0) x = 0; I bits; std::memcpy(&bits, &x, 8); u64(bits); }
    void value(Z x) { real(x.real()); real(x.imag()); }
    void string(const std::string& s) { u64(s.size()); h_.update(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
    std::string finish() { return h_.finish_hex(); }
private: detail::Sha256 h_;
};
}  // namespace

PeriodicCorrelationIAOOptimizerMemoryPlan plan_periodic_correlation_iao_optimizer(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationDiabaticSeed& seed,
    I r, const Options& o) {
    require_seed(reference, seed); require_options(o);
    const auto pm = plan_periodic_correlation_iao_pm(reference, r);
    const I k = pm.n_points, n = pm.n_active, b = reference.state().n_basis();
    const I nn = mul(n, n), kn = mul(k, n), knn = mul(k, nn), g = mul(16, knn);
    PeriodicCorrelationIAOOptimizerMemoryPlan m;
    m.n_points = k; m.n_basis = b; m.n_active = n; m.n_minimal = r;
    m.output_numerical_bytes = add(g, mul(8, kn));
    m.fixed_optimizer_bytes = add(mul(5, g), mul(8, kn));
    m.exponential_workspace_bytes = add(mul(32, nn), mul(24, n));
    m.pm_owned_peak_bytes = pm.peak_owned_numerical_bytes;
    m.peak_owned_numerical_bytes = add(m.fixed_optimizer_bytes, add(m.exponential_workspace_bytes, m.pm_owned_peak_bytes));
    m.borrowed_seed_numerical_bytes = add(g, mul(8, add(kn, n)));
    if (m.borrowed_seed_numerical_bytes != seed.memory().output_numerical_bytes)
        throw std::invalid_argument("IAO optimizer seed numerical inventory mismatch");
    m.live_iao_numerical_bytes = pm.live_iao_numerical_bytes;
    m.borrowed_owner_pointer_bytes = pm.borrowed_owner_pointer_bytes;
    const I live = add(m.peak_owned_numerical_bytes,
        add(m.borrowed_seed_numerical_bytes, add(m.live_iao_numerical_bytes, m.borrowed_owner_pointer_bytes)));
    const auto& d = reference.dimensions(); const auto& budget = reference.budget();
    m.required_node_memory_bytes = add(add(d.external_bytes, d.shared_bytes),
        add(mul(budget.mpi_ranks, add(d.per_rank_bytes, d.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), live)));
    const I cubic = mul(nn, n);
    const I preflight_terms = add(add(mul(r, nn), mul(b, nn)), add(mul(pm.n_occupied, n), add(r, nn)));
    m.preflight_work_units = mul(512, add(mul(k, preflight_terms), 1));
    m.projection_work_units = mul(512, mul(k, add(add(cubic, nn), 1)));
    m.trial_retraction_work_units = mul(k, add(mul(1024, mul(o.jacobi_max_sweeps, cubic)),
        mul(512, add(add(cubic, mul(b, nn)), nn))));
    m.pm_work_units = pm.work_units;
    m.initial_work_units = add(m.preflight_work_units, add(m.pm_work_units, m.projection_work_units));
    m.work_units_per_trial = add(m.trial_retraction_work_units, add(m.pm_work_units, m.projection_work_units));
    m.maximum_work_units = add(m.initial_work_units, mul(mul(o.maximum_iterations, o.maximum_line_search_trials), m.work_units_per_trial));
    const I limit = static_cast<I>(std::numeric_limits<std::ptrdiff_t>::max());
    const I hash_limit = std::numeric_limits<I>::max() / 8U - 4096U;
    if (m.peak_owned_numerical_bytes > limit || m.borrowed_owner_pointer_bytes > limit
        || m.output_numerical_bytes > hash_limit || mul(72, k) > hash_limit
        || knn > std::vector<Z>().max_size()) throw std::length_error("IAO optimizer arrays exceed native address/hash range");
    return m;
}

void PeriodicCorrelationIAOOptimizerResult::require_live() const {
    if (!state_ || gauges_.size() != memory_.n_points * memory_.n_active * memory_.n_active
        || active_indices_.size() != memory_.n_points * memory_.n_active)
        throw std::logic_error("IAO optimizer result owner is consumed");
}
const PeriodicRestrictedMeanFieldState& PeriodicCorrelationIAOOptimizerResult::state() const { require_live(); return *state_; }
const Z* PeriodicCorrelationIAOOptimizerResult::gauges_data() const { require_live(); return gauges_.data(); }
Z PeriodicCorrelationIAOOptimizerResult::gauge(std::size_t k, std::size_t i, std::size_t j) const {
    require_live(); if (k >= memory_.n_points || i >= memory_.n_active || j >= memory_.n_active)
        throw std::out_of_range("IAO optimizer gauge index out of range");
    return gauges_[(k * memory_.n_active + i) * memory_.n_active + j];
}
I PeriodicCorrelationIAOOptimizerResult::active_band(std::size_t k, std::size_t i) const {
    require_live(); if (k >= memory_.n_points || i >= memory_.n_active)
        throw std::out_of_range("IAO optimizer active index out of range");
    return active_indices_[k * memory_.n_active + i];
}

PeriodicCorrelationIAOOptimizerResult optimize_periodic_correlation_iao_pm(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationDiabaticSeed& seed,
    const PeriodicCorrelationBlochIAO* const* points, std::size_t point_count,
    const PeriodicCorrelationIAOPMOptions& pm_options, const Options& o,
    const PeriodicCorrelationIAOOptimizerCaps& caps,
    PeriodicCorrelationIAOOptimizerCallback callback, void* context) {
    require_seed(reference, seed);
    const I kcount = reference.state().n_kpoints(), n = reference.state().n_correlated_occupied();
    const I pointer_bytes = mul(sizeof(*points), kcount);
    const auto pointer_address = reinterpret_cast<std::uintptr_t>(points);
    if (!points || point_count != kcount || pointer_address % alignof(const PeriodicCorrelationBlochIAO*)
        || pointer_bytes > static_cast<I>(std::numeric_limits<std::ptrdiff_t>::max())
        || pointer_bytes > std::numeric_limits<std::uintptr_t>::max() - pointer_address)
        throw std::invalid_argument("IAO optimizer owner pointer view or extent is invalid");
    if (!points[0]) throw std::invalid_argument("IAO optimizer requires live one-k IAO owners");
    const I r = points[0]->memory().n_minimal;
    const auto memory = plan_periodic_correlation_iao_optimizer(reference, seed, r, o);
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < memory.peak_owned_numerical_bytes
        || caps.maximum_work_units < memory.initial_work_units)
        throw std::length_error("IAO optimizer byte or initial work cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || memory.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("IAO optimizer live numerical owners exceed admitted node memory");
    for (const double x : {pm_options.gauge_unitarity_tolerance, pm_options.charge_normalization_tolerance})
        if (!std::isfinite(x) || x <= 0 || x >= 1) throw std::invalid_argument("IAO optimizer PM controls must lie in (0,1)");
    const auto pm_plan = plan_periodic_correlation_iao_pm(reference, r);
    const I p = pm_plan.n_occupied, knn = kcount * n * n;
    for (I k = 0; k < kcount; ++k) {
        if (!points[k] || points[k]->state_handle().get() != reference.state_handle().get()
            || points[k]->allocation_identity() != reference.dimensions().allocation_identity
            || points[k]->point() != k || points[k]->memory().n_minimal != r || points[k]->memory().n_occupied != p
            || points[k]->memory().output_numerical_bytes != memory.live_iao_numerical_bytes / kcount
            || points[k]->declared_provenance().minimal_basis_identity != points[0]->declared_provenance().minimal_basis_identity)
            throw std::invalid_argument("IAO optimizer IAO owners or declared minimal bases differ");
        (void) points[k]->occupied_coefficients_data();
        for (I rho = 0; rho < r; ++rho) if (points[k]->atom_label(rho) != points[0]->atom_label(rho))
            throw std::invalid_argument("IAO optimizer IAO atom labels differ across points");
    }
    PeriodicCorrelationIAOOptimizerResult result; result.memory_ = memory;
    auto& d = result.diagnostics_; d.charged_work_units = memory.initial_work_units;
    result.gauges_.resize(knn); result.active_indices_.resize(kcount * n);
    std::vector<Z> v(knn), trial_v(knn), trial_u(knn), h(knn);
    std::vector<Z> matrix(n * n), eigenvectors(n * n), row(n);
    std::vector<double> eigenvalues(n);
    for (I k = 0; k < kcount; ++k) for (I j = 0; j < n; ++j) {
        const I band = seed.active_band(k, j);
        if (band >= reference.state().n_effective_orbitals() || !reference.state().correlated_occupied_mask(k)[band])
            throw std::invalid_argument("IAO optimizer seed has an invalid active-band map");
        I column = 0; while (column < p && points[k]->occupied_band(column) != band) ++column;
        if (column == p) throw std::invalid_argument("IAO optimizer cannot map a seed active band into the all-occupied IAO frame");
        result.active_indices_[k * n + j] = column; // later reused for retained full-band output
        for (I i = 0; i < n; ++i) {
            result.gauges_[(k * n + j) * n + i] = seed.gauge(k, j, i);
            v[(k * n + j) * n + i] = double(j == i);
        }
    }
    const RegularKMesh mesh(reference.state().mesh());
    physical_audit(seed, mesh, result.gauges_.data(), o, d);
    charge_frame_audit(seed, mesh, points, result.active_indices_.data(), r, p, n, o, d);
    const PeriodicCorrelationIAOPMCaps pm_caps{memory.pm_owned_peak_bytes, memory.pm_work_units};
    double slope = 0;
    {
        auto evaluation = evaluate_periodic_correlation_iao_pm(reference, points, point_count,
            result.gauges_.data(), knn, pm_options, pm_caps);
        d.objective_evaluations = 1; d.initial_objective = d.final_objective = evaluation.objective;
        slope = tangent(mesh, n, result.gauges_.data(), evaluation.gradient.data(), h.data(), matrix.data(), o, d);
    } // no previous PM gradient can coexist with the next charged full P arena
    const auto emit = [&](Event event, double step, double trial_objective) {
        if (callback) callback({event, result.status_, d.accepted_steps, d.objective_evaluations,
            d.line_search_trials, d.charged_work_units, d.final_objective, d.riemannian_gradient_norm,
            step, trial_objective}, context);
    };
    if (d.riemannian_gradient_norm <= o.riemannian_gradient_tolerance) result.status_ = Status::Converged;
    emit(Event::Initial, 0, 0);
    while (result.status_ != Status::Converged && d.accepted_steps < o.maximum_iterations) {
        bool accepted = false, work_limit = false;
        double step = o.initial_step;
        for (I trial = 0; trial < o.maximum_line_search_trials && step >= o.minimum_step; ++trial) {
            if (memory.work_units_per_trial > caps.maximum_work_units - d.charged_work_units) { work_limit = true; break; }
            d.charged_work_units += memory.work_units_per_trial;
            ++d.line_search_trials;
            retract(seed, mesh, n, v.data(), h.data(), step, trial_v.data(), trial_u.data(),
                matrix.data(), eigenvectors.data(), eigenvalues.data(), row.data(), o, d);
            auto evaluation = evaluate_periodic_correlation_iao_pm(reference, points, point_count,
                trial_u.data(), knn, pm_options, pm_caps);
            ++d.objective_evaluations;
            const double model_gain = finite(step * slope);
            const double gain = finite(o.armijo_fraction * model_gain);
            const double actual_gain = finite(evaluation.objective - d.final_objective);
            // Compare the increment, not f_trial >= round(f + c1*t*g^2):
            // the small Armijo fraction can make that bound round back to f
            // while a genuine trial improves by thousands of ULPs. Nearby
            // positive objective subtraction is exact by Sterbenz. Still
            // reject a model displacement that itself is unrepresentable;
            // numerical reconstruction noise is not a tiny-step success.
            const bool model_representable = finite(d.final_objective + model_gain) > d.final_objective;
            if (gain > 0 && model_representable && actual_gain > 0 && actual_gain >= gain) {
                v.swap(trial_v); result.gauges_.swap(trial_u);
                d.final_objective = evaluation.objective; d.last_accepted_step = step; ++d.accepted_steps;
                slope = tangent(mesh, n, result.gauges_.data(), evaluation.gradient.data(), h.data(), matrix.data(), o, d);
                if (d.riemannian_gradient_norm <= o.riemannian_gradient_tolerance) result.status_ = Status::Converged;
                emit(Event::Accepted, step, evaluation.objective); accepted = true; break;
            }
            ++d.rejected_trials; emit(Event::Rejected, step, evaluation.objective);
            step = finite(step * o.backtracking_factor);
        }
        if (work_limit) { result.status_ = Status::WorkLimit; break; }
        if (!accepted) { result.status_ = Status::LineSearchFailed; break; }
    }
    for (I k = 0; k < kcount; ++k) for (I j = 0; j < n; ++j) result.active_indices_[k * n + j] = seed.active_band(k, j);
    result.state_ = reference.state_handle(); result.allocation_identity_ = reference.dimensions().allocation_identity;
    result.seed_identity_ = seed.seed_identity_sha256();
    Digest source("vibeqc.periodic.correlation.iao-optimizer.source");
    source.string(reference.state().state_identity_sha256()); source.string(result.seed_identity_); source.u64(kcount);
    for (I k = 0; k < kcount; ++k) source.string(points[k]->iao_identity_sha256());
    result.source_identity_ = source.finish();
    Digest output("vibeqc.periodic.correlation.iao-optimizer.gauges");
    output.u64(kcount); output.u64(n);
    for (const auto x : result.gauges_) output.value(x);
    for (const auto band : result.active_indices_) output.u64(band);
    result.gauge_digest_ = output.finish();
    Digest identity("vibeqc.periodic.correlation.iao-optimizer.identity");
    identity.string(result.source_identity_); identity.string(result.allocation_identity_); identity.string(result.gauge_digest_);
    identity.string("full-mesh-TR-tangent;right-exponential;Armijo-steepest-ascent;real-Mulliken-p4;no-overlap-source-certification");
    identity.u32(static_cast<std::uint32_t>(result.status_));
    for (const double x : {o.initial_step, o.minimum_step, o.backtracking_factor, o.armijo_fraction,
            o.riemannian_gradient_tolerance, o.absolute_tolerance, o.relative_tolerance, o.jacobi_relative_tolerance,
            pm_options.gauge_unitarity_tolerance, pm_options.charge_normalization_tolerance,
            d.final_objective, d.riemannian_gradient_norm}) identity.real(x);
    for (const I x : {o.maximum_iterations, o.maximum_line_search_trials, o.jacobi_max_sweeps,
            caps.maximum_owned_numerical_bytes, caps.maximum_work_units, d.accepted_steps, d.objective_evaluations}) identity.u64(x);
    result.identity_digest_ = identity.finish();
    emit(Event::Finished, d.last_accepted_step, 0);
    return result;
}
}  // namespace vibeqc
