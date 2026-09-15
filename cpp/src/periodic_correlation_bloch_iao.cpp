#include "vibeqc/periodic_correlation_bloch_iao.hpp"

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

namespace vibeqc {
namespace {
using Z = std::complex<double>;
using Options = PeriodicCorrelationBlochIAOOptions;
using Diagnostics = PeriodicCorrelationBlochIAODiagnostics;
using State = PeriodicRestrictedMeanFieldState;
static_assert(sizeof(double) == 8U && sizeof(Z) == 16U
              && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
              "Bloch IAO requires IEEE binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Bloch IAO forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Bloch IAO requires binary64 evaluation"
#endif
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("Bloch IAO count/byte/work product overflow");
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("Bloch IAO count/byte/work sum overflow");
    return a + b;
}
bool finite(Z value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); }
double norm(Z value) {
    const auto a = std::abs(value);
    if (!std::isfinite(a)) throw std::overflow_error("Bloch IAO non-finite arithmetic");
    return a;
}
struct Sum {
    double re = 0.0, im = 0.0, rc = 0.0, ic = 0.0;
    static void lane(double t, double& x, double& c) {
        const double next = x + t;
        c += std::abs(x) >= std::abs(t) ? (x - next) + t : (t - next) + x;
        x = next;
        if (!std::isfinite(x) || !std::isfinite(c)) throw std::overflow_error("Bloch IAO compensated sum overflow");
    }
    void push(Z t) {
        if (!finite(t)) throw std::overflow_error("Bloch IAO scalar product overflow");
        lane(t.real(), re, rc); lane(t.imag(), im, ic);
    }
    Z value() const {
        const Z result(re + rc, im + ic);
        if (!finite(result)) throw std::overflow_error("Bloch IAO contraction overflow");
        return result;
    }
};
void check(Z a, Z b, double& maximum, const Options& o, const char* message) {
    const double error = norm(a - b); maximum = std::max(maximum, error);
    if (error > o.validation_absolute_tolerance + o.validation_relative_tolerance * std::max(norm(a), norm(b))) {
        throw std::invalid_argument(message);
    }
}
void charge(std::uint64_t amount, const Options& o, Diagnostics& d) {
    const auto next = add(d.charged_work_units, amount);
    if (next > o.maximum_work_units) throw std::length_error("Bloch IAO scalar-loop work budget exceeded");
    d.charged_work_units = next;
}
template<class T> void storage(const T* ptr, std::size_t extent, std::uint64_t required) {
    const auto bytes = mul(sizeof(T), required);
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    if (!ptr || extent < required || address % alignof(T)
        || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::invalid_argument("Bloch IAO input pointer, alignment or accessible extent is invalid");
    }
}
bool digest_id(const std::string& s) {
    return s.size() == 64U && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
void validate_options(const Options& o) {
    const std::array<std::array<double, 2>, 5> pairs{{
        {o.minimal_rank_absolute_floor, o.minimal_rank_relative_floor},
        {o.depolarized_rank_absolute_floor, o.depolarized_rank_relative_floor},
        {o.iao_rank_absolute_floor, o.iao_rank_relative_floor},
        {o.schur_negative_absolute_tolerance, o.schur_negative_relative_tolerance},
        {o.validation_absolute_tolerance, o.validation_relative_tolerance}}};
    for (const auto& pair : pairs) {
        if (!std::isfinite(pair[0]) || pair[0] < 0.0 || !std::isfinite(pair[1])
            || pair[1] < 0.0 || pair[1] >= 1.0 || (pair[0] == 0.0 && pair[1] == 0.0)) {
            throw std::invalid_argument("Bloch IAO absolute/relative controls must be finite, nonnegative and nonzero as a pair");
        }
    }
    if (o.validation_absolute_tolerance >= 1.0 || !std::isfinite(o.jacobi_relative_tolerance)
        || o.jacobi_relative_tolerance <= 0.0 || o.jacobi_relative_tolerance >= 1.0
        || o.jacobi_max_sweeps == 0U || o.maximum_work_units == 0U) {
        throw std::invalid_argument("Bloch IAO requires explicit positive solver/work controls");
    }
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0.0) || std::fma(tiny, one, zero) != tiny) {
        throw std::invalid_argument("Bloch IAO requires round-to-nearest and gradual underflow");
    }
}
void validate_reference(const PeriodicCorrelationAdmittedReference& r, std::size_t point) {
    if (!r.state_handle()) throw std::invalid_argument("Bloch IAO requires a live admitted state");
    const auto& s = r.state(); const auto& d = r.dimensions(); const auto& p = r.plan();
    if (point >= s.n_kpoints()) throw std::out_of_range("Bloch IAO k-point index out of range");
    if (r.contract_version() != kPeriodicCorrelationAdmittedReferenceContractVersion
        || s.contract_version() != kPeriodicRestrictedMeanFieldStateContractVersion
        || s.normalization() != PeriodicMeanFieldNormalizationConvention::UnnormalizedAoBlochSumsUniformFullBzWeights
        || s.reference_kind() != PeriodicMeanFieldReferenceKind::RestrictedHartreeFock
        || p.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || p.admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || d.symmetry_reduction_requested || !d.symmetry_mapping_identity.empty()
        || d.symmetry_representative_count != d.n_kpoints || d.symmetry_weight_sum != d.n_kpoints
        || p.allocation_contract_version != d.allocation_contract_version
        || p.calculation_identity != d.calculation_identity || p.allocation_identity != d.allocation_identity
        || s.calculation_identity() != d.calculation_identity || s.mesh() != d.mesh || s.is_shift() != d.is_shift
        || s.n_basis() != d.n_basis || s.n_kpoints() != d.n_kpoints
        || s.n_effective_orbitals() != d.n_effective_orbitals
        || s.n_correlated_occupied() != d.n_home_occupied
        || s.n_frozen_core() + s.n_correlated_occupied() != d.n_home_total_occupied) {
        throw std::invalid_argument("Bloch IAO admitted-reference seals mismatch");
    }
}
// Preserve equal subnormals; reject loss when scaling unequal values rather
// than silently rounding away a component before Hermitian averaging.
double half_sum(double a, double b) {
    if (a == b) return a;
    const double largest = std::max(std::abs(a), std::abs(b));
    int exponent = 0; if (largest) std::frexp(largest, &exponent);
    const double x = std::scalbn(a, -exponent), y = std::scalbn(b, -exponent);
    if (std::scalbn(x, exponent) != a || std::scalbn(y, exponent) != b) {
        throw std::overflow_error("Bloch IAO Hermitian averaging loses input range");
    }
    const double answer = std::scalbn(x + y, exponent - 1);
    if (!std::isfinite(answer)) throw std::overflow_error("Bloch IAO Hermitian averaging overflow");
    return answer;
}
void hermitize(Z* a, std::size_t n, const Options& o, Diagnostics& d) {
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) {
        const Z x = a[i * n + j], y = std::conj(a[j * n + i]);
        check(x, y, d.maximum_derived_hermitian_defect, o, "Bloch IAO derived metric failed raw Hermitian audit");
        const Z v(half_sum(x.real(), y.real()), i == j ? 0.0 : half_sum(x.imag(), y.imag()));
        d.maximum_derived_hermitization_correction = std::max(d.maximum_derived_hermitization_correction,
            std::max(norm(v - x), norm(v - y)));
        a[i * n + j] = v; a[j * n + i] = std::conj(v);
    }
}
void solve(const Z* original, std::size_t n, Z* m, Z* u, double* eigen,
           const Options& o, Diagnostics& d) {
    double scale = 0.0;
    for (std::size_t t = 0; t < n * n; ++t) { m[t] = original[t]; scale = std::max(scale, norm(original[t])); }
    const auto answer = hermitian_jacobi_in_place(m, n * n, u, n * n, eigen, n, n,
        {o.jacobi_max_sweeps, o.jacobi_relative_tolerance});
    if (answer.status != HermitianJacobiStatus::Success || answer.scaling_underflow_components != 0U) {
        throw std::runtime_error("Bloch IAO Jacobi failed or lost input range");
    }
    d.jacobi_sweeps = add(d.jacobi_sweeps, answer.sweeps);
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        Sum av, gram;
        for (std::size_t t = 0; t < n; ++t) {
            av.push(original[i * n + t] * u[t * n + j]);
            gram.push(std::conj(u[t * n + i]) * u[t * n + j]);
        }
        const double error = norm(av.value() - eigen[j] * u[i * n + j]);
        const double relative = scale ? error / scale : error;
        d.maximum_eigensystem_relative_residual = std::max(d.maximum_eigensystem_relative_residual, relative);
        if (!std::isfinite(relative) || relative > o.validation_absolute_tolerance + n * o.validation_relative_tolerance
            || norm(gram.value() - double(i == j)) > o.validation_absolute_tolerance + o.validation_relative_tolerance) {
            throw std::invalid_argument("Bloch IAO original eigensystem/unitarity audit failed");
        }
    }
}
void full_rank(const double* eigen, std::size_t n, double absolute, double relative,
               double& minimum, double& boundary, const char* message) {
    minimum = eigen[0]; boundary = std::max(absolute, relative * std::max(0.0, eigen[n - 1]));
    if (minimum <= boundary) throw std::invalid_argument(message);
}
void inverse(const Z* u, const double* eigen, std::size_t n, Z* out) {
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        Sum value;
        for (std::size_t t = 0; t < n; ++t) value.push((u[i * n + t] / eigen[t]) * std::conj(u[j * n + t]));
        out[i * n + j] = value.value();
    }
}
class Digest {
public:
    explicit Digest(const char* domain) { text(domain); u32(kPeriodicCorrelationBlochIAOContractVersion); }
    void u32(std::uint32_t x) { std::array<std::uint8_t, 4> b{}; for (unsigned i = 0; i < 4; ++i) b[i] = x >> (24U - 8U * i); h_.update(b.data(), b.size()); }
    void u64(std::uint64_t x) { std::array<std::uint8_t, 8> b{}; for (unsigned i = 0; i < 8; ++i) b[i] = x >> (56U - 8U * i); h_.update(b.data(), b.size()); }
    void real(double x) { if (!std::isfinite(x)) throw std::overflow_error("Bloch IAO digest is non-finite"); if (x == 0.0) x = 0.0; std::uint64_t bits; std::memcpy(&bits, &x, 8); u64(bits); }
    void value(Z x) { real(x.real()); real(x.imag()); }
    void text(const std::string& s) { u64(s.size()); h_.update(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
    std::string finish() { return h_.finish_hex(); }
private: detail::Sha256 h_;
};
}  // namespace

PeriodicCorrelationBlochIAOMemoryPlan plan_periodic_correlation_bloch_iao(
    std::uint64_t b, std::uint64_t e, std::uint64_t p, std::uint64_t r, std::uint64_t sweeps) {
    if (p == 0U || p > r || r > e || e > b || sweeps == 0U) {
        throw std::invalid_argument("Bloch IAO requires 0 < all-occupied <= minimal <= retained <= AO and positive sweeps");
    }
    PeriodicCorrelationBlochIAOMemoryPlan m;
    m.n_basis = b; m.n_effective = e; m.n_occupied = p; m.n_minimal = r;
    const auto er = mul(e, r), ep = mul(e, p), rr = mul(r, r), rp = mul(r, p), br = mul(b, r);
    m.borrowed_input_bytes = add(mul(16U, add(br, rr)), mul(8U, r));
    m.output_numerical_bytes = add(mul(16U, add(add(er, rr), mul(2U, rp))), mul(8U, add(r, p)));
    m.projection_workspace_bytes = add(mul(16U, er), mul(32U, ep));
    m.matrix_workspace_bytes = mul(48U, rr);
    m.scalar_workspace_bytes = add(mul(8U, r), mul(32U, b));
    m.peak_owned_numerical_bytes = add(m.output_numerical_bytes,
        add(m.projection_workspace_bytes, add(m.matrix_workspace_bytes, m.scalar_workspace_bytes)));
    // Deliberately conservative scalar-loop bounds (including contraction
    // audits and four Jacobi solves), not hardware FLOP or timing estimates.
    const auto cubic = mul(rr, r), jacobi = mul(1024U, mul(sweeps, cubic));
    m.input_preflight_work_units = mul(128U, add(add(br, rr), add(mul(mul(b, b), e), mul(b, mul(e, e)))));
    m.projection_work_units = add(mul(2U, jacobi), mul(256U, add(mul(b, er), add(mul(e, rr), cubic))));
    m.construction_work_units = add(mul(2U, jacobi), mul(512U, add(mul(e, rr), add(mul(e, mul(p, p)), add(cubic, mul(rr, p))))));
    const auto ao_validation = add(mul(b, mul(e, rr)), mul(mul(b, b), r));
    const auto retained_validation = add(mul(mul(e, e), p), mul(e, rp));
    const auto metric_validation = add(mul(rr, p), mul(r, mul(p, p)));
    m.validation_work_units = mul(256U, add(ao_validation, add(retained_validation, metric_validation)));
    m.maximum_work_units = add(add(m.input_preflight_work_units, m.projection_work_units),
        add(m.construction_work_units, m.validation_work_units));
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto hash_limit = std::numeric_limits<std::uint64_t>::max() / 8U - 4096U;
    if (m.peak_owned_numerical_bytes > limit || m.borrowed_input_bytes > limit
        || m.output_numerical_bytes > hash_limit || m.borrowed_input_bytes > hash_limit
        || er > std::vector<Z>().max_size() || rr > std::vector<Z>().max_size()) {
        throw std::length_error("Bloch IAO numerical arrays exceed native vector/address/hash extents");
    }
    return m;
}

void PeriodicCorrelationBlochIAO::require_live() const {
    if (!state_ || a_.size() != memory_.n_effective * memory_.n_minimal
        || g_.size() != memory_.n_minimal * memory_.n_minimal
        || b_.size() != memory_.n_minimal * memory_.n_occupied || d_.size() != b_.size()
        || atom_labels_.size() != memory_.n_minimal || occupied_indices_.size() != memory_.n_occupied) {
        throw std::logic_error("Bloch IAO owner is consumed");
    }
}
const State& PeriodicCorrelationBlochIAO::state() const { require_live(); return *state_; }
const Z* PeriodicCorrelationBlochIAO::retained_coefficients_data() const { require_live(); return a_.data(); }
const Z* PeriodicCorrelationBlochIAO::metric_data() const { require_live(); return g_.data(); }
const Z* PeriodicCorrelationBlochIAO::occupied_coefficients_data() const { require_live(); return b_.data(); }
const Z* PeriodicCorrelationBlochIAO::occupied_covariant_data() const { require_live(); return d_.data(); }
std::uint64_t PeriodicCorrelationBlochIAO::atom_label(std::size_t i) const {
    require_live(); if (i >= memory_.n_minimal) throw std::out_of_range("Bloch IAO atom label index out of range"); return atom_labels_[i];
}
std::uint64_t PeriodicCorrelationBlochIAO::occupied_band(std::size_t i) const {
    require_live(); if (i >= memory_.n_occupied) throw std::out_of_range("Bloch IAO occupied index out of range"); return occupied_indices_[i];
}
Z PeriodicCorrelationBlochIAO::ao_coefficient(std::size_t mu, std::size_t rho) const {
    require_live();
    if (mu >= memory_.n_basis || rho >= memory_.n_minimal) throw std::out_of_range("Bloch IAO coefficient index out of range");
    Sum value;
    for (std::size_t j = 0; j < memory_.n_effective; ++j) value.push(state_->coefficients(point_)(mu, j) * a_[j * memory_.n_minimal + rho]);
    return value.value();
}

PeriodicCorrelationBlochIAO make_periodic_correlation_bloch_iao(
    const PeriodicCorrelationAdmittedReference& reference, std::size_t point,
    const Z* cross, std::size_t cross_extent, const Z* minimal, std::size_t minimal_extent,
    const std::uint64_t* labels, std::size_t label_extent, std::size_t r,
    const PeriodicCorrelationBlochIAOInputProvenance& provenance, std::uint64_t cap, const Options& o) {
    validate_reference(reference, point); validate_options(o);
    const auto& state = reference.state();
    const auto b = static_cast<std::size_t>(state.n_basis()), e = static_cast<std::size_t>(state.n_effective_orbitals());
    const auto p = static_cast<std::size_t>(state.n_frozen_core() + state.n_correlated_occupied());
    const auto memory = plan_periodic_correlation_bloch_iao(b, e, p, r, o.jacobi_max_sweeps);
    if (!cap || cap < memory.peak_owned_numerical_bytes) throw std::length_error("Bloch IAO numerical byte cap is missing or exceeded");
    if (!digest_id(provenance.cross_overlap_identity) || !digest_id(provenance.minimal_basis_identity)
        || provenance.caller_live_numerical_bytes < memory.borrowed_input_bytes) {
        throw std::invalid_argument("Bloch IAO caller-declared provenance or live input inventory is invalid");
    }
    const auto& dims = reference.dimensions(); const auto& budget = reference.budget();
    const auto required = add(add(dims.external_bytes, dims.shared_bytes),
        add(mul(budget.mpi_ranks, add(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), add(memory.peak_owned_numerical_bytes, provenance.caller_live_numerical_bytes))));
    if (!budget.memory_limit_bytes || required > budget.memory_limit_bytes) throw std::length_error("Bloch IAO reference/input/owned payload exceeds admitted node memory");
    storage(cross, cross_extent, mul(b, r)); storage(minimal, minimal_extent, mul(r, r)); storage(labels, label_extent, r);
    Diagnostics d; d.required_node_memory_bytes = required;
    charge(memory.input_preflight_work_units, o, d);
    for (std::size_t i = 0; i < b * r; ++i) if (!finite(cross[i])) throw std::invalid_argument("Bloch IAO S12 must be finite");
    for (std::size_t i = 0; i < r; ++i) for (std::size_t j = 0; j < r; ++j) {
        if (!finite(minimal[i * r + j]) || minimal[i * r + j] != std::conj(minimal[j * r + i])) {
            throw std::invalid_argument("Bloch IAO S22 must be finite and exactly Hermitian");
        }
    }
    const auto& active = state.correlated_occupied_mask(point); const auto& core = state.frozen_core_mask(point);
    const auto& virt = state.virtual_mask(point); const auto& y = state.coefficients(point);
    if (active.size() != e || core.size() != e || virt.size() != e) throw std::invalid_argument("Bloch IAO state mask shape mismatch");
    std::size_t occupied_count = 0;
    for (std::size_t j = 0; j < e; ++j) {
        if (active[j] > 1U || core[j] > 1U || virt[j] > 1U || active[j] + core[j] + virt[j] != 1U) {
            throw std::invalid_argument("Bloch IAO masks must be an exhaustive disjoint partition");
        }
        occupied_count += active[j] + core[j];
    }
    if (occupied_count != p) throw std::invalid_argument("Bloch IAO all-occupied count mismatch");
    // Fixed, simultaneously live numerical arena. Inputs are never copied
    // wholesale; only S22's consumed eigensolver work is copied into M.
    PeriodicCorrelationBlochIAO result;
    result.a_.resize(e * r); result.g_.resize(r * r); result.b_.resize(r * p); result.d_.resize(r * p);
    result.atom_labels_.resize(r); result.occupied_indices_.resize(p);
    std::vector<Z> t(e * r), z(e * p), q(e * p), inv(r * r), m(r * r), u(r * r), ao(2 * b);
    std::vector<double> eigen(r);
    for (std::size_t i = 0; i < r; ++i) result.atom_labels_[i] = labels[i];
    std::size_t occupied = 0;
    for (std::size_t j = 0; j < e; ++j) if (active[j] || core[j]) result.occupied_indices_[occupied++] = j;
    for (std::size_t j = 0; j < e; ++j) {
        for (std::size_t mu = 0; mu < b; ++mu) {
            Sum value; for (std::size_t nu = 0; nu < b; ++nu) value.push(state.overlap(point)(mu, nu) * y(nu, j));
            ao[mu] = value.value();
        }
        for (std::size_t i = 0; i < e; ++i) {
            Sum value; for (std::size_t mu = 0; mu < b; ++mu) value.push(std::conj(y(mu, i)) * ao[mu]);
            check(value.value(), double(i == j), d.maximum_retained_metric_residual, o, "Bloch IAO retained frame is not metric orthonormal");
        }
    }
    charge(memory.projection_work_units, o, d);
    for (std::size_t i = 0; i < e; ++i) for (std::size_t j = 0; j < r; ++j) {
        Sum value; for (std::size_t mu = 0; mu < b; ++mu) value.push(std::conj(y(mu, i)) * cross[mu * r + j]);
        t[i * r + j] = value.value();
    }
    for (std::size_t i = 0; i < r; ++i) for (std::size_t j = 0; j < r; ++j) {
        Sum value; value.push(minimal[i * r + j]);
        for (std::size_t a = 0; a < e; ++a) value.push(-std::conj(t[a * r + i]) * t[a * r + j]);
        inv[i * r + j] = value.value();
    }
    hermitize(inv.data(), r, o, d); solve(inv.data(), r, m.data(), u.data(), eigen.data(), o, d);
    d.minimum_schur_eigenvalue = eigen[0];
    d.schur_negative_boundary = o.schur_negative_absolute_tolerance
        + o.schur_negative_relative_tolerance * std::max(std::abs(eigen[0]), std::abs(eigen[r - 1]));
    if (!std::isfinite(d.schur_negative_boundary)) throw std::overflow_error("Bloch IAO Schur-complement tolerance overflow");
    if (eigen[0] < -d.schur_negative_boundary) throw std::invalid_argument("Bloch IAO cross/minimal overlaps fail Schur-complement PSD consistency");
    solve(minimal, r, m.data(), u.data(), eigen.data(), o, d);
    full_rank(eigen.data(), r, o.minimal_rank_absolute_floor, o.minimal_rank_relative_floor,
        d.minimum_minimal_eigenvalue, d.minimal_rank_boundary, "Bloch IAO minimal overlap lost full atomic rank");
    inverse(u.data(), eigen.data(), r, inv.data());
    charge(memory.construction_work_units, o, d);
    // The final B output slot initially holds K=S22^-1 T_occ^H [r,p].
    for (std::size_t a = 0; a < r; ++a) for (std::size_t i = 0; i < p; ++i) {
        Sum value;
        for (std::size_t c = 0; c < r; ++c) value.push(inv[a * r + c] * std::conj(t[result.occupied_indices_[i] * r + c]));
        result.b_[a * p + i] = value.value();
    }
    for (std::size_t a = 0; a < e; ++a) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t c = 0; c < r; ++c) value.push(t[a * r + c] * result.b_[c * p + i]);
        z[a * p + i] = value.value();
    }
    for (std::size_t i = 0; i < p; ++i) for (std::size_t j = 0; j < p; ++j) {
        Sum value; for (std::size_t a = 0; a < e; ++a) value.push(std::conj(z[a * p + i]) * z[a * p + j]);
        inv[i * p + j] = value.value();
    }
    hermitize(inv.data(), p, o, d); solve(inv.data(), p, m.data(), u.data(), eigen.data(), o, d);
    full_rank(eigen.data(), p, o.depolarized_rank_absolute_floor, o.depolarized_rank_relative_floor,
        d.minimum_depolarized_gram_eigenvalue, d.depolarized_rank_boundary, "Bloch IAO depolarized Gram lost occupied rank");
    for (std::size_t a = 0; a < e; ++a) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t j = 0; j < p; ++j) value.push(z[a * p + j] * u[j * p + i]);
        q[a * p + i] = value.value() / std::sqrt(eigen[i]);
        if (!finite(q[a * p + i])) throw std::overflow_error("Bloch IAO depolarized normalization overflow");
    }
    // Canonical Q differs from symmetric orth(Z) by a unitary only. Eq17
    // consumes QQ^H, so no eigenvector gauge or rotated atomic label escapes.
    for (std::size_t i = 0; i < p; ++i) for (std::size_t j = 0; j < p; ++j) {
        Sum value; for (std::size_t a = 0; a < e; ++a) value.push(std::conj(q[a * p + i]) * q[a * p + j]);
        check(value.value(), double(i == j), d.maximum_depolarized_metric_residual, o, "Bloch IAO depolarized metric audit failed");
    }
    // inv's [p,p] prefix is Q^H Z for an independent range audit.
    for (std::size_t i = 0; i < p; ++i) for (std::size_t j = 0; j < p; ++j) {
        Sum value; for (std::size_t a = 0; a < e; ++a) value.push(std::conj(q[a * p + i]) * z[a * p + j]);
        inv[i * p + j] = value.value();
    }
    for (std::size_t a = 0; a < e; ++a) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t j = 0; j < p; ++j) value.push(q[a * p + j] * inv[j * p + i]);
        check(value.value(), z[a * p + i], d.maximum_depolarized_span_residual, o, "Bloch IAO depolarized occupied span changed");
    }
    // Reuse final B's r*p elements as V=Q^H T [p,r].
    for (std::size_t i = 0; i < p; ++i) for (std::size_t rho = 0; rho < r; ++rho) {
        Sum value; for (std::size_t a = 0; a < e; ++a) value.push(std::conj(q[a * p + i]) * t[a * r + rho]);
        result.b_[i * r + rho] = value.value();
    }
    for (std::size_t a = 0; a < e; ++a) for (std::size_t rho = 0; rho < r; ++rho) {
        Sum value; for (std::size_t i = 0; i < p; ++i) value.push(q[a * p + i] * result.b_[i * r + rho]);
        result.a_[a * r + rho] = (active[a] || core[a]) ? value.value() : t[a * r + rho] - value.value();
    }
    for (std::size_t i = 0; i < r; ++i) for (std::size_t j = 0; j < r; ++j) {
        Sum value; for (std::size_t a = 0; a < e; ++a) value.push(std::conj(result.a_[a * r + i]) * result.a_[a * r + j]);
        result.g_[i * r + j] = value.value();
    }
    hermitize(result.g_.data(), r, o, d); solve(result.g_.data(), r, m.data(), u.data(), eigen.data(), o, d);
    full_rank(eigen.data(), r, o.iao_rank_absolute_floor, o.iao_rank_relative_floor,
        d.minimum_iao_gram_eigenvalue, d.iao_rank_boundary, "Bloch IAO final Gram lost minimal atomic rank");
    inverse(u.data(), eigen.data(), r, inv.data());
    for (std::size_t rho = 0; rho < r; ++rho) for (std::size_t i = 0; i < p; ++i) result.d_[rho * p + i] = std::conj(result.a_[result.occupied_indices_[i] * r + rho]);
    for (std::size_t rho = 0; rho < r; ++rho) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t sigma = 0; sigma < r; ++sigma) value.push(inv[rho * r + sigma] * result.d_[sigma * p + i]);
        result.b_[rho * p + i] = value.value();
    }
    charge(memory.validation_work_units, o, d);
    // Q workspace now holds G*B [r,p], independently checked against D.
    for (std::size_t rho = 0; rho < r; ++rho) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t sigma = 0; sigma < r; ++sigma) value.push(result.g_[rho * r + sigma] * result.b_[sigma * p + i]);
        q[rho * p + i] = value.value();
        check(q[rho * p + i], result.d_[rho * p + i], d.maximum_covariant_residual, o, "Bloch IAO G*B=D audit failed");
    }
    for (std::size_t i = 0; i < p; ++i) for (std::size_t j = 0; j < p; ++j) {
        Sum value; for (std::size_t rho = 0; rho < r; ++rho) value.push(std::conj(result.b_[rho * p + i]) * q[rho * p + j]);
        check(value.value(), double(i == j), d.maximum_occupied_metric_residual, o, "Bloch IAO occupied metric audit failed");
    }
    // Original Z workspace is reused for A*B, never a full projector.
    for (std::size_t a = 0; a < e; ++a) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t rho = 0; rho < r; ++rho) value.push(result.a_[a * r + rho] * result.b_[rho * p + i]);
        z[a * p + i] = value.value();
        check(z[a * p + i], double(a == result.occupied_indices_[i]), d.maximum_retained_reconstruction_residual,
            o, "Bloch IAO lost an original occupied band in retained reconstruction");
    }
    for (std::size_t a = 0; a < e; ++a) for (std::size_t c = 0; c < e; ++c) {
        Sum value; for (std::size_t i = 0; i < p; ++i) value.push(z[a * p + i] * std::conj(z[c * p + i]));
        check(value.value(), double(a == c && (active[a] || core[a])), d.maximum_occupied_projector_residual,
            o, "Bloch IAO changed the full frozen-plus-active occupied projector");
    }
    for (std::size_t mu = 0; mu < b; ++mu) for (std::size_t i = 0; i < p; ++i) {
        Sum value; for (std::size_t a = 0; a < e; ++a) value.push(y(mu, a) * z[a * p + i]);
        check(value.value(), y(mu, result.occupied_indices_[i]), d.maximum_ao_reconstruction_residual,
            o, "Bloch IAO original AO occupied reconstruction failed");
    }
    // Independently evaluate (Y*A)^H S1 (Y*A) with two AO columns.
    for (std::size_t j = 0; j < r; ++j) {
        for (std::size_t mu = 0; mu < b; ++mu) {
            Sum value; for (std::size_t a = 0; a < e; ++a) value.push(y(mu, a) * result.a_[a * r + j]);
            ao[mu] = value.value();
        }
        for (std::size_t mu = 0; mu < b; ++mu) {
            Sum value; for (std::size_t nu = 0; nu < b; ++nu) value.push(state.overlap(point)(mu, nu) * ao[nu]);
            ao[b + mu] = value.value();
        }
        for (std::size_t i = 0; i < r; ++i) {
            Sum value;
            for (std::size_t mu = 0; mu < b; ++mu) {
                Sum ai; for (std::size_t a = 0; a < e; ++a) ai.push(y(mu, a) * result.a_[a * r + i]);
                value.push(std::conj(ai.value()) * ao[b + mu]);
            }
            check(value.value(), result.g_[i * r + j], d.maximum_ao_iao_metric_residual, o, "Bloch IAO original AO metric audit failed");
        }
    }
    result.point_ = point; result.state_ = reference.state_handle(); result.memory_ = memory;
    result.options_ = o; result.diagnostics_ = d; result.provenance_ = provenance;
    result.allocation_identity_ = dims.allocation_identity;
    Digest source("vibeqc.periodic.correlation.bloch-iao.input");
    source.u64(point); source.u64(b); source.u64(r);
    for (std::size_t i = 0; i < b * r; ++i) source.value(cross[i]);
    for (std::size_t i = 0; i < r * r; ++i) source.value(minimal[i]);
    for (std::size_t i = 0; i < r; ++i) source.u64(labels[i]);
    result.input_digest_ = source.finish();
    Digest output("vibeqc.periodic.correlation.bloch-iao.output");
    output.u64(e); output.u64(p); output.u64(r);
    for (const auto* vector : {&result.a_, &result.g_, &result.b_, &result.d_}) for (const auto value : *vector) output.value(value);
    for (const auto label : result.atom_labels_) output.u64(label);
    for (const auto band : result.occupied_indices_) output.u64(band);
    result.output_digest_ = output.finish();
    Digest identity("vibeqc.periodic.correlation.bloch-iao.identity");
    identity.text(state.state_identity_sha256()); identity.text(dims.allocation_identity);
    identity.text(provenance.cross_overlap_identity); identity.text(provenance.minimal_basis_identity);
    identity.text(result.input_digest_); identity.text(result.output_digest_);
    identity.text("caller-declared-one-k-overlaps;unorthogonalized-retained-space-IAO;all-occupied;no-physical-source-certification");
    identity.u64(point); identity.u64(provenance.caller_live_numerical_bytes);
    for (const double value : {o.minimal_rank_absolute_floor, o.minimal_rank_relative_floor,
            o.depolarized_rank_absolute_floor, o.depolarized_rank_relative_floor,
            o.iao_rank_absolute_floor, o.iao_rank_relative_floor,
            o.schur_negative_absolute_tolerance, o.schur_negative_relative_tolerance,
            o.validation_absolute_tolerance, o.validation_relative_tolerance, o.jacobi_relative_tolerance}) identity.real(value);
    identity.u64(o.jacobi_max_sweeps); identity.u64(o.maximum_work_units);
    result.identity_digest_ = identity.finish();
    return result;
}
}  // namespace vibeqc
