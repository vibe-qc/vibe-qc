#include "vibeqc/periodic_correlation_diabatic_seed.hpp"

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
using Complex = std::complex<double>;
using Options = PeriodicCorrelationDiabaticSeedOptions;
using Diagnostics = PeriodicCorrelationDiabaticSeedDiagnostics;
using State = PeriodicRestrictedMeanFieldState;

static_assert(sizeof(double) == 8U && sizeof(Complex) == 16U
              && std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits == 53,
              "Diabatic seed requires IEEE binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Diabatic seed forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Diabatic seed requires binary64 evaluation"
#endif

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("Diabatic seed count/byte/work product overflow");
    }
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("Diabatic seed count/byte/work sum overflow");
    }
    return a + b;
}
bool finite(Complex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }
void require_finite(Complex z) {
    if (!finite(z)) throw std::overflow_error("Diabatic seed contraction is non-finite");
}
double magnitude(Complex z) {
    const double result = std::abs(z);
    if (!std::isfinite(result)) throw std::overflow_error("Diabatic seed magnitude overflow");
    return result;
}
struct StableSum {
    double re = 0.0, im = 0.0, rc = 0.0, ic = 0.0;
    static void lane(double t, double& x, double& c) {
        const double next = x + t;
        c += std::abs(x) >= std::abs(t) ? (x - next) + t : (t - next) + x;
        x = next;
        if (!std::isfinite(x) || !std::isfinite(c)) {
            throw std::overflow_error("Diabatic seed compensated sum overflow");
        }
    }
    void push(Complex z) { require_finite(z); lane(z.real(), re, rc); lane(z.imag(), im, ic); }
    Complex value() const { Complex z(re + rc, im + ic); require_finite(z); return z; }
};

void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0.0)
        || std::fma(tiny, one, zero) != tiny) {
        throw std::invalid_argument("Diabatic seed requires round-to-nearest and gradual underflow");
    }
}
void options_valid(const Options& o) {
    for (const double v : {o.absolute_tolerance, o.relative_tolerance,
             o.unitarity_tolerance, o.cholesky_relative_floor, o.singular_relative_floor,
             o.jacobi_relative_tolerance}) {
        if (!std::isfinite(v) || v < 0.0 || v >= 1.0) {
            throw std::invalid_argument("Diabatic seed tolerances/floors must be finite in [0,1)");
        }
    }
    for (const double floor : {o.cholesky_absolute_floor, o.singular_absolute_floor}) {
        if (!std::isfinite(floor) || floor < 0.0) {
            throw std::invalid_argument("Diabatic seed absolute rank floors must be finite and nonnegative");
        }
    }
    if (o.unitarity_tolerance == 0.0 || o.jacobi_relative_tolerance == 0.0
        || o.jacobi_max_sweeps == 0U || o.maximum_work_units == 0U
        || (o.absolute_tolerance == 0.0 && o.relative_tolerance == 0.0)
        || (o.cholesky_absolute_floor == 0.0 && o.cholesky_relative_floor == 0.0)
        || (o.singular_absolute_floor == 0.0 && o.singular_relative_floor == 0.0)) {
        throw std::invalid_argument("Diabatic seed requires explicit positive numerical/work controls");
    }
}
void charge(std::uint64_t work, const Options& o, Diagnostics& d) {
    const auto next = add(d.charged_work_units, work);
    if (next > o.maximum_work_units) throw std::length_error("Diabatic seed scalar-loop work budget exceeded");
    d.charged_work_units = next;
}
void compare(Complex a, Complex b, double& maximum, const Options& o, const char* message) {
    const double error = magnitude(a - b);
    maximum = std::max(maximum, error);
    if (error > o.absolute_tolerance + o.relative_tolerance * std::max(magnitude(a), magnitude(b))) {
        throw std::invalid_argument(message);
    }
}
void reference_valid(const PeriodicCorrelationAdmittedReference& r) {
    if (!r.state_handle()) throw std::invalid_argument("Diabatic seed requires a live admitted reference");
    const auto& s = r.state(); const auto& d = r.dimensions(); const auto& p = r.plan();
    if (s.is_shift() != std::array<int, 3>{0, 0, 0}) {
        throw std::invalid_argument("Diabatic seed requires an exact Gamma-centered full mesh");
    }
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
        || s.n_kpoints() != d.n_kpoints || s.n_basis() != d.n_basis
        || s.n_effective_orbitals() != d.n_effective_orbitals
        || s.n_correlated_occupied() != d.n_home_occupied
        || s.n_frozen_core() + s.n_correlated_occupied() != d.n_home_total_occupied) {
        throw std::invalid_argument("Diabatic seed admitted-reference seals are inconsistent");
    }
}

Complex density(const State& state, std::size_t k, std::size_t mu,
                std::size_t nu, bool frozen = false) {
    const auto& mask = frozen ? state.frozen_core_mask(k) : state.correlated_occupied_mask(k);
    const auto& c = state.coefficients(k);
    StableSum total;
    for (std::size_t band = 0; band < mask.size(); ++band) {
        if (mask[band]) total.push(c(mu, band) * std::conj(c(nu, band)));
    }
    return total.value();
}
void preflight(const State& s, const Options& o, Diagnostics& d) {
    const RegularKMesh mesh(s.mesh());
    const auto b = static_cast<std::size_t>(s.n_basis());
    for (std::size_t k = 0; k < s.n_kpoints(); ++k) {
        const auto& active = s.correlated_occupied_mask(k);
        const auto& core = s.frozen_core_mask(k); const auto& virt = s.virtual_mask(k);
        if (active.size() != s.n_effective_orbitals() || core.size() != active.size() || virt.size() != active.size()) {
            throw std::invalid_argument("Diabatic seed occupied-mask extents are inconsistent");
        }
        std::uint64_t na = 0, nf = 0;
        for (std::size_t j = 0; j < active.size(); ++j) {
            if (active[j] > 1U || core[j] > 1U || virt[j] > 1U || active[j] + core[j] + virt[j] != 1U) {
                throw std::invalid_argument("Diabatic seed masks must be disjoint and exhaustive");
            }
            na += active[j]; nf += core[j];
            for (std::size_t mu = 0; mu < b; ++mu) require_finite(s.coefficients(k)(mu, j));
        }
        if (na != s.n_correlated_occupied() || nf != s.n_frozen_core()) {
            throw std::invalid_argument("Diabatic seed active/frozen ranks vary across k");
        }
        const auto partner = mesh.negate_index(k);
        for (std::size_t mu = 0; mu < b; ++mu) {
            for (std::size_t nu = 0; nu < b; ++nu) {
                compare(s.overlap(k)(mu, nu), std::conj(s.overlap(partner)(mu, nu)),
                    d.maximum_overlap_time_reversal_residual, o, "Diabatic seed overlap failed time reversal");
                compare(s.fock(k)(mu, nu), std::conj(s.fock(partner)(mu, nu)),
                    d.maximum_fock_time_reversal_residual, o, "Diabatic seed Fock failed time reversal");
                compare(density(s, k, mu, nu), std::conj(density(s, partner, mu, nu)),
                    d.maximum_active_projector_time_reversal_residual, o,
                    "Diabatic seed active projector failed time reversal");
                compare(density(s, k, mu, nu, true), std::conj(density(s, partner, mu, nu, true)),
                    d.maximum_frozen_projector_time_reversal_residual, o,
                    "Diabatic seed frozen projector failed time reversal");
            }
        }
    }
}

// C C^H is a rank-o positive matrix without a spin factor. At Gamma it
// is real under the audited TR contract. Pivoted Cholesky produces L=C O;
// all o columns must be recovered, and both density and metric are audited.
void anchor(const State& s, Complex* l, double* diagonal, std::uint64_t* pivots,
            const Options& o, Diagnostics& d) {
    const auto b = static_cast<std::size_t>(s.n_basis()), n = static_cast<std::size_t>(s.n_correlated_occupied());
    double largest = 0.0;
    for (std::size_t mu = 0; mu < b; ++mu) {
        diagonal[mu] = density(s, 0, mu, mu).real();
        largest = std::max(largest, diagonal[mu]);
    }
    const double floor = std::max(o.cholesky_absolute_floor, o.cholesky_relative_floor * largest);
    d.minimum_cholesky_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n; ++j) {
        std::size_t pivot = 0;
        for (std::size_t mu = 1; mu < b; ++mu) if (diagonal[mu] > diagonal[pivot]) pivot = mu;
        const double p = diagonal[pivot];
        if (!std::isfinite(p) || p <= floor) {
            throw std::invalid_argument("Diabatic seed Gamma Cholesky lost active rank at its strict floor");
        }
        pivots[j] = pivot; d.minimum_cholesky_pivot = std::min(d.minimum_cholesky_pivot, p);
        const double root = std::sqrt(p);
        for (std::size_t mu = 0; mu < b; ++mu) {
            const Complex raw = density(s, 0, mu, pivot);
            d.maximum_anchor_imaginary_correction = std::max(d.maximum_anchor_imaginary_correction, std::abs(raw.imag()));
            StableSum value; value.push(raw.real());
            for (std::size_t t = 0; t < j; ++t) value.push(-l[mu * n + t] * l[pivot * n + t]);
            const double v = value.value().real() / root;
            if (!std::isfinite(v)) throw std::overflow_error("Diabatic seed Gamma Cholesky overflow");
            l[mu * n + j] = Complex(v, 0.0);
        }
        for (std::size_t mu = 0; mu < b; ++mu) {
            if (diagonal[mu] < 0.0) continue;  // already selected
            const double v = l[mu * n + j].real();
            const double next = std::fma(-v, v, diagonal[mu]);
            if (!std::isfinite(next) || next < -(o.absolute_tolerance + o.relative_tolerance * largest)) {
                throw std::invalid_argument("Diabatic seed Gamma density has a negative Cholesky remainder");
            }
            diagonal[mu] = std::max(0.0, next);
        }
        diagonal[pivot] = -1.0;
    }
    for (std::size_t mu = 0; mu < b; ++mu) {
        for (std::size_t nu = 0; nu < b; ++nu) {
            StableSum value;
            for (std::size_t i = 0; i < n; ++i) value.push(l[mu * n + i] * std::conj(l[nu * n + i]));
            compare(value.value(), density(s, 0, mu, nu), d.maximum_anchor_density_residual,
                o, "Diabatic seed Gamma Cholesky does not preserve the active density");
        }
    }
}

void unitary(const Complex* u, std::size_t n, const Options& o, Diagnostics& d) {
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        StableSum left, right;
        for (std::size_t q = 0; q < n; ++q) {
            left.push(std::conj(u[q * n + i]) * u[q * n + j]);
            right.push(u[i * n + q] * std::conj(u[j * n + q]));
        }
        const double error = std::max(magnitude(left.value() - double(i == j)), magnitude(right.value() - double(i == j)));
        d.maximum_unitarity_residual = std::max(d.maximum_unitarity_residual, error);
        if (error > o.unitarity_tolerance) throw std::invalid_argument("Diabatic seed gauge is not unitary");
    }
}
void metric_audit(const State& s, std::size_t k, const Complex* frame, Complex* column,
                  const Options& o, Diagnostics& d) {
    const auto b = static_cast<std::size_t>(s.n_basis()), n = static_cast<std::size_t>(s.n_correlated_occupied());
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t mu = 0; mu < b; ++mu) {
            StableSum value;
            for (std::size_t nu = 0; nu < b; ++nu) value.push(s.overlap(k)(mu, nu) * frame[nu * n + j]);
            column[mu] = value.value();
        }
        for (std::size_t i = 0; i < n; ++i) {
            StableSum value;
            for (std::size_t mu = 0; mu < b; ++mu) value.push(std::conj(frame[mu * n + i]) * column[mu]);
            const double error = magnitude(value.value() - double(i == j));
            d.maximum_metric_orthonormality_residual = std::max(d.maximum_metric_orthonormality_residual, error);
            if (error > o.unitarity_tolerance) throw std::invalid_argument("Diabatic seed physical frame is not metric orthonormal");
        }
    }
}
void reconstruct(const State& s, std::size_t k, const std::uint64_t* index,
                 const Complex* u, Complex* frame) {
    const auto b = static_cast<std::size_t>(s.n_basis()), n = static_cast<std::size_t>(s.n_correlated_occupied());
    for (std::size_t mu = 0; mu < b; ++mu) for (std::size_t j = 0; j < n; ++j) {
        StableSum value;
        for (std::size_t i = 0; i < n; ++i) value.push(s.coefficients(k)(mu, index[i]) * u[i * n + j]);
        frame[mu * n + j] = value.value();
    }
}
// Projection into a certified complete active frame. Partner sewing is not
// an assumption about incoming canonical eigenvector phases.
void metric_project(const State& s, std::size_t k, const std::uint64_t* index,
                    const Complex* frame, bool conjugate, Complex* column, Complex* u) {
    const auto b = static_cast<std::size_t>(s.n_basis()), n = static_cast<std::size_t>(s.n_correlated_occupied());
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t mu = 0; mu < b; ++mu) {
            StableSum value;
            for (std::size_t nu = 0; nu < b; ++nu) {
                const Complex z = frame[nu * n + j];
                value.push(s.overlap(k)(mu, nu) * (conjugate ? std::conj(z) : z));
            }
            column[mu] = value.value();
        }
        for (std::size_t i = 0; i < n; ++i) {
            StableSum value;
            for (std::size_t mu = 0; mu < b; ++mu) value.push(std::conj(s.coefficients(k)(mu, index[i])) * column[mu]);
            u[i * n + j] = value.value();
        }
    }
}

void polar(const State& s, std::size_t k, const std::uint64_t* index,
           const Complex* l, Complex* a, Complex* h, Complex* v, double* eigen,
           Complex* out, const Options& o, Diagnostics& d) {
    const auto b = static_cast<std::size_t>(s.n_basis()), n = static_cast<std::size_t>(s.n_correlated_occupied());
    const std::size_t size = 2 * n;
    std::fill(h, h + size * size, Complex(0.0, 0.0));
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        StableSum value;
        for (std::size_t mu = 0; mu < b; ++mu) value.push(std::conj(s.coefficients(k)(mu, index[i])) * l[mu * n + j]);
        a[i * n + j] = value.value(); scale = std::max(scale, magnitude(a[i * n + j]));
        h[i * size + n + j] = a[i * n + j];
        h[(n + j) * size + i] = std::conj(a[i * n + j]);
    }
    const auto solved = hermitian_jacobi_in_place(h, size * size, v, size * size,
        eigen, size, size, {o.jacobi_max_sweeps, o.jacobi_relative_tolerance});
    if (solved.status != HermitianJacobiStatus::Success || solved.scaling_underflow_components != 0U) {
        throw std::runtime_error("Diabatic seed polar Jacobi failed or lost input range");
    }
    d.jacobi_sweeps = add(d.jacobi_sweeps, solved.sweeps);
    const double largest = std::max(std::abs(eigen[0]), std::abs(eigen[size - 1]));
    const double floor = std::max(o.singular_absolute_floor, o.singular_relative_floor * largest);
    if (!std::isfinite(largest) || largest <= floor) {
        throw std::invalid_argument("Diabatic seed Procrustes similarity has deficient rank");
    }
    for (std::size_t t = 0; t < n; ++t) {
        if (eigen[t] >= -floor || eigen[n + t] <= floor) {
            throw std::invalid_argument("Diabatic seed Procrustes similarity has deficient rank at strict singular floor");
        }
        d.minimum_similarity_singular_value = std::min(d.minimum_similarity_singular_value,
            std::min(-eigen[t], eigen[n + t]));
        d.maximum_similarity_singular_value = std::max(d.maximum_similarity_singular_value,
            std::max(-eigen[t], eigen[n + t]));
    }
    // Reconstruct ORIGINAL dilation residual from A, never trust only the
    // transformed Jacobi off-diagonal. Independent of eigenvector gauges.
    for (std::size_t row = 0; row < size; ++row) for (std::size_t t = 0; t < size; ++t) {
        StableSum value;
        for (std::size_t j = 0; j < n; ++j) {
            value.push(row < n ? a[row * n + j] * v[(n + j) * size + t]
                : std::conj(a[j * n + row - n]) * v[j * size + t]);
        }
        const double error = magnitude(value.value() - eigen[t] * v[row * size + t]);
        const double normalized = scale == 0.0 ? error : error / scale;
        d.maximum_dilation_eigen_residual = std::max(d.maximum_dilation_eigen_residual, normalized);
        if (!std::isfinite(normalized) || normalized > o.absolute_tolerance + o.relative_tolerance * size) {
            throw std::invalid_argument("Diabatic seed original dilation eigensystem failed audit");
        }
    }
    // Off-diagonal block of sign(H). Both spectral halves are used.
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        StableSum value;
        for (std::size_t t = 0; t < size; ++t) {
            value.push((t < n ? -1.0 : 1.0) * v[i * size + t] * std::conj(v[(n + j) * size + t]));
        }
        out[i * n + j] = value.value();
    }
    unitary(out, n, o, d);
    StableSum trace, optimum;
    for (std::size_t i = 0; i < n; ++i) {
        optimum.push(eigen[n + i]);
        for (std::size_t j = 0; j < n; ++j) trace.push(std::conj(out[i * n + j]) * a[i * n + j]);
    }
    compare(trace.value(), optimum.value(), d.maximum_procrustes_trace_residual, o,
        "Diabatic seed polar does not attain the Procrustes trace optimum");
    ++d.polar_point_count;
}

class Digest {
public:
    explicit Digest(const char* domain) { string(domain); u32(kPeriodicCorrelationDiabaticSeedContractVersion); }
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> b{};
        for (unsigned i = 0; i < 4; ++i) b[i] = value >> (24U - 8U * i);
        h_.update(b.data(), b.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i < 8; ++i) b[i] = value >> (56U - 8U * i);
        h_.update(b.data(), b.size());
    }
    void real(double value) {
        if (!std::isfinite(value)) throw std::overflow_error("Diabatic seed digest is non-finite");
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); u64(bits);
    }
    void complex(Complex value) { real(value.real()); real(value.imag()); }
    void string(const std::string& text) {
        u64(text.size()); h_.update(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    }
    std::string finish() { return h_.finish_hex(); }
private:
    detail::Sha256 h_;
};
}  // namespace

PeriodicCorrelationDiabaticSeedMemoryPlan plan_periodic_correlation_diabatic_seed(
    std::array<int, 3> mesh, std::uint64_t b, std::uint64_t n,
    std::uint64_t effective, std::uint64_t sweeps) {
    const RegularKMesh addressing(mesh);
    if (b == 0U || n == 0U || n > effective || effective > b || sweeps == 0U) {
        throw std::invalid_argument("Diabatic seed requires 0 < active <= effective <= basis and positive sweeps");
    }
    (void) estimate_periodic_correlation_translation_pair_counts(mesh, n);
    PeriodicCorrelationDiabaticSeedMemoryPlan p;
    p.n_cells = addressing.size(); p.n_basis = b; p.n_active = n; p.n_effective_orbitals = effective;
    p.self_inverse_count = 1U;
    for (const int axis : mesh) if (axis % 2 == 0) p.self_inverse_count *= 2U;
    const auto nn = mul(n, n), bn = mul(b, n), bb = mul(b, b);
    p.gauge_count = mul(p.n_cells, nn); p.active_index_count = mul(p.n_cells, n);
    p.retained_gauge_bytes = mul(16U, p.gauge_count);
    p.retained_index_bytes = mul(8U, add(p.active_index_count, n));
    p.output_numerical_bytes = add(p.retained_gauge_bytes, p.retained_index_bytes);
    p.frame_workspace_bytes = mul(32U, bn);
    p.polar_workspace_bytes = mul(144U, nn); // A[o,o], H[2o,2o], V[2o,2o]
    p.scalar_workspace_bytes = add(mul(16U, n), mul(24U, b)); // eigen[2o], diag[B], column[B]
    p.peak_owned_numerical_bytes = add(p.output_numerical_bytes,
        add(p.frame_workspace_bytes, add(p.polar_workspace_bytes, p.scalar_workspace_bytes)));
    p.reference_preflight_work_units = mul(p.n_cells,
        add(mul(16U, mul(bb, effective)), add(mul(16U, bb), mul(16U, mul(b, effective)))));
    p.anchor_work_units = add(mul(64U, mul(bb, n)), add(mul(128U, mul(b, nn)), mul(128U, bn)));
    const auto two_n = mul(2U, n);
    p.polar_work_units_per_point = add(mul(1024U, mul(sweeps, mul(mul(two_n, two_n), two_n))),
        add(mul(128U, mul(b, nn)), add(mul(128U, mul(nn, n)), mul(128U, nn))));
    p.physical_work_units_per_point = add(mul(64U, mul(bb, n)),
        add(mul(128U, mul(b, nn)), add(mul(128U, mul(nn, n)), mul(64U, bn))));
    const auto polar_points = add(p.n_cells, p.self_inverse_count) / 2U - 1U;
    p.maximum_work_units = add(p.reference_preflight_work_units, add(p.anchor_work_units,
        add(mul(polar_points, p.polar_work_units_per_point), mul(p.n_cells, p.physical_work_units_per_point))));
    const auto max_address = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto max_hash = std::numeric_limits<std::uint64_t>::max() / 8U - 4096U;
    if (p.peak_owned_numerical_bytes > max_address || p.output_numerical_bytes > max_hash
        || p.gauge_count > std::vector<Complex>().max_size()
        || mul(4U, nn) > std::vector<Complex>().max_size() || bn > std::vector<Complex>().max_size()
        || p.active_index_count > std::vector<std::uint64_t>().max_size()) {
        throw std::length_error("Diabatic seed arrays exceed native address/vector/hash extents");
    }
    return p;
}

const State& PeriodicCorrelationDiabaticSeed::state() const {
    if (!state_) throw std::logic_error("Diabatic seed state owner is consumed");
    return *state_;
}
const Complex* PeriodicCorrelationDiabaticSeed::gauges_data() const {
    if (!state_ || gauges_.size() != memory_.gauge_count || memory_.gauge_count == 0U
        || active_indices_.size() != memory_.active_index_count || pivots_.size() != memory_.n_active) {
        throw std::logic_error("Diabatic seed numerical owner is consumed");
    }
    return gauges_.data();
}
const Complex* PeriodicCorrelationDiabaticSeed::point_gauge(std::size_t point) const {
    const auto* data = gauges_data();
    if (point >= memory_.n_cells) throw std::out_of_range("Diabatic seed point index out of range");
    return data + point * memory_.n_active * memory_.n_active;
}
Complex PeriodicCorrelationDiabaticSeed::gauge(std::size_t k, std::size_t i, std::size_t j) const {
    if (i >= memory_.n_active || j >= memory_.n_active) throw std::out_of_range("Diabatic seed gauge index out of range");
    return point_gauge(k)[i * memory_.n_active + j];
}
std::uint64_t PeriodicCorrelationDiabaticSeed::active_band(std::size_t k, std::size_t i) const {
    (void) point_gauge(k);
    if (i >= memory_.n_active) throw std::out_of_range("Diabatic seed active index out of range");
    return active_indices_[k * memory_.n_active + i];
}
std::uint64_t PeriodicCorrelationDiabaticSeed::gamma_pivot(std::size_t i) const {
    (void) gauges_data();
    if (i >= memory_.n_active) throw std::out_of_range("Diabatic seed pivot index out of range");
    return pivots_[i];
}

PeriodicCorrelationDiabaticSeed make_periodic_correlation_diabatic_seed(
    const PeriodicCorrelationAdmittedReference& reference, std::uint64_t cap, const Options& options) {
    reference_valid(reference); options_valid(options);
    const auto& state = reference.state();
    const auto memory = plan_periodic_correlation_diabatic_seed(state.mesh(), state.n_basis(),
        state.n_correlated_occupied(), state.n_effective_orbitals(), options.jacobi_max_sweeps);
    if (cap == 0U || cap < memory.peak_owned_numerical_bytes) {
        throw std::length_error("Diabatic seed numerical byte cap is missing or exceeded");
    }
    const auto& dims = reference.dimensions(); const auto& budget = reference.budget();
    const auto required = add(add(dims.external_bytes, dims.shared_bytes),
        add(mul(budget.mpi_ranks, add(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), memory.peak_owned_numerical_bytes)));
    if (budget.memory_limit_bytes == 0U || required > budget.memory_limit_bytes) {
        throw std::length_error("Diabatic seed peak plus immutable reference inventory exceeds admitted node memory");
    }
    Diagnostics diagnostics; diagnostics.required_node_memory_bytes = required;
    diagnostics.minimum_similarity_singular_value = std::numeric_limits<double>::infinity();
    environment(); charge(memory.reference_preflight_work_units, options, diagnostics);
    preflight(state, options, diagnostics);
    // All numerical allocations below, including retained map/pivots, are
    // simultaneous and were admitted together. No Eigen/BLAS expressions.
    const auto kcount = static_cast<std::size_t>(memory.n_cells);
    const auto b = static_cast<std::size_t>(memory.n_basis), n = static_cast<std::size_t>(memory.n_active);
    PeriodicCorrelationDiabaticSeed result;
    result.gauges_.resize(memory.gauge_count); result.active_indices_.resize(memory.active_index_count);
    result.pivots_.resize(n);
    std::vector<Complex> l(b * n), current(b * n), a(n * n), h(4 * n * n), vectors(4 * n * n), column(b);
    std::vector<double> eigen(2 * n), diagonal(b);
    for (std::size_t k = 0; k < kcount; ++k) {
        std::size_t count = 0;
        const auto& mask = state.correlated_occupied_mask(k);
        for (std::size_t j = 0; j < mask.size(); ++j) if (mask[j]) result.active_indices_[k * n + count++] = j;
    }
    charge(memory.anchor_work_units, options, diagnostics);
    anchor(state, l.data(), diagonal.data(), result.pivots_.data(), options, diagnostics);
    metric_audit(state, 0, l.data(), column.data(), options, diagnostics);
    const RegularKMesh mesh(state.mesh());
    for (std::size_t k = 0; k < kcount; ++k) {
        const auto minus = mesh.negate_index(k);
        if (k > minus) continue;
        auto* u = result.gauges_.data() + k * n * n;
        const auto* index = result.active_indices_.data() + k * n;
        charge(memory.physical_work_units_per_point, options, diagnostics);
        if (k == 0U) {
            metric_project(state, k, index, l.data(), false, column.data(), u);
        } else {
            charge(memory.polar_work_units_per_point, options, diagnostics);
            polar(state, k, index, l.data(), a.data(), h.data(), vectors.data(), eigen.data(), u, options, diagnostics);
        }
        unitary(u, n, options, diagnostics);
        reconstruct(state, k, index, u, current.data());
        metric_audit(state, k, current.data(), column.data(), options, diagnostics);
        if (k == 0U) for (std::size_t entry = 0; entry < b * n; ++entry) {
            compare(current[entry], l[entry], diagnostics.maximum_physical_reconstruction_residual,
                options, "Diabatic seed Gamma gauge does not reconstruct its anchor");
        }
        if (k == minus) {
            ++diagnostics.self_inverse_point_count;
            for (const auto z : current) {
                diagnostics.maximum_self_inverse_imaginary_magnitude = std::max(
                    diagnostics.maximum_self_inverse_imaginary_magnitude, std::abs(z.imag()));
                compare(z, std::conj(z), diagnostics.maximum_physical_time_reversal_residual,
                    options, "Diabatic seed physical self-inverse frame is not real");
            }
        } else {
            charge(memory.physical_work_units_per_point, options, diagnostics);
            auto* partner_u = result.gauges_.data() + minus * n * n;
            const auto* partner_index = result.active_indices_.data() + minus * n;
            metric_project(state, minus, partner_index, current.data(), true, column.data(), partner_u);
            unitary(partner_u, n, options, diagnostics);
            // Compare each reconstructed partner component without a third
            // B*o frame. Its target is the conjugate of current.
            for (std::size_t mu = 0; mu < b; ++mu) for (std::size_t j = 0; j < n; ++j) {
                StableSum value;
                for (std::size_t i = 0; i < n; ++i) value.push(state.coefficients(minus)(mu, partner_index[i]) * partner_u[i * n + j]);
                compare(value.value(), std::conj(current[mu * n + j]),
                    diagnostics.maximum_physical_time_reversal_residual, options,
                    "Diabatic seed physical partner sewing failed reconstruction");
            }
            // The representative is no longer needed after the independent
            // sewing check: reuse that frame for the partner's metric audit.
            reconstruct(state, minus, partner_index, partner_u, current.data());
            metric_audit(state, minus, current.data(), column.data(), options, diagnostics);
            ++diagnostics.sewn_pair_count;
        }
    }
    if (diagnostics.polar_point_count == 0U) diagnostics.minimum_similarity_singular_value = 0.0;
    result.state_ = reference.state_handle(); result.memory_ = memory; result.options_ = options;
    result.diagnostics_ = diagnostics; result.allocation_identity_ = dims.allocation_identity;
    Digest payload("vibeqc.periodic.correlation.diabatic-seed.gauges");
    payload.u64(kcount); payload.u64(n);
    for (const auto z : result.gauges_) payload.complex(z);
    result.gauge_digest_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.diabatic-seed.identity");
    identity.string(state.state_identity_sha256()); identity.string(dims.allocation_identity);
    identity.string(result.gauge_digest_);
    for (const auto i : result.active_indices_) identity.u64(i);
    for (const auto i : result.pivots_) identity.u64(i);
    identity.string("Gamma-active-density-pivoted-Cholesky;coefficient-polar;physical-TR-sewing;seed-only");
    for (const double value : {options.absolute_tolerance, options.relative_tolerance, options.unitarity_tolerance,
            options.cholesky_absolute_floor, options.cholesky_relative_floor,
            options.singular_absolute_floor, options.singular_relative_floor, options.jacobi_relative_tolerance}) identity.real(value);
    identity.u64(options.jacobi_max_sweeps); identity.u64(options.maximum_work_units);
    result.identity_digest_ = identity.finish();
    return result;
}

}  // namespace vibeqc
