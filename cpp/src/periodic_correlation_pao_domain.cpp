#include "vibeqc/periodic_correlation_pao_domain.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
using Options = PeriodicCorrelationPAODomainOptions;
using Diagnostics = PeriodicCorrelationPAODomainDiagnostics;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;
static_assert(sizeof(PeriodicPAODomainColumn) == 16U && sizeof(Complex) == 16U,
              "PAO domain inventory requires two uint64 indices and complex128");
static_assert(std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits == 53 && sizeof(double) == 8U,
              "PAO domain wire and arithmetic require IEEE binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "PAO domain contractions forbid fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "PAO domain contractions require binary64 evaluation"
#endif

std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("PAO domain count or byte product overflows uint64");
    }
    return a * b;
}

std::uint64_t sum(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("PAO domain memory sum overflows uint64");
    }
    return a + b;
}

bool finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

double magnitude(Complex value) {
    const double result = std::abs(value);
    if (!finite(value) || !std::isfinite(result)) {
        throw std::overflow_error("PAO domain contraction or diagnostic is non-finite");
    }
    return result;
}

void add_lane(double term, double& value, double& correction) {
    const double next = value + term;
    correction += std::abs(value) >= std::abs(term)
        ? (value - next) + term : (term - next) + value;
    value = next;
    if (!std::isfinite(value) || !std::isfinite(correction)) {
        throw std::overflow_error("PAO domain compensated contraction overflowed");
    }
}

void add_complex(Complex term, Complex& value, Complex& correction) {
    double real = value.real(), imag = value.imag();
    double rc = correction.real(), ic = correction.imag();
    add_lane(term.real(), real, rc);
    add_lane(term.imag(), imag, ic);
    value = Complex(real, imag);
    correction = Complex(rc, ic);
}

struct ComplexSum {
    Complex value{}, correction{};
    void add(Complex term) { add_complex(term, value, correction); }
    Complex result() const {
        const auto answer = value + correction;
        if (!finite(answer)) throw std::overflow_error("PAO domain sum is non-finite");
        return answer;
    }
};

bool compare(Complex left, Complex right, double absolute, double relative,
             double& maximum) {
    const double residual = magnitude(left - right);
    const double scale = std::max(magnitude(left), magnitude(right));
    const double bound = absolute + relative * scale;
    if (!std::isfinite(bound)) throw std::overflow_error("PAO domain tolerance bound overflowed");
    maximum = std::max(maximum, residual);
    return residual <= bound;
}

class PAODigest {
public:
    explicit PAODigest(const char* domain) {
        string(domain);
        u32(kPeriodicCorrelationPAODomainContractVersion);
    }
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (unsigned i = 0; i < 4U; ++i) bytes[i] = value >> (24U - 8U * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned i = 0; i < 8U; ++i) bytes[i] = value >> (56U - 8U * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void real(double value) {
        if (!std::isfinite(value)) throw std::overflow_error("PAO domain digest is non-finite");
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void complex(Complex value) { real(value.real()); real(value.imag()); }
    void string(const std::string& value) {
        u64(value.size());
        hash_.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
};

void require_options(const Options& options) {
    const std::array<double, 6> controls{
        options.hermitian_absolute_tolerance, options.hermitian_relative_tolerance,
        options.time_reversal_absolute_tolerance, options.time_reversal_relative_tolerance,
        options.real_absolute_tolerance, options.real_relative_tolerance};
    for (double value : controls) {
        if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
            throw std::invalid_argument("PAO domain tolerances must be finite and in [0,1)");
        }
    }
    if (options.hermitian_absolute_tolerance == 0.0
        && options.hermitian_relative_tolerance == 0.0) {
        throw std::invalid_argument("PAO domain requires a positive Hermitian tolerance");
    }
    if (std::fegetround() != FE_TONEAREST) {
        throw std::invalid_argument("PAO domain requires round-to-nearest arithmetic");
    }
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, zero = 0.0;
    if (!(tiny > 0.0) || std::fma(tiny, one, zero) != tiny) {
        throw std::invalid_argument("PAO domain requires gradual underflow");
    }
}

void require_reference(const PeriodicCorrelationAdmittedReference& reference) {
    if (!reference.state_handle()) throw std::invalid_argument("PAO domain requires a live reference");
    const auto& state = reference.state();
    const auto& dims = reference.dimensions();
    const auto& plan = reference.plan();
    if (state.is_shift() != std::array<int, 3>{0, 0, 0}) {
        throw std::invalid_argument("PAO domain v1 requires the exact Gamma-centered mesh");
    }
    if (reference.contract_version() != kPeriodicCorrelationAdmittedReferenceContractVersion
        || state.contract_version() != kPeriodicRestrictedMeanFieldStateContractVersion
        || state.normalization() != PeriodicMeanFieldNormalizationConvention::
            UnnormalizedAoBlochSumsUniformFullBzWeights
        || state.reference_kind() != PeriodicMeanFieldReferenceKind::RestrictedHartreeFock
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || dims.symmetry_reduction_requested || !dims.symmetry_mapping_identity.empty()
        || dims.symmetry_representative_count != dims.n_kpoints
        || dims.symmetry_weight_sum != dims.n_kpoints
        || plan.allocation_contract_version != dims.allocation_contract_version
        || plan.calculation_identity != dims.calculation_identity
        || plan.allocation_identity != dims.allocation_identity
        || state.calculation_identity() != dims.calculation_identity
        || state.mesh() != dims.mesh || state.is_shift() != dims.is_shift
        || state.n_kpoints() != dims.n_kpoints || state.n_basis() != dims.n_basis
        || state.n_effective_orbitals() != dims.n_effective_orbitals
        || state.n_virtual() != dims.n_home_virtual
        || state.n_correlated_occupied() != dims.n_home_occupied
        || state.n_frozen_core() + state.n_correlated_occupied() != dims.n_home_total_occupied) {
        throw std::logic_error("PAO domain received inconsistent admitted-reference seals");
    }
}

void audit_state(const PeriodicRestrictedMeanFieldState& state,
                 const RegularKMesh& mesh, const Options& options,
                 Diagnostics& diagnostics) {
    for (std::size_t k = 0; k < state.n_kpoints(); ++k) {
        const auto& frozen = state.frozen_core_mask(k);
        const auto& active = state.correlated_occupied_mask(k);
        const auto& virtuals = state.virtual_mask(k);
        if (frozen.size() != state.n_effective_orbitals() || active.size() != frozen.size()
            || virtuals.size() != frozen.size()) {
            throw std::logic_error("PAO domain mask extents are inconsistent");
        }
        std::uint64_t nf = 0, no = 0, nv = 0;
        for (std::size_t band = 0; band < frozen.size(); ++band) {
            if (frozen[band] > 1U || active[band] > 1U || virtuals[band] > 1U
                || frozen[band] + active[band] + virtuals[band] != 1U) {
                throw std::logic_error("PAO domain masks are not an exhaustive disjoint partition");
            }
            nf += frozen[band]; no += active[band]; nv += virtuals[band];
            for (std::size_t mu = 0; mu < state.n_basis(); ++mu) {
                if (!finite(state.coefficients(k)(mu, band))) {
                    throw std::invalid_argument("PAO domain state contains a non-finite coefficient");
                }
            }
        }
        if (nf != state.n_frozen_core() || no != state.n_correlated_occupied() || nv != state.n_virtual()) {
            throw std::logic_error("PAO domain mask counts vary over the mesh");
        }
        const auto minus_k = mesh.negate_index(k);
        for (std::size_t mu = 0; mu < state.n_basis(); ++mu) {
            for (std::size_t nu = 0; nu < state.n_basis(); ++nu) {
                const auto s = state.overlap(k)(mu, nu), f = state.fock(k)(mu, nu);
                const bool sh = compare(s, std::conj(state.overlap(k)(nu, mu)),
                    options.hermitian_absolute_tolerance, options.hermitian_relative_tolerance,
                    diagnostics.maximum_input_overlap_hermitian_defect);
                const bool fh = compare(f, std::conj(state.fock(k)(nu, mu)),
                    options.hermitian_absolute_tolerance, options.hermitian_relative_tolerance,
                    diagnostics.maximum_input_fock_hermitian_defect);
                if (!sh || !fh) throw std::invalid_argument("PAO domain input S/F failed Hermitian audit");
                const bool st = compare(s, std::conj(state.overlap(minus_k)(mu, nu)),
                    options.time_reversal_absolute_tolerance, options.time_reversal_relative_tolerance,
                    diagnostics.maximum_overlap_time_reversal_residual);
                const bool ft = compare(f, std::conj(state.fock(minus_k)(mu, nu)),
                    options.time_reversal_absolute_tolerance, options.time_reversal_relative_tolerance,
                    diagnostics.maximum_fock_time_reversal_residual);
                diagnostics.time_reversal_compatible &= st && ft;
            }
        }
    }
    if (options.require_time_reversal && !diagnostics.time_reversal_compatible) {
        throw std::invalid_argument("PAO domain reference overlap/Fock failed time reversal");
    }
}

// No selected-column cache: q and correction are disjoint nao-sized views.
void projector_column(const PeriodicRestrictedMeanFieldState& state,
                      std::size_t k, std::size_t ao, Complex* q, Complex* correction) {
    const auto n = static_cast<std::size_t>(state.n_basis());
    std::fill_n(q, n, Complex{});
    std::fill_n(correction, n, Complex{});
    const auto& c = state.coefficients(k);
    const auto& s = state.overlap(k);
    const auto& mask = state.virtual_mask(k);
    for (std::size_t band = 0; band < mask.size(); ++band) {
        if (mask[band] == 0U) continue;
        ComplexSum overlap;
        for (std::size_t nu = 0; nu < n; ++nu) {
            overlap.add(std::conj(c(nu, band)) * s(nu, ao));
        }
        const auto value = overlap.result();
        for (std::size_t nu = 0; nu < n; ++nu) {
            add_complex(c(nu, band) * value, q[nu], correction[nu]);
        }
    }
    for (std::size_t nu = 0; nu < n; ++nu) {
        q[nu] += correction[nu];
        if (!finite(q[nu])) throw std::overflow_error("PAO domain projector column is non-finite");
    }
}

Complex bilinear(const PeriodicMeanFieldComplexMatrix& m, const Complex* left,
                 const Complex* right, Complex* scratch, std::size_t n) {
    for (std::size_t mu = 0; mu < n; ++mu) {
        ComplexSum row;
        for (std::size_t nu = 0; nu < n; ++nu) row.add(m(mu, nu) * right[nu]);
        scratch[mu] = row.result();
    }
    ComplexSum result;
    for (std::size_t mu = 0; mu < n; ++mu) result.add(std::conj(left[mu]) * scratch[mu]);
    return result.result();
}

// exp(+ik.R) on the exact Gamma torus; no phase table. Shared mathematical
// convention with the Wannier primitive. k/-k use one representative and
// self-inverse characters are exact integer parities, including even meshes.
Complex character(const RegularKMesh& mesh, std::size_t k, std::size_t cell) {
    const auto minus = mesh.negate_index(k);
    const auto canonical = std::min(k, minus);
    const auto m = mesh.address(canonical).doubled;
    const auto r = mesh.address(cell).doubled;
    if (k == minus) {
        std::int64_t parity = 0;
        for (std::size_t d = 0; d < 3U; ++d) if (m[d] != 0) parity += r[d] / 2;
        return Complex(parity % 2 == 0 ? 1.0 : -1.0, 0.0);
    }
    double turns = 0.0;
    for (std::size_t d = 0; d < 3U; ++d) {
        const auto modulus = static_cast<std::int64_t>(mesh.mesh()[d]);
        const auto residue = ((static_cast<std::int64_t>(m[d]) / 2)
            * (static_cast<std::int64_t>(r[d]) / 2)) % modulus;
        turns += static_cast<double>(residue) / static_cast<double>(modulus);
    }
    turns -= std::round(turns);
    const Complex phase(std::cos(kTwoPi * turns), std::sin(kTwoPi * turns));
    return k == canonical ? phase : std::conj(phase);
}

Complex hermitize(Complex ab, Complex ba, bool diagonal, double& maximum_defect,
                  double& maximum_correction, const Options& options) {
    if (!compare(ab, std::conj(ba), options.hermitian_absolute_tolerance,
                 options.hermitian_relative_tolerance, maximum_defect)) {
        throw std::runtime_error("PAO domain raw S/F failed Hermitian audit");
    }
    const auto answer = diagonal ? Complex(ab.real(), 0.0)
        : ab * 0.5 + std::conj(ba) * 0.5;
    maximum_correction = std::max({maximum_correction,
        magnitude(answer - ab), magnitude(std::conj(answer) - ba)});
    return answer;
}

void audit_real(Complex value, double& maximum, const Options& options,
                Diagnostics& diagnostics) {
    const double imaginary = std::abs(value.imag());
    maximum = std::max(maximum, imaginary);
    diagnostics.real_matrices_compatible &= imaginary <= options.real_absolute_tolerance
        + options.real_relative_tolerance * magnitude(value);
}

}  // namespace

PeriodicCorrelationPAODomainMemoryPlan plan_periodic_correlation_pao_domain(
    std::array<int, 3> mesh, std::uint64_t n_basis, std::uint64_t domain_dimension) {
    const RegularKMesh addressing(mesh);
    if (n_basis == 0U) throw std::invalid_argument("PAO domain requires n_basis > 0");
    PeriodicCorrelationPAODomainMemoryPlan result;
    result.n_cells = addressing.size();
    result.n_basis = n_basis;
    result.domain_dimension = domain_dimension;
    result.maximum_unique_domain_columns = product(result.n_cells, n_basis);
    if (domain_dimension > result.maximum_unique_domain_columns) {
        throw std::invalid_argument("PAO domain dimension exceeds the unique finite-torus columns");
    }
    result.matrix_element_count = product(domain_dimension, domain_dimension);
    result.caller_domain_index_bytes = product(domain_dimension, sizeof(PeriodicPAODomainColumn));
    result.retained_domain_index_bytes = result.caller_domain_index_bytes;
    result.retained_matrix_bytes = product(result.matrix_element_count, 2U * sizeof(Complex));
    result.temporary_column_bytes = domain_dimension == 0U ? 0U : product(n_basis, 3U * sizeof(Complex));
    result.peak_owned_numerical_bytes = sum(result.retained_domain_index_bytes,
        sum(result.retained_matrix_bytes, result.temporary_column_bytes));
    constexpr auto maximum_hash_payload = std::numeric_limits<std::uint64_t>::max() / 8U - 4096U;
    if (result.retained_matrix_bytes > maximum_hash_payload
        || result.retained_domain_index_bytes > maximum_hash_payload) {
        throw std::overflow_error("PAO domain payload exceeds SHA-256's bit-length domain");
    }
    const auto pointer_limit = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (result.matrix_element_count > std::vector<Complex>().max_size()
        || domain_dimension > std::vector<PeriodicPAODomainColumn>().max_size()
        || product(n_basis, 3U) > std::vector<Complex>().max_size()
        || result.retained_matrix_bytes > pointer_limit
        || result.retained_domain_index_bytes > pointer_limit
        || product(n_basis, sizeof(Complex)) > pointer_limit) {
        throw std::length_error("PAO domain payload exceeds native vector or address extent");
    }
    return result;
}

PeriodicPAODomainColumn PeriodicCorrelationPAODomain::column(std::size_t index) const {
    if (!state_ || columns_.size() != domain_dimension()) {
        throw std::logic_error("PAO domain column requested from a consumed owner");
    }
    if (index >= domain_dimension()) throw std::out_of_range("PAO domain column index out of range");
    return columns_[index];
}

Complex PeriodicCorrelationPAODomain::overlap(std::size_t row, std::size_t col) const {
    if (!state_ || overlap_.size() != memory_.matrix_element_count) {
        throw std::logic_error("PAO domain overlap requested from a consumed owner");
    }
    if (row >= domain_dimension() || col >= domain_dimension()) {
        throw std::out_of_range("PAO domain matrix index out of range");
    }
    return overlap_[row * domain_dimension() + col];
}

Complex PeriodicCorrelationPAODomain::fock(std::size_t row, std::size_t col) const {
    if (!state_ || fock_.size() != memory_.matrix_element_count) {
        throw std::logic_error("PAO domain Fock requested from a consumed owner");
    }
    if (row >= domain_dimension() || col >= domain_dimension()) {
        throw std::out_of_range("PAO domain matrix index out of range");
    }
    return fock_[row * domain_dimension() + col];
}

PeriodicCorrelationPAODomain make_periodic_correlation_pao_domain(
    const PeriodicCorrelationAdmittedReference& reference,
    const std::uint64_t* cell_ao_indices, std::size_t accessible_index_count,
    std::size_t domain_dimension, std::uint64_t owned_numerical_byte_cap,
    const Options& options) {
    require_reference(reference);
    require_options(options);
    const auto& state = reference.state();
    const auto memory = plan_periodic_correlation_pao_domain(
        state.mesh(), state.n_basis(), domain_dimension);
    if (owned_numerical_byte_cap < memory.peak_owned_numerical_bytes) {
        throw std::length_error("PAO domain numerical byte cap is missing or exceeded");
    }
    if (accessible_index_count < product(domain_dimension, 2U)
        || (domain_dimension != 0U && (cell_ao_indices == nullptr
            || reinterpret_cast<std::uintptr_t>(cell_ao_indices) % alignof(std::uint64_t) != 0U))) {
        throw std::invalid_argument("PAO domain input view has an invalid pointer, alignment or extent");
    }
    if (domain_dimension != 0U && memory.caller_domain_index_bytes
        > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(cell_ao_indices)) {
        throw std::overflow_error("PAO domain input view exceeds native pointer range");
    }
    const auto& dims = reference.dimensions();
    const auto& budget = reference.budget();
    const auto required_node = sum(sum(dims.external_bytes, dims.shared_bytes),
        sum(product(budget.mpi_ranks, sum(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            product(product(budget.mpi_ranks, budget.workers_per_rank),
                sum(memory.peak_owned_numerical_bytes, memory.caller_domain_index_bytes))));
    if (budget.memory_limit_bytes == 0U || required_node > budget.memory_limit_bytes) {
        throw std::length_error("PAO domain peak and caller inventory exceed admitted node memory");
    }
    const auto column_at = [cell_ao_indices](std::size_t index) {
        return PeriodicPAODomainColumn{cell_ao_indices[2U * index], cell_ao_indices[2U * index + 1U]};
    };
    for (std::size_t a = 0; a < domain_dimension; ++a) {
        const auto column = column_at(a);
        if (column.cell >= memory.n_cells || column.ao >= memory.n_basis) {
            throw std::invalid_argument("PAO domain column is outside canonical modular cell/AO extents");
        }
        for (std::size_t b = 0; b < a; ++b) {
            const auto previous = column_at(b);
            if (column.cell == previous.cell && column.ao == previous.ao) {
                throw std::invalid_argument("PAO domain contains a duplicate modular cell/AO column");
            }
        }
    }
    const RegularKMesh mesh(state.mesh());
    Diagnostics diagnostics;
    diagnostics.required_node_memory_bytes = required_node;
    audit_state(state, mesh, options, diagnostics);
    PeriodicCorrelationPAODomain result;
    result.state_ = reference.state_handle();
    result.memory_ = memory;
    result.options_ = options;
    result.allocation_identity_ = dims.allocation_identity;
    if (domain_dimension != 0U) {
        result.columns_.resize(domain_dimension);
        for (std::size_t a = 0; a < domain_dimension; ++a) result.columns_[a] = column_at(a);
        const auto* columns = result.columns_.data();
        result.overlap_.resize(static_cast<std::size_t>(memory.matrix_element_count));
        result.fock_.resize(static_cast<std::size_t>(memory.matrix_element_count));
        const auto n = static_cast<std::size_t>(state.n_basis());
        // Exactly 3*nao complex entries. The third column is compensation
        // during projection and matrix-vector storage during contraction.
        std::vector<Complex> workspace(3U * n);
        auto* qa = workspace.data();
        auto* qb = qa + n;
        auto* scratch = qb + n;
        for (std::size_t k = 0; k < state.n_kpoints(); ++k) {
            const auto minus_k = mesh.negate_index(k);
            if (minus_k < k) continue;
            for (std::size_t a = 0; a < domain_dimension; ++a) {
                // Repeated AOs in different cells do not need repeated Q audits.
                bool already = false;
                for (std::size_t b = 0; b < a; ++b) already |= columns[b].ao == columns[a].ao;
                if (already) continue;
                projector_column(state, k, columns[a].ao, qa, scratch);
                projector_column(state, minus_k, columns[a].ao, qb, scratch);
                for (std::size_t mu = 0; mu < n; ++mu) {
                    diagnostics.time_reversal_compatible &= compare(qa[mu], std::conj(qb[mu]),
                        options.time_reversal_absolute_tolerance, options.time_reversal_relative_tolerance,
                        diagnostics.maximum_selected_projector_time_reversal_residual);
                }
            }
        }
        if (options.require_time_reversal && !diagnostics.time_reversal_compatible) {
            throw std::invalid_argument("PAO domain selected virtual projector failed time reversal");
        }
        const double weight = state.uniform_weight();
        for (std::size_t a = 0; a < domain_dimension; ++a) {
            for (std::size_t b = a; b < domain_dimension; ++b) {
                ComplexSum sab, sba, fab, fba;
                // transfer_index(left,right) is right-left; these indices
                // name real-space cells, so this is precisely R_a-R_b.
                const auto difference = mesh.transfer_index(columns[b].cell, columns[a].cell);
                for (std::size_t k = 0; k < state.n_kpoints(); ++k) {
                    projector_column(state, k, columns[a].ao, qa, scratch);
                    projector_column(state, k, columns[b].ao, qb, scratch);
                    const auto phase = character(mesh, k, difference);
                    const auto skab = bilinear(state.overlap(k), qa, qb, scratch, n);
                    const auto fkab = bilinear(state.fock(k), qa, qb, scratch, n);
                    sab.add(phase * (skab * weight));
                    fab.add(phase * (fkab * weight));
                    if (a != b) {
                        sba.add(std::conj(phase) * (bilinear(state.overlap(k), qb, qa, scratch, n) * weight));
                        fba.add(std::conj(phase) * (bilinear(state.fock(k), qb, qa, scratch, n) * weight));
                    }
                }
                const auto s = hermitize(sab.result(), a == b ? sab.result() : sba.result(), a == b,
                    diagnostics.maximum_raw_overlap_hermitian_defect,
                    diagnostics.maximum_overlap_hermitization_correction, options);
                const auto f = hermitize(fab.result(), a == b ? fab.result() : fba.result(), a == b,
                    diagnostics.maximum_raw_fock_hermitian_defect,
                    diagnostics.maximum_fock_hermitization_correction, options);
                result.overlap_[a * domain_dimension + b] = s;
                result.overlap_[b * domain_dimension + a] = std::conj(s);
                result.fock_[a * domain_dimension + b] = f;
                result.fock_[b * domain_dimension + a] = std::conj(f);
                audit_real(s, diagnostics.maximum_overlap_imaginary_magnitude, options, diagnostics);
                audit_real(f, diagnostics.maximum_fock_imaginary_magnitude, options, diagnostics);
            }
        }
    }
    if (options.require_real_matrices && !diagnostics.real_matrices_compatible) {
        throw std::invalid_argument("PAO domain matrices failed the real tolerance");
    }
    result.diagnostics_ = diagnostics;
    // Fixed big-endian wire, with signed-zero canonicalization. The domain
    // digest binds mesh, nao, D and ordered (cell,ao) pairs. The matrix digest
    // binds D followed by row-major S then F. Identity binds state/options.
    PAODigest domain_digest("vibeqc.periodic.correlation.pao.domain");
    for (int value : state.mesh()) domain_digest.u32(value);
    domain_digest.u64(state.n_basis()); domain_digest.u64(domain_dimension);
    for (const auto column : result.columns_) {
        domain_digest.u64(column.cell); domain_digest.u64(column.ao);
    }
    result.domain_digest_ = domain_digest.finish();
    PAODigest matrix_digest("vibeqc.periodic.correlation.pao.matrices");
    matrix_digest.u64(domain_dimension);
    for (const auto value : result.overlap_) matrix_digest.complex(value);
    for (const auto value : result.fock_) matrix_digest.complex(value);
    result.matrix_digest_ = matrix_digest.finish();
    PAODigest identity("vibeqc.periodic.correlation.pao.identity");
    identity.u32(state.digest_version());
    identity.string(state.state_identity_sha256());
    identity.string(dims.calculation_identity); identity.string(dims.allocation_identity);
    identity.string(result.domain_digest_); identity.string(result.matrix_digest_);
    identity.string("retained-virtual-projector;no-spin2;plus-k-Ra-minus-Rb;uniform-full-gamma");
    identity.real(options.hermitian_absolute_tolerance); identity.real(options.hermitian_relative_tolerance);
    identity.real(options.time_reversal_absolute_tolerance); identity.real(options.time_reversal_relative_tolerance);
    identity.real(options.real_absolute_tolerance); identity.real(options.real_relative_tolerance);
    identity.u32(options.require_time_reversal ? 1U : 0U);
    identity.u32(options.require_real_matrices ? 1U : 0U);
    result.identity_digest_ = identity.finish();
    return result;
}

}  // namespace vibeqc
