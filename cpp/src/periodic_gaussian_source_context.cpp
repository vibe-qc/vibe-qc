#include "vibeqc/periodic_gaussian_source_context.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/cart_to_sph_data.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "vibeqc/periodic_correlation_metric_factorization.hpp"
#include "vibeqc/periodic_correlation_three_center.hpp"

namespace vibeqc {
namespace {

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Gaussian source contexts require IEEE binary64");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t), "address count width");
static_assert(kAuxiliaryBasisContentDigestVersion == 1U,
              "update source-context basis-wire census for a new digest version");
constexpr double kTwoPi = 6.283185307179586476925286766559005768;
constexpr std::uint64_t kBasisPrefixBytes = 75U;
constexpr std::uint64_t kMaximumShaBytes =
    std::numeric_limits<std::uint64_t>::max() / 8U;

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("Gaussian source-context count overflow");
    }
    return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("Gaussian source-context count overflow");
    }
    return a * b;
}
void limit(std::uint64_t n, std::uint64_t cap, const char* what) {
    if (n > cap) throw std::length_error(what);
}

void require_caps(const PeriodicGaussianSourceCaps& c, std::uint64_t nk) {
    if (c.maximum_context_storage_bytes == 0 || c.maximum_kpoint_count == 0
        || c.maximum_shell_count == 0 || c.maximum_contraction_count == 0
        || c.maximum_primitive_numeric_lanes == 0
        || c.maximum_basis_content_wire_bytes == 0
        || c.maximum_borrowed_active_numeric_bytes == 0 || c.maximum_work_units == 0) {
        throw std::invalid_argument("Gaussian source-context caps must all be positive");
    }
    limit(sizeof(PeriodicGaussianSourceContext), c.maximum_context_storage_bytes,
          "Gaussian source-context fixed storage exceeds cap");
    limit(nk, c.maximum_kpoint_count, "Gaussian source-context kpoint count exceeds cap");
}

struct Counts {
    std::uint64_t shells = 0, contractions = 0, primitives = 0;
    std::uint64_t wire = 2U * kBasisPrefixBytes, borrowed = 0;
    std::uint64_t work() const {
        return add(add(mul(128U, wire), mul(64U, add(shells, contractions))), 4096U);
    }
    void check(const PeriodicGaussianSourceCaps& caps) const {
        limit(shells, caps.maximum_shell_count, "Gaussian source-context shell count exceeds cap");
        limit(contractions, caps.maximum_contraction_count,
              "Gaussian source-context contraction count exceeds cap");
        limit(primitives, caps.maximum_primitive_numeric_lanes,
              "Gaussian source-context primitive lanes exceed cap");
        limit(wire, std::min(caps.maximum_basis_content_wire_bytes, kMaximumShaBytes),
              "Gaussian source-context basis wire exceeds cap or SHA extent");
        limit(borrowed, caps.maximum_borrowed_active_numeric_bytes,
              "Gaussian source-context borrowed numeric payload exceeds cap");
        limit(work(), caps.maximum_work_units, "Gaussian source-context scan work exceeds cap");
    }
};

PeriodicGaussianBasisInventory inspect_basis(
    const BasisSet& basis, Counts& total, const PeriodicGaussianSourceCaps& caps) {
    PeriodicGaussianBasisInventory result;
    result.content_wire_bytes = kBasisPrefixBytes;
    if (basis.libint().empty()) throw std::invalid_argument("Gaussian source context requires nonempty bases");
    // Cheap extent check before traversing a potentially large shell vector.
    limit(add(total.shells, basis.nshells()), caps.maximum_shell_count,
          "Gaussian source-context shell count exceeds cap");
    for (const auto& shell : basis.libint()) {
        total.shells = add(total.shells, 1U);
        result.shell_count = add(result.shell_count, 1U);
        const auto np = static_cast<std::uint64_t>(shell.alpha.size());
        if (np == 0 || shell.contr.empty()) {
            throw std::invalid_argument("Gaussian source context has an empty shell or contraction");
        }
        total.primitives = add(total.primitives, np);
        total.borrowed = add(total.borrowed, mul(8U, add(3U, np)));
        total.wire = add(total.wire, 8U);
        result.exponent_count = add(result.exponent_count, np);
        result.content_wire_bytes = add(result.content_wire_bytes, 8U);
        total.check(caps);
        limit(add(total.contractions, shell.contr.size()), caps.maximum_contraction_count,
              "Gaussian source-context contraction count exceeds cap");
        for (const auto& contraction : shell.contr) {
            // Reject unsupported angular conventions before any value/hash scan.
            if (contraction.l < 0 || contraction.l > cart_to_sph_data::kMaxL
                || (!contraction.pure && contraction.l != 0)) {
                throw std::invalid_argument("Gaussian source context supports spherical L<=6 and Cartesian s only");
            }
            if (contraction.coeff.size() != shell.alpha.size()) {
                throw std::invalid_argument("Gaussian source-context exponent/coefficient extents differ");
            }
            total.contractions = add(total.contractions, 1U);
            total.primitives = add(total.primitives, np);
            total.borrowed = add(total.borrowed, mul(8U, np));
            const auto record_wire = add(45U, mul(16U, np));
            total.wire = add(total.wire, record_wire);
            result.contraction_count = add(result.contraction_count, 1U);
            result.coefficient_count = add(result.coefficient_count, np);
            result.content_wire_bytes = add(result.content_wire_bytes, record_wire);
            result.function_count = add(result.function_count,
                static_cast<std::uint64_t>(2 * contraction.l + 1));
            total.check(caps);
        }
    }
    if (result.function_count != basis.nbasis()) {
        throw std::invalid_argument("Gaussian source-context shell/function count mismatch");
    }
    result.borrowed_active_numeric_bytes = mul(8U,
        add(mul(3U, result.shell_count), add(result.exponent_count, result.coefficient_count)));
    return result;
}

PeriodicGaussianSourceInventory census_basis_inputs(
    const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicGaussianSourceCaps& caps, std::uint64_t nk) {
    require_caps(caps, nk);
    Counts counts;
    counts.check(caps);
    PeriodicGaussianSourceInventory result;
    result.fixed_context_storage_bytes = sizeof(PeriodicGaussianSourceContext);
    result.ao = inspect_basis(ao, counts, caps);
    result.auxiliary = inspect_basis(auxiliary, counts, caps);
    result.combined_borrowed_active_numeric_bytes = counts.borrowed;
    result.work_units_upper_bound = counts.work();
    return result;
}

std::array<char, 64> digest_array(const std::string& digest) {
    if (digest.size() != 64U || !std::all_of(digest.begin(), digest.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) throw std::logic_error("Gaussian source context received an invalid native digest");
    std::array<char, 64> result{};
    std::copy(digest.begin(), digest.end(), result.begin());
    return result;
}

class Digest {
public:
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> b{};
        for (unsigned i = 0; i != 4; ++i) b[i] = value >> (24U - 8U * i);
        hash_.update(b.data(), b.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i != 8; ++i) b[i] = value >> (56U - 8U * i);
        hash_.update(b.data(), b.size());
    }
    void bytes(const char* p, std::size_t n) {
        u64(n);
        hash_.update(reinterpret_cast<const std::uint8_t*>(p), n);
    }
    template<std::size_t N> void text(const char (&s)[N]) { bytes(s, N - 1U); }
    void identity(const std::array<char, 64>& s) { bytes(s.data(), s.size()); }
    void real(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("Gaussian source context nonfinite identity lane");
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, 8U);
        u64(bits);
    }
    std::array<char, 64> finish() { return digest_array(hash_.finish_hex()); }
private:
    detail::Sha256 hash_;
};

void require_options(const PeriodicGaussianSourceOptions& o) {
    if (!std::isfinite(o.reciprocal_energy_cutoff) || o.reciprocal_energy_cutoff <= 0.0
        || !std::isfinite(o.ao_pair_image_cutoff_bohr) || o.ao_pair_image_cutoff_bohr <= 0.0
        || !std::isfinite(o.metric_absolute_eigenvalue_threshold)
        || o.metric_absolute_eigenvalue_threshold <= 0.0
        || !std::isfinite(o.metric_negative_tolerance) || o.metric_negative_tolerance <= 0.0
        || o.metric_negative_tolerance > o.metric_absolute_eigenvalue_threshold) {
        throw std::invalid_argument("Gaussian source context requires positive finite cutoffs and negative tolerance <= rank cutoff");
    }
    if (!std::isfinite(2.0 * o.reciprocal_energy_cutoff)
        || !std::isfinite(o.ao_pair_image_cutoff_bohr * o.ao_pair_image_cutoff_bohr)) {
        throw std::overflow_error("Gaussian source-context squared cutoff is not finite");
    }
}

void audit_geometry(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    // Long-double accumulation is diagnostic only. The sealed B remains the
    // existing native derivation; this check never repairs/averages geometry.
    for (int i = 0; i != 3; ++i) {
        for (int j = 0; j != 3; ++j) {
            long double sum = 0.0L, scale = kTwoPi;
            for (int d = 0; d != 3; ++d) {
                const long double product = static_cast<long double>(a(d, i)) * b(d, j);
                sum += product;
                scale += std::abs(product);
            }
            const long double expected = i == j ? kTwoPi : 0.0L;
            const long double tolerance = 256.0L * std::numeric_limits<double>::epsilon() * scale;
            if (!std::isfinite(sum) || !std::isfinite(scale)
                || std::abs(sum - expected) > tolerance) {
                throw std::invalid_argument("Gaussian source-context direct/reciprocal duality audit failed");
            }
        }
    }
}

std::array<double, 3> cartesian(
    const Eigen::Matrix3d& b, const std::array<double, 3>& f) {
    std::array<double, 3> result{};
    for (int r = 0; r != 3; ++r) {
        result[r] = std::fma(b(r, 0), f[0],
            std::fma(b(r, 1), f[1], std::fma(b(r, 2), f[2], 0.0)));
        if (!std::isfinite(result[r])) throw std::overflow_error("Gaussian source-context Cartesian momentum is not finite");
        if (result[r] == 0.0) result[r] = 0.0;
    }
    return result;
}

} // namespace

PeriodicGaussianSourceContext make_periodic_gaussian_source_context(
    const PeriodicSystem& system, const BasisSet& ao, const BasisSet& auxiliary,
    const RegularKMesh& mesh, const PeriodicGaussianSourceOptions& options,
    const PeriodicGaussianSourceCaps& caps) {
    require_caps(caps, mesh.size());
    require_options(options);
    if (system.dim != 3 || mesh.is_shift() != std::array<int, 3>{0, 0, 0}) {
        throw std::invalid_argument("Gaussian source context requires 3D Gamma-centered mesh");
    }
    if (std::fegetround() != FE_TONEAREST) {
        throw std::invalid_argument("Gaussian source context requires round-to-nearest floating point");
    }
    if (!system.lattice.allFinite()) {
        throw std::invalid_argument("Gaussian source context requires a finite direct lattice");
    }
    PeriodicGaussianSourceContext result;
    result.inventory_ = census_basis_inputs(ao, auxiliary, caps, mesh.size());
    result.mesh_ = mesh;
    result.options_ = options;
    result.direct_ = system.lattice;
    result.reciprocal_ = system.reciprocal_lattice();
    if (!result.reciprocal_.allFinite()) {
        throw std::invalid_argument("Gaussian source context requires a finite reciprocal lattice");
    }
    for (int row = 0; row != 3; ++row) {
        for (int col = 0; col != 3; ++col) {
            if (result.direct_(row, col) == 0.0) result.direct_(row, col) = 0.0;
            if (result.reciprocal_(row, col) == 0.0) result.reciprocal_(row, col) = 0.0;
        }
    }
    audit_geometry(result.direct_, result.reciprocal_);
    // Both complete metadata censuses and all caps precede these full scans.
    result.ao_digest_ = digest_array(auxiliary_basis_content_identity_sha256(ao));
    result.auxiliary_digest_ = digest_array(auxiliary_basis_content_identity_sha256(auxiliary));
    result.producer_digest_ = digest_array(periodic_correlation_reciprocal_metric_producer_identity_sha256());
    result.backend_digest_ = digest_array(periodic_correlation_metric_factorization_backend_identity_sha256());
    Digest digest;
    digest.text("vibeqc.periodic.gaussian-source-context");
    digest.u32(kPeriodicGaussianSourceContextVersion);
    digest.u32(3U);
    digest.text(kPeriodicGaussianSourceGeometryPolicy);
    digest.text(kPeriodicGaussianSourceHamiltonianPolicy);
    digest.text(kPeriodicCorrelationThreeCenterImagePolicy);
    digest.text(kPeriodicGaussianSourceReciprocalPolicy);
    digest.u32(kPeriodicCorrelationReciprocalMetricSourceContractVersion);
    digest.u32(kAuxiliaryBasisContentDigestVersion);
    digest.identity(result.ao_digest_);
    digest.identity(result.auxiliary_digest_);
    digest.identity(result.producer_digest_);
    digest.identity(result.backend_digest_);
    for (int value : mesh.mesh()) digest.u32(static_cast<std::uint32_t>(value));
    for (int value : mesh.is_shift()) digest.u32(static_cast<std::uint32_t>(value));
    for (const auto* matrix : {&result.direct_, &result.reciprocal_}) {
        for (int row = 0; row != 3; ++row) {
            for (int col = 0; col != 3; ++col) digest.real((*matrix)(row, col));
        }
    }
    digest.real(options.reciprocal_energy_cutoff);
    digest.real(options.ao_pair_image_cutoff_bohr);
    digest.real(options.metric_absolute_eigenvalue_threshold);
    digest.real(options.metric_negative_tolerance);
    digest.u64(result.inventory_.ao.function_count);
    digest.u64(result.inventory_.auxiliary.function_count);
    result.context_digest_ = digest.finish();
    return result;
}

std::string PeriodicGaussianSourceContext::ao_basis_identity_sha256() const {
    return {ao_digest_.begin(), ao_digest_.end()};
}
std::string PeriodicGaussianSourceContext::auxiliary_basis_identity_sha256() const {
    return {auxiliary_digest_.begin(), auxiliary_digest_.end()};
}
std::string PeriodicGaussianSourceContext::reciprocal_producer_identity_sha256() const {
    return {producer_digest_.begin(), producer_digest_.end()};
}
std::string PeriodicGaussianSourceContext::factorization_backend_identity_sha256() const {
    return {backend_digest_.begin(), backend_digest_.end()};
}
std::string PeriodicGaussianSourceContext::source_context_identity_sha256() const {
    return {context_digest_.begin(), context_digest_.end()};
}

PeriodicGaussianKRecord PeriodicGaussianSourceContext::k_record(std::uint64_t index) const {
    if (index >= mesh_.size()) throw std::out_of_range("Gaussian source-context k index");
    PeriodicGaussianKRecord result;
    result.index = index;
    const auto address = mesh_.address(static_cast<std::size_t>(index));
    result.modular_doubled_address = address.doubled;
    const auto fractional = mesh_.fractional(address);
    for (int d = 0; d != 3; ++d) result.fractional[d] = fractional[d];
    result.cartesian = cartesian(reciprocal_, result.fractional);
    return result;
}

PeriodicGaussianTransferRecord PeriodicGaussianSourceContext::transfer_record(std::uint64_t index) const {
    if (index >= mesh_.size()) throw std::out_of_range("Gaussian source-context q index");
    PeriodicGaussianTransferRecord result;
    result.index = index;
    const auto address = mesh_.transfer_address(static_cast<std::size_t>(index));
    result.modular_doubled_address = address.doubled;
    result.gamma = index == 0U;
    result.self_conjugate = mesh_.negate(address) == address;
    for (int d = 0; d != 3; ++d) {
        const bool wrap = address.doubled[d] >= mesh_.mesh()[d];
        result.centered_reciprocal_wrap[d] = wrap ? 1 : 0;
        result.centered_doubled_numerator[d] = address.doubled[d]
            - (wrap ? mesh_.doubled_modulus()[d] : 0);
        result.fractional[d] = static_cast<double>(result.centered_doubled_numerator[d])
            / static_cast<double>(mesh_.doubled_modulus()[d]);
    }
    result.cartesian = cartesian(reciprocal_, result.fractional);
    return result;
}

std::uint64_t PeriodicGaussianSourceContext::ket_index(
    std::uint64_t bra, std::uint64_t q) const {
    if (bra >= mesh_.size() || q >= mesh_.size()) throw std::out_of_range("Gaussian source-context ket input index");
    return mesh_.add_transfer_index(static_cast<std::size_t>(bra), static_cast<std::size_t>(q));
}

PeriodicGaussianSourceInventory PeriodicGaussianSourceContext::verify_bases(
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicGaussianSourceCaps& caps) const {
    const auto result = census_basis_inputs(ao, auxiliary, caps, mesh_.size());
    if (digest_array(auxiliary_basis_content_identity_sha256(ao)) != ao_digest_
        || digest_array(auxiliary_basis_content_identity_sha256(auxiliary)) != auxiliary_digest_) {
        throw std::invalid_argument("Gaussian source-context borrowed basis content mismatch");
    }
    return result;
}

} // namespace vibeqc
