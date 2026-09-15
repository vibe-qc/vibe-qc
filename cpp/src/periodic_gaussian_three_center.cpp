#include "vibeqc/periodic_gaussian_three_center.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/detail/periodic_gaussian_three_center_numeric.hpp"
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {
using Complex = std::complex<double>;
static_assert(sizeof(double) == 8 && sizeof(Complex) == 16,
              "Gaussian tile inventory requires binary64/complex128 storage");
constexpr char kPayloadDomain[] = "vibeqc.periodic.gaussian-three-center.payload";
constexpr char kWhitenerPayloadDomain[] = "vibeqc.periodic.gaussian-whitener.payload";
constexpr std::uint64_t kShaMax = std::numeric_limits<std::uint64_t>::max() / 8U;
// Four SHA strings; two policy strings; cutoff; A; bra/ket full records;
// descriptor; image/retained/reciprocal/output counts. No variable arrays.
constexpr std::uint64_t kPayloadPrefix = 648U + sizeof(kPayloadDomain) - 1U
    + sizeof(kPeriodicGaussianThreeCenterImagePolicy) - 1U
    + sizeof(kPeriodicGaussianThreeCenterLatticePolicy) - 1U;
constexpr std::uint64_t kWhitenerPayloadPrefix = 228U + sizeof(kWhitenerPayloadDomain) - 1U;

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("Gaussian tile count overflow");
    return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("Gaussian tile count overflow");
    return a * b;
}
void limit(std::uint64_t n, std::uint64_t cap, const char* message) {
    if (n > cap) throw std::length_error(message);
}
std::array<char, 64> ascii(const std::string& s) {
    if (s.size() != 64 || !std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    })) throw std::invalid_argument("Gaussian tile requires native lowercase SHA identities");
    std::array<char, 64> a{};
    std::copy(s.begin(), s.end(), a.begin());
    return a;
}
class Digest {
public:
    void bytes(const std::uint8_t* p, std::size_t n) {
        if (n > kShaMax - extent_) throw std::length_error("Gaussian tile SHA extent overflow");
        hash_.update(p, n); extent_ += n;
    }
    void u32(std::uint32_t n) {
        std::array<std::uint8_t, 4> b{};
        for (unsigned i = 0; i != 4; ++i) b[i] = n >> (24U - 8U * i);
        bytes(b.data(), b.size());
    }
    void u64(std::uint64_t n) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i != 8; ++i) b[i] = n >> (56U - 8U * i);
        bytes(b.data(), b.size());
    }
    void real(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("Gaussian tile payload must be finite");
        if (x == 0.0) x = 0.0;
        std::uint64_t n; std::memcpy(&n, &x, 8U); u64(n);
    }
    void text(const std::string& s) { u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
    std::uint64_t extent() const noexcept { return extent_; }
    std::array<char, 64> finish() { return ascii(hash_.finish_hex()); }
private:
    detail::Sha256 hash_;
    std::uint64_t extent_ = 0;
};
void descriptor(Digest& h, const PeriodicGaussianThreeCenterDescriptor& d) {
    h.u64(d.q_index); h.u64(d.conjugate_q_index); h.u64(d.k_bra_index); h.u64(d.k_ket_index);
    for (int x : d.k_ket_reciprocal_wrap) h.u32(static_cast<std::uint32_t>(x));
    for (auto n : {d.ao_pair_begin, d.ao_pair_count, d.auxiliary_begin, d.auxiliary_count, d.element_count}) h.u64(n);
}
void k_record(Digest& h, const PeriodicGaussianKRecord& k) {
    h.u64(k.index);
    for (int x : k.modular_doubled_address) h.u32(static_cast<std::uint32_t>(x));
    for (double x : k.fractional) h.real(x);
    for (double x : k.cartesian) h.real(x);
}
void resource_caps(Digest& h, const PeriodicGaussianMetricCaps& c) {
    for (auto n : {c.maximum_owned_numeric_bytes, c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes, c.maximum_candidate_evaluations, c.maximum_work_units}) h.u64(n);
}
void hash_plan(Digest& h, const PeriodicGaussianThreeCenterPlan& p) {
    descriptor(h, p.descriptor);
    for (auto n : {p.n_basis, p.n_auxiliary, p.accepted_vector_count, p.reciprocal_candidate_count,
        p.reciprocal_capacity, p.reciprocal_panel_count, p.resident_whitener_bytes, p.raw_panel_bytes,
        p.compensation_bytes, p.reciprocal_panel_bytes, p.double_auxiliary_fourier_bytes, p.ao_fourier_bytes,
        p.fixed_fourier_numeric_workspace_bytes, p.output_bytes, p.assembly_owned_numeric_peak_bytes,
        p.whitening_owned_numeric_peak_bytes, p.owned_numeric_peak_bytes, p.borrowed_basis_active_numeric_bytes,
        p.fixed_inventoried_object_bytes, p.per_replica_inventoried_bytes, p.node_inventoried_bytes,
        p.image_candidate_count, p.retained_pair_image_count, p.reciprocal_candidate_evaluations,
        p.image_candidate_evaluations, p.primitive_pair_evaluations_upper_bound, p.raw_contraction_term_count,
        p.whitening_term_count, p.preflight_work_units_upper_bound, p.work_units_upper_bound}) h.u64(n);
    h.u64(p.config.reciprocal_block);
    const auto& b = p.config.basis_verification_caps;
    for (auto n : {b.maximum_context_storage_bytes, b.maximum_kpoint_count, b.maximum_shell_count,
        b.maximum_contraction_count, b.maximum_primitive_numeric_lanes, b.maximum_basis_content_wire_bytes,
        b.maximum_borrowed_active_numeric_bytes, b.maximum_work_units}) h.u64(n);
    for (auto n : {p.live.replicas_per_node, p.live.other_retained_bytes_per_replica,
        p.live.other_transient_bytes_per_replica, p.live.fixed_backend_margin_bytes_per_replica,
        p.live.external_node_bytes}) h.u64(n);
    h.u64(p.caps.maximum_image_candidates); resource_caps(h, p.caps.resources);
}
void verify_whitener_payload(const PeriodicGaussianMetricWhitener& w) {
    Digest h;
    h.text(kWhitenerPayloadDomain); h.u32(1U);
    h.text(w.input_payload_identity_sha256()); h.u64(w.q_index()); h.u64(w.conjugate_q_index());
    const auto& d = w.diagnostics();
    for (auto n : {d.n_auxiliary, d.retained_rank, d.sweeps, d.rotations}) h.u64(n);
    for (double x : {d.rank_cutoff, d.negative_tolerance, d.minimum_eigenvalue, d.maximum_eigenvalue,
        d.smallest_retained_eigenvalue, d.largest_discarded_eigenvalue, d.input_scale,
        d.scaled_initial_frobenius_norm, d.scaled_final_offdiagonal_norm, d.orthogonality_frobenius_error,
        d.maximum_self_conjugate_imaginary_residual}) h.real(x);
    h.u64(w.matrix_row_major().size());
    if (h.extent() != kWhitenerPayloadPrefix) throw std::logic_error("Gaussian tile whitener prefix wire changed");
    for (const auto z : w.matrix_row_major()) { h.real(z.real()); h.real(z.imag()); }
    if (h.finish() != ascii(w.payload_identity_sha256())) throw std::invalid_argument("Gaussian tile whitener payload mismatch");
}
std::uint64_t basis_scan(const PeriodicGaussianBasisInventory& b) {
    return add(b.borrowed_active_numeric_bytes / 8U, add(b.shell_count, b.contraction_count));
}
PeriodicGaussianSourceCaps bounded_basis_caps(
    const PeriodicGaussianSourceContext& c, const PeriodicGaussianSourceCaps& supplied) {
    const auto& a = c.inventory().ao;
    const auto& x = c.inventory().auxiliary;
    PeriodicGaussianSourceCaps b;
    b.maximum_context_storage_bytes = sizeof(PeriodicGaussianSourceContext);
    b.maximum_kpoint_count = c.mesh().size();
    b.maximum_shell_count = add(a.shell_count, x.shell_count);
    b.maximum_contraction_count = add(a.contraction_count, x.contraction_count);
    b.maximum_primitive_numeric_lanes = add(add(a.exponent_count, a.coefficient_count),
        add(x.exponent_count, x.coefficient_count));
    b.maximum_basis_content_wire_bytes = add(a.content_wire_bytes, x.content_wire_bytes);
    b.maximum_borrowed_active_numeric_bytes = c.inventory().combined_borrowed_active_numeric_bytes;
    b.maximum_work_units = c.inventory().work_units_upper_bound;
    const std::array<std::uint64_t, 8> expected{{b.maximum_context_storage_bytes, b.maximum_kpoint_count,
        b.maximum_shell_count, b.maximum_contraction_count, b.maximum_primitive_numeric_lanes,
        b.maximum_basis_content_wire_bytes, b.maximum_borrowed_active_numeric_bytes, b.maximum_work_units}};
    const std::array<std::uint64_t, 8> caps{{supplied.maximum_context_storage_bytes, supplied.maximum_kpoint_count,
        supplied.maximum_shell_count, supplied.maximum_contraction_count, supplied.maximum_primitive_numeric_lanes,
        supplied.maximum_basis_content_wire_bytes, supplied.maximum_borrowed_active_numeric_bytes, supplied.maximum_work_units}};
    for (std::size_t i = 0; i != caps.size(); ++i) {
        if (caps[i] == 0) throw std::invalid_argument("Gaussian tile basis verification caps must be positive");
        limit(expected[i], caps[i], "Gaussian tile basis verification exceeds cap");
    }
    return b;
}
Eigen::Vector3d ket_cartesian(const PeriodicGaussianSourceContext& c, std::uint64_t ket) {
    const auto r = c.k_record(ket);
    return {r.cartesian[0], r.cartesian[1], r.cartesian[2]};
}
PeriodicSystem original_system(const PeriodicGaussianSourceContext& c) {
    PeriodicSystem s; s.dim = 3; s.lattice = c.direct_lattice();
    return s;
}

PeriodicGaussianThreeCenterPlan basic_plan(
    const PeriodicGaussianReciprocalSource& source, const PeriodicGaussianMetricWhitener& w,
    const PeriodicGaussianThreeCenterSelection& selection, const PeriodicGaussianThreeCenterConfig& config,
    const PeriodicGaussianMetricLiveInventory& live, const PeriodicGaussianThreeCenterCaps& caps) {
    const auto& r = caps.resources;
    if (caps.maximum_image_candidates == 0 || r.maximum_owned_numeric_bytes == 0
        || r.maximum_per_replica_inventoried_bytes == 0 || r.maximum_node_inventoried_bytes == 0
        || r.maximum_candidate_evaluations == 0 || r.maximum_work_units == 0
        || live.replicas_per_node == 0 || live.fixed_backend_margin_bytes_per_replica == 0) {
        throw std::invalid_argument("Gaussian tile caps, replicas and backend margin must be positive");
    }
    if (!source.context_handle() || !w.context_handle()
        || source.context_handle().get() != w.context_handle().get()
        || source.q_index() != w.q_index() || source.conjugate_q_index() != w.conjugate_q_index()
        || source.source_identity_sha256() != w.source_identity_sha256()
        || source.conjugate_source_identity_sha256() != w.conjugate_source_identity_sha256()
        || !source.reciprocal_conjugate_closure_certified()) {
        throw std::invalid_argument("Gaussian tile context/source/whitener provenance mismatch");
    }
    const auto& c = *source.context_handle();
    const auto n = c.inventory().ao.function_count, a = c.inventory().auxiliary.function_count;
    const auto square = mul(n, n), w_square = mul(a, a), count = source.accepted_vector_count();
    if (n == 0 || a == 0 || count == 0 || config.reciprocal_block == 0
        || selection.k_bra_index >= c.mesh().size() || selection.ao_pair_count == 0
        || selection.ao_pair_begin > square || selection.ao_pair_count > square - selection.ao_pair_begin
        || selection.auxiliary_count == 0 || selection.auxiliary_begin > a
        || selection.auxiliary_count > a - selection.auxiliary_begin
        || w.diagnostics().n_auxiliary != a || w.diagnostics().retained_rank == 0
        || w.diagnostics().retained_rank > a || w.matrix_row_major().size() != w_square) {
        throw std::invalid_argument("Gaussian tile selection or native whitener shape is invalid");
    }
    if (w.diagnostics().rank_cutoff != c.options().metric_absolute_eigenvalue_threshold
        || w.diagnostics().negative_tolerance != c.options().metric_negative_tolerance
        || c.factorization_backend_identity_sha256() != periodic_correlation_metric_factorization_backend_identity_sha256()) {
        throw std::invalid_argument("Gaussian tile whitener scientific backend mismatch");
    }
    PeriodicGaussianThreeCenterPlan p;
    p.config = config; p.live = live; p.caps = caps;
    auto& d = p.descriptor;
    d.q_index = source.q_index(); d.conjugate_q_index = source.conjugate_q_index();
    d.k_bra_index = selection.k_bra_index; d.k_ket_index = c.ket_index(d.k_bra_index, d.q_index);
    const auto ket_address = c.mesh().add_transfer_with_wrap(
        c.mesh().address(d.k_bra_index), c.mesh().transfer_address(d.q_index));
    for (int axis = 0; axis != 3; ++axis) d.k_ket_reciprocal_wrap[axis] = ket_address.wrap[axis];
    d.ao_pair_begin = selection.ao_pair_begin; d.ao_pair_count = selection.ao_pair_count;
    d.auxiliary_begin = selection.auxiliary_begin; d.auxiliary_count = selection.auxiliary_count;
    d.element_count = mul(d.ao_pair_count, d.auxiliary_count);
    p.n_basis = n; p.n_auxiliary = a; p.accepted_vector_count = count;
    p.reciprocal_candidate_count = source.candidate_count();
    p.reciprocal_capacity = std::min(count, config.reciprocal_block);
    p.reciprocal_panel_count = count / p.reciprocal_capacity + (count % p.reciprocal_capacity != 0);
    const auto pairs = d.ao_pair_count, g = p.reciprocal_capacity, raw = mul(a, pairs);
    for (auto elements : {w_square, raw, mul(a, g), mul(pairs, g), d.element_count}) {
        limit(elements, std::vector<Complex>().max_size(), "Gaussian tile vector extent exceeds cap");
    }
    limit(mul(5U, g), std::vector<double>().max_size(), "Gaussian tile reciprocal vector extent exceeds cap");
    limit(d.element_count, (kShaMax - kPayloadPrefix) / 16U, "Gaussian tile payload SHA extent exceeds cap");
    // Verify W's independent payload extent before its scan as well.
    limit(w_square, (kShaMax - kWhitenerPayloadPrefix) / 16U, "Gaussian tile whitener SHA extent exceeds cap");
    p.resident_whitener_bytes = mul(16U, w_square); p.raw_panel_bytes = mul(16U, raw);
    p.compensation_bytes = p.raw_panel_bytes; p.reciprocal_panel_bytes = mul(40U, g);
    p.double_auxiliary_fourier_bytes = mul(32U, mul(a, g)); p.ao_fourier_bytes = mul(16U, mul(pairs, g));
    p.fixed_fourier_numeric_workspace_bytes = ao_pair_fourier_fixed_numeric_workspace_bytes();
    p.output_bytes = mul(16U, d.element_count);
    p.assembly_owned_numeric_peak_bytes = add(add(p.raw_panel_bytes, p.compensation_bytes),
        add(p.reciprocal_panel_bytes, add(p.double_auxiliary_fourier_bytes,
            add(p.ao_fourier_bytes, p.fixed_fourier_numeric_workspace_bytes))));
    p.whitening_owned_numeric_peak_bytes = add(p.raw_panel_bytes, p.output_bytes);
    p.owned_numeric_peak_bytes = std::max(p.assembly_owned_numeric_peak_bytes, p.whitening_owned_numeric_peak_bytes);
    p.borrowed_basis_active_numeric_bytes = c.inventory().combined_borrowed_active_numeric_bytes;
    p.fixed_inventoried_object_bytes = sizeof(PeriodicGaussianSourceContext) + sizeof(PeriodicGaussianReciprocalSource)
        + sizeof(PeriodicGaussianMetricWhitener) + sizeof(PeriodicGaussianThreeCenterPlan)
        + sizeof(PeriodicGaussianThreeCenterTile) + sizeof(PeriodicSystem);
    p.per_replica_inventoried_bytes = add(add(p.resident_whitener_bytes, p.owned_numeric_peak_bytes),
        add(p.borrowed_basis_active_numeric_bytes, add(p.fixed_inventoried_object_bytes,
            add(live.other_retained_bytes_per_replica, add(live.other_transient_bytes_per_replica,
                live.fixed_backend_margin_bytes_per_replica)))));
    p.node_inventoried_bytes = add(live.external_node_bytes, mul(live.replicas_per_node, p.per_replica_inventoried_bytes));
    p.reciprocal_candidate_evaluations = mul(2U, source.candidate_count());
    p.raw_contraction_term_count = mul(count, raw);
    p.whitening_term_count = mul(a, d.element_count);
    const auto scan = add(mul(2U, basis_scan(c.inventory().ao)), basis_scan(c.inventory().auxiliary));
    const auto lookup = add(add(c.inventory().ao.shell_count, c.inventory().ao.contraction_count), 1U);
    p.preflight_work_units_upper_bound = c.inventory().work_units_upper_bound;
    for (auto term : {mul(128U, w_square), mul(1024U, source.candidate_count()), mul(128U, scan),
        mul(512U, mul(pairs, lookup)), mul(128U, caps.maximum_image_candidates), std::uint64_t(4096U)}) {
        p.preflight_work_units_upper_bound = add(p.preflight_work_units_upper_bound, term);
    }
    limit(p.owned_numeric_peak_bytes, r.maximum_owned_numeric_bytes, "Gaussian tile owned numerical memory exceeds cap");
    limit(p.per_replica_inventoried_bytes, r.maximum_per_replica_inventoried_bytes, "Gaussian tile per-replica inventory exceeds cap");
    limit(p.node_inventoried_bytes, r.maximum_node_inventoried_bytes, "Gaussian tile node inventory exceeds cap");
    limit(p.reciprocal_candidate_evaluations, r.maximum_candidate_evaluations, "Gaussian tile reciprocal candidates exceed cap");
    limit(p.preflight_work_units_upper_bound, r.maximum_work_units, "Gaussian tile preflight work exceeds cap");
    return p;
}

} // namespace

PeriodicGaussianThreeCenterPlan plan_periodic_gaussian_three_center_tile(
    const PeriodicGaussianReciprocalSource& source, const PeriodicGaussianMetricWhitener& w,
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicGaussianThreeCenterSelection& selection,
    const PeriodicGaussianThreeCenterConfig& config, const PeriodicGaussianMetricLiveInventory& live,
    const PeriodicGaussianThreeCenterCaps& caps) {
    auto p = basic_plan(source, w, selection, config, live, caps);
    const auto& c = *source.context_handle();
    // Do not let generous caller caps authorize scans exceeding this plan's
    // stored-census work/borrowed inventory when an actual basis is changed.
    c.verify_bases(ao, auxiliary, bounded_basis_caps(c, config.basis_verification_caps));
    verify_whitener_payload(w);
    visit_periodic_gaussian_reciprocal_source(source, source.candidate_count(),
        [](const std::array<std::int64_t, 3>&, const std::array<double, 5>&, void*) {}, nullptr);
    const auto system = original_system(c);
    const auto census = ao_pair_gaussian_fourier_panel(ao, system, AuxiliaryFourierVectorView{},
        ket_cartesian(c, p.descriptor.k_ket_index), p.descriptor.ao_pair_begin, p.descriptor.ao_pair_count,
        c.options().ao_pair_image_cutoff_bohr, caps.maximum_image_candidates, 1U);
    const auto empty_auxiliary = auxiliary_gaussian_fourier_panel(auxiliary, AuxiliaryFourierVectorView{}, 1U);
    if (!census.data.empty() || !empty_auxiliary.data.empty()
        || census.fixed_numeric_workspace_bytes != p.fixed_fourier_numeric_workspace_bytes) {
        throw std::logic_error("Gaussian tile allocation-free census changed workspace");
    }
    p.image_candidate_count = census.image_candidate_count;
    p.retained_pair_image_count = census.retained_pair_image_count;
    const auto n = p.accepted_vector_count, k = p.reciprocal_panel_count, a = p.n_auxiliary;
    const auto pairs = p.descriptor.ao_pair_count;
    p.image_candidate_evaluations = mul(p.image_candidate_count, add(add(1U, k), n));
    p.primitive_pair_evaluations_upper_bound = mul(mul(n, p.retained_pair_image_count),
        mul(c.inventory().ao.exponent_count, c.inventory().ao.exponent_count));
    const auto scan = add(mul(2U, basis_scan(c.inventory().ao)), basis_scan(c.inventory().auxiliary));
    const auto lookup = add(add(c.inventory().ao.shell_count, c.inventory().ao.contraction_count), 1U);
    p.work_units_upper_bound = p.preflight_work_units_upper_bound;
    // 2^20 abstract units dominate L<=6 primitive MD, <=28² Cartesian pairs,
    // three <=13-term polynomial axes and scalar radial/phase checks. This
    // deliberately loose upper bound is not an instruction-count estimate.
    for (auto term : {mul(1024U, p.reciprocal_candidate_count), mul(128U, mul(k, scan)),
        mul(512U, mul(mul(k, pairs), lookup)), mul(256U, mul(add(k, 1U), pairs)),
        mul(128U, mul(p.image_candidate_count, add(k, n))),
        mul(1048576U, p.primitive_pair_evaluations_upper_bound),
        mul(128U, mul(n, c.inventory().auxiliary.coefficient_count)),
        mul(512U, mul(28U, mul(n, a))), mul(64U, p.raw_contraction_term_count),
        mul(64U, mul(n, a)), mul(128U, p.whitening_term_count),
        mul(128U, add(mul(a, pairs), p.descriptor.element_count)), std::uint64_t(4096U)}) {
        p.work_units_upper_bound = add(p.work_units_upper_bound, term);
    }
    limit(p.work_units_upper_bound, caps.resources.maximum_work_units, "Gaussian tile full numerical work exceeds cap");
    Digest h; h.text("vibeqc.periodic.gaussian-three-center.plan"); h.u32(1U);
    h.text(c.source_context_identity_sha256()); h.text(source.source_identity_sha256());
    h.text(source.conjugate_source_identity_sha256()); h.text(w.payload_identity_sha256());
    hash_plan(h, p); p.identity_ascii = h.finish();
    return p;
}

PeriodicGaussianThreeCenterTile build_periodic_gaussian_three_center_tile(
    const PeriodicGaussianReciprocalSource& source, const PeriodicGaussianMetricWhitener& w,
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicGaussianThreeCenterSelection& selection,
    const PeriodicGaussianThreeCenterConfig& config, const PeriodicGaussianMetricLiveInventory& live,
    const PeriodicGaussianThreeCenterCaps& caps) {
    const auto p = plan_periodic_gaussian_three_center_tile(source, w, ao, auxiliary, selection, config, live, caps);
    const auto& c = *source.context_handle();
    const auto system = original_system(c);
    const auto& d = p.descriptor;
    const detail::ThreeCenterNumericalSelection numeric_selection{
        d.ao_pair_begin, d.ao_pair_count, d.auxiliary_begin, d.auxiliary_count};
    auto numeric = detail::contract_three_center_numeric(ao, auxiliary, system, ket_cartesian(c, d.k_ket_index),
        numeric_selection, w.matrix_row_major(), p.accepted_vector_count, p.reciprocal_capacity,
        c.options().ao_pair_image_cutoff_bohr, caps.maximum_image_candidates,
        p.image_candidate_count, p.retained_pair_image_count, p.owned_numeric_peak_bytes,
        [](detail::PeriodicReciprocalRecordCallback callback, void* user, const void* pointer) {
            const auto& s = *static_cast<const PeriodicGaussianReciprocalSource*>(pointer);
            return visit_periodic_gaussian_reciprocal_source(s, s.candidate_count(), callback, user);
        }, &source);
    if (numeric.visited_reciprocal_count != p.accepted_vector_count || numeric.values.size() != d.element_count) {
        throw std::logic_error("Gaussian tile numerical result changed admitted shape");
    }
    PeriodicGaussianThreeCenterTile result;
    result.context_ = source.context_handle(); result.plan_ = p;
    result.source_ = ascii(source.source_identity_sha256()); result.opposite_ = ascii(source.conjugate_source_identity_sha256());
    result.whitener_ = ascii(w.payload_identity_sha256()); result.values_ = std::move(numeric.values);
    Digest h; h.text(kPayloadDomain); h.u32(1U);
    h.text(c.source_context_identity_sha256()); h.text(source.source_identity_sha256());
    h.text(source.conjugate_source_identity_sha256()); h.text(w.payload_identity_sha256());
    h.text(kPeriodicGaussianThreeCenterImagePolicy); h.text(kPeriodicGaussianThreeCenterLatticePolicy);
    h.real(c.options().ao_pair_image_cutoff_bohr);
    for (int i = 0; i != 3; ++i) for (int j = 0; j != 3; ++j) h.real(c.direct_lattice()(i, j));
    k_record(h, c.k_record(d.k_bra_index)); k_record(h, c.k_record(d.k_ket_index));
    descriptor(h, d);
    h.u64(p.image_candidate_count); h.u64(p.retained_pair_image_count);
    h.u64(p.accepted_vector_count); h.u64(p.output_bytes);
    if (h.extent() != kPayloadPrefix) throw std::logic_error("Gaussian tile payload prefix wire changed");
    for (const auto z : result.values_) { h.real(z.real()); h.real(z.imag()); }
    result.payload_ = h.finish();
    return result;
}

#define VIBEQC_GAUSSIAN_TILE_ID(Name, Field) \
std::string PeriodicGaussianThreeCenterTile::Name() const { return {Field.begin(), Field.end()}; }
VIBEQC_GAUSSIAN_TILE_ID(source_identity_sha256, source_)
VIBEQC_GAUSSIAN_TILE_ID(conjugate_source_identity_sha256, opposite_)
VIBEQC_GAUSSIAN_TILE_ID(whitener_payload_identity_sha256, whitener_)
VIBEQC_GAUSSIAN_TILE_ID(payload_identity_sha256, payload_)
#undef VIBEQC_GAUSSIAN_TILE_ID

} // namespace vibeqc
