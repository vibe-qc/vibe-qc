#pragma once

// Actual finite-source common-PAO pair spaces for Nejad2025
// doi:10.1063/5.0290816 Eqs.37-44. This is projected pair data, not a
// coupled-MP2 solution, automatic distinct PAO domains or production DLPNO.
#include <optional>
#include "vibeqc/periodic_gaussian_pair_pno_frame.hpp"

namespace vibeqc {

struct PeriodicGaussianPairSpaceOptions {
    // Only a physical diagonal pair is averaged to exact symmetry. The
    // represented C^T G C is symmetric analytically; this explicit bound
    // limits the measured Frobenius raw-to-symmetric roundoff projection.
    double maximum_diagonal_symmetry_projection_error = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianPairSpaceLiveInventory {
    // Extra live owners beyond reference baseline, PNO input, real basis
    // and (legacy/embedded only) provider rows. Other retained pair spaces
    // go here. The direct PAO-Gram source does not accept a provider.
    std::uint64_t other_live_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianPairSpaceCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianPairSpaceStorage {
    std::uint64_t virtual_count = 0, retained_dimension = 0, integral_calls = 0;
    std::uint64_t generation_dimension = 0;
    std::uint64_t construction_owned_bytes = 0, borrowed_pno_bytes = 0;
    std::uint64_t retained_output_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t construction_live_numerical_bytes = 0;
    std::uint64_t numerical_work_units = 0, fixed_control_storage_bytes = 0;
};
// Allocation-free upper inventory usable BEFORE PNO generation with r=n.
// Construction: G'/comp16r² + row/comp16r, borrowing P=8nr+8r+8n.
// Publish transfers P without copying after row/comp release, retains P+8r².
// Owned peak=max(16r²+16r,P+8r²); simultaneously live=P+16r²+16r,
// NOT owned_peak+P. Fixed control allowance is not an exact RSS claim.
PeriodicGaussianPairSpaceStorage periodic_gaussian_pair_space_storage(
    std::uint64_t virtual_count,std::uint64_t retained_dimension);
// Embedded pair generation keeps actual D[m,r] and m original occupations,
// not n padded occupations: P=8(n*r+m*r+r+m), 0<=r<=m<=n. Unlike the
// two-argument legacy API, this counts a distinct local D even when m=n.
// No generation-domain density or local coefficient matrix is rebuilt.
PeriodicGaussianPairSpaceStorage periodic_gaussian_pair_space_storage(
    std::uint64_t virtual_count,std::uint64_t generation_dimension,std::uint64_t retained_dimension);
// Direct source already owns G_PNO: P=8(n*r+m*r+r+m+r*r). No numerical
// scratch, G copy, new projection or integral callback. Before transfer P
// is borrowed, after the final noexcept move P is retained. The owned cap
// includes this transferred P, so rank zero still retains 8*m occupations.
// Resource-plan receipts may change with sizeof(tagged owner); the older
// numerical/source wires and arithmetic remain unchanged.
PeriodicGaussianPairSpaceStorage periodic_gaussian_direct_gram_pair_space_storage(
    std::uint64_t virtual_count,std::uint64_t generation_dimension,std::uint64_t retained_dimension);

struct PeriodicGaussianPairSpacePlan : PeriodicGaussianPairSpaceStorage {
    std::uint64_t occupied_count = 0, occupied_slot_i = 0, occupied_slot_j = 0;
    std::uint64_t provider_work_units = 0, work_units = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0;
    std::uint64_t state_resident_bytes = 0, replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    PeriodicGaussianPairSpaceOptions options;
    PeriodicGaussianPairSpaceLiveInventory live;
    PeriodicGaussianPairSpaceCaps caps;
    std::array<char,64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(),identity_ascii.end()}; }
};
struct PeriodicGaussianPairSpaceDiagnostics {
    std::uint64_t completed_integral_calls = 0, product_underflow_count = 0;
    bool source_integrals_replayed = false, diagonal_symmetry_projection_applied = false;
    double maximum_scalar_integral_roundoff_error = 0.0;
    // Provider bound remains in the COMMON basis; not claimed to be a
    // projected-integral or correlation-energy error bound.
    double common_basis_integral_projection_error_bound = 0.0;
    double maximum_raw_diagonal_asymmetry = 0.0;
    double diagonal_symmetry_projection_frobenius_bound = 0.0;
};
class PeriodicGaussianPairSpace {
public:
    PeriodicGaussianPairSpace(const PeriodicGaussianPairSpace&) = delete;
    PeriodicGaussianPairSpace& operator=(const PeriodicGaussianPairSpace&) = delete;
    PeriodicGaussianPairSpace(PeriodicGaussianPairSpace&&) noexcept = default;
    PeriodicGaussianPairSpace& operator=(PeriodicGaussianPairSpace&&) = delete;
    const PeriodicGaussianPairSpacePlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianPairSpaceDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    PeriodicGaussianPairPNOFrameView frame() const;
    // Source-specific native getters reject the wrong tag. Python exposes
    // only a read-only frame view, never a consumable borrowed PNO owner.
    const PeriodicGaussianPairPNOResult& pnos() const;
    const PeriodicGaussianEmbeddedPairPNOResult& embedded_pnos() const;
    const PeriodicGaussianGramPairPNOResult& direct_gram_pnos() const;
    const std::vector<double>& exchange_integrals() const;
    const std::vector<double>& coefficients() const { return frame().coefficients(); }
    const std::vector<double>& energies() const { return frame().energies(); }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const { return frame().state_handle(); }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const { return frame().context_handle(); }
    PeriodicCorrelationPlacedOccupied occupied_i() const { return frame().occupied_i(); }
    PeriodicCorrelationPlacedOccupied occupied_j() const { return frame().occupied_j(); }
    bool diagonal_pair() const { return frame().diagonal_pair(); }
    bool coupled_mp2_solution() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& raw_exchange_integral_identity_sha256() const noexcept { return raw_integrals_; }
    const std::string& exchange_integral_identity_sha256() const noexcept { return integrals_; }
    const std::string& allocation_identity() const noexcept { return allocation_; }
    const std::string& local_basis_identity_sha256() const noexcept { return local_basis_; }
    const std::string& consumed_sources_identity_sha256() const noexcept { return sources_; }
private:
    PeriodicGaussianPairSpace() = default;
    PeriodicGaussianPairSpacePlan memory_;
    PeriodicGaussianPairSpaceDiagnostics diagnostics_;
    std::optional<PeriodicGaussianPairPNOFrame> pnos_;
    std::vector<double> exchange_;
    std::string identity_,payload_,raw_integrals_,integrals_,allocation_,local_basis_,sources_;
    static PeriodicGaussianPairSpace build(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,PeriodicGaussianPairPNOFrameView,
        const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
        const PeriodicGaussianPairSpaceCaps&);
    friend PeriodicGaussianPairSpace make_periodic_gaussian_pair_space(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,PeriodicGaussianPairPNOResult&&,
        const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
        const PeriodicGaussianPairSpaceCaps&);
    friend PeriodicGaussianPairSpace make_periodic_gaussian_pair_space(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        PeriodicGaussianGramPairPNOResult&&,
        const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
        const PeriodicGaussianPairSpaceCaps&);
    friend PeriodicGaussianPairSpace make_periodic_gaussian_pair_space(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,PeriodicGaussianEmbeddedPairPNOResult&&,
        const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
        const PeriodicGaussianPairSpaceCaps&);
};

PeriodicGaussianPairSpacePlan plan_periodic_gaussian_pair_space(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairPNOResult&,
    const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairSpaceCaps&);
// Immutable input is moved ONLY after successful computation, receipt
// validation and allocation. A rejected build leaves its PNO input intact.
// Exactly n² native scalar calls for r>0, zero for r=0. All raw G elements
// are replayed/hashed for nonempty r and must match the generating PNO seal.
// Stream G row -> G*C row -> C^T*(G*C); never allocate n² or n*r scratch.
PeriodicGaussianPairSpace make_periodic_gaussian_pair_space(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,PeriodicGaussianPairPNOResult&&,
    const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairSpaceCaps&);

// Direct authentic PAO-Gram result: no common-factor provider is accepted.
// Existing G_PNO is validated and borrowed after transfer; it is neither
// reconstructed from C nor copied/averaged. The explicit diagonal budget
// gates the source's already-measured retained-space projection. Zero
// integral calls are required and maximum_integral_calls may therefore be0.
// All caps precede payload scans; failure leaves the source owner intact.
PeriodicGaussianPairSpacePlan plan_periodic_gaussian_pair_space(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianGramPairPNOResult&,
    const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairSpaceCaps&);
PeriodicGaussianPairSpace make_periodic_gaussian_pair_space(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    PeriodicGaussianGramPairPNOResult&&,
    const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairSpaceCaps&);
PeriodicGaussianPairSpacePlan plan_periodic_gaussian_pair_space(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianEmbeddedPairPNOResult&,
    const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairSpaceCaps&);
PeriodicGaussianPairSpace make_periodic_gaussian_pair_space(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,PeriodicGaussianEmbeddedPairPNOResult&&,
    const PeriodicGaussianPairSpaceOptions&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairSpaceCaps&);

struct PeriodicGaussianPairOverlapCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianPairOverlapPlan {
    std::uint64_t virtual_count = 0, target_dimension = 0, source_dimension = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, borrowed_pair_space_bytes = 0;
    std::uint64_t work_units = 0, fixed_control_storage_bytes = 0;
    std::uint64_t state_resident_bytes = 0, replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    bool same_object_borrower = false;
    PeriodicGaussianPairSpaceLiveInventory live;
    PeriodicGaussianPairOverlapCaps caps;
    std::array<char,64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(),identity_ascii.end()}; }
};
class PeriodicGaussianPairOverlap {
public:
    PeriodicGaussianPairOverlap(const PeriodicGaussianPairOverlap&) = delete;
    PeriodicGaussianPairOverlap& operator=(const PeriodicGaussianPairOverlap&) = delete;
    PeriodicGaussianPairOverlap(PeriodicGaussianPairOverlap&&) noexcept = default;
    PeriodicGaussianPairOverlap& operator=(PeriodicGaussianPairOverlap&&) = delete;
    const PeriodicGaussianPairOverlapPlan& memory() const noexcept { return memory_; }
    const std::vector<double>& overlaps() const;
    const std::string& target_identity_sha256() const noexcept { return target_; }
    const std::string& source_identity_sha256() const noexcept { return source_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    std::uint64_t product_underflow_count() const noexcept { return underflows_; }
private:
    PeriodicGaussianPairOverlap() = default;
    PeriodicGaussianPairOverlapPlan memory_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<double> values_;
    std::string target_,source_,identity_,payload_;
    std::uint64_t underflows_ = 0;
    friend PeriodicGaussianPairOverlap make_periodic_gaussian_pair_overlap(
        const PeriodicCorrelationAdmittedReference&,const PeriodicGaussianPairSpace&,
        const PeriodicGaussianPairSpace&,const PeriodicGaussianPairSpaceLiveInventory&,
        const PeriodicGaussianPairOverlapCaps&);
};
// O=C_target^T*C_source, row-major [r,s], 8rs owned bytes and scalar
// compensation. Both immutable pair spaces must have the exact same
// state/context/allocation/common-basis/HF origin. The older two kinds also
// require identical provider/source receipts. Direct/direct overlaps keep
// each pair's independent selected-Gram receipts, which need not match.
// Mixing a direct source with an older kind is not qualified and rejects.
// Different
// occupied pairs and ranks are allowed. Exact same-object borrowing is
// counted once; no copied coefficients, no user-labelled overlap arrays.
PeriodicGaussianPairOverlapPlan plan_periodic_gaussian_pair_overlap(
    const PeriodicCorrelationAdmittedReference&,const PeriodicGaussianPairSpace&,
    const PeriodicGaussianPairSpace&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairOverlapCaps&);
PeriodicGaussianPairOverlap make_periodic_gaussian_pair_overlap(
    const PeriodicCorrelationAdmittedReference&,const PeriodicGaussianPairSpace&,
    const PeriodicGaussianPairSpace&,const PeriodicGaussianPairSpaceLiveInventory&,
    const PeriodicGaussianPairOverlapCaps&);

} // namespace vibeqc
