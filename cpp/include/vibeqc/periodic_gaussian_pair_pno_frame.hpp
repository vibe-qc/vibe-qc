#pragma once

// Sealed native ownership adapter, not another numerical PNO representation.
// Legacy generation uses m=n; embedded and direct PAO-Gram generation have
// m<=n and retain ALL m original occupations. No branch pads occupations or
// copies payloads. Direct generation also owns its retained-space integrals;
// it has no legacy common-provider or common ordered-integral receipt.
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include "vibeqc/periodic_gaussian_embedded_pair_pnos.hpp"
#include "vibeqc/periodic_gaussian_gram_pair_pnos.hpp"

namespace vibeqc {

enum class PeriodicGaussianPairPNOFrameKind { Common, Embedded, DirectPAOGram };
class PeriodicGaussianPairPNOFrameView;
struct PeriodicGaussianPairPNOFrameValidationPlan {
    std::uint64_t numerical_lanes = 0, work_units = 0, control_storage_bytes = 0;
};
PeriodicGaussianPairPNOFrameValidationPlan plan_periodic_gaussian_pair_pno_frame_validation(
    const PeriodicGaussianPairPNOFrameView&);
// Caller MUST admit the published plan before this scan. This verifies the
// source-specific numerical payload receipt, not a new physical certificate.
// A constant-size digest/hex receipt is temporary; no numerical heap/copy.
void validate_periodic_gaussian_pair_pno_frame_payload(const PeriodicGaussianPairPNOFrameView&);

// Allocation-free const borrow from an authentic native result. The source
// must remain alive at the same address and unmoved throughout every use.
// Metadata access checks live owners and extents, not numerical payloads or
// physical-source hashes; separately budget any full validation/replay.
// Do not expose the typed result getters through Python: a binding which
// discards const could otherwise pass the child to a consuming native factory.
class PeriodicGaussianPairPNOFrameView {
public:
    explicit PeriodicGaussianPairPNOFrameView(const PeriodicGaussianPairPNOResult& value) noexcept : source_(&value) {}
    explicit PeriodicGaussianPairPNOFrameView(const PeriodicGaussianEmbeddedPairPNOResult& value) noexcept : source_(&value) {}
    explicit PeriodicGaussianPairPNOFrameView(const PeriodicGaussianGramPairPNOResult& value) noexcept : source_(&value) {}
    PeriodicGaussianPairPNOFrameView(PeriodicGaussianPairPNOResult&&) = delete;
    PeriodicGaussianPairPNOFrameView(const PeriodicGaussianPairPNOResult&&) = delete;
    PeriodicGaussianPairPNOFrameView(PeriodicGaussianEmbeddedPairPNOResult&&) = delete;
    PeriodicGaussianPairPNOFrameView(const PeriodicGaussianEmbeddedPairPNOResult&&) = delete;
    PeriodicGaussianPairPNOFrameView(PeriodicGaussianGramPairPNOResult&&) = delete;
    PeriodicGaussianPairPNOFrameView(const PeriodicGaussianGramPairPNOResult&&) = delete;

    PeriodicGaussianPairPNOFrameKind kind() const noexcept {
        return static_cast<PeriodicGaussianPairPNOFrameKind>(source_.index());
    }
    bool is_embedded() const noexcept { return kind()==PeriodicGaussianPairPNOFrameKind::Embedded; }
    bool is_direct_gram() const noexcept { return kind()==PeriodicGaussianPairPNOFrameKind::DirectPAOGram; }
    const PeriodicGaussianPairPNOResult& legacy() const {
        if(is_direct_gram()) throw std::logic_error("direct PAO-Gram PNO frame has no legacy result");
        if(is_embedded()) throw std::logic_error("embedded PNO frame has no legacy result");
        require_live();return *std::get<0>(source_);
    }
    const PeriodicGaussianEmbeddedPairPNOResult& embedded() const {
        if(is_direct_gram()) throw std::logic_error("direct PAO-Gram PNO frame has no embedded result");
        if(!is_embedded()) throw std::logic_error("common PNO frame has no embedded result");
        require_live();return *std::get<1>(source_);
    }
    const PeriodicGaussianGramPairPNOResult& direct_gram() const {
        if(!is_direct_gram()) throw std::logic_error("PNO frame has no direct PAO-Gram result");
        require_live();return *std::get<2>(source_);
    }
    std::uint64_t common_virtual_dimension() const {
        require_live();return std::visit([](const auto* p) { return common_dimension(*p); },source_);
    }
    std::uint64_t generation_dimension() const {
        require_live();return std::visit([](const auto* p) { return generation_dimension_of(*p); },source_);
    }
    std::uint64_t retained_dimension() const {
        require_live();return std::visit([](const auto* p) { return p->diagnostics().retained_dimension; },source_);
    }
    std::uint64_t occupied_count() const {
        require_live();return std::visit([](const auto* p) { return p->memory().occupied_count; },source_);
    }
    std::uint64_t occupied_slot_i() const {
        require_live();return std::visit([](const auto* p) { return p->memory().occupied_slot_i; },source_);
    }
    std::uint64_t occupied_slot_j() const {
        require_live();return std::visit([](const auto* p) { return p->memory().occupied_slot_j; },source_);
    }
    PeriodicCorrelationPlacedOccupied occupied_i() const {
        require_live();return std::visit([](const auto* p) { return p->occupied_i(); },source_);
    }
    PeriodicCorrelationPlacedOccupied occupied_j() const {
        require_live();return std::visit([](const auto* p) { return p->occupied_j(); },source_);
    }
    bool diagonal_pair() const { return occupied_slot_i()==occupied_slot_j(); }
    bool complete_generation_pair_space() const { return retained_dimension()==generation_dimension(); }
    bool complete_common_virtual_space() const { return retained_dimension()==common_virtual_dimension(); }
    const std::vector<double>& coefficients() const {
        require_live();return std::visit([](const auto* p)->const std::vector<double>& { return p->coefficients(); },source_);
    }
    const std::vector<double>& energies() const {
        require_live();return std::visit([](const auto* p)->const std::vector<double>& { return p->energies(); },source_);
    }
    const std::vector<double>& original_pno_occupations() const {
        require_live();return std::visit([](const auto* p)->const std::vector<double>& { return p->original_pno_occupations(); },source_);
    }
    const PeriodicGaussianPairPNOOptions& options() const {
        require_live();
        if(is_direct_gram()) return std::get<2>(source_)->options().pno.pno;
        if(is_embedded()) return std::get<1>(source_)->options().pno;
        return std::get<0>(source_)->memory().options;
    }
    // Embedded generation diagnostics describe LOCAL D[m,r], not both local
    // and exported n*r payloads. Use retained_numerical_bytes() for the owner.
    const PeriodicGaussianPairPNODiagnostics& generation_diagnostics() const {
        require_live();
        if(is_direct_gram()) return std::get<2>(source_)->diagnostics().generation;
        if(is_embedded()) return std::get<1>(source_)->diagnostics().generation;
        return std::get<0>(source_)->diagnostics();
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const {
        require_live();return std::visit([](const auto* p)->const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& { return p->state_handle(); },source_);
    }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const {
        require_live();return std::visit([](const auto* p)->const std::shared_ptr<const PeriodicGaussianSourceContext>& { return p->context_handle(); },source_);
    }
    const std::string& identity_sha256() const {
        require_live();return std::visit([](const auto* p)->const std::string& { return p->identity_sha256(); },source_);
    }
    const std::string& payload_sha256() const {
        require_live();return std::visit([](const auto* p)->const std::string& { return p->payload_sha256(); },source_);
    }
    const std::string& basis_identity_sha256() const {
        require_live();return std::visit([](const auto* p)->const std::string& { return p->basis_identity_sha256(); },source_);
    }
    const std::string& provider_identity_sha256() const {
        require_live();
        if(is_direct_gram()) throw std::logic_error("direct PAO-Gram PNO frame has no legacy provider receipt");
        return is_embedded() ? std::get<1>(source_)->provider_identity_sha256()
            : std::get<0>(source_)->provider_identity_sha256();
    }
    const std::string& hf_reference_source_identity_sha256() const {
        require_live();return std::visit([](const auto* p)->const std::string& { return p->hf_reference_source_identity_sha256(); },source_);
    }
    // Exactly one common ordered-integral wire for both older sources. Embedded
    // projected G and raw pair-frame G remain distinct, typed receipts.
    const std::string& common_exchange_integral_identity_sha256() const {
        require_live();
        if(is_direct_gram()) throw std::logic_error("direct PAO-Gram PNO frame has no common ordered-integral receipt");
        if(is_embedded()) return std::get<1>(source_)->common_exchange_integral_identity_sha256();
        return std::get<0>(source_)->exchange_integral_identity_sha256();
    }
    const std::string& generation_exchange_integral_identity_sha256() const {
        require_live();
        if(is_direct_gram()) return std::get<2>(source_)->projected_exchange_identity_sha256();
        if(is_embedded()) return std::get<1>(source_)->projected_exchange_identity_sha256();
        return std::get<0>(source_)->exchange_integral_identity_sha256();
    }
    const std::string& initial_amplitude_identity_sha256() const {
        require_live();return std::visit([](const auto* p)->const std::string& { return p->initial_amplitude_identity_sha256(); },source_);
    }
    const std::string& density_identity_sha256() const {
        require_live();return std::visit([](const auto* p)->const std::string& { return p->density_identity_sha256(); },source_);
    }
    // Numerical payload only; source state/context are shared, not counted
    // here. Caller separately inventories sizeof(owner), receipt storage and
    // unique state/context lifetimes. No allocator/RSS exactness is claimed.
    std::uint64_t retained_numerical_bytes() const {
        require_live();return std::visit([](const auto* p) {
            auto lanes=checked_add(checked_add(p->coefficients().size(),p->energies().size()),p->original_pno_occupations().size());
            if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*p)>>,
                PeriodicGaussianEmbeddedPairPNOResult>
                || std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*p)>>,
                    PeriodicGaussianGramPairPNOResult>)
                lanes=checked_add(lanes,p->generation_coefficients().size());
            if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*p)>>,
                PeriodicGaussianGramPairPNOResult>)
                lanes=checked_add(lanes,p->exchange_integrals().size());
            return checked_mul(8,lanes);
        },source_);
    }
    // Every factory receipt is a fixed 64-character SHA plus its terminator;
    // std::string objects are already part of sizeof(the tagged owner).
    std::uint64_t retained_receipt_payload_bytes() const {
        require_live();
        if(is_direct_gram()) return std::get<2>(source_)->retained_receipt_payload_bytes();
        return (is_embedded()?14U:8U)*65U;
    }
    void verify_payload() const { validate_periodic_gaussian_pair_pno_frame_payload(*this); }
    bool coupled_mp2_solution() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }

private:
    std::variant<const PeriodicGaussianPairPNOResult*,const PeriodicGaussianEmbeddedPairPNOResult*,
        const PeriodicGaussianGramPairPNOResult*> source_;
    static std::uint64_t checked_mul(std::uint64_t a,std::uint64_t b) {
        if(a && b>std::numeric_limits<std::uint64_t>::max()/a) throw std::overflow_error("PNO frame extent product overflows");
        return a*b;
    }
    static std::uint64_t checked_add(std::uint64_t a,std::uint64_t b) {
        if(b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("PNO frame extent sum overflows");
        return a+b;
    }
    static std::uint64_t common_dimension(const PeriodicGaussianPairPNOResult& p) noexcept { return p.memory().virtual_count; }
    static std::uint64_t common_dimension(const PeriodicGaussianEmbeddedPairPNOResult& p) noexcept { return p.memory().common_virtual_dimension; }
    static std::uint64_t common_dimension(const PeriodicGaussianGramPairPNOResult& p) noexcept { return p.memory().common_virtual_dimension; }
    static std::uint64_t generation_dimension_of(const PeriodicGaussianPairPNOResult& p) noexcept { return p.memory().virtual_count; }
    static std::uint64_t generation_dimension_of(const PeriodicGaussianEmbeddedPairPNOResult& p) noexcept { return p.memory().generation_dimension; }
    static std::uint64_t generation_dimension_of(const PeriodicGaussianGramPairPNOResult& p) noexcept { return p.memory().generation_dimension; }
    void require_live() const {
        std::visit([](const auto* p) {
            const auto n=common_dimension(*p),m=generation_dimension_of(*p),r=p->diagnostics().retained_dimension;
            if(!p->state_handle() || !p->context_handle() || !n || !m || m>n || r>m
                || !p->memory().occupied_count || p->memory().occupied_slot_i>=p->memory().occupied_count
                || p->memory().occupied_slot_j>=p->memory().occupied_count
                || p->coefficients().size()!=checked_mul(n,r) || p->energies().size()!=r
                || p->original_pno_occupations().size()!=m)
                throw std::logic_error("PNO frame source is consumed or malformed");
            if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*p)>>,
                PeriodicGaussianEmbeddedPairPNOResult>
                || std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*p)>>,
                    PeriodicGaussianGramPairPNOResult>)
                if(p->generation_coefficients().size()!=checked_mul(m,r))
                    throw std::logic_error("embedded PNO generation frame is consumed or malformed");
            if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*p)>>,
                PeriodicGaussianGramPairPNOResult>)
                if(p->exchange_integrals().size()!=checked_mul(r,r))
                    throw std::logic_error("direct PAO-Gram PNO integral payload is consumed or malformed");
        },source_);
    }
};

// No default/arbitrary-input constructor. Construction is only a noexcept
// native-result move, allowing callers to transfer ownership as the LAST
// step after all potentially failing admission, validation and allocations.
class PeriodicGaussianPairPNOFrame {
public:
    explicit PeriodicGaussianPairPNOFrame(PeriodicGaussianPairPNOResult&& p) noexcept
        : source_(std::in_place_type<PeriodicGaussianPairPNOResult>,std::move(p)) {}
    explicit PeriodicGaussianPairPNOFrame(PeriodicGaussianEmbeddedPairPNOResult&& p) noexcept
        : source_(std::in_place_type<PeriodicGaussianEmbeddedPairPNOResult>,std::move(p)) {}
    explicit PeriodicGaussianPairPNOFrame(PeriodicGaussianGramPairPNOResult&& p) noexcept
        : source_(std::in_place_type<PeriodicGaussianGramPairPNOResult>,std::move(p)) {}
    PeriodicGaussianPairPNOFrame(const PeriodicGaussianPairPNOFrame&) = delete;
    PeriodicGaussianPairPNOFrame& operator=(const PeriodicGaussianPairPNOFrame&) = delete;
    PeriodicGaussianPairPNOFrame(PeriodicGaussianPairPNOFrame&&) noexcept = default;
    PeriodicGaussianPairPNOFrame& operator=(PeriodicGaussianPairPNOFrame&&) = delete;
    PeriodicGaussianPairPNOFrameView view() const & {
        return std::visit([](const auto& p) { return PeriodicGaussianPairPNOFrameView(p); },source_);
    }
    PeriodicGaussianPairPNOFrameView view() const && = delete;
    const PeriodicGaussianPairPNOResult& legacy() const & { return view().legacy(); }
    const PeriodicGaussianEmbeddedPairPNOResult& embedded() const & { return view().embedded(); }
    const PeriodicGaussianGramPairPNOResult& direct_gram() const & { return view().direct_gram(); }
    const PeriodicGaussianPairPNOResult& legacy() const && = delete;
    const PeriodicGaussianEmbeddedPairPNOResult& embedded() const && = delete;
    const PeriodicGaussianGramPairPNOResult& direct_gram() const && = delete;
private:
    std::variant<PeriodicGaussianPairPNOResult,PeriodicGaussianEmbeddedPairPNOResult,
        PeriodicGaussianGramPairPNOResult> source_;
};
static_assert(std::is_nothrow_move_constructible<PeriodicGaussianPairPNOFrame>::value,
    "PNO frame transfer must preserve final-transfer exception guarantees");

} // namespace vibeqc
