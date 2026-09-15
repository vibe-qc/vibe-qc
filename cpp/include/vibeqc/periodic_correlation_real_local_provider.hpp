#pragma once

/// Bounded internal real-row provider for a selected finite-source RI
/// reference, not a production periodic CCSD/DLPNO method. It consumes native
/// immutable unified panels and a native real-basis certificate, not caller
/// factor arrays. Sun doi:10.1063/1.4998644 and Nejad doi:10.1063/5.0290816
/// fix the inherited finite-BvK Coulomb/normalization convention.
///
/// For each unordered density x, n_x=||L_x||_2 and
/// r_x=||L_reverse(x)-L_x||_2 use ALL actual q/original-auxiliary rows.
/// Keep each q representative unchanged and replace ONLY its partner by the
/// conjugate; at self-inverse q remove the explicitly gated imaginary lane.
/// d_x is the norm of this full-q change (no averaging assumption).
/// Through the unordered original Gram, both-orientation error is bounded
/// by 2*n*r+r^2, and covariance projection by 2*n*d+d^2, using maxima over x.
/// Their sum is 2*n*(r+d)+r^2+d^2, with no unneeded 2*r*d cross term.
///
/// Non-self-inverse pairs give TWO real rows sqrt(2)*Re L(q) and
/// sqrt(2)*Im L(q); neither complex lane is lost. A conservative conversion
/// norm bound c adds 2*(n+d)*c+c^2. Scalar compensated dots have a separate
/// outward interval/error gate. Image-tail and HF-operator matching remain
/// explicitly UNCERTIFIED, regardless of these numerical certificates.

#include "vibeqc/bounded_restricted_ccsd_target.hpp"
#include "vibeqc/periodic_correlation_real_local_basis.hpp"

namespace vibeqc {

namespace periodic_correlation_real_local_detail { struct ProviderAccess; }

enum class PeriodicCorrelationRealLocalProviderSourceKind : std::uint32_t {
    PrivateFactorStore = 0, MatchedGaussianHF = 1
};

inline constexpr std::uint32_t kPeriodicCorrelationRealLocalProviderContractVersion = 1;

struct PeriodicCorrelationRealLocalProviderOptions {
    double reversal_absolute_tolerance = 0.0, reversal_relative_tolerance = 0.0;
    double conjugacy_absolute_tolerance = 0.0, conjugacy_relative_tolerance = 0.0;
    double self_q_absolute_tolerance = 0.0, self_q_relative_tolerance = 0.0;
    double maximum_eri_projection_error = 0.0;
    double maximum_scalar_roundoff_error = 0.0;
};

struct PeriodicCorrelationRealLocalProviderCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_work_units = 0;
    std::uint64_t maximum_factor_panels = 0, maximum_tile_visits = 0, maximum_reader_tile_bytes = 0;
    std::uint64_t maximum_scalar_work_units = 0;
};

struct PeriodicCorrelationRealLocalProviderMemoryPlan {
    std::uint64_t n_cells = 0, n_auxiliary = 0, occupied_count = 0, virtual_count = 0, orbital_count = 0;
    std::uint64_t density_count = 0, row_count = 0, self_inverse_q_count = 0;
    std::uint64_t retained_row_bytes = 0, norm_workspace_bytes = 0;
    std::uint64_t retained_partner_panel_bytes = 0, building_panel_peak_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t live_basis_bytes = 0, basis_index_alias_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0, live_wannier_bytes = 0, live_domain_bytes = 0, live_space_bytes = 0;
    std::uint64_t live_reader_numeric_bytes = 0, live_reader_control_bytes = 0;
    std::uint64_t maximum_reader_tile_bytes = 0, factor_panels = 0, tile_visits = 0;
    std::uint64_t work_units = 0, scalar_work_units = 0, required_node_memory_bytes = 0;
};

struct PeriodicCorrelationRealLocalProviderDiagnostics {
    std::uint64_t factor_panels_built = 0, tile_visits = 0;
    double maximum_reversal_error = 0.0, maximum_conjugacy_error = 0.0, maximum_self_q_imaginary_magnitude = 0.0;
    double maximum_original_density_norm = 0.0, maximum_reversal_norm = 0.0, maximum_covariance_projection_norm = 0.0;
    double maximum_real_row_conversion_norm = 0.0;
    double orientation_error_bound = 0.0, covariance_error_bound = 0.0, conversion_error_bound = 0.0;
    double maximum_eri_projection_error_bound = 0.0;
};

struct PeriodicCorrelationRealLocalIntegral {
    double value = 0.0;
    double roundoff_error_bound = 0.0;
};

/// Owns only real rows [Nk*Naux,m*(m+1)/2] plus fixed sealed metadata/state.
/// Canonical density order is left<=right, row-major upper triangle. Row
/// slot q*Naux+P stores cosine when q<qbar, sine when q>qbar, and the real
/// self-q row otherwise. No extra q map, all-q complex tensor or ERI cache.
/// Scalar calls allocate nothing and never revisit the AO store. Their loop
/// work is admitted per call; the enclosing CCSD leaf caps total call count.
class PeriodicCorrelationRealLocalProvider {
public:
    PeriodicCorrelationRealLocalProvider(const PeriodicCorrelationRealLocalProvider&) = delete;
    PeriodicCorrelationRealLocalProvider& operator=(const PeriodicCorrelationRealLocalProvider&) = delete;
    PeriodicCorrelationRealLocalProvider(PeriodicCorrelationRealLocalProvider&&) noexcept = default;
    PeriodicCorrelationRealLocalProvider& operator=(PeriodicCorrelationRealLocalProvider&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationRealLocalProviderContractVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicCorrelationRealLocalProviderMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationRealLocalProviderOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationRealLocalProviderDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const double* rows_data() const;
    double row(std::size_t row, std::size_t left, std::size_t right) const;
    PeriodicCorrelationRealLocalIntegral integral(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const;
    /// Rows and basis labels are declared provider-retained; packed basis F
    /// is counted by the leaf's borrowed OO/VV/OV input views, not twice.
    /// Both owners must remain alive during the synchronous leaf call.
    BoundedRestrictedCCSDIntegralProvider integral_provider(const PeriodicCorrelationRealLocalBasis&) const;
    bool finite_image_reference() const noexcept { return true; }
    bool hf_hamiltonian_match_certified() const noexcept { return false; }
    bool ao_image_source_certified() const noexcept { return false; }
    PeriodicCorrelationRealLocalProviderSourceKind source_kind() const noexcept { return source_kind_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept {
        return state_ && source_kind_ == PeriodicCorrelationRealLocalProviderSourceKind::MatchedGaussianHF;
    }
    bool bitwise_hf_factor_consumption_verified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& local_basis_identity_sha256() const noexcept { return basis_; }
    const std::string& basis_certificate_identity_sha256() const noexcept { return certificate_; }
    const std::string& store_identity_sha256() const;
    const std::string& source_context_identity_sha256() const;
    const std::string& hf_reference_source_identity_sha256() const;
    const std::string& consumed_panels_identity_sha256() const noexcept { return consumed_; }
private:
    PeriodicCorrelationRealLocalProvider() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationRealLocalProviderMemoryPlan memory_;
    PeriodicCorrelationRealLocalProviderOptions options_;
    PeriodicCorrelationRealLocalProviderDiagnostics diagnostics_;
    std::vector<double> rows_;
    std::string identity_, payload_, basis_, certificate_, store_, consumed_;
    PeriodicCorrelationRealLocalProviderSourceKind source_kind_ = PeriodicCorrelationRealLocalProviderSourceKind::PrivateFactorStore;
    std::string source_context_, hf_source_;
    friend struct periodic_correlation_real_local_detail::ProviderAccess;
    friend PeriodicCorrelationRealLocalProvider make_periodic_correlation_real_local_provider(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicCorrelationRealLocalProviderOptions&, const PeriodicCorrelationRealLocalProviderCaps&);
};

PeriodicCorrelationRealLocalProviderMemoryPlan plan_periodic_correlation_real_local_provider(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationRealLocalBasis&);
PeriodicCorrelationRealLocalProvider make_periodic_correlation_real_local_provider(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_count, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicCorrelationRealLocalProviderOptions&, const PeriodicCorrelationRealLocalProviderCaps&);

}  // namespace vibeqc
