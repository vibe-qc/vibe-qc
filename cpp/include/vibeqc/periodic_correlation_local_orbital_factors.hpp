#pragma once

/// One-q global-auxiliary density factors for an explicitly selected LOCAL
/// occupied set and one common orthonormal PAO virtual block. Sun et al.,
/// doi:10.1063/1.4998644, Eqs.(3),(4),(13),(16),(21), and Nejad et al.,
/// doi:10.1063/5.0290816, Eqs.(2)-(10),(26)-(28):
/// L_lr(q,P)=Nk^-3/2 sum_k,mu,nu conj(C_l(mu,k))*C_r(nu,k+q)*B(q,P,mu,nu;k).
/// All ordered OO/OV/VO/VV entries are actually evaluated in one traversal
/// of the verified AO tiles intersecting the chosen q/auxiliary slice.
/// Occupied labels precede virtual labels. No all-q cache, four-index tensor,
/// real cast, same-q transpose, q/-q shortcut or energy calculation occurs.
/// Diagonal densities retain the source's omitted-zero-mode Hamiltonian;
/// there is no local chargeless-auxiliary approximation or added constant.

#include "vibeqc/periodic_correlation_density_factors.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationLocalOrbitalFactorContractVersion = 1;

struct PeriodicCorrelationLocalOrbitalFactorMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0;
    std::uint64_t occupied_count = 0, virtual_count = 0, orbital_count = 0, auxiliary_count = 0;
    std::uint64_t retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t coefficient_panel_bytes = 0, coefficient_scratch_bytes = 0;
    std::uint64_t retained_index_bytes = 0, caller_index_bytes = 0;
    /// 32*Ab*m*m +32*nao*m +32*nao +16*occupied_count, m=o+v.
    /// Explicit heap numerical/index payloads; fixed stack metadata/SHA
    /// states and string/allocator bookkeeping are excluded. The reader's
    /// conservative fixed control allowance is separately inventoried below.
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0, live_wannier_bytes = 0;
    std::uint64_t live_domain_bytes = 0, live_space_bytes = 0;
    std::uint64_t live_reader_numeric_bytes = 0, live_reader_control_bytes = 0;
    std::uint64_t maximum_reader_tile_bytes = 0, reader_payload_bytes = 0;
    std::uint64_t tile_visits = 0, work_units = 0, required_node_memory_bytes = 0;
};

/// Move-only complex [Ab,m,m] and one retained occupied index list. Numerical
/// source provenance and actual orbital-basis selection have separate seals.
/// Neither is a certification of real orbitals, factor covariance, matching
/// HF operator provenance or infinite-image convergence. All construction
/// controls are resource caps; there are no caller-asserted scientific flags.
///
/// SHA wire uses length-prefixed strings, BE u32/u64, finite binary64 lanes
/// and canonical signed zero. Prefix "vibeqc.periodic.correlation.local-orbital-factors",
/// suffixed by: ".selection": version1 u32, occupied_count u64 and ordered
/// (active index,cell) u64 pairs, then virtual begin/count/translation u64;
/// ".basis": v1 and state/calculation/allocation/selection/Wannier/gauge/
/// domain/space SHA strings; ".payload": v1, q/aux_begin/Ab/m u64 and
/// row-major complex lanes; ".consumed-tiles": v1, store string, count u64,
/// ordered sequence u64/tile SHA strings; ".identity": v1, local-basis/AO/
/// auxiliary/store/source/whitener/consumed/payload SHA strings and the fixed
/// numerical-policy string. Domain itself is the first prefixed string.
class PeriodicCorrelationLocalOrbitalFactorPanel {
public:
    PeriodicCorrelationLocalOrbitalFactorPanel(const PeriodicCorrelationLocalOrbitalFactorPanel&) = delete;
    PeriodicCorrelationLocalOrbitalFactorPanel& operator=(const PeriodicCorrelationLocalOrbitalFactorPanel&) = delete;
    PeriodicCorrelationLocalOrbitalFactorPanel(PeriodicCorrelationLocalOrbitalFactorPanel&&) noexcept = default;
    PeriodicCorrelationLocalOrbitalFactorPanel& operator=(PeriodicCorrelationLocalOrbitalFactorPanel&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationLocalOrbitalFactorContractVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicCorrelationLocalOrbitalFactorMemoryPlan& memory() const noexcept { return memory_; }
    std::uint64_t q_index() const noexcept { return q_; }
    std::uint64_t auxiliary_begin() const noexcept { return auxiliary_begin_; }
    PeriodicCorrelationPlacedOccupied occupied(std::size_t index) const;
    PeriodicCorrelationVirtualBlockSelection virtual_selection() const noexcept { return virtual_; }
    std::complex<double> element(std::size_t auxiliary, std::size_t left, std::size_t right) const;
    const std::complex<double>* data() const;
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    bool density_symmetry_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& selection_identity_sha256() const noexcept { return selection_; }
    const std::string& local_basis_identity_sha256() const noexcept { return basis_; }
    const std::string& ao_basis_identity_sha256() const noexcept { return ao_; }
    const std::string& auxiliary_basis_identity_sha256() const noexcept { return auxiliary_; }
    const std::string& source_identity_sha256() const noexcept { return source_; }
    const std::string& whitener_payload_identity_sha256() const noexcept { return whitener_; }
    const std::string& store_identity_sha256() const noexcept { return store_; }
    const std::string& consumed_tiles_identity_sha256() const noexcept { return consumed_; }
private:
    PeriodicCorrelationLocalOrbitalFactorPanel() = default;
    PeriodicCorrelationLocalOrbitalFactorMemoryPlan memory_;
    std::uint64_t q_ = 0, auxiliary_begin_ = 0;
    PeriodicCorrelationVirtualBlockSelection virtual_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> values_;
    std::vector<PeriodicCorrelationPlacedOccupied> occupied_;
    std::string identity_, payload_, selection_, basis_, ao_, auxiliary_, source_, whitener_, store_, consumed_;
    friend PeriodicCorrelationLocalOrbitalFactorPanel build_periodic_correlation_local_orbital_factor_panel(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const std::uint64_t*, std::size_t, std::size_t,
        const PeriodicCorrelationVirtualBlockSelection&, std::uint64_t, std::uint64_t, std::uint64_t,
        const PeriodicCorrelationLocalFactorCaps&);
};

/// O(1) sealed-object/count/overflow planning, without gauge/index scans or
/// tile reads. Lists and virtual ranks must be positive. Both caller and
/// retained index storage are charged; other live caller storage must already
/// be in admitted inventory. All native caps precede size-dependent work.
PeriodicCorrelationLocalOrbitalFactorMemoryPlan plan_periodic_correlation_local_orbital_factor_panel(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    std::uint64_t occupied_count, const PeriodicCorrelationVirtualBlockSelection&,
    std::uint64_t q_index, std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count);

/// Gauges and occupied labels are aligned immutable nonowning views. Labels
/// are contiguous uint64 [occupied_count,2]: (active occupied index, canonical
/// cell). Accessible count is the exact number of uint64 elements, not bytes.
/// Duplicate labels and modular aliases are rejected after cap admission.
PeriodicCorrelationLocalOrbitalFactorPanel build_periodic_correlation_local_orbital_factor_panel(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const std::uint64_t* occupied_indices, std::size_t accessible_count, std::size_t occupied_count,
    const PeriodicCorrelationVirtualBlockSelection&, std::uint64_t q_index,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count, const PeriodicCorrelationLocalFactorCaps&);

}  // namespace vibeqc
