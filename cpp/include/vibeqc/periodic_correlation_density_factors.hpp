#pragma once

/// Bounded OO/VV global-auxiliary density factors. Same finite-BvK convention
/// as periodic_correlation_local_factors.hpp, Sun doi:10.1063/1.4998644,
/// and Nejad doi:10.1063/5.0290816, Eqs.(2)-(10),(26)-(28):
/// L_lr^q[P] = Nk^-3/2 sum_k,mu,nu conj(C_l(mu,k))*C_r(nu,k+q)*B_mu,nu^q[P;k].
/// The bra and ket orbitals/translation phases are independently evaluated.
/// General complex Coulomb I(p,q,r,s)=sum_QP conj(L_qp^Q[P])*L_rs^Q[P].
/// No symmetry/real/Gram shortcut, energy adapter or all-q/system tensor.
/// Diagonal OO/VV densities can be charged. Retain the source's omitted-zero-
/// mode Coulomb/background convention: no local chargeless-aux projection,
/// inferred Madelung constant or surface correction is added here.
///
/// Canonical digests use length-prefixed strings, big-endian u32/u64 and
/// binary64 complex lanes with signed zero normalized. Each domain begins
/// with u32 version 1. The ".selection" domain has kind u32, then for OO
/// left/right counts u64 and ordered (occupied,cell) u64 pairs; for VV it has
/// the two block selections in declaration order. The ".payload" domain
/// has kind u32, q/aux_begin/Ab/left_count/right_count u64, then row-major
/// complex values. Both domain prefixes are
/// "vibeqc.periodic.correlation.density-factors". The ".consumed-tiles"
/// domain binds store SHA, visit count and ordered (sequence,tile SHA).
/// The ".identity" domain binds state/calculation/allocation/selection/
/// store/source/whitener/consumed/payload SHA strings, the fixed numerical
/// policy string, then Wannier/gauge (OO) or domain/space (VV) SHA strings.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_local_factors.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationDensityFactorContractVersion = 1;

enum class PeriodicCorrelationDensityFactorKind : std::uint32_t {
    OccupiedOccupied = 0, VirtualVirtual = 1,
};

struct PeriodicCorrelationPlacedOccupied {
    std::uint64_t occupied_index = 0;
    std::uint64_t cell = 0;
};

struct PeriodicCorrelationVirtualBlockSelection {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
    std::uint64_t translation_cell = 0;
};

struct PeriodicCorrelationDensityFactorMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0;
    std::uint64_t left_count = 0, right_count = 0, auxiliary_count = 0;
    std::uint64_t retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t coefficient_panel_bytes = 0, coefficient_scratch_bytes = 0;
    std::uint64_t retained_index_bytes = 0, caller_index_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0, live_wannier_bytes = 0;
    std::uint64_t live_domain_bytes = 0, live_space_bytes = 0;
    std::uint64_t live_reader_numeric_bytes = 0, live_reader_control_bytes = 0;
    std::uint64_t maximum_reader_tile_bytes = 0, reader_payload_bytes = 0;
    std::uint64_t tile_visits = 0, work_units = 0, required_node_memory_bytes = 0;
};

/// Move-only row-major [auxiliary_count,left_count,right_count] owner. OO
/// retains precisely the selected index lists; VV stores its two fixed-size
/// block selections. Known live borrowed inputs are charged separately.
/// Other live caller allocations must already be in admitted inventory.
class PeriodicCorrelationDensityFactorBlock {
public:
    PeriodicCorrelationDensityFactorBlock(const PeriodicCorrelationDensityFactorBlock&) = delete;
    PeriodicCorrelationDensityFactorBlock& operator=(const PeriodicCorrelationDensityFactorBlock&) = delete;
    PeriodicCorrelationDensityFactorBlock(PeriodicCorrelationDensityFactorBlock&&) noexcept = default;
    PeriodicCorrelationDensityFactorBlock& operator=(PeriodicCorrelationDensityFactorBlock&&) noexcept = default;
    PeriodicCorrelationDensityFactorKind kind() const noexcept { return kind_; }
    const PeriodicCorrelationDensityFactorMemoryPlan& memory() const noexcept { return memory_; }
    std::uint64_t q_index() const noexcept { return q_; }
    std::uint64_t auxiliary_begin() const noexcept { return auxiliary_begin_; }
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    PeriodicCorrelationPlacedOccupied left_occupied(std::size_t index) const;
    PeriodicCorrelationPlacedOccupied right_occupied(std::size_t index) const;
    PeriodicCorrelationVirtualBlockSelection left_virtual() const;
    PeriodicCorrelationVirtualBlockSelection right_virtual() const;
    std::complex<double> element(std::size_t auxiliary, std::size_t left, std::size_t right) const;
    const std::complex<double>* data() const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& selection_identity_sha256() const noexcept { return selection_; }
    const std::string& source_identity_sha256() const noexcept { return source_; }
    const std::string& whitener_payload_identity_sha256() const noexcept { return whitener_; }
    const std::string& store_identity_sha256() const noexcept { return store_; }
    const std::string& consumed_tiles_identity_sha256() const noexcept { return consumed_; }
private:
    PeriodicCorrelationDensityFactorBlock() = default;
    PeriodicCorrelationDensityFactorKind kind_ = PeriodicCorrelationDensityFactorKind::OccupiedOccupied;
    PeriodicCorrelationDensityFactorMemoryPlan memory_;
    std::uint64_t q_ = 0, auxiliary_begin_ = 0;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> values_;
    std::vector<PeriodicCorrelationPlacedOccupied> left_, right_;
    PeriodicCorrelationVirtualBlockSelection left_virtual_, right_virtual_;
    std::string identity_, payload_, selection_, source_, whitener_, store_, consumed_;
    friend PeriodicCorrelationDensityFactorBlock build_periodic_correlation_occupied_density_factor_block(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t,
        const std::uint64_t*, std::size_t, std::size_t, const std::uint64_t*, std::size_t, std::size_t,
        std::uint64_t, std::uint64_t, std::uint64_t, const PeriodicCorrelationLocalFactorCaps&);
    friend PeriodicCorrelationDensityFactorBlock build_periodic_correlation_virtual_density_factor_block(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationVirtualBlockSelection&,
        const PeriodicCorrelationVirtualBlockSelection&, std::uint64_t, std::uint64_t, std::uint64_t,
        const PeriodicCorrelationLocalFactorCaps&);
};

/// No size-dependent allocation, index/gauge scan or tile read. OO counts
/// must be positive and no greater than Nk*nactive; uniqueness is checked
/// during construction after the explicit work and byte caps are admitted.
PeriodicCorrelationDensityFactorMemoryPlan plan_periodic_correlation_occupied_density_factor_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    std::uint64_t left_count, std::uint64_t right_count, std::uint64_t q_index,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count);
PeriodicCorrelationDensityFactorMemoryPlan plan_periodic_correlation_virtual_density_factor_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationVirtualBlockSelection& left,
    const PeriodicCorrelationVirtualBlockSelection& right, std::uint64_t q_index,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count);

/// Nonowning aligned contiguous OO lists contain uint64 [count,2], columns
/// (active occupied index, canonical cell). Accessible counts are uint64
/// elements. Lists remain immutable during the call. Reject duplicates within
/// each list; overlap between left/right lists is allowed and includes l=r.
/// Retained index bytes are included in maximum_owned_numerical_bytes.
PeriodicCorrelationDensityFactorBlock build_periodic_correlation_occupied_density_factor_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    const std::uint64_t* left_indices, std::size_t left_accessible_count, std::size_t left_count,
    const std::uint64_t* right_indices, std::size_t right_accessible_count, std::size_t right_count,
    std::uint64_t q_index, std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count,
    const PeriodicCorrelationLocalFactorCaps& caps);
PeriodicCorrelationDensityFactorBlock build_periodic_correlation_virtual_density_factor_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationVirtualBlockSelection& left,
    const PeriodicCorrelationVirtualBlockSelection& right, std::uint64_t q_index,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count,
    const PeriodicCorrelationLocalFactorCaps& caps);

}  // namespace vibeqc
