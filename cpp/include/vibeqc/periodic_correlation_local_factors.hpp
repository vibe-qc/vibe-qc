#pragma once

/// Selected local orbitals and global-auxiliary RI factors on the finite torus.
/// Sun et al., doi:10.1063/1.4998644, Eqs. (3),(4),(13),(16),(17),(21),
/// uses unnormalized AO Bloch sums and the primitive-cell Coulomb kernel.
/// Nejad et al., doi:10.1063/5.0290816, Eqs. (2)-(10),(26)-(28), fixes
/// the inverse-Bloch 1/Nk convention for local occupied/PAO coefficients.
/// Consequently L_ia(q)=Nk^(-3/2) sum_k,mu,nu c_i(mu,k)^* c_a(nu,k+q)
/// B(P,mu,nu;k,q). This is GLOBAL auxiliary RI in selected local orbital
/// spaces, not the local chargeless auxiliary-domain DF of Nejad Eqs.45-65.
///
/// For complex orbitals (i* a|j* b)=sum_q,P conj(L_ai(q))*L_jb(q).
/// OV and VO are computed independently. No q/-q, real-orbital, or source
/// conjugation shortcut is assumed; imaginary components are never dropped.
/// Current store factors have an explicitly finite AO-image source, without
/// an infinite-image tail certificate or a production correlation claim.
/// Known live Wannier/gauge/domain/space/reader payloads are charged explicitly.
/// Other live caller allocations must already be in the admitted inventory.
///
/// Digests use length-prefixed strings, big-endian u32/u64 and binary64
/// real/imaginary lanes with signed zero normalized. Every domain starts with
/// u32 version 1. Coefficient payload: domain
/// "vibeqc.periodic.correlation.local-coefficients.payload", nao,V,ko,kv
/// as u64, then column-major complex values. Factor payload: domain
/// "vibeqc.periodic.correlation.local-factors.payload", q,aux_begin,Ab,V
/// as u64, orientation u32, then row-major complex values. Consumed-tile
/// payload: domain "vibeqc.periodic.correlation.local-factors.consumed-tiles",
/// store SHA string, visit count u64, then (sequence u64,tile SHA string)
/// in visitation order. Identity domains end in ".identity" and bind the
/// state/calculation/allocation/Wannier/gauge/domain/space SHA strings and
/// Selection fields in declaration order, then their explicit k/q/slice,
/// source/payload and fixed normalization/contraction-policy strings.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_factor_store.hpp"
#include "vibeqc/periodic_correlation_pao_space.hpp"
#include "vibeqc/periodic_correlation_wannier.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationLocalFactorContractVersion = 1;

enum class PeriodicCorrelationLocalFactorOrientation : std::uint32_t {
    OccupiedVirtual = 0, VirtualOccupied = 1,
};

/// Cell indices are canonical modular addresses, never signed aliases.
/// The virtual translation applies to every selected PAO domain column;
/// the occupied translation is independent. No q-only phase replaces either.
struct PeriodicCorrelationLocalOrbitalSelection {
    std::uint64_t occupied_index = 0;
    std::uint64_t occupied_cell = 0;
    std::uint64_t virtual_begin = 0;
    std::uint64_t virtual_count = 0;
    std::uint64_t virtual_translation_cell = 0;
};

struct PeriodicCorrelationLocalFactorCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    /// Conservative loop/contraction units, not CPU time or floating FLOPs.
    /// Includes input hashing, coefficient expansion, tile payload traversal,
    /// factor multiply-adds and output hashing. Positive and mandatory.
    std::uint64_t maximum_work_units = 0;
    /// Required only by the factor consumer, not the coefficient primitive.
    std::uint64_t maximum_tile_visits = 0;
    std::uint64_t maximum_reader_tile_bytes = 0;
};

struct PeriodicCorrelationLocalFactorMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_home_occupied = 0;
    std::uint64_t domain_dimension = 0, retained_virtual_dimension = 0;
    std::uint64_t virtual_count = 0, auxiliary_count = 0;
    std::uint64_t coefficient_panel_bytes = 0;
    std::uint64_t coefficient_scratch_bytes = 0;
    std::uint64_t retained_output_bytes = 0;
    std::uint64_t compensation_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0;
    std::uint64_t live_wannier_bytes = 0;
    std::uint64_t live_domain_bytes = 0;
    std::uint64_t live_space_bytes = 0;
    std::uint64_t live_reader_numeric_bytes = 0;
    std::uint64_t live_reader_control_bytes = 0;
    std::uint64_t maximum_reader_tile_bytes = 0;
    std::uint64_t tile_visits = 0;
    std::uint64_t reader_payload_bytes = 0;
    std::uint64_t work_units = 0;
    std::uint64_t required_node_memory_bytes = 0;
};

/// Panel layout: column-major nao*(1+virtual_count), occupied column first.
/// The occupied and virtual k addresses may differ. Only a compact panel is
/// retained, no all-k orbitals or complete AO projector.
class PeriodicCorrelationLocalCoefficientPanel {
public:
    PeriodicCorrelationLocalCoefficientPanel(const PeriodicCorrelationLocalCoefficientPanel&) = delete;
    PeriodicCorrelationLocalCoefficientPanel& operator=(const PeriodicCorrelationLocalCoefficientPanel&) = delete;
    PeriodicCorrelationLocalCoefficientPanel(PeriodicCorrelationLocalCoefficientPanel&&) noexcept = default;
    PeriodicCorrelationLocalCoefficientPanel& operator=(PeriodicCorrelationLocalCoefficientPanel&&) noexcept = default;
    const PeriodicCorrelationLocalFactorMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationLocalOrbitalSelection& selection() const noexcept { return selection_; }
    std::uint64_t occupied_k_index() const noexcept { return occupied_k_; }
    std::uint64_t virtual_k_index() const noexcept { return virtual_k_; }
    std::complex<double> coefficient(std::size_t ao, std::size_t column) const;
    const std::complex<double>* data() const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
private:
    PeriodicCorrelationLocalCoefficientPanel() = default;
    PeriodicCorrelationLocalFactorMemoryPlan memory_;
    PeriodicCorrelationLocalOrbitalSelection selection_;
    std::uint64_t occupied_k_ = 0, virtual_k_ = 0;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> values_;
    std::string identity_, payload_;
    friend PeriodicCorrelationLocalCoefficientPanel make_periodic_correlation_local_coefficient_panel(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationLocalOrbitalSelection&,
        std::uint64_t, std::uint64_t, const PeriodicCorrelationLocalFactorCaps&);
};

/// One original-auxiliary-AO slice, row-major [auxiliary_count,virtual_count].
/// Translation, orientation and source identities are part of the identity.
class PeriodicCorrelationLocalFactorBlock {
public:
    PeriodicCorrelationLocalFactorBlock(const PeriodicCorrelationLocalFactorBlock&) = delete;
    PeriodicCorrelationLocalFactorBlock& operator=(const PeriodicCorrelationLocalFactorBlock&) = delete;
    PeriodicCorrelationLocalFactorBlock(PeriodicCorrelationLocalFactorBlock&&) noexcept = default;
    PeriodicCorrelationLocalFactorBlock& operator=(PeriodicCorrelationLocalFactorBlock&&) noexcept = default;
    const PeriodicCorrelationLocalFactorMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationLocalOrbitalSelection& selection() const noexcept { return selection_; }
    std::uint64_t q_index() const noexcept { return q_; }
    std::uint64_t auxiliary_begin() const noexcept { return auxiliary_begin_; }
    PeriodicCorrelationLocalFactorOrientation orientation() const noexcept { return orientation_; }
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    std::complex<double> element(std::size_t auxiliary_row, std::size_t virtual_column) const;
    const std::complex<double>* data() const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& source_identity_sha256() const noexcept { return source_; }
    const std::string& whitener_payload_identity_sha256() const noexcept { return whitener_; }
    const std::string& store_identity_sha256() const noexcept { return store_; }
    const std::string& consumed_tiles_identity_sha256() const noexcept { return consumed_; }
private:
    PeriodicCorrelationLocalFactorBlock() = default;
    PeriodicCorrelationLocalFactorMemoryPlan memory_;
    PeriodicCorrelationLocalOrbitalSelection selection_;
    PeriodicCorrelationLocalFactorOrientation orientation_ = PeriodicCorrelationLocalFactorOrientation::OccupiedVirtual;
    std::uint64_t q_ = 0, auxiliary_begin_ = 0;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> values_;
    std::string identity_, payload_, source_, whitener_, store_, consumed_;
    friend PeriodicCorrelationLocalFactorBlock build_periodic_correlation_local_factor_block(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationLocalOrbitalSelection&,
        std::uint64_t, std::uint64_t, std::uint64_t, PeriodicCorrelationLocalFactorOrientation,
        const PeriodicCorrelationLocalFactorCaps&);
};

/// Object-based preflights with no size-dependent allocations. They check
/// structural identities and all shape/work overflows, but do not hash caller
/// gauges or visit tiles. Fixed-size identity strings may allocate.
PeriodicCorrelationLocalFactorMemoryPlan plan_periodic_correlation_local_coefficient_panel(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationLocalOrbitalSelection&);
PeriodicCorrelationLocalFactorMemoryPlan plan_periodic_correlation_local_factor_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationLocalOrbitalSelection&, std::uint64_t q_index,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count);

/// Nonowning aligned contiguous gauges [Nk,nactive,nactive] must remain
/// immutable. Exact content must match the sealed Wannier gauge. Q(k) uses
/// every retained virtual MO; frozen AND active occupied MOs are excluded.
/// Admission and positive work caps precede size-dependent scans/allocations.
PeriodicCorrelationLocalCoefficientPanel make_periodic_correlation_local_coefficient_panel(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationLocalOrbitalSelection&, std::uint64_t occupied_k_index,
    std::uint64_t virtual_k_index, const PeriodicCorrelationLocalFactorCaps&);
PeriodicCorrelationLocalFactorBlock build_periodic_correlation_local_factor_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationLocalOrbitalSelection&, std::uint64_t q_index,
    std::uint64_t auxiliary_begin, std::uint64_t auxiliary_count,
    PeriodicCorrelationLocalFactorOrientation, const PeriodicCorrelationLocalFactorCaps&);

}  // namespace vibeqc
