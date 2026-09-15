#pragma once

/// \file periodic_correlation_admitted_reference.hpp
/// \brief Fail-closed bridge from a certified RHF state to correlation work.
///
/// This object is the only contract-v1 entry point from a retained periodic
/// RHF state into native local-correlation construction.  It owns immutable
/// shared state, copies the static dimensions and resource budget, adds the
/// state's exact retained numerical bytes to the caller's external inventory,
/// and stores a static plan produced internally by the resource planner.
///
/// Version 1 is deliberately in-process and nonserializable: the mean-field
/// state carries canonicalized payload and provenance digests, but no checkpoint
/// schema, loader, or load-time verification contract exists yet. It also
/// rejects symmetry reduction until a verified full-mesh reconstruction and
/// orbital-sewing object exists.

#include <cstdint>
#include <memory>

#include "vibeqc/periodic_correlation_resource.hpp"
#include "vibeqc/periodic_mean_field_state.hpp"

namespace vibeqc {

inline constexpr std::uint32_t
    kPeriodicCorrelationAdmittedReferenceContractVersion = 1;

class PeriodicCorrelationAdmittedReference {
public:
    PeriodicCorrelationAdmittedReference(
        const PeriodicCorrelationAdmittedReference&) = delete;
    PeriodicCorrelationAdmittedReference& operator=(
        const PeriodicCorrelationAdmittedReference&) = delete;
    PeriodicCorrelationAdmittedReference(
        PeriodicCorrelationAdmittedReference&&) noexcept = default;
    PeriodicCorrelationAdmittedReference& operator=(
        PeriodicCorrelationAdmittedReference&&) noexcept = default;
    ~PeriodicCorrelationAdmittedReference() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationAdmittedReferenceContractVersion;
    }
    const PeriodicRestrictedMeanFieldState& state() const noexcept {
        return *state_;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept {
        return state_;
    }
    const PeriodicCorrelationStaticDimensions& dimensions() const noexcept {
        return dimensions_;
    }
    const PeriodicCorrelationResourceBudget& budget() const noexcept {
        return budget_;
    }
    const PeriodicCorrelationResourcePlan& plan() const noexcept {
        return plan_;
    }
    PeriodicCorrelationByteCount state_resident_bytes() const noexcept {
        return state_resident_bytes_;
    }

private:
    PeriodicCorrelationAdmittedReference(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        PeriodicCorrelationStaticDimensions dimensions,
        PeriodicCorrelationResourceBudget budget,
        PeriodicCorrelationResourcePlan plan,
        PeriodicCorrelationByteCount state_resident_bytes);

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationStaticDimensions dimensions_;
    PeriodicCorrelationResourceBudget budget_;
    PeriodicCorrelationResourcePlan plan_;
    PeriodicCorrelationByteCount state_resident_bytes_ = 0;

    friend PeriodicCorrelationAdmittedReference
    make_periodic_correlation_admitted_reference(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        PeriodicCorrelationStaticDimensions dimensions,
        PeriodicCorrelationResourceBudget budget);
};

/// Validate a certified state against the complete static inventory and admit
/// the next native pair-domain-census stage.  A caller-supplied resource plan
/// is intentionally impossible: this factory always computes and checks it.
PeriodicCorrelationAdmittedReference
make_periodic_correlation_admitted_reference(
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
    PeriodicCorrelationStaticDimensions dimensions,
    PeriodicCorrelationResourceBudget budget);

}  // namespace vibeqc
