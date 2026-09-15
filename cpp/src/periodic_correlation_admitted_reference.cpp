#include "vibeqc/periodic_correlation_admitted_reference.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace vibeqc {

namespace {

const char* admission_name(PeriodicCorrelationAdmissionCode admission) {
    switch (admission) {
        case PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus:
            return "ReadyForPairDomainCensus";
        case PeriodicCorrelationAdmissionCode::ReadyForTripleDomainCensus:
            return "ReadyForTripleDomainCensus";
        case PeriodicCorrelationAdmissionCode::Admitted:
            return "Admitted";
        case PeriodicCorrelationAdmissionCode::MissingMemoryLimit:
            return "MissingMemoryLimit";
        case PeriodicCorrelationAdmissionCode::MissingScratchLimit:
            return "MissingScratchLimit";
        case PeriodicCorrelationAdmissionCode::MemoryExceeded:
            return "MemoryExceeded";
        case PeriodicCorrelationAdmissionCode::ScratchExceeded:
            return "ScratchExceeded";
        case PeriodicCorrelationAdmissionCode::ArithmeticOverflow:
            return "ArithmeticOverflow";
    }
    return "Unknown";
}

void require_state_dimensions_match(
    const PeriodicRestrictedMeanFieldState& state,
    const PeriodicCorrelationStaticDimensions& dimensions) {
    if (state.contract_version()
        != kPeriodicRestrictedMeanFieldStateContractVersion) {
        throw std::invalid_argument(
            "periodic correlation reference has an unsupported mean-field "
            "state contract version");
    }
    if (!state.converged()) {
        throw std::invalid_argument(
            "periodic correlation reference requires a converged "
            "mean-field state");
    }
    if (state.reference_kind()
        != PeriodicMeanFieldReferenceKind::RestrictedHartreeFock) {
        throw std::invalid_argument(
            "periodic correlation reference contract v1 requires RHF");
    }
    if (state.normalization()
        != PeriodicMeanFieldNormalizationConvention::
            UnnormalizedAoBlochSumsUniformFullBzWeights) {
        throw std::invalid_argument(
            "periodic correlation reference normalization does not match "
            "contract v1");
    }
    if (dimensions.symmetry_reduction_requested) {
        throw std::invalid_argument(
            "periodic correlation admitted-reference contract v1 rejects "
            "symmetry reduction until a native verified full-mesh "
            "reconstruction and orbital-sewing object exists");
    }
    if (dimensions.calculation_identity != state.calculation_identity()) {
        throw std::invalid_argument(
            "periodic correlation calculation identity does not match the "
            "mean-field state");
    }
    if (dimensions.periodic_dimension != state.periodic_dimension()) {
        throw std::invalid_argument(
            "periodic correlation periodic dimension does not match the "
            "mean-field state");
    }
    if (dimensions.mesh != state.mesh()) {
        throw std::invalid_argument(
            "periodic correlation mesh does not match the mean-field state");
    }
    if (dimensions.is_shift != state.is_shift()) {
        throw std::invalid_argument(
            "periodic correlation k-point shift does not match the "
            "mean-field state");
    }
    if (dimensions.n_kpoints
        != static_cast<std::uint64_t>(state.n_kpoints())) {
        throw std::invalid_argument(
            "periodic correlation k-point count does not match the "
            "mean-field state");
    }
    if (dimensions.n_basis != state.n_basis()) {
        throw std::invalid_argument(
            "periodic correlation AO-basis size does not match the "
            "mean-field state");
    }
    if (dimensions.n_effective_orbitals
        != state.n_effective_orbitals()) {
        throw std::invalid_argument(
            "periodic correlation effective-orbital count does not match "
            "the mean-field state");
    }
    const auto frozen = state.n_frozen_core();
    const auto correlated = state.n_correlated_occupied();
    if (correlated > std::numeric_limits<std::uint64_t>::max() - frozen) {
        throw std::logic_error(
            "validated mean-field occupied count overflows uint64");
    }
    const auto total_occupied = frozen + correlated;
    if (dimensions.n_home_total_occupied != total_occupied
        || total_occupied != state.electrons_per_cell() / 2U) {
        throw std::invalid_argument(
            "periodic correlation total occupied count does not match the "
            "mean-field frozen-core partition");
    }
    if (dimensions.n_home_occupied != correlated) {
        throw std::invalid_argument(
            "periodic correlation correlated occupied count does not match "
            "the mean-field masks");
    }
    if (dimensions.n_home_virtual != state.n_virtual()) {
        throw std::invalid_argument(
            "periodic correlation virtual count does not match the "
            "mean-field masks");
    }
    if (dimensions.n_spin_channels != 1U) {
        throw std::invalid_argument(
            "periodic correlation admitted-reference contract v1 is "
            "closed-shell only");
    }
}

void require_ready_plan(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationResourceBudget& budget,
    const PeriodicCorrelationResourcePlan& plan) {
    if (plan.admission
        != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus) {
        std::string message =
            "periodic correlation reference was not admitted: static "
            "resource planner returned ";
        message += admission_name(plan.admission);
        message += "; required_memory_bytes="
            + std::to_string(plan.required_memory_bytes)
            + ", memory_limit_bytes="
            + std::to_string(budget.memory_limit_bytes)
            + ", required_scratch_bytes="
            + std::to_string(plan.required_scratch_bytes)
            + ", scratch_limit_bytes="
            + std::to_string(budget.scratch_limit_bytes);
        if (!plan.failure_detail.empty()) {
            message += " (" + plan.failure_detail + ")";
        }
        if (plan.admission
            == PeriodicCorrelationAdmissionCode::ArithmeticOverflow) {
            throw std::overflow_error(message);
        }
        throw std::runtime_error(message);
    }
    if (plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.allocation_contract_version
            != dimensions.allocation_contract_version
        || plan.calculation_identity != dimensions.calculation_identity
        || plan.allocation_identity != dimensions.allocation_identity
        || !plan.census_identity.empty()
        || plan.pair_domain_census_complete
        || plan.triple_domain_census_complete) {
        throw std::logic_error(
            "periodic correlation static planner returned an inconsistent "
            "admission seal");
    }
}

}  // namespace

PeriodicCorrelationAdmittedReference::
    PeriodicCorrelationAdmittedReference(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        PeriodicCorrelationStaticDimensions dimensions,
        PeriodicCorrelationResourceBudget budget,
        PeriodicCorrelationResourcePlan plan,
        PeriodicCorrelationByteCount state_resident_bytes)
    : state_(std::move(state)),
      dimensions_(std::move(dimensions)),
      budget_(std::move(budget)),
      plan_(std::move(plan)),
      state_resident_bytes_(state_resident_bytes) {}

PeriodicCorrelationAdmittedReference
make_periodic_correlation_admitted_reference(
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
    PeriodicCorrelationStaticDimensions dimensions,
    PeriodicCorrelationResourceBudget budget) {
    if (!state) {
        throw std::invalid_argument(
            "periodic correlation admitted reference requires a non-null "
            "mean-field state");
    }
    require_state_dimensions_match(*state, dimensions);

    const auto state_resident_bytes = state->resident_bytes();
    if (state_resident_bytes
        > std::numeric_limits<PeriodicCorrelationByteCount>::max()
              - dimensions.external_bytes) {
        throw std::overflow_error(
            "periodic correlation admitted-reference external byte "
            "accounting overflow");
    }
    dimensions.external_bytes += state_resident_bytes;

    auto plan = plan_periodic_correlation_static_resources(
        dimensions, budget);
    require_ready_plan(dimensions, budget, plan);
    return PeriodicCorrelationAdmittedReference(
        std::move(state),
        std::move(dimensions),
        std::move(budget),
        std::move(plan),
        state_resident_bytes);
}

static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationAdmittedReference>::value,
    "an admitted periodic correlation reference must never copy its state");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationAdmittedReference>::value,
    "an admitted periodic correlation reference must be cheaply movable");

}  // namespace vibeqc
