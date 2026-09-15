#pragma once

// Native finite-image Gaussian overlaps feeding one-k Bloch IAOs.
// Sun et al., doi:10.1063/1.4998644, Eqs.10/16; Zhu and Tew,
// doi:10.1021/acs.jpca.4c04555, Eqs.13-20. Explicit AO/minimal bases;
// no minimal-basis default, infinite-image error bound, or HF Hamiltonian
// authentication. An S11 numerical match is not an AO-basis content receipt
// from the mean-field producer. All occupied bands, including frozen, enter
// the existing retained-space IAO constructor unchanged.

#include "vibeqc/periodic_correlation_bloch_iao.hpp"
#include "vibeqc/periodic_gaussian_source_context.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicGaussianBlochIAOVersion = 1U;

struct PeriodicGaussianBlochIAOOptions {
    double image_cutoff_bohr = 0.0;
    double geometry_absolute_tolerance = 0.0, geometry_relative_tolerance = 0.0;
    double overlap_absolute_tolerance = 0.0, overlap_relative_tolerance = 0.0;
    double structural_absolute_tolerance = 0.0, structural_relative_tolerance = 0.0;
    // Maximum correction to ANY S12/S22 element during audited Hermitian
    // and (at TRIM only) real projection. No correction to admitted S11.
    double projection_absolute_tolerance = 0.0, projection_relative_tolerance = 0.0;
};

struct PeriodicGaussianBlochIAOCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_work_units = 0;
    std::uint64_t maximum_atom_count = 0, maximum_shell_count = 0;
    std::uint64_t maximum_contraction_count = 0, maximum_primitive_numeric_lanes = 0;
    std::uint64_t maximum_basis_content_wire_bytes = 0, maximum_borrowed_active_numeric_bytes = 0;
    std::uint64_t maximum_panel_pairs = 0, maximum_total_image_candidates = 0;
};

struct PeriodicGaussianBlochIAOMemoryPlan {
    std::uint64_t n_basis = 0, n_effective = 0, n_occupied = 0, n_minimal = 0;
    std::uint64_t point = 0, conjugate_point = 0, panel_pairs = 0, panel_calls = 0;
    PeriodicGaussianBasisInventory ao, minimal;
    // Both basis roles are inventoried, even if they alias the same owner.
    // Active lanes only, NOT allocator capacity or C++ object overhead.
    std::uint64_t borrowed_basis_numeric_bytes = 0, borrowed_geometry_numeric_bytes = 0;
    std::uint64_t other_live_numerical_bytes = 0, total_borrowed_numerical_bytes = 0;
    std::uint64_t overlap_and_label_bytes = 0, panel_output_bytes = 0;
    std::uint64_t fixed_fourier_workspace_bytes = 0, iao_owned_peak_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, output_numerical_bytes = 0;
    std::uint64_t required_node_memory_bytes = 0;
    std::uint64_t image_candidate_count = 0, retained_pair_image_count = 0;
    // Preflight is a conservative reservation using the supplied total
    // candidate cap; evaluation uses counted candidates. Counts include
    // every S11/S12/S22/S21/partner stream, not only retained output values.
    std::uint64_t basis_scan_work_units = 0, preflight_work_units = 0;
    std::uint64_t overlap_evaluation_work_units = 0, iao_work_units = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicGaussianBlochIAODiagnostics {
    std::uint64_t charged_work_units = 0, evaluated_panel_count = 0;
    std::uint64_t evaluated_image_candidate_count = 0, evaluated_pair_image_count = 0;
    double maximum_geometry_residual = 0.0, maximum_s11_reference_residual = 0.0;
    double maximum_s11_reference_conjugacy_residual = 0.0;
    double maximum_cross_adjoint_residual = 0.0, maximum_overlap_time_reversal_residual = 0.0;
    double maximum_minimal_hermitian_defect = 0.0, maximum_trim_imaginary_magnitude = 0.0;
    double maximum_projection_correction = 0.0;
};

// Owns exactly one native IAO, not S11/S12/S22, bases, geometry, seed, or
// other points. Its .iao() reference remains valid only while this owner is
// live and unmoved. Python uses reference_internal to retain that lifetime.
class PeriodicGaussianBlochIAOPoint {
public:
    PeriodicGaussianBlochIAOPoint(const PeriodicGaussianBlochIAOPoint&) = delete;
    PeriodicGaussianBlochIAOPoint& operator=(const PeriodicGaussianBlochIAOPoint&) = delete;
    PeriodicGaussianBlochIAOPoint(PeriodicGaussianBlochIAOPoint&&) noexcept = default;
    PeriodicGaussianBlochIAOPoint& operator=(PeriodicGaussianBlochIAOPoint&&) noexcept = default;
    const PeriodicCorrelationBlochIAO& iao() const;
    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianBlochIAOVersion; }
    bool hf_basis_source_authenticated() const noexcept { return false; }
    bool infinite_image_tail_certified() const noexcept { return false; }
    const PeriodicGaussianBlochIAOMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianBlochIAODiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const PeriodicGaussianBlochIAOOptions& options() const noexcept { return options_; }
    const std::string& ao_basis_identity_sha256() const noexcept { return ao_identity_; }
    const std::string& minimal_basis_identity_sha256() const noexcept { return minimal_identity_; }
    const std::string& source_identity_sha256() const noexcept { return source_identity_; }
    const std::string& raw_overlap_payload_sha256() const noexcept { return raw_identity_; }
    const std::string& reference_match_identity_sha256() const noexcept { return match_identity_; }
private:
    explicit PeriodicGaussianBlochIAOPoint(PeriodicCorrelationBlochIAO&&);
    PeriodicCorrelationBlochIAO iao_;
    PeriodicGaussianBlochIAOMemoryPlan memory_;
    PeriodicGaussianBlochIAODiagnostics diagnostics_;
    PeriodicGaussianBlochIAOOptions options_;
    std::string ao_identity_, minimal_identity_, source_identity_, raw_identity_, match_identity_;
    friend PeriodicGaussianBlochIAOPoint make_periodic_gaussian_bloch_iao(
        const PeriodicCorrelationAdmittedReference&, const BasisSet&, const BasisSet&,
        const PeriodicSystem&, std::size_t, std::uint64_t,
        const PeriodicGaussianBlochIAOOptions&, const PeriodicCorrelationBlochIAOOptions&,
        const PeriodicGaussianBlochIAOCaps&);
};

// No variable numerical allocation: validate/census actual bases and count
// all finite-image panels using a zero-vector Fourier view. The original
// direct cell is used, never reconstructed from the state's reciprocal.
// Positive byte, scan/work and candidate caps precede enumeration. Repeated
// basis scans and flattened-AO lookups are charged. A later factory repeats
// this preflight; plans are not reusable authentication of mutable inputs.
PeriodicGaussianBlochIAOMemoryPlan plan_periodic_gaussian_bloch_iao(
    const PeriodicCorrelationAdmittedReference&, const BasisSet& ao, const BasisSet& minimal,
    const PeriodicSystem&, std::size_t point, std::uint64_t other_live_numerical_bytes,
    const PeriodicGaussianBlochIAOOptions&, const PeriodicCorrelationBlochIAOOptions&,
    const PeriodicGaussianBlochIAOCaps&);

// p=0 Fourier overlap with positive ket phase and NO Nk/volume factor.
// S11 is streamed against the admitted overlap; cross-adjoint and opposite-k
// sources are independently regenerated. Raw defects precede any audited
// S22 Hermitian or TRIM real projection. Source SHA authenticates actual
// Gaussian inputs/finite policy/raw values; reference-match SHA separately
// binds the admitted state and measured audits, without an HF source claim.
// Other live owners (e.g. preceding point IAOs and seed), source capacities,
// and unrelated numerical buffers must be admitted via the explicit other
// inventory or reference external inventory. No callback/input mutation is
// allowed during this synchronous call. No hidden Eigen/BLAS dense work.
// Fixed scalar geometry, SHA/string/control storage and return metadata are
// separate backend costs, not counted as variable numerical array payload.
PeriodicGaussianBlochIAOPoint make_periodic_gaussian_bloch_iao(
    const PeriodicCorrelationAdmittedReference&, const BasisSet& ao, const BasisSet& minimal,
    const PeriodicSystem&, std::size_t point, std::uint64_t other_live_numerical_bytes,
    const PeriodicGaussianBlochIAOOptions&, const PeriodicCorrelationBlochIAOOptions&,
    const PeriodicGaussianBlochIAOCaps&);
}  // namespace vibeqc
