#pragma once

/// \file periodic_mean_field_state.hpp
/// \brief Validated full-Brillouin-zone restricted mean-field snapshots.
///
/// Contract version 1 owns one complete regular k mesh in the exact
/// last-axis-fast order of RegularKMesh.  It deliberately rejects an
/// irreducible-zone-only input: orbital symmetry transport is not part of
/// this contract and an IBZ representative map alone cannot reconstruct an
/// AO or MO gauge at the omitted points.

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {

using PeriodicMeanFieldComplexMatrix =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

inline constexpr std::uint32_t
    kPeriodicRestrictedMeanFieldStateContractVersion = 1;
inline constexpr std::uint32_t
    kPeriodicMeanFieldStateDigestVersion = 2;
inline constexpr std::uint32_t
    kPeriodicMeanFieldValidationToleranceVersion = 1;

/// Bloch normalization and Brillouin-zone integration convention.
///
/// The sole v1 value is the convention used by vibe-qc's native periodic
/// kernels: AO Bloch matrices are unnormalized lattice sums and full-mesh
/// integrals carry equal 1/Nk weights.  The regular-grid coordinates are the
/// Monkhorst-Pack construction (Monkhorst and Pack, 1976,
/// doi:10.1103/PhysRevB.13.5188), with vibe-qc's independently selected
/// per-axis half-step flags.
enum class PeriodicMeanFieldNormalizationConvention {
    UnnormalizedAoBlochSumsUniformFullBzWeights,
};

/// Reference type accepted by the restricted state contract.
enum class PeriodicMeanFieldReferenceKind {
    RestrictedHartreeFock,
};

/// Fixed, versioned numerical gates for state contract v1.
///
/// Callers select only the version.  They cannot relax individual checks and
/// thereby label incompatible snapshots with the same contract version.
struct PeriodicMeanFieldValidationTolerances {
    std::uint32_t version = kPeriodicMeanFieldValidationToleranceVersion;
    double matrix_absolute = 1.0e-10;
    double matrix_relative = 1.0e-8;
    double coordinate_absolute = 5.0e-13;
    double coordinate_relative = 5.0e-13;
    double scalar_absolute = 1.0e-12;
    double scalar_relative = 1.0e-10;
    double reciprocal_lattice_relative_volume_floor = 1.0e-12;
    double overlap_eigenvalue_relative_floor = 1.0e-10;
};

/// Maximum raw residuals measured while validating a snapshot.
///
/// Matrix diagnostics are maximum absolute element residuals.  Cartesian
/// coordinates use the component infinity norm; scalar diagnostics are
/// absolute differences.  Each gate is residual <= absolute + relative *
/// max(1, scale), where scale is the largest magnitude of the compared
/// operands.  The reciprocal-lattice volume floor is applied to
/// abs(det(B / max(abs(B)))).  band_gap_hartree is the global indirect gap,
/// min_{k,a} epsilon_a(k) - max_{k,i} epsilon_i(k).
struct PeriodicMeanFieldValidationDiagnostics {
    double maximum_kpoint_cartesian_residual = 0.0;
    double maximum_weight_residual = 0.0;
    double maximum_overlap_hermiticity_residual = 0.0;
    double maximum_fock_hermiticity_residual = 0.0;
    double minimum_overlap_eigenvalue = 0.0;
    double maximum_metric_orthonormality_residual = 0.0;
    double maximum_roothaan_residual = 0.0;
    double maximum_energy_order_violation = 0.0;
    double maximum_occupation_residual = 0.0;
    double electron_count_residual = 0.0;
    double band_gap_hartree = 0.0;
};

class PeriodicRestrictedMeanFieldState;

/// Mutable assembly object.  Per-k storage is append-only and intentionally
/// has no writable vector property on the Python boundary.
class PeriodicRestrictedMeanFieldInput {
public:
    PeriodicRestrictedMeanFieldInput() = default;
    PeriodicRestrictedMeanFieldInput(
        const PeriodicRestrictedMeanFieldInput&) = delete;
    PeriodicRestrictedMeanFieldInput& operator=(
        const PeriodicRestrictedMeanFieldInput&) = delete;
    PeriodicRestrictedMeanFieldInput(
        PeriodicRestrictedMeanFieldInput&&) noexcept = default;
    PeriodicRestrictedMeanFieldInput& operator=(
        PeriodicRestrictedMeanFieldInput&&) noexcept = default;
    ~PeriodicRestrictedMeanFieldInput() = default;

    /// The lowercase SHA-256-shaped calculation identity also used by
    /// PeriodicCorrelationStaticDimensions. It seals the physical periodic
    /// dimension, cell and AO ordering, orbital and auxiliary bases,
    /// finite-periodic Hamiltonian and singularity treatment, exact k mesh,
    /// RHF root, thresholds, and the complete frozen/correlated/virtual
    /// selection. It is caller-supplied calculation provenance, not a digest
    /// of this particular converged matrix payload.
    std::string calculation_identity;
    PeriodicMeanFieldReferenceKind reference_kind =
        PeriodicMeanFieldReferenceKind::RestrictedHartreeFock;
    PeriodicMeanFieldNormalizationConvention normalization =
        PeriodicMeanFieldNormalizationConvention::
            UnnormalizedAoBlochSumsUniformFullBzWeights;
    /// Physical periodicity: 1 for polymers, 2 for slabs, or 3 for bulk.
    /// Zero is an unset, fail-closed assembly default.
    int periodic_dimension = 0;
    std::array<int, 3> mesh = {1, 1, 1};
    std::array<int, 3> is_shift = {0, 0, 0};
    /// Reciprocal-lattice vectors are columns, in bohr^-1, and include 2*pi.
    /// Point i must be the unwrapped product B * fractional_at(i).
    Eigen::Matrix3d reciprocal_lattice = Eigen::Matrix3d::Identity();
    /// Only the native SCF capture path may assert this for production use.
    /// The supplied Fock matrices must be the physical Hermitian F[D]
    /// operators associated with the stored orbitals, without DIIS
    /// extrapolation, mixing, or a level shift.  A producer that projects
    /// finite-domain raw matrices must separately fail closed on any material
    /// pre-projection Hermiticity defect.
    bool converged = false;
    /// Must be false in contract v1.  A shorter point list is rejected too,
    /// so failing to set this flag cannot smuggle an IBZ-only state through.
    bool symmetry_reduced_input = false;
    /// Must also be false.  V1 does not accept reconstructed full meshes until
    /// the AO representation and orbital-gauge transport are separately sealed.
    bool symmetry_reconstructed_input = false;
    std::uint64_t n_basis = 0;
    std::uint64_t n_effective_orbitals = 0;
    /// Closed-shell electron count.  V1 requires a positive even integer.
    std::uint64_t electrons_per_cell = 0;
    /// Converged RHF reference energy per primitive cell, in Hartree.
    double reference_energy_per_cell = 0.0;
    /// Required lower bound for the global insulating band gap, in Hartree.
    double minimum_band_gap_hartree = 0.0;
    std::uint32_t validation_tolerance_version =
        kPeriodicMeanFieldValidationToleranceVersion;

    /// Append the next point in RegularKMesh order.  Each mask must be binary;
    /// the three masks are disjoint and exhaustive, frozen/correlated entries
    /// carry occupation 2, virtual entries carry occupation 0, and their
    /// respective counts must be identical at every k point.
    void add_kpoint(
        Eigen::Vector3d k_cartesian,
        double weight,
        PeriodicMeanFieldComplexMatrix overlap,
        PeriodicMeanFieldComplexMatrix fock,
        PeriodicMeanFieldComplexMatrix coefficients,
        Eigen::VectorXd orbital_energies,
        Eigen::VectorXd occupations,
        std::vector<std::uint8_t> frozen_core_mask,
        std::vector<std::uint8_t> correlated_occupied_mask,
        std::vector<std::uint8_t> virtual_mask);

    std::size_t kpoint_count() const noexcept { return kpoints_.size(); }

private:
    struct KPointInput {
        Eigen::Vector3d k_cartesian;
        double weight = 0.0;
        PeriodicMeanFieldComplexMatrix overlap;
        PeriodicMeanFieldComplexMatrix fock;
        PeriodicMeanFieldComplexMatrix coefficients;
        Eigen::VectorXd orbital_energies;
        Eigen::VectorXd occupations;
        std::vector<std::uint8_t> frozen_core_mask;
        std::vector<std::uint8_t> correlated_occupied_mask;
        std::vector<std::uint8_t> virtual_mask;
    };

    std::vector<KPointInput> kpoints_;

    friend std::shared_ptr<const PeriodicRestrictedMeanFieldState>
    make_periodic_restricted_mean_field_state(
        PeriodicRestrictedMeanFieldInput&& input);
};

/// Immutable owner of one validated restricted mean-field state.
///
/// The C++ API exposes const references to one k-point block at a time.  The
/// internal Python binding returns copies of those individual blocks, so a
/// caller cannot mutate the stored snapshot or request an accidental all-k
/// copy. The validated object is neither copyable nor movable after its
/// factory installs it behind shared_ptr<const>; this prevents a mutable alias
/// from replacing or moving out the payload after resource admission.
/// Digest version 2 seals the canonicalized logical numerical payload and the
/// declared provenance separately. These are content identities suitable for
/// future checkpoint verification, not physical-wavefunction equivalence
/// tests: independent SCF runs may differ by orbital phases, degenerate
/// rotations, or last-bit arithmetic.
class PeriodicRestrictedMeanFieldState {
public:
    PeriodicRestrictedMeanFieldState(
        const PeriodicRestrictedMeanFieldState&) = delete;
    PeriodicRestrictedMeanFieldState& operator=(
        const PeriodicRestrictedMeanFieldState&) = delete;
    PeriodicRestrictedMeanFieldState(
        PeriodicRestrictedMeanFieldState&&) = delete;
    PeriodicRestrictedMeanFieldState& operator=(
        PeriodicRestrictedMeanFieldState&&) = delete;
    ~PeriodicRestrictedMeanFieldState() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicRestrictedMeanFieldStateContractVersion;
    }
    bool converged() const noexcept { return true; }
    const std::string& calculation_identity() const noexcept {
        return calculation_identity_;
    }
    std::uint32_t digest_version() const noexcept {
        return kPeriodicMeanFieldStateDigestVersion;
    }
    const std::string& numerical_payload_sha256() const noexcept {
        return numerical_payload_sha256_;
    }
    const std::string& state_identity_sha256() const noexcept {
        return state_identity_sha256_;
    }
    PeriodicMeanFieldNormalizationConvention normalization() const noexcept {
        return normalization_;
    }
    PeriodicMeanFieldReferenceKind reference_kind() const noexcept {
        return reference_kind_;
    }
    int periodic_dimension() const noexcept { return periodic_dimension_; }
    const std::array<int, 3>& mesh() const noexcept {
        return addressing_.mesh();
    }
    const std::array<int, 3>& is_shift() const noexcept {
        return addressing_.is_shift();
    }
    const Eigen::Matrix3d& reciprocal_lattice() const noexcept {
        return reciprocal_lattice_;
    }
    std::size_t n_kpoints() const noexcept { return kpoints_.size(); }
    std::uint64_t n_basis() const noexcept { return n_basis_; }
    std::uint64_t n_effective_orbitals() const noexcept {
        return n_effective_orbitals_;
    }
    std::uint64_t n_frozen_core() const noexcept { return n_frozen_core_; }
    std::uint64_t n_correlated_occupied() const noexcept {
        return n_correlated_occupied_;
    }
    std::uint64_t n_virtual() const noexcept { return n_virtual_; }
    std::uint64_t electrons_per_cell() const noexcept {
        return electrons_per_cell_;
    }
    double reference_energy_per_cell() const noexcept {
        return reference_energy_per_cell_;
    }
    double requested_minimum_band_gap_hartree() const noexcept {
        return requested_minimum_band_gap_hartree_;
    }
    double band_gap_hartree() const noexcept {
        return diagnostics_.band_gap_hartree;
    }
    double uniform_weight() const noexcept { return uniform_weight_; }
    std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }
    const PeriodicMeanFieldValidationTolerances& validation_tolerances()
        const noexcept {
        return validation_tolerances_;
    }
    const PeriodicMeanFieldValidationDiagnostics& diagnostics()
        const noexcept {
        return diagnostics_;
    }

    const Eigen::Vector3d& kpoint_cartesian(std::size_t index) const;
    double weight(std::size_t index) const;
    const PeriodicMeanFieldComplexMatrix& overlap(std::size_t index) const;
    const PeriodicMeanFieldComplexMatrix& fock(std::size_t index) const;
    const PeriodicMeanFieldComplexMatrix& coefficients(
        std::size_t index) const;
    const Eigen::VectorXd& orbital_energies(std::size_t index) const;
    const Eigen::VectorXd& occupations(std::size_t index) const;
    const std::vector<std::uint8_t>& frozen_core_mask(
        std::size_t index) const;
    const std::vector<std::uint8_t>& correlated_occupied_mask(
        std::size_t index) const;
    const std::vector<std::uint8_t>& virtual_mask(std::size_t index) const;

private:
    struct KPointBlock {
        Eigen::Vector3d k_cartesian;
        double weight = 0.0;
        PeriodicMeanFieldComplexMatrix overlap;
        PeriodicMeanFieldComplexMatrix fock;
        PeriodicMeanFieldComplexMatrix coefficients;
        Eigen::VectorXd orbital_energies;
        Eigen::VectorXd occupations;
        std::vector<std::uint8_t> frozen_core_mask;
        std::vector<std::uint8_t> correlated_occupied_mask;
        std::vector<std::uint8_t> virtual_mask;
    };

    PeriodicRestrictedMeanFieldState(
        std::string calculation_identity,
        std::string numerical_payload_sha256,
        std::string state_identity_sha256,
        PeriodicMeanFieldReferenceKind reference_kind,
        PeriodicMeanFieldNormalizationConvention normalization,
        int periodic_dimension,
        RegularKMesh addressing,
        Eigen::Matrix3d reciprocal_lattice,
        std::uint64_t n_basis,
        std::uint64_t n_effective_orbitals,
        std::uint64_t n_frozen_core,
        std::uint64_t n_correlated_occupied,
        std::uint64_t n_virtual,
        std::uint64_t electrons_per_cell,
        double reference_energy_per_cell,
        double requested_minimum_band_gap_hartree,
        double uniform_weight,
        std::uint64_t resident_bytes,
        PeriodicMeanFieldValidationTolerances validation_tolerances,
        PeriodicMeanFieldValidationDiagnostics diagnostics,
        std::vector<KPointBlock> kpoints);

    const KPointBlock& block(std::size_t index) const;

    std::string calculation_identity_;
    std::string numerical_payload_sha256_;
    std::string state_identity_sha256_;
    PeriodicMeanFieldReferenceKind reference_kind_;
    PeriodicMeanFieldNormalizationConvention normalization_;
    int periodic_dimension_ = 0;
    RegularKMesh addressing_;
    Eigen::Matrix3d reciprocal_lattice_;
    std::uint64_t n_basis_ = 0;
    std::uint64_t n_effective_orbitals_ = 0;
    std::uint64_t n_frozen_core_ = 0;
    std::uint64_t n_correlated_occupied_ = 0;
    std::uint64_t n_virtual_ = 0;
    std::uint64_t electrons_per_cell_ = 0;
    double reference_energy_per_cell_ = 0.0;
    double requested_minimum_band_gap_hartree_ = 0.0;
    double uniform_weight_ = 0.0;
    std::uint64_t resident_bytes_ = 0;
    PeriodicMeanFieldValidationTolerances validation_tolerances_;
    PeriodicMeanFieldValidationDiagnostics diagnostics_;
    std::vector<KPointBlock> kpoints_;

    friend std::shared_ptr<const PeriodicRestrictedMeanFieldState>
    make_periodic_restricted_mean_field_state(
        PeriodicRestrictedMeanFieldInput&& input);
};

/// Validate and move an assembly object into immutable shared state.
std::shared_ptr<const PeriodicRestrictedMeanFieldState>
make_periodic_restricted_mean_field_state(
    PeriodicRestrictedMeanFieldInput&& input);

/// Checked byte count for every bulk numerical field retained by v1.
///
/// The count includes the reciprocal lattice and, for every k point, the
/// Cartesian vector, integration weight, S/F/C matrices, orbital energies,
/// occupations, and all three uint8 masks.  Fixed C++ object/string/allocator
/// bookkeeping is excluded because it is implementation-dependent.  No
/// container or matrix is allocated by this function.
std::uint64_t estimate_periodic_restricted_mean_field_resident_bytes(
    std::array<int, 3> mesh,
    std::uint64_t n_basis,
    std::uint64_t n_effective_orbitals);

}  // namespace vibeqc
