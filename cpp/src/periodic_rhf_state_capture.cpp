#include "vibeqc/periodic_rhf_state_capture.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {

namespace {

bool is_lower_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char digit) {
        return (digit >= '0' && digit <= '9')
            || (digit >= 'a' && digit <= 'f');
    });
}

bool finite_complex_matrix(const PeriodicMeanFieldComplexMatrix& matrix) {
    return matrix.real().allFinite() && matrix.imag().allFinite();
}

bool within_scaled_limit(double residual,
                         double absolute,
                         double relative,
                         double scale) {
    if (!std::isfinite(residual) || !std::isfinite(scale)) return false;
    const double limit = absolute + relative * std::max(1.0, scale);
    return std::isfinite(limit) && residual <= limit;
}

void require_square_shape(const PeriodicMeanFieldComplexMatrix& matrix,
                          Eigen::Index expected,
                          const char* label) {
    if (matrix.rows() != expected || matrix.cols() != expected) {
        throw std::invalid_argument(
            std::string("periodic RHF capture ") + label
            + " must be a common square AO matrix");
    }
}

}  // namespace

PeriodicRHFStateCapturePreflight preflight_periodic_rhf_state_capture(
    const PeriodicRHFStateCaptureRequest& request,
    int periodic_dimension,
    const std::array<int, 3>& mesh,
    const std::array<int, 3>& is_shift,
    const Eigen::Matrix3d& reciprocal_lattice,
    const std::vector<Eigen::Vector3d>& kpoints,
    const std::vector<double>& weights,
    bool symmetry_reduced_or_reconstructed,
    std::uint64_t n_basis,
    std::uint64_t electrons_per_cell) {
    if (periodic_dimension < 1 || periodic_dimension > 3) {
        throw std::invalid_argument(
            "periodic RHF capture periodic_dimension must be 1, 2, or 3");
    }
    for (int axis = periodic_dimension; axis < 3; ++axis) {
        if (mesh[static_cast<std::size_t>(axis)] != 1
            || is_shift[static_cast<std::size_t>(axis)] != 0) {
            throw std::invalid_argument(
                "periodic RHF capture inactive mesh axes must have extent "
                "one and zero shift");
        }
    }
    if (!is_lower_sha256(request.calculation_identity)) {
        throw std::invalid_argument(
            "periodic RHF capture calculation_identity must be a lowercase "
            "SHA-256-shaped 64-character hexadecimal label");
    }
    if (request.maximum_retained_numerical_payload_bytes == 0U) {
        throw std::invalid_argument(
            "periodic RHF capture requires a positive retained numerical "
            "payload byte limit");
    }
    if (!std::isfinite(request.minimum_band_gap_hartree)
        || request.minimum_band_gap_hartree <= 0.0) {
        throw std::invalid_argument(
            "periodic RHF capture minimum_band_gap_hartree must be finite "
            "and strictly positive");
    }
    if (symmetry_reduced_or_reconstructed) {
        throw std::invalid_argument(
            "periodic RHF capture contract v1 requires a directly computed "
            "full Brillouin-zone mesh; symmetry-reduced or reconstructed "
            "input is not accepted");
    }
    if (n_basis == 0U || electrons_per_cell == 0U
        || electrons_per_cell % 2U != 0U) {
        throw std::invalid_argument(
            "periodic RHF capture requires a nonzero AO basis and a positive "
            "even electron count");
    }
    const std::uint64_t n_occupied = electrons_per_cell / 2U;
    if (n_occupied >= n_basis) {
        throw std::invalid_argument(
            "periodic RHF capture requires at least one virtual band");
    }
    if (!reciprocal_lattice.allFinite()) {
        throw std::invalid_argument(
            "periodic RHF capture reciprocal lattice contains a non-finite "
            "value");
    }

    const PeriodicMeanFieldValidationTolerances tolerances;
    const double reciprocal_scale = reciprocal_lattice.cwiseAbs().maxCoeff();
    double relative_volume = 0.0;
    if (reciprocal_scale > 0.0) {
        relative_volume = std::abs(
            (reciprocal_lattice / reciprocal_scale).determinant());
    }
    if (!std::isfinite(relative_volume)
        || !(relative_volume
             > tolerances.reciprocal_lattice_relative_volume_floor)) {
        throw std::invalid_argument(
            "periodic RHF capture reciprocal lattice must have three "
            "linearly independent vectors");
    }

    const RegularKMesh addressing(mesh, is_shift);
    if (kpoints.size() != addressing.size()
        || weights.size() != addressing.size()) {
        throw std::invalid_argument(
            "periodic RHF capture requires every point and weight of the "
            "regular full Brillouin-zone mesh");
    }
    const double uniform_weight =
        1.0 / static_cast<double>(addressing.size());
    for (std::size_t index = 0; index < addressing.size(); ++index) {
        if (!kpoints[index].allFinite()) {
            throw std::invalid_argument(
                "periodic RHF capture k-point coordinate is non-finite");
        }
        const Eigen::Vector3d expected =
            reciprocal_lattice * addressing.fractional_at(index);
        const double coordinate_residual =
            (kpoints[index] - expected).cwiseAbs().maxCoeff();
        const double coordinate_scale = std::max(
            kpoints[index].cwiseAbs().maxCoeff(),
            expected.cwiseAbs().maxCoeff());
        if (!within_scaled_limit(
                coordinate_residual,
                tolerances.coordinate_absolute,
                tolerances.coordinate_relative,
                coordinate_scale)) {
            throw std::invalid_argument(
                "periodic RHF capture k points are not in exact "
                "RegularKMesh order");
        }
        if (!std::isfinite(weights[index]) || weights[index] <= 0.0) {
            throw std::invalid_argument(
                "periodic RHF capture integration weight is not finite and "
                "positive");
        }
        const double weight_residual =
            std::abs(weights[index] - uniform_weight);
        if (!within_scaled_limit(
                weight_residual,
                tolerances.scalar_absolute,
                tolerances.scalar_relative,
                uniform_weight)) {
            throw std::invalid_argument(
                "periodic RHF capture requires exact uniform full-BZ "
                "weights");
        }
    }

    if (n_basis > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "periodic RHF capture basis dimension exceeds size_t");
    }
    const std::size_t n_basis_size = static_cast<std::size_t>(n_basis);
    if (request.frozen_core_mask_per_k.size() != addressing.size()) {
        throw std::invalid_argument(
            "periodic RHF capture requires one explicit frozen-core mask per "
            "full-mesh k point");
    }
    std::uint64_t common_frozen = 0;
    bool have_common_frozen = false;
    for (std::size_t kpoint = 0; kpoint < addressing.size(); ++kpoint) {
        const auto& mask = request.frozen_core_mask_per_k[kpoint];
        if (mask.size() != n_basis_size) {
            throw std::invalid_argument(
                "periodic RHF capture frozen-core mask length must equal "
                "the full band count");
        }
        std::uint64_t frozen = 0;
        for (std::size_t orbital = 0; orbital < mask.size(); ++orbital) {
            if (mask[orbital] > 1U) {
                throw std::invalid_argument(
                    "periodic RHF capture frozen-core masks must be binary");
            }
            if (mask[orbital] != 0U) {
                if (orbital >= n_occupied) {
                    throw std::invalid_argument(
                        "periodic RHF capture cannot freeze a virtual band");
                }
                ++frozen;
            }
        }
        if (!have_common_frozen) {
            common_frozen = frozen;
            have_common_frozen = true;
        } else if (frozen != common_frozen) {
            throw std::invalid_argument(
                "periodic RHF capture frozen-core rank must be identical at "
                "every k point");
        }
    }
    if (common_frozen >= n_occupied) {
        throw std::invalid_argument(
            "periodic RHF capture requires at least one correlated occupied "
            "band");
    }

    const auto retained_bytes =
        estimate_periodic_restricted_mean_field_resident_bytes(
            mesh, n_basis, n_basis);
    if (retained_bytes
        > request.maximum_retained_numerical_payload_bytes) {
        throw std::runtime_error(
            "periodic RHF capture retained numerical payload requires "
            + std::to_string(retained_bytes)
            + " bytes, above the explicit limit of "
            + std::to_string(
                request.maximum_retained_numerical_payload_bytes)
            + " bytes");
    }

    return PeriodicRHFStateCapturePreflight{
        retained_bytes,
        common_frozen,
        n_occupied - common_frozen,
        n_basis - n_occupied,
    };
}

PeriodicRHFPhysicalDensityDiagnostics periodic_rhf_physical_density_diagnostics(
    const PeriodicMeanFieldComplexMatrix& overlap,
    const PeriodicMeanFieldComplexMatrix& physical_fock,
    const PeriodicMeanFieldComplexMatrix& spin_summed_density) {
    const Eigen::Index n_basis = overlap.rows();
    require_square_shape(overlap, n_basis, "overlap");
    require_square_shape(physical_fock, n_basis, "physical Fock");
    require_square_shape(spin_summed_density, n_basis, "density");
    if (n_basis == 0 || !finite_complex_matrix(overlap)
        || !finite_complex_matrix(physical_fock)
        || !finite_complex_matrix(spin_summed_density)) {
        throw std::invalid_argument(
            "periodic RHF capture physical-density inputs must be nonempty "
            "and finite");
    }

    PeriodicMeanFieldComplexMatrix left =
        physical_fock * spin_summed_density * overlap;
    PeriodicMeanFieldComplexMatrix right =
        overlap * spin_summed_density * physical_fock;
    const double left_scale = left.norm();
    const double right_scale = right.norm();
    left -= right;
    const double commutator_norm = left.norm();

    left = spin_summed_density * overlap * spin_summed_density;
    right = 2.0 * spin_summed_density;
    const double idempotency_left_scale = left.norm();
    const double idempotency_right_scale = right.norm();
    left -= right;
    const double idempotency_norm = left.norm();
    const double commutator_scale = std::max(
        {1.0, left_scale, right_scale});
    const double idempotency_scale = std::max(
        {1.0, idempotency_left_scale, idempotency_right_scale});
    const double roundoff_allowance =
        64.0 * std::numeric_limits<double>::epsilon()
        * static_cast<double>(n_basis) * commutator_scale;
    const double idempotency_relative =
        idempotency_norm / idempotency_scale;
    if (!std::isfinite(commutator_norm)
        || !std::isfinite(roundoff_allowance)
        || !std::isfinite(idempotency_norm)
        || !std::isfinite(idempotency_relative)) {
        throw std::overflow_error(
            "periodic RHF capture physical-density residual overflow");
    }
    return {
        commutator_norm,
        roundoff_allowance,
        idempotency_norm,
        idempotency_relative,
    };
}

PeriodicRHFDensityFixedPointDiagnostics
periodic_rhf_density_fixed_point_diagnostics(
    const PeriodicMeanFieldComplexMatrix& spin_summed_density,
    const PeriodicMeanFieldComplexMatrix& coefficients,
    std::uint64_t n_occupied) {
    const Eigen::Index n_basis = spin_summed_density.rows();
    require_square_shape(spin_summed_density, n_basis, "density");
    if (n_basis == 0 || coefficients.rows() != n_basis
        || coefficients.cols() != n_basis
        || n_occupied == 0U
        || n_occupied >= static_cast<std::uint64_t>(n_basis)
        || !finite_complex_matrix(spin_summed_density)
        || !finite_complex_matrix(coefficients)) {
        throw std::invalid_argument(
            "periodic RHF capture fixed-point inputs have incompatible "
            "dimensions or non-finite values");
    }
    const Eigen::Index n_occ = static_cast<Eigen::Index>(n_occupied);
    const PeriodicMeanFieldComplexMatrix canonical_density =
        2.0 * coefficients.leftCols(n_occ)
            * coefficients.leftCols(n_occ).adjoint();
    const double residual = (spin_summed_density - canonical_density).norm();
    const double scale = std::max(
        {1.0, spin_summed_density.norm(), canonical_density.norm()});
    const double relative = residual / scale;
    if (!std::isfinite(residual) || !std::isfinite(relative)) {
        throw std::overflow_error(
            "periodic RHF capture fixed-point residual overflow");
    }
    return {residual, relative};
}

}  // namespace vibeqc
