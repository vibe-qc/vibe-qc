#include "vibeqc/periodic_mean_field_state.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {

namespace {

using ByteCount = std::uint64_t;

class CanonicalStateHasher {
public:
    explicit CanonicalStateHasher(const std::string& domain) {
        add_string(domain);
        add_u32(kPeriodicMeanFieldStateDigestVersion);
    }

    void add_u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> encoded{};
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            encoded[i] = static_cast<std::uint8_t>(
                value >> (24U - 8U * i));
        }
        hasher_.update(encoded.data(), encoded.size());
    }

    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            encoded[i] = static_cast<std::uint8_t>(
                value >> (56U - 8U * i));
        }
        hasher_.update(encoded.data(), encoded.size());
    }

    void add_double(double value) {
        if (!std::isfinite(value)) {
            throw std::logic_error(
                "validated periodic mean-field state contains a non-finite "
                "value during digest construction");
        }
        if (value == 0.0) value = 0.0;
        static_assert(std::numeric_limits<double>::is_iec559,
                      "periodic state digest requires IEEE-754 arithmetic");
        static_assert(sizeof(double) == sizeof(std::uint64_t),
                      "periodic state digest requires IEEE-754 binary64");
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        add_u64(bits);
    }

    void add_complex(std::complex<double> value) {
        add_double(value.real());
        add_double(value.imag());
    }

    void add_string(const std::string& value) {
        add_u64(static_cast<std::uint64_t>(value.size()));
        hasher_.update(
            reinterpret_cast<const std::uint8_t*>(value.data()),
            value.size());
    }

    void add_vector3(const Eigen::Vector3d& vector) {
        add_u64(3U);
        for (Eigen::Index index = 0; index < vector.size(); ++index) {
            add_double(vector[index]);
        }
    }

    void add_matrix3(const Eigen::Matrix3d& matrix) {
        add_u64(3U);
        add_u64(3U);
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
                add_double(matrix(row, column));
            }
        }
    }

    void add_complex_matrix(const PeriodicMeanFieldComplexMatrix& matrix) {
        add_u64(static_cast<std::uint64_t>(matrix.rows()));
        add_u64(static_cast<std::uint64_t>(matrix.cols()));
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
                add_complex(matrix(row, column));
            }
        }
    }

    void add_real_vector(const Eigen::VectorXd& vector) {
        add_u64(static_cast<std::uint64_t>(vector.size()));
        for (Eigen::Index index = 0; index < vector.size(); ++index) {
            add_double(vector[index]);
        }
    }

    void add_mask(const std::vector<std::uint8_t>& mask) {
        add_u64(static_cast<std::uint64_t>(mask.size()));
        hasher_.update(mask.data(), mask.size());
    }

    std::string finish() { return hasher_.finish_hex(); }

private:
    detail::Sha256 hasher_;
};

const char* reference_kind_token(PeriodicMeanFieldReferenceKind kind) {
    switch (kind) {
        case PeriodicMeanFieldReferenceKind::RestrictedHartreeFock:
            return "restricted-hartree-fock";
    }
    throw std::logic_error("unsupported periodic mean-field reference kind");
}

const char* normalization_token(
    PeriodicMeanFieldNormalizationConvention normalization) {
    switch (normalization) {
        case PeriodicMeanFieldNormalizationConvention::
            UnnormalizedAoBlochSumsUniformFullBzWeights:
            return "unnormalized-ao-bloch-sums-uniform-full-bz-weights";
    }
    throw std::logic_error(
        "unsupported periodic mean-field normalization convention");
}

std::string make_state_identity_sha256(
    const std::string& calculation_identity,
    PeriodicMeanFieldReferenceKind reference_kind,
    PeriodicMeanFieldNormalizationConvention normalization,
    double requested_minimum_band_gap_hartree,
    const PeriodicMeanFieldValidationTolerances& tolerances,
    const std::string& numerical_payload_sha256) {
    CanonicalStateHasher hasher(
        "vibeqc.periodic.restricted-mean-field-state.identity");
    hasher.add_u32(kPeriodicRestrictedMeanFieldStateContractVersion);
    hasher.add_string(calculation_identity);
    hasher.add_string(reference_kind_token(reference_kind));
    hasher.add_string(normalization_token(normalization));
    hasher.add_double(requested_minimum_band_gap_hartree);
    hasher.add_u32(tolerances.version);
    hasher.add_double(tolerances.matrix_absolute);
    hasher.add_double(tolerances.matrix_relative);
    hasher.add_double(tolerances.coordinate_absolute);
    hasher.add_double(tolerances.coordinate_relative);
    hasher.add_double(tolerances.scalar_absolute);
    hasher.add_double(tolerances.scalar_relative);
    hasher.add_double(tolerances.reciprocal_lattice_relative_volume_floor);
    hasher.add_double(tolerances.overlap_eigenvalue_relative_floor);
    hasher.add_string(numerical_payload_sha256);
    return hasher.finish();
}

ByteCount checked_add(ByteCount left,
                      ByteCount right,
                      const char* context) {
    if (left > std::numeric_limits<ByteCount>::max() - right) {
        throw std::overflow_error(
            std::string("periodic mean-field resident-byte overflow in ") +
            context);
    }
    return left + right;
}

ByteCount checked_multiply(ByteCount left,
                           ByteCount right,
                           const char* context) {
    if (left != 0 && right > std::numeric_limits<ByteCount>::max() / left) {
        throw std::overflow_error(
            std::string("periodic mean-field resident-byte overflow in ") +
            context);
    }
    return left * right;
}

bool is_lower_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const char character : value) {
        const bool digit = character >= '0' && character <= '9';
        const bool lower_hex = character >= 'a' && character <= 'f';
        if (!digit && !lower_hex) return false;
    }
    return true;
}

bool finite_complex_matrix(const PeriodicMeanFieldComplexMatrix& matrix) {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            const std::complex<double> value = matrix(row, column);
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
                return false;
            }
        }
    }
    return true;
}

bool finite_real_vector(const Eigen::VectorXd& vector) {
    for (Eigen::Index index = 0; index < vector.size(); ++index) {
        if (!std::isfinite(vector[index])) return false;
    }
    return true;
}

bool finite_vector3(const Eigen::Vector3d& vector) {
    return std::isfinite(vector[0]) && std::isfinite(vector[1]) &&
           std::isfinite(vector[2]);
}

bool finite_matrix3(const Eigen::Matrix3d& matrix) {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            if (!std::isfinite(matrix(row, column))) return false;
        }
    }
    return true;
}

double matrix_max_abs(const PeriodicMeanFieldComplexMatrix& matrix) {
    double maximum = 0.0;
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            const double magnitude = std::abs(matrix(row, column));
            if (!std::isfinite(magnitude)) {
                return std::numeric_limits<double>::infinity();
            }
            maximum = std::max(maximum, magnitude);
        }
    }
    return maximum;
}

double vector_max_abs(const Eigen::VectorXd& vector) {
    double maximum = 0.0;
    for (Eigen::Index index = 0; index < vector.size(); ++index) {
        const double magnitude = std::abs(vector[index]);
        if (!std::isfinite(magnitude)) {
            return std::numeric_limits<double>::infinity();
        }
        maximum = std::max(maximum, magnitude);
    }
    return maximum;
}

bool within_scaled_limit(double residual,
                         double absolute,
                         double relative,
                         double scale) {
    if (!std::isfinite(residual) || !std::isfinite(scale)) return false;
    const double relative_term = relative * std::max(1.0, scale);
    const double limit = absolute + relative_term;
    return std::isfinite(relative_term) && std::isfinite(limit) &&
           residual <= limit;
}

std::string kpoint_context(std::size_t index, const char* quantity) {
    return "periodic restricted mean-field k point " +
           std::to_string(index) + " has invalid " + quantity;
}

void require_shape(const PeriodicMeanFieldComplexMatrix& matrix,
                   Eigen::Index rows,
                   Eigen::Index columns,
                   std::size_t kpoint,
                   const char* name) {
    if (matrix.rows() != rows || matrix.cols() != columns) {
        throw std::invalid_argument(kpoint_context(kpoint, name) +
                                    " shape");
    }
}

void require_vector_shape(const Eigen::VectorXd& vector,
                          Eigen::Index size,
                          std::size_t kpoint,
                          const char* name) {
    if (vector.size() != size) {
        throw std::invalid_argument(kpoint_context(kpoint, name) +
                                    " length");
    }
}

void require_mask_shape_and_values(const std::vector<std::uint8_t>& mask,
                                   std::size_t size,
                                   std::size_t kpoint,
                                   const char* name) {
    if (mask.size() != size) {
        throw std::invalid_argument(kpoint_context(kpoint, name) +
                                    " length");
    }
    for (const std::uint8_t value : mask) {
        if (value != 0U && value != 1U) {
            throw std::invalid_argument(kpoint_context(kpoint, name) +
                                        " value (expected only 0 or 1)");
        }
    }
}

double hermiticity_residual(const PeriodicMeanFieldComplexMatrix& matrix) {
    double maximum = 0.0;
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            const double residual = std::abs(
                matrix(row, column) - std::conj(matrix(column, row)));
            if (!std::isfinite(residual)) {
                return std::numeric_limits<double>::infinity();
            }
            maximum = std::max(maximum, residual);
        }
    }
    return maximum;
}

double identity_residual(const PeriodicMeanFieldComplexMatrix& matrix) {
    double maximum = 0.0;
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            const std::complex<double> expected =
                row == column ? std::complex<double>(1.0, 0.0)
                              : std::complex<double>(0.0, 0.0);
            const double residual = std::abs(matrix(row, column) - expected);
            if (!std::isfinite(residual)) {
                return std::numeric_limits<double>::infinity();
            }
            maximum = std::max(maximum, residual);
        }
    }
    return maximum;
}

}  // namespace

static_assert(
    !std::is_copy_constructible<PeriodicRestrictedMeanFieldInput>::value,
    "the full-k mean-field input must never copy its matrix payload");
static_assert(
    !std::is_copy_constructible<PeriodicRestrictedMeanFieldState>::value,
    "the full-k mean-field state must never copy its matrix payload");
static_assert(
    !std::is_move_constructible<PeriodicRestrictedMeanFieldState>::value
        && !std::is_move_assignable<PeriodicRestrictedMeanFieldState>::value,
    "the validated full-k mean-field state must be immutable");

void PeriodicRestrictedMeanFieldInput::add_kpoint(
    Eigen::Vector3d k_cartesian,
    double weight,
    PeriodicMeanFieldComplexMatrix overlap,
    PeriodicMeanFieldComplexMatrix fock,
    PeriodicMeanFieldComplexMatrix coefficients,
    Eigen::VectorXd orbital_energies,
    Eigen::VectorXd occupations,
    std::vector<std::uint8_t> frozen_core_mask,
    std::vector<std::uint8_t> correlated_occupied_mask,
    std::vector<std::uint8_t> virtual_mask) {
    KPointInput point;
    point.k_cartesian = std::move(k_cartesian);
    point.weight = weight;
    point.overlap = std::move(overlap);
    point.fock = std::move(fock);
    point.coefficients = std::move(coefficients);
    point.orbital_energies = std::move(orbital_energies);
    point.occupations = std::move(occupations);
    point.frozen_core_mask = std::move(frozen_core_mask);
    point.correlated_occupied_mask = std::move(correlated_occupied_mask);
    point.virtual_mask = std::move(virtual_mask);
    kpoints_.push_back(std::move(point));
}

std::uint64_t estimate_periodic_restricted_mean_field_resident_bytes(
    std::array<int, 3> mesh,
    std::uint64_t n_basis,
    std::uint64_t n_effective_orbitals) {
    if (n_basis == 0) {
        throw std::invalid_argument(
            "periodic mean-field byte estimate requires n_basis > 0");
    }
    if (n_effective_orbitals == 0 || n_effective_orbitals > n_basis) {
        throw std::invalid_argument(
            "periodic mean-field byte estimate requires 0 < "
            "n_effective_orbitals <= n_basis");
    }

    const RegularKMesh addressing(mesh);
    if (addressing.size() > std::numeric_limits<ByteCount>::max()) {
        throw std::overflow_error(
            "periodic mean-field k-point count exceeds uint64_t");
    }
    const ByteCount n_kpoints =
        static_cast<ByteCount>(addressing.size());
    const ByteCount basis_square =
        checked_multiply(n_basis, n_basis, "basis square");
    const ByteCount coefficient_elements = checked_multiply(
        n_basis, n_effective_orbitals, "coefficient elements");
    const ByteCount square_matrix_elements = checked_multiply(
        2U, basis_square, "overlap and Fock elements");
    const ByteCount complex_elements = checked_add(
        square_matrix_elements,
        coefficient_elements,
        "complex elements per k point");
    const ByteCount complex_bytes = checked_multiply(
        complex_elements,
        static_cast<ByteCount>(sizeof(std::complex<double>)),
        "complex bytes per k point");

    const ByteCount orbital_real_elements = checked_multiply(
        2U, n_effective_orbitals, "energy and occupation elements");
    const ByteCount kpoint_real_elements = checked_add(
        orbital_real_elements, 4U, "real elements per k point");
    const ByteCount real_bytes = checked_multiply(
        kpoint_real_elements,
        static_cast<ByteCount>(sizeof(double)),
        "real bytes per k point");
    const ByteCount mask_bytes = checked_multiply(
        3U,
        checked_multiply(
            n_effective_orbitals,
            static_cast<ByteCount>(sizeof(std::uint8_t)),
            "one mask per k point"),
        "three masks per k point");

    ByteCount per_kpoint = checked_add(
        complex_bytes, real_bytes, "complex and real bytes per k point");
    per_kpoint = checked_add(
        per_kpoint, mask_bytes, "all bytes per k point");
    const ByteCount all_kpoints = checked_multiply(
        n_kpoints, per_kpoint, "full-mesh retained fields");
    const ByteCount reciprocal_lattice_bytes = checked_multiply(
        9U,
        static_cast<ByteCount>(sizeof(double)),
        "reciprocal lattice bytes");
    return checked_add(
        all_kpoints,
        reciprocal_lattice_bytes,
        "full resident numerical payload");
}

PeriodicRestrictedMeanFieldState::PeriodicRestrictedMeanFieldState(
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
    std::vector<KPointBlock> kpoints)
    : calculation_identity_(std::move(calculation_identity)),
      numerical_payload_sha256_(std::move(numerical_payload_sha256)),
      state_identity_sha256_(std::move(state_identity_sha256)),
      reference_kind_(reference_kind),
      normalization_(normalization),
      periodic_dimension_(periodic_dimension),
      addressing_(std::move(addressing)),
      reciprocal_lattice_(std::move(reciprocal_lattice)),
      n_basis_(n_basis),
      n_effective_orbitals_(n_effective_orbitals),
      n_frozen_core_(n_frozen_core),
      n_correlated_occupied_(n_correlated_occupied),
      n_virtual_(n_virtual),
      electrons_per_cell_(electrons_per_cell),
      reference_energy_per_cell_(reference_energy_per_cell),
      requested_minimum_band_gap_hartree_(
          requested_minimum_band_gap_hartree),
      uniform_weight_(uniform_weight),
      resident_bytes_(resident_bytes),
      validation_tolerances_(validation_tolerances),
      diagnostics_(diagnostics),
      kpoints_(std::move(kpoints)) {}

const PeriodicRestrictedMeanFieldState::KPointBlock&
PeriodicRestrictedMeanFieldState::block(std::size_t index) const {
    if (index >= kpoints_.size()) {
        throw std::out_of_range(
            "periodic restricted mean-field k-point index " +
            std::to_string(index) + " is outside [0, " +
            std::to_string(kpoints_.size()) + ")");
    }
    return kpoints_[index];
}

const Eigen::Vector3d& PeriodicRestrictedMeanFieldState::kpoint_cartesian(
    std::size_t index) const {
    return block(index).k_cartesian;
}

double PeriodicRestrictedMeanFieldState::weight(std::size_t index) const {
    return block(index).weight;
}

const PeriodicMeanFieldComplexMatrix&
PeriodicRestrictedMeanFieldState::overlap(std::size_t index) const {
    return block(index).overlap;
}

const PeriodicMeanFieldComplexMatrix&
PeriodicRestrictedMeanFieldState::fock(std::size_t index) const {
    return block(index).fock;
}

const PeriodicMeanFieldComplexMatrix&
PeriodicRestrictedMeanFieldState::coefficients(std::size_t index) const {
    return block(index).coefficients;
}

const Eigen::VectorXd& PeriodicRestrictedMeanFieldState::orbital_energies(
    std::size_t index) const {
    return block(index).orbital_energies;
}

const Eigen::VectorXd& PeriodicRestrictedMeanFieldState::occupations(
    std::size_t index) const {
    return block(index).occupations;
}

const std::vector<std::uint8_t>&
PeriodicRestrictedMeanFieldState::frozen_core_mask(std::size_t index) const {
    return block(index).frozen_core_mask;
}

const std::vector<std::uint8_t>&
PeriodicRestrictedMeanFieldState::correlated_occupied_mask(
    std::size_t index) const {
    return block(index).correlated_occupied_mask;
}

const std::vector<std::uint8_t>&
PeriodicRestrictedMeanFieldState::virtual_mask(std::size_t index) const {
    return block(index).virtual_mask;
}

std::shared_ptr<const PeriodicRestrictedMeanFieldState>
make_periodic_restricted_mean_field_state(
    PeriodicRestrictedMeanFieldInput&& input) {
    if (!is_lower_sha256(input.calculation_identity)) {
        throw std::invalid_argument(
            "periodic restricted mean-field calculation_identity must be a "
            "lowercase SHA-256 digest");
    }
    if (input.reference_kind !=
        PeriodicMeanFieldReferenceKind::RestrictedHartreeFock) {
        throw std::invalid_argument(
            "periodic restricted mean-field state contract v1 accepts only "
            "a restricted Hartree-Fock reference");
    }
    if (input.normalization !=
        PeriodicMeanFieldNormalizationConvention::
            UnnormalizedAoBlochSumsUniformFullBzWeights) {
        throw std::invalid_argument(
            "periodic restricted mean-field state contract v1 accepts only "
            "unnormalized AO Bloch sums with uniform full-BZ weights");
    }
    if (input.periodic_dimension < 1 || input.periodic_dimension > 3) {
        throw std::invalid_argument(
            "periodic restricted mean-field periodic_dimension must be "
            "1, 2, or 3");
    }
    for (int axis = input.periodic_dimension; axis < 3; ++axis) {
        if (input.mesh[static_cast<std::size_t>(axis)] != 1
            || input.is_shift[static_cast<std::size_t>(axis)] != 0) {
            throw std::invalid_argument(
                "periodic restricted mean-field inactive mesh axes must "
                "have extent one and zero shift");
        }
    }
    if (input.validation_tolerance_version !=
        kPeriodicMeanFieldValidationToleranceVersion) {
        throw std::invalid_argument(
            "periodic restricted mean-field validation-tolerance version is "
            "not supported by state contract v1");
    }
    const PeriodicMeanFieldValidationTolerances tolerances;
    if (!input.converged) {
        throw std::invalid_argument(
            "periodic restricted mean-field state requires a converged SCF "
            "snapshot");
    }
    if (input.symmetry_reduced_input || input.symmetry_reconstructed_input) {
        throw std::invalid_argument(
            "periodic restricted mean-field state contract v1 rejects "
            "symmetry-reduced, IBZ-only, or symmetry-reconstructed input; "
            "supply a directly computed complete regular Brillouin-zone mesh "
            "in RegularKMesh order");
    }
    if (!finite_matrix3(input.reciprocal_lattice)) {
        throw std::invalid_argument(
            "periodic restricted mean-field reciprocal lattice contains a "
            "non-finite value");
    }
    const double reciprocal_scale =
        input.reciprocal_lattice.cwiseAbs().maxCoeff();
    double relative_reciprocal_volume = 0.0;
    if (reciprocal_scale > 0.0) {
        relative_reciprocal_volume = std::abs(
            (input.reciprocal_lattice / reciprocal_scale).determinant());
    }
    if (!std::isfinite(relative_reciprocal_volume) ||
        !(relative_reciprocal_volume >
          tolerances.reciprocal_lattice_relative_volume_floor)) {
        throw std::invalid_argument(
            "periodic restricted mean-field reciprocal lattice must have "
            "three linearly independent vectors");
    }
    if (input.electrons_per_cell == 0U ||
        input.electrons_per_cell % 2U != 0U) {
        throw std::invalid_argument(
            "periodic restricted mean-field electrons_per_cell must be a "
            "positive even integer");
    }
    if (!std::isfinite(input.reference_energy_per_cell)) {
        throw std::invalid_argument(
            "periodic restricted mean-field reference_energy_per_cell must "
            "be finite");
    }
    if (!std::isfinite(input.minimum_band_gap_hartree) ||
        input.minimum_band_gap_hartree <= 0.0) {
        throw std::invalid_argument(
            "periodic restricted mean-field minimum_band_gap_hartree must be "
            "finite and strictly positive");
    }
    if (input.n_basis == 0 || input.n_effective_orbitals == 0 ||
        input.n_effective_orbitals > input.n_basis) {
        throw std::invalid_argument(
            "periodic restricted mean-field requires 0 < "
            "n_effective_orbitals <= n_basis");
    }
    if (input.n_basis >
            static_cast<std::uint64_t>(
                std::numeric_limits<Eigen::Index>::max()) ||
        input.n_effective_orbitals >
            static_cast<std::uint64_t>(
                std::numeric_limits<Eigen::Index>::max()) ||
        input.n_effective_orbitals >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "periodic restricted mean-field dimensions exceed native index "
            "limits");
    }

    const RegularKMesh addressing(input.mesh, input.is_shift);
    if (input.kpoints_.size() != addressing.size()) {
        throw std::invalid_argument(
            "periodic restricted mean-field state contract v1 requires all "
            "points of the regular Brillouin-zone mesh; received " +
            std::to_string(input.kpoints_.size()) + " for a mesh containing " +
            std::to_string(addressing.size()) +
            " points (IBZ-only input is not accepted)");
    }

    // Finish all scalar, count, shape, mask-value, and finiteness checks
    // before eigensolvers or matrix products allocate validation workspaces.
    const Eigen::Index n_basis =
        static_cast<Eigen::Index>(input.n_basis);
    const Eigen::Index n_effective =
        static_cast<Eigen::Index>(input.n_effective_orbitals);
    const std::size_t n_effective_size =
        static_cast<std::size_t>(input.n_effective_orbitals);
    for (std::size_t kpoint = 0; kpoint < input.kpoints_.size(); ++kpoint) {
        const auto& point = input.kpoints_[kpoint];
        if (!finite_vector3(point.k_cartesian)) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "Cartesian coordinate finiteness"));
        }
        if (!std::isfinite(point.weight) || point.weight <= 0.0) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "integration weight"));
        }
        require_shape(
            point.overlap, n_basis, n_basis, kpoint, "overlap matrix");
        require_shape(point.fock, n_basis, n_basis, kpoint, "Fock matrix");
        require_shape(
            point.coefficients,
            n_basis,
            n_effective,
            kpoint,
            "coefficient matrix");
        require_vector_shape(
            point.orbital_energies,
            n_effective,
            kpoint,
            "orbital-energy vector");
        require_vector_shape(
            point.occupations,
            n_effective,
            kpoint,
            "occupation vector");
        require_mask_shape_and_values(
            point.frozen_core_mask,
            n_effective_size,
            kpoint,
            "frozen-core mask");
        require_mask_shape_and_values(
            point.correlated_occupied_mask,
            n_effective_size,
            kpoint,
            "correlated-occupied mask");
        require_mask_shape_and_values(
            point.virtual_mask,
            n_effective_size,
            kpoint,
            "virtual mask");
        if (!finite_complex_matrix(point.overlap) ||
            !finite_complex_matrix(point.fock) ||
            !finite_complex_matrix(point.coefficients) ||
            !finite_real_vector(point.orbital_energies) ||
            !finite_real_vector(point.occupations)) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "matrix/vector finiteness"));
        }
    }

    const std::uint64_t resident_bytes =
        estimate_periodic_restricted_mean_field_resident_bytes(
            input.mesh, input.n_basis, input.n_effective_orbitals);
    PeriodicMeanFieldValidationDiagnostics diagnostics;
    diagnostics.minimum_overlap_eigenvalue =
        std::numeric_limits<double>::infinity();

    const double uniform_weight =
        1.0 / static_cast<double>(addressing.size());
    double weighted_electron_count = 0.0;
    double global_homo = -std::numeric_limits<double>::infinity();
    double global_lumo = std::numeric_limits<double>::infinity();
    std::uint64_t common_frozen = 0;
    std::uint64_t common_correlated = 0;
    std::uint64_t common_virtual = 0;
    bool have_common_counts = false;

    // Complete every O(Nk * (nAO^2 + nMO)) large-workspace-free gate over the
    // full mesh before the first eigensolver or matrix product.  A malformed
    // late k point must not make a target mesh perform earlier O(nAO^3) work.
    for (std::size_t kpoint = 0; kpoint < input.kpoints_.size(); ++kpoint) {
        const auto& point = input.kpoints_[kpoint];
        const Eigen::Vector3d expected_k =
            input.reciprocal_lattice * addressing.fractional_at(kpoint);
        const double coordinate_residual =
            (point.k_cartesian - expected_k).cwiseAbs().maxCoeff();
        diagnostics.maximum_kpoint_cartesian_residual = std::max(
            diagnostics.maximum_kpoint_cartesian_residual,
            coordinate_residual);
        const double coordinate_scale = std::max(
            point.k_cartesian.cwiseAbs().maxCoeff(),
            expected_k.cwiseAbs().maxCoeff());
        if (!within_scaled_limit(
                coordinate_residual,
                tolerances.coordinate_absolute,
                tolerances.coordinate_relative,
                coordinate_scale)) {
            throw std::invalid_argument(
                kpoint_context(
                    kpoint,
                    "Cartesian coordinate or RegularKMesh ordering"));
        }

        const double weight_residual = std::abs(point.weight - uniform_weight);
        diagnostics.maximum_weight_residual = std::max(
            diagnostics.maximum_weight_residual, weight_residual);
        if (!within_scaled_limit(
                weight_residual,
                tolerances.scalar_absolute,
                tolerances.scalar_relative,
                uniform_weight)) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "uniform full-BZ weight"));
        }

        const double overlap_hermiticity =
            hermiticity_residual(point.overlap);
        diagnostics.maximum_overlap_hermiticity_residual = std::max(
            diagnostics.maximum_overlap_hermiticity_residual,
            overlap_hermiticity);
        if (!within_scaled_limit(
                overlap_hermiticity,
                tolerances.matrix_absolute,
                tolerances.matrix_relative,
                matrix_max_abs(point.overlap))) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "overlap Hermiticity"));
        }

        const double fock_hermiticity = hermiticity_residual(point.fock);
        diagnostics.maximum_fock_hermiticity_residual = std::max(
            diagnostics.maximum_fock_hermiticity_residual,
            fock_hermiticity);
        if (!within_scaled_limit(
                fock_hermiticity,
                tolerances.matrix_absolute,
                tolerances.matrix_relative,
                matrix_max_abs(point.fock))) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "Fock Hermiticity"));
        }

        const double energy_scale = vector_max_abs(point.orbital_energies);
        for (Eigen::Index orbital = 1; orbital < n_effective; ++orbital) {
            const double violation = std::max(
                0.0,
                point.orbital_energies[orbital - 1] -
                    point.orbital_energies[orbital]);
            diagnostics.maximum_energy_order_violation = std::max(
                diagnostics.maximum_energy_order_violation, violation);
            if (!within_scaled_limit(
                    violation,
                    tolerances.scalar_absolute,
                    tolerances.scalar_relative,
                    energy_scale)) {
                throw std::invalid_argument(
                    kpoint_context(kpoint, "ascending orbital energies"));
            }
        }

        std::uint64_t n_frozen = 0;
        std::uint64_t n_correlated = 0;
        std::uint64_t n_virtual = 0;
        double kpoint_electrons = 0.0;
        for (std::size_t orbital = 0; orbital < n_effective_size; ++orbital) {
            const unsigned membership =
                static_cast<unsigned>(point.frozen_core_mask[orbital]) +
                static_cast<unsigned>(
                    point.correlated_occupied_mask[orbital]) +
                static_cast<unsigned>(point.virtual_mask[orbital]);
            if (membership != 1U) {
                throw std::invalid_argument(
                    kpoint_context(
                        kpoint,
                        "disjoint and complete orbital partition masks"));
            }
            const bool occupied =
                point.frozen_core_mask[orbital] != 0U ||
                point.correlated_occupied_mask[orbital] != 0U;
            const double expected_occupation = occupied ? 2.0 : 0.0;
            const double occupation_residual = std::abs(
                point.occupations[static_cast<Eigen::Index>(orbital)] -
                expected_occupation);
            diagnostics.maximum_occupation_residual = std::max(
                diagnostics.maximum_occupation_residual,
                occupation_residual);
            if (!within_scaled_limit(
                    occupation_residual,
                    tolerances.scalar_absolute,
                    tolerances.scalar_relative,
                    expected_occupation)) {
                throw std::invalid_argument(
                    kpoint_context(
                        kpoint,
                        "closed-shell 2/0 occupations and masks"));
            }

            const double energy =
                point.orbital_energies[static_cast<Eigen::Index>(orbital)];
            if (point.frozen_core_mask[orbital] != 0U) {
                ++n_frozen;
                global_homo = std::max(global_homo, energy);
            } else if (point.correlated_occupied_mask[orbital] != 0U) {
                ++n_correlated;
                global_homo = std::max(global_homo, energy);
            } else {
                ++n_virtual;
                global_lumo = std::min(global_lumo, energy);
            }
            kpoint_electrons +=
                point.occupations[static_cast<Eigen::Index>(orbital)];
        }
        if (n_correlated == 0 || n_virtual == 0) {
            throw std::invalid_argument(
                kpoint_context(
                    kpoint,
                    "orbital partition (at least one correlated occupied and "
                    "one virtual are required)"));
        }
        if (!have_common_counts) {
            common_frozen = n_frozen;
            common_correlated = n_correlated;
            common_virtual = n_virtual;
            have_common_counts = true;
        } else if (n_frozen != common_frozen ||
                   n_correlated != common_correlated ||
                   n_virtual != common_virtual) {
            throw std::invalid_argument(
                kpoint_context(
                    kpoint,
                    "orbital-mask ranks (must be common across the mesh)"));
        }
        weighted_electron_count += uniform_weight * kpoint_electrons;
    }

    diagnostics.electron_count_residual = std::abs(
        weighted_electron_count -
        static_cast<double>(input.electrons_per_cell));
    if (!within_scaled_limit(
            diagnostics.electron_count_residual,
            tolerances.scalar_absolute,
            tolerances.scalar_relative,
            std::max(
                std::abs(weighted_electron_count),
                static_cast<double>(input.electrons_per_cell)))) {
        throw std::invalid_argument(
            "periodic restricted mean-field weighted electron count does not "
            "match electrons_per_cell");
    }

    diagnostics.band_gap_hartree = global_lumo - global_homo;
    if (!std::isfinite(diagnostics.band_gap_hartree) ||
        !(diagnostics.band_gap_hartree >
          input.minimum_band_gap_hartree)) {
        throw std::invalid_argument(
            "periodic restricted mean-field global band gap " +
            std::to_string(diagnostics.band_gap_hartree) +
            " Ha is not strictly above the requested minimum " +
            std::to_string(input.minimum_band_gap_hartree) + " Ha");
    }

    for (std::size_t kpoint = 0; kpoint < input.kpoints_.size(); ++kpoint) {
        const auto& point = input.kpoints_[kpoint];
        const PeriodicMeanFieldComplexMatrix symmetrized_overlap =
            0.5 * point.overlap + 0.5 * point.overlap.adjoint().eval();
        Eigen::SelfAdjointEigenSolver<PeriodicMeanFieldComplexMatrix>
            overlap_solver(symmetrized_overlap, Eigen::EigenvaluesOnly);
        if (overlap_solver.info() != Eigen::Success) {
            throw std::runtime_error(
                kpoint_context(kpoint, "overlap eigendecomposition"));
        }
        const double minimum_overlap =
            overlap_solver.eigenvalues().minCoeff();
        const double maximum_overlap =
            overlap_solver.eigenvalues().cwiseAbs().maxCoeff();
        diagnostics.minimum_overlap_eigenvalue = std::min(
            diagnostics.minimum_overlap_eigenvalue, minimum_overlap);
        const double overlap_floor =
            tolerances.overlap_eigenvalue_relative_floor *
            std::max(1.0, maximum_overlap);
        if (!std::isfinite(minimum_overlap) ||
            !std::isfinite(maximum_overlap) ||
            !std::isfinite(overlap_floor) ||
            !(minimum_overlap > overlap_floor)) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "positive-definite overlap metric"));
        }

        const PeriodicMeanFieldComplexMatrix gram =
            point.coefficients.adjoint() * point.overlap * point.coefficients;
        const double orthonormality_residual = identity_residual(gram);
        diagnostics.maximum_metric_orthonormality_residual = std::max(
            diagnostics.maximum_metric_orthonormality_residual,
            orthonormality_residual);
        if (!within_scaled_limit(
                orthonormality_residual,
                tolerances.matrix_absolute,
                tolerances.matrix_relative,
                matrix_max_abs(gram))) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "C^H S C orthonormality"));
        }

        const PeriodicMeanFieldComplexMatrix fock_coefficients =
            point.fock * point.coefficients;
        PeriodicMeanFieldComplexMatrix metric_coefficients =
            point.overlap * point.coefficients;
        for (Eigen::Index orbital = 0; orbital < n_effective; ++orbital) {
            metric_coefficients.col(orbital) *=
                point.orbital_energies[orbital];
        }
        const double roothaan_residual = matrix_max_abs(
            fock_coefficients - metric_coefficients);
        diagnostics.maximum_roothaan_residual = std::max(
            diagnostics.maximum_roothaan_residual, roothaan_residual);
        const double roothaan_scale = std::max(
            matrix_max_abs(fock_coefficients),
            matrix_max_abs(metric_coefficients));
        if (!within_scaled_limit(
                roothaan_residual,
                tolerances.matrix_absolute,
                tolerances.matrix_relative,
                roothaan_scale)) {
            throw std::invalid_argument(
                kpoint_context(kpoint, "F C = S C epsilon residual"));
        }
    }

    // Hash the accepted logical values, never Eigen's raw storage. Matrix
    // entries use explicit column-major traversal, vectors use increasing
    // indices, and every variable shape/length is encoded. This is a streaming
    // pass over the retained payload with constant additional memory.
    CanonicalStateHasher payload_hasher(
        "vibeqc.periodic.restricted-mean-field-state.numerical-payload");
    payload_hasher.add_u32(
        static_cast<std::uint32_t>(input.periodic_dimension));
    for (const int value : input.mesh) {
        payload_hasher.add_u64(static_cast<std::uint64_t>(value));
    }
    for (const int value : input.is_shift) {
        payload_hasher.add_u64(static_cast<std::uint64_t>(value));
    }
    payload_hasher.add_u64(
        static_cast<std::uint64_t>(input.kpoints_.size()));
    payload_hasher.add_u64(input.n_basis);
    payload_hasher.add_u64(input.n_effective_orbitals);
    payload_hasher.add_matrix3(input.reciprocal_lattice);
    payload_hasher.add_double(input.reference_energy_per_cell);
    for (const auto& point : input.kpoints_) {
        payload_hasher.add_vector3(point.k_cartesian);
        payload_hasher.add_double(uniform_weight);
        payload_hasher.add_complex_matrix(point.overlap);
        payload_hasher.add_complex_matrix(point.fock);
        payload_hasher.add_complex_matrix(point.coefficients);
        payload_hasher.add_real_vector(point.orbital_energies);
        payload_hasher.add_real_vector(point.occupations);
        payload_hasher.add_mask(point.frozen_core_mask);
        payload_hasher.add_mask(point.correlated_occupied_mask);
        payload_hasher.add_mask(point.virtual_mask);
    }
    const std::string numerical_payload_sha256 = payload_hasher.finish();
    const std::string state_identity_sha256 = make_state_identity_sha256(
        input.calculation_identity,
        input.reference_kind,
        input.normalization,
        input.minimum_band_gap_hartree,
        tolerances,
        numerical_payload_sha256);
    if (!is_lower_sha256(numerical_payload_sha256) ||
        !is_lower_sha256(state_identity_sha256)) {
        throw std::logic_error(
            "periodic mean-field state digest construction failed");
    }

    std::vector<PeriodicRestrictedMeanFieldState::KPointBlock> blocks;
    blocks.reserve(input.kpoints_.size());
    for (auto& point : input.kpoints_) {
        PeriodicRestrictedMeanFieldState::KPointBlock block;
        block.k_cartesian = std::move(point.k_cartesian);
        // Accepted roundoff is not retained: the state advertises and stores
        // the exact equal-weight integration convention.
        block.weight = uniform_weight;
        block.overlap = std::move(point.overlap);
        block.fock = std::move(point.fock);
        block.coefficients = std::move(point.coefficients);
        block.orbital_energies = std::move(point.orbital_energies);
        block.occupations = std::move(point.occupations);
        block.frozen_core_mask = std::move(point.frozen_core_mask);
        block.correlated_occupied_mask =
            std::move(point.correlated_occupied_mask);
        block.virtual_mask = std::move(point.virtual_mask);
        blocks.push_back(std::move(block));
    }

    return std::shared_ptr<const PeriodicRestrictedMeanFieldState>(
        new const PeriodicRestrictedMeanFieldState(
            std::move(input.calculation_identity),
            numerical_payload_sha256,
            state_identity_sha256,
            input.reference_kind,
            input.normalization,
            input.periodic_dimension,
            addressing,
            std::move(input.reciprocal_lattice),
            input.n_basis,
            input.n_effective_orbitals,
            common_frozen,
            common_correlated,
            common_virtual,
            input.electrons_per_cell,
            input.reference_energy_per_cell,
            input.minimum_band_gap_hartree,
            uniform_weight,
            resident_bytes,
            tolerances,
            diagnostics,
            std::move(blocks)));
}

}  // namespace vibeqc
