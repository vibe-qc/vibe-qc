#include "vibeqc/periodic_auxiliary_fourier.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/cart_to_sph_data.hpp"
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {

static_assert(
    sizeof(std::size_t) <= sizeof(std::uint64_t),
    "auxiliary Fourier extents require size_t to fit in uint64_t");
static_assert(
    std::numeric_limits<double>::is_iec559 && sizeof(double) == 8U,
    "auxiliary Fourier panels require IEEE-754 binary64");
static_assert(
    sizeof(std::complex<double>) == 16U,
    "auxiliary Fourier panel byte accounting requires 16-byte complex128");

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr char kBasisDigestDomain[] =
    "vibeqc.periodic.auxiliary-basis-content";

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* context) {
    if (left != 0U
        && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(context);
    }
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(context);
    }
    return left + right;
}

class CanonicalBasisHasher {
public:
    CanonicalBasisHasher() {
        add_bytes_with_length(
            kBasisDigestDomain, sizeof(kBasisDigestDomain) - 1U);
        add_u32(kAuxiliaryBasisContentDigestVersion);
    }

    void add_u8(std::uint8_t value) {
        hasher_.update(&value, 1U);
    }

    void add_u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (24U - 8U * static_cast<unsigned>(index)));
        }
        hasher_.update(encoded.data(), encoded.size());
    }

    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (56U - 8U * static_cast<unsigned>(index)));
        }
        hasher_.update(encoded.data(), encoded.size());
    }

    void add_double(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "auxiliary basis identity requires finite binary64 values");
        }
        // IEEE equality deliberately maps both signs of zero to +0.0.
        if (value == 0.0) value = 0.0;
        static_assert(std::numeric_limits<double>::is_iec559,
                      "auxiliary basis identity requires IEEE-754 binary64");
        static_assert(sizeof(double) == sizeof(std::uint64_t),
                      "auxiliary basis identity requires IEEE-754 binary64");
        std::uint64_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(bits));
        add_u64(bits);
    }

    void add_bytes_with_length(const char* value, std::size_t size) {
        add_u64(static_cast<std::uint64_t>(size));
        hasher_.update(
            reinterpret_cast<const std::uint8_t*>(value), size);
    }

    std::string finish_hex() { return hasher_.finish_hex(); }

private:
    detail::Sha256 hasher_;
};

std::uint64_t shell_function_count(int angular_momentum, bool pure) {
    if (angular_momentum < 0) {
        throw std::invalid_argument(
            "auxiliary basis contains a shell with negative angular "
            "momentum");
    }
    const auto angular = static_cast<std::uint64_t>(angular_momentum);
    if (pure) {
        return checked_add(
            checked_multiply(
                2U,
                angular,
                "pure auxiliary shell function count overflows uint64"),
            1U,
            "pure auxiliary shell function count overflows uint64");
    }
    const std::uint64_t first = checked_add(
        angular,
        1U,
        "Cartesian auxiliary shell function count overflows uint64");
    const std::uint64_t second = checked_add(
        angular,
        2U,
        "Cartesian auxiliary shell function count overflows uint64");
    return checked_multiply(
               first,
               second,
               "Cartesian auxiliary shell function count overflows uint64")
        / 2U;
}

struct BasisContentCounts {
    std::uint64_t function_count = 0U;
    std::uint64_t contraction_record_count = 0U;
};

BasisContentCounts validate_basis_content(const BasisSet& basis) {
    const auto& shells = basis.libint();
    if (shells.empty()) {
        throw std::invalid_argument(
            "auxiliary basis must contain at least one libint shell");
    }
    BasisContentCounts counts;
    for (std::size_t shell_index = 0;
         shell_index < shells.size();
         ++shell_index) {
        const auto& shell = shells[shell_index];
        if (basis.shell_atom_index(shell_index) < 0) {
            throw std::invalid_argument(
                "auxiliary basis contains a negative atom index");
        }
        if (shell.alpha.empty()) {
            throw std::invalid_argument(
                "auxiliary basis shell has no Gaussian primitives");
        }
        for (double exponent : shell.alpha) {
            if (!std::isfinite(exponent) || exponent <= 0.0) {
                throw std::invalid_argument(
                    "auxiliary basis exponents must be finite and positive");
            }
        }
        for (double coordinate : shell.O) {
            if (!std::isfinite(coordinate)) {
                throw std::invalid_argument(
                    "auxiliary basis shell origins must be finite");
            }
        }
        if (shell.contr.empty()) {
            throw std::invalid_argument(
                "auxiliary basis shell has no contractions");
        }
        for (const auto& contraction : shell.contr) {
            if (contraction.coeff.size() != shell.alpha.size()) {
                throw std::invalid_argument(
                    "auxiliary basis shell exponent/coefficient counts "
                    "differ");
            }
            for (double coefficient : contraction.coeff) {
                if (!std::isfinite(coefficient)) {
                    throw std::invalid_argument(
                        "auxiliary basis coefficients must be finite");
                }
            }
            counts.function_count = checked_add(
                counts.function_count,
                shell_function_count(contraction.l, contraction.pure),
                "auxiliary basis function count overflows uint64");
            counts.contraction_record_count = checked_add(
                counts.contraction_record_count,
                1U,
                "auxiliary basis contraction-record count overflows uint64");
        }
    }
    if (counts.function_count
        != static_cast<std::uint64_t>(basis.nbasis())) {
        throw std::logic_error(
            "auxiliary basis contractions do not cover BasisSet::nbasis");
    }
    return counts;
}

void require_fourier_shell_convention(int angular_momentum, bool pure) {
    if (angular_momentum > cart_to_sph_data::kMaxL) {
        throw std::invalid_argument(
            "auxiliary Gaussian Fourier panel shell with L="
            + std::to_string(angular_momentum)
            + " exceeds Cartesian-to-spherical table coverage 0.."
            + std::to_string(cart_to_sph_data::kMaxL));
    }
    if (!pure && angular_momentum > 0) {
        throw std::invalid_argument(
            "auxiliary Gaussian Fourier panel does not support Cartesian "
            "shells with L>0");
    }
}

double integer_power(double base, int exponent) noexcept {
    double result = 1.0;
    for (int factor = 0; factor < exponent; ++factor) result *= base;
    return result;
}

double fixed_binary64_squared_norm(double x, double y, double z) noexcept {
    double value = std::fma(z, z, 0.0);
    value = std::fma(y, y, value);
    return std::fma(x, x, value);
}

std::complex<double> negative_i_power(int exponent) noexcept {
    switch (exponent % 4) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, -1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, 1.0};
    }
}

void require_addressable_lane(std::size_t count,
                              std::ptrdiff_t stride) {
    if (stride <= 0) {
        throw std::invalid_argument(
            "auxiliary Gaussian Fourier vector strides must be positive");
    }
    if (count > 1U) {
        const auto last = static_cast<std::uint64_t>(count - 1U);
        const auto positive_stride = static_cast<std::uint64_t>(stride);
        if (last
            > static_cast<std::uint64_t>(
                  std::numeric_limits<std::ptrdiff_t>::max())
                / positive_stride) {
            throw std::overflow_error(
                "auxiliary Gaussian Fourier vector stride overflows "
                "ptrdiff_t");
        }
    }
}

}  // namespace

std::string auxiliary_basis_content_identity_sha256(const BasisSet& basis) {
    const BasisContentCounts counts = validate_basis_content(basis);
    const auto& shells = basis.libint();

    CanonicalBasisHasher digest;
    digest.add_u64(static_cast<std::uint64_t>(basis.nbasis()));
    digest.add_u64(static_cast<std::uint64_t>(basis.nshells()));
    digest.add_u64(counts.contraction_record_count);
    for (std::size_t shell_index = 0;
         shell_index < shells.size();
         ++shell_index) {
        const auto& shell = shells[shell_index];
        digest.add_u64(static_cast<std::uint64_t>(shell.contr.size()));
        for (const auto& contraction : shell.contr) {
            digest.add_u64(static_cast<std::uint64_t>(
                basis.shell_atom_index(shell_index)));
            digest.add_u32(static_cast<std::uint32_t>(contraction.l));
            digest.add_u8(contraction.pure ? 1U : 0U);
            digest.add_u64(static_cast<std::uint64_t>(shell.alpha.size()));
            for (double exponent : shell.alpha) digest.add_double(exponent);
            for (double coefficient : contraction.coeff) {
                digest.add_double(coefficient);
            }
            for (double coordinate : shell.O) digest.add_double(coordinate);
        }
    }
    return digest.finish_hex();
}

AuxiliaryFourierPanel auxiliary_gaussian_fourier_panel(
    const BasisSet& basis,
    AuxiliaryFourierVectorView vectors,
    std::uint64_t output_byte_cap) {
    if (output_byte_cap == 0U) {
        throw std::invalid_argument(
            "auxiliary Gaussian Fourier output-byte cap must be positive");
    }

    // Do the complete output-extent admission before inspecting shell content,
    // validating vectors, or evaluating a single Gaussian.  In particular, a
    // cap-minus-one request is a cheap deterministic failure.
    const std::uint64_t n_auxiliary =
        static_cast<std::uint64_t>(basis.nbasis());
    const std::uint64_t n_vectors = static_cast<std::uint64_t>(vectors.count);
    const std::uint64_t element_count = checked_multiply(
        n_auxiliary,
        n_vectors,
        "auxiliary Gaussian Fourier element count overflows uint64");
    const std::uint64_t output_bytes = checked_multiply(
        element_count,
        static_cast<std::uint64_t>(sizeof(std::complex<double>)),
        "auxiliary Gaussian Fourier output byte count overflows uint64");
    if (output_bytes > output_byte_cap) {
        throw std::length_error(
            "auxiliary Gaussian Fourier output exceeds output-byte cap");
    }
    if (element_count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "auxiliary Gaussian Fourier element count exceeds size_t");
    }
    const auto size = static_cast<std::size_t>(element_count);
    if (size > std::vector<std::complex<double>>().max_size()) {
        throw std::length_error(
            "auxiliary Gaussian Fourier output exceeds vector max_size");
    }

    if (vectors.count > 0U
        && (vectors.x == nullptr || vectors.y == nullptr
            || vectors.z == nullptr)) {
        throw std::invalid_argument(
            "auxiliary Gaussian Fourier vector lanes must not be null");
    }
    require_addressable_lane(vectors.count, vectors.x_stride);
    require_addressable_lane(vectors.count, vectors.y_stride);
    require_addressable_lane(vectors.count, vectors.z_stride);
    const BasisContentCounts counts = validate_basis_content(basis);
    if (counts.function_count != n_auxiliary) {
        throw std::logic_error(
            "auxiliary Gaussian Fourier basis extent changed during "
            "validation");
    }
    const auto& shells = basis.libint();
    for (const auto& shell : shells) {
        for (const auto& contraction : shell.contr) {
            require_fourier_shell_convention(
                contraction.l, contraction.pure);
        }
    }
    for (std::size_t vector = 0; vector < vectors.count; ++vector) {
        const double px = vectors.x[
            static_cast<std::ptrdiff_t>(vector) * vectors.x_stride];
        const double py = vectors.y[
            static_cast<std::ptrdiff_t>(vector) * vectors.y_stride];
        const double pz = vectors.z[
            static_cast<std::ptrdiff_t>(vector) * vectors.z_stride];
        if (!std::isfinite(px) || !std::isfinite(py)
            || !std::isfinite(pz)) {
            throw std::invalid_argument(
                "auxiliary Gaussian Fourier vectors must be finite");
        }
        const double p_squared =
            fixed_binary64_squared_norm(px, py, pz);
        if (!std::isfinite(p_squared)) {
            throw std::invalid_argument(
                "auxiliary Gaussian Fourier squared vector norm must be "
                "finite");
        }
        for (const auto& shell : shells) {
            const double center_argument =
                px * shell.O[0] + py * shell.O[1] + pz * shell.O[2];
            if (!std::isfinite(center_argument)) {
                throw std::invalid_argument(
                    "auxiliary Gaussian Fourier center phase argument must "
                    "be finite");
            }
        }
    }

    AuxiliaryFourierPanel panel;
    panel.n_auxiliary = static_cast<std::size_t>(n_auxiliary);
    panel.n_vectors = static_cast<std::size_t>(n_vectors);
    panel.output_bytes = output_bytes;
    panel.data.resize(size, std::complex<double>(0.0, 0.0));

    std::size_t function_offset = 0U;
    for (const auto& shell : shells) {
        for (const auto& contraction : shell.contr) {
            const int angular_momentum = contraction.l;
            const std::size_t n_components =
                cart_to_sph_data::n_sph_for_l(angular_momentum);
            const std::size_t n_cartesian =
                cart_to_sph_data::n_cart_for_l(angular_momentum);
            const auto* cartesian =
                cart_to_sph_data::cart_table_for_l(angular_momentum);
            const double* spherical =
                cart_to_sph_data::sph_table_for_l(angular_momentum);
            if (cartesian == nullptr || spherical == nullptr) {
                throw std::logic_error(
                    "auxiliary Gaussian Fourier table dispatch failed");
            }

            const double convention_scale = angular_momentum == 0
                ? 1.0
                : std::sqrt(
                    4.0 * kPi
                    / static_cast<double>(2 * angular_momentum + 1));
            const std::complex<double> angular_phase =
                negative_i_power(angular_momentum);

            for (std::size_t vector = 0;
                 vector < panel.n_vectors;
                 ++vector) {
                const auto offset = static_cast<std::ptrdiff_t>(vector);
                const double px = vectors.x[offset * vectors.x_stride];
                const double py = vectors.y[offset * vectors.y_stride];
                const double pz = vectors.z[offset * vectors.z_stride];
                const double p_squared =
                    fixed_binary64_squared_norm(px, py, pz);
                const double center_argument =
                    px * shell.O[0]
                    + py * shell.O[1]
                    + pz * shell.O[2];
                const std::complex<double> center_phase(
                    std::cos(center_argument), -std::sin(center_argument));

                double contracted_radial = 0.0;
                for (std::size_t primitive = 0;
                     primitive < shell.alpha.size();
                     ++primitive) {
                    const double exponent = shell.alpha[primitive];
                    const double inverse_two_exponent_power = integer_power(
                        1.0 / (2.0 * exponent), angular_momentum);
                    const double primitive_radial =
                        std::pow(kPi / exponent, 1.5)
                        * std::exp(-p_squared / (4.0 * exponent))
                        * inverse_two_exponent_power;
                    contracted_radial +=
                        contraction.coeff[primitive] * primitive_radial;
                }
                if (!std::isfinite(contracted_radial)) {
                    throw std::overflow_error(
                        "auxiliary Gaussian Fourier radial value is "
                        "non-finite");
                }

                for (std::size_t component = 0;
                     component < n_components;
                     ++component) {
                    double solid_harmonic = 0.0;
                    for (std::size_t term = 0;
                         term < n_cartesian;
                         ++term) {
                        const auto powers = cartesian[term];
                        solid_harmonic +=
                            spherical[component * n_cartesian + term]
                            * integer_power(px, powers.i)
                            * integer_power(py, powers.j)
                            * integer_power(pz, powers.k);
                    }
                    const std::complex<double> value =
                        convention_scale
                        * angular_phase
                        * solid_harmonic
                        * contracted_radial
                        * center_phase;
                    if (!std::isfinite(value.real())
                        || !std::isfinite(value.imag())) {
                        throw std::overflow_error(
                            "auxiliary Gaussian Fourier value is non-finite");
                    }
                    panel(function_offset + component, vector) = value;
                }
            }
            function_offset += n_components;
        }
    }
    if (function_offset != panel.n_auxiliary) {
        throw std::logic_error(
            "auxiliary Gaussian Fourier shell walk did not cover nbasis");
    }
    return panel;
}

AuxiliaryFourierPanel auxiliary_gaussian_fourier_panel(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& vectors,
    std::uint64_t output_byte_cap) {
    AuxiliaryFourierVectorView view;
    view.count = static_cast<std::size_t>(vectors.rows());
    view.x_stride = 3;
    view.y_stride = 3;
    view.z_stride = 3;
    if (view.count > 0U) {
        view.x = vectors.data();
        view.y = vectors.data() + 1;
        view.z = vectors.data() + 2;
    }
    return auxiliary_gaussian_fourier_panel(basis, view, output_byte_cap);
}

}  // namespace vibeqc
