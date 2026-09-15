// Native kernels for the experimental aiccm2026dev-b Python route.
//
// These helpers deliberately live in the pybind module rather than the
// published C++ core interface: they accelerate B-stream finite-torus tensor
// transformations without changing the stable molecular/periodic APIs.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "vibeqc/kmesh_address.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

using Complex = std::complex<double>;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXcd =
    Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr double kTwoPi = 6.283185307179586476925286766559;

template <typename T>
std::string shape_string(const py::buffer_info& info) {
    (void)sizeof(T);
    std::string out = "(";
    for (py::ssize_t i = 0; i < info.ndim; ++i) {
        if (i != 0) out += ", ";
        out += std::to_string(info.shape[static_cast<std::size_t>(i)]);
    }
    out += ")";
    return out;
}

void require_rank(const py::buffer_info& info,
                  py::ssize_t rank,
                  const char* name) {
    if (info.ndim != rank) {
        throw std::runtime_error(
            std::string(name) + " must have rank " + std::to_string(rank) +
            ", got shape " + shape_string<double>(info));
    }
}

py::ssize_t checked_positive_mesh_product(py::ssize_t n0,
                                          py::ssize_t n1,
                                          py::ssize_t n2,
                                          const char* context) {
    if (n0 <= 0 || n1 <= 0 || n2 <= 0) {
        throw std::runtime_error(
            std::string("aiccm2026dev-b ") + context +
            " requires a positive cyclic mesh");
    }
    const py::ssize_t maximum = std::numeric_limits<py::ssize_t>::max();
    if (n0 > maximum / n1 || n0 * n1 > maximum / n2) {
        throw std::runtime_error(
            std::string("aiccm2026dev-b ") + context +
            " cyclic mesh product exceeds the supported index range");
    }
    return n0 * n1 * n2;
}

py::ssize_t checked_positive_product(py::ssize_t left,
                                     py::ssize_t right,
                                     const char* context) {
    if (left <= 0 || right <= 0) {
        throw std::runtime_error(
            std::string("aiccm2026dev-b ") + context +
            " requires positive dimensions");
    }
    if (left > std::numeric_limits<py::ssize_t>::max() / right) {
        throw std::runtime_error(
            std::string("aiccm2026dev-b ") + context +
            " exceeds the supported index range");
    }
    return left * right;
}

std::vector<Complex> finite_group_phases(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& kfrac,
    const py::array_t<std::int64_t,
                      py::array::c_style | py::array::forcecast>& translations) {
    const auto k = kfrac.unchecked<2>();
    const auto t = translations.unchecked<2>();
    const py::ssize_t n_k = k.shape(0);
    const py::ssize_t n_cells = t.shape(0);
    if (k.shape(1) != 3 || t.shape(1) != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b kernels require kfrac and translations with "
            "three Cartesian/fractional components");
    }
    if (n_k != n_cells) {
        throw std::runtime_error(
            "aiccm2026dev-b finite-torus kernels require one character "
            "per cyclic translation");
    }
    std::vector<Complex> phase(static_cast<std::size_t>(n_k * n_cells));
    for (py::ssize_t ik = 0; ik < n_k; ++ik) {
        for (py::ssize_t ir = 0; ir < n_cells; ++ir) {
            const double dot =
                k(ik, 0) * static_cast<double>(t(ir, 0)) +
                k(ik, 1) * static_cast<double>(t(ir, 1)) +
                k(ik, 2) * static_cast<double>(t(ir, 2));
            phase[static_cast<std::size_t>(ik * n_cells + ir)] =
                std::polar(1.0, kTwoPi * dot);
        }
    }
    return phase;
}

std::vector<Complex> auxiliary_translation_phases(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& kfrac,
    const py::array_t<std::int64_t,
                      py::array::c_style | py::array::forcecast>& translations) {
    const auto k = kfrac.unchecked<2>();
    const auto t = translations.unchecked<2>();
    const py::ssize_t n_k = k.shape(0);
    const py::ssize_t n_cells = t.shape(0);
    std::vector<Complex> aux(static_cast<std::size_t>(n_cells * n_k * n_k));
    for (py::ssize_t it = 0; it < n_cells; ++it) {
        for (py::ssize_t ki = 0; ki < n_k; ++ki) {
            for (py::ssize_t ka = 0; ka < n_k; ++ka) {
                const double dot =
                    (k(ka, 0) - k(ki, 0)) * static_cast<double>(t(it, 0)) +
                    (k(ka, 1) - k(ki, 1)) * static_cast<double>(t(it, 1)) +
                    (k(ka, 2) - k(ki, 2)) * static_cast<double>(t(it, 2));
                aux[static_cast<std::size_t>((it * n_k + ki) * n_k + ka)] =
                    std::polar(1.0, kTwoPi * dot);
            }
        }
    }
    return aux;
}

py::tuple matrix_to_real_supercell(
    py::array_t<Complex, py::array::c_style | py::array::forcecast> matrices,
    py::array_t<double, py::array::c_style | py::array::forcecast> kfrac,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations) {
    const py::buffer_info minfo = matrices.request();
    const py::buffer_info kinfo = kfrac.request();
    const py::buffer_info tinfo = translations.request();
    require_rank(minfo, 3, "matrices");
    require_rank(kinfo, 2, "kfrac");
    require_rank(tinfo, 2, "translations");

    const py::ssize_t n_k = minfo.shape[0];
    const py::ssize_t nbf = minfo.shape[1];
    if (minfo.shape[2] != nbf) {
        throw std::runtime_error(
            "aiccm2026dev-b matrix transform requires square k blocks, got " +
            shape_string<Complex>(minfo));
    }
    if (kinfo.shape[0] != n_k || tinfo.shape[0] != n_k) {
        throw std::runtime_error(
            "aiccm2026dev-b matrix transform shape mismatch: matrices has " +
            std::to_string(n_k) + " k blocks, kfrac shape is " +
            shape_string<double>(kinfo) + ", translations shape is " +
            shape_string<std::int64_t>(tinfo));
    }

    const py::ssize_t n_cells = n_k;
    const py::ssize_t n_ao = n_cells * nbf;
    py::array_t<double> out({n_ao, n_ao});
    auto out_u = out.mutable_unchecked<2>();
    const auto mat = matrices.unchecked<3>();
    const auto phase = finite_group_phases(kfrac, translations);
    const double scale = 1.0 / static_cast<double>(n_cells);
    double max_imag = 0.0;

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(2) schedule(static) reduction(max : max_imag)
        for (py::ssize_t row = 0; row < n_ao; ++row) {
            for (py::ssize_t col0 = 0; col0 < n_ao; ++col0) {
                if (col0 < row) {
                    continue;
                }
                const py::ssize_t r_cell = row / nbf;
                const py::ssize_t s_cell = col0 / nbf;
                const py::ssize_t mu = row % nbf;
                const py::ssize_t nu = col0 % nbf;
                Complex value{0.0, 0.0};
                Complex value_t{0.0, 0.0};
                for (py::ssize_t ik = 0; ik < n_k; ++ik) {
                    const Complex p_rs =
                        phase[static_cast<std::size_t>(ik * n_cells + r_cell)] *
                        std::conj(phase[static_cast<std::size_t>(
                            ik * n_cells + s_cell)]);
                    const Complex p_sr = std::conj(p_rs);
                    value += p_rs * mat(ik, mu, nu);
                    value_t += p_sr * mat(ik, nu, mu);
                }
                value *= scale;
                value_t *= scale;
                max_imag = std::max(max_imag, std::abs(value.imag()));
                max_imag = std::max(max_imag, std::abs(value_t.imag()));
                const double real_sym = 0.5 * (value.real() + value_t.real());
                out_u(row, col0) = real_sym;
                out_u(col0, row) = real_sym;
            }
        }
    }
    return py::make_tuple(out, max_imag);
}

py::array_t<Complex> inverse_bloch_transform_blocks(
    py::array_t<Complex, py::array::c_style | py::array::forcecast> matrices,
    py::array_t<double, py::array::c_style | py::array::forcecast> kfrac,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights) {
    const py::buffer_info minfo = matrices.request();
    const py::buffer_info kinfo = kfrac.request();
    const py::buffer_info tinfo = translations.request();
    const py::buffer_info winfo = weights.request();
    require_rank(minfo, 3, "matrices");
    require_rank(kinfo, 2, "kfrac");
    require_rank(tinfo, 2, "translations");
    require_rank(winfo, 1, "weights");

    const py::ssize_t n_k = minfo.shape[0];
    const py::ssize_t nbf = minfo.shape[1];
    if (minfo.shape[2] != nbf || kinfo.shape[0] != n_k ||
        kinfo.shape[1] != 3 || winfo.shape[0] != n_k ||
        tinfo.shape[1] != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b inverse Bloch transform shape mismatch: "
            "matrices " + shape_string<Complex>(minfo) + ", kfrac " +
            shape_string<double>(kinfo) + ", translations " +
            shape_string<std::int64_t>(tinfo) + ", weights " +
            shape_string<double>(winfo));
    }
    const py::ssize_t n_translations = tinfo.shape[0];
    py::array_t<Complex> out({n_translations, nbf, nbf});
    const auto mat = matrices.unchecked<3>();
    const auto k = kfrac.unchecked<2>();
    const auto t = translations.unchecked<2>();
    const auto w = weights.unchecked<1>();
    auto out_u = out.mutable_unchecked<3>();

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(2) schedule(static)
        for (py::ssize_t r = 0; r < n_translations; ++r) {
            for (py::ssize_t row = 0; row < nbf; ++row) {
                for (py::ssize_t col = 0; col < nbf; ++col) {
                    Complex value{0.0, 0.0};
                    for (py::ssize_t ik = 0; ik < n_k; ++ik) {
                        const double dot =
                            k(ik, 0) * static_cast<double>(t(r, 0)) +
                            k(ik, 1) * static_cast<double>(t(r, 1)) +
                            k(ik, 2) * static_cast<double>(t(r, 2));
                        value += std::polar(w(ik), -kTwoPi * dot) *
                                 mat(ik, row, col);
                    }
                    out_u(r, row, col) = value;
                }
            }
        }
    }
    return out;
}

py::array_t<Complex> canonical_wannier_coefficients(
    py::array_t<Complex, py::array::c_style | py::array::forcecast>
        coefficients_k,
    py::array_t<double, py::array::c_style | py::array::forcecast> kfrac,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations,
    py::ssize_t n_occ) {
    const py::buffer_info cinfo = coefficients_k.request();
    const py::buffer_info kinfo = kfrac.request();
    const py::buffer_info tinfo = translations.request();
    require_rank(cinfo, 3, "coefficients_k");
    require_rank(kinfo, 2, "kfrac");
    require_rank(tinfo, 2, "translations");
    const py::ssize_t n_k = cinfo.shape[0];
    const py::ssize_t nbf = cinfo.shape[1];
    const py::ssize_t n_mo = cinfo.shape[2];
    if (kinfo.shape[0] != n_k || tinfo.shape[0] != n_k ||
        kinfo.shape[1] != 3 || tinfo.shape[1] != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b Wannier transform requires coefficients_k "
            "shape (n_k, nbf, n_mo) and matching kfrac/translations "
            "shape (n_k, 3)");
    }
    if (n_occ < 1 || n_occ > n_mo) {
        throw std::runtime_error(
            "aiccm2026dev-b Wannier transform got an invalid occupied rank");
    }
    const py::ssize_t n_cells = n_k;
    py::array_t<Complex> out({n_cells * nbf, n_cells * n_occ});
    const auto coeff = coefficients_k.unchecked<3>();
    auto out_u = out.mutable_unchecked<2>();
    const auto phase = finite_group_phases(kfrac, translations);
    const double scale = 1.0 / static_cast<double>(n_cells);

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(4) schedule(static)
        for (py::ssize_t t = 0; t < n_cells; ++t) {
            for (py::ssize_t mu = 0; mu < nbf; ++mu) {
                for (py::ssize_t r = 0; r < n_cells; ++r) {
                    for (py::ssize_t n = 0; n < n_occ; ++n) {
                        Complex value{0.0, 0.0};
                        for (py::ssize_t ik = 0; ik < n_k; ++ik) {
                            value +=
                                phase[static_cast<std::size_t>(
                                    ik * n_cells + t)] *
                                coeff(ik, mu, n) *
                                std::conj(phase[static_cast<std::size_t>(
                                    ik * n_cells + r)]);
                        }
                        out_u(t * nbf + mu, r * n_occ + n) = value * scale;
                    }
                }
            }
        }
    }
    return out;
}

template <typename FockScalar>
Complex localization_one_particle_energy(
    const FockScalar* fock,
    const Complex* coefficients,
    py::ssize_t n_ao,
    py::ssize_t n_occ) {
    double energy_real = 0.0;
    double energy_imaginary = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) \
        reduction(+ : energy_real, energy_imaginary)
    for (py::ssize_t row = 0; row < n_ao; ++row) {
        for (py::ssize_t occ = 0; occ < n_occ; ++occ) {
            Complex transformed{0.0, 0.0};
            for (py::ssize_t col = 0; col < n_ao; ++col) {
                transformed +=
                    fock[static_cast<std::size_t>(row * n_ao + col)] *
                    coefficients[static_cast<std::size_t>(col * n_occ + occ)];
            }
            const Complex contribution =
                std::conj(coefficients[static_cast<std::size_t>(
                    row * n_occ + occ)]) *
                transformed;
            energy_real += contribution.real();
            energy_imaginary += contribution.imag();
        }
    }
    return {energy_real, energy_imaginary};
}

py::tuple localization_projector_audit(
    py::array_t<Complex, py::array::c_style | py::array::forcecast> canonical,
    py::array_t<Complex, py::array::c_style | py::array::forcecast> localized,
    py::object fock,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh,
    py::ssize_t nbf) {
    const py::buffer_info cinfo = canonical.request();
    const py::buffer_info linfo = localized.request();
    const py::buffer_info tinfo = translations.request();
    const py::buffer_info minfo = mesh.request();
    require_rank(cinfo, 2, "canonical");
    require_rank(linfo, 2, "localized");
    require_rank(tinfo, 2, "translations");
    require_rank(minfo, 1, "mesh");
    if (cinfo.shape != linfo.shape) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit requires canonical "
            "and localized coefficients with the same shape");
    }
    const py::ssize_t n_ao = cinfo.shape[0];
    const py::ssize_t n_occ = cinfo.shape[1];
    if (n_ao < 1 || n_occ < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit requires non-empty "
            "coefficient matrices");
    }
    if (tinfo.shape[1] != 3 || minfo.shape[0] != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit requires "
            "translations shape (n_cells, 3) and mesh shape (3,)");
    }
    if (nbf < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit requires a positive "
            "primitive AO rank");
    }

    const auto mesh_u = mesh.unchecked<1>();
    const py::ssize_t n0 = static_cast<py::ssize_t>(mesh_u(0));
    const py::ssize_t n1 = static_cast<py::ssize_t>(mesh_u(1));
    const py::ssize_t n2 = static_cast<py::ssize_t>(mesh_u(2));
    if (n0 < 1 || n1 < 1 || n2 < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit mesh entries must "
            "be positive");
    }
    const py::ssize_t n_cells = n0 * n1 * n2;
    if (tinfo.shape[0] != n_cells || n_ao != n_cells * nbf) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit coefficient, "
            "translation, mesh, and AO dimensions are inconsistent");
    }

    const auto trans = translations.unchecked<2>();
    std::vector<py::ssize_t> index_by_residue(
        static_cast<std::size_t>(n_cells), -1);
    for (py::ssize_t cell = 0; cell < n_cells; ++cell) {
        const py::ssize_t x = static_cast<py::ssize_t>(trans(cell, 0));
        const py::ssize_t y = static_cast<py::ssize_t>(trans(cell, 1));
        const py::ssize_t z = static_cast<py::ssize_t>(trans(cell, 2));
        if (x < 0 || x >= n0 || y < 0 || y >= n1 || z < 0 || z >= n2) {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit translation "
                "lies outside the declared mesh");
        }
        const py::ssize_t residue = (x * n1 + y) * n2 + z;
        if (index_by_residue[static_cast<std::size_t>(residue)] >= 0) {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit translations "
                "contain a duplicate residue");
        }
        index_by_residue[static_cast<std::size_t>(residue)] = cell;
    }

    std::array<std::vector<py::ssize_t>, 3> shifted_sources;
    const std::array<py::ssize_t, 3> mesh_shape{n0, n1, n2};
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        auto& sources = shifted_sources[static_cast<std::size_t>(axis)];
        sources.resize(static_cast<std::size_t>(n_cells));
        for (py::ssize_t cell = 0; cell < n_cells; ++cell) {
            std::array<py::ssize_t, 3> shifted{
                static_cast<py::ssize_t>(trans(cell, 0)),
                static_cast<py::ssize_t>(trans(cell, 1)),
                static_cast<py::ssize_t>(trans(cell, 2)),
            };
            shifted[static_cast<std::size_t>(axis)] =
                (shifted[static_cast<std::size_t>(axis)] - 1 +
                 mesh_shape[static_cast<std::size_t>(axis)]) %
                mesh_shape[static_cast<std::size_t>(axis)];
            const py::ssize_t residue =
                (shifted[0] * n1 + shifted[1]) * n2 + shifted[2];
            sources[static_cast<std::size_t>(cell)] =
                index_by_residue[static_cast<std::size_t>(residue)];
        }
    }

    py::object fock_owner = py::none();
    const double* fock_real_ptr = nullptr;
    const Complex* fock_complex_ptr = nullptr;
    if (!fock.is_none()) {
        py::array raw = py::array::ensure(fock);
        if (!raw) {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit requires an "
                "array-like Fock matrix");
        }
        if (raw.ndim() != 2 || raw.shape(0) != n_ao || raw.shape(1) != n_ao) {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit Fock matrix "
                "has the wrong AO shape");
        }
        if (raw.dtype().is(py::dtype::of<double>())) {
            using RealArray =
                py::array_t<double, py::array::c_style | py::array::forcecast>;
            RealArray converted = RealArray::ensure(raw);
            fock_owner = converted;
            fock_real_ptr = converted.data();
        } else if (raw.dtype().is(py::dtype::of<Complex>())) {
            using ComplexArray = py::array_t<
                Complex,
                py::array::c_style | py::array::forcecast>;
            ComplexArray converted = ComplexArray::ensure(raw);
            fock_owner = converted;
            fock_complex_ptr = converted.data();
        } else {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit Fock matrix "
                "must have dtype float64 or complex128");
        }
    }

    const Complex* canonical_ptr = canonical.data();
    const Complex* localized_ptr = localized.data();
    Eigen::Map<const RowMatrixXcd> c0(canonical_ptr, n_ao, n_occ);
    Eigen::Map<const RowMatrixXcd> c1(localized_ptr, n_ao, n_occ);
    if (!c0.allFinite() || !c1.allFinite()) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit coefficients must "
            "be finite");
    }
    if (fock_real_ptr != nullptr) {
        Eigen::Map<const RowMatrixXd> f(fock_real_ptr, n_ao, n_ao);
        if (!f.allFinite()) {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit Fock matrix "
                "must be finite");
        }
    } else if (fock_complex_ptr != nullptr) {
        Eigen::Map<const RowMatrixXcd> f(fock_complex_ptr, n_ao, n_ao);
        if (!f.allFinite()) {
            throw std::runtime_error(
                "aiccm2026dev-b localization projector audit Fock matrix "
                "must be finite");
        }
    }

    double density_squared = 0.0;
    double density_scale = 0.0;
    double energy_error = 0.0;
    std::array<double, 3> translation_squared{0.0, 0.0, 0.0};
    std::array<double, 3> translation_scale{0.0, 0.0, 0.0};
    {
        py::gil_scoped_release release;
        const auto stable_projector_difference_squared = [
            n_ao,
            n_occ](const Complex* left, const Complex* right) -> double {
            double total = 0.0;
            #pragma omp parallel for collapse(2) schedule(static) reduction(+ : total)
            for (py::ssize_t row = 0; row < n_ao; ++row) {
                for (py::ssize_t col = 0; col < n_ao; ++col) {
                    Complex left_value{0.0, 0.0};
                    Complex right_value{0.0, 0.0};
                    for (py::ssize_t occ = 0; occ < n_occ; ++occ) {
                        const std::size_t row_index = static_cast<std::size_t>(
                            row * n_occ + occ);
                        const std::size_t col_index = static_cast<std::size_t>(
                            col * n_occ + occ);
                        left_value += left[row_index] * std::conj(left[col_index]);
                        right_value +=
                            right[row_index] * std::conj(right[col_index]);
                    }
                    total += std::norm(left_value - right_value);
                }
            }
            return total;
        };
        const auto is_near_cancellation = [](double value, double scale) {
            return std::abs(value) <= 1.0e-10 * std::max(1.0, scale);
        };

        RowMatrixXcd gram0 = c0.adjoint() * c0;
        RowMatrixXcd gram1 = c1.adjoint() * c1;
        RowMatrixXcd cross = c1.adjoint() * c0;
        const double norm0 = gram0.squaredNorm();
        const double norm1 = gram1.squaredNorm();
        const double norm_cross = cross.squaredNorm();
        density_squared = norm0 + norm1 - 2.0 * norm_cross;
        density_scale = norm0 + norm1 + 2.0 * norm_cross;
        if (is_near_cancellation(density_squared, density_scale)) {
            density_squared = stable_projector_difference_squared(
                canonical_ptr,
                localized_ptr);
        }

        if (fock_real_ptr != nullptr) {
            const Complex energy0 = localization_one_particle_energy(
                fock_real_ptr,
                canonical_ptr,
                n_ao,
                n_occ);
            const Complex energy1 = localization_one_particle_energy(
                fock_real_ptr,
                localized_ptr,
                n_ao,
                n_occ);
            energy_error = std::abs((energy1 - energy0).real());
        } else if (fock_complex_ptr != nullptr) {
            const Complex energy0 = localization_one_particle_energy(
                fock_complex_ptr,
                canonical_ptr,
                n_ao,
                n_occ);
            const Complex energy1 = localization_one_particle_energy(
                fock_complex_ptr,
                localized_ptr,
                n_ao,
                n_occ);
            energy_error = std::abs((energy1 - energy0).real());
        }

        if (n_cells > 1) {
            RowMatrixXcd shifted(n_ao, n_occ);
            for (py::ssize_t axis = 0; axis < 3; ++axis) {
                if (mesh_shape[static_cast<std::size_t>(axis)] == 1) {
                    translation_squared[static_cast<std::size_t>(axis)] = 0.0;
                    translation_scale[static_cast<std::size_t>(axis)] =
                        4.0 * norm1;
                    continue;
                }
                const auto& sources =
                    shifted_sources[static_cast<std::size_t>(axis)];
                #pragma omp parallel for schedule(static)
                for (py::ssize_t cell = 0; cell < n_cells; ++cell) {
                    const py::ssize_t source =
                        sources[static_cast<std::size_t>(cell)];
                    const std::size_t count =
                        static_cast<std::size_t>(nbf * n_occ);
                    std::copy_n(
                        localized_ptr + static_cast<std::size_t>(
                            source * nbf * n_occ),
                        count,
                        shifted.data() + static_cast<std::size_t>(
                            cell * nbf * n_occ));
                }
                cross.noalias() = shifted.adjoint() * c1;
                const double translated_cross = cross.squaredNorm();
                translation_squared[static_cast<std::size_t>(axis)] =
                    2.0 * norm1 - 2.0 * translated_cross;
                translation_scale[static_cast<std::size_t>(axis)] =
                    2.0 * norm1 + 2.0 * translated_cross;
                if (is_near_cancellation(
                        translation_squared[static_cast<std::size_t>(axis)],
                        translation_scale[static_cast<std::size_t>(axis)])) {
                    translation_squared[static_cast<std::size_t>(axis)] =
                        stable_projector_difference_squared(
                            shifted.data(),
                            localized_ptr);
                }
            }
        }
    }

    const auto checked_norm = [](double value,
                                 double scale,
                                 const char* label) -> double {
        if (!std::isfinite(value) ||
            value < -1.0e-10 * std::max(1.0, scale)) {
            throw std::runtime_error(
                std::string("aiccm2026dev-b localization projector audit ") +
                label + " norm is numerically invalid");
        }
        return std::sqrt(std::max(0.0, value));
    };
    const double density_error =
        checked_norm(density_squared, density_scale, "density");
    double translation_error = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        translation_error = std::max(
            translation_error,
            checked_norm(
                translation_squared[axis],
                translation_scale[axis],
                "translation"));
    }
    if (!std::isfinite(energy_error)) {
        throw std::runtime_error(
            "aiccm2026dev-b localization projector audit energy is not finite");
    }
    return py::make_tuple(density_error, energy_error, translation_error);
}

py::tuple lpq_cache_to_real_supercell_impl(
    py::dict lpq_cache,
    py::array_t<double, py::array::c_style | py::array::forcecast> kfrac,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations,
    bool home_auxiliary_only) {
    const py::buffer_info kinfo = kfrac.request();
    const py::buffer_info tinfo = translations.request();
    require_rank(kinfo, 2, "kfrac");
    require_rank(tinfo, 2, "translations");
    if (kinfo.shape[1] != 3 || tinfo.shape[1] != 3 ||
        kinfo.shape[0] != tinfo.shape[0]) {
        throw std::runtime_error(
            "aiccm2026dev-b RI transform requires matching kfrac and "
            "translation arrays with shape (n_cells, 3)");
    }

    const py::ssize_t n_cells = kinfo.shape[0];
    const py::ssize_t n_k = n_cells;
    using ComplexArray =
        py::array_t<Complex, py::array::c_style | py::array::forcecast>;
    std::vector<ComplexArray> arrays(static_cast<std::size_t>(n_k * n_k));
    std::vector<const Complex*> blocks(static_cast<std::size_t>(n_k * n_k));

    py::ssize_t n_aux = -1;
    py::ssize_t nbf = -1;
    for (py::ssize_t ki = 0; ki < n_k; ++ki) {
        for (py::ssize_t ka = 0; ka < n_k; ++ka) {
            const py::object key = py::make_tuple(ki, ka);
            if (!lpq_cache.contains(key)) {
                throw std::runtime_error(
                    "aiccm2026dev-b RI transform missing lpq_cache[(" +
                    std::to_string(ki) + ", " + std::to_string(ka) + ")]");
            }
            ComplexArray arr = ComplexArray::ensure(lpq_cache[key]);
            if (!arr) {
                throw std::runtime_error(
                    "aiccm2026dev-b RI transform could not cast lpq_cache "
                    "entry to complex128 C-contiguous array");
            }
            const py::buffer_info info = arr.request();
            require_rank(info, 3, "lpq_cache entry");
            if (info.shape[1] != info.shape[2]) {
                throw std::runtime_error(
                    "aiccm2026dev-b RI transform requires square AO pair "
                    "blocks, got " + shape_string<Complex>(info));
            }
            if (n_aux < 0) {
                n_aux = info.shape[0];
                nbf = info.shape[1];
            } else if (info.shape[0] != n_aux || info.shape[1] != nbf) {
                throw std::runtime_error(
                    "aiccm2026dev-b RI transform got inconsistent LPQ "
                    "entry shape " + shape_string<Complex>(info));
            }
            const std::size_t index = static_cast<std::size_t>(ki * n_k + ka);
            arrays[index] = std::move(arr);
            blocks[index] = arrays[index].data();
        }
    }
    if (n_aux <= 0 || nbf <= 0) {
        throw std::runtime_error("aiccm2026dev-b RI transform got empty LPQ cache");
    }

    const py::ssize_t n_aux_super = n_cells * n_aux;
    const py::ssize_t n_ao = n_cells * nbf;
    const py::ssize_t n_aux_out = home_auxiliary_only ? n_aux : n_aux_super;
    py::array_t<double> out({n_aux_out, n_ao, n_ao});
    auto out_u = out.mutable_unchecked<3>();

    const auto phase = finite_group_phases(kfrac, translations);
    const auto aux_phase = home_auxiliary_only
        ? std::vector<Complex>{}
        : auxiliary_translation_phases(kfrac, translations);
    const double scale =
        1.0 / static_cast<double>(n_cells * n_cells);
    double max_imag = 0.0;
    double max_sym = 0.0;

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(2) schedule(static) \
            reduction(max : max_imag, max_sym)
        for (py::ssize_t ps = 0; ps < n_aux_out; ++ps) {
            for (py::ssize_t row = 0; row < n_ao; ++row) {
                const py::ssize_t t_cell = home_auxiliary_only ? 0 : ps / n_aux;
                const py::ssize_t p_aux = ps % n_aux;
                const py::ssize_t r_cell = row / nbf;
                const py::ssize_t mu = row % nbf;
                for (py::ssize_t col = row; col < n_ao; ++col) {
                    const py::ssize_t s_cell = col / nbf;
                    const py::ssize_t nu = col % nbf;
                    Complex value{0.0, 0.0};
                    Complex value_t{0.0, 0.0};
                    for (py::ssize_t ki = 0; ki < n_k; ++ki) {
                        const Complex phi_ki_r =
                            phase[static_cast<std::size_t>(
                                ki * n_cells + r_cell)];
                        const Complex phi_ki_s =
                            phase[static_cast<std::size_t>(
                                ki * n_cells + s_cell)];
                        for (py::ssize_t ka = 0; ka < n_k; ++ka) {
                            const Complex auxp = home_auxiliary_only
                                ? Complex{1.0, 0.0}
                                : aux_phase[static_cast<std::size_t>(
                                      (t_cell * n_k + ki) * n_k + ka)];
                            const Complex phase_rs =
                                auxp * phi_ki_r *
                                std::conj(phase[static_cast<std::size_t>(
                                    ka * n_cells + s_cell)]);
                            const Complex phase_sr =
                                auxp * phi_ki_s *
                                std::conj(phase[static_cast<std::size_t>(
                                    ka * n_cells + r_cell)]);
                            const Complex* block =
                                blocks[static_cast<std::size_t>(ki * n_k + ka)];
                            const std::size_t base =
                                static_cast<std::size_t>(p_aux * nbf * nbf);
                            value += phase_rs *
                                block[base + static_cast<std::size_t>(
                                    mu * nbf + nu)];
                            value_t += phase_sr *
                                block[base + static_cast<std::size_t>(
                                    nu * nbf + mu)];
                        }
                    }
                    value *= scale;
                    value_t *= scale;
                    max_imag = std::max(max_imag, std::abs(value.imag()));
                    max_imag = std::max(max_imag, std::abs(value_t.imag()));
                    max_sym = std::max(max_sym, std::abs(value - value_t));
                    const double real_sym =
                        0.5 * (value.real() + value_t.real());
                    out_u(ps, row, col) = real_sym;
                    out_u(ps, col, row) = real_sym;
                }
            }
        }
    }

    return py::make_tuple(out, max_imag, max_sym);
}

py::tuple lpq_cache_to_real_supercell(
    py::dict lpq_cache,
    py::array_t<double, py::array::c_style | py::array::forcecast> kfrac,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations) {
    return lpq_cache_to_real_supercell_impl(
        std::move(lpq_cache),
        std::move(kfrac),
        std::move(translations),
        false);
}

void validate_home_auxiliary_group(
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        kfrac,
    const py::array_t<std::int64_t,
                      py::array::c_style | py::array::forcecast>& translations,
    const py::array_t<std::int64_t,
                      py::array::c_style | py::array::forcecast>& mesh) {
    const py::buffer_info kinfo = kfrac.request();
    const py::buffer_info tinfo = translations.request();
    const py::buffer_info minfo = mesh.request();
    require_rank(kinfo, 2, "kfrac");
    require_rank(tinfo, 2, "translations");
    require_rank(minfo, 1, "mesh");
    if (kinfo.shape[1] != 3 || tinfo.shape[1] != 3 || minfo.shape[0] != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b home-auxiliary transform requires kfrac and "
            "translations with shape (n_cells, 3) and mesh shape (3,)");
    }
    const auto k = kfrac.unchecked<2>();
    const auto t = translations.unchecked<2>();
    const auto m = mesh.unchecked<1>();
    const py::ssize_t n0 = static_cast<py::ssize_t>(m(0));
    const py::ssize_t n1 = static_cast<py::ssize_t>(m(1));
    const py::ssize_t n2 = static_cast<py::ssize_t>(m(2));
    const py::ssize_t n_cells = checked_positive_mesh_product(
        n0, n1, n2, "home-auxiliary transform");
    if (kinfo.shape[0] != n_cells || tinfo.shape[0] != n_cells) {
        throw std::runtime_error(
            "aiccm2026dev-b home-auxiliary transform requires one complete "
            "character and translation per cyclic residue");
    }

    py::ssize_t index = 0;
    for (py::ssize_t i = 0; i < n0; ++i) {
        for (py::ssize_t j = 0; j < n1; ++j) {
            for (py::ssize_t l = 0; l < n2; ++l, ++index) {
                if (t(index, 0) != i || t(index, 1) != j || t(index, 2) != l) {
                    throw std::runtime_error(
                        "aiccm2026dev-b home-auxiliary transform requires "
                        "the complete lexicographic cyclic translation group");
                }
            }
        }
    }

    std::vector<unsigned char> seen(static_cast<std::size_t>(n_cells), 0);
    const std::array<py::ssize_t, 3> dimensions = {n0, n1, n2};
    for (py::ssize_t ik = 0; ik < n_cells; ++ik) {
        std::array<py::ssize_t, 3> residue{};
        for (py::ssize_t axis = 0; axis < 3; ++axis) {
            const py::ssize_t dimension =
                dimensions[static_cast<std::size_t>(axis)];
            const double scaled = k(ik, axis) * static_cast<double>(dimension);
            const long double scaled_wide =
                static_cast<long double>(k(ik, axis)) *
                static_cast<long double>(dimension);
            const long double lower = static_cast<long double>(
                std::numeric_limits<py::ssize_t>::lowest());
            const long double upper = static_cast<long double>(
                std::numeric_limits<py::ssize_t>::max());
            if (!std::isfinite(scaled_wide) || scaled_wide < lower ||
                scaled_wide > upper) {
                throw std::runtime_error(
                    "aiccm2026dev-b home-auxiliary transform requires finite "
                    "dual cyclic characters in the supported integer range");
            }
            const double nearest = std::nearbyint(scaled);
            if (std::abs(scaled - nearest) > 1.0e-9) {
                throw std::runtime_error(
                    "aiccm2026dev-b home-auxiliary transform requires kfrac "
                    "to lie on the dual cyclic character mesh");
            }
            const auto integer = static_cast<py::ssize_t>(nearest);
            residue[static_cast<std::size_t>(axis)] =
                (integer % dimension + dimension) % dimension;
        }
        const std::size_t character = static_cast<std::size_t>(
            (residue[0] * n1 + residue[1]) * n2 + residue[2]);
        if (seen[character]) {
            throw std::runtime_error(
                "aiccm2026dev-b home-auxiliary transform requires distinct "
                "dual cyclic characters");
        }
        seen[character] = 1;
    }
}

py::tuple lpq_cache_to_real_home_auxiliary(
    py::dict lpq_cache,
    py::array_t<double, py::array::c_style | py::array::forcecast> kfrac,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        translations,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh) {
    validate_home_auxiliary_group(kfrac, translations, mesh);
    return lpq_cache_to_real_supercell_impl(
        std::move(lpq_cache),
        std::move(kfrac),
        std::move(translations),
        true);
}

py::array_t<double> three_index_mo_transform_real(
    py::array_t<double, py::array::c_style | py::array::forcecast> factors,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_left,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_right) {
    const py::buffer_info binfo = factors.request();
    const py::buffer_info linfo = c_left.request();
    const py::buffer_info rinfo = c_right.request();
    require_rank(binfo, 3, "factors");
    require_rank(linfo, 2, "c_left");
    require_rank(rinfo, 2, "c_right");

    const py::ssize_t n_aux = binfo.shape[0];
    const py::ssize_t n_ao = binfo.shape[1];
    if (binfo.shape[2] != n_ao || linfo.shape[0] != n_ao ||
        rinfo.shape[0] != n_ao) {
        throw std::runtime_error(
            "aiccm2026dev-b real MO transform shape mismatch: factors " +
            shape_string<double>(binfo) + ", c_left " +
            shape_string<double>(linfo) + ", c_right " +
            shape_string<double>(rinfo));
    }
    const py::ssize_t n_left = linfo.shape[1];
    const py::ssize_t n_right = rinfo.shape[1];
    py::array_t<double> out({n_aux, n_left, n_right});

    const double* bptr = static_cast<const double*>(binfo.ptr);
    const double* lptr = static_cast<const double*>(linfo.ptr);
    const double* rptr = static_cast<const double*>(rinfo.ptr);
    double* optr = static_cast<double*>(out.request().ptr);

    {
        py::gil_scoped_release release;
        Eigen::Map<const RowMatrixXd> L(lptr, n_ao, n_left);
        Eigen::Map<const RowMatrixXd> R(rptr, n_ao, n_right);
        #pragma omp parallel for schedule(static)
        for (py::ssize_t p = 0; p < n_aux; ++p) {
            Eigen::Map<const RowMatrixXd> B(
                bptr + static_cast<std::size_t>(p * n_ao * n_ao),
                n_ao, n_ao);
            RowMatrixXd tmp = B * R;
            Eigen::Map<RowMatrixXd> O(
                optr + static_cast<std::size_t>(p * n_left * n_right),
                n_left, n_right);
            O.noalias() = L.transpose() * tmp;
        }
    }
    return out;
}

py::array_t<Complex> three_index_mo_transform_complex(
    py::array_t<Complex, py::array::c_style | py::array::forcecast> factors,
    py::array_t<Complex, py::array::c_style | py::array::forcecast> c_left,
    py::array_t<Complex, py::array::c_style | py::array::forcecast> c_right) {
    const py::buffer_info binfo = factors.request();
    const py::buffer_info linfo = c_left.request();
    const py::buffer_info rinfo = c_right.request();
    require_rank(binfo, 3, "factors");
    require_rank(linfo, 2, "c_left");
    require_rank(rinfo, 2, "c_right");

    const py::ssize_t n_aux = binfo.shape[0];
    const py::ssize_t n_ao = binfo.shape[1];
    if (binfo.shape[2] != n_ao || linfo.shape[0] != n_ao ||
        rinfo.shape[0] != n_ao) {
        throw std::runtime_error(
            "aiccm2026dev-b complex MO transform shape mismatch: factors " +
            shape_string<Complex>(binfo) + ", c_left " +
            shape_string<Complex>(linfo) + ", c_right " +
            shape_string<Complex>(rinfo));
    }
    const py::ssize_t n_left = linfo.shape[1];
    const py::ssize_t n_right = rinfo.shape[1];
    py::array_t<Complex> out({n_aux, n_left, n_right});

    const Complex* bptr = static_cast<const Complex*>(binfo.ptr);
    const Complex* lptr = static_cast<const Complex*>(linfo.ptr);
    const Complex* rptr = static_cast<const Complex*>(rinfo.ptr);
    Complex* optr = static_cast<Complex*>(out.request().ptr);

    {
        py::gil_scoped_release release;
        Eigen::Map<const RowMatrixXcd> L(lptr, n_ao, n_left);
        Eigen::Map<const RowMatrixXcd> R(rptr, n_ao, n_right);
        #pragma omp parallel for schedule(static)
        for (py::ssize_t p = 0; p < n_aux; ++p) {
            Eigen::Map<const RowMatrixXcd> B(
                bptr + static_cast<std::size_t>(p * n_ao * n_ao),
                n_ao, n_ao);
            RowMatrixXcd tmp = B * R;
            Eigen::Map<RowMatrixXcd> O(
                optr + static_cast<std::size_t>(p * n_left * n_right),
                n_left, n_right);
            O.noalias() = L.conjugate().transpose() * tmp;
        }
    }
    return out;
}

py::tuple canonical_mp2_energy_from_lov(
    py::dict lov_cache,
    py::array_t<double, py::array::c_style | py::array::forcecast> energies,
    std::array<int, 3> mesh,
    std::array<int, 3> is_shift,
    double denominator_tolerance) {
    const py::buffer_info einfo = energies.request();
    require_rank(einfo, 2, "energies");

    const py::ssize_t n_k = einfo.shape[0];
    const vibeqc::RegularKMesh regular_mesh(mesh, is_shift);
    if (n_k <= 0 ||
        static_cast<std::size_t>(n_k) != regular_mesh.size()) {
        throw std::runtime_error(
            "aiccm2026dev-b MP2 kernel regular mesh contains " +
            std::to_string(regular_mesh.size()) +
            " points, but energies has " + std::to_string(n_k) +
            " k-point rows");
    }

    using ComplexArray =
        py::array_t<Complex, py::array::c_style | py::array::forcecast>;
    std::vector<ComplexArray> arrays(static_cast<std::size_t>(n_k * n_k));
    std::vector<const Complex*> blocks(static_cast<std::size_t>(n_k * n_k));

    py::ssize_t n_aux = -1;
    py::ssize_t n_occ = -1;
    py::ssize_t n_vir = -1;
    for (py::ssize_t ki = 0; ki < n_k; ++ki) {
        for (py::ssize_t ka = 0; ka < n_k; ++ka) {
            const py::object key = py::make_tuple(ki, ka);
            if (!lov_cache.contains(key)) {
                throw std::runtime_error(
                    "aiccm2026dev-b MP2 kernel missing lov_cache[(" +
                    std::to_string(ki) + ", " + std::to_string(ka) + ")]");
            }
            ComplexArray arr = ComplexArray::ensure(lov_cache[key]);
            if (!arr) {
                throw std::runtime_error(
                    "aiccm2026dev-b MP2 kernel could not cast LOV cache "
                    "entry to complex128 C-contiguous array");
            }
            const py::buffer_info info = arr.request();
            require_rank(info, 3, "lov_cache entry");
            if (n_aux < 0) {
                n_aux = info.shape[0];
                n_occ = info.shape[1];
                n_vir = info.shape[2];
            } else if (info.shape[0] != n_aux || info.shape[1] != n_occ ||
                       info.shape[2] != n_vir) {
                throw std::runtime_error(
                    "aiccm2026dev-b MP2 kernel got inconsistent LOV shape " +
                    shape_string<Complex>(info));
            }
            const std::size_t index = static_cast<std::size_t>(ki * n_k + ka);
            arrays[index] = std::move(arr);
            blocks[index] = arrays[index].data();
        }
    }
    if (n_aux <= 0 || n_occ <= 0 || n_vir <= 0) {
        throw std::runtime_error("aiccm2026dev-b MP2 kernel got an empty LOV cache");
    }
    if (einfo.shape[1] != n_occ + n_vir) {
        throw std::runtime_error(
            "aiccm2026dev-b MP2 kernel energy shape does not match LOV "
            "occupied/virtual ranks");
    }

    const auto eps = energies.unchecked<2>();
    const double inv_nk = 1.0 / static_cast<double>(n_k);
    double e_ss = 0.0;
    double e_os = 0.0;
    double max_imag = 0.0;
    int bad_denominator = 0;

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(3) schedule(dynamic) \
            reduction(+ : e_ss, e_os) \
            reduction(max : max_imag, bad_denominator)
        for (py::ssize_t ki = 0; ki < n_k; ++ki) {
            for (py::ssize_t kj = 0; kj < n_k; ++kj) {
                for (py::ssize_t ka = 0; ka < n_k; ++ka) {
                    const py::ssize_t kb = static_cast<py::ssize_t>(
                        regular_mesh.conserved_index(
                            static_cast<std::size_t>(ki),
                            static_cast<std::size_t>(ka),
                            static_cast<std::size_t>(kj)));
                    const Complex* lov_ia =
                        blocks[static_cast<std::size_t>(ki * n_k + ka)];
                    const Complex* lov_jb =
                        blocks[static_cast<std::size_t>(kj * n_k + kb)];
                    const Complex* lov_ib =
                        blocks[static_cast<std::size_t>(ki * n_k + kb)];
                    const Complex* lov_ja =
                        blocks[static_cast<std::size_t>(kj * n_k + ka)];
                    Complex direct_sum{0.0, 0.0};
                    Complex exchange_sum{0.0, 0.0};
                    for (py::ssize_t i = 0; i < n_occ; ++i) {
                        const double eps_i = eps(ki, i);
                        for (py::ssize_t j = 0; j < n_occ; ++j) {
                            const double eps_ij = eps_i + eps(kj, j);
                            for (py::ssize_t a = 0; a < n_vir; ++a) {
                                const double eps_a = eps(ka, n_occ + a);
                                for (py::ssize_t b = 0; b < n_vir; ++b) {
                                    const double denom =
                                        eps_ij - eps_a - eps(kb, n_occ + b);
                                    if (denom >= -denominator_tolerance) {
                                        bad_denominator = 1;
                                        continue;
                                    }
                                    Complex gijab{0.0, 0.0};
                                    Complex gijba_exchange{0.0, 0.0};
                                    for (py::ssize_t p = 0; p < n_aux; ++p) {
                                        const std::size_t pia =
                                            static_cast<std::size_t>(
                                                (p * n_occ + i) * n_vir + a);
                                        const std::size_t pjb =
                                            static_cast<std::size_t>(
                                                (p * n_occ + j) * n_vir + b);
                                        const std::size_t pib =
                                            static_cast<std::size_t>(
                                                (p * n_occ + i) * n_vir + b);
                                        const std::size_t pja =
                                            static_cast<std::size_t>(
                                                (p * n_occ + j) * n_vir + a);
                                        gijab += lov_ia[pia] * lov_jb[pjb];
                                        gijba_exchange +=
                                            lov_ib[pib] * lov_ja[pja];
                                    }
                                    gijab *= inv_nk;
                                    gijba_exchange *= inv_nk;
                                    const Complex amplitude =
                                        std::conj(gijab / denom);
                                    direct_sum += 2.0 * amplitude * gijab;
                                    exchange_sum -= amplitude * gijba_exchange;
                                }
                            }
                        }
                    }
                    max_imag = std::max(max_imag, std::abs(direct_sum.imag()));
                    max_imag = std::max(max_imag, std::abs(exchange_sum.imag()));
                    const double direct = direct_sum.real();
                    const double exchange = exchange_sum.real();
                    e_ss += 0.5 * direct + exchange;
                    e_os += 0.5 * direct;
                }
            }
        }
    }

    if (bad_denominator) {
        throw std::runtime_error(
            "aiccm2026dev-b MP2 kernel encountered a non-negative "
            "orbital energy denominator");
    }

    e_ss *= inv_nk;
    e_os *= inv_nk;
    return py::make_tuple(e_ss, e_os, max_imag);
}

double real_mp2_energy_from_lov(
    py::array_t<double, py::array::c_style | py::array::forcecast> lov,
    py::array_t<double, py::array::c_style | py::array::forcecast> eps_occ,
    py::array_t<double, py::array::c_style | py::array::forcecast> eps_vir,
    double denominator_tolerance) {
    const py::buffer_info linfo = lov.request();
    const py::buffer_info oinfo = eps_occ.request();
    const py::buffer_info vinfo = eps_vir.request();
    require_rank(linfo, 3, "lov");
    require_rank(oinfo, 1, "eps_occ");
    require_rank(vinfo, 1, "eps_vir");
    const py::ssize_t n_aux = linfo.shape[0];
    const py::ssize_t n_occ = linfo.shape[1];
    const py::ssize_t n_vir = linfo.shape[2];
    if (n_aux <= 0 || n_occ <= 0 || n_vir <= 0 ||
        oinfo.shape[0] != n_occ || vinfo.shape[0] != n_vir) {
        throw std::runtime_error(
            "aiccm2026dev-b real MP2 kernel shape mismatch: lov " +
            shape_string<double>(linfo) + ", eps_occ " +
            shape_string<double>(oinfo) + ", eps_vir " +
            shape_string<double>(vinfo));
    }

    const double* lptr = static_cast<const double*>(linfo.ptr);
    const double* optr = static_cast<const double*>(oinfo.ptr);
    const double* vptr = static_cast<const double*>(vinfo.ptr);
    double energy = 0.0;
    int bad_denominator = 0;

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(2) schedule(dynamic) \
            reduction(+ : energy) reduction(max : bad_denominator)
        for (py::ssize_t i = 0; i < n_occ; ++i) {
            for (py::ssize_t j = 0; j < n_occ; ++j) {
                const double eps_ij = optr[i] + optr[j];
                double pair_energy = 0.0;
                for (py::ssize_t a = 0; a < n_vir; ++a) {
                    for (py::ssize_t b = 0; b < n_vir; ++b) {
                        const double denom = eps_ij - vptr[a] - vptr[b];
                        if (denom >= -denominator_tolerance) {
                            bad_denominator = 1;
                            continue;
                        }
                        double gijab = 0.0;
                        double gijba = 0.0;
                        for (py::ssize_t p = 0; p < n_aux; ++p) {
                            const std::size_t pia =
                                static_cast<std::size_t>(
                                    (p * n_occ + i) * n_vir + a);
                            const std::size_t pjb =
                                static_cast<std::size_t>(
                                    (p * n_occ + j) * n_vir + b);
                            const std::size_t pib =
                                static_cast<std::size_t>(
                                    (p * n_occ + i) * n_vir + b);
                            const std::size_t pja =
                                static_cast<std::size_t>(
                                    (p * n_occ + j) * n_vir + a);
                            gijab += lptr[pia] * lptr[pjb];
                            gijba += lptr[pib] * lptr[pja];
                        }
                        const double amplitude = gijab / denom;
                        pair_energy += amplitude * (2.0 * gijab - gijba);
                    }
                }
                energy += pair_energy;
            }
        }
    }

    if (bad_denominator) {
        throw std::runtime_error(
            "aiccm2026dev-b real MP2 kernel encountered a non-negative "
            "orbital energy denominator");
    }
    return energy;
}

std::size_t cyclic_residue_index(py::ssize_t i,
                                 py::ssize_t j,
                                 py::ssize_t k,
                                 py::ssize_t n0,
                                 py::ssize_t n1,
                                 py::ssize_t n2) {
    const py::ssize_t ii = (i % n0 + n0) % n0;
    const py::ssize_t jj = (j % n1 + n1) % n1;
    const py::ssize_t kk = (k % n2 + n2) % n2;
    return static_cast<std::size_t>((ii * n1 + jj) * n2 + kk);
}

template <typename Scalar>
py::array_t<Scalar> home_auxiliary_three_index_mo_transform_impl(
    py::array_t<Scalar, py::array::c_style | py::array::forcecast>
        home_factors,
    py::array_t<Scalar, py::array::c_style | py::array::forcecast> c_left,
    py::array_t<Scalar, py::array::c_style | py::array::forcecast> c_right,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh,
    const char* scalar_label) {
    const py::buffer_info binfo = home_factors.request();
    const py::buffer_info linfo = c_left.request();
    const py::buffer_info rinfo = c_right.request();
    const py::buffer_info minfo = mesh.request();
    require_rank(binfo, 3, "home_factors");
    require_rank(linfo, 2, "c_left");
    require_rank(rinfo, 2, "c_right");
    require_rank(minfo, 1, "mesh");
    if (minfo.shape[0] != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b home-auxiliary MO transform requires mesh "
            "shape (3,)");
    }

    const auto mesh_u = mesh.unchecked<1>();
    const py::ssize_t n0 = static_cast<py::ssize_t>(mesh_u(0));
    const py::ssize_t n1 = static_cast<py::ssize_t>(mesh_u(1));
    const py::ssize_t n2 = static_cast<py::ssize_t>(mesh_u(2));
    const py::ssize_t n_cells = checked_positive_mesh_product(
        n0, n1, n2, "home-auxiliary MO transform");
    const py::ssize_t n_aux_home = binfo.shape[0];
    const py::ssize_t n_ao = binfo.shape[1];
    if (n_aux_home <= 0 || n_ao <= 0 || binfo.shape[2] != n_ao ||
        n_ao % n_cells != 0 || linfo.shape[0] != n_ao ||
        rinfo.shape[0] != n_ao) {
        throw std::runtime_error(
            std::string("aiccm2026dev-b ") + scalar_label +
            " home-auxiliary MO transform shape mismatch: factors " +
            shape_string<Scalar>(binfo) + ", c_left " +
            shape_string<Scalar>(linfo) + ", c_right " +
            shape_string<Scalar>(rinfo));
    }

    const py::ssize_t nbf = n_ao / n_cells;
    const py::ssize_t n_left = linfo.shape[1];
    const py::ssize_t n_right = rinfo.shape[1];
    const py::ssize_t n_aux_super = checked_positive_product(
        n_cells, n_aux_home, "home-auxiliary logical auxiliary rank");
    py::array_t<Scalar> out({n_aux_super, n_left, n_right});

    std::vector<std::array<py::ssize_t, 3>> residues(
        static_cast<std::size_t>(n_cells));
    for (py::ssize_t i = 0; i < n0; ++i) {
        for (py::ssize_t j = 0; j < n1; ++j) {
            for (py::ssize_t k = 0; k < n2; ++k) {
                residues[cyclic_residue_index(i, j, k, n0, n1, n2)] = {i, j, k};
            }
        }
    }

    const Scalar* bptr = static_cast<const Scalar*>(binfo.ptr);
    const Scalar* lptr = static_cast<const Scalar*>(linfo.ptr);
    const Scalar* rptr = static_cast<const Scalar*>(rinfo.ptr);
    Scalar* optr = static_cast<Scalar*>(out.request().ptr);

    using RowMatrixXs =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    {
        py::gil_scoped_release release;
        #pragma omp parallel
        {
            RowMatrixXs L(n_ao, n_left);
            RowMatrixXs R(n_ao, n_right);
            py::ssize_t loaded_translation = -1;
            #pragma omp for schedule(static)
            for (py::ssize_t ps = 0; ps < n_aux_super; ++ps) {
                const py::ssize_t t_cell = ps / n_aux_home;
                const py::ssize_t p_aux = ps % n_aux_home;
                if (t_cell != loaded_translation) {
                    const auto rt = residues[static_cast<std::size_t>(t_cell)];
                    for (py::ssize_t row = 0; row < n_ao; ++row) {
                        const auto rr =
                            residues[static_cast<std::size_t>(row / nbf)];
                        const py::ssize_t source_cell =
                            static_cast<py::ssize_t>(cyclic_residue_index(
                                rr[0] + rt[0],
                                rr[1] + rt[1],
                                rr[2] + rt[2],
                                n0,
                                n1,
                                n2));
                        const py::ssize_t source_row =
                            source_cell * nbf + row % nbf;
                        std::copy_n(
                            lptr + static_cast<std::size_t>(source_row * n_left),
                            static_cast<std::size_t>(n_left),
                            L.data() + static_cast<std::size_t>(row * n_left));
                        std::copy_n(
                            rptr + static_cast<std::size_t>(source_row * n_right),
                            static_cast<std::size_t>(n_right),
                            R.data() + static_cast<std::size_t>(row * n_right));
                    }
                    loaded_translation = t_cell;
                }
                Eigen::Map<const RowMatrixXs> B(
                    bptr + static_cast<std::size_t>(p_aux * n_ao * n_ao),
                    n_ao,
                    n_ao);
                RowMatrixXs tmp = B * R;
                Eigen::Map<RowMatrixXs> O(
                    optr + static_cast<std::size_t>(ps * n_left * n_right),
                    n_left,
                    n_right);
                if constexpr (std::is_same_v<Scalar, double>) {
                    O.noalias() = L.transpose() * tmp;
                } else {
                    O.noalias() = L.adjoint() * tmp;
                }
            }
        }
    }
    return out;
}

py::array_t<double> home_auxiliary_three_index_mo_transform_real(
    py::array_t<double, py::array::c_style | py::array::forcecast>
        home_factors,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_left,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_right,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh) {
    return home_auxiliary_three_index_mo_transform_impl<double>(
        std::move(home_factors),
        std::move(c_left),
        std::move(c_right),
        std::move(mesh),
        "real");
}

py::array_t<Complex> home_auxiliary_three_index_mo_transform_complex(
    py::array_t<Complex, py::array::c_style | py::array::forcecast>
        home_factors,
    py::array_t<Complex, py::array::c_style | py::array::forcecast> c_left,
    py::array_t<Complex, py::array::c_style | py::array::forcecast> c_right,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh) {
    return home_auxiliary_three_index_mo_transform_impl<Complex>(
        std::move(home_factors),
        std::move(c_left),
        std::move(c_right),
        std::move(mesh),
        "complex");
}

std::pair<py::ssize_t, py::ssize_t> canonical_occupied_pair(
    py::ssize_t i,
    py::ssize_t j) {
    return (i <= j) ? std::make_pair(i, j) : std::make_pair(j, i);
}

std::size_t unordered_pair_index(py::ssize_t i,
                                 py::ssize_t j,
                                 py::ssize_t n_orbitals) {
    if (i > j) std::swap(i, j);
    return static_cast<std::size_t>(
        i * n_orbitals - (i * (i - 1)) / 2 + (j - i));
}

py::array_t<std::int64_t> translation_permutations(
    py::ssize_t n_bands,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh) {
    const py::buffer_info minfo = mesh.request();
    require_rank(minfo, 1, "mesh");
    if (minfo.shape[0] != 3) {
        throw std::runtime_error(
            "aiccm2026dev-b translation permutations require mesh shape (3,)");
    }
    if (n_bands < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b translation permutations need at least one band");
    }
    const auto mesh_u = mesh.unchecked<1>();
    const py::ssize_t n0 = static_cast<py::ssize_t>(mesh_u(0));
    const py::ssize_t n1 = static_cast<py::ssize_t>(mesh_u(1));
    const py::ssize_t n2 = static_cast<py::ssize_t>(mesh_u(2));
    if (n0 < 1 || n1 < 1 || n2 < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b translation permutation mesh entries must be positive");
    }
    const py::ssize_t n_cells = n0 * n1 * n2;
    const py::ssize_t n_orbitals = n_cells * n_bands;
    py::array_t<std::int64_t> out({n_cells, n_orbitals});
    auto out_u = out.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (py::ssize_t shift_index = 0; shift_index < n_cells; ++shift_index) {
            const py::ssize_t sx = shift_index / (n1 * n2);
            const py::ssize_t rem_s = shift_index - sx * n1 * n2;
            const py::ssize_t sy = rem_s / n2;
            const py::ssize_t sz = rem_s - sy * n2;
            for (py::ssize_t source_index = 0;
                 source_index < n_cells;
                 ++source_index) {
                const py::ssize_t cx = source_index / (n1 * n2);
                const py::ssize_t rem_c = source_index - cx * n1 * n2;
                const py::ssize_t cy = rem_c / n2;
                const py::ssize_t cz = rem_c - cy * n2;
                const py::ssize_t tx = (cx + sx) % n0;
                const py::ssize_t ty = (cy + sy) % n1;
                const py::ssize_t tz = (cz + sz) % n2;
                const py::ssize_t target_index = (tx * n1 + ty) * n2 + tz;
                for (py::ssize_t band = 0; band < n_bands; ++band) {
                    out_u(shift_index, source_index * n_bands + band) =
                        static_cast<std::int64_t>(
                            target_index * n_bands + band);
                }
            }
        }
    }

    return out;
}

py::tuple pair_orbits_from_permutations(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        permutations,
    py::ssize_t n_cells) {
    const py::buffer_info pinfo = permutations.request();
    require_rank(pinfo, 2, "permutations");
    if (n_cells < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b pair orbit n_cells must be positive");
    }
    const py::ssize_t n_generators = pinfo.shape[0];
    const py::ssize_t n_orbitals = pinfo.shape[1];
    if (n_generators < 1 || n_orbitals < 1) {
        throw std::runtime_error(
            "aiccm2026dev-b pair orbits require at least one permutation "
            "over at least one occupied orbital");
    }
    const auto perm = permutations.unchecked<2>();

    int invalid_mapping = 0;
    #pragma omp parallel for schedule(static) reduction(max : invalid_mapping)
    for (py::ssize_t g = 0; g < n_generators; ++g) {
        std::vector<unsigned char> seen(static_cast<std::size_t>(n_orbitals), 0);
        for (py::ssize_t i = 0; i < n_orbitals; ++i) {
            const py::ssize_t target = static_cast<py::ssize_t>(perm(g, i));
            if (target < 0 || target >= n_orbitals ||
                seen[static_cast<std::size_t>(target)] != 0) {
                invalid_mapping = 1;
                break;
            }
            seen[static_cast<std::size_t>(target)] = 1;
        }
    }
    if (invalid_mapping) {
        throw std::runtime_error(
            "aiccm2026dev-b pair mapping is not a permutation");
    }

    const py::ssize_t n_pairs = n_orbitals * (n_orbitals + 1) / 2;
    std::vector<unsigned char> assigned(static_cast<std::size_t>(n_pairs), 0);
    std::vector<int> orbit_marks(static_cast<std::size_t>(n_pairs), -1);
    std::vector<std::vector<std::pair<py::ssize_t, py::ssize_t>>> orbits;
    orbits.reserve(static_cast<std::size_t>(n_pairs));

    {
        py::gil_scoped_release release;
        int orbit_id = 0;
        for (py::ssize_t i_seed = 0; i_seed < n_orbitals; ++i_seed) {
            for (py::ssize_t j_seed = i_seed; j_seed < n_orbitals; ++j_seed) {
                const std::size_t seed_index =
                    unordered_pair_index(i_seed, j_seed, n_orbitals);
                if (assigned[seed_index] != 0) {
                    continue;
                }

                std::vector<std::pair<py::ssize_t, py::ssize_t>> members;
                std::vector<std::pair<py::ssize_t, py::ssize_t>> frontier;
                members.emplace_back(i_seed, j_seed);
                frontier.emplace_back(i_seed, j_seed);
                orbit_marks[seed_index] = orbit_id;

                while (!frontier.empty()) {
                    const auto current = frontier.back();
                    frontier.pop_back();
                    for (py::ssize_t g = 0; g < n_generators; ++g) {
                        auto mapped = canonical_occupied_pair(
                            static_cast<py::ssize_t>(perm(g, current.first)),
                            static_cast<py::ssize_t>(perm(g, current.second)));
                        const std::size_t mapped_index =
                            unordered_pair_index(
                                mapped.first, mapped.second, n_orbitals);
                        if (orbit_marks[mapped_index] != orbit_id) {
                            orbit_marks[mapped_index] = orbit_id;
                            members.push_back(mapped);
                            frontier.push_back(mapped);
                        }
                    }
                }

                std::sort(members.begin(), members.end());
                for (const auto& member : members) {
                    assigned[unordered_pair_index(
                        member.first, member.second, n_orbitals)] = 1;
                }
                orbits.push_back(std::move(members));
                ++orbit_id;
            }
        }
    }

    std::size_t member_count = 0;
    for (const auto& orbit : orbits) {
        member_count += orbit.size();
    }
    if (member_count != static_cast<std::size_t>(n_pairs)) {
        throw std::runtime_error(
            "aiccm2026dev-b pair orbits do not partition pair space");
    }

    py::array_t<std::int64_t> representatives(
        {static_cast<py::ssize_t>(orbits.size()), py::ssize_t{2}});
    py::array_t<std::int64_t> offsets(
        {static_cast<py::ssize_t>(orbits.size() + 1)});
    py::array_t<std::int64_t> members(
        {static_cast<py::ssize_t>(member_count), py::ssize_t{2}});
    auto reps_u = representatives.mutable_unchecked<2>();
    auto offsets_u = offsets.mutable_unchecked<1>();
    auto members_u = members.mutable_unchecked<2>();

    py::ssize_t cursor = 0;
    offsets_u(0) = 0;
    for (py::ssize_t orbit_index = 0;
         orbit_index < static_cast<py::ssize_t>(orbits.size());
         ++orbit_index) {
        const auto& orbit = orbits[static_cast<std::size_t>(orbit_index)];
        reps_u(orbit_index, 0) = static_cast<std::int64_t>(orbit.front().first);
        reps_u(orbit_index, 1) = static_cast<std::int64_t>(orbit.front().second);
        for (const auto& member : orbit) {
            members_u(cursor, 0) = static_cast<std::int64_t>(member.first);
            members_u(cursor, 1) = static_cast<std::int64_t>(member.second);
            ++cursor;
        }
        offsets_u(orbit_index + 1) = static_cast<std::int64_t>(cursor);
    }

    return py::make_tuple(representatives, offsets, members);
}

py::array_t<double> mayer_pair_tensor_from_blocks(
    py::array_t<double, py::array::c_style | py::array::forcecast>
        density_blocks,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        overlap_blocks,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        ao_atoms_unit,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh,
    py::object density_beta_blocks_obj) {
    const py::buffer_info dinfo = density_blocks.request();
    const py::buffer_info sinfo = overlap_blocks.request();
    const py::buffer_info ainfo = ao_atoms_unit.request();
    const py::buffer_info minfo = mesh.request();
    require_rank(dinfo, 3, "density_blocks");
    require_rank(sinfo, 3, "overlap_blocks");
    require_rank(ainfo, 1, "ao_atoms_unit");
    require_rank(minfo, 1, "mesh");
    if (minfo.shape[0] != 3) {
        throw std::runtime_error("mesh must have shape (3,)");
    }
    const py::ssize_t n_cells = dinfo.shape[0];
    const py::ssize_t nbf = dinfo.shape[1];
    if (n_cells <= 0 || nbf <= 0 || dinfo.shape[2] != nbf) {
        throw std::runtime_error(
            "density_blocks must have shape (n_cells, nbf, nbf)");
    }
    if (sinfo.shape[0] != n_cells || sinfo.shape[1] != nbf ||
        sinfo.shape[2] != nbf) {
        throw std::runtime_error(
            "overlap_blocks shape does not match density_blocks");
    }
    if (ainfo.shape[0] != nbf) {
        throw std::runtime_error(
            "ao_atoms_unit length must match the AO dimension");
    }

    const auto mesh_u = mesh.unchecked<1>();
    const py::ssize_t n0 = static_cast<py::ssize_t>(mesh_u(0));
    const py::ssize_t n1 = static_cast<py::ssize_t>(mesh_u(1));
    const py::ssize_t n2 = static_cast<py::ssize_t>(mesh_u(2));
    if (n0 <= 0 || n1 <= 0 || n2 <= 0 || n0 * n1 * n2 != n_cells) {
        throw std::runtime_error(
            "mesh product must match density_blocks.shape[0]");
    }

    using DoubleArray =
        py::array_t<double, py::array::c_style | py::array::forcecast>;
    DoubleArray beta_blocks;
    bool unrestricted = !density_beta_blocks_obj.is_none();
    if (unrestricted) {
        beta_blocks = DoubleArray::ensure(density_beta_blocks_obj);
        if (!beta_blocks) {
            throw std::runtime_error(
                "density_beta_blocks could not be cast to a float64 "
                "C-contiguous array");
        }
        const py::buffer_info binfo = beta_blocks.request();
        require_rank(binfo, 3, "density_beta_blocks");
        if (binfo.shape[0] != n_cells || binfo.shape[1] != nbf ||
            binfo.shape[2] != nbf) {
            throw std::runtime_error(
                "density_beta_blocks shape does not match density_blocks");
        }
    }

    const auto atoms = ao_atoms_unit.unchecked<1>();
    py::ssize_t n_atoms = 0;
    for (py::ssize_t mu = 0; mu < nbf; ++mu) {
        if (atoms(mu) < 0) {
            throw std::runtime_error("ao_atoms_unit contains a negative atom index");
        }
        n_atoms = std::max(n_atoms, static_cast<py::ssize_t>(atoms(mu)) + 1);
    }
    std::vector<std::vector<py::ssize_t>> ao_by_atom(
        static_cast<std::size_t>(n_atoms));
    for (py::ssize_t mu = 0; mu < nbf; ++mu) {
        ao_by_atom[static_cast<std::size_t>(atoms(mu))].push_back(mu);
    }

    std::vector<std::array<py::ssize_t, 3>> residues(
        static_cast<std::size_t>(n_cells));
    for (py::ssize_t i = 0; i < n0; ++i) {
        for (py::ssize_t j = 0; j < n1; ++j) {
            for (py::ssize_t k = 0; k < n2; ++k) {
                residues[cyclic_residue_index(i, j, k, n0, n1, n2)] = {i, j, k};
            }
        }
    }

    const double* dptr = static_cast<const double*>(dinfo.ptr);
    const double* sptr = static_cast<const double*>(sinfo.ptr);
    const double* bptr =
        unrestricted ? static_cast<const double*>(beta_blocks.request().ptr)
                     : nullptr;
    const std::size_t block_size =
        static_cast<std::size_t>(nbf * nbf);
    std::vector<RowMatrixXd> ps_alpha(static_cast<std::size_t>(n_cells));
    std::vector<RowMatrixXd> ps_beta(
        unrestricted ? static_cast<std::size_t>(n_cells) : 0);

    py::array_t<double> out({n_cells, n_atoms, n_atoms});
    auto out_u = out.mutable_unchecked<3>();

    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(static)
        for (py::ssize_t d = 0; d < n_cells; ++d) {
            RowMatrixXd accum = RowMatrixXd::Zero(nbf, nbf);
            const auto delta = residues[static_cast<std::size_t>(d)];
            for (py::ssize_t g = 0; g < n_cells; ++g) {
                const auto gamma = residues[static_cast<std::size_t>(g)];
                const std::size_t h = cyclic_residue_index(
                    delta[0] - gamma[0],
                    delta[1] - gamma[1],
                    delta[2] - gamma[2],
                    n0,
                    n1,
                    n2);
                Eigen::Map<const RowMatrixXd> P(
                    dptr + static_cast<std::size_t>(g) * block_size,
                    nbf,
                    nbf);
                Eigen::Map<const RowMatrixXd> S(
                    sptr + h * block_size,
                    nbf,
                    nbf);
                accum.noalias() += P * S;
            }
            ps_alpha[static_cast<std::size_t>(d)] = std::move(accum);
        }

        if (unrestricted) {
            #pragma omp parallel for schedule(static)
            for (py::ssize_t d = 0; d < n_cells; ++d) {
                RowMatrixXd accum = RowMatrixXd::Zero(nbf, nbf);
                const auto delta = residues[static_cast<std::size_t>(d)];
                for (py::ssize_t g = 0; g < n_cells; ++g) {
                    const auto gamma = residues[static_cast<std::size_t>(g)];
                    const std::size_t h = cyclic_residue_index(
                        delta[0] - gamma[0],
                        delta[1] - gamma[1],
                        delta[2] - gamma[2],
                        n0,
                        n1,
                        n2);
                    Eigen::Map<const RowMatrixXd> P(
                        bptr + static_cast<std::size_t>(g) * block_size,
                        nbf,
                        nbf);
                    Eigen::Map<const RowMatrixXd> S(
                        sptr + h * block_size,
                        nbf,
                        nbf);
                    accum.noalias() += P * S;
                }
                ps_beta[static_cast<std::size_t>(d)] = std::move(accum);
            }
        }

        #pragma omp parallel for schedule(static)
        for (py::ssize_t d = 0; d < n_cells; ++d) {
            const auto delta = residues[static_cast<std::size_t>(d)];
            const std::size_t neg = cyclic_residue_index(
                -delta[0],
                -delta[1],
                -delta[2],
                n0,
                n1,
                n2);
            const RowMatrixXd& psd = ps_alpha[static_cast<std::size_t>(d)];
            const RowMatrixXd& psn = ps_alpha[neg];
            const RowMatrixXd* psd_beta = nullptr;
            const RowMatrixXd* psn_beta = nullptr;
            if (unrestricted) {
                psd_beta = &ps_beta[static_cast<std::size_t>(d)];
                psn_beta = &ps_beta[neg];
            }
            for (py::ssize_t atom_a = 0; atom_a < n_atoms; ++atom_a) {
                const auto& aos_a = ao_by_atom[static_cast<std::size_t>(atom_a)];
                for (py::ssize_t atom_b = 0; atom_b < n_atoms; ++atom_b) {
                    const auto& aos_b =
                        ao_by_atom[static_cast<std::size_t>(atom_b)];
                    double value_alpha = 0.0;
                    double value_beta = 0.0;
                    for (py::ssize_t mu : aos_a) {
                        for (py::ssize_t nu : aos_b) {
                            value_alpha += psd(mu, nu) * psn(nu, mu);
                            if (unrestricted) {
                                value_beta += (*psd_beta)(mu, nu) *
                                              (*psn_beta)(nu, mu);
                            }
                        }
                    }
                    const double value =
                        unrestricted ? 2.0 * (value_alpha + value_beta)
                                     : value_alpha;
                    out_u(d, atom_a, atom_b) = value;
                }
            }
        }
    }

    return out;
}

py::array_t<double> mayer_pair_matrix_from_blocks(
    py::array_t<double, py::array::c_style | py::array::forcecast>
        density_blocks,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        overlap_blocks,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        ao_atoms_unit,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> mesh,
    py::object density_beta_blocks_obj) {
    py::array_t<double> tensor = mayer_pair_tensor_from_blocks(
        density_blocks,
        overlap_blocks,
        ao_atoms_unit,
        mesh,
        density_beta_blocks_obj);
    const py::buffer_info tinfo = tensor.request();
    const py::buffer_info minfo = mesh.request();
    require_rank(tinfo, 3, "Mayer pair tensor");
    require_rank(minfo, 1, "mesh");
    if (minfo.shape[0] != 3) {
        throw std::runtime_error("mesh must have shape (3,)");
    }
    const auto mesh_u = mesh.unchecked<1>();
    const py::ssize_t n0 = static_cast<py::ssize_t>(mesh_u(0));
    const py::ssize_t n1 = static_cast<py::ssize_t>(mesh_u(1));
    const py::ssize_t n2 = static_cast<py::ssize_t>(mesh_u(2));
    const py::ssize_t n_cells = tinfo.shape[0];
    const py::ssize_t n_atoms = tinfo.shape[1];
    if (tinfo.shape[2] != n_atoms || n0 * n1 * n2 != n_cells) {
        throw std::runtime_error(
            "Mayer pair tensor shape does not match the requested mesh");
    }

    std::vector<std::array<py::ssize_t, 3>> residues(
        static_cast<std::size_t>(n_cells));
    for (py::ssize_t i = 0; i < n0; ++i) {
        for (py::ssize_t j = 0; j < n1; ++j) {
            for (py::ssize_t k = 0; k < n2; ++k) {
                residues[cyclic_residue_index(i, j, k, n0, n1, n2)] = {i, j, k};
            }
        }
    }

    const py::ssize_t n_full_atoms = n_cells * n_atoms;
    py::array_t<double> out({n_full_atoms, n_full_atoms});
    auto out_u = out.mutable_unchecked<2>();
    const auto tensor_u = tensor.unchecked<3>();

    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(2) schedule(static)
        for (py::ssize_t origin = 0; origin < n_cells; ++origin) {
            for (py::ssize_t target = 0; target < n_cells; ++target) {
                const auto ro = residues[static_cast<std::size_t>(origin)];
                const auto rt = residues[static_cast<std::size_t>(target)];
                const std::size_t delta = cyclic_residue_index(
                    rt[0] - ro[0],
                    rt[1] - ro[1],
                    rt[2] - ro[2],
                    n0,
                    n1,
                    n2);
                for (py::ssize_t atom_a = 0; atom_a < n_atoms; ++atom_a) {
                    const py::ssize_t row = origin * n_atoms + atom_a;
                    for (py::ssize_t atom_b = 0; atom_b < n_atoms; ++atom_b) {
                        const py::ssize_t col = target * n_atoms + atom_b;
                        out_u(row, col) = tensor_u(delta, atom_a, atom_b);
                    }
                }
            }
        }
    }

    return out;
}

}  // namespace

void bind_aiccm2026dev_b_kernels(py::module_& m) {
    m.def(
        "aiccm2026dev_b_matrix_to_real_supercell",
        &matrix_to_real_supercell,
        py::arg("matrices_k"),
        py::arg("kpoints_frac"),
        py::arg("translations"),
        "OpenMP finite-character inverse transform for B-stream one-body "
        "matrices. Returns (real_symmetric_matrix, max_imaginary_residual).");

    m.def(
        "aiccm2026dev_b_inverse_bloch_transform",
        &inverse_bloch_transform_blocks,
        py::arg("matrices_k"),
        py::arg("kpoints_frac"),
        py::arg("translations"),
        py::arg("weights"),
        "OpenMP finite-character inverse Bloch transform for χ-CCM residue "
        "blocks, M(R)=sum_k w_k exp(-2*pi*i*k.R) M(k).");

    m.def(
        "aiccm2026dev_b_canonical_wannier_coefficients",
        &canonical_wannier_coefficients,
        py::arg("coefficients_k"),
        py::arg("kpoints_frac"),
        py::arg("translations"),
        py::arg("n_occ"),
        "OpenMP finite-character back-transform from Bloch occupied "
        "coefficients to canonical real-torus Wannier coefficients.");

    m.def(
        "aiccm2026dev_b_localization_projector_audit",
        &localization_projector_audit,
        py::arg("canonical"),
        py::arg("localized"),
        py::arg("fock"),
        py::arg("translations"),
        py::arg("mesh"),
        py::arg("nbf"),
        "Native low-rank projector invariants for χ-CCM occupied "
        "localization. Returns density, one-particle-energy, and cyclic "
        "translation-projector errors without materialising AO projectors.");

    m.def(
        "aiccm2026dev_b_lpq_to_real_supercell",
        &lpq_cache_to_real_supercell,
        py::arg("lpq_cache"),
        py::arg("kpoints_frac"),
        py::arg("translations"),
        "OpenMP streaming inverse transform of pair-resolved RI factors for "
        "aiccm2026dev-b. Returns (real_symmetric_factors, max_imag, max_sym) "
        "without materialising the historical 6-D complex tensor.");

    m.def(
        "aiccm2026dev_b_lpq_to_real_home_auxiliary",
        &lpq_cache_to_real_home_auxiliary,
        py::arg("lpq_cache"),
        py::arg("kpoints_frac"),
        py::arg("translations"),
        py::arg("mesh"),
        "OpenMP inverse transform of pair-resolved χ-CCM RI factors for the "
        "home auxiliary cell on a complete cyclic character group. A shared "
        "canonical primitive-auxiliary frame is required; translation "
        "covariance supplies every other logical auxiliary row.");

    m.def(
        "aiccm2026dev_b_3index_mo_transform",
        &three_index_mo_transform_real,
        py::arg("factors"),
        py::arg("c_left"),
        py::arg("c_right"),
        "OpenMP real three-index AO-to-MO transform "
        "B[P,i,j] = C_left.T @ B[P] @ C_right.");

    m.def(
        "aiccm2026dev_b_3index_mo_transform_complex",
        &three_index_mo_transform_complex,
        py::arg("factors"),
        py::arg("c_left"),
        py::arg("c_right"),
        "OpenMP complex three-index AO-to-MO transform "
        "B[P,i,j] = C_left.conj().T @ B[P] @ C_right.");

    m.def(
        "aiccm2026dev_b_home_auxiliary_3index_mo_transform",
        &home_auxiliary_three_index_mo_transform_real,
        py::arg("home_factors"),
        py::arg("c_left"),
        py::arg("c_right"),
        py::arg("mesh"),
        "OpenMP exact circulant AO-to-MO transform from home-auxiliary "
        "χ-CCM factors. Returns all translated auxiliary rows.");

    m.def(
        "aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex",
        &home_auxiliary_three_index_mo_transform_complex,
        py::arg("home_factors"),
        py::arg("c_left"),
        py::arg("c_right"),
        py::arg("mesh"),
        "OpenMP exact complex circulant AO-to-MO transform from "
        "home-auxiliary χ-CCM factors. Returns all translated auxiliary "
        "rows with conjugation on the left coefficients.");

    m.def(
        "aiccm2026dev_b_mp2_energy_from_lov",
        &canonical_mp2_energy_from_lov,
        py::arg("lov_cache"),
        py::arg("energies"),
        py::arg("mesh"),
        py::arg("is_shift") = std::array<int, 3>{0, 0, 0},
        py::arg("denominator_tolerance") = 1.0e-12,
        "OpenMP streaming B-stream RI-MP2 energy contraction from "
        "momentum-pair LOV factors. The conserved fourth k index is computed "
        "exactly and on demand from the regular mesh; no N_k^3 table is "
        "materialised. Returns (E_ss, E_os, max_imag) without materialising "
        "per-k oovv scratch tensors.");

    m.def(
        "aiccm2026dev_b_real_mp2_energy_from_lov",
        &real_mp2_energy_from_lov,
        py::arg("lov"),
        py::arg("eps_occ"),
        py::arg("eps_vir"),
        py::arg("denominator_tolerance") = 1.0e-12,
        "OpenMP streaming real-torus RI-MP2 energy contraction for the "
        "χ-CCM complete-domain local-correlation audit. The input is "
        "L[P,i,a]; the kernel avoids materialising the full ijab tensor.");

    m.def(
        "aiccm2026dev_b_pair_orbits",
        &pair_orbits_from_permutations,
        py::arg("permutations"),
        py::arg("n_cells"),
        "Native finite-group partition of unordered occupied pairs for "
        "χ-CCM local correlation. Returns "
        "(representatives, member_offsets, members) for the supplied "
        "permutation generators without Python set bookkeeping.");

    m.def(
        "aiccm2026dev_b_translation_permutations",
        &translation_permutations,
        py::arg("n_bands"),
        py::arg("mesh"),
        "Native finite-translation occupied-index permutations for χ-CCM "
        "local-correlation setup. Returns an int64 matrix with one row per "
        "cyclic translation in lexicographic finite-torus order.");

    m.def(
        "aiccm2026dev_b_mayer_pair_tensor",
        &mayer_pair_tensor_from_blocks,
        py::arg("density_blocks"),
        py::arg("overlap_blocks"),
        py::arg("ao_atoms_unit"),
        py::arg("mesh"),
        py::arg("density_beta_blocks") = py::none(),
        "OpenMP block-circulant Mayer atom-pair tensor for χ-CCM. "
        "The input blocks are finite-torus real-space density/overlap blocks "
        "ordered like itertools.product(range(n0), range(n1), range(n2)); "
        "the output has shape (n_cells, n_atoms, n_atoms) and avoids both "
        "full AO and atom supercell matrices.");

    m.def(
        "aiccm2026dev_b_mayer_pair_matrix",
        &mayer_pair_matrix_from_blocks,
        py::arg("density_blocks"),
        py::arg("overlap_blocks"),
        py::arg("ao_atoms_unit"),
        py::arg("mesh"),
        py::arg("density_beta_blocks") = py::none(),
        "OpenMP block-circulant Mayer atom-pair contraction for χ-CCM. "
        "The input blocks are finite-torus real-space density/overlap blocks "
        "ordered like itertools.product(range(n0), range(n1), range(n2)); "
        "the output is the folded full atom-pair matrix without materialising "
        "full AO supercell density and overlap matrices.");
}
