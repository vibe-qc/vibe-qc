// Tiny internal diagnostics for the caller-owned scalar Jacobi eigensolver.
// Included by bindings.cpp; not a standalone translation unit.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cfenv>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "vibeqc/hermitian_jacobi.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_hermitian_jacobi(py::module_& m) {
    using Complex = std::complex<double>;
    using Status = vibeqc::HermitianJacobiStatus;
    py::enum_<Status>(m, "_HermitianJacobiStatus")
        .value("SUCCESS", Status::Success)
        .value("INVALID_INPUT", Status::InvalidInput)
        .value("NON_FINITE_INPUT", Status::NonFiniteInput)
        .value("NOT_HERMITIAN", Status::NotHermitian)
        .value("NO_CONVERGENCE", Status::NoConvergence)
        .value("NUMERICAL_FAILURE", Status::NumericalFailure);
    m.attr("_HERMITIAN_JACOBI_DIAGNOSTIC_MAXIMUM_DIMENSION") = py::int_(32);
    m.attr("_HERMITIAN_JACOBI_DIAGNOSTIC_MAXIMUM_SWEEPS") = py::int_(1024);
    m.def("_hermitian_jacobi_non_nearest_probe", []() {
        // Keep floating-environment testing native and tightly scoped: no
        // Python callback or array allocation runs under the altered mode.
        Complex matrix(1.0, 0.0);
        Complex vectors(0.0, 0.0);
        double value = 0.0;
        vibeqc::HermitianJacobiOptions options;
        options.max_sweeps = 1;
        options.relative_offdiagonal_tolerance = 1.0e-12;
        const int original = std::fegetround();
        if (original == -1 || std::fesetround(FE_DOWNWARD) != 0) {
            throw std::runtime_error("native rounding-mode probe is unavailable");
        }
        const auto result = vibeqc::hermitian_jacobi_in_place(
            &matrix, 1, &vectors, 1, &value, 1, 1, options);
        const int restore = std::fesetround(original);
        if (restore != 0) {
            throw std::runtime_error("native rounding-mode probe could not restore mode");
        }
        return result.status;
    });
    m.def(
        "_hermitian_jacobi_diagnostic",
        [](const py::array& input, std::uint64_t max_sweeps,
           double relative_offdiagonal_tolerance) {
            // All shape/type/cap checks precede any matrix copy. No implicit
            // array conversion, non-contiguous staging, or large diagnostic.
            if (input.ndim() != 2 || input.shape(0) != input.shape(1)
                || input.shape(0) < 1 || input.shape(0) > 32) {
                throw std::invalid_argument(
                    "Jacobi diagnostic requires a square matrix of dimension 1..32");
            }
            if (!input.dtype().is(py::dtype::of<Complex>())
                || (input.flags() & py::array::c_style) == 0) {
                throw std::invalid_argument(
                    "Jacobi diagnostic requires C-contiguous native complex128 input");
            }
            if (max_sweeps > 1024U) {
                throw std::invalid_argument(
                    "Jacobi diagnostic allows at most 1024 sweeps");
            }
            const auto n = input.shape(0);
            const auto count = static_cast<std::size_t>(n * n);
            py::array_t<Complex> work({n, n});
            py::array_t<Complex> vectors({n, n});
            py::array_t<double> values(n);
            std::memcpy(work.mutable_data(), input.data(), count * sizeof(Complex));
            // Invalid input returns a status without filling outputs. Keep
            // those diagnostic outputs initialized, never expose heap bytes.
            std::fill_n(vectors.mutable_data(), count, Complex(0.0, 0.0));
            std::fill_n(values.mutable_data(), static_cast<std::size_t>(n), 0.0);
            vibeqc::HermitianJacobiOptions options;
            options.max_sweeps = max_sweeps;
            options.relative_offdiagonal_tolerance = relative_offdiagonal_tolerance;
            vibeqc::HermitianJacobiResult solve;
            {
                py::gil_scoped_release release;
                solve = vibeqc::hermitian_jacobi_in_place(
                    work.mutable_data(), count, vectors.mutable_data(), count,
                    values.mutable_data(), static_cast<std::size_t>(n),
                    static_cast<std::size_t>(n), options);
            }
            py::dict result;
            result["status"] = solve.status;
            result["sweeps"] = solve.sweeps;
            result["rotations"] = solve.rotations;
            result["input_scale_exponent"] = solve.input_scale_exponent;
            result["input_scale"] = solve.input_scale;
            result["scaled_input_frobenius_norm"] = solve.scaled_input_frobenius_norm;
            result["scaled_offdiagonal_frobenius_norm"] =
                solve.scaled_offdiagonal_frobenius_norm;
            result["orthogonality_frobenius_error"] =
                solve.orthogonality_frobenius_error;
            result["scaling_underflow_components"] = solve.scaling_underflow_components;
            result["matrix_scaled"] = std::move(work);
            result["eigenvectors"] = std::move(vectors);
            result["eigenvalues"] = std::move(values);
            return result;
        },
        py::arg("matrix").noconvert(), py::arg("max_sweeps"),
        py::arg("relative_offdiagonal_tolerance"),
        "Bounded internal eigensolver diagnostic, dimension <= 32. "
        "Copies a complex128 input; numerical controls are explicit.");
}
