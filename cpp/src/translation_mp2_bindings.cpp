#include "vibeqc/translation_mp2.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_translation_mp2(py::module_& module) {
    module.def("dlpno_translation_mp2_solve",
        [](const std::vector<int>& mesh, int bands,
           const std::vector<int>& pair_i, const std::vector<int>& pair_j,
           py::sequence integrals, py::sequence fock, py::sequence seeds,
           py::sequence coefficients, py::object occupied_fock, py::object overlap,
           const std::vector<double>& weights, int maximum_iterations,
           double energy_tolerance, double residual_tolerance, double damping,
           double symmetry_tolerance) {
            const auto matrix = [](py::handle object) {
                const auto array = py::array::ensure(object);
                if (!array || array.ndim() != 2
                    || (array.dtype().kind() != 'f' && array.dtype().kind() != 'i'
                        && array.dtype().kind() != 'u'))
                    throw std::invalid_argument("translation MP2 requires real numeric matrices");
                return array.cast<Eigen::MatrixXd>();
            };
            const auto matrices = [&](py::sequence objects) {
                std::vector<Eigen::MatrixXd> result;
                for (const auto object : objects) result.push_back(matrix(object));
                return result;
            };
            const auto k = matrices(integrals), f = matrices(fock), t = matrices(seeds), v = matrices(coefficients);
            const auto foo = matrix(occupied_fock), s = matrix(overlap);
            vibeqc::TranslationMP2Result result;
            {
                py::gil_scoped_release release;
                result = vibeqc::translation_mp2_solve(mesh, bands, pair_i, pair_j,
                    k, f, t, v, foo, s, weights, maximum_iterations,
                    energy_tolerance, residual_tolerance, damping, symmetry_tolerance);
            }
            py::list trace;
            for (const auto& step : result.trace)
                trace.append(py::make_tuple(step.iteration, step.energy,
                                             step.energy_change, step.maximum_residual));
            return py::make_tuple(result.amplitudes, result.residuals, result.converged, trace);
        },
        py::arg("mesh"), py::arg("n_bands"), py::arg("pair_i"), py::arg("pair_j"),
        py::arg("K"), py::arg("F_vv"), py::arg("T"), py::arg("V"),
        py::arg("F_oo"), py::arg("S_ao"), py::arg("energy_weights"),
        py::arg("maximum_iterations"), py::arg("energy_tolerance"),
        py::arg("residual_tolerance"), py::arg("damping"), py::arg("symmetry_tolerance"),
        "Experimental representative finite-torus MP2 Jacobi solve; raw snapshot convergence.");
}
