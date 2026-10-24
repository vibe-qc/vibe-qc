#include "vibeqc/translation_mp2.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_translation_mp2(py::module_& module) {
    py::class_<vibeqc::TranslationMP2Transport>(module, "_TranslationMP2Transport")
        .def_property_readonly("n_links", [](const vibeqc::TranslationMP2Transport& value) { return value.links.size(); })
        .def_property_readonly("n_couplings", [](const vibeqc::TranslationMP2Transport& value) { return value.couplings.size(); })
        .def_property_readonly("n_representatives", [](const vibeqc::TranslationMP2Transport& value) { return value.virtual_fock.size(); })
        .def_readonly("compact", &vibeqc::TranslationMP2Transport::compact)
        .def_readonly("recipe_only", &vibeqc::TranslationMP2Transport::recipe_only)
        .def_readonly("cyclic_metric", &vibeqc::TranslationMP2Transport::cyclic_metric)
        .def_readonly("stable_links", &vibeqc::TranslationMP2Transport::stable_links)
        .def_readonly("retained_bytes", &vibeqc::TranslationMP2Transport::retained_bytes)
        .def_readonly("workspace_bytes", &vibeqc::TranslationMP2Transport::workspace_bytes)
        .def_readonly("required_bytes", &vibeqc::TranslationMP2Transport::required_bytes);
    const auto real_matrix = [](py::handle object) {
        const auto array = py::array::ensure(object);
        if (!array || array.ndim() != 2
            || (array.dtype().kind() != 'f' && array.dtype().kind() != 'i' && array.dtype().kind() != 'u'))
            throw std::invalid_argument("translation MP2 requires real numeric matrices");
        return array.cast<Eigen::MatrixXd>();
    };
    const auto real_matrices = [real_matrix](py::sequence objects) {
        std::vector<Eigen::MatrixXd> result;
        for (const auto object : objects) result.push_back(real_matrix(object));
        return result;
    };
    module.def("dlpno_prepare_translation_transport",
        [real_matrix, real_matrices](const std::vector<int>& mesh, int bands,
           const std::vector<int>& pair_i, const std::vector<int>& pair_j,
           py::sequence fock, py::sequence coefficients, py::object occupied_fock,
           py::object overlap, const std::vector<double>& weights, double tolerance,
           std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
           py::sequence local, py::sequence projected, py::object all_occupied, bool occupied_home) {
            const auto f = real_matrices(fock), v = real_matrices(coefficients);
            const auto foo = real_matrix(occupied_fock), s = real_matrix(overlap);
            const auto a = real_matrices(local), b = real_matrices(projected);
            const auto c = all_occupied.is_none() ? Eigen::MatrixXd() : real_matrix(all_occupied);
            py::gil_scoped_release release;
            return vibeqc::prepare_translation_mp2_transport(mesh, bands, pair_i, pair_j,
                f, v, foo, s, weights, tolerance, maximum_bytes, domains, a, b, c, occupied_home);
        },
        py::arg("mesh"), py::arg("n_bands"), py::arg("pair_i"), py::arg("pair_j"),
        py::arg("F_vv"), py::arg("V"), py::arg("F_oo"), py::arg("S_ao"),
        py::arg("energy_weights"), py::arg("symmetry_tolerance"), py::arg("maximum_bytes"),
        py::arg("domains") = std::vector<std::vector<int>>{},
        py::arg("local_coefficients") = py::tuple(), py::arg("occupied_coefficients") = py::tuple(),
        py::arg("all_occupied") = py::none(), py::arg("occupied_home") = false,
        "Prepare unique translated pair-space overlaps under an explicit transport byte budget.");
    module.def("dlpno_prepare_translation_transport_blocks",
        [real_matrix, real_matrices](const std::vector<int>& mesh, int bands,
           const std::vector<int>& pair_i, const std::vector<int>& pair_j,
           py::sequence fock, py::sequence coefficients, py::object occupied_fock,
           py::sequence overlap_blocks, const std::vector<double>& weights, double tolerance,
           std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
           py::sequence local, py::sequence projected, py::object all_occupied, bool occupied_home) {
            const auto f = real_matrices(fock), v = real_matrices(coefficients);
            const auto foo = real_matrix(occupied_fock);
            const auto s = real_matrices(overlap_blocks);
            const auto a = real_matrices(local), b = real_matrices(projected);
            const auto c = all_occupied.is_none() ? Eigen::MatrixXd() : real_matrix(all_occupied);
            py::gil_scoped_release release;
            return vibeqc::prepare_translation_mp2_transport_blocks(mesh, bands, pair_i, pair_j,
                f, v, foo, s, weights, tolerance, maximum_bytes, domains, a, b, c, occupied_home);
        },
        py::arg("mesh"), py::arg("n_bands"), py::arg("pair_i"), py::arg("pair_j"),
        py::arg("F_vv"), py::arg("V"), py::arg("F_oo"), py::arg("S_home_blocks"),
        py::arg("energy_weights"), py::arg("symmetry_tolerance"), py::arg("maximum_bytes"),
        py::arg("domains") = std::vector<std::vector<int>>{},
        py::arg("local_coefficients") = py::tuple(), py::arg("occupied_coefficients") = py::tuple(),
        py::arg("all_occupied") = py::none(), py::arg("occupied_home") = false,
        "Prepare translated pair overlaps from cyclic metric blocks without dense AO expansion.");
    module.def("dlpno_translation_mp2_solve_prepared",
        [real_matrices](const vibeqc::TranslationMP2Transport& transport,
           py::sequence integrals, py::sequence seeds, int maximum_iterations,
           double energy_tolerance, double residual_tolerance, double damping) {
            const auto k = real_matrices(integrals), t = real_matrices(seeds);
            vibeqc::TranslationMP2Result result;
            {
                py::gil_scoped_release release;
                result = vibeqc::translation_mp2_solve_prepared(transport, k, t,
                    maximum_iterations, energy_tolerance, residual_tolerance, damping);
            }
            py::list trace;
            for (const auto& step : result.trace)
                trace.append(py::make_tuple(step.iteration, step.energy, step.energy_change, step.maximum_residual));
            return py::make_tuple(result.amplitudes, result.residuals, result.converged, trace);
        },
        py::arg("transport"), py::arg("K"), py::arg("T"), py::arg("maximum_iterations"),
        py::arg("energy_tolerance"), py::arg("residual_tolerance"), py::arg("damping"),
        "Solve using prepared pair-space links, with no AO arrays in the iterative kernel.");
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
