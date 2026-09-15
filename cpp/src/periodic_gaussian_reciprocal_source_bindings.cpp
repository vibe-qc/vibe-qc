// Included by bindings.cpp; tiny source diagnostics, not an SCF method.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "vibeqc/periodic_gaussian_reciprocal_source.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_reciprocal_source(py::module_& m) {
    using Caps = vibeqc::PeriodicGaussianReciprocalSourceCaps;
    using Inventory = vibeqc::PeriodicGaussianReciprocalSourceInventory;
    using Source = vibeqc::PeriodicGaussianReciprocalSource;
    py::class_<Caps>(m, "_PeriodicGaussianReciprocalSourceCaps")
        .def(py::init<>())
        .def_readwrite("maximum_fixed_storage_bytes", &Caps::maximum_fixed_storage_bytes)
        .def_readwrite("maximum_candidates_per_source", &Caps::maximum_candidates_per_source)
        .def_readwrite("maximum_candidate_evaluations", &Caps::maximum_candidate_evaluations)
        .def_readwrite("maximum_source_wire_bytes", &Caps::maximum_source_wire_bytes);
    py::class_<Inventory>(m, "_PeriodicGaussianReciprocalSourceInventory")
        .def_readonly("fixed_source_storage_bytes", &Inventory::fixed_source_storage_bytes)
        .def_readonly("retained_context_storage_bytes", &Inventory::retained_context_storage_bytes)
        .def_readonly("factory_workspace_storage_bytes", &Inventory::factory_workspace_storage_bytes)
        .def_readonly("inventoried_fixed_storage_bytes", &Inventory::inventoried_fixed_storage_bytes)
        .def_readonly("variable_owned_numeric_bytes", &Inventory::variable_owned_numeric_bytes)
        .def_readonly("candidate_evaluations_upper_bound", &Inventory::candidate_evaluations_upper_bound)
        .def_readonly("candidate_evaluations_performed", &Inventory::candidate_evaluations_performed)
        .def_readonly("source_wire_bytes", &Inventory::source_wire_bytes)
        .def_readonly("conjugate_source_wire_bytes", &Inventory::conjugate_source_wire_bytes);
    py::class_<Source>(m, "_PeriodicGaussianReciprocalSource")
        .def_property_readonly("contract_version", &Source::contract_version)
        .def_property_readonly("context", &Source::context_handle)
        .def_property_readonly("reciprocal_conjugate_closure_certified", &Source::reciprocal_conjugate_closure_certified)
        .def_property_readonly("ao_image_source_certified", &Source::ao_image_source_certified)
        .def_property_readonly("q_index", &Source::q_index)
        .def_property_readonly("conjugate_q_index", &Source::conjugate_q_index)
        .def_property_readonly("self_conjugate_transfer", &Source::self_conjugate_transfer)
        .def_property_readonly("centered_doubled_numerator", &Source::centered_doubled_numerator)
        .def_property_readonly("centered_reciprocal_wrap", &Source::centered_reciprocal_wrap)
        .def_property_readonly("q_fractional", [](const Source& s) { return Eigen::Vector3d(s.q_fractional()); })
        .def_property_readonly("q_cartesian", [](const Source& s) { return Eigen::Vector3d(s.q_cartesian()); })
        .def_property_readonly("reciprocal_lattice", [](const Source& s) { return Eigen::Matrix3d(s.reciprocal_lattice()); })
        .def_property_readonly("reciprocal_energy_cutoff", &Source::reciprocal_energy_cutoff)
        .def_property_readonly("maximum_reciprocal_radius", &Source::maximum_reciprocal_radius)
        .def_property_readonly("radial_boundary_tolerance", &Source::radial_boundary_tolerance)
        .def_property_readonly("cell_volume_bohr3", &Source::cell_volume_bohr3)
        .def_property_readonly("lower_bounds", &Source::lower_bounds)
        .def_property_readonly("upper_bounds", &Source::upper_bounds)
        .def_property_readonly("candidate_count", &Source::candidate_count)
        .def_property_readonly("conjugate_candidate_count", &Source::conjugate_candidate_count)
        .def_property_readonly("accepted_vector_count", &Source::accepted_vector_count)
        .def_property_readonly("zero_mode_excluded_count", &Source::zero_mode_excluded_count)
        .def_property_readonly("conjugacy_audited_vector_count", &Source::conjugacy_audited_vector_count)
        .def_property_readonly("inventory", [](const Source& s) { return Inventory(s.inventory()); })
        .def_property_readonly("source_identity_sha256", &Source::source_identity_sha256)
        .def_property_readonly("conjugate_source_identity_sha256", &Source::conjugate_source_identity_sha256)
        .def_property_readonly("source_context_identity_sha256", &Source::source_context_identity_sha256);
    m.def("_make_periodic_gaussian_reciprocal_source", [](
        std::shared_ptr<const vibeqc::PeriodicGaussianSourceContext> context,
        std::uint64_t q, const Caps& caps) {
        if (caps.maximum_candidates_per_source > 65536U
            || caps.maximum_candidate_evaluations > 400000U) {
            throw std::length_error("Gaussian reciprocal diagnostic source exceeds hard work caps");
        }
        py::gil_scoped_release release;
        return vibeqc::make_periodic_gaussian_reciprocal_source(std::move(context), q, caps);
    }, py::arg("context"), py::arg("q_index"), py::arg("caps"));
    m.def("_periodic_gaussian_reciprocal_source_wire_bytes",
          &vibeqc::periodic_gaussian_reciprocal_source_wire_bytes, py::arg("accepted_count"));
    m.def("_periodic_gaussian_reciprocal_source_records", [](const Source& source, std::uint64_t cap) {
        if (source.accepted_vector_count() > 512U || source.candidate_count() > 65536U || cap > 65536U) {
            throw std::length_error("Gaussian reciprocal diagnostic record copy exceeds tiny shape");
        }
        if (cap == 0U || source.candidate_count() > cap) {
            throw std::length_error("Gaussian reciprocal diagnostic record copy candidate cap");
        }
        const auto n = static_cast<py::ssize_t>(source.accepted_vector_count());
        py::array_t<std::int64_t> labels({n, py::ssize_t(3)});
        py::array_t<double> lanes({n, py::ssize_t(5)});
        struct Copy {
            std::int64_t* labels;
            double* lanes;
            std::uint64_t n, cursor = 0;
        } copy{labels.mutable_data(), lanes.mutable_data(), source.accepted_vector_count()};
        {
            py::gil_scoped_release release;
            vibeqc::visit_periodic_gaussian_reciprocal_source(source, cap,
                [](const std::array<std::int64_t, 3>& label, const std::array<double, 5>& values, void* p) {
                    auto& copy = *static_cast<Copy*>(p);
                    if (copy.cursor >= copy.n) throw std::logic_error("Gaussian reciprocal record copy overflow");
                    for (std::size_t d = 0; d != 3; ++d) copy.labels[3 * copy.cursor + d] = label[d];
                    for (std::size_t d = 0; d != 5; ++d) copy.lanes[5 * copy.cursor + d] = values[d];
                    ++copy.cursor;
                }, &copy);
        }
        return py::make_tuple(std::move(labels), std::move(lanes));
    }, py::arg("source"), py::arg("maximum_candidates"));
    // Tiny callback/cancellation diagnostic only. The Python callback's own
    // allocations/side effects are outside source resource accounting.
    m.def("_visit_periodic_gaussian_reciprocal_source", [](
        const Source& source, std::uint64_t cap, py::function callback) {
        if (source.accepted_vector_count() > 512U || source.candidate_count() > 65536U || cap > 65536U) {
            throw std::length_error("Gaussian reciprocal callback diagnostic exceeds tiny shape");
        }
        return vibeqc::visit_periodic_gaussian_reciprocal_source(source, cap,
            [](const std::array<std::int64_t, 3>& label, const std::array<double, 5>& lanes, void* p) {
                (*static_cast<py::function*>(p))(label, lanes);
            }, &callback);
    }, py::arg("source"), py::arg("maximum_candidates"), py::arg("callback"));
}
