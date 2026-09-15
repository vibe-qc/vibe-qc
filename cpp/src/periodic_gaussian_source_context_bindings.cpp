// Included by bindings.cpp. Authenticated source inputs, not an SCF route.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

#include "vibeqc/periodic_gaussian_source_context.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_source_context(py::module_& m) {
    using Options = vibeqc::PeriodicGaussianSourceOptions;
    using Caps = vibeqc::PeriodicGaussianSourceCaps;
    using BasisInventory = vibeqc::PeriodicGaussianBasisInventory;
    using Inventory = vibeqc::PeriodicGaussianSourceInventory;
    using K = vibeqc::PeriodicGaussianKRecord;
    using Q = vibeqc::PeriodicGaussianTransferRecord;
    using Context = vibeqc::PeriodicGaussianSourceContext;
    m.attr("_PERIODIC_GAUSSIAN_SOURCE_CONTEXT_STORAGE_BYTES") = py::int_(sizeof(Context));
    m.attr("_PERIODIC_GAUSSIAN_SOURCE_GEOMETRY_POLICY") = py::str(vibeqc::kPeriodicGaussianSourceGeometryPolicy);
    m.attr("_PERIODIC_GAUSSIAN_SOURCE_HAMILTONIAN_POLICY") = py::str(vibeqc::kPeriodicGaussianSourceHamiltonianPolicy);
    m.attr("_PERIODIC_GAUSSIAN_SOURCE_RECIPROCAL_POLICY") = py::str(vibeqc::kPeriodicGaussianSourceReciprocalPolicy);
    py::class_<Options>(m, "_PeriodicGaussianSourceOptions")
        .def(py::init<>())
        .def_readwrite("reciprocal_energy_cutoff", &Options::reciprocal_energy_cutoff)
        .def_readwrite("ao_pair_image_cutoff_bohr", &Options::ao_pair_image_cutoff_bohr)
        .def_readwrite("metric_absolute_eigenvalue_threshold", &Options::metric_absolute_eigenvalue_threshold)
        .def_readwrite("metric_negative_tolerance", &Options::metric_negative_tolerance);
    py::class_<Caps>(m, "_PeriodicGaussianSourceCaps")
        .def(py::init<>())
        .def_readwrite("maximum_context_storage_bytes", &Caps::maximum_context_storage_bytes)
        .def_readwrite("maximum_kpoint_count", &Caps::maximum_kpoint_count)
        .def_readwrite("maximum_shell_count", &Caps::maximum_shell_count)
        .def_readwrite("maximum_contraction_count", &Caps::maximum_contraction_count)
        .def_readwrite("maximum_primitive_numeric_lanes", &Caps::maximum_primitive_numeric_lanes)
        .def_readwrite("maximum_basis_content_wire_bytes", &Caps::maximum_basis_content_wire_bytes)
        .def_readwrite("maximum_borrowed_active_numeric_bytes", &Caps::maximum_borrowed_active_numeric_bytes)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    py::class_<BasisInventory>(m, "_PeriodicGaussianBasisInventory")
        .def_readonly("function_count", &BasisInventory::function_count)
        .def_readonly("shell_count", &BasisInventory::shell_count)
        .def_readonly("contraction_count", &BasisInventory::contraction_count)
        .def_readonly("exponent_count", &BasisInventory::exponent_count)
        .def_readonly("coefficient_count", &BasisInventory::coefficient_count)
        .def_readonly("borrowed_active_numeric_bytes", &BasisInventory::borrowed_active_numeric_bytes)
        .def_readonly("content_wire_bytes", &BasisInventory::content_wire_bytes);
    py::class_<Inventory>(m, "_PeriodicGaussianSourceInventory")
        .def_readonly("fixed_context_storage_bytes", &Inventory::fixed_context_storage_bytes)
        .def_readonly("variable_owned_numeric_bytes", &Inventory::variable_owned_numeric_bytes)
        .def_readonly("ao", &Inventory::ao)
        .def_readonly("auxiliary", &Inventory::auxiliary)
        .def_readonly("combined_borrowed_active_numeric_bytes", &Inventory::combined_borrowed_active_numeric_bytes)
        .def_readonly("work_units_upper_bound", &Inventory::work_units_upper_bound);
    py::class_<K>(m, "_PeriodicGaussianKRecord")
        .def_readonly("index", &K::index)
        .def_readonly("modular_doubled_address", &K::modular_doubled_address)
        .def_readonly("fractional", &K::fractional)
        .def_readonly("cartesian", &K::cartesian);
    py::class_<Q>(m, "_PeriodicGaussianTransferRecord")
        .def_readonly("index", &Q::index)
        .def_readonly("modular_doubled_address", &Q::modular_doubled_address)
        .def_readonly("centered_doubled_numerator", &Q::centered_doubled_numerator)
        .def_readonly("centered_reciprocal_wrap", &Q::centered_reciprocal_wrap)
        .def_readonly("fractional", &Q::fractional)
        .def_readonly("cartesian", &Q::cartesian)
        .def_readonly("gamma", &Q::gamma)
        .def_readonly("self_conjugate", &Q::self_conjugate);
    py::class_<Context, std::shared_ptr<Context>>(m, "_PeriodicGaussianSourceContext")
        .def_property_readonly("contract_version", &Context::contract_version)
        .def_property_readonly("density_independent", &Context::density_independent)
        .def_property_readonly("enumerated_source_certified", &Context::enumerated_source_certified)
        .def_property_readonly("ao_image_source_certified", &Context::ao_image_source_certified)
        .def_property_readonly("nuclear_hcore_certified", &Context::nuclear_hcore_certified)
        .def_property_readonly("direct_lattice", [](const Context& c) { return Eigen::Matrix3d(c.direct_lattice()); })
        .def_property_readonly("reciprocal_lattice", [](const Context& c) { return Eigen::Matrix3d(c.reciprocal_lattice()); })
        .def_property_readonly("mesh", [](const Context& c) { return c.mesh().mesh(); })
        .def_property_readonly("is_shift", [](const Context& c) { return c.mesh().is_shift(); })
        .def_property_readonly("n_kpoints", [](const Context& c) { return c.mesh().size(); })
        .def_property_readonly("options", [](const Context& c) { return Options(c.options()); })
        .def_property_readonly("inventory", [](const Context& c) { return Inventory(c.inventory()); })
        .def_property_readonly("ao_basis_identity_sha256", &Context::ao_basis_identity_sha256)
        .def_property_readonly("auxiliary_basis_identity_sha256", &Context::auxiliary_basis_identity_sha256)
        .def_property_readonly("reciprocal_producer_identity_sha256", &Context::reciprocal_producer_identity_sha256)
        .def_property_readonly("factorization_backend_identity_sha256", &Context::factorization_backend_identity_sha256)
        .def_property_readonly("source_context_identity_sha256", &Context::source_context_identity_sha256)
        .def("k_record", &Context::k_record, py::arg("index"))
        .def("transfer_record", &Context::transfer_record, py::arg("index"))
        .def("ket_index", &Context::ket_index, py::arg("bra_index"), py::arg("q_index"))
        .def("verify_bases", &Context::verify_bases, py::arg("ao_basis"),
             py::arg("auxiliary_basis"), py::arg("caps"), py::call_guard<py::gil_scoped_release>());
    m.def("_make_periodic_gaussian_source_context", &vibeqc::make_periodic_gaussian_source_context,
          py::arg("system"), py::arg("ao_basis"), py::arg("auxiliary_basis"),
          py::arg("mesh"), py::arg("options"), py::arg("caps"),
          py::call_guard<py::gil_scoped_release>());
}
