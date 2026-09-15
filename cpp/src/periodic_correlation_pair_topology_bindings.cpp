// Internal diagnostic bindings for the native translation-pair topology.

#include <pybind11/pybind11.h>

#include <cstddef>
#include <memory>

#include "vibeqc/periodic_correlation_pair_topology.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pair_topology(py::module_& m) {
    using Classification = vibeqc::PeriodicCorrelationPairClassification;
    using Pair = vibeqc::PeriodicCorrelationTranslationPair;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Topology =
        vibeqc::PeriodicCorrelationTranslationPairTopology;
    using Resolution = vibeqc::PeriodicCorrelationPlacedPairResolution;
    py::class_<Resolution>(m, "_PeriodicCorrelationPlacedPairResolution")
        .def_readonly("row_index", &Resolution::row_index)
        .def_readonly("common_translation_cell", &Resolution::common_translation_cell)
        .def_readonly("transpose", &Resolution::transpose);

    m.attr(
        "_PERIODIC_CORRELATION_TRANSLATION_PAIR_TOPOLOGY_CONTRACT_VERSION") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationTranslationPairTopologyContractVersion);
    m.attr("_PERIODIC_CORRELATION_TRANSLATION_PAIR_ROW_BYTES") =
        py::int_(vibeqc::kPeriodicCorrelationTranslationPairRowBytes);

    py::enum_<Classification>(
        m, "_PeriodicCorrelationPairClassification")
        .value("UNCLASSIFIED", Classification::Unclassified);

    py::class_<Pair>(m, "_PeriodicCorrelationTranslationPair")
        .def_readonly("home_orbital", &Pair::home_orbital)
        .def_readonly("partner_orbital", &Pair::partner_orbital)
        .def_readonly(
            "translation_linear_index", &Pair::translation_linear_index)
        .def_readonly("placed_multiplicity", &Pair::placed_multiplicity)
        .def_property_readonly("classification", &Pair::classification);

    py::class_<Topology>(m, "_PeriodicCorrelationTranslationPairTopology")
        .def_property_readonly("contract_version", &Topology::contract_version)
        .def_property_readonly(
            "state",
            [](const Topology& topology) -> std::shared_ptr<State> {
                return std::const_pointer_cast<State>(topology.state_handle());
            })
        .def_property_readonly(
            "calculation_identity", &Topology::calculation_identity)
        .def_property_readonly(
            "allocation_identity", &Topology::allocation_identity)
        .def_property_readonly(
            "state_identity_sha256", &Topology::state_identity_sha256)
        .def_property_readonly(
            "topology_identity_sha256", &Topology::topology_identity_sha256)
        .def_property_readonly("mesh", &Topology::mesh)
        .def_property_readonly("is_shift", &Topology::is_shift)
        .def_property_readonly("n_cells", &Topology::n_cells)
        .def_property_readonly(
            "self_inverse_translation_count",
            &Topology::self_inverse_translation_count)
        .def_property_readonly(
            "n_home_occupied", &Topology::n_home_occupied)
        .def_property_readonly("row_count", &Topology::row_count)
        .def_property_readonly(
            "placed_pair_count", &Topology::placed_pair_count)
        .def(
            "row",
            [](const Topology& topology, std::size_t index) {
                return topology.row(index);
            },
            py::arg("index"))
        .def("classification", &Topology::classification, py::arg("index"));

    m.def(
        "_make_periodic_correlation_translation_pair_topology",
        &vibeqc::make_periodic_correlation_translation_pair_topology,
        py::arg("reference"));
    m.def("_resolve_periodic_correlation_placed_pair",
        &vibeqc::resolve_periodic_correlation_placed_pair, py::arg("topology"),
        py::arg("first_orbital"), py::arg("first_cell"),
        py::arg("second_orbital"), py::arg("second_cell"));
    m.def("_periodic_correlation_translation_pair_energy_weight",
        &vibeqc::periodic_correlation_translation_pair_energy_weight,
        py::arg("topology"), py::arg("row_index"));
}
