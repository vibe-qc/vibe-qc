// Included by bindings.cpp; tiny diagnostics for native finite-source tiles.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_gaussian_three_center.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_three_center(py::module_& m) {
    using Selection = vibeqc::PeriodicGaussianThreeCenterSelection;
    using Descriptor = vibeqc::PeriodicGaussianThreeCenterDescriptor;
    using Config = vibeqc::PeriodicGaussianThreeCenterConfig;
    using Caps = vibeqc::PeriodicGaussianThreeCenterCaps;
    using Plan = vibeqc::PeriodicGaussianThreeCenterPlan;
    using Tile = vibeqc::PeriodicGaussianThreeCenterTile;
    using Source = vibeqc::PeriodicGaussianReciprocalSource;
    using White = vibeqc::PeriodicGaussianMetricWhitener;
    using Live = vibeqc::PeriodicGaussianMetricLiveInventory;
    py::class_<Selection>(m, "_PeriodicGaussianThreeCenterSelection")
        .def(py::init<>())
        .def_readwrite("k_bra_index", &Selection::k_bra_index)
        .def_readwrite("ao_pair_begin", &Selection::ao_pair_begin)
        .def_readwrite("ao_pair_count", &Selection::ao_pair_count)
        .def_readwrite("auxiliary_begin", &Selection::auxiliary_begin)
        .def_readwrite("auxiliary_count", &Selection::auxiliary_count);
    py::class_<Descriptor>(m, "_PeriodicGaussianThreeCenterDescriptor")
        .def_readonly("q_index", &Descriptor::q_index)
        .def_readonly("conjugate_q_index", &Descriptor::conjugate_q_index)
        .def_readonly("k_bra_index", &Descriptor::k_bra_index)
        .def_readonly("k_ket_index", &Descriptor::k_ket_index)
        .def_readonly("k_ket_reciprocal_wrap", &Descriptor::k_ket_reciprocal_wrap)
        .def_readonly("ao_pair_begin", &Descriptor::ao_pair_begin)
        .def_readonly("ao_pair_count", &Descriptor::ao_pair_count)
        .def_readonly("auxiliary_begin", &Descriptor::auxiliary_begin)
        .def_readonly("auxiliary_count", &Descriptor::auxiliary_count)
        .def_readonly("element_count", &Descriptor::element_count);
    py::class_<Config>(m, "_PeriodicGaussianThreeCenterConfig")
        .def(py::init<>())
        .def_readwrite("reciprocal_block", &Config::reciprocal_block)
        .def_readwrite("basis_verification_caps", &Config::basis_verification_caps);
    py::class_<Caps>(m, "_PeriodicGaussianThreeCenterCaps")
        .def(py::init<>())
        .def_readwrite("maximum_image_candidates", &Caps::maximum_image_candidates)
        .def_readwrite("resources", &Caps::resources);
    py::class_<Plan>(m, "_PeriodicGaussianThreeCenterPlan")
        .def_property_readonly("descriptor", [](const Plan& p) { return Descriptor(p.descriptor); })
        .def_readonly("n_basis", &Plan::n_basis)
        .def_readonly("n_auxiliary", &Plan::n_auxiliary)
        .def_readonly("accepted_vector_count", &Plan::accepted_vector_count)
        .def_readonly("reciprocal_candidate_count", &Plan::reciprocal_candidate_count)
        .def_readonly("reciprocal_capacity", &Plan::reciprocal_capacity)
        .def_readonly("reciprocal_panel_count", &Plan::reciprocal_panel_count)
        .def_readonly("resident_whitener_bytes", &Plan::resident_whitener_bytes)
        .def_readonly("raw_panel_bytes", &Plan::raw_panel_bytes)
        .def_readonly("compensation_bytes", &Plan::compensation_bytes)
        .def_readonly("reciprocal_panel_bytes", &Plan::reciprocal_panel_bytes)
        .def_readonly("double_auxiliary_fourier_bytes", &Plan::double_auxiliary_fourier_bytes)
        .def_readonly("ao_fourier_bytes", &Plan::ao_fourier_bytes)
        .def_readonly("fixed_fourier_numeric_workspace_bytes", &Plan::fixed_fourier_numeric_workspace_bytes)
        .def_readonly("output_bytes", &Plan::output_bytes)
        .def_readonly("assembly_owned_numeric_peak_bytes", &Plan::assembly_owned_numeric_peak_bytes)
        .def_readonly("whitening_owned_numeric_peak_bytes", &Plan::whitening_owned_numeric_peak_bytes)
        .def_readonly("owned_numeric_peak_bytes", &Plan::owned_numeric_peak_bytes)
        .def_readonly("borrowed_basis_active_numeric_bytes", &Plan::borrowed_basis_active_numeric_bytes)
        .def_readonly("fixed_inventoried_object_bytes", &Plan::fixed_inventoried_object_bytes)
        .def_readonly("per_replica_inventoried_bytes", &Plan::per_replica_inventoried_bytes)
        .def_readonly("node_inventoried_bytes", &Plan::node_inventoried_bytes)
        .def_readonly("image_candidate_count", &Plan::image_candidate_count)
        .def_readonly("retained_pair_image_count", &Plan::retained_pair_image_count)
        .def_readonly("reciprocal_candidate_evaluations", &Plan::reciprocal_candidate_evaluations)
        .def_readonly("image_candidate_evaluations", &Plan::image_candidate_evaluations)
        .def_readonly("primitive_pair_evaluations_upper_bound", &Plan::primitive_pair_evaluations_upper_bound)
        .def_readonly("raw_contraction_term_count", &Plan::raw_contraction_term_count)
        .def_readonly("whitening_term_count", &Plan::whitening_term_count)
        .def_readonly("preflight_work_units_upper_bound", &Plan::preflight_work_units_upper_bound)
        .def_readonly("work_units_upper_bound", &Plan::work_units_upper_bound)
        .def_property_readonly("config", [](const Plan& p) { return Config(p.config); })
        .def_property_readonly("live", [](const Plan& p) { return Live(p.live); })
        .def_property_readonly("caps", [](const Plan& p) { return Caps(p.caps); })
        .def_property_readonly("plan_identity_sha256", &Plan::plan_identity_sha256);
    py::class_<Tile>(m, "_PeriodicGaussianThreeCenterTile")
        .def_property_readonly("context", &Tile::context_handle)
        .def_property_readonly("plan", [](const Tile& t) { return Plan(t.plan()); })
        .def_property_readonly("descriptor", [](const Tile& t) { return Descriptor(t.descriptor()); })
        .def_property_readonly("finite_image_reference", &Tile::finite_image_reference)
        .def_property_readonly("ao_image_source_certified", &Tile::ao_image_source_certified)
        .def_property_readonly("source_identity_sha256", &Tile::source_identity_sha256)
        .def_property_readonly("conjugate_source_identity_sha256", &Tile::conjugate_source_identity_sha256)
        .def_property_readonly("whitener_payload_identity_sha256", &Tile::whitener_payload_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Tile::payload_identity_sha256)
        .def_property_readonly("matrix", [](const Tile& t) {
            const auto& d = t.descriptor();
            if (!t.context_handle() || d.auxiliary_count > 16 || d.ao_pair_count > 256
                || t.matrix_row_major().size() != d.element_count) {
                throw std::invalid_argument("Gaussian tile diagnostic copy is outside tiny shape or moved");
            }
            py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(d.auxiliary_count),
                                                 static_cast<py::ssize_t>(d.ao_pair_count)});
            std::memcpy(a.mutable_data(), t.matrix_row_major().data(), d.element_count * 16U);
            return a;
        });
    const auto require_tiny = [](const Source& source, const vibeqc::BasisSet& ao,
                                 const vibeqc::BasisSet& aux, const Selection& s, const Caps& caps) {
        if (ao.nbasis() > 16 || aux.nbasis() > 16 || s.ao_pair_count > 256
            || source.accepted_vector_count() > 512 || source.candidate_count() > 65536
            || caps.maximum_image_candidates > 65536) {
            throw std::length_error("Gaussian three-center diagnostic exceeds tiny shape/work caps");
        }
    };
    m.def("_plan_periodic_gaussian_three_center_tile", [require_tiny](
        const Source& source, const White& white, const vibeqc::BasisSet& ao,
        const vibeqc::BasisSet& aux, Selection selection, Config config,
        Live live, Caps caps) {
        require_tiny(source, ao, aux, selection, caps);
        py::gil_scoped_release release;
        return vibeqc::plan_periodic_gaussian_three_center_tile(source, white, ao, aux, selection, config, live, caps);
    }, py::arg("source"), py::arg("whitener"), py::arg("ao_basis"), py::arg("auxiliary_basis"),
       py::arg("selection"), py::arg("config"), py::arg("live"), py::arg("caps"));
    m.def("_build_periodic_gaussian_three_center_tile", [require_tiny](
        const Source& source, const White& white, const vibeqc::BasisSet& ao,
        const vibeqc::BasisSet& aux, Selection selection, Config config,
        Live live, Caps caps) {
        require_tiny(source, ao, aux, selection, caps);
        py::gil_scoped_release release;
        return vibeqc::build_periodic_gaussian_three_center_tile(source, white, ao, aux, selection, config, live, caps);
    }, py::arg("source"), py::arg("whitener"), py::arg("ao_basis"), py::arg("auxiliary_basis"),
       py::arg("selection"), py::arg("config"), py::arg("live"), py::arg("caps"));
    m.attr("_PERIODIC_GAUSSIAN_THREE_CENTER_LATTICE_POLICY") = vibeqc::kPeriodicGaussianThreeCenterLatticePolicy;
    m.attr("_PERIODIC_GAUSSIAN_THREE_CENTER_IMAGE_POLICY") = vibeqc::kPeriodicGaussianThreeCenterImagePolicy;
}
