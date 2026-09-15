// Included by bindings.cpp; tiny native pre-SCF metric diagnostics.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_gaussian_metric.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_metric(py::module_& m) {
    using Config = vibeqc::PeriodicGaussianMetricConfig;
    using Live = vibeqc::PeriodicGaussianMetricLiveInventory;
    using Caps = vibeqc::PeriodicGaussianMetricCaps;
    using Plan = vibeqc::PeriodicGaussianMetricPlan;
    using Phase = vibeqc::PeriodicGaussianMetricPhase;
    using Raw = vibeqc::PeriodicGaussianReciprocalMetric;
    using White = vibeqc::PeriodicGaussianMetricWhitener;
    py::enum_<Phase>(m, "_PeriodicGaussianMetricPhase")
        .value("RAW_METRIC", Phase::RawMetric).value("PRINCIPAL_WHITENING", Phase::PrincipalWhitening);
    py::class_<Config>(m, "_PeriodicGaussianMetricConfig")
        .def(py::init<>())
        .def_readwrite("reciprocal_block", &Config::reciprocal_block)
        .def_readwrite("whitener_column_block", &Config::whitener_column_block)
        .def_readwrite("basis_verification_caps", &Config::basis_verification_caps);
    py::class_<Live>(m, "_PeriodicGaussianMetricLiveInventory")
        .def(py::init<>())
        .def_readwrite("replicas_per_node", &Live::replicas_per_node)
        .def_readwrite("other_retained_bytes_per_replica", &Live::other_retained_bytes_per_replica)
        .def_readwrite("other_transient_bytes_per_replica", &Live::other_transient_bytes_per_replica)
        .def_readwrite("fixed_backend_margin_bytes_per_replica", &Live::fixed_backend_margin_bytes_per_replica)
        .def_readwrite("external_node_bytes", &Live::external_node_bytes);
    py::class_<Caps>(m, "_PeriodicGaussianMetricCaps")
        .def(py::init<>())
        .def_readwrite("maximum_owned_numeric_bytes", &Caps::maximum_owned_numeric_bytes)
        .def_readwrite("maximum_per_replica_inventoried_bytes", &Caps::maximum_per_replica_inventoried_bytes)
        .def_readwrite("maximum_node_inventoried_bytes", &Caps::maximum_node_inventoried_bytes)
        .def_readwrite("maximum_candidate_evaluations", &Caps::maximum_candidate_evaluations)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    auto plan = py::class_<Plan>(m, "_PeriodicGaussianMetricPlan");
#define VIBEQC_GMETRIC_PLAN_FIELD(Name) plan.def_readonly(#Name, &Plan::Name)
    VIBEQC_GMETRIC_PLAN_FIELD(phase);
    VIBEQC_GMETRIC_PLAN_FIELD(n_auxiliary);
    VIBEQC_GMETRIC_PLAN_FIELD(accepted_vector_count);
    VIBEQC_GMETRIC_PLAN_FIELD(candidate_count);
    VIBEQC_GMETRIC_PLAN_FIELD(reciprocal_panel_capacity);
    VIBEQC_GMETRIC_PLAN_FIELD(auxiliary_validation_passes);
    VIBEQC_GMETRIC_PLAN_FIELD(matrix_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(reciprocal_panel_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(double_fourier_panel_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(metric_owned_numeric_peak_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(factorization_owned_numeric_peak_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(owned_numeric_peak_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(returned_numeric_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(borrowed_basis_active_numeric_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(fixed_inventoried_object_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(per_replica_inventoried_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(node_inventoried_bytes);
    VIBEQC_GMETRIC_PLAN_FIELD(metric_product_count);
    VIBEQC_GMETRIC_PLAN_FIELD(fourier_radial_term_count);
    VIBEQC_GMETRIC_PLAN_FIELD(harmonic_term_count_upper_bound);
    VIBEQC_GMETRIC_PLAN_FIELD(jacobi_rotation_count_upper_bound);
    VIBEQC_GMETRIC_PLAN_FIELD(whitener_term_count_upper_bound);
    VIBEQC_GMETRIC_PLAN_FIELD(candidate_evaluations);
    VIBEQC_GMETRIC_PLAN_FIELD(work_units_upper_bound);
#undef VIBEQC_GMETRIC_PLAN_FIELD
    plan.def_property_readonly("config", [](const Plan& p) { return Config(p.config); })
        .def_property_readonly("live", [](const Plan& p) { return Live(p.live); })
        .def_property_readonly("caps", [](const Plan& p) { return Caps(p.caps); })
        .def_property_readonly("plan_identity_sha256", &Plan::plan_identity_sha256);
    const auto copy_matrix = [](const auto& result) {
        const auto a = result.plan().n_auxiliary;
        if (!result.context_handle() || a == 0 || a > 16 || result.matrix_row_major().size() != a * a) {
            throw std::invalid_argument("Gaussian metric matrix copy is consumed or outside tiny dimension");
        }
        py::array_t<std::complex<double>> array({static_cast<py::ssize_t>(a), static_cast<py::ssize_t>(a)});
        std::memcpy(array.mutable_data(), result.matrix_row_major().data(), a * a * 16U);
        return array;
    };
    py::class_<Raw>(m, "_PeriodicGaussianReciprocalMetric")
        .def_property_readonly("consumed", &Raw::consumed)
        .def_property_readonly("context", &Raw::context_handle)
        .def_property_readonly("plan", [](const Raw& r) { return Plan(r.plan()); })
        .def_property_readonly("q_index", &Raw::q_index)
        .def_property_readonly("conjugate_q_index", &Raw::conjugate_q_index)
        .def_property_readonly("self_conjugate_transfer", &Raw::self_conjugate_transfer)
        .def_property_readonly("completed_panel_count", &Raw::completed_panel_count)
        .def_property_readonly("maximum_self_conjugate_imaginary_residual", &Raw::maximum_self_conjugate_imaginary_residual)
        .def_property_readonly("source_identity_sha256", &Raw::source_identity_sha256)
        .def_property_readonly("conjugate_source_identity_sha256", &Raw::conjugate_source_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Raw::payload_identity_sha256)
        .def_property_readonly("matrix", [copy_matrix](const Raw& r) { return copy_matrix(r); });
    py::class_<White>(m, "_PeriodicGaussianMetricWhitener")
        .def_property_readonly("context", &White::context_handle)
        .def_property_readonly("plan", [](const White& r) { return Plan(r.plan()); })
        .def_property_readonly("q_index", &White::q_index)
        .def_property_readonly("conjugate_q_index", &White::conjugate_q_index)
        .def_property_readonly("self_conjugate_transfer", &White::self_conjugate_transfer)
        .def_property_readonly("diagnostics", &White::diagnostics)
        .def_property_readonly("source_identity_sha256", &White::source_identity_sha256)
        .def_property_readonly("conjugate_source_identity_sha256", &White::conjugate_source_identity_sha256)
        .def_property_readonly("input_payload_identity_sha256", &White::input_payload_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &White::payload_identity_sha256)
        .def_property_readonly("matrix", [copy_matrix](const White& r) { return copy_matrix(r); });
    m.def("_plan_periodic_gaussian_metric", &vibeqc::plan_periodic_gaussian_metric,
          py::arg("source"), py::arg("config"), py::arg("live"), py::arg("caps"), py::arg("phase"));
    m.def("_build_periodic_gaussian_reciprocal_metric", [](
        const vibeqc::PeriodicGaussianReciprocalSource& source,
        const vibeqc::BasisSet& ao, const vibeqc::BasisSet& auxiliary,
        const Config& config, const Live& live, const Caps& caps) {
        if (ao.nbasis() > 16 || auxiliary.nbasis() > 16 || source.accepted_vector_count() > 512
            || source.candidate_count() > 65536) throw std::length_error("Gaussian metric diagnostic exceeds tiny shape");
        py::gil_scoped_release release;
        return vibeqc::build_periodic_gaussian_reciprocal_metric(source, ao, auxiliary, config, live, caps);
    }, py::arg("source"), py::arg("ao_basis"), py::arg("auxiliary_basis"),
       py::arg("config"), py::arg("live"), py::arg("caps"));
    m.def("_factorize_periodic_gaussian_metric", [](Raw& raw, std::uint64_t columns, const Live& live, const Caps& caps) {
        if (raw.plan().n_auxiliary > 16) throw std::length_error("Gaussian whitening diagnostic exceeds tiny shape");
        py::gil_scoped_release release;
        return vibeqc::factorize_periodic_gaussian_metric(std::move(raw), columns, live, caps);
    }, py::arg("raw"), py::arg("whitener_column_block"), py::arg("live"), py::arg("caps"));
}
