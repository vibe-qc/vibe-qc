// Capped reference-optimizer diagnostics; included by bindings.cpp.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_correlation_iao_optimizer.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_iao_optimizer(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationIAOOptimizerOptions;
    using Caps = PeriodicCorrelationIAOOptimizerCaps;
    using Memory = PeriodicCorrelationIAOOptimizerMemoryPlan;
    using Diagnostics = PeriodicCorrelationIAOOptimizerDiagnostics;
    using Progress = PeriodicCorrelationIAOOptimizerProgress;
    using Result = PeriodicCorrelationIAOOptimizerResult;
    using Status = PeriodicCorrelationIAOOptimizerStatus;
    using Event = PeriodicCorrelationIAOOptimizerEvent;
    using Z = std::complex<double>;
    py::enum_<Status>(m, "_PeriodicCorrelationIAOOptimizerStatus")
        .value("CONVERGED", Status::Converged).value("ITERATION_LIMIT", Status::IterationLimit)
        .value("LINE_SEARCH_FAILED", Status::LineSearchFailed).value("WORK_LIMIT", Status::WorkLimit);
    py::enum_<Event>(m, "_PeriodicCorrelationIAOOptimizerEvent")
        .value("INITIAL", Event::Initial).value("ACCEPTED", Event::Accepted)
        .value("REJECTED", Event::Rejected).value("FINISHED", Event::Finished);
    auto options = py::class_<Options>(m, "_PeriodicCorrelationIAOOptimizerOptions"); options.def(py::init<>());
#define IAO_OPT_OPTION(field) options.def_readwrite(#field, &Options::field)
    IAO_OPT_OPTION(maximum_iterations); IAO_OPT_OPTION(maximum_line_search_trials);
    IAO_OPT_OPTION(initial_step); IAO_OPT_OPTION(minimum_step);
    IAO_OPT_OPTION(backtracking_factor); IAO_OPT_OPTION(armijo_fraction);
    IAO_OPT_OPTION(riemannian_gradient_tolerance); IAO_OPT_OPTION(absolute_tolerance);
    IAO_OPT_OPTION(relative_tolerance); IAO_OPT_OPTION(jacobi_max_sweeps);
    IAO_OPT_OPTION(jacobi_relative_tolerance);
#undef IAO_OPT_OPTION
    py::class_<Caps>(m, "_PeriodicCorrelationIAOOptimizerCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    auto memory = py::class_<Memory>(m, "_PeriodicCorrelationIAOOptimizerMemoryPlan");
#define IAO_OPT_MEMORY(field) memory.def_readonly(#field, &Memory::field)
    IAO_OPT_MEMORY(n_points); IAO_OPT_MEMORY(n_basis); IAO_OPT_MEMORY(n_active); IAO_OPT_MEMORY(n_minimal);
    IAO_OPT_MEMORY(output_numerical_bytes); IAO_OPT_MEMORY(fixed_optimizer_bytes);
    IAO_OPT_MEMORY(exponential_workspace_bytes); IAO_OPT_MEMORY(pm_owned_peak_bytes);
    IAO_OPT_MEMORY(peak_owned_numerical_bytes); IAO_OPT_MEMORY(borrowed_seed_numerical_bytes);
    IAO_OPT_MEMORY(live_iao_numerical_bytes); IAO_OPT_MEMORY(borrowed_owner_pointer_bytes);
    IAO_OPT_MEMORY(required_node_memory_bytes); IAO_OPT_MEMORY(preflight_work_units);
    IAO_OPT_MEMORY(projection_work_units); IAO_OPT_MEMORY(trial_retraction_work_units);
    IAO_OPT_MEMORY(pm_work_units); IAO_OPT_MEMORY(initial_work_units);
    IAO_OPT_MEMORY(work_units_per_trial); IAO_OPT_MEMORY(maximum_work_units);
#undef IAO_OPT_MEMORY
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicCorrelationIAOOptimizerDiagnostics");
#define IAO_OPT_DIAGNOSTIC(field) diagnostics.def_readonly(#field, &Diagnostics::field)
    IAO_OPT_DIAGNOSTIC(charged_work_units); IAO_OPT_DIAGNOSTIC(objective_evaluations);
    IAO_OPT_DIAGNOSTIC(accepted_steps); IAO_OPT_DIAGNOSTIC(line_search_trials);
    IAO_OPT_DIAGNOSTIC(rejected_trials); IAO_OPT_DIAGNOSTIC(jacobi_sweeps);
    IAO_OPT_DIAGNOSTIC(initial_objective); IAO_OPT_DIAGNOSTIC(final_objective);
    IAO_OPT_DIAGNOSTIC(riemannian_gradient_norm); IAO_OPT_DIAGNOSTIC(last_accepted_step);
    IAO_OPT_DIAGNOSTIC(maximum_relative_unitarity_residual); IAO_OPT_DIAGNOSTIC(maximum_raw_unitarity_residual);
    IAO_OPT_DIAGNOSTIC(maximum_charge_frame_time_reversal_residual);
    IAO_OPT_DIAGNOSTIC(maximum_physical_time_reversal_residual); IAO_OPT_DIAGNOSTIC(maximum_trim_physical_imaginary_magnitude);
    IAO_OPT_DIAGNOSTIC(maximum_trim_relative_imaginary_correction);
    IAO_OPT_DIAGNOSTIC(maximum_exponential_eigen_relative_residual);
    IAO_OPT_DIAGNOSTIC(maximum_exponential_eigenvector_unitarity_residual);
    IAO_OPT_DIAGNOSTIC(maximum_tangent_slope_residual);
#undef IAO_OPT_DIAGNOSTIC
    auto progress = py::class_<Progress>(m, "_PeriodicCorrelationIAOOptimizerProgress");
#define IAO_OPT_PROGRESS(field) progress.def_readonly(#field, &Progress::field)
    IAO_OPT_PROGRESS(event); IAO_OPT_PROGRESS(status); IAO_OPT_PROGRESS(accepted_steps);
    IAO_OPT_PROGRESS(objective_evaluations); IAO_OPT_PROGRESS(line_search_trials);
    IAO_OPT_PROGRESS(charged_work_units); IAO_OPT_PROGRESS(objective);
    IAO_OPT_PROGRESS(riemannian_gradient_norm); IAO_OPT_PROGRESS(step); IAO_OPT_PROGRESS(trial_objective);
#undef IAO_OPT_PROGRESS
    py::class_<Result>(m, "_PeriodicCorrelationIAOOptimizerResult")
        .def_property_readonly("contract_version", &Result::contract_version)
        .def_property_readonly("status", &Result::status).def_property_readonly("converged", &Result::converged)
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("state", [](const Result& r) { (void)r.state(); return std::const_pointer_cast<PeriodicRestrictedMeanFieldState>(r.state_handle()); })
        .def_property_readonly("allocation_identity", &Result::allocation_identity)
        .def_property_readonly("seed_identity_sha256", &Result::seed_identity_sha256)
        .def_property_readonly("source_identity_sha256", &Result::source_identity_sha256)
        .def_property_readonly("gauge_payload_sha256", &Result::gauge_payload_sha256)
        .def_property_readonly("optimizer_identity_sha256", &Result::optimizer_identity_sha256)
        .def("active_band", &Result::active_band).def("gauge", &Result::gauge)
        .def("gauges_copy", [](const Result& r) {
            const auto k = r.memory().n_points, o = r.memory().n_active;
            if (k > 8 || o > 4) throw std::length_error("IAO optimizer diagnostic copy exceeds tiny dimensions");
            const auto* source = r.gauges_data();
            py::array_t<Z> out({static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(o), static_cast<py::ssize_t>(o)});
            std::memcpy(out.mutable_data(), source, k * o * o * sizeof(Z)); return out;
        });
    m.def("_plan_periodic_correlation_iao_optimizer", &plan_periodic_correlation_iao_optimizer,
        py::arg("reference"), py::arg("seed"), py::arg("n_minimal"), py::arg("options"));
    m.def("_optimize_periodic_correlation_iao_pm", [](
        const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationDiabaticSeed& seed,
        const py::list& points, PeriodicCorrelationIAOPMOptions pm_options, Options options,
        Caps caps, const py::object& callback) {
        if (!reference.state_handle()) throw std::invalid_argument("IAO optimizer requires a live reference");
        const auto& state = reference.state();
        const auto k = state.n_kpoints();
        const auto o = state.n_correlated_occupied();
        if (k > 8 || o > 4 || state.n_basis() > 12 || points.size() != k
            || options.maximum_iterations > 256 || options.maximum_line_search_trials > 16
            || options.jacobi_max_sweeps > 64 || caps.maximum_owned_numerical_bytes > (4U << 20)
            || caps.maximum_work_units > 10000000000ULL)
            throw std::length_error("IAO optimizer diagnostic exceeds tiny dimensions or iteration/work limits");
        if (!callback.is_none() && !PyCallable_Check(callback.ptr())) throw std::invalid_argument("IAO optimizer callback must be callable or None");
        // Pin each immutable owner independently of the caller's mutable list:
        // a progress callback may clear or replace that list. This tiny tuple
        // and copied scalar wrappers are bounded Python reference/control
        // metadata, separate from the exact numerical inventory; no numerical
        // payload is copied. The K-native-pointer view is charged by the core.
        const py::tuple pinned_points(points);
        std::vector<const PeriodicCorrelationBlochIAO*> owners; owners.reserve(k);
        for (const auto value : pinned_points) owners.push_back(&value.cast<const PeriodicCorrelationBlochIAO&>());
        const auto notify = [](const Progress& event, void* context) {
            (*static_cast<const py::object*>(context))(py::cast(Progress(event), py::return_value_policy::move));
        };
        // Hold GIL and copy scalar control wrappers by value: a Python progress
        // callback must not mutate the active numerical controls mid-iteration.
        return optimize_periodic_correlation_iao_pm(reference, seed, owners.data(), owners.size(),
            pm_options, options, caps, callback.is_none() ? nullptr : +notify,
            callback.is_none() ? nullptr : const_cast<py::object*>(&callback));
    }, py::arg("reference"), py::arg("seed"), py::arg("points"), py::arg("pm_options"),
       py::arg("options"), py::arg("caps"), py::arg("callback") = py::none());
}
