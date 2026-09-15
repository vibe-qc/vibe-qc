// Tiny actual-source coupled local-triples diagnostics, no supplied tensors.
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gaussian_local_triples.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_local_triples(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicGaussianLocalTriplesOptions;
    using Live = PeriodicGaussianLocalTriplesLiveInventory;
    using Caps = PeriodicGaussianLocalTriplesCaps;
    using Plan = PeriodicGaussianLocalTriplesPlan;
    using Diagnostics = PeriodicGaussianLocalTriplesDiagnostics;
    using Stage = PeriodicGaussianLocalTriplesStage;
    using Progress = PeriodicGaussianLocalTriplesProgress;
    using Result = PeriodicGaussianLocalTriplesResult;
    py::class_<Options>(m, "_PeriodicGaussianLocalTriplesOptions").def(py::init<>())
        .def_readwrite("moments", &Options::moments).def_readwrite("solver", &Options::solver)
        .def_readwrite("maximum_brillouin_projection_norm", &Options::maximum_brillouin_projection_norm);
    auto live = py::class_<Live>(m, "_PeriodicGaussianLocalTriplesLiveInventory");
    live.def(py::init<>());
#define GLT_LIVE(f) live.def_readwrite(#f, &Live::f)
    GLT_LIVE(other_live_numerical_bytes_per_worker); GLT_LIVE(other_live_control_bytes_per_worker);
    GLT_LIVE(fixed_backend_margin_bytes_per_worker);
#undef GLT_LIVE
    auto caps = py::class_<Caps>(m, "_PeriodicGaussianLocalTriplesCaps");
    caps.def(py::init<>());
#define GLT_CAP(f) caps.def_readwrite(#f, &Caps::f)
    GLT_CAP(moments); GLT_CAP(solver); GLT_CAP(maximum_pair_count); GLT_CAP(maximum_triple_count);
    GLT_CAP(maximum_owned_numerical_bytes); GLT_CAP(maximum_control_storage_bytes_per_worker);
    GLT_CAP(maximum_per_worker_inventoried_bytes); GLT_CAP(maximum_node_inventoried_bytes);
    GLT_CAP(maximum_common_integral_calls); GLT_CAP(maximum_progress_callbacks); GLT_CAP(maximum_work_units);
#undef GLT_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicGaussianLocalTriplesPlan");
#define GLT_PLAN(f) plan.def_readonly(#f, &Plan::f)
    GLT_PLAN(occupied_count); GLT_PLAN(common_virtual_dimension); GLT_PLAN(pair_count); GLT_PLAN(triple_count);
    GLT_PLAN(maximum_retained_rank); GLT_PLAN(total_retained_rank); GLT_PLAN(total_amplitude_elements);
    GLT_PLAN(moments_required); GLT_PLAN(complete_borrowed_owner_numerical_bytes); GLT_PLAN(complete_borrowed_owner_control_bytes);
    GLT_PLAN(projected_zero_fov_bytes); GLT_PLAN(reader_table_bytes); GLT_PLAN(triple_table_bytes);
    GLT_PLAN(reader_borrowed_numerical_bytes); GLT_PLAN(solver_borrowed_numerical_bytes);
    GLT_PLAN(maximum_moment_transient_numerical_bytes); GLT_PLAN(reader_borrowed_rank_padding_bytes);
    GLT_PLAN(solver_borrowed_rank_padding_bytes); GLT_PLAN(rank_padding_reservation_bytes);
    GLT_PLAN(peak_owned_numerical_bytes); GLT_PLAN(control_storage_reservation_bytes);
    GLT_PLAN(replicas_per_node); GLT_PLAN(reference_base_node_bytes);
    GLT_PLAN(per_worker_inventoried_bytes); GLT_PLAN(required_node_memory_bytes);
    GLT_PLAN(common_integral_calls_upper_bound); GLT_PLAN(progress_callback_upper_bound);
    GLT_PLAN(driver_work_units); GLT_PLAN(work_units_upper_bound);
#undef GLT_PLAN
    plan.def_property_readonly("reader_upper", [](const Plan& p) { return p.reader_upper; })
        .def_property_readonly("moments_upper", [](const Plan& p) { return p.moments_upper; })
        .def_property_readonly("solver_upper", [](const Plan& p) { return p.solver_upper; });
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicGaussianLocalTriplesDiagnostics");
#define GLT_DIAG(f) diagnostics.def_readonly(#f, &Diagnostics::f)
    GLT_DIAG(completed_progress_callbacks); GLT_DIAG(completed_common_integral_calls);
    GLT_DIAG(brillouin_projection_frobenius_norm); GLT_DIAG(complete_common_finite_torus_basis);
    GLT_DIAG(all_retained_triples_full_common_rank);
#undef GLT_DIAG
    py::enum_<Stage>(m, "_PeriodicGaussianLocalTriplesStage")
        .value("BEGIN", Stage::Begin).value("SOLVER", Stage::Solver).value("FINISHED", Stage::Finished);
    py::class_<Progress>(m, "_PeriodicGaussianLocalTriplesProgress")
        .def_readonly("stage", &Progress::stage).def_readonly("callback_count", &Progress::callback_count)
        .def_readonly("common_integral_calls", &Progress::common_integral_calls)
        .def_property_readonly("solver", [](const Progress& p) { return p.solver; });
    py::class_<Result>(m, "_PeriodicGaussianLocalTriplesResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        // The numerical solver exposes only scalar records and copied
        // amplitudes; there is no consuming or mutating solver API.
        .def_property_readonly("solver", &Result::solver, py::return_value_policy::reference_internal)
        .def_property_readonly("converged", &Result::converged)
        .def_property_readonly("periodic_energy_per_cell", &Result::periodic_energy_per_cell)
        .def_property_readonly("triples_energy_per_cell", &Result::triples_energy_per_cell)
        .def_property_readonly("correlation_energy_per_cell", &Result::correlation_energy_per_cell)
        .def_property_readonly("total_energy_per_cell", &Result::total_energy_per_cell)
        .def_property_readonly("production_dlpno", &Result::production_dlpno)
        .def_property_readonly("infinite_source_accuracy_certified", &Result::infinite_source_accuracy_certified)
        .def_property_readonly("identity_sha256", &Result::identity_sha256)
        .def_property_readonly("ccsd_identity_sha256", &Result::ccsd_identity_sha256)
        .def_property_readonly("triple_spaces_identity_sha256", &Result::triple_spaces_identity_sha256)
        .def_property_readonly("consumed_moments_receipt_sha256", &Result::consumed_moments_receipt_sha256);
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianRealLocalProvider& provider, const PeriodicGaussianPairMP2Result& mp2,
        const PeriodicGaussianPairCCSDResult& ccsd, const PeriodicGaussianTripleSpaces& spaces,
        const Options& options, const Live& live, const Caps& caps) {
        if (ref.state().n_kpoints() > 8 || basis.memory().occupied_count > 4 || basis.memory().virtual_count > 4
            || options.solver.maximum_iterations > 128)
            throw std::length_error("Gaussian local-triples diagnostic exceeds tiny dimensions or iterations");
        auto p = plan_periodic_gaussian_local_triples(ref,basis,provider,mp2,ccsd,spaces,options,live,caps);
        if (p.peak_owned_numerical_bytes > (16U << 20) || p.required_node_memory_bytes > (128U << 20)
            || p.work_units_upper_bound > 1000000000000000ULL || p.progress_callback_upper_bound > 1024)
            throw std::length_error("Gaussian local-triples diagnostic exceeds tiny memory, work or callbacks");
        return p;
    };
    m.def("_plan_periodic_gaussian_local_triples", tiny, py::arg("reference"), py::arg("basis"),
        py::arg("provider"), py::arg("mp2"), py::arg("ccsd"), py::arg("spaces"),
        py::arg("options"), py::arg("live"), py::arg("caps"));
    m.def("_run_periodic_gaussian_local_triples", [tiny](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairMP2Result& mp2, const PeriodicGaussianPairCCSDResult& ccsd,
        const PeriodicGaussianTripleSpaces& spaces, Options options, Live live, Caps caps, py::object callback) {
        (void)tiny(ref,basis,provider,mp2,ccsd,spaces,options,live,caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian local-triples progress must be callable");
        const auto progress = [](const Progress& p, void* context) {
            (*static_cast<py::object*>(context))(py::cast(Progress(p)));
        };
        // Immutable native owners stay pinned. Copy every mutable control
        // before scalar progress callbacks; the tiny diagnostic keeps the GIL.
        return run_periodic_gaussian_local_triples(ref,basis,provider,mp2,ccsd,spaces,options,live,caps,
            callback.is_none() ? nullptr : +progress, callback.is_none() ? nullptr : &callback);
    }, py::arg("reference"), py::arg("basis"), py::arg("provider"), py::arg("mp2"), py::arg("ccsd"),
       py::arg("spaces"), py::arg("options"), py::arg("live"), py::arg("caps"), py::arg("progress") = py::none());
}
