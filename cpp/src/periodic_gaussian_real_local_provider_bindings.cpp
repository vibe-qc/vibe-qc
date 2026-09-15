// Tiny inspection boundary; actual native HF/source owners only, no rows input.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vibeqc/periodic_gaussian_real_local_provider.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_real_local_provider(py::module_& m) {
    using namespace vibeqc;
    using Config=PeriodicGaussianRealLocalProviderConfig;
    using Live=PeriodicGaussianRealLocalProviderLiveInventory;
    using Caps=PeriodicGaussianRealLocalProviderCaps;
    using Plan=PeriodicGaussianRealLocalProviderPlan;
    using Receipt=PeriodicGaussianRealLocalProviderReceipt;
    using Progress=PeriodicGaussianRealLocalProviderProgress;
    using Stage=PeriodicGaussianRealLocalProviderStage;
    using Provider=PeriodicGaussianRealLocalProvider;
    using Complex=std::complex<double>;
    py::class_<Config>(m,"_PeriodicGaussianRealLocalProviderConfig").def(py::init<>())
        .def_readwrite("auxiliary_block",&Config::auxiliary_block).def_readwrite("panel",&Config::panel);
    py::class_<Live>(m,"_PeriodicGaussianRealLocalProviderLiveInventory").def(py::init<>())
        .def_readwrite("other_retained_bytes_per_worker",&Live::other_retained_bytes_per_worker)
        .def_readwrite("other_transient_bytes_per_worker",&Live::other_transient_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
#define GAUSSIAN_REAL_CAP(name) .def_readwrite(#name,&Caps::name)
    py::class_<Caps>(m,"_PeriodicGaussianRealLocalProviderCaps").def(py::init<>())
        GAUSSIAN_REAL_CAP(resources) GAUSSIAN_REAL_CAP(metric) GAUSSIAN_REAL_CAP(panel)
        GAUSSIAN_REAL_CAP(maximum_factor_panels) GAUSSIAN_REAL_CAP(maximum_tile_calls)
        GAUSSIAN_REAL_CAP(maximum_image_candidate_evaluations) GAUSSIAN_REAL_CAP(maximum_progress_callbacks)
        GAUSSIAN_REAL_CAP(maximum_scalar_work_units);
#undef GAUSSIAN_REAL_CAP
#define GAUSSIAN_REAL_PLAN(name) .def_readonly(#name,&Plan::name)
    py::class_<Plan>(m,"_PeriodicGaussianRealLocalProviderPlan")
        GAUSSIAN_REAL_PLAN(n_cells) GAUSSIAN_REAL_PLAN(n_basis) GAUSSIAN_REAL_PLAN(n_auxiliary)
        GAUSSIAN_REAL_PLAN(occupied_count) GAUSSIAN_REAL_PLAN(virtual_count) GAUSSIAN_REAL_PLAN(orbital_count)
        GAUSSIAN_REAL_PLAN(density_count) GAUSSIAN_REAL_PLAN(row_count) GAUSSIAN_REAL_PLAN(self_inverse_q_count)
        GAUSSIAN_REAL_PLAN(auxiliary_block_count) GAUSSIAN_REAL_PLAN(factor_panels) GAUSSIAN_REAL_PLAN(tile_calls)
        GAUSSIAN_REAL_PLAN(retained_row_bytes) GAUSSIAN_REAL_PLAN(norm_workspace_bytes)
        GAUSSIAN_REAL_PLAN(whitener_bytes) GAUSSIAN_REAL_PLAN(maximum_live_whitener_bytes)
        GAUSSIAN_REAL_PLAN(retained_partner_panel_bytes) GAUSSIAN_REAL_PLAN(panel_driver_owned_bytes)
        GAUSSIAN_REAL_PLAN(metric_phase_owned_upper_bound) GAUSSIAN_REAL_PLAN(panel_phase_owned_upper_bound)
        GAUSSIAN_REAL_PLAN(peak_owned_numerical_bytes) GAUSSIAN_REAL_PLAN(borrowed_basis_active_numeric_bytes)
        GAUSSIAN_REAL_PLAN(caller_gauge_bytes) GAUSSIAN_REAL_PLAN(live_wannier_bytes)
        GAUSSIAN_REAL_PLAN(live_domain_bytes) GAUSSIAN_REAL_PLAN(live_space_bytes)
        GAUSSIAN_REAL_PLAN(live_basis_bytes) GAUSSIAN_REAL_PLAN(basis_index_alias_bytes)
        GAUSSIAN_REAL_PLAN(macro_fixed_object_bytes) GAUSSIAN_REAL_PLAN(maximum_leaf_fixed_object_bytes)
        GAUSSIAN_REAL_PLAN(replicas_per_node) GAUSSIAN_REAL_PLAN(reference_base_node_bytes)
        GAUSSIAN_REAL_PLAN(per_replica_inventoried_bytes) GAUSSIAN_REAL_PLAN(required_node_memory_bytes)
        GAUSSIAN_REAL_PLAN(source_factory_calls) GAUSSIAN_REAL_PLAN(metric_calls) GAUSSIAN_REAL_PLAN(whitening_calls)
        GAUSSIAN_REAL_PLAN(reciprocal_candidate_evaluations_upper_bound)
        GAUSSIAN_REAL_PLAN(image_candidate_evaluations_upper_bound) GAUSSIAN_REAL_PLAN(progress_callback_upper_bound)
        GAUSSIAN_REAL_PLAN(input_check_work_units) GAUSSIAN_REAL_PLAN(driver_work_units)
        GAUSSIAN_REAL_PLAN(work_units) GAUSSIAN_REAL_PLAN(scalar_work_units)
        .def_property_readonly("plan_identity_sha256",&Plan::plan_identity_sha256);
#undef GAUSSIAN_REAL_PLAN
#define GAUSSIAN_REAL_RECEIPT(name) .def_readonly(#name,&Receipt::name)
    py::class_<Receipt>(m,"_PeriodicGaussianRealLocalProviderReceipt")
        GAUSSIAN_REAL_RECEIPT(completed_source_count) GAUSSIAN_REAL_RECEIPT(completed_metric_count)
        GAUSSIAN_REAL_RECEIPT(completed_whitening_count) GAUSSIAN_REAL_RECEIPT(completed_factor_panels)
        GAUSSIAN_REAL_RECEIPT(completed_tile_calls) GAUSSIAN_REAL_RECEIPT(progress_callback_count)
        GAUSSIAN_REAL_RECEIPT(source_factory_candidate_evaluations) GAUSSIAN_REAL_RECEIPT(reciprocal_candidate_evaluations)
        GAUSSIAN_REAL_RECEIPT(image_candidate_evaluations) GAUSSIAN_REAL_RECEIPT(charged_work_units_upper_bound)
        GAUSSIAN_REAL_RECEIPT(maximum_observed_owned_numerical_bytes)
        GAUSSIAN_REAL_RECEIPT(maximum_observed_per_replica_inventoried_bytes);
#undef GAUSSIAN_REAL_RECEIPT
    py::enum_<Stage>(m,"_PeriodicGaussianRealLocalProviderStage")
        .value("Begin",Stage::Begin).value("Source",Stage::Source).value("Metric",Stage::Metric)
        .value("Whitening",Stage::Whitening).value("Panel",Stage::Panel)
        .value("PairComplete",Stage::PairComplete).value("Complete",Stage::Complete);
#define GAUSSIAN_REAL_PROGRESS(name) .def_readonly(#name,&Progress::name)
    py::class_<Progress>(m,"_PeriodicGaussianRealLocalProviderProgress")
        GAUSSIAN_REAL_PROGRESS(stage) GAUSSIAN_REAL_PROGRESS(q_index) GAUSSIAN_REAL_PROGRESS(auxiliary_begin)
        GAUSSIAN_REAL_PROGRESS(completed_source_count) GAUSSIAN_REAL_PROGRESS(completed_factor_panels)
        GAUSSIAN_REAL_PROGRESS(completed_tile_calls) GAUSSIAN_REAL_PROGRESS(charged_work_units_upper_bound);
#undef GAUSSIAN_REAL_PROGRESS
    py::class_<Provider>(m,"_PeriodicGaussianRealLocalProvider")
        .def_property_readonly("provider",&Provider::provider,py::return_value_policy::reference_internal)
        .def_property_readonly("memory",[](const Provider& p) { return p.memory(); })
        .def_property_readonly("receipt",[](const Provider& p) { return p.receipt(); })
        .def_property_readonly("identity_sha256",&Provider::identity_sha256)
        .def_property_readonly("source_context_identity_sha256",&Provider::source_context_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256",&Provider::hf_reference_source_identity_sha256)
        .def_property_readonly("consumed_sources_identity_sha256",&Provider::consumed_sources_identity_sha256)
        .def_property_readonly("matched_finite_gaussian_hf_recipe",&Provider::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("bitwise_hf_factor_consumption_verified",&Provider::bitwise_hf_factor_consumption_verified)
        .def_property_readonly("infinite_source_accuracy_certified",&Provider::infinite_source_accuracy_certified);
    m.def("_plan_periodic_gaussian_real_local_provider",&plan_periodic_gaussian_real_local_provider,
        py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("domain"),py::arg("space"),py::arg("basis"),
        py::arg("config"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_real_local_provider",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
        const py::array& gauges,const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
        const PeriodicCorrelationRealLocalBasis& basis,const Config& config,
        const PeriodicCorrelationRealLocalProviderOptions& options,const Live& live,const Caps& caps,py::object progress) {
        const auto& state=reference.state();
        if (state.n_kpoints()>8 || state.n_basis()>8 || state.n_correlated_occupied()>4
            || reference.dimensions().n_auxiliary>8 || domain.domain_dimension()>32 || basis.memory().orbital_count>16
            || caps.resources.maximum_owned_numeric_bytes>(1ULL<<24) || caps.resources.maximum_work_units>1000000000000000000ULL
            || caps.maximum_factor_panels>64 || caps.maximum_tile_calls>4096 || caps.maximum_progress_callbacks>512
            || caps.maximum_scalar_work_units>1000000)
            throw std::length_error("Gaussian real provider diagnostic exceeds tiny shape/work cap");
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim()!=3 || !(gauges.flags()&py::array::c_style)
            || gauges.shape(0)!=static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1)!=static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2)!=static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("Gaussian real provider gauges require C-contiguous complex128 [K,active,active]");
        if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            throw std::invalid_argument("Gaussian real provider progress must be callable or None");
        // Controls are immutable local snapshots. Numerical inputs are NOT
        // copied: argument references pin their owners for the synchronous
        // call; every callback boundary rechecks gauges and actual bases.
        // Python callback/event/reference metadata is separate from the
        // native numerical inventory and bounded by the tiny shape/call caps.
        const auto settings=config; const auto policy=options; const auto inventory=live; const auto limits=caps;
        struct Bridge {
            py::object callback;
            static void call(const Progress& event,void* context) {
                py::gil_scoped_acquire acquire;
                static_cast<Bridge*>(context)->callback(py::cast(event,py::return_value_policy::copy));
            }
        } bridge{progress};
        const auto callback=progress.is_none()?nullptr:&Bridge::call;
        const auto* data=static_cast<const Complex*>(gauges.data()); const auto count=static_cast<std::size_t>(gauges.size());
        py::gil_scoped_release release;
        return make_periodic_gaussian_real_local_provider(hf,reference,ao,auxiliary,wannier,data,count,domain,space,basis,
            settings,policy,inventory,limits,callback,&bridge);
    },py::arg("hf"),py::arg("reference"),py::arg("ao"),py::arg("auxiliary"),py::arg("wannier"),
        py::arg("gauges").noconvert(),py::arg("domain"),py::arg("space"),py::arg("basis"),py::arg("config"),
        py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("progress")=py::none());
}
