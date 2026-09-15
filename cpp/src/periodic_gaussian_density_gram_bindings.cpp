// Tiny callback-free inspection boundary; no caller rows or forged source.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_density_gram.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace periodic_gaussian_density_gram_python {
using namespace vibeqc;
using Range = PeriodicGaussianDensityRange;
using Selection = PeriodicGaussianDensityGramSelection;
using Config = PeriodicGaussianDensityGramConfig;
using Caps = PeriodicGaussianDensityGramCaps;
using Result = PeriodicGaussianDensityGramBlock;

void tiny(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationRealLocalBasis& basis,
    const Selection& selection,const Config& config,const Caps& caps) {
    if (!ref.state_handle() || !hf.context_handle())
        throw std::invalid_argument("Gaussian density Gram diagnostic requires live native source owners");
    if (hf.plan().n_kpoints>8 || hf.plan().n_basis>4
        || hf.context_handle()->inventory().auxiliary.function_count>4
        || basis.memory().orbital_count>8 || domain.domain_dimension()>32
        || config.auxiliary_block>4 || selection.left.count>36 || selection.right.count>36
        || caps.resources.maximum_owned_numeric_bytes>(1ULL<<23)
        || caps.resources.maximum_node_inventoried_bytes>(1ULL<<27)
        || caps.resources.maximum_work_units>10000000000000000000ULL
        || hf.plan().fock.config.source_caps.maximum_candidate_evaluations>65536)
        throw std::length_error("Gaussian density Gram diagnostic exceeds tiny shape/source/resource bounds");
}

py::array_t<double> copy(const Result& result) {
    const auto& s=result.memory().selection;
    if (s.left.count>36 || s.right.count>36)
        throw std::length_error("Gaussian density Gram diagnostic copy exceeds tiny shape");
    const auto* data=result.data();
    py::array_t<double> output({static_cast<py::ssize_t>(s.left.count),static_cast<py::ssize_t>(s.right.count)});
    if (s.left.count && s.right.count) std::memcpy(output.mutable_data(),data,8*s.left.count*s.right.count);
    output.attr("setflags")(false);
    return output;
}
} // namespace periodic_gaussian_density_gram_python

void bind_periodic_gaussian_density_gram(py::module_& m) {
    using namespace vibeqc;
    namespace binding=periodic_gaussian_density_gram_python;
    using Range=PeriodicGaussianDensityRange;
    using Selection=PeriodicGaussianDensityGramSelection;
    using Config=PeriodicGaussianDensityGramConfig;
    using Options=PeriodicGaussianDensityGramOptions;
    using Live=PeriodicGaussianDensityGramLiveInventory;
    using Caps=PeriodicGaussianDensityGramCaps;
    using Plan=PeriodicGaussianDensityGramPlan;
    using Diagnostics=PeriodicGaussianDensityGramDiagnostics;
    using Result=PeriodicGaussianDensityGramBlock;
    using Complex=std::complex<double>;
    // Options and Live are aliases of registered provider/panel classes.
    // They must not be registered a second time under new Python names.
    py::class_<Range>(m,"_PeriodicGaussianDensityRange").def(py::init<>())
        .def_readwrite("begin",&Range::begin).def_readwrite("count",&Range::count);
    py::class_<Selection>(m,"_PeriodicGaussianDensityGramSelection").def(py::init<>())
        .def_readwrite("left",&Selection::left).def_readwrite("right",&Selection::right);
    py::class_<Config>(m,"_PeriodicGaussianDensityGramConfig").def(py::init<>())
        .def_readwrite("auxiliary_block",&Config::auxiliary_block).def_readwrite("panel",&Config::panel);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianDensityGramCaps"); caps.def(py::init<>());
#define DGRAM_CAP(f) caps.def_readwrite(#f,&Caps::f)
    DGRAM_CAP(resources);DGRAM_CAP(metric);DGRAM_CAP(panel);DGRAM_CAP(maximum_left_density_count);
    DGRAM_CAP(maximum_right_density_count);DGRAM_CAP(maximum_output_elements);DGRAM_CAP(maximum_factor_panels);
    DGRAM_CAP(maximum_tile_calls);DGRAM_CAP(maximum_image_candidate_evaluations);DGRAM_CAP(maximum_leaf_control_bytes);
#undef DGRAM_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianDensityGramPlan");
    plan.def_property_readonly("selection",[](const Plan& p){return p.selection;});
#define DGRAM_PLAN(f) plan.def_readonly(#f,&Plan::f)
    DGRAM_PLAN(n_cells);DGRAM_PLAN(n_basis);DGRAM_PLAN(n_auxiliary);DGRAM_PLAN(occupied_count);
    DGRAM_PLAN(virtual_count);DGRAM_PLAN(orbital_count);DGRAM_PLAN(common_density_count);
    DGRAM_PLAN(selected_density_count);DGRAM_PLAN(output_elements);DGRAM_PLAN(left_run_count);
    DGRAM_PLAN(right_run_count);DGRAM_PLAN(maximum_run_length);DGRAM_PLAN(row_count);
    DGRAM_PLAN(self_inverse_q_count);DGRAM_PLAN(auxiliary_block_count);DGRAM_PLAN(factor_panels);DGRAM_PLAN(tile_calls);
    DGRAM_PLAN(retained_output_bytes);DGRAM_PLAN(integral_accumulator_bytes);DGRAM_PLAN(norm_workspace_bytes);
    DGRAM_PLAN(row_slab_bytes);DGRAM_PLAN(factor_phase_retained_bytes);DGRAM_PLAN(whitener_bytes);
    DGRAM_PLAN(maximum_live_whitener_bytes);DGRAM_PLAN(retained_panel_upper_bytes);DGRAM_PLAN(panel_driver_owned_upper_bytes);
    DGRAM_PLAN(metric_phase_upper_bytes);DGRAM_PLAN(panel_phase_upper_bytes);DGRAM_PLAN(peak_owned_numerical_bytes);
    DGRAM_PLAN(borrowed_local_numerical_bytes);DGRAM_PLAN(borrowed_basis_active_numeric_bytes);
    DGRAM_PLAN(macro_control_storage_bytes);DGRAM_PLAN(minimum_leaf_control_bytes);DGRAM_PLAN(control_storage_reservation_bytes);
    DGRAM_PLAN(replicas_per_node);DGRAM_PLAN(reference_base_node_bytes);DGRAM_PLAN(per_worker_inventoried_bytes);
    DGRAM_PLAN(required_node_memory_bytes);DGRAM_PLAN(source_factory_calls);DGRAM_PLAN(metric_calls);DGRAM_PLAN(whitening_calls);
    DGRAM_PLAN(reciprocal_candidate_evaluations_upper_bound);DGRAM_PLAN(image_candidate_evaluations_upper_bound);
    DGRAM_PLAN(input_validation_work_units);DGRAM_PLAN(driver_work_units);DGRAM_PLAN(work_units);
#undef DGRAM_PLAN
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianDensityGramDiagnostics");
#define DGRAM_DIAG(f) diagnostics.def_readonly(#f,&Diagnostics::f)
    DGRAM_DIAG(completed_source_count);DGRAM_DIAG(completed_metric_count);DGRAM_DIAG(completed_whitening_count);
    DGRAM_DIAG(completed_factor_panels);DGRAM_DIAG(completed_tile_calls);DGRAM_DIAG(reciprocal_candidate_evaluations);
    DGRAM_DIAG(image_candidate_evaluations);DGRAM_DIAG(charged_work_units_upper_bound);
    DGRAM_DIAG(maximum_observed_owned_numerical_bytes);DGRAM_DIAG(maximum_observed_per_worker_inventoried_bytes);
    DGRAM_DIAG(maximum_integral_roundoff_error);
#undef DGRAM_DIAG
    diagnostics.def_property_readonly("real_projection",[](const Diagnostics& d){return d.real_projection;});
    auto result=py::class_<Result>(m,"_PeriodicGaussianDensityGramBlock");
    result.def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("state",&Result::state_handle)
        .def_property_readonly("context",&Result::context_handle)
        .def("matrix_copy",&binding::copy)
        .def("element",&Result::element,py::arg("left"),py::arg("right"))
        .def("left_density",&Result::left_density,py::arg("index"))
        .def("right_density",&Result::right_density,py::arg("index"));
#define DGRAM_RESULT(f) result.def_property_readonly(#f,&Result::f)
    DGRAM_RESULT(identity_sha256);DGRAM_RESULT(payload_sha256);DGRAM_RESULT(hf_reference_source_identity_sha256);
    DGRAM_RESULT(source_context_identity_sha256);DGRAM_RESULT(basis_identity_sha256);DGRAM_RESULT(consumed_sources_identity_sha256);
    DGRAM_RESULT(matched_finite_gaussian_hf_recipe);DGRAM_RESULT(original_provider_projection_reproduced_bitwise);
    DGRAM_RESULT(all_common_densities_audited);DGRAM_RESULT(infinite_source_accuracy_certified);DGRAM_RESULT(production_dlpno);
#undef DGRAM_RESULT
    m.def("_plan_periodic_gaussian_density_gram_diagnostic",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
        const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
        Selection selection,Config config,Options options,Live live,Caps caps) {
        binding::tiny(hf,reference,domain,basis,selection,config,caps);
        return plan_periodic_gaussian_density_gram(hf,reference,wannier,domain,space,basis,
            selection,config,options,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("domain"),py::arg("space"),py::arg("basis"),
        py::arg("selection"),py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_build_periodic_gaussian_density_gram_diagnostic",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
        const py::array& gauges,const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
        const PeriodicCorrelationRealLocalBasis& basis,Selection selection,Config config,Options options,Live live,Caps caps) {
        binding::tiny(hf,reference,domain,basis,selection,config,caps);
        const auto& state=reference.state();
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || !(gauges.flags()&py::array::c_style)
            || gauges.ndim()!=3 || gauges.shape(0)!=static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1)!=static_cast<py::ssize_t>(state.n_correlated_occupied()) || gauges.shape(2)!=gauges.shape(1)
            || reinterpret_cast<std::uintptr_t>(gauges.data())%alignof(Complex))
            throw py::type_error("Gaussian density Gram gauges require exact C-contiguous complex128 [K,active,active]");
        // The GIL stays held. Argument owners and the exact ndarray are
        // pinned; copied scalar controls cannot change during this native
        // callback-free call. No numerical input or factor rows are copied.
        return build_periodic_gaussian_density_gram(hf,reference,ao,auxiliary,wannier,
            static_cast<const Complex*>(gauges.data()),static_cast<std::size_t>(gauges.size()),
            domain,space,basis,selection,config,options,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),
        py::arg("gauges").noconvert(),py::arg("domain"),py::arg("space"),py::arg("basis"),py::arg("selection"),
        py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
}
