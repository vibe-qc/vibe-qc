// Tiny diagnostic entry: numerical construction and iteration stay native.
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gaussian_pair_mp2.hpp"
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

void bind_periodic_gaussian_pair_mp2(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianPairMP2Options;
    using Live=PeriodicGaussianPairMP2LiveInventory;
    using Caps=PeriodicGaussianPairMP2Caps;
    using Plan=PeriodicGaussianPairMP2Plan;
    using Diagnostics=PeriodicGaussianPairMP2Diagnostics;
    using Stage=PeriodicGaussianPairMP2Stage;
    using Progress=PeriodicGaussianPairMP2Progress;
    using Result=PeriodicGaussianPairMP2Result;
    using SourceKind=PeriodicGaussianPairMP2SourceKind;
    py::enum_<SourceKind>(m,"_PeriodicGaussianPairMP2SourceKind")
        .value("COMMON_FACTOR_ROWS",SourceKind::CommonFactorRows)
        .value("DIRECT_PAO_GRAM",SourceKind::DirectPAOGram);
    py::class_<Options>(m,"_PeriodicGaussianPairMP2Options").def(py::init<>())
        .def_readwrite("pnos",&Options::pnos).def_readwrite("projection",&Options::projection)
        .def_readwrite("solver",&Options::solver)
        .def_readwrite("maximum_occupied_virtual_fock_norm",&Options::maximum_occupied_virtual_fock_norm);
    py::class_<Live>(m,"_PeriodicGaussianPairMP2LiveInventory").def(py::init<>())
        .def_readwrite("other_live_bytes_per_worker",&Live::other_live_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianPairMP2Caps"); caps.def(py::init<>());
#define GPM_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GPM_CAP(pnos); GPM_CAP(projection); GPM_CAP(solver); GPM_CAP(maximum_owned_numerical_bytes);
    GPM_CAP(maximum_per_worker_inventoried_bytes); GPM_CAP(maximum_node_inventoried_bytes);
    GPM_CAP(maximum_pair_count); GPM_CAP(maximum_integral_calls); GPM_CAP(maximum_progress_callbacks); GPM_CAP(maximum_work_units);
#undef GPM_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianPairMP2Plan");
#define GPM_PLAN(f) plan.def_readonly(#f,&Plan::f)
    GPM_PLAN(occupied_count); GPM_PLAN(common_virtual_dimension); GPM_PLAN(pair_count);
    GPM_PLAN(source_kind);
    GPM_PLAN(retained_pair_output_upper_bytes); GPM_PLAN(pair_generation_phase_upper_bytes);
    GPM_PLAN(solver_phase_owned_upper_bytes); GPM_PLAN(peak_owned_numerical_bytes);
    GPM_PLAN(borrowed_basis_bytes); GPM_PLAN(borrowed_provider_row_bytes);
    GPM_PLAN(borrowed_gaussian_bytes);GPM_PLAN(borrowed_wannier_bytes);GPM_PLAN(borrowed_gauge_bytes);
    GPM_PLAN(control_storage_reservation_bytes); GPM_PLAN(retained_pair_seal_bytes);
    GPM_PLAN(domain_generated); GPM_PLAN(borrowed_domain_builder_bytes); GPM_PLAN(retained_generation_embedding_upper_bytes);
    GPM_PLAN(retained_pair_geometry_upper_bytes); GPM_PLAN(retained_pair_geometry_control_upper_bytes);
    GPM_PLAN(replicas_per_node); GPM_PLAN(reference_base_node_bytes);
    GPM_PLAN(per_worker_inventoried_bytes); GPM_PLAN(required_node_memory_bytes);
    GPM_PLAN(integral_calls_upper_bound); GPM_PLAN(progress_callback_upper_bound);
    GPM_PLAN(gram_builds_upper_bound);GPM_PLAN(factor_panels_upper_bound);GPM_PLAN(tile_calls_upper_bound);
    GPM_PLAN(driver_work_units); GPM_PLAN(work_units_upper_bound);
#undef GPM_PLAN
    plan.def_property_readonly("solver_upper",[](const Plan& p){return p.solver_upper;});
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianPairMP2Diagnostics");
#define GPM_DIAG(f) diagnostics.def_readonly(#f,&Diagnostics::f)
    GPM_DIAG(completed_pairs); GPM_DIAG(completed_integral_calls); GPM_DIAG(completed_progress_callbacks);
    GPM_DIAG(source_kind);GPM_DIAG(completed_gram_builds);GPM_DIAG(completed_factor_panels);GPM_DIAG(completed_tile_calls);
    GPM_DIAG(retained_pair_bytes); GPM_DIAG(minimum_pair_rank); GPM_DIAG(maximum_pair_rank); GPM_DIAG(zero_rank_pairs);
    GPM_DIAG(domain_generated); GPM_DIAG(generation_dimension_sum);
    GPM_DIAG(retained_generation_coefficient_bytes);
    GPM_DIAG(diagonal_generation_dimension_sum); GPM_DIAG(retained_generation_embedding_bytes);
    GPM_DIAG(retained_pair_geometry_bytes); GPM_DIAG(retained_pair_geometry_control_bytes);
    GPM_DIAG(complete_common_finite_torus_basis); GPM_DIAG(all_pairs_full_rank);
    GPM_DIAG(occupied_virtual_fock_norm_upper_bound); GPM_DIAG(maximum_diagonal_integral_projection_norm);
#undef GPM_DIAG
    py::enum_<Stage>(m,"_PeriodicGaussianPairMP2Stage").value("BEGIN",Stage::Begin)
        .value("PAIR_COMPLETE",Stage::PairComplete).value("SOLVER",Stage::Solver).value("FINISHED",Stage::Finished)
        .value("PAIR_DOMAIN_READY",Stage::PairDomainReady);
    py::class_<Progress>(m,"_PeriodicGaussianPairMP2Progress")
        .def_readonly("stage",&Progress::stage).def_readonly("callback_count",&Progress::callback_count)
        .def_readonly("completed_pairs",&Progress::completed_pairs)
        .def_readonly("occupied_i",&Progress::occupied_i).def_readonly("occupied_j",&Progress::occupied_j)
        .def_readonly("pair_rank",&Progress::pair_rank)
        .def_property_readonly("solver",[](const Progress& p){return p.solver;});
    py::class_<Result>(m,"_PeriodicGaussianPairMP2Result")
        .def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("solver",&Result::solver,py::return_value_policy::reference_internal)
        .def("pair",&Result::pair,py::return_value_policy::reference_internal)
        .def_property_readonly("domain_generated",&Result::domain_generated)
        .def_property_readonly("source_kind",&Result::source_kind)
        .def_property_readonly("direct_gram_source",&Result::direct_gram_source)
        .def("diagonal_generation_embedding",&Result::diagonal_generation_embedding,py::return_value_policy::reference_internal)
        .def("pair_generation_geometry_view",[](const Result& r,std::uint64_t i,std::uint64_t j){
            return r.pair_generation_geometry_view(i,j);},py::keep_alive<0,1>())
        .def("pair_generation_geometry_metadata",[](const Result& r,std::uint64_t i,std::uint64_t j) {
            const auto& g=r.pair_generation_geometry(i,j);py::dict out;
            out["domain_dimension"]=g.domain().domain_dimension();out["generation_dimension"]=g.real_space().space().retained_dimension();
            out["retained_numerical_bytes"]=g.retained_numerical_bytes();out["retained_control_storage_bytes"]=g.retained_control_storage_bytes();
            out["embedding_bytes"]=g.embedding().memory().output_numerical_bytes;
            out["identity_sha256"]=g.identity_sha256();out["embedding_identity_sha256"]=g.embedding().identity_sha256();
            out["builder_identity_sha256"]=g.builder_identity_sha256();return out;
        })
        .def_property_readonly("converged",&Result::converged)
        .def_property_readonly("periodic_energy_per_cell",&Result::periodic_energy_per_cell)
        .def_property_readonly("correlation_energy_per_cell",&Result::correlation_energy_per_cell)
        .def_property_readonly("total_energy_per_cell",&Result::total_energy_per_cell)
        .def_property_readonly("matched_finite_gaussian_hf_recipe",&Result::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("production_dlpno",&Result::production_dlpno)
        .def_property_readonly("infinite_source_accuracy_certified",&Result::infinite_source_accuracy_certified)
        .def_property_readonly("identity_sha256",&Result::identity_sha256)
        .def_property_readonly("pair_spaces_identity_sha256",&Result::pair_spaces_identity_sha256)
        .def_property_readonly("provider_identity_sha256",&Result::provider_identity_sha256)
        .def_property_readonly("gram_sources_identity_sha256",&Result::gram_sources_identity_sha256)
        .def_property_readonly("domain_builder_identity_sha256",&Result::domain_builder_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256",&Result::hf_reference_source_identity_sha256);
    const auto tiny=[](const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianRealLocalProvider& provider,const Options& options,const Live& live,const Caps& caps) {
        if(ref.state().n_kpoints()>8 || basis.memory().occupied_count>4 || basis.memory().virtual_count>4
            || options.pnos.pno_eigensolver.max_sweeps>200 || options.pnos.semicanonical_eigensolver.max_sweeps>200
            || options.solver.maximum_iterations>256)
            throw std::length_error("Gaussian pair MP2 diagnostic exceeds tiny dimensions or iterations");
        auto p=plan_periodic_gaussian_pair_mp2(ref,basis,provider,options,live,caps);
        if(p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units_upper_bound>1000000000000000ULL || p.progress_callback_upper_bound>1024)
            throw std::length_error("Gaussian pair MP2 diagnostic exceeds tiny memory, work or callbacks");
        return p;
    };
    m.def("_plan_periodic_gaussian_pair_mp2",tiny,py::arg("reference"),py::arg("basis"),py::arg("provider"),
        py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_run_periodic_gaussian_pair_mp2",[tiny](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        Options options,Live live,Caps caps,py::object callback) {
        (void)tiny(ref,basis,provider,options,live,caps);
        if(!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian pair MP2 progress must be callable");
        const auto progress=[](const Progress& p,void* context) {
            (*static_cast<py::object*>(context))(py::cast(Progress(p)));
        };
        // No borrowed NumPy inputs; native owners are pinned for this call.
        // Keep GIL for the deliberately tiny scalar-only diagnostic callback.
        return run_periodic_gaussian_pair_mp2(ref,basis,provider,options,live,caps,
            callback.is_none()?nullptr:+progress,callback.is_none()?nullptr:&callback);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("options"),py::arg("live"),
      py::arg("caps"),py::arg("progress")=py::none());
}
