// Tiny actual-source connection tests, not the production DLPNO frontend.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "vibeqc/periodic_gaussian_selected_local_ccsd_t.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_selected_local_ccsd_t(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianSelectedLocalCCSDTOptions;
    using Live=PeriodicGaussianSelectedLocalCCSDTLiveInventory;
    using Caps=PeriodicGaussianSelectedLocalCCSDTCaps;
    using Plan=PeriodicGaussianSelectedLocalCCSDTPlan;
    using Stage=PeriodicGaussianSelectedLocalCCSDTStage;
    using Progress=PeriodicGaussianSelectedLocalCCSDTProgress;
    using Diagnostics=PeriodicGaussianSelectedLocalCCSDTDiagnostics;
    using Result=PeriodicGaussianSelectedLocalCCSDTResult;
    using HF=PeriodicGaussianRHFResult;
    using Reference=PeriodicCorrelationAdmittedReference;
    auto options=py::class_<Options>(m,"_PeriodicGaussianSelectedLocalCCSDTOptions"); options.def(py::init<>());
#define GSL_OPTION(f) options.def_readwrite(#f,&Options::f)
    GSL_OPTION(localization); GSL_OPTION(domain); GSL_OPTION(space); GSL_OPTION(basis);
    GSL_OPTION(factors); GSL_OPTION(real_factors); GSL_OPTION(correlation);
    GSL_OPTION(pair_local_correlation); GSL_OPTION(pair_mp2); GSL_OPTION(pair_ccsd);
    GSL_OPTION(selected_pair_domains); GSL_OPTION(occupied_domains); GSL_OPTION(domain_pair_mp2);
    GSL_OPTION(physical_particle_hole_ccsd); GSL_OPTION(particle_hole); GSL_OPTION(particle_hole_options);
    GSL_OPTION(triple_spaces); GSL_OPTION(local_triples);
#undef GSL_OPTION
    py::class_<Live>(m,"_PeriodicGaussianSelectedLocalCCSDTLiveInventory").def(py::init<>())
        .def_readwrite("other_live_bytes_per_worker",&Live::other_live_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianSelectedLocalCCSDTCaps"); caps.def(py::init<>());
#define GSL_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GSL_CAP(iao_source); GSL_CAP(localization); GSL_CAP(maximum_domain_owned_numerical_bytes);
    GSL_CAP(space); GSL_CAP(basis); GSL_CAP(factors); GSL_CAP(correlation);
    GSL_CAP(pair_mp2); GSL_CAP(pair_ccsd); GSL_CAP(triple_spaces); GSL_CAP(local_triples);
    GSL_CAP(pair_domain_builder); GSL_CAP(domain_pair_mp2);
    GSL_CAP(particle_hole);
    GSL_CAP(maximum_pair_control_storage_bytes); GSL_CAP(maximum_pair_rank_padding_bytes);
    GSL_CAP(maximum_owned_numerical_bytes); GSL_CAP(maximum_per_worker_inventoried_bytes);
    GSL_CAP(maximum_node_inventoried_bytes); GSL_CAP(maximum_factor_control_storage_bytes);
    GSL_CAP(maximum_progress_callbacks); GSL_CAP(maximum_work_units);
#undef GSL_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianSelectedLocalCCSDTPlan");
#define GSL_PLAN(f) plan.def_readonly(#f,&Plan::f)
    GSL_PLAN(n_kpoints); GSL_PLAN(n_basis); GSL_PLAN(domain_count); GSL_PLAN(occupied_count);
    GSL_PLAN(caller_index_bytes); GSL_PLAN(auxiliary_basis_active_bytes);
    GSL_PLAN(additional_minimal_geometry_bytes_upper_bound); GSL_PLAN(borrowed_input_bytes_upper_bound);
    GSL_PLAN(control_storage_reservation_bytes); GSL_PLAN(localization_phase_owned_upper_bound);
    GSL_PLAN(pair_local_correlation); GSL_PLAN(pair_rank_padding_reservation_bytes);
    GSL_PLAN(selected_pair_domains); GSL_PLAN(pair_domain_preparation_phase_owned_upper_bound);
    GSL_PLAN(physical_particle_hole_ccsd); GSL_PLAN(pair_mp2_phase_owned_upper_bound);
    GSL_PLAN(physical_pair_ccsd_phase_owned_upper_bound); GSL_PLAN(physical_pair_ccsd_metadata_work_units_upper_bound);
    GSL_PLAN(domain_phase_owned_upper_bound); GSL_PLAN(space_phase_owned_upper_bound);
    GSL_PLAN(basis_phase_owned_upper_bound); GSL_PLAN(factors_phase_owned_upper_bound);
    GSL_PLAN(correlation_phase_owned_upper_bound); GSL_PLAN(owned_numerical_upper_bound);
    GSL_PLAN(per_worker_inventoried_bytes_upper_bound); GSL_PLAN(node_inventoried_bytes_upper_bound);
    GSL_PLAN(replicas_per_node); GSL_PLAN(reference_base_node_bytes); GSL_PLAN(input_check_work_units);
    GSL_PLAN(domain_work_units_upper_bound); GSL_PLAN(progress_callback_upper_bound); GSL_PLAN(work_units_upper_bound);
#undef GSL_PLAN
    py::enum_<Stage>(m,"_PeriodicGaussianSelectedLocalCCSDTStage")
        .value("BEGIN",Stage::Begin).value("LOCALIZATION",Stage::Localization)
        .value("DOMAIN_READY",Stage::DomainReady).value("SPACE_READY",Stage::SpaceReady)
        .value("BASIS_READY",Stage::BasisReady).value("FACTORS",Stage::Factors)
        .value("FACTORS_READY",Stage::FactorsReady).value("CORRELATION",Stage::Correlation)
        .value("FINISHED",Stage::Finished).value("PAIR_MP2",Stage::PairMP2)
        .value("PAIR_CCSD",Stage::PairCCSD).value("TRIPLE_SPACES",Stage::TripleSpaces)
        .value("LOCAL_TRIPLES",Stage::LocalTriples).value("PAIR_DOMAINS_READY",Stage::PairDomainsReady);
    py::class_<Progress>(m,"_PeriodicGaussianSelectedLocalCCSDTProgress")
        .def_readonly("stage",&Progress::stage).def_readonly("callback_count",&Progress::callback_count)
        .def_readonly("retained_virtual_count",&Progress::retained_virtual_count)
        .def_property_readonly("localization",[](const Progress& p) { return p.localization; })
        .def_property_readonly("factors",[](const Progress& p) { return p.factors; })
        .def_property_readonly("correlation",[](const Progress& p) { return p.correlation; })
        .def_property_readonly("pair_mp2",[](const Progress& p) { return p.pair_mp2; })
        .def_property_readonly("pair_ccsd",[](const Progress& p) { return p.pair_ccsd; })
        .def_property_readonly("triple_spaces",[](const Progress& p) { return p.triple_spaces; })
        .def_property_readonly("local_triples",[](const Progress& p) { return p.local_triples; });
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianSelectedLocalCCSDTDiagnostics");
#define GSL_DIAG(f) diagnostics.def_readonly(#f,&Diagnostics::f)
    GSL_DIAG(completed_progress_callbacks); GSL_DIAG(completed_input_checks); GSL_DIAG(retained_virtual_count);
    GSL_DIAG(complete_finite_torus_basis); GSL_DIAG(hf_energy_per_cell);
    GSL_DIAG(physical_particle_hole_evaluated);
#undef GSL_DIAG
#define GSL_DIAG_VALUE(f) diagnostics.def_property_readonly(#f,[](const Diagnostics& d) { return d.f; })
    GSL_DIAG_VALUE(localization_memory); GSL_DIAG_VALUE(localization); GSL_DIAG_VALUE(domain);
    GSL_DIAG_VALUE(space); GSL_DIAG_VALUE(basis); GSL_DIAG_VALUE(factors_memory); GSL_DIAG_VALUE(factors);
    GSL_DIAG_VALUE(pair_domain_builder_memory); GSL_DIAG_VALUE(pair_domain_builder);
    GSL_DIAG_VALUE(physical_particle_hole_memory);
#undef GSL_DIAG_VALUE
    py::class_<Result>(m,"_PeriodicGaussianSelectedLocalCCSDTResult")
        .def_property_readonly("plan",[](const Result& r) { return r.plan(); })
        .def_property_readonly("diagnostics",[](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("correlation_evaluated",&Result::correlation_evaluated)
        .def_property_readonly("converged",&Result::converged)
        .def_property_readonly("pair_mp2_evaluated",&Result::pair_mp2_evaluated)
        .def_property_readonly("pair_ccsd_evaluated",&Result::pair_ccsd_evaluated)
        .def_property_readonly("triple_spaces_evaluated",&Result::triple_spaces_evaluated)
        .def_property_readonly("local_triples_evaluated",&Result::local_triples_evaluated)
        .def_property_readonly("pair_mp2",&Result::pair_mp2,py::return_value_policy::reference_internal)
        .def_property_readonly("pair_ccsd",&Result::pair_ccsd,py::return_value_policy::reference_internal)
        .def_property_readonly("triple_spaces",&Result::triple_spaces,py::return_value_policy::reference_internal)
        .def_property_readonly("local_triples",&Result::local_triples,py::return_value_policy::reference_internal)
        .def_property_readonly("unfinished_localization",&Result::unfinished_localization,py::return_value_policy::reference_internal)
        .def_property_readonly("correlation",&Result::correlation,py::return_value_policy::reference_internal)
        .def_property_readonly("matched_finite_gaussian_hf_recipe",&Result::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("bitwise_hf_factor_consumption_verified",&Result::bitwise_hf_factor_consumption_verified)
        .def_property_readonly("infinite_source_accuracy_certified",&Result::infinite_source_accuracy_certified)
        .def_property_readonly("production_dlpno",&Result::production_dlpno)
        .def_property_readonly("periodic_energy_per_cell",&Result::periodic_energy_per_cell)
        .def_property_readonly("correlation_energy_per_cell",&Result::correlation_energy_per_cell)
        .def_property_readonly("total_energy_per_cell",&Result::total_energy_per_cell)
        .def_property_readonly("hf_reference_source_identity_sha256",&Result::hf_reference_source_identity_sha256)
        .def_property_readonly("input_identity_sha256",&Result::input_identity_sha256)
        .def_property_readonly("localization_identity_sha256",&Result::localization_identity_sha256)
        .def_property_readonly("provider_identity_sha256",&Result::provider_identity_sha256)
        .def_property_readonly("identity_sha256",&Result::identity_sha256);
    const auto selection=[](const py::array& domain,const py::array& occupied,std::uint64_t translation) {
        for (const auto* a:{&domain,&occupied}) {
            if (!a->dtype().is(py::dtype::of<std::uint64_t>())||!(a->flags()&py::array::c_style)
                ||a->ndim()!=2||a->shape(0)<1||a->shape(1)!=2||a->strides(1)!=8||a->strides(0)!=16
                ||reinterpret_cast<std::uintptr_t>(a->data())%alignof(std::uint64_t))
                throw std::invalid_argument("Gaussian selected-local diagnostic requires aligned contiguous uint64 [count,2] indices");
        }
        if (domain.shape(0)>32||occupied.shape(0)>4)
            throw std::length_error("Gaussian selected-local diagnostic exceeds tiny selection bounds");
        return PeriodicGaussianSelectedLocalSelection{static_cast<const std::uint64_t*>(domain.data()),
            static_cast<std::size_t>(domain.size()),static_cast<std::size_t>(domain.shape(0)),
            static_cast<const std::uint64_t*>(occupied.data()),static_cast<std::size_t>(occupied.size()),
            static_cast<std::size_t>(occupied.shape(0)),translation};
    };
    const auto tiny_plan=[selection](const HF& hf,const Reference& reference,const py::array& domain,
        const py::array& occupied,std::uint64_t translation,const Options& o,const Live& l,const Caps& c) {
        const auto view=selection(domain,occupied,translation);
        if (!reference.state_handle())
            throw std::invalid_argument("Gaussian selected-local diagnostic requires a live reference owner");
        const auto& state=reference.state();
        if (state.n_basis()>4||state.n_kpoints()>8
            ||std::min<std::uint64_t>(view.domain_count,state.n_kpoints()*state.n_virtual())>4
            ||o.localization.optimizer.maximum_iterations>256||o.localization.optimizer.maximum_line_search_trials>16
            ||o.correlation.ccsd.maximum_iterations>128||o.correlation.triples.maximum_iterations>128
            ||o.pair_mp2.solver.maximum_iterations>128||o.pair_ccsd.solver.maximum_iterations>128
            ||o.domain_pair_mp2.solver.maximum_iterations>128
            ||o.local_triples.solver.maximum_iterations>128
            ||o.triple_spaces.union_eigensolver.max_sweeps>200||o.triple_spaces.density_eigensolver.max_sweeps>200
            ||o.triple_spaces.fock_eigensolver.max_sweeps>200
            ||c.iao_source.maximum_total_image_candidates>250000||c.localization.maximum_total_image_candidates>2000000)
            throw std::length_error("Gaussian selected-local diagnostic exceeds tiny state/rank/iteration/source bounds");
        auto p=plan_periodic_gaussian_selected_local_ccsd_t(hf,reference,view,o,l,c);
        // The physical PH producer reserves each actual all-q target's work
        // across the full iteration cap. Keep the same tiny memory/shape
        // limits; only this explicitly selected work ceiling is larger.
        const std::uint64_t work_limit = o.physical_particle_hole_ccsd
            ? 10000000000000000000ULL : 1000000000000000000ULL;
        if (p.owned_numerical_upper_bound>(16U<<20)||p.node_inventoried_bytes_upper_bound>(128U<<20)
            ||p.work_units_upper_bound>work_limit||p.progress_callback_upper_bound>1000000)
            throw std::length_error("Gaussian selected-local diagnostic exceeds tiny memory/work bounds");
        return p;
    };
    m.def("_plan_periodic_gaussian_selected_local_ccsd_t",[tiny_plan](const HF& hf,const Reference& reference,
        py::array domain,py::array occupied,std::uint64_t translation,Options o,Live l,Caps c) {
        return tiny_plan(hf,reference,domain,occupied,translation,o,l,c);
    },py::arg("hf"),py::arg("reference"),py::arg("domain").noconvert(),py::arg("occupied").noconvert(),
      py::arg("virtual_translation_cell"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_run_periodic_gaussian_selected_local_ccsd_t",[selection,tiny_plan](const HF& hf,const Reference& reference,
        const BasisSet& ao,const BasisSet& auxiliary,const BasisSet& minimal,const PeriodicSystem& system,
        py::array domain,py::array occupied,std::uint64_t translation,Options o,Live l,Caps c,py::object callback) {
        if (ao.nbasis()>4||auxiliary.nbasis()>4||minimal.nbasis()>4||system.unit_cell.size()>4)
            throw std::length_error("Gaussian selected-local diagnostic exceeds tiny actual basis/atom bounds");
        tiny_plan(hf,reference,domain,occupied,translation,o,l,c);
        if (!callback.is_none()&&!PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian selected-local progress must be callable or None");
        const auto labels=selection(domain,occupied,translation);
        struct Callback {
            py::object& function;
            const py::array& domain;
            const py::array& occupied;
            PeriodicGaussianSelectedLocalSelection labels;
            static void notify(const Progress& event,void* opaque) {
                auto& self=*static_cast<Callback*>(opaque);
                self.function(py::cast(Progress(event),py::return_value_policy::move));
                // resize(refcheck=False) can invalidate a pinned ndarray's
                // original data pointer. Reject descriptors BEFORE the C++
                // physical-input checker dereferences its borrowed labels.
                const auto check=[](const py::array& a,const std::uint64_t* pointer,std::size_t count) {
                    if (!a.dtype().is(py::dtype::of<std::uint64_t>()) || !(a.flags()&py::array::c_style)
                        || a.ndim()!=2 || a.shape(0)!=static_cast<py::ssize_t>(count) || a.shape(1)!=2
                        || a.strides(0)!=16 || a.strides(1)!=8 || a.data()!=pointer)
                        throw std::invalid_argument("Gaussian selected-local progress changed borrowed selection descriptors");
                };
                check(self.domain,self.labels.domain_indices,self.labels.domain_count);
                check(self.occupied,self.labels.occupied_indices,self.labels.occupied_count);
            }
        };
        Callback progress{callback,domain,occupied,labels};
        // Hold the GIL and every actual borrowed owner. Native callback
        // boundaries rehash original physical data and selection payloads;
        // controls were copied on entry, no Fock/ERI/amplitude callback exists.
        return run_periodic_gaussian_selected_local_ccsd_t(hf,reference,ao,auxiliary,minimal,system,
            labels,o,l,c,callback.is_none()?nullptr:&Callback::notify,
            callback.is_none()?nullptr:&progress);
    },py::arg("hf"),py::arg("reference"),py::arg("ao"),py::arg("auxiliary"),py::arg("minimal"),py::arg("system"),
      py::arg("domain").noconvert(),py::arg("occupied").noconvert(),py::arg("virtual_translation_cell"),
      py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("callback")=py::none());
}
