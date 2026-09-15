// Included by bindings.cpp. Tiny native-owner integration diagnostics only.
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_correlation_real_local_ccsd_t.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_real_local_ccsd_t(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationRealLocalCCSDTOptions;
    using Inventory = PeriodicCorrelationRealLocalCCSDTInventory;
    using Caps = PeriodicCorrelationRealLocalCCSDTCaps;
    using Plan = PeriodicCorrelationRealLocalCCSDTMemoryPlan;
    using Result = PeriodicCorrelationRealLocalCCSDTResult;
    using Progress = PeriodicCorrelationRealLocalCCSDTProgress;
    using Stage = PeriodicCorrelationRealLocalCCSDTStage;
    py::class_<Options>(m,"_PeriodicCorrelationRealLocalCCSDTOptions").def(py::init<>())
        .def_readwrite("ccsd",&Options::ccsd).def_readwrite("triples",&Options::triples)
        .def_readwrite("maximum_additional_fock_projection_norm",&Options::maximum_additional_fock_projection_norm);
    py::class_<Inventory>(m,"_PeriodicCorrelationRealLocalCCSDTInventory").def(py::init<>())
        .def_readwrite("other_live_numerical_bytes_per_replica",&Inventory::other_live_numerical_bytes_per_replica);
    py::class_<Caps>(m,"_PeriodicCorrelationRealLocalCCSDTCaps").def(py::init<>())
#define REAL_CCSDT_CAP(field) .def_readwrite(#field,&Caps::field)
        REAL_CCSDT_CAP(maximum_owned_numerical_bytes) REAL_CCSDT_CAP(maximum_total_numerical_bytes)
        REAL_CCSDT_CAP(maximum_node_numerical_bytes) REAL_CCSDT_CAP(maximum_integral_calls) REAL_CCSDT_CAP(maximum_work_units);
#undef REAL_CCSDT_CAP
    py::class_<Plan>(m,"_PeriodicCorrelationRealLocalCCSDTMemoryPlan")
#define REAL_CCSDT_PLAN(field) .def_readonly(#field,&Plan::field)
        REAL_CCSDT_PLAN(occupied_count) REAL_CCSDT_PLAN(virtual_count) REAL_CCSDT_PLAN(projected_fock_bytes)
        REAL_CCSDT_PLAN(borrowed_basis_bytes) REAL_CCSDT_PLAN(borrowed_factor_row_bytes)
        REAL_CCSDT_PLAN(other_live_numerical_bytes_per_replica) REAL_CCSDT_PLAN(peak_owned_numerical_bytes)
        REAL_CCSDT_PLAN(retained_output_bytes_upper_bound) REAL_CCSDT_PLAN(total_live_numerical_bytes)
        REAL_CCSDT_PLAN(required_node_numerical_bytes) REAL_CCSDT_PLAN(numerical_replicas)
        REAL_CCSDT_PLAN(external_node_numerical_bytes) REAL_CCSDT_PLAN(integral_calls_upper_bound)
        REAL_CCSDT_PLAN(provider_work_units_upper_bound) REAL_CCSDT_PLAN(work_units_upper_bound)
        REAL_CCSDT_PLAN(ccsd) REAL_CCSDT_PLAN(triples);
#undef REAL_CCSDT_PLAN
    py::enum_<Stage>(m,"_PeriodicCorrelationRealLocalCCSDTStage").value("CCSD",Stage::CCSD).value("TRIPLES",Stage::Triples);
    py::class_<Progress>(m,"_PeriodicCorrelationRealLocalCCSDTProgress")
        .def_readonly("stage",&Progress::stage).def_readonly("ccsd",&Progress::ccsd).def_readonly("triples",&Progress::triples);
    py::class_<Result>(m,"_PeriodicCorrelationRealLocalCCSDTResult")
        .def_property_readonly("memory",&Result::memory,py::return_value_policy::reference_internal)
        .def_property_readonly("ccsd",&Result::ccsd,py::return_value_policy::reference_internal)
        .def_property_readonly("triples",&Result::triples,py::return_value_policy::reference_internal)
#define REAL_CCSDT_RESULT(field) .def_property_readonly(#field,&Result::field)
        REAL_CCSDT_RESULT(triples_evaluated) REAL_CCSDT_RESULT(converged)
        REAL_CCSDT_RESULT(hf_hamiltonian_match_certified) REAL_CCSDT_RESULT(periodic_energy_per_cell)
        REAL_CCSDT_RESULT(additional_fock_projection_norm_bound) REAL_CCSDT_RESULT(total_fock_projection_norm_bound)
        REAL_CCSDT_RESULT(identity_sha256) REAL_CCSDT_RESULT(payload_sha256)
        REAL_CCSDT_RESULT(provider_identity_sha256) REAL_CCSDT_RESULT(basis_identity_sha256);
#undef REAL_CCSDT_RESULT
    m.def("_plan_periodic_correlation_real_local_ccsd_t",&plan_periodic_correlation_real_local_ccsd_t,
        py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("options"),py::arg("inventory"));
    m.def("_solve_periodic_correlation_real_local_ccsd_t",[](
        const PeriodicCorrelationAdmittedReference& reference,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicCorrelationRealLocalProvider& provider,Options options,Inventory inventory,Caps caps,
        const py::object& progress) {
        // Input owners are immutable; copied controls cannot be mutated by a
        // callback during the released-GIL calculation. Numerical output array
        // copies inherit the existing tiny common-space solver restrictions.
        if (basis.memory().occupied_count>3 || basis.memory().virtual_count>4 || provider.memory().row_count>128
            || options.ccsd.maximum_iterations>256 || options.triples.maximum_iterations>256
            || reference.budget().mpi_ranks>8 || reference.budget().workers_per_rank>8
            || caps.maximum_owned_numerical_bytes>(1U<<20) || caps.maximum_total_numerical_bytes>(2U<<20)
            || caps.maximum_node_numerical_bytes>(8U<<20) || caps.maximum_integral_calls>100000000ULL
            || caps.maximum_work_units>20000000000ULL || inventory.other_live_numerical_bytes_per_replica>(1U<<20))
            throw std::length_error("real-local CCSD(T) diagnostic exceeds tiny integration limits");
        if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            throw std::invalid_argument("real-local CCSD(T) progress must be callable or absent");
        struct Sink {
            const py::object* callable;
            static void receive(const Progress& event,void* opaque) {
                const auto& self=*static_cast<const Sink*>(opaque);
                py::gil_scoped_acquire acquire;
                (*self.callable)(py::cast(event,py::return_value_policy::copy));
            }
        } sink{&progress};
        const auto callback=progress.is_none()?nullptr:&Sink::receive;
        py::gil_scoped_release release;
        return solve_periodic_correlation_real_local_ccsd_t(reference,basis,provider,options,inventory,caps,callback,&sink);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("options"),py::arg("inventory"),py::arg("caps"),
      py::arg("progress")=py::none());
}
