// Tiny actual-source HF integration diagnostic; the numerical producer is C++.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gaussian_rhf.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_rhf(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianRHFOptions; using Caps=PeriodicGaussianRHFCaps;
    using Plan=PeriodicGaussianRHFPlan; using Diagnostics=PeriodicGaussianRHFDiagnostics;
    using Progress=PeriodicGaussianRHFProgress; using Stage=PeriodicGaussianRHFStage;
    using Result=PeriodicGaussianRHFResult; using Context=PeriodicGaussianSourceContext;
    using Config=PeriodicGaussianFockConfig; using Live=PeriodicGaussianMetricLiveInventory;
    py::class_<Options>(m,"_PeriodicGaussianRHFOptions").def(py::init<>())
        .def_readwrite("one_electron_pair_block",&Options::one_electron_pair_block)
        .def_readwrite("one_electron",&Options::one_electron)
        .def_readwrite("nuclear",&Options::nuclear).def_readwrite("scf",&Options::scf);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianRHFCaps"); caps.def(py::init<>());
#define GRHF_CAP(field) caps.def_readwrite(#field,&Caps::field)
    GRHF_CAP(one_electron); GRHF_CAP(nuclear); GRHF_CAP(fock); GRHF_CAP(scf);
    GRHF_CAP(maximum_owned_numeric_bytes); GRHF_CAP(maximum_per_replica_inventoried_bytes);
    GRHF_CAP(maximum_node_inventoried_bytes); GRHF_CAP(maximum_state_numeric_bytes);
    GRHF_CAP(maximum_one_electron_panel_calls); GRHF_CAP(maximum_progress_callbacks); GRHF_CAP(maximum_work_units);
#undef GRHF_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianRHFPlan");
#define GRHF_PLAN(field) plan.def_readonly(#field,&Plan::field)
    GRHF_PLAN(n_kpoints); GRHF_PLAN(n_basis); GRHF_PLAN(electrons_per_cell); GRHF_PLAN(atom_count);
    GRHF_PLAN(frozen_selection_bytes); GRHF_PLAN(overlap_hcore_bytes); GRHF_PLAN(state_bytes_upper_bound);
    GRHF_PLAN(state_validation_workspace_reservation_bytes); GRHF_PLAN(one_electron_panel_calls);
    GRHF_PLAN(one_electron_owned_upper_bound); GRHF_PLAN(scf_owned_upper_bound); GRHF_PLAN(capture_owned_upper_bound);
    GRHF_PLAN(owned_numeric_upper_bound); GRHF_PLAN(borrowed_input_numeric_bytes);
    GRHF_PLAN(control_storage_reservation_bytes); GRHF_PLAN(per_replica_inventoried_bytes); GRHF_PLAN(node_inventoried_bytes);
    GRHF_PLAN(progress_callback_upper_bound); GRHF_PLAN(input_check_work_units); GRHF_PLAN(work_units_upper_bound);
#undef GRHF_PLAN
    plan.def_property_readonly("fock",[](const Plan& p){return p.fock;})
        .def_property_readonly("scf",[](const Plan& p){return p.scf;});
    auto diagnostic=py::class_<Diagnostics>(m,"_PeriodicGaussianRHFDiagnostics");
#define GRHF_DIAG(field) diagnostic.def_readonly(#field,&Diagnostics::field)
    GRHF_DIAG(completed_panel_calls); GRHF_DIAG(completed_fock_calls); GRHF_DIAG(completed_progress_callbacks);
    GRHF_DIAG(nuclear_energy_per_cell); GRHF_DIAG(captured_energy_per_cell);
    GRHF_DIAG(maximum_capture_fock_hermiticity_defect); GRHF_DIAG(maximum_capture_fock_change); GRHF_DIAG(capture_energy_change);
    GRHF_DIAG(capture_commutator_frobenius_rms); GRHF_DIAG(maximum_capture_projected_eigen_relative_residual);
    GRHF_DIAG(maximum_capture_coefficient_metric_error);
#undef GRHF_DIAG
    diagnostic.def_property_readonly("last_scf_snapshot",[](const Diagnostics& d){return d.last_scf_snapshot;})
        .def_property_readonly("scf",[](const Diagnostics& d){return d.scf;});
    py::enum_<Stage>(m,"_PeriodicGaussianRHFStage")
        .value("BEGIN",Stage::Begin).value("NUCLEAR_ENERGY",Stage::NuclearEnergy).value("ONE_ELECTRON",Stage::OneElectron)
        .value("TWO_ELECTRON",Stage::TwoElectron).value("SCF",Stage::SCF).value("CAPTURE_REBUILD",Stage::CaptureRebuild)
        .value("CAPTURED",Stage::Captured).value("FINISHED",Stage::Finished);
    py::class_<Progress>(m,"_PeriodicGaussianRHFProgress")
        .def_readonly("stage",&Progress::stage).def_readonly("k_index",&Progress::k_index)
        .def_readonly("pair_begin",&Progress::pair_begin).def_readonly("completed_panel_calls",&Progress::completed_panel_calls)
        .def_readonly("fock_call",&Progress::fock_call)
        .def_property_readonly("fock",[](const Progress& p){return p.fock;})
        .def_property_readonly("scf",[](const Progress& p){return p.scf;});
    py::class_<Result>(m,"_PeriodicGaussianRHFResult")
        .def_property_readonly("context",&Result::context_handle)
        .def_property_readonly("state",&Result::state_handle)
        .def_property_readonly("unfinished",&Result::unconverged_numerical_result,py::return_value_policy::reference_internal)
        .def_property_readonly("converged",&Result::converged)
        .def_property_readonly("matched_finite_gaussian_hf_source",&Result::matched_finite_gaussian_hf_source)
        .def_property_readonly("infinite_source_accuracy_certified",&Result::infinite_source_accuracy_certified)
        .def_property_readonly("plan",[](const Result& r){return r.plan();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("original_input_identity_sha256",&Result::original_input_identity_sha256)
        .def_property_readonly("one_electron_source_identity_sha256",&Result::one_electron_source_identity_sha256)
        .def_property_readonly("final_fock_source_identity_sha256",&Result::final_fock_source_identity_sha256)
        .def_property_readonly("reference_source_identity_sha256",&Result::reference_source_identity_sha256)
        .def("verify_physical_inputs",&Result::verify_physical_inputs);
    const auto selection=[](const Context& context,const PeriodicSystem& system,const py::array& mask,const Options& o) {
        if (context.mesh().size()>8 || context.inventory().ao.function_count>4
            || context.inventory().auxiliary.function_count>4 || system.unit_cell.size()>4
            || o.scf.maximum_iterations>64 || o.scf.jacobi_max_sweeps>64)
            throw std::length_error("Gaussian RHF diagnostic exceeds tiny shape/iteration bounds");
        if (!mask.dtype().is(py::dtype::of<std::uint8_t>()) || !(mask.flags()&py::array::c_style)
            || mask.ndim()!=2 || mask.shape(0)!=static_cast<py::ssize_t>(context.mesh().size())
            || mask.shape(1)<1 || mask.shape(1)>=static_cast<py::ssize_t>(context.inventory().ao.function_count))
            throw std::invalid_argument("Gaussian RHF diagnostic requires contiguous uint8 [Nk,nocc] frozen selection");
        return PeriodicGaussianRHFFrozenSelection{static_cast<const std::uint8_t*>(mask.data()),static_cast<std::size_t>(mask.size())};
    };
    const auto tiny_plan=[selection](const Context& c,const PeriodicSystem& s,const py::array& mask,
                                    const Options& o,const Config& f,const Live& l,const Caps& caps) {
        const auto view=selection(c,s,mask,o);
        const auto p=plan_periodic_gaussian_rhf(c,s,view,o,f,l,caps);
        if (p.owned_numeric_upper_bound>(16U<<20) || p.node_inventoried_bytes>(128U<<20)
            || p.fock.total_tile_calls>4096 || p.work_units_upper_bound>1000000000000000000ULL
            || f.source_caps.maximum_candidates_per_source>65536 || f.tile_caps.maximum_image_candidates>65536
            || caps.nuclear.maximum_reciprocal_candidates>65536 || caps.nuclear.maximum_atom_count>4)
            throw std::length_error("Gaussian RHF diagnostic exceeds tiny memory/source/work bounds");
        return p;
    };
    m.def("_plan_periodic_gaussian_rhf",[tiny_plan](const Context& c,const PeriodicSystem& s,py::array mask,
        Options o,Config f,Live l,Caps caps){return tiny_plan(c,s,mask,o,f,l,caps);},
        py::arg("context"),py::arg("system"),py::arg("frozen_selection").noconvert(),py::arg("options"),
        py::arg("fock_config"),py::arg("live"),py::arg("caps"));
    m.def("_run_periodic_gaussian_rhf",[tiny_plan,selection](std::shared_ptr<const Context> c,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicSystem& s,py::array mask,
        Options o,Config f,Live l,Caps caps,py::object callback) {
        if (!c) throw std::invalid_argument("Gaussian RHF requires a native source context");
        tiny_plan(*c,s,mask,o,f,l,caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian RHF progress must be callable or None");
        const auto notify=[](const Progress& p,void* opaque) {
            (*static_cast<py::object*>(opaque))(py::cast(Progress(p),py::return_value_policy::move));
        };
        // Tiny diagnostic holds the GIL. Mask/bases/system are pinned borrowed
        // inputs; root rechecks bounded actual content after every callback.
        // No Fock/matrix callback, supplied S/H, or converged flag is accepted.
        return run_periodic_gaussian_rhf(c,ao,auxiliary,s,selection(*c,s,mask,o),o,f,l,caps,
            callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:&callback);
    },py::arg("context"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("system"),
      py::arg("frozen_selection").noconvert(),py::arg("options"),py::arg("fock_config"),py::arg("live"),py::arg("caps"),
      py::arg("progress")=py::none());
}
