// Included by bindings.cpp. Actual-source tiny diagnostic, no G/T/F arrays.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_pair_pnos.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

void bind_periodic_gaussian_pair_pnos(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianPairPNOOptions;
    using Live=PeriodicGaussianPairPNOLiveInventory;
    using Caps=PeriodicGaussianPairPNOCaps;
    using Plan=PeriodicGaussianPairPNOPlan;
    using Diagnostics=PeriodicGaussianPairPNODiagnostics;
    using Result=PeriodicGaussianPairPNOResult;
    py::class_<Options>(m,"_PeriodicGaussianPairPNOOptions").def(py::init<>())
        .def_readwrite("occupation_cutoff",&Options::occupation_cutoff)
        .def_readwrite("denominator_floor",&Options::denominator_floor)
        .def_readwrite("maximum_initial_fvv_offdiagonal_norm",&Options::maximum_initial_fvv_offdiagonal_norm)
        .def_readwrite("semicanonical_orthonormality_tolerance",&Options::semicanonical_orthonormality_tolerance)
        .def_property("pno_max_sweeps",[](const Options& o){return o.pno_eigensolver.max_sweeps;},
            [](Options& o,std::uint64_t v){o.pno_eigensolver.max_sweeps=v;})
        .def_property("pno_relative_eigensolver_tolerance",[](const Options& o){return o.pno_eigensolver.relative_offdiagonal_tolerance;},
            [](Options& o,double v){o.pno_eigensolver.relative_offdiagonal_tolerance=v;})
        .def_property("semicanonical_max_sweeps",[](const Options& o){return o.semicanonical_eigensolver.max_sweeps;},
            [](Options& o,std::uint64_t v){o.semicanonical_eigensolver.max_sweeps=v;})
        .def_property("semicanonical_relative_eigensolver_tolerance",[](const Options& o){return o.semicanonical_eigensolver.relative_offdiagonal_tolerance;},
            [](Options& o,double v){o.semicanonical_eigensolver.relative_offdiagonal_tolerance=v;});
    py::class_<Live>(m,"_PeriodicGaussianPairPNOLiveInventory").def(py::init<>())
        .def_readwrite("other_live_bytes_per_worker",&Live::other_live_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
    py::class_<Caps>(m,"_PeriodicGaussianPairPNOCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes",&Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_per_worker_inventoried_bytes",&Caps::maximum_per_worker_inventoried_bytes)
        .def_readwrite("maximum_node_inventoried_bytes",&Caps::maximum_node_inventoried_bytes)
        .def_readwrite("maximum_integral_calls",&Caps::maximum_integral_calls)
        .def_readwrite("maximum_work_units",&Caps::maximum_work_units);
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianPairPNOPlan");
#define GPAIR_PLAN(field) plan.def_readonly(#field,&Plan::field)
    GPAIR_PLAN(occupied_count); GPAIR_PLAN(virtual_count); GPAIR_PLAN(occupied_slot_i); GPAIR_PLAN(occupied_slot_j);
    GPAIR_PLAN(integral_calls); GPAIR_PLAN(provider_work_units); GPAIR_PLAN(numerical_work_units); GPAIR_PLAN(work_units);
    GPAIR_PLAN(integral_amplitude_phase_bytes); GPAIR_PLAN(density_phase_bytes); GPAIR_PLAN(semicanonical_phase_upper_bytes);
    GPAIR_PLAN(peak_owned_numerical_bytes); GPAIR_PLAN(retained_output_upper_bytes); GPAIR_PLAN(borrowed_basis_bytes);
    GPAIR_PLAN(borrowed_provider_row_bytes); GPAIR_PLAN(state_resident_bytes); GPAIR_PLAN(fixed_control_storage_bytes);
    GPAIR_PLAN(replicas_per_node); GPAIR_PLAN(reference_base_node_bytes); GPAIR_PLAN(per_worker_inventoried_bytes);
    GPAIR_PLAN(required_node_memory_bytes);
#undef GPAIR_PLAN
    plan.def_property_readonly("options",[](const Plan& p){return p.options;})
        .def_property_readonly("live",[](const Plan& p){return p.live;})
        .def_property_readonly("caps",[](const Plan& p){return p.caps;})
        .def_property_readonly("plan_identity_sha256",&Plan::plan_identity_sha256);
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianPairPNODiagnostics");
#define GPAIR_DIAG(field) diagnostics.def_readonly(#field,&Diagnostics::field)
    GPAIR_DIAG(completed_integral_calls); GPAIR_DIAG(retained_dimension); GPAIR_DIAG(retained_output_bytes);
    GPAIR_DIAG(actual_semicanonical_phase_bytes); GPAIR_DIAG(initial_fvv_offdiagonal_norm_upper_bound);
    GPAIR_DIAG(ignored_occupied_offdiagonal_norm_upper_bound); GPAIR_DIAG(maximum_scalar_integral_roundoff_error);
    GPAIR_DIAG(integral_projection_error_bound); GPAIR_DIAG(minimum_denominator); GPAIR_DIAG(maximum_denominator);
    GPAIR_DIAG(maximum_absolute_initial_amplitude); GPAIR_DIAG(maximum_initial_residual);
    GPAIR_DIAG(initial_residual_frobenius_norm); GPAIR_DIAG(maximum_diagonal_integral_asymmetry);
    GPAIR_DIAG(maximum_diagonal_amplitude_asymmetry); GPAIR_DIAG(density_trace); GPAIR_DIAG(discarded_occupation_sum);
    GPAIR_DIAG(minimum_occupation); GPAIR_DIAG(negative_occupation_count); GPAIR_DIAG(amplitude_scaling_underflow_count);
    GPAIR_DIAG(density_underflow_entry_count); GPAIR_DIAG(semicanonical_fock_scaling_underflow_count);
    GPAIR_DIAG(density_eigensystem_relative_residual); GPAIR_DIAG(density_eigenvector_orthogonality_error);
    GPAIR_DIAG(semicanonical_input_orthonormality_error); GPAIR_DIAG(semicanonical_output_orthonormality_error);
    GPAIR_DIAG(semicanonical_reduced_relative_residual); GPAIR_DIAG(semicanonical_projected_fock_relative_residual);
    GPAIR_DIAG(semicanonical_subspace_projector_error); GPAIR_DIAG(semicanonical_full_space_relative_residual);
#undef GPAIR_DIAG
    diagnostics.def_property_readonly("pno_eigensolver_sweeps",[](const Diagnostics& d){return d.pno_eigensolver.sweeps;})
        .def_property_readonly("pno_eigensolver_rotations",[](const Diagnostics& d){return d.pno_eigensolver.rotations;})
        .def_property_readonly("semicanonical_eigensolver_sweeps",[](const Diagnostics& d){return d.semicanonical_eigensolver.sweeps;})
        .def_property_readonly("semicanonical_eigensolver_rotations",[](const Diagnostics& d){return d.semicanonical_eigensolver.rotations;});
    py::class_<Result>(m,"_PeriodicGaussianPairPNOResult")
        .def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("state",&Result::state_handle).def_property_readonly("context",&Result::context_handle)
        .def_property_readonly("occupied_i",[](const Result& r){const auto i=r.occupied_i(); return py::make_tuple(i.occupied_index,i.cell);})
        .def_property_readonly("occupied_j",[](const Result& r){const auto i=r.occupied_j(); return py::make_tuple(i.occupied_index,i.cell);})
        .def_property_readonly("diagonal_pair",&Result::diagonal_pair)
        .def_property_readonly("semicanonical_generation_approximation",&Result::semicanonical_generation_approximation)
        .def_property_readonly("coupled_mp2_solution",&Result::coupled_mp2_solution)
        .def_property_readonly("production_dlpno",&Result::production_dlpno)
        .def_property_readonly("matched_finite_gaussian_hf_recipe",&Result::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("infinite_source_accuracy_certified",&Result::infinite_source_accuracy_certified)
        .def_property_readonly("identity_sha256",&Result::identity_sha256)
        .def_property_readonly("payload_sha256",&Result::payload_sha256)
        .def_property_readonly("basis_identity_sha256",&Result::basis_identity_sha256)
        .def_property_readonly("provider_identity_sha256",&Result::provider_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256",&Result::hf_reference_source_identity_sha256)
        .def_property_readonly("exchange_integral_identity_sha256",&Result::exchange_integral_identity_sha256)
        .def_property_readonly("initial_amplitude_identity_sha256",&Result::initial_amplitude_identity_sha256)
        .def_property_readonly("density_identity_sha256",&Result::density_identity_sha256)
        .def("coefficients_copy",[](const Result& r){
            const auto& values=r.coefficients(); const auto n=r.memory().virtual_count,v=r.diagnostics().retained_dimension;
            if(n>8 || v>8) throw std::length_error("Gaussian pair PNO copy exceeds tiny diagnostic dimension");
            py::array_t<double> a({static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(v)});
            if(!values.empty()) std::memcpy(a.mutable_data(),values.data(),values.size()*8); return a;
        })
        .def("energies_copy",[](const Result& r){
            const auto& values=r.energies();
            if(values.size()>8) throw std::length_error("Gaussian pair PNO energy copy exceeds tiny dimension");
            py::array_t<double> a(static_cast<py::ssize_t>(values.size()));
            if(!values.empty()) std::memcpy(a.mutable_data(),values.data(),values.size()*8); return a;
        })
        .def("original_pno_occupations_copy",[](const Result& r){
            const auto& values=r.original_pno_occupations();
            if(values.size()>8) throw std::length_error("Gaussian pair PNO occupation copy exceeds tiny dimension");
            py::array_t<double> a(static_cast<py::ssize_t>(values.size()));
            std::memcpy(a.mutable_data(),values.data(),values.size()*8); return a;
        });
    const auto tiny=[](const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const Options& o){
        if(ref.state().n_kpoints()>8 || basis.memory().occupied_count>8 || basis.memory().virtual_count>8
            || o.pno_eigensolver.max_sweeps>200 || o.semicanonical_eigensolver.max_sweeps>200)
            throw std::length_error("Gaussian pair PNO diagnostic exceeds tiny dimensions or sweeps");
    };
    const auto bounded=[](const Plan& p){
        if(p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units>100000000000000ULL)
            throw std::length_error("Gaussian pair PNO diagnostic exceeds tiny memory or work");
    };
    m.def("_plan_periodic_gaussian_pair_pnos",[tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        std::uint64_t i,std::uint64_t j,Options options,Live live,Caps caps){
        tiny(ref,basis,options); auto p=plan_periodic_gaussian_pair_pnos(ref,basis,provider,i,j,options,live,caps);
        bounded(p); return p;
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("occupied_slot_i"),py::arg("occupied_slot_j"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_pnos",[tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        std::uint64_t i,std::uint64_t j,Options options,Live live,Caps caps){
        tiny(ref,basis,options); bounded(plan_periodic_gaussian_pair_pnos(ref,basis,provider,i,j,options,live,caps));
        // Native owned inputs only, no hidden NumPy cast or callback. Hold GIL.
        return make_periodic_gaussian_pair_pnos(ref,basis,provider,i,j,options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("occupied_slot_i"),py::arg("occupied_slot_j"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
}
