// Tiny linear-response RHF diagnostics; no physical-source or MF factory.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include "vibeqc/bounded_periodic_rhf_solver.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_periodic_rhf_solver(py::module_& m) {
    using namespace vibeqc;
    using Options=BoundedPeriodicRHFOptions;
    using Caps=BoundedPeriodicRHFCaps;
    using Memory=BoundedPeriodicRHFMemoryPlan;
    using Diagnostics=BoundedPeriodicRHFDiagnostics;
    using Snapshot=BoundedPeriodicRHFSnapshot;
    using Result=BoundedPeriodicRHFResult;
    using Status=BoundedPeriodicRHFStatus;
    using Z=std::complex<double>;
    auto options=py::class_<Options>(m,"_BoundedPeriodicRHFOptions"); options.def(py::init<>());
#define BRHF_OPTION(field) options.def_readwrite(#field,&Options::field)
    BRHF_OPTION(maximum_iterations); BRHF_OPTION(maximum_diis_history); BRHF_OPTION(jacobi_max_sweeps);
    BRHF_OPTION(jacobi_relative_tolerance); BRHF_OPTION(overlap_rank_absolute_floor); BRHF_OPTION(overlap_rank_relative_floor);
    BRHF_OPTION(overlap_negative_absolute_tolerance); BRHF_OPTION(overlap_negative_relative_tolerance);
    BRHF_OPTION(hermitian_absolute_tolerance); BRHF_OPTION(hermitian_relative_tolerance);
    BRHF_OPTION(maximum_overlap_projection_error); BRHF_OPTION(maximum_hcore_projection_error); BRHF_OPTION(maximum_fock_projection_error);
    BRHF_OPTION(algebra_absolute_tolerance); BRHF_OPTION(algebra_relative_tolerance); BRHF_OPTION(eigen_relative_tolerance);
    BRHF_OPTION(commutator_tolerance); BRHF_OPTION(density_closure_absolute_tolerance); BRHF_OPTION(density_closure_relative_tolerance);
    BRHF_OPTION(energy_change_tolerance); BRHF_OPTION(minimum_band_gap_hartree);
#undef BRHF_OPTION
    auto caps=py::class_<Caps>(m,"_BoundedPeriodicRHFCaps"); caps.def(py::init<>());
#define BRHF_CAP(field) caps.def_readwrite(#field,&Caps::field)
    BRHF_CAP(maximum_owned_numerical_bytes); BRHF_CAP(maximum_total_numerical_bytes);
    BRHF_CAP(maximum_work_units); BRHF_CAP(maximum_fock_calls); BRHF_CAP(other_live_numerical_bytes);
#undef BRHF_CAP
    auto memory=py::class_<Memory>(m,"_BoundedPeriodicRHFMemoryPlan");
#define BRHF_MEMORY(field) memory.def_readonly(#field,&Memory::field)
    BRHF_MEMORY(n_kpoints); BRHF_MEMORY(n_basis); BRHF_MEMORY(n_occupied); BRHF_MEMORY(retained_rank);
    BRHF_MEMORY(overlap_discovery_owned_bytes); BRHF_MEMORY(orthogonalizer_bytes); BRHF_MEMORY(density_bytes);
    BRHF_MEMORY(fock_bytes); BRHF_MEMORY(coefficient_bytes); BRHF_MEMORY(orbital_energy_bytes);
    BRHF_MEMORY(jacobi_workspace_bytes); BRHF_MEMORY(contraction_workspace_bytes); BRHF_MEMORY(peak_owned_numerical_bytes);
    BRHF_MEMORY(output_numerical_bytes); BRHF_MEMORY(borrowed_input_bytes); BRHF_MEMORY(provider_retained_bytes);
    BRHF_MEMORY(provider_workspace_bytes); BRHF_MEMORY(other_live_numerical_bytes); BRHF_MEMORY(total_numerical_bytes);
    BRHF_MEMORY(input_validation_work_units); BRHF_MEMORY(overlap_discovery_work_units);
    BRHF_MEMORY(orthogonalizer_work_units); BRHF_MEMORY(initial_density_work_units);
    BRHF_MEMORY(work_units_per_evaluated_iteration); BRHF_MEMORY(maximum_work_units);
#undef BRHF_MEMORY
    auto diagnostics=py::class_<Diagnostics>(m,"_BoundedPeriodicRHFDiagnostics");
#define BRHF_DIAGNOSTIC(field) diagnostics.def_readonly(#field,&Diagnostics::field)
    BRHF_DIAGNOSTIC(jacobi_sweeps); BRHF_DIAGNOSTIC(maximum_raw_overlap_hermitian_defect);
    BRHF_DIAGNOSTIC(maximum_raw_hcore_hermitian_defect); BRHF_DIAGNOSTIC(maximum_raw_fock_hermitian_defect);
    BRHF_DIAGNOSTIC(maximum_overlap_projection_frobenius); BRHF_DIAGNOSTIC(maximum_hcore_projection_frobenius);
    BRHF_DIAGNOSTIC(maximum_fock_projection_frobenius); BRHF_DIAGNOSTIC(maximum_reduced_operator_hermitian_defect);
    BRHF_DIAGNOSTIC(maximum_reduced_operator_projection_frobenius); BRHF_DIAGNOSTIC(minimum_overlap_eigenvalue);
    BRHF_DIAGNOSTIC(minimum_retained_overlap_eigenvalue); BRHF_DIAGNOSTIC(maximum_overlap_eigen_relative_residual);
    BRHF_DIAGNOSTIC(maximum_orthogonalizer_metric_error);
#undef BRHF_DIAGNOSTIC
    py::enum_<Status>(m,"_BoundedPeriodicRHFStatus")
        .value("CONVERGED",Status::Converged).value("ITERATION_LIMIT",Status::IterationLimit)
        .value("WORK_LIMIT",Status::WorkLimit).value("FOCK_CALL_LIMIT",Status::FockCallLimit)
        .value("CANCELLED",Status::Cancelled).value("NON_INSULATING",Status::NonInsulating);
    auto snapshot=py::class_<Snapshot>(m,"_BoundedPeriodicRHFSnapshot");
#define BRHF_SNAPSHOT(field) snapshot.def_readonly(#field,&Snapshot::field)
    BRHF_SNAPSHOT(status); BRHF_SNAPSHOT(evaluated_iteration); BRHF_SNAPSHOT(fock_calls); BRHF_SNAPSHOT(charged_work_units);
    BRHF_SNAPSHOT(converged); BRHF_SNAPSHOT(has_energy_change); BRHF_SNAPSHOT(energy_per_cell); BRHF_SNAPSHOT(electronic_energy_per_cell);
    BRHF_SNAPSHOT(energy_change); BRHF_SNAPSHOT(raw_energy_per_cell); BRHF_SNAPSHOT(energy_projection_change);
    BRHF_SNAPSHOT(commutator_frobenius_rms); BRHF_SNAPSHOT(maximum_commutator_element);
    BRHF_SNAPSHOT(maximum_density_closure_frobenius); BRHF_SNAPSHOT(maximum_density_closure_relative);
    BRHF_SNAPSHOT(maximum_metric_idempotency_error); BRHF_SNAPSHOT(maximum_electron_count_error);
    BRHF_SNAPSHOT(maximum_coefficient_metric_error); BRHF_SNAPSHOT(maximum_projected_eigen_relative_residual);
    BRHF_SNAPSHOT(maximum_full_ao_eigen_residual); BRHF_SNAPSHOT(global_homo); BRHF_SNAPSHOT(global_lumo); BRHF_SNAPSHOT(global_band_gap);
#undef BRHF_SNAPSHOT
    const auto tiny_result=[](const Result& r){
        if (r.memory().n_kpoints>8 || r.memory().n_basis>4 || r.memory().retained_rank>4)
            throw std::length_error("bounded RHF diagnostic output copy exceeds tiny dimensions");
    };
    py::class_<Result>(m,"_BoundedPeriodicRHFResult")
        .def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("final_snapshot",[](const Result& r){return r.final_snapshot();})
        .def_property_readonly("converged",&Result::converged)
        .def("density_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.density_data();
            py::array_t<Z> a({static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
            std::memcpy(a.mutable_data(),data,p.density_bytes); return a;
        })
        .def("fock_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.fock_data();
            py::array_t<Z> a({static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
            std::memcpy(a.mutable_data(),data,p.fock_bytes); return a;
        })
        .def("coefficients_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.coefficients_data();
            py::array_t<Z> a({static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.retained_rank)});
            std::memcpy(a.mutable_data(),data,p.coefficient_bytes); return a;
        })
        .def("orbital_energies_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.orbital_energies_data();
            py::array_t<double> a({static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.retained_rank)});
            std::memcpy(a.mutable_data(),data,p.orbital_energy_bytes); return a;
        });
    m.def("_plan_bounded_periodic_rhf_linear_model",[](
        std::uint64_t k,std::uint64_t n,std::uint64_t electrons,std::uint64_t rank,
        Options o,std::uint64_t other){
        if (!k || k>8 || n<2 || n>4 || o.maximum_iterations>128 || o.jacobi_max_sweeps>64 || other>(8U<<20))
            throw std::length_error("bounded RHF diagnostic exceeds tiny shape/iteration caps");
        const auto noop=[](const Z*,std::size_t,Z*,std::size_t,void*){};
        const std::uint64_t count=k*n*n;
        const BoundedPeriodicRHFTwoElectronProvider p{+noop,nullptr,16*(count*count+count),0,128*(count*count+count+1)};
        return plan_bounded_periodic_rhf_solver(k,n,electrons,rank,p,o,other);
    },py::arg("n_kpoints"),py::arg("n_basis"),py::arg("electrons"),py::arg("retained_rank"),
      py::arg("options"),py::arg("other_live_numerical_bytes")=0);
    m.def("_solve_bounded_periodic_rhf_linear_model",[](
        const py::array& overlap,const py::array& hcore,const py::array& kernel,const py::array& bias,
        std::uint64_t electrons,double nuclear,Options o,Caps caps,const py::object& callback,
        std::uint64_t fail_on_call,bool omit_last_output){
        const auto valid=[](const py::array& a){return a.dtype().is(py::dtype::of<Z>())
            && (a.flags()&py::array::c_style) && reinterpret_cast<std::uintptr_t>(a.data())%alignof(Z)==0;};
        if (!valid(overlap) || !valid(hcore) || !valid(kernel) || !valid(bias) || overlap.ndim()!=3
            || overlap.shape(0)<1 || overlap.shape(0)>8 || overlap.shape(1)<2 || overlap.shape(1)>4
            || overlap.shape(2)!=overlap.shape(1))
            throw std::invalid_argument("bounded RHF diagnostic requires aligned contiguous complex128 tiny matrix arrays");
        const std::uint64_t k=overlap.shape(0),n=overlap.shape(1),count=k*n*n;
        if (hcore.ndim()!=3 || bias.ndim()!=3 || kernel.ndim()!=2
            || hcore.shape(0)!=static_cast<py::ssize_t>(k) || bias.shape(0)!=static_cast<py::ssize_t>(k)
            || hcore.shape(1)!=static_cast<py::ssize_t>(n) || hcore.shape(2)!=static_cast<py::ssize_t>(n)
            || bias.shape(1)!=static_cast<py::ssize_t>(n) || bias.shape(2)!=static_cast<py::ssize_t>(n)
            || kernel.shape(0)!=static_cast<py::ssize_t>(count) || kernel.shape(1)!=static_cast<py::ssize_t>(count))
            throw std::invalid_argument("bounded RHF diagnostic linear model shapes disagree");
        if (o.maximum_iterations>128 || o.jacobi_max_sweeps>64 || caps.maximum_owned_numerical_bytes>(8U<<20)
            || caps.maximum_total_numerical_bytes>(16U<<20) || caps.other_live_numerical_bytes>(8U<<20)
            || caps.maximum_work_units>10000000000ULL || caps.maximum_fock_calls>128)
            throw std::length_error("bounded RHF diagnostic exceeds tiny memory/work/iteration caps");
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("bounded RHF diagnostic progress must be callable or None");
        struct Model { const Z* kernel; const Z* bias; std::size_t count; std::uint64_t calls,fail; bool omit; };
        Model model{static_cast<const Z*>(kernel.data()),static_cast<const Z*>(bias.data()),static_cast<std::size_t>(count),0,fail_on_call,omit_last_output};
        const auto evaluate=[](const Z* density,std::size_t dn,Z* output,std::size_t on,void* pointer){
            auto& p=*static_cast<Model*>(pointer);
            if (dn!=p.count || on!=p.count) throw std::logic_error("bounded RHF diagnostic callback extent mismatch");
            ++p.calls; if (p.calls==p.fail) throw std::runtime_error("bounded RHF diagnostic provider failure");
            for (std::size_t i=0;i<p.count-(p.omit?1:0);++i) {
                // Independent small-model implementation; native solver has
                // no knowledge of this linear kernel or its numerical form.
                Z value=p.bias[i]; for (std::size_t j=0;j<p.count;++j) value+=p.kernel[i*p.count+j]*density[j];
                output[i]=value;
            }
        };
        BoundedPeriodicRHFInput input{k,n,electrons,nuclear,static_cast<const Z*>(overlap.data()),static_cast<std::size_t>(count),
            static_cast<const Z*>(hcore.data()),static_cast<std::size_t>(count)};
        const BoundedPeriodicRHFTwoElectronProvider provider{+evaluate,&model,16*(count*count+count),0,128*(count*count+count+1)};
        const auto notify=[](const Snapshot& value,void* context){
            const py::object response=(*static_cast<const py::object*>(context))(py::cast(Snapshot(value),py::return_value_policy::move));
            return response.is_none() || response.cast<bool>();
        };
        // No native numerical views cross into callbacks. All matrix inputs
        // remain borrowed immutable call arguments; scalar controls are copied.
        return solve_bounded_periodic_rhf(input,provider,o,caps,callback.is_none()?nullptr:+notify,
            callback.is_none()?nullptr:const_cast<py::object*>(&callback));
    },py::arg("overlap"),py::arg("hcore"),py::arg("kernel"),py::arg("bias"),py::arg("electrons"),
      py::arg("nuclear_energy"),py::arg("options"),py::arg("caps"),py::arg("callback")=py::none(),
      py::arg("fail_on_call")=0,py::arg("omit_last_output")=false);
}
