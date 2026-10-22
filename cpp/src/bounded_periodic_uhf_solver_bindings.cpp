// Tiny linear-response UHF diagnostics; no physical-source or MF factory.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include "vibeqc/bounded_periodic_uhf_solver.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_periodic_uhf_solver(py::module_& m) {
    using namespace vibeqc;
    using Options=BoundedPeriodicUHFOptions;
    using Caps=BoundedPeriodicUHFCaps;
    using Memory=BoundedPeriodicUHFMemoryPlan;
    using Diagnostics=BoundedPeriodicUHFDiagnostics;
    using Snapshot=BoundedPeriodicUHFSnapshot;
    using Result=BoundedPeriodicUHFResult;
    using Status=BoundedPeriodicUHFStatus;
    using Z=std::complex<double>;
    py::class_<Memory,BoundedPeriodicRHFMemoryPlan>(m,"_BoundedPeriodicUHFMemoryPlan")
        .def_readonly("n_alpha",&Memory::n_alpha).def_readonly("n_beta",&Memory::n_beta);
    py::class_<Snapshot,BoundedPeriodicRHFSnapshot>(m,"_BoundedPeriodicUHFSnapshot")
        .def_readonly("spin_homo",&Snapshot::spin_homo)
        .def_readonly("spin_lumo",&Snapshot::spin_lumo)
        .def_readonly("spin_band_gap",&Snapshot::spin_band_gap);
    const auto tiny_result=[](const Result& r){
        if (r.memory().n_kpoints>8 || r.memory().n_basis>4 || r.memory().retained_rank>4)
            throw std::length_error("bounded UHF diagnostic output copy exceeds tiny dimensions");
    };
    py::class_<Result>(m,"_BoundedPeriodicUHFResult")
        .def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("final_snapshot",[](const Result& r){return r.final_snapshot();})
        .def_property_readonly("converged",&Result::converged)
        .def("density_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.density_data();
            py::array_t<Z> a({static_cast<py::ssize_t>(2),static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
            std::memcpy(a.mutable_data(),data,p.density_bytes); return a;
        })
        .def("fock_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.fock_data();
            py::array_t<Z> a({static_cast<py::ssize_t>(2),static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
            std::memcpy(a.mutable_data(),data,p.fock_bytes); return a;
        })
        .def("coefficients_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.coefficients_data();
            py::array_t<Z> a({static_cast<py::ssize_t>(2),static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.retained_rank)});
            std::memcpy(a.mutable_data(),data,p.coefficient_bytes); return a;
        })
        .def("orbital_energies_copy",[tiny_result](const Result& r){
            tiny_result(r);
            const auto& p=r.memory(); const auto* data=r.orbital_energies_data();
            py::array_t<double> a({static_cast<py::ssize_t>(2),static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.retained_rank)});
            std::memcpy(a.mutable_data(),data,p.orbital_energy_bytes); return a;
        });
    m.def("_plan_bounded_periodic_uhf_linear_model",[](
        std::uint64_t k,std::uint64_t n,std::uint64_t alpha,std::uint64_t beta,std::uint64_t rank,
        Options o,std::uint64_t other){
        if (!k || k>8 || n<2 || n>4 || o.maximum_iterations>128 || o.jacobi_max_sweeps>64 || other>(8U<<20))
            throw std::length_error("bounded UHF diagnostic exceeds tiny shape/iteration caps");
        const auto noop=[](const Z*,std::size_t,Z*,std::size_t,void*){};
        const std::uint64_t count=2*k*n*n;
        const BoundedPeriodicUHFTwoElectronProvider p{+noop,nullptr,16*(count*count+count),0,128*(count*count+count+1)};
        return plan_bounded_periodic_uhf_solver(k,n,alpha,beta,rank,p,o,other);
    },py::arg("n_kpoints"),py::arg("n_basis"),py::arg("n_alpha"),py::arg("n_beta"),py::arg("retained_rank"),
      py::arg("options"),py::arg("other_live_numerical_bytes")=0);
    m.def("_solve_bounded_periodic_uhf_linear_model",[](
        const py::array& overlap,const py::array& hcore,const py::array& kernel,const py::array& bias,
        std::uint64_t alpha,std::uint64_t beta,double nuclear,Options o,Caps caps,const py::object& callback,
        std::uint64_t fail_on_call,bool omit_last_output){
        const auto valid=[](const py::array& a){return a.dtype().is(py::dtype::of<Z>())
            && (a.flags()&py::array::c_style) && reinterpret_cast<std::uintptr_t>(a.data())%alignof(Z)==0;};
        if (!valid(overlap) || !valid(hcore) || !valid(kernel) || !valid(bias) || overlap.ndim()!=3
            || overlap.shape(0)<1 || overlap.shape(0)>8 || overlap.shape(1)<2 || overlap.shape(1)>4
            || overlap.shape(2)!=overlap.shape(1))
            throw std::invalid_argument("bounded UHF diagnostic requires aligned contiguous complex128 tiny matrix arrays");
        const std::uint64_t k=overlap.shape(0),n=overlap.shape(1),count=2*k*n*n;
        if (hcore.ndim()!=3 || bias.ndim()!=4 || kernel.ndim()!=2
            || hcore.shape(0)!=static_cast<py::ssize_t>(k) || bias.shape(0)!=2 || bias.shape(1)!=static_cast<py::ssize_t>(k)
            || hcore.shape(1)!=static_cast<py::ssize_t>(n) || hcore.shape(2)!=static_cast<py::ssize_t>(n)
            || bias.shape(2)!=static_cast<py::ssize_t>(n) || bias.shape(3)!=static_cast<py::ssize_t>(n)
            || kernel.shape(0)!=static_cast<py::ssize_t>(count) || kernel.shape(1)!=static_cast<py::ssize_t>(count))
            throw std::invalid_argument("bounded UHF diagnostic linear model shapes disagree");
        if (o.maximum_iterations>128 || o.jacobi_max_sweeps>64 || caps.maximum_owned_numerical_bytes>(8U<<20)
            || caps.maximum_total_numerical_bytes>(16U<<20) || caps.other_live_numerical_bytes>(8U<<20)
            || caps.maximum_work_units>10000000000ULL || caps.maximum_fock_calls>128)
            throw std::length_error("bounded UHF diagnostic exceeds tiny memory/work/iteration caps");
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("bounded UHF diagnostic progress must be callable or None");
        struct Model { const Z* kernel; const Z* bias; std::size_t count; std::uint64_t calls,fail; bool omit; };
        Model model{static_cast<const Z*>(kernel.data()),static_cast<const Z*>(bias.data()),static_cast<std::size_t>(count),0,fail_on_call,omit_last_output};
        const auto evaluate=[](const Z* density,std::size_t dn,Z* output,std::size_t on,void* pointer){
            auto& p=*static_cast<Model*>(pointer);
            if (dn!=p.count || on!=p.count) throw std::logic_error("bounded UHF diagnostic callback extent mismatch");
            ++p.calls; if (p.calls==p.fail) throw std::runtime_error("bounded UHF diagnostic provider failure");
            for (std::size_t i=0;i<p.count-(p.omit?1:0);++i) {
                // Independent small-model implementation; native solver has
                // no knowledge of this linear kernel or its numerical form.
                Z value=p.bias[i]; for (std::size_t j=0;j<p.count;++j) value+=p.kernel[i*p.count+j]*density[j];
                output[i]=value;
            }
        };
        BoundedPeriodicUHFInput input{k,n,alpha,beta,nuclear,static_cast<const Z*>(overlap.data()),static_cast<std::size_t>(count/2),
            static_cast<const Z*>(hcore.data()),static_cast<std::size_t>(count/2)};
        const BoundedPeriodicUHFTwoElectronProvider provider{+evaluate,&model,16*(count*count+count),0,128*(count*count+count+1)};
        const auto notify=[](const Snapshot& value,void* context){
            const py::object response=(*static_cast<const py::object*>(context))(py::cast(Snapshot(value),py::return_value_policy::move));
            return response.is_none() || response.cast<bool>();
        };
        // No native numerical views cross into callbacks. All matrix inputs
        // remain borrowed immutable call arguments; scalar controls are copied.
        return solve_bounded_periodic_uhf(input,provider,o,caps,callback.is_none()?nullptr:+notify,
            callback.is_none()?nullptr:const_cast<py::object*>(&callback));
    },py::arg("overlap"),py::arg("hcore"),py::arg("kernel"),py::arg("bias"),py::arg("n_alpha"),py::arg("n_beta"),
      py::arg("nuclear_energy"),py::arg("options"),py::arg("caps"),py::arg("callback")=py::none(),
      py::arg("fail_on_call")=0,py::arg("omit_last_output")=false);
}
