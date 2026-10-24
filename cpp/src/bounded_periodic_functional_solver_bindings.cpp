// Tiny nonlinear functional diagnostics, not physical XC or a Python callback provider.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "vibeqc/bounded_periodic_functional_solver.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif
void bind_bounded_periodic_functional_solver(py::module_& m) {
    using namespace vibeqc;
    using Z=std::complex<double>;
    using I=std::uint64_t;
    using O=BoundedPeriodicRHFOptions;
    using C=BoundedPeriodicRHFCaps;
    const auto shape=[](I k,I n,I alpha,I beta,bool opened,const O& o) {
        if(!k || k>8 || n<2 || n>4 || o.maximum_iterations>128 || o.jacobi_max_sweeps>64)
            throw std::length_error("functional diagnostic exceeds tiny dimensions/iterations");
        // Bound each population before the restricted 2*alpha conversion.
        // Checking the doubled value would accept uint64 wraparound.
        if(alpha>=n || beta>=n || (!alpha && !beta))
            throw std::invalid_argument("functional diagnostic requires occupied and virtual bands");
        if(!opened && alpha!=beta) throw std::invalid_argument("restricted functional requires equal populations");
    };
    const auto provider=[](I count) {
        const auto noop=[](const Z*,std::size_t,Z*,std::size_t,double*,void*){};
        return BoundedPeriodicFunctionalProvider{+noop,nullptr,16*count+16,64,256*(count+1)};
    };
    m.def("_plan_bounded_periodic_functional_model",[shape,provider](I k,I n,I alpha,I beta,bool opened,I rank,O o,I other)->py::object {
        shape(k,n,alpha,beta,opened,o);
        if(other>(8U<<20)) throw std::length_error("functional diagnostic other memory exceeds cap");
        auto p=provider((opened?2:1)*k*n*n);
        if(opened) return py::cast(plan_bounded_periodic_uks_solver(k,n,alpha,beta,rank,p,o,other));
        return py::cast(plan_bounded_periodic_rks_solver(k,n,2*alpha,rank,p,o,other));
    },py::arg("k"),py::arg("n"),py::arg("alpha"),py::arg("beta"),py::arg("opened"),py::arg("rank"),
        py::arg("options"),py::arg("other")=0);
    m.def("_solve_bounded_periodic_functional_model",[shape,provider](const py::array& overlap,const py::array& hcore,
        const py::array& op,I alpha,I beta,bool opened,double quadratic,double cubic,double nuclear,O o,C caps,
        const py::object& callback,I omit_energy_on_call,bool omit_potential)->py::object {
        const auto valid=[](const py::array& a){return a.dtype().is(py::dtype::of<Z>()) &&
            (a.flags()&py::array::c_style) && reinterpret_cast<std::uintptr_t>(a.data())%alignof(Z)==0;};
        if(!valid(overlap) || !valid(hcore) || !valid(op) || overlap.ndim()!=3)
            throw std::invalid_argument("functional diagnostic requires aligned contiguous complex128 arrays");
        const I k=overlap.shape(0),n=overlap.shape(1);
        shape(k,n,alpha,beta,opened,o);
        const I count=(opened?2:1)*k*n*n;
        if(overlap.shape(2)!=static_cast<py::ssize_t>(n) || hcore.ndim()!=3 || hcore.shape(0)!=static_cast<py::ssize_t>(k)
            || hcore.shape(1)!=static_cast<py::ssize_t>(n) || hcore.shape(2)!=static_cast<py::ssize_t>(n)
            || op.ndim()!=(opened?4:3) || op.size()!=static_cast<py::ssize_t>(count)
            || (opened && op.shape(0)!=2) || op.shape(opened?1:0)!=static_cast<py::ssize_t>(k)
            || op.shape(opened?2:1)!=static_cast<py::ssize_t>(n) || op.shape(opened?3:2)!=static_cast<py::ssize_t>(n))
            throw std::invalid_argument("functional diagnostic array shapes disagree");
        if(caps.maximum_owned_numerical_bytes>(8U<<20) || caps.maximum_total_numerical_bytes>(16U<<20)
            || caps.other_live_numerical_bytes>(8U<<20) || caps.maximum_work_units>10000000000ULL || caps.maximum_fock_calls>128)
            throw std::length_error("functional diagnostic exceeds resource caps");
        if(!callback.is_none() && !PyCallable_Check(callback.ptr())) throw std::invalid_argument("progress must be callable");
        struct Model { const Z* a; I count,n,k,calls,omit; double quadratic,cubic; bool omit_potential; };
        Model model{static_cast<const Z*>(op.data()),count,n,k,0,omit_energy_on_call,quadratic,cubic,omit_potential};
        const auto evaluate=[](const Z* d,std::size_t dn,Z* g,std::size_t gn,double* energy,void* pointer) {
            auto& p=*static_cast<Model*>(pointer);
            if(dn!=p.count || gn!=p.count) throw std::logic_error("functional diagnostic extent mismatch");
            ++p.calls; double rho=0.;
            for(I block=0;block<p.count/(p.n*p.n);++block) for(I i=0;i<p.n;++i) for(I j=0;j<p.n;++j)
                rho+=(d[block*p.n*p.n+i*p.n+j]*p.a[block*p.n*p.n+j*p.n+i]).real()/p.k;
            for(I i=0;i<p.count-(p.omit_potential?1:0);++i) g[i]=(p.quadratic*rho+p.cubic*rho*rho)*p.a[i];
            if(p.calls!=p.omit) *energy=p.quadratic*rho*rho/2+p.cubic*rho*rho*rho/3;
        };
        auto p=provider(count); p.evaluate=+evaluate; p.context=&model;
        const auto s=static_cast<const Z*>(overlap.data()),h=static_cast<const Z*>(hcore.data());
        if(opened) {
            const auto notify=[](const BoundedPeriodicUHFSnapshot& step,void* pointer) {
                const auto value=(*static_cast<const py::object*>(pointer))(py::cast(step));
                return value.is_none() || value.cast<bool>();
            };
            return py::cast(solve_bounded_periodic_uks({k,n,alpha,beta,nuclear,s,k*n*n,h,k*n*n},p,o,caps,
                callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:const_cast<py::object*>(&callback)));
        }
        const auto notify=[](const BoundedPeriodicRHFSnapshot& step,void* pointer) {
            const auto value=(*static_cast<const py::object*>(pointer))(py::cast(step));
            return value.is_none() || value.cast<bool>();
        };
        return py::cast(solve_bounded_periodic_rks({k,n,2*alpha,nuclear,s,k*n*n,h,k*n*n},p,o,caps,
            callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:const_cast<py::object*>(&callback)));
    },py::arg("overlap"),py::arg("hcore"),py::arg("operator"),py::arg("alpha"),py::arg("beta"),py::arg("opened"),
        py::arg("quadratic"),py::arg("cubic"),py::arg("nuclear"),py::arg("options"),py::arg("caps"),
        py::arg("callback")=py::none(),py::arg("omit_energy_on_call")=0,py::arg("omit_potential")=false);
}
