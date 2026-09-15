// Included by bindings.cpp. No native borrowed reader escapes to Python:
// the GIL is held and array owners live until every synchronous query ends.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include "vibeqc/bounded_restricted_pair_ccsd_amplitudes.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

namespace bounded_pair_ccsd_binding {
using namespace vibeqc;
using I=std::uint64_t;
using Options=BoundedRestrictedPairCCSDAmplitudesOptions;
using Inventory=BoundedRestrictedPairCCSDAmplitudesInventory;
using Caps=BoundedRestrictedPairCCSDAmplitudesCaps;
using Plan=BoundedRestrictedPairCCSDAmplitudesMemoryPlan;
I sum(I a,I b) {
    if(b>std::numeric_limits<I>::max()-a) throw std::overflow_error("ragged CCSD diagnostic count overflow");
    return a+b;
}
I product(I a,I b) {
    if(a && b>std::numeric_limits<I>::max()/a) throw std::overflow_error("ragged CCSD diagnostic count overflow");
    return a*b;
}
void array(const py::array& a,int ndim,bool integer=false) {
    if(!(integer ? a.dtype().is(py::dtype::of<I>()) : a.dtype().is(py::dtype::of<double>()))
        || a.ndim()!=ndim || !(a.flags()&py::array::c_style)
        || reinterpret_cast<std::uintptr_t>(a.data())%8)
        throw std::invalid_argument("ragged CCSD diagnostic requires aligned contiguous exact dtype arrays");
}
BoundedRestrictedCCSDRealView view(const py::array& a) {
    return {static_cast<const double*>(a.data()),static_cast<std::size_t>(a.size())};
}
struct Borrowed {
    I n=0,o=0,p=0;
    std::array<BoundedRestrictedPairCCSDSinglesView,4> singles;
    std::array<BoundedRestrictedPairCCSDPairView,10> pairs;
    // Keep original array objects alive even if the diagnostic callback
    // replaces a Python list entry. No data copy is made.
    std::array<py::array,28> owners;
    BoundedRestrictedPairCCSDAmplitudesInput input() const {
        return {o,n,singles.data(),static_cast<std::size_t>(o),pairs.data(),static_cast<std::size_t>(p)};
    }
    void validate_python_descriptors() const {
        const auto check=[](const py::array& a,BoundedRestrictedCCSDRealView v,int ndim,I rows,I columns) {
            array(a,ndim);
            if(a.data()!=v.data || static_cast<I>(a.size())!=v.element_count
                || static_cast<I>(a.shape(0))!=rows || (ndim==2 && static_cast<I>(a.shape(1))!=columns))
                throw std::invalid_argument("ragged CCSD diagnostic callback changed array descriptors");
        };
        for(I i=0;i<o;++i) {
            check(owners[2*i],singles[i].coefficients,2,n,singles[i].rank);
            check(owners[2*i+1],singles[i].amplitudes,1,singles[i].rank,0);
        }
        for(I i=0;i<p;++i) {
            check(owners[2*o+2*i],pairs[i].coefficients,2,n,pairs[i].rank);
            check(owners[2*o+2*i+1],pairs[i].amplitudes,2,pairs[i].rank,pairs[i].rank);
        }
    }
};
Borrowed borrow(I n,const py::list& sc,const py::list& st,const py::list& pc,const py::list& pt,
    const Inventory& inventory,const Caps& caps) {
    Borrowed b; b.n=n; b.o=sc.size();
    if(!b.o || b.o>4 || !n || n>4 || st.size()!=b.o)
        throw std::invalid_argument("ragged CCSD diagnostic singles table has invalid tiny shape");
    b.p=b.o*(b.o+1)/2;
    if(pc.size()!=b.p || pt.size()!=b.p) throw std::invalid_argument("ragged CCSD diagnostic unordered pair table has wrong count");
    const auto base=plan_bounded_restricted_pair_ccsd_amplitudes_upper(b.o,n,0,0,inventory);
    if(!caps.maximum_occupied_count || b.o>caps.maximum_occupied_count
        || !caps.maximum_common_virtual_dimension || n>caps.maximum_common_virtual_dimension
        || !caps.maximum_pair_count || b.p>caps.maximum_pair_count
        || !caps.maximum_table_bytes || base.borrowed_table_bytes>caps.maximum_table_bytes
        || !caps.maximum_validation_work_units || base.metadata_work_units>caps.maximum_validation_work_units)
        throw std::length_error("ragged CCSD diagnostic table/count cap exceeded");
    for(I i=0;i<b.o;++i) {
        if(!py::isinstance<py::array>(sc[i]) || !py::isinstance<py::array>(st[i]))
            throw std::invalid_argument("ragged CCSD diagnostic requires actual arrays, not converted lists");
        const auto c=py::reinterpret_borrow<py::array>(sc[i]),t=py::reinterpret_borrow<py::array>(st[i]);
        array(c,2); array(t,1);
        if(c.shape(0)!=static_cast<py::ssize_t>(n) || c.shape(1)!=t.shape(0))
            throw std::invalid_argument("ragged CCSD singles numerical shape mismatch");
        b.singles[i]={static_cast<I>(t.size()),view(c),view(t)};
        b.owners[2*i]=c; b.owners[2*i+1]=t;
    }
    for(I i=0;i<b.p;++i) {
        if(!py::isinstance<py::array>(pc[i]) || !py::isinstance<py::array>(pt[i]))
            throw std::invalid_argument("ragged CCSD diagnostic requires actual arrays, not converted lists");
        const auto c=py::reinterpret_borrow<py::array>(pc[i]),t=py::reinterpret_borrow<py::array>(pt[i]);
        array(c,2); array(t,2);
        if(c.shape(0)!=static_cast<py::ssize_t>(n) || c.shape(1)!=t.shape(0) || t.shape(0)!=t.shape(1))
            throw std::invalid_argument("ragged CCSD pair numerical shape mismatch");
        b.pairs[i]={static_cast<I>(t.shape(0)),view(c),view(t)};
        b.owners[2*b.o+2*i]=c; b.owners[2*b.o+2*i+1]=t;
    }
    return b;
}
void tiny(const Plan& p) {
    if(p.borrowed_numerical_bytes>(1U<<20) || p.required_node_inventoried_bytes>(128U<<20)
        || p.validation_work_units>1000000000ULL)
        throw std::length_error("ragged CCSD diagnostic exceeds tiny memory or work");
}
} // namespace bounded_pair_ccsd_binding

void bind_bounded_restricted_pair_ccsd_amplitudes(py::module_& m) {
    using namespace bounded_pair_ccsd_binding;
    py::class_<Options>(m,"_BoundedRestrictedPairCCSDAmplitudesOptions").def(py::init<>())
        .def_readwrite("coefficient_orthogonality_tolerance",&Options::coefficient_orthogonality_tolerance);
    py::class_<Inventory>(m,"_BoundedRestrictedPairCCSDAmplitudesInventory").def(py::init<>())
        .def_readwrite("numerical_replicas",&Inventory::numerical_replicas)
        .def_readwrite("external_node_bytes",&Inventory::external_node_bytes)
        .def_readwrite("other_live_bytes_per_replica",&Inventory::other_live_bytes_per_replica)
        .def_readwrite("fixed_backend_margin_bytes_per_replica",&Inventory::fixed_backend_margin_bytes_per_replica);
    auto c=py::class_<Caps>(m,"_BoundedRestrictedPairCCSDAmplitudesCaps").def(py::init<>());
#define RCC_CAP(f) c.def_readwrite(#f,&Caps::f)
    RCC_CAP(maximum_occupied_count); RCC_CAP(maximum_common_virtual_dimension); RCC_CAP(maximum_pair_count);
    RCC_CAP(maximum_singles_rank); RCC_CAP(maximum_pair_rank); RCC_CAP(maximum_table_bytes);
    RCC_CAP(maximum_borrowed_numerical_bytes); RCC_CAP(maximum_per_replica_inventoried_bytes);
    RCC_CAP(maximum_node_inventoried_bytes); RCC_CAP(maximum_validation_work_units);
    RCC_CAP(maximum_singles_work_units_per_query); RCC_CAP(maximum_doubles_work_units_per_query);
#undef RCC_CAP
    auto p=py::class_<Plan>(m,"_BoundedRestrictedPairCCSDAmplitudesMemoryPlan");
#define RCC_PLAN(f) p.def_readonly(#f,&Plan::f)
    RCC_PLAN(uniform_rank_upper_bound); RCC_PLAN(n_occupied); RCC_PLAN(common_virtual_dimension); RCC_PLAN(pair_count);
    RCC_PLAN(maximum_singles_rank); RCC_PLAN(maximum_pair_rank); RCC_PLAN(singles_amplitude_elements);
    RCC_PLAN(pair_amplitude_elements); RCC_PLAN(singles_coefficient_elements); RCC_PLAN(pair_coefficient_elements);
    RCC_PLAN(borrowed_singles_table_bytes); RCC_PLAN(borrowed_pair_table_bytes); RCC_PLAN(borrowed_table_bytes);
    RCC_PLAN(borrowed_singles_numerical_bytes); RCC_PLAN(borrowed_pair_numerical_bytes); RCC_PLAN(borrowed_numerical_bytes);
    RCC_PLAN(owned_numerical_bytes); RCC_PLAN(fixed_control_storage_bytes); RCC_PLAN(metadata_work_units);
    RCC_PLAN(validation_work_units); RCC_PLAN(maximum_singles_work_units_per_query); RCC_PLAN(maximum_doubles_work_units_per_query);
    RCC_PLAN(numerical_replicas); RCC_PLAN(external_node_bytes); RCC_PLAN(per_replica_inventoried_bytes);
    RCC_PLAN(required_node_inventoried_bytes);
#undef RCC_PLAN
    using Diagnostics=BoundedRestrictedPairCCSDAmplitudesDiagnostics;
    py::class_<Diagnostics>(m,"_BoundedRestrictedPairCCSDAmplitudesDiagnostics")
        .def_readonly("maximum_coefficient_orthogonality_error",&Diagnostics::maximum_coefficient_orthogonality_error)
        .def_readonly("validated_singles",&Diagnostics::validated_singles)
        .def_readonly("validated_pairs",&Diagnostics::validated_pairs);
    m.def("_plan_bounded_restricted_pair_ccsd_amplitudes_upper",&plan_bounded_restricted_pair_ccsd_amplitudes_upper,
        py::arg("n_occupied"),py::arg("common_virtual_dimension"),py::arg("maximum_singles_rank"),
        py::arg("maximum_pair_rank"),py::arg("inventory"));
    m.def("_plan_bounded_restricted_pair_ccsd_amplitudes_diagnostic",[](I n,py::list sc,py::list st,py::list pc,py::list pt,
        Options options,Inventory inventory,Caps caps) {
        const auto b=borrow(n,sc,st,pc,pt,inventory,caps);
        auto p=plan_bounded_restricted_pair_ccsd_amplitudes(b.input(),options,inventory,caps); tiny(p); return p;
    });
    m.def("_bounded_restricted_pair_ccsd_amplitudes_query_diagnostic",[](I n,py::list sc,py::list st,py::list pc,py::list pt,
        Options options,Inventory inventory,Caps caps,py::array sq,py::array dq,I query_work_cap,py::object callback) {
        array(sq,2,true); array(dq,2,true);
        if(sq.shape(1)!=2 || dq.shape(1)!=4 || sq.shape(0)>256 || dq.shape(0)>4096)
            throw std::length_error("ragged CCSD diagnostic scalar queries exceed tiny shape/count caps");
        const I ns=sq.shape(0),nd=dq.shape(0),extra=sum(sum(sq.nbytes(),dq.nbytes()),product(8,sum(ns,nd)));
        inventory.other_live_bytes_per_replica=sum(inventory.other_live_bytes_per_replica,extra);
        const auto b=borrow(n,sc,st,pc,pt,inventory,caps);
        auto plan=plan_bounded_restricted_pair_ccsd_amplitudes(b.input(),options,inventory,caps); tiny(plan);
        const I work=sum(product(ns,plan.maximum_singles_work_units_per_query),product(nd,plan.maximum_doubles_work_units_per_query));
        if(!query_work_cap || work>query_work_cap) throw std::length_error("ragged CCSD diagnostic aggregate query work cap exceeded");
        auto reader=make_bounded_restricted_pair_ccsd_amplitudes(b.input(),options,inventory,caps);
        if(!callback.is_none()) {
            callback(); b.validate_python_descriptors(); reader.validate_immutable_snapshot();
            array(sq,2,true); array(dq,2,true);
            if(sq.shape(0)!=static_cast<py::ssize_t>(ns) || sq.shape(1)!=2
                || dq.shape(0)!=static_cast<py::ssize_t>(nd) || dq.shape(1)!=4)
                throw std::invalid_argument("ragged CCSD diagnostic callback changed query descriptors");
        }
        py::array_t<double> one(static_cast<py::ssize_t>(ns)),two(static_cast<py::ssize_t>(nd));
        const auto* s=static_cast<const I*>(sq.data()); const auto* d=static_cast<const I*>(dq.data());
        for(I i=0;i<ns;++i) one.mutable_data()[i]=reader.singles(s[2*i],s[2*i+1]);
        for(I i=0;i<nd;++i) two.mutable_data()[i]=reader.doubles(d[4*i],d[4*i+1],d[4*i+2],d[4*i+3]);
        const auto provider=reader.amplitude_provider();
        py::dict result; result["memory"]=reader.memory(); result["diagnostics"]=reader.diagnostics();
        result["snapshot_identity_sha256"]=reader.snapshot_identity_sha256(); result["singles"]=one; result["doubles"]=two;
        result["query_work_units"]=work; result["wrapper_extra_live_bytes"]=extra;
        result["provider_retained_numerical_bytes"]=provider.retained_numerical_bytes;
        result["provider_transient_numerical_bytes"]=provider.maximum_transient_numerical_bytes;
        return result;
    },py::arg("common_virtual_dimension"),py::arg("singles_coefficients"),py::arg("singles_amplitudes"),
      py::arg("pair_coefficients"),py::arg("pair_amplitudes"),py::arg("options"),py::arg("inventory"),py::arg("caps"),
      py::arg("singles_queries"),py::arg("doubles_queries"),py::arg("maximum_query_work_units"),py::arg("callback")=py::none());

    m.def("_bounded_restricted_pair_ccsd_target_diagnostic",[](I n,py::list sc,py::list st,py::list pc,py::list pt,
        Options options,Inventory inventory,Caps caps,py::array foo,py::array fvv,py::array fov,py::array eri,
        I i,I j,BoundedRestrictedCCSDTargetAccessorCaps target_caps) {
        array(foo,2); array(fvv,2); array(fov,2); array(eri,4);
        const auto b=borrow(n,sc,st,pc,pt,inventory,caps); const auto o=b.o,morb=o+n;
        if(foo.shape(0)!=static_cast<py::ssize_t>(o) || foo.shape(1)!=static_cast<py::ssize_t>(o)
            || fvv.shape(0)!=static_cast<py::ssize_t>(n) || fvv.shape(1)!=static_cast<py::ssize_t>(n)
            || fov.shape(0)!=static_cast<py::ssize_t>(o) || fov.shape(1)!=static_cast<py::ssize_t>(n))
            throw std::invalid_argument("ragged CCSD target diagnostic Fock shape mismatch");
        for(int axis=0;axis<4;++axis) if(eri.shape(axis)!=static_cast<py::ssize_t>(morb))
            throw std::invalid_argument("ragged CCSD target diagnostic integral shape mismatch");
        const auto initial=plan_bounded_restricted_pair_ccsd_amplitudes(b.input(),options,inventory,caps); tiny(initial);
        BoundedRestrictedCCSDAmplitudeProvider shape;
        shape.retained_numerical_bytes=initial.borrowed_numerical_bytes;
        shape.maximum_singles_work_units_per_query=initial.maximum_singles_work_units_per_query;
        shape.maximum_doubles_work_units_per_query=initial.maximum_doubles_work_units_per_query;
        const auto target=plan_bounded_restricted_ccsd_target_accessor(o,n,shape,eri.nbytes(),0);
        const I extra=sum(target.kernel.peak_owned_numerical_bytes,sum(eri.nbytes(),sum(foo.nbytes(),sum(fvv.nbytes(),fov.nbytes()))));
        inventory.other_live_bytes_per_replica=sum(inventory.other_live_bytes_per_replica,extra);
        tiny(plan_bounded_restricted_pair_ccsd_amplitudes(b.input(),options,inventory,caps));
        // Cap target work BEFORE the floating reader scan, despite no reader
        // allocation. The native target repeats its authoritative admission.
        if(target.kernel.peak_owned_numerical_bytes>target_caps.kernel.maximum_owned_numerical_bytes
            || target.kernel.total_live_numerical_bytes>target_caps.kernel.maximum_total_numerical_bytes
            || target.kernel.integral_calls_upper_bound>target_caps.kernel.maximum_integral_calls
            || target.kernel.kernel_work_units_upper_bound>target_caps.kernel.maximum_kernel_work_units
            || target.singles_calls_upper_bound>target_caps.maximum_singles_calls
            || target.doubles_calls_upper_bound>target_caps.maximum_doubles_calls
            || target.amplitude_work_units_upper_bound>target_caps.maximum_amplitude_work_units
            || target.amplitude_work_units_upper_bound>100000000000ULL)
            throw std::length_error("ragged CCSD target diagnostic memory/work cap exceeded");
        auto reader=make_bounded_restricted_pair_ccsd_amplitudes(b.input(),options,inventory,caps);
        struct Direct {
            const double* data; I n;
            static double value(I p,I q,I r,I s,void* opaque) {
                const auto& d=*static_cast<const Direct*>(opaque); return d.data[((p*d.n+q)*d.n+r)*d.n+s];
            }
        } direct{static_cast<const double*>(eri.data()),morb};
        BoundedRestrictedCCSDIntegralProvider integrals{Direct::value,&direct,static_cast<I>(eri.nbytes()),0};
        BoundedRestrictedCCSDTargetOperatorInput input{o,n,i,j,view(foo),view(fvv),view(fov)};
        return bounded_restricted_ccsd_target_residual_accessor(input,reader.amplitude_provider(),integrals,target_caps);
    });
}
