#include "vibeqc/manta_periodic_ewald.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
void shape(const py::array& a,const py::dtype& type,int dimensions) {
    if(!a.dtype().is(type) || a.ndim()!=dimensions || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic Ewald needs contiguous arrays with the documented dtype and shape");
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T value; std::memcpy(&value,static_cast<const char*>(a.data())+offset,sizeof(T)); return value;
}
Count count(const py::array& a) {
    shape(a,py::dtype::of<std::int64_t>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA periodic Ewald indices require (count,3) shape");
    return a.shape(0);
}
std::vector<LatticeIndex> indices(const py::array& a) {
    std::vector<LatticeIndex> result(a.shape(0));
    for(py::ssize_t i=0;i<a.shape(0);++i) for(int j=0;j<3;++j)
        result[i][j]=item<std::int64_t>(a,i*a.strides(0)+j*a.strides(1));
    return result;
}
} // namespace
void bind_manta_periodic_ewald(py::module_& m) {
    using namespace vibeqc::manta;
    auto p=py::class_<PeriodicEwaldPlan>(m,"PeriodicEwaldPlan");
#define P(n) p.def_readonly(#n,&PeriodicEwaldPlan::n)
    P(n_basis); P(momenta); P(pairs); P(pair_interactions); P(real_cells); P(reciprocal_vectors);
    P(reciprocal_batch); P(reciprocal_batches); P(real_shell_calls); P(fourier_shell_calls);
    P(overlap_shell_calls); P(shell_calls); P(primitive_products); P(scalar_terms); P(kernel_terms); P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicEwaldReceipt>(m,"PeriodicEwaldReceipt");
#define R(n) r.def_readonly(#n,&PeriodicEwaldReceipt::n)
    R(provider); R(provider_version); R(normalization); R(algorithm); R(boundary); R(qualification);
    R(source_identity); R(pair_source_identity); R(kernel_identity); R(density_identity);
    R(real_domain_identity); R(reciprocal_domain_identity); R(real_shell_calls); R(fourier_shell_calls);
    R(overlap_shell_calls); R(primitive_products); R(vector_primitive_products); R(coefficient_panels); R(modeled_peak_bytes);
    R(eta); R(cell_volume); R(gauge_constant); R(charge); R(real_energy); R(reciprocal_energy);
    R(gauge_energy); R(energy); R(trace_energy); R(density_hermiticity_residual); R(response_hermiticity_residual);
#undef R
    py::class_<PeriodicEwaldResult>(m,"PeriodicEwaldResult")
        .def_readonly("plan",&PeriodicEwaldResult::plan).def_readonly("receipt",&PeriodicEwaldResult::receipt)
        .def_property_readonly("values",[](const PeriodicEwaldResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.n_basis);
            py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(r.plan.momenta),n,n});
            if(!r.values.empty()) std::memcpy(a.mutable_data(),r.values.data(),r.values.size()*sizeof(std::complex<double>));
            return a;
        });
    m.def("plan_periodic_ewald",&PeriodicEwaldSource::plan_source,py::arg("source"),py::arg("real_cells"),
        py::arg("reciprocal_vectors"),py::arg("reciprocal_batch")=64,py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicEwaldSource,std::shared_ptr<PeriodicEwaldSource>>(m,"PeriodicEwaldSource")
        .def(py::init([](std::shared_ptr<const PeriodicPairSource> s,double eta,const py::array& real,const py::array& reciprocal,
            Count batch,const ResourceBudget& budget) {
            if(!s) throw py::value_error("MANTA periodic Ewald requires a native periodic pair source");
            PeriodicEwaldSource::plan_source(*s,count(real),count(reciprocal),batch,budget);
            const auto rs=indices(real),gs=indices(reciprocal); const auto b=budget;
            py::gil_scoped_release release; return std::make_shared<PeriodicEwaldSource>(std::move(s),eta,rs,gs,batch,b);
        }),py::arg("source"),py::arg("eta"),py::arg("real_indices").noconvert(),py::arg("reciprocal_indices").noconvert(),
           py::arg("reciprocal_batch")=64,py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const PeriodicEwaldSource& s) { return PeriodicEwaldPlan(s.plan()); })
        .def_property_readonly("source_identity",&PeriodicEwaldSource::source_identity)
        .def_property_readonly("pair_source_identity",&PeriodicEwaldSource::pair_source_identity)
        .def_property_readonly("kernel_identity",&PeriodicEwaldSource::kernel_identity)
        .def("build",[](const PeriodicEwaldSource& s,const py::array& d) {
            shape(d,py::dtype::of<std::complex<double>>(),3); const auto& p=s.plan();
            if(d.shape(0)!=static_cast<py::ssize_t>(p.momenta) || d.shape(1)!=static_cast<py::ssize_t>(p.n_basis) || d.shape(2)!=static_cast<py::ssize_t>(p.n_basis))
                throw py::value_error("MANTA periodic Ewald density requires (momenta,AO,AO) shape");
            std::vector<std::complex<double>> values(p.momenta*p.n_basis*p.n_basis);
            for(Count k=0;k<p.momenta;++k) for(Count i=0;i<p.n_basis;++i) for(Count j=0;j<p.n_basis;++j)
                values[(k*p.n_basis+i)*p.n_basis+j]=item<std::complex<double>>(d,k*d.strides(0)+i*d.strides(1)+j*d.strides(2));
            py::gil_scoped_release release; return s.build(values);
        },py::arg("densities").noconvert());
}
