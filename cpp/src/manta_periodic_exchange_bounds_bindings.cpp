#include "vibeqc/manta_periodic_exchange_bounds.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
template<class T>T item(const py::array& a,py::ssize_t offset){T x;std::memcpy(&x,static_cast<const char*>(a.data())+offset,sizeof(T));return x;}
void check(const py::array& a,const py::dtype& type,int rank){if(!a.dtype().is(type)||a.ndim()!=rank||!(a.flags()&(py::array::c_style|py::array::f_style)))throw py::value_error("MANTA exchange bounds require contiguous float64 vectors, int64 offsets and complex128 densities");}
py::array_t<double> owned(const md::Matrix& m){py::array_t<double>a({py::ssize_t(m.rows()),py::ssize_t(m.cols())});auto v=a.mutable_unchecked<2>();for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)v(i,j)=m(i,j);return a;}
py::array_t<double> owned(const PeriodicExchangeBoundResult& r,const std::vector<double>& v){const auto& p=r.plan;py::array_t<double>a({py::ssize_t(p.densities),py::ssize_t(p.momenta),py::ssize_t(p.n_basis),py::ssize_t(p.n_basis)});if(!v.empty())std::memcpy(a.mutable_data(),v.data(),v.size()*sizeof(double));return a;}
}
void bind_manta_periodic_exchange_bounds(py::module_& m){
    auto p=py::class_<PeriodicExchangeBoundPlan>(m,"PeriodicExchangeBoundPlan");
#define P(n) p.def_readonly(#n,&PeriodicExchangeBoundPlan::n)
    P(n_basis);P(momenta);P(densities);P(reciprocal_cube_radius);P(reciprocal_cube_cells);P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicExchangeBoundReceipt>(m,"PeriodicExchangeBoundReceipt");
#define R(n) r.def_readonly(#n,&PeriodicExchangeBoundReceipt::n)
    R(qualification);R(source_identity);R(pair_source_identity);R(density_identity);R(fourier_decay);R(maximum_error);R(exchange_scale);R(energy_error_bounds);R(roundoff_included);
#undef R
    py::class_<PeriodicExchangeBoundResult>(m,"PeriodicExchangeBoundResult").def_readonly("plan",&PeriodicExchangeBoundResult::plan).def_readonly("receipt",&PeriodicExchangeBoundResult::receipt)
        .def_property_readonly("error_bound",[](const PeriodicExchangeBoundResult& r){return owned(r,r.error_bound);})
        .def_property_readonly("pair_image_error_bound",[](const PeriodicExchangeBoundResult& r){return owned(r,r.pair_image_error_bound);})
        .def_property_readonly("reciprocal_error_bound",[](const PeriodicExchangeBoundResult& r){return owned(r,r.reciprocal_error_bound);});
    py::class_<PeriodicExchangeBoundSource>(m,"PeriodicExchangeBoundSource")
        .def(py::init([](const PeriodicPairSource& s,const py::array& q,const py::array& offsets,Count radius,const ResourceBudget& b){
            check(q,py::dtype::of<double>(),2);check(offsets,py::dtype::of<std::int64_t>(),1);
            const Count nk=s.plan().momenta,pairs=admission::square(nk);
            if(q.shape(1)!=3||Count(offsets.shape(0))!=admission::add(pairs,1))throw py::value_error("MANTA exchange bound vector/offset shape mismatch");
            PeriodicExchangeBoundSource::plan_source(s,q.shape(0),radius,1,b);
            md::Matrix wave(q.shape(0),3);for(Eigen::Index i=0;i<wave.rows();++i)for(int j=0;j<3;++j)wave(i,j)=item<double>(q,i*q.strides(0)+j*q.strides(1));
            std::vector<Count> o(pairs+1);for(Count i=0;i<=pairs;++i){const auto x=item<std::int64_t>(offsets,i*offsets.strides(0));if(x<0||x>q.shape(0))throw py::value_error("MANTA exchange bound offsets out of range");o[i]=x;}
            py::gil_scoped_release release;return PeriodicExchangeBoundSource(s,wave,o,radius,b);
        }),py::arg("source"),py::arg("vectors").noconvert(),py::arg("offsets").noconvert(),py::arg("reciprocal_cube_radius"),py::arg("budget")=ResourceBudget{})
        .def_static("plan_source",&PeriodicExchangeBoundSource::plan_source,py::arg("source"),py::arg("vectors"),py::arg("reciprocal_cube_radius"),py::arg("densities")=1,py::arg("budget")=ResourceBudget{})
        .def("plan_build",&PeriodicExchangeBoundSource::plan_build,py::arg("densities")=1,py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",&PeriodicExchangeBoundSource::plan,py::return_value_policy::copy)
        .def_property_readonly("receipt",&PeriodicExchangeBoundSource::receipt,py::return_value_policy::copy)
        .def_property_readonly("retained_amplitudes",[](const PeriodicExchangeBoundSource& s){return owned(s.retained_amplitudes());})
        .def_property_readonly("omitted_amplitudes",[](const PeriodicExchangeBoundSource& s){return owned(s.omitted_amplitudes());})
        .def_property_readonly("reciprocal_coefficients",[](const PeriodicExchangeBoundSource& s){return owned(s.reciprocal_coefficients());})
        .def("build",[](const PeriodicExchangeBoundSource& s,const py::array& d,const ResourceBudget& b){
            check(d,py::dtype::of<std::complex<double>>(),4);const auto& p=s.plan();
            if(Count(d.shape(1))!=p.momenta||Count(d.shape(2))!=p.n_basis||Count(d.shape(3))!=p.n_basis)throw py::value_error("MANTA exchange bound density shape mismatch");
            const Count nd=d.shape(0);s.plan_build(nd,b);std::vector<std::complex<double>> values(d.size());Count at=0;
            for(py::ssize_t a=0;a<d.shape(0);++a)for(py::ssize_t k=0;k<d.shape(1);++k)for(py::ssize_t i=0;i<d.shape(2);++i)for(py::ssize_t j=0;j<d.shape(3);++j)values[at++]=item<std::complex<double>>(d,a*d.strides(0)+k*d.strides(1)+i*d.strides(2)+j*d.strides(3));
            py::gil_scoped_release release;return s.build(values,nd,b);
        },py::arg("densities").noconvert(),py::arg("budget")=ResourceBudget{});
}
