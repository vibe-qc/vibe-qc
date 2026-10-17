#include "vibeqc/manta_periodic_electrostatic.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
void shape(const py::array& a,const py::dtype& type,int ndim) {
    if(!a.dtype().is(type) || a.ndim()!=ndim || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA electrostatics needs contiguous arrays with the documented dtype and shape");
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T x;std::memcpy(&x,static_cast<const char*>(a.data())+offset,sizeof(T));return x;
}
py::array_t<std::complex<double>> tensor(const std::vector<std::complex<double>>& v,const PeriodicElectrostaticPlan& p) {
    const auto n=static_cast<py::ssize_t>(p.n_basis);py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(p.momenta),n,n});
    if(!v.empty())std::memcpy(a.mutable_data(),v.data(),v.size()*sizeof(std::complex<double>));return a;
}
py::array_t<double> matrix(const md::Matrix& m) {
    py::array_t<double> a({m.rows(),m.cols()});std::memcpy(a.mutable_data(),m.data(),m.size()*sizeof(double));return a;
}
}
void bind_manta_periodic_electrostatic(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<PeriodicElectrostaticOptions>(m,"PeriodicElectrostaticOptions").def(py::init<>())
        .def_readwrite("charge_policy",&PeriodicElectrostaticOptions::charge_policy)
        .def_readwrite("charge_tolerance",&PeriodicElectrostaticOptions::charge_tolerance);
    auto p=py::class_<PeriodicElectrostaticPlan>(m,"PeriodicElectrostaticPlan");
#define P(n) p.def_readonly(#n,&PeriodicElectrostaticPlan::n)
    P(n_basis);P(momenta);P(points);P(pairs);P(nuclear_shell_calls);P(overlap_shell_calls);P(fourier_shell_calls);P(point_pair_terms);
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicElectrostaticReceipt>(m,"PeriodicElectrostaticReceipt");
#define R(n) r.def_readonly(#n,&PeriodicElectrostaticReceipt::n)
    R(source_identity);R(electronic_source_identity);R(pair_source_identity);R(kernel_identity);R(point_source_identity);R(density_identity);
    R(qualification);R(boundary);R(charge_policy);R(charge_tolerance);R(nuclear_charge);R(nuclear_absolute_charge);R(electron_charge);
    R(retained_net_charge);R(charge_uncertainty_bound);R(maximum_net_charge_bound);R(background_charge);
    R(nuclear_real);R(nuclear_reciprocal);R(nuclear_self);R(nuclear_gauge);R(nuclear_energy);
    R(attraction_real);R(attraction_reciprocal);R(attraction_gauge);R(attraction_energy);R(electronic_energy);R(electronic_gauge);
    R(total_gauge);R(energy);R(trace_energy);R(nuclear_domain_error_bound);R(attraction_domain_error_bound);R(attraction_image_error_bound);
    R(point_kernel_bound);R(energy_error_bound);R(maximum_j_error_bound);R(hermiticity_residual);
    R(nuclear_shell_calls);R(fourier_shell_calls);R(overlap_shell_calls);R(point_pair_terms);R(gaussian_point_bounds);R(point_pair_bounds);
    R(modeled_peak_bytes);R(roundoff_included);
#undef R
    py::class_<PeriodicElectrostaticResult>(m,"PeriodicElectrostaticResult")
        .def_readonly("plan",&PeriodicElectrostaticResult::plan).def_readonly("receipt",&PeriodicElectrostaticResult::receipt)
        .def_readonly("electronic",&PeriodicElectrostaticResult::electronic)
        .def_property_readonly("values",[](const PeriodicElectrostaticResult& r){return tensor(r.values,r.plan);})
        .def_property_readonly("j_error_bound",[](const PeriodicElectrostaticResult& r){return matrix(r.j_error_bound);})
        .def_property_readonly("nuclear_j_error_bound",[](const PeriodicElectrostaticResult& r){return matrix(r.nuclear_j_error_bound);});
    m.def("plan_periodic_electrostatic",&PeriodicElectrostaticSource::plan_source,py::arg("electronic"),py::arg("points"),
        py::arg("options")=PeriodicElectrostaticOptions{},py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicElectrostaticSource,std::shared_ptr<PeriodicElectrostaticSource>>(m,"PeriodicElectrostaticSource")
        .def(py::init([](std::shared_ptr<const PeriodicBupoSource> b,const py::array& positions,const py::array& charges,
            const PeriodicElectrostaticOptions& options,const ResourceBudget& budget) {
            if(!b)throw py::value_error("MANTA electrostatics requires a native periodic BUPO source");
            shape(positions,py::dtype::of<double>(),2);shape(charges,py::dtype::of<double>(),1);
            if(positions.shape(1)!=3 || positions.shape(0)!=charges.shape(0))
                throw py::value_error("MANTA electrostatic point positions and charges require (points,3) and (points) shapes");
            PeriodicElectrostaticSource::plan_source(*b,charges.shape(0),options,budget);
            md::Matrix points(positions.shape(0),3);std::vector<double> values(charges.shape(0));
            for(py::ssize_t i=0;i<charges.shape(0);++i) {
                values[i]=item<double>(charges,i*charges.strides(0));
                for(int j=0;j<3;++j)points(i,j)=item<double>(positions,i*positions.strides(0)+j*positions.strides(1));
            }
            const auto opts=options;const auto limits=budget;py::gil_scoped_release release;
            return std::make_shared<PeriodicElectrostaticSource>(std::move(b),points,values,opts,limits);
        }),py::arg("electronic"),py::arg("positions").noconvert(),py::arg("charges").noconvert(),
           py::arg("options")=PeriodicElectrostaticOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const PeriodicElectrostaticSource& s){return PeriodicElectrostaticPlan(s.plan());})
        .def_property_readonly("source_identity",&PeriodicElectrostaticSource::source_identity)
        .def_property_readonly("overlap",[](const PeriodicElectrostaticSource& s){return tensor(s.overlap(),s.plan());})
        .def_property_readonly("nuclear_values",[](const PeriodicElectrostaticSource& s){return tensor(s.nuclear_values(),s.plan());})
        .def("build",[](const PeriodicElectrostaticSource& s,const py::array& d) {
            shape(d,py::dtype::of<std::complex<double>>(),3);const auto& p=s.plan();
            if(d.shape(0)!=static_cast<py::ssize_t>(p.momenta) || d.shape(1)!=static_cast<py::ssize_t>(p.n_basis) || d.shape(2)!=static_cast<py::ssize_t>(p.n_basis))
                throw py::value_error("MANTA electrostatic density requires (momenta,AO,AO) shape");
            std::vector<std::complex<double>> values(p.momenta*p.n_basis*p.n_basis);
            for(Count k=0;k<p.momenta;++k)for(Count i=0;i<p.n_basis;++i)for(Count j=0;j<p.n_basis;++j)
                values[(k*p.n_basis+i)*p.n_basis+j]=item<std::complex<double>>(d,k*d.strides(0)+i*d.strides(1)+j*d.strides(2));
            py::gil_scoped_release release;return s.build(values);
        },py::arg("densities").noconvert());
}
