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
    P(n_basis);P(momenta);P(densities);P(ao_cube_radius);P(reciprocal_cube_radius);P(reciprocal_cube_cells);P(fourier_decay_bins);P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicExchangeBoundReceipt>(m,"PeriodicExchangeBoundReceipt");
#define R(n) r.def_readonly(#n,&PeriodicExchangeBoundReceipt::n)
    R(kernel);R(physical_omega);R(qualification);R(source_identity);R(pair_source_identity);R(density_identity);R(fourier_decay);R(maximum_error);R(exchange_scale);R(energy_error_bounds);R(roundoff_included);
#undef R
    py::class_<PeriodicExchangeBoundResult>(m,"PeriodicExchangeBoundResult").def_readonly("plan",&PeriodicExchangeBoundResult::plan).def_readonly("receipt",&PeriodicExchangeBoundResult::receipt)
        .def_property_readonly("error_bound",[](const PeriodicExchangeBoundResult& r){return owned(r,r.error_bound);})
        .def_property_readonly("pair_image_error_bound",[](const PeriodicExchangeBoundResult& r){return owned(r,r.pair_image_error_bound);})
        .def_property_readonly("reciprocal_error_bound",[](const PeriodicExchangeBoundResult& r){return owned(r,r.reciprocal_error_bound);});
    auto e=py::class_<PeriodicExchangeEnvelopeResult>(m,"PeriodicExchangeEnvelopeResult");
#define E(n) e.def_readonly(#n,&PeriodicExchangeEnvelopeResult::n)
    E(kernel);E(physical_omega);E(plan);E(density);E(qualification);E(source_identity);E(pair_source_identity);E(density_envelope_identity);E(certificate_identity);
    E(maximum_error);E(energy_error_bound);E(entry_energy_error_bound);E(l1_energy_error_bound);E(roundoff_included);
#undef E
    auto envelope_array=[](const PeriodicExchangeEnvelopeResult& e,const std::vector<double>& v) {
        py::array_t<double> a({py::ssize_t(e.plan.momenta),py::ssize_t(e.plan.n_basis),py::ssize_t(e.plan.n_basis)});
        std::memcpy(a.mutable_data(),v.data(),v.size()*sizeof(double));a.attr("flags").attr("writeable")=false;return a;
    };
    e.def_property_readonly("error_bound",[envelope_array](const PeriodicExchangeEnvelopeResult& e){return envelope_array(e,e.error_bound);})
     .def_property_readonly("pair_image_error_bound",[envelope_array](const PeriodicExchangeEnvelopeResult& e){return envelope_array(e,e.pair_image_error_bound);})
     .def_property_readonly("reciprocal_error_bound",[envelope_array](const PeriodicExchangeEnvelopeResult& e){return envelope_array(e,e.reciprocal_error_bound);});
    py::class_<PeriodicExchangeAccuracyOptions>(m,"PeriodicExchangeAccuracyOptions").def(py::init<>())
        .def_readwrite("matrix_error",&PeriodicExchangeAccuracyOptions::matrix_error)
        .def_readwrite("energy_error",&PeriodicExchangeAccuracyOptions::energy_error)
        .def_readwrite("ao_completion_cube_radius",&PeriodicExchangeAccuracyOptions::ao_completion_cube_radius)
        .def_readwrite("reciprocal_completion_cube_radius",&PeriodicExchangeAccuracyOptions::reciprocal_completion_cube_radius);
    auto ap=py::class_<PeriodicExchangeAccuracyPlan>(m,"PeriodicExchangeAccuracyPlan");
#define AP(n) ap.def_readonly(#n,&PeriodicExchangeAccuracyPlan::n)
    AP(pairs);AP(exchange);AP(images);AP(modeled_peak_bytes);AP(shell_calls);AP(primitive_pairs);AP(scalar_terms);AP(kernel_terms);
#undef AP
    auto ar=py::class_<PeriodicExchangeAccuracyResult>(m,"PeriodicExchangeAccuracyResult");
#define AR(n) ar.def_readonly(#n,&PeriodicExchangeAccuracyResult::n)
    AR(exchange);AR(images);AR(overlap_identity);AR(certificate_identity);AR(qualified);AR(bvk_correction);AR(method_scale);AR(fixed_madelung);
    AR(maximum_error);AR(energy_error_bound);AR(maximum_correction_error);AR(correction_energy_error_bound);AR(shell_calls);AR(primitive_pairs);
#undef AR
    ar.def_property_readonly("targets",[](const PeriodicExchangeAccuracyResult& r){return r.targets;})
        .def_property_readonly("error_bound",[envelope_array](const PeriodicExchangeAccuracyResult& r){return envelope_array(r.exchange,r.error_bound);})
        .def_property_readonly("correction_error_bound",[envelope_array](const PeriodicExchangeAccuracyResult& r){return envelope_array(r.exchange,r.correction_error_bound);});
    py::class_<PeriodicExchangeBoundSource>(m,"PeriodicExchangeBoundSource")
        .def(py::init([](const PeriodicPairSource& s,const py::array& q,const py::array& offsets,Count radius,const ResourceBudget& b,std::optional<Count> ao,const std::string& kernel,double omega,Count bins){
            check(q,py::dtype::of<double>(),2);check(offsets,py::dtype::of<std::int64_t>(),1);
            const Count nk=s.plan().momenta,pairs=admission::square(nk);
            if(q.shape(1)!=3||Count(offsets.shape(0))!=admission::add(pairs,1))throw py::value_error("MANTA exchange bound vector/offset shape mismatch");
            const Count completion=ao?*ao:PeriodicExchangeBoundSource::plan_source(s,q.shape(0),radius,1,b).ao_cube_radius;
            const auto p=PeriodicExchangeBoundSource::plan_source(s,q.shape(0),radius,1,completion,b,bins);
            md::Matrix wave(q.shape(0),3);for(Eigen::Index i=0;i<wave.rows();++i)for(int j=0;j<3;++j)wave(i,j)=item<double>(q,i*q.strides(0)+j*q.strides(1));
            std::vector<Count> o(pairs+1);for(Count i=0;i<=pairs;++i){const auto x=item<std::int64_t>(offsets,i*offsets.strides(0));if(x<0||x>q.shape(0))throw py::value_error("MANTA exchange bound offsets out of range");o[i]=x;}
            py::gil_scoped_release release;return PeriodicExchangeBoundSource(s,wave,o,radius,p.ao_cube_radius,b,kernel,omega,bins);
        }),py::arg("source"),py::arg("vectors").noconvert(),py::arg("offsets").noconvert(),py::arg("reciprocal_cube_radius"),py::arg("budget")=ResourceBudget{},py::arg("ao_completion_cube_radius")=py::none(),py::arg("kernel")="coulomb",py::arg("omega")=0.,py::arg("fourier_decay_bins")=1)
        .def_static("plan_source",[](const PeriodicPairSource& s,Count vectors,Count radius,Count nd,const ResourceBudget& b,std::optional<Count> ao,Count bins){
            const Count completion=ao?*ao:PeriodicExchangeBoundSource::plan_source(s,vectors,radius,nd,b).ao_cube_radius;
            return PeriodicExchangeBoundSource::plan_source(s,vectors,radius,nd,completion,b,bins);
        },py::arg("source"),py::arg("vectors"),py::arg("reciprocal_cube_radius"),py::arg("densities")=1,py::arg("budget")=ResourceBudget{},py::arg("ao_completion_cube_radius")=py::none(),py::arg("fourier_decay_bins")=1)
        .def("plan_build",&PeriodicExchangeBoundSource::plan_build,py::arg("densities")=1,py::arg("budget")=ResourceBudget{})
        .def("plan_density_envelope",&PeriodicExchangeBoundSource::plan_density_envelope,py::arg("budget")=ResourceBudget{})
        .def("bound_density_envelope",[](const PeriodicExchangeBoundSource& s,const vibeqc::BoundedPeriodicDensityEnvelope& d,const ResourceBudget& b){
            py::gil_scoped_release release;return s.bound_density_envelope(d,b);
        },py::arg("density_envelope"),py::arg("budget")=ResourceBudget{})
        .def("plan_occupied_subspaces",&PeriodicExchangeBoundSource::plan_occupied_subspaces,
            py::arg("n_alpha"),py::arg("n_beta"),py::arg("scf"),py::arg("budget")=ResourceBudget{})
        .def("bound_occupied_subspaces",[](const PeriodicExchangeBoundSource& s,const py::array& overlap,
            Count na,Count nb,const vibeqc::BoundedPeriodicRHFOptions& o,const ResourceBudget& b){
            check(overlap,py::dtype::of<std::complex<double>>(),3);const auto& p=s.plan();
            if(Count(overlap.shape(0))!=p.momenta||Count(overlap.shape(1))!=p.n_basis||Count(overlap.shape(2))!=p.n_basis)
                throw py::value_error("MANTA exchange envelope overlap shape mismatch");
            s.plan_occupied_subspaces(na,nb,o,b);std::vector<std::complex<double>> values(overlap.size());Count at=0;
            for(py::ssize_t k=0;k<overlap.shape(0);++k)for(py::ssize_t i=0;i<overlap.shape(1);++i)for(py::ssize_t j=0;j<overlap.shape(2);++j)
                values[at++]=item<std::complex<double>>(overlap,k*overlap.strides(0)+i*overlap.strides(1)+j*overlap.strides(2));
            py::gil_scoped_release release;return s.bound_occupied_subspaces(values,na,nb,o,b);
        },py::arg("overlap").noconvert(),py::arg("n_alpha"),py::arg("n_beta"),py::arg("scf"),py::arg("budget")=ResourceBudget{})
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
