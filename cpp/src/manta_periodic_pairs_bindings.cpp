#include "vibeqc/manta_periodic_pairs.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void check(const py::array& a,const py::dtype& type,int ndim) {
    if(!a.dtype().is(type) || a.ndim()!=ndim || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic pairs require contiguous arrays with the documented dtype and shape");
}
Count rows(const py::array& a,const py::dtype& type) {
    check(a,type,2);
    if(a.shape(1)!=3) throw py::value_error("MANTA periodic pair vectors/images require (count,3) shape");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T value; std::memcpy(&value,static_cast<const char*>(a.data())+offset,sizeof(T)); return value;
}
md::Matrix matrix(const py::array& a) {
    md::Matrix out(a.shape(0),a.shape(1));
    for(Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
        out(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
py::array_t<double> owned(const md::Matrix& m) {
    py::array_t<double> a({static_cast<py::ssize_t>(m.rows()),static_cast<py::ssize_t>(m.cols())});
    auto v=a.mutable_unchecked<2>();
    for(Eigen::Index i=0;i<m.rows();++i) for(Eigen::Index j=0;j<m.cols();++j) v(i,j)=m(i,j);
    return a;
}
} // namespace

void bind_manta_periodic_pairs(py::module_& m) {
    py::class_<PeriodicPairOptions>(m,"PeriodicPairOptions")
        .def(py::init<>()).def_readwrite("order",&PeriodicPairOptions::order)
        .def_readwrite("extents",&PeriodicPairOptions::extents)
        .def_readwrite("pair_cutoff_bohr",&PeriodicPairOptions::pair_cutoff_bohr);
    auto p=py::class_<PeriodicPairPlan>(m,"PeriodicPairPlan");
#define P(n) p.def_readonly(#n,&PeriodicPairPlan::n)
    P(n_basis); P(cells); P(momenta); P(maximum_pairs); P(moments); P(shell_calls); P(primitive_pairs);
    P(scalar_terms); P(kernel_terms); P(retained_bytes); P(driver_live_bytes); P(modeled_peak_bytes);
#undef P
    auto e=py::class_<PeriodicPairEntry>(m,"PeriodicPairEntry");
#define E(n) e.def_readonly(#n,&PeriodicPairEntry::n)
    E(shells); E(offsets); E(sizes); E(image); E(translation); E(paired); E(extent);
#undef E
    auto r=py::class_<PeriodicPairReceipt>(m,"PeriodicPairReceipt");
#define R(n) r.def_readonly(#n,&PeriodicPairReceipt::n)
    R(source_identity); R(basis_identity); R(domain_identity); R(convention); R(qualification);
    R(pairs); R(ordered_pairs); R(skipped_ordered_pairs); R(primitive_pairs); R(envelope_evaluations);
    R(always_near_pairs); R(modeled_peak_bytes);
#undef R
    auto d=py::class_<PeriodicPairDensityReceipt>(m,"PeriodicPairDensityReceipt");
#define D(n) d.def_readonly(#n,&PeriodicPairDensityReceipt::n)
    D(source_identity); D(density_identity); D(shell_calls); D(primitive_pairs); D(phase_products);
    D(contraction_terms); D(modeled_peak_bytes); D(charge); D(coefficient_l1); D(absolute_charge_bound);
    D(density_hermiticity_residual);
#undef D
    py::class_<PeriodicPairDensity>(m,"PeriodicPairDensity")
        .def_readonly("receipt",&PeriodicPairDensity::receipt)
        .def_property_readonly("moments",[](const PeriodicPairDensity& d) { return owned(d.moments); })
        .def_property_readonly("pair_coefficient_l1",[](const PeriodicPairDensity& d) {
            py::array_t<double> out(d.pair_coefficient_l1.size());
            if(!d.pair_coefficient_l1.empty()) std::memcpy(out.mutable_data(),d.pair_coefficient_l1.data(),d.pair_coefficient_l1.size()*8);
            return out;
        });
    auto a=py::class_<PeriodicPairAdjoint>(m,"PeriodicPairAdjoint");
#define A(n) a.def_readonly(#n,&PeriodicPairAdjoint::n)
    A(n_basis); A(momenta); A(shell_calls); A(primitive_pairs); A(phase_products); A(contraction_terms);
    A(hermiticity_residual); A(source_identity); A(potential_identity);
#undef A
    a.def_property_readonly("values",[](const PeriodicPairAdjoint& result) {
        const auto n=static_cast<py::ssize_t>(result.n_basis);
        py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(result.momenta),n,n});
        if(!result.values.empty()) std::memcpy(out.mutable_data(),result.values.data(),result.values.size()*sizeof(std::complex<double>));
        return out;
    });
    m.def("plan_periodic_pairs",&plan_periodic_pairs,py::arg("basis"),py::arg("cells"),py::arg("momenta"),
        py::arg("options")=PeriodicPairOptions{},py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicPairSource,std::shared_ptr<PeriodicPairSource>>(m,"PeriodicPairSource")
        .def(py::init([](const vibeqc::BasisSet& basis,const py::array& lattice,const py::array& images,
            const py::array& k,const py::array& weights,const PeriodicPairOptions& options,const ResourceBudget& budget) {
            if(rows(lattice,py::dtype::of<double>())!=3) throw py::value_error("MANTA periodic pair lattice requires (3,3) shape");
            const auto nc=rows(images,py::dtype::of<std::int64_t>()),nk=rows(k,py::dtype::of<double>());
            check(weights,py::dtype::of<double>(),1);
            if(weights.shape(0)!=static_cast<py::ssize_t>(nk)) throw py::value_error("MANTA periodic pair weights must match momenta");
            const auto opts=options; const auto limits=budget;
            plan_periodic_pairs(basis,nc,nk,opts,limits); // no numerical payload read/copy yet
            const Eigen::Matrix3d cell=matrix(lattice); const auto points=matrix(k);
            std::vector<LatticeIndex> indices(nc); std::vector<double> w(nk);
            for(Count i=0;i<nc;++i) for(int j=0;j<3;++j)
                indices[i][j]=item<std::int64_t>(images,i*images.strides(0)+j*images.strides(1));
            for(Count i=0;i<nk;++i) w[i]=item<double>(weights,i*weights.strides(0));
            py::gil_scoped_release release;
            return std::make_shared<PeriodicPairSource>(basis,cell,indices,points,w,opts,limits);
        }),py::arg("basis"),py::arg("lattice").noconvert(),py::arg("images").noconvert(),
            py::arg("momenta").noconvert(),py::arg("weights").noconvert(),
            py::arg("options")=PeriodicPairOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",&PeriodicPairSource::plan,py::return_value_policy::reference_internal)
        .def_property_readonly("receipt",&PeriodicPairSource::receipt,py::return_value_policy::reference_internal)
        .def_property_readonly("entries",&PeriodicPairSource::entries)
        .def_property_readonly("extents",&PeriodicPairSource::extents)
        .def("moment_panel",&PeriodicPairSource::moment_panel,py::arg("pair"),py::call_guard<py::gil_scoped_release>())
        .def("contract",[](const PeriodicPairSource& source,const py::array& d) {
            check(d,py::dtype::of<std::complex<double>>(),3);
            const auto& plan=source.plan(); const Count nk=plan.momenta,n=plan.n_basis;
            if(d.shape(0)!=static_cast<py::ssize_t>(nk) || d.shape(1)!=static_cast<py::ssize_t>(n) || d.shape(2)!=static_cast<py::ssize_t>(n))
                throw py::value_error("MANTA periodic pair density requires (momenta,AO,AO) shape");
            std::vector<std::complex<double>> values(nk*n*n);
            for(Count k=0;k<nk;++k) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
                values[(k*n+i)*n+j]=item<std::complex<double>>(d,k*d.strides(0)+i*d.strides(1)+j*d.strides(2));
            py::gil_scoped_release release; return source.contract(values);
        },py::arg("densities").noconvert())
        .def("adjoint",[](const PeriodicPairSource& source,const py::array& potentials) {
            check(potentials,py::dtype::of<double>(),2);
            if(potentials.shape(0)!=static_cast<py::ssize_t>(source.receipt().pairs)
                || potentials.shape(1)!=static_cast<py::ssize_t>(source.plan().moments))
                throw py::value_error("MANTA periodic pair potentials require (pairs,moments) shape");
            const auto values=matrix(potentials);
            py::gil_scoped_release release; return source.adjoint(values);
        },py::arg("local_potentials").noconvert());
}
