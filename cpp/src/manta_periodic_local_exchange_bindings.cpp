#include "vibeqc/manta_periodic_local_exchange.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
void check(const py::array& a,const py::dtype& dtype,int dimensions) {
    if(!a.dtype().is(dtype) || a.ndim()!=dimensions || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA local exchange requires contiguous float64 source arrays and complex128 D[spin,k,AO,AO]");
}
Count vectors(const py::array& a) {
    check(a,py::dtype::of<double>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA local exchange vectors require shape (count,3)");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T out; std::memcpy(&out,static_cast<const char*>(a.data())+offset,sizeof(T)); return out;
}
Eigen::MatrixXd copy(const py::array& a) {
    Eigen::MatrixXd out(a.shape(0),a.shape(1));
    for(Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
        out(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
}
void bind_manta_periodic_local_exchange(py::module_& m) {
    auto p=py::class_<PeriodicLocalExchangePlan>(m,"PeriodicLocalExchangePlan");
#define P(x) p.def_readonly(#x,&PeriodicLocalExchangePlan::x)
    P(n_basis); P(spins); P(momenta); P(images); P(grid_points); P(grid_batch);
    P(shell_calls); P(primitive_points); P(scalar_terms); P(kernel_terms);
    P(panel_bytes); P(input_bytes); P(output_bytes); P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicLocalExchangeReceipt>(m,"PeriodicLocalExchangeReceipt");
#define R(x) r.def_readonly(#x,&PeriodicLocalExchangeReceipt::x)
    R(algorithm); R(functional); R(source_identity); R(density_identity); R(snapshot_identity); R(convention); R(support);
    R(energy); R(electrons_grid); R(density_hermiticity_residual); R(minimum_density);
    R(panel_calls); R(shell_calls); R(primitive_points); R(rounded_negative_points); R(modeled_peak_bytes);
#undef R
    py::class_<PeriodicLocalExchangeResult>(m,"PeriodicLocalExchangeResult")
        .def_readonly("plan",&PeriodicLocalExchangeResult::plan)
        .def_readonly("receipt",&PeriodicLocalExchangeResult::receipt)
        .def_property_readonly("values",[](const PeriodicLocalExchangeResult& r) {
            py::array_t<Complex> out({static_cast<py::ssize_t>(r.plan.spins),static_cast<py::ssize_t>(r.plan.momenta),
                static_cast<py::ssize_t>(r.plan.n_basis),static_cast<py::ssize_t>(r.plan.n_basis)});
            std::memcpy(out.mutable_data(),r.values.data(),r.values.size()*sizeof(Complex)); return out;
        });
    m.def("plan_periodic_local_exchange",&plan_periodic_local_exchange,
        py::arg("basis"),py::arg("images"),py::arg("momenta"),py::arg("grid_points"),py::arg("spins"),
        py::arg("grid_batch")=64,py::arg("budget")=ResourceBudget{});
    m.def("compute_periodic_local_exchange",[](const vibeqc::BasisSet& basis,const py::array& lattice,
        const py::array& images,const py::array& k,const py::array& weights,const py::array& points,
        const py::array& qw,const py::array& d,Count batch,const ResourceBudget& budget) {
        using namespace admission;
        if(vectors(lattice)!=3) throw py::value_error("MANTA local exchange lattice requires shape (3,3)");
        const Count ni=vectors(images),nk=vectors(k),ng=vectors(points),n=basis.nbasis();
        check(weights,py::dtype::of<double>(),1); check(qw,py::dtype::of<double>(),1); check(d,py::dtype::of<Complex>(),4);
        if(weights.shape(0)!=static_cast<py::ssize_t>(nk) || qw.shape(0)!=static_cast<py::ssize_t>(ng)
           || d.shape(1)!=static_cast<py::ssize_t>(nk) || d.shape(2)!=static_cast<py::ssize_t>(n) || d.shape(3)!=static_cast<py::ssize_t>(n))
            throw py::value_error("MANTA local exchange source extents disagree");
        const Count ns=d.shape(0); plan_periodic_local_exchange(basis,ni,nk,ng,ns,batch,budget);
        // Count-only admission precedes every numerical copy. Keep the GIL:
        // these borrowed Python owners cannot be cooperatively mutated mid-call.
        const auto a=copy(lattice),t=copy(images),momenta=copy(k),grid=copy(points);
        std::vector<double> w(nk),q(ng); std::vector<Complex> density(mul(mul(ns,nk),square(n)));
        for(Count i=0;i<nk;++i) w[i]=item<double>(weights,i*weights.strides(0));
        for(Count i=0;i<ng;++i) q[i]=item<double>(qw,i*qw.strides(0));
        for(Count spin=0;spin<ns;++spin) for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
            density[((spin*nk+ik)*n+i)*n+j]=item<Complex>(d,spin*d.strides(0)+ik*d.strides(1)+i*d.strides(2)+j*d.strides(3));
        return compute_periodic_local_exchange(basis,a,t,momenta,w,grid,q,density,ns,batch,budget);
    },py::arg("basis"),py::arg("lattice").noconvert(),py::arg("images").noconvert(),py::arg("momenta").noconvert(),
      py::arg("weights").noconvert(),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("densities").noconvert(),py::arg("grid_batch")=64,py::arg("budget")=ResourceBudget{});
}
