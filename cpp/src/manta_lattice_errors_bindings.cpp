#include "vibeqc/manta_lattice_errors.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
void shape(const py::array& a,const py::dtype& dtype,int ndim) {
    if(!a.dtype().is(dtype) || a.ndim()!=ndim || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA lattice Gaussian requires contiguous arrays with the documented dtype and shape");
}
Count count(const py::array& a) {
    shape(a,py::dtype::of<std::int64_t>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA lattice Gaussian domain shape must be (count,3)");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T x; std::memcpy(&x,static_cast<const char*>(a.data())+offset,sizeof(T)); return x;
}
std::vector<LatticeIndex> copy(const py::array& a) {
    std::vector<LatticeIndex> out(a.shape(0));
    for(py::ssize_t i=0;i<a.shape(0);++i) for(int j=0;j<3;++j)
        out[i][j]=item<std::int64_t>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
template<class A,class B> void bind_interaction(py::module_& m) {
    m.def("erfc_gaussian_interaction",[](double omega,const A& target,const B& source,int order,const ResourceBudget& budget) {
        const auto limits=budget;
        plan_erfc_gaussian_interaction(order,limits);
        py::gil_scoped_release release;
        return erfc_gaussian_interaction(omega,target,source,order,limits);
    },py::arg("omega"),py::arg("target").noconvert(),py::arg("source").noconvert(),py::arg("order"),py::arg("budget")=ResourceBudget{});
    m.def("lattice_gaussian_interaction",[](const py::array& cell,double eta,
        const py::array& real,const py::array& reciprocal,const py::array& near,
        const A& target,const B& source,int order,const ResourceBudget& budget) {
        shape(cell,py::dtype::of<double>(),2);
        if(cell.shape(0)!=3 || cell.shape(1)!=3)
            throw py::value_error("MANTA lattice Gaussian cell shape must be (3,3)");
        const auto nr=count(real),ng=count(reciprocal),nn=count(near);
        const auto admitted_budget=budget;
        plan_lattice_gaussian_interaction(order,nr,ng,nn,admitted_budget); // no payload read/copy before admission
        Eigen::Matrix3d a;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
            a(i,j)=item<double>(cell,i*cell.strides(0)+j*cell.strides(1));
        const auto rs=copy(real),gs=copy(reciprocal),ns=copy(near);
        py::gil_scoped_release release;
        return lattice_gaussian_interaction(a,eta,rs,gs,ns,target,source,order,admitted_budget);
    },py::arg("cell").noconvert(),py::arg("eta"),py::arg("real_indices").noconvert(),
        py::arg("reciprocal_indices").noconvert(),py::arg("near_indices").noconvert(),
        py::arg("target").noconvert(),py::arg("source").noconvert(),py::arg("order"),
        py::arg("budget")=ResourceBudget{});
}
} // namespace
void bind_manta_lattice_errors(py::module_& m) {
    using namespace vibeqc::manta;
    auto p=py::class_<LatticeGaussianPlan>(m,"LatticeGaussianPlan");
#define P(name) p.def_readonly(#name,&LatticeGaussianPlan::name)
    P(coefficients); P(compact); P(scalar_terms); P(kernel_terms);
    P(envelope_bytes); P(workspace_bytes); P(modeled_peak_bytes);
#undef P
    auto e=py::class_<LatticeGaussianError>(m,"LatticeGaussianError");
#define E(name) e.def_readonly(#name,&LatticeGaussianError::name)
    E(compact_remainder); E(tail_penetration); E(full_moment_tail); E(total); E(bounded_kernel_sup);
    E(source_identity); E(target_envelope_identity); E(source_envelope_identity);
    E(qualification); E(roundoff_included); E(moment_products_evaluated); E(reciprocal_terms_evaluated);
#undef E
    py::class_<LatticeGaussianInteraction>(m,"LatticeGaussianInteraction")
        .def_readonly("plan",&LatticeGaussianInteraction::plan)
        .def_readonly("matrix",&LatticeGaussianInteraction::matrix)
        .def_readonly("compact",&LatticeGaussianInteraction::compact)
        .def_readonly("error",&LatticeGaussianInteraction::error);
    m.def("plan_lattice_gaussian_interaction",&plan_lattice_gaussian_interaction,
        py::arg("order"),py::arg("real_cells"),py::arg("reciprocal_vectors"),py::arg("near_cells"),
        py::arg("budget")=ResourceBudget{});
    m.def("plan_erfc_gaussian_interaction",&plan_erfc_gaussian_interaction,py::arg("order"),py::arg("budget")=ResourceBudget{});
    bind_interaction<PairExtent,PairExtent>(m);
    bind_interaction<PairExtent,AuxiliaryExtent>(m);
    bind_interaction<AuxiliaryExtent,PairExtent>(m);
    bind_interaction<AuxiliaryExtent,AuxiliaryExtent>(m);
}
