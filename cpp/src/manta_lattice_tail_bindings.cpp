#include "vibeqc/manta_lattice_tail.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
void shape(const py::array& a,const py::dtype& dtype,int dimensions) {
    if(!a.dtype().is(dtype) || a.ndim()!=dimensions || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA lattice-tail requires contiguous arrays with the documented dtype and shape");
}
vibeqc::manta::Count count(const py::array& a) {
    shape(a,py::dtype::of<std::int64_t>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA lattice-tail domain shape must be (count,3)");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T value; std::memcpy(&value,static_cast<const char*>(a.data())+offset,sizeof(T)); return value;
}
std::vector<vibeqc::manta::LatticeIndex> copy(const py::array& a) {
    std::vector<vibeqc::manta::LatticeIndex> out(a.shape(0));
    for(py::ssize_t i=0;i<a.shape(0);++i) for(int j=0;j<3;++j)
        out[i][j]=item<std::int64_t>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
} // namespace
void bind_manta_lattice_tail(py::module_& m) {
    using namespace vibeqc::manta;
    auto p=py::class_<LatticeTailPlan>(m,"LatticeTailPlan");
#define P(name) p.def_readonly(#name,&LatticeTailPlan::name)
    P(order); P(moments); P(real_cells); P(reciprocal_vectors); P(near_cells);
    P(scalar_terms); P(kernel_terms); P(output_bytes); P(workspace_bytes); P(domain_bytes); P(modeled_peak_bytes);
#undef P
    auto r=py::class_<LatticeTailMatrix>(m,"LatticeTailMatrix");
    r.def_property_readonly("values",[](const LatticeTailMatrix& a) { return Eigen::MatrixXd(a.values); });
#define R(name) r.def_readonly(#name,&LatticeTailMatrix::name)
    R(indices); R(plan); R(source_identity); R(kernel_identity); R(real_domain_identity);
    R(reciprocal_domain_identity); R(near_domain_identity); R(normalization); R(boundary); R(qualification);
    R(eta); R(cell_volume); R(gauge_constant); R(radial_derivatives_evaluated);
    R(reciprocal_coefficients_evaluated); R(matrix_coefficients_evaluated);
#undef R
    m.def("plan_lattice_tail",&plan_lattice_tail,py::arg("order"),py::arg("real_cells"),
        py::arg("reciprocal_vectors"),py::arg("near_cells"),py::arg("budget")=ResourceBudget{});
    m.def("multipole_lattice_tail",[](const py::array& cell,const py::array& r,double eta,
        const py::array& real,const py::array& reciprocal,const py::array& near,int order,const ResourceBudget& budget) {
        shape(cell,py::dtype::of<double>(),2); shape(r,py::dtype::of<double>(),1);
        if(cell.shape(0)!=3 || cell.shape(1)!=3 || r.shape(0)!=3)
            throw py::value_error("MANTA lattice-tail cell/displacement shapes must be (3,3) and (3,)");
        const auto nr=count(real),ng=count(reciprocal),nn=count(near);
        plan_lattice_tail(order,nr,ng,nn,budget); // before any payload read or copy
        Eigen::Matrix3d a; md::Point x;
        for(int i=0;i<3;++i) {
            x[i]=item<double>(r,i*r.strides(0));
            for(int j=0;j<3;++j) a(i,j)=item<double>(cell,i*cell.strides(0)+j*cell.strides(1));
        }
        const auto rs=copy(real),gs=copy(reciprocal),ns=copy(near);
        py::gil_scoped_release release;
        return multipole_lattice_tail(a,x,eta,rs,gs,ns,order,budget);
    },py::arg("cell").noconvert(),py::arg("target_minus_source").noconvert(),py::arg("eta"),
        py::arg("real_indices").noconvert(),py::arg("reciprocal_indices").noconvert(),
        py::arg("near_indices").noconvert(),py::arg("order"),py::arg("budget")=ResourceBudget{});
}
