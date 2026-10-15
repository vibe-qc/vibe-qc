#include "vibeqc/manta_lattice_tail.hpp"
#include "vibeqc/manta_gaussian_majorants.hpp"
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
    auto gp=py::class_<LatticeGaussianDomainPlan>(m,"LatticeGaussianDomainPlan");
#define P(name) gp.def_readonly(#name,&LatticeGaussianDomainPlan::name)
    P(domain); P(target_terms); P(source_terms); P(primitive_products); P(scalar_terms); P(kernel_terms); P(majorant_bytes); P(modeled_peak_bytes);
#undef P
    auto gr=py::class_<LatticeGaussianDomainRemainder>(m,"LatticeGaussianDomainRemainder");
#define R(name) gr.def_readonly(#name,&LatticeGaussianDomainRemainder::name)
    R(plan); R(physical_eta); R(minimum_effective_eta); R(minimum_omitted_real_distance);
    R(real_remainder); R(reciprocal_remainder); R(total); R(majorant_pairs_evaluated);
    R(real_cube_radius); R(reciprocal_cube_radius); R(source_identity); R(kernel_identity);
    R(target_majorant_identity); R(source_majorant_identity); R(real_domain_identity);
    R(reciprocal_domain_identity); R(near_domain_identity); R(qualification); R(roundoff_included);
#undef R
    m.def("plan_lattice_gaussian_domain_remainder",&plan_lattice_gaussian_domain_remainder,
        py::arg("target_terms"),py::arg("source_terms"),py::arg("real_cells"),py::arg("reciprocal_vectors"),py::arg("near_cells"),py::arg("budget")=ResourceBudget{});
    m.def("lattice_gaussian_domain_remainder",[](const py::array& cell,double eta,
        const py::array& real,const py::array& reciprocal,const py::array& near,
        const GaussianMajorant& target,const GaussianMajorant& source,const ResourceBudget& budget) {
        shape(cell,py::dtype::of<double>(),2);
        if(cell.shape(0)!=3 || cell.shape(1)!=3) throw py::value_error("MANTA Gaussian domain cell shape must be (3,3)");
        const auto nr=count(real),ng=count(reciprocal),nn=count(near); const auto limits=budget;
        plan_lattice_gaussian_domain_remainder(target.terms().size(),source.terms().size(),nr,ng,nn,limits);
        Eigen::Matrix3d a;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) a(i,j)=item<double>(cell,i*cell.strides(0)+j*cell.strides(1));
        const auto rs=copy(real),gs=copy(reciprocal),ns=copy(near);
        py::gil_scoped_release release;
        return lattice_gaussian_domain_remainder(a,eta,rs,gs,ns,target,source,limits);
    },py::arg("cell").noconvert(),py::arg("eta"),py::arg("real_indices").noconvert(),
        py::arg("reciprocal_indices").noconvert(),py::arg("near_indices").noconvert(),
        py::arg("target"),py::arg("source"),py::arg("budget")=ResourceBudget{});
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
    auto ep=py::class_<LatticeTailRemainderPlan>(m,"LatticeTailRemainderPlan");
#define P(name) ep.def_readonly(#name,&LatticeTailRemainderPlan::name)
    P(order); P(real_cells); P(reciprocal_vectors); P(near_cells); P(radial_integrals);
    P(scalar_terms); P(kernel_terms); P(output_bytes); P(domain_bytes); P(workspace_bytes); P(modeled_peak_bytes);
#undef P
    auto e=py::class_<LatticeTailRemainder>(m,"LatticeTailRemainder");
#define R(name) e.def_readonly(#name,&LatticeTailRemainder::name)
    R(plan); R(target_radius); R(source_radius); R(target_norm); R(source_norm);
    R(near_remainder); R(real_remainder); R(reciprocal_remainder); R(total); R(minimum_non_near_distance);
    R(source_identity); R(expansion_identity); R(kernel_identity); R(real_domain_identity);
    R(reciprocal_domain_identity); R(near_domain_identity); R(qualification); R(roundoff_included);
    R(radial_integrals_evaluated); R(reciprocal_terms_evaluated);
#undef R
    m.def("plan_lattice_tail_remainder",&plan_lattice_tail_remainder,py::arg("order"),py::arg("real_cells"),
        py::arg("reciprocal_vectors"),py::arg("near_cells"),py::arg("budget")=ResourceBudget{});
    m.def("lattice_tail_remainder",[](const py::array& cell,const py::array& r,double eta,
        const py::array& real,const py::array& reciprocal,const py::array& near,int order,
        double ra,double rb,double qa,double qb,const ResourceBudget& budget) {
        shape(cell,py::dtype::of<double>(),2); shape(r,py::dtype::of<double>(),1);
        if(cell.shape(0)!=3 || cell.shape(1)!=3 || r.shape(0)!=3)
            throw py::value_error("MANTA lattice-tail cell/displacement shapes must be (3,3) and (3,)");
        const auto nr=count(real),ng=count(reciprocal),nn=count(near);
        plan_lattice_tail_remainder(order,nr,ng,nn,budget); // before any payload read or copy
        Eigen::Matrix3d a; md::Point x;
        for(int i=0;i<3;++i) {
            x[i]=item<double>(r,i*r.strides(0));
            for(int j=0;j<3;++j) a(i,j)=item<double>(cell,i*cell.strides(0)+j*cell.strides(1));
        }
        const auto rs=copy(real),gs=copy(reciprocal),ns=copy(near);
        py::gil_scoped_release release;
        return lattice_tail_remainder(a,x,eta,rs,gs,ns,order,ra,rb,qa,qb,budget);
    },py::arg("cell").noconvert(),py::arg("target_minus_source").noconvert(),py::arg("eta"),
        py::arg("real_indices").noconvert(),py::arg("reciprocal_indices").noconvert(),
        py::arg("near_indices").noconvert(),py::arg("order"),py::arg("target_radius"),py::arg("source_radius"),
        py::arg("target_norm"),py::arg("source_norm"),py::arg("budget")=ResourceBudget{});
    auto dp=py::class_<LatticeDomainPlan>(m,"LatticeDomainPlan");
#define P(name) dp.def_readonly(#name,&LatticeDomainPlan::name)
    P(real_cells); P(reciprocal_vectors); P(near_cells); P(scalar_terms); P(kernel_terms);
    P(output_bytes); P(domain_bytes); P(workspace_bytes); P(modeled_peak_bytes);
#undef P
    auto dr=py::class_<LatticeDomainRemainder>(m,"LatticeDomainRemainder");
#define R(name) dr.def_readonly(#name,&LatticeDomainRemainder::name)
    R(plan); R(real_cube_radius); R(reciprocal_cube_radius);
    R(target_radius); R(source_radius); R(target_norm); R(source_norm); R(displacement_bound);
    R(real_metric_lower); R(reciprocal_metric_lower); R(minimum_omitted_real_distance);
    R(real_remainder); R(reciprocal_remainder); R(total); R(source_identity); R(kernel_identity);
    R(real_domain_identity); R(reciprocal_domain_identity); R(near_domain_identity);
    R(qualification); R(roundoff_included);
#undef R
    m.def("plan_lattice_domain_remainder",&plan_lattice_domain_remainder,py::arg("real_cells"),
        py::arg("reciprocal_vectors"),py::arg("near_cells"),py::arg("budget")=ResourceBudget{});
    m.def("lattice_domain_remainder",[](const py::array& cell,const py::array& r,double eta,
        const py::array& real,const py::array& reciprocal,const py::array& near,
        double ra,double rb,double qa,double qb,const ResourceBudget& budget) {
        shape(cell,py::dtype::of<double>(),2); shape(r,py::dtype::of<double>(),1);
        if(cell.shape(0)!=3 || cell.shape(1)!=3 || r.shape(0)!=3)
            throw py::value_error("MANTA lattice-domain cell/displacement shapes must be (3,3) and (3,)");
        const auto nr=count(real),ng=count(reciprocal),nn=count(near); const auto limits=budget;
        plan_lattice_domain_remainder(nr,ng,nn,limits); // before payload reads/copies
        Eigen::Matrix3d a; md::Point x;
        for(int i=0;i<3;++i) {
            x[i]=item<double>(r,i*r.strides(0));
            for(int j=0;j<3;++j) a(i,j)=item<double>(cell,i*cell.strides(0)+j*cell.strides(1));
        }
        const auto rs=copy(real),gs=copy(reciprocal),ns=copy(near);
        py::gil_scoped_release release;
        return lattice_domain_remainder(a,x,eta,rs,gs,ns,ra,rb,qa,qb,limits);
    },py::arg("cell").noconvert(),py::arg("target_minus_source").noconvert(),py::arg("eta"),
        py::arg("real_indices").noconvert(),py::arg("reciprocal_indices").noconvert(),
        py::arg("near_indices").noconvert(),py::arg("target_radius"),py::arg("source_radius"),
        py::arg("target_norm"),py::arg("source_norm"),py::arg("budget")=ResourceBudget{});
}
