#include "vibeqc/manta_periodic_hartree.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void check(const py::array& a,const py::dtype& dtype,int dimensions) {
    if(!a.dtype().is(dtype) || a.ndim()!=dimensions)
        throw py::value_error("MANTA periodic Hartree requires float64 vectors/weights and complex128 densities with the documented shapes");
    if(!(a.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA periodic Hartree inputs must be C- or F-contiguous");
}
Count vectors(const py::array& a) {
    check(a,py::dtype::of<double>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA periodic Hartree vectors must have shape (count,3)");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T out; std::memcpy(&out,static_cast<const char*>(a.data())+offset,sizeof(T)); return out;
}
md::Matrix copy(const py::array& a) {
    md::Matrix out(a.shape(0),a.shape(1));
    for(Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
        out(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
py::array_t<std::complex<double>> owned(const std::vector<std::complex<double>>& values,Count count,Count n) {
    py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(count),static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)});
    if(!values.empty()) std::memcpy(a.mutable_data(),values.data(),values.size()*sizeof(std::complex<double>));
    return a;
}
} // namespace
void bind_manta_periodic_hartree(py::module_& m) {
    auto p=py::class_<PeriodicHartreePlan>(m,"PeriodicHartreePlan");
#define P(name) p.def_readonly(#name,&PeriodicHartreePlan::name)
    P(n_basis); P(vectors); P(cells); P(momenta); P(reciprocal_batch); P(batches); P(panel_calls);
    P(shell_calls); P(primitive_pairs); P(vector_primitive_products); P(contraction_terms);
    P(scalar_terms); P(kernel_terms); P(output_bytes); P(driver_live_bytes); P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicHartreeReceipt>(m,"PeriodicHartreeReceipt");
#define R(name) r.def_readonly(#name,&PeriodicHartreeReceipt::name)
    R(provider); R(provider_version); R(normalization); R(algorithm); R(boundary); R(source_identity);
    R(density_identity); R(panel_calls); R(shell_calls); R(skipped_shell_pairs); R(primitive_pairs);
    R(vector_primitive_products); R(phase_products); R(phase_accumulations); R(contraction_terms);
    R(modeled_peak_bytes); R(energy); R(density_hermiticity_residual); R(cell_imaginary_residual);
    R(response_hermiticity_residual); R(cell_volume);
#undef R
    py::class_<PeriodicHartreeResult>(m,"PeriodicHartreeResult")
        .def_readonly("plan",&PeriodicHartreeResult::plan)
        .def_readonly("receipt",&PeriodicHartreeResult::receipt)
        .def_property_readonly("cell_values",[](const PeriodicHartreeResult& r) { return owned(r.cells,r.plan.cells,r.plan.n_basis); })
        .def_property_readonly("values",[](const PeriodicHartreeResult& r) { return owned(r.momenta,r.plan.momenta,r.plan.n_basis); });
    m.def("plan_periodic_hartree",&plan_periodic_hartree,py::arg("basis"),py::arg("vectors"),py::arg("cells"),py::arg("momenta"),
          py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
    m.def("compute_periodic_hartree",[](const vibeqc::BasisSet& basis,const py::array& lattice,const py::array& g,
        const py::array& cells,const py::array& k,const py::array& weights,const py::array& d,
        Count batch,std::optional<double> cutoff,const ResourceBudget& budget) {
        if(vectors(lattice)!=3) throw py::value_error("MANTA periodic Hartree lattice must have shape (3,3)");
        const Count ng=vectors(g),nc=vectors(cells),nk=vectors(k),n=basis.nbasis();
        check(weights,py::dtype::of<double>(),1); check(d,py::dtype::of<std::complex<double>>(),3);
        if(weights.shape(0)!=static_cast<py::ssize_t>(nk) || d.shape(0)!=static_cast<py::ssize_t>(nk)
           || d.shape(1)!=static_cast<py::ssize_t>(n) || d.shape(2)!=static_cast<py::ssize_t>(n))
            throw py::value_error("MANTA periodic Hartree weights/densities have incompatible shapes");
        plan_periodic_hartree(basis,ng,nc,nk,batch,cutoff,budget);
        const auto a=copy(lattice),q=copy(g),images=copy(cells),momenta=copy(k);
        std::vector<double> w(nk); std::vector<std::complex<double>> density(nk*n*n);
        for(Count ik=0;ik<nk;++ik) {
            w[ik]=item<double>(weights,ik*weights.strides(0));
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
                density[(ik*n+i)*n+j]=item<std::complex<double>>(d,ik*d.strides(0)+i*d.strides(1)+j*d.strides(2));
        }
        return compute_periodic_hartree(basis,a,q,images,momenta,w,density,batch,cutoff,budget);
    },py::arg("basis"),py::arg("lattice").noconvert(),py::arg("wave_vectors").noconvert(),py::arg("translations").noconvert(),
      py::arg("momenta").noconvert(),py::arg("weights").noconvert(),py::arg("densities").noconvert(),
      py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
}
