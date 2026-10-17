#include "vibeqc/manta_periodic_nuclear.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void check(const py::array& a,const py::dtype& dtype,int dimensions) {
    if(!a.dtype().is(dtype) || a.ndim()!=dimensions)
        throw py::value_error("MANTA periodic nuclear requires float64 vectors with the documented shapes");
    if(!(a.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA periodic nuclear inputs must be C- or F-contiguous");
}
Count vectors(const py::array& a) {
    check(a,py::dtype::of<double>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA periodic nuclear vectors must have shape (count,3)");
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
template<class T> py::array_t<T> owned(const std::vector<T>& values,Count count,Count n) {
    py::array_t<T> a({static_cast<py::ssize_t>(count),static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)});
    if(!values.empty()) std::memcpy(a.mutable_data(),values.data(),values.size()*sizeof(T));
    return a;
}
} // namespace
void bind_manta_periodic_nuclear(py::module_& m) {
    auto p=py::class_<PeriodicNuclearPlan>(m,"PeriodicNuclearPlan");
#define P(name) p.def_readonly(#name,&PeriodicNuclearPlan::name)
    P(n_basis); P(vectors); P(cells); P(momenta); P(reciprocal_batch); P(batches);
    P(maximum_nuclear_candidates_per_pair); P(nuclear_image_candidates_upper_bound);
    P(nuclear_shell_calls); P(overlap_shell_calls); P(fourier_shell_calls);
    P(shell_calls); P(primitive_pairs); P(primitive_charge_products_upper_bound);
    P(vector_primitive_products); P(contraction_terms); P(scalar_terms); P(kernel_terms);
    P(output_bytes); P(driver_live_bytes); P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicNuclearReceipt>(m,"PeriodicNuclearReceipt");
#define R(name) r.def_readonly(#name,&PeriodicNuclearReceipt::name)
    R(provider); R(provider_version); R(normalization); R(algorithm); R(boundary); R(nuclear_support); R(source_identity);
    R(nuclear_shell_calls); R(overlap_shell_calls); R(fourier_shell_calls); R(skipped_shell_pairs);
    R(primitive_pairs); R(primitive_charge_products); R(nuclear_images_retained); R(vector_primitive_products);
    R(contraction_terms); R(modeled_peak_bytes); R(alpha); R(nuclear_cutoff_bohr); R(cell_volume);
    R(nuclear_charge); R(constant_coefficient); R(cell_imaginary_residual); R(response_hermiticity_residual);
#undef R
    py::class_<PeriodicNuclearResult>(m,"PeriodicNuclearResult")
        .def_readonly("plan",&PeriodicNuclearResult::plan)
        .def_readonly("receipt",&PeriodicNuclearResult::receipt)
        .def_property_readonly("cell_values",[](const PeriodicNuclearResult& r) { return owned(r.cells,r.plan.cells,r.plan.n_basis); })
        .def_property_readonly("values",[](const PeriodicNuclearResult& r) { return owned(r.momenta,r.plan.momenta,r.plan.n_basis); })
        .def_property_readonly("short_range_cells",[](const PeriodicNuclearResult& r) { return owned(r.short_range_cells,r.plan.cells,r.plan.n_basis); })
        .def_property_readonly("reciprocal_cells",[](const PeriodicNuclearResult& r) { return owned(r.reciprocal_cells,r.plan.cells,r.plan.n_basis); })
        .def_property_readonly("constant_cells",[](const PeriodicNuclearResult& r) { return owned(r.constant_cells,r.plan.cells,r.plan.n_basis); });
    m.def("plan_periodic_nuclear",&plan_periodic_nuclear,py::arg("basis"),py::arg("system"),
          py::arg("vectors"),py::arg("cells"),py::arg("momenta"),py::arg("alpha"),py::arg("nuclear_cutoff_bohr"),
          py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
    m.def("compute_periodic_nuclear",[](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const py::array& g,const py::array& cells,const py::array& k,double alpha,double nuclear_cutoff,
        Count batch,std::optional<double> cutoff,const ResourceBudget& budget) {
        const Count ng=vectors(g),nc=vectors(cells),nk=vectors(k);
        plan_periodic_nuclear(basis,system,ng,nc,nk,alpha,nuclear_cutoff,batch,cutoff,budget);
        const auto q=copy(g),images=copy(cells),momenta=copy(k);
        return compute_periodic_nuclear(basis,system,q,images,momenta,alpha,nuclear_cutoff,batch,cutoff,budget);
    },py::arg("basis"),py::arg("system"),py::arg("wave_vectors").noconvert(),py::arg("translations").noconvert(),
      py::arg("momenta").noconvert(),py::arg("alpha"),py::arg("nuclear_cutoff_bohr"),
      py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
}
