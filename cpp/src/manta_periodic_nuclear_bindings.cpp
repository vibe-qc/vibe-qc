#include "vibeqc/manta_periodic_nuclear.hpp"
#include "vibeqc/manta_periodic_nuclear_bounds.hpp"
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
py::array_t<double> bound_matrix(const md::Matrix& values) {
    py::array_t<double> a({py::ssize_t(values.rows()),py::ssize_t(values.cols())});
    auto out=a.mutable_unchecked<2>();for(Eigen::Index i=0;i<values.rows();++i)for(Eigen::Index j=0;j<values.cols();++j)out(i,j)=values(i,j);
    a.attr("setflags")(false);return a;
}
} // namespace
void bind_manta_periodic_nuclear(py::module_& m) {
    auto options=py::class_<PeriodicNuclearBoundOptions>(m,"PeriodicNuclearBoundOptions").def(py::init<>());
#define O(name) options.def_readwrite(#name,&PeriodicNuclearBoundOptions::name)
    O(ao_completion_cube_radius);O(real_completion_cube_radius);O(reciprocal_completion_cube_radius);
#undef O
    auto ao=py::class_<PeriodicNuclearAccuracyOptions,PeriodicNuclearBoundOptions>(m,"PeriodicNuclearAccuracyOptions").def(py::init<>());
    ao.def_readwrite("matrix_error",&PeriodicNuclearAccuracyOptions::matrix_error)
        .def_readwrite("attraction_energy_error",&PeriodicNuclearAccuracyOptions::attraction_energy_error)
        .def_readwrite("nuclear_energy_error",&PeriodicNuclearAccuracyOptions::nuclear_energy_error);
    auto ap=py::class_<PeriodicNuclearAccuracyPlan>(m,"PeriodicNuclearAccuracyPlan");
#define P(name) ap.def_readonly(#name,&PeriodicNuclearAccuracyPlan::name)
    P(nuclear);P(modeled_peak_bytes);P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);
#undef P
    auto ar=py::class_<PeriodicNuclearAccuracyResult>(m,"PeriodicNuclearAccuracyResult");
#define R(name) ar.def_readonly(#name,&PeriodicNuclearAccuracyResult::name)
    R(nuclear);R(qualified);R(entry_attraction_energy_error_bound);R(l1_attraction_energy_error_bound);
    R(attraction_energy_error_bound);R(total_energy_error_bound);R(density_l1_limit);
    R(density_envelope_identity);R(certificate_identity);R(shell_calls);R(primitive_pairs);
#undef R
    ar.def_property_readonly("targets",[](const PeriodicNuclearAccuracyResult& r){return r.targets;});
    auto bp=py::class_<PeriodicNuclearBoundPlan>(m,"PeriodicNuclearBoundPlan");
#define P(name) bp.def_readonly(#name,&PeriodicNuclearBoundPlan::name)
    P(pairs);P(n_basis);P(momenta);P(vectors);P(points);P(ao_cube_cells);P(real_cube_cells);P(reciprocal_cube_cells);
    P(maximum_nuclear_candidates_per_pair);P(maximum_ordered_pairs);P(maximum_gaussian_terms);P(real_terms);
    P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto br=py::class_<PeriodicNuclearBoundReceipt>(m,"PeriodicNuclearBoundReceipt");
#define R(name) br.def_readonly(#name,&PeriodicNuclearBoundReceipt::name)
    R(qualification);R(boundary);R(nuclear_support);R(source_identity);R(nuclear_source_identity);R(pair_source_identity);R(pair_image_identity);
    R(shell_calls);R(primitive_pairs);R(ordered_pairs);R(majorant_primitive_pairs);R(real_terms);R(alpha);R(nuclear_cutoff_bohr);R(nuclear_absolute_charge);
    R(point_kernel_bound);R(reciprocal_weight_bound);R(maximum_potential_error);R(nuclear_real_error_bound);
    R(nuclear_reciprocal_error_bound);R(nuclear_energy_error_bound);R(roundoff_included);
#undef R
    auto result=py::class_<PeriodicNuclearBoundResult>(m,"PeriodicNuclearBoundResult");
    result.def_readonly("plan",&PeriodicNuclearBoundResult::plan).def_readonly("receipt",&PeriodicNuclearBoundResult::receipt)
        .def_property_readonly("options",[](const PeriodicNuclearBoundResult& r){return r.options;});
#define A(name) result.def_property_readonly(#name,[](const PeriodicNuclearBoundResult& r){return bound_matrix(r.name);})
    A(potential_error_bound);A(real_error_bound);A(reciprocal_error_bound);A(pair_image_error_bound);
#undef A
    m.def("plan_periodic_nuclear_bounds",&plan_periodic_nuclear_bounds,py::arg("basis"),py::arg("system"),
        py::arg("vectors"),py::arg("cells"),py::arg("momenta"),py::arg("alpha"),py::arg("nuclear_cutoff_bohr"),
        py::arg("pair_cutoff_bohr")=py::none(),py::arg("options")=PeriodicNuclearBoundOptions{},py::arg("budget")=ResourceBudget{});
    m.def("bound_periodic_nuclear",[](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const py::array& g,const py::array& cells,const py::array& k,double alpha,double cutoff,std::optional<double> paircut,
        const PeriodicNuclearBoundOptions& options,const ResourceBudget& budget) {
        plan_periodic_nuclear_bounds(basis,system,vectors(g),vectors(cells),vectors(k),alpha,cutoff,paircut,options,budget);
        const auto q=copy(g),r=copy(cells),momenta=copy(k);py::gil_scoped_release release;
        return bound_periodic_nuclear(basis,system,q,r,momenta,alpha,cutoff,paircut,options,budget);
    },py::arg("basis"),py::arg("system"),py::arg("wave_vectors").noconvert(),py::arg("translations").noconvert(),
        py::arg("momenta").noconvert(),py::arg("alpha"),py::arg("nuclear_cutoff_bohr"),py::arg("pair_cutoff_bohr")=py::none(),
        py::arg("options")=PeriodicNuclearBoundOptions{},py::arg("budget")=ResourceBudget{});
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
