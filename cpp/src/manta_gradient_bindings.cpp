#include "vibeqc/manta_gradient.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
namespace {
void validate_array(const py::array& array,py::ssize_t n) {
    if(!array.dtype().is(py::dtype::of<double>()) || array.ndim()!=2 || array.shape(0)!=n || array.shape(1)!=n)
        throw py::value_error("MANTA HF gradient requires square float64 ndarrays matching the basis");
    if(!(array.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA HF gradient matrices must be C- or F-contiguous");
}
Eigen::MatrixXd copy_matrix(const py::array& array) {
    const auto n=array.shape(0); Eigen::MatrixXd matrix(n,n);
    const auto* bytes=static_cast<const char*>(array.data());
    if(array.flags() & py::array::f_style) std::memcpy(matrix.data(),bytes,array.size()*sizeof(double));
    else for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<n;++j)
        std::memcpy(&matrix(i,j),bytes+(i*n+j)*sizeof(double),sizeof(double));
    return matrix;
}
}
void bind_manta_gradient(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<HFGradientPlan>(m,"HFGradientPlan")
        .def_readonly("n_atoms",&HFGradientPlan::n_atoms)
        .def_readonly("n_basis",&HFGradientPlan::n_basis)
        .def_readonly("shell_calls",&HFGradientPlan::shell_calls)
        .def_readonly("primitive_evaluations_upper_bound",&HFGradientPlan::primitive_evaluations_upper_bound)
        .def_readonly("scalar_terms",&HFGradientPlan::scalar_terms)
        .def_readonly("kernel_terms",&HFGradientPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&HFGradientPlan::modeled_peak_bytes);
    py::class_<HFGradientReceipt>(m,"HFGradientReceipt")
        .def_readonly("provider",&HFGradientReceipt::provider)
        .def_readonly("provider_version",&HFGradientReceipt::provider_version)
        .def_readonly("algorithm",&HFGradientReceipt::algorithm)
        .def_readonly("source_identity",&HFGradientReceipt::source_identity)
        .def_readonly("density_identity",&HFGradientReceipt::density_identity)
        .def_readonly("normalization",&HFGradientReceipt::normalization)
        .def_readonly("shell_calls",&HFGradientReceipt::shell_calls)
        .def_readonly("primitive_integrals_evaluated",&HFGradientReceipt::primitive_integrals_evaluated)
        .def_readonly("modeled_peak_bytes",&HFGradientReceipt::modeled_peak_bytes);
    py::class_<HFGradientResult>(m,"HFGradientResult")
        .def_readonly("values",&HFGradientResult::values)
        .def_readonly("nuclear",&HFGradientResult::nuclear)
        .def_readonly("one_electron",&HFGradientResult::one_electron)
        .def_readonly("two_electron",&HFGradientResult::two_electron)
        .def_readonly("pulay",&HFGradientResult::pulay)
        .def_readonly("receipt",&HFGradientResult::receipt);
    m.def("plan_hf_lagrangian_gradient",&plan_hf_lagrangian_gradient,
          py::arg("molecule"),py::arg("basis"),py::arg("budget")=ResourceBudget{});
    m.def("compute_hf_lagrangian_gradient",[](const vibeqc::Molecule& mol,const vibeqc::BasisSet& basis,
        const py::array& alpha,const py::array& beta,const py::array& w,const ResourceBudget& budget) {
        for(const auto* array:{&alpha,&beta,&w}) validate_array(*array,basis.nbasis());
        plan_hf_lagrangian_gradient(mol,basis,budget);
        return compute_hf_lagrangian_gradient(mol,basis,copy_matrix(alpha),copy_matrix(beta),copy_matrix(w),budget);
    },py::arg("molecule"),py::arg("basis"),py::arg("density_alpha").noconvert(),
      py::arg("density_beta").noconvert(),py::arg("energy_weighted_density").noconvert(),py::arg("budget")=ResourceBudget{});
    py::class_<RIHFGradientResult,HFGradientResult>(m,"RIHFGradientResult")
        .def_readonly("coulomb",&RIHFGradientResult::coulomb)
        .def_readonly("exchange",&RIHFGradientResult::exchange)
        .def_readonly("coulomb_ri",&RIHFGradientResult::coulomb_ri)
        .def_readonly("exchange_ri",&RIHFGradientResult::exchange_ri)
        .def_readonly("coulomb_metric",&RIHFGradientResult::coulomb_metric)
        .def_readonly("exchange_metric",&RIHFGradientResult::exchange_metric);
    m.def("plan_hf_ri_lagrangian_gradient",&plan_hf_ri_lagrangian_gradient,
          py::arg("molecule"),py::arg("basis"),py::arg("auxiliary"),py::arg("options")=RIOptions{},
          py::arg("fit_j")=true,py::arg("fit_k")=true,py::arg("budget")=ResourceBudget{});
    m.def("compute_hf_ri_lagrangian_gradient",[](const vibeqc::Molecule& mol,const vibeqc::BasisSet& basis,
        const vibeqc::BasisSet& auxiliary,const py::array& alpha,const py::array& beta,const py::array& w,
        const RIOptions& options,bool fit_j,bool fit_k,const ResourceBudget& budget) {
        for(const auto* array:{&alpha,&beta,&w}) validate_array(*array,basis.nbasis());
        plan_hf_ri_lagrangian_gradient(mol,basis,auxiliary,options,fit_j,fit_k,budget);
        return compute_hf_ri_lagrangian_gradient(mol,basis,auxiliary,copy_matrix(alpha),copy_matrix(beta),copy_matrix(w),options,fit_j,fit_k,budget);
    },py::arg("molecule"),py::arg("basis"),py::arg("auxiliary"),py::arg("density_alpha").noconvert(),
      py::arg("density_beta").noconvert(),py::arg("energy_weighted_density").noconvert(),py::arg("options")=RIOptions{},
      py::arg("fit_j")=true,py::arg("fit_k")=true,py::arg("budget")=ResourceBudget{});

}
