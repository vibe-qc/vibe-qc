#include "vibeqc/manta_bloch_one_electron.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
Count check_vectors(const py::array& a) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=2 || a.shape(1)!=3)
        throw py::value_error("MANTA Bloch inputs must be float64 ndarrays of shape (n,3)");
    if(!(a.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA Bloch inputs must be C- or F-contiguous");
    return a.shape(0);
}
md::Matrix copy_vectors(const py::array& a) {
    md::Matrix out(a.shape(0),3);
    const auto* ptr=static_cast<const char*>(a.data());
    if(a.flags() & py::array::c_style) {
        if(a.size()) std::memcpy(out.data(),ptr,a.size()*sizeof(double));
    } else for(Eigen::Index i=0;i<out.rows();++i) for(int j=0;j<3;++j)
        std::memcpy(&out(i,j),ptr+(j*out.rows()+i)*sizeof(double),sizeof(double));
    return out;
}
} // namespace
void bind_manta_bloch_one_electron(py::module_& m) {
    py::class_<BlochOneElectronPlan>(m,"BlochOneElectronPlan")
        .def_readonly("shape",&BlochOneElectronPlan::shape)
        .def_readonly("cell_shape",&BlochOneElectronPlan::cell_shape)
        .def_readonly("candidate_shell_pairs",&BlochOneElectronPlan::candidate_shell_pairs)
        .def_readonly("shell_calls",&BlochOneElectronPlan::shell_calls)
        .def_readonly("primitive_pairs",&BlochOneElectronPlan::primitive_pairs)
        .def_readonly("phase_products",&BlochOneElectronPlan::phase_products)
        .def_readonly("phase_accumulations",&BlochOneElectronPlan::phase_accumulations)
        .def_readonly("integral_elements",&BlochOneElectronPlan::integral_elements)
        .def_readonly("scalar_terms",&BlochOneElectronPlan::scalar_terms)
        .def_readonly("kernel_terms",&BlochOneElectronPlan::kernel_terms)
        .def_readonly("output_bytes",&BlochOneElectronPlan::output_bytes)
        .def_readonly("modeled_peak_bytes",&BlochOneElectronPlan::modeled_peak_bytes);
    py::class_<BlochOneElectronReceipt>(m,"BlochOneElectronReceipt")
        .def_readonly("provider",&BlochOneElectronReceipt::provider)
        .def_readonly("provider_version",&BlochOneElectronReceipt::provider_version)
        .def_readonly("normalization",&BlochOneElectronReceipt::normalization)
        .def_readonly("convention",&BlochOneElectronReceipt::convention)
        .def_readonly("operation",&BlochOneElectronReceipt::operation)
        .def_readonly("source_identity",&BlochOneElectronReceipt::source_identity)
        .def_readonly("candidate_shell_pairs",&BlochOneElectronReceipt::candidate_shell_pairs)
        .def_readonly("skipped_shell_pairs",&BlochOneElectronReceipt::skipped_shell_pairs)
        .def_readonly("shell_calls",&BlochOneElectronReceipt::shell_calls)
        .def_readonly("primitive_pairs",&BlochOneElectronReceipt::primitive_pairs)
        .def_readonly("phase_products",&BlochOneElectronReceipt::phase_products)
        .def_readonly("phase_accumulations",&BlochOneElectronReceipt::phase_accumulations)
        .def_readonly("integral_elements",&BlochOneElectronReceipt::integral_elements)
        .def_readonly("modeled_peak_bytes",&BlochOneElectronReceipt::modeled_peak_bytes)
        .def_readonly("hermiticity_residual",&BlochOneElectronReceipt::hermiticity_residual);
    py::class_<BlochOneElectronResult>(m,"BlochOneElectronResult")
        .def_readonly("plan",&BlochOneElectronResult::plan)
        .def_readonly("receipt",&BlochOneElectronResult::receipt)
        .def_property_readonly("values",[](const BlochOneElectronResult& r) {
            std::vector<py::ssize_t> shape(r.plan.shape.begin(),r.plan.shape.end());
            py::array_t<std::complex<double>> out(shape);
            if(!r.values.empty()) std::memcpy(out.mutable_data(),r.values.data(),r.values.size()*sizeof(std::complex<double>));
            return out;
        })
        .def_property_readonly("cell_values",[](const BlochOneElectronResult& r) {
            std::vector<py::ssize_t> shape(r.plan.cell_shape.begin(),r.plan.cell_shape.end());
            py::array_t<double> out(shape);
            if(!r.cell_values.empty()) std::memcpy(out.mutable_data(),r.cell_values.data(),r.cell_values.size()*sizeof(double));
            return out;
        });
    m.def("plan_bloch_one_electron",&plan_bloch_one_electron,py::arg("basis"),py::arg("operation"),
        py::arg("cells"),py::arg("momenta"),py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
    m.def("compute_bloch_one_electron",[](const vibeqc::BasisSet& b,const std::string& operation,
        const py::array& cells,const py::array& k,std::optional<double> cutoff,const ResourceBudget& budget) {
            const auto nc=check_vectors(cells),nk=check_vectors(k);
            plan_bloch_one_electron(b,operation,nc,nk,cutoff,budget);
            const auto images=copy_vectors(cells),momenta=copy_vectors(k);
            return compute_bloch_one_electron(b,operation,images,momenta,cutoff,budget);
        },py::arg("basis"),py::arg("operation"),py::arg("translations").noconvert(),
        py::arg("momenta").noconvert(),py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
}
