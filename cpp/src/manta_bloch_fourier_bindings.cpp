#include "vibeqc/manta_bloch_fourier.hpp"
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
void bind_manta_bloch_fourier(py::module_& m) {
    py::class_<BlochFourierPlan>(m,"BlochFourierPlan")
        .def_readonly("shape",&BlochFourierPlan::shape)
        .def_readonly("cells",&BlochFourierPlan::cells)
        .def_readonly("candidate_shell_pairs",&BlochFourierPlan::candidate_shell_pairs)
        .def_readonly("shell_calls",&BlochFourierPlan::shell_calls)
        .def_readonly("primitive_pairs",&BlochFourierPlan::primitive_pairs)
        .def_readonly("vector_primitive_products",&BlochFourierPlan::vector_primitive_products)
        .def_readonly("phase_products",&BlochFourierPlan::phase_products)
        .def_readonly("phase_accumulations",&BlochFourierPlan::phase_accumulations)
        .def_readonly("scalar_terms",&BlochFourierPlan::scalar_terms)
        .def_readonly("kernel_terms",&BlochFourierPlan::kernel_terms)
        .def_readonly("output_bytes",&BlochFourierPlan::output_bytes)
        .def_readonly("modeled_peak_bytes",&BlochFourierPlan::modeled_peak_bytes);
    py::class_<BlochFourierReceipt>(m,"BlochFourierReceipt")
        .def_readonly("provider",&BlochFourierReceipt::provider)
        .def_readonly("provider_version",&BlochFourierReceipt::provider_version)
        .def_readonly("normalization",&BlochFourierReceipt::normalization)
        .def_readonly("convention",&BlochFourierReceipt::convention)
        .def_readonly("source_identity",&BlochFourierReceipt::source_identity)
        .def_readonly("candidate_shell_pairs",&BlochFourierReceipt::candidate_shell_pairs)
        .def_readonly("skipped_shell_pairs",&BlochFourierReceipt::skipped_shell_pairs)
        .def_readonly("shell_calls",&BlochFourierReceipt::shell_calls)
        .def_readonly("primitive_pairs",&BlochFourierReceipt::primitive_pairs)
        .def_readonly("vector_primitive_products",&BlochFourierReceipt::vector_primitive_products)
        .def_readonly("phase_products",&BlochFourierReceipt::phase_products)
        .def_readonly("phase_accumulations",&BlochFourierReceipt::phase_accumulations)
        .def_readonly("integral_elements",&BlochFourierReceipt::integral_elements)
        .def_readonly("modeled_peak_bytes",&BlochFourierReceipt::modeled_peak_bytes);
    py::class_<BlochFourierResult>(m,"BlochFourierResult")
        .def_readonly("shape",&BlochFourierResult::shape)
        .def_readonly("receipt",&BlochFourierResult::receipt)
        .def_property_readonly("values",[](const BlochFourierResult& r) {
            std::vector<py::ssize_t> shape(r.shape.begin(),r.shape.end());
            py::array_t<std::complex<double>> out(shape);
            if(!r.values.empty()) std::memcpy(out.mutable_data(),r.values.data(),r.values.size()*sizeof(std::complex<double>));
            return out;
        });
    m.def("plan_bloch_fourier",&plan_bloch_fourier,py::arg("basis"),py::arg("vectors"),py::arg("cells"),
        py::arg("momenta"),py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
    m.def("compute_bloch_fourier_panel",[](const vibeqc::BasisSet& b,const py::array& q,const py::array& cells,
        const py::array& k,std::optional<double> cutoff,const ResourceBudget& budget) {
            const auto nq=check_vectors(q),nc=check_vectors(cells),nk=check_vectors(k);
            plan_bloch_fourier(b,nq,nc,nk,cutoff,budget);
            const auto wave=copy_vectors(q),images=copy_vectors(cells),momenta=copy_vectors(k);
            return compute_bloch_fourier_panel(b,wave,images,momenta,cutoff,budget);
        },py::arg("basis"),py::arg("wave_vectors").noconvert(),py::arg("translations").noconvert(),
        py::arg("momenta").noconvert(),py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{});
}
