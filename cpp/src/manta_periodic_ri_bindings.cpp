#include "vibeqc/manta_periodic_ri.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
py::array_t<Complex> owned(const std::vector<Complex>& values,std::vector<py::ssize_t> shape) {
    py::array_t<Complex> out(shape);
    if(!values.empty()) std::memcpy(out.mutable_data(),values.data(),values.size()*sizeof(Complex));
    return out;
}
py::array_t<Complex> owned_matrix(const Eigen::MatrixXcd& values) {
    py::array_t<Complex> out({values.rows(),values.cols()});
    auto a=out.mutable_unchecked<2>();
    for(Eigen::Index i=0;i<values.rows();++i)for(Eigen::Index j=0;j<values.cols();++j)a(i,j)=values(i,j);
    return out;
}
void shape(const PeriodicRIQPlan& p,const py::array& d,const py::array& w) {
    if(!d.dtype().is(py::dtype::of<Complex>()) || d.ndim()!=4 || d.shape(0)!=py::ssize_t(p.spins) ||
       d.shape(1)!=py::ssize_t(p.momenta) || d.shape(2)!=py::ssize_t(p.n_basis) || d.shape(3)!=py::ssize_t(p.n_basis))
        throw py::value_error("MANTA periodic RI densities require complex128 (spin,ket,AO,AO) arrays");
    if(!w.dtype().is(py::dtype::of<double>()) || w.ndim()!=1 || w.shape(0)!=py::ssize_t(p.momenta))
        throw py::value_error("MANTA periodic RI weights require float64 (ket,) arrays");
    if(!(d.flags() & (py::array::c_style|py::array::f_style)) || !(w.flags() & (py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic RI inputs must be C- or F-contiguous");
}
}

void bind_manta_periodic_ri(py::module_& m) {
    auto p=py::class_<PeriodicRIQPlan>(m,"PeriodicRIQPlan");
#define P(name) p.def_readonly(#name,&PeriodicRIQPlan::name)
    P(n_basis);P(n_auxiliary);P(momenta);P(spins);P(maximum_rank);P(mode_block);P(maximum_coupling_passes);
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto i=py::class_<PeriodicRIQMetricInfo>(m,"PeriodicRIQMetricInfo");
#define I(name) i.def_readonly(#name,&PeriodicRIQMetricInfo::name)
    I(rank);I(shell_calls);I(primitive_products);I(cutoff);I(negative_limit);I(minimum_eigenvalue);I(maximum_eigenvalue);
    I(hermiticity_residual);I(zero_transfer);I(metric_source_identity);I(canonical_metric_identity);I(factor_identity);
    I(projector_identity);I(source_identity);
#undef I
    auto r=py::class_<PeriodicRIQReceipt>(m,"PeriodicRIQReceipt");
#define R(name) r.def_readonly(#name,&PeriodicRIQReceipt::name)
    R(provider);R(algorithm);R(kernel);R(convention);R(source_identity);R(projector_identity);R(density_identity);
    R(weight_identity);R(snapshot_identity);R(source_frame_identity);R(target_frame_identity);R(retained_rank);
    R(coupling_passes);R(shell_calls);R(primitive_products);R(modeled_peak_bytes);R(coulomb_energy);
    R(physical_omega);R(zero_transfer);R(physical_zero_included);
    R(metric_rhs_norm);R(metric_residual_norm);R(hermiticity_residual);R(has_j);R(has_k);R(metric_residual_evaluated);
#undef R
    py::class_<PeriodicRIQResult>(m,"PeriodicRIQResult")
        .def_readonly("receipt",&PeriodicRIQResult::receipt)
        .def_property_readonly("J",[](const PeriodicRIQResult& r){return owned(r.J,
            {py::ssize_t(r.receipt.has_j?r.momenta:0),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});})
        .def_property_readonly("K",[](const PeriodicRIQResult& r){return owned(r.K,
            {py::ssize_t(r.receipt.has_k?r.spins:0),py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});});
    m.def("plan_periodic_ri_q",[](const PeriodicGDFTiles& s,Count spins,const RIOptions& o,const ResourceBudget& b){return plan_periodic_ri_q(s,spins,o,b);},py::arg("source"),py::arg("spins")=2,
        py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicRIQBuilder>(m,"PeriodicRIQBuilder")
        .def(py::init([](std::shared_ptr<PeriodicGDFTiles> s,Count spins,const RIOptions& options,const ResourceBudget& budget) {
            return PeriodicRIQBuilder(std::move(s),spins,options,budget);
        }),py::arg("source"),py::arg("spins")=2,py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",&PeriodicRIQBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("metric_info",&PeriodicRIQBuilder::metric_info,py::return_value_policy::copy)
        .def_property_readonly("metric",[](const PeriodicRIQBuilder& b){return owned_matrix(b.metric());})
        .def_property_readonly("factor",[](const PeriodicRIQBuilder& b){return owned_matrix(b.factor());})
        .def_property_readonly("projector",[](const PeriodicRIQBuilder& b){return owned_matrix(b.projector());})
        .def("build",[](const PeriodicRIQBuilder& b,const py::array& d,const py::array& w,bool j,bool k) {
            b.validate_request(j,k);shape(b.plan(),d,w); // Before numerical copies.
            std::vector<Complex> density(d.size());std::vector<double> weights(w.size());
            const auto* raw=static_cast<const char*>(d.data());Count out=0;
            for(py::ssize_t s=0;s<d.shape(0);++s)for(py::ssize_t q=0;q<d.shape(1);++q)
                for(py::ssize_t i=0;i<d.shape(2);++i)for(py::ssize_t n=0;n<d.shape(3);++n)
                    std::memcpy(&density[out++],raw+s*d.strides(0)+q*d.strides(1)+i*d.strides(2)+n*d.strides(3),sizeof(Complex));
            std::memcpy(weights.data(),w.data(),weights.size()*sizeof(double));
            py::gil_scoped_release release;
            return b.build(density,weights,j,k);
        },py::arg("densities").noconvert(),py::arg("weights").noconvert(),py::arg("make_j")=true,py::arg("make_k")=true);
}
