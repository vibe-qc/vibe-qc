#include "vibeqc/manta_ri_gradient.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
void bind_manta_ri_gradient(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<RIJGradientPlan>(m,"RIJGradientPlan")
        .def_readonly("n_atoms",&RIJGradientPlan::n_atoms)
        .def_readonly("n_basis",&RIJGradientPlan::n_basis)
        .def_readonly("n_auxiliary",&RIJGradientPlan::n_auxiliary)
        .def_readonly("shell_calls",&RIJGradientPlan::shell_calls)
        .def_readonly("primitive_evaluations_upper_bound",&RIJGradientPlan::primitive_evaluations_upper_bound)
        .def_readonly("scalar_terms",&RIJGradientPlan::scalar_terms)
        .def_readonly("kernel_terms",&RIJGradientPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&RIJGradientPlan::modeled_peak_bytes);
    py::class_<RIJGradientReceipt>(m,"RIJGradientReceipt")
        .def_readonly("provider",&RIJGradientReceipt::provider)
        .def_readonly("provider_version",&RIJGradientReceipt::provider_version)
        .def_readonly("algorithm",&RIJGradientReceipt::algorithm)
        .def_readonly("source_identity",&RIJGradientReceipt::source_identity)
        .def_readonly("ri_source_identity",&RIJGradientReceipt::ri_source_identity)
        .def_readonly("density_identity",&RIJGradientReceipt::density_identity)
        .def_readonly("kernel",&RIJGradientReceipt::kernel)
        .def_readonly("omega",&RIJGradientReceipt::omega)
        .def_readonly("coulomb_energy",&RIJGradientReceipt::coulomb_energy)
        .def_readonly("metric_rhs_norm",&RIJGradientReceipt::metric_rhs_norm)
        .def_readonly("metric_residual_norm",&RIJGradientReceipt::metric_residual_norm)
        .def_readonly("metric_residual_evaluated",&RIJGradientReceipt::metric_residual_evaluated)
        .def_readonly("metric_shell_calls",&RIJGradientReceipt::metric_shell_calls)
        .def_readonly("coupling_shell_calls",&RIJGradientReceipt::coupling_shell_calls)
        .def_readonly("metric_derivative_shell_calls",&RIJGradientReceipt::metric_derivative_shell_calls)
        .def_readonly("coupling_derivative_shell_calls",&RIJGradientReceipt::coupling_derivative_shell_calls)
        .def_readonly("primitive_integrals_evaluated",&RIJGradientReceipt::primitive_integrals_evaluated)
        .def_readonly("integral_elements_consumed",&RIJGradientReceipt::integral_elements_consumed)
        .def_readonly("modeled_peak_bytes",&RIJGradientReceipt::modeled_peak_bytes);
    py::class_<RIJGradientResult>(m,"RIJGradientResult")
        .def_readonly("values",&RIJGradientResult::values)
        .def_readonly("coupling",&RIJGradientResult::coupling)
        .def_readonly("metric",&RIJGradientResult::metric)
        .def_readonly("metric_response",&RIJGradientResult::metric_response)
        .def_readonly("receipt",&RIJGradientResult::receipt);
    m.def("plan_ri_coulomb_gradient",&plan_ri_coulomb_gradient,
          py::arg("molecule"),py::arg("orbital"),py::arg("auxiliary"),py::arg("options")=RIOptions{},
          py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    m.def("compute_ri_coulomb_gradient",[](const vibeqc::Molecule& mol,const vibeqc::BasisSet& orbital,
        const vibeqc::BasisSet& auxiliary,const py::array& density,const RIOptions& options,
        const KernelRequest& request,const ResourceBudget& budget) {
        const auto n=static_cast<py::ssize_t>(orbital.nbasis());
        if(!density.dtype().is(py::dtype::of<double>()) || density.ndim()!=2 || density.shape(0)!=n || density.shape(1)!=n)
            throw py::value_error("MANTA RI Coulomb gradient density must be a square float64 ndarray matching the orbital basis");
        if(!(density.flags() & (py::array::c_style | py::array::f_style)))
            throw py::value_error("MANTA RI Coulomb gradient density must be C- or F-contiguous");
        plan_ri_coulomb_gradient(mol,orbital,auxiliary,options,request,budget);
        Eigen::MatrixXd d(n,n);
        const auto* data=static_cast<const char*>(density.data());
        if(density.flags() & py::array::f_style) std::memcpy(d.data(),data,density.size()*sizeof(double));
        else for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<n;++j)
            std::memcpy(&d(i,j),data+(i*n+j)*sizeof(double),sizeof(double));
        return compute_ri_coulomb_gradient(mol,orbital,auxiliary,d,options,request,budget);
    },py::arg("molecule"),py::arg("orbital"),py::arg("auxiliary"),py::arg("density").noconvert(),
      py::arg("options")=RIOptions{},py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    py::class_<RIKGradientPlan,RIJGradientPlan>(m,"RIKGradientPlan")
        .def_readonly("mode_block",&RIKGradientPlan::mode_block)
        .def_readonly("auxiliary_blocks",&RIKGradientPlan::auxiliary_blocks)
        .def_readonly("maximum_coupling_passes",&RIKGradientPlan::maximum_coupling_passes)
        .def_readonly("maximum_derivative_passes",&RIKGradientPlan::maximum_derivative_passes);
    py::class_<RIKGradientReceipt>(m,"RIKGradientReceipt")
        .def_readonly("provider",&RIKGradientReceipt::provider)
        .def_readonly("provider_version",&RIKGradientReceipt::provider_version)
        .def_readonly("algorithm",&RIKGradientReceipt::algorithm)
        .def_readonly("source_identity",&RIKGradientReceipt::source_identity)
        .def_readonly("ri_source_identity",&RIKGradientReceipt::ri_source_identity)
        .def_readonly("density_identity",&RIKGradientReceipt::density_identity)
        .def_readonly("density_identities",&RIKGradientReceipt::density_identities)
        .def_readonly("kernel",&RIKGradientReceipt::kernel)
        .def_readonly("omega",&RIKGradientReceipt::omega)
        .def_readonly("exchange_energy",&RIKGradientReceipt::exchange_energy)
        .def_readonly("metric_shell_calls",&RIKGradientReceipt::metric_shell_calls)
        .def_readonly("coupling_shell_calls",&RIKGradientReceipt::coupling_shell_calls)
        .def_readonly("metric_derivative_shell_calls",&RIKGradientReceipt::metric_derivative_shell_calls)
        .def_readonly("coupling_derivative_shell_calls",&RIKGradientReceipt::coupling_derivative_shell_calls)
        .def_readonly("coupling_passes",&RIKGradientReceipt::coupling_passes)
        .def_readonly("coupling_derivative_passes",&RIKGradientReceipt::coupling_derivative_passes)
        .def_readonly("primitive_integrals_evaluated",&RIKGradientReceipt::primitive_integrals_evaluated)
        .def_readonly("integral_elements_consumed",&RIKGradientReceipt::integral_elements_consumed)
        .def_readonly("modeled_peak_bytes",&RIKGradientReceipt::modeled_peak_bytes);
    py::class_<RIKGradientResult>(m,"RIKGradientResult")
        .def_readonly("values",&RIKGradientResult::values)
        .def_readonly("coupling",&RIKGradientResult::coupling)
        .def_readonly("metric",&RIKGradientResult::metric)
        .def_readonly("metric_response",&RIKGradientResult::metric_response)
        .def_readonly("receipt",&RIKGradientResult::receipt);
    m.def("plan_ri_exchange_gradient",&plan_ri_exchange_gradient,
          py::arg("molecule"),py::arg("orbital"),py::arg("auxiliary"),py::arg("options")=RIOptions{},
          py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    m.def("compute_ri_exchange_gradient",[](const vibeqc::Molecule& mol,const vibeqc::BasisSet& orbital,
        const vibeqc::BasisSet& auxiliary,const py::array& alpha,const py::array& beta,const RIOptions& options,
        const KernelRequest& request,const ResourceBudget& budget) {
        const auto n=static_cast<py::ssize_t>(orbital.nbasis());
        for(const auto* d:{&alpha,&beta}) {
            if(!d->dtype().is(py::dtype::of<double>()) || d->ndim()!=2 || d->shape(0)!=n || d->shape(1)!=n)
                throw py::value_error("MANTA RI exchange gradient requires square float64 ndarrays matching the orbital basis");
            if(!(d->flags() & (py::array::c_style | py::array::f_style)))
                throw py::value_error("MANTA RI exchange gradient densities must be C- or F-contiguous");
        }
        plan_ri_exchange_gradient(mol,orbital,auxiliary,options,request,budget);
        auto copy=[&](const py::array& array) {
            Eigen::MatrixXd d(n,n);
            const auto* data=static_cast<const char*>(array.data());
            if(array.flags() & py::array::f_style) std::memcpy(d.data(),data,array.size()*sizeof(double));
            else for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<n;++j)
                std::memcpy(&d(i,j),data+(i*n+j)*sizeof(double),sizeof(double));
            return d;
        };
        const auto da=copy(alpha),db=copy(beta);
        return compute_ri_exchange_gradient(mol,orbital,auxiliary,da,db,options,request,budget);
    },py::arg("molecule"),py::arg("orbital"),py::arg("auxiliary"),py::arg("density_alpha").noconvert(),
      py::arg("density_beta").noconvert(),py::arg("options")=RIOptions{},py::arg("request")=KernelRequest{},
      py::arg("budget")=ResourceBudget{});

}
