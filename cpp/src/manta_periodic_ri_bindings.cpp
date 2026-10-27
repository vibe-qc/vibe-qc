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
    auto a=py::class_<PeriodicRIAccuracyOptions>(m,"PeriodicRIAccuracyOptions").def(py::init<>());
#define A(name) a.def_readwrite(#name,&PeriodicRIAccuracyOptions::name)
    A(j_error);A(energy_error);A(density_l1_limit);A(maximum_attempts);A(density_entry_limits);A(metric_reference_cutoff);
#undef A
    auto c=py::class_<PeriodicRIAccuracyReceipt>(m,"PeriodicRIAccuracyReceipt");
#define C(name) c.def_readonly(#name,&PeriodicRIAccuracyReceipt::name)
    C(enabled);C(qualified);C(j_target);C(energy_target);C(density_l1_limit);C(selected_coupling_error);
    C(j_bound);C(energy_bound);C(attempts);C(shell_calls);C(primitive_products);
    C(coupling_j_bound);C(coupling_energy_bound);C(metric_truncation_certified);C(metric_reference_cutoff);
    C(metric_j_bound);C(metric_energy_bound);C(metric_reference_rank);C(metric_discarded_rank);
    C(metric_discarded_factor_identity);C(metric_reference_identity);
    C(reference_coupling_identity);C(selected_coupling_identity);C(certificate_identity);C(component_density_envelope);C(density_envelope_identity);
#undef C
    auto p=py::class_<PeriodicRIQPlan>(m,"PeriodicRIQPlan");
#define P(name) p.def_readonly(#name,&PeriodicRIQPlan::name)
    P(n_basis);P(n_auxiliary);P(momenta);P(spins);P(maximum_rank);P(mode_block);P(maximum_coupling_passes);P(maximum_coupling_tile_elements);P(explicit_coupling_action);P(coupling_norm_supported);P(coupling_plan);
    P(shell_calls);P(primitive_products);    P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);P(accuracy_selection_passes);
#undef P
    auto i=py::class_<PeriodicRIQMetricInfo>(m,"PeriodicRIQMetricInfo");
#define I(name) i.def_readonly(#name,&PeriodicRIQMetricInfo::name)
    I(assembly_representation_included);I(assembly_integrals_included);I(assembly_fourier_included);I(assembly_prefactor_included);I(assembly_weight_included);I(assembly_phase_included);I(assembly_response);I(assembly_rank_requested);I(assembly_rank_qualified);I(assembly_perturbation_bound);I(assembly_certificate_identity);
    I(inverse_certificate);I(spectrum);I(accuracy);I(rank);I(shell_calls);I(primitive_products);I(cutoff);I(negative_limit);I(minimum_eigenvalue);I(maximum_eigenvalue);
    I(hermiticity_residual);I(zero_transfer);I(metric_source_identity);I(canonical_metric_identity);I(factor_identity);
    I(projector_identity);I(source_identity);
#undef I
    auto r=py::class_<PeriodicRIQReceipt>(m,"PeriodicRIQReceipt");
#define R(name) r.def_readonly(#name,&PeriodicRIQReceipt::name)
    R(retained_j_roundoff_evaluated);R(retained_k_roundoff_evaluated);R(retained_j_roundoff_bound);R(retained_k_roundoff_bound);R(retained_energy_roundoff_bound);
    R(provider);R(algorithm);R(kernel);R(convention);R(coulomb_contraction);R(source_identity);R(projector_identity);R(density_identity);
    R(weight_identity);R(snapshot_identity);R(source_frame_identity);R(target_frame_identity);R(retained_rank);
    R(coupling_passes);R(shell_calls);R(primitive_products);R(modeled_peak_bytes);R(coulomb_energy);
    R(physical_omega);R(zero_transfer);R(physical_zero_included);
    R(weighted_input_roundoff_evaluated);R(weighted_input_roundoff_bound);R(coupling_forward_input_error_included);R(paired_input_roundoff_included);
    R(forward_metric_roundoff_evaluated);R(forward_metric_coefficient_roundoff_bound);R(forward_metric_energy_roundoff_bound);R(paired_roundoff_evaluated);R(paired_j_roundoff_bound);R(paired_energy_roundoff_bound);
    R(coupling_roundoff_evaluated);R(coupling_forward_roundoff_bound);R(coupling_adjoint_roundoff_bound);R(metric_roundoff_evaluated);R(metric_j_roundoff_evaluated);R(metric_j_roundoff_bound);R(metric_projection_roundoff_bound);R(metric_coefficient_roundoff_bound);R(metric_energy_roundoff_bound);
    R(reference_coupling_identity);R(coupling_error_bound);R(metric_inverse_norm_bound);R(forward_error_l2_bound);
    R(approximation_j_bound);R(approximation_energy_bound);R(coupling_multipole_partition_builds);R(coupling_multipole_partition_reuses);R(coupling_multipole_partition_bypasses);R(coupling_multipole_trials);R(coupling_multipole_images);R(coupling_multipole_tree_nodes);R(coupling_multipole_far_blocks);R(coupling_multipole_translations);R(coupling_multipole_moment_panels);R(coupling_multipole_envelope_evaluations);R(coupling_multipole_reused_moment_panels);R(coupling_multipole_reused_extents);R(coupling_multipole_global_pairs);R(coupling_multipole_global_pair_reuses);R(coupling_multipole_global_auxiliary);R(coupling_multipole_global_projections);R(coupling_multipole_global_auxiliary_projections);R(coupling_multipole_pair_groups);R(coupling_multipole_compressed_pairs);R(coupling_multipole_compressed_nodes);R(coupling_multipole_action_products);R(coupling_multipole_action_translations);
    R(accuracy_certified);R(accuracy_density_l1_bound);R(metric_rhs_norm);R(metric_residual_norm);R(hermiticity_residual);R(has_j);R(has_k);R(metric_residual_evaluated);
#undef R
    py::class_<PeriodicRIQResult>(m,"PeriodicRIQResult")
        .def_readonly("receipt",&PeriodicRIQResult::receipt)
        .def_property_readonly("J",[](const PeriodicRIQResult& r){return owned(r.J,
            {py::ssize_t(r.receipt.has_j?r.momenta:0),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});})
        .def_property_readonly("J_roundoff_bounds",[](const PeriodicRIQResult& r) {
            py::array_t<double> out({py::ssize_t(r.receipt.retained_j_roundoff_evaluated?r.momenta:0),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});
            std::copy(r.J_roundoff_bounds.begin(),r.J_roundoff_bounds.end(),out.mutable_data());return out;})
        .def_property_readonly("K_roundoff_bounds",[](const PeriodicRIQResult& r) {
            py::array_t<double> out({py::ssize_t(r.receipt.retained_k_roundoff_evaluated?r.spins:0),py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});
            std::copy(r.K_roundoff_bounds.begin(),r.K_roundoff_bounds.end(),out.mutable_data());return out;})
        .def_property_readonly("K",[](const PeriodicRIQResult& r){return owned(r.K,
            {py::ssize_t(r.receipt.has_k?r.spins:0),py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});});
    m.def("plan_periodic_ri_q",[](const PeriodicGDFTiles& s,Count spins,const RIOptions& o,const ResourceBudget& b,const PeriodicRIAccuracyOptions& a){return plan_periodic_ri_q(s,spins,o,b,a);},py::arg("source"),py::arg("spins")=2,
        py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{},py::arg("accuracy")=PeriodicRIAccuracyOptions{});
    py::class_<PeriodicRIQBuilder,std::shared_ptr<PeriodicRIQBuilder>>(m,"PeriodicRIQBuilder")
        .def(py::init([](std::shared_ptr<PeriodicGDFTiles> s,Count spins,const RIOptions& options,const ResourceBudget& budget,const PeriodicRIAccuracyOptions& accuracy) {
            return std::make_shared<PeriodicRIQBuilder>(std::move(s),spins,options,budget,accuracy);
        }),py::arg("source"),py::arg("spins")=2,py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{},py::arg("accuracy")=PeriodicRIAccuracyOptions{})
        .def_property_readonly("plan",&PeriodicRIQBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("metric_info",&PeriodicRIQBuilder::metric_info,py::return_value_policy::copy)
        .def_property_readonly("metric",[](const PeriodicRIQBuilder& b){return owned_matrix(b.metric());})
        .def("inverse_action",&PeriodicRIQBuilder::inverse_action,py::arg("budget")=ResourceBudget{})
        .def_property_readonly("factor",[](const PeriodicRIQBuilder& b){return owned_matrix(b.factor());})
        .def_property_readonly("projector",[](const PeriodicRIQBuilder& b){return owned_matrix(b.projector());})
        .def_property_readonly("metric_discarded_factor",[](const PeriodicRIQBuilder& b){return owned_matrix(b.metric_discarded_factor());})
        .def("build",[](const PeriodicRIQBuilder& b,const py::array& d,const py::array& w,bool j,bool k,bool audit) {
            b.validate_request(j,k);shape(b.plan(),d,w); // Before numerical copies.
            std::vector<Complex> density(d.size());std::vector<double> weights(w.size());
            const auto* raw=static_cast<const char*>(d.data());Count out=0;
            for(py::ssize_t s=0;s<d.shape(0);++s)for(py::ssize_t q=0;q<d.shape(1);++q)
                for(py::ssize_t i=0;i<d.shape(2);++i)for(py::ssize_t n=0;n<d.shape(3);++n)
                    std::memcpy(&density[out++],raw+s*d.strides(0)+q*d.strides(1)+i*d.strides(2)+n*d.strides(3),sizeof(Complex));
            std::memcpy(weights.data(),w.data(),weights.size()*sizeof(double));
            py::gil_scoped_release release;
            return b.build(density,weights,j,k,audit);
        },py::arg("densities").noconvert(),py::arg("weights").noconvert(),py::arg("make_j")=true,py::arg("make_k")=true,py::arg("audit_metric_arithmetic")=false);
}
