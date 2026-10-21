#include "vibeqc/manta_periodic_ri_fitting_bounds.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void shape(const py::array& a,int ndim) {
    if(!a.dtype().is(py::dtype::of<std::complex<double>>()) || a.ndim()!=ndim ||
       !(a.flags()&py::array::c_style))throw py::value_error("MANTA fitting densities require C-contiguous complex128 spin/k/AO/AO arrays");
}
}
void bind_manta_periodic_ri_fitting_bounds(py::module_& m) {
    py::class_<PeriodicRIFittingReferenceOptions>(m,"PeriodicRIFittingReferenceOptions").def(py::init<>())
        .def_readwrite("selective_refinement",&PeriodicRIFittingReferenceOptions::selective_refinement)
        .def_readwrite("initial",&PeriodicRIFittingReferenceOptions::initial)
        .def_readwrite("budget",&PeriodicRIFittingReferenceOptions::budget)
        .def_readwrite("maximum_refinements",&PeriodicRIFittingReferenceOptions::maximum_refinements)
        .def_readwrite("maximum_cube_radius",&PeriodicRIFittingReferenceOptions::maximum_cube_radius)
        .def_readwrite("growth_factor",&PeriodicRIFittingReferenceOptions::growth_factor)
        .def_readwrite("maximum_pair_cutoff_bohr",&PeriodicRIFittingReferenceOptions::maximum_pair_cutoff_bohr)
        .def_readwrite("reference_fraction",&PeriodicRIFittingReferenceOptions::reference_fraction);
    m.def("validate_periodic_ri_reference_options",&validate_periodic_ri_reference_options);
    auto domain=py::class_<PeriodicRIFittingDomainOptions>(m,"PeriodicRIFittingDomainOptions").def(py::init<>());
#define D(n) domain.def_readwrite(#n,&PeriodicRIFittingDomainOptions::n)
    D(budget);D(initial_pair_cutoff_bohr);D(initial_auxiliary_cutoff_bohr);D(initial_reciprocal_cube_radius);
    D(maximum_refinements);D(maximum_reciprocal_cube_radius);D(growth_factor);D(maximum_pair_cutoff_bohr);D(maximum_auxiliary_cutoff_bohr);
#undef D
    m.def("validate_periodic_ri_domain_options",&validate_periodic_ri_domain_options);
    auto auxiliary=py::class_<PeriodicRIFittingAuxiliaryOptions>(m,"PeriodicRIFittingAuxiliaryOptions").def(py::init<>());
#define A(n) auxiliary.def_readwrite(#n,&PeriodicRIFittingAuxiliaryOptions::n)
    A(initial);A(generation_limits);A(maximum_refinements);A(budget);A(prepared);
#undef A
    m.def("validate_periodic_ri_auxiliary_options",&validate_periodic_ri_auxiliary_options);
    m.def("prepare_periodic_ri_auxiliary_selection",&prepare_periodic_ri_auxiliary_selection,
        py::arg("orbital"),py::arg("seed"),py::arg("system"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("periodic_ri_fitting_search_budget",&periodic_ri_fitting_search_budget);
    py::class_<PeriodicRIFittingAccuracyOptions>(m,"PeriodicRIFittingAccuracyOptions").def(py::init<>())
        .def_readwrite("reference",&PeriodicRIFittingAccuracyOptions::reference)
        .def_readwrite("reference_selection",&PeriodicRIFittingAccuracyOptions::reference_selection)
        .def_readwrite("domain_selection",&PeriodicRIFittingAccuracyOptions::domain_selection)
        .def_readwrite("auxiliary_selection",&PeriodicRIFittingAccuracyOptions::auxiliary_selection)
        .def_readwrite("j_error",&PeriodicRIFittingAccuracyOptions::j_error)
        .def_readwrite("energy_error",&PeriodicRIFittingAccuracyOptions::energy_error)
        .def_readwrite("probe_batch_size",&PeriodicRIFittingAccuracyOptions::probe_batch_size)
        .def_readwrite("maximum_probes",&PeriodicRIFittingAccuracyOptions::maximum_probes)
        .def_readwrite("maximum_metric_refinements",&PeriodicRIFittingAccuracyOptions::maximum_metric_refinements)
        .def_readwrite("metric_cutoff_reduction",&PeriodicRIFittingAccuracyOptions::metric_cutoff_reduction)
        .def_readwrite("metric_action_j_error",&PeriodicRIFittingAccuracyOptions::metric_action_j_error)
        .def_readwrite("metric_action_energy_error",&PeriodicRIFittingAccuracyOptions::metric_action_energy_error)
        .def_readwrite("metric_rank_perturbation",&PeriodicRIFittingAccuracyOptions::metric_rank_perturbation)
        .def_readwrite("minimum_metric_cutoff",&PeriodicRIFittingAccuracyOptions::minimum_metric_cutoff);
    auto o=py::class_<PeriodicRIFittingBoundOptions>(m,"PeriodicRIFittingBoundOptions").def(py::init<>());
#define O(n) o.def_readwrite(#n,&PeriodicRIFittingBoundOptions::n)
    O(metric_action_j_error);O(metric_action_energy_error);O(metric_rank_perturbation);O(density_l1_limit);O(density_entry_limits);O(j_error);O(energy_error);O(maximum_probes);O(probe_batch_size);
#undef O
    auto p=py::class_<PeriodicRIFittingBoundPlan>(m,"PeriodicRIFittingBoundPlan");
#define P(n) p.def_readonly(#n,&PeriodicRIFittingBoundPlan::n)
    P(n_basis);P(momenta);P(spins);P(maximum_probes);P(maximum_trials);P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicRIFittingBoundReceipt>(m,"PeriodicRIFittingBoundReceipt");
#define R(n) r.def_readonly(#n,&PeriodicRIFittingBoundReceipt::n)
    R(qualification);R(ri_source_identity);R(reference_source_identity);R(pair_source_identity);R(certificate_identity);
    R(reference_batches);R(reference_source_passes);R(reference_shell_calls);R(reference_majorant_pairs);
    R(probes);R(density_l1_limit);R(j_error_bound);R(energy_error_bound);R(finite_difference_j_bound);R(reference_remainder_j_bound);
    R(finite_difference_energy_bound);R(reference_remainder_energy_bound);
    R(reference_approximation_j_bound);R(reference_approximation_energy_bound);
    R(reference_real_domain_j_bound);R(reference_real_domain_energy_bound);
    R(reference_reciprocal_domain_j_bound);R(reference_reciprocal_domain_energy_bound);
    R(reference_pair_image_j_bound);R(reference_pair_image_energy_bound);

    R(metric_action_requested);R(metric_action_qualified);R(metric_action_j_target);R(metric_action_energy_target);
    R(metric_action_j_bound);R(metric_action_energy_bound);R(metric_action_columns);R(metric_action_shell_calls);R(metric_action_primitive_products);
    R(metric_rank_perturbation);R(metric_spectrum);R(metric_rank_qualified);R(roundoff_included);R(component_density_envelope);R(targets_requested);
#undef R
    m.def("plan_periodic_ri_fitting_bound",&PeriodicRIFittingBoundSource::plan_source,py::arg("ri"),py::arg("reference"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicRIFittingBoundSource,std::shared_ptr<PeriodicRIFittingBoundSource>>(m,"PeriodicRIFittingBoundSource")
        .def(py::init([](std::shared_ptr<const PeriodicRIQBuilder> ri,std::shared_ptr<const PeriodicBupoSource> reference,
            const PeriodicRIFittingBoundOptions& options,const ResourceBudget& budget) {
            if(!ri || !reference)throw py::value_error("MANTA fitting bounds require native sources");
            PeriodicRIFittingBoundSource::plan_source(*ri,*reference,options,budget); // before vector snapshot
            const auto opts=options;const auto limits=budget;py::gil_scoped_release release;
            return std::make_shared<PeriodicRIFittingBoundSource>(std::move(ri),std::move(reference),opts,limits);
        }),py::arg("ri"),py::arg("reference"),py::arg("options"),py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const PeriodicRIFittingBoundSource& s){return PeriodicRIFittingBoundPlan(s.plan());})
        .def_property_readonly("receipt",[](const PeriodicRIFittingBoundSource& s){return PeriodicRIFittingBoundReceipt(s.receipt());})
        .def_property_readonly("j_error_bounds",[](const PeriodicRIFittingBoundSource& s) {
            const auto& p=s.plan();py::array_t<double> out({p.momenta,p.n_basis,p.n_basis});
            if(!s.j_error_bounds().empty())std::memcpy(out.mutable_data(),s.j_error_bounds().data(),8*s.j_error_bounds().size());return out;
        })
        .def_property_readonly("metric_action_j_bounds",[](const PeriodicRIFittingBoundSource& s){return s.metric_action_j_bounds();})
        .def("build",[](const PeriodicRIFittingBoundSource& s,const py::array& d,const py::array& weights) {
            shape(d,4);const auto& p=s.plan();
            if(d.shape(0)!=static_cast<py::ssize_t>(p.spins) || d.shape(1)!=static_cast<py::ssize_t>(p.momenta) ||
               d.shape(2)!=static_cast<py::ssize_t>(p.n_basis) || d.shape(3)!=static_cast<py::ssize_t>(p.n_basis))throw py::value_error("MANTA fitting density shape mismatch");
            if(!weights.dtype().is(py::dtype::of<double>()) || weights.ndim()!=1 ||
               !(weights.flags()&py::array::c_style) || weights.shape(0)!=static_cast<py::ssize_t>(p.momenta))
                throw py::value_error("MANTA fitting weights require one contiguous float64 momentum vector");
            const auto* raw=static_cast<const double*>(weights.data());const std::vector<double> w(raw,raw+weights.size());const auto* data=static_cast<const std::complex<double>*>(d.data());std::vector<std::complex<double>> copy(data,data+d.size());
            py::gil_scoped_release release;return s.build(copy,w);
        },py::arg("densities").noconvert(),py::arg("weights").noconvert());
}
