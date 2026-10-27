#include "vibeqc/manta_periodic_support_selection.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace vibeqc::manta;
void bind_manta_periodic_support_selection(py::module_& m) {
    auto o=py::class_<PeriodicSupportSelectionOptions>(m,"PeriodicSupportSelectionOptions").def(py::init<>());
#define O(name) o.def_readwrite(#name,&PeriodicSupportSelectionOptions::name)
    O(maximum_attempts);O(maximum_completion_cube_radius);O(growth_factor);O(maximum_pair_cutoff_bohr);O(maximum_nuclear_cutoff_bohr);O(maximum_reciprocal_energy_cutoff);
#undef O
    auto q=py::class_<PeriodicSupportRequest,PeriodicRHFOptions>(m,"PeriodicSupportRequest").def(py::init<>());
#define Q(name) q.def_readwrite(#name,&PeriodicSupportRequest::name)
    Q(reciprocal_energy_cutoff);Q(exchange_scale);Q(one_electron_accuracy);Q(nuclear_accuracy);Q(exchange_accuracy);
#undef Q
    auto t=py::class_<PeriodicSupportTrial>(m,"PeriodicSupportTrial");
#define T(name) t.def_readonly(#name,&PeriodicSupportTrial::name)
    T(reciprocal_energy_cutoff);T(pair_cutoff_bohr);T(nuclear_cutoff_bohr);T(ao_cube_radius);T(real_cube_radius);T(reciprocal_cube_radius);
    T(overlap_image_rank_requested);T(overlap_image_rank_qualified);T(overlap_image_perturbation_bound);T(overlap_image_rank_radius);
    T(overlap_error);T(kinetic_error);T(kinetic_energy_error);T(nuclear_matrix_error);T(attraction_energy_error);T(nuclear_energy_error);
    T(exchange_matrix_error);T(exchange_energy_error);T(qualified);T(grow_pair);T(grow_nuclear);T(grow_reciprocal);
    T(overlap_identity);T(nuclear_identity);T(one_electron_certificate);T(nuclear_certificate);T(exchange_certificate);T(density_envelope_identity);
    T(modeled_peak_bytes);T(shell_calls);T(primitive_pairs);T(scalar_terms);T(kernel_terms);
#undef T
    auto p=py::class_<PeriodicSupportDomainPlan>(m,"PeriodicSupportDomainPlan");
#define P(name) p.def_readonly(#name,&PeriodicSupportDomainPlan::name)
    P(reciprocal_energy_cutoff);P(pair_cutoff_bohr);P(translation_radius_bohr);P(momenta);P(translation_candidates);P(hartree_candidates);
    P(maximum_exchange_candidates);P(exchange_candidates);P(candidate_visits);P(domain_live_bytes);P(domain_scalar_terms);P(domain_kernel_terms);
    P(modeled_peak_bytes);P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);
#undef P
    auto result=py::class_<PeriodicSupportSelectionResult>(m,"PeriodicSupportSelectionResult");
#define R(name) result.def_readonly(#name,&PeriodicSupportSelectionResult::name)
    R(preparation_scalar_terms);R(preparation_kernel_terms);R(qualified);R(termination);R(certificate_identity);R(retained_bytes);R(modeled_peak_bytes);R(shell_calls);R(primitive_pairs);R(scalar_terms);R(kernel_terms);
#undef R
    result.def_property_readonly("selected",[](const PeriodicSupportSelectionResult& r){return r.selected;})
        .def_property_readonly("policy",[](const PeriodicSupportSelectionResult& r){return r.policy;})
        .def_property_readonly("trials",[](const PeriodicSupportSelectionResult& r){return r.trials;});
    m.def("plan_periodic_support_trial",&plan_periodic_support_trial,py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("request"),py::arg("budget")=ResourceBudget{});
    m.def("select_periodic_supports",&select_periodic_supports,py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("request"),
        py::arg("policy")=PeriodicSupportSelectionOptions{},py::arg("budget")=ResourceBudget{});
}
