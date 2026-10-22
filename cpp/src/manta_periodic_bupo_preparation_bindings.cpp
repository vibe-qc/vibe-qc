#include "vibeqc/manta_periodic_bupo_preparation.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;

void bind_manta_periodic_bupo_preparation(py::module_& m) {
    using namespace vibeqc::manta;
    auto o=py::class_<PeriodicBupoSourceOptions>(m,"PeriodicBupoSourceOptions").def(py::init<>());
#define O(n) o.def_readwrite(#n,&PeriodicBupoHFSourceOptions::n)
    O(pairs);O(bubbles);O(bupo);O(pair_cube_radius);O(real_cube_radius);O(reciprocal_cube_radius);O(near_cube_radius);O(eta);
#undef O
    m.attr("PeriodicBupoHFSourceOptions")=m.attr("PeriodicBupoSourceOptions");
    auto p=py::class_<PeriodicBupoSourcePreparationPlan>(m,"PeriodicBupoSourcePreparationPlan");
#define P(n) p.def_readonly(#n,&PeriodicBupoSourcePreparationPlan::n)
    P(pairs);P(bubbles);P(ewald);P(bupo);P(generator_live_bytes);P(generator_scalar_terms);P(generator_kernel_terms);
    P(source_live_bytes);P(source_shell_calls);P(source_primitive_pairs);P(source_scalar_terms);P(source_kernel_terms);
    P(modeled_peak_bytes);P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);
#undef P
    py::class_<PeriodicBupoHFPreparationPlan,PeriodicBupoSourcePreparationPlan>(m,"PeriodicBupoHFPreparationPlan")
        .def_readonly("exchange_domains",&PeriodicBupoHFPreparationPlan::exchange_domains);
    auto r=py::class_<PeriodicBupoSourcePreparationReceipt>(m,"PeriodicBupoSourcePreparationReceipt");
#define R(n) r.def_readonly(#n,&PeriodicBupoHFPreparationReceipt::n)
    R(identity);R(pair_source_identity);R(tree_identity);R(kernel_identity);R(electronic_source_identity);
    R(pair_cube_radius);R(real_cube_radius);R(reciprocal_cube_radius);R(near_cube_radius);R(pairs);R(nodes);R(leaf_capacity);
    R(order);R(branching);R(maximum_depth);R(eta);R(opening);R(integral_tolerance);R(pair_cutoff_bohr);
    R(pointwise_tolerance);R(tail_mass_tolerance);R(minimum_radius);R(maximum_radius);
#undef R
    m.attr("PeriodicBupoHFPreparationReceipt")=m.attr("PeriodicBupoSourcePreparationReceipt");
    py::class_<PreparedPeriodicBupoSource,std::shared_ptr<PreparedPeriodicBupoSource>>(m,"PreparedPeriodicBupoSource")
        .def_property_readonly("plan",&PreparedPeriodicBupoSource::plan,py::return_value_policy::reference_internal)
        .def_property_readonly("receipt",&PreparedPeriodicBupoSource::receipt,py::return_value_policy::reference_internal)
        .def_property_readonly("source",&PreparedPeriodicBupoSource::source)
        .def_property_readonly("source_options",[](const PreparedPeriodicBupoSource& p){return p.source_options();})
        .def_property_readonly("remaining_budget",&PreparedPeriodicBupoSource::remaining_budget);
    m.def("prepare_periodic_bupo_source",[](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const PeriodicBupoSourceOptions& source_options,Count reciprocal_batch,
        const ResourceBudget& budget) {
        // Constructor captures caller inputs under the GIL, with cumulative
        // admission before each original producer allocation.
        return std::make_shared<PreparedPeriodicBupoSource>(basis,system,mesh,source_options,reciprocal_batch,budget);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("source_options")=PeriodicBupoSourceOptions{},
       py::arg("reciprocal_batch")=64,py::arg("budget")=ResourceBudget{});
    py::class_<PreparedPeriodicBupoHF,std::shared_ptr<PreparedPeriodicBupoHF>>(m,"PreparedPeriodicBupoHF")
        .def_property_readonly("plan",&PreparedPeriodicBupoHF::plan,py::return_value_policy::reference_internal)
        .def_property_readonly("receipt",&PreparedPeriodicBupoHF::receipt,py::return_value_policy::reference_internal)
        .def_property_readonly("options",[](const PreparedPeriodicBupoHF& p){return p.options();})
        .def_property_readonly("source_options",[](const PreparedPeriodicBupoHF& p){return p.source_options();})
        .def("run",[](const PreparedPeriodicBupoHF& p){py::gil_scoped_release release;return p.run();});
    PeriodicBupoHFOptions defaults;defaults.bound_exchange=true;
    m.def("prepare_periodic_bupo_hf",[](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,double cutoff,const PeriodicBupoHFSourceOptions& source_options,
        const PeriodicBupoHFOptions& options,const ResourceBudget& budget) {
        // Keep the GIL while the constructor admits and captures caller-owned
        // physical inputs. The eventual run uses only immutable owned inputs.
        return std::make_shared<PreparedPeriodicBupoHF>(basis,system,mesh,cutoff,source_options,options,budget);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),
       py::arg("source_options")=PeriodicBupoHFSourceOptions{},py::arg("options")=defaults,py::arg("budget")=ResourceBudget{});
}
