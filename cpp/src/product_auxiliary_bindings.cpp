#include "vibeqc/product_auxiliary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace vibeqc;
void bind_product_auxiliary(py::module_& m) {
    auto o=py::class_<ProductAuxiliaryOptions>(m,"ProductAuxiliaryOptions").def(py::init<>());
    o.def_readwrite("beta",&ProductAuxiliaryOptions::beta).def_readwrite("refinement",&ProductAuxiliaryOptions::refinement)
        .def_readwrite("maximum_angular_momentum",&ProductAuxiliaryOptions::maximum_angular_momentum);
    auto b=py::class_<ProductAuxiliaryLimits>(m,"ProductAuxiliaryLimits").def(py::init<>());
#define B(n) b.def_readwrite(#n,&ProductAuxiliaryLimits::n)
    B(memory_bytes);B(work_units);B(maximum_shells);B(maximum_functions);
#undef B
    auto p=py::class_<ProductAuxiliaryPlan>(m,"ProductAuxiliaryPlan");
#define P(n) p.def_readonly(#n,&ProductAuxiliaryPlan::n)
    P(centers);P(channels);P(original_shells);P(original_functions);P(maximum_generated_shells);P(maximum_generated_functions);
    P(maximum_shells);P(maximum_functions);P(modeled_peak_bytes);P(work_units);
#undef P
    auto s=py::class_<ProductAuxiliarySpan>(m,"ProductAuxiliarySpan");
#define S(n) s.def_readonly(#n,&ProductAuxiliarySpan::n)
    S(atom);S(coarse_intervals);S(angular_momentum);S(minimum_exponent);S(maximum_product_exponent);S(upper_grid_exponent);
#undef S
    auto r=py::class_<ProductAuxiliaryReceipt>(m,"ProductAuxiliaryReceipt");
#define R(n) r.def_readonly(#n,&ProductAuxiliaryReceipt::n)
    R(algorithm);R(qualification);R(citation_method);R(orbital_identity);R(seed_identity);R(basis_identity);R(identity);
    R(options);R(added_shells);R(added_functions);R(duplicate_seed_shells);R(spans);
#undef R
    py::class_<ProductAuxiliaryBasis>(m,"ProductAuxiliaryBasis")
        .def("matches_seed",&ProductAuxiliaryBasis::matches_seed,py::arg("seed"))
        .def("matches_basis",&ProductAuxiliaryBasis::matches_basis,py::arg("basis"))
        .def("matches_orbital",&ProductAuxiliaryBasis::matches_orbital,py::arg("orbital"))
        .def_property_readonly("basis",&ProductAuxiliaryBasis::basis,py::return_value_policy::reference_internal)
        .def_property_readonly("plan",[](const ProductAuxiliaryBasis& s){return ProductAuxiliaryPlan(s.plan());})
        .def_property_readonly("receipt",[](const ProductAuxiliaryBasis& s){return ProductAuxiliaryReceipt(s.receipt());});
    m.def("plan_product_auxiliary_basis",&plan_product_auxiliary_basis,py::arg("molecule"),py::arg("orbital"),py::arg("seed")=nullptr,
        py::arg("options")=ProductAuxiliaryOptions{},py::arg("limits")=ProductAuxiliaryLimits{});
    m.def("prepare_product_auxiliary_basis",[](const Molecule& mol,const BasisSet& orbital,const BasisSet* seed,
        const ProductAuxiliaryOptions& options,const ProductAuxiliaryLimits& limits) {
        // No Python snapshots before native count admission. Keep the GIL
        // while borrowing input metadata; the returned source owns its basis.
        return std::make_unique<ProductAuxiliaryBasis>(mol,orbital,seed,options,limits);
    },py::arg("molecule"),py::arg("orbital"),py::arg("seed")=nullptr,py::arg("options")=ProductAuxiliaryOptions{},py::arg("limits")=ProductAuxiliaryLimits{});
    auto seqplan=py::class_<ProductAuxiliarySequencePlan>(m,"ProductAuxiliarySequencePlan");
#define S(n) seqplan.def_readonly(#n,&ProductAuxiliarySequencePlan::n)
    S(candidates);S(modeled_peak_bytes);S(work_units);S(maximum_shells);S(maximum_functions);
#undef S
    py::class_<ProductAuxiliarySequence,std::shared_ptr<ProductAuxiliarySequence>>(m,"ProductAuxiliarySequence")
        .def_property_readonly("plan",[](const ProductAuxiliarySequence& s){return s.plan();})
        .def_property_readonly("initial",[](const ProductAuxiliarySequence& s){return s.initial();})
        .def_property_readonly("maximum_refinements",&ProductAuxiliarySequence::maximum_refinements)
        .def("candidate",&ProductAuxiliarySequence::candidate,py::return_value_policy::reference_internal)
        .def("matches_sources",&ProductAuxiliarySequence::matches_sources);
    m.def("plan_product_auxiliary_sequence",&plan_product_auxiliary_sequence,py::arg("molecule"),py::arg("orbital"),py::arg("seed"),
        py::arg("initial")=ProductAuxiliaryOptions{},py::arg("maximum_refinements")=2,py::arg("limits")=ProductAuxiliaryLimits{});
    m.def("prepare_product_auxiliary_sequence",[](const Molecule& mol,const BasisSet& orbital,const BasisSet& seed,
        const ProductAuxiliaryOptions& initial,std::uint64_t refinements,const ProductAuxiliaryLimits& limits) {
        return std::make_shared<ProductAuxiliarySequence>(mol,orbital,seed,initial,refinements,limits);
    },py::arg("molecule"),py::arg("orbital"),py::arg("seed"),py::arg("initial")=ProductAuxiliaryOptions{},
      py::arg("maximum_refinements")=2,py::arg("limits")=ProductAuxiliaryLimits{});

}
