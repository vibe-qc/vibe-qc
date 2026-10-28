#include "vibeqc/manta_molecular_source.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
void check(const py::array& a,const py::dtype& dtype,int rank) {
    if(!a.dtype().is(dtype) || a.ndim()!=rank || !(a.flags()&py::array::c_style))
        throw py::value_error("molecular source support requires C-contiguous float64 operations and int64 destinations");
}
}
void bind_manta_molecular_source(py::module_& m) {
    using namespace vibeqc::manta;
    auto qp=py::class_<MolecularQuartetConstructionPlan>(m,"MolecularQuartetConstructionPlan");
#define QP(n) qp.def_readonly(#n,&MolecularQuartetConstructionPlan::n)
    QP(shell_calls); QP(primitive_products); QP(scalar_terms); QP(kernel_terms);
    QP(additional_bytes); QP(modeled_peak_bytes);
#undef QP
    auto qc=py::class_<MolecularQuartetConstruction>(m,"MolecularQuartetConstruction");
#define QC(n) qc.def_property_readonly(#n,&MolecularQuartetConstruction::n)
    QC(plan); QC(source_identity); QC(key); QC(evaluated); QC(lower); QC(upper); QC(error_bound); QC(null_tile);
#undef QC
    qc.def("validate",&MolecularQuartetConstruction::validate)
      .def_property_readonly("arithmetic_equivariance_certified",[](const MolecularQuartetConstruction&){return false;})
      .def_property_readonly("qualification_scope",[](const MolecularQuartetConstruction&){return "finite-molecular-s-quartet-construction";});
    m.def("plan_molecular_quartet_construction",&plan_molecular_quartet_construction,
        py::arg("provider"),py::arg("key"),py::arg("budget"));
    m.def("certify_molecular_quartet_construction",[](std::shared_ptr<OrderedJKBuilder> b,
        const std::array<std::size_t,4>& key,const ResourceBudget& budget) {
        return MolecularQuartetConstruction(std::move(b),key,budget);
    },py::arg("provider"),py::arg("key"),py::arg("budget"));
    auto plan=py::class_<MolecularSourceSupportPlan>(m,"MolecularSourceSupportPlan");
#define P(n) plan.def_readonly(#n,&MolecularSourceSupportPlan::n)
    P(operations); P(shell_checks); P(radial_values); P(additional_bytes); P(modeled_peak_bytes); P(scalar_terms);
#undef P
    py::class_<MolecularSourceSupport,std::shared_ptr<MolecularSourceSupport>>(m,"MolecularSourceSupport")
        .def_property_readonly("plan",[](const MolecularSourceSupport& c){return c.plan();})
        .def_property_readonly("identity",&MolecularSourceSupport::identity)
        .def_property_readonly("source_identity",&MolecularSourceSupport::source_identity)
        .def_property_readonly("maximum_center_residual",&MolecularSourceSupport::maximum_center_residual)
        .def_property_readonly("geometry_tolerance",&MolecularSourceSupport::geometry_tolerance)
        .def_property_readonly("arithmetic_equivariance_certified",[](const MolecularSourceSupport&){return false;})
        .def_property_readonly("qualification_scope",[](const MolecularSourceSupport&){return "finite-molecular-source-support";})
        .def("validate",&MolecularSourceSupport::validate)
        .def("validate_maps",[](const MolecularSourceSupport& c,const py::array& d) {
            check(d,py::dtype::of<std::int64_t>(),2);
            if(reinterpret_cast<std::uintptr_t>(d.data())%alignof(std::int64_t))
                throw py::value_error("molecular source map validation requires aligned storage");
            c.validate_maps(d.shape(0),d.shape(1),static_cast<const std::int64_t*>(d.data()));
        },py::arg("destinations").noconvert());
    m.def("plan_molecular_source_support",&plan_molecular_source_support,
        py::arg("provider"),py::arg("operations"),py::arg("budget"));
    m.def("certify_molecular_source_support",[](std::shared_ptr<OrderedJKBuilder> b,
        const py::array& r,const py::array& t,const py::array& d,double tolerance,const ResourceBudget& budget) {
        check(r,py::dtype::of<double>(),3); check(t,py::dtype::of<double>(),2); check(d,py::dtype::of<std::int64_t>(),2);
        if(!b || r.shape(1)!=3 || r.shape(2)!=3 || t.shape(0)!=r.shape(0) || t.shape(1)!=3 ||
            d.shape(0)!=r.shape(0) || Count(d.shape(1))!=b->plan().n_shells)
            throw py::value_error("molecular source support operation shape mismatch");
        plan_molecular_source_support(*b,r.shape(0),budget); // Before payload copies, including unaligned input.
        std::vector<double> rotations(r.size()),translations(t.size());
        std::vector<std::int64_t> destinations(d.size());
        std::memcpy(rotations.data(),r.data(),r.nbytes()); std::memcpy(translations.data(),t.data(),t.nbytes());
        std::memcpy(destinations.data(),d.data(),d.nbytes());
        return MolecularSourceSupport(std::move(b),r.shape(0),rotations.data(),translations.data(),destinations.data(),tolerance,budget);
    },py::arg("provider"),py::arg("rotations").noconvert(),py::arg("translations").noconvert(),
        py::arg("destinations").noconvert(),py::arg("geometry_tolerance"),py::arg("budget"));
}
