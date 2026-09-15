// Tiny diagnostic boundary, not a dense/all-k Hcore constructor.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_gaussian_one_electron.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_one_electron(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianOneElectronOptions;
    using Live=PeriodicGaussianOneElectronLiveInventory;
    using Caps=PeriodicGaussianOneElectronCaps;
    using Plan=PeriodicGaussianOneElectronPlan;
    using Diagnostics=PeriodicGaussianOneElectronDiagnostics;
    using Panel=PeriodicGaussianOneElectronPanel;
    m.attr("_PERIODIC_GAUSSIAN_ONE_ELECTRON_VERSION")=py::int_(kPeriodicGaussianOneElectronVersion);
    m.attr("_PERIODIC_GAUSSIAN_ONE_ELECTRON_POLICY")=py::str(kPeriodicGaussianOneElectronPolicy);
    py::class_<Options>(m,"_PeriodicGaussianOneElectronOptions").def(py::init<>())
        .def_readwrite("basis_verification_caps",&Options::basis_verification_caps)
        .def_readwrite("structural_absolute_tolerance",&Options::structural_absolute_tolerance)
        .def_readwrite("structural_relative_tolerance",&Options::structural_relative_tolerance);
    py::class_<Live>(m,"_PeriodicGaussianOneElectronLiveInventory").def(py::init<>())
        .def_readwrite("replicas_per_node",&Live::replicas_per_node)
        .def_readwrite("other_retained_bytes_per_replica",&Live::other_retained_bytes_per_replica)
        .def_readwrite("other_transient_bytes_per_replica",&Live::other_transient_bytes_per_replica)
        .def_readwrite("fixed_backend_margin_bytes_per_replica",&Live::fixed_backend_margin_bytes_per_replica)
        .def_readwrite("external_node_bytes",&Live::external_node_bytes);
    py::class_<Caps>(m,"_PeriodicGaussianOneElectronCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numeric_bytes",&Caps::maximum_owned_numeric_bytes)
        .def_readwrite("maximum_per_replica_inventoried_bytes",&Caps::maximum_per_replica_inventoried_bytes)
        .def_readwrite("maximum_node_inventoried_bytes",&Caps::maximum_node_inventoried_bytes)
        .def_readwrite("maximum_pair_count",&Caps::maximum_pair_count)
        .def_readwrite("maximum_candidate_evaluations",&Caps::maximum_candidate_evaluations)
        .def_readwrite("maximum_work_units",&Caps::maximum_work_units);
#define ONE_ELECTRON_PLAN(field) .def_readonly(#field,&Plan::field)
    py::class_<Plan>(m,"_PeriodicGaussianOneElectronPlan")
        ONE_ELECTRON_PLAN(n_basis) ONE_ELECTRON_PLAN(k_index) ONE_ELECTRON_PLAN(opposite_k_index)
        ONE_ELECTRON_PLAN(pair_begin) ONE_ELECTRON_PLAN(pair_count)
        ONE_ELECTRON_PLAN(output_numeric_bytes) ONE_ELECTRON_PLAN(fixed_numeric_workspace_bytes)
        ONE_ELECTRON_PLAN(owned_numeric_peak_bytes) ONE_ELECTRON_PLAN(borrowed_basis_active_numeric_bytes)
        ONE_ELECTRON_PLAN(fixed_inventoried_object_bytes) ONE_ELECTRON_PLAN(per_replica_inventoried_bytes)
        ONE_ELECTRON_PLAN(node_inventoried_bytes) ONE_ELECTRON_PLAN(candidate_evaluations)
        ONE_ELECTRON_PLAN(primitive_pair_evaluations_upper_bound) ONE_ELECTRON_PLAN(md_table_cells_upper_bound)
        ONE_ELECTRON_PLAN(cartesian_pair_terms_upper_bound) ONE_ELECTRON_PLAN(image_identity_wire_bytes_upper_bound)
        ONE_ELECTRON_PLAN(basis_scan_work_units) ONE_ELECTRON_PLAN(preflight_work_units)
        ONE_ELECTRON_PLAN(work_units_upper_bound)
        .def_property_readonly("options",[](const Plan& p) { return p.options; })
        .def_property_readonly("live",[](const Plan& p) { return p.live; })
        .def_property_readonly("caps",[](const Plan& p) { return p.caps; });
#undef ONE_ELECTRON_PLAN
#define ONE_ELECTRON_DIAGNOSTIC(field) .def_readonly(#field,&Diagnostics::field)
    py::class_<Diagnostics>(m,"_PeriodicGaussianOneElectronDiagnostics")
        ONE_ELECTRON_DIAGNOSTIC(retained_primary_pair_images)
        ONE_ELECTRON_DIAGNOSTIC(retained_audit_pair_images)
        ONE_ELECTRON_DIAGNOSTIC(completed_candidate_evaluations)
        ONE_ELECTRON_DIAGNOSTIC(completed_primitive_pair_evaluations)
        ONE_ELECTRON_DIAGNOSTIC(image_identity_wire_bytes)
        ONE_ELECTRON_DIAGNOSTIC(maximum_overlap_hermitian_error)
        ONE_ELECTRON_DIAGNOSTIC(maximum_kinetic_hermitian_error)
        ONE_ELECTRON_DIAGNOSTIC(maximum_overlap_time_reversal_error)
        ONE_ELECTRON_DIAGNOSTIC(maximum_kinetic_time_reversal_error)
        ONE_ELECTRON_DIAGNOSTIC(maximum_diagonal_imaginary_magnitude)
        ONE_ELECTRON_DIAGNOSTIC(maximum_trim_imaginary_magnitude);
#undef ONE_ELECTRON_DIAGNOSTIC
    py::class_<Panel>(m,"_PeriodicGaussianOneElectronPanel")
        .def_property_readonly("contract_version",&Panel::contract_version)
        .def_property_readonly("context",&Panel::context_handle)
        .def_property_readonly("plan",[](const Panel& p) { return p.plan(); })
        .def_property_readonly("diagnostics",[](const Panel& p) { return p.diagnostics(); })
        .def_property_readonly("nuclear_hcore_certified",&Panel::nuclear_hcore_certified)
        .def_property_readonly("infinite_image_tail_certified",&Panel::infinite_image_tail_certified)
        .def_property_readonly("ao_image_source_certified",&Panel::ao_image_source_certified)
        .def_property_readonly("image_source_identity_sha256",&Panel::image_source_identity_sha256)
        .def_property_readonly("operator_source_identity_sha256",&Panel::operator_source_identity_sha256)
        .def_property_readonly("payload_identity_sha256",&Panel::payload_identity_sha256)
        .def("element",&Panel::element)
        .def("values_copy",[](const Panel& p) {
            const auto n=p.plan().pair_count;
            if (!p.context_handle() || n>256 || p.values().size()!=2U*n)
                throw std::length_error("Gaussian one-electron diagnostic copy exceeds tiny shape");
            py::array_t<std::complex<double>> output({py::ssize_t(2),static_cast<py::ssize_t>(n)});
            std::memcpy(output.mutable_data(),p.values().data(),static_cast<std::size_t>(32U*n));
            return output;
        });
    m.def("_plan_periodic_gaussian_one_electron_panel",&plan_periodic_gaussian_one_electron_panel,
        py::arg("context"),py::arg("ao"),py::arg("auxiliary"),py::arg("system"),
        py::arg("k_index"),py::arg("pair_begin"),py::arg("pair_count"),
        py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_build_periodic_gaussian_one_electron_panel",[](
        std::shared_ptr<const PeriodicGaussianSourceContext> context,const BasisSet& ao,const BasisSet& auxiliary,
        const PeriodicSystem& system,std::uint64_t k,std::uint64_t begin,std::uint64_t count,
        const Options& options,const Live& live,const Caps& caps) {
        if (!context || context->mesh().size()>16 || ao.nbasis()>32 || auxiliary.nbasis()>32 || count>256
            || caps.maximum_candidate_evaluations>1000000 || caps.maximum_owned_numeric_bytes>1048576
            || caps.maximum_work_units>2000000000)
            throw std::length_error("Gaussian one-electron diagnostic exceeds tiny shape or caps");
        const auto settings=options; const auto inventory=live; const auto limits=caps;
        py::gil_scoped_release release;
        return build_periodic_gaussian_one_electron_panel(std::move(context),ao,auxiliary,system,k,begin,count,settings,inventory,limits);
    },py::arg("context"),py::arg("ao"),py::arg("auxiliary"),py::arg("system"),
       py::arg("k_index"),py::arg("pair_begin"),py::arg("pair_count"),
       py::arg("options"),py::arg("live"),py::arg("caps"));
}
