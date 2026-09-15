// Tiny diagnostics only. No Python Hcore/SCF or physical assertion bypass.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include "periodic_gaussian_nuclear_internal.hpp"
#include "vibeqc/periodic_gaussian_nuclear.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_nuclear(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianNuclearOptions;
    using Live=PeriodicGaussianNuclearLiveInventory;
    using Caps=PeriodicGaussianNuclearCaps;
    using Plan=PeriodicGaussianNuclearPlan;
    using EPlan=PeriodicGaussianNuclearEwaldPlan;
    using Diagnostics=PeriodicGaussianNuclearDiagnostics;
    using Panel=PeriodicGaussianNuclearPanel;
    using Ewald=PeriodicGaussianNuclearEwald;
    m.attr("_PERIODIC_GAUSSIAN_NUCLEAR_VERSION")=py::int_(kPeriodicGaussianNuclearVersion);
    m.attr("_PERIODIC_GAUSSIAN_NUCLEAR_POLICY")=py::str(kPeriodicGaussianNuclearPolicy);
    m.attr("_PERIODIC_GAUSSIAN_NUCLEAR_BOYS_POLICY")=py::str(kPeriodicGaussianNuclearBoysPolicy);
    m.attr("_PERIODIC_GAUSSIAN_NUCLEAR_FLOATING_POINT_POLICY")=py::str(kPeriodicGaussianNuclearFloatingPointPolicy);
#define NUCLEAR_RW(type,field) .def_readwrite(#field,&type::field)
    py::class_<Options>(m,"_PeriodicGaussianNuclearOptions").def(py::init<>())
        NUCLEAR_RW(Options,alpha) NUCLEAR_RW(Options,real_cutoff_bohr)
        NUCLEAR_RW(Options,reciprocal_cutoff_bohr_inverse)
        NUCLEAR_RW(Options,structural_absolute_tolerance) NUCLEAR_RW(Options,structural_relative_tolerance)
        NUCLEAR_RW(Options,basis_verification_caps);
    py::class_<Live>(m,"_PeriodicGaussianNuclearLiveInventory").def(py::init<>())
        NUCLEAR_RW(Live,replicas_per_node) NUCLEAR_RW(Live,other_retained_bytes_per_replica)
        NUCLEAR_RW(Live,other_transient_bytes_per_replica)
        NUCLEAR_RW(Live,fixed_backend_margin_bytes_per_replica) NUCLEAR_RW(Live,external_node_bytes);
    py::class_<Caps>(m,"_PeriodicGaussianNuclearCaps").def(py::init<>())
        NUCLEAR_RW(Caps,maximum_owned_numeric_bytes) NUCLEAR_RW(Caps,maximum_per_replica_inventoried_bytes)
        NUCLEAR_RW(Caps,maximum_node_inventoried_bytes) NUCLEAR_RW(Caps,maximum_atom_count)
        NUCLEAR_RW(Caps,maximum_pair_count) NUCLEAR_RW(Caps,maximum_ao_image_candidates)
        NUCLEAR_RW(Caps,maximum_nuclear_image_candidates) NUCLEAR_RW(Caps,maximum_reciprocal_candidates)
        NUCLEAR_RW(Caps,maximum_ewald_pair_candidates) NUCLEAR_RW(Caps,maximum_work_units);
#undef NUCLEAR_RW
#define NUCLEAR_RO(type,field) .def_readonly(#field,&type::field)
    py::class_<Plan>(m,"_PeriodicGaussianNuclearPlan")
        NUCLEAR_RO(Plan,n_basis) NUCLEAR_RO(Plan,atom_count) NUCLEAR_RO(Plan,k_index)
        NUCLEAR_RO(Plan,opposite_k_index) NUCLEAR_RO(Plan,pair_begin) NUCLEAR_RO(Plan,pair_count)
        NUCLEAR_RO(Plan,output_numeric_bytes) NUCLEAR_RO(Plan,boys_table_numeric_bytes)
        NUCLEAR_RO(Plan,os_workspace_numeric_bytes) NUCLEAR_RO(Plan,selected_shell_numeric_bytes_upper_bound)
        NUCLEAR_RO(Plan,primitive_and_shell_output_numeric_bytes) NUCLEAR_RO(Plan,fourier_numeric_workspace_bytes)
        NUCLEAR_RO(Plan,owned_numeric_peak_bytes) NUCLEAR_RO(Plan,borrowed_basis_active_numeric_bytes)
        NUCLEAR_RO(Plan,borrowed_system_active_numeric_bytes) NUCLEAR_RO(Plan,fixed_inventoried_object_bytes)
        NUCLEAR_RO(Plan,per_replica_inventoried_bytes) NUCLEAR_RO(Plan,node_inventoried_bytes)
        NUCLEAR_RO(Plan,ao_image_candidates_upper_bound) NUCLEAR_RO(Plan,primitive_pair_evaluations_upper_bound)
        NUCLEAR_RO(Plan,nuclear_image_candidates_upper_bound) NUCLEAR_RO(Plan,reciprocal_box_candidates)
        NUCLEAR_RO(Plan,reciprocal_candidate_evaluations_upper_bound) NUCLEAR_RO(Plan,ao_fourier_calls_upper_bound)
        NUCLEAR_RO(Plan,basis_scan_work_units) NUCLEAR_RO(Plan,preflight_work_units)
        NUCLEAR_RO(Plan,work_units_upper_bound) NUCLEAR_RO(Plan,maximum_selected_angular_momentum)
        .def_property_readonly("options",[](const Plan& p) { return p.options; })
        .def_property_readonly("live",[](const Plan& p) { return p.live; })
        .def_property_readonly("caps",[](const Plan& p) { return p.caps; });
    py::class_<EPlan>(m,"_PeriodicGaussianNuclearEwaldPlan")
        NUCLEAR_RO(EPlan,atom_count) NUCLEAR_RO(EPlan,packed_input_numeric_bytes)
        NUCLEAR_RO(EPlan,periodic_positions_numeric_bytes) NUCLEAR_RO(EPlan,reciprocal_reserved_numeric_bytes)
        NUCLEAR_RO(EPlan,owned_numeric_peak_bytes) NUCLEAR_RO(EPlan,borrowed_system_active_numeric_bytes)
        NUCLEAR_RO(EPlan,fixed_inventoried_object_bytes) NUCLEAR_RO(EPlan,per_replica_inventoried_bytes)
        NUCLEAR_RO(EPlan,node_inventoried_bytes) NUCLEAR_RO(EPlan,atom_pair_count)
        NUCLEAR_RO(EPlan,real_pair_candidates_upper_bound) NUCLEAR_RO(EPlan,reciprocal_box_candidates)
        NUCLEAR_RO(EPlan,preflight_work_units) NUCLEAR_RO(EPlan,work_units_upper_bound)
        .def_property_readonly("options",[](const EPlan& p) { return p.options; })
        .def_property_readonly("live",[](const EPlan& p) { return p.live; })
        .def_property_readonly("caps",[](const EPlan& p) { return p.caps; });
    py::class_<Diagnostics>(m,"_PeriodicGaussianNuclearDiagnostics")
        NUCLEAR_RO(Diagnostics,completed_ao_image_candidates) NUCLEAR_RO(Diagnostics,retained_ao_images)
        NUCLEAR_RO(Diagnostics,completed_nuclear_image_candidates) NUCLEAR_RO(Diagnostics,retained_nuclear_images)
        NUCLEAR_RO(Diagnostics,retained_reciprocal_vectors) NUCLEAR_RO(Diagnostics,completed_ao_fourier_calls)
        NUCLEAR_RO(Diagnostics,maximum_boys_series_iterations)
        NUCLEAR_RO(Diagnostics,maximum_erfc_seed_term_magnitude)
        NUCLEAR_RO(Diagnostics,minimum_nonzero_erfc_seed_difference_ratio)
        NUCLEAR_RO(Diagnostics,rounded_zero_erfc_seed_differences)
        NUCLEAR_RO(Diagnostics,maximum_hermitian_error) NUCLEAR_RO(Diagnostics,maximum_time_reversal_error)
        NUCLEAR_RO(Diagnostics,maximum_diagonal_imaginary_magnitude)
        NUCLEAR_RO(Diagnostics,maximum_trim_imaginary_magnitude);
#undef NUCLEAR_RO
    py::class_<Panel>(m,"_PeriodicGaussianNuclearPanel")
        .def_property_readonly("contract_version",&Panel::contract_version)
        .def_property_readonly("context",&Panel::context_handle)
        .def_property_readonly("plan",[](const Panel& p) { return p.plan(); })
        .def_property_readonly("diagnostics",[](const Panel& p) { return p.diagnostics(); })
        .def_property_readonly("infinite_image_tail_certified",&Panel::infinite_image_tail_certified)
        .def_property_readonly("whole_hf_hamiltonian_certified",&Panel::whole_hf_hamiltonian_certified)
        .def_property_readonly("propagated_roundoff_error_certified",&Panel::propagated_roundoff_error_certified)
        .def_property_readonly("nuclei_policy_identity_sha256",&Panel::nuclei_policy_identity_sha256)
        .def_property_readonly("panel_source_identity_sha256",&Panel::panel_source_identity_sha256)
        .def_property_readonly("payload_identity_sha256",&Panel::payload_identity_sha256)
        .def("element",&Panel::element)
        .def("values_copy",[](const Panel& p) {
            const auto n=p.plan().pair_count;
            if (!p.context_handle()||n>256||p.values().size()!=4U*n)
                throw std::length_error("Gaussian nuclear diagnostic copy exceeds tiny shape");
            py::array_t<std::complex<double>> values({py::ssize_t(4),static_cast<py::ssize_t>(n)});
            std::memcpy(values.mutable_data(),p.values().data(),static_cast<std::size_t>(64U*n)); return values;
        });
    py::class_<Ewald>(m,"_PeriodicGaussianNuclearEwald")
        .def_property_readonly("contract_version",&Ewald::contract_version)
        .def_property_readonly("context",&Ewald::context_handle)
        .def_property_readonly("plan",[](const Ewald& p) { return p.plan(); })
        .def_property_readonly("energy",&Ewald::energy)
        .def_property_readonly("infinite_image_tail_certified",&Ewald::infinite_image_tail_certified)
        .def_property_readonly("whole_hf_hamiltonian_certified",&Ewald::whole_hf_hamiltonian_certified)
        .def_property_readonly("cross_rank_bitwise_replay_certified",&Ewald::cross_rank_bitwise_replay_certified)
        .def_property_readonly("nuclei_policy_identity_sha256",&Ewald::nuclei_policy_identity_sha256)
        .def_property_readonly("scalar_source_identity_sha256",&Ewald::scalar_source_identity_sha256)
        .def_property_readonly("payload_identity_sha256",&Ewald::payload_identity_sha256);
    m.def("_plan_periodic_gaussian_nuclear_panel",&plan_periodic_gaussian_nuclear_panel);
    m.def("_plan_periodic_gaussian_nuclear_ewald",&plan_periodic_gaussian_nuclear_ewald);
    m.def("_build_periodic_gaussian_nuclear_panel",[](
        std::shared_ptr<const PeriodicGaussianSourceContext> context,const BasisSet& ao,const BasisSet& auxiliary,
        const PeriodicSystem& system,std::uint64_t k,std::uint64_t begin,std::uint64_t count,
        const Options& options,const Live& live,const Caps& caps) {
        if (!context||context->mesh().size()>16||ao.nbasis()>32||auxiliary.nbasis()>32||count>256
            ||system.unit_cell.size()>8||caps.maximum_owned_numeric_bytes>16777216
            ||caps.maximum_nuclear_image_candidates>5000000||caps.maximum_ao_image_candidates>5000000
            ||caps.maximum_reciprocal_candidates>1000000||caps.maximum_work_units>1000000000000000ULL)
            throw std::length_error("Gaussian nuclear diagnostic exceeds tiny shape or caps");
        if (caps.maximum_work_units>1000000000000ULL) {
            // Explicit isolated high-L regression only. The conservative
            // rejected-candidate times full-OS bound remains unchanged.
            if (count!=1||context->mesh().size()!=1||system.unit_cell.size()!=1
                ||ao.libint().size()!=1||ao.libint()[0].alpha.size()!=1
                ||ao.libint()[0].contr.size()!=1||ao.libint()[0].contr[0].l<3
                ||context->options().ao_pair_image_cutoff_bohr>.5
                ||options.real_cutoff_bohr>2||options.reciprocal_cutoff_bohr_inverse>.1
                ||!system.lattice.isDiagonal()||system.lattice.diagonal().cwiseAbs().minCoeff()<30)
                throw std::length_error("Gaussian nuclear expanded diagnostic work requires an isolated one-pair high-L witness");
        }
        const auto o=options; const auto l=live; const auto c=caps;
        py::gil_scoped_release release;
        return build_periodic_gaussian_nuclear_panel(std::move(context),ao,auxiliary,system,k,begin,count,o,l,c);
    });
    m.def("_build_periodic_gaussian_nuclear_ewald",[](
        std::shared_ptr<const PeriodicGaussianSourceContext> context,const PeriodicSystem& system,
        const Options& options,const Live& live,const Caps& caps) {
        if (!context||system.unit_cell.size()>8||caps.maximum_owned_numeric_bytes>16777216
            ||caps.maximum_ewald_pair_candidates>5000000||caps.maximum_reciprocal_candidates>1000000
            ||caps.maximum_work_units>1000000000000ULL)
            throw std::length_error("Gaussian nuclear Ewald diagnostic exceeds tiny shape or caps");
        const auto o=options; const auto l=live; const auto c=caps;
        py::gil_scoped_release release;
        return build_periodic_gaussian_nuclear_ewald(std::move(context),system,o,l,c);
    });
    m.def("_periodic_gaussian_nuclear_boys_diagnostic",[](int order,py::array values,
        std::uint64_t maximum_owned_numeric_bytes,std::uint64_t maximum_work_units) {
        if (!values.dtype().is(py::dtype::of<double>())||values.ndim()!=1||values.size()<1||values.size()>256
            ||values.strides(0)!=sizeof(double)||maximum_owned_numeric_bytes>2097152||maximum_work_units>2000000000ULL)
            throw std::invalid_argument("Gaussian nuclear Boys diagnostic expects a tiny contiguous binary64 vector");
        const auto* input=static_cast<const double*>(values.data());
        const auto n=static_cast<std::uint64_t>(values.size());
        std::vector<double> result;
        { py::gil_scoped_release release;
          result=periodic_gaussian_nuclear_detail::boys_diagnostic(order,input,n,maximum_owned_numeric_bytes,maximum_work_units); }
        py::array_t<double> out(values.size());
        std::memcpy(out.mutable_data(),result.data(),result.size()*sizeof(double)); return out;
    });
}
