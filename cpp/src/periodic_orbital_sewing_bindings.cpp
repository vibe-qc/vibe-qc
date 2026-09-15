// Tiny numerical witnesses only. No physical-source symmetry admission.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_orbital_sewing.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_orbital_sewing(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicOrbitalSewingOptions;
    using Inventory = PeriodicOrbitalSewingInventory;
    using Caps = PeriodicOrbitalSewingCaps;
    using Plan = PeriodicOrbitalSewingPlan;
    using Subspace = PeriodicOrbitalSubspace;
    using SubDiagnostics = PeriodicOrbitalSubspaceSewingDiagnostics;
    using Diagnostics = PeriodicOrbitalSewingDiagnostics;
    using Result = PeriodicOrbitalSewingResult;
    using State = PeriodicRestrictedMeanFieldState;
    using U = std::uint64_t;
    using C = std::complex<double>;
    py::enum_<Subspace>(m, "_PeriodicOrbitalSubspace")
        .value("FROZEN_CORE", Subspace::FrozenCore)
        .value("CORRELATED_OCCUPIED", Subspace::CorrelatedOccupied)
        .value("VIRTUAL", Subspace::Virtual);
    auto options = py::class_<Options>(m, "_PeriodicOrbitalSewingOptions").def(py::init<>());
    options.def_readwrite("ao_transport", &Options::ao_transport);
    options.def_readwrite("request_full_ao_scope", &Options::request_full_ao_scope);
    options.def_readwrite("maximum_reciprocal_lattice_residual", &Options::maximum_reciprocal_lattice_residual);
    options.def_readwrite("maximum_source_metric_residual", &Options::maximum_source_metric_residual);
    options.def_readwrite("maximum_target_metric_residual", &Options::maximum_target_metric_residual);
    options.def_readwrite("maximum_transported_metric_residual", &Options::maximum_transported_metric_residual);
    options.def_readwrite("maximum_unitarity_residual", &Options::maximum_unitarity_residual);
    options.def_readwrite("maximum_reconstruction_residual", &Options::maximum_reconstruction_residual);
    options.def_readwrite("maximum_cross_subspace_overlap", &Options::maximum_cross_subspace_overlap);
    options.def_readwrite("maximum_roothaan_residual", &Options::maximum_roothaan_residual);
    options.def_readwrite("maximum_energy_intertwining_residual", &Options::maximum_energy_intertwining_residual);
    auto inventory = py::class_<Inventory>(m, "_PeriodicOrbitalSewingInventory").def(py::init<>());
    inventory.def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    inventory.def_readwrite("external_node_bytes", &Inventory::external_node_bytes);
    inventory.def_readwrite("other_live_numerical_bytes_per_replica", &Inventory::other_live_numerical_bytes_per_replica);
    inventory.def_readwrite("other_live_control_bytes_per_replica", &Inventory::other_live_control_bytes_per_replica);
    inventory.def_readwrite("backend_margin_bytes_per_replica", &Inventory::backend_margin_bytes_per_replica);
    auto caps = py::class_<Caps>(m, "_PeriodicOrbitalSewingCaps").def(py::init<>());
    caps.def_readwrite("ao_transport", &Caps::ao_transport);
    caps.def_readwrite("maximum_kpoints", &Caps::maximum_kpoints);
    caps.def_readwrite("maximum_basis_functions", &Caps::maximum_basis_functions);
    caps.def_readwrite("maximum_effective_orbitals", &Caps::maximum_effective_orbitals);
    caps.def_readwrite("maximum_subspace_rank", &Caps::maximum_subspace_rank);
    caps.def_readwrite("maximum_transport_calls", &Caps::maximum_transport_calls);
    caps.def_readwrite("maximum_borrowed_numerical_bytes", &Caps::maximum_borrowed_numerical_bytes);
    caps.def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes);
    caps.def_readwrite("maximum_control_storage_bytes", &Caps::maximum_control_storage_bytes);
    caps.def_readwrite("maximum_per_replica_inventoried_bytes", &Caps::maximum_per_replica_inventoried_bytes);
    caps.def_readwrite("maximum_node_inventoried_bytes", &Caps::maximum_node_inventoried_bytes);
    caps.def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    auto plan = py::class_<Plan>(m, "_PeriodicOrbitalSewingPlan");
    plan.def_readonly("n_basis", &Plan::n_basis);
    plan.def_readonly("n_effective_orbitals", &Plan::n_effective_orbitals);
    plan.def_readonly("n_kpoints", &Plan::n_kpoints);
    plan.def_readonly("source_index", &Plan::source_index);
    plan.def_readonly("target_index", &Plan::target_index);
    plan.def_readonly("maximum_rank", &Plan::maximum_rank);
    plan.def_readonly("subspace_ranks", &Plan::subspace_ranks);
    plan.def_readonly("target_doubled_address", &Plan::target_doubled_address);
    plan.def_readonly("reciprocal_wrap", &Plan::reciprocal_wrap);
    plan.def_readonly("time_reversal", &Plan::time_reversal);
    plan.def_readonly("full_ao_scope", &Plan::full_ao_scope);
    plan.def_readonly("state_resident_numerical_bytes", &Plan::state_resident_numerical_bytes);
    plan.def_readonly("state_control_storage_bytes", &Plan::state_control_storage_bytes);
    plan.def_readonly("borrowed_basis_numeric_bytes", &Plan::borrowed_basis_numeric_bytes);
    plan.def_readonly("borrowed_geometry_numeric_bytes", &Plan::borrowed_geometry_numeric_bytes);
    plan.def_readonly("borrowed_numerical_bytes", &Plan::borrowed_numerical_bytes);
    plan.def_readonly("retained_sewing_bytes", &Plan::retained_sewing_bytes);
    plan.def_readonly("index_workspace_bytes", &Plan::index_workspace_bytes);
    plan.def_readonly("maximum_packed_source_bytes", &Plan::maximum_packed_source_bytes);
    plan.def_readonly("column_workspace_bytes", &Plan::column_workspace_bytes);
    plan.def_readonly("fixed_scalar_numerical_bytes", &Plan::fixed_scalar_numerical_bytes);
    plan.def_readonly("transport_phase_owned_upper_bound", &Plan::transport_phase_owned_upper_bound);
    plan.def_readonly("audit_phase_owned_upper_bound", &Plan::audit_phase_owned_upper_bound);
    plan.def_readonly("peak_owned_numerical_bytes", &Plan::peak_owned_numerical_bytes);
    plan.def_readonly("control_storage_bytes", &Plan::control_storage_bytes);
    plan.def_readonly("per_replica_inventoried_bytes", &Plan::per_replica_inventoried_bytes);
    plan.def_readonly("required_node_inventoried_bytes", &Plan::required_node_inventoried_bytes);
    plan.def_readonly("transport_calls", &Plan::transport_calls);
    plan.def_readonly("transport_work_units_upper_bound", &Plan::transport_work_units_upper_bound);
    plan.def_readonly("validation_work_units_upper_bound", &Plan::validation_work_units_upper_bound);
    plan.def_readonly("work_units_upper_bound", &Plan::work_units_upper_bound);
    plan.def_readonly("transport_upper", &Plan::transport_upper);
    auto sub = py::class_<SubDiagnostics>(m, "_PeriodicOrbitalSubspaceSewingDiagnostics");
    sub.def_readonly("rank", &SubDiagnostics::rank);
    sub.def_readonly("source_metric_residual", &SubDiagnostics::source_metric_residual);
    sub.def_readonly("target_metric_residual", &SubDiagnostics::target_metric_residual);
    sub.def_readonly("transported_metric_residual", &SubDiagnostics::transported_metric_residual);
    sub.def_readonly("unitarity_residual", &SubDiagnostics::unitarity_residual);
    sub.def_readonly("reconstruction_residual", &SubDiagnostics::reconstruction_residual);
    sub.def_readonly("cross_subspace_overlap", &SubDiagnostics::cross_subspace_overlap);
    sub.def_readonly("roothaan_residual", &SubDiagnostics::roothaan_residual);
    sub.def_readonly("source_roothaan_residual", &SubDiagnostics::source_roothaan_residual);
    sub.def_readonly("target_roothaan_residual", &SubDiagnostics::target_roothaan_residual);
    sub.def_readonly("energy_intertwining_residual", &SubDiagnostics::energy_intertwining_residual);
    sub.def_readonly("transport", &SubDiagnostics::transport);
    auto diag = py::class_<Diagnostics>(m, "_PeriodicOrbitalSewingDiagnostics");
    diag.def_readonly("reciprocal_lattice_residual", &Diagnostics::reciprocal_lattice_residual);
    diag.def_readonly("completed_transport_calls", &Diagnostics::completed_transport_calls);
    diag.def_readonly("audited_source_columns", &Diagnostics::audited_source_columns);
    diag.def_readonly("subspaces", &Diagnostics::subspaces);
    py::class_<Result>(m, "_PeriodicOrbitalSewingResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("state", [](const Result& r) {
            return std::const_pointer_cast<State>(r.state_handle());
        })
        .def_property_readonly("state_identity_sha256", &Result::state_identity_sha256)
        .def_property_readonly("full_ao_scope_audited", &Result::full_ao_scope_audited)
        .def_property_readonly("physical_source_symmetry_certified", &Result::physical_source_symmetry_certified)
        .def("element", &Result::element)
        .def("source_band", &Result::source_band)
        .def("target_band", &Result::target_band)
        .def("sewing_copy", [](const Result& r, Subspace subspace) {
            const auto* source = r.sewing_data(subspace); // validates enum and moved-from state
            const U rank = r.memory().subspace_ranks.at(static_cast<U>(subspace));
            if (rank > 32) throw std::length_error("Orbital sewing diagnostic copy exceeds tiny bounds");
            py::array_t<C> out({py::ssize_t(rank), py::ssize_t(rank)});
            if (rank) std::memcpy(out.mutable_data(), source, sizeof(C)*rank*rank);
            out.attr("setflags")(false);
            return out;
        });
    const auto small = [](const std::shared_ptr<const State>& s,
                          const BasisSet& b, const PeriodicSystem& system) {
        if (!s) throw std::invalid_argument("Orbital sewing requires an immutable state");
        if (s->n_kpoints() > 64 || s->n_basis() > 64 || s->n_effective_orbitals() > 32
            || b.nbasis() > 64 || b.nshells() > 128 || system.unit_cell.size() > 32)
            throw std::length_error("Orbital sewing diagnostic exceeds tiny shape bounds");
    };
    const auto bounded = [](const Plan& p) {
        if (p.peak_owned_numerical_bytes > (16U << 20)
            || p.required_node_inventoried_bytes > (128U << 20)
            || p.work_units_upper_bound > 1000000000ULL)
            throw std::length_error("Orbital sewing diagnostic exceeds tiny memory/work bounds");
    };
    m.def("_plan_periodic_orbital_sewing", [small, bounded](
        std::shared_ptr<State> state, const BasisSet& basis, const PeriodicSystem& system,
        const SymmetryOp& op, U source, bool tr, const Options& options,
        const Inventory& inventory, const Caps& caps) {
        small(state, basis, system);
        auto p = plan_periodic_orbital_sewing(state, basis, system, op, source, tr, options, inventory, caps);
        bounded(p);
        return p;
    }, py::arg("state"), py::arg("basis"), py::arg("system"), py::arg("operation"),
       py::arg("source_index"), py::arg("time_reversal"), py::arg("options"),
       py::arg("inventory"), py::arg("caps"));
    m.def("_make_periodic_orbital_sewing", [small, bounded](
        std::shared_ptr<State> state, const BasisSet& basis, const PeriodicSystem& system,
        const SymmetryOp& op, U source, bool tr, const Options& options,
        const Inventory& inventory, const Caps& caps) {
        small(state, basis, system);
        auto p = plan_periodic_orbital_sewing(state, basis, system, op, source, tr, options, inventory, caps);
        bounded(p);
        // Retain GIL: no callbacks/conversions or mutable borrowed-input windows.
        return make_periodic_orbital_sewing(state, basis, system, op, source, tr, options, inventory, caps);
    }, py::arg("state"), py::arg("basis"), py::arg("system"), py::arg("operation"),
       py::arg("source_index"), py::arg("time_reversal"), py::arg("options"),
       py::arg("inventory"), py::arg("caps"));
}
