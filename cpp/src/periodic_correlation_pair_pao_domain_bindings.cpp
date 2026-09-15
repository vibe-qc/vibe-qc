// Tiny typed native-domain union diagnostics; included by bindings.cpp.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_correlation_pair_pao_domain.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pair_pao_domain(py::module_& m) {
    using namespace vibeqc;
    using Inventory = PeriodicCorrelationPairPAODomainInventory;
    using Caps = PeriodicCorrelationPairPAODomainCaps;
    using Plan = PeriodicCorrelationPairPAODomainMemoryPlan;
    using Diagnostics = PeriodicCorrelationPairPAODomainDiagnostics;
    using Result = PeriodicCorrelationPairPAODomain;
    py::class_<Inventory>(m, "_PeriodicCorrelationPairPAODomainInventory").def(py::init<>())
        .def_readwrite("other_live_numerical_bytes", &Inventory::other_live_numerical_bytes)
        .def_readwrite("other_live_control_bytes", &Inventory::other_live_control_bytes)
        .def_readwrite("backend_allowance_bytes", &Inventory::backend_allowance_bytes);
    auto caps = py::class_<Caps>(m, "_PeriodicCorrelationPairPAODomainCaps"); caps.def(py::init<>());
#define PPD_CAP(field) caps.def_readwrite(#field, &Caps::field)
    PPD_CAP(maximum_atom_cells); PPD_CAP(maximum_union_atom_cells); PPD_CAP(maximum_ao_columns);
    PPD_CAP(maximum_topology_rows); PPD_CAP(maximum_owned_numerical_bytes); PPD_CAP(maximum_control_storage_bytes);
    PPD_CAP(maximum_worker_bytes); PPD_CAP(maximum_work_units);
#undef PPD_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicCorrelationPairPAODomainMemoryPlan");
#define PPD_PLAN(field) plan.def_readonly(#field, &Plan::field)
    PPD_PLAN(n_cells); PPD_PLAN(n_basis); PPD_PLAN(n_atoms); PPD_PLAN(atom_cell_count);
    PPD_PLAN(union_atom_count_upper); PPD_PLAN(ao_column_count_upper); PPD_PLAN(distinct_domain_owners);
    PPD_PLAN(borrowed_domain_numerical_bytes); PPD_PLAN(borrowed_topology_row_bytes); PPD_PLAN(caller_mapping_bytes);
    PPD_PLAN(mark_bytes); PPD_PLAN(retained_atom_bytes_upper); PPD_PLAN(retained_ao_bytes_upper);
    PPD_PLAN(peak_owned_numerical_bytes); PPD_PLAN(fixed_control_storage_bytes); PPD_PLAN(borrowed_owner_control_bytes);
    PPD_PLAN(worker_bytes); PPD_PLAN(required_node_memory_bytes); PPD_PLAN(validation_work_units);
    PPD_PLAN(union_work_units); PPD_PLAN(planned_work_units);
#undef PPD_PLAN
    auto diag = py::class_<Diagnostics>(m, "_PeriodicCorrelationPairPAODomainDiagnostics");
#define PPD_DIAG(field) diag.def_readonly(#field, &Diagnostics::field)
    PPD_DIAG(union_atom_count); PPD_DIAG(ao_column_count); PPD_DIAG(retained_numerical_bytes);
    PPD_DIAG(actual_peak_owned_numerical_bytes); PPD_DIAG(charged_work_units);
#undef PPD_DIAG
    const auto copy_bound = [](const Result& r) {
        if (r.memory().n_cells > 8 || r.memory().n_atoms > 8 || r.memory().n_basis > 12
            || r.diagnostics().union_atom_count > 64 || r.diagnostics().ao_column_count > 96)
            throw std::length_error("pair PAO diagnostic copy exceeds tiny dimensions");
    };
    py::class_<Result>(m, "_PeriodicCorrelationPairPAODomain")
        .def_property_readonly("contract_version", &Result::contract_version)
        .def_property_readonly("state", [](const Result& r) {
            (void)r.cell_ao_indices_data(); return std::const_pointer_cast<PeriodicRestrictedMeanFieldState>(r.state_handle());
        })
        .def_property_readonly("row_index", &Result::row_index).def_property_readonly("row", &Result::row)
        .def_property_readonly("energy_weight", &Result::energy_weight)
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("extended_ccsd_domain", &Result::extended_ccsd_domain)
        .def_property_readonly("hf_basis_source_authenticated", &Result::hf_basis_source_authenticated)
        .def_property_readonly("allocation_identity", &Result::allocation_identity)
        .def_property_readonly("topology_identity_sha256", &Result::topology_identity_sha256)
        .def_property_readonly("row_identity_sha256", &Result::row_identity_sha256)
        .def_property_readonly("home_domain_identity_sha256", &Result::home_domain_identity_sha256)
        .def_property_readonly("partner_domain_identity_sha256", &Result::partner_domain_identity_sha256)
        .def_property_readonly("mapping_identity_sha256", &Result::mapping_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("pair_pao_domain_identity_sha256", &Result::pair_pao_domain_identity_sha256)
        .def("atom", [](const Result& r, std::size_t i) { const auto a = r.atom(i); return py::make_tuple(a.cell, a.atom); })
        .def("column", [](const Result& r, std::size_t i) { const auto a = r.column(i); return py::make_tuple(a.cell, a.ao); })
        .def("translated_column", [](const Result& r, std::size_t i, std::uint64_t cell) {
            const auto a = r.translated_column(i, cell); return py::make_tuple(a.cell, a.ao);
        }, py::arg("index"), py::arg("common_translation_cell"))
        .def("atoms_copy", [copy_bound](const Result& r) {
            copy_bound(r); const auto* source = r.atom_cells_data(); const auto count = r.diagnostics().union_atom_count;
            py::array_t<std::uint64_t> out({py::ssize_t(count), py::ssize_t(2)});
            for (std::uint64_t i = 0; i < count; ++i) {
                out.mutable_data()[2*i] = source[i].cell; out.mutable_data()[2*i+1] = source[i].atom;
            }
            return out;
        })
        .def("columns_copy", [copy_bound](const Result& r) {
            copy_bound(r); const auto* source = r.cell_ao_indices_data(); const auto count = r.diagnostics().ao_column_count;
            py::array_t<std::uint64_t> out({py::ssize_t(count), py::ssize_t(2)});
            if (count) std::memcpy(out.mutable_data(), source, 16*count);
            return out;
        });
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationTranslationPairTopology& topology, std::uint64_t atoms) {
        if (!ref.state_handle() || ref.state().n_kpoints() > 8 || ref.state().n_basis() > 12
            || ref.state().n_correlated_occupied() > 4 || atoms > 8 || topology.row_count() > 512)
            throw std::length_error("pair PAO domain exceeds tiny diagnostic dimensions");
    };
    const auto bounded = [](const Plan& p) {
        if (p.peak_owned_numerical_bytes > (1U<<20) || p.required_node_memory_bytes > (128U<<20)
            || p.planned_work_units > 1000000000000ULL)
            throw std::length_error("pair PAO domain exceeds tiny diagnostic memory/work bounds");
    };
    m.def("_plan_periodic_correlation_pair_pao_domain", [tiny,bounded](
        const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationTranslationPairTopology& topology,
        std::uint64_t row, const PeriodicCorrelationOccupiedPAODomain& home, const PeriodicCorrelationOccupiedPAODomain& partner,
        std::uint64_t atoms, Inventory i, Caps c) {
        tiny(ref,topology,atoms); const auto p=plan_periodic_correlation_pair_pao_domain(ref,topology,row,home,partner,atoms,i,c);
        bounded(p); return p;
    }, py::arg("reference"),py::arg("topology"),py::arg("row_index"),py::arg("home"),py::arg("partner"),
       py::arg("atom_count"),py::arg("inventory"),py::arg("caps"));
    m.def("_make_periodic_correlation_pair_pao_domain", [tiny,bounded](
        const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationTranslationPairTopology& topology,
        std::uint64_t row, const PeriodicCorrelationOccupiedPAODomain& home, const PeriodicCorrelationOccupiedPAODomain& partner,
        const py::array& map, std::uint64_t atoms, Inventory i, Caps c) {
        tiny(ref,topology,atoms);
        bounded(plan_periodic_correlation_pair_pao_domain(ref,topology,row,home,partner,atoms,i,c));
        if (!map.dtype().is(py::dtype::of<std::uint64_t>()) || map.ndim()!=1 || !(map.flags()&py::array::c_style))
            throw std::invalid_argument("pair PAO mapping requires contiguous uint64[nao]");
        // All exact argument owners pinned with the GIL held; no callbacks,
        // forcecasts, array copies or released-GIL Python source mutation.
        return make_periodic_correlation_pair_pao_domain(ref,topology,row,home,partner,
            static_cast<const std::uint64_t*>(map.data()),map.size(),atoms,i,c);
    }, py::arg("reference"),py::arg("topology"),py::arg("row_index"),py::arg("home"),py::arg("partner"),
       py::arg("ao_to_atom").noconvert(),py::arg("atom_count"),py::arg("inventory"),py::arg("caps"));
}
