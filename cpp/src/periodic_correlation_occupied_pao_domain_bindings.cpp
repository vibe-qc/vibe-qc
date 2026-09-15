// Tiny typed-owner diagnostics only; included by bindings.cpp.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_correlation_occupied_pao_domain.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_occupied_pao_domain(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationOccupiedPAODomainOptions;
    using Inventory = PeriodicCorrelationOccupiedPAODomainInventory;
    using Caps = PeriodicCorrelationOccupiedPAODomainCaps;
    using Plan = PeriodicCorrelationOccupiedPAODomainMemoryPlan;
    using Diagnostics = PeriodicCorrelationOccupiedPAODomainDiagnostics;
    using Result = PeriodicCorrelationOccupiedPAODomain;
    auto options = py::class_<Options>(m, "_PeriodicCorrelationOccupiedPAODomainOptions");
    options.def(py::init<>());
#define OPD_OPTION(field) options.def_readwrite(#field, &Options::field)
    OPD_OPTION(full_domain); OPD_OPTION(mulliken_population_cutoff); OPD_OPTION(pao_tail_cutoff);
    OPD_OPTION(normalization_tolerance); OPD_OPTION(maximum_population_imaginary_magnitude);
    OPD_OPTION(maximum_pao_imaginary_magnitude); OPD_OPTION(maximum_negative_absolute_population);
    OPD_OPTION(maximum_seed_omitted_absolute_population); OPD_OPTION(maximum_expanded_omitted_absolute_population);
#undef OPD_OPTION
    py::class_<Inventory>(m, "_PeriodicCorrelationOccupiedPAODomainInventory").def(py::init<>())
        .def_readwrite("other_live_numerical_bytes", &Inventory::other_live_numerical_bytes)
        .def_readwrite("other_live_control_bytes", &Inventory::other_live_control_bytes)
        .def_readwrite("backend_allowance_bytes", &Inventory::backend_allowance_bytes);
    auto caps = py::class_<Caps>(m, "_PeriodicCorrelationOccupiedPAODomainCaps"); caps.def(py::init<>());
#define OPD_CAP(field) caps.def_readwrite(#field, &Caps::field)
    OPD_CAP(maximum_atom_cells); OPD_CAP(maximum_seed_atom_cells); OPD_CAP(maximum_expanded_atom_cells);
    OPD_CAP(maximum_owned_numerical_bytes); OPD_CAP(maximum_control_storage_bytes);
    OPD_CAP(maximum_worker_bytes); OPD_CAP(maximum_work_units);
#undef OPD_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicCorrelationOccupiedPAODomainMemoryPlan");
#define OPD_PLAN(field) plan.def_readonly(#field, &Plan::field)
    OPD_PLAN(n_cells); OPD_PLAN(n_basis); OPD_PLAN(n_atoms); OPD_PLAN(n_active); OPD_PLAN(atom_cell_count);
    OPD_PLAN(seed_count_upper); OPD_PLAN(expanded_count_upper); OPD_PLAN(borrowed_optimizer_numerical_bytes);
    OPD_PLAN(borrowed_wannier_numerical_bytes); OPD_PLAN(caller_mapping_bytes); OPD_PLAN(retained_population_bytes);
    OPD_PLAN(retained_domain_bytes_upper); OPD_PLAN(coefficient_panel_bytes); OPD_PLAN(shared_scratch_bytes);
    OPD_PLAN(selection_mark_bytes); OPD_PLAN(tail_score_bytes_upper); OPD_PLAN(population_phase_owned_bytes);
    OPD_PLAN(tail_phase_owned_bytes_upper); OPD_PLAN(output_phase_owned_bytes_upper);
    OPD_PLAN(peak_owned_numerical_bytes); OPD_PLAN(fixed_control_storage_bytes); OPD_PLAN(borrowed_owner_control_bytes);
    OPD_PLAN(worker_bytes); OPD_PLAN(required_node_memory_bytes); OPD_PLAN(validation_work_units);
    OPD_PLAN(population_work_units); OPD_PLAN(tail_work_units_upper); OPD_PLAN(planned_work_units);
#undef OPD_PLAN
    auto diag = py::class_<Diagnostics>(m, "_PeriodicCorrelationOccupiedPAODomainDiagnostics");
#define OPD_DIAG(field) diag.def_readonly(#field, &Diagnostics::field)
    OPD_DIAG(seed_count); OPD_DIAG(expanded_count); OPD_DIAG(retained_numerical_bytes);
    OPD_DIAG(actual_peak_owned_numerical_bytes); OPD_DIAG(charged_work_units); OPD_DIAG(projected_pao_columns);
    OPD_DIAG(population_sum); OPD_DIAG(normalization_residual); OPD_DIAG(maximum_population_imaginary_magnitude);
    OPD_DIAG(maximum_pao_imaginary_magnitude); OPD_DIAG(negative_absolute_population);
    OPD_DIAG(seed_omitted_absolute_population); OPD_DIAG(expanded_omitted_absolute_population);
    OPD_DIAG(maximum_pao_tail_strength);
#undef OPD_DIAG
    const auto copy_bound = [](const Result& r) {
        if (r.memory().n_cells > 8 || r.memory().n_atoms > 8 || r.memory().atom_cell_count > 64)
            throw std::length_error("occupied PAO domain diagnostic copy exceeds tiny dimensions");
    };
    py::class_<Result>(m, "_PeriodicCorrelationOccupiedPAODomain")
        .def_property_readonly("contract_version", &Result::contract_version)
        .def_property_readonly("home_occupied_index", &Result::home_occupied_index)
        .def_property_readonly("hf_basis_source_authenticated", &Result::hf_basis_source_authenticated)
        .def_property_readonly("extended_ccsd_domain", &Result::extended_ccsd_domain)
        .def_property_readonly("state", [](const Result& r) {
            (void) r.populations_data(); return std::const_pointer_cast<PeriodicRestrictedMeanFieldState>(r.state_handle());
        })
        .def_property_readonly("allocation_identity", &Result::allocation_identity)
        .def_property_readonly("optimizer_identity_sha256", &Result::optimizer_identity_sha256)
        .def_property_readonly("wannier_identity_sha256", &Result::wannier_identity_sha256)
        .def_property_readonly("mapping_identity_sha256", &Result::mapping_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("occupied_pao_domain_identity_sha256", &Result::occupied_pao_domain_identity_sha256)
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("options", [](const Result& r) { return r.options(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def("population", &Result::population, py::arg("cell"), py::arg("atom"))
        .def("seed", [](const Result& r, std::size_t i) { const auto x = r.seed(i); return py::make_tuple(x.cell, x.atom); })
        .def("expanded", [](const Result& r, std::size_t i) { const auto x = r.expanded(i); return py::make_tuple(x.cell, x.atom); })
        .def("populations_copy", [copy_bound](const Result& r) {
            copy_bound(r); const auto* source = r.populations_data();
            py::array_t<double> out({py::ssize_t(r.memory().n_cells), py::ssize_t(r.memory().n_atoms)});
            std::memcpy(out.mutable_data(), source, r.memory().retained_population_bytes); return out;
        })
        .def("seeds_copy", [copy_bound](const Result& r) {
            copy_bound(r); const auto* source = r.seeds_data(); const auto count = r.diagnostics().seed_count;
            py::array_t<std::uint64_t> out({py::ssize_t(count), py::ssize_t(2)});
            for (std::uint64_t i = 0; i < count; ++i) {
                out.mutable_data()[2*i] = source[i].cell; out.mutable_data()[2*i+1] = source[i].atom;
            }
            return out;
        })
        .def("expanded_copy", [copy_bound](const Result& r) {
            copy_bound(r); const auto* source = r.expanded_data(); const auto count = r.diagnostics().expanded_count;
            py::array_t<std::uint64_t> out({py::ssize_t(count), py::ssize_t(2)});
            for (std::uint64_t i = 0; i < count; ++i) {
                out.mutable_data()[2*i] = source[i].cell; out.mutable_data()[2*i+1] = source[i].atom;
            }
            return out;
        });
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref, std::uint64_t atoms) {
        if (!ref.state_handle() || ref.state().n_kpoints() > 8 || ref.state().n_basis() > 12
            || ref.state().n_correlated_occupied() > 4 || atoms > 8)
            throw std::length_error("occupied PAO domain exceeds tiny diagnostic dimensions");
    };
    const auto bounded = [](const Plan& p) {
        if (p.peak_owned_numerical_bytes > (8U << 20) || p.required_node_memory_bytes > (128U << 20)
            || p.planned_work_units > 100000000000000ULL)
            throw std::length_error("occupied PAO domain exceeds tiny diagnostic memory/work bounds");
    };
    m.def("_plan_periodic_correlation_occupied_pao_domain", [tiny, bounded](
        const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationIAOOptimizerResult& opt,
        const PeriodicCorrelationWannier& w, std::uint64_t atoms, Options o, Inventory i, Caps c) {
        tiny(ref, atoms); const auto p = plan_periodic_correlation_occupied_pao_domain(ref, opt, w, atoms, o, i, c);
        bounded(p); return p;
    }, py::arg("reference"), py::arg("optimizer"), py::arg("wannier"), py::arg("atom_count"),
        py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_select_periodic_correlation_occupied_pao_domain", [tiny, bounded](
        const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationIAOOptimizerResult& opt,
        const PeriodicCorrelationWannier& w, const py::array& map, std::uint64_t atoms,
        std::uint64_t occupied, Options o, Inventory i, Caps c) {
        tiny(ref, atoms);
        bounded(plan_periodic_correlation_occupied_pao_domain(ref, opt, w, atoms, o, i, c));
        if (!map.dtype().is(py::dtype::of<std::uint64_t>()) || map.ndim() != 1
            || !(map.flags() & py::array::c_style))
            throw std::invalid_argument("occupied PAO domain mapping requires contiguous uint64[nao]");
        // GIL remains held; original typed numerical owners and the exact
        // NumPy mapping owner stay pinned for the complete native call.
        return select_periodic_correlation_occupied_pao_domain(ref, opt, w,
            static_cast<const std::uint64_t*>(map.data()), map.size(), atoms, occupied, o, i, c);
    }, py::arg("reference"), py::arg("optimizer"), py::arg("wannier"), py::arg("ao_to_atom").noconvert(),
        py::arg("atom_count"), py::arg("home_active_occupied_index"), py::arg("options"),
        py::arg("inventory"), py::arg("caps"));
}
