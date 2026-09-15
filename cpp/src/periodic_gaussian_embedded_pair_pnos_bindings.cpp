// Native immutable-source diagnostics: no caller G/F/T or mutable child.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_embedded_pair_pnos.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_embedded_pair_pnos(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicGaussianEmbeddedPairPNOOptions;
    using Live = PeriodicGaussianEmbeddedPairPNOLiveInventory;
    using Caps = PeriodicGaussianEmbeddedPairPNOCaps;
    using Plan = PeriodicGaussianEmbeddedPairPNOPlan;
    using Diagnostics = PeriodicGaussianEmbeddedPairPNODiagnostics;
    using Result = PeriodicGaussianEmbeddedPairPNOResult;
    using CountPlan = PeriodicGaussianEmbeddedPairPNOCountPlan;
    auto options = py::class_<Options>(m, "_PeriodicGaussianEmbeddedPairPNOOptions");
    options.def(py::init<>()).def_property("pno", [](const Options& o) { return o.pno; },
        [](Options& o, const PeriodicGaussianPairPNOOptions& p) { o.pno = p; });
#define GEPNO_OPTION(f) options.def_readwrite(#f, &Options::f)
    GEPNO_OPTION(maximum_diagonal_exchange_projection_norm); GEPNO_OPTION(maximum_fock_symmetry_projection_norm);
    GEPNO_OPTION(maximum_exported_gram_error); GEPNO_OPTION(maximum_exported_fock_error);
    GEPNO_OPTION(maximum_exported_subspace_error);
#undef GEPNO_OPTION
    auto live = py::class_<Live>(m, "_PeriodicGaussianEmbeddedPairPNOLiveInventory");
    live.def(py::init<>());
#define GEPNO_LIVE(f) live.def_readwrite(#f, &Live::f)
    GEPNO_LIVE(other_live_numerical_bytes_per_worker); GEPNO_LIVE(other_live_control_bytes_per_worker);
    GEPNO_LIVE(fixed_backend_margin_bytes_per_worker);
#undef GEPNO_LIVE
    auto caps = py::class_<Caps>(m, "_PeriodicGaussianEmbeddedPairPNOCaps");
    caps.def(py::init<>());
#define GEPNO_CAP(f) caps.def_readwrite(#f, &Caps::f)
    GEPNO_CAP(maximum_common_dimension); GEPNO_CAP(maximum_generation_dimension);
    GEPNO_CAP(maximum_owned_numerical_bytes); GEPNO_CAP(maximum_control_storage_bytes_per_worker);
    GEPNO_CAP(maximum_per_worker_inventoried_bytes); GEPNO_CAP(maximum_node_inventoried_bytes);
    GEPNO_CAP(maximum_integral_calls); GEPNO_CAP(maximum_work_units);
#undef GEPNO_CAP
    auto count = py::class_<CountPlan>(m, "_PeriodicGaussianEmbeddedPairPNOCountPlan");
#define GEPNO_COUNT(f) count.def_readonly(#f, &CountPlan::f)
    GEPNO_COUNT(integral_calls); GEPNO_COUNT(provider_work_units); GEPNO_COUNT(numerical_work_units); GEPNO_COUNT(work_units);
    GEPNO_COUNT(projection_phase_bytes); GEPNO_COUNT(amplitude_phase_bytes); GEPNO_COUNT(density_phase_bytes);
    GEPNO_COUNT(semicanonical_phase_upper_bytes); GEPNO_COUNT(export_phase_upper_bytes);
    GEPNO_COUNT(peak_owned_numerical_bytes); GEPNO_COUNT(retained_output_upper_bytes);
    GEPNO_COUNT(retained_generation_coefficient_upper_bytes);
    GEPNO_COUNT(borrowed_basis_bytes); GEPNO_COUNT(borrowed_provider_row_bytes); GEPNO_COUNT(borrowed_embedding_bytes);
    GEPNO_COUNT(complete_borrowed_numerical_bytes); GEPNO_COUNT(control_storage_reservation_bytes);
#undef GEPNO_COUNT
    auto plan = py::class_<Plan>(m, "_PeriodicGaussianEmbeddedPairPNOPlan");
#define GEPNO_PLAN(f) plan.def_readonly(#f, &Plan::f)
    GEPNO_PLAN(occupied_count); GEPNO_PLAN(common_virtual_dimension); GEPNO_PLAN(generation_dimension);
    GEPNO_PLAN(occupied_slot_i); GEPNO_PLAN(occupied_slot_j); GEPNO_PLAN(integral_calls);
    GEPNO_PLAN(provider_work_units); GEPNO_PLAN(numerical_work_units); GEPNO_PLAN(work_units);
    GEPNO_PLAN(projection_phase_bytes); GEPNO_PLAN(amplitude_phase_bytes); GEPNO_PLAN(density_phase_bytes);
    GEPNO_PLAN(semicanonical_phase_upper_bytes); GEPNO_PLAN(export_phase_upper_bytes);
    GEPNO_PLAN(peak_owned_numerical_bytes); GEPNO_PLAN(retained_output_upper_bytes);
    GEPNO_PLAN(retained_generation_coefficient_upper_bytes);
    GEPNO_PLAN(borrowed_basis_bytes); GEPNO_PLAN(borrowed_provider_row_bytes); GEPNO_PLAN(borrowed_embedding_bytes);
    GEPNO_PLAN(complete_borrowed_numerical_bytes); GEPNO_PLAN(control_storage_reservation_bytes);
    GEPNO_PLAN(replicas_per_node); GEPNO_PLAN(reference_base_node_bytes);
    GEPNO_PLAN(per_worker_inventoried_bytes); GEPNO_PLAN(required_node_memory_bytes);
#undef GEPNO_PLAN
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicGaussianEmbeddedPairPNODiagnostics");
    diagnostics.def_property_readonly("generation", [](const Diagnostics& d) { return d.generation; });
#define GEPNO_DIAG(f) diagnostics.def_readonly(#f, &Diagnostics::f)
    GEPNO_DIAG(retained_dimension); GEPNO_DIAG(retained_output_bytes);
    GEPNO_DIAG(retained_generation_coefficient_bytes);
    GEPNO_DIAG(actual_semicanonical_phase_bytes); GEPNO_DIAG(actual_export_phase_bytes);
    GEPNO_DIAG(maximum_raw_exchange_transpose_defect); GEPNO_DIAG(diagonal_exchange_projection_frobenius_upper_bound);
    GEPNO_DIAG(maximum_raw_fock_transpose_defect); GEPNO_DIAG(fock_symmetry_projection_frobenius_upper_bound);
    GEPNO_DIAG(exported_gram_frobenius_upper_bound); GEPNO_DIAG(exported_fock_frobenius_upper_bound);
    GEPNO_DIAG(exported_subspace_frobenius_upper_bound);
#undef GEPNO_DIAG
    auto result = py::class_<Result>(m, "_PeriodicGaussianEmbeddedPairPNOResult");
    result.def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("options", [](const Result& r) { return r.options(); })
        .def_property_readonly("state", &Result::state_handle).def_property_readonly("context", &Result::context_handle)
        .def_property_readonly("occupied_i", [](const Result& r) { const auto i = r.occupied_i(); return py::make_tuple(i.occupied_index, i.cell); })
        .def_property_readonly("occupied_j", [](const Result& r) { const auto j = r.occupied_j(); return py::make_tuple(j.occupied_index, j.cell); });
#define GEPNO_RESULT(f) result.def_property_readonly(#f, &Result::f)
    GEPNO_RESULT(diagonal_pair); GEPNO_RESULT(complete_generation_pair_space); GEPNO_RESULT(complete_common_virtual_space);
    GEPNO_RESULT(coupled_mp2_solution); GEPNO_RESULT(production_dlpno); GEPNO_RESULT(infinite_source_accuracy_certified);
    GEPNO_RESULT(identity_sha256); GEPNO_RESULT(payload_sha256); GEPNO_RESULT(basis_identity_sha256);
    GEPNO_RESULT(provider_identity_sha256); GEPNO_RESULT(hf_reference_source_identity_sha256);
    GEPNO_RESULT(common_exchange_integral_identity_sha256);
    GEPNO_RESULT(embedding_identity_sha256); GEPNO_RESULT(raw_exchange_identity_sha256);
    GEPNO_RESULT(projected_exchange_identity_sha256); GEPNO_RESULT(raw_fock_identity_sha256);
    GEPNO_RESULT(projected_fock_identity_sha256); GEPNO_RESULT(initial_amplitude_identity_sha256);
    GEPNO_RESULT(density_identity_sha256); GEPNO_RESULT(source_payload_receipt_sha256);
#undef GEPNO_RESULT
    result.def("coefficients_copy", [](const Result& r) {
        const auto& values = r.coefficients();
        const auto n = r.memory().common_virtual_dimension, t = r.diagnostics().retained_dimension;
        if (n > 8 || t > 8) throw std::length_error("embedded pair PNO copy exceeds tiny dimension");
        py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(t)});
        if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
        return out;
    }).def("generation_coefficients_copy", [](const Result& r) {
        const auto& values = r.generation_coefficients();
        const auto rows = r.memory().generation_dimension, t = r.diagnostics().retained_dimension;
        if (rows > 8 || t > 8) throw std::length_error("embedded pair PNO generation copy exceeds tiny dimension");
        py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(t)});
        if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
        out.attr("setflags")(py::arg("write")=false);
        return out;
    }).def("energies_copy", [](const Result& r) {
        const auto& values = r.energies();
        if (values.size() > 8) throw std::length_error("embedded pair PNO copy exceeds tiny dimension");
        py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
        if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
        return out;
    }).def("original_pno_occupations_copy", [](const Result& r) {
        const auto& values = r.original_pno_occupations();
        if (values.size() > 8) throw std::length_error("embedded pair PNO copy exceeds tiny dimension");
        py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
        if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
        return out;
    });
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicCorrelationRealPAOEmbedding& embedding,
        const Options& options) {
        if (ref.state().n_kpoints() > 8 || ref.state().n_basis() > 12 || basis.memory().occupied_count > 8
            || basis.memory().virtual_count > 8 || embedding.memory().pair_dimension > 8
            || options.pno.pno_eigensolver.max_sweeps > 200 || options.pno.semicanonical_eigensolver.max_sweeps > 200)
            throw std::length_error("embedded pair PNO diagnostic exceeds tiny dimensions or sweeps");
    };
    const auto bounded = [](const Plan& p) {
        if (p.peak_owned_numerical_bytes > (16U << 20) || p.required_node_memory_bytes > (128U << 20)
            || p.work_units > 100000000000000ULL)
            throw std::length_error("embedded pair PNO diagnostic exceeds tiny memory/work envelope");
    };
    m.def("_plan_periodic_gaussian_embedded_pair_pnos", [tiny, bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicCorrelationRealPAOEmbedding& embedding, std::uint64_t i, std::uint64_t j,
        Options options, Live live, Caps caps) {
        tiny(ref, basis, embedding, options);
        auto p = plan_periodic_gaussian_embedded_pair_pnos(ref, basis, provider, embedding, i, j, options, live, caps);
        bounded(p); return p;
    }, py::arg("reference"), py::arg("basis"), py::arg("provider"), py::arg("embedding"),
       py::arg("occupied_slot_i"), py::arg("occupied_slot_j"), py::arg("options"), py::arg("live"), py::arg("caps"));
    m.def("_make_periodic_gaussian_embedded_pair_pnos", [tiny, bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicCorrelationRealPAOEmbedding& embedding, std::uint64_t i, std::uint64_t j,
        Options options, Live live, Caps caps) {
        tiny(ref, basis, embedding, options);
        bounded(plan_periodic_gaussian_embedded_pair_pnos(ref, basis, provider, embedding, i, j, options, live, caps));
        // Keep immutable native inputs pinned and hold the GIL. All mutable
        // controls are copied; this diagnostic has no callback or array casts.
        return make_periodic_gaussian_embedded_pair_pnos(ref, basis, provider, embedding, i, j, options, live, caps);
    }, py::arg("reference"), py::arg("basis"), py::arg("provider"), py::arg("embedding"),
       py::arg("occupied_slot_i"), py::arg("occupied_slot_j"), py::arg("options"), py::arg("live"), py::arg("caps"));
}
