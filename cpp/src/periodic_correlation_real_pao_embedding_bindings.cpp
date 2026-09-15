// Bounded immutable common-to-pair PAO geometry diagnostics.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include "vibeqc/periodic_correlation_real_pao_embedding.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_real_pao_embedding(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationRealPAOEmbeddingOptions;
    using Live = PeriodicCorrelationRealPAOEmbeddingLiveInventory;
    using Caps = PeriodicCorrelationRealPAOEmbeddingCaps;
    using Plan = PeriodicCorrelationRealPAOEmbeddingMemoryPlan;
    using Diagnostics = PeriodicCorrelationRealPAOEmbeddingDiagnostics;
    using Result = PeriodicCorrelationRealPAOEmbedding;
    using Selection = PeriodicCorrelationVirtualBlockSelection;
    auto options = py::class_<Options>(m, "_PeriodicCorrelationRealPAOEmbeddingOptions");
    options.def(py::init<>());
#define RPE_OPTION(f) options.def_readwrite(#f, &Options::f)
    RPE_OPTION(maximum_cross_overlap_imaginary_norm); RPE_OPTION(maximum_embedding_gram_error);
    RPE_OPTION(maximum_pair_metric_error); RPE_OPTION(maximum_containment_norm);
    RPE_OPTION(maximum_pair_fock_error); RPE_OPTION(maximum_common_fock_error); RPE_OPTION(maximum_embedded_fock_error);
#undef RPE_OPTION
    auto live = py::class_<Live>(m, "_PeriodicCorrelationRealPAOEmbeddingLiveInventory");
    live.def(py::init<>());
#define RPE_LIVE(f) live.def_readwrite(#f, &Live::f)
    RPE_LIVE(other_live_numerical_bytes_per_worker); RPE_LIVE(other_live_control_bytes_per_worker);
    RPE_LIVE(fixed_backend_margin_bytes_per_worker);
#undef RPE_LIVE
    auto caps = py::class_<Caps>(m, "_PeriodicCorrelationRealPAOEmbeddingCaps");
    caps.def(py::init<>());
#define RPE_CAP(f) caps.def_readwrite(#f, &Caps::f)
    RPE_CAP(maximum_common_dimension); RPE_CAP(maximum_pair_dimension); RPE_CAP(maximum_owned_numerical_bytes);
    RPE_CAP(maximum_control_storage_bytes_per_worker); RPE_CAP(maximum_per_worker_inventoried_bytes);
    RPE_CAP(maximum_node_inventoried_bytes); RPE_CAP(maximum_work_units);
#undef RPE_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicCorrelationRealPAOEmbeddingMemoryPlan");
#define RPE_PLAN(f) plan.def_readonly(#f, &Plan::f)
    RPE_PLAN(n_cells); RPE_PLAN(n_basis); RPE_PLAN(common_dimension); RPE_PLAN(pair_dimension);
    RPE_PLAN(unique_domain_owners); RPE_PLAN(unique_space_owners); RPE_PLAN(unique_real_wrapper_owners);
    RPE_PLAN(live_domain_bytes); RPE_PLAN(live_space_bytes); RPE_PLAN(live_basis_bytes);
    RPE_PLAN(complete_borrowed_numerical_bytes); RPE_PLAN(output_numerical_bytes);
    RPE_PLAN(overlap_phase_bytes); RPE_PLAN(conversion_phase_bytes); RPE_PLAN(physical_audit_phase_bytes);
    RPE_PLAN(peak_owned_numerical_bytes); RPE_PLAN(control_storage_reservation_bytes);
    RPE_PLAN(replicas_per_node); RPE_PLAN(reference_base_node_bytes);
    RPE_PLAN(per_worker_inventoried_bytes); RPE_PLAN(required_node_memory_bytes);
    RPE_PLAN(source_payload_validation_work_units); RPE_PLAN(physical_audit_work_units); RPE_PLAN(work_units);
#undef RPE_PLAN
    plan.def_property_readonly("overlap", [](const Plan& p) { return p.overlap; });
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicCorrelationRealPAOEmbeddingDiagnostics");
#define RPE_DIAG(f) diagnostics.def_readonly(#f, &Diagnostics::f)
    RPE_DIAG(cross_overlap_imaginary_frobenius_upper_bound); RPE_DIAG(embedding_gram_frobenius_upper_bound);
    RPE_DIAG(pair_metric_frobenius_upper_bound); RPE_DIAG(containment_metric_quadratic_real);
    RPE_DIAG(containment_metric_quadratic_imaginary); RPE_DIAG(containment_s_absolute_upper_bound);
    RPE_DIAG(pair_original_fock_frobenius_upper_bound); RPE_DIAG(common_projected_fock_frobenius_upper_bound);
    RPE_DIAG(embedded_original_fock_frobenius_upper_bound);
#undef RPE_DIAG
    py::class_<Result>(m, "_PeriodicCorrelationRealPAOEmbedding")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("options", [](const Result& r) { return r.options(); })
        .def_property_readonly("state", &Result::state_handle)
        .def_property_readonly("common_selection", &Result::common_selection)
        .def_property_readonly("pair_selection", &Result::pair_selection)
        .def_property_readonly("common_basis_identity_sha256", &Result::common_basis_identity_sha256)
        .def_property_readonly("common_frame_identity_sha256", &Result::common_frame_identity_sha256)
        .def_property_readonly("pair_frame_identity_sha256", &Result::pair_frame_identity_sha256)
        .def_property_readonly("overlap_identity_sha256", &Result::overlap_identity_sha256)
        .def_property_readonly("source_payload_receipt_sha256", &Result::source_payload_receipt_sha256)
        .def_property_readonly("payload_sha256", &Result::payload_sha256)
        .def_property_readonly("identity_sha256", &Result::identity_sha256)
        .def_property_readonly("allocation_identity", &Result::allocation_identity)
        .def_property_readonly("production_distinct_domain_scaling", &Result::production_distinct_domain_scaling)
        .def("coefficient", &Result::coefficient).def("energy", &Result::energy)
        .def("coefficients_copy", [](const Result& r) {
            const auto n = r.memory().common_dimension, rank = r.memory().pair_dimension;
            if (n > 8 || rank > 8) throw std::length_error("real PAO embedding diagnostic copy exceeds tiny shape");
            const auto* values = r.coefficients_data();
            py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(rank)});
            if (n*rank) std::memcpy(out.mutable_data(),values,n*rank*sizeof(double));
            return out;
        })
        .def("energies_copy", [](const Result& r) {
            const auto rank = r.memory().pair_dimension;
            if (rank > 8) throw std::length_error("real PAO embedding diagnostic copy exceeds tiny shape");
            const auto* values = r.energies_data();
            py::array_t<double> out(static_cast<py::ssize_t>(rank));
            if (rank) std::memcpy(out.mutable_data(),values,rank*sizeof(double));
            return out;
        });
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicCorrelationPAODomain& common_domain, const PeriodicCorrelationRealPAOSpace& common_space,
        const Selection& common, const PeriodicCorrelationPAODomain& pair_domain,
        const PeriodicCorrelationRealPAOSpace& pair_space, const Selection& pair,
        const Options& options, const Live& live, const Caps& caps) {
        if (ref.state().n_kpoints() > 8 || ref.state().n_basis() > 12 || common.count > 8 || pair.count > 8)
            throw std::length_error("real PAO embedding diagnostic exceeds tiny dimensions");
        auto p = plan_periodic_correlation_real_pao_embedding(ref,basis,common_domain,common_space,common,
            pair_domain,pair_space,pair,options,live,caps);
        if (p.peak_owned_numerical_bytes > (8U << 20) || p.required_node_memory_bytes > (128U << 20)
            || p.work_units > 1000000000000ULL)
            throw std::length_error("real PAO embedding diagnostic exceeds tiny memory/work envelope");
        return p;
    };
    m.def("_plan_periodic_correlation_real_pao_embedding",tiny,
        py::arg("reference"),py::arg("basis"),py::arg("common_domain"),py::arg("common_space"),py::arg("common_selection"),
        py::arg("pair_domain"),py::arg("pair_space"),py::arg("pair_selection"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_correlation_real_pao_embedding",[tiny](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicCorrelationPAODomain& common_domain,
        const PeriodicCorrelationRealPAOSpace& common_space, Selection common,
        const PeriodicCorrelationPAODomain& pair_domain, const PeriodicCorrelationRealPAOSpace& pair_space, Selection pair,
        Options options, Live live, Caps caps) {
        (void)tiny(ref,basis,common_domain,common_space,common,pair_domain,pair_space,pair,options,live,caps);
        // Native immutable owners remain pinned. No arbitrary array or
        // callback enters this factory, and mutable controls are copied.
        return make_periodic_correlation_real_pao_embedding(ref,basis,common_domain,common_space,common,
            pair_domain,pair_space,pair,options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("common_domain"),py::arg("common_space"),py::arg("common_selection"),
      py::arg("pair_domain"),py::arg("pair_space"),py::arg("pair_selection"),py::arg("options"),py::arg("live"),py::arg("caps"));
}
