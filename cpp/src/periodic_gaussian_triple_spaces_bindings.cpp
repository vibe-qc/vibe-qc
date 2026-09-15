// Actual-source TNO batch diagnostics. No supplied numerical tensors.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_triple_spaces.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_triple_spaces(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = BoundedRestrictedTNOOptions;
    using Live = PeriodicGaussianTripleSpacesLiveInventory;
    using Caps = PeriodicGaussianTripleSpacesCaps;
    using Plan = PeriodicGaussianTripleSpacesPlan;
    using Diagnostics = PeriodicGaussianTripleSpacesDiagnostics;
    using Stage = PeriodicGaussianTripleSpacesStage;
    using Progress = PeriodicGaussianTripleSpacesProgress;
    using Result = PeriodicGaussianTripleSpaces;
    auto live = py::class_<Live>(m, "_PeriodicGaussianTripleSpacesLiveInventory");
    live.def(py::init<>());
#define GTS_LIVE(f) live.def_readwrite(#f, &Live::f)
    GTS_LIVE(other_live_numerical_bytes_per_worker); GTS_LIVE(other_live_control_bytes_per_worker);
    GTS_LIVE(fixed_backend_margin_bytes_per_worker);
#undef GTS_LIVE
    auto caps = py::class_<Caps>(m, "_PeriodicGaussianTripleSpacesCaps");
    caps.def(py::init<>());
#define GTS_CAP(f) caps.def_readwrite(#f, &Caps::f)
    GTS_CAP(geometry); GTS_CAP(maximum_pair_count); GTS_CAP(maximum_triple_count);
    GTS_CAP(maximum_owned_numerical_bytes); GTS_CAP(maximum_control_storage_bytes_per_worker);
    GTS_CAP(maximum_per_worker_inventoried_bytes); GTS_CAP(maximum_node_inventoried_bytes);
    GTS_CAP(maximum_progress_callbacks); GTS_CAP(maximum_work_units);
#undef GTS_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicGaussianTripleSpacesPlan");
#define GTS_PLAN(f) plan.def_readonly(#f, &Plan::f)
    GTS_PLAN(occupied_count); GTS_PLAN(common_virtual_dimension); GTS_PLAN(pair_count); GTS_PLAN(triple_count);
    GTS_PLAN(borrowed_basis_bytes); GTS_PLAN(borrowed_provider_row_bytes); GTS_PLAN(borrowed_mp2_numerical_bytes);
    GTS_PLAN(borrowed_ccsd_numerical_bytes); GTS_PLAN(borrowed_mp2_control_bytes); GTS_PLAN(borrowed_ccsd_control_bytes);
    GTS_PLAN(retained_triple_output_upper_bytes); GTS_PLAN(serialized_leaf_owned_ceiling); GTS_PLAN(peak_owned_numerical_bytes);
    GTS_PLAN(retained_triple_object_bytes); GTS_PLAN(retained_triple_seal_bytes); GTS_PLAN(leaf_fixed_control_reservation_bytes);
    GTS_PLAN(control_storage_reservation_bytes); GTS_PLAN(replicas_per_node); GTS_PLAN(reference_base_node_bytes);
    GTS_PLAN(per_worker_inventoried_bytes); GTS_PLAN(required_node_memory_bytes); GTS_PLAN(progress_callback_upper_bound);
    GTS_PLAN(driver_work_units); GTS_PLAN(work_units_upper_bound); GTS_PLAN(exact_rank_plan_output_upper_bytes);
    GTS_PLAN(exact_rank_plan_peak_owned_bytes); GTS_PLAN(exact_rank_plan_geometry_work_units);
#undef GTS_PLAN
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicGaussianTripleSpacesDiagnostics");
#define GTS_DIAG(f) diagnostics.def_readonly(#f, &Diagnostics::f)
    GTS_DIAG(completed_triples); GTS_DIAG(completed_progress_callbacks); GTS_DIAG(retained_numerical_bytes);
    GTS_DIAG(geometry_work_units); GTS_DIAG(empty_union_count); GTS_DIAG(empty_retained_space_count);
    GTS_DIAG(total_retained_rank); GTS_DIAG(total_union_rank); GTS_DIAG(maximum_retained_rank);
    GTS_DIAG(all_equal_tuple_count); GTS_DIAG(complete_common_finite_torus_basis);
    GTS_DIAG(all_unions_full_common_rank); GTS_DIAG(all_retained_full_common_rank);
#undef GTS_DIAG
    py::enum_<Stage>(m, "_PeriodicGaussianTripleSpacesStage")
        .value("BEGIN", Stage::Begin).value("TRIPLE_COMPLETE", Stage::TripleComplete).value("FINISHED", Stage::Finished);
    auto progress = py::class_<Progress>(m, "_PeriodicGaussianTripleSpacesProgress");
#define GTS_PROGRESS(f) progress.def_readonly(#f, &Progress::f)
    GTS_PROGRESS(stage); GTS_PROGRESS(callback_count); GTS_PROGRESS(completed_triples); GTS_PROGRESS(occupied);
    GTS_PROGRESS(union_rank); GTS_PROGRESS(retained_rank); GTS_PROGRESS(retained_numerical_bytes); GTS_PROGRESS(geometry_work_units);
#undef GTS_PROGRESS
    py::class_<Result>(m, "_PeriodicGaussianTripleSpaces")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("options", [](const Result& r) { return r.options(); })
        .def_property_readonly("state", &Result::state_handle)
        .def_property_readonly("context", &Result::context_handle)
        .def_property_readonly("identity_sha256", &Result::identity_sha256)
        .def_property_readonly("spaces_identity_sha256", &Result::spaces_identity_sha256)
        .def_property_readonly("ccsd_identity_sha256", &Result::ccsd_identity_sha256)
        .def_property_readonly("ccsd_amplitude_payload_sha256", &Result::ccsd_amplitude_payload_sha256)
        .def_property_readonly("warmstart_identity_sha256", &Result::warmstart_identity_sha256)
        .def_property_readonly("pair_spaces_identity_sha256", &Result::pair_spaces_identity_sha256)
        .def_property_readonly("basis_identity_sha256", &Result::basis_identity_sha256)
        .def_property_readonly("provider_identity_sha256", &Result::provider_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256", &Result::hf_reference_source_identity_sha256)
        .def_property_readonly("allocation_identity", &Result::allocation_identity)
        .def_property_readonly("matched_finite_gaussian_hf_recipe", &Result::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("production_dlpno", &Result::production_dlpno)
        .def_property_readonly("includes_triples_energy", &Result::includes_triples_energy)
        .def_property_readonly("infinite_source_accuracy_certified", &Result::infinite_source_accuracy_certified)
        // The native generic TNO and semicanonical owners never escape.
        // A mutable generic wrapper could invalidate the actual-source seal.
        .def("triple_diagnostics", [](const Result& r, U i, U j, U k) {
            const auto& t = r.triple(i,j,k);
            py::dict out;
#define GTS_TRIPLE(f) out[#f] = t.f
            GTS_TRIPLE(union_rank); GTS_TRIPLE(retained_rank); GTS_TRIPLE(usable);
            GTS_TRIPLE(maximum_input_orthonormality_error); GTS_TRIPLE(union_orthonormality_error);
            GTS_TRIPLE(union_column_reconstruction_frobenius_error); GTS_TRIPLE(minimum_occupation);
            GTS_TRIPLE(discarded_occupation_sum); GTS_TRIPLE(negative_occupation_count); GTS_TRIPLE(amplitude_scale_exponent);
            GTS_TRIPLE(output_numerical_bytes); GTS_TRIPLE(input_identity_sha256); GTS_TRIPLE(result_identity_sha256);
#undef GTS_TRIPLE
            out["memory"] = py::cast(BoundedRestrictedTNOMemoryPlan(t.memory));
            out["union_audit"] = py::cast(BoundedRestrictedTNOEigensystemAudit(t.union_audit));
            out["density_audit"] = py::cast(BoundedRestrictedTNOEigensystemAudit(t.density_audit));
            out["semicanonical_input_orthonormality_error"] = t.semicanonical.input_orthonormality_error;
            out["semicanonical_output_orthonormality_error"] = t.semicanonical.output_orthonormality_error;
            out["semicanonical_projected_fock_relative_residual"] = t.semicanonical.projected_fock_relative_residual;
            out["semicanonical_subspace_projector_frobenius_error"] = t.semicanonical.subspace_projector_frobenius_error;
            out["semicanonical_full_space_relative_residual"] = t.semicanonical.full_space_relative_residual;
            return out;
        })
        .def("triple_coefficients_copy", [](const Result& r, U i, U j, U k) {
            const auto& t = r.triple(i,j,k);
            const auto n = r.memory().common_virtual_dimension, rank = t.retained_rank;
            if (n > 4 || rank > 4 || t.semicanonical.coefficients.size() != n*rank)
                throw std::length_error("Gaussian triple-space diagnostic coefficient copy exceeds tiny extent");
            py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(rank)});
            if (n*rank) std::memcpy(out.mutable_data(), t.semicanonical.coefficients.data(), n*rank*sizeof(double));
            return out;
        })
        .def("triple_energies_copy", [](const Result& r, U i, U j, U k) {
            const auto& t = r.triple(i,j,k);
            if (t.retained_rank > 4 || t.semicanonical.energies.size() != t.retained_rank)
                throw std::length_error("Gaussian triple-space diagnostic energy copy exceeds tiny extent");
            py::array_t<double> out(static_cast<py::ssize_t>(t.retained_rank));
            if (t.retained_rank) std::memcpy(out.mutable_data(), t.semicanonical.energies.data(), t.retained_rank*sizeof(double));
            return out;
        })
        .def("triple_occupations_copy", [](const Result& r, U i, U j, U k) {
            const auto& t = r.triple(i,j,k);
            if (t.union_rank > 4 || t.occupations.size() != t.union_rank)
                throw std::length_error("Gaussian triple-space diagnostic occupation copy exceeds tiny extent");
            py::array_t<double> out(static_cast<py::ssize_t>(t.union_rank));
            if (t.union_rank) std::memcpy(out.mutable_data(), t.occupations.data(), t.union_rank*sizeof(double));
            return out;
        });
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianRealLocalProvider& provider, const PeriodicGaussianPairMP2Result& mp2,
        const PeriodicGaussianPairCCSDResult& ccsd, const Options& options, const Live& live, const Caps& caps) {
        if (ref.state().n_kpoints() > 8 || basis.memory().occupied_count > 4 || basis.memory().virtual_count > 4
            || options.union_eigensolver.max_sweeps > 200 || options.density_eigensolver.max_sweeps > 200
            || options.fock_eigensolver.max_sweeps > 200)
            throw std::length_error("Gaussian triple-space diagnostic exceeds tiny dimensions or eigensolver sweeps");
        auto p = plan_periodic_gaussian_triple_spaces(ref, basis, provider, mp2, ccsd, options, live, caps);
        if (p.peak_owned_numerical_bytes > (16U << 20) || p.required_node_memory_bytes > (128U << 20)
            || p.work_units_upper_bound > 1000000000000000ULL || p.progress_callback_upper_bound > 1024)
            throw std::length_error("Gaussian triple-space diagnostic exceeds tiny memory, work or callbacks");
        return p;
    };
    m.def("_plan_periodic_gaussian_triple_spaces", tiny, py::arg("reference"), py::arg("basis"),
        py::arg("provider"), py::arg("mp2"), py::arg("ccsd"), py::arg("options"), py::arg("live"), py::arg("caps"));
    m.def("_make_periodic_gaussian_triple_spaces", [tiny](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairMP2Result& mp2, const PeriodicGaussianPairCCSDResult& ccsd,
        Options options, Live live, Caps caps, py::object callback) {
        (void)tiny(ref,basis,provider,mp2,ccsd,options,live,caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian triple-space progress must be callable");
        const auto progress = [](const Progress& p, void* context) {
            (*static_cast<py::object*>(context))(py::cast(Progress(p)));
        };
        // Exact immutable native owners are pinned through return. All mutable
        // controls are copied before callbacks. No numerical Python callback.
        auto result = make_periodic_gaussian_triple_spaces(ref,basis,provider,mp2,ccsd,options,live,caps,
            callback.is_none() ? nullptr : +progress, callback.is_none() ? nullptr : &callback);
        validate_periodic_gaussian_triple_spaces(ref,basis,provider,mp2,ccsd,result);
        return result;
    }, py::arg("reference"), py::arg("basis"), py::arg("provider"), py::arg("mp2"), py::arg("ccsd"),
       py::arg("options"), py::arg("live"), py::arg("caps"), py::arg("progress") = py::none());
}
