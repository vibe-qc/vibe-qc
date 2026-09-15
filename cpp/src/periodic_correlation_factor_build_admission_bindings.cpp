// Internal diagnostic bindings for the native periodic factor-build
// admission contract. The production driver consumes the C++ API directly.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/periodic_correlation_factor_build_admission.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_factor_build_admission(py::module_& m) {
    using Admission =
        vibeqc::PeriodicCorrelationFactorBuildAdmissionCode;
    using Backend =
        vibeqc::PeriodicCorrelationFactorBuildBackendInventory;
    using Backing = vibeqc::PeriodicCorrelationFactorBackingMode;
    using Census = vibeqc::PeriodicCorrelationFactorBuildCensus;
    using Codec = vibeqc::PeriodicCorrelationFactorBuildCodecInventory;
    using Components =
        vibeqc::PeriodicCorrelationFactorBuildComponents;
    using Config = vibeqc::PeriodicCorrelationFactorBuildConfig;
    using Phase = vibeqc::PeriodicCorrelationFactorBuildPhase;
    using PhaseEstimate =
        vibeqc::PeriodicCorrelationFactorBuildPhaseEstimate;
    using Plan = vibeqc::PeriodicCorrelationFactorBuildPlan;
    using Producer = vibeqc::PeriodicCorrelationFactorProducerMode;
    using Publisher = vibeqc::PeriodicCorrelationFactorPublisherMode;
    using QRecord = vibeqc::PeriodicCorrelationFactorBuildQRecord;
    using RowConvention =
        vibeqc::PeriodicCorrelationFactorRowConvention;
    using Shape = vibeqc::PeriodicCorrelationFactorBuildShape;
    using ShortRange = vibeqc::PeriodicCorrelationShortRangePolicy;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using TransferConvention =
        vibeqc::PeriodicCorrelationTransferRepresentativeConvention;

    m.attr("_PERIODIC_CORRELATION_FACTOR_BUILD_ADMISSION_CONTRACT_VERSION") =
        py::int_(
            vibeqc::kPeriodicCorrelationFactorBuildAdmissionContractVersion);

    py::enum_<Producer>(m, "_PeriodicCorrelationFactorProducerMode")
        .value(
            "FULL_COULOMB_ALL_RECIPROCAL_REFERENCE",
            Producer::FullCoulombAllReciprocalReference)
        .value("RANGE_SEPARATED_GDF", Producer::RangeSeparatedGdf);
    py::enum_<ShortRange>(m, "_PeriodicCorrelationShortRangePolicy")
        .value(
            "DISABLED_ALL_RECIPROCAL",
            ShortRange::DisabledAllReciprocal)
        .value(
            "DOUBLE_CELL_RECOMPUTE_PER_K_PAIR_REFERENCE",
            ShortRange::DoubleCellRecomputePerKPairReference)
        .value(
            "DOUBLE_CELL_BVK_BLOCKED_TRANSFORM",
            ShortRange::DoubleCellBvkBlockedTransform)
        .value(
            "UNSUPPORTED_SINGLE_CELL_GAMMA_SUM",
            ShortRange::UnsupportedSingleCellGammaSum);
    py::enum_<RowConvention>(m, "_PeriodicCorrelationFactorRowConvention")
        .value(
            "ORIGINAL_AUXILIARY_AO_HERMITIAN_PRINCIPAL_"
            "PSEUDOINVERSE_SQUARE_ROOT",
            RowConvention::
                OriginalAuxiliaryAoHermitianPrincipalPseudoinverseSquareRoot)
        .value(
            "COMPACT_METRIC_EIGENMODES_UNSUPPORTED",
            RowConvention::CompactMetricEigenmodesUnsupported);
    py::enum_<TransferConvention>(
        m, "_PeriodicCorrelationTransferRepresentativeConvention")
        .value(
            "CENTERED_HALF_OPEN_NEGATIVE_NYQUIST",
            TransferConvention::CenteredHalfOpenNegativeNyquist)
        .value(
            "UNSPECIFIED_UNSUPPORTED",
            TransferConvention::UnspecifiedUnsupported);
    py::enum_<Backing>(m, "_PeriodicCorrelationFactorBackingMode")
        .value("DISK", Backing::Disk)
        .value("MEMORY", Backing::Memory);
    py::enum_<Publisher>(m, "_PeriodicCorrelationFactorPublisherMode")
        .value(
            "RANDOM_ACCESS_EXACTLY_ONCE",
            Publisher::RandomAccessExactlyOnce)
        .value(
            "SEQUENTIAL_APPEND_UNSUPPORTED",
            Publisher::SequentialAppendUnsupported)
        .value("CANONICAL_SEQUENTIAL_EXACTLY_ONCE", Publisher::CanonicalSequentialExactlyOnce);
    py::enum_<Phase>(m, "_PeriodicCorrelationFactorBuildPhase")
        .value("RECIPROCAL_METRIC", Phase::ReciprocalMetric)
        .value("SHORT_RANGE_METRIC", Phase::ShortRangeMetric)
        .value("METRIC_FACTORIZATION", Phase::MetricFactorization)
        .value("METRIC_WHITENER", Phase::MetricWhitener)
        .value("RECIPROCAL_THREE_CENTER", Phase::ReciprocalThreeCenter)
        .value("SHORT_RANGE_THREE_CENTER", Phase::ShortRangeThreeCenter)
        .value("WHITENING", Phase::Whitening)
        .value("TRANSPOSE", Phase::Transpose)
        .value("PUBLISH", Phase::Publish);
    py::enum_<Admission>(m, "_PeriodicCorrelationFactorBuildAdmissionCode")
        .value("ADMITTED", Admission::Admitted)
        .value("MISSING_MEMORY_LIMIT", Admission::MissingMemoryLimit)
        .value("MISSING_SCRATCH_LIMIT", Admission::MissingScratchLimit)
        .value("MEMORY_EXCEEDED", Admission::MemoryExceeded)
        .value("SCRATCH_EXCEEDED", Admission::ScratchExceeded)
        .value("ARITHMETIC_OVERFLOW", Admission::ArithmeticOverflow);

    py::class_<QRecord>(m, "_PeriodicCorrelationFactorBuildQRecord")
        .def(py::init<>())
        .def_readwrite("q_index", &QRecord::q_index)
        .def_readwrite(
            "centered_doubled_numerator",
            &QRecord::centered_doubled_numerator)
        .def_readwrite(
            "centered_reciprocal_wrap", &QRecord::centered_reciprocal_wrap)
        .def_readwrite(
            "base_reciprocal_vector_count",
            &QRecord::base_reciprocal_vector_count)
        .def_readwrite(
            "tail_reciprocal_vector_count",
            &QRecord::tail_reciprocal_vector_count)
        .def_readwrite(
            "zero_mode_excluded_count",
            &QRecord::zero_mode_excluded_count)
        .def_readwrite(
            "short_range_metric_cell_count",
            &QRecord::short_range_metric_cell_count)
        .def_readwrite(
            "short_range_metric_task_count",
            &QRecord::short_range_metric_task_count)
        .def_readwrite(
            "short_range_three_center_cell_pair_count",
            &QRecord::short_range_three_center_cell_pair_count)
        .def_readwrite(
            "short_range_three_center_task_count",
            &QRecord::short_range_three_center_task_count)
        .def_readwrite(
            "source_manifest_bytes", &QRecord::source_manifest_bytes);

    py::class_<Backend>(
        m, "_PeriodicCorrelationFactorBuildBackendInventory")
        .def(py::init<>())
        .def_readwrite("complete", &Backend::complete)
        .def_readwrite(
            "per_thread_short_range_fixed_workspace_bytes",
            &Backend::per_thread_short_range_fixed_workspace_bytes)
        .def_readwrite(
            "per_thread_fourier_transform_fixed_workspace_bytes",
            &Backend::per_thread_fourier_transform_fixed_workspace_bytes)
        .def_readwrite(
            "eigensolver_workspace_bytes",
            &Backend::eigensolver_workspace_bytes)
        .def_readwrite(
            "whitener_workspace_bytes", &Backend::whitener_workspace_bytes)
        .def_readwrite("gemm_workspace_bytes", &Backend::gemm_workspace_bytes)
        .def_readwrite(
            "publisher_workspace_bytes", &Backend::publisher_workspace_bytes)
        .def_readwrite(
            "short_range_staging_memory_bytes",
            &Backend::short_range_staging_memory_bytes)
        .def_readwrite(
            "folded_source_memory_bytes", &Backend::folded_source_memory_bytes)
        .def_readwrite(
            "short_range_staging_scratch_bytes",
            &Backend::short_range_staging_scratch_bytes)
        .def_readwrite(
            "folded_source_scratch_bytes",
            &Backend::folded_source_scratch_bytes)
        .def_readwrite(
            "exact_extra_retained_bytes",
            &Backend::exact_extra_retained_bytes)
        .def_readwrite(
            "exact_extra_control_bytes", &Backend::exact_extra_control_bytes)
        .def_readwrite(
            "exact_extra_scratch_bytes", &Backend::exact_extra_scratch_bytes);

    py::class_<Codec>(m, "_PeriodicCorrelationFactorBuildCodecInventory")
        .def(py::init<>())
        .def_readwrite("complete", &Codec::complete)
        .def_readwrite("fixed_header_bytes", &Codec::fixed_header_bytes)
        .def_readwrite("fixed_footer_bytes", &Codec::fixed_footer_bytes)
        .def_readwrite("fixed_manifest_bytes", &Codec::fixed_manifest_bytes)
        .def_readwrite(
            "integrity_bytes_per_tile", &Codec::integrity_bytes_per_tile)
        .def_readwrite("rank_bytes_per_q", &Codec::rank_bytes_per_q)
        .def_readwrite(
            "digest_bytes_per_tile", &Codec::digest_bytes_per_tile)
        .def_readwrite("journal_header_bytes", &Codec::journal_header_bytes)
        .def_readwrite(
            "journal_record_bytes_per_tile",
            &Codec::journal_record_bytes_per_tile)
        .def_readwrite(
            "checkpoint_record_bytes", &Codec::checkpoint_record_bytes)
        .def_readwrite(
            "checkpoint_records_per_generation",
            &Codec::checkpoint_records_per_generation)
        .def_readwrite(
            "existing_generation_count", &Codec::existing_generation_count);

    py::class_<Config>(m, "_PeriodicCorrelationFactorBuildConfig")
        .def(py::init<>())
        .def_readwrite("producer_mode", &Config::producer_mode)
        .def_readwrite("short_range_policy", &Config::short_range_policy)
        .def_readwrite("row_convention", &Config::row_convention)
        .def_readwrite("transfer_convention", &Config::transfer_convention)
        .def_readwrite("backing_mode", &Config::backing_mode)
        .def_readwrite("publisher_mode", &Config::publisher_mode)
        .def_readwrite(
            "ao_basis_identity_sha256", &Config::ao_basis_identity_sha256)
        .def_readwrite(
            "auxiliary_basis_identity_sha256",
            &Config::auxiliary_basis_identity_sha256)
        .def_readwrite(
            "producer_identity_sha256", &Config::producer_identity_sha256)
        .def_readwrite(
            "backend_identity_sha256", &Config::backend_identity_sha256)
        .def_readwrite(
            "codec_identity_sha256", &Config::codec_identity_sha256)
        .def_readwrite(
            "metric_absolute_eigenvalue_threshold",
            &Config::metric_absolute_eigenvalue_threshold)
        .def_readwrite(
            "integral_absolute_screening_threshold",
            &Config::integral_absolute_screening_threshold)
        .def_readwrite(
            "reciprocal_energy_cutoff", &Config::reciprocal_energy_cutoff)
        .def_readwrite(
            "short_range_real_space_cutoff",
            &Config::short_range_real_space_cutoff)
        .def_readwrite(
            "range_separation_omega", &Config::range_separation_omega)
        .def_readwrite("ao_pair_block", &Config::ao_pair_block)
        .def_readwrite("auxiliary_block", &Config::auxiliary_block)
        .def_readwrite("reciprocal_block", &Config::reciprocal_block)
        .def_readwrite(
            "whitener_column_block", &Config::whitener_column_block)
        .def_readwrite("q_concurrency", &Config::q_concurrency)
        .def_readwrite("native_threads", &Config::native_threads)
        .def_readwrite(
            "ao_pair_fourier_staging_required",
            &Config::ao_pair_fourier_staging_required)
        .def_readwrite(
            "transpose_before_publish", &Config::transpose_before_publish)
        .def_readwrite(
            "publisher_buffer_count", &Config::publisher_buffer_count)
        .def_readwrite("backend", &Config::backend)
        .def_readwrite("codec", &Config::codec);

    py::class_<Shape>(m, "_PeriodicCorrelationFactorBuildShape")
        .def_readonly("n_kpoints", &Shape::n_kpoints)
        .def_readonly("n_basis", &Shape::n_basis)
        .def_readonly("n_auxiliary", &Shape::n_auxiliary)
        .def_readonly("n_ao_pairs", &Shape::n_ao_pairs)
        .def_readonly("ao_pair_block", &Shape::ao_pair_block)
        .def_readonly("auxiliary_block", &Shape::auxiliary_block)
        .def_readonly("reciprocal_block", &Shape::reciprocal_block)
        .def_readonly(
            "whitener_column_block", &Shape::whitener_column_block)
        .def_readonly("tile_count", &Shape::tile_count)
        .def_readonly(
            "logical_element_count", &Shape::logical_element_count)
        .def_readonly("logical_bytes", &Shape::logical_bytes)
        .def_readonly("maximum_tile_bytes", &Shape::maximum_tile_bytes);

    py::class_<Census>(m, "_PeriodicCorrelationFactorBuildCensus")
        .def_property_readonly("contract_version", &Census::contract_version)
        .def_property_readonly(
            "state",
            [](const Census& census) -> std::shared_ptr<State> {
                const auto& state = census.state_handle();
                if (!state) {
                    throw std::logic_error(
                        "moved-from periodic factor-build census has no "
                        "state");
                }
                return std::const_pointer_cast<State>(state);
            })
        .def_property_readonly(
            "state_identity_sha256", &Census::state_identity_sha256)
        .def_property_readonly(
            "calculation_identity", &Census::calculation_identity)
        .def_property_readonly(
            "allocation_identity", &Census::allocation_identity)
        .def_property_readonly(
            "schedule_identity_sha256", &Census::schedule_identity_sha256)
        .def_property_readonly(
            "census_identity_sha256", &Census::census_identity_sha256)
        .def_property_readonly("mesh", &Census::mesh)
        .def_property_readonly("is_shift", &Census::is_shift)
        .def_property_readonly(
            "shape", [](const Census& census) { return census.shape(); })
        .def_property_readonly(
            "config", [](const Census& census) { return census.config(); })
        .def_property_readonly(
            "q_records",
            [](const Census& census) { return census.q_records(); });

    py::class_<Components>(
        m, "_PeriodicCorrelationFactorBuildComponents")
        .def_readonly(
            "admitted_baseline_retained_bytes",
            &Components::admitted_baseline_retained_bytes)
        .def_readonly(
            "in_memory_output_bytes", &Components::in_memory_output_bytes)
        .def_readonly(
            "source_manifest_total_bytes",
            &Components::source_manifest_total_bytes)
        .def_readonly(
            "source_manifest_maximum_bytes",
            &Components::source_manifest_maximum_bytes)
        .def_readonly(
            "publisher_bitmap_bytes", &Components::publisher_bitmap_bytes)
        .def_readonly(
            "publisher_digest_table_bytes",
            &Components::publisher_digest_table_bytes)
        .def_readonly(
            "publisher_control_bytes", &Components::publisher_control_bytes)
        .def_readonly(
            "folded_source_retained_bytes",
            &Components::folded_source_retained_bytes)
        .def_readonly(
            "reciprocal_g_panel_bytes", &Components::reciprocal_g_panel_bytes)
        .def_readonly(
            "auxiliary_fourier_double_panel_bytes",
            &Components::auxiliary_fourier_double_panel_bytes)
        .def_readonly(
            "ao_pair_fourier_panel_bytes",
            &Components::ao_pair_fourier_panel_bytes)
        .def_readonly(
            "ao_pair_fourier_staging_bytes",
            &Components::ao_pair_fourier_staging_bytes)
        .def_readonly(
            "metric_matrix_bytes", &Components::metric_matrix_bytes)
        .def_readonly(
            "metric_eigenvector_bytes",
            &Components::metric_eigenvector_bytes)
        .def_readonly(
            "metric_eigenvalue_bytes",
            &Components::metric_eigenvalue_bytes)
        .def_readonly(
            "whitener_matrix_bytes", &Components::whitener_matrix_bytes)
        .def_readonly(
            "whitener_column_panel_bytes",
            &Components::whitener_column_panel_bytes)
        .def_readonly(
            "raw_three_center_panel_bytes",
            &Components::raw_three_center_panel_bytes)
        .def_readonly(
            "whitening_input_panel_bytes",
            &Components::whitening_input_panel_bytes)
        .def_readonly(
            "whitening_output_panel_bytes",
            &Components::whitening_output_panel_bytes)
        .def_readonly(
            "transpose_staging_bytes",
            &Components::transpose_staging_bytes)
        .def_readonly(
            "publisher_buffer_bytes", &Components::publisher_buffer_bytes)
        .def_readonly(
            "threaded_short_range_workspace_bytes",
            &Components::threaded_short_range_workspace_bytes)
        .def_readonly(
            "threaded_fourier_workspace_bytes",
            &Components::threaded_fourier_workspace_bytes)
        .def_readonly(
            "encoded_generation_bytes", &Components::encoded_generation_bytes)
        .def_readonly("journal_bytes", &Components::journal_bytes)
        .def_readonly("checkpoint_bytes", &Components::checkpoint_bytes)
        .def_readonly(
            "disk_generation_bytes", &Components::disk_generation_bytes);

    py::class_<PhaseEstimate>(
        m, "_PeriodicCorrelationFactorBuildPhaseEstimate")
        .def_readonly("phase", &PhaseEstimate::phase)
        .def_readonly("retained_bytes", &PhaseEstimate::retained_bytes)
        .def_readonly("phase_extra_bytes", &PhaseEstimate::phase_extra_bytes)
        .def_readonly(
            "peak_memory_bytes", &PhaseEstimate::peak_memory_bytes);

    py::class_<Plan>(m, "_PeriodicCorrelationFactorBuildPlan")
        .def_readonly("contract_version", &Plan::contract_version)
        .def_readonly("admission", &Plan::admission)
        .def_readonly("calculation_identity", &Plan::calculation_identity)
        .def_readonly("allocation_identity", &Plan::allocation_identity)
        .def_readonly(
            "schedule_identity_sha256", &Plan::schedule_identity_sha256)
        .def_readonly("census_identity_sha256", &Plan::census_identity_sha256)
        .def_readonly("plan_identity_sha256", &Plan::plan_identity_sha256)
        .def_readonly("shape", &Plan::shape)
        .def_readonly(
            "total_base_reciprocal_vectors",
            &Plan::total_base_reciprocal_vectors)
        .def_readonly(
            "total_tail_reciprocal_vectors",
            &Plan::total_tail_reciprocal_vectors)
        .def_readonly(
            "maximum_reciprocal_vectors_per_q",
            &Plan::maximum_reciprocal_vectors_per_q)
        .def_readonly(
            "total_short_range_metric_tasks",
            &Plan::total_short_range_metric_tasks)
        .def_readonly(
            "total_short_range_three_center_tasks",
            &Plan::total_short_range_three_center_tasks)
        .def_readonly(
            "maximum_short_range_metric_tasks_per_q",
            &Plan::maximum_short_range_metric_tasks_per_q)
        .def_readonly(
            "maximum_short_range_three_center_tasks_per_q",
            &Plan::maximum_short_range_three_center_tasks_per_q)
        .def_readonly("logical_factor_bytes", &Plan::logical_factor_bytes)
        .def_readonly(
            "encoded_generation_bytes", &Plan::encoded_generation_bytes)
        .def_readonly(
            "modeled_peak_memory_bytes", &Plan::modeled_peak_memory_bytes)
        .def_readonly("required_memory_bytes", &Plan::required_memory_bytes)
        .def_readonly(
            "modeled_scratch_bytes", &Plan::modeled_scratch_bytes)
        .def_readonly("required_scratch_bytes", &Plan::required_scratch_bytes)
        .def_readonly("components", &Plan::components)
        .def_readonly("phases", &Plan::phases)
        .def_readonly("failure_detail", &Plan::failure_detail);

    m.def(
        "_make_periodic_correlation_factor_build_census",
        [](const vibeqc::PeriodicCorrelationAdmittedReference& reference,
           const vibeqc::PeriodicCorrelationFactorStreamSchedule& schedule,
           Config config,
           std::vector<QRecord> q_records) {
            return vibeqc::make_periodic_correlation_factor_build_census(
                reference,
                schedule,
                std::move(config),
                std::move(q_records));
        },
        py::arg("reference"), py::arg("schedule"), py::arg("config"),
        py::arg("q_records"));
    m.def(
        "_plan_periodic_correlation_factor_build",
        &vibeqc::plan_periodic_correlation_factor_build,
        py::arg("reference"), py::arg("schedule"), py::arg("census"));
}
