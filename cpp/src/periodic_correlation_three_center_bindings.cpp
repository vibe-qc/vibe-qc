// Included by bindings.cpp. Tiny diagnostics, not a user-facing method route.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/periodic_correlation_three_center.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_three_center(py::module_& m) {
    using Tile = vibeqc::PeriodicCorrelationThreeCenterTile;
    m.attr("_PERIODIC_CORRELATION_THREE_CENTER_IMAGE_POLICY") =
        py::str(vibeqc::kPeriodicCorrelationThreeCenterImagePolicy);
    m.attr("_PERIODIC_CORRELATION_THREE_CENTER_LATTICE_POLICY") =
        py::str(vibeqc::kPeriodicCorrelationThreeCenterLatticePolicy);
    m.attr("_PERIODIC_CORRELATION_THREE_CENTER_SOURCE_AUDIT_CONTROL_BYTES") =
        py::int_(sizeof(vibeqc::PeriodicCorrelationReciprocalMetricSourceManifest));
    py::class_<Tile>(m, "_PeriodicCorrelationThreeCenterTile")
        .def_property_readonly("contract_version", &Tile::contract_version)
        .def_property_readonly("finite_image_reference", &Tile::finite_image_reference)
        .def_property_readonly("ao_image_source_certified", &Tile::ao_image_source_certified)
        .def_property_readonly("descriptor", &Tile::descriptor)
        .def_property_readonly("image_cutoff_bohr", &Tile::image_cutoff_bohr)
        .def_property_readonly("image_candidate_count", &Tile::image_candidate_count)
        .def_property_readonly("retained_pair_image_count", &Tile::retained_pair_image_count)
        .def_property_readonly("reciprocal_vector_count", &Tile::reciprocal_vector_count)
        .def_property_readonly("output_bytes", &Tile::output_bytes)
        .def_property_readonly("numerical_peak_bytes", &Tile::numerical_peak_bytes)
        .def_property_readonly("admitted_assembly_peak_bytes", &Tile::admitted_assembly_peak_bytes)
        .def_property_readonly("admitted_whitening_peak_bytes", &Tile::admitted_whitening_peak_bytes)
        .def_property_readonly("source_identity_sha256", &Tile::source_identity_sha256)
        .def_property_readonly("whitener_payload_identity_sha256", &Tile::whitener_payload_identity_sha256)
        .def_property_readonly("ao_basis_identity_sha256", &Tile::ao_basis_identity_sha256)
        .def_property_readonly("auxiliary_basis_identity_sha256", &Tile::auxiliary_basis_identity_sha256)
        .def_property_readonly("census_identity_sha256", &Tile::census_identity_sha256)
        .def_property_readonly("plan_identity_sha256", &Tile::plan_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Tile::payload_identity_sha256)
        .def_property_readonly("matrix", [](const Tile& tile) {
            const auto& d = tile.descriptor();
            if (d.auxiliary_count > 16 || d.ao_pair_count > 64
                || d.auxiliary_count == 0 || d.ao_pair_count == 0
                || tile.matrix_row_major().size() != d.auxiliary_count * d.ao_pair_count) {
                throw std::length_error("three-center tile copy exceeds diagnostic shape or has been consumed");
            }
            py::array_t<std::complex<double>> result({
                static_cast<py::ssize_t>(d.auxiliary_count),
                static_cast<py::ssize_t>(d.ao_pair_count)});
            std::memcpy(result.mutable_data(), tile.matrix_row_major().data(), tile.output_bytes());
            return result;
        });
    m.def("_build_periodic_correlation_three_center_tile", [](
        const vibeqc::PeriodicCorrelationAdmittedReference& reference,
        const vibeqc::PeriodicCorrelationFactorStreamSchedule& schedule,
        const vibeqc::PeriodicCorrelationFactorBuildCensus& census,
        const vibeqc::PeriodicCorrelationReciprocalMetricSourceManifest& source,
        const vibeqc::PeriodicCorrelationMetricFactorizationResult& whitener,
        const vibeqc::BasisSet& ao, const vibeqc::BasisSet& auxiliary,
        std::uint64_t sequence, double cutoff, std::uint64_t candidate_cap) {
        const auto d = schedule.descriptor(sequence);
        if (ao.nbasis() > 16 || auxiliary.nbasis() > 16
            || d.ao_pair_count > 64 || source.accepted_vector_count() > 512
            || source.candidate_count() > 65536
            || census.config().reciprocal_block > 64 || candidate_cap > 65536) {
            throw std::length_error("three-center tile diagnostic exceeds tiny shape or source cap");
        }
        std::uint64_t maximum_primitives = 0;
        for (const auto& shell : ao.libint()) {
            maximum_primitives = std::max(maximum_primitives,
                static_cast<std::uint64_t>(shell.alpha.size()));
        }
        for (const auto& shell : auxiliary.libint()) {
            maximum_primitives = std::max(maximum_primitives,
                static_cast<std::uint64_t>(shell.alpha.size()));
        }
        if (maximum_primitives > 16 || candidate_cap * source.accepted_vector_count()
            * maximum_primitives * maximum_primitives > 2000000) {
            throw std::length_error("three-center tile diagnostic exceeds AO-image Fourier work cap");
        }
        py::gil_scoped_release release;
        return vibeqc::build_periodic_correlation_three_center_tile(
            reference, schedule, census, source, whitener, ao, auxiliary,
            sequence, cutoff, candidate_cap);
    }, py::arg("reference"), py::arg("schedule"), py::arg("census"),
       py::arg("source"), py::arg("whitener"), py::arg("ao_basis"),
       py::arg("auxiliary_basis"), py::arg("sequence_index"),
       py::arg("image_cutoff_bohr"), py::arg("maximum_image_candidates"));

    using Caps = vibeqc::PeriodicCorrelationThreeCenterStreamCaps;
    using Event = vibeqc::PeriodicCorrelationThreeCenterStreamProgress;
    using Receipt = vibeqc::PeriodicCorrelationThreeCenterStreamReceipt;
    using Stage = vibeqc::PeriodicCorrelationThreeCenterStreamStage;
    py::class_<Caps>(m, "_PeriodicCorrelationThreeCenterStreamCaps")
        .def(py::init<>())
        .def_readwrite("maximum_tile_count", &Caps::maximum_tile_count)
        .def_readwrite("maximum_logical_bytes", &Caps::maximum_logical_bytes)
        .def_readwrite("maximum_tile_bytes", &Caps::maximum_tile_bytes)
        .def_readwrite("maximum_reciprocal_candidates_per_q", &Caps::maximum_reciprocal_candidates_per_q)
        .def_readwrite("maximum_image_candidates_per_tile", &Caps::maximum_image_candidates_per_tile)
        .def_readwrite("receiver_retained_numeric_bytes", &Caps::receiver_retained_numeric_bytes);
    py::enum_<Stage>(m, "_PeriodicCorrelationThreeCenterStreamStage")
        .value("SOURCE", Stage::Source).value("METRIC", Stage::Metric)
        .value("FACTORIZATION", Stage::Factorization).value("TILES", Stage::Tiles)
        .value("Q_COMPLETE", Stage::QComplete).value("COMPLETE", Stage::Complete);
    py::class_<Event>(m, "_PeriodicCorrelationThreeCenterStreamProgress")
        .def_readonly("stage", &Event::stage)
        .def_readonly("q_index", &Event::q_index)
        .def_readonly("completed_q_count", &Event::completed_q_count)
        .def_readonly("completed_tile_count", &Event::completed_tile_count)
        .def_readonly("completed_logical_bytes", &Event::completed_logical_bytes)
        .def_readonly("total_q_count", &Event::total_q_count)
        .def_readonly("total_tile_count", &Event::total_tile_count)
        .def_readonly("total_logical_bytes", &Event::total_logical_bytes);
    py::class_<Receipt>(m, "_PeriodicCorrelationThreeCenterStreamReceipt")
        .def_readonly("contract_version", &Receipt::contract_version)
        .def_readonly("finite_image_reference", &Receipt::finite_image_reference)
        .def_readonly("ao_image_source_certified", &Receipt::ao_image_source_certified)
        .def_readonly("completed_q_count", &Receipt::completed_q_count)
        .def_readonly("completed_tile_count", &Receipt::completed_tile_count)
        .def_readonly("completed_element_count", &Receipt::completed_element_count)
        .def_readonly("completed_logical_bytes", &Receipt::completed_logical_bytes)
        .def_readonly("maximum_tile_bytes", &Receipt::maximum_tile_bytes)
        .def_readonly("admitted_peak_memory_bytes", &Receipt::admitted_peak_memory_bytes)
        .def_readonly("schedule_identity_sha256", &Receipt::schedule_identity_sha256)
        .def_readonly("census_identity_sha256", &Receipt::census_identity_sha256)
        .def_readonly("plan_identity_sha256", &Receipt::plan_identity_sha256)
        .def_readonly("payload_identity_sha256", &Receipt::payload_identity_sha256);
    // Allocation-free native receiver: the diagnostic exercises the actual
    // complete producer but never returns/collects a full factor tensor.
    m.def("_periodic_correlation_three_center_stream_diagnostic", [](
        const vibeqc::PeriodicCorrelationAdmittedReference& reference,
        const vibeqc::PeriodicCorrelationFactorStreamSchedule& schedule,
        const vibeqc::PeriodicCorrelationFactorBuildCensus& census,
        const vibeqc::BasisSet& ao, const vibeqc::BasisSet& auxiliary,
        double cutoff, double negative, const Caps& caps,
        std::uint64_t fail_before_sequence, py::object progress) {
        const auto& shape = schedule.shape();
        if (shape.n_basis > 8 || shape.n_auxiliary > 8 || shape.n_kpoints > 8
            || shape.tile_count > 256 || shape.logical_bytes > 65536
            || caps.maximum_reciprocal_candidates_per_q > 65536
            || caps.maximum_image_candidates_per_tile > 65536
            || census.config().reciprocal_block > 64) {
            throw std::length_error("physical factor stream diagnostic exceeds tiny work limits");
        }
        std::uint64_t primitives = 0, maximum_vectors = 0;
        for (const auto& shell : ao.libint()) primitives = std::max(primitives,
            static_cast<std::uint64_t>(shell.alpha.size()));
        for (const auto& shell : auxiliary.libint()) primitives = std::max(primitives,
            static_cast<std::uint64_t>(shell.alpha.size()));
        for (const auto& q : census.q_records()) maximum_vectors = std::max(
            maximum_vectors, q.base_reciprocal_vector_count);
        if (primitives > 8 || maximum_vectors > 512
            || caps.maximum_image_candidates_per_tile * maximum_vectors
                * primitives * primitives * shape.tile_count > 100000000) {
            throw std::length_error("physical factor stream diagnostic exceeds total Fourier work cap");
        }
        if (!progress.is_none() && !PyCallable_Check(progress.ptr())) {
            throw std::invalid_argument("physical factor stream progress must be callable or None");
        }
        struct Sink {
            std::uint64_t fail_before_sequence;
            std::uint64_t delivered = 0;
            static void receive(const Tile& tile, void* pointer) {
                auto& self = *static_cast<Sink*>(pointer);
                if (tile.descriptor().sequence_index != self.delivered) {
                    throw std::logic_error("diagnostic sink received an out-of-sequence tile");
                }
                if (self.delivered == self.fail_before_sequence) {
                    throw std::runtime_error("diagnostic sink cancelled before tile acceptance");
                }
                ++self.delivered;
            }
            static void report(const Event& event, void* pointer) {
                py::gil_scoped_acquire acquire;
                (*static_cast<py::object*>(pointer))(py::cast(event, py::return_value_policy::copy));
            }
        } sink{fail_before_sequence};
        py::gil_scoped_release release;
        auto result = vibeqc::stream_periodic_correlation_three_center_tiles(
            reference, schedule, census, ao, auxiliary, cutoff, negative, caps,
            &Sink::receive, &sink, progress.is_none() ? nullptr : &Sink::report, &progress);
        if (result.completed_tile_count != sink.delivered) {
            throw std::logic_error("diagnostic sink and producer coverage disagree");
        }
        return result;
    }, py::arg("reference"), py::arg("schedule"), py::arg("census"),
       py::arg("ao_basis"), py::arg("auxiliary_basis"), py::arg("image_cutoff_bohr"),
       py::arg("negative_tolerance"), py::arg("caps"),
       py::arg("fail_before_sequence") = std::numeric_limits<std::uint64_t>::max(),
       py::arg("progress") = py::none());
}
