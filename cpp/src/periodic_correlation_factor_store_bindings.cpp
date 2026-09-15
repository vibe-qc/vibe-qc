// Included by bindings.cpp; private scratch diagnostics, no user artifact API.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/periodic_correlation_factor_store.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_factor_store(py::module_& m) {
    using Caps = vibeqc::PeriodicCorrelationPrivateFactorStoreCaps;
    using Writer = vibeqc::PeriodicCorrelationPrivateFactorWriter;
    using Reader = vibeqc::PeriodicCorrelationPrivateFactorReader;
    using State = vibeqc::PeriodicCorrelationPrivateFactorStoreState;
    m.def("_periodic_correlation_private_factor_store_codec_identity_sha256",
          &vibeqc::periodic_correlation_private_factor_store_codec_identity_sha256);
    m.def("_periodic_correlation_private_factor_store_codec_inventory",
          &vibeqc::periodic_correlation_private_factor_store_codec_inventory);
    m.def("_periodic_correlation_private_factor_store_fixed_control_bytes",
          &vibeqc::periodic_correlation_private_factor_store_fixed_control_bytes);
    m.def("_periodic_correlation_private_factor_store_receiver_bytes",
          &vibeqc::periodic_correlation_private_factor_store_receiver_bytes);
    m.def("_periodic_correlation_private_factor_store_file_bytes",
          &vibeqc::periodic_correlation_private_factor_store_file_bytes);
    m.def("_periodic_correlation_private_factor_store_record_offset",
          &vibeqc::periodic_correlation_private_factor_store_record_offset);
    m.def("_private_factor_store_live_owned_descriptors_diagnostic",
          &vibeqc::private_factor_store_live_owned_descriptors_diagnostic);
    m.def("_private_factor_store_corrupt_diagnostic", &vibeqc::private_factor_store_corrupt_diagnostic);
    m.def("_private_factor_store_truncate_diagnostic", &vibeqc::private_factor_store_truncate_diagnostic);
    m.def("_private_factor_store_bytes_diagnostic", [](
        const Reader& reader, std::uint64_t offset, std::uint64_t count) {
        const auto bytes = vibeqc::private_factor_store_bytes_diagnostic(reader, offset, count);
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    });
    m.def("_private_factor_reader_callback_failure_diagnostic", [](
        const Reader& reader, std::uint64_t sequence, bool recursive) {
        if (reader.file_bytes() > 1048576 || reader.tile_count() > 128)
            throw std::length_error("private factor callback diagnostic exceeds tiny store cap");
        struct Context {
            const Reader& reader;
            std::uint64_t sequence;
            bool recursive;
            static void receive(const vibeqc::PeriodicCorrelationPrivateFactorTileView&, void* pointer) {
                auto& self = *static_cast<Context*>(pointer);
                if (self.recursive) self.reader.visit_tile(self.sequence, 8192, &Context::receive, &self);
                throw std::runtime_error("private factor reader diagnostic callback failed");
            }
        } context{reader, sequence, recursive};
        py::gil_scoped_release release;
        reader.visit_tile(sequence, 8192, &Context::receive, &context);
    });
    py::class_<Caps>(m, "_PeriodicCorrelationPrivateFactorStoreCaps")
        .def(py::init<>())
        .def_readwrite("maximum_file_bytes", &Caps::maximum_file_bytes)
        .def_readwrite("maximum_tile_bytes", &Caps::maximum_tile_bytes)
        .def_readwrite("maximum_tile_count", &Caps::maximum_tile_count);
    py::enum_<State>(m, "_PeriodicCorrelationPrivateFactorStoreState")
        .value("OPEN", State::Open).value("FAILED", State::Failed)
        .value("ABORTED", State::Aborted).value("FINALIZED", State::Finalized);
    py::class_<Reader>(m, "_PeriodicCorrelationPrivateFactorReader")
        .def_property_readonly("file_bytes", &Reader::file_bytes)
        .def_property_readonly("tile_count", &Reader::tile_count)
        .def_property_readonly("storage_identity_sha256", &Reader::storage_identity_sha256)
        .def_property_readonly("producer_completion_identity_sha256", &Reader::producer_completion_identity_sha256)
        .def_property_readonly("state_identity_sha256", &Reader::state_identity_sha256)
        .def_property_readonly("calculation_identity", &Reader::calculation_identity)
        .def_property_readonly("allocation_identity", &Reader::allocation_identity)
        .def_property_readonly("schedule_identity_sha256", &Reader::schedule_identity_sha256)
        .def_property_readonly("census_identity_sha256", &Reader::census_identity_sha256)
        .def_property_readonly("plan_identity_sha256", &Reader::plan_identity_sha256)
        .def_property_readonly("ao_basis_identity_sha256", &Reader::ao_basis_identity_sha256)
        .def_property_readonly("auxiliary_basis_identity_sha256", &Reader::auxiliary_basis_identity_sha256)
        .def_property_readonly("image_cutoff_bohr", &Reader::image_cutoff_bohr)
        .def_property_readonly("image_policy", &Reader::image_policy)
        .def_property_readonly("finite_image_reference", &Reader::finite_image_reference)
        .def_property_readonly("ao_image_source_certified", &Reader::ao_image_source_certified)
        .def_property_readonly("restart_supported", [](const Reader&) { return false; })
        .def("descriptor", &Reader::descriptor)
        .def("read_tile", [](const Reader& reader, std::uint64_t sequence, std::uint64_t cap) {
            const auto d = reader.descriptor(sequence);
            if (reader.file_bytes() > 1048576 || reader.tile_count() > 128
                || d.auxiliary_count > 8 || d.ao_pair_count > 64)
                throw std::length_error("private factor reader diagnostic exceeds tiny shape cap");
            if (cap == 0 || d.element_count > cap / 16)
                throw std::invalid_argument("private factor reader diagnostic has insufficient explicit tile cap");
            py::array_t<std::complex<double>> result({static_cast<py::ssize_t>(d.auxiliary_count),
                                                     static_cast<py::ssize_t>(d.ao_pair_count)});
            struct Output {
                std::complex<double>* pointer;
                std::uint64_t count;
                static void receive(const vibeqc::PeriodicCorrelationPrivateFactorTileView& tile, void* context) {
                    const auto& out = *static_cast<Output*>(context);
                    if (tile.element_count != out.count) throw std::logic_error("private store diagnostic shape changed");
                    std::memcpy(out.pointer, tile.data, tile.payload_bytes);
                }
            } output{result.mutable_data(), d.element_count};
            {
                py::gil_scoped_release release;
                reader.visit_tile(sequence, cap, &Output::receive, &output);
            }
            return result;
        }, py::arg("sequence"), py::arg("maximum_tile_bytes"));
    py::class_<Writer>(m, "_PeriodicCorrelationPrivateFactorWriter")
        .def_property_readonly("state", &Writer::state)
        .def_property_readonly("accepted_tile_count", &Writer::accepted_tile_count)
        .def_property_readonly("file_bytes", &Writer::file_bytes)
        .def("accept", &Writer::accept, py::call_guard<py::gil_scoped_release>())
        .def("finish", &Writer::finish, py::call_guard<py::gil_scoped_release>())
        .def("abort", &Writer::abort);
    m.def("_make_periodic_correlation_private_factor_writer", [](
        const vibeqc::PeriodicCorrelationAdmittedReference& reference,
        const vibeqc::PeriodicCorrelationFactorStreamSchedule& schedule,
        const vibeqc::PeriodicCorrelationFactorBuildCensus& census,
        double cutoff, double negative, const std::string& directory, const Caps& caps) {
        const auto& shape = schedule.shape();
        if (shape.n_basis > 8 || shape.n_auxiliary > 8 || shape.n_kpoints > 8
            || shape.tile_count > 128 || shape.maximum_tile_bytes > 8192
            || caps.maximum_file_bytes > 1048576 || caps.maximum_tile_bytes > 8192
            || caps.maximum_tile_count > 128)
            throw std::length_error("private factor writer diagnostic exceeds tiny store cap");
        py::gil_scoped_release release;
        return vibeqc::make_periodic_correlation_private_factor_writer(
            reference, schedule, census, cutoff, negative, directory, caps);
    }, py::arg("reference"), py::arg("schedule"), py::arg("census"),
       py::arg("image_cutoff_bohr"), py::arg("negative_tolerance"),
       py::arg("private_scratch_directory"), py::arg("caps"));
    m.def("_periodic_correlation_private_factor_store_stream_diagnostic", [](
        Writer& writer, const vibeqc::PeriodicCorrelationAdmittedReference& reference,
        const vibeqc::PeriodicCorrelationFactorStreamSchedule& schedule,
        const vibeqc::PeriodicCorrelationFactorBuildCensus& census,
        const vibeqc::BasisSet& ao, const vibeqc::BasisSet& auxiliary,
        double cutoff, double negative, const vibeqc::PeriodicCorrelationThreeCenterStreamCaps& caps,
        std::uint64_t fail_before_sequence) {
        try {
            const auto& shape = schedule.shape();
            if (shape.n_basis > 8 || shape.n_auxiliary > 8 || shape.n_kpoints > 8
                || shape.tile_count > 128 || shape.logical_bytes > 65536
                || caps.maximum_reciprocal_candidates_per_q > 65536
                || caps.maximum_image_candidates_per_tile > 65536 || census.config().reciprocal_block > 64)
                throw std::length_error("private factor stream diagnostic exceeds tiny work limits");
            if (caps.receiver_retained_numeric_bytes
                < vibeqc::periodic_correlation_private_factor_store_receiver_bytes(schedule))
                throw std::invalid_argument("private factor stream requires its receiver workspace declaration");
            std::uint64_t primitives = 0, vectors = 0;
            for (const auto& shell : ao.libint()) primitives = std::max(primitives,
                static_cast<std::uint64_t>(shell.alpha.size()));
            for (const auto& shell : auxiliary.libint()) primitives = std::max(primitives,
                static_cast<std::uint64_t>(shell.alpha.size()));
            for (const auto& q : census.q_records()) vectors = std::max(vectors, q.base_reciprocal_vector_count);
            if (primitives > 8 || vectors > 512 || caps.maximum_image_candidates_per_tile
                * vectors * primitives * primitives * shape.tile_count > 100000000)
                throw std::length_error("private factor stream diagnostic exceeds Fourier work cap");
            struct Sink {
                Writer& writer;
                std::uint64_t fail;
                static void receive(const vibeqc::PeriodicCorrelationThreeCenterTile& tile, void* context) {
                    auto& self = *static_cast<Sink*>(context);
                    if (tile.descriptor().sequence_index == self.fail)
                        throw std::runtime_error("private factor stream diagnostic receiver cancelled");
                    self.writer.accept(tile);
                }
            } sink{writer, fail_before_sequence};
            py::gil_scoped_release release;
            return vibeqc::stream_periodic_correlation_three_center_tiles(
                reference, schedule, census, ao, auxiliary, cutoff, negative, caps,
                &Sink::receive, &sink);
        } catch (...) { writer.abort(); throw; }
    }, py::arg("writer"), py::arg("reference"), py::arg("schedule"), py::arg("census"),
       py::arg("ao_basis"), py::arg("auxiliary_basis"), py::arg("image_cutoff_bohr"),
       py::arg("negative_tolerance"), py::arg("caps"),
       py::arg("fail_before_sequence") = std::numeric_limits<std::uint64_t>::max());
}
