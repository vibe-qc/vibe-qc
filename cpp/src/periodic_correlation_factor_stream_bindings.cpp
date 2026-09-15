// Internal diagnostic bindings for the native periodic factor-stream contract.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "vibeqc/periodic_correlation_factor_stream.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

py::ssize_t checked_py_ssize_t(std::uint64_t value, const char* context) {
    if (value
        > static_cast<std::uint64_t>(
            std::numeric_limits<py::ssize_t>::max())) {
        throw std::overflow_error(
            std::string("periodic factor-stream Python size overflow: ")
            + context);
    }
    return static_cast<py::ssize_t>(value);
}

std::uint64_t checked_array_count(py::ssize_t value, const char* context) {
    if (value < 0) {
        throw std::overflow_error(
            std::string("periodic factor-stream negative Python size: ")
            + context);
    }
    return static_cast<std::uint64_t>(value);
}

bool same_descriptor(
    const vibeqc::PeriodicCorrelationFactorTileDescriptor& left,
    const vibeqc::PeriodicCorrelationFactorTileDescriptor& right) noexcept {
    return left.sequence_index == right.sequence_index
        && left.q_index == right.q_index
        && left.k_bra_index == right.k_bra_index
        && left.k_ket_index == right.k_ket_index
        && left.k_ket_reciprocal_wrap == right.k_ket_reciprocal_wrap
        && left.ao_pair_begin == right.ao_pair_begin
        && left.ao_pair_count == right.ao_pair_count
        && left.auxiliary_begin == right.auxiliary_begin
        && left.auxiliary_count == right.auxiliary_count
        && left.element_count == right.element_count;
}

}  // namespace

void bind_periodic_correlation_factor_stream(py::module_& m) {
    using Caps =
        vibeqc::PeriodicCorrelationSyntheticFactorExecutionCaps;
    using Descriptor = vibeqc::PeriodicCorrelationFactorTileDescriptor;
    using PayloadKind = vibeqc::PeriodicCorrelationFactorPayloadKind;
    using Receipt = vibeqc::PeriodicCorrelationSyntheticFactorReceipt;
    using Schedule = vibeqc::PeriodicCorrelationFactorStreamSchedule;
    using Shape = vibeqc::PeriodicCorrelationFactorStreamShape;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Transaction =
        vibeqc::PeriodicCorrelationSyntheticFactorTransaction;
    using TransactionState =
        vibeqc::PeriodicCorrelationSyntheticTransactionState;

    m.attr("_PERIODIC_CORRELATION_FACTOR_STREAM_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicCorrelationFactorStreamContractVersion);
    m.attr("_PERIODIC_CORRELATION_SYNTHETIC_FACTOR_PATTERN_VERSION") =
        py::int_(
            vibeqc::kPeriodicCorrelationSyntheticFactorPatternVersion);
    m.attr("_PERIODIC_CORRELATION_SYNTHETIC_FACTOR_TRANSACTION_VERSION") =
        py::int_(
            vibeqc::kPeriodicCorrelationSyntheticFactorTransactionVersion);

    py::class_<Shape>(m, "_PeriodicCorrelationFactorStreamShape")
        .def_readonly("n_kpoints", &Shape::n_kpoints)
        .def_readonly("n_basis", &Shape::n_basis)
        .def_readonly("n_auxiliary", &Shape::n_auxiliary)
        .def_readonly("n_ao_pairs", &Shape::n_ao_pairs)
        .def_readonly("auxiliary_block", &Shape::auxiliary_block)
        .def_readonly("ao_pair_block", &Shape::ao_pair_block)
        .def_readonly(
            "auxiliary_tile_count", &Shape::auxiliary_tile_count)
        .def_readonly("ao_pair_tile_count", &Shape::ao_pair_tile_count)
        .def_readonly("tiles_per_k_bra", &Shape::tiles_per_k_bra)
        .def_readonly("tiles_per_q", &Shape::tiles_per_q)
        .def_readonly("tile_count", &Shape::tile_count)
        .def_readonly(
            "logical_element_count", &Shape::logical_element_count)
        .def_readonly("logical_bytes", &Shape::logical_bytes)
        .def_readonly(
            "maximum_tile_element_count",
            &Shape::maximum_tile_element_count)
        .def_readonly("maximum_tile_bytes", &Shape::maximum_tile_bytes)
        .def_readonly(
            "two_buffer_workspace_bytes",
            &Shape::two_buffer_workspace_bytes);

    py::class_<Descriptor>(m, "_PeriodicCorrelationFactorTileDescriptor")
        .def_readonly("sequence_index", &Descriptor::sequence_index)
        .def_readonly("q_index", &Descriptor::q_index)
        .def_readonly("k_bra_index", &Descriptor::k_bra_index)
        .def_readonly("k_ket_index", &Descriptor::k_ket_index)
        .def_readonly(
            "k_ket_reciprocal_wrap",
            &Descriptor::k_ket_reciprocal_wrap)
        .def_readonly("ao_pair_begin", &Descriptor::ao_pair_begin)
        .def_readonly("ao_pair_count", &Descriptor::ao_pair_count)
        .def_readonly("auxiliary_begin", &Descriptor::auxiliary_begin)
        .def_readonly("auxiliary_count", &Descriptor::auxiliary_count)
        .def_readonly("element_count", &Descriptor::element_count);

    py::class_<Schedule>(m, "_PeriodicCorrelationFactorStreamSchedule")
        .def_property_readonly("contract_version", &Schedule::contract_version)
        .def_property_readonly(
            "state",
            [](const Schedule& schedule) -> std::shared_ptr<State> {
                const auto& state = schedule.state_handle();
                if (!state) {
                    throw std::logic_error(
                        "moved-from periodic factor-stream schedule has no "
                        "state");
                }
                return std::const_pointer_cast<State>(state);
            })
        .def_property_readonly(
            "calculation_identity", &Schedule::calculation_identity)
        .def_property_readonly(
            "allocation_identity", &Schedule::allocation_identity)
        .def_property_readonly(
            "state_identity_sha256", &Schedule::state_identity_sha256)
        .def_property_readonly(
            "schedule_identity_sha256", &Schedule::schedule_identity_sha256)
        .def_property_readonly("mesh", &Schedule::mesh)
        .def_property_readonly("is_shift", &Schedule::is_shift)
        .def_property_readonly(
            "shape", [](const Schedule& schedule) { return schedule.shape(); })
        .def("descriptor", &Schedule::descriptor, py::arg("sequence_index"))
        .def("ao_pair_index", &Schedule::ao_pair_index,
             py::arg("mu"), py::arg("nu"))
        .def("ao_pair_indices", &Schedule::ao_pair_indices,
             py::arg("ao_pair_index"))
        .def("payload_offset", &Schedule::payload_offset,
             py::arg("descriptor"), py::arg("auxiliary"),
             py::arg("ao_pair"));

    py::enum_<TransactionState>(
        m, "_PeriodicCorrelationSyntheticTransactionState")
        .value("OPEN", TransactionState::Open)
        .value("FAILED", TransactionState::Failed)
        .value("ABORTED", TransactionState::Aborted)
        .value("COMMITTED", TransactionState::Committed);
    py::enum_<PayloadKind>(m, "_PeriodicCorrelationFactorPayloadKind")
        .value(
            "SYNTHETIC_NONPHYSICAL_TEST_PATTERN",
            PayloadKind::SyntheticNonPhysicalTestPattern);

    py::class_<Receipt>(m, "_PeriodicCorrelationSyntheticFactorReceipt")
        .def_readonly(
            "transaction_contract_version",
            &Receipt::transaction_contract_version)
        .def_readonly(
            "synthetic_pattern_version", &Receipt::synthetic_pattern_version)
        .def_readonly("payload_kind", &Receipt::payload_kind)
        .def_readonly(
            "synthetic_nonphysical", &Receipt::synthetic_nonphysical)
        .def_readonly(
            "schedule_identity_sha256", &Receipt::schedule_identity_sha256)
        .def_readonly(
            "source_identity_sha256", &Receipt::source_identity_sha256)
        .def_readonly(
            "payload_identity_sha256", &Receipt::payload_identity_sha256)
        .def_readonly(
            "receipt_identity_sha256", &Receipt::receipt_identity_sha256)
        .def_readonly(
            "committed_tile_count", &Receipt::committed_tile_count)
        .def_readonly(
            "committed_element_count", &Receipt::committed_element_count)
        .def_readonly(
            "committed_logical_bytes", &Receipt::committed_logical_bytes);

    py::class_<Transaction>(
        m, "_PeriodicCorrelationSyntheticFactorTransaction")
        .def_property_readonly("contract_version", &Transaction::contract_version)
        .def_property_readonly("state", &Transaction::state)
        .def_property_readonly(
            "next_sequence_index", &Transaction::next_sequence_index)
        .def_property_readonly(
            "accepted_tile_count", &Transaction::accepted_tile_count)
        .def_property_readonly(
            "accepted_element_count", &Transaction::accepted_element_count)
        .def_property_readonly(
            "schedule_identity_sha256",
            &Transaction::schedule_identity_sha256)
        .def_property_readonly(
            "source_identity_sha256", &Transaction::source_identity_sha256)
        .def(
            "accept",
            [](Transaction& transaction,
               const Descriptor& descriptor,
               const py::object& payload_object,
               std::uint64_t max_payload_bytes) {
                if (transaction.state()
                    != TransactionState::Open) {
                    throw std::logic_error(
                        "periodic synthetic factor transaction is not open");
                }
                if (!py::isinstance<py::array>(payload_object)) {
                    throw std::invalid_argument(
                        "periodic synthetic factor transaction requires an "
                        "exact NumPy ndarray payload");
                }
                const auto payload =
                    py::reinterpret_borrow<py::array>(payload_object);
                if (payload.ndim() != 1
                    || !payload.dtype().is(
                        py::dtype::of<std::complex<double>>())
                    || (payload.flags() & py::array::c_style) == 0) {
                    throw std::invalid_argument(
                        "periodic synthetic factor transaction payload must "
                        "be one-dimensional, C-contiguous complex128");
                }
                if (reinterpret_cast<std::uintptr_t>(payload.data())
                    % alignof(std::complex<double>) != 0U) {
                    throw std::invalid_argument(
                        "periodic synthetic factor transaction payload is "
                        "not correctly aligned");
                }
                const std::uint64_t payload_count = checked_array_count(
                    payload.size(), "synthetic transaction payload");
                transaction.accept(
                    descriptor,
                    static_cast<const std::complex<double>*>(payload.data()),
                    payload_count,
                    max_payload_bytes);
            },
            py::arg("descriptor"), py::arg("payload"),
            py::arg("max_payload_bytes"))
        .def("commit", &Transaction::commit)
        .def("abort", &Transaction::abort);

    py::class_<Caps>(m, "_PeriodicCorrelationSyntheticFactorExecutionCaps")
        .def(py::init<>())
        .def_readwrite("max_tile_bytes", &Caps::max_tile_bytes)
        .def_readwrite("max_workspace_bytes", &Caps::max_workspace_bytes)
        .def_readwrite("max_tile_count", &Caps::max_tile_count)
        .def_readwrite("max_logical_bytes", &Caps::max_logical_bytes);

    m.def(
        "_estimate_periodic_correlation_factor_stream_shape",
        &vibeqc::estimate_periodic_correlation_factor_stream_shape,
        py::arg("mesh"), py::arg("n_basis"), py::arg("n_auxiliary"),
        py::arg("auxiliary_block"), py::arg("ao_pair_block"));
    m.def(
        "_estimate_periodic_correlation_synthetic_payload_wire_bytes",
        &vibeqc::estimate_periodic_correlation_synthetic_payload_wire_bytes,
        py::arg("shape"));
    m.def(
        "_make_periodic_correlation_factor_stream_schedule",
        &vibeqc::make_periodic_correlation_factor_stream_schedule,
        py::arg("reference"));
    m.def(
        "_periodic_correlation_synthetic_factor_tile",
        [](const Schedule& schedule,
           const Descriptor& descriptor,
           std::uint64_t max_tile_bytes) {
            const auto expected = schedule.descriptor(
                descriptor.sequence_index);
            if (!same_descriptor(descriptor, expected)) {
                throw std::invalid_argument(
                    "synthetic factor tile received a noncanonical "
                    "descriptor");
            }
            if (max_tile_bytes == 0U) {
                throw std::invalid_argument(
                    "synthetic factor tile requires an explicit nonzero "
                    "binding cap");
            }
            if (expected.element_count
                > max_tile_bytes / sizeof(std::complex<double>)) {
                throw std::runtime_error(
                    "synthetic factor tile exceeds explicit binding cap");
            }
            if (expected.element_count
                > std::numeric_limits<std::size_t>::max()
                    / sizeof(std::complex<double>)) {
                throw std::overflow_error(
                    "synthetic factor tile byte count exceeds size_t");
            }
            const py::ssize_t count = checked_py_ssize_t(
                expected.element_count,
                "synthetic factor tile element count");
            py::array_t<std::complex<double>> payload(count);
            vibeqc::fill_periodic_correlation_synthetic_factor_tile(
                schedule,
                descriptor,
                payload.mutable_data(),
                expected.element_count);
            return payload;
        },
        py::arg("schedule"), py::arg("descriptor"),
        py::arg("max_tile_bytes"));
    m.def(
        "_make_periodic_correlation_synthetic_factor_transaction",
        &vibeqc::make_periodic_correlation_synthetic_factor_transaction,
        py::arg("schedule"));
    m.def(
        "_execute_periodic_correlation_synthetic_factor_stream",
        &vibeqc::execute_periodic_correlation_synthetic_factor_stream,
        py::arg("schedule"), py::arg("caps"));
}
