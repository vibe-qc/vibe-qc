// Tiny float64-only algebra diagnostic. No physical pair topology is inferred.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "vibeqc/periodic_correlation_pair_residual.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pair_residual(py::module_& m) {
    using Controls = vibeqc::PairResidualControls;
    using Memory = vibeqc::PairResidualMemoryPlan;
    using Diagnostics = vibeqc::PairResidualDiagnostics;
    using Result = vibeqc::PairResidualResult;
    using Accumulator = vibeqc::PairResidualAccumulator;
    using Leg = vibeqc::PairResidualOccupiedLeg;

    py::enum_<Leg>(m, "_PairResidualOccupiedLeg")
        .value("FIRST", Leg::First)
        .value("SECOND", Leg::Second);
    py::class_<Controls>(m, "_PairResidualControls")
        .def(py::init<>())
        .def_readwrite("occupied_label_count", &Controls::occupied_label_count)
        .def_readwrite("target_first", &Controls::target_first)
        .def_readwrite("target_second", &Controls::target_second)
        .def_readwrite("expected_coupling_count", &Controls::expected_coupling_count)
        .def_readwrite("maximum_source_dimension", &Controls::maximum_source_dimension)
        .def_readwrite("maximum_scalar_products", &Controls::maximum_scalar_products)
        .def_readwrite("denominator_floor", &Controls::denominator_floor);
    py::class_<Memory>(m, "_PairResidualMemoryPlan")
        .def_readonly("target_dimension", &Memory::target_dimension)
        .def_readonly("maximum_source_dimension", &Memory::maximum_source_dimension)
        .def_readonly("expected_coupling_count", &Memory::expected_coupling_count)
        .def_readonly("borrowed_target_input_bytes", &Memory::borrowed_target_input_bytes)
        .def_readonly("maximum_borrowed_source_input_bytes", &Memory::maximum_borrowed_source_input_bytes)
        .def_readonly("residual_bytes", &Memory::residual_bytes)
        .def_readonly("compensation_bytes", &Memory::compensation_bytes)
        .def_readonly("projection_workspace_bytes", &Memory::projection_workspace_bytes)
        .def_readonly("peak_owned_numerical_bytes", &Memory::peak_owned_numerical_bytes)
        .def_readonly("output_numerical_bytes", &Memory::output_numerical_bytes)
        .def_readonly("maximum_scalar_products_per_source", &Memory::maximum_scalar_products_per_source)
        .def_readonly("maximum_total_scalar_products", &Memory::maximum_total_scalar_products);
    py::class_<Diagnostics>(m, "_PairResidualDiagnostics")
        .def_readonly("accepted_coupling_count", &Diagnostics::accepted_coupling_count)
        .def_readonly("scalar_product_count", &Diagnostics::scalar_product_count)
        .def_readonly("transposed_source_count", &Diagnostics::transposed_source_count)
        .def_readonly("zero_rank_source_count", &Diagnostics::zero_rank_source_count)
        .def_readonly("maximum_observed_source_dimension", &Diagnostics::maximum_observed_source_dimension)
        .def_readonly("maximum_observed_source_input_bytes", &Diagnostics::maximum_observed_source_input_bytes)
        .def_readonly("product_underflow_count", &Diagnostics::product_underflow_count)
        .def_readonly("minimum_denominator", &Diagnostics::minimum_denominator)
        .def_readonly("maximum_denominator", &Diagnostics::maximum_denominator)
        .def_readonly("maximum_absolute_residual", &Diagnostics::maximum_absolute_residual)
        .def_readonly("residual_frobenius_norm", &Diagnostics::residual_frobenius_norm);
    py::class_<Result>(m, "_PairResidualResult")
        .def_property_readonly("target_dimension", &Result::target_dimension)
        .def_property_readonly("memory", [](const Result& value) { return value.memory(); })
        .def_property_readonly("controls", [](const Result& value) { return value.controls(); })
        .def_property_readonly("diagnostics", [](const Result& value) { return value.diagnostics(); })
        .def("residual", &Result::residual, py::arg("row"), py::arg("column"))
        .def("residual_copy", [](const Result& value) {
            const auto n = value.target_dimension();
            if (n > 16U) throw std::length_error("Pair residual diagnostic copy is limited to rank 16");
            const auto* source = value.residual_data();
            py::array_t<double> answer({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            std::memcpy(answer.mutable_data(), source, n * n * sizeof(double));
            return answer;
        });
    py::class_<Accumulator>(m, "_PairResidualAccumulator")
        .def_property_readonly("is_open", &Accumulator::is_open)
        .def_property_readonly("accepted_coupling_count", &Accumulator::accepted_coupling_count)
        .def_property_readonly("memory", [](const Accumulator& value) { return value.memory(); })
        .def("abort", &Accumulator::abort)
        .def("finish", [](Accumulator& value) {
            // Serialize this tiny mutable diagnostic against Python abort or
            // another accumulation. Native callers own synchronization.
            return value.finish();
        })
        .def("accumulate", [](Accumulator& value, Leg leg, std::uint64_t substituted,
             std::uint64_t stored_first, std::uint64_t stored_second,
             const py::array& source_amplitudes, const py::array& overlap, double fock) {
            try {
                const auto n = value.memory().target_dimension;
                if (!source_amplitudes.dtype().is(py::dtype::of<double>())
                    || source_amplitudes.ndim() != 2 || !(source_amplitudes.flags() & py::array::c_style)
                    || source_amplitudes.shape(0) != source_amplitudes.shape(1)
                    || !overlap.dtype().is(py::dtype::of<double>()) || overlap.ndim() != 2
                    || !(overlap.flags() & py::array::c_style)
                    || overlap.shape(0) != static_cast<py::ssize_t>(n)
                    || overlap.shape(1) != source_amplitudes.shape(0)) {
                    throw std::invalid_argument("Pair residual source needs existing C-contiguous float64 T[m,m] and O[n,m]");
                }
                if (n > 16U || source_amplitudes.shape(0) > 16) {
                    throw std::length_error("Pair residual diagnostic target/source rank is limited to 16");
                }
                vibeqc::PairResidualSource source;
                source.leg = leg;
                source.substituted_occupied = substituted;
                source.stored_first = stored_first;
                source.stored_second = stored_second;
                source.source_dimension = static_cast<std::size_t>(source_amplitudes.shape(0));
                source.occupied_fock_coupling = fock;
                source.amplitudes = static_cast<const double*>(source_amplitudes.data());
                source.amplitude_elements = static_cast<std::size_t>(source_amplitudes.size());
                source.target_source_overlap = static_cast<const double*>(overlap.data());
                source.overlap_elements = static_cast<std::size_t>(overlap.size());
                value.accumulate(source);
            } catch (...) {
                value.abort();
                throw;
            }
        }, py::arg("leg"), py::arg("substituted_occupied"), py::arg("stored_first"),
        py::arg("stored_second"), py::arg("source_amplitudes").noconvert(),
        py::arg("target_source_overlap").noconvert(), py::arg("occupied_fock_coupling"),
        "Borrow source arrays for one arithmetic slot; any error aborts the native accumulator.");
    m.def("_plan_pair_residual", &vibeqc::plan_pair_residual,
          py::arg("target_dimension"), py::arg("maximum_source_dimension"),
          py::arg("expected_coupling_count"));
    m.def("_initialize_pair_residual",
        [](const py::array& g, const py::array& t, const py::array& eps,
           double fii, double fjj, const Controls& controls, std::uint64_t cap) {
            if (!g.dtype().is(py::dtype::of<double>()) || g.ndim() != 2
                || !(g.flags() & py::array::c_style) || g.shape(0) != g.shape(1)
                || !t.dtype().is(py::dtype::of<double>()) || t.ndim() != 2
                || !(t.flags() & py::array::c_style) || t.shape(0) != g.shape(0) || t.shape(1) != g.shape(1)
                || !eps.dtype().is(py::dtype::of<double>()) || eps.ndim() != 1
                || !(eps.flags() & py::array::c_style) || eps.shape(0) != g.shape(0)) {
                throw std::invalid_argument("Pair residual target needs existing C-contiguous float64 G[n,n], T[n,n], eps[n]");
            }
            if (g.shape(0) > 16 || controls.maximum_source_dimension > 16U
                || controls.expected_coupling_count > 128U || controls.occupied_label_count > 64U) {
                throw std::length_error("Pair residual diagnostic is limited to rank 16, 64 occupied labels and 128 couplings");
            }
            const auto n = static_cast<std::size_t>(g.shape(0));
            const auto* gp = static_cast<const double*>(g.data());
            const auto* tp = static_cast<const double*>(t.data());
            const auto* ep = static_cast<const double*>(eps.data());
            py::gil_scoped_release release;
            return vibeqc::initialize_pair_residual(gp, static_cast<std::size_t>(g.size()),
                tp, static_cast<std::size_t>(t.size()), ep, static_cast<std::size_t>(eps.size()),
                n, fii, fjj, controls, cap);
        }, py::arg("exchange_integrals").noconvert(), py::arg("target_amplitudes").noconvert(),
        py::arg("virtual_energies").noconvert(), py::arg("occupied_fock_ii"),
        py::arg("occupied_fock_jj"), py::arg("controls"), py::arg("owned_numerical_byte_cap"),
        "Initialize bounded real MP2 target residual algebra; not a torus map, convergence decision or energy.");
}
