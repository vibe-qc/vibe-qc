// Included by bindings.cpp. Tiny direct-array numerical oracles only.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/bounded_restricted_triples_target.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_triples_target(py::module_& m) {
    using Plan = vibeqc::BoundedRestrictedTriplesTargetMemoryPlan;
    using Caps = vibeqc::BoundedRestrictedTriplesTargetCaps;
    using Result = vibeqc::BoundedRestrictedTriplesTargetResult;
    py::class_<Plan>(m, "_BoundedRestrictedTriplesTargetMemoryPlan")
#define BOUNDED_TRIPLES_PLAN(field) .def_readonly(#field, &Plan::field)
        BOUNDED_TRIPLES_PLAN(n_occupied) BOUNDED_TRIPLES_PLAN(n_virtual)
        BOUNDED_TRIPLES_PLAN(maximum_source_virtual_dimension)
        BOUNDED_TRIPLES_PLAN(borrowed_target_bytes)
        BOUNDED_TRIPLES_PLAN(active_provider_view_bytes_upper_bound)
        BOUNDED_TRIPLES_PLAN(provider_retained_numerical_bytes)
        BOUNDED_TRIPLES_PLAN(provider_maximum_transient_numerical_bytes)
        BOUNDED_TRIPLES_PLAN(output_bytes) BOUNDED_TRIPLES_PLAN(compensation_bytes)
        BOUNDED_TRIPLES_PLAN(projection_workspace_bytes)
        BOUNDED_TRIPLES_PLAN(peak_owned_numerical_bytes)
        BOUNDED_TRIPLES_PLAN(total_live_numerical_bytes)
        BOUNDED_TRIPLES_PLAN(provider_visits_upper_bound)
        BOUNDED_TRIPLES_PLAN(kernel_work_units_upper_bound);
#undef BOUNDED_TRIPLES_PLAN
    py::class_<Caps>(m, "_BoundedRestrictedTriplesTargetCaps")
        .def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_total_numerical_bytes", &Caps::maximum_total_numerical_bytes)
        .def_readwrite("maximum_provider_visits", &Caps::maximum_provider_visits)
        .def_readwrite("maximum_kernel_work_units", &Caps::maximum_kernel_work_units);
    py::class_<Result>(m, "_BoundedRestrictedTriplesTargetResult")
        .def_readonly("memory", &Result::memory)
        .def_readonly("occupied", &Result::occupied)
        .def_readonly("amplitude_snapshot_id", &Result::amplitude_snapshot_id)
        .def_readonly("provider_visits", &Result::provider_visits)
        .def_readonly("exactly_zero_couplings_skipped", &Result::exactly_zero_couplings_skipped)
        .def_readonly("largest_source_virtual_dimension", &Result::largest_source_virtual_dimension)
        .def_readonly("maximum_absolute_residual", &Result::maximum_absolute_residual)
        .def_readonly("residual_frobenius_norm", &Result::residual_frobenius_norm)
        .def_readonly("raw_energy_contraction", &Result::raw_energy_contraction)
        .def_property_readonly("residual", [](const Result& result) {
            const auto v = result.memory.n_virtual;
            if (!v || v > 5 || result.residual.size() != v * v * v)
                throw std::length_error("bounded triples residual copy exceeds tiny extent");
            py::array_t<double> copy({static_cast<py::ssize_t>(v), static_cast<py::ssize_t>(v), static_cast<py::ssize_t>(v)});
            std::memcpy(copy.mutable_data(), result.residual.data(), result.residual.size() * sizeof(double));
            return copy;
        });
    m.def("_plan_bounded_restricted_triples_target", &vibeqc::plan_bounded_restricted_triples_target,
        py::arg("n_occupied"), py::arg("n_virtual"), py::arg("maximum_source_virtual_dimension"),
        py::arg("provider_retained_numerical_bytes") = 0,
        py::arg("provider_maximum_transient_numerical_bytes") = 0);

    m.def("_bounded_restricted_triples_target_residual_diagnostic", [](
        py::array amplitudes, py::array connected, py::array singles, py::array energies,
        py::array fock_rows, std::array<std::uint64_t, 3> occupied,
        py::list source_amplitudes, py::list overlaps, std::uint64_t maximum_source_dimension,
        const Caps& caps, std::uint64_t snapshot, std::uint64_t fail_before_visit,
        std::uint32_t protocol_fault, std::uint64_t provider_transient_bytes) {
        using U = std::uint64_t;
        using View = vibeqc::BoundedRestrictedTriplesRealView;
        const auto require_array = [](const py::array& array, int rank) {
            if (!array.dtype().is(py::dtype::of<double>()) || array.ndim() != rank
                || !(array.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
                throw std::invalid_argument("bounded triples diagnostic requires aligned C-contiguous binary64 arrays");
        };
        const auto dim = [](const py::array& array, int axis, U expected) {
            if (static_cast<U>(array.shape(axis)) != expected)
                throw std::invalid_argument("bounded triples diagnostic shape mismatch");
        };
        const auto view = [](const py::array& array) {
            return View{static_cast<const double*>(array.data()), static_cast<std::size_t>(array.size())};
        };
        require_array(amplitudes, 3); require_array(connected, 3); require_array(singles, 3);
        require_array(energies, 1); require_array(fock_rows, 2);
        const U v = static_cast<U>(amplitudes.shape(0)), o = static_cast<U>(fock_rows.shape(1));
        if (!o || !v || o > 4 || v > 5 || !maximum_source_dimension || maximum_source_dimension > 5
            || source_amplitudes.size() != 3 * o || overlaps.size() != 3 * o
            || caps.maximum_owned_numerical_bytes > (1U << 20)
            || caps.maximum_total_numerical_bytes > (2U << 20)
            || caps.maximum_provider_visits > 12 || caps.maximum_kernel_work_units > 100000000U
            || provider_transient_bytes > (1U << 20) || protocol_fault > 7)
            throw std::length_error("bounded triples diagnostic exceeds tiny extent or work limits");
        for (int axis = 0; axis < 3; ++axis) {
            dim(amplitudes, axis, v); dim(connected, axis, v); dim(singles, axis, v);
        }
        dim(energies, 0, v); dim(fock_rows, 0, 3);
        struct Slot { bool present = false; U dimension = 0; View amplitudes, overlap; };
        struct Direct {
            std::array<Slot, 12> slots;
            U occupied_count = 0, visits = 0, fail_before = 0;
            std::uint32_t fault = 0;
            static void visit(const vibeqc::BoundedRestrictedTriplesNeighbourRequest& request,
                              vibeqc::BoundedRestrictedTriplesNeighbourReceiver receiver,
                              void* receiver_context, void* provider_context) {
                auto& self = *static_cast<Direct*>(provider_context);
                if (self.visits == self.fail_before)
                    throw std::runtime_error("bounded triples diagnostic injected provider failure");
                ++self.visits;
                if (self.fault == 1) return; // missing receiver invocation
                const auto& slot = self.slots[request.replaced_axis * self.occupied_count + request.replacement_occupied];
                if (!slot.present)
                    throw std::runtime_error("bounded triples diagnostic has a missing nonzero-coupling neighbour");
                vibeqc::BoundedRestrictedTriplesNeighbourView borrowed{
                    request.ordered_occupied, request.amplitude_snapshot_id, slot.dimension, slot.amplitudes, slot.overlap};
                if (self.fault == 3) ++borrowed.amplitude_snapshot_id;
                if (self.fault == 4) borrowed.ordered_occupied[request.replaced_axis] = self.occupied_count;
                if (self.fault == 6) ++borrowed.amplitudes.element_count;
                if (self.fault == 7) ++borrowed.target_source_overlap.element_count;
                if (self.fault == 5) {
                    ++borrowed.amplitude_snapshot_id;
                    try { receiver(borrowed, receiver_context); } catch (...) {}
                    return; // maliciously swallowed receiver error must fail
                }
                receiver(borrowed, receiver_context);
                if (self.fault == 2) receiver(borrowed, receiver_context);
            }
        } direct{};
        direct.occupied_count = o; direct.fail_before = fail_before_visit; direct.fault = protocol_fault;
        U retained = 0;
        for (U index = 0; index < 3 * o; ++index) {
            if (source_amplitudes[index].is_none() && overlaps[index].is_none()) continue;
            if (!py::isinstance<py::array>(source_amplitudes[index]) || !py::isinstance<py::array>(overlaps[index]))
                throw std::invalid_argument("bounded triples diagnostic requires array or None neighbour pairs");
            const auto tensor = py::reinterpret_borrow<py::array>(source_amplitudes[index]);
            const auto overlap = py::reinterpret_borrow<py::array>(overlaps[index]);
            require_array(tensor, 3); require_array(overlap, 2);
            const U d = static_cast<U>(tensor.shape(0));
            if (d > 5) throw std::length_error("bounded triples diagnostic source exceeds tiny extent");
            dim(tensor, 1, d); dim(tensor, 2, d); dim(overlap, 0, v); dim(overlap, 1, d);
            direct.slots[index] = {true, d, view(tensor), view(overlap)};
            retained += static_cast<U>(tensor.nbytes() + overlap.nbytes());
        }
        vibeqc::BoundedRestrictedTriplesTargetInput input{o, v, occupied, snapshot,
            view(amplitudes), view(connected), view(singles), view(energies), view(fock_rows)};
        vibeqc::BoundedRestrictedTriplesNeighbourProvider provider{
            &Direct::visit, &direct, maximum_source_dimension, retained, provider_transient_bytes};
        py::gil_scoped_release release;
        return vibeqc::bounded_restricted_triples_target_residual(input, provider, caps);
    }, py::arg("amplitudes"), py::arg("connected_moment"), py::arg("singles_moment"),
       py::arg("virtual_energies"), py::arg("occupied_fock_rows"), py::arg("occupied"),
       py::arg("source_amplitudes"), py::arg("target_source_overlaps"), py::arg("maximum_source_dimension"),
       py::arg("caps"), py::arg("amplitude_snapshot_id") = 1,
       py::arg("fail_before_visit") = std::numeric_limits<std::uint64_t>::max(),
       py::arg("protocol_fault") = 0, py::arg("provider_transient_bytes") = 0);
}
