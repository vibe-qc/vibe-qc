// Included by bindings.cpp. Strictly tiny common-space reference diagnostic.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <limits>
#include "vibeqc/bounded_restricted_triples_solver.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_triples_solver(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = BoundedRestrictedTriplesSolverOptions;
    using Inventory = BoundedRestrictedTriplesSolverInventory;
    using Caps = BoundedRestrictedTriplesSolverCaps;
    using Plan = BoundedRestrictedTriplesSolverMemoryPlan;
    using Progress = BoundedRestrictedTriplesSolverProgress;
    using Result = BoundedRestrictedTriplesSolverResult;
    py::class_<Options>(m, "_BoundedRestrictedTriplesSolverOptions").def(py::init<>())
#define TRIPLES_SOLVER_OPTION(field) .def_readwrite(#field, &Options::field)
        TRIPLES_SOLVER_OPTION(maximum_iterations) TRIPLES_SOLVER_OPTION(denominator_floor)
        TRIPLES_SOLVER_OPTION(residual_tolerance) TRIPLES_SOLVER_OPTION(energy_tolerance)
        TRIPLES_SOLVER_OPTION(input_symmetry_tolerance);
#undef TRIPLES_SOLVER_OPTION
    py::class_<Inventory>(m, "_BoundedRestrictedTriplesSolverInventory").def(py::init<>())
        .def_readwrite("external_node_numerical_bytes", &Inventory::external_node_numerical_bytes)
        .def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    py::class_<Caps>(m, "_BoundedRestrictedTriplesSolverCaps").def(py::init<>())
#define TRIPLES_SOLVER_CAP(field) .def_readwrite(#field, &Caps::field)
        TRIPLES_SOLVER_CAP(maximum_owned_numerical_bytes) TRIPLES_SOLVER_CAP(maximum_total_numerical_bytes)
        TRIPLES_SOLVER_CAP(maximum_node_numerical_bytes) TRIPLES_SOLVER_CAP(maximum_integral_calls)
        TRIPLES_SOLVER_CAP(maximum_kernel_work_units);
#undef TRIPLES_SOLVER_CAP
    py::class_<Plan>(m, "_BoundedRestrictedTriplesSolverMemoryPlan")
#define TRIPLES_SOLVER_PLAN(field) .def_readonly(#field, &Plan::field)
        TRIPLES_SOLVER_PLAN(n_occupied) TRIPLES_SOLVER_PLAN(n_virtual) TRIPLES_SOLVER_PLAN(maximum_iterations)
        TRIPLES_SOLVER_PLAN(amplitude_snapshot_bytes) TRIPLES_SOLVER_PLAN(candidate_snapshot_bytes)
        TRIPLES_SOLVER_PLAN(borrowed_input_bytes) TRIPLES_SOLVER_PLAN(orbital_workspace_bytes)
        TRIPLES_SOLVER_PLAN(peak_owned_numerical_bytes) TRIPLES_SOLVER_PLAN(total_live_numerical_bytes)
        TRIPLES_SOLVER_PLAN(external_node_numerical_bytes) TRIPLES_SOLVER_PLAN(numerical_replicas)
        TRIPLES_SOLVER_PLAN(required_node_numerical_bytes) TRIPLES_SOLVER_PLAN(target_evaluations_upper_bound)
        TRIPLES_SOLVER_PLAN(neighbour_visits_upper_bound) TRIPLES_SOLVER_PLAN(integral_calls_upper_bound)
        TRIPLES_SOLVER_PLAN(kernel_work_units_upper_bound) TRIPLES_SOLVER_PLAN(moments) TRIPLES_SOLVER_PLAN(target);
#undef TRIPLES_SOLVER_PLAN
    py::class_<Progress>(m, "_BoundedRestrictedTriplesSolverProgress")
#define TRIPLES_SOLVER_PROGRESS(field) .def_readonly(#field, &Progress::field)
        TRIPLES_SOLVER_PROGRESS(iteration) TRIPLES_SOLVER_PROGRESS(integral_calls)
        TRIPLES_SOLVER_PROGRESS(target_evaluations) TRIPLES_SOLVER_PROGRESS(neighbour_visits)
        TRIPLES_SOLVER_PROGRESS(triples_energy) TRIPLES_SOLVER_PROGRESS(energy_change)
        TRIPLES_SOLVER_PROGRESS(maximum_absolute_residual) TRIPLES_SOLVER_PROGRESS(residual_frobenius_norm)
        TRIPLES_SOLVER_PROGRESS(has_previous_energy) TRIPLES_SOLVER_PROGRESS(converged);
#undef TRIPLES_SOLVER_PROGRESS
    py::class_<Result>(m, "_BoundedRestrictedTriplesSolverResult")
        .def_readonly("memory", &Result::memory).def_readonly("final_snapshot", &Result::final_snapshot)
        .def_readonly("ccsd_snapshot_id", &Result::ccsd_snapshot_id)
        .def_readonly("minimum_denominator", &Result::minimum_denominator)
        .def_readonly("maximum_denominator", &Result::maximum_denominator)
        .def_property_readonly("amplitudes", [](const Result& r) {
            const auto o = r.memory.n_occupied, v = r.memory.n_virtual;
            // Authenticated o=v=4 results copy at most 4096 doubles
            // (32768 bytes). This does not broaden solver-input admission.
            if (!o || !v || o > 4 || v > 4 || r.amplitudes.size() != o * o * o * v * v * v)
                throw std::length_error("bounded triples solver copy exceeds tiny extent");
            const auto oi = static_cast<py::ssize_t>(o), vi = static_cast<py::ssize_t>(v);
            py::array_t<double> out({oi, oi, oi, vi, vi, vi});
            std::memcpy(out.mutable_data(), r.amplitudes.data(), r.amplitudes.size() * sizeof(double));
            return out;
        });
    m.def("_plan_bounded_restricted_triples_solver", &plan_bounded_restricted_triples_solver,
        py::arg("n_occupied"), py::arg("n_virtual"), py::arg("maximum_iterations"), py::arg("inventory"),
        py::arg("provider_retained_numerical_bytes") = 0, py::arg("provider_maximum_transient_numerical_bytes") = 0);
    m.def("_bounded_restricted_triples_solve_diagnostic", [](
        const py::array& t1, const py::array& t2, const py::array& foo, const py::array& fvv,
        const py::array& fov, const py::array& eri, U snapshot,
        const Options& options, const Inventory& inventory, const Caps& caps,
        const py::object& progress, U fail_before_call, U transient) {
        const auto array = [](const py::array& a, int ndim) {
            if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != ndim
                || !(a.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double))
                throw std::invalid_argument("bounded triples solver diagnostic needs aligned C-contiguous binary64 arrays");
        };
        array(t1, 2); array(t2, 4); array(foo, 2); array(fvv, 2); array(fov, 2); array(eri, 4);
        const U o = foo.shape(0), v = fvv.shape(0), n = o + v;
        if (!o || !v || o > 3 || v > 4 || options.maximum_iterations > 256
            || inventory.numerical_replicas > 8 || inventory.external_node_numerical_bytes > (2U << 20)
            || caps.maximum_owned_numerical_bytes > (1U << 20)
            || caps.maximum_total_numerical_bytes > (2U << 20)
            || caps.maximum_node_numerical_bytes > (8U << 20)
            || caps.maximum_integral_calls > 100000000U
            || caps.maximum_kernel_work_units > 20000000000ULL || transient > (1U << 20))
            throw std::length_error("bounded triples solver diagnostic exceeds tiny shape or work limits");
        const auto dim = [](const py::array& a, int axis, U size) {
            if (static_cast<U>(a.shape(axis)) != size)
                throw std::invalid_argument("bounded triples solver diagnostic shape mismatch");
        };
        dim(t1, 0, o); dim(t1, 1, v); dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
        dim(foo, 1, o); dim(fvv, 1, v); dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(eri, axis, n);
        const auto as_view = [](const py::array& a) {
            return BoundedRestrictedCCSDRealView{static_cast<const double*>(a.data()), static_cast<std::size_t>(a.size())};
        };
        const BoundedRestrictedTriplesSolverInput input{o, v, as_view(t1), as_view(t2), as_view(foo),
            as_view(fvv), as_view(fov), snapshot};
        if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            throw std::invalid_argument("bounded triples solver progress must be callable or absent");
        struct Direct {
            const double* data;
            U n, calls, fail;
            static double value(U p, U q, U r, U s, void* context) {
                auto& self = *static_cast<Direct*>(context);
                if (self.calls == self.fail) throw std::runtime_error("injected triples integral callback failure");
                ++self.calls;
                if (p >= self.n || q >= self.n || r >= self.n || s >= self.n)
                    throw std::logic_error("triples diagnostic integral index out of range");
                return self.data[((p * self.n + q) * self.n + r) * self.n + s];
            }
        } direct{static_cast<const double*>(eri.data()), n, 0, fail_before_call};
        struct Sink {
            const py::object& callable;
            static void receive(const Progress& record, void* context) {
                const auto& self = *static_cast<Sink*>(context);
                py::gil_scoped_acquire acquire;
                self.callable(py::cast(record, py::return_value_policy::copy));
            }
        } sink{progress};
        const BoundedRestrictedCCSDIntegralProvider provider{&Direct::value, &direct, static_cast<U>(eri.nbytes()), transient};
        const auto selected = options; const auto owners = inventory; const auto limits = caps;
        const auto callback = progress.is_none() ? nullptr : &Sink::receive;
        py::gil_scoped_release release;
        return bounded_restricted_triples_solve(input, provider, selected, owners, limits, callback, &sink);
    }, py::arg("t1").noconvert(), py::arg("t2").noconvert(), py::arg("f_oo").noconvert(),
        py::arg("f_vv").noconvert(), py::arg("f_ov").noconvert(), py::arg("integrals").noconvert(),
        py::arg("ccsd_snapshot_id"), py::arg("options"), py::arg("inventory"), py::arg("caps"),
        py::arg("progress") = py::none(), py::arg("fail_before_call") = std::numeric_limits<U>::max(),
        py::arg("provider_transient_bytes") = 0);
}
