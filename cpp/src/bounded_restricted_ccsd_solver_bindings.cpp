// Included by bindings.cpp. Tiny direct-integral reference diagnostics only.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/bounded_restricted_ccsd_solver.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_ccsd_solver(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = BoundedRestrictedCCSDSolverOptions;
    using Inventory = BoundedRestrictedCCSDSolverInventory;
    using Caps = BoundedRestrictedCCSDSolverCaps;
    using Plan = BoundedRestrictedCCSDSolverMemoryPlan;
    using Progress = BoundedRestrictedCCSDSolverProgress;
    using Result = BoundedRestrictedCCSDSolverResult;
    py::class_<Options>(m, "_BoundedRestrictedCCSDSolverOptions")
        .def(py::init<>())
#define CCSD_SOLVER_OPTION(field) .def_readwrite(#field, &Options::field)
        CCSD_SOLVER_OPTION(maximum_iterations) CCSD_SOLVER_OPTION(denominator_floor)
        CCSD_SOLVER_OPTION(singles_residual_tolerance) CCSD_SOLVER_OPTION(doubles_residual_tolerance)
        CCSD_SOLVER_OPTION(energy_tolerance) CCSD_SOLVER_OPTION(input_symmetry_tolerance);
#undef CCSD_SOLVER_OPTION
    py::class_<Inventory>(m, "_BoundedRestrictedCCSDSolverInventory")
        .def(py::init<>())
        .def_readwrite("external_node_numerical_bytes", &Inventory::external_node_numerical_bytes)
        .def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    py::class_<Caps>(m, "_BoundedRestrictedCCSDSolverCaps")
        .def(py::init<>())
#define CCSD_SOLVER_CAP(field) .def_readwrite(#field, &Caps::field)
        CCSD_SOLVER_CAP(maximum_owned_numerical_bytes) CCSD_SOLVER_CAP(maximum_total_numerical_bytes)
        CCSD_SOLVER_CAP(maximum_node_numerical_bytes) CCSD_SOLVER_CAP(maximum_integral_calls)
        CCSD_SOLVER_CAP(maximum_kernel_work_units);
#undef CCSD_SOLVER_CAP
    py::class_<Plan>(m, "_BoundedRestrictedCCSDSolverMemoryPlan")
#define CCSD_SOLVER_PLAN(field) .def_readonly(#field, &Plan::field)
        CCSD_SOLVER_PLAN(n_occupied) CCSD_SOLVER_PLAN(n_virtual) CCSD_SOLVER_PLAN(maximum_iterations)
        CCSD_SOLVER_PLAN(supplied_initial_amplitudes) CCSD_SOLVER_PLAN(amplitude_snapshot_bytes)
        CCSD_SOLVER_PLAN(candidate_snapshot_bytes) CCSD_SOLVER_PLAN(borrowed_fock_bytes)
        CCSD_SOLVER_PLAN(borrowed_initial_amplitude_bytes) CCSD_SOLVER_PLAN(peak_owned_numerical_bytes)
        CCSD_SOLVER_PLAN(total_live_numerical_bytes) CCSD_SOLVER_PLAN(external_node_numerical_bytes)
        CCSD_SOLVER_PLAN(numerical_replicas) CCSD_SOLVER_PLAN(required_node_numerical_bytes)
        CCSD_SOLVER_PLAN(target_evaluations_upper_bound) CCSD_SOLVER_PLAN(integral_calls_upper_bound)
        CCSD_SOLVER_PLAN(kernel_work_units_upper_bound) CCSD_SOLVER_PLAN(target);
#undef CCSD_SOLVER_PLAN
    py::class_<Progress>(m, "_BoundedRestrictedCCSDSolverProgress")
#define CCSD_SOLVER_PROGRESS(field) .def_readonly(#field, &Progress::field)
        CCSD_SOLVER_PROGRESS(iteration) CCSD_SOLVER_PROGRESS(integral_calls)
        CCSD_SOLVER_PROGRESS(target_evaluations) CCSD_SOLVER_PROGRESS(correlation_energy)
        CCSD_SOLVER_PROGRESS(energy_change) CCSD_SOLVER_PROGRESS(has_previous_energy)
        CCSD_SOLVER_PROGRESS(converged) CCSD_SOLVER_PROGRESS(singles_max_residual)
        CCSD_SOLVER_PROGRESS(doubles_max_residual) CCSD_SOLVER_PROGRESS(singles_residual_norm)
        CCSD_SOLVER_PROGRESS(doubles_residual_norm);
#undef CCSD_SOLVER_PROGRESS
    py::class_<Result>(m, "_BoundedRestrictedCCSDSolverResult")
        .def_readonly("memory", &Result::memory)
        .def_readonly("final_snapshot", &Result::final_snapshot)
        .def_readonly("minimum_denominator", &Result::minimum_denominator)
        .def_readonly("maximum_denominator", &Result::maximum_denominator)
        .def_property_readonly("t1", [](const Result& r) {
            const U o = r.memory.n_occupied, v = r.memory.n_virtual;
            // The authenticated full two-cell He2 reference has o=v=4.
            // This copy is at most 16 doubles; solver-input caps below
            // remain independently restricted to their diagnostic shape.
            if (!o || !v || o > 4 || v > 4 || r.t1.size() != o * v)
                throw std::length_error("bounded CCSD solver copy exceeds tiny extent");
            py::array_t<double> out({static_cast<py::ssize_t>(o), static_cast<py::ssize_t>(v)});
            std::memcpy(out.mutable_data(), r.t1.data(), r.t1.size() * 8);
            return out;
        })
        .def_property_readonly("t2", [](const Result& r) {
            const U o = r.memory.n_occupied, v = r.memory.n_virtual;
            // At most 256 doubles (2048 bytes), only from an owned result.
            if (!o || !v || o > 4 || v > 4 || r.t2.size() != o * o * v * v)
                throw std::length_error("bounded CCSD solver copy exceeds tiny extent");
            py::array_t<double> out({static_cast<py::ssize_t>(o), static_cast<py::ssize_t>(o),
                                    static_cast<py::ssize_t>(v), static_cast<py::ssize_t>(v)});
            std::memcpy(out.mutable_data(), r.t2.data(), r.t2.size() * 8);
            return out;
        });
    m.def("_plan_bounded_restricted_ccsd_solver", &plan_bounded_restricted_ccsd_solver,
        py::arg("n_occupied"), py::arg("n_virtual"), py::arg("maximum_iterations"),
        py::arg("supplied_initial_amplitudes"), py::arg("inventory"),
        py::arg("provider_retained_numerical_bytes") = 0,
        py::arg("provider_maximum_transient_numerical_bytes") = 0);
    m.def("_bounded_restricted_ccsd_solve_diagnostic", [](
        const py::array& foo, const py::array& fvv, const py::array& fov, const py::array& eri,
        const Options& options, const Inventory& inventory, const Caps& caps,
        const py::object& initial_t1, const py::object& initial_t2, const py::object& progress,
        U fail_before_call, U provider_transient_bytes) {
        const auto array = [](const py::array& a, int ndim) {
            if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != ndim
                || !(a.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double))
                throw std::invalid_argument("bounded CCSD solver diagnostic requires aligned C-contiguous binary64 arrays");
        };
        array(foo, 2); array(fvv, 2); array(fov, 2); array(eri, 4);
        const U o = static_cast<U>(foo.shape(0)), v = static_cast<U>(fvv.shape(0)), n = o + v;
        if (!o || !v || o > 3 || v > 4 || options.maximum_iterations > 1000
            || inventory.numerical_replicas > 8 || inventory.external_node_numerical_bytes > (2U << 20)
            || caps.maximum_owned_numerical_bytes > (1U << 20)
            || caps.maximum_total_numerical_bytes > (2U << 20)
            || caps.maximum_node_numerical_bytes > (8U << 20)
            || caps.maximum_integral_calls > 100000000U
            || caps.maximum_kernel_work_units > 20000000000ULL
            || provider_transient_bytes > (1U << 20))
            throw std::length_error("bounded CCSD solver diagnostic exceeds tiny shape or work limits");
        const auto dim = [](const py::array& a, int axis, U size) {
            if (static_cast<U>(a.shape(axis)) != size)
                throw std::invalid_argument("bounded CCSD solver diagnostic input shape mismatch");
        };
        dim(foo, 1, o); dim(fvv, 1, v); dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(eri, axis, n);
        const auto view = [](const py::array& a) {
            return BoundedRestrictedCCSDRealView{static_cast<const double*>(a.data()), static_cast<std::size_t>(a.size())};
        };
        BoundedRestrictedCCSDSolverInput input{o, v, view(foo), view(fvv), view(fov), {}, {}};
        py::array t1, t2;
        if (!initial_t1.is_none() || !initial_t2.is_none()) {
            if (!py::isinstance<py::array>(initial_t1) || !py::isinstance<py::array>(initial_t2))
                throw std::invalid_argument("bounded CCSD solver diagnostic initial amplitudes must both be arrays or both absent");
            t1 = py::reinterpret_borrow<py::array>(initial_t1);
            t2 = py::reinterpret_borrow<py::array>(initial_t2);
            array(t1, 2); array(t2, 4);
            dim(t1, 0, o); dim(t1, 1, v); dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
            input.initial_t1 = view(t1); input.initial_t2 = view(t2);
        }
        if (!progress.is_none() && !PyCallable_Check(progress.ptr()))
            throw std::invalid_argument("bounded CCSD solver diagnostic progress must be callable or absent");
        struct Direct {
            const double* data;
            U n, calls = 0, fail_before;
            static double value(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                if (d.calls == d.fail_before)
                    throw std::runtime_error("bounded CCSD solver diagnostic injected integral callback failure");
                ++d.calls;
                if (p >= d.n || q >= d.n || r >= d.n || s >= d.n)
                    throw std::logic_error("bounded CCSD solver diagnostic integral index is out of range");
                return d.data[((p * d.n + q) * d.n + r) * d.n + s];
            }
        } direct{static_cast<const double*>(eri.data()), n, 0, fail_before_call};
        struct Sink {
            const py::object* callable;
            static void receive(const Progress& p, void* context) {
                auto& sink = *static_cast<Sink*>(context);
                py::gil_scoped_acquire acquire;
                // Copy the scalar event, never expose the stack reference.
                (*sink.callable)(py::cast(p, py::return_value_policy::copy));
            }
        } sink{&progress};
        const BoundedRestrictedCCSDIntegralProvider provider{&Direct::value, &direct,
            static_cast<U>(eri.nbytes()), provider_transient_bytes};
        const Options selected = options; const Inventory owners = inventory; const Caps limits = caps;
        const auto callback = progress.is_none() ? nullptr : &Sink::receive;
        py::gil_scoped_release release;
        return bounded_restricted_ccsd_solve(input, provider, selected, owners, limits, callback, &sink);
    }, py::arg("f_oo").noconvert(), py::arg("f_vv").noconvert(), py::arg("f_ov").noconvert(),
       py::arg("integrals").noconvert(), py::arg("options"), py::arg("inventory"), py::arg("caps"),
       py::arg("initial_t1") = py::none(), py::arg("initial_t2") = py::none(), py::arg("progress") = py::none(),
       py::arg("fail_before_call") = std::numeric_limits<U>::max(), py::arg("provider_transient_bytes") = 0);
}
