// Tiny ragged triples diagnostics. Supplied moments are not physical certificates.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <limits>
#include <stdexcept>
#include "vibeqc/bounded_restricted_local_triples_solver.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace bounded_local_triples_binding {
using namespace vibeqc;
using U = std::uint64_t;
using View = BoundedRestrictedTriplesRealView;
using Request = BoundedRestrictedLocalTriplesMomentRequest;
using Receiver = BoundedRestrictedLocalTriplesMomentReceiver;
using Progress = BoundedRestrictedLocalTriplesSolverProgress;
py::array array(py::handle object, int ndim) {
    if (!py::isinstance<py::array>(object))
        throw std::invalid_argument("local triples diagnostic requires existing arrays");
    auto a = py::reinterpret_borrow<py::array>(object);
    if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != ndim
        || !(a.flags() & py::array::c_style)
        || (a.size() && reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double)))
        throw std::invalid_argument("local triples diagnostic requires aligned C-contiguous binary64 arrays");
    return a;
}
View view(const py::array& a) {
    return {static_cast<const double*>(a.data()), static_cast<std::size_t>(a.size())};
}
struct Direct {
    std::array<py::array, 82> owners;
    std::array<BoundedRestrictedLocalTriplesSpaceView, 20> spaces{};
    std::array<std::array<U,3>, 20> labels{};
    std::array<View, 20> connected{}, singles{};
    View foo, fvv;
    U o = 0, n = 0, count = 0, retained = 0, visits = 0;
    U fail_before = std::numeric_limits<U>::max();
    std::uint32_t fault = 0;
    void validate_descriptors() const {
        const auto check = [](const py::array& a, View original, int ndim) {
            (void)array(a, ndim);
            if (static_cast<std::size_t>(a.size()) != original.element_count
                || (a.size() && a.data() != original.data))
                throw std::invalid_argument("local triples diagnostic input array storage changed at callback");
        };
        check(owners[80], foo, 2); check(owners[81], fvv, 2);
        if (owners[80].shape(0) != static_cast<py::ssize_t>(o) || owners[80].shape(1) != static_cast<py::ssize_t>(o)
            || owners[81].shape(0) != static_cast<py::ssize_t>(n) || owners[81].shape(1) != static_cast<py::ssize_t>(n))
            throw std::invalid_argument("local triples diagnostic Fock shape changed at callback");
        for (U k = 0; k < count; ++k) {
            const auto r = static_cast<py::ssize_t>(spaces[k].rank);
            check(owners[4*k], spaces[k].coefficients, 2); check(owners[4*k+1], spaces[k].energies, 1);
            check(owners[4*k+2], connected[k], 3); check(owners[4*k+3], singles[k], 3);
            if (owners[4*k].shape(0) != static_cast<py::ssize_t>(n) || owners[4*k].shape(1) != r
                || owners[4*k+1].shape(0) != r)
                throw std::invalid_argument("local triples diagnostic space shape changed at callback");
            for (int axis = 0; axis < 3; ++axis)
                if (owners[4*k+2].shape(axis) != r || owners[4*k+3].shape(axis) != r)
                    throw std::invalid_argument("local triples diagnostic moment shape changed at callback");
        }
    }
    static void visit(const Request& request, Receiver receiver, void* receiver_context, void* context) {
        auto& self = *static_cast<Direct*>(context);
        if (self.visits == self.fail_before)
            throw std::runtime_error("local triples diagnostic injected moment provider failure");
        ++self.visits;
        if (self.fault == 1) return;
        U slot = 0;
        while (slot < self.count && self.labels[slot] != request.occupied) ++slot;
        if (slot == self.count || request.rank != self.spaces[slot].rank)
            throw std::logic_error("local triples diagnostic moment request is not a canonical stored space");
        BoundedRestrictedLocalTriplesMomentView data{request, self.connected[slot], self.singles[slot]};
        if (self.fault == 3 || self.fault == 5) ++data.request.ccsd_snapshot_id;
        if (self.fault == 4) data.request.occupied[0] = self.o;
        if (self.fault == 6) ++data.connected.element_count;
        if (self.fault == 7) ++data.request.rank;
        if (self.fault == 5) {
            try { receiver(data, receiver_context); } catch (...) {}
            return; // The native solver must reject swallowed receiver failures.
        }
        receiver(data, receiver_context);
        if (self.fault == 2) receiver(data, receiver_context);
    }
};
} // namespace bounded_local_triples_binding

void bind_bounded_restricted_local_triples_solver(py::module_& m) {
    using namespace vibeqc;
    using namespace bounded_local_triples_binding;
    using Options = BoundedRestrictedLocalTriplesSolverOptions;
    using Inventory = BoundedRestrictedLocalTriplesSolverInventory;
    using Caps = BoundedRestrictedLocalTriplesSolverCaps;
    using Plan = BoundedRestrictedLocalTriplesSolverMemoryPlan;
    using Result = BoundedRestrictedLocalTriplesSolverResult;
    auto options = py::class_<Options>(m, "_BoundedRestrictedLocalTriplesSolverOptions");
    options.def(py::init<>());
#define BLT_OPTION(f) options.def_readwrite(#f, &Options::f)
    BLT_OPTION(maximum_iterations); BLT_OPTION(denominator_floor); BLT_OPTION(residual_tolerance);
    BLT_OPTION(energy_tolerance); BLT_OPTION(coefficient_orthogonality_tolerance);
    BLT_OPTION(maximum_projected_fock_error); BLT_OPTION(maximum_repeated_moment_defect_norm);
    BLT_OPTION(maximum_repeated_update_defect_norm);
#undef BLT_OPTION
    auto inventory = py::class_<Inventory>(m, "_BoundedRestrictedLocalTriplesSolverInventory");
    inventory.def(py::init<>());
#define BLT_INV(f) inventory.def_readwrite(#f, &Inventory::f)
    BLT_INV(numerical_replicas); BLT_INV(external_node_bytes); BLT_INV(other_live_bytes_per_replica);
    BLT_INV(fixed_backend_margin_bytes_per_replica);
#undef BLT_INV
    auto caps = py::class_<Caps>(m, "_BoundedRestrictedLocalTriplesSolverCaps");
    caps.def(py::init<>());
#define BLT_CAP(f) caps.def_readwrite(#f, &Caps::f)
    BLT_CAP(maximum_occupied_count); BLT_CAP(maximum_common_virtual_dimension);
    BLT_CAP(maximum_triple_count); BLT_CAP(maximum_rank); BLT_CAP(maximum_owned_numerical_bytes);
    BLT_CAP(maximum_per_replica_inventoried_bytes); BLT_CAP(maximum_node_inventoried_bytes);
    BLT_CAP(maximum_moment_visits); BLT_CAP(maximum_work_units);
#undef BLT_CAP
    auto plan = py::class_<Plan>(m, "_BoundedRestrictedLocalTriplesSolverMemoryPlan");
#define BLT_PLAN(f) plan.def_readonly(#f, &Plan::f)
    BLT_PLAN(uniform_rank_upper_bound); BLT_PLAN(n_occupied); BLT_PLAN(common_virtual_dimension);
    BLT_PLAN(triple_count); BLT_PLAN(nonempty_triple_count); BLT_PLAN(maximum_rank); BLT_PLAN(maximum_iterations);
    BLT_PLAN(total_amplitude_elements); BLT_PLAN(total_rank); BLT_PLAN(amplitude_snapshot_bytes);
    BLT_PLAN(candidate_snapshot_bytes); BLT_PLAN(retained_record_bytes); BLT_PLAN(copied_space_table_bytes);
    BLT_PLAN(borrowed_space_table_bytes); BLT_PLAN(borrowed_numerical_bytes); BLT_PLAN(neighbour_workspace_bytes);
    BLT_PLAN(occupied_row_bytes); BLT_PLAN(target_owned_peak_bytes); BLT_PLAN(moment_digest_bytes);
    BLT_PLAN(output_numerical_bytes); BLT_PLAN(peak_owned_numerical_bytes); BLT_PLAN(control_storage_reservation_bytes);
    BLT_PLAN(per_replica_inventoried_bytes); BLT_PLAN(required_node_inventoried_bytes);
    BLT_PLAN(moment_visits_upper_bound); BLT_PLAN(neighbour_visits_upper_bound);
    BLT_PLAN(validation_work_units); BLT_PLAN(work_units_upper_bound);
#undef BLT_PLAN
    auto progress = py::class_<Progress>(m, "_BoundedRestrictedLocalTriplesSolverProgress");
#define BLT_PROGRESS(f) progress.def_readonly(#f, &Progress::f)
    BLT_PROGRESS(iteration); BLT_PROGRESS(moment_visits); BLT_PROGRESS(neighbour_visits);
    BLT_PROGRESS(has_previous_energy); BLT_PROGRESS(converged); BLT_PROGRESS(triples_energy);
    BLT_PROGRESS(energy_change); BLT_PROGRESS(maximum_absolute_residual); BLT_PROGRESS(residual_frobenius_norm);
    BLT_PROGRESS(repeated_update_defect_norm); BLT_PROGRESS(maximum_repeated_moment_defect_norm);
#undef BLT_PROGRESS
    py::class_<Result>(m, "_BoundedRestrictedLocalTriplesSolverResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("final_snapshot", [](const Result& r) { return r.final_snapshot(); })
        .def_property_readonly("minimum_denominator", &Result::minimum_denominator)
        .def_property_readonly("maximum_denominator", &Result::maximum_denominator)
        .def_property_readonly("input_identity_sha256", &Result::input_identity_sha256)
        .def_property_readonly("payload_sha256", &Result::payload_sha256)
        .def("rank", &Result::rank).def("amplitude", &Result::amplitude)
        .def("amplitudes_copy", [](const Result& r, U i, U j, U k) {
            const U rank = r.rank(i, j, k);
            if (rank > 4) throw std::length_error("local triples diagnostic result copy exceeds tiny rank");
            const auto dim = static_cast<py::ssize_t>(rank);
            py::array_t<double> out({dim, dim, dim});
            for (U a = 0; a < rank; ++a) for (U b = 0; b < rank; ++b) for (U c = 0; c < rank; ++c)
                out.mutable_data()[(a*rank+b)*rank+c] = r.amplitude(i, j, k, a, b, c);
            return out;
        });
    m.def("_plan_bounded_restricted_local_triples_solver_upper", [](U o, U n, U rank,
        const Options& selected, const Inventory& owners, U retained, U transient, U work) {
        const BoundedRestrictedLocalTriplesMomentProvider provider{nullptr, nullptr, retained, transient, work};
        return plan_bounded_restricted_local_triples_solver_upper(o, n, rank, provider, selected, owners);
    }, py::arg("n_occupied"), py::arg("common_virtual_dimension"), py::arg("maximum_rank"),
       py::arg("options"), py::arg("inventory"), py::arg("provider_retained_bytes"),
       py::arg("provider_transient_bytes"), py::arg("provider_work_units_per_visit"));
    const auto dispatch = [](py::array foo, py::array fvv, py::list coefficients, py::list energies,
        py::list connected, py::list singles, U snapshot, Options selected, Inventory owners, Caps limits,
        py::object callback, U fail_before, std::uint32_t fault, U transient, bool planning) -> py::object {
        foo = array(foo, 2); fvv = array(fvv, 2);
        if (foo.shape(0) < 1 || foo.shape(0) > 4 || fvv.shape(0) < 1 || fvv.shape(0) > 4
            || foo.shape(1) != foo.shape(0) || fvv.shape(1) != fvv.shape(0)
            || selected.maximum_iterations > 128 || owners.numerical_replicas > 8
            || limits.maximum_owned_numerical_bytes > (1U << 20)
            || limits.maximum_node_inventoried_bytes > (128U << 20)
            || limits.maximum_work_units > 1000000000000000ULL || transient > (1U << 20) || fault > 7)
            throw std::length_error("local triples diagnostic exceeds tiny dimensions, memory or work");
        Direct direct;
        direct.o = static_cast<U>(foo.shape(0)); direct.n = static_cast<U>(fvv.shape(0));
        direct.count = direct.o*(direct.o+1)*(direct.o+2)/6;
        direct.fail_before = fail_before; direct.fault = fault;
        if (coefficients.size() != direct.count || energies.size() != direct.count
            || connected.size() != direct.count || singles.size() != direct.count)
            throw std::invalid_argument("local triples diagnostic needs every lexicographic occupied multiset");
        direct.owners[80] = foo; direct.owners[81] = fvv; direct.foo = view(foo); direct.fvv = view(fvv);
        U index = 0;
        for (U i = 0; i < direct.o; ++i) for (U j = i; j < direct.o; ++j) for (U k = j; k < direct.o; ++k) {
            direct.labels[index] = {i,j,k};
            auto& c = direct.owners[4*index]; c = array(coefficients[index], 2);
            auto& e = direct.owners[4*index+1]; e = array(energies[index], 1);
            auto& w = direct.owners[4*index+2]; w = array(connected[index], 3);
            auto& u = direct.owners[4*index+3]; u = array(singles[index], 3);
            const auto rank = e.shape(0);
            if (rank < 0 || rank > static_cast<py::ssize_t>(direct.n)
                || c.shape(0) != static_cast<py::ssize_t>(direct.n) || c.shape(1) != rank)
                throw std::invalid_argument("local triples diagnostic space shape mismatch");
            for (int axis = 0; axis < 3; ++axis)
                if (w.shape(axis) != rank || u.shape(axis) != rank)
                    throw std::invalid_argument("local triples diagnostic moment shape mismatch");
            direct.spaces[index] = {static_cast<U>(rank), view(c), view(e)};
            direct.connected[index] = view(w); direct.singles[index] = view(u);
            direct.retained += static_cast<U>(w.nbytes()+u.nbytes());
            ++index;
        }
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("local triples diagnostic progress must be callable");
        // 20 comparisons and scalar control per request; no numeric arithmetic
        // or data copy in this provider. W/U are retained, not transient twice.
        const BoundedRestrictedLocalTriplesMomentProvider provider{&Direct::visit, &direct, direct.retained, transient, 256};
        const BoundedRestrictedLocalTriplesSolverInput input{direct.o, direct.n, direct.spaces.data(),
            static_cast<std::size_t>(direct.count), direct.foo, direct.fvv, snapshot};
        if (planning) return py::cast(plan_bounded_restricted_local_triples_solver(input, provider, selected, owners, limits));
        const auto plan = plan_bounded_restricted_local_triples_solver(input, provider, selected, owners, limits);
        if (plan.peak_owned_numerical_bytes > (1U << 20) || plan.required_node_inventoried_bytes > (128U << 20))
            throw std::length_error("local triples diagnostic admitted plan exceeds tiny memory");
        struct Sink {
            Direct* direct;
            py::object* callback;
            static void receive(const Progress& p, void* context) {
                auto& self = *static_cast<Sink*>(context);
                (*self.callback)(py::cast(Progress(p)));
                // Reject resize/reallocation before the native solver resumes
                // borrowed pointer reads. Numeric mutations are native gates.
                self.direct->validate_descriptors();
            }
        } sink{&direct, &callback};
        // Fixed tiny owner arrays pin every original allocation even if the
        // callback replaces list elements. Hold the GIL; no Python per-number
        // provider and no NumPy coercion or numerical copy is hidden here.
        return py::cast(bounded_restricted_local_triples_solve(input, provider, selected, owners, limits,
            callback.is_none() ? nullptr : &Sink::receive, &sink));
    };
    m.def("_plan_bounded_restricted_local_triples_solver", [dispatch](py::array foo, py::array fvv,
        py::list c, py::list e, py::list w, py::list u, U snapshot, Options o, Inventory i, Caps caps, U transient) {
        return dispatch(foo, fvv, c, e, w, u, snapshot, o, i, caps, py::none(), std::numeric_limits<U>::max(), 0, transient, true);
    }, py::arg("f_oo").noconvert(), py::arg("f_vv").noconvert(), py::arg("coefficients"), py::arg("energies"),
       py::arg("connected_moments"), py::arg("singles_moments"), py::arg("ccsd_snapshot_id"),
       py::arg("options"), py::arg("inventory"), py::arg("caps"), py::arg("provider_transient_bytes") = 0);
    m.def("_bounded_restricted_local_triples_solve_diagnostic", [dispatch](py::array foo, py::array fvv,
        py::list c, py::list e, py::list w, py::list u, U snapshot, Options o, Inventory i, Caps caps,
        py::object progress, U fail, std::uint32_t fault, U transient) {
        return dispatch(foo, fvv, c, e, w, u, snapshot, o, i, caps, progress, fail, fault, transient, false);
    }, py::arg("f_oo").noconvert(), py::arg("f_vv").noconvert(), py::arg("coefficients"), py::arg("energies"),
       py::arg("connected_moments"), py::arg("singles_moments"), py::arg("ccsd_snapshot_id"),
       py::arg("options"), py::arg("inventory"), py::arg("caps"), py::arg("progress") = py::none(),
       py::arg("fail_before_visit") = std::numeric_limits<U>::max(), py::arg("protocol_fault") = 0,
       py::arg("provider_transient_bytes") = 0);
}
