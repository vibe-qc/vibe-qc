// Included by bindings.cpp. Tiny borrowed-ERI diagnostics, no method route.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <cfenv>
#include <limits>
#include <stdexcept>

#include "vibeqc/bounded_restricted_triples_moments.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_triples_moments_accessor(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Plan = BoundedRestrictedTriplesMomentsAccessorMemoryPlan;
    using Caps = BoundedRestrictedTriplesMomentsAccessorCaps;
    using Result = BoundedRestrictedTriplesMomentsAccessorResult;
    py::class_<Plan>(m, "_BoundedRestrictedTriplesMomentsAccessorMemoryPlan")
#define TMA_PLAN(field) .def_readonly(#field, &Plan::field)
        TMA_PLAN(kernel) TMA_PLAN(amplitude_retained_numerical_bytes)
        TMA_PLAN(amplitude_maximum_transient_numerical_bytes) TMA_PLAN(singles_calls)
        TMA_PLAN(contraction_doubles_calls) TMA_PLAN(reverse_audit_doubles_calls)
        TMA_PLAN(doubles_calls) TMA_PLAN(amplitude_work_units_upper_bound)
        TMA_PLAN(fixed_control_storage_bytes);
#undef TMA_PLAN
    py::class_<Caps>(m, "_BoundedRestrictedTriplesMomentsAccessorCaps")
        .def(py::init<>())
#define TMA_CAP(field) .def_readwrite(#field, &Caps::field)
        TMA_CAP(kernel) TMA_CAP(maximum_singles_calls) TMA_CAP(maximum_doubles_calls)
        TMA_CAP(maximum_amplitude_work_units) TMA_CAP(maximum_control_storage_bytes);
#undef TMA_CAP
    py::class_<Result>(m, "_BoundedRestrictedTriplesMomentsAccessorResult")
#define TMA_RESULT(field) .def_readonly(#field, &Result::field)
        TMA_RESULT(memory) TMA_RESULT(moments) TMA_RESULT(singles_calls) TMA_RESULT(doubles_calls)
        TMA_RESULT(reverse_symmetry_audits) TMA_RESULT(charged_amplitude_work_units)
        TMA_RESULT(maximum_consumed_amplitude_symmetry_error);
#undef TMA_RESULT
    m.def("_plan_bounded_restricted_triples_moments_accessor", [](
        U o, U v, U amplitude_retained, U amplitude_transient, U singles_work, U doubles_work,
        U integral_retained, U integral_transient) {
        BoundedRestrictedCCSDAmplitudeProvider amplitudes;
        amplitudes.retained_numerical_bytes = amplitude_retained;
        amplitudes.maximum_transient_numerical_bytes = amplitude_transient;
        amplitudes.maximum_singles_work_units_per_query = singles_work;
        amplitudes.maximum_doubles_work_units_per_query = doubles_work;
        return plan_bounded_restricted_triples_moments_accessor(o, v, amplitudes, integral_retained, integral_transient);
    }, py::arg("n_occupied"), py::arg("n_virtual"),
       py::arg("amplitude_retained_numerical_bytes"), py::arg("amplitude_maximum_transient_numerical_bytes"),
       py::arg("maximum_singles_work_units_per_query"), py::arg("maximum_doubles_work_units_per_query"),
       py::arg("integral_retained_numerical_bytes") = 0, py::arg("integral_maximum_transient_numerical_bytes") = 0);
    // Nonphysical direct-array adapter only: no array-to-certified-state path.
    // Arrays are pinned and borrowed without a numerical copy or amplitude scan.
    m.def("_bounded_restricted_triples_moments_accessor_diagnostic", [](
        const py::array& supplied_t1, const py::array& supplied_t2, const py::array& supplied_fov,
        const py::array& supplied_integrals, const std::array<U, 3>& occupied, U snapshot, double tolerance,
        const Caps& supplied_caps, U singles_work, U doubles_work,
        U amplitude_transient, U integral_transient, U fail_singles, U fail_doubles, U fail_integral,
        unsigned disabled_callbacks, bool mutate_controls, unsigned floating_fault) {
        const py::array t1 = supplied_t1, t2 = supplied_t2, fov = supplied_fov, integrals = supplied_integrals;
        const auto array = [](const py::array& a, int ndim) {
            if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != ndim
                || !(a.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double))
                throw std::invalid_argument("bounded triples accessor diagnostic requires aligned C-contiguous binary64 arrays");
        };
        array(t1, 2); array(t2, 4); array(fov, 2); array(integrals, 4);
        const U o = static_cast<U>(t1.shape(0)), v = static_cast<U>(t1.shape(1)), n = o + v;
        if (!o || !v || o > 4 || v > 5 || n > 8
            || supplied_caps.kernel.maximum_owned_numerical_bytes > (1U << 20)
            || supplied_caps.kernel.maximum_total_numerical_bytes > (2U << 20)
            || supplied_caps.kernel.maximum_integral_calls > 1000000U
            || supplied_caps.kernel.maximum_kernel_work_units > 200000000U
            || supplied_caps.maximum_singles_calls > 1000000U || supplied_caps.maximum_doubles_calls > 1000000U
            || supplied_caps.maximum_amplitude_work_units > 1000000000U
            || supplied_caps.maximum_control_storage_bytes > 65536U
            || singles_work > 1000000U || doubles_work > 1000000U
            || amplitude_transient > (1U << 20) || integral_transient > (1U << 20)
            || disabled_callbacks > 7 || floating_fault > 4)
            throw std::length_error("bounded triples accessor diagnostic exceeds tiny extent or work limits");
        const auto dim = [](const py::array& a, int axis, U value) {
            if (static_cast<U>(a.shape(axis)) != value)
                throw std::invalid_argument("bounded triples accessor diagnostic input shape mismatch");
        };
        dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
        dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(integrals, axis, n);
        BoundedRestrictedTriplesMomentsOperatorInput op{o, v, occupied, snapshot,
            {static_cast<const double*>(fov.data()), static_cast<std::size_t>(fov.size())}, tolerance};
        Caps caps = supplied_caps;
        BoundedRestrictedCCSDAmplitudeProvider amplitudes;
        BoundedRestrictedCCSDIntegralProvider provider;
        struct Direct {
            const double* t1; const double* t2; const double* eri;
            U o, v, n, fail_singles, fail_doubles, fail_integral;
            bool mutate;
            unsigned floating_fault;
            BoundedRestrictedTriplesMomentsOperatorInput* op;
            Caps* caps;
            BoundedRestrictedCCSDAmplitudeProvider* amplitudes;
            BoundedRestrictedCCSDIntegralProvider* provider;
            mutable U singles_calls = 0, doubles_calls = 0, integral_calls = 0;
            void controls(unsigned lane) const {
                if (mutate) {
                    *op = {}; *caps = {}; *amplitudes = {}; *provider = {};
                }
                if (floating_fault == lane && std::fesetround(FE_UPWARD) != 0)
                    throw std::runtime_error("triples diagnostic cannot set floating environment fault");
            }
            static double singles(U i, U a, const void* context) {
                const auto& d = *static_cast<const Direct*>(context);
                if (d.singles_calls == d.fail_singles)
                    throw std::runtime_error("bounded triples diagnostic injected singles callback failure");
                ++d.singles_calls;
                if (i >= d.o || a >= d.v) throw std::logic_error("invalid triples singles index");
                d.controls(1);
                return d.t1[i * d.v + a];
            }
            static double doubles(U i, U j, U a, U b, const void* context) {
                const auto& d = *static_cast<const Direct*>(context);
                if (d.doubles_calls == d.fail_doubles)
                    throw std::runtime_error("bounded triples diagnostic injected doubles callback failure");
                ++d.doubles_calls;
                if (i >= d.o || j >= d.o || a >= d.v || b >= d.v)
                    throw std::logic_error("invalid triples doubles index");
                d.controls(2);
                return d.t2[((i * d.o + j) * d.v + a) * d.v + b];
            }
            static double integral(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                if (d.integral_calls == d.fail_integral)
                    throw std::runtime_error("bounded triples diagnostic injected integral callback failure");
                ++d.integral_calls;
                if (p >= d.n || q >= d.n || r >= d.n || s >= d.n)
                    throw std::logic_error("invalid triples integral index");
                d.controls(3);
                return d.eri[((p * d.n + q) * d.n + r) * d.n + s];
            }
        } direct{static_cast<const double*>(t1.data()), static_cast<const double*>(t2.data()),
            static_cast<const double*>(integrals.data()), o, v, n, fail_singles, fail_doubles, fail_integral,
            mutate_controls, floating_fault, &op, &caps, &amplitudes, &provider};
        amplitudes = {disabled_callbacks & 1 ? nullptr : &Direct::singles,
            disabled_callbacks & 2 ? nullptr : &Direct::doubles, &direct,
            static_cast<U>(t1.nbytes() + t2.nbytes()), amplitude_transient, singles_work, doubles_work};
        provider = {disabled_callbacks & 4 ? nullptr : &Direct::integral, &direct,
            static_cast<U>(integrals.nbytes()), integral_transient};
        struct RestoreEnvironment {
            std::fenv_t saved;
            RestoreEnvironment() {
                if (std::fegetenv(&saved) != 0)
                    throw std::runtime_error("triples diagnostic cannot save floating environment");
            }
            ~RestoreEnvironment() { std::fesetenv(&saved); }
        } restore;
        if (floating_fault == 4 && std::fesetround(FE_UPWARD) != 0)
            throw std::runtime_error("triples diagnostic cannot set floating environment fault");
        py::gil_scoped_release release;
        return bounded_restricted_triples_moments_accessor(op, amplitudes, provider, caps);
    }, py::arg("t1").noconvert(), py::arg("t2").noconvert(), py::arg("f_ov").noconvert(),
       py::arg("integrals").noconvert(), py::arg("occupied"), py::arg("amplitude_snapshot_id"),
       py::arg("amplitude_symmetry_tolerance"), py::arg("caps"),
       py::arg("singles_work_units") = 1, py::arg("doubles_work_units") = 1,
       py::arg("amplitude_transient_bytes") = 0, py::arg("integral_transient_bytes") = 0,
       py::arg("fail_before_singles") = std::numeric_limits<U>::max(),
       py::arg("fail_before_doubles") = std::numeric_limits<U>::max(),
       py::arg("fail_before_integral") = std::numeric_limits<U>::max(),
       py::arg("disabled_callbacks") = 0, py::arg("mutate_control_objects") = false,
       py::arg("floating_environment_fault") = 0);
}

void bind_bounded_restricted_triples_moments(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Plan = BoundedRestrictedTriplesMomentsMemoryPlan;
    using Caps = BoundedRestrictedTriplesMomentsCaps;
    using Result = BoundedRestrictedTriplesMomentsResult;
    py::class_<Plan>(m, "_BoundedRestrictedTriplesMomentsMemoryPlan")
#define TRIPLES_MOMENTS_PLAN(field) .def_readonly(#field, &Plan::field)
        TRIPLES_MOMENTS_PLAN(n_occupied) TRIPLES_MOMENTS_PLAN(n_virtual)
        TRIPLES_MOMENTS_PLAN(borrowed_input_bytes) TRIPLES_MOMENTS_PLAN(provider_retained_numerical_bytes)
        TRIPLES_MOMENTS_PLAN(provider_maximum_transient_numerical_bytes)
        TRIPLES_MOMENTS_PLAN(connected_output_bytes) TRIPLES_MOMENTS_PLAN(singles_output_bytes)
        TRIPLES_MOMENTS_PLAN(peak_owned_numerical_bytes) TRIPLES_MOMENTS_PLAN(total_live_numerical_bytes)
        TRIPLES_MOMENTS_PLAN(integral_calls) TRIPLES_MOMENTS_PLAN(kernel_work_units_upper_bound);
#undef TRIPLES_MOMENTS_PLAN
    py::class_<Caps>(m, "_BoundedRestrictedTriplesMomentsCaps")
        .def(py::init<>())
#define TRIPLES_MOMENTS_CAP(field) .def_readwrite(#field, &Caps::field)
        TRIPLES_MOMENTS_CAP(maximum_owned_numerical_bytes) TRIPLES_MOMENTS_CAP(maximum_total_numerical_bytes)
        TRIPLES_MOMENTS_CAP(maximum_integral_calls) TRIPLES_MOMENTS_CAP(maximum_kernel_work_units);
#undef TRIPLES_MOMENTS_CAP
    const auto copy = [](const Result& r, bool connected) {
        const U v = r.memory.n_virtual;
        const auto& values = connected ? r.connected : r.singles;
        if (!v || v > 5 || values.size() != v * v * v)
            throw std::length_error("bounded triples moments copy exceeds tiny extent");
        py::array_t<double> out({static_cast<py::ssize_t>(v), static_cast<py::ssize_t>(v), static_cast<py::ssize_t>(v)});
        std::memcpy(out.mutable_data(), values.data(), values.size() * 8);
        return out;
    };
    py::class_<Result>(m, "_BoundedRestrictedTriplesMomentsResult")
        .def_readonly("memory", &Result::memory)
        .def_readonly("occupied", &Result::occupied)
        .def_readonly("amplitude_snapshot_id", &Result::amplitude_snapshot_id)
        .def_readonly("integral_calls", &Result::integral_calls)
        .def_property_readonly("connected", [copy](const Result& r) { return copy(r, true); })
        .def_property_readonly("singles", [copy](const Result& r) { return copy(r, false); });
    m.def("_plan_bounded_restricted_triples_moments", &plan_bounded_restricted_triples_moments,
        py::arg("n_occupied"), py::arg("n_virtual"),
        py::arg("provider_retained_numerical_bytes") = 0,
        py::arg("provider_maximum_transient_numerical_bytes") = 0);
    m.def("_bounded_restricted_triples_moments_diagnostic", [](
        const py::array& t1, const py::array& t2, const py::array& fov, const py::array& integrals,
        const std::array<U, 3>& occupied, U snapshot, double tolerance, const Caps& caps,
        U fail_before_call, U provider_transient_bytes) {
        const auto array = [](const py::array& a, int ndim) {
            if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != ndim
                || !(a.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double))
                throw std::invalid_argument("bounded triples moments diagnostic requires aligned C-contiguous binary64 arrays");
        };
        array(t1, 2); array(t2, 4); array(fov, 2); array(integrals, 4);
        const U o = static_cast<U>(t1.shape(0)), v = static_cast<U>(t1.shape(1)), n = o + v;
        if (!o || !v || o > 4 || v > 5 || n > 8
            || caps.maximum_owned_numerical_bytes > (1U << 20)
            || caps.maximum_total_numerical_bytes > (2U << 20)
            || caps.maximum_integral_calls > 1000000U
            || caps.maximum_kernel_work_units > 200000000U
            || provider_transient_bytes > (1U << 20))
            throw std::length_error("bounded triples moments diagnostic exceeds tiny shape or work limits");
        const auto dim = [](const py::array& a, int axis, U value) {
            if (static_cast<U>(a.shape(axis)) != value)
                throw std::invalid_argument("bounded triples moments diagnostic input shape mismatch");
        };
        dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
        dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(integrals, axis, n);
        const auto view = [](const py::array& a) {
            return BoundedRestrictedCCSDRealView{static_cast<const double*>(a.data()), static_cast<std::size_t>(a.size())};
        };
        const BoundedRestrictedTriplesMomentsInput input{o, v, occupied, snapshot, view(t1), view(t2), view(fov), tolerance};
        struct Direct {
            const double* data;
            U n, calls = 0, fail_before;
            static double value(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                if (d.calls == d.fail_before)
                    throw std::runtime_error("bounded triples moments diagnostic injected integral callback failure");
                ++d.calls;
                if (p >= d.n || q >= d.n || r >= d.n || s >= d.n)
                    throw std::logic_error("bounded triples moments diagnostic invalid integral index");
                return d.data[((p * d.n + q) * d.n + r) * d.n + s];
            }
        } direct{static_cast<const double*>(integrals.data()), n, 0, fail_before_call};
        const BoundedRestrictedCCSDIntegralProvider provider{&Direct::value, &direct,
            static_cast<U>(integrals.nbytes()), provider_transient_bytes};
        const Caps limits = caps;
        py::gil_scoped_release release;
        return bounded_restricted_triples_moments(input, provider, limits);
    }, py::arg("t1").noconvert(), py::arg("t2").noconvert(), py::arg("f_ov").noconvert(),
       py::arg("integrals").noconvert(), py::arg("occupied"), py::arg("amplitude_snapshot_id"),
       py::arg("amplitude_symmetry_tolerance"), py::arg("caps"),
       py::arg("fail_before_call") = std::numeric_limits<U>::max(), py::arg("provider_transient_bytes") = 0);
    bind_bounded_restricted_triples_moments_accessor(m);
}
