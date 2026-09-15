// Included by bindings.cpp. Tiny numerical diagnostics, no physical driver.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/bounded_restricted_ccsd_target.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_ccsd_target(py::module_& m) {
    using Plan = vibeqc::BoundedRestrictedCCSDTargetMemoryPlan;
    using Caps = vibeqc::BoundedRestrictedCCSDTargetCaps;
    using Result = vibeqc::BoundedRestrictedCCSDTargetResult;
    using AccessorPlan = vibeqc::BoundedRestrictedCCSDTargetAccessorMemoryPlan;
    using AccessorCaps = vibeqc::BoundedRestrictedCCSDTargetAccessorCaps;
    using AccessorResult = vibeqc::BoundedRestrictedCCSDTargetAccessorResult;
    py::class_<Plan>(m, "_BoundedRestrictedCCSDTargetMemoryPlan")
#define BOUNDED_CCSD_PLAN(field) .def_readonly(#field, &Plan::field)
        BOUNDED_CCSD_PLAN(n_occupied) BOUNDED_CCSD_PLAN(n_virtual)
        BOUNDED_CCSD_PLAN(borrowed_input_bytes)
        BOUNDED_CCSD_PLAN(provider_retained_numerical_bytes)
        BOUNDED_CCSD_PLAN(provider_maximum_transient_numerical_bytes)
        BOUNDED_CCSD_PLAN(output_bytes) BOUNDED_CCSD_PLAN(workspace_bytes)
        BOUNDED_CCSD_PLAN(peak_owned_numerical_bytes)
        BOUNDED_CCSD_PLAN(total_live_numerical_bytes)
        BOUNDED_CCSD_PLAN(integral_calls_upper_bound)
        BOUNDED_CCSD_PLAN(kernel_work_units_upper_bound)
        BOUNDED_CCSD_PLAN(bare_particle_hole_excluded)
        BOUNDED_CCSD_PLAN(omitted_bare_integral_calls) BOUNDED_CCSD_PLAN(omitted_bare_kernel_work_units);
#undef BOUNDED_CCSD_PLAN
    py::class_<Caps>(m, "_BoundedRestrictedCCSDTargetCaps")
        .def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_total_numerical_bytes", &Caps::maximum_total_numerical_bytes)
        .def_readwrite("maximum_integral_calls", &Caps::maximum_integral_calls)
        .def_readwrite("maximum_kernel_work_units", &Caps::maximum_kernel_work_units);
    py::class_<Result>(m, "_BoundedRestrictedCCSDTargetResult")
        .def_readonly("memory", &Result::memory)
        .def_readonly("target_i", &Result::target_i)
        .def_readonly("target_j", &Result::target_j)
        .def_readonly("integral_calls", &Result::integral_calls)
        .def_property_readonly("complete_target_residual", &Result::complete_target_residual)
        .def_property_readonly("singles", [](const Result& result) {
            if (result.memory.n_virtual > 8 || result.singles.size() != result.memory.n_virtual)
                throw std::length_error("bounded CCSD target copy exceeds tiny extent");
            py::array_t<double> copy(static_cast<py::ssize_t>(result.singles.size()));
            std::memcpy(copy.mutable_data(), result.singles.data(), result.singles.size() * sizeof(double));
            return copy;
        })
        .def_property_readonly("doubles", [](const Result& result) {
            const auto v = result.memory.n_virtual;
            if (v > 8 || result.doubles.size() != v * v)
                throw std::length_error("bounded CCSD target copy exceeds tiny extent");
            py::array_t<double> copy({static_cast<py::ssize_t>(v), static_cast<py::ssize_t>(v)});
            std::memcpy(copy.mutable_data(), result.doubles.data(), result.doubles.size() * sizeof(double));
            return copy;
        });
    m.def("_plan_bounded_restricted_ccsd_target", &vibeqc::plan_bounded_restricted_ccsd_target,
        py::arg("n_occupied"), py::arg("n_virtual"),
        py::arg("provider_retained_numerical_bytes") = 0,
        py::arg("provider_maximum_transient_numerical_bytes") = 0);
    m.def("_plan_bounded_restricted_ccsd_target_without_bare_particle_hole",
        &vibeqc::plan_bounded_restricted_ccsd_target_without_bare_particle_hole,
        py::arg("n_occupied"), py::arg("n_virtual"),
        py::arg("provider_retained_numerical_bytes") = 0,
        py::arg("provider_maximum_transient_numerical_bytes") = 0);

    py::class_<AccessorPlan>(m, "_BoundedRestrictedCCSDTargetAccessorMemoryPlan")
#define BOUNDED_CCSD_ACCESSOR_PLAN(field) .def_readonly(#field, &AccessorPlan::field)
        BOUNDED_CCSD_ACCESSOR_PLAN(kernel)
        BOUNDED_CCSD_ACCESSOR_PLAN(amplitude_retained_numerical_bytes)
        BOUNDED_CCSD_ACCESSOR_PLAN(amplitude_maximum_transient_numerical_bytes)
        BOUNDED_CCSD_ACCESSOR_PLAN(singles_calls_upper_bound)
        BOUNDED_CCSD_ACCESSOR_PLAN(doubles_calls_upper_bound)
        BOUNDED_CCSD_ACCESSOR_PLAN(amplitude_work_units_upper_bound);
#undef BOUNDED_CCSD_ACCESSOR_PLAN
    py::class_<AccessorCaps>(m, "_BoundedRestrictedCCSDTargetAccessorCaps")
        .def(py::init<>())
        .def_readwrite("kernel", &AccessorCaps::kernel)
        .def_readwrite("maximum_singles_calls", &AccessorCaps::maximum_singles_calls)
        .def_readwrite("maximum_doubles_calls", &AccessorCaps::maximum_doubles_calls)
        .def_readwrite("maximum_amplitude_work_units", &AccessorCaps::maximum_amplitude_work_units);
    py::class_<AccessorResult>(m, "_BoundedRestrictedCCSDTargetAccessorResult")
        .def_readonly("memory", &AccessorResult::memory)
        .def_readonly("target", &AccessorResult::target)
        .def_readonly("singles_calls", &AccessorResult::singles_calls)
        .def_readonly("doubles_calls", &AccessorResult::doubles_calls)
        .def_readonly("charged_amplitude_work_units", &AccessorResult::charged_amplitude_work_units);
    for (const bool excluded : {false, true}) {
    m.def(excluded ? "_plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole"
                   : "_plan_bounded_restricted_ccsd_target_accessor", [excluded](
        std::uint64_t o, std::uint64_t v, std::uint64_t amplitude_retained,
        std::uint64_t amplitude_transient, std::uint64_t singles_work,
        std::uint64_t doubles_work, std::uint64_t integral_retained,
        std::uint64_t integral_transient) {
        vibeqc::BoundedRestrictedCCSDAmplitudeProvider amplitudes;
        amplitudes.retained_numerical_bytes = amplitude_retained;
        amplitudes.maximum_transient_numerical_bytes = amplitude_transient;
        amplitudes.maximum_singles_work_units_per_query = singles_work;
        amplitudes.maximum_doubles_work_units_per_query = doubles_work;
        if (excluded) return vibeqc::plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole(
            o, v, amplitudes, integral_retained, integral_transient);
        return vibeqc::plan_bounded_restricted_ccsd_target_accessor(
            o, v, amplitudes, integral_retained, integral_transient);
    }, py::arg("n_occupied"), py::arg("n_virtual"),
       py::arg("amplitude_retained_numerical_bytes"),
       py::arg("amplitude_maximum_transient_numerical_bytes"),
       py::arg("maximum_singles_work_units_per_query"),
       py::arg("maximum_doubles_work_units_per_query"),
       py::arg("integral_retained_numerical_bytes") = 0,
       py::arg("integral_maximum_transient_numerical_bytes") = 0);
    }

    // Native direct-ERI adapter avoids Python callback overhead and never
    // builds/copies an integral tensor. This is strictly a tiny diagnostic;
    // caller-supplied numbers do not acquire physical provenance here.
    for (const bool excluded : {false, true}) {
    m.def(excluded ? "_bounded_restricted_ccsd_target_residual_without_bare_particle_hole_diagnostic"
                   : "_bounded_restricted_ccsd_target_residual_diagnostic", [excluded](
        py::array t1, py::array t2, py::array foo, py::array fvv,
        py::array fov, py::array integrals, std::uint64_t target_i,
        std::uint64_t target_j, const Caps& caps, std::uint64_t fail_before_call,
        std::uint64_t provider_transient_bytes) {
        using U = std::uint64_t;
        const auto require_array = [](const py::array& array, int dimensions) {
            if (!array.dtype().is(py::dtype::of<double>())
                || array.ndim() != dimensions || !(array.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
                throw std::invalid_argument("bounded CCSD diagnostic requires aligned C-contiguous binary64 arrays");
        };
        require_array(t1, 2); require_array(t2, 4); require_array(foo, 2);
        require_array(fvv, 2); require_array(fov, 2); require_array(integrals, 4);
        const U o = static_cast<U>(t1.shape(0)), v = static_cast<U>(t1.shape(1));
        if (!o || !v || o > 4 || v > 8 || o + v > 10
            || caps.maximum_owned_numerical_bytes > (1U << 20)
            || caps.maximum_total_numerical_bytes > (2U << 20)
            || caps.maximum_integral_calls > 10000000U
            || caps.maximum_kernel_work_units > 2000000000U
            || provider_transient_bytes > (1U << 20))
            throw std::length_error("bounded CCSD diagnostic exceeds tiny extent or work limits");
        const auto dim = [](const py::array& array, int axis, U expected) {
            if (static_cast<U>(array.shape(axis)) != expected)
                throw std::invalid_argument("bounded CCSD diagnostic input shape mismatch");
        };
        dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
        dim(foo, 0, o); dim(foo, 1, o); dim(fvv, 0, v); dim(fvv, 1, v);
        dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(integrals, axis, o + v);
        const auto view = [](const py::array& array) {
            return vibeqc::BoundedRestrictedCCSDRealView{
                static_cast<const double*>(array.data()), static_cast<std::size_t>(array.size())};
        };
        vibeqc::BoundedRestrictedCCSDTargetInput input{
            o, v, target_i, target_j, view(t1), view(t2), view(foo), view(fvv), view(fov)};
        struct Direct {
            const double* data;
            U dimension, calls = 0, fail_before;
            static double value(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                if (d.calls == d.fail_before)
                    throw std::runtime_error("bounded CCSD diagnostic injected integral callback failure");
                ++d.calls;
                if (p >= d.dimension || q >= d.dimension || r >= d.dimension || s >= d.dimension)
                    throw std::logic_error("bounded CCSD integral callback received an invalid orbital label");
                return d.data[((p * d.dimension + q) * d.dimension + r) * d.dimension + s];
            }
        } direct{static_cast<const double*>(integrals.data()), o + v, 0, fail_before_call};
        vibeqc::BoundedRestrictedCCSDIntegralProvider provider{
            &Direct::value, &direct, static_cast<U>(integrals.nbytes()), provider_transient_bytes};
        py::gil_scoped_release release;
        if (excluded) return vibeqc::bounded_restricted_ccsd_target_residual_without_bare_particle_hole(input, provider, caps);
        return vibeqc::bounded_restricted_ccsd_target_residual(input, provider, caps);
    }, py::arg("t1"), py::arg("t2"), py::arg("f_oo"), py::arg("f_vv"),
       py::arg("f_ov"), py::arg("integrals"), py::arg("target_i"), py::arg("target_j"),
       py::arg("caps"), py::arg("fail_before_call") = std::numeric_limits<std::uint64_t>::max(),
       py::arg("provider_transient_bytes") = 0);
    }

    // Deliberately nonphysical direct-array amplitude accessor. Arrays remain
    // borrowed/pinned and are not copied or pre-scanned. Failure injection and
    // local descriptor mutation exercise callback contracts, not provenance.
    for (const bool excluded : {false, true}) {
    m.def(excluded ? "_bounded_restricted_ccsd_target_residual_accessor_without_bare_particle_hole_diagnostic"
                   : "_bounded_restricted_ccsd_target_residual_accessor_diagnostic", [excluded](
        py::array t1, py::array t2, py::array foo, py::array fvv, py::array fov,
        py::array integrals, std::uint64_t target_i, std::uint64_t target_j,
        const AccessorCaps& supplied_caps, std::uint64_t singles_work,
        std::uint64_t doubles_work, std::uint64_t amplitude_transient_bytes,
        std::uint64_t integral_transient_bytes, std::uint64_t fail_before_singles,
        std::uint64_t fail_before_doubles, std::uint64_t fail_before_integral,
        unsigned disabled_amplitude_callbacks, bool mutate_control_objects) {
        using U = std::uint64_t;
        const auto require_array = [](const py::array& array, int dimensions) {
            if (!array.dtype().is(py::dtype::of<double>())
                || array.ndim() != dimensions || !(array.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
                throw std::invalid_argument("bounded CCSD accessor diagnostic requires aligned C-contiguous binary64 arrays");
        };
        require_array(t1, 2); require_array(t2, 4); require_array(foo, 2);
        require_array(fvv, 2); require_array(fov, 2); require_array(integrals, 4);
        const U o = static_cast<U>(t1.shape(0)), v = static_cast<U>(t1.shape(1));
        if (!o || !v || o > 4 || v > 8 || o + v > 10
            || supplied_caps.kernel.maximum_owned_numerical_bytes > (1U << 20)
            || supplied_caps.kernel.maximum_total_numerical_bytes > (2U << 20)
            || supplied_caps.kernel.maximum_integral_calls > 10000000U
            || supplied_caps.kernel.maximum_kernel_work_units > 2000000000U
            || supplied_caps.maximum_singles_calls > 10000000U
            || supplied_caps.maximum_doubles_calls > 10000000U
            || supplied_caps.maximum_amplitude_work_units > 1000000000000ULL
            || amplitude_transient_bytes > (1U << 20) || integral_transient_bytes > (1U << 20)
            || singles_work > 1000000U || doubles_work > 1000000U
            || disabled_amplitude_callbacks > 3)
            throw std::length_error("bounded CCSD accessor diagnostic exceeds tiny extent or work limits");
        const auto dim = [](const py::array& array, int axis, U expected) {
            if (static_cast<U>(array.shape(axis)) != expected)
                throw std::invalid_argument("bounded CCSD accessor diagnostic input shape mismatch");
        };
        dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
        dim(foo, 0, o); dim(foo, 1, o); dim(fvv, 0, v); dim(fvv, 1, v);
        dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(integrals, axis, o + v);
        const auto view = [](const py::array& array) {
            return vibeqc::BoundedRestrictedCCSDRealView{
                static_cast<const double*>(array.data()), static_cast<std::size_t>(array.size())};
        };
        vibeqc::BoundedRestrictedCCSDTargetOperatorInput op{
            o, v, target_i, target_j, view(foo), view(fvv), view(fov)};
        auto caps = supplied_caps;
        vibeqc::BoundedRestrictedCCSDAmplitudeProvider amplitudes;
        vibeqc::BoundedRestrictedCCSDIntegralProvider provider;
        struct Direct {
            const double *t1, *t2, *integrals;
            U o, v, n, fail_singles, fail_doubles, fail_integral;
            bool mutate;
            vibeqc::BoundedRestrictedCCSDTargetOperatorInput* op;
            AccessorCaps* caps;
            vibeqc::BoundedRestrictedCCSDAmplitudeProvider* amplitudes;
            vibeqc::BoundedRestrictedCCSDIntegralProvider* provider;
            mutable U singles_calls = 0, doubles_calls = 0, integral_calls = 0;
            void mutate_descriptors() const {
                if (!mutate) return;
                *op = {}; *caps = {}; *amplitudes = {}; *provider = {};
            }
            static double singles(U i, U a, const void* context) {
                const auto& d = *static_cast<const Direct*>(context);
                if (d.singles_calls == d.fail_singles)
                    throw std::runtime_error("bounded CCSD diagnostic injected singles callback failure");
                ++d.singles_calls;
                if (i >= d.o || a >= d.v) throw std::logic_error("invalid singles label");
                d.mutate_descriptors();
                return d.t1[i * d.v + a];
            }
            static double doubles(U i, U j, U a, U b, const void* context) {
                const auto& d = *static_cast<const Direct*>(context);
                if (d.doubles_calls == d.fail_doubles)
                    throw std::runtime_error("bounded CCSD diagnostic injected doubles callback failure");
                ++d.doubles_calls;
                if (i >= d.o || j >= d.o || a >= d.v || b >= d.v)
                    throw std::logic_error("invalid doubles label");
                d.mutate_descriptors();
                return d.t2[((i * d.o + j) * d.v + a) * d.v + b];
            }
            static double integral(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                if (d.integral_calls == d.fail_integral)
                    throw std::runtime_error("bounded CCSD diagnostic injected integral callback failure");
                ++d.integral_calls;
                if (p >= d.n || q >= d.n || r >= d.n || s >= d.n)
                    throw std::logic_error("invalid integral label");
                d.mutate_descriptors();
                return d.integrals[((p * d.n + q) * d.n + r) * d.n + s];
            }
        } direct{static_cast<const double*>(t1.data()), static_cast<const double*>(t2.data()),
            static_cast<const double*>(integrals.data()), o, v, o + v,
            fail_before_singles, fail_before_doubles, fail_before_integral, mutate_control_objects,
            &op, &caps, &amplitudes, &provider};
        amplitudes = {disabled_amplitude_callbacks & 1 ? nullptr : &Direct::singles,
            disabled_amplitude_callbacks & 2 ? nullptr : &Direct::doubles, &direct,
            static_cast<U>(t1.nbytes() + t2.nbytes()), amplitude_transient_bytes, singles_work, doubles_work};
        provider = {&Direct::integral, &direct, static_cast<U>(integrals.nbytes()), integral_transient_bytes};
        py::gil_scoped_release release;
        if (excluded) return vibeqc::bounded_restricted_ccsd_target_residual_accessor_without_bare_particle_hole(op, amplitudes, provider, caps);
        return vibeqc::bounded_restricted_ccsd_target_residual_accessor(op, amplitudes, provider, caps);
    }, py::arg("t1"), py::arg("t2"), py::arg("f_oo"), py::arg("f_vv"), py::arg("f_ov"),
       py::arg("integrals"), py::arg("target_i"), py::arg("target_j"), py::arg("caps"),
       py::arg("singles_work_units") = 1, py::arg("doubles_work_units") = 1,
       py::arg("amplitude_transient_bytes") = 0, py::arg("integral_transient_bytes") = 0,
       py::arg("fail_before_singles") = std::numeric_limits<std::uint64_t>::max(),
       py::arg("fail_before_doubles") = std::numeric_limits<std::uint64_t>::max(),
       py::arg("fail_before_integral") = std::numeric_limits<std::uint64_t>::max(),
       py::arg("disabled_amplitude_callbacks") = 0, py::arg("mutate_control_objects") = false);
    }
}
