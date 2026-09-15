// Included by bindings.cpp. Tiny borrowed-array numerical diagnostic only.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <limits>
#include <stdexcept>
#include <string>
#include "vibeqc/bounded_restricted_ccsd_energy.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_ccsd_energy(py::module_& m) {
    using U = std::uint64_t;
    using Plan = vibeqc::BoundedRestrictedCCSDEnergyMemoryPlan;
    using Caps = vibeqc::BoundedRestrictedCCSDEnergyCaps;
    using Inventory = vibeqc::BoundedRestrictedCCSDEnergyInventory;
    using Result = vibeqc::BoundedRestrictedCCSDEnergyResult;
    py::class_<Inventory>(m, "_BoundedRestrictedCCSDEnergyInventory")
        .def(py::init<>())
        .def_readwrite("other_live_numerical_bytes", &Inventory::other_live_numerical_bytes)
        .def_readwrite("other_live_control_bytes", &Inventory::other_live_control_bytes)
        .def_readwrite("maximum_integral_work_units_per_query", &Inventory::maximum_integral_work_units_per_query);
    py::class_<Caps>(m, "_BoundedRestrictedCCSDEnergyCaps")
        .def(py::init<>())
#define CCSD_ENERGY_CAP(field) .def_readwrite(#field, &Caps::field)
        CCSD_ENERGY_CAP(maximum_total_numerical_bytes) CCSD_ENERGY_CAP(maximum_control_storage_bytes)
        CCSD_ENERGY_CAP(maximum_singles_calls) CCSD_ENERGY_CAP(maximum_doubles_calls)
        CCSD_ENERGY_CAP(maximum_integral_calls) CCSD_ENERGY_CAP(maximum_total_work_units);
#undef CCSD_ENERGY_CAP
    py::class_<Plan>(m, "_BoundedRestrictedCCSDEnergyMemoryPlan")
#define CCSD_ENERGY_PLAN(field) .def_readonly(#field, &Plan::field)
        CCSD_ENERGY_PLAN(n_occupied) CCSD_ENERGY_PLAN(n_virtual) CCSD_ENERGY_PLAN(borrowed_f_ov_bytes)
        CCSD_ENERGY_PLAN(amplitude_retained_numerical_bytes) CCSD_ENERGY_PLAN(amplitude_maximum_transient_numerical_bytes)
        CCSD_ENERGY_PLAN(integral_retained_numerical_bytes) CCSD_ENERGY_PLAN(integral_maximum_transient_numerical_bytes)
        CCSD_ENERGY_PLAN(other_live_numerical_bytes) CCSD_ENERGY_PLAN(peak_owned_numerical_bytes)
        CCSD_ENERGY_PLAN(total_live_numerical_bytes) CCSD_ENERGY_PLAN(fixed_inventoried_object_bytes)
        CCSD_ENERGY_PLAN(other_live_control_bytes) CCSD_ENERGY_PLAN(total_control_storage_bytes)
        CCSD_ENERGY_PLAN(singles_calls) CCSD_ENERGY_PLAN(doubles_calls) CCSD_ENERGY_PLAN(integral_calls)
        CCSD_ENERGY_PLAN(kernel_work_units_upper_bound) CCSD_ENERGY_PLAN(amplitude_work_units_upper_bound)
        CCSD_ENERGY_PLAN(integral_work_units_upper_bound) CCSD_ENERGY_PLAN(total_work_units_upper_bound);
#undef CCSD_ENERGY_PLAN
    py::class_<Result>(m, "_BoundedRestrictedCCSDEnergyResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory; })
#define CCSD_ENERGY_RESULT(field) .def_readonly(#field, &Result::field)
        CCSD_ENERGY_RESULT(correlation_energy) CCSD_ENERGY_RESULT(singles_fock_energy)
        CCSD_ENERGY_RESULT(doubles_and_disconnected_energy) CCSD_ENERGY_RESULT(singles_calls)
        CCSD_ENERGY_RESULT(doubles_calls) CCSD_ENERGY_RESULT(integral_calls)
        CCSD_ENERGY_RESULT(charged_amplitude_work_units) CCSD_ENERGY_RESULT(charged_integral_work_units)
        CCSD_ENERGY_RESULT(charged_total_work_units);
#undef CCSD_ENERGY_RESULT
    m.def("_plan_bounded_restricted_ccsd_energy", [](U o, U v, const Inventory& inventory,
        U amplitude_retained, U integral_retained, U amplitude_transient, U integral_transient,
        U singles_work, U doubles_work) {
        vibeqc::BoundedRestrictedCCSDAmplitudeProvider a;
        a.retained_numerical_bytes = amplitude_retained;
        a.maximum_transient_numerical_bytes = amplitude_transient;
        a.maximum_singles_work_units_per_query = singles_work;
        a.maximum_doubles_work_units_per_query = doubles_work;
        vibeqc::BoundedRestrictedCCSDIntegralProvider g;
        g.retained_numerical_bytes = integral_retained;
        g.maximum_transient_numerical_bytes = integral_transient;
        return vibeqc::plan_bounded_restricted_ccsd_energy(o, v, a, g, inventory);
    }, py::arg("n_occupied"), py::arg("n_virtual"), py::arg("inventory"),
       py::arg("amplitude_retained_bytes") = 0, py::arg("integral_retained_bytes") = 0,
       py::arg("amplitude_transient_bytes") = 0, py::arg("integral_transient_bytes") = 0,
       py::arg("singles_work") = 1, py::arg("doubles_work") = 1);

    m.def("_bounded_restricted_ccsd_energy_diagnostic", [](py::array t1, py::array t2,
        py::array fov, py::array eri, const Inventory& inventory, const Caps& caps,
        U amplitude_transient, U integral_transient, U singles_work, U doubles_work,
        const std::string& fail_kind, U fail_before, bool alter_external_controls) {
        const auto array = [](const py::array& a, int dimensions) {
            if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != dimensions
                || !(a.flags() & py::array::c_style)
                || reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double))
                throw std::invalid_argument("CCSD energy diagnostic requires aligned C-contiguous binary64 arrays");
        };
        array(t1, 2); array(t2, 4); array(fov, 2); array(eri, 4);
        const U o = static_cast<U>(t1.shape(0)), v = static_cast<U>(t1.shape(1));
        if (!o || !v || o > 4 || v > 8 || o + v > 10
            || caps.maximum_total_numerical_bytes > (4U << 20)
            || caps.maximum_control_storage_bytes > (1U << 20)
            || caps.maximum_singles_calls > 100000U || caps.maximum_doubles_calls > 100000U
            || caps.maximum_integral_calls > 100000U || caps.maximum_total_work_units > 100000000U
            || amplitude_transient > (1U << 20) || integral_transient > (1U << 20))
            throw std::length_error("CCSD energy diagnostic exceeds tiny count/work limits");
        const auto dim = [](const py::array& a, int axis, U expected) {
            if (static_cast<U>(a.shape(axis)) != expected)
                throw std::invalid_argument("CCSD energy diagnostic input shape mismatch");
        };
        dim(t2, 0, o); dim(t2, 1, o); dim(t2, 2, v); dim(t2, 3, v);
        dim(fov, 0, o); dim(fov, 1, v);
        for (int axis = 0; axis < 4; ++axis) dim(eri, axis, o + v);
        unsigned fail = 0;
        if (fail_kind == "singles") fail = 1;
        else if (fail_kind == "doubles") fail = 2;
        else if (fail_kind == "integrals") fail = 3;
        else if (fail_kind != "none") throw std::invalid_argument("CCSD energy diagnostic unknown failure kind");
        Caps supplied_caps = caps;
        Inventory supplied_inventory = inventory;
        struct Direct {
            const double *t1, *t2, *eri;
            U o, v, singles_count = 0, doubles_count = 0, integral_count = 0;
            unsigned fail;
            U fail_before;
            bool alter;
            Caps* supplied_caps;
            Inventory* supplied_inventory;
            void before(unsigned kind, U& count) {
                if (alter) {
                    supplied_caps->maximum_total_work_units = 0;
                    supplied_inventory->maximum_integral_work_units_per_query = std::numeric_limits<U>::max();
                    alter = false;
                }
                if (kind == fail && count == fail_before)
                    throw std::runtime_error("CCSD energy diagnostic injected callback failure");
                ++count;
            }
            static double singles(U i, U a, const void* context) {
                auto& d = *const_cast<Direct*>(static_cast<const Direct*>(context));
                d.before(1, d.singles_count);
                if (i >= d.o || a >= d.v) throw std::logic_error("CCSD energy invalid singles labels");
                return d.t1[i*d.v+a];
            }
            static double doubles(U i, U j, U a, U b, const void* context) {
                auto& d = *const_cast<Direct*>(static_cast<const Direct*>(context));
                d.before(2, d.doubles_count);
                if (i >= d.o || j >= d.o || a >= d.v || b >= d.v)
                    throw std::logic_error("CCSD energy invalid doubles labels");
                return d.t2[((i*d.o+j)*d.v+a)*d.v+b];
            }
            static double integral(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                d.before(3, d.integral_count);
                const U n = d.o + d.v;
                if (p >= n || q >= n || r >= n || s >= n)
                    throw std::logic_error("CCSD energy invalid integral labels");
                return d.eri[((p*n+q)*n+r)*n+s];
            }
        } direct{static_cast<const double*>(t1.data()), static_cast<const double*>(t2.data()),
            static_cast<const double*>(eri.data()), o, v, 0, 0, 0, fail, fail_before,
            alter_external_controls, &supplied_caps, &supplied_inventory};
        vibeqc::BoundedRestrictedCCSDAmplitudeProvider amplitudes{
            &Direct::singles, &Direct::doubles, &direct, static_cast<U>(t1.nbytes()+t2.nbytes()),
            amplitude_transient, singles_work, doubles_work};
        vibeqc::BoundedRestrictedCCSDIntegralProvider integrals{
            &Direct::integral, &direct, static_cast<U>(eri.nbytes()), integral_transient};
        vibeqc::BoundedRestrictedCCSDEnergyInput input{o, v,
            {static_cast<const double*>(fov.data()), static_cast<std::size_t>(fov.size())}};
        // Keep the GIL: all four borrowed NumPy owners stay pinned and no Python
        // callbacks are exposed. Concurrent native mutation remains forbidden.
        // This adapter's fixed Direct/descriptor stack and Python wrappers are
        // bounded diagnostic metadata, separate from the published native leaf
        // inventory; no numerical payload is copied or newly allocated here.
        return vibeqc::bounded_restricted_ccsd_energy(input, amplitudes, integrals,
            supplied_inventory, supplied_caps);
    }, py::arg("t1"), py::arg("t2"), py::arg("f_ov"), py::arg("integrals"),
       py::arg("inventory"), py::arg("caps"), py::arg("amplitude_transient_bytes") = 0,
       py::arg("integral_transient_bytes") = 0, py::arg("singles_work") = 1,
       py::arg("doubles_work") = 1, py::arg("fail_kind") = "none",
       py::arg("fail_before_call") = 0, py::arg("alter_external_controls") = false);
}
