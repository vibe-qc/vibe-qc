// Included by bindings.cpp. Tiny algebra diagnostic, no physical owner path.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cfenv>
#include <cstring>
#include <stdexcept>
#include "vibeqc/bounded_restricted_pair_ccsd_interaction.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_pair_ccsd_interaction(py::module_& m) {
    using U = std::uint64_t;
    using Input = vibeqc::BoundedRestrictedPairCCSDInteractionInput;
    using Inventory = vibeqc::BoundedRestrictedPairCCSDInteractionInventory;
    using Caps = vibeqc::BoundedRestrictedPairCCSDInteractionCaps;
    using Plan = vibeqc::BoundedRestrictedPairCCSDInteractionMemoryPlan;
    using Diagnostics = vibeqc::BoundedRestrictedPairCCSDInteractionDiagnostics;
    using Result = vibeqc::BoundedRestrictedPairCCSDInteractionResult;
    py::class_<Inventory>(m, "_BoundedRestrictedPairCCSDInteractionInventory")
        .def(py::init<>())
#define PAIR_INTERACTION_INVENTORY(field) .def_readwrite(#field, &Inventory::field)
        PAIR_INTERACTION_INVENTORY(numerical_replicas) PAIR_INTERACTION_INVENTORY(external_node_bytes)
        PAIR_INTERACTION_INVENTORY(other_live_numerical_bytes_per_replica)
        PAIR_INTERACTION_INVENTORY(other_live_control_bytes_per_replica)
        PAIR_INTERACTION_INVENTORY(backend_margin_bytes_per_replica);
#undef PAIR_INTERACTION_INVENTORY
    py::class_<Caps>(m, "_BoundedRestrictedPairCCSDInteractionCaps")
        .def(py::init<>())
#define PAIR_INTERACTION_CAP(field) .def_readwrite(#field, &Caps::field)
        PAIR_INTERACTION_CAP(maximum_target_dimension) PAIR_INTERACTION_CAP(maximum_source_dimension)
        PAIR_INTERACTION_CAP(maximum_borrowed_numerical_bytes) PAIR_INTERACTION_CAP(maximum_owned_numerical_bytes)
        PAIR_INTERACTION_CAP(maximum_control_storage_bytes) PAIR_INTERACTION_CAP(maximum_per_replica_inventoried_bytes)
        PAIR_INTERACTION_CAP(maximum_node_inventoried_bytes) PAIR_INTERACTION_CAP(maximum_scalar_products)
        PAIR_INTERACTION_CAP(maximum_work_units);
#undef PAIR_INTERACTION_CAP
    py::class_<Plan>(m, "_BoundedRestrictedPairCCSDInteractionMemoryPlan")
#define PAIR_INTERACTION_PLAN(field) .def_readonly(#field, &Plan::field)
        PAIR_INTERACTION_PLAN(target_dimension) PAIR_INTERACTION_PLAN(source_dimension)
        PAIR_INTERACTION_PLAN(borrowed_input_elements) PAIR_INTERACTION_PLAN(borrowed_numerical_bytes)
        PAIR_INTERACTION_PLAN(output_bytes) PAIR_INTERACTION_PLAN(intermediate_bytes)
        PAIR_INTERACTION_PLAN(peak_owned_numerical_bytes) PAIR_INTERACTION_PLAN(fixed_control_storage_bytes)
        PAIR_INTERACTION_PLAN(total_control_storage_bytes) PAIR_INTERACTION_PLAN(other_live_numerical_bytes_per_replica)
        PAIR_INTERACTION_PLAN(backend_margin_bytes_per_replica) PAIR_INTERACTION_PLAN(numerical_replicas)
        PAIR_INTERACTION_PLAN(external_node_bytes) PAIR_INTERACTION_PLAN(per_replica_inventoried_bytes)
        PAIR_INTERACTION_PLAN(required_node_inventoried_bytes) PAIR_INTERACTION_PLAN(intermediate_contraction_terms)
        PAIR_INTERACTION_PLAN(residual_contraction_terms) PAIR_INTERACTION_PLAN(scalar_products)
        PAIR_INTERACTION_PLAN(input_scan_elements) PAIR_INTERACTION_PLAN(output_scan_elements)
        PAIR_INTERACTION_PLAN(work_units_upper_bound);
#undef PAIR_INTERACTION_PLAN
    py::class_<Diagnostics>(m, "_BoundedRestrictedPairCCSDInteractionDiagnostics")
#define PAIR_INTERACTION_DIAGNOSTIC(field) .def_readonly(#field, &Diagnostics::field)
        PAIR_INTERACTION_DIAGNOSTIC(scalar_products) PAIR_INTERACTION_DIAGNOSTIC(product_underflow_count)
        PAIR_INTERACTION_DIAGNOSTIC(intermediate_contraction_terms) PAIR_INTERACTION_DIAGNOSTIC(residual_contraction_terms)
        PAIR_INTERACTION_DIAGNOSTIC(charged_work_units) PAIR_INTERACTION_DIAGNOSTIC(maximum_absolute_intermediate)
        PAIR_INTERACTION_DIAGNOSTIC(maximum_absolute_residual);
#undef PAIR_INTERACTION_DIAGNOSTIC
    py::class_<Result>(m, "_BoundedRestrictedPairCCSDInteractionResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("target_dimension", &Result::target_dimension)
        .def_property_readonly("source_dimension", &Result::source_dimension)
        .def_property_readonly("source_transposed", &Result::source_transposed)
        .def_property_readonly("physical_source_certified", &Result::physical_source_certified)
        .def_property_readonly("input_payload_identity_sha256", &Result::input_payload_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("identity_sha256", &Result::identity_sha256)
        .def("residual", &Result::residual)
        .def("residual_copy", [](const Result& r) {
            const U a = r.target_dimension();
            if (a > 16) throw std::length_error("pair CCSD interaction copy exceeds tiny rank guard");
            py::array_t<double> output({static_cast<py::ssize_t>(a), static_cast<py::ssize_t>(a)});
            if (a) std::memcpy(output.mutable_data(), r.residual_data(), 8 * a * a);
            output.attr("setflags")(py::arg("write") = false);
            return output;
        });
    m.def("_plan_bounded_restricted_pair_ccsd_interaction",
        &vibeqc::plan_bounded_restricted_pair_ccsd_interaction,
        py::arg("target_dimension"), py::arg("source_dimension"), py::arg("inventory"));
    m.def("_bounded_restricted_pair_ccsd_interaction_diagnostic",
        [](py::array k, py::array j, py::array overlap, py::array t,
           const Inventory& supplied_inventory, const Caps& supplied_caps,
           bool source_transposed, unsigned view_fault, unsigned rounding_fault) {
            // Arrays are accepted without forcecast, copied owners stay pinned
            // and the GIL remains held throughout the callback-free native call.
            const Inventory inventory = supplied_inventory; const Caps caps = supplied_caps;
            for (const py::array* x : {&k, &j, &overlap, &t})
                if (!x->dtype().is(py::dtype::of<double>()) || x->ndim() != 2
                    || !(x->flags() & py::array::c_style)
                    || (x->size() && reinterpret_cast<std::uintptr_t>(x->data()) % alignof(double)))
                    throw std::invalid_argument("pair CCSD interaction diagnostic requires aligned C-contiguous binary64 matrices");
            const U a = static_cast<U>(k.shape(0)), b = static_cast<U>(k.shape(1));
            if (a > 16 || b > 16 || caps.maximum_target_dimension > 16 || caps.maximum_source_dimension > 16
                || caps.maximum_owned_numerical_bytes > (1U << 20)
                || caps.maximum_borrowed_numerical_bytes > (1U << 20)
                || caps.maximum_control_storage_bytes > (1U << 20)
                || caps.maximum_per_replica_inventoried_bytes > (128U << 20)
                || caps.maximum_node_inventoried_bytes > (128U << 20)
                || caps.maximum_scalar_products > 1000000 || caps.maximum_work_units > 1000000000)
                throw std::length_error("pair CCSD interaction diagnostic exceeds tiny rank/resource guards");
            const auto shape = [](const py::array& x, U rows, U cols) {
                if (static_cast<U>(x.shape(0)) != rows || static_cast<U>(x.shape(1)) != cols)
                    throw std::invalid_argument("pair CCSD interaction diagnostic matrix shape mismatch");
            };
            shape(j, b, a); shape(overlap, b, a); shape(t, b, b);
            if (view_fault > 4 || rounding_fault > 3)
                throw std::invalid_argument("pair CCSD interaction diagnostic unknown fault");
            const auto v = [](const py::array& x) {
                return vibeqc::BoundedRestrictedCCSDRealView{static_cast<const double*>(x.data()), static_cast<std::size_t>(x.size())};
            };
            Input in{a, b, v(k), v(j), v(overlap), v(t), source_transposed};
            if (view_fault == 1) in.k_ab.element_count = a * b ? a * b - 1 : 1;
            if (view_fault == 2 && a * b)
                in.j_ba.data = reinterpret_cast<const double*>(reinterpret_cast<const unsigned char*>(j.data()) + 1);
            if (view_fault == 3) in.overlap_ba.data = nullptr;
            if (view_fault == 4) ++in.t_bb.element_count;
            struct RestoreRounding {
                int original = std::fegetround();
                ~RestoreRounding() { std::fesetround(original); }
            } restore;
            if (rounding_fault) {
                const int mode = rounding_fault == 1 ? FE_DOWNWARD : rounding_fault == 2 ? FE_UPWARD : FE_TOWARDZERO;
                if (std::fesetround(mode)) throw std::runtime_error("pair CCSD interaction diagnostic cannot set rounding mode");
            }
            return vibeqc::bounded_restricted_pair_ccsd_interaction(in, inventory, caps);
        }, py::arg("k_ab"), py::arg("j_ba"), py::arg("overlap_ba"), py::arg("t_bb"),
        py::arg("inventory"), py::arg("caps"), py::arg("source_transposed") = false,
        py::arg("view_fault") = 0, py::arg("rounding_fault") = 0);
}
