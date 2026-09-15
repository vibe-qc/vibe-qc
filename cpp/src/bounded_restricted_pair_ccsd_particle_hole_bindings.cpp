// Included by bindings.cpp. Tiny algebra diagnostics, no physical receipt.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cfenv>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "vibeqc/bounded_restricted_pair_ccsd_particle_hole.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_pair_ccsd_particle_hole(py::module_& m) {
    using U = std::uint64_t;
    using Source = vibeqc::BoundedRestrictedPairCCSDParticleHoleSource;
    using Inventory = vibeqc::BoundedRestrictedPairCCSDParticleHoleInventory;
    using Caps = vibeqc::BoundedRestrictedPairCCSDParticleHoleCaps;
    using SourcePlan = vibeqc::BoundedRestrictedPairCCSDParticleHoleSourcePlan;
    using Plan = vibeqc::BoundedRestrictedPairCCSDParticleHoleMemoryPlan;
    using Diagnostics = vibeqc::BoundedRestrictedPairCCSDParticleHoleDiagnostics;
    using Result = vibeqc::BoundedRestrictedPairCCSDParticleHoleResult;
    using Accumulator = vibeqc::BoundedRestrictedPairCCSDParticleHoleAccumulator;
    py::class_<Inventory>(m, "_BoundedRestrictedPairCCSDParticleHoleInventory")
        .def(py::init<>())
#define PAIR_PH_INVENTORY(field) .def_readwrite(#field, &Inventory::field)
        PAIR_PH_INVENTORY(numerical_replicas) PAIR_PH_INVENTORY(external_node_bytes)
        PAIR_PH_INVENTORY(other_live_numerical_bytes_per_replica)
        PAIR_PH_INVENTORY(other_live_control_bytes_per_replica)
        PAIR_PH_INVENTORY(backend_margin_bytes_per_replica);
#undef PAIR_PH_INVENTORY
    py::class_<Caps>(m, "_BoundedRestrictedPairCCSDParticleHoleCaps")
        .def(py::init<>())
#define PAIR_PH_CAP(field) .def_readwrite(#field, &Caps::field)
        PAIR_PH_CAP(maximum_target_dimension) PAIR_PH_CAP(maximum_source_dimension)
        PAIR_PH_CAP(maximum_occupied_count) PAIR_PH_CAP(maximum_borrowed_numerical_bytes)
        PAIR_PH_CAP(maximum_owned_numerical_bytes) PAIR_PH_CAP(maximum_control_storage_bytes)
        PAIR_PH_CAP(maximum_per_replica_inventoried_bytes) PAIR_PH_CAP(maximum_node_inventoried_bytes)
        PAIR_PH_CAP(maximum_scalar_products) PAIR_PH_CAP(maximum_work_units);
#undef PAIR_PH_CAP
    py::class_<SourcePlan>(m, "_BoundedRestrictedPairCCSDParticleHoleSourcePlan")
#define PAIR_PH_SOURCE_PLAN(field) .def_readonly(#field, &SourcePlan::field)
        PAIR_PH_SOURCE_PLAN(target_dimension) PAIR_PH_SOURCE_PLAN(source_dimension)
        PAIR_PH_SOURCE_PLAN(borrowed_input_elements) PAIR_PH_SOURCE_PLAN(borrowed_numerical_bytes)
        PAIR_PH_SOURCE_PLAN(intermediate_contraction_terms) PAIR_PH_SOURCE_PLAN(projection_contraction_terms)
        PAIR_PH_SOURCE_PLAN(scalar_products) PAIR_PH_SOURCE_PLAN(input_scan_elements)
        PAIR_PH_SOURCE_PLAN(work_units_upper_bound);
#undef PAIR_PH_SOURCE_PLAN
    py::class_<Plan>(m, "_BoundedRestrictedPairCCSDParticleHoleMemoryPlan")
#define PAIR_PH_PLAN(field) .def_readonly(#field, &Plan::field)
        PAIR_PH_PLAN(target_dimension) PAIR_PH_PLAN(maximum_source_dimension) PAIR_PH_PLAN(occupied_count)
        PAIR_PH_PLAN(source_slots) PAIR_PH_PLAN(maximum_borrowed_numerical_bytes)
        PAIR_PH_PLAN(output_bytes) PAIR_PH_PLAN(compensation_bytes) PAIR_PH_PLAN(intermediate_bytes)
        PAIR_PH_PLAN(peak_owned_numerical_bytes) PAIR_PH_PLAN(fixed_control_storage_bytes)
        PAIR_PH_PLAN(total_control_storage_bytes) PAIR_PH_PLAN(other_live_numerical_bytes_per_replica)
        PAIR_PH_PLAN(backend_margin_bytes_per_replica) PAIR_PH_PLAN(numerical_replicas)
        PAIR_PH_PLAN(external_node_bytes) PAIR_PH_PLAN(per_replica_inventoried_bytes)
        PAIR_PH_PLAN(required_node_inventoried_bytes) PAIR_PH_PLAN(scalar_products_upper_bound)
        PAIR_PH_PLAN(input_scan_elements_upper_bound) PAIR_PH_PLAN(finish_work_units) PAIR_PH_PLAN(work_units_upper_bound);
#undef PAIR_PH_PLAN
    py::class_<Diagnostics>(m, "_BoundedRestrictedPairCCSDParticleHoleDiagnostics")
#define PAIR_PH_DIAGNOSTIC(field) .def_readonly(#field, &Diagnostics::field)
        PAIR_PH_DIAGNOSTIC(visited_sources) PAIR_PH_DIAGNOSTIC(zero_rank_sources)
        PAIR_PH_DIAGNOSTIC(scalar_products) PAIR_PH_DIAGNOSTIC(input_scan_elements)
        PAIR_PH_DIAGNOSTIC(product_underflow_count) PAIR_PH_DIAGNOSTIC(charged_work_units)
        PAIR_PH_DIAGNOSTIC(maximum_absolute_intermediate) PAIR_PH_DIAGNOSTIC(maximum_absolute_residual);
#undef PAIR_PH_DIAGNOSTIC
    py::class_<Result>(m, "_BoundedRestrictedPairCCSDParticleHoleResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("target_i", &Result::target_i)
        .def_property_readonly("target_j", &Result::target_j)
        .def_property_readonly("physical_source_certified", &Result::physical_source_certified)
        .def_property_readonly("entire_ccsd_residual", &Result::entire_ccsd_residual)
        .def_property_readonly("input_stream_identity_sha256", &Result::input_stream_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("identity_sha256", &Result::identity_sha256)
        .def("residual", &Result::residual)
        .def("residual_copy", [](const Result& r) {
            const U a = r.memory().target_dimension;
            if (a > 8) throw std::length_error("particle-hole copy exceeds tiny rank guard");
            py::array_t<double> output({static_cast<py::ssize_t>(a), static_cast<py::ssize_t>(a)});
            if (a) std::memcpy(output.mutable_data(), r.residual_data(), 8 * a * a);
            output.attr("setflags")(py::arg("write") = false); return output;
        });
    m.def("_plan_bounded_restricted_pair_ccsd_particle_hole_source",
        &vibeqc::plan_bounded_restricted_pair_ccsd_particle_hole_source,
        py::arg("target_dimension"), py::arg("source_dimension"));
    m.def("_plan_bounded_restricted_pair_ccsd_particle_hole",
        &vibeqc::plan_bounded_restricted_pair_ccsd_particle_hole,
        py::arg("target_dimension"), py::arg("occupied_count"), py::arg("maximum_source_dimension"), py::arg("inventory"));
    py::class_<Accumulator>(m, "_BoundedRestrictedPairCCSDParticleHoleAccumulator")
        .def(py::init([](U a, U o, U i, U j, U maximum_rank, const Inventory& inventory, const Caps& caps) {
            if (a > 8 || o > 4 || maximum_rank > 8 || caps.maximum_target_dimension > 8
                || caps.maximum_source_dimension > 8 || caps.maximum_occupied_count > 4
                || caps.maximum_owned_numerical_bytes > (1U << 20)
                || caps.maximum_borrowed_numerical_bytes > (1U << 20)
                || caps.maximum_control_storage_bytes > (1U << 20)
                || caps.maximum_per_replica_inventoried_bytes > (128U << 20)
                || caps.maximum_node_inventoried_bytes > (128U << 20)
                || caps.maximum_scalar_products > 1000000 || caps.maximum_work_units > 1000000000)
                throw std::length_error("particle-hole diagnostic exceeds tiny rank/resource guards");
            return std::make_unique<Accumulator>(a, o, i, j, maximum_rank, inventory, caps);
        }), py::arg("target_dimension"), py::arg("occupied_count"), py::arg("target_i"), py::arg("target_j"),
            py::arg("maximum_source_dimension"), py::arg("inventory"), py::arg("caps"))
        .def_property_readonly("memory", [](const Accumulator& r) { return r.memory(); })
        .def_property_readonly("failed", &Accumulator::failed)
        .def_property_readonly("finished", &Accumulator::finished)
        .def("finish", &Accumulator::finish)
        .def("accumulate", [](Accumulator& accumulator, U leg, U occupied_index,
            py::array k, py::array j, py::array overlap, py::array t,
            bool source_transposed, unsigned view_fault, unsigned rounding_fault) {
            // By-value array owners pin all inputs during the GIL-held call.
            // No forcecast, retained borrow, callback, or Python arithmetic.
            for (const py::array* x : {&k, &j, &overlap, &t})
                if (!x->dtype().is(py::dtype::of<double>()) || x->ndim() != 2
                    || !(x->flags() & py::array::c_style)
                    || (x->size() && reinterpret_cast<std::uintptr_t>(x->data()) % alignof(double)))
                    throw std::invalid_argument("particle-hole diagnostic requires aligned C-contiguous binary64 matrices");
            const U b = static_cast<U>(k.shape(0)), a = static_cast<U>(k.shape(1));
            if (a > 8 || b > 8) throw std::length_error("particle-hole diagnostic source exceeds tiny rank guards");
            if (a != accumulator.memory().target_dimension)
                throw std::invalid_argument("particle-hole diagnostic target matrix shape mismatch");
            const auto shape = [](const py::array& x, U rows, U cols) {
                if (static_cast<U>(x.shape(0)) != rows || static_cast<U>(x.shape(1)) != cols)
                    throw std::invalid_argument("particle-hole diagnostic matrix shape mismatch");
            };
            shape(j, b, a); shape(overlap, b, a); shape(t, b, b);
            if (view_fault > 4 || rounding_fault > 3) throw std::invalid_argument("particle-hole diagnostic unknown fault");
            const auto v = [](const py::array& x) {
                return vibeqc::BoundedRestrictedCCSDRealView{static_cast<const double*>(x.data()), static_cast<std::size_t>(x.size())};
            };
            Source source{leg, occupied_index, b, v(k), v(j), v(overlap), v(t), source_transposed};
            if (view_fault == 1) ++source.k_ba.element_count;
            if (view_fault == 2 && a * b)
                source.j_ba.data = reinterpret_cast<const double*>(reinterpret_cast<const unsigned char*>(j.data()) + 1);
            if (view_fault == 3) source.overlap_ba.data = nullptr;
            if (view_fault == 4) ++source.t_bb.element_count;
            struct RestoreRounding {
                int original = std::fegetround();
                ~RestoreRounding() { std::fesetround(original); }
            } restore;
            if (rounding_fault) {
                const int mode = rounding_fault == 1 ? FE_DOWNWARD : rounding_fault == 2 ? FE_UPWARD : FE_TOWARDZERO;
                if (std::fesetround(mode)) throw std::runtime_error("particle-hole diagnostic cannot set rounding mode");
            }
            accumulator.accumulate(source);
        }, py::arg("leg"), py::arg("occupied_index"), py::arg("k_ba"), py::arg("j_ba"),
            py::arg("overlap_ba"), py::arg("t_bb"), py::arg("source_transposed") = false,
            py::arg("view_fault") = 0, py::arg("rounding_fault") = 0);
}
