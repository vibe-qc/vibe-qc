// Tiny real numerical TNO diagnostics. These arrays do not certify CCSD.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cstring>
#include "vibeqc/bounded_restricted_triple_natural_orbitals.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_triple_natural_orbitals(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = BoundedRestrictedTNOOptions;
    using Inventory = BoundedRestrictedTNOInventory;
    using Caps = BoundedRestrictedTNOCaps;
    using Plan = BoundedRestrictedTNOMemoryPlan;
    using Audit = BoundedRestrictedTNOEigensystemAudit;
    using Result = BoundedRestrictedTNOResult;
    auto options = py::class_<Options>(m, "_BoundedRestrictedTNOOptions");
    options.def(py::init<>());
#define BTNO_OPTION(f) options.def_readwrite(#f, &Options::f)
    BTNO_OPTION(occupation_cutoff); BTNO_OPTION(union_absolute_rank_cutoff); BTNO_OPTION(union_relative_rank_cutoff);
    BTNO_OPTION(maximum_union_column_reconstruction_error); BTNO_OPTION(input_orthonormality_tolerance);
    BTNO_OPTION(eigensystem_relative_reconstruction_tolerance); BTNO_OPTION(eigenvector_orthogonality_tolerance);
    BTNO_OPTION(union_negative_absolute_tolerance); BTNO_OPTION(union_negative_relative_tolerance);
    BTNO_OPTION(density_negative_absolute_tolerance); BTNO_OPTION(density_negative_relative_tolerance);
    BTNO_OPTION(occupation_ambiguity_absolute_guard); BTNO_OPTION(occupation_ambiguity_relative_guard);
#undef BTNO_OPTION
#define BTNO_EIGEN(f) \
    options.def_property(#f "_max_sweeps", [](const Options& o) { return o.f##_eigensolver.max_sweeps; }, \
        [](Options& o, U value) { o.f##_eigensolver.max_sweeps = value; }) \
        .def_property(#f "_relative_eigensolver_tolerance", \
            [](const Options& o) { return o.f##_eigensolver.relative_offdiagonal_tolerance; }, \
            [](Options& o, double value) { o.f##_eigensolver.relative_offdiagonal_tolerance = value; })
    BTNO_EIGEN(union); BTNO_EIGEN(density); BTNO_EIGEN(fock);
#undef BTNO_EIGEN
    auto inventory = py::class_<Inventory>(m, "_BoundedRestrictedTNOInventory");
    inventory.def(py::init<>());
#define BTNO_INV(f) inventory.def_readwrite(#f, &Inventory::f)
    BTNO_INV(numerical_replicas); BTNO_INV(external_node_bytes); BTNO_INV(other_live_numerical_bytes_per_replica);
    BTNO_INV(other_live_control_bytes_per_replica); BTNO_INV(backend_margin_bytes_per_replica);
#undef BTNO_INV
    auto caps = py::class_<Caps>(m, "_BoundedRestrictedTNOCaps");
    caps.def(py::init<>());
#define BTNO_CAP(f) caps.def_readwrite(#f, &Caps::f)
    BTNO_CAP(maximum_common_dimension); BTNO_CAP(maximum_union_columns); BTNO_CAP(maximum_owned_numerical_bytes);
    BTNO_CAP(maximum_node_bytes); BTNO_CAP(maximum_control_storage_bytes_per_replica); BTNO_CAP(maximum_work_units);
#undef BTNO_CAP
    auto plan = py::class_<Plan>(m, "_BoundedRestrictedTNOMemoryPlan");
#define BTNO_PLAN(f) plan.def_readonly(#f, &Plan::f)
    BTNO_PLAN(common_dimension); BTNO_PLAN(unique_edge_count); BTNO_PLAN(union_columns); BTNO_PLAN(union_rank_upper_bound);
    BTNO_PLAN(maximum_edge_rank); BTNO_PLAN(borrowed_numerical_bytes); BTNO_PLAN(gram_phase_bytes); BTNO_PLAN(union_phase_bytes);
    BTNO_PLAN(density_phase_bytes); BTNO_PLAN(density_eigen_phase_bytes); BTNO_PLAN(selection_phase_bytes);
    BTNO_PLAN(semicanonical_phase_bytes); BTNO_PLAN(peak_owned_numerical_bytes); BTNO_PLAN(output_numerical_bytes_upper_bound);
    BTNO_PLAN(fixed_inventoried_object_bytes); BTNO_PLAN(control_storage_bytes_per_replica);
    BTNO_PLAN(total_node_bytes); BTNO_PLAN(work_units_upper_bound);
#undef BTNO_PLAN
    auto audit = py::class_<Audit>(m, "_BoundedRestrictedTNOEigensystemAudit");
#define BTNO_AUDIT(f) audit.def_readonly(#f, &Audit::f)
    BTNO_AUDIT(scaled_matrix_frobenius_norm); BTNO_AUDIT(scaled_reconstruction_frobenius_error);
    BTNO_AUDIT(relative_reconstruction_error); BTNO_AUDIT(orthogonality_frobenius_error);
    BTNO_AUDIT(scaled_selection_error_guard); BTNO_AUDIT(matrix_scale_exponent);
#undef BTNO_AUDIT
    audit.def_property_readonly("eigensolver_sweeps", [](const Audit& a) { return a.eigensolver.sweeps; });
    auto result = py::class_<Result>(m, "_BoundedRestrictedTNOResult");
#define BTNO_RESULT(f) result.def_readonly(#f, &Result::f)
    BTNO_RESULT(occupied); BTNO_RESULT(union_rank); BTNO_RESULT(retained_rank); BTNO_RESULT(usable);
    BTNO_RESULT(maximum_input_orthonormality_error); BTNO_RESULT(union_orthonormality_error);
    BTNO_RESULT(union_column_reconstruction_frobenius_error); BTNO_RESULT(minimum_occupation);
    BTNO_RESULT(discarded_occupation_sum); BTNO_RESULT(negative_occupation_count); BTNO_RESULT(amplitude_scale_exponent);
    BTNO_RESULT(output_numerical_bytes); BTNO_RESULT(input_identity_sha256); BTNO_RESULT(result_identity_sha256);
#undef BTNO_RESULT
    result.def_property_readonly("options", [](const Result& r) { return r.options; })
        .def_property_readonly("memory", [](const Result& r) { return r.memory; })
        .def_property_readonly("union_audit", [](const Result& r) { return r.union_audit; })
        .def_property_readonly("density_audit", [](const Result& r) { return r.density_audit; })
        .def_property_readonly("semicanonical", [](const Result& r) -> const RestrictedPairSemicanonicalResult& {
            return r.semicanonical;
        }, py::return_value_policy::reference_internal)
        .def("coefficients_copy", [](const Result& r) {
            const U n = r.memory.common_dimension, rank = r.retained_rank;
            if (n > 8 || rank > 8 || r.semicanonical.coefficients.size() != n*rank)
                throw std::length_error("TNO diagnostic coefficient copy exceeds tiny extent");
            py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(rank)});
            if (n*rank) std::memcpy(out.mutable_data(), r.semicanonical.coefficients.data(), n*rank*sizeof(double));
            return out;
        })
        .def("energies_copy", [](const Result& r) {
            if (r.retained_rank > 8 || r.semicanonical.energies.size() != r.retained_rank)
                throw std::length_error("TNO diagnostic energy copy exceeds tiny extent");
            py::array_t<double> out(static_cast<py::ssize_t>(r.retained_rank));
            if (r.retained_rank) std::memcpy(out.mutable_data(), r.semicanonical.energies.data(), r.retained_rank*sizeof(double));
            return out;
        })
        .def("occupations_copy", [](const Result& r) {
            if (r.union_rank > 8 || r.occupations.size() != r.union_rank)
                throw std::length_error("TNO diagnostic occupation copy exceeds tiny extent");
            py::array_t<double> out(static_cast<py::ssize_t>(r.union_rank));
            if (r.union_rank) std::memcpy(out.mutable_data(), r.occupations.data(), r.union_rank*sizeof(double));
            return out;
        });
    const auto dispatch = [](U o, std::array<U,3> occupied, py::array fvv, py::list coefficients,
        py::list amplitudes, Options selected, Inventory owners, Caps limits, bool planning) -> py::object {
        const auto require = [](py::handle object) {
            if (!py::isinstance<py::array>(object))
                throw std::invalid_argument("TNO diagnostic requires existing arrays");
            auto a = py::reinterpret_borrow<py::array>(object);
            if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != 2
                || !(a.flags() & py::array::c_style)
                || (a.size() && reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double)))
                throw std::invalid_argument("TNO diagnostic requires aligned C-contiguous binary64 matrices");
            return a;
        };
        fvv = require(fvv);
        if (!o || o > 4 || fvv.shape(0) < 1 || fvv.shape(0) > 8 || fvv.shape(1) != fvv.shape(0)
            || selected.union_eigensolver.max_sweeps > 200 || selected.density_eigensolver.max_sweeps > 200
            || selected.fock_eigensolver.max_sweeps > 200 || owners.numerical_replicas > 8)
            throw std::length_error("TNO diagnostic exceeds tiny dimensions or eigensolver sweeps");
        if (coefficients.size() != 3 || amplitudes.size() != 3)
            throw std::invalid_argument("TNO diagnostic requires exactly three edge occurrences (ij,ik,jk)");
        const auto view = [](const py::array& a) { return BoundedRestrictedTNORealView{
            static_cast<const double*>(a.data()), static_cast<std::size_t>(a.size())}; };
        BoundedRestrictedTNOInput input;
        input.n_occupied = o; input.common_dimension = static_cast<U>(fvv.shape(0));
        input.occupied = occupied; input.virtual_fock = view(fvv);
        std::array<py::array,6> pinned;
        const std::array<U,3> left{occupied[0],occupied[0],occupied[1]}, right{occupied[1],occupied[2],occupied[2]};
        for (U k = 0; k < 3; ++k) {
            auto& c = pinned[2*k]; auto& t = pinned[2*k+1];
            c = require(coefficients[k]); t = require(amplitudes[k]);
            if (t.shape(0) != t.shape(1) || c.shape(0) != fvv.shape(0)
                || c.shape(1) != t.shape(0) || t.shape(0) > fvv.shape(0))
                throw std::invalid_argument("TNO diagnostic edge matrix shape mismatch");
            input.edges[k] = {left[k], right[k], static_cast<U>(t.shape(0)), view(c), view(t)};
        }
        const auto plan = plan_bounded_restricted_triple_natural_orbitals(input, selected, owners);
        if (plan.peak_owned_numerical_bytes > (1U << 20) || plan.total_node_bytes > (128U << 20)
            || plan.work_units_upper_bound > 1000000000000ULL)
            throw std::length_error("TNO diagnostic exceeds tiny memory or work");
        if (planning) return py::cast(plan);
        // No callbacks or GIL release: original arrays remain pinned and no
        // hidden casts/copies replace repeated-edge pointer identity.
        return py::cast(bounded_restricted_triple_natural_orbitals(input, selected, owners, limits));
    };
    m.def("_plan_bounded_restricted_triple_natural_orbitals", [dispatch](U o, std::array<U,3> occupied,
        py::array fvv, py::list c, py::list t, Options options, Inventory inventory) {
        return dispatch(o, occupied, fvv, c, t, options, inventory, Caps{}, true);
    }, py::arg("n_occupied"), py::arg("occupied"), py::arg("virtual_fock").noconvert(),
       py::arg("coefficients"), py::arg("connected_doubles"), py::arg("options"), py::arg("inventory"));
    m.def("_bounded_restricted_triple_natural_orbitals_diagnostic", [dispatch](U o, std::array<U,3> occupied,
        py::array fvv, py::list c, py::list t, Options options, Inventory inventory, Caps caps) {
        return dispatch(o, occupied, fvv, c, t, options, inventory, caps, false);
    }, py::arg("n_occupied"), py::arg("occupied"), py::arg("virtual_fock").noconvert(),
       py::arg("coefficients"), py::arg("connected_doubles"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
}
