// Internal tiny numerical diagnostics, not a production symmetry admission.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_ao_bloch_transport.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_ao_bloch_transport(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicAOBlochTransportOptions;
    using Inventory = PeriodicAOBlochTransportInventory;
    using Caps = PeriodicAOBlochTransportCaps;
    using Plan = PeriodicAOBlochTransportPlan;
    using Diagnostics = PeriodicAOBlochTransportDiagnostics;
    using Result = PeriodicAOBlochTransportResult;
    using U = std::uint64_t;
    using C = std::complex<double>;
    // SymmetryOp is intentionally read-only in the general Python API.
    // This diagnostic factory makes a raw descriptor, not a symmetry
    // certificate; the numerical leaf still validates the complete action.
    m.def("_make_periodic_ao_bloch_operation", [](py::array rotation, py::array translation) {
        if (!rotation.dtype().is(py::dtype::of<int>()) || rotation.ndim() != 2
            || rotation.shape(0) != 3 || rotation.shape(1) != 3
            || !(rotation.flags() & py::array::c_style)
            || !translation.dtype().is(py::dtype::of<double>()) || translation.ndim() != 1
            || translation.shape(0) != 3 || !(translation.flags() & py::array::c_style))
            throw std::invalid_argument("AO Bloch operation requires C-contiguous int32 [3,3] and float64 [3]");
        SymmetryOp op;
        // memcpy accepts unaligned raw descriptors without an unsafe cast.
        for (int i = 0; i < 3; ++i) {
            std::memcpy(&op.translation(i), static_cast<const char*>(translation.data())
                + sizeof(double) * i, sizeof(double));
            for (int j = 0; j < 3; ++j)
                std::memcpy(&op.rotation(i,j), static_cast<const char*>(rotation.data())
                    + sizeof(int) * (3*i+j), sizeof(int));
        }
        return op;
    }, py::arg("rotation").noconvert(), py::arg("translation").noconvert());
    auto options = py::class_<Options>(m, "_PeriodicAOBlochTransportOptions").def(py::init<>());
#define AO_OPT(x) options.def_readwrite(#x, &Options::x)
    AO_OPT(maximum_atom_mapping_residual_bohr); AO_OPT(maximum_basis_origin_residual_bohr);
    AO_OPT(maximum_rotation_orthogonality_residual); AO_OPT(maximum_polynomial_reconstruction_residual);
    AO_OPT(maximum_pure_rotation_unitarity_residual); AO_OPT(minimum_relative_lattice_volume);
#undef AO_OPT
    auto inventory = py::class_<Inventory>(m, "_PeriodicAOBlochTransportInventory").def(py::init<>());
#define AO_INV(x) inventory.def_readwrite(#x, &Inventory::x)
    AO_INV(numerical_replicas); AO_INV(external_node_bytes);
    AO_INV(other_live_numerical_bytes_per_replica); AO_INV(other_live_control_bytes_per_replica);
    AO_INV(backend_margin_bytes_per_replica);
#undef AO_INV
    auto caps = py::class_<Caps>(m, "_PeriodicAOBlochTransportCaps").def(py::init<>());
#define AO_CAP(x) caps.def_readwrite(#x, &Caps::x)
    AO_CAP(maximum_atoms); AO_CAP(maximum_shells); AO_CAP(maximum_contractions);
    AO_CAP(maximum_basis_functions); AO_CAP(maximum_columns); AO_CAP(maximum_basis_numeric_lanes);
    AO_CAP(maximum_borrowed_numerical_bytes); AO_CAP(maximum_owned_numerical_bytes);
    AO_CAP(maximum_control_storage_bytes); AO_CAP(maximum_per_replica_inventoried_bytes);
    AO_CAP(maximum_node_inventoried_bytes); AO_CAP(maximum_work_units);
#undef AO_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicAOBlochTransportPlan");
#define AO_PLAN(x) plan.def_readonly(#x, &Plan::x)
    AO_PLAN(n_atoms); AO_PLAN(n_shells); AO_PLAN(n_contractions); AO_PLAN(n_basis); AO_PLAN(n_columns);
    AO_PLAN(basis_numeric_lanes); AO_PLAN(n_kpoints); AO_PLAN(source_index); AO_PLAN(target_index);
    AO_PLAN(mesh); AO_PLAN(is_shift); AO_PLAN(target_doubled_address); AO_PLAN(reciprocal_wrap);
    AO_PLAN(time_reversal); AO_PLAN(retained_output_bytes); AO_PLAN(borrowed_coefficient_bytes);
    AO_PLAN(retained_mapping_bytes);
    AO_PLAN(borrowed_basis_numeric_bytes); AO_PLAN(borrowed_geometry_numeric_bytes);
    AO_PLAN(borrowed_numerical_bytes); AO_PLAN(mapping_workspace_bytes);
    AO_PLAN(fixed_numerical_workspace_bytes); AO_PLAN(peak_owned_numerical_bytes);
    AO_PLAN(control_storage_bytes); AO_PLAN(per_replica_inventoried_bytes);
    AO_PLAN(required_node_inventoried_bytes); AO_PLAN(work_units_upper_bound);
#undef AO_PLAN
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicAOBlochTransportDiagnostics");
#define AO_DIAG(x) diagnostics.def_readonly(#x, &Diagnostics::x)
    AO_DIAG(mapped_atoms); AO_DIAG(mapped_shells); AO_DIAG(rotated_contractions);
    AO_DIAG(pure_rotation_blocks); AO_DIAG(cartesian_rotation_blocks);
    AO_DIAG(maximum_atom_mapping_residual_bohr); AO_DIAG(maximum_basis_origin_residual_bohr);
    AO_DIAG(rotation_orthogonality_residual); AO_DIAG(maximum_polynomial_reconstruction_residual);
    AO_DIAG(maximum_pure_rotation_unitarity_residual); AO_DIAG(maximum_output_magnitude);
#undef AO_DIAG
    const auto small = [](const BasisSet& basis, const PeriodicSystem& system, U columns) {
        if (system.unit_cell.size() > 32 || basis.nshells() > 128 || basis.nbasis() > 256 || columns > 32)
            throw std::length_error("AO Bloch transport diagnostic exceeds tiny shape bounds");
    };
    const auto bounded = [](const Plan& p) {
        if (p.peak_owned_numerical_bytes > (16U << 20)
            || p.required_node_inventoried_bytes > (128U << 20)
            || p.work_units_upper_bound > 1000000000ULL)
            throw std::length_error("AO Bloch transport diagnostic exceeds tiny memory/work bounds");
    };
    py::class_<Result>(m, "_PeriodicAOBlochTransportResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("physical_orbital_sewing_certified", &Result::physical_orbital_sewing_certified)
        .def("coefficient", &Result::coefficient)
        .def("atom_destination", &Result::atom_destination)
        .def("atom_lattice_shift", &Result::atom_lattice_shift)
        .def("shell_destination", &Result::shell_destination)
        .def("coefficients_copy", [](const Result& r) {
            const auto& p = r.memory();
            if (p.n_basis > 256 || p.n_columns > 32 || p.retained_output_bytes > (1U << 20))
                throw std::length_error("AO Bloch transport diagnostic copy exceeds tiny bounds");
            const auto* data = r.data();
            py::array_t<C> out({py::ssize_t(p.n_basis), py::ssize_t(p.n_columns)});
            if (p.retained_output_bytes) std::memcpy(out.mutable_data(), data, p.retained_output_bytes);
            out.attr("setflags")(false);
            return out;
        });
    m.def("_plan_periodic_ao_bloch_transport", [small, bounded](
        const BasisSet& basis, const PeriodicSystem& system, const SymmetryOp& operation,
        const RegularKMesh& mesh, U source, bool tr, U columns,
        const Options& options, const Inventory& inventory, const Caps& caps) {
        small(basis, system, columns);
        auto p = plan_periodic_ao_bloch_transport(basis, system, operation, mesh,
            source, tr, columns, options, inventory, caps);
        bounded(p);
        return p;
    }, py::arg("basis"), py::arg("system"), py::arg("operation"), py::arg("mesh"),
       py::arg("source_index"), py::arg("time_reversal"), py::arg("n_columns"),
       py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_apply_periodic_ao_bloch_operation", [small, bounded](
        const BasisSet& basis, const PeriodicSystem& system, const SymmetryOp& operation,
        const RegularKMesh& mesh, U source, bool tr, py::array coefficients,
        const Options& options, const Inventory& inventory, const Caps& caps) {
        if (!coefficients.dtype().is(py::dtype::of<C>()) || coefficients.ndim() != 2
            || !(coefficients.flags() & py::array::c_style)
            || coefficients.shape(0) != py::ssize_t(basis.nbasis())
            || reinterpret_cast<std::uintptr_t>(coefficients.data()) % alignof(C))
            throw std::invalid_argument("AO Bloch coefficients require aligned C-contiguous complex128 [nao,columns]");
        const U columns = U(coefficients.shape(1));
        small(basis, system, columns);
        auto p = plan_periodic_ao_bloch_transport(basis, system, operation, mesh,
            source, tr, columns, options, inventory, caps);
        bounded(p);
        // Keep the GIL: there are no callbacks, conversions or concurrent
        // Python mutation windows for these tiny borrowed diagnostic panels.
        return apply_periodic_ao_bloch_operation(basis, system, operation, mesh, source, tr,
            static_cast<const C*>(coefficients.data()), U(coefficients.size()), columns,
            options, inventory, caps);
    }, py::arg("basis"), py::arg("system"), py::arg("operation"), py::arg("mesh"),
       py::arg("source_index"), py::arg("time_reversal"), py::arg("coefficients").noconvert(),
       py::arg("options"), py::arg("inventory"), py::arg("caps"));
}
