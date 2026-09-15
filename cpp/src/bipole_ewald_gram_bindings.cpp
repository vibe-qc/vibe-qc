// Bounded integral-only witnesses. Included by bindings.cpp, not compiled alone.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <memory>
#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/bipole_ewald_gram.hpp"
#include "vibeqc/bipole_erfc_panel.hpp"
#include "vibeqc/bipole_erfc_bloch.hpp"
#include "vibeqc/bipole_finite_source.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {
using BipoleDiagnosticComplex = std::complex<double>;
py::array_t<BipoleDiagnosticComplex> bipole_diagnostic_array(
    std::vector<BipoleDiagnosticComplex> data, std::uint64_t rows, std::uint64_t columns) {
    auto owner = std::make_unique<std::vector<BipoleDiagnosticComplex>>(std::move(data));
    py::capsule lifetime(owner.get(), [](void* p) {
        delete static_cast<std::vector<BipoleDiagnosticComplex>*>(p);
    });
    auto* values = owner.release();
    py::array_t<BipoleDiagnosticComplex> result(
        {py::ssize_t(rows), py::ssize_t(columns)},
        {py::ssize_t(columns * sizeof(BipoleDiagnosticComplex)),
         py::ssize_t(sizeof(BipoleDiagnosticComplex))}, values->data(), lifetime);
    result.attr("setflags")(false);
    return result;
}
vibeqc::AOPairFourierCellView bipole_diagnostic_cells(
    const py::array_t<std::int64_t, py::array::c_style>& cells) {
    if (cells.ndim() != 2 || cells.shape(1) != 3)
        throw std::invalid_argument("BIPOLE cells must be exact int64 (n,3)");
    if (cells.shape(0) > 128)
        throw std::length_error("BIPOLE diagnostic exceeds tiny cell bounds");
    return {cells.data(), std::uint64_t(cells.shape(0)), std::uint64_t(cells.size())};
}
} // namespace

#include "bipole_finite_source_bindings.cpp"

void bind_bipole_erfc_panel(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Selection = BipoleErfcPanelSelection;
    auto selection = py::class_<Selection>(m, "_BipoleErfcPanelSelection").def(py::init<>());
    selection.def_readwrite("left_pair_begin", &Selection::left_pair_begin);
    selection.def_readwrite("left_pair_count", &Selection::left_pair_count);
    selection.def_readwrite("right_pair_begin", &Selection::right_pair_begin);
    selection.def_readwrite("right_pair_count", &Selection::right_pair_count);
    using Options = BipoleErfcPanelOptions;
    auto options = py::class_<Options>(m, "_BipoleErfcPanelOptions").def(py::init<>());
    options.def_readwrite("omega", &Options::omega);
    using Inventory = BipoleErfcPanelInventory;
    auto inventory = py::class_<Inventory>(m, "_BipoleErfcPanelInventory").def(py::init<>());
    inventory.def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    inventory.def_readwrite("external_node_bytes", &Inventory::external_node_bytes);
    inventory.def_readwrite("other_live_numerical_bytes_per_replica", &Inventory::other_live_numerical_bytes_per_replica);
    inventory.def_readwrite("other_live_control_bytes_per_replica", &Inventory::other_live_control_bytes_per_replica);
    inventory.def_readwrite("backend_margin_bytes_per_replica", &Inventory::backend_margin_bytes_per_replica);
    using Caps = BipoleErfcPanelCaps;
    auto caps = py::class_<Caps>(m, "_BipoleErfcPanelCaps").def(py::init<>());
    caps.def_readwrite("maximum_images", &Caps::maximum_images);
    caps.def_readwrite("maximum_shell_quartet_calls", &Caps::maximum_shell_quartet_calls);
    caps.def_readwrite("maximum_primitive_quartet_visits", &Caps::maximum_primitive_quartet_visits);
    caps.def_readwrite("maximum_borrowed_numerical_bytes", &Caps::maximum_borrowed_numerical_bytes);
    caps.def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes);
    caps.def_readwrite("maximum_control_storage_bytes", &Caps::maximum_control_storage_bytes);
    caps.def_readwrite("maximum_worker_bytes", &Caps::maximum_worker_bytes);
    caps.def_readwrite("maximum_node_bytes", &Caps::maximum_node_bytes);
    caps.def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    using Plan = BipoleErfcPanelPlan;
    auto plan = py::class_<Plan>(m, "_BipoleErfcPanelPlan");
    plan.def_readonly("n_basis", &Plan::n_basis);
    plan.def_readonly("n_shells", &Plan::n_shells);
    plan.def_readonly("image_count", &Plan::image_count);
    plan.def_readonly("selection", &Plan::selection);
    plan.def_readonly("maximum_primitives", &Plan::maximum_primitives);
    plan.def_readonly("maximum_angular_momentum", &Plan::maximum_angular_momentum);
    plan.def_readonly("backend_maximum_angular_momentum", &Plan::backend_maximum_angular_momentum);
    plan.def_readonly("output_elements", &Plan::output_elements);
    plan.def_readonly("retained_output_bytes", &Plan::retained_output_bytes);
    plan.def_readonly("shell_quartet_calls_upper_bound", &Plan::shell_quartet_calls_upper_bound);
    plan.def_readonly("primitive_quartet_visits_upper_bound", &Plan::primitive_quartet_visits_upper_bound);
    plan.def_readonly("maximum_shell_quartet_elements", &Plan::maximum_shell_quartet_elements);
    plan.def_readonly("engine_primitive_bytes", &Plan::engine_primitive_bytes);
    plan.def_readonly("engine_pair_bytes", &Plan::engine_pair_bytes);
    plan.def_readonly("engine_stack_bytes", &Plan::engine_stack_bytes);
    plan.def_readonly("engine_scratch_bytes", &Plan::engine_scratch_bytes);
    plan.def_readonly("engine_core_numeric_bytes", &Plan::engine_core_numeric_bytes);
    plan.def_readonly("shell_copy_numeric_bytes", &Plan::shell_copy_numeric_bytes);
    plan.def_readonly("fixed_numeric_workspace_bytes", &Plan::fixed_numeric_workspace_bytes);
    plan.def_readonly("borrowed_basis_numeric_bytes", &Plan::borrowed_basis_numeric_bytes);
    plan.def_readonly("borrowed_image_bytes", &Plan::borrowed_image_bytes);
    plan.def_readonly("borrowed_geometry_numeric_bytes", &Plan::borrowed_geometry_numeric_bytes);
    plan.def_readonly("borrowed_numerical_bytes", &Plan::borrowed_numerical_bytes);
    plan.def_readonly("peak_owned_numerical_bytes", &Plan::peak_owned_numerical_bytes);
    plan.def_readonly("control_storage_bytes", &Plan::control_storage_bytes);
    plan.def_readonly("per_replica_inventoried_bytes", &Plan::per_replica_inventoried_bytes);
    plan.def_readonly("required_node_inventoried_bytes", &Plan::required_node_inventoried_bytes);
    plan.def_readonly("metadata_work_units", &Plan::metadata_work_units);
    plan.def_readonly("validation_work_units", &Plan::validation_work_units);
    plan.def_readonly("contraction_work_units", &Plan::contraction_work_units);
    plan.def_readonly("work_units_upper_bound", &Plan::work_units_upper_bound);
    using Diagnostics = BipoleErfcPanelDiagnostics;
    auto diagnostics = py::class_<Diagnostics>(m, "_BipoleErfcPanelDiagnostics");
    diagnostics.def_readonly("shell_quartet_calls", &Diagnostics::shell_quartet_calls);
    diagnostics.def_readonly("null_shell_quartet_buffers", &Diagnostics::null_shell_quartet_buffers);
    diagnostics.def_readonly("maximum_integral_magnitude", &Diagnostics::maximum_integral_magnitude);
    diagnostics.def_readonly("primitive_screening_precision", &Diagnostics::primitive_screening_precision);
    using Result = BipoleErfcPanelResult;
    py::class_<Result>(m, "_BipoleErfcPanelResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("input_identity_sha256", &Result::input_identity_sha256)
        .def_property_readonly("source_identity_sha256", &Result::source_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &Result::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &Result::symmetry_certified)
        .def_property_readonly("citation_numerics", [](const Result&) {
            return std::vector<std::string>{"bipole_erfc_panel"};
        })
        .def("element", &Result::element)
        .def("values_copy", [](const Result& r) {
            const auto& p = r.memory();
            if (p.image_count > 64 || p.selection.left_pair_count > 64 || p.selection.right_pair_count > 64)
                throw std::length_error("BIPOLE erfc copy exceeds tiny bounds");
            py::array_t<double> out({py::ssize_t(p.image_count),
                py::ssize_t(p.selection.left_pair_count), py::ssize_t(p.selection.right_pair_count)});
            if (p.output_elements) std::copy_n(r.data(), p.output_elements, out.mutable_data());
            out.attr("setflags")(false);
            return out;
        });
    auto images = [](const py::array_t<std::int64_t, py::array::c_style>& a) {
        if (a.ndim() != 3 || a.shape(1) != 3 || a.shape(2) != 3)
            throw std::invalid_argument("BIPOLE erfc images must be exact int64 (n,3,3)");
        if (a.shape(0) > 64) throw std::length_error("BIPOLE erfc diagnostic exceeds tiny image bounds");
        return BipoleErfcImageView{a.data(), U(a.shape(0)), U(a.size())};
    };
    auto admit = [](const BasisSet& b, const PeriodicSystem& system, BipoleErfcImageView v,
                    const Selection& s, const Options& o, const Inventory& i, const Caps& c) {
        if (b.nbasis() > 64 || b.nshells() > 128 || s.left_pair_count > 64 || s.right_pair_count > 64)
            throw std::length_error("BIPOLE erfc diagnostic exceeds tiny basis/pair bounds");
        auto p = plan_bipole_erfc_panel(b, system, v, s, o, i, c);
        if (p.peak_owned_numerical_bytes > (128U << 20) || p.required_node_inventoried_bytes > (512U << 20)
            || p.work_units_upper_bound > 100000000000ULL || p.shell_quartet_calls_upper_bound > 100000)
            throw std::length_error("BIPOLE erfc diagnostic exceeds tiny memory/work bounds");
        return p;
    };
    m.def("_plan_bipole_erfc_panel", [images, admit](const BasisSet& b, const PeriodicSystem& system,
        const py::array_t<std::int64_t, py::array::c_style>& a, const Selection& s,
        const Options& o, const Inventory& i, const Caps& c) {
        return admit(b, system, images(a), s, o, i, c);
    }, py::arg("basis"), py::arg("system"), py::arg("images").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_make_bipole_erfc_panel", [images, admit](const BasisSet& b, const PeriodicSystem& system,
        const py::array_t<std::int64_t, py::array::c_style>& a, const Selection& s,
        const Options& o, const Inventory& i, const Caps& c) {
        const auto v = images(a);
        admit(b, system, v, s, o, i, c);
        // GIL stays held and no callbacks run: borrowed input owners remain live.
        return make_bipole_erfc_panel(b, system, v, s, o, i, c);
    }, py::arg("basis"), py::arg("system"), py::arg("images").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
}

void bind_bipole_erfc_bloch(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Selection = BipoleEwaldGramSelection;
    using Options = BipoleErfcBlochOptions;
    auto options = py::class_<Options>(m, "_BipoleErfcBlochOptions").def(py::init<>());
    options.def_readwrite("raw", &Options::raw);
    options.def_readwrite("require_image_permutation_closure", &Options::require_image_permutation_closure);
    using Inventory = BipoleErfcBlochInventory;
    auto inventory = py::class_<Inventory>(m, "_BipoleErfcBlochInventory").def(py::init<>());
    inventory.def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    inventory.def_readwrite("external_node_bytes", &Inventory::external_node_bytes);
    inventory.def_readwrite("other_live_numerical_bytes_per_replica", &Inventory::other_live_numerical_bytes_per_replica);
    inventory.def_readwrite("other_live_control_bytes_per_replica", &Inventory::other_live_control_bytes_per_replica);
    inventory.def_readwrite("backend_margin_bytes_per_replica", &Inventory::backend_margin_bytes_per_replica);
    using Caps = BipoleErfcBlochCaps;
    auto caps = py::class_<Caps>(m, "_BipoleErfcBlochCaps").def(py::init<>());
    caps.def_readwrite("raw", &Caps::raw);
    caps.def_readwrite("maximum_kpoints", &Caps::maximum_kpoints);
    caps.def_readwrite("maximum_phase_evaluations", &Caps::maximum_phase_evaluations);
    caps.def_readwrite("maximum_support_image_comparisons", &Caps::maximum_support_image_comparisons);
    caps.def_readwrite("maximum_borrowed_numerical_bytes", &Caps::maximum_borrowed_numerical_bytes);
    caps.def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes);
    caps.def_readwrite("maximum_control_storage_bytes", &Caps::maximum_control_storage_bytes);
    caps.def_readwrite("maximum_worker_bytes", &Caps::maximum_worker_bytes);
    caps.def_readwrite("maximum_node_bytes", &Caps::maximum_node_bytes);
    caps.def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    using Plan = BipoleErfcBlochPlan;
    auto plan = py::class_<Plan>(m, "_BipoleErfcBlochPlan");
    plan.def_readonly("n_basis", &Plan::n_basis);
    plan.def_readonly("n_kpoints", &Plan::n_kpoints);
    plan.def_readonly("image_count", &Plan::image_count);
    plan.def_readonly("selection", &Plan::selection);
    plan.def_readonly("output_elements", &Plan::output_elements);
    plan.def_readonly("retained_output_bytes", &Plan::retained_output_bytes);
    plan.def_readonly("compensation_bytes", &Plan::compensation_bytes);
    plan.def_readonly("phase_evaluations", &Plan::phase_evaluations);
    plan.def_readonly("folded_scalar_terms", &Plan::folded_scalar_terms);
    plan.def_readonly("support_image_comparisons_upper_bound", &Plan::support_image_comparisons_upper_bound);
    plan.def_readonly("support_work_units", &Plan::support_work_units);
    plan.def_readonly("fixed_numeric_workspace_bytes", &Plan::fixed_numeric_workspace_bytes);
    plan.def_readonly("wrapper_control_storage_bytes", &Plan::wrapper_control_storage_bytes);
    plan.def_readonly("raw_phase_owned_numerical_bytes", &Plan::raw_phase_owned_numerical_bytes);
    plan.def_readonly("fold_phase_owned_numerical_bytes", &Plan::fold_phase_owned_numerical_bytes);
    plan.def_readonly("peak_owned_numerical_bytes", &Plan::peak_owned_numerical_bytes);
    plan.def_readonly("borrowed_numerical_bytes", &Plan::borrowed_numerical_bytes);
    plan.def_readonly("control_storage_bytes", &Plan::control_storage_bytes);
    plan.def_readonly("per_replica_inventoried_bytes", &Plan::per_replica_inventoried_bytes);
    plan.def_readonly("required_node_inventoried_bytes", &Plan::required_node_inventoried_bytes);
    plan.def_readonly("phase_work_units", &Plan::phase_work_units);
    plan.def_readonly("folding_work_units", &Plan::folding_work_units);
    plan.def_readonly("identity_work_units", &Plan::identity_work_units);
    plan.def_readonly("work_units_upper_bound", &Plan::work_units_upper_bound);
    plan.def_readonly("raw", &Plan::raw);
    using Diagnostics = BipoleErfcBlochDiagnostics;
    auto diagnostics = py::class_<Diagnostics>(m, "_BipoleErfcBlochDiagnostics");
    diagnostics.def_readonly("evaluated_images", &Diagnostics::evaluated_images);
    diagnostics.def_readonly("folded_scalar_terms", &Diagnostics::folded_scalar_terms);
    diagnostics.def_readonly("support_image_comparisons", &Diagnostics::support_image_comparisons);
    diagnostics.def_readonly("maximum_phase_modulus_residual", &Diagnostics::maximum_phase_modulus_residual);
    diagnostics.def_readonly("maximum_integral_magnitude", &Diagnostics::maximum_integral_magnitude);
    diagnostics.def_readonly("raw", &Diagnostics::raw);
    using Result = BipoleErfcBlochResult;
    py::class_<Result>(m, "_BipoleErfcBlochResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("input_identity_sha256", &Result::input_identity_sha256)
        .def_property_readonly("source_identity_sha256", &Result::source_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("raw_input_identity_sha256", &Result::raw_input_identity_sha256)
        .def_property_readonly("raw_source_identity_sha256", &Result::raw_source_identity_sha256)
        .def_property_readonly("raw_payload_identity_sha256", &Result::raw_payload_identity_sha256)
        .def_property_readonly("image_permutation_support_certified", &Result::image_permutation_support_certified)
        .def_property_readonly("image_permutation_support_identity_sha256", &Result::image_permutation_support_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &Result::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &Result::symmetry_certified)
        .def_property_readonly("citation_numerics", [](const Result&) {
            return std::vector<std::string>{"bipole_erfc_bloch"};
        })
        .def("element", &Result::element)
        .def("values_copy", [](const Result& r) {
            const auto& p = r.memory();
            if (p.selection.left_pair_count > 64 || p.selection.right_pair_count > 64)
                throw std::length_error("BIPOLE erfc Bloch copy exceeds tiny bounds");
            py::array_t<BipoleDiagnosticComplex> out({py::ssize_t(p.selection.left_pair_count),
                py::ssize_t(p.selection.right_pair_count)});
            if (p.output_elements) std::copy_n(r.data(), p.output_elements, out.mutable_data());
            out.attr("setflags")(false);
            return out;
        });
    auto images = [](const py::array_t<std::int64_t, py::array::c_style>& a) {
        if (a.ndim() != 3 || a.shape(1) != 3 || a.shape(2) != 3)
            throw std::invalid_argument("BIPOLE erfc Bloch images must be exact int64 (n,3,3)");
        if (a.shape(0) > 64) throw std::length_error("BIPOLE erfc Bloch diagnostic exceeds tiny image bounds");
        return BipoleErfcImageView{a.data(), U(a.shape(0)), U(a.size())};
    };
    auto admit = [](const BasisSet& b, const PeriodicSystem& system, const RegularKMesh& mesh,
                    BipoleErfcImageView v, const Selection& s, const Options& o,
                    const Inventory& i, const Caps& c) {
        if (b.nbasis() > 64 || b.nshells() > 128 || mesh.size() > 4096
            || s.left_pair_count > 64 || s.right_pair_count > 64)
            throw std::length_error("BIPOLE erfc Bloch diagnostic exceeds tiny shape bounds");
        auto p = plan_bipole_erfc_bloch(b, system, mesh, v, s, o, i, c);
        if (p.peak_owned_numerical_bytes > (128U << 20) || p.required_node_inventoried_bytes > (512U << 20)
            || p.work_units_upper_bound > 100000000000ULL || p.raw.shell_quartet_calls_upper_bound > 100000)
            throw std::length_error("BIPOLE erfc Bloch diagnostic exceeds tiny memory/work bounds");
        return p;
    };
    m.def("_plan_bipole_erfc_bloch", [images, admit](const BasisSet& b, const PeriodicSystem& system,
        const RegularKMesh& mesh, const py::array_t<std::int64_t, py::array::c_style>& a,
        const Selection& s, const Options& o, const Inventory& i, const Caps& c) {
        return admit(b, system, mesh, images(a), s, o, i, c);
    }, py::arg("basis"), py::arg("system"), py::arg("mesh"), py::arg("images").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_make_bipole_erfc_bloch", [images, admit](const BasisSet& b, const PeriodicSystem& system,
        const RegularKMesh& mesh, const py::array_t<std::int64_t, py::array::c_style>& a,
        const Selection& s, const Options& o, const Inventory& i, const Caps& c) {
        const auto v = images(a);
        admit(b, system, mesh, v, s, o, i, c);
        return make_bipole_erfc_bloch(b, system, mesh, v, s, o, i, c);
    }, py::arg("basis"), py::arg("system"), py::arg("mesh"), py::arg("images").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
    bind_bipole_finite_source(m);
}

void bind_bipole_ewald_gram(py::module_& m) {
    using namespace vibeqc;
    m.def("_bipole_ewald_weighted_product_diagnostic", &bipole_ewald_weighted_product_diagnostic,
        py::arg("left"), py::arg("right"), py::arg("weight"),
        "Scalar arithmetic witness only; no physical factor-source admission.");
    using U = std::uint64_t;
    using CellCaps = AOPairFourierCellPanelCaps;
    using CellPlan = AOPairFourierCellPanelPlan;
    auto cc = py::class_<CellCaps>(m, "_AOPairFourierCellPanelCaps").def(py::init<>());
    cc.def_readwrite("maximum_cells", &CellCaps::maximum_cells);
    cc.def_readwrite("maximum_pair_cell_visits", &CellCaps::maximum_pair_cell_visits);
    cc.def_readwrite("maximum_work_units", &CellCaps::maximum_work_units);
    cc.def_readwrite("maximum_output_bytes", &CellCaps::maximum_output_bytes);
    auto cp = py::class_<CellPlan>(m, "_AOPairFourierCellPanelPlan");
#define VIBEQC_CELL_PLAN_FIELD(field) cp.def_readonly(#field, &CellPlan::field)
    VIBEQC_CELL_PLAN_FIELD(n_basis);
    VIBEQC_CELL_PLAN_FIELD(n_vectors);
    VIBEQC_CELL_PLAN_FIELD(pair_begin);
    VIBEQC_CELL_PLAN_FIELD(n_pairs);
    VIBEQC_CELL_PLAN_FIELD(cell_count);
    VIBEQC_CELL_PLAN_FIELD(output_bytes);
    VIBEQC_CELL_PLAN_FIELD(borrowed_basis_numeric_bytes);
    VIBEQC_CELL_PLAN_FIELD(borrowed_cell_bytes);
    VIBEQC_CELL_PLAN_FIELD(basis_control_storage_bytes);
    VIBEQC_CELL_PLAN_FIELD(fixed_control_storage_bytes);
    VIBEQC_CELL_PLAN_FIELD(fixed_numeric_workspace_bytes);
    VIBEQC_CELL_PLAN_FIELD(fixed_scalar_numeric_bytes);
    VIBEQC_CELL_PLAN_FIELD(pair_cell_visits);
    VIBEQC_CELL_PLAN_FIELD(duplicate_comparisons);
    VIBEQC_CELL_PLAN_FIELD(metadata_work_units);
    VIBEQC_CELL_PLAN_FIELD(validation_work_units);
    VIBEQC_CELL_PLAN_FIELD(contraction_work_units);
    VIBEQC_CELL_PLAN_FIELD(work_units);
#undef VIBEQC_CELL_PLAN_FIELD
    auto small_cell_plan = [](const BasisSet& b, U vectors, U begin, U count,
                              U cells, const CellCaps& caps) {
        if (b.nbasis() > 64 || b.nshells() > 128 || count > 64 || vectors > 64 || cells > 128)
            throw std::length_error("BIPOLE cell diagnostic exceeds tiny shape bounds");
        auto p = plan_ao_pair_gaussian_fourier_cell_panel(b, vectors, begin, count, cells, caps);
        if (p.work_units > 2000000000ULL || p.output_bytes > (1U << 20))
            throw std::length_error("BIPOLE cell diagnostic exceeds tiny memory/work bounds");
        return p;
    };
    m.def("_plan_ao_pair_gaussian_fourier_cell_panel", small_cell_plan,
        py::arg("basis"), py::arg("n_vectors"), py::arg("pair_begin"),
        py::arg("pair_count"), py::arg("cell_count"), py::arg("caps"));
    m.def("_ao_pair_gaussian_fourier_cell_panel", [small_cell_plan](
        const BasisSet& b, const PeriodicSystem& s,
        const py::array_t<double, py::array::c_style>& vectors,
        const py::array_t<double, py::array::c_style>& k,
        U begin, U count, const py::array_t<std::int64_t, py::array::c_style>& cells,
        const CellCaps& caps) {
        if (vectors.ndim() != 2 || vectors.shape(1) != 3 || k.ndim() != 1 || k.shape(0) != 3)
            throw std::invalid_argument("BIPOLE vectors must be (n,3) and k must be (3,)");
        const auto cv = bipole_diagnostic_cells(cells);
        const auto p = small_cell_plan(b, vectors.shape(0), begin, count, cv.cell_count, caps);
        AuxiliaryFourierVectorView vv;
        vv.count = vectors.shape(0);
        vv.x_stride = vv.y_stride = vv.z_stride = 3;
        if (vv.count) { vv.x = vectors.data(); vv.y = vv.x + 1; vv.z = vv.x + 2; }
        // Count/work gates precede k payload reads; GIL remains held, no callbacks.
        const Eigen::Vector3d kc(k.data()[0], k.data()[1], k.data()[2]);
        auto r = ao_pair_gaussian_fourier_cell_panel(b, s, vv, kc, begin, count, cv, caps);
        py::dict out;
        out["plan"] = p;
        out["values"] = bipole_diagnostic_array(std::move(r.data), count, vv.count);
        return out;
    }, py::arg("basis"), py::arg("system"), py::arg("vectors").noconvert(),
       py::arg("k").noconvert(), py::arg("pair_begin"), py::arg("pair_count"),
       py::arg("cells").noconvert(), py::arg("caps"));

    using Result = BipoleEwaldGramResult;
    using Options = BipoleEwaldGramOptions;
    auto gramOptions = py::class_<Options>(m, "_BipoleEwaldGramOptions").def(py::init<>());
    gramOptions.def_readwrite("omega", &Options::omega);
    gramOptions.def_readwrite("reciprocal_energy_cutoff", &Options::reciprocal_energy_cutoff);
    gramOptions.def_readwrite("reciprocal_block_size", &Options::reciprocal_block_size);
    gramOptions.def_readwrite("require_reciprocal_conjugacy", &Options::require_reciprocal_conjugacy);
    gramOptions.def_readwrite("require_cell_inversion_closure", &Options::require_cell_inversion_closure);
    using Selection = BipoleEwaldGramSelection;
    auto gramSelection = py::class_<Selection>(m, "_BipoleEwaldGramSelection").def(py::init<>());
    gramSelection.def_readwrite("q_index", &Selection::q_index);
    gramSelection.def_readwrite("left_k_index", &Selection::left_k_index);
    gramSelection.def_readwrite("right_k_index", &Selection::right_k_index);
    gramSelection.def_readwrite("left_pair_begin", &Selection::left_pair_begin);
    gramSelection.def_readwrite("left_pair_count", &Selection::left_pair_count);
    gramSelection.def_readwrite("right_pair_begin", &Selection::right_pair_begin);
    gramSelection.def_readwrite("right_pair_count", &Selection::right_pair_count);
    using Inventory = BipoleEwaldGramInventory;
    auto gramInventory = py::class_<Inventory>(m, "_BipoleEwaldGramInventory").def(py::init<>());
    gramInventory.def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    gramInventory.def_readwrite("external_node_bytes", &Inventory::external_node_bytes);
    gramInventory.def_readwrite("other_live_numerical_bytes_per_replica", &Inventory::other_live_numerical_bytes_per_replica);
    gramInventory.def_readwrite("other_live_control_bytes_per_replica", &Inventory::other_live_control_bytes_per_replica);
    gramInventory.def_readwrite("backend_margin_bytes_per_replica", &Inventory::backend_margin_bytes_per_replica);
    using Caps = BipoleEwaldGramCaps;
    auto gramCaps = py::class_<Caps>(m, "_BipoleEwaldGramCaps").def(py::init<>());
    gramCaps.def_readwrite("ao_panel", &Caps::ao_panel);
    gramCaps.def_readwrite("maximum_kpoints", &Caps::maximum_kpoints);
    gramCaps.def_readwrite("maximum_reciprocal_candidates", &Caps::maximum_reciprocal_candidates);
    gramCaps.def_readwrite("maximum_accepted_vectors", &Caps::maximum_accepted_vectors);
    gramCaps.def_readwrite("maximum_reciprocal_blocks", &Caps::maximum_reciprocal_blocks);
    gramCaps.def_readwrite("maximum_borrowed_numerical_bytes", &Caps::maximum_borrowed_numerical_bytes);
    gramCaps.def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes);
    gramCaps.def_readwrite("maximum_control_storage_bytes", &Caps::maximum_control_storage_bytes);
    gramCaps.def_readwrite("maximum_per_replica_inventoried_bytes", &Caps::maximum_per_replica_inventoried_bytes);
    gramCaps.def_readwrite("maximum_node_inventoried_bytes", &Caps::maximum_node_inventoried_bytes);
    gramCaps.def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    using Plan = BipoleEwaldGramPlan;
    auto gramPlan = py::class_<Plan>(m, "_BipoleEwaldGramPlan");
    gramPlan.def_readonly("n_basis", &Plan::n_basis);
    gramPlan.def_readonly("n_kpoints", &Plan::n_kpoints);
    gramPlan.def_readonly("cell_count", &Plan::cell_count);
    gramPlan.def_readonly("left_cell_count", &Plan::left_cell_count);
    gramPlan.def_readonly("right_cell_count", &Plan::right_cell_count);
    gramPlan.def_readonly("shared_cell_storage", &Plan::shared_cell_storage);
    gramPlan.def_readonly("selection", &Plan::selection);
    gramPlan.def_readonly("opposite_q_index", &Plan::opposite_q_index);
    gramPlan.def_readonly("centered_q_doubled", &Plan::centered_q_doubled);
    gramPlan.def_readonly("centered_q_wrap", &Plan::centered_q_wrap);
    gramPlan.def_readonly("reciprocal_candidates", &Plan::reciprocal_candidates);
    gramPlan.def_readonly("opposite_reciprocal_candidates", &Plan::opposite_reciprocal_candidates);
    gramPlan.def_readonly("accepted_vectors_upper_bound", &Plan::accepted_vectors_upper_bound);
    gramPlan.def_readonly("reciprocal_blocks_upper_bound", &Plan::reciprocal_blocks_upper_bound);
    gramPlan.def_readonly("block_vectors", &Plan::block_vectors);
    gramPlan.def_readonly("retained_output_bytes", &Plan::retained_output_bytes);
    gramPlan.def_readonly("compensation_bytes", &Plan::compensation_bytes);
    gramPlan.def_readonly("reciprocal_buffer_bytes", &Plan::reciprocal_buffer_bytes);
    gramPlan.def_readonly("maximum_ao_panel_bytes", &Plan::maximum_ao_panel_bytes);
    gramPlan.def_readonly("fixed_numerical_workspace_bytes", &Plan::fixed_numerical_workspace_bytes);
    gramPlan.def_readonly("borrowed_basis_numeric_bytes", &Plan::borrowed_basis_numeric_bytes);
    gramPlan.def_readonly("borrowed_cell_bytes", &Plan::borrowed_cell_bytes);
    gramPlan.def_readonly("borrowed_geometry_numeric_bytes", &Plan::borrowed_geometry_numeric_bytes);
    gramPlan.def_readonly("borrowed_numerical_bytes", &Plan::borrowed_numerical_bytes);
    gramPlan.def_readonly("peak_owned_numerical_bytes", &Plan::peak_owned_numerical_bytes);
    gramPlan.def_readonly("control_storage_bytes", &Plan::control_storage_bytes);
    gramPlan.def_readonly("per_replica_inventoried_bytes", &Plan::per_replica_inventoried_bytes);
    gramPlan.def_readonly("required_node_inventoried_bytes", &Plan::required_node_inventoried_bytes);
    gramPlan.def_readonly("reciprocal_traversal_work_units", &Plan::reciprocal_traversal_work_units);
    gramPlan.def_readonly("contraction_work_units", &Plan::contraction_work_units);
    gramPlan.def_readonly("validation_work_units", &Plan::validation_work_units);
    gramPlan.def_readonly("work_units_upper_bound", &Plan::work_units_upper_bound);
    gramPlan.def_readonly("left_panel", &Plan::left_panel);
    gramPlan.def_readonly("right_panel", &Plan::right_panel);
    using Diagnostics = BipoleEwaldGramDiagnostics;
    auto gramDiagnostics = py::class_<Diagnostics>(m, "_BipoleEwaldGramDiagnostics");
    gramDiagnostics.def_readonly("accepted_vectors", &Diagnostics::accepted_vectors);
    gramDiagnostics.def_readonly("evaluated_blocks", &Diagnostics::evaluated_blocks);
    gramDiagnostics.def_readonly("zero_weight_vectors", &Diagnostics::zero_weight_vectors);
    gramDiagnostics.def_readonly("cell_inversion_closed", &Diagnostics::cell_inversion_closed);
    gramDiagnostics.def_readonly("left_cell_inversion_closed", &Diagnostics::left_cell_inversion_closed);
    gramDiagnostics.def_readonly("right_cell_inversion_closed", &Diagnostics::right_cell_inversion_closed);
    gramDiagnostics.def_readonly("reciprocal_conjugacy_audited", &Diagnostics::reciprocal_conjugacy_audited);
    gramDiagnostics.def_readonly("maximum_weight", &Diagnostics::maximum_weight);
    gramDiagnostics.def_readonly("maximum_integral_magnitude", &Diagnostics::maximum_integral_magnitude);
    py::class_<Result>(m, "_BipoleEwaldGramResult")
        .def_property_readonly("citation_numerics", [](const Result&) {
            return std::vector<std::string>{"bipole_ewald_gram"};
        })
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("input_identity_sha256", &Result::input_identity_sha256)
        .def_property_readonly("reciprocal_source_identity_sha256", &Result::reciprocal_source_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &Result::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &Result::symmetry_certified)
        .def("element", &Result::element)
        .def("values_copy", [](const Result& r) {
            const auto& s = r.memory().selection;
            if (s.left_pair_count > 64 || s.right_pair_count > 64)
                throw std::length_error("BIPOLE Gram copy exceeds tiny bounds");
            py::array_t<BipoleDiagnosticComplex> out(
                {py::ssize_t(s.left_pair_count), py::ssize_t(s.right_pair_count)});
            const U n = s.left_pair_count * s.right_pair_count;
            if (n) std::copy_n(r.data(), n, out.mutable_data());
            out.attr("setflags")(false);
            return out;
        });
    auto small_gram = [](const BasisSet& b, const RegularKMesh& mesh,
                          const Selection& s, const Options& o) {
        if (b.nbasis() > 64 || b.nshells() > 128 || mesh.size() > 4096
            || s.left_pair_count > 64 || s.right_pair_count > 64
            || o.reciprocal_block_size > 64)
            throw std::length_error("BIPOLE Gram diagnostic exceeds tiny shape bounds");
    };
    auto bounded_gram = [](const Plan& p) {
        if (p.peak_owned_numerical_bytes > (16U << 20)
            || p.required_node_inventoried_bytes > (128U << 20)
            || p.work_units_upper_bound > 100000000000ULL
            || p.reciprocal_candidates > 65536)
            throw std::length_error("BIPOLE Gram diagnostic exceeds tiny memory/work bounds");
    };
    m.def("_plan_bipole_ewald_gram", [small_gram, bounded_gram](
        const BasisSet& b, const PeriodicSystem& system, const RegularKMesh& mesh,
        const py::array_t<std::int64_t, py::array::c_style>& cells,
        const Selection& selection, const Options& options, const Inventory& inventory,
        const Caps& caps) {
        small_gram(b, mesh, selection, options);
        auto p = plan_bipole_ewald_gram(b, system, mesh, bipole_diagnostic_cells(cells),
                                      selection, options, inventory, caps);
        bounded_gram(p);
        return p;
    }, py::arg("basis"), py::arg("system"), py::arg("mesh"), py::arg("cells").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_make_bipole_ewald_gram", [small_gram, bounded_gram](
        const BasisSet& b, const PeriodicSystem& system, const RegularKMesh& mesh,
        const py::array_t<std::int64_t, py::array::c_style>& cells,
        const Selection& selection, const Options& options, const Inventory& inventory,
        const Caps& caps) {
        small_gram(b, mesh, selection, options);
        const auto cv = bipole_diagnostic_cells(cells);
        auto p = plan_bipole_ewald_gram(b, system, mesh, cv, selection, options, inventory, caps);
        bounded_gram(p);
        // No GIL release or callbacks: all borrowed native/array owners stay live.
        return make_bipole_ewald_gram(b, system, mesh, cv, selection, options, inventory, caps);
    }, py::arg("basis"), py::arg("system"), py::arg("mesh"), py::arg("cells").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_plan_bipole_ewald_product_gram", [small_gram, bounded_gram](
        const BasisSet& b, const PeriodicSystem& system, const RegularKMesh& mesh,
        const py::array_t<std::int64_t, py::array::c_style>& cells,
        const py::array_t<std::int64_t, py::array::c_style>& right_cells,
        const Selection& selection, const Options& options, const Inventory& inventory,
        const Caps& caps) {
        small_gram(b, mesh, selection, options);
        auto p = plan_bipole_ewald_product_gram(b, system, mesh, bipole_diagnostic_cells(cells),
                                      bipole_diagnostic_cells(right_cells), selection, options, inventory, caps);
        bounded_gram(p);
        return p;
    }, py::arg("basis"), py::arg("system"), py::arg("mesh"), py::arg("left_cells").noconvert(), py::arg("right_cells").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));
    m.def("_make_bipole_ewald_product_gram", [small_gram, bounded_gram](
        const BasisSet& b, const PeriodicSystem& system, const RegularKMesh& mesh,
        const py::array_t<std::int64_t, py::array::c_style>& cells,
        const py::array_t<std::int64_t, py::array::c_style>& right_cells,
        const Selection& selection, const Options& options, const Inventory& inventory,
        const Caps& caps) {
        small_gram(b, mesh, selection, options);
        const auto cv = bipole_diagnostic_cells(cells);
        auto p = plan_bipole_ewald_product_gram(b, system, mesh, cv, bipole_diagnostic_cells(right_cells), selection, options, inventory, caps);
        bounded_gram(p);
        // No GIL release or callbacks: all borrowed native/array owners stay live.
        return make_bipole_ewald_product_gram(b, system, mesh, cv, bipole_diagnostic_cells(right_cells), selection, options, inventory, caps);
    }, py::arg("basis"), py::arg("system"), py::arg("mesh"), py::arg("left_cells").noconvert(), py::arg("right_cells").noconvert(),
       py::arg("selection"), py::arg("options"), py::arg("inventory"), py::arg("caps"));

}
