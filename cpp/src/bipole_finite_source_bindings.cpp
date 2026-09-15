// Included after the bounded BIPOLE binding helpers, not compiled alone.
#include <cstring>

void bind_bipole_finite_source(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    py::enum_<BipoleFiniteZeroMode>(m, "_BipoleFiniteZeroMode")
        .value("Unspecified", BipoleFiniteZeroMode::Unspecified)
        .value("G0Omitted", BipoleFiniteZeroMode::G0Omitted);
    using Options = BipoleFiniteSourceOptions;
    auto options = py::class_<Options>(m, "_BipoleFiniteSourceOptions").def(py::init<>());
    options.def_readwrite("omega", &Options::omega);
    options.def_readwrite("reciprocal_energy_cutoff", &Options::reciprocal_energy_cutoff);
    options.def_readwrite("zero_mode", &Options::zero_mode);
    options.def_readwrite("require_image_permutation_closure", &Options::require_image_permutation_closure);
    options.def_readwrite("require_cell_inversion_closure", &Options::require_cell_inversion_closure);
    options.def_readwrite("require_reciprocal_conjugacy", &Options::require_reciprocal_conjugacy);
    using SourceCaps = BipoleFiniteSourceCaps;
    auto sourcecaps = py::class_<SourceCaps>(m, "_BipoleFiniteSourceCaps").def(py::init<>());
    sourcecaps.def_readwrite("maximum_kpoints", &SourceCaps::maximum_kpoints);
    sourcecaps.def_readwrite("maximum_images", &SourceCaps::maximum_images);
    sourcecaps.def_readwrite("maximum_cells", &SourceCaps::maximum_cells);
    sourcecaps.def_readwrite("maximum_context_storage_bytes", &SourceCaps::maximum_context_storage_bytes);
    sourcecaps.def_readwrite("maximum_borrowed_numerical_bytes", &SourceCaps::maximum_borrowed_numerical_bytes);
    sourcecaps.def_readwrite("maximum_work_units", &SourceCaps::maximum_work_units);
    using Domain = BipoleFiniteProductDomain;
    auto domain = py::class_<Domain>(m, "_BipoleFiniteProductDomain").def(py::init<>());
    domain.def_readwrite("left_pair_begin", &Domain::left_pair_begin);
    domain.def_readwrite("left_pair_count", &Domain::left_pair_count);
    domain.def_readwrite("right_pair_begin", &Domain::right_pair_begin);
    domain.def_readwrite("right_pair_count", &Domain::right_pair_count);
    using SourcePlan = BipoleFiniteSourcePlan;
    auto sourceplan = py::class_<SourcePlan>(m, "_BipoleFiniteSourcePlan");
    sourceplan.def_readonly("n_basis", &SourcePlan::n_basis);
    sourceplan.def_readonly("n_shells", &SourcePlan::n_shells);
    sourceplan.def_readonly("n_kpoints", &SourcePlan::n_kpoints);
    sourceplan.def_readonly("image_count", &SourcePlan::image_count);
    sourceplan.def_readonly("cell_count", &SourcePlan::cell_count);
    sourceplan.def_readonly("left_cell_count", &SourcePlan::left_cell_count);
    sourceplan.def_readonly("right_cell_count", &SourcePlan::right_cell_count);
    sourceplan.def_readonly("product_resolved", &SourcePlan::product_resolved);
    sourceplan.def_readonly("product_domain", &SourcePlan::product_domain);
    sourceplan.def_readonly("context_storage_bytes", &SourcePlan::context_storage_bytes);
    sourceplan.def_readonly("borrowed_basis_numeric_bytes", &SourcePlan::borrowed_basis_numeric_bytes);
    sourceplan.def_readonly("borrowed_image_bytes", &SourcePlan::borrowed_image_bytes);
    sourceplan.def_readonly("borrowed_cell_bytes", &SourcePlan::borrowed_cell_bytes);
    sourceplan.def_readonly("borrowed_numerical_bytes", &SourcePlan::borrowed_numerical_bytes);
    sourceplan.def_readonly("work_units_upper_bound", &SourcePlan::work_units_upper_bound);
    sourceplan.def_readonly("basis_census", &SourcePlan::basis_census);
    using Source = BipoleFiniteSource;
    py::class_<Source, std::shared_ptr<Source>>(m, "_BipoleFiniteSource")
        .def_property_readonly("memory", [](const Source& s) { return s.memory(); })
        .def_property_readonly("options", [](const Source& s) { return s.options(); })
        .def_property_readonly("mesh", [](const Source& s) { return s.mesh(); })
        .def_property_readonly("direct_lattice", [](const Source& s) { return s.direct_lattice(); })
        .def_property_readonly("reciprocal_lattice", [](const Source& s) { return s.reciprocal_lattice(); })
        .def_property_readonly("cell_volume", &Source::cell_volume)
        .def_property_readonly("zero_mode_coefficient", &Source::zero_mode_coefficient)
        .def_property_readonly("source_identity_sha256", &Source::source_identity_sha256)
        .def_property_readonly("basis_identity_sha256", &Source::basis_identity_sha256)
        .def_property_readonly("images_identity_sha256", &Source::images_identity_sha256)
        .def_property_readonly("cells_identity_sha256", &Source::cells_identity_sha256)
        .def_property_readonly("right_cells_identity_sha256", &Source::right_cells_identity_sha256)
        .def_property_readonly("product_resolved", &Source::product_resolved)
        .def_property_readonly("product_domain", [](const Source& s) { return s.product_domain(); })
        .def_property_readonly("physical_hamiltonian_certified", &Source::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &Source::symmetry_certified);
    using Inventory = BipoleFinitePanelInventory;
    auto inventory = py::class_<Inventory>(m, "_BipoleFinitePanelInventory").def(py::init<>());
    inventory.def_readwrite("numerical_replicas", &Inventory::numerical_replicas);
    inventory.def_readwrite("external_node_bytes", &Inventory::external_node_bytes);
    inventory.def_readwrite("other_live_numerical_bytes_per_replica", &Inventory::other_live_numerical_bytes_per_replica);
    inventory.def_readwrite("other_live_control_bytes_per_replica", &Inventory::other_live_control_bytes_per_replica);
    inventory.def_readwrite("backend_margin_bytes_per_replica", &Inventory::backend_margin_bytes_per_replica);
    inventory.def_readwrite("reciprocal_block_size", &Inventory::reciprocal_block_size);
    using Caps = BipoleFinitePanelCaps;
    auto caps = py::class_<Caps>(m, "_BipoleFinitePanelCaps").def(py::init<>());
    caps.def_readwrite("source", &Caps::source);
    caps.def_readwrite("short_range", &Caps::short_range);
    caps.def_readwrite("long_range", &Caps::long_range);
    caps.def_readwrite("zero_mode", &Caps::zero_mode);
    caps.def_readwrite("maximum_borrowed_numerical_bytes", &Caps::maximum_borrowed_numerical_bytes);
    caps.def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes);
    caps.def_readwrite("maximum_control_storage_bytes", &Caps::maximum_control_storage_bytes);
    caps.def_readwrite("maximum_worker_bytes", &Caps::maximum_worker_bytes);
    caps.def_readwrite("maximum_node_bytes", &Caps::maximum_node_bytes);
    caps.def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    using Plan = BipoleFinitePanelPlan;
    auto plan = py::class_<Plan>(m, "_BipoleFinitePanelPlan");
    plan.def_readonly("selection", &Plan::selection);
    plan.def_readonly("source", &Plan::source);
    plan.def_readonly("short_range", &Plan::short_range);
    plan.def_readonly("long_range", &Plan::long_range);
    plan.def_readonly("zero_left", &Plan::zero_left);
    plan.def_readonly("zero_right", &Plan::zero_right);
    plan.def_readonly("subtract_zero_mode", &Plan::subtract_zero_mode);
    plan.def_readonly("output_elements", &Plan::output_elements);
    plan.def_readonly("retained_output_bytes", &Plan::retained_output_bytes);
    plan.def_readonly("compensation_bytes", &Plan::compensation_bytes);
    plan.def_readonly("fixed_numeric_workspace_bytes", &Plan::fixed_numeric_workspace_bytes);
    plan.def_readonly("wrapper_control_storage_bytes", &Plan::wrapper_control_storage_bytes);
    plan.def_readonly("zero_mode_workspace_bytes", &Plan::zero_mode_workspace_bytes);
    plan.def_readonly("peak_owned_numerical_bytes", &Plan::peak_owned_numerical_bytes);
    plan.def_readonly("borrowed_numerical_bytes", &Plan::borrowed_numerical_bytes);
    plan.def_readonly("control_storage_bytes", &Plan::control_storage_bytes);
    plan.def_readonly("per_replica_inventoried_bytes", &Plan::per_replica_inventoried_bytes);
    plan.def_readonly("required_node_inventoried_bytes", &Plan::required_node_inventoried_bytes);
    plan.def_readonly("wrapper_work_units", &Plan::wrapper_work_units);
    plan.def_readonly("work_units_upper_bound", &Plan::work_units_upper_bound);
    using Diagnostics = BipoleFinitePanelDiagnostics;
    auto diagnostics = py::class_<Diagnostics>(m, "_BipoleFinitePanelDiagnostics");
    diagnostics.def_readonly("short_range", &Diagnostics::short_range);
    diagnostics.def_readonly("long_range", &Diagnostics::long_range);
    diagnostics.def_readonly("zero_mode_pair_values", &Diagnostics::zero_mode_pair_values);
    diagnostics.def_readonly("image_permutation_support_certified", &Diagnostics::image_permutation_support_certified);
    diagnostics.def_readonly("maximum_zero_mode_magnitude", &Diagnostics::maximum_zero_mode_magnitude);
    diagnostics.def_readonly("maximum_integral_magnitude", &Diagnostics::maximum_integral_magnitude);
    using Result = BipoleFinitePanelResult;
    py::class_<Result>(m, "_BipoleFinitePanelResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("source", &Result::source, py::return_value_policy::reference_internal)
        .def_property_readonly("input_identity_sha256", &Result::input_identity_sha256)
        .def_property_readonly("short_range_payload_identity_sha256", &Result::short_range_payload_identity_sha256)
        .def_property_readonly("long_range_payload_identity_sha256", &Result::long_range_payload_identity_sha256)
        .def_property_readonly("zero_mode_identity_sha256", &Result::zero_mode_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &Result::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &Result::symmetry_certified)
        .def_property_readonly("citation_numerics", [](const Result&) {
            return std::vector<std::string>{"bipole_finite_panel"};
        })
        .def("element", &Result::element)
        .def("values_copy", [](const Result& r) {
            const auto& p=r.memory();
            if (p.selection.left_pair_count>64 || p.selection.right_pair_count>64)
                throw std::length_error("BIPOLE finite panel copy exceeds tiny bounds");
            const auto* first=r.data();
            std::vector<std::complex<double>> copy;
            if (p.output_elements) copy.assign(first,first+p.output_elements);
            return bipole_diagnostic_array(std::move(copy),p.selection.left_pair_count,p.selection.right_pair_count);
        });
    auto images=[](const py::array_t<std::int64_t,py::array::c_style>& a) {
        if (a.ndim()!=3 || a.shape(1)!=3 || a.shape(2)!=3)
            throw std::invalid_argument("BIPOLE finite images must be exact int64 (n,3,3)");
        if (a.shape(0)>64) throw std::length_error("BIPOLE finite diagnostic exceeds tiny image bounds");
        return BipoleErfcImageView{a.data(),U(a.shape(0)),U(a.size())};
    };
    auto tiny_basis=[](const BasisSet& b, const RegularKMesh& mesh) {
        if (b.nbasis()>64 || b.nshells()>128 || mesh.size()>4096)
            throw std::length_error("BIPOLE finite diagnostic exceeds tiny basis/mesh bounds");
    };
    auto tiny_source=[](const SourcePlan& p) {
        if (p.context_storage_bytes>(1U<<20) || p.borrowed_numerical_bytes>(16U<<20) || p.work_units_upper_bound>100000000000ULL)
            throw std::length_error("BIPOLE finite source diagnostic exceeds tiny memory/work bounds");
    };
    auto tiny_panel=[](const Plan& p) {
        if (p.selection.left_pair_count>64 || p.selection.right_pair_count>64 ||
            p.peak_owned_numerical_bytes>(128U<<20) || p.required_node_inventoried_bytes>(512U<<20) ||
            p.work_units_upper_bound>100000000000ULL || p.long_range.reciprocal_candidates>100000 ||
            p.short_range.raw.shell_quartet_calls_upper_bound>100000)
            throw std::length_error("BIPOLE finite panel diagnostic exceeds tiny memory/work bounds");
    };
    m.def("_plan_bipole_finite_source", [images,tiny_basis,tiny_source](const BasisSet& b,
        const RegularKMesh& mesh, const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells, const Options& o, const SourceCaps& c) {
        tiny_basis(b,mesh);
        auto p=plan_bipole_finite_source(b,mesh,images(a),bipole_diagnostic_cells(cells),o,c);
        tiny_source(p); return p;
    },py::arg("basis"),py::arg("mesh"),py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("options"),py::arg("caps"));
    m.def("_make_bipole_finite_source", [images,tiny_basis,tiny_source](const BasisSet& b,
        const PeriodicSystem& system, const RegularKMesh& mesh, const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells, const Options& o, const SourceCaps& c) {
        tiny_basis(b,mesh); const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        tiny_source(plan_bipole_finite_source(b,mesh,iv,cv,o,c));
        return make_bipole_finite_source(b,system,mesh,iv,cv,o,c);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("options"),py::arg("caps"));
    m.def("_plan_bipole_finite_product_source", [images,tiny_basis,tiny_source](const BasisSet& b,
        const RegularKMesh& mesh, const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::int64_t,py::array::c_style>& right_cells, const Domain& d, const Options& o, const SourceCaps& c) {
        tiny_basis(b,mesh);
        auto p=plan_bipole_finite_product_source(b,mesh,images(a),bipole_diagnostic_cells(cells),bipole_diagnostic_cells(right_cells),d,o,c);
        tiny_source(p); return p;
    },py::arg("basis"),py::arg("mesh"),py::arg("images").noconvert(),py::arg("left_cells").noconvert(),py::arg("right_cells").noconvert(),py::arg("domain"),py::arg("options"),py::arg("caps"));
    m.def("_make_bipole_finite_product_source", [images,tiny_basis,tiny_source](const BasisSet& b,
        const PeriodicSystem& system, const RegularKMesh& mesh, const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::int64_t,py::array::c_style>& right_cells, const Domain& d, const Options& o, const SourceCaps& c) {
        tiny_basis(b,mesh); const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        tiny_source(plan_bipole_finite_product_source(b,mesh,iv,cv,bipole_diagnostic_cells(right_cells),d,o,c));
        return make_bipole_finite_product_source(b,system,mesh,iv,cv,bipole_diagnostic_cells(right_cells),d,o,c);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("images").noconvert(),py::arg("left_cells").noconvert(),py::arg("right_cells").noconvert(),py::arg("domain"),py::arg("options"),py::arg("caps"));
    auto overlapcaps=py::class_<BipoleFiniteOverlapCaps>(m,"_BipoleFiniteOverlapCaps").def(py::init<>());
    overlapcaps.def_readwrite("source",&BipoleFiniteOverlapCaps::source);
    overlapcaps.def_readwrite("fourier",&BipoleFiniteOverlapCaps::fourier);
    overlapcaps.def_readwrite("maximum_per_replica_inventoried_bytes",&BipoleFiniteOverlapCaps::maximum_per_replica_inventoried_bytes);
    overlapcaps.def_readwrite("maximum_node_inventoried_bytes",&BipoleFiniteOverlapCaps::maximum_node_inventoried_bytes);
    overlapcaps.def_readwrite("maximum_work_units",&BipoleFiniteOverlapCaps::maximum_work_units);
    auto overlapplan=py::class_<BipoleFiniteOverlapPlan>(m,"_BipoleFiniteOverlapPlan");
    overlapplan.def_readonly("source",&BipoleFiniteOverlapPlan::source);
    overlapplan.def_readonly("fourier",&BipoleFiniteOverlapPlan::fourier);
    overlapplan.def_readonly("k_index",&BipoleFiniteOverlapPlan::k_index);
    overlapplan.def_readonly("overlap_elements",&BipoleFiniteOverlapPlan::overlap_elements);
    overlapplan.def_readonly("borrowed_overlap_bytes",&BipoleFiniteOverlapPlan::borrowed_overlap_bytes);
    overlapplan.def_readonly("fixed_workspace_bytes",&BipoleFiniteOverlapPlan::fixed_workspace_bytes);
    overlapplan.def_readonly("per_replica_inventoried_bytes",&BipoleFiniteOverlapPlan::per_replica_inventoried_bytes);
    overlapplan.def_readonly("required_node_inventoried_bytes",&BipoleFiniteOverlapPlan::required_node_inventoried_bytes);
    overlapplan.def_readonly("work_units_upper_bound",&BipoleFiniteOverlapPlan::work_units_upper_bound);
    auto overlapdiagnostics=py::class_<BipoleFiniteOverlapDiagnostics>(m,"_BipoleFiniteOverlapDiagnostics");
    overlapdiagnostics.def_readonly("memory",&BipoleFiniteOverlapDiagnostics::memory);
    overlapdiagnostics.def_readonly("absolute_tolerance",&BipoleFiniteOverlapDiagnostics::absolute_tolerance);
    overlapdiagnostics.def_readonly("maximum_direct_residual",&BipoleFiniteOverlapDiagnostics::maximum_direct_residual);
    overlapdiagnostics.def_readonly("maximum_dual_residual",&BipoleFiniteOverlapDiagnostics::maximum_dual_residual);
    overlapdiagnostics.def_readonly("direct_matches",&BipoleFiniteOverlapDiagnostics::direct_matches);
    overlapdiagnostics.def_readonly("dual_matches",&BipoleFiniteOverlapDiagnostics::dual_matches);
    overlapdiagnostics.def_readonly("source_identity_sha256",&BipoleFiniteOverlapDiagnostics::source_identity_sha256);
    overlapdiagnostics.def_readonly("overlap_identity_sha256",&BipoleFiniteOverlapDiagnostics::overlap_identity_sha256);
    overlapdiagnostics.def_property_readonly("physical_hamiltonian_certified",&BipoleFiniteOverlapDiagnostics::physical_hamiltonian_certified);
    overlapdiagnostics.def_property_readonly("symmetry_certified",&BipoleFiniteOverlapDiagnostics::symmetry_certified);
    const auto overlap_view=[](const py::array_t<std::complex<double>,py::array::c_style>& a, U n) {
        if (a.ndim()!=2 || static_cast<U>(a.shape(0))!=n || static_cast<U>(a.shape(1))!=n)
            throw std::invalid_argument("BIPOLE overlap requires an Nao by Nao panel");
        return a.data();
    };
    const auto tiny_overlap=[](const BipoleFiniteOverlapPlan& p) {
        if (p.required_node_inventoried_bytes>(128U<<20) || p.work_units_upper_bound>1000000000ULL)
            throw std::length_error("BIPOLE overlap diagnostic exceeds tiny memory/work bounds");
    };
    m.def("_plan_bipole_finite_overlap", [images,tiny_basis,overlap_view,tiny_overlap](
        const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::complex<double>,py::array::c_style>& overlap, U k, double tolerance,
        const PeriodicAOBlochTransportInventory& i, const BipoleFiniteOverlapCaps& c) {
        tiny_basis(b,source.mesh());
        const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        const auto* data=overlap_view(overlap,b.nbasis());
        const auto count=static_cast<U>(overlap.size());
        const auto p=plan_bipole_finite_overlap(source,b,iv,cv,data,count,k,tolerance,i,c);
        tiny_overlap(p);
        // GIL held; the caller keeps borrowed source/S payloads immutable.
        return p;
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),
      py::arg("cells").noconvert(),py::arg("overlap").noconvert(),py::arg("k_index"),
      py::arg("absolute_tolerance"),py::arg("inventory"),py::arg("caps"));
    m.def("_audit_bipole_finite_overlap", [images,tiny_basis,overlap_view,tiny_overlap](
        const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::complex<double>,py::array::c_style>& overlap, U k, double tolerance,
        const PeriodicAOBlochTransportInventory& i, const BipoleFiniteOverlapCaps& c) {
        tiny_basis(b,source.mesh());
        const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        const auto* data=overlap_view(overlap,b.nbasis());
        const auto count=static_cast<U>(overlap.size());
        const auto p=plan_bipole_finite_overlap(source,b,iv,cv,data,count,k,tolerance,i,c);
        tiny_overlap(p);
        // GIL held; the caller keeps borrowed source/S payloads immutable.
        return audit_bipole_finite_overlap(source,b,iv,cv,data,count,k,tolerance,i,c);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),
      py::arg("cells").noconvert(),py::arg("overlap").noconvert(),py::arg("k_index"),
      py::arg("absolute_tolerance"),py::arg("inventory"),py::arg("caps"));
    using SupportCaps = BipoleFiniteSupportCaps;
    py::class_<SupportCaps>(m, "_BipoleFiniteSupportCaps").def(py::init<>())
        .def_readwrite("source", &SupportCaps::source)
        .def_readwrite("maximum_label_comparisons", &SupportCaps::maximum_label_comparisons)
        .def_readwrite("maximum_inventoried_bytes", &SupportCaps::maximum_inventoried_bytes)
        .def_readwrite("maximum_work_units", &SupportCaps::maximum_work_units);
    using SupportPlan = BipoleFiniteSupportPlan;
    py::class_<SupportPlan>(m, "_BipoleFiniteSupportPlan")
        .def_readonly("source", &SupportPlan::source)
        .def_readonly("label_comparisons", &SupportPlan::label_comparisons)
        .def_readonly("fixed_workspace_bytes", &SupportPlan::fixed_workspace_bytes)
        .def_readonly("inventoried_bytes", &SupportPlan::inventoried_bytes)
        .def_readonly("work_units_upper_bound", &SupportPlan::work_units_upper_bound);
    using SupportDiagnostics = BipoleFiniteSupportDiagnostics;
    py::class_<SupportDiagnostics>(m, "_BipoleFiniteSupportDiagnostics")
        .def_readonly("memory", &SupportDiagnostics::memory)
        .def_readonly("left_cells_closed", &SupportDiagnostics::left_cells_closed)
        .def_readonly("right_cells_closed", &SupportDiagnostics::right_cells_closed)
        .def_readonly("images_closed", &SupportDiagnostics::images_closed)
        .def_readonly("first_left_cell", &SupportDiagnostics::first_left_cell)
        .def_readonly("first_right_cell", &SupportDiagnostics::first_right_cell)
        .def_readonly("first_image", &SupportDiagnostics::first_image)
        .def_readonly("label_comparisons", &SupportDiagnostics::label_comparisons)
        .def_readonly("source_identity_sha256", &SupportDiagnostics::source_identity_sha256)
        .def_readonly("action_identity_sha256", &SupportDiagnostics::action_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &SupportDiagnostics::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &SupportDiagnostics::symmetry_certified);
    m.def("_plan_bipole_finite_support_action", [images,tiny_basis](const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells, const SupportCaps& c) {
        tiny_basis(b,source.mesh());
        return plan_bipole_finite_support_action(source,b,images(a),bipole_diagnostic_cells(cells),c);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("caps"));
    m.def("_audit_bipole_finite_support_action", [images,tiny_basis](const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::int32_t,py::array::c_style>& rotation,
        const py::array_t<std::int64_t,py::array::c_style>& shifts, const SupportCaps& c) {
        tiny_basis(b,source.mesh());
        const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        (void)plan_bipole_finite_support_action(source,b,iv,cv,c);
        if (rotation.ndim()!=2 || rotation.shape(0)!=3 || rotation.shape(1)!=3 ||
            shifts.ndim()!=2 || shifts.shape(0)!=4 || shifts.shape(1)!=3)
            throw std::invalid_argument("BIPOLE finite support requires int32 (3,3) rotation and int64 (4,3) shifts");
        BipoleFiniteSupportAction action;
        // Only these fixed descriptors are copied; unaligned NumPy buffers
        // remain safe, and neither labels nor AO numerical data are copied.
        std::memcpy(action.rotation.data(),rotation.data(),sizeof(action.rotation));
        std::memcpy(action.atom_shifts.data(),shifts.data(),sizeof(action.atom_shifts));
        return audit_bipole_finite_support_action(source,b,iv,cv,action,c);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("cells").noconvert(),
       py::arg("rotation").noconvert(),py::arg("atom_shifts").noconvert(),py::arg("caps"));
    using MappedSelection = BipoleFiniteMappedSupportSelection;
    py::class_<MappedSelection>(m, "_BipoleFiniteMappedSupportSelection").def(py::init<>())
        .def_readwrite("source_shells", &MappedSelection::source_shells)
        .def_readwrite("source_k_index", &MappedSelection::source_k_index)
        .def_readwrite("time_reversal", &MappedSelection::time_reversal);
    using MappedCaps = BipoleFiniteMappedSupportCaps;
    py::class_<MappedCaps>(m, "_BipoleFiniteMappedSupportCaps").def(py::init<>())
        .def_readwrite("support", &MappedCaps::support)
        .def_readwrite("transport", &MappedCaps::transport)
        .def_readwrite("maximum_per_replica_inventoried_bytes", &MappedCaps::maximum_per_replica_inventoried_bytes)
        .def_readwrite("maximum_node_inventoried_bytes", &MappedCaps::maximum_node_inventoried_bytes)
        .def_readwrite("maximum_work_units", &MappedCaps::maximum_work_units);
    using MappedPlan = BipoleFiniteMappedSupportPlan;
    py::class_<MappedPlan>(m, "_BipoleFiniteMappedSupportPlan")
        .def_readonly("support", &MappedPlan::support)
        .def_readonly("transport", &MappedPlan::transport)
        .def_readonly("fixed_workspace_bytes", &MappedPlan::fixed_workspace_bytes)
        .def_readonly("per_replica_inventoried_bytes", &MappedPlan::per_replica_inventoried_bytes)
        .def_readonly("required_node_inventoried_bytes", &MappedPlan::required_node_inventoried_bytes)
        .def_readonly("work_units_upper_bound", &MappedPlan::work_units_upper_bound);
    using MappedDiagnostics = BipoleFiniteMappedSupportDiagnostics;
    py::class_<MappedDiagnostics>(m, "_BipoleFiniteMappedSupportDiagnostics")
        .def_readonly("memory", &MappedDiagnostics::memory)
        .def_readonly("support", &MappedDiagnostics::support)
        .def_readonly("transport", &MappedDiagnostics::transport)
        .def_readonly("source_shells", &MappedDiagnostics::source_shells)
        .def_readonly("destination_shells", &MappedDiagnostics::destination_shells)
        .def_readonly("source_atoms", &MappedDiagnostics::source_atoms)
        .def_readonly("destination_atoms", &MappedDiagnostics::destination_atoms)
        .def_readonly("atom_shifts", &MappedDiagnostics::atom_shifts)
        .def_readonly("mapping_identity_sha256", &MappedDiagnostics::mapping_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &MappedDiagnostics::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &MappedDiagnostics::symmetry_certified);
    const auto tiny_mapped=[](const PeriodicSystem& system, const MappedPlan& p) {
        if (system.unit_cell.size()>32 || p.required_node_inventoried_bytes>(128U<<20) ||
            p.work_units_upper_bound>1000000000ULL)
            throw std::length_error("BIPOLE mapped support diagnostic exceeds tiny memory/work bounds");
    };
    m.def("_plan_bipole_finite_mapped_support", [images,tiny_basis,tiny_mapped](
        const Source& source, const BasisSet& b, const PeriodicSystem& system, const SymmetryOp& op,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells, const MappedSelection& s,
        const PeriodicAOBlochTransportOptions& o, const PeriodicAOBlochTransportInventory& i,
        const MappedCaps& c) {
        tiny_basis(b,source.mesh());
        if (system.unit_cell.size()>32) throw std::length_error("BIPOLE mapped support tiny atom bound");
        auto p=plan_bipole_finite_mapped_support(source,b,system,op,images(a),bipole_diagnostic_cells(cells),s,o,i,c);
        tiny_mapped(system,p); return p;
    },py::arg("source"),py::arg("basis"),py::arg("system"),py::arg("operation"),
       py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("selection"),
       py::arg("options"),py::arg("inventory"),py::arg("caps"));
    m.def("_audit_bipole_finite_mapped_support", [images,tiny_basis,tiny_mapped](
        const Source& source, const BasisSet& b, const PeriodicSystem& system, const SymmetryOp& op,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::int64_t,py::array::c_style>& cells, const MappedSelection& s,
        const PeriodicAOBlochTransportOptions& o, const PeriodicAOBlochTransportInventory& i,
        const MappedCaps& c) {
        tiny_basis(b,source.mesh());
        if (system.unit_cell.size()>32) throw std::length_error("BIPOLE mapped support tiny atom bound");
        const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        tiny_mapped(system,plan_bipole_finite_mapped_support(source,b,system,op,iv,cv,s,o,i,c));
        // Keep the GIL; there are no conversions or callbacks between the
        // validators. Caller-owned borrowed payloads must remain immutable.
        return audit_bipole_finite_mapped_support(source,b,system,op,iv,cv,s,o,i,c);
    },py::arg("source"),py::arg("basis"),py::arg("system"),py::arg("operation"),
       py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("selection"),
       py::arg("options"),py::arg("inventory"),py::arg("caps"));
    m.def("_plan_bipole_finite_panel", [images,tiny_basis,tiny_panel](const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a, const py::array_t<std::int64_t,py::array::c_style>& cells,
        const BipoleEwaldGramSelection& s, const Inventory& i, const Caps& c) {
        tiny_basis(b,source.mesh());
        if (s.left_pair_count>64 || s.right_pair_count>64 || i.reciprocal_block_size>64)
            throw std::length_error("BIPOLE finite diagnostic exceeds tiny pair/block bounds");
        auto p=plan_bipole_finite_panel(source,b,images(a),bipole_diagnostic_cells(cells),s,i,c);
        tiny_panel(p); return p;
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("selection"),py::arg("inventory"),py::arg("caps"));
    m.def("_make_bipole_finite_panel", [images,tiny_basis,tiny_panel](std::shared_ptr<Source> source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a, const py::array_t<std::int64_t,py::array::c_style>& cells,
        const BipoleEwaldGramSelection& s, const Inventory& i, const Caps& c) {
        if (!source) throw std::invalid_argument("BIPOLE finite panel needs a source owner");
        tiny_basis(b,source->mesh());
        if (s.left_pair_count>64 || s.right_pair_count>64 || i.reciprocal_block_size>64)
            throw std::length_error("BIPOLE finite diagnostic exceeds tiny pair/block bounds");
        const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        tiny_panel(plan_bipole_finite_panel(*source,b,iv,cv,s,i,c));
        // GIL held, no callbacks: borrowed owners stay immutable and live.
        return make_bipole_finite_panel(std::move(source),b,iv,cv,s,i,c);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("cells").noconvert(),py::arg("selection"),py::arg("inventory"),py::arg("caps"));

    m.def("_plan_bipole_finite_product_panel", [images,tiny_basis,tiny_panel](const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a, const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::int64_t,py::array::c_style>& right_cells,
        const BipoleEwaldGramSelection& s, const Inventory& i, const Caps& c) {
        tiny_basis(b,source.mesh());
        if (s.left_pair_count>64 || s.right_pair_count>64 || i.reciprocal_block_size>64)
            throw std::length_error("BIPOLE finite diagnostic exceeds tiny pair/block bounds");
        auto p=plan_bipole_finite_product_panel(source,b,images(a),bipole_diagnostic_cells(cells),bipole_diagnostic_cells(right_cells),s,i,c);
        tiny_panel(p); return p;
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("left_cells").noconvert(),py::arg("right_cells").noconvert(),py::arg("selection"),py::arg("inventory"),py::arg("caps"));
    m.def("_make_bipole_finite_product_panel", [images,tiny_basis,tiny_panel](std::shared_ptr<Source> source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a, const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::int64_t,py::array::c_style>& right_cells,
        const BipoleEwaldGramSelection& s, const Inventory& i, const Caps& c) {
        if (!source) throw std::invalid_argument("BIPOLE finite panel needs a source owner");
        tiny_basis(b,source->mesh());
        if (s.left_pair_count>64 || s.right_pair_count>64 || i.reciprocal_block_size>64)
            throw std::length_error("BIPOLE finite diagnostic exceeds tiny pair/block bounds");
        const auto iv=images(a); const auto cv=bipole_diagnostic_cells(cells);
        tiny_panel(plan_bipole_finite_product_panel(*source,b,iv,cv,bipole_diagnostic_cells(right_cells),s,i,c));
        // GIL held, no callbacks: borrowed owners stay immutable and live.
        return make_bipole_finite_product_panel(std::move(source),b,iv,cv,bipole_diagnostic_cells(right_cells),s,i,c);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("left_cells").noconvert(),py::arg("right_cells").noconvert(),py::arg("selection"),py::arg("inventory"),py::arg("caps"));

    auto jks=py::class_<BipoleFiniteJKSelection>(m, "_BipoleFiniteJKSelection").def(py::init<>());
    jks.def_readwrite("target_k_index", &BipoleFiniteJKSelection::target_k_index);
    jks.def_readwrite("pair_begin", &BipoleFiniteJKSelection::pair_begin);
    jks.def_readwrite("pair_count", &BipoleFiniteJKSelection::pair_count);
    jks.def_readwrite("density_pair_block_size", &BipoleFiniteJKSelection::density_pair_block_size);
    auto jkc=py::class_<BipoleFiniteJKCaps>(m, "_BipoleFiniteJKCaps").def(py::init<>());
    jkc.def_readwrite("panel", &BipoleFiniteJKCaps::panel);
    jkc.def_readwrite("maximum_panel_calls", &BipoleFiniteJKCaps::maximum_panel_calls);
    jkc.def_readwrite("maximum_work_units", &BipoleFiniteJKCaps::maximum_work_units);
    auto jkp=py::class_<BipoleFiniteJKPlan>(m, "_BipoleFiniteJKPlan");
    jkp.def_readonly("selection", &BipoleFiniteJKPlan::selection);
    jkp.def_readonly("n_basis", &BipoleFiniteJKPlan::n_basis);
    jkp.def_readonly("n_kpoints", &BipoleFiniteJKPlan::n_kpoints);
    jkp.def_readonly("density_bytes", &BipoleFiniteJKPlan::density_bytes);
    jkp.def_readonly("panel_calls", &BipoleFiniteJKPlan::panel_calls);
    jkp.def_readonly("contracted_terms", &BipoleFiniteJKPlan::contracted_terms);
    jkp.def_readonly("output_elements", &BipoleFiniteJKPlan::output_elements);
    jkp.def_readonly("retained_output_bytes", &BipoleFiniteJKPlan::retained_output_bytes);
    jkp.def_readonly("compensation_bytes", &BipoleFiniteJKPlan::compensation_bytes);
    jkp.def_readonly("fixed_numeric_workspace_bytes", &BipoleFiniteJKPlan::fixed_numeric_workspace_bytes);
    jkp.def_readonly("wrapper_control_storage_bytes", &BipoleFiniteJKPlan::wrapper_control_storage_bytes);
    jkp.def_readonly("borrowed_numerical_bytes", &BipoleFiniteJKPlan::borrowed_numerical_bytes);
    jkp.def_readonly("peak_owned_numerical_bytes", &BipoleFiniteJKPlan::peak_owned_numerical_bytes);
    jkp.def_readonly("control_storage_bytes", &BipoleFiniteJKPlan::control_storage_bytes);
    jkp.def_readonly("per_replica_inventoried_bytes", &BipoleFiniteJKPlan::per_replica_inventoried_bytes);
    jkp.def_readonly("required_node_inventoried_bytes", &BipoleFiniteJKPlan::required_node_inventoried_bytes);
    jkp.def_readonly("wrapper_work_units", &BipoleFiniteJKPlan::wrapper_work_units);
    jkp.def_readonly("work_units_upper_bound", &BipoleFiniteJKPlan::work_units_upper_bound);
    using JKResult=BipoleFiniteJKResult;
    py::class_<JKResult>(m, "_BipoleFiniteJKResult")
        .def_property_readonly("memory", [](const JKResult& r) { return r.memory(); })
        .def_property_readonly("source", &JKResult::source, py::return_value_policy::reference_internal)
        .def_property_readonly("input_identity_sha256", &JKResult::input_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &JKResult::payload_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified", &JKResult::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified", &JKResult::symmetry_certified)
        .def_property_readonly("citation_numerics", [](const JKResult&) {
            return std::vector<std::string>{"bipole_finite_jk"};
        })
        .def("element", &JKResult::element)
        .def("values_copy", [](const JKResult& r) {
            const auto n=r.memory().selection.pair_count;
            if (n>64) throw std::length_error("BIPOLE finite J/K copy exceeds tiny bounds");
            const auto* first=r.data();
            return bipole_diagnostic_array(std::vector<std::complex<double>>(first,first+2*n),2,n);
        });
    auto jk_inputs=[images](const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a,
        const py::array_t<std::complex<double>,py::array::c_style>& d,
        const BipoleFiniteJKSelection& s, const Inventory& i, BipoleFiniteJKCaps c) {
        if (b.nbasis()>8 || b.nshells()>32 || source.mesh().size()>64 ||
            s.pair_count>64 || s.density_pair_block_size>64 || i.reciprocal_block_size>64)
            throw std::length_error("BIPOLE finite J/K diagnostic exceeds tiny bounds");
        (void)images(a);
        if (d.ndim()!=3 || d.shape(0)!=py::ssize_t(source.mesh().size()) ||
            d.shape(1)!=py::ssize_t(b.nbasis()) || d.shape(2)!=py::ssize_t(b.nbasis()))
            throw std::invalid_argument("BIPOLE finite J/K density must be exact complex128 (Nk,Nao,Nao)");
        // Hard private witness bounds apply BEFORE any planning traversal or
        // payload scan, even if a caller inflates all explicit resource caps.
        c.maximum_panel_calls=std::min<U>(c.maximum_panel_calls,32768);
        c.maximum_work_units=std::min<U>(c.maximum_work_units,100000000000ULL);
        c.panel.maximum_owned_numerical_bytes=std::min<U>(c.panel.maximum_owned_numerical_bytes,128U<<20);
        c.panel.maximum_borrowed_numerical_bytes=std::min<U>(c.panel.maximum_borrowed_numerical_bytes,16U<<20);
        c.panel.maximum_control_storage_bytes=std::min<U>(c.panel.maximum_control_storage_bytes,32U<<20);
        c.panel.maximum_worker_bytes=std::min<U>(c.panel.maximum_worker_bytes,256U<<20);
        c.panel.maximum_node_bytes=std::min<U>(c.panel.maximum_node_bytes,512U<<20);
        c.panel.short_range.raw.maximum_shell_quartet_calls=std::min<U>(c.panel.short_range.raw.maximum_shell_quartet_calls,100000);
        c.panel.long_range.maximum_reciprocal_candidates=std::min<U>(c.panel.long_range.maximum_reciprocal_candidates,100000);
        return c;
    };
    m.def("_plan_bipole_finite_jk", [images,jk_inputs](const Source& source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a, const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::complex<double>,py::array::c_style>& d,
        const BipoleFiniteJKSelection& s, const Inventory& i, const BipoleFiniteJKCaps& c) {
        const auto caps=jk_inputs(source,b,a,d,s,i,c);
        const BipoleFiniteDensityView density{d.data(),U(d.shape(0)),U(d.shape(1)),U(d.size())};
        return plan_bipole_finite_jk(source,b,images(a),bipole_diagnostic_cells(cells),density,s,i,caps);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("cells").noconvert(),
       py::arg("density").noconvert(),py::arg("selection"),py::arg("inventory"),py::arg("caps"));
    m.def("_make_bipole_finite_jk", [images,jk_inputs](std::shared_ptr<Source> source, const BasisSet& b,
        const py::array_t<std::int64_t,py::array::c_style>& a, const py::array_t<std::int64_t,py::array::c_style>& cells,
        const py::array_t<std::complex<double>,py::array::c_style>& d,
        const BipoleFiniteJKSelection& s, const Inventory& i, const BipoleFiniteJKCaps& c) {
        if (!source) throw std::invalid_argument("BIPOLE finite J/K needs a source owner");
        const auto caps=jk_inputs(*source,b,a,d,s,i,c);
        const BipoleFiniteDensityView density{d.data(),U(d.shape(0)),U(d.shape(1)),U(d.size())};
        // GIL retained, no callbacks or implicit copies of borrowed arrays.
        return make_bipole_finite_jk(std::move(source),b,images(a),bipole_diagnostic_cells(cells),density,s,i,caps);
    },py::arg("source"),py::arg("basis"),py::arg("images").noconvert(),py::arg("cells").noconvert(),
       py::arg("density").noconvert(),py::arg("selection"),py::arg("inventory"),py::arg("caps"));


    using StreamCaps=BipoleProductJKStreamCaps;
    auto sc=py::class_<StreamCaps>(m,"_BipoleProductJKStreamCaps").def(py::init<>());
    sc.def_readwrite("maximum_panel_calls",&StreamCaps::maximum_panel_calls);
    sc.def_readwrite("maximum_density_bytes",&StreamCaps::maximum_density_bytes);
    sc.def_readwrite("maximum_state_bytes",&StreamCaps::maximum_state_bytes);
    sc.def_readwrite("maximum_panel_inventoried_bytes",&StreamCaps::maximum_panel_inventoried_bytes);
    sc.def_readwrite("maximum_panel_work_units",&StreamCaps::maximum_panel_work_units);
    sc.def_readwrite("maximum_node_bytes",&StreamCaps::maximum_node_bytes);
    sc.def_readwrite("maximum_work_units",&StreamCaps::maximum_work_units);
    using StreamPlan=BipoleProductJKStreamPlan;
    auto sp=py::class_<StreamPlan>(m,"_BipoleProductJKStreamPlan");
    sp.def_readonly("n_basis",&StreamPlan::n_basis);
    sp.def_readonly("n_kpoints",&StreamPlan::n_kpoints);
    sp.def_readonly("target_k_index",&StreamPlan::target_k_index);
    sp.def_readonly("quartet_count",&StreamPlan::quartet_count);
    sp.def_readonly("panel_calls",&StreamPlan::panel_calls);
    sp.def_readonly("density_bytes",&StreamPlan::density_bytes);
    sp.def_readonly("output_elements",&StreamPlan::output_elements);
    sp.def_readonly("state_bytes",&StreamPlan::state_bytes);
    sp.def_readonly("per_replica_inventoried_bytes",&StreamPlan::per_replica_inventoried_bytes);
    sp.def_readonly("required_node_inventoried_bytes",&StreamPlan::required_node_inventoried_bytes);
    sp.def_readonly("wrapper_work_units",&StreamPlan::wrapper_work_units);
    sp.def_readonly("work_units_upper_bound",&StreamPlan::work_units_upper_bound);
    using Stream=BipoleProductJKStream;
    py::class_<Stream,std::shared_ptr<Stream>>(m,"_BipoleProductJKStream")
        .def_property_readonly("memory",[](const Stream& r) { return r.memory(); })
        .def_property_readonly("declaration",&Stream::declaration,py::return_value_policy::reference_internal)
        .def_property_readonly("accepted_panels",&Stream::accepted_panels)
        .def_property_readonly("complete",&Stream::complete)
        .def_property_readonly("finalized",&Stream::finalized)
        .def_property_readonly("declared_policy_identity_sha256",&Stream::declared_policy_identity_sha256)
        .def_property_readonly("input_identity_sha256",&Stream::input_identity_sha256)
        .def_property_readonly("payload_identity_sha256",&Stream::payload_identity_sha256)
        .def_property_readonly("physical_hamiltonian_certified",&Stream::physical_hamiltonian_certified)
        .def_property_readonly("symmetry_certified",&Stream::symmetry_certified)
        .def_property_readonly("citation_numerics",[](const Stream&) {
            return std::vector<std::string>{"bipole_finite_jk"};
        })
        .def("next_selection",&Stream::next_selection)
        .def("consume",&Stream::consume)
        .def("finalize",&Stream::finalize)
        .def("element",&Stream::element)
        .def("values_copy",[](const Stream& r) {
            const U n2=r.memory().n_basis*r.memory().n_basis;
            if (n2>64) throw std::length_error("BIPOLE product J/K stream copy exceeds tiny bounds");
            const auto* first=r.data();
            return bipole_diagnostic_array(std::vector<std::complex<double>>(first,first+2*n2),2,n2);
        });
    auto stream_inputs=[](const Source& source,
        const py::array_t<std::complex<double>,py::array::c_style>& d, StreamCaps c) {
        const U n=source.memory().n_basis, nk=source.mesh().size();
        if (n>8 || source.memory().n_shells>32 || nk>64)
            throw std::length_error("BIPOLE product J/K stream diagnostic exceeds tiny bounds");
        if (d.ndim()!=3 || d.shape(0)!=py::ssize_t(nk) || d.shape(1)!=py::ssize_t(n) || d.shape(2)!=py::ssize_t(n))
            throw std::invalid_argument("BIPOLE product J/K stream density must be exact complex128 (Nk,Nao,Nao)");
        c.maximum_panel_calls=std::min<U>(c.maximum_panel_calls,32768);
        c.maximum_density_bytes=std::min<U>(c.maximum_density_bytes,16U<<20);
        c.maximum_state_bytes=std::min<U>(c.maximum_state_bytes,128U<<20);
        c.maximum_panel_inventoried_bytes=std::min<U>(c.maximum_panel_inventoried_bytes,512U<<20);
        c.maximum_panel_work_units=std::min<U>(c.maximum_panel_work_units,1000000000ULL);
        c.maximum_node_bytes=std::min<U>(c.maximum_node_bytes,U{1}<<30);
        c.maximum_work_units=std::min<U>(c.maximum_work_units,100000000000ULL);
        return c;
    };
    m.def("_plan_bipole_product_jk_stream",[stream_inputs](const Source& source,
        const py::array_t<std::complex<double>,py::array::c_style>& d,U target,
        const Inventory& i,const StreamCaps& c) {
        const auto caps=stream_inputs(source,d,c);
        return plan_bipole_product_jk_stream(source,{d.data(),U(d.shape(0)),U(d.shape(1)),U(d.size())},target,i,caps);
    },py::arg("declaration"),py::arg("density").noconvert(),py::arg("target_k_index"),py::arg("inventory"),py::arg("caps"));
    m.def("_make_bipole_product_jk_stream",[stream_inputs](std::shared_ptr<Source> source,
        const py::array_t<std::complex<double>,py::array::c_style>& d,U target,
        const py::str& policy,const Inventory& i,const StreamCaps& c) {
        if (!source) throw std::invalid_argument("BIPOLE product J/K stream needs a declaration");
        const auto caps=stream_inputs(*source,d,c);
        if (py::len(policy)!=64) throw std::invalid_argument("BIPOLE product J/K stream needs a SHA256 policy declaration");
        return make_bipole_product_jk_stream(std::move(source),{d.data(),U(d.shape(0)),U(d.shape(1)),U(d.size())},
            target,policy.cast<std::string>(),i,caps);
    },py::arg("declaration"),py::arg("density").noconvert(),py::arg("target_k_index"),
       py::arg("declared_policy_identity").noconvert(),py::arg("inventory"),py::arg("caps"));

}
