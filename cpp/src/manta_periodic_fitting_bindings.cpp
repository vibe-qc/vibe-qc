#include "vibeqc/manta_periodic_gdf_pair_tree.hpp"
#include "vibeqc/manta_periodic_fitting.hpp"
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void check(const py::array& a,int dimensions) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=dimensions ||
       !(a.flags() & (py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic fitting requires C- or F-contiguous float64 q and ket momenta");
}
double item(const py::array& a,py::ssize_t offset) {
    double value;std::memcpy(&value,static_cast<const char*>(a.data())+offset,sizeof(double));return value;
}
}
void bind_manta_periodic_fitting(py::module_& m) {
    auto o=py::class_<PeriodicGDFShortRangeOptions>(m,"PeriodicGDFShortRangeOptions").def(py::init<>());
#define O(name) o.def_readwrite(#name,&PeriodicGDFShortRangeOptions::name)
    O(omega);O(pair_cutoff_bohr);O(auxiliary_cutoff_bohr);O(integral_screen_error);O(coupling_screen_error);O(coupling_multipole_error);O(coupling_multipole_order);O(coupling_multipole_hierarchy);O(coupling_multipole_reuse);O(coupling_multipole_compress);O(coupling_multipole_global_actions);O(coupling_multipole_pair_tree);O(coupling_multipole_partition_bytes);O(coupling_multipole_extents);O(image_candidate_cap);O(compute_metric);O(metric_assembly_rank);
#undef O
    auto mp=py::class_<PeriodicGDFMultipolePlan>(m,"PeriodicGDFMultipolePlan");
#define P(name) mp.def_readonly(#name,&PeriodicGDFMultipolePlan::name)
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto p=py::class_<PeriodicGDFShortRangePlan>(m,"PeriodicGDFShortRangePlan");
#define P(name) p.def_readonly(#name,&PeriodicGDFShortRangePlan::name)
    P(n_aux);P(n_orb);P(momenta);P(compute_metric);P(metric_image_candidates);P(pair_image_candidates);P(auxiliary_image_candidates);
    P(metric_shell_calls);P(three_center_shell_calls);P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);
    P(output_bytes);P(driver_live_bytes);P(modeled_peak_bytes);P(metric);P(three_center);P(multipole);
#undef P
    auto r=py::class_<PeriodicGDFShortRangeReceipt>(m,"PeriodicGDFShortRangeReceipt");
#define R(name) r.def_readonly(#name,&PeriodicGDFShortRangeReceipt::name)
    R(provider);R(provider_version);R(normalization);R(algorithm);R(kernel);R(convention);
    R(orbital_basis_identity);R(auxiliary_basis_identity);R(metric_source_identity);R(three_center_source_identity);R(source_identity);
    R(omega);R(pair_cutoff_bohr);R(auxiliary_cutoff_bohr);R(integral_screen_error);R(coupling_screen_error);R(coupling_multipole_error);R(evaluated_multipole_error);R(coupling_multipole_hierarchy);R(coupling_multipole_pair_tree);
    R(multipole_partition_builds);R(multipole_partition_reuses);R(multipole_partition_bypasses);R(multipole_trials);R(multipole_images);R(multipole_moment_panels);R(multipole_tree_nodes);R(multipole_far_blocks);R(multipole_translations);R(multipole_reused_extents);R(multipole_reused_moment_panels);R(multipole_primitive_products);R(multipole_envelope_evaluations);R(multipole_candidate_images);R(metric_shell_calls);R(three_center_shell_calls);
    R(pair_shell_images);R(primitive_products);R(phase_products);R(phase_accumulations);R(modeled_peak_bytes);R(threads);
#undef R
    py::class_<PeriodicGDFShortRangeResult>(m,"PeriodicGDFShortRangeResult")
        .def_readonly("plan",&PeriodicGDFShortRangeResult::plan)
        .def_readonly("receipt",&PeriodicGDFShortRangeResult::receipt)
        .def_property_readonly("metric",[](const PeriodicGDFShortRangeResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.compute_metric?r.plan.n_aux:0);
            py::array_t<std::complex<double>> out({n,n});
            if(n) std::memcpy(out.mutable_data(),r.metric_values.data(),r.metric_values.size()*sizeof(std::complex<double>));return out;
        })
        .def_property_readonly("three_center",[](const PeriodicGDFShortRangeResult& r) {
            py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(r.plan.momenta),static_cast<py::ssize_t>(r.plan.n_aux),
                static_cast<py::ssize_t>(r.plan.n_orb),static_cast<py::ssize_t>(r.plan.n_orb)});
            std::memcpy(out.mutable_data(),r.three_center_values.data(),r.three_center_values.size()*sizeof(std::complex<double>));return out;
        });
    m.def("plan_periodic_gdf_short_range",&plan_periodic_gdf_short_range,
        py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("momenta"),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{});
    m.def("compute_periodic_gdf_short_range",[](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,
        const vibeqc::PeriodicSystem& system,const py::array& q,const py::array& k,
        const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget) {
        check(q,1);check(k,2);
        if(q.shape(0)!=3 || k.shape(1)!=3) throw py::value_error("MANTA periodic fitting q must have shape (3,) and ket momenta (nk,3)");
        // Complete joint admission precedes either numerical input copy. Keep
        // the GIL; the borrowed bases/system/options must remain unchanged.
        plan_periodic_gdf_short_range(orbital,auxiliary,system,k.shape(0),o,budget);
        Eigen::Vector3d transfer;md::Matrix momenta(k.shape(0),3);
        for(int j=0;j<3;++j) transfer[j]=item(q,j*q.strides(0));
        for(Eigen::Index i=0;i<momenta.rows();++i)for(int j=0;j<3;++j)momenta(i,j)=item(k,i*k.strides(0)+j*k.strides(1));
        return compute_periodic_gdf_short_range(orbital,auxiliary,system,transfer,momenta,o,budget);
    },py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("q").noconvert(),py::arg("ket_kpoints").noconvert(),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{});

    py::class_<PeriodicGDFKernelOptions>(m,"PeriodicGDFKernelOptions").def(py::init<>())
        .def_readwrite("kernel",&PeriodicGDFKernelOptions::kernel)
        .def_readwrite("omega",&PeriodicGDFKernelOptions::omega);
    auto gp=py::class_<PeriodicGDFPlan>(m,"PeriodicGDFPlan");
#define P(name) gp.def_readonly(#name,&PeriodicGDFPlan::name)
    P(n_aux);P(n_orb);P(momenta);P(vectors);P(panel_vectors);P(panel_count);P(compute_metric);
    P(shell_calls);P(primitive_products);P(vector_primitive_products);P(scalar_terms);P(kernel_terms);
    P(output_bytes);P(modeled_peak_bytes);P(short_range);P(auxiliary_fourier);P(ao_pair_fourier);
#undef P
    py::class_<PeriodicGDFWeightPlan>(m,"PeriodicGDFWeightPlan")
        .def_readonly("modeled_peak_bytes",&PeriodicGDFWeightPlan::modeled_peak_bytes)
        .def_readonly("scalar_terms",&PeriodicGDFWeightPlan::scalar_terms)
        .def_readonly("kernel_terms",&PeriodicGDFWeightPlan::kernel_terms);
    py::class_<PeriodicGDFWeightCertificate>(m,"PeriodicGDFWeightCertificate")
        .def_readonly("lower",&PeriodicGDFWeightCertificate::lower)
        .def_readonly("upper",&PeriodicGDFWeightCertificate::upper)
        .def_readonly("error_bound",&PeriodicGDFWeightCertificate::error_bound);
    m.def("plan_periodic_gdf_weight",&plan_periodic_gdf_weight,py::arg("budget")=ResourceBudget{});
    m.def("certify_periodic_gdf_weight",&certify_periodic_gdf_weight,py::arg("momentum"),py::arg("omega"),
        py::arg("reciprocal_prefactor"),py::arg("physical_kernel"),py::arg("zero"),py::arg("evaluated_weight"),py::arg("budget")=ResourceBudget{});
    auto gr=py::class_<PeriodicGDFReceipt>(m,"PeriodicGDFReceipt");
#define R(name) gr.def_readonly(#name,&PeriodicGDFReceipt::name)
    R(provider);R(provider_version);R(normalization);R(algorithm);R(kernel);R(convention);
    R(metric_source_identity);R(three_center_source_identity);R(source_identity);R(short_range);
    R(metric_weight_included);R(maximum_metric_weight_error);R(metric_phase_included);R(maximum_metric_phase_error);R(metric_assembly_evaluated);R(maximum_metric_assembly_error);R(metric_assembly_certificate_identity);
    R(reference_three_center_source_identity);R(coupling_error_bound);R(evaluated_coupling_error_bound);R(coupling_error_evaluated);
    R(zero_transfer);R(physical_zero_included);R(real_short_range);R(physical_omega);R(finite_zero_removed);R(reciprocal_vectors);R(auxiliary_fourier_shell_calls);R(pair_fourier_shell_calls);
    R(shell_calls);R(primitive_products);R(vector_primitive_products);R(phase_products);R(phase_accumulations);
    R(contraction_terms);R(modeled_peak_bytes);R(threads);
#undef R
    py::class_<PeriodicGDFResult>(m,"PeriodicGDFResult")
        .def_readonly("plan",&PeriodicGDFResult::plan)
        .def_readonly("receipt",&PeriodicGDFResult::receipt)
        .def_property_readonly("metric",[](const PeriodicGDFResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.compute_metric?r.plan.n_aux:0);
            py::array_t<std::complex<double>> out({n,n});
            if(n) std::memcpy(out.mutable_data(),r.metric_values.data(),r.metric_values.size()*sizeof(std::complex<double>));return out;
        })
        .def_property_readonly("three_center",[](const PeriodicGDFResult& r) {
            py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(r.plan.momenta),static_cast<py::ssize_t>(r.plan.n_aux),
                static_cast<py::ssize_t>(r.plan.n_orb),static_cast<py::ssize_t>(r.plan.n_orb)});
            std::memcpy(out.mutable_data(),r.three_center_values.data(),r.three_center_values.size()*sizeof(std::complex<double>));return out;
        });
    m.def("plan_periodic_gdf",&plan_periodic_gdf,py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),
        py::arg("vectors"),py::arg("momenta"),py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});
    m.def("compute_periodic_gdf",[](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,
        const vibeqc::PeriodicSystem& system,const py::array& q,const py::array& k,const py::array& v,
        const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,const PeriodicGDFKernelOptions& physical) {
        check(q,1);check(k,2);check(v,2);
        if(q.shape(0)!=3 || k.shape(1)!=3 || v.shape(1)!=3)
            throw py::value_error("MANTA GDF q must have shape (3,), momenta and vectors (n,3)");
        plan_periodic_gdf(orbital,auxiliary,system,v.shape(0),k.shape(0),o,budget,physical);
        Eigen::Vector3d transfer;md::Matrix momenta(k.shape(0),3),vectors(v.shape(0),3);
        for(int j=0;j<3;++j) transfer[j]=item(q,j*q.strides(0));
        for(Eigen::Index i=0;i<momenta.rows();++i)for(int j=0;j<3;++j)momenta(i,j)=item(k,i*k.strides(0)+j*k.strides(1));
        for(Eigen::Index i=0;i<vectors.rows();++i)for(int j=0;j<3;++j)vectors(i,j)=item(v,i*v.strides(0)+j*v.strides(1));
        return compute_periodic_gdf(orbital,auxiliary,system,transfer,momenta,vectors,o,budget,physical);
    },py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("q").noconvert(),
        py::arg("ket_kpoints").noconvert(),py::arg("vectors").noconvert(),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});

    auto tp=py::class_<PeriodicGDFTilesPlan>(m,"PeriodicGDFTilesPlan");
#define P(name) tp.def_readonly(#name,&PeriodicGDFTilesPlan::name)
    P(n_aux);P(n_orb);P(momenta);P(vectors);P(panel_vectors);P(compute_metric);
    P(metric_shell_calls);P(coupling_shell_calls);P(metric_primitive_products);P(coupling_primitive_products);
    P(metric_scalar_terms);P(coupling_scalar_terms);P(metric_kernel_terms);P(coupling_kernel_terms);
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(maximum_tile_elements);
    P(pair_tree_bytes);P(pair_tree_action_bytes);P(pair_tree_scalar_terms);P(partition_cache_bytes);P(partition_workspace_bytes);P(partition_scalar_terms);P(global_action_bytes);P(global_action_scalar_terms);P(global_pair_slots);P(global_auxiliary_slots);P(compressed_action_bytes);P(compressed_action_scalar_terms);P(preparation_cache_bytes);P(preparation_pair_slots);P(preparation_auxiliary_slots);P(preparation_support_slots);P(retained_bytes);P(metric_peak_bytes);P(coupling_peak_bytes);P(modeled_peak_bytes);P(external_live_bytes);P(coupling_error_bound);
#undef P
    auto preparation=py::class_<PeriodicGDFPreparationInfo>(m,"PeriodicGDFPreparationInfo");
#define P(name) preparation.def_readonly(#name,&PeriodicGDFPreparationInfo::name)
    P(enabled);P(source_identity);P(pair_extents);P(auxiliary_extents);P(moment_panels);P(moment_elements);P(support_tiles);P(modeled_live_bytes);P(capacity_bytes);
#undef P
    auto partitions=py::class_<PeriodicGDFPartitionInfo>(m,"PeriodicGDFPartitionInfo");
#define P(name) partitions.def_readonly(#name,&PeriodicGDFPartitionInfo::name)
    P(enabled);P(source_identity);P(entries);P(evictions);P(modeled_live_bytes);P(capacity_bytes);
#undef P
    py::class_<PeriodicGDFTile>(m,"PeriodicGDFTile")
        .def_readonly("metric_assembly_error_bounds",&PeriodicGDFTile::metric_assembly_error_bounds)
        .def_readonly("shape",&PeriodicGDFTile::shape).def_readonly("offsets",&PeriodicGDFTile::offsets)
        .def_readonly("receipt",&PeriodicGDFTile::receipt)
        .def_property_readonly("values",[](const PeriodicGDFTile& tile) {
            std::vector<py::ssize_t> shape(tile.shape.begin(),tile.shape.end());py::array_t<std::complex<double>> out(shape);
            if(!tile.values.empty()) std::memcpy(out.mutable_data(),tile.values.data(),tile.values.size()*sizeof(std::complex<double>));return out;
        });
    m.def("plan_periodic_gdf_tiles",&plan_periodic_gdf_tiles,py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),
        py::arg("vectors"),py::arg("momenta"),py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});
    auto cp=py::class_<PeriodicGDFCouplingPlan>(m,"PeriodicGDFCouplingPlan");
#define P(name) cp.def_readonly(#name,&PeriodicGDFCouplingPlan::name)
    P(roundoff_workspace_bytes);P(roundoff_scalar_terms);P(roundoff_kernel_terms);P(n_aux);P(n_orb);P(momenta);P(maximum_tile_elements);P(shell_calls);P(primitive_products);
    P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto cr=py::class_<PeriodicGDFCouplingReceipt>(m,"PeriodicGDFCouplingReceipt");
#define R(name) cr.def_readonly(#name,&PeriodicGDFCouplingReceipt::name)
    R(provider);R(algorithm);R(convention);R(direction);R(coupling_identity);R(input_identity);R(snapshot_identity);
    R(coupling_passes);R(shell_tiles);R(shell_calls);R(primitive_products);
    R(multipole_partition_builds);R(multipole_partition_reuses);R(multipole_partition_bypasses);R(multipole_trials);R(multipole_images);R(multipole_moment_panels);R(multipole_tree_nodes);R(multipole_far_blocks);R(multipole_translations);R(multipole_reused_extents);R(multipole_reused_moment_panels);R(multipole_envelope_evaluations);R(multipole_global_pairs);R(multipole_global_pair_reuses);R(multipole_global_auxiliary);R(multipole_global_projections);R(multipole_global_auxiliary_projections);R(multipole_pair_groups);R(multipole_compressed_pairs);R(multipole_compressed_nodes);R(multipole_action_products);R(multipole_action_translations);R(integral_elements_consumed);
    R(roundoff_evaluated);R(input_error_included);R(roundoff_bound);R(contraction_products);R(modeled_peak_bytes);
    R(approximation_evaluated);R(reference_coupling_identity);R(coupling_error_bound);R(input_l1_bound);R(approximation_bound);R(approximation_l2_bound);R(maximum_coupling);
#undef R
    py::class_<PeriodicGDFCouplingResult>(m,"PeriodicGDFCouplingResult")
        .def_readonly("plan",&PeriodicGDFCouplingResult::plan).def_readonly("receipt",&PeriodicGDFCouplingResult::receipt)
        .def_readonly("roundoff_bounds",&PeriodicGDFCouplingResult::roundoff_bounds)
        .def_readonly("metric_norm_qualified",&PeriodicGDFCouplingResult::metric_norm_qualified)
        .def_readonly("metric_column_norm_bounds",&PeriodicGDFCouplingResult::metric_column_norm_bounds)
        .def_property_readonly("values",[](const PeriodicGDFCouplingResult& r) {
            std::vector<py::ssize_t> shape(r.shape.begin(),r.shape.end());py::array_t<std::complex<double>> out(shape);
            std::memcpy(out.mutable_data(),r.values.data(),r.values.size()*sizeof(std::complex<double>));return out;
        })
        .def_property_readonly("approximation_bounds",[](const PeriodicGDFCouplingResult& r) {
            std::vector<py::ssize_t> shape(r.shape.begin(),r.shape.end());py::array_t<double> out(shape);
            std::memcpy(out.mutable_data(),r.approximation_bounds.data(),r.approximation_bounds.size()*sizeof(double));
            out.attr("setflags")(py::arg("write")=false);return out;
        })
        .def_property_readonly("reference_action_bounds",[](const PeriodicGDFCouplingResult& r) {
            py::array_t<double> out(r.reference_action_bounds.size());
            if(!r.reference_action_bounds.empty())
                std::memcpy(out.mutable_data(),r.reference_action_bounds.data(),r.reference_action_bounds.size()*sizeof(double));
            out.attr("flags").attr("writeable")=false;return out;
        })
        .def_property_readonly("reference_magnitude_bounds",[](const PeriodicGDFCouplingResult& r) {
            py::array_t<double> out(r.reference_magnitude_bounds.size());
            if(!r.reference_magnitude_bounds.empty())
                std::memcpy(out.mutable_data(),r.reference_magnitude_bounds.data(),r.reference_magnitude_bounds.size()*sizeof(double));
            out.attr("setflags")(py::arg("write")=false);return out;
        });
    auto action=[](const PeriodicGDFTiles& s,const py::array& input,const ResourceBudget& budget,bool forward,bool audit_roundoff,const py::object& input_errors) {
        using Complex=std::complex<double>;
        const auto& p=s.plan();
        if(!input.dtype().is(py::dtype::of<Complex>()) ||
            (forward?(input.ndim()!=3 || input.shape(0)!=py::ssize_t(p.momenta) ||
                input.shape(1)!=py::ssize_t(p.n_orb) || input.shape(2)!=py::ssize_t(p.n_orb)):
                (input.ndim()!=1 || input.shape(0)!=py::ssize_t(p.n_aux))))
            throw py::value_error("MANTA GDF coupling requires complex128 (ket,AO,AO) forward or (auxiliary,) adjoint input");
        if(!(input.flags() & (py::array::c_style|py::array::f_style)))
            throw py::value_error("MANTA GDF coupling input must be C- or F-contiguous");
        if(audit_roundoff)s.plan_coupling_roundoff(budget);else s.plan_coupling(budget); // Before input copies.
        std::vector<double> errors;
        if(!input_errors.is_none()) {
            if(!audit_roundoff || !py::isinstance<py::array>(input_errors))
                throw py::value_error("MANTA coupling input errors require an audited float64 array");
            const auto e=py::cast<py::array>(input_errors);
            if(!e.dtype().is(py::dtype::of<double>()) || e.ndim()!=1 || e.size()!=input.size() || !(e.flags()&py::array::c_style))
                throw py::value_error("MANTA coupling input errors require a matching contiguous float64 vector");
            errors.resize(e.size());std::memcpy(errors.data(),e.data(),errors.size()*sizeof(double));
        }
        std::vector<Complex> values(input.size());const auto* raw=static_cast<const char*>(input.data());Count out=0;
        if(forward) for(py::ssize_t k=0;k<input.shape(0);++k)for(py::ssize_t i=0;i<input.shape(1);++i)
            for(py::ssize_t j=0;j<input.shape(2);++j)
                std::memcpy(&values[out++],raw+k*input.strides(0)+i*input.strides(1)+j*input.strides(2),sizeof(Complex));
        else std::memcpy(values.data(),raw,values.size()*sizeof(Complex));
        const ResourceBudget owned=budget;
        py::gil_scoped_release release;
        return forward?s.forward(values,owned,audit_roundoff,errors):s.adjoint(values,owned,audit_roundoff,errors);
    };
    auto ppt=py::class_<PeriodicGDFPairTreePlan>(m,"PeriodicGDFPairTreePlan");
#define P(name) ppt.def_readonly(#name,&PeriodicGDFPairTreePlan::name)
    P(pair_capacity);P(node_capacity);P(auxiliary_image_capacity);P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto entry=py::class_<PeriodicGDFPairTreeEntry>(m,"PeriodicGDFPairTreeEntry");
#define E(name) entry.def_readonly(#name,&PeriodicGDFPairTreeEntry::name)
    E(bra_shell);E(ket_shell);E(pair_ordinal);E(ket_image);E(bra_frame);E(bra_center);E(ket_center);E(translation);E(frame_translation);
#undef E
    py::class_<PeriodicGDFCapsuleMask>(m,"PeriodicGDFCapsuleMask")
        .def_readonly("membership",&PeriodicGDFCapsuleMask::membership).def_readonly("pairs",&PeriodicGDFCapsuleMask::pairs);
    py::class_<PeriodicGDFCapsulePartition>(m,"PeriodicGDFCapsulePartition")
        .def_readonly("included_nodes",&PeriodicGDFCapsulePartition::included_nodes)
        .def_readonly("included_pairs",&PeriodicGDFCapsulePartition::included_pairs)
        .def_readonly("excluded_pairs",&PeriodicGDFCapsulePartition::excluded_pairs)
        .def_readonly("visited_nodes",&PeriodicGDFCapsulePartition::visited_nodes)
        .def_readonly("refined_nodes",&PeriodicGDFCapsulePartition::refined_nodes)
        .def_readonly("source_identity",&PeriodicGDFCapsulePartition::source_identity);
    py::class_<PeriodicGDFPairTree,std::shared_ptr<PeriodicGDFPairTree>>(m,"PeriodicGDFPairTree")
        .def_property_readonly("plan",&PeriodicGDFPairTree::plan)
        .def_property_readonly("source_identity",&PeriodicGDFPairTree::source_identity)
        .def_property_readonly("entries",&PeriodicGDFPairTree::entries)
        .def_property_readonly("tree",[](const PeriodicGDFPairTree& t){return std::const_pointer_cast<BubbleTree>(t.tree());})
        .def("classify",&PeriodicGDFPairTree::classify,py::arg("node"),py::arg("auxiliary_shell"),py::arg("image"))
        .def("auxiliary_images",&PeriodicGDFPairTree::auxiliary_images,py::arg("auxiliary_shell"),py::call_guard<py::gil_scoped_release>())
        .def("partition",&PeriodicGDFPairTree::partition,py::arg("auxiliary_shell"),py::arg("image"),py::call_guard<py::gil_scoped_release>())
        .def("pair_phase",&PeriodicGDFPairTree::pair_phase,py::arg("entry"))
        .def("auxiliary_phase",&PeriodicGDFPairTree::auxiliary_phase,py::arg("auxiliary_shell"),py::arg("image"))
        .def("source_auxiliary_image",&PeriodicGDFPairTree::source_auxiliary_image,py::arg("entry"),py::arg("auxiliary_shell"),py::arg("image"));
    m.def("plan_periodic_gdf_pair_tree",&plan_periodic_gdf_pair_tree,py::arg("source"),py::arg("budget")=ResourceBudget{});
    m.def("build_periodic_gdf_pair_tree",&build_periodic_gdf_pair_tree,py::arg("source"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    auto domains=py::class_<PeriodicGDFDomainPlan>(m,"PeriodicGDFDomainPlan");
#define D(n) domains.def_readonly(#n,&PeriodicGDFDomainPlan::n)
    D(source);D(maximum_vectors);D(generation_bytes);D(generation_scalar_terms);D(generation_kernel_terms);
    D(modeled_peak_bytes);D(scalar_terms);D(kernel_terms);
#undef D
    py::class_<PeriodicGDFTiles,std::shared_ptr<PeriodicGDFTiles>>(m,"PeriodicGDFTiles")
        .def(py::init([](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,
            const vibeqc::PeriodicSystem& system,const py::array& q,const py::array& k,const py::array& v,
            const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,const PeriodicGDFKernelOptions& physical) {
            check(q,1);check(k,2);check(v,2);
            if(q.shape(0)!=3 || k.shape(1)!=3 || v.shape(1)!=3) throw py::value_error("MANTA GDF tile vectors require three components");
            plan_periodic_gdf_tiles(orbital,auxiliary,system,v.shape(0),k.shape(0),o,budget,physical);
            Eigen::Vector3d transfer;md::Matrix momenta(k.shape(0),3),vectors(v.shape(0),3);
            for(int j=0;j<3;++j) transfer[j]=item(q,j*q.strides(0));
            for(Eigen::Index i=0;i<momenta.rows();++i)for(int j=0;j<3;++j)momenta(i,j)=item(k,i*k.strides(0)+j*k.strides(1));
            for(Eigen::Index i=0;i<vectors.rows();++i)for(int j=0;j<3;++j)vectors(i,j)=item(v,i*v.strides(0)+j*v.strides(1));
            return std::make_shared<PeriodicGDFTiles>(orbital,auxiliary,system,transfer,momenta,vectors,o,budget,physical);
        }),py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("q").noconvert(),
            py::arg("ket_kpoints").noconvert(),py::arg("vectors").noconvert(),
            py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{})
        .def_property_readonly("plan",&PeriodicGDFTiles::plan)
        .def("plan_auxiliary_domains",&PeriodicGDFTiles::plan_auxiliary_domains,py::arg("auxiliary"),py::arg("pair_cutoff"),py::arg("auxiliary_cutoff"),py::arg("cube"),py::arg("budget")=ResourceBudget{})
        .def("with_auxiliary_domains",&PeriodicGDFTiles::with_auxiliary_domains,py::arg("auxiliary"),py::arg("pair_cutoff"),py::arg("auxiliary_cutoff"),py::arg("cube"),py::arg("budget")=ResourceBudget{})
        .def("plan_domains",&PeriodicGDFTiles::plan_domains,py::arg("pair_cutoff"),py::arg("auxiliary_cutoff"),py::arg("cube"),py::arg("budget")=ResourceBudget{})
        .def("with_domains",&PeriodicGDFTiles::with_domains,py::arg("pair_cutoff"),py::arg("auxiliary_cutoff"),py::arg("cube"),py::arg("budget")=ResourceBudget{})
        .def_property_readonly("vectors",[](const PeriodicGDFTiles& s) {
            const auto& v=s.vectors();py::array_t<double> out({static_cast<py::ssize_t>(v.rows()),py::ssize_t{3}});
            auto a=out.mutable_unchecked<2>();for(Eigen::Index i=0;i<v.rows();++i)for(int j=0;j<3;++j)a(i,j)=v(i,j);
            return out;
        })
        .def_property_readonly("preparation",&PeriodicGDFTiles::preparation_info)
        .def_property_readonly("partitions",&PeriodicGDFTiles::partition_info)
        .def_property_readonly("receipt",[](const PeriodicGDFTiles& s){return s.source();})
        .def_property_readonly("source_identity",[](const PeriodicGDFTiles& s){return s.source().source_identity;})
        .def_property_readonly("metric_source_identity",[](const PeriodicGDFTiles& s){return s.source().metric_source_identity;})
        .def_property_readonly("three_center_source_identity",[](const PeriodicGDFTiles& s){return s.source().three_center_source_identity;})
        .def_property_readonly("orbital_basis_identity",[](const PeriodicGDFTiles& s){return s.source().short_range.orbital_basis_identity;})
        .def_property_readonly("auxiliary_basis_identity",[](const PeriodicGDFTiles& s){return s.source().short_range.auxiliary_basis_identity;})
        .def_property_readonly("shell_extents",[](const PeriodicGDFTiles& s){return std::vector<Count>{s.auxiliary().nshells(),s.orbital().nshells(),s.orbital().nshells()};})
        .def("plan_coupling",&PeriodicGDFTiles::plan_coupling,py::arg("budget")=ResourceBudget{})
        .def("forward",[action](const PeriodicGDFTiles& s,const py::array& input,const ResourceBudget& budget,bool audit_roundoff,const py::object& input_errors) {
            return action(s,input,budget,true,audit_roundoff,input_errors);
        },py::arg("input").noconvert(),py::arg("budget")=ResourceBudget{},py::arg("audit_roundoff")=false,py::arg("input_errors")=py::none())
        .def("adjoint",[action](const PeriodicGDFTiles& s,const py::array& input,const ResourceBudget& budget,bool audit_roundoff,const py::object& input_errors) {
            return action(s,input,budget,false,audit_roundoff,input_errors);
        },py::arg("input").noconvert(),py::arg("budget")=ResourceBudget{},py::arg("audit_roundoff")=false,py::arg("input_errors")=py::none())
        .def("metric",&PeriodicGDFTiles::metric)
        .def("plan_coupling_roundoff",&PeriodicGDFTiles::plan_coupling_roundoff,py::arg("budget")=ResourceBudget{})
        .def("plan_coupling_norm_bounds",&PeriodicGDFTiles::plan_coupling_norm_bounds,py::arg("budget")=ResourceBudget{})
        .def("coupling_norm_bounds",&PeriodicGDFTiles::coupling_norm_bounds,py::arg("budget")=ResourceBudget{})
        .def("evaluate",&PeriodicGDFTiles::evaluate,py::arg("auxiliary_shell"),py::arg("bra_shell"),py::arg("ket_shell"));
}
