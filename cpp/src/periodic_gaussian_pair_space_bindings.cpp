// Included by bindings.cpp. Tiny native-owner diagnostics, no caller G/C/O.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_pair_space.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

void bind_periodic_gaussian_pair_space(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianPairSpaceOptions;
    using Live=PeriodicGaussianPairSpaceLiveInventory;
    using Caps=PeriodicGaussianPairSpaceCaps;
    using Storage=PeriodicGaussianPairSpaceStorage;
    using Plan=PeriodicGaussianPairSpacePlan;
    using Diagnostics=PeriodicGaussianPairSpaceDiagnostics;
    using Space=PeriodicGaussianPairSpace;
    using OverlapCaps=PeriodicGaussianPairOverlapCaps;
    using OverlapPlan=PeriodicGaussianPairOverlapPlan;
    using Overlap=PeriodicGaussianPairOverlap;
    py::class_<Options>(m,"_PeriodicGaussianPairSpaceOptions").def(py::init<>())
        .def_readwrite("maximum_diagonal_symmetry_projection_error",&Options::maximum_diagonal_symmetry_projection_error);
    py::class_<Live>(m,"_PeriodicGaussianPairSpaceLiveInventory").def(py::init<>())
        .def_readwrite("other_live_bytes_per_worker",&Live::other_live_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
    py::class_<Caps>(m,"_PeriodicGaussianPairSpaceCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes",&Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_per_worker_inventoried_bytes",&Caps::maximum_per_worker_inventoried_bytes)
        .def_readwrite("maximum_node_inventoried_bytes",&Caps::maximum_node_inventoried_bytes)
        .def_readwrite("maximum_integral_calls",&Caps::maximum_integral_calls)
        .def_readwrite("maximum_work_units",&Caps::maximum_work_units);
    auto storage=py::class_<Storage>(m,"_PeriodicGaussianPairSpaceStorage");
#define GPS_STORAGE(field) storage.def_readonly(#field,&Storage::field)
    GPS_STORAGE(virtual_count); GPS_STORAGE(generation_dimension); GPS_STORAGE(retained_dimension); GPS_STORAGE(integral_calls);
    GPS_STORAGE(construction_owned_bytes); GPS_STORAGE(borrowed_pno_bytes); GPS_STORAGE(retained_output_bytes);
    GPS_STORAGE(peak_owned_numerical_bytes); GPS_STORAGE(construction_live_numerical_bytes);
    GPS_STORAGE(numerical_work_units); GPS_STORAGE(fixed_control_storage_bytes);
#undef GPS_STORAGE
    m.def("_periodic_gaussian_pair_space_storage",[](std::uint64_t n,std::uint64_t r){
        return periodic_gaussian_pair_space_storage(n,r);
    },
        py::arg("virtual_count"),py::arg("retained_dimension"));
    m.def("_periodic_gaussian_pair_space_storage",[](std::uint64_t n,std::uint64_t m,std::uint64_t r){
        return periodic_gaussian_pair_space_storage(n,m,r);
    },py::arg("virtual_count"),py::arg("generation_dimension"),py::arg("retained_dimension"));
    auto plan=py::class_<Plan,Storage>(m,"_PeriodicGaussianPairSpacePlan");
#define GPS_PLAN(field) plan.def_readonly(#field,&Plan::field)
    GPS_PLAN(occupied_count); GPS_PLAN(occupied_slot_i); GPS_PLAN(occupied_slot_j); GPS_PLAN(provider_work_units);
    GPS_PLAN(work_units); GPS_PLAN(borrowed_basis_bytes); GPS_PLAN(borrowed_provider_row_bytes);
    GPS_PLAN(state_resident_bytes); GPS_PLAN(replicas_per_node); GPS_PLAN(reference_base_node_bytes);
    GPS_PLAN(per_worker_inventoried_bytes); GPS_PLAN(required_node_memory_bytes);
#undef GPS_PLAN
    plan.def_property_readonly("options",[](const Plan& p){return p.options;})
        .def_property_readonly("live",[](const Plan& p){return p.live;})
        .def_property_readonly("caps",[](const Plan& p){return p.caps;})
        .def_property_readonly("plan_identity_sha256",&Plan::plan_identity_sha256);
    auto diag=py::class_<Diagnostics>(m,"_PeriodicGaussianPairSpaceDiagnostics");
#define GPS_DIAG(field) diag.def_readonly(#field,&Diagnostics::field)
    GPS_DIAG(completed_integral_calls); GPS_DIAG(product_underflow_count); GPS_DIAG(source_integrals_replayed);
    GPS_DIAG(diagonal_symmetry_projection_applied); GPS_DIAG(maximum_scalar_integral_roundoff_error);
    GPS_DIAG(common_basis_integral_projection_error_bound); GPS_DIAG(maximum_raw_diagonal_asymmetry);
    GPS_DIAG(diagonal_symmetry_projection_frobenius_bound);
#undef GPS_DIAG
    const auto copy=[](const std::vector<double>& values,std::uint64_t rows,std::uint64_t columns) {
        if(rows>8 || columns>8 || values.size()!=rows*columns)
            throw std::length_error("Gaussian pair space copy exceeds tiny diagnostic dimensions");
        py::array_t<double> a({static_cast<py::ssize_t>(rows),static_cast<py::ssize_t>(columns)});
        if(!values.empty()) std::memcpy(a.mutable_data(),values.data(),values.size()*8);
        return a;
    };
    py::class_<Space>(m,"_PeriodicGaussianPairSpace")
        .def_property_readonly("memory",[](const Space& s){return s.memory();})
        .def_property_readonly("diagnostics",[](const Space& s){return s.diagnostics();})
        .def_property_readonly("state",&Space::state_handle).def_property_readonly("context",&Space::context_handle)
        .def_property_readonly("occupied_i",[](const Space& s){const auto i=s.occupied_i();return py::make_tuple(i.occupied_index,i.cell);})
        .def_property_readonly("occupied_j",[](const Space& s){const auto i=s.occupied_j();return py::make_tuple(i.occupied_index,i.cell);})
        .def_property_readonly("diagonal_pair",&Space::diagonal_pair)
        .def_property_readonly("coupled_mp2_solution",&Space::coupled_mp2_solution)
        .def_property_readonly("production_dlpno",&Space::production_dlpno)
        .def_property_readonly("infinite_source_accuracy_certified",&Space::infinite_source_accuracy_certified)
        .def_property_readonly("identity_sha256",&Space::identity_sha256)
        .def_property_readonly("payload_sha256",&Space::payload_sha256)
        .def_property_readonly("raw_exchange_integral_identity_sha256",&Space::raw_exchange_integral_identity_sha256)
        .def_property_readonly("exchange_integral_identity_sha256",&Space::exchange_integral_identity_sha256)
        .def_property_readonly("allocation_identity",&Space::allocation_identity)
        .def_property_readonly("local_basis_identity_sha256",&Space::local_basis_identity_sha256)
        .def_property_readonly("consumed_sources_identity_sha256",&Space::consumed_sources_identity_sha256)
        .def_property_readonly("embedded_generation",[](const Space& s){return s.frame().is_embedded();})
        .def_property_readonly("direct_gram_generation",[](const Space& s){return s.frame().is_direct_gram();})
        .def_property_readonly("complete_generation_pair_space",[](const Space& s){return s.frame().complete_generation_pair_space();})
        .def_property_readonly("complete_common_virtual_space",[](const Space& s){return s.frame().complete_common_virtual_space();})
        .def_property_readonly("pno_identity_sha256",[](const Space& s){return s.frame().identity_sha256();})
        .def_property_readonly("pno_payload_sha256",[](const Space& s){return s.frame().payload_sha256();})
        .def_property_readonly("pno_diagnostics",[](const Space& s){return s.frame().generation_diagnostics();})
        .def_property_readonly("pno_options",[](const Space& s){return s.frame().options();})
        .def_property_readonly("embedded_pno_diagnostics",[](const Space& s){return s.embedded_pnos().diagnostics();})
        .def_property_readonly("embedded_pno_options",[](const Space& s){return s.embedded_pnos().options();})
        .def_property_readonly("embedding_identity_sha256",[](const Space& s){return s.embedded_pnos().embedding_identity_sha256();})
        .def_property_readonly("common_source_exchange_integral_identity_sha256",[](const Space& s){return s.frame().common_exchange_integral_identity_sha256();})
        .def_property_readonly("basis_identity_sha256",[](const Space& s){return s.frame().basis_identity_sha256();})
        .def_property_readonly("provider_identity_sha256",[](const Space& s){return s.frame().provider_identity_sha256();})
        .def_property_readonly("hf_reference_source_identity_sha256",[](const Space& s){return s.frame().hf_reference_source_identity_sha256();})
        .def("exchange_integrals_copy",[copy](const Space& s){return copy(s.exchange_integrals(),s.memory().retained_dimension,s.memory().retained_dimension);})
        .def("coefficients_copy",[copy](const Space& s){return copy(s.coefficients(),s.memory().virtual_count,s.memory().retained_dimension);})
        .def("energies_copy",[](const Space& s){
            const auto& v=s.energies();
            if(v.size()>8) throw std::length_error("Gaussian pair space energy copy exceeds tiny dimension");
            py::array_t<double> a(static_cast<py::ssize_t>(v.size()));
            if(!v.empty()) std::memcpy(a.mutable_data(),v.data(),v.size()*8); return a;
        })
        .def("original_pno_occupations_copy",[](const Space& s){
            const auto& v=s.frame().original_pno_occupations();
            if(v.size()>8) throw std::length_error("Gaussian pair space occupation copy exceeds tiny dimension");
            py::array_t<double> a(static_cast<py::ssize_t>(v.size()));
            if(!v.empty()) std::memcpy(a.mutable_data(),v.data(),v.size()*8); return a;
        });
    const auto tiny=[](const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis) {
        if(ref.state().n_kpoints()>8 || basis.memory().occupied_count>8 || basis.memory().virtual_count>8)
            throw std::length_error("Gaussian pair space diagnostic exceeds tiny dimensions");
    };
    const auto bounded=[](const auto& p) {
        if(p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units>100000000000000ULL)
            throw std::length_error("Gaussian pair space diagnostic exceeds tiny memory or work");
    };
    m.def("_plan_periodic_gaussian_pair_space",[tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairPNOResult& pno,Options options,Live live,Caps caps){
        tiny(ref,basis); auto p=plan_periodic_gaussian_pair_space(ref,basis,provider,pno,options,live,caps);
        bounded(p); return p;
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("pnos"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_plan_periodic_gaussian_pair_space",[tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianEmbeddedPairPNOResult& pno,Options options,Live live,Caps caps){
        tiny(ref,basis); auto p=plan_periodic_gaussian_pair_space(ref,basis,provider,pno,options,live,caps);
        bounded(p); return p;
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("pnos"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_space",[tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        PeriodicGaussianEmbeddedPairPNOResult& pno,Options options,Live live,Caps caps){
        tiny(ref,basis); bounded(plan_periodic_gaussian_pair_space(ref,basis,provider,pno,options,live,caps));
        return make_periodic_gaussian_pair_space(ref,basis,provider,std::move(pno),options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("pnos"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_space",[tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        PeriodicGaussianPairPNOResult& pno,Options options,Live live,Caps caps){
        tiny(ref,basis); bounded(plan_periodic_gaussian_pair_space(ref,basis,provider,pno,options,live,caps));
        // GIL held; native immutable inputs, copied controls, no hidden casts.
        // Successful return consumes the Python PNO owner's native payload.
        return make_periodic_gaussian_pair_space(ref,basis,provider,std::move(pno),options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("pnos"),
      py::arg("options"),py::arg("live"),py::arg("caps"));

    py::class_<OverlapCaps>(m,"_PeriodicGaussianPairOverlapCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes",&OverlapCaps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_per_worker_inventoried_bytes",&OverlapCaps::maximum_per_worker_inventoried_bytes)
        .def_readwrite("maximum_node_inventoried_bytes",&OverlapCaps::maximum_node_inventoried_bytes)
        .def_readwrite("maximum_work_units",&OverlapCaps::maximum_work_units);
    auto op=py::class_<OverlapPlan>(m,"_PeriodicGaussianPairOverlapPlan");
#define GPS_OVERLAP(field) op.def_readonly(#field,&OverlapPlan::field)
    GPS_OVERLAP(virtual_count); GPS_OVERLAP(target_dimension); GPS_OVERLAP(source_dimension);
    GPS_OVERLAP(peak_owned_numerical_bytes); GPS_OVERLAP(borrowed_pair_space_bytes); GPS_OVERLAP(work_units);
    GPS_OVERLAP(fixed_control_storage_bytes); GPS_OVERLAP(state_resident_bytes); GPS_OVERLAP(replicas_per_node);
    GPS_OVERLAP(reference_base_node_bytes); GPS_OVERLAP(per_worker_inventoried_bytes);
    GPS_OVERLAP(required_node_memory_bytes); GPS_OVERLAP(same_object_borrower);
#undef GPS_OVERLAP
    op.def_property_readonly("live",[](const OverlapPlan& p){return p.live;})
        .def_property_readonly("caps",[](const OverlapPlan& p){return p.caps;})
        .def_property_readonly("plan_identity_sha256",&OverlapPlan::plan_identity_sha256);
    py::class_<Overlap>(m,"_PeriodicGaussianPairOverlap")
        .def_property_readonly("memory",[](const Overlap& s){return s.memory();})
        .def_property_readonly("target_identity_sha256",&Overlap::target_identity_sha256)
        .def_property_readonly("source_identity_sha256",&Overlap::source_identity_sha256)
        .def_property_readonly("identity_sha256",&Overlap::identity_sha256)
        .def_property_readonly("payload_sha256",&Overlap::payload_sha256)
        .def_property_readonly("product_underflow_count",&Overlap::product_underflow_count)
        .def("overlaps_copy",[copy](const Overlap& s){return copy(s.overlaps(),s.memory().target_dimension,s.memory().source_dimension);});
    const auto overlap_tiny=[](const Space& a,const Space& b) {
        if(a.memory().virtual_count>8 || b.memory().virtual_count>8)
            throw std::length_error("Gaussian pair overlap diagnostic exceeds tiny dimensions");
    };
    m.def("_plan_periodic_gaussian_pair_overlap",[overlap_tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const Space& target,const Space& source,Live live,OverlapCaps caps){
        overlap_tiny(target,source); auto p=plan_periodic_gaussian_pair_overlap(ref,target,source,live,caps);
        bounded(p); return p;
    },py::arg("reference"),py::arg("target"),py::arg("source"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_overlap",[overlap_tiny,bounded](const PeriodicCorrelationAdmittedReference& ref,
        const Space& target,const Space& source,Live live,OverlapCaps caps){
        overlap_tiny(target,source); bounded(plan_periodic_gaussian_pair_overlap(ref,target,source,live,caps));
        return make_periodic_gaussian_pair_overlap(ref,target,source,live,caps);
    },py::arg("reference"),py::arg("target"),py::arg("source"),py::arg("live"),py::arg("caps"));
}
