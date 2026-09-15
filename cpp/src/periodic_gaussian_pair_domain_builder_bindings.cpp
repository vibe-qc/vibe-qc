// Included by bindings.cpp. Tiny actual-source diagnostics; no child-owner escape.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "vibeqc/periodic_gaussian_pair_domain_builder.hpp"
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

void bind_periodic_gaussian_pair_domain_builder(py::module_& m) {
    using namespace vibeqc;
    using Live=PeriodicGaussianPairDomainBuilderLive;
    using Caps=PeriodicGaussianPairDomainBuilderCaps;
    using Plan=PeriodicGaussianPairDomainBuilderPlan;
    using Diagnostics=PeriodicGaussianPairDomainBuilderDiagnostics;
    using Builder=PeriodicGaussianPairDomainBuilder;
    using Geometry=PeriodicGaussianPairDomainGeometry;
    using PairOptions=PeriodicGaussianPairDomainOptions;
    using PairCaps=PeriodicGaussianPairDomainCaps;
    using PairPlan=PeriodicGaussianPairDomainEmbeddingPlan;
    using OccupiedOptions=PeriodicCorrelationOccupiedPAODomainOptions;
    auto live=py::class_<Live>(m,"_PeriodicGaussianPairDomainBuilderLive").def(py::init<>());
#define GPDB_LIVE(f) live.def_readwrite(#f,&Live::f)
    GPDB_LIVE(other_live_numerical_bytes_per_worker); GPDB_LIVE(other_live_control_bytes_per_worker);
    GPDB_LIVE(backend_margin_bytes_per_worker);
#undef GPDB_LIVE
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianPairDomainBuilderCaps").def(py::init<>());
#define GPDB_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GPDB_CAP(occupied); GPDB_CAP(maximum_home_domains); GPDB_CAP(maximum_topology_rows);
    GPDB_CAP(maximum_owned_numerical_bytes); GPDB_CAP(maximum_control_storage_bytes);
    GPDB_CAP(maximum_worker_bytes); GPDB_CAP(maximum_node_bytes); GPDB_CAP(maximum_work_units);
#undef GPDB_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianPairDomainBuilderPlan");
#define GPDB_PLAN(f) plan.def_readonly(#f,&Plan::f)
    GPDB_PLAN(n_cells); GPDB_PLAN(n_basis); GPDB_PLAN(n_atoms); GPDB_PLAN(n_home_occupied);
    GPDB_PLAN(common_virtual_dimension); GPDB_PLAN(topology_rows); GPDB_PLAN(atom_mapping_bytes);
    GPDB_PLAN(topology_row_bytes); GPDB_PLAN(home_domain_output_upper_bytes); GPDB_PLAN(common_geometry_numerical_bytes);
    GPDB_PLAN(borrowed_basis_bytes); GPDB_PLAN(borrowed_localization_bytes); GPDB_PLAN(borrowed_gaussian_bytes);
    GPDB_PLAN(retained_numerical_upper_bytes); GPDB_PLAN(peak_owned_numerical_bytes);
    GPDB_PLAN(control_storage_reservation_bytes); GPDB_PLAN(retained_control_upper_bytes);
    GPDB_PLAN(input_validation_work_units); GPDB_PLAN(domain_work_units); GPDB_PLAN(work_units);
    GPDB_PLAN(replicas_per_node); GPDB_PLAN(reference_base_node_bytes); GPDB_PLAN(worker_bytes);
    GPDB_PLAN(required_node_memory_bytes);
#undef GPDB_PLAN
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianPairDomainBuilderDiagnostics");
#define GPDB_DIAG(f) diagnostics.def_readonly(#f,&Diagnostics::f)
    GPDB_DIAG(retained_home_domain_bytes); GPDB_DIAG(retained_numerical_bytes);
    GPDB_DIAG(retained_control_storage_bytes); GPDB_DIAG(constructed_home_domains);
#undef GPDB_DIAG
    py::class_<Builder>(m,"_PeriodicGaussianPairDomainBuilder")
        .def_property_readonly("memory",[](const Builder& b){return b.memory();})
        .def_property_readonly("diagnostics",[](const Builder& b){return b.diagnostics();})
        .def_property_readonly("state",&Builder::state_handle)
        .def_property_readonly("context",&Builder::context_handle)
        .def_property_readonly("basis_identity_sha256",&Builder::basis_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256",&Builder::hf_reference_source_identity_sha256)
        .def_property_readonly("localization_identity_sha256",&Builder::localization_identity_sha256)
        .def_property_readonly("identity_sha256",&Builder::identity_sha256)
        .def_property_readonly("payload_sha256",&Builder::payload_sha256)
        .def_property_readonly("allocation_identity",&Builder::allocation_identity)
        .def_property_readonly("retained_numerical_bytes",&Builder::retained_numerical_bytes)
        .def_property_readonly("retained_control_storage_bytes",&Builder::retained_control_storage_bytes)
        .def_property_readonly("production_distinct_domain_scaling",&Builder::production_distinct_domain_scaling);
    auto options=py::class_<PairOptions>(m,"_PeriodicGaussianPairDomainOptions").def(py::init<>());
    options.def_readwrite("domain",&PairOptions::domain)
        .def_readwrite("real_space",&PairOptions::real_space)
        .def_readwrite("embedding",&PairOptions::embedding);
    auto pcaps=py::class_<PairCaps>(m,"_PeriodicGaussianPairDomainCaps").def(py::init<>());
#define GPDB_PAIR_CAP(f) pcaps.def_readwrite(#f,&PairCaps::f)
    GPDB_PAIR_CAP(pair_union); GPDB_PAIR_CAP(maximum_pao_domain_owned_numerical_bytes);
    GPDB_PAIR_CAP(real_space); GPDB_PAIR_CAP(embedding); GPDB_PAIR_CAP(maximum_owned_numerical_bytes);
    GPDB_PAIR_CAP(maximum_control_storage_bytes); GPDB_PAIR_CAP(maximum_worker_bytes);
    GPDB_PAIR_CAP(maximum_node_bytes); GPDB_PAIR_CAP(maximum_work_units);
#undef GPDB_PAIR_CAP
    auto pplan=py::class_<PairPlan>(m,"_PeriodicGaussianPairDomainEmbeddingPlan");
#define GPDB_PAIR_PLAN(f) pplan.def_readonly(#f,&PairPlan::f)
    GPDB_PAIR_PLAN(n_cells); GPDB_PAIR_PLAN(n_basis); GPDB_PAIR_PLAN(common_virtual_dimension);
    GPDB_PAIR_PLAN(maximum_generation_dimension); GPDB_PAIR_PLAN(occupied_slot_i); GPDB_PAIR_PLAN(occupied_slot_j);
    GPDB_PAIR_PLAN(borrowed_builder_numerical_bytes); GPDB_PAIR_PLAN(borrowed_basis_numerical_bytes);
    GPDB_PAIR_PLAN(retained_embedding_upper_bytes); GPDB_PAIR_PLAN(peak_owned_numerical_bytes);
    GPDB_PAIR_PLAN(maximum_domain_dimension); GPDB_PAIR_PLAN(retained_pair_geometry_upper_bytes);
    GPDB_PAIR_PLAN(retained_geometry_control_upper_bytes);
    GPDB_PAIR_PLAN(control_storage_reservation_bytes); GPDB_PAIR_PLAN(metadata_validation_work_units);
    GPDB_PAIR_PLAN(pao_domain_work_upper); GPDB_PAIR_PLAN(child_work_upper); GPDB_PAIR_PLAN(work_units);
    GPDB_PAIR_PLAN(replicas_per_node); GPDB_PAIR_PLAN(reference_base_node_bytes); GPDB_PAIR_PLAN(worker_bytes);
    GPDB_PAIR_PLAN(required_node_memory_bytes);
#undef GPDB_PAIR_PLAN
    const auto geometry_tiny=[](const Geometry& g) {
        if(g.domain().domain_dimension()>96 || g.real_space().space().retained_dimension()>4
            || g.embedding().memory().common_dimension>4)
            throw std::length_error("Gaussian pair geometry copy exceeds tiny diagnostic dimensions");
    };
    auto geometry=py::class_<Geometry>(m,"_PeriodicGaussianPairDomainGeometry");
    geometry.def_property_readonly("state",&Geometry::state_handle)
        .def_property_readonly("context",&Geometry::context_handle)
        .def_property_readonly("occupied_slot_i",&Geometry::occupied_slot_i)
        .def_property_readonly("occupied_slot_j",&Geometry::occupied_slot_j)
        .def_property_readonly("domain_dimension",[](const Geometry& g){return g.domain().domain_dimension();})
        .def_property_readonly("generation_dimension",[](const Geometry& g){return g.real_space().space().retained_dimension();})
        .def_property_readonly("retained_numerical_bytes",&Geometry::retained_numerical_bytes)
        .def_property_readonly("retained_control_storage_bytes",&Geometry::retained_control_storage_bytes)
        .def_property_readonly("identity_sha256",&Geometry::identity_sha256)
        .def_property_readonly("builder_identity_sha256",&Geometry::builder_identity_sha256)
        .def_property_readonly("basis_identity_sha256",&Geometry::basis_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256",&Geometry::hf_reference_source_identity_sha256)
        .def_property_readonly("allocation_identity",&Geometry::allocation_identity)
        .def_property_readonly("embedding_identity_sha256",[](const Geometry& g){return g.embedding().identity_sha256();})
        .def_property_readonly("pair_translation",[](const Geometry& g){return g.embedding().pair_selection().translation_cell;})
        .def("geometry_view",[](const Geometry& g){return g.geometry_view();},py::keep_alive<0,1>())
        .def("domain_columns_copy",[geometry_tiny](const Geometry& g) {
            geometry_tiny(g);const auto& d=g.domain();py::array_t<std::uint64_t> out({static_cast<py::ssize_t>(d.domain_dimension()),py::ssize_t(2)});
            for(std::size_t i=0;i<d.domain_dimension();++i) {const auto c=d.column(i);out.mutable_data()[2*i]=c.cell;out.mutable_data()[2*i+1]=c.ao;}
            out.attr("setflags")(false);return out;
        })
        .def("real_coefficients_copy",[geometry_tiny](const Geometry& g) {
            geometry_tiny(g);const auto& s=g.real_space().space();
            py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(s.domain_dimension()),static_cast<py::ssize_t>(s.retained_dimension())});
            std::copy_n(s.coefficients_data(),out.size(),out.mutable_data());out.attr("setflags")(false);return out;
        })
        .def("embedding_coefficients_copy",[geometry_tiny](const Geometry& g) {
            geometry_tiny(g);const auto& e=g.embedding();py::array_t<double> out({static_cast<py::ssize_t>(e.memory().common_dimension),
                static_cast<py::ssize_t>(e.memory().pair_dimension)});std::copy_n(e.coefficients_data(),out.size(),out.mutable_data());
            out.attr("setflags")(false);return out;
        })
        .def("energies_copy",[geometry_tiny](const Geometry& g) {
            geometry_tiny(g);const auto& s=g.real_space().space();py::array_t<double> out(static_cast<py::ssize_t>(s.retained_dimension()));
            std::copy_n(s.energies_data(),out.size(),out.mutable_data());out.attr("setflags")(false);return out;
        });
    const auto tiny=[](const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis) {
        if(!ref.state_handle() || ref.state().n_basis()>12 || ref.state().n_kpoints()>8
            || basis.memory().occupied_count>4 || basis.memory().virtual_count>4)
            throw std::length_error("Gaussian pair domain builder exceeds tiny diagnostic dimensions");
    };
    const auto bounded=[](const auto& p) {
        if(p.peak_owned_numerical_bytes>(8U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units>100000000000000ULL)
            throw std::length_error("Gaussian pair domain builder exceeds tiny diagnostic memory/work limits");
    };
    m.def("_plan_periodic_gaussian_pair_domain_builder",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicGaussianLocalizationResult& localization,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationRealPAOSpace& space,
        OccupiedOptions options,Live live,Caps caps) {
        tiny(ref,basis);
        const auto p=plan_periodic_gaussian_pair_domain_builder(hf,ref,localization,basis,domain,space,options,live,caps);
        bounded(p);return p;
    },py::arg("hf"),py::arg("reference"),py::arg("localization"),py::arg("basis"),
      py::arg("common_domain"),py::arg("common_space"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_domain_builder",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
        const BasisSet& ao,const BasisSet& auxiliary,const BasisSet& minimal,const PeriodicSystem& system,
        const PeriodicGaussianLocalizationResult& localization,const PeriodicCorrelationRealLocalBasis& basis,
        PeriodicCorrelationPAODomain& domain,PeriodicCorrelationRealPAOSpace& space,
        OccupiedOptions options,Live live,Caps caps) {
        tiny(ref,basis);
        bounded(plan_periodic_gaussian_pair_domain_builder(hf,ref,localization,basis,domain,space,options,live,caps));
        // GIL held and controls copied; native input arguments pin every
        // borrowed owner through both validation and final noexcept moves.
        return make_periodic_gaussian_pair_domain_builder(hf,ref,ao,auxiliary,minimal,system,
            localization,basis,std::move(domain),std::move(space),options,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("minimal_basis"),
      py::arg("system"),py::arg("localization"),py::arg("basis"),py::arg("common_domain"),py::arg("common_space"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_plan_periodic_gaussian_pair_domain_embedding",[tiny,bounded](
        const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const Builder& builder,std::uint64_t i,std::uint64_t j,PairOptions options,Live live,PairCaps caps) {
        tiny(ref,basis);
        const auto p=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options,live,caps);
        bounded(p);return p;
    },py::arg("reference"),py::arg("basis"),py::arg("builder"),py::arg("occupied_slot_i"),py::arg("occupied_slot_j"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_domain_embedding",[tiny,bounded](
        const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const Builder& builder,std::uint64_t i,std::uint64_t j,PairOptions options,Live live,PairCaps caps) {
        tiny(ref,basis);
        bounded(plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options,live,caps));
        return make_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("builder"),py::arg("occupied_slot_i"),py::arg("occupied_slot_j"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_pair_domain_geometry",[tiny,bounded](
        const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const Builder& builder,std::uint64_t i,std::uint64_t j,PairOptions options,Live live,PairCaps caps) {
        tiny(ref,basis);bounded(plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options,live,caps));
        return make_periodic_gaussian_pair_domain_geometry(ref,basis,builder,i,j,options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("builder"),py::arg("occupied_slot_i"),py::arg("occupied_slot_j"),
      py::arg("options"),py::arg("live"),py::arg("caps"));
}
