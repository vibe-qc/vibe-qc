// Included by bindings.cpp. Bounded diagnostics over actual immutable native
// PNO/PairSpace owners; no caller coefficient arrays or mutable child escape.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"
#include "vibeqc/periodic_gaussian_pair_space.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_mixed_pair_factors(py::module_& m) {
    using namespace vibeqc;
    using Frame=PeriodicGaussianPairPNOFrameView;
    using Geometry=PeriodicGaussianPairPNOGeometryView;
    using Storage=PeriodicGaussianMixedPairFactorStorage;
    using Plan=PeriodicGaussianMixedPairFactorPlan;
    using Panel=PeriodicGaussianMixedPairFactorPanel;
    using Config=PeriodicGaussianLocalOrbitalFactorConfig;
    using Live=PeriodicGaussianLocalOrbitalFactorLiveInventory;
    using Caps=PeriodicGaussianLocalOrbitalFactorCaps;
    using U=std::uint64_t;
    using C=std::complex<double>;
    py::class_<Geometry>(m,"_PeriodicGaussianPairPNOGeometryView")
        .def(py::init<const PeriodicCorrelationPAODomain&,const PeriodicCorrelationRealPAOSpace&,
            const PeriodicCorrelationRealPAOEmbedding&>(),
            py::arg("domain"),py::arg("real_space"),py::arg("embedding"),
            py::keep_alive<1,2>(),py::keep_alive<1,3>(),py::keep_alive<1,4>());
    const auto add=[](U a,U b) {
        if(b>std::numeric_limits<U>::max()-a) throw std::overflow_error("mixed-factor diagnostic inventory overflows");
        return a+b;
    };
    const auto frame=[](const py::object& value)->Frame {
        if(py::isinstance<PeriodicGaussianPairPNOResult>(value))
            return Frame(value.cast<const PeriodicGaussianPairPNOResult&>());
        if(py::isinstance<PeriodicGaussianEmbeddedPairPNOResult>(value))
            return Frame(value.cast<const PeriodicGaussianEmbeddedPairPNOResult&>());
        if(py::isinstance<PeriodicGaussianGramPairPNOResult>(value))
            return Frame(value.cast<const PeriodicGaussianGramPairPNOResult&>());
        if(py::isinstance<PeriodicGaussianPairSpace>(value))
            return value.cast<const PeriodicGaussianPairSpace&>().frame();
        throw std::invalid_argument("mixed factors require an actual native PNO or PairSpace owner");
    };
    const auto live_with_wrappers=[add](const py::object& a,const py::object& b,Live live) {
        const PeriodicGaussianPairSpace* previous=nullptr;
        for(const auto* value:{&a,&b}) {
            if(!py::isinstance<PeriodicGaussianPairSpace>(*value)) continue;
            const auto* space=&value->cast<const PeriodicGaussianPairSpace&>();
            if(space==previous) continue;
            previous=space;
            // PNO numerical payload is counted by the native leaf. The
            // immediate legacy PairSpace also retains G; direct Gram frames
            // already own it, so do not count that same array twice. Count
            // entire wrapper/receipt controls (including embedded controls).
            const auto lanes=space->frame().is_direct_gram()?0:space->exchange_integrals().size();
            if(lanes>std::numeric_limits<U>::max()/8) throw std::overflow_error("mixed-factor PairSpace extent overflows");
            live.other_retained_bytes_per_worker=add(live.other_retained_bytes_per_worker,
                add(U(lanes)*8,sizeof(PeriodicGaussianPairSpace)+7*65));
        }
        return live;
    };
    const auto tiny=[](const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianReciprocalSource& source,const Frame& a,const Frame& b) {
        if(!hf.context_handle() || hf.plan().n_kpoints>8 || hf.plan().n_basis>4
            || hf.context_handle()->inventory().auxiliary.function_count>4
            || basis.memory().occupied_count>8 || basis.memory().virtual_count>8
            || a.retained_dimension()>8 || b.retained_dimension()>8 || source.candidate_count()>65536)
            throw std::length_error("Gaussian mixed-pair diagnostic exceeds tiny shape/source bounds");
    };
    const auto bounded=[](const Plan& p) {
        if(p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.tile_calls>512 || p.work_units>1000000000000000ULL)
            throw std::length_error("Gaussian mixed-pair diagnostic exceeds tiny memory/work bounds");
    };
    auto storage=py::class_<Storage>(m,"_PeriodicGaussianMixedPairFactorStorage");
#define MIX_STORAGE(field) storage.def_readonly(#field,&Storage::field)
    MIX_STORAGE(orbital_count);MIX_STORAGE(pno_column_count);MIX_STORAGE(tile_calls);MIX_STORAGE(contraction_term_count);
    MIX_STORAGE(retained_factor_bytes);MIX_STORAGE(retained_output_bytes);MIX_STORAGE(compensation_bytes);
    MIX_STORAGE(coefficient_panel_bytes);MIX_STORAGE(coefficient_helper_workspace_bytes);
    MIX_STORAGE(driver_owned_numerical_bytes);MIX_STORAGE(coefficient_work_units_per_kpoint);MIX_STORAGE(coefficient_work_units);
#undef MIX_STORAGE
    auto plan=py::class_<Plan,Storage>(m,"_PeriodicGaussianMixedPairFactorPlan");
#define MIX_PLAN(field) plan.def_readonly(#field,&Plan::field)
    MIX_PLAN(n_cells);MIX_PLAN(n_basis);MIX_PLAN(n_auxiliary);MIX_PLAN(occupied_count);
    MIX_PLAN(common_virtual_dimension);MIX_PLAN(common_domain_dimension);MIX_PLAN(rank_a);MIX_PLAN(rank_b);
    MIX_PLAN(occupied_slot_i);MIX_PLAN(occupied_slot_k);MIX_PLAN(q_index);MIX_PLAN(auxiliary_begin);MIX_PLAN(auxiliary_count);
    MIX_PLAN(ao_pair_block);MIX_PLAN(same_frame_owner);MIX_PLAN(borrowed_frame_numerical_bytes);
    MIX_PLAN(direct_gram_frames);
    MIX_PLAN(borrowed_frame_control_bytes);MIX_PLAN(frame_validation_work_units);
    MIX_PLAN(local_geometry_a);MIX_PLAN(local_geometry_b);MIX_PLAN(borrowed_geometry_numerical_bytes);
    MIX_PLAN(borrowed_geometry_control_bytes);MIX_PLAN(geometry_validation_work_units);
    MIX_PLAN(maximum_tile_owned_numerical_bytes);MIX_PLAN(peak_owned_numerical_bytes);MIX_PLAN(resident_whitener_bytes);
    MIX_PLAN(borrowed_basis_active_numeric_bytes);MIX_PLAN(caller_gauge_bytes);MIX_PLAN(live_wannier_bytes);
    MIX_PLAN(live_domain_bytes);MIX_PLAN(live_space_bytes);MIX_PLAN(live_basis_bytes);MIX_PLAN(state_resident_bytes);
    MIX_PLAN(macro_fixed_object_bytes);MIX_PLAN(tile_fixed_object_bytes);MIX_PLAN(replicas_per_node);
    MIX_PLAN(reference_base_node_bytes);MIX_PLAN(per_replica_inventoried_bytes);MIX_PLAN(required_node_memory_bytes);
    MIX_PLAN(reciprocal_candidate_evaluations);MIX_PLAN(image_candidate_evaluations_upper_bound);
    MIX_PLAN(driver_work_units);MIX_PLAN(work_units);
#undef MIX_PLAN
    plan.def_property_readonly("config",[](const Plan& p){return p.config;})
        .def_property_readonly("live",[](const Plan& p){return p.live;})
        .def_property_readonly("caps",[](const Plan& p){return p.caps;})
        .def_property_readonly("tile_config",[](const Plan& p){return p.tile_config;})
        .def_property_readonly("plan_identity_sha256",&Plan::plan_identity_sha256);
    auto panel=py::class_<Panel>(m,"_PeriodicGaussianMixedPairFactorPanel");
    panel.def_property_readonly("context",&Panel::context_handle)
        .def_property_readonly("state",&Panel::state_handle)
        .def_property_readonly("memory",[](const Panel& p){return p.memory();})
        .def_property_readonly("diagnostics",[](const Panel& p){return p.diagnostics();})
        .def_property_readonly("q_index",&Panel::q_index)
        .def_property_readonly("auxiliary_begin",&Panel::auxiliary_begin)
        .def_property_readonly("auxiliary_count",&Panel::auxiliary_count)
        .def_property_readonly("orbital_count",&Panel::orbital_count)
        .def_property_readonly("rank_a",&Panel::rank_a).def_property_readonly("rank_b",&Panel::rank_b)
        .def_property_readonly("occupied_i",[](const Panel& p){const auto x=p.occupied_i();return py::make_tuple(x.occupied_index,x.cell);})
        .def_property_readonly("occupied_k",[](const Panel& p){const auto x=p.occupied_k();return py::make_tuple(x.occupied_index,x.cell);})
        .def("element",&Panel::element)
        .def("tensor_copy",[](const Panel& p){
            const auto& plan=p.memory();
            if(plan.n_cells>8 || plan.n_basis>4 || plan.n_auxiliary>4 || plan.orbital_count>18
                || plan.retained_factor_bytes>(1U<<20))
                throw std::length_error("Gaussian mixed-pair diagnostic copy exceeds tiny shape");
            const auto* values=p.data();
            py::array_t<C> result({py::ssize_t(p.auxiliary_count()),py::ssize_t(p.orbital_count()),py::ssize_t(p.orbital_count())});
            std::memcpy(result.mutable_data(),values,plan.retained_factor_bytes);
            result.attr("setflags")(false);return result;
        });
#define MIX_RESULT(field) panel.def_property_readonly(#field,&Panel::field)
    MIX_RESULT(finite_image_reference);MIX_RESULT(matched_finite_gaussian_hf_recipe);
    MIX_RESULT(jointly_orthonormal_basis_certified);MIX_RESULT(density_symmetry_certified);
    MIX_RESULT(bitwise_origin_provider_factors_verified);MIX_RESULT(production_dlpno);
    MIX_RESULT(identity_sha256);MIX_RESULT(payload_sha256);MIX_RESULT(hf_reference_source_identity_sha256);
    MIX_RESULT(local_basis_identity_sha256);MIX_RESULT(basis_certificate_identity_sha256);
    MIX_RESULT(source_identity_sha256);MIX_RESULT(conjugate_source_identity_sha256);
    MIX_RESULT(whitener_payload_identity_sha256);MIX_RESULT(consumed_tiles_identity_sha256);
    MIX_RESULT(frame_a_identity_sha256);MIX_RESULT(frame_b_identity_sha256);
    MIX_RESULT(frame_a_payload_sha256);MIX_RESULT(frame_b_payload_sha256);
    MIX_RESULT(frame_a_provider_identity_sha256);MIX_RESULT(frame_b_provider_identity_sha256);
    MIX_RESULT(direct_gram_frames);MIX_RESULT(frame_a_gram_identity_sha256);MIX_RESULT(frame_b_gram_identity_sha256);
#undef MIX_RESULT
    m.def("_periodic_gaussian_mixed_pair_factor_storage",&periodic_gaussian_mixed_pair_factor_storage,
        py::arg("n_cells"),py::arg("n_basis"),py::arg("n_effective_orbitals"),py::arg("n_common"),py::arg("d_common"),
        py::arg("rank_a"),py::arg("rank_b"),py::arg("auxiliary_count"),py::arg("ao_pair_block"));
    m.def("_plan_periodic_gaussian_mixed_pair_factor_panel",[frame,live_with_wrappers,tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
        const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
        py::object a,py::object b,U i,U k,U begin,U count,Config config,Live live,Caps caps,
        const Geometry* ga,const Geometry* gb) {
        const auto fa=frame(a),fb=frame(b);tiny(hf,basis,source,fa,fb);
        live=live_with_wrappers(a,b,live);
        const auto p=plan_periodic_gaussian_mixed_pair_factor_panel(hf,ref,source,w,wannier,domain,space,basis,
            fa,fb,i,k,begin,count,config,live,caps,ga,gb);bounded(p);return p;
    },py::arg("hf"),py::arg("reference"),py::arg("source"),py::arg("whitener"),py::arg("wannier"),
      py::arg("domain"),py::arg("space"),py::arg("basis"),py::arg("frame_a"),py::arg("frame_b"),
      py::arg("occupied_slot_i"),py::arg("occupied_slot_k"),py::arg("auxiliary_begin"),py::arg("auxiliary_count"),
      py::arg("config"),py::arg("live"),py::arg("caps"),
      py::arg("geometry_a")=nullptr,py::arg("geometry_b")=nullptr);
    m.def("_build_periodic_gaussian_mixed_pair_factor_panel",[frame,live_with_wrappers,tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
        py::array gauges,const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
        const PeriodicCorrelationRealLocalBasis& basis,py::object a,py::object b,
        U i,U k,U begin,U count,Config config,Live live,Caps caps,const Geometry* ga,const Geometry* gb) {
        const auto fa=frame(a),fb=frame(b);tiny(hf,basis,source,fa,fb);
        live=live_with_wrappers(a,b,live);
        const auto p=plan_periodic_gaussian_mixed_pair_factor_panel(hf,ref,source,w,wannier,domain,space,basis,
            fa,fb,i,k,begin,count,config,live,caps,ga,gb);bounded(p);
        if(!gauges.dtype().is(py::dtype::of<C>()) || !(gauges.flags()&py::array::c_style)
            || reinterpret_cast<std::uintptr_t>(gauges.data())%alignof(C) || gauges.ndim()!=3
            || gauges.shape(0)!=py::ssize_t(p.n_cells) || gauges.shape(1)!=py::ssize_t(ref.state().n_correlated_occupied())
            || gauges.shape(2)!=gauges.shape(1))
            throw std::invalid_argument("Gaussian mixed-pair gauges require aligned contiguous complex128 [Nk,no,no]");
        // GIL and original Python owners stay pinned throughout the native
        // call. No callback, forcecast, provider or numerical frame copy.
        return build_periodic_gaussian_mixed_pair_factor_panel(hf,ref,source,w,ao,auxiliary,wannier,
            static_cast<const C*>(gauges.data()),std::size_t(gauges.size()),domain,space,basis,
            fa,fb,i,k,begin,count,config,live,caps,ga,gb);
    },py::arg("hf"),py::arg("reference"),py::arg("source"),py::arg("whitener"),py::arg("ao_basis"),
      py::arg("auxiliary_basis"),py::arg("wannier"),py::arg("gauges").noconvert(),py::arg("domain"),py::arg("space"),
      py::arg("basis"),py::arg("frame_a"),py::arg("frame_b"),py::arg("occupied_slot_i"),py::arg("occupied_slot_k"),
      py::arg("auxiliary_begin"),py::arg("auxiliary_count"),py::arg("config"),py::arg("live"),py::arg("caps"),
      py::arg("geometry_a")=nullptr,py::arg("geometry_b")=nullptr);
}
