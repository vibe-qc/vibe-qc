// Tiny actual all-q CCSD diagnostics. No supplied amplitudes, matrices,
// factor provider callback, or caller-controlled source qualification.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <limits>
#include "vibeqc/periodic_gaussian_pair_ccsd_physical_particle_hole.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace periodic_gaussian_pair_ccsd_physical_particle_hole_python {
using namespace vibeqc;
using U = std::uint64_t;
using Complex = std::complex<double>;
using HF = PeriodicGaussianRHFResult;
using Reference = PeriodicCorrelationAdmittedReference;
using Wannier = PeriodicCorrelationWannier;
using Domain = PeriodicCorrelationPAODomain;
using Space = PeriodicCorrelationPAOSpace;
using Builder = PeriodicGaussianPairDomainBuilder;
using Basis = PeriodicCorrelationRealLocalBasis;
using Provider = PeriodicGaussianRealLocalProvider;
using Warm = PeriodicGaussianPairMP2Result;
using Config = PeriodicGaussianPairInteractionConfig;
using InteractionOptions = PeriodicGaussianPairInteractionOptions;
using PHCaps = PeriodicGaussianPairParticleHoleCaps;
using Options = PeriodicGaussianPairCCSDOptions;
using Live = PeriodicGaussianPairCCSDLiveInventory;
using Caps = PeriodicGaussianPairCCSDCaps;
using Plan = PeriodicGaussianPairCCSDPhysicalParticleHolePlan;
using Progress = PeriodicGaussianPairCCSDProgress;

U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a)
        throw std::overflow_error("physical particle-hole CCSD binding inventory overflows");
    return a+b;
}

struct Common {
    const Domain* domain = nullptr;
    const Space* space = nullptr;
    Live live;
};

// Scalar descriptor pin only; no copied gauge payload. A Python callback can
// resize an ndarray even while it is pinned, so descriptor validation must
// run BEFORE the native callback boundary may dereference the old pointer.
struct GaugePin {
    const py::array* array = nullptr;
    const Complex* data = nullptr;
    std::size_t count = 0;
    std::array<py::ssize_t, 3> shape{}, strides{};

    GaugePin(const py::array& a, const HF& hf) : array(&a) {
        const auto state = hf.state_handle();
        if (!state || !a.dtype().is(py::dtype::of<Complex>()) || !(a.flags()&py::array::c_style)
            || a.ndim()!=3 || a.shape(0)!=static_cast<py::ssize_t>(state->n_kpoints())
            || a.shape(1)!=static_cast<py::ssize_t>(state->n_correlated_occupied())
            || a.shape(2)!=a.shape(1) || reinterpret_cast<std::uintptr_t>(a.data())%alignof(Complex))
            throw py::type_error("physical particle-hole CCSD gauges require aligned contiguous complex128 [K,nocc,nocc]");
        data = static_cast<const Complex*>(a.data());
        count = static_cast<std::size_t>(a.size());
        for (std::size_t d=0;d<3;++d) { shape[d]=a.shape(d); strides[d]=a.strides(d); }
    }

    void validate() const {
        const auto& a = *array;
        if (!a.dtype().is(py::dtype::of<Complex>()) || !(a.flags()&py::array::c_style)
            || a.ndim()!=3 || a.data()!=data || static_cast<std::size_t>(a.size())!=count)
            throw std::invalid_argument("physical particle-hole CCSD progress changed gauge descriptors");
        for (std::size_t d=0;d<3;++d)
            if (a.shape(d)!=shape[d] || a.strides(d)!=strides[d])
                throw std::invalid_argument("physical particle-hole CCSD progress changed gauge descriptors");
    }
};

struct Callback {
    py::object* function = nullptr;
    const GaugePin* gauges = nullptr;
    static void notify(const Progress& event, void* opaque) {
        auto& self = *static_cast<Callback*>(opaque);
        (*self.function)(py::cast(Progress(event), py::return_value_policy::move));
        self.gauges->validate();
    }
};

constexpr U fixed_binding_control_bytes = 4096+sizeof(Common)+sizeof(GaugePin)+sizeof(Callback)
    +2*sizeof(Config)+2*sizeof(InteractionOptions)+2*sizeof(PHCaps)
    +2*sizeof(Options)+2*sizeof(Live)+2*sizeof(Caps)+2*sizeof(Plan)+8*sizeof(py::object);

Common common(const py::object& owner, const py::object& space, Live live) {
    live.other_live_bytes_per_worker=add(live.other_live_bytes_per_worker,fixed_binding_control_bytes);
    if (py::isinstance<Builder>(owner)) {
        if (!space.is_none())
            throw py::type_error("physical particle-hole CCSD builder common_geometry requires space=None");
        const auto& b = owner.cast<const Builder&>();
        const auto& d = b.common_domain();
        const auto& s = b.common_real_space().space();
        const U common_bytes=add(d.memory().retained_domain_index_bytes,
            add(d.memory().retained_matrix_bytes,s.memory().output_numerical_bytes));
        const U whole=b.retained_numerical_bytes();
        if (common_bytes>whole)
            throw std::logic_error("physical particle-hole CCSD common geometry exceeds builder inventory");
        // CCSD Live is a combined byte channel. The native wrapper already
        // counts common domain/space numerics; preserve every other builder
        // byte plus its complete controls, with no hidden owner subtraction.
        live.other_live_bytes_per_worker=add(live.other_live_bytes_per_worker,
            add(whole-common_bytes,b.retained_control_storage_bytes()));
        return {&d,&s,live};
    }
    if (!py::isinstance<Domain>(owner) || !py::isinstance<Space>(space))
        throw py::type_error("physical particle-hole CCSD common_geometry requires PAODomain/PAOSpace or Builder/None");
    const auto& d = owner.cast<const Domain&>();
    const auto& s = space.cast<const Space&>();
    if (!d.state_handle() || !s.state_handle())
        throw std::invalid_argument("physical particle-hole CCSD common geometry is moved or incomplete");
    return {&d,&s,live};
}

void tiny_shape(const HF& hf, const Reference& ref, const Basis& basis,
    const Options& options) {
    if (!hf.context_handle() || !ref.state_handle() || ref.state().n_kpoints()>8 || hf.plan().n_kpoints>8
        || hf.plan().n_basis>4 || hf.context_handle()->inventory().auxiliary.function_count>8
        || basis.memory().occupied_count>4 || basis.memory().virtual_count>4
        || hf.plan().fock.config.source_caps.maximum_candidate_evaluations>65536
        || options.singles.pno_eigensolver.max_sweeps>200
        || options.singles.semicanonical_eigensolver.max_sweeps>200
        || options.solver.maximum_iterations>128)
        throw std::length_error("physical particle-hole CCSD diagnostic exceeds tiny shape/source/iteration bounds");
}

Plan plan(const HF& hf, const Reference& ref, const Wannier& wannier,
    const Common& common_owner, const Basis& basis, const Provider& provider, const Warm& warm,
    const Config& config, const InteractionOptions& interaction_options, const PHCaps& ph_caps,
    const Options& options, const Caps& caps) {
    const auto p=plan_periodic_gaussian_pair_ccsd_physical_particle_hole(hf,ref,wannier,
        *common_owner.domain,*common_owner.space,basis,provider,warm,config,interaction_options,ph_caps,
        options,common_owner.live,caps);
    if (p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
        || p.work_units_upper_bound>10000000000000000000ULL
        || p.ccsd.progress_callback_upper_bound>1024 || p.particle_hole_calls_upper_bound>1280)
        throw std::length_error("physical particle-hole CCSD diagnostic exceeds tiny memory/work/callback bounds");
    return p;
}
} // namespace periodic_gaussian_pair_ccsd_physical_particle_hole_python

void bind_periodic_gaussian_pair_ccsd_physical_particle_hole(py::module_& m) {
    namespace binding=periodic_gaussian_pair_ccsd_physical_particle_hole_python;
    using namespace binding;
    auto p=py::class_<Plan>(m,"_PeriodicGaussianPairCCSDPhysicalParticleHolePlan");
    p.def_property_readonly("ccsd",[](const Plan& plan){return plan.ccsd;});
#define GPCCPH_PLAN(field) p.def_readonly(#field,&Plan::field)
    GPCCPH_PLAN(occupied_count);GPCCPH_PLAN(common_virtual_dimension);GPCCPH_PLAN(pair_count);
    GPCCPH_PLAN(maximum_pair_rank);GPCCPH_PLAN(pair_coefficient_bytes);
    GPCCPH_PLAN(maximum_frame_numerical_bytes);GPCCPH_PLAN(maximum_geometry_numerical_bytes);
    GPCCPH_PLAN(domain_generated);GPCCPH_PLAN(borrowed_warmstart_numerical_bytes);GPCCPH_PLAN(borrowed_basis_numerical_bytes);
    GPCCPH_PLAN(additional_retained_numerical_bytes);GPCCPH_PLAN(duplicate_frame_geometry_padding_bytes);
    GPCCPH_PLAN(duplicate_t_role_padding_bytes);GPCCPH_PLAN(producer_transient_upper_bytes);
    GPCCPH_PLAN(wrapper_fixed_control_storage_bytes);GPCCPH_PLAN(producer_additional_control_storage_bytes);
    GPCCPH_PLAN(metadata_work_units);GPCCPH_PLAN(source_validation_work_units_per_pass);
    GPCCPH_PLAN(source_validation_passes_upper_bound);GPCCPH_PLAN(finalization_work_units);
    GPCCPH_PLAN(wrapper_work_units_upper_bound);GPCCPH_PLAN(work_units_upper_bound);
    GPCCPH_PLAN(particle_hole_calls_upper_bound);GPCCPH_PLAN(factor_panels_upper_bound);GPCCPH_PLAN(tile_calls_upper_bound);
    GPCCPH_PLAN(reciprocal_candidates_upper_bound);GPCCPH_PLAN(image_candidates_upper_bound);
    GPCCPH_PLAN(peak_owned_numerical_bytes);GPCCPH_PLAN(per_worker_inventoried_bytes);GPCCPH_PLAN(required_node_memory_bytes);
#undef GPCCPH_PLAN
    m.def("_plan_periodic_gaussian_pair_ccsd_physical_particle_hole_diagnostic",[](const HF& hf,
        const Reference& ref,const Wannier& wannier,py::object geometry,py::object space,
        const Basis& basis,const Provider& provider,const Warm& warm,Config config,
        InteractionOptions interaction_options,PHCaps ph_caps,Options options,Live live,Caps caps) {
        tiny_shape(hf,ref,basis,options);
        const auto c=common(geometry,space,live);
        return plan(hf,ref,wannier,c,basis,provider,warm,config,interaction_options,ph_caps,options,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("common_geometry"),py::arg("space"),
        py::arg("basis"),py::arg("provider"),py::arg("warmstart"),py::arg("config"),py::arg("interaction_options"),
        py::arg("particle_hole_caps"),py::arg("options"),py::arg("live"),py::arg("caps"));

    m.def("_run_periodic_gaussian_pair_ccsd_physical_particle_hole_diagnostic",[](const HF& hf,
        const Reference& ref,const BasisSet& ao,const BasisSet& auxiliary,const Wannier& wannier,
        py::array gauges,py::object geometry,py::object space,const Basis& basis,const Provider& provider,
        const Warm& warm,Config config,InteractionOptions interaction_options,PHCaps ph_caps,
        Options options,Live live,Caps caps,py::object callback) {
        tiny_shape(hf,ref,basis,options);
        if (ao.nbasis()>4 || auxiliary.nbasis()>8)
            throw std::length_error("physical particle-hole CCSD diagnostic exceeds tiny actual basis bounds");
        const auto c=common(geometry,space,live);
        (void)plan(hf,ref,wannier,c,basis,provider,warm,config,interaction_options,ph_caps,options,caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("physical particle-hole CCSD progress must be callable or None");
        const GaugePin pin(gauges,hf);
        Callback progress{&callback,&pin};
        // All mutable controls above are value copies. Keep the GIL and all
        // authentic owners pinned through scalar callbacks. Native source
        // boundaries audit gauge VALUES and physical owner payloads too.
        return run_periodic_gaussian_pair_ccsd_physical_particle_hole(hf,ref,ao,auxiliary,wannier,
            pin.data,pin.count,*c.domain,*c.space,basis,provider,warm,config,interaction_options,ph_caps,
            options,c.live,caps,callback.is_none()?nullptr:&Callback::notify,
            callback.is_none()?nullptr:&progress);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),
        py::arg("gauges").noconvert(),py::arg("common_geometry"),py::arg("space"),py::arg("basis"),py::arg("provider"),
        py::arg("warmstart"),py::arg("config"),py::arg("interaction_options"),py::arg("particle_hole_caps"),
        py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("progress")=py::none());
}
