// Tiny actual-source diagnostics. Numerical selection/PNOs/iteration stay C++.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include "vibeqc/periodic_gaussian_domain_pair_mp2.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

namespace gaussian_gram_domain_pair_mp2_python {
using namespace vibeqc;
using Complex=std::complex<double>;
using Config=PeriodicGaussianGramDomainPairMP2Config;
using Options=PeriodicGaussianGramDomainPairMP2Options;
using Caps=PeriodicGaussianGramDomainPairMP2Caps;
using Live=PeriodicGaussianPairMP2LiveInventory;
using Progress=PeriodicGaussianPairMP2Progress;

PeriodicGaussianPairMP2Plan tiny(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairDomainBuilder& builder,
    const Config& config,const Options& options,const Live& live,const Caps& caps) {
    if (!ref.state_handle() || !hf.context_handle() || ref.state().n_kpoints()>8
        || ref.state().n_basis()>4 || hf.context_handle()->inventory().auxiliary.function_count>4
        || basis.memory().occupied_count>4 || basis.memory().virtual_count>4
        || options.solver.maximum_iterations>256 || options.pnos.pno.pno.pno_eigensolver.max_sweeps>200
        || options.pnos.pno.pno.semicanonical_eigensolver.max_sweeps>200
        || hf.plan().fock.config.source_caps.maximum_candidate_evaluations>65536)
        throw std::length_error("Gaussian Gram domain pair MP2 exceeds tiny diagnostic source/dimension/iteration bounds");
    auto p=plan_periodic_gaussian_domain_pair_mp2(hf,ref,wannier,basis,builder,config,options,live,caps);
    if (p.peak_owned_numerical_bytes>(16ULL<<20) || p.required_node_memory_bytes>(128ULL<<20)
        || p.work_units_upper_bound>10000000000000000000ULL || p.progress_callback_upper_bound>1024)
        throw std::length_error("Gaussian Gram domain pair MP2 exceeds tiny diagnostic memory/work/progress bounds");
    return p;
}

struct GaugePin {
    const py::array& array;
    const void* data;
    std::array<py::ssize_t,3> shape,strides;
    py::object callback;
    GaugePin(const PeriodicCorrelationAdmittedReference& ref,const py::array& a,py::object cb)
        : array(a),data(a.data()),shape{},strides{},callback(std::move(cb)) {
        const auto& s=ref.state();
        if (!a.dtype().is(py::dtype::of<Complex>()) || !(a.flags()&py::array::c_style)
            || a.ndim()!=3 || a.shape(0)!=static_cast<py::ssize_t>(s.n_kpoints())
            || a.shape(1)!=static_cast<py::ssize_t>(s.n_correlated_occupied()) || a.shape(2)!=a.shape(1)
            || reinterpret_cast<std::uintptr_t>(data)%alignof(Complex))
            throw py::type_error("Gaussian Gram domain pair MP2 gauges require exact C-contiguous complex128 [K,active,active]");
        for(std::size_t i=0;i<3;++i) { shape[i]=a.shape(i);strides[i]=a.strides(i); }
    }
    void verify() const {
        if (array.data()!=data || !array.dtype().is(py::dtype::of<Complex>())
            || array.ndim()!=3 || !(array.flags()&py::array::c_style))
            throw std::invalid_argument("Gaussian Gram domain pair MP2 gauge descriptors changed across progress");
        for(std::size_t i=0;i<3;++i) if(array.shape(i)!=shape[i] || array.strides(i)!=strides[i])
            throw std::invalid_argument("Gaussian Gram domain pair MP2 gauge descriptors changed across progress");
    }
    static void notify(const Progress& p,void* context) {
        auto& pin=*static_cast<GaugePin*>(context);
        pin.callback(py::cast(Progress(p)));
        // Detect resize(refcheck=False), dtype and stride changes BEFORE
        // native source validation can inspect the previously pinned pointer.
        pin.verify();
    }
};
} // namespace gaussian_gram_domain_pair_mp2_python

void bind_periodic_gaussian_domain_pair_mp2(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianDomainPairMP2Options;
    using Caps=PeriodicGaussianDomainPairMP2Caps;
    using Live=PeriodicGaussianPairMP2LiveInventory;
    using Builder=PeriodicGaussianPairDomainBuilder;
    using Progress=PeriodicGaussianPairMP2Progress;
    py::class_<Options>(m,"_PeriodicGaussianDomainPairMP2Options").def(py::init<>())
        .def_readwrite("domain",&Options::domain).def_readwrite("pnos",&Options::pnos)
        .def_readwrite("projection",&Options::projection).def_readwrite("solver",&Options::solver)
        .def_readwrite("maximum_occupied_virtual_fock_norm",&Options::maximum_occupied_virtual_fock_norm);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianDomainPairMP2Caps").def(py::init<>());
#define GDPM_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GDPM_CAP(domain); GDPM_CAP(pnos); GDPM_CAP(projection); GDPM_CAP(solver);
    GDPM_CAP(maximum_owned_numerical_bytes); GDPM_CAP(maximum_control_storage_bytes);
    GDPM_CAP(maximum_per_worker_inventoried_bytes); GDPM_CAP(maximum_node_inventoried_bytes);
    GDPM_CAP(maximum_pair_count); GDPM_CAP(maximum_integral_calls); GDPM_CAP(maximum_progress_callbacks);
    GDPM_CAP(maximum_work_units);
#undef GDPM_CAP
    const auto tiny=[](const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianRealLocalProvider& provider,const Builder& builder,const Options& o,const Live& l,const Caps& c) {
        if (!ref.state_handle() || ref.state().n_kpoints()>8 || ref.state().n_basis()>12
            || basis.memory().occupied_count>4 || basis.memory().virtual_count>4 || o.solver.maximum_iterations>256
            || o.pnos.pno.pno_eigensolver.max_sweeps>200 || o.pnos.pno.semicanonical_eigensolver.max_sweeps>200)
            throw std::length_error("Gaussian domain pair MP2 exceeds tiny diagnostic dimensions/iterations");
        const auto p=plan_periodic_gaussian_domain_pair_mp2(ref,basis,provider,builder,o,l,c);
        if (p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units_upper_bound>1000000000000000ULL || p.progress_callback_upper_bound>1024)
            throw std::length_error("Gaussian domain pair MP2 exceeds tiny diagnostic memory/work/progress bounds");
        return p;
    };
    m.def("_plan_periodic_gaussian_domain_pair_mp2",tiny,py::arg("reference"),py::arg("basis"),py::arg("provider"),
        py::arg("builder"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_run_periodic_gaussian_domain_pair_mp2",[tiny](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
        const Builder& builder,Options o,Live l,Caps c,py::object callback) {
        (void)tiny(ref,basis,provider,builder,o,l,c);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian domain pair MP2 progress must be callable");
        const auto progress=[](const Progress& p,void* context) { (*static_cast<py::object*>(context))(py::cast(Progress(p))); };
        // GIL held; no borrowed NumPy input and no caller numerical overrides.
        return run_periodic_gaussian_domain_pair_mp2(ref,basis,provider,builder,o,l,c,
            callback.is_none()?nullptr:+progress,callback.is_none()?nullptr:&callback);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("builder"),py::arg("options"),
      py::arg("live"),py::arg("caps"),py::arg("progress")=py::none());

    using GramConfig=PeriodicGaussianGramDomainPairMP2Config;
    using GramOptions=PeriodicGaussianGramDomainPairMP2Options;
    using GramCaps=PeriodicGaussianGramDomainPairMP2Caps;
    namespace gram=gaussian_gram_domain_pair_mp2_python;
    py::class_<GramConfig>(m,"_PeriodicGaussianGramDomainPairMP2Config").def(py::init<>())
        .def_readwrite("pnos",&GramConfig::pnos);
    py::class_<GramOptions>(m,"_PeriodicGaussianGramDomainPairMP2Options").def(py::init<>())
        .def_readwrite("domain",&GramOptions::domain).def_readwrite("pnos",&GramOptions::pnos)
        .def_readwrite("projection",&GramOptions::projection).def_readwrite("solver",&GramOptions::solver)
        .def_readwrite("maximum_occupied_virtual_fock_norm",&GramOptions::maximum_occupied_virtual_fock_norm);
    auto gram_caps=py::class_<GramCaps>(m,"_PeriodicGaussianGramDomainPairMP2Caps").def(py::init<>());
#define GGDPM_CAP(f) gram_caps.def_readwrite(#f,&GramCaps::f)
    GGDPM_CAP(domain);GGDPM_CAP(pnos);GGDPM_CAP(projection);GGDPM_CAP(solver);
    GGDPM_CAP(maximum_pno_leaf_control_storage_bytes);GGDPM_CAP(maximum_owned_numerical_bytes);
    GGDPM_CAP(maximum_control_storage_bytes);GGDPM_CAP(maximum_per_worker_inventoried_bytes);
    GGDPM_CAP(maximum_node_inventoried_bytes);GGDPM_CAP(maximum_pair_count);GGDPM_CAP(maximum_gram_builds);
    GGDPM_CAP(maximum_factor_panels);GGDPM_CAP(maximum_tile_calls);GGDPM_CAP(maximum_progress_callbacks);
    GGDPM_CAP(maximum_work_units);
#undef GGDPM_CAP
    m.def("_plan_periodic_gaussian_gram_domain_pair_mp2_diagnostic",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationRealLocalBasis& basis,const Builder& builder,
        GramConfig config,GramOptions options,Live live,GramCaps caps) {
        return gram::tiny(hf,reference,wannier,basis,builder,config,options,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("basis"),py::arg("builder"),
        py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_run_periodic_gaussian_gram_domain_pair_mp2_diagnostic",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,const py::array& gauges,
        const PeriodicCorrelationRealLocalBasis& basis,const Builder& builder,
        GramConfig config,GramOptions options,Live live,GramCaps caps,py::object callback) {
        (void)gram::tiny(hf,reference,wannier,basis,builder,config,options,live,caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian Gram domain pair MP2 progress must be callable");
        gram::GaugePin pin(reference,gauges,std::move(callback));
        // GIL held and exact input owners pinned. Mutable scalar controls
        // are value copies; no AO/gauge or integral/amplitude array is copied.
        return run_periodic_gaussian_domain_pair_mp2(hf,reference,ao,auxiliary,wannier,
            static_cast<const gram::Complex*>(pin.data),static_cast<std::size_t>(gauges.size()),basis,builder,
            config,options,live,caps,pin.callback.is_none()?nullptr:&gram::GaugePin::notify,
            pin.callback.is_none()?nullptr:&pin);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),
        py::arg("gauges").noconvert(),py::arg("basis"),py::arg("builder"),py::arg("config"),
        py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("progress")=py::none());
}
