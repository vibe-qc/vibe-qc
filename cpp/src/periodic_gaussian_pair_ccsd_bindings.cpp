// Tiny actual-source diagnostics. No caller Fock/integral/amplitude arrays.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_pair_ccsd.hpp"
#include "periodic_correlation_real_local_internal.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

// This explicitly tiny reference producer transforms scalar integrals from
// the existing actual selected-common Gaussian provider. It is not the
// production direct all-q mixed-panel source, and cannot qualify a generic
// split result as physically certified. No Python numerical callback, dense
// common T2, four-index ERI tensor, or retained transformed-integral cache.
namespace gaussian_pair_ccsd_split_diagnostic {
using namespace vibeqc;
namespace real = periodic_correlation_real_local_detail;
using U = std::uint64_t;
using Producer = BoundedRestrictedPairCCSDParticleHoleProducer;
using Reader = BoundedRestrictedPairCCSDAmplitudes;
using PHPlan = BoundedRestrictedPairCCSDParticleHoleMemoryPlan;
using PHCaps = BoundedRestrictedPairCCSDParticleHoleCaps;
using PHInventory = BoundedRestrictedPairCCSDParticleHoleInventory;
using real::add;
using real::mul;
struct Sum {
    double value = 0.0, correction = 0.0;
    void include(double term) {
        real::finite(term);
        const double next = real::finite(value + term);
        const double error = std::abs(value) >= std::abs(term)
            ? real::finite(real::finite(value-next)+term) : real::finite(real::finite(term-next)+value);
        correction = real::finite(correction+error); value = next;
    }
    double total() const { return real::finite(value+correction); }
};
struct Context {
    const PeriodicCorrelationRealLocalBasis* basis = nullptr;
    const PeriodicGaussianRealLocalProvider* provider = nullptr;
    const PeriodicGaussianPairMP2Result* warm = nullptr;
    PHInventory inventory;
    PHCaps caps;
    U o = 0, n = 0, maximum_integral_calls = 0;
    U declared_extra_bytes = 0;
    unsigned fault = 0;
    static void produce(const Reader& reader, const BoundedRestrictedPairCCSDSolverMemoryPlan&, U i, U j, U snapshot,
        BoundedRestrictedPairCCSDParticleHoleVisitor visit, void* visitor_context, const void* raw) {
        const auto& c = *static_cast<const Context*>(raw);
        if (c.fault == 1) throw std::runtime_error("injected native Gaussian scalar PH producer failure");
        if (c.fault == 2) return;
        real::float_environment();
        const auto integrals = c.provider->provider().integral_provider(*c.basis);
        if (!snapshot || reader.memory().n_occupied != c.o || reader.memory().common_virtual_dimension != c.n)
            throw std::invalid_argument("Gaussian scalar PH diagnostic reader dimensions or snapshot differ");
        const auto frame = [&](U l, U m) {
            if (l > m) std::swap(l,m);
            const auto v = reader.canonical_pair_view(l,m);
            const auto& original = c.warm->pair(l,m).coefficients();
            if (v.rank != c.warm->pair(l,m).memory().retained_dimension
                || v.coefficients.data != original.data() || v.coefficients.element_count != original.size())
                throw std::invalid_argument("Gaussian scalar PH diagnostic requires exact warm-start frame pointers");
            return v;
        };
        const auto target = frame(i,j);
        const U a = target.rank, n = c.n, o = c.o;
        const auto current = plan_bounded_restricted_pair_ccsd_particle_hole(a,o,n,c.inventory);
        if (current.peak_owned_numerical_bytes > c.caps.maximum_owned_numerical_bytes)
            throw std::length_error("Gaussian scalar PH diagnostic accumulator exceeds admitted phase");
        BoundedRestrictedPairCCSDParticleHoleAccumulator accumulator(a,o,i,j,n,c.inventory,c.caps);
        // A single slot allocation, reused for K/J/O at every source rank.
        std::vector<double> slot(static_cast<std::size_t>(mul(3,mul(a,n))),0.0);
        U integral_calls = 0;
        const auto eri = [&](U p,U q,U r,U s) {
            if (integral_calls >= c.maximum_integral_calls)
                throw std::length_error("Gaussian scalar PH diagnostic integral cap exceeded");
            ++integral_calls;
            return real::finite(integrals.value(p,q,r,s,integrals.context));
        };
        for (U leg=0;leg<2;++leg) {
            const U l=leg ? j : i, r=leg ? i : j;
            for (U m=0;m<o;++m) {
                const auto source = frame(l,m);
                const U b=source.rank, ab=mul(a,b);
                for (U x=0;x<b;++x) for (U y=0;y<a;++y) {
                    Sum k, exchange, overlap;
                    for (U u=0;u<n;++u) {
                        overlap.include(real::finite(source.coefficients.data[u*b+x]*target.coefficients.data[u*a+y]));
                        for (U v=0;v<n;++v) {
                            const double weight=real::finite(source.coefficients.data[u*b+x]*target.coefficients.data[v*a+y]);
                            k.include(real::finite(weight*eri(m,o+u,r,o+v)));
                            exchange.include(real::finite(weight*eri(m,r,o+u,o+v)));
                        }
                    }
                    slot[x*a+y]=k.total(); slot[ab+x*a+y]=exchange.total(); slot[2*ab+x*a+y]=overlap.total();
                }
                const auto view = [&](U offset) -> BoundedRestrictedCCSDRealView {
                    return {ab ? slot.data()+offset : nullptr,static_cast<std::size_t>(ab)};
                };
                accumulator.accumulate({leg,m,b,view(0),view(ab),view(2*ab),source.amplitudes,l>m});
            }
        }
        auto result=accumulator.finish();
        const auto receipt = c.fault == 4 ? std::string(64,'0') : reader.snapshot_identity_sha256();
        visit(result,receipt,visitor_context);
        if (c.fault == 3) visit(result,receipt,visitor_context);
    }
};
Producer make(Context& c,const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicGaussianRealLocalProvider& provider,const PeriodicGaussianPairMP2Result& warm,
    U extra_bytes,unsigned fault) {
    c.o=basis.memory().occupied_count; c.n=basis.memory().virtual_count;
    if (!c.o || !c.n || c.o>4 || c.n>4 || extra_bytes>(1U<<20) || fault>4)
        throw std::length_error("Gaussian scalar PH diagnostic exceeds tiny counts, inventory or fault range");
    c.basis=&basis; c.provider=&provider; c.warm=&warm; c.fault=fault; c.declared_extra_bytes=extra_bytes;
    c.inventory.numerical_replicas=1; c.inventory.backend_margin_bytes_per_replica=65536;
    const auto p=plan_bounded_restricted_pair_ccsd_particle_hole(c.n,c.o,c.n,c.inventory);
    auto& caps=c.caps;
    caps.maximum_target_dimension=caps.maximum_source_dimension=c.n; caps.maximum_occupied_count=c.o;
    caps.maximum_borrowed_numerical_bytes=p.maximum_borrowed_numerical_bytes;
    caps.maximum_owned_numerical_bytes=p.peak_owned_numerical_bytes;
    caps.maximum_control_storage_bytes=p.total_control_storage_bytes;
    caps.maximum_per_replica_inventoried_bytes=p.per_replica_inventoried_bytes;
    caps.maximum_node_inventoried_bytes=p.required_node_inventoried_bytes;
    caps.maximum_scalar_products=p.scalar_products_upper_bound; caps.maximum_work_units=p.work_units_upper_bound;
    const U n2=mul(c.n,c.n), slot_bytes=mul(24,n2);
    c.maximum_integral_calls=mul(4,mul(c.o,mul(n2,n2)));
    Producer out; out.produce=&Context::produce; out.context=&c;
    // Explicit extra is a diagnostic live-inventory reservation only, not an
    // invented owner payload or a source certificate. Default is exactly0.
    out.additional_retained_numerical_bytes=extra_bytes;
    out.maximum_transient_numerical_bytes_per_target=add(slot_bytes,p.peak_owned_numerical_bytes);
    out.additional_control_storage_bytes=add(p.total_control_storage_bytes,
        65536U+sizeof(Context)+2U*sizeof(PHPlan)+sizeof(Producer)+3U*65U);
    out.maximum_work_units_per_target=add(p.work_units_upper_bound,
        add(mul(c.maximum_integral_calls,add(provider.memory().scalar_work_units,256)),
        mul(4096,add(1024,add(mul(2,mul(c.o,mul(n2,c.n))),mul(c.o,n2))))));
    real::Digest identity("vibeqc.periodic.gaussian-pair-ccsd.scalar-PH-diagnostic");
    identity.string(basis.identity_sha256()); identity.string(provider.identity_sha256());
    identity.string(warm.pair_spaces_identity_sha256()); identity.u64(c.o); identity.u64(c.n);
    identity.string("selected-common-scalar-reference;no-all-q-producer-certificate");
    const auto digest=identity.finish(); std::copy(digest.begin(),digest.end(),out.identity_sha256.begin());
    return out;
}
} // namespace gaussian_pair_ccsd_split_diagnostic

void bind_periodic_gaussian_pair_ccsd(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = PeriodicGaussianPairCCSDOptions;
    using Live = PeriodicGaussianPairCCSDLiveInventory;
    using Caps = PeriodicGaussianPairCCSDCaps;
    using Plan = PeriodicGaussianPairCCSDPlan;
    using Diagnostics = PeriodicGaussianPairCCSDDiagnostics;
    using Stage = PeriodicGaussianPairCCSDStage;
    using Progress = PeriodicGaussianPairCCSDProgress;
    using Result = PeriodicGaussianPairCCSDResult;
    using EmbeddingOptions = PeriodicGaussianPairCCSDEmbeddingOptions;
    auto embedding = py::class_<EmbeddingOptions>(m, "_PeriodicGaussianPairCCSDEmbeddingOptions");
    embedding.def(py::init<>());
#define GPCC_EMBED(f) embedding.def_readwrite(#f, &EmbeddingOptions::f)
    GPCC_EMBED(maximum_diagonal_exchange_projection_norm); GPCC_EMBED(maximum_fock_symmetry_projection_norm);
    GPCC_EMBED(maximum_exported_gram_error); GPCC_EMBED(maximum_exported_fock_error); GPCC_EMBED(maximum_exported_subspace_error);
#undef GPCC_EMBED
    py::class_<Options>(m, "_PeriodicGaussianPairCCSDOptions").def(py::init<>())
        .def_readwrite("singles", &Options::singles).def_readwrite("singles_embedding", &Options::singles_embedding)
        .def_readwrite("solver", &Options::solver);
    py::class_<Live>(m, "_PeriodicGaussianPairCCSDLiveInventory").def(py::init<>())
        .def_readwrite("other_live_bytes_per_worker", &Live::other_live_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker", &Live::fixed_backend_margin_bytes_per_worker);
    auto caps = py::class_<Caps>(m, "_PeriodicGaussianPairCCSDCaps");
    caps.def(py::init<>());
#define GPCC_CAP(f) caps.def_readwrite(#f, &Caps::f)
    GPCC_CAP(singles); GPCC_CAP(embedded_singles); GPCC_CAP(solver); GPCC_CAP(maximum_owned_numerical_bytes);
    GPCC_CAP(maximum_per_worker_inventoried_bytes); GPCC_CAP(maximum_node_inventoried_bytes);
    GPCC_CAP(maximum_pair_count); GPCC_CAP(maximum_integral_calls);
    GPCC_CAP(maximum_progress_callbacks); GPCC_CAP(maximum_work_units);
#undef GPCC_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicGaussianPairCCSDPlan");
#define GPCC_PLAN(f) plan.def_readonly(#f, &Plan::f)
    GPCC_PLAN(occupied_count); GPCC_PLAN(common_virtual_dimension); GPCC_PLAN(pair_count);
    GPCC_PLAN(domain_generated); GPCC_PLAN(singles_generation_dimension_sum); GPCC_PLAN(borrowed_generation_embedding_bytes);
    GPCC_PLAN(split_bare_particle_hole); GPCC_PLAN(borrowed_particle_hole_additional_numerical_bytes);
    GPCC_PLAN(borrowed_basis_bytes); GPCC_PLAN(borrowed_provider_row_bytes);
    GPCC_PLAN(borrowed_mp2_numerical_bytes); GPCC_PLAN(borrowed_mp2_pair_ct_bytes);
    GPCC_PLAN(borrowed_mp2_control_bytes); GPCC_PLAN(singles_output_upper_bytes);
    GPCC_PLAN(zero_singles_upper_bytes); GPCC_PLAN(singles_generation_phase_upper_bytes);
    GPCC_PLAN(solver_phase_owned_upper_bytes); GPCC_PLAN(peak_owned_numerical_bytes);
    GPCC_PLAN(solver_rank_padding_upper_bytes); GPCC_PLAN(control_storage_reservation_bytes);
    GPCC_PLAN(replicas_per_node); GPCC_PLAN(reference_base_node_bytes);
    GPCC_PLAN(per_worker_inventoried_bytes); GPCC_PLAN(required_node_memory_bytes);
    GPCC_PLAN(integral_calls_upper_bound); GPCC_PLAN(progress_callback_upper_bound);
    GPCC_PLAN(driver_work_units); GPCC_PLAN(work_units_upper_bound);
#undef GPCC_PLAN
    plan.def_property_readonly("singles_upper", [](const Plan& p) { return p.singles_upper; })
        .def_property_readonly("embedded_singles_upper", [](const Plan& p) { return p.embedded_singles_upper; })
        .def_property_readonly("solver_upper", [](const Plan& p) { return p.solver_upper; });
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicGaussianPairCCSDDiagnostics");
#define GPCC_DIAG(f) diagnostics.def_readonly(#f, &Diagnostics::f)
    GPCC_DIAG(completed_singles); GPCC_DIAG(completed_integral_calls); GPCC_DIAG(completed_progress_callbacks);
    GPCC_DIAG(retained_singles_bytes); GPCC_DIAG(zero_initial_singles_bytes);
    GPCC_DIAG(retained_singles_generation_coefficient_bytes);
    GPCC_DIAG(minimum_singles_rank); GPCC_DIAG(maximum_singles_rank); GPCC_DIAG(zero_rank_singles);
    GPCC_DIAG(singles_generation_dimension_sum);
    GPCC_DIAG(complete_common_finite_torus_basis); GPCC_DIAG(all_singles_full_rank); GPCC_DIAG(all_pairs_full_rank);
    GPCC_DIAG(split_bare_particle_hole); GPCC_DIAG(completed_particle_hole_calls);
    GPCC_DIAG(completed_particle_hole_visits); GPCC_DIAG(charged_particle_hole_work_units);
#undef GPCC_DIAG
    py::enum_<Stage>(m, "_PeriodicGaussianPairCCSDStage")
        .value("BEGIN", Stage::Begin).value("SINGLES_COMPLETE", Stage::SinglesComplete)
        .value("SOLVER", Stage::Solver).value("FINISHED", Stage::Finished);
    py::class_<Progress>(m, "_PeriodicGaussianPairCCSDProgress")
        .def_readonly("stage", &Progress::stage).def_readonly("callback_count", &Progress::callback_count)
        .def_readonly("completed_singles", &Progress::completed_singles)
        .def_readonly("occupied_i", &Progress::occupied_i).def_readonly("singles_rank", &Progress::singles_rank)
        .def_property_readonly("solver", [](const Progress& p) { return p.solver; });
    py::class_<Result>(m, "_PeriodicGaussianPairCCSDResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics", [](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("solver", &Result::solver, py::return_value_policy::reference_internal)
        .def_property_readonly("converged", &Result::converged)
        .def_property_readonly("periodic_energy_per_cell", &Result::periodic_energy_per_cell)
        .def_property_readonly("correlation_energy_per_cell", &Result::correlation_energy_per_cell)
        .def_property_readonly("total_energy_per_cell", &Result::total_energy_per_cell)
        .def_property_readonly("matched_finite_gaussian_hf_recipe", &Result::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("split_bare_particle_hole", &Result::split_bare_particle_hole)
        .def_property_readonly("particle_hole_physical_source_certified", &Result::particle_hole_physical_source_certified)
        .def_property_readonly("physical_particle_hole_source_identity_sha256", &Result::physical_particle_hole_source_identity_sha256)
        .def_property_readonly("split_execution_identity_sha256", &Result::split_execution_identity_sha256)
        .def_property_readonly("consumed_particle_hole_identity_sha256", &Result::consumed_particle_hole_identity_sha256)
        .def_property_readonly("production_dlpno", &Result::production_dlpno)
        .def_property_readonly("includes_triples", &Result::includes_triples)
        .def_property_readonly("infinite_source_accuracy_certified", &Result::infinite_source_accuracy_certified)
        .def_property_readonly("identity_sha256", &Result::identity_sha256)
        .def_property_readonly("warmstart_identity_sha256", &Result::warmstart_identity_sha256)
        .def_property_readonly("pair_spaces_identity_sha256", &Result::pair_spaces_identity_sha256)
        .def_property_readonly("singles_spaces_identity_sha256", &Result::singles_spaces_identity_sha256)
        .def_property_readonly("basis_identity_sha256", &Result::basis_identity_sha256)
        .def_property_readonly("provider_identity_sha256", &Result::provider_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256", &Result::hf_reference_source_identity_sha256)
        // Never expose the nested native PNO object. Python const references
        // are not immutable against the separate consuming pair-space API.
        .def("singles_diagnostics", [](const Result& r, U i) { return r.singles_frame(i).generation_diagnostics(); })
        .def("singles_options", [](const Result& r, U i) { return r.singles_frame(i).options(); })
        .def("singles_identity_sha256", [](const Result& r, U i) { return r.singles_frame(i).identity_sha256(); })
        .def("singles_payload_sha256", [](const Result& r, U i) { return r.singles_frame(i).payload_sha256(); })
        .def("singles_domain_generated", [](const Result& r, U i) { return r.singles_frame(i).is_embedded(); })
        .def("singles_generation_dimension", [](const Result& r, U i) { return r.singles_frame(i).generation_dimension(); })
        .def("singles_retained_numerical_bytes", [](const Result& r, U i) { return r.singles_frame(i).retained_numerical_bytes(); })
        .def("singles_embedding_identity_sha256", [](const Result& r, U i) { return r.singles_frame(i).embedded().embedding_identity_sha256(); })
        .def("singles_embedded_diagnostics", [](const Result& r, U i) { return r.singles_frame(i).embedded().diagnostics(); })
        .def("singles_coefficients_copy", [](const Result& r, U i) {
            const auto p = r.singles_frame(i);
            const auto n = p.common_virtual_dimension(), rank = p.retained_dimension();
            if (n > 4 || rank > 4) throw std::length_error("Gaussian pair CCSD singles copy exceeds tiny dimension");
            py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(rank)});
            const auto& values = p.coefficients();
            if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
            return out;
        })
        .def("singles_energies_copy", [](const Result& r, U i) {
            const auto& values = r.singles_frame(i).energies();
            if (values.size() > 4) throw std::length_error("Gaussian pair CCSD energy copy exceeds tiny dimension");
            py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
            if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
            return out;
        })
        .def("singles_original_pno_occupations_copy", [](const Result& r, U i) {
            const auto& values = r.singles_frame(i).original_pno_occupations();
            if (values.size() > 4) throw std::length_error("Gaussian pair CCSD occupation copy exceeds tiny dimension");
            py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
            if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size()*sizeof(double));
            return out;
        });
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairMP2Result& warmstart, const Options& options, const Live& live, const Caps& caps) {
        if (ref.state().n_kpoints() > 8 || basis.memory().occupied_count > 4 || basis.memory().virtual_count > 4
            || options.singles.pno_eigensolver.max_sweeps > 200
            || options.singles.semicanonical_eigensolver.max_sweeps > 200 || options.solver.maximum_iterations > 128)
            throw std::length_error("Gaussian pair CCSD diagnostic exceeds tiny dimensions or iterations");
        auto p = plan_periodic_gaussian_pair_ccsd(ref, basis, provider, warmstart, options, live, caps);
        if (p.peak_owned_numerical_bytes > (16U << 20) || p.required_node_memory_bytes > (128U << 20)
            || p.work_units_upper_bound > 1000000000000000ULL || p.progress_callback_upper_bound > 1024)
            throw std::length_error("Gaussian pair CCSD diagnostic exceeds tiny memory, work or callbacks");
        return p;
    };
    m.def("_plan_periodic_gaussian_pair_ccsd", tiny,
        py::arg("reference"), py::arg("basis"), py::arg("provider"), py::arg("warmstart"),
        py::arg("options"), py::arg("live"), py::arg("caps"));
    m.def("_run_periodic_gaussian_pair_ccsd", [tiny](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairMP2Result& warmstart, Options options, Live live, Caps caps, py::object callback) {
        (void)tiny(ref, basis, provider, warmstart, options, live, caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian pair CCSD progress must be callable");
        const auto progress = [](const Progress& p, void* context) {
            (*static_cast<py::object*>(context))(py::cast(Progress(p)));
        };
        // Native immutable owners are pinned for the call. Mutable controls
        // above are copied before any callback; keep the GIL for this tiny
        // diagnostic, whose only Python callbacks are scalar progress.
        return run_periodic_gaussian_pair_ccsd(ref, basis, provider, warmstart, options, live, caps,
            callback.is_none() ? nullptr : +progress, callback.is_none() ? nullptr : &callback);
    }, py::arg("reference"), py::arg("basis"), py::arg("provider"), py::arg("warmstart"),
       py::arg("options"), py::arg("live"), py::arg("caps"), py::arg("progress") = py::none());

    // Separate, unambiguously diagnostic selected-common scalar producer.
    // Unlike arbitrary callback injection, this seam accepts no numerical
    // arrays or caller producer certification and pins all genuine owners.
    const auto split_tiny = [](const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairMP2Result& warm, const BoundedRestrictedPairCCSDParticleHoleProducer& producer,
        const Options& options, const Live& live, const Caps& caps) {
        if (ref.state().n_kpoints()>8 || basis.memory().occupied_count>4 || basis.memory().virtual_count>4
            || options.singles.pno_eigensolver.max_sweeps>200
            || options.singles.semicanonical_eigensolver.max_sweeps>200 || options.solver.maximum_iterations>128)
            throw std::length_error("Gaussian pair CCSD split diagnostic exceeds tiny dimensions or iterations");
        auto p=plan_periodic_gaussian_pair_ccsd(ref,basis,provider,warm,producer,options,live,caps);
        if (p.peak_owned_numerical_bytes>(16U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units_upper_bound>1000000000000000ULL || p.progress_callback_upper_bound>1024)
            throw std::length_error("Gaussian pair CCSD split diagnostic exceeds tiny memory, work or callbacks");
        return p;
    };
    m.def("_plan_periodic_gaussian_pair_ccsd_split_scalar_reference", [split_tiny](
        const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianRealLocalProvider& provider,const PeriodicGaussianPairMP2Result& warm,
        Options options,Live live,Caps caps,U extra) {
        gaussian_pair_ccsd_split_diagnostic::Context context;
        const auto producer=gaussian_pair_ccsd_split_diagnostic::make(context,basis,provider,warm,extra,0);
        return split_tiny(ref,basis,provider,warm,producer,options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("warmstart"),
        py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("declared_extra_retained_numerical_bytes")=0);
    m.def("_run_periodic_gaussian_pair_ccsd_split_scalar_reference", [split_tiny](
        const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianRealLocalProvider& provider,const PeriodicGaussianPairMP2Result& warm,
        Options options,Live live,Caps caps,py::object callback,U extra,unsigned fault) {
        gaussian_pair_ccsd_split_diagnostic::Context context;
        const auto producer=gaussian_pair_ccsd_split_diagnostic::make(context,basis,provider,warm,extra,fault);
        (void)split_tiny(ref,basis,provider,warm,producer,options,live,caps);
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian pair CCSD progress must be callable");
        const auto progress=[](const Progress& p,void* raw) {
            (*static_cast<py::object*>(raw))(py::cast(Progress(p)));
        };
        return run_periodic_gaussian_pair_ccsd(ref,basis,provider,warm,producer,options,live,caps,
            callback.is_none()?nullptr:+progress,callback.is_none()?nullptr:&callback);
    },py::arg("reference"),py::arg("basis"),py::arg("provider"),py::arg("warmstart"),
        py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("progress")=py::none(),
       py::arg("declared_extra_retained_numerical_bytes")=0,py::arg("producer_fault")=0);

    // Deliberately invalidates original F storage through the owner's real
    // native move assignment, without changing any certified numerical
    // content or receipt. This run-only negative seam never exposes a generic
    // owner mutator and accepts no user callback or numerical arrays.
    m.def("_run_periodic_gaussian_pair_ccsd_basis_replacement_diagnostic", [tiny,split_tiny](
        const PeriodicCorrelationAdmittedReference& ref,PeriodicCorrelationRealLocalBasis& basis,
        PeriodicCorrelationRealLocalBasis& replacement,const PeriodicGaussianRealLocalProvider& provider,
        const PeriodicGaussianPairMP2Result& warm,Options options,Live live,Caps caps,
        Stage replacement_stage,bool split) {
        namespace real=periodic_correlation_real_local_detail;
        if (&basis==&replacement || basis.state_handle()!=replacement.state_handle()
            || basis.identity_sha256()!=replacement.identity_sha256()
            || basis.local_basis_identity_sha256()!=replacement.local_basis_identity_sha256()
            || basis.memory().retained_output_bytes!=replacement.memory().retained_output_bytes
            || basis.f_oo_data()==replacement.f_oo_data()
            || basis.occupied_indices_data()==replacement.occupied_indices_data())
            throw std::invalid_argument("basis replacement diagnostic requires distinct equal-content native owners");
        if (replacement_stage!=Stage::Begin && replacement_stage!=Stage::SinglesComplete
            && replacement_stage!=Stage::Solver && replacement_stage!=Stage::Finished)
            throw std::invalid_argument("basis replacement diagnostic requires a known stage");
        struct Replacement {
            PeriodicCorrelationRealLocalBasis& basis;
            PeriodicCorrelationRealLocalBasis& replacement;
            Stage stage;
            bool consumed=false;
            static void notify(const Progress& p,void* context) {
                auto& self=*static_cast<Replacement*>(context);
                if (p.stage==self.stage && !self.consumed) {
                    self.basis=std::move(self.replacement);self.consumed=true;
                }
            }
        } replacement_event{basis,replacement,replacement_stage};
        // Count both original and replacement owner until the invalidating
        // event; retaining the upper afterwards is conservative.
        live.other_live_bytes_per_worker=real::add(live.other_live_bytes_per_worker,
            real::add(replacement.memory().retained_output_bytes,
                sizeof(PeriodicCorrelationRealLocalBasis)+sizeof(Replacement)+8U*65U));
        if (split) {
            gaussian_pair_ccsd_split_diagnostic::Context context;
            const auto producer=gaussian_pair_ccsd_split_diagnostic::make(context,basis,provider,warm,0,0);
            (void)split_tiny(ref,basis,provider,warm,producer,options,live,caps);
            return run_periodic_gaussian_pair_ccsd(ref,basis,provider,warm,producer,options,live,caps,
                &Replacement::notify,&replacement_event);
        }
        (void)tiny(ref,basis,provider,warm,options,live,caps);
        return run_periodic_gaussian_pair_ccsd(ref,basis,provider,warm,options,live,caps,
            &Replacement::notify,&replacement_event);
    },py::arg("reference"),py::arg("basis"),py::arg("replacement"),py::arg("provider"),
      py::arg("warmstart"),py::arg("options"),py::arg("live"),py::arg("caps"),
      py::arg("replacement_stage"),py::arg("split")=false);
}
