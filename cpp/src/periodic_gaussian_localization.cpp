#include "vibeqc/periodic_gaussian_localization.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"

namespace vibeqc {
namespace {
using I = std::uint64_t;
using SourcePlan = PeriodicGaussianBlochIAOMemoryPlan;
using SourceCaps = PeriodicGaussianBlochIAOCaps;
using Memory = PeriodicGaussianLocalizationMemoryPlan;
using Diagnostics = PeriodicGaussianLocalizationDiagnostics;
using Progress = PeriodicGaussianLocalizationProgress;
using Stage = PeriodicGaussianLocalizationStage;
constexpr I fixed_control_reservation = 65536;
static_assert(sizeof(void*) == 8 && sizeof(int) == 4 && sizeof(double) == 8
              && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
              "Gaussian localization inventory requires 64-bit pointers/binary64 and int32");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian localization forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Gaussian localization requires binary64 evaluation"
#endif
I add(I a, I b) {
    if (b > std::numeric_limits<I>::max() - a) throw std::overflow_error("Gaussian localization count sum overflow");
    return a + b;
}
I mul(I a, I b) {
    if (a && b > std::numeric_limits<I>::max() / a) throw std::overflow_error("Gaussian localization count product overflow");
    return a * b;
}
void limit(I value, I cap, const char* message) { if (value > cap) throw std::length_error(message); }
void float_environment() {
    volatile double tiny=std::numeric_limits<double>::denorm_min(), one=1.0, zero=0.0;
    if (std::fegetround()!=FE_TONEAREST || !(tiny>0) || std::fma(tiny,one,zero)!=tiny)
        throw std::invalid_argument("Gaussian localization requires round-to-nearest and gradual underflow");
}
class Work {
public:
    explicit Work(I cap) : cap_(cap) {}
    void require(I value) const { limit(add(used_, value), cap_, "Gaussian localization total work cap exceeded"); }
    void charge(I value) { require(value); used_ += value; }
    I used() const { return used_; }
    I remaining() const { return cap_ - used_; }
private: I cap_, used_ = 0;
};
class Digest {
public:
    explicit Digest(const char* domain) { text(domain); u64(kPeriodicGaussianLocalizationVersion); }
    void u64(I x) { std::array<std::uint8_t, 8> b{}; for (unsigned j = 0; j < 8; ++j) b[j] = x >> (56 - 8*j); h_.update(b.data(), b.size()); }
    void real(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("Gaussian localization input is not finite");
        if (x == 0) x = 0;
        I bits; std::memcpy(&bits, &x, 8); u64(bits);
    }
    void text(const std::string& x) { u64(x.size()); h_.update(reinterpret_cast<const std::uint8_t*>(x.data()), x.size()); }
    std::string finish() { return h_.finish_hex(); }
private: detail::Sha256 h_;
};
I baseline(const PeriodicCorrelationAdmittedReference& ref) {
    const auto& d = ref.dimensions();
    return add(add(d.external_bytes, d.shared_bytes),
               mul(ref.budget().mpi_ranks, add(d.per_rank_bytes, d.localization_window_bytes_per_rank)));
}
I node(const PeriodicCorrelationAdmittedReference& ref, const Memory& m, I own) {
    const I replica = add(add(m.borrowed_gaussian_numerical_bytes, m.other_live_numerical_bytes),
                          add(own, m.peak_control_storage_bytes));
    const I value = add(baseline(ref), mul(mul(ref.budget().mpi_ranks, ref.budget().workers_per_rank), replica));
    limit(value, ref.budget().memory_limit_bytes, "Gaussian localization enclosing node memory cap exceeded");
    return value;
}
// A conservative source-preflight reservation made BEFORE calling its actual
// census/image walker. Initial unknown metadata use explicit caller caps;
// after the first census these caps are clamped to admitted actual totals.
I source_plan_work(I b, I r, const SourceCaps& c) {
    const I panel = std::min(c.maximum_panel_pairs, mul(b, b));
    if (!panel || !r) throw std::invalid_argument("Gaussian localization requires nonempty explicit bases/panels");
    const I bb = mul(b,b), br = mul(b,r), rr = mul(r,r);
    const auto panels = [panel](I n) { return add(n / panel, n % panel != 0); };
    const I calls = add(add(mul(2, panels(bb)), mul(3, panels(br))), mul(2, panels(rr)));
    const I pairs = add(add(mul(2,bb), mul(3,br)), mul(2,rr));
    const I sc = add(c.maximum_shell_count, c.maximum_contraction_count);
    const I scan = add(add(mul(128, c.maximum_basis_content_wire_bytes), mul(64,sc)),
                       add(4096, mul(512,c.maximum_atom_count)));
    return add(scan, mul(512, add(add(mul(calls, add(c.maximum_basis_content_wire_bytes, add(sc,256))),
                                     mul(pairs, add(sc,32))), c.maximum_total_image_candidates)));
}
SourceCaps clamp_source(const SourceCaps& c, const SourcePlan& p, I atoms) {
    SourceCaps out = c;
    out.maximum_atom_count = atoms;
    out.maximum_shell_count = add(p.ao.shell_count, p.minimal.shell_count);
    out.maximum_contraction_count = add(p.ao.contraction_count, p.minimal.contraction_count);
    out.maximum_primitive_numeric_lanes = add(add(p.ao.exponent_count, p.ao.coefficient_count),
                                              add(p.minimal.exponent_count, p.minimal.coefficient_count));
    out.maximum_basis_content_wire_bytes = add(p.ao.content_wire_bytes, p.minimal.content_wire_bytes);
    out.maximum_borrowed_active_numeric_bytes = add(p.borrowed_basis_numeric_bytes, p.borrowed_geometry_numeric_bytes);
    return out;
}
// Bounded metadata re-census BEFORE the content hash. Changed/oversized
// objects may not use generous original caps to escape the outer work model.
void same_inventory(const BasisSet& basis, const PeriodicGaussianBasisInventory& p) {
    if (basis.nbasis() != p.function_count || basis.nshells() != p.shell_count)
        throw std::invalid_argument("Gaussian localization basis dimensions changed during the call");
    I contractions = 0, exponents = 0, coefficients = 0;
    for (const auto& shell : basis.libint()) {
        contractions = add(contractions, shell.contr.size());
        exponents = add(exponents, shell.alpha.size());
        limit(contractions, p.contraction_count, "Gaussian localization admitted contraction census changed");
        limit(exponents, p.exponent_count, "Gaussian localization admitted exponent census changed");
        for (const auto& contraction : shell.contr) {
            coefficients = add(coefficients, contraction.coeff.size());
            limit(coefficients, p.coefficient_count, "Gaussian localization admitted coefficient census changed");
        }
    }
    if (contractions != p.contraction_count || exponents != p.exponent_count || coefficients != p.coefficient_count)
        throw std::invalid_argument("Gaussian localization admitted basis census changed");
}
std::string input_digest(const PeriodicCorrelationAdmittedReference& ref,
                         const BasisSet& ao, const BasisSet& minimal, const PeriodicSystem& system,
                         const SourcePlan& p, I atom_count, double cutoff) {
    if (system.unit_cell.size() != atom_count || system.dim != 3)
        throw std::invalid_argument("Gaussian localization original cell/atom count changed during the call");
    same_inventory(ao,p.ao); same_inventory(minimal,p.minimal);
    Digest digest("vibeqc.periodic.gaussian-localization.inputs");
    digest.text(auxiliary_basis_content_identity_sha256(ao));
    digest.text(auxiliary_basis_content_identity_sha256(minimal));
    for (int n : ref.state().mesh()) digest.u64(static_cast<I>(n));
    digest.real(cutoff); digest.u64(atom_count);
    for (unsigned row=0; row<3; ++row) for (unsigned col=0; col<3; ++col) digest.real(system.lattice(row,col));
    for (const auto& atom : system.unit_cell) {
        if (atom.Z <= 0 || atom.Z > 118) throw std::invalid_argument("Gaussian localization invalid atom label");
        digest.u64(static_cast<I>(atom.Z)); for (double x : atom.xyz) digest.real(x);
    }
    for (const auto* basis : {&ao,&minimal}) {
        digest.u64(basis->nshells());
        for (std::size_t s=0; s<basis->nshells(); ++s) {
            const int atom=basis->shell_atom_index(s);
            if (atom < 0 || static_cast<I>(atom) >= atom_count)
                throw std::invalid_argument("Gaussian localization invalid shell atom label");
            digest.u64(static_cast<I>(atom));
        }
    }
    return digest.finish();
}
I wannier_work(const PeriodicRestrictedMeanFieldState& s) {
    const I k=s.n_kpoints(), b=s.n_basis(), o=s.n_correlated_occupied(), e=s.n_effective_orbitals();
    I sum=add(mul(k,mul(o,mul(o,o))), mul(k,mul(b,b)));
    sum=add(sum,mul(k,mul(b,mul(o,e)))); sum=add(sum,mul(k,mul(k,mul(b,o))));
    sum=add(sum,add(mul(k,e),add(mul(k,mul(b,o)),mul(k,mul(o,o)))));
    return mul(512,add(sum,1));
}
void aggregate(PeriodicGaussianBlochIAODiagnostics& a, const PeriodicGaussianBlochIAODiagnostics& b) {
    a.charged_work_units=add(a.charged_work_units,b.charged_work_units);
    a.evaluated_panel_count=add(a.evaluated_panel_count,b.evaluated_panel_count);
    a.evaluated_image_candidate_count=add(a.evaluated_image_candidate_count,b.evaluated_image_candidate_count);
    a.evaluated_pair_image_count=add(a.evaluated_pair_image_count,b.evaluated_pair_image_count);
#define GLS_MAX(field) a.field=std::max(a.field,b.field)
    GLS_MAX(maximum_geometry_residual); GLS_MAX(maximum_s11_reference_residual);
    GLS_MAX(maximum_s11_reference_conjugacy_residual); GLS_MAX(maximum_cross_adjoint_residual);
    GLS_MAX(maximum_overlap_time_reversal_residual); GLS_MAX(maximum_minimal_hermitian_defect);
    GLS_MAX(maximum_trim_imaginary_magnitude); GLS_MAX(maximum_projection_correction);
#undef GLS_MAX
}
struct CallbackContext {
    const PeriodicCorrelationAdmittedReference& reference;
    const BasisSet& ao; const BasisSet& minimal; const PeriodicSystem& system;
    const SourcePlan& source_plan; I atoms; double cutoff;
    const std::string& identity; Memory& memory; Diagnostics& diagnostics; Work& work;
    PeriodicGaussianLocalizationCallback callback; void* context;
    void emit(Stage stage, const PeriodicCorrelationIAOOptimizerProgress* optimizer=nullptr) {
        if (!callback) return;
        Progress event; event.stage=stage; event.completed_points=diagnostics.constructed_iao_points;
        event.point_count=memory.n_points; event.charged_work_units=work.used();
        if (optimizer) { event.optimizer=*optimizer; event.charged_work_units=add(event.charged_work_units,optimizer->charged_work_units); }
        callback(event,context);
        if (input_digest(reference,ao,minimal,system,source_plan,atoms,cutoff) != identity)
            throw std::invalid_argument("Gaussian localization callback changed original Gaussian inputs");
        diagnostics.callback_input_checks=add(diagnostics.callback_input_checks,1);
    }
    static void relay(const PeriodicCorrelationIAOOptimizerProgress& event, void* context) {
        static_cast<CallbackContext*>(context)->emit(Stage::Optimization,&event);
    }
};
} // namespace

const PeriodicCorrelationIAOOptimizerResult& PeriodicGaussianLocalizationResult::optimizer() const {
    if (!optimizer_) throw std::logic_error("Gaussian localization result has no live optimizer");
    (void)optimizer_->state(); return *optimizer_;
}
bool PeriodicGaussianLocalizationResult::converged() const { return optimizer().converged(); }
const PeriodicCorrelationWannier& PeriodicGaussianLocalizationResult::wannier() const {
    if (!optimizer().converged() || !wannier_ || !wannier_->state_handle())
        throw std::logic_error("Gaussian localization has no converged live Wannier output");
    return *wannier_;
}
void PeriodicGaussianLocalizationResult::verify_original_inputs(
    const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao,
    const BasisSet& minimal, const PeriodicSystem& system) const {
    float_environment();
    const auto& opt=optimizer();
    if (!reference.state_handle() || opt.state_handle().get()!=reference.state_handle().get()
        || opt.allocation_identity()!=reference.dimensions().allocation_identity)
        throw std::invalid_argument("Gaussian localization original-input check requires the exact admitted state/allocation");
    if (opt.converged()) {
        const auto& w=wannier();
        if (w.state_handle().get()!=reference.state_handle().get()
            || w.allocation_identity()!=reference.dimensions().allocation_identity
            || w.localization_identity_sha256()!=opt.optimizer_identity_sha256())
            throw std::invalid_argument("Gaussian localization original-input check found inconsistent output owners");
    }
    if (input_digest(reference,ao,minimal,system,source_plan_,atom_count_,image_cutoff_bohr_)!=input_identity_)
        throw std::invalid_argument("Gaussian localization original Gaussian inputs changed");
}

PeriodicGaussianLocalizationResult localize_periodic_gaussian_occupied(
    const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao, const BasisSet& minimal,
    const PeriodicSystem& system, I other, const PeriodicGaussianLocalizationOptions& options,
    const SourceCaps& source_caps, const PeriodicGaussianLocalizationCaps& caps,
    PeriodicGaussianLocalizationCallback callback, void* callback_context) {
    if (!reference.state_handle()) throw std::invalid_argument("Gaussian localization requires an immutable admitted reference");
    // Same environment as all constituent numerical leaves; in particular
    // input receipts and callback equality checks may not silently flush a
    // physically supplied subnormal lane or run under directed rounding.
    float_environment();
    for (I cap : {caps.maximum_owned_numerical_bytes,caps.maximum_control_storage_bytes,caps.maximum_point_owners,
            caps.maximum_work_units,caps.maximum_total_image_candidates,caps.maximum_optimizer_work_units})
        if (!cap) throw std::invalid_argument("Gaussian localization caps must all be positive");
    if (!options.wannier.require_time_reversal || !options.wannier.require_real_home_coefficients)
        throw std::invalid_argument("Gaussian localization requires physical TR and real-home Wannier audits");
    const auto& state=reference.state(); const I k=state.n_kpoints(), b=state.n_basis();
    limit(k,caps.maximum_point_owners,"Gaussian localization point-owner cap exceeded");
    PeriodicGaussianLocalizationResult result; auto& m=result.memory_; auto& d=result.diagnostics_;
    m.n_points=k; m.n_basis=b; m.n_minimal=minimal.nbasis(); m.n_active=state.n_correlated_occupied();
    m.other_live_numerical_bytes=other;
    m.point_plan_record_bytes=mul(k,sizeof(SourcePlan));
    m.point_owner_record_bytes=mul(k,sizeof(PeriodicGaussianBlochIAOPoint));
    // Eleven 64-character identities per source wrapper including its IAO;
    // allocation identity is itself contractually a lowercase SHA-256.
    m.source_identity_character_bytes=mul(k,11*64);
    m.fixed_control_reservation_bytes=fixed_control_reservation;
    m.peak_control_storage_bytes=add(fixed_control_reservation,
        add(m.point_plan_record_bytes,add(m.point_owner_record_bytes,m.source_identity_character_bytes)));
    limit(m.peak_control_storage_bytes,caps.maximum_control_storage_bytes,"Gaussian localization control storage cap exceeded");
    limit(m.peak_control_storage_bytes,static_cast<I>(std::numeric_limits<std::ptrdiff_t>::max()),
          "Gaussian localization metadata address extent exceeded");
    Work work(caps.maximum_work_units);
    SourceCaps initial_caps=source_caps;
    initial_caps.maximum_total_image_candidates=std::min(initial_caps.maximum_total_image_candidates,caps.maximum_total_image_candidates);
    const I initial_plan_work=source_plan_work(b,m.n_minimal,initial_caps);
    work.charge(initial_plan_work);
    const SourcePlan first=plan_periodic_gaussian_bloch_iao(reference,ao,minimal,system,0,other,
        options.source,options.iao,initial_caps);
    m.source_planning_work_units=initial_plan_work;
    m.borrowed_gaussian_numerical_bytes=add(first.borrowed_basis_numeric_bytes,first.borrowed_geometry_numeric_bytes);
    const I atoms=system.unit_cell.size(); SourceCaps tight=clamp_source(source_caps,first,atoms);
    result.source_plan_=first; result.atom_count_=atoms;
    result.image_cutoff_bohr_=options.source.image_cutoff_bohr;
    m.input_verification_work_units=add(mul(2,first.basis_scan_work_units),4096);
    work.charge(m.input_verification_work_units);
    result.input_identity_=input_digest(reference,ao,minimal,system,first,atoms,options.source.image_cutoff_bohr);
    const auto seed_plan=plan_periodic_correlation_diabatic_seed(state.mesh(),b,m.n_active,
        state.n_effective_orbitals(),options.seed.jacobi_max_sweeps);
    m.seed_phase_owned_bytes=seed_plan.peak_owned_numerical_bytes;
    limit(m.seed_phase_owned_bytes,caps.maximum_owned_numerical_bytes,"Gaussian localization owned seed phase cap exceeded");
    m.seed_required_node_bytes=node(reference,m,m.seed_phase_owned_bytes);
    work.charge(seed_plan.maximum_work_units);
    std::optional<PeriodicCorrelationDiabaticSeed> seed;
    seed.emplace(make_periodic_correlation_diabatic_seed(reference,caps.maximum_owned_numerical_bytes,options.seed));
    const auto opt_plan=plan_periodic_correlation_iao_optimizer(reference,*seed,m.n_minimal,options.optimizer);
    const auto w_plan=plan_periodic_correlation_wannier(state.mesh(),b,m.n_active);
    const I seed_live=seed_plan.output_numerical_bytes, iao_live=mul(k,first.output_numerical_bytes);
    const I pointer_bytes=mul(k,sizeof(const PeriodicCorrelationBlochIAO*));
    m.source_phase_owned_bytes=add(seed_live,add(mul(k-1,first.output_numerical_bytes),first.peak_owned_numerical_bytes));
    m.optimizer_phase_owned_bytes=add(seed_live,add(iao_live,add(pointer_bytes,opt_plan.peak_owned_numerical_bytes)));
    m.wannier_phase_owned_bytes=add(opt_plan.output_numerical_bytes,w_plan.peak_owned_numerical_bytes);
    m.peak_owned_numerical_bytes=std::max({m.seed_phase_owned_bytes,m.source_phase_owned_bytes,
        m.optimizer_phase_owned_bytes,m.wannier_phase_owned_bytes});
    limit(m.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian localization enclosing owned numerical cap exceeded");
    m.source_required_node_bytes=node(reference,m,m.source_phase_owned_bytes);
    m.optimizer_required_node_bytes=node(reference,m,m.optimizer_phase_owned_bytes);
    m.wannier_required_node_bytes=node(reference,m,m.wannier_phase_owned_bytes);
    m.required_node_memory_bytes=std::max({m.seed_required_node_bytes,m.source_required_node_bytes,
        m.optimizer_required_node_bytes,m.wannier_required_node_bytes});
    m.wannier_work_reservation=wannier_work(state);
    if (callback) {
        const I maximum_events=add(add(k,6),mul(options.optimizer.maximum_iterations,options.optimizer.maximum_line_search_trials));
        m.callback_work_reservation=mul(maximum_events,m.input_verification_work_units);
        work.charge(m.callback_work_reservation);
    }
    d.image_candidate_visits=first.image_candidate_count;
    m.planned_total_image_visits=first.image_candidate_count;
    std::vector<SourcePlan> plans; plans.reserve(static_cast<std::size_t>(k));
    for (I point=0; point<k; ++point) {
        SourceCaps planning=tight;
        const I remaining=caps.maximum_total_image_candidates-m.planned_total_image_visits;
        // A completed point requires this plan + factory preflight + two
        // value walks. Reserve all four before permitting this image walk.
        planning.maximum_total_image_candidates=std::min(planning.maximum_total_image_candidates,remaining/4);
        if (!planning.maximum_total_image_candidates) throw std::length_error("Gaussian localization total image visit cap exceeded");
        const I reservation=source_plan_work(b,m.n_minimal,planning);
        work.charge(reservation); m.source_planning_work_units=add(m.source_planning_work_units,reservation);
        auto plan=plan_periodic_gaussian_bloch_iao(reference,ao,minimal,system,point,add(other,seed_live),
            options.source,options.iao,planning);
        m.planned_source_image_candidates=add(m.planned_source_image_candidates,plan.image_candidate_count);
        m.planned_total_image_visits=add(m.planned_total_image_visits,mul(4,plan.image_candidate_count));
        d.image_candidate_visits=add(d.image_candidate_visits,plan.image_candidate_count);
        m.source_factory_work_reservation=add(m.source_factory_work_reservation,plan.maximum_work_units);
        plans.push_back(std::move(plan));
    }
    // All remaining numerical owners and the minimum valid optimizer state
    // are admitted before constructing the K source wrappers.
    work.require(add(m.source_factory_work_reservation,add(opt_plan.initial_work_units,m.wannier_work_reservation)));
    CallbackContext context{reference,ao,minimal,system,first,atoms,options.source.image_cutoff_bohr,
        result.input_identity_,m,d,work,callback,callback_context};
    context.emit(Stage::SeedReady);
    Digest sources("vibeqc.periodic.gaussian-localization.ordered-sources");
    Digest matches("vibeqc.periodic.gaussian-localization.ordered-reference-matches");
    sources.u64(k); matches.u64(k);
    {
        std::vector<PeriodicGaussianBlochIAOPoint> points; points.reserve(static_cast<std::size_t>(k));
        for (I point=0; point<k; ++point) {
            SourceCaps actual=tight; actual.maximum_total_image_candidates=plans[point].image_candidate_count;
            work.require(plans[point].maximum_work_units);
            const I previous=mul(point,first.output_numerical_bytes);
            auto value=make_periodic_gaussian_bloch_iao(reference,ao,minimal,system,point,
                add(other,add(seed_live,previous)),options.source,options.iao,actual);
            if (value.memory().image_candidate_count != plans[point].image_candidate_count
                || value.memory().output_numerical_bytes != first.output_numerical_bytes)
                throw std::logic_error("Gaussian localization actual source differs from admitted point plan");
            limit(value.diagnostics().charged_work_units,plans[point].maximum_work_units,
                  "Gaussian localization source exceeded reserved work");
            work.charge(value.diagnostics().charged_work_units);
            d.image_candidate_visits=add(d.image_candidate_visits,mul(3,value.memory().image_candidate_count));
            aggregate(d.source,value.diagnostics()); sources.u64(point); matches.u64(point);
            sources.text(value.source_identity_sha256()); matches.text(value.reference_match_identity_sha256());
            points.push_back(std::move(value)); ++d.constructed_iao_points;
            context.emit(Stage::PointReady);
        }
        result.source_identity_=sources.finish(); result.match_identity_=matches.finish();
        // Pointers are created ONLY after all exact-K reserved owners exist.
        std::vector<const PeriodicCorrelationBlochIAO*> pointers; pointers.reserve(static_cast<std::size_t>(k));
        for (const auto& point : points) pointers.push_back(&point.iao());
        work.require(add(opt_plan.initial_work_units,m.wannier_work_reservation));
        m.optimizer_work_cap=std::min(caps.maximum_optimizer_work_units,work.remaining()-m.wannier_work_reservation);
        const PeriodicCorrelationIAOOptimizerCaps optimizer_caps{caps.maximum_owned_numerical_bytes,m.optimizer_work_cap};
        result.optimizer_.emplace(optimize_periodic_correlation_iao_pm(reference,*seed,pointers.data(),pointers.size(),
            options.pm,options.optimizer,optimizer_caps,callback ? &CallbackContext::relay : nullptr,&context));
        work.charge(result.optimizer_->diagnostics().charged_work_units);
    }
    seed.reset();
    // Source plans are metadata only, but free their logical record storage
    // before Wannier as well. No IAO owner/provenance is retained by output.
    std::vector<SourcePlan>().swap(plans);
    m.output_numerical_bytes=opt_plan.output_numerical_bytes;
    if (result.optimizer_->converged()) {
        work.charge(m.wannier_work_reservation);
        result.wannier_.emplace(make_periodic_correlation_wannier_from_iao_optimizer(reference,
            *result.optimizer_,caps.maximum_owned_numerical_bytes,options.wannier));
        m.output_numerical_bytes=add(m.output_numerical_bytes,w_plan.retained_coefficient_bytes);
        context.emit(Stage::WannierReady);
    }
    context.emit(Stage::Finished);
    d.charged_work_units=work.used();
    if (d.image_candidate_visits != m.planned_total_image_visits)
        throw std::logic_error("Gaussian localization image visit inventory mismatch");
    Digest identity("vibeqc.periodic.gaussian-localization.result");
    identity.text(result.input_identity_); identity.text(result.source_identity_); identity.text(result.match_identity_);
    identity.text(result.optimizer_->optimizer_identity_sha256()); identity.u64(result.optimizer_->converged());
    if (result.wannier_) identity.text(result.wannier_->wannier_identity_sha256());
    result.identity_=identity.finish();
    return result;
}
} // namespace vibeqc
