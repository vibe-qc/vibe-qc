#include "vibeqc/periodic_gaussian_selected_local_ccsd_t.hpp"

#include <cfloat>
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace arithmetic = periodic_correlation_real_local_detail;
using I = std::uint64_t;
using arithmetic::add;
using arithmetic::mul;
using arithmetic::Digest;
using Selection = PeriodicGaussianSelectedLocalSelection;
using Options = PeriodicGaussianSelectedLocalCCSDTOptions;
using Caps = PeriodicGaussianSelectedLocalCCSDTCaps;
using Live = PeriodicGaussianSelectedLocalCCSDTLiveInventory;
using Plan = PeriodicGaussianSelectedLocalCCSDTPlan;
using Diagnostics = PeriodicGaussianSelectedLocalCCSDTDiagnostics;
using Stage = PeriodicGaussianSelectedLocalCCSDTStage;
using Progress = PeriodicGaussianSelectedLocalCCSDTProgress;
constexpr I fixed_controls = 65536 + sizeof(Options) + sizeof(Caps) + sizeof(Plan)
    + sizeof(Diagnostics) + 2*sizeof(Progress) + sizeof(PeriodicGaussianSelectedLocalCCSDTResult);
static_assert(sizeof(std::size_t) == 8 && sizeof(int) == 4,
              "Gaussian selected-local inventory requires 64-bit extents and int32");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian selected-local connection forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Gaussian selected-local connection requires binary64 evaluation"
#endif
void limit(I x, I cap, const char* message) {
    if (x > cap) throw std::length_error(message);
}
I subtract(I a, I b) {
    if (b>a) throw std::logic_error("Gaussian selected-local ownership subtraction is inconsistent");
    return a-b;
}
I triangle(I n) { return n%2 ? mul(n,add(n,1)/2) : mul(n/2,add(n,1)); }
void physical_particle_hole_controls(const Options& options, const Caps& caps) {
    const auto& c=caps.particle_hole;
    const auto& r=options.particle_hole_options.real_projection;
    arithmetic::tolerance_pair(r.reversal_absolute_tolerance,r.reversal_relative_tolerance);
    arithmetic::tolerance_pair(r.conjugacy_absolute_tolerance,r.conjugacy_relative_tolerance);
    arithmetic::tolerance_pair(r.self_q_absolute_tolerance,r.self_q_relative_tolerance);
    for (double x : {r.maximum_eri_projection_error,r.maximum_scalar_roundoff_error,
                    options.particle_hole_options.maximum_overlap_imaginary_norm})
        if (!std::isfinite(x) || x<=0)
            throw std::invalid_argument("Gaussian selected-local physical particle-hole projection controls must be positive and finite");
    for (I x : {options.particle_hole.auxiliary_block,options.particle_hole.panel.ao_pair_block,
            c.maximum_source_slots,c.maximum_pair_count,c.maximum_owned_numerical_bytes,c.maximum_control_storage_bytes,
            c.maximum_worker_bytes,c.maximum_node_bytes,c.maximum_work_units,c.maximum_interaction_control_bytes,
            c.maximum_factor_panels,c.maximum_tile_calls,c.maximum_reciprocal_candidates,c.maximum_image_candidates,
            c.interaction.maximum_factor_panels,c.interaction.maximum_tile_calls,
            c.interaction.maximum_image_candidate_evaluations,c.interaction.maximum_leaf_control_bytes,
            c.interaction.panel.maximum_tile_calls,c.interaction.panel.maximum_image_candidate_evaluations,
            c.interaction.panel.tile.maximum_image_candidates})
        if (!x) throw std::invalid_argument("Gaussian selected-local physical particle-hole caps must be positive");
    for (const auto* rsrc : {&c.interaction.resources,&c.interaction.metric,
            &c.interaction.panel.resources,&c.interaction.panel.tile.resources})
        for (I x : {rsrc->maximum_owned_numeric_bytes,rsrc->maximum_per_replica_inventoried_bytes,
                rsrc->maximum_node_inventoried_bytes,rsrc->maximum_candidate_evaluations,rsrc->maximum_work_units})
            if (!x) throw std::invalid_argument("Gaussian selected-local physical particle-hole nested caps must be positive");
}
void same_state(const PeriodicGaussianRHFResult& hf, const PeriodicCorrelationAdmittedReference& ref) {
    if (!hf.converged() || !hf.context_handle() || !ref.state_handle()
        || hf.state_handle().get() != ref.state_handle().get())
        throw std::invalid_argument("Gaussian selected-local correlation requires the exact captured HF state owner");
}
void view(const std::uint64_t* data, std::size_t count, std::size_t elements) {
    const I bytes = mul(16,count);
    arithmetic::extent(bytes);
    if (!count || !data || elements != mul(2,count)
        || reinterpret_cast<std::uintptr_t>(data) % alignof(std::uint64_t))
        throw std::invalid_argument("Gaussian selected-local labels require exact aligned nonempty uint64 [count,2] views");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(data))
        throw std::overflow_error("Gaussian selected-local label pointer extent overflow");
}
// Bounded minimal-basis census. The cap's combined AO/minimal census is
// reduced by the immutable HF AO inventory before any mutable content scan.
PeriodicGaussianBasisInventory minimal_inventory(const BasisSet& minimal,
    const PeriodicGaussianBasisInventory& ao, I atoms, I effective,
    const PeriodicGaussianBlochIAOCaps& caps) {
    PeriodicGaussianBasisInventory p;
    p.function_count=minimal.nbasis(); p.shell_count=minimal.nshells();
    if (!p.function_count || p.function_count > effective || !p.shell_count)
        throw std::invalid_argument("Gaussian selected-local minimal basis exceeds the retained HF space or is empty");
    limit(atoms,caps.maximum_atom_count,"Gaussian selected-local atom census cap exceeded");
    limit(add(ao.shell_count,p.shell_count),caps.maximum_shell_count,"Gaussian selected-local shell census cap exceeded");
    if (minimal.libint().size() != p.shell_count)
        throw std::invalid_argument("Gaussian selected-local minimal shell metadata is inconsistent");
    for (const auto& shell : minimal.libint()) {
        p.contraction_count=add(p.contraction_count,shell.contr.size());
        p.exponent_count=add(p.exponent_count,shell.alpha.size());
        limit(add(ao.contraction_count,p.contraction_count),caps.maximum_contraction_count,
              "Gaussian selected-local contraction census cap exceeded");
        limit(add(add(ao.exponent_count,ao.coefficient_count),add(p.exponent_count,p.coefficient_count)),
              caps.maximum_primitive_numeric_lanes,"Gaussian selected-local primitive census cap exceeded");
        for (const auto& contraction : shell.contr) {
            p.coefficient_count=add(p.coefficient_count,contraction.coeff.size());
            limit(add(add(ao.exponent_count,ao.coefficient_count),add(p.exponent_count,p.coefficient_count)),
                  caps.maximum_primitive_numeric_lanes,"Gaussian selected-local coefficient census cap exceeded");
        }
    }
    p.borrowed_active_numeric_bytes=mul(8,add(mul(3,p.shell_count),add(p.exponent_count,p.coefficient_count)));
    p.content_wire_bytes=add(75,add(mul(8,p.shell_count),add(mul(45,p.contraction_count),mul(16,p.coefficient_count))));
    limit(add(ao.content_wire_bytes,p.content_wire_bytes),caps.maximum_basis_content_wire_bytes,
          "Gaussian selected-local basis wire cap exceeded");
    limit(add(add(72,mul(28,atoms)),add(mul(4,add(ao.shell_count,p.shell_count)),
              add(ao.borrowed_active_numeric_bytes,p.borrowed_active_numeric_bytes))),
          caps.maximum_borrowed_active_numeric_bytes,"Gaussian selected-local borrowed basis/geometry cap exceeded");
    return p;
}
PeriodicGaussianBlochIAOCaps tight_source_caps(const PeriodicGaussianBlochIAOCaps& caps,
    const PeriodicGaussianBasisInventory& ao, const PeriodicGaussianBasisInventory& minimal, I atoms) {
    auto c=caps;
    c.maximum_atom_count=atoms;
    c.maximum_shell_count=add(ao.shell_count,minimal.shell_count);
    c.maximum_contraction_count=add(ao.contraction_count,minimal.contraction_count);
    c.maximum_primitive_numeric_lanes=add(add(ao.exponent_count,ao.coefficient_count),
                                        add(minimal.exponent_count,minimal.coefficient_count));
    c.maximum_basis_content_wire_bytes=add(ao.content_wire_bytes,minimal.content_wire_bytes);
    c.maximum_borrowed_active_numeric_bytes=add(add(72,mul(28,atoms)),
        add(mul(4,add(ao.shell_count,minimal.shell_count)),
            add(ao.borrowed_active_numeric_bytes,minimal.borrowed_active_numeric_bytes)));
    return c;
}
std::string input_identity(const PeriodicGaussianRHFResult& hf, const PeriodicCorrelationAdmittedReference& ref,
    const BasisSet& ao, const BasisSet& auxiliary, const BasisSet& minimal, const PeriodicSystem& system,
    Selection selection, const PeriodicGaussianBlochIAOCaps& tight, const PeriodicGaussianBasisInventory& expected) {
    same_state(hf,ref);
    hf.verify_physical_inputs(ao,auxiliary,system);
    const auto now=minimal_inventory(minimal,hf.context_handle()->inventory().ao,system.unit_cell.size(),
                                    ref.state().n_effective_orbitals(),tight);
    if (now.function_count != expected.function_count || now.shell_count != expected.shell_count
        || now.contraction_count != expected.contraction_count || now.exponent_count != expected.exponent_count
        || now.coefficient_count != expected.coefficient_count)
        throw std::invalid_argument("Gaussian selected-local minimal basis census changed during the call");
    Digest d("vibeqc.periodic.gaussian-selected-local.inputs.v1");
    d.string(hf.reference_source_identity_sha256());
    d.string(auxiliary_basis_content_identity_sha256(minimal));
    for (const auto* basis : {&ao,&minimal}) {
        d.u64(basis->nshells());
        for (std::size_t s=0;s<basis->nshells();++s) {
            const int atom=basis->shell_atom_index(s);
            if (atom < 0 || static_cast<I>(atom) >= system.unit_cell.size())
                throw std::invalid_argument("Gaussian selected-local shell atom label is out of range");
            d.u64(static_cast<I>(atom));
        }
    }
    d.u64(selection.domain_count);
    for (I a=0;a<selection.domain_count;++a) {
        const I cell=selection.domain_indices[2*a], orbital=selection.domain_indices[2*a+1];
        if (cell >= ref.state().n_kpoints() || orbital >= ref.state().n_basis())
            throw std::out_of_range("Gaussian selected-local domain label is out of range");
        for (I b=0;b<a;++b)
            if (cell==selection.domain_indices[2*b] && orbital==selection.domain_indices[2*b+1])
                throw std::invalid_argument("Gaussian selected-local domain contains duplicate labels");
        d.u64(cell); d.u64(orbital);
    }
    arithmetic::indices(selection.occupied_indices,selection.occupied_elements,selection.occupied_count,ref.state());
    d.u64(selection.occupied_count);
    for (I a=0;a<mul(2,selection.occupied_count);++a) d.u64(selection.occupied_indices[a]);
    d.u64(selection.virtual_translation_cell);
    return d.finish();
}
struct Events {
    const PeriodicGaussianRHFResult& hf;
    const PeriodicCorrelationAdmittedReference& reference;
    const BasisSet& ao; const BasisSet& auxiliary; const BasisSet& minimal; const PeriodicSystem& system;
    Selection selection;
    const PeriodicGaussianBlochIAOCaps& source_caps;
    const PeriodicGaussianBasisInventory& minimal_census;
    const std::string& inputs;
    const Plan& plan; Diagnostics& diagnostics;
    PeriodicGaussianSelectedLocalCCSDTCallback callback; void* context;
    void recheck() {
        if (input_identity(hf,reference,ao,auxiliary,minimal,system,selection,source_caps,minimal_census) != inputs)
            throw std::invalid_argument("Gaussian selected-local callback changed original physical inputs or selections");
        diagnostics.completed_input_checks=add(diagnostics.completed_input_checks,1);
    }
    void emit(Progress event) {
        if (!callback) return;
        limit(add(diagnostics.completed_progress_callbacks,1),plan.progress_callback_upper_bound,
              "Gaussian selected-local progress callback bound exceeded");
        event.callback_count=add(diagnostics.completed_progress_callbacks,1);
        event.retained_virtual_count=diagnostics.retained_virtual_count;
        callback(event,context);
        diagnostics.completed_progress_callbacks=event.callback_count;
        recheck();
    }
    void emit(Stage stage) { Progress p; p.stage=stage; emit(p); }
    static void localization(const PeriodicGaussianLocalizationProgress& p, void* context) {
        Progress e; e.stage=Stage::Localization; e.localization=p; static_cast<Events*>(context)->emit(e);
    }
    static void factors(const PeriodicGaussianRealLocalProviderProgress& p, void* context) {
        Progress e; e.stage=Stage::Factors; e.factors=p; static_cast<Events*>(context)->emit(e);
    }
    static void correlation(const PeriodicCorrelationRealLocalCCSDTProgress& p, void* context) {
        Progress e; e.stage=Stage::Correlation; e.correlation=p; static_cast<Events*>(context)->emit(e);
    }
    static void pair_mp2(const PeriodicGaussianPairMP2Progress& p, void* context) {
        Progress e; e.stage=Stage::PairMP2; e.pair_mp2=p; static_cast<Events*>(context)->emit(e);
    }
    static void pair_ccsd(const PeriodicGaussianPairCCSDProgress& p, void* context) {
        Progress e; e.stage=Stage::PairCCSD; e.pair_ccsd=p; static_cast<Events*>(context)->emit(e);
    }
    static void triple_spaces(const PeriodicGaussianTripleSpacesProgress& p, void* context) {
        Progress e; e.stage=Stage::TripleSpaces; e.triple_spaces=p; static_cast<Events*>(context)->emit(e);
    }
    static void local_triples(const PeriodicGaussianLocalTriplesProgress& p, void* context) {
        Progress e; e.stage=Stage::LocalTriples; e.local_triples=p; static_cast<Events*>(context)->emit(e);
    }
};
} // namespace

PeriodicGaussianSelectedLocalCCSDTPlan plan_periodic_gaussian_selected_local_ccsd_t(
    const PeriodicGaussianRHFResult& hf, const PeriodicCorrelationAdmittedReference& reference,
    Selection selection, const Options& options, const Live& live, const Caps& caps) {
    same_state(hf,reference);
    for (I cap : {caps.maximum_domain_owned_numerical_bytes,caps.space.maximum_owned_numerical_bytes,
            caps.space.maximum_work_units,caps.basis.maximum_owned_numerical_bytes,caps.basis.maximum_work_units,
            caps.factors.resources.maximum_owned_numeric_bytes,caps.factors.resources.maximum_work_units,
            caps.factors.maximum_progress_callbacks,caps.maximum_owned_numerical_bytes,
            caps.maximum_per_worker_inventoried_bytes,caps.maximum_node_inventoried_bytes,
            caps.maximum_factor_control_storage_bytes,caps.maximum_progress_callbacks,caps.maximum_work_units,
            caps.localization.maximum_owned_numerical_bytes,caps.localization.maximum_control_storage_bytes,
            caps.localization.maximum_work_units,caps.iao_source.maximum_shell_count,
            caps.iao_source.maximum_contraction_count,caps.iao_source.maximum_primitive_numeric_lanes,
            caps.iao_source.maximum_basis_content_wire_bytes,caps.iao_source.maximum_borrowed_active_numeric_bytes,
            caps.iao_source.maximum_atom_count,live.fixed_backend_margin_bytes_per_worker})
        if (!cap) throw std::invalid_argument("Gaussian selected-local enclosing and leaf caps/margin must be positive");
    I correlation_owned=0,correlation_work=0,correlation_events=0;
    if (options.selected_pair_domains && !options.pair_local_correlation)
        throw std::invalid_argument("Selected pair domains require the explicit pair-local correlation branch");
    if (options.physical_particle_hole_ccsd && !options.pair_local_correlation)
        throw std::invalid_argument("Physical particle-hole CCSD requires the explicit pair-local correlation branch");
    if (options.physical_particle_hole_ccsd) physical_particle_hole_controls(options,caps);
    const I mp2_owned=options.selected_pair_domains ? caps.domain_pair_mp2.maximum_owned_numerical_bytes
                                                   : caps.pair_mp2.maximum_owned_numerical_bytes;
    const I mp2_work=options.selected_pair_domains ? caps.domain_pair_mp2.maximum_work_units : caps.pair_mp2.maximum_work_units;
    const I mp2_events=options.selected_pair_domains ? caps.domain_pair_mp2.maximum_progress_callbacks
                                                   : caps.pair_mp2.maximum_progress_callbacks;
    if (options.pair_local_correlation) {
        for (I cap : {mp2_owned,mp2_work,mp2_events,caps.pair_ccsd.maximum_owned_numerical_bytes,
                caps.pair_ccsd.maximum_work_units,caps.pair_ccsd.maximum_progress_callbacks,
                caps.triple_spaces.maximum_owned_numerical_bytes,caps.triple_spaces.maximum_work_units,
                caps.triple_spaces.maximum_progress_callbacks,caps.local_triples.maximum_owned_numerical_bytes,
                caps.local_triples.maximum_work_units,caps.local_triples.maximum_progress_callbacks,
                caps.maximum_pair_control_storage_bytes})
            if (!cap) throw std::invalid_argument("Gaussian selected-local pair branch requires positive stage/control caps");
        correlation_owned=add(add(mp2_owned,caps.pair_ccsd.maximum_owned_numerical_bytes),
            add(caps.triple_spaces.maximum_owned_numerical_bytes,caps.local_triples.maximum_owned_numerical_bytes));
        // Two admission passes, the stage call and final lineage validation.
        // A complete stage work ceiling safely bounds each metadata pass.
        if (options.physical_particle_hole_ccsd) {
            // The physical wrapper cap ALREADY encloses every producer and
            // callback source replay. Charge it once, not four times. The
            // two external metadata-only plans are added below separately.
            correlation_work=add(caps.pair_ccsd.maximum_work_units,mul(4,
                add(mp2_work,add(caps.triple_spaces.maximum_work_units,caps.local_triples.maximum_work_units))));
        } else correlation_work=mul(4,add(add(mp2_work,caps.pair_ccsd.maximum_work_units),
            add(caps.triple_spaces.maximum_work_units,caps.local_triples.maximum_work_units)));
        correlation_events=add(add(mp2_events,caps.pair_ccsd.maximum_progress_callbacks),
            add(caps.triple_spaces.maximum_progress_callbacks,caps.local_triples.maximum_progress_callbacks));
        if (options.selected_pair_domains) {
            for (I cap : {caps.pair_domain_builder.maximum_owned_numerical_bytes,
                    caps.pair_domain_builder.maximum_work_units,caps.pair_domain_builder.maximum_control_storage_bytes})
                if (!cap) throw std::invalid_argument("Selected pair domains require explicit builder resource caps");
            if (!options.physical_particle_hole_ccsd)
                correlation_owned=add(correlation_owned,caps.pair_domain_builder.maximum_owned_numerical_bytes);
            correlation_work=add(correlation_work,mul(4,caps.pair_domain_builder.maximum_work_units));
            correlation_events=add(correlation_events,1);
        }
    } else {
        if (!caps.correlation.maximum_owned_numerical_bytes || !caps.correlation.maximum_work_units)
            throw std::invalid_argument("Gaussian selected-local common correlation caps must be positive");
        correlation_owned=caps.correlation.maximum_owned_numerical_bytes;
        correlation_work=caps.correlation.maximum_work_units;
        correlation_events=mul(2,add(options.correlation.ccsd.maximum_iterations,options.correlation.triples.maximum_iterations));
    }
    view(selection.domain_indices,selection.domain_count,selection.domain_elements);
    view(selection.occupied_indices,selection.occupied_count,selection.occupied_elements);
    const auto& state=reference.state(); const auto& ctx=*hf.context_handle();
    Plan p; p.n_kpoints=state.n_kpoints(); p.n_basis=state.n_basis();
    p.pair_local_correlation=options.pair_local_correlation;
    p.selected_pair_domains=options.selected_pair_domains;
    p.physical_particle_hole_ccsd=options.physical_particle_hole_ccsd;
    p.pair_rank_padding_reservation_bytes=options.pair_local_correlation ? caps.maximum_pair_rank_padding_bytes : 0;
    p.domain_count=selection.domain_count; p.occupied_count=selection.occupied_count;
    if (p.domain_count > mul(p.n_kpoints,p.n_basis)
        || p.occupied_count > mul(p.n_kpoints,state.n_correlated_occupied())
        || selection.virtual_translation_cell >= p.n_kpoints)
        throw std::invalid_argument("Gaussian selected-local selection exceeds the finite torus");
    if (options.physical_particle_hole_ccsd) {
        // Mirrors the physical wrapper's bounded receipt/metadata census.
        // Its plans invoke no amplitude, AO or gauge value callbacks; native
        // child count planners and immutable extent getters are included.
        p.physical_pair_ccsd_metadata_work_units_upper_bound=mul(1048576,
            add(1024,add(triangle(p.occupied_count),p.occupied_count)));
        correlation_work=add(correlation_work,mul(2,p.physical_pair_ccsd_metadata_work_units_upper_bound));
        limit(triangle(p.occupied_count),caps.particle_hole.maximum_pair_count,
            "Gaussian selected-local physical particle-hole pair cap exceeded");
        limit(mul(2,p.occupied_count),caps.particle_hole.maximum_source_slots,
            "Gaussian selected-local physical particle-hole source-slot cap exceeded");
    }
    if (options.localization.source.image_cutoff_bohr != ctx.options().ao_pair_image_cutoff_bohr)
        throw std::invalid_argument("Gaussian selected-local localization must use the actual HF AO-image cutoff");
    if (!options.domain.require_time_reversal || !options.domain.require_real_matrices)
        throw std::invalid_argument("Gaussian selected-local real PAOs require actual domain time-reversal and reality audits");
    const I ao_bytes=ctx.inventory().ao.borrowed_active_numeric_bytes;
    if (caps.iao_source.maximum_borrowed_active_numeric_bytes < add(ao_bytes,
            add(add(72,mul(28,hf.plan().atom_count)),mul(4,ctx.inventory().ao.shell_count))))
        throw std::length_error("Gaussian selected-local source cap cannot cover actual HF AO/geometry payload");
    p.additional_minimal_geometry_bytes_upper_bound=caps.iao_source.maximum_borrowed_active_numeric_bytes-ao_bytes;
    p.auxiliary_basis_active_bytes=ctx.inventory().auxiliary.borrowed_active_numeric_bytes;
    p.caller_index_bytes=mul(16,add(p.domain_count,p.occupied_count));
    p.borrowed_input_bytes_upper_bound=add(ctx.inventory().combined_borrowed_active_numeric_bytes,
        add(p.additional_minimal_geometry_bytes_upper_bound,p.caller_index_bytes));
    // All lower control reservations stay conservatively live, even after
    // individual phase objects are freed. They are NOT numerical array bytes.
    p.control_storage_reservation_bytes=add(fixed_controls,add(caps.localization.maximum_control_storage_bytes,
                                                              caps.maximum_factor_control_storage_bytes));
    if (options.pair_local_correlation)
        p.control_storage_reservation_bytes=add(p.control_storage_reservation_bytes,caps.maximum_pair_control_storage_bytes);
    const auto domain=plan_periodic_correlation_pao_domain(state.mesh(),p.n_basis,p.domain_count);
    limit(domain.peak_owned_numerical_bytes,caps.maximum_domain_owned_numerical_bytes,
          "Gaussian selected-local domain owned cap exceeded");
    const I domain_output=add(domain.retained_domain_index_bytes,domain.retained_matrix_bytes);
    const I loc=caps.localization.maximum_owned_numerical_bytes;
    const I space=caps.space.maximum_owned_numerical_bytes;
    const I basis=caps.basis.maximum_owned_numerical_bytes;
    const I factors=caps.factors.resources.maximum_owned_numeric_bytes;
    p.localization_phase_owned_upper_bound=loc;
    p.domain_phase_owned_upper_bound=add(loc,domain.peak_owned_numerical_bytes);
    p.space_phase_owned_upper_bound=add(add(loc,domain_output),space);
    p.basis_phase_owned_upper_bound=add(p.space_phase_owned_upper_bound,basis);
    p.factors_phase_owned_upper_bound=add(p.basis_phase_owned_upper_bound,factors);
    if (options.selected_pair_domains)
        p.pair_domain_preparation_phase_owned_upper_bound=add(p.factors_phase_owned_upper_bound,
            caps.pair_domain_builder.maximum_owned_numerical_bytes);
    // The default branch releases the source geometry before correlation.
    // Physical PH retains it only through MP2 and CCSD; its owned cap already
    // includes the maximum of common-remainder and local-PH workspaces.
    if (options.physical_particle_hole_ccsd) {
        const I geometry=options.selected_pair_domains ? caps.pair_domain_builder.maximum_owned_numerical_bytes
                                                       : add(domain_output,space);
        p.pair_mp2_phase_owned_upper_bound=add(add(add(basis,factors),add(loc,geometry)),mp2_owned);
        p.physical_pair_ccsd_phase_owned_upper_bound=add(p.pair_mp2_phase_owned_upper_bound,
            caps.pair_ccsd.maximum_owned_numerical_bytes);
    }
    p.correlation_phase_owned_upper_bound=add(add(basis,factors),correlation_owned);
    p.owned_numerical_upper_bound=std::max({p.localization_phase_owned_upper_bound,p.domain_phase_owned_upper_bound,
        p.space_phase_owned_upper_bound,p.basis_phase_owned_upper_bound,p.factors_phase_owned_upper_bound,
        p.pair_domain_preparation_phase_owned_upper_bound,
        p.pair_mp2_phase_owned_upper_bound,p.physical_pair_ccsd_phase_owned_upper_bound,
        p.correlation_phase_owned_upper_bound});
    p.per_worker_inventoried_bytes_upper_bound=add(add(p.owned_numerical_upper_bound,p.pair_rank_padding_reservation_bytes),
        add(p.borrowed_input_bytes_upper_bound,add(p.control_storage_reservation_bytes,
        add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    const auto& d=reference.dimensions(); const auto& b=reference.budget();
    p.replicas_per_node=mul(b.mpi_ranks,b.workers_per_rank);
    p.reference_base_node_bytes=add(add(d.external_bytes,d.shared_bytes),
        mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    p.node_inventoried_bytes_upper_bound=add(p.reference_base_node_bytes,
        mul(p.replicas_per_node,p.per_worker_inventoried_bytes_upper_bound));
    limit(p.owned_numerical_upper_bound,caps.maximum_owned_numerical_bytes,"Gaussian selected-local whole-call owned cap exceeded");
    limit(p.per_worker_inventoried_bytes_upper_bound,caps.maximum_per_worker_inventoried_bytes,
          "Gaussian selected-local whole-call worker cap exceeded");
    limit(p.node_inventoried_bytes_upper_bound,caps.maximum_node_inventoried_bytes,
          "Gaussian selected-local whole-call node cap exceeded");
    limit(p.node_inventoried_bytes_upper_bound,b.memory_limit_bytes,
          "Gaussian selected-local admitted-reference node budget exceeded");
    arithmetic::extent(p.owned_numerical_upper_bound);
    const I labels=add(p.domain_count,p.occupied_count);
    p.input_check_work_units=add(hf.plan().input_check_work_units,
        add(mul(512,add(caps.iao_source.maximum_basis_content_wire_bytes,
            add(caps.iao_source.maximum_shell_count,caps.iao_source.maximum_contraction_count))),
            mul(512,add(4096,mul(add(labels,1),add(labels,1))))));
    // Bounds the domain's pair-major full-k projector contractions, input
    // S/F/TR audits, unique-label scans and digest passes before it starts.
    p.domain_work_units_upper_bound=mul(4096,mul(p.n_kpoints,mul(mul(add(p.domain_count,1),add(p.domain_count,1)),
        mul(add(p.n_basis,1),mul(add(p.n_basis,1),add(p.n_basis,1))))));
    const I optimizer_events=mul(2,mul(options.localization.optimizer.maximum_iterations,
                                      add(options.localization.optimizer.maximum_line_search_trials,1)));
    p.progress_callback_upper_bound=add(32,add(p.n_kpoints,add(optimizer_events,
        add(caps.factors.maximum_progress_callbacks,correlation_events))));
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,
          "Gaussian selected-local enclosing callback cap exceeded");
    p.work_units_upper_bound=add(mul(add(p.progress_callback_upper_bound,2),p.input_check_work_units),
        add(caps.localization.maximum_work_units,add(p.domain_work_units_upper_bound,
        add(caps.space.maximum_work_units,add(caps.basis.maximum_work_units,
        add(caps.factors.resources.maximum_work_units,correlation_work))))));
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian selected-local enclosing work cap exceeded");
    return p;
}

const PeriodicGaussianLocalizationResult& PeriodicGaussianSelectedLocalCCSDTResult::unfinished_localization() const {
    if (!unfinished_) throw std::logic_error("Gaussian selected-local result has no unfinished localization");
    return *unfinished_;
}
const PeriodicCorrelationRealLocalCCSDTResult& PeriodicGaussianSelectedLocalCCSDTResult::correlation() const {
    if (!correlation_) {
        if (unfinished_) throw std::logic_error("Gaussian selected-local localization did not converge; correlation was not evaluated");
        throw std::logic_error("Gaussian selected-local common-space correlation was not evaluated");
    }
    return *correlation_;
}
bool PeriodicGaussianSelectedLocalCCSDTResult::converged() const {
    return plan_.pair_local_correlation ? local_triples_ && local_triples_->converged()
                                      : correlation_ && correlation_->converged();
}
const PeriodicGaussianPairMP2Result& PeriodicGaussianSelectedLocalCCSDTResult::pair_mp2() const {
    if (!pair_mp2_) throw std::logic_error("Gaussian selected-local pair MP2 was not evaluated");
    return *pair_mp2_;
}
const PeriodicGaussianPairCCSDResult& PeriodicGaussianSelectedLocalCCSDTResult::pair_ccsd() const {
    if (!pair_ccsd_) throw std::logic_error("Gaussian selected-local pair CCSD was not evaluated");
    return *pair_ccsd_;
}
const PeriodicGaussianTripleSpaces& PeriodicGaussianSelectedLocalCCSDTResult::triple_spaces() const {
    if (!triple_spaces_) throw std::logic_error("Gaussian selected-local TNO spaces were not evaluated");
    return *triple_spaces_;
}
const PeriodicGaussianLocalTriplesResult& PeriodicGaussianSelectedLocalCCSDTResult::local_triples() const {
    if (!local_triples_) throw std::logic_error("Gaussian selected-local triples were not evaluated");
    return *local_triples_;
}
bool PeriodicGaussianSelectedLocalCCSDTResult::periodic_energy_per_cell() const {
    return converged() && diagnostics_.complete_finite_torus_basis;
}
double PeriodicGaussianSelectedLocalCCSDTResult::correlation_energy_per_cell() const {
    if (!periodic_energy_per_cell())
        throw std::logic_error("Per-cell correlation energy requires a converged complete finite-torus basis");
    if (plan_.pair_local_correlation) return local_triples_->correlation_energy_per_cell();
    return arithmetic::finite(arithmetic::finite(correlation_->ccsd().final_snapshot.correlation_energy
        +correlation_->triples().final_snapshot.triples_energy)/static_cast<double>(plan_.n_kpoints));
}
double PeriodicGaussianSelectedLocalCCSDTResult::total_energy_per_cell() const {
    return arithmetic::finite(diagnostics_.hf_energy_per_cell+correlation_energy_per_cell());
}

PeriodicGaussianSelectedLocalCCSDTResult run_periodic_gaussian_selected_local_ccsd_t(
    const PeriodicGaussianRHFResult& hf, const PeriodicCorrelationAdmittedReference& reference,
    const BasisSet& ao, const BasisSet& auxiliary, const BasisSet& minimal, const PeriodicSystem& system,
    Selection selection, const Options& supplied_options, const Live& supplied_live, const Caps& supplied_caps,
    PeriodicGaussianSelectedLocalCCSDTCallback callback, void* callback_context) {
    const Options options=supplied_options; const Live live=supplied_live; const Caps caps=supplied_caps;
    const auto p=plan_periodic_gaussian_selected_local_ccsd_t(hf,reference,selection,options,live,caps);
    arithmetic::float_environment();
    const auto& state=reference.state(); const auto& context=*hf.context_handle();
    const auto minimal_census=minimal_inventory(minimal,context.inventory().ao,hf.plan().atom_count,
                                               state.n_effective_orbitals(),caps.iao_source);
    const auto tight=tight_source_caps(caps.iao_source,context.inventory().ao,minimal_census,hf.plan().atom_count);
    PeriodicGaussianSelectedLocalCCSDTResult result;
    result.plan_=p; result.hf_=hf.reference_source_identity_sha256();
    result.inputs_=input_identity(hf,reference,ao,auxiliary,minimal,system,selection,tight,minimal_census);
    auto& diagnostics=result.diagnostics_;
    diagnostics.completed_input_checks=1;
    diagnostics.hf_energy_per_cell=hf.diagnostics().captured_energy_per_cell;
    Events events{hf,reference,ao,auxiliary,minimal,system,selection,tight,minimal_census,result.inputs_,
                  result.plan_,diagnostics,callback,callback_context};
    events.emit(Stage::Begin);
    const I external=add(live.other_live_bytes_per_worker,p.control_storage_reservation_bytes);
    const I indices_and_external=add(p.caller_index_bytes,external);
    // The localizer owns/carries AO+minimal+geometry already. Only auxiliary
    // active lanes, original index views and explicit caller/control excess
    // are added, so numerical payload is not duplicated through aliases.
    // The child charges its own control storage. Omit that reserved ceiling
    // from its caller-live excess; the enclosing plan retains the ceiling.
    const I localizer_other=add(add(p.auxiliary_basis_active_bytes,
                                  indices_and_external-caps.localization.maximum_control_storage_bytes),
                                live.fixed_backend_margin_bytes_per_worker);
    std::optional<PeriodicGaussianLocalizationResult> localization;
    localization.emplace(localize_periodic_gaussian_occupied(reference,ao,minimal,system,localizer_other,
        options.localization,tight,caps.localization,callback ? &Events::localization : nullptr,&events));
    diagnostics.localization_memory=localization->memory(); diagnostics.localization=localization->diagnostics();
    result.localization_=localization->localization_identity_sha256();
    const auto finish_identity=[&](const std::string& domain, const std::string& space, const std::string& basis) {
        Digest d("vibeqc.periodic.gaussian-selected-local.ccsd-t.v1");
        d.string(result.hf_); d.string(result.inputs_); d.string(result.localization_);
        d.string(domain); d.string(space); d.string(basis); d.string(result.provider_);
        d.u32(result.correlation_evaluated()); d.u32(diagnostics.complete_finite_torus_basis);
        if (result.correlation_) d.string(result.correlation_->identity_sha256());
        if (options.pair_local_correlation) {
            if (options.selected_pair_domains) d.string("actual-selected-atomic-pair-domains;pair-frame-density-before-common-export");
            // Opt-in receipt is present even if localization/MP2 stops early.
            // Leaving the default codec untouched preserves legacy receipts.
            if (options.physical_particle_hole_ccsd)
                d.string("native-physical-all-q-bare-particle-hole-CCSD;dressed-remainder-common-frame;not-production-DLPNO");
            d.string("matched-finite-Gaussian-HF-recipe;global-RI;pair-Galerkin-CCSD;preprojected-TNO-coupled-triples;not-production-DLPNO");
            for (bool stage : {result.pair_mp2_evaluated(),result.pair_ccsd_evaluated(),
                    result.triple_spaces_evaluated(),result.local_triples_evaluated()}) d.u32(stage);
            if (result.pair_mp2_) d.string(result.pair_mp2_->identity_sha256());
            if (result.pair_ccsd_) d.string(result.pair_ccsd_->identity_sha256());
            if (result.triple_spaces_) d.string(result.triple_spaces_->identity_sha256());
            if (result.local_triples_) d.string(result.local_triples_->identity_sha256());
        } else d.string("matched-finite-Gaussian-HF-recipe;global-RI;common-selected-space;coupled-triples;not-production-DLPNO");
        d.real(diagnostics.hf_energy_per_cell); result.identity_=d.finish();
    };
    if (!localization->converged()) {
        result.unfinished_.emplace(std::move(*localization)); localization.reset();
        events.recheck(); finish_identity("","",""); events.emit(Stage::Finished);
        return result;
    }
    // The whole-call cap admits every still-live owner for these legacy
    // leaves, whose individual node inventories intentionally remain partial.
    std::optional<PeriodicCorrelationPAODomain> domain;
    domain.emplace(make_periodic_correlation_pao_domain(reference,selection.domain_indices,selection.domain_elements,
        selection.domain_count,caps.maximum_domain_owned_numerical_bytes,options.domain));
    diagnostics.domain=domain->diagnostics(); events.emit(Stage::DomainReady);
    std::optional<PeriodicCorrelationRealPAOSpace> real_space;
    real_space.emplace(make_periodic_correlation_real_pao_space(reference,*domain,options.space,caps.space));
    const auto& space=real_space->space();
    diagnostics.space=real_space->diagnostics(); diagnostics.retained_virtual_count=space.retained_dimension();
    diagnostics.complete_finite_torus_basis=selection.domain_count==mul(p.n_kpoints,p.n_basis)
        && selection.occupied_count==mul(p.n_kpoints,state.n_correlated_occupied())
        && space.retained_dimension()==mul(p.n_kpoints,state.n_virtual());
    events.emit(Stage::SpaceReady);
    const auto& optimizer=localization->optimizer(); const auto& wannier=localization->wannier();
    const I gauge_count=mul(p.n_kpoints,mul(state.n_correlated_occupied(),state.n_correlated_occupied()));
    const PeriodicCorrelationVirtualBlockSelection virtuals{0,space.retained_dimension(),selection.virtual_translation_cell};
    auto basis=make_periodic_correlation_real_local_basis(reference,wannier,optimizer.gauges_data(),gauge_count,
        *domain,space,selection.occupied_indices,selection.occupied_elements,selection.occupied_count,
        virtuals,options.basis,caps.basis);
    diagnostics.basis=basis.diagnostics(); events.emit(Stage::BasisReady);
    // Gauge payload is counted by the factor producer; only the optimizer's
    // separately retained active-band indices belong in additional storage.
    const I optimizer_index_bytes=mul(8,mul(p.n_kpoints,state.n_correlated_occupied()));
    // Original A and AO/minimal shell-to-atom labels remain borrowed too;
    // neither belongs to the context's active Gaussian-lane inventory.
    const I actual_minimal_geometry=add(minimal_census.borrowed_active_numeric_bytes,
        add(add(72,mul(28,hf.plan().atom_count)),
            mul(4,add(context.inventory().ao.shell_count,minimal_census.shell_count))));
    const PeriodicGaussianRealLocalProviderLiveInventory factor_live{
        add(add(indices_and_external-caps.maximum_factor_control_storage_bytes,actual_minimal_geometry),
            optimizer_index_bytes),0,
        live.fixed_backend_margin_bytes_per_worker};
    const auto factor_plan=plan_periodic_gaussian_real_local_provider(hf,reference,wannier,*domain,space,basis,
        options.factors,factor_live,caps.factors);
    limit(add(factor_plan.macro_fixed_object_bytes,factor_plan.maximum_leaf_fixed_object_bytes),
          caps.maximum_factor_control_storage_bytes,"Gaussian selected-local factor control reservation exceeded");
    auto provider=make_periodic_gaussian_real_local_provider(hf,reference,ao,auxiliary,wannier,
        optimizer.gauges_data(),gauge_count,*domain,space,basis,options.factors,options.real_factors,
        factor_live,caps.factors,callback ? &Events::factors : nullptr,&events);
    diagnostics.factors_memory=provider.memory(); diagnostics.factors=provider.receipt();
    result.provider_=provider.identity_sha256(); events.emit(Stage::FactorsReady);
    const std::string domain_identity=domain->pao_domain_identity_sha256();
    const std::string space_identity=real_space->identity_sha256();
    const std::string basis_identity=basis.identity_sha256();
    const I retained_localization=options.physical_particle_hole_ccsd ? localization->memory().output_numerical_bytes : 0;
    const I retained_common_geometry=options.physical_particle_hole_ccsd
        ? add(add(domain->memory().retained_domain_index_bytes,domain->memory().retained_matrix_bytes),
            space.memory().output_numerical_bytes) : 0;
    if (options.physical_particle_hole_ccsd
        && retained_localization!=add(optimizer_index_bytes,add(wannier.memory().caller_gauge_bytes,
            wannier.memory().retained_coefficient_bytes)))
        throw std::logic_error("Gaussian selected-local retained localization inventory differs");
    std::optional<PeriodicGaussianPairDomainBuilder> pair_domain_builder;
    if (options.selected_pair_domains) {
        // Preparation borrows the original Gaussian, localization, common
        // geometry and basis roles itself. Only source rows, index views and
        // additional caller storage remain external to that native census.
        PeriodicGaussianPairDomainBuilderLive builder_live{
            add(provider.memory().retained_row_bytes,add(p.caller_index_bytes,live.other_live_bytes_per_worker)),
            0,live.fixed_backend_margin_bytes_per_worker};
        const auto before=plan_periodic_gaussian_pair_domain_builder(hf,reference,*localization,basis,*domain,*real_space,
            options.occupied_domains,builder_live,caps.pair_domain_builder);
        limit(before.control_storage_reservation_bytes,caps.maximum_pair_control_storage_bytes,
            "Gaussian selected-local pair-domain control reservation cap exceeded");
        builder_live.other_live_control_bytes_per_worker=p.control_storage_reservation_bytes-before.control_storage_reservation_bytes;
        const auto admitted=plan_periodic_gaussian_pair_domain_builder(hf,reference,*localization,basis,*domain,*real_space,
            options.occupied_domains,builder_live,caps.pair_domain_builder);
        limit(admitted.worker_bytes,p.per_worker_inventoried_bytes_upper_bound,
            "Gaussian selected-local domain preparation exceeds whole worker cap");
        limit(admitted.required_node_memory_bytes,p.node_inventoried_bytes_upper_bound,
            "Gaussian selected-local domain preparation exceeds whole node cap");
        pair_domain_builder.emplace(make_periodic_gaussian_pair_domain_builder(hf,reference,ao,auxiliary,minimal,system,
            *localization,basis,std::move(*domain),std::move(*real_space),options.occupied_domains,
            builder_live,caps.pair_domain_builder));
        diagnostics.pair_domain_builder_memory=pair_domain_builder->memory();
        diagnostics.pair_domain_builder=pair_domain_builder->diagnostics();
        events.emit(Stage::PairDomainsReady);
    }
    // Physical PH requires the original localized coefficients and common
    // geometry through CCSD. In the builder branch, original optionals are
    // moved-from: reacquire child references from the builder when needed.
    // The default route retains its original release point and arithmetic.
    if (!options.physical_particle_hole_ccsd) {
        localization.reset(); real_space.reset(); domain.reset();
    }
    if (options.pair_local_correlation) {
        // Original geometry/bases/selection views still belong to the caller.
        // Every child already counts the common Fock/labels and Gaussian
        // rows, plus every retained preceding pair-local owner it borrows.
        const I originals=add(add(context.inventory().combined_borrowed_active_numeric_bytes,actual_minimal_geometry),
            add(p.caller_index_bytes,live.other_live_bytes_per_worker));
        const auto control_excess=[&](I child_controls) {
            limit(child_controls,caps.maximum_pair_control_storage_bytes,
                  "Gaussian selected-local pair control reservation cap exceeded");
            return p.control_storage_reservation_bytes-child_controls;
        };
        const auto phase_gate=[&](I worker,I node,I padding=0) {
            limit(padding,p.pair_rank_padding_reservation_bytes,
                  "Gaussian selected-local pair rank padding cap exceeded");
            limit(worker,p.per_worker_inventoried_bytes_upper_bound,
                  "Gaussian selected-local pair phase exceeds enclosing worker inventory");
            limit(node,p.node_inventoried_bytes_upper_bound,
                  "Gaussian selected-local pair phase exceeds enclosing node inventory");
        };
        const auto finish=[&]() {
            events.recheck();finish_identity(domain_identity,space_identity,basis_identity);events.emit(Stage::Finished);
        };
        const I mp2_originals=add(originals,add(retained_localization,
            options.selected_pair_domains ? 0 : retained_common_geometry));
        PeriodicGaussianPairMP2LiveInventory mp2_live{mp2_originals,live.fixed_backend_margin_bytes_per_worker};
        if (options.selected_pair_domains) {
            const auto before=plan_periodic_gaussian_domain_pair_mp2(reference,basis,provider,*pair_domain_builder,
                options.domain_pair_mp2,mp2_live,caps.domain_pair_mp2);
            mp2_live.other_live_bytes_per_worker=add(mp2_originals,control_excess(before.control_storage_reservation_bytes));
            const auto admitted=plan_periodic_gaussian_domain_pair_mp2(reference,basis,provider,*pair_domain_builder,
                options.domain_pair_mp2,mp2_live,caps.domain_pair_mp2);
            phase_gate(admitted.per_worker_inventoried_bytes,admitted.required_node_memory_bytes);
            result.pair_mp2_.emplace(run_periodic_gaussian_domain_pair_mp2(reference,basis,provider,*pair_domain_builder,
                options.domain_pair_mp2,mp2_live,caps.domain_pair_mp2,callback ? &Events::pair_mp2 : nullptr,&events));
            if (!options.physical_particle_hole_ccsd)
                pair_domain_builder.reset(); // MP2 owns original pair geometry, but not the common PH source.
        } else {
            const auto mp2_pre=plan_periodic_gaussian_pair_mp2(reference,basis,provider,options.pair_mp2,mp2_live,caps.pair_mp2);
            mp2_live.other_live_bytes_per_worker=add(mp2_originals,control_excess(mp2_pre.control_storage_reservation_bytes));
            const auto mp2_plan=plan_periodic_gaussian_pair_mp2(reference,basis,provider,options.pair_mp2,mp2_live,caps.pair_mp2);
            phase_gate(mp2_plan.per_worker_inventoried_bytes,mp2_plan.required_node_memory_bytes);
            result.pair_mp2_.emplace(run_periodic_gaussian_pair_mp2(reference,basis,provider,options.pair_mp2,mp2_live,
                caps.pair_mp2,callback ? &Events::pair_mp2 : nullptr,&events));
        }
        const auto release_physical_sources=[&]() {
            if (options.physical_particle_hole_ccsd) {
                pair_domain_builder.reset(); localization.reset(); real_space.reset(); domain.reset();
            }
        };
        if (!result.pair_mp2_->converged()) { release_physical_sources();finish();return result; }

        auto ccsd_options=options.pair_ccsd;
        ccsd_options.solver.maximum_integral_work_units_per_call=provider.memory().scalar_work_units;
        if (options.physical_particle_hole_ccsd) {
            const auto& ph_domain=options.selected_pair_domains ? pair_domain_builder->common_domain() : *domain;
            const auto& ph_space=options.selected_pair_domains ? pair_domain_builder->common_real_space().space()
                                                               : real_space->space();
            const auto& ph_wannier=localization->wannier();
            const auto& ph_optimizer=localization->optimizer();
            const I builder_remainder=options.selected_pair_domains
                ? subtract(pair_domain_builder->retained_numerical_bytes(),retained_common_geometry) : 0;
            const I builder_controls=options.selected_pair_domains ? pair_domain_builder->retained_control_storage_bytes() : 0;
            // The physical wrapper already borrows A+G+W+C itself. Only E,
            // optimizer active indices I, and builder's (J-C) remain here.
            const I ph_originals=add(subtract(originals,context.inventory().combined_borrowed_active_numeric_bytes),
                add(optimizer_index_bytes,builder_remainder));
            PeriodicGaussianPairCCSDLiveInventory ccsd_live{add(ph_originals,builder_controls),
                live.fixed_backend_margin_bytes_per_worker};
            const auto before=plan_periodic_gaussian_pair_ccsd_physical_particle_hole(hf,reference,
                ph_wannier,ph_domain,ph_space,basis,provider,*result.pair_mp2_,options.particle_hole,
                options.particle_hole_options,caps.particle_hole,ccsd_options,ccsd_live,caps.pair_ccsd);
            limit(before.metadata_work_units,p.physical_pair_ccsd_metadata_work_units_upper_bound,
                "Gaussian selected-local physical particle-hole metadata plan exceeds admission");
            // Include the COMPLETE retained builder controls in the child
            // union before subtracting it from the enclosing reservation.
            const I child_controls=add(before.ccsd.control_storage_reservation_bytes,builder_controls);
            ccsd_live.other_live_bytes_per_worker=add(add(ph_originals,builder_controls),control_excess(child_controls));
            const auto admitted=plan_periodic_gaussian_pair_ccsd_physical_particle_hole(hf,reference,
                ph_wannier,ph_domain,ph_space,basis,provider,*result.pair_mp2_,options.particle_hole,
                options.particle_hole_options,caps.particle_hole,ccsd_options,ccsd_live,caps.pair_ccsd);
            limit(admitted.metadata_work_units,p.physical_pair_ccsd_metadata_work_units_upper_bound,
                "Gaussian selected-local physical particle-hole metadata plan exceeds admission");
            phase_gate(admitted.per_worker_inventoried_bytes,admitted.required_node_memory_bytes,
                admitted.ccsd.solver_rank_padding_upper_bytes);
            diagnostics.physical_particle_hole_memory=admitted;
            result.pair_ccsd_.emplace(run_periodic_gaussian_pair_ccsd_physical_particle_hole(hf,reference,
                ao,auxiliary,ph_wannier,ph_optimizer.gauges_data(),gauge_count,ph_domain,ph_space,basis,provider,
                *result.pair_mp2_,options.particle_hole,options.particle_hole_options,caps.particle_hole,
                ccsd_options,ccsd_live,caps.pair_ccsd,callback ? &Events::pair_ccsd : nullptr,&events));
            if (!result.pair_ccsd_->split_bare_particle_hole()
                || !result.pair_ccsd_->particle_hole_physical_source_certified()
                || result.pair_ccsd_->physical_particle_hole_source_identity_sha256().size()!=64)
                throw std::logic_error("Gaussian selected-local physical particle-hole source certificate is missing");
            diagnostics.physical_particle_hole_evaluated=true;
        } else {
            PeriodicGaussianPairCCSDLiveInventory ccsd_live{originals,live.fixed_backend_margin_bytes_per_worker};
            const auto ccsd_pre=plan_periodic_gaussian_pair_ccsd(reference,basis,provider,*result.pair_mp2_,
                ccsd_options,ccsd_live,caps.pair_ccsd);
            ccsd_live.other_live_bytes_per_worker=add(originals,control_excess(ccsd_pre.control_storage_reservation_bytes));
            const auto ccsd_plan=plan_periodic_gaussian_pair_ccsd(reference,basis,provider,*result.pair_mp2_,
                ccsd_options,ccsd_live,caps.pair_ccsd);
            phase_gate(ccsd_plan.per_worker_inventoried_bytes,ccsd_plan.required_node_memory_bytes,ccsd_plan.solver_rank_padding_upper_bytes);
            result.pair_ccsd_.emplace(run_periodic_gaussian_pair_ccsd(reference,basis,provider,*result.pair_mp2_,
                ccsd_options,ccsd_live,caps.pair_ccsd,callback ? &Events::pair_ccsd : nullptr,&events));
        }
        release_physical_sources();
        if (!result.pair_ccsd_->converged()) { finish();return result; }

        PeriodicGaussianTripleSpacesLiveInventory spaces_live{originals,0,live.fixed_backend_margin_bytes_per_worker};
        const auto spaces_pre=plan_periodic_gaussian_triple_spaces(reference,basis,provider,*result.pair_mp2_,*result.pair_ccsd_,
            options.triple_spaces,spaces_live,caps.triple_spaces);
        spaces_live.other_live_control_bytes_per_worker=control_excess(spaces_pre.control_storage_reservation_bytes);
        const auto spaces_plan=plan_periodic_gaussian_triple_spaces(reference,basis,provider,*result.pair_mp2_,*result.pair_ccsd_,
            options.triple_spaces,spaces_live,caps.triple_spaces);
        phase_gate(spaces_plan.per_worker_inventoried_bytes,spaces_plan.required_node_memory_bytes);
        result.triple_spaces_.emplace(make_periodic_gaussian_triple_spaces(reference,basis,provider,*result.pair_mp2_,
            *result.pair_ccsd_,options.triple_spaces,spaces_live,caps.triple_spaces,
            callback ? &Events::triple_spaces : nullptr,&events));

        auto triples_options=options.local_triples;
        triples_options.moments.maximum_integral_work_units_per_call=provider.memory().scalar_work_units;
        PeriodicGaussianLocalTriplesLiveInventory triples_live{originals,0,live.fixed_backend_margin_bytes_per_worker};
        const auto triples_pre=plan_periodic_gaussian_local_triples(reference,basis,provider,*result.pair_mp2_,*result.pair_ccsd_,
            *result.triple_spaces_,triples_options,triples_live,caps.local_triples);
        triples_live.other_live_control_bytes_per_worker=control_excess(triples_pre.control_storage_reservation_bytes);
        const auto triples_plan=plan_periodic_gaussian_local_triples(reference,basis,provider,*result.pair_mp2_,*result.pair_ccsd_,
            *result.triple_spaces_,triples_options,triples_live,caps.local_triples);
        phase_gate(triples_plan.per_worker_inventoried_bytes,triples_plan.required_node_memory_bytes,
            triples_plan.rank_padding_reservation_bytes);
        result.local_triples_.emplace(run_periodic_gaussian_local_triples(reference,basis,provider,*result.pair_mp2_,
            *result.pair_ccsd_,*result.triple_spaces_,triples_options,triples_live,caps.local_triples,
            callback ? &Events::local_triples : nullptr,&events));
        // Original native lineage remains checked even for all-zero retained
        // TNO ranks and an exhausted (unconverged) triples iteration.
        validate_periodic_gaussian_triple_spaces(reference,basis,provider,*result.pair_mp2_,*result.pair_ccsd_,*result.triple_spaces_);
        finish();return result;
    }
    const PeriodicCorrelationRealLocalCCSDTInventory correlation_live{
        add(add(context.inventory().combined_borrowed_active_numeric_bytes,actual_minimal_geometry),
            add(indices_and_external,live.fixed_backend_margin_bytes_per_worker))};
    result.correlation_.emplace(solve_periodic_correlation_real_local_ccsd_t(reference,basis,provider.provider(),
        options.correlation,correlation_live,caps.correlation,callback ? &Events::correlation : nullptr,&events));
    events.recheck(); finish_identity(domain_identity,space_identity,basis_identity); events.emit(Stage::Finished);
    return result;
}
} // namespace vibeqc
