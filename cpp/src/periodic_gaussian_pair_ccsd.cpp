#include "vibeqc/periodic_gaussian_pair_ccsd.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace real = periodic_correlation_real_local_detail;
using U = std::uint64_t;
using real::add;
using real::mul;
using real::Digest;
using Options = PeriodicGaussianPairCCSDOptions;
using Live = PeriodicGaussianPairCCSDLiveInventory;
using Caps = PeriodicGaussianPairCCSDCaps;
using Plan = PeriodicGaussianPairCCSDPlan;
using Result = PeriodicGaussianPairCCSDResult;
using SolverPlan = BoundedRestrictedPairCCSDSolverMemoryPlan;
using SolverInventory = BoundedRestrictedPairCCSDSolverInventory;
using SinglesView = BoundedRestrictedPairCCSDSinglesView;
using PairView = BoundedRestrictedPairCCSDPairView;
using PHProducer = BoundedRestrictedPairCCSDParticleHoleProducer;
static_assert(sizeof(double) == 8 && FLT_EVAL_METHOD == 0,
    "Gaussian pair CCSD requires binary64 evaluation");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian pair CCSD forbids fast/finite-only math"
#endif
constexpr char policy[] = "actual-finite-Gaussian;all-unordered-pairs;full-common-frame-Galerkin-CCSD;"
    "independent-diagonal-singles-periodic-Eq39-cutoff;no-molecular-0.03-preset;"
    "unprojected-disconnected-singles-energy;no-pair-correction;no-triples;no-symmetry-reduction";
U subtract(U a, U b) {
    if (b > a) throw std::logic_error("Gaussian pair CCSD ownership subtraction is inconsistent");
    return a - b;
}
U triangular(U n) { return n % 2 ? mul(n, add(n, 1) / 2) : mul(n / 2, add(n, 1)); }
bool placed_same(PeriodicCorrelationPlacedOccupied a, PeriodicCorrelationPlacedOccupied b) {
    return a.occupied_index == b.occupied_index && a.cell == b.cell;
}
U mp2_control_bytes(const PeriodicGaussianPairMP2Result& warm, U, U pairs) {
    U bytes = add(sizeof(PeriodicGaussianPairMP2Result), add(mul(pairs, sizeof(PeriodicGaussianPairSpace)),
        add(warm.memory().retained_pair_seal_bytes, 6U*65U)));
    bytes = add(bytes, warm.diagnostics().retained_pair_geometry_control_bytes);
    return bytes;
}
PeriodicGaussianEmbeddedPairPNOOptions embedded_options(const Options& options) {
    PeriodicGaussianEmbeddedPairPNOOptions out;
    out.pno = options.singles;
    const auto& e = options.singles_embedding;
    out.maximum_diagonal_exchange_projection_norm = e.maximum_diagonal_exchange_projection_norm;
    out.maximum_fock_symmetry_projection_norm = e.maximum_fock_symmetry_projection_norm;
    out.maximum_exported_gram_error = e.maximum_exported_gram_error;
    out.maximum_exported_fock_error = e.maximum_exported_fock_error;
    out.maximum_exported_subspace_error = e.maximum_exported_subspace_error;
    return out;
}
U frame_validation_upper(U n) { return mul(512, add(add(mul(2,mul(n,n)), mul(2,n)),1024)); }
void verify_frame_payload(const PeriodicGaussianPairPNOFrameView& frame, U n) {
    const auto p=plan_periodic_gaussian_pair_pno_frame_validation(frame);
    // The existing fixed64KiB driver control reservation includes this
    // sequential digest. Numerical payloads are borrowed and never copied.
    if (p.work_units>frame_validation_upper(n) || p.control_storage_bytes>65536)
        throw std::logic_error("Gaussian pair CCSD frame verifier exceeds admitted bound");
    frame.verify_payload();
}
void limit(U value, U cap, const char* message) {
    if (!cap || value > cap) throw std::length_error(message);
}
void sha(const std::string& s) {
    if (s.size() != 64) throw std::invalid_argument("Gaussian pair CCSD requires native SHA-256 receipts");
    for (unsigned char c : s)
        if (!(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f'))
            throw std::invalid_argument("Gaussian pair CCSD receipt is not lowercase SHA-256");
}
void warm_metadata(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, U o, U n, U pairs) {
    const auto& native = provider.provider();
    (void) native.integral_provider(basis); // Also catches moved basis/provider owners after callbacks.
    if (basis.state_handle().get() != ref.state_handle().get()
        || native.state_handle().get() != ref.state_handle().get()
        || basis.allocation_identity() != ref.dimensions().allocation_identity)
        throw std::invalid_argument("Gaussian pair CCSD reference/basis/provider owner changed");
    if (!warm.converged() || !warm.matched_finite_gaussian_hf_recipe()
        || warm.memory().occupied_count != o || warm.memory().common_virtual_dimension != n
        || warm.memory().pair_count != pairs || warm.diagnostics().completed_pairs != pairs
        || warm.provider_identity_sha256() != provider.identity_sha256()
        || warm.hf_reference_source_identity_sha256() != provider.hf_reference_source_identity_sha256())
        throw std::invalid_argument("Gaussian pair CCSD requires converged matching actual-source MP2");
    const auto& solver = warm.solver(); const auto& first = warm.pair(0, 0);
    if (solver.memory().n_occupied != o || solver.memory().common_virtual_dimension != n
        || solver.memory().pair_count != pairs
        || first.state_handle().get() != ref.state_handle().get()
        || first.context_handle().get() != provider.context_handle().get()
        || first.allocation_identity() != ref.dimensions().allocation_identity
        || first.frame().basis_identity_sha256() != basis.identity_sha256()
        || warm.state_handle().get() != ref.state_handle().get()
        || warm.context_handle().get() != provider.context_handle().get())
        throw std::invalid_argument("Gaussian pair CCSD warm-start state/context or common frame differs");
    for (const auto* s : {&warm.identity_sha256(), &warm.pair_spaces_identity_sha256(),
        &warm.provider_identity_sha256(), &warm.hf_reference_source_identity_sha256(),
        &solver.input_identity_sha256(), &solver.payload_sha256()}) sha(*s);
}
U warm_pair_ct(const PeriodicGaussianPairMP2Result& warm, U o, U n, U pairs) {
    const auto& m = warm.solver().memory();
    const U r2 = m.total_amplitude_elements;
    if (m.borrowed_pair_numeric_bytes % 8 || r2 > mul(pairs, mul(n, n)))
        throw std::logic_error("Gaussian pair CCSD MP2 pair count inventory is malformed");
    // MP2's borrowed per-pair data are C[n,r], G[r,r], eps[r].
    const U rank_numerator = subtract(m.borrowed_pair_numeric_bytes / 8, r2);
    if (rank_numerator % add(n, 1))
        throw std::logic_error("Gaussian pair CCSD MP2 rank inventory is inconsistent");
    const U rank_sum = rank_numerator / add(n, 1);
    const U generation_sum = warm.diagnostics().generation_dimension_sum;
    const U generation_coefficients = warm.diagnostics().retained_generation_coefficient_bytes;
    const U expected_spaces = add(generation_coefficients,
        mul(8, add(add(mul(n, rank_sum), rank_sum), add(generation_sum, r2))));
    if (rank_sum > mul(pairs, n) || expected_spaces != warm.diagnostics().retained_pair_bytes
        || generation_sum > mul(pairs, n) || (!warm.domain_generated() && generation_sum != mul(pairs, n))
        || generation_coefficients > mul(8,mul(n,rank_sum)) || generation_coefficients%8
        || (!warm.domain_generated() && generation_coefficients)
        || m.amplitude_snapshot_bytes != mul(8, r2)
        || m.retained_pair_record_bytes != mul(16, pairs)
        || m.output_numerical_bytes != add(m.amplitude_snapshot_bytes, m.retained_pair_record_bytes)
        || m.borrowed_fock_bytes != mul(8, mul(o, o)))
        throw std::logic_error("Gaussian pair CCSD retained warm-start payload inventory differs");
    return mul(8, add(mul(n, rank_sum), r2));
}
void verify_pairs(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const Plan& p) {
    warm_metadata(ref, basis, provider, warm, p.occupied_count, p.common_virtual_dimension, p.pair_count);
    U retained = 0, pair_ct = 0, maximum_rank = 0, minimum_rank = p.common_virtual_dimension, zero_ranks = 0;
    U generation_sum = 0, diagonal_generation_sum = 0, embedding_bytes = 0, receipt_bytes = 0, generation_coefficients = 0;
    U geometry_bytes = 0, geometry_controls = 0;
    const U o = p.occupied_count, n = p.common_virtual_dimension;
    for (U i = 0; i < o; ++i) for (U j = i; j < o; ++j) {
        const auto& pair = warm.pair(i, j); const auto pn = pair.frame();
        const U r = pair.memory().retained_dimension;
        const U m = pn.generation_dimension();
        maximum_rank = std::max(maximum_rank, r); minimum_rank = std::min(minimum_rank, r);
        if (!r) ++zero_ranks;
        const auto t = warm.solver().stored_amplitudes_view(i, j);
        if (r > n || warm.solver().pair_rank(i, j) != r
            || pair.state_handle().get() != ref.state_handle().get()
            || pair.context_handle().get() != provider.context_handle().get()
            || pair.allocation_identity() != ref.dimensions().allocation_identity
            || pair.local_basis_identity_sha256() != basis.local_basis_identity_sha256()
            || pair.consumed_sources_identity_sha256() != provider.consumed_sources_identity_sha256()
            || pn.basis_identity_sha256() != basis.identity_sha256()
            || pn.provider_identity_sha256() != provider.identity_sha256()
            || pn.hf_reference_source_identity_sha256() != provider.hf_reference_source_identity_sha256()
            || pn.occupied_count() != o
            || pn.occupied_slot_i() != i || pn.occupied_slot_j() != j
            || !placed_same(pn.occupied_i(), basis.occupied(i)) || !placed_same(pn.occupied_j(), basis.occupied(j))
            || pn.common_virtual_dimension() != n || pn.retained_dimension() != r || !m || r > m || m > n
            || pair.memory().virtual_count != n || pair.memory().generation_dimension != m
            || pair.memory().occupied_slot_i != i
            || pair.memory().occupied_slot_j != j || pair.coefficients().size() != mul(n, r)
            || pair.exchange_integrals().size() != mul(r, r) || pair.energies().size() != r
            || pn.original_pno_occupations().size() != m || t.element_count != mul(r, r))
            throw std::invalid_argument("Gaussian pair CCSD warm-start pair owner, frame or extent differs");
        for (const auto* s : {&pair.identity_sha256(), &pair.payload_sha256(),
            &pair.raw_exchange_integral_identity_sha256(), &pair.exchange_integral_identity_sha256(),
            &pair.allocation_identity(), &pair.local_basis_identity_sha256(), &pair.consumed_sources_identity_sha256(),
            &pn.identity_sha256(), &pn.payload_sha256(), &pn.basis_identity_sha256(), &pn.provider_identity_sha256(),
            &pn.hf_reference_source_identity_sha256(), &pn.common_exchange_integral_identity_sha256(),
            &pn.initial_amplitude_identity_sha256(), &pn.density_identity_sha256()}) sha(*s);
        const U bytes = add(pn.retained_numerical_bytes(), mul(8, mul(r, r)));
        if (bytes != pair.memory().retained_output_bytes)
            throw std::logic_error("Gaussian pair CCSD individual retained pair inventory differs");
        generation_sum = add(generation_sum, m);
        receipt_bytes = add(receipt_bytes, add(7U*65U, pn.retained_receipt_payload_bytes()));
        if (warm.domain_generated()) {
            const auto& geometry=warm.pair_generation_geometry(i,j);
            geometry_bytes=add(geometry_bytes,geometry.retained_numerical_bytes());
            geometry_controls=add(geometry_controls,geometry.retained_control_storage_bytes());
            for(const auto* receipt:{&geometry.identity_sha256(),&geometry.builder_identity_sha256(),
                &geometry.basis_identity_sha256(),&geometry.hf_reference_source_identity_sha256(),&geometry.allocation_identity()}) sha(*receipt);
            const auto& embedded = pn.embedded(); // Strict native tag; never a relabelled legacy PNO.
            generation_coefficients = add(generation_coefficients,mul(8,mul(m,r)));
            for (const auto* receipt : {&embedded.embedding_identity_sha256(), &embedded.raw_exchange_identity_sha256(),
                &embedded.projected_exchange_identity_sha256(), &embedded.raw_fock_identity_sha256(),
                &embedded.projected_fock_identity_sha256(), &embedded.source_payload_receipt_sha256()}) sha(*receipt);
            if (i == j) {
                const auto& e = warm.diagonal_generation_embedding(i);
                if (e.state_handle().get() != ref.state_handle().get()
                    || e.allocation_identity() != ref.dimensions().allocation_identity
                    || e.common_basis_identity_sha256() != basis.identity_sha256()
                    || e.memory().common_dimension != n || e.memory().pair_dimension != m
                    || e.identity_sha256() != embedded.embedding_identity_sha256())
                    throw std::invalid_argument("Gaussian pair CCSD original diagonal generation embedding differs");
                (void)e.coefficients_data(); (void)e.energies_data();
                embedding_bytes = add(embedding_bytes, e.memory().output_numerical_bytes);
                diagonal_generation_sum = add(diagonal_generation_sum, m);
            }
        } else {
            (void)pn.legacy();
            if (m != n) throw std::logic_error("Gaussian pair CCSD legacy generation dimension differs");
        }
        retained = add(retained, bytes);
        pair_ct = add(pair_ct, mul(8, add(mul(n, r), mul(r, r))));
    }
    if (retained != warm.diagnostics().retained_pair_bytes || pair_ct != p.borrowed_mp2_pair_ct_bytes
        || minimum_rank != warm.diagnostics().minimum_pair_rank
        || maximum_rank != warm.diagnostics().maximum_pair_rank || zero_ranks != warm.diagnostics().zero_rank_pairs
        || warm.diagnostics().all_pairs_full_rank != (minimum_rank == n)
        || generation_sum != warm.diagnostics().generation_dimension_sum
        || generation_coefficients != warm.diagnostics().retained_generation_coefficient_bytes
        || diagonal_generation_sum != warm.diagnostics().diagonal_generation_dimension_sum
        || embedding_bytes != warm.diagnostics().retained_generation_embedding_bytes
        || geometry_bytes != warm.diagnostics().retained_pair_geometry_bytes
        || geometry_controls != warm.diagnostics().retained_pair_geometry_control_bytes
        || receipt_bytes != warm.memory().retained_pair_seal_bytes
        || warm.domain_generated() != p.domain_generated
        || p.singles_generation_dimension_sum != (warm.domain_generated() ? diagonal_generation_sum : mul(o, n)))
        throw std::logic_error("Gaussian pair CCSD warm-start total inventory differs");
}
void solver_caps(const SolverPlan& p, const BoundedRestrictedPairCCSDSolverCaps& c) {
    limit(p.n_occupied, c.maximum_occupied_count, "Gaussian pair CCSD solver occupied cap exceeded");
    limit(p.common_virtual_dimension, c.maximum_common_virtual_dimension, "Gaussian pair CCSD solver common cap exceeded");
    limit(p.pair_count, c.maximum_pair_count, "Gaussian pair CCSD solver pair cap exceeded");
    limit(p.maximum_singles_rank, c.maximum_singles_rank, "Gaussian pair CCSD solver singles rank cap exceeded");
    limit(p.maximum_pair_rank, c.maximum_pair_rank, "Gaussian pair CCSD solver pair rank cap exceeded");
    limit(p.peak_owned_numerical_bytes, c.maximum_owned_numerical_bytes, "Gaussian pair CCSD solver owned cap exceeded");
    limit(p.per_replica_inventoried_bytes, c.maximum_per_replica_inventoried_bytes, "Gaussian pair CCSD solver worker cap exceeded");
    limit(p.required_node_inventoried_bytes, c.maximum_node_inventoried_bytes, "Gaussian pair CCSD solver node cap exceeded");
    limit(p.integral_calls_upper_bound, c.maximum_integral_calls, "Gaussian pair CCSD solver integral cap exceeded");
    limit(p.singles_calls_upper_bound, c.maximum_singles_calls, "Gaussian pair CCSD solver singles call cap exceeded");
    limit(p.doubles_calls_upper_bound, c.maximum_doubles_calls, "Gaussian pair CCSD solver doubles call cap exceeded");
    limit(p.work_units_upper_bound, c.maximum_work_units, "Gaussian pair CCSD solver work cap exceeded");
    if (p.split_bare_particle_hole)
        limit(p.particle_hole_calls_upper_bound, c.maximum_particle_hole_calls,
            "Gaussian pair CCSD solver particle-hole call cap exceeded");
}
SolverPlan solver_upper(U o, U n, const Options& options, const SolverInventory& inventory,
    const BoundedRestrictedCCSDIntegralProvider& integrals, const PHProducer* ph) {
    if (ph) return plan_bounded_restricted_pair_ccsd_solver_upper(o, n, n, n,
        options.solver, inventory, *ph, integrals.retained_numerical_bytes,
        integrals.maximum_transient_numerical_bytes);
    return plan_bounded_restricted_pair_ccsd_solver_upper(o, n, n, n,
        options.solver, inventory, integrals.retained_numerical_bytes,
        integrals.maximum_transient_numerical_bytes);
}
SolverInventory solver_inventory(const Plan& p, const Live& live, U singles_rank_sum,
    U solver_control, U solver_owned_tables, U solver_borrowed_tables, U singles_generation_coefficients) {
    SolverInventory inv{p.replicas_per_node, p.reference_base_node_bytes, 0,
        live.fixed_backend_margin_bytes_per_worker};
    // Generic solver counts all original F, and its provider includes the
    // occupied-label array alongside rows, exactly completing the basis.
    // Borrowed singles C and zero T1 are also already generic input roles.
    // On the split route, the generic solver ALSO counts producer additional
    // retained numerics and controls. Never add those roles here a second time.
    const U singles_non_c = add(singles_generation_coefficients,
        mul(8, add(singles_rank_sum, p.singles_generation_dimension_sum)));
    const U controls = subtract(p.control_storage_reservation_bytes,
        add(solver_control, add(solver_owned_tables, solver_borrowed_tables)));
    inv.other_live_bytes_per_replica = add(live.other_live_bytes_per_worker,
        add(subtract(p.borrowed_mp2_numerical_bytes, p.borrowed_mp2_pair_ct_bytes),
        add(singles_non_c, controls)));
    return inv;
}
void options_wire(Digest& h, const Options& o) {
    for (double x : {o.singles.occupation_cutoff, o.singles.denominator_floor,
        o.singles.maximum_initial_fvv_offdiagonal_norm, o.singles.semicanonical_orthonormality_tolerance,
        o.solver.denominator_floor, o.solver.singles_residual_tolerance, o.solver.doubles_residual_tolerance,
        o.solver.energy_tolerance, o.solver.coefficient_orthogonality_tolerance,
        o.solver.maximum_diagonal_update_antisymmetry_norm}) h.real(x);
    for (const auto& e : {o.singles.pno_eigensolver, o.singles.semicanonical_eigensolver}) {
        h.u64(e.max_sweeps); h.real(e.relative_offdiagonal_tolerance);
    }
    h.u64(o.solver.maximum_iterations); h.u64(o.solver.maximum_integral_work_units_per_call);
}
struct Events {
    PeriodicGaussianPairCCSDProgress event;
    PeriodicGaussianPairCCSDCallback callback = nullptr;
    void* context = nullptr;
    U maximum = 0;
    const PeriodicCorrelationAdmittedReference* reference;
    const PeriodicCorrelationRealLocalBasis* basis;
    const PeriodicGaussianRealLocalProvider* provider;
    const PeriodicGaussianPairMP2Result* warm;
    const Plan* plan;
    std::array<char,64> warm_sha{}, provider_sha{};
    // Equal-content move replacement must not leave the solver's original
    // borrowed F views dangling. Check storage before any callback re-scan.
    std::array<const void*,4> basis_views{};
    void pin_basis() {
        basis_views={basis->occupied_indices_data(),basis->f_oo_data(),
            basis->f_vv_data(),basis->f_ov_data()};
    }
    void emit(PeriodicGaussianPairCCSDStage stage) {
        if (event.callback_count >= maximum) throw std::length_error("Gaussian pair CCSD progress cap exceeded");
        event.stage = stage; ++event.callback_count;
        if (callback) callback(event, context);
        real::float_environment();
        if (basis_views!=std::array<const void*,4>{basis->occupied_indices_data(),basis->f_oo_data(),
            basis->f_vv_data(),basis->f_ov_data()})
            throw std::invalid_argument("Gaussian pair CCSD original basis storage changed across progress");
        verify_pairs(*reference, *basis, *provider, *warm, *plan);
        if (!std::equal(warm_sha.begin(), warm_sha.end(), warm->identity_sha256().begin())
            || provider->identity_sha256().size() != 64
            || !std::equal(provider_sha.begin(), provider_sha.end(), provider->identity_sha256().begin()))
            throw std::invalid_argument("Gaussian pair CCSD immutable source changed at callback");
    }
    static void solver(const BoundedRestrictedPairCCSDSolverProgress& progress, void* context) {
        auto& self = *static_cast<Events*>(context); self.event.solver = progress;
        self.emit(PeriodicGaussianPairCCSDStage::Solver);
    }
};

Plan embedded_ccsd_plan(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const Options& options, const Live& live, const Caps& caps,
    const PHProducer* ph) {
    Plan p; p.domain_generated = true;
    p.split_bare_particle_hole = ph != nullptr;
    p.borrowed_particle_hole_additional_numerical_bytes = ph ? ph->additional_retained_numerical_bytes : 0;
    const U o = p.occupied_count = basis.memory().occupied_count, n = p.common_virtual_dimension = basis.memory().virtual_count;
    if (!o || !n) throw std::invalid_argument("Gaussian pair CCSD occupied/common dimensions must be positive");
    const U P = p.pair_count = triangular(o), nn = mul(n, n), on = mul(o, n);
    limit(P, caps.maximum_pair_count, "Gaussian pair CCSD outer pair count cap exceeded");
    warm_metadata(ref, basis, provider, warm, o, n, P);
    const auto eo = embedded_options(options);
    const PeriodicGaussianEmbeddedPairPNOLiveInventory minimal{0, 0, live.fixed_backend_margin_bytes_per_worker};
    const auto upper = plan_periodic_gaussian_embedded_pair_pno_counts(
        {o, n, n, provider.memory().scalar_work_units, provider.memory().retained_row_bytes}, eo, minimal);
    p.embedded_singles_upper = upper;
    p.singles_generation_dimension_sum = warm.diagnostics().diagonal_generation_dimension_sum;
    p.borrowed_generation_embedding_bytes = warm.diagnostics().retained_generation_embedding_bytes;
    if (p.singles_generation_dimension_sum < o || p.singles_generation_dimension_sum > on
        || p.borrowed_generation_embedding_bytes != mul(8, mul(add(n, 1), p.singles_generation_dimension_sum)))
        throw std::invalid_argument("Gaussian pair CCSD original diagonal generation census differs");
    p.borrowed_mp2_pair_ct_bytes = warm_pair_ct(warm, o, n, P);
    p.borrowed_mp2_numerical_bytes = add(warm.diagnostics().retained_pair_geometry_bytes,
        add(warm.diagnostics().retained_pair_bytes, warm.solver().memory().output_numerical_bytes));
    p.borrowed_mp2_control_bytes = mp2_control_bytes(warm, o, P);
    p.borrowed_basis_bytes = upper.borrowed_basis_bytes; p.borrowed_provider_row_bytes = upper.borrowed_provider_row_bytes;
    const auto& dims = ref.dimensions(); const auto& budget = ref.budget();
    p.replicas_per_node = mul(budget.mpi_ranks, budget.workers_per_rank);
    p.reference_base_node_bytes = add(add(dims.external_bytes, dims.shared_bytes),
        mul(budget.mpi_ranks, add(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)));
    const auto callback = provider.provider().integral_provider(basis);
    const SolverInventory initial{p.replicas_per_node, p.reference_base_node_bytes, 0, live.fixed_backend_margin_bytes_per_worker};
    const auto solver = solver_upper(o, n, options, initial, callback, ph);
    // Full-rank common C/eps upper roles plus m_i*n local D and exactly m_i
    // occupations. The generic solver consumes neither D nor occupations.
    const U generation_coefficients_upper = mul(8,mul(n,p.singles_generation_dimension_sum));
    p.singles_output_upper_bytes = add(generation_coefficients_upper,
        mul(8, add(mul(add(n, 1), on), p.singles_generation_dimension_sum)));
    p.zero_singles_upper_bytes = mul(8, on);
    p.singles_generation_phase_upper_bytes = add(mul(o-1, upper.retained_output_upper_bytes), upper.peak_owned_numerical_bytes);
    p.solver_phase_owned_upper_bytes = add(p.singles_output_upper_bytes,
        add(p.zero_singles_upper_bytes, solver.peak_owned_numerical_bytes));
    p.peak_owned_numerical_bytes = std::max(p.singles_generation_phase_upper_bytes, p.solver_phase_owned_upper_bytes);
    p.solver_rank_padding_upper_bytes = subtract(mul(16, mul(P, nn)), p.borrowed_mp2_pair_ct_bytes);
    p.control_storage_reservation_bytes = add(65536U+sizeof(Plan)+sizeof(Result)+sizeof(Options)+sizeof(Live)+sizeof(Caps)
        +sizeof(Events)+7U*65U, add(p.borrowed_mp2_control_bytes,
        add(mul(o, add(sizeof(PeriodicGaussianPairPNOFrame), 14U*65U)),
        add(upper.control_storage_reservation_bytes, add(solver.control_storage_reservation_bytes,
        add(solver.owned_snapshot_table_bytes, solver.borrowed_input_table_bytes))))));
    if (ph) p.control_storage_reservation_bytes = add(p.control_storage_reservation_bytes, 2U*sizeof(PHProducer)+65U);
    const U borrowed = add(p.borrowed_particle_hole_additional_numerical_bytes,
        add(p.borrowed_mp2_numerical_bytes, add(p.borrowed_basis_bytes, p.borrowed_provider_row_bytes)));
    const U extras = add(p.control_storage_reservation_bytes, add(live.other_live_bytes_per_worker, live.fixed_backend_margin_bytes_per_worker));
    p.per_worker_inventoried_bytes = add(borrowed, add(extras, std::max(p.singles_generation_phase_upper_bytes,
        add(p.solver_phase_owned_upper_bytes, p.solver_rank_padding_upper_bytes))));
    p.required_node_memory_bytes = add(p.reference_base_node_bytes, mul(p.replicas_per_node, p.per_worker_inventoried_bytes));
    p.integral_calls_upper_bound = add(mul(o, upper.integral_calls), solver.integral_calls_upper_bound);
    p.progress_callback_upper_bound = add(add(o, options.solver.maximum_iterations), 2);
    p.driver_work_units = mul(32768, mul(add(add(add(P, o), n), 1), add(p.progress_callback_upper_bound, 4)));
    p.driver_work_units = add(p.driver_work_units, mul(add(P,o),frame_validation_upper(n)));
    p.work_units_upper_bound = add(p.driver_work_units, add(mul(o, upper.work_units), solver.work_units_upper_bound));
    for (U bytes : {p.peak_owned_numerical_bytes, p.control_storage_reservation_bytes,
        p.borrowed_mp2_numerical_bytes, p.solver_rank_padding_upper_bytes}) real::extent(bytes);
    if (o > std::vector<PeriodicGaussianPairPNOFrame>().max_size() || o > std::vector<SinglesView>().max_size()
        || P > std::vector<PairView>().max_size() || on > std::vector<double>().max_size())
        throw std::length_error("Gaussian pair CCSD native table extent exceeded");
    limit(p.peak_owned_numerical_bytes, caps.maximum_owned_numerical_bytes, "Gaussian pair CCSD outer owned cap exceeded");
    limit(p.per_worker_inventoried_bytes, caps.maximum_per_worker_inventoried_bytes, "Gaussian pair CCSD outer worker cap exceeded");
    limit(p.required_node_memory_bytes, caps.maximum_node_inventoried_bytes, "Gaussian pair CCSD outer node cap exceeded");
    limit(p.required_node_memory_bytes, budget.memory_limit_bytes, "Gaussian pair CCSD admitted-reference node cap exceeded");
    limit(p.integral_calls_upper_bound, caps.maximum_integral_calls, "Gaussian pair CCSD outer integral cap exceeded");
    limit(p.progress_callback_upper_bound, caps.maximum_progress_callbacks, "Gaussian pair CCSD outer progress cap exceeded");
    limit(p.work_units_upper_bound, caps.maximum_work_units, "Gaussian pair CCSD outer work cap exceeded");
    // Conservative inner caps cover m=n before any per-i owner traversal.
    // Actual leaf plans below use the real embedding, not this count census.
    const auto& c = caps.embedded_singles;
    limit(n, c.maximum_common_dimension, "Gaussian pair CCSD embedded common cap exceeded");
    limit(n, c.maximum_generation_dimension, "Gaussian pair CCSD embedded generation upper cap exceeded");
    limit(upper.peak_owned_numerical_bytes, c.maximum_owned_numerical_bytes, "Gaussian pair CCSD embedded owned cap exceeded");
    limit(p.control_storage_reservation_bytes, c.maximum_control_storage_bytes_per_worker, "Gaussian pair CCSD embedded control cap exceeded");
    limit(p.per_worker_inventoried_bytes, c.maximum_per_worker_inventoried_bytes, "Gaussian pair CCSD embedded worker cap exceeded");
    limit(p.required_node_memory_bytes, c.maximum_node_inventoried_bytes, "Gaussian pair CCSD embedded node cap exceeded");
    limit(upper.integral_calls, c.maximum_integral_calls, "Gaussian pair CCSD embedded integral cap exceeded");
    limit(upper.work_units, c.maximum_work_units, "Gaussian pair CCSD embedded work cap exceeded");
    const auto inv = solver_inventory(p, live, on, solver.control_storage_reservation_bytes,
        solver.owned_snapshot_table_bytes, solver.borrowed_input_table_bytes,generation_coefficients_upper);
    p.solver_upper = solver_upper(o, n, options, inv, callback, ph);
    solver_caps(p.solver_upper, caps.solver);
    if (p.solver_upper.per_replica_inventoried_bytes != add(borrowed, add(extras,
        add(p.solver_phase_owned_upper_bytes, p.solver_rank_padding_upper_bytes))))
        throw std::logic_error("Gaussian pair CCSD embedded enclosing/solver inventory identity differs");
    verify_pairs(ref, basis, provider, warm, p);
    for (U i=0; i<o; ++i) {
        const auto& e = warm.diagonal_generation_embedding(i);
        PeriodicGaussianEmbeddedPairPNOLiveInventory ilive;
        ilive.fixed_backend_margin_bytes_per_worker = live.fixed_backend_margin_bytes_per_worker;
        ilive.other_live_numerical_bytes_per_worker = add(p.borrowed_particle_hole_additional_numerical_bytes,
            add(subtract(p.borrowed_mp2_numerical_bytes, e.memory().output_numerical_bytes),
            add(mul(o-1, upper.retained_output_upper_bytes), live.other_live_bytes_per_worker)));
        ilive.other_live_control_bytes_per_worker = subtract(p.control_storage_reservation_bytes, upper.control_storage_reservation_bytes);
        const auto exact = plan_periodic_gaussian_embedded_pair_pnos(ref, basis, provider, e, i, i, eo, ilive, c);
        if (exact.required_node_memory_bytes > p.required_node_memory_bytes || exact.work_units > upper.work_units)
            throw std::logic_error("Gaussian pair CCSD actual embedded leaf exceeds macro admission");
    }
    return p;
}
} // namespace

static Plan pair_ccsd_plan_impl(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const Options& options, const Live& live, const Caps& caps,
    const PHProducer* ph) {
    real::float_environment();
    if (!live.fixed_backend_margin_bytes_per_worker
        || options.singles.denominator_floor != options.solver.denominator_floor
        || options.solver.maximum_integral_work_units_per_call != provider.memory().scalar_work_units)
        throw std::invalid_argument("Gaussian pair CCSD requires explicit backend margin and matching native work/denominator controls");
    if (warm.domain_generated()) return embedded_ccsd_plan(ref, basis, provider, warm, options, live, caps, ph);
    PeriodicGaussianPairPNOLiveInventory pno_live{0, live.fixed_backend_margin_bytes_per_worker};
    const auto pno = plan_periodic_gaussian_pair_pnos(ref, basis, provider, 0, 0, options.singles, pno_live, caps.singles);
    Plan p;
    p.split_bare_particle_hole = ph != nullptr;
    p.borrowed_particle_hole_additional_numerical_bytes = ph ? ph->additional_retained_numerical_bytes : 0;
    const U o = p.occupied_count = pno.occupied_count, n = p.common_virtual_dimension = pno.virtual_count;
    p.singles_generation_dimension_sum = mul(o, n);
    const U pairs = p.pair_count = triangular(o), nn = mul(n, n);
    limit(pairs, caps.maximum_pair_count, "Gaussian pair CCSD outer pair count cap exceeded");
    warm_metadata(ref, basis, provider, warm, o, n, pairs);
    p.borrowed_mp2_pair_ct_bytes = warm_pair_ct(warm, o, n, pairs);
    p.borrowed_mp2_numerical_bytes = add(warm.diagnostics().retained_pair_bytes, warm.solver().memory().output_numerical_bytes);
    p.borrowed_mp2_control_bytes = mp2_control_bytes(warm, o, pairs);
    p.borrowed_basis_bytes = pno.borrowed_basis_bytes; p.borrowed_provider_row_bytes = pno.borrowed_provider_row_bytes;
    p.replicas_per_node = pno.replicas_per_node; p.reference_base_node_bytes = pno.reference_base_node_bytes;
    const auto callback = provider.provider().integral_provider(basis);
    SolverInventory initial{p.replicas_per_node, p.reference_base_node_bytes, 0, live.fixed_backend_margin_bytes_per_worker};
    const auto solver = solver_upper(o, n, options, initial, callback, ph);
    p.singles_output_upper_bytes = mul(o, pno.retained_output_upper_bytes);
    p.zero_singles_upper_bytes = mul(8, mul(o, n));
    p.singles_generation_phase_upper_bytes = add(mul(o - 1, pno.retained_output_upper_bytes), pno.peak_owned_numerical_bytes);
    p.solver_phase_owned_upper_bytes = add(p.singles_output_upper_bytes,
        add(p.zero_singles_upper_bytes, solver.peak_owned_numerical_bytes));
    p.peak_owned_numerical_bytes = std::max(p.singles_generation_phase_upper_bytes, p.solver_phase_owned_upper_bytes);
    p.solver_rank_padding_upper_bytes = subtract(mul(16, mul(pairs, nn)), p.borrowed_mp2_pair_ct_bytes);
    p.control_storage_reservation_bytes = add(65536U + sizeof(Plan) + sizeof(Result) + sizeof(Options)
        + sizeof(Live) + sizeof(Caps) + sizeof(Events) + 7U*65U,
        add(p.borrowed_mp2_control_bytes, add(mul(o, add(sizeof(PeriodicGaussianPairPNOFrame), 8U*65U)),
        add(pno.fixed_control_storage_bytes, add(solver.control_storage_reservation_bytes,
        add(solver.owned_snapshot_table_bytes, solver.borrowed_input_table_bytes))))));
    if (ph) p.control_storage_reservation_bytes = add(p.control_storage_reservation_bytes, 2U*sizeof(PHProducer)+65U);
    const U borrowed = add(p.borrowed_particle_hole_additional_numerical_bytes,
        add(p.borrowed_mp2_numerical_bytes, add(p.borrowed_basis_bytes, p.borrowed_provider_row_bytes)));
    const U extras = add(p.control_storage_reservation_bytes,
        add(live.other_live_bytes_per_worker, live.fixed_backend_margin_bytes_per_worker));
    // The rank padding only belongs to the solver phase, not generation.
    p.per_worker_inventoried_bytes = add(borrowed, add(extras,
        std::max(p.singles_generation_phase_upper_bytes,
            add(p.solver_phase_owned_upper_bytes, p.solver_rank_padding_upper_bytes))));
    p.required_node_memory_bytes = add(p.reference_base_node_bytes, mul(p.replicas_per_node, p.per_worker_inventoried_bytes));
    p.integral_calls_upper_bound = add(mul(o, pno.integral_calls), solver.integral_calls_upper_bound);
    p.progress_callback_upper_bound = add(add(o, options.solver.maximum_iterations), 2);
    // All pair-metadata/receipt checks before/after every external callback,
    // plus count/hash/descriptor preparation. No full numeric replay here.
    // Each pair includes fifteen 64-character receipt scans and their
    // comparisons, not merely fifteen scalar metadata operations.
    p.driver_work_units = mul(32768, mul(add(add(add(pairs, o), n), 1), add(p.progress_callback_upper_bound, 4)));
    p.work_units_upper_bound = add(p.driver_work_units, add(mul(o, pno.work_units), solver.work_units_upper_bound));
    for (U bytes : {p.peak_owned_numerical_bytes, p.control_storage_reservation_bytes,
        p.borrowed_mp2_numerical_bytes, p.solver_rank_padding_upper_bytes}) real::extent(bytes);
    if (o > std::vector<PeriodicGaussianPairPNOFrame>().max_size()
        || o > std::vector<SinglesView>().max_size() || pairs > std::vector<PairView>().max_size()
        || mul(o, n) > std::vector<double>().max_size())
        throw std::length_error("Gaussian pair CCSD native table extent exceeded");
    limit(p.peak_owned_numerical_bytes, caps.maximum_owned_numerical_bytes, "Gaussian pair CCSD outer owned cap exceeded");
    limit(p.per_worker_inventoried_bytes, caps.maximum_per_worker_inventoried_bytes, "Gaussian pair CCSD outer worker cap exceeded");
    limit(p.required_node_memory_bytes, caps.maximum_node_inventoried_bytes, "Gaussian pair CCSD outer node cap exceeded");
    limit(p.required_node_memory_bytes, ref.budget().memory_limit_bytes, "Gaussian pair CCSD admitted-reference node cap exceeded");
    limit(p.integral_calls_upper_bound, caps.maximum_integral_calls, "Gaussian pair CCSD outer integral cap exceeded");
    limit(p.progress_callback_upper_bound, caps.maximum_progress_callbacks, "Gaussian pair CCSD outer progress cap exceeded");
    limit(p.work_units_upper_bound, caps.maximum_work_units, "Gaussian pair CCSD outer work cap exceeded");
    pno_live.other_live_bytes_per_worker = add(p.borrowed_particle_hole_additional_numerical_bytes, add(p.borrowed_mp2_numerical_bytes,
        add(mul(o - 1, pno.retained_output_upper_bytes), add(live.other_live_bytes_per_worker,
        subtract(p.control_storage_reservation_bytes, pno.fixed_control_storage_bytes)))));
    p.singles_upper = plan_periodic_gaussian_pair_pnos(ref, basis, provider, 0, 0,
        options.singles, pno_live, caps.singles);
    const auto inv = solver_inventory(p, live, mul(o, n), solver.control_storage_reservation_bytes,
        solver.owned_snapshot_table_bytes, solver.borrowed_input_table_bytes,0);
    p.solver_upper = solver_upper(o, n, options, inv, callback, ph);
    solver_caps(p.solver_upper, caps.solver);
    const U expected_solver_worker = add(borrowed, add(extras,
        add(p.solver_phase_owned_upper_bytes, p.solver_rank_padding_upper_bytes)));
    if (p.solver_upper.per_replica_inventoried_bytes != expected_solver_worker)
        throw std::logic_error("Gaussian pair CCSD enclosing/solver inventory identity changed");
    return p;
}

Plan plan_periodic_gaussian_pair_ccsd(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const Options& options, const Live& live, const Caps& caps) {
    return pair_ccsd_plan_impl(ref, basis, provider, warm, options, live, caps, nullptr);
}
Plan plan_periodic_gaussian_pair_ccsd(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const PHProducer& input_ph,
    const Options& options, const Live& live, const Caps& caps) {
    const auto ph = input_ph;
    return pair_ccsd_plan_impl(ref, basis, provider, warm, options, live, caps, &ph);
}

const BoundedRestrictedPairCCSDSolverResult& Result::solver() const {
    if (!state_ || !context_ || !solver_ || singles_.size() != memory_.occupied_count)
        throw std::logic_error("Gaussian pair CCSD result is moved or incomplete");
    return *solver_;
}
const PeriodicGaussianPairPNOResult& Result::singles(U i) const {
    return singles_frame(i).legacy();
}
PeriodicGaussianPairPNOFrameView Result::singles_frame(U i) const {
    (void) solver();
    if (i >= memory_.occupied_count) throw std::out_of_range("Gaussian pair CCSD singles index out of range");
    return singles_.at(static_cast<std::size_t>(i)).view();
}
bool Result::converged() const { return solver().final_snapshot().converged; }
bool Result::periodic_energy_per_cell() const {
    return (!memory_.split_bare_particle_hole || physical_particle_hole_source_)
        && diagnostics_.complete_common_finite_torus_basis && converged();
}
const std::string& Result::split_execution_identity_sha256() const {
    return solver().split_execution_identity_sha256();
}
const std::string& Result::consumed_particle_hole_identity_sha256() const {
    return solver().consumed_particle_hole_identity_sha256();
}
double Result::correlation_energy_per_cell() const {
    if (!periodic_energy_per_cell())
        throw std::logic_error("Gaussian pair CCSD per-cell energy requires complete common basis, convergence and a certified particle-hole source");
    return real::finite(solver().final_snapshot().correlation_energy / static_cast<double>(state_->n_kpoints()));
}
double Result::total_energy_per_cell() const {
    const double correlation = correlation_energy_per_cell();
    return real::finite(state_->reference_energy_per_cell() + correlation);
}

Result Result::run_impl(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const PHProducer* input_ph,
    const Options& input_options, const Live& input_live,
    const Caps& input_caps, PeriodicGaussianPairCCSDCallback callback, void* callback_context) {
    const auto options = input_options; const auto live = input_live; const auto caps = input_caps;
    const std::optional<PHProducer> copied_ph = input_ph ? std::optional<PHProducer>(*input_ph) : std::nullopt;
    const auto* ph = copied_ph ? &*copied_ph : nullptr;
    if (ph && !ph->produce) throw std::invalid_argument("Gaussian pair CCSD split route requires a native particle-hole producer");
    const auto p = pair_ccsd_plan_impl(ref, basis, provider, warm, options, live, caps, ph);
    verify_pairs(ref, basis, provider, warm, p);
    if (p.domain_generated) for (U i=0;i<p.occupied_count;++i) for (U j=i;j<p.occupied_count;++j)
        verify_frame_payload(warm.pair(i,j).frame(),p.common_virtual_dimension);
    Result result; result.memory_ = p; result.state_ = ref.state_handle(); result.context_ = provider.context_handle();
    result.warmstart_ = warm.identity_sha256(); result.pairs_ = warm.pair_spaces_identity_sha256();
    result.basis_ = basis.identity_sha256(); result.provider_ = provider.identity_sha256();
    result.hf_ = provider.hf_reference_source_identity_sha256();
    const U o = p.occupied_count, n = p.common_virtual_dimension;
    auto& d = result.diagnostics_;
    d.split_bare_particle_hole = p.split_bare_particle_hole;
    d.minimum_singles_rank = n; d.all_singles_full_rank = true;
    d.all_pairs_full_rank = warm.diagnostics().all_pairs_full_rank;
    d.complete_common_finite_torus_basis = warm.diagnostics().complete_common_finite_torus_basis
        && o == mul(ref.state().n_kpoints(), ref.state().n_correlated_occupied())
        && n == mul(ref.state().n_kpoints(), ref.state().n_virtual());
    Events events{{}, callback, callback_context, p.progress_callback_upper_bound, &ref, &basis, &provider, &warm, &p, {}, {}};
    std::copy(result.warmstart_.begin(), result.warmstart_.end(), events.warm_sha.begin());
    std::copy(result.provider_.begin(), result.provider_.end(), events.provider_sha.begin());
    events.pin_basis();
    events.emit(PeriodicGaussianPairCCSDStage::Begin);
    result.singles_.reserve(static_cast<std::size_t>(o));
    Digest singles_sha("vibeqc.periodic.gaussian-pair-ccsd.singles"); singles_sha.u64(o);
    U rank_sum = 0;
    for (U i = 0; i < o; ++i) {
        auto generate = [&]() -> PeriodicGaussianPairPNOFrame {
            if (p.domain_generated) {
                const auto& e = warm.diagonal_generation_embedding(i);
                PeriodicGaussianEmbeddedPairPNOLiveInventory ilive;
                ilive.fixed_backend_margin_bytes_per_worker = live.fixed_backend_margin_bytes_per_worker;
                ilive.other_live_numerical_bytes_per_worker = add(p.borrowed_particle_hole_additional_numerical_bytes,
                    add(subtract(p.borrowed_mp2_numerical_bytes, e.memory().output_numerical_bytes),
                    add(d.retained_singles_bytes, live.other_live_bytes_per_worker)));
                ilive.other_live_control_bytes_per_worker = subtract(p.control_storage_reservation_bytes,
                    p.embedded_singles_upper.control_storage_reservation_bytes);
                return PeriodicGaussianPairPNOFrame(make_periodic_gaussian_embedded_pair_pnos(ref, basis, provider,
                    e, i, i, embedded_options(options), ilive, caps.embedded_singles));
            }
            PeriodicGaussianPairPNOLiveInventory ilive;
            ilive.fixed_backend_margin_bytes_per_worker = live.fixed_backend_margin_bytes_per_worker;
            ilive.other_live_bytes_per_worker = add(p.borrowed_particle_hole_additional_numerical_bytes, add(p.borrowed_mp2_numerical_bytes,
                add(d.retained_singles_bytes, add(live.other_live_bytes_per_worker,
                subtract(p.control_storage_reservation_bytes, p.singles_upper.fixed_control_storage_bytes)))));
            return PeriodicGaussianPairPNOFrame(make_periodic_gaussian_pair_pnos(ref, basis, provider, i, i,
                options.singles, ilive, caps.singles));
        };
        auto singles = generate();
        const auto frame = singles.view();
        if (p.domain_generated) verify_frame_payload(frame,n);
        const U r = frame.retained_dimension();
        rank_sum = add(rank_sum, r);
        d.singles_generation_dimension_sum = add(d.singles_generation_dimension_sum, frame.generation_dimension());
        d.retained_singles_bytes = add(d.retained_singles_bytes, frame.retained_numerical_bytes());
        if(frame.is_embedded()) d.retained_singles_generation_coefficient_bytes = add(
            d.retained_singles_generation_coefficient_bytes,mul(8,mul(frame.generation_dimension(),r)));
        d.completed_integral_calls = add(d.completed_integral_calls, frame.generation_diagnostics().completed_integral_calls);
        d.minimum_singles_rank = std::min(d.minimum_singles_rank, r);
        d.maximum_singles_rank = std::max(d.maximum_singles_rank, r);
        if (!r) ++d.zero_rank_singles;
        d.all_singles_full_rank = d.all_singles_full_rank && r == n;
        if (d.retained_singles_bytes > p.singles_output_upper_bytes || rank_sum > mul(o, n))
            throw std::logic_error("Gaussian pair CCSD singles payload exceeds admitted upper bound");
        singles_sha.u64(i); singles_sha.string(frame.identity_sha256());
        result.singles_.push_back(std::move(singles)); ++d.completed_singles;
        events.event.completed_singles = d.completed_singles; events.event.occupied_i = i; events.event.singles_rank = r;
        events.emit(PeriodicGaussianPairCCSDStage::SinglesComplete);
    }
    if (d.singles_generation_dimension_sum != p.singles_generation_dimension_sum)
        throw std::logic_error("Gaussian pair CCSD singles generation census differs");
    result.singles_sha_ = singles_sha.finish();
    std::vector<double> zero_singles(static_cast<std::size_t>(rank_sum), 0.0);
    d.zero_initial_singles_bytes = mul(8, rank_sum);
    std::vector<SinglesView> singles_views; singles_views.reserve(static_cast<std::size_t>(o));
    std::vector<PairView> pair_views; pair_views.reserve(static_cast<std::size_t>(p.pair_count));
    U offset = 0;
    for (const auto& singles : result.singles_) {
        const auto frame = singles.view();
        const U r = frame.retained_dimension();
        const auto& c = frame.coefficients();
        singles_views.push_back({r, {c.data(), c.size()},
            {r ? zero_singles.data() + offset : nullptr, static_cast<std::size_t>(r)}});
        offset += r;
    }
    for (U i = 0; i < o; ++i) for (U j = i; j < o; ++j) {
        const auto& pair = warm.pair(i, j); const auto& c = pair.coefficients();
        const auto t = warm.solver().stored_amplitudes_view(i, j);
        pair_views.push_back({pair.memory().retained_dimension, {c.data(), c.size()}, {t.data, t.element_count}});
    }
    BoundedRestrictedPairCCSDSolverInput input;
    input.initial = {o, n, singles_views.data(), singles_views.size(), pair_views.data(), pair_views.size()};
    input.f_oo = {basis.f_oo_data(), static_cast<std::size_t>(mul(o, o))};
    input.f_vv = {basis.f_vv_data(), static_cast<std::size_t>(mul(n, n))};
    input.f_ov = {basis.f_ov_data(), static_cast<std::size_t>(mul(o, n))};
    const auto inv = solver_inventory(p, live, rank_sum, p.solver_upper.control_storage_reservation_bytes,
        p.solver_upper.owned_snapshot_table_bytes, p.solver_upper.borrowed_input_table_bytes,
        d.retained_singles_generation_coefficient_bytes);
    const auto provider_callback = provider.provider().integral_provider(basis);
    const auto exact = ph
        ? plan_bounded_restricted_pair_ccsd_solver(input, options.solver, inv, caps.solver, *ph,
            provider_callback.retained_numerical_bytes, provider_callback.maximum_transient_numerical_bytes)
        : plan_bounded_restricted_pair_ccsd_solver(input, options.solver, inv, caps.solver,
            provider_callback.retained_numerical_bytes, provider_callback.maximum_transient_numerical_bytes);
    if (exact.per_replica_inventoried_bytes > p.per_worker_inventoried_bytes
        || exact.peak_owned_numerical_bytes > p.solver_upper.peak_owned_numerical_bytes
        || exact.work_units_upper_bound > p.solver_upper.work_units_upper_bound)
        throw std::logic_error("Gaussian pair CCSD exact solver exceeds enclosing admission");
    if (ph) result.solver_.emplace(bounded_restricted_pair_ccsd_solve(input, provider_callback, *ph,
        options.solver, inv, caps.solver, &Events::solver, &events));
    else result.solver_.emplace(bounded_restricted_pair_ccsd_solve(input, provider_callback,
        options.solver, inv, caps.solver, &Events::solver, &events));
    const auto& final_snapshot = result.solver_->final_snapshot();
    d.completed_particle_hole_calls = final_snapshot.particle_hole_calls;
    d.completed_particle_hole_visits = final_snapshot.particle_hole_visits;
    d.charged_particle_hole_work_units = final_snapshot.charged_particle_hole_work_units;
    d.completed_integral_calls = add(d.completed_integral_calls, result.solver_->final_snapshot().integral_calls);
    if (d.completed_integral_calls > p.integral_calls_upper_bound)
        throw std::logic_error("Gaussian pair CCSD integral census exceeds enclosing admission");
    Digest identity("vibeqc.periodic.gaussian-pair-ccsd.identity");
    for (const auto* s : std::array<const std::string*,10>{&ref.state().state_identity_sha256(), &ref.dimensions().allocation_identity,
        &result.basis_, &result.provider_, &result.hf_, &result.warmstart_, &result.pairs_, &result.singles_sha_,
        &result.solver_->input_identity_sha256(),
        &result.solver_->payload_sha256()}) identity.string(*s);
    identity.string(result.context_->source_context_identity_sha256());
    options_wire(identity, options); identity.u64(d.complete_common_finite_torus_basis);
    identity.u64(d.all_singles_full_rank); identity.u64(d.all_pairs_full_rank); identity.string(policy);
    if (p.domain_generated) {
        identity.string("embedded-original-diagonal-generation;independent-singles-cut;no-common-density-fallback");
        identity.u64(d.singles_generation_dimension_sum);
        identity.u64(d.retained_singles_generation_coefficient_bytes);
        const auto& e = options.singles_embedding;
        for (double x : {e.maximum_diagonal_exchange_projection_norm, e.maximum_fock_symmetry_projection_norm,
            e.maximum_exported_gram_error, e.maximum_exported_fock_error, e.maximum_exported_subspace_error}) identity.real(x);
    }
    if (ph) {
        sha(result.solver_->split_execution_identity_sha256());
        sha(result.solver_->consumed_particle_hole_identity_sha256());
        identity.string("generic-native-bare-PH-replacement;not-entire-full-common-arithmetic;uncertified-PH-source");
        identity.string(result.solver_->split_execution_identity_sha256());
        identity.string(result.solver_->consumed_particle_hole_identity_sha256());
    }
    result.identity_ = identity.finish();
    events.emit(PeriodicGaussianPairCCSDStage::Finished);
    d.completed_progress_callbacks = events.event.callback_count;
    return result;
}

Result run_periodic_gaussian_pair_ccsd(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const Options& options, const Live& live, const Caps& caps,
    PeriodicGaussianPairCCSDCallback callback, void* context) {
    return Result::run_impl(ref, basis, provider, warm, nullptr, options, live, caps, callback, context);
}
Result run_periodic_gaussian_pair_ccsd(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis, const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm, const PHProducer& ph,
    const Options& options, const Live& live, const Caps& caps,
    PeriodicGaussianPairCCSDCallback callback, void* context) {
    return Result::run_impl(ref, basis, provider, warm, &ph, options, live, caps, callback, context);
}
} // namespace vibeqc
