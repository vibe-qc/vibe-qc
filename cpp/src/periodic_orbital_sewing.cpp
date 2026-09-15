#include "vibeqc/periodic_orbital_sewing.hpp"

#include <algorithm>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Z = std::complex<double>;
using State = PeriodicRestrictedMeanFieldState;
using Plan = PeriodicOrbitalSewingPlan;
using Options = PeriodicOrbitalSewingOptions;
using Inventory = PeriodicOrbitalSewingInventory;
using Caps = PeriodicOrbitalSewingCaps;
using Subspace = PeriodicOrbitalSubspace;
using SubDiagnostics = PeriodicOrbitalSubspaceSewingDiagnostics;
constexpr double TwoPi = 6.283185307179586476925286766559005768;
constexpr U ScalarNumericalBytes = 1024;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a)
        throw std::overflow_error("Orbital sewing count overflow");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a)
        throw std::overflow_error("Orbital sewing count overflow");
    return a*b;
}
void cap(U value, U bound, const char* message) { if (value > bound) invalid(message); }
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("Orbital sewing nonfinite numerical value");
    return x;
}
Z finite(Z x) { finite(x.real()); finite(x.imag()); return x; }
double magnitude(Z x) { return finite(std::abs(finite(x))); }
void update(double& residual, Z difference, double bound, const char* message) {
    const double r = magnitude(difference);
    residual = std::max(residual,r);
    if (r > bound) invalid(message);
}
void real_push(double x, double& s, double& c) {
    finite(x);
    const double t = finite(s+x);
    c = finite(c+(std::abs(s) >= std::abs(x) ? (s-t)+x : (x-t)+s));
    s = t;
}
struct Sum {
    double r = 0, i = 0, rc = 0, ic = 0;
    void push(Z x) { real_push(x.real(),r,rc); real_push(x.imag(),i,ic); }
    Z value() const { return {finite(r+rc),finite(i+ic)}; }
};
U ordinal(Subspace subspace) {
    const U x = static_cast<U>(subspace);
    if (x >= 3) invalid("Orbital sewing invalid subspace selector");
    return x;
}
const std::vector<std::uint8_t>& mask(const State& state, U k, U x) {
    if (x == 0) return state.frozen_core_mask(k);
    if (x == 1) return state.correlated_occupied_mask(k);
    if (x == 2) return state.virtual_mask(k);
    invalid("Orbital sewing invalid subspace selector");
}

void environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    invalid("Orbital sewing requires strict floating point compilation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || sizeof(double) != 8 ||
        std::numeric_limits<double>::digits != 53 ||
        std::numeric_limits<double>::max_exponent != 1024 || FLT_RADIX != 2 ||
        sizeof(Z) != 16 || std::fegetround() != FE_TONEAREST)
        invalid("Orbital sewing requires binary64 round-to-nearest arithmetic");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1, zero = 0;
    volatile double normal = std::numeric_limits<double>::min(), half = 0.5;
    if (!(tiny > 0) || tiny*one != tiny || tiny+zero != tiny ||
        std::fma(tiny,one,zero) != tiny || std::nextafter(0.0,1.0) != tiny || normal*half == 0)
        invalid("Orbital sewing requires gradual underflow");
}
void controls(const Options& options, const Inventory& inventory) {
    environment();
    const double bounds[] = {
        options.maximum_reciprocal_lattice_residual,
        options.maximum_source_metric_residual,options.maximum_target_metric_residual,
        options.maximum_transported_metric_residual,options.maximum_unitarity_residual,
        options.maximum_reconstruction_residual,options.maximum_cross_subspace_overlap,
        options.maximum_roothaan_residual,options.maximum_energy_intertwining_residual};
    for (double bound : bounds)
        if (!std::isfinite(bound) || bound < 0)
            invalid("Orbital sewing requires explicit finite nonnegative audit controls");
    if (options.maximum_reciprocal_lattice_residual >= 1 ||
        options.maximum_source_metric_residual >= 1 || options.maximum_target_metric_residual >= 1 ||
        options.maximum_transported_metric_residual >= 1 || options.maximum_unitarity_residual >= 1 ||
        options.maximum_cross_subspace_overlap >= 1)
        invalid("Orbital sewing dimensionless audit controls must be less than one");
    if (!inventory.numerical_replicas || !inventory.backend_margin_bytes_per_replica)
        invalid("Orbital sewing requires positive replica count and backend margin");
}

// V1's resident_bytes already counts every bulk numerical field of EVERY k.
// Add all per-k descriptor roles without reading any per-k floating payload.
// Padding is an explicit conservative ABI control allowance, not heap proof.
U state_controls(const State& state) {
    constexpr U PointDescriptors = 3*sizeof(PeriodicMeanFieldComplexMatrix)+
        2*sizeof(Eigen::VectorXd)+3*sizeof(std::vector<std::uint8_t>)+
        sizeof(Eigen::Vector3d)+sizeof(double)+128;
    U bytes = add(sizeof(State)+64,mul(state.n_kpoints(),PointDescriptors));
    bytes = add(bytes,add(state.calculation_identity().size(),1));
    bytes = add(bytes,add(state.numerical_payload_sha256().size(),1));
    return add(bytes,add(state.state_identity_sha256().size(),1));
}
U macro_controls() {
    return 8192+sizeof(PeriodicOrbitalSewingResult)+2*sizeof(Plan)+
        sizeof(Options)+sizeof(Inventory)+sizeof(Caps)+sizeof(RegularKMesh)+
        3*sizeof(std::vector<Z>)+2*sizeof(std::vector<U>)+sizeof(PeriodicAOBlochTransportResult)+
        2*sizeof(PeriodicAOBlochTransportInventory)+sizeof(PeriodicAOBlochTransportPlan);
}
PeriodicAOBlochTransportInventory transport_inventory(const Plan& p, const Inventory& live) {
    PeriodicAOBlochTransportInventory out;
    out.numerical_replicas = live.numerical_replicas;
    out.external_node_bytes = live.external_node_bytes;
    out.backend_margin_bytes_per_replica = live.backend_margin_bytes_per_replica;
    out.other_live_numerical_bytes_per_replica = add(live.other_live_numerical_bytes_per_replica,
        add(p.state_resident_numerical_bytes,add(p.retained_sewing_bytes,
            add(p.index_workspace_bytes,p.fixed_scalar_numerical_bytes))));
    out.other_live_control_bytes_per_replica = add(live.other_live_control_bytes_per_replica,
        add(p.state_control_storage_bytes,macro_controls()));
    return out;
}
void point_shape(const State& state, U k) {
    const auto n = static_cast<Eigen::Index>(state.n_basis());
    const auto e = static_cast<Eigen::Index>(state.n_effective_orbitals());
    if (state.overlap(k).rows() != n || state.overlap(k).cols() != n ||
        state.fock(k).rows() != n || state.fock(k).cols() != n ||
        state.coefficients(k).rows() != n || state.coefficients(k).cols() != e ||
        state.orbital_energies(k).size() != e || state.occupations(k).size() != e)
        invalid("Orbital sewing selected state matrix extent mismatch");
    for (U x = 0; x < 3; ++x)
        if (mask(state,k,x).size() != state.n_effective_orbitals())
            invalid("Orbital sewing selected state mask extent mismatch");
}
void state_metadata(const std::shared_ptr<const State>& state, const BasisSet& basis,
                    const PeriodicSystem& system, U source, const Options& options,
                    const Caps& caps, Plan& p) {
    if (!state) invalid("Orbital sewing requires a live immutable mean-field state");
    if (state->contract_version() != kPeriodicRestrictedMeanFieldStateContractVersion ||
        !state->converged() ||
        state->normalization() != PeriodicMeanFieldNormalizationConvention::UnnormalizedAoBlochSumsUniformFullBzWeights ||
        state->reference_kind() != PeriodicMeanFieldReferenceKind::RestrictedHartreeFock)
        invalid("Orbital sewing unsupported immutable state contract");
    p.n_basis = state->n_basis(); p.n_effective_orbitals = state->n_effective_orbitals();
    p.n_kpoints = state->n_kpoints(); p.source_index = source;
    cap(p.n_kpoints,caps.maximum_kpoints,"Orbital sewing k-point cap");
    cap(p.n_basis,caps.maximum_basis_functions,"Orbital sewing basis-function cap");
    cap(p.n_effective_orbitals,caps.maximum_effective_orbitals,"Orbital sewing effective-orbital cap");
    if (!p.n_basis || !p.n_effective_orbitals || p.n_effective_orbitals > p.n_basis ||
        p.n_basis > static_cast<U>(std::numeric_limits<Eigen::Index>::max()) ||
        p.n_basis != basis.nbasis() || state->periodic_dimension() != system.dim)
        invalid("Orbital sewing state/basis/system dimension metadata mismatch");
    const RegularKMesh mesh(state->mesh(),state->is_shift());
    if (mesh.size() != p.n_kpoints || source >= p.n_kpoints)
        invalid("Orbital sewing source/whole-mesh extent mismatch");
    p.subspace_ranks = {state->n_frozen_core(),state->n_correlated_occupied(),state->n_virtual()};
    U sum = 0;
    for (U rank : p.subspace_ranks) {
        cap(rank,caps.maximum_subspace_rank,"Orbital sewing subspace-rank cap");
        sum = add(sum,rank);
        p.maximum_rank = std::max(p.maximum_rank,rank);
        p.retained_sewing_bytes = add(p.retained_sewing_bytes,mul(16,mul(rank,rank)));
        if (rank) ++p.transport_calls;
    }
    if (sum != p.n_effective_orbitals || !p.maximum_rank)
        invalid("Orbital sewing state subspace-rank metadata mismatch");
    cap(p.transport_calls,caps.maximum_transport_calls,"Orbital sewing transport-call cap");
    p.full_ao_scope = p.n_effective_orbitals == p.n_basis;
    if (options.request_full_ao_scope && !p.full_ao_scope)
        invalid("Orbital sewing full-AO scope requested for a retained-only state");
    p.state_resident_numerical_bytes = state->resident_bytes();
    if (p.state_resident_numerical_bytes != estimate_periodic_restricted_mean_field_resident_bytes(
            state->mesh(),p.n_basis,p.n_effective_orbitals))
        invalid("Orbital sewing immutable state resident inventory mismatch");
    p.state_control_storage_bytes = state_controls(*state);
    p.index_workspace_bytes = mul(16,p.maximum_rank);
    p.maximum_packed_source_bytes = mul(16,mul(p.n_basis,p.maximum_rank));
    p.column_workspace_bytes = mul(32,p.n_basis);
    p.fixed_scalar_numerical_bytes = ScalarNumericalBytes;
    point_shape(*state,source);
}

void physical_metadata_audit(const State& state, const PeriodicSystem& system,
                            const Options& o, PeriodicOrbitalSewingDiagnostics& d) {
    // A and B have Cartesian lattice vectors as columns. No basis-content
    // authentication is inferred from this numerical reciprocal-cell check.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Sum value;
            for (int k = 0; k < 3; ++k)
                value.push(finite(system.lattice(k,i))*finite(state.reciprocal_lattice()(k,j)));
            update(d.reciprocal_lattice_residual,value.value()/TwoPi-(i==j ? 1.0 : 0.0),
                   o.maximum_reciprocal_lattice_residual,
                   "Orbital sewing state/system reciprocal lattice audit");
        }
}
void point_payload(const State& state, U k, const Plan& p) {
    std::array<U,3> ranks = {0,0,0};
    for (U band = 0; band < p.n_effective_orbitals; ++band) {
        U count = 0;
        for (U x = 0; x < 3; ++x) {
            const auto bit = mask(state,k,x)[band];
            if (bit > 1) invalid("Orbital sewing masks must be binary");
            ranks[x] += bit; count += bit;
        }
        if (count != 1) invalid("Orbital sewing masks must be disjoint and exhaustive");
        finite(state.orbital_energies(k)[band]);
        const double expected = mask(state,k,2)[band] ? 0.0 : 2.0;
        if (finite(state.occupations(k)[band]) != expected)
            invalid("Orbital sewing mask/occupation mismatch");
        for (U mu = 0; mu < p.n_basis; ++mu) finite(state.coefficients(k)(mu,band));
    }
    if (ranks != p.subspace_ranks) invalid("Orbital sewing exact mask ranks differ from state metadata");
    for (U mu = 0; mu < p.n_basis; ++mu)
        for (U nu = 0; nu < p.n_basis; ++nu) {
            finite(state.overlap(k)(mu,nu)); finite(state.fock(k)(mu,nu));
        }
}
void indices(const State& state, U k, U x, U expected, U* out) {
    U count = 0;
    for (U band = 0; band < state.n_effective_orbitals(); ++band)
        if (mask(state,k,x)[band]) {
            if (count >= expected) invalid("Orbital sewing mask index overrun");
            out[count++] = band;
        }
    if (count != expected) invalid("Orbital sewing mask index underrun");
}

// Source and target canonical metrics/energies are audited at this leaf's
// explicit tolerances, not merely trusted at the older state-v1 tolerance.
void original_audit(const State& state, U k, const U* bands, U rank, Z* sc, Z* fc,
                    double metric_bound, const Options& options,
                    double& metric_residual, double& roothaan_residual) {
    const U n = state.n_basis(), e = state.n_effective_orbitals();
    const auto& c = state.coefficients(k);
    for (U j = 0; j < rank; ++j) {
        const U band = bands[j];
        for (U mu = 0; mu < n; ++mu) {
            Sum s, f;
            for (U nu = 0; nu < n; ++nu) {
                s.push(state.overlap(k)(mu,nu)*c(nu,band));
                f.push(state.fock(k)(mu,nu)*c(nu,band));
            }
            sc[mu] = s.value(); fc[mu] = f.value();
            update(roothaan_residual,finite(fc[mu]-finite(sc[mu]*state.orbital_energies(k)[band])),
                   options.maximum_roothaan_residual,
                   "Orbital sewing original source/target Roothaan audit");
        }
        // Include ALL retained bands: source/target cross-mask metric errors
        // cannot hide behind a successful within-mask normalization check.
        for (U b = 0; b < e; ++b) {
            Sum s;
            for (U mu = 0; mu < n; ++mu) s.push(std::conj(c(mu,b))*sc[mu]);
            update(metric_residual,s.value()-(b==band ? 1.0 : 0.0),metric_bound,
                   "Orbital sewing source/target metric audit");
        }
    }
}
void transported_audit(const State& state, const Plan& plan, U x,
                       const U* source_bands, const U* target_bands,
                       const Z* t, Z* u, Z* st, Z* ft,
                       const Options& options, SubDiagnostics& d) {
    const U n = plan.n_basis, e = plan.n_effective_orbitals, r = plan.subspace_ranks[x];
    const U source = plan.source_index, target = plan.target_index;
    const auto& ct = state.coefficients(target);
    for (U j = 0; j < r; ++j) {
        for (U mu = 0; mu < n; ++mu) {
            Sum s, f;
            for (U nu = 0; nu < n; ++nu) {
                s.push(state.overlap(target)(mu,nu)*t[nu*r+j]);
                f.push(state.fock(target)(mu,nu)*t[nu*r+j]);
            }
            st[mu] = s.value(); ft[mu] = f.value();
            update(d.roothaan_residual,
                   finite(ft[mu]-finite(st[mu]*state.orbital_energies(source)[source_bands[j]])),
                   options.maximum_roothaan_residual,"Orbital sewing transported target Roothaan audit");
        }
        for (U a = 0; a < r; ++a) {
            Sum s;
            for (U mu = 0; mu < n; ++mu) s.push(std::conj(t[mu*r+a])*st[mu]);
            update(d.transported_metric_residual,s.value()-(a==j ? 1.0 : 0.0),
                   options.maximum_transported_metric_residual,"Orbital sewing transported metric audit");
        }
        // Casassa2006 Sec.2 Eq.3 permits a full subspace matrix. Here its
        // numerically determined coefficients are C_t^dagger S_t Q_g C_s.
        U target_ordinal = 0;
        for (U band = 0; band < e; ++band) {
            Sum s;
            for (U mu = 0; mu < n; ++mu) s.push(std::conj(ct(mu,band))*st[mu]);
            const Z value = s.value();
            if (mask(state,target,x)[band]) {
                if (target_ordinal >= r || target_bands[target_ordinal] != band)
                    invalid("Orbital sewing target mask ordinal mismatch");
                u[target_ordinal*r+j] = value;
                ++target_ordinal;
            } else {
                update(d.cross_subspace_overlap,value,options.maximum_cross_subspace_overlap,
                       "Orbital sewing cross-subspace mask leakage");
            }
        }
        if (target_ordinal != r) invalid("Orbital sewing target mask ordinal extent");
    }
    for (U a = 0; a < r; ++a)
        for (U b = 0; b < r; ++b) {
            Sum left, right;
            for (U k = 0; k < r; ++k) {
                left.push(std::conj(u[k*r+a])*u[k*r+b]);
                right.push(u[a*r+k]*std::conj(u[b*r+k]));
            }
            const double delta = a==b ? 1.0 : 0.0;
            update(d.unitarity_residual,left.value()-delta,options.maximum_unitarity_residual,
                   "Orbital sewing subspace unitarity audit");
            update(d.unitarity_residual,right.value()-delta,options.maximum_unitarity_residual,
                   "Orbital sewing subspace unitarity audit");
            // Dovesi1986 Eqs.40-41 plus the canonical Roothaan equations:
            // eps_t U = U eps_s. No energy-based grouping or phase repair.
            const double et = state.orbital_energies(target)[target_bands[a]];
            const double es = state.orbital_energies(source)[source_bands[b]];
            const Z difference = finite(finite(et*u[a*r+b])-finite(u[a*r+b]*es));
            update(d.energy_intertwining_residual,difference,
                   options.maximum_energy_intertwining_residual,"Orbital sewing energy-intertwining audit");
        }
    for (U mu = 0; mu < n; ++mu)
        for (U j = 0; j < r; ++j) {
            Sum rebuilt;
            for (U a = 0; a < r; ++a) rebuilt.push(ct(mu,target_bands[a])*u[a*r+j]);
            update(d.reconstruction_residual,finite(rebuilt.value()-t[mu*r+j]),
                   options.maximum_reconstruction_residual,"Orbital sewing subspace reconstruction audit");
        }
}

} // namespace

Plan plan_periodic_orbital_sewing(
    const std::shared_ptr<const State>& state, const BasisSet& basis,
    const PeriodicSystem& system, const SymmetryOp& op, U source, bool tr,
    const Options& options, const Inventory& live, const Caps& caps) {
    controls(options,live);
    Plan p;
    state_metadata(state,basis,system,source,options,caps,p);
    p.time_reversal = tr;
    const RegularKMesh mesh(state->mesh(),state->is_shift());
    // This is count-only: it checks all shell/contraction metadata and exact
    // whole-mesh action, but reads no primitive, coefficient or S/F payload.
    p.transport_upper = plan_periodic_ao_bloch_transport(basis,system,op,mesh,source,tr,p.maximum_rank,
        options.ao_transport,transport_inventory(p,live),caps.ao_transport);
    p.target_index = p.transport_upper.target_index;
    p.target_doubled_address = p.transport_upper.target_doubled_address;
    p.reciprocal_wrap = p.transport_upper.reciprocal_wrap;
    point_shape(*state,p.target_index);
    p.borrowed_basis_numeric_bytes = p.transport_upper.borrowed_basis_numeric_bytes;
    p.borrowed_geometry_numeric_bytes = p.transport_upper.borrowed_geometry_numeric_bytes;
    p.borrowed_numerical_bytes = add(p.state_resident_numerical_bytes,
        add(p.borrowed_basis_numeric_bytes,p.borrowed_geometry_numeric_bytes));
    const U retained_and_indices = add(p.retained_sewing_bytes,
        add(p.index_workspace_bytes,p.fixed_scalar_numerical_bytes));
    p.transport_phase_owned_upper_bound = add(retained_and_indices,
        add(p.maximum_packed_source_bytes,p.transport_upper.peak_owned_numerical_bytes));
    // Packed input is destroyed before these two columns are allocated;
    // transport scratch is destroyed; its coefficients and maps survive.
    p.audit_phase_owned_upper_bound = add(retained_and_indices,
        add(p.transport_upper.retained_mapping_bytes,
            add(p.maximum_packed_source_bytes,p.column_workspace_bytes)));
    p.peak_owned_numerical_bytes = std::max(p.transport_phase_owned_upper_bound,p.audit_phase_owned_upper_bound);
    p.control_storage_bytes = p.transport_upper.control_storage_bytes;
    // Includes the macro's count-only upper leaf plan and each actual call's
    // own planner. The extra leaf allowance is conservative, not elapsed work.
    p.transport_work_units_upper_bound = mul(add(p.transport_calls,1),p.transport_upper.work_units_upper_bound);
    U cubes = 0;
    for (U r : p.subspace_ranks) cubes = add(cubes,mul(mul(r,r),r));
    const U n = p.n_basis, e = p.n_effective_orbitals;
    U terms = add(mul(16,mul(mul(n,n),e)),mul(16,mul(n,mul(e,e))));
    terms = add(terms,mul(16,cubes));
    terms = add(terms,add(mul(32,mul(n,n)),add(mul(64,mul(n,e)),mul(128,e))));
    p.validation_work_units_upper_bound = add(65536,mul(256,add(1024,terms)));
    p.work_units_upper_bound = add(p.transport_work_units_upper_bound,p.validation_work_units_upper_bound);
    p.per_replica_inventoried_bytes = add(p.peak_owned_numerical_bytes,add(p.borrowed_numerical_bytes,
        add(live.other_live_numerical_bytes_per_replica,add(p.control_storage_bytes,live.backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes = add(live.external_node_bytes,
        mul(live.numerical_replicas,p.per_replica_inventoried_bytes));
    cap(p.borrowed_numerical_bytes,caps.maximum_borrowed_numerical_bytes,"Orbital sewing borrowed numerical byte cap");
    cap(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Orbital sewing owned numerical byte cap");
    cap(p.control_storage_bytes,caps.maximum_control_storage_bytes,"Orbital sewing control storage cap");
    cap(p.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,"Orbital sewing per-replica inventoried byte cap");
    cap(p.required_node_inventoried_bytes,caps.maximum_node_inventoried_bytes,"Orbital sewing node inventoried byte cap");
    cap(p.work_units_upper_bound,caps.maximum_work_units,"Orbital sewing work cap");
    if (p.peak_owned_numerical_bytes > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Orbital sewing allocation address extent");
    return p;
}

PeriodicOrbitalSewingResult make_periodic_orbital_sewing(
    std::shared_ptr<const State> state, const BasisSet& basis, const PeriodicSystem& system,
    const SymmetryOp& op, U source, bool tr, const Options& options,
    const Inventory& live, const Caps& caps) {
    const Plan p = plan_periodic_orbital_sewing(state,basis,system,op,source,tr,options,live,caps);
    PeriodicOrbitalSewingResult result;
    result.plan_ = p;
    // Only after COMPLETE count/shape/resource/work admission do payload
    // comparisons, primitive scans, allocations or transport calls begin.
    physical_metadata_audit(*state,system,options,result.diagnostics_);
    point_payload(*state,p.source_index,p);
    point_payload(*state,p.target_index,p);
    for (U x = 0; x < 3; ++x) {
        result.sewing_[x].resize(mul(p.subspace_ranks[x],p.subspace_ranks[x]));
        result.diagnostics_.subspaces[x].rank = p.subspace_ranks[x];
    }
    std::vector<U> source_bands(p.maximum_rank), target_bands(p.maximum_rank);
    const RegularKMesh mesh(state->mesh(),state->is_shift());
    for (U x = 0; x < 3; ++x) {
        const U r = p.subspace_ranks[x];
        if (!r) continue;
        indices(*state,p.source_index,x,r,source_bands.data());
        indices(*state,p.target_index,x,r,target_bands.data());
        // Lambda lifetime is important: input packing dies before audit
        // columns are allocated. No previous class's T/columns survive here.
        auto transported = [&]() {
            std::vector<Z> packed(mul(p.n_basis,r));
            for (U mu = 0; mu < p.n_basis; ++mu)
                for (U j = 0; j < r; ++j)
                    packed[mu*r+j] = state->coefficients(source)(mu,source_bands[j]);
            return apply_periodic_ao_bloch_operation(basis,system,op,mesh,source,tr,
                packed.data(),packed.size(),r,options.ao_transport,transport_inventory(p,live),caps.ao_transport);
        }();
        if (transported.memory().target_index != p.target_index ||
            transported.memory().target_doubled_address != p.target_doubled_address ||
            transported.memory().reciprocal_wrap != p.reciprocal_wrap)
            invalid("Orbital sewing transport changed its admitted exact k action");
        auto& d = result.diagnostics_.subspaces[x];
        d.transport = transported.diagnostics();
        ++result.diagnostics_.completed_transport_calls;
        std::vector<Z> columns(mul(2,p.n_basis));
        Z* sc = columns.data();
        Z* fc = columns.data()+p.n_basis;
        original_audit(*state,p.source_index,source_bands.data(),r,sc,fc,
            options.maximum_source_metric_residual,options,d.source_metric_residual,d.source_roothaan_residual);
        original_audit(*state,p.target_index,target_bands.data(),r,sc,fc,
            options.maximum_target_metric_residual,options,d.target_metric_residual,d.target_roothaan_residual);
        transported_audit(*state,p,x,source_bands.data(),target_bands.data(),transported.data(),
            result.sewing_[x].data(),sc,fc,options,d);
        result.diagnostics_.audited_source_columns = add(result.diagnostics_.audited_source_columns,r);
    }
    if (result.diagnostics_.completed_transport_calls != p.transport_calls ||
        result.diagnostics_.audited_source_columns != p.n_effective_orbitals)
        invalid("Orbital sewing incomplete subspace audit");
    result.state_ = std::move(state);
    return result;
}

const std::shared_ptr<const State>& PeriodicOrbitalSewingResult::state_handle() const {
    if (!state_) throw std::logic_error("Orbital sewing result is moved from or has no state owner");
    return state_;
}
const std::string& PeriodicOrbitalSewingResult::state_identity_sha256() const {
    return state_handle()->state_identity_sha256();
}
bool PeriodicOrbitalSewingResult::full_ao_scope_audited() const {
    (void)state_handle();
    return plan_.full_ao_scope;
}
const Z* PeriodicOrbitalSewingResult::sewing_data(Subspace subspace) const {
    (void)state_handle();
    const U x = ordinal(subspace), r = plan_.subspace_ranks[x];
    if (sewing_[x].size() != mul(r,r)) throw std::logic_error("Orbital sewing result storage extent");
    return sewing_[x].data();
}
Z PeriodicOrbitalSewingResult::element(Subspace subspace, U target, U source) const {
    const auto* data = sewing_data(subspace);
    const U r = plan_.subspace_ranks[ordinal(subspace)];
    if (target >= r || source >= r) throw std::out_of_range("Orbital sewing matrix ordinal");
    return data[target*r+source];
}
namespace {
U band_at(const State& state, U k, U x, U wanted) {
    U at = 0;
    for (U band = 0; band < state.n_effective_orbitals(); ++band)
        if (mask(state,k,x)[band] && at++ == wanted) return band;
    throw std::out_of_range("Orbital sewing subspace band ordinal");
}
} // namespace
U PeriodicOrbitalSewingResult::source_band(Subspace subspace, U wanted) const {
    return band_at(*state_handle(),plan_.source_index,ordinal(subspace),wanted);
}
U PeriodicOrbitalSewingResult::target_band(Subspace subspace, U wanted) const {
    return band_at(*state_handle(),plan_.target_index,ordinal(subspace),wanted);
}

} // namespace vibeqc
