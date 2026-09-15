#include "vibeqc/bounded_restricted_pair_mp2_solver.hpp"
#include "vibeqc/periodic_correlation_pair_residual.hpp"
#include "vibeqc/pair_natural_orbitals.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Input = BoundedRestrictedPairMP2SolverInput;
using Options = BoundedRestrictedPairMP2SolverOptions;
using Inventory = BoundedRestrictedPairMP2SolverInventory;
using Caps = BoundedRestrictedPairMP2SolverCaps;
using Plan = BoundedRestrictedPairMP2SolverMemoryPlan;
using Progress = BoundedRestrictedPairMP2SolverProgress;
using View = BoundedRestrictedPairMP2RealView;
using Pair = BoundedRestrictedPairMP2PairView;
static_assert(sizeof(double) == 8 && sizeof(std::size_t) == 8
    && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
    "pair MP2 requires IEEE binary64 and 64-bit extents");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "pair MP2 forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "pair MP2 requires binary64 evaluation"
#endif
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a) throw std::overflow_error("pair MP2 count sum overflows");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a) throw std::overflow_error("pair MP2 count product overflows");
    return a*b;
}
U triangular(U o) { return o%2 ? mul(o,add(o,1)/2) : mul(o/2,add(o,1)); }
U pair_index(U i, U j, U o) {
    if (i >= o || j >= o) throw std::out_of_range("pair MP2 occupied label out of range");
    if (i > j) std::swap(i,j);
    // o(o+1)/2-(o-i)(o-i+1)/2 gives row start without 2*o overflow.
    return add(triangular(o)-triangular(o-i),j-i);
}
void limit(U count, U cap, const char* message) {
    if (count > cap) throw std::length_error(message);
}
void extent(U bytes) {
    if (bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<U>::max()/8-16384)
        throw std::length_error("pair MP2 storage or SHA extent exceeded");
}
double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("pair MP2 arithmetic is non-finite");
    return value;
}
void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0.0) || tiny*one != tiny
        || tiny+zero != tiny || std::fma(tiny,one,zero) != tiny)
        throw std::invalid_argument("pair MP2 requires round-to-nearest and gradual underflow");
}
struct Sum {
    double value = 0, correction = 0;
    void include(double term) {
        finite(term);
        const auto next = finite(value+term);
        correction = finite(correction+(std::abs(value) >= std::abs(term)
            ? (value-next)+term : (term-next)+value));
        value = next;
    }
    double total() const { return finite(value+correction); }
};
void storage(const void* pointer, U count, U width, U alignment) {
    if (!count) return;
    const auto bytes = mul(count,width);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    extent(bytes);
    if (!pointer || address%alignment || bytes > std::numeric_limits<std::uintptr_t>::max()-address)
        throw std::invalid_argument("pair MP2 input pointer, alignment or extent is invalid");
}
void view(View v, U required) {
    if (v.element_count != required) throw std::invalid_argument("pair MP2 requires exact view extents");
    storage(v.data,required,8,alignof(double));
}
void scan(View v) {
    for (U k=0;k<v.element_count;++k)
        if (!std::isfinite(v.data[k])) throw std::invalid_argument("pair MP2 input must be finite");
}
double gap(double ea, double eb, double fii, double fjj, double floor) {
    // Same power-of-two, compensated four-term convention as both reused
    // semicanonical initializer and residual leaf. No shift or half sums.
    const std::array<double,4> values{ea,eb,-fii,-fjj};
    double largest = 0;
    for (double x:values) largest = std::max(largest,std::abs(x));
    int exponent = 0;
    if (largest) std::frexp(largest,&exponent);
    Sum total;
    for (double x:values) {
        const auto scaled = std::scalbn(x,-exponent);
        if (std::scalbn(scaled,exponent) != x)
            throw std::overflow_error("pair MP2 denominator scaling loses range");
        total.include(scaled);
    }
    const auto result = std::scalbn(total.total(),exponent);
    if (!std::isfinite(result) || result <= floor)
        throw std::invalid_argument("pair MP2 denominator must strictly exceed its positive floor");
    return result;
}
double average(double a, double b) {
    if (a == b) return a;
    const double largest = std::max(std::abs(a),std::abs(b));
    int exponent = 0;
    if (largest) std::frexp(largest,&exponent);
    const auto x = std::scalbn(a,-exponent), y = std::scalbn(b,-exponent);
    if (std::scalbn(x,exponent) != a || std::scalbn(y,exponent) != b)
        throw std::overflow_error("pair MP2 symmetric update scaling loses range");
    return finite(std::scalbn((x+y)*0.5,exponent));
}
U residual_products(U r, U s) {
    if (!r || !s) return 0;
    return add(add(mul(r,mul(s,s)),mul(mul(r,r),s)),mul(r,r));
}
constexpr U fixed_controls = 4096+sizeof(Input)+sizeof(Options)+sizeof(Inventory)+sizeof(Caps)
    +2*sizeof(Plan)+2*sizeof(Progress)+sizeof(PairResidualAccumulator)+sizeof(PairResidualResult)
    +sizeof(RestrictedPairMP2Result)+4*sizeof(std::vector<double>)+2*sizeof(detail::Sha256);

Plan base(U o, U n, U rank, U iterations, const Inventory& inventory) {
    if (!o || !n || rank > n || !iterations || !inventory.numerical_replicas
        || !inventory.fixed_backend_margin_bytes_per_replica)
        throw std::invalid_argument("pair MP2 dimensions, iterations, replicas or backend allowance invalid");
    Plan p;
    p.n_occupied=o; p.common_virtual_dimension=n; p.maximum_pair_rank=rank;
    p.maximum_iterations=iterations; p.pair_count=triangular(o);
    p.borrowed_pair_table_bytes=mul(sizeof(Pair),p.pair_count);
    p.retained_pair_record_bytes=mul(16,p.pair_count);
    p.borrowed_fock_bytes=mul(8,mul(o,o));
    p.fixed_control_storage_bytes=fixed_controls;
    p.numerical_replicas=inventory.numerical_replicas;
    p.external_node_bytes=inventory.external_node_bytes;
    p.target_evaluations_upper_bound=mul(iterations,p.pair_count);
    p.coupling_slots_upper_bound=mul(p.target_evaluations_upper_bound,mul(2,o-1));
    p.metadata_work_units=mul(512,add(add(p.pair_count,mul(p.pair_count,mul(2,o-1))),1));
    extent(p.borrowed_pair_table_bytes); extent(p.retained_pair_record_bytes); extent(p.borrowed_fock_bytes);
    return p;
}
void finish_plan(Plan& p, const Inventory& inventory, U rank_sum, U gram_products,
    U overlap_products, U residual_product_count, U source_scan_lanes) {
    const auto r2=mul(p.maximum_pair_rank,p.maximum_pair_rank), a=p.total_amplitude_elements;
    p.amplitude_snapshot_bytes=p.candidate_snapshot_bytes=mul(8,a);
    p.output_numerical_bytes=add(p.amplitude_snapshot_bytes,p.retained_pair_record_bytes);
    const auto snapshots=add(mul(2,p.amplitude_snapshot_bytes),p.retained_pair_record_bytes);
    p.initialization_phase_owned_bytes=add(snapshots,mul(8,r2));
    p.residual_workspace_bytes=mul(24,r2); // R/compensation plus r*R projection scratch.
    p.overlap_workspace_bytes=mul(8,r2);
    p.iteration_phase_owned_bytes=add(snapshots,add(p.residual_workspace_bytes,p.overlap_workspace_bytes));
    p.peak_owned_numerical_bytes=std::max(p.initialization_phase_owned_bytes,p.iteration_phase_owned_bytes);
    p.borrowed_pair_numeric_bytes=mul(8,add(a,add(rank_sum,mul(p.common_virtual_dimension,rank_sum))));
    const auto payload=add(p.borrowed_fock_bytes,p.borrowed_pair_numeric_bytes);
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(payload,
        add(p.borrowed_pair_table_bytes,add(p.fixed_control_storage_bytes,
        add(inventory.other_live_bytes_per_replica,inventory.fixed_backend_margin_bytes_per_replica)))));
    p.required_node_inventoried_bytes=add(p.external_node_bytes,mul(p.numerical_replicas,p.per_replica_inventoried_bytes));
    // Conservative abstract work: 512 units per scalar/metadata lane cover
    // finite/address/hash passes, stable sums/gaps, norms and event handling.
    // Reused leaf scans every source T/O per slot, included explicitly.
    const auto payload_lanes=payload/8;
    p.validation_work_units=mul(512,add(add(payload_lanes,gram_products),add(a,add(p.pair_count,1))));
    p.initialization_work_units=mul(512,add(add(a,rank_sum),add(p.pair_count,1)));
    const auto slots=p.coupling_slots_upper_bound/p.maximum_iterations;
    p.work_units_per_snapshot=add(p.metadata_work_units,mul(512,
        add(add(payload_lanes,mul(8,a)),add(add(overlap_products,residual_product_count),
        add(source_scan_lanes,add(p.pair_count,add(slots,1)))))));
    p.work_units_upper_bound=add(p.metadata_work_units,add(p.validation_work_units,
        add(p.initialization_work_units,mul(p.maximum_iterations,p.work_units_per_snapshot))));
    p.overlap_scalar_products_upper_bound=mul(p.maximum_iterations,overlap_products);
    p.residual_scalar_products_upper_bound=mul(p.maximum_iterations,residual_product_count);
    for (U x:{p.peak_owned_numerical_bytes,p.borrowed_pair_numeric_bytes,p.per_replica_inventoried_bytes}) extent(x);
    if (a > std::vector<double>().max_size()) throw std::length_error("pair MP2 amplitude vector extent exceeded");
}
void cap_plan(const Plan& p, const Caps& c) {
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"pair MP2 owned numerical cap exceeded");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"pair MP2 per-replica cap exceeded");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"pair MP2 node cap exceeded");
    limit(p.coupling_slots_upper_bound,c.maximum_coupling_slots,"pair MP2 coupling slot cap exceeded");
    limit(p.work_units_upper_bound,c.maximum_work_units,"pair MP2 work cap exceeded");
}
struct Digest {
    detail::Sha256 sha;
    explicit Digest(const char* domain) { text(domain); integer(1); }
    void integer(U value) {
        std::array<std::uint8_t,8> bytes{};
        for (U k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(value>>(56-8*k));
        sha.update(bytes.data(),bytes.size());
    }
    void text(const std::string& value) {
        integer(value.size()); sha.update(reinterpret_cast<const std::uint8_t*>(value.data()),value.size());
    }
    void number(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("pair MP2 hash input is non-finite");
        if (x == 0) x=0;
        U bits=0; std::memcpy(&bits,&x,8); integer(bits);
    }
};
std::string input_hash(const Input& in, const Options& options, const Inventory& inventory, const Caps& caps) {
    // Re-admit mutable table descriptors before dereferencing their payloads.
    const auto p=plan_bounded_restricted_pair_mp2_solver(in,options,inventory,caps);
    Digest h("vibeqc.bounded-restricted-pair-mp2.input");
    h.integer(p.n_occupied); h.integer(p.common_virtual_dimension); h.integer(p.pair_count);
    for (U k=0;k<in.occupied_fock.element_count;++k) h.number(in.occupied_fock.data[k]);
    for (U k=0;k<p.pair_count;++k) {
        const auto& pair=in.pairs[k]; h.integer(pair.rank);
        for (auto v:{pair.integrals,pair.virtual_energies,pair.coefficients}) {
            h.integer(v.element_count);
            for (U x=0;x<v.element_count;++x) h.number(v.data[x]);
        }
    }
    return h.sha.finish_hex();
}
double energy_term(double gab, double gba, double t, U weight, Progress& record) {
    if (t == 0 || (gab == 0 && gba == 0)) return 0;
    int eg=0,et=0;
    std::frexp(std::max(std::abs(gab),std::abs(gba)),&eg);
    const double st=std::frexp(t,&et), x=std::scalbn(gab,-eg), y=std::scalbn(gba,-eg);
    if (std::scalbn(x,eg) != gab || std::scalbn(y,eg) != gba)
        throw std::overflow_error("pair MP2 energy input scaling loses range");
    const auto scaled=(2*x-y)*st*static_cast<double>(weight);
    const auto value=finite(std::scalbn(scaled,eg+et));
    if (scaled != 0 && value == 0) ++record.arithmetic_underflow_count;
    return value;
}
} // namespace

Plan plan_bounded_restricted_pair_mp2_solver_upper(U o,U n,U r,U iterations,const Inventory& inventory) {
    auto p=base(o,n,r,iterations,inventory); p.uniform_rank_upper_bound=true;
    const auto r2=mul(r,r), slots=mul(p.pair_count,mul(2,o-1));
    p.total_amplitude_elements=mul(p.pair_count,r2);
    finish_plan(p,inventory,mul(p.pair_count,r),mul(p.pair_count,mul(n,r2)),
        mul(slots,mul(n,r2)),mul(slots,residual_products(r,r)),mul(slots,mul(2,r2)));
    return p;
}

Plan plan_bounded_restricted_pair_mp2_solver(const Input& in,const Options& options,
    const Inventory& inventory,const Caps& caps) {
    if (!caps.maximum_occupied_count || !caps.maximum_common_virtual_dimension || !caps.maximum_pair_count
        || !caps.maximum_owned_numerical_bytes || !caps.maximum_per_replica_inventoried_bytes
        || !caps.maximum_node_inventoried_bytes || !caps.maximum_work_units)
        throw std::invalid_argument("pair MP2 positive caps must be explicit");
    limit(in.n_occupied,caps.maximum_occupied_count,"pair MP2 occupied count cap exceeded");
    limit(in.common_virtual_dimension,caps.maximum_common_virtual_dimension,"pair MP2 common virtual cap exceeded");
    auto p=base(in.n_occupied,in.common_virtual_dimension,0,options.maximum_iterations,inventory);
    limit(p.pair_count,caps.maximum_pair_count,"pair MP2 pair table count cap exceeded");
    limit(p.metadata_work_units,caps.maximum_work_units,"pair MP2 metadata work cap exceeded");
    limit(p.coupling_slots_upper_bound,caps.maximum_coupling_slots,"pair MP2 coupling slot cap exceeded");
    if (in.pair_count != p.pair_count) throw std::invalid_argument("pair MP2 table must contain every unordered pair exactly once");
    storage(in.pairs,p.pair_count,sizeof(Pair),alignof(Pair));
    view(in.occupied_fock,mul(in.n_occupied,in.n_occupied));
    U rank_sum=0,gram=0,overlap=0,residual=0,scans=0;
    for (U k=0;k<p.pair_count;++k) {
        const auto& pair=in.pairs[k]; const auto r=pair.rank;
        if (r > in.common_virtual_dimension) throw std::invalid_argument("pair MP2 rank exceeds common virtual dimension");
        limit(r,caps.maximum_pair_rank,"pair MP2 pair rank cap exceeded");
        const auto r2=mul(r,r);
        view(pair.integrals,r2); view(pair.virtual_energies,r); view(pair.coefficients,mul(in.common_virtual_dimension,r));
        rank_sum=add(rank_sum,r); p.total_amplitude_elements=add(p.total_amplitude_elements,r2);
        gram=add(gram,mul(in.common_virtual_dimension,r2));
        p.maximum_pair_rank=std::max(p.maximum_pair_rank,r);
    }
    for (U i=0;i<in.n_occupied;++i) for (U j=i;j<in.n_occupied;++j) {
        const auto r=in.pairs[pair_index(i,j,in.n_occupied)].rank;
        for (U leg=0;leg<2;++leg) for (U k=0;k<in.n_occupied;++k) {
            if (k == (leg ? j : i)) continue;
            const auto s=in.pairs[pair_index(leg?i:k,leg?k:j,in.n_occupied)].rank;
            overlap=add(overlap,mul(in.common_virtual_dimension,mul(r,s)));
            residual=add(residual,residual_products(r,s));
            if (r) scans=add(scans,add(mul(s,s),mul(r,s)));
        }
    }
    finish_plan(p,inventory,rank_sum,gram,overlap,residual,scans); cap_plan(p,caps);
    return p;
}

std::uint64_t BoundedRestrictedPairMP2SolverResult::pair_rank(U i,U j) const {
    const auto index=pair_index(i,j,memory_.n_occupied);
    if (records_.size() != memory_.pair_count) throw std::logic_error("pair MP2 result is consumed");
    return records_[index].rank;
}
double BoundedRestrictedPairMP2SolverResult::amplitude(U i,U j,U a,U b) const {
    const auto r=pair_rank(i,j);
    if (a >= r || b >= r) throw std::out_of_range("pair MP2 amplitude index out of range");
    if (i > j) std::swap(a,b);
    const auto& record=records_[pair_index(i,j,memory_.n_occupied)];
    return amplitudes_.at(record.offset+a*r+b);
}

BoundedRestrictedPairMP2RealView BoundedRestrictedPairMP2SolverResult::stored_amplitudes_view(U i,U j) const {
    if (i>j) throw std::invalid_argument("pair MP2 contiguous amplitude borrow requires i<=j");
    (void) pair_rank(i,j);
    const auto& r=records_.at(pair_index(i,j,memory_.n_occupied));
    const auto count=mul(r.rank,r.rank);
    if (r.offset>amplitudes_.size() || count>amplitudes_.size()-r.offset)
        throw std::logic_error("pair MP2 amplitude owner is moved or incomplete");
    return {count ? amplitudes_.data()+r.offset : nullptr,static_cast<std::size_t>(count)};
}

BoundedRestrictedPairMP2SolverResult bounded_restricted_pair_mp2_solve(const Input& input,
    const Options& supplied_options,const Inventory& supplied_inventory,const Caps& supplied_caps,
    BoundedRestrictedPairMP2SolverCallback callback,void* callback_context) {
    const auto in=input; const auto options=supplied_options; const auto inventory=supplied_inventory; const auto caps=supplied_caps;
    const auto p=plan_bounded_restricted_pair_mp2_solver(in,options,inventory,caps);
    for (double x:{options.denominator_floor,options.residual_tolerance,options.energy_tolerance,
                   options.coefficient_orthogonality_tolerance})
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("pair MP2 tolerances and denominator floor must be positive finite");
    if (options.coefficient_orthogonality_tolerance >= 1.0)
        throw std::invalid_argument("pair MP2 coefficient orthogonality tolerance must be less than one");
    if (!std::isfinite(options.maximum_diagonal_update_antisymmetry_norm)
        || options.maximum_diagonal_update_antisymmetry_norm < 0)
        throw std::invalid_argument("pair MP2 diagonal update antisymmetry budget must be explicit finite nonnegative");
    environment(); scan(in.occupied_fock);
    const auto o=p.n_occupied,n=p.common_virtual_dimension;
    for (U i=0;i<o;++i) for (U j=0;j<i;++j)
        if (in.occupied_fock.data[i*o+j] != in.occupied_fock.data[j*o+i])
            throw std::invalid_argument("pair MP2 occupied Fock must be exactly symmetric");
    double minimum=std::numeric_limits<double>::infinity(),maximum=0;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) {
        const auto& pair=in.pairs[pair_index(i,j,o)]; const auto r=pair.rank;
        scan(pair.integrals); scan(pair.virtual_energies); scan(pair.coefficients);
        for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
            if (i == j && pair.integrals.data[a*r+b] != pair.integrals.data[b*r+a])
                throw std::invalid_argument("pair MP2 diagonal-pair G must be exactly symmetric");
            Sum s;
            for (U mu=0;mu<n;++mu) s.include(finite(pair.coefficients.data[mu*r+a]*pair.coefficients.data[mu*r+b]));
            if (std::abs(finite(s.total()-(a == b ? 1.0 : 0.0))) > options.coefficient_orthogonality_tolerance)
                throw std::invalid_argument("pair MP2 coefficient columns are not orthonormal");
            const auto delta=gap(pair.virtual_energies.data[a],pair.virtual_energies.data[b],
                in.occupied_fock.data[i*o+i],in.occupied_fock.data[j*o+j],options.denominator_floor);
            minimum=std::min(minimum,delta); maximum=std::max(maximum,delta);
        }
    }
    BoundedRestrictedPairMP2SolverResult result;
    static_assert(sizeof(BoundedRestrictedPairMP2SolverResult::PairRecord) == 16,"pair record inventory mismatch");
    result.memory_=p; result.minimum_denominator_=p.total_amplitude_elements ? minimum : 0;
    result.maximum_denominator_=maximum;
    result.input_identity_=input_hash(in,options,inventory,caps);
    result.records_.resize(p.pair_count); result.amplitudes_.resize(p.total_amplitude_elements);
    std::vector<double> candidate(p.total_amplitude_elements);
    U offset=0;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) {
        const auto index=pair_index(i,j,o); const auto& pair=in.pairs[index]; const auto r=pair.rank;
        result.records_[index]={r,offset};
        if (r) {
            const auto initialized=restricted_pair_semicanonical_mp2(pair.integrals.data,pair.integrals.element_count,
                pair.virtual_energies.data,pair.virtual_energies.element_count,r,
                in.occupied_fock.data[i*o+i],in.occupied_fock.data[j*o+j],options.denominator_floor,mul(8,mul(r,r)));
            std::copy(initialized.amplitudes.begin(),initialized.amplitudes.end(),result.amplitudes_.begin()+offset);
        }
        offset=add(offset,mul(r,r));
    }
    std::vector<double> overlap(p.overlap_workspace_bytes/8);
    Progress record;
    record.input_checks=1;
    record.charged_work_units=add(p.metadata_work_units,add(p.validation_work_units,p.initialization_work_units));
    double previous=0;
    for (U iteration=1;iteration<=options.maximum_iterations;++iteration) {
        environment();
        record.iteration=iteration; record.has_previous_energy=iteration>1;
        record.maximum_absolute_residual=record.residual_frobenius_norm=0;
        record.charged_work_units=add(record.charged_work_units,p.work_units_per_snapshot);
        Sum energy;
        for (U i=0;i<o;++i) for (U j=i;j<o;++j) {
            ++record.target_evaluations;
            const auto index=pair_index(i,j,o); const auto& pair=in.pairs[index];
            const auto r=pair.rank, start=result.records_[index].offset;
            if (!r) {
                ++record.zero_rank_targets;
                record.coupling_slots=add(record.coupling_slots,mul(2,o-1));
                continue;
            }
            PairResidualControls control;
            control.occupied_label_count=o; control.target_first=i; control.target_second=j;
            control.expected_coupling_count=mul(2,o-1); control.maximum_source_dimension=p.maximum_pair_rank;
            control.maximum_scalar_products=mul(control.expected_coupling_count,residual_products(r,p.maximum_pair_rank));
            control.denominator_floor=options.denominator_floor;
            auto accumulator=initialize_pair_residual(pair.integrals.data,pair.integrals.element_count,
                result.amplitudes_.data()+start,r*r,pair.virtual_energies.data,r,r,
                in.occupied_fock.data[i*o+i],in.occupied_fock.data[j*o+j],control,
                plan_pair_residual(r,p.maximum_pair_rank,control.expected_coupling_count).peak_owned_numerical_bytes);
            for (U leg=0;leg<2;++leg) for (U k=0;k<o;++k) {
                if (k == (leg ? j : i)) continue;
                const auto first=leg?i:k,second=leg?k:j,source_index=pair_index(first,second,o);
                const auto& source=in.pairs[source_index]; const auto s=source.rank;
                for (U a=0;a<r;++a) for (U b=0;b<s;++b) {
                    Sum sum;
                    for (U mu=0;mu<n;++mu) {
                        const auto left=pair.coefficients.data[mu*r+a],right=source.coefficients.data[mu*s+b];
                        const auto product=finite(left*right);
                        if (left != 0 && right != 0 && product == 0) ++record.arithmetic_underflow_count;
                        sum.include(product);
                    }
                    overlap[a*s+b]=sum.total();
                }
                PairResidualSource term;
                term.leg=leg ? PairResidualOccupiedLeg::Second : PairResidualOccupiedLeg::First;
                term.substituted_occupied=k; term.stored_first=std::min(first,second); term.stored_second=std::max(first,second);
                term.source_dimension=s; term.occupied_fock_coupling=in.occupied_fock.data[k*o+(leg?j:i)];
                term.amplitudes=s ? result.amplitudes_.data()+result.records_[source_index].offset : nullptr;
                term.amplitude_elements=s*s; term.target_source_overlap=s ? overlap.data() : nullptr; term.overlap_elements=r*s;
                accumulator.accumulate(term);
                ++record.coupling_slots;
                record.overlap_scalar_products=add(record.overlap_scalar_products,mul(n,mul(r,s)));
            }
            const auto residual=accumulator.finish(); const auto& diagnostics=residual.diagnostics();
            record.transposed_sources=add(record.transposed_sources,diagnostics.transposed_source_count);
            record.zero_rank_sources=add(record.zero_rank_sources,diagnostics.zero_rank_source_count);
            record.residual_scalar_products=add(record.residual_scalar_products,diagnostics.scalar_product_count);
            record.arithmetic_underflow_count=add(record.arithmetic_underflow_count,diagnostics.product_underflow_count);
            for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
                const auto ab=a*r+b;
                const auto value=residual.residual_data()[ab];
                candidate[start+ab]=value;
                record.maximum_absolute_residual=std::max(record.maximum_absolute_residual,std::abs(value));
                record.residual_frobenius_norm=finite(std::hypot(record.residual_frobenius_norm,value));
                energy.include(energy_term(pair.integrals.data[ab],pair.integrals.data[b*r+a],
                    result.amplitudes_[start+ab],i == j ? 1 : 2,record));
            }
        }
        record.correlation_energy=energy.total(); record.energy_change=iteration>1 ? finite(record.correlation_energy-previous) : 0;
        record.converged=iteration>1 && record.maximum_absolute_residual<=options.residual_tolerance
            && std::abs(record.energy_change)<=options.energy_tolerance;
        const bool update=!record.converged && iteration<options.maximum_iterations;
        if (update) {
            for (U i=0;i<o;++i) for (U j=i;j<o;++j) {
                const auto index=pair_index(i,j,o); const auto& pair=in.pairs[index];
                const auto r=pair.rank,start=result.records_[index].offset;
                for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
                    const auto ab=start+a*r+b;
                    const auto delta=gap(pair.virtual_energies.data[a],pair.virtual_energies.data[b],
                        in.occupied_fock.data[i*o+i],in.occupied_fock.data[j*o+j],options.denominator_floor);
                    const auto change=finite(candidate[ab]/delta);
                    if (candidate[ab] != 0 && change == 0) ++record.arithmetic_underflow_count;
                    candidate[ab]=finite(result.amplitudes_[ab]-change);
                }
                if (i == j) {
                    double discarded=0;
                    for (U a=0;a<r;++a) for (U b=a+1;b<r;++b) {
                        const auto ab=start+a*r+b,ba=start+b*r+a;
                        const auto mean=average(candidate[ab],candidate[ba]);
                        discarded=finite(std::hypot(discarded,finite(candidate[ab]-mean)));
                        discarded=finite(std::hypot(discarded,finite(candidate[ba]-mean)));
                        candidate[ab]=candidate[ba]=mean;
                    }
                    record.maximum_diagonal_update_antisymmetry_norm=std::max(record.maximum_diagonal_update_antisymmetry_norm,discarded);
                    if (discarded>options.maximum_diagonal_update_antisymmetry_norm)
                        throw std::invalid_argument("pair MP2 diagonal update antisymmetry budget exceeded");
                }
            }
        }
        if (callback) callback(record,callback_context);
        environment();
        if (input_hash(in,options,inventory,caps) != result.input_identity_)
            throw std::invalid_argument("pair MP2 immutable input changed during callback");
        ++record.input_checks;
        result.snapshot_=record;
        if (!update) break;
        previous=record.correlation_energy;
        result.amplitudes_.swap(candidate);
    }
    Digest payload("vibeqc.bounded-restricted-pair-mp2.payload");
    payload.text(result.input_identity_); payload.integer(result.snapshot_.iteration);
    for (const auto& r:result.records_) { payload.integer(r.rank); payload.integer(r.offset); }
    for (double t:result.amplitudes_) payload.number(t);
    result.payload_=payload.sha.finish_hex();
    return result;
}
} // namespace vibeqc
