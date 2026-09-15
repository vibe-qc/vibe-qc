#include "vibeqc/bounded_restricted_pair_ccsd_solver.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Input = BoundedRestrictedPairCCSDSolverInput;
using Options = BoundedRestrictedPairCCSDSolverOptions;
using Inventory = BoundedRestrictedPairCCSDSolverInventory;
using Caps = BoundedRestrictedPairCCSDSolverCaps;
using Plan = BoundedRestrictedPairCCSDSolverMemoryPlan;
using Progress = BoundedRestrictedPairCCSDSolverProgress;
using View = BoundedRestrictedCCSDRealView;
using Singles = BoundedRestrictedPairCCSDSinglesView;
using Pair = BoundedRestrictedPairCCSDPairView;
using Reader = BoundedRestrictedPairCCSDAmplitudes;
using ReaderPlan = BoundedRestrictedPairCCSDAmplitudesMemoryPlan;
using ReaderCaps = BoundedRestrictedPairCCSDAmplitudesCaps;
using ReaderInventory = BoundedRestrictedPairCCSDAmplitudesInventory;
using PHProducer = BoundedRestrictedPairCCSDParticleHoleProducer;
using PHResult = BoundedRestrictedPairCCSDParticleHoleResult;
static_assert(sizeof(double) == 8 && sizeof(std::size_t) == 8
    && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
    "pair CCSD requires IEEE binary64 and 64-bit extents");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "pair CCSD forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "pair CCSD requires binary64 evaluation"
#endif
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a) throw std::overflow_error("pair CCSD count sum overflows");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a) throw std::overflow_error("pair CCSD count product overflows");
    return a*b;
}
void sha(const std::string& value) {
    if(value.size()!=64) throw std::invalid_argument("pair CCSD split receipt requires 64 hex characters");
    for(char c:value) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("pair CCSD split receipt is malformed");
}
void producer_valid(const PHProducer& p,U maximum_rank) {
    for(char c:p.identity_sha256) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("pair CCSD split producer identity must be SHA-256 ASCII");
    if(!p.maximum_work_units_per_target
        || p.maximum_transient_numerical_bytes_per_target<mul(8,mul(maximum_rank,maximum_rank))
        || p.additional_control_storage_bytes<sizeof(PHResult)+3U*65U)
        throw std::invalid_argument("pair CCSD split producer must inventory output, controls and positive work");
}
U split_control_bytes();
U triangular(U o) { return o%2 ? mul(o,add(o,1)/2) : mul(o/2,add(o,1)); }
U pair_index(U i,U j,U o) {
    if (i>=o || j>=o) throw std::out_of_range("pair CCSD occupied label out of range");
    if (i>j) std::swap(i,j);
    return add(triangular(o)-triangular(o-i),j-i);
}
void limit(U value,U cap,const char* message) {
    if (value>cap) throw std::length_error(message);
}
void extent(U bytes) {
    if (bytes>static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes>std::numeric_limits<U>::max()/8-16384)
        throw std::length_error("pair CCSD storage or SHA extent exceeded");
}
void view(View v,U required) {
    if (v.element_count!=required) throw std::invalid_argument("pair CCSD requires exact view extents");
    if (!required) return;
    const auto bytes=mul(8,required);
    const auto address=reinterpret_cast<std::uintptr_t>(v.data);
    extent(bytes);
    if (!v.data || address%alignof(double) || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
        throw std::invalid_argument("pair CCSD input pointer, alignment or extent invalid");
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("pair CCSD arithmetic is non-finite");
    return x;
}
void environment() {
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1,zero=0;
    if (std::fegetround()!=FE_TONEAREST || !(tiny>0) || tiny*one!=tiny
        || tiny+zero!=tiny || std::fma(tiny,one,zero)!=tiny)
        throw std::invalid_argument("pair CCSD requires round-to-nearest and gradual underflow");
}
struct Sum {
    double value=0,correction=0;
    void include(double term) {
        finite(term);
        const auto next=finite(value+term);
        correction=finite(correction+(std::abs(value)>=std::abs(term)
            ? (value-next)+term : (term-next)+value));
        value=next;
    }
    double total() const { return finite(value+correction); }
};
double gap(double ea,double eb,double ei,double ej,double floor) {
    const std::array<double,4> values{ea,eb,-ei,-ej};
    double largest=0;
    for (auto x:values) largest=std::max(largest,std::abs(x));
    int exponent=0;
    if (largest) std::frexp(largest,&exponent);
    Sum total;
    for (auto x:values) {
        const auto scaled=std::scalbn(x,-exponent);
        if (std::scalbn(scaled,exponent)!=x)
            throw std::overflow_error("pair CCSD denominator scaling loses range");
        total.include(scaled);
    }
    const auto value=std::scalbn(total.total(),exponent);
    if (!std::isfinite(value) || value<=floor)
        throw std::invalid_argument("pair CCSD denominator must strictly exceed its positive floor");
    return value;
}
double average(double a,double b) {
    if (a==b) return a;
    int exponent=0;
    const auto largest=std::max(std::abs(a),std::abs(b));
    if (largest) std::frexp(largest,&exponent);
    const auto x=std::scalbn(a,-exponent),y=std::scalbn(b,-exponent);
    if (std::scalbn(x,exponent)!=a || std::scalbn(y,exponent)!=b)
        throw std::overflow_error("pair CCSD symmetric candidate scaling loses range");
    return finite(std::scalbn((x+y)*0.5,exponent));
}
void options_valid(const Options& o) {
    if (!o.maximum_iterations || !o.maximum_integral_work_units_per_call)
        throw std::invalid_argument("pair CCSD iterations and integral work declaration must be positive");
    for (double x:{o.denominator_floor,o.singles_residual_tolerance,o.doubles_residual_tolerance,
        o.energy_tolerance,o.coefficient_orthogonality_tolerance})
        if (!std::isfinite(x) || x<=0) throw std::invalid_argument("pair CCSD tolerances must be positive finite");
    if (o.coefficient_orthogonality_tolerance>=1
        || !std::isfinite(o.maximum_diagonal_update_antisymmetry_norm)
        || o.maximum_diagonal_update_antisymmetry_norm<0)
        throw std::invalid_argument("pair CCSD orthogonality or diagonal candidate budget invalid");
}
ReaderInventory reader_inventory(const Inventory& inv) {
    // A subset admission; the OUTER plan inventories all concurrent owners.
    return {1,0,0,inv.fixed_backend_margin_bytes_per_replica};
}
ReaderCaps reader_caps(const ReaderPlan& p) {
    return {p.n_occupied,p.common_virtual_dimension,p.pair_count,p.maximum_singles_rank,p.maximum_pair_rank,
        p.borrowed_table_bytes,p.borrowed_numerical_bytes,p.per_replica_inventoried_bytes,
        p.required_node_inventoried_bytes,p.validation_work_units,
        p.maximum_singles_work_units_per_query,p.maximum_doubles_work_units_per_query};
}
BoundedRestrictedCCSDAmplitudeProvider amplitude_metadata(const ReaderPlan& p) {
    BoundedRestrictedCCSDAmplitudeProvider a;
    a.retained_numerical_bytes=p.borrowed_numerical_bytes;
    a.maximum_singles_work_units_per_query=p.maximum_singles_work_units_per_query;
    a.maximum_doubles_work_units_per_query=p.maximum_doubles_work_units_per_query;
    return a;
}
constexpr U fixed_controls=65536+sizeof(Input)+sizeof(Options)+sizeof(Inventory)+sizeof(Caps)
    +3*sizeof(Plan)+3*sizeof(Progress)+2*sizeof(Reader)+8*sizeof(std::vector<double>)
    +4*sizeof(detail::Sha256);
Plan finish_plan(const ReaderPlan& r,const Options& options,const Inventory& inventory,U retained,U transient,
    const PHProducer* ph=nullptr) {
    options_valid(options);
    if (!inventory.numerical_replicas || !inventory.fixed_backend_margin_bytes_per_replica)
        throw std::invalid_argument("pair CCSD replicas and backend allowance must be positive");
    Plan p;
    if(ph) producer_valid(*ph,r.maximum_pair_rank);
    p.split_bare_particle_hole=ph!=nullptr;
    p.amplitudes=r; p.uniform_rank_upper_bound=r.uniform_rank_upper_bound;
    p.n_occupied=r.n_occupied; p.common_virtual_dimension=r.common_virtual_dimension; p.pair_count=r.pair_count;
    p.maximum_singles_rank=r.maximum_singles_rank; p.maximum_pair_rank=r.maximum_pair_rank;
    p.maximum_iterations=options.maximum_iterations;
    p.total_singles_elements=r.singles_amplitude_elements; p.total_doubles_elements=r.pair_amplitude_elements;
    const auto o=p.n_occupied,n=p.common_virtual_dimension,n2=mul(n,n),pcount=p.pair_count;
    const auto a=add(p.total_singles_elements,p.total_doubles_elements);
    const auto rank_sum=add(r.singles_amplitude_elements,r.pair_coefficient_elements/n);
    p.amplitude_snapshot_bytes=p.candidate_snapshot_bytes=mul(8,a);
    p.projected_fock_diagonal_bytes=mul(8,rank_sum);
    p.retained_record_bytes=mul(16,add(o,pcount));
    p.owned_snapshot_table_bytes=p.borrowed_input_table_bytes=r.borrowed_table_bytes;
    p.borrowed_initial_numerical_bytes=r.borrowed_numerical_bytes;
    p.borrowed_fock_bytes=mul(8,add(add(mul(o,o),n2),mul(o,n)));
    p.provider_retained_numerical_bytes=retained; p.provider_transient_numerical_bytes=transient;
    p.target=ph?plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole(o,n,amplitude_metadata(r),retained,transient)
        :plan_bounded_restricted_ccsd_target_accessor(o,n,amplitude_metadata(r),retained,transient);
    BoundedRestrictedCCSDIntegralProvider ep;
    ep.retained_numerical_bytes=retained; ep.maximum_transient_numerical_bytes=transient;
    p.energy=plan_bounded_restricted_ccsd_energy(o,n,amplitude_metadata(r),ep,
        {0,0,options.maximum_integral_work_units_per_call});
    p.target_owned_peak_bytes=p.target.kernel.peak_owned_numerical_bytes;
    p.output_numerical_bytes=add(p.amplitude_snapshot_bytes,p.retained_record_bytes);
    p.peak_owned_numerical_bytes=add(add(mul(2,p.amplitude_snapshot_bytes),p.retained_record_bytes),
        add(p.projected_fock_diagonal_bytes,p.target_owned_peak_bytes));
    p.control_storage_reservation_bytes=add(fixed_controls,
        add(mul(2,r.fixed_control_storage_bytes),p.energy.total_control_storage_bytes));
    U phase_peak=p.peak_owned_numerical_bytes;
    U payload=add(add(p.borrowed_initial_numerical_bytes,p.borrowed_fock_bytes),add(retained,transient));
    if(ph) {
        p.particle_hole_retained_numerical_bytes=ph->additional_retained_numerical_bytes;
        p.particle_hole_transient_numerical_bytes=ph->maximum_transient_numerical_bytes_per_target;
        p.particle_hole_control_storage_bytes=add(ph->additional_control_storage_bytes,split_control_bytes());
        p.control_storage_reservation_bytes=add(p.control_storage_reservation_bytes,p.particle_hole_control_storage_bytes);
        const U resident=p.peak_owned_numerical_bytes-p.target_owned_peak_bytes;
        p.remaining_target_phase_bytes=add(p.target_owned_peak_bytes,transient);
        p.local_particle_hole_phase_bytes=p.particle_hole_transient_numerical_bytes;
        p.peak_owned_numerical_bytes=add(resident,std::max(p.target_owned_peak_bytes,p.local_particle_hole_phase_bytes));
        phase_peak=add(resident,std::max(p.remaining_target_phase_bytes,p.local_particle_hole_phase_bytes));
        payload=add(payload-transient,p.particle_hole_retained_numerical_bytes);
    }
    p.per_replica_inventoried_bytes=add(phase_peak,add(payload,
        add(add(p.owned_snapshot_table_bytes,p.borrowed_input_table_bytes),
        add(p.control_storage_reservation_bytes,add(inventory.other_live_bytes_per_replica,
        inventory.fixed_backend_margin_bytes_per_replica)))));
    p.required_node_inventoried_bytes=add(inventory.external_node_bytes,
        mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    p.metadata_work_units=r.metadata_work_units;
    // Includes all finite/Gram/hash/address passes, both initial and current
    // reader validation, original F hashing and bounded control handling.
    const auto lanes=add(add(r.borrowed_numerical_bytes/8,p.borrowed_fock_bytes/8),add(o,pcount));
    p.validation_work_units=add(mul(4,r.validation_work_units),mul(4096,add(lanes,1)));
    p.initialization_work_units=mul(4096,add(add(mul(n2,rank_sum),a),add(rank_sum,1)));
    p.projection_work_units_per_snapshot=mul(4096,
        add(add(mul(n,p.total_singles_elements),mul(n2,p.total_doubles_elements)),add(a,1)));
    const auto e1=p.energy.singles_calls,e2=p.energy.doubles_calls,ei=p.energy.integral_calls;
    const auto ew=p.energy.total_work_units_upper_bound;
    const auto ti=p.target.kernel.integral_calls_upper_bound;
    const auto tw=add(add(p.target.kernel.kernel_work_units_upper_bound,p.target.amplitude_work_units_upper_bound),
        mul(ti,options.maximum_integral_work_units_per_call));
    p.target_evaluations_upper_bound=mul(options.maximum_iterations,pcount);
    p.integral_calls_upper_bound=mul(options.maximum_iterations,add(mul(pcount,ti),ei));
    p.singles_calls_upper_bound=mul(options.maximum_iterations,add(mul(pcount,p.target.singles_calls_upper_bound),e1));
    p.doubles_calls_upper_bound=mul(options.maximum_iterations,add(mul(pcount,p.target.doubles_calls_upper_bound),e2));
    p.work_units_per_snapshot=add(p.validation_work_units,
        add(p.projection_work_units_per_snapshot,add(mul(pcount,tw),ew)));
    if(ph) {
        p.particle_hole_calls_upper_bound=p.target_evaluations_upper_bound;
        p.particle_hole_work_units_per_target=ph->maximum_work_units_per_target;
        // Current/initial Reader plus original F checks before and after
        // the external producer; output scan, receipts, norms and addition.
        p.particle_hole_guard_work_units_per_target=add(mul(2,p.validation_work_units),
            mul(4096,add(1024,add(mul(p.maximum_pair_rank,p.maximum_pair_rank),add(o,pcount)))));
        p.work_units_per_snapshot=add(p.work_units_per_snapshot,mul(pcount,
            add(p.particle_hole_work_units_per_target,p.particle_hole_guard_work_units_per_target)));
        p.omitted_bare_integral_calls_upper_bound=mul(p.target_evaluations_upper_bound,p.target.kernel.omitted_bare_integral_calls);
    }
    p.work_units_upper_bound=add(p.metadata_work_units,add(p.validation_work_units,
        add(p.initialization_work_units,mul(options.maximum_iterations,p.work_units_per_snapshot))));
    for (auto x:{p.peak_owned_numerical_bytes,p.borrowed_initial_numerical_bytes,p.borrowed_fock_bytes,
        p.per_replica_inventoried_bytes,p.required_node_inventoried_bytes}) extent(x);
    if (a>std::vector<double>().max_size() || rank_sum>std::vector<double>().max_size())
        throw std::length_error("pair CCSD vector extent exceeded");
    return p;
}
void cap_plan(const Plan& p,const Caps& c) {
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"pair CCSD owned numerical cap exceeded");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"pair CCSD per-replica cap exceeded");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"pair CCSD node cap exceeded");
    limit(p.integral_calls_upper_bound,c.maximum_integral_calls,"pair CCSD integral count cap exceeded");
    limit(p.singles_calls_upper_bound,c.maximum_singles_calls,"pair CCSD singles count cap exceeded");
    limit(p.doubles_calls_upper_bound,c.maximum_doubles_calls,"pair CCSD doubles count cap exceeded");
    limit(p.work_units_upper_bound,c.maximum_work_units,"pair CCSD work cap exceeded");
    if(p.split_bare_particle_hole)
        limit(p.particle_hole_calls_upper_bound,c.maximum_particle_hole_calls,"pair CCSD particle-hole call cap exceeded");
}
struct Digest {
    detail::Sha256 h;
    explicit Digest(const char* domain) { text(domain); integer(1); }
    void integer(U x) {
        std::array<std::uint8_t,8> b{};
        for (U k=0;k<8;++k) b[k]=static_cast<std::uint8_t>(x>>(56-8*k));
        h.update(b.data(),b.size());
    }
    void text(const std::string& s) {
        integer(s.size()); h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
    void number(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("pair CCSD hash input must be finite");
        if (x==0) x=0;
        U bits=0; std::memcpy(&bits,&x,8); integer(bits);
    }
};
struct PHVisit {
    U i=0,j=0,o=0,rank=0,iteration=0,maximum_output=0;
    U maximum_control=0,maximum_work=0;
    double* candidate=nullptr;
    const std::string& snapshot;
    Digest& consumed;
    U attempts=0,visits=0,sources=0;
    bool failed=false;
    static void visit(const PHResult& result,const std::string& snapshot,void* context) {
        auto& self=*static_cast<PHVisit*>(context);self.failed=true;
        if(self.attempts++) throw std::invalid_argument("pair CCSD particle-hole producer visited more than once");
        const auto& m=result.memory();const auto& d=result.diagnostics();
        if(result.target_i()!=self.i || result.target_j()!=self.j || m.occupied_count!=self.o
            || m.target_dimension!=self.rank || m.source_slots!=mul(2,self.o)
            || d.visited_sources!=mul(2,self.o) || m.output_bytes!=mul(8,mul(self.rank,self.rank))
            || m.output_bytes>self.maximum_output || m.peak_owned_numerical_bytes>self.maximum_output
            || m.fixed_control_storage_bytes>self.maximum_control
            || d.charged_work_units>self.maximum_work || snapshot!=self.snapshot)
            throw std::invalid_argument("pair CCSD particle-hole target, completed slots or snapshot differs");
        sha(snapshot);sha(result.input_stream_identity_sha256());sha(result.payload_identity_sha256());sha(result.identity_sha256());
        const U lanes=mul(self.rank,self.rank);const auto* values=result.residual_data();
        view({values,static_cast<std::size_t>(lanes)},lanes);
        // Validate the ENTIRE borrowed output before any candidate write.
        for(U at=0;at<lanes;++at) finite(values[at]);
        self.consumed.integer(self.iteration);self.consumed.integer(self.i);self.consumed.integer(self.j);
        self.consumed.text(snapshot);self.consumed.text(result.identity_sha256());
        self.consumed.text(result.input_stream_identity_sha256());self.consumed.text(result.payload_identity_sha256());
        for(U at=0;at<lanes;++at) self.candidate[at]=finite(self.candidate[at]+values[at]);
        self.sources=d.visited_sources;self.visits=1;self.failed=false;
    }
};
U split_control_bytes() {
    return 65536U+2U*sizeof(PHProducer)+sizeof(PHVisit)+3U*sizeof(Digest)+8U*65U;
}
std::string input_identity(const Input& in,const Reader& reader) {
    Digest d("vibeqc.bounded-restricted-pair-ccsd.input");
    d.text(reader.snapshot_identity_sha256());
    for (auto v:{in.f_oo,in.f_vv,in.f_ov}) {
        d.integer(v.element_count);
        for (U k=0;k<v.element_count;++k) d.number(v.data[k]);
    }
    return d.h.finish_hex();
}
void fock_valid(const Input& in) {
    for (auto v:{in.f_oo,in.f_vv,in.f_ov})
        for (U k=0;k<v.element_count;++k)
            if (!std::isfinite(v.data[k])) throw std::invalid_argument("pair CCSD Fock inputs must be finite");
    const auto o=in.initial.n_occupied,n=in.initial.common_virtual_dimension;
    for (U i=0;i<o;++i) for (U j=0;j<i;++j)
        if (in.f_oo.data[i*o+j]!=in.f_oo.data[j*o+i])
            throw std::invalid_argument("pair CCSD original occupied Fock must be exactly symmetric");
    for (U a=0;a<n;++a) for (U b=0;b<a;++b)
        if (in.f_vv.data[a*n+b]!=in.f_vv.data[b*n+a])
            throw std::invalid_argument("pair CCSD original virtual Fock must be exactly symmetric");
}
struct IntegralCounter {
    BoundedRestrictedCCSDIntegralProvider provider;
    U count=0,limit=0;
    static double value(U p,U q,U r,U s,void* context) {
        auto& c=*static_cast<IntegralCounter*>(context);
        if (c.count>=c.limit) throw std::length_error("pair CCSD exhausted integral count");
        ++c.count;
        return finite(c.provider.value(p,q,r,s,c.provider.context));
    }
};
BoundedRestrictedCCSDTargetAccessorCaps target_caps(const BoundedRestrictedCCSDTargetAccessorMemoryPlan& p) {
    return {{p.kernel.peak_owned_numerical_bytes,p.kernel.total_live_numerical_bytes,
        p.kernel.integral_calls_upper_bound,p.kernel.kernel_work_units_upper_bound},
        p.singles_calls_upper_bound,p.doubles_calls_upper_bound,p.amplitude_work_units_upper_bound};
}
BoundedRestrictedCCSDEnergyResult pair_ccsd_energy(U o,U n,View fov,
    const BoundedRestrictedCCSDAmplitudeProvider& amplitudes,
    const BoundedRestrictedCCSDIntegralProvider& integrals,const Options& options) {
    const BoundedRestrictedCCSDEnergyInventory inventory{0,0,options.maximum_integral_work_units_per_call};
    const auto p=plan_bounded_restricted_ccsd_energy(o,n,amplitudes,integrals,inventory);
    const BoundedRestrictedCCSDEnergyCaps caps{p.total_live_numerical_bytes,p.total_control_storage_bytes,
        p.singles_calls,p.doubles_calls,p.integral_calls,p.total_work_units_upper_bound};
    return bounded_restricted_ccsd_energy({o,n,fov},amplitudes,integrals,inventory,caps);
}
} // namespace

Plan plan_bounded_restricted_pair_ccsd_solver_upper(U o,U n,U s,U r,const Options& options,
    const Inventory& inventory,U retained,U transient) {
    return finish_plan(plan_bounded_restricted_pair_ccsd_amplitudes_upper(o,n,s,r,reader_inventory(inventory)),
        options,inventory,retained,transient);
}
Plan plan_bounded_restricted_pair_ccsd_solver_upper(U o,U n,U s,U r,const Options& options,
    const Inventory& inventory,const PHProducer& ph,U retained,U transient) {
    return finish_plan(plan_bounded_restricted_pair_ccsd_amplitudes_upper(o,n,s,r,reader_inventory(inventory)),
        options,inventory,retained,transient,&ph);
}
static Plan plan_pair_solver_impl(const Input& in,const Options& options,
    const Inventory& inventory,const Caps& caps,U retained,U transient,const PHProducer* ph) {
    if (!caps.maximum_occupied_count || !caps.maximum_common_virtual_dimension || !caps.maximum_pair_count
        || !caps.maximum_owned_numerical_bytes || !caps.maximum_per_replica_inventoried_bytes
        || !caps.maximum_node_inventoried_bytes || !caps.maximum_integral_calls || !caps.maximum_singles_calls
        || !caps.maximum_doubles_calls || !caps.maximum_work_units)
        throw std::invalid_argument("pair CCSD positive caps must be explicit");
    options_valid(options);
    const auto o=in.initial.n_occupied,n=in.initial.common_virtual_dimension;
    limit(o,caps.maximum_occupied_count,"pair CCSD occupied cap exceeded");
    limit(n,caps.maximum_common_virtual_dimension,"pair CCSD common virtual cap exceeded");
    limit(triangular(o),caps.maximum_pair_count,"pair CCSD pair count cap exceeded");
    // The nested metadata plan checks traversal work/counts and exact view
    // extents BEFORE floating reads. Its interim byte caps are outer ceilings,
    // not evidence that the complete simultaneous inventory fits.
    const ReaderCaps rc{caps.maximum_occupied_count,caps.maximum_common_virtual_dimension,caps.maximum_pair_count,
        caps.maximum_singles_rank,caps.maximum_pair_rank,caps.maximum_per_replica_inventoried_bytes,
        caps.maximum_per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,
        caps.maximum_node_inventoried_bytes,caps.maximum_work_units,caps.maximum_work_units,caps.maximum_work_units};
    const auto rp=plan_bounded_restricted_pair_ccsd_amplitudes(in.initial,
        {options.coefficient_orthogonality_tolerance},reader_inventory(inventory),rc);
    auto p=finish_plan(rp,options,inventory,retained,transient,ph);
    cap_plan(p,caps);
    view(in.f_oo,mul(o,o)); view(in.f_vv,mul(n,n)); view(in.f_ov,mul(o,n));
    return p;
}
Plan plan_bounded_restricted_pair_ccsd_solver(const Input& in,const Options& options,
    const Inventory& inventory,const Caps& caps,U retained,U transient) {
    return plan_pair_solver_impl(in,options,inventory,caps,retained,transient,nullptr);
}
Plan plan_bounded_restricted_pair_ccsd_solver(const Input& in,const Options& options,
    const Inventory& inventory,const Caps& caps,const PHProducer& ph,U retained,U transient) {
    return plan_pair_solver_impl(in,options,inventory,caps,retained,transient,&ph);
}

BoundedRestrictedPairCCSDSolverResult BoundedRestrictedPairCCSDSolverResult::solve_impl(const Input& source,
    const BoundedRestrictedCCSDIntegralProvider& source_provider,const Options& source_options,
    const Inventory& source_inventory,const Caps& source_caps,const PHProducer* source_ph,
    BoundedRestrictedPairCCSDSolverCallback callback,void* callback_context) {
    const Input in=source;
    const auto provider=source_provider;
    const Options options=source_options;
    const Inventory inventory=source_inventory;
    const Caps caps=source_caps;
    const PHProducer ph=source_ph?*source_ph:PHProducer{};
    const auto p=plan_pair_solver_impl(in,options,inventory,caps,
        provider.retained_numerical_bytes,provider.maximum_transient_numerical_bytes,source_ph?&ph:nullptr);
    if (!provider.value) throw std::invalid_argument("pair CCSD integral callback is required");
    if(source_ph && !ph.produce) throw std::invalid_argument("pair CCSD particle-hole producer is required");
    environment(); fock_valid(in);
    const auto ri=reader_inventory(inventory);
    const auto rc=reader_caps(p.amplitudes);
    const BoundedRestrictedPairCCSDAmplitudesOptions ro{options.coefficient_orthogonality_tolerance};
    const auto initial_reader=make_bounded_restricted_pair_ccsd_amplitudes(in.initial,ro,ri,rc);
    const auto initial_identity=input_identity(in,initial_reader);
    const auto o=p.n_occupied,n=p.common_virtual_dimension,count=p.pair_count;
    BoundedRestrictedPairCCSDSolverResult result;
    result.memory_=p; result.input_identity_=initial_identity;
    if(source_ph) {
        Digest execution("vibeqc.bounded-restricted-pair-ccsd.split-execution");execution.text(initial_identity);
        execution.text(std::string(ph.identity_sha256.begin(),ph.identity_sha256.end()));
        execution.text("replace-only-complete-bare-particle-hole;declared-producer-not-physical-certificate");
        execution.integer(ph.additional_retained_numerical_bytes);execution.integer(ph.maximum_transient_numerical_bytes_per_target);
        execution.integer(ph.additional_control_storage_bytes);execution.integer(ph.maximum_work_units_per_target);
        execution.integer(options.maximum_iterations);
        for(double x:{options.denominator_floor,options.singles_residual_tolerance,options.doubles_residual_tolerance,
            options.energy_tolerance,options.coefficient_orthogonality_tolerance,options.maximum_diagonal_update_antisymmetry_norm}) execution.number(x);
        result.split_identity_=execution.h.finish_hex();
    }
    std::optional<Digest> consumed_ph;
    if(source_ph) consumed_ph.emplace("vibeqc.bounded-restricted-pair-ccsd.consumed-particle-hole");
    result.singles_.resize(o); result.pairs_.resize(count);
    result.amplitudes_.resize(p.amplitude_snapshot_bytes/8);
    std::vector<double> candidate(result.amplitudes_.size());
    std::vector<double> eps(p.projected_fock_diagonal_bytes/8);
    std::vector<Singles> singles(o);
    std::vector<Pair> pairs(count);
    U offset=0,epos=0;
    const auto project_diagonal=[&](View c,U rank) {
        for (U a=0;a<rank;++a) {
            Sum value;
            for (U x=0;x<n;++x) for (U y=0;y<n;++y)
                value.include(finite(c.data[x*rank+a]*finite(in.f_vv.data[x*n+y]*c.data[y*rank+a])));
            eps[epos++]=value.total();
        }
    };
    for (U i=0;i<o;++i) {
        const auto& s=in.initial.singles[i];
        result.singles_[i]={s.rank,offset};
        if (s.rank) std::copy_n(s.amplitudes.data,s.rank,result.amplitudes_.data()+offset);
        singles[i]={s.rank,s.coefficients,{nullptr,s.rank}};
        offset+=s.rank; project_diagonal(s.coefficients,s.rank);
    }
    for (U k=0;k<count;++k) {
        const auto& s=in.initial.pairs[k];
        result.pairs_[k]={s.rank,offset};
        if (s.rank) std::copy_n(s.amplitudes.data,s.rank*s.rank,result.amplitudes_.data()+offset);
        pairs[k]={s.rank,s.coefficients,{nullptr,static_cast<std::size_t>(s.rank*s.rank)}};
        offset+=s.rank*s.rank; project_diagonal(s.coefficients,s.rank);
    }
    if (offset!=result.amplitudes_.size() || epos!=eps.size())
        throw std::logic_error("pair CCSD snapshot census changed after admission");
    double minimum=std::numeric_limits<double>::infinity(),maximum=0;
    const auto record_gap=[&](double x) { minimum=std::min(minimum,x); maximum=std::max(maximum,x); };
    epos=0;
    for (U i=0;i<o;++i) {
        for (U a=0;a<singles[i].rank;++a)
            record_gap(gap(eps[epos+a],0,in.f_oo.data[i*o+i],0,options.denominator_floor));
        epos+=singles[i].rank;
    }
    U pk=0;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j,++pk) {
        const auto r=pairs[pk].rank;
        for (U a=0;a<r;++a) for (U b=0;b<r;++b)
            record_gap(gap(eps[epos+a],eps[epos+b],in.f_oo.data[i*o+i],in.f_oo.data[j*o+j],options.denominator_floor));
        epos+=r;
    }
    result.minimum_denominator_=maximum?minimum:0; result.maximum_denominator_=maximum;
    IntegralCounter counter{provider,0,p.integral_calls_upper_bound};
    const BoundedRestrictedCCSDIntegralProvider counted{&IntegralCounter::value,&counter,
        provider.retained_numerical_bytes,provider.maximum_transient_numerical_bytes};
    const BoundedRestrictedPairCCSDAmplitudesInput current{o,n,singles.data(),singles.size(),pairs.data(),pairs.size()};
    U target_count=0,singles_count=0,doubles_count=0;
    U ph_calls=0,ph_visits=0,ph_sources=0,ph_work=0;
    double previous_energy=0,discarded=0;
    for (U iteration=1;iteration<=options.maximum_iterations;++iteration) {
        for (U i=0;i<o;++i) singles[i].amplitudes.data=singles[i].rank
            ? result.amplitudes_.data()+result.singles_[i].offset : nullptr;
        for (U k=0;k<count;++k) pairs[k].amplitudes.data=pairs[k].rank
            ? result.amplitudes_.data()+result.pairs_[k].offset : nullptr;
        const auto reader=make_bounded_restricted_pair_ccsd_amplitudes(current,ro,ri,rc);
        const auto amplitudes=reader.amplitude_provider();
        Progress progress;
        progress.iteration=iteration;
        const bool update=iteration<options.maximum_iterations;
        U k=0;
        for (U i=0;i<o;++i) {
            for (U j=i;j<o;++j,++k) {
                const BoundedRestrictedCCSDTargetOperatorInput target{o,n,i,j,in.f_oo,in.f_vv,in.f_ov};
                {
                const auto raw=source_ph
                    ?bounded_restricted_ccsd_target_residual_accessor_without_bare_particle_hole(target,amplitudes,counted,target_caps(p.target))
                    :bounded_restricted_ccsd_target_residual_accessor(target,amplitudes,counted,target_caps(p.target));
                ++target_count;
                singles_count=add(singles_count,raw.singles_calls); doubles_count=add(doubles_count,raw.doubles_calls);
                // A single common-space residual owner. Project only now,
                // never the coupled source amplitudes or their integrals.
                if (i==j) {
                    const auto& s=singles[i];
                    for (U a=0;a<s.rank;++a) {
                        Sum value;
                        for (U x=0;x<n;++x) value.include(s.coefficients.data[x*s.rank+a]*raw.target.singles[x]);
                        const auto residual=value.total();
                        progress.singles_max_residual=std::max(progress.singles_max_residual,std::abs(residual));
                        progress.singles_residual_norm=finite(std::hypot(progress.singles_residual_norm,residual));
                        candidate[result.singles_[i].offset+a]=residual;
                    }
                }
                const auto& s=pairs[k];
                for (U a=0;a<s.rank;++a) for (U b=0;b<s.rank;++b) {
                    Sum value;
                    for (U x=0;x<n;++x) for (U y=0;y<n;++y)
                        value.include(finite(s.coefficients.data[x*s.rank+a]*finite(raw.target.doubles[x*n+y]*s.coefficients.data[y*s.rank+b])));
                    const auto residual=value.total();
                    if(!source_ph) {
                        progress.doubles_max_residual=std::max(progress.doubles_max_residual,std::abs(residual));
                        progress.doubles_residual_norm=finite(std::hypot(progress.doubles_residual_norm,residual));
                        if (i!=j) progress.doubles_residual_norm=finite(std::hypot(progress.doubles_residual_norm,residual));
                    }
                    candidate[result.pairs_[k].offset+a*s.rank+b]=residual;
                }
                } // Destroy raw common R before any local PH factory scratch.
                if(source_ph) {
                    const auto check=[&]() {
                        environment();reader.validate_immutable_snapshot();initial_reader.validate_immutable_snapshot();
                        if(input_identity(in,initial_reader)!=initial_identity)
                            throw std::invalid_argument("pair CCSD initial inputs changed during particle-hole production");
                    };
                    check();const auto snapshot=reader.snapshot_identity_sha256();
                    if(ph_calls>=caps.maximum_particle_hole_calls || ph_calls>=p.particle_hole_calls_upper_bound)
                        throw std::length_error("pair CCSD exhausted particle-hole producer calls");
                    ++ph_calls;ph_work=add(ph_work,add(p.particle_hole_work_units_per_target,p.particle_hole_guard_work_units_per_target));
                    limit(ph_work,p.work_units_upper_bound,"pair CCSD particle-hole work charge exceeds admission");
                    const auto& s=pairs[k];
                    PHVisit visitor{i,j,o,s.rank,iteration,ph.maximum_transient_numerical_bytes_per_target,
                        ph.additional_control_storage_bytes,ph.maximum_work_units_per_target,
                        s.rank?candidate.data()+result.pairs_[k].offset:nullptr,snapshot,*consumed_ph};
                    ph.produce(reader,p,i,j,iteration,&PHVisit::visit,&visitor,ph.context);
                    check();
                    if(visitor.failed || visitor.attempts!=1 || visitor.visits!=1)
                        throw std::invalid_argument("pair CCSD particle-hole producer requires exactly one successful visit");
                    ph_visits=add(ph_visits,visitor.visits);ph_sources=add(ph_sources,visitor.sources);
                    for(U a=0;a<s.rank;++a) for(U b=0;b<s.rank;++b) {
                        const double residual=candidate[result.pairs_[k].offset+a*s.rank+b];
                        progress.doubles_max_residual=std::max(progress.doubles_max_residual,std::abs(residual));
                        progress.doubles_residual_norm=finite(std::hypot(progress.doubles_residual_norm,residual));
                        if(i!=j) progress.doubles_residual_norm=finite(std::hypot(progress.doubles_residual_norm,residual));
                    }
                }
            }
        }
        // Energy adapter invocation is below; it must retain the complete
        // common-frame singles product even outside every target PNO space.
        const auto energy=pair_ccsd_energy(o,n,in.f_ov,amplitudes,counted,options);
        singles_count=add(singles_count,energy.singles_calls); doubles_count=add(doubles_count,energy.doubles_calls);
        limit(singles_count,p.singles_calls_upper_bound,"pair CCSD aggregate singles count exceeded");
        limit(doubles_count,p.doubles_calls_upper_bound,"pair CCSD aggregate doubles count exceeded");
        progress.correlation_energy=energy.correlation_energy;
        progress.has_previous_energy=iteration>1;
        progress.energy_change=progress.has_previous_energy?finite(progress.correlation_energy-previous_energy):0;
        progress.converged=progress.has_previous_energy && std::abs(progress.energy_change)<=options.energy_tolerance
            && progress.singles_max_residual<=options.singles_residual_tolerance
            && progress.doubles_max_residual<=options.doubles_residual_tolerance;
        // An already converged snapshot does not need an unevaluated next
        // candidate. Audit/disclose only updates that will actually be used.
        if (!progress.converged && update) {
            // Reuse the ragged residual snapshot for the next amplitudes.
            // Even candidate arithmetic is deferred: an irrelevant R/gap
            // could overflow despite a valid converged current snapshot.
            U at=0;
            for (U i=0;i<o;++i) {
                const auto& s=singles[i]; const auto start=result.singles_[i].offset;
                for (U a=0;a<s.rank;++a) candidate[start+a]=finite(s.amplitudes.data[a]-finite(candidate[start+a]/
                    gap(eps[at+a],0,in.f_oo.data[i*o+i],0,options.denominator_floor)));
                at+=s.rank;
            }
            U pk=0;
            for (U i=0;i<o;++i) for (U j=i;j<o;++j,++pk) {
                const auto& s=pairs[pk]; const auto start=result.pairs_[pk].offset;
                for (U a=0;a<s.rank;++a) for (U b=0;b<s.rank;++b) {
                    const auto ab=a*s.rank+b;
                    candidate[start+ab]=finite(s.amplitudes.data[ab]-finite(candidate[start+ab]/
                        gap(eps[at+a],eps[at+b],in.f_oo.data[i*o+i],in.f_oo.data[j*o+j],options.denominator_floor)));
                }
                if (i==j) for (U a=0;a<s.rank;++a) for (U b=0;b<a;++b) {
                    auto& x=candidate[start+a*s.rank+b]; auto& y=candidate[start+b*s.rank+a];
                    const auto mean=average(x,y);
                    discarded=finite(std::hypot(discarded,finite(x-mean)));
                    discarded=finite(std::hypot(discarded,finite(y-mean)));
                    x=y=mean;
                }
                at+=s.rank;
            }
            if (discarded>options.maximum_diagonal_update_antisymmetry_norm)
                throw std::invalid_argument("pair CCSD diagonal candidate antisymmetry budget exceeded");
        }
        progress.target_evaluations=target_count; progress.integral_calls=counter.count;
        progress.singles_calls=singles_count; progress.doubles_calls=doubles_count;
        progress.particle_hole_calls=ph_calls;progress.particle_hole_visits=ph_visits;progress.particle_hole_source_slots=ph_sources;
        progress.charged_particle_hole_work_units=ph_work;
        progress.maximum_diagonal_update_antisymmetry_norm=discarded;
        progress.input_checks=add(iteration+1,mul(2,ph_calls));
        progress.charged_work_units=add(p.metadata_work_units,add(p.validation_work_units,
            add(p.initialization_work_units,mul(iteration,p.work_units_per_snapshot))));
        reader.validate_immutable_snapshot();
        if (callback) callback(progress,callback_context);
        environment();
        initial_reader.validate_immutable_snapshot();
        if (input_identity(in,initial_reader)!=initial_identity)
            throw std::invalid_argument("pair CCSD initial inputs changed during callback or evaluation");
        result.snapshot_=progress;
        if (progress.converged || !update) {
            Digest digest("vibeqc.bounded-restricted-pair-ccsd.amplitudes");
            digest.text(initial_identity);
            for (double x:result.amplitudes_) digest.number(x);
            result.payload_=digest.h.finish_hex();
            if(source_ph) result.particle_hole_identity_=consumed_ph->h.finish_hex();
            return result;
        }
        previous_energy=progress.correlation_energy;
        result.amplitudes_.swap(candidate);
    }
    throw std::logic_error("pair CCSD snapshot loop did not terminate");
}

BoundedRestrictedPairCCSDSolverResult bounded_restricted_pair_ccsd_solve(const Input& in,
    const BoundedRestrictedCCSDIntegralProvider& integrals,const Options& options,const Inventory& inventory,
    const Caps& caps,BoundedRestrictedPairCCSDSolverCallback callback,void* context) {
    return BoundedRestrictedPairCCSDSolverResult::solve_impl(in,integrals,options,inventory,caps,nullptr,callback,context);
}
BoundedRestrictedPairCCSDSolverResult bounded_restricted_pair_ccsd_solve(const Input& in,
    const BoundedRestrictedCCSDIntegralProvider& integrals,const PHProducer& ph,const Options& options,const Inventory& inventory,
    const Caps& caps,BoundedRestrictedPairCCSDSolverCallback callback,void* context) {
    return BoundedRestrictedPairCCSDSolverResult::solve_impl(in,integrals,options,inventory,caps,&ph,callback,context);
}

BoundedRestrictedPairCCSDSolverPayloadValidationPlan
plan_bounded_restricted_pair_ccsd_solver_payload_validation(const BoundedRestrictedPairCCSDSolverResult& result) {
    using Validation = BoundedRestrictedPairCCSDSolverPayloadValidationPlan;
    const auto& m=result.memory_;
    Validation p;
    p.n_occupied=m.n_occupied; p.pair_count=m.pair_count;
    if (!m.n_occupied || !m.common_virtual_dimension || m.pair_count!=triangular(m.n_occupied)
        || result.singles_.size()!=m.n_occupied || result.pairs_.size()!=m.pair_count
        || result.input_identity_.size()!=64 || result.payload_.size()!=64
        || !result.snapshot_.iteration || result.snapshot_.iteration>m.maximum_iterations
        || m.maximum_singles_rank>m.common_virtual_dimension || m.maximum_pair_rank>m.common_virtual_dimension)
        throw std::invalid_argument("pair CCSD payload owner is consumed or malformed");
    p.record_count=add(m.n_occupied,m.pair_count);
    p.singles_elements=m.total_singles_elements; p.doubles_elements=m.total_doubles_elements;
    p.numerical_lanes=add(p.singles_elements,p.doubles_elements);
    p.retained_numerical_bytes=mul(8,p.numerical_lanes);
    p.retained_record_bytes=mul(sizeof(BoundedRestrictedPairCCSDSolverResult::Record),p.record_count);
    if (result.amplitudes_.size()!=p.numerical_lanes
        || m.amplitude_snapshot_bytes!=p.retained_numerical_bytes
        || m.retained_record_bytes!=p.retained_record_bytes
        || m.output_numerical_bytes!=add(p.retained_numerical_bytes,p.retained_record_bytes)
        || m.amplitudes.singles_amplitude_elements!=p.singles_elements
        || m.amplitudes.pair_amplitude_elements!=p.doubles_elements)
        throw std::invalid_argument("pair CCSD payload numerical/record extents are malformed");
    // Existing exact codec: length/domain, u64 version, length/input SHA,
    // then contiguous binary64 singles followed by canonical pair doubles.
    constexpr char domain[]="vibeqc.bounded-restricted-pair-ccsd.amplitudes";
    p.payload_message_bytes=add(8+sizeof(domain)-1+8+8+64,p.retained_numerical_bytes);
    (void)mul(8,p.payload_message_bytes);
    extent(p.retained_numerical_bytes); extent(p.retained_record_bytes);
    p.fixed_codec_payload_bytes=65; // fixed finish_hex string, not zero heap
    p.fixed_control_storage_bytes=4096+sizeof(Digest)+2*sizeof(Validation)
        +p.fixed_codec_payload_bytes+2*sizeof(std::string);
    p.metadata_work_units=mul(512,add(1024,p.record_count));
    p.payload_work_units=mul(512,add(1024,p.numerical_lanes));
    p.work_units=add(p.metadata_work_units,p.payload_work_units);
    return p;
}

void verify_bounded_restricted_pair_ccsd_solver_payload(const BoundedRestrictedPairCCSDSolverResult& result,
    U maximum_work_units) {
    // Admission precedes record traversal, floating reads, fixed codec
    // allocation and all hashing. The O(1) plan never follows a payload view.
    const auto p=plan_bounded_restricted_pair_ccsd_solver_payload_validation(result);
    if (!maximum_work_units) throw std::invalid_argument("pair CCSD payload verification work cap must be positive");
    limit(p.work_units,maximum_work_units,"pair CCSD payload verification work cap exceeded");
    environment();
    for (const auto* identity:{&result.input_identity_,&result.payload_})
        for (char x:*identity)
            if (!((x>='0' && x<='9') || (x>='a' && x<='f')))
                throw std::invalid_argument("pair CCSD payload identity is malformed");
    U offset=0,singles=0,doubles=0,maximum_singles=0,maximum_pair=0;
    for (const auto& r:result.singles_) {
        if (r.offset!=offset || r.rank>result.memory_.maximum_singles_rank)
            throw std::invalid_argument("pair CCSD payload singles record is malformed");
        maximum_singles=std::max(maximum_singles,r.rank);
        offset=add(offset,r.rank); singles=add(singles,r.rank);
    }
    // Storage order IS lexicographic i<=j. No reversed duplicate or padding.
    for (const auto& r:result.pairs_) {
        if (r.offset!=offset || r.rank>result.memory_.maximum_pair_rank)
            throw std::invalid_argument("pair CCSD payload pair record is malformed");
        maximum_pair=std::max(maximum_pair,r.rank);
        const U lanes=mul(r.rank,r.rank);
        offset=add(offset,lanes); doubles=add(doubles,lanes);
    }
    if (offset!=p.numerical_lanes || singles!=p.singles_elements || doubles!=p.doubles_elements
        || maximum_singles!=result.memory_.maximum_singles_rank || maximum_pair!=result.memory_.maximum_pair_rank)
        throw std::invalid_argument("pair CCSD payload canonical layout does not match its plan");
    Digest digest("vibeqc.bounded-restricted-pair-ccsd.amplitudes");
    digest.text(result.input_identity_);
    for (double x:result.amplitudes_) digest.number(x);
    if (digest.h.finish_hex()!=result.payload_)
        throw std::invalid_argument("pair CCSD final snapshot payload digest mismatch");
}

U BoundedRestrictedPairCCSDSolverResult::singles_rank(U i) const {
    if (i>=singles_.size()) throw std::out_of_range("pair CCSD singles label out of range");
    return singles_[i].rank;
}
U BoundedRestrictedPairCCSDSolverResult::pair_rank(U i,U j) const {
    return pairs_.at(pair_index(i,j,memory_.n_occupied)).rank;
}
double BoundedRestrictedPairCCSDSolverResult::singles_amplitude(U i,U a) const {
    if (a>=singles_rank(i)) throw std::out_of_range("pair CCSD singles amplitude label out of range");
    return amplitudes_.at(singles_[i].offset+a);
}
double BoundedRestrictedPairCCSDSolverResult::doubles_amplitude(U i,U j,U a,U b) const {
    const auto k=pair_index(i,j,memory_.n_occupied);
    const auto& r=pairs_.at(k);
    if (a>=r.rank || b>=r.rank) throw std::out_of_range("pair CCSD doubles amplitude label out of range");
    if (i>j) std::swap(a,b);
    return amplitudes_.at(r.offset+a*r.rank+b);
}
BoundedRestrictedCCSDRealView BoundedRestrictedPairCCSDSolverResult::stored_singles_view(U i) const {
    (void)singles_rank(i);
    const auto& r=singles_.at(i);
    if (r.offset>amplitudes_.size() || r.rank>amplitudes_.size()-r.offset)
        throw std::logic_error("pair CCSD singles owner is moved or incomplete");
    return {r.rank ? amplitudes_.data()+r.offset : nullptr,static_cast<std::size_t>(r.rank)};
}
BoundedRestrictedCCSDRealView BoundedRestrictedPairCCSDSolverResult::stored_pair_view(U i,U j) const {
    if (i>j) throw std::invalid_argument("pair CCSD contiguous pair borrow requires i<=j");
    const auto& r=pairs_.at(pair_index(i,j,memory_.n_occupied));
    const auto count=mul(r.rank,r.rank);
    if (r.offset>amplitudes_.size() || count>amplitudes_.size()-r.offset)
        throw std::logic_error("pair CCSD pair owner is moved or incomplete");
    return {count ? amplitudes_.data()+r.offset : nullptr,static_cast<std::size_t>(count)};
}
} // namespace vibeqc
