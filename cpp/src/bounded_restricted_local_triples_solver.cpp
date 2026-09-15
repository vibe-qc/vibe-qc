#include "vibeqc/bounded_restricted_local_triples_solver.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using View = BoundedRestrictedTriplesRealView;
using Space = BoundedRestrictedLocalTriplesSpaceView;
using Input = BoundedRestrictedLocalTriplesSolverInput;
using Options = BoundedRestrictedLocalTriplesSolverOptions;
using Inventory = BoundedRestrictedLocalTriplesSolverInventory;
using Caps = BoundedRestrictedLocalTriplesSolverCaps;
using Plan = BoundedRestrictedLocalTriplesSolverMemoryPlan;
using Progress = BoundedRestrictedLocalTriplesSolverProgress;
using Provider = BoundedRestrictedLocalTriplesMomentProvider;
using Request = BoundedRestrictedLocalTriplesMomentRequest;
using Moment = BoundedRestrictedLocalTriplesMomentView;
using Triple = std::array<U,3>;
static_assert(sizeof(double)==8 && sizeof(std::size_t)==8 && sizeof(Space)==40
    && std::numeric_limits<double>::is_iec559, "local triples requires 64-bit IEEE storage");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
#error "local triples requires strict binary64 evaluation"
#endif
U add(U a,U b) {
    if (b>std::numeric_limits<U>::max()-a) throw std::overflow_error("local triples count sum overflow");
    return a+b;
}
U mul(U a,U b) {
    if (a && b>std::numeric_limits<U>::max()/a) throw std::overflow_error("local triples count product overflow");
    return a*b;
}
U square(U x) { return mul(x,x); }
U cube(U x) { return mul(square(x),x); }
U triangular(U x) { return x%2 ? mul(x,add(x,1)/2) : mul(x/2,add(x,1)); }
U triples(U o) {
    std::array<U,3> a{o,add(o,1),add(o,2)};
    for (auto divisor:{2U,3U}) for (auto& x:a) if (x%divisor==0) { x/=divisor; break; }
    return mul(mul(a[0],a[1]),a[2]);
}
U index(Triple x,U o) {
    for (auto v:x) if (v>=o) throw std::out_of_range("local triples occupied label out of range");
    std::sort(x.begin(),x.end());
    return add(add(triples(o)-triples(o-x[0]),
        triangular(o-x[0])-triangular(o-x[1])),x[2]-x[1]);
}
void ordered(Triple& occupied,Triple& abc) {
    for (U end=3;end>1;--end) for (U a=0;a+1<end;++a)
        if (occupied[a]>occupied[a+1]) {
            std::swap(occupied[a],occupied[a+1]); std::swap(abc[a],abc[a+1]);
        }
}
U at(Triple x,U r) { return (x[0]*r+x[1])*r+x[2]; }
void limit(U value,U cap,const char* message) { if (value>cap) throw std::length_error(message); }
void extent(U bytes) {
    if (bytes>static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes>std::numeric_limits<U>::max()/8-65536)
        throw std::overflow_error("local triples storage/SHA extent overflow");
}
void pointer(const void* p,U bytes,U alignment) {
    if (!bytes) return;
    extent(bytes);
    const auto address=reinterpret_cast<std::uintptr_t>(p);
    if (!p || address%alignment || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
        throw std::invalid_argument("local triples invalid pointer/alignment/extent");
}
void view(View v,U count) {
    if (v.element_count!=count) throw std::invalid_argument("local triples requires exact view extents");
    pointer(v.data,mul(8,count),alignof(double));
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("local triples nonfinite arithmetic");
    return x;
}
void environment() {
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1,zero=0;
    if (std::fegetround()!=FE_TONEAREST || !(tiny>0) || tiny*one!=tiny
        || tiny+zero!=tiny || std::fma(tiny,one,zero)!=tiny)
        throw std::invalid_argument("local triples requires nearest rounding and gradual underflow");
}
struct Sum {
    double value=0,correction=0;
    void include(double x) {
        finite(x); const auto next=finite(value+x);
        correction=finite(correction+(std::abs(value)>=std::abs(x)
            ? (value-next)+x : (x-next)+value)); value=next;
    }
    double total() const { return finite(value+correction); }
};
struct Digest {
    detail::Sha256 h;
    explicit Digest(const char* domain) { h.update(reinterpret_cast<const std::uint8_t*>(domain),std::strlen(domain)); integer(1); }
    void integer(U x) {
        std::array<std::uint8_t,8> b{};
        for (U k=0;k<8;++k) b[k]=static_cast<std::uint8_t>(x>>(56-8*k));
        h.update(b.data(),b.size());
    }
    void number(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("local triples input must be finite");
        if (x==0) x=0;
        U bits=0; std::memcpy(&bits,&x,8); integer(bits);
    }
    void values(View v) { integer(v.element_count); for (U i=0;i<v.element_count;++i) number(v.data[i]); }
};
void options_valid(const Options& o) {
    if (!o.maximum_iterations) throw std::invalid_argument("local triples iterations must be positive");
    for (double x:{o.denominator_floor,o.residual_tolerance,o.energy_tolerance,o.coefficient_orthogonality_tolerance})
        if (!std::isfinite(x) || x<=0) throw std::invalid_argument("local triples tolerances must be positive finite");
    if (o.coefficient_orthogonality_tolerance>=1)
        throw std::invalid_argument("local triples orthogonality tolerance must be below one");
    for (double x:{o.maximum_projected_fock_error,o.maximum_repeated_moment_defect_norm,o.maximum_repeated_update_defect_norm})
        if (!std::isfinite(x) || x<0) throw std::invalid_argument("local triples explicit error budgets must be nonnegative finite");
}
constexpr U fixed_controls=65536+3*sizeof(Plan)+3*sizeof(Progress)+2*sizeof(Input)+2*sizeof(Options)
    +2*sizeof(Inventory)+2*sizeof(Caps)+2*sizeof(Provider)+8*sizeof(std::vector<double>)+4*sizeof(Digest);
Plan finish(Plan p,const Provider& provider,const Options& options,const Inventory& inventory) {
    options_valid(options);
    if (!inventory.numerical_replicas || !inventory.fixed_backend_margin_bytes_per_replica
        || !provider.maximum_work_units_per_visit)
        throw std::invalid_argument("local triples replicas/backend/producer work must be positive");
    const auto o=p.n_occupied,n=p.common_virtual_dimension,q=p.triple_count,r=p.maximum_rank;
    const auto r2=square(r),r3=cube(r);
    p.maximum_iterations=options.maximum_iterations;
    p.amplitude_snapshot_bytes=p.candidate_snapshot_bytes=mul(8,p.total_amplitude_elements);
    p.retained_record_bytes=mul(16,q);
    p.copied_space_table_bytes=p.borrowed_space_table_bytes=mul(sizeof(Space),q);
    p.moment_digest_bytes=mul(64,q);
    p.borrowed_numerical_bytes=mul(8,add(add(square(o),square(n)),mul(add(n,1),p.total_rank)));
    p.neighbour_workspace_bytes=mul(8,add(r3,r2));
    p.occupied_row_bytes=mul(24,o);
    U target_work=0,target_visits=0;
    if (r) {
        const auto t=plan_bounded_restricted_triples_target(o,r,r);
        p.target_owned_peak_bytes=t.peak_owned_numerical_bytes;
        target_work=t.kernel_work_units_upper_bound; target_visits=t.provider_visits_upper_bound;
    }
    p.output_numerical_bytes=add(p.amplitude_snapshot_bytes,p.retained_record_bytes);
    p.peak_owned_numerical_bytes=add(add(mul(2,p.amplitude_snapshot_bytes),p.retained_record_bytes),
        add(add(p.neighbour_workspace_bytes,p.occupied_row_bytes),p.target_owned_peak_bytes));
    p.control_storage_reservation_bytes=add(fixed_controls,
        add(add(p.copied_space_table_bytes,p.borrowed_space_table_bytes),p.moment_digest_bytes));
    const auto producer_bytes=add(provider.retained_numerical_bytes,provider.maximum_transient_numerical_bytes);
    // A metadata-only producer inventory check, before the first W/U visit.
    if (producer_bytes<mul(16,r3)) throw std::length_error("local triples producer inventory omits active moments");
    p.per_replica_inventoried_bytes=add(add(p.peak_owned_numerical_bytes,p.borrowed_numerical_bytes),
        add(add(producer_bytes,p.control_storage_reservation_bytes),
            add(inventory.other_live_bytes_per_replica,inventory.fixed_backend_margin_bytes_per_replica)));
    p.required_node_inventoried_bytes=add(inventory.external_node_bytes,
        mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    p.moment_visits_upper_bound=mul(options.maximum_iterations,p.nonempty_triple_count);
    p.neighbour_visits_upper_bound=mul(p.moment_visits_upper_bound,target_visits);
    const auto scan=mul(4096,add(add(p.borrowed_numerical_bytes/8,q),1));
    p.validation_work_units=add(scan,mul(1024,mul(square(n),mul(r,p.total_rank))));
    const auto per_neighbour=mul(1024,add(add(mul(n,r2),r3),add(q,1)));
    const auto per_target=add(add(target_work,provider.maximum_work_units_per_visit),
        add(mul(2,scan),add(mul(target_visits,per_neighbour),mul(4096,add(r3,1)))));
    const auto per_snapshot=add(mul(p.nonempty_triple_count,per_target),
        add(scan,mul(4096,add(add(p.total_amplitude_elements,q),1))));
    p.work_units_upper_bound=add(mul(2,p.validation_work_units),mul(options.maximum_iterations,per_snapshot));
    for (U bytes:{p.peak_owned_numerical_bytes,p.borrowed_numerical_bytes,p.control_storage_reservation_bytes,
            p.required_node_inventoried_bytes}) extent(bytes);
    return p;
}
void caps_valid(const Plan& p,const Caps& c) {
    limit(p.n_occupied,c.maximum_occupied_count,"local triples occupied cap exceeded");
    limit(p.common_virtual_dimension,c.maximum_common_virtual_dimension,"local triples common dimension cap exceeded");
    limit(p.triple_count,c.maximum_triple_count,"local triples count cap exceeded");
    limit(p.maximum_rank,c.maximum_rank,"local triples rank cap exceeded");
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"local triples owned memory cap exceeded");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"local triples replica memory cap exceeded");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"local triples node memory cap exceeded");
    limit(p.moment_visits_upper_bound,c.maximum_moment_visits,"local triples moment visit cap exceeded");
    limit(p.work_units_upper_bound,c.maximum_work_units,"local triples work cap exceeded");
}
Plan counts(U o,U n) {
    if (!o || !n) throw std::invalid_argument("local triples occupied/common dimensions must be positive");
    Plan p; p.n_occupied=o; p.common_virtual_dimension=n; p.triple_count=triples(o);
    extent(mul(sizeof(Space),p.triple_count)); return p;
}
std::string identity(const Input& in) {
    Digest d("vibeqc.bounded-local-triples.input");
    d.integer(in.n_occupied); d.integer(in.common_virtual_dimension); d.integer(in.ccsd_snapshot_id);
    d.values(in.f_oo); d.values(in.f_vv);
    for (U t=0;t<in.space_count;++t) {
        d.integer(in.spaces[t].rank); d.values(in.spaces[t].coefficients); d.values(in.spaces[t].energies);
    }
    return d.h.finish_hex();
}
void geometry(const Input& in,const Options& options) {
    const auto o=in.n_occupied,n=in.common_virtual_dimension;
    for (const auto& item:std::array<std::pair<View,U>,2>{{{in.f_oo,o},{in.f_vv,n}}}) {
        for (U a=0;a<item.second;++a) for (U b=0;b<a;++b)
            if (item.first.data[a*item.second+b]!=item.first.data[b*item.second+a])
                throw std::invalid_argument("local triples original Fock must be exactly symmetric");
    }
    for (U t=0;t<in.space_count;++t) {
        const auto& s=in.spaces[t]; const auto r=s.rank;
        double gram_error=0,fock_error=0;
        for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
            Sum gram,fock;
            for (U x=0;x<n;++x) {
                gram.include(s.coefficients.data[x*r+a]*s.coefficients.data[x*r+b]);
                Sum column;
                for (U y=0;y<n;++y) column.include(in.f_vv.data[x*n+y]*s.coefficients.data[y*r+b]);
                fock.include(s.coefficients.data[x*r+a]*column.total());
            }
            gram_error=finite(std::hypot(gram_error,finite(gram.total()-(a==b ? 1.0 : 0.0))));
            fock_error=finite(std::hypot(fock_error,finite(fock.total()-(a==b ? s.energies.data[a] : 0.0))));
        }
        if (gram_error>options.coefficient_orthogonality_tolerance)
            throw std::invalid_argument("local triples frame is not orthonormal");
        if (fock_error>options.maximum_projected_fock_error)
            throw std::invalid_argument("local triples projected original Fock error exceeds budget");
    }
}
double gap(const Space& space,Triple abc,Triple ijk,const Input& in,double floor) {
    std::array<double,6> x{};
    for (U a=0;a<3;++a) { x[a]=space.energies.data[abc[a]]; x[a+3]=-in.f_oo.data[ijk[a]*in.n_occupied+ijk[a]]; }
    double largest=0; for (auto v:x) largest=std::max(largest,std::abs(v));
    int exponent=0; if (largest) std::frexp(largest,&exponent);
    Sum s;
    for (auto v:x) {
        const auto scaled=std::scalbn(v,-exponent);
        if (std::scalbn(scaled,exponent)!=v) throw std::overflow_error("local triples denominator scaling loses range");
        s.include(scaled);
    }
    const auto d=std::scalbn(s.total(),exponent);
    if (!std::isfinite(d) || d<=floor)
        throw std::invalid_argument("local triples denominator must strictly exceed its positive floor");
    return d;
}
constexpr std::array<Triple,6> permutations{{{{0,1,2}},{{0,2,1}},{{1,0,2}},{{1,2,0}},{{2,0,1}},{{2,1,0}}}};
// Each orbit is visited once. No extra cube and no sequential in-place
// averaging: collect its <=6 original values before replacing any member.
double symmetrize(double* values,const double* readonly,U r,Triple ijk,bool apply) {
    double norm=0;
    for (U a=0;a<r;++a) for (U b=0;b<r;++b) for (U c=0;c<r;++c) {
        const Triple abc{a,b,c}; const auto current=at(abc,r);
        std::array<U,6> orbit{}; U count=0;
        for (const auto& p:permutations) {
            if (ijk[p[0]]!=ijk[0] || ijk[p[1]]!=ijk[1] || ijk[p[2]]!=ijk[2]) continue;
            const auto next=at({abc[p[0]],abc[p[1]],abc[p[2]]},r);
            if (std::find(orbit.begin(),orbit.begin()+count,next)==orbit.begin()+count) orbit[count++]=next;
        }
        if (current!=*std::min_element(orbit.begin(),orbit.begin()+count)) continue;
        if (count==1) continue;
        std::array<double,6> original{}; double largest=0;
        for (U t=0;t<count;++t) { original[t]=readonly[orbit[t]]; largest=std::max(largest,std::abs(original[t])); }
        int exponent=0; if (largest) std::frexp(largest,&exponent);
        Sum mean;
        for (U t=0;t<count;++t) {
            const auto x=std::scalbn(original[t],-exponent);
            if (std::scalbn(x,exponent)!=original[t]) throw std::overflow_error("local triples symmetry scaling loses range");
            mean.include(x);
        }
        const auto average=finite(std::scalbn(mean.total()/static_cast<double>(count),exponent));
        for (U t=0;t<count;++t) {
            norm=finite(std::hypot(norm,finite(original[t]-average)));
            if (apply) values[orbit[t]]=average;
        }
    }
    return norm;
}
} // namespace

Plan plan_bounded_restricted_local_triples_solver_upper(U o,U n,U rank,const Provider& provider,
    const Options& options,const Inventory& inventory) {
    auto p=counts(o,n);
    if (rank>n) throw std::invalid_argument("local triples rank exceeds common dimension");
    p.uniform_rank_upper_bound=true; p.maximum_rank=rank;
    p.nonempty_triple_count=rank ? p.triple_count : 0;
    p.total_rank=mul(p.triple_count,rank); p.total_amplitude_elements=mul(p.triple_count,cube(rank));
    return finish(p,provider,options,inventory);
}
Plan plan_bounded_restricted_local_triples_solver(const Input& in,const Provider& provider,
    const Options& options,const Inventory& inventory,const Caps& caps) {
    auto p=counts(in.n_occupied,in.common_virtual_dimension);
    // No space-table reads until even the minimal metadata traversal fits.
    const auto minimal=finish(p,provider,options,inventory); caps_valid(minimal,caps);
    if (in.space_count!=p.triple_count) throw std::invalid_argument("local triples requires the complete unordered triple table");
    pointer(in.spaces,mul(sizeof(Space),p.triple_count),alignof(Space));
    for (U t=0;t<p.triple_count;++t) {
        const auto& s=in.spaces[t];
        if (s.rank>p.common_virtual_dimension) throw std::invalid_argument("local triples rank exceeds common dimension");
        p.maximum_rank=std::max(p.maximum_rank,s.rank);
        p.total_rank=add(p.total_rank,s.rank); p.total_amplitude_elements=add(p.total_amplitude_elements,cube(s.rank));
        p.nonempty_triple_count+=s.rank!=0;
        view(s.coefficients,mul(p.common_virtual_dimension,s.rank)); view(s.energies,s.rank);
    }
    p=finish(p,provider,options,inventory); caps_valid(p,caps);
    view(in.f_oo,square(p.n_occupied)); view(in.f_vv,square(p.common_virtual_dimension));
    return p;
}

namespace {
using Result = BoundedRestrictedLocalTriplesSolverResult;
struct Neighbours {
    const Input& input;
    const Result& current;
    const Space& target;
    U snapshot;
    double* scratch;
    static void visit(const BoundedRestrictedTriplesNeighbourRequest& request,
        BoundedRestrictedTriplesNeighbourReceiver receive,void* receiving,void* context) {
        const auto& self=*static_cast<Neighbours*>(context);
        if (request.amplitude_snapshot_id!=self.snapshot || request.replaced_axis>=3
            || request.replacement_occupied>=self.input.n_occupied)
            throw std::logic_error("local triples internal neighbour request mismatch");
        const auto& source=self.input.spaces[index(request.ordered_occupied,self.input.n_occupied)];
        const auto d=source.rank,r=self.target.rank,n=self.input.common_virtual_dimension;
        BoundedRestrictedTriplesNeighbourView result;
        result.ordered_occupied=request.ordered_occupied;
        result.amplitude_snapshot_id=self.snapshot; result.source_virtual_dimension=d;
        if (d) {
            auto* amplitudes=self.scratch; auto* overlap=amplitudes+cube(d);
            const auto ijk=request.ordered_occupied;
            for (U a=0;a<d;++a) for (U b=0;b<d;++b) for (U c=0;c<d;++c)
                amplitudes[(a*d+b)*d+c]=self.current.amplitude(ijk[0],ijk[1],ijk[2],a,b,c);
            for (U a=0;a<r;++a) for (U b=0;b<d;++b) {
                Sum sum;
                for (U x=0;x<n;++x) sum.include(self.target.coefficients.data[x*r+a]*source.coefficients.data[x*d+b]);
                overlap[a*d+b]=sum.total();
            }
            result.amplitudes={amplitudes,static_cast<std::size_t>(cube(d))};
            result.target_source_overlap={overlap,static_cast<std::size_t>(r*d)};
        }
        receive(result,receiving);
    }
};
struct MomentReceiver {
    Request expected;
    const Input& input;
    const Options& options;
    const Space& space;
    Neighbours& neighbours;
    const double* amplitudes;
    double* residual;
    const double* rows;
    std::array<char,64>& digest;
    U iteration,visits=0;
    bool failed=false;
    double energy=0,max_residual=0,norm=0,moment_defect=0;
    U neighbour_visits=0;
    static void receive(const Moment& moment,void* context) {
        auto& self=*static_cast<MomentReceiver*>(context);
        self.failed=true;
        if (++self.visits!=1) throw std::runtime_error("local triples moment provider must visit exactly once");
        environment();
        const auto& a=moment.request; const auto& b=self.expected;
        if (a.occupied!=b.occupied || a.rank!=b.rank || a.ccsd_snapshot_id!=b.ccsd_snapshot_id)
            throw std::invalid_argument("local triples moment request/snapshot mismatch");
        const auto r=self.space.rank,r3=cube(r);
        view(moment.connected,r3); view(moment.singles,r3);
        Digest d("vibeqc.bounded-local-triples.moments");
        for (auto x:a.occupied) d.integer(x);
        d.integer(a.rank); d.integer(a.ccsd_snapshot_id); d.values(moment.connected); d.values(moment.singles);
        const auto sha=d.h.finish_hex();
        if (self.iteration==1) std::copy(sha.begin(),sha.end(),self.digest.begin());
        else if (!std::equal(sha.begin(),sha.end(),self.digest.begin()))
            throw std::invalid_argument("local triples moments changed between immutable CCSD snapshots");
        self.moment_defect=std::max(symmetrize(nullptr,moment.connected.data,r,a.occupied,false),
            symmetrize(nullptr,moment.singles.data,r,a.occupied,false));
        if (self.moment_defect>self.options.maximum_repeated_moment_defect_norm)
            throw std::invalid_argument("local triples repeated-axis moment defect exceeds budget");
        BoundedRestrictedTriplesNeighbourProvider provider;
        provider.visit=&Neighbours::visit; provider.context=&self.neighbours;
        provider.maximum_source_virtual_dimension=self.neighbours.current.memory().maximum_rank;
        const auto p=plan_bounded_restricted_triples_target(self.input.n_occupied,r,provider.maximum_source_virtual_dimension);
        const BoundedRestrictedTriplesTargetCaps caps{p.peak_owned_numerical_bytes,p.total_live_numerical_bytes,
            std::max<U>(1,p.provider_visits_upper_bound),p.kernel_work_units_upper_bound};
        const BoundedRestrictedTriplesTargetInput target{self.input.n_occupied,r,a.occupied,self.iteration,
            {self.amplitudes,static_cast<std::size_t>(r3)},moment.connected,moment.singles,self.space.energies,
            {self.rows,static_cast<std::size_t>(3*self.input.n_occupied)}};
        const auto result=bounded_restricted_triples_target_residual(target,provider,caps);
        environment();
        std::copy(result.residual.begin(),result.residual.end(),self.residual);
        self.energy=result.raw_energy_contraction; self.max_residual=result.maximum_absolute_residual;
        self.norm=result.residual_frobenius_norm; self.neighbour_visits=result.provider_visits;
        self.failed=false;
    }
};
bool same_view(View a,View b) { return a.data==b.data && a.element_count==b.element_count; }
} // namespace

U BoundedRestrictedLocalTriplesSolverResult::rank(U i,U j,U k) const {
    const auto t=index({i,j,k},memory_.n_occupied);
    if (t>=records_.size()) throw std::logic_error("local triples result owner is moved or invalid");
    return records_[t].rank;
}
double BoundedRestrictedLocalTriplesSolverResult::amplitude(U i,U j,U k,U a,U b,U c) const {
    Triple ijk{i,j,k},abc{a,b,c}; const auto t=index(ijk,memory_.n_occupied);
    if (t>=records_.size()) throw std::logic_error("local triples result owner is moved or invalid");
    const auto& record=records_[t];
    for (auto v:abc) if (v>=record.rank) throw std::out_of_range("local triples virtual label out of range");
    ordered(ijk,abc);
    const auto offset=add(record.offset,at(abc,record.rank));
    if (offset>=amplitudes_.size()) throw std::logic_error("local triples amplitude storage is invalid");
    return amplitudes_[offset];
}

BoundedRestrictedLocalTriplesSolverResult bounded_restricted_local_triples_solve(
    const Input& supplied,const Provider& supplied_provider,const Options& supplied_options,
    const Inventory& supplied_inventory,const Caps& supplied_caps,
    BoundedRestrictedLocalTriplesSolverCallback progress,void* progress_context) {
    // Descriptor controls are copied before any externally supplied callback.
    auto input=supplied; const auto provider=supplied_provider; const auto options=supplied_options;
    const auto inventory=supplied_inventory; const auto caps=supplied_caps;
    const auto p=plan_bounded_restricted_local_triples_solver(input,provider,options,inventory,caps);
    environment();
    if (!input.ccsd_snapshot_id || (p.nonempty_triple_count && !provider.visit))
        throw std::invalid_argument("local triples requires a positive CCSD snapshot and moment producer");
    const auto* original_table=input.spaces;
    std::vector<Space> spaces(input.spaces,input.spaces+input.space_count);
    input.spaces=spaces.data();
    const auto initial_identity=identity(input); // scans every floating input
    geometry(input,options);
    const auto check_input=[&]() {
        environment();
        for (U t=0;t<p.triple_count;++t) {
            const auto& old=spaces[t]; const auto& now=original_table[t];
            if (now.rank!=old.rank || !same_view(now.coefficients,old.coefficients) || !same_view(now.energies,old.energies))
                throw std::invalid_argument("local triples borrowed space descriptors changed during callback");
        }
        if (identity(input)!=initial_identity)
            throw std::invalid_argument("local triples immutable input changed during callback");
    };
    BoundedRestrictedLocalTriplesSolverResult result;
    result.memory_=p; result.input_=initial_identity;
    result.records_.resize(static_cast<std::size_t>(p.triple_count));
    result.amplitudes_.resize(static_cast<std::size_t>(p.total_amplitude_elements));
    std::vector<double> candidate(result.amplitudes_.size());
    std::vector<double> workspace(static_cast<std::size_t>((p.neighbour_workspace_bytes+p.occupied_row_bytes)/8));
    std::vector<std::array<char,64>> moment_digests(static_cast<std::size_t>(p.triple_count));
    auto* rows=workspace.data(); auto* neighbour_scratch=rows+3*p.n_occupied;
    U offset=0;
    for (U t=0;t<p.triple_count;++t) {
        result.records_[t]={spaces[t].rank,offset}; offset=add(offset,cube(spaces[t].rank));
    }
    if (offset!=p.total_amplitude_elements) throw std::logic_error("local triples amplitude census mismatch");
    const auto o=p.n_occupied;
    bool has_gap=false;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k) {
        const Triple ijk{i,j,k}; const auto& s=spaces[index(ijk,o)]; const auto r=s.rank;
        for (U a=0;a<r;++a) for (U b=0;b<r;++b) for (U c=0;c<r;++c) {
            const auto value=gap(s,{a,b,c},ijk,input,options.denominator_floor);
            if (!has_gap) { result.minimum_denominator_=value; has_gap=true; }
            result.minimum_denominator_=std::min(result.minimum_denominator_,value);
            result.maximum_denominator_=std::max(result.maximum_denominator_,value);
        }
    }
    double previous_energy=0,update_defect=0;
    U visits=0,neighbour_visits=0;
    for (U iteration=1;iteration<=options.maximum_iterations;++iteration) {
        Progress record; record.iteration=iteration; record.repeated_update_defect_norm=update_defect;
        Sum energy;
        for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k) {
            const Triple ijk{i,j,k}; const auto t=index(ijk,o); const auto& s=spaces[t];
            if (!s.rank) continue;
            for (U axis=0;axis<3;++axis) std::copy_n(input.f_oo.data+ijk[axis]*o,o,rows+axis*o);
            Neighbours neighbours{input,result,s,iteration,neighbour_scratch};
            const Request request{ijk,s.rank,input.ccsd_snapshot_id};
            MomentReceiver receiver{request,input,options,s,neighbours,
                result.amplitudes_.data()+result.records_[t].offset,candidate.data()+result.records_[t].offset,
                rows,moment_digests[t],iteration};
            if (visits>=p.moment_visits_upper_bound) throw std::length_error("local triples exhausted moment visit cap");
            ++visits; // Charge BEFORE calling the opaque producer.
            check_input();
            provider.visit(request,&MomentReceiver::receive,&receiver,provider.context);
            check_input();
            if (receiver.visits!=1 || receiver.failed)
                throw std::runtime_error("local triples moment producer violated exactly-once protocol");
            neighbour_visits=add(neighbour_visits,receiver.neighbour_visits);
            if (neighbour_visits>p.neighbour_visits_upper_bound)
                throw std::logic_error("local triples neighbour visit census exceeded");
            const U multiplicity=i==k ? 1 : (i==j || j==k ? 3 : 6);
            record.maximum_absolute_residual=std::max(record.maximum_absolute_residual,receiver.max_residual);
            record.residual_frobenius_norm=finite(std::hypot(record.residual_frobenius_norm,
                finite(std::sqrt(static_cast<double>(multiplicity))*receiver.norm)));
            record.maximum_repeated_moment_defect_norm=std::max(record.maximum_repeated_moment_defect_norm,receiver.moment_defect);
            energy.include((2-static_cast<int>(i==j)-static_cast<int>(j==k))*receiver.energy);
        }
        record.moment_visits=visits; record.neighbour_visits=neighbour_visits;
        record.triples_energy=energy.total(); record.has_previous_energy=iteration>1;
        record.energy_change=record.has_previous_energy ? finite(record.triples_energy-previous_energy) : 0;
        record.converged=record.has_previous_energy && record.maximum_absolute_residual<=options.residual_tolerance
            && std::abs(record.energy_change)<=options.energy_tolerance;
        result.snapshot_=record;
        if (progress) progress(record,progress_context);
        check_input();
        if (record.converged || iteration==options.maximum_iterations) {
            Digest payload("vibeqc.bounded-local-triples.output");
            payload.integer(input.ccsd_snapshot_id); payload.integer(iteration);
            payload.number(record.triples_energy); payload.integer(record.converged);
            for (const auto& r:result.records_) { payload.integer(r.rank); payload.integer(r.offset); }
            payload.values({result.amplitudes_.data(),result.amplitudes_.size()});
            result.payload_=payload.h.finish_hex();
            return result;
        }
        // Candidate contains RAW residuals until the evaluated-snapshot
        // stopping decision. Never compute an unnecessary overflowing step.
        for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k) {
            const Triple ijk{i,j,k}; const auto t=index(ijk,o); const auto& s=spaces[t];
            const auto r=s.rank,start=result.records_[t].offset;
            for (U a=0;a<r;++a) for (U b=0;b<r;++b) for (U c=0;c<r;++c) {
                const auto x=start+(a*r+b)*r+c;
                candidate[x]=finite(result.amplitudes_[x]-candidate[x]/gap(s,{a,b,c},ijk,input,options.denominator_floor));
            }
            if (r) update_defect=finite(std::hypot(update_defect,
                symmetrize(candidate.data()+start,candidate.data()+start,r,ijk,true)));
        }
        if (update_defect>options.maximum_repeated_update_defect_norm)
            throw std::invalid_argument("local triples repeated-axis update projection exceeds cumulative budget");
        previous_energy=record.triples_energy; result.amplitudes_.swap(candidate);
    }
    throw std::logic_error("local triples exited without an evaluated snapshot");
}

} // namespace vibeqc
