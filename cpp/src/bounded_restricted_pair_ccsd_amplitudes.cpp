#include "vibeqc/bounded_restricted_pair_ccsd_amplitudes.hpp"

#include <algorithm>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {
using I=std::uint64_t;
using Input=BoundedRestrictedPairCCSDAmplitudesInput;
using Options=BoundedRestrictedPairCCSDAmplitudesOptions;
using Inventory=BoundedRestrictedPairCCSDAmplitudesInventory;
using Caps=BoundedRestrictedPairCCSDAmplitudesCaps;
using Plan=BoundedRestrictedPairCCSDAmplitudesMemoryPlan;
using Reader=BoundedRestrictedPairCCSDAmplitudes;
using Diagnostics=BoundedRestrictedPairCCSDAmplitudesDiagnostics;
constexpr char policy[]="restricted-connected-alpha-beta;separate-singles;unordered-pair-transpose;diagonal-common-upper;role-conservative-borrowed;no-physical-certificate";
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559 && FLT_EVAL_METHOD==0,
    "ragged CCSD amplitudes require IEEE binary64 evaluation");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "ragged CCSD amplitudes do not support fast-math"
#endif
I add(I a,I b) {
    if(b>std::numeric_limits<I>::max()-a) throw std::overflow_error("ragged CCSD count sum overflow");
    return a+b;
}
I mul(I a,I b) {
    if(a && b>std::numeric_limits<I>::max()/a) throw std::overflow_error("ragged CCSD count product overflow");
    return a*b;
}
I triangular(I o) { return o%2 ? mul(o,add(o,1)/2) : mul(o/2,add(o,1)); }
void cap(I x,I bound,const char* message) { if(!bound || x>bound) throw std::length_error(message); }
double finite(double x) {
    if(!std::isfinite(x)) throw std::overflow_error("ragged CCSD amplitude arithmetic is non-finite");
    return x;
}
void environment() {
    if(std::fegetround()!=FE_TONEAREST) throw std::invalid_argument("ragged CCSD amplitudes require round-to-nearest");
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1.0,zero=0.0;
    if(!(tiny>0) || tiny*one!=tiny || tiny+zero!=tiny || std::fma(tiny,one,zero)!=tiny)
        throw std::invalid_argument("ragged CCSD amplitudes require gradual underflow");
}
struct Sum {
    double sum=0,correction=0;
    void include(double x) {
        finite(x); const auto next=finite(sum+x);
        const auto tail=std::abs(sum)>=std::abs(x) ? finite(finite(sum-next)+x) : finite(finite(x-next)+sum);
        correction=finite(correction+tail); sum=next;
    }
    double value() const { return finite(sum+correction); }
};
class Hash {
public:
    explicit Hash(const char* domain) {
        text(domain); const std::uint8_t version[4]={0,0,0,1}; hash_.update(version,4);
    }
    void u64(I value) {
        std::uint8_t wire[8]; for(unsigned i=0;i<8;++i) wire[i]=value>>(56-8*i); hash_.update(wire,8);
    }
    void real(double value) {
        finite(value); if(value==0) value=0;
        I bits; std::memcpy(&bits,&value,8); u64(bits);
    }
    void text(const char* value) {
        const auto n=std::strlen(value); u64(n); hash_.update(reinterpret_cast<const std::uint8_t*>(value),n);
    }
    std::array<char,64> finish() {
        const auto value=hash_.finish_hex(); std::array<char,64> out{};
        std::copy(value.begin(),value.end(),out.begin()); return out;
    }
private:
    detail::Sha256 hash_;
};
void extent(I bytes) {
    if(bytes>static_cast<I>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes>std::numeric_limits<I>::max()/8-65536)
        throw std::length_error("ragged CCSD numerical or SHA address extent exceeded");
}
template<class T> void pointer(const T* p,I count,const char* message) {
    const auto bytes=mul(sizeof(T),count); extent(bytes);
    if(!count) return;
    const auto address=reinterpret_cast<std::uintptr_t>(p);
    if(!p || address%alignof(T) || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
        throw std::invalid_argument(message);
}
void view(BoundedRestrictedCCSDRealView v,I count) {
    if(v.element_count!=count) throw std::invalid_argument("ragged CCSD numerical view has wrong exact extent");
    pointer(v.data,count,"ragged CCSD numerical view must be aligned and accessible");
}
void controls(const Options& o,const Inventory& inventory) {
    if(!std::isfinite(o.coefficient_orthogonality_tolerance)
        || o.coefficient_orthogonality_tolerance<=0 || o.coefficient_orthogonality_tolerance>=1)
        throw std::invalid_argument("ragged CCSD coefficient orthogonality tolerance must be explicit in (0,1)");
    if(!inventory.numerical_replicas || !inventory.fixed_backend_margin_bytes_per_replica)
        throw std::invalid_argument("ragged CCSD replica count and backend margin must be positive");
}
Plan base(I o,I n,const Inventory& inventory) {
    if(!o || !n) throw std::invalid_argument("ragged CCSD occupied/common virtual dimensions must be positive");
    if(!inventory.numerical_replicas || !inventory.fixed_backend_margin_bytes_per_replica)
        throw std::invalid_argument("ragged CCSD replica count and backend margin must be positive");
    Plan p; p.n_occupied=o; p.common_virtual_dimension=n; p.pair_count=triangular(o);
    p.borrowed_singles_table_bytes=mul(o,sizeof(BoundedRestrictedPairCCSDSinglesView));
    p.borrowed_pair_table_bytes=mul(p.pair_count,sizeof(BoundedRestrictedPairCCSDPairView));
    p.borrowed_table_bytes=add(p.borrowed_singles_table_bytes,p.borrowed_pair_table_bytes);
    p.fixed_control_storage_bytes=16384+sizeof(Reader)+sizeof(Input)+sizeof(Plan)+sizeof(Options)
        +sizeof(Inventory)+sizeof(Caps)+sizeof(Diagnostics);
    p.metadata_work_units=add(256,mul(128,add(o,p.pair_count)));
    p.numerical_replicas=inventory.numerical_replicas; p.external_node_bytes=inventory.external_node_bytes;
    extent(p.borrowed_table_bytes); return p;
}
void finish(Plan& p,const Inventory& inventory,I gram_work) {
    p.borrowed_singles_numerical_bytes=mul(8,add(p.singles_coefficient_elements,p.singles_amplitude_elements));
    p.borrowed_pair_numerical_bytes=mul(8,add(p.pair_coefficient_elements,p.pair_amplitude_elements));
    p.borrowed_numerical_bytes=add(p.borrowed_singles_numerical_bytes,p.borrowed_pair_numerical_bytes);
    p.validation_work_units=add(p.metadata_work_units,add(4096,mul(256,
        add(add(p.borrowed_numerical_bytes/8,gram_work),add(p.pair_amplitude_elements,1)))));
    p.maximum_singles_work_units_per_query=mul(32,add(p.maximum_singles_rank,1));
    p.maximum_doubles_work_units_per_query=mul(64,add(add(mul(p.maximum_pair_rank,p.maximum_pair_rank),p.maximum_pair_rank),1));
    p.per_replica_inventoried_bytes=add(p.borrowed_numerical_bytes,add(p.borrowed_table_bytes,
        add(p.fixed_control_storage_bytes,add(inventory.other_live_bytes_per_replica,inventory.fixed_backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes=add(inventory.external_node_bytes,mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    extent(p.borrowed_numerical_bytes);
}
void count_caps(const Plan& p,const Caps& c) {
    cap(p.n_occupied,c.maximum_occupied_count,"ragged CCSD occupied count cap exceeded");
    cap(p.common_virtual_dimension,c.maximum_common_virtual_dimension,"ragged CCSD common dimension cap exceeded");
    cap(p.pair_count,c.maximum_pair_count,"ragged CCSD pair count cap exceeded");
    cap(p.borrowed_table_bytes,c.maximum_table_bytes,"ragged CCSD table byte cap exceeded");
    cap(p.metadata_work_units,c.maximum_validation_work_units,"ragged CCSD metadata work cap exceeded");
}
void exact_caps(const Plan& p,const Caps& c) {
    if(p.borrowed_numerical_bytes>c.maximum_borrowed_numerical_bytes)
        throw std::length_error("ragged CCSD borrowed numerical cap exceeded");
    cap(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"ragged CCSD replica memory cap exceeded");
    cap(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"ragged CCSD node memory cap exceeded");
    cap(p.validation_work_units,c.maximum_validation_work_units,"ragged CCSD validation work cap exceeded");
    cap(p.maximum_singles_work_units_per_query,c.maximum_singles_work_units_per_query,"ragged CCSD singles query work cap exceeded");
    cap(p.maximum_doubles_work_units_per_query,c.maximum_doubles_work_units_per_query,"ragged CCSD doubles query work cap exceeded");
}
std::array<char,64> descriptors(const Input& input) {
    Hash h("vibeqc.bounded-restricted-pair-ccsd-amplitudes.descriptors");
    h.u64(input.n_occupied); h.u64(input.common_virtual_dimension);
    h.u64(reinterpret_cast<std::uintptr_t>(input.singles)); h.u64(input.singles_count);
    h.u64(reinterpret_cast<std::uintptr_t>(input.pairs)); h.u64(input.pair_count);
    const auto record=[&h](const auto& p) {
        h.u64(p.rank);
        for(const auto& v:{p.coefficients,p.amplitudes}) {
            h.u64(v.element_count); h.u64(v.element_count ? reinterpret_cast<std::uintptr_t>(v.data) : 0);
        }
    };
    for(I i=0;i<input.singles_count;++i) record(input.singles[i]);
    for(I p=0;p<input.pair_count;++p) record(input.pairs[p]);
    return h.finish();
}
void orthogonal(const double* c,I n,I r,double tolerance,Diagnostics& diagnostics) {
    for(I a=0;a<r;++a) for(I b=0;b<r;++b) {
        Sum sum;
        for(I k=0;k<n;++k) sum.include(finite(c[k*r+a]*c[k*r+b]));
        const auto error=std::abs(finite(sum.value()-double(a==b)));
        diagnostics.maximum_coefficient_orthogonality_error=std::max(diagnostics.maximum_coefficient_orthogonality_error,error);
        if(error>tolerance) throw std::invalid_argument("ragged CCSD coefficient columns are not orthonormal within explicit tolerance");
    }
}
std::array<char,64> validate(const Input& input,const Options& options,Diagnostics& diagnostics) {
    environment(); Hash h("vibeqc.bounded-restricted-pair-ccsd-amplitudes.snapshot");
    h.u64(input.n_occupied); h.u64(input.common_virtual_dimension);
    const auto record=[&](const auto& p) {
        h.u64(p.rank);
        for(const auto& v:{p.coefficients,p.amplitudes}) {
            h.u64(v.element_count); for(I x=0;x<v.element_count;++x) h.real(v.data[x]);
        }
        orthogonal(p.coefficients.data,input.common_virtual_dimension,p.rank,
            options.coefficient_orthogonality_tolerance,diagnostics);
    };
    for(I i=0;i<input.singles_count;++i) { record(input.singles[i]); ++diagnostics.validated_singles; }
    I at=0;
    for(I i=0;i<input.n_occupied;++i) for(I j=i;j<input.n_occupied;++j,++at) {
        const auto& p=input.pairs[at]; record(p);
        if(i==j) for(I a=0;a<p.rank;++a) for(I b=a+1;b<p.rank;++b)
            if(p.amplitudes.data[a*p.rank+b]!=p.amplitudes.data[b*p.rank+a])
                throw std::invalid_argument("ragged CCSD diagonal pair amplitudes must be exactly symmetric");
        ++diagnostics.validated_pairs;
    }
    h.real(options.coefficient_orthogonality_tolerance); h.text(policy); return h.finish();
}
} // namespace

Plan plan_bounded_restricted_pair_ccsd_amplitudes_upper(I o,I n,I s,I r,const Inventory& inventory) {
    auto p=base(o,n,inventory);
    if(s>n || r>n) throw std::invalid_argument("ragged CCSD uniform ranks exceed common dimension");
    p.uniform_rank_upper_bound=true; p.maximum_singles_rank=s; p.maximum_pair_rank=r;
    p.singles_amplitude_elements=mul(o,s); p.pair_amplitude_elements=mul(p.pair_count,mul(r,r));
    p.singles_coefficient_elements=mul(n,p.singles_amplitude_elements);
    p.pair_coefficient_elements=mul(n,mul(p.pair_count,r));
    const auto gram=mul(n,add(mul(o,mul(s,s)),mul(p.pair_count,mul(r,r))));
    finish(p,inventory,gram); return p;
}
Plan plan_bounded_restricted_pair_ccsd_amplitudes(const Input& input,const Options& options,
    const Inventory& inventory,const Caps& caps) {
    controls(options,inventory); auto p=base(input.n_occupied,input.common_virtual_dimension,inventory);
    count_caps(p,caps);
    if(input.singles_count!=p.n_occupied || input.pair_count!=p.pair_count)
        throw std::invalid_argument("ragged CCSD tables require exact singles and unordered pair counts");
    pointer(input.singles,input.singles_count,"ragged CCSD singles table pointer invalid or unaligned");
    pointer(input.pairs,input.pair_count,"ragged CCSD pair table pointer invalid or unaligned");
    I gram=0; const auto n=p.common_virtual_dimension;
    for(I i=0;i<input.singles_count;++i) {
        const auto& s=input.singles[i];
        if(s.rank>n || s.rank>caps.maximum_singles_rank) throw std::length_error("ragged CCSD singles rank cap exceeded");
        const auto nr=mul(n,s.rank); view(s.coefficients,nr); view(s.amplitudes,s.rank);
        p.maximum_singles_rank=std::max(p.maximum_singles_rank,s.rank);
        p.singles_amplitude_elements=add(p.singles_amplitude_elements,s.rank);
        p.singles_coefficient_elements=add(p.singles_coefficient_elements,nr);
        gram=add(gram,mul(n,mul(s.rank,s.rank)));
    }
    for(I i=0;i<input.pair_count;++i) {
        const auto& s=input.pairs[i];
        if(s.rank>n || s.rank>caps.maximum_pair_rank) throw std::length_error("ragged CCSD pair rank cap exceeded");
        const auto nr=mul(n,s.rank),rr=mul(s.rank,s.rank); view(s.coefficients,nr); view(s.amplitudes,rr);
        p.maximum_pair_rank=std::max(p.maximum_pair_rank,s.rank);
        p.pair_amplitude_elements=add(p.pair_amplitude_elements,rr);
        p.pair_coefficient_elements=add(p.pair_coefficient_elements,nr);
        gram=add(gram,mul(n,rr));
    }
    finish(p,inventory,gram); exact_caps(p,caps); return p;
}
Reader make_bounded_restricted_pair_ccsd_amplitudes(const Input& input,const Options& options,
    const Inventory& inventory,const Caps& caps) {
    Reader reader; reader.memory_=plan_bounded_restricted_pair_ccsd_amplitudes(input,options,inventory,caps);
    reader.input_=input; reader.options_=options; reader.inventory_=inventory; reader.caps_=caps;
    reader.descriptors_=descriptors(input); reader.snapshot_=validate(input,options,reader.diagnostics_); return reader;
}
Reader::BoundedRestrictedPairCCSDAmplitudes(Reader&& other) noexcept
    :input_(other.input_),options_(other.options_),inventory_(other.inventory_),caps_(other.caps_),
     memory_(other.memory_),diagnostics_(other.diagnostics_),snapshot_(other.snapshot_),descriptors_(other.descriptors_) {
    other.input_={};
}
double Reader::singles(I i,I a) const {
    if(!input_.n_occupied) throw std::logic_error("ragged CCSD reader is moved or incomplete");
    if(i>=input_.n_occupied || a>=input_.common_virtual_dimension)
        throw std::out_of_range("ragged CCSD singles label out of range");
    const auto& p=input_.singles[i]; Sum sum;
    for(I x=0;x<p.rank;++x) sum.include(finite(p.coefficients.data[a*p.rank+x]*p.amplitudes.data[x]));
    return sum.value();
}
double Reader::doubles(I i,I j,I a,I b) const {
    if(!input_.n_occupied) throw std::logic_error("ragged CCSD reader is moved or incomplete");
    if(i>=input_.n_occupied || j>=input_.n_occupied || a>=input_.common_virtual_dimension || b>=input_.common_virtual_dimension)
        throw std::out_of_range("ragged CCSD doubles label out of range");
    if(i>j) { std::swap(i,j); std::swap(a,b); }
    if(i==j && a>b) std::swap(a,b);
    const auto at=memory_.pair_count-triangular(input_.n_occupied-i)+(j-i);
    const auto& p=input_.pairs[at]; Sum sum;
    for(I x=0;x<p.rank;++x) {
        Sum row;
        for(I y=0;y<p.rank;++y) row.include(finite(p.amplitudes.data[x*p.rank+y]*p.coefficients.data[b*p.rank+y]));
        sum.include(finite(p.coefficients.data[a*p.rank+x]*row.value()));
    }
    return sum.value();
}
BoundedRestrictedCCSDAmplitudeProvider Reader::amplitude_provider() const {
    if(!input_.n_occupied) throw std::logic_error("ragged CCSD reader is moved or incomplete");
    return {[](I i,I a,const void* p){return static_cast<const Reader*>(p)->singles(i,a);},
        [](I i,I j,I a,I b,const void* p){return static_cast<const Reader*>(p)->doubles(i,j,a,b);},this,
        memory_.borrowed_numerical_bytes,0,memory_.maximum_singles_work_units_per_query,memory_.maximum_doubles_work_units_per_query};
}
BoundedRestrictedPairCCSDSinglesView Reader::singles_view(I i) const & {
    if(!input_.n_occupied) throw std::logic_error("ragged CCSD reader is moved or incomplete");
    if(i>=input_.n_occupied) throw std::out_of_range("ragged CCSD singles view label out of range");
    return input_.singles[i];
}
BoundedRestrictedPairCCSDPairView Reader::canonical_pair_view(I i,I j) const & {
    if(!input_.n_occupied) throw std::logic_error("ragged CCSD reader is moved or incomplete");
    if(i>j) throw std::invalid_argument("ragged CCSD contiguous pair view requires i<=j");
    if(j>=input_.n_occupied) throw std::out_of_range("ragged CCSD pair view label out of range");
    const auto at=memory_.pair_count-triangular(input_.n_occupied-i)+(j-i);
    return input_.pairs[at];
}
void Reader::validate_immutable_snapshot() const {
    if(!input_.n_occupied) throw std::logic_error("ragged CCSD reader is moved or incomplete");
    // Validate changed extents under original caps before hashing descriptors
    // or scanning any possibly changed numerical view.
    (void)plan_bounded_restricted_pair_ccsd_amplitudes(input_,options_,inventory_,caps_);
    if(descriptors(input_)!=descriptors_) throw std::invalid_argument("ragged CCSD immutable snapshot descriptors changed");
    Diagnostics diagnostic;
    if(validate(input_,options_,diagnostic)!=snapshot_)
        throw std::invalid_argument("ragged CCSD immutable amplitude snapshot changed");
}

} // namespace vibeqc
