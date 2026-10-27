#include "vibeqc/symmetry_quartets_source.hpp"
#include "vibeqc/manta_provider.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <typeinfo>

namespace vibeqc {
namespace {
using U=std::uint64_t;
using I=std::int64_t;
void require(bool condition,const char* message) {
    if(!condition) throw std::invalid_argument(message);
}
U add(U a,U b) {
    if(b>std::numeric_limits<U>::max()-a) throw std::overflow_error("quartet source count overflow");
    return a+b;
}
U mul(U a,U b) {
    if(a && b>std::numeric_limits<U>::max()/a) throw std::overflow_error("quartet source count overflow");
    return a*b;
}
const manta::IntegralProvider& admitted(const manta::OrderedJKBuilder& builder) {
    const auto& p=builder.provider();
    require(typeid(p)==typeid(manta::MantaProvider) || typeid(p)==typeid(manta::LibintProvider),
            "immutable native molecular quartet provider required");
    const auto r=p.kernel_request();
    require(r.kernel=="coulomb" && r.omega==0. && r.scalar=="real" && r.centers==4 && r.derivative==0 &&
            r.images==decltype(r.images){},"unqualified molecular quartet provider or kernel");
    require(p.plan().provider==p.name() && p.plan().kernel==r.kernel && p.plan().omega==r.omega,
            "quartet provider plan mismatch");
    return p;
}
}

SymmetryPlan plan_symmetry_quartet_source(const manta::OrderedJKBuilder& builder,const SymmetryBudget& b) {
    const auto& p=admitted(builder).plan();
    SymmetryPlan result;
    // Charge a second workspace while the plan's Python audit session stays
    // live. The whole provider reservation conservatively covers its opaque
    // engine; the additional zero tile handles null libint views explicitly.
    result.bytes=add(65536,add(p.modeled_peak_bytes,mul(8,p.tile_elements)));
    const U n2=mul(p.n_basis,p.n_basis);
    result.work=add(4096,add(p.kernel_terms,add(mul(64,p.shell_calls),mul(64,mul(n2,n2)))));
    require(b.maximum_bytes && b.maximum_work,"quartet source execution requires positive budgets");
    if(result.bytes>b.maximum_bytes || result.work>b.maximum_work)
        throw std::length_error("quartet source execution budget exceeded");
    if(result.bytes>std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("quartet source address extent overflow");
    return result;
}

SymmetryQuartetSourceAccumulator::SymmetryQuartetSourceAccumulator(SymmetryQuartetAccumulator& a,
    std::shared_ptr<manta::OrderedJKBuilder> builder,const std::string& source,
    const std::string& expected,U representatives,const SymmetryBudget& budget)
    :accumulator_(a),builder_(std::move(builder)),expected_representatives_(representatives) {
    try {
        require(bool(builder_),"quartet source builder required");
        plan_symmetry_quartet_source(*builder_,budget); // Before workspace/payload allocation.
        const auto& p=builder_->provider();
        require(source==p.source_identity(),"quartet source identity changed; requalify the finite source");
        require(expected.size()==64 && std::all_of(expected.begin(),expected.end(),[](char c) {
            return (c>='0' && c<='9') || (c>='a' && c<='f'); }),"invalid quartet representative digest");
        require(representatives<=p.plan().shell_calls,"quartet representative census exceeds source");
        const auto& offsets=a.shell_offsets();
        require(offsets.size()==p.basis().nshells()+1,"quartet source shell layout mismatch");
        U at=0;
        for(U s=0;s<p.basis().nshells();++s) {
            require(offsets[s]==at,"quartet source shell layout mismatch");
            at=add(at,p.basis().libint()[s].size());
        }
        require(offsets.back()==at && at==p.plan().n_basis,"quartet source AO layout mismatch");
        source_=source; expected_digest_=expected;
        workspace_=std::make_unique<manta::Workspace>(p.plan());
        zeros_.resize(p.plan().tile_elements,0.);
    } catch(...) { invalidate(); throw; }
}

SymmetryQuartetSourceAccumulator::~SymmetryQuartetSourceAccumulator() {
    if(!finished_) invalidate();
}
void SymmetryQuartetSourceAccumulator::invalidate() noexcept {
    valid_=false; accumulator_.invalidate();
}
void SymmetryQuartetSourceAccumulator::consume_orbit(const std::array<I,4>& representative,
    U operations,const I* mapped,U identity,std::uint8_t mode) {
    try {
        require(!support_,"certified quartet maps cannot be overridden");
        consume(representative,operations,mapped,identity,mode);
    } catch(...) { invalidate(); throw; }
}
void SymmetryQuartetSourceAccumulator::bind_support(std::shared_ptr<const manta::MolecularSourceSupport> s,
    U ng,U ns,const I* destinations,U identity,std::uint8_t mode) {
    try {
        require(valid_ && !finished_ && !support_ && work_[0]==0,
            "quartet support must be bound once before evaluation");
        require(bool(s),"quartet source support certificate required");
        require(ng==accumulator_.operations() && ns==builder_->plan().n_shells && identity<ng && mode<=2,
            "quartet support descriptor mismatch");
        s->validate(*builder_); s->validate_maps(ng,ns,destinations);
        for(U shell=0;shell<ns;++shell)
            require(destinations[identity*ns+shell]==static_cast<I>(shell),"quartet support identity map mismatch");
        // Replaces the transient Python (operation,4) map inside the existing
        // accumulator's per-operation metadata reservation; no quartet cache.
        mapped_.resize(mul(4,ng));
        support_=std::move(s); identity_=identity; mode_=mode;
    } catch(...) { invalidate(); throw; }
}
void SymmetryQuartetSourceAccumulator::consume_key(const std::array<I,4>& representative) {
    try {
        require(valid_ && !finished_ && bool(support_),"quartet source support is absent or invalid");
        const U ns=builder_->plan().n_shells,ng=support_->plan().operations;
        for(const auto shell:representative)
            require(shell>=0 && static_cast<U>(shell)<ns,"quartet source shell index");
        const auto& maps=support_->destinations();
        for(U g=0;g<ng;++g) for(U axis=0;axis<4;++axis)
            mapped_[4*g+axis]=maps[g*ns+representative[axis]];
        consume(representative,ng,mapped_.data(),identity_,mode_);
    } catch(...) { invalidate(); throw; }
}
void SymmetryQuartetSourceAccumulator::consume(const std::array<I,4>& representative,
    U operations,const I* mapped,U identity,std::uint8_t mode) {
    try {
        require(valid_ && !finished_,"quartet source accumulator is invalid or finished");
        require(work_[0]<expected_representatives_,"quartet source representative census exceeded");
        const auto& p=builder_->provider();
        require(p.source_identity()==source_,"quartet source identity changed during build");
        std::array<std::size_t,4> key{};
        std::array<U,4> shape{};
        std::array<std::uint8_t,32> encoded{};
        U count=1,primitives=1;
        for(U axis=0;axis<4;++axis) {
            require(representative[axis]>=0 && static_cast<U>(representative[axis])<p.basis().nshells(),
                    "quartet source shell index");
            key[axis]=static_cast<std::size_t>(representative[axis]);
            const auto& shell=p.basis().libint()[key[axis]];
            shape[axis]=shell.size(); count=mul(count,shape[axis]);
            primitives=mul(primitives,shell.alpha.size());
            for(U byte=0;byte<8;++byte) encoded[axis*8+byte]=static_cast<U>(key[axis])>>(56-8*byte);
        }
        const auto tile=p.evaluate(key,*workspace_);
        require(std::equal(shape.begin(),shape.end(),tile.shape.begin()),"quartet source tile shape mismatch");
        require(count<=zeros_.size(),"quartet source tile capacity mismatch");
        const auto* values=tile.values?tile.values:zeros_.data();
        for(U i=0;i<count;++i) require(std::isfinite(values[i]),"nonfinite quartet source tile");
        // Match Python _digest_tile: big-endian int64 key, native float64
        // tile bytes. Consume the borrowed view before its next evaluation.
        digest_.update(encoded.data(),encoded.size());
        digest_.update(reinterpret_cast<const std::uint8_t*>(values),mul(count,sizeof(double)));
        accumulator_.accumulate_orbit(representative,shape,values,operations,mapped,identity,mode);
        work_[0]=add(work_[0],1); work_[1]=add(work_[1],1);
        work_[2]=add(work_[2],primitives); work_[3]=add(work_[3],tile.values?0:1);
        work_[4]=add(work_[4],count);
    } catch(...) { invalidate(); throw; }
}
SymmetryQuartetTargets SymmetryQuartetSourceAccumulator::finish() {
    try {
        require(valid_ && !finished_,"quartet source accumulator is invalid or finished");
        require(work_[0]==expected_representatives_,"quartet source representative census mismatch");
        require(builder_->provider().source_identity()==source_,"quartet source identity changed during build");
        require(digest_.finish_hex()==expected_digest_,"quartet representative changed; requalify the finite source");
        auto result=accumulator_.finish(); finished_=true; return result;
    } catch(...) { invalidate(); throw; }
}
} // namespace vibeqc
