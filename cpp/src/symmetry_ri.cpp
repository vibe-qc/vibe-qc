#include "vibeqc/symmetry_ri.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using I = std::int64_t;
using Z = std::complex<double>;
using Matrix = Eigen::Matrix<Z,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using ConstMap = Eigen::Map<const Matrix>;
using Map = Eigen::Map<Matrix>;
U add(U a,U b) {
    if (b>std::numeric_limits<U>::max()-a) throw std::overflow_error("RI block count overflow");
    return a+b;
}
U mul(U a,U b) {
    if (a && b>std::numeric_limits<U>::max()/a) throw std::overflow_error("RI block count overflow");
    return a*b;
}
void require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
void finite(Z z) {
    require(std::isfinite(z.real()) && std::isfinite(z.imag()),"nonfinite RI block payload or arithmetic");
}
template<class T> void pointer(const T* p,U count) {
    require(!count || (p && reinterpret_cast<std::uintptr_t>(p)%alignof(T)==0),"unaligned RI block payload");
}
void admit(const SymmetryPlan& plan,const SymmetryBudget& budget) {
    require(budget.maximum_bytes && budget.maximum_work,"RI blocks require positive budgets");
    if (plan.bytes>budget.maximum_bytes) throw std::length_error("RI block byte budget exceeded");
    if (plan.work>budget.maximum_work) throw std::length_error("RI block work budget exceeded");
    if (plan.bytes>static_cast<U>(std::numeric_limits<Eigen::Index>::max()))
        throw std::overflow_error("RI block address extent overflow");
}
}

SymmetryPlan plan_symmetry_ri_blocks(U n,U rank,U groups,U blocks,U values,const SymmetryBudget& budget) {
    require(n && rank && groups,"RI blocks require nonempty coordinate spaces");
    const U square=mul(n,n), cube=mul(square,n);
    require(blocks<=mul(mul(groups,groups),groups) && values<=mul(rank,square),
        "RI block counts exceed coordinate extents");
    // Borrowed + owned packed factors, descriptors and group indices; one
    // auxiliary panel for exchange, density, outputs and Eigen work buffers.
    // No rank*n*n intermediate or complete dense factor reconstruction.
    const U bytes=add(65536,add(mul(64,add(add(groups,rank),1)),
        add(mul(256,blocks),add(mul(32,values),mul(128,square)))));
    // Arbitrary nonoverlapping layouts need not obey an Abelian selection
    // rule. The panel-pair schedule can repeat B*D for several right blocks;
    // values*n*n bounds that work without assuming one partner per irrep.
    const U work=mul(128,add(mul(add(rank,1),cube),
        add(mul(values,square),add(mul(blocks,blocks),add(values,groups)))));
    SymmetryPlan plan{bytes,work}; admit(plan,budget); return plan;
}

SymmetryRIBlocks::SymmetryRIBlocks(U n,U rank,U groups,U nb,U nv,
    const I* orbital,const I* auxiliary,const I* keys,const Z* factors,const SymmetryBudget& budget)
    : n_(n),rank_(rank),groups_(groups),memory_(plan_symmetry_ri_blocks(n,rank,groups,nb,nv,budget)) {
    pointer(orbital,groups+1); pointer(auxiliary,groups+1);
    pointer(keys,3*nb); pointer(factors,nv);
    require(orbital[0]==0 && orbital[groups]==static_cast<I>(n) &&
        auxiliary[0]==0 && auxiliary[groups]==static_cast<I>(rank),"RI offsets must partition their coordinate spaces");
    for (U g=0;g<groups;++g)
        require(orbital[g]>=0 && orbital[g+1]>=orbital[g] && auxiliary[g]>=0 && auxiliary[g+1]>=auxiliary[g],
            "RI offsets must be nondecreasing");
    U consumed=0;
    // Validate the complete descriptor and payload before allocating copies.
    for (U b=0;b<nb;++b) {
        for (U i=0;i<3;++i) require(keys[3*b+i]>=0 && static_cast<U>(keys[3*b+i])<groups,"RI block key out of range");
        const U p=keys[3*b],q=keys[3*b+1],a=keys[3*b+2];
        const U extent=mul(mul(orbital[p+1]-orbital[p],orbital[q+1]-orbital[q]),auxiliary[a+1]-auxiliary[a]);
        require(extent>0,"RI blocks must have nonempty axes");
        consumed=add(consumed,extent);
        require(consumed<=nv,"RI packed factor extent mismatch");
        for (U previous=0;previous<b;++previous)
            require(keys[3*previous]!=static_cast<I>(p) || keys[3*previous+1]!=static_cast<I>(q) ||
                keys[3*previous+2]!=static_cast<I>(a),"duplicate RI block key");
    }
    require(consumed==nv,"RI packed factor extent mismatch");
    for (U i=0;i<nv;++i) finite(factors[i]);
    orbital_.assign(orbital,orbital+groups+1); auxiliary_.assign(auxiliary,auxiliary+groups+1);
    if (nv) factors_.assign(factors,factors+nv);
    blocks_.reserve(nb); by_auxiliary_.resize(groups);
    U offset=0;
    for (U b=0;b<nb;++b) {
        const U p=keys[3*b],q=keys[3*b+1],a=keys[3*b+2];
        blocks_.push_back({p,q,a,offset}); by_auxiliary_[a].push_back(b);
        offset+=(orbital_[p+1]-orbital_[p])*(orbital_[q+1]-orbital_[q])*(auxiliary_[a+1]-auxiliary_[a]);
    }
}

SymmetryRIResult SymmetryRIBlocks::contract(const Z* density,bool make_j,bool make_k,const SymmetryBudget& budget) const {
    admit(memory_,budget); // Before reading density or allocating outputs.
    require(make_j || make_k,"RI contraction requires J or K");
    pointer(density,n_*n_);
    for (U i=0;i<n_*n_;++i) finite(density[i]);
    ConstMap d(density,n_,n_);
    SymmetryRIResult result;
    if (make_j) result.j.resize(n_*n_);
    if (make_k) result.k.resize(n_*n_);
    // Melega et al., DOI 10.1021/acs.jpca.5c08579, Eqs. (3)-(5), and
    // Kitsaras/Stopkowicz arXiv:2602.12457v1, Eqs. (16)-(17):
    // J = sum_A B_A tr(B_A^dagger D), K = sum_A B_A D B_A^dagger.
    // D can be complex and nonsymmetric. Spatial conjugate partners do not
    // authorize conjugating D or applying antiunitary time reversal.
    for (U c=0;c<groups_;++c) {
        const U na=auxiliary_[c+1]-auxiliary_[c];
        const auto& group=by_auxiliary_[c];
        if (make_j && !group.empty()) {
            std::vector<Z> rho(na);
            for (U index:group) {
                const auto& b=blocks_[index];
                const U p0=orbital_[b.p],q0=orbital_[b.q],np=orbital_[b.p+1]-p0,nq=orbital_[b.q+1]-q0;
                for (U a=0;a<na;++a) for (U p=0;p<np;++p) for (U q=0;q<nq;++q)
                    rho[a]+=std::conj(factors_[b.offset+(a*np+p)*nq+q])*d(p0+p,q0+q);
                result.scalar_products+=na*np*nq; result.block_contractions+=na;
            }
            for (Z r:rho) finite(r);
            for (U index:group) {
                const auto& b=blocks_[index];
                const U p0=orbital_[b.p],q0=orbital_[b.q],np=orbital_[b.p+1]-p0,nq=orbital_[b.q+1]-q0;
                for (U a=0;a<na;++a) for (U p=0;p<np;++p) for (U q=0;q<nq;++q)
                    result.j[(p0+p)*n_+q0+q]+=rho[a]*factors_[b.offset+(a*np+p)*nq+q];
                result.scalar_products+=na*np*nq; result.block_contractions+=na;
            }
        }
        if (make_k) for (U li:group) for (U ri:group) {
            const auto& left=blocks_[li]; const auto& right=blocks_[ri];
            const U p0=orbital_[left.p],q0=orbital_[left.q],s0=orbital_[right.p],r0=orbital_[right.q];
            const U np=orbital_[left.p+1]-p0,nq=orbital_[left.q+1]-q0;
            const U ns=orbital_[right.p+1]-s0,nr=orbital_[right.q+1]-r0;
            Matrix temporary(np,nr);
            Map k(result.k.data(),n_,n_);
            for (U a=0;a<na;++a) {
                ConstMap l(factors_.data()+left.offset+a*np*nq,np,nq);
                ConstMap r(factors_.data()+right.offset+a*ns*nr,ns,nr);
                temporary.noalias()=l*d.block(q0,r0,nq,nr);
                require(temporary.allFinite(),"nonfinite RI block exchange intermediate");
                k.block(p0,s0,np,ns).noalias()+=temporary*r.adjoint();
            }
            result.scalar_products+=na*np*nr*(nq+ns); result.block_contractions+=2*na;
        }
    }
    for (Z x:result.j) finite(x);
    for (Z x:result.k) finite(x);
    return result;
}
} // namespace vibeqc
