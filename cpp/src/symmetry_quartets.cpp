#include "vibeqc/symmetry_quartets.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using I = std::int64_t;
using Z = std::complex<double>;
constexpr std::array<std::array<int,4>,8> permutations{{
    {{0,1,2,3}}, {{1,0,2,3}}, {{0,1,3,2}}, {{1,0,3,2}},
    {{2,3,0,1}}, {{3,2,0,1}}, {{2,3,1,0}}, {{3,2,1,0}}
}};
U add(U a,U b) {
    if (b>std::numeric_limits<U>::max()-a) throw std::overflow_error("quartet count overflow");
    return a+b;
}
U mul(U a,U b) {
    if (a && b>std::numeric_limits<U>::max()/a) throw std::overflow_error("quartet count overflow");
    return a*b;
}
U fourth(U n) { return mul(mul(n,n),mul(n,n)); }
U target_count(U count,U nd) { return count==std::numeric_limits<U>::max()?nd:count; }
void finite(double v) {
    if (!std::isfinite(v)) throw std::invalid_argument("nonfinite quartet payload or arithmetic");
}
void finite(Z v) { finite(v.real()); finite(v.imag()); }
void require(bool condition,const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class T> void pointer(const T* p) {
    require(p && reinterpret_cast<std::uintptr_t>(p)%alignof(T)==0,"unaligned quartet payload");
}

template<bool Coulomb>
void contract_selected(const std::array<U,4>& shape,const std::array<int,4>& permutation,
    const std::array<U,4>& origin,U n,const double* tile,const Z* density,Z* target,
    const std::vector<U>& channels) {
    // Dupuis-King shell transport has already produced this member's real
    // tile. Resolve the requested contraction's two matrix addresses once,
    // outside the scalar loop, without materializing a permuted tile.
    // J[pq] = E[pqrs] D[sr]; K[ps] = E[pqrs] D[qr]. No density symmetry is used.
    constexpr int output_column=Coulomb?1:3, density_row=Coulomb?3:1;
    std::array<U,4> output_stride{}, density_stride{};
    output_stride[permutation[0]]=n;
    output_stride[permutation[output_column]]=1;
    density_stride[permutation[density_row]]=n;
    density_stride[permutation[2]]=1;
    const U output_base=origin[0]*n+origin[output_column];
    const U density_base=origin[density_row]*n+origin[2];
    const U panel=n*n;
    U index=0;
    for (U a=0;a<shape[0];++a) for (U b=0;b<shape[1];++b) {
        const U output_ab=output_base+a*output_stride[0]+b*output_stride[1];
        const U density_ab=density_base+a*density_stride[0]+b*density_stride[1];
        for (U c=0;c<shape[2];++c) {
            const U output_abc=output_ab+c*output_stride[2];
            const U density_abc=density_ab+c*density_stride[2];
            for (U d=0;d<shape[3];++d) {
                const U output=output_abc+d*output_stride[3];
                const U input=density_abc+d*density_stride[3];
                const double value=tile[index++];
                for (U channel=0;channel<channels.size();++channel) {
                    Z& result=target[channel*panel+output];
                    const Z next=result+value*density[channels[channel]*panel+input];
                    finite(next); result=next;
                }
            }
        }
    }
}
}

SymmetryPlan plan_symmetry_quartet_accumulator(U n,U ns,U ng,U nm,U maximum,const SymmetryBudget& b,U nd,U nj,U nk) {
    require(n && ns && ng && nm && maximum && nd && ns<=n && maximum<=n,
            "positive consistent quartet extents required");
    require(nm>=ns && nm<=mul(ns,mul(maximum,maximum)),"invalid packed quartet matrix extent");
    nj=target_count(nj,nd); nk=target_count(nk,nd);
    const U targets=add(nj,nk);
    require(nj<=nd && nk<=nd && targets>=nd,"each quartet density requires J, K or both targets");
    SymmetryPlan p;
    // Borrowed + owned density/targets/exports, packed actions, tile scratch,
    // metadata and one bounded orbit. Opaque integral engines are not owned.
    p.bytes=add(add(65536,mul(128,nd-1)),add(mul(32,mul(add(nd,targets),mul(n,n))),add(mul(16,mul(ng,nm)),
        add(mul(64,add(ns,1)),add(mul(80,fourth(maximum)),mul(2048,ng))))));
    p.work=add(4096,add(mul(64,mul(ng,nm)),add(mul(256,ns),
        add(mul(add(mul(16,targets),mul(16,mul(ng,add(maximum,1)))),fourth(n)),mul(mul(64,ng),fourth(ns))))));
    require(b.maximum_bytes && b.maximum_work,"quartet execution requires positive budgets");
    if (p.bytes>b.maximum_bytes) throw std::length_error("quartet execution byte budget exceeded");
    if (p.work>b.maximum_work) throw std::length_error("quartet execution work budget exceeded");
    if (p.bytes>std::numeric_limits<std::size_t>::max()) throw std::overflow_error("quartet address extent overflow");
    return p;
}

SymmetryQuartetAccumulator::SymmetryQuartetAccumulator(U n,U ns,U ng,U nm,U maximum,
    const I* offsets,const double* matrices,const Z* density,const SymmetryBudget& b,U nd,U nj,U nk,const std::uint8_t* targets)
    : memory_(plan_symmetry_quartet_accumulator(n,ns,ng,nm,maximum,b,nd,nj,nk)),
      n_(n),shells_(ns),operations_(ng),matrix_elements_(nm),maximum_shell_(maximum) {
    pointer(offsets); pointer(matrices); pointer(density);
    nj=target_count(nj,nd); nk=target_count(nk,nd);
    // Verify target declarations before allocating snapshots. A null selector
    // retains the original all-J/all-K API; every density must be used.
    U actual_j=0, actual_k=0;
    for (U channel=0;channel<nd;++channel) {
        const auto flags=targets?targets[channel]:3;
        require(flags>=1 && flags<=3,"quartet target flags must request J, K or both");
        actual_j+=(flags&1)!=0; actual_k+=(flags&2)!=0;
    }
    require(actual_j==nj && actual_k==nk,"quartet target counts disagree with selectors");
    require(offsets[0]==0 && offsets[ns]==static_cast<I>(n),"quartet shell offsets must partition the density");
    U elements=0, actual_maximum=0;
    for (U s=0;s<ns;++s) {
        require(offsets[s]>=0 && offsets[s+1]>offsets[s],"strictly increasing quartet offsets required");
        const U size=static_cast<U>(offsets[s+1]-offsets[s]);
        actual_maximum=std::max(actual_maximum,size);
        elements=add(elements,mul(size,size));
    }
    require(actual_maximum==maximum && elements==nm,"quartet count-only extents disagree with offsets");
    for (U i=0;i<mul(ng,nm);++i) finite(matrices[i]);
    const U panels=mul(nd,mul(n,n));
    for (U i=0;i<panels;++i) finite(density[i]);
    offsets_.assign(offsets,offsets+ns+1);
    sizes_.resize(ns); matrix_offsets_.resize(ns);
    elements=0;
    for (U s=0;s<ns;++s) {
        sizes_[s]=offsets_[s+1]-offsets_[s];
        matrix_offsets_[s]=elements; elements+=sizes_[s]*sizes_[s];
    }
    matrices_.assign(matrices,matrices+ng*nm);
    density_.assign(density,density+panels);
    j_.resize(mul(nj,mul(n,n))); k_.resize(mul(nk,mul(n,n)));
    j_channels_.reserve(nj); k_channels_.reserve(nk);
    for (U channel=0;channel<nd;++channel) {
        const auto flags=targets?targets[channel]:3;
        if (flags&1) j_channels_.push_back(channel);
        if (flags&2) k_channels_.push_back(channel);
    }
    first_.resize(fourth(maximum)); second_.resize(fourth(maximum));
    identity_blocks_.assign(ng*ns,1);
    for (U g=0;g<ng;++g) for (U s=0;s<ns;++s) {
        const U size=sizes_[s];
        const double* w=matrices_.data()+g*nm+matrix_offsets_[s];
        for (U a=0;a<size;++a) for (U v=0;v<size;++v)
            if (w[a*size+v]!=(a==v?1.:0.)) identity_blocks_[g*ns+s]=0;
    }
}

void SymmetryQuartetAccumulator::accumulate(const std::array<I,4>& representative,
    const std::array<U,4>& shape,const double* tile,U members,const I* images) {
    require(valid_ && !finished_,"quartet accumulator is failed or finished");
    // Any native failure poisons the private partial targets.
    valid_=false;
    pointer(tile); pointer(images);
    require(members && members<=mul(8,operations_),"quartet member extent exceeds one orbit");
    U elements=1;
    for (int axis=0;axis<4;++axis) {
        require(representative[axis]>=0 && static_cast<U>(representative[axis])<shells_,"quartet representative index");
        require(shape[axis]==sizes_[representative[axis]],"quartet tile shape mismatch");
        elements=mul(elements,shape[axis]);
    }
    U extra_transport=0, previous=operations_;
    for (U member=0;member<members;++member) {
        const I* m=images+6*member;
        require(m[0]>=0 && static_cast<U>(m[0])<operations_ && m[1]>=0 && m[1]<8,"quartet operation or permutation index");
        const U g=static_cast<U>(m[0]);
        for (int axis=0;axis<4;++axis) {
            require(m[axis+2]>=0 && static_cast<U>(m[axis+2])<shells_,"quartet image shell index");
            require(sizes_[m[axis+2]]==shape[permutations[m[1]][axis]],"quartet image shell extent mismatch");
            if (g!=previous && !identity_blocks_[g*shells_+representative[axis]])
                extra_transport=add(extra_transport,mul(elements,shape[axis]));
        }
        previous=g;
    }
    const U targets=add(j_channels_.size(),k_channels_.size());
    const U channels=density_.size()/(n_*n_);
    const bool all_targets=j_channels_.size()==channels && k_channels_.size()==channels;
    const U extra_blocks=mul(targets,members);
    const U extra_scalar=mul(elements,extra_blocks);
    require(add(counts_.block_contractions,extra_blocks)<=mul(targets,fourth(shells_)),"quartet build member census exceeded");
    require(add(counts_.scalar_products,extra_scalar)<=mul(targets,fourth(n_)),"quartet build contraction census exceeded");
    const U transport=add(counts_.transport_scalar_products,extra_transport);
    require(add(add(counts_.scalar_products,extra_scalar),transport)<=memory_.work,"quartet cumulative work exceeded");
    for (U i=0;i<elements;++i) finite(tile[i]);

    const double* rotated=tile;
    previous=operations_;
    for (U member=0;member<members;++member) {
        const I* m=images+6*member;
        const U g=static_cast<U>(m[0]);
        if (g!=previous) {
            rotated=tile;
            // E_target = W* W W* W E_source, W=U^-1. W is real here.
            // Never substitute U^T for the metric-aware inverse action.
            for (int axis=0;axis<4;++axis) {
                const U s=representative[axis], size=shape[axis];
                if (identity_blocks_[g*shells_+s]) continue;
                double* target=rotated==first_.data()?second_.data():first_.data();
                const double* w=matrices_.data()+g*matrix_elements_+matrix_offsets_[s];
                U stride=1;
                for (int k=axis+1;k<4;++k) stride*=shape[k];
                const U outer=elements/(size*stride);
                for (U o=0;o<outer;++o) for (U v=0;v<size;++v) for (U t=0;t<stride;++t) {
                    double sum=0.;
                    for (U a=0;a<size;++a) sum+=rotated[(o*size+a)*stride+t]*w[a*size+v];
                    finite(sum); target[(o*size+v)*stride+t]=sum;
                }
                rotated=target;
            }
            previous=g;
        }
        const auto& p=permutations[m[1]];
        if (!all_targets) {
            // Channel/family selection is per member, not per integral value.
            // Each output still receives contributions in the original order.
            const std::array<U,4> origin{{offsets_[m[2]],offsets_[m[3]],offsets_[m[4]],offsets_[m[5]]}};
            const U panel=n_*n_;
            if (elements==1) {
                // Scalar shells need no four-axis loop or stride inventory.
                const U oj=origin[0]*n_+origin[1], dj=origin[3]*n_+origin[2];
                const U ok=origin[0]*n_+origin[3], dk=origin[1]*n_+origin[2];
                for (U target=0;target<j_channels_.size();++target) {
                    Z& value=j_[target*panel+oj];
                    const Z next=value+rotated[0]*density_[j_channels_[target]*panel+dj];
                    finite(next); value=next;
                }
                for (U target=0;target<k_channels_.size();++target) {
                    Z& value=k_[target*panel+ok];
                    const Z next=value+rotated[0]*density_[k_channels_[target]*panel+dk];
                    finite(next); value=next;
                }
            } else {
                if (!j_channels_.empty())
                    contract_selected<true>(shape,p,origin,n_,rotated,
                        density_.data(),j_.data(),j_channels_);
                if (!k_channels_.empty())
                    contract_selected<false>(shape,p,origin,n_,rotated,
                        density_.data(),k_.data(),k_channels_);
            }
            continue;
        }
        U index=0;
        for (U a=0;a<shape[0];++a) for (U b=0;b<shape[1];++b)
        for (U c=0;c<shape[2];++c) for (U d=0;d<shape[3];++d) {
            const std::array<U,4> x{{a,b,c,d}};
            const U i=offsets_[m[2]]+x[p[0]], j=offsets_[m[3]]+x[p[1]];
            const U k=offsets_[m[4]]+x[p[2]], l=offsets_[m[5]]+x[p[3]];
            const double v=rotated[index++];
            // Independent spin/response channels share the transported tile.
            // A failure in any channel invalidates the whole private batch.
            // Preserve the paired loop for the existing default request.
            for (U channel=0;channel<channels;++channel) {
                const U base=channel*n_*n_;
                const Z next_j=j_[base+i*n_+j]+v*density_[base+l*n_+k];
                const Z next_k=k_[base+i*n_+l]+v*density_[base+j*n_+k];
                finite(next_j); finite(next_k);
                j_[base+i*n_+j]=next_j; k_[base+i*n_+l]=next_k;
            }
        }
    }
    counts_.representatives=add(counts_.representatives,1);
    counts_.scalar_products=add(counts_.scalar_products,extra_scalar);
    counts_.block_contractions=add(counts_.block_contractions,extra_blocks);
    counts_.transport_scalar_products=transport;
    valid_=true;
}

SymmetryQuartetTargets SymmetryQuartetAccumulator::finish() {
    require(valid_ && !finished_,"quartet accumulator is failed or finished");
    finished_=true;
    return {n_,j_channels_.size(),k_channels_.size(),std::move(j_),std::move(k_),counts_};
}
} // namespace vibeqc
