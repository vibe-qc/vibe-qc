#include "vibeqc/manta_shells.hpp"
#include <cmath>
#include <map>
#include <algorithm>
#include <stdexcept>

namespace vibeqc::manta::md {
namespace {
int angular(const libint2::Shell* s) { return s?s->contr[0].l:0; }
std::size_t primitives(const libint2::Shell* s) { return s?s->alpha.size():1; }
double exponent(const libint2::Shell* s,std::size_t p) { return s?s->alpha[p]:0.; }
double coefficient(const libint2::Shell* s,std::size_t p) { return s?s->contr[0].coeff[p]:1.; }
Point center(const libint2::Shell* s) { return s?s->O:Point{}; }
Matrix transform(const libint2::Shell* s) { return representation(angular(s),s && s->contr[0].pure); }
Matrix pair_transform(const libint2::Shell* a,const libint2::Shell* b) {
    const auto ca=transform(a),cb=transform(b);
    Matrix result(ca.rows()*cb.rows(),ca.cols()*cb.cols());
    for(Eigen::Index i=0;i<ca.rows();++i) for(Eigen::Index j=0;j<cb.rows();++j)
        for(Eigen::Index k=0;k<ca.cols();++k) for(Eigen::Index l=0;l<cb.cols();++l)
            result(i*cb.rows()+j,k*cb.cols()+l)=ca(i,k)*cb(j,l);
    return result;
}
void accumulate(Matrix& sum,Matrix& correction,const Matrix& value,double factor) {
    for(Eigen::Index i=0;i<sum.size();++i) {
        const double term=factor*value.data()[i],before=sum.data()[i],after=before+term;
        correction.data()[i]+=std::abs(before)>=std::abs(term)?(before-after)+term:(term-after)+before;
        sum.data()[i]=after;
    }
}
void check(const Matrix& result) {
    if(!result.allFinite()) throw std::overflow_error("MANTA MD nonfinite contracted shell result");
}
std::size_t ncart(const libint2::Shell* s) {
    const int l=angular(s); return (l+1)*(l+2)/2;
}
} // namespace

Matrix shell_coulomb(const std::array<const libint2::Shell*,4>& s,RadialKernel kernel,double omega) {
    const auto nbra=ncart(s[0])*ncart(s[1]),nket=ncart(s[2])*ncart(s[3]);
    Matrix sum=Matrix::Zero(nbra,nket),correction=Matrix::Zero(nbra,nket);
    // Segmented early contraction: E/R/GEMM workspace contains only one
    // primitive quartet. No global primitive or AO four-index tensor.
    // The ket primitive-pair expansions of this shell quartet are built once
    // and read by every bra primitive pair; the Hermite kernel and the two
    // E products reuse one scratch. Same arithmetic and accumulation order
    // as one pair/coulomb call per primitive quartet (#638).
    const auto nc=primitives(s[2]),nd=primitives(s[3]);
    std::vector<Pair> kets; kets.reserve(nc*nd);
    for(std::size_t c=0;c<nc;++c) for(std::size_t d=0;d<nd;++d)
        kets.push_back(pair(angular(s[2]),angular(s[3]),exponent(s[2],c),exponent(s[3],d),center(s[2]),center(s[3])));
    CoulombScratch scratch; Matrix value;
    for(std::size_t a=0;a<primitives(s[0]);++a) for(std::size_t b=0;b<primitives(s[1]);++b) {
        const auto ab=pair(angular(s[0]),angular(s[1]),exponent(s[0],a),exponent(s[1],b),center(s[0]),center(s[1]));
        const double cab=coefficient(s[0],a)*coefficient(s[1],b);
        for(std::size_t c=0;c<nc;++c) for(std::size_t d=0;d<nd;++d) {
            coulomb_into(ab,kets[c*nd+d],kernel,omega,scratch,value);
            accumulate(sum,correction,value,cab*coefficient(s[2],c)*coefficient(s[3],d));
        }
    }
    sum+=correction;
    // Keep Cartesian paths free of identity matrix multiplications. Folded
    // E transforms and shared general contractions are separate schedules.
    if((s[0] && s[0]->contr[0].pure) || (s[1] && s[1]->contr[0].pure))
        sum=(pair_transform(s[0],s[1])*sum).eval();
    if((s[2] && s[2]->contr[0].pure) || (s[3] && s[3]->contr[0].pure))
        sum=(sum*pair_transform(s[2],s[3]).transpose()).eval();
    check(sum);
    return sum;
}
SharedCoulombTiles shell_coulomb_shared(const std::vector<std::array<const libint2::Shell*,4>>& channels,
                                      RadialKernel kernel,double omega) {
    if(channels.empty()) throw std::invalid_argument("MANTA empty contraction channel group");
    const auto& shape=channels.front();
    for(const auto& quartet:channels) for(int axis=0;axis<4;++axis)
        if(!quartet[axis] || !shape[axis] || angular(quartet[axis])!=angular(shape[axis]) || quartet[axis]->O!=shape[axis]->O)
            throw std::invalid_argument("MANTA incompatible contraction channel group");
    using Coefficients=std::map<double,double>;
    std::vector<std::array<Coefficients,4>> coefficients(channels.size());
    for(std::size_t channel=0;channel<channels.size();++channel) for(int axis=0;axis<4;++axis) {
        const auto* shell=channels[channel][axis];
        // Equal exponents denote the same primitive polynomial, including
        // repeated exponents within one channel. Preserve zero entries so
        // first-occurrence ownership is independent of signed coefficients.
        std::map<double,std::pair<double,double>> merged;
        for(std::size_t p=0;p<shell->alpha.size();++p) {
            auto& entry=merged[shell->alpha[p]];
            const double term=coefficient(shell,p),next=entry.first+term;
            entry.second+=std::abs(entry.first)>=std::abs(term)?(entry.first-next)+term:(term-next)+entry.first;
            entry.first=next;
        }
        for(const auto& entry:merged) {
            const double value=entry.second.first+entry.second.second;
            if(!std::isfinite(value)) throw std::overflow_error("MANTA nonfinite shared contraction coefficient");
            coefficients[channel][axis].emplace(entry.first,value);
        }
    }
    SharedCoulombTiles result;
    const auto rows=ncart(shape[0])*ncart(shape[1]),cols=ncart(shape[2])*ncart(shape[3]);
    result.values.assign(channels.size(),Matrix::Zero(rows,cols));
    std::vector<Matrix> corrections= result.values;
    std::vector<double> weights(channels.size());
    // Neese, SHARK (2023), sections 3.2-3.4: common primitive integrals feed
    // several contraction channels. This bounded implementation enumerates
    // only products present in a channel, and gives each product to its first
    // containing channel. It stores no primitive-quartet cache or union tensor.
    // Exact exponent membership also supports partially shared contractions.
    for(std::size_t owner=0;owner<channels.size();++owner)
    for(const auto& a:coefficients[owner][0]) for(const auto& b:coefficients[owner][1])
    for(const auto& c:coefficients[owner][2]) for(const auto& d:coefficients[owner][3]) {
        const std::array<double,4> exponents{a.first,b.first,c.first,d.first};
        bool seen=false;
        for(std::size_t prior=0;prior<owner && !seen;++prior) {
            bool contains=true;
            for(int axis=0;axis<4;++axis) contains=contains && coefficients[prior][axis].count(exponents[axis]);
            seen=contains;
        }
        if(seen) continue;
        bool active=false;
        for(std::size_t channel=owner;channel<channels.size();++channel) {
            double weight=1.;
            for(int axis=0;axis<4;++axis) {
                const auto entry=coefficients[channel][axis].find(exponents[axis]);
                if(entry==coefficients[channel][axis].end()) { weight=0.; break; }
                weight*=entry->second;
            }
            if(!std::isfinite(weight)) throw std::overflow_error("MANTA nonfinite shared contraction weight");
            weights[channel]=weight; active=active || weight!=0.;
        }
        if(!active) continue;
        const auto ab=pair(angular(shape[0]),angular(shape[1]),a.first,b.first,center(shape[0]),center(shape[1]));
        const auto cd=pair(angular(shape[2]),angular(shape[3]),c.first,d.first,center(shape[2]),center(shape[3]));
        const auto value=coulomb(ab,cd,kernel,omega);
        ++result.primitive_evaluations;
        for(std::size_t channel=owner;channel<channels.size();++channel) if(weights[channel]!=0.)
            accumulate(result.values[channel],corrections[channel],value,weights[channel]);
    }
    for(std::size_t channel=0;channel<channels.size();++channel) {
        auto& value=result.values[channel];
        const auto& shells=channels[channel];
        value+=corrections[channel];
        if(shells[0]->contr[0].pure || shells[1]->contr[0].pure)
            value=(pair_transform(shells[0],shells[1])*value).eval();
        if(shells[2]->contr[0].pure || shells[3]->contr[0].pure)
            value=(value*pair_transform(shells[2],shells[3]).transpose()).eval();
        check(value);
    }
    return result;
}
CoulombDerivative shell_coulomb_derivative(const std::array<const libint2::Shell*,4>& shells,int target,int axis,
                                           RadialKernel kernel,double omega) {
    if(target<0 || target>=4 || !shells[target] || axis<0 || axis>=3)
        throw std::invalid_argument("MANTA Coulomb derivative center/axis");
    std::array<int,4> angulars{};
    std::array<std::size_t,4> sizes{};
    for(int k=0;k<4;++k) { angulars[k]=angular(shells[k]); sizes[k]=ncart(shells[k]); }
    const auto original=cartesians(angulars[target]),upper=cartesians(angulars[target]+1);
    const auto lower=angulars[target]?cartesians(angulars[target]-1):std::vector<Index>{};
    std::vector<std::size_t> up(original.size()),down(original.size());
    for(std::size_t i=0;i<original.size();++i) {
        auto powers=original[i]; ++powers[axis];
        up[i]=std::find(upper.begin(),upper.end(),powers)-upper.begin();
        if(original[i][axis]) {
            powers=original[i]; --powers[axis];
            down[i]=std::find(lower.begin(),lower.end(),powers)-lower.begin();
        }
    }
    Matrix sum=Matrix::Zero(sizes[0]*sizes[1],sizes[2]*sizes[3]),correction=sum;
    CoulombDerivative result;
    for(std::size_t a=0;a<primitives(shells[0]);++a) for(std::size_t b=0;b<primitives(shells[1]);++b)
    for(std::size_t c=0;c<primitives(shells[2]);++c) for(std::size_t d=0;d<primitives(shells[3]);++d) {
        const std::array<std::size_t,4> primitives_at{a,b,c,d};
        auto evaluate_shift=[&](int delta) {
            auto shifted=angulars; shifted[target]+=delta;
            const auto bra=pair(shifted[0],shifted[1],exponent(shells[0],a),exponent(shells[1],b),center(shells[0]),center(shells[1]));
            const auto ket=pair(shifted[2],shifted[3],exponent(shells[2],c),exponent(shells[3],d),center(shells[2]),center(shells[3]));
            ++result.primitive_evaluations;
            return coulomb(bra,ket,kernel,omega);
        };
        const Matrix raised=evaluate_shift(1);
        const Matrix lowered=angulars[target]?evaluate_shift(-1):Matrix{};
        auto upper_sizes=sizes,lower_sizes=sizes;
        upper_sizes[target]=upper.size(); lower_sizes[target]=lower.size();
        Matrix value(sum.rows(),sum.cols());
        // MD center derivative: d/dA_x [(x-A_x)^a exp(-alpha*r_A^2)]
        // = 2*alpha*g_(a+1) - a*g_(a-1). Shifted primitives retain the
        // ORIGINAL channel coefficients; no shifted-shell normalization.
        for(std::size_t i=0;i<sizes[0];++i) for(std::size_t j=0;j<sizes[1];++j)
        for(std::size_t k=0;k<sizes[2];++k) for(std::size_t l=0;l<sizes[3];++l) {
            std::array<std::size_t,4> indices{i,j,k,l};
            const auto component=indices[target];
            indices[target]=up[component];
            double v=2*exponent(shells[target],primitives_at[target])*
                raised(indices[0]*upper_sizes[1]+indices[1],indices[2]*upper_sizes[3]+indices[3]);
            if(original[component][axis]) {
                indices[target]=down[component];
                v-=original[component][axis]*lowered(indices[0]*lower_sizes[1]+indices[1],indices[2]*lower_sizes[3]+indices[3]);
            }
            value(i*sizes[1]+j,k*sizes[3]+l)=v;
        }
        const double factor=coefficient(shells[0],a)*coefficient(shells[1],b)*coefficient(shells[2],c)*coefficient(shells[3],d);
        accumulate(sum,correction,value,factor);
    }
    sum+=correction;
    if((shells[0] && shells[0]->contr[0].pure) || (shells[1] && shells[1]->contr[0].pure))
        sum=(pair_transform(shells[0],shells[1])*sum).eval();
    if((shells[2] && shells[2]->contr[0].pure) || (shells[3] && shells[3]->contr[0].pure))
        sum=(sum*pair_transform(shells[2],shells[3]).transpose()).eval();
    check(sum); result.values=std::move(sum);
    return result;
}
namespace {
Matrix primitive_one_electron(int la,int lb,double alpha,double beta,const Point& a,const Point& b,
                             const std::string& operation,const Charges& charges,const Point& origin,const Index& powers,
                             RadialKernel radial,double omega) {
    if(operation=="overlap") return overlap(la,lb,alpha,beta,a,b);
    if(operation=="kinetic") return kinetic(la,lb,alpha,beta,a,b);
    if(operation=="moment") return moment(la,lb,alpha,beta,a,b,origin,powers);
    if(operation!="potential") throw std::invalid_argument("MANTA MD one-electron operation");
    const auto e=pair(la,lb,alpha,beta,a,b);
    const auto na=cartesians(la).size(),nb=cartesians(lb).size();
    Matrix v=Matrix::Zero(na*nb,1),error=v;
    for(const auto& charge:charges) accumulate(v,error,potential(e,charge.second,charge.first,radial,omega),1.);
    v+=error;
    return Eigen::Map<const Matrix>(v.data(),na,nb);
}
Matrix finish_one_electron(Matrix sum,const Matrix& correction,const libint2::Shell& a,const libint2::Shell& b) {
    sum+=correction;
    if(a.contr[0].pure) sum=(transform(&a)*sum).eval();
    if(b.contr[0].pure) sum=(sum*transform(&b).transpose()).eval();
    check(sum);
    return sum;
}
}
Matrix shell_one_electron(const libint2::Shell& a,const libint2::Shell& b,
                          const std::string& operation,const Charges& charges,
                          const Point& origin,const Index& powers,RadialKernel radial,double omega) {
    Matrix sum=Matrix::Zero(ncart(&a),ncart(&b)),correction=sum;
    for(std::size_t i=0;i<a.alpha.size();++i) for(std::size_t j=0;j<b.alpha.size();++j) {
        const Matrix value=primitive_one_electron(angular(&a),angular(&b),a.alpha[i],b.alpha[j],a.O,b.O,
                                                 operation,charges,origin,powers,radial,omega);
        accumulate(sum,correction,value,coefficient(&a,i)*coefficient(&b,j));
    }
    return finish_one_electron(std::move(sum),correction,a,b);
}
Matrix shell_moments(const libint2::Shell& a,const libint2::Shell& b,const Point& origin,int maximum_degree) {
    const auto count=hermites(maximum_degree).size();
    Matrix sum=Matrix::Zero(ncart(&a)*ncart(&b),count),correction=sum;
    for(std::size_t i=0;i<a.alpha.size();++i) for(std::size_t j=0;j<b.alpha.size();++j) {
        const Matrix panel=moments(angular(&a),angular(&b),a.alpha[i],b.alpha[j],a.O,b.O,origin,maximum_degree);
        accumulate(sum,correction,panel,coefficient(&a,i)*coefficient(&b,j));
    }
    sum+=correction;
    if(a.contr[0].pure || b.contr[0].pure) sum=(pair_transform(&a,&b)*sum).eval();
    check(sum); return sum;
}
Matrix shell_auxiliary_moments(const libint2::Shell& a,const Point& origin,int maximum_degree) {
    const auto count=hermites(maximum_degree).size();
    Matrix sum=Matrix::Zero(ncart(&a),count),correction=sum;
    // One Gaussian times a unit function, not an orbital self product.
    // The primitive Expansion is shared across every raw moment column.
    for(std::size_t i=0;i<a.alpha.size();++i) {
        const Matrix panel=moments(angular(&a),0,a.alpha[i],0.,a.O,a.O,origin,maximum_degree);
        accumulate(sum,correction,panel,coefficient(&a,i));
    }
    sum+=correction;
    if(a.contr[0].pure) sum=(transform(&a)*sum).eval();
    check(sum); return sum;
}
Matrix shell_one_electron_derivative(const libint2::Shell& a,const libint2::Shell& b,
                          const std::string& operation,const Charges& charges,const Point& origin,const Index& powers,
                          const std::string& target,int axis,std::size_t charge_index,RadialKernel radial,double omega) {
    const int la=angular(&a),lb=angular(&b);
    const bool bra=target=="bra",center=bra || target=="ket";
    const int angular_target=bra?la:lb;
    const auto original=center?cartesians(angular_target):std::vector<Index>{};
    const auto upper=center?cartesians(angular_target+1):std::vector<Index>{};
    const auto lower=center && angular_target?cartesians(angular_target-1):std::vector<Index>{};
    std::vector<std::size_t> up(original.size()),down(original.size());
    for(std::size_t i=0;i<original.size();++i) {
        auto t=original[i]; ++t[axis]; up[i]=std::find(upper.begin(),upper.end(),t)-upper.begin();
        if(original[i][axis]) { t=original[i]; --t[axis]; down[i]=std::find(lower.begin(),lower.end(),t)-lower.begin(); }
    }
    Matrix sum=Matrix::Zero(ncart(&a),ncart(&b)),correction=sum;
    for(std::size_t i=0;i<a.alpha.size();++i) for(std::size_t j=0;j<b.alpha.size();++j) {
        Matrix value;
        if(center) {
            auto shifted=[&](int delta) {
                return primitive_one_electron(la+(bra?delta:0),lb+(bra?0:delta),a.alpha[i],b.alpha[j],a.O,b.O,
                                              operation,charges,origin,powers,radial,omega);
            };
            const Matrix raised=shifted(1),lowered=angular_target?shifted(-1):Matrix{};
            value.resize(sum.rows(),sum.cols());
            // McMurchie-Davidson center derivative: 2*alpha*g_(a+1)-a*g_(a-1).
            // Original normalized contraction coefficients and transforms apply.
            for(Eigen::Index row=0;row<sum.rows();++row) for(Eigen::Index col=0;col<sum.cols();++col) {
                const auto k=bra?row:col;
                double v=2*(bra?a.alpha[i]:b.alpha[j])*raised(bra?up[k]:row,bra?col:up[k]);
                if(original[k][axis]) v-=original[k][axis]*lowered(bra?down[k]:row,bra?col:down[k]);
                value(row,col)=v;
            }
        } else if(target=="charge") {
            const auto& charge=charges.at(charge_index);
            const Matrix v=potential_charge_derivative(pair(la,lb,a.alpha[i],b.alpha[j],a.O,b.O),charge.second,charge.first,axis,radial,omega);
            value=Eigen::Map<const Matrix>(v.data(),sum.rows(),sum.cols());
        } else if(target=="origin") {
            if(powers[axis]) {
                auto lowered=powers; --lowered[axis];
                value=-powers[axis]*moment(la,lb,a.alpha[i],b.alpha[j],a.O,b.O,origin,lowered);
            } else value=Matrix::Zero(sum.rows(),sum.cols());
        } else throw std::invalid_argument("MANTA one-electron derivative target");
        accumulate(sum,correction,value,coefficient(&a,i)*coefficient(&b,j));
    }
    return finish_one_electron(std::move(sum),correction,a,b);
}
ComplexMatrix shell_fourier_panel(const libint2::Shell* a,const libint2::Shell* b,const Matrix& vectors) {
    ComplexMatrix sum=ComplexMatrix::Zero(ncart(a)*ncart(b),vectors.rows()),correction=sum;
    for(std::size_t i=0;i<primitives(a);++i) for(std::size_t j=0;j<primitives(b);++j) {
        const auto e=pair(angular(a),angular(b),exponent(a,i),exponent(b,j),center(a),center(b));
        const ComplexMatrix value=(coefficient(a,i)*coefficient(b,j))*fourier_panel(e,vectors);
        const ComplexMatrix adjusted=value-correction;
        ComplexMatrix next=sum+adjusted;
        correction=(next-sum)-adjusted;
        sum=std::move(next);
    }
    if((a && a->contr[0].pure) || (b && b->contr[0].pure))
        sum=(pair_transform(a,b).cast<std::complex<double>>()*sum).eval();
    if(!sum.allFinite()) throw std::overflow_error("MANTA MD nonfinite contracted Fourier panel");
    return sum;
}
Eigen::VectorXcd shell_fourier(const libint2::Shell& a,const libint2::Shell& b,const Point& q) {
    Eigen::VectorXcd sum=Eigen::VectorXcd::Zero(ncart(&a)*ncart(&b));
    Eigen::VectorXcd correction=sum;
    for(std::size_t i=0;i<a.alpha.size();++i) for(std::size_t j=0;j<b.alpha.size();++j) {
        const auto e=pair(angular(&a),angular(&b),a.alpha[i],b.alpha[j],a.O,b.O);
        const Eigen::VectorXcd value=(coefficient(&a,i)*coefficient(&b,j))*fourier(e,q);
        // Kahan accumulation in each real and imaginary channel.
        const Eigen::VectorXcd adjusted=value-correction;
        const Eigen::VectorXcd next=sum+adjusted;
        correction=(next-sum)-adjusted;
        sum=next;
    }
    if(a.contr[0].pure || b.contr[0].pure)
        sum=(pair_transform(&a,&b).cast<std::complex<double>>()*sum).eval();
    if(!sum.allFinite()) throw std::overflow_error("MANTA MD nonfinite Fourier result");
    return sum;
}
} // namespace vibeqc::manta::md
