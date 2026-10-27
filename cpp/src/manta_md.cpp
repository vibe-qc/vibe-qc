#include "vibeqc/manta_md.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::manta::md {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
void finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("MANTA MD nonfinite intermediate");
}
void order(int l) {
    if (l<0 || l>maximum_order) throw std::invalid_argument("MANTA MD angular order");
}
int degree(const Index& t) { return t[0]+t[1]+t[2]; }
double norm2(const Point& x) { return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]; }
} // namespace
void boys_into(int n,double t,std::vector<double>& f);
void radial_boys_into(int n,double t,double rho,RadialKernel kernel,double omega,std::vector<double>& f);
namespace {
struct Axis {
    int nb, nt;
    std::vector<double> e;
    Axis(int la,int lb,double p,double pa,double pb,double seed)
        : nb(lb+1),nt(la+lb+1),e((la+1)*nb*nt,0.) {
        at(0,0,0)=seed;
        for(int i=0;i<=la;++i) for(int j=0;j<=lb;++j) {
            if(i==0 && j==0) continue;
            const int u=i-(j==0), v=j-(j!=0);
            const double shift=j==0?pa:pb;
            for(int t=0;t<=i+j;++t)
                at(i,j,t)=shift*get(u,v,t)+get(u,v,t-1)/(2*p)+(t+1)*get(u,v,t+1);
        }
    }
    double& at(int i,int j,int t) { return e[(i*nb+j)*nt+t]; }
    double get(int i,int j,int t) const {
        return i<0 || j<0 || t<0 || t>=nt ? 0. : e[(i*nb+j)*nt+t];
    }
};
struct Expansion {
    double p;
    Point center;
    std::array<Axis,3> axes;
    Expansion(int la,int lb,double alpha,double beta,const Point& a,const Point& b)
        : p(alpha+beta), center{}, axes{make(0,la,lb,alpha,beta,a,b),
          make(1,la,lb,alpha,beta,a,b),make(2,la,lb,alpha,beta,a,b)} {
        for(int k=0;k<3;++k) center[k]=a[k]+beta/p*(b[k]-a[k]);
    }
    static Axis make(int k,int la,int lb,double alpha,double beta,const Point& a,const Point& b) {
        order(la); order(lb); order(la+lb);
        if(!std::isfinite(alpha) || !std::isfinite(beta) || alpha<0 || beta<0 || alpha+beta<=0)
            throw std::invalid_argument("MANTA MD primitive exponents");
        finite(a[k]); finite(b[k]); finite(alpha+beta);
        const double p=alpha+beta, delta=b[k]-a[k];
        const double pa=beta/p*delta, pb=-alpha/p*delta;
        const double exponent=-(alpha/p)*beta*delta*delta;
        finite(exponent);
        return Axis(la,lb,p,pa,pb,std::exp(exponent));
    }
    double coefficient(const Index& a,const Index& b,const Index& t) const {
        return axes[0].get(a[0],b[0],t[0])*axes[1].get(a[1],b[1],t[1])*axes[2].get(a[2],b[2],t[2]);
    }
    double overlap(const Index& a,const Index& b) const {
        for(int k=0;k<3;++k) if(a[k]<0 || b[k]<0) return 0.;
        return std::pow(pi/p,1.5)*coefficient(a,b,{0,0,0});
    }
};

// McMurchie and Davidson, J. Comput. Phys. 26, 218 (1978),
// doi:10.1016/0021-9991(78)90092-X. R000^n=(-2 rho)^n F_n(T).
// Memoization bounds the recurrence by the admitted total angular degree.
struct Auxiliary {
    int width;
    Point displacement;
    std::vector<double> seeds, cache;
    Auxiliary(int l,double rho,const Point& r,RadialKernel kernel=RadialKernel::Coulomb,double omega=0.)
        : width(l+1),displacement(r),seeds(radial_boys(l,rho*norm2(r),rho,kernel,omega)),
          cache(static_cast<std::size_t>(width)*width*width*width,std::numeric_limits<double>::quiet_NaN()) {
        scale_seeds(l,rho);
    }
    // Storage handed in by the caller (moved in, moved back out by the
    // caller) so repeated quartets reuse their capacity. Same values as above.
    Auxiliary(int l,double rho,const Point& r,RadialKernel kernel,double omega,
              std::vector<double>&& seed_storage,std::vector<double>&& cache_storage)
        : width(l+1),displacement(r),seeds(std::move(seed_storage)),cache(std::move(cache_storage)) {
        radial_boys_into(l,rho*norm2(r),rho,kernel,omega,seeds);
        cache.assign(static_cast<std::size_t>(width)*width*width*width,std::numeric_limits<double>::quiet_NaN());
        scale_seeds(l,rho);
    }
    void scale_seeds(int l,double rho) {
        double power=1.;
        for(int n=0;n<=l;++n) {
            seeds[n]*=power; finite(seeds[n]);
            if(n<l) power*=-2*rho;
        }
    }
    double get(int x,int y,int z,int n=0) {
        if(x==0 && y==0 && z==0) return seeds[n];
        auto& v=cache[((x*width+y)*width+z)*width+n];
        if(!std::isnan(v)) return v;
        if(x) v=displacement[0]*get(x-1,y,z,n+1)+(x>1?(x-1)*get(x-2,y,z,n+1):0.);
        else if(y) v=displacement[1]*get(x,y-1,z,n+1)+(y>1?(y-1)*get(x,y-2,z,n+1):0.);
        else v=displacement[2]*get(x,y,z-1,n+1)+(z>1?(z-1)*get(x,y,z-2,n+1):0.);
        finite(v);
        return v;
    }
};
} // namespace

void boys_into(int n,double t,std::vector<double>& f) {
    order(n);
    if(!std::isfinite(t) || t<0) throw std::invalid_argument("MANTA Boys argument");
    f.assign(n+1,0.);
    if(t==0) {
        for(int m=0;m<=n;++m) f[m]=1./(2*m+1);
    } else if(t>n+30.) {
        // Upward recurrence only where its subtractions are well conditioned.
        f[0]=.5*std::sqrt(pi/t)*std::erf(std::sqrt(t));
        const double exponential=std::exp(-t);
        for(int m=1;m<=n;++m) f[m]=((2*m-1)*f[m-1]-exponential)/t*.5;
    } else {
        // Positive-term expansion about the upper endpoint avoids the
        // cancellation of the alternating Taylor series at moderate T.
        double term=1.,sum=1.;
        int k=0;
        do {
            ++k; term*=2*t/(2*n+2*k+1); sum+=term;
        } while(term>sum*std::numeric_limits<double>::epsilon()*.25 && k<512);
        if(k==512) throw std::runtime_error("MANTA Boys series failed to converge");
        const double exponential=std::exp(-t);
        f[n]=exponential*sum/(2*n+1);
        for(int m=n;m>0;--m) f[m-1]=(2*t*f[m]+exponential)/(2*m-1);
    }
}
std::vector<double> boys(int n,double t) {
    std::vector<double> f; boys_into(n,t,f); return f;
}

namespace {
template<int N> struct GaussRule {
    std::array<double,N> nodes{},weights{};
    GaussRule() {
        for(int i=0;i<(N+1)/2;++i) {
            double x=std::cos(pi*(i+.75)/(N+.5)),derivative=0.;
            for(int iteration=0;iteration<32;++iteration) {
                double previous=1.,current=x;
                for(int k=2;k<=N;++k) {
                    const double next=((2*k-1)*x*current-(k-1)*previous)/k;
                    previous=current; current=next;
                }
                derivative=N*(x*current-previous)/(x*x-1);
                const double shift=current/derivative;
                x-=shift;
                if(std::abs(shift)<2e-16) break;
            }
            nodes[i]=-x; nodes[N-1-i]=x;
            weights[i]=weights[N-1-i]=2/((1-x*x)*derivative*derivative);
        }
    }
};
using Seeds=std::array<double,maximum_order+1>;
struct ShortRangeIntegral {
    int n,visits=0;
    double t,theta,width;
    Seeds tolerance{};
    template<int N> Seeds panel(double left,double right) const {
        static const GaussRule<N> rule;
        Seeds result{};
        const double midpoint=(left+right)*.5,halfwidth=(right-left)*.5;
        for(int i=0;i<N;++i) {
            const double s=midpoint+halfwidth*rule.nodes[i],u=theta+width*s;
            double value=halfwidth*rule.weights[i]*std::exp(-t*width*s*(2*theta+width*s));
            for(int m=0;m<=n;++m) { result[m]+=value; value*=u*u; }
        }
        return result;
    }
    Seeds integrate(double left,double right,int depth=0) {
        if(++visits>256 || depth>20) throw std::runtime_error("MANTA erfc quadrature work limit");
        const auto low=panel<16>(left,right),high=panel<32>(left,right);
        if(depth==0) for(int m=0;m<=n;++m) tolerance[m]=2e-14*std::abs(high[m]);
        bool converged=true;
        for(int m=0;m<=n;++m)
            if(std::abs(high[m]-low[m])>tolerance[m]*(right-left)
               +8*std::numeric_limits<double>::epsilon()*std::abs(high[m])
               +std::numeric_limits<double>::denorm_min())
                converged=false;
        if(converged) return high;
        auto a=integrate(left,(left+right)*.5,depth+1);
        const auto b=integrate((left+right)*.5,right,depth+1);
        for(int m=0;m<=n;++m) a[m]+=b[m];
        return a;
    }
};
} // namespace

void radial_boys_into(int n,double t,double rho,RadialKernel kernel,double omega,std::vector<double>& f) {
    if(kernel==RadialKernel::Coulomb) { boys_into(n,t,f); return; }
    order(n);
    if(!std::isfinite(t) || t<0 || !std::isfinite(rho) || rho<=0 || std::isnan(omega) || omega<0)
        throw std::invalid_argument("MANTA range-separated Boys arguments");
    if(omega==0) { if(kernel==RadialKernel::Erfc) boys_into(n,t,f); else f.assign(n+1,0.); return; }
    if(std::isinf(omega)) { if(kernel==RadialKernel::Erf) boys_into(n,t,f); else f.assign(n+1,0.); return; }
    const double denominator=std::hypot(omega,std::sqrt(rho));
    const double theta=omega/denominator,kappa=theta*theta;
    if(kernel==RadialKernel::Erf) {
        boys_into(n,kappa*t,f);
        double factor=theta;
        for(auto& v:f) { v*=factor; factor*=kappa; }
        return;
    }
    // SHARK Eq. 54, doi:10.1002/jcc.26942. For erfc, integrate the
    // positive interval [theta,1] directly rather than subtract two nearly
    // equal Boys values. Retain 1-theta via (1-kappa)/(1+theta), including
    // ranges so large that theta rounds to one in binary64.
    const double ratio=std::sqrt(rho)/denominator;
    double width=ratio*ratio/(1+theta);
    if(t>0) {
        // Trim only an exponentially negligible tail. The cutoff includes
        // the largest admitted polynomial power, not just the n=0 layer.
        const double limit=4*n+100.;
        const double root=std::hypot(theta,std::sqrt(limit/t));
        const double distance=(limit/t)/(root+theta);
        width=std::min(width,distance);
    }
    ShortRangeIntegral quadrature{n,0,t,theta,width};
    auto integral=quadrature.integrate(0.,1.);
    const double factor=width*std::exp(-t*kappa);
    f.assign(n+1,0.);
    for(int m=0;m<=n;++m) f[m]=factor*integral[m];
}
std::vector<double> radial_boys(int n,double t,double rho,RadialKernel kernel,double omega) {
    std::vector<double> f; radial_boys_into(n,t,rho,kernel,omega,f); return f;
}
namespace {
std::vector<Index> build_cartesians(int l) {
    std::vector<Index> result;
    for(int x=l;x>=0;--x) for(int y=l-x;y>=0;--y) result.push_back({x,y,l-x-y});
    return result;
}
} // namespace
const std::vector<Index>& cartesians(int l) {
    order(l);
    static const std::vector<std::vector<Index>> tables=[] {
        std::vector<std::vector<Index>> t;
        for(int order_=0;order_<=maximum_order;++order_) t.push_back(build_cartesians(order_));
        return t;
    }();
    return tables[l];
}
const std::vector<Index>& hermites(int l) {
    order(l);
    static const std::vector<std::vector<Index>> tables=[] {
        std::vector<std::vector<Index>> t;
        for(int order_=0;order_<=maximum_order;++order_) {
            std::vector<Index> result;
            for(int d=0;d<=order_;++d) for(const auto& index:build_cartesians(d)) result.push_back(index);
            t.push_back(std::move(result));
        }
        return t;
    }();
    return tables[l];
}
Matrix representation(int l,bool pure) {
    if(l<0 || l>7) throw std::invalid_argument("MANTA MD representation angular order");
    const auto powers=cartesians(l);
    const auto nc=powers.size();
    if(!pure) return Matrix::Identity(nc,nc);
    // NIST DLMF 14.30.1 and 14.30.3 (https://dlmf.nist.gov/14.30):
    // expand (x+i*y)^m d^m P_l(z/r) r^(l-m). Real/sine rows are
    // ordered m=-l..l and omit the Condon-Shortley phase, matching the
    // repository's real harmonics. Racah normalization consumes Cartesian
    // monomials with common axial shell normalization, without component
    // renormalization. Polynomial coefficients are exact dyadic rationals
    // before the final square root; no sampled/fitted transform is used.
    std::array<double,15> factorial{}; factorial[0]=1.;
    for(int i=1;i<15;++i) factorial[i]=i*factorial[i-1];
    Matrix c=Matrix::Zero(2*l+1,nc);
    for(int row=-l;row<=l;++row) {
        const int m=std::abs(row);
        const double scale=std::sqrt((m?2.:1.)*factorial[l-m]/factorial[l+m]);
        for(int k=0;k<=(l-m)/2;++k) {
            const double legendre=(k%2?-1.:1.)*factorial[2*l-2*k]
                /(std::ldexp(1.,l)*factorial[k]*factorial[l-k]*factorial[l-m-2*k]);
            for(int j=0;j<=m;++j) {
                const int phase=row<0?(j%4==1?1:j%4==3?-1:0):(j%4==0?1:j%4==2?-1:0);
                if(!phase) continue;
                const double azimuth=phase*factorial[m]/(factorial[j]*factorial[m-j]);
                for(int x=0;x<=k;++x) for(int y=0;y<=k-x;++y) {
                    const int z=k-x-y;
                    const Index term{m-j+2*x,j+2*y,l-m-2*k+2*z};
                    const auto col=std::find(powers.begin(),powers.end(),term)-powers.begin();
                    c(row+l,col)+=legendre*azimuth*factorial[k]/(factorial[x]*factorial[y]*factorial[z]);
                }
            }
        }
        c.row(row+l)*=scale;
    }
    return c;
}
Pair pair(int la,int lb,double alpha,double beta,const Point& a,const Point& b) {
    const Expansion e(la,lb,alpha,beta,a,b);
    const auto& ca=cartesians(la); const auto& cb=cartesians(lb);
    Pair result;
    result.exponent=e.p; result.center=e.center; result.indices=hermites(la+lb);
    result.coefficients.resize(ca.size()*cb.size(),result.indices.size());
    for(std::size_t i=0;i<ca.size();++i) for(std::size_t j=0;j<cb.size();++j)
        for(std::size_t k=0;k<result.indices.size();++k)
            result.coefficients(i*cb.size()+j,k)=e.coefficient(ca[i],cb[j],result.indices[k]);
    if(!result.coefficients.allFinite()) throw std::overflow_error("MANTA MD E coefficients");
    return result;
}
void coulomb_into(const Pair& a,const Pair& b,RadialKernel radial,double omega,CoulombScratch& scratch,Matrix& result) {
    const double p=a.exponent,q=b.exponent,rho=(p/(p+q))*q;
    Point r{}; for(int k=0;k<3;++k) r[k]=a.center[k]-b.center[k];
    const int l=degree(a.indices.back())+degree(b.indices.back());
    Auxiliary auxiliary(l,rho,r,radial,omega,std::move(scratch.seeds),std::move(scratch.cache));
    Matrix& kernel=scratch.kernel;
    kernel.resize(a.indices.size(),b.indices.size());
    const double scale=2*std::pow(pi,2.5)/(p*q*std::sqrt(p+q));
    for(std::size_t i=0;i<a.indices.size();++i) for(std::size_t j=0;j<b.indices.size();++j) {
        const auto& t=a.indices[i]; const auto& u=b.indices[j];
        kernel(i,j)=(degree(u)%2?-scale:scale)*auxiliary.get(t[0]+u[0],t[1]+u[1],t[2]+u[2]);
    }
    // Neese, J. Comput. Chem. 44, 381 (2023), doi:10.1002/jcc.26942:
    // rectangular E-R-E matrix formulation; one primitive quartet at a time.
    // The two products are the same expressions coulomb() evaluated, into
    // reused storage.
    scratch.intermediate.noalias()=a.coefficients*kernel;
    result.noalias()=scratch.intermediate*b.coefficients.transpose();
    if(!result.allFinite()) throw std::overflow_error("MANTA MD Coulomb result");
    scratch.seeds=std::move(auxiliary.seeds); scratch.cache=std::move(auxiliary.cache);
}
Matrix coulomb(const Pair& a,const Pair& b,RadialKernel radial,double omega) {
    CoulombScratch scratch; Matrix result;
    coulomb_into(a,b,radial,omega,scratch,result);
    return result;
}
Matrix overlap(int la,int lb,double alpha,double beta,const Point& a,const Point& b) {
    const Expansion e(la,lb,alpha,beta,a,b);
    const auto ca=cartesians(la),cb=cartesians(lb);
    Matrix result(ca.size(),cb.size());
    for(std::size_t i=0;i<ca.size();++i) for(std::size_t j=0;j<cb.size();++j) result(i,j)=e.overlap(ca[i],cb[j]);
    return result;
}
Matrix kinetic(int la,int lb,double alpha,double beta,const Point& a,const Point& b) {
    // Integrate the gradient product. For very unequal exponents this avoids
    // cancellation between beta*S and beta^2*S(b+2) in the Laplacian form.
    const Expansion e(la+1,lb+1,alpha,beta,a,b);
    const auto ca=cartesians(la),cb=cartesians(lb);
    Matrix result=Matrix::Zero(ca.size(),cb.size());
    for(std::size_t i=0;i<ca.size();++i) for(std::size_t j=0;j<cb.size();++j) for(int k=0;k<3;++k) {
        auto al=ca[i],au=ca[i],bl=cb[j],bu=cb[j];
        --al[k]; ++au[k]; --bl[k]; ++bu[k];
        result(i,j)+=.5*ca[i][k]*cb[j][k]*e.overlap(al,bl)
            -alpha*cb[j][k]*e.overlap(au,bl)-beta*ca[i][k]*e.overlap(al,bu)
            +2*alpha*beta*e.overlap(au,bu);
    }
    return result;
}
Matrix potential(const Pair& a,const Point& position,double charge,RadialKernel radial,double omega) {
    Point r{}; for(int k=0;k<3;++k) { finite(position[k]); r[k]=a.center[k]-position[k]; }
    finite(charge);
    // Point-charge limit of the same range-separated Hermite Coulomb kernel:
    // rho=p and theta=omega/sqrt(p+omega^2), with the positive erfc interval.
    Auxiliary auxiliary(degree(a.indices.back()),a.exponent,r,radial,omega);
    Eigen::VectorXd kernel(a.indices.size());
    for(std::size_t i=0;i<a.indices.size();++i) {
        const auto& t=a.indices[i]; kernel[i]=-charge*(2*pi/a.exponent)*auxiliary.get(t[0],t[1],t[2]);
    }
    return a.coefficients*kernel;
}
Matrix potential_charge_derivative(const Pair& a,const Point& position,double charge,int axis,RadialKernel radial,double omega) {
    if(axis<0 || axis>2) throw std::invalid_argument("MANTA potential derivative axis");
    Point r{}; for(int k=0;k<3;++k) { finite(position[k]); r[k]=a.center[k]-position[k]; }
    finite(charge);
    // MD R is differentiated in P-C. Moving C supplies a minus sign,
    // cancelling the minus sign of the attractive -q/r potential.
    Auxiliary auxiliary(degree(a.indices.back())+1,a.exponent,r,radial,omega);
    Eigen::VectorXd kernel(a.indices.size());
    for(std::size_t i=0;i<a.indices.size();++i) {
        auto t=a.indices[i]; ++t[axis];
        kernel[i]=charge*(2*pi/a.exponent)*auxiliary.get(t[0],t[1],t[2]);
    }
    return a.coefficients*kernel;
}
Matrix moment(int la,int lb,double alpha,double beta,const Point& a,const Point& b,const Point& origin,const Index& powers) {
    for(auto n:powers) order(n);
    const Expansion e(la+degree(powers),lb,alpha,beta,a,b);
    const auto ca=cartesians(la),cb=cartesians(lb);
    Matrix result=Matrix::Zero(ca.size(),cb.size());
    for(std::size_t i=0;i<ca.size();++i) for(std::size_t j=0;j<cb.size();++j) {
        double product=std::pow(pi/e.p,1.5);
        for(int k=0;k<3;++k) {
            finite(origin[k]); double axis=0.,binomial=1.;
            for(int m=0;m<=powers[k];++m) {
                axis+=binomial*std::pow(a[k]-origin[k],powers[k]-m)*e.axes[k].get(ca[i][k]+m,cb[j][k],0);
                binomial*=static_cast<double>(powers[k]-m)/(m+1);
            }
            product*=axis;
        }
        result(i,j)=product;
    }
    return result;
}
Matrix moments(int la,int lb,double alpha,double beta,const Point& a,const Point& b,const Point& origin,int maximum_degree) {
    order(maximum_degree);
    for(auto x:origin) finite(x);
    const Expansion e(la+maximum_degree,lb,alpha,beta,a,b);
    const auto ca=cartesians(la),cb=cartesians(lb),indices=hermites(maximum_degree);
    Matrix result(ca.size()*cb.size(),indices.size());
    // McMurchie-Davidson polynomial moments, reusing one primitive Expansion
    // over the full panel. This is the same shared-source/multiple-target
    // principle as Neese, SHARK (2023), DOI 10.1002/jcc.26942.
    for(std::size_t q=0;q<indices.size();++q) {
        const auto& powers=indices[q];
        for(std::size_t i=0;i<ca.size();++i) for(std::size_t j=0;j<cb.size();++j) {
            double value=std::pow(pi/e.p,1.5);
            for(int axis=0;axis<3;++axis) {
                double sum=0.,correction=0.,binomial=1.;
                for(int k=0;k<=powers[axis];++k) {
                    const double term=binomial*std::pow(a[axis]-origin[axis],powers[axis]-k)*e.axes[axis].get(ca[i][axis]+k,cb[j][axis],0);
                    const double next=sum+term;
                    correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;
                    sum=next; binomial*=double(powers[axis]-k)/(k+1);
                }
                value*=sum+correction;
            }
            result(i*cb.size()+j,q)=value;
        }
    }
    return result;
}
ComplexMatrix fourier_panel(const Pair& a,const Matrix& vectors) {
    if(vectors.cols()!=3 || !vectors.allFinite()) throw std::invalid_argument("MANTA Fourier panel vectors");
    const std::complex<double> minus_i(0.,-1.);
    ComplexMatrix kernel(a.indices.size(),vectors.rows());
    for(Eigen::Index column=0;column<vectors.rows();++column) {
        const Point q{vectors(column,0),vectors(column,1),vectors(column,2)};
        double phase=0.; for(int k=0;k<3;++k) phase+=q[k]*a.center[k];
        const auto factor=std::pow(pi/a.exponent,1.5)*std::exp(-norm2(q)/(4*a.exponent))*std::exp(minus_i*phase);
        for(std::size_t i=0;i<a.indices.size();++i) {
            auto value=factor;
            for(int k=0;k<3;++k) for(int n=0;n<a.indices[i][k];++n) value*=minus_i*q[k];
            kernel(i,column)=value;
        }
    }
    // Neese, J. Comput. Chem. 44, 381 (2023), doi:10.1002/jcc.26942.
    // One E expansion is shared by every q column; the Fourier transform of
    // a Hermite derivative is (-i*q)^t times the Gaussian Fourier seed.
    ComplexMatrix result=a.coefficients.cast<std::complex<double>>()*kernel;
    if(!result.allFinite()) throw std::overflow_error("MANTA nonfinite Fourier primitive panel");
    return result;
}
Eigen::VectorXcd fourier(const Pair& a,const Point& q) {
    for(auto v:q) finite(v);
    const std::complex<double> minus_i(0.,-1.);
    double phase=0.; for(int k=0;k<3;++k) phase+=q[k]*a.center[k];
    const auto factor=std::pow(pi/a.exponent,1.5)*std::exp(-norm2(q)/(4*a.exponent))*std::exp(minus_i*phase);
    Eigen::VectorXcd kernel(a.indices.size());
    for(std::size_t i=0;i<a.indices.size();++i) {
        auto v=factor;
        for(int k=0;k<3;++k) for(int n=0;n<a.indices[i][k];++n) v*=minus_i*q[k];
        kernel[i]=v;
    }
    return a.coefficients.cast<std::complex<double>>()*kernel;
}
} // namespace vibeqc::manta::md
