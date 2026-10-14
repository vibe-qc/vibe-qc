#include "vibeqc/manta_periodic_exchange_bounds.hpp"
#include "manta_pair_image_internal.hpp"
#include "manta_lattice_geometry.hpp"

namespace vibeqc::manta {
// Gaussian Bloch Fourier convention: Sun et al. (2017), Sec.II.A,
// Eqs.3,10,16, DOI10.1063/1.4998644. Absolute image/reciprocal and density
// contraction enclosures are derived in scientific-contract Section5.13.
namespace {
using namespace admission;
using Real=long double;
using LogMatrix=Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
constexpr Real neg=-std::numeric_limits<Real>::infinity();
constexpr Real eps=std::numeric_limits<double>::epsilon();
Real addlog(Real a,Real b){if(a==neg)return b;if(b==neg)return a;const Real hi=std::max(a,b);return hi+std::log1p(std::exp(std::min(a,b)-hi));}
Real logvalue(double x){return x==0?neg:std::log(Real(x));}
double upper(Real x){if(x==neg)return 0;const double v=std::nextafter(double(std::exp(x)*(1+8192*eps)),std::numeric_limits<double>::infinity());if(!std::isfinite(v))throw std::overflow_error("MANTA exchange error bound exceeds binary64");return v;}
Real norm(std::complex<double> x){return std::hypot(Real(x.real()),Real(x.imag()));}
LogMatrix logs(const md::Matrix& m){LogMatrix out(m.rows(),m.cols());for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)out(i,j)=logvalue(m(i,j));return out;}
LogMatrix logs(const std::complex<double>* x,Count n){LogMatrix out(n,n);for(Count i=0;i<n;++i)for(Count j=0;j<n;++j){const Real v=norm(x[i*n+j]);out(i,j)=v?std::log(v):neg;}return out;}
LogMatrix sum(const LogMatrix& a,const LogMatrix& b){LogMatrix out(a.rows(),a.cols());for(Eigen::Index i=0;i<a.size();++i)out.data()[i]=addlog(a.data()[i],b.data()[i]);return out;}
LogMatrix product(const LogMatrix& a,const LogMatrix& b){LogMatrix out=LogMatrix::Constant(a.rows(),b.cols(),neg);for(Eigen::Index i=0;i<a.rows();++i)for(Eigen::Index j=0;j<b.cols();++j)for(Eigen::Index k=0;k<a.cols();++k)out(i,j)=addlog(out(i,j),a(i,k)+b(k,j));return out;}
LogMatrix sandwich(const LogMatrix& a,const LogMatrix& d,const LogMatrix& b){return product(a.transpose().eval(),product(d,b));}
PeriodicExchangeBoundPlan density_plan(PeriodicExchangeBoundPlan p,Count nd,const ResourceBudget& b){
    margin(b);if(!nd)throw std::invalid_argument("MANTA exchange bounds require positive density count");p.densities=nd;
    const Count n2=square(p.n_basis),elements=mul(nd,mul(p.momenta,n2));
    cap(elements,std::numeric_limits<Eigen::Index>::max(),"MANTA exchange bound output is not addressable");
    p.scalar_terms=add(p.scalar_terms,mul(1024,mul(elements,add(p.n_basis,p.momenta))));
    p.kernel_terms=add(p.kernel_terms,mul(128,p.scalar_terms));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(b.other_live_bytes,add(mul(256,elements),mul(512,n2))));
    cap(p.shell_calls,b.shell_calls,"MANTA exchange bound shell budget exceeded");
    cap(p.primitive_pairs,b.primitive_quartets,"MANTA exchange bound primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA exchange bound scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA exchange bound kernel budget exceeded");
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA exchange bound memory budget exceeded");return p;
}
LatticeIndex label(const Eigen::Vector3d& x,Count radius){LatticeIndex n{};for(int a=0;a<3;++a){const double v=x[a],r=std::round(v);if(!std::isfinite(v)||std::abs(r)>double(radius)||std::abs(v-r)>1e-10)throw std::invalid_argument("MANTA exchange vector lies outside its declared completion cube or reciprocal lattice");n[a]=static_cast<std::int64_t>(r);}return n;}
} // namespace

PeriodicExchangeBoundPlan PeriodicExchangeBoundSource::plan_source(const PeriodicPairSource& s,Count vectors,Count radius,Count nd,const ResourceBudget& b){
    if(radius>2147483647)throw std::invalid_argument("MANTA exchange bound cube exceeds signed 31-bit labels");
    const auto image=pair_image_detail::plan(s.basis_,s.entries_,s.images_,s.plan_.momenta,b);
    PeriodicExchangeBoundPlan p;p.n_basis=s.plan_.n_basis;p.momenta=s.plan_.momenta;p.reciprocal_cube_radius=radius;
    const Count width=add(mul(2,radius),1);p.reciprocal_cube_cells=mul(square(width),width);
    p.shell_calls=image.shell_calls;p.primitive_pairs=image.primitive_products;
    p.scalar_terms=add(mul(4,image.scalar_terms),mul(1024,add(vectors,mul(square(p.momenta),p.reciprocal_cube_cells))));
    p.kernel_terms=mul(128,p.scalar_terms);
    p.modeled_peak_bytes=add(image.modeled_peak_bytes,add(s.plan_.retained_bytes,add(32768,add(mul(128,vectors),mul(128,square(p.momenta))))));
    // image already included the caller's live bytes.
    p.modeled_peak_bytes-=b.other_live_bytes;
    return density_plan(p,nd,b);
}
PeriodicExchangeBoundSource::PeriodicExchangeBoundSource(const PeriodicPairSource& s,const md::Matrix& q,
    const std::vector<Count>& offsets,Count radius,const ResourceBudget& b){
    plan_=plan_source(s,q.rows(),radius,1,b);const Count nk=s.plan_.momenta,pairs=square(nk);
    if(q.cols()!=3||!q.allFinite()||offsets.size()!=add(pairs,1)||offsets.front()!=0||offsets.back()!=Count(q.rows()))
        throw std::invalid_argument("MANTA exchange bounds require finite Q vectors and complete transfer offsets");
    std::vector<std::vector<LatticeIndex>> labels(pairs);const Eigen::Matrix3d inverse=s.lattice_.transpose()/(2*std::acos(-1.));
    lattice_detail::Hash h;h.text("manta.periodic-exchange-bound.v1;full-images;full-reciprocal;fixed-mesh;q0-omitted");h.text(s.receipt_.source_identity);h.word(radius);
    for(Count t=0;t<nk;++t)for(Count a=0;a<nk;++a){const Count pair=t*nk+a;if(offsets[pair]>offsets[pair+1])throw std::invalid_argument("MANTA exchange bound offsets must be monotone");
        auto& v=labels[pair];v.reserve(offsets[pair+1]-offsets[pair]);h.word(offsets[pair]);
        for(Count g=offsets[pair];g<offsets[pair+1];++g){if(q.row(g).squaredNorm()==0||!std::isfinite(q.row(g).squaredNorm()))throw std::invalid_argument("MANTA full-Coulomb exchange bounds require resolved nonzero Q");
            v.push_back(label(inverse*(q.row(g)-s.momenta_.row(t)+s.momenta_.row(a)).transpose(),radius));for(int d=0;d<3;++d)h.number(q(g,d));}
        std::sort(v.begin(),v.end());if(std::adjacent_find(v.begin(),v.end())!=v.end())throw std::invalid_argument("MANTA exchange bound vectors must be unique");}
    for(Count t=0;t<nk;++t)for(Count a=0;a<nk;++a){const auto& reverse=labels[a*nk+t];for(const auto& n:labels[t*nk+a])if(!std::binary_search(reverse.begin(),reverse.end(),LatticeIndex{-n[0],-n[1],-n[2]}))throw std::invalid_argument("MANTA exchange bound supports require transfer reversal");}
    const auto ip=pair_image_detail::plan(s.basis_,s.entries_,s.images_,nk,b);
    auto images=pair_image_detail::fourier_images(s.basis_,s.lattice_,s.entries_,ip);
    retained_=std::move(images.retained);omitted_=std::move(images.omitted);reciprocal_=md::Matrix(pairs,2);weights_=s.weights_;
    receipt_.fourier_decay=images.decay;receipt_.pair_source_identity=s.receipt_.source_identity;
    for(Count t=0;t<nk;++t)for(Count a=0;a<nk;++a){const Count pair=t*nk+a;
        const auto values=pair_image_detail::reciprocal_exchange_sums(s.lattice_,(s.momenta_.row(t)-s.momenta_.row(a)).transpose(),labels[pair],images.decay,radius);
        for(int j=0;j<2;++j){reciprocal_(pair,j)=values[j];h.number(values[j]);}}
    h.number(images.decay);for(Eigen::Index i=0;i<retained_.size();++i){h.number(retained_.data()[i]);h.number(omitted_.data()[i]);}
    receipt_.source_identity=h.value.finish_hex();
}
PeriodicExchangeBoundPlan PeriodicExchangeBoundSource::plan_build(Count nd,const ResourceBudget& b)const{
    PeriodicExchangeBoundPlan p=plan_;p.shell_calls=p.primitive_pairs=p.scalar_terms=p.kernel_terms=0;
    return density_plan(p,nd,b);
}
PeriodicExchangeBoundResult PeriodicExchangeBoundSource::build(const std::vector<std::complex<double>>& d,Count nd,const ResourceBudget& b)const{
    PeriodicExchangeBoundResult out;out.plan=plan_build(nd,b);out.receipt=receipt_;const Count n=plan_.n_basis,n2=square(n),nk=plan_.momenta,elements=mul(nk,n2);
    if(d.size()!=mul(nd,elements))throw std::invalid_argument("MANTA exchange bound density extent mismatch");
    lattice_detail::Hash hash;hash.text(receipt_.source_identity);hash.word(nd);
    for(Count s=0;s<nd;++s)for(Count k=0;k<nk;++k){double residual=0,scale=0;for(Count i=0;i<n;++i)for(Count j=0;j<n;++j){const auto z=d[s*elements+k*n2+i*n+j];require_finite(z.real());require_finite(z.imag());hash.number(z.real());hash.number(z.imag());scale=std::max(scale,std::abs(z));residual=std::max(residual,std::abs(z-std::conj(d[s*elements+k*n2+j*n+i])));}if(!std::isfinite(scale)||!std::isfinite(residual)||residual>1e-12*std::max(1.,scale))throw std::invalid_argument("MANTA exchange bounds require finite Hermitian densities");}
    const auto retained=logs(retained_),omitted=logs(omitted_),full=sum(retained,omitted);
    out.error_bound.resize(d.size());out.pair_image_error_bound.resize(d.size());out.reciprocal_error_bound.resize(d.size());
    std::vector<Real> image(d.size(),neg),reciprocal(d.size(),neg);
    for(Count state=0;state<nd;++state)for(Count source=0;source<nk;++source){if(!weights_[source])continue;
        const auto density=logs(d.data()+state*elements+source*n2,n);
        const auto image_matrix=sum(sum(sandwich(retained,density,omitted),sandwich(omitted,density,retained)),sandwich(omitted,density,omitted));
        const auto reciprocal_matrix=sandwich(full,density,full);
        for(Count target=0;target<nk;++target){const Count pair=target*nk+source,offset=state*elements+target*n2;
            const Real common=logvalue(weights_[source]),a=common+logvalue(reciprocal_(pair,0)),g=common+logvalue(reciprocal_(pair,1));
            for(Count i=0;i<n2;++i){image[offset+i]=addlog(image[offset+i],a+image_matrix.data()[i]);reciprocal[offset+i]=addlog(reciprocal[offset+i],g+reciprocal_matrix.data()[i]);}}
    }
    for(Count i=0;i<d.size();++i){out.pair_image_error_bound[i]=upper(image[i]);out.reciprocal_error_bound[i]=upper(reciprocal[i]);out.error_bound[i]=positive_bound_sum(out.pair_image_error_bound[i],out.reciprocal_error_bound[i]);out.receipt.maximum_error=std::max(out.receipt.maximum_error,out.error_bound[i]);}
    out.receipt.density_identity=hash.value.finish_hex();out.receipt.energy_error_bounds.resize(nd);
    for(Count state=0;state<nd;++state){Real energy=neg;for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j){const auto index=state*elements+k*n2+i*n+j;const Real v=norm(d[state*elements+k*n2+j*n+i]);if(v)energy=addlog(energy,logvalue(weights_[k])-std::log(2.L)+std::log(v)+logvalue(out.error_bound[index]));}out.receipt.energy_error_bounds[state]=upper(energy);}return out;
}
double positive_bound_sum(double a,double b){return upper(addlog(logvalue(a),logvalue(b)));}
void scale_exchange_bound(PeriodicExchangeBoundResult& out,double scale){
    if(!std::isfinite(scale)||scale<0)throw std::invalid_argument("MANTA exchange bound scale must be finite nonnegative");
    out.receipt.exchange_scale*=scale;
    const Real factor=logvalue(scale);out.receipt.maximum_error=0;
    for(auto* values:{&out.error_bound,&out.pair_image_error_bound,&out.reciprocal_error_bound,&out.receipt.energy_error_bounds})for(auto& x:*values)x=upper(factor+logvalue(x));
    for(auto x:out.error_bound)out.receipt.maximum_error=std::max(out.receipt.maximum_error,x);
}
std::vector<double> bound_bvk_overlap_correction(const PeriodicOneElectronImageResult& images,
    const std::vector<std::complex<double>>& overlap,const std::vector<std::complex<double>>& d,Count nd,
    const std::vector<double>& weights,double scaled_madelung){
    const Count n=images.plan.n_basis,n2=n*n,nk=weights.size(),elements=nk*n2;std::vector<double> out(d.size());
    if(overlap.size()!=elements||d.size()!=nd*elements)throw std::logic_error("MANTA BvK image bound extent mismatch");
    const auto missing=logs(images.overlap_error_bound);const Real factor=logvalue(std::abs(scaled_madelung));
    for(Count s=0;s<nd;++s)for(Count k=0;k<nk;++k){const auto S=logs(overlap.data()+k*n2,n),D=logs(d.data()+s*elements+k*n2,n);
        const auto correction=sum(sum(product(S,product(D,missing)),product(missing,product(D,S))),product(missing,product(D,missing)));
        for(Count i=0;i<n2;++i)out[s*elements+k*n2+i]=upper(factor+correction.data()[i]);}return out;
}
double weighted_exchange_energy_bound(const std::vector<double>& bounds,const std::vector<std::complex<double>>& d,Count n,const std::vector<double>& weights){
    if(bounds.size()!=d.size())throw std::logic_error("MANTA exchange energy bound extent mismatch");const Count elements=weights.size()*n*n;Real energy=neg;
    for(Count s=0;s<d.size()/elements;++s)for(Count k=0;k<weights.size();++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j){const Real v=norm(d[s*elements+(k*n+j)*n+i]);if(v)energy=addlog(energy,logvalue(weights[k])-std::log(2.L)+std::log(v)+logvalue(bounds[s*elements+(k*n+i)*n+j]));}return upper(energy);
}
} // namespace vibeqc::manta
