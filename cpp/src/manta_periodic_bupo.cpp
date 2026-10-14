#include "vibeqc/manta_periodic_bupo.hpp"
#include "vibeqc/manta_lattice_errors.hpp"
#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/manta_provider.hpp"
#include "manta_bupo_interaction.hpp"
#include "manta_ewald_internal.hpp"
#include "manta_gaussian_domain_internal.hpp"
#include "manta_pair_image_internal.hpp"

namespace vibeqc::manta {
// BUPO near/far separation: Neese et al. (2025), DOI 10.1021/acs.jpca.4c07415.
// This finite periodic extension uses Ewald (1921), IV.1, Eqs. 37-40,
// DOI 10.1002/andp.19213690304, and the q=0 coefficients of Sun et al.
// (2017), Eqs. 16-17, DOI 10.1063/1.4998644. Full Gaussian error ledgers
// and a fixed density-independent partition qualify only the retained source.
namespace {
using namespace admission;
using namespace ewald_detail;
template<class T> const T& required_source(const std::shared_ptr<const T>& p) {
    if(!p) throw std::invalid_argument("MANTA periodic BUPO requires native hierarchy and direct Ewald sources");
    return *p;
}
template<class F> void pairs(const PeriodicBubbleTree& tree,const BubbleBlock& block,F f) {
    const auto& a=tree.nodes()[block.target]; const auto& b=tree.nodes()[block.source];
    for(Count i=a.begin;i<a.end;++i) for(Count j=b.begin;j<(block.target==block.source?i+1:b.end);++j) {
        auto p=tree.pair_order()[i],q=tree.pair_order()[j]; if(p>q) std::swap(p,q); f(p,q);
    }
}
constexpr Real minus_inf=-std::numeric_limits<Real>::infinity();
Real logadd(Real a,Real b) {
    if(a==minus_inf) return b; if(b==minus_inf) return a;
    const auto high=std::max(a,b); return high+std::log1p(std::exp(std::min(a,b)-high));
}
double upper(Real logarithm) {
    if(logarithm==minus_inf) return 0;
    const Real x=std::exp(logarithm)*(1+8192*std::numeric_limits<double>::epsilon());
    const double out=std::nextafter(number(x),std::numeric_limits<double>::infinity());
    if(!std::isfinite(out)) throw std::overflow_error("MANTA periodic BUPO error bound exceeds binary64");
    return out;
}
double positive_add(double a,double b) {
    return upper(logadd(a?std::log(Real(a)):minus_inf,b?std::log(Real(b)):minus_inf));
}
double positive_product(double a,double b,double c=1.) {
    return a==0 || b==0 || c==0?0:upper(std::log(Real(a))+std::log(Real(b))+std::log(Real(c)));
}
double diagonal_product(double bound,double norm) {
    return bound==0 || norm==0?0:upper(std::log(Real(bound))+2*std::log(Real(norm))-std::log(Real(2)));
}
void scatter_bound(const PeriodicPairEntry& e,double bound,md::Matrix& values) {
    for(Count i=0;i<e.sizes[0];++i) for(Count j=0;j<e.sizes[1];++j) {
        auto& a=values(e.offsets[0]+i,e.offsets[1]+j); a=positive_add(a,bound);
        if(e.paired) { auto& b=values(e.offsets[1]+j,e.offsets[0]+i); b=positive_add(b,bound); }
    }
}
PairExtent envelope(const BubbleNode& n,const BubbleEnvelope& e) {
    PairExtent out; out.center=n.center; out.radius=n.radius; out.always_near=n.always_near;
    out.full_moment_bounds=e.full_moments; out.tail_moment_bounds=e.tail_moments; out.potential_bound=e.potential;
    return out; // Native internal adapter, never a caller-provided envelope.
}
void record(PeriodicBupoPartition& p,const BubbleBlock& block,bool far,double error) {
    if(far) {
        ++p.approximate_blocks; p.approximate_pairs+=block.pair_interactions;
        p.maximum_integral_error=std::max(p.maximum_integral_error,error);
    } else { ++p.direct_blocks; p.direct_pairs+=block.pair_interactions; }
}
void finish(PeriodicBupoPartition& p,const BubbleTraversalReceipt& r) {
    p.traversal=r;
    if(!r.complete || r.failed || p.approximate_pairs+p.direct_pairs!=r.pair_interactions)
        throw std::logic_error("MANTA periodic BUPO partition is incomplete");
    lattice_detail::Hash h; h.text("manta.periodic-bupo-partition.v1"); h.text(r.partition_identity);
    h.word(p.approximate_blocks); h.word(p.direct_blocks); h.word(p.approximate_pairs); h.word(p.direct_pairs);
    h.number(p.maximum_integral_error); p.identity=h.value.finish_hex();
}
} // namespace

PeriodicBupoPlan PeriodicBupoSource::plan_source(const PeriodicBubbleTree& tree,const PeriodicEwaldSource& direct,
    Count near_cells,const PeriodicBupoOptions& options,const ResourceBudget& budget) {
    margin(budget); require_finite(options.opening); require_finite(options.integral_tolerance);
    if(options.opening<0 || options.opening>=1 || options.integral_tolerance<0 || !near_cells || !(near_cells%2))
        throw std::invalid_argument("MANTA periodic BUPO opening, tolerance or near count is invalid");
    if(tree.pair_source_identity()!=direct.pair_source_identity())
        throw std::invalid_argument("MANTA periodic BUPO hierarchy and Ewald pair sources must match");
    if(near_cells>direct.plan_.real_cells) throw std::invalid_argument("MANTA periodic BUPO near count exceeds real domain");
    if(options.bound_pair_images && !options.bound_infinite_domain)
        throw std::invalid_argument("MANTA pair-image bounds require the retained-source infinite-domain bound");
    PeriodicBupoPlan p; p.order=tree.source_->order(); p.n_basis=direct.plan_.n_basis; p.momenta=direct.plan_.momenta;
    p.pairs=tree.plan_.pairs; p.nodes=tree.nodes().size(); p.moments=tree.plan_.moments;
    p.maximum_queries=mul(p.nodes,add(p.nodes,1))/2;
    const Count interactions=mul(p.pairs,add(p.pairs,1))/2,dn=mul(p.momenta,square(p.n_basis)),nm=mul(p.nodes,p.moments);
    const auto leaf=unlimited(budget);
    const auto lp=plan_lattice_gaussian_interaction(p.order,direct.plan_.real_cells,direct.plan_.reciprocal_vectors,near_cells,leaf);
    const auto mp=plan_multipoles(p.order,leaf);
    const auto ep=plan_bubble_errors(*tree.tree_,p.order,leaf);
    const auto c=census(tree.source_->basis_,6); const Count m=square(c.maximum_size);
    KernelRequest request; request.kernel="erfc"; request.omega=direct.receipt_.eta;
    const auto tile=plan_md_tiles(tree.source_->basis_,request,leaf);
    const auto fp=plan_fourier(tree.source_->basis_,&tree.source_->basis_,std::max<Count>(1,direct.plan_.reciprocal_batch),leaf);
    const Count real_calls=mul(interactions,direct.plan_.real_cells);
    const Count panel_calls=mul(mul(2,interactions),add(direct.plan_.reciprocal_batches,1));
    p.shell_calls=add(tree.plan_.shell_calls,add(real_calls,panel_calls));
    p.primitive_products=add(tree.plan_.primitive_pairs,add(mul(real_calls,tile.primitive_quartets_upper_bound),mul(panel_calls,square(tile.maximum_primitives))));
    const Count query_scalar=add(lp.scalar_terms,mul(near_cells,mul(4,mp.scalar_terms)));
    const Count query_work=add(lp.kernel_terms,mul(near_cells,add(mp.kernel_terms,mul(256,mp.scalar_terms))));
    p.scalar_terms=add(add(tree.plan_.scalar_terms,ep.scalar_terms),add(mul(2,mul(p.maximum_queries,query_scalar)),
        add(mul(64,mul(interactions,add(mul(square(m),add(direct.plan_.real_cells,direct.plan_.reciprocal_vectors)),mul(m,p.momenta)))),
        add(mul(64,dn),mul(64,nm)))));
    p.kernel_terms=add(add(tree.plan_.kernel_terms,ep.kernel_terms),add(mul(2,mul(p.maximum_queries,query_work)),
        add(mul(real_calls,tile.kernel_terms),add(mul(panel_calls,fp.kernel_terms),mul(64,p.scalar_terms)))));
    p.modeled_peak_bytes=add(budget.other_live_bytes,budget.backend_margin_bytes);
    for(Count bytes:{tree.plan_.modeled_peak_bytes,direct.plan_.modeled_peak_bytes,ep.modeled_peak_bytes,
        lp.modeled_peak_bytes,mp.modeled_peak_bytes,fp.modeled_peak_bytes,tile.modeled_peak_bytes,
        mul(96,near_cells),mul(96,square(m)),mul(128,dn),mul(64,nm),mul(128,p.nodes)})
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,bytes);
    if(options.bound_infinite_domain) {
        p.domain_pair_interactions=interactions;
        Count maximum_terms=0,producer_bytes=0,producer_scalar=0,producer_work=0;
        for(const auto& e:direct.source_->entries_) {
            const auto gp=plan_pair_gaussian_majorant(direct.source_->basis_,e.shells,leaf);
            p.domain_majorant_terms=add(p.domain_majorant_terms,gp.maximum_terms);
            maximum_terms=std::max(maximum_terms,gp.maximum_terms);
            producer_bytes=std::max(producer_bytes,gp.modeled_peak_bytes);
            producer_scalar=add(producer_scalar,gp.scalar_terms);
            producer_work=add(producer_work,gp.kernel_terms);
        }
        const auto dp=plan_lattice_gaussian_domain_remainder(maximum_terms,maximum_terms,
            direct.plan_.real_cells,direct.plan_.reciprocal_vectors,1,leaf);
        // Construction validates every center separation. Build streams the
        // same interactions again, without retaining an O(pair_count^2) cache.
        const Count evaluations=mul(2,interactions);
        p.shell_calls=add(p.shell_calls,p.pairs);
        p.primitive_products=add(p.primitive_products,add(p.domain_majorant_terms,mul(evaluations,dp.primitive_products)));
        p.scalar_terms=add(p.scalar_terms,add(producer_scalar,add(mul(evaluations,dp.scalar_terms),
            add(mul(128,mul(p.pairs,mul(m,p.momenta))),mul(16,square(p.n_basis))))));
        p.kernel_terms=add(p.kernel_terms,add(producer_work,add(mul(evaluations,dp.kernel_terms),
            mul(128,add(mul(p.pairs,mul(m,p.momenta)),square(p.n_basis))))));
        for(Count bytes:{producer_bytes,dp.modeled_peak_bytes,mul(512,p.pairs),
            mul(3*sizeof(GaussianMajorantTerm),p.domain_majorant_terms),mul(32,square(p.n_basis))})
            p.modeled_peak_bytes=add(p.modeled_peak_bytes,bytes);
    }
    if(options.bound_pair_images) {
        const auto ip=pair_image_detail::plan(direct.source_->basis_,direct.source_->entries_,direct.source_->images_,p.momenta,leaf);
        p.image_cube_radius=ip.radius; p.image_cells=ip.cells; p.image_primitive_products=ip.primitive_products;
        p.shell_calls=add(p.shell_calls,ip.shell_calls);
        p.primitive_products=add(p.primitive_products,ip.primitive_products);
        p.scalar_terms=add(p.scalar_terms,ip.scalar_terms); p.kernel_terms=add(p.kernel_terms,ip.kernel_terms);
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,ip.modeled_peak_bytes);
    }
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic BUPO shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA periodic BUPO primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic BUPO scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic BUPO work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic BUPO memory budget exceeded");
    return p;
}

bool PeriodicBupoSource::accepted(const BubbleBlock& block,Far* output) const {
    if(!block.far_candidate || options_.integral_tolerance==0) return false;
    const auto& nodes=tree_->nodes(); const auto& envelopes=errors_->envelopes();
    const auto a=envelope(nodes[block.target],envelopes[block.target]);
    const auto b=envelope(nodes[block.source],envelopes[block.source]);
    // Every bare near image and every singular non-near erfc image must be
    // disjoint. A central-cell geometric candidate is not periodic admission.
    for(const auto& n:direct_->real_) {
        std::array<Real,3> x{},absolute{};
        for(int i=0;i<3;++i) {
            x[i]=a.center[i]-b.center[i]; absolute[i]=std::abs(Real(a.center[i]))+std::abs(Real(b.center[i]));
            for(int j=0;j<3;++j) { const Real v=Real(direct_->source_->lattice_(i,j))*n[j]; x[i]+=v; absolute[i]+=std::abs(v); }
        }
        const Real distance=std::hypot(x[0],x[1],x[2]),eps=std::numeric_limits<double>::epsilon();
        const Real guard=256*eps*std::hypot(absolute[0],absolute[1],absolute[2]);
        if(!std::isfinite(distance) || distance-guard<=(Real(a.radius)+b.radius)*(1+64*eps)) return false;
    }
    try {
        const auto tail=lattice_gaussian_interaction(direct_->source_->lattice_,direct_->receipt_.eta,
            direct_->real_,direct_->reciprocal_,near_,a,b,plan_.order,workspace_budget_);
        Far result; result.values=tail.matrix.values; result.error=tail.error.total;
        Eigen::MatrixXd correction=Eigen::MatrixXd::Zero(result.values.rows(),result.values.cols());
        for(const auto& n:near_) {
            const auto r=vector(direct_->source_->lattice_,n); auto shifted_b=b.center;
            for(int axis=0;axis<3;++axis) shifted_b[axis]-=r[axis];
            const auto near=bupo_detail::interaction(a.center,a.radius,envelopes[block.target],
                shifted_b,b.radius,envelopes[block.source],plan_.order,workspace_budget_);
            result.error=positive_add(result.error,near.error.total);
            for(Eigen::Index i=0;i<result.values.size();++i)
                accumulate(near.matrix.values.data()[i],result.values.data()[i],correction.data()[i]);
        }
        if(result.error>options_.integral_tolerance) return false;
        if(output) *output=std::move(result);
        return true;
    } catch(const std::overflow_error&) {
        return false; // Refine a nonrepresentable far field; never drop it.
    }
}

PeriodicBupoSource::PeriodicBupoSource(std::shared_ptr<const PeriodicBubbleTree> tree,
    std::shared_ptr<const PeriodicEwaldSource> direct,const std::vector<LatticeIndex>& near,
    const PeriodicBupoOptions& options,const ResourceBudget& budget)
    :plan_(plan_source(required_source(tree),required_source(direct),near.size(),options,budget)),
     tree_(std::move(tree)),direct_(std::move(direct)),options_(options),workspace_budget_(unlimited(budget)) {
    near_=lattice_detail::domain(near,true);
    if(!std::includes(direct_->real_.begin(),direct_->real_.end(),near_.begin(),near_.end()))
        throw std::invalid_argument("MANTA periodic BUPO near domain must be contained in the real domain");
    errors_=std::make_shared<BubbleErrorModel>(tree_->tree_,plan_.order,workspace_budget_);
    if(options_.bound_infinite_domain) {
        domain_geometry_=std::make_shared<gaussian_domain_detail::Geometry>(
            gaussian_domain_detail::prepare(direct_->source_->lattice_,direct_->receipt_.eta,
                direct_->real_,direct_->reciprocal_,{{0,0,0}}));
        domain_receipt_.emplace(); auto& r=*domain_receipt_; const auto& g=*domain_geometry_;
        r.pair_source_identity=direct_->pair_source_identity(); r.kernel_identity=direct_->kernel_identity();
        r.real_domain_identity=g.real_identity; r.reciprocal_domain_identity=g.reciprocal_identity;
        r.real_cube_radius=g.real_cube_radius; r.reciprocal_cube_radius=g.reciprocal_cube_radius;
        lattice_detail::Hash envelopes; envelopes.text("manta.periodic-bupo-domain-envelopes.v1");
        envelopes.text(r.pair_source_identity);
        majorants_.reserve(plan_.pairs);
        for(const auto& e:direct_->source_->entries_) {
            majorants_.push_back(compute_pair_gaussian_majorant(direct_->source_->basis_,e.shells,e.translation,workspace_budget_));
            const auto& a=majorants_.back(); envelopes.text(a.source_identity());
            r.majorant_terms+=a.terms().size(); ++r.majorant_shell_calls;
        }
        r.envelope_identity=envelopes.value.finish_hex();
        for(Count p=0;p<plan_.pairs;++p) for(Count q=p;q<plan_.pairs;++q) {
            const auto bound=gaussian_domain_detail::evaluate(g,majorants_[p],majorants_[q],workspace_budget_);
            r.maximum_integral_error=std::max(r.maximum_integral_error,bound.total);
            r.minimum_effective_eta=std::min(r.minimum_effective_eta,bound.minimum_effective_eta);
            r.minimum_omitted_real_distance=std::min(r.minimum_omitted_real_distance,bound.minimum_omitted_real_distance);
        }
        lattice_detail::Hash identity; identity.text("manta.periodic-bupo-infinite-domain.v1;full-gaussian;retained-pairs");
        identity.text(direct_->source_identity()); identity.text(r.envelope_identity);
        r.source_identity=identity.value.finish_hex();
    }
    if(options_.bound_pair_images) {
        const auto& s=*direct_->source_;
        image_envelope_=std::make_shared<pair_image_detail::Envelope>(pair_image_detail::produce(s.basis_,s.lattice_,s.entries_,
            *domain_geometry_,s.receipt_.source_identity,direct_->kernel_identity(),
            pair_image_detail::plan(s.basis_,s.entries_,s.images_,plan_.momenta,workspace_budget_)));
    }
    auto traversal=tree_->traversal(options_.opening,workspace_budget_);
    while(const auto pending=traversal->peek()) {
        Far far; const bool approx=accepted(*pending,&far);
        if(pending->far_candidate && !approx) { traversal->refine(); continue; }
        record(partition_,*pending,approx,far.error); traversal->accept();
    }
    finish(partition_,traversal->receipt());
    lattice_detail::Hash h; h.text("manta.periodic-bupo-source.v1;fixed-partition;full-raw-cartesian;gaussian-ledger");
    h.text(tree_->source_identity()); h.text(direct_->source_identity()); h.text(errors_->source_identity());
    h.text(lattice_detail::domain_identity("near",near_)); h.text(partition_.identity);
    h.number(options.opening); h.number(options.integral_tolerance);
    if(domain_receipt_) {
        h.text("infinite-domain-bound.v1"); h.text(domain_receipt_->source_identity);
    }
    if(image_envelope_) { h.text("pair-image-bound.v1"); h.text(image_envelope_->receipt.source_identity); }
    source_identity_=h.value.finish_hex();
}

void PeriodicBupoSource::bound_domain(const std::vector<Complex>& d,PeriodicBupoResult& out) const {
    auto& r=out.receipt; r.finite_energy_error_bound=r.energy_error_bound;
    r.finite_maximum_j_error_bound=r.maximum_j_error_bound;
    if(!domain_receipt_) return;
    r.qualification="periodic-bupo-j-with-retained-source-ewald-bound";
    r.domain=domain_receipt_; auto& domain=*r.domain; domain.density_identity=r.density_identity;
    const auto& source=*direct_->source_; const auto& entries=source.entries_;
    std::vector<double> norms(plan_.pairs),potential(plan_.pairs);
    for(Count p=0;p<plan_.pairs;++p) {
        Real norm=minus_inf;
        for(const auto& c:coefficients(entries[p],plan_.n_basis,source.momenta_,source.weights_,d))
            if(c.normalized!=0 && c.scale!=0)
                norm=logadd(norm,std::log(std::abs(c.normalized))+std::log(Real(c.scale)));
        norms[p]=upper(norm);
    }
    // The infinite-minus-finite integral is uniformly bounded for every AO
    // component by native full Gaussian majorants. Ewald1921 IV.1 Eqs.37-40;
    // the Gaussian convolution and density aggregation are derived in the
    // scientific contract, not attributed to the point-charge formula.
    for(Count p=0;p<plan_.pairs;++p) for(Count q=p;q<plan_.pairs;++q) {
        const auto bound=gaussian_domain_detail::evaluate(*domain_geometry_,majorants_[p],majorants_[q],workspace_budget_);
        const auto energy=[&](double delta) { return p==q?diagonal_product(delta,norms[p]):positive_product(delta,norms[p],norms[q]); };
        domain.real_energy_error_bound=positive_add(domain.real_energy_error_bound,energy(bound.real_remainder));
        domain.reciprocal_energy_error_bound=positive_add(domain.reciprocal_energy_error_bound,energy(bound.reciprocal_remainder));
        potential[p]=positive_add(potential[p],positive_product(bound.total,norms[q]));
        if(p!=q) potential[q]=positive_add(potential[q],positive_product(bound.total,norms[p]));
        ++domain.pair_interactions; domain.majorant_pairs_evaluated+=bound.majorant_pairs_evaluated;
    }
    out.domain_j_error_bound=md::Matrix::Zero(plan_.n_basis,plan_.n_basis);
    for(Count p=0;p<plan_.pairs;++p) scatter_bound(entries[p],potential[p],*out.domain_j_error_bound);
    domain.maximum_j_error_bound=out.domain_j_error_bound->maxCoeff();
    domain.energy_error_bound=positive_add(domain.real_energy_error_bound,domain.reciprocal_energy_error_bound);
    r.energy_error_bound=positive_add(r.finite_energy_error_bound,domain.energy_error_bound);
    for(Eigen::Index i=0;i<out.j_error_bound.size();++i)
        out.j_error_bound.data()[i]=positive_add(out.j_error_bound.data()[i],out.domain_j_error_bound->data()[i]);
    r.maximum_j_error_bound=out.j_error_bound.maxCoeff();
    bound_pair_images(d,out);
}

void PeriodicBupoSource::bound_pair_images(const std::vector<Complex>& d,PeriodicBupoResult& out) const {
    if(!image_envelope_) return;
    auto& r=out.receipt; r.pair_images=image_envelope_->receipt; auto& image=*r.pair_images;
    r.qualification="periodic-bupo-j-with-full-pair-image-ewald-bound";
    image.density_identity=r.density_identity;
    out.retained_pair_mass_bound=image_envelope_->retained; out.omitted_pair_mass_bound=image_envelope_->omitted;
    out.pair_image_j_error_bound=md::Matrix::Zero(plan_.n_basis,plan_.n_basis);
    const auto& keep=image_envelope_->retained; const auto& omit=image_envelope_->omitted;
    const auto& w=direct_->source_->weights_; const Count n=plan_.n_basis;
    Real qr=minus_inf,qm=minus_inf;
    for(Count a=0;a<n;++a) for(Count b=0;b<n;++b) {
        Real norm=minus_inf;
        for(Count k=0;k<plan_.momenta;++k) {
            const auto x=d[(k*n+a)*n+b],y=d[(k*n+b)*n+a];
            // Exact Hermitian densities have identical magnitudes. Taking
            // the larger arm also covers the accepted Hermiticity tolerance.
            const Real magnitude=std::max(std::hypot(Real(x.real()),Real(x.imag())),std::hypot(Real(y.real()),Real(y.imag())));
            if(w[k]!=0 && magnitude!=0) norm=logadd(norm,std::log(Real(w[k]))+std::log(magnitude));
        }
        if(keep(a,b)!=0 && norm!=minus_inf) qr=logadd(qr,norm+std::log(Real(keep(a,b))));
        if(omit(a,b)!=0 && norm!=minus_inf) qm=logadd(qm,norm+std::log(Real(omit(a,b))));
    }
    image.retained_density_mass_bound=upper(qr); image.omitted_density_mass_bound=upper(qm);
    const Real c=std::log(Real(image.kernel_bound)),qfull=logadd(qr,qm);
    image.energy_error_bound=qm==minus_inf?0:upper(c+qm+logadd(qr,qm-std::log(2.L)));
    for(Count a=0;a<n;++a) for(Count b=0;b<n;++b) {
        // Include missing targets as well as missing density. The ordered
        // image masses already contain both canonical arms, including twice
        // for a nonzero-image diagonal AO component.
        const Real retained=keep(a,b)?std::log(Real(keep(a,b)))+qm:minus_inf;
        const Real missing=omit(a,b)?std::log(Real(omit(a,b)))+qfull:minus_inf;
        const double bound=upper(c+logadd(retained,missing));
        (*out.pair_image_j_error_bound)(a,b)=bound;
        out.j_error_bound(a,b)=positive_add(out.j_error_bound(a,b),bound);
    }
    image.maximum_j_error_bound=out.pair_image_j_error_bound->maxCoeff();
    r.energy_error_bound=positive_add(r.energy_error_bound,image.energy_error_bound);
    r.maximum_j_error_bound=out.j_error_bound.maxCoeff();
}

PeriodicBupoResult PeriodicBupoSource::build(const std::vector<Complex>& d) const {
    PeriodicBupoResult out; out.plan=plan_; auto& r=out.receipt;
    r.partition=partition_; r.source_identity=source_identity_; r.tree_identity=tree_->source_identity();
    r.pair_source_identity=tree_->pair_source_identity(); r.kernel_identity=direct_->kernel_identity(); r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    const auto& source=*direct_->source_; const auto& entries=source.entries_; const auto& k=source.momenta_; const auto& weights=source.weights_;
    const Count n=plan_.n_basis,nk=plan_.momenta; out.j_error_bound=md::Matrix::Zero(n,n);
    if(!partition_.approximate_blocks) {
        auto all=direct_->build(d); out.values=std::move(all.values); r.direct=std::move(all.receipt);
        r.direct_energy=r.energy=r.direct.energy; r.trace_energy=r.direct.trace_energy; r.charge=r.direct.charge;
        r.density_identity=r.direct.density_identity; r.hermiticity_residual=r.direct.response_hermiticity_residual;
        bound_domain(d,out); return out;
    }
    r.direct=direct_->receipt_; r.direct.qualification="finite-ewald-direct-partition-subset";
    lattice_detail::Hash subset; subset.text("manta.periodic-ewald-direct-subset.v1");
    subset.text(direct_->source_identity()); subset.text(partition_.identity);
    r.direct.source_identity=subset.value.finish_hex();
    validate_density(source,d,r.direct); r.density_identity=r.direct.density_identity;
    const auto forward=tree_->contract(d); r.charge=forward.pair_receipt.charge;
    r.direct.charge=r.charge; // Charge of the retained source, not a block energy.
    r.forward_shell_calls=forward.pair_receipt.shell_calls; r.translation_calls=forward.translation_calls;
    out.values.resize(nk*n*n); std::vector<Complex> correction(out.values.size());
    md::Matrix local=md::Matrix::Zero(plan_.nodes,plan_.moments),local_error=local;
    std::vector<double> node_error(plan_.nodes);
    std::array<Real,3> energies{},energy_correction{}; Real far_energy=0,far_correction=0;
    auto traversal=tree_->traversal(options_.opening,workspace_budget_); PeriodicBupoPartition executed;
    while(const auto pending=traversal->peek()) {
        const auto block=*pending; Far far; const bool approx=accepted(block,&far);
        if(block.far_candidate && !approx) { traversal->refine(); continue; }
        if(approx) {
            const auto a=block.target,b=block.source;
            for(Count i=0;i<plan_.moments;++i) for(Count j=0;j<plan_.moments;++j) {
                const double value=far.values(i,j); if(value==0) continue;
                accumulate(number(Real(value)*forward.moments(b,j)),local(a,i),local_error(a,i));
                accumulate(number(Real(value)*forward.moments(a,i)),local(b,j),local_error(b,j));
                accumulate(Real(value)*forward.moments(a,i)*forward.moments(b,j),far_energy,far_correction);
                ++r.far_matrix_terms;
            }
            const auto la=forward.coefficient_l1[a],lb=forward.coefficient_l1[b];
            r.energy_error_bound=positive_add(r.energy_error_bound,positive_product(far.error,la,lb));
            node_error[a]=positive_add(node_error[a],positive_product(far.error,lb));
            node_error[b]=positive_add(node_error[b],positive_product(far.error,la));
        } else pairs(*tree_,block,[&](Count p,Count q) {
            const auto cp=coefficients(entries[p],n,k,weights,d),cq=coefficients(entries[q],n,k,weights,d);
            r.direct.coefficient_panels+=2;
            const auto matrices=direct_->interaction(p,q,r.direct);
            std::vector<Real> vp(cp.size()),ep(cp.size()),vq(cq.size()),eq(cq.size());
            for(int component=0;component<3;++component) {
                const auto& matrix=matrices[component];
                for(Count i=0;i<cp.size();++i) {
                    Real sum=0,error=0;
                    for(Count j=0;j<cq.size();++j) {
                        const auto x=cq[j].times(matrix(i,j)); accumulate(x,sum,error); accumulate(x,vp[i],ep[i]);
                        if(p!=q) accumulate(cp[i].times(matrix(i,j)),vq[j],eq[j]);
                    }
                    accumulate((p==q?Real(.5):Real(1))*cp[i].times(number(sum)),energies[component],energy_correction[component]);
                }
            }
            scatter(entries[p],n,k,vp,out.values,correction);
            if(p!=q) scatter(entries[q],n,k,vq,out.values,correction);
        });
        record(executed,block,approx,far.error); traversal->accept();
    }
    finish(executed,traversal->receipt());
    if(executed.identity!=partition_.identity) throw std::logic_error("MANTA periodic BUPO partition changed between setup and build");
    Real direct_trace=0,direct_trace_error=0; double direct_scale=0;
    for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=out.values[(ik*n+i)*n+j],x=d[(ik*n+j)*n+i]; finite(v); require_finite(std::abs(v));
        direct_scale=std::max(direct_scale,std::abs(v));
        r.direct.response_hermiticity_residual=std::max(r.direct.response_hermiticity_residual,
            std::abs(v-std::conj(out.values[(ik*n+j)*n+i])));
        accumulate(Real(.5)*weights[ik]*(Real(x.real())*v.real()-Real(x.imag())*v.imag()),direct_trace,direct_trace_error);
    }
    r.direct.trace_energy=number(direct_trace); require_finite(r.direct.response_hermiticity_residual);
    if(r.direct.response_hermiticity_residual>1e-11*std::max(1.,direct_scale))
        throw std::runtime_error("MANTA periodic BUPO direct subset violates Hermiticity");
    const auto reverse=tree_->adjoint(local); r.adjoint_shell_calls=reverse.pair.shell_calls; r.translation_calls+=reverse.translation_calls;
    for(Count i=0;i<out.values.size();++i) accumulate(reverse.pair.values[i],out.values[i],correction[i]);
    // Ancestor block errors add on each descendant. The absolute phase scatter
    // counts both conjugate arms without using cancellation or TR symmetry.
    for(Count id=0;id<plan_.nodes;++id) {
        const auto& node=tree_->nodes()[id];
        if(id) node_error[id]=positive_add(node_error[id],node_error[node.parent]);
        if(!node.child_count) for(Count pos=node.begin;pos<node.end;++pos) {
            const auto& e=entries[tree_->pair_order()[pos]];
            scatter_bound(e,node_error[id],out.j_error_bound);
        }
    }
    r.direct.real_energy=number(energies[0]); r.direct.reciprocal_energy=number(energies[1]); r.direct.gauge_energy=number(energies[2]);
    r.direct.energy=r.direct_energy=number(energies[0]+energies[1]+energies[2]); r.far_energy=number(far_energy);
    r.energy=number(Real(r.direct_energy)+r.far_energy); r.maximum_j_error_bound=out.j_error_bound.maxCoeff();
    Real trace=0,trace_error=0; double scale=0;
    for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=out.values[(ik*n+i)*n+j]; finite(v); require_finite(std::abs(v)); scale=std::max(scale,std::abs(v));
        r.hermiticity_residual=std::max(r.hermiticity_residual,std::abs(v-std::conj(out.values[(ik*n+j)*n+i])));
        const auto x=d[(ik*n+j)*n+i];
        accumulate(Real(.5)*weights[ik]*(Real(x.real())*v.real()-Real(x.imag())*v.imag()),trace,trace_error);
    }
    r.trace_energy=number(trace); require_finite(r.hermiticity_residual);
    if(r.hermiticity_residual>1e-11*std::max(1.,scale)) throw std::runtime_error("MANTA periodic BUPO response violates Hermiticity");
    bound_domain(d,out); return out;
}
} // namespace vibeqc::manta
