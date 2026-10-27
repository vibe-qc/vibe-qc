#include "vibeqc/manta_periodic_nuclear_bounds.hpp"
#include "manta_periodic_nuclear_internal.hpp"
#include "manta_pair_image_internal.hpp"
#include "manta_lattice_geometry.hpp"
#include "vibeqc/lattice_pair_cells.hpp"

namespace vibeqc::manta {
// Ewald (1921), IV.1 Eqs.37-40, DOI10.1002/andp.19213690304, and
// Sun et al. (2017), Sec.II.A Eq.16, DOI10.1063/1.4998644 define the
// kernel/phase conventions. The completion proof is derived in the
// scientific contract Section5.17; it does not change the physical source.
namespace {
using namespace admission;
using Real=long double;
constexpr Real neg=-std::numeric_limits<Real>::infinity();
constexpr Real eps=std::numeric_limits<double>::epsilon();
Real addlog(Real a,Real b) { if(a==neg)return b;if(b==neg)return a;const Real hi=std::max(a,b);return hi+std::log1p(std::exp(std::min(a,b)-hi)); }
Real logarithm(double x) { return x?std::log(Real(x)):neg; }
double upper(Real x) { if(x==neg)return 0;const double y=std::nextafter(double(std::exp(x)*(1+8192*eps)),std::numeric_limits<double>::infinity());if(!std::isfinite(y))throw std::overflow_error("MANTA nuclear bound exceeds binary64");return y; }
double sum(double a,double b) { return upper(addlog(logarithm(a),logarithm(b))); }
double product(double a,double b) { return a&&b?upper(logarithm(a)+logarithm(b)):0.; }
ResourceBudget unlimited(const ResourceBudget& b) { auto u=b;u.other_live_bytes=0;u.memory_bytes=u.shell_calls=u.primitive_quartets=u.scalar_terms=u.kernel_terms=std::numeric_limits<Count>::max();return u; }
Count cube(Count r) { if(r>2147483647)throw std::invalid_argument("MANTA nuclear completion cube exceeds signed 31-bit labels");const Count w=add(mul(2,r),1);return mul(square(w),w); }
void caps(const PeriodicNuclearBoundPlan& p,const ResourceBudget& b) {
    cap(p.shell_calls,b.shell_calls,"MANTA nuclear bound shell budget exceeded");
    cap(p.primitive_pairs,b.primitive_quartets,"MANTA nuclear bound primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA nuclear bound scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA nuclear bound kernel budget exceeded");
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA nuclear bound memory budget exceeded");
}
struct ShiftedPoint { md::Point point;double uncertainty; };
ShiftedPoint shifted(const PeriodicSystem& s,Count atom,const LatticeIndex& n) {
    ShiftedPoint out;Real error=0;
    for(int a=0;a<3;++a) {
        Real value=s.unit_cell[atom].xyz[a],scale=std::abs(value);
        for(int b=0;b<3;++b) { const Real x=Real(s.lattice(a,b))*n[b];value+=x;scale+=std::abs(x); }
        out.point[a]=double(value);require_finite(out.point[a]);
        error=std::hypot(error,256*eps*scale+256*std::numeric_limits<double>::denorm_min());
    }
    out.uncertainty=std::nextafter(double(error*(1+256*eps)),std::numeric_limits<double>::infinity());
    require_finite(out.uncertainty);return out;
}
LatticeIndex recenter(const PeriodicSystem& s,const Eigen::Matrix3d& inverse,Count atom,const Eigen::Vector3d& center,Count radius) {
    const auto& xyz=s.unit_cell[atom].xyz;
    const Eigen::Vector3d f=inverse*(center-Eigen::Vector3d(xyz[0],xyz[1],xyz[2]));
    LatticeIndex n{};
    for(int a=0;a<3;++a) { const double x=std::round(f[a]);if(!std::isfinite(x)||std::abs(x)>double(std::numeric_limits<int>::max())-double(radius)-2)throw std::invalid_argument("MANTA nuclear completion index exceeds supported range");n[a]=static_cast<std::int64_t>(x); }
    return n;
}
using Labels=std::vector<std::array<int,4>>;
bool retained(const Labels& labels,Count atom,const LatticeIndex& n) {
    return std::binary_search(labels.begin(),labels.end(),std::array<int,4>{int(atom),int(n[0]),int(n[1]),int(n[2])});
}
void scatter(md::Matrix& m,Count a,Count b,Count na,Count nb,double value) {
    for(Count i=0;i<na;++i)for(Count j=0;j<nb;++j)m(a+i,b+j)=sum(m(a+i,b+j),value);
}
} // namespace

PeriodicNuclearBoundPlan plan_periodic_nuclear_bounds(const BasisSet& basis,const PeriodicSystem& s,
    Count ng,Count nc,Count nk,double alpha,double cutoff,std::optional<double> paircut,
    const PeriodicNuclearBoundOptions& o,const ResourceBudget& b) {
    margin(b);if(!nc||!nk)throw std::invalid_argument("MANTA nuclear bounds require nonempty AO and k supports");
    if(s.dim!=3||!s.lattice.allFinite()||!s.lattice.inverse().allFinite()||!std::isfinite(alpha)||alpha<=0||!std::isfinite(cutoff)||cutoff<0)
        throw std::invalid_argument("MANTA nuclear bounds require a finite nonsingular 3D lattice and valid split/cutoff");
    cap(s.unit_cell.size(),std::numeric_limits<int>::max(),"MANTA nuclear bound point count exceeds supported range");
    const auto leaf=unlimited(b);PeriodicNuclearBoundPlan p;
    p.n_basis=basis.nbasis();p.points=s.unit_cell.size();p.momenta=nk;p.vectors=ng;
    p.ao_cube_cells=cube(o.ao_completion_cube_radius);p.real_cube_cells=cube(o.real_completion_cube_radius);p.reciprocal_cube_cells=cube(o.reciprocal_completion_cube_radius);
    PeriodicPairOptions po;po.order=0;po.pair_cutoff_bohr=paircut;
    p.pairs=plan_periodic_pairs(basis,nc,nk,po,leaf);
    p.maximum_ordered_pairs=mul(2,p.pairs.maximum_pairs);
    p.maximum_nuclear_candidates_per_pair=nuclear_detail::candidate_bound(s,cutoff);
    const auto images=pair_image_detail::plan(basis,p.pairs.maximum_pairs,nk,o.ao_completion_cube_radius,leaf);
    Count majorant_bytes=0,majorant_scalar=0,majorant_kernel=0,majorant_primitives=0;
    for(Count a=0;a<basis.nshells();++a)for(Count c=a;c<basis.nshells();++c) {
        const auto gp=plan_pair_gaussian_majorant(basis,{a,c},leaf);
        p.maximum_gaussian_terms=std::max(p.maximum_gaussian_terms,gp.maximum_terms);
        majorant_bytes=std::max(majorant_bytes,gp.modeled_peak_bytes);
        majorant_scalar=std::max(majorant_scalar,gp.scalar_terms);majorant_kernel=std::max(majorant_kernel,gp.kernel_terms);
        majorant_primitives=std::max(majorant_primitives,gp.primitive_pairs);
    }
    const Count queries=add(p.maximum_ordered_pairs,p.points);
    p.real_terms=mul(mul(queries,p.points),add(p.real_cube_cells,1));
    p.shell_calls=add(p.pairs.shell_calls,add(images.shell_calls,p.maximum_ordered_pairs));
    p.primitive_pairs=add(p.pairs.primitive_pairs,add(images.primitive_products,mul(p.maximum_ordered_pairs,majorant_primitives)));
    Count levels=1;for(Count count=std::max({ng,nc,p.maximum_nuclear_candidates_per_pair});count>1;count>>=1)++levels;
    const Count scalar=add(mul(p.maximum_ordered_pairs,majorant_scalar),add(mul(1024,mul(p.real_terms,add(p.maximum_gaussian_terms,levels))),
        add(mul(1024,mul(queries,mul(p.maximum_nuclear_candidates_per_pair,levels))),mul(1024,mul(levels,add(ng,add(nc,p.reciprocal_cube_cells)))))));
    p.scalar_terms=add(p.pairs.scalar_terms,add(images.scalar_terms,add(scalar,mul(256,mul(nk,square(p.n_basis))))));
    p.kernel_terms=add(p.pairs.kernel_terms,add(images.kernel_terms,add(mul(p.maximum_ordered_pairs,majorant_kernel),mul(128,p.scalar_terms))));
    p.modeled_peak_bytes=add(b.other_live_bytes,add(p.pairs.modeled_peak_bytes,add(images.modeled_peak_bytes,add(majorant_bytes,
        add(65536,add(mul(128,p.maximum_nuclear_candidates_per_pair),add(mul(256,square(p.n_basis)),mul(256,add(ng,add(nc,add(nk,p.points)))))))))));
    caps(p,b);return p;
}

PeriodicNuclearBoundResult bound_periodic_nuclear(const BasisSet& basis,const PeriodicSystem& s,
    const md::Matrix& g,const md::Matrix& cells,const md::Matrix& k,double alpha,double cutoff,
    std::optional<double> paircut,const PeriodicNuclearBoundOptions& o,const ResourceBudget& b) {
    const auto p=plan_periodic_nuclear_bounds(basis,s,g.rows(),cells.rows(),k.rows(),alpha,cutoff,paircut,o,b);
    if(g.cols()!=3||cells.cols()!=3||k.cols()!=3||!g.allFinite()||!cells.allFinite()||!k.allFinite())throw std::invalid_argument("MANTA nuclear bounds require finite Cartesian vectors");
    validate_basis(basis);for(const auto& atom:s.unit_cell)for(double x:atom.xyz)require_finite(x);
    const auto glabels=nuclear_detail::closed_support(g,s.lattice.transpose()/(2*std::acos(-1.)),true);
    // Physical vectors and integer reciprocal labels must describe the same
    // source to roundoff, not merely have approximately integral fractions.
    const Eigen::Matrix3d reciprocal=2*std::acos(-1.)*s.lattice.inverse().transpose();
    for(Eigen::Index row=0;row<g.rows();++row) {
        const Eigen::Vector3d fractional=s.lattice.transpose()*g.row(row).transpose()/(2*std::acos(-1.));
        for(int axis=0;axis<3;++axis) {
            Real value=0,scale=1+std::abs(Real(g(row,axis)));
            for(int d=0;d<3;++d) { const Real t=Real(reciprocal(axis,d))*std::round(fractional[d]);value+=t;scale+=std::abs(t); }
            if(std::abs(value-g(row,axis))>256*eps*scale)throw std::invalid_argument("MANTA nuclear reciprocal support differs from its lattice images beyond roundoff");
        }
    }
    auto leaf=unlimited(b); // Every nested operation is reserved by the joint plan.
    const auto pairs=make_periodic_accuracy_pair_source(basis,s.lattice,cells,k,paircut,leaf);
    // Only lattice metrics and the uniform periodized point kernel are used
    // from this shared geometry. The actual noncubic G inventory is handled
    // by reciprocal_exchange_sums, including its holes and exterior.
    auto geometry=gaussian_domain_detail::prepare(s.lattice,alpha,{{0,0,0}},{},{{0,0,0}});
    geometry.real_cube_radius=o.real_completion_cube_radius;
    const auto image_plan=pair_image_detail::plan(basis,pairs.entries().size(),k.rows(),o.ao_completion_cube_radius,leaf);
    const auto images=pair_image_detail::produce(basis,s.lattice,pairs.entries(),geometry,pairs.receipt().source_identity,geometry.kernel_identity,image_plan);
    PeriodicNuclearBoundResult out;out.plan=p;out.options=o;auto& r=out.receipt;
    r.alpha=alpha;r.nuclear_cutoff_bohr=cutoff;r.pair_source_identity=pairs.receipt().source_identity;r.pair_image_identity=images.receipt.source_identity;
    r.nuclear_source_identity=nuclear_detail::source_identity(basis,s,g,cells,k,alpha,cutoff,paircut);
    Real charge=neg;for(const auto& atom:s.unit_cell)if(atom.Z)charge=addlog(charge,std::log(std::abs(Real(atom.Z))));r.nuclear_absolute_charge=upper(charge);
    const Real logdecay=-std::log(8.L)-2*std::log(Real(alpha));
    const double decay=std::nextafter(double(std::exp(logdecay)*(1-256*eps)),0.);
    if(!std::isfinite(decay)||decay<=0)throw std::invalid_argument("MANTA nuclear reciprocal decay is not representable");
    r.reciprocal_weight_bound=pair_image_detail::reciprocal_exchange_sums(s.lattice,Eigen::Vector3d::Zero(),glabels,decay,o.reciprocal_completion_cube_radius)[1];
    r.point_kernel_bound=pair_image_detail::point_kernel_bound(images,geometry);
    const Count n=p.n_basis;out.real_error_bound=out.reciprocal_error_bound=out.pair_image_error_bound=out.potential_error_bound=md::Matrix::Zero(n,n);
    // Reuse physical selector labels rather than reimplementing its floating
    // sphere boundary. Each ordered arm gets its original Cartesian cell.
    const Eigen::Matrix3d inverse=s.lattice.inverse();
    std::vector<std::pair<LatticeIndex,Count>> ordered;ordered.reserve(cells.rows());
    for(Count c=0;c<Count(cells.rows());++c) { const Eigen::Vector3d f=inverse*cells.row(c).transpose();LatticeIndex label{};for(int a=0;a<3;++a)label[a]=static_cast<std::int64_t>(std::round(f[a]));ordered.push_back({label,c}); }
    std::sort(ordered.begin(),ordered.end());
    const PairNuclearImageSelector selector(s,cutoff);const auto radius=static_cast<std::int64_t>(o.real_completion_cube_radius);
    auto select=[&](const Eigen::Vector3d& center) { Labels labels;selector.select(center,nullptr,&labels,p.maximum_nuclear_candidates_per_pair,true);std::sort(labels.begin(),labels.end());return labels; };
    auto arm=[&](Count a,Count c,Count oa,Count oc,Count na,Count nc,const LatticeIndex& label) {
        const auto it=std::lower_bound(ordered.begin(),ordered.end(),std::make_pair(label,Count(0)));
        if(it==ordered.end()||it->first!=label)throw std::logic_error("MANTA nuclear bound lost an AO image");
        md::Point translation{};for(int d=0;d<3;++d)translation[d]=cells(it->second,d);
        auto ket=basis.libint()[c];for(int d=0;d<3;++d)ket.O[d]+=translation[d];
        const auto& bra=basis.libint()[a];const Eigen::Vector3d center(.5*(bra.O[0]+ket.O[0]),.5*(bra.O[1]+ket.O[1]),.5*(bra.O[2]+ket.O[2]));
        const auto labels=select(center);const auto majorant=compute_pair_gaussian_majorant(basis,{a,c},translation,leaf);
        ++r.ordered_pairs;r.majorant_primitive_pairs+=majorant.plan().primitive_pairs;Real real=neg;
        for(Count atom=0;atom<s.unit_cell.size();++atom)if(s.unit_cell[atom].Z) {
            const Real logz=std::log(std::abs(Real(s.unit_cell[atom].Z)));const auto n0=recenter(s,inverse,atom,center,o.real_completion_cube_radius);
            for(std::int64_t i=-radius;i<=radius;++i)for(std::int64_t j=-radius;j<=radius;++j)for(std::int64_t l=-radius;l<=radius;++l) {
                const LatticeIndex image{n0[0]+i,n0[1]+j,n0[2]+l};++r.real_terms;if(retained(labels,atom,image))continue;
                const auto point=shifted(s,atom,image);real=addlog(real,logz+logarithm(gaussian_domain_detail::gaussian_point_real_image(alpha,majorant,point.point,point.uncertainty)));
            }
            const auto point=shifted(s,atom,n0);++r.real_terms;
            real=addlog(real,logz+logarithm(gaussian_domain_detail::gaussian_point_real_tail(geometry,majorant,point.point,point.uncertainty)));
        }
        scatter(out.real_error_bound,oa,oc,na,nc,upper(real));
        scatter(out.reciprocal_error_bound,oa,oc,na,nc,product(product(majorant.charge_bound(),r.nuclear_absolute_charge),r.reciprocal_weight_bound));
    };
    for(const auto& e:pairs.entries()) {
        arm(e.shells[0],e.shells[1],e.offsets[0],e.offsets[1],e.sizes[0],e.sizes[1],e.image);
        if(e.paired)arm(e.shells[1],e.shells[0],e.offsets[1],e.offsets[0],e.sizes[1],e.sizes[0],{-e.image[0],-e.image[1],-e.image[2]});
    }
    for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
        out.pair_image_error_bound(i,j)=product(product(r.point_kernel_bound,r.nuclear_absolute_charge),images.omitted(i,j));
        out.potential_error_bound(i,j)=sum(sum(out.real_error_bound(i,j),out.reciprocal_error_bound(i,j)),out.pair_image_error_bound(i,j));
        r.maximum_potential_error=std::max(r.maximum_potential_error,out.potential_error_bound(i,j));
    }
    Real nn=neg;
    for(Count a=0;a<s.unit_cell.size();++a)if(s.unit_cell[a].Z) {
        const auto& xyz=s.unit_cell[a].xyz;const md::Point origin{xyz[0],xyz[1],xyz[2]};const Eigen::Vector3d center(xyz[0],xyz[1],xyz[2]);const auto labels=select(center);
        for(Count c=0;c<s.unit_cell.size();++c)if(s.unit_cell[c].Z) {
            const Real logz=std::log(std::abs(Real(s.unit_cell[a].Z)))+std::log(std::abs(Real(s.unit_cell[c].Z)))-std::log(2.L);
            const auto n0=recenter(s,inverse,c,center,o.real_completion_cube_radius);
            for(std::int64_t i=-radius;i<=radius;++i)for(std::int64_t j=-radius;j<=radius;++j)for(std::int64_t l=-radius;l<=radius;++l) {
                const LatticeIndex image{n0[0]+i,n0[1]+j,n0[2]+l};++r.real_terms;if(a==c&&image==LatticeIndex{})continue;
                const auto point=shifted(s,c,image);
                // Also validate retained neighbors: a singular infinite source
                // cannot acquire a certificate by omitting its finite error.
                const double value=gaussian_domain_detail::point_real_image(alpha,origin,point.point,point.uncertainty);
                if(!retained(labels,c,image))nn=addlog(nn,logz+logarithm(value));
            }
            const auto point=shifted(s,c,n0);++r.real_terms;
            nn=addlog(nn,logz+logarithm(gaussian_domain_detail::point_real_tail(geometry,origin,point.point,point.uncertainty)));
        }
    }
    r.nuclear_real_error_bound=upper(nn);
    r.nuclear_reciprocal_error_bound=upper(2*charge+logarithm(r.reciprocal_weight_bound)-std::log(2.L));
    r.nuclear_energy_error_bound=sum(r.nuclear_real_error_bound,r.nuclear_reciprocal_error_bound);
    lattice_detail::Hash h;h.text("manta.periodic-nuclear-bound.v1;fixed-source;full-images;zero-average;no-roundoff");
    h.text(r.nuclear_source_identity);h.text(r.pair_source_identity);h.text(r.pair_image_identity);
    h.word(o.ao_completion_cube_radius);h.word(o.real_completion_cube_radius);h.word(o.reciprocal_completion_cube_radius);
    for(const auto* matrix:{&out.real_error_bound,&out.reciprocal_error_bound,&out.pair_image_error_bound,&out.potential_error_bound})for(Eigen::Index i=0;i<matrix->size();++i)h.number(matrix->data()[i]);
    h.number(r.nuclear_real_error_bound);h.number(r.nuclear_reciprocal_error_bound);r.source_identity=h.value.finish_hex();
    r.shell_calls=add(pairs.receipt().pairs,add(images.receipt.shell_pairs_evaluated,r.ordered_pairs));
    r.primitive_pairs=add(pairs.receipt().primitive_pairs,add(images.receipt.primitive_pairs_evaluated,r.majorant_primitive_pairs));
    if(r.shell_calls>p.shell_calls||r.primitive_pairs>p.primitive_pairs||r.ordered_pairs>p.maximum_ordered_pairs||r.real_terms>p.real_terms)throw std::logic_error("MANTA nuclear bound count reservation violated");
    return out;
}

PeriodicNuclearAccuracyPlan plan_periodic_nuclear_accuracy(const BasisSet& basis,const PeriodicSystem& s,
    Count ng,Count nc,Count nk,double alpha,double cutoff,std::optional<double> paircut,
    const PeriodicNuclearAccuracyOptions& o,const ResourceBudget& b) {
    margin(b);
    for(double x:{o.matrix_error,o.attraction_energy_error,o.nuclear_energy_error})
        if(!std::isfinite(x)||x<0)throw std::invalid_argument("MANTA nuclear accuracy targets must be finite nonnegative");
    PeriodicNuclearAccuracyPlan p;if(!o.enabled())return p;
    if(o.matrix_error<=0||o.attraction_energy_error<=0||o.nuclear_energy_error<=0)
        throw std::invalid_argument("MANTA nuclear accuracy requires three positive targets");
    p.nuclear=plan_periodic_nuclear_bounds(basis,s,ng,nc,nk,alpha,cutoff,paircut,o,unlimited(b));
    const Count mesh=mul(nk,square(basis.nbasis()));
    // The parent owns one canonical envelope. Reserve positive trace scratch,
    // retained receipts and owned result exports before any payload copy.
    p.modeled_peak_bytes=add(b.other_live_bytes,add(p.nuclear.modeled_peak_bytes,add(32768,mul(64,mesh))));
    p.shell_calls=p.nuclear.shell_calls;p.primitive_pairs=p.nuclear.primitive_pairs;
    p.scalar_terms=add(p.nuclear.scalar_terms,mul(256,mesh));
    p.kernel_terms=add(p.nuclear.kernel_terms,mul(128,p.scalar_terms));
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA nuclear accuracy memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA nuclear accuracy shell budget exceeded");
    cap(p.primitive_pairs,b.primitive_quartets,"MANTA nuclear accuracy primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA nuclear accuracy scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA nuclear accuracy kernel budget exceeded");return p;
}
PeriodicNuclearAccuracyResult bound_periodic_nuclear_accuracy(const BasisSet& basis,const PeriodicSystem& s,
    const md::Matrix& g,const md::Matrix& cells,const md::Matrix& k,double alpha,double cutoff,
    std::optional<double> paircut,const BoundedPeriodicDensityEnvelope& density,
    const PeriodicNuclearAccuracyOptions& o,const ResourceBudget& b,bool enforce_targets) {
    const auto p=plan_periodic_nuclear_accuracy(basis,s,g.rows(),cells.rows(),k.rows(),alpha,cutoff,paircut,o,b);
    if(!o.enabled())throw std::invalid_argument("MANTA nuclear accuracy requires enabled targets");
    const Count n=basis.nbasis(),nk=k.rows(),mesh=mul(nk,square(n));
    if(density.weighted_entry_bounds.size()!=mesh||!std::isfinite(density.density_l1_limit)||density.density_l1_limit<=0)
        throw std::invalid_argument("MANTA nuclear accuracy requires the canonical density envelope");
    std::vector<std::complex<double>> envelope(mesh);lattice_detail::Hash identity;
    identity.text("manta.canonical-weighted-density-envelope.v1");identity.word(nk);identity.word(n);identity.number(density.density_l1_limit);
    for(Count q=0;q<mesh;++q) {
        const double h=density.weighted_entry_bounds[q];if(!std::isfinite(h)||h<0)throw std::invalid_argument("MANTA density envelope must be finite nonnegative");
        envelope[q]=h;identity.number(h);
    }
    PeriodicNuclearAccuracyResult out;out.targets=o;out.density_l1_limit=density.density_l1_limit;
    out.density_envelope_identity=identity.value.finish_hex();
    auto leaf=b;leaf.other_live_bytes=p.modeled_peak_bytes-p.nuclear.modeled_peak_bytes;
    out.nuclear=bound_periodic_nuclear(basis,s,g,cells,k,alpha,cutoff,paircut,o,leaf);
    const auto& r=out.nuclear.receipt;
    if(enforce_targets && r.maximum_potential_error>o.matrix_error)
        throw std::invalid_argument("MANTA nuclear matrix accuracy target failed; enlarge physical support or completion cube");
    if(enforce_targets && r.nuclear_energy_error_bound>o.nuclear_energy_error)
        throw std::invalid_argument("MANTA nuclear repulsion energy accuracy target failed; enlarge physical support or completion cube");
    // One-body energy is Tr(D V), with no half factor. H already includes
    // k weights and the spin sum; this is not an evaluated SCF density.
    out.entry_attraction_energy_error_bound=pair_image_detail::density_trace_bound(out.nuclear.potential_error_bound,envelope,std::vector<double>(nk,1.));
    out.l1_attraction_energy_error_bound=pair_image_detail::density_trace_bound(md::Matrix::Constant(1,1,r.maximum_potential_error),{{density.density_l1_limit,0.}},{1.});
    out.attraction_energy_error_bound=std::min(out.entry_attraction_energy_error_bound,out.l1_attraction_energy_error_bound);
    if(enforce_targets && out.attraction_energy_error_bound>o.attraction_energy_error)
        throw std::invalid_argument("MANTA nuclear attraction energy accuracy target failed; enlarge physical support or completion cube");
    out.total_energy_error_bound=sum(out.attraction_energy_error_bound,r.nuclear_energy_error_bound);
    out.shell_calls=r.shell_calls;out.primitive_pairs=r.primitive_pairs;
    lattice_detail::Hash certificate;certificate.text("manta.nuclear-density-envelope-accuracy.v1;no-roundoff;no-relaxation");
    certificate.text(r.source_identity);certificate.text(out.density_envelope_identity);
    for(double x:{o.matrix_error,o.attraction_energy_error,o.nuclear_energy_error,out.entry_attraction_energy_error_bound,
        out.l1_attraction_energy_error_bound,out.attraction_energy_error_bound,out.total_energy_error_bound})certificate.number(x);
    out.certificate_identity=certificate.value.finish_hex();out.qualified=r.maximum_potential_error<=o.matrix_error && r.nuclear_energy_error_bound<=o.nuclear_energy_error && out.attraction_energy_error_bound<=o.attraction_energy_error;return out;
}

} // namespace vibeqc::manta
