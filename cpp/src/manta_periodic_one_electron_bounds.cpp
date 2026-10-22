#include "vibeqc/manta_periodic_one_electron_bounds.hpp"
#include "manta_pair_image_internal.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_lattice_geometry.hpp"
#include <algorithm>

namespace vibeqc::manta {
struct PeriodicOneElectronImageDriver {
    static PeriodicOneElectronImagePlan plan(const PeriodicPairSource& s,const ResourceBudget& budget) {
        return plan(s,radius(s),budget);
    }
    static Count radius(const PeriodicPairSource& s) {
        Count radius=0;for(const auto& n:s.images_)for(auto x:n)radius=std::max(radius,Count(std::abs(x)));
        return radius;
    }
    static PeriodicOneElectronImagePlan plan(const BasisSet& basis,Count entries,Count nk,Count radius,
        Count retained,const ResourceBudget& budget) {
        using namespace admission;
        const auto base=pair_image_detail::plan(basis,entries,nk,radius,budget);
        PeriodicOneElectronImagePlan p;p.n_basis=basis.nbasis();p.momenta=nk;
        p.cube_radius=base.radius;p.cube_cells=base.cells;
        p.shell_calls=base.shell_calls;p.primitive_pairs=base.primitive_products;
        p.scalar_terms=add(4096,mul(2,base.scalar_terms));p.kernel_terms=mul(128,p.scalar_terms);
        // The source stays alive alongside primitive gradient majorants,
        // both error matrices, owned exports and checked lattice geometry.
        p.modeled_peak_bytes=add(base.modeled_peak_bytes,add(retained,
            add(16384,mul(128,square(p.n_basis)))));
        cap(p.scalar_terms,budget.scalar_terms,"MANTA one-electron image scalar budget exceeded");
        cap(p.kernel_terms,budget.kernel_terms,"MANTA one-electron image kernel budget exceeded");
        cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA one-electron image memory budget exceeded");
        return p;
    }
    static PeriodicOneElectronImagePlan plan(const PeriodicPairSource& s,Count radius,const ResourceBudget& budget) {
        return plan(s.basis_,s.entries_.size(),s.plan_.momenta,radius,s.plan_.retained_bytes,budget);
    }
    static PeriodicOneElectronImageResult run(const PeriodicPairSource& s,const ResourceBudget& budget) {
        return run(s,radius(s),budget);
    }
    static PeriodicOneElectronImageResult run(const PeriodicPairSource& s,Count radius,const ResourceBudget& budget) {
        const auto p=plan(s,radius,budget);
        const auto base=pair_image_detail::plan(s.basis_,s.entries_.size(),s.plan_.momenta,radius,budget);
        auto out=pair_image_detail::one_electron(s.basis_,s.lattice_,s.entries_,s.receipt_.source_identity,base);
        out.plan=p;return out;
    }
};
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const PeriodicPairSource& s,const ResourceBudget& b) {
    return PeriodicOneElectronImageDriver::plan(s,b);
}
PeriodicOneElectronImageResult bound_periodic_one_electron_images(const PeriodicPairSource& s,const ResourceBudget& b) {
    return PeriodicOneElectronImageDriver::run(s,b);
}
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const PeriodicPairSource& s,Count radius,const ResourceBudget& b) {
    return PeriodicOneElectronImageDriver::plan(s,radius,b);
}
PeriodicOneElectronImageResult bound_periodic_one_electron_images(const PeriodicPairSource& s,Count radius,const ResourceBudget& b) {
    return PeriodicOneElectronImageDriver::run(s,radius,b);
}
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const BasisSet& basis,Count entries,Count nk,Count radius,
    Count retained,const ResourceBudget& budget) {
    return PeriodicOneElectronImageDriver::plan(basis,entries,nk,radius,retained,budget);
}
PeriodicOneElectronAccuracyPlan plan_periodic_one_electron_accuracy(const BasisSet& basis,Count nc,Count nk,
    std::optional<double> cutoff,const PeriodicOneElectronAccuracyOptions& o,const ResourceBudget& budget) {
    using namespace admission; margin(budget);
    for(double value:{o.overlap_error,o.kinetic_error,o.kinetic_energy_error})
        if(!std::isfinite(value) || value<0.) throw std::invalid_argument("MANTA one-electron accuracy targets must be finite nonnegative");
    PeriodicOneElectronAccuracyPlan p;if(!o.enabled())return p;
    if(o.overlap_error<=0. || o.kinetic_error<=0. || o.kinetic_energy_error<=0.)
        throw std::invalid_argument("MANTA one-electron accuracy requires three positive targets");
    auto leaf=budget;leaf.other_live_bytes=0;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    PeriodicPairOptions pairs;pairs.order=0;pairs.pair_cutoff_bohr=cutoff;
    p.pairs=plan_periodic_pairs(basis,nc,nk,pairs,leaf);
    p.images=PeriodicOneElectronImageDriver::plan(basis,p.pairs.maximum_pairs,nk,o.completion_cube_radius,p.pairs.retained_bytes,leaf);
    // Count-only admission precedes image-label allocation, pair snapshots,
    // Gaussian preparation and bounds. Parent owns the canonical envelope.
    const Count mesh=mul(nk,square(basis.nbasis()));
    const Count extra=add(32768,add(mul(128,add(nc,nk)),mul(32,mesh)));
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(extra,add(p.pairs.modeled_peak_bytes,p.images.modeled_peak_bytes)));
    p.shell_calls=add(p.pairs.shell_calls,p.images.shell_calls);
    p.primitive_pairs=add(p.pairs.primitive_pairs,p.images.primitive_pairs);
    Count levels=1;for(auto remaining=nc;remaining>1;remaining>>=1)++levels;
    p.scalar_terms=add(p.pairs.scalar_terms,add(p.images.scalar_terms,
        add(mul(128,mesh),mul(1024,mul(levels,p.pairs.maximum_pairs)))));
    p.kernel_terms=add(p.pairs.kernel_terms,add(p.images.kernel_terms,mul(128,p.scalar_terms)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA one-electron accuracy memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA one-electron accuracy shell budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA one-electron accuracy primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA one-electron accuracy scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA one-electron accuracy kernel budget exceeded");
    return p;
}
PeriodicPairSource make_periodic_accuracy_pair_source(const BasisSet& basis,const Eigen::Matrix3d& lattice,
    const md::Matrix& cells,const md::Matrix& k,std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    PeriodicPairOptions admission_options;admission_options.order=0;admission_options.pair_cutoff_bohr=cutoff;
    plan_periodic_pairs(basis,cells.rows(),k.rows(),admission_options,budget);
    if(cells.cols()!=3 || k.cols()!=3 || !cells.allFinite() || !k.allFinite() || !lattice.allFinite())
        throw std::invalid_argument("MANTA periodic accuracy requires finite 3D vectors");
    const auto inverse=lattice.inverse().eval();
    if(!inverse.allFinite())throw std::invalid_argument("MANTA periodic accuracy requires an invertible lattice");
    const Count nk=k.rows();
    std::vector<LatticeIndex> labels(cells.rows());
    for(Eigen::Index c=0;c<cells.rows();++c) {
        const Eigen::Vector3d fractional=inverse*cells.row(c).transpose();
        for(int axis=0;axis<3;++axis) {
            const double v=fractional[axis],nearest=std::round(v);
            if(!std::isfinite(v) || std::abs(nearest)>0x1p40 || std::abs(v-nearest)>1e-10)
                throw std::invalid_argument("MANTA periodic accuracy requires integer lattice translations");
            labels[c][axis]=static_cast<std::int64_t>(nearest);
        }
        for(int axis=0;axis<3;++axis) {
            long double value=0.,scale=1.+std::abs(cells(c,axis));
            for(int d=0;d<3;++d) {const long double term=static_cast<long double>(lattice(axis,d))*labels[c][d];value+=term;scale+=std::abs(term);}
            if(std::abs(value-cells(c,axis))>256*std::numeric_limits<double>::epsilon()*scale)
                throw std::invalid_argument("MANTA periodic accuracy translations differ from their lattice images beyond roundoff");
        }
    }
    PeriodicPairOptions pairs;pairs.order=0;pairs.pair_cutoff_bohr=cutoff;
    PeriodicPairSource source(basis,lattice,labels,k,std::vector<double>(nk,1./nk),pairs,budget);
    // The bound inventory must include no block absent from the actual Bloch
    // producer. Refuse a cutoff decision changed by lattice reconstruction;
    // roundoff in integral values remains outside this image certificate.
    if(cutoff) {
        std::vector<std::pair<LatticeIndex,Count>> ordered;ordered.reserve(labels.size());
        for(Count c=0;c<labels.size();++c)ordered.push_back({labels[c],c});std::sort(ordered.begin(),ordered.end());
        auto audit=[&](Count ia,Count ib,const LatticeIndex& image) {
            const auto it=std::lower_bound(ordered.begin(),ordered.end(),std::make_pair(image,Count(0)));
            if(it==ordered.end() || it->first!=image)throw std::logic_error("MANTA image accuracy lost a retained image");
            const Count c=it->second;const auto& a=basis.libint()[ia];const auto& b=basis.libint()[ib];
            const double distance=std::hypot(a.O[0]-(b.O[0]+cells(c,0)),a.O[1]-(b.O[1]+cells(c,1)),a.O[2]-(b.O[2]+cells(c,2)));
            if(distance>*cutoff)throw std::invalid_argument("MANTA periodic accuracy cutoff membership is unresolved by image reconstruction");
        };
        for(const auto& entry:source.entries()) {
            audit(entry.shells[0],entry.shells[1],entry.image);
            if(entry.paired)audit(entry.shells[1],entry.shells[0],{-entry.image[0],-entry.image[1],-entry.image[2]});
        }
    }
    return source;
}
PeriodicOneElectronAccuracyResult bound_periodic_one_electron_accuracy(const BasisSet& basis,const Eigen::Matrix3d& lattice,
    const md::Matrix& cells,const md::Matrix& k,std::optional<double> cutoff,
    const BoundedPeriodicDensityEnvelope& density,const PeriodicOneElectronAccuracyOptions& o,const ResourceBudget& budget,bool enforce_targets) {
    using namespace admission;
    const auto p=plan_periodic_one_electron_accuracy(basis,cells.rows(),k.rows(),cutoff,o,budget);
    if(!o.enabled())throw std::invalid_argument("MANTA one-electron accuracy requires enabled targets");
    const Count n=basis.nbasis(),nk=k.rows(),mesh=mul(nk,square(n));
    if(density.weighted_entry_bounds.size()!=mesh || !std::isfinite(density.density_l1_limit) || !(density.density_l1_limit>0.))
        throw std::invalid_argument("MANTA one-electron accuracy requires the canonical density envelope");
    auto leaf=budget;leaf.other_live_bytes=p.modeled_peak_bytes-p.pairs.modeled_peak_bytes;
    const auto source=make_periodic_accuracy_pair_source(basis,lattice,cells,k,cutoff,leaf);
    leaf.other_live_bytes=p.modeled_peak_bytes-p.images.modeled_peak_bytes;
    PeriodicOneElectronAccuracyResult result;result.targets=o;
    result.images=bound_periodic_one_electron_images(source,o.completion_cube_radius,leaf);
    const auto& images=result.images.receipt;
    if(enforce_targets && (images.maximum_overlap_error>o.overlap_error || images.maximum_kinetic_error>o.kinetic_error))
        throw std::invalid_argument("MANTA one-electron image matrix accuracy target failed; enlarge physical support or completion cube");
    std::vector<std::complex<double>> envelope(mesh);lattice_detail::Hash identity;
    identity.text("manta.canonical-weighted-density-envelope.v1");identity.word(nk);identity.word(n);identity.number(density.density_l1_limit);
    for(Count q=0;q<mesh;++q) {
        const double h=density.weighted_entry_bounds[q];if(!std::isfinite(h) || h<0.)throw std::invalid_argument("MANTA density envelope must be finite nonnegative");
        envelope[q]=h;identity.number(h);
    }
    result.density_l1_limit=density.density_l1_limit;result.density_envelope_identity=identity.value.finish_hex();
    // The existing log-positive trace producer preserves subnormal bounds.
    // H already includes k weights, hence unit weights in this contraction.
    result.kinetic_energy_error_bound=std::min(
        pair_image_detail::density_trace_bound(result.images.kinetic_error_bound,envelope,std::vector<double>(nk,1.)),
        pair_image_detail::density_trace_bound(md::Matrix::Constant(1,1,images.maximum_kinetic_error),
            {{density.density_l1_limit,0.}},{1.}));
    if(enforce_targets && result.kinetic_energy_error_bound>o.kinetic_energy_error)
        throw std::invalid_argument("MANTA one-electron image energy accuracy target failed; enlarge physical support or completion cube");
    result.shell_calls=add(source.receipt().pairs,images.shell_pairs_evaluated);
    result.primitive_pairs=add(source.receipt().primitive_pairs,images.primitive_pairs_evaluated);
    lattice_detail::Hash certificate;certificate.text("manta.one-electron-density-envelope-accuracy.v1");
    certificate.text(images.source_identity);certificate.text(result.density_envelope_identity);
    for(double value:{o.overlap_error,o.kinetic_error,o.kinetic_energy_error,result.kinetic_energy_error_bound})certificate.number(value);
    certificate.word(o.completion_cube_radius);result.certificate_identity=certificate.value.finish_hex();result.qualified=images.maximum_overlap_error<=o.overlap_error && images.maximum_kinetic_error<=o.kinetic_error && result.kinetic_energy_error_bound<=o.kinetic_energy_error;
    return result;
}
} // namespace vibeqc::manta
