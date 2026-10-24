#include "vibeqc/manta_periodic_fitting.hpp"
#include "vibeqc/manta_admission.hpp"
#include <algorithm>
#include <array>

namespace vibeqc::manta {
namespace {
using namespace admission;
using Label=std::array<std::int64_t,3>;
ResourceBudget unlimited(ResourceBudget b) {
    b.other_live_bytes=0;
    b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max();return b;
}
}
Count periodic_gdf_domain_vector_capacity(Count original,Count cube) {
    if(cube>128)throw std::invalid_argument("MANTA fitting domain reciprocal cube exceeds 128");
    const Count side=add(mul(2,cube),1);return add(original,mul(side,square(side))-1);
}
PeriodicGDFDomainPlan PeriodicGDFTiles::plan_domains(double pair,double auxiliary,Count cube,const ResourceBudget& b) const {
    return plan_auxiliary_domains(auxiliary_,pair,auxiliary,cube,b);
}
PeriodicGDFDomainPlan PeriodicGDFTiles::plan_auxiliary_domains(const BasisSet& candidate,double pair,double auxiliary,Count cube,const ResourceBudget& b) const {
    if(system_.dim!=3 || !source_.zero_transfer || physical_.kernel!="coulomb" ||
       !std::isfinite(pair) || pair<options_.pair_cutoff_bohr ||
       !std::isfinite(auxiliary) || auxiliary<options_.auxiliary_cutoff_bohr)
        throw std::invalid_argument("MANTA fitting domains require nested finite radii and a 3D Gamma Coulomb source");
    auto options=options_;options.pair_cutoff_bohr=pair;options.auxiliary_cutoff_bohr=auxiliary;
    PeriodicGDFDomainPlan p;p.maximum_vectors=periodic_gdf_domain_vector_capacity(vectors_.rows(),cube);
    p.source=plan_periodic_gdf_tiles(orbital_,candidate,system_,p.maximum_vectors,k_.rows(),options,unlimited(b),physical_);
    // Original labels, the enlarged vector matrix, constructor copies and
    // bounded sorting/lookup scratch. No integral values precede admission.
    p.generation_bytes=add(65536,mul(256,p.maximum_vectors));
    p.generation_scalar_terms=add(65536,mul(8192,add(p.maximum_vectors,k_.rows())));
    p.generation_kernel_terms=mul(16,p.generation_scalar_terms);
    p.modeled_peak_bytes=add(b.other_live_bytes,add(plan_.modeled_peak_bytes,add(p.generation_bytes,p.source.modeled_peak_bytes)));
    p.scalar_terms=add(p.generation_scalar_terms,p.source.scalar_terms);
    p.kernel_terms=add(p.generation_kernel_terms,p.source.kernel_terms);
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA fitting domain memory budget exceeded");
    cap(p.source.shell_calls,b.shell_calls,"MANTA fitting domain shell budget exceeded");
    cap(p.source.primitive_products,b.primitive_quartets,"MANTA fitting domain primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA fitting domain scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA fitting domain kernel budget exceeded");return p;
}
std::shared_ptr<const PeriodicGDFTiles> PeriodicGDFTiles::with_domains(double pair,double auxiliary,Count cube,const ResourceBudget& b) const {
    return with_auxiliary_domains(auxiliary_,pair,auxiliary,cube,b);
}
std::shared_ptr<const PeriodicGDFTiles> PeriodicGDFTiles::with_auxiliary_domains(const BasisSet& candidate,double pair,double auxiliary,Count cube,const ResourceBudget& b) const {
    const auto p=plan_auxiliary_domains(candidate,pair,auxiliary,cube,b);
    std::vector<Label> labels;labels.reserve(vectors_.rows());
    for(Eigen::Index i=0;i<vectors_.rows();++i) {
        const Eigen::Vector3d f=system_.lattice.transpose()*vectors_.row(i).transpose()/(2*std::acos(-1.));
        Label label;
        for(int d=0;d<3;++d) {
            if(!std::isfinite(f[d]) || std::abs(f[d])>0x1p40 || std::abs(f[d]-std::round(f[d]))>1e-10)
                throw std::invalid_argument("MANTA fitting domain source requires reciprocal lattice vectors");
            label[d]=static_cast<std::int64_t>(std::round(f[d]));
        }
        if(label==Label{0,0,0})throw std::invalid_argument("MANTA fitting domain source excludes reciprocal zero");
        labels.push_back(label);
    }
    std::sort(labels.begin(),labels.end());
    if(std::adjacent_find(labels.begin(),labels.end())!=labels.end())
        throw std::invalid_argument("MANTA fitting domain source has duplicate reciprocal vectors");
    for(const auto& n:labels)if(!std::binary_search(labels.begin(),labels.end(),Label{-n[0],-n[1],-n[2]}))
        throw std::invalid_argument("MANTA fitting domain source must close under reversal");
    md::Matrix vectors(p.maximum_vectors,3);vectors.topRows(vectors_.rows())=vectors_;
    const Eigen::Matrix3d reciprocal=2*std::acos(-1.)*system_.lattice.inverse().transpose();
    Eigen::Index count=vectors_.rows();const auto r=static_cast<std::int64_t>(cube);
    for(auto x=-r;x<=r;++x)for(auto y=-r;y<=r;++y)for(auto z=-r;z<=r;++z) {
        const Label n{x,y,z};if(n==Label{0,0,0} || std::binary_search(labels.begin(),labels.end(),n))continue;
        vectors.row(count++)=(reciprocal*Eigen::Vector3d(x,y,z)).transpose();
    }
    vectors.conservativeResize(count,3);
    auto options=options_;options.pair_cutoff_bohr=pair;options.auxiliary_cutoff_bohr=auxiliary;
    return std::make_shared<PeriodicGDFTiles>(orbital_,candidate,system_,q_,k_,vectors,options,unlimited(b),physical_);
}
} // namespace vibeqc::manta
