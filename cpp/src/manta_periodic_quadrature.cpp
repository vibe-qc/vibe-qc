#include "vibeqc/manta_periodic_quadrature.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/lebedev_data.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace vibeqc::manta {
namespace {
using namespace admission;
GridOptions options(const PeriodicQuadratureOptions& o) {
    GridOptions g;g.n_radial=o.radial_points;g.angular=AngularScheme::Lebedev;
    g.lebedev_order=o.lebedev_order;g.partition=o.stratmann?AtomicPartition::Stratmann:AtomicPartition::Becke;
    return g;
}
Eigen::Vector3d fold(const Eigen::Matrix3d& a,const Eigen::Matrix3d& inverse,const Eigen::Vector3d& r) {
    Eigen::Vector3d f=inverse*r;
    if(!f.allFinite() || f.cwiseAbs().maxCoeff()>1e6)
        throw std::invalid_argument("MANTA quadrature fractional coordinate exceeds supported range");
    f.array()-=f.array().floor();return a*f;
}
Count box(const Eigen::Matrix3d& inverse,double radius,Count atoms) {
    Count result=atoms;
    // The shared visitor pads fractional boxes by 1e-10*(1+|f|+reach).
    // Its accepted image indices are below INT_MAX/4, so a whole extra
    // integer on each face encloses that padding and floor/ceil rounding.
    for(int i=0;i<3;++i) {
        const double reach=std::ceil(radius*inverse.row(i).norm()+2.);
        if(!std::isfinite(reach) || reach>1e6)
            throw std::invalid_argument("MANTA quadrature image extent exceeds supported range");
        result=mul(result,add(mul(2,static_cast<Count>(reach)),1));
    }
    return result;
}
struct Digest {
    detail::Sha256 hash;
    void word(Count v) {std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8);}
    void real(double v) {if(v==0)v=0.;Count b;std::memcpy(&b,&v,8);word(b);}
};
}
PeriodicQuadraturePlan plan_periodic_quadrature(const PeriodicSystem& s,const PeriodicQuadratureOptions& o,const ResourceBudget& b) {
    margin(b);PeriodicQuadraturePlan p;p.atoms=s.unit_cell.size();
    const Count metadata=add(32768,mul(256,square(p.atoms)));
    cap(metadata,b.scalar_terms,"MANTA quadrature metadata scalar budget exceeded");
    cap(mul(32,metadata),b.kernel_terms,"MANTA quadrature metadata kernel budget exceeded");
    cap(add(b.other_live_bytes,32768),b.memory_bytes,"MANTA quadrature metadata memory budget exceeded");
    if(!p.atoms || p.atoms>static_cast<Count>(std::numeric_limits<int>::max()/118) || s.dim!=3 || !s.lattice.allFinite())
        throw std::invalid_argument("MANTA quadrature requires a nonempty finite 3D cell");
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(s.lattice);
    const auto singular=svd.singularValues();
    if(singular.minCoeff()<1e-4 || singular.maxCoeff()>1e4 || singular.maxCoeff()/singular.minCoeff()>1e4)
        throw std::invalid_argument("MANTA quadrature lattice scale or conditioning exceeds supported range");
    if(o.radial_points<1 || o.radial_points>10000 || !std::isfinite(o.partition_radius_bohr)
       || o.partition_radius_bohr<=0 || o.partition_radius_bohr>1e4)
        throw std::invalid_argument("MANTA quadrature requires 1..10000 radial points and a positive bounded partition radius");
    const LebedevTier* tier=nullptr;
    for(std::size_t i=0;i<lebedev_tier_count();++i)if(lebedev_tiers()[i].order==o.lebedev_order)tier=&lebedev_tiers()[i];
    if(!tier)throw std::invalid_argument("MANTA quadrature requires a bundled Lebedev order");
    for(std::size_t i=0;i<tier->n_points;++i)if(!(tier->data[4*i+3]>0))
        throw std::invalid_argument("MANTA quadrature requires a positive-weight Lebedev rule");
    p.angular_points=tier->n_points;
    const Count per_atom=mul(o.radial_points,p.angular_points);
    cap(per_atom,std::numeric_limits<int>::max(),"MANTA quadrature per-atom extent exceeds int range");
    p.grid_points=mul(p.atoms,per_atom);
    cap(p.grid_points,std::numeric_limits<Eigen::Index>::max()/3,"MANTA quadrature output extent exceeds index range");
    p.threads=1;
#ifdef _OPENMP
    p.threads=std::max(1,omp_get_max_threads());
#endif
    const Eigen::Matrix3d inverse=s.lattice.inverse();
    for(const auto& atom:s.unit_cell) {
        if(atom.Z<1 || atom.Z>118)throw std::invalid_argument("MANTA quadrature requires atomic numbers 1..118");
        fold(s.lattice,inverse,Eigen::Vector3d(atom.xyz[0],atom.xyz[1],atom.xyz[2]));
    }
    const double cover=s.lattice.col(0).norm()+s.lattice.col(1).norm()+s.lattice.col(2).norm();
    // Rounded fractional labels put an image within cover/2 of every point.
    // Both the nearest-image search and max(R,2*d_nearest) fit this radius.
    // A 1% enclosure covers floating arithmetic for the admitted lattice scale.
    const double radius=std::max(o.partition_radius_bohr,cover)*1.01;
    p.atlas_candidates=box(inverse,2.02*o.partition_radius_bohr,p.atoms);
    p.neighbor_candidates=box(inverse,radius,p.atoms);
    p.candidate_visits=add(mul(p.atoms,p.atlas_candidates),mul(p.grid_points,add(p.atlas_candidates,mul(2,p.neighbor_candidates))));
    p.scalar_terms=add(metadata,add(mul(128,p.candidate_visits),mul(128,mul(p.grid_points,square(p.neighbor_candidates)))));
    p.kernel_terms=mul(32,p.scalar_terms);
    // Full shared Grid, owned points/weight exports, atom metadata, angular
    // tables, radial scratch, vector capacity doubling, and concurrent local
    // distance/product/neighbor lists. No global OpenMP settings are changed.
    p.output_bytes=mul(76,p.grid_points);
    p.workspace_bytes=add(32768,add(mul(256,add(p.atoms,p.atlas_candidates)),
        add(mul(64,p.angular_points),mul(p.threads,add(mul(64,o.radial_points),mul(128,p.neighbor_candidates))))));
    p.modeled_peak_bytes=add(b.other_live_bytes,add(b.backend_margin_bytes,add(p.output_bytes,p.workspace_bytes)));
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA quadrature memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA quadrature scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA quadrature kernel budget exceeded");
    return p;
}
PeriodicQuadratureResult build_periodic_quadrature(const PeriodicSystem& s,const PeriodicQuadratureOptions& o,const ResourceBudget& b) {
    PeriodicQuadratureResult out;out.plan=plan_periodic_quadrature(s,o,b);
    const Eigen::Matrix3d inverse=s.lattice.inverse();PeriodicSystem canonical;
    canonical.dim=3;canonical.lattice=s.lattice;canonical.unit_cell=s.unit_cell;
    Digest digest;digest.word(1);digest.word(o.radial_points);digest.word(o.lebedev_order);
    digest.word(o.stratmann);digest.real(o.partition_radius_bohr);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)digest.real(s.lattice(i,j));
    for(auto& atom:canonical.unit_cell) {
        const auto r=fold(s.lattice,inverse,Eigen::Vector3d(atom.xyz[0],atom.xyz[1],atom.xyz[2]));
        digest.word(atom.Z);for(int i=0;i<3;++i) {atom.xyz[i]=r[i];digest.real(r[i]);}
    }
    out.grid=build_periodic_point_grid(canonical,o.partition_radius_bohr,options(o));
    if(static_cast<Count>(out.grid.points.rows())!=out.plan.grid_points)
        throw std::logic_error("MANTA quadrature producer changed admitted count");
    double sum=0.,correction=0.;
    for(Eigen::Index i=0;i<out.grid.points.rows();++i) {
        const Eigen::Vector3d point=fold(s.lattice,inverse,out.grid.points.row(i).transpose());
        out.grid.points.row(i)=point.transpose();const double w=out.grid.weights[i];
        if(!std::isfinite(w) || w<0)throw std::runtime_error("MANTA quadrature produced an invalid weight");
        out.receipt.nonzero_weights+=w>0;
        for(int j=0;j<3;++j)digest.real(point[j]);digest.real(w);
        const double y=w-correction,t=sum+y;correction=(t-sum)-y;sum=t;
    }
    out.receipt.weight_sum=sum;out.receipt.cell_volume=std::abs(s.lattice.determinant());
    out.receipt.source_identity="manta.periodic-quadrature.v1:"+digest.hash.finish_hex();return out;
}
} // namespace vibeqc::manta
