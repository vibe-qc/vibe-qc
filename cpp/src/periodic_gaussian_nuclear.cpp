#include "vibeqc/periodic_gaussian_nuclear.hpp"

#include <algorithm>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "aopair_ft_internal.hpp"
#include "periodic_gaussian_nuclear_internal.hpp"
#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/cosx_kernel.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/ewald.hpp"

namespace vibeqc {
namespace {
namespace pair = aopair_ft_detail;
using Complex = std::complex<double>;
using Options = PeriodicGaussianNuclearOptions;
using Live = PeriodicGaussianNuclearLiveInventory;
using Caps = PeriodicGaussianNuclearCaps;
using Plan = PeriodicGaussianNuclearPlan;
using EPlan = PeriodicGaussianNuclearEwaldPlan;
using Diagnostics = PeriodicGaussianNuclearDiagnostics;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr std::uint64_t sha_max = std::numeric_limits<std::uint64_t>::max()/8U;
constexpr std::uint64_t boys_rows = 10001U, boys_iterations = 512U;
static_assert(sizeof(Complex)==16 && sizeof(int)==4 && sizeof(Atom)>=28,
              "nuclear inventory requires binary64 and 32-bit atomic numbers");
void float_environment() {
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559
        &&std::numeric_limits<double>::radix==2 &&std::numeric_limits<double>::digits==53,
        "Gaussian nuclear requires IEEE754 binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    throw std::invalid_argument("Gaussian nuclear does not permit fast-math arithmetic");
#endif
#if FLT_EVAL_METHOD != 0
    throw std::invalid_argument("Gaussian nuclear requires FLT_EVAL_METHOD=0");
#endif
    if (std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("Gaussian nuclear requires round-to-nearest arithmetic");
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1.0,zero=0.0;
    volatile double smallest_normal=std::numeric_limits<double>::min(),half=.5;
    if (!(tiny>0)||tiny*one!=tiny||tiny+zero!=tiny||std::fma(tiny,one,zero)!=tiny
        ||std::nextafter(0.0,1.0)!=tiny||smallest_normal*half==0)
        throw std::invalid_argument("Gaussian nuclear requires gradual underflow");
}
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if (b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("Gaussian nuclear count overflow");
    return a+b;
}
std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if (a && b>std::numeric_limits<std::uint64_t>::max()/a) throw std::overflow_error("Gaussian nuclear count overflow");
    return a*b;
}
void limit(std::uint64_t n,std::uint64_t cap,const char* message) {
    if (n>cap) throw std::length_error(message);
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::runtime_error("Gaussian nuclear numerical value is not finite");
    return x;
}
Complex finite(Complex z) { finite(z.real()); finite(z.imag()); return z; }
class Digest {
public:
    void bytes(const std::uint8_t* p,std::size_t n) {
        limit(n,sha_max-extent_,"Gaussian nuclear SHA extent overflow");
        hash_.update(p,n); extent_+=n;
    }
    void u32(std::uint32_t n) {
        std::array<std::uint8_t,4> b{};
        for (unsigned i=0;i<4;++i) b[i]=n>>(24U-8U*i);
        bytes(b.data(),b.size());
    }
    void u64(std::uint64_t n) {
        std::array<std::uint8_t,8> b{};
        for (unsigned i=0;i<8;++i) b[i]=n>>(56U-8U*i);
        bytes(b.data(),b.size());
    }
    void real(double x) {
        finite(x); if (x==0) x=0;
        std::uint64_t bits; std::memcpy(&bits,&x,8U); u64(bits);
    }
    void text(const std::string& s) {
        u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
    std::array<char,64> finish() {
        const auto s=hash_.finish_hex(); std::array<char,64> out{};
        if (s.size()!=out.size()) throw std::logic_error("Gaussian nuclear SHA extent changed");
        std::copy(s.begin(),s.end(),out.begin()); return out;
    }
private:
    detail::Sha256 hash_; std::uint64_t extent_=0;
};
struct Sum {
    double value=0,correction=0;
    void add(double x) {
        finite(x); const double next=finite(value+x);
        correction=finite(correction+(std::abs(value)>=std::abs(x)?(value-next)+x:(x-next)+value)); value=next;
    }
    double result() const { return finite(value+correction); }
};
struct CSum {
    Sum re,im;
    void add(Complex z) { re.add(z.real()); im.add(z.imag()); }
    Complex result() const { return {re.result(),im.result()}; }
};
void validate_options(const Options& o,const Live& live,const Caps& caps) {
    float_environment();
    for (double x:{o.alpha,o.real_cutoff_bohr,o.reciprocal_cutoff_bohr_inverse})
        if (!std::isfinite(x)||x<=0) throw std::invalid_argument("Gaussian nuclear alpha and cutoffs must be finite and positive");
    for (double x:{o.structural_absolute_tolerance,o.structural_relative_tolerance})
        if (!std::isfinite(x)||x<0||x>=1) throw std::invalid_argument("Gaussian nuclear structural tolerances must be finite in [0,1)");
    if (o.structural_absolute_tolerance==0 && o.structural_relative_tolerance==0)
        throw std::invalid_argument("Gaussian nuclear structural tolerances cannot both be zero");
    for (auto n:{live.replicas_per_node,live.fixed_backend_margin_bytes_per_replica,
        caps.maximum_owned_numeric_bytes,caps.maximum_per_replica_inventoried_bytes,
        caps.maximum_node_inventoried_bytes,caps.maximum_atom_count,caps.maximum_pair_count,
        caps.maximum_ao_image_candidates,caps.maximum_nuclear_image_candidates,
        caps.maximum_reciprocal_candidates,caps.maximum_ewald_pair_candidates,caps.maximum_work_units})
        if (!n) throw std::invalid_argument("Gaussian nuclear caps, replicas and backend margin must be positive");
    // Both the legacy Ewald and OS rho formulas form these products.
    for (double x:{o.alpha*o.alpha,4.0*o.alpha*o.alpha,o.real_cutoff_bohr*o.real_cutoff_bohr,
        o.reciprocal_cutoff_bohr_inverse*o.reciprocal_cutoff_bohr_inverse})
        if (!std::isfinite(x)||x<=0) throw std::invalid_argument("Gaussian nuclear squared controls underflow or overflow");
}
struct Geometry {
    pair::LongMatrix inverse_long;
    Eigen::Matrix3d inverse,reciprocal;
    double volume;
};
Geometry geometry(const PeriodicGaussianSourceContext& c,const PeriodicSystem& s) {
    if (s.dim!=3||!s.lattice.allFinite()||s.lattice!=c.direct_lattice())
        throw std::invalid_argument("Gaussian nuclear original cell does not match source context");
    const auto a=s.lattice.cast<long double>().eval();
    Geometry g; g.inverse_long=a.inverse(); g.inverse=s.lattice.inverse();
    g.reciprocal=2.0*pi*g.inverse.transpose(); g.volume=std::abs(s.lattice.determinant());
    const auto condition=a.norm()*g.inverse_long.norm();
    if (!g.inverse_long.allFinite()||!g.inverse.allFinite()||!g.reciprocal.allFinite()
        ||!std::isfinite(condition)||condition>1e10L||!std::isfinite(g.volume)||g.volume<1e-14
        ||g.reciprocal!=c.reciprocal_lattice())
        throw std::invalid_argument("Gaussian nuclear lattice exceeds conditioning or reciprocal coherence policy");
    return g;
}
void derived_controls(const Geometry& g,const Options& o) {
    for (double x:{4.0*pi/g.volume,1.0/(4.0*o.alpha*o.alpha),
        g.volume*o.alpha*o.alpha,pi/(g.volume*o.alpha*o.alpha)})
        if (!std::isfinite(x)||x<=0)
            throw std::invalid_argument("Gaussian nuclear derived prefactor underflow or overflow");
}
void require_context(const std::shared_ptr<const PeriodicGaussianSourceContext>& c) {
    if (!c||c->contract_version()!=kPeriodicGaussianSourceContextVersion)
        throw std::invalid_argument("Gaussian nuclear requires a live native source context");
}
std::uint64_t system_count(const PeriodicSystem& s,const Caps& caps) {
    const auto n=static_cast<std::uint64_t>(s.unit_cell.size());
    if (!n) throw std::invalid_argument("Gaussian nuclear requires a nonempty actual nucleus list");
    limit(n,caps.maximum_atom_count,"Gaussian nuclear atom count exceeds cap");
    return n;
}
void validate_atoms(const PeriodicSystem& s,const Geometry& g) {
    if (s.multiplicity<1) throw std::invalid_argument("Gaussian nuclear original multiplicity must be positive");
    for (const auto& atom:s.unit_cell) {
        if (atom.Z<0||atom.Z>118) throw std::invalid_argument("Gaussian nuclear requires actual all-electron atomic numbers 0..118");
        Eigen::Vector3d x;
        for (int d=0;d<3;++d) x[d]=finite(atom.xyz[d]);
        const auto f=(g.inverse*x).eval();
        if (!f.allFinite()||f.cwiseAbs().maxCoeff()>0x1p40)
            throw std::invalid_argument("Gaussian nuclear unwrapped atom exceeds fractional-coordinate policy");
    }
}
Eigen::Vector3d centered(const PeriodicSystem& s,const Geometry& g,const Atom& atom) {
    Eigen::Vector3d x(atom.xyz[0],atom.xyz[1],atom.xyz[2]);
    Eigen::Vector3d f=g.inverse*x;
    for (int d=0;d<3;++d) f[d]-=std::nearbyint(f[d]);
    const Eigen::Vector3d out=s.lattice*f;
    if (!out.allFinite()) throw std::runtime_error("Gaussian nuclear periodic representative is not finite");
    return out;
}
Atom centered_atom(const PeriodicSystem& s,const Geometry& g,const Atom& atom) {
    // Same operations and order as ewald.cpp's one-time positions packing.
    // Only this one stack object is live; the original unit_cell is borrowed
    // and retained in the input digest, never replaced by a wrapped copy.
    const auto position=centered(s,g,atom);
    return Atom{atom.Z,{position[0],position[1],position[2]}};
}
std::array<char,64> nuclei_digest(const PeriodicGaussianSourceContext& c,const PeriodicSystem& s,const Options& o) {
    Digest d; d.text("vibeqc.periodic.gaussian-nuclear.nuclei-policy"); d.u32(kPeriodicGaussianNuclearVersion);
    d.text(kPeriodicGaussianNuclearPolicy); d.text(kPeriodicGaussianNuclearBoysPolicy);
    d.text(kPeriodicGaussianNuclearFloatingPointPolicy);
    d.text(c.source_context_identity_sha256()); d.u32(s.dim);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) d.real(s.lattice(i,j));
    d.u32(static_cast<std::uint32_t>(s.charge)); d.u32(static_cast<std::uint32_t>(s.multiplicity));
    d.u64(s.unit_cell.size());
    for (const auto& a:s.unit_cell) { d.u32(a.Z); for (double x:a.xyz) d.real(x); }
    d.real(o.alpha); d.real(o.real_cutoff_bohr); d.real(o.reciprocal_cutoff_bohr_inverse);
    return d.finish();
}
double interplanar(const Eigen::Matrix3d& m,int axis) {
    const Eigen::Vector3d a=m.col((axis+1)%3),b=m.col((axis+2)%3);
    const double norm=finite(a.cross(b).norm()),det=finite(m.determinant());
    if (norm<1e-14) throw std::invalid_argument("Gaussian nuclear degenerate lattice plane");
    const double h=finite(std::abs(det)/norm);
    if (h<=0) throw std::invalid_argument("Gaussian nuclear zero lattice-plane separation");
    return h;
}
struct IntBox {
    std::array<int,3> low{},high{};
    std::uint64_t candidates=1;
};
int checked_int(double x) {
    if (!std::isfinite(x)||x<=static_cast<double>(std::numeric_limits<int>::min())
        ||x>=static_cast<double>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Gaussian nuclear native Ewald integer-loop bound exceeded");
    return static_cast<int>(x);
}
IntBox reciprocal_box(const Geometry& g,const Options& o) {
    IntBox b;
    for (int d=0;d<3;++d) {
        b.high[d]=checked_int(std::ceil(o.reciprocal_cutoff_bohr_inverse/interplanar(g.reciprocal,d)));
        b.low[d]=-b.high[d]; b.candidates=mul(b.candidates,2U*static_cast<std::uint64_t>(b.high[d])+1U);
    }
    // Existing accepted-vector count is converted to int. Bounding the full
    // box is conservative and also protects each native loop increment.
    limit(b.candidates,std::numeric_limits<int>::max(),"Gaussian nuclear reciprocal box exceeds native int count");
    return b;
}
template<class Visitor> void walk_reciprocal(const Geometry& g,const Options& o,const IntBox& box,Visitor&& visit) {
    const double cut2=o.reciprocal_cutoff_bohr_inverse*o.reciprocal_cutoff_bohr_inverse;
    for (int x=box.low[0];x<=box.high[0];++x) for (int y=box.low[1];y<=box.high[1];++y)
        for (int z=box.low[2];z<=box.high[2];++z) {
            if (!x&&!y&&!z) continue;
            // Deliberately the original Ewald arithmetic, not a new source
            // sphere or the correlation q-source's separate numerical rule.
            const Eigen::Vector3d v=static_cast<double>(x)*g.reciprocal.col(0)
                +static_cast<double>(y)*g.reciprocal.col(1)+static_cast<double>(z)*g.reciprocal.col(2);
            const double squared=finite(v.squaredNorm());
            if (squared<=cut2) {
                if (squared<=0) throw std::runtime_error("Gaussian nuclear nonzero reciprocal label has zero norm");
                visit(std::array<int,3>{x,y,z},v,squared);
            }
        }
}
std::array<double,3> ewald_spans(const PeriodicSystem& s,const Options& o) {
    std::array<double,3> spans{};
    for (int d=0;d<3;++d) spans[d]=finite(o.real_cutoff_bohr/interplanar(s.lattice,d));
    return spans;
}
Eigen::Vector3d pair_fractional(const Geometry& g,const Eigen::Vector3d& difference) {
    Eigen::Vector3d f=g.inverse*difference;
    for (int d=0;d<3;++d) f[d]-=std::nearbyint(f[d]);
    if (!f.allFinite()) throw std::runtime_error("Gaussian nuclear Ewald pair fractional coordinate is not finite");
    return f;
}
IntBox ewald_pair_box(const Eigen::Vector3d& f,const std::array<double,3>& spans) {
    IntBox b;
    for (int d=0;d<3;++d) {
        const double low=-f[d]-spans[d],high=-f[d]+spans[d];
        const double padding=16.0*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(low),std::abs(high)});
        b.low[d]=checked_int(std::ceil(low-padding)); b.high[d]=checked_int(std::floor(high+padding));
        if (b.high[d]<b.low[d]) { b.candidates=0; continue; }
        b.candidates=mul(b.candidates,static_cast<std::uint64_t>(static_cast<std::int64_t>(b.high[d])-b.low[d]+1));
    }
    return b;
}
std::uint64_t ewald_uniform_candidates(const std::array<double,3>& spans) {
    IntBox b;
    for (int d=0;d<3;++d) {
        const double edge=finite(spans[d]+0.5);
        const double pad=16.0*std::numeric_limits<double>::epsilon()*std::max(1.0,edge);
        const auto high=checked_int(std::ceil(edge+pad));
        b.candidates=mul(b.candidates,2U*static_cast<std::uint64_t>(high)+1U);
    }
    return b.candidates;
}
template<class P> void memory_limits(P& p,const Live& live,const Caps& caps,std::uint64_t basis_bytes=0) {
    const auto borrowed=add(basis_bytes,p.borrowed_system_active_numeric_bytes);
    p.per_replica_inventoried_bytes=add(add(p.owned_numeric_peak_bytes,borrowed),
        add(p.fixed_inventoried_object_bytes,add(live.other_retained_bytes_per_replica,
        add(live.other_transient_bytes_per_replica,live.fixed_backend_margin_bytes_per_replica))));
    p.node_inventoried_bytes=add(live.external_node_bytes,mul(live.replicas_per_node,p.per_replica_inventoried_bytes));
    limit(p.owned_numeric_peak_bytes,caps.maximum_owned_numeric_bytes,"Gaussian nuclear owned numerical byte cap exceeded");
    limit(p.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,"Gaussian nuclear per-replica byte cap exceeded");
    limit(p.node_inventoried_bytes,caps.maximum_node_inventoried_bytes,"Gaussian nuclear node byte cap exceeded");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian nuclear work cap exceeded");
}
std::uint64_t system_controls(std::uint64_t n) {
    // Atomic active numeric lanes are exactly 28N; owning padding/control
    // is separate. Optional symmetry and excess owner capacities belong in
    // the caller's explicit live inventory, never counted as physical data.
    return add(sizeof(PeriodicSystem),mul(n,sizeof(Atom)-28U));
}
PeriodicGaussianSourceCaps clamped_basis_caps(const PeriodicGaussianSourceContext& c,const PeriodicGaussianSourceCaps& caller) {
    const auto& v=c.inventory(); const auto& a=v.ao; const auto& b=v.auxiliary;
    PeriodicGaussianSourceCaps exact;
    exact.maximum_context_storage_bytes=sizeof(PeriodicGaussianSourceContext);
    exact.maximum_kpoint_count=c.mesh().size();
    exact.maximum_shell_count=add(a.shell_count,b.shell_count);
    exact.maximum_contraction_count=add(a.contraction_count,b.contraction_count);
    exact.maximum_primitive_numeric_lanes=add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count));
    exact.maximum_basis_content_wire_bytes=add(a.content_wire_bytes,b.content_wire_bytes);
    exact.maximum_borrowed_active_numeric_bytes=v.combined_borrowed_active_numeric_bytes;
    exact.maximum_work_units=v.work_units_upper_bound;
#define NUCLEAR_BASIS_CAP(f) if (!caller.f) throw std::invalid_argument("Gaussian nuclear basis caps must be positive"); \
    limit(exact.f,caller.f,"Gaussian nuclear basis verification cap is insufficient")
    NUCLEAR_BASIS_CAP(maximum_context_storage_bytes); NUCLEAR_BASIS_CAP(maximum_kpoint_count);
    NUCLEAR_BASIS_CAP(maximum_shell_count); NUCLEAR_BASIS_CAP(maximum_contraction_count);
    NUCLEAR_BASIS_CAP(maximum_primitive_numeric_lanes); NUCLEAR_BASIS_CAP(maximum_basis_content_wire_bytes);
    NUCLEAR_BASIS_CAP(maximum_borrowed_active_numeric_bytes); NUCLEAR_BASIS_CAP(maximum_work_units);
#undef NUCLEAR_BASIS_CAP
    return exact;
}
} // namespace

EPlan plan_periodic_gaussian_nuclear_ewald(std::shared_ptr<const PeriodicGaussianSourceContext> context,
    const PeriodicSystem& system,const Options& options,const Live& live,const Caps& caps) {
    require_context(context); validate_options(options,live,caps);
    EPlan p; p.options=options; p.live=live; p.caps=caps;
    p.atom_count=system_count(system,caps); p.atom_pair_count=mul(p.atom_count,p.atom_count);
    limit(p.atom_pair_count,std::numeric_limits<Eigen::Index>::max(),"Gaussian nuclear atom-pair extent exceeds Eigen index");
    limit(mul(3U,p.atom_count),std::numeric_limits<Eigen::Index>::max(),"Gaussian nuclear positions exceed Eigen index");
    p.borrowed_system_active_numeric_bytes=mul(28U,p.atom_count);
    p.packed_input_numeric_bytes=mul(32U,p.atom_count); p.periodic_positions_numeric_bytes=mul(24U,p.atom_count);
    p.owned_numeric_peak_bytes=add(p.packed_input_numeric_bytes,p.periodic_positions_numeric_bytes);
    p.fixed_inventoried_object_bytes=add(system_controls(p.atom_count),sizeof(PeriodicGaussianSourceContext)
        +sizeof(PeriodicGaussianNuclearEwald)+sizeof(EPlan)+sizeof(Geometry)+3U*sizeof(Digest)
        +sizeof(Eigen::Matrix3Xd)+sizeof(Eigen::VectorXd)+sizeof(std::vector<Eigen::Vector3d>));
    p.preflight_work_units=add(4096U,add(mul(1024U,p.atom_count),mul(2048U,p.atom_pair_count)));
    p.work_units_upper_bound=p.preflight_work_units;
    memory_limits(p,live,caps); // no full atom or pair scan before this gate
    const auto g=geometry(*context,system); derived_controls(g,options);
    const auto spans=ewald_spans(system,options);
    const auto box=reciprocal_box(g,options); p.reciprocal_box_candidates=box.candidates;
    limit(box.candidates,caps.maximum_reciprocal_candidates,"Gaussian nuclear reciprocal candidate cap exceeded");
    limit(box.candidates,std::vector<Eigen::Vector3d>().max_size(),"Gaussian nuclear reciprocal reserve extent exceeded");
    p.reciprocal_reserved_numeric_bytes=mul(24U,box.candidates);
    p.owned_numeric_peak_bytes=add(p.owned_numeric_peak_bytes,p.reciprocal_reserved_numeric_bytes);
    p.real_pair_candidates_upper_bound=mul(p.atom_pair_count,ewald_uniform_candidates(spans));
    // Both the source-receipt replay and legacy evaluation traverse them.
    limit(mul(2U,p.real_pair_candidates_upper_bound),caps.maximum_ewald_pair_candidates,"Gaussian nuclear Ewald pair candidate cap exceeded");
    p.work_units_upper_bound=add(p.preflight_work_units,
        add(mul(4096U,p.real_pair_candidates_upper_bound),mul(2048U,mul(box.candidates,add(p.atom_count,1U)))));
    limit(add(4096U,add(mul(28U,p.atom_count),mul(256U,add(p.atom_pair_count,
        add(p.real_pair_candidates_upper_bound,p.reciprocal_box_candidates))))),sha_max,
        "Gaussian nuclear Ewald source SHA extent exceeded");
    memory_limits(p,live,caps);
    validate_atoms(system,g);
    std::uint64_t exact=0;
    for (std::uint64_t a=0;a<p.atom_count;++a) for (std::uint64_t b=0;b<p.atom_count;++b) {
        const auto f=pair_fractional(g,centered(system,g,system.unit_cell[a])-centered(system,g,system.unit_cell[b]));
        if (a!=b && system.unit_cell[a].Z && system.unit_cell[b].Z && (system.lattice*f).norm()<1e-14)
            throw std::invalid_argument("Gaussian nuclear distinct nonzero nuclei are periodically coincident");
        exact=add(exact,ewald_pair_box(f,spans).candidates);
    }
    if (exact>p.real_pair_candidates_upper_bound) throw std::logic_error("Gaussian nuclear Ewald uniform pair admission failed");
    return p;
}

PeriodicGaussianNuclearEwald build_periodic_gaussian_nuclear_ewald(
    std::shared_ptr<const PeriodicGaussianSourceContext> context,const PeriodicSystem& system,
    const Options& options,const Live& live,const Caps& caps) {
    auto p=plan_periodic_gaussian_nuclear_ewald(context,system,options,live,caps);
    const auto g=geometry(*context,system); const auto spans=ewald_spans(system,options);
    PeriodicGaussianNuclearEwald out; out.context_=std::move(context); out.plan_=p;
    out.nuclei_=nuclei_digest(*out.context_,system,options);
    Digest source; source.text("vibeqc.periodic.gaussian-nuclear.ewald-source"); source.u32(kPeriodicGaussianNuclearVersion);
    source.text(out.nuclei_policy_identity_sha256());
    source.text("native-ewald-centered-pair-box;native-reciprocal-box;OMP-scalar-reduction-v1");
    for (std::uint64_t a=0;a<p.atom_count;++a) for (std::uint64_t b=0;b<p.atom_count;++b) {
        const auto f=pair_fractional(g,centered(system,g,system.unit_cell[a])-centered(system,g,system.unit_cell[b]));
        const auto box=ewald_pair_box(f,spans); source.u64(a); source.u64(b);
        for (int d=0;d<3;++d) { source.u32(box.low[d]); source.u32(box.high[d]); }
        for (int x=box.low[0];x<=box.high[0];++x) for (int y=box.low[1];y<=box.high[1];++y)
            for (int z=box.low[2];z<=box.high[2];++z) {
                const Eigen::Vector3d d=system.lattice*(f+Eigen::Vector3d(x,y,z));
                const double r=finite(d.norm());
                if (r<1e-14||r>options.real_cutoff_bohr) continue;
                source.u32(x); source.u32(y); source.u32(z); source.real(r);
            }
    }
    walk_reciprocal(g,options,reciprocal_box(g,options),[&](const auto& label,const auto& v,double squared) {
        for (int n:label) source.u32(n); for (int d=0;d<3;++d) source.real(v[d]); source.real(squared);
    });
    out.source_=source.finish();
    Eigen::Matrix3Xd positions(3,static_cast<Eigen::Index>(p.atom_count));
    Eigen::VectorXd charges(static_cast<Eigen::Index>(p.atom_count));
    for (std::uint64_t a=0;a<p.atom_count;++a) {
        for (int d=0;d<3;++d) positions(d,static_cast<Eigen::Index>(a))=system.unit_cell[a].xyz[d];
        charges[static_cast<Eigen::Index>(a)]=system.unit_cell[a].Z;
    }
    EwaldOptions native; native.alpha=options.alpha; native.real_cutoff_bohr=options.real_cutoff_bohr;
    native.recip_cutoff_bohr_inv=options.reciprocal_cutoff_bohr_inverse;
    out.energy_=finite(ewald_point_charge_energy(system.lattice,positions,charges,native));
    Digest payload; payload.text("vibeqc.periodic.gaussian-nuclear.ewald-payload"); payload.u32(kPeriodicGaussianNuclearVersion);
    payload.text(out.scalar_source_identity_sha256()); payload.real(out.energy_); out.payload_=payload.finish();
    return out;
}
std::string PeriodicGaussianNuclearEwald::nuclei_policy_identity_sha256() const { return {nuclei_.begin(),nuclei_.end()}; }
std::string PeriodicGaussianNuclearEwald::scalar_source_identity_sha256() const { return {source_.begin(),source_.end()}; }
std::string PeriodicGaussianNuclearEwald::payload_identity_sha256() const { return {payload_.begin(),payload_.end()}; }

namespace {
std::uint64_t cartesian(std::uint64_t l) { return (l+1U)*(l+2U)/2U; }
std::uint64_t os_bytes(std::uint64_t l) {
    const auto t=2U*l+1U,b=l+1U,c=cartesian(l);
    auto doubles=add(mul(mul(t,t),mul(t,t)),add(mul(mul(t,t),mul(t,mul(mul(b,b),b))),2U*c*c));
    std::uint64_t integers=0;
    for (std::uint64_t j=0;j<=l;++j) { doubles=add(doubles,(2U*j+1U)*cartesian(j)); integers=add(integers,3U*cartesian(j)); }
    return add(mul(8U,doubles),mul(sizeof(int),integers));
}
std::uint64_t nuclear_uniform_box(const Geometry& g,const Options& o) {
    std::uint64_t count=1;
    for (int d=0;d<3;++d) {
        const long double radius=static_cast<long double>(o.real_cutoff_bohr)*g.inverse_long.row(d).norm();
        const long double pad=1.0L+64.0L*std::numeric_limits<long double>::epsilon()*(1.0L+0x1p44L+radius);
        const long double half=std::ceil(radius+pad)+1.0L;
        if (!std::isfinite(half)||half<0||half>0x1p48L)
            throw std::overflow_error("Gaussian nuclear image box width exceeds policy");
        count=mul(count,2U*static_cast<std::uint64_t>(half)+1U);
    }
    return count;
}
pair::ImageBox nuclear_box(const Eigen::Vector3d& p,const Atom& atom,const Geometry& g,const Options& o) {
    Eigen::Matrix<long double,3,1> delta;
    for (int d=0;d<3;++d) delta[d]=static_cast<long double>(p[d])-atom.xyz[d];
    const auto center=(g.inverse_long*delta).eval();
    pair::ImageBox box;
    for (int d=0;d<3;++d) {
        if (!std::isfinite(center[d])||std::abs(center[d])>0x1p44L)
            throw std::length_error("Gaussian nuclear primitive-product image center exceeds policy");
        const long double radius=static_cast<long double>(o.real_cutoff_bohr)*g.inverse_long.row(d).norm();
        const long double pad=1.0L+64.0L*std::numeric_limits<long double>::epsilon()*(1.0L+std::abs(center[d])+radius);
        const long double low=std::floor(center[d]-radius-pad),high=std::ceil(center[d]+radius+pad);
        if (!std::isfinite(low)||!std::isfinite(high)||low< -0x1p52L||high>0x1p52L)
            throw std::length_error("Gaussian nuclear image label exceeds exact binary64 range");
        box.lower[d]=static_cast<std::int64_t>(low); box.upper[d]=static_cast<std::int64_t>(high);
        box.candidates=mul(box.candidates,static_cast<std::uint64_t>(box.upper[d]-box.lower[d])+1U);
    }
    return box;
}
template<class Visitor> void walk_nuclear(const PeriodicSystem& system,const Atom& atom,
    const Eigen::Vector3d& p,const pair::ImageBox& box,double cut2,Visitor&& visit) {
    for (std::int64_t x=box.lower[0];x<=box.upper[0];++x)
        for (std::int64_t y=box.lower[1];y<=box.upper[1];++y)
            for (std::int64_t z=box.lower[2];z<=box.upper[2];++z) {
                std::array<double,3> c{},delta{};
                const std::array<std::int64_t,3> label{x,y,z};
                for (int d=0;d<3;++d) {
                    double translation=0;
                    for (int axis=2;axis>=0;--axis) translation=std::fma(system.lattice(d,axis),static_cast<double>(label[axis]),translation);
                    c[d]=finite(atom.xyz[d]+translation); delta[d]=finite(p[d]-c[d]);
                }
                double squared=0;
                for (int d=2;d>=0;--d) squared=std::fma(delta[d],delta[d],squared);
                finite(squared); if (squared<=cut2) visit(label,c,squared);
            }
}
BoysTable make_boys(unsigned l,Diagnostics& d) {
    BoysTable table; table.n_max=2*static_cast<int>(l); table.t_max=100.0; table.dt=.01;
    table.n_grid=static_cast<int>(boys_rows);
    const int top=table.n_max+6,stride=table.n_max+7;
    table.values.resize(static_cast<std::size_t>(boys_rows)*stride);
    for (std::uint64_t row=0;row<boys_rows;++row) {
        // Grid locations are the SAME represented index*dt used by eval.
        const double td=static_cast<double>(row)*table.dt;
        const long double t=td;
        long double term=1.0L/(2*top+1),sum=term;
        bool converged=t==0;
        unsigned used=0;
        for (unsigned k=1;k<=boys_iterations && !converged;++k) {
            term*=2.0L*t/(2*top+2*k+1); sum+=term; used=k;
            const long double ratio=2.0L*t/(2*top+2*k+3);
            // Subsequent ratios decrease: term*ratio/(1-ratio) bounds
            // the remaining POSITIVE mathematical series, not FP roundoff.
            if (ratio<1 && term*ratio/(1-ratio)<=std::numeric_limits<long double>::epsilon()*sum/8)
                converged=true;
            if (!std::isfinite(term)||!std::isfinite(sum)) throw std::runtime_error("Gaussian nuclear Boys positive series overflow");
        }
        if (!converged) throw std::runtime_error("Gaussian nuclear Boys positive series did not converge within cap");
        d.maximum_boys_series_iterations=std::max(d.maximum_boys_series_iterations,used);
        const long double exponential=std::exp(-t);
        long double value=exponential*sum;
        table.values[static_cast<std::size_t>(row)*stride+top]=finite(static_cast<double>(value));
        for (int n=top;n>0;--n) {
            value=(2*t*value+exponential)/(2*n-1);
            table.values[static_cast<std::size_t>(row)*stride+n-1]=finite(static_cast<double>(value));
        }
    }
    return table;
}
void reserve_shell(libint2::Shell& shell,std::size_t primitives) {
    shell.alpha.reserve(primitives); shell.max_ln_coeff.reserve(primitives);
    shell.contr.resize(1); shell.contr[0].coeff.reserve(primitives);
}
void copy_shell(libint2::Shell& dst,const pair::Ao& src,const Eigen::Vector3d& translation) {
    dst.alpha.resize(src.shell->alpha.size()); dst.max_ln_coeff.resize(src.shell->alpha.size());
    dst.contr[0].coeff.resize(src.shell->alpha.size()); dst.contr[0].l=src.contraction->l;
    dst.contr[0].pure=src.contraction->pure;
    for (int d=0;d<3;++d) dst.O[d]=finite(src.shell->O[d]+translation[d]);
    for (std::size_t p=0;p<src.shell->alpha.size();++p) {
        dst.alpha[p]=src.shell->alpha[p]; dst.contr[0].coeff[p]=src.contraction->coeff[p];
        dst.max_ln_coeff[p]=src.contraction->coeff[p]==0 ? -std::numeric_limits<double>::infinity()
            :std::log(std::abs(src.contraction->coeff[p]));
    }
}
void seed_diagnostics(const PrimitivePairData& p,const std::array<double,3>& c,const BoysTable& table,
    int order,double omega,Diagnostics& d) {
    const double dx=p.P[0]-c[0],dy=p.P[1]-c[1],dz=p.P[2]-c[2];
    const double t=finite(p.alpha*(dx*dx+dy*dy+dz*dz));
    const double denominator=finite(omega*omega+p.alpha),rho=omega*omega/denominator;
    if (!(rho>0 && rho<1)) throw std::runtime_error("Gaussian nuclear erfc ratio is not representable strictly inside (0,1)");
    double power=std::sqrt(rho);
    for (int m=0;m<=order;++m) {
        const double f=finite(table.eval(m,t)),lr=finite(power*table.eval(m,rho*t));
        const double difference=finite(f-lr),scale=std::max(std::abs(f),std::abs(lr));
        d.maximum_erfc_seed_term_magnitude=std::max(d.maximum_erfc_seed_term_magnitude,
            finite(std::abs(p.prefactor)*scale));
        if (difference==0 && scale>0) ++d.rounded_zero_erfc_seed_differences;
        else if (scale>0) d.minimum_nonzero_erfc_seed_difference_ratio=std::min(d.minimum_nonzero_erfc_seed_difference_ratio,std::abs(difference)/scale);
        power*=rho;
    }
}
struct ShortWorkspace {
    BoysTable boys;
    CosxKernelWorkspace os;
    libint2::Shell a,b;
    std::vector<PrimitivePairData> primitive;
    std::vector<double> output;
};
Complex short_pair(const pair::Ao& a,const pair::Ao& b,const PeriodicSystem& system,const Geometry& g,
    const Eigen::Vector3d& k,double ao_cutoff,const Options& options,const Plan& plan,
    ShortWorkspace& w,Digest& source,Diagnostics& d) {
    const auto box=pair::image_box(a,b,g.inverse_long,ao_cutoff);
    source.u64(box.candidates);
    d.completed_ao_image_candidates=add(d.completed_ao_image_candidates,box.candidates);
    for (int axis=0;axis<3;++axis) { source.u64(box.lower[axis]); source.u64(box.upper[axis]); }
    CSum sum;
    const auto images=pair::walk_images(a,b,system.lattice,box,ao_cutoff*ao_cutoff,
        [&](const auto& label,const Eigen::Vector3d& translation,const Eigen::Vector3d&,double squared) {
            for (auto n:label) source.u64(n);
            copy_shell(w.a,a,Eigen::Vector3d::Zero()); copy_shell(w.b,b,translation);
            const double angle=finite(k.dot(translation)); const Complex phase(std::cos(angle),std::sin(angle));
            for (std::size_t ia=0;ia<w.a.alpha.size();++ia) for (std::size_t ib=0;ib<w.b.alpha.size();++ib) {
                auto& primitive=w.primitive[0];
                const double aa=w.a.alpha[ia],bb=w.b.alpha[ib]; primitive.alpha=finite(aa+bb);
                primitive.inv_alpha=finite(1.0/primitive.alpha); primitive.inv_two_alpha=.5*primitive.inv_alpha;
                const double wa=aa/primitive.alpha,wb=bb/primitive.alpha;
                Eigen::Vector3d p;
                for (int axis=0;axis<3;++axis) p[axis]=primitive.P[axis]=finite(wa*w.a.O[axis]+wb*w.b.O[axis]);
                primitive.prefactor=finite(w.a.contr[0].coeff[ia]*w.b.contr[0].coeff[ib]
                    *(2.0*pi*primitive.inv_alpha)*std::exp(-wa*bb*squared));
                source.u64(ia); source.u64(ib);
                for (std::size_t atom=0;atom<system.unit_cell.size();++atom) {
                    const auto nucleus=centered_atom(system,g,system.unit_cell[atom]);
                    const auto nuclear=nuclear_box(p,nucleus,g,options);
                    d.completed_nuclear_image_candidates=add(d.completed_nuclear_image_candidates,nuclear.candidates);
                    if (d.completed_nuclear_image_candidates>plan.nuclear_image_candidates_upper_bound)
                        throw std::logic_error("Gaussian nuclear source exceeded admitted image bound");
                    source.u64(atom); source.u64(nuclear.candidates);
                    for (int axis=0;axis<3;++axis) { source.u64(nuclear.lower[axis]); source.u64(nuclear.upper[axis]); }
                    walk_nuclear(system,nucleus,p,nuclear,options.real_cutoff_bohr*options.real_cutoff_bohr,
                        [&](const auto& nuclear_label,const auto& center,double) {
                            for (auto n:nuclear_label) source.u64(n);
                            ++d.retained_nuclear_images;
                            seed_diagnostics(primitive,center,w.boys,a.contraction->l+b.contraction->l,options.alpha,d);
                            cosx_nuclear_pair_into(w.a,w.b,w.primitive,center,w.boys,w.os,w.output.data(),options.alpha);
                            const auto n_b=static_cast<std::size_t>(2*b.contraction->l+1);
                            const double value=finite(w.output[a.component*n_b+b.component]);
                            sum.add(finite(-static_cast<double>(nucleus.Z)*value*phase));
                        });
                }
            }
        });
    d.retained_ao_images=add(d.retained_ao_images,images);
    return sum.result();
}
Complex fourier(const BasisSet& basis,const PeriodicSystem& system,const Eigen::Vector3d& k,
    const Eigen::Vector3d& vector,std::uint64_t index,double cut,std::uint64_t candidate_cap,
    Diagnostics& d) {
    AuxiliaryFourierVectorView view{&vector[0],&vector[1],&vector[2],1,1,1,1};
    auto panel=ao_pair_gaussian_fourier_panel(basis,system,view,k,index,1,cut,candidate_cap,16);
    ++d.completed_ao_fourier_calls;
    // Native FT performs both a source census and the numerical walk.
    d.completed_ao_image_candidates=add(d.completed_ao_image_candidates,mul(2U,panel.image_candidate_count));
    if (panel.data.size()!=1) throw std::logic_error("Gaussian nuclear one-vector FT extent changed");
    return finite(panel.data[0]);
}
std::array<Complex,4> evaluate_pair(const BasisSet& basis,const PeriodicSystem& system,const Geometry& g,
    const Eigen::Vector3d& k,std::uint64_t index,std::uint32_t role,double ao_cutoff,
    const Plan& p,ShortWorkspace& w,Digest& source,Diagnostics& d) {
    const auto a=pair::ao(basis,index/p.n_basis),b=pair::ao(basis,index%p.n_basis);
    source.u64(index); source.u32(role);
    const Complex short_value=short_pair(a,b,system,g,k,ao_cutoff,p.options,p,w,source,d);
    CSum long_value;
    walk_reciprocal(g,p.options,reciprocal_box(g,p.options),[&](const auto& label,const Eigen::Vector3d& v,double squared) {
        for (int n:label) source.u32(n); for (int axis=0;axis<3;++axis) source.real(v[axis]);
        ++d.retained_reciprocal_vectors;
        CSum structure;
        for (const auto& atom:system.unit_cell) {
            const double phase=finite(v.dot(centered(system,g,atom)));
            structure.add(static_cast<double>(atom.Z)*Complex(std::cos(phase),-std::sin(phase)));
        }
        const double weight=finite((-4.0*pi/g.volume)*std::exp(-squared/(4.0*p.options.alpha*p.options.alpha))/squared);
        const Eigen::Vector3d negative=-v;
        const auto rho=fourier(basis,system,k,negative,index,ao_cutoff,p.caps.maximum_ao_image_candidates,d);
        long_value.add(finite(weight*structure.result()*rho));
    });
    double q=0; for (const auto& atom:system.unit_cell) q=finite(q+atom.Z);
    const auto overlap=fourier(basis,system,k,Eigen::Vector3d::Zero(),index,ao_cutoff,p.caps.maximum_ao_image_candidates,d);
    const auto background=finite((pi*q/(g.volume*p.options.alpha*p.options.alpha))*overlap);
    CSum total; total.add(short_value); total.add(long_value.result()); total.add(background);
    return {short_value,long_value.result(),background,total.result()};
}
double audit(Complex a,Complex b,const Options& o,const char* message) {
    const double error=finite(std::abs(finite(a-b))),scale=std::max(finite(std::abs(a)),finite(std::abs(b)));
    if (error>finite(o.structural_absolute_tolerance+o.structural_relative_tolerance*scale)) throw std::runtime_error(message);
    return error;
}
} // namespace

Plan plan_periodic_gaussian_nuclear_panel(std::shared_ptr<const PeriodicGaussianSourceContext> context,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicSystem& system,
    std::uint64_t k,std::uint64_t begin,std::uint64_t count,const Options& options,const Live& live,const Caps& caps) {
    require_context(context); validate_options(options,live,caps);
    const auto basis_caps=clamped_basis_caps(*context,options.basis_verification_caps);
    Plan p; p.options=options; p.live=live; p.caps=caps;
    p.atom_count=system_count(system,caps); p.n_basis=context->inventory().ao.function_count;
    const auto square=mul(p.n_basis,p.n_basis);
    if (!count||begin>square||count>square-begin||k>=context->mesh().size())
        throw std::invalid_argument("Gaussian nuclear k or nonempty pair interval is invalid");
    limit(count,caps.maximum_pair_count,"Gaussian nuclear pair count exceeds cap");
    limit(mul(4U,count),std::vector<Complex>().max_size(),"Gaussian nuclear panel extent exceeded");
    p.k_index=k; p.opposite_k_index=context->mesh().negate_index(k); p.pair_begin=begin; p.pair_count=count;
    p.output_numeric_bytes=mul(64U,count); p.owned_numeric_peak_bytes=p.output_numeric_bytes;
    p.borrowed_basis_active_numeric_bytes=context->inventory().combined_borrowed_active_numeric_bytes;
    p.borrowed_system_active_numeric_bytes=mul(28U,p.atom_count);
    p.fixed_inventoried_object_bytes=add(system_controls(p.atom_count),sizeof(PeriodicGaussianSourceContext)
        +sizeof(PeriodicGaussianNuclearPanel)+sizeof(Plan)+sizeof(Geometry)+3U*sizeof(Digest)
        +sizeof(ShortWorkspace)+sizeof(AOPairFourierPanel)+2U*sizeof(libint2::Shell::Contraction));
    p.basis_scan_work_units=context->inventory().work_units_upper_bound;
    const auto lookup=add(context->inventory().ao.shell_count,context->inventory().ao.contraction_count);
    p.preflight_work_units=add(p.basis_scan_work_units,add(4096U,add(mul(1024U,p.atom_count),mul(512U,mul(count,lookup)))));
    p.work_units_upper_bound=add(p.preflight_work_units,mul(128U,p.output_numeric_bytes));
    memory_limits(p,live,caps,p.borrowed_basis_active_numeric_bytes);
    const auto g=geometry(*context,system);
    derived_controls(g,options);
    const auto reciprocal=reciprocal_box(g,options);
    p.reciprocal_box_candidates=reciprocal.candidates;
    p.reciprocal_candidate_evaluations_upper_bound=mul(3U,mul(count,reciprocal.candidates));
    limit(p.reciprocal_candidate_evaluations_upper_bound,caps.maximum_reciprocal_candidates,"Gaussian nuclear reciprocal candidate cap exceeded");
    const auto nuclear_candidates=nuclear_uniform_box(g,options);
    p.ao_fourier_calls_upper_bound=mul(3U,mul(count,add(reciprocal.candidates,1U)));
    p.work_units_upper_bound=add(p.work_units_upper_bound,
        add(mul(256U,mul(boys_rows,boys_iterations)),mul(p.ao_fourier_calls_upper_bound,
            add(mul(2U,p.basis_scan_work_units),add(mul(2048U,lookup),mul(1024U,p.atom_count))))));
    memory_limits(p,live,caps,p.borrowed_basis_active_numeric_bytes);
    context->verify_bases(ao,auxiliary,basis_caps); validate_atoms(system,g);
    std::uint64_t maximum_primitives=0,short_candidates=0;
    long double atom_fractional_bound=0;
    for (const auto& atom:system.unit_cell) {
        const auto position=centered(system,g,atom);
        const Eigen::Matrix<long double,3,1> x(position[0],position[1],position[2]);
        atom_fractional_bound=std::max(atom_fractional_bound,(g.inverse_long*x).cwiseAbs().maxCoeff());
    }
    for (std::uint64_t row=0;row<count;++row) {
        const auto a=pair::ao(ao,(begin+row)/p.n_basis),b=pair::ao(ao,(begin+row)%p.n_basis);
        p.maximum_selected_angular_momentum=std::max(p.maximum_selected_angular_momentum,
            static_cast<std::uint32_t>(std::max(a.contraction->l,b.contraction->l)));
        maximum_primitives=std::max(maximum_primitives,static_cast<std::uint64_t>(std::max(a.shell->alpha.size(),b.shell->alpha.size())));
        const auto forward=pair::image_box(a,b,g.inverse_long,context->options().ao_pair_image_cutoff_bohr);
        const auto reverse=pair::image_box(b,a,g.inverse_long,context->options().ao_pair_image_cutoff_bohr);
        const auto candidates=add(mul(2U,forward.candidates),reverse.candidates);
        short_candidates=add(short_candidates,candidates);
        const auto primitive=mul(candidates,mul(a.shell->alpha.size(),b.shell->alpha.size()));
        p.primitive_pair_evaluations_upper_bound=add(p.primitive_pair_evaluations_upper_bound,primitive);
        p.nuclear_image_candidates_upper_bound=add(p.nuclear_image_candidates_upper_bound,
            mul(primitive,mul(p.atom_count,nuclear_candidates)));
        limit(p.nuclear_image_candidates_upper_bound,caps.maximum_nuclear_image_candidates,"Gaussian nuclear nuclear-image candidate cap exceeded");
        const auto na=cartesian(a.contraction->l),nb=cartesian(b.contraction->l);
        const auto cells=3U*(a.contraction->l+1U)*(b.contraction->l+1U)*(a.contraction->l+b.contraction->l+1U);
        const auto ft_primitive=mul(primitive,add(reciprocal.candidates,1U));
        p.work_units_upper_bound=add(p.work_units_upper_bound,
            add(mul(4096U,mul(candidates,add(1U,mul(2U,add(reciprocal.candidates,1U))))),
                mul(ft_primitive,add(2048U,add(mul(128U,cells),mul(1024U,mul(na,nb)))))));
        // Product centers are convex combinations of translated AO centers.
        // Precheck a conservative absolute fractional bound before scans of
        // AO images/primitive pairs/nuclear images and numerical allocation.
        long double ao_bound=0,label_bound=0;
        for (const auto* shell:{a.shell,b.shell}) {
            const Eigen::Matrix<long double,3,1> x(shell->O[0],shell->O[1],shell->O[2]);
            ao_bound=std::max(ao_bound,(g.inverse_long*x).cwiseAbs().maxCoeff());
        }
        for (int axis=0;axis<3;++axis) for (auto v:{forward.lower[axis],forward.upper[axis],reverse.lower[axis],reverse.upper[axis]})
            label_bound=std::max(label_bound,std::abs(static_cast<long double>(v)));
        if (!std::isfinite(ao_bound)||ao_bound+label_bound+atom_fractional_bound+1>0x1p44L)
            throw std::length_error("Gaussian nuclear AO/nuclear image center preflight exceeds policy");
        limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian nuclear work cap exceeded before image enumeration");
    }
    p.ao_image_candidates_upper_bound=mul(short_candidates,add(1U,mul(2U,add(reciprocal.candidates,1U))));
    limit(p.ao_image_candidates_upper_bound,caps.maximum_ao_image_candidates,"Gaussian nuclear AO-image candidate cap exceeded");
    const auto l=p.maximum_selected_angular_momentum;
    p.boys_table_numeric_bytes=mul(8U,mul(boys_rows,2U*l+7U));
    p.os_workspace_numeric_bytes=os_bytes(l);
    p.selected_shell_numeric_bytes_upper_bound=add(48U,mul(48U,maximum_primitives));
    p.primitive_and_shell_output_numeric_bytes=add(sizeof(PrimitivePairData),mul(8U,mul(2U*l+1U,2U*l+1U)));
    p.fourier_numeric_workspace_bytes=add(16U,ao_pair_fourier_fixed_numeric_workspace_bytes());
    p.owned_numeric_peak_bytes=add(p.output_numeric_bytes,add(p.boys_table_numeric_bytes,
        add(p.os_workspace_numeric_bytes,add(p.selected_shell_numeric_bytes_upper_bound,
        add(p.primitive_and_shell_output_numeric_bytes,p.fourier_numeric_workspace_bytes)))));
    p.fixed_inventoried_object_bytes=add(p.fixed_inventoried_object_bytes,
        mul(l+1U,3U*sizeof(std::vector<int>)+sizeof(std::vector<double>)));
    p.work_units_upper_bound=add(p.work_units_upper_bound,
        mul(p.nuclear_image_candidates_upper_bound,add(4096U,mul(128U,add(1U,p.os_workspace_numeric_bytes/4U)))));
    // Full stream hash includes at most a fixed 256-byte record per tested
    // candidate/primitive plus output. Check the SHA bit-length too.
    limit(add(4096U,add(p.output_numeric_bytes,mul(256U,add(p.ao_image_candidates_upper_bound,
        add(p.nuclear_image_candidates_upper_bound,p.reciprocal_candidate_evaluations_upper_bound))))),sha_max,
        "Gaussian nuclear stream SHA extent exceeds representation");
    memory_limits(p,live,caps,p.borrowed_basis_active_numeric_bytes);
    return p;
}

PeriodicGaussianNuclearPanel build_periodic_gaussian_nuclear_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext> context,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicSystem& system,std::uint64_t k,std::uint64_t begin,std::uint64_t count,
    const Options& options,const Live& live,const Caps& caps) {
    const auto p=plan_periodic_gaussian_nuclear_panel(context,ao,auxiliary,system,k,begin,count,options,live,caps);
    const auto g=geometry(*context,system);
    PeriodicGaussianNuclearPanel out; out.context_=std::move(context); out.plan_=p;
    out.nuclei_=nuclei_digest(*out.context_,system,options);
    out.values_.resize(static_cast<std::size_t>(4U*count));
    ShortWorkspace w; w.boys=make_boys(p.maximum_selected_angular_momentum,out.diagnostics_);
    w.os.reserve(p.maximum_selected_angular_momentum,p.maximum_selected_angular_momentum);
    const auto max_primitives=static_cast<std::size_t>((p.selected_shell_numeric_bytes_upper_bound-48U)/48U);
    reserve_shell(w.a,max_primitives); reserve_shell(w.b,max_primitives); w.primitive.resize(1);
    w.output.resize(static_cast<std::size_t>((p.primitive_and_shell_output_numeric_bytes-sizeof(PrimitivePairData))/8U));
    const auto point=out.context_->k_record(k),partner=out.context_->k_record(p.opposite_k_index);
    const Eigen::Vector3d kv(point.cartesian[0],point.cartesian[1],point.cartesian[2]);
    const Eigen::Vector3d kb(partner.cartesian[0],partner.cartesian[1],partner.cartesian[2]);
    Digest source; source.text("vibeqc.periodic.gaussian-nuclear.panel-source"); source.u32(kPeriodicGaussianNuclearVersion);
    source.text(out.nuclei_policy_identity_sha256()); source.text(out.context_->ao_basis_identity_sha256());
    for (auto n:{k,p.opposite_k_index,begin,count}) source.u64(n);
    const double cut=out.context_->options().ao_pair_image_cutoff_bohr;
    auto& d=out.diagnostics_;
    for (std::uint64_t row=0;row<count;++row) {
        const auto mu=(begin+row)/p.n_basis,nu=(begin+row)%p.n_basis;
        const auto original=evaluate_pair(ao,system,g,kv,begin+row,0,cut,p,w,source,d);
        const auto reverse=evaluate_pair(ao,system,g,kv,nu*p.n_basis+mu,1,cut,p,w,source,d);
        const auto opposite=evaluate_pair(ao,system,g,kb,begin+row,2,cut,p,w,source,d);
        for (std::size_t op=0;op<4;++op) {
            d.maximum_hermitian_error[op]=std::max(d.maximum_hermitian_error[op],audit(original[op],std::conj(reverse[op]),options,
                "Gaussian nuclear raw Hermiticity audit failed"));
            d.maximum_time_reversal_error[op]=std::max(d.maximum_time_reversal_error[op],audit(original[op],std::conj(opposite[op]),options,
                "Gaussian nuclear raw time-reversal audit failed"));
            if (mu==nu) d.maximum_diagonal_imaginary_magnitude=std::max(d.maximum_diagonal_imaginary_magnitude,
                audit(original[op],Complex(original[op].real(),0),options,"Gaussian nuclear diagonal imaginary audit failed"));
            if (k==p.opposite_k_index) d.maximum_trim_imaginary_magnitude=std::max(d.maximum_trim_imaginary_magnitude,
                audit(original[op],Complex(original[op].real(),0),options,"Gaussian nuclear TRIM imaginary audit failed"));
            out.values_[op*count+row]=original[op];
        }
    }
    if (d.completed_ao_image_candidates>p.ao_image_candidates_upper_bound
        ||d.completed_nuclear_image_candidates>p.nuclear_image_candidates_upper_bound
        ||d.completed_ao_fourier_calls>p.ao_fourier_calls_upper_bound)
        throw std::logic_error("Gaussian nuclear traversal exceeded admitted census");
    out.source_=source.finish();
    Digest payload; payload.text("vibeqc.periodic.gaussian-nuclear.panel-payload"); payload.u32(kPeriodicGaussianNuclearVersion);
    payload.text(out.panel_source_identity_sha256()); payload.real(options.structural_absolute_tolerance);
    payload.real(options.structural_relative_tolerance); payload.u64(out.values_.size());
    for (auto z:out.values_) { payload.real(z.real()); payload.real(z.imag()); }
    out.payload_=payload.finish(); return out;
}
Complex PeriodicGaussianNuclearPanel::element(std::uint32_t component,std::uint64_t pair_index) const {
    if (!context_||component>3||pair_index>=plan_.pair_count||values_.size()!=4U*plan_.pair_count)
        throw std::out_of_range("Gaussian nuclear panel element is consumed or out of range");
    return values_[component*plan_.pair_count+pair_index];
}
std::string PeriodicGaussianNuclearPanel::nuclei_policy_identity_sha256() const { return {nuclei_.begin(),nuclei_.end()}; }
std::string PeriodicGaussianNuclearPanel::panel_source_identity_sha256() const { return {source_.begin(),source_.end()}; }
std::string PeriodicGaussianNuclearPanel::payload_identity_sha256() const { return {payload_.begin(),payload_.end()}; }

std::vector<double> periodic_gaussian_nuclear_detail::boys_diagnostic(int order,const double* values,
    std::uint64_t count,std::uint64_t maximum_owned_numeric_bytes,std::uint64_t maximum_work_units) {
    float_environment();
    if (order<0||order>12||!count||count>256||!values
        ||reinterpret_cast<std::uintptr_t>(values)%alignof(double)
        ||!maximum_owned_numeric_bytes||!maximum_work_units)
        throw std::invalid_argument("Gaussian nuclear Boys diagnostic requires bounded aligned finite inputs");
    const auto l=static_cast<unsigned>((order+1)/2);
    limit(add(mul(8U,mul(boys_rows,2U*l+7U)),mul(8U,count)),maximum_owned_numeric_bytes,
        "Gaussian nuclear Boys diagnostic byte cap exceeded");
    limit(add(mul(256U,mul(boys_rows,boys_iterations)),mul(1024U,count)),maximum_work_units,
        "Gaussian nuclear Boys diagnostic work cap exceeded");
    for (std::uint64_t i=0;i<count;++i)
        if (!std::isfinite(values[i])||values[i]<0) throw std::invalid_argument("Gaussian nuclear Boys diagnostic T must be finite and nonnegative");
    Diagnostics diagnostics; const auto table=make_boys(l,diagnostics);
    std::vector<double> result(static_cast<std::size_t>(count));
    for (std::uint64_t i=0;i<count;++i) result[i]=finite(table.eval(order,values[i]));
    return result;
}
} // namespace vibeqc
