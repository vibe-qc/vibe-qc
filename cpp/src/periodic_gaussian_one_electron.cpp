#include "vibeqc/periodic_gaussian_one_electron.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "aopair_ft_internal.hpp"
#include "vibeqc/cart_to_sph_data.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_correlation_three_center.hpp"

namespace vibeqc {
namespace {
namespace pair = aopair_ft_detail;
using Complex = std::complex<double>;
using Plan = PeriodicGaussianOneElectronPlan;
using Options = PeriodicGaussianOneElectronOptions;
using Live = PeriodicGaussianOneElectronLiveInventory;
using Caps = PeriodicGaussianOneElectronCaps;
using Diagnostics = PeriodicGaussianOneElectronDiagnostics;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::size_t kAxisExtent = 8U * 8U * 15U;
constexpr char kImageDomain[] = "vibeqc.periodic.gaussian-one-electron.images";
constexpr std::uint64_t kImagePrefix = 8U + sizeof(kImageDomain) - 1U + 4U + 5U * 8U;
constexpr std::uint64_t kShaMax = std::numeric_limits<std::uint64_t>::max() / 8U;
struct Workspace { std::array<double, 3U * kAxisExtent> md; };
static_assert(cart_to_sph_data::kMaxL == 6, "update one-electron derivative workspace");
static_assert(sizeof(Workspace) == 23040 && sizeof(Complex) == 16,
              "one-electron panel accounting requires binary64");
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("Gaussian one-electron count overflow");
    return a+b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max()/a) throw std::overflow_error("Gaussian one-electron count overflow");
    return a*b;
}
void limit(std::uint64_t value, std::uint64_t cap, const char* message) {
    if (value > cap) throw std::length_error(message);
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::runtime_error("Gaussian one-electron numerical value is not finite");
    return x;
}
Complex finite(Complex z) { finite(z.real()); finite(z.imag()); return z; }
class Digest {
public:
    void bytes(const std::uint8_t* p, std::size_t n) {
        limit(n,kShaMax-extent_,"Gaussian one-electron SHA extent overflow");
        hash_.update(p,n); extent_ += n;
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
        std::uint64_t n=0; std::memcpy(&n,&x,8U); u64(n);
    }
    void text(const std::string& s) {
        u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
    std::uint64_t extent() const noexcept { return extent_; }
    std::array<char,64> finish() {
        const auto s=hash_.finish_hex();
        if (s.size()!=64) throw std::logic_error("Gaussian one-electron SHA result extent");
        std::array<char,64> result{}; std::copy(s.begin(),s.end(),result.begin()); return result;
    }
private:
    detail::Sha256 hash_;
    std::uint64_t extent_=0;
};

PeriodicGaussianSourceCaps clamped_basis_caps(const PeriodicGaussianSourceContext& context,
    const PeriodicGaussianSourceCaps& caller) {
    const auto& v=context.inventory(); const auto& a=v.ao; const auto& b=v.auxiliary;
    PeriodicGaussianSourceCaps exact;
    exact.maximum_context_storage_bytes=sizeof(PeriodicGaussianSourceContext);
    exact.maximum_kpoint_count=context.mesh().size();
    exact.maximum_shell_count=add(a.shell_count,b.shell_count);
    exact.maximum_contraction_count=add(a.contraction_count,b.contraction_count);
    exact.maximum_primitive_numeric_lanes=add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count));
    exact.maximum_basis_content_wire_bytes=add(a.content_wire_bytes,b.content_wire_bytes);
    exact.maximum_borrowed_active_numeric_bytes=v.combined_borrowed_active_numeric_bytes;
    exact.maximum_work_units=v.work_units_upper_bound;
#define ONE_ELECTRON_BASIS_CAP(field) \
    if (!caller.field) throw std::invalid_argument("Gaussian one-electron basis caps must be positive"); \
    limit(exact.field,caller.field,"Gaussian one-electron basis verification cap is insufficient")
    ONE_ELECTRON_BASIS_CAP(maximum_context_storage_bytes);
    ONE_ELECTRON_BASIS_CAP(maximum_kpoint_count);
    ONE_ELECTRON_BASIS_CAP(maximum_shell_count);
    ONE_ELECTRON_BASIS_CAP(maximum_contraction_count);
    ONE_ELECTRON_BASIS_CAP(maximum_primitive_numeric_lanes);
    ONE_ELECTRON_BASIS_CAP(maximum_basis_content_wire_bytes);
    ONE_ELECTRON_BASIS_CAP(maximum_borrowed_active_numeric_bytes);
    ONE_ELECTRON_BASIS_CAP(maximum_work_units);
#undef ONE_ELECTRON_BASIS_CAP
    return exact;
}
void options_valid(const Options& o, const Live& live, const Caps& caps) {
    for (double x : {o.structural_absolute_tolerance,o.structural_relative_tolerance})
        if (!std::isfinite(x) || x<0 || x>=1) throw std::invalid_argument("Gaussian one-electron structural tolerances must be finite in [0,1)");
    if (o.structural_absolute_tolerance==0 && o.structural_relative_tolerance==0)
        throw std::invalid_argument("Gaussian one-electron structural tolerances cannot both be zero");
    for (auto n : {live.replicas_per_node,live.fixed_backend_margin_bytes_per_replica,
        caps.maximum_owned_numeric_bytes,caps.maximum_per_replica_inventoried_bytes,
        caps.maximum_node_inventoried_bytes,caps.maximum_pair_count,caps.maximum_candidate_evaluations,caps.maximum_work_units})
        if (!n) throw std::invalid_argument("Gaussian one-electron caps, replicas and backend margin must be positive");
    if (std::fegetround()!=FE_TONEAREST) throw std::invalid_argument("Gaussian one-electron requires round-to-nearest arithmetic");
}
pair::LongMatrix geometry(const PeriodicGaussianSourceContext& context, const PeriodicSystem& system) {
    if (system.dim!=3 || !system.lattice.allFinite() || system.lattice!=context.direct_lattice())
        throw std::invalid_argument("Gaussian one-electron original cell does not match source context");
    const pair::LongMatrix a=system.lattice.cast<long double>();
    const long double determinant=a.determinant();
    if (!std::isfinite(determinant) || determinant==0) throw std::invalid_argument("Gaussian one-electron singular direct lattice");
    const pair::LongMatrix inverse=a.inverse();
    const long double condition=a.norm()*inverse.norm();
    if (!inverse.allFinite() || !std::isfinite(condition) || condition>1.0e10L)
        throw std::invalid_argument("Gaussian one-electron lattice exceeds numerical conditioning policy");
    return inverse;
}
void memory_limits(const Plan& p) {
    limit(p.owned_numeric_peak_bytes,p.caps.maximum_owned_numeric_bytes,"Gaussian one-electron owned numerical byte cap exceeded");
    limit(p.per_replica_inventoried_bytes,p.caps.maximum_per_replica_inventoried_bytes,"Gaussian one-electron per-replica byte cap exceeded");
    limit(p.node_inventoried_bytes,p.caps.maximum_node_inventoried_bytes,"Gaussian one-electron node byte cap exceeded");
    limit(p.work_units_upper_bound,p.caps.maximum_work_units,"Gaussian one-electron work cap exceeded");
}
void add_pair_work(Plan& p, const pair::Ao& a, const pair::Ao& b, std::uint64_t candidates) {
    p.candidate_evaluations=add(p.candidate_evaluations,candidates);
    limit(p.candidate_evaluations,p.caps.maximum_candidate_evaluations,"Gaussian one-electron image candidate cap exceeded");
    const auto primitives=mul(candidates,mul(a.shell->alpha.size(),b.shell->alpha.size()));
    const auto la=static_cast<std::uint64_t>(a.contraction->l), lb=static_cast<std::uint64_t>(b.contraction->l);
    const auto cells=mul(primitives,mul(3U,mul(mul(la+2U,lb+2U),la+lb+3U)));
    const auto cart=mul(primitives,mul(cart_to_sph_data::n_cart_for_l(static_cast<int>(la)),cart_to_sph_data::n_cart_for_l(static_cast<int>(lb))));
    p.primitive_pair_evaluations_upper_bound=add(p.primitive_pair_evaluations_upper_bound,primitives);
    p.md_table_cells_upper_bound=add(p.md_table_cells_upper_bound,cells);
    p.cartesian_pair_terms_upper_bound=add(p.cartesian_pair_terms_upper_bound,cart);
    p.work_units_upper_bound=add(p.work_units_upper_bound,
        add(mul(4096U,candidates),add(mul(512U,primitives),add(mul(32U,cells),mul(512U,cart)))));
    limit(p.work_units_upper_bound,p.caps.maximum_work_units,"Gaussian one-electron work cap exceeded before image enumeration");
}
struct Sum {
    double value=0, correction=0;
    void add(double x) {
        finite(x); const double next=finite(value+x);
        correction=finite(correction+(std::abs(value)>=std::abs(x) ? (value-next)+x : (x-next)+value));
        value=next;
    }
    double result() const { return finite(value+correction); }
};
struct ComplexSum {
    Sum re,im;
    void add(Complex z) { re.add(z.real()); im.add(z.imag()); }
    Complex result() const { return {re.result(),im.result()}; }
};

// Return S/T primitive polynomial multipliers. The extended MD table holds
// original polynomial powers and their FIRST derivatives on BOTH centers.
// The original pure-shell transform remains unchanged for every term.
std::array<double,2> polynomials(const pair::Ao& a, const pair::Ao& b,
    double alpha, double beta, const Workspace& w) {
    const auto la=a.contraction->l, lb=b.contraction->l;
    const auto na=cart_to_sph_data::n_cart_for_l(la), nb=cart_to_sph_data::n_cart_for_l(lb);
    const auto* ac=cart_to_sph_data::cart_table_for_l(la); const auto* bc=cart_to_sph_data::cart_table_for_l(lb);
    const auto* at=cart_to_sph_data::sph_table_for_l(la)+a.component*na;
    const auto* bt=cart_to_sph_data::sph_table_for_l(lb)+b.component*nb;
    const auto stride=static_cast<std::size_t>(lb+2), nt=static_cast<std::size_t>(la+lb+3);
    const auto e=[&](std::size_t axis,int i,int j) {
        if (i<0 || j<0) return 0.0;
        return w.md[axis*kAxisExtent+(static_cast<std::size_t>(i)*stride+static_cast<std::size_t>(j))*nt];
    };
    Sum overlap,kinetic;
    for (std::size_t ia=0;ia<na;++ia) {
        if (at[ia]==0) continue;
        const std::array<int,3> ap{ac[ia].i,ac[ia].j,ac[ia].k};
        for (std::size_t ib=0;ib<nb;++ib) {
            if (bt[ib]==0) continue;
            const std::array<int,3> bp{bc[ib].i,bc[ib].j,bc[ib].k};
            std::array<double,3> s{};
            for (std::size_t d=0;d<3;++d) s[d]=finite(e(d,ap[d],bp[d]));
            const double weight=at[ia]*bt[ib];
            overlap.add(weight*s[0]*s[1]*s[2]);
            for (std::size_t d=0;d<3;++d) {
                const int i=ap[d], j=bp[d];
                // McMurchie-Davidson Eq2.37; -1/2 Laplacian equals
                // +1/2 gradient dot gradient after integration by parts.
                const double derivative=finite(0.5*(
                    static_cast<double>(i*j)*e(d,i-1,j-1)
                    -2.0*static_cast<double>(i)*beta*e(d,i-1,j+1)
                    -2.0*alpha*static_cast<double>(j)*e(d,i+1,j-1)
                    +4.0*alpha*beta*e(d,i+1,j+1)));
                kinetic.add(weight*derivative*s[(d+1)%3]*s[(d+2)%3]);
            }
        }
    }
    return {overlap.result(),kinetic.result()};
}
std::array<Complex,2> evaluate(const pair::Ao& a, const pair::Ao& b,
    const PeriodicSystem& system, const pair::LongMatrix& inverse, double cutoff,
    const Eigen::Vector3d& k, std::uint64_t original_pair, std::uint32_t role,
    Workspace& workspace, Digest& images, Diagnostics& diagnostics) {
    const auto box=pair::image_box(a,b,inverse,cutoff);
    images.u64(original_pair); images.u32(role);
    for (auto v : box.lower) images.u64(static_cast<std::uint64_t>(v));
    for (auto v : box.upper) images.u64(static_cast<std::uint64_t>(v));
    images.u64(box.candidates);
    diagnostics.completed_candidate_evaluations=add(diagnostics.completed_candidate_evaluations,box.candidates);
    const int la=a.contraction->l,lb=b.contraction->l;
    const double scale_a=la==0 ? 1.0 : std::sqrt(4.0*kPi/(2*la+1));
    const double scale_b=lb==0 ? 1.0 : std::sqrt(4.0*kPi/(2*lb+1));
    std::array<ComplexSum,2> sums;
    const auto retained=pair::walk_images(a,b,system.lattice,box,cutoff*cutoff,
        [&](const std::array<std::int64_t,3>& label, const Eigen::Vector3d& translation,
            const Eigen::Vector3d& separation, double squared) {
            for (auto v : label) images.u64(static_cast<std::uint64_t>(v));
            const double kr=finite(k.dot(translation));
            const Complex phase{std::cos(kr),std::sin(kr)};
            for (std::size_t ia=0;ia<a.shell->alpha.size();++ia) {
                const double alpha=a.shell->alpha[ia];
                for (std::size_t ib=0;ib<b.shell->alpha.size();++ib) {
                    diagnostics.completed_primitive_pair_evaluations=add(diagnostics.completed_primitive_pair_evaluations,1U);
                    const double beta=b.shell->alpha[ib], gamma=finite(alpha+beta);
                    const double af=alpha/gamma,bf=beta/gamma;
                    for (std::size_t d=0;d<3;++d)
                        pair::md_coefficients(la+1,lb+1,gamma,-bf*separation[d],af*separation[d],workspace.md.data()+d*kAxisExtent);
                    const double radial=finite(a.contraction->coeff[ia]*b.contraction->coeff[ib]
                        *std::pow(kPi/gamma,1.5)*std::exp(-af*beta*squared));
                    const auto polynomial=polynomials(a,b,alpha,beta,workspace);
                    for (std::size_t op=0;op<2;++op)
                        sums[op].add(finite((scale_a*scale_b*radial)*phase*polynomial[op]));
                }
            }
        });
    images.u64(retained);
    if (role==0) diagnostics.retained_primary_pair_images=add(diagnostics.retained_primary_pair_images,retained);
    else diagnostics.retained_audit_pair_images=add(diagnostics.retained_audit_pair_images,retained);
    return {sums[0].result(),sums[1].result()};
}
double audit(Complex a, Complex b, const Options& options, const char* message) {
    const double error=finite(std::abs(finite(a-b)));
    const double scale=std::max(finite(std::abs(a)),finite(std::abs(b)));
    const double tolerance=finite(options.structural_absolute_tolerance+options.structural_relative_tolerance*scale);
    if (error>tolerance) throw std::runtime_error(message);
    return error;
}
} // namespace

Plan plan_periodic_gaussian_one_electron_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext> context, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicSystem& system, std::uint64_t k, std::uint64_t begin, std::uint64_t count,
    const Options& options, const Live& live, const Caps& caps) {
    if (!context || context->contract_version()!=kPeriodicGaussianSourceContextVersion)
        throw std::invalid_argument("Gaussian one-electron requires a live native source context");
    options_valid(options,live,caps);
    const auto basis_caps=clamped_basis_caps(*context,options.basis_verification_caps);
    Plan p; p.options=options; p.live=live; p.caps=caps;
    p.n_basis=context->inventory().ao.function_count;
    const auto square=mul(p.n_basis,p.n_basis);
    if (!count || begin>square || count>square-begin || k>=context->mesh().size())
        throw std::invalid_argument("Gaussian one-electron k or nonempty pair interval is invalid");
    limit(count,caps.maximum_pair_count,"Gaussian one-electron pair count exceeds cap");
    p.k_index=k; p.opposite_k_index=context->mesh().negate_index(k); p.pair_begin=begin; p.pair_count=count;
    p.output_numeric_bytes=mul(32U,count);
    limit(mul(2U,count),std::vector<Complex>().max_size(),"Gaussian one-electron output vector extent exceeded");
    limit(add(p.output_numeric_bytes,4096U),kShaMax,"Gaussian one-electron payload SHA extent exceeded");
    p.fixed_numeric_workspace_bytes=sizeof(Workspace);
    p.owned_numeric_peak_bytes=add(p.output_numeric_bytes,p.fixed_numeric_workspace_bytes);
    p.borrowed_basis_active_numeric_bytes=context->inventory().combined_borrowed_active_numeric_bytes;
    p.fixed_inventoried_object_bytes=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianOneElectronPanel)
        +sizeof(Plan)+3U*sizeof(Digest)+sizeof(pair::LongMatrix);
    p.per_replica_inventoried_bytes=add(add(p.owned_numeric_peak_bytes,p.borrowed_basis_active_numeric_bytes),
        add(p.fixed_inventoried_object_bytes,add(live.other_retained_bytes_per_replica,
            add(live.other_transient_bytes_per_replica,live.fixed_backend_margin_bytes_per_replica))));
    p.node_inventoried_bytes=add(live.external_node_bytes,mul(live.replicas_per_node,p.per_replica_inventoried_bytes));
    p.basis_scan_work_units=context->inventory().work_units_upper_bound;
    p.preflight_work_units=add(p.basis_scan_work_units,add(mul(256U,mul(count,
        add(context->inventory().ao.shell_count,context->inventory().ao.contraction_count))),4096U));
    p.work_units_upper_bound=add(p.preflight_work_units,mul(128U,p.output_numeric_bytes));
    memory_limits(p); // precedes any full borrowed-basis value/hash scan
    context->verify_bases(ao,auxiliary,basis_caps);
    const auto inverse=geometry(*context,system);
    const double cutoff=context->options().ao_pair_image_cutoff_bohr;
    for (std::uint64_t row=0;row<count;++row) {
        const auto a=pair::ao(ao,(begin+row)/p.n_basis), b=pair::ao(ao,(begin+row)%p.n_basis);
        const auto forward=pair::image_box(a,b,inverse,cutoff);
        const auto reverse=pair::image_box(b,a,inverse,cutoff);
        add_pair_work(p,a,b,mul(2U,forward.candidates));
        add_pair_work(p,b,a,reverse.candidates);
    }
    p.image_identity_wire_bytes_upper_bound=add(kImagePrefix,add(mul(228U,count),mul(24U,p.candidate_evaluations)));
    limit(p.image_identity_wire_bytes_upper_bound,kShaMax,"Gaussian one-electron image SHA extent exceeded");
    memory_limits(p);
    return p;
}

PeriodicGaussianOneElectronPanel build_periodic_gaussian_one_electron_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext> context, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicSystem& system, std::uint64_t k, std::uint64_t begin, std::uint64_t count,
    const Options& options, const Live& live, const Caps& caps) {
    auto p=plan_periodic_gaussian_one_electron_panel(context,ao,auxiliary,system,k,begin,count,options,live,caps);
    const auto inverse=geometry(*context,system);
    const auto point=context->k_record(k),opposite=context->k_record(p.opposite_k_index);
    const Eigen::Vector3d kv(point.cartesian[0],point.cartesian[1],point.cartesian[2]);
    const Eigen::Vector3d kb(opposite.cartesian[0],opposite.cartesian[1],opposite.cartesian[2]);
    PeriodicGaussianOneElectronPanel result;
    result.context_=std::move(context); result.plan_=p;
    result.values_.resize(static_cast<std::size_t>(2U*count));
    Workspace workspace;
    Digest images; images.text(kImageDomain); images.u32(kPeriodicGaussianOneElectronVersion);
    for (auto n : {p.n_basis,p.k_index,p.opposite_k_index,p.pair_begin,p.pair_count}) images.u64(n);
    if (images.extent()!=kImagePrefix) throw std::logic_error("Gaussian one-electron image wire prefix changed");
    auto& d=result.diagnostics_;
    const double cutoff=result.context_->options().ao_pair_image_cutoff_bohr;
    for (std::uint64_t row=0;row<count;++row) {
        const auto mu=(begin+row)/p.n_basis,nu=(begin+row)%p.n_basis;
        const auto a=pair::ao(ao,mu),b=pair::ao(ao,nu);
        const auto original=evaluate(a,b,system,inverse,cutoff,kv,begin+row,0,workspace,images,d);
        const auto reverse=evaluate(b,a,system,inverse,cutoff,kv,begin+row,1,workspace,images,d);
        const auto partner=evaluate(a,b,system,inverse,cutoff,kb,begin+row,2,workspace,images,d);
        for (std::size_t op=0;op<2;++op) {
            const auto h=audit(original[op],std::conj(reverse[op]),options,"Gaussian one-electron raw Hermiticity audit failed");
            const auto tr=audit(original[op],std::conj(partner[op]),options,"Gaussian one-electron raw time-reversal audit failed");
            if (op==0) { d.maximum_overlap_hermitian_error=std::max(d.maximum_overlap_hermitian_error,h); d.maximum_overlap_time_reversal_error=std::max(d.maximum_overlap_time_reversal_error,tr); }
            else { d.maximum_kinetic_hermitian_error=std::max(d.maximum_kinetic_hermitian_error,h); d.maximum_kinetic_time_reversal_error=std::max(d.maximum_kinetic_time_reversal_error,tr); }
            if (mu==nu) d.maximum_diagonal_imaginary_magnitude=std::max(d.maximum_diagonal_imaginary_magnitude,
                audit(original[op],Complex(original[op].real(),0),options,"Gaussian one-electron diagonal imaginary audit failed"));
            if (k==p.opposite_k_index) d.maximum_trim_imaginary_magnitude=std::max(d.maximum_trim_imaginary_magnitude,
                audit(original[op],Complex(original[op].real(),0),options,"Gaussian one-electron TRIM imaginary audit failed"));
            result.values_[static_cast<std::size_t>(op*count+row)]=original[op];
        }
    }
    d.image_identity_wire_bytes=images.extent();
    const auto retained=add(d.retained_primary_pair_images,d.retained_audit_pair_images);
    const auto exact_wire=add(kImagePrefix,add(mul(228U,count),mul(24U,retained)));
    if (d.completed_candidate_evaluations!=p.candidate_evaluations
        || d.completed_primitive_pair_evaluations>p.primitive_pair_evaluations_upper_bound
        || d.image_identity_wire_bytes!=exact_wire || exact_wire>p.image_identity_wire_bytes_upper_bound)
        throw std::logic_error("Gaussian one-electron traversal differs from admitted census");
    result.images_=images.finish();
    Digest source; source.text("vibeqc.periodic.gaussian-one-electron.source"); source.u32(kPeriodicGaussianOneElectronVersion);
    source.text(kPeriodicGaussianOneElectronPolicy); source.text(kPeriodicCorrelationThreeCenterImagePolicy);
    source.text(result.context_->source_context_identity_sha256()); source.text(result.context_->ao_basis_identity_sha256());
    source.text(result.image_source_identity_sha256());
    for (auto n : {k,p.opposite_k_index,begin,count}) source.u64(n);
    source.real(cutoff); result.source_=source.finish();
    Digest payload; payload.text("vibeqc.periodic.gaussian-one-electron.payload"); payload.u32(kPeriodicGaussianOneElectronVersion);
    payload.text(result.operator_source_identity_sha256());
    payload.real(options.structural_absolute_tolerance); payload.real(options.structural_relative_tolerance);
    payload.u64(result.values_.size());
    for (auto z : result.values_) { payload.real(z.real()); payload.real(z.imag()); }
    result.payload_=payload.finish();
    return result;
}
Complex PeriodicGaussianOneElectronPanel::element(std::uint32_t op,std::uint64_t pair_index) const {
    if (!context_ || op>1 || pair_index>=plan_.pair_count || values_.size()!=2U*plan_.pair_count)
        throw std::out_of_range("Gaussian one-electron panel element is consumed or out of range");
    return values_[static_cast<std::size_t>(op*plan_.pair_count+pair_index)];
}
std::string PeriodicGaussianOneElectronPanel::image_source_identity_sha256() const { return {images_.begin(),images_.end()}; }
std::string PeriodicGaussianOneElectronPanel::operator_source_identity_sha256() const { return {source_.begin(),source_.end()}; }
std::string PeriodicGaussianOneElectronPanel::payload_identity_sha256() const { return {payload_.begin(),payload_.end()}; }
} // namespace vibeqc
