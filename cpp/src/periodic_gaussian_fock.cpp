#include "vibeqc/periodic_gaussian_fock.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {
using Complex = std::complex<double>;
static_assert(sizeof(double) == 8 && sizeof(Complex) == 16, "Gaussian Fock requires binary64/complex128");
constexpr std::uint64_t kShaMax = std::numeric_limits<std::uint64_t>::max() / 8U;
constexpr char kDensityDomain[] = "vibeqc.periodic.gaussian-fock.density";
constexpr char kConsumedDomain[] = "vibeqc.periodic.gaussian-fock.consumed-factors";
constexpr char kPayloadDomain[] = "vibeqc.periodic.gaussian-fock.payload";
constexpr std::uint64_t kDensityPrefix = 116U + sizeof(kDensityDomain) - 1U;
constexpr std::uint64_t kConsumedPrefix = 188U + sizeof(kConsumedDomain) - 1U;
constexpr std::uint64_t kPayloadPrefix = 356U + sizeof(kPayloadDomain) - 1U;
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("Gaussian Fock count overflow");
    return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("Gaussian Fock count overflow");
    return a*b;
}
std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) { return a/b + (a%b != 0); }
void limit(std::uint64_t n, std::uint64_t cap, const char* message) {
    if (n > cap) throw std::length_error(message);
}
void finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("Gaussian Fock numerical value is nonfinite");
}
void finite(Complex z) { finite(z.real()); finite(z.imag()); }
void accumulate(double x, double& s, double& c) {
    finite(x);
    const double next = s+x;
    c += std::abs(s) >= std::abs(x) ? (s-next)+x : (x-next)+s;
    s = next; finite(s); finite(c);
}
void accumulate(Complex x, Complex& s, Complex& c) {
    double sr=s.real(), si=s.imag(), cr=c.real(), ci=c.imag();
    accumulate(x.real(), sr, cr); accumulate(x.imag(), si, ci);
    s={sr,si}; c={cr,ci};
}
Complex completed(Complex s, Complex c) {
    s += c; finite(s);
    return {s.real() == 0.0 ? 0.0 : s.real(), s.imag() == 0.0 ? 0.0 : s.imag()};
}
std::array<char,64> ascii(const std::string& s) {
    if (s.size()!=64 || !std::all_of(s.begin(),s.end(),[](char c) {
        return (c>='0' && c<='9') || (c>='a' && c<='f');
    })) throw std::invalid_argument("Gaussian Fock requires native lowercase SHA identities");
    std::array<char,64> r{}; std::copy(s.begin(),s.end(),r.begin()); return r;
}
class Digest {
public:
    void bytes(const std::uint8_t* p,std::size_t n) {
        if (n>kShaMax-extent_) throw std::length_error("Gaussian Fock SHA extent overflow");
        h_.update(p,n); extent_+=n;
    }
    void u32(std::uint32_t n) {
        std::array<std::uint8_t,4> b{};
        for(unsigned i=0;i!=4;++i) b[i]=n>>(24U-8U*i);
        bytes(b.data(),b.size());
    }
    void u64(std::uint64_t n) {
        std::array<std::uint8_t,8> b{};
        for(unsigned i=0;i!=8;++i) b[i]=n>>(56U-8U*i);
        bytes(b.data(),b.size());
    }
    void real(double x) {
        finite(x); if(x==0.0) x=0.0;
        std::uint64_t n; std::memcpy(&n,&x,8); u64(n);
    }
    void text(const std::string& s) { u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    std::uint64_t extent() const noexcept { return extent_; }
    std::array<char,64> finish() { return ascii(h_.finish_hex()); }
private:
    detail::Sha256 h_; std::uint64_t extent_=0;
};
void resource_wire(Digest& h,const PeriodicGaussianMetricCaps& c) {
    for(auto n:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) h.u64(n);
}
void require_resources(const PeriodicGaussianMetricCaps& c) {
    if (!c.maximum_owned_numeric_bytes || !c.maximum_per_replica_inventoried_bytes
        || !c.maximum_node_inventoried_bytes || !c.maximum_candidate_evaluations || !c.maximum_work_units) {
        throw std::invalid_argument("Gaussian Fock resource caps must be positive");
    }
}
void basis_wire(Digest& h,const PeriodicGaussianSourceCaps& c) {
    for(auto n:{c.maximum_context_storage_bytes,c.maximum_kpoint_count,c.maximum_shell_count,
        c.maximum_contraction_count,c.maximum_primitive_numeric_lanes,c.maximum_basis_content_wire_bytes,
        c.maximum_borrowed_active_numeric_bytes,c.maximum_work_units}) h.u64(n);
}
PeriodicGaussianSourceCaps bounded_basis_caps(const PeriodicGaussianSourceContext& c,
                                             const PeriodicGaussianSourceCaps& supplied) {
    const auto& a=c.inventory().ao; const auto& b=c.inventory().auxiliary;
    PeriodicGaussianSourceCaps r;
    r.maximum_context_storage_bytes=sizeof(PeriodicGaussianSourceContext);
    r.maximum_kpoint_count=c.mesh().size(); r.maximum_shell_count=add(a.shell_count,b.shell_count);
    r.maximum_contraction_count=add(a.contraction_count,b.contraction_count);
    r.maximum_primitive_numeric_lanes=add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count));
    r.maximum_basis_content_wire_bytes=add(a.content_wire_bytes,b.content_wire_bytes);
    r.maximum_borrowed_active_numeric_bytes=c.inventory().combined_borrowed_active_numeric_bytes;
    r.maximum_work_units=c.inventory().work_units_upper_bound;
    const std::array<std::uint64_t,8> expected{{r.maximum_context_storage_bytes,r.maximum_kpoint_count,
        r.maximum_shell_count,r.maximum_contraction_count,r.maximum_primitive_numeric_lanes,
        r.maximum_basis_content_wire_bytes,r.maximum_borrowed_active_numeric_bytes,r.maximum_work_units}};
    const std::array<std::uint64_t,8> caps{{supplied.maximum_context_storage_bytes,supplied.maximum_kpoint_count,
        supplied.maximum_shell_count,supplied.maximum_contraction_count,supplied.maximum_primitive_numeric_lanes,
        supplied.maximum_basis_content_wire_bytes,supplied.maximum_borrowed_active_numeric_bytes,supplied.maximum_work_units}};
    for(std::size_t i=0;i!=8;++i) {
        if(!caps[i]) throw std::invalid_argument("Gaussian Fock basis caps must be positive");
        limit(expected[i],caps[i],"Gaussian Fock basis census exceeds cap");
    }
    return r;
}
void plan_wire(Digest& h,const PeriodicGaussianFockPlan& p) {
    for(auto n:{p.n_kpoints,p.n_basis,p.n_auxiliary,p.auxiliary_block,p.ao_column_block,
        p.auxiliary_block_count,p.ao_column_block_count,p.density_element_count,p.borrowed_density_bytes,
        p.response_bytes,p.response_compensation_bytes,p.hartree_vector_bytes,p.exchange_double_panel_bytes,
        p.resident_whitener_bytes,p.metric_phase_owned_numeric_upper_bound,p.tile_phase_owned_numeric_upper_bound,
        p.owned_numeric_upper_bound,p.borrowed_basis_active_numeric_bytes,p.macro_fixed_object_bytes,
        p.maximum_leaf_fixed_inventory_bytes,p.per_replica_inventoried_bytes,p.node_inventoried_bytes,
        p.hartree_tile_calls,p.exchange_tile_calls,p.total_tile_calls,p.progress_callback_upper_bound,
        p.driver_contraction_terms,p.reciprocal_candidate_evaluations_upper_bound,
        p.image_candidate_evaluations_upper_bound,p.driver_work_units_upper_bound,p.work_units_upper_bound}) h.u64(n);
    const auto& c=p.config;
    h.u64(c.auxiliary_block); h.u64(c.ao_column_block);
    for(auto n:{c.source_caps.maximum_fixed_storage_bytes,c.source_caps.maximum_candidates_per_source,
        c.source_caps.maximum_candidate_evaluations,c.source_caps.maximum_source_wire_bytes}) h.u64(n);
    h.u64(c.metric.reciprocal_block); h.u64(c.metric.whitener_column_block);
    basis_wire(h,c.metric.basis_verification_caps); resource_wire(h,c.metric_caps);
    h.u64(c.tile.reciprocal_block); basis_wire(h,c.tile.basis_verification_caps);
    h.u64(c.tile_caps.maximum_image_candidates); resource_wire(h,c.tile_caps.resources);
    for(auto n:{p.live.replicas_per_node,p.live.other_retained_bytes_per_replica,p.live.other_transient_bytes_per_replica,
        p.live.fixed_backend_margin_bytes_per_replica,p.live.external_node_bytes}) h.u64(n);
    resource_wire(h,p.caps.resources);
    h.u64(p.caps.maximum_tile_calls); h.u64(p.caps.maximum_progress_callbacks);
    h.u64(p.caps.maximum_image_candidate_evaluations);
}
std::uint64_t source_work(std::uint64_t candidate_evaluations) { return add(mul(2048U,candidate_evaluations),65536U); }
std::array<char,64> density_hash(const PeriodicGaussianSourceContext& c,PeriodicGaussianFockDensityView d) {
    Digest h; h.text(kDensityDomain); h.u32(1U);
    h.text(c.source_context_identity_sha256()); h.u64(c.mesh().size()); h.u64(c.inventory().ao.function_count);
    h.u64(d.element_count); h.u64(mul(16U,d.element_count));
    if(h.extent()!=kDensityPrefix) throw std::logic_error("Gaussian Fock density wire prefix changed");
    for(std::uint64_t i=0;i<d.element_count;++i) { h.real(d.data[i].real()); h.real(d.data[i].imag()); }
    return h.finish();
}
PeriodicGaussianFockMatrixDiagnostics matrix_diagnostics(const PeriodicGaussianSourceContext& c,
    const Complex* data,bool require_exact_hermitian) {
    PeriodicGaussianFockMatrixDiagnostics d;
    const auto n=c.inventory().ao.function_count, nk=static_cast<std::uint64_t>(c.mesh().size());
    for(std::uint64_t k=0;k<nk;++k) for(std::uint64_t i=0;i<n;++i) for(std::uint64_t j=0;j<n;++j) {
        const auto x=data[(k*n+i)*n+j]; finite(x);
        const auto y=std::conj(data[(k*n+j)*n+i]);
        if(require_exact_hermitian && x!=y) throw std::invalid_argument("Gaussian Fock density must be exactly Hermitian");
        const auto opposite=std::conj(data[(c.mesh().negate_index(k)*n+i)*n+j]);
        const double scale=std::abs(x), herm=std::abs(x-y), tr=std::abs(x-opposite);
        finite(scale); finite(herm); finite(tr);
        d.maximum_magnitude=std::max(d.maximum_magnitude,scale);
        d.maximum_hermiticity_residual=std::max(d.maximum_hermiticity_residual,herm);
        d.maximum_time_reversal_residual=std::max(d.maximum_time_reversal_residual,tr);
        if(i==j) d.maximum_diagonal_imaginary=std::max(d.maximum_diagonal_imaginary,std::abs(x.imag()));
    }
    if(d.maximum_magnitude!=0.0) {
        d.relative_hermiticity_residual=d.maximum_hermiticity_residual/d.maximum_magnitude;
        d.relative_time_reversal_residual=d.maximum_time_reversal_residual/d.maximum_magnitude;
    }
    return d;
}
void diagnostics_wire(Digest& h,const PeriodicGaussianFockMatrixDiagnostics& d) {
    for(double x:{d.maximum_magnitude,d.maximum_hermiticity_residual,d.relative_hermiticity_residual,
        d.maximum_time_reversal_residual,d.relative_time_reversal_residual,d.maximum_diagonal_imaginary}) h.real(x);
}
} // namespace

PeriodicGaussianFockPlan plan_periodic_gaussian_fock(const PeriodicGaussianSourceContext& context,
    const PeriodicGaussianFockConfig& config,const PeriodicGaussianMetricLiveInventory& live,
    const PeriodicGaussianFockCaps& caps) {
    require_resources(caps.resources); require_resources(config.metric_caps); require_resources(config.tile_caps.resources);
    if(!caps.maximum_tile_calls || !caps.maximum_progress_callbacks || !caps.maximum_image_candidate_evaluations
        || !live.replicas_per_node || !live.fixed_backend_margin_bytes_per_replica
        || !config.source_caps.maximum_fixed_storage_bytes || !config.source_caps.maximum_candidates_per_source
        || !config.source_caps.maximum_candidate_evaluations || !config.source_caps.maximum_source_wire_bytes
        || !config.tile_caps.maximum_image_candidates) throw std::invalid_argument("Gaussian Fock controls and caps must be positive");
    bounded_basis_caps(context,config.metric.basis_verification_caps);
    bounded_basis_caps(context,config.tile.basis_verification_caps);
    const auto n=context.inventory().ao.function_count, a=context.inventory().auxiliary.function_count;
    const auto nk=static_cast<std::uint64_t>(context.mesh().size());
    if(!n || !a || !nk || !config.auxiliary_block || config.auxiliary_block>a
        || !config.ao_column_block || config.ao_column_block>n || !config.metric.reciprocal_block
        || !config.metric.whitener_column_block || config.metric.whitener_column_block>a
        || !config.tile.reciprocal_block) throw std::invalid_argument("Gaussian Fock shape or blocks are invalid");
    PeriodicGaussianFockPlan p; p.config=config; p.live=live; p.caps=caps;
    p.n_kpoints=nk; p.n_basis=n; p.n_auxiliary=a;
    p.auxiliary_block=config.auxiliary_block; p.ao_column_block=config.ao_column_block;
    const auto u=ceil_div(a,p.auxiliary_block), l=ceil_div(n,p.ao_column_block), n2=mul(n,n), nk2=mul(nk,nk);
    p.auxiliary_block_count=u; p.ao_column_block_count=l; p.density_element_count=mul(nk,n2);
    p.borrowed_density_bytes=mul(16U,p.density_element_count);
    p.response_bytes=p.borrowed_density_bytes; p.response_compensation_bytes=p.response_bytes;
    p.hartree_vector_bytes=mul(32U,p.auxiliary_block);
    p.exchange_double_panel_bytes=mul(32U,mul(p.auxiliary_block,mul(n,p.ao_column_block)));
    p.resident_whitener_bytes=mul(16U,mul(a,a));
    const auto output=add(p.response_bytes,p.response_compensation_bytes);
    p.metric_phase_owned_numeric_upper_bound=add(output,config.metric_caps.maximum_owned_numeric_bytes);
    p.tile_phase_owned_numeric_upper_bound=add(add(output,p.resident_whitener_bytes),
        add(std::max(p.hartree_vector_bytes,p.exchange_double_panel_bytes),config.tile_caps.resources.maximum_owned_numeric_bytes));
    p.owned_numeric_upper_bound=std::max(p.metric_phase_owned_numeric_upper_bound,p.tile_phase_owned_numeric_upper_bound);
    p.borrowed_basis_active_numeric_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    p.macro_fixed_object_bytes=sizeof(PeriodicGaussianFockPlan)+sizeof(PeriodicGaussianFockResult)
        +sizeof(PeriodicGaussianFockReceipt)+sizeof(PeriodicGaussianFockProgress)+sizeof(PeriodicGaussianFockConfig)
        +sizeof(PeriodicGaussianFockCaps)+sizeof(PeriodicGaussianMetricLiveInventory)+sizeof(PeriodicGaussianFockDensityView);
    const std::uint64_t metric_fixed=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianReciprocalMetric)+sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianMetricPlan);
    const std::uint64_t tile_fixed=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianThreeCenterPlan)
        +sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem);
    p.maximum_leaf_fixed_inventory_bytes=std::max({metric_fixed,tile_fixed,config.source_caps.maximum_fixed_storage_bytes});
    p.per_replica_inventoried_bytes=add(add(p.owned_numeric_upper_bound,p.borrowed_density_bytes),
        add(p.borrowed_basis_active_numeric_bytes,add(p.macro_fixed_object_bytes,add(p.maximum_leaf_fixed_inventory_bytes,
            add(live.other_retained_bytes_per_replica,add(live.other_transient_bytes_per_replica,live.fixed_backend_margin_bytes_per_replica))))));
    p.node_inventoried_bytes=add(live.external_node_bytes,mul(live.replicas_per_node,p.per_replica_inventoried_bytes));
    p.hartree_tile_calls=mul(2U,mul(u,mul(nk,mul(n,l))));
    p.exchange_tile_calls=mul(nk2,mul(u,mul(l,mul(n,add(1U,l)))));
    p.total_tile_calls=add(p.hartree_tile_calls,p.exchange_tile_calls);
    p.progress_callback_upper_bound=add(add(2U,mul(4U,nk)),add(mul(2U,mul(u,nk)),mul(nk2,mul(u,l))));
    p.driver_contraction_terms=add(mul(2U,mul(nk,mul(a,n2))),mul(2U,mul(nk2,mul(a,mul(n2,n)))));
    p.reciprocal_candidate_evaluations_upper_bound=add(mul(nk,config.source_caps.maximum_candidate_evaluations),
        add(mul(nk,config.metric_caps.maximum_candidate_evaluations),mul(p.total_tile_calls,config.tile_caps.resources.maximum_candidate_evaluations)));
    const auto max_n=config.source_caps.maximum_candidates_per_source;
    p.image_candidate_evaluations_upper_bound=mul(p.total_tile_calls,mul(config.tile_caps.maximum_image_candidates,
        add(add(1U,max_n),ceil_div(max_n,config.tile.reciprocal_block))));
    p.driver_work_units_upper_bound=context.inventory().work_units_upper_bound;
    for(auto term:{mul(512U,p.density_element_count),mul(128U,p.driver_contraction_terms),
        mul(256U,mul(p.progress_callback_upper_bound,p.density_element_count)),
        mul(64U,mul(p.total_tile_calls,mul(p.auxiliary_block,p.ao_column_block))),
        mul(4096U,add(add(p.total_tile_calls,mul(3U,nk)),p.progress_callback_upper_bound)),std::uint64_t(4096U)}) {
        p.driver_work_units_upper_bound=add(p.driver_work_units_upper_bound,term);
    }
    p.work_units_upper_bound=add(p.driver_work_units_upper_bound,add(mul(nk,source_work(config.source_caps.maximum_candidate_evaluations)),
        add(mul(mul(2U,nk),config.metric_caps.maximum_work_units),mul(p.total_tile_calls,config.tile_caps.resources.maximum_work_units))));
    limit(p.density_element_count,std::vector<Complex>().max_size(),"Gaussian Fock output vector exceeds cap");
    limit(p.exchange_double_panel_bytes/32U,std::vector<Complex>().max_size(),"Gaussian Fock exchange vector exceeds cap");
    // Full consumption trace has296 bytes/q and80 bytes/tile, no hash table.
    limit(p.density_element_count,(kShaMax-kDensityPrefix)/16U,"Gaussian Fock density SHA extent exceeds cap");
    limit(p.density_element_count,(kShaMax-kPayloadPrefix)/16U,"Gaussian Fock payload SHA extent exceeds cap");
    limit(add(mul(296U,nk),mul(80U,p.total_tile_calls)),kShaMax-kConsumedPrefix,"Gaussian Fock factor trace SHA extent exceeds cap");
    limit(p.owned_numeric_upper_bound,caps.resources.maximum_owned_numeric_bytes,"Gaussian Fock owned numerical memory exceeds cap");
    limit(p.per_replica_inventoried_bytes,caps.resources.maximum_per_replica_inventoried_bytes,"Gaussian Fock per-replica inventory exceeds cap");
    limit(p.node_inventoried_bytes,caps.resources.maximum_node_inventoried_bytes,"Gaussian Fock node inventory exceeds cap");
    limit(p.total_tile_calls,caps.maximum_tile_calls,"Gaussian Fock tile calls exceed cap");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian Fock progress calls exceed cap");
    limit(p.reciprocal_candidate_evaluations_upper_bound,caps.resources.maximum_candidate_evaluations,"Gaussian Fock cumulative reciprocal candidates exceed cap");
    limit(p.image_candidate_evaluations_upper_bound,caps.maximum_image_candidate_evaluations,"Gaussian Fock cumulative image candidates exceed cap");
    limit(p.work_units_upper_bound,caps.resources.maximum_work_units,"Gaussian Fock cumulative work exceeds cap");
    Digest h; h.text("vibeqc.periodic.gaussian-fock.plan"); h.u32(1U); h.text(context.source_context_identity_sha256());
    plan_wire(h,p); p.identity_ascii=h.finish(); return p;
}

PeriodicGaussianFockResult build_periodic_gaussian_fock(
    std::shared_ptr<const PeriodicGaussianSourceContext> context,const BasisSet& ao,const BasisSet& auxiliary,
    PeriodicGaussianFockDensityView density,PeriodicGaussianFockConfig config,
    PeriodicGaussianMetricLiveInventory live,PeriodicGaussianFockCaps caps,
    void (*progress)(const PeriodicGaussianFockProgress&,void*),void* progress_context) {
    if(!context) throw std::invalid_argument("Gaussian Fock requires a native source context");
    const auto plan=plan_periodic_gaussian_fock(*context,config,live,caps);
    if(!density.data || density.element_count!=plan.density_element_count) throw std::invalid_argument("Gaussian Fock density view shape mismatch");
    context->verify_bases(ao,auxiliary,bounded_basis_caps(*context,config.metric.basis_verification_caps));
    PeriodicGaussianFockResult result; result.context_=context; result.plan_=plan;
    result.density_diagnostics_=matrix_diagnostics(*context,density.data,true);
    result.density_=density_hash(*context,density);
    result.values_.resize(static_cast<std::size_t>(plan.density_element_count));
    std::vector<Complex> correction(static_cast<std::size_t>(plan.density_element_count));
    auto& receipt=result.receipt_; receipt.charged_work_units_upper_bound=plan.driver_work_units_upper_bound;
    const auto nk=plan.n_kpoints,n=plan.n_basis,a=plan.n_auxiliary,ab=plan.auxiliary_block,b=plan.ao_column_block;
    const auto retained_output=add(plan.response_bytes,plan.response_compensation_bytes);
    const double inverse_nk=1.0/static_cast<double>(nk);
    Digest consumed; consumed.text(kConsumedDomain); consumed.u32(1U);
    consumed.text(context->source_context_identity_sha256()); consumed.text(result.density_identity_sha256());
    consumed.u64(nk); consumed.u64(n); consumed.u64(a); consumed.u64(plan.total_tile_calls);
    if(consumed.extent()!=kConsumedPrefix) throw std::logic_error("Gaussian Fock factor trace prefix changed");
    const auto notify=[&](PeriodicGaussianFockStage stage,std::uint64_t q=0,std::uint64_t aux=0,
                          std::uint64_t bra=0,std::uint64_t ket=0,std::uint64_t sigma=0) {
        if(!progress) return;
        limit(add(receipt.progress_callback_count,1U),caps.maximum_progress_callbacks,"Gaussian Fock progress count exceeds cap");
        PeriodicGaussianFockProgress event{stage,q,aux,bra,ket,sigma,receipt.completed_q_count,
            receipt.completed_tile_count,receipt.charged_work_units_upper_bound};
        progress(event,progress_context);
        if(density_hash(*context,density)!=result.density_) throw std::invalid_argument("Gaussian Fock density changed during progress callback");
        ++receipt.progress_callback_count;
    };
    const auto reserve=[&](std::uint64_t work,std::uint64_t reciprocal,std::uint64_t images=0) {
        limit(add(receipt.charged_work_units_upper_bound,work),caps.resources.maximum_work_units,"Gaussian Fock remaining work cap");
        limit(add(receipt.reciprocal_candidate_evaluations,reciprocal),caps.resources.maximum_candidate_evaluations,"Gaussian Fock remaining reciprocal cap");
        limit(add(receipt.image_candidate_evaluations,images),caps.maximum_image_candidate_evaluations,"Gaussian Fock remaining image cap");
    };
    const auto nested_live=[&](std::uint64_t scratch) {
        auto current=live;
        current.other_retained_bytes_per_replica=add(current.other_retained_bytes_per_replica,
            add(plan.borrowed_density_bytes,add(retained_output,add(scratch,plan.macro_fixed_object_bytes))));
        return current;
    };
    const auto observe=[&](std::uint64_t owned,std::uint64_t per_replica) {
        if(owned>plan.owned_numeric_upper_bound || per_replica>plan.per_replica_inventoried_bytes) {
            throw std::logic_error("Gaussian Fock nested lifetime exceeded conservative macro admission");
        }
        receipt.maximum_observed_owned_numeric_bytes=std::max(receipt.maximum_observed_owned_numeric_bytes,owned);
        receipt.maximum_observed_per_replica_inventoried_bytes=std::max(receipt.maximum_observed_per_replica_inventoried_bytes,per_replica);
    };
    const auto add_output=[&](std::uint64_t k,std::uint64_t i,std::uint64_t j,Complex value) {
        const auto index=(k*n+i)*n+j; accumulate(value,result.values_[index],correction[index]);
    };
    notify(PeriodicGaussianFockStage::Begin);
    for(std::uint64_t q=0;q<nk;++q) {
        notify(PeriodicGaussianFockStage::Source,q);
        reserve(source_work(config.source_caps.maximum_candidate_evaluations),config.source_caps.maximum_candidate_evaluations);
        const auto source=make_periodic_gaussian_reciprocal_source(context,q,config.source_caps);
        observe(retained_output,add(add(retained_output,plan.borrowed_density_bytes),
            add(plan.borrowed_basis_active_numeric_bytes,add(plan.macro_fixed_object_bytes,
                add(source.inventory().inventoried_fixed_storage_bytes,add(live.other_retained_bytes_per_replica,
                    add(live.other_transient_bytes_per_replica,live.fixed_backend_margin_bytes_per_replica)))))));
        const auto source_evaluations=source.inventory().candidate_evaluations_performed;
        receipt.source_factory_candidate_evaluations=add(receipt.source_factory_candidate_evaluations,source_evaluations);
        receipt.reciprocal_candidate_evaluations=add(receipt.reciprocal_candidate_evaluations,source_evaluations);
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,source_work(source_evaluations));
        notify(PeriodicGaussianFockStage::Metric,q);
        reserve(config.metric_caps.maximum_work_units,config.metric_caps.maximum_candidate_evaluations);
        auto raw=build_periodic_gaussian_reciprocal_metric(source,ao,auxiliary,config.metric,nested_live(0),config.metric_caps);
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,raw.plan().work_units_upper_bound);
        receipt.reciprocal_candidate_evaluations=add(receipt.reciprocal_candidate_evaluations,raw.plan().candidate_evaluations);
        observe(add(retained_output,raw.plan().owned_numeric_peak_bytes),raw.plan().per_replica_inventoried_bytes);
        consumed.u64(q); consumed.text(source.source_identity_sha256()); consumed.text(source.conjugate_source_identity_sha256());
        consumed.text(raw.payload_identity_sha256());
        notify(PeriodicGaussianFockStage::Whitening,q);
        reserve(config.metric_caps.maximum_work_units,0);
        const auto whitener=factorize_periodic_gaussian_metric(std::move(raw),config.metric.whitener_column_block,nested_live(0),config.metric_caps);
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,whitener.plan().work_units_upper_bound);
        observe(add(retained_output,whitener.plan().owned_numeric_peak_bytes),whitener.plan().per_replica_inventoried_bytes);
        consumed.text(whitener.payload_identity_sha256());
        const auto tile=[&](std::uint64_t bra,std::uint64_t aux,std::uint64_t rows,std::uint64_t mu,
                            std::uint64_t nu,std::uint64_t columns,std::uint64_t scratch) {
            limit(add(receipt.completed_tile_count,1U),caps.maximum_tile_calls,"Gaussian Fock remaining tile-call cap");
            const auto max_images=mul(config.tile_caps.maximum_image_candidates,
                add(add(1U,source.accepted_vector_count()),ceil_div(source.accepted_vector_count(),config.tile.reciprocal_block)));
            reserve(config.tile_caps.resources.maximum_work_units,config.tile_caps.resources.maximum_candidate_evaluations,max_images);
            const PeriodicGaussianThreeCenterSelection selection{bra,mu*n+nu,columns,aux,rows};
            auto t=build_periodic_gaussian_three_center_tile(source,whitener,ao,auxiliary,selection,
                config.tile,nested_live(scratch),config.tile_caps);
            if(t.descriptor().k_ket_index!=context->ket_index(bra,q) || t.descriptor().element_count!=rows*columns) {
                throw std::logic_error("Gaussian Fock received an inconsistent native tile");
            }
            const auto& p=t.plan();
            receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,p.work_units_upper_bound);
            receipt.reciprocal_candidate_evaluations=add(receipt.reciprocal_candidate_evaluations,p.reciprocal_candidate_evaluations);
            receipt.image_candidate_evaluations=add(receipt.image_candidate_evaluations,p.image_candidate_evaluations);
            observe(add(add(retained_output,scratch),add(p.resident_whitener_bytes,p.owned_numeric_peak_bytes)),p.per_replica_inventoried_bytes);
            consumed.u64(receipt.completed_tile_count); consumed.text(t.payload_identity_sha256());
            ++receipt.completed_tile_count;
            return t;
        };
        if(q==0) {
            std::vector<Complex> z(static_cast<std::size_t>(ab)),zc(static_cast<std::size_t>(ab));
            for(std::uint64_t aux=0;aux<a;aux+=ab) {
                const auto rows=std::min(ab,a-aux);
                std::fill(z.begin(),z.end(),Complex{}); std::fill(zc.begin(),zc.end(),Complex{});
                for(std::uint64_t k=0;k<nk;++k) {
                    notify(PeriodicGaussianFockStage::HartreeDensity,q,aux,k,k);
                    for(std::uint64_t mu=0;mu<n;++mu) for(std::uint64_t nu=0;nu<n;nu+=b) {
                        const auto columns=std::min(b,n-nu);
                        const auto t=tile(k,aux,rows,mu,nu,columns,plan.hartree_vector_bytes);
                        for(std::uint64_t p=0;p<rows;++p) for(std::uint64_t s=0;s<columns;++s) {
                            accumulate(density.data[(k*n+nu+s)*n+mu]*t.matrix_row_major()[p*columns+s],z[p],zc[p]);
                        }
                    }
                }
                for(std::uint64_t p=0;p<rows;++p) { z[p]=completed(z[p],zc[p])*inverse_nk; finite(z[p]); }
                for(std::uint64_t k=0;k<nk;++k) {
                    notify(PeriodicGaussianFockStage::HartreeApply,q,aux,k,k);
                    for(std::uint64_t mu=0;mu<n;++mu) for(std::uint64_t nu=0;nu<n;nu+=b) {
                        const auto columns=std::min(b,n-nu);
                        const auto t=tile(k,aux,rows,mu,nu,columns,plan.hartree_vector_bytes);
                        for(std::uint64_t p=0;p<rows;++p) for(std::uint64_t s=0;s<columns;++s) {
                            add_output(k,nu+s,mu,std::conj(t.matrix_row_major()[p*columns+s])*z[p]);
                        }
                    }
                }
            }
        }
        {
            const auto elements=plan.exchange_double_panel_bytes/32U;
            std::vector<Complex> first(static_cast<std::size_t>(elements)),dm(static_cast<std::size_t>(elements));
            const auto load=[&](std::uint64_t bra,std::uint64_t aux,std::uint64_t rows,std::uint64_t nu,std::uint64_t columns) {
                for(std::uint64_t mu=0;mu<n;++mu) {
                    const auto t=tile(bra,aux,rows,mu,nu,columns,plan.exchange_double_panel_bytes);
                    for(std::uint64_t p=0;p<rows;++p) for(std::uint64_t s=0;s<columns;++s) {
                        first[(p*n+mu)*b+s]=t.matrix_row_major()[p*columns+s];
                    }
                }
            };
            for(std::uint64_t bra=0;bra<nk;++bra) {
                const auto ket=context->ket_index(bra,q);
                for(std::uint64_t aux=0;aux<a;aux+=ab) {
                    const auto rows=std::min(ab,a-aux);
                    for(std::uint64_t sigma=0;sigma<n;sigma+=b) {
                        const auto sigmas=std::min(b,n-sigma);
                        notify(PeriodicGaussianFockStage::Exchange,q,aux,bra,ket,sigma);
                        load(bra,aux,rows,sigma,sigmas);
                        for(std::uint64_t p=0;p<rows;++p) for(std::uint64_t mu=0;mu<n;++mu) for(std::uint64_t s=0;s<sigmas;++s) {
                            Complex sum{},comp{};
                            for(std::uint64_t lambda=0;lambda<n;++lambda) {
                                accumulate(density.data[(bra*n+mu)*n+lambda]*first[(p*n+lambda)*b+s],sum,comp);
                            }
                            dm[(p*n+mu)*b+s]=completed(sum,comp);
                        }
                        for(std::uint64_t nu=0;nu<n;nu+=b) {
                            const auto columns=std::min(b,n-nu);
                            load(bra,aux,rows,nu,columns);
                            for(std::uint64_t p=0;p<rows;++p) for(std::uint64_t v=0;v<columns;++v) for(std::uint64_t s=0;s<sigmas;++s) {
                                Complex sum{},comp{};
                                for(std::uint64_t mu=0;mu<n;++mu) {
                                    accumulate(std::conj(first[(p*n+mu)*b+v])*dm[(p*n+mu)*b+s],sum,comp);
                                }
                                add_output(ket,nu+v,sigma+s,completed(sum,comp)*(-0.5*inverse_nk));
                            }
                        }
                    }
                }
            }
        }
        ++receipt.completed_q_count;
        notify(PeriodicGaussianFockStage::QComplete,q);
        // first/dm and all tiles are dead; W/source die here before next q.
    }
    if(receipt.completed_q_count!=nk || receipt.completed_tile_count!=plan.total_tile_calls) {
        throw std::logic_error("Gaussian Fock execution did not complete its admitted sequence");
    }
    for(std::size_t i=0;i<result.values_.size();++i) result.values_[i]=completed(result.values_[i],correction[i]);
    std::vector<Complex>().swap(correction);
    result.response_diagnostics_=matrix_diagnostics(*context,result.values_.data(),false);
    if(consumed.extent()!=add(kConsumedPrefix,add(mul(296U,nk),mul(80U,plan.total_tile_calls)))) {
        throw std::logic_error("Gaussian Fock factor trace extent changed");
    }
    result.consumed_=consumed.finish();
    notify(PeriodicGaussianFockStage::Complete,nk-1);
    if(density_hash(*context,density)!=result.density_) throw std::invalid_argument("Gaussian Fock density changed during execution");
    if(progress && receipt.progress_callback_count!=plan.progress_callback_upper_bound) {
        throw std::logic_error("Gaussian Fock progress sequence changed");
    }
    Digest payload; payload.text(kPayloadDomain); payload.u32(1U);
    payload.text(context->source_context_identity_sha256()); payload.text(result.density_identity_sha256());
    payload.text(result.consumed_factor_identity_sha256());
    payload.u64(nk); payload.u64(n); payload.u64(result.values_.size()); payload.u64(plan.response_bytes);
    diagnostics_wire(payload,result.density_diagnostics_); diagnostics_wire(payload,result.response_diagnostics_);
    if(payload.extent()!=kPayloadPrefix) throw std::logic_error("Gaussian Fock payload prefix changed");
    for(const auto z:result.values_) { payload.real(z.real()); payload.real(z.imag()); }
    result.payload_=payload.finish(); return result;
}

#define VIBEQC_GFOCK_ID(Name,Field) \
std::string PeriodicGaussianFockResult::Name() const { return {Field.begin(),Field.end()}; }
VIBEQC_GFOCK_ID(density_identity_sha256,density_)
VIBEQC_GFOCK_ID(consumed_factor_identity_sha256,consumed_)
VIBEQC_GFOCK_ID(payload_identity_sha256,payload_)
#undef VIBEQC_GFOCK_ID

} // namespace vibeqc
