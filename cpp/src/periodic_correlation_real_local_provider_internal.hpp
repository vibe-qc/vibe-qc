#pragma once

// Arithmetic only. This leaf neither creates nor certifies a QC provider;
// its tiny diagnostic binding has no physical provenance or factory access.
#include "vibeqc/periodic_correlation_real_local_provider.hpp"
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace periodic_correlation_real_local_detail {

// Private native-factory bridge only. No binding exposes this construction
// access or admits caller-labelled factor/row arrays as a physical source.
struct ProviderAccess {
    static PeriodicCorrelationRealLocalProvider gaussian(
        const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationRealLocalProviderMemoryPlan& memory,
        const PeriodicCorrelationRealLocalProviderOptions& options,
        const PeriodicCorrelationRealLocalBasis& basis,
        const std::string& context,const std::string& hf) {
        if (context.size()!=64 || hf.size()!=64)
            throw std::logic_error("Gaussian real provider lacks native source identities");
        PeriodicCorrelationRealLocalProvider p;
        p.state_=reference.state_handle(); p.memory_=memory; p.options_=options;
        p.basis_=basis.local_basis_identity_sha256(); p.certificate_=basis.identity_sha256();
        p.source_kind_=PeriodicCorrelationRealLocalProviderSourceKind::MatchedGaussianHF;
        p.source_context_=context; p.hf_source_=hf;
        p.rows_.resize(static_cast<std::size_t>(memory.retained_row_bytes/8));
        return p;
    }
    static std::vector<double>& rows(PeriodicCorrelationRealLocalProvider& p) { return p.rows_; }
    static PeriodicCorrelationRealLocalProviderDiagnostics& diagnostics(PeriodicCorrelationRealLocalProvider& p) {
        return p.diagnostics_;
    }
    static void seal(PeriodicCorrelationRealLocalProvider& p,std::string consumed,std::string payload,std::string identity) {
        if (consumed.size()!=64 || payload.size()!=64 || identity.size()!=64)
            throw std::logic_error("Gaussian real provider numerical seal is incomplete");
        p.consumed_=std::move(consumed); p.payload_=std::move(payload); p.identity_=std::move(identity);
    }
};

inline std::uint64_t density_index(std::uint64_t left,std::uint64_t right,std::uint64_t n) {
    if (left >= n || right >= n) throw std::out_of_range("real-local provider orbital index is out of range");
    if (left > right) std::swap(left,right);
    return left*n-left*(left+1)/2+right;
}

// Extracted unchanged from the original store-backed factory. Panel shape,
// owners and physical source seals must already have been validated by the
// enclosing native factory. The template is private, with no runtime sink,
// callback or externally supplied numerical panel route.
template<class NativePanel>
inline void convert_provider_panel_pair(const NativePanel& panel,const NativePanel* partner,
    std::uint64_t q,std::uint64_t partner_q,std::uint64_t begin,std::uint64_t count,
    const PeriodicCorrelationRealLocalProviderMemoryPlan& memory,
    const PeriodicCorrelationRealLocalProviderOptions& options,
    PeriodicCorrelationRealLocalProviderDiagnostics& diagnostic,double* rows,double* norms) {
    const auto n=memory.orbital_count,h=memory.density_count,aux=memory.n_auxiliary;
    auto* original=norms; auto* reversal=original+h; auto* covariance=reversal+h;
    for (std::uint64_t p = 0; p < count; ++p) {
        for (std::uint64_t i = 0; i < n; ++i) for (std::uint64_t j = 0; j < n; ++j) {
            const auto x = panel.element(p,i,j), reversed = panel.element(p,j,i);
            diagnostic.maximum_reversal_error = std::max(diagnostic.maximum_reversal_error,defect_up(x,reversed));
            if (!within(x,reversed,options.reversal_absolute_tolerance,options.reversal_relative_tolerance))
                throw std::invalid_argument("real-local actual same-q density reversal exceeds tolerance");
            if (partner) {
                const auto y = partner->element(p,i,j), yr = partner->element(p,j,i);
                diagnostic.maximum_reversal_error = std::max(diagnostic.maximum_reversal_error,defect_up(y,yr));
                if (!within(y,yr,options.reversal_absolute_tolerance,options.reversal_relative_tolerance))
                    throw std::invalid_argument("real-local actual same-q density reversal exceeds tolerance");
                diagnostic.maximum_conjugacy_error = std::max(diagnostic.maximum_conjugacy_error,defect_up(y,std::conj(x)));
                if (!within(y,std::conj(x),options.conjugacy_absolute_tolerance,options.conjugacy_relative_tolerance))
                    throw std::invalid_argument("real-local actual original-row q conjugacy exceeds tolerance");
            } else {
                diagnostic.maximum_self_q_imaginary_magnitude = std::max(diagnostic.maximum_self_q_imaginary_magnitude,std::abs(x.imag()));
                if (!within(x,Complex(x.real(),0),options.self_q_absolute_tolerance,options.self_q_relative_tolerance))
                    throw std::invalid_argument("real-local actual self-q imaginary lane exceeds tolerance");
            }
        }
        for (std::uint64_t i = 0; i < n; ++i) for (std::uint64_t j = i; j < n; ++j) {
            const auto index = density_index(i,j,n);
            const auto x = panel.element(p,i,j);
            norm_value(original[index],x); norm_difference(reversal[index],panel.element(p,j,i),x);
            if (partner) {
                const auto y = partner->element(p,i,j);
                norm_value(original[index],y); norm_difference(reversal[index],partner->element(p,j,i),y);
                norm_difference(covariance[index],y,std::conj(x));
                constexpr double sqrt_two = 0x1.6a09e667f3bcdp+0;
                rows[(q*aux+begin+p)*h+index] = finite(sqrt_two*x.real());
                rows[(partner_q*aux+begin+p)*h+index] = finite(sqrt_two*x.imag());
            } else {
                norm_lane(covariance[index],std::abs(x.imag()));
                rows[(q*aux+begin+p)*h+index] = x.real();
            }
        }
    }
}

inline void finish_provider_norms(const PeriodicCorrelationRealLocalProviderMemoryPlan& memory,
    PeriodicCorrelationRealLocalProviderDiagnostics& diagnostic,const double* norms) {
    const auto h=memory.density_count;
    const auto* original=norms; const auto* reversal=original+h; const auto* covariance=reversal+h;
    for (std::uint64_t index = 0; index < h; ++index) {
        diagnostic.maximum_original_density_norm = std::max(diagnostic.maximum_original_density_norm,sqrt_up(original[index]));
        diagnostic.maximum_reversal_norm = std::max(diagnostic.maximum_reversal_norm,sqrt_up(reversal[index]));
        diagnostic.maximum_covariance_projection_norm = std::max(diagnostic.maximum_covariance_projection_norm,sqrt_up(covariance[index]));
    }
}
inline double provider_quadratic_error(double n,double defect) {
    return add_up(mul_up(2.0,mul_up(n,defect)),mul_up(defect,defect));
}
inline void finish_provider_projection(const PeriodicCorrelationRealLocalProviderMemoryPlan& memory,
    const PeriodicCorrelationRealLocalProviderOptions& options,PeriodicCorrelationRealLocalProviderDiagnostics& diagnostic) {
    const auto original_norm = diagnostic.maximum_original_density_norm;
    const auto covariance_norm = diagnostic.maximum_covariance_projection_norm;
    diagnostic.orientation_error_bound = provider_quadratic_error(original_norm,diagnostic.maximum_reversal_norm);
    diagnostic.covariance_error_bound = provider_quadratic_error(original_norm,covariance_norm);
    if (memory.self_inverse_q_count != memory.n_cells) {
        constexpr double four_u = 0x1p-51;
        diagnostic.maximum_real_row_conversion_norm = add_up(
            mul_up(four_u,add_up(original_norm,covariance_norm)),
            mul_up(sqrt_up(up(static_cast<double>(memory.row_count))),std::numeric_limits<double>::denorm_min()));
    }
    diagnostic.conversion_error_bound = provider_quadratic_error(add_up(original_norm,covariance_norm),
        diagnostic.maximum_real_row_conversion_norm);
    diagnostic.maximum_eri_projection_error_bound = add_up(add_up(
        diagnostic.orientation_error_bound,diagnostic.covariance_error_bound),diagnostic.conversion_error_bound);
    if (diagnostic.maximum_eri_projection_error_bound > options.maximum_eri_projection_error)
        throw std::invalid_argument("real-local ERI projection exceeds its explicit error budget");
}

inline std::uint64_t dot_interval_work(std::uint64_t count) { return add(mul(256,count),64); }

inline void dot_view(const double* data, std::size_t accessible, std::uint64_t count, std::uint64_t stride) {
    if (!data || !stride || reinterpret_cast<std::uintptr_t>(data)%alignof(double))
        throw std::invalid_argument("real-local scalar dot requires aligned nonnull views and positive strides");
    const auto needed = add(mul(count-1,stride),1), bytes = mul(8,needed);
    extent(bytes);
    if (accessible < needed) throw std::length_error("real-local scalar dot accessible extent is too short");
    if (bytes > std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(data))
        throw std::overflow_error("real-local scalar dot pointer extent overflows");
}

inline PeriodicCorrelationRealLocalIntegral dot_interval(
    const double* a, std::size_t accessible_a, std::uint64_t stride_a,
    const double* b, std::size_t accessible_b, std::uint64_t stride_b,
    std::uint64_t count, std::uint64_t maximum_work, double maximum_error) {
    if (!count) throw std::invalid_argument("real-local scalar dot requires at least one lane");
    if (!maximum_work || dot_interval_work(count) > maximum_work)
        throw std::length_error("real-local scalar dot work cap is missing or exceeded");
    if (!std::isfinite(maximum_error) || maximum_error <= 0)
        throw std::invalid_argument("real-local scalar dot error budget must be finite positive");
    dot_view(a,accessible_a,count,stride_a); dot_view(b,accessible_b,count,stride_b);
    float_environment();
    double sum = 0.0, correction = 0.0, lower = 0.0, upper = 0.0;
    for (std::uint64_t k = 0; k < count; ++k) {
        const auto x = finite(a[k*stride_a]), y = finite(b[k*stride_b]);
        if (x == 0.0 || y == 0.0) continue;  // Exact zero, not an underflow product.
        const auto product = finite(x*y), next = finite(sum+product);
        const auto adjustment = std::abs(sum) >= std::abs(product)
            ? (sum-next)+product : (product-next)+sum;
        correction = finite(correction+adjustment); sum = next;
        // One outward step contains the exact product, including subnormal
        // or rounded-to-zero results; another contains endpoint addition.
        lower = down(finite(lower+down(product)));
        upper = up(finite(upper+up(product)));
    }
    PeriodicCorrelationRealLocalIntegral out;
    out.value = finite(sum+correction);
    out.roundoff_error_bound = std::max(difference_up(out.value,lower),difference_up(out.value,upper));
    if (out.roundoff_error_bound > maximum_error)
        throw std::invalid_argument("real-local scalar integral roundoff exceeds its error budget");
    return out;
}

}  // namespace periodic_correlation_real_local_detail
}  // namespace vibeqc
