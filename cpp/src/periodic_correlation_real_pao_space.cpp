#include "vibeqc/periodic_correlation_real_pao_space.hpp"

#include "periodic_correlation_real_pao_space_internal.hpp"

namespace vibeqc {
namespace {

namespace local = periodic_correlation_local_detail;
namespace real = periodic_correlation_real_local_detail;
namespace interval = periodic_correlation_real_pao_detail;
using Complex = std::complex<double>;
using I = interval::Interval;
using CI = interval::ComplexInterval;
using Options = PeriodicCorrelationRealPAOSpaceOptions;
using Memory = PeriodicCorrelationRealPAOSpaceMemoryPlan;
using real::add;
using real::mul;

void options_valid(const Options& o) {
    const auto& a = o.algebra;
    for (auto value : {a.rank_absolute_cutoff,a.negative_absolute_tolerance})
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("real PAO absolute rank/negative controls must be finite nonnegative");
    for (auto value : {a.rank_relative_cutoff,a.negative_relative_tolerance})
        if (!std::isfinite(value) || value < 0 || value >= 1)
            throw std::invalid_argument("real PAO relative rank/negative controls must be finite in [0,1)");
    if ((a.rank_absolute_cutoff == 0 && a.rank_relative_cutoff == 0)
        || (a.negative_absolute_tolerance == 0 && a.negative_relative_tolerance == 0))
        throw std::invalid_argument("real PAO rank/negative pairs need a positive control");
    real::tolerance_pair(a.validation_absolute_tolerance,a.validation_relative_tolerance);
    real::tolerance_pair(o.coefficient_tr_absolute_tolerance,o.coefficient_tr_relative_tolerance);
    if (!a.eigensolver.max_sweeps || !std::isfinite(a.eigensolver.relative_offdiagonal_tolerance)
        || a.eigensolver.relative_offdiagonal_tolerance <= 0 || a.eigensolver.relative_offdiagonal_tolerance >= 1)
        throw std::invalid_argument("real PAO eigensolver controls must be explicit and positive");
    for (auto value : {o.maximum_overlap_projection_error,o.maximum_fock_projection_error,
        o.maximum_overlap_spectral_uncertainty,o.maximum_original_metric_error,o.maximum_original_fock_error,
        o.maximum_original_projector_relation_error})
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("real PAO error budgets must be finite positive");
}
void inputs_valid(const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationPAODomain& domain) {
    real::float_environment(); local::validate_reference(reference);
    if (!domain.state_handle() || domain.state_handle().get() != reference.state_handle().get()
        || domain.contract_version() != kPeriodicCorrelationPAODomainContractVersion
        || domain.allocation_identity() != reference.dimensions().allocation_identity)
        throw std::invalid_argument("real PAO reference/domain state owners or allocation differ");
    if (!domain.domain_dimension()) throw std::invalid_argument("real PAO preparation requires a nonempty domain");
    if (!domain.options().require_time_reversal || !domain.options().require_real_matrices
        || !domain.diagnostics().time_reversal_compatible || !domain.diagnostics().real_matrices_compatible)
        throw std::invalid_argument("real PAO preparation requires proven domain TR and real-matrix gates");
    const auto p = plan_periodic_correlation_pao_domain(reference.state().mesh(),reference.state().n_basis(),domain.domain_dimension());
    if (domain.memory().retained_matrix_bytes != p.retained_matrix_bytes
        || domain.memory().retained_domain_index_bytes != p.retained_domain_index_bytes
        || domain.memory().matrix_element_count != p.matrix_element_count)
        throw std::logic_error("real PAO original domain inventory differs from its seal");
}
void admit(const PeriodicCorrelationAdmittedReference& reference, const Memory& m,
           const PeriodicCorrelationRealPAOSpaceCaps& c) {
    if (!c.maximum_owned_numerical_bytes || m.peak_owned_numerical_bytes > c.maximum_owned_numerical_bytes)
        throw std::length_error("real PAO owned numerical byte cap is missing or exceeded");
    if (!c.maximum_work_units || m.work_units > c.maximum_work_units)
        throw std::length_error("real PAO work cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || m.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("real PAO live inventory exceeds admitted node memory");
}
void error_gate(double error, double budget, const char* message) {
    real::finite(error);
    if (error > budget) throw std::runtime_error(message);
}
double projected_matrix_audit(const Complex* matrix, std::size_t n) {
    double squared = 0;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        const auto z = real::finite(matrix[i*n+j]);
        if (z != std::conj(matrix[j*n+i]))
            throw std::invalid_argument("real PAO original matrices must be exactly Hermitian");
        real::norm_lane(squared,std::abs(z.imag()));
    }
    return real::sqrt_up(squared);
}
struct EigenBounds { double gram = 0, reconstruction = 0, relative_residual = 0; };
template<class Matrix>
EigenBounds eigen_bounds(Matrix matrix, const Complex* u, const double* lambda,
                         std::size_t n, const Options& options) {
    for (std::size_t k = 0; k < n*n; ++k)
        if (real::finite(u[k]).imag() != 0)
            throw std::runtime_error("real PAO exactly-real eigensolver input yielded a non-real eigenvector");
    double gram = 0, reconstruction = 0, residual = 0, matrix_squared_lower = 0;
    bool nonzero_matrix = false;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
        const auto value = real::finite(matrix(i,j));
        nonzero_matrix = nonzero_matrix || value != 0;
        matrix_squared_lower = real::add_down(matrix_squared_lower,real::mul_down(std::abs(value),std::abs(value)));
        I g, t, r;
        for (std::size_t k = 0; k < n; ++k) {
            g = interval::plus(g,interval::product(u[k*n+i].real(),u[k*n+j].real()));
            t = interval::plus(t,interval::scale(interval::product(u[i*n+k].real(),lambda[k]),u[j*n+k].real()));
            r = interval::plus(r,interval::product(matrix(i,k),u[k*n+j].real()));
        }
        interval::norm(gram,g,i == j ? 1 : 0);
        interval::norm(reconstruction,t,value);
        interval::norm(residual,interval::minus(r,interval::product(u[i*n+j].real(),lambda[j])));
    }
    EigenBounds result;
    result.gram = real::sqrt_up(gram); result.reconstruction = real::sqrt_up(reconstruction);
    const auto lower = matrix_squared_lower == 0 ? 0 : std::max(0.0,real::down(std::sqrt(matrix_squared_lower)));
    if (nonzero_matrix && lower == 0)
        throw std::overflow_error("real PAO nonzero matrix norm lower bound underflows");
    result.relative_residual = interval::relative_up(real::sqrt_up(residual),lower);
    const auto tolerance = real::add_down(options.algebra.validation_absolute_tolerance,options.algebra.validation_relative_tolerance);
    error_gate(result.gram,tolerance,"real PAO independent eigenvector Gram gate failed");
    error_gate(result.relative_residual,tolerance,"real PAO independent eigensystem residual gate failed");
    return result;
}
double raw_bilinear_audit(const Complex* matrix, const Complex* c, const double* diagonal,
                         std::size_t n, std::size_t r, CI* column) {
    double squared = 0;
    for (std::size_t b = 0; b < r; ++b) {
        for (std::size_t mu = 0; mu < n; ++mu) {
            CI sum;
            for (std::size_t nu = 0; nu < n; ++nu)
                sum = interval::plus(sum,interval::scale(CI::point(matrix[mu*n+nu]),c[nu*r+b].real()));
            column[mu] = sum;
        }
        for (std::size_t a = 0; a < r; ++a) {
            CI sum;
            for (std::size_t mu = 0; mu < n; ++mu)
                sum = interval::plus(sum,interval::scale(column[mu],c[mu*r+a].real()));
            const auto target = a == b ? (diagonal ? diagonal[b] : 1) : 0;
            interval::norm(squared,sum,Complex(target,0));
        }
    }
    return real::sqrt_up(squared);
}
double raw_projector_audit(const Complex* s, const Complex* c, const Complex* x, const double* lambda,
                          std::size_t n, std::size_t r, CI* column) {
    double squared = 0;
    for (std::size_t b = 0; b < n; ++b) {
        for (std::size_t k = 0; k < r; ++k) {
            CI sum;
            for (std::size_t nu = 0; nu < n; ++nu)
                sum = interval::plus(sum,interval::scale(CI::point(s[nu*n+b]),c[nu*r+k].real()));
            column[k] = sum;
        }
        for (std::size_t a = 0; a < n; ++a) {
            CI actual; I target;
            for (std::size_t k = 0; k < r; ++k) {
                actual = interval::plus(actual,interval::scale(column[k],c[a*r+k].real()));
                target = interval::plus(target,interval::scale(interval::product(x[a*r+k].real(),lambda[k]),x[b*r+k].real()));
            }
            interval::norm(squared,CI{interval::minus(actual.re,target),actual.im});
        }
    }
    return real::sqrt_up(squared);
}
void options_wire(real::Digest& h, const Options& o) {
    const auto& a = o.algebra;
    for (auto value : {a.rank_absolute_cutoff,a.rank_relative_cutoff,a.negative_absolute_tolerance,
        a.negative_relative_tolerance,a.validation_absolute_tolerance,a.validation_relative_tolerance,
        a.eigensolver.relative_offdiagonal_tolerance,o.maximum_overlap_projection_error,o.maximum_fock_projection_error,
        o.maximum_overlap_spectral_uncertainty,o.maximum_original_metric_error,o.maximum_original_fock_error,
        o.maximum_original_projector_relation_error,o.coefficient_tr_absolute_tolerance,o.coefficient_tr_relative_tolerance}) h.real(value);
    h.u64(a.eigensolver.max_sweeps);
}

}  // namespace

Memory plan_periodic_correlation_real_pao_space(const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationPAODomain& domain, std::uint64_t r, const Options& options) {
    inputs_valid(reference,domain); options_valid(options);
    const auto n = domain.domain_dimension(), nn = mul(n,n), nnn = mul(nn,n);
    Memory m;
    m.compact = plan_periodic_correlation_pao_space(n,r);
    m.n_cells = reference.state().n_kpoints(); m.n_basis = reference.state().n_basis();
    m.work_units = mul(512,add(mul(options.algebra.eigensolver.max_sweeps,nnn),add(mul(8,nnn),add(mul(32,nn),mul(16,n)))));
    if (r) {
        m.compact.validation_phase_bytes = add(m.compact.validation_phase_bytes,mul(16,n));
        m.compact.peak_owned_numerical_bytes = std::max(m.compact.peak_owned_numerical_bytes,m.compact.validation_phase_bytes);
        m.coefficient_tr_phase_bytes = add(m.compact.output_numerical_bytes,mul(64,m.n_basis));
        const auto rr = mul(r,r), rrr = mul(rr,r);
        const auto vc = add(mul(m.n_basis,n),add(mul(m.n_basis,m.n_basis),
            add(mul(mul(2,m.n_basis),reference.state().n_effective_orbitals()),mul(4,m.n_basis))));
        const auto extra = add(mul(options.algebra.eigensolver.max_sweeps,rrr),
            add(mul(8,rrr),add(mul(12,mul(nn,r)),add(mul(16,mul(n,rr)),mul(mul(m.n_cells,r),add(mul(2,vc),m.n_basis))))));
        m.work_units = add(m.work_units,mul(512,extra));
    }
    m.peak_owned_numerical_bytes = std::max(m.compact.peak_owned_numerical_bytes,m.coefficient_tr_phase_bytes);
    real::extent(m.peak_owned_numerical_bytes);
    if (n > std::vector<CI>().max_size()) throw std::length_error("real PAO interval column exceeds vector extent");
    const auto& d = reference.dimensions(); const auto& b = reference.budget();
    m.required_node_memory_bytes = add(add(d.external_bytes,d.shared_bytes),
        add(mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)),
            mul(mul(b.mpi_ranks,b.workers_per_rank),add(m.compact.borrowed_domain_bytes,m.peak_owned_numerical_bytes))));
    return m;
}

const PeriodicCorrelationPAOSpace& PeriodicCorrelationRealPAOSpace::space() const {
    (void) space_.coefficients_data(); return space_;
}

PeriodicCorrelationRealPAOSpace make_periodic_correlation_real_pao_space(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationPAODomain& domain,
    const Options& options, const PeriodicCorrelationRealPAOSpaceCaps& caps) {
    auto memory = plan_periodic_correlation_real_pao_space(reference,domain,0,options);
    admit(reference,memory,caps);
    const auto n = static_cast<std::size_t>(domain.domain_dimension());
    (void) domain.overlap(n-1,n-1); (void) domain.fock(n-1,n-1);
    const auto* s = domain.overlap_data(); const auto* f = domain.fock_data();
    PeriodicCorrelationRealPAOSpaceDiagnostics diagnostic;
    diagnostic.overlap_projection_frobenius_upper_bound = projected_matrix_audit(s,n);
    diagnostic.fock_projection_frobenius_upper_bound = projected_matrix_audit(f,n);
    error_gate(diagnostic.overlap_projection_frobenius_upper_bound,options.maximum_overlap_projection_error,
               "real PAO overlap projection exceeds its error budget");
    error_gate(diagnostic.fock_projection_frobenius_upper_bound,options.maximum_fock_projection_error,
               "real PAO Fock projection exceeds its error budget");
    PeriodicCorrelationPAOSpace base;
    base.state_ = reference.state_handle(); base.state_digest_ = reference.state().state_identity_sha256();
    base.domain_index_digest_ = domain.domain_index_sha256(); base.domain_digest_ = domain.pao_domain_identity_sha256();
    base.allocation_identity_ = reference.dimensions().allocation_identity; base.options_ = options.algebra;
    base.overlap_eigenvalues_.resize(n);
    std::vector<Complex> u(n*n);
    {
        std::vector<Complex> work(n*n);
        for (std::size_t k = 0; k < n*n; ++k) work[k] = Complex(s[k].real(),0);
        base.diagnostics_.overlap_eigensolver = hermitian_jacobi_in_place(work.data(),work.size(),u.data(),u.size(),
            base.overlap_eigenvalues_.data(),n,n,options.algebra.eigensolver);
        if (base.diagnostics_.overlap_eigensolver.status != HermitianJacobiStatus::Success)
            throw std::runtime_error("real PAO overlap eigensolver failed to converge or represent its result");
    }
    const auto* lambda = base.overlap_eigenvalues_.data();
    const auto overlap_bounds = eigen_bounds([&](std::size_t i,std::size_t j) { return s[i*n+j].real(); },u.data(),lambda,n,options);
    diagnostic.overlap_eigenvector_gram_error_upper_bound = overlap_bounds.gram;
    diagnostic.overlap_reconstruction_error_upper_bound = overlap_bounds.reconstruction;
    diagnostic.overlap_polar_distance_upper_bound = interval::polar_distance(overlap_bounds.gram);
    const auto max_lambda = std::max(std::abs(lambda[0]),std::abs(lambda[n-1]));
    const auto e = diagnostic.overlap_polar_distance_upper_bound;
    diagnostic.overlap_eigenvalue_error_upper_bound = real::add_up(diagnostic.overlap_projection_frobenius_upper_bound,
        real::add_up(overlap_bounds.reconstruction,real::mul_up(max_lambda,
            real::add_up(real::mul_up(2,e),real::mul_up(e,e)))));
    error_gate(diagnostic.overlap_eigenvalue_error_upper_bound,options.maximum_overlap_spectral_uncertainty,
               "real PAO overlap spectral uncertainty exceeds its error budget");
    const auto epsilon = diagnostic.overlap_eigenvalue_error_upper_bound;
    const auto eigen_interval = [&](double value) { return interval::plus(I::point(value),I{-epsilon,epsilon}); };
    const auto largest = eigen_interval(lambda[n-1]);
    diagnostic.rank_cutoff_lower_bound = std::max(options.algebra.rank_absolute_cutoff,
        real::mul_down(options.algebra.rank_relative_cutoff,std::max(0.0,largest.lower)));
    diagnostic.rank_cutoff_upper_bound = std::max(options.algebra.rank_absolute_cutoff,
        real::mul_up(options.algebra.rank_relative_cutoff,std::max(0.0,largest.upper)));
    const auto maximum_abs_lower = std::max(0.0,interval::minus(I::point(max_lambda),I::point(epsilon)).lower);
    diagnostic.negative_tolerance_lower_bound = real::add_down(options.algebra.negative_absolute_tolerance,
        real::mul_down(options.algebra.negative_relative_tolerance,maximum_abs_lower));
    if (eigen_interval(lambda[0]).lower < -diagnostic.negative_tolerance_lower_bound)
        throw std::runtime_error("real PAO original-overlap negative boundary is not certified");
    std::size_t first = n;
    diagnostic.minimum_rank_margin_lower_bound = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < n; ++i) {
        const auto bounds = eigen_interval(lambda[i]);
        double margin;
        if (bounds.lower > diagnostic.rank_cutoff_upper_bound) {
            if (first == n) first = i;
            margin = std::max(0.0,interval::minus(I::point(bounds.lower),I::point(diagnostic.rank_cutoff_upper_bound)).lower);
        } else if (bounds.upper <= diagnostic.rank_cutoff_lower_bound) {
            margin = std::max(0.0,interval::minus(I::point(diagnostic.rank_cutoff_lower_bound),I::point(bounds.upper)).lower);
        } else {
            throw std::runtime_error("real PAO overlap rank is ambiguous within the certified spectral interval");
        }
        diagnostic.minimum_rank_margin_lower_bound = std::min(diagnostic.minimum_rank_margin_lower_bound,margin);
        base.diagnostics_.negative_overlap_eigenvalue_count += lambda[i] < 0;
    }
    const auto r = n-first;
    if (!r) throw std::runtime_error("real PAO nonempty domain has zero retained virtual rank");
    memory = plan_periodic_correlation_real_pao_space(reference,domain,r,options);
    admit(reference,memory,caps);
    base.memory_ = memory.compact; base.diagnostics_.required_node_memory_bytes = memory.required_node_memory_bytes;
    base.diagnostics_.minimum_overlap_eigenvalue = lambda[0]; base.diagnostics_.maximum_overlap_eigenvalue = lambda[n-1];
    base.diagnostics_.effective_rank_cutoff = diagnostic.rank_cutoff_upper_bound;
    base.diagnostics_.effective_negative_tolerance = diagnostic.negative_tolerance_lower_bound;
    base.diagnostics_.overlap_eigensystem_relative_residual = overlap_bounds.relative_residual;
    base.diagnostics_.overlap_eigenvector_orthogonality_error = overlap_bounds.gram;
    std::vector<Complex> x(n*r);
    double reconstructed_u_squared = 0, original_u_squared = 0;
    for (std::size_t b = 0; b < r; ++b) {
        const auto root = real::finite(std::sqrt(lambda[first+b]));
        const auto factor = real::finite(1/root);
        const I root_interval{std::max(0.0,real::down(root)),real::up(root)};
        for (std::size_t a = 0; a < n; ++a) {
            const auto original = u[a*n+first+b].real();
            x[a*r+b] = Complex(real::finite(original*factor),0);
            real::norm_lane(original_u_squared,std::abs(original));
            interval::norm(reconstructed_u_squared,interval::scale(root_interval,x[a*r+b].real()),original);
        }
    }
    // Preserve a bound to the ACTUAL retained U before freeing it. The
    // later target X Lambda X^T is not silently equated to U_r U_r^T.
    const auto du = real::sqrt_up(reconstructed_u_squared), nu = real::sqrt_up(original_u_squared);
    diagnostic.projected_projector_reconstruction_error_upper_bound = real::add_up(
        real::mul_up(2,real::mul_up(nu,du)),real::mul_up(du,du));
    std::vector<Complex>().swap(u);
    std::vector<Complex> h(r*r);
    {
        std::vector<Complex> column(n);
        for (std::size_t b = 0; b < r; ++b) {
            for (std::size_t a = 0; a < n; ++a) {
                real::Sum sum;
                for (std::size_t k = 0; k < n; ++k) sum.include(Complex(f[a*n+k].real()*x[k*r+b].real(),0));
                column[a] = sum.value();
            }
            for (std::size_t a = 0; a < r; ++a) {
                real::Sum sum;
                for (std::size_t k = 0; k < n; ++k) sum.include(Complex(x[k*r+a].real()*column[k].real(),0));
                h[a*r+b] = sum.value();
            }
        }
    }
    for (std::size_t a = 0; a < r; ++a) for (std::size_t b = a; b < r; ++b) {
        const auto left = h[a*r+b], right = h[b*r+a];
        if (!real::within(left,right,options.algebra.validation_absolute_tolerance,options.algebra.validation_relative_tolerance))
            throw std::runtime_error("real PAO projected Fock symmetry gate failed");
        base.diagnostics_.maximum_projected_fock_hermitian_defect = std::max(
            base.diagnostics_.maximum_projected_fock_hermitian_defect,real::defect_up(left,right));
        const auto value = Complex(real::finite(.5*left.real()+.5*right.real()),0);
        base.diagnostics_.maximum_projected_fock_hermitization_correction = std::max({
            base.diagnostics_.maximum_projected_fock_hermitization_correction,real::defect_up(value,left),real::defect_up(value,right)});
        h[a*r+b] = h[b*r+a] = value;
    }
    base.energies_.resize(r);
    {
        std::vector<Complex> v(r*r);
        {
            std::vector<Complex> work(h);
            base.diagnostics_.fock_eigensolver = hermitian_jacobi_in_place(work.data(),work.size(),v.data(),v.size(),
                base.energies_.data(),r,r,options.algebra.eigensolver);
            if (base.diagnostics_.fock_eigensolver.status != HermitianJacobiStatus::Success)
                throw std::runtime_error("real PAO Fock eigensolver failed to converge or represent its result");
        }
        const auto bounds = eigen_bounds([&](std::size_t a,std::size_t b) { return h[a*r+b].real(); },v.data(),base.energies_.data(),r,options);
        base.diagnostics_.fock_eigensystem_relative_residual = bounds.relative_residual;
        base.diagnostics_.fock_eigenvector_orthogonality_error = bounds.gram;
        std::vector<Complex>().swap(h);
        base.coefficients_.resize(n*r);
        for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < r; ++b) {
            real::Sum sum;
            for (std::size_t k = 0; k < r; ++k) sum.include(Complex(x[a*r+k].real()*v[k*r+b].real(),0));
            base.coefficients_[a*r+b] = sum.value();
        }
    }
    for (std::size_t b = 0; b < r; ++b) {
        std::size_t pivot = 0;
        for (std::size_t a = 0; a < n; ++a) {
            if (base.coefficients_[a*r+b].imag() != 0)
                throw std::runtime_error("real PAO rotation yielded a non-real coefficient");
            if (std::abs(base.coefficients_[a*r+b].real()) > std::abs(base.coefficients_[pivot*r+b].real())) pivot = a;
        }
        if (base.coefficients_[pivot*r+b].real() == 0) throw std::runtime_error("real PAO rotation produced a zero orbital");
        if (base.coefficients_[pivot*r+b].real() < 0)
            for (std::size_t a = 0; a < n; ++a) base.coefficients_[a*r+b] = -base.coefficients_[a*r+b];
    }
    {
        std::vector<CI> column(n);
        base.diagnostics_.canonical_metric_frobenius_residual = raw_bilinear_audit(s,x.data(),nullptr,n,r,column.data());
        error_gate(base.diagnostics_.canonical_metric_frobenius_residual,options.maximum_original_metric_error,
                   "real PAO canonical original metric exceeds its error budget");
        diagnostic.original_metric_frobenius_error_upper_bound = raw_bilinear_audit(s,base.coefficients_.data(),nullptr,n,r,column.data());
        diagnostic.original_fock_frobenius_error_upper_bound = raw_bilinear_audit(f,base.coefficients_.data(),base.energies_.data(),n,r,column.data());
        diagnostic.original_projector_relation_frobenius_error_upper_bound = real::add_up(
            raw_projector_audit(s,base.coefficients_.data(),x.data(),lambda+first,n,r,column.data()),
            diagnostic.projected_projector_reconstruction_error_upper_bound);
    }
    error_gate(diagnostic.original_metric_frobenius_error_upper_bound,options.maximum_original_metric_error,
               "real PAO original metric exceeds its error budget");
    error_gate(diagnostic.original_fock_frobenius_error_upper_bound,options.maximum_original_fock_error,
               "real PAO original Fock exceeds its error budget");
    error_gate(diagnostic.original_projector_relation_frobenius_error_upper_bound,options.maximum_original_projector_relation_error,
               "real PAO original retained-projector relation exceeds its error budget");
    base.diagnostics_.final_metric_frobenius_residual = diagnostic.original_metric_frobenius_error_upper_bound;
    double energy_squared_lower = 0;
    bool nonzero_energy = false;
    for (auto energy : base.energies_) {
        nonzero_energy = nonzero_energy || energy != 0;
        energy_squared_lower = real::add_down(energy_squared_lower,real::mul_down(std::abs(energy),std::abs(energy)));
    }
    const auto energy_norm_lower = energy_squared_lower == 0 ? 0 : std::max(0.0,real::down(std::sqrt(energy_squared_lower)));
    if (nonzero_energy && energy_norm_lower == 0)
        throw std::overflow_error("real PAO nonzero energy norm lower bound underflows");
    base.diagnostics_.final_projected_fock_relative_residual = interval::relative_up(
        diagnostic.original_fock_frobenius_error_upper_bound,energy_norm_lower);
    base.diagnostics_.retained_projector_relative_residual = interval::relative_up(
        diagnostic.original_projector_relation_frobenius_error_upper_bound,real::down(std::sqrt(static_cast<double>(r))));
    std::vector<Complex>().swap(x);
    {
        const auto nao = reference.state().n_basis();
        std::vector<Complex> columns(mul(4,nao));
        auto* left = columns.data(); auto* right = left+nao; auto* scratch = right+nao;
        double squared = 0;
        for (std::size_t k = 0; k < reference.state().n_kpoints(); ++k) {
            const auto bar = real::negative_cell(k,reference.state().mesh());
            ++diagnostic.inspected_kpoints; diagnostic.inspected_trim_points += k == bar;
            for (std::size_t a = 0; a < r; ++a) {
                local::fill_virtual_columns(reference.state(),domain,base,a,1,0,k,left,scratch);
                local::fill_virtual_columns(reference.state(),domain,base,a,1,0,bar,right,scratch);
                for (std::size_t mu = 0; mu < nao; ++mu) {
                    const auto defect = real::defect_up(right[mu],std::conj(left[mu]));
                    diagnostic.maximum_coefficient_tr_error = std::max(diagnostic.maximum_coefficient_tr_error,defect);
                    real::norm_difference(squared,right[mu],std::conj(left[mu]));
                    if (!real::within(right[mu],std::conj(left[mu]),options.coefficient_tr_absolute_tolerance,options.coefficient_tr_relative_tolerance))
                        throw std::runtime_error("real PAO actual expanded coefficients violate physical time reversal");
                }
            }
        }
        diagnostic.coefficient_tr_frobenius_upper_bound = real::sqrt_up(squared);
    }
    real::Digest payload("vibeqc.periodic.correlation.pao.space.payload");
    payload.u64(n); payload.u64(r);
    for (auto z : base.coefficients_) payload.complex(z);
    for (auto z : base.energies_) payload.real(z);
    for (auto z : base.overlap_eigenvalues_) payload.real(z);
    base.payload_digest_ = payload.finish();
    real::Digest identity("vibeqc.periodic.correlation.real-pao-space.identity");
    identity.string(base.state_digest_); identity.string(base.domain_index_digest_); identity.string(base.domain_digest_);
    identity.string(base.allocation_identity_); identity.string(base.payload_digest_); identity.string(real::kFloatPolicy);
    identity.string("explicit-real-S0-F0-projection;interval-qualified-rank;original-S-F-relations;actual-expanded-TR");
    options_wire(identity,options);
    for (auto value : {diagnostic.overlap_projection_frobenius_upper_bound,diagnostic.fock_projection_frobenius_upper_bound,
        diagnostic.overlap_eigenvector_gram_error_upper_bound,diagnostic.overlap_reconstruction_error_upper_bound,
        diagnostic.overlap_polar_distance_upper_bound,diagnostic.overlap_eigenvalue_error_upper_bound,
        diagnostic.rank_cutoff_lower_bound,diagnostic.rank_cutoff_upper_bound,diagnostic.negative_tolerance_lower_bound,
        diagnostic.minimum_rank_margin_lower_bound,diagnostic.projected_projector_reconstruction_error_upper_bound,
        diagnostic.original_metric_frobenius_error_upper_bound,diagnostic.original_fock_frobenius_error_upper_bound,
        diagnostic.original_projector_relation_frobenius_error_upper_bound,diagnostic.maximum_coefficient_tr_error,
        diagnostic.coefficient_tr_frobenius_upper_bound}) identity.real(value);
    identity.u64(diagnostic.inspected_kpoints); identity.u64(diagnostic.inspected_trim_points);
    base.identity_digest_ = identity.finish();
    PeriodicCorrelationRealPAOSpace result(std::move(base));
    result.options_ = options; result.memory_ = memory; result.diagnostics_ = diagnostic;
    result.identity_ = result.space_.pao_space_identity_sha256();
    return result;
}

}  // namespace vibeqc
