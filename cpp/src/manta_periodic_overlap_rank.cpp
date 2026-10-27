#include "vibeqc/manta_periodic_one_electron_bounds.hpp"
#include "vibeqc/manta_admission.hpp"
#include "bounded_periodic_scf_numerics.hpp"
#include "manta_ri_interval.hpp"
#include "manta_lattice_geometry.hpp"

namespace vibeqc::manta {
namespace {
RIOptions rank_options(const BoundedPeriodicRHFOptions& scf) {
    RIOptions o;o.absolute_cutoff=scf.overlap_rank_absolute_floor;
    o.relative_cutoff=scf.overlap_rank_relative_floor;return o;
}
}
HermitianSpectrumPlan plan_periodic_overlap_image_rank(Count n,Count nk,
    const BoundedPeriodicRHFOptions& scf,const ResourceBudget& budget) {
    namespace a=admission;
    a::margin(budget);
    // The same shared canonical-overlap work bound and numerical controls.
    const auto discovery=plan_bounded_periodic_density_envelope(nk,n,0,0,scf);
    auto leaf=budget;leaf.other_live_bytes=0;
    leaf.memory_bytes=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    auto p=plan_hermitian_spectrum(n,rank_options(scf),leaf);
    // Raw/entry-bound binding copies, canonical A/Q, Jacobi buffers, interval
    // norms and owned/exported per-k receipts coexist. No AO-pair tensor.
    p.modeled_peak_bytes=a::add(budget.other_live_bytes,a::add(p.modeled_peak_bytes,
        a::add(discovery.total_numerical_bytes,a::add(a::mul(128,a::mul(nk,a::square(n))),a::mul(2048,nk)))));
    p.scalar_terms=a::add(discovery.maximum_work_units,a::mul(nk,a::add(p.scalar_terms,a::mul(512,a::square(n)))));
    p.kernel_terms=a::mul(8,p.scalar_terms);
    a::cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA overlap image rank memory budget exceeded");
    a::cap(p.scalar_terms,budget.scalar_terms,"MANTA overlap image rank scalar budget exceeded");
    a::cap(p.kernel_terms,budget.kernel_terms,"MANTA overlap image rank kernel budget exceeded");
    return p;
}
PeriodicOverlapImageRankResult bound_periodic_overlap_image_rank(Count n,Count nk,
    const std::vector<std::complex<double>>& overlap,const md::Matrix& errors,
    Count expected_rank,const BoundedPeriodicRHFOptions& scf,const ResourceBudget& budget) {
    namespace a=admission;namespace si=spectral_interval;
    const auto plan=plan_periodic_overlap_image_rank(n,nk,scf,budget);si::require_arithmetic();
    const Count nn=a::square(n);
    if(overlap.size()!=a::mul(nk,nn) || errors.rows()!=n || errors.cols()!=n || expected_rank>n ||
       !errors.allFinite() || (errors.array()<0.).any())
        throw std::invalid_argument("MANTA overlap image rank requires matching finite nonnegative bounds");
    for(const auto z:overlap)if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("MANTA overlap image rank requires finite overlap");
    PeriodicOverlapImageRankResult out;out.qualified=true;
    out.minimum_stability_radius=std::numeric_limits<double>::infinity();
    out.spectra.reserve(nk);out.perturbation_bounds.reserve(nk);out.projection_bounds.reserve(nk);
    lattice_detail::Hash identity;identity.text("manta.overlap-image-rank.v1");
    identity.word(n);identity.word(nk);identity.word(expected_rank);
    identity.number(scf.overlap_rank_absolute_floor);identity.number(scf.overlap_rank_relative_floor);
    for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)identity.number(errors(i,j));
    std::vector<std::complex<double>> work(nn),vectors(nn);std::vector<double> values(n);
    Eigen::MatrixXcd matrix(n,n),q(n,n);Eigen::VectorXd eigenvalues(n);
    auto leaf=budget;
    const auto spectrum_plan=plan_hermitian_spectrum(n,rank_options(scf),budget);
    leaf.other_live_bytes=plan.modeled_peak_bytes-spectrum_plan.modeled_peak_bytes+budget.other_live_bytes;
    for(Count k=0;k<nk;++k) {
        const auto* raw=overlap.data()+k*nn;
        // Reuse exactly the shared solver's canonical projection, Jacobi,
        // negativity and residual checks; no independent rank convention.
        BoundedPeriodicRHFDiagnostics diagnostics;
        const auto selected=vibeqc::overlap_eigen(raw,n,work.data(),vectors.data(),values.data(),scf,diagnostics);
        std::vector<double> rows(n),columns(n),projection_rows(n),projection_columns(n);
        for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
            const auto z=vibeqc::hermitian(raw,n,i,j);matrix(i,j)=z;q(i,j)=vectors[i*n+j];
            identity.number(raw[i*n+j].real());identity.number(raw[i*n+j].imag());
            // Image bounds refer to the raw finite producer. Include the
            // actual canonical projection displacement, not an assumption
            // that numerical conjugate lanes were exactly identical.
            const double projection=z==raw[i*n+j]?0.:si::magnitude(si::sub(si::point(z),si::point(raw[i*n+j])));
            const double total=si::plus_up(errors(i,j),projection);
            rows[i]=si::plus_up(rows[i],total);columns[j]=si::plus_up(columns[j],total);
            projection_rows[i]=si::plus_up(projection_rows[i],projection);
            projection_columns[j]=si::plus_up(projection_columns[j],projection);
        }
        for(Count i=0;i<n;++i)eigenvalues[i]=values[i];
        auto c=certify_hermitian_spectrum(matrix,eigenvalues,q,rank_options(scf),leaf);
        double perturbation=0.,projection=0.;
        for(Count i=0;i<n;++i) {
            perturbation=std::max({perturbation,rows[i],columns[i]});
            projection=std::max({projection,projection_rows[i],projection_columns[i]});
        }
        // Rump/Lange (2023), doi:10.1016/j.cam.2023.115332, Secs.3-4,
        // plus Weyl: ||Delta S||2 <= max(row/column entry-bound sums).
        // The shared radius includes motion of the relative cutoff. Equality
        // refuses. This conditional image proof excludes source arithmetic,
        // eigenvector/occupied-space motion and any relaxed energy change.
        const bool qualified=c.rank_certified && c.rank==selected && selected==expected_rank &&
            perturbation<c.rank_stability_radius;
        out.qualified=out.qualified && qualified;
        out.maximum_perturbation_bound=std::max(out.maximum_perturbation_bound,perturbation);
        out.minimum_stability_radius=std::min(out.minimum_stability_radius,c.rank_stability_radius);
        out.spectra.push_back(c);out.perturbation_bounds.push_back(perturbation);out.projection_bounds.push_back(projection);
        identity.word(qualified);identity.word(c.rank);identity.number(perturbation);identity.number(projection);
        identity.number(c.rank_stability_radius);identity.number(c.eigenvalue_error_bound);
    }
    out.certificate_identity=identity.value.finish_hex();return out;
}
} // namespace vibeqc::manta
