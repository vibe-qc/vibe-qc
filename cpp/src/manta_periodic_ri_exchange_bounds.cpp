#include "manta_lattice_cube.hpp"
#include "vibeqc/manta_periodic_ri_exchange_bounds.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_periodic_error.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "manta_ri_interval.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
using namespace admission;
using spectral_interval::plus_up;
using spectral_interval::times_up;
struct Digest {
    vibeqc::detail::Sha256 value;
    void word(Count v){std::array<std::uint8_t,8> a{};for(int i=0;i<8;++i)a[i]=v>>(56-8*i);value.update(a.data(),8);}
    void text(const std::string& s){word(s.size());value.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}
    void real(double x){if(x==0.)x=0.;Count v;std::memcpy(&v,&x,8);word(v);}
};
double norm(Complex z) {return periodic_error::upper(std::hypot(static_cast<long double>(z.real()),static_cast<long double>(z.imag())));}
ResourceBudget unlimited(ResourceBudget b){b.other_live_bytes=0;b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max();return b;}
void validate(const PeriodicRIExchangeFittingOptions& o,Count mesh,Count n) {
    if(!std::isfinite(o.density_l1_limit) || o.density_l1_limit<=0.)throw std::invalid_argument("MANTA exchange fitting requires a positive finite density L1 limit");
    for(double x:{o.matrix_error,o.energy_error,o.metric_assembly_matrix_error,o.metric_assembly_energy_error})if(!std::isfinite(x)||x<0.)throw std::invalid_argument("MANTA exchange fitting targets require finite nonnegative values");
    if((o.matrix_error>0.)!=(o.energy_error>0.))throw std::invalid_argument("MANTA exchange fitting requires both matrix and energy targets");
    if((o.metric_assembly_matrix_error>0.)!=(o.metric_assembly_energy_error>0.))throw std::invalid_argument("MANTA metric assembly exchange requires both matrix and energy targets");
    if(!o.fourier_decay_bins || o.fourier_decay_bins>8)throw std::invalid_argument("MANTA exchange Fourier decay bins must be in 1..8");
    if(!o.maximum_probes || mesh>o.maximum_probes || !o.reciprocal_batch)throw std::invalid_argument("MANTA exchange fitting probe or batch limit exceeded");
    if(!o.density_entry_limits.empty() && o.density_entry_limits.size()!=mesh)throw std::invalid_argument("MANTA exchange fitting entry-limit shape mismatch");
    for(Count q=0;q<o.density_entry_limits.size();++q) {
        const double x=o.density_entry_limits[q];const Count local=q%(n*n),transpose=q-local+(local%n)*n+local/n;
        if(!std::isfinite(x)||x<0. || x!=o.density_entry_limits[transpose])throw std::invalid_argument("MANTA exchange fitting entry limits must be finite nonnegative and symmetric");
    }
}
}
struct PeriodicRIExchangeFittingDriver {
    static void compatible(const PeriodicRIFockBuilder& ri,const PeriodicPairSource& source) {
        if(!ri.plan().has_k)throw std::invalid_argument("MANTA exchange fitting requires a fitted K owner");
        const auto& fit=ri.gamma_->coupling_source();
        if(fit.source().short_range.orbital_basis_identity!=source.receipt().basis_identity ||
           fit.momenta().rows()!=source.momenta().rows() ||
           !(fit.lattice().array()==source.lattice().array()).all() ||
           !(fit.momenta().array()==source.momenta().array()).all() || ri.weights_!=source.weights())
            throw std::invalid_argument("MANTA exchange fitting basis, cell, momentum or weight mismatch");
    }
    static PeriodicRIExchangeFittingPlan plan(const PeriodicRIFockBuilder& ri,const PeriodicPairSource& source,Count vectors,
        const PeriodicRIExchangeFittingOptions& o,const ResourceBudget& b) {
        compatible(ri,source);margin(b);const auto& a=ri.plan();const Count mesh=mul(a.momenta,square(a.n_basis));validate(o,mesh,a.n_basis);
        const auto leaf=unlimited(b);
        const auto tail=PeriodicExchangeBoundSource::plan_source(source,vectors,o.reciprocal_completion_cube_radius,1,o.ao_completion_cube_radius,leaf,o.fourier_decay_bins);
        // Count-only preflight uses total reference vectors as a conservative
        // per-channel ceiling; no vector payload or coefficient is read here.
        const auto exact=plan_periodic_exchange(source.basis_,vectors,source.images_.size(),a.momenta,1,o.reciprocal_batch,source.options_.pair_cutoff_bohr,leaf,ri.info().exchange_kernel,ri.info().physical_omega,true);
        return aggregate(a,tail,exact,source.plan().retained_bytes,source.images_.size(),vectors,
            ri.fitting_bound_?ri.fitting_bound_->plan().modeled_peak_bytes:0,nullptr,o,b);
    }
    static PeriodicRIExchangeFittingPlan aggregate(const PeriodicRIFockPlan& a,const PeriodicExchangeBoundPlan& tail,
        const PeriodicExchangePlan& exact,Count retained_pairs,Count cells,Count vectors,Count retained_fitting,
        const PeriodicPairPlan* preparation,const PeriodicRIExchangeFittingOptions& o,const ResourceBudget& b) {
        const Count mesh=mul(a.momenta,square(a.n_basis));
        PeriodicRIExchangeFittingPlan p;p.n_basis=a.n_basis;p.momenta=a.momenta;p.maximum_probes=mesh;
        p.shell_calls=add(mul(mesh,add(a.build_shell_calls,exact.shell_calls)),tail.shell_calls);
        p.primitive_products=add(mul(mesh,add(a.build_primitive_products,exact.primitive_pairs)),tail.primitive_pairs);
        const Count extra=add(4096,mul(16384,square(mesh)));
        p.scalar_terms=add(extra,mul(mesh,add(add(a.scalar_terms,exact.scalar_terms),tail.scalar_terms)));
        p.kernel_terms=add(mul(8,extra),mul(mesh,add(add(a.kernel_terms,exact.kernel_terms),tail.kernel_terms)));
        p.modeled_peak_bytes=add(add(b.other_live_bytes,b.backend_margin_bytes),add(add(a.modeled_peak_bytes,exact.modeled_peak_bytes),
            add(tail.modeled_peak_bytes,add(retained_pairs,add(16384,add(mul(2048,mesh),mul(64,add(vectors,cells))))))));
        // A previously attached J certificate can retain a separate full
        // Coulomb reference even though these K-only probes do not use it.
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,retained_fitting);
        if(preparation) {
            p.shell_calls=add(p.shell_calls,preparation->shell_calls);
            p.primitive_products=add(p.primitive_products,preparation->primitive_pairs);
            p.scalar_terms=add(p.scalar_terms,preparation->scalar_terms);
            p.kernel_terms=add(p.kernel_terms,preparation->kernel_terms);
            p.modeled_peak_bytes=add(p.modeled_peak_bytes,preparation->modeled_peak_bytes);
        }
        if(o.metric_assembly_matrix_error>0.) {
            const auto leaf=unlimited(b);
            auto include=[&](const PeriodicRIQPlan& q,Count count) {
                if(!count)return;
                if(!q.coupling_norm_supported)throw std::invalid_argument("MANTA metric assembly exchange requires qualified coupling norms");
                const auto norm=plan_periodic_gdf_coupling_norm_bounds(q.coupling_plan,leaf);
                p.shell_calls=add(p.shell_calls,mul(count,norm.shell_calls));
                p.primitive_products=add(p.primitive_products,mul(count,norm.primitive_products));
                p.scalar_terms=add(p.scalar_terms,mul(count,norm.scalar_terms));
                p.kernel_terms=add(p.kernel_terms,mul(count,norm.kernel_terms));
                // Conservative simultaneous reservation; execution streams pair norms.
                p.modeled_peak_bytes=add(p.modeled_peak_bytes,norm.modeled_peak_bytes);
            };
            include(a.gamma,1);include(a.pair,square(a.momenta)-a.momenta);
            const Count work=add(16384,mul(8192,square(mesh)));
            p.scalar_terms=add(p.scalar_terms,work);p.kernel_terms=add(p.kernel_terms,mul(8,work));
            p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(4096,mul(512,mesh)));
        }
        cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA exchange fitting memory budget exceeded");
        cap(p.shell_calls,b.shell_calls,"MANTA exchange fitting shell budget exceeded");
        cap(p.primitive_products,b.primitive_quartets,"MANTA exchange fitting primitive budget exceeded");
        cap(p.scalar_terms,b.scalar_terms,"MANTA exchange fitting scalar budget exceeded");
        cap(p.kernel_terms,b.kernel_terms,"MANTA exchange fitting kernel budget exceeded");return p;
    }
    static void metric_assembly(const PeriodicRIFockBuilder& ri,const PeriodicRIExchangeFittingOptions& o,
        PeriodicRIExchangeFittingResult& r,const ResourceBudget& b) {
        if(o.metric_assembly_matrix_error==0.)return;
        spectral_interval::require_arithmetic();r.metric_assembly_requested=true;
        const Count n=r.plan.n_basis,n2=square(n),nk=r.plan.momenta,mesh=mul(nk,n2);
        std::vector<double> maximum(mesh),sum(mesh);
        Digest id;id.text("manta.exchange-metric-assembly.v1;fixed-stored-coupling;unweighted-spin-family");
        id.text(ri.info().source_identity);id.real(o.density_l1_limit);
        for(double x:o.density_entry_limits)id.real(x);
        auto norms=[&](const PeriodicRIQBuilder& q) {
            const auto& metric=q.metric_info();
            if(!metric.assembly_rank_requested || !metric.assembly_rank_qualified || !metric.assembly_response.qualified)
                throw std::runtime_error("MANTA metric assembly exchange requires qualified metric_assembly_rank response on every channel");
            id.text(metric.source_identity);id.text(metric.assembly_certificate_identity);
            id.real(metric.assembly_response.inverse_error_bound);++r.metric_assembly_channels;
            auto c=q.coupling_source().coupling_norm_bounds(b);
            if(!c.metric_norm_qualified)throw std::logic_error("MANTA exchange coupling norm qualification lost");
            r.ri_shell_calls=add(r.ri_shell_calls,c.receipt.shell_calls);
            for(double x:c.metric_column_norm_bounds)id.real(x);
            return c.metric_column_norm_bounds;
        };
        const auto gamma=norms(*ri.gamma_);
        // Sun2017 q-channel T and the same B D B* map as the retained K owner.
        // |delta K_mu,nu| <= sum_l,a,b w_l |D_l,a,b| eta_kl n_mu,a n_nu,b.
        // Each spin is bounded by the declared spin-absolute family. The
        // metric difference is exact f(A+E)-f(A), not the rounded stored factor.
        // Contract 5.58; no primitive/coupling or executed-action error implied.
        for(Count k=0;k<nk;++k)for(Count l=0;l<nk;++l) {
            const auto& q=k==l?*ri.gamma_:*ri.pairs_[k*nk+l];
            const auto pair=k==l?std::vector<double>{}:norms(q);
            const auto& v=k==l?gamma:pair;const Count offset=k==l?k*n2:0;
            if(v.size()!=(k==l?mesh:n2))throw std::logic_error("MANTA exchange norm extent mismatch");
            const double eta=times_up(q.metric_info().assembly_response.inverse_error_bound,ri.weights_[l]);
            id.real(ri.weights_[l]);
            for(Count mu=0;mu<n;++mu)for(Count nu=0;nu<n;++nu) {
                const Count out=k*n2+mu*n+nu;
                for(Count a=0;a<n;++a)for(Count c=0;c<n;++c) {
                    const double coefficient=times_up(eta,times_up(v[offset+mu*n+a],v[offset+nu*n+c]));
                    maximum[out]=std::max(maximum[out],coefficient);
                    if(!o.density_entry_limits.empty())sum[out]=plus_up(sum[out],times_up(coefficient,o.density_entry_limits[l*n2+a*n+c]));
                }
            }
        }
        r.metric_assembly_error_bounds.resize(mesh);double weighted_max=0.,entry_energy=0.;
        for(Count x=0;x<mesh;++x) {
            double value=times_up(o.density_l1_limit,maximum[x]);
            if(!o.density_entry_limits.empty())value=std::min(value,sum[x]);
            if(!std::isfinite(value))throw std::runtime_error("MANTA metric assembly exchange bound is not finite");
            r.metric_assembly_error_bounds[x]=value;r.metric_assembly_matrix_bound=std::max(r.metric_assembly_matrix_bound,value);
            const double weighted=times_up(ri.weights_[x/n2],value);weighted_max=std::max(weighted_max,weighted);
            if(!o.density_entry_limits.empty())entry_energy=plus_up(entry_energy,times_up(o.density_entry_limits[x],weighted));
            id.real(value);
        }
        r.metric_assembly_energy_bound=times_up(.5,times_up(o.density_l1_limit,weighted_max));
        if(!o.density_entry_limits.empty())r.metric_assembly_energy_bound=std::min(r.metric_assembly_energy_bound,times_up(.5,entry_energy));
        if(!std::isfinite(r.metric_assembly_energy_bound))throw std::runtime_error("MANTA metric assembly exchange energy bound is not finite");
        r.metric_assembly_qualified=r.metric_assembly_matrix_bound<=o.metric_assembly_matrix_error && r.metric_assembly_energy_bound<=o.metric_assembly_energy_error;
        if(!r.metric_assembly_qualified)throw std::runtime_error("MANTA metric assembly exchange targets unmet");
        id.real(r.metric_assembly_energy_bound);r.metric_assembly_identity=id.value.finish_hex();
    }
    static PeriodicRIExchangeFittingResult run(const PeriodicRIFockBuilder& ri,const PeriodicPairSource& source,
        const md::Matrix& vectors,const std::vector<Count>& offsets,const PeriodicRIExchangeFittingOptions& o,const ResourceBudget& b) {
        PeriodicRIExchangeFittingResult r;r.plan=plan(ri,source,vectors.rows(),o,b);
        const Count n=r.plan.n_basis,n2=square(n),nk=r.plan.momenta,mesh=mul(nk,n2);
        const auto leaf=unlimited(b);
        metric_assembly(ri,o,r,leaf);
        PeriodicExchangeBoundSource tail(source,vectors,offsets,o.reciprocal_completion_cube_radius,o.ao_completion_cube_radius,leaf,ri.info().exchange_kernel,ri.info().physical_omega,o.fourier_decay_bins);
        md::Matrix translations(source.images_.size(),3);
        for(Count i=0;i<source.images_.size();++i)for(int axis=0;axis<3;++axis){long double x=0;for(int c=0;c<3;++c)x+=static_cast<long double>(source.lattice_(axis,c))*source.images_[i][c];translations(i,axis)=double(x);}
        r.ri_source_identity=ri.info().source_identity;r.reference_source_identity=tail.receipt().source_identity;r.pair_source_identity=source.receipt().source_identity;
        r.kernel=ri.info().exchange_kernel;r.physical_omega=ri.info().physical_omega;
        if(r.kernel!="coulomb")r.qualification="physical-"+r.kernel+"-exchange-on-unweighted-hermitian-spin-density-family;screened-zero-accounted;stored-integrals;no-madelung";
        r.density_l1_limit=o.density_l1_limit;r.density_entry_limits=o.density_entry_limits;r.targets_requested=o.matrix_error>0.;
        std::array<std::vector<double>,2> sum,maximum;
        for(auto* arrays:{&sum,&maximum})for(auto& v:*arrays)v.resize(mesh);
        std::vector<Complex> density(2*mesh);
        Digest digest;digest.text("manta.ri-exchange-fitting-bound.v1;unweighted-spin-absolute-family;full-coulomb;q0-omitted");
        if(r.kernel!="coulomb"){digest.text(r.kernel);digest.real(r.physical_omega);}
        digest.text(r.ri_source_identity);digest.text(r.reference_source_identity);digest.real(o.density_l1_limit);
        for(double x:o.density_entry_limits)digest.real(x);
        auto probe=[&](Count k,Count i,Count j,bool imaginary) {
            std::fill(density.begin(),density.end(),Complex{});
            density[k*n2+i*n+j]=imaginary?Complex(0.,1.):Complex(1.);
            density[k*n2+j*n+i]=std::conj(density[k*n2+i*n+j]);
            const auto fit=ri.build_exchange(density,true);
            const std::vector<Complex> one(density.begin(),density.begin()+mesh);
            const auto exact=compute_periodic_exchange(source.basis_,source.lattice_,vectors,offsets,translations,source.momenta_,source.weights_,one,1,o.reciprocal_batch,source.options_.pair_cutoff_bohr,leaf,ri.info().exchange_kernel,ri.info().physical_omega,true);
            if(r.finite_reference_source_identity.empty())r.finite_reference_source_identity=exact.receipt.source_identity;
            else if(r.finite_reference_source_identity!=exact.receipt.source_identity)throw std::logic_error("MANTA exchange fitting reference changed during probing");
            const auto remainder=tail.build(one,1,leaf);++r.probes;
            r.ri_shell_calls=add(r.ri_shell_calls,fit.receipt.shell_calls);r.reference_shell_calls=add(r.reference_shell_calls,exact.receipt.shell_calls);
            if(!fit.receipt.fitted_exchange_roundoff_evaluated || !exact.receipt.roundoff_evaluated)throw std::logic_error("MANTA exchange fitting probe lacks arithmetic enclosure");
            std::array<std::vector<double>,2> errors;for(auto& v:errors)v.resize(mesh);
            for(Count q=0;q<mesh;++q) {
                const auto delta=spectral_interval::sub(spectral_interval::point(fit.K[q]),spectral_interval::point(exact.values[q]));
                errors[0][q]=plus_up(spectral_interval::magnitude(delta),plus_up(fit.K_roundoff_bounds[q],exact.roundoff_bounds[q]));
                errors[1][q]=remainder.error_bound[q];
                for(int c=0;c<2;++c)digest.real(errors[c][q]);
            }
            return errors;
        };
        for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=i;j<n;++j) {
            auto real=probe(k,i,j,false);std::array<std::vector<double>,2> imaginary;
            if(i!=j)imaginary=probe(k,i,j,true);
            const double limit=o.density_entry_limits.empty()?0.:o.density_entry_limits[k*n2+i*n+j];
            for(int c=0;c<2;++c)for(Count q=0;q<mesh;++q) {
                const double coefficient=i==j?real[c][q]:periodic_error::upper(std::hypot(static_cast<long double>(real[c][q]),static_cast<long double>(imaginary[c][q])));
                maximum[c][q]=std::max(maximum[c][q],times_up(i==j?1.:.5,coefficient));
                if(!o.density_entry_limits.empty())sum[c][q]=plus_up(sum[c][q],times_up(limit,coefficient));
            }
        }
        r.error_bounds.resize(mesh);r.finite_difference_bounds.resize(mesh);r.reference_remainder_bounds.resize(mesh);
        std::array<double,2> energy{},max_error{};
        for(int c=0;c<2;++c) {
            auto& bounds=c==0?r.finite_difference_bounds:r.reference_remainder_bounds;
            double entry_energy=0.,weighted_max=0.;
            for(Count q=0;q<mesh;++q) {
                double value=times_up(maximum[c][q],o.density_l1_limit);
                if(!o.density_entry_limits.empty())value=std::min(value,sum[c][q]);
                if(!std::isfinite(value))throw std::runtime_error("MANTA exchange fitting bound is not finite");
                bounds[q]=value;max_error[c]=std::max(max_error[c],value);
                const double weighted=times_up(source.weights_[q/n2],value);weighted_max=std::max(weighted_max,weighted);
                if(!o.density_entry_limits.empty())entry_energy=plus_up(entry_energy,times_up(o.density_entry_limits[q],weighted));
            }
            energy[c]=times_up(.5,times_up(o.density_l1_limit,weighted_max));
            if(!o.density_entry_limits.empty())energy[c]=std::min(energy[c],times_up(.5,entry_energy));
        }
        r.finite_difference_maximum=max_error[0];r.reference_remainder_maximum=max_error[1];
        r.finite_difference_energy_bound=energy[0];r.reference_remainder_energy_bound=energy[1];r.energy_error_bound=plus_up(energy[0],energy[1]);
        for(Count q=0;q<mesh;++q){r.error_bounds[q]=plus_up(r.finite_difference_bounds[q],r.reference_remainder_bounds[q]);r.maximum_error=std::max(r.maximum_error,r.error_bounds[q]);digest.real(r.error_bounds[q]);}
        if(!std::isfinite(r.maximum_error)||!std::isfinite(r.energy_error_bound))throw std::runtime_error("MANTA exchange fitting bound is not finite");
        r.qualified=!r.targets_requested || (r.maximum_error<=o.matrix_error && r.energy_error_bound<=o.energy_error);
        if(!r.qualified)throw std::runtime_error("MANTA exchange fitting targets unmet");
        digest.text(r.finite_reference_source_identity);digest.real(r.energy_error_bound);r.certificate_identity=digest.value.finish_hex();
        cap(add(r.ri_shell_calls,r.reference_shell_calls),r.plan.shell_calls,"MANTA exchange fitting executed shell bound exceeded");return r;
    }
};
PeriodicRIExchangeFittingPlan plan_periodic_ri_exchange_fitting_capacity(const BasisSet& basis,
    const PeriodicRIFockPlan& ri,Count cells,Count vectors,const PeriodicPairOptions& pairs,
    const PeriodicRIExchangeFittingOptions& o,Count retained_fitting,const ResourceBudget& b) {
    margin(b);
    if(!ri.has_k || ri.n_basis!=basis.nbasis() || !ri.momenta)
        throw std::invalid_argument("MANTA exchange fitting capacity requires matching basis and a K plan");
    validate(o,mul(ri.momenta,square(ri.n_basis)),ri.n_basis);
    const auto leaf=unlimited(b);
    const auto pair=plan_periodic_pairs(basis,cells,ri.momenta,pairs,leaf);
    const auto tail=PeriodicExchangeBoundSource::plan_source_capacity(basis,pair.maximum_pairs,ri.momenta,
        pair.retained_bytes,vectors,o.reciprocal_completion_cube_radius,1,o.ao_completion_cube_radius,leaf,o.fourier_decay_bins);
    const auto exact=plan_periodic_exchange(basis,vectors,cells,ri.momenta,1,o.reciprocal_batch,pairs.pair_cutoff_bohr,leaf,"coulomb",0.,true);
    return PeriodicRIExchangeFittingDriver::aggregate(ri,tail,exact,pair.retained_bytes,cells,vectors,retained_fitting,&pair,o,b);
}
PeriodicRIExchangeFittingPlan plan_periodic_ri_exchange_fitting(const PeriodicRIFockBuilder& r,const PeriodicPairSource& s,Count n,const PeriodicRIExchangeFittingOptions& o,const ResourceBudget& b) {return PeriodicRIExchangeFittingDriver::plan(r,s,n,o,b);}
PeriodicRIExchangeFittingResult certify_periodic_ri_exchange_fitting(const PeriodicRIFockBuilder& r,const PeriodicPairSource& s,const md::Matrix& v,const std::vector<Count>& offsets,const PeriodicRIExchangeFittingOptions& o,const ResourceBudget& b) {return PeriodicRIExchangeFittingDriver::run(r,s,v,offsets,o,b);}
namespace {
PeriodicPairOptions accuracy_pairs() {PeriodicPairOptions o;o.order=0;o.extents.moment_order=0;return o;}
PeriodicRIExchangeFittingOptions accuracy_options(const PeriodicRIExchangeAccuracyOptions& o) {
    if(!std::isfinite(o.matrix_error)||!std::isfinite(o.energy_error)||o.matrix_error<=0.||o.energy_error<=0.)
        throw std::invalid_argument("MANTA fitted exchange accuracy requires positive finite matrix and energy targets");
    if(o.ao_completion_cube_radius<o.reference_ao_cube_radius)
        throw std::invalid_argument("MANTA fitted exchange completion must contain the reference images");
    PeriodicRIExchangeFittingOptions a;a.density_l1_limit=1.;a.matrix_error=o.matrix_error;a.energy_error=o.energy_error;
    a.metric_assembly_matrix_error=o.metric_assembly_matrix_error;a.metric_assembly_energy_error=o.metric_assembly_energy_error;
    a.maximum_probes=o.maximum_probes;a.reciprocal_batch=o.reciprocal_batch;a.fourier_decay_bins=o.fourier_decay_bins;
    a.ao_completion_cube_radius=o.ao_completion_cube_radius;a.reciprocal_completion_cube_radius=o.reference_reciprocal_cube_radius;return a;
}
}
PeriodicRIExchangeFittingPlan plan_periodic_ri_exchange_accuracy(const BasisSet& basis,const PeriodicRIFockPlan& ri,
    const PeriodicRIExchangeAccuracyOptions& o,Count retained_fitting,const ResourceBudget& b,const PeriodicGDFKernelOptions& physical) {
    if((physical.kernel!="coulomb" && physical.kernel!="erf" && physical.kernel!="erfc") || !std::isfinite(physical.omega) ||
       (physical.kernel=="coulomb"?physical.omega!=0.:physical.omega<=0.))
        throw std::invalid_argument("MANTA fitted exchange accuracy requires an admitted physical kernel/range");
    const auto a=accuracy_options(o);
    const Count images=lattice_cube::cube_cells(o.reference_ao_cube_radius),labels=lattice_cube::cube_cells(o.reference_reciprocal_cube_radius);
    const Count vectors=mul(square(ri.momenta),labels)-(physical.kernel=="erfc"?0:ri.momenta);
    auto p=plan_periodic_ri_exchange_fitting_capacity(basis,ri,images,vectors,accuracy_pairs(),a,retained_fitting,unlimited(b));
    // Labels, Q/offset owners, family conversion and returned certificate
    // arrays coexist with the sources; enumerate only after aggregate caps.
    const Count generation=add(32768,mul(1024,add(add(images,labels),add(vectors,mul(ri.momenta,square(ri.n_basis))))));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(b.other_live_bytes,generation));
    p.scalar_terms=add(p.scalar_terms,generation);p.kernel_terms=add(p.kernel_terms,mul(32,generation));
    // SHA-256 family tag, dimensions, L1 bound and weighted entries. Keep
    // the automatic canonical family distinct from the raw probe family.
    const Count identity_work=mul(4096,add(mul(ri.momenta,square(ri.n_basis)),8));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,512);
    p.scalar_terms=add(p.scalar_terms,identity_work);p.kernel_terms=add(p.kernel_terms,mul(8,identity_work));
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA fitted exchange accuracy memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA fitted exchange accuracy shell budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA fitted exchange accuracy primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA fitted exchange accuracy scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA fitted exchange accuracy kernel budget exceeded");return p;
}
PeriodicRIExchangeFittingResult bound_periodic_ri_exchange_accuracy(const BasisSet& basis,const PeriodicSystem& system,
    const md::Matrix& k,const PeriodicRIFockBuilder& ri,const BoundedPeriodicDensityEnvelope& density,
    const PeriodicRIExchangeAccuracyOptions& o,Count retained_fitting,const ResourceBudget& b,bool enforce_targets) {
    const PeriodicGDFKernelOptions physical{ri.info().exchange_kernel,ri.info().physical_omega};
    const auto plan=plan_periodic_ri_exchange_accuracy(basis,ri.plan(),o,retained_fitting,b,physical);
    const Count nk=plan.momenta,n=plan.n_basis,n2=square(n),mesh=mul(nk,n2);
    if(k.rows()!=Eigen::Index(nk)||k.cols()!=3||!k.allFinite()||density.memory.n_kpoints!=nk||density.memory.n_basis!=n||density.weighted_entry_bounds.size()!=mesh)
        throw std::invalid_argument("MANTA fitted exchange density envelope frame mismatch");
    auto a=accuracy_options(o);if(!enforce_targets)a.matrix_error=a.energy_error=0.;a.density_l1_limit=periodic_error::upper(static_cast<long double>(nk)*density.density_l1_limit);
    a.density_entry_limits.resize(mesh);
    for(Count q=0;q<mesh;++q){const Count local=q%n2,t=q-local+(local%n)*n+local/n;
        a.density_entry_limits[q]=periodic_error::upper(static_cast<long double>(nk)*std::max(density.weighted_entry_bounds[q],density.weighted_entry_bounds[t]));}
    const auto images=lattice_cube::cube(o.reference_ao_cube_radius),labels=lattice_cube::cube(o.reference_reciprocal_cube_radius);
    const auto leaf=unlimited(b);std::vector<double> weights(nk,1./nk);
    PeriodicPairSource pairs(basis,system.lattice,images,k,weights,accuracy_pairs(),leaf);
    const Count vectors=mul(square(nk),labels.size())-(physical.kernel=="erfc"?0:nk);md::Matrix q(vectors,3);std::vector<Count> offsets(add(square(nk),1));Count row=0;
    const auto reciprocal=system.reciprocal_lattice();
    for(Count i=0;i<nk;++i)for(Count j=0;j<nk;++j){
        for(const auto& label:labels){if(i==j && label==LatticeIndex{} && physical.kernel!="erfc")continue;
            for(int axis=0;axis<3;++axis){long double x=0.;for(int c=0;c<3;++c)x+=static_cast<long double>(reciprocal(axis,c))*label[c];
                q(row,axis)=double(x)+k(i,axis)-k(j,axis);}++row;}
        offsets[i*nk+j+1]=row;}
    auto result=certify_periodic_ri_exchange_fitting(ri,pairs,q,offsets,a,leaf);
    for(auto field:{&PeriodicRIExchangeFittingPlan::modeled_peak_bytes,&PeriodicRIExchangeFittingPlan::shell_calls,
        &PeriodicRIExchangeFittingPlan::primitive_products,&PeriodicRIExchangeFittingPlan::scalar_terms,&PeriodicRIExchangeFittingPlan::kernel_terms})
        if(result.plan.*field>plan.*field)throw std::logic_error("MANTA fitted exchange certificate exceeds preparation capacity");
    // Same original weighted family as the J, kinetic and nuclear ledgers.
    // Hash before outward spin-family conversion; do not alter the existing
    // numerical probe certificate or source identity (contract 5.47).
    Digest family;family.text("manta.canonical-weighted-density-envelope.v1");
    family.word(nk);family.word(n);family.real(density.density_l1_limit);
    for(double x:density.weighted_entry_bounds)family.real(x);
    result.canonical_density_envelope_identity=family.value.finish_hex();
    result.plan=plan;return result;
}
void PeriodicRIExchangeFittingResult::validate_density(const std::vector<Complex>& d,Count spins) const {
    const Count n=plan.n_basis,n2=square(n),mesh=mul(plan.momenta,n2);
    if((spins!=1 && spins!=2) || d.size()!=mul(spins,mesh))throw std::invalid_argument("MANTA exchange fitting density extent mismatch");
    long double total=0.;
    for(Count q=0;q<mesh;++q) {
        double entry=0.;const Count local=q%n2,transpose=q-local+(local%n)*n+local/n;
        for(Count s=0;s<spins;++s){const auto z=d[s*mesh+q];if(!std::isfinite(z.real())||!std::isfinite(z.imag())||z!=std::conj(d[s*mesh+transpose]))throw std::invalid_argument("MANTA exchange fitting requires exactly Hermitian finite densities");entry=plus_up(entry,norm(z));}
        total+=entry;if(!density_entry_limits.empty() && entry>density_entry_limits[q])throw std::invalid_argument("MANTA exchange fitting density exceeds entry family");
    }
    if(periodic_error::upper(total)>density_l1_limit)throw std::invalid_argument("MANTA exchange fitting density exceeds L1 family");
}
} // namespace vibeqc::manta
