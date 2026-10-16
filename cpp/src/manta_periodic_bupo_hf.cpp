#include "vibeqc/manta_periodic_bupo_hf.hpp"
#include "manta_periodic_hf_source.hpp"
#include "manta_pair_image_internal.hpp"
#include "manta_periodic_bupo_hf_inputs.hpp"

namespace vibeqc::manta {
namespace {
using namespace admission;
Count populations(const BasisSet& basis,const PeriodicSystem& system,const PeriodicBupoHFOptions& o) {
    if(system.dim!=3 || system.charge!=0 || system.unit_cell.empty() || system.multiplicity<1
       || (o.spin_mode!="restricted" && o.spin_mode!="unrestricted")
       || (o.spin_mode=="restricted" && system.multiplicity!=1))
        throw std::invalid_argument("MANTA BUPO HF requires a neutral all-electron 3D system and a compatible restricted/unrestricted spin mode");
    for(auto x:{o.charge_tolerance,o.maximum_electrostatic_energy_error,o.maximum_electrostatic_matrix_error,
                o.maximum_overlap_image_error,o.maximum_kinetic_image_error,o.maximum_kinetic_energy_error,
                o.maximum_exchange_matrix_error,o.maximum_exchange_energy_error,
                o.confirmation_fock_tolerance,o.confirmation_energy_tolerance})
        if(!std::isfinite(x) || x<=0)throw std::invalid_argument("MANTA BUPO HF tolerances must be finite positive");
    if((o.exchange_q0!="strict-zero-mode" && o.exchange_q0!="BvK-ewald")
       || !std::isfinite(o.madelung_tolerance) || o.madelung_tolerance<=0 || o.madelung_tolerance>=1
       || (o.madelung_alpha && (!std::isfinite(*o.madelung_alpha) || *o.madelung_alpha<=0)))
        throw std::invalid_argument("MANTA BUPO HF exchange zero-mode and Madelung settings are invalid");
    Count ne=0;
    for(const auto& atom:system.unit_cell) {
        if(atom.Z<=0)throw std::invalid_argument("MANTA BUPO HF requires positive all-electron nuclear charges");
        for(auto x:atom.xyz)require_finite(x);
        ne=add(ne,atom.Z);
    }
    const Count unpaired=system.multiplicity-1;
    if(unpaired>ne || (ne-unpaired)%2 || add(ne,unpaired)/2>=basis.nbasis())
        throw std::invalid_argument("MANTA BUPO HF populations need compatible spin counts and at least one virtual per spin");
    return ne;
}
PeriodicElectrostaticOptions charge_options(const PeriodicBupoHFOptions& o) {
    PeriodicElectrostaticOptions result;result.charge_tolerance=o.charge_tolerance;return result;
}
struct Adapter {
    const PeriodicElectrostaticSource& electrostatic;
    const BasisSet& basis;const md::Matrix& lattice;const md::Matrix& q;const std::vector<Count>& offsets;
    const md::Matrix& cells;const md::Matrix& k;const std::vector<double>& weights;
    const std::vector<Complex>& overlap;const std::vector<Complex>& h;
    const PeriodicOneElectronImageResult& images;
    const PeriodicExchangeBoundSource* exchange_bounds;
    std::optional<double> cutoff;const PeriodicBupoHFOptions& options;
    ResourceBudget leaf;PeriodicBupoHFReceipt& receipt;Count spins,pair_primitives;
    double trace(const std::vector<Complex>& d,const std::vector<Complex>& a,Count channel=0) const {
        const Count n=basis.nbasis(),elements=h.size();Sum sum;
        for(Count ik=0;ik<weights.size();++ik)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)
            sum.add(weights[ik]*std::real(d[channel*elements+(ik*n+j)*n+i]*a[(ik*n+i)*n+j]));
        return sum.value;
    }
    std::vector<Complex> build(const Complex* data,Count size) {
        const Count elements=h.size(),n=basis.nbasis(),n2=square(n);
        if(size!=mul(spins,elements))throw std::logic_error("MANTA BUPO HF callback spin extent mismatch");
        std::vector<Complex> d(data,data+size),total(elements);
        for(Count i=0;i<elements;++i)total[i]=d[i]+(spins==2?d[elements+i]:Complex{});
        receipt.kinetic_energy_error_bound=pair_image_detail::density_trace_bound(images.kinetic_error_bound,total,weights);
        if(!std::isfinite(receipt.kinetic_energy_error_bound)
           || receipt.kinetic_energy_error_bound>options.maximum_kinetic_energy_error)
            throw std::invalid_argument("MANTA BUPO HF kinetic image energy accuracy target failed; enlarge the source or change the declared target");
        receipt.maximum_observed_kinetic_energy_error=std::max(receipt.maximum_observed_kinetic_energy_error,receipt.kinetic_energy_error_bound);
        auto es=electrostatic.build(total);
        // Fixed domains and partition: fail the requested accuracy, never
        // silently repartition according to the evolving SCF density.
        if(es.receipt.energy_error_bound>options.maximum_electrostatic_energy_error
           || es.receipt.maximum_j_error_bound>options.maximum_electrostatic_matrix_error)
            throw std::invalid_argument("MANTA BUPO HF electrostatic accuracy target failed; enlarge source/Ewald domains or change the declared target");
        const double scale=spins==1?.5:1.;
        if(exchange_bounds) {
            auto bounds=exchange_bounds->build(d,spins,leaf);scale_exchange_bound(bounds,scale);
            receipt.exchange_bounds=bounds.receipt;
            if(options.exchange_q0=="BvK-ewald") {
                const auto correction=bound_bvk_overlap_correction(images,overlap,d,spins,weights,scale*receipt.madelung);
                receipt.exchange_correction_matrix_error=*std::max_element(correction.begin(),correction.end());
                receipt.exchange_correction_energy_error=weighted_exchange_energy_bound(correction,d,n,weights);
                for(Count i=0;i<size;++i)bounds.error_bound[i]=positive_bound_sum(bounds.error_bound[i],correction[i]);
            }
            receipt.exchange_matrix_error_bound=*std::max_element(bounds.error_bound.begin(),bounds.error_bound.end());
            receipt.exchange_energy_error_bound=weighted_exchange_energy_bound(bounds.error_bound,d,n,weights);
            if(receipt.exchange_matrix_error_bound>options.maximum_exchange_matrix_error
               || receipt.exchange_energy_error_bound>options.maximum_exchange_energy_error)
                throw std::invalid_argument("MANTA BUPO HF exchange accuracy target failed; enlarge AO/reciprocal support or change the declared target");
            receipt.maximum_observed_exchange_matrix_error=std::max(receipt.maximum_observed_exchange_matrix_error,receipt.exchange_matrix_error_bound);
            receipt.maximum_observed_exchange_energy_error=std::max(receipt.maximum_observed_exchange_energy_error,receipt.exchange_energy_error_bound);
            receipt.exchange_accuracy_passed=true;
        }
        auto exchange=compute_periodic_exchange(basis,lattice,q,offsets,cells,k,weights,d,spins,
            options.reciprocal_batch,cutoff,leaf);
        // HF spin functional: Helgaker/Jorgensen/Olsen (2000), Ch.10;
        // Gaussian Bloch exchange: Sun et al. (2017), Sec.II,
        // DOI 10.1063/1.4998644. Periodic BUPO is the project derivation
        // in scientific-contract Sections 5.7-5.11, not a published HF route.
        std::vector<Complex> f(size);
        for(Count s=0;s<spins;++s)for(Count i=0;i<elements;++i)
            f[s*elements+i]=h[i]+es.electronic.values[i]-scale*exchange.values[s*elements+i];
        Sum ex;for(auto value:exchange.receipt.energies)ex.add(scale*value);
        receipt.uncorrected_exchange_energy=ex.value;Sum correction;
        if(options.exchange_q0=="BvK-ewald") {
            // Sundararaman & Arias (2013), Sec.II.A, DOI10.1103/PhysRevB.87.165122.
            // Full reciprocal exchange: no extra erfc finite-zero subtraction.
            using Matrix=Eigen::Matrix<Complex,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
            for(Count s=0;s<spins;++s)for(Count ik=0;ik<weights.size();++ik) {
                const Count offset=ik*n2,block=s*elements+offset;
                const Eigen::Map<const Matrix> S(overlap.data()+offset,n,n),D(d.data()+block,n,n);
                const Matrix v=-scale*receipt.madelung*(S*D*S).eval();
                for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
                    f[block+i*n+j]+=v(i,j);require_finite(f[block+i*n+j].real());require_finite(f[block+i*n+j].imag());
                    correction.add(.5*weights[ik]*std::real(d[block+j*n+i]*v(i,j)));
                }
            }
        }
        receipt.exchange_correction_energy=correction.value;receipt.exchange_energy=ex.value+correction.value;
        receipt.one_electron_energy=trace(total,h);receipt.coulomb_energy=es.receipt.electronic_energy;
        receipt.energy=receipt.nuclear_energy+receipt.one_electron_energy+receipt.coulomb_energy+receipt.exchange_energy;
        require_finite(receipt.energy);receipt.electrostatic=es.receipt;receipt.exchange=exchange.receipt;
        receipt.coulomb=es.electronic.receipt;
        receipt.maximum_observed_electrostatic_energy_error=std::max(receipt.maximum_observed_electrostatic_energy_error,es.receipt.energy_error_bound);
        receipt.maximum_observed_electrostatic_matrix_error=std::max(receipt.maximum_observed_electrostatic_matrix_error,es.receipt.maximum_j_error_bound);
        receipt.electrostatic_accuracy_passed=true;++receipt.fock_calls;
        receipt.shell_calls=add(receipt.shell_calls,add(es.electronic.receipt.direct.real_shell_calls,exchange.receipt.shell_calls));
        // The full admitted counters include all domain/envelope work. Here
        // record actual native integral calls from the electronic/exchange receipts.
        receipt.shell_calls=add(receipt.shell_calls,add(es.electronic.receipt.direct.fourier_shell_calls,es.electronic.receipt.direct.overlap_shell_calls));
        receipt.shell_calls=add(receipt.shell_calls,add(es.electronic.receipt.forward_shell_calls,es.electronic.receipt.adjoint_shell_calls));
        receipt.primitive_pairs=add(receipt.primitive_pairs,add(mul(2,pair_primitives),add(es.electronic.receipt.direct.primitive_products,exchange.receipt.primitive_pairs)));
        Digest source;source.text("manta.periodic-bupo-hf.v1;matched-electrostatics;finite-exact-K");
        source.text(electrostatic.source_identity());source.text(exchange.receipt.source_identity);
        source.text(receipt.kinetic_identity);source.text(receipt.exchange_correction_identity);
        source.text(options.spin_mode);source.word(receipt.n_alpha);source.word(receipt.n_beta);
        receipt.source_identity=source.hash.finish_hex();
        Digest density;density.text(receipt.source_identity);density.text(exchange.receipt.density_identity);
        receipt.final_density_identity=density.hash.finish_hex();
        Digest snapshot;snapshot.text(receipt.source_identity);snapshot.text(receipt.final_density_identity);
        for(auto value:f)snapshot.value(value);snapshot.real(receipt.energy);
        receipt.final_fock_identity=snapshot.hash.finish_hex();
        return f;
    }
    static void evaluate(const Complex* d,std::size_t dn,Complex* out,std::size_t on,void* pointer) {
        auto& a=*static_cast<Adapter*>(pointer);if(dn!=on)throw std::logic_error("MANTA BUPO HF output extent mismatch");
        const auto f=a.build(d,dn);for(Count i=0;i<on;++i)out[i]=f[i]-a.h[i%a.h.size()];
    }
};
template<class Snapshot> struct Trace {
    std::vector<Snapshot>& rows;Count capacity;
    static bool append(const Snapshot& snapshot,void* pointer) {
        auto& trace=*static_cast<Trace*>(pointer);
        if(trace.rows.size()>=trace.capacity || snapshot.evaluated_iteration!=trace.rows.size()+1)
            throw std::logic_error("MANTA BUPO HF history exceeded its admitted extent");
        trace.rows.push_back(snapshot);return true;
    }
};
} // namespace

void bupo_hf_detail::validate_inputs(const BasisSet& basis,const PeriodicSystem& system,const PeriodicBupoHFOptions& options) {
    populations(basis,system,options);
    validate_centers(basis,system);
}

struct PeriodicBupoHFDriver {
    static const PeriodicPairSource& pairs(const PeriodicBupoSource& source) { return *source.direct_->source_; }
    static PeriodicBupoHFPlan plan(const PeriodicBupoSource& source,const PeriodicSystem& system,const RegularKMesh& mesh,
                                  Count nq,const PeriodicBupoHFOptions& o,const ResourceBudget& budget) {
        margin(budget);const auto& s=pairs(source);const auto leaf=unlimited(budget);
        PeriodicBupoHFPlan p;p.electrons=populations(s.basis_,system,o);p.n_basis=s.basis_.nbasis();p.momenta=mesh.size();
        p.spins=o.spin_mode=="restricted"?1:2;p.n_beta=(p.electrons-(system.multiplicity-1))/2;p.n_alpha=p.electrons-p.n_beta;
        p.cells=s.images_.size();
        if(p.momenta!=s.plan_.momenta || (system.lattice.array()!=s.lattice_.array()).any())
            throw std::invalid_argument("MANTA BUPO HF requires the original source lattice and uniform full mesh");
        for(Count ik=0;ik<p.momenta;++ik) {
            const auto expected=system.reciprocal_lattice()*mesh.fractional_at(ik);
            if(std::abs(s.weights_[ik]-1./p.momenta)>1e-14
               || (s.momenta_.row(ik).transpose()-expected).norm()>1e-12*std::max(1.,expected.norm()))
                throw std::invalid_argument("MANTA BUPO HF requires source momenta/weights in original uniform full-mesh order");
        }
        validate_centers(s.basis_,system);
        p.electrostatic=PeriodicElectrostaticSource::plan_source(source,system.unit_cell.size(),charge_options(o),leaf);
        p.one_electron_images=plan_periodic_one_electron_images(s,leaf);
        p.kinetic=plan_bloch_one_electron(s.basis_,"kinetic",p.cells,p.momenta,s.options_.pair_cutoff_bohr,leaf);
        p.exchange=plan_periodic_exchange(s.basis_,nq,p.cells,p.momenta,p.spins,o.reciprocal_batch,s.options_.pair_cutoff_bohr,leaf);
        if(o.bound_exchange)p.exchange_bounds=PeriodicExchangeBoundSource::plan_source(s,mul(nq,square(p.momenta)),o.exchange_bound_cube_radius,p.spins,leaf);
        p.madelung=plan_madelung(system,mesh,o);
        const Count elements=mul(p.momenta,square(p.n_basis)),correction=mul(512,mul(elements,mul(p.n_basis,p.spins)));
        p.provider_kernel_terms_per_call=add(p.exchange_bounds.kernel_terms,add(p.electrostatic.kernel_terms,add(p.exchange.kernel_terms,correction)));
        const Count provider_bytes=add(mul(2,p.exchange_bounds.modeled_peak_bytes),add(p.electrostatic.modeled_peak_bytes,p.exchange.modeled_peak_bytes));
        BoundedPeriodicRHFTwoElectronProvider provider{Adapter::evaluate,nullptr,0,provider_bytes,p.provider_kernel_terms_per_call};
        if(p.spins==1)static_cast<BoundedPeriodicRHFMemoryPlan&>(p.solver)=plan_bounded_periodic_rhf_solver(p.momenta,p.n_basis,p.electrons,p.n_basis,provider,o.scf,0);
        else p.solver=plan_bounded_periodic_uhf_solver(p.momenta,p.n_basis,p.n_alpha,p.n_beta,p.n_basis,provider,o.scf,0);
        p.solver.n_alpha=p.n_alpha;p.solver.n_beta=p.n_beta;
        p.history_capacity=o.scf.maximum_iterations;
        p.history_live_bytes=mul(p.history_capacity,add(mul(3,sizeof(BoundedPeriodicUHFSnapshot)),256));
        p.driver_live_bytes=add(65536,add(p.history_live_bytes,add(mul(1024,elements),add(mul(128,p.cells),mul(256,system.unit_cell.size())))));
        p.modeled_peak_bytes=add(p.one_electron_images.modeled_peak_bytes,add(budget.other_live_bytes,add(p.driver_live_bytes,add(p.solver.peak_owned_numerical_bytes,
            add(provider_bytes,add(p.kinetic.modeled_peak_bytes,p.madelung.modeled_peak_bytes))))));
        p.fock_calls_upper_bound=add(o.scf.maximum_iterations,1);
        p.shell_calls=add(p.one_electron_images.shell_calls,add(p.kinetic.shell_calls,mul(p.fock_calls_upper_bound,add(p.electrostatic.shell_calls,p.exchange.shell_calls))));
        p.primitive_pairs=add(p.one_electron_images.primitive_pairs,add(p.kinetic.primitive_pairs,mul(p.fock_calls_upper_bound,add(p.electrostatic.primitive_products,p.exchange.primitive_pairs))));
        p.scalar_terms=add(p.one_electron_images.scalar_terms,add(p.kinetic.scalar_terms,add(p.solver.maximum_work_units,mul(p.fock_calls_upper_bound,
            add(correction,add(p.electrostatic.scalar_terms,p.exchange.scalar_terms))))));
        p.kernel_terms=add(p.one_electron_images.kernel_terms,add(p.kinetic.kernel_terms,add(p.madelung.kernel_terms,add(p.solver.maximum_work_units,p.provider_kernel_terms_per_call))));
        p.shell_calls=add(p.shell_calls,p.exchange_bounds.shell_calls);p.primitive_pairs=add(p.primitive_pairs,p.exchange_bounds.primitive_pairs);
        p.scalar_terms=add(p.scalar_terms,mul(add(1,p.fock_calls_upper_bound),p.exchange_bounds.scalar_terms));
        p.kernel_terms=add(p.kernel_terms,p.exchange_bounds.kernel_terms);
        cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA BUPO HF aggregate memory budget exceeded");
        cap(p.shell_calls,budget.shell_calls,"MANTA BUPO HF aggregate shell-call budget exceeded");
        cap(p.primitive_pairs,budget.primitive_quartets,"MANTA BUPO HF aggregate primitive budget exceeded");
        cap(p.scalar_terms,budget.scalar_terms,"MANTA BUPO HF aggregate scalar budget exceeded");
        cap(p.kernel_terms,budget.kernel_terms,"MANTA BUPO HF aggregate kernel budget exceeded");return p;
    }
    static PeriodicBupoHFResult run(std::shared_ptr<const PeriodicBupoSource> source,const PeriodicSystem& system,const RegularKMesh& mesh,
        const md::Matrix& q,const std::vector<Count>& offsets,const PeriodicBupoHFOptions& options,const ResourceBudget& budget) {
        if(!source)throw std::invalid_argument("MANTA BUPO HF requires a native electronic source");
        const auto o=options;const Count nk=mesh.size(),channels=square(nk);
        if(offsets.size()!=add(channels,1) || offsets.front()!=0 || offsets.back()!=Count(q.rows()))
            throw std::invalid_argument("MANTA BUPO HF offsets must delimit every exchange transfer");
        Count nq=0;for(Count i=0;i<channels;++i) {
            if(offsets[i]>offsets[i+1])throw std::invalid_argument("MANTA BUPO HF offsets must be monotone");
            nq=std::max(nq,offsets[i+1]-offsets[i]);
        }
        PeriodicBupoHFResult result;result.plan=plan(*source,system,mesh,nq,o,budget);const auto& p=result.plan;
        const auto& s=pairs(*source);auto& r=result.receipt;const Count elements=mul(nk,square(p.n_basis));const auto leaf=unlimited(budget);
        r.n_alpha=p.n_alpha;r.n_beta=p.n_beta;r.modeled_peak_bytes=p.modeled_peak_bytes;r.bupo_source_identity=source->source_identity();
        r.maximum_electrostatic_energy_error=o.maximum_electrostatic_energy_error;r.maximum_electrostatic_matrix_error=o.maximum_electrostatic_matrix_error;
        r.maximum_overlap_image_error=o.maximum_overlap_image_error;r.maximum_kinetic_image_error=o.maximum_kinetic_image_error;
        r.exchange_bound_cube_radius=o.bound_exchange?o.exchange_bound_cube_radius:0;
        r.maximum_kinetic_energy_error=o.maximum_kinetic_energy_error;
        r.maximum_exchange_matrix_error=o.maximum_exchange_matrix_error;r.maximum_exchange_energy_error=o.maximum_exchange_energy_error;
        if(o.bound_exchange)r.accuracy_qualification="electrostatic-one-electron-and-exchange-at-evaluated-density;fixed-mesh-and-madelung";
        std::optional<PeriodicExchangeBoundSource> exchange_bounds;
        if(o.bound_exchange)exchange_bounds.emplace(s,q,offsets,o.exchange_bound_cube_radius,leaf);
        result.one_electron_images=bound_periodic_one_electron_images(s,leaf);
        const auto& image_bounds=result.one_electron_images.receipt;
        if(image_bounds.maximum_overlap_error>o.maximum_overlap_image_error || image_bounds.maximum_kinetic_error>o.maximum_kinetic_image_error)
            throw std::invalid_argument("MANTA BUPO HF overlap/kinetic image accuracy target failed; enlarge the source or change the declared target");
        r.one_electron_accuracy_passed=true;
        md::Matrix positions(system.unit_cell.size(),3),cells(p.cells,3);std::vector<double> charges(system.unit_cell.size());
        for(Count i=0;i<charges.size();++i) {charges[i]=system.unit_cell[i].Z;for(int j=0;j<3;++j)positions(i,j)=system.unit_cell[i].xyz[j];}
        for(Count i=0;i<p.cells;++i)for(int axis=0;axis<3;++axis) {
            long double value=0;for(int j=0;j<3;++j)value+=static_cast<long double>(s.lattice_(axis,j))*s.images_[i][j];
            cells(i,axis)=static_cast<double>(value);require_finite(cells(i,axis));
        }
        PeriodicElectrostaticSource electrostatic(source,positions,charges,charge_options(o),leaf);
        result.overlap=electrostatic.overlap();r.overlap_identity=s.receipt_.source_identity+":overlap";
        auto kinetic=compute_bloch_one_electron(s.basis_,"kinetic",cells,s.momenta_,s.options_.pair_cutoff_bohr,leaf);
        r.kinetic_identity=kinetic.receipt.source_identity;result.hcore=std::move(kinetic.values);
        for(Count i=0;i<elements;++i)result.hcore[i]+=electrostatic.nuclear_values()[i];
        const auto& nr=electrostatic.receipt_;
        r.nuclear_identity=nr.point_source_identity;r.nuclear_energy_identity=nr.source_identity;
        r.nuclear_real=nr.nuclear_real;r.nuclear_reciprocal=nr.nuclear_reciprocal;r.nuclear_self=nr.nuclear_self;
        r.nuclear_background=nr.nuclear_gauge;r.nuclear_energy=nr.nuclear_energy;r.nuclear_pair_images=nr.point_pair_terms;
        r.shell_calls=add(kinetic.receipt.shell_calls,add(nr.nuclear_shell_calls,add(nr.overlap_shell_calls,nr.fourier_shell_calls)));
        r.primitive_pairs=add(kinetic.receipt.primitive_pairs,mul(s.receipt_.primitive_pairs,
            add(1,add(mul(source->direct_->real_.size(),charges.size()),source->direct_->plan_.reciprocal_batches))));
        compute_madelung(system,mesh,o,p.madelung,r);
        const md::Matrix lattice=s.lattice_;
        Adapter adapter{electrostatic,s.basis_,lattice,q,offsets,cells,s.momenta_,s.weights_,result.overlap,result.hcore,result.one_electron_images,
            exchange_bounds?&*exchange_bounds:nullptr,s.options_.pair_cutoff_bohr,o,leaf,r,p.spins,s.receipt_.primitive_pairs};
        const Count provider_bytes=add(mul(2,p.exchange_bounds.modeled_peak_bytes),add(p.electrostatic.modeled_peak_bytes,p.exchange.modeled_peak_bytes));
        BoundedPeriodicRHFTwoElectronProvider provider{Adapter::evaluate,&adapter,0,provider_bytes,p.provider_kernel_terms_per_call};
        BoundedPeriodicRHFCaps caps;caps.maximum_owned_numerical_bytes=p.solver.peak_owned_numerical_bytes;
        caps.maximum_total_numerical_bytes=p.modeled_peak_bytes;caps.maximum_work_units=p.solver.maximum_work_units;
        caps.maximum_fock_calls=o.scf.maximum_iterations;
        caps.other_live_numerical_bytes=p.modeled_peak_bytes-p.solver.peak_owned_numerical_bytes-provider_bytes-mul(32,elements);
        const Complex *d,*f;double energy=0;bool converged=false;
        if(p.spins==1) {
            result.rhf_history.reserve(p.history_capacity);Trace<BoundedPeriodicRHFSnapshot> trace{result.rhf_history,p.history_capacity};
            result.rhf.emplace(solve_bounded_periodic_rhf({nk,p.n_basis,p.electrons,r.nuclear_energy,result.overlap.data(),elements,result.hcore.data(),elements},provider,o.scf,caps,Trace<BoundedPeriodicRHFSnapshot>::append,&trace));
            d=result.rhf->density_data();f=result.rhf->fock_data();energy=result.rhf->final_snapshot().energy_per_cell;converged=result.rhf->converged();
        } else {
            result.uhf_history.reserve(p.history_capacity);Trace<BoundedPeriodicUHFSnapshot> trace{result.uhf_history,p.history_capacity};
            result.uhf.emplace(solve_bounded_periodic_uhf({nk,p.n_basis,p.n_alpha,p.n_beta,r.nuclear_energy,result.overlap.data(),elements,result.hcore.data(),elements},provider,o.scf,caps,Trace<BoundedPeriodicUHFSnapshot>::append,&trace));
            d=result.uhf->density_data();f=result.uhf->fock_data();energy=result.uhf->final_snapshot().energy_per_cell;converged=result.uhf->converged();
        }
        if(converged) {
            const auto rebuilt=adapter.build(d,mul(p.spins,elements));
            for(Count i=0;i<rebuilt.size();++i)r.confirmation_fock_error=std::max(r.confirmation_fock_error,std::abs(rebuilt[i]-f[i]));
            r.confirmation_energy_error=std::abs(r.energy-energy);
            r.confirmed=r.confirmation_fock_error<=o.confirmation_fock_tolerance && r.confirmation_energy_error<=o.confirmation_energy_tolerance;
        }
        r.domain_translations=p.cells;r.domain_exchange_vectors=q.rows();r.domain_identity=s.receipt_.domain_identity;
        r.domain_selection="native-bupo-pair-images-and-explicit-exchange";return result;
    }
};
PeriodicBupoHFPlan plan_periodic_bupo_hf(const PeriodicBupoSource& source,const PeriodicSystem& system,const RegularKMesh& mesh,
    Count nq,const PeriodicBupoHFOptions& o,const ResourceBudget& budget) {return PeriodicBupoHFDriver::plan(source,system,mesh,nq,o,budget);}
PeriodicBupoHFResult run_periodic_bupo_hf(std::shared_ptr<const PeriodicBupoSource> source,const PeriodicSystem& system,const RegularKMesh& mesh,
    const md::Matrix& q,const std::vector<Count>& offsets,const PeriodicBupoHFOptions& o,const ResourceBudget& budget) {
    return PeriodicBupoHFDriver::run(std::move(source),system,mesh,q,offsets,o,budget);
}
} // namespace vibeqc::manta
