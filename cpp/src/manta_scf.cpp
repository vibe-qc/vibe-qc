#include "vibeqc/manta_scf.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_one_electron.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/xc.hpp"
#include <libint2/engine.h>
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <chrono>
#include <cstring>

namespace vibeqc::manta {
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
std::string matrix_identity(const Eigen::MatrixXd& matrix) {
    vibeqc::detail::Sha256 hash;
    auto word=[&](std::uint64_t value) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(value>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    };
    word(matrix.rows()); word(matrix.cols());
    for(Eigen::Index i=0;i<matrix.rows();++i) for(Eigen::Index j=0;j<matrix.cols();++j) {
        double value=matrix(i,j); if(value==0.) value=0.;
        std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits)); word(bits);
    }
    return hash.finish_hex();
}
std::string hash_text(const std::string& text) {
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(text.data()),text.size()); return hash.finish_hex();
}
std::string molecule_identity(const Molecule& mol) {
    Eigen::MatrixXd atoms(mol.atoms().size(),4);
    for(std::size_t i=0;i<mol.atoms().size();++i) {
        atoms(i,0)=mol.atoms()[i].Z;
        for(int axis=0;axis<3;++axis) atoms(i,axis+1)=mol.atoms()[i].xyz[axis];
    }
    return hash_text("molecule-v1;"+std::to_string(mol.charge())+";"+std::to_string(mol.multiplicity())+";"+matrix_identity(atoms));
}
template<class Options> void validate_options(const Options& opts,const MolecularHFEngine& engine) {
    validate_initial_guess(opts.initial_guess);
    if(engine.backend!="manta" && engine.backend!="libint") throw std::invalid_argument("MANTA molecular engine backend must be manta or libint");
    if((engine.coulomb!="exact" && engine.coulomb!="ri" && engine.coulomb!="bupo" && engine.coulomb!="ri_bupo") || (engine.exchange!="exact" && engine.exchange!="ri"))
        throw std::invalid_argument("MANTA molecular SCF requires exact, ri, bupo or ri_bupo J and exact or ri K");
    if(engine.backend=="libint" && (engine.coulomb!="exact" || engine.exchange!="exact"))
        throw std::invalid_argument("The bounded libint molecular reference currently supports exact J and K only");
    if(opts.initial_guess!=InitialGuess::HCORE && opts.initial_guess!=InitialGuess::READ)
        throw std::invalid_argument("Selected molecular engines currently require explicit HCORE or supplied-density READ guess");
    if(!opts.read_path.empty()) throw std::invalid_argument("Selected molecular engines require an already supplied READ density, not a restart file");
    if(!opts.ecp_centers.empty() || !opts.ecp_library.empty() || !opts.ecp_primitive_blocks.empty()
       || !opts.ecp_primitive_centers.empty() || !opts.ecp_effective_charges.empty() || opts.ecp_total_ncore)
        throw std::invalid_argument("Selected molecular engines currently require an all-electron Hamiltonian without ECP options");
    if(opts.cosx || !opts.dft_plus_u_sites.empty())
        throw std::invalid_argument("Selected molecular SCF engines do not yet admit COSX or DFT+U");
    if(opts.density_fit || !opts.aux_basis.empty())
        throw std::invalid_argument("Use the selected engine's Coulomb/exchange algorithms and auxiliary basis, not legacy density_fit/aux_basis");
    if(opts.scf_mode==SCFMode::CONVENTIONAL)
        throw std::invalid_argument("Selected molecular engines use bounded direct/streamed builds, not conventional AO4 storage");
    if(opts.max_iter<0 || !std::isfinite(opts.conv_tol_energy) || opts.conv_tol_energy<=0
       || !std::isfinite(opts.conv_tol_grad) || opts.conv_tol_grad<=0)
        throw std::invalid_argument("Selected molecular SCF requires nonnegative iterations and finite positive convergence tolerances");
    if(engine.channel_batch<1 || engine.channel_batch>64 || (engine.backend=="libint" && engine.channel_batch!=1))
        throw std::invalid_argument("MANTA channel batch must be 1..64; libint reference requires one");
}
void validate_geometry(const Molecule& mol,const BasisSet& basis) {
    if(mol.atoms().empty()) throw std::invalid_argument("Molecular engine requires a nonempty molecule");
    for(const auto& atom:mol.atoms()) for(auto value:atom.xyz) admission::require_finite(value);
    for(std::size_t i=0;i<mol.atoms().size();++i) for(std::size_t j=0;j<i;++j)
        if(mol.atoms()[i].Z && mol.atoms()[j].Z && mol.atoms()[i].xyz==mol.atoms()[j].xyz)
            throw std::invalid_argument("Selected molecular engine has coincident charged nuclei");
    for(std::size_t i=0;i<basis.nshells();++i) {
        const int owner=basis.shell_atom_index(i);
        if(owner<0 || static_cast<std::size_t>(owner)>=mol.atoms().size() || basis.libint()[i].O!=mol.atoms()[owner].xyz)
            throw std::invalid_argument("Molecular engine basis centers/atom map do not match the molecule");
    }
}
class SelectedJK final : public JKBuilder {
public:
    SelectedJK(const BasisSet& basis,const BasisSet* aux,const MolecularHFEngine& engine,MolecularEngineProvenance& receipt,Count one_electron_peak,const KernelRequest* exchange_range=nullptr)
        : j_ri_(engine.coulomb=="ri"),k_ri_(engine.exchange=="ri"),j_bupo_(engine.coulomb=="bupo"),j_ri_bupo_(engine.coulomb=="ri_bupo"),receipt_(receipt) {
        using namespace admission;
        const bool direct=(!j_ri_ && !j_ri_bupo_) || !k_ri_,ri=j_ri_ || k_ri_;
        if((ri || j_ri_bupo_) && !aux) throw std::invalid_argument("Selected RI algorithm requires an explicit auxiliary basis");
        if(!ri && !j_ri_bupo_ && aux) throw std::invalid_argument("An auxiliary basis requires a selected RI algorithm");
        const auto& budget=engine.budget;
        Count direct_peak=0,ri_peak=0,range_peak=0,bupo_peak=0;
        if(direct) {
            const auto p=engine.backend=="manta"?plan_md(basis,{},budget,engine.channel_batch):plan_libint(basis,{},budget);
            // Paired K includes one extra bounded density/target set.
            direct_peak=add(p.modeled_peak_bytes,add(mul(64,square(basis.nbasis())),512));
        }
        if(ri) ri_peak=plan_ri(basis,*aux,engine.ri,{},budget).modeled_peak_bytes;
        if(exchange_range) {
            if(engine.backend!="manta") throw std::invalid_argument("Selected libint reference does not yet admit range-separated exchange");
            range_request_=*exchange_range;
            range_peak=k_ri_?plan_ri(basis,*aux,engine.ri,*exchange_range,budget).modeled_peak_bytes
                :plan_md(basis,*exchange_range,budget,engine.channel_batch).modeled_peak_bytes;
        }
        provider_peak_=add(add(direct_peak,ri_peak),range_peak);
        Count peak=add(provider_peak_,one_electron_peak);
        cap(peak,budget.memory_bytes,"Selected molecular engine combined J/K memory budget exceeded");
        if(j_bupo_) {
            // Keep the exact provider admitted for the terminal confirmation,
            // including when K itself uses RI. BUPO owns its separate fixed map.
            auto bupo_budget=budget; bupo_budget.other_live_bytes=add(budget.other_live_bytes,peak);
            const auto start=Clock::now();
            bupo_=std::make_unique<BupoJBuilder>(basis,engine.bupo,bupo_budget);
            receipt_.bupo_setup_wall_seconds=seconds(start); receipt_.jk_wall_seconds+=receipt_.bupo_setup_wall_seconds;
            bupo_peak=bupo_->plan().modeled_peak_bytes-peak;
            provider_peak_=add(provider_peak_,bupo_peak); peak=add(provider_peak_,one_electron_peak);
            cap(peak,budget.memory_bytes,"Selected molecular engine combined BUPO memory budget exceeded");
            receipt_.bupo_partition_identity=bupo_->plan().partition_identity;
            receipt_.bupo_near_provider=engine.bupo.near_backend; receipt_.bupo_order=engine.bupo.order;
            receipt_.bupo_integral_tolerance=engine.bupo.integral_tolerance;
            receipt_.bupo_maximum_integral_error=bupo_->plan().maximum_integral_error;
            if(engine.bupo.near_backend=="libint") receipt_.primitive_evaluations_measured=false;
        }
        if(j_ri_bupo_) {
            auto bupo_budget=budget; bupo_budget.other_live_bytes=add(budget.other_live_bytes,peak);
            const auto start=Clock::now();
            ri_bupo_=std::make_unique<BupoRIJBuilder>(basis,*aux,engine.ri_bupo,engine.ri,bupo_budget);
            receipt_.bupo_setup_wall_seconds=seconds(start); receipt_.jk_wall_seconds+=receipt_.bupo_setup_wall_seconds;
            bupo_peak=ri_bupo_->plan().modeled_peak_bytes-peak;
            provider_peak_=add(provider_peak_,bupo_peak); peak=add(provider_peak_,one_electron_peak);
            cap(peak,budget.memory_bytes,"Selected molecular engine combined RI-BUPO memory budget exceeded");
            receipt_.bupo_partition_identity=ri_bupo_->coupling_plan().partition_identity;
            receipt_.bupo_near_provider="manta"; receipt_.bupo_order=engine.ri_bupo.order;
            receipt_.bupo_integral_tolerance=engine.ri_bupo.coupling_tolerance;
            receipt_.bupo_maximum_integral_error=ri_bupo_->coupling_plan().maximum_coupling_error;
            receipt_.ri_bupo_metric_rank=ri_bupo_->metric_info().rank;
            receipt_.ri_bupo_metric_factor_identity=ri_bupo_->metric_info().factor_identity;
            receipt_.jk_shell_calls=add(receipt_.jk_shell_calls,ri_bupo_->metric_info().shell_calls);
            receipt_.primitive_integrals_evaluated=add(receipt_.primitive_integrals_evaluated,ri_bupo_->metric_info().primitive_products);
        }
        receipt_.integral_modeled_peak_bytes=std::max(receipt_.integral_modeled_peak_bytes,peak);
        // Conservatively charge the other provider's entire plan while one
        // is active. No assumption that an eigensolver workspace disappears.
        auto direct_budget=budget,ri_budget=budget,range_budget=budget;
        direct_budget.other_live_bytes=add(budget.other_live_bytes,add(add(ri_peak,range_peak),add(bupo_peak,one_electron_peak)));
        ri_budget.other_live_bytes=add(budget.other_live_bytes,add(add(direct_peak,range_peak),add(bupo_peak,one_electron_peak)));
        range_budget.other_live_bytes=add(budget.other_live_bytes,add(add(direct_peak,ri_peak),add(bupo_peak,one_electron_peak)));
        if(direct) {
            if(engine.backend=="manta") direct_=std::make_unique<DirectJKBuilder>(basis,KernelRequest{},direct_budget,engine.channel_batch);
            else direct_=std::make_unique<ReferenceJKBuilder>(basis,KernelRequest{},direct_budget);
            direct_->plan_batch(2,false,true);
        }
        if(ri) {
            const auto start=Clock::now();
            ri_=std::make_unique<RIJKBuilder>(basis,*aux,engine.ri,KernelRequest{},ri_budget);
            receipt_.jk_wall_seconds+=seconds(start);
            receipt_.jk_shell_calls=add(receipt_.jk_shell_calls,ri_->metric_info().shell_calls);
            receipt_.primitive_integrals_evaluated=add(receipt_.primitive_integrals_evaluated,ri_->metric_info().primitive_products);
        }
        if(exchange_range) {
            const auto start=Clock::now();
            if(k_ri_) {
                range_ri_=std::make_unique<RIJKBuilder>(basis,*aux,engine.ri,*exchange_range,range_budget);
                receipt_.range_exchange_source_identity=range_ri_->source_identity();
                receipt_.jk_shell_calls=add(receipt_.jk_shell_calls,range_ri_->metric_info().shell_calls);
                receipt_.range_exchange_shell_calls=range_ri_->metric_info().shell_calls;
                receipt_.primitive_integrals_evaluated=add(receipt_.primitive_integrals_evaluated,range_ri_->metric_info().primitive_products);
            } else {
                range_direct_=std::make_unique<DirectJKBuilder>(basis,*exchange_range,range_budget,engine.channel_batch);
                receipt_.range_exchange_source_identity=range_direct_->provider().source_identity();
            }
            receipt_.jk_wall_seconds+=seconds(start);
        }
        receipt_.coulomb_source_identity=j_ri_bupo_?ri_bupo_->source_identity():j_bupo_?bupo_->source_identity():j_ri_?ri_->source_identity():direct_->provider().source_identity();
        receipt_.exchange_source_identity=k_ri_?ri_->source_identity():direct_->provider().source_identity();
        receipt_.confirmation_coulomb_algorithm=j_ri_bupo_?"ri":j_bupo_?"exact":engine.coulomb;
        receipt_.confirmation_coulomb_source_identity=j_ri_bupo_?ri_bupo_->exact_ri_source_identity():j_bupo_?direct_->provider().source_identity():receipt_.coulomb_source_identity;
    }
    Count provider_peak() const { return provider_peak_; }
    Eigen::MatrixXd build_J(const Eigen::MatrixXd& d) const override {
        const auto start=Clock::now(); Eigen::MatrixXd matrix;
        if(j_ri_bupo_) { auto r=ri_bupo_->build(d); charge(r.receipt); matrix=std::move(r.J); }
        else if(j_bupo_) { auto r=bupo_->build(d); charge(r.receipt); matrix=std::move(r.J); }
        else if(j_ri_) { auto r=ri_->build(d,true,false); charge(r.receipt); matrix=std::move(r.J); }
        else { auto r=direct_->build(d,true,false); charge(r.receipt); matrix=std::move(r.J); }
        receipt_.jk_wall_seconds+=seconds(start); return matrix;
    }
    Eigen::MatrixXd build_confirmation_J(const Eigen::MatrixXd& d) const {
        receipt_.confirmation_coulomb_build_calls=admission::add(receipt_.confirmation_coulomb_build_calls,1);
        if(j_ri_bupo_) {
            const auto start=Clock::now(); auto result=ri_bupo_->build_exact(d); charge(result.receipt);
            receipt_.jk_wall_seconds+=seconds(start); return std::move(result.J);
        }
        if(!j_bupo_) return build_J(d);
        const auto start=Clock::now(); auto result=direct_->build(d,true,false); charge(result.receipt);
        receipt_.jk_wall_seconds+=seconds(start); return std::move(result.J);
    }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& d) const override {
        const auto start=Clock::now(); Eigen::MatrixXd matrix;
        if(k_ri_) { auto r=ri_->build(d,false,true); charge(r.receipt); matrix=std::move(r.K); }
        else { auto r=direct_->build(d,false,true); charge(r.receipt); matrix=std::move(r.K); }
        receipt_.jk_wall_seconds+=seconds(start); return matrix;
    }
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> build_K_pair(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) const override {
        const auto start=Clock::now(); std::pair<Eigen::MatrixXd,Eigen::MatrixXd> matrices;
        if(k_ri_) {
            auto r=ri_->build_exchange_pair(a,b); charge(r.receipt);
            matrices={std::move(r.K_alpha),std::move(r.K_beta)};
        } else {
            auto r=direct_->build_batch({a,b},false,true); charge(r.receipt);
            matrices={std::move(r.K[0]),std::move(r.K[1])};
        }
        receipt_.jk_wall_seconds+=seconds(start); return matrices;
    }
    Eigen::MatrixXd build_K_erf(const Eigen::MatrixXd& d,double omega) const override {
        if((!range_direct_ && !range_ri_) || omega!=range_request_.omega)
            throw std::invalid_argument("Selected exchange range differs from the admitted source");
        const auto start=Clock::now(); Eigen::MatrixXd matrix;
        const auto calls=receipt_.jk_shell_calls;
        if(range_ri_) { auto r=range_ri_->build(d,false,true); charge(r.receipt); matrix=std::move(r.K); }
        else { auto r=range_direct_->build(d,false,true); charge(r.receipt); matrix=std::move(r.K); }
        receipt_.range_exchange_shell_calls=admission::add(receipt_.range_exchange_shell_calls,receipt_.jk_shell_calls-calls);
        receipt_.range_exchange_build_calls=admission::add(receipt_.range_exchange_build_calls,1);
        receipt_.jk_wall_seconds+=seconds(start); return matrix;
    }
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& d,double fraction=1.) const override {
        if(j_bupo_ || j_ri_bupo_) {
            Eigen::MatrixXd matrix=build_J(d); if(fraction!=0.) matrix-=.5*fraction*build_K(d);
            return matrix;
        }
        if(j_ri_!=k_ri_) return JKBuilder::build_g_rhf(d,fraction);
        const auto start=Clock::now(); Eigen::MatrixXd matrix;
        if(j_ri_) { auto r=ri_->build(d,true,fraction!=0.); charge(r.receipt); matrix=std::move(r.J); if(fraction!=0.) matrix-=.5*fraction*r.K; }
        else { auto r=direct_->build(d,true,fraction!=0.); charge(r.receipt); matrix=std::move(r.J); if(fraction!=0.) matrix-=.5*fraction*r.K; }
        receipt_.jk_wall_seconds+=seconds(start); return matrix;
    }
private:
    void charge(const BupoRIReceipt& r) const {
        using namespace admission;
        receipt_.jk_build_calls=add(receipt_.jk_build_calls,1);
        receipt_.bupo_build_calls=add(receipt_.bupo_build_calls,1);
        for(const auto* pass:{&r.forward,&r.adjoint}) {
            const Count panels=add(pass->pair_moment_panels,pass->auxiliary_moment_panels);
            receipt_.jk_shell_calls=add(receipt_.jk_shell_calls,add(pass->near_shell_calls,panels));
            receipt_.primitive_integrals_evaluated=add(receipt_.primitive_integrals_evaluated,
                add(pass->near_primitive_products,mul(pass->moment_primitive_products,ri_bupo_->coupling_plan().moments)));
            receipt_.bupo_near_shell_calls=add(receipt_.bupo_near_shell_calls,pass->near_shell_calls);
            receipt_.bupo_far_blocks=add(receipt_.bupo_far_blocks,pass->far_blocks);
            receipt_.bupo_moment_panels=add(receipt_.bupo_moment_panels,panels);
            receipt_.bupo_moment_primitive_products=add(receipt_.bupo_moment_primitive_products,pass->moment_primitive_products);
        }
        receipt_.bupo_last_j_bound=r.approximation_j_bound;
        receipt_.ri_bupo_forward_source_identity=r.forward.source_identity;
        receipt_.ri_bupo_adjoint_source_identity=r.adjoint.source_identity;
        receipt_.ri_bupo_last_forward_bound=r.forward.approximation_bound;
        receipt_.ri_bupo_last_adjoint_bound=r.adjoint.approximation_bound;
        receipt_.ri_bupo_last_energy_bound=r.approximation_energy_bound;
        receipt_.ri_bupo_last_metric_residual=r.metric_residual_norm;
        receipt_.ri_bupo_metric_inverse_norm_bound=r.metric_inverse_norm_bound;
    }
    void charge(const BupoReceipt& r) const {
        using namespace admission;
        receipt_.jk_build_calls=add(receipt_.jk_build_calls,1);
        receipt_.jk_shell_calls=add(receipt_.jk_shell_calls,add(r.near_shell_calls,r.moment_panels));
        receipt_.primitive_integrals_evaluated=add(receipt_.primitive_integrals_evaluated,
            add(r.near_primitive_evaluations_measured?r.near_primitive_quartets:0,mul(r.moment_primitive_pairs,bupo_->plan().moments)));
        receipt_.bupo_build_calls=add(receipt_.bupo_build_calls,1);
        receipt_.bupo_near_shell_calls=add(receipt_.bupo_near_shell_calls,r.near_shell_calls);
        receipt_.bupo_far_blocks=add(receipt_.bupo_far_blocks,r.far_blocks);
        receipt_.bupo_moment_panels=add(receipt_.bupo_moment_panels,r.moment_panels);
        receipt_.bupo_moment_primitive_pairs=add(receipt_.bupo_moment_primitive_pairs,r.moment_primitive_pairs);
        receipt_.bupo_moment_primitive_products=add(receipt_.bupo_moment_primitive_products,r.moment_primitive_pairs);
        receipt_.bupo_last_j_bound=r.approximation_j_bound;
    }
    void charge(const BuildReceipt& r) const {
        receipt_.jk_build_calls=admission::add(receipt_.jk_build_calls,1);
        receipt_.jk_shell_calls=admission::add(receipt_.jk_shell_calls,r.shell_calls);
        receipt_.primitive_integrals_evaluated=admission::add(receipt_.primitive_integrals_evaluated,r.primitive_quartets_evaluated);
    }
    void charge(const RIReceipt& r) const {
        receipt_.jk_build_calls=admission::add(receipt_.jk_build_calls,1);
        receipt_.jk_shell_calls=admission::add(receipt_.jk_shell_calls,r.shell_calls);
        receipt_.primitive_integrals_evaluated=admission::add(receipt_.primitive_integrals_evaluated,r.primitive_products);
    }
    Count provider_peak_=0;
    bool j_ri_,k_ri_,j_bupo_,j_ri_bupo_;
    MolecularEngineProvenance& receipt_;
    std::unique_ptr<OrderedJKBuilder> direct_;
    std::unique_ptr<RIJKBuilder> ri_,range_ri_;
    std::unique_ptr<BupoJBuilder> bupo_;
    std::unique_ptr<BupoRIJBuilder> ri_bupo_;
    std::unique_ptr<OrderedJKBuilder> range_direct_;
    KernelRequest range_request_;
};
// A new immutable confirmation view is created after the SCF solve. It uses
// exact J for BUPO or exact RI-J for RI-BUPO, preserving full/range K and native XC.
// No density-dependent operator captured by the solver is mutated.
class ConfirmationJK final : public JKBuilder {
public:
    explicit ConfirmationJK(const SelectedJK& selected):selected_(selected) {}
    Eigen::MatrixXd build_J(const Eigen::MatrixXd& d) const override { return selected_.build_confirmation_J(d); }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& d) const override { return selected_.build_K(d); }
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> build_K_pair(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) const override { return selected_.build_K_pair(a,b); }
    Eigen::MatrixXd build_K_erf(const Eigen::MatrixXd& d,double omega) const override { return selected_.build_K_erf(d,omega); }
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& d,double fraction=1.) const override {
        Eigen::MatrixXd matrix=build_J(d); if(fraction!=0.) matrix-=.5*fraction*build_K(d); return matrix;
    }
private:
    const SelectedJK& selected_;
};
// Serial ordered reference: the reported call count and zero primitive
// screening policy match execution, without a process-wide engine pool.
Eigen::MatrixXd reference_one_electron(const BasisSet& basis,libint2::Operator op,const md::Charges& charges={}) {
    ensure_libint_initialized();
    const auto& shells=basis.libint();
    libint2::Engine engine(op,shells.max_nprim(),shells.max_l(),0);
    engine.set_precision(0.);
    if(op==libint2::Operator::nuclear) engine.set_params(charges);
    Eigen::MatrixXd matrix=Eigen::MatrixXd::Zero(basis.nbasis(),basis.nbasis());
    Eigen::Index row=0;
    for(const auto& a:shells) {
        Eigen::Index column=0;
        for(const auto& b:shells) {
            engine.compute(a,b);
            const auto* values=engine.results()[0];
            if(values) for(std::size_t i=0;i<a.size();++i) for(std::size_t j=0;j<b.size();++j)
                matrix(row+i,column+j)=values[i*b.size()+j];
            column+=b.size();
        }
        row+=a.size();
    }
    if(!matrix.allFinite()) throw std::overflow_error("Selected libint one-electron reference returned nonfinite values");
    return matrix;
}
struct Hamiltonian {
    Eigen::MatrixXd overlap,core;
    double nuclear=0.;
};
Count admit_hamiltonian(const Molecule& mol,const BasisSet& basis,const MolecularHFEngine& engine) {
    OneElectronRequest potential; potential.operation="potential";
    for(const auto& a:mol.atoms()) potential.charges.push_back({static_cast<double>(a.Z),a.xyz});
    Count peak=0;
    for(const auto* op:{"overlap","kinetic","potential"}) {
        auto request=std::string(op)=="potential"?potential:OneElectronRequest{}; request.operation=op;
        const auto plan=plan_one_electron(basis,basis,request,engine.budget);
        peak=std::max(peak,plan.modeled_peak_bytes);
    }
    return peak;
}
Hamiltonian hamiltonian(const Molecule& mol,const BasisSet& basis,const MolecularHFEngine& engine,MolecularEngineProvenance& receipt) {
    OneElectronRequest potential; potential.operation="potential";
    for(const auto& a:mol.atoms()) potential.charges.push_back({static_cast<double>(a.Z),a.xyz});
    Hamiltonian h; const auto start=Clock::now();
    if(engine.backend=="manta") {
        OneElectronRequest request;
        auto s=compute_one_electron(basis,basis,request,engine.budget); h.overlap=std::move(s.values);
        request.operation="kinetic"; auto t=compute_one_electron(basis,basis,request,engine.budget);
        auto v=compute_one_electron(basis,basis,potential,engine.budget); h.core=t.values+v.values;
        receipt.one_electron_source_identity=hash_text(s.receipt.source_identity+":"+t.receipt.source_identity+":"+v.receipt.source_identity);
        receipt.primitive_integrals_evaluated=admission::add(receipt.primitive_integrals_evaluated,
            admission::add(s.receipt.primitive_pairs,admission::add(t.receipt.primitive_pairs,v.receipt.primitive_pairs)));
    } else {
        h.overlap=reference_one_electron(basis,libint2::Operator::overlap);
        h.core=reference_one_electron(basis,libint2::Operator::kinetic)+reference_one_electron(basis,libint2::Operator::nuclear,potential.charges);
        receipt.one_electron_source_identity=hash_text("libint:"+auxiliary_basis_content_identity_sha256(basis)+":"+matrix_identity(h.core)+":"+matrix_identity(h.overlap));
    }
    h.nuclear=mol.nuclear_repulsion(); admission::require_finite(h.nuclear);
    receipt.one_electron_shell_calls=admission::mul(3,admission::square(basis.nshells()));
    receipt.one_electron_wall_seconds=seconds(start);
    return h;
}
bool retains_gradient_source(const MolecularEngineProvenance& receipt) {
    return (receipt.method=="rhf" || receipt.method=="uhf") && receipt.requested_backend=="manta"
        && (receipt.coulomb_algorithm=="exact" || receipt.coulomb_algorithm=="ri")
        && (receipt.exchange_algorithm=="exact" || receipt.exchange_algorithm=="ri");
}
void retain_gradient_source(const MolecularHFEngine& requested,const BasisSet* auxiliary,MolecularEngineProvenance& receipt) {
    if(!retains_gradient_source(receipt)) return;
    auto source=std::make_shared<MolecularGradientSource>();
    source->budget=requested.budget; source->ri=requested.ri;
    source->coulomb=requested.coulomb; source->exchange=requested.exchange;
    if(auxiliary) source->auxiliary=*auxiliary;
    receipt.gradient_source=std::move(source);
}
MolecularHFEngine prepare_engine(const Molecule& mol,const BasisSet& basis,const MolecularHFEngine& requested,MolecularEngineProvenance& receipt,const BasisSet* auxiliary) {
    auto engine=requested;
    receipt.requested_backend=engine.backend;
    receipt.provider_version=engine.backend=="manta"?"md-fp64-v1":LIBINT_VERSION;
    receipt.channel_batch=engine.channel_batch;
    receipt.one_electron_provider=receipt.coulomb_provider=receipt.exchange_provider=engine.backend;
    receipt.coulomb_algorithm=engine.coulomb; receipt.exchange_algorithm=engine.exchange;
    receipt.primitive_evaluations_measured=engine.backend=="manta";
    // Account for Hamiltonian/confirmation matrices kept around the producer.
    // Solver/DIIS allocations have their existing SCF resource policy; this
    // is explicitly an integral-build budget, not a process-RSS guarantee.
    admission::cap(admission::square(mol.atoms().size()),engine.budget.scalar_terms,"Molecular nuclear-pair work budget exceeded");
    const Count known_live=admission::add(admission::mul(64,admission::square(basis.nbasis())),admission::mul(128,mol.atoms().size()));
    engine.budget.other_live_bytes=admission::add(engine.budget.other_live_bytes,known_live);
    // Reserve the immutable post-SCF derivative context before producer
    // admission. The auxiliary copy is made only after successful SCF setup.
    if(retains_gradient_source(receipt)) {
        const Count context=admission::add(sizeof(MolecularGradientSource)+128,auxiliary?admission::mul(2,admission::census(*auxiliary,7).bytes):0);
        engine.budget.other_live_bytes=admission::add(engine.budget.other_live_bytes,context);
    }
    admission::cap(admission::add(engine.budget.other_live_bytes,engine.budget.backend_margin_bytes),
        engine.budget.memory_bytes,"Molecular engine input/Hamiltonian memory budget exceeded");
    return engine;
}
void identify_sources(const Molecule& mol,const BasisSet& basis,const BasisSet* aux,MolecularEngineProvenance& receipt) {
    validate_geometry(mol,basis);
    if(aux) {
        validate_geometry(mol,*aux);
        std::vector<bool> coverage(mol.atoms().size(),false);
        for(std::size_t i=0;i<aux->nshells();++i) coverage[aux->shell_atom_index(i)]=true;
        for(std::size_t i=0;i<basis.nshells();++i) if(!coverage[basis.shell_atom_index(i)])
            throw std::invalid_argument("Selected molecular RI auxiliary basis has incomplete atom coverage");
    }
    receipt.orbital_basis_identity=auxiliary_basis_content_identity_sha256(basis);
    receipt.molecule_identity=molecule_identity(mol);
}
void validate_ks_functional(const Functional& f,const MolecularHFEngine& engine) {
    if(f.is_external() || f.is_double_hybrid())
        throw std::invalid_argument("Selected KS currently requires a libxc functional without a double-hybrid post-SCF correction");
    double full=0.,range=0.;
    for(const auto& [kernel,weight,omega]:f.exchange_components()) {
        if(!std::isfinite(weight) || !std::isfinite(omega)) throw std::invalid_argument("Nonfinite KS exchange source");
        if(kernel=="coulomb") full+=weight;
        else if(kernel=="erfc" && omega==f.rsh_omega()) { full+=weight; range-=weight; }
        else throw std::invalid_argument("Selected KS requires a single Coulomb/erf range, not mixed ranges or Yukawa exchange");
    }
    if(std::abs(full-f.cam_alpha())>1e-13 || std::abs(range-f.cam_beta())>1e-13)
        throw std::invalid_argument("Selected KS resolved exchange differs from the native CAM contract");
    if(range!=0. && engine.backend!="manta")
        throw std::invalid_argument("Selected libint reference does not yet admit range-separated exchange");
}
std::string grid_identity(const Grid& grid) {
    return hash_text(matrix_identity(grid.points)+":"+matrix_identity(grid.weights));
}
void identify_xc(const Functional& functional,const Grid& grid,const Grid* vv10,MolecularEngineProvenance& receipt) {
    receipt.functional=functional.name(); receipt.xc_provider=functional.needs_vv10()?"libxc+vibeqc-vv10":"libxc";
    receipt.full_exchange_fraction=functional.cam_alpha();
    receipt.range_exchange_fraction=functional.cam_beta(); receipt.range_omega=functional.rsh_omega();
    receipt.exchange_active=functional.is_hybrid();
    receipt.xc_grid_points=grid.points.rows(); receipt.vv10_grid_points=vv10?vv10->points.rows():0;
    const auto components=functional.components();
    Eigen::MatrixXd resolved(components.size()+1,6); resolved.setZero();
    resolved(0,0)=functional.cam_alpha(); resolved(0,1)=functional.cam_beta();
    resolved(0,2)=functional.rsh_omega(); resolved(0,3)=functional.spin();
    resolved(0,4)=functional.vv10_b(); resolved(0,5)=functional.vv10_C();
    for(std::size_t i=0;i<components.size();++i) { resolved(i+1,0)=components[i].first; resolved(i+1,1)=components[i].second; }
    receipt.xc_source_identity=hash_text("molecular-xc-v1:"+matrix_identity(resolved)+":"+grid_identity(grid)
        +":"+(vv10?grid_identity(*vv10):"same-grid"));
}
void confirm(MolecularEngineProvenance& p,double energy_delta,double fock_delta,double error,double tolerance,bool& converged) {
    p.terminal_energy_delta=energy_delta; p.terminal_fock_delta=fock_delta; p.terminal_commutator_norm=error;
    p.full_build_confirmed=std::isfinite(energy_delta) && std::isfinite(fock_delta) && energy_delta<=1e-9 && fock_delta<=1e-9;
    p.stationarity_confirmed=p.full_build_confirmed && std::isfinite(error) && error<=tolerance;
    converged=converged && p.stationarity_confirmed;
}
} // namespace
bool matches_molecular_engine_density(const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta,const MolecularEngineProvenance& p) {
    return p.terminal_density_alpha_identity==matrix_identity(alpha) && p.terminal_density_beta_identity==matrix_identity(beta);
}
bool matches_molecular_engine_source(const Molecule& mol,const BasisSet& basis,const MolecularEngineProvenance& p) {
    return p.orbital_basis_identity==auxiliary_basis_content_identity_sha256(basis) && p.molecule_identity==molecule_identity(mol);
}
RHFResult run_rhf_with_engine(const Molecule& mol,const BasisSet& basis,const RHFOptions& opts,const MolecularHFEngine& requested,const BasisSet* aux) {
    validate_options(opts,requested);
    if(mol.n_electrons()<0 || mol.n_electrons()%2 || mol.multiplicity()!=1)
        throw std::invalid_argument("Selected RHF requires nonnegative even electron count and multiplicity one");
    MolecularEngineProvenance receipt; receipt.method="rhf";
    auto engine=prepare_engine(mol,basis,requested,receipt,aux);
    const Count one_peak=admit_hamiltonian(mol,basis,engine);
    identify_sources(mol,basis,aux,receipt);
    SelectedJK jk(basis,aux,engine,receipt,one_peak);
    engine.budget.other_live_bytes=admission::add(engine.budget.other_live_bytes,jk.provider_peak());
    const auto h=hamiltonian(mol,basis,engine,receipt);
    auto result=run_rhf_scf_with_jk(basis,mol.n_electrons(),h.overlap,h.core,h.nuclear,jk,opts,{},&mol);
    const auto j=jk.build_confirmation_J(result.density),k=jk.build_K(result.density);
    const Eigen::MatrixXd f=h.core+j-.5*k;
    const double energy=h.nuclear+(result.density.array()*h.core.array()).sum()+.5*(result.density.array()*j.array()).sum()-.25*(result.density.array()*k.array()).sum();
    const double error=(f*result.density*h.overlap-h.overlap*result.density*f).norm();
    receipt.terminal_density_alpha_identity=receipt.terminal_density_beta_identity=matrix_identity(.5*result.density);
    confirm(receipt,std::abs(energy-result.energy),(f-result.fock).cwiseAbs().maxCoeff(),error,opts.conv_tol_grad,result.converged);
    retain_gradient_source(requested,aux,receipt);
    result.ecp_provenance_verified=true; result.engine_provenance=std::move(receipt); return result;
}
UHFResult run_uhf_with_engine(const Molecule& mol,const BasisSet& basis,const UHFOptions& opts,const MolecularHFEngine& requested,const BasisSet* aux) {
    validate_options(opts,requested);
    if(!opts.atomic_spins.empty() || opts.spinlock_mode!=SpinlockMode::OFF || !opts.multi_guess_seeds.empty())
        throw std::invalid_argument("Selected UHF does not yet admit atomic spin seeds, spin locking or multi-guess schedules");
    const int ne=mol.n_electrons(),spin=mol.multiplicity()-1;
    if(ne<0 || spin<0 || spin>ne || (ne+spin)%2) throw std::invalid_argument("Selected UHF electron count/multiplicity mismatch");
    MolecularEngineProvenance receipt; receipt.method="uhf";
    auto engine=prepare_engine(mol,basis,requested,receipt,aux);
    const Count one_peak=admit_hamiltonian(mol,basis,engine);
    identify_sources(mol,basis,aux,receipt);
    SelectedJK jk(basis,aux,engine,receipt,one_peak);
    engine.budget.other_live_bytes=admission::add(engine.budget.other_live_bytes,jk.provider_peak());
    const auto h=hamiltonian(mol,basis,engine,receipt);
    auto result=run_uhf_scf_with_jk(basis,(ne+spin)/2,(ne-spin)/2,h.overlap,h.core,h.nuclear,jk,opts,{},{},&mol);
    const Eigen::MatrixXd d=result.density_alpha+result.density_beta;
    const auto j=jk.build_confirmation_J(d); const auto k=jk.build_K_pair(result.density_alpha,result.density_beta);
    const Eigen::MatrixXd fa=h.core+j-k.first,fb=h.core+j-k.second;
    const double energy=h.nuclear+(d.array()*h.core.array()).sum()+.5*(d.array()*j.array()).sum()
        -.5*((result.density_alpha.array()*k.first.array()).sum()+(result.density_beta.array()*k.second.array()).sum());
    const double error=std::max((fa*result.density_alpha*h.overlap-h.overlap*result.density_alpha*fa).norm(),
                               (fb*result.density_beta*h.overlap-h.overlap*result.density_beta*fb).norm());
    receipt.terminal_density_alpha_identity=matrix_identity(result.density_alpha);
    receipt.terminal_density_beta_identity=matrix_identity(result.density_beta);
    const double delta=std::max((fa-result.fock_alpha).cwiseAbs().maxCoeff(),(fb-result.fock_beta).cwiseAbs().maxCoeff());
    confirm(receipt,std::abs(energy-result.energy),delta,error,opts.conv_tol_grad,result.converged);
    retain_gradient_source(requested,aux,receipt);
    result.ecp_provenance_verified=true; result.engine_provenance=std::move(receipt); return result;
}
RKSResult run_rks_with_engine(const Molecule& mol,const BasisSet& basis,const RKSOptions& opts,const MolecularHFEngine& requested,const BasisSet* aux) {
    validate_options(opts,requested);
    if(mol.n_electrons()<0 || mol.n_electrons()%2 || mol.multiplicity()!=1)
        throw std::invalid_argument("Selected RKS requires nonnegative even electron count and multiplicity one");
    const Functional functional(opts.functional);
    validate_ks_functional(functional,requested);
    if(!std::isfinite(opts.grid.vv10_grid_factor)) throw std::invalid_argument("Nonfinite VV10 grid factor");
    MolecularEngineProvenance receipt; receipt.method="rks";
    auto engine=prepare_engine(mol,basis,requested,receipt,aux);
    const Count one_peak=admit_hamiltonian(mol,basis,engine);
    identify_sources(mol,basis,aux,receipt);
    KernelRequest range; range.kernel="erf"; range.omega=functional.rsh_omega();
    SelectedJK jk(basis,aux,engine,receipt,one_peak,functional.cam_beta()!=0.?&range:nullptr);
    engine.budget.other_live_bytes=admission::add(engine.budget.other_live_bytes,jk.provider_peak());
    const auto h=hamiltonian(mol,basis,engine,receipt);
    const auto grid=build_grid(mol,opts.grid);
    std::unique_ptr<Grid> vv10;
    if(functional.needs_vv10() && opts.grid.vv10_grid_factor>0. && opts.grid.vv10_grid_factor!=1.)
        vv10=std::make_unique<Grid>(build_vv10_grid(mol,opts.grid));
    identify_xc(functional,grid,vv10.get(),receipt);
    auto result=run_rks_scf_with_jk(basis,mol.n_electrons(),h.overlap,h.core,h.nuclear,jk,grid,opts,{},vv10.get(),nullptr,nullptr,&mol);
    // Evaluate exactly the returned density with the native XC composition,
    // including any nonlocal grid. NOITER performs no density update. Force
    // its ordinary evaluation branch even after an orbital optimizer solve.
    auto final_opts=opts; final_opts.max_iter=0; final_opts.initial_guess=InitialGuess::READ;
    final_opts.orbital_optimizer="native"; final_opts.read_density=result.density;
    const ConfirmationJK confirmation(jk);
    const auto final=run_rks_scf_with_jk(basis,mol.n_electrons(),h.overlap,h.core,h.nuclear,confirmation,grid,final_opts,result.density,vv10.get(),nullptr,nullptr,&mol);
    const double error=(final.fock*result.density*h.overlap-h.overlap*result.density*final.fock).norm();
    receipt.terminal_density_alpha_identity=receipt.terminal_density_beta_identity=matrix_identity(.5*result.density);
    const double energy_delta=std::max({std::abs(final.energy-result.energy),std::abs(final.e_coulomb-result.e_coulomb),
        std::abs(final.e_hf_exchange-result.e_hf_exchange),std::abs(final.e_xc-result.e_xc)});
    confirm(receipt,energy_delta,(final.fock-result.fock).cwiseAbs().maxCoeff(),error,opts.conv_tol_grad,result.converged);
    retain_gradient_source(requested,aux,receipt);
    result.ecp_provenance_verified=true; result.engine_provenance=std::move(receipt); return result;
}
UKSResult run_uks_with_engine(const Molecule& mol,const BasisSet& basis,const UKSOptions& opts,const MolecularHFEngine& requested,const BasisSet* aux) {
    validate_options(opts,requested);
    if(!opts.atomic_spins.empty() || opts.spinlock_mode!=SpinlockMode::OFF || !opts.multi_guess_seeds.empty())
        throw std::invalid_argument("Selected UKS does not yet admit atomic spin seeds, spin locking or multi-guess schedules");
    const int ne=mol.n_electrons(),spin=mol.multiplicity()-1;
    if(ne<0 || spin<0 || spin>ne || (ne+spin)%2) throw std::invalid_argument("Selected UKS electron count/multiplicity mismatch");
    const Functional functional(opts.functional,2);
    validate_ks_functional(functional,requested);
    if(!std::isfinite(opts.grid.vv10_grid_factor)) throw std::invalid_argument("Nonfinite VV10 grid factor");
    MolecularEngineProvenance receipt; receipt.method="uks";
    auto engine=prepare_engine(mol,basis,requested,receipt,aux);
    const Count one_peak=admit_hamiltonian(mol,basis,engine);
    identify_sources(mol,basis,aux,receipt);
    KernelRequest range; range.kernel="erf"; range.omega=functional.rsh_omega();
    SelectedJK jk(basis,aux,engine,receipt,one_peak,functional.cam_beta()!=0.?&range:nullptr);
    engine.budget.other_live_bytes=admission::add(engine.budget.other_live_bytes,jk.provider_peak());
    const auto h=hamiltonian(mol,basis,engine,receipt);
    const auto grid=build_grid(mol,opts.grid);
    std::unique_ptr<Grid> vv10;
    if(functional.needs_vv10() && opts.grid.vv10_grid_factor>0. && opts.grid.vv10_grid_factor!=1.)
        vv10=std::make_unique<Grid>(build_vv10_grid(mol,opts.grid));
    identify_xc(functional,grid,vv10.get(),receipt);
    auto result=run_uks_scf_with_jk(basis,(ne+spin)/2,(ne-spin)/2,h.overlap,h.core,h.nuclear,jk,grid,opts,{},{},vv10.get(),&mol);
    auto final_opts=opts; final_opts.max_iter=0; final_opts.initial_guess=InitialGuess::READ;
    final_opts.orbital_optimizer="native";
    final_opts.read_density_alpha=result.density_alpha; final_opts.read_density_beta=result.density_beta;
    const ConfirmationJK confirmation(jk);
    const auto final=run_uks_scf_with_jk(basis,(ne+spin)/2,(ne-spin)/2,h.overlap,h.core,h.nuclear,confirmation,grid,final_opts,
        result.density_alpha,result.density_beta,vv10.get(),&mol);
    const double error=std::max((final.fock_alpha*result.density_alpha*h.overlap-h.overlap*result.density_alpha*final.fock_alpha).norm(),
        (final.fock_beta*result.density_beta*h.overlap-h.overlap*result.density_beta*final.fock_beta).norm());
    receipt.terminal_density_alpha_identity=matrix_identity(result.density_alpha);
    receipt.terminal_density_beta_identity=matrix_identity(result.density_beta);
    const double delta=std::max((final.fock_alpha-result.fock_alpha).cwiseAbs().maxCoeff(),(final.fock_beta-result.fock_beta).cwiseAbs().maxCoeff());
    const double energy_delta=std::max({std::abs(final.energy-result.energy),std::abs(final.e_coulomb-result.e_coulomb),
        std::abs(final.e_hf_exchange-result.e_hf_exchange),std::abs(final.e_xc-result.e_xc)});
    confirm(receipt,energy_delta,delta,error,opts.conv_tol_grad,result.converged);
    retain_gradient_source(requested,aux,receipt);
    result.ecp_provenance_verified=true; result.engine_provenance=std::move(receipt); return result;
}
} // namespace vibeqc::manta
