#include "vibeqc/manta_bupo.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <chrono>
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
// Disjoint optional timing regions. Disabled profiling avoids clock reads in
// every tile/node; it is execution metadata and never enters source identity.
class SectionClock {
    using Clock=std::chrono::steady_clock;
    bool enabled_;
    Clock::time_point previous_;
public:
    explicit SectionClock(bool enabled):enabled_(enabled),previous_(enabled?Clock::now():Clock::time_point{}) {}
    void finish(double& elapsed) {
        if(enabled_) {
            const auto now=Clock::now();
            elapsed+=std::chrono::duration<double>(now-previous_).count(); previous_=now;
        }
    }
};
void accumulate(double value,double& sum,double& correction) {
    const double next=sum+value;
    correction+=std::abs(sum)>=std::abs(value)?(sum-next)+value:(value-next)+sum;
    sum=next;
}
void add_column(Eigen::MatrixXd& sum,Eigen::MatrixXd& correction,Count column,const Eigen::VectorXd& values) {
    for(Eigen::Index i=0;i<values.size();++i) accumulate(values[i],sum(i,column),correction(i,column));
}
void hash_word(vibeqc::detail::Sha256& hash,std::uint64_t value) {
    std::array<std::uint8_t,8> bytes{}; for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(value>>(56-8*k));
    hash.update(bytes.data(),bytes.size());
}
void hash_real(vibeqc::detail::Sha256& hash,double value) {
    if(value==0.) value=0.; std::uint64_t word; std::memcpy(&word,&value,sizeof(word)); hash_word(hash,word);
}
void hash_text(vibeqc::detail::Sha256& hash,const std::string& text) {
    hash_word(hash,text.size()); hash.update(reinterpret_cast<const std::uint8_t*>(text.data()),text.size());
}
md::Point displacement(const md::Point& a,const md::Point& b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
ResourceBudget traversal_budget(ResourceBudget budget) {
    // The consuming J plan charges actual matrix products and contractions;
    // represented pair count is a coverage diagnostic, not an execution cost.
    budget.scalar_terms=std::numeric_limits<Count>::max(); return budget;
}
template<class Callback>
void near_pairs(const BubbleTree& tree,const BubbleBlock& block,Callback callback) {
    const auto& a=tree.nodes()[block.target]; const auto& b=tree.nodes()[block.source];
    const auto& order=tree.pair_order();
    for(Count i=a.begin;i<a.end;++i) {
        const auto end=block.target==block.source?i+1:b.end;
        for(Count j=b.begin;j<end;++j) callback(order[i],order[j]);
    }
}
double error_product(double bound,double density_norm) {
    if(bound==0. || density_norm==0.) return 0.;
    const double product=bound*density_norm;
    const auto value=std::nextafter(product*(1+32*std::numeric_limits<double>::epsilon()),std::numeric_limits<double>::infinity());
    if(!std::isfinite(value)) throw std::overflow_error("MANTA BUPO density-weighted approximation bound exceeds binary64");
    return value;
}
} // namespace
bool BupoJBuilder::accepted(const BubbleBlock& block,BubbleInteraction* output) const {
    if(!block.far_candidate) return false;
    try {
        auto interaction=errors_->interaction(block.target,block.source);
        if(interaction.error.total>options_.integral_tolerance) return false;
        if(output) *output=std::move(interaction);
        return true;
    } catch(const std::overflow_error&) {
        // A far-field representation failure is handled by direct refinement,
        // never by deleting a distribution or accepting an infinite bound.
        return false;
    }
}
BupoJBuilder::BupoJBuilder(const BasisSet& basis,const BupoOptions& options,const ResourceBudget& budget)
    :options_(options),budget_(budget) {
    using namespace admission;
    SectionClock setup(options.collect_timings),section(options.collect_timings);
    plan_.timings_collected=options.collect_timings;
    margin(budget); require_finite(options.opening); require_finite(options.integral_tolerance);
    if(options.order<0 || options.order>12 || options.opening<0 || options.opening>=1. || options.integral_tolerance<0)
        throw std::invalid_argument("MANTA BUPO order, opening or integral tolerance is invalid");
    if(options.near_backend!="manta" && options.near_backend!="libint")
        throw std::invalid_argument("MANTA BUPO near backend must be manta or libint");
    plan_.n_basis=basis.nbasis();
    // Staged admission: reserve density/target/export matrices before copying
    // any source, then charge every still-live setup object to the next stage.
    ResourceBudget stage=budget;
    stage.other_live_bytes=add(stage.other_live_bytes,mul(64,square(plan_.n_basis)));
    if(options.near_backend=="manta") provider_=std::make_shared<MantaProvider>(basis,KernelRequest{},stage,1,true);
    else provider_=std::make_shared<LibintProvider>(basis,KernelRequest{},stage,true);
    section.finish(plan_.provider_setup_wall_seconds);
    const auto& tile_plan=provider_->plan();
    stage.other_live_bytes=tile_plan.modeled_peak_bytes-budget.backend_margin_bytes;
    auto extents_options=options.extents;
    extents_options.moment_order=options.order; extents_options.retain_full_moments=true;
    const auto extent_plan=plan_pair_extents(basis,extents_options,stage);
    const auto extents=compute_pair_extents(basis,extents_options,stage);
    section.finish(plan_.extent_setup_wall_seconds);
    stage.other_live_bytes=extent_plan.modeled_peak_bytes-budget.backend_margin_bytes;
    tree_=std::make_shared<BubbleTree>(extents,options.bubbles,stage);
    section.finish(plan_.tree_setup_wall_seconds);
    plan_.shell_pairs=tree_->pairs().size(); plan_.nodes=tree_->nodes().size();
    const auto multipoles=plan_multipoles(options.order,budget);
    plan_.moments=multipoles.moments;
    const Count node_vectors=mul(32,mul(plan_.nodes,plan_.moments));
    const Count stack_bytes=add(2048,mul(16,add(1,mul(70,tree_->depth()))));
    stage.other_live_bytes=add(tree_->plan().modeled_peak_bytes-budget.backend_margin_bytes,add(node_vectors,stack_bytes));
    errors_=std::make_shared<BubbleErrorModel>(tree_,options.order,stage);
    section.finish(plan_.error_model_setup_wall_seconds);
    plan_.modeled_peak_bytes=errors_->plan().modeled_peak_bytes;
    // One global-angular panel bounds all pair workspaces. Scale its work by
    // the maximum primitive extent when that belongs to another angular shell.
    const auto& shells=basis.libint(); std::size_t maximum_l_shell=0;
    Count maximum_ao_size=0;
    for(std::size_t i=0;i<shells.size();++i) {
        maximum_ao_size=std::max<Count>(maximum_ao_size,shells[i].size());
        if(shells[i].contr[0].l>shells[maximum_l_shell].contr[0].l) maximum_l_shell=i;
    }
    ResourceBudget standalone=budget; standalone.other_live_bytes=0;
    const auto moment_plan=plan_pair_moments(basis,{maximum_l_shell,maximum_l_shell},options.order,standalone);
    plan_.modeled_peak_bytes=add(plan_.modeled_peak_bytes,moment_plan.modeled_peak_bytes-budget.backend_margin_bytes);
    cap(plan_.modeled_peak_bytes,budget.memory_bytes,"MANTA BUPO combined workspace memory budget exceeded");
    const Count moment_scale=std::max<Count>(1,(square(tile_plan.maximum_primitives)+moment_plan.primitive_pairs-1)/moment_plan.primitive_pairs);
    const Count moment_work=mul(moment_plan.kernel_terms,moment_scale);
    const Count query_work=add(multipoles.kernel_terms,mul(128,multipoles.scalar_terms));
    plan_.kernel_terms=add(extent_plan.kernel_terms,add(tree_->plan().kernel_terms,errors_->plan().kernel_terms));
    cap(plan_.kernel_terms,budget.kernel_terms,"MANTA BUPO setup work budget exceeded");
    BubbleTraversal traversal(tree_,options.opening,traversal_budget(budget));
    while(const auto pending=traversal.peek()) {
        const auto block=*pending;
        BubbleInteraction interaction;
        if(block.far_candidate) {
            plan_.kernel_terms=add(plan_.kernel_terms,mul(2,query_work));
            cap(plan_.kernel_terms,budget.kernel_terms,"MANTA BUPO aggregate far-admission work budget exceeded");
            if(!accepted(block,&interaction)) { traversal.refine(); continue; }
            ++plan_.far_blocks;
            plan_.maximum_integral_error=std::max(plan_.maximum_integral_error,interaction.error.total);
            plan_.maximum_compact_error=std::max(plan_.maximum_compact_error,interaction.error.compact_remainder);
            plan_.maximum_penetration_error=std::max(plan_.maximum_penetration_error,interaction.error.tail_penetration);
            plan_.maximum_moment_tail_error=std::max(plan_.maximum_moment_tail_error,interaction.error.full_moment_tail);
        } else near_pairs(*tree_,block,[&](Count p,Count q) {
            const auto& a=tree_->pairs()[p].shells; const auto& b=tree_->pairs()[q].shells;
            Count primitives=1,elements=1;
            for(auto s:{a[0],a[1],b[0],b[1]}) { primitives=mul(primitives,shells[s].alpha.size()); elements=mul(elements,shells[s].size()); }
            plan_.near_shell_calls=add(plan_.near_shell_calls,1);
            plan_.near_primitive_quartets=add(plan_.near_primitive_quartets,primitives);
            plan_.scalar_terms=add(plan_.scalar_terms,mul(p==q?1:2,elements));
            cap(plan_.near_shell_calls,budget.shell_calls,"MANTA BUPO near shell-call budget exceeded");
            cap(plan_.near_primitive_quartets,budget.primitive_quartets,"MANTA BUPO near primitive budget exceeded");
            cap(plan_.scalar_terms,budget.scalar_terms,"MANTA BUPO near contraction budget exceeded");
        });
        traversal.accept();
    }
    section.finish(plan_.partition_setup_wall_seconds);
    plan_.traversal_visits=traversal.receipt().visited_node_pairs;
    plan_.partition_identity=traversal.receipt().partition_identity;
    if(plan_.far_blocks) {
        plan_.moment_panels=mul(2,plan_.shell_pairs);
        plan_.moment_primitive_pairs=mul(2,extent_plan.primitive_pairs);
        const Count translation_calls=mul(2,plan_.nodes-1);
        plan_.kernel_terms=add(plan_.kernel_terms,add(mul(plan_.moment_panels,moment_work),mul(translation_calls,multipoles.kernel_terms)));
        const Count moment_scalar_upper=mul(square(maximum_ao_size),plan_.moments);
        plan_.scalar_terms=add(plan_.scalar_terms,add(mul(plan_.moment_panels,moment_scalar_upper),mul(add(translation_calls,mul(2,plan_.far_blocks)),multipoles.scalar_terms)));
    }
    plan_.shell_calls=add(extent_plan.shell_pairs,add(mul(2,plan_.traversal_visits),add(plan_.near_shell_calls,plan_.moment_panels)));
    plan_.primitive_products=add(extent_plan.primitive_pairs,add(plan_.near_primitive_quartets,plan_.moment_primitive_pairs));
    plan_.scalar_terms=add(plan_.scalar_terms,add(extent_plan.scalar_terms,add(tree_->plan().scalar_terms,errors_->plan().scalar_terms)));
    // The reported cap must also admit every staged workspace preflight,
    // including a moment panel when the final partition happens to be all near.
    plan_.scalar_terms=std::max(plan_.scalar_terms,moment_plan.scalar_terms);
    plan_.kernel_terms=add(plan_.kernel_terms,add(mul(plan_.near_shell_calls,tile_plan.kernel_terms),add(mul(256,plan_.traversal_visits),mul(4,plan_.scalar_terms))));
    plan_.kernel_terms=std::max(plan_.kernel_terms,moment_plan.kernel_terms);
    cap(plan_.shell_calls,budget.shell_calls,"MANTA BUPO aggregate shell/traversal-call budget exceeded");
    cap(plan_.primitive_products,budget.primitive_quartets,"MANTA BUPO aggregate primitive-product budget exceeded");
    cap(plan_.scalar_terms,budget.scalar_terms,"MANTA BUPO aggregate scalar-contraction budget exceeded");
    cap(plan_.kernel_terms,budget.kernel_terms,"MANTA BUPO aggregate work budget exceeded");
    offsets_.reserve(shells.size()); Eigen::Index offset=0;
    for(const auto& shell:shells) { offsets_.push_back(offset); offset+=shell.size(); }
    vibeqc::detail::Sha256 source;
    hash_text(source,"manta.bupo-j.v1;paired-canonical-pairs;full-density;no-screening;raw-cartesian-moments;");
    hash_text(source,provider_->source_identity()); hash_text(source,errors_->source_identity());
    hash_text(source,plan_.partition_identity); hash_real(source,options.integral_tolerance);
    source_=source.finish_hex();
    setup.finish(plan_.setup_wall_seconds);
}
Eigen::MatrixXd BupoJBuilder::build_K(const Eigen::MatrixXd&) const {
    throw std::invalid_argument("MANTA BUPO-J does not supply exchange; compose an independent K provider");
}
Eigen::MatrixXd BupoJBuilder::build_g_rhf(const Eigen::MatrixXd& density,double alpha) const {
    if(alpha!=0.) throw std::invalid_argument("MANTA BUPO-J requires an independent exchange provider");
    return build_J(density);
}
BupoResult BupoJBuilder::build(const Eigen::MatrixXd& density) const {
    const auto started=std::chrono::steady_clock::now();
    SectionClock validation(options_.collect_timings);
    const auto n=static_cast<Eigen::Index>(plan_.n_basis);
    if(density.rows()!=n || density.cols()!=n || !density.allFinite())
        throw std::invalid_argument("MANTA BUPO requires a finite density matching the basis");
    const double scale=std::max(1.,density.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(density(i,j)-density(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA BUPO requires a symmetric real density");
    BupoResult result; auto& receipt=result.receipt;
    receipt.timings_collected=options_.collect_timings;
    receipt.source_identity=source_; receipt.partition_identity=plan_.partition_identity;
    receipt.near_provider=provider_->name(); receipt.near_primitive_evaluations_measured=provider_->name()=="manta";
    receipt.modeled_peak_bytes=plan_.modeled_peak_bytes;
    vibeqc::detail::Sha256 density_hash;
    double norm=0.,norm_correction=0.;
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<n;++j) {
        hash_real(density_hash,density(i,j)); accumulate(std::abs(density(i,j)),norm,norm_correction);
    }
    receipt.density_identity=density_hash.finish_hex(); receipt.density_l1=norm+norm_correction;
    if(!std::isfinite(receipt.density_l1)) throw std::overflow_error("MANTA BUPO density norm exceeds binary64");
    receipt.approximation_j_bound=error_product(plan_.maximum_integral_error,receipt.density_l1);
    receipt.compact_j_bound=error_product(plan_.maximum_compact_error,receipt.density_l1);
    receipt.penetration_j_bound=error_product(plan_.maximum_penetration_error,receipt.density_l1);
    receipt.moment_tail_j_bound=error_product(plan_.maximum_moment_tail_error,receipt.density_l1);
    result.J=Eigen::MatrixXd::Zero(n,n); Eigen::MatrixXd correction=result.J;
    const auto& pairs=tree_->pairs(); const auto& nodes=tree_->nodes(); const auto& order=tree_->pair_order();
    const auto& shells=provider_->basis().libint();
    auto pair_density=[&](Count id) {
        const auto& s=pairs[id].shells; const auto na=shells[s[0]].size(),nb=shells[s[1]].size();
        Eigen::VectorXd value(na*nb);
        for(std::size_t i=0;i<na;++i) for(std::size_t j=0;j<nb;++j) {
            const auto a=offsets_[s[0]]+i,b=offsets_[s[1]]+j;
            value[i*nb+j]=density(a,b)+(s[0]==s[1]?0.:density(b,a));
        }
        return value;
    };
    auto add_pair=[&](Count id,Eigen::Index element,double value) {
        const auto& s=pairs[id].shells; const auto nb=shells[s[1]].size();
        const auto a=offsets_[s[0]]+element/nb,b=offsets_[s[1]]+element%nb;
        accumulate(value,result.J(a,b),correction(a,b));
    };
    auto moments_at=[&](Count pair,const md::Point& center) {
        const auto& s=pairs[pair].shells;
        ++receipt.moment_panels; receipt.moment_primitive_pairs+=shells[s[0]].alpha.size()*shells[s[1]].alpha.size();
        return md::shell_moments(shells[s[0]],shells[s[1]],center,options_.order);
    };
    validation.finish(receipt.validation_wall_seconds);
    Eigen::MatrixXd moments,moment_error,fields,field_error;
    if(plan_.far_blocks) {
        SectionClock moment_clock(options_.collect_timings);
        moments=Eigen::MatrixXd::Zero(plan_.moments,plan_.nodes); moment_error=moments;
        fields=moments; field_error=moments;
        for(Count node=0;node<nodes.size();++node) if(!nodes[node].child_count)
            for(Count k=nodes[node].begin;k<nodes[node].end;++k) {
                const Count pair=order[k]; const auto panel=moments_at(pair,nodes[node].center);
                add_column(moments,moment_error,node,panel.transpose()*pair_density(pair));
            }
        moment_clock.finish(receipt.leaf_moments_wall_seconds);
        // BUPO upward moment translation and its downward field adjoint:
        // Neese et al. (2025), DOI 10.1021/acs.jpca.4c07415. Cartesian raw
        // moments and complete total-degree truncation are MANTA conventions.
        for(Count reverse=nodes.size();reverse>0;--reverse) {
            const Count node=reverse-1; moments.col(node)+=moment_error.col(node);
            if(node) {
                const Count parent=nodes[node].parent;
                const auto w=multipole_translation(displacement(nodes[node].center,nodes[parent].center),options_.order,budget_);
                add_column(moments,moment_error,parent,w.values*moments.col(node));
            }
        }
        moment_clock.finish(receipt.upward_wall_seconds);
    }
    Workspace workspace(provider_->plan());
    BubbleTraversal traversal(tree_,options_.opening,traversal_budget(budget_));
    while(const auto pending=traversal.peek()) {
        const auto block=*pending;
        if(block.far_candidate) {
            SectionClock far_clock(options_.collect_timings);
            BubbleInteraction interaction;
            const bool accept=accepted(block,&interaction);
            far_clock.finish(receipt.far_admission_wall_seconds);
            if(!accept) { traversal.refine(); continue; }
            add_column(fields,field_error,block.target,interaction.matrix.values*moments.col(block.source));
            add_column(fields,field_error,block.source,interaction.matrix.values.transpose()*moments.col(block.target));
            ++receipt.far_blocks;
            far_clock.finish(receipt.far_apply_wall_seconds);
        } else near_pairs(*tree_,block,[&](Count p,Count q) {
            SectionClock near_clock(options_.collect_timings);
            const auto& a=pairs[p].shells; const auto& b=pairs[q].shells;
            const auto tile=provider_->evaluate({a[0],a[1],b[0],b[1]},workspace);
            near_clock.finish(receipt.near_integrals_wall_seconds);
            ++receipt.near_shell_calls; Count primitives=1;
            for(auto s:{a[0],a[1],b[0],b[1]}) primitives*=shells[s].alpha.size();
            receipt.near_primitive_quartets+=primitives;
            if(!tile.values) { ++receipt.near_null_tiles; near_clock.finish(receipt.near_digest_wall_seconds); return; }
            const auto dp=pair_density(p),dq=pair_density(q);
            for(Eigen::Index i=0;i<dp.size();++i) for(Eigen::Index j=0;j<dq.size();++j) {
                const double value=tile.values[i*dq.size()+j];
                add_pair(p,i,value*dq[j]); if(p!=q) add_pair(q,j,value*dp[i]);
            }
            near_clock.finish(receipt.near_digest_wall_seconds);
        });
        traversal.accept();
    }
    receipt.traversal_visits=traversal.receipt().visited_node_pairs;
    if(traversal.receipt().partition_identity!=plan_.partition_identity)
        throw std::logic_error("MANTA BUPO immutable partition changed during execution");
    if(plan_.far_blocks) {
        for(Count node=0;node<nodes.size();++node) {
            SectionClock reverse_clock(options_.collect_timings);
            fields.col(node)+=field_error.col(node);
            for(int k=0;k<nodes[node].child_count;++k) {
                const Count child=nodes[node].children[k];
                const auto w=multipole_translation(displacement(nodes[child].center,nodes[node].center),options_.order,budget_);
                add_column(fields,field_error,child,w.values.transpose()*fields.col(node));
            }
            reverse_clock.finish(receipt.downward_wall_seconds);
            if(!nodes[node].child_count) for(Count k=nodes[node].begin;k<nodes[node].end;++k) {
                const Count pair=order[k]; const auto panel=moments_at(pair,nodes[node].center);
                const Eigen::VectorXd value=panel*fields.col(node);
                for(Eigen::Index i=0;i<value.size();++i) add_pair(pair,i,value[i]);
            }
            reverse_clock.finish(receipt.reverse_moments_wall_seconds);
        }
    }
    SectionClock final_clock(options_.collect_timings);
    result.J+=correction;
    // Canonical shell-pair reconstruction, with no post-build averaging.
    for(const auto& pair:pairs) {
        const auto a=pair.shells[0],b=pair.shells[1]; if(a==b) continue;
        result.J.block(offsets_[b],offsets_[a],shells[b].size(),shells[a].size())=
            result.J.block(offsets_[a],offsets_[b],shells[a].size(),shells[b].size()).transpose();
    }
    if(!result.J.allFinite()) throw std::overflow_error("MANTA BUPO nonfinite Coulomb result");
    double energy=0.,energy_error=0.;
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<n;++j) accumulate(.5*density(i,j)*result.J(i,j),energy,energy_error);
    receipt.coulomb_energy=energy+energy_error;
    if(!std::isfinite(receipt.coulomb_energy)) throw std::overflow_error("MANTA BUPO Coulomb energy exceeds binary64");
    if(receipt.near_shell_calls!=plan_.near_shell_calls || receipt.moment_panels!=plan_.moment_panels || receipt.far_blocks!=plan_.far_blocks)
        throw std::logic_error("MANTA BUPO executed work disagrees with its immutable plan");
    final_clock.finish(receipt.finalize_wall_seconds);
    receipt.wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return result;
}
} // namespace vibeqc::manta
