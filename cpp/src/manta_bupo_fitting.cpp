#include "vibeqc/manta_bupo_fitting.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
void word(vibeqc::detail::Sha256& h,std::uint64_t value) {
    std::array<std::uint8_t,8> bytes{};
    for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(value>>(56-8*k));
    h.update(bytes.data(),bytes.size());
}
void real(vibeqc::detail::Sha256& h,double x) {
    if(x==0.) x=0.; std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(h,bits);
}
void text(vibeqc::detail::Sha256& h,const std::string& s) {
    word(h,s.size()); h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
}
void accumulate(double value,double& sum,double& correction) {
    const double next=sum+value;
    correction+=std::abs(sum)>=std::abs(value)?(sum-next)+value:(value-next)+sum; sum=next;
}
void column(Eigen::MatrixXd& sum,Eigen::MatrixXd& correction,Count col,const Eigen::VectorXd& v) {
    for(Eigen::Index i=0;i<v.size();++i) accumulate(v[i],sum(i,col),correction(i,col));
}
md::Point delta(const md::Point& a,const md::Point& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
double norm(const md::Point& d) { return std::hypot(d[0],d[1],d[2]); }
double weighted(double error,double magnitude) {
    if(error==0. || magnitude==0.) return 0.;
    const double result=std::nextafter(error*magnitude*(1+32*std::numeric_limits<double>::epsilon()),std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA BUPO fitting approximation bound overflow");
    return result;
}
BupoFittingReceipt receipt(const Eigen::MatrixXd& input,const std::string& source,const char* direction) {
    BupoFittingReceipt r; r.source_identity=source; r.direction=direction;
    vibeqc::detail::Sha256 h; text(h,direction); word(h,input.rows()); word(h,input.cols());
    double correction=0.;
    for(Eigen::Index i=0;i<input.rows();++i) for(Eigen::Index j=0;j<input.cols();++j) {
        real(h,input(i,j)); accumulate(std::abs(input(i,j)),r.input_l1,correction);
    }
    r.input_identity=h.finish_hex(); r.input_l1+=correction;
    if(!std::isfinite(r.input_l1)) throw std::overflow_error("MANTA BUPO fitting input norm overflow");
    return r;
}
std::size_t angular_shell(const BasisSet& b) {
    std::size_t chosen=0;
    for(std::size_t i=1;i<b.nshells();++i) if(b.libint()[i].contr[0].l>b.libint()[chosen].contr[0].l) chosen=i;
    return chosen;
}
} // namespace
std::string BupoFittingCoupling::walk(const Consume& consume,Count& visits,Count* queries) const {
    const auto& nodes=tree_->nodes();
    std::vector<Count> stack; stack.reserve(1+7*tree_->depth());
    vibeqc::detail::Sha256 hash; text(hash,partition_seed_); visits=0;
    const auto multipoles=plan_multipoles(options_.order,budget_);
    const Count query_work=add(multipoles.kernel_terms,mul(128,multipoles.scalar_terms));
    for(Count p=0;p<auxiliary_.shells.size();++p) {
        const auto& aux=auxiliary_.shells[p]; stack.push_back(0);
        while(!stack.empty()) {
            const Count index=stack.back(); stack.pop_back();
            visits=add(visits,1); cap(visits,budget_.shell_calls,"MANTA BUPO fitting traversal budget exceeded");
            const auto& node=nodes[index];
            const double distance=norm(delta(node.center,aux.center));
            bool far=false; BubbleInteraction interaction;
            if(!node.always_near && !aux.always_near && distance>node.radius+aux.radius
               && (node.radius+aux.radius)/distance<options_.opening) {
                if(queries) {
                    *queries=add(*queries,1);
                    cap(add(plan_.kernel_terms,mul(3,mul(*queries,query_work))),budget_.kernel_terms,
                        "MANTA BUPO fitting admission work budget exceeded");
                }
                try {
                    interaction=errors_->auxiliary_interaction(index,aux);
                    far=interaction.error.total<=options_.coupling_tolerance;
                } catch(const std::overflow_error&) { /* Refine an unrepresentable far expansion. */ }
            }
            if(!far && node.child_count) {
                for(int child=node.child_count;child>0;--child) stack.push_back(node.children[child-1]);
                continue;
            }
            word(hash,p); word(hash,index); word(hash,far);
            consume(p,index,far?&interaction:nullptr);
        }
    }
    return hash.finish_hex();
}
BupoFittingCoupling::BupoFittingCoupling(const BasisSet& orbital,const BasisSet& auxiliary,
    const BupoFittingOptions& options,const ResourceBudget& budget):options_(options),budget_(budget) {
    margin(budget); require_finite(options.opening); require_finite(options.coupling_tolerance);
    if(options.order<0 || options.order>12 || options.opening<0. || options.opening>=1. || options.coupling_tolerance<0.)
        throw std::invalid_argument("MANTA BUPO fitting order, opening or coupling tolerance is invalid");
    const auto oc=census(orbital,6),ac=census(auxiliary,7);
    plan_.n_basis=orbital.nbasis(); plan_.n_auxiliary=auxiliary.nbasis();
    ResourceBudget stage=budget;
    // Input/export/compensated AO matrices, auxiliary vectors, and controls.
    stage.other_live_bytes=add(stage.other_live_bytes,add(mul(64,square(plan_.n_basis)),mul(128,plan_.n_auxiliary)));
    provider_=std::make_shared<FittingTiles>(std::vector<const BasisSet*>{&auxiliary,&orbital,&orbital},stage);
    const auto& fitting=provider_->plan();
    stage.other_live_bytes=fitting.modeled_peak_bytes-budget.backend_margin_bytes;
    auto ext=options.extents; ext.moment_order=options.order; ext.retain_full_moments=true;
    const auto pair_plan=plan_pair_extents(orbital,ext,stage);
    auto pairs=compute_pair_extents(orbital,ext,stage);
    stage.other_live_bytes=pair_plan.modeled_peak_bytes-budget.backend_margin_bytes;
    const auto aux_plan=plan_auxiliary_extents(auxiliary,ext,stage);
    auxiliary_=compute_auxiliary_extents(auxiliary,ext,stage);
    // |(pair|aux)| <= min(L1_pair * potential_aux, L1_aux * potential_pair).
    // A global uniform bound needs only linear envelope scans, not all pairs.
    double pair_l1=0.,pair_potential=0.,aux_l1=0.,aux_potential=0.;
    for(const auto& p:pairs.pairs) { pair_l1=std::max(pair_l1,p.l1_norm_bound); pair_potential=std::max(pair_potential,p.potential_bound); }
    for(const auto& p:auxiliary_.shells) { aux_l1=std::max(aux_l1,p.l1_norm_bound); aux_potential=std::max(aux_potential,p.potential_bound); }
    auto log_product=[](double a,double b) { return a==0. || b==0.?-std::numeric_limits<double>::infinity():std::log(a)+std::log(b); };
    const double logarithm=std::min(log_product(pair_l1,aux_potential),log_product(aux_l1,pair_potential));
    if(logarithm!=-std::numeric_limits<double>::infinity())
        plan_.maximum_exact_coupling=weighted(std::max(std::numeric_limits<double>::denorm_min(),
            std::exp(logarithm+512*std::numeric_limits<double>::epsilon()*(1+std::abs(logarithm)))),1.);
    stage.other_live_bytes=aux_plan.modeled_peak_bytes-budget.backend_margin_bytes;
    tree_=std::make_shared<BubbleTree>(pairs,options.bubbles,stage);
    plan_.shell_pairs=tree_->pairs().size(); plan_.nodes=tree_->nodes().size();
    const auto mp=plan_multipoles(options.order,budget); plan_.moments=mp.moments;
    stage.other_live_bytes=add(tree_->plan().modeled_peak_bytes-budget.backend_margin_bytes,
        add(mul(32,mul(plan_.nodes,plan_.moments)),add(2048,mul(64,tree_->depth()+1))));
    errors_=std::make_shared<BubbleErrorModel>(tree_,options.order,stage);
    ResourceBudget standalone=budget; standalone.other_live_bytes=0;
    const auto os=angular_shell(orbital),as=angular_shell(auxiliary);
    const auto pm=plan_pair_moments(orbital,{os,os},options.order,standalone);
    const auto am=plan_auxiliary_moments(auxiliary,as,options.order,standalone);
    // Both panel workspaces can coexist with the query matrix in a far pass.
    plan_.modeled_peak_bytes=add(errors_->plan().modeled_peak_bytes,
        add(pm.modeled_peak_bytes-budget.backend_margin_bytes,am.modeled_peak_bytes-budget.backend_margin_bytes));
    cap(plan_.modeled_peak_bytes,budget.memory_bytes,"MANTA BUPO fitting combined workspace memory budget exceeded");
    plan_.kernel_terms=add(pair_plan.kernel_terms,add(aux_plan.kernel_terms,add(tree_->plan().kernel_terms,errors_->plan().kernel_terms)));
    cap(plan_.kernel_terms,budget.kernel_terms,"MANTA BUPO fitting setup work budget exceeded");
    partition_seed_="manta.bupo-fitting.partition.v1;"+errors_->source_identity()+";"+auxiliary_.receipt.source_identity;
    Count queries=0;
    plan_.partition_identity=walk([&](Count p,Count index,const BubbleInteraction* far) {
        if(far) {
            ++plan_.far_blocks;
            plan_.maximum_coupling_error=std::max(plan_.maximum_coupling_error,far->error.total);
            plan_.maximum_compact_error=std::max(plan_.maximum_compact_error,far->error.compact_remainder);
            plan_.maximum_penetration_error=std::max(plan_.maximum_penetration_error,far->error.tail_penetration);
            plan_.maximum_moment_tail_error=std::max(plan_.maximum_moment_tail_error,far->error.full_moment_tail);
        } else {
            const auto& node=tree_->nodes()[index];
            for(Count k=node.begin;k<node.end;++k) {
                const auto& s=tree_->pairs()[tree_->pair_order()[k]].shells;
                ++plan_.near_shell_calls;
                plan_.near_primitive_products=add(plan_.near_primitive_products,mul(auxiliary.libint()[p].alpha.size(),
                    mul(orbital.libint()[s[0]].alpha.size(),orbital.libint()[s[1]].alpha.size())));
            }
        }
    },plan_.traversal_visits,&queries);
    Count moment_work=0,translation_calls=0;
    if(plan_.far_blocks) {
        plan_.pair_moment_panels=plan_.shell_pairs;
        plan_.auxiliary_moment_panels=auxiliary.nshells();
        plan_.moment_primitive_products=add(pair_plan.primitive_pairs,ac.primitives);
        const Count pscale=std::max<Count>(1,(pair_plan.maximum_pair_primitives+pm.primitive_pairs-1)/pm.primitive_pairs);
        const Count ascale=std::max<Count>(1,(aux_plan.maximum_shell_primitives+am.primitives-1)/am.primitives);
        moment_work=add(mul(plan_.pair_moment_panels,mul(pm.kernel_terms,pscale)),mul(plan_.auxiliary_moment_panels,mul(am.kernel_terms,ascale)));
        translation_calls=plan_.nodes-1;
    }
    const Count setup_calls=add(pair_plan.shell_pairs,aux_plan.shells);
    const Count moment_calls=add(plan_.pair_moment_panels,plan_.auxiliary_moment_panels);
    plan_.shell_calls=add(setup_calls,add(mul(3,plan_.traversal_visits),mul(2,add(plan_.near_shell_calls,moment_calls))));
    plan_.primitive_products=add(add(pair_plan.primitive_pairs,aux_plan.primitives),mul(2,add(plan_.near_primitive_products,plan_.moment_primitive_products)));
    plan_.scalar_terms=add(pair_plan.scalar_terms,add(aux_plan.scalar_terms,add(tree_->plan().scalar_terms,errors_->plan().scalar_terms)));
    plan_.scalar_terms=add(plan_.scalar_terms,mul(2,add(fitting.integral_elements,add(
        mul(plan_.pair_moment_panels,mul(square(oc.maximum_size),plan_.moments)),add(
        mul(plan_.auxiliary_moment_panels,mul(ac.maximum_size,plan_.moments)),mul(add(translation_calls,plan_.far_blocks),mp.scalar_terms))))));
    plan_.scalar_terms=std::max({plan_.scalar_terms,pm.scalar_terms,am.scalar_terms,mp.scalar_terms});
    const Count query_work=add(mp.kernel_terms,mul(128,mp.scalar_terms));
    plan_.kernel_terms=add(plan_.kernel_terms,add(mul(3,mul(queries,query_work)),add(mul(2,add(fitting.kernel_terms,
        add(moment_work,mul(translation_calls,mp.kernel_terms)))),add(mul(768,plan_.traversal_visits),mul(4,plan_.scalar_terms)))));
    plan_.kernel_terms=std::max({plan_.kernel_terms,pm.kernel_terms,am.kernel_terms,mp.kernel_terms});
    // Every preflight is covered even when the selected partition is all far.
    plan_.shell_calls=std::max(plan_.shell_calls,fitting.shell_calls);
    plan_.primitive_products=std::max({plan_.primitive_products,fitting.primitive_products,pm.primitive_pairs,am.primitives});
    cap(plan_.shell_calls,budget.shell_calls,"MANTA BUPO fitting aggregate shell/traversal budget exceeded");
    cap(plan_.primitive_products,budget.primitive_quartets,"MANTA BUPO fitting aggregate primitive budget exceeded");
    cap(plan_.scalar_terms,budget.scalar_terms,"MANTA BUPO fitting aggregate scalar budget exceeded");
    cap(plan_.kernel_terms,budget.kernel_terms,"MANTA BUPO fitting aggregate work budget exceeded");
    vibeqc::detail::Sha256 hash; text(hash,"manta.bupo-fitting.v1;canonical-pairs;raw-cartesian;full-coulomb;");
    text(hash,provider_->source_identity()); text(hash,plan_.partition_identity); real(hash,options.coupling_tolerance);
    source_=hash.finish_hex();
}
void BupoFittingCoupling::finish(BupoFittingReceipt& r) const {
    if(r.partition_identity!=plan_.partition_identity || r.traversal_visits!=plan_.traversal_visits
       || r.near_shell_calls!=plan_.near_shell_calls || r.near_primitive_products!=plan_.near_primitive_products
       || r.far_blocks!=plan_.far_blocks || r.pair_moment_panels!=plan_.pair_moment_panels
       || r.auxiliary_moment_panels!=plan_.auxiliary_moment_panels || r.moment_primitive_products!=plan_.moment_primitive_products)
        throw std::logic_error("MANTA BUPO fitting paired partition/work identity mismatch");
    r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    r.approximation_bound=weighted(plan_.maximum_coupling_error,r.input_l1);
    r.compact_bound=weighted(plan_.maximum_compact_error,r.input_l1);
    r.penetration_bound=weighted(plan_.maximum_penetration_error,r.input_l1);
    r.moment_tail_bound=weighted(plan_.maximum_moment_tail_error,r.input_l1);
}
BupoFittingForward BupoFittingCoupling::forward(const Eigen::MatrixXd& density) const {
    if(density.rows()!=Eigen::Index(plan_.n_basis) || density.cols()!=density.rows() || !density.allFinite())
        throw std::invalid_argument("MANTA BUPO fitting density must be finite and match the orbital basis");
    BupoFittingForward result; auto& r=result.receipt; r=receipt(density,source_,"forward");
    result.values=Eigen::VectorXd::Zero(plan_.n_auxiliary); Eigen::VectorXd correction=result.values;
    const auto& orbital=provider_->basis(1).libint(); const auto& auxiliary=provider_->basis(0).libint();
    const auto oo=orbital.shell2bf(),ao=auxiliary.shell2bf();
    const auto& nodes=tree_->nodes(); const auto& pairs=tree_->pairs(); const auto& order=tree_->pair_order();
    auto weights=[&](Count p) {
        const auto& s=pairs[p].shells; const Count na=orbital[s[0]].size(),nb=orbital[s[1]].size();
        Eigen::VectorXd value(na*nb);
        for(Count i=0;i<na;++i) for(Count j=0;j<nb;++j)
            value[i*nb+j]=density(oo[s[0]]+i,oo[s[1]]+j)+(s[0]==s[1]?0.:density(oo[s[1]]+j,oo[s[0]]+i));
        return value;
    };
    Eigen::MatrixXd moments,compensation;
    if(plan_.far_blocks) {
        moments=Eigen::MatrixXd::Zero(plan_.moments,plan_.nodes); compensation=moments;
        for(Count n=0;n<nodes.size();++n) if(!nodes[n].child_count)
            for(Count k=nodes[n].begin;k<nodes[n].end;++k) {
                const Count p=order[k]; const auto& s=pairs[p].shells;
                const auto panel=md::shell_moments(orbital[s[0]],orbital[s[1]],nodes[n].center,options_.order);
                column(moments,compensation,n,panel.transpose()*weights(p));
                ++r.pair_moment_panels; r.moment_primitive_products+=orbital[s[0]].alpha.size()*orbital[s[1]].alpha.size();
            }
        // BUPO upward moments, Neese et al. (2025), DOI 10.1021/acs.jpca.4c07415.
        // The adjoint below uses exactly the same W and T matrices transposed.
        for(Count reverse=nodes.size();reverse>0;--reverse) {
            const Count n=reverse-1; moments.col(n)+=compensation.col(n);
            if(n) {
                const Count parent=nodes[n].parent;
                const auto w=multipole_translation(delta(nodes[n].center,nodes[parent].center),options_.order,budget_);
                column(moments,compensation,parent,w.values*moments.col(n));
            }
        }
    }
    Count active=std::numeric_limits<Count>::max(); md::Matrix aux_panel;
    r.partition_identity=walk([&](Count p,Count n,const BubbleInteraction* far) {
        if(plan_.far_blocks && active!=p) {
            aux_panel=md::shell_auxiliary_moments(auxiliary[p],auxiliary_.shells[p].center,options_.order);
            active=p; ++r.auxiliary_moment_panels; r.moment_primitive_products+=auxiliary[p].alpha.size();
        }
        if(far) {
            const Eigen::VectorXd contribution=aux_panel*(far->matrix.values.transpose()*moments.col(n));
            for(Eigen::Index i=0;i<contribution.size();++i) accumulate(contribution[i],result.values[ao[p]+i],correction[ao[p]+i]);
            ++r.far_blocks;
        } else for(Count k=nodes[n].begin;k<nodes[n].end;++k) {
            const Count pair=order[k]; const auto& s=pairs[pair].shells;
            const auto tile=provider_->evaluate({p,s[0],s[1]});
            const Eigen::VectorXd contribution=tile.values*weights(pair);
            for(Eigen::Index i=0;i<contribution.size();++i) accumulate(contribution[i],result.values[ao[p]+i],correction[ao[p]+i]);
            ++r.near_shell_calls; r.near_primitive_products+=tile.primitive_products;
        }
    },r.traversal_visits);
    result.values+=correction; finish(r);
    if(!result.values.allFinite()) throw std::overflow_error("MANTA BUPO fitting forward contraction overflow");
    return result;
}
BupoFittingAdjoint BupoFittingCoupling::adjoint(const Eigen::VectorXd& input) const {
    if(input.size()!=Eigen::Index(plan_.n_auxiliary) || !input.allFinite())
        throw std::invalid_argument("MANTA BUPO fitting coefficients must be finite and match the auxiliary basis");
    BupoFittingAdjoint result; auto& r=result.receipt; r=receipt(input,source_,"adjoint");
    result.values=Eigen::MatrixXd::Zero(plan_.n_basis,plan_.n_basis); Eigen::MatrixXd correction=result.values;
    const auto& orbital=provider_->basis(1).libint(); const auto& auxiliary=provider_->basis(0).libint();
    const auto oo=orbital.shell2bf(),ao=auxiliary.shell2bf();
    const auto& nodes=tree_->nodes(); const auto& pairs=tree_->pairs(); const auto& order=tree_->pair_order();
    auto add_pair=[&](Count p,const Eigen::VectorXd& values) {
        const auto& s=pairs[p].shells; const Count nb=orbital[s[1]].size();
        for(Eigen::Index k=0;k<values.size();++k) {
            const Count a=oo[s[0]]+k/nb,b=oo[s[1]]+k%nb;
            accumulate(values[k],result.values(a,b),correction(a,b));
        }
    };
    Eigen::MatrixXd fields,compensation;
    if(plan_.far_blocks) { fields=Eigen::MatrixXd::Zero(plan_.moments,plan_.nodes); compensation=fields; }
    Count active=std::numeric_limits<Count>::max(); Eigen::VectorXd aux_moments;
    r.partition_identity=walk([&](Count p,Count n,const BubbleInteraction* far) {
        if(plan_.far_blocks && active!=p) {
            const auto panel=md::shell_auxiliary_moments(auxiliary[p],auxiliary_.shells[p].center,options_.order);
            aux_moments=panel.transpose()*input.segment(ao[p],auxiliary[p].size());
            active=p; ++r.auxiliary_moment_panels; r.moment_primitive_products+=auxiliary[p].alpha.size();
        }
        if(far) {
            column(fields,compensation,n,far->matrix.values*aux_moments); ++r.far_blocks;
        } else for(Count k=nodes[n].begin;k<nodes[n].end;++k) {
            const Count pair=order[k]; const auto& s=pairs[pair].shells;
            const auto tile=provider_->evaluate({p,s[0],s[1]});
            add_pair(pair,tile.values.transpose()*input.segment(ao[p],auxiliary[p].size()));
            ++r.near_shell_calls; r.near_primitive_products+=tile.primitive_products;
        }
    },r.traversal_visits);
    if(plan_.far_blocks) for(Count n=0;n<nodes.size();++n) {
        if(n) {
            const Count parent=nodes[n].parent;
            const auto w=multipole_translation(delta(nodes[n].center,nodes[parent].center),options_.order,budget_);
            column(fields,compensation,n,w.values.transpose()*fields.col(parent));
        }
        fields.col(n)+=compensation.col(n);
        if(!nodes[n].child_count) for(Count k=nodes[n].begin;k<nodes[n].end;++k) {
            const Count pair=order[k]; const auto& s=pairs[pair].shells;
            const auto panel=md::shell_moments(orbital[s[0]],orbital[s[1]],nodes[n].center,options_.order);
            add_pair(pair,panel*fields.col(n));
            ++r.pair_moment_panels; r.moment_primitive_products+=orbital[s[0]].alpha.size()*orbital[s[1]].alpha.size();
        }
    }
    result.values+=correction;
    for(const auto& pair:pairs) {
        const auto& s=pair.shells; if(s[0]==s[1]) continue;
        result.values.block(oo[s[1]],oo[s[0]],orbital[s[1]].size(),orbital[s[0]].size())=
            result.values.block(oo[s[0]],oo[s[1]],orbital[s[0]].size(),orbital[s[1]].size()).transpose();
    }
    finish(r);
    if(!result.values.allFinite()) throw std::overflow_error("MANTA BUPO fitting adjoint contraction overflow");
    return result;
}
} // namespace vibeqc::manta
