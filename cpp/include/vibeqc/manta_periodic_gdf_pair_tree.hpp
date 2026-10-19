#pragma once
#include "vibeqc/manta_periodic_fitting.hpp"
#include "vibeqc/manta_bubbles.hpp"

namespace vibeqc::manta {
struct PeriodicGDFPairTreePlan {
    Count pair_capacity=0,node_capacity=0,auxiliary_image_capacity=0,shell_calls=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicGDFPairTreeEntry {
    Count bra_shell=0,ket_shell=0,pair_ordinal=0;
    std::array<long long,3> ket_image{},bra_frame{};
    md::Point bra_center{},ket_center{},translation{},frame_translation{};
};
struct PeriodicGDFCapsuleMask {
    std::string membership;
    Count pairs=0;
};
struct PeriodicGDFCapsulePartition {
    std::vector<Count> included_nodes;
    Count included_pairs=0,excluded_pairs=0,visited_nodes=0,refined_nodes=0;
    std::string source_identity;
};
class PeriodicGDFTiles;
class PeriodicGDFPairTree {
public:
    const PeriodicGDFPairTreePlan& plan() const {return plan_;}
    const std::string& source_identity() const {return source_;}
    const std::vector<PeriodicGDFPairTreeEntry>& entries() const {return entries_;}
    std::shared_ptr<const BubbleTree> tree() const {return tree_;}
    PeriodicGDFCapsuleMask classify(Count node,Count auxiliary_shell,const std::array<long long,3>& image) const;
    std::vector<std::array<long long,3>> auxiliary_images(Count auxiliary_shell) const;
    PeriodicGDFCapsulePartition partition(Count auxiliary_shell,const std::array<long long,3>& image) const;
    const PeriodicGDFShortRangeReceipt& preparation_receipt() const {return preparation_;}
    md::Point auxiliary_center(Count auxiliary,const std::array<long long,3>& image) const;
    std::vector<std::complex<double>> pair_phase(Count entry) const;
    std::complex<double> auxiliary_phase(Count auxiliary_shell,const std::array<long long,3>& image) const;
    std::array<long long,3> source_auxiliary_image(Count entry,Count auxiliary_shell,const std::array<long long,3>& image) const;
private:
    friend std::shared_ptr<PeriodicGDFPairTree> build_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);
    struct Capsule {md::Point begin{},end{};double spread=0.;};
    PeriodicGDFPairTreePlan plan_;
    PeriodicGDFShortRangeReceipt preparation_;
    std::string source_;
    std::vector<PeriodicGDFPairTreeEntry> entries_;
    std::shared_ptr<const BubbleTree> tree_;
    std::vector<Capsule> capsules_;
    std::vector<md::Point> auxiliary_centers_,bra_centers_;
    std::vector<std::array<long long,3>> auxiliary_frames_;
    Eigen::Matrix3d lattice_;
    Eigen::Vector3d q_;
    md::Matrix k_;
    double cutoff_=0.,pair_cutoff_=0.,roundoff_guard_=0.;
    Count image_cap_=0;
    Eigen::Vector3d position(Count auxiliary_shell,const std::array<long long,3>& image) const;
};
PeriodicGDFPairTreePlan plan_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);
std::shared_ptr<PeriodicGDFPairTree> build_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);

} // namespace vibeqc::manta
