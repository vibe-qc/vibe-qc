#pragma once
#include "vibeqc/manta_fitting.hpp"
#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/manta_extents.hpp"
#include "vibeqc/periodic.hpp"
#include <memory>

namespace vibeqc::manta {
// Raw erfc pieces of Ye and Berkelbach, JCP 154,131104 (2021), Eqs.9,12.
// The existing GDF sphere/capsule selector owns the numerical image domain.
// Finite SR zero modes remain included; no jellium subtraction or exchange
// singular correction belongs to this source. No fitting solve is performed.
struct PeriodicGDFShortRangeOptions {
    double omega=0.5,pair_cutoff_bohr=6.,auxiliary_cutoff_bohr=6.;
    double integral_screen_error=0.;
    // Separate coupling-only omission relative to the zero-screen finite
    // source. The metric stays exact; cannot combine with common screening.
    double coupling_screen_error=0.;
    // Experimental ordered-image erfc multipoles; exact metric and direct
    // fallback. Mutually exclusive with either screening control.
    double coupling_multipole_error=0.;
    int coupling_multipole_order=6;
    bool coupling_multipole_hierarchy=false;
    bool coupling_multipole_reuse=true; // False preserves the lower-storage streaming path.
    Count coupling_multipole_partition_bytes=64*1024*1024; // Bounded retained geometry/error decisions; zero streams.
    bool coupling_multipole_pair_tree=false; // Global spatial pair aggregation; changes the approximate coupling identity.
    bool coupling_multipole_global_actions=false; // Call-wide pair charges/fields; requires compression and preparation reuse.
    bool coupling_multipole_compress=false; // Hierarchical paired actions contract density before translation.
    ExtentOptions coupling_multipole_extents;
    Count image_candidate_cap=1000000;
    bool compute_metric=true;
};
// Physical interaction is independent of the numerical Coulomb split above.
// erfc retains its finite p=0 coefficient; Coulomb and erf omit their poles.
struct PeriodicGDFKernelOptions {
    std::string kernel="coulomb";
    double omega=0.;
};
struct PeriodicGDFMultipolePlan {
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicGDFShortRangePlan {
    Count n_aux=0,n_orb=0,momenta=0;
    bool compute_metric=true;
    Count metric_image_candidates=0,pair_image_candidates=0,auxiliary_image_candidates=0;
    Count metric_shell_calls=0,three_center_shell_calls=0,shell_calls=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,driver_live_bytes=0,modeled_peak_bytes=0;
    FittingPlan metric,three_center;
    PeriodicGDFMultipolePlan multipole;
};
struct PeriodicGDFShortRangeReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",normalization="libint-shell-axial-v1";
    std::string algorithm="finite-3d-gdf-short-range",kernel="erfc";
    std::string convention="raw-erfc;finite-zero-included;no-jellium-subtraction;no-exchange-correction";
    std::string orbital_basis_identity,auxiliary_basis_identity;
    std::string metric_source_identity,three_center_source_identity,source_identity;
    double omega=0.,pair_cutoff_bohr=0.,auxiliary_cutoff_bohr=0.,integral_screen_error=0.;
    double coupling_screen_error=0.,coupling_multipole_error=0.;
    bool coupling_multipole_hierarchy=false,coupling_multipole_pair_tree=false;
    double evaluated_multipole_error=0.; // Maximum image-summed shell-tile bound.
    Count multipole_trials=0,multipole_images=0,multipole_moment_panels=0,multipole_primitive_products=0;
    Count multipole_envelope_evaluations=0,multipole_candidate_images=0;
    Count multipole_tree_nodes=0,multipole_far_blocks=0,multipole_translations=0;
    Count multipole_reused_extents=0,multipole_reused_moment_panels=0;
    Count multipole_partition_builds=0,multipole_partition_reuses=0,multipole_partition_bypasses=0;
    Count metric_shell_calls=0,three_center_shell_calls=0,pair_shell_images=0,primitive_products=0;
    Count phase_products=0,phase_accumulations=0,modeled_peak_bytes=0,threads=1;
};
struct PeriodicGDFShortRangeResult {
    PeriodicGDFShortRangePlan plan;
    PeriodicGDFShortRangeReceipt receipt;
    // Row-major [P,Q] and [ket k,P,mu,nu], individually owned.
    std::vector<std::complex<double>> metric_values,three_center_values;
};
PeriodicGDFShortRangePlan plan_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,Count momenta,const PeriodicGDFShortRangeOptions&,const ResourceBudget&);
PeriodicGDFShortRangeResult compute_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,const Eigen::Vector3d& q,const md::Matrix& ket_kpoints,
    const PeriodicGDFShortRangeOptions&,const ResourceBudget&);

// Complete finite Coulomb source, still unfactored. The reciprocal vectors
// are the caller's explicit G+q domain. The shared GDF jellium kernel removes
// the finite SR zero exactly once and omits the LR zero. No exchange seam.
// Ye and Berkelbach, JCP 154,131104 (2021), Eq.12; Bloch Fourier phases as
// Sun et al., JCP 147,164119 (2017), Eq.16.
struct PeriodicGDFPlan {
    Count n_aux=0,n_orb=0,momenta=0,vectors=0,panel_vectors=0,panel_count=0;
    bool compute_metric=true;
    Count shell_calls=0,primitive_products=0,vector_primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,modeled_peak_bytes=0;
    PeriodicGDFShortRangePlan short_range;
    FourierPlan auxiliary_fourier,ao_pair_fourier;
};
struct PeriodicGDFReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",normalization="libint-shell-axial-v1";
    std::string algorithm="finite-3d-range-separated-gdf",kernel="coulomb";
    std::string convention="jellium;finite-SR-zero-removed-once;LR-zero-omitted;no-exchange-correction";
    std::string metric_source_identity,three_center_source_identity,source_identity;
    // Same numerical domains/kernel, zero coupling screening, unchanged M.
    std::string reference_three_center_source_identity;
    double coupling_error_bound=0.,evaluated_coupling_error_bound=0.;
    bool coupling_error_evaluated=false;
    PeriodicGDFShortRangeReceipt short_range;
    bool finite_zero_removed=false;
    bool zero_transfer=false,physical_zero_included=false,real_short_range=true;
    double physical_omega=0.;
    Count reciprocal_vectors=0,auxiliary_fourier_shell_calls=0,pair_fourier_shell_calls=0;
    Count shell_calls=0,primitive_products=0,vector_primitive_products=0;
    Count phase_products=0,phase_accumulations=0,contraction_terms=0;
    Count modeled_peak_bytes=0,threads=1;
};
struct PeriodicGDFResult {
    PeriodicGDFPlan plan;
    PeriodicGDFReceipt receipt;
    std::vector<std::complex<double>> metric_values,three_center_values;
};
PeriodicGDFPlan plan_periodic_gdf(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,Count vectors,Count momenta,const PeriodicGDFShortRangeOptions&,const ResourceBudget&,
    const PeriodicGDFKernelOptions& = {});
PeriodicGDFResult compute_periodic_gdf(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,const Eigen::Vector3d& q,const md::Matrix& ket_kpoints,const md::Matrix& vectors,
    const PeriodicGDFShortRangeOptions&,const ResourceBudget&,const PeriodicGDFKernelOptions& = {});

struct PeriodicGDFTilesPlan {
    Count n_aux=0,n_orb=0,momenta=0,vectors=0,panel_vectors=0;
    Count metric_shell_calls=0,coupling_shell_calls=0,metric_primitive_products=0,coupling_primitive_products=0;
    Count metric_scalar_terms=0,coupling_scalar_terms=0,metric_kernel_terms=0,coupling_kernel_terms=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0;
    Count maximum_tile_elements=0,retained_bytes=0,metric_peak_bytes=0,coupling_peak_bytes=0,modeled_peak_bytes=0;
    Count external_live_bytes=0;
    Count preparation_cache_bytes=0,preparation_pair_slots=0,preparation_auxiliary_slots=0,preparation_support_slots=0;
    Count compressed_action_bytes=0,compressed_action_scalar_terms=0;
    Count global_action_bytes=0,global_action_scalar_terms=0,global_pair_slots=0,global_auxiliary_slots=0;
    Count pair_tree_bytes=0,pair_tree_action_bytes=0,pair_tree_scalar_terms=0;
    Count partition_cache_bytes=0,partition_workspace_bytes=0,partition_scalar_terms=0;
    double coupling_error_bound=0.;
    bool compute_metric=true;
};
struct PeriodicGDFPreparationInfo {
    bool enabled=false;
    std::string source_identity;
    Count pair_extents=0,auxiliary_extents=0,moment_panels=0,moment_elements=0,support_tiles=0;
    Count modeled_live_bytes=0,capacity_bytes=0;
};
struct PeriodicGDFPartitionInfo {
    bool enabled=false;
    std::string source_identity;
    Count entries=0,evictions=0,modeled_live_bytes=0,capacity_bytes=0;
};
struct PeriodicGDFPairState;
struct PeriodicGDFPairContractions;
struct PeriodicGDFPartitionCache;
struct PeriodicGDFMomentCache; // Private, source-owned lazy immutable panels.
struct PeriodicGDFMultipoleAction; // Private per-call compressed input/field state.
struct PeriodicGDFTile {
    std::vector<Count> shape,offsets;
    std::vector<std::complex<double>> values;
    PeriodicGDFReceipt receipt;
};
PeriodicGDFTilesPlan plan_periodic_gdf_tiles(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,Count vectors,Count momenta,const PeriodicGDFShortRangeOptions&,const ResourceBudget&,
    const PeriodicGDFKernelOptions& = {});

// One unweighted forward OR adjoint action over the fixed raw coupling.
// Consumers admit repeated actions and their other retained storage jointly.
struct PeriodicGDFCouplingPlan {
    Count n_aux=0,n_orb=0,momenta=0,maximum_tile_elements=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicGDFCouplingReceipt {
    std::string provider="manta",algorithm="ordered-periodic-gdf-coupling-v1";
    std::string convention="unweighted;forward=conj(T):X;adjoint=T*c;T(k-q,k)";
    std::string direction,coupling_identity,input_identity,snapshot_identity;
    std::string reference_coupling_identity;
    // Absolute per-output omission bound; excludes arithmetic roundoff.
    // maximum_coupling bounds max |T|; compressed actions use a factored triangle envelope.
    double coupling_error_bound=0.,input_l1_bound=0.,approximation_bound=0.,maximum_coupling=0.;
    // Euclidean output-error envelope; pair-tree actions retain shell-block
    // information. Other sources use sqrt(output size) * approximation_bound.
    double approximation_l2_bound=0.;
    Count multipole_compressed_pairs=0,multipole_compressed_nodes=0,multipole_action_products=0,multipole_action_translations=0;
    Count multipole_global_pairs=0,multipole_global_pair_reuses=0,multipole_global_auxiliary=0,multipole_global_projections=0,multipole_global_auxiliary_projections=0;
    Count multipole_pair_groups=0; // Accepted interactions representing multiple ordered pair images.
    bool approximation_evaluated=false; // Norm/magnitude diagnostics are inactive otherwise.
    Count coupling_passes=1,shell_tiles=0,shell_calls=0,primitive_products=0;
    Count multipole_trials=0,multipole_images=0,multipole_moment_panels=0;
    Count multipole_envelope_evaluations=0;
    Count multipole_tree_nodes=0,multipole_far_blocks=0,multipole_translations=0;
    Count multipole_reused_extents=0,multipole_reused_moment_panels=0;
    Count multipole_partition_builds=0,multipole_partition_reuses=0,multipole_partition_bypasses=0;
    Count integral_elements_consumed=0,contraction_products=0,modeled_peak_bytes=0;
};
struct PeriodicGDFCouplingResult {
    PeriodicGDFCouplingPlan plan;
    PeriodicGDFCouplingReceipt receipt;
    std::vector<Count> shape;
    std::vector<std::complex<double>> values;
    // Per-output omission envelopes, in values' order (zero for exact maps).
    // For an approximate source, reference_magnitude_bounds[P] bounds
    // max_{k,mu,nu}|T_exact(k,P,mu,nu)| for the same finite reference.
    // Empty for exact sources. Neither vector includes arithmetic error.
    std::vector<double> approximation_bounds,reference_magnitude_bounds;
    // Per-output absolute action envelope: |T_exact|^T |input| (forward)
    // or |T_exact| |input| (adjoint), with no cancellation credit. Thus it
    // also bounds every input dominated entrywise by |input|. Approximate
    // sources only; the same finite reference, excluding arithmetic error.
    std::vector<double> reference_action_bounds;
    // Audit only: exact stored residual tiles plus the exact real-arithmetic
    // moment DAG. Excludes integral construction and actual action roundoff.
    bool metric_norm_qualified=false;
    std::vector<double> metric_column_norm_bounds; // Auxiliary L1 norm per k/AO pair.
};
PeriodicGDFCouplingPlan plan_periodic_gdf_coupling(const PeriodicGDFTilesPlan&,const ResourceBudget&);
PeriodicGDFCouplingPlan plan_periodic_gdf_coupling_norm_bounds(const PeriodicGDFCouplingPlan&,const ResourceBudget&);

// Immutable finite source. One full dense metric or one [k,P,mu,nu] shell
// tile may be live/exported under this plan; no complete coupling tensor.
// Lazy source-owned extent/moment panels are published once and then read
// immutably; numerical source identity never depends on cache history.
// Work bounds cover one metric and one complete shell traversal, not a
// lifetime call quota. Consumers admit all of their repeated passes jointly.
class PeriodicGDFTiles;
struct PeriodicGDFPairTreePlan;
class PeriodicGDFPairTree;
PeriodicGDFPairTreePlan plan_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);
std::shared_ptr<PeriodicGDFPairTree> build_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);

// Count-only union of the original Gamma vectors and a complete nonzero
// reciprocal cube. The original order and numerical vectors are preserved.
struct PeriodicGDFDomainPlan {
    PeriodicGDFTilesPlan source;
    Count maximum_vectors=0,generation_bytes=0,generation_scalar_terms=0,generation_kernel_terms=0;
    Count modeled_peak_bytes=0,scalar_terms=0,kernel_terms=0;
};
Count periodic_gdf_domain_vector_capacity(Count original,Count cube);
class PeriodicGDFTiles {
public:
    PeriodicGDFTiles(const BasisSet& orbital,const BasisSet& auxiliary,const PeriodicSystem&,
        const Eigen::Vector3d& q,const md::Matrix& ket_kpoints,const md::Matrix& vectors,
        const PeriodicGDFShortRangeOptions&,const ResourceBudget&,const PeriodicGDFKernelOptions& = {});
    const PeriodicGDFTilesPlan& plan() const { return plan_; }
    const PeriodicGDFReceipt& source() const { return source_; }
    const BasisSet& orbital() const { return orbital_; }
    const BasisSet& auxiliary() const { return auxiliary_; }
    const Eigen::Matrix3d& lattice() const { return system_.lattice; }
    const md::Matrix& momenta() const { return k_; }
    const Eigen::Vector3d& transfer() const { return q_; }
    const PeriodicGDFShortRangeOptions& options() const { return options_; }
    const md::Matrix& vectors() const { return vectors_; }
    PeriodicGDFDomainPlan plan_domains(double pair_cutoff,double auxiliary_cutoff,Count cube,const ResourceBudget&) const;
    std::shared_ptr<const PeriodicGDFTiles> with_domains(double pair_cutoff,double auxiliary_cutoff,Count cube,const ResourceBudget&) const;
    PeriodicGDFDomainPlan plan_auxiliary_domains(const BasisSet&,double pair_cutoff,double auxiliary_cutoff,Count cube,const ResourceBudget&) const;
    std::shared_ptr<const PeriodicGDFTiles> with_auxiliary_domains(const BasisSet&,double pair_cutoff,double auxiliary_cutoff,Count cube,const ResourceBudget&) const;
    PeriodicGDFPreparationInfo preparation_info() const;
    PeriodicGDFPartitionInfo partition_info() const;
    // Recreate the same finite source with a tighter positive multipole
    // tolerance. Metric, physical kernel, geometry and extent policy stay fixed.
    std::shared_ptr<const PeriodicGDFTiles> refine_multipole(double,const ResourceBudget&) const;
    PeriodicGDFTile metric() const;
    PeriodicGDFTile evaluate(Count auxiliary_shell,Count bra_shell,Count ket_shell) const;
    PeriodicGDFCouplingPlan plan_coupling(const ResourceBudget&) const;
    PeriodicGDFCouplingPlan plan_coupling_norm_bounds(const ResourceBudget&) const;
    PeriodicGDFCouplingResult coupling_norm_bounds(const ResourceBudget&) const;
    // Arbitrary complex inputs; no Hermiticity, spin sum, weights or metric
    // solve is supplied here. The two maps are adjoints under the ordinary
    // complex Frobenius pairing. This is not a symmetry certificate.
    PeriodicGDFCouplingResult forward(const std::vector<std::complex<double>>&,const ResourceBudget&) const;
    PeriodicGDFCouplingResult adjoint(const std::vector<std::complex<double>>&,const ResourceBudget&) const;
private:
    friend PeriodicGDFPairTreePlan plan_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);
    friend std::shared_ptr<PeriodicGDFPairTree> build_periodic_gdf_pair_tree(const PeriodicGDFTiles&,const ResourceBudget&);
    friend struct PeriodicGDFPairContractions;
    friend PeriodicGDFShortRangeResult compute_periodic_gdf_short_range(const BasisSet&,const BasisSet&,
        const PeriodicSystem&,const Eigen::Vector3d&,const md::Matrix&,const PeriodicGDFShortRangeOptions&,const ResourceBudget&);
    PeriodicGDFTile pair_tree_tile(Count,Count,Count) const;
    PeriodicGDFTile pair_tree_tile_impl(Count,Count,Count) const;
    void prepare_pair_tree(PeriodicGDFShortRangeReceipt&) const;
    double apply_pair_tree(const std::vector<std::complex<double>>&,std::vector<std::complex<double>>&,
        std::vector<std::complex<double>>&,bool,PeriodicGDFCouplingReceipt&,
        std::vector<double>&,std::vector<double>&,std::vector<double>&) const;
    PeriodicGDFTile evaluate_impl(Count,Count,Count,PeriodicGDFMultipoleAction*,bool residual=false) const;
    PeriodicGDFCouplingResult apply_coupling(const std::vector<std::complex<double>>&,bool,const ResourceBudget&,bool verified=false) const;
    PeriodicGDFTilesPlan plan_;
    BasisSet orbital_,auxiliary_;
    PeriodicSystem system_; // lattice/dimension only; no implicit symmetry state
    Eigen::Vector3d q_;
    md::Matrix k_,vectors_;
    PeriodicGDFShortRangeOptions options_;
    PeriodicGDFKernelOptions physical_;
    PeriodicGDFReceipt source_;
    std::shared_ptr<PeriodicGDFMomentCache> preparation_;
    std::shared_ptr<PeriodicGDFPartitionCache> partitions_;
    std::shared_ptr<PeriodicGDFPairState> pair_state_;
};
} // namespace vibeqc::manta
