#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vibeqc/pair_natural_orbitals.hpp"

namespace vibeqc {

struct BoundedRestrictedTNORealView {
    const double* data = nullptr;
    std::size_t element_count = 0;
};

struct BoundedRestrictedTNOEdge {
    std::uint64_t occupied_i = 0, occupied_j = 0, rank = 0;
    BoundedRestrictedTNORealView coefficients;  // common_dimension * rank
    BoundedRestrictedTNORealView connected_doubles;  // rank * rank
};

struct BoundedRestrictedTNOInput {
    std::uint64_t n_occupied = 0, common_dimension = 0;
    // SORTED MULTISET, not a set of distinct occupied labels.
    std::array<std::uint64_t, 3> occupied{};
    // Fixed (ij,ik,jk) order. Repeated edges must have exactly identical
    // rank, pointers and extents. Storage/union columns are counted once;
    // all THREE density occurrences, including repetitions, are retained.
    std::array<BoundedRestrictedTNOEdge, 3> edges;
    BoundedRestrictedTNORealView virtual_fock;  // exact n*n, real symmetric
};

struct BoundedRestrictedTNOOptions {
    // Scientific selection controls. No molecular/periodic preset implied.
    // Strict occupation > cutoff; zero explicitly retains the COMPLETE UNION
    // including null occupations. It does NOT complete the common space.
    double occupation_cutoff = 0.0;
    // Gram eigenvalue (SQUARED singular-value) units; threshold=max(abs,
    // rel*largest). Nonempty union rank cannot be zero. Original UNIQUE
    // columns M are independently checked by ||M-Q Q^T M||_F below.
    double union_absolute_rank_cutoff = 0.0;
    double union_relative_rank_cutoff = 0.0;
    double maximum_union_column_reconstruction_error = 0.0;  // (0,1)
    // Numerical audit controls, distinct from scientific occupation cutoffs.
    double input_orthonormality_tolerance = 0.0;  // (0,1)
    double eigensystem_relative_reconstruction_tolerance = 0.0;  // (0,1)
    double eigenvector_orthogonality_tolerance = 0.0;  // (0,1)
    double union_negative_absolute_tolerance = 0.0;
    double union_negative_relative_tolerance = 0.0;
    double density_negative_absolute_tolerance = 0.0;  // physical occupations
    double density_negative_relative_tolerance = 0.0;
    double occupation_ambiguity_absolute_guard = 0.0;  // physical occupations
    double occupation_ambiguity_relative_guard = 0.0;
    HermitianJacobiOptions union_eigensolver, density_eigensolver, fock_eigensolver;
};

struct BoundedRestrictedTNOInventory {
    std::uint64_t numerical_replicas = 1;
    std::uint64_t external_node_bytes = 0;
    // Caller must count full retained owners not represented by selected
    // edge views/Fvv here; no physical provenance or alias discovery occurs.
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    std::uint64_t backend_margin_bytes_per_replica = 0;
};

struct BoundedRestrictedTNOCaps {
    std::uint64_t maximum_common_dimension = 0;
    std::uint64_t maximum_union_columns = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_node_bytes = 0;
    std::uint64_t maximum_control_storage_bytes_per_replica = 0;
    std::uint64_t maximum_work_units = 0;
};

struct BoundedRestrictedTNOMemoryPlan {
    std::uint64_t common_dimension = 0, unique_edge_count = 0;
    std::uint64_t union_columns = 0, union_rank_upper_bound = 0, maximum_edge_rank = 0;
    std::uint64_t borrowed_numerical_bytes = 0;
    std::uint64_t gram_phase_bytes = 0, union_phase_bytes = 0;
    std::uint64_t density_phase_bytes = 0, density_eigen_phase_bytes = 0;
    std::uint64_t selection_phase_bytes = 0, semicanonical_phase_bytes = 0;
    // Count-only upper bounds use u=t=min(n,m), never a guessed numerical rank.
    std::uint64_t peak_owned_numerical_bytes = 0, output_numerical_bytes_upper_bound = 0;
    std::uint64_t fixed_inventoried_object_bytes = 0, control_storage_bytes_per_replica = 0;
    std::uint64_t total_node_bytes = 0, work_units_upper_bound = 0;
};

struct BoundedRestrictedTNOEigensystemAudit {
    double scaled_matrix_frobenius_norm = 0.0;
    double scaled_reconstruction_frobenius_error = 0.0;
    double relative_reconstruction_error = 0.0;
    double orthogonality_frobenius_error = 0.0;
    // Reconstruction error plus polar-orthogonalization correction, computed
    // from measured binary64 norms. This is a conservative numerical decision
    // guard, NOT a rigorously rounded spectral interval certificate.
    double scaled_selection_error_guard = 0.0;
    int matrix_scale_exponent = 0;
    HermitianJacobiResult eigensolver;
};

struct BoundedRestrictedTNOResult {
    BoundedRestrictedTNOResult() = default;
    BoundedRestrictedTNOResult(const BoundedRestrictedTNOResult&) = delete;
    BoundedRestrictedTNOResult& operator=(const BoundedRestrictedTNOResult&) = delete;
    BoundedRestrictedTNOResult(BoundedRestrictedTNOResult&&) noexcept = default;
    BoundedRestrictedTNOResult& operator=(BoundedRestrictedTNOResult&&) noexcept = default;
    std::array<std::uint64_t, 3> occupied{};
    std::uint64_t union_rank = 0, retained_rank = 0;
    bool usable = false;
    BoundedRestrictedTNOOptions options;
    BoundedRestrictedTNOMemoryPlan memory;
    // Original density occupations in DESCENDING order, before Fock rotation.
    // Includes accepted tiny negative eigenvalues verbatim. They are not
    // columnwise labels of the subsequently semicanonicalized coefficients.
    std::vector<double> occupations;
    RestrictedPairSemicanonicalResult semicanonical;
    // Density audit matrix is in power-of-two-scaled amplitude-squared
    // units; physical density restores 2*amplitude_scale_exponent in addition
    // to the audit's own matrix_scale_exponent. Union Gram is dimensionless.
    BoundedRestrictedTNOEigensystemAudit union_audit, density_audit;
    double maximum_input_orthonormality_error = 0.0;
    double union_orthonormality_error = 0.0;
    double union_column_reconstruction_frobenius_error = 0.0;
    double minimum_occupation = 0.0, discarded_occupation_sum = 0.0;
    std::uint64_t negative_occupation_count = 0;
    int amplitude_scale_exponent = 0;
    std::uint64_t output_numerical_bytes = 0;
    std::string input_identity_sha256, result_identity_sha256;
};

// Metadata-only: reads fixed three edge descriptors/options, never numerical
// payload. Positive n/o, exact extents, checked native-address arithmetic.
// m=0 is valid and needs zero owned numerical bytes (zero owned/column caps
// then valid); Fvv, control, input scans and work are still nonzero.
BoundedRestrictedTNOMemoryPlan plan_bounded_restricted_triple_natural_orbitals(
    const BoundedRestrictedTNOInput&, const BoundedRestrictedTNOOptions&,
    const BoundedRestrictedTNOInventory&);

// Riplinger et al., JCP 139,134101 (2013), doi:10.1063/1.4821834,
// Eqs.10-13: D_T=(Dbar_ij+Dbar_ik+Dbar_jk)/3,
// Ttilde=(4T-2T^T)/(1+delta), Dbar=Ttilde*T^T+Ttilde^T*T.
// Equivalent PSD form [4*S*S^T+12*A*A^T]/(1+delta), S=(T+T^T)/2,
// A=(T-T^T)/2, projected into the orthonormal union of the three pair frames.
// Diagonal T is EXACT symmetric: Dbar_ii=2*T*T^T, NOT the periodic-MP2-PNO
// normalization. Connected CCSD doubles only; no MP2/IEPA/trace normalization,
// no disconnected singles product. This pure numerical leaf cannot certify
// convergence, source authenticity, locality, triples energy or (T) accuracy.
//
// All caps precede floating input scans and size-dependent allocation. No
// separately expanded common-space density/T2, concatenated n*m column
// matrix, Eigen or BLAS workspace. Only the m*m union Gram and u*u density
// are materialized; a complete union can of course have u=n. Real
// orthonormal input only; never discard complex lanes.
// Immutable snapshots/payload rehash and binary64/rounding/range gates fail
// closed. Scale losses that change amplitudes/density/occupations are errors.
// A positive cutoff may produce rank zero; no fallback or minimum rank.
BoundedRestrictedTNOResult bounded_restricted_triple_natural_orbitals(
    const BoundedRestrictedTNOInput&, const BoundedRestrictedTNOOptions&,
    const BoundedRestrictedTNOInventory&, const BoundedRestrictedTNOCaps&);

}  // namespace vibeqc
