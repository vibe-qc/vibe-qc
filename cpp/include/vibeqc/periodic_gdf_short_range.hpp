// Raw, finite-cutoff RSGDF short-range integrals. Ye and Berkelbach,
// JCP 154, 131104 (2021), doi:10.1063/5.0046617, Eqs. (9), (12)-(14).
// These sources include the finite SR zero mode. The combined SR+LR
// assembler must remove it once; no exchange correction belongs here.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>
#include <Eigen/Dense>
#include "basis.hpp"
#include "periodic.hpp"

namespace vibeqc {

// Workspace estimate for the requested SR team, using the linked Libint ABI
// and recurrence stacks. Value and derivative calls use the same admission
// calculation. Concurrent outputs and LR panels are charged separately.
std::size_t gdf_short_range_workspace_bytes(
    const BasisSet& orbital, const BasisSet& aux, int n_threads,
    int derivative_order = 0, std::size_t n_atoms = 0);

struct GDFShortRangeBatch {
    std::size_t n_k = 0, n_aux = 0, n_orb = 0;
    // Row-major [ket k, P, mu, nu], no image-indexed dense storage.
    std::vector<std::complex<double>> data;
};

Eigen::MatrixXcd compute_gdf_sr_metric(
    const BasisSet& aux, const PeriodicSystem& system,
    const Eigen::Vector3d& q, double omega, double cutoff,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap = 256U * 1024U * 1024U,
    double integral_screen_error = 0.0);

// T(k-q,k) = sum_{R,T} exp(+i k.R-i q.T) (P_T | erfc(omega r)/r | mu_0 nu_R).
// AO-pair separation <= pair_cutoff; auxiliary distance to the line segment
// joining the AO centers <= auxiliary_cutoff. The latter contains every
// primitive Gaussian product center. Both cutoffs need convergence checks.
// A shared-q batch reuses each molecular integral across its ket momenta.
// Shell triples own disjoint output slices; no thread-private full tensor.
GDFShortRangeBatch compute_gdf_sr_three_center(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints, double omega,
    double pair_cutoff, double auxiliary_cutoff,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap = 256U * 1024U * 1024U,
    double integral_screen_error = 0.0);

struct GDFShortRangeWeightView {
    const std::complex<double>* data = nullptr;
    std::size_t n_k = 0, n_aux = 0, n_orb = 0;
};

// Atomic derivatives of Re sum(weight * raw integral), without conjugating
// weight. Shared image domains and fixed screening membership match the
// energy sources above. No atom-by-integral derivative tensor is allocated.
// Caller-owned weights/bases are borrowed; workspace admits engine estimates,
// shell scratch and one atom-gradient accumulator per worker.
Eigen::MatrixXd compute_gdf_sr_metric_gradient_weighted(
    const BasisSet& aux, const PeriodicSystem& system,
    const Eigen::Vector3d& q, double omega, double cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& weight,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap, double integral_screen_error = 0.0);

Eigen::MatrixXd compute_gdf_sr_three_center_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux, const PeriodicSystem& system,
    const Eigen::Vector3d& q, const Eigen::MatrixXd& ket_kpoints,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    GDFShortRangeWeightView weight,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap, double integral_screen_error = 0.0);

struct GDFRangeSeparatedIntegrals {
    Eigen::MatrixXcd metric;
    GDFShortRangeBatch three_center;
    std::size_t reciprocal_vector_count = 0;
};

struct GDFPlaneWaveProjection {
    Eigen::MatrixXcd metric;
    GDFShortRangeBatch three_center;
    // Same layout, with the second index labelling the supplied PW vectors.
    // Singular vectors have zero factors and are not counted as PW modes.
    GDFShortRangeBatch factors;
    std::size_t reciprocal_vector_count = 0;
};

// Selected bare-Coulomb PW Gram blocks and factors for a shared-q MDF fit.
// Uses the same AO-pair distance domain/normalization as the SR/LR source.
// No zero-mode restoration or Madelung term belongs in this projection.
GDFPlaneWaveProjection compute_gdf_plane_wave_projection(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double pair_cutoff, std::size_t output_byte_cap,
    std::size_t transient_byte_cap, std::size_t image_candidate_cap,
    bool compute_metric = true);

// Private atomic derivative of Re(sum(WM*M_PW) + sum(WT*T_PW)
// + sum(WF*F_PW)). Weights are unconjugated borrowed arrays. The second
// index of factor_weight labels the supplied reciprocal vectors, including
// zero vectors (whose factors and derivatives vanish). Fixed cell/q/k/G
// and fixed image membership: this is not a stress or cutoff derivative.
Eigen::MatrixXd compute_gdf_plane_wave_projection_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double pair_cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
    GDFShortRangeWeightView three_center_weight,
    GDFShortRangeWeightView factor_weight,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap);

// Full jellium Coulomb M(q), T(k-q,k), with the same finite AO-image
// domain in SR, LR and the zero-mode overlap. Reciprocal vectors are p=G+q;
// callers supply a converged omega-sized mesh. Explicit output and transient
// caps cover numeric arrays and an admitted libint engine workspace estimate.
// Borrowed bases, runtime/global library storage and allocator overhead are
// separate; the workspace cap is not a whole-process RSS limit.
GDFRangeSeparatedIntegrals compute_gdf_range_separated_integrals(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap, double integral_screen_error = 0.0,
    bool compute_metric = true);

// Derivative of Re(sum(metric_weight*M) + sum(three_center_weight*T)).
// Uses the energy source's pair-distance image domain and finite SR zero
// mode, including the derivative of its Bloch overlap. Weights are borrowed.
Eigen::MatrixXd compute_gdf_range_separated_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
    GDFShortRangeWeightView three_center_weight,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap, double integral_screen_error = 0.0);

}  // namespace vibeqc
