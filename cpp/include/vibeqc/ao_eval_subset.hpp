// Shell-subset AO evaluation for screened quadratures.
//
// Kept out of ao_eval.hpp so that adding it does not rebuild every AO
// consumer. The periodic XC kernels use it to evaluate only the shells of a
// lattice image whose Gaussian extent reaches a grid batch
// (compute_shell_radial_cutoffs); the dense evaluators stay the reference.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "basis.hpp"

namespace vibeqc {

// Evaluate the listed shells (indices into basis.libint(), any order) at
// `points` and pack their basis functions contiguously in list order:
// `values` becomes (n_points, sum of shell sizes). When `gradients` is not
// null it receives the three Cartesian gradient blocks of the same shape.
// Each column is bit-identical to the matching column of evaluate_ao /
// evaluate_ao_with_gradient: both call the same per-point shell kernel.
// Serial; callers parallelise over images or batches.
void evaluate_ao_shell_subset(const BasisSet& basis,
                              const Eigen::MatrixX3d& points,
                              const std::vector<int>& shells,
                              Eigen::MatrixXd& values,
                              std::array<Eigen::MatrixXd, 3>* gradients);

}  // namespace vibeqc
