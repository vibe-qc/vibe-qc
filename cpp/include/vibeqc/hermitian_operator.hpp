// Shared explicit Hermitian solve boundary. See docs/design/operator_contract.md.
#pragma once

#include "vibeqc/davidson.hpp"
#include <Eigen/Core>

namespace vibeqc {

enum class OperatorKind { Fock, KohnSham, OneElectron, Response, CI };

// Native callers borrow const matrices only for the duration of a synchronous
// solve. No pointer or factorization survives the call. Builders, occupations,
// density updates and SCF acceleration remain owned by the calling method.
template<class Scalar> struct HermitianSolution {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::VectorXd eigenvalues;
    Matrix eigenvectors;
    Eigen::VectorXd residuals;
    double orthogonality_error = 0;
    int n_iter = 0;
    bool used_dense = true;
};

// Standard-space residuals, Euclidean normalization, ascending roots. Dense
// means a full spectrum. A non-null Davidson option selects lowest roots and
// preserves its documented small-explicit-problem dense fallback. Both paths
// validate the returned eigenpairs and throw on failure.
HermitianSolution<double> solve_hermitian_matrix(
    const Eigen::MatrixXd& matrix, OperatorKind kind,
    const DavidsonOptions* davidson = nullptr);
HermitianSolution<std::complex<double>> solve_hermitian_matrix(
    const Eigen::MatrixXcd& matrix, OperatorKind kind,
    const DavidsonOptions* davidson = nullptr);

// X is supplied by the method's established overlap policy and may be
// rectangular. Eigenvectors returned here are in the orthonormal X frame;
// the caller back-transforms with X. No overlap truncation is hidden here.
HermitianSolution<double> solve_fock_matrix(
    const Eigen::MatrixXd& fock, const Eigen::MatrixXd& X,
    const DavidsonOptions* davidson = nullptr,
    OperatorKind kind = OperatorKind::Fock);
HermitianSolution<std::complex<double>> solve_fock_matrix(
    const Eigen::MatrixXcd& fock, const Eigen::MatrixXcd& X,
    const DavidsonOptions* davidson = nullptr,
    OperatorKind kind = OperatorKind::Fock);

// Strict SPD metric, with no rank truncation. AO eigenvectors and raw AO
// residuals are returned; convergence is checked in the whitened frame.
HermitianSolution<double> solve_one_electron_metric(
    const Eigen::MatrixXd& matrix, const Eigen::MatrixXd& overlap);

} // namespace vibeqc
