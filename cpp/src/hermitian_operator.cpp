#include "vibeqc/hermitian_operator.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace vibeqc {
namespace {
template<class Matrix> void validate(const Matrix& a) {
    if (a.rows() == 0 || a.cols() != a.rows() || !a.allFinite())
        throw std::invalid_argument("Hermitian operator must be finite, square and nonempty");
    if ((a - a.adjoint()).norm() > 1e-10 * std::max(1.0, a.norm()))
        throw std::invalid_argument("Hermitian operator has a non-Hermitian component");
}

template<class Scalar> HermitianSolution<Scalar> solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& a,
    const DavidsonOptions* opts) {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    validate(a);
    HermitianSolution<Scalar> out;
    const double tol = opts ? opts->conv_tol : 1e-7;
    if (!std::isfinite(tol) || tol <= 0)
        throw std::invalid_argument("Hermitian solve tolerance must be finite and positive");
    if (opts) {
        auto result = [&]() {
            if constexpr (std::is_same_v<Scalar, double>)
                return davidson_solve(a, *opts);
            else
                return davidson_solve_hermitian(a, *opts);
        }();
        if (!result.converged)
            throw std::runtime_error("Hermitian operator: Davidson did not converge");
        out.eigenvalues = result.eigenvalues;
        out.eigenvectors = result.eigenvectors;
        out.n_iter = result.n_iter;
        out.used_dense = result.n_iter == 0;
    } else {
        Eigen::SelfAdjointEigenSolver<Matrix> es(a);
        if (es.info() != Eigen::Success)
            throw std::runtime_error("Hermitian operator: dense diagonalization failed");
        out.eigenvalues = es.eigenvalues();
        out.eigenvectors = es.eigenvectors();
        out.n_iter = 1;
    }
    const Matrix residual = a * out.eigenvectors
        - out.eigenvectors * out.eigenvalues.asDiagonal();
    out.residuals = residual.colwise().norm();
    out.orthogonality_error = (out.eigenvectors.adjoint() * out.eigenvectors
        - Matrix::Identity(out.eigenvalues.size(), out.eigenvalues.size())).norm();
    if (!out.residuals.allFinite() || out.residuals.maxCoeff() > tol
        || out.orthogonality_error > 1e-7)
        throw std::runtime_error("Hermitian operator: residual or orthogonality check failed");
    return out;
}

template<class Scalar> HermitianSolution<Scalar> fock_solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& f,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& x,
    const DavidsonOptions* opts) {
    validate(f);
    if (x.rows() != f.rows() || x.cols() == 0 || !x.allFinite())
        throw std::invalid_argument("Fock orthogonalizer has incompatible dimensions or values");
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    // validate(f) above is the refusal for a non-Hermitian builder. It runs on
    // f exactly as supplied, so a builder error is never repaired by dropping
    // its anti-Hermitian part. Only an f that passed it is projected, and it
    // is projected through its Hermitian part (QC672).
    //
    // X^H A X amplifies an anti-Hermitian roundoff A by up to
    // sigma_max(X)^2 = 1/lambda_min(S). A search Fock is a Pulay combination
    // sum_i c_i F_i (Pulay 1980, Eq. 3; 1982, procedure step 3) with
    // coefficients admitted up to 1e6 (diis.cpp), which scale the ~1e-16
    // builder roundoff of each F_i[D]; it measured 1.1e-12 in the rejected
    // f of QC672. The diffuse CH doublet (aug-cc-pVTZ,
    // sigma_max(X) 69, cond(S) 2.4e4) then lifted that to 1.02e-10 in the X
    // frame and crossed the guard, although every physical operator was
    // symmetric to 7e-16. F C = S C e is defined for the Hermitian F, and
    // LAPACK ?syev/?heev and Eigen's SelfAdjointEigenSolver read a single
    // triangle, i.e. the same Hermitian part. The unchanged guard below now
    // checks only the projection's own arithmetic.
    const Matrix f_hermitian = 0.5 * (f + f.adjoint());
    const Matrix projected = x.adjoint() * f_hermitian * x;
    validate(projected);
    const Matrix hermitian = 0.5 * (projected + projected.adjoint());
    return solve<Scalar>(hermitian, opts);
}
} // namespace

HermitianSolution<double> solve_hermitian_matrix(
    const Eigen::MatrixXd& a, OperatorKind, const DavidsonOptions* opts) {
    return solve<double>(a, opts);
}
HermitianSolution<std::complex<double>> solve_hermitian_matrix(
    const Eigen::MatrixXcd& a, OperatorKind, const DavidsonOptions* opts) {
    return solve<std::complex<double>>(a, opts);
}
HermitianSolution<double> solve_fock_matrix(
    const Eigen::MatrixXd& f, const Eigen::MatrixXd& x,
    const DavidsonOptions* opts, OperatorKind) {
    return fock_solve<double>(f, x, opts);
}
HermitianSolution<std::complex<double>> solve_fock_matrix(
    const Eigen::MatrixXcd& f, const Eigen::MatrixXcd& x,
    const DavidsonOptions* opts, OperatorKind) {
    return fock_solve<std::complex<double>>(f, x, opts);
}
HermitianSolution<double> solve_one_electron_metric(
    const Eigen::MatrixXd& a, const Eigen::MatrixXd& s) {
    validate(a);
    validate(s);
    if (s.rows() != a.rows())
        throw std::invalid_argument("Operator and metric dimensions differ");
    Eigen::LLT<Eigen::MatrixXd> llt(s);
    if (llt.info() != Eigen::Success)
        throw std::invalid_argument("One-electron overlap must be positive definite");
    const Eigen::MatrixXd x = llt.matrixU().solve(Eigen::MatrixXd::Identity(s.rows(), s.cols()));
    auto out = fock_solve<double>(a, x, nullptr);
    out.eigenvectors = (x * out.eigenvectors).eval();
    out.residuals = (a * out.eigenvectors
        - s * out.eigenvectors * out.eigenvalues.asDiagonal()).colwise().norm();
    out.orthogonality_error = (out.eigenvectors.transpose() * s * out.eigenvectors
        - Eigen::MatrixXd::Identity(s.rows(), s.cols())).norm();
    if (!out.residuals.allFinite() || out.orthogonality_error > 1e-7)
        throw std::runtime_error("One-electron metric normalization failed");
    return out;
}
} // namespace vibeqc
