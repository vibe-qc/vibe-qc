#pragma once
#include "vibeqc/semiempirical/methods/indo/indo_engine.hpp"
#include <Eigen/SVD>
#include <limits>
namespace vibeqc::semiempirical::indo {

// Variational continuation of an integer-occupied CCM seed. Use the existing
// Neese (2000) SOSCF L-BFGS direction, with Armijo descent on the same quadratic
// INDO functional. Recover the seed's occupied subspace before optimizing: the
// Hcore eigenvectors do not represent a supplied frontier projector.
//
// The SVD gives principal occupied/virtual rotations. Evaluate their energy
// difference using sin(theta)^2 and sin(theta)*cos(theta), not subtraction of
// large total energies or nearly equal projectors. Scale the stored quasi-Newton step
// after a backtrack so its next curvature pair uses the accepted rotation.
// Every trial Fock consumes the remaining budget. On exhaustion, return the
// last accepted physical density and its energy as unconverged. The initial
// Fock/energy were already counted during seed construction.
// Twin: msindo_ccm_orbitals.py. References: Neese, Chem. Phys. Lett. 325, 93-98
// (2000); Lehtola et al., Molecules 25, 1218 (2020), doi:10.3390/molecules25051218.

inline MsindoResult ccm_minimize_orbitals(
    Eigen::MatrixXd P, Eigen::MatrixXd F, double energy, int nocc, int budget, double tolerance,
    const std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd &)>
        &evaluate) {
    const int n = P.rows(), nv = n - nocc;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> projector(P);
    Eigen::MatrixXd Co = projector.eigenvectors().rightCols(nocc);
    Eigen::MatrixXd Cv = projector.eigenvectors().leftCols(nv);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> occ(Co.transpose() * F * Co);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> vir(Cv.transpose() * F * Cv);
    Eigen::MatrixXd C(n, n);
    C.leftCols(nocc) = Co * occ.eigenvectors();
    C.rightCols(nv) = Cv * vir.eigenvectors();
    Eigen::VectorXd eps(n);
    eps.head(nocc) = occ.eigenvalues();
    eps.tail(nv) = vir.eigenvalues();
    SOSCF optimizer;
    double previous_energy = std::numeric_limits<double>::infinity();
    MsindoResult out;
    while (out.n_iter < budget) {
        if ((F * P - P * F).cwiseAbs().maxCoeff() < tolerance &&
            std::abs(energy - previous_energy) < tolerance) {
            out.converged = true;
            break;
        }
        previous_energy = energy;
        auto proposed = optimizer.step(F, C, eps, nocc);
        out.orbital_optimization_used = true;
        Co = C.leftCols(nocc);
        Cv = C.rightCols(nv);
        Eigen::MatrixXd overlap = Cv.transpose() * (proposed.C.leftCols(nocc) - Co);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(overlap,
                                              Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::MatrixXd U = svd.matrixU(), V = svd.matrixV();
        Eigen::VectorXd angles = svd.singularValues().array().max(0.).min(1.).asin();
        const Eigen::MatrixXd A = Co * V, B = Cv * U;
        const Eigen::VectorXd gaps = (B.array() * (F * B).array()).colwise().sum().transpose() -
                                     (A.array() * (F * A).array()).colwise().sum().transpose();
        const Eigen::VectorXd cross = (A.array() * (F * B).array()).colwise().sum().transpose();
        const double slope = 4 * angles.dot(cross);
        double alpha = 1.;
        bool accepted = false;
        Eigen::MatrixXd Cot, Cvt, Pt, Ft;
        double Et = energy;
        for (int line = 0; line < 24 && out.n_iter < budget; ++line) {
            const Eigen::VectorXd theta = alpha * angles;
            const Eigen::VectorXd sn = theta.array().sin(), cs = theta.array().cos();
            const Eigen::VectorXd cm = -2 * (.5 * theta.array()).sin().square();
            Cot = Co + (A * cm.asDiagonal() + B * sn.asDiagonal()) * V.transpose();
            Cvt = Cv + (B * cm.asDiagonal() - A * sn.asDiagonal()) * U.transpose();
            Pt = 2 * Cot * Cot.transpose();
            auto next = evaluate(Pt);
            ++out.n_iter;
            Ft = std::move(next.first);
            Et = next.second;
            const Eigen::MatrixXd As = A * sn.asDiagonal(), Bs = B * sn.asDiagonal();
            const Eigen::MatrixXd Ac = A * cs.asDiagonal();
            const Eigen::MatrixXd delta = 2 * (Bs * Bs.transpose() - As * As.transpose() +
                                               Ac * Bs.transpose() + Bs * Ac.transpose());
            const double change = 2 * (sn.array().square() * gaps.array() +
                                       2 * sn.array() * cs.array() * cross.array())
                                          .sum() +
                                  .5 * (delta.array() * (Ft - F).array()).sum();
            if (change <= 1e-4 * alpha * slope) {
                accepted = true;
                break;
            }
            alpha *= .5;
        }
        if (!accepted)
            break;
        if (alpha != 1.)
            optimizer.scale_previous_step(alpha);
        C.leftCols(nocc) = Cot;
        C.rightCols(nv) = Cvt;
        P = std::move(Pt);
        F = std::move(Ft);
        energy = Et;
        eps = (C.transpose() * F * C).diagonal();
    }
    out.density = P;
    out.electronic_energy = energy;
    out.mo_energies = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(F).eigenvalues();
    return out;
}
} // namespace vibeqc::semiempirical::indo
