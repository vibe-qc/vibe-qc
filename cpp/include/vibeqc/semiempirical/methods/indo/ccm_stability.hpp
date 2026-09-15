#pragma once

#include "vibeqc/davidson.hpp"
#include "vibeqc/semiempirical/methods/indo/indo_engine.hpp"

namespace vibeqc::semiempirical::indo {

// Restricted (same alpha/beta rotation) orbital stability in the orthogonal
// INDO basis. Lehtola, Blockhuys and Van Alsenoy, Molecules 25, 1218 (2020),
// Sec. 10, doi:10.3390/molecules25051218. The Hessian below is one quarter
// of d^2 E / d kappa^2, with delta P = 2 (Cv kappa Co^T + transpose).
// The Madelung callback is affine in P; subtract its zero-density value
// when taking the response, including the electronic embedding response.
struct CCMOrbitalStability {
    bool converged = false;
    double curvature = 0.0;
    Eigen::MatrixXd occupied, virtuals, direction;
};

using CCMFockExtra = std::function<std::pair<Eigen::MatrixXd, double>(
    const Eigen::MatrixXd&)>;

inline CCMOrbitalStability ccm_orbital_stability(
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
    const std::vector<std::array<int, 2>>& blocks, const std::vector<int>& Z,
    int nocc, const MsindoParameterSet& p, const Eigen::MatrixXd& P,
    const CCMFockExtra& extra) {
    CCMOrbitalStability out;
    const int n = H.rows(), nv = n - nocc, dim = nv * nocc;
    if (dim == 0) { out.converged = true; return out; }
    std::vector<int> uat(n);
    for (std::size_t a = 0; a < blocks.size(); ++a)
        for (int i = blocks[a][0]; i < blocks[a][1]; ++i) uat[i] = a;
    Eigen::MatrixXd F = build_fock(H, G, P, uat, blocks, Z, p);
    Eigen::MatrixXd extra_zero;
    if (extra) {
        F += extra(P).first;
        extra_zero = extra(Eigen::MatrixXd::Zero(n, n)).first;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F);
    if (es.info() != Eigen::Success) return out;
    out.occupied = es.eigenvectors().leftCols(nocc);
    out.virtuals = es.eigenvectors().rightCols(nv);
    Eigen::MatrixXd gaps(nv, nocc);
    for (int i = 0; i < nocc; ++i)
        gaps.col(i) = es.eigenvalues().tail(nv).array() - es.eigenvalues()(i);
    const auto action = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Map<const Eigen::MatrixXd> k(x.data(), nv, nocc);
        Eigen::MatrixXd dP = 2.0 * out.virtuals * k * out.occupied.transpose();
        dP = (dP + dP.transpose()).eval();
        Eigen::MatrixXd dF = build_fock(H, G, dP, uat, blocks, Z, p) - H;
        if (extra) dF += extra(dP).first - extra_zero;
        Eigen::MatrixXd y = out.virtuals.transpose() * dF * out.occupied;
        y.array() += gaps.array() * k.array();
        return Eigen::Map<const Eigen::VectorXd>(y.data(), dim);
    };
    DavidsonOptions opts;
    opts.n_eig = 1;
    opts.n_guess = std::min(8, dim);
    opts.max_subspace = std::min(32, dim);
    opts.max_iter = 150;
    opts.conv_tol = 1e-8;
    // Dense deterministic seeds reach symmetry sectors that diagonal unit
    // seeds alone can miss. No random state or atom-coordinate rounding.
    opts.guess_vectors.resize(dim, opts.n_guess);
    for (int j = 0; j < opts.n_guess; ++j)
        for (int i = 0; i < dim; ++i)
            opts.guess_vectors(i, j) = std::sin((i + 1.0) * (j + 1.0));
    const auto eig = davidson_solve_matvec(
        dim, action, Eigen::Map<const Eigen::VectorXd>(gaps.data(), dim), opts);
    out.converged = eig.converged;
    if (eig.eigenvalues.size()) {
        out.curvature = eig.eigenvalues(0);
        out.direction = Eigen::Map<const Eigen::MatrixXd>(
            eig.eigenvectors.col(0).data(), nv, nocc);
    }
    return out;
}

inline Eigen::MatrixXd ccm_stability_rotated_density(
    const CCMOrbitalStability& stab, double theta) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        stab.direction, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd angles = theta * svd.singularValues();
    Eigen::MatrixXd occ = stab.occupied
        + stab.occupied * svd.matrixV()
          * (angles.array().cos() - 1.0).matrix().asDiagonal()
          * svd.matrixV().transpose()
        + stab.virtuals * svd.matrixU() * angles.array().sin().matrix().asDiagonal()
          * svd.matrixV().transpose();
    return 2.0 * occ * occ.transpose();
}

inline MsindoResult scf_rhf_ccm_driver(
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
    const std::vector<std::array<int, 2>>& blocks, const std::vector<int>& Z,
    int nocc, const MsindoParameterSet& p, int max_iter, double conv_tol,
    const CCMFockExtra& extra) {
    MsindoResult result = scf_rhf_driver(
        H, G, blocks, Z, nocc, p, max_iter, conv_tol, extra);
    if (!result.converged || nocc == 0 || nocc == H.rows()) return result;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> guess(H);
    // Same resolvable-gap scale as the semiempirical occupation guard.
    // A gapped final spectrum does not repair a roundoff-selected Hcore
    // starting projector (#249). Audit this ambiguous-guess domain before
    // returning energy OR deriving forces from its density.
    if (guess.eigenvalues()(nocc) - guess.eigenvalues()(nocc - 1) > 1e-6)
        return result;
    std::vector<int> uat(H.rows());
    for (std::size_t a = 0; a < blocks.size(); ++a)
        for (int i = blocks[a][0]; i < blocks[a][1]; ++i) uat[i] = a;
    const auto energy = [&](const Eigen::MatrixXd& P) {
        Eigen::MatrixXd F = build_fock(H, G, P, uat, blocks, Z, p);
        double e_add = 0.0;
        if (extra) { auto fe = extra(P); F += fe.first; e_add = fe.second; }
        return 0.5 * (P.array() * (H + F).array()).sum() + e_add;
    };
    int used = result.n_iter;
    for (int restart = 0; ; ++restart) {
        const auto stab = ccm_orbital_stability(
            H, G, blocks, Z, nocc, p, result.density, extra);
        result.stability_checked = true;
        result.stability_analysis_converged = stab.converged;
        result.stability_eigenvalue = 4.0 * stab.curvature;
        result.n_stability_restarts = restart;
        result.n_iter = used;
        if (!stab.converged)
            throw std::runtime_error("MSINDO CCM orbital stability analysis did not converge");
        if (stab.curvature >= -1e-6) return result;
        if (restart >= 5 || used >= max_iter)
            throw std::runtime_error("MSINDO CCM found an unstable SCF saddle; stability restart budget exhausted");
        MsindoResult best = result;
        bool descended = false;
        // Search both signs; an eigenvector's sign is arbitrary. Re-converge
        // each lower-energy seed before comparing stationary energies.
        for (int sign_index = 0; sign_index < 2; ++sign_index) {
            const double sign = sign_index == 0 ? -1.0 : 1.0;
            Eigen::MatrixXd seed;
            double seed_energy = result.electronic_energy;
            for (double angle : {0.1, 0.25, 0.5, 1.0, 2.0}) {
                Eigen::MatrixXd P = ccm_stability_rotated_density(stab, sign * angle);
                const double e = energy(P);
                if (e < seed_energy) { seed_energy = e; seed = std::move(P); }
            }
            const int budget = (max_iter - used) / (2 - sign_index);
            if (!seed.size() || budget <= 0) continue;
            MsindoResult candidate = scf_rhf_driver(
                H, G, blocks, Z, nocc, p, budget, conv_tol, extra, &seed);
            used += candidate.n_iter;
            if (candidate.converged && candidate.electronic_energy
                    < best.electronic_energy - std::max(1e-10, conv_tol)) {
                best = std::move(candidate);
                descended = true;
            }
        }
        if (!descended)
            throw std::runtime_error("MSINDO CCM found an unstable SCF saddle; no lower stationary solution converged within max_iter");
        result = std::move(best);
    }
}

}  // namespace vibeqc::semiempirical::indo
