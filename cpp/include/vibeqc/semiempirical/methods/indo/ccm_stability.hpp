#pragma once

#include "vibeqc/davidson.hpp"
#include "vibeqc/semiempirical/methods/indo/ccm_orbitals.hpp"
#include "vibeqc/semiempirical/methods/indo/ccm_frontier.hpp"
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
    const auto solve = [&](int n_eig, int n_guess, int max_subspace, int max_iter) {
        DavidsonOptions opts;
        opts.n_eig = std::min(n_eig, dim);
        opts.n_guess = std::min(n_guess, dim);
        opts.max_subspace = std::min(max_subspace, dim);
        opts.max_iter = max_iter;
        opts.conv_tol = 1e-8;
        // Dense deterministic seeds reach symmetry sectors that diagonal unit
        // seeds alone can miss. No random state or atom-coordinate rounding.
        opts.guess_vectors.resize(dim, opts.n_guess);
        for (int j = 0; j < opts.n_guess; ++j)
            for (int i = 0; i < dim; ++i)
                opts.guess_vectors(i, j) = std::sin((i + 1.0) * (j + 1.0));
        return davidson_solve_matvec(
            dim, action, Eigen::Map<const Eigen::VectorXd>(gaps.data(), dim), opts);
    };
    auto eig = solve(1, 8, 32, 150);
    if (!eig.converged) {
        // A slightly displaced high-symmetry cluster splits a degenerate
        // lowest curvature into a near-degenerate cluster, where a
        // one-vector solve can stall (MgO 2x2x2 with Madelung). Resolve the
        // cluster as a block; the lowest pair is taken from it. Every case
        // the first solve converges keeps its exact result.
        eig = solve(6, 12, 96, 400);
    }
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

// Rotation angles (radians along the unstable mode) scanned for a restart
// seed, and the energy a mode must be able to release before its saddle
// triggers a flag or restart. A negative mode without enough scanned descent
// is accepted by the numerical policy, with its negative curvature retained;
// this does not prove the orbital Hessian is nonnegative.
// Twin of msindo_ccm_stability.py; keep both in step.
constexpr double kCcmRestartAngles[] = {0.1, 0.25, 0.5, 1.0, 2.0};
constexpr double kCcmStabilityDescentTol = 1e-6;

// SCF plus a restricted stability audit of every converged state (#249).
//
// Two policies. The default, reference policy keeps the reference program's
// Hcore/DIIS trajectory: it repairs a saddle only where that trajectory is
// ill-defined, an unresolved Hcore frontier (gap <= 1e-6 Ha), and otherwise
// returns the converged state with internal_instability set, never silently.
// The frozen MSINDO oracle clusters include such saddles, so this is the
// policy that reproduces the reference program. lowest_stable attempts descent
// from detected saddles within the cumulative budget. It guarantees neither
// convergence nor branch continuity or a global minimum. For example,
// a C3 chain bent by 3e-3 bohr has a resolved gap (1e-5 Ha) yet converges
// from the Hcore guess 16.6 mHa above the continuous lower branch.
//
// max_iter bounds the cumulative iterations including every restart. The
// lower-energy rotation sign is re-converged first with the whole remaining
// budget; the other sign only when the first does not descend.
inline MsindoResult scf_rhf_ccm_driver(const Eigen::MatrixXd &H, const Eigen::MatrixXd &G,
                                       const std::vector<std::array<int, 2>> &blocks,
                                       const std::vector<int> &Z, int nocc,
                                       const MsindoParameterSet &p, int max_iter,
                                       double conv_tol, const CCMFockExtra &extra,
                                       bool lowest_stable = false,
                                       const Eigen::MatrixXd *initial_density = nullptr) {
    if (max_iter < 1)
        throw std::invalid_argument("max_iter must be positive");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> guess(H);
    const bool unresolved_guess =
        nocc > 0 && nocc < H.rows() &&
        guess.eigenvalues()(nocc) - guess.eigenvalues()(nocc - 1) <= 1e-6;
    const bool descend = lowest_stable || unresolved_guess;
    // Variational seeds continue in their own occupied subspace below.
    // Unresolved frontiers outside the rank-one seed scope retain the
    // established 29-DIIS-update fallback before orbital optimization.
    // Resolved reference trajectories and explicit densities retain DIIS.
    // Both phases and all restarts share the public cumulative budget.
    const int optimize_after = unresolved_guess && !initial_density ? 30 : 0;
    std::vector<int> seed_uat(H.rows());
    for (size_t a = 0; a < blocks.size(); ++a)
        for (int i = blocks[a][0]; i < blocks[a][1]; ++i)
            seed_uat[i] = a;
    const auto seed_eval = [&](const Eigen::MatrixXd &P) {
        Eigen::MatrixXd F = build_fock(H, G, P, seed_uat, blocks, Z, p);
        double add = 0;
        if (extra) {
            auto fe = extra(P);
            F += fe.first;
            add = fe.second;
        }
        double e = .5 * (P.array() * (H + F).array()).sum() + add;
        return std::make_pair(F, e);
    };
    const auto seed =
        initial_density ? CCMFrontierSeed{} : ccm_frontier_seed(H, nocc, max_iter, seed_eval, lowest_stable);
    if (seed.applicable && (!seed.completed || seed.evaluations >= max_iter)) {
        MsindoResult failed;
        failed.density = seed.density;
        failed.electronic_energy = seed.energy;
        failed.n_iter = seed.evaluations;
        failed.frontier_seed_evaluations = seed.evaluations;
        failed.frontier_seed_completed = seed.completed;
        failed.frontier_seed_orbitals = seed.orbital_end - seed.orbital_begin;
        failed.frontier_seed_expansions = seed.expansions;
        failed.mo_energies =
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(seed.fock).eigenvalues();
        return failed;
    }
    MsindoResult result = seed.applicable
        ? ccm_minimize_orbitals(seed.density, seed.fock, seed.energy, nocc,
                               max_iter - seed.evaluations, conv_tol, seed_eval)
        : scf_rhf_driver(H, G, blocks, Z, nocc, p, max_iter, conv_tol, extra,
                         initial_density, optimize_after);
    result.n_iter += seed.evaluations;
    result.frontier_seed_evaluations = seed.evaluations;
    result.frontier_seed_completed = seed.completed;
    result.frontier_seed_orbitals = seed.applicable ? seed.orbital_end - seed.orbital_begin : 0;
    result.frontier_seed_expansions = seed.expansions;
    const bool orbital_optimization_used = result.orbital_optimization_used;
    if (!result.converged || nocc == 0 || nocc == H.rows())
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
    struct Seed { double energy; double sign; Eigen::MatrixXd density; };
    int used = result.n_iter;
    for (int restart = 0; ; ++restart) {
        const auto stab = ccm_orbital_stability(
            H, G, blocks, Z, nocc, p, result.density, extra);
        result.stability_checked = true;
        result.stability_analysis_converged = stab.converged;
        result.stability_eigenvalue = 4.0 * stab.curvature;
        result.n_stability_restarts = restart;
        result.n_iter = used;
        result.frontier_seed_evaluations = seed.evaluations;
        result.frontier_seed_completed = seed.completed;
        result.frontier_seed_orbitals = seed.applicable ? seed.orbital_end - seed.orbital_begin : 0;
        result.frontier_seed_expansions = seed.expansions;
        result.orbital_optimization_used = orbital_optimization_used;
        if (!stab.converged) {
            // Under the reference policy at a resolved gap the audit only
            // reports; an unverified verdict is recorded, not an error. The
            // driver cannot act on an unconverged mode, so it raises there.
            if (!descend) return result;
            throw std::runtime_error("MSINDO CCM orbital stability analysis did not converge");
        }
        if (stab.curvature >= -1e-6) return result;
        // One seed per sign of the mode: the lowest-energy scanned rotation.
        std::vector<Seed> seeds;
        for (double sign : {-1.0, 1.0}) {
            Eigen::MatrixXd seed;
            double seed_energy = result.electronic_energy;
            for (double angle : kCcmRestartAngles) {
                Eigen::MatrixXd P = ccm_stability_rotated_density(stab, sign * angle);
                const double e = energy(P);
                if (e < seed_energy) { seed_energy = e; seed = std::move(P); }
            }
            if (seed.size()) seeds.push_back({seed_energy, sign, std::move(seed)});
        }
        double lowest = result.electronic_energy;
        for (const auto& s : seeds) lowest = std::min(lowest, s.energy);
        if (seeds.empty() || result.electronic_energy - lowest < kCcmStabilityDescentTol)
            return result;
        if (!descend) {
            result.internal_instability = true;
            return result;
        }
        if (restart >= 5 || used >= max_iter)
            throw std::runtime_error("MSINDO CCM found an unstable SCF saddle; stability restart budget exhausted");
        // Lower seed first; a tie within 1e-10 keeps the negative sign first.
        if (seeds.size() == 2 && seeds[1].energy < seeds[0].energy - 1e-10)
            std::swap(seeds[0], seeds[1]);
        MsindoResult best = result;
        bool descended = false;
        for (auto& s : seeds) {
            const int budget = max_iter - used;
            if (budget <= 0) break;
            MsindoResult candidate = scf_rhf_driver(
                H, G, blocks, Z, nocc, p, budget, conv_tol, extra, &s.density);
            used += candidate.n_iter;
            if (candidate.converged && candidate.electronic_energy
                    < best.electronic_energy - std::max(1e-10, conv_tol)) {
                best = std::move(candidate);
                descended = true;
                break;
            }
        }
        if (!descended)
            throw std::runtime_error("MSINDO CCM found an unstable SCF saddle; no lower stationary solution converged within max_iter");
        result = std::move(best);
    }
}

}  // namespace vibeqc::semiempirical::indo
