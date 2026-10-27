#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace vibeqc::semiempirical::indo {

// Variational rank-one occupied/frontier-hole initial projector. The Fock
// is the derivative of the quadratic INDO density functional. Minimize in
// the unresolved Hcore subspace with a modified Newton step and exact
// spherical Hessian. Negative-curvature steps escape initial saddles. This selects a starting
// projector, not a global SCF minimum. Lehtola et al., Molecules 25, 1218 (2020),
// doi:10.3390/molecules25051218. Twin of msindo_ccm_frontier.py. Every Fock evaluation,
// including seed curvature and line searches, consumes the caller's cumulative budget. Blocks
// other than one occupied orbital or one hole, or larger than eight, retain the established
// Hcore path; explicit starting densities bypass it.
struct CCMFrontierSeed {
    Eigen::MatrixXd density, fock;
    double energy = 0.;
    int evaluations = 0;
    int orbital_begin = 0, orbital_end = 0, expansions = 0;
    bool applicable = false, completed = false;
};
inline CCMFrontierSeed ccm_frontier_seed_in_block(
    const Eigen::MatrixXd &H, int nocc, int budget,
    const std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd &)> &evaluate,
    int override_begin = -1, int override_end = -1,
    const Eigen::MatrixXd *initial_density = nullptr) {
    CCMFrontierSeed out;
    const int n = H.rows();
    if (nocc == 0 || nocc == n)
        return out;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    const Eigen::VectorXd eps = es.eigenvalues();
    const Eigen::MatrixXd C = es.eigenvectors();
    if (override_begin < 0 && eps(nocc) - eps(nocc - 1) > 1e-6)
        return out;
    const double center = .5 * (eps(nocc) + eps(nocc - 1));
    int lo = nocc - 1, hi = nocc;
    while (lo > 0 && std::abs(eps(lo - 1) - center) < 1e-6)
        --lo;
    while (hi < n && std::abs(eps(hi) - center) < 1e-6)
        ++hi;
    if (override_begin >= 0) {
        lo = override_begin;
        hi = override_end;
    }
    out.orbital_begin = lo;
    out.orbital_end = hi;
    const int g = hi - lo, q = nocc - lo;
    if (g > 8 || (q != 1 && q != g - 1))
        return out;
    if (budget < 1)
        throw std::invalid_argument("max_iter must be positive");
    out.applicable = true;
    const bool hole = q != 1;
    const double sign = hole ? -1. : 1.;
    const int fixed = hole ? hi : lo;
    const Eigen::MatrixXd base = 2 * C.leftCols(fixed) * C.leftCols(fixed).transpose();
    const Eigen::MatrixXd B = C.middleCols(lo, g);
    Eigen::VectorXd ref = Eigen::VectorXd::Zero(g);
    const int ref_axis = hole ? g - 1 : 0;
    ref(ref_axis) = 1.;
    if (initial_density) {
        Eigen::MatrixXd D = sign * .5 * B.transpose() * (*initial_density - base) * B;
        D = (.5 * (D + D.transpose())).eval();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> projector(D);
        ref = projector.eigenvectors().col(g - 1);
    }
    const auto density = [&](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
        Eigen::VectorXd u = B * x.normalized();
        return base + sign * 2 * u * u.transpose();
    };
    out.density = density(ref);
    auto initial = evaluate(out.density);
    ++out.evaluations;
    out.fock = initial.first;
    out.energy = initial.second;
    const Eigen::MatrixXd Fref = B.transpose() * out.fock * B;
    const Eigen::MatrixXd A = Fref - Fref.trace() / g * Eigen::MatrixXd::Identity(g, g);
    struct Value {
        double energy;
        Eigen::VectorXd gradient;
        Eigen::MatrixXd density, fock;
        double absolute_energy;
    };
    struct Exhausted {};
    const auto value = [&](const Eigen::VectorXd &x) -> Value {
        if (out.evaluations >= budget)
            throw Exhausted{};
        Eigen::VectorXd v = x.normalized();
        Eigen::MatrixXd P = density(x);
        auto fe = evaluate(P);
        ++out.evaluations;
        const Eigen::MatrixXd response = B.transpose() * fe.first * B - Fref;
        const Eigen::MatrixXd delta = sign * 2 * (v * v.transpose() - ref * ref.transpose());
        const double energy = (delta.array() * (A + .5 * response).array()).sum();
        Eigen::VectorXd grad = sign * 4 * (A + response) * v;
        grad = (grad - v * v.dot(grad)).eval() / x.norm();
        out.density = P;
        out.fock = fe.first;
        out.energy = fe.second;
        return {energy, grad, P, fe.first, fe.second};
    };
    Eigen::VectorXd refgrad = sign * 4 * A * ref;
    refgrad = (refgrad - ref * ref.dot(refgrad)).eval();
    const Value initial_value{0., refgrad, out.density, out.fock, out.energy};
    Value best = initial_value;
    bool found = false;
    try {
        for (int start = 0; start < (initial_density ? 1 : g + 2); ++start) {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(g);
            if (start == 0)
                x = ref;
            else if (start < g)
                x((start + ref_axis) % g) = 1.;
            else
                for (int j = 0; j < g; ++j)
                    x(j) = start == g ? 1. : (j % 2 ? -1. : 1.);
            x.normalize();
            Value current = start == 0 ? initial_value : value(x);
            bool stationary_minimum = false;
            for (int iteration = 0; iteration < 100; ++iteration) {
                const Eigen::MatrixXd tangent =
                    Eigen::MatrixXd::Identity(g, g) - x * x.transpose();
                const Eigen::VectorXd u = B * x;
                Eigen::MatrixXd Fsub = B.transpose() * current.fock * B;
                Fsub -= Fsub.trace() / g * Eigen::MatrixXd::Identity(g, g);
                const double lambda = sign * 4 * x.dot(Fsub * x);
                Eigen::MatrixXd hessian(g, g);
                for (int j = 0; j < g; ++j) {
                    if (out.evaluations >= budget)
                        throw Exhausted{};
                    Eigen::VectorXd t = tangent.col(j), du = B * t;
                    Eigen::MatrixXd dP = sign * 2 * (du * u.transpose() + u * du.transpose());
                    auto response = evaluate(current.density + dP);
                    ++out.evaluations;
                    Eigen::MatrixXd dF = response.first - current.fock;
                    hessian.col(j) =
                        tangent * (sign * 4 * (Fsub * t + B.transpose() * dF * u) - lambda * t);
                }
                hessian = (.5 * (hessian + hessian.transpose())).eval();
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> curvature(hessian);
                if (current.gradient.norm() < 1e-9 && curvature.eigenvalues()(0) >= -1e-8) {
                    stationary_minimum = true;
                    break;
                }
                // Keep physically resolved soft modes; a relative 1e-4 floor stalls
                // the bent-C3 seed even with a gradient above the unchanged threshold.
                const double floor =
                    std::max(1e-8, 64 * std::numeric_limits<double>::epsilon() *
                                       curvature.eigenvalues().cwiseAbs().maxCoeff());
                Eigen::VectorXd inverse_values =
                    curvature.eigenvalues().cwiseAbs().cwiseMax(floor).cwiseInverse();
                const Eigen::MatrixXd inverse = curvature.eigenvectors() *
                                                inverse_values.asDiagonal() *
                                                curvature.eigenvectors().transpose();
                Eigen::VectorXd direction = -tangent * inverse * current.gradient;
                if (direction.norm() > 1.)
                    direction /= direction.norm();
                if (curvature.eigenvalues()(0) < -1e-8) {
                    Eigen::VectorXd unstable = tangent * curvature.eigenvectors().col(0);
                    unstable.normalize();
                    if (current.gradient.dot(unstable) > 0)
                        unstable = -unstable;
                    const double predicted = current.gradient.dot(direction) +
                                             .5 * direction.dot(hessian * direction);
                    const double unstable_predicted =
                        current.gradient.dot(unstable) + .5 * curvature.eigenvalues()(0);
                    if (unstable_predicted < predicted)
                        direction = unstable;
                }
                direction = (direction - x * x.dot(direction)).eval();
                const double slope = current.gradient.dot(direction);
                double alpha = 1.;
                bool accepted = false;
                Eigen::VectorXd trial;
                Value next = current;
                for (int line = 0; line < 24; ++line) {
                    trial = (x + alpha * direction).normalized();
                    next = value(trial);
                    // Exact normalized-projector difference, avoiding cancellation
                    // between outer products when the step becomes small.
                    const Eigen::VectorXd move = alpha * direction;
                    const double move2 = move.squaredNorm();
                    const Eigen::MatrixXd delta =
                        sign * 2 *
                        (x * move.transpose() + move * x.transpose() + move * move.transpose() -
                         move2 * x * x.transpose()) /
                        (1 + move2);
                    Eigen::MatrixXd response = B.transpose() * (next.fock - current.fock) * B;
                    // Compare against the current iterate, not two nearly equal
                    // reference-relative energies. The quadratic difference is exact.
                    const double delta_energy =
                        (delta.array() *
                         (Fsub - x.dot(Fsub * x) * Eigen::MatrixXd::Identity(g, g) +
                          .5 * response)
                             .array())
                            .sum();
                    if (delta_energy <= 1e-4 * alpha * slope) {
                        accepted = true;
                        break;
                    }
                    alpha *= .5;
                }
                if (!accepted)
                    break;
                x = trial;
                current = next;
            }
            if (stationary_minimum && (!found || current.energy < best.energy - 1e-12)) {
                best = current;
                found = true;
            }
        }
    } catch (const Exhausted &) {
        return out;
    }
    if (!found)
        throw std::runtime_error("MSINDO CCM initial-frontier optimization did not converge");
    out.density = best.density;
    out.fock = best.fock;
    out.energy = best.absolute_energy;
    out.completed = true;
    return out;
}

// Screen adjacent occupied/virtual blocks for rank-one negative curvature.
// Expand only whole blocks within the eight-orbital bound, then continue
// from the optimized projector. Screening consumes the cumulative budget.
// This improves the initial guess, not a certificate of a global SCF minimum.
inline CCMFrontierSeed ccm_frontier_seed(
    const Eigen::MatrixXd &H, int nocc, int budget,
    const std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd &)> &evaluate,
    bool allow_resolved = false) {
    int first = -1, last = -1;
    if (allow_resolved && nocc > 0 && nocc < H.rows()) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(H);
        const Eigen::VectorXd eps = spectrum.eigenvalues();
        if (eps(nocc) - eps(nocc - 1) > 1e-6) {
            first = nocc - 1;
            last = nocc + 1;
            while (first > 0 && std::abs(eps(first - 1) - eps(nocc - 1)) < 1e-6)
                --first;
            while (last < H.rows() && std::abs(eps(last) - eps(nocc)) < 1e-6)
                ++last;
        }
    }
    auto seed = ccm_frontier_seed_in_block(H, nocc, budget, evaluate, first, last);
    if (!seed.applicable || !seed.completed)
        return seed;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    const Eigen::VectorXd eps = es.eigenvalues();
    const Eigen::MatrixXd C = es.eigenvectors();
    int used = seed.evaluations, expansions = 0;
    while (true) {
        const int lo = seed.orbital_begin, hi = seed.orbital_end, g = hi - lo, q = nocc - lo;
        double most_negative = -1e-8;
        int selected_begin = -1, selected_end = -1;
        for (bool hole : {false, true}) {
            if ((!hole && q != 1) || (hole && q != g - 1))
                continue;
            int begin, end, new_begin = lo, new_end = hi;
            if (hole) {
                if (lo == 0)
                    continue;
                begin = lo - 1;
                end = lo;
                while (begin > 0 && std::abs(eps(begin - 1) - eps(lo - 1)) < 1e-6)
                    --begin;
                new_begin = begin;
            } else {
                if (hi == H.rows())
                    continue;
                begin = hi;
                end = hi + 1;
                while (end < H.rows() && std::abs(eps(end) - eps(hi)) < 1e-6)
                    ++end;
                new_end = end;
            }
            if (new_end - new_begin > 8)
                continue;
            const double sign = hole ? -1. : 1.;
            const int fixed = hole ? hi : lo;
            const Eigen::MatrixXd base = 2 * C.leftCols(fixed) * C.leftCols(fixed).transpose();
            const Eigen::MatrixXd B = C.middleCols(lo, g);
            Eigen::MatrixXd D = sign * .5 * B.transpose() * (seed.density - base) * B;
            D = (.5 * (D + D.transpose())).eval();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> projector(D);
            const Eigen::VectorXd u = B * projector.eigenvectors().col(g - 1);
            const Eigen::MatrixXd Y = C.middleCols(begin, end - begin);
            const int k = Y.cols();
            const double multiplier = 4 * sign * u.dot(seed.fock * u);
            Eigen::MatrixXd hessian = 4 * sign * Y.transpose() * seed.fock * Y -
                                      multiplier * Eigen::MatrixXd::Identity(k, k);
            for (int j = 0; j < k; ++j) {
                if (used >= budget) {
                    seed.completed = false;
                    seed.evaluations = used;
                    return seed;
                }
                Eigen::VectorXd y = Y.col(j);
                Eigen::MatrixXd delta = 2 * sign * (y * u.transpose() + u * y.transpose());
                auto response = evaluate(seed.density + delta);
                ++used;
                hessian.col(j) += 4 * sign * Y.transpose() * (response.first - seed.fock) * u;
            }
            hessian = (.5 * (hessian + hessian.transpose())).eval();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> curvature(hessian);
            if (curvature.eigenvalues()(0) < most_negative) {
                most_negative = curvature.eigenvalues()(0);
                selected_begin = new_begin;
                selected_end = new_end;
            }
        }
        seed.evaluations = used;
        if (selected_begin < 0)
            return seed;
        if (used >= budget) {
            seed.completed = false;
            return seed;
        }
        seed = ccm_frontier_seed_in_block(H, nocc, budget - used, evaluate, selected_begin,
                                          selected_end, &seed.density);
        used += seed.evaluations;
        seed.evaluations = used;
        seed.expansions = ++expansions;
        if (!seed.completed)
            return seed;
    }
}

} // namespace vibeqc::semiempirical::indo
