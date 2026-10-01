// Standalone regression for the noncanonical UHF/UKS TRAH Hessian (#55).
// Also run by tests/native/CMakeLists.txt in native-build CI.
// Build with the Eigen/libint headers used by the native core:
// c++ -std=c++17 -O1 -Icpp/include $(pkg-config --cflags eigen3 libint2) \
//   tests/test_trah_noncanonical.cpp cpp/src/trah.cpp \
//   cpp/src/quadratic_scf.cpp $(pkg-config --libs libint2) -o /tmp/trah-noncanonical
// /tmp/trah-noncanonical
#include "vibeqc/trah.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/quadratic_scf.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using Matrix = Eigen::MatrixXd;

// A fixed one-electron Hamiltonian isolates the Fock-block contribution
// from the response terms. A two-electron case below tests the J/K coupling.
struct ZeroJK : vibeqc::JKBuilder {
    Matrix build_J(const Matrix& d) const override {
        return Matrix::Zero(d.rows(), d.cols());
    }
    Matrix build_K(const Matrix& d) const override {
        return Matrix::Zero(d.rows(), d.cols());
    }
};

// A positive semidefinite two-electron tensor (mu nu|lambda sigma)
// = B(mu,nu) B(lambda,sigma). Contract it without the production Hessian
// helpers, so finite differences below provide an independent response oracle.
struct TensorJK : vibeqc::JKBuilder {
    Matrix b;
    TensorJK() : b(4, 4) {
        b << .21, .03, .04, -.02,
             .03, .17, -.01, .05,
             .04, -.01, .13, .02,
             -.02, .05, .02, .19;
    }
    Matrix build_J(const Matrix& d) const override {
        return b * (b.cwiseProduct(d)).sum();
    }
    Matrix build_K(const Matrix& d) const override {
        return b * d * b;
    }
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

Matrix density(const Matrix& c, int occupied) {
    return c.leftCols(occupied) * c.leftCols(occupied).transpose();
}

Matrix block_rotation(int occupied) {
    Matrix q = Matrix::Identity(4, 4);
    for (const auto block : {std::pair<int, int>{0, occupied}, {occupied, 4}}) {
        for (int i = block.first; i + 1 < block.second; ++i) {
            Matrix r = Matrix::Identity(4, 4);
            const double t = 0.37 + 0.13 * i;
            r(i, i) = r(i + 1, i + 1) = std::cos(t);
            r(i, i + 1) = -std::sin(t);
            r(i + 1, i) = std::sin(t);
            q = q * r;
        }
    }
    return q;
}

Matrix rotate(const Eigen::VectorXd& x, int offset, int nocc) {
    Matrix k = Matrix::Zero(4, 4);
    for (int i = 0; i < nocc; ++i)
        for (int a = nocc; a < 4; ++a) {
            k(a, i) = x(offset++);
            k(i, a) = -k(a, i);
        }
    return vibeqc::expm_skew(k);
}

// Compare an actual coupled UHF step with a dense Hessian obtained by
// finite differences of physical Focks in rotated orbitals. This detects
// omitted J/K responses even if both orbital gauges omit the same term.
void response_oracle(const vibeqc::TRAHOptions& opts) {
    TensorJK jk;
    const Matrix c = Matrix::Identity(4, 4);
    const Matrix da = density(c, 2), db = density(c, 1);
    Matrix h = (Eigen::Vector4d(-1.2, -.4, .2, 1.4)).asDiagonal();
    h(0, 2) = h(2, 0) = .012;
    h(1, 3) = h(3, 1) = -.021;
    auto gradient = [&](const Eigen::VectorXd& x) {
        const Matrix ca = rotate(x, 0, 2), cb = rotate(x, 4, 1);
        const Matrix a = density(ca, 2), b = density(cb, 1);
        const Matrix j = jk.build_J(a + b);
        const Matrix fa = ca.transpose() * (h + j - jk.build_K(a)) * ca;
        const Matrix fb = cb.transpose() * (h + j - jk.build_K(b)) * cb;
        Eigen::VectorXd g(7);
        int n = 0;
        for (int i = 0; i < 2; ++i)
            for (int v = 2; v < 4; ++v) g(n++) = fa(v, i);
        for (int v = 1; v < 4; ++v) g(n++) = fb(v, 0);
        return g;
    };
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);
    Matrix hess(7, 7);
    for (int i = 0; i < 7; ++i) {
        Eigen::VectorXd dx = zero;
        dx(i) = 1e-5;
        hess.col(i) = (gradient(dx) - gradient(-dx)) / 2e-5;
    }
    require((hess - hess.transpose()).norm() < 1e-8,
            "finite-difference Hessian is not symmetric");
    require(Eigen::SelfAdjointEigenSolver<Matrix>(hess).eigenvalues().minCoeff() > .1,
            "response oracle must have an interior positive Hessian");
    const Eigen::VectorXd k = hess.ldlt().solve(-gradient(zero));
    const Matrix fa = h + jk.build_J(da + db) - jk.build_K(da);
    const Matrix fb = h + jk.build_J(da + db) - jk.build_K(db);
    for (const bool change_gauge : {false, true}) {
        const Matrix qa = change_gauge ? block_rotation(2) : c;
        const Matrix qb = change_gauge ? block_rotation(1) : c;
        const auto step = vibeqc::uhf_trah_step(
            fa, fb, qa, qb, (qa.transpose() * fa * qa).diagonal(),
            (qb.transpose() * fb * qb).diagonal(), 2, 1, jk, opts, .3);
        const double error = std::max(
            (density(step.C_alpha, 2) - density(rotate(k, 0, 2), 2)).norm(),
            (density(step.C_beta, 1) - density(rotate(k, 4, 1), 1)).norm());
        std::cout << "response_oracle gauge=" << change_gauge
                  << " density_error=" << error << " cg_iter=" << step.cg_iter << '\n';
        require(step.cg_converged && !step.on_boundary && error < 1e-8,
                "TRAH disagrees with finite-difference J/K Hessian");
    }
}

// Analytic hard-case reference: alpha has curvature -0.4 with zero
// gradient along that eigenvector. At radius 0.3 the minimum has shift
// lambda=0.4 and fills the remaining norm in either sign of that direction.
// Both signs have the same model value; comparing their densities is invalid.
int hard_case_regression() {
    Matrix fa = (Eigen::Vector4d(-1.2, .6, .2, 1.4)).asDiagonal();
    fa(0, 2) = fa(2, 0) = .012;
    fa(1, 3) = fa(3, 1) = -.021;
    fa(0, 3) = fa(3, 0) = .008;
    Matrix fb = fa;
    fb(0, 1) = fb(1, 0) = .015;
    ZeroJK jk;
    vibeqc::TRAHOptions opts;
    opts.cg_tol = 1e-12; opts.cg_max_iter = 100;
    opts.ms_tol = 1e-10; opts.ms_max_iter = 80;
    constexpr double radius = .3, shift = .4;
    const double alpha_part = .012*.012/1.8 + .008*.008/3.0 + .021*.021/1.2;
    const double alpha_norm2 = std::pow(.012/1.8, 2) + std::pow(.008/3.0, 2)
                              + std::pow(.021/1.2, 2);
    const Eigen::Vector3d gb(.015, .012, .008);
    const Matrix shifted_beta = fb.bottomRightCorner(3, 3)
                                + 1.6 * Matrix::Identity(3, 3);
    const Eigen::Vector3d beta_part = shifted_beta.ldlt().solve(gb);
    bool failed = false;
    for (const int nb : {1, 0}) {
        require(alpha_norm2 + (nb ? beta_part.squaredNorm() : 0.0) < radius*radius,
                "fixture must lie in the hard-case regime");
        const double optimum = .5 * (alpha_part + (nb ? gb.dot(beta_part) : 0.0)
                                      + shift * radius * radius);
        for (const bool gauge : {false, true}) {
            const Matrix qa = gauge ? block_rotation(2) : Matrix::Identity(4, 4);
            const Matrix qb = gauge ? block_rotation(nb) : Matrix::Identity(4, 4);
            const auto step = vibeqc::uhf_trah_step(
                fa, fb, qa, qb, (qa.transpose()*fa*qa).diagonal(),
                (qb.transpose()*fb*qb).diagonal(), 2, nb, jk, opts, radius);
            const bool ok = step.cg_converged && step.on_boundary
                && std::abs(step.kappa_norm - radius) <= 1e-10 * radius
                && std::abs(step.predicted_decrease - optimum) <= 1e-10 * optimum;
            std::cout << "hard_case beta=" << nb << " gauge=" << gauge
                      << " decrease=" << step.predicted_decrease
                      << " optimum=" << optimum << " norm=" << step.kappa_norm
                      << " boundary=" << step.on_boundary << '\n';
            failed = failed || !ok;
        }
    }
    if (failed) {
        std::cerr << "KNOWN_FAILURE_424: hard-case model optimum or boundary missed\n";
        return 86;
    }
    return 0;
}

int main(int argc, char** argv) {
    const bool hard_case = argc == 2 && std::string(argv[1]) == "--hard-case";
    try {
        if (hard_case) return hard_case_regression();
        Matrix fa = (Eigen::Vector4d(-1.2, -0.4, 0.2, 1.4)).asDiagonal();
        fa(0, 2) = fa(2, 0) = 0.012;
        fa(1, 3) = fa(3, 1) = -0.021;
        fa(0, 3) = fa(3, 0) = 0.008;
        Matrix fb = fa;
        fb(1, 1) = -0.1;
        fb(0, 1) = fb(1, 0) = 0.015;
        ZeroJK jk;
        vibeqc::TRAHOptions opts;
        opts.cg_tol = 1e-12;
        opts.cg_max_iter = 100;
        opts.ms_tol = 1e-10;
        opts.ms_max_iter = 80;
        const Matrix identity = Matrix::Identity(4, 4);
        response_oracle(opts);
        for (const bool indefinite : {false, true}) {
            if (indefinite) {
                // Negative occupied-to-virtual curvature with a nonzero gradient.
                fa(1, 1) = .6;
                fa(1, 2) = fa(2, 1) = .017;
                fb = fa;
                fb(0, 1) = fb(1, 0) = .015;
            }
            for (const auto occupation : {std::pair<int, int>{2, 1}, {4, 1}, {2, 0}}) {
                const int na = occupation.first, nb = occupation.second;
                const Matrix qa = block_rotation(na), qb = block_rotation(nb);
                require((density(qa, na) - density(identity, na)).norm() < 1e-14,
                        "occupied-space gauge changed the starting density");
                for (const double radius : {0.3, 0.007}) {
                    const auto reference = vibeqc::uhf_trah_step(
                        fa, fb, identity, identity, fa.diagonal(), fb.diagonal(),
                        na, nb, jk, opts, radius);
                    const auto rotated = vibeqc::uhf_trah_step(
                        fa, fb, qa, qb, (qa.transpose() * fa * qa).diagonal(),
                        (qb.transpose() * fb * qb).diagonal(),
                        na, nb, jk, opts, radius);
                    const double error = std::max(
                        (density(reference.C_alpha, na) - density(rotated.C_alpha, na)).norm(),
                        (density(reference.C_beta, nb) - density(rotated.C_beta, nb)).norm());
                    std::cout << "indefinite=" << indefinite << " occupations=" << na << ',' << nb
                              << " radius=" << radius << " density_error=" << error
                              << " cg_iter=" << rotated.cg_iter << '\n';
                    require(reference.cg_converged && rotated.cg_converged,
                            "inner solve did not converge");
                    require(error < 1e-9, "TRAH step depends on occupied/virtual orbital gauge");
                    require(std::abs(reference.kappa_norm - rotated.kappa_norm) < 1e-9,
                            "TRAH step norm depends on orbital gauge");
                    require(rotated.kappa_norm <= radius * (1.0 + 1e-8),
                            "TRAH step exceeds trust radius");
                    const double energy_before =
                        (fa * density(identity, na)).trace()
                        + (fb * density(identity, nb)).trace();
                    const double energy_after =
                        (fa * density(rotated.C_alpha, na)).trace()
                        + (fb * density(rotated.C_beta, nb)).trace();
                    require(energy_after < energy_before - 1e-8,
                            "TRAH did not lower the one-electron energy");
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
