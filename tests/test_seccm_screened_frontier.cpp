// Standalone regression for the retained-spectrum contract (#151).
// Build from the repository root, using the same Eigen/libint headers as
// the tested core:
// c++ -std=c++17 -O1 -Icpp/include -Icpp/src \
//   $(pkg-config --cflags eigen3 libint2) \
//   tests/test_seccm_screened_frontier.cpp $(pkg-config --libs libint2) \
//   -o /tmp/seccm-screened-frontier
// /tmp/seccm-screened-frontier
#include "semiempirical/seccm/seccm_common.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace detail = vibeqc::semiempirical::seccm::detail;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        // Rotate a known indefinite metric so correctness also requires
        // transforming coefficients back to the original AO coordinates.
        Eigen::Matrix4d rotation = Eigen::Matrix4d::Identity();
        rotation(0, 0) = rotation(2, 2) = 0.6;
        rotation(0, 2) = 0.8;
        rotation(2, 0) = -0.8;
        const Eigen::Vector4d metric(-1.0, -0.5, 0.7, 1.1);
        const Eigen::Vector4d diagonal_h(0.8, 0.2, 1.4, 3.3);
        const Eigen::MatrixXd S = rotation * metric.asDiagonal()
            * rotation.transpose();
        const Eigen::MatrixXd H = rotation * diagonal_h.asDiagonal()
            * rotation.transpose();

        bool occupied_lost = false;
        try {
            detail::canonical_orthogonalizer(S, 1e-10, 3, "test");
        } catch (const std::runtime_error&) {
            occupied_lost = true;
        }
        require(occupied_lost, "insufficient occupied space was accepted");

        const Eigen::MatrixXd X =
            detail::canonical_orthogonalizer(S, 1e-10, 2, "test");
        require(X.rows() == 4 && X.cols() == 2,
                "exact occupied-space equality must remain valid");
        const auto solution = detail::solve_canonical_orthogonalized(H, X, "test");
        require((solution.coefficients.transpose() * S * solution.coefficients
                 - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff() < 1e-13,
                "screened eigenvectors are not metric orthonormal");
        require((solution.eigenvalues - Eigen::Vector2d(2.0, 3.0)).norm() < 1e-13,
                "canonical eigenvalues differ from the known spectrum");
        require(std::isnan(detail::finite_torus_homo_lumo_gap(
                    solution.eigenvalues, 2)),
                "fully occupied retained space must have unavailable gap");
        require(std::abs(detail::finite_torus_homo_lumo_gap(
                    solution.eigenvalues, 1) - 1.0) < 1e-13,
                "retained virtual orbital must supply the actual gap");

        // Guard every boundary before indexing, including invalid callers.
        for (int occupied : {-1, 0, 2, 3}) {
            require(std::isnan(detail::finite_torus_homo_lumo_gap(
                        solution.eigenvalues, occupied)),
                    "unavailable frontier was read or reported as zero");
        }
        require(std::isnan(detail::finite_torus_homo_lumo_gap(
                    Eigen::VectorXd(), 1)), "empty spectrum was indexed");

        const Eigen::MatrixXd positive_metric = Eigen::Matrix4d::Identity();
        require(detail::canonical_orthogonalizer(
                    positive_metric, 1e-10, 2, "test").size() == 0,
                "ordinary SPD solve must retain its existing path");
        const Eigen::Vector4d full_spectrum(-2.0, -1.0, 0.5, 2.0);
        require(detail::finite_torus_homo_lumo_gap(full_spectrum, 2) == 1.5,
                "ordinary SPD frontier changed");
        Eigen::Vector2d degenerate(-1.0, -1.0);
        require(detail::finite_torus_homo_lumo_gap(degenerate, 1) == 0.0,
                "real degeneracy must remain distinct from absent frontier");
        std::cout << "PASS: occupied-space loss/equality/virtual space, metric "
                     "orthonormality, unavailable frontiers, SPD and degeneracy\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
