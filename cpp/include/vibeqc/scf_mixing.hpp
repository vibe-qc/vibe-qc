// Shared SCF mixing helpers.
//
// These are intentionally tiny and header-only: RHF/RKS/UHF/UKS and the
// periodic drivers should use the same validation and Fock-mixing semantics
// instead of each method inventing a private convention.

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vibeqc {

inline void validate_fraction_01(const char* name, double value) {
    if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and in [0, 1), got "
            + std::to_string(value));
    }
}

inline Eigen::MatrixXd mix_fock_matrices(const Eigen::MatrixXd& current,
                                         const Eigen::MatrixXd& previous,
                                         double previous_weight) {
    if (previous_weight == 0.0) {
        return current;
    }
    Eigen::MatrixXd mixed =
        (1.0 - previous_weight) * current + previous_weight * previous;
    return 0.5 * (mixed + mixed.transpose());
}

}  // namespace vibeqc
