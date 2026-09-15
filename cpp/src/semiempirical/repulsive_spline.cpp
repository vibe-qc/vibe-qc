#include "vibeqc/semiempirical/repulsive_spline.hpp"

#include <algorithm>
#include <Eigen/Dense>

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Natural cubic spline builder
// ---------------------------------------------------------------------------

void RepulsiveSpline::build(const std::vector<double>& R,
                             const std::vector<double>& V) {
    if (R.size() < 2) {
        throw std::invalid_argument("RepulsiveSpline: need at least 2 knots");
    }
    if (R.size() != V.size()) {
        throw std::invalid_argument("RepulsiveSpline: R and V must have same length");
    }
    for (std::size_t i = 1; i < R.size(); ++i) {
        if (R[i] <= R[i-1]) {
            throw std::invalid_argument("RepulsiveSpline: R must be strictly increasing");
        }
    }

    R_ = R;
    V_ = V;
    const int n = static_cast<int>(R.size()) - 1;  // number of intervals

    a_.resize(n);
    b_.resize(n);
    c_.resize(n);
    d_.resize(n);

    // a_i = V_i (value at left knot)
    for (int i = 0; i < n; ++i) {
        a_[i] = V[i];
    }

    // Tridiagonal system for c_i (second derivatives / 2)
    // Natural spline: c_0 = c_n = 0
    Eigen::VectorXd h(n);
    for (int i = 0; i < n; ++i) {
        h(i) = R[i+1] - R[i];
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n-1, n-1);
    Eigen::VectorXd rhs(n-1);

    for (int i = 0; i < n-1; ++i) {
        if (i > 0) A(i, i-1) = h(i) / 6.0;
        A(i, i) = (h(i) + h(i+1)) / 3.0;
        if (i < n-2) A(i, i+1) = h(i+1) / 6.0;

        rhs(i) = (V[i+2] - V[i+1]) / h(i+1) - (V[i+1] - V[i]) / h(i);
    }

    Eigen::VectorXd c_sol = A.colPivHouseholderQr().solve(rhs);

    // c_i for intervals 1..n-1; c_0 = c_n = 0
    std::vector<double> c_full(n + 1, 0.0);
    for (int i = 0; i < n-1; ++i) {
        c_full[i+1] = c_sol(i);
    }

    for (int i = 0; i < n; ++i) {
        c_[i] = c_full[i] / 2.0;  // our c = y''/2
        d_[i] = (c_full[i+1] - c_full[i]) / (6.0 * h(i));
        b_[i] = (V[i+1] - V[i]) / h(i) - h(i) * (2.0 * c_full[i] + c_full[i+1]) / 6.0;
    }
}

double RepulsiveSpline::evaluate(double R) const {
    if (R_ .empty()) return 0.0;
    if (R >= R_.back()) return 0.0;  // beyond cutoff
    if (R <= R_.front()) {
        // below first knot: linear extrapolation using first interval
        double h = R_[1] - R_[0];
        if (h > 0) return V_[0] + (V_[1] - V_[0]) * (R - R_[0]) / h;
        return V_[0];
    }

    // Find interval
    auto it = std::upper_bound(R_.begin(), R_.end(), R);
    int i = static_cast<int>(it - R_.begin()) - 1;
    if (i < 0) i = 0;
    if (i >= static_cast<int>(a_.size())) i = static_cast<int>(a_.size()) - 1;

    double dx = R - R_[i];
    return a_[i] + b_[i] * dx + c_[i] * dx * dx + d_[i] * dx * dx * dx;
}

double RepulsiveSpline::derivative(double R) const {
    if (R_.empty()) return 0.0;
    if (R >= R_.back()) return 0.0;
    if (R <= R_.front()) {
        double h = R_[1] - R_[0];
        if (h > 0) return (V_[1] - V_[0]) / h;
        return 0.0;
    }

    auto it = std::upper_bound(R_.begin(), R_.end(), R);
    int i = static_cast<int>(it - R_.begin()) - 1;
    if (i < 0) i = 0;
    if (i >= static_cast<int>(a_.size())) i = static_cast<int>(a_.size()) - 1;

    double dx = R - R_[i];
    return b_[i] + 2.0 * c_[i] * dx + 3.0 * d_[i] * dx * dx;
}

}  // namespace semiempirical
}  // namespace vibeqc
