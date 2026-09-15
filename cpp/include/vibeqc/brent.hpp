// Brent's 1-D minimisation without derivatives.
//
// Classic Brent (1973) algorithm combining golden-section search with
// inverse parabolic interpolation.  Guaranteed to converge for any
// unimodal function within the initial bracket.
//
// Reference
// ---------
// Brent, R. P. *Algorithms for Minimization without Derivatives*,
//   Prentice-Hall (1973), Chapter 5.

#pragma once

#include <cstdint>
#include <functional>
#include <tuple>

namespace vibeqc {

/// Result of a 1-D Brent minimisation.
struct BrentResult {
    double x_min = 0.0;   ///< location of the minimum
    double f_min = 0.0;   ///< function value at the minimum
    int    n_eval = 0;    ///< number of function evaluations (≥3)
};

/// Minimise a scalar function f(x) within the bracketing triplet
/// a < b < c where f(b) <= f(a) and f(b) <= f(c).
///
/// \param f        Scalar function to minimise.
/// \param a, b, c  Bracketing triplet.  Must satisfy a < b < c and
///                 f(b) <= f(a), f(b) <= f(c).
/// \param tol      Absolute tolerance on x.  Converges when the
///                 bracket width is ~ 2*tol.
/// \param max_iter Maximum number of function evaluations.
///
/// \returns BrentResult with the minimum location, value, and the
///          number of evaluations (including the three bracket points).
BrentResult brent_minimize(
    const std::function<double(double)>& f,
    double a,
    double b,
    double c,
    double tol = 1e-5,
    int    max_iter = 100
);

}  // namespace vibeqc
