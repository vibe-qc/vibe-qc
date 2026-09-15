#include "vibeqc/brent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vibeqc {

namespace {

// Golden-section constant: (3 - sqrt(5)) / 2
constexpr double kGold = 0.3819660112501051;

// Machine epsilon safety margin.
constexpr double kEps = 1e-12;

}  // namespace

BrentResult brent_minimize(
    const std::function<double(double)>& f,
    double a,
    double b,
    double c,
    double tol,
    int    max_iter)
{
    BrentResult result;

    // ---- validate bracket --------------------------------------------------
    if (!(a < b && b < c)) {
        // Sort a, b, c so a < b < c.
        double xs[3] = {a, b, c};
        std::sort(xs, xs + 3);
        a = xs[0]; b = xs[1]; c = xs[2];
    }

    double fa = f(a);
    double fb = f(b);
    double fc = f(c);
    int n_eval = 3;

    // Ensure b is the lowest point.
    if (fb > fa || fb > fc) {
        // The bracket is invalid — return the best of the three.
        if (fa <= fb && fa <= fc) {
            result.x_min = a; result.f_min = fa;
        } else if (fc <= fa && fc <= fb) {
            result.x_min = c; result.f_min = fc;
        } else {
            result.x_min = b; result.f_min = fb;
        }
        result.n_eval = n_eval;
        return result;
    }

    // Degenerate bracket — all function values identical.
    if (std::abs(fb - fa) < 1e-300 && std::abs(fb - fc) < 1e-300) {
        result.x_min = b;
        result.f_min = fb;
        result.n_eval = n_eval;
        return result;
    }

    // ---- initialise --------------------------------------------------------
    // x  — current best point (lowest f)
    // w  — second-best point
    // v  — previous value of w
    double x = b,  fx = fb;
    double w = b,  fw = fb;
    double v = b,  fv = fb;
    double e = 0.0;   // step size from previous iteration
    double d = 0.0;   // pending step

    for (int iter = 1; iter <= max_iter; ++iter) {
        double xm = 0.5 * (a + c);
        double tol1 = tol * std::abs(x) + kEps;
        double tol2 = 2.0 * tol1;

        // Convergence check.
        if (std::abs(x - xm) <= tol2 - 0.5 * (c - a)) {
            result.x_min = x;
            result.f_min = fx;
            result.n_eval = n_eval;
            return result;
        }

        // ---- try parabolic interpolation -----------------------------------
        if (std::abs(e) > tol1) {
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::abs(q);
            double etemp = e;
            e = d;

            if (std::abs(p) >= std::abs(0.5 * q * etemp) ||
                p <= q * (a - x) ||
                p >= q * (c - x)) {
                // Reject parabolic — fall through to golden-section.
                if (x >= xm)
                    e = a - x;
                else
                    e = c - x;
                d = kGold * e;
            } else {
                // Accept parabolic step.
                d = p / q;
                double u = x + d;
                if (u - a < tol2 || c - u < tol2)
                    d = (xm >= x ? tol1 : -tol1);
            }
        } else {
            // Golden-section step.
            if (x >= xm)
                e = a - x;
            else
                e = c - x;
            d = kGold * e;
        }

        // ---- take the step -------------------------------------------------
        double u;
        if (std::abs(d) >= tol1)
            u = x + d;
        else
            u = x + ((d > 0.0) ? tol1 : -tol1);

        double fu = f(u);
        ++n_eval;

        // ---- update bracket and best points --------------------------------
        if (fu <= fx) {
            if (u >= x)
                a = x;
            else
                c = x;
            v = w;  fv = fw;
            w = x;  fw = fx;
            x = u;  fx = fu;
        } else {
            if (u < x)
                a = u;
            else
                c = u;
            if (fu <= fw || std::abs(w - x) < 1e-15) {
                v = w;  fv = fw;
                w = u;  fw = fu;
            } else if (fu <= fv || std::abs(v - x) < 1e-15 ||
                       std::abs(v - w) < 1e-15) {
                v = u;  fv = fu;
            }
        }
    }

    // Max iterations reached — return best point found.
    result.x_min = x;
    result.f_min = fx;
    result.n_eval = n_eval;
    return result;
}

}  // namespace vibeqc
