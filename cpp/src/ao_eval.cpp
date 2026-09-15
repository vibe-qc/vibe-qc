#include "vibeqc/ao_eval.hpp"

#include <libint2/shell.h>
#include <libint2/solidharmonics.h>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/thread_pool.hpp"

namespace vibeqc {

// --- Cartesian normalization convention -----------------------------------
//
// These evaluators must reproduce, on a grid, exactly the basis functions
// libint's integral engine uses: MO coefficients, density matrices and every
// operator matrix live in that basis, so any mismatch shows up as a wrong
// density rather than as an error.
//
// libint applies ONE normalization per shell, derived from the total l (the
// norm of the axial (l, 0, 0) component), and does not vary it per Cartesian
// component. Mixed components are therefore NOT individually unit-normalized:
// for a single Cartesian d shell, compute_overlap gives <xy|xy> = 1/3 against
// <xx|xx> = 1. (Verified directly against a raw libint2::Engine; see
// tests/test_ao_convention_invariants.py.)
//
// So the monomials below carry no per-component factor. Applying one --
// sqrt((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!)), which reaches true unit norm
// -- is what this file used to do for `pure = false` shells, and it made
// grid-evaluated Cartesian AOs disagree with every integral by that factor
// (3x in the density for d_xy). Pure shells were unaffected: the transform
// below is defined against these same unnormalized monomials, so the old code
// multiplied the factor in and then divided it straight back out.
//
// QVF spec Appendix A.1 defines the identical convention for `basis.json`
// coefficients, so a QVF consumer and this evaluator now agree by
// construction.

namespace {

struct CartIndex {
    int lx;
    int ly;
    int lz;
};

// Libint "standard" Cartesian ordering, matching the FOR_CART macro:
//   for lx = l .. 0 (step -1)
//     for ly = l-lx .. 0 (step -1)
//       lz = l - lx - ly
std::vector<CartIndex> enumerate_cartesians(int l) {
    std::vector<CartIndex> out;
    out.reserve((l + 1) * (l + 2) / 2);
    for (int lx = l; lx >= 0; --lx) {
        for (int ly = l - lx; ly >= 0; --ly) {
            out.push_back({lx, ly, l - lx - ly});
        }
    }
    return out;
}

// Pow for small non-negative integer exponents, used for (dx)^lx etc.
double ipow(double x, int n) {
    double r = 1.0;
    for (int k = 0; k < n; ++k) r *= x;
    return r;
}

// Build the Cartesian-to-spherical transform matrix T of size
// (2l+1, n_cart), with T(m+l, i) = coeff of i-th Cartesian in m-th pure
// function. libint's SolidHarmonicsCoefficients::coeff already includes the
// sqrt((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!)) conversion (solidharmonics.h),
// so it consumes Cartesian components in libint's shell-wide *axial*
// normalization — exactly what the bare monomials above plus libint's stored
// contraction coefficients produce — and *no* per-Cartesian prefactor on T
// is needed here.
//
// The `m = -l..+l` row order below hardcodes libint's STANDARD
// solid-harmonic ordering. `coeff(l, m, ...)` is the raw per-(l, m)
// coefficient and does not itself know the shell ordering, so nothing here
// would complain under GAUSSIAN ordering — the rows would just come out
// permuted relative to the integral engine's own pure shells. That invariant
// is asserted centrally: see the static_assert in vibeqc/basis.hpp and the
// runtime check in ensure_libint_initialized() (cpp/src/init.cpp).
Eigen::MatrixXd cart_to_pure_transform(int l,
                                       const std::vector<CartIndex>& cart) {
    const int n_cart = static_cast<int>(cart.size());
    Eigen::MatrixXd T(2 * l + 1, n_cart);
    using libint2::solidharmonics::SolidHarmonicsCoefficients;
    for (int m = -l; m <= l; ++m) {
        for (int i = 0; i < n_cart; ++i) {
            T(m + l, i) = SolidHarmonicsCoefficients<double>::coeff(
                l, m, cart[i].lx, cart[i].ly, cart[i].lz);
        }
    }
    return T;
}

}  // namespace

Eigen::MatrixXd evaluate_ao(const BasisSet& basis,
                            const Eigen::MatrixX3d& points) {
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto n_pts = points.rows();
    const auto n_bf = static_cast<Eigen::Index>(basis.nbasis());

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_pts, n_bf);

    // One parallel region around the whole shell loop. The per-shell grid-point
    // work is distributed with `omp for` + `nowait`: each shell writes a
    // disjoint column block out(:, bf..bf+nm), so no inter-shell barrier is
    // needed, and the team forks/joins ONCE instead of once per shell. (The
    // old per-shell `parallel for` measured ~1.9x on 4 threads / cc-pVTZ; this
    // recovers the scaling.) Per-shell setup is small and runs once per thread.
    const Eigen::Index n_pts_i = n_pts;
    #pragma omp parallel if(!omp_in_parallel_region())
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const auto& shell = shells[s];
        const auto& contr = shell.contr[0];   // we use simple (non-SP) shells
        const int l = contr.l;
        const bool pure = contr.pure;
        const auto bf = shell2bf[s];
        const auto n_prim = shell.alpha.size();
        const auto& alpha = shell.alpha;
        const auto& coeff = contr.coeff;
        const auto& O = shell.O;

        const auto cart = enumerate_cartesians(l);
        const int n_cart = static_cast<int>(cart.size());

        // Transform to apply to cartesian values if the shell is pure.
        const Eigen::MatrixXd T =
            pure ? cart_to_pure_transform(l, cart) : Eigen::MatrixXd();

        #pragma omp for schedule(static) nowait
        for (Eigen::Index g = 0; g < n_pts_i; ++g) {
            const double dx = points(g, 0) - O[0];
            const double dy = points(g, 1) - O[1];
            const double dz = points(g, 2) - O[2];
            const double r2 = dx * dx + dy * dy + dz * dz;

            // Contracted radial: Σ_k c_k exp(-α_k r²)
            double radial = 0.0;
            for (std::size_t k = 0; k < n_prim; ++k) {
                radial += coeff[k] * std::exp(-alpha[k] * r2);
            }
            if (radial == 0.0) continue;

            // Cartesian values for this shell, on the libint-normalized
            // monomials (see the convention note at the top of this file).
            Eigen::VectorXd cart_vals(n_cart);
            for (int i = 0; i < n_cart; ++i) {
                cart_vals(i) = ipow(dx, cart[i].lx)
                             * ipow(dy, cart[i].ly)
                             * ipow(dz, cart[i].lz) * radial;
            }

            if (pure) {
                // T contains coeff(l,m,lx,ly,lz) directly, which is defined
                // against exactly these monomials.
                Eigen::VectorXd pure_vals = T * cart_vals;
                for (int m = 0; m < 2 * l + 1; ++m) {
                    out(g, bf + m) = pure_vals(m);
                }
            } else {
                for (int i = 0; i < n_cart; ++i) {
                    out(g, bf + i) = cart_vals(i);
                }
            }
        }
    }
    return out;
}

AOValues evaluate_ao_with_gradient(const BasisSet& basis,
                                   const Eigen::MatrixX3d& points) {
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto n_pts = points.rows();
    const auto n_bf = static_cast<Eigen::Index>(basis.nbasis());

    AOValues result;
    result.values = Eigen::MatrixXd::Zero(n_pts, n_bf);
    for (int c = 0; c < 3; ++c) {
        result.gradients[c] = Eigen::MatrixXd::Zero(n_pts, n_bf);
    }

    const Eigen::Index n_pts_i = n_pts;
    #pragma omp parallel if(!omp_in_parallel_region())
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const auto& shell = shells[s];
        const auto& contr = shell.contr[0];
        const int l = contr.l;
        const bool pure = contr.pure;
        const auto bf = shell2bf[s];
        const auto n_prim = shell.alpha.size();
        const auto& alpha = shell.alpha;
        const auto& coeff = contr.coeff;
        const auto& O = shell.O;

        const auto cart = enumerate_cartesians(l);
        const int n_cart = static_cast<int>(cart.size());
        const Eigen::MatrixXd T =
            pure ? cart_to_pure_transform(l, cart) : Eigen::MatrixXd();

        #pragma omp for schedule(static) nowait
        for (Eigen::Index g = 0; g < n_pts_i; ++g) {
            const double dx = points(g, 0) - O[0];
            const double dy = points(g, 1) - O[1];
            const double dz = points(g, 2) - O[2];
            const double r2 = dx * dx + dy * dy + dz * dz;

            // Two radial sums we need:
            //   R0 = Σ_k c_k exp(-α_k r²)
            //   R1 = Σ_k c_k (-2α_k) exp(-α_k r²)    (for gradient via -2α dx χ term)
            double R0 = 0.0, R1 = 0.0;
            for (std::size_t k = 0; k < n_prim; ++k) {
                const double e = coeff[k] * std::exp(-alpha[k] * r2);
                R0 += e;
                R1 += -2.0 * alpha[k] * e;
            }
            if (R0 == 0.0 && R1 == 0.0) continue;

            // Cartesian values and gradient components. For a Cartesian
            // primitive (dx)^lx (dy)^ly (dz)^lz × exp(-α r²), the spatial
            // gradient in x is:
            //   [lx (dx)^{lx-1} - 2α dx (dx)^lx] (dy)^ly (dz)^lz exp(-α r²)
            // Summing over primitives with normalized coefficients yields
            //   value = (dx)^lx (dy)^ly (dz)^lz R0
            //   grad_x = lx (dx)^{lx-1} (dy)^ly (dz)^lz R0
            //          + (dx)^lx (dy)^ly (dz)^lz × dx × R1
            Eigen::VectorXd v(n_cart);
            Eigen::VectorXd gx(n_cart), gy(n_cart), gz(n_cart);
            for (int i = 0; i < n_cart; ++i) {
                const int lx = cart[i].lx;
                const int ly = cart[i].ly;
                const int lz = cart[i].lz;
                const double xlx  = ipow(dx, lx);
                const double ylm  = ipow(dy, ly);
                const double zln  = ipow(dz, lz);
                const double xlxm = (lx > 0) ? ipow(dx, lx - 1) : 0.0;
                const double ylym = (ly > 0) ? ipow(dy, ly - 1) : 0.0;
                const double zlzm = (lz > 0) ? ipow(dz, lz - 1) : 0.0;
                const double mon = xlx * ylm * zln;

                v(i)  = mon * R0;
                gx(i) = lx * xlxm * ylm * zln * R0 + mon * dx * R1;
                gy(i) = ly * xlx * ylym * zln * R0 + mon * dy * R1;
                gz(i) = lz * xlx * ylm * zlzm * R0 + mon * dz * R1;
            }

            if (pure) {
                // T is plain coeff(l,m,lx,ly,lz), defined against the
                // libint-normalized monomials v, gx, gy, gz computed above.
                Eigen::VectorXd pv  = T * v;
                Eigen::VectorXd pgx = T * gx;
                Eigen::VectorXd pgy = T * gy;
                Eigen::VectorXd pgz = T * gz;
                for (int m = 0; m < 2 * l + 1; ++m) {
                    result.values      (g, bf + m) = pv(m);
                    result.gradients[0](g, bf + m) = pgx(m);
                    result.gradients[1](g, bf + m) = pgy(m);
                    result.gradients[2](g, bf + m) = pgz(m);
                }
            } else {
                for (int i = 0; i < n_cart; ++i) {
                    result.values      (g, bf + i) = v (i);
                    result.gradients[0](g, bf + i) = gx(i);
                    result.gradients[1](g, bf + i) = gy(i);
                    result.gradients[2](g, bf + i) = gz(i);
                }
            }
        }
    }
    return result;
}

AOValuesWithHessian evaluate_ao_with_hessian(const BasisSet& basis,
                                              const Eigen::MatrixX3d& points) {
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto n_pts = points.rows();
    const auto n_bf = static_cast<Eigen::Index>(basis.nbasis());

    AOValuesWithHessian r;
    r.values = Eigen::MatrixXd::Zero(n_pts, n_bf);
    for (int c = 0; c < 3; ++c) {
        r.gradients[c] = Eigen::MatrixXd::Zero(n_pts, n_bf);
    }
    for (int h = 0; h < 6; ++h) {
        r.hessians[h] = Eigen::MatrixXd::Zero(n_pts, n_bf);
    }

    // Hessian index layout: 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz.
    constexpr int H_XX = 0, H_XY = 1, H_XZ = 2, H_YY = 3, H_YZ = 4, H_ZZ = 5;

    const Eigen::Index n_pts_i = n_pts;
    #pragma omp parallel if(!omp_in_parallel_region())
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const auto& shell = shells[s];
        const auto& contr = shell.contr[0];
        const int l = contr.l;
        const bool pure = contr.pure;
        const auto bf = shell2bf[s];
        const auto n_prim = shell.alpha.size();
        const auto& alpha = shell.alpha;
        const auto& coeff = contr.coeff;
        const auto& O = shell.O;

        const auto cart = enumerate_cartesians(l);
        const int n_cart = static_cast<int>(cart.size());
        const Eigen::MatrixXd T =
            pure ? cart_to_pure_transform(l, cart) : Eigen::MatrixXd();

        #pragma omp for schedule(static) nowait
        for (Eigen::Index g = 0; g < n_pts_i; ++g) {
            const double dx = points(g, 0) - O[0];
            const double dy = points(g, 1) - O[1];
            const double dz = points(g, 2) - O[2];
            const double r2 = dx * dx + dy * dy + dz * dz;

            // Contracted radial and its r²-derivatives:
            //   R_n(r²) = Σ_k c_k α_k^n exp(-α_k r²)
            double R0 = 0.0, R1 = 0.0, R2 = 0.0;
            for (std::size_t k = 0; k < n_prim; ++k) {
                const double e = coeff[k] * std::exp(-alpha[k] * r2);
                R0 += e;
                R1 += alpha[k] * e;
                R2 += alpha[k] * alpha[k] * e;
            }
            if (R0 == 0.0 && R1 == 0.0 && R2 == 0.0) continue;

            Eigen::VectorXd vv (n_cart);
            Eigen::VectorXd gx (n_cart), gy (n_cart), gz (n_cart);
            Eigen::VectorXd hxx(n_cart), hxy(n_cart), hxz(n_cart),
                           hyy(n_cart), hyz(n_cart), hzz(n_cart);
            for (int i = 0; i < n_cart; ++i) {
                const int lx = cart[i].lx;
                const int ly = cart[i].ly;
                const int lz = cart[i].lz;

                // Powers of (dx,dy,dz) with some "missing" at low orders
                // where the derivative drops the monomial.
                const double x0 = ipow(dx, lx);
                const double y0 = ipow(dy, ly);
                const double z0 = ipow(dz, lz);
                const double x1 = (lx > 0) ? ipow(dx, lx - 1) : 0.0;
                const double y1 = (ly > 0) ? ipow(dy, ly - 1) : 0.0;
                const double z1 = (lz > 0) ? ipow(dz, lz - 1) : 0.0;
                const double x2 = (lx > 1) ? ipow(dx, lx - 2) : 0.0;
                const double y2 = (ly > 1) ? ipow(dy, ly - 2) : 0.0;
                const double z2 = (lz > 1) ? ipow(dz, lz - 2) : 0.0;

                const double M   = x0 * y0 * z0;
                const double dMx = lx * x1 * y0 * z0;
                const double dMy = ly * x0 * y1 * z0;
                const double dMz = lz * x0 * y0 * z1;
                const double dMxx = lx * (lx - 1) * x2 * y0 * z0;  // 0 if lx<2
                const double dMyy = ly * (ly - 1) * x0 * y2 * z0;
                const double dMzz = lz * (lz - 1) * x0 * y0 * z2;
                const double dMxy = lx * ly * x1 * y1 * z0;
                const double dMxz = lx * lz * x1 * y0 * z1;
                const double dMyz = ly * lz * x0 * y1 * z1;

                // Value  χ = M R_0
                vv(i) = M * R0;

                // Gradient  ∂_c χ = (∂_c M) R_0 − 2 dx_c M R_1
                gx(i) = dMx * R0 - 2.0 * dx * M * R1;
                gy(i) = dMy * R0 - 2.0 * dy * M * R1;
                gz(i) = dMz * R0 - 2.0 * dz * M * R1;

                // Hessian  ∂²χ/∂x_c∂x_d
                //   = ∂_c∂_d M × R_0
                //   − 2 R_1 [dx_c ∂_d M + dx_d ∂_c M + δ_{cd} M]
                //   + 4 dx_c dx_d M R_2
                hxx(i) = dMxx * R0 - 2.0 * R1 * (2.0 * dx * dMx + M)
                       + 4.0 * dx * dx * M * R2;
                hyy(i) = dMyy * R0 - 2.0 * R1 * (2.0 * dy * dMy + M)
                       + 4.0 * dy * dy * M * R2;
                hzz(i) = dMzz * R0 - 2.0 * R1 * (2.0 * dz * dMz + M)
                       + 4.0 * dz * dz * M * R2;
                hxy(i) = dMxy * R0 - 2.0 * R1 * (dx * dMy + dy * dMx)
                       + 4.0 * dx * dy * M * R2;
                hxz(i) = dMxz * R0 - 2.0 * R1 * (dx * dMz + dz * dMx)
                       + 4.0 * dx * dz * M * R2;
                hyz(i) = dMyz * R0 - 2.0 * R1 * (dy * dMz + dz * dMy)
                       + 4.0 * dy * dz * M * R2;
            }

            auto write = [&](const Eigen::VectorXd& cart_vec,
                             Eigen::MatrixXd& out) {
                if (pure) {
                    Eigen::VectorXd pv = T * cart_vec;
                    for (int m = 0; m < 2 * l + 1; ++m) out(g, bf + m) = pv(m);
                } else {
                    for (int i = 0; i < n_cart; ++i) {
                        out(g, bf + i) = cart_vec(i);
                    }
                }
            };
            write(vv,  r.values);
            write(gx,  r.gradients[0]);
            write(gy,  r.gradients[1]);
            write(gz,  r.gradients[2]);
            write(hxx, r.hessians[H_XX]);
            write(hxy, r.hessians[H_XY]);
            write(hxz, r.hessians[H_XZ]);
            write(hyy, r.hessians[H_YY]);
            write(hyz, r.hessians[H_YZ]);
            write(hzz, r.hessians[H_ZZ]);
        }
    }
    return r;
}

Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
evaluate_bloch_ao(
    const BasisSet& basis,
    const Eigen::MatrixX3d& points,
    const Eigen::Vector3d& k_cart,
    const Eigen::MatrixX3d& lattice_translations) {

    if (lattice_translations.cols() != 3) {
        throw std::runtime_error(
            "evaluate_bloch_ao: lattice_translations must be (n_T, 3)");
    }

    const auto n_bf = static_cast<Eigen::Index>(basis.nbasis());
    const Eigen::Index n_pts = points.rows();
    const Eigen::Index n_T   = lattice_translations.rows();

    // Real and imaginary accumulators for χ^k(r); the complex result is
    // assembled at the end. χ_μ is real, so each translation contributes
    // (cos φ · χ) to the real part and (sin φ · χ) to the imaginary part.
    Eigen::MatrixXd chi_re = Eigen::MatrixXd::Zero(n_pts, n_bf);
    Eigen::MatrixXd chi_im = Eigen::MatrixXd::Zero(n_pts, n_bf);

    Eigen::MatrixX3d shifted_points(n_pts, 3);

    for (Eigen::Index t = 0; t < n_T; ++t) {
        const double Tx = lattice_translations(t, 0);
        const double Ty = lattice_translations(t, 1);
        const double Tz = lattice_translations(t, 2);

        shifted_points.col(0) = points.col(0).array() - Tx;
        shifted_points.col(1) = points.col(1).array() - Ty;
        shifted_points.col(2) = points.col(2).array() - Tz;

        // χ_μ(r − T) for every basis function and every grid point.
        const Eigen::MatrixXd chi = evaluate_ao(basis, shifted_points);

        // Bloch phase exp(i k·T) = cos φ + i sin φ.
        const double phase = k_cart(0) * Tx + k_cart(1) * Ty + k_cart(2) * Tz;
        const double cp = std::cos(phase);
        const double sp = std::sin(phase);

        chi_re.noalias() += cp * chi;
        chi_im.noalias() += sp * chi;
    }

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
        chi_k(n_pts, n_bf);
    chi_k.real() = chi_re;
    chi_k.imag() = chi_im;
    return chi_k;
}

}  // namespace vibeqc
