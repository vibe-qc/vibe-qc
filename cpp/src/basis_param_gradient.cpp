#include "vibeqc/basis_param_gradient.hpp"

#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"  // compute_overlap / kinetic / nuclear
#include "vibeqc/cart_to_sph_data.hpp"

#include <libint2/engine.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vibeqc {

// dS/dα_q via second-moment (emultipole2) integrals about the differentiated
// shell's own centre A. With μ = N_c · Σ_p c_p Ñ(α_p) Y_lm e^{−α_p r²}
// (libint primitive- *and* contracted-normalised), differentiating one
// exponent gives, for the q-th primitive,
//
//   ∂φ̂(α_q)/∂α_q = (2l+3)/(4α_q) φ̂(α_q) − r²_A φ̂(α_q),
//
// (the first term is ∂lnÑ/∂α_q = (2l+3)/(4α_q); the second is
// ∂/∂α e^{−αr²} = −r² e^{−αr²}). Hence
//
//   ∂S_{μν}/∂α_q = [ (2l+3)/(4α_q)·⟨aux_q|ν⟩ − ⟨aux_q|r²_A|ν⟩ ]
//                + (∂lnN_c/∂α_q)·S_{μν},
//
// where aux_q is the single primitive A_q·(bare Gaussian) (A_q = μ's stored
// libint coeff for q), ⟨·|r²_A|·⟩ is the trace of the second-moment tensor
// about A (= Q_xx+Q_yy+Q_zz from emultipole2 with origin A), and the
// contracted-renormalisation response is, by ⟨μ|μ⟩ ≡ 1,
//
//   ∂lnN_c/∂α_q = −[ (2l+3)/(4α_q)·⟨aux_q|μ⟩ − ⟨aux_q|r²_A|μ⟩ ].
//
// This needs no l+2 shells and no cartesian↔spherical transform; libint's
// emultipole2 already outputs in the spherical basis. (Kinetic / nuclear /
// ERI exponent derivatives — Phase 1b — do need the r²-weighted bra as l+2
// Gaussians and are added separately.)
Eigen::MatrixXd overlap_exponent_derivative(const BasisSet& basis,
                                            int shell_idx, int prim_idx) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nsh = static_cast<int>(shells.size());
    if (shell_idx < 0 || shell_idx >= nsh) {
        throw std::out_of_range("overlap_exponent_derivative: shell_idx");
    }
    const auto& mu = shells[shell_idx];
    if (prim_idx < 0 || prim_idx >= static_cast<int>(mu.alpha.size())) {
        throw std::out_of_range("overlap_exponent_derivative: prim_idx");
    }

    const int l = mu.contr[0].l;
    const bool pure = mu.contr[0].pure;
    const double alpha_q = mu.alpha[prim_idx];
    const double coeff_q = mu.contr[0].coeff[prim_idx];  // libint-internal A_q
    const std::array<double, 3> O = mu.O;
    const double pref = (2.0 * l + 3.0) / (4.0 * alpha_q);

    // aux_q = single primitive A_q·(bare Gaussian), coeff taken as-given
    // (do_enforce_unit_normalization = false) so it is exactly the q-th term
    // as it sits inside μ.
    libint2::svector<libint2::Shell::real_t> aux_alpha{alpha_q};
    libint2::svector<libint2::Shell::real_t> aux_coeff{coeff_q};
    libint2::svector<libint2::Shell::Contraction> aux_contrs;
    aux_contrs.push_back(libint2::Shell::Contraction{l, pure, std::move(aux_coeff)});
    libint2::Shell aux{std::move(aux_alpha), std::move(aux_contrs),
                       {O[0], O[1], O[2]}, /*do_enforce_unit_normalization=*/false};

    const auto nbf = basis.nbasis();
    const auto shell2bf = shells.shell2bf();
    const auto mubf = shell2bf[shell_idx];
    const std::size_t nmu = mu.size();

    libint2::Engine engine(libint2::Operator::emultipole2,
                           std::max<std::size_t>(shells.max_nprim(), 1),
                           shells.max_l(), 0);
    engine.set_params(std::array<double, 3>{O[0], O[1], O[2]});
    const auto& buf = engine.results();

    const Eigen::MatrixXd S = compute_overlap(basis);

    // Per-primitive term, accumulated over the μ rows (nmu × nbf).
    Eigen::MatrixXd perprim =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nmu), nbf);
    for (int s2 = 0; s2 < nsh; ++s2) {
        const auto bf2 = shell2bf[s2];
        const auto n2 = shells[s2].size();
        engine.compute(aux, shells[s2]);
        if (!buf[0]) continue;
        for (std::size_t a = 0; a < nmu; ++a) {
            for (std::size_t b = 0; b < n2; ++b) {
                const std::size_t idx = a * n2 + b;
                const double ov = buf[0][idx];
                const double qt = buf[4][idx] + buf[7][idx] + buf[9][idx];
                perprim(static_cast<Eigen::Index>(a),
                        static_cast<Eigen::Index>(bf2 + b)) = pref * ov - qt;
            }
        }
    }

    // Contracted-renormalisation response (scalar; diagonal m-component).
    engine.compute(aux, mu);
    double dln_nc = 0.0;
    if (buf[0]) {
        const double ov = buf[0][0];
        const double qt = buf[4][0] + buf[7][0] + buf[9][0];
        dln_nc = -(pref * ov - qt);
    }

    // Assemble symmetric ∂S = perprim + dlnN_c·S on the μ block.
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(nbf, nbf);
    for (std::size_t a = 0; a < nmu; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(mubf + a);
        for (Eigen::Index k = 0; k < nbf; ++k) {
            const double v = perprim(static_cast<Eigen::Index>(a), k)
                             + dln_nc * S(row, k);
            Z(row, k) = v;
            Z(k, row) = v;
        }
    }
    return Z;
}

// ---------------------------------------------------------------------------
// Phase 1b: kinetic / nuclear exponent derivatives (operator between the
// r²-weighted bra and ν). The r²·φ̂ bra is a degree-(l+2) Gaussian, so we
// build it from libint integrals over a bare cartesian (l+2) shell and map
// back to the spherical μ axis with the generated cart_to_sph tables:
//
//   ⟨r²·φ̂_lm | Ô | ν⟩ = A_q · Σ_a C_l[m,a]·N_cart(a,α_q)
//                              · Σ_{shift∈{x,y,z}} ⟨bare_cart_{l+2,a+shift}|Ô|ν⟩
//
// φ̂_lm = Σ_a C_l[m,a]·N_cart(a)·(bare cart_l monomial a); r²·(bare monomial
// (i,j,k)) = (i+2,j,k)+(i,j+2,k)+(i,j,k+2). The full μ-side derivative is
//   d_m,k = (2l+3)/(4α_q)⟨aux_m|Ô|k⟩ − ⟨r²·aux_m|Ô|k⟩ + (∂lnN_c/∂α_q)·I_mk
// and ∂I = symmetrise(d), with the μ-μ block carrying BOTH bra and ket
// derivatives (∂I_μμ = ⟨∂μ|Ô|μ⟩ + ⟨μ|Ô|∂μ⟩), unlike overlap where it vanishes.
namespace {

const cart_to_sph_data::CartIdx* cart_list(int l, int& n) {
    using namespace cart_to_sph_data;
    switch (l) {
        case 0: n = 1;  return kCart0.data();
        case 1: n = 3;  return kCart1.data();
        case 2: n = 6;  return kCart2.data();
        case 3: n = 10; return kCart3.data();
        case 4: n = 15; return kCart4.data();
        case 5: n = 21; return kCart5.data();
        case 6: n = 28; return kCart6.data();
        default: n = 0; return nullptr;
    }
}

int cart_index(int i, int j, int k) {
    int n = 0;
    const auto* lst = cart_list(i + j + k, n);
    for (int a = 0; a < n; ++a) {
        if (lst[a].i == i && lst[a].j == j && lst[a].k == k) return a;
    }
    return -1;
}

Eigen::MatrixXd op_exp_deriv_impl(
    const BasisSet& basis, int shell_idx, int prim_idx, libint2::Operator op,
    const std::vector<std::pair<double, std::array<double, 3>>>* nuclei,
    const Eigen::MatrixXd& Iop) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nsh = static_cast<int>(shells.size());
    if (shell_idx < 0 || shell_idx >= nsh) {
        throw std::out_of_range("op_exponent_derivative: shell_idx");
    }
    const auto& mu = shells[shell_idx];
    if (prim_idx < 0 || prim_idx >= static_cast<int>(mu.alpha.size())) {
        throw std::out_of_range("op_exponent_derivative: prim_idx");
    }
    const int l = mu.contr[0].l;
    const bool pure = mu.contr[0].pure;
    const double alpha_q = mu.alpha[prim_idx];
    const double coeff_q = mu.contr[0].coeff[prim_idx];
    const std::array<double, 3> O = mu.O;
    const double pref = (2.0 * l + 3.0) / (4.0 * alpha_q);
    if (l + 2 > cart_to_sph_data::kMaxL) {
        throw std::runtime_error("op_exponent_derivative: l+2 exceeds kMaxL");
    }

    libint2::Shell aux{libint2::svector<libint2::Shell::real_t>{alpha_q},
        {libint2::Shell::Contraction{l, pure,
            libint2::svector<libint2::Shell::real_t>{coeff_q}}},
        {O[0], O[1], O[2]}, /*do_enforce_unit_normalization=*/false};
    libint2::Shell cart2{libint2::svector<libint2::Shell::real_t>{alpha_q},
        {libint2::Shell::Contraction{l + 2, /*pure=*/false,
            libint2::svector<libint2::Shell::real_t>{1.0}}},
        {O[0], O[1], O[2]}, /*do_enforce_unit_normalization=*/false};

    const auto nbf = basis.nbasis();
    const auto shell2bf = shells.shell2bf();
    const auto mubf = shell2bf[shell_idx];
    const std::size_t nmu = mu.size();

    // Renorm scalar ∂lnN_c/∂α_q via emultipole2 (operator-independent).
    double dln_nc = 0.0;
    {
        libint2::Engine me(libint2::Operator::emultipole2,
                           std::max<std::size_t>(shells.max_nprim(), 1),
                           shells.max_l(), 0);
        me.set_params(std::array<double, 3>{O[0], O[1], O[2]});
        const auto& mb = me.results();
        me.compute(aux, mu);
        if (mb[0]) {
            dln_nc = -(pref * mb[0][0] - (mb[4][0] + mb[7][0] + mb[9][0]));
        }
    }

    libint2::Engine engine(op, std::max<std::size_t>(shells.max_nprim(), 1),
                           std::max<int>(shells.max_l(), l + 2), 0);
    if (op == libint2::Operator::nuclear && nuclei) engine.set_params(*nuclei);
    const auto& buf = engine.results();

    int ncl = 0;
    const auto* cl = cart_list(l, ncl);

    // libint's do_enforce=false cartesian shells carry an L-dependent
    // prefactor K_L (a-independent within a shell), so r²·(cart_l) =
    // (K_l/K_{l+2})·Σ_shift cart_{l+2}. Measure the ratio from the (L,0,0)
    // self-overlaps (no hard-coded libint convention):
    //   K_L² = selfov_L(L,0,0) / I_bare(L);  I_bare(l+2)/I_bare(l) =
    //   (2l+3)(2l+1)/(4α)²  ⇒  K_l/K_{l+2} =
    //   √(selfov_l/selfov_{l+2})·√((2l+3)(2l+1))/(4α).
    auto ll0_selfov = [&](int L) -> double {
        libint2::Shell sh{libint2::svector<libint2::Shell::real_t>{alpha_q},
            {libint2::Shell::Contraction{L, false,
                libint2::svector<libint2::Shell::real_t>{1.0}}},
            {O[0], O[1], O[2]}, false};
        libint2::Engine se(libint2::Operator::overlap, 1, L, 0);
        se.compute(sh, sh);
        return se.results()[0][0];  // (L,0,0)-(L,0,0) element
    };
    const double kfac = std::sqrt(ll0_selfov(l) / ll0_selfov(l + 2))
        * std::sqrt((2.0 * l + 3.0) * (2.0 * l + 1.0)) / (4.0 * alpha_q);

    // Express aux (the pure m-components, do_enforce=false, coeff A_q baked
    // in) in the do_enforce=false cartesian-l basis EMPIRICALLY, so the r²
    // mapping stays consistent with libint's actual convention rather than
    // assuming the *normalised* cart_to_sph (kSph). T = ⟨aux|cart_l⟩·
    // ⟨cart_l|cart_l⟩⁻¹  (nmu × ncl). Then
    //   ⟨r²·aux_m|Ô|ν⟩ = kfac · Σ_a T[m,a] · Σ_shift ⟨cart_{l+2,shift}|Ô|ν⟩.
    Eigen::MatrixXd T;
    {
        libint2::Shell cart_l{libint2::svector<libint2::Shell::real_t>{alpha_q},
            {libint2::Shell::Contraction{l, /*pure=*/false,
                libint2::svector<libint2::Shell::real_t>{1.0}}},
            {O[0], O[1], O[2]}, false};
        libint2::Engine te(libint2::Operator::overlap, 1, l, 0);
        const auto& tb = te.results();
        Eigen::MatrixXd A(static_cast<Eigen::Index>(nmu), ncl);
        te.compute(aux, cart_l);
        for (std::size_t m = 0; m < nmu; ++m)
            for (int a = 0; a < ncl; ++a)
                A(static_cast<Eigen::Index>(m), a) = tb[0][m * ncl + a];
        Eigen::MatrixXd G(ncl, ncl);
        te.compute(cart_l, cart_l);
        for (int a = 0; a < ncl; ++a)
            for (int a2 = 0; a2 < ncl; ++a2)
                G(a, a2) = tb[0][a * ncl + a2];
        T = A * G.inverse();
    }

    // d(m,k) = full μ-side derivative ⟨∂μ_m|Ô|k⟩.
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nmu), nbf);
    for (int s2 = 0; s2 < nsh; ++s2) {
        const auto bf2 = shell2bf[s2];
        const auto n2 = shells[s2].size();

        engine.compute(aux, shells[s2]);
        if (!buf[0]) continue;
        std::vector<double> clean(buf[0], buf[0] + nmu * n2);

        engine.compute(cart2, shells[s2]);
        const double* cb = buf[0];

        for (std::size_t m = 0; m < nmu; ++m) {
            for (std::size_t b = 0; b < n2; ++b) {
                // libint's do_enforce=false cartesian shell carries the
                // *radial-only* prefactor (2α/π)^¾ — the SAME for degree l and
                // l+2 — so r²·(cart_l monomial) = Σ_shift cart_{l+2,shift}
                // exactly (coefficient 1), and ⟨r²·φ̂_lm|Ô|ν⟩ = Σ_a C_l[m,a]·
                // Σ_shift ⟨cart_{l+2,shift}|Ô|ν⟩. (Verified against the
                // emultipole2 reference once the spurious per-shift factor was
                // dropped.)
                double r2 = 0.0;
                for (int a = 0; a < ncl; ++a) {
                    const double tma = T(static_cast<Eigen::Index>(m), a);
                    const int i = cl[a].i, j = cl[a].j, k = cl[a].k;
                    r2 += tma * (cb[cart_index(i + 2, j, k) * n2 + b]
                               + cb[cart_index(i, j + 2, k) * n2 + b]
                               + cb[cart_index(i, j, k + 2) * n2 + b]);
                }
                r2 *= kfac;  // coeff_q is already baked into T (built from aux)
                const double row_v = pref * clean[m * n2 + b] - r2
                    + dln_nc * Iop(static_cast<Eigen::Index>(mubf + m),
                                   static_cast<Eigen::Index>(bf2 + b));
                d(static_cast<Eigen::Index>(m),
                  static_cast<Eigen::Index>(bf2 + b)) = row_v;
            }
        }
    }

    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(nbf, nbf);
    for (std::size_t m = 0; m < nmu; ++m) {
        const Eigen::Index row = static_cast<Eigen::Index>(mubf + m);
        for (Eigen::Index k = 0; k < nbf; ++k) {
            const bool in_mu = (k >= static_cast<Eigen::Index>(mubf)
                                && k < static_cast<Eigen::Index>(mubf + nmu));
            if (in_mu) {
                const std::size_t mp = static_cast<std::size_t>(k) - mubf;
                Z(row, k) = d(static_cast<Eigen::Index>(m), k)
                    + d(static_cast<Eigen::Index>(mp),
                        static_cast<Eigen::Index>(mubf + m));
            } else {
                Z(row, k) = d(static_cast<Eigen::Index>(m), k);
                Z(k, row) = d(static_cast<Eigen::Index>(m), k);
            }
        }
    }
    return Z;
}

}  // namespace

Eigen::MatrixXd kinetic_exponent_derivative(const BasisSet& basis,
                                            int shell_idx, int prim_idx) {
    return op_exp_deriv_impl(basis, shell_idx, prim_idx,
                             libint2::Operator::kinetic, nullptr,
                             compute_kinetic(basis));
}

Eigen::MatrixXd nuclear_exponent_derivative(const BasisSet& basis,
                                            const Molecule& mol,
                                            int shell_idx, int prim_idx) {
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        q.emplace_back(static_cast<double>(a.Z), a.xyz);
    }
    return op_exp_deriv_impl(basis, shell_idx, prim_idx,
                             libint2::Operator::nuclear, &q,
                             compute_nuclear(basis, mol));
}

// ---------------------------------------------------------------------------
// Phase 1b: two-electron (ERI) exponent derivative. Returns the dense flat
// (nbf⁴, row-major) tensor ∂(μν|λσ)/∂α_q. Reuses the one-electron r²-bra
// machinery (aux + cart_{l+2} + measured T + kfac + renorm) on the coulomb
// engine, with the differentiated shell in position 1 (D1), then symmetrises
// over the 4 ERI positions:
//   ∂(μν|λσ) = D1[μνλσ] + D1[νμλσ] + D1[λσμν] + D1[σλμν]
// (each term is the derivative w.r.t. that position's shell exponent; the sum
// covers a shell appearing in several positions, e.g. (SS|λσ)).
std::vector<double> eri_exponent_derivative(const BasisSet& basis,
                                            int shell_idx, int prim_idx) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nsh = static_cast<int>(shells.size());
    if (shell_idx < 0 || shell_idx >= nsh) {
        throw std::out_of_range("eri_exponent_derivative: shell_idx");
    }
    const auto& mu = shells[shell_idx];
    if (prim_idx < 0 || prim_idx >= static_cast<int>(mu.alpha.size())) {
        throw std::out_of_range("eri_exponent_derivative: prim_idx");
    }
    const int l = mu.contr[0].l;
    const bool pure = mu.contr[0].pure;
    const double alpha_q = mu.alpha[prim_idx];
    const double coeff_q = mu.contr[0].coeff[prim_idx];
    const std::array<double, 3> O = mu.O;
    const double pref = (2.0 * l + 3.0) / (4.0 * alpha_q);
    if (l + 2 > cart_to_sph_data::kMaxL) {
        throw std::runtime_error("eri_exponent_derivative: l+2 exceeds kMaxL");
    }

    libint2::Shell aux{libint2::svector<libint2::Shell::real_t>{alpha_q},
        {libint2::Shell::Contraction{l, pure,
            libint2::svector<libint2::Shell::real_t>{coeff_q}}},
        {O[0], O[1], O[2]}, false};
    libint2::Shell cart2{libint2::svector<libint2::Shell::real_t>{alpha_q},
        {libint2::Shell::Contraction{l + 2, false,
            libint2::svector<libint2::Shell::real_t>{1.0}}},
        {O[0], O[1], O[2]}, false};

    const auto nbf = basis.nbasis();
    const auto shell2bf = shells.shell2bf();
    const auto sbf = shell2bf[shell_idx];
    const std::size_t nmu = mu.size();
    int ncl = 0;
    const auto* cl = cart_list(l, ncl);

    auto ll0 = [&](int L) -> double {
        libint2::Shell sh{libint2::svector<libint2::Shell::real_t>{alpha_q},
            {libint2::Shell::Contraction{L, false,
                libint2::svector<libint2::Shell::real_t>{1.0}}},
            {O[0], O[1], O[2]}, false};
        libint2::Engine se(libint2::Operator::overlap, 1, L, 0);
        se.compute(sh, sh);
        return se.results()[0][0];
    };
    const double kfac = std::sqrt(ll0(l) / ll0(l + 2))
        * std::sqrt((2.0 * l + 3.0) * (2.0 * l + 1.0)) / (4.0 * alpha_q);

    Eigen::MatrixXd T;
    {
        libint2::Shell cart_l{libint2::svector<libint2::Shell::real_t>{alpha_q},
            {libint2::Shell::Contraction{l, false,
                libint2::svector<libint2::Shell::real_t>{1.0}}},
            {O[0], O[1], O[2]}, false};
        libint2::Engine te(libint2::Operator::overlap, 1, l, 0);
        const auto& tb = te.results();
        Eigen::MatrixXd A(static_cast<Eigen::Index>(nmu), ncl);
        te.compute(aux, cart_l);
        for (std::size_t m = 0; m < nmu; ++m)
            for (int a = 0; a < ncl; ++a)
                A(static_cast<Eigen::Index>(m), a) = tb[0][m * ncl + a];
        Eigen::MatrixXd G(ncl, ncl);
        te.compute(cart_l, cart_l);
        for (int a = 0; a < ncl; ++a)
            for (int a2 = 0; a2 < ncl; ++a2) G(a, a2) = tb[0][a * ncl + a2];
        T = A * G.inverse();
    }

    double dln_nc = 0.0;
    {
        libint2::Engine me(libint2::Operator::emultipole2,
                           std::max<std::size_t>(shells.max_nprim(), 1),
                           shells.max_l(), 0);
        me.set_params(std::array<double, 3>{O[0], O[1], O[2]});
        const auto& mb = me.results();
        me.compute(aux, mu);
        if (mb[0]) {
            dln_nc = -(pref * mb[0][0] - (mb[4][0] + mb[7][0] + mb[9][0]));
        }
    }

    const Eri4D eri_full = compute_eri(basis);
    const std::size_t N = static_cast<std::size_t>(nbf);
    auto fidx = [N](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        return ((a * N + b) * N + c) * N + d;
    };

    // D1: derivative w.r.t. the differentiated shell in position 1.
    std::vector<double> D1(N * N * N * N, 0.0);
    libint2::Engine ce(libint2::Operator::coulomb,
                       std::max<std::size_t>(shells.max_nprim(), 1),
                       std::max<int>(shells.max_l(), l + 2), 0);
    const auto& cbuf = ce.results();
    for (int s2 = 0; s2 < nsh; ++s2) {
        const auto bf2 = shell2bf[s2];
        const auto n2 = shells[s2].size();
        for (int s3 = 0; s3 < nsh; ++s3) {
            const auto bf3 = shell2bf[s3];
            const auto n3 = shells[s3].size();
            for (int s4 = 0; s4 < nsh; ++s4) {
                const auto bf4 = shell2bf[s4];
                const auto n4 = shells[s4].size();

                ce.compute(aux, shells[s2], shells[s3], shells[s4]);
                if (!cbuf[0]) continue;
                std::vector<double> ba(cbuf[0], cbuf[0] + nmu * n2 * n3 * n4);
                ce.compute(cart2, shells[s2], shells[s3], shells[s4]);
                const double* bc = cbuf[0];

                for (std::size_t m = 0; m < nmu; ++m)
                    for (std::size_t j = 0; j < n2; ++j)
                        for (std::size_t k = 0; k < n3; ++k)
                            for (std::size_t ll = 0; ll < n4; ++ll) {
                                auto bi = [&](std::size_t c) {
                                    return ((c * n2 + j) * n3 + k) * n4 + ll;
                                };
                                double r2 = 0.0;
                                for (int a = 0; a < ncl; ++a) {
                                    const double tma = T(static_cast<Eigen::Index>(m), a);
                                    const int i = cl[a].i, jj = cl[a].j, kk = cl[a].k;
                                    r2 += tma * (bc[bi(cart_index(i + 2, jj, kk))]
                                               + bc[bi(cart_index(i, jj + 2, kk))]
                                               + bc[bi(cart_index(i, jj, kk + 2))]);
                                }
                                const std::size_t g = fidx(sbf + m, bf2 + j,
                                                           bf3 + k, bf4 + ll);
                                D1[g] = pref * ba[bi(m)] - kfac * r2
                                        + dln_nc * eri_full.data[g];
                            }
            }
        }
    }

    // Symmetrise over the 4 positions.
    std::vector<double> dE(N * N * N * N);
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b)
            for (std::size_t c = 0; c < N; ++c)
                for (std::size_t d = 0; d < N; ++d)
                    dE[fidx(a, b, c, d)] =
                        D1[fidx(a, b, c, d)] + D1[fidx(b, a, c, d)]
                        + D1[fidx(c, d, a, b)] + D1[fidx(d, c, a, b)];
    return dE;
}

}  // namespace vibeqc
