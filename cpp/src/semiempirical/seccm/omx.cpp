// OMx-SECCM: WS-weighted supercell Fock for cyclic clusters.
//
// The supercell Fock is the molecular OMx Hamiltonian evaluated over the WS
// record set (Bredow-Geudtner-Jug construction), the same generalization
// pm6.cpp applies to the molecular PM6 driver:
//
//   * One-center blocks per supercell atom: the unchanged molecular terms.
//   * Two-center terms per directed record (a -> b image, weight x_ab):
//     penetration-corrected core attraction (eqs 13-14 of Dral 2016),
//     eq-17 resonance, two-electron Coulomb/exchange (eq 18), core-core
//     repulsion (eq 20), and the eq-8 F1/F2 one-center VORT corrections.
//   * Three-center terms per (mu, nu; C): the eq-9 G1/G2 VORT corrections
//     and the OM2/OM3 ECP (Weber 2000 eqs 60/63), weighted by the
//     Peintinger-Bredow 2014 scheme (JCC 35, 839, eqs 10-14):
//       - peintinger_eq13 (default): x_muNuC = x_muNu (x_muC + x_nuC)/2,
//         center weights from the union WSSC(MN) = WSSC(M) u WSSC(N), the
//         C-sum over the union images (interaction range extended to +-t).
//       - janetzko_eq10 (opt-in):   x_muNuC = x_muNu x_muC x_nuC /
//         (x_muC + x_nuC), center weights from the individual WSSCs.
//     The ECP is the same three-center pattern with mu, nu on one atom
//     (x_muNu = 1) and C the core-carrying center.
//
// The molecular limit (one replica, zero-translation records with full
// weight) reproduces nddo::run_omx_v2 on the same molecule.  The molecular
// driver files are frozen; this adapter keeps verbatim copies of the small
// anonymous-namespace helpers it needs.

#include "vibeqc/semiempirical/seccm/omx.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/semiempirical/methods/nddo/omx_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/slater_overlap.hpp"

#include "seccm_common.h"

namespace vibeqc {
namespace semiempirical {
namespace seccm {
namespace {

using nddo::nddo_sto_overlap;
using nddo::omx_resonance_integral;
using nddo::slater_dir_cos;
using nddo::valence_n_sp;

constexpr double EV2HA = 1.0 / 27.2114;
constexpr double HERMITICITY_TOLERANCE = 1.0e-10;

// Valence electron count (verbatim from omx_fock.cpp).
static int omx_valence_electrons(int Z) {
    switch (Z) {
        case  1: return 1;   // H
        case  6: return 4;   // C
        case  7: return 5;   // N
        case  8: return 6;   // O
        case  9: return 7;   // F
        case 16: return 6;   // S
        case 17: return 7;   // Cl
        default: return Z - 2; // rough fallback for main-group
    }
}

// Radial OMx resonance kernel, eq 17 of Dral 2016 (verbatim from
// omx_fock.cpp; returns eV).
static inline double omx_beta_radial(double beta_a, double alpha_a,
                                     double beta_b, double alpha_b,
                                     double R) {
    return 0.5 * (beta_a + beta_b) * std::sqrt(R)
           * std::exp(-(alpha_a + alpha_b) * R * R);
}

// Klopman-Ohno monopole gamma_AB (Ha), eq 19 of Dral 2016 (verbatim from
// omx_fock.cpp).
static inline double omx_gamma_k(const nddo::OMxElementData& ed_a,
                                 const nddo::OMxElementData& ed_b,
                                 double R) {
    double gss_a = ed_a.gss > 0.0 ? ed_a.gss * EV2HA : 0.367;
    double gss_b = ed_b.gss > 0.0 ? ed_b.gss * EV2HA : 0.367;
    double eta = 0.5 * (1.0 / gss_a + 1.0 / gss_b);
    return 1.0 / std::sqrt(R * R + eta * eta);
}

// sigma/pi channel weights of an AO's density relative to a pair axis
// (verbatim from omx_fock.cpp).
struct ChannelWeights {
    double w[3];  // [s, psigma, ppi]
};

static inline ChannelWeights ao_channel_weights(int t, double dx, double dy,
                                                double dz, double R) {
    ChannelWeights cw{{0.0, 0.0, 0.0}};
    if (t == 0) {
        cw.w[0] = 1.0;
    } else {
        const double c = slater_dir_cos(t - 1, dx, dy, dz, R);
        cw.w[1] = c * c;
        cw.w[2] = 1.0 - c * c;
    }
    return cw;
}

// f_KO + channel-resolved Coulomb/attraction integrals for one atom pair at
// distance R (verbatim from the omx_setup pair loop in omx_fock.cpp).
struct OmxPairIntegrals {
    double f_ko = 1.0;
    double gee[3][3];
    double vat_a[3];
    double vat_b[3];
    double vs_a_s = 0.0;
    double vs_a_p = 0.0;
    double vs_b_s = 0.0;
    double vs_b_p = 0.0;
};

static OmxPairIntegrals compute_pair_integrals(
    const nddo::OMxElementData* ed_a, int n_a, double zeta_a,
    const nddo::OMxElementData* ed_b, int n_b, double zeta_b, double R) {
    namespace msi = ::vibeqc::semiempirical::indo;
    OmxPairIntegrals pi;

    // f_KO (eq 19): Klopman monopole over analytic (ss,ss).
    const double g_k = omx_gamma_k(*ed_a, *ed_b, R);
    const double g_a_ss =
        msi::c2int(n_a, 0, 0, zeta_a, n_b, 0, 0, zeta_b, R);
    pi.f_ko = (g_a_ss > 1e-12) ? g_k / g_a_ss : 1.0;

    // Channel-resolved (mu mu, nu nu)^a (eq 18): rows/cols = [s, psigma, ppi].
    const int l_of[3] = {0, 1, 1};
    const int m_of[3] = {0, 0, 1};
    const bool has_p_a = (n_a > 1), has_p_b = (n_b > 1);
    for (int ia = 0; ia < 3; ++ia) {
        for (int ib = 0; ib < 3; ++ib) {
            if ((ia > 0 && !has_p_a) || (ib > 0 && !has_p_b)) {
                pi.gee[ia][ib] = g_a_ss;
                continue;
            }
            pi.gee[ia][ib] =
                msi::c2int(n_a, l_of[ia], m_of[ia], zeta_a,
                           n_b, l_of[ib], m_of[ib], zeta_b, R);
        }
    }

    // Channel-resolved attraction <mu|1/r_X|mu> (eqs 13-14).
    for (int i = 0; i < 3; ++i) {
        pi.vat_a[i] = (i > 0 && !has_p_a)
                          ? msi::v2int(n_a, 0, 0, zeta_a, R)
                          : msi::v2int(n_a, l_of[i], m_of[i], zeta_a, R);
        pi.vat_b[i] = (i > 0 && !has_p_b)
                          ? msi::v2int(n_b, 0, 0, zeta_b, R)
                          : msi::v2int(n_b, l_of[i], m_of[i], zeta_b, R);
    }

    // Spherically-averaged (mu mu, s_X s_X) for the eq-10 local energies.
    pi.vs_a_s = pi.gee[0][0];
    pi.vs_a_p = (pi.gee[1][0] + 2.0 * pi.gee[2][0]) / 3.0;
    pi.vs_b_s = pi.gee[0][0];
    pi.vs_b_p = (pi.gee[0][1] + 2.0 * pi.gee[0][2]) / 3.0;
    return pi;
}

// Local pair energy H^loc_mumu,X = U_mu + V^s_mumu,X (eq 10 of Dral 2016).
// ``vs`` is the spherically-averaged (mu mu, s_X s_X) value for the AO's own
// atom side of the pair (see vs_of below).
static inline double h_loc_vs(double U_mu, double Z_X, double f_ko, double vs) {
    return U_mu - Z_X * f_ko * vs;
}

// Side-resolved accessors into an (lo, hi)-ordered pair (lo = min atom
// index, hi = max atom index), matching the molecular driver's omx_setup
// pair loop.  The MSINDO c2int kernel is not numerically symmetric under
// argument exchange, so the argument order matters at the ~1e-13 level and
// the molecular convention must be reproduced exactly.
static inline double vs_of(const OmxPairIntegrals& pi, int atom, int lo,
                           int t) {
    if (atom == lo) return (t == 0) ? pi.vs_a_s : pi.vs_a_p;
    return (t == 0) ? pi.vs_b_s : pi.vs_b_p;
}

static inline double vat_of(const OmxPairIntegrals& pi, int atom, int lo,
                            int i) {
    return (atom == lo) ? pi.vat_a[i] : pi.vat_b[i];
}

// Pulay DIIS extrapolation state (verbatim from omx_fock.cpp; residual =
// [F, P] in the orthogonal NDDO basis).
struct DiisState {
    std::vector<Eigen::MatrixXd> F, e;
    static constexpr std::size_t MAX = 8;

    void push(const Eigen::MatrixXd& Fi, const Eigen::MatrixXd& ei) {
        F.push_back(Fi);
        e.push_back(ei);
        if (F.size() > MAX) {
            F.erase(F.begin());
            e.erase(e.begin());
        }
    }

    void clear() {
        F.clear();
        e.clear();
    }

    Eigen::MatrixXd extrapolate(const Eigen::MatrixXd& F_in) const {
        const int n = static_cast<int>(F.size());
        if (n < 2) return F_in;
        Eigen::MatrixXd B = Eigen::MatrixXd::Constant(n + 1, n + 1, -1.0);
        B(n, n) = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j)
                B(i, j) = B(j, i) = e[i].cwiseProduct(e[j]).sum();
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
        rhs(n) = -1.0;
        Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);
        if (!c.allFinite()) return F_in;
        Eigen::MatrixXd F_out = Eigen::MatrixXd::Zero(F_in.rows(), F_in.cols());
        for (int i = 0; i < n; ++i) F_out += c(i) * F[i];
        return F_out;
    }
};

// One union image of WSSC(MN) = WSSC(M) u WSSC(N-image): (origin, absolute
// shell label) with the displacement from the real atom M.  The absolute
// labels can reach +-2 shells (one full cluster translation, the paper's
// +-t interaction range).
struct UnionImage {
    int origin = -1;
    Eigen::Vector3d disp_from_a = Eigen::Vector3d::Zero();
    double weight_from_a = 0.0;
    double weight_from_b = 0.0;
};

using UnionKey = std::pair<int, ImageShellLabel>;

std::map<UnionKey, UnionImage> build_ws_union(
    const WSTopology& topology, int a, int b, const ImageShellLabel& pair_label,
    const Eigen::Vector3d& pair_disp) {
    std::map<UnionKey, UnionImage> images;
    for (const auto& image : topology.cells[a]) {
        UnionKey key{image.origin, image.image_shell_label};
        auto [it, inserted] = images.emplace(
            key,
            UnionImage{image.origin, image.disp, image.weight, 0.0});
        if (!inserted) it->second.weight_from_a = image.weight;
    }
    for (const auto& image : topology.cells[b]) {
        ImageShellLabel absolute = image.image_shell_label;
        for (int axis = 0; axis < 3; ++axis) absolute[axis] += pair_label[axis];
        UnionKey key{image.origin, absolute};
        auto [it, inserted] = images.emplace(
            key,
            UnionImage{
                image.origin, pair_disp + image.disp, 0.0, image.weight});
        if (!inserted) it->second.weight_from_b = image.weight;
    }
    return images;
}

// Image-resolved center weight for C in the pair (a, b-image).  The endpoint
// weights are the actual x_muC and x_nuC values of this absolute C image in
// WSSC(M) and the translated WSSC(N), respectively.  Peintinger-Bredow eq 13
// averages those endpoint weights image by image; it does not redistribute a
// unit weight uniformly over all images having the same origin.
static double center_weight(
    OMxThreeCenterWeighting weighting, int c, const UnionImage& image,
    const std::map<int, int>& ws_multiplicity_a,
    const std::map<int, int>& ws_multiplicity_b) {
    if (weighting == OMxThreeCenterWeighting::janetzko_eq10) {
        // Preserve the original Janetzko count convention: n_MC and n_NC
        // count occurrences of atom C in each endpoint WSSC, rather than
        // membership of one absolute image in the eq-13 union.
        const auto it_a = ws_multiplicity_a.find(c);
        const auto it_b = ws_multiplicity_b.find(c);
        const double x_mc =
            (it_a != ws_multiplicity_a.end() && it_a->second > 0)
                ? 1.0 / static_cast<double>(it_a->second)
                : 0.0;
        const double x_nc =
            (it_b != ws_multiplicity_b.end() && it_b->second > 0)
                ? 1.0 / static_cast<double>(it_b->second)
                : 0.0;
        return omx_three_center_endpoint_weight(weighting, x_mc, x_nc);
    }
    return omx_three_center_endpoint_weight(
        weighting, image.weight_from_a, image.weight_from_b);
}

}  // anonymous namespace

OMxSECCMResult run_omx_seccm(
    const Molecule& mol,
    const nddo::OMxParameterSet& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    const OMxSECCMOptions& opts,
    double gap_tolerance) {
    if (params.variant() == nddo::OMxVariant::OM1) {
        throw std::invalid_argument(
            "OM1-SECCM is unavailable because the published analytic "
            "core-valence ECP is not implemented; use OM2- or OM3-SECCM");
    }
    detail::validate_common_inputs(
        mol,
        [&params](int Z) { return params.has_element(Z); },
        topology,
        group_order,
        geometry_tolerance,
        HERMITICITY_TOLERANCE,
        gap_tolerance,
        "OMx-SECCM",
        [&params](int Z) {
            if (!params.omx_data(Z)) {
                throw std::invalid_argument(
                    "OMx-SECCM: element Z=" + std::to_string(Z)
                    + " not in the OMx parameter set");
            }
        },
        [](int, int) {});
    if (opts.max_iter < 1 || !std::isfinite(opts.conv_tol)
        || opts.conv_tol <= 0.0) {
        throw std::invalid_argument(
            "OMx-SECCM options must be positive and finite");
    }
    const bool truncated_electrostatics_required =
        group_order > 1 || !detail::is_trivial_molecular_records(topology);
    if (truncated_electrostatics_required
        && !opts.allow_truncated_electrostatics) {
        throw std::invalid_argument(
            "OM2-/OM3-SECCM cyclic-image calculations require "
            "self-consistent long-range Madelung/Ewald electrostatics, which "
            "this adapter does not implement. The exact molecular limit "
            "(one replica, zero-translation records with full weights) "
            "remains available. Set allow_truncated_electrostatics=True only "
            "to acknowledge an algorithm-mechanics-only calculation that is "
            "not a quantitative solid-state result.");
    }

    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());

    // ---- AO basis: s, px, py, pz per main-group atom ----
    std::vector<int> ao_Z, ao_atom, ao_type;
    int n_basis = 0;
    std::vector<std::vector<int>> aos_on(n_atoms);
    for (int a = 0; a < n_atoms; ++a) {
        const int Z = atoms[a].Z;
        const auto* ed = params.omx_data(Z);
        const int n_ao = (Z <= 2) ? 1 : (ed ? ed->n_orbitals : 4);
        for (int i = 0; i < n_ao; ++i) {
            ao_Z.push_back(Z);
            ao_atom.push_back(a);
            ao_type.push_back(i);
            aos_on[a].push_back(n_basis + i);
        }
        n_basis += n_ao;
    }

    int n_elec = 0;
    for (const auto& atom : atoms) n_elec += omx_valence_electrons(atom.Z);
    const int n_occ = n_elec / 2;
    if (n_occ >= n_basis || n_occ < 1) {
        throw std::invalid_argument(
            "OMx-SECCM occupancy does not fit the finite-cluster AO space "
            "or leaves no LUMO");
    }
    if (n_elec % 2 != 0) {
        throw std::invalid_argument(
            "OMx-SECCM requires a positive even valence-electron count");
    }

    // ---- Pre-load per-AO one-center integrals (eV -> Ha) ----
    std::vector<double> U_s(n_basis, 0.0);
    for (int mu = 0; mu < n_basis; ++mu) {
        const auto* ed = params.omx_data(ao_Z[mu]);
        if (!ed) continue;
        const bool is_s = (ao_type[mu] == 0);
        U_s[mu] = (is_s ? ed->uss : ed->upp) * EV2HA;
    }

    std::vector<int> core_charge(n_atoms);
    for (int a = 0; a < n_atoms; ++a)
        core_charge[a] = omx_valence_electrons(atoms[a].Z);

    const OMxThreeCenterWeighting weighting = opts.three_center_weighting;

    // ---- WS-weighted one-electron core Hamiltonian, Gamma_ee, E_core ----
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd Gamma_ee = Eigen::MatrixXd::Zero(n_basis, n_basis);
    double E_core = 0.0;
    int n_records = 0;

    for (int mu = 0; mu < n_basis; ++mu) H(mu, mu) += U_s[mu];

    for (int a = 0; a < n_atoms; ++a) {
        const auto* ed_a = params.omx_data(atoms[a].Z);
        const int n_a = valence_n_sp(atoms[a].Z);
        const double zeta_a = (ed_a->zeta > 0.0) ? ed_a->zeta : 1.0;
        const auto& idx_a = aos_on[a];
        const int n_ao_a = static_cast<int>(idx_a.size());
        const double F1 = ed_a->F1;
        const double F2 = ed_a->F2;

        for (const auto& image : topology.cells[a]) {
            const int b = image.origin;
            const double w = image.weight;
            const Eigen::Vector3d d_work = -image.disp;
            const double dx = d_work[0], dy = d_work[1], dz = d_work[2];
            const double R = d_work.norm();
            if (R < 1e-12) continue;

            const auto* ed_b = params.omx_data(atoms[b].Z);
            const int n_b = valence_n_sp(atoms[b].Z);
            const double zeta_b = (ed_b->zeta > 0.0) ? ed_b->zeta : 1.0;
            const auto& idx_b = aos_on[b];
            const int n_ao_b = static_cast<int>(idx_b.size());

            // Pair integrals in the molecular driver's (min, max) atom-index
            // order: the MSINDO c2int kernel is not bit-symmetric under
            // argument exchange, so omx_setup's loop order is the reference.
            const int lo = std::min(a, b);
            const int hi = std::max(a, b);
            const auto* ed_lo = (lo == a) ? ed_a : ed_b;
            const auto* ed_hi = (hi == a) ? ed_a : ed_b;
            const int n_lo = (lo == a) ? n_a : n_b;
            const int n_hi = (hi == a) ? n_a : n_b;
            const double zeta_lo = (lo == a) ? zeta_a : zeta_b;
            const double zeta_hi = (hi == a) ? zeta_a : zeta_b;

            const OmxPairIntegrals pi =
                compute_pair_integrals(ed_lo, n_lo, zeta_lo, ed_hi, n_hi, zeta_hi, R);

            ++n_records;

            // (1) Penetration-corrected core attraction on atom A (eqs
            // 13-14 of Dral 2016).
            for (int mu : idx_a) {
                const auto cw = ao_channel_weights(ao_type[mu], dx, dy, dz, R);
                double v = 0.0;
                for (int i = 0; i < 3; ++i) v += cw.w[i] * vat_of(pi, a, lo, i);
                H(mu, mu) -= w * static_cast<double>(core_charge[b])
                             * (pi.f_ko * v);
            }

            // (2) Inter-atomic resonance beta_mu lam (eq 17), directed; the
            // reverse record supplies the transpose and the final
            // symmetrization averages the two halves.
            for (int mu : idx_a) {
                for (int nu : idx_b) {
                    const double beta_val = omx_resonance_integral(
                        ao_type[mu], ao_type[nu], ao_Z[mu], ao_Z[nu],
                        R, dx, dy, dz, params);
                    H(mu, nu) += w * beta_val;
                }
            }

            // (3) Two-center ORT corrections to the one-center block of A
            // (eq 8 of Dral 2016).  S and beta between the a and b-image AOs
            // are computed once per record.
            if (F1 != 0.0 || F2 != 0.0) {
                std::vector<double> S_ab(n_ao_a * n_ao_b, 0.0);
                std::vector<double> beta_ab(n_ao_a * n_ao_b, 0.0);
                for (int i = 0; i < n_ao_a; ++i) {
                    for (int j = 0; j < n_ao_b; ++j) {
                        S_ab[i * n_ao_b + j] = nddo_sto_overlap(
                            n_a, ao_type[idx_a[i]], zeta_a, zeta_a,
                            n_b, ao_type[idx_b[j]], zeta_b, zeta_b,
                            dx, dy, dz, R);
                        beta_ab[i * n_ao_b + j] = omx_resonance_integral(
                            ao_type[idx_a[i]], ao_type[idx_b[j]],
                            ao_Z[idx_a[i]], ao_Z[idx_b[j]],
                            R, dx, dy, dz, params);
                    }
                }
                for (int i = 0; i < n_ao_a; ++i) {
                    const int mu = idx_a[i];
                    const double h_mu_b = h_loc_vs(
                        U_s[mu], static_cast<double>(core_charge[b]),
                        pi.f_ko, vs_of(pi, a, lo, ao_type[mu]));
                    for (int j = i; j < n_ao_a; ++j) {
                        const int nu = idx_a[j];
                        const double h_nu_b = h_loc_vs(
                            U_s[nu], static_cast<double>(core_charge[b]),
                            pi.f_ko, vs_of(pi, a, lo, ao_type[nu]));
                        double v1 = 0.0, v2 = 0.0;
                        for (int k = 0; k < n_ao_b; ++k) {
                            const int rho = idx_b[k];
                            const double h_rho_a = h_loc_vs(
                                U_s[rho], static_cast<double>(core_charge[a]),
                                pi.f_ko, vs_of(pi, b, lo, ao_type[rho]));
                            const double s_mu_rho = S_ab[i * n_ao_b + k];
                            const double s_nu_rho = S_ab[j * n_ao_b + k];
                            const double b_mu_rho = beta_ab[i * n_ao_b + k];
                            const double b_nu_rho = beta_ab[j * n_ao_b + k];
                            v1 += s_mu_rho * b_nu_rho + b_mu_rho * s_nu_rho;
                            v2 += s_mu_rho * s_nu_rho
                                  * (h_mu_b + h_nu_b - 2.0 * h_rho_a);
                        }
                        const double vort =
                            w * (-0.5 * F1 * v1 + 0.125 * F2 * v2);
                        H(mu, nu) += vort;
                        if (nu != mu) H(nu, mu) += vort;
                    }
                }
            }

            // (4) Three-center ORT corrections to the (a, b-image) block
            // (eq 9 of Dral 2016), C-sum over the union WSSC(MN).
            const double G1 = 0.5 * (ed_a->G1 + ed_b->G1);  // eq 11
            const double G2 = 0.5 * (ed_a->G2 + ed_b->G2);  // eq 12
            if (G1 != 0.0 || G2 != 0.0) {
                const auto union_images = build_ws_union(
                    topology, a, b, image.image_shell_label, image.disp);
                std::map<int, int> ws_multiplicity_a, ws_multiplicity_b;
                for (const auto& rec : topology.cells[a])
                    ++ws_multiplicity_a[rec.origin];
                for (const auto& rec : topology.cells[b])
                    ++ws_multiplicity_b[rec.origin];

                for (const auto& entry : union_images) {
                    const int c = entry.first.first;
                    const auto& cimg = entry.second;
                    // The orbital pair lives on the real atom a and the
                    // b image: those two centers are not third centers
                    // (the molecular c != a, b rule).
                    if (c == a && entry.first.second == ImageShellLabel{0, 0, 0})
                        continue;
                    if (c == b && entry.first.second == image.image_shell_label)
                        continue;
                    const auto* ed_c = params.omx_data(atoms[c].Z);
                    if (!ed_c) continue;
                    const double x_C = center_weight(
                        weighting, c, cimg,
                        ws_multiplicity_a, ws_multiplicity_b);
                    if (x_C <= 0.0) continue;
                    const double total_weight = w * x_C;

                    const int n_c = valence_n_sp(atoms[c].Z);
                    const double zeta_c =
                        (ed_c->zeta > 0.0) ? ed_c->zeta : 1.0;
                    const auto& idx_c = aos_on[c];
                    const int n_ao_c = static_cast<int>(idx_c.size());

                    const Eigen::Vector3d d1_work = -cimg.disp_from_a;
                    const Eigen::Vector3d d2_work =
                        cimg.disp_from_a - image.disp;
                    const double R1 = d1_work.norm();
                    const double R2 = d2_work.norm();
                    if (R1 < 1e-12 || R2 < 1e-12) continue;
                    // (min, max) pair order, as in the record pair above.
                    const int lo_ac = std::min(a, c);
                    const int hi_ac = std::max(a, c);
                    const auto* ed_lo_ac = (lo_ac == a) ? ed_a : ed_c;
                    const auto* ed_hi_ac = (hi_ac == a) ? ed_a : ed_c;
                    const int n_lo_ac = (lo_ac == a) ? n_a : n_c;
                    const int n_hi_ac = (hi_ac == a) ? n_a : n_c;
                    const double zeta_lo_ac = (lo_ac == a) ? zeta_a : zeta_c;
                    const double zeta_hi_ac = (hi_ac == a) ? zeta_a : zeta_c;
                    const OmxPairIntegrals pi_ac = compute_pair_integrals(
                        ed_lo_ac, n_lo_ac, zeta_lo_ac,
                        ed_hi_ac, n_hi_ac, zeta_hi_ac, R1);
                    const int lo_bc = std::min(b, c);
                    const int hi_bc = std::max(b, c);
                    const auto* ed_lo_bc = (lo_bc == b) ? ed_b : ed_c;
                    const auto* ed_hi_bc = (hi_bc == b) ? ed_b : ed_c;
                    const int n_lo_bc = (lo_bc == b) ? n_b : n_c;
                    const int n_hi_bc = (hi_bc == b) ? n_b : n_c;
                    const double zeta_lo_bc = (lo_bc == b) ? zeta_b : zeta_c;
                    const double zeta_hi_bc = (hi_bc == b) ? zeta_b : zeta_c;
                    const OmxPairIntegrals pi_bc = compute_pair_integrals(
                        ed_lo_bc, n_lo_bc, zeta_lo_bc,
                        ed_hi_bc, n_hi_bc, zeta_hi_bc, R2);

                    // Per-AO overlaps and resonance between the C image and
                    // each side of the orbital pair, plus the eq-10 local
                    // energies.
                    std::vector<double> s_mu(n_ao_a * n_ao_c, 0.0);
                    std::vector<double> b_mu(n_ao_a * n_ao_c, 0.0);
                    std::vector<double> h_mu_c(n_ao_a, 0.0);
                    for (int i = 0; i < n_ao_a; ++i) {
                        const int mu = idx_a[i];
                        h_mu_c[i] = h_loc_vs(
                            U_s[mu], static_cast<double>(core_charge[c]),
                            pi_ac.f_ko, vs_of(pi_ac, a, lo_ac, ao_type[mu]));
                        for (int k = 0; k < n_ao_c; ++k) {
                            const int rho = idx_c[k];
                            s_mu[i * n_ao_c + k] = nddo_sto_overlap(
                                n_a, ao_type[mu], zeta_a, zeta_a,
                                n_c, ao_type[rho], zeta_c, zeta_c,
                                d1_work[0], d1_work[1], d1_work[2], R1);
                            b_mu[i * n_ao_c + k] = omx_resonance_integral(
                                ao_type[mu], ao_type[rho], ao_Z[mu], ao_Z[rho],
                                R1, d1_work[0], d1_work[1], d1_work[2], params);
                        }
                    }
                    std::vector<double> s_lam(n_ao_b * n_ao_c, 0.0);
                    std::vector<double> b_lam(n_ao_b * n_ao_c, 0.0);
                    std::vector<double> h_lam_c(n_ao_b, 0.0);
                    std::vector<double> h_rho_a(n_ao_c, 0.0);
                    std::vector<double> h_rho_b(n_ao_c, 0.0);
                    for (int j = 0; j < n_ao_b; ++j) {
                        const int lam = idx_b[j];
                        h_lam_c[j] = h_loc_vs(
                            U_s[lam], static_cast<double>(core_charge[c]),
                            pi_bc.f_ko, vs_of(pi_bc, b, lo_bc, ao_type[lam]));
                        for (int k = 0; k < n_ao_c; ++k) {
                            const int rho = idx_c[k];
                            // Both legs carry the matrix-transpose
                            // convention of the molecular driver, where
                            // S(rho, lam) and beta(rho, lam) are evaluated
                            // with d = pos[rho] - pos[lam] = +d2.  The
                            // value S(lam, rho; -d2) equals that S(rho,
                            // lam; d2) element (S(lam, rho; pos[lam] -
                            // pos[rho]) = S(rho, lam; pos[rho] - pos[lam])),
                            // and likewise for the resonance matrix beta,
                            // which is elementwise symmetric in (mu, nu)
                            // with the element displacement.
                            s_lam[j * n_ao_c + k] = nddo_sto_overlap(
                                n_b, ao_type[lam], zeta_b, zeta_b,
                                n_c, ao_type[rho], zeta_c, zeta_c,
                                -d2_work[0], -d2_work[1], -d2_work[2], R2);
                            b_lam[j * n_ao_c + k] = omx_resonance_integral(
                                ao_type[lam], ao_type[rho], ao_Z[lam], ao_Z[rho],
                                R2, -d2_work[0], -d2_work[1], -d2_work[2], params);
                        }
                    }
                    for (int k = 0; k < n_ao_c; ++k) {
                        const int rho = idx_c[k];
                        h_rho_a[k] = h_loc_vs(
                            U_s[rho], static_cast<double>(core_charge[a]),
                            pi_ac.f_ko, vs_of(pi_ac, c, lo_ac, ao_type[rho]));
                        h_rho_b[k] = h_loc_vs(
                            U_s[rho], static_cast<double>(core_charge[b]),
                            pi_bc.f_ko, vs_of(pi_bc, c, lo_bc, ao_type[rho]));
                    }

                    for (int i = 0; i < n_ao_a; ++i) {
                        const int mu = idx_a[i];
                        for (int j = 0; j < n_ao_b; ++j) {
                            const int lam = idx_b[j];
                            double v1 = 0.0, v2 = 0.0;
                            for (int k = 0; k < n_ao_c; ++k) {
                                const double s_mu_rho = s_mu[i * n_ao_c + k];
                                const double s_lam_rho = s_lam[j * n_ao_c + k];
                                const double b_mu_rho = b_mu[i * n_ao_c + k];
                                const double b_lam_rho = b_lam[j * n_ao_c + k];
                                v1 += s_mu_rho * b_lam_rho
                                      + b_mu_rho * s_lam_rho;
                                v2 += s_mu_rho * s_lam_rho
                                      * (h_mu_c[i] + h_lam_c[j]
                                         - h_rho_a[k] - h_rho_b[k]);
                            }
                            const double vort = total_weight
                                * (-0.5 * G1 * v1 + 0.125 * G2 * v2);
                            H(mu, lam) += vort;  // directed; symmetrized below
                        }
                    }
                }
            }

            // (5) Semiempirical ECP, OM2/OM3 (eqs 60/63 of Weber 2000): the
            // core of the b image corrects the one-center block of A.  The
            // pair (mu, nu) is one-center (x_muNu = 1), so eq 13 gives the
            // plain two-center weight and eq 10 its half.
            if (ed_b->zeta_alpha > 0.0) {
                const double F_aa = ed_b->F_alpha_alpha * EV2HA;
                const double w_ecp =
                    (weighting == OMxThreeCenterWeighting::janetzko_eq10)
                        ? 0.5 * w
                        : w;
                std::vector<double> S_va(n_ao_a, 0.0);
                std::vector<double> G_va(n_ao_a, 0.0);
                for (int i = 0; i < n_ao_a; ++i) {
                    const int mu = idx_a[i];
                    const int t_mu = ao_type[mu];
                    S_va[i] = nddo_sto_overlap(
                        n_a, t_mu, zeta_a, zeta_a,
                        1, 0, ed_b->zeta_alpha, ed_b->zeta_alpha,
                        dx, dy, dz, R);
                    // G_mu alpha (eq 63): the valence orbital's plain
                    // (non-X-H) beta/alpha on a, the core orbital's
                    // beta_alpha/alpha_alpha on b, with the same
                    // rotation/sign convention as the valence resonance.
                    if (t_mu == 0) {
                        G_va[i] = omx_beta_radial(
                                      ed_a->beta_s, ed_a->alpha_s,
                                      ed_b->beta_alpha, ed_b->alpha_alpha, R)
                                  * EV2HA;
                    } else {
                        const double c =
                            slater_dir_cos(t_mu - 1, dx, dy, dz, R);
                        G_va[i] = -c
                                  * omx_beta_radial(
                                        ed_a->beta_p, ed_a->alpha_p,
                                        ed_b->beta_alpha, ed_b->alpha_alpha, R)
                                  * EV2HA;
                    }
                }
                // V^ECP_mu nu,B = -(S_ma G_an + G_ma S_an) - S_ma S_an F_aa
                // (eq 60).
                for (int i = 0; i < n_ao_a; ++i) {
                    for (int j = i; j < n_ao_a; ++j) {
                        const double v = -(S_va[i] * G_va[j] + G_va[i] * S_va[j])
                                         - S_va[i] * S_va[j] * F_aa;
                        H(idx_a[i], idx_a[j]) += w_ecp * v;
                        if (j != i) H(idx_a[j], idx_a[i]) += w_ecp * v;
                    }
                }
            }

            // (6) Two-center Coulomb/exchange integrals (eq 18, channel
            // resolved), directed accumulation; symmetrized below.  The
            // channel weights are assigned by atom-index side (lo/hi) and
            // the gee block is in (lo, hi) order, so both directed records
            // of a pair produce the bitwise-identical molecular value.
            for (int mu : idx_a) {
                for (int nu : idx_b) {
                    const int t_lo_ao = (a == lo) ? ao_type[mu] : ao_type[nu];
                    const int t_hi_ao = (a == lo) ? ao_type[nu] : ao_type[mu];
                    const auto cw_lo = ao_channel_weights(t_lo_ao, dx, dy, dz, R);
                    const auto cw_hi = ao_channel_weights(t_hi_ao, dx, dy, dz, R);
                    double g = 0.0;
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            g += cw_lo.w[i] * cw_hi.w[j] * pi.gee[i][j];
                    Gamma_ee(mu, nu) += w * pi.f_ko * g;
                }
            }

            // (7) Core-core repulsion (eq 20).
            E_core += 0.5 * w * pi.f_ko
                      * static_cast<double>(core_charge[a])
                      * static_cast<double>(core_charge[b]) / R;
        }
    }

    // Check the directed result before cleanup: unconditional symmetrization
    // must not hide a missing reverse image or an image-weighting error.
    if (detail::max_abs(H - H.transpose()) > HERMITICITY_TOLERANCE
        || detail::max_abs(Gamma_ee - Gamma_ee.transpose())
            > HERMITICITY_TOLERANCE) {
        throw std::runtime_error(
            "OMx-SECCM reverse-image assembly is not Hermitian");
    }

    // Retain the numerical cleanup before the self-adjoint eigensolver.
    H = (0.5 * (H + H.transpose())).eval();
    Gamma_ee = (0.5 * (Gamma_ee + Gamma_ee.transpose())).eval();

    // =====================================================================
    // SCF: the molecular run_omx_v2 loop on the WS-weighted Hamiltonian.
    // =====================================================================
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::VectorXd eps;
    Eigen::MatrixXd C;
    DiisState diis;
    double e_last = 0.0;
    OMxSECCMResult result;
    result.three_center_weighting =
        omx_three_center_weighting_name(weighting);
    result.truncated_electrostatics_acknowledged =
        truncated_electrostatics_required
        && opts.allow_truncated_electrostatics;

    const int damping_start = std::min(64, std::max(2, opts.max_iter / 2));
    const int diis_restart =
        std::min(opts.max_iter, damping_start + 64);

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        Eigen::MatrixXd F = H;

        // ---- One-center two-electron contributions (Pople NDDO) ----
        for (int a = 0; a < n_atoms; ++a) {
            const auto& idx = aos_on[a];
            const int n_ao_a = static_cast<int>(idx.size());
            const int Z_a = atoms[a].Z;
            const auto* ed = params.omx_data(Z_a);
            if (!ed) continue;
            const double G_ss = ed->gss * EV2HA;
            const double G_pp = ed->gpp * EV2HA;
            const double G_sp = ed->gsp * EV2HA;
            const double G_p2 = ed->gp2 * EV2HA;
            const double H_sp = ed->hsp * EV2HA;

            for (int i = 0; i < n_ao_a; ++i) {
                const int mu = idx[i], t_mu = ao_type[mu];

                // Coulomb: sum_nu P_nunu (mu mu | nu nu)
                for (int j = 0; j < n_ao_a; ++j) {
                    const int nu = idx[j], t_nu = ao_type[nu];
                    double coul = 0.0;
                    if (t_mu == 0 && t_nu == 0) coul = G_ss;
                    else if (t_mu == 0 && t_nu > 0) coul = G_sp;
                    else if (t_mu > 0 && t_nu == 0) coul = G_sp;
                    else if (t_mu > 0 && t_nu > 0 && t_mu == t_nu) coul = G_pp;
                    else coul = G_p2;
                    F(mu, mu) += D(nu, nu) * coul;
                }

                // Exchange: -1/2 sum_nu P_nu nu (mu nu | mu nu)
                for (int j = 0; j < n_ao_a; ++j) {
                    const int nu = idx[j], t_nu = ao_type[nu];
                    double exch = 0.0;
                    if (mu == nu) {
                        exch = (t_mu == 0) ? G_ss : G_pp;
                    } else if (t_mu == 0 && t_nu > 0) {
                        exch = H_sp;
                    } else if (t_mu > 0 && t_nu == 0) {
                        exch = H_sp;
                    } else {
                        exch = 0.5 * (G_pp - G_p2);
                    }
                    F(mu, mu) -= 0.5 * D(nu, nu) * exch;
                }
            }

            // Off-diagonal one-center
            for (int i = 0; i < n_ao_a; ++i) {
                for (int j = i + 1; j < n_ao_a; ++j) {
                    const int mu = idx[i], nu = idx[j];
                    const int t_mu = ao_type[mu], t_nu = ao_type[nu];
                    double exch = 0.0, coul = 0.0;
                    if (t_mu == 0 && t_nu > 0) {
                        exch = H_sp;
                        coul = G_sp;
                    } else if (t_mu > 0 && t_nu == 0) {
                        exch = H_sp;
                        coul = G_sp;
                    } else {
                        exch = 0.5 * (G_pp - G_p2);
                        coul = G_p2;
                    }
                    const double f_off =
                        0.5 * D(mu, nu) * (3.0 * exch - coul);
                    F(mu, nu) += f_off;
                    F(nu, mu) += f_off;
                }
            }
        }

        // ---- Two-center Coulomb (eq 18) ----
        for (int mu = 0; mu < n_basis; ++mu) {
            double v = 0.0;
            for (int nu = 0; nu < n_basis; ++nu) {
                if (ao_atom[nu] == ao_atom[mu]) continue;
                v += D(nu, nu) * Gamma_ee(mu, nu);
            }
            F(mu, mu) += v;
        }

        // ---- Two-center exchange: -1/2 P_mu nu Gamma_mu nu ----
        for (int mu = 0; mu < n_basis; ++mu) {
            const int a_mu = ao_atom[mu];
            for (int nu = mu + 1; nu < n_basis; ++nu) {
                if (ao_atom[nu] == a_mu) continue;
                const double ex = -0.5 * D(mu, nu) * Gamma_ee(mu, nu);
                F(mu, nu) += ex;
                F(nu, mu) += ex;
            }
        }

        // ---- DIIS residual: [F, P] (orthogonal basis) ----
        Eigen::MatrixXd err = F * D - D * F;
        const double res = err.cwiseAbs().maxCoeff();

        // ---- Energy at the current density: E = 1/2 Tr[P(H + F)] + E_core ----
        double E_total = 0.0;
        for (int mu = 0; mu < n_basis; ++mu) {
            E_total += D(mu, mu) * (H(mu, mu) + F(mu, mu));
            for (int nu = mu + 1; nu < n_basis; ++nu)
                E_total += 2.0 * D(mu, nu) * (H(mu, nu) + F(mu, nu));
        }
        E_total = 0.5 * E_total + E_core;

        const bool converged =
            (iter > 1 && res < opts.conv_tol
             && std::abs(E_total - e_last) < opts.conv_tol);
        e_last = E_total;

        if (converged || iter == opts.max_iter) {
            if (converged) {
                const double gap = eps(n_occ) - eps(n_occ - 1);
                if (!std::isfinite(gap) || gap <= gap_tolerance) {
                    throw std::runtime_error(
                        "OMx-SECCM requires a positive finite-torus "
                        "HOMO-LUMO gap");
                }
                result.homo_lumo_gap = gap;
            }

            const double normalization = static_cast<double>(group_order);
            result.energy = E_total / normalization;
            result.e_electronic = (E_total - E_core) / normalization;
            result.e_core = E_core / normalization;
            result.total_cyclic_energy = E_total;
            result.cyclic_core_energy = E_core;
            result.mo_energies = eps;
            result.mo_coeffs = C;
            result.density = D;
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_iter = iter;
            result.group_order = group_order;
            result.n_records = n_records;
            result.converged = converged;
            return result;
        }

        const bool use_damping =
            iter >= damping_start && iter < diis_restart;
        if (iter == damping_start || iter == diis_restart) diis.clear();

        // Skip the zero-density first iterate (zero commutator residual).
        if (iter > 1 && !use_damping) diis.push(F, err);
        Eigen::MatrixXd F_eff = use_damping ? F : diis.extrapolate(F);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_eff);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("OMx-SECCM: diagonalisation failed");
        eps = solver.eigenvalues();
        C = solver.eigenvectors();
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        Eigen::MatrixXd D_new = 2.0 * C_occ * C_occ.transpose();
        D = use_damping ? 0.5 * D + 0.5 * D_new : D_new;
    }

    result.n_iter = opts.max_iter;
    result.density = D;
    result.converged = false;
    return result;
}

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
