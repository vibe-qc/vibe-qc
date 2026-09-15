// OMx Fock builder — proper OM1/OM2/OM3 Hamiltonian implementation.
//
// The OMx methods extend the MNDO-type NDDO model with explicit
// orthogonalization corrections (the "ORT" pseudopotential terms) and, for
// OM2/OM3, a semiempirical effective core potential (ECP).  The secular
// equation is solved directly in the (assumed-orthogonal) NDDO basis,
//   F C = C ε                       (eq 2 of Dral 2016),
// i.e. there is NO Löwdin S^{-1/2} transformation of the Fock matrix here:
// the overlap matrix S enters only through the explicit correction terms.
// (Transforming a ZDO-parametrized Fock by the real S^{-1/2} amplifies the
// parametrized diagonal by 1/λ_min(S) and over-bound H₂O by ~4×; that bug —
// plus a sign error and a double eV→Ha conversion in the old VORT term, and a
// missing √R / summed-α in the resonance — is why this file was rewritten
// against the published equations on 2026-06-10.)
//
// References (equation numbers used in the comments below):
//   Dral, Wu, Spörkel, Koslowski, Weber, Steiger, Scholten & Thiel,
//     J. Chem. Theory Comput. 12, 1082 (2016), doi:10.1021/acs.jctc.5b01046
//     — "Dral 2016", the definitive OMx formalism + parameter paper.
//   Weber & Thiel, Theor. Chem. Acc. 103, 495 (2000),
//     doi:10.1007/s002149900083 — "Weber 2000", the ORT/ECP derivation (OM2).
//   Kolb & Thiel, J. Comput. Chem. 14, 775 (1993), doi:10.1002/jcc.540140704
//     — OM1.
//   Scholten, PhD thesis, Universität Düsseldorf (2003) — OM3.
//
// Implementation validated against published reference values: the resonance
// integrals and three-centre ORT corrections reproduce the OM2 H₃⁻ matrix
// elements of Weber 2000 Table 3 (β_AC = −2.786 eV, β_AB = −0.710 eV,
// V_AB,C = 0.610 eV, V_AC,B = 0.114 eV at 180°/1.1 Å) — pinned in
// tests/test_semiempirical.py::TestOMxV2AgainstWeber2000.
//
// Two-centre Coulomb-class integrals follow the published Klopman–Ohno-scaled
// analytic scheme (Dral 2016 eqs 13–14 and 18–20) with STO densities standing
// in for the scaled Gaussian basis:
//   f_KO(A,B) = γ^K_AB / (s_A s_A, s_B s_B)^a                     (eq 19)
//   (μμ,νν)^s = f_KO · (μμ,νν)^a            [σ/π-resolved c2int]  (eq 18)
//   V^s+V^PI  = −Z_B^core f_KO ⟨μ|1/r_B|μ⟩  [σ/π-resolved v2int]  (eqs 13–14)
//   E^core_AB = f_KO Z_A^core Z_B^core / R                        (eq 20)
// where γ^K is the Klopman–Ohno ss monopole from the one-centre Gss.  This
// keeps the MNDO one-centre limit, restores the analytic penetration
// behaviour at bonding distances, and (unlike MNDO/PM6) needs NO empirical
// exponential core-repulsion terms — the Pauli wall is carried by the
// explicit ORT/ECP corrections, the penetration balance by f_KO.
//
// Documented approximations relative to the published OMx methods (see
// the semiempirical PES hardening pass):
//   * STO radial densities (MSINDO s2int/c2int/v2int kernels) replace the
//     ζ-scaled Gaussian STO-3G/ECP-3G basis of Dral 2016 §2.3.
//   * (μμ|νν) cross-channel (pσpπ|pσpπ)-type quadrupole couplings and the
//     two-centre exchange integrals (μν|μν) keep the spherical Klopman–Ohno
//     monopole value.
//   * OM1's core–valence ECP (analytic integrals, Klopman–Ohno scaled) is
//     omitted — OM1 publishes no semiempirical ECP parameters (Dral 2016
//     Table 1).  OM2/OM3 use the published ECP formula (Weber 2000 eq 60).

#include "vibeqc/semiempirical/methods/nddo/omx_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/gradient_fd_checks.hpp"
#include "vibeqc/semiempirical/methods/nddo/slater_overlap.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {
namespace semiempirical {
namespace nddo {

namespace {

constexpr double EV2HA = 1.0 / 27.2114;

void validate_omx_scf_controls(
    int max_iter,
    double conv_tol,
    const char* driver) {
    if (max_iter < 1 || !std::isfinite(conv_tol) || conv_tol <= 0.0) {
        throw std::invalid_argument(
            std::string(driver)
            + ": max_iter and conv_tol must be positive and finite");
    }
}

// Valence electron count (same convention as pm6_fock.cpp).
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

// ---------------------------------------------------------------------------
// Resonance-parameter selection: (β, α) for an orbital type on one atom,
// honouring the X–H pair-specific parameters of Dral 2016 Tables 1–3 when
// the partner atom is hydrogen (and the parameters are present).
// ---------------------------------------------------------------------------
static std::pair<double, double>
pick_beta_alpha(int t,                 // orbital type: 0=s, >0 = p
                bool partner_is_h,     // X–H pair-specific set applies?
                const OMxElementData& ed) {
    if (t == 0) {
        if (partner_is_h && ed.beta_s_xh != 0.0)
            return {ed.beta_s_xh, ed.alpha_s_xh};
        return {ed.beta_s, ed.alpha_s};
    }
    if (partner_is_h && ed.beta_p_xh != 0.0)
        return {ed.beta_p_xh, ed.alpha_p_xh};
    return {ed.beta_p, ed.alpha_p};
}

// Radial OMx resonance kernel, eq 17 of Dral 2016 (= eq 27 of Weber 2000):
//   β_μλ^loc = ½ (β_μ^A + β_λ^B) · √R_AB · exp[−(α_μ^A + α_λ^B) · R_AB²]
// β in eV·bohr^{-1/2} (hence the explicit √R), α in bohr^{-2}; the α's are
// SUMMED, not averaged.  Returns eV.
// Published target: OM2 H–H at 1.1 Å gives β_ss = −2.786 eV and at 2.2 Å
// −0.710 eV (Weber 2000 Table 3, H₃⁻ rows 2/6 at 180°).
static inline double omx_beta_radial(double beta_a, double alpha_a,
                                     double beta_b, double alpha_b,
                                     double R) {
    return 0.5 * (beta_a + beta_b) * std::sqrt(R)
           * std::exp(-(alpha_a + alpha_b) * R * R);
}

}  // anonymous namespace

// ===========================================================================
// OMx resonance integral in the molecular frame (Ha).
//
// Eq 17 of Dral 2016 defines the LOCAL diatomic-frame integrals (ss, spσ,
// pσpσ, pπpπ) with "a suitable phase factor implied"; the molecular-frame
// elements follow by the standard Slater–Koster rotation.  The phase is fixed
// by the requirement that the resonance integral and the overlap integral
// carry OPPOSITE signs element-by-element (Weber 2000 §3.2.1: "overlap and
// resonance integrals normally have opposite signs"; their products in the
// ORT terms below must be negative for the correction to be repulsive).  The
// rotation below therefore mirrors nddo_sto_overlap() exactly — same
// direction cosines, same s–p convention minus — with the σ channels
// sign-flipped relative to the σ overlaps (s2int's pσ/spσ radial values are
// negative where the β kernel is negative, so the σ channels need an explicit
// flip to land opposite to S; the ss and pπpπ channels are already opposite).
// ===========================================================================

double omx_resonance_integral(
    int t_mu, int t_nu,
    int Z_mu, int Z_nu,
    double R,
    double dx, double dy, double dz,
    const OMxParameterSet& params) {

    if (R < 1e-12) return 0.0;

    const auto* ed_mu = params.omx_data(Z_mu);
    const auto* ed_nu = params.omx_data(Z_nu);
    if (!ed_mu || !ed_nu) return 0.0;

    // X–H pair-specific parameters live on the heavy atom of an X–H pair.
    const bool xh_mu = (Z_nu == 1) && (Z_mu != 1);
    const bool xh_nu = (Z_mu == 1) && (Z_nu != 1);

    if (t_mu == 0 && t_nu == 0) {
        // s–s channel: opposite to S_ss (>0) already — no flip.
        auto [b_mu, a_mu] = pick_beta_alpha(0, xh_mu, *ed_mu);
        auto [b_nu, a_nu] = pick_beta_alpha(0, xh_nu, *ed_nu);
        return omx_beta_radial(b_mu, a_mu, b_nu, a_nu, R) * EV2HA;
    }

    if (t_mu == 0 || t_nu == 0) {
        // s–p channel.  s2int's local-frame value is ANTISYMMETRIC in the
        // argument order (s2int(s,pσ) = −s2int(pσ,s); verified numerically:
        // ∓0.539 for 1s/2p at ζ=1.47/1.42, R=2.0), and the overlap engine
        // applies −c to both orders, so S(s_A,p_B) = +c·|f| while
        // S(p_A,s_B) = −c·|f|.  β must stay elementwise OPPOSITE to S in
        // both orders — getting one order wrong flips the inter-atomic
        // S·β ORT products from repulsive to attractive for every p–s
        // pairing and inverted the ethane rotation barrier to −28 kcal/mol
        // (published OM2: +2.8, SI Table S4 of Dral 2016).
        const bool s_is_mu = (t_mu == 0);
        const int  d  = (s_is_mu ? t_nu : t_mu) - 1;
        const double c = slater_dir_cos(d, dx, dy, dz, R);
        auto [b_s, a_s] = pick_beta_alpha(0, s_is_mu ? xh_mu : xh_nu,
                                          s_is_mu ? *ed_mu : *ed_nu);
        auto [b_p, a_p] = pick_beta_alpha(1, s_is_mu ? xh_nu : xh_mu,
                                          s_is_mu ? *ed_nu : *ed_mu);
        const double val = omx_beta_radial(b_s, a_s, b_p, a_p, R) * EV2HA;
        // (s_A, p_B): S = +c|f| → β = +c·val (val < 0).
        // (p_A, s_B): S = −c|f| → β = −c·val.
        return s_is_mu ? c * val : -c * val;
    }

    // p–p channels: σ via both direction cosines (flip), π without (no flip).
    auto [b_p_mu, a_p_mu] = pick_beta_alpha(1, xh_mu, *ed_mu);
    auto [b_p_nu, a_p_nu] = pick_beta_alpha(1, xh_nu, *ed_nu);
    const double val_sig = omx_beta_radial(b_p_mu, a_p_mu, b_p_nu, a_p_nu, R) * EV2HA;
    const double val_pi  = omx_beta_radial(ed_mu->beta_pi, ed_mu->alpha_pi,
                                           ed_nu->beta_pi, ed_nu->alpha_pi, R) * EV2HA;

    const int d_mu = t_mu - 1, d_nu = t_nu - 1;
    const double c_mu = slater_dir_cos(d_mu, dx, dy, dz, R);
    const double c_nu = slater_dir_cos(d_nu, dx, dy, dz, R);
    const double delta = (d_mu == d_nu) ? 1.0 : 0.0;

    // σ: opposite to S's c·c·s2int(pσpσ) (s2int σ value < 0) → flip sign.
    // π: opposite to S's (δ−c·c)·s2int(pπpπ) (>0) → keep val (<0).
    return -c_mu * c_nu * val_sig + (delta - c_mu * c_nu) * val_pi;
}

// ===========================================================================
// OMx overlap matrix.
//
// The OMx basis is a minimal valence set whose radial scale is the single
// per-element parameter ζ (Dral 2016 §2.3, Tables 1–3, "ζ (au)"); we
// represent each valence orbital as an STO with exponent ζ and the period's
// principal quantum number (the published ECP core exponents ζ_α ≈ Slater
// 1s values, e.g. ζ_α(O) = 7.59 vs Slater-rule 7.7, confirm ζ is an absolute
// exponent).  One-centre blocks are exactly orthonormal (identity).
// ===========================================================================

Eigen::MatrixXd build_omx_overlap(
    const Molecule& mol,
    const OMxParameterSet& params,
    const std::vector<int>& ao_Z,
    const std::vector<int>& ao_atom,
    const std::vector<int>& ao_type) {

    const auto& atoms = mol.atoms();
    int n_basis = static_cast<int>(ao_Z.size());
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n_basis, n_basis);

    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu], t_mu = ao_type[mu];
        for (int nu = mu + 1; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu], t_nu = ao_type[nu];
            if (a_mu == a_nu) continue;

            double dx = atoms[a_mu].xyz[0] - atoms[a_nu].xyz[0];
            double dy = atoms[a_mu].xyz[1] - atoms[a_nu].xyz[1];
            double dz = atoms[a_mu].xyz[2] - atoms[a_nu].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;

            const auto* ed_mu = params.omx_data(ao_Z[mu]);
            const auto* ed_nu = params.omx_data(ao_Z[nu]);
            double zeta_a = (ed_mu && ed_mu->zeta > 0.0) ? ed_mu->zeta : 1.0;
            double zeta_b = (ed_nu && ed_nu->zeta > 0.0) ? ed_nu->zeta : 1.0;
            int n_a = valence_n_sp(ao_Z[mu]), n_b = valence_n_sp(ao_Z[nu]);
            double S_val = nddo_sto_overlap(n_a, t_mu, zeta_a, zeta_a,
                                            n_b, t_nu, zeta_b, zeta_b,
                                            dx, dy, dz, R);

            S(mu, nu) = S_val;
            S(nu, mu) = S_val;
        }
    }

    return S;
}

namespace {

// Full inter-atomic resonance matrix β (Ha); one-centre blocks are zero
// (resonance is purely inter-atomic in NDDO).
Eigen::MatrixXd build_omx_resonance(
    const Molecule& mol,
    const OMxParameterSet& params,
    const std::vector<int>& ao_Z,
    const std::vector<int>& ao_atom,
    const std::vector<int>& ao_type) {

    const auto& atoms = mol.atoms();
    const int n_basis = static_cast<int>(ao_Z.size());
    Eigen::MatrixXd Beta = Eigen::MatrixXd::Zero(n_basis, n_basis);

    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        for (int nu = mu + 1; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            if (a_mu == a_nu) continue;
            double dx = atoms[a_mu].xyz[0] - atoms[a_nu].xyz[0];
            double dy = atoms[a_mu].xyz[1] - atoms[a_nu].xyz[1];
            double dz = atoms[a_mu].xyz[2] - atoms[a_nu].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;
            double b = omx_resonance_integral(ao_type[mu], ao_type[nu],
                                              ao_Z[mu], ao_Z[nu],
                                              R, dx, dy, dz, params);
            Beta(mu, nu) = b;
            Beta(nu, mu) = b;
        }
    }
    return Beta;
}

// Klopman–Ohno monopole γ_AB between two atoms (Ha), from the one-centre Gss:
//   γ^K_AB = 1 / sqrt(R² + η²),  η = ½(1/Gss_A + 1/Gss_B)
// — the MNDO ss limit the f_KO scaling is anchored to (Dral 2016 eq 19).
inline double omx_gamma_k(const OMxElementData& ed_a,
                          const OMxElementData& ed_b, double R) {
    double gss_a = ed_a.gss > 0.0 ? ed_a.gss * EV2HA : 0.367;
    double gss_b = ed_b.gss > 0.0 ? ed_b.gss * EV2HA : 0.367;
    double eta = 0.5 * (1.0 / gss_a + 1.0 / gss_b);
    return 1.0 / std::sqrt(R * R + eta * eta);
}

// σ/π channel weights of an AO's density relative to a pair axis:
// s → (1, 0, 0);  p_d → (0, c², 1−c²) with c the direction cosine.
// (The (pσpπ) cross density carries no monopole, so it drops out of the
// Coulomb-class integrals retained here.)
struct ChannelWeights {
    double w[3];  // [s, pσ, pπ]
};

inline ChannelWeights ao_channel_weights(int t, double dx, double dy,
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

// Per-geometry, density-independent context shared by the v2 SCF drivers.
struct OMxContext {
    std::vector<int> ao_Z, ao_atom, ao_type;
    int n_basis = 0;
    int n_atoms = 0;
    int n_val = 0;
    std::vector<double> U_s, G_ss, G_pp, G_sp, G_p2, H_sp, rho;
    std::vector<int> core_charge;
    Eigen::MatrixXd S;         // valence overlap (corrections only)
    Eigen::MatrixXd Gamma_ee;  // f_KO·(μμ|νν)^a, σ/π-resolved; 0 one-centre
    Eigen::MatrixXd Vat;       // n_basis × n_atoms: f_KO·⟨μ|1/r_X|μ⟩ (eq 14)
    Eigen::MatrixXd Vs_loc;    // n_basis × n_atoms: −Z_X f_KO (μμ, s_X s_X),
                               // p sph-averaged (eq 10 local energies)
    Eigen::MatrixXd H_core;
    double E_core = 0.0;       // Σ f_KO Z_A Z_B / R (eq 20)
};

// Forward declaration (defined below; needs OMxContext).
Eigen::MatrixXd build_omx_core_hamiltonian_v2(const Molecule& mol,
                                              const OMxParameterSet& params,
                                              const OMxContext& ctx);

OMxContext omx_setup(const Molecule& mol, const OMxParameterSet& params,
                     const char* who) {
    if (mol.atoms().empty())
        throw std::invalid_argument(std::string(who) + ": empty molecule");
    // Element coverage check.
    {
        std::vector<int> missing;
        for (const auto& atom : mol.atoms()) {
            if (!params.has_element(atom.Z)
                || params.omx_data(atom.Z) == nullptr) {
                missing.push_back(atom.Z);
            }
        }
        if (!missing.empty()) {
            std::string msg = std::string(who) + ": element(s) not in parameter set: ";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += "Z=" + std::to_string(missing[i]);
            }
            msg += ".  OMx currently supports H, C, N, O, F only.";
            throw std::invalid_argument(msg);
        }
    }

    const auto& atoms = mol.atoms();
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            double distance_squared = 0.0;
            for (int d = 0; d < 3; ++d) {
                const double delta = atoms[a].xyz[d] - atoms[b].xyz[d];
                distance_squared += delta * delta;
            }
            if (distance_squared < 1.0e-24) {
                throw std::invalid_argument(
                    std::string(who) + ": distinct atoms "
                    + std::to_string(a) + " and " + std::to_string(b)
                    + " are coincident");
            }
        }
    }
    OMxContext ctx;
    ctx.n_atoms = static_cast<int>(atoms.size());

    // AO basis: H/He carry a single 1s valence orbital; C–F get the sp set.
    for (int a = 0; a < ctx.n_atoms; ++a) {
        int Z = atoms[a].Z;
        const auto* ed = params.omx_data(Z);
        int n_ao = (Z <= 2) ? 1 : (ed ? ed->n_orbitals : 4);
        for (int i = 0; i < n_ao; ++i) {
            ctx.ao_Z.push_back(Z);
            ctx.ao_atom.push_back(a);
            ctx.ao_type.push_back(i);
        }
        ctx.n_basis += n_ao;
    }

    // NDDO is valence-only: fill Σ core charges − net charge.
    for (const auto& atom : atoms) ctx.n_val += omx_valence_electrons(atom.Z);
    ctx.n_val -= mol.charge();

    ctx.U_s.assign(ctx.n_basis, 0.0);
    ctx.G_ss.assign(ctx.n_basis, 0.0);
    ctx.G_pp.assign(ctx.n_basis, 0.0);
    ctx.G_sp.assign(ctx.n_basis, 0.0);
    ctx.G_p2.assign(ctx.n_basis, 0.0);
    ctx.H_sp.assign(ctx.n_basis, 0.0);
    ctx.rho.assign(ctx.n_basis, 0.0);
    for (int mu = 0; mu < ctx.n_basis; ++mu) {
        const auto* ed = params.omx_data(ctx.ao_Z[mu]);
        if (!ed) continue;
        bool is_s = (ctx.ao_type[mu] == 0);
        ctx.U_s[mu] = (is_s ? ed->uss : ed->upp) * EV2HA;
        ctx.G_ss[mu] = ed->gss * EV2HA;
        ctx.G_pp[mu] = ed->gpp * EV2HA;
        ctx.G_sp[mu] = ed->gsp * EV2HA;
        ctx.G_p2[mu] = ed->gp2 * EV2HA;
        ctx.H_sp[mu] = ed->hsp * EV2HA;
        ctx.rho[mu] = (ed->gss > 0.0) ? 1.0 / (ed->gss * EV2HA) : 2.72;
    }

    ctx.core_charge.resize(ctx.n_atoms);
    for (int a = 0; a < ctx.n_atoms; ++a)
        ctx.core_charge[a] = omx_valence_electrons(atoms[a].Z);

    ctx.S = build_omx_overlap(mol, params, ctx.ao_Z, ctx.ao_atom, ctx.ao_type);

    // ---- Klopman-scaled analytic monopole integrals (eqs 13–14, 18–20) ----
    // Channel-resolved STO Coulomb/attraction integrals per atom pair,
    // uniformly scaled by f_KO = γ^K_ss / (ss,ss)^a so the ss integral matches
    // the MNDO monopole at all distances (eq 19) and the one-centre limit is
    // preserved.
    ctx.Gamma_ee = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_basis);
    ctx.Vat = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_atoms);
    ctx.Vs_loc = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_atoms);
    ctx.E_core = 0.0;

    namespace msi = ::vibeqc::semiempirical::indo;

    for (int a = 0; a < ctx.n_atoms; ++a) {
        const auto* ed_a = params.omx_data(atoms[a].Z);
        if (!ed_a) continue;
        const double zeta_a = (ed_a->zeta > 0.0) ? ed_a->zeta : 1.0;
        const int n_a = valence_n_sp(atoms[a].Z);

        for (int b = a + 1; b < ctx.n_atoms; ++b) {
            const auto* ed_b = params.omx_data(atoms[b].Z);
            if (!ed_b) continue;
            const double zeta_b = (ed_b->zeta > 0.0) ? ed_b->zeta : 1.0;
            const int n_b = valence_n_sp(atoms[b].Z);

            const double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            const double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            const double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            const double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;

            // f_KO (eq 19): Klopman monopole over analytic (ss,ss).
            const double g_k = omx_gamma_k(*ed_a, *ed_b, R);
            const double g_a_ss = msi::c2int(n_a, 0, 0, zeta_a,
                                             n_b, 0, 0, zeta_b, R);
            const double f_ko = (g_a_ss > 1e-12) ? g_k / g_a_ss : 1.0;

            // Channel-resolved (μμ,νν)^a (eq 18): rows/cols = [s, pσ, pπ].
            double gee[3][3];
            const int l_of[3] = {0, 1, 1};
            const int m_of[3] = {0, 0, 1};
            const bool has_p_a = (n_a > 1), has_p_b = (n_b > 1);
            for (int ia = 0; ia < 3; ++ia) {
                for (int ib = 0; ib < 3; ++ib) {
                    if ((ia > 0 && !has_p_a) || (ib > 0 && !has_p_b)) {
                        gee[ia][ib] = g_a_ss;
                        continue;
                    }
                    gee[ia][ib] = msi::c2int(n_a, l_of[ia], m_of[ia], zeta_a,
                                             n_b, l_of[ib], m_of[ib], zeta_b, R);
                }
            }

            // Channel-resolved attraction ⟨μ|1/r_X|μ⟩ (eqs 13–14).
            double vat_a[3], vat_b[3];
            for (int i = 0; i < 3; ++i) {
                vat_a[i] = (i > 0 && !has_p_a)
                               ? msi::v2int(n_a, 0, 0, zeta_a, R)
                               : msi::v2int(n_a, l_of[i], m_of[i], zeta_a, R);
                vat_b[i] = (i > 0 && !has_p_b)
                               ? msi::v2int(n_b, 0, 0, zeta_b, R)
                               : msi::v2int(n_b, l_of[i], m_of[i], zeta_b, R);
            }

            // Spherically-averaged (μμ, s_X s_X) for the eq-10 local energies
            // ("average local matrix elements ... for the three p functions").
            const double vs_a_s = gee[0][0];
            const double vs_a_p = (gee[1][0] + 2.0 * gee[2][0]) / 3.0;
            const double vs_b_s = gee[0][0];
            const double vs_b_p = (gee[0][1] + 2.0 * gee[0][2]) / 3.0;

            // Scatter into per-AO arrays with σ/π direction weights.
            for (int mu = 0; mu < ctx.n_basis; ++mu) {
                if (ctx.ao_atom[mu] == a) {
                    const auto cw = ao_channel_weights(ctx.ao_type[mu],
                                                       dx, dy, dz, R);
                    double v = 0.0;
                    for (int i = 0; i < 3; ++i) v += cw.w[i] * vat_a[i];
                    ctx.Vat(mu, b) = f_ko * v;
                    ctx.Vs_loc(mu, b) = -static_cast<double>(ctx.core_charge[b])
                                        * f_ko
                                        * ((ctx.ao_type[mu] == 0) ? vs_a_s : vs_a_p);
                    for (int nu = 0; nu < ctx.n_basis; ++nu) {
                        if (ctx.ao_atom[nu] != b) continue;
                        const auto cwb = ao_channel_weights(ctx.ao_type[nu],
                                                            dx, dy, dz, R);
                        double g = 0.0;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j)
                                g += cw.w[i] * cwb.w[j] * gee[i][j];
                        ctx.Gamma_ee(mu, nu) = f_ko * g;
                        ctx.Gamma_ee(nu, mu) = f_ko * g;
                    }
                } else if (ctx.ao_atom[mu] == b) {
                    const auto cw = ao_channel_weights(ctx.ao_type[mu],
                                                       dx, dy, dz, R);
                    double v = 0.0;
                    for (int i = 0; i < 3; ++i) v += cw.w[i] * vat_b[i];
                    ctx.Vat(mu, a) = f_ko * v;
                    ctx.Vs_loc(mu, a) = -static_cast<double>(ctx.core_charge[a])
                                        * f_ko
                                        * ((ctx.ao_type[mu] == 0) ? vs_b_s : vs_b_p);
                }
            }

            // Core–core repulsion (eq 20).
            ctx.E_core += f_ko
                          * static_cast<double>(ctx.core_charge[a])
                          * static_cast<double>(ctx.core_charge[b]) / R;
        }
    }

    ctx.H_core = build_omx_core_hamiltonian_v2(mol, params, ctx);
    return ctx;
}

// ===========================================================================
// OMx core Hamiltonian (Ha).
//
// One-centre block (μ,ν on atom A), eq 6 of Dral 2016:
//   H_μν = U_μ δ_μν + Σ_{B≠A} [ (V^s+V^PI)_μν,B + V^ORT_μν,B + V^ECP_μν,B ]
// with the penetration-corrected core attraction (eqs 13–14)
//   (V^s+V^PI)_μμ,B = −Z_B^core f_KO ⟨μ|1/r_B|μ⟩   [ctx.Vat]
// (diagonal in the σ/π-monopole model used here).
//
// Two-centre block (μ on A, λ on B), eq 7 of Dral 2016:
//   H_μλ = β_μλ + Σ_{C≠A,B} V^ORT_μλ,C
//
// Two-centre ORT correction to the ONE-centre block, eq 8 of Dral 2016:
//   V^ORT_μν,B = −½ F1^A Σ_{ρ∈B} (S_μρ β_ρν + β_μρ S_ρν)
//                + ⅛ F2^A Σ_{ρ∈B} S_μρ S_ρν (H^loc_μμ,B + H^loc_νν,B − 2 H^loc_ρρ,A)
//
// Three-centre ORT correction to the TWO-centre block, eq 9 of Dral 2016
// (= eq 50 of Weber 2000; reproduces Weber Table 3 H₃⁻ reference values):
//   V^ORT_μλ,C = −½ G1^AB Σ_{ρ∈C} (S_μρ β_ρλ + β_μρ S_ρλ)
//                + ⅛ G2^AB Σ_{ρ∈C} S_μρ S_ρλ (H^loc_μμ,C + H^loc_λλ,C
//                                              − H^loc_ρρ,A − H^loc_ρρ,B)
//   with G1^AB = ½(G1^A + G1^B), G2^AB = ½(G2^A + G2^B)   (eqs 11–12).
//
// Local pair energies, eq 10 of Dral 2016:
//   H^loc_μμ,X = U_μ + V^s_μμ,X    [ctx.Vs_loc; p spherically averaged]
//
// Semiempirical ECP (OM2/OM3 only; first-row core = one 1s orbital α on B),
// eq 60 of Weber 2000 (= eq 15 of Dral 2016):
//   V^ECP_μν,B = −(S_μα G_αν + G_μα S_αν) − S_μα S_αν F_αα
// where S_μα is the valence–core overlap (core: 1s STO, exponent ζ_α) and
// G_μα follows the resonance form, eq 63 of Weber 2000 (= eq 16 of Dral 2016):
//   G_μα = ½(β_μ^A + β_α^B) √R_AB exp[−(α_μ^A + α_α^B) R_AB²]
// rotated to the molecular frame exactly like β and S (G must stay
// elementwise-opposite to S_μα for the ECP to be repulsive).
// ===========================================================================

Eigen::MatrixXd build_omx_core_hamiltonian_v2(const Molecule& mol,
                                              const OMxParameterSet& params,
                                              const OMxContext& ctx) {
    const auto& atoms = mol.atoms();
    const int n_atoms = ctx.n_atoms;
    const int n_basis = ctx.n_basis;
    const Eigen::MatrixXd& S = ctx.S;

    // AO indices per atom.
    std::vector<std::vector<int>> aos_on(n_atoms);
    for (int mu = 0; mu < n_basis; ++mu) aos_on[ctx.ao_atom[mu]].push_back(mu);

    // Local pair energy H^loc_μμ,X (eq 10).
    auto h_loc = [&](int mu, int X) -> double {
        return ctx.U_s[mu] + ctx.Vs_loc(mu, X);
    };

    // Inter-atomic resonance matrix β (Ha).
    const Eigen::MatrixXd Beta =
        build_omx_resonance(mol, params, ctx.ao_Z, ctx.ao_atom, ctx.ao_type);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_basis, n_basis);

    // ---- One-centre energies + penetration-corrected core attraction ----
    for (int mu = 0; mu < n_basis; ++mu) {
        H(mu, mu) += ctx.U_s[mu];
        const int a = ctx.ao_atom[mu];
        for (int b = 0; b < n_atoms; ++b) {
            if (b == a) continue;
            // (V^s + V^PI)_μμ,B = −Z_B^core f_KO ⟨μ|1/r_B|μ⟩  (eqs 13–14).
            H(mu, mu) -= static_cast<double>(ctx.core_charge[b]) * ctx.Vat(mu, b);
        }
    }

    // ---- Two-centre resonance β_μλ (eq 7, first term) ----
    H += Beta;

    // ---- Two-centre ORT corrections to the one-centre blocks (eq 8) ----
    for (int a = 0; a < n_atoms; ++a) {
        const auto* ed_a = params.omx_data(atoms[a].Z);
        if (!ed_a) continue;
        const double F1 = ed_a->F1;
        const double F2 = ed_a->F2;
        if (F1 == 0.0 && F2 == 0.0) continue;

        for (int mu : aos_on[a]) {
            for (int nu : aos_on[a]) {
                if (nu < mu) continue;  // symmetric — fill both at once
                double v1 = 0.0, v2 = 0.0;
                for (int b = 0; b < n_atoms; ++b) {
                    if (b == a) continue;
                    const double h_mu_b = h_loc(mu, b);
                    const double h_nu_b = h_loc(nu, b);
                    for (int rho : aos_on[b]) {
                        v1 += S(mu, rho) * Beta(rho, nu) + Beta(mu, rho) * S(rho, nu);
                        v2 += S(mu, rho) * S(rho, nu)
                              * (h_mu_b + h_nu_b - 2.0 * h_loc(rho, a));
                    }
                }
                const double vort = -0.5 * F1 * v1 + 0.125 * F2 * v2;
                H(mu, nu) += vort;
                if (nu != mu) H(nu, mu) += vort;
            }
        }
    }

    // ---- Three-centre ORT corrections to the two-centre blocks (eq 9) ----
    // OM1 publishes no G parameters (G1 = G2 = 0 → no-op); OM3 has G2 = 0.
    for (int a = 0; a < n_atoms; ++a) {
        const auto* ed_a = params.omx_data(atoms[a].Z);
        if (!ed_a) continue;
        for (int b = a + 1; b < n_atoms; ++b) {
            const auto* ed_b = params.omx_data(atoms[b].Z);
            if (!ed_b) continue;
            const double G1 = 0.5 * (ed_a->G1 + ed_b->G1);  // eq 11
            const double G2 = 0.5 * (ed_a->G2 + ed_b->G2);  // eq 12
            if (G1 == 0.0 && G2 == 0.0) continue;

            for (int mu : aos_on[a]) {
                for (int lam : aos_on[b]) {
                    double v1 = 0.0, v2 = 0.0;
                    for (int c = 0; c < n_atoms; ++c) {
                        if (c == a || c == b) continue;
                        const double h_mu_c = h_loc(mu, c);
                        const double h_lam_c = h_loc(lam, c);
                        for (int rho : aos_on[c]) {
                            v1 += S(mu, rho) * Beta(rho, lam)
                                  + Beta(mu, rho) * S(rho, lam);
                            v2 += S(mu, rho) * S(rho, lam)
                                  * (h_mu_c + h_lam_c
                                     - h_loc(rho, a) - h_loc(rho, b));
                        }
                    }
                    const double vort = -0.5 * G1 * v1 + 0.125 * G2 * v2;
                    H(mu, lam) += vort;
                    H(lam, mu) += vort;
                }
            }
        }
    }

    // ---- Semiempirical ECP, OM2/OM3 (eq 60 + 63 of Weber 2000) ----
    // For each atom B with a published core (ζ_α > 0): one 1s core orbital α.
    for (int b = 0; b < n_atoms; ++b) {
        const auto* ed_b = params.omx_data(atoms[b].Z);
        if (!ed_b || ed_b->zeta_alpha <= 0.0) continue;
        const double F_aa = ed_b->F_alpha_alpha * EV2HA;

        for (int a = 0; a < n_atoms; ++a) {
            if (a == b) continue;
            const auto* ed_a = params.omx_data(atoms[a].Z);
            if (!ed_a) continue;
            const double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            const double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            const double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            const double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;
            const double zeta_a = (ed_a->zeta > 0.0) ? ed_a->zeta : 1.0;
            const int n_a = valence_n_sp(atoms[a].Z);

            // Per-AO valence–core overlap S_μα and resonance-form G_μα for
            // every μ on A against the 1s core (n=1, s-type) on B.
            const auto& idx_a = aos_on[a];
            const int n_ao_a = static_cast<int>(idx_a.size());
            std::vector<double> S_va(n_ao_a), G_va(n_ao_a);
            for (int i = 0; i < n_ao_a; ++i) {
                const int mu = idx_a[i];
                const int t_mu = ctx.ao_type[mu];
                S_va[i] = nddo_sto_overlap(n_a, t_mu, zeta_a, zeta_a,
                                           1, 0, ed_b->zeta_alpha, ed_b->zeta_alpha,
                                           dx, dy, dz, R);
                // G_μα (eq 63): valence orbital's plain (non-X–H) β/α on A,
                // core orbital's β_α/α_α on B; same rotation/sign convention
                // as the valence resonance.  μ here is the FIRST index of a
                // (valence_A, core-s_B) pair, so the p case is the (p_A, s_B)
                // order: S_μα = −c·|f| → G must be −c·val to stay
                // elementwise opposite to S (see omx_resonance_integral).
                if (t_mu == 0) {
                    G_va[i] = omx_beta_radial(ed_a->beta_s, ed_a->alpha_s,
                                              ed_b->beta_alpha, ed_b->alpha_alpha,
                                              R) * EV2HA;
                } else {
                    const double c = slater_dir_cos(t_mu - 1, dx, dy, dz, R);
                    G_va[i] = -c * omx_beta_radial(ed_a->beta_p, ed_a->alpha_p,
                                                   ed_b->beta_alpha, ed_b->alpha_alpha,
                                                   R) * EV2HA;
                }
            }

            // V^ECP_μν,B = −(S_μα G_αν + G_μα S_αν) − S_μα S_αν F_αα  (eq 60)
            for (int i = 0; i < n_ao_a; ++i) {
                for (int j = i; j < n_ao_a; ++j) {
                    const double v = -(S_va[i] * G_va[j] + G_va[i] * S_va[j])
                                     - S_va[i] * S_va[j] * F_aa;
                    H(idx_a[i], idx_a[j]) += v;
                    if (j != i) H(idx_a[j], idx_a[i]) += v;
                }
            }
        }
    }

    return H;
}

// ===========================================================================
// Two-electron part of the closed-shell Fock matrix: F = H_core + G(D).
//
// One-centre: standard Pople NDDO closed-shell terms.
// Two-centre Coulomb: F_μμ += Σ_{ν∉A} P_νν · Γ_μν with the Klopman-scaled
//   σ/π-resolved (μμ,νν)^s of eq 18 [ctx.Gamma_ee].
// Two-centre exchange: F_μν += −½ P_μν Γ_μν — the NDDO monopole limit of
//   −½ Σ_{λ∈A,σ∈B} P_λσ (μλ,νσ): only the (μμ,νν) monopole products
//   survive, so the SAME Γ_μν integrals enter the exchange pattern.
// ===========================================================================

Eigen::MatrixXd build_omx_fock_v2(const OMxContext& ctx,
                                  const Eigen::MatrixXd& D) {
    const int n_atoms = ctx.n_atoms;
    const int n_basis = ctx.n_basis;

    Eigen::MatrixXd F = ctx.H_core;

    // ---- One-centre two-electron contributions ----
    for (int a = 0; a < n_atoms; ++a) {
        std::vector<int> idx;
        for (int mu = 0; mu < n_basis; ++mu)
            if (ctx.ao_atom[mu] == a) idx.push_back(mu);
        int n_ao_a = static_cast<int>(idx.size());

        for (int i = 0; i < n_ao_a; ++i) {
            int mu = idx[i], t_mu = ctx.ao_type[mu];

            // Coulomb: Σ_ν P_νν (μμ|νν)
            for (int j = 0; j < n_ao_a; ++j) {
                int nu = idx[j], t_nu = ctx.ao_type[nu];
                double coul = 0.0;
                if (t_mu == 0 && t_nu == 0) coul = ctx.G_ss[mu];
                else if (t_mu == 0 && t_nu > 0) coul = ctx.G_sp[mu];
                else if (t_mu > 0 && t_nu == 0) coul = ctx.G_sp[nu];
                else if (t_mu > 0 && t_nu > 0 && t_mu == t_nu) coul = ctx.G_pp[mu];
                else coul = ctx.G_p2[mu];
                F(mu, mu) += D(nu, nu) * coul;
            }

            // Exchange: −½ Σ_ν P_νν (μν|μν)
            for (int j = 0; j < n_ao_a; ++j) {
                int nu = idx[j], t_nu = ctx.ao_type[nu];
                double exch = 0.0;
                if (mu == nu) {
                    exch = (t_mu == 0) ? ctx.G_ss[mu] : ctx.G_pp[mu];
                } else if (t_mu == 0 && t_nu > 0) {
                    exch = ctx.H_sp[mu];
                } else if (t_mu > 0 && t_nu == 0) {
                    exch = ctx.H_sp[nu];
                } else {
                    exch = 0.5 * (ctx.G_pp[mu] - ctx.G_p2[mu]);
                }
                F(mu, mu) -= 0.5 * D(nu, nu) * exch;
            }
        }

        // Off-diagonal one-centre
        for (int i = 0; i < n_ao_a; ++i) {
            for (int j = i + 1; j < n_ao_a; ++j) {
                int mu = idx[i], nu = idx[j];
                int t_mu = ctx.ao_type[mu], t_nu = ctx.ao_type[nu];
                double exch = 0.0, coul = 0.0;
                if (t_mu == 0 && t_nu > 0) {
                    exch = ctx.H_sp[mu];
                    coul = ctx.G_sp[mu];
                } else if (t_mu > 0 && t_nu == 0) {
                    exch = ctx.H_sp[nu];
                    coul = ctx.G_sp[nu];
                } else {
                    exch = 0.5 * (ctx.G_pp[mu] - ctx.G_p2[mu]);
                    coul = ctx.G_p2[mu];
                }
                double f_off = 0.5 * D(mu, nu) * (3.0 * exch - coul);
                F(mu, nu) += f_off;
                F(nu, mu) += f_off;
            }
        }
    }

    // ---- Two-centre Coulomb (eq 18, σ/π-resolved) ----
    for (int mu = 0; mu < n_basis; ++mu) {
        double v = 0.0;
        for (int nu = 0; nu < n_basis; ++nu) {
            if (ctx.ao_atom[nu] == ctx.ao_atom[mu]) continue;
            v += D(nu, nu) * ctx.Gamma_ee(mu, nu);
        }
        F(mu, mu) += v;
    }

    // ---- Two-centre exchange: −½ P_μν Γ_μν ----
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ctx.ao_atom[mu];
        for (int nu = mu + 1; nu < n_basis; ++nu) {
            int a_nu = ctx.ao_atom[nu];
            if (a_mu == a_nu) continue;
            double ex = -0.5 * D(mu, nu) * ctx.Gamma_ee(mu, nu);
            F(mu, nu) += ex;
            F(nu, mu) += ex;
        }
    }

    return F;
}

// Pulay DIIS extrapolation state (residual = [F, P] in the orthogonal NDDO
// basis, where S = I by construction).
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

    // Returns the extrapolated Fock (or the input if the system is singular).
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

}  // anonymous namespace

// ===========================================================================
// Closed-shell OMx SCF (v2).  Solves F C = C ε directly in the NDDO basis
// (eq 2 of Dral 2016) with Pulay DIIS; the overlap matrix never enters the
// secular equation.
// ===========================================================================

PM6Result run_omx_v2(
    const Molecule& mol,
    const OMxParameterSet& params,
    int max_iter,
    double conv_tol) {

    validate_omx_scf_controls(max_iter, conv_tol, "OMx_v2");
    if (mol.multiplicity() != 1)
        throw std::invalid_argument("OMx_v2: only closed-shell supported");

    OMxContext ctx = omx_setup(mol, params, "OMx_v2");
    if (ctx.n_val < 0 || ctx.n_val % 2 != 0
        || ctx.n_val > 2 * ctx.n_basis) {
        throw std::invalid_argument(
            "OMx_v2: electron count is incompatible with the closed-shell "
            "NDDO basis");
    }
    const int n_occ = ctx.n_val / 2;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_basis);
    Eigen::VectorXd eps;
    Eigen::MatrixXd C;
    DiisState diis;
    double e_last = 0.0;
    PM6Result result;

    // Pulay DIIS is fast for the ordinary OMx cases, but near-degenerate
    // closed-shell densities can enter a persistent occupation cycle.  Keep
    // an initial bounded part of the caller's iteration budget unchanged,
    // then discard the stale DIIS history and enter a damped fixed-point map.
    // Once that map has crossed the nonlinear occupation-cycle region, clear
    // its history again and restart DIIS to converge the slow linear tail.
    // These phases change only the path to the same
    // zero-temperature idempotent density; the converged energy expression is
    // unchanged.
    const int damping_start = std::min(64, std::max(2, max_iter / 2));
    const int diis_restart = std::min(max_iter, damping_start + 64);

    for (int iter = 1; iter <= max_iter; ++iter) {
        Eigen::MatrixXd F = build_omx_fock_v2(ctx, D);

        // DIIS residual: [F, P] (orthogonal basis → no S factors).
        Eigen::MatrixXd err = F * D - D * F;
        double res = err.cwiseAbs().maxCoeff();

        // Energy at the current density: E = ½ Tr[P(H + F)] + E_core.
        double E_total = 0.0;
        for (int mu = 0; mu < ctx.n_basis; ++mu) {
            E_total += D(mu, mu) * (ctx.H_core(mu, mu) + F(mu, mu));
            for (int nu = mu + 1; nu < ctx.n_basis; ++nu)
                E_total += 2.0 * D(mu, nu) * (ctx.H_core(mu, nu) + F(mu, nu));
        }
        E_total = 0.5 * E_total + ctx.E_core;

        const bool converged =
            (iter > 1 && res < conv_tol && std::abs(E_total - e_last) < conv_tol);
        e_last = E_total;

        if (converged || iter == max_iter) {
            result.energy = E_total;
            result.e_electronic = E_total - ctx.E_core;
            result.e_core = ctx.E_core;
            result.mo_energies = eps;
            result.mo_coeffs = C;
            result.density = D;
            result.n_basis = ctx.n_basis;
            result.n_occ = n_occ;
            result.n_iter = iter;
            result.converged = converged;
            return result;
        }

        const bool use_damping =
            iter >= damping_start && iter < diis_restart;
        if (iter == damping_start || iter == diis_restart) diis.clear();

        // Skip the zero-density first iterate: its commutator residual is
        // exactly zero, which poisons the DIIS equations.  The recovery phase
        // uses the physical Fock directly so old extrapolation coefficients
        // cannot keep the occupation cycle alive.
        if (iter > 1 && !use_damping) diis.push(F, err);
        Eigen::MatrixXd F_eff = use_damping ? F : diis.extrapolate(F);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_eff);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("OMx_v2: diagonalisation failed");
        eps = solver.eigenvalues();
        C = solver.eigenvectors();
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        Eigen::MatrixXd D_new = 2.0 * C_occ * C_occ.transpose();
        D = use_damping ? 0.5 * D + 0.5 * D_new : D_new;
    }

    result.n_iter = max_iter;
    result.converged = false;
    return result;
}

// ===========================================================================
// Unrestricted OMx SCF (v2) — UHF on the same corrected OMx Hamiltonian.
// The one-electron H_core (incl. ORT + ECP) is spin-free; the two-electron
// part follows the unrestricted Pople NDDO form (cf. run_upm6):
//   Fσ_μμ  = H_μμ + Σ_ν P_νν (μμ|νν) − Σ_ν Pσ_νν (μν|μν)        (one-centre)
//   Fσ_μν += (Pσ + 2Pσ̄)_μν (μν|μν) − Pσ_μν (μμ|νν)             (one-centre, μ≠ν)
//   Fσ_μμ += Σ_{ν∉A} P_νν Γ_μν                                  (two-centre)
//   Fσ_μν += −Pσ_μν γ^K                                          (two-centre)
// ===========================================================================

UPM6Result run_uomx_v2(
    const Molecule& mol,
    const OMxParameterSet& params,
    int max_iter,
    double conv_tol) {

    validate_omx_scf_controls(max_iter, conv_tol, "UOMx_v2");
    const int mult = mol.multiplicity();
    if (mult < 1)
        throw std::invalid_argument("UOMx_v2: multiplicity must be >= 1");

    OMxContext ctx = omx_setup(mol, params, "UOMx_v2");
    const int n_unpaired = mult - 1;
    if (ctx.n_val < 0 || ctx.n_val < n_unpaired
        || (ctx.n_val - n_unpaired) % 2 != 0)
        throw std::invalid_argument("UOMx_v2: invalid multiplicity for electron count");
    const int n_beta = (ctx.n_val - n_unpaired) / 2;
    const int n_alpha = n_beta + n_unpaired;
    if (n_alpha > ctx.n_basis || n_beta > ctx.n_basis) {
        throw std::invalid_argument(
            "UOMx_v2: electron count exceeds the NDDO basis capacity");
    }

    const int n_atoms = ctx.n_atoms;

    Eigen::MatrixXd Da = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_basis);
    Eigen::MatrixXd Db = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_basis);
    DiisState diis;  // stacked [Fa|Fb] with stacked residuals
    double e_last = 0.0;
    UPM6Result result;

    for (int iter = 1; iter <= max_iter; ++iter) {
        Eigen::MatrixXd D_tot = Da + Db;
        Eigen::MatrixXd Fa = ctx.H_core;
        Eigen::MatrixXd Fb = ctx.H_core;

        // ---- One-centre two-electron (unrestricted Pople NDDO) ----
        for (int a = 0; a < n_atoms; ++a) {
            std::vector<int> idx;
            for (int mu = 0; mu < ctx.n_basis; ++mu)
                if (ctx.ao_atom[mu] == a) idx.push_back(mu);
            const int n_ao_a = static_cast<int>(idx.size());

            for (int i = 0; i < n_ao_a; ++i) {
                int mu = idx[i], t_mu = ctx.ao_type[mu];
                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j], t_nu = ctx.ao_type[nu];
                    double coul = 0.0;
                    if (t_mu == 0 && t_nu == 0) coul = ctx.G_ss[mu];
                    else if (t_mu == 0 && t_nu > 0) coul = ctx.G_sp[mu];
                    else if (t_mu > 0 && t_nu == 0) coul = ctx.G_sp[nu];
                    else if (t_mu > 0 && t_nu > 0 && t_mu == t_nu) coul = ctx.G_pp[mu];
                    else coul = ctx.G_p2[mu];
                    double c = D_tot(nu, nu) * coul;
                    Fa(mu, mu) += c;
                    Fb(mu, mu) += c;
                }
                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j], t_nu = ctx.ao_type[nu];
                    double exch = 0.0;
                    if (mu == nu) exch = (t_mu == 0) ? ctx.G_ss[mu] : ctx.G_pp[mu];
                    else if (t_mu == 0 && t_nu > 0) exch = ctx.H_sp[mu];
                    else if (t_mu > 0 && t_nu == 0) exch = ctx.H_sp[nu];
                    else exch = 0.5 * (ctx.G_pp[mu] - ctx.G_p2[mu]);
                    Fa(mu, mu) -= Da(nu, nu) * exch;
                    Fb(mu, mu) -= Db(nu, nu) * exch;
                }
            }
            for (int i = 0; i < n_ao_a; ++i) {
                for (int j = i + 1; j < n_ao_a; ++j) {
                    int mu = idx[i], nu = idx[j];
                    int t_mu = ctx.ao_type[mu], t_nu = ctx.ao_type[nu];
                    double exch = 0.0, coul = 0.0;
                    if (t_mu == 0 && t_nu > 0) { exch = ctx.H_sp[mu]; coul = ctx.G_sp[mu]; }
                    else if (t_mu > 0 && t_nu == 0) { exch = ctx.H_sp[nu]; coul = ctx.G_sp[nu]; }
                    else { exch = 0.5 * (ctx.G_pp[mu] - ctx.G_p2[mu]); coul = ctx.G_p2[mu]; }
                    double fa = (Da(mu, nu) + 2.0 * Db(mu, nu)) * exch - Da(mu, nu) * coul;
                    double fb = (2.0 * Da(mu, nu) + Db(mu, nu)) * exch - Db(mu, nu) * coul;
                    Fa(mu, nu) += fa; Fa(nu, mu) += fa;
                    Fb(mu, nu) += fb; Fb(nu, mu) += fb;
                }
            }
        }

        // ---- Two-centre Coulomb (eq 18) + exchange ----
        for (int mu = 0; mu < ctx.n_basis; ++mu) {
            double v = 0.0;
            for (int nu = 0; nu < ctx.n_basis; ++nu) {
                if (ctx.ao_atom[nu] == ctx.ao_atom[mu]) continue;
                v += D_tot(nu, nu) * ctx.Gamma_ee(mu, nu);
            }
            Fa(mu, mu) += v;
            Fb(mu, mu) += v;
        }
        for (int mu = 0; mu < ctx.n_basis; ++mu) {
            int a_mu = ctx.ao_atom[mu];
            for (int nu = mu + 1; nu < ctx.n_basis; ++nu) {
                int a_nu = ctx.ao_atom[nu];
                if (a_mu == a_nu) continue;
                double exa = -Da(mu, nu) * ctx.Gamma_ee(mu, nu);
                double exb = -Db(mu, nu) * ctx.Gamma_ee(mu, nu);
                Fa(mu, nu) += exa; Fa(nu, mu) += exa;
                Fb(mu, nu) += exb; Fb(nu, mu) += exb;
            }
        }

        // ---- DIIS residuals (orthogonal basis: [F, P] per spin) ----
        Eigen::MatrixXd err_a = Fa * Da - Da * Fa;
        Eigen::MatrixXd err_b = Fb * Db - Db * Fb;
        double res = std::max(err_a.cwiseAbs().maxCoeff(),
                              err_b.cwiseAbs().maxCoeff());

        // ---- Energy: E = ½ Σσ Tr[Pσ(H + Fσ)] + E_core ----
        double E_total = 0.0;
        for (int mu = 0; mu < ctx.n_basis; ++mu) {
            E_total += Da(mu, mu) * (ctx.H_core(mu, mu) + Fa(mu, mu));
            E_total += Db(mu, mu) * (ctx.H_core(mu, mu) + Fb(mu, mu));
            for (int nu = mu + 1; nu < ctx.n_basis; ++nu) {
                E_total += 2.0 * Da(mu, nu) * (ctx.H_core(mu, nu) + Fa(mu, nu));
                E_total += 2.0 * Db(mu, nu) * (ctx.H_core(mu, nu) + Fb(mu, nu));
            }
        }
        E_total = 0.5 * E_total + ctx.E_core;

        const bool converged =
            (iter > 1 && res < conv_tol && std::abs(E_total - e_last) < conv_tol);
        e_last = E_total;

        if (converged || iter == max_iter) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_a(Fa);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_b(Fb);
            if (solver_a.info() != Eigen::Success
                || solver_b.info() != Eigen::Success) {
                throw std::runtime_error(
                    "UOMx_v2: final diagonalisation failed");
            }
            const auto eps_alpha = solver_a.eigenvalues();
            const auto eps_beta = solver_b.eigenvalues();
            result.energy = E_total;
            result.e_electronic = E_total - ctx.E_core;
            result.e_core = ctx.E_core;
            result.mo_energies.resize(2 * ctx.n_basis);
            result.mo_energies.head(ctx.n_basis) = eps_alpha;
            result.mo_energies.tail(ctx.n_basis) = eps_beta;
            result.mo_coeffs.resize(ctx.n_basis, 2 * ctx.n_basis);
            result.mo_coeffs.leftCols(ctx.n_basis) = solver_a.eigenvectors();
            result.mo_coeffs.rightCols(ctx.n_basis) = solver_b.eigenvectors();
            result.density_alpha = Da;
            result.density_beta = Db;
            result.n_basis = ctx.n_basis;
            result.n_alpha = n_alpha;
            result.n_beta = n_beta;
            result.n_iter = iter;
            result.converged = converged;
            return result;
        }

        // ---- DIIS extrapolation on the stacked spin Fock ----
        Eigen::MatrixXd F_stack(2 * ctx.n_basis, ctx.n_basis);
        F_stack.topRows(ctx.n_basis) = Fa;
        F_stack.bottomRows(ctx.n_basis) = Fb;
        Eigen::MatrixXd e_stack(2 * ctx.n_basis, ctx.n_basis);
        e_stack.topRows(ctx.n_basis) = err_a;
        e_stack.bottomRows(ctx.n_basis) = err_b;
        // Zero-density first iterate: see run_omx_v2.
        if (iter > 1) diis.push(F_stack, e_stack);
        Eigen::MatrixXd F_eff = diis.extrapolate(F_stack);
        Eigen::MatrixXd Fa_eff = F_eff.topRows(ctx.n_basis);
        Eigen::MatrixXd Fb_eff = F_eff.bottomRows(ctx.n_basis);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_a(Fa_eff);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_b(Fb_eff);
        if (solver_a.info() != Eigen::Success || solver_b.info() != Eigen::Success)
            throw std::runtime_error("UOMx_v2: diagonalisation failed");
        Eigen::MatrixXd Ca_occ = solver_a.eigenvectors().leftCols(n_alpha);
        Eigen::MatrixXd Cb_occ = solver_b.eigenvectors().leftCols(n_beta);
        Da = Ca_occ * Ca_occ.transpose();
        Db = Cb_occ * Cb_occ.transpose();
    }

    result.n_iter = max_iter;
    result.converged = false;
    return result;
}

// =========================================================================
// Finite-difference OMx gradients (v2)
// =========================================================================

Eigen::MatrixXd compute_omx_v2_gradient_fd(
    const Molecule& mol,
    const OMxParameterSet& params,
    double h,
    int max_iter,
    double conv_tol) {
    return detail::checked_central_fd_gradient(
        mol, params, h, max_iter, conv_tol, params.method_name().c_str(),
        run_omx_v2);
}

Eigen::MatrixXd compute_uomx_v2_gradient_fd(
    const Molecule& mol,
    const OMxParameterSet& params,
    double h,
    int max_iter,
    double conv_tol) {
    const std::string method = "U" + params.method_name();
    return detail::checked_central_fd_gradient(
        mol, params, h, max_iter, conv_tol, method.c_str(), run_uomx_v2);
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
