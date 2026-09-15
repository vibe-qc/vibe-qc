#include "vibeqc/semiempirical/seccm/pm6.hpp"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "vibeqc/semiempirical/methods/nddo/diatomic_gammas.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"
#include "vibeqc/semiempirical/methods/indo/ccm_engine.hpp"
#include "vibeqc/semiempirical/methods/nddo/slater_overlap.hpp"

#include "ewald_1d.h"
#include "seccm_common.h"

namespace vibeqc {
namespace semiempirical {
namespace seccm {
namespace {

using nddo::DiatomicGammas;
using nddo::compute_diatomic_gammas;
using nddo::gamma_ao_pair;
using nddo::nddo_sto_overlap;
using nddo::slater_dir_cos;
using nddo::valence_n_sp;

constexpr double HERMITICITY_TOLERANCE = 1.0e-10;

inline double cos_dir(int d, double dx, double dy, double dz, double R) {
    return slater_dir_cos(d, dx, dy, dz, R);
}

// Multi-term Gaussian gamma (verbatim from pm6_fock.cpp / periodic_pm6.cpp).
double gamma_ab_multi(
    double R,
    const std::vector<nddo::NDDOElementData::GammaTerm>& ga,
    const std::vector<nddo::NDDOElementData::GammaTerm>& gb) {
    double g = 0.0;
    for (const auto& ta : ga) {
        for (const auto& tb : gb) {
            double dAB = ta.exponent + tb.exponent;
            double cAB = ta.coeff * tb.coeff;
            double R2 = R * R;
            double denom = std::sqrt(R2 + dAB * dAB);

            double eta2 = ta.exponent * ta.exponent + tb.exponent * tb.exponent;
            if (eta2 > 0.0 && R2 < eta2) {
                double g_simple = 1.0 / std::sqrt(R2 + eta2);
                g += cAB * g_simple;
            } else if (denom > 0.0) {
                g += cAB / denom;
            }
        }
    }
    return g;
}

}  // namespace

PM6SECCMResult run_pm6_seccm(
    const Molecule& mol,
    const nddo::PM6ParameterSet& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    const PM6SECCMOptions& opts,
    double gap_tolerance) {
    if (params.method_name() != "pm6") {
        throw std::invalid_argument(
            "PM6-SECCM requires PM6 parameters, not "
            + params.method_name());
    }
    detail::validate_common_inputs(
        mol,
        [&params](int Z) { return params.has_element(Z); },
        topology,
        group_order,
        geometry_tolerance,
        HERMITICITY_TOLERANCE,
        gap_tolerance,
        "PM6-SECCM",
        [&params](int Z) {
            if (Z > 86) {
                throw std::invalid_argument(
                    "PM6-SECCM: actinide element Z=" + std::to_string(Z)
                    + " is unavailable because its principal-shell "
                      "convention is not implemented");
            }
            const auto* element = params.element_data(Z);
            if (element == nullptr) {
                throw std::invalid_argument(
                    "PM6-SECCM: element Z=" + std::to_string(Z)
                    + " not in the PM6 parameter set");
            }
            if (Z > 2
                && (element->has_d || element->n_orbitals != 4)) {
                throw std::invalid_argument(
                    "PM6-SECCM: element Z=" + std::to_string(Z)
                    + " requires PM6 d orbitals or has an unsupported AO "
                      "count, but the SECCM PM6 Hamiltonian currently "
                      "implements only s/p channels");
            }
            if (!nddo::pm6_has_complete_sp_parameters(*element)) {
                throw std::invalid_argument(
                    "PM6-SECCM: element Z=" + std::to_string(Z)
                    + " has an incomplete placeholder parameter record");
            }
        },
        [](int, int) {});
    if (opts.max_iter < 1 || !std::isfinite(opts.conv_tol)
        || opts.conv_tol <= 0.0) {
        throw std::invalid_argument(
            "PM6-SECCM options must be positive and finite");
    }

    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());

    // ---- AO basis: s, px, py, pz per main-group atom ----
    std::vector<int> ao_Z;
    std::vector<int> ao_atom;
    std::vector<int> ao_type;
    std::vector<int> atom_ao_start(n_atoms, 0);
    std::vector<int> atom_n_ao(n_atoms, 0);
    int n_basis = 0;
    for (int a = 0; a < n_atoms; ++a) {
        int Z = atoms[a].Z;
        const auto* ed = params.element_data(Z);
        int n_ao = (Z <= 2) ? 1 : (ed ? ed->n_orbitals : 4);
        atom_ao_start[a] = n_basis;
        atom_n_ao[a] = n_ao;
        for (int i = 0; i < n_ao; ++i) {
            ao_Z.push_back(Z);
            ao_atom.push_back(a);
            ao_type.push_back(i);
        }
        n_basis += n_ao;
    }

    int n_elec = 0;
    for (const auto& atom : atoms)
        n_elec += nddo::pm6_core_charge(atom.Z);
    int n_occ = n_elec / 2;
    if (n_occ > n_basis) n_occ = n_basis;
    if (n_occ >= n_basis || n_occ < 1) {
        throw std::invalid_argument(
            "PM6-SECCM occupancy does not fit the finite-cluster AO space "
            "or leaves no LUMO");
    }
    if (n_elec % 2 != 0) {
        throw std::invalid_argument(
            "PM6-SECCM requires a positive even valence-electron count");
    }

    // ---- Pre-load per-AO integrals (eV -> Ha) ----
    double ev2ha = 1.0 / 27.2114;
    std::vector<double> U_s(n_basis, 0.0),
                        G_ss(n_basis, 0.0),
                        G_pp(n_basis, 0.0),
                        G_sp(n_basis, 0.0),
                        G_p2(n_basis, 0.0),
                        H_sp(n_basis, 0.0),
                        beta(n_basis, 0.0),
                        rho(n_basis, 0.0);

    for (int mu = 0; mu < n_basis; ++mu) {
        int Z = ao_Z[mu];
        const auto* ed = params.element_data(Z);
        if (!ed) continue;
        bool is_s = (ao_type[mu] == 0);
        U_s[mu]  = (is_s ? ed->uss : ed->upp) * ev2ha;
        G_ss[mu] = ed->gss * ev2ha;
        G_pp[mu] = ed->gpp * ev2ha;
        G_sp[mu] = ed->gsp * ev2ha;
        G_p2[mu] = ed->gp2 * ev2ha;
        H_sp[mu] = ed->hsp * ev2ha;
        beta[mu] = (is_s ? ed->betas : ed->betap) * ev2ha;
        rho[mu]  = (ed->gss > 0.0) ? 1.0 / (ed->gss * ev2ha) : 2.72;
    }

    std::vector<int> core_charge(n_atoms);
    for (int a = 0; a < n_atoms; ++a)
        core_charge[a] = nddo::pm6_core_charge(atoms[a].Z);

    // ---- Opt-in CCM Madelung/Ewald embedding ----
    //
    // A finite cyclic cluster carries the Coulomb interaction only out to
    // the Wigner-Seitz cell; the conditionally convergent lattice tail
    // beyond it is missing. MSINDO restores it as a classical point-charge
    // field added to the DIAGONAL of the Fock matrix (ccmfockcl.f:
    // FA(K,K) = H(K,K) - MADELATOM(I)) with the self-energy half
    // accumulated on the CORE charges (MADELENRG += CZ(I)*0.5*MADELATOM(I)).
    // The potential itself is the full Ewald lattice sum minus the
    // WS-internal bare 1/r, WS-weighted with the self image excluded
    // (madelsum.f, SMADEL branch), because the NDDO two-centre integrals
    // already carry everything inside the WS cell:
    //
    //   V_mad(I) = sum_J q_J * MADKONST(I,J)
    //            - sum_{J in WS(I), |r_IJ| > 0} q_J w_IJ / |r_IJ|
    //
    // indo::_madelung_potential_ewald is the verified port of exactly that.
    //
    // 1-D uses the converged background-corrected wire Ewald rather than
    // MSINDO's frozen truncated +-2-shell sum; 2-D uses Parry/Heyes. 3-D
    // fails closed: the WS-folded remainder it would leave has no
    // thermodynamic limit (issues #211, #425, #444), so an embedded 3-D
    // number would drift with cluster size rather than converge.
    const int dim = static_cast<int>(topology.translations.size());
    const bool trivial_records = detail::is_trivial_molecular_records(topology);
    if (opts.madelung && dim == 3) {
        throw std::invalid_argument(
            "PM6-SECCM 3-D Madelung embedding is not implemented: the "
            "WS-folded remainder has no thermodynamic limit (issues #211, "
            "#425, #444); 1-D and 2-D embedding are available");
    }
    if (!opts.madelung && !trivial_records
        && !opts.allow_truncated_electrostatics) {
        throw std::invalid_argument(
            "PM6-SECCM on a nontrivial cyclic topology omits the "
            "long-range Coulomb tail without madelung=true: the "
            "WS-truncated monopole sum reproduces the exact rocksalt "
            "Madelung constant only to -43%..+38% with the sign flipping "
            "on replica parity, so the result is not quantitative. Enable "
            "madelung=true (1-D/2-D), or set "
            "allow_truncated_electrostatics=true to acknowledge a "
            "non-quantitative algorithm-mechanics probe");
    }
    std::vector<std::vector<indo::WSNeighbor>> ews;
    Eigen::MatrixXd madkonst;
    if (opts.madelung) {
        ews = indo::_ewald_ws_cells(topology);
        if (dim == 1) {
            madkonst =
                detail::wire_madkonst_1d(ews, topology.translations[0], n_atoms);
        } else {
            madkonst = indo::_madkonst_2d(ews, topology.translations, n_atoms);
        }
        if (!madkonst.allFinite()) {
            throw std::runtime_error(
                "PM6-SECCM Madelung-constant matrix produced non-finite "
                "values");
        }
    }
    // Net atomic charges q_I = Z_core(I) - population(I), the MSINDO
    // EWALDCHARGES convention.
    auto net_charges = [&](const Eigen::MatrixXd& density) {
        Eigen::VectorXd q(n_atoms);
        for (int a = 0; a < n_atoms; ++a) q(a) = core_charge[a];
        for (int mu = 0; mu < n_basis; ++mu)
            q(ao_atom[mu]) -= density(mu, mu);
        return q;
    };

    // ---- SCF ----
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_basis, n_basis);
    PM6SECCMResult result;

    // Pulay DIIS on the commutator [F, P], with the molecular driver's
    // bounded damped-mixing window for hard Aufbau cycles.
    std::vector<Eigen::MatrixXd> diis_F, diis_e;
    const std::size_t DIIS_MAX = 8;
    const int damping_length = 32;
    const int damping_start = std::min(64, std::max(2, opts.max_iter / 3));
    const int diis_restart =
        std::min(opts.max_iter, damping_start + damping_length);

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n_basis, n_basis);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_basis, n_basis);

        // ===========================================================
        // ONE-CENTRE (unchanged molecular PM6 blocks)
        // ===========================================================
        for (int a = 0; a < n_atoms; ++a) {
            std::vector<int> idx;
            for (int mu = 0; mu < n_basis; ++mu)
                if (ao_atom[mu] == a) idx.push_back(mu);
            int n_ao_a = static_cast<int>(idx.size());

            for (int i = 0; i < n_ao_a; ++i) {
                int mu = idx[i];
                int t_mu = ao_type[mu];

                F(mu, mu) += U_s[mu];
                H(mu, mu) += U_s[mu];

                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j];
                    int t_nu = ao_type[nu];
                    double coul = 0.0;
                    if (t_mu == 0 && t_nu == 0)
                        coul = G_ss[mu];
                    else if (t_mu == 0 && t_nu > 0)
                        coul = G_sp[mu];
                    else if (t_mu > 0 && t_nu == 0)
                        coul = G_sp[nu];
                    else if (t_mu > 0 && t_nu > 0 && t_mu == t_nu)
                        coul = G_pp[mu];
                    else
                        coul = G_p2[mu];
                    F(mu, mu) += D(nu, nu) * coul;
                }

                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j];
                    int t_nu = ao_type[nu];
                    double exch = 0.0;
                    if (mu == nu) {
                        exch = (t_mu == 0) ? G_ss[mu] : G_pp[mu];
                    } else if (t_mu == 0 && t_nu > 0) {
                        exch = H_sp[mu];
                    } else if (t_mu > 0 && t_nu == 0) {
                        exch = H_sp[nu];
                    } else {
                        exch = 0.5 * (G_pp[mu] - G_p2[mu]);
                    }
                    F(mu, mu) -= 0.5 * D(nu, nu) * exch;
                }
            }

            for (int i = 0; i < n_ao_a; ++i) {
                for (int j = i + 1; j < n_ao_a; ++j) {
                    int mu = idx[i], nu = idx[j];
                    int t_mu = ao_type[mu], t_nu = ao_type[nu];

                    double exch = 0.0, coul = 0.0;
                    if (t_mu == 0 && t_nu > 0) {
                        exch = H_sp[mu]; coul = G_sp[mu];
                    } else if (t_mu > 0 && t_nu == 0) {
                        exch = H_sp[nu]; coul = G_sp[nu];
                    } else {
                        exch = 0.5 * (G_pp[mu] - G_p2[mu]);
                        coul = G_p2[mu];
                    }
                    double f_off = 0.5 * D(mu, nu) * (3.0 * exch - coul);
                    F(mu, nu) += f_off;
                    F(nu, mu) += f_off;
                }
            }
        }

        // ===========================================================
        // TWO-CENTRE: WS-weighted directed records. The displacement
        // convention mirrors the molecular driver: for the record
        // (a -> b image) the working vector is d = R_a - (R_b + T),
        // which is -image.disp.
        // ===========================================================
        for (int central = 0; central < n_atoms; ++central) {
            const int a = central;
            const auto* ed_a = params.element_data(atoms[a].Z);
            for (const auto& image : topology.cells[central]) {
                const int b = image.origin;
                const double w = image.weight;
                const Eigen::Vector3d d_work = -image.disp;
                const double dx = d_work[0];
                const double dy = d_work[1];
                const double dz = d_work[2];
                const double R = d_work.norm();
                if (R < 1e-12) continue;

                const auto* ed_b = params.element_data(atoms[b].Z);

                // The published PM6 H--X s/p integrals retain dipole and
                // quadrupole components that the generic population-gamma
                // approximation below omits.  Assemble the exact tensor per
                // directed image record: X -> H owns the X on-site block,
                // while H -> X owns the H diagonal.  The reverse record then
                // supplies the transpose of the interatomic exchange block.
                const bool a_is_sp_heavy = atoms[a].Z > 2
                    && atom_n_ao[a] == 4 && ed_a && !ed_a->has_d;
                const bool b_is_sp_heavy = atoms[b].Z > 2
                    && atom_n_ao[b] == 4 && ed_b && !ed_b->has_d;
                const bool exact_sp_hydrogen = params.method_name() == "pm6"
                    && ((a_is_sp_heavy && atoms[b].Z == 1)
                        || (atoms[a].Z == 1 && b_is_sp_heavy));
                nddo::PM6SPHydrogenPair exact_pair;
                int exact_heavy_start = -1;
                int exact_hydrogen_ao = -1;
                if (exact_sp_hydrogen) {
                    const bool heavy_is_central = a_is_sp_heavy;
                    const int heavy = heavy_is_central ? a : b;
                    const int hydrogen = heavy_is_central ? b : a;
                    const Eigen::Vector3d heavy_to_hydrogen =
                        heavy_is_central ? image.disp : -image.disp;
                    exact_heavy_start = atom_ao_start[heavy];
                    exact_hydrogen_ao = atom_ao_start[hydrogen];
                    exact_pair = nddo::pm6_sp_hydrogen_pair(
                        atoms[heavy].Z,
                        heavy_to_hydrogen,
                        *params.element_data(atoms[heavy].Z),
                        *params.element_data(1));
                }
                const bool exact_sp_sp = params.method_name() == "pm6"
                    && a_is_sp_heavy && b_is_sp_heavy;
                nddo::PM6SPSPPair exact_sp_sp_integrals;
                if (exact_sp_sp) {
                    exact_sp_sp_integrals = nddo::pm6_sp_sp_pair(
                        atoms[a].Z,
                        atoms[b].Z,
                        image.disp,
                        *ed_a,
                        *ed_b);
                }

                double q_B = 0.0;
                for (int nu = 0; nu < n_basis; ++nu)
                    if (ao_atom[nu] == b) q_B += D(nu, nu);

                double g_ab = 0.0;
                if (ed_a && ed_b && !ed_a->gamma_terms.empty()
                    && !ed_b->gamma_terms.empty()) {
                    g_ab = gamma_ab_multi(
                        R, ed_a->gamma_terms, ed_b->gamma_terms);
                }
                if (g_ab == 0.0) {
                    int mu_a0 = 0, mu_b0 = 0;
                    for (int mu = 0; mu < n_basis; ++mu) {
                        if (ao_atom[mu] == a) { mu_a0 = mu; break; }
                    }
                    for (int mu = 0; mu < n_basis; ++mu) {
                        if (ao_atom[mu] == b) { mu_b0 = mu; break; }
                    }
                    double eta = 0.5 * (rho[mu_a0] + rho[mu_b0]);
                    g_ab = 1.0 / std::sqrt(R * R + eta * eta);
                }

                DiatomicGammas dg;
                if (ed_a && ed_b && !ed_a->gamma_terms.empty()
                    && !ed_b->gamma_terms.empty()) {
                    double zr_a = ed_a->zs > 0 ? ed_a->zp / ed_a->zs : 1.0;
                    double zr_b = ed_b->zs > 0 ? ed_b->zp / ed_b->zs : 1.0;
                    dg = compute_diatomic_gammas(
                        R, ed_a->gamma_terms, ed_b->gamma_terms, zr_a, zr_b);
                }

                if (exact_sp_hydrogen) {
                    const auto& eri = exact_pair.electron_repulsion;
                    if (a_is_sp_heavy) {
                        const double d_hh = D(
                            exact_hydrogen_ao, exact_hydrogen_ao);
                        F.block<4, 4>(
                            exact_heavy_start, exact_heavy_start)
                            += w * (d_hh * eri + exact_pair.heavy_core);
                        H.block<4, 4>(
                            exact_heavy_start, exact_heavy_start)
                            += w * exact_pair.heavy_core;
                    } else {
                        const double j_h = (D.block<4, 4>(
                            exact_heavy_start, exact_heavy_start)
                            .cwiseProduct(eri)).sum();
                        F(exact_hydrogen_ao, exact_hydrogen_ao)
                            += w * (j_h + exact_pair.hydrogen_core);
                        H(exact_hydrogen_ao, exact_hydrogen_ao)
                            += w * exact_pair.hydrogen_core;
                    }
                } else if (exact_sp_sp) {
                    const int a0 = atom_ao_start[a];
                    const int b0 = atom_ao_start[b];
                    Eigen::Matrix4d j_a = Eigen::Matrix4d::Zero();
                    const auto& eri =
                        exact_sp_sp_integrals.electron_repulsion;
                    for (int mu = 0; mu < 4; ++mu) {
                        for (int nu = 0; nu < 4; ++nu) {
                            for (int la = 0; la < 4; ++la) {
                                for (int si = 0; si < 4; ++si) {
                                    j_a(mu, nu) += D(b0 + la, b0 + si)
                                        * eri(4 * mu + nu, 4 * la + si);
                                }
                            }
                        }
                    }
                    F.block<4, 4>(a0, a0) += w
                        * (j_a + exact_sp_sp_integrals.first_core);
                    H.block<4, 4>(a0, a0) +=
                        w * exact_sp_sp_integrals.first_core;
                } else {
                    // Nuclear attraction + electron-core attraction on atom A.
                    double g_s, g_p_sigma, g_p_pi;
                    if (dg.g_ss != 0.0) {
                        g_s = dg.g_ss;
                        g_p_sigma = dg.g_pp_ss;
                        g_p_pi = dg.g_sp_p;
                    } else {
                        g_s = g_p_sigma = g_p_pi = g_ab;
                    }
                    const double z_core = core_charge[b];
                    const double g_core = nddo::pm6_electron_core_gamma(
                        R, *ed_a, *ed_b);

                    for (int mu = 0; mu < n_basis; ++mu) {
                        if (ao_atom[mu] != a) continue;
                        double g_mu;
                        if (ao_type[mu] == 0 || g_s == g_p_sigma) {
                            g_mu = g_s;
                        } else {
                            int d = ao_type[mu] - 1;
                            double ca = cos_dir(d, dx, dy, dz, R);
                            double ca2 = ca * ca;
                            g_mu = g_p_sigma * ca2
                                + g_p_pi * (1.0 - ca2);
                        }
                        F(mu, mu) +=
                            w * (q_B * g_mu - z_core * g_core);
                        H(mu, mu) += w * (-z_core) * g_core;
                    }
                }

                if (exact_sp_hydrogen) {
                    const Eigen::Vector4d d_xh = D.block<4, 1>(
                        exact_heavy_start, exact_hydrogen_ao);
                    const Eigen::Vector4d exchange =
                        -0.5 * exact_pair.electron_repulsion * d_xh;
                    if (a_is_sp_heavy) {
                        F.block<4, 1>(
                            exact_heavy_start, exact_hydrogen_ao)
                            += w * exchange;
                    } else {
                        F.block<1, 4>(
                            exact_hydrogen_ao, exact_heavy_start)
                            += w * exchange.transpose();
                    }
                } else if (exact_sp_sp) {
                    const int a0 = atom_ao_start[a];
                    const int b0 = atom_ao_start[b];
                    Eigen::Matrix4d exchange = Eigen::Matrix4d::Zero();
                    const auto& eri =
                        exact_sp_sp_integrals.electron_repulsion;
                    for (int mu = 0; mu < 4; ++mu) {
                        for (int la = 0; la < 4; ++la) {
                            for (int nu = 0; nu < 4; ++nu) {
                                for (int si = 0; si < 4; ++si) {
                                    exchange(mu, la) -= 0.5
                                        * D(a0 + nu, b0 + si)
                                        * eri(4 * mu + nu, 4 * la + si);
                                }
                            }
                        }
                    }
                    F.block<4, 4>(a0, b0) += w * exchange;
                }

                // Resonance + exchange, directed (a -> b image) only; the
                // reverse record supplies the transpose.
                for (int mu = 0; mu < n_basis; ++mu) {
                    if (ao_atom[mu] != a) continue;
                    int t_mu = ao_type[mu];
                    for (int nu = 0; nu < n_basis; ++nu) {
                        if (ao_atom[nu] != b) continue;
                        int t_nu = ao_type[nu];

                        double zs_a = ed_a ? ed_a->zs : 1.0;
                        double zp_a = ed_a ? ed_a->zp : 1.0;
                        double zs_b = ed_b ? ed_b->zs : 1.0;
                        double zp_b = ed_b ? ed_b->zp : 1.0;
                        if (zs_a <= 0.0) zs_a = 1.0;
                        if (zp_a <= 0.0) zp_a = zs_a;
                        if (zs_b <= 0.0) zs_b = 1.0;
                        if (zp_b <= 0.0) zp_b = zs_b;
                        double S_val = nddo_sto_overlap(
                            valence_n_sp(ao_Z[mu]), t_mu, zs_a, zp_a,
                            valence_n_sp(ao_Z[nu]), t_nu, zs_b, zp_b,
                            dx, dy, dz, R);

                        double beta_val =
                            0.5 * (beta[mu] + beta[nu]) * S_val;
                        H(mu, nu) += w * beta_val;
                        F(mu, nu) += w * beta_val;

                        if (exact_sp_hydrogen || exact_sp_sp) continue;

                        double g_ex = 0.0;
                        if (dg.g_ss != 0.0) {
                            double ca = 0.0, cb = 0.0;
                            if (t_mu > 0)
                                ca = cos_dir(t_mu - 1, dx, dy, dz, R);
                            if (t_nu > 0)
                                cb = cos_dir(t_nu - 1, dx, dy, dz, R);
                            g_ex = gamma_ao_pair(dg, t_mu, t_nu, ca, cb);
                        }
                        if (g_ex == 0.0) {
                            double eta_ex = 0.5 * (rho[mu] + rho[nu]);
                            g_ex = 1.0 / std::sqrt(R * R + eta_ex * eta_ex);
                        }
                        F(mu, nu) += w * (-0.5 * D(mu, nu) * g_ex);
                    }
                }
            }
        }

        // Check the directed result before cleanup: unconditional
        // symmetrization must not hide a missing reverse image or a directed
        // assembly error.
        if (detail::max_abs(F - F.transpose()) > HERMITICITY_TOLERANCE
            || detail::max_abs(H - H.transpose()) > HERMITICITY_TOLERANCE) {
            throw std::runtime_error(
                "PM6-SECCM reverse-image assembly is not Hermitian");
        }

        // Retain the numerical cleanup before the self-adjoint eigensolver.
        F = (0.5 * (F + F.transpose())).eval();
        H = (0.5 * (H + H.transpose())).eval();

        // CCM Madelung deposit, MSINDO ccmfockcl.f convention:
        //   FA(K,K) = H(K,K) - MADELATOM(I)
        // A real diagonal shift, so it cannot disturb the Hermiticity the
        // guard above just checked, and it is exactly the derivative of
        // the 1/2 q.V_mad energy with respect to the population, which
        // keeps the embedded SCF variational.
        Eigen::VectorXd mad_iter;
        if (opts.madelung) {
            mad_iter = indo::_madelung_potential_ewald(
                net_charges(D), madkonst, ews);
            if (!mad_iter.allFinite()) {
                throw std::runtime_error(
                    "PM6-SECCM Madelung potential produced non-finite "
                    "values");
            }
            for (int mu = 0; mu < n_basis; ++mu)
                F(mu, mu) -= mad_iter(ao_atom[mu]);
        }

        Eigen::MatrixXd err = F * D - D * F;
        const bool use_damping =
            iter >= damping_start && iter < diis_restart;
        if (iter == damping_start || iter == diis_restart) {
            diis_F.clear();
            diis_e.clear();
        }
        if (iter > 1 && !use_damping) {
            diis_F.push_back(F);
            diis_e.push_back(err);
            if (diis_F.size() > DIIS_MAX) {
                diis_F.erase(diis_F.begin());
                diis_e.erase(diis_e.begin());
            }
        }
        Eigen::MatrixXd F_eff = F;
        if (!use_damping) {
            int n = static_cast<int>(diis_F.size());
            if (n >= 2) {
                Eigen::MatrixXd B =
                    Eigen::MatrixXd::Constant(n + 1, n + 1, -1.0);
                B(n, n) = 0.0;
                for (int i = 0; i < n; ++i)
                    for (int j = i; j < n; ++j)
                        B(i, j) = B(j, i) =
                            diis_e[i].cwiseProduct(diis_e[j]).sum();
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
                rhs(n) = -1.0;
                Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);
                if (c.allFinite()) {
                    F_eff.setZero();
                    for (int i = 0; i < n; ++i) F_eff += c(i) * diis_F[i];
                }
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_eff);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("PM6-SECCM: diagonalisation failed");
        }
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        Eigen::MatrixXd D_new = 2.0 * C_occ * C_occ.transpose();

        double delta = (D_new - D).cwiseAbs().maxCoeff();
        if (!std::isfinite(delta)) {
            throw std::runtime_error(
                "PM6-SECCM: non-finite density change at iteration "
                + std::to_string(iter));
        }
        D = use_damping ? 0.5 * D + 0.5 * D_new : D_new;
        Eigen::MatrixXd comm = F_eff * D - D * F_eff;
        double comm_max = comm.cwiseAbs().maxCoeff();

        if (delta < opts.conv_tol && comm_max < opts.conv_tol) {
            const double gap = eps(n_occ) - eps(n_occ - 1);
            if (!std::isfinite(gap) || gap <= gap_tolerance) {
                throw std::runtime_error(
                    "PM6-SECCM requires a positive finite-torus HOMO-LUMO "
                    "gap");
            }

            // Core-core repulsion, WS-weighted.
            double cyclic_core = 0.0;
            int n_records = 0;
            for (int central = 0; central < n_atoms; ++central) {
                const auto* ed_a =
                    params.element_data(atoms[central].Z);
                for (const auto& image : topology.cells[central]) {
                    const auto* ed_b =
                        params.element_data(atoms[image.origin].Z);
                    if (!ed_a || !ed_b) continue;
                    const double R = image.disp.norm();
                    if (R < 1e-12) continue;
                    const auto* dp = params.diatomic(
                        atoms[central].Z, atoms[image.origin].Z);
                    cyclic_core += 0.5 * image.weight
                        * nddo::pm6_core_core_repulsion(
                            atoms[central].Z,
                            atoms[image.origin].Z,
                            R,
                            *ed_a,
                            *ed_b,
                            dp);
                    ++n_records;
                }
            }

            // E = 1/2 Tr[P(H+F)] + E_core on the supercell.
            double cyclic_total = 0.0;
            for (int mu = 0; mu < n_basis; ++mu) {
                cyclic_total += D(mu, mu) * (H(mu, mu) + F(mu, mu));
                for (int nu = mu + 1; nu < n_basis; ++nu)
                    cyclic_total +=
                        2.0 * D(mu, nu) * (H(mu, nu) + F(mu, nu));
            }
            cyclic_total *= 0.5;
            cyclic_total += cyclic_core;

            // Madelung self-energy, MSINDO ccmfockcl.f:
            //   MADELENRG += CZ(I) * 0.5 * MADELATOM(I)
            // The electronic half already rides in 1/2 Tr[P(H+F)] through
            // the diagonal deposit above, so only the core-charge half is
            // added here; together they give 1/2 sum_I q_I V_mad(I).
            // Recomputed from the CONVERGED density rather than reusing
            // the pre-update iterate, so the reported component matches
            // 1/2 q.V_mad at result.density exactly instead of to the
            // convergence tolerance.
            double cyclic_madelung = 0.0;
            if (opts.madelung) {
                const Eigen::VectorXd mad_final =
                    indo::_madelung_potential_ewald(
                        net_charges(D), madkonst, ews);
                for (int a = 0; a < n_atoms; ++a)
                    cyclic_madelung +=
                        0.5 * static_cast<double>(core_charge[a])
                        * mad_final(a);
                cyclic_total += cyclic_madelung;
            }

            const double normalization = static_cast<double>(group_order);
            result.energy = cyclic_total / normalization;
            result.e_electronic =
                (cyclic_total - cyclic_core) / normalization;
            result.e_core = cyclic_core / normalization;
            result.e_madelung = cyclic_madelung / normalization;
            result.total_cyclic_energy = cyclic_total;
            result.cyclic_core_energy = cyclic_core;
            result.cyclic_madelung_energy = cyclic_madelung;
            result.homo_lumo_gap = gap;
            result.mo_energies = eps;
            result.mo_coeffs = C;
            result.density = D;
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_iter = iter;
            result.group_order = group_order;
            result.n_records = n_records;
            result.converged = true;
            return result;
        }
    }

    result.n_iter = opts.max_iter;
    result.density = D;
    result.converged = false;
    return result;
}

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
