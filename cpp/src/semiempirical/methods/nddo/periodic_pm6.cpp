// Periodic PM6 NDDO Fock builder.
//
// Lattice-summed analogue of the molecular PM6 driver (pm6_fock.cpp).
// Follows the same pattern as periodic SCC-DFTB.

#include "vibeqc/semiempirical/methods/nddo/periodic_pm6.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/slater_overlap.hpp"
#include "vibeqc/diis.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// Direction cosine helper.  The old 1s-shaped slater_overlap_ss/sp/pp_*
// wrappers are removed — use the n-dependent nddo_sto_overlap() from
// slater_overlap.hpp instead (the MSINDO s2int kernel, validated against
// numerical 3-D integration).
inline double cos_dir(int d, double dx, double dy, double dz, double R) {
    return slater_dir_cos(d, dx, dy, dz, R);
}

// Multi-term Gaussian gamma
static inline double gamma_ab_multi(
    double R,
    const std::vector<NDDOElementData::GammaTerm>& ga,
    const std::vector<NDDOElementData::GammaTerm>& gb);

// ---------------------------------------------------------------------------
// Multi-term Gaussian gamma expansion
// ---------------------------------------------------------------------------
static inline double gamma_ab_multi(
    double R,
    const std::vector<NDDOElementData::GammaTerm>& ga,
    const std::vector<NDDOElementData::GammaTerm>& gb) {
    double g = 0.0;
    for (const auto& ta : ga) {
        for (const auto& tb : gb) {
            double dAB = ta.exponent + tb.exponent;
            double cAB = ta.coeff * tb.coeff;
            double R2 = R * R;
            double denom = std::sqrt(R2 + dAB * dAB);

            // MOPAC PM6: convert to eV, compute 1/R(gamma) limit, convert back
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

static double electronic_gamma_ab(
    double R,
    const NDDOElementData* ed_a,
    const NDDOElementData* ed_b,
    double rho_a,
    double rho_b) {
    double gamma = 0.0;
    if (ed_a && ed_b &&
        !ed_a->gamma_terms.empty() && !ed_b->gamma_terms.empty()) {
        gamma = gamma_ab_multi(R, ed_a->gamma_terms, ed_b->gamma_terms);
    }
    if (gamma == 0.0) {
        const double eta = 0.5 * (rho_a + rho_b);
        gamma = 1.0 / std::sqrt(R * R + eta * eta);
    }
    return gamma;
}

// =====================================================================

PeriodicPM6Result run_pm6_gamma_with_cells(
    const PeriodicSystem& system,
    const PM6ParameterSet& params,
    const PeriodicPM6Options& opts,
    const std::vector<LatticeCell>& cells) {

    const std::string active_method = params.method_name();
    if (active_method != "pm6") {
        throw std::invalid_argument(
            "Periodic PM6 driver requires PM6 parameters, not "
            + active_method);
    }
    if (opts.max_iter < 1 || !std::isfinite(opts.conv_tol)
        || opts.conv_tol <= 0.0 || !std::isfinite(opts.cutoff_bohr)
        || opts.cutoff_bohr < 0.0) {
        throw std::invalid_argument(
            "Periodic PM6 options must be positive and finite, with a "
            "nonnegative finite cutoff");
    }
    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());
    if (n_atoms == 0)
        throw std::invalid_argument("Periodic PM6: empty unit cell");
    if (system.charge != 0) {
        throw std::invalid_argument(
            "Periodic PM6: charged unit cells require a compensating "
            "electrostatics convention and are not supported");
    }
    if (system.multiplicity != 1) {
        throw std::invalid_argument(
            "Periodic PM6: only closed-shell multiplicity 1 is supported");
    }
    for (const auto& atom : atoms) {
        if (atom.Z > 86) {
            throw std::invalid_argument(
                "Periodic PM6: actinide element Z="
                + std::to_string(atom.Z)
                + " is unavailable because its principal-shell convention "
                  "is not implemented");
        }
        const auto* ed = params.element_data(atom.Z);
        if (!ed) {
            throw std::invalid_argument(
                "Periodic PM6: element Z=" + std::to_string(atom.Z)
                + " is not in the parameter set");
        }
        if (atom.Z > 2 && (ed->has_d || ed->n_orbitals != 4)) {
            throw std::invalid_argument(
                "Periodic PM6: d-shell element Z="
                + std::to_string(atom.Z)
                + " or unsupported AO count is not supported by the "
                  "s/p-only periodic kernel");
        }
        if (!pm6_has_complete_sp_parameters(*ed)) {
            throw std::invalid_argument(
                "Periodic PM6: element Z=" + std::to_string(atom.Z)
                + " has an incomplete placeholder parameter record");
        }
    }

    // ---- AO basis: s, px, py, pz per atom ----
    std::vector<int> ao_Z;
    std::vector<int> ao_atom;
    std::vector<int> ao_type;   // 0=s, 1=px, 2=py, 3=pz
    std::vector<int> first_ao(n_atoms, -1);
    std::vector<int> atom_n_ao(n_atoms, 0);
    int n_basis = 0;
    for (int a = 0; a < n_atoms; ++a) {
        int Z = atoms[a].Z;
        const auto* ed = params.element_data(Z);
        int n_ao = ed ? ed->n_orbitals : 4;
        // H and He have only a 1s shell — 4 AOs (s+p) would give phantom
        // 2p AOs that sink the valence density once the e-core attraction
        // is added to H (below), causing population collapse.
        if (Z <= 2) n_ao = 1;
        first_ao[a] = n_basis;
        atom_n_ao[a] = n_ao;
        for (int i = 0; i < n_ao; ++i) {
            ao_Z.push_back(Z);
            ao_atom.push_back(a);
            ao_type.push_back(i);
        }
        n_basis += n_ao;
    }

    int n_elec = 0;
    for (int a = 0; a < n_atoms; ++a)
        n_elec += pm6_core_charge(atoms[a].Z);  // valence-only count, not all-electron
    n_elec -= system.charge;
    if (n_elec < 0 || n_elec % 2 != 0 || n_elec > 2 * n_basis) {
        throw std::invalid_argument(
            "Periodic PM6: electron count is incompatible with the "
            "closed-shell NDDO basis");
    }
    const int n_occ = n_elec / 2;

    // ---- Pre-load per-AO integrals (eV → Ha) ----
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

    // Effective core charges
    std::vector<int> core_charge(n_atoms);
    for (int a = 0; a < n_atoms; ++a)
        core_charge[a] = pm6_core_charge(atoms[a].Z);

    if (cells.empty()) {
        throw std::invalid_argument(
            "Periodic PM6: prepared lattice-cell list is empty");
    }
    int n_cells = static_cast<int>(cells.size());

    for (const auto& cell : cells) {
        const bool is_zero_cell = (cell.index.array() == 0).all();
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = 0; b < n_atoms; ++b) {
                if (is_zero_cell && a == b) continue;
                double distance_squared = 0.0;
                for (int d = 0; d < 3; ++d) {
                    const double delta = atoms[a].xyz[d]
                        - (atoms[b].xyz[d] + cell.r_cart[d]);
                    distance_squared += delta * delta;
                }
                if (distance_squared < 1.0e-24) {
                    throw std::invalid_argument(
                        "Periodic PM6: distinct atoms/images "
                        + std::to_string(a) + " and " + std::to_string(b)
                        + " are coincident");
                }
            }
        }
    }

    const bool has_nonzero_cell = std::any_of(
        cells.begin(), cells.end(), [](const LatticeCell& cell) {
            return (cell.index.array() != 0).any();
        });
    if (!opts.gamma_only_0 && !has_nonzero_cell) {
        require_nonzero_lattice_image(
            cells, opts.cutoff_bohr, "run_pm6_gamma");
    }
    const bool has_hydrogen = std::any_of(
        atoms.begin(), atoms.end(), [](const Atom& atom) {
            return atom.Z == 1;
        });
    const bool has_sp_heavy = std::any_of(
        atoms.begin(), atoms.end(), [&params](const Atom& atom) {
            const auto* ed = params.element_data(atom.Z);
            return atom.Z > 2 && ed && !ed->has_d && ed->n_orbitals == 4;
        });
    if (params.method_name() == "pm6" && has_nonzero_cell
        && has_hydrogen && has_sp_heavy) {
        throw std::invalid_argument(
            "Periodic PM6: nonzero-image H-X exchange requires a "
            "cell-resolved density and is not implemented; use PM6-SECCM "
            "for a finite cyclic cluster or gamma_only_0 for the molecular "
            "limit");
    }

    // Resolve the source-verifiable PM6 multipole block only for the
    // s/p-heavy--H subclass covered by pm6_sp_hydrogen_pair(). PM7 has a
    // distinct feathered Hamiltonian and is rejected before this driver.
    auto make_exact_sp_hydrogen_pair = [&params, &atoms, &atom_n_ao](
        int a,
        int b,
        const Eigen::Vector3d& b_translation,
        int& heavy,
        int& hydrogen,
        PM6SPHydrogenPair& integrals) {
        if (params.method_name() != "pm6") return false;
        if (atoms[a].Z == 1 && atoms[b].Z > 2) {
            hydrogen = a;
            heavy = b;
        } else if (atoms[b].Z == 1 && atoms[a].Z > 2) {
            hydrogen = b;
            heavy = a;
        } else {
            return false;
        }
        const auto* ed_x = params.element_data(atoms[heavy].Z);
        const auto* ed_h = params.element_data(1);
        if (!ed_x || !ed_h || ed_x->has_d || atom_n_ao[heavy] != 4)
            return false;

        Eigen::Vector3d heavy_position(
            atoms[heavy].xyz[0],
            atoms[heavy].xyz[1],
            atoms[heavy].xyz[2]);
        Eigen::Vector3d hydrogen_position(
            atoms[hydrogen].xyz[0],
            atoms[hydrogen].xyz[1],
            atoms[hydrogen].xyz[2]);
        if (heavy == b) heavy_position += b_translation;
        if (hydrogen == b) hydrogen_position += b_translation;
        integrals = pm6_sp_hydrogen_pair(
            atoms[heavy].Z,
            hydrogen_position - heavy_position,
            *ed_x,
            *ed_h);
        return true;
    };

    auto make_exact_sp_sp_pair = [&params, &atoms, &atom_n_ao](
        int a,
        int b,
        const Eigen::Vector3d& b_translation,
        PM6SPSPPair& integrals) {
        if (params.method_name() != "pm6" || atoms[a].Z <= 2
            || atoms[b].Z <= 2 || atom_n_ao[a] != 4
            || atom_n_ao[b] != 4) {
            return false;
        }
        const auto* ed_a = params.element_data(atoms[a].Z);
        const auto* ed_b = params.element_data(atoms[b].Z);
        if (!ed_a || !ed_b || ed_a->has_d || ed_b->has_d) return false;
        Eigen::Vector3d displacement;
        for (int d = 0; d < 3; ++d) {
            displacement[d] = atoms[b].xyz[d] + b_translation[d]
                - atoms[a].xyz[d];
        }
        integrals = pm6_sp_sp_pair(
            atoms[a].Z, atoms[b].Z, displacement, *ed_a, *ed_b);
        return true;
    };

    // ---- SCF ----
    // Seed with the spherically averaged neutral-atom valence density:
    // every AO of an atom carries core_charge / n_ao, the MOPAC/NDDO
    // minimal-valence-basis atomic guess, not D = 0.
    //
    // A zero seed makes iteration 1 diagonalise the bare core Hamiltonian,
    // whose Aufbau ordering carries no electron repulsion.  Under the exact
    // heavy-heavy tensor (issue #419) that ordering fills s twice on the
    // expanded article graphene cell, and Pulay extrapolation then converges
    // to a sigma-only self-consistent solution 0.48 Ha above the
    // pi-occupied one found at every smaller scale (measured at scale 1.06:
    // -8.133 Ha against the -8.611 Ha continuing the 0.94 .. 1.04 branch).
    //
    // The seed must be spherically averaged, not shell-ordered: an
    // s-first/p-remainder seed carries the same sigma bias as the bare-core
    // ordering and does not converge at all on scales 1.02 .. 1.06 (100
    // iterations, no fixed point).  With the uniform valence seed the whole
    // sweep converges in 2 iterations on the pi-occupied branch.  The seed
    // selects the branch; it changes no fixed point.
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int a = 0; a < n_atoms; ++a) {
        int n_ao = 0;
        for (int mu = 0; mu < n_basis; ++mu) {
            if (ao_atom[mu] == a) ++n_ao;
        }
        if (n_ao == 0) continue;
        const double per_ao =
            static_cast<double>(core_charge[a]) / static_cast<double>(n_ao);
        for (int mu = 0; mu < n_basis; ++mu) {
            if (ao_atom[mu] == a) D(mu, mu) = per_ao;
        }
    }
    PeriodicPM6Result result;
    result.n_basis = n_basis;
    result.n_occ   = n_occ;
    result.n_cells = n_cells;

    // Pulay DIIS on the physical commutator [F, P]. The NDDO basis is
    // orthogonal, so no overlap factors enter the residual. Direct Aufbau
    // iteration can otherwise settle into an exact charge-transfer
    // two-cycle for polar crystals even though neither density is a
    // stationary SCF solution. Use the shared DIIS implementation rather
    // than an unchecked local bordered solve: it shortens linearly dependent
    // histories and rejects coefficients dominated by cancellation
    // round-off before falling back to the physical Fock matrix.
    DIIS diis(8);

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n_basis, n_basis);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_basis, n_basis);

        // ===========================================================
        // ONE-CENTRE  (only on g=0 cell — same as molecular PM6)
        // ===========================================================
        for (int a = 0; a < n_atoms; ++a) {
            std::vector<int> idx;
            for (int mu = 0; mu < n_basis; ++mu)
                if (ao_atom[mu] == a) idx.push_back(mu);
            int n_ao_a = static_cast<int>(idx.size());

            // --- Diagonal ----
            for (int i = 0; i < n_ao_a; ++i) {
                int mu = idx[i];
                int t_mu = ao_type[mu];

                F(mu, mu) += U_s[mu];
                H(mu, mu) += U_s[mu];

                // Coulomb Σ_ν P_νν (μμ|νν)
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

                // Exchange −½ Σ_ν P_νν (μν|μν)
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

            // --- Off-diagonal one-centre ----
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
        // TWO-CENTRE: lattice-summed over cells
        // ===========================================================
        for (const auto& cell : cells) {
            bool is_zero_cell = (cell.index.array() == 0).all();

            // --- Nuclear attraction (every a,b pair, a≠b in zero cell) ---
            //
            // Electrostatics are truncated per image CELL, not per atom
            // pair (issue #419).  The pair-distance inventory (issue #316)
            // decides which image cells enter the sum; inside an admitted
            // cell every atom pair contributes, so each image always enters
            // as the neutral group it is.  Dropping the pairs of an
            // admitted cell that fall beyond the cutoff leaves that image
            // with a net charge: CO in a 16-bohr cube at cutoff 15 then saw
            // only the cross C-O pairs of its +-x images (13.9 bohr) and
            // not their C-C / O-O partners (16 bohr), over-binding by
            // 12.8 mHa, and in a 10-bohr cube at cutoff 13 the eight
            // half-seen diagonal images drove a charge-transfer runaway
            // (-2.52 Ha).  The core-core sum below uses the same rule so
            // the neutral-pair monopoles cancel term by term.
            for (int a = 0; a < n_atoms; ++a) {
                for (int b = 0; b < n_atoms; ++b) {
                    if (is_zero_cell && a == b) continue;

                    double bx = atoms[b].xyz[0] + cell.r_cart[0];
                    double by = atoms[b].xyz[1] + cell.r_cart[1];
                    double bz = atoms[b].xyz[2] + cell.r_cart[2];
                    double dx = atoms[a].xyz[0] - bx;
                    double dy = atoms[a].xyz[1] - by;
                    double dz = atoms[a].xyz[2] - bz;
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (R < 1e-12) continue;

                    int heavy = -1;
                    int hydrogen = -1;
                    PM6SPHydrogenPair exact_integrals;
                    if (make_exact_sp_hydrogen_pair(
                            a, b, cell.r_cart, heavy, hydrogen,
                            exact_integrals)) {
                        const int x0 = first_ao[heavy];
                        const int h = first_ao[hydrogen];
                        if (a == heavy) {
                            F.block<4, 4>(x0, x0) +=
                                D(h, h) * exact_integrals.electron_repulsion
                                + exact_integrals.heavy_core;
                            H.block<4, 4>(x0, x0) +=
                                exact_integrals.heavy_core;
                        } else {
                            const double j_h =
                                (D.block<4, 4>(x0, x0).cwiseProduct(
                                    exact_integrals.electron_repulsion)).sum();
                            F(h, h) += j_h + exact_integrals.hydrogen_core;
                            H(h, h) += exact_integrals.hydrogen_core;
                        }
                        continue;
                    }

                    // Exact heavy-heavy s/p Coulomb block for every
                    // ordered pair (A, B, g), including non-zero images
                    // (issue #419).  At Gamma the Fock matrix is
                    // F(mu,nu) = sum_g F_{mu,0;nu,g}; the NDDO Coulomb term
                    // of an electron on A in the field of B's image g
                    // (Dewar & Thiel, Theor. Chim. Acta 46, 89 (1977),
                    // Eq. 1 multipole expansion of (mu nu|la si)) is
                    //   J_A(mu,nu) += sum_{la,si in B} P(la,si)
                    //                 (mu_0 nu_0 | la_g si_g)
                    //                 - Z_B (mu_0 nu_0 | s_B,g s_B,g),
                    // and it lands only on A's on-site block.  The
                    // (B, A, -g) record supplies B's block, so the
                    // directed image sum carries no double count here --
                    // exactly the bookkeeping of the collapsed monopole
                    // fallback below, which this block replaces.  The
                    // interatomic exchange block is a separate matter: see
                    // the resonance/exchange loop.
                    PM6SPSPPair exact_sp_sp_integrals;
                    if (make_exact_sp_sp_pair(
                            a, b, cell.r_cart, exact_sp_sp_integrals)) {
                        const int a0 = first_ao[a];
                        const int b0 = first_ao[b];
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
                        F.block<4, 4>(a0, a0) +=
                            j_a + exact_sp_sp_integrals.first_core;
                        H.block<4, 4>(a0, a0) +=
                            exact_sp_sp_integrals.first_core;
                        continue;
                    }

                    double q_B = 0.0;
                    for (int nu = 0; nu < n_basis; ++nu)
                        if (ao_atom[nu] == b) q_B += D(nu, nu);

                    const auto* ed_a = params.element_data(atoms[a].Z);
                    const auto* ed_b = params.element_data(atoms[b].Z);
                    const double g_ab = electronic_gamma_ab(
                        R, ed_a, ed_b, rho[first_ao[a]], rho[first_ao[b]]);

                    const double g_core = params.method_name() == "pm6"
                        ? pm6_electron_core_gamma(R, *ed_a, *ed_b)
                        : g_ab;
                    double V_AB = q_B * g_ab - core_charge[b] * g_core;
                    double V_core = -core_charge[b] * g_core;
                    for (int mu = 0; mu < n_basis; ++mu)
                        if (ao_atom[mu] == a) {
                            F(mu, mu) += V_AB;
                            H(mu, mu) += V_core;
                        }
                }
            }

            // --- Resonance + exchange (a < b in zero cell, all directed
            //     pairs in image cells) ---
            //
            // At Gamma, H(mu,nu) = sum_g H_{mu,0;nu,g}.  Non-zero image
            // blocks are directed: their transpose is supplied by the
            // opposite lattice cell.  Mirroring every directed block here
            // and then visiting -g counts image hopping/exchange twice.
            for (int a = 0; a < n_atoms; ++a) {
                int b0 = (is_zero_cell ? a + 1 : 0);
                for (int b = b0; b < n_atoms; ++b) {
                    if (is_zero_cell && a == b) continue;

                    double bx = atoms[b].xyz[0] + cell.r_cart[0];
                    double by = atoms[b].xyz[1] + cell.r_cart[1];
                    double bz = atoms[b].xyz[2] + cell.r_cart[2];
                    double dx = atoms[a].xyz[0] - bx;
                    double dy = atoms[a].xyz[1] - by;
                    double dz = atoms[a].xyz[2] - bz;
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (R < 1e-12) continue;
                    // Pair-distance cutoff (issue #316;
                    // pair_lattice.hpp (*)).
                    if (R > opts.cutoff_bohr) continue;

                    int heavy = -1;
                    int hydrogen = -1;
                    PM6SPHydrogenPair exact_integrals;
                    const bool exact_sp_hydrogen =
                        make_exact_sp_hydrogen_pair(
                            a, b, cell.r_cart, heavy, hydrogen,
                            exact_integrals);
                    // The exact heavy-heavy exchange block
                    //   K(mu,la) = -1/2 sum_{nu,si} P_{nu,0;si,g}
                    //              (mu_0 nu_0 | la_g si_g)
                    // needs the cell-resolved density P_{nu,0;si,g}.  A
                    // Gamma-only density is the same in every cell,
                    // P_{nu,0;si,g} = D(nu,si), and the monopole part of the
                    // tensor decays only as 1/R, so summing the undamped
                    // block over images diverges with the direct-space
                    // cutoff (measured on CO in a 12-bohr cube, cutoff 15:
                    // -20.086 Ha against the -15.174 Ha molecular limit,
                    // issue #419).  The full tensor therefore fires in the
                    // zero cell only, where P is exact and the block is
                    // mirrored once; non-zero images keep the S^2-damped
                    // monopole exchange model below (issue #419).
                    PM6SPSPPair exact_sp_sp_integrals;
                    const bool exact_sp_sp = is_zero_cell
                        && make_exact_sp_sp_pair(
                            a, b, cell.r_cart, exact_sp_sp_integrals);
                    for (int mu = 0; mu < n_basis; ++mu) {
                        if (ao_atom[mu] != a) continue;
                        int t_mu = ao_type[mu];
                        for (int nu = 0; nu < n_basis; ++nu) {
                            if (ao_atom[nu] != b) continue;
                            int t_nu = ao_type[nu];

                            // n-dependent Slater overlap (nddo_sto_overlap via
                            // MSINDO s2int kernel).  The old 1s-shaped wrappers
                            // gave wrong magnitude and sigma/pi sign for any
                            // 2nd-row (or heavier) atom.
                            const auto* ed_a2 = params.element_data(ao_Z[mu]);
                            const auto* ed_b2 = params.element_data(ao_Z[nu]);
                            double zs_a = ed_a2 ? ed_a2->zs : 1.0;
                            double zp_a = ed_a2 ? ed_a2->zp : 1.0;
                            double zs_b = ed_b2 ? ed_b2->zs : 1.0;
                            double zp_b = ed_b2 ? ed_b2->zp : 1.0;
                            if (zs_a <= 0.0) zs_a = 1.0;
                            if (zp_a <= 0.0) zp_a = zs_a;
                            if (zs_b <= 0.0) zs_b = 1.0;
                            if (zp_b <= 0.0) zp_b = zs_b;
                            int n_a = valence_n_sp(ao_Z[mu]);
                            int n_b = valence_n_sp(ao_Z[nu]);
                            double S_val = nddo_sto_overlap(
                                n_a, t_mu, zs_a, zp_a,
                                n_b, t_nu, zs_b, zp_b,
                                dx, dy, dz, R);

                            // Resonance: β_μν = ½(β_A+β_B)·S_μν
                            double beta_val = 0.5 * (beta[mu] + beta[nu]) * S_val;
                            H(mu, nu) += beta_val;
                            F(mu, nu) += beta_val;
                            if (is_zero_cell) {
                                H(nu, mu) += beta_val;
                                F(nu, mu) += beta_val;
                            }

                            if (exact_sp_hydrogen || exact_sp_sp) {
                                continue;
                            }

                            // Exchange: −½ P_μν γ_AB
                            double eta_ex = 0.5 * (rho[mu] + rho[nu]);
                            double g_ex = 1.0 / std::sqrt(R * R + eta_ex * eta_ex);
                            // A non-zero-cell exchange integral couples the
                            // transition density phi_mu(0) phi_nu(g), whose
                            // monopole scales as S_mu,nu(g). Its self-Coulomb
                            // interaction therefore vanishes as S^2/R. Using
                            // the molecular gamma_AB form without that overlap
                            // factor leaves a spurious -1/R exchange tail at
                            // Gamma and makes a neutral lattice energy diverge
                            // with the direct-space cutoff.
                            if (!is_zero_cell) {
                                g_ex *= S_val * S_val;
                            }
                            double ex_val = -0.5 * D(mu, nu) * g_ex;
                            F(mu, nu) += ex_val;
                            if (is_zero_cell) {
                                F(nu, mu) += ex_val;
                            }
                        }
                    }

                    if (exact_sp_hydrogen) {
                        const int x0 = first_ao[heavy];
                        const int h = first_ao[hydrogen];
                        const Eigen::Vector4d d_xh =
                            D.block<4, 1>(x0, h);
                        const Eigen::Vector4d exchange =
                            -0.5
                            * exact_integrals.electron_repulsion
                            * d_xh;
                        if (a == heavy) {
                            F.block<4, 1>(x0, h) += exchange;
                            if (is_zero_cell) {
                                F.block<1, 4>(h, x0) +=
                                    exchange.transpose();
                            }
                        } else {
                            F.block<1, 4>(h, x0) += exchange.transpose();
                            if (is_zero_cell) {
                                F.block<4, 1>(x0, h) += exchange;
                            }
                        }
                    }
                    if (exact_sp_sp) {
                        const int a0 = first_ao[a];
                        const int b0 = first_ao[b];
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
                        F.block<4, 4>(a0, b0) += exchange;
                        F.block<4, 4>(b0, a0) += exchange.transpose();
                    }
                }
            }
        }

        // The inversion-symmetric image list makes the directed Gamma sum
        // Hermitian analytically. Remove only floating-point antisymmetry
        // before passing the matrix to the self-adjoint eigensolver.
        F = (0.5 * (F + F.transpose())).eval();
        H = (0.5 * (H + H.transpose())).eval();

        // Skip the zero-density first iterate: its commutator is exactly
        // zero and would make the initial DIIS system singular.
        Eigen::MatrixXd err = F * D - D * F;
        double residual = err.cwiseAbs().maxCoeff();
        Eigen::MatrixXd F_eff = F;
        if (iter > 1) {
            F_eff = diis.extrapolate(F, err);
        }

        // ---- Diagonalise ----
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_eff);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Periodic PM6: diagonalisation failed");
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();

        // ---- New density ----
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        Eigen::MatrixXd D_new = 2.0 * C_occ * C_occ.transpose();

        // ---- Convergence ----
        double delta = (D_new - D).cwiseAbs().maxCoeff();
        D = D_new;

        // Density-only convergence can accept a repeated extrapolated Fock
        // that is not a solution of the physical SCF equations. Require the
        // commutator residual from the unextrapolated Fock as well.
        if (delta < opts.conv_tol && residual < opts.conv_tol) {
            // ---- Core-core repulsion (lattice-summed, with diatomic pairs) ----
            double E_core = 0.0;
            for (const auto& cell : cells) {
                bool is_zero_cell = (cell.index.array() == 0).all();
                for (int a = 0; a < n_atoms; ++a) {
                    const auto* ed_a = params.element_data(atoms[a].Z);
                    if (!ed_a) continue;
                    for (int b = (is_zero_cell ? a + 1 : 0); b < n_atoms; ++b) {
                        const auto* ed_b = params.element_data(atoms[b].Z);
                        if (!ed_b) continue;
                        double bx = atoms[b].xyz[0] + cell.r_cart[0];
                        double by = atoms[b].xyz[1] + cell.r_cart[1];
                        double bz = atoms[b].xyz[2] + cell.r_cart[2];
                        double dx = atoms[a].xyz[0] - bx;
                        double dy = atoms[a].xyz[1] - by;
                        double dz = atoms[a].xyz[2] - bz;
                        double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (R < 1e-12) continue;
                        // Every pair of an admitted image cell, matching
                        // the electronic Coulomb sum (issue #419).

                        const auto* dp = params.diatomic(atoms[a].Z, atoms[b].Z);
                        const double pair_weight = is_zero_cell ? 1.0 : 0.5;
                        // Reuse the molecular PM6 expression verbatim.  The
                        // one-half factor applies only to directed non-zero
                        // image pairs; the reverse cell supplies the other
                        // half.  Sharing this function pins the Gamma-only
                        // molecular limit and prevents drift in the neutral
                        // long-range cancellation.
                        const double e_pair = pm6_core_core_repulsion(
                            atoms[a].Z, atoms[b].Z, R,
                            *ed_a, *ed_b, dp);
                        E_core += pair_weight * e_pair;
                    }
                }
            }

            // ---- Total energy ----
            // E_elec = ½ Tr[P·(H + F)]
            double E_elec = 0.0;
            for (int mu = 0; mu < n_basis; ++mu) {
                for (int nu = 0; nu < n_basis; ++nu) {
                    E_elec += D(mu, nu) * (H(mu, nu) + F(mu, nu));
                }
            }
            E_elec *= 0.5;

            double E_total = E_elec + E_core;

            result.energy       = E_total;
            result.e_electronic = E_elec;
            result.e_core       = E_core;
            result.mo_energies  = eps;
            result.mo_coeffs    = C;
            result.density      = D;
            result.fock_gamma   = F;
            result.n_iter       = iter;
            result.converged    = true;
            return result;
        }
    }

    // Did not converge
    result.density = D;
    result.n_iter  = opts.max_iter;
    result.converged = false;
    return result;
}

PeriodicPM6Result run_pm6_gamma(
    const PeriodicSystem& system,
    const PM6ParameterSet& params,
    const PeriodicPM6Options& opts) {
    if (system.unit_cell.empty()) {
        throw std::invalid_argument("Periodic PM6: empty unit cell");
    }
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, opts.cutoff_bohr);
    if (opts.gamma_only_0) {
        cells = {cells[0]};
    }
    return run_pm6_gamma_with_cells(system, params, opts, cells);
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
