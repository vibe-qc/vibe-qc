#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/dispersion.hpp"
#include "vibeqc/eeq_charges.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"

namespace vibeqc {
namespace semiempirical {

namespace {

double gfn2_h0_cov_radius(int Z) {
    // GFN2 H0 has historically consumed the same pre-scaled D3/D4 covalent
    // radii as the H-Ar dispersion table. Preserve that validated H-Ar surface
    // and extend the same 4/3-scaled Pyykko convention with the full EEQ table
    // for post-Ar elements instead of letting D3's NaN sentinel enter H0.
    const double radius = (Z >= 1 && Z <= max_supported_Z())
        ? rcov(Z)
        : (4.0 / 3.0) * eeq_covalent_radius(Z);
    if (!std::isfinite(radius) || radius <= 0.0) {
        throw std::runtime_error(
            "GFN2 H0 coordination number: no finite covalent radius for Z="
            + std::to_string(Z));
    }
    return radius;
}

double gfn2_h0_cn_pair_weight(double R_cov, double R) {
    // R_cov is the sum of gfn2_h0_cov_radius(Z_i)+gfn2_h0_cov_radius(Z_j),
    // which already includes the k2=4/3 scaling absorbed from D3's rcov().
    // xtb's GFN2 H0 CN uses ka=10, kb=20, r_shift=2 with the same
    // pre-scaled radii, so the argument is simply R_cov/R - 1.
    const double x1 = R_cov / R - 1.0;
    const double x2 = (R_cov + 2.0) / R - 1.0;
    const double f1 = 1.0 / (1.0 + std::exp(-10.0 * x1));
    const double f2 = 1.0 / (1.0 + std::exp(-20.0 * x2));
    return f1 * f2;
}

double gfn2_h0_cn_pair_derivative(double R_cov, double R) {
    const double x1 = R_cov / R - 1.0;
    const double x2 = (R_cov + 2.0) / R - 1.0;
    const double f1 = 1.0 / (1.0 + std::exp(-10.0 * x1));
    const double f2 = 1.0 / (1.0 + std::exp(-20.0 * x2));
    const double df1 = 10.0 * f1 * (1.0 - f1)
        * (-R_cov / (R * R));
    const double df2 = 20.0 * f2 * (1.0 - f2)
        * (-(R_cov + 2.0) / (R * R));
    return df1 * f2 + f1 * df2;
}

// Pauling electronegativity for the GFN2 H0 off-site EN factor
// (1 + 0.02 * dEN^2). Defined further down in this file; the energy path
// above it shares the single table with the derivative path.
double gfn2_h0_en_pauling(int Z);

}  // namespace

double gfn2_h0_coordination_pair_contribution(
    int Z_a, int Z_b, double distance_bohr) {
    constexpr double kCoordinationCutoffBohr = 15.0;
    if (!std::isfinite(distance_bohr) || distance_bohr <= 1.0e-12) {
        throw std::invalid_argument(
            "GFN2 H0 coordination pair distance must be finite and positive");
    }
    if (distance_bohr > kCoordinationCutoffBohr) {
        return 0.0;
    }
    const double R_cov = gfn2_h0_cov_radius(Z_a)
        + gfn2_h0_cov_radius(Z_b);
    return gfn2_h0_cn_pair_weight(R_cov, distance_bohr);
}

Eigen::VectorXd gfn2_h0_coordination_numbers(const Molecule& mol) {
    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());
    Eigen::VectorXd cn = Eigen::VectorXd::Zero(n_atoms);
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = 0; b < n_atoms; ++b) {
            if (a == b) continue;
            const double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            const double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            const double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
            cn(a) += gfn2_h0_coordination_pair_contribution(
                atoms[a].Z, atoms[b].Z, R);
        }
    }
    return cn;
}

Eigen::VectorXd gfn2_h0_periodic_coordination_numbers(
    const PeriodicSystem& system) {
    constexpr double kCoordinationCutoffBohr = 15.0;
    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());
    Eigen::VectorXd cn = Eigen::VectorXd::Zero(n_atoms);
    Eigen::VectorXd compensation = Eigen::VectorXd::Zero(n_atoms);

    // A physical pair can be within the CN cutoff even when its lattice
    // translation lies outside the origin-centred cutoff sphere. Pad by the
    // largest intra-cell atom offset, then filter on the actual pair distance
    // below (the same pair-complete construction as lattice_pair_cells.hpp).
    double max_atom_offset = 0.0;
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = 0; b < n_atoms; ++b) {
            const double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            const double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            const double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            max_atom_offset = std::max(
                max_atom_offset, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    const std::vector<LatticeCell> cells = direct_lattice_cells(
        system, kCoordinationCutoffBohr + max_atom_offset);
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = 0; b < n_atoms; ++b) {
            for (const auto& cell : cells) {
                const bool home_cell = (cell.index.array() == 0).all();
                if (home_cell && a == b) continue;
                const double dx = atoms[a].xyz[0]
                    - (atoms[b].xyz[0] + cell.r_cart[0]);
                const double dy = atoms[a].xyz[1]
                    - (atoms[b].xyz[1] + cell.r_cart[1]);
                const double dz = atoms[a].xyz[2]
                    - (atoms[b].xyz[2] + cell.r_cart[2]);
                const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (R > kCoordinationCutoffBohr) continue;
                if (!(R > 1.0e-12)) {
                    // Two distinct atoms (or an atom and its own image)
                    // coincide modulo the declared lattice.  Name the pair
                    // and the image instead of failing on the generic
                    // pair-distance guard: on anisotropic supercells this
                    // is the signature of a lattice handed to
                    // PeriodicSystem as ROWS (issue #408 reproduced the fcc
                    // Cu 2x2x4 refusal exactly that way; the lattice
                    // vectors are COLUMNS).
                    throw std::invalid_argument(
                        "GFN2 periodic H0 coordination: atoms "
                        + std::to_string(a) + " and " + std::to_string(b)
                        + " coincide modulo the lattice (image "
                        + std::to_string(cell.index[0]) + ","
                        + std::to_string(cell.index[1]) + ","
                        + std::to_string(cell.index[2])
                        + "); distinct atoms must not be lattice-equivalent."
                          " PeriodicSystem takes lattice vectors as COLUMNS;"
                          " a transposed (row-fed) anisotropic supercell"
                          " produces exactly this coincidence");
                }
                const double contribution =
                    gfn2_h0_coordination_pair_contribution(
                        atoms[a].Z, atoms[b].Z, R);
                const double corrected = contribution - compensation(a);
                const double updated = cn(a) + corrected;
                compensation(a) = (updated - cn(a)) - corrected;
                cn(a) = updated;
            }
        }
    }
    return cn;
}

// ---------------------------------------------------------------------------
// DFTB Hamiltonian builder — delegates to existing implementation
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_dftb_hamiltonian_zero(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    return SemiempiricalHamiltonianBuilder::build_hamiltonian_zero(
        basis, S, mol, params);
}

// ---------------------------------------------------------------------------
// DFTB gamma — delegates to existing implementation
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_dftb_gamma(
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    return SemiempiricalHamiltonianBuilder::gamma_matrix(mol, params);
}

// ---------------------------------------------------------------------------
// GFN2-xTB Hamiltonian builder — electronegativity-based
//
// H⁰_{μμ} = EN_A(l) + k_EN * Γ_AA * q_A  (on-site)
// H⁰_{μν} = 0.5 * k_ll' * (H⁰_{μμ} + H⁰_{νν}) * S_{μν}  (off-site, A=B)
// H⁰_{μν} = 0.5 * S_{μν} * (k_{EN,A} + k_{EN,B})   (off-site, A≠B, simplified)
//          * (H⁰_{μμ}/k_{EN,A} + H⁰_{νν}/k_{EN,B})
//
// This is a simplified version.  The full GFN2-xTB Hamiltonian
// includes multipole electrostatic and exchange contributions
// that will be added in subsequent iterations.
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_gfn2_hamiltonian_zero(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params) {
    return build_gfn2_hamiltonian_zero_image(
        basis, S, mol, params, Eigen::Vector3d::Zero());
}

Eigen::MatrixXd build_gfn2_hamiltonian_zero_image(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const Eigen::Vector3d& image_shift) {

    return build_gfn2_hamiltonian_zero_image_with_cn(
        basis, S, mol, params, image_shift,
        gfn2_h0_coordination_numbers(mol));
}

Eigen::MatrixXd build_gfn2_hamiltonian_zero_image_with_cn(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const Eigen::Vector3d& image_shift,
    const Eigen::VectorXd& coordination_numbers) {

    // The molecular caller remains paired with
    // gfn2_h0_image_derivative_terms below. Periodic and WS callers inject a
    // boundary-specific CN vector; their derivative path must consume that
    // same vector before it can claim parity (tracked by GitLab issue #338).

    const int n_basis = static_cast<int>(basis.nbasis());
    const int n_atoms = static_cast<int>(mol.atoms().size());
    Eigen::MatrixXd H0 = Eigen::MatrixXd::Zero(n_basis, n_basis);

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto& basis_shells = basis.shells();
    const auto& atoms = mol.atoms();

    // Map AOs to atoms and shells
    struct AoInfo {
        int atom_idx;
        int l;
        double zeta;
    };
    std::vector<AoInfo> ao_map(n_basis);
    for (int s = 0; s < static_cast<int>(basis_shells.size()); ++s) {
        int n_funcs = shells[s].size();
        int bf_start = shell2bf[s];
        int l = basis_shells[s].l;
        int atom_idx = basis_shells[s].atom_index;
        double zeta = 0.0;
        const auto* elem = params.element_data(atoms[atom_idx].Z);
        if (elem) {
            for (const auto& sh : elem->shells) {
                if (sh.l == l) {
                    zeta = sh.zeta;
                    break;
                }
            }
        }
        for (int i = 0; i < n_funcs; ++i) {
            ao_map[bf_start + i] = {atom_idx, l, zeta};
        }
    }

    if (coordination_numbers.size() != n_atoms
        || !coordination_numbers.allFinite()
        || (coordination_numbers.array() < 0.0).any()) {
        throw std::invalid_argument(
            "GFN2 H0 coordination vector must be finite, nonnegative, and "
            "match the atom count");
    }
    const Eigen::VectorXd& cn = coordination_numbers;

    // ---- Per-element atomic radius (CRC, bohr) and Pauling EN ----
    // Matches xtb's atomicRad table (CRC Handbook, 91st ed., 2010).
    // Used in the shell distance polynomial: rr = √(R_AB/(rad_A+rad_B)).
    // Distinct from p_rad (multipole radius table in GFN2 parameters).
    static const double atomic_rad[] = {
        0.0,        // Z=0
        0.604712,   // H  0.32 Å
        0.699199,   // He 0.37 Å
        2.456644,   // Li 1.30 Å
        1.870829,   // Be 0.99 Å
        1.587370,   // B  0.84 Å
        1.417295,   // C  0.75 Å
        1.341706,   // N  0.71 Å
        1.209425,   // O  0.64 Å
        1.133836,   // F  0.60 Å
        1.171630,   // Ne 0.62 Å
        3.023561,   // Na 1.60 Å
        2.645616,   // Mg 1.40 Å
        2.343260,   // Al 1.24 Å
        2.154287,   // Si 1.14 Å
        2.059800,   // P  1.09 Å
        1.965314,   // S  1.04 Å
        1.889726,   // Cl 1.00 Å
        1.908623,   // Ar 1.01 Å
        3.779452,   // K  2.00 Å
        3.288123,   // Ca 1.74 Å
        3.004664,   // Sc 1.59 Å
        2.796794,   // Ti 1.48 Å
        2.721205,   // V  1.44 Å
        2.456644,   // Cr 1.30 Å
        2.437746,   // Mn 1.29 Å
        2.343260,   // Fe 1.24 Å
        2.229877,   // Co 1.18 Å
        2.210979,   // Ni 1.17 Å
        2.305466,   // Cu 1.22 Å
        2.267671,   // Zn 1.20 Å
        2.324363,   // Ga 1.23 Å
        2.267671,   // Ge 1.20 Å
        2.267671,   // As 1.20 Å
        2.229877,   // Se 1.18 Å
        2.210979,   // Br 1.17 Å
        2.192082,   // Kr 1.16 Å
        4.062822,   // Rb 2.15 Å
        3.590480,   // Sr 1.90 Å
        3.325918,   // Y  1.76 Å
        3.099151,   // Zr 1.64 Å
        2.947972,   // Nb 1.56 Å
        2.758999,   // Mo 1.46 Å
        2.607822,   // Tc 1.38 Å
        2.570027,   // Ru 1.36 Å
        2.532233,   // Rh 1.34 Å
        2.456644,   // Pd 1.30 Å
        2.570027,   // Ag 1.36 Å
        2.645616,   // Cd 1.40 Å
        2.759000,   // In 1.46 Å
        2.683411,   // Sn 1.42 Å
        2.645616,   // Sb 1.40 Å
        2.570027,   // Te 1.36 Å
        2.494438,   // I  1.32 Å
        2.456644,   // Xe 1.30 Å
        4.440857,   // Cs 2.35 Å
        3.968334,   // Ba 2.10 Å
        3.533787,   // La 1.87 Å
        3.458198,   // Ce 1.83 Å
        3.439301,   // Pr 1.82 Å
        3.458198,   // Nd 1.83 Å
        3.420404,   // Pm 1.81 Å
        3.382609,   // Sm 1.79 Å
        3.779452,   // Eu 2.00 Å
        3.363712,   // Gd 1.78 Å
        3.288123,   // Tb 1.74 Å
        3.325918,   // Dy 1.76 Å
        3.212534,   // Ho 1.70 Å
        3.193637,   // Er 1.69 Å
        3.136945,   // Tm 1.66 Å
        3.250329,   // Yb 1.72 Å
        3.212534,   // Lu 1.70 Å
        2.985767,   // Hf 1.58 Å
        2.796794,   // Ta 1.48 Å
        2.645616,   // W  1.40 Å
        2.551130,   // Re 1.35 Å
        2.513335,   // Os 1.33 Å
        2.475541,   // Ir 1.31 Å
        2.456644,   // Pt 1.30 Å
        2.381055,   // Au 1.26 Å
        2.418849,   // Hg 1.28 Å
        2.740102,   // Tl 1.45 Å
        2.759000,   // Pb 1.46 Å
        2.796794,   // Bi 1.48 Å
        2.645616,   // Po 1.40 Å
        2.834589,   // At 1.50 Å
        2.834589,   // Rn 1.50 Å
    };
    static constexpr int atomic_rad_max = sizeof(atomic_rad)/sizeof(atomic_rad[0]) - 1;
    auto get_atomic_rad = [](int Z) {
        if (Z >= 1 && Z <= atomic_rad_max) return atomic_rad[Z];
        return 3.0;  // fallback for superheavy elements
    };

    // ---- Wolfsberg-Helmholtz K_{ll'} — from the shared inline helper ----

    // Precompute diagonal H⁰ values before any off-diagonal access.
    std::vector<double> h_diag(n_basis, 0.0);
    std::vector<int> ao_l(n_basis, 0);
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_map[mu].atom_idx;
        int l_mu = ao_map[mu].l;
        int Z_mu = atoms[a_mu].Z;
        ao_l[mu] = l_mu;

        const auto* elem_mu = params.element_data(Z_mu);
        if (!elem_mu) continue;

        double en_mu = 0.0, kcn_mu = 0.0;
        for (const auto& sh : elem_mu->shells) {
            if (sh.l == l_mu) {
                en_mu = sh.en;
                kcn_mu = sh.kcn;
                break;
            }
        }

        // Self-energy: h_l(A) = EN_l(A) − kcn_l(A)·CN_A
        h_diag[mu] = en_mu - kcn_mu * cn(a_mu);
        H0(mu, mu) = h_diag[mu] * S(mu, mu);
    }

    const bool home_block = image_shift.norm() < 1e-14;

    // Build off-diagonals with H0 shape terms.  Nonzero image blocks are not
    // symmetric in AO order (H(g) = H(-g)^T), so every AO pair is evaluated.
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_map[mu].atom_idx;
        int l_mu = ao_l[mu];
        int Z_mu = atoms[a_mu].Z;
        double en_mu = gfn2_h0_en_pauling(Z_mu);
        double rad_mu = get_atomic_rad(Z_mu);
        double poly_mu = 0.0;
        {
            const auto* e = params.element_data(Z_mu);
            if (e) for (const auto& sh : e->shells)
                if (sh.l == l_mu) { poly_mu = sh.poly; break; }
        }

        const int nu_start = home_block ? mu + 1 : 0;
        for (int nu = nu_start; nu < n_basis; ++nu) {
            int a_nu = ao_map[nu].atom_idx;
            int l_nu = ao_l[nu];
            int Z_nu = atoms[a_nu].Z;

            double s_val = S(mu, nu);
            if (std::abs(s_val) < 1e-15) continue;

            double k = gfn2_kshell(l_mu, l_nu,
                params.element_data(Z_mu) ? params.element_data(Z_mu)->gam3 : 0.0,
                params.element_data(Z_nu) ? params.element_data(Z_nu)->gam3 : 0.0);
            double h_base = 0.5 * k * s_val * (h_diag[mu] + h_diag[nu]);

            // Eq. 16 Slater-exponent compactness factor.  Equal exponents
            // give one; unlike compactness reduces the EHT coupling.
            double zeta_factor = 1.0;
            const double zeta_mu = ao_map[mu].zeta;
            const double zeta_nu = ao_map[nu].zeta;
            if (zeta_mu > 0.0 && zeta_nu > 0.0) {
                zeta_factor = std::sqrt(
                    2.0 * std::sqrt(zeta_mu * zeta_nu) / (zeta_mu + zeta_nu));
            }

            // EN factor (A≠B only): (1 + 0.02·ΔEN²)
            double en_factor = 1.0;
            const bool intercentre = !home_block || a_mu != a_nu;
            if (intercentre) {
                double en_nu = gfn2_h0_en_pauling(Z_nu);
                double dEN = en_mu - en_nu;
                en_factor = 1.0 + 0.02 * dEN * dEN;
            }

            // Shell distance-polynomial (A≠B only):
            //   Π = (1 + p_l(A)·rr)(1 + p_l'(B)·rr), rr = √(R_AB / (rad_A + rad_B))
            double pi_factor = 1.0;
            if (intercentre) {
                double dx = atoms[a_mu].xyz[0] - (atoms[a_nu].xyz[0] + image_shift[0]);
                double dy = atoms[a_mu].xyz[1] - (atoms[a_nu].xyz[1] + image_shift[1]);
                double dz = atoms[a_mu].xyz[2] - (atoms[a_nu].xyz[2] + image_shift[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                double rad_nu = get_atomic_rad(Z_nu);
                double rr = std::sqrt(R / (rad_mu + rad_nu));
                double poly_nu = 0.0;
                {
                    const auto* e = params.element_data(Z_nu);
                    if (e) for (const auto& sh : e->shells)
                        if (sh.l == l_nu) { poly_nu = sh.poly; break; }
                }
                pi_factor = (1.0 + poly_mu * rr) * (1.0 + poly_nu * rr);
            }

            H0(mu, nu) = h_base * zeta_factor * en_factor * pi_factor;
        }
    }

    if (home_block) {
        for (int mu = 0; mu < n_basis; ++mu) {
            for (int nu = mu + 1; nu < n_basis; ++nu) {
                H0(nu, mu) = H0(mu, nu);
            }
        }
    }

    return H0;
}

// ---------------------------------------------------------------------------
// GFN2-xTB H⁰ geometry derivatives.
//
// Mirrors build_gfn2_hamiltonian_zero_image factor-for-factor (same CN', the
// same per-element atomic radius / Pauling-EN tables, the same |S| < 1e-15 skip) so
// the derivative cannot drift from the energy expression.  Keep the two in
// sync when touching either.
// ---------------------------------------------------------------------------

namespace {

// Per-element tables shared with build_gfn2_hamiltonian_zero_image.
// CRC atomic radii (bohr), matching xtb's atomicRad table.
inline double gfn2_h0_atomic_rad(int Z) {
    static const double rad[] = {
        0.0, 0.604712, 0.699199, 2.456644, 1.870829, 1.587370,
        1.417295, 1.341706, 1.209425, 1.133836, 1.171630,
        3.023561, 2.645616, 2.343260, 2.154287, 2.059800,
        1.965314, 1.889726, 1.908623, 3.779452, 3.288123,
        3.004664, 2.796794, 2.721205, 2.456644, 2.437746,
        2.343260, 2.229877, 2.210979, 2.305466, 2.267671,
        2.324363, 2.267671, 2.267671, 2.229877, 2.210979,
        2.192082, 4.062822, 3.590480, 3.325918, 3.099151,
        2.947972, 2.758999, 2.607822, 2.570027, 2.532233,
        2.456644, 2.570027, 2.645616, 2.759000, 2.683411,
        2.645616, 2.570027, 2.494438, 2.456644, 4.440857,
        3.968334, 3.533787, 3.458198, 3.439301, 3.458198,
        3.420404, 3.382609, 3.779452, 3.363712, 3.288123,
        3.325918, 3.212534, 3.193637, 3.136945, 3.250329,
        3.212534, 2.985767, 2.796794, 2.645616, 2.551130,
        2.513335, 2.475541, 2.456644, 2.381055, 2.418849,
        2.740102, 2.759000, 2.796794, 2.645616, 2.834589, 2.834589,
    };
    static constexpr int max_z = sizeof(rad)/sizeof(rad[0]) - 1;
    return (Z >= 1 && Z <= max_z) ? rad[Z] : 3.0;
}
inline double gfn2_h0_en_pauling(int Z) {
    // Pauling electronegativities, the full mctc-lib set (Z <= 86), the
    // same values tblite consumes through get_pauling_en (issue #433).
    // The table behind the GFN2 H0 off-site EN factor used to stop at
    // Z = 10 and returned 1.0 for every heavier element, so heteronuclear
    // pairs involving Na or beyond computed dEN = 0 and a flat en_factor
    // instead of 1 + 0.02 * dEN^2.
    static const double pauling_en[87] = {
        0.0,        // Z=0 (unused)
        2.20,       // H
        3.00,       // He
        0.98,       // Li
        1.57,       // Be
        2.04,       // B
        2.55,       // C
        3.04,       // N
        3.44,       // O
        3.98,       // F
        4.50,       // Ne
        0.93,       // Na
        1.31,       // Mg
        1.61,       // Al
        1.90,       // Si
        2.19,       // P
        2.58,       // S
        3.16,       // Cl
        3.50,       // Ar
        0.82,       // K
        1.00,       // Ca
        1.36,       // Sc
        1.54,       // Ti
        1.63,       // V
        1.66,       // Cr
        1.55,       // Mn
        1.83,       // Fe
        1.88,       // Co
        1.91,       // Ni
        1.90,       // Cu
        1.65,       // Zn
        1.81,       // Ga
        2.01,       // Ge
        2.18,       // As
        2.55,       // Se
        2.96,       // Br
        3.00,       // Kr
        0.82,       // Rb
        0.95,       // Sr
        1.22,       // Y
        1.33,       // Zr
        1.60,       // Nb
        2.16,       // Mo
        1.90,       // Tc
        2.20,       // Ru
        2.28,       // Rh
        2.20,       // Pd
        1.93,       // Ag
        1.69,       // Cd
        1.78,       // In
        1.96,       // Sn
        2.05,       // Sb
        2.10,       // Te
        2.66,       // I
        2.60,       // Xe
        0.79,       // Cs
        0.89,       // Ba
        1.10,       // La
        1.12,       // Ce
        1.13,       // Pr
        1.14,       // Nd
        1.15,       // Pm
        1.17,       // Sm
        1.18,       // Eu
        1.20,       // Gd
        1.21,       // Tb
        1.22,       // Dy
        1.23,       // Ho
        1.24,       // Er
        1.25,       // Tm
        1.26,       // Yb
        1.27,       // Lu
        1.30,       // Hf
        1.50,       // Ta
        2.36,       // W
        1.90,       // Re
        2.20,       // Os
        2.20,       // Ir
        2.28,       // Pt
        2.54,       // Au
        2.00,       // Hg
        1.62,       // Tl
        2.33,       // Pb
        2.02,       // Bi
        2.00,       // Po
        2.20,       // At
        2.20,       // Rn
    };
    return (Z >= 1 && Z <= 86) ? pauling_en[Z] : 1.0;
}

}  // namespace

Eigen::MatrixXd gfn2_h0_image_derivative_terms(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& D,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const Eigen::Vector3d& image_shift,
    const Eigen::VectorXd& coordination_numbers,
    Eigen::VectorXd& dE_dCN,
    Eigen::MatrixXd& grad_poly,
    Eigen::Matrix3d& strain_poly) {

    const int n_basis = static_cast<int>(basis.nbasis());
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto& basis_shells = basis.shells();
    const auto& atoms = mol.atoms();

    // AO → (atom, l, zeta, kcn, poly) map, as in the H⁰ builder.
    struct AoInfo { int atom_idx; int l; double zeta; double kcn; double poly; };
    std::vector<AoInfo> ao_map(n_basis);
    for (int s = 0; s < static_cast<int>(basis_shells.size()); ++s) {
        int n_funcs = shells[s].size();
        int bf_start = shell2bf[s];
        int l = basis_shells[s].l;
        int atom_idx = basis_shells[s].atom_index;
        double zeta = 0.0, kcn = 0.0, poly = 0.0;
        const auto* elem = params.element_data(atoms[atom_idx].Z);
        if (elem) {
            for (const auto& sh : elem->shells) {
                if (sh.l == l) { zeta = sh.zeta; kcn = sh.kcn; poly = sh.poly; break; }
            }
        }
        for (int i = 0; i < n_funcs; ++i)
            ao_map[bf_start + i] = {atom_idx, l, zeta, kcn, poly};
    }

    if (coordination_numbers.size() != static_cast<Eigen::Index>(atoms.size())
        || !coordination_numbers.allFinite()
        || (coordination_numbers.array() < 0.0).any()) {
        throw std::invalid_argument(
            "GFN2 H0 derivative coordination vector must be finite, "
            "nonnegative, and match the atom count");
    }
    const Eigen::VectorXd& cn = coordination_numbers;

    // Self-energies h_μ = EN_l(A) − k_CN,l(A)·CN'_A.
    std::vector<double> h_diag(n_basis, 0.0);
    for (int mu = 0; mu < n_basis; ++mu) {
        const auto* elem = params.element_data(atoms[ao_map[mu].atom_idx].Z);
        if (!elem) continue;
        double en_mu = 0.0;
        for (const auto& sh : elem->shells)
            if (sh.l == ao_map[mu].l) { en_mu = sh.en; break; }
        h_diag[mu] = en_mu - ao_map[mu].kcn * cn(ao_map[mu].atom_idx);
    }

    const bool home_block = image_shift.norm() < 1e-14;
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n_basis, n_basis);

    for (int mu = 0; mu < n_basis; ++mu) {
        const int a_mu = ao_map[mu].atom_idx;
        const int l_mu = ao_map[mu].l;
        const int Z_mu = atoms[a_mu].Z;
        const double en_p_mu = gfn2_h0_en_pauling(Z_mu);
        const double rad_mu = gfn2_h0_atomic_rad(Z_mu);
        const double poly_mu = ao_map[mu].poly;

        if (home_block) {
            // Diagonal H⁰_{μμ} = h_μ·S_{μμ}: S_{μμ} is a normalization
            // constant (dS_{μμ}/dR = 0), so only the CN chain through h_μ
            // contributes.
            F(mu, mu) = h_diag[mu];
            dE_dCN(a_mu) += D(mu, mu) * S(mu, mu) * (-ao_map[mu].kcn);
        }

        const int nu_start = home_block ? mu + 1 : 0;
        for (int nu = nu_start; nu < n_basis; ++nu) {
            const int a_nu = ao_map[nu].atom_idx;
            const int l_nu = ao_map[nu].l;
            const int Z_nu = atoms[a_nu].Z;

            const double s_val = S(mu, nu);
            // NOTE: unlike the H⁰ builder (which may zero H⁰_μν = base·S_μν
            // for |S| < 1e-15 — a continuous clamp, since base·S → 0 as
            // S → 0), the derivative F_μν = ∂H⁰/∂S = base does NOT vanish at
            // S = 0.  Skipping such pairs here (as the builder does) drops a
            // real force term whenever an overlap is exactly zero by symmetry
            // (e.g. an O–H bond aligned with a global axis: S(p_y, s_H) = 0
            // but dS(p_y, s_H)/dR ≠ 0), producing a discrete jump in the
            // analytic gradient at the aligned geometry (~2.7e-2 Ha/bohr on
            // water, GFN2-GRADIENT-FD-RESIDUAL).  Keep every pair; the
            // s_val-weighted chains below (dE_dCN, grad_poly) vanish
            // continuously at s_val = 0 on their own.

            const double k = gfn2_kshell(l_mu, l_nu,
                params.element_data(Z_mu) ? params.element_data(Z_mu)->gam3 : 0.0,
                params.element_data(Z_nu) ? params.element_data(Z_nu)->gam3 : 0.0);

            double zeta_factor = 1.0;
            const double zeta_mu = ao_map[mu].zeta;
            const double zeta_nu = ao_map[nu].zeta;
            if (zeta_mu > 0.0 && zeta_nu > 0.0) {
                zeta_factor = std::sqrt(
                    2.0 * std::sqrt(zeta_mu * zeta_nu) / (zeta_mu + zeta_nu));
            }

            double en_factor = 1.0;
            const bool intercentre = !home_block || a_mu != a_nu;
            if (intercentre) {
                double dEN = en_p_mu - gfn2_h0_en_pauling(Z_nu);
                en_factor = 1.0 + 0.02 * dEN * dEN;
            }

            double pi_factor = 1.0;
            double dpi_dR = 0.0;
            double dx = 0.0, dy = 0.0, dz = 0.0, R = 0.0;
            if (intercentre) {
                dx = atoms[a_mu].xyz[0] - (atoms[a_nu].xyz[0] + image_shift[0]);
                dy = atoms[a_mu].xyz[1] - (atoms[a_nu].xyz[1] + image_shift[1]);
                dz = atoms[a_mu].xyz[2] - (atoms[a_nu].xyz[2] + image_shift[2]);
                R = std::sqrt(dx*dx + dy*dy + dz*dz);
                double rad_nu = gfn2_h0_atomic_rad(Z_nu);
                double rr = std::sqrt(R / (rad_mu + rad_nu));
                double poly_nu = 0.0;
                {
                    const auto* e = params.element_data(Z_nu);
                    if (e) for (const auto& sh : e->shells)
                        if (sh.l == l_nu) { poly_nu = sh.poly; break; }
                }
                pi_factor = (1.0 + poly_mu * rr) * (1.0 + poly_nu * rr);
                // dΠ/dR with drr/dR = rr/(2R):
                if (R > 1e-12) {
                    dpi_dR = (poly_mu * (1.0 + poly_nu * rr)
                              + poly_nu * (1.0 + poly_mu * rr)) * rr / (2.0 * R);
                }
            }

            const double base = 0.5 * k * zeta_factor * en_factor * pi_factor;
            F(mu, nu) = base * (h_diag[mu] + h_diag[nu]);
            if (home_block) F(nu, mu) = F(mu, nu);

            // The home-block μ<ν loop covers each ordered pair once; the
            // energy Σ_{μν} D·H⁰ counts both orders, hence weight 2.
            const double weight = home_block ? 2.0 : 1.0;

            // CN chain: ∂H⁰_{μν}/∂h = ½ K S X_ζ EN Π, ∂h_μ/∂CN'_A = −k_CN,μ.
            const double dE_dh = weight * D(mu, nu) * base * s_val;
            dE_dCN(a_mu) += dE_dh * (-ao_map[mu].kcn);
            dE_dCN(a_nu) += dE_dh * (-ao_map[nu].kcn);

            // Distance-polynomial pair force.
            if (intercentre && R > 1e-12 && dpi_dR != 0.0) {
                const double dH0_dR = weight * D(mu, nu)
                    * 0.5 * k * zeta_factor * en_factor
                    * s_val * (h_diag[mu] + h_diag[nu]) * dpi_dR;
                const double invR = 1.0 / R;
                grad_poly(a_mu, 0) += dH0_dR * dx * invR;
                grad_poly(a_mu, 1) += dH0_dR * dy * invR;
                grad_poly(a_mu, 2) += dH0_dR * dz * invR;
                grad_poly(a_nu, 0) -= dH0_dR * dx * invR;
                grad_poly(a_nu, 1) -= dH0_dR * dy * invR;
                grad_poly(a_nu, 2) -= dH0_dR * dz * invR;
                const double dR[3] = {dx, dy, dz};
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        strain_poly(i, j) += dH0_dR * dR[i] * dR[j] * invR;
            }
        }
    }

    return F;
}

Eigen::MatrixXd gfn2_cn_chain_gradient(
    const Molecule& mol,
    const Eigen::VectorXd& dE_dCN) {

    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);
    const double cn_cutoff = 15.0;

    for (int a = 0; a < n_atoms; ++a) {
        for (int b = a + 1; b < n_atoms; ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R > cn_cutoff || R < 1e-12) continue;

            const double R_cov = gfn2_h0_cov_radius(atoms[a].Z)
                + gfn2_h0_cov_radius(atoms[b].Z);
            const double dfdR = gfn2_h0_cn_pair_derivative(R_cov, R);

            // The pair term enters both CN'_a and CN'_b symmetrically.
            double w = (dE_dCN(a) + dE_dCN(b)) * dfdR / R;
            grad(a, 0) += w * dx;
            grad(a, 1) += w * dy;
            grad(a, 2) += w * dz;
            grad(b, 0) -= w * dx;
            grad(b, 1) -= w * dy;
            grad(b, 2) -= w * dz;
        }
    }
    return grad;
}

GFN2PeriodicCNDerivatives gfn2_periodic_cn_chain_derivatives(
    const PeriodicSystem& system,
    const Eigen::VectorXd& dE_dCN) {

    constexpr double kCoordinationCutoffBohr = 15.0;
    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());
    if (dE_dCN.size() != n_atoms) {
        throw std::invalid_argument(
            "periodic GFN2 H0 CN derivative must match the atom count");
    }

    GFN2PeriodicCNDerivatives out;
    out.atomic_gradient = Eigen::MatrixXd::Zero(n_atoms, 3);

    double max_atom_offset = 0.0;
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = 0; b < n_atoms; ++b) {
            const Eigen::Vector3d dR(
                atoms[a].xyz[0] - atoms[b].xyz[0],
                atoms[a].xyz[1] - atoms[b].xyz[1],
                atoms[a].xyz[2] - atoms[b].xyz[2]);
            max_atom_offset = std::max(max_atom_offset, dR.norm());
        }
    }
    const auto cells = direct_lattice_cells(
        system, kCoordinationCutoffBohr + max_atom_offset);

    // Bannwarth, Ehlert & Grimme, JCTC 2019, Eqs. 17-19: H0 depends on
    // the directed Eq. 18 coordination number of its centre.  Differentiate
    // the exact atom/image list used by gfn2_h0_periodic_coordination_numbers.
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = 0; b < n_atoms; ++b) {
            for (const auto& cell : cells) {
                const bool home_cell = (cell.index.array() == 0).all();
                if (home_cell && a == b) continue;
                const Eigen::Vector3d dR(
                    atoms[a].xyz[0] - (atoms[b].xyz[0] + cell.r_cart[0]),
                    atoms[a].xyz[1] - (atoms[b].xyz[1] + cell.r_cart[1]),
                    atoms[a].xyz[2] - (atoms[b].xyz[2] + cell.r_cart[2]));
                const double R = dR.norm();
                if (R <= 1.0e-12 || R > kCoordinationCutoffBohr) continue;
                const double R_cov = gfn2_h0_cov_radius(atoms[a].Z)
                    + gfn2_h0_cov_radius(atoms[b].Z);
                const double scale = dE_dCN(a)
                    * gfn2_h0_cn_pair_derivative(R_cov, R) / R;
                const Eigen::Vector3d contribution = scale * dR;
                out.atomic_gradient.row(a) += contribution.transpose();
                out.atomic_gradient.row(b) -= contribution.transpose();
                out.strain_derivative += scale * dR * dR.transpose();
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// GFN2-xTB gamma — Klopman-Ohno with Gaussian damping
//
// γ_{AB} = 1 / sqrt(R_AB² + η_AB²)  (same Ohno-Klopman form as DFTB
// but with element-specific hardness parameters from the GFN2 set)
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_gfn2_gamma(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params) {

    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index a = 0; a < n; ++a) {
        const auto* elem_a = params.element_data(atoms[a].Z);
        double Ua = elem_a ? elem_a->gam : 0.5;
        gamma(a, a) = Ua;

        for (Eigen::Index b = a + 1; b < n; ++b) {
            const auto* elem_b = params.element_data(atoms[b].Z);
            double Ub = elem_b ? elem_b->gam : 0.5;

            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);

            // GFN2's effective Coulomb kernel uses the arithmetic average of
            // shell/atomic hardnesses with gexp=2:
            //   gamma_AB = 1 / sqrt(R_AB^2 + gamma_ab^{-2})
            // where gamma_ab = 0.5 * (gamma_a + gamma_b).
            const double gab = 0.5 * (Ua + Ub);
            double g = 1.0 / std::sqrt(R*R + 1.0 / (gab * gab));

            gamma(a, b) = g;
            gamma(b, a) = g;
        }
    }
    return gamma;
}

// ---------------------------------------------------------------------------
// Shell enumeration for GFN2-xTB
// ---------------------------------------------------------------------------

std::vector<GFN2ShellInfo> gfn2_enumerate_shells(
    const BasisSet& basis,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params) {

    std::vector<GFN2ShellInfo> shells;
    const auto& basis_shells = basis.shells();
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();
    const auto& atoms = mol.atoms();

    for (int s = 0; s < static_cast<int>(basis_shells.size()); ++s) {
        int atom_idx = basis_shells[s].atom_index;
        int n_funcs = lib_shells[s].size();
        int bf_start = shell2bf[s];
        int l = basis_shells[s].l;
        int Z = atoms[atom_idx].Z;
        const auto* elem = params.element_data(Z);

        double k_en = 1.0;
        double kcn = 0.0;
        double poly = 0.0;
        double gam = elem ? elem->gam : 0.5;
        if (elem) {
            for (const auto& sh : elem->shells) {
                if (sh.l == l) {
                    k_en = sh.k_en;
                    kcn = sh.kcn;
                    poly = sh.poly;
                    break;
                }
            }
        }

        GFN2ShellInfo info;
        info.atom_idx = atom_idx;
        info.Z = Z;
        info.l = l;
        info.bf_start = bf_start;
        info.n_funcs = n_funcs;
        info.k_en = k_en;
        info.hardness = gam * k_en;
        info.kcn = kcn;
        info.poly = poly;
        shells.push_back(info);
    }
    return shells;
}

// ---------------------------------------------------------------------------
// Shell-pair gamma for GFN2-xTB H¹
// ---------------------------------------------------------------------------

double gfn2_shell_gamma_at_distance(
    const GFN2ShellInfo& shell_a,
    const GFN2ShellInfo& shell_b,
    double distance_bohr) {
    const double ha = shell_a.hardness > 0.0 ? shell_a.hardness : 0.5;
    const double hb = shell_b.hardness > 0.0 ? shell_b.hardness : 0.5;
    const double gab = 0.5 * (ha + hb);
    return 1.0 / std::sqrt(
        distance_bohr * distance_bohr + 1.0 / (gab * gab));
}

Eigen::MatrixXd build_gfn2_shell_gamma(
    const std::vector<GFN2ShellInfo>& shells,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params) {

    const int n_shells = static_cast<int>(shells.size());
    const auto& atoms = mol.atoms();
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n_shells, n_shells);

    for (int si = 0; si < n_shells; ++si) {
        const auto& sa = shells[si];

        // On-site: gamma_AA^{ll'} is the same arithmetic-averaged hardness
        // used by the off-site Klopman-Ohno kernel at R=0.
        for (int sj = 0; sj < n_shells; ++sj) {
            const auto& sb = shells[sj];
            if (sa.atom_idx == sb.atom_idx) {
                gamma(si, sj) =
                    gfn2_shell_gamma_at_distance(sa, sb, 0.0);
            }
        }

        // Off-site GFN2 effective Coulomb kernel:
        //   gamma_AB^{ll'} = 1 / sqrt(R_AB^2 + gamma_ll'^{-2})
        // with gamma_ll' the arithmetic average of shell hardnesses. This
        // matches tblite's effective_coulomb(gexp=2, arithmetic_average).
        for (int sj = si + 1; sj < n_shells; ++sj) {
            const auto& sb = shells[sj];
            if (sa.atom_idx == sb.atom_idx) continue;

            int a = sa.atom_idx, b = sb.atom_idx;
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);

            const double g =
                gfn2_shell_gamma_at_distance(sa, sb, R);

            gamma(si, sj) = g;
            gamma(sj, si) = g;
        }
    }
    return gamma;
}

}  // namespace semiempirical
}  // namespace vibeqc
