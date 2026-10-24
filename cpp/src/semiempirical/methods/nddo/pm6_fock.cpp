#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vibeqc/semiempirical/methods/nddo/gradient_fd_checks.hpp"
#include "vibeqc/semiempirical/methods/nddo/slater_overlap.hpp"
#include "vibeqc/semiempirical/methods/nddo/diatomic_gammas.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// Two-center electron repulsion (multi-term Gaussian or Ohno-Klopman)
//   If gamma_terms available: γ = ΣᵢΣⱼ cᵢcⱼ / √(R² + (dᵢ+dⱼ)²)
//   Otherwise: γ = 1 / √(R² + ¼(ρ_A+ρ_B)²)
// ---------------------------------------------------------------------------

static inline double gamma_ab_rho(double R, double rho_a, double rho_b) {
    double eta = 0.5 * (rho_a + rho_b);
    return 1.0 / std::sqrt(R * R + eta * eta);
}

static inline double gamma_ab_multi(
    double R,
    const std::vector<NDDOElementData::GammaTerm>& terms_a,
    const std::vector<NDDOElementData::GammaTerm>& terms_b) {
    if (terms_a.empty() || terms_b.empty())
        return 0.0;  // fallback: caller should use simple gamma
    double g = 0.0;
    for (const auto& ta : terms_a) {
        for (const auto& tb : terms_b) {
            double eta2 = (ta.exponent + tb.exponent);
            eta2 *= eta2;
            g += ta.coeff * tb.coeff / std::sqrt(R * R + eta2);
        }
    }
    // Reject unphysical values: gamma must be positive (two-electron repulsion)
    // and have sufficient magnitude.  If the expansion produces a negative or
    // negligible value, the multi-term form is not correctly parameterised
    // (e.g. MOPAC's gues61/gues62 arrays have a different meaning than the
    // simple Gaussian-overlap formula used here) and we fall back to
    // Ohno-Klopman.
    if (g <= 0.0)
        return 0.0;
    double g_simple = 1.0 / std::sqrt(R * R + 1.0);  // ≈ Ohno-Klopman scale
    if (g < 1e-4 * g_simple)
        return 0.0;  // too small, use simple gamma
    return g;
}

double gamma_ab(double R, double rho_a, double rho_b) {
    return gamma_ab_rho(R, rho_a, rho_b);
}

namespace {

constexpr double EV_TO_HARTREE = 1.0 / 27.2114;

double pm6_po1(const NDDOElementData& ed) {
    const double gss = ed.gss > 0.0 ? ed.gss * EV_TO_HARTREE : 0.367;
    return 0.5 / gss;
}

double pm6_po9(const NDDOElementData& ed) {
    return ed.pcore > 1.0e-5 ? ed.pcore : pm6_po1(ed);
}

struct SPMultipoleParameters {
    double dipole_separation = 0.0;
    double quadrupole_separation = 0.0;
    double po1 = 0.0;
    double po2 = 0.0;
    double po3 = 0.0;
    double po9 = 0.0;
};

SPMultipoleParameters pm6_sp_multipole_parameters(
    int Z,
    const NDDOElementData& ed) {
    const double n = static_cast<double>(valence_n_sp(Z));
    const double dd = (2.0 * n + 1.0)
        * std::pow(4.0 * ed.zs * ed.zp, n + 0.5)
        / std::pow(ed.zs + ed.zp, 2.0 * n + 2.0)
        / std::sqrt(3.0);
    const double qq = std::sqrt(
        (4.0 * n * n + 6.0 * n + 2.0) / 20.0) / ed.zp;
    const double hsp = std::max(ed.hsp, 1.0e-7) * EV_TO_HARTREE;
    const double hpp = std::max(0.5 * (ed.gpp - ed.gp2), 0.1)
        * EV_TO_HARTREE;

    auto hsp_value = [dd](double a) {
        return 0.5 * a
            - 0.5 / std::sqrt(4.0 * dd * dd + 1.0 / (a * a));
    };
    double ad1 = std::cbrt(hsp / (dd * dd));
    double ad2 = ad1 + 0.04;
    for (int i = 0; i < 5; ++i) {
        const double f1 = hsp_value(ad1);
        const double f2 = hsp_value(ad2);
        if (std::abs(f2 - f1) < 1.0e-25) break;
        const double ad3 = ad1 + (ad2 - ad1) * (hsp - f1) / (f2 - f1);
        ad1 = ad2;
        ad2 = ad3;
    }

    auto hpp_value = [qq](double a) {
        return 0.25 * a
            - 0.5 / std::sqrt(4.0 * qq * qq + 1.0 / (a * a))
            + 0.25 / std::sqrt(8.0 * qq * qq + 1.0 / (a * a));
    };
    double aq1 = std::pow(hpp / (3.0 * std::pow(qq, 4.0)), 0.2);
    double aq2 = aq1 + 0.04;
    for (int i = 0; i < 5; ++i) {
        const double f1 = hpp_value(aq1);
        const double f2 = hpp_value(aq2);
        if (std::abs(f2 - f1) < 1.0e-25) break;
        const double aq3 = aq1 + (aq2 - aq1) * (hpp - f1) / (f2 - f1);
        aq1 = aq2;
        aq2 = aq3;
    }
    if (!(ad2 > 0.0) || !(aq2 > 0.0)
        || !std::isfinite(ad2) || !std::isfinite(aq2)) {
        throw std::runtime_error("PM6 s/p multipole radii did not converge");
    }
    return {dd, qq, pm6_po1(ed), 0.5 / ad2, 0.5 / aq2, pm6_po9(ed)};
}

struct MultipolePoint {
    double weight = 0.0;
    Eigen::Vector3d offset = Eigen::Vector3d::Zero();
    double radius = 0.0;
};

struct AOProductMultipole {
    std::array<MultipolePoint, 4> points{};
    int size = 0;
};

AOProductMultipole pm6_ao_product_multipole(
    int mu,
    int nu,
    const SPMultipoleParameters& mp) {
    if (mu > nu) std::swap(mu, nu);
    AOProductMultipole result;
    auto add = [&result](
        double weight,
        const Eigen::Vector3d& offset,
        double radius) {
        result.points[result.size++] = {weight, offset, radius};
    };
    if (mu == 0 && nu == 0) {
        add(1.0, Eigen::Vector3d::Zero(), mp.po1);
        return result;
    }
    if (mu == 0) {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis[nu - 1] = 1.0;
        add(0.5, mp.dipole_separation * axis, mp.po2);
        add(-0.5, -mp.dipole_separation * axis, mp.po2);
        return result;
    }
    if (mu == nu) {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis[mu - 1] = 1.0;
        add(1.0, Eigen::Vector3d::Zero(), mp.po1);
        add(0.25, 2.0 * mp.quadrupole_separation * axis, mp.po3);
        add(0.25, -2.0 * mp.quadrupole_separation * axis, mp.po3);
        add(-0.5, Eigen::Vector3d::Zero(), mp.po3);
        return result;
    }

    Eigen::Vector3d first_axis = Eigen::Vector3d::Zero();
    Eigen::Vector3d second_axis = Eigen::Vector3d::Zero();
    first_axis[mu - 1] = 1.0;
    second_axis[nu - 1] = 1.0;
    if (mu == 1) {
        // In the diatomic frame, p_sigma*p_pi is the l=2, |m|=1 point
        // configuration.  Its four charges lie on the two bisectors.
        const Eigen::Vector3d same = mp.quadrupole_separation
            * (first_axis + second_axis);
        const Eigen::Vector3d opposite = mp.quadrupole_separation
            * (first_axis - second_axis);
        add(0.25, same, mp.po3);
        add(0.25, -same, mp.po3);
        add(-0.25, opposite, mp.po3);
        add(-0.25, -opposite, mp.po3);
        return result;
    }
    // 2 p_i p_j = p_u^2 - p_v^2 for the normalized bisector axes
    // u=(i+j)/sqrt(2), v=(i-j)/sqrt(2).  Build the off-diagonal product
    // from that polarization identity so it has the same quadrupole moment
    // and additive radius as the diagonal p products.
    const Eigen::Vector3d same = std::sqrt(2.0)
        * mp.quadrupole_separation * (first_axis + second_axis);
    const Eigen::Vector3d opposite = std::sqrt(2.0)
        * mp.quadrupole_separation * (first_axis - second_axis);
    add(0.125, same, mp.po3);
    add(0.125, -same, mp.po3);
    add(-0.125, opposite, mp.po3);
    add(-0.125, -opposite, mp.po3);
    return result;
}

double pm6_multipole_interaction(
    const AOProductMultipole& first,
    const AOProductMultipole& second,
    const Eigen::Vector3d& first_to_second) {
    double value = 0.0;
    for (int i = 0; i < first.size; ++i) {
        for (int j = 0; j < second.size; ++j) {
            const auto& a = first.points[i];
            const auto& b = second.points[j];
            const Eigen::Vector3d separation =
                first_to_second + b.offset - a.offset;
            const double radius = a.radius + b.radius;
            value += a.weight * b.weight /
                std::sqrt(separation.squaredNorm() + radius * radius);
        }
    }
    return value;
}

double pm6_multipole_core_attraction(
    const AOProductMultipole& electron,
    const Eigen::Vector3d& electron_to_core,
    double core_radius,
    int core_charge) {
    double value = 0.0;
    for (int i = 0; i < electron.size; ++i) {
        const auto& point = electron.points[i];
        const Eigen::Vector3d separation = electron_to_core - point.offset;
        const double radius = point.radius + core_radius;
        value -= static_cast<double>(core_charge) * point.weight /
            std::sqrt(separation.squaredNorm() + radius * radius);
    }
    return value;
}

Eigen::Matrix3d pm6_sp_diatomic_rotation(const Eigen::Vector3d& unit_bond) {
    const double sb = std::hypot(unit_bond.x(), unit_bond.y());
    double ca = 0.0;
    double sa = 0.0;
    double cb = 0.0;
    if (sb > 1.0e-7) {
        ca = unit_bond.x() / sb;
        sa = unit_bond.y() / sb;
        cb = unit_bond.z();
    } else {
        ca = unit_bond.z() < 0.0 ? -1.0 : 1.0;
        cb = ca;
    }
    Eigen::Matrix3d rotation;
    rotation <<
        ca * sb, ca * cb, -sa,
        sa * sb, sa * cb,  ca,
             cb,      -sb, 0.0;
    return rotation;
}

}  // namespace

double pm6_electron_core_gamma(
    double R,
    const NDDOElementData& electron_atom,
    const NDDOElementData& core_atom) {
    if (!std::isfinite(R) || R <= 1.0e-12) {
        throw std::invalid_argument(
            "PM6 electron-core gamma requires a positive finite distance");
    }
    if (!pm6_has_complete_base_parameters(electron_atom)
        || !pm6_has_complete_base_parameters(core_atom)) {
        throw std::invalid_argument(
            "PM6 electron-core gamma requires complete executable element "
            "records");
    }
    const double radius = pm6_po1(electron_atom) + pm6_po9(core_atom);
    return 1.0 / std::sqrt(R * R + radius * radius);
}

PM6SPHydrogenPair pm6_sp_hydrogen_pair(
    int Z_heavy,
    const Eigen::Vector3d& heavy_to_hydrogen,
    const NDDOElementData& heavy_atom,
    const NDDOElementData& hydrogen_atom) {
    const double R = heavy_to_hydrogen.norm();
    if (Z_heavy <= 2 || Z_heavy > 86 || heavy_atom.Z != Z_heavy
        || hydrogen_atom.Z != 1) {
        throw std::invalid_argument(
            "PM6 s/p--hydrogen pair element records do not match the "
            "requested heavy--hydrogen pair");
    }
    if (!heavy_to_hydrogen.allFinite() || !std::isfinite(R)
        || R < 1.0e-12)
        throw std::invalid_argument(
            "PM6 s/p--hydrogen pair requires a heavy atom and nonzero distance");
    if (!pm6_has_complete_sp_parameters(heavy_atom)
        || !pm6_has_complete_sp_parameters(hydrogen_atom)) {
        throw std::invalid_argument(
            "PM6 s/p--hydrogen pair requires complete multipole parameters");
    }

    // NDDO represents an AO-product distribution by collinear point
    // multipoles.  Their separations (D, Q) follow from the normalized
    // Slater orbitals, while the additive Klopman--Ohno radii reproduce the
    // one-centre HSP and HPP integrals.  Work directly in atomic units so
    // the resulting matrices are Hartree.
    const double n = static_cast<double>(valence_n_sp(Z_heavy));
    const double dd = (2.0 * n + 1.0)
        * std::pow(4.0 * heavy_atom.zs * heavy_atom.zp, n + 0.5)
        / std::pow(heavy_atom.zs + heavy_atom.zp, 2.0 * n + 2.0)
        / std::sqrt(3.0);
    const double qq = std::sqrt(
        (4.0 * n * n + 6.0 * n + 2.0) / 20.0) / heavy_atom.zp;
    const double hsp = std::max(heavy_atom.hsp, 1.0e-7) * EV_TO_HARTREE;
    const double hpp = std::max(
        0.5 * (heavy_atom.gpp - heavy_atom.gp2), 0.1)
        * EV_TO_HARTREE;

    auto hsp_value = [dd](double a) {
        return 0.5 * a
            - 0.5 / std::sqrt(4.0 * dd * dd + 1.0 / (a * a));
    };
    double ad1 = std::cbrt(hsp / (dd * dd));
    double ad2 = ad1 + 0.04;
    for (int i = 0; i < 5; ++i) {
        const double f1 = hsp_value(ad1);
        const double f2 = hsp_value(ad2);
        if (std::abs(f2 - f1) < 1.0e-25) break;
        const double ad3 = ad1 + (ad2 - ad1) * (hsp - f1) / (f2 - f1);
        ad1 = ad2;
        ad2 = ad3;
    }

    auto hpp_value = [qq](double a) {
        return 0.25 * a
            - 0.5 / std::sqrt(4.0 * qq * qq + 1.0 / (a * a))
            + 0.25 / std::sqrt(8.0 * qq * qq + 1.0 / (a * a));
    };
    double aq1 = std::pow(hpp / (3.0 * std::pow(qq, 4.0)), 0.2);
    double aq2 = aq1 + 0.04;
    for (int i = 0; i < 5; ++i) {
        const double f1 = hpp_value(aq1);
        const double f2 = hpp_value(aq2);
        if (std::abs(f2 - f1) < 1.0e-25) break;
        const double aq3 = aq1 + (aq2 - aq1) * (hpp - f1) / (f2 - f1);
        aq1 = aq2;
        aq2 = aq3;
    }

    if (!(ad2 > 0.0) || !(aq2 > 0.0) ||
        !std::isfinite(ad2) || !std::isfinite(aq2)) {
        throw std::runtime_error(
            "PM6 s/p--hydrogen multipole radii did not converge");
    }

    const double po1_x = pm6_po1(heavy_atom);
    const double po1_h = pm6_po1(hydrogen_atom);
    const double po2_x = 0.5 / ad2;
    const double po3_x = 0.5 / aq2;
    const double po9_x = pm6_po9(heavy_atom);
    const double po9_h = pm6_po9(hydrogen_atom);
    const double two_q = 2.0 * qq;
    const double R2 = R * R;
    const Eigen::Vector3d u = heavy_to_hydrogen / R;

    const double aee = std::pow(po1_x + po1_h, 2.0);
    const double ade = std::pow(po2_x + po1_h, 2.0);
    const double aqe = std::pow(po3_x + po1_h, 2.0);
    const double ri1 = 1.0 / std::sqrt(R2 + aee);
    const double ri2 =
        0.5 / std::sqrt(std::pow(R - dd, 2.0) + ade)
        - 0.5 / std::sqrt(std::pow(R + dd, 2.0) + ade);
    const double ri3 = ri1
        + 0.25 / std::sqrt(std::pow(R + two_q, 2.0) + aqe)
        + 0.25 / std::sqrt(std::pow(R - two_q, 2.0) + aqe)
        - 0.5 / std::sqrt(R2 + aqe);
    const double ri4 = ri1
        + 0.5 / std::sqrt(R2 + aqe + two_q * two_q)
        - 0.5 / std::sqrt(R2 + aqe);

    PM6SPHydrogenPair pair;
    pair.electron_repulsion(0, 0) = ri1;
    pair.electron_repulsion.block<1, 3>(0, 1) = ri2 * u.transpose();
    pair.electron_repulsion.block<3, 1>(1, 0) = ri2 * u;
    pair.electron_repulsion.block<3, 3>(1, 1) =
        ri4 * Eigen::Matrix3d::Identity()
        + (ri3 - ri4) * (u * u.transpose());

    const double z_h = static_cast<double>(pm6_core_charge(1));
    const double z_x = static_cast<double>(pm6_core_charge(Z_heavy));
    const double adj = std::pow(po2_x + po9_h, 2.0);
    const double aqj = std::pow(po3_x + po9_h, 2.0);
    const double c_ss = -z_h /
        std::sqrt(R2 + std::pow(po9_h + po1_x, 2.0));
    const double c_sp = -z_h * (
        -0.5 / std::sqrt(std::pow(R + dd, 2.0) + adj)
        + 0.5 / std::sqrt(std::pow(R - dd, 2.0) + adj));
    const double core_x1 = 1.0 /
        std::sqrt(R2 + std::pow(po9_h + po1_x, 2.0));
    const double core_x2 = -0.5 / std::sqrt(R2 + aqj);
    const double c_p_sigma = -z_h * (
        core_x1 + core_x2
        + 0.25 / std::sqrt(std::pow(R - two_q, 2.0) + aqj)
        + 0.25 / std::sqrt(std::pow(R + two_q, 2.0) + aqj));
    const double c_p_pi = -z_h * (
        core_x1 + core_x2
        + 0.5 / std::sqrt(R2 + two_q * two_q + aqj));

    pair.heavy_core(0, 0) = c_ss;
    pair.heavy_core.block<1, 3>(0, 1) = c_sp * u.transpose();
    pair.heavy_core.block<3, 1>(1, 0) = c_sp * u;
    pair.heavy_core.block<3, 3>(1, 1) =
        c_p_pi * Eigen::Matrix3d::Identity()
        + (c_p_sigma - c_p_pi) * (u * u.transpose());
    pair.hydrogen_core = -z_x /
        std::sqrt(R2 + std::pow(po9_x + po1_h, 2.0));
    return pair;
}

PM6SPSPPair pm6_sp_sp_pair(
    int Z_first,
    int Z_second,
    const Eigen::Vector3d& first_to_second,
    const NDDOElementData& first_atom,
    const NDDOElementData& second_atom) {
    const double R = first_to_second.norm();
    if (Z_first <= 2 || Z_first > 86 || Z_second <= 2 || Z_second > 86
        || first_atom.Z != Z_first || second_atom.Z != Z_second) {
        throw std::invalid_argument(
            "PM6 s/p--s/p pair element records do not match the requested "
            "heavy-atom pair");
    }
    if (!first_to_second.allFinite() || !std::isfinite(R) || R < 1.0e-12) {
        throw std::invalid_argument(
            "PM6 s/p--s/p pair requires distinct atoms at finite positions");
    }
    if (!pm6_has_complete_sp_parameters(first_atom)
        || !pm6_has_complete_sp_parameters(second_atom)
        || first_atom.has_d || second_atom.has_d) {
        throw std::invalid_argument(
            "PM6 s/p--s/p pair requires complete s/p-only multipole "
            "parameters");
    }

    // Dewar and Thiel, Theor. Chim. Acta 46, 89 (1977), represent every
    // retained NDDO AO product by one monopole, two dipole charges, or four
    // quadrupole charges.  Evaluating those point distributions directly in
    // the molecular frame is algebraically the 22-integral diatomic-frame
    // construction followed by its AO rotation, without a scalar gamma path.
    const auto first_mp = pm6_sp_multipole_parameters(Z_first, first_atom);
    const auto second_mp = pm6_sp_multipole_parameters(Z_second, second_atom);
    std::array<AOProductMultipole, 16> first_products;
    std::array<AOProductMultipole, 16> second_products;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            const int pair_index = 4 * mu + nu;
            first_products[pair_index] =
                pm6_ao_product_multipole(mu, nu, first_mp);
            second_products[pair_index] =
                pm6_ao_product_multipole(mu, nu, second_mp);
        }
    }

    // The finite point-charge quadrupoles are an approximation in the
    // diatomic frame; rotating the charges first is not algebraically the
    // same approximation.  Evaluate the complete local tensor on the bond
    // axis, then rotate its AO indices, as required by the NDDO definition.
    const Eigen::Vector3d local_displacement(R, 0.0, 0.0);
    PM6SPSPPair local_pair;
    for (int first_pair = 0; first_pair < 16; ++first_pair) {
        for (int second_pair = 0; second_pair < 16; ++second_pair) {
            local_pair.electron_repulsion(first_pair, second_pair) =
                pm6_multipole_interaction(
                    first_products[first_pair],
                    second_products[second_pair],
                    local_displacement);
        }
    }
    const int first_core_charge = pm6_core_charge(Z_first);
    const int second_core_charge = pm6_core_charge(Z_second);
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            const int pair_index = 4 * mu + nu;
            local_pair.first_core(mu, nu) = pm6_multipole_core_attraction(
                first_products[pair_index],
                local_displacement,
                second_mp.po9,
                second_core_charge);
            local_pair.second_core(mu, nu) = pm6_multipole_core_attraction(
                second_products[pair_index],
                -local_displacement,
                first_mp.po9,
                first_core_charge);
        }
    }

    Eigen::Matrix4d ao_rotation = Eigen::Matrix4d::Zero();
    ao_rotation(0, 0) = 1.0;
    const Eigen::Matrix3d canonical_rotation =
        pm6_sp_diatomic_rotation(Eigen::Vector3d::UnitX());
    ao_rotation.block<3, 3>(1, 1) =
        pm6_sp_diatomic_rotation(first_to_second / R)
        * canonical_rotation.transpose();

    Eigen::Matrix<double, 16, 16> product_rotation =
        Eigen::Matrix<double, 16, 16>::Zero();
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    product_rotation(4 * mu + nu, 4 * i + j) =
                        ao_rotation(mu, i) * ao_rotation(nu, j);
                }
            }
        }
    }

    PM6SPSPPair pair;
    pair.electron_repulsion = product_rotation
        * local_pair.electron_repulsion
        * product_rotation.transpose();
    pair.first_core = ao_rotation * local_pair.first_core
        * ao_rotation.transpose();
    pair.second_core = ao_rotation * local_pair.second_core
        * ao_rotation.transpose();
    return pair;
}

// ---------------------------------------------------------------------------
// Core-core repulsion
//   E_AB = Z_A^eff · Z_B^eff · γ_AB · (1 + exp(−α_A R) + exp(−α_B R))
// ---------------------------------------------------------------------------

// Number of VALENCE electrons in the molecule = Σ core charges − net charge.
// NDDO is a valence-only theory: the SCF must fill valence electrons, NOT the
// all-electron count returned by Molecule::n_electrons() (which counts the
// frozen cores too — e.g. 10 for H₂O instead of 8).  Using the all-electron
// count over-fills the valence shell by one MO per heavy-atom core pair and
// leaves Σ pop ≠ Σ core_charge, i.e. a spurious net charge that makes the
// electrostatics diverge.  MSINDO does this correctly (msindo.py: nelec =
// sum(eff_core_charge) − charge); this brings PM6/OMx into agreement.
static int valence_electron_count(const Molecule& mol) {
    int n = 0;
    for (const auto& atom : mol.atoms())
        n += pm6_core_charge(atom.Z);
    return n - mol.charge();
}

static void reject_coincident_distinct_atoms(
    const Molecule& mol,
    const char* driver) {
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
                    std::string(driver) + ": distinct atoms "
                    + std::to_string(a) + " and " + std::to_string(b)
                    + " are coincident");
            }
        }
    }
}

static int restricted_occupation_count(
    int n_electrons,
    int n_basis,
    const char* driver) {
    if (n_electrons < 0 || n_electrons % 2 != 0
        || n_electrons > 2 * n_basis) {
        throw std::invalid_argument(
            std::string(driver)
            + ": electron count is incompatible with the closed-shell "
              "NDDO basis");
    }
    return n_electrons / 2;
}

static std::pair<int, int> unrestricted_occupation_counts(
    int n_electrons,
    int multiplicity,
    int n_basis,
    const char* driver) {
    const int n_unpaired = multiplicity - 1;
    if (n_electrons < 0 || n_unpaired < 0 || n_electrons < n_unpaired
        || (n_electrons - n_unpaired) % 2 != 0) {
        throw std::invalid_argument(
            std::string(driver)
            + ": invalid multiplicity for electron count");
    }
    const int n_beta = (n_electrons - n_unpaired) / 2;
    const int n_alpha = n_beta + n_unpaired;
    if (n_alpha > n_basis || n_beta > n_basis) {
        throw std::invalid_argument(
            std::string(driver)
            + ": electron count exceeds the NDDO basis capacity");
    }
    return {n_alpha, n_beta};
}

static void validate_scf_controls(
    int max_iter,
    double conv_tol,
    const char* driver) {
    if (max_iter < 1 || !std::isfinite(conv_tol) || conv_tol <= 0.0) {
        throw std::invalid_argument(
            std::string(driver)
            + ": max_iter and conv_tol must be positive and finite");
    }
}

static void reject_unsupported_nddo_d_shells(
    const Molecule& mol,
    const PM6ParameterSet& params,
    const char* driver) {
    for (const auto& atom : mol.atoms()) {
        if (atom.Z > 86) {
            throw std::invalid_argument(
                std::string(driver) + ": actinide element Z="
                + std::to_string(atom.Z)
                + " is unavailable because its PM6 principal-shell "
                  "convention is not implemented");
        }
        const auto* ed = params.element_data(atom.Z);
        if (ed && atom.Z > 2 && (ed->has_d || ed->n_orbitals != 4)) {
            throw std::invalid_argument(
                std::string(driver) + ": d-shell element Z="
                + std::to_string(atom.Z)
                + " or unsupported AO count is unavailable because the "
                  "PM6 kernel implements exactly the s/p Hamiltonian");
        }
        if (ed && !pm6_has_complete_sp_parameters(*ed)) {
            throw std::invalid_argument(
                std::string(driver) + ": element Z="
                + std::to_string(atom.Z)
                + " has an incomplete placeholder parameter record");
        }
    }
}

double pm6_core_core_repulsion(
    int Za, int Zb, double R,
    const NDDOElementData& ed_a, const NDDOElementData& ed_b,
    const NDDODiatomicParams* dp) {
    if (R < 1e-12) return 0.0;
    const double ev2ha = EV_TO_HARTREE;
    int nval_a = pm6_core_charge(Za);
    int nval_b = pm6_core_charge(Zb);

    // Core–core repulsion uses the CORE charge (number of valence electrons,
    // O→6, C→4, H→1), NOT the nuclear Z.  This is the MNDO/PM6 "EN" convention
    // (Stewart 2007) and — crucially — it must match the Z_B^core used in the
    // electron–core attraction (core_charge[] = pm6_core_charge) so that a
    // neutral atom pair's monopole electrostatics cancel to (Δq_A·Δq_B)·γ.
    // MOPAC constructs the core Klopman-Ohno radius as po(9)=EV/(2*GSS),
    // with the sparse poc_ parameter as an explicit override.  ``polvo`` is
    // atomic polarizability and must not enter this interaction.
    const double po9_a = pm6_po9(ed_a);
    const double po9_b = pm6_po9(ed_b);
    double gab = 1.0 / std::sqrt(
        R * R + (po9_a + po9_b) * (po9_a + po9_b));
    double enuc = nval_a * nval_b * gab;
    double a0 = 0.529177;
    double r_ang = R * a0;

    const bool has_fitted_pair = dp && std::abs(dp->d2) > 1.0e-5;
    double energy;
    if (has_fitted_pair) {
        double r_ang6 = r_ang * r_ang * r_ang * r_ang * r_ang * r_ang;
        double d  = dp->d1 < 1.0e-6 ? 1.2 : dp->d1;
        double xf = dp->d2;
        double exponent_distance = r_ang + 0.0003 * r_ang6;
        const int z_lo = std::min(Za, Zb);
        const int z_hi = std::max(Za, Zb);
        if (z_lo == 1 && (z_hi == 6 || z_hi == 7 || z_hi == 8))
            exponent_distance = r_ang * r_ang;
        double scale = 1.0 + 2.0 * xf * std::exp(-d * exponent_distance);
        if (Za == 6 && Zb == 6)
            scale += 9.278465 * std::exp(-5.983752 * r_ang);
        if ((Za == 8 && Zb == 14) || (Za == 14 && Zb == 8))
            scale -= 0.0007 * std::exp(-(r_ang - 2.9) * (r_ang - 2.9));
        energy = enuc * scale;
    } else {
        // MOPAC's PM6 fallback for a missing diatomic pair.  Production
        // parameter sets define the fitted pair, but custom/incomplete sets
        // must use the documented generic wall rather than MNDO alpha terms.
        const bool lanthanide_pair =
            (Za > 56 && Za < 72) || (Zb > 56 && Zb < 72);
        const double exponent = lanthanide_pair ? 3.0 : 2.18;
        energy = enuc * (1.0 + 10.0 * std::exp(-exponent * r_ang));
    }

    auto gaussian_term = [&](const NDDOElementData::GaussianTerm& gt) {
        double dr = r_ang - gt.factor;
        double ax = gt.exponent * dr * dr;
        if (ax < 25.0) {
            // MOPAC accumulates this correction in eV.  The main monopole
            // above is already in Hartree, so convert explicitly here.
            energy += nval_a * nval_b / r_ang * gt.coeff
                    * std::exp(-ax) * ev2ha;
        }
    };
    if (!ed_a.gaussian_terms.empty())
        gaussian_term(ed_a.gaussian_terms.front());
    if (!ed_b.gaussian_terms.empty())
        gaussian_term(ed_b.gaussian_terms.front());
    if (!has_fitted_pair) {
        // This mirrors the upstream generic-pair branch, including its
        // additional four-term AM1-style correction loop.
        for (std::size_t i = 0;
             i < std::min<std::size_t>(4, ed_a.gaussian_terms.size()); ++i)
            gaussian_term(ed_a.gaussian_terms[i]);
        for (std::size_t i = 0;
             i < std::min<std::size_t>(4, ed_b.gaussian_terms.size()); ++i)
            gaussian_term(ed_b.gaussian_terms[i]);
    }

    // Stewart 2007 Eq. (7): negligible at normal distances, but prevents
    // collapse when atoms are forced inside their unpolarizable cores.
    double core_scale = r_ang /
        (std::pow(static_cast<double>(Za), 0.3333)
         + std::pow(static_cast<double>(Zb), 0.3333));
    if (core_scale < 3.0) {
        double wall_ev = 1.0e-8 / std::pow(core_scale, 12.0);
        energy += std::min(wall_ev, 1.0e5) * ev2ha;
    }
    return energy;
}

// Thin wrappers — shared slater_*() functions in the header take zR directly.
inline double slater_overlap_ss(double zeta, double R) { return slater_ss(zeta * R); }
inline double slater_overlap_sp(double zeta, double R) { return slater_sp(zeta * R); }
inline double slater_overlap_pp_sigma(double zeta, double R) { return slater_pp_sigma(zeta * R); }
inline double slater_overlap_pp_pi(double zeta, double R) { return slater_pp_pi(zeta * R); }
inline double cos_dir(int axis, double dx, double dy, double dz, double R) {
    return slater_dir_cos(axis, dx, dy, dz, R);
}

// Correct two-centre STO overlap for AO (atom Z_a, type t_a) — AO (atom Z_b,
// type t_b), pulling the s/p exponents from the parameter set and the principal
// quantum number from Z.  Templated so it serves both PM6ParameterSet and
// OMxParameterSet (both expose element_data(Z)->zs/zp).
template <typename ParamSet>
inline double nddo_overlap_from_Z(const ParamSet& params, int Z_a, int t_a,
                                  int Z_b, int t_b,
                                  double dx, double dy, double dz, double R) {
    const auto* ed_a = params.element_data(Z_a);
    const auto* ed_b = params.element_data(Z_b);
    double zs_a = ed_a ? ed_a->zs : 1.0, zp_a = ed_a ? ed_a->zp : 1.0;
    double zs_b = ed_b ? ed_b->zs : 1.0, zp_b = ed_b ? ed_b->zp : 1.0;
    if (zs_a <= 0.0) zs_a = 1.0;
    if (zp_a <= 0.0) zp_a = zs_a;
    if (zs_b <= 0.0) zs_b = 1.0;
    if (zp_b <= 0.0) zp_b = zs_b;
    const int n_a = valence_n_sp(Z_a), n_b = valence_n_sp(Z_b);
    return nddo_sto_overlap(n_a, t_a, zs_a, zp_a, n_b, t_b, zs_b, zp_b,
                            dx, dy, dz, R);
}

// ---------------------------------------------------------------------------
// Build PM6 Fock matrix and run SCF
// ---------------------------------------------------------------------------

PM6Result run_pm6_impl(
    const Molecule& mol,
    const PM6ParameterSet& params,
    int max_iter,
    double conv_tol,
    const Eigen::MatrixXd* initial_density,
    double electronic_temperature = 0.0) {

    // ---- Validate ----
    const std::string active_method = params.method_name();
    if (active_method != "pm6") {
        throw std::invalid_argument(
            "PM6 driver requires PM6 parameters, not "
            + active_method);
    }
    validate_scf_controls(max_iter, conv_tol, "PM6");
    if (mol.multiplicity() != 1)
        throw std::invalid_argument("PM6: only closed-shell supported");
    if (mol.atoms().empty())
        throw std::invalid_argument("PM6: empty molecule");

    // Check all elements are in the parameter set
    {
        std::vector<int> missing;
        for (const auto& atom : mol.atoms()) {
            if (!params.has_element(atom.Z))
                missing.push_back(atom.Z);
        }
        if (!missing.empty()) {
            std::string msg = "PM6: element(s) not in parameter set: ";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += "Z=" + std::to_string(missing[i]);
            }
            msg += ".  Try load_pm6_mopac_params() for 75 chemical-element "
                   "records.";
            throw std::invalid_argument(msg);
        }
    }

    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());
    reject_coincident_distinct_atoms(mol, "PM6");
    reject_unsupported_nddo_d_shells(mol, params, "PM6");

    // ---- AO basis: s, px, py, pz per main-group atom ----
    std::vector<int> ao_Z;
    std::vector<int> ao_atom;
    std::vector<int> ao_type;   // 0=s, 1=px, 2=py, 3=pz
    std::vector<int> atom_ao_start(n_atoms, 0);
    std::vector<int> atom_n_ao(n_atoms, 0);
    int n_basis = 0;
    for (int a = 0; a < n_atoms; ++a) {
        int Z = atoms[a].Z;
        const auto* ed = params.element_data(Z);
        // H/He carry only a 1s valence orbital; everything else gets the sp set.
        // (NDDOElementData::n_orbitals defaults to 4 and the loaders never
        // override it for H, which silently gave H a spurious 2p shell — once the
        // electron–core attraction entered H, the SCF dumped the whole valence
        // density onto that phantom H-p shell.  MSINDO gives H one orbital.)
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

    const int n_occ = restricted_occupation_count(
        valence_electron_count(mol), n_basis, "PM6");

    // ---- Pre-load per-AO integrals (eV → Ha) ----
    double ev2ha = 1.0 / 27.2114;
    std::vector<double> U_s(n_basis, 0.0),     // one-centre core (Ha)
                        G_ss(n_basis, 0.0),
                        G_pp(n_basis, 0.0),
                        G_sp(n_basis, 0.0),
                        G_p2(n_basis, 0.0),
                        H_sp(n_basis, 0.0),
                        beta(n_basis, 0.0),    // resonance
                        rho(n_basis, 0.0);      // 1/Gss for gamma

    for (int mu = 0; mu < n_basis; ++mu) {
        int Z = ao_Z[mu];
        const auto* ed = params.element_data(Z);
        if (!ed) continue;
        bool is_s = (ao_type[mu] == 0);
        U_s[mu] = (is_s ? ed->uss : ed->upp) * ev2ha;
        G_ss[mu] = ed->gss * ev2ha;
        G_pp[mu] = ed->gpp * ev2ha;
        G_sp[mu] = ed->gsp * ev2ha;
        G_p2[mu] = ed->gp2 * ev2ha;
        H_sp[mu] = ed->hsp * ev2ha;
        beta[mu] = (is_s ? ed->betas : ed->betap) * ev2ha;
        rho[mu] = (ed->gss > 0.0) ? 1.0 / (ed->gss * ev2ha) : 2.72;
    }

    // Effective core charges
    std::vector<int> core_charge(n_atoms);
    for (int a = 0; a < n_atoms; ++a)
        core_charge[a] = pm6_core_charge(atoms[a].Z);

    struct ExactSPHydrogenPair {
        int heavy_atom;
        int hydrogen_atom;
        int heavy_start;
        int hydrogen_ao;
        PM6SPHydrogenPair integrals;
    };
    struct ExactSPSPPair {
        int first_start;
        int second_start;
        PM6SPSPPair integrals;
    };
    std::vector<ExactSPHydrogenPair> exact_sp_hydrogen_pairs;
    std::vector<ExactSPSPPair> exact_sp_sp_pairs;
    std::vector<std::vector<int>> exact_pair_index(
        n_atoms, std::vector<int>(n_atoms, -1));
    if (params.method_name() == "pm6") {
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = a + 1; b < n_atoms; ++b) {
                int heavy = -1;
                int hydrogen = -1;
                if (atoms[a].Z == 1 && atoms[b].Z > 2) {
                    hydrogen = a;
                    heavy = b;
                } else if (atoms[b].Z == 1 && atoms[a].Z > 2) {
                    hydrogen = b;
                    heavy = a;
                }
                if (heavy < 0 || atom_n_ao[heavy] != 4) continue;
                const auto* ed_x = params.element_data(atoms[heavy].Z);
                const auto* ed_h = params.element_data(1);
                if (!ed_x || !ed_h || ed_x->has_d) continue;
                Eigen::Vector3d displacement;
                for (int d = 0; d < 3; ++d) {
                    displacement[d] = atoms[hydrogen].xyz[d]
                        - atoms[heavy].xyz[d];
                }
                const int index = static_cast<int>(
                    exact_sp_hydrogen_pairs.size());
                exact_sp_hydrogen_pairs.push_back({
                    heavy,
                    hydrogen,
                    atom_ao_start[heavy],
                    atom_ao_start[hydrogen],
                    pm6_sp_hydrogen_pair(
                        atoms[heavy].Z, displacement, *ed_x, *ed_h)});
                exact_pair_index[a][b] = index;
                exact_pair_index[b][a] = index;
            }
        }
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = a + 1; b < n_atoms; ++b) {
                if (atoms[a].Z <= 2 || atoms[b].Z <= 2
                    || atom_n_ao[a] != 4 || atom_n_ao[b] != 4) {
                    continue;
                }
                const auto* ed_a = params.element_data(atoms[a].Z);
                const auto* ed_b = params.element_data(atoms[b].Z);
                if (!ed_a || !ed_b || ed_a->has_d || ed_b->has_d) continue;
                Eigen::Vector3d displacement;
                for (int d = 0; d < 3; ++d) {
                    displacement[d] = atoms[b].xyz[d] - atoms[a].xyz[d];
                }
                exact_sp_sp_pairs.push_back({
                    atom_ao_start[a],
                    atom_ao_start[b],
                    pm6_sp_sp_pair(
                        atoms[a].Z,
                        atoms[b].Z,
                        displacement,
                        *ed_a,
                        *ed_b)});
                exact_pair_index[a][b] = 0;
                exact_pair_index[b][a] = 0;
            }
        }
    }

    // ---- SCF initialisation ----
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_basis, n_basis);
    if (initial_density != nullptr) {
        if (initial_density->rows() != n_basis ||
            initial_density->cols() != n_basis) {
            throw std::invalid_argument(
                "PM6: initial density dimension does not match AO basis");
        }
        if (!initial_density->allFinite()) {
            throw std::invalid_argument(
                "PM6: initial density must contain only finite values");
        }
        D = *initial_density;
    }
    PM6Result result;
    result.n_basis = n_basis;
    result.n_occ = n_occ;

    // Pulay DIIS on the commutator [F, P] (the NDDO basis is orthogonal, so
    // no S factors).  Plain fixed-point iteration oscillates for polar
    // multiple bonds (CO), so DIIS is the standard accelerator here.
    std::vector<Eigen::MatrixXd> diis_F, diis_e;
    const std::size_t DIIS_MAX = 8;
    // Hard Aufbau occupations can leave a near-degenerate closed-shell
    // density in a persistent Pulay cycle.  A bounded damped-mixing window
    // helps cross that nonlinear region.
    //
    // The window is a FIXED schedule, deliberately independent of max_iter
    // (#154).  Deriving damping_start from max_iter/3 made the converged
    // answer a function of the iteration budget rather than of the input:
    // PM6/norbornadiene converges at iteration 49 for max_iter >= 150, but
    // for max_iter <= 100 damping began at max_iter/3 < 49, derailed that
    // same trajectory, and the SCF then failed to converge at all -- a
    // smaller budget changed the result instead of merely bounding the work.
    // With a fixed schedule every budget walks the same iterate sequence, so
    // a larger max_iter can only ever take more steps along one path.
    const int damping_length = 32;
    const int damping_start = 64;
    const int diis_restart = damping_start + damping_length;

    for (int iter = 1; iter <= max_iter; ++iter) {
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n_basis, n_basis);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_basis, n_basis); // core Hamiltonian (track for energy)

        // ---------------------------------------------------------------
        // ONE-CENTRE Fock elements (Pople NDDO, closed-shell)
        //
        //  Diagonal:  F_μμ = U_μμ + Σ_ν∈A P_νν[(μμ|νν) − ½(μν|μν)]
        //  Off-diag:  F_μν = ½ P_μν[3(μν|μν) − (μμ|νν)]   (μ≠ν, same atom)
        // ---------------------------------------------------------------

        for (int a = 0; a < n_atoms; ++a) {
            // Collect AO indices for atom a
            std::vector<int> idx;
            for (int mu = 0; mu < n_basis; ++mu)
                if (ao_atom[mu] == a) idx.push_back(mu);
            int n_ao_a = static_cast<int>(idx.size());

            // --- Diagonal ----
            for (int i = 0; i < n_ao_a; ++i) {
                int mu = idx[i];
                int t_mu = ao_type[mu];  // 0=s, 1+=p

                // Core integral U_μμ
                F(mu, mu) += U_s[mu];
                H(mu, mu) += U_s[mu];

                // Coulomb:  Σ_ν P_νν (μμ|νν)
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
                    else // t_mu > 0, t_nu > 0, t_mu != t_nu
                        coul = G_p2[mu];
                    F(mu, mu) += D(nu, nu) * coul;
                }

                // Exchange:  −½ Σ_ν P_νν (μν|μν)
                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j];
                    int t_nu = ao_type[nu];
                    double exch = 0.0;
                    if (mu == nu) {
                        if (t_mu == 0) exch = G_ss[mu];
                        else           exch = G_pp[mu];
                    } else if (t_mu == 0 && t_nu > 0) {
                        exch = H_sp[mu];   // (sp|sp)
                    } else if (t_mu > 0 && t_nu == 0) {
                        exch = H_sp[nu];   // (sp|sp)
                    } else {
                        // t_mu > 0, t_nu > 0, t_mu != t_nu
                        exch = 0.5 * (G_pp[mu] - G_p2[mu]); // (p_x p_y|p_x p_y)
                    }
                    F(mu, mu) -= 0.5 * D(nu, nu) * exch;
                }
            }

            // --- Off-diagonal one-centre ----
            for (int i = 0; i < n_ao_a; ++i) {
                for (int j = i + 1; j < n_ao_a; ++j) {
                    int mu = idx[i], nu = idx[j];
                    int t_mu = ao_type[mu], t_nu = ao_type[nu];

                    double exch = 0.0;  // (μν|μν)
                    double coul = 0.0;   // (μμ|νν)

                    if (t_mu == 0 && t_nu > 0) {
                        exch = H_sp[mu];
                        coul = G_sp[mu];
                    } else if (t_mu > 0 && t_nu == 0) {
                        exch = H_sp[nu];
                        coul = G_sp[nu];
                    } else {
                        // both p, different spatial components
                        exch = 0.5 * (G_pp[mu] - G_p2[mu]);
                        coul = G_p2[mu];
                    }

                    // F_μν = ½ P_μν [3·(μν|μν) − (μμ|νν)]
                    double f_off = 0.5 * D(mu, nu) * (3.0 * exch - coul);
                    F(mu, nu) += f_off;
                    F(nu, mu) += f_off;
                }
            }
        }

        // ---------------------------------------------------------------
        // Precompute sigma/pi diatomic gammas for each atom pair
        // ---------------------------------------------------------------
        std::vector<std::vector<DiatomicGammas>> dg(n_atoms, std::vector<DiatomicGammas>(n_atoms));
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = a + 1; b < n_atoms; ++b) {
                const auto* ed_a = params.element_data(atoms[a].Z);
                const auto* ed_b = params.element_data(atoms[b].Z);
                if (ed_a && ed_b && !ed_a->gamma_terms.empty() && !ed_b->gamma_terms.empty()) {
                    double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
                    double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
                    double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                    double zr_a = ed_a->zs > 0 ? ed_a->zp / ed_a->zs : 1.0;
                    double zr_b = ed_b->zs > 0 ? ed_b->zp / ed_b->zs : 1.0;
                    dg[a][b] = compute_diatomic_gammas(R, ed_a->gamma_terms, ed_b->gamma_terms, zr_a, zr_b);
                    dg[b][a] = dg[a][b];
                }
            }
        }

        // ---------------------------------------------------------------
        // TWO-CENTRE  nuclear attraction  +  electron repulsion
        //   V_AB = (q_B − Z_B) · γ_AB   added to every diagonal on atom A
        // ---------------------------------------------------------------
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = 0; b < n_atoms; ++b) {
                if (a == b) continue;
                if (exact_pair_index[a][b] >= 0) continue;
                double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
                double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
                double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;

                // Total electron population on B
                double q_B = 0.0;
                for (int nu = 0; nu < n_basis; ++nu)
                    if (ao_atom[nu] == b) q_B += D(nu, nu);

                // Two-center gamma: multi-term if available, else Ohno-Klopman
                double g_ab = 0.0;
                const auto* ed_a = params.element_data(atoms[a].Z);
                const auto* ed_b = params.element_data(atoms[b].Z);
                if (ed_a && ed_b && !ed_a->gamma_terms.empty() && !ed_b->gamma_terms.empty()) {
                    g_ab = gamma_ab_multi(R, ed_a->gamma_terms, ed_b->gamma_terms);
                }
                if (g_ab == 0.0) {
                    // Find first AO of each atom for correct rho indexing
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

                // Per-AO Coulomb using sigma/pi gammas when available.
                // The two-centre potential V_μμ,B = (q_B − Z_B^core)·γ_AB splits
                // into a *two-electron* part (+q_B·γ, density-dependent → Fock
                // only) and a *one-electron* electron–core attraction
                // (−Z_B^core·γ).  The latter is part of the core Hamiltonian and
                // MUST also enter H, so the energy ½Tr[P(H+F)] counts it at full
                // weight.  Putting it in F alone halved it — the missing
                // ½·Σ pop_A·Z_B^core·γ of attraction is what made the PM6/OMx PES
                // dissociative (cf. MSINDO _v2core_one_side, which lands the same
                // −Z^core·γ attraction in its core Hamiltonian).
                double g_s, g_p_sigma, g_p_pi;
                if (dg[a][b].g_ss != 0.0) {
                    g_s = dg[a][b].g_ss;
                    g_p_sigma = dg[a][b].g_pp_ss;
                    g_p_pi = dg[a][b].g_sp_p;
                } else {
                    g_s = g_p_sigma = g_p_pi = g_ab;
                }
                const double z_core = core_charge[b];
                const bool audited_pm6_core = params.method_name() == "pm6";
                const double g_core = audited_pm6_core
                    ? pm6_electron_core_gamma(R, *ed_a, *ed_b)
                    : 0.0;

                for (int mu = 0; mu < n_basis; ++mu) {
                    if (ao_atom[mu] != a) continue;
                    double g_mu;
                    if (ao_type[mu] == 0 || g_s == g_p_sigma) {
                        g_mu = g_s;
                    } else {
                        // p-type AO: use sigma/pi weighted gamma
                        int d = ao_type[mu] - 1;
                        double ca = cos_dir(d, dx, dy, dz, R);
                        double ca2 = ca * ca;
                        g_mu = g_p_sigma * ca2 + g_p_pi * (1.0 - ca2);
                    }
                    const double directed_core_gamma =
                        audited_pm6_core ? g_core : g_mu;
                    F(mu, mu) += q_B * g_mu
                        - z_core * directed_core_gamma;
                    H(mu, mu) += -z_core * directed_core_gamma;
                }
            }
        }

        // The spherical population approximation above drops the dipole and
        // quadrupole components that are nonzero for an s/p heavy atom paired
        // with hydrogen.  Contract the exact NDDO pair block for this common
        // subclass, including its AO-resolved electron-core matrices.
        for (const auto& pair : exact_sp_hydrogen_pairs) {
            const int x0 = pair.heavy_start;
            const int h = pair.hydrogen_ao;
            const auto& eri = pair.integrals.electron_repulsion;
            const auto& core_x = pair.integrals.heavy_core;
            const double d_hh = D(h, h);
            F.block<4, 4>(x0, x0) += d_hh * eri + core_x;
            H.block<4, 4>(x0, x0) += core_x;
            const double j_h =
                (D.block<4, 4>(x0, x0).cwiseProduct(eri)).sum();
            F(h, h) += j_h + pair.integrals.hydrogen_core;
            H(h, h) += pair.integrals.hydrogen_core;
            const Eigen::Vector4d d_xh = D.block<4, 1>(x0, h);
            const Eigen::Vector4d exchange = -0.5 * eri * d_xh;
            F.block<4, 1>(x0, h) += exchange;
            F.block<1, 4>(h, x0) += exchange.transpose();
        }

        for (const auto& pair : exact_sp_sp_pairs) {
            const int a0 = pair.first_start;
            const int b0 = pair.second_start;
            const auto& eri = pair.integrals.electron_repulsion;
            Eigen::Matrix4d j_a = Eigen::Matrix4d::Zero();
            Eigen::Matrix4d j_b = Eigen::Matrix4d::Zero();
            Eigen::Matrix4d exchange = Eigen::Matrix4d::Zero();
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    const int first_pair = 4 * mu + nu;
                    for (int la = 0; la < 4; ++la) {
                        for (int si = 0; si < 4; ++si) {
                            const double integral =
                                eri(first_pair, 4 * la + si);
                            j_a(mu, nu) += D(b0 + la, b0 + si) * integral;
                            j_b(la, si) += D(a0 + mu, a0 + nu) * integral;
                            exchange(mu, la) -= 0.5
                                * D(a0 + nu, b0 + si) * integral;
                        }
                    }
                }
            }
            const auto& core_a = pair.integrals.first_core;
            const auto& core_b = pair.integrals.second_core;
            F.block<4, 4>(a0, a0) += j_a + core_a;
            F.block<4, 4>(b0, b0) += j_b + core_b;
            H.block<4, 4>(a0, a0) += core_a;
            H.block<4, 4>(b0, b0) += core_b;
            F.block<4, 4>(a0, b0) += exchange;
            F.block<4, 4>(b0, a0) += exchange.transpose();
        }

        // ---------------------------------------------------------------
        // TWO-CENTRE  resonance  +  exchange
        //   F_μν = β_μν − ½ P_μν γ_AB
        //   β_μν = ½ (β_A + β_B) · S_μν
        // ---------------------------------------------------------------
        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = ao_atom[mu];
            int t_mu = ao_type[mu];
            for (int nu = mu + 1; nu < n_basis; ++nu) {
                int a_nu = ao_atom[nu];
                if (a_mu == a_nu) continue;
                int t_nu = ao_type[nu];

                double dx = atoms[a_mu].xyz[0] - atoms[a_nu].xyz[0];
                double dy = atoms[a_mu].xyz[1] - atoms[a_nu].xyz[1];
                double dz = atoms[a_mu].xyz[2] - atoms[a_nu].xyz[2];
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;

                // Correct n-dependent two-centre STO overlap (Slater–Koster
                // rotation over the MSINDO STO kernel).  The old 1s-shaped
                // closed forms gave 2s/2p overlaps far too small and σ/π
                // sign-scrambled, collapsing the resonance for every second-row
                // atom (N₂'s triple-bond resonance fell to ~0 → dissociative).
                double S_val = nddo_overlap_from_Z(params, ao_Z[mu], t_mu,
                                                   ao_Z[nu], t_nu, dx, dy, dz, R);

                // Resonance: β_μν = ½ (β_A + β_B) · S_μν
                double beta_val = 0.5 * (beta[mu] + beta[nu]) * S_val;
                // Track in H_core for energy formula
                H(mu, nu) = beta_val;
                H(nu, mu) = beta_val;
                F(mu, nu) += beta_val;
                F(nu, mu) += beta_val;

                if (exact_pair_index[a_mu][a_nu] >= 0) continue;

                // Exchange:  −½ P_μν γ_AB
                // Use sigma/pi diatomic gammas when available
                double g_ex = 0.0;
                if (dg[a_mu][a_nu].g_ss != 0.0) {
                    double ca = 0.0, cb = 0.0;
                    if (t_mu > 0) ca = cos_dir(t_mu - 1, dx, dy, dz, R);
                    if (t_nu > 0) cb = cos_dir(t_nu - 1, dx, dy, dz, R);
                    g_ex = gamma_ao_pair(dg[a_mu][a_nu], t_mu, t_nu, ca, cb);
                }
                if (g_ex == 0.0) {
                    double eta_ex = 0.5 * (rho[mu] + rho[nu]);
                    g_ex = 1.0 / std::sqrt(R * R + eta_ex * eta_ex);
                }
                double ex = -0.5 * D(mu, nu) * g_ex;
                F(mu, nu) += ex;
                F(nu, mu) += ex;
            }
        }

        // ---- Pulay DIIS extrapolation ----
        // Skip the zero-density first iterate: its commutator residual is
        // exactly zero, which poisons the DIIS equations (all weight lands
        // on the seed Fock and the density freezes).
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
                Eigen::MatrixXd B = Eigen::MatrixXd::Constant(n + 1, n + 1, -1.0);
                B(n, n) = 0.0;
                for (int i = 0; i < n; ++i)
                    for (int j = i; j < n; ++j)
                        B(i, j) = B(j, i) = diis_e[i].cwiseProduct(diis_e[j]).sum();
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
                rhs(n) = -1.0;
                Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);
                if (c.allFinite()) {
                    F_eff.setZero();
                    for (int i = 0; i < n; ++i) F_eff += c(i) * diis_F[i];
                }
            }
        }

        // ---- Diagonalise ----
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_eff);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("PM6: diagonalisation failed");
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();

        // ---- New density ----
        Eigen::MatrixXd D_new;
        if (electronic_temperature > 0.0) {
            KPointOccupationOptions occupation_options;
            occupation_options.smearing_temperature =
                electronic_temperature;
            const auto occupation =
                compute_closed_shell_kpoint_occupations(
                    {eps}, {1.0}, 2.0 * n_occ, n_occ,
                    occupation_options);
            D_new =
                C * occupation.occupations_per_k.front().asDiagonal()
                * C.transpose();
        } else {
            Eigen::MatrixXd C_occ = C.leftCols(n_occ);
            D_new = 2.0 * C_occ * C_occ.transpose();
        }

        // ---- Convergence ----
        // An SCF solution requires BOTH the density to stop changing AND
        // the Fock matrix to commute with the density ([F, D] = 0).
        // Checking only delta_D can false-converge to a non-SCF state.
        double delta = (D_new - D).cwiseAbs().maxCoeff();

        // Guard against NaN/Inf in the density — a pathological DIIS
        // extrapolation or near-degenerate root can inject non-finite
        // values and the SCF will silently cycle to max_iter (BUG-024).
        if (!std::isfinite(delta)) {
            std::ostringstream oss;
            oss << "PM6: non-finite density change at iteration " << iter
                << " (delta=" << delta << ", n_basis=" << n_basis
                << ", n_occ=" << n_occ
                << ", last F.diagonal().minCoeff()=" << F.diagonal().minCoeff()
                << ", F.hasNaN=" << !F.allFinite()
                << ", D.trace()=" << D.trace()
                << "). The DIIS extrapolation or Fock build produced "
                << "NaN/Inf; this is a numerical instability, not a "
                << "physics convergence failure.  If reproducible, "
                << "retry with a higher smearing temperature "
                << "(run_pm6_with_smearing).";
            throw std::runtime_error(oss.str());
        }

        D = use_damping ? 0.5 * D + 0.5 * D_new : D_new;
        Eigen::MatrixXd comm = F_eff * D - D * F_eff;
        double comm_max = comm.cwiseAbs().maxCoeff();

        // Also guard the commutator
        if (!std::isfinite(comm_max)) {
            std::ostringstream oss;
            oss << "PM6: non-finite commutator [F,D] at iteration " << iter
                << " (comm_max=" << comm_max
                << ", F_eff.hasNaN=" << !F_eff.allFinite()
                << ", D.hasNaN=" << !D.allFinite()
                << "). Retry with smearing (run_pm6_with_smearing).";
            throw std::runtime_error(oss.str());
        }

        if (delta < conv_tol && comm_max < conv_tol) {
            // ---- Core-core repulsion ----
            double E_core = 0.0;
            for (int a = 0; a < n_atoms; ++a) {
                const auto* ed_a = params.element_data(atoms[a].Z);
                for (int b = a + 1; b < n_atoms; ++b) {
                    const auto* ed_b = params.element_data(atoms[b].Z);
                    if (!ed_a || !ed_b) continue;
                    double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
                    double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
                    double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (R < 1e-12) continue;
                    const auto* dp = params.diatomic(atoms[a].Z, atoms[b].Z);
                    E_core += pm6_core_core_repulsion(
                        atoms[a].Z, atoms[b].Z, R,
                        *ed_a, *ed_b, dp);
                }
            }

            // ---- Total energy:  E = ½ Tr[P·(H + F)] + E_core ----
            double E_total = 0.0;
            for (int mu = 0; mu < n_basis; ++mu) {
                E_total += D(mu, mu) * (H(mu, mu) + F(mu, mu));
                for (int nu = mu + 1; nu < n_basis; ++nu)
                    E_total += 2.0 * D(mu, nu) * (H(mu, nu) + F(mu, nu));
            }
            E_total *= 0.5;
            E_total += E_core;

            result.energy = E_total;
            result.e_electronic = E_total - E_core;
            result.e_core = E_core;
            result.mo_energies = eps;
            result.mo_coeffs = C;
            result.density = D;
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_iter = iter;
            result.converged = true;
            return result;
        }
    }

    result.n_iter = max_iter;
    result.density = D;
    return result;
}

PM6Result run_pm6(
    const Molecule& mol,
    const PM6ParameterSet& params,
    int max_iter,
    double conv_tol) {
    return run_pm6_impl(mol, params, max_iter, conv_tol, nullptr);
}

PM6Result run_pm6_with_density(
    const Molecule& mol,
    const PM6ParameterSet& params,
    const Eigen::MatrixXd& initial_density,
    int max_iter,
    double conv_tol) {
    return run_pm6_impl(
        mol, params, max_iter, conv_tol, &initial_density);
}

PM6Result run_pm6_with_smearing(
    const Molecule& mol,
    const PM6ParameterSet& params,
    double electronic_temperature,
    int max_iter,
    double conv_tol) {
    if (!(electronic_temperature > 0.0)
        || !std::isfinite(electronic_temperature)) {
        throw std::invalid_argument(
            "PM6 smearing recovery needs finite electronic_temperature > 0");
    }
    return run_pm6_impl(
        mol, params, max_iter, conv_tol, nullptr,
        electronic_temperature);
}

Eigen::MatrixXd compute_pm6_gradient_fd_from_result(
    const Molecule& mol,
    const PM6ParameterSet& params,
    const PM6Result& base,
    double h,
    int max_iter,
    double conv_tol) {
    if (!(h > 0.0) || !std::isfinite(h)) {
        throw std::invalid_argument(
            "PM6 finite-difference gradient needs finite h > 0.");
    }
    const auto n_atoms = static_cast<int>(mol.atoms().size());
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);

    (void)detail::checked_fd_energy(
        base, "PM6", mol, -1, -1, "base", h, max_iter);

    for (int a = 0; a < n_atoms; ++a) {
        for (int d = 0; d < 3; ++d) {
            double pair_h = h;
            for (int refinement = 0;
                 refinement <= detail::kFdGradientMaxStepRefinements;
                 ++refinement) {
                std::vector<Atom> atoms_p(
                    mol.atoms().begin(), mol.atoms().end());
                std::vector<Atom> atoms_m(
                    mol.atoms().begin(), mol.atoms().end());
                atoms_p[static_cast<size_t>(a)].xyz[static_cast<size_t>(d)]
                    += pair_h;
                atoms_m[static_cast<size_t>(a)].xyz[static_cast<size_t>(d)]
                    -= pair_h;
                Molecule mol_p(atoms_p, mol.charge(), mol.multiplicity());
                Molecule mol_m(atoms_m, mol.charge(), mol.multiplicity());
                const auto result_p = run_pm6_impl(
                    mol_p, params, max_iter, conv_tol, &base.density);
                const auto result_m = run_pm6_impl(
                    mol_m, params, max_iter, conv_tol, &base.density);
                const bool energies_valid =
                    result_p.converged && result_m.converged
                    && std::isfinite(result_p.energy)
                    && std::isfinite(result_m.energy);
                if (!energies_valid
                    && refinement < detail::kFdGradientMaxStepRefinements) {
                    pair_h *= 0.1;
                    continue;
                }
                const double ep = detail::checked_fd_energy(
                    result_p, "PM6", mol, a, d, "+h", pair_h, max_iter);
                const double em = detail::checked_fd_energy(
                    result_m, "PM6", mol, a, d, "-h", pair_h, max_iter);
                if (std::abs(ep - em) <= detail::kFdGradientMaxPairDeltaHa) {
                    grad(a, d) = (ep - em) / (2.0 * pair_h);
                    break;
                }
                if (refinement == detail::kFdGradientMaxStepRefinements) {
                    detail::check_fd_pair_delta(
                        ep, em, "PM6", mol, a, d, pair_h);
                }
                pair_h *= 0.1;
            }
        }
    }
    return grad;
}

Eigen::MatrixXd compute_pm6_gradient_fd(
    const Molecule& mol,
    const PM6ParameterSet& params,
    double h,
    int max_iter,
    double conv_tol) {
    const auto base = run_pm6(mol, params, max_iter, conv_tol);
    return compute_pm6_gradient_fd_from_result(
        mol, params, base, h, max_iter, conv_tol);
}

// ---------------------------------------------------------------------------
// Unrestricted PM6 (UPM6)
// ---------------------------------------------------------------------------

UPM6Result run_upm6(
    const Molecule& mol,
    const PM6ParameterSet& params,
    int max_iter,
    double conv_tol) {

    // ---- Validate ----
    const std::string active_method = params.method_name();
    if (active_method != "pm6") {
        throw std::invalid_argument(
            "UPM6 driver requires PM6 parameters, not "
            + active_method);
    }
    validate_scf_controls(max_iter, conv_tol, "UPM6");
    int mult = mol.multiplicity();
    if (mult < 1)
        throw std::invalid_argument("UPM6: multiplicity must be >= 1");
    if (mol.atoms().empty())
        throw std::invalid_argument("UPM6: empty molecule");

    // Check elements
    {
        std::vector<int> missing;
        for (const auto& atom : mol.atoms())
            if (!params.has_element(atom.Z)) missing.push_back(atom.Z);
        if (!missing.empty()) {
            std::string msg = "UPM6: element(s) not in parameter set: ";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += "Z=" + std::to_string(missing[i]);
            }
            throw std::invalid_argument(msg);
        }
    }

    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());
    reject_coincident_distinct_atoms(mol, "UPM6");
    reject_unsupported_nddo_d_shells(mol, params, "UPM6");

    // ---- AO basis ----
    std::vector<int> ao_Z, ao_atom, ao_type;
    std::vector<int> atom_ao_start(n_atoms, 0);
    std::vector<int> atom_n_ao(n_atoms, 0);
    int n_basis = 0;
    for (int a = 0; a < n_atoms; ++a) {
        int Z = atoms[a].Z;
        const auto* ed = params.element_data(Z);
        // H/He carry only a 1s valence orbital; everything else gets the sp set.
        // (NDDOElementData::n_orbitals defaults to 4 and the loaders never
        // override it for H, which silently gave H a spurious 2p shell — once the
        // electron–core attraction entered H, the SCF dumped the whole valence
        // density onto that phantom H-p shell.  MSINDO gives H one orbital.)
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

    const int n_el = valence_electron_count(mol);
    const auto [n_alpha, n_beta] = unrestricted_occupation_counts(
        n_el, mult, n_basis, "UPM6");

    // ---- Pre-load integrals ----
    double ev2ha = 1.0 / 27.2114;
    std::vector<double> U_s(n_basis, 0.0), G_ss(n_basis, 0.0),
                        G_pp(n_basis, 0.0), G_sp(n_basis, 0.0),
                        G_p2(n_basis, 0.0), H_sp(n_basis, 0.0),
                        beta(n_basis, 0.0), rho(n_basis, 0.0);
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
        core_charge[a] = pm6_core_charge(atoms[a].Z);

    struct ExactSPHydrogenPair {
        int heavy_start;
        int hydrogen_ao;
        PM6SPHydrogenPair integrals;
    };
    struct ExactSPSPPair {
        int first_start;
        int second_start;
        PM6SPSPPair integrals;
    };
    std::vector<ExactSPHydrogenPair> exact_sp_hydrogen_pairs;
    std::vector<ExactSPSPPair> exact_sp_sp_pairs;
    std::vector<std::vector<int>> exact_pair_index(
        n_atoms, std::vector<int>(n_atoms, -1));
    if (params.method_name() == "pm6") {
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = a + 1; b < n_atoms; ++b) {
                int heavy = -1;
                int hydrogen = -1;
                if (atoms[a].Z == 1 && atoms[b].Z > 2) {
                    hydrogen = a;
                    heavy = b;
                } else if (atoms[b].Z == 1 && atoms[a].Z > 2) {
                    hydrogen = b;
                    heavy = a;
                }
                if (heavy < 0 || atom_n_ao[heavy] != 4) continue;
                const auto* ed_x = params.element_data(atoms[heavy].Z);
                const auto* ed_h = params.element_data(1);
                if (!ed_x || !ed_h || ed_x->has_d) continue;
                Eigen::Vector3d displacement;
                for (int d = 0; d < 3; ++d) {
                    displacement[d] = atoms[hydrogen].xyz[d]
                        - atoms[heavy].xyz[d];
                }
                const int index = static_cast<int>(
                    exact_sp_hydrogen_pairs.size());
                exact_sp_hydrogen_pairs.push_back({
                    atom_ao_start[heavy],
                    atom_ao_start[hydrogen],
                    pm6_sp_hydrogen_pair(
                        atoms[heavy].Z, displacement, *ed_x, *ed_h)});
                exact_pair_index[a][b] = index;
                exact_pair_index[b][a] = index;
            }
        }
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = a + 1; b < n_atoms; ++b) {
                if (atoms[a].Z <= 2 || atoms[b].Z <= 2
                    || atom_n_ao[a] != 4 || atom_n_ao[b] != 4) {
                    continue;
                }
                const auto* ed_a = params.element_data(atoms[a].Z);
                const auto* ed_b = params.element_data(atoms[b].Z);
                if (!ed_a || !ed_b || ed_a->has_d || ed_b->has_d) continue;
                Eigen::Vector3d displacement;
                for (int d = 0; d < 3; ++d) {
                    displacement[d] = atoms[b].xyz[d] - atoms[a].xyz[d];
                }
                exact_sp_sp_pairs.push_back({
                    atom_ao_start[a],
                    atom_ao_start[b],
                    pm6_sp_sp_pair(
                        atoms[a].Z,
                        atoms[b].Z,
                        displacement,
                        *ed_a,
                        *ed_b)});
                exact_pair_index[a][b] = 0;
                exact_pair_index[b][a] = 0;
            }
        }
    }

    // ---- SCF ----
    Eigen::MatrixXd Da = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd Db = Eigen::MatrixXd::Zero(n_basis, n_basis);
    UPM6Result result;
    result.n_basis = n_basis;
    result.n_alpha = n_alpha;
    result.n_beta = n_beta;

    for (int iter = 1; iter <= max_iter; ++iter) {
        Eigen::MatrixXd D_tot = Da + Db;
        Eigen::MatrixXd Fa = Eigen::MatrixXd::Zero(n_basis, n_basis);
        Eigen::MatrixXd Fb = Eigen::MatrixXd::Zero(n_basis, n_basis);
        Eigen::MatrixXd H  = Eigen::MatrixXd::Zero(n_basis, n_basis);

        // ---- ONE-CENTRE ----
        for (int a = 0; a < n_atoms; ++a) {
            std::vector<int> idx;
            for (int mu = 0; mu < n_basis; ++mu)
                if (ao_atom[mu] == a) idx.push_back(mu);
            int n_ao_a = static_cast<int>(idx.size());

            for (int i = 0; i < n_ao_a; ++i) {
                int mu = idx[i];
                int t_mu = ao_type[mu];
                Fa(mu, mu) += U_s[mu]; Fb(mu, mu) += U_s[mu];
                H(mu, mu)  += U_s[mu];

                // Coulomb: Σ_ν D_tot(ν,ν) (μμ|νν)
                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j];
                    int t_nu = ao_type[nu];
                    double coul = 0.0;
                    if (t_mu == 0 && t_nu == 0) coul = G_ss[mu];
                    else if (t_mu == 0 && t_nu > 0) coul = G_sp[mu];
                    else if (t_mu > 0 && t_nu == 0) coul = G_sp[nu];
                    else if (t_mu > 0 && t_nu > 0 && t_mu == t_nu) coul = G_pp[mu];
                    else coul = G_p2[mu];
                    double c = D_tot(nu, nu) * coul;
                    Fa(mu, mu) += c; Fb(mu, mu) += c;
                }

                // Exchange: −Σ_ν D_spin(ν,ν) (μν|μν)
                for (int j = 0; j < n_ao_a; ++j) {
                    int nu = idx[j];
                    int t_nu = ao_type[nu];
                    double exch = 0.0;
                    if (mu == nu) exch = (t_mu == 0) ? G_ss[mu] : G_pp[mu];
                    else if (t_mu == 0 && t_nu > 0) exch = H_sp[mu];
                    else if (t_mu > 0 && t_nu == 0) exch = H_sp[nu];
                    else exch = 0.5 * (G_pp[mu] - G_p2[mu]);
                    Fa(mu, mu) -= Da(nu, nu) * exch;
                    Fb(mu, mu) -= Db(nu, nu) * exch;
                }
            }

            // Off-diagonal one-centre
            for (int i = 0; i < n_ao_a; ++i) {
                for (int j = i + 1; j < n_ao_a; ++j) {
                    int mu = idx[i], nu = idx[j];
                    int t_mu = ao_type[mu], t_nu = ao_type[nu];
                    double exch = 0.0, coul = 0.0;
                    if (t_mu == 0 && t_nu > 0) { exch = H_sp[mu]; coul = G_sp[mu]; }
                    else if (t_mu > 0 && t_nu == 0) { exch = H_sp[nu]; coul = G_sp[nu]; }
                    else { exch = 0.5 * (G_pp[mu] - G_p2[mu]); coul = G_p2[mu]; }
                    // Unrestricted NDDO one-center off-diagonal:
                    // Fα_μν = (Pα + 2Pβ)·(μν|μν) − Pα·(μμ|νν)
                    // Fβ_μν = (2Pα + Pβ)·(μν|μν) − Pβ·(μμ|νν)
                    double fa = (Da(mu, nu) + 2.0 * Db(mu, nu)) * exch - Da(mu, nu) * coul;
                    double fb = (2.0 * Da(mu, nu) + Db(mu, nu)) * exch - Db(mu, nu) * coul;
                    Fa(mu, nu) += fa; Fa(nu, mu) += fa;
                    Fb(mu, nu) += fb; Fb(nu, mu) += fb;
                }
            }
        }

        // ---- TWO-CENTRE: nuclear attraction ----
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = 0; b < n_atoms; ++b) {
                if (a == b) continue;
                if (exact_pair_index[a][b] >= 0) continue;
                double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
                double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
                double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                double q_B = 0.0;
                for (int nu = 0; nu < n_basis; ++nu)
                    if (ao_atom[nu] == b) q_B += D_tot(nu, nu);

                // Use AO-indexed rho (matching fixed PM6)
                int mu_a0 = 0, mu_b0 = 0;
                for (int mu = 0; mu < n_basis; ++mu)
                    if (ao_atom[mu] == a) { mu_a0 = mu; break; }
                for (int mu = 0; mu < n_basis; ++mu)
                    if (ao_atom[mu] == b) { mu_b0 = mu; break; }
                double g_ab = gamma_ab(R, rho[mu_a0], rho[mu_b0]);
                const auto* ed_a = params.element_data(atoms[a].Z);
                const auto* ed_b = params.element_data(atoms[b].Z);
                const double g_core = params.method_name() == "pm6"
                    ? pm6_electron_core_gamma(R, *ed_a, *ed_b)
                    : g_ab;
                double V_AB = q_B * g_ab - core_charge[b] * g_core;
                double V_core = -core_charge[b] * g_core;
                for (int mu = 0; mu < n_basis; ++mu)
                    if (ao_atom[mu] == a) {
                        Fa(mu, mu) += V_AB; Fb(mu, mu) += V_AB;
                        H(mu, mu) += V_core;
                    }
            }
        }

        for (const auto& pair : exact_sp_hydrogen_pairs) {
            const int x0 = pair.heavy_start;
            const int h = pair.hydrogen_ao;
            const auto& eri = pair.integrals.electron_repulsion;
            const auto& core_x = pair.integrals.heavy_core;
            const double d_hh = D_tot(h, h);
            Fa.block<4, 4>(x0, x0) += d_hh * eri + core_x;
            Fb.block<4, 4>(x0, x0) += d_hh * eri + core_x;
            H.block<4, 4>(x0, x0) += core_x;
            const double j_h =
                (D_tot.block<4, 4>(x0, x0).cwiseProduct(eri)).sum();
            Fa(h, h) += j_h + pair.integrals.hydrogen_core;
            Fb(h, h) += j_h + pair.integrals.hydrogen_core;
            H(h, h) += pair.integrals.hydrogen_core;
            const Eigen::Vector4d exchange_a =
                -eri * Da.block<4, 1>(x0, h);
            const Eigen::Vector4d exchange_b =
                -eri * Db.block<4, 1>(x0, h);
            Fa.block<4, 1>(x0, h) += exchange_a;
            Fa.block<1, 4>(h, x0) += exchange_a.transpose();
            Fb.block<4, 1>(x0, h) += exchange_b;
            Fb.block<1, 4>(h, x0) += exchange_b.transpose();
        }

        for (const auto& pair : exact_sp_sp_pairs) {
            const int a0 = pair.first_start;
            const int b0 = pair.second_start;
            const auto& eri = pair.integrals.electron_repulsion;
            Eigen::Matrix4d j_a = Eigen::Matrix4d::Zero();
            Eigen::Matrix4d j_b = Eigen::Matrix4d::Zero();
            Eigen::Matrix4d exchange_a = Eigen::Matrix4d::Zero();
            Eigen::Matrix4d exchange_b = Eigen::Matrix4d::Zero();
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    const int first_pair = 4 * mu + nu;
                    for (int la = 0; la < 4; ++la) {
                        for (int si = 0; si < 4; ++si) {
                            const double integral =
                                eri(first_pair, 4 * la + si);
                            j_a(mu, nu) +=
                                D_tot(b0 + la, b0 + si) * integral;
                            j_b(la, si) +=
                                D_tot(a0 + mu, a0 + nu) * integral;
                            exchange_a(mu, la) -=
                                Da(a0 + nu, b0 + si) * integral;
                            exchange_b(mu, la) -=
                                Db(a0 + nu, b0 + si) * integral;
                        }
                    }
                }
            }
            const auto& core_a = pair.integrals.first_core;
            const auto& core_b = pair.integrals.second_core;
            Fa.block<4, 4>(a0, a0) += j_a + core_a;
            Fb.block<4, 4>(a0, a0) += j_a + core_a;
            Fa.block<4, 4>(b0, b0) += j_b + core_b;
            Fb.block<4, 4>(b0, b0) += j_b + core_b;
            H.block<4, 4>(a0, a0) += core_a;
            H.block<4, 4>(b0, b0) += core_b;
            Fa.block<4, 4>(a0, b0) += exchange_a;
            Fa.block<4, 4>(b0, a0) += exchange_a.transpose();
            Fb.block<4, 4>(a0, b0) += exchange_b;
            Fb.block<4, 4>(b0, a0) += exchange_b.transpose();
        }

        // ---- TWO-CENTRE: resonance + exchange ----
        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = ao_atom[mu];
            int t_mu = ao_type[mu];
            for (int nu = mu + 1; nu < n_basis; ++nu) {
                int a_nu = ao_atom[nu];
                if (a_mu == a_nu) continue;
                int t_nu = ao_type[nu];
                double dx = atoms[a_mu].xyz[0] - atoms[a_nu].xyz[0];
                double dy = atoms[a_mu].xyz[1] - atoms[a_nu].xyz[1];
                double dz = atoms[a_mu].xyz[2] - atoms[a_nu].xyz[2];
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;

                double S_val = nddo_overlap_from_Z(params, ao_Z[mu], t_mu,
                                                   ao_Z[nu], t_nu, dx, dy, dz, R);
                double beta_val = 0.5 * (beta[mu] + beta[nu]) * S_val;
                H(mu, nu) = beta_val; H(nu, mu) = beta_val;
                Fa(mu, nu) += beta_val; Fa(nu, mu) += beta_val;
                Fb(mu, nu) += beta_val; Fb(nu, mu) += beta_val;

                if (exact_pair_index[a_mu][a_nu] >= 0) continue;

                double eta_ex = 0.5 * (rho[mu] + rho[nu]);
                double g_ex = 1.0 / std::sqrt(R * R + eta_ex * eta_ex);
                Fa(mu, nu) -= Da(mu, nu) * g_ex;
                Fa(nu, mu) -= Da(nu, mu) * g_ex;
                Fb(mu, nu) -= Db(mu, nu) * g_ex;
                Fb(nu, mu) -= Db(nu, mu) * g_ex;
            }
        }

        // ---- Diagonalize ----
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_a(Fa);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_b(Fb);
        if (solver_a.info() != Eigen::Success || solver_b.info() != Eigen::Success)
            throw std::runtime_error("UPM6: diagonalisation failed");

        Eigen::MatrixXd Ca_occ = solver_a.eigenvectors().leftCols(n_alpha);
        Eigen::MatrixXd Cb_occ = solver_b.eigenvectors().leftCols(n_beta);
        Eigen::MatrixXd Da_new = Ca_occ * Ca_occ.transpose();
        Eigen::MatrixXd Db_new = Cb_occ * Cb_occ.transpose();

        double delta_a = (Da_new - Da).cwiseAbs().maxCoeff();
        double delta_b = (Db_new - Db).cwiseAbs().maxCoeff();
        Da = Da_new; Db = Db_new;

        if (std::max(delta_a, delta_b) < conv_tol) {
            // Core-core repulsion
            double E_core = 0.0;
            for (int a = 0; a < n_atoms; ++a) {
                const auto* ed_a = params.element_data(atoms[a].Z);
                for (int b = a + 1; b < n_atoms; ++b) {
                    const auto* ed_b = params.element_data(atoms[b].Z);
                    if (!ed_a || !ed_b) continue;
                    double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
                    double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
                    double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (R < 1e-12) continue;
                    const auto* dp =
                        params.diatomic(atoms[a].Z, atoms[b].Z);
                    E_core += pm6_core_core_repulsion(
                        atoms[a].Z, atoms[b].Z, R,
                        *ed_a, *ed_b, dp);
                }
            }

            double E_total = 0.0;
            D_tot = Da + Db;
            for (int mu = 0; mu < n_basis; ++mu) {
                E_total += Da(mu, mu) * (H(mu, mu) + Fa(mu, mu));
                E_total += Db(mu, mu) * (H(mu, mu) + Fb(mu, mu));
                for (int nu = mu + 1; nu < n_basis; ++nu) {
                    E_total += 2.0 * Da(mu, nu) * (H(mu, nu) + Fa(mu, nu));
                    E_total += 2.0 * Db(mu, nu) * (H(mu, nu) + Fb(mu, nu));
                }
            }
            E_total *= 0.5;
            E_total += E_core;

            const auto eps_alpha = solver_a.eigenvalues();
            const auto eps_beta = solver_b.eigenvalues();
            result.e_electronic = E_total - E_core;
            result.mo_energies.resize(2 * n_basis);
            result.mo_energies.head(n_basis) = eps_alpha;
            result.mo_energies.tail(n_basis) = eps_beta;
            result.mo_coeffs.resize(n_basis, 2 * n_basis);
            result.mo_coeffs.leftCols(n_basis) = solver_a.eigenvectors();
            result.mo_coeffs.rightCols(n_basis) = solver_b.eigenvectors();

            result.energy = E_total;
            result.e_core = E_core;
            result.density_alpha = Da;
            result.density_beta = Db;
            result.n_basis = n_basis;
            result.n_alpha = n_alpha;
            result.n_beta  = n_beta;
            result.n_iter  = iter;
            result.converged = true;
            return result;
        }
    }

    result.density_alpha = Da;
    result.density_beta = Db;
    result.n_iter = max_iter;
    result.converged = false;
    return result;
}

Eigen::MatrixXd compute_upm6_gradient_fd(
    const Molecule& mol,
    const PM6ParameterSet& params,
    double h,
    int max_iter,
    double conv_tol) {
    return detail::checked_central_fd_gradient(
        mol, params, h, max_iter, conv_tol, "UPM6", run_upm6);
}


}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
