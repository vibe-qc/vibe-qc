#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/molecule.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {
namespace detail {

constexpr double kFdGradientMaxPairDeltaHa = 2.0e-2;
constexpr int kFdGradientMaxStepRefinements = 3;

inline const char* fd_axis_label(int axis) {
    switch (axis) {
        case 0: return "x";
        case 1: return "y";
        case 2: return "z";
        default: return "?";
    }
}

inline std::string fd_scalar(double value) {
    std::ostringstream out;
    out << std::setprecision(15) << value;
    return out.str();
}

inline std::string fd_element_list(const Molecule& mol) {
    std::ostringstream out;
    out << "Z=[";
    const auto& atoms = mol.atoms();
    for (size_t i = 0; i < atoms.size(); ++i) {
        if (i > 0) out << ",";
        out << atoms[i].Z;
    }
    out << "]";
    return out.str();
}

inline std::string fd_location(
    const Molecule& mol,
    int atom_index,
    int axis) {
    if (atom_index < 0) {
        return "base geometry";
    }
    const auto& atom = mol.atoms()[static_cast<size_t>(atom_index)];
    std::ostringstream out;
    out << "atom " << (atom_index + 1) << " (Z=" << atom.Z << "), axis "
        << fd_axis_label(axis);
    return out.str();
}

template <typename Result>
inline double checked_fd_energy(
    const Result& result,
    const char* method,
    const Molecule& mol,
    int atom_index,
    int axis,
    const char* displacement_label,
    double h,
    int max_iter) {
    if (!result.converged) {
        std::ostringstream msg;
        msg << method << " finite-difference gradient SCF did not converge "
            << "for " << fd_location(mol, atom_index, axis) << " ("
            << displacement_label << ", h=" << fd_scalar(h)
            << " bohr) after " << result.n_iter << " of " << max_iter
            << " iterations; " << fd_element_list(mol)
            << ". Refusing to return an untrusted finite-difference force.";
        throw std::runtime_error(msg.str());
    }
    if (!std::isfinite(result.energy)) {
        std::ostringstream msg;
        msg << method << " finite-difference gradient SCF returned a "
            << "non-finite energy for " << fd_location(mol, atom_index, axis)
            << " (" << displacement_label << ", h=" << fd_scalar(h)
            << " bohr); " << fd_element_list(mol)
            << ". Refusing to return an untrusted finite-difference force.";
        throw std::runtime_error(msg.str());
    }
    return result.energy;
}

inline void check_fd_pair_delta(
    double ep,
    double em,
    const char* method,
    const Molecule& mol,
    int atom_index,
    int axis,
    double h) {
    const double delta = std::abs(ep - em);
    if (delta <= kFdGradientMaxPairDeltaHa) {
        return;
    }
    std::ostringstream msg;
    msg << method << " finite-difference gradient is discontinuous for "
        << fd_location(mol, atom_index, axis) << ": E(+h)=" << fd_scalar(ep)
        << " Ha, E(-h)=" << fd_scalar(em) << " Ha, |delta|="
        << fd_scalar(delta) << " Ha over h=" << fd_scalar(h)
        << " bohr exceeds " << fd_scalar(kFdGradientMaxPairDeltaHa)
        << " Ha; " << fd_element_list(mol)
        << ". This indicates an SCC root switch or discontinuous energy "
        << "surface, so the optimizer must fail closed instead of using the "
        << "derived force.";
    throw std::runtime_error(msg.str());
}

template <typename ParamSet, typename RunFn>
Eigen::MatrixXd checked_central_fd_gradient(
    const Molecule& mol,
    const ParamSet& params,
    double h,
    int max_iter,
    double conv_tol,
    const char* method,
    RunFn run) {
    if (!(h > 0.0) || !std::isfinite(h)) {
        throw std::invalid_argument(
            std::string(method) + " finite-difference gradient needs finite h > 0."
        );
    }
    const auto n_atoms = static_cast<int>(mol.atoms().size());
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);

    const auto base = run(mol, params, max_iter, conv_tol);
    (void)checked_fd_energy(base, method, mol, -1, -1, "base", h, max_iter);

    for (int a = 0; a < n_atoms; ++a) {
        for (int d = 0; d < 3; ++d) {
            double pair_h = h;
            for (int refinement = 0;
                 refinement <= kFdGradientMaxStepRefinements;
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
                const auto result_p = run(mol_p, params, max_iter, conv_tol);
                const auto result_m = run(mol_m, params, max_iter, conv_tol);
                const bool energies_valid =
                    result_p.converged && result_m.converged
                    && std::isfinite(result_p.energy)
                    && std::isfinite(result_m.energy);
                if (!energies_valid
                    && refinement < kFdGradientMaxStepRefinements) {
                    pair_h *= 0.1;
                    continue;
                }
                const double ep = checked_fd_energy(
                    result_p, method, mol, a, d, "+h", pair_h, max_iter);
                const double em = checked_fd_energy(
                    result_m, method, mol, a, d, "-h", pair_h, max_iter);
                if (std::abs(ep - em) <= kFdGradientMaxPairDeltaHa) {
                    grad(a, d) = (ep - em) / (2.0 * pair_h);
                    break;
                }
                if (refinement == kFdGradientMaxStepRefinements) {
                    check_fd_pair_delta(
                        ep, em, method, mol, a, d, pair_h);
                }
                pair_h *= 0.1;
            }
        }
    }
    return grad;
}

}  // namespace detail
}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
