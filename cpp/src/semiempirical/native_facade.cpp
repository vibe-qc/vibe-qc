#include "vibeqc/semiempirical/native_facade.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/dftb0.hpp"
#include "vibeqc/semiempirical/methods/indo/indo_engine.hpp"
#include "vibeqc/semiempirical/methods/nddo/omx_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/scc_dftb.hpp"

namespace vibeqc {
namespace semiempirical {
namespace {

// MSINDO pins its parameters and reference outputs to the 1986-CODATA length
// convention declared by indo::ANGSTROM_TO_BOHR.  This adapter must use the
// exact inverse: a generic current-CODATA conversion followed by the engine's
// legacy conversion silently changes the molecular geometry.
constexpr double kMsindoBohrToAngstrom = 1.0 / indo::ANGSTROM_TO_BOHR;

template <typename Derived>
std::size_t eigen_bytes(const Eigen::DenseBase<Derived>& value) {
    return static_cast<std::size_t>(value.size()) *
           sizeof(typename Derived::Scalar);
}

bool molecule_is_unrestricted(const Molecule& molecule) {
    return molecule.multiplicity() != 1 || molecule.n_electrons() % 2 != 0;
}

void validate_route(const SemiempiricalNativeRoute& route,
                    const Molecule& molecule) {
    if (route.max_iter < 1) {
        throw std::invalid_argument("semiempirical native max_iter must be >= 1");
    }
    if (!std::isfinite(route.conv_tol) || route.conv_tol <= 0.0) {
        throw std::invalid_argument(
            "semiempirical native conv_tol must be finite and > 0");
    }
    if (!std::isfinite(route.charge_mixing)
        || route.charge_mixing <= 0.0 || route.charge_mixing > 1.0) {
        throw std::invalid_argument(
            "semiempirical native charge_mixing must be in (0, 1]");
    }

    const bool requested_unrestricted =
        route.spin == SemiempiricalNativeSpin::Unrestricted;
    if (requested_unrestricted != molecule_is_unrestricted(molecule)) {
        throw std::invalid_argument(
            "semiempirical native route spin does not match the molecule");
    }
}

SemiempiricalNativeResult base_result(
    const SemiempiricalNativeRoute& route) {
    SemiempiricalNativeResult result;
    result.route = route;
    return result;
}

void require_method(bool condition, const char* parameter_family) {
    if (!condition) {
        throw std::invalid_argument(
            std::string("semiempirical native route method does not match ") +
            parameter_family + " parameters");
    }
}

}  // namespace

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const SemiempiricalParameters& parameters) {
    validate_route(route, molecule);
    require_method(
        route.method == SemiempiricalNativeMethod::DFTB0 ||
            route.method == SemiempiricalNativeMethod::SCCDFTB,
        "DFTB");

    auto result = base_result(route);
    const bool unrestricted =
        route.spin == SemiempiricalNativeSpin::Unrestricted;
    if (route.method == SemiempiricalNativeMethod::DFTB0) {
        if (unrestricted) {
            const auto direct = run_udftb0(molecule, parameters);
            result.energy = direct.energy;
            result.e_electronic = direct.e_electronic;
            result.e_repulsive = direct.e_repulsive;
            result.n_basis = direct.n_basis;
            result.n_iter = 1;
            result.converged = true;
            result.retained_result_bytes =
                eigen_bytes(direct.mo_energies) +
                eigen_bytes(direct.mo_coeffs) +
                eigen_bytes(direct.density_alpha) +
                eigen_bytes(direct.density_beta) +
                eigen_bytes(direct.overlap) +
                eigen_bytes(direct.hamiltonian);
        } else {
            const auto direct = run_dftb0(molecule, parameters);
            result.energy = direct.energy;
            result.e_electronic = direct.e_electronic;
            result.e_repulsive = direct.e_repulsive;
            result.n_basis = direct.n_basis;
            result.n_iter = 1;
            result.converged = true;
            result.retained_result_bytes =
                eigen_bytes(direct.mo_energies) +
                eigen_bytes(direct.mo_coeffs) +
                eigen_bytes(direct.density) + eigen_bytes(direct.overlap) +
                eigen_bytes(direct.hamiltonian);
        }
        return result;
    }

    SCCOptions options;
    options.max_iter = route.max_iter;
    options.conv_tol_charge = route.conv_tol;
    options.charge_mixing = route.charge_mixing;
    if (unrestricted) {
        const auto direct = run_uscc_dftb(molecule, parameters, options);
        result.energy = direct.energy;
        result.e_electronic = direct.e_electronic;
        result.e_repulsive = direct.e_repulsive;
        result.e_scc = direct.e_scc;
        result.n_basis = direct.n_basis;
        result.n_iter = direct.n_iter;
        result.converged = direct.converged;
        result.retained_result_bytes =
            eigen_bytes(direct.mo_energies) +
            eigen_bytes(direct.mo_coeffs) +
            eigen_bytes(direct.density_alpha) +
            eigen_bytes(direct.density_beta) + eigen_bytes(direct.overlap) +
            eigen_bytes(direct.hamiltonian) + eigen_bytes(direct.charges);
    } else {
        const auto direct = run_scc_dftb(molecule, parameters, options);
        result.energy = direct.energy;
        result.e_electronic = direct.e_electronic;
        result.e_repulsive = direct.e_repulsive;
        result.e_scc = direct.e_scc;
        result.n_basis = direct.n_basis;
        result.n_iter = direct.n_iter;
        result.converged = direct.converged;
        result.retained_result_bytes =
            eigen_bytes(direct.mo_energies) +
            eigen_bytes(direct.mo_coeffs) + eigen_bytes(direct.density) +
            eigen_bytes(direct.overlap) + eigen_bytes(direct.hamiltonian) +
            eigen_bytes(direct.charges);
    }
    return result;
}

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const xtb::GFN2ParameterSet& parameters) {
    validate_route(route, molecule);
    require_method(route.method == SemiempiricalNativeMethod::GFN2XTB, "GFN2");
    if (route.spin == SemiempiricalNativeSpin::Unrestricted) {
        throw std::invalid_argument(
            "the unified native GFN2 route is closed-shell only");
    }

    xtb::XTBSccOptions options;
    options.max_iter = route.max_iter;
    options.conv_tol_charge = route.conv_tol;
    options.charge_mixing = route.charge_mixing;
    const auto direct = xtb::run_gfn2_xtb(molecule, parameters, options);

    auto result = base_result(route);
    result.energy = direct.energy;
    result.e_electronic = direct.e_electronic;
    result.e_repulsive = direct.e_repulsive;
    result.e_scc = direct.e_scc;
    result.n_basis = direct.n_basis;
    result.n_iter = direct.n_iter;
    result.converged = direct.converged;
    result.retained_result_bytes =
        eigen_bytes(direct.mo_energies) + eigen_bytes(direct.mo_coeffs) +
        eigen_bytes(direct.density) + eigen_bytes(direct.overlap) +
        eigen_bytes(direct.hamiltonian) + eigen_bytes(direct.charges);
    return result;
}

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const nddo::PM6ParameterSet& parameters) {
    validate_route(route, molecule);
    require_method(route.method == SemiempiricalNativeMethod::PM6, "PM6");
    require_method(parameters.method_name() == "pm6", "PM6 method-specific");

    auto result = base_result(route);
    if (route.spin == SemiempiricalNativeSpin::Unrestricted) {
        const auto direct = nddo::run_upm6(
            molecule, parameters, route.max_iter, route.conv_tol);
        result.energy = direct.energy;
        result.e_electronic = direct.e_electronic;
        result.e_core = direct.e_core;
        result.n_basis = direct.n_basis;
        result.n_iter = direct.n_iter;
        result.converged = direct.converged;
        result.retained_result_bytes =
            eigen_bytes(direct.mo_energies) +
            eigen_bytes(direct.mo_coeffs) +
            eigen_bytes(direct.density_alpha) +
            eigen_bytes(direct.density_beta);
    } else {
        const auto direct = nddo::run_pm6(
            molecule, parameters, route.max_iter, route.conv_tol);
        result.energy = direct.energy;
        result.e_electronic = direct.e_electronic;
        result.e_core = direct.e_core;
        result.n_basis = direct.n_basis;
        result.n_iter = direct.n_iter;
        result.converged = direct.converged;
        result.retained_result_bytes =
            eigen_bytes(direct.mo_energies) +
            eigen_bytes(direct.mo_coeffs) + eigen_bytes(direct.density);
    }
    return result;
}

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const nddo::OMxParameterSet& parameters) {
    validate_route(route, molecule);
    require_method(
        route.method == SemiempiricalNativeMethod::OM1 ||
            route.method == SemiempiricalNativeMethod::OM2 ||
            route.method == SemiempiricalNativeMethod::OM3,
        "OMx");
    std::string expected_method;
    switch (route.method) {
        case SemiempiricalNativeMethod::OM1: expected_method = "om1"; break;
        case SemiempiricalNativeMethod::OM2: expected_method = "om2"; break;
        case SemiempiricalNativeMethod::OM3: expected_method = "om3"; break;
        default: break;
    }
    require_method(
        parameters.method_name() == expected_method,
        "route-specific OMx");

    auto result = base_result(route);
    if (route.spin == SemiempiricalNativeSpin::Unrestricted) {
        const auto direct = nddo::run_uomx_v2(
            molecule, parameters, route.max_iter, route.conv_tol);
        result.energy = direct.energy;
        result.e_electronic = direct.e_electronic;
        result.e_core = direct.e_core;
        result.n_basis = direct.n_basis;
        result.n_iter = direct.n_iter;
        result.converged = direct.converged;
        result.retained_result_bytes =
            eigen_bytes(direct.mo_energies) +
            eigen_bytes(direct.mo_coeffs) +
            eigen_bytes(direct.density_alpha) +
            eigen_bytes(direct.density_beta);
    } else {
        const auto direct = nddo::run_omx_v2(
            molecule, parameters, route.max_iter, route.conv_tol);
        result.energy = direct.energy;
        result.e_electronic = direct.e_electronic;
        result.e_core = direct.e_core;
        result.n_basis = direct.n_basis;
        result.n_iter = direct.n_iter;
        result.converged = direct.converged;
        result.retained_result_bytes =
            eigen_bytes(direct.mo_energies) +
            eigen_bytes(direct.mo_coeffs) + eigen_bytes(direct.density);
    }
    return result;
}

SemiempiricalNativeResult run_native(
    const SemiempiricalNativeRoute& route,
    const Molecule& molecule,
    const indo::MsindoParameterSet& parameters) {
    validate_route(route, molecule);
    require_method(
        route.method == SemiempiricalNativeMethod::MSINDO ||
            route.method == SemiempiricalNativeMethod::MSINDONDDO,
        "MSINDO");

    const bool nddo = route.method == SemiempiricalNativeMethod::MSINDONDDO;
    const bool unrestricted =
        route.spin == SemiempiricalNativeSpin::Unrestricted;
    if (nddo && unrestricted) {
        throw std::invalid_argument(
            "the unified native MSINDO NDDO route is closed-shell only");
    }

    std::vector<int> atomic_numbers;
    std::vector<std::array<double, 3>> coordinates_angstrom;
    atomic_numbers.reserve(molecule.atoms().size());
    coordinates_angstrom.reserve(molecule.atoms().size());
    for (const auto& atom : molecule.atoms()) {
        atomic_numbers.push_back(atom.Z);
        coordinates_angstrom.push_back({
            atom.xyz[0] * kMsindoBohrToAngstrom,
            atom.xyz[1] * kMsindoBohrToAngstrom,
            atom.xyz[2] * kMsindoBohrToAngstrom,
        });
    }

    indo::MsindoResult direct;
    if (unrestricted) {
        direct = indo::run_msindo_core_uhf(
            atomic_numbers, coordinates_angstrom, parameters,
            route.max_iter, route.conv_tol, molecule.multiplicity(), nddo,
            molecule.charge());
    } else {
        direct = indo::run_msindo_core(
            atomic_numbers, coordinates_angstrom, parameters,
            route.max_iter, route.conv_tol, nddo, molecule.charge());
    }

    auto result = base_result(route);
    result.energy = direct.total_energy;
    result.e_electronic = direct.electronic_energy;
    result.binding_energy = direct.binding_energy;
    result.n_basis = static_cast<int>(direct.density.rows());
    result.n_iter = direct.n_iter;
    result.converged = direct.converged;
    result.retained_result_bytes =
        eigen_bytes(direct.mo_energies) + eigen_bytes(direct.density);
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
