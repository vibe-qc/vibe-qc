#include "vibeqc/semiempirical/methods/nddo/periodic_fd_batch.hpp"

#include "vibeqc/semiempirical/core/pair_lattice.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibeqc {
namespace semiempirical {
namespace nddo {
namespace {

void validate_options(const PeriodicSystem& system,
                      const PeriodicNDDOFDBatchOptions& options) {
    if (!options.compute_gradient && !options.compute_stress) {
        throw std::invalid_argument(
            "periodic NDDO FD batch must request gradient or stress");
    }
    if (options.compute_gradient &&
        (!std::isfinite(options.coordinate_step) ||
         options.coordinate_step <= 0.0)) {
        throw std::invalid_argument(
            "periodic NDDO FD coordinate_step must be > 0");
    }
    if (options.compute_stress &&
        (!std::isfinite(options.strain_step) || options.strain_step <= 0.0)) {
        throw std::invalid_argument(
            "periodic NDDO FD strain_step must be > 0");
    }
    if (system.dim < 1 || system.dim > 3) {
        throw std::invalid_argument(
            "periodic NDDO FD system dim must be 1, 2, or 3");
    }
    if (system.unit_cell.empty()) {
        throw std::invalid_argument(
            "periodic NDDO FD batch requires at least one atom");
    }
    const double determinant = system.lattice.determinant();
    if (options.compute_stress &&
        (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-14)) {
        throw std::invalid_argument(
            "periodic NDDO FD stress requires nonzero cell volume");
    }
}

std::vector<LatticeCell> prepare_cells(const PeriodicSystem& system,
                                       double cutoff_bohr,
                                       bool gamma_only_0) {
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)):
    // the FD batch must displace the same interaction set the SCF sums.
    auto cells = atom_pair_interaction_cells(system, cutoff_bohr);
    if (gamma_only_0) {
        cells = {cells[0]};
    }
    return cells;
}

template <typename Result>
double require_converged_energy(Result&& result, const char* method) {
    if (!result.converged) {
        throw std::runtime_error(
            std::string(method)
            + " displaced SCF did not converge; finite-difference "
              "derivatives are unavailable");
    }
    return result.energy;
}

template <typename CoordinateEnergyFunction, typename StrainEnergyFunction>
PeriodicNDDOFDBatchResult compute_fd_batch(
    const PeriodicSystem& system,
    const PeriodicNDDOFDBatchOptions& options,
    CoordinateEnergyFunction&& coordinate_energy,
    StrainEnergyFunction&& strain_energy) {
    PeriodicNDDOFDBatchResult result;
    PeriodicSystem working = system;
    result.system_workspace_copies = 1;
    result.workspace_bytes =
        sizeof(PeriodicSystem) +
        working.unit_cell.capacity() * sizeof(Atom);

    const int n_atoms = static_cast<int>(system.unit_cell.size());
    if (options.compute_gradient) {
        result.gradient = Eigen::MatrixXd::Zero(n_atoms, 3);
        for (int atom = 0; atom < n_atoms; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const double coordinate = system.unit_cell[atom].xyz[axis];
                working.unit_cell[atom].xyz[axis] =
                    coordinate + options.coordinate_step;
                const double energy_plus = coordinate_energy(working);
                ++result.energy_evaluations;

                working.unit_cell[atom].xyz[axis] =
                    coordinate - options.coordinate_step;
                const double energy_minus = coordinate_energy(working);
                ++result.energy_evaluations;

                result.gradient(atom, axis) =
                    (energy_plus - energy_minus) /
                    (2.0 * options.coordinate_step);
                working.unit_cell[atom].xyz[axis] = coordinate;
            }
        }
    }

    if (options.compute_stress) {
        const Eigen::Matrix3d lattice = system.lattice;
        const double volume = std::abs(lattice.determinant());
        std::vector<std::array<double, 3>> coordinates;
        coordinates.reserve(system.unit_cell.size());
        for (const auto& atom : system.unit_cell) {
            coordinates.push_back(atom.xyz);
        }
        result.workspace_bytes +=
            coordinates.capacity() * sizeof(std::array<double, 3>);

        for (int row = 0; row < system.dim; ++row) {
            for (int column = 0; column < system.dim; ++column) {
                Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
                strain(row, column) = options.strain_step;

                const Eigen::Matrix3d transform_plus =
                    Eigen::Matrix3d::Identity() + strain;
                working.lattice = transform_plus * lattice;
                if (options.strain_positions) {
                    for (int atom = 0; atom < n_atoms; ++atom) {
                        const Eigen::Vector3d position(
                            coordinates[atom][0], coordinates[atom][1],
                            coordinates[atom][2]);
                        const Eigen::Vector3d transformed =
                            transform_plus * position;
                        for (int axis = 0; axis < 3; ++axis) {
                            working.unit_cell[atom].xyz[axis] =
                                transformed[axis];
                        }
                    }
                }
                const double energy_plus = strain_energy(working);
                ++result.energy_evaluations;

                const Eigen::Matrix3d transform_minus =
                    Eigen::Matrix3d::Identity() - strain;
                working.lattice = transform_minus * lattice;
                if (options.strain_positions) {
                    for (int atom = 0; atom < n_atoms; ++atom) {
                        const Eigen::Vector3d position(
                            coordinates[atom][0], coordinates[atom][1],
                            coordinates[atom][2]);
                        const Eigen::Vector3d transformed =
                            transform_minus * position;
                        for (int axis = 0; axis < 3; ++axis) {
                            working.unit_cell[atom].xyz[axis] =
                                transformed[axis];
                        }
                    }
                }
                const double energy_minus = strain_energy(working);
                ++result.energy_evaluations;

                result.stress(row, column) =
                    (energy_plus - energy_minus) /
                    (2.0 * options.strain_step * volume);
                working.lattice = lattice;
                working.unit_cell = system.unit_cell;
            }
        }
    }

    return result;
}

}  // namespace

PeriodicNDDOFDBatchResult compute_pm6_gamma_fd_batch(
    const PeriodicSystem& system,
    const PM6ParameterSet& params,
    const PeriodicPM6Options& scf_options,
    const PeriodicNDDOFDBatchOptions& fd_options) {
    if (params.method_name() != "pm6") {
        throw std::invalid_argument(
            "Periodic PM6 derivatives require PM6 parameters, not "
            + params.method_name());
    }
    validate_options(system, fd_options);
    std::vector<LatticeCell> coordinate_cells;
    if (fd_options.compute_gradient) {
        coordinate_cells = prepare_cells(
            system, scf_options.cutoff_bohr, scf_options.gamma_only_0);
    }
    auto result = compute_fd_batch(
        system, fd_options,
        [&params, &scf_options, &coordinate_cells](
            const PeriodicSystem& displaced) {
            return require_converged_energy(
                run_pm6_gamma_with_cells(
                    displaced, params, scf_options, coordinate_cells),
                "Periodic PM6");
        },
        [&params, &scf_options](const PeriodicSystem& displaced) {
            return require_converged_energy(
                run_pm6_gamma(displaced, params, scf_options),
                "Periodic PM6");
        });
    result.workspace_bytes +=
        coordinate_cells.capacity() * sizeof(LatticeCell);
    result.lattice_cell_setups =
        (fd_options.compute_gradient ? 1 : 0) +
        (fd_options.compute_stress ? 2 * system.dim * system.dim : 0);
    return result;
}

PeriodicNDDOFDBatchResult compute_omx_gamma_fd_batch(
    const PeriodicSystem& system,
    const OMxParameterSet& params,
    const PeriodicOMxOptions& scf_options,
    const PeriodicNDDOFDBatchOptions& fd_options) {
    (void)system;
    (void)params;
    (void)scf_options;
    (void)fd_options;
    throw std::logic_error(
        "Bloch-periodic OMx derivatives are gated with the underlying "
        "periodic Hamiltonian; use the explicitly topology-bound "
        "OMx-SECCM route");
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
