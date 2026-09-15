#include "vibeqc/semiempirical/kpoints_dftb_fd_batch.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {
namespace semiempirical {
namespace {

void validate_fd_options(
    const PeriodicSystem& system,
    const KPointDFTBFDBatchOptions& options,
    const char* caller) {
    const std::string prefix = std::string(caller) + ": ";
    if (!options.compute_gradient && !options.compute_stress) {
        throw std::invalid_argument(
            prefix + "must request gradient or stress");
    }
    if (options.compute_gradient &&
        (!std::isfinite(options.coordinate_step) ||
         options.coordinate_step <= 0.0)) {
        throw std::invalid_argument(
            prefix + "coordinate_step must be finite and > 0");
    }
    if (options.compute_stress &&
        (!std::isfinite(options.strain_step) ||
         options.strain_step <= 0.0)) {
        throw std::invalid_argument(
            prefix + "strain_step must be finite and > 0");
    }
    if (system.dim < 1 || system.dim > 3) {
        throw std::invalid_argument(prefix + "system dim must be 1, 2, or 3");
    }
    const double determinant = system.lattice.determinant();
    if (options.compute_stress &&
        (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-14)) {
        throw std::invalid_argument(
            prefix + "stress requires nonzero cell volume");
    }
}

void transform_kmesh(
    const BlochKMesh& source,
    const Eigen::Matrix3d& deformation,
    BlochKMesh& target) {
    const Eigen::Matrix3d reciprocal_transform =
        deformation.inverse().transpose();
    for (std::size_t index = 0; index < source.kpoints.size(); ++index) {
        target.kpoints[index] = reciprocal_transform * source.kpoints[index];
    }
}

template <typename EnergyFunction>
KPointDFTBFDBatchResult compute_fd_batch(
    const PeriodicSystem& system,
    const BlochKMesh& kmesh,
    const KPointDFTBFDBatchOptions& options,
    EnergyFunction&& energy) {
    KPointDFTBFDBatchResult result;
    PeriodicSystem working_system = system;
    BlochKMesh working_kmesh = kmesh;
    result.system_workspace_copies = 1;
    result.kmesh_workspace_copies = 1;
    result.workspace_bytes =
        sizeof(PeriodicSystem) +
        working_system.unit_cell.capacity() * sizeof(Atom) +
        sizeof(BlochKMesh) +
        working_kmesh.kpoints.capacity() * sizeof(Eigen::Vector3d) +
        working_kmesh.weights.capacity() * sizeof(double) +
        working_kmesh.ir_mapping.capacity() * sizeof(int);

    const int n_atoms = static_cast<int>(system.unit_cell.size());
    if (options.compute_gradient) {
        result.gradient = Eigen::MatrixXd::Zero(n_atoms, 3);
        for (int atom = 0; atom < n_atoms; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const double coordinate = system.unit_cell[atom].xyz[axis];
                working_system.unit_cell[atom].xyz[axis] =
                    coordinate + options.coordinate_step;
                const double energy_plus = energy(working_system, kmesh);
                ++result.energy_evaluations;

                working_system.unit_cell[atom].xyz[axis] =
                    coordinate - options.coordinate_step;
                const double energy_minus = energy(working_system, kmesh);
                ++result.energy_evaluations;

                result.gradient(atom, axis) =
                    (energy_plus - energy_minus) /
                    (2.0 * options.coordinate_step);
                working_system.unit_cell[atom].xyz[axis] = coordinate;
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
                working_system.lattice = transform_plus * lattice;
                if (options.strain_positions) {
                    for (int atom = 0; atom < n_atoms; ++atom) {
                        const Eigen::Vector3d position(
                            coordinates[atom][0], coordinates[atom][1],
                            coordinates[atom][2]);
                        const Eigen::Vector3d transformed =
                            transform_plus * position;
                        for (int axis = 0; axis < 3; ++axis) {
                            working_system.unit_cell[atom].xyz[axis] =
                                transformed[axis];
                        }
                    }
                }
                transform_kmesh(kmesh, transform_plus, working_kmesh);
                const double energy_plus =
                    energy(working_system, working_kmesh);
                ++result.energy_evaluations;

                const Eigen::Matrix3d transform_minus =
                    Eigen::Matrix3d::Identity() - strain;
                working_system.lattice = transform_minus * lattice;
                if (options.strain_positions) {
                    for (int atom = 0; atom < n_atoms; ++atom) {
                        const Eigen::Vector3d position(
                            coordinates[atom][0], coordinates[atom][1],
                            coordinates[atom][2]);
                        const Eigen::Vector3d transformed =
                            transform_minus * position;
                        for (int axis = 0; axis < 3; ++axis) {
                            working_system.unit_cell[atom].xyz[axis] =
                                transformed[axis];
                        }
                    }
                }
                transform_kmesh(kmesh, transform_minus, working_kmesh);
                const double energy_minus =
                    energy(working_system, working_kmesh);
                ++result.energy_evaluations;

                result.stress(row, column) =
                    (energy_plus - energy_minus) /
                    (2.0 * options.strain_step * volume);
                working_system.lattice = lattice;
                for (int atom = 0; atom < n_atoms; ++atom) {
                    working_system.unit_cell[atom].xyz = coordinates[atom];
                }
            }
        }
    }

    result.lattice_cell_setups = result.energy_evaluations;
    return result;
}

}  // namespace

KPointDFTBFDBatchResult compute_dftb0_kpoints_fd_batch(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    double cutoff_bohr,
    const KPointDFTBFDBatchOptions& fd_options,
    const KPointOccupationOptions& occupation_options) {
    validate_fd_options(system, fd_options, "DFTB0 k-point FD batch");
    validate_kpoint_occupation_options(
        occupation_options, "DFTB0 k-point FD batch");
    validate_dftb_kpoint_request(
        system, params, kmesh, cutoff_bohr, "DFTB0 k-point FD batch");
    auto result = compute_fd_batch(
        system, kmesh, fd_options,
        [&params, &occupation_options, cutoff_bohr](
            const PeriodicSystem& displaced,
            const BlochKMesh& displaced_kmesh) {
            return run_dftb0_kpoints(
                       displaced, params, displaced_kmesh, cutoff_bohr,
                       occupation_options)
                .free_energy;
        });
    result.differentiated_free_energy =
        occupation_options.smearing_temperature > 0.0;
    return result;
}

KPointDFTBFDBatchResult compute_scc_dftb_kpoints_fd_batch(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    const SCCOptions& scc_options,
    double cutoff_bohr,
    const KPointDFTBFDBatchOptions& fd_options,
    const KPointOccupationOptions& occupation_options) {
    validate_fd_options(system, fd_options, "SCC-DFTB k-point FD batch");
    validate_kpoint_occupation_options(
        occupation_options, "SCC-DFTB k-point FD batch");
    validate_dftb_kpoint_request(
        system, params, kmesh, cutoff_bohr, "SCC-DFTB k-point FD batch");
    validate_scc_dftb_kpoint_options(
        scc_options, "SCC-DFTB k-point FD batch");
    auto result = compute_fd_batch(
        system, kmesh, fd_options,
        [&params, &scc_options, &occupation_options, cutoff_bohr](
            const PeriodicSystem& displaced,
            const BlochKMesh& displaced_kmesh) {
            const auto result = run_scc_dftb_kpoints(
                displaced, params, displaced_kmesh, scc_options, cutoff_bohr,
                occupation_options);
            if (!result.converged) {
                throw std::runtime_error(
                    "SCC-DFTB k-point FD batch: displaced SCC did not converge");
            }
            return result.free_energy;
        });
    result.differentiated_free_energy =
        occupation_options.smearing_temperature > 0.0;
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
