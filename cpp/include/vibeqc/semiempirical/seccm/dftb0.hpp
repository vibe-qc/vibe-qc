// DFTB0 adapter for a frozen semiempirical cyclic-cluster topology.
//
// This adapter consumes Hamiltonian-independent WS records. It does not
// rebuild topology, add SCC electrostatics, or reuse MSINDO CCM energy terms.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

struct DFTB0SECCMResult : ParameterIdentifiedResult {
    double energy = 0.0;  // Total energy per primitive cell.
    double electronic_energy = 0.0;
    double repulsive_energy = 0.0;
    double long_range_energy = 0.0;
    double dispersion_energy = 0.0;
    double specific_energy = 0.0;
    double total_cyclic_energy = 0.0;
    double cyclic_electronic_energy = 0.0;
    double cyclic_repulsive_energy = 0.0;
    double homo_lumo_gap = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd hamiltonian;
    Eigen::MatrixXd gradient;
    bool has_gradient = false;
    int n_basis = 0;
    int n_occ = 0;
    int group_order = 0;
    int n_records = 0;
};

DFTB0SECCMResult run_dftb0_seccm(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    double hermiticity_tolerance = 1.0e-10,
    double gap_tolerance = 1.0e-8,
    bool compute_gradient = false);

Eigen::MatrixXd compute_dftb0_seccm_gradient(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    const DFTB0SECCMResult& result,
    int group_order,
    double geometry_tolerance);

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
