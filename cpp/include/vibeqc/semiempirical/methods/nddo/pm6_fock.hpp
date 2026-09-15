// PM6 Fock matrix builder (NDDO formalism).
//
// Builds the Fock matrix for closed-shell PM6 using parameterized
// 1-center and 2-center integrals.  The basis is orthogonal (S=I).

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// PM6 energy result
// ---------------------------------------------------------------------------

struct PM6Result : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;     // variational 1/2 Tr[P(H+F)] component
    double e_core = 0.0;         // core-core repulsion
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    bool converged = false;
};

// ---------------------------------------------------------------------------
// Build PM6 Fock matrix and compute energy
// ---------------------------------------------------------------------------

PM6Result run_pm6(
    const Molecule& mol,
    const PM6ParameterSet& params,
    int max_iter = 100,
    double conv_tol = 1e-7);

PM6Result run_pm6_with_density(
    const Molecule& mol,
    const PM6ParameterSet& params,
    const Eigen::MatrixXd& initial_density,
    int max_iter = 100,
    double conv_tol = 1e-7);

PM6Result run_pm6_with_smearing(
    const Molecule& mol,
    const PM6ParameterSet& params,
    double electronic_temperature,
    int max_iter = 100,
    double conv_tol = 1e-7);

// ---------------------------------------------------------------------------
// Unrestricted PM6 (UPM6)
// ---------------------------------------------------------------------------

struct UPM6Result : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;     // variational 1/2 sum_sigma Tr[P(H+F)]
    double e_core = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density_alpha;
    Eigen::MatrixXd density_beta;
    int n_basis = 0;
    int n_alpha = 0;
    int n_beta = 0;
    int n_iter = 0;
    bool converged = false;
};

UPM6Result run_upm6(
    const Molecule& mol,
    const PM6ParameterSet& params,
    int max_iter = 100,
    double conv_tol = 1e-7);

// Two-center electron repulsion (Ohno-Klopman)
double gamma_ab(double R, double rho_a, double rho_b);

// Directed PM6 electron-core monopole.  The electron distribution on A uses
// po(1)=1/(2*GSS_A), while the core on B uses po(9), including the sparse
// MOPAC poc_ override.  This is deliberately not symmetric in A and B.
double pm6_electron_core_gamma(
    double R,
    const NDDOElementData& electron_atom,
    const NDDOElementData& core_atom);

// Exact NDDO s/p multipole block for a PM6 heavy-atom--hydrogen pair.
// ``electron_repulsion(mu,nu)`` is (mu nu | h h), ``heavy_core`` is the
// attraction of the heavy-atom AO pair to the hydrogen core, and
// ``hydrogen_core`` is <h | V_heavy-core | h>.  All values are Hartree.
struct PM6SPHydrogenPair {
    Eigen::Matrix4d electron_repulsion = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d heavy_core = Eigen::Matrix4d::Zero();
    double hydrogen_core = 0.0;
};

PM6SPHydrogenPair pm6_sp_hydrogen_pair(
    int Z_heavy,
    const Eigen::Vector3d& heavy_to_hydrogen,
    const NDDOElementData& heavy_atom,
    const NDDOElementData& hydrogen_atom);

// Exact NDDO multipole block for two PM6 s/p-heavy atoms.  A compound AO-pair
// index is ``4 * mu + nu``.  Thus electron_repulsion(4*mu+nu, 4*la+si) is
// (mu nu | la si); first_core and second_core are the directed attractions of
// each atom's AO products to the other atom's core.  All values are Hartree.
struct PM6SPSPPair {
    Eigen::Matrix<double, 16, 16> electron_repulsion =
        Eigen::Matrix<double, 16, 16>::Zero();
    Eigen::Matrix4d first_core = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d second_core = Eigen::Matrix4d::Zero();
};

PM6SPSPPair pm6_sp_sp_pair(
    int Z_first,
    int Z_second,
    const Eigen::Vector3d& first_to_second,
    const NDDOElementData& first_atom,
    const NDDOElementData& second_atom);

// Published PM6 core-core repulsion.
double pm6_core_core_repulsion(
    int Za, int Zb, double R,
    const NDDOElementData& ed_a, const NDDOElementData& ed_b,
    const NDDODiatomicParams* dp = nullptr);

// Finite-difference PM6 gradient
Eigen::MatrixXd compute_pm6_gradient_fd(
    const Molecule& mol,
    const PM6ParameterSet& params,
    double h = 0.001,
    int max_iter = 100,
    double conv_tol = 1e-7);

Eigen::MatrixXd compute_pm6_gradient_fd_from_result(
    const Molecule& mol,
    const PM6ParameterSet& params,
    const PM6Result& base,
    double h = 0.001,
    int max_iter = 100,
    double conv_tol = 1e-7);

// Finite-difference UPM6 gradient
Eigen::MatrixXd compute_upm6_gradient_fd(
    const Molecule& mol,
    const PM6ParameterSet& params,
    double h = 0.001,
    int max_iter = 100,
    double conv_tol = 1e-7);

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
