#include "vibeqc/periodic_scf.hpp"

#include "vibeqc/diis.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/dynamic_damping.hpp"
#include "vibeqc/ecp.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/periodic_fock.hpp"
#include "vibeqc/periodic_rhf_state_capture.hpp"
#include "vibeqc/periodic_xc.hpp"
#include "vibeqc/scf_mixing.hpp"
#include "vibeqc/scf_convergence.hpp"
#include "vibeqc/xc.hpp"

#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace vibeqc {

namespace {

// One-electron operators keep their physical pair values. Their zero
// extension shares the exchange output/density ordering used by the SCF.
LatticeMatrixSet on_eri_cells(LatticeMatrixSet source,
                            const BasisSet& basis,
                            const PeriodicSystem& system,
                            const LatticeSumOptions& opts) {
    if (!opts.pair_complete_1e) return source;
    const auto cells = pair_complete_eri_cells(system, opts, basis.libint());
    std::map<std::array<int, 3>, std::size_t> positions;
    for (std::size_t c = 0; c < source.cells.size(); ++c) {
        const auto& i = source.cells[c].index;
        positions[{i[0], i[1], i[2]}] = c;
    }
    std::vector<Eigen::MatrixXd> blocks;
    blocks.reserve(cells.size());
    for (const auto& cell : cells) {
        const auto& i = cell.index;
        const auto found = positions.find({i[0], i[1], i[2]});
        if (found == positions.end()) blocks.push_back(Eigen::MatrixXd::Zero(source.nbf, source.nbf));
        else blocks.push_back(std::move(source.blocks[found->second]));
    }
    source.cells = cells;
    source.blocks = std::move(blocks);
    return source;
}

std::string capture_scientific(double value) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(6) << value;
    return stream.str();
}

Eigen::MatrixXd s_inverse_sqrt_real(const Eigen::MatrixXd& S,
                                    double lindep_threshold = 1.0e-8) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    const auto& eigs = solver.eigenvalues();
    if (eigs.minCoeff() < lindep_threshold) {
        throw std::runtime_error(
            "run_rhf_periodic: S(Γ) near-singular (min eig = "
            + std::to_string(eigs.minCoeff()) + ")");
    }
    return solver.eigenvectors()
           * eigs.unaryExpr([](double v) { return 1.0 / std::sqrt(v); })
                 .asDiagonal()
           * solver.eigenvectors().transpose();
}

// Fold a real-space lattice matrix to a complex k-matrix (Bloch sum).
// Duplicates bloch_sum's body here with a lighter API to avoid an extra
// allocation per call; the hot SCF loop calls this per k per iteration.
ComplexMatrix bloch_sum_fast(const LatticeMatrixSet& M,
                             const Eigen::Vector3d& k_cart) {
    const int n = M.nbf;
    ComplexMatrix out = ComplexMatrix::Zero(n, n);
    for (std::size_t c = 0; c < M.cells.size(); ++c) {
        const double phase = k_cart.dot(M.cells[c].r_cart);
        out += std::complex<double>(std::cos(phase), std::sin(phase))
               * M.blocks[c];
    }
    return out;
}

void mix_lattice_fock_set(LatticeMatrixSet& current,
                          const LatticeMatrixSet& previous,
                          double previous_weight) {
    if (previous_weight == 0.0) return;
    if (current.blocks.size() != previous.blocks.size()) {
        throw std::runtime_error(
            "mix_lattice_fock_set: mismatched lattice block counts");
    }
    const double current_weight = 1.0 - previous_weight;
    for (std::size_t c = 0; c < current.blocks.size(); ++c) {
        current.blocks[c] =
            current_weight * current.blocks[c]
            + previous_weight * previous.blocks[c];
    }
}

// Symmetric orthogonalisation at a single k-point.
ComplexMatrix s_inverse_sqrt_complex(const ComplexMatrix& Sk,
                                     double lindep_threshold = 1.0e-8) {
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver(Sk);
    const auto& eigs = solver.eigenvalues();
    if (eigs.minCoeff() < lindep_threshold) {
        throw std::runtime_error(
            "run_rhf_periodic: S(k) near-singular (min eig = "
            + std::to_string(eigs.minCoeff()) + ")");
    }
    const ComplexMatrix& U = solver.eigenvectors();
    Eigen::VectorXd s_inv_sqrt =
        eigs.unaryExpr([](double v) { return 1.0 / std::sqrt(v); });
    return U * s_inv_sqrt.cast<std::complex<double>>().asDiagonal()
             * U.adjoint();
}

struct KDiag {
    Eigen::VectorXd energies;
    ComplexMatrix C;
};

KDiag diag_k(const ComplexMatrix& F_k, const ComplexMatrix& X_k,
             const DavidsonOptions* dav_opts = nullptr) {
    ComplexMatrix F_prime = X_k.adjoint() * F_k * X_k;
    // Symmetrize the tiny numerical drift.
    F_prime = 0.5 * (F_prime + F_prime.adjoint().eval());
    if (dav_opts != nullptr && dav_opts->n_eig > 0) {
        DavidsonResultComplex dres =
            davidson_solve_hermitian(F_prime, *dav_opts);
        if (!dres.converged) {
            throw std::runtime_error(
                "run_rhf_periodic: Davidson diagonalisation failed at k");
        }
        return KDiag{dres.eigenvalues, X_k * dres.eigenvectors};
    }
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver(F_prime);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("run_rhf_periodic: F(k) diagonalisation failed");
    }
    return KDiag{solver.eigenvalues(), X_k * solver.eigenvectors()};
}

double capture_idempotency_limit(double scf_gradient_tolerance,
                                 Eigen::Index n_basis) {
    const double arithmetic_floor =
        64.0 * std::numeric_limits<double>::epsilon()
        * static_cast<double>(n_basis);
    return std::max(
        arithmetic_floor,
        std::min(1.0e-8, scf_gradient_tolerance));
}

double capture_matrix_hermiticity_residual(const ComplexMatrix& matrix) {
    if (!matrix.real().allFinite() || !matrix.imag().allFinite()) {
        return std::numeric_limits<double>::infinity();
    }
    return (matrix - matrix.adjoint().eval()).cwiseAbs().maxCoeff();
}

bool capture_matrix_is_hermitian(const ComplexMatrix& matrix,
                                 double residual) {
    const PeriodicMeanFieldValidationTolerances tolerances;
    const double scale = matrix.cwiseAbs().maxCoeff();
    const double limit = tolerances.matrix_absolute
        + tolerances.matrix_relative * std::max(1.0, scale);
    return std::isfinite(residual) && std::isfinite(limit)
        && residual <= limit;
}

ComplexMatrix capture_spin_summed_density(
    const ComplexMatrix& coefficients,
    int n_occupied) {
    return 2.0 * coefficients.leftCols(n_occupied)
        * coefficients.leftCols(n_occupied).adjoint();
}

}  // namespace

LatticeMatrixSet real_space_density_from_kpoints_fractional(
    const std::vector<ComplexMatrix>& C_per_k,
    const std::vector<Eigen::VectorXd>& occ_per_k,
    const BlochKMesh& kmesh,
    const std::vector<LatticeCell>& cells) {
    if (C_per_k.size() != kmesh.kpoints.size() ||
        occ_per_k.size() != kmesh.kpoints.size() ||
        kmesh.weights.size() != kmesh.kpoints.size()) {
        throw std::runtime_error(
            "real_space_density_from_kpoints_fractional: size mismatch "
            "across k inputs");
    }
    const int nbf = C_per_k.empty() ? 0
                                    : static_cast<int>(C_per_k[0].rows());

    LatticeMatrixSet P_set;
    P_set.nbf = nbf;
    P_set.cells = cells;
    P_set.blocks.assign(cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));

    // P(k) = C · diag(occ) · C†  with each occ_i ∈ [0, 2] for closed-
    // shell RHF (or [0, 1] for spin-orbital). Reduces to the integer-
    // occupation case when occ_i ∈ {0, 2}.
    //
    // Two-phase layout so both phases parallelise without races: the
    // per-k P(k) builds are independent over k, and the per-cell folds
    // are independent over g. The inner k order per cell is unchanged,
    // so each output block accumulates in the exact historical order.
    const std::size_t n_k = kmesh.kpoints.size();
    std::vector<Eigen::MatrixXd> Pk_re(n_k), Pk_im(n_k);

    for (std::size_t ik = 0; ik < n_k; ++ik) {
        if (static_cast<int>(occ_per_k[ik].size()) !=
            static_cast<int>(C_per_k[ik].cols())) {
            throw std::runtime_error(
                "real_space_density_from_kpoints_fractional: occ_per_k "
                "size must equal number of MO columns");
        }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t ik = 0; ik < n_k; ++ik) {
        const ComplexMatrix& C = C_per_k[ik];
        const Eigen::VectorXd& occ = occ_per_k[ik];
        // P(k) = Σ_i occ_i · C_i · C_i†.  Vectorised via diag-multiply.
        const ComplexMatrix C_scaled = C.array().rowwise() *
            occ.transpose().cast<std::complex<double>>().array();
        const ComplexMatrix P_k = C_scaled * C.adjoint();
        Pk_re[ik] = P_k.real();
        Pk_im[ik] = P_k.imag();
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t c = 0; c < cells.size(); ++c) {
        for (std::size_t ik = 0; ik < n_k; ++ik) {
            const double w = kmesh.weights[ik];
            const double phase = kmesh.kpoints[ik].dot(cells[c].r_cart);
            const std::complex<double> z(std::cos(phase), -std::sin(phase));
            P_set.blocks[c] +=
                w * (z.real() * Pk_re[ik] - z.imag() * Pk_im[ik]);
        }
    }
    return P_set;
}

LatticeMatrixSet real_space_density_from_kpoints(
    const std::vector<ComplexMatrix>& C_per_k,
    const std::vector<int>& n_occ_per_k,
    const BlochKMesh& kmesh,
    const std::vector<LatticeCell>& cells) {
    if (C_per_k.size() != kmesh.kpoints.size() ||
        n_occ_per_k.size() != kmesh.kpoints.size() ||
        kmesh.weights.size() != kmesh.kpoints.size()) {
        throw std::runtime_error(
            "real_space_density_from_kpoints: size mismatch across k inputs");
    }
    const int nbf = C_per_k.empty() ? 0
                                    : static_cast<int>(C_per_k[0].rows());

    LatticeMatrixSet P_set;
    P_set.nbf = nbf;
    P_set.cells = cells;
    P_set.blocks.assign(cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));

    // For each k, build P(k) = 2 C_occ C_occ† and accumulate
    //   P(g) += w_k · Re[ exp(−i k · g) · P(k) ]
    // The Re[] implements time-reversal averaging: for a time-reversal-
    // symmetric Hamiltonian P(-k) = P(k)*, so the (k, -k) pair contribution
    // is the real part of the exp(-i k.g) * P(k) term.
    //
    // Γ-only caveat (kmesh = (1,1,1)): with a single k-point at k = 0
    // and unit weight, this formula reduces to P(g) = P(Γ) at EVERY
    // cell g — the same matrix block at every lattice position. That's
    // mathematically correct as a Bloch sum (there's only one k, so no
    // k-information to recover g-dependence from), but it is NOT the
    // density-LOCALITY convention required by CRYSTAL-style direct-space
    // BIPOLE Fock builds, which assume P(g ≠ 0) = 0 for an iter-1
    // SAD-like density. For BIPOLE-style locality at Γ-only, callers
    // (e.g. ``run_pbc_bipole_rhf`` when ``use_ewald_j_split=True`` and
    // ``n_k == 1``) short-circuit this function and manually set
    // P_real(g=0) = 2·C_occ·C_occ†, P_real(g≠0) = 0. For multi-k
    // (N_k > 1) the k-averaging Σ_k w_k · exp(−i k · g) → δ_{g, 0}
    // naturally recovers the locality convention without intervention.
    // Two-phase layout (see the fractional variant above): per-k P(k)
    // builds parallelise over k, per-cell folds parallelise over g, and
    // the inner k order per cell keeps the historical accumulation order.
    // A k with nocc == 0 is skipped in both phases, exactly as before.
    const std::size_t n_k = kmesh.kpoints.size();
    std::vector<Eigen::MatrixXd> Pk_re(n_k), Pk_im(n_k);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t ik = 0; ik < n_k; ++ik) {
        if (n_occ_per_k[ik] == 0) continue;
        const ComplexMatrix Cocc = C_per_k[ik].leftCols(n_occ_per_k[ik]);
        // P(k) — Hermitian nbf × nbf.
        const ComplexMatrix P_k = 2.0 * Cocc * Cocc.adjoint();
        Pk_re[ik] = P_k.real();
        Pk_im[ik] = P_k.imag();
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t c = 0; c < cells.size(); ++c) {
        for (std::size_t ik = 0; ik < n_k; ++ik) {
            if (n_occ_per_k[ik] == 0) continue;
            const double w = kmesh.weights[ik];
            const double phase = kmesh.kpoints[ik].dot(cells[c].r_cart);
            const std::complex<double> z(std::cos(phase), -std::sin(phase));
            // Real part of (z · P_k): P_real = cos·Re(P_k) + sin·Im(P_k).
            // With z = cos − i sin, z.imag() = −sin, so subtracting
            // z.imag()·Im P_k flips sign correctly.
            P_set.blocks[c] +=
                w * (z.real() * Pk_re[ik] - z.imag() * Pk_im[ik]);
        }
    }
    return P_set;
}

namespace {

LatticeMatrixSet fock_initial_density_multik(
    const LatticeMatrixSet& F_guess, const std::vector<LatticeCell>& cells,
    const BlochKMesh& kmesh, const std::vector<ComplexMatrix>& X_k,
    int nocc) {
    const std::size_t nk = kmesh.kpoints.size();
    std::vector<ComplexMatrix> C_k(nk);
    std::vector<int> n_occ_per_k(nk, nocc);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        ComplexMatrix Fk = bloch_sum_fast(F_guess, kmesh.kpoints[ik]);
        Fk = 0.5 * (Fk + Fk.adjoint().eval());
        C_k[ik] = diag_k(Fk, X_k[ik]).C;
    }
    return real_space_density_from_kpoints(C_k, n_occ_per_k, kmesh, cells);
}

LatticeMatrixSet local_density_from_home_block(
    const Eigen::MatrixXd& D_home,
    const std::vector<LatticeCell>& cells) {
    LatticeMatrixSet P_set;
    P_set.nbf = static_cast<int>(D_home.rows());
    P_set.cells = cells;
    P_set.blocks.assign(cells.size(), Eigen::MatrixXd::Zero(
        D_home.rows(), D_home.cols()));
    if (!P_set.blocks.empty()) {
        // direct_lattice_cells orders the home cell first.
        P_set.blocks[0] = D_home;
    }
    return P_set;
}

// Periodic SAP initial density (multi-k): F_SAP(k) = T(k) + V_SAP(k),
// diagonalised at each k and folded to a real-space density. The periodic
// analogue of the molecular SAP guess (V_SAP, the lattice-summed SAP potential
// from compute_vsap_lattice, replaces V_ne). A molecular Becke grid on the unit
// cell is adequate for V_SAP's smooth long-range part; the analytical erfc
// short-range carries the sharp on-site structure.
LatticeMatrixSet sap_initial_density_multik(
    const BasisSet& basis, const PeriodicSystem& system,
    const LatticeMatrixSet& T_set, const std::vector<LatticeCell>& cells,
    const BlochKMesh& kmesh, const std::vector<ComplexMatrix>& X_k,
    int nocc, const LatticeSumOptions& lattice_opts, const GuessECPContext& ecp) {
    const Grid grid = build_grid(system.unit_cell_molecule());
    const auto Vsap = on_eri_cells(ecp.active()
        ? compute_vsap_ecp_lattice(basis, system, lattice_opts, ecp)
        : compute_vsap_lattice(basis, system, grid, "sap_helfem_large", lattice_opts),
        basis, system, lattice_opts);
    LatticeMatrixSet Fsap = T_set;
    if (Vsap.blocks.size() > Fsap.blocks.size())
        throw std::invalid_argument("SAP: potential cell domain exceeds the kinetic domain");
    for (std::size_t c = 0; c < Vsap.blocks.size(); ++c) {
        Fsap.blocks[c] += Vsap.blocks[c];
    }
    return fock_initial_density_multik(Fsap, cells, kmesh, X_k, nocc);
}

// Periodic PATOM initial density (multi-k): start from the local SAD density
// D(g=0), build one HF-like in-field Fock in real space,
// F_PATOM(g) = Hcore(g) + J[D](g) - 1/2 K[D](g), then diagonalise it at each
// k point and fold the resulting density back to real space.
LatticeMatrixSet patom_initial_density_multik(
    const BasisSet& basis, const PeriodicSystem& system,
    const LatticeMatrixSet& Hcore_set, const std::vector<LatticeCell>& cells,
    const BlochKMesh& kmesh, const std::vector<ComplexMatrix>& X_k,
    int nocc, const LatticeSumOptions& lattice_opts,
    const Eigen::MatrixXd& S_effective, const GuessECPContext& ecp) {
    const Eigen::MatrixXd D_sad = normalize_guess_density(
        sad_density(system.unit_cell_molecule(), basis, ecp), S_effective, 2.0 * nocc);
    LatticeMatrixSet P_sad =
        local_density_from_home_block(D_sad, cells);
    LatticeMatrixSet F_patom = build_fock_2e_real_space(
        basis, system, lattice_opts, P_sad, /*exchange_scale=*/1.0);
    // The zero-extended Hcore shares the in-field exchange ordering.
    if (F_patom.blocks.size() > Hcore_set.blocks.size()) {
        throw std::runtime_error(
            "periodic PATOM: in-field Fock cell list is longer than Hcore's");
    }
    LatticeMatrixSet F_wide = Hcore_set;
    for (std::size_t c = 0; c < F_patom.blocks.size(); ++c) {
        F_wide.blocks[c] += F_patom.blocks[c];
    }
    return fock_initial_density_multik(F_wide, cells, kmesh, X_k, nocc);
}

}  // namespace

namespace {

void validate_guess_kmesh(const BlochKMesh& kmesh) {
    if (kmesh.kpoints.empty() || kmesh.weights.size() != kmesh.kpoints.size()) {
        throw std::invalid_argument("periodic initial guess: k points and weights must match");
    }
    double weight_sum = 0.0;
    for (std::size_t k = 0; k < kmesh.kpoints.size(); ++k) {
        const double w = kmesh.weights[k];
        if (!kmesh.kpoints[k].allFinite() || !std::isfinite(w) || w < 0.0) {
            throw std::invalid_argument("periodic initial guess: invalid k point or weight");
        }
        weight_sum += w;
    }
    if (std::abs(weight_sum - 1.0) > 1e-12) {
        throw std::invalid_argument("periodic initial guess: k weights must sum to one");
    }
}

// Validate the state before the ordinary finite-lattice Fourier fold. In
// particular, taking Re[D(g)] must not silently remove broken time reversal.
LatticeMatrixSet read_initial_density_multik(
    const std::vector<ComplexMatrix>& densities,
    const std::vector<ComplexMatrix>& overlaps, const BlochKMesh& mesh,
    const PeriodicSystem& system, const std::vector<LatticeCell>& cells,
    int electrons) {
    if (densities.size() != overlaps.size() || densities.empty())
        throw std::invalid_argument("READ: supply one density per target k point");
    const auto nbf = overlaps.front().rows();
    double population = 0;
    for (std::size_t k = 0; k < densities.size(); ++k) {
        const auto& d = densities[k];
        if (d.rows() != nbf || d.cols() != nbf || !d.allFinite()
            || (d - d.adjoint()).norm() > 1e-8)
            throw std::invalid_argument("READ: density must be finite, Hermitian and match the AO basis");
        Eigen::SelfAdjointEigenSolver<ComplexMatrix> eig(d, Eigen::EigenvaluesOnly);
        if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() < -1e-8)
            throw std::invalid_argument("READ: density must be positive semidefinite");
        population += mesh.weights[k] * std::real((d * overlaps[k]).trace());
        int partner = -1;
        for (std::size_t j = 0; j < densities.size(); ++j) {
            const Eigen::Vector3d frac = system.lattice.transpose() * (mesh.kpoints[k] + mesh.kpoints[j])
                / (2.0 * std::acos(-1.0));
            if ((frac.array() - frac.array().round()).abs().maxCoeff() < 1e-8) {
                if (partner >= 0) throw std::invalid_argument("READ: duplicate target k points");
                partner = static_cast<int>(j);
            }
        }
        if (partner < 0 || std::abs(mesh.weights[k] - mesh.weights[partner]) > 1e-12
            || densities[partner].rows() != nbf || densities[partner].cols() != nbf
            || (d - densities[partner].conjugate()).norm() > 1e-8)
            throw std::invalid_argument("READ: real lattice requires a complete time-reversal-compatible density");
    }
    if (!std::isfinite(population) || (electrons > 0 && population <= 0))
        throw std::invalid_argument("READ: positive electrons require positive weighted population");
    const double scale = electrons == 0 ? 0 : electrons / population;
    LatticeMatrixSet result;
    result.nbf = nbf;
    result.cells = cells;
    for (const auto& cell : cells) {
        ComplexMatrix block = ComplexMatrix::Zero(nbf, nbf);
        for (std::size_t k = 0; k < densities.size(); ++k)
            block += mesh.weights[k] * std::exp(std::complex<double>(0, -mesh.kpoints[k].dot(cell.r_cart)))
                * (0.5 * (densities[k] + densities[k].adjoint()).eval()) * scale;
        if (!block.allFinite() || block.imag().norm() > 1e-8)
            throw std::invalid_argument("READ: density cannot be represented by real lattice blocks");
        result.blocks.push_back(block.real());
    }
    return result;
}

PeriodicRHFResult run_rhf_periodic_impl(
    const PeriodicSystem& system,
    const BasisSet& basis,
    const BlochKMesh& kmesh,
    const PeriodicSCFOptions& opts,
    PeriodicRHFStateCaptureRequest* capture_request) {
    validate_guess_kmesh(kmesh);
    std::vector<InitialGuess> guess_capabilities{
        InitialGuess::HCORE, InitialGuess::SAD, InitialGuess::PATOM,
        InitialGuess::HUECKEL, InitialGuess::MINAO, InitialGuess::READ};
    if (system.dim == 3) guess_capabilities.push_back(InitialGuess::SAP);
    const InitialGuess effective_guess = GuessEngine::resolve_auto_for_molecule(
        system.unit_cell_molecule(), opts.initial_guess, true, false, guess_capabilities);
    const auto guess_mol = system.unit_cell_molecule();
    const auto guess_ecp = periodic_guess_ecp_context(opts, &guess_mol);
    validate_guess_ecp(effective_guess, guess_ecp, &guess_mol);
    const int n_elec = system.n_electrons();
    const bool _has_ecp = !opts.ecp_primitive_blocks.empty();
    const int n_elec_valence = _has_ecp
        ? n_elec - opts.ecp_total_ncore
        : n_elec;
    if (n_elec_valence % 2 != 0) {
        throw std::invalid_argument(
            "run_rhf_periodic: closed-shell RHF requires even electron "
            "count per unit cell, got " + std::to_string(n_elec_valence) +
            (_has_ecp ? " (valence, with ECP ncore=" +
             std::to_string(opts.ecp_total_ncore) + ")" : ""));
    }
    if (system.multiplicity != 1) {
        throw std::invalid_argument(
            "run_rhf_periodic: closed-shell RHF requires multiplicity = 1");
    }
    if (kmesh.kpoints.empty()) {
        throw std::invalid_argument("run_rhf_periodic: empty k-mesh");
    }
    validate_fraction_01("run_rhf_periodic: damping", opts.damping);
    validate_fraction_01("run_rhf_periodic: fock_mixing",
                         opts.fock_mixing);
    const int nocc = n_elec_valence / 2;

    // The correlation-capture entry point has a stricter, allocation-free
    // preflight than ordinary SCF. It cannot be mistaken for full resource
    // admission: the byte cap covers only the exact retained state payload.
    int capture_periodic_dimension = 0;
    Eigen::Matrix3d capture_reciprocal_lattice = Eigen::Matrix3d::Identity();
    if (capture_request != nullptr) {
        constexpr double kMaximumCaptureEnergyTolerance = 1.0e-8;
        constexpr double kMaximumCaptureGradientTolerance = 1.0e-6;
        if (!std::isfinite(opts.smearing_temperature)
            || opts.smearing_temperature != 0.0) {
            throw std::invalid_argument(
                "run_rhf_periodic_with_state_capture: smearing is not "
                "compatible with the hard 2/0 RHF state contract");
        }
        if (!std::isfinite(opts.conv_tol_energy)
            || opts.conv_tol_energy <= 0.0
            || opts.conv_tol_energy > kMaximumCaptureEnergyTolerance
            || !std::isfinite(opts.conv_tol_grad)
            || opts.conv_tol_grad <= 0.0
            || opts.conv_tol_grad > kMaximumCaptureGradientTolerance) {
            throw std::invalid_argument(
                "run_rhf_periodic_with_state_capture: convergence "
                "tolerances must be positive and no looser than 1e-8 Ha "
                "and 1e-6, respectively");
        }
        if (n_elec_valence <= 0) {
            throw std::invalid_argument(
                "run_rhf_periodic_with_state_capture: effective electron "
                "count must be positive");
        }
        const bool density_damping_can_activate =
            opts.damping > 0.0
            || (opts.dynamic_damping && opts.dynamic_damping_max > 0.0);
        const bool diis_disables_damping_before_iteration_cap =
            opts.use_diis && opts.diis_start_iter <= opts.max_iter;
        if (density_damping_can_activate
            && !diis_disables_damping_before_iteration_cap) {
            throw std::invalid_argument(
                "run_rhf_periodic_with_state_capture: density damping "
                "requires DIIS to become active before max_iter, or all "
                "density damping must be disabled, so final F[D] uses the "
                "captured determinant exactly");
        }
        capture_periodic_dimension = system.dim;
        capture_reciprocal_lattice = system.reciprocal_lattice();
        (void)preflight_periodic_rhf_state_capture(
            *capture_request,
            capture_periodic_dimension,
            kmesh.mesh,
            kmesh.is_shift,
            capture_reciprocal_lattice,
            kmesh.kpoints,
            kmesh.weights,
            !kmesh.ir_mapping.empty(),
            static_cast<std::uint64_t>(basis.nbasis()),
            static_cast<std::uint64_t>(n_elec_valence));
    }

    // ---- One-electron lattice-summed integrals ---------------------------
    const auto S_set = on_eri_cells(compute_overlap_lattice(basis, system, opts.lattice_opts),
                                    basis, system, opts.lattice_opts);
    const bool use_davidson_here =
        opts.use_davidson && S_set.nbf >= opts.davidson_min_dim;
    DavidsonOptions dav_opts = opts.davidson;
    if (use_davidson_here) {
        if (dav_opts.n_eig == 0) dav_opts.n_eig = S_set.nbf;
    }
    const DavidsonOptions* dav_ptr =
        use_davidson_here ? &dav_opts : nullptr;
    const auto T_set = on_eri_cells(compute_kinetic_lattice(basis, system, opts.lattice_opts),
                                    basis, system, opts.lattice_opts);
    LatticeMatrixSet V_set;
    LatticeMatrixSet V_ecp_set;
    if (_has_ecp) {
        V_set = on_eri_cells(compute_nuclear_lattice_with_charges(
            basis, system, opts.lattice_opts, opts.ecp_effective_charges), basis, system, opts.lattice_opts);
        V_ecp_set = on_eri_cells(compute_ecp_lattice_from_primitives(
            basis, system, opts.lattice_opts,
            opts.ecp_home_centers, opts.ecp_primitive_blocks), basis, system, opts.lattice_opts);
    } else {
        V_set = on_eri_cells(compute_nuclear_lattice(basis, system, opts.lattice_opts), basis, system, opts.lattice_opts);
    }

    // Real-space Hcore(g) = T(g) + V(g) (+ V_ecp(g) when ECP).
    LatticeMatrixSet Hcore_set;
    Hcore_set.nbf = S_set.nbf;
    Hcore_set.cells = S_set.cells;
    Hcore_set.blocks.resize(S_set.cells.size());
    for (std::size_t c = 0; c < S_set.cells.size(); ++c) {
        Hcore_set.blocks[c] = T_set.blocks[c] + V_set.blocks[c];
        if (_has_ecp && c < V_ecp_set.blocks.size()) {
            Hcore_set.blocks[c] += V_ecp_set.blocks[c];
        }
    }

    // Pre-compute S(k) and X(k) at each k-point — these are iteration-
    // invariant. Same for Hcore(k).
    const std::size_t nk = kmesh.kpoints.size();
    std::vector<ComplexMatrix> S_k(nk), X_k(nk), H_k(nk);
    Eigen::MatrixXd S_effective = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        S_k[ik] = bloch_sum_fast(S_set, kmesh.kpoints[ik]);
        // Hermitian-ise against tiny imaginary drift in the diagonal.
        S_k[ik] = 0.5 * (S_k[ik] + S_k[ik].adjoint().eval());
        S_effective += kmesh.weights[ik] * S_k[ik].real();
        X_k[ik] = s_inverse_sqrt_complex(S_k[ik]);
        H_k[ik] = bloch_sum_fast(Hcore_set, kmesh.kpoints[ik]);
        H_k[ik] = 0.5 * (H_k[ik] + H_k[ik].adjoint().eval());
    }

    // ---- Initial guess: Hcore at each k -> diagonalize -> density --------
    std::vector<ComplexMatrix> C_k(nk);
    std::vector<int> n_occ_per_k(nk, nocc);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        const auto dk = diag_k(H_k[ik], X_k[ik], dav_ptr);
        C_k[ik] = dk.C;
    }
    LatticeMatrixSet P_set =
        real_space_density_from_kpoints(C_k, n_occ_per_k, kmesh, S_set.cells);

    // Optionally replace with a density-mode initial guess via the
    // engine. For periodic, density lives at g=0 (other cells zero);
    // Bloch-summing the g=0 block recovers a sensible D(k) for the
    // first Fock build. Fock-mode periodic guesses (SAP, HUECKEL) are
    // handled by the branches below because the generic molecular engine
    // has no lattice-Fock output channel.
    if (effective_guess == InitialGuess::READ) {
        P_set = read_initial_density_multik(opts.read_density_k, S_k, kmesh,
                                          system, S_set.cells, 2 * nocc);
    } else if (effective_guess == InitialGuess::SAP) {
        // Periodic SAP Fock-mode guess, wired through the multi-k path:
        // F_SAP(k) = T(k) + V_SAP(k), diagonalised at each k.
        P_set = sap_initial_density_multik(
            basis, system, T_set, S_set.cells, kmesh, X_k, nocc,
            opts.lattice_opts, guess_ecp);
    } else if (effective_guess == InitialGuess::HUECKEL) {
        // Periodic HUECKEL Fock-mode guess:
        // F_GWH(g) from S(g) and atomic AO energies, diagonalised at each k.
        const auto F_huckel = compute_huckel_fock_lattice(
            system.unit_cell_molecule(), basis, S_set, guess_ecp);
        P_set = fock_initial_density_multik(
            F_huckel, S_set.cells, kmesh, X_k, nocc);
    } else if (effective_guess == InitialGuess::MINAO) {
        // Periodic MINAO density-mode guess: project the ANO-RCC reference
        // density onto the working basis with the lattice-summed Γ overlaps,
        // place it on the g=0 cell (Bloch-summing recovers a k-independent
        // D(k) for the first Fock build), exactly like SAD.
        Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
        for (const auto& b : S_set.blocks) S_gamma += b;
        const Eigen::MatrixXd D_minao = compute_minao_density_periodic(
            system.unit_cell_molecule(), basis, system, S_gamma,
            2 * nocc, opts.lattice_opts, guess_ecp);
        P_set = local_density_from_home_block(
            normalize_guess_density(D_minao, S_effective, 2.0 * nocc),
            S_set.cells);
    } else if (effective_guess == InitialGuess::PATOM) {
        P_set = patom_initial_density_multik(
            basis, system, Hcore_set, S_set.cells, kmesh, X_k, nocc,
            opts.lattice_opts, S_effective, guess_ecp);
    } else {
        const auto g = GuessEngine::build_closed_shell(
            system.unit_cell_molecule(), basis, nocc,
            effective_guess,
            /*S=*/&S_effective, /*Hcore=*/nullptr, /*jk=*/nullptr,
            SystemHints{/*is_periodic=*/true}, 1e-7, guess_ecp);
        if (g.D.size() != 0) {
            for (auto& b : P_set.blocks) b.setZero();
            // g=0 is always the first cell by direct_lattice_cells' ordering.
            P_set.blocks[0] = g.D;
        }
    }

    // H(k) is only an initial-guess input. Release its all-k storage before
    // entering SCF so an eventual opt-in S/F/C capture never stacks three
    // retained matrices on top of an otherwise dead fourth matrix.
    std::vector<ComplexMatrix>().swap(H_k);

    // ---- Nuclear repulsion -----------------------------------------------
    double e_nuc;
    if (_has_ecp && !opts.ecp_effective_charges.empty()) {
        const auto rep_cells = direct_lattice_cells(
            system, opts.lattice_opts.nuclear_cutoff_bohr);
        double enuc = 0.0;
        const auto& atoms = system.unit_cell;
        for (const auto& rc : rep_cells) {
            for (std::size_t i = 0; i < atoms.size(); ++i) {
                double qi = i < opts.ecp_effective_charges.size()
                    ? opts.ecp_effective_charges[i]
                    : static_cast<double>(atoms[i].Z);
                std::size_t j0 = (rc.r_cart.squaredNorm() < 1e-24) ? i + 1 : 0;
                for (std::size_t j = j0; j < atoms.size(); ++j) {
                    double qj = j < opts.ecp_effective_charges.size()
                        ? opts.ecp_effective_charges[j]
                        : static_cast<double>(atoms[j].Z);
                    double dx = atoms[i].xyz[0] + rc.r_cart[0] - atoms[j].xyz[0];
                    double dy = atoms[i].xyz[1] + rc.r_cart[1] - atoms[j].xyz[1];
                    double dz = atoms[i].xyz[2] + rc.r_cart[2] - atoms[j].xyz[2];
                    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (r > 1e-12) enuc += qi * qj / r;
                }
            }
        }
        e_nuc = 0.5 * enuc;
    } else {
        e_nuc = nuclear_repulsion_per_cell(system, opts.lattice_opts);
    }

    // ---- DIIS / EDIIS / EDIIS+DIIS operate on Γ-folded real matrices ----
    // KDIIS multi-k path needs a per-k MO basis to project the orbital-
    // rotation gradient; deferred. Selecting KDIIS here raises a clear
    // error so users aren't silently downgraded.
    DIIS diis = make_diis(opts);
    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);

    PeriodicRHFResult result;
    result.mo_energies_k.resize(nk);
    result.restart_basis = basis;
    result.restart_lattice = system.lattice;
    result.guess_selection = {opts.initial_guess, effective_guess, effective_guess};
    result.restart_kpoints.resize(kmesh.kpoints.size(), 3);
    result.restart_weights.resize(kmesh.weights.size());
    for (std::size_t ik = 0; ik < kmesh.kpoints.size(); ++ik) {
        result.restart_kpoints.row(ik) = kmesh.kpoints[ik].transpose();
        result.restart_weights(ik) = kmesh.weights[ik];
    }

    result.e_nuclear = e_nuc;
    // Fold S to Γ for the output.
    Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
    for (const auto& b : S_set.blocks) S_gamma += b;
    result.overlap = S_gamma;

    LatticeMatrixSet P_prev = P_set;
    LatticeMatrixSet F_prev_mixed;
    bool have_prev_fock = false;
    double E_prev = 0.0;

    // Dynamic-damping state. Inactive by default.
    double current_damping = opts.damping;
    bool have_prev_E = false;

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter;

        // Damping on the real-space density (block-wise). current_damping
        // mirrors opts.damping when dynamic_damping is off; otherwise
        // updated at the bottom of each iteration from the energy
        // decrease (Zerner-Hehenberger 1979).
        LatticeMatrixSet P_used = P_set;
        if (iter > 1 && current_damping > 0.0 && !diis_active) {
            for (std::size_t c = 0; c < P_set.blocks.size(); ++c) {
                P_used.blocks[c] =
                    current_damping * P_prev.blocks[c]
                    + (1.0 - current_damping) * P_set.blocks[c];
            }
        }

        // F^{2e}(g) = J(g) − ½ K(g) via the general real-space build.
        const auto F2e_set =
            build_fock_2e_real_space(basis, system, opts.lattice_opts, P_used);

        // Both operators share the declared output ordering. Hcore is
        // zero beyond its physical one-electron pair support.
        if (F2e_set.blocks.size() > Hcore_set.blocks.size()) {
            throw std::runtime_error(
                "run_rhf_periodic: 2e Fock cell list is longer than Hcore's");
        }
        LatticeMatrixSet F_set = Hcore_set;  // copy, then add F^{2e}
        for (std::size_t c = 0; c < F2e_set.blocks.size(); ++c) {
            F_set.blocks[c] += F2e_set.blocks[c];
        }

        // Electronic energy — lattice-sum version of
        //   E_elec = (1/2) Σ_g tr( P(g) · [Hcore(g) + F(g)]ᵀ ).
        // For real P and real F at any given g, trace(AB) = sum of
        // element-wise products (where we use the transpose to match
        // the Hermitian convention P_μν(g) = P_νμ*(-g)).
        double E_elec = 0.0;
        for (std::size_t c = 0; c < F_set.blocks.size(); ++c) {
            E_elec += 0.5 * (P_used.blocks[c].array()
                             * (Hcore_set.blocks[c] + F_set.blocks[c]).array())
                             .sum();
        }
        const double E_total = E_elec + e_nuc;

        // Γ-folded matrices for DIIS + convergence test.
        Eigen::MatrixXd F_gamma = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
        Eigen::MatrixXd P_used_gamma = F_gamma;
        for (std::size_t c = 0; c < F_set.blocks.size(); ++c) {
            F_gamma += F_set.blocks[c];
            P_used_gamma += P_used.blocks[c];
        }
        const Eigen::MatrixXd error =
            F_gamma * P_used_gamma * S_gamma - S_gamma * P_used_gamma * F_gamma;
        const double grad_norm = error.norm();
        const double dE = E_total - E_prev;

        VIBEQC_DIAG("periodic-rhf", vibeqc::DiagLevel::VERBOSE,
            "iter %3d  E=% .10f  dE=%+.3e  |grad|=%.3e",
            iter, E_total, (iter == 1 ? 0.0 : dE), grad_norm);

        bool converged = is_scf_converged(
            iter, dE, grad_norm,
            opts.conv_tol_energy, opts.conv_tol_grad);

        double capture_commutator_max = 0.0;
        double capture_commutator_rms = 0.0;
        double capture_projector_closure_max = 0.0;
        double capture_idempotency_relative_max = 0.0;
        double capture_fock_hermiticity_max = 0.0;
        if (converged && capture_request != nullptr) {
            double weighted_commutator_square = 0.0;
            double weighted_roundoff_square = 0.0;
            double global_homo = -std::numeric_limits<double>::infinity();
            double global_lumo = std::numeric_limits<double>::infinity();
            const double idempotency_limit =
                capture_idempotency_limit(opts.conv_tol_grad, S_set.nbf);
            // P_set is the real-space fold of C_k.  A damped P_used is not a
            // determinant, so it cannot define the correlation snapshot.
            // The option preflight guarantees DIIS will eventually provide
            // an undamped cycle; defer until that exact F[P_set] cycle rather
            // than accepting approximate mixed-density provenance.
            const bool capture_density_was_damped =
                iter > 1 && current_damping > 0.0 && !diis_active;
            bool physical_density_ready = !capture_density_was_damped;
            for (std::size_t ik = 0;
                 physical_density_ready && ik < nk;
                 ++ik) {
                const ComplexMatrix raw_fock_k =
                    bloch_sum_fast(F_set, kmesh.kpoints[ik]);
                const double fock_hermiticity =
                    capture_matrix_hermiticity_residual(raw_fock_k);
                capture_fock_hermiticity_max = std::max(
                    capture_fock_hermiticity_max, fock_hermiticity);
                // The direct finite cell-domain contraction need not be
                // exactly self-adjoint before its established Bloch-space
                // projection: transposing the g block shifts the internal
                // lambda/sigma domain from C to C+g.  Do not silently accept
                // a material boundary defect, however.  The raw residual
                // must first fit the fixed state-contract matrix tolerance.
                // Fixed-point diagnostics then use the Hermitian operator
                // that the ordinary SCF path actually diagonalizes.
                if (!capture_matrix_is_hermitian(
                        raw_fock_k, fock_hermiticity)) {
                    throw std::runtime_error(
                        "run_rhf_periodic_with_state_capture: raw Fock at "
                        "full-mesh k point " + std::to_string(ik)
                        + " has finite-domain Hermiticity residual "
                        + capture_scientific(fock_hermiticity)
                        + " above the fixed capture tolerance; increase the "
                        "lattice domains or use a translation-covariant "
                        "periodic Hamiltonian");
                }
                const ComplexMatrix physical_fock_k = 0.5 * (
                    raw_fock_k + raw_fock_k.adjoint().eval());
                const ComplexMatrix physical_density_k =
                    capture_spin_summed_density(C_k[ik], nocc);
                const double density_hermiticity =
                    capture_matrix_hermiticity_residual(physical_density_k);
                if (!capture_matrix_is_hermitian(
                        physical_density_k, density_hermiticity)) {
                    physical_density_ready = false;
                    break;
                }
                const auto diagnostics =
                    periodic_rhf_physical_density_diagnostics(
                        S_k[ik], physical_fock_k, physical_density_k);
                capture_commutator_max = std::max(
                    capture_commutator_max,
                    diagnostics.commutator_frobenius);
                capture_idempotency_relative_max = std::max(
                    capture_idempotency_relative_max,
                    diagnostics.metric_idempotency_relative);
                weighted_commutator_square +=
                    kmesh.weights[ik]
                    * diagnostics.commutator_frobenius
                    * diagnostics.commutator_frobenius;
                weighted_roundoff_square +=
                    kmesh.weights[ik]
                    * diagnostics.commutator_roundoff_allowance
                    * diagnostics.commutator_roundoff_allowance;
                if (diagnostics.commutator_frobenius
                        > opts.conv_tol_grad
                            + diagnostics.commutator_roundoff_allowance
                    || diagnostics.metric_idempotency_relative
                        > idempotency_limit) {
                    physical_density_ready = false;
                }
                // A small commutator alone does not prove that the current
                // occupied projector is the Aufbau projector of F.  In
                // particular, its implied projector error is gap dependent.
                // Apply the same explicit closure gate used by final capture
                // here, so a merely underconverged candidate continues SCF
                // rather than throwing after convergence was accepted.
                const auto diagonal =
                    diag_k(physical_fock_k, X_k[ik], nullptr);
                global_homo = std::max(
                    global_homo, diagonal.energies[nocc - 1]);
                global_lumo = std::min(
                    global_lumo, diagonal.energies[nocc]);
                const auto fixed_point =
                    periodic_rhf_density_fixed_point_diagnostics(
                        physical_density_k,
                        diagonal.C,
                        static_cast<std::uint64_t>(nocc));
                capture_projector_closure_max = std::max(
                    capture_projector_closure_max,
                    fixed_point.relative);
                if (fixed_point.relative > idempotency_limit) {
                    physical_density_ready = false;
                }
            }
            capture_commutator_rms =
                std::sqrt(weighted_commutator_square);
            const double weighted_roundoff =
                std::sqrt(weighted_roundoff_square);
            if (capture_commutator_rms
                > opts.conv_tol_grad + weighted_roundoff) {
                physical_density_ready = false;
            }
            if (physical_density_ready) {
                const double global_gap = global_lumo - global_homo;
                if (!std::isfinite(global_gap)
                    || !(global_gap
                         > capture_request->minimum_band_gap_hartree)) {
                    throw std::runtime_error(
                        "run_rhf_periodic_with_state_capture: converged "
                        "RHF global band gap " + std::to_string(global_gap)
                        + " Ha does not exceed the requested minimum of "
                        + std::to_string(
                            capture_request->minimum_band_gap_hartree)
                        + " Ha");
                }
            }
            if (!physical_density_ready) {
                converged = false;
                VIBEQC_DIAG(
                    "periodic-rhf", vibeqc::DiagLevel::DEBUG,
                    "correlation-state capture deferred at iter %d: "
                    "density-damped=%d, full-k commutator max=%.3e "
                    "rms=%.3e, idempotency max=%.3e, projector closure "
                    "max=%.3e, raw pre-projection Fock Hermiticity "
                    "max=%.3e",
                    iter,
                    capture_density_was_damped ? 1 : 0,
                    capture_commutator_max,
                    capture_commutator_rms,
                    capture_idempotency_relative_max,
                    capture_projector_closure_max,
                    capture_fock_hermiticity_max);
            }
        }
        if (converged) {
            VIBEQC_DIAG("periodic-rhf", vibeqc::DiagLevel::DEBUG,
                "converged at iter %d: |dE|=%.3e < %.0e, |grad|=%.3e < %.0e",
                iter, std::abs(dE), opts.conv_tol_energy,
                grad_norm, opts.conv_tol_grad);
        }

        if (converged && capture_request != nullptr) {
            // The readiness pass above has compared each exact C_k occupied
            // projector with a fresh diagonalization of the same gated-and-
            // projected physical Fock.  The old coefficients are therefore
            // dead as a full mesh; free them in bulk before state assembly
            // starts retaining S/F/C.
            std::vector<ComplexMatrix>().swap(C_k);

            PeriodicRestrictedMeanFieldInput input;
            input.calculation_identity =
                std::move(capture_request->calculation_identity);
            input.reference_kind =
                PeriodicMeanFieldReferenceKind::RestrictedHartreeFock;
            input.normalization =
                PeriodicMeanFieldNormalizationConvention::
                    UnnormalizedAoBlochSumsUniformFullBzWeights;
            input.periodic_dimension = capture_periodic_dimension;
            input.mesh = kmesh.mesh;
            input.is_shift = kmesh.is_shift;
            input.reciprocal_lattice = capture_reciprocal_lattice;
            input.converged = true;
            input.symmetry_reduced_input = false;
            input.symmetry_reconstructed_input = false;
            input.n_basis = static_cast<std::uint64_t>(S_set.nbf);
            input.n_effective_orbitals =
                static_cast<std::uint64_t>(S_set.nbf);
            input.electrons_per_cell =
                static_cast<std::uint64_t>(n_elec_valence);
            input.reference_energy_per_cell = E_total;
            input.minimum_band_gap_hartree =
                capture_request->minimum_band_gap_hartree;

            const double uniform_weight = 1.0 / static_cast<double>(nk);
            for (std::size_t ik = 0; ik < nk; ++ik) {
                const ComplexMatrix raw_fock_k =
                    bloch_sum_fast(F_set, kmesh.kpoints[ik]);
                const double fock_hermiticity =
                    capture_matrix_hermiticity_residual(raw_fock_k);
                if (!capture_matrix_is_hermitian(
                        raw_fock_k, fock_hermiticity)) {
                    throw std::runtime_error(
                        "run_rhf_periodic_with_state_capture: raw Fock "
                        "exceeded the finite-domain Hermiticity gate during "
                        "final capture");
                }
                ComplexMatrix physical_fock_k = 0.5 * (
                    raw_fock_k + raw_fock_k.adjoint().eval());
                auto diagonal = diag_k(physical_fock_k, X_k[ik], nullptr);
                X_k[ik].resize(0, 0);

                if (ik == 0U) {
                    // Legacy output means mesh-index zero, which is not Gamma
                    // on a shifted mesh. The retained state below is the
                    // authoritative full-complex correlation reference.
                    result.mo_energies = diagonal.energies;
                    result.mo_coeffs = diagonal.C.real();
                }

                Eigen::VectorXd occupations =
                    Eigen::VectorXd::Zero(S_set.nbf);
                occupations.head(nocc).setConstant(2.0);
                std::vector<std::uint8_t> correlated(
                    static_cast<std::size_t>(S_set.nbf), 0U);
                std::vector<std::uint8_t> virtual_mask(
                    static_cast<std::size_t>(S_set.nbf), 0U);
                const auto& frozen =
                    capture_request->frozen_core_mask_per_k[ik];
                for (int orbital = 0; orbital < S_set.nbf; ++orbital) {
                    if (orbital < nocc
                        && frozen[static_cast<std::size_t>(orbital)] == 0U) {
                        correlated[static_cast<std::size_t>(orbital)] = 1U;
                    } else if (orbital >= nocc) {
                        virtual_mask[static_cast<std::size_t>(orbital)] = 1U;
                    }
                }

                input.add_kpoint(
                    kmesh.kpoints[ik],
                    uniform_weight,
                    std::move(S_k[ik]),
                    std::move(physical_fock_k),
                    std::move(diagonal.C),
                    std::move(diagonal.energies),
                    std::move(occupations),
                    std::move(capture_request->frozen_core_mask_per_k[ik]),
                    std::move(correlated),
                    std::move(virtual_mask));
            }

            // Release the empty orthogonalizer container before the state
            // factory's one-k-at-a-time validation.
            std::vector<ComplexMatrix>().swap(X_k);
            result.mean_field_state =
                make_periodic_restricted_mean_field_state(std::move(input));
            result.scf_trace.push_back(SCFIteration{
                iter,
                E_total,
                (iter == 1) ? 0.0 : dE,
                std::max(grad_norm, capture_commutator_max),
                0});
            VIBEQC_DIAG(
                "progress", vibeqc::DiagLevel::STANDARD,
                "iter=%d energy=%.10f grad=%.3e diis=%d",
                iter, E_total,
                std::max(grad_norm, capture_commutator_max), 0);
            result.density = P_used.blocks[0];
            result.fock = F_gamma;
            result.energy = E_total;
            result.e_electronic = E_elec;
            result.n_iter = iter;
            result.converged = true;
            return result;
        }

        int diis_subspace = 0;
        if (opts.use_diis) {
            // Accelerator dispatch. KDIIS multi-k is deferred (needs
            // per-k MO-basis design); raise an explicit error rather
            // than silently fall back. EDIIS / ADIIS run with full
            // per-cell history — the QP cross-term ⟨F_i | D_j⟩ sums
            // over the real-space cell list, which is exactly the
            // periodic energy bilinear form
            //   E_elec = ½ Σ_g (P(g) ⊙ [H(g) + F(g)]).sum().
            // DIIS still extrapolates on the Γ-folded Fock against the
            // Γ-folded commutator error; promoting it would need a
            // per-cell commutator error metric (Phase 12c+ tune-up).
            Eigen::MatrixXd F_ext_gamma;
            std::vector<Eigen::MatrixXd> F_ext_blocks;
            bool have_ext_per_cell = false;
            bool have_ext_gamma = false;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the DIIS extrapolation; the
                // adaptive depth policy is baked into ``diis`` (make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    F_ext_gamma = diis.extrapolate(F_gamma, error);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    have_ext_gamma = true;
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    throw std::runtime_error(
                        "run_rhf_periodic: KDIIS not yet supported for "
                        "multi-k SCF (needs a per-k MO-basis design). "
                        "Use DIIS, EDIIS, or EDIIS_DIIS for multi-k.");
                }
                case SCFAccelerator::EDIIS: {
                    F_ext_blocks = ediis.extrapolate(
                        F_set.blocks, P_used.blocks, E_total);
                    diis_subspace = static_cast<int>(ediis.subspace_size());
                    have_ext_per_cell = true;
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    // Keep both histories warm so the asymptotic switch
                    // is smooth. EDIIS regime: per-cell extrap. DIIS
                    // regime: Γ-folded extrap (asymptotic — the
                    // commutator-error correction is tiny by then, so
                    // the residual Γ-fold approximation is bounded by
                    // the convergence threshold).
                    F_ext_gamma = diis.extrapolate(F_gamma, error);
                    auto F_e_blocks = ediis.extrapolate(
                        F_set.blocks, P_used.blocks, E_total);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    // Intensive RMS metric (ediis.hpp) -- the raw
                    // Frobenius norm is size-extensive and over-holds
                    // the EDIIS regime (ddd8d325 defect class).
                    if (ediis_diis_switch_metric(grad_norm, F_gamma.rows())
                            > opts.ediis_diis_switch_threshold) {
                        F_ext_blocks = std::move(F_e_blocks);
                        have_ext_per_cell = true;
                    } else {
                        // A DIIS-branch cycle discards F_e_blocks; retract
                        // it so the anti-replay guard keys on consumed
                        // returns only.
                        ediis.discard_last_extrapolation();
                        have_ext_gamma = true;
                    }
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    F_ext_blocks = adiis.extrapolate(
                        F_set.blocks, P_used.blocks);
                    diis_subspace = static_cast<int>(adiis.subspace_size());
                    have_ext_per_cell = true;
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    // ADIIS analogue of EDIIS_DIIS: ADIIS (per-cell) far
                    // from convergence, DIIS (Γ-folded) in the asymptotic
                    // regime. See the EDIIS_DIIS case above.
                    F_ext_gamma = diis.extrapolate(F_gamma, error);
                    auto F_a_blocks = adiis.extrapolate(
                        F_set.blocks, P_used.blocks);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    if (ediis_diis_switch_metric(grad_norm, F_gamma.rows())
                            > opts.ediis_diis_switch_threshold) {
                        F_ext_blocks = std::move(F_a_blocks);
                        have_ext_per_cell = true;
                    } else {
                        adiis.discard_last_extrapolation();
                        have_ext_gamma = true;
                    }
                    break;
                }
            }
            if (diis_active) {
                if (have_ext_per_cell) {
                    F_set.blocks = std::move(F_ext_blocks);
                    F_gamma.setZero();
                    for (const auto& b : F_set.blocks) F_gamma += b;
                } else if (have_ext_gamma) {
                    // Γ-folded path (DIIS, or EDIIS_DIIS asymptotic):
                    // distribute the Γ correction uniformly across
                    // cells. Equivalent to the v0.9.0 multi-k DIIS
                    // path on insulators where damping alone carried.
                    const Eigen::MatrixXd delta = F_ext_gamma - F_gamma;
                    const double scale =
                        1.0 / static_cast<double>(F_set.blocks.size());
                    for (auto& b : F_set.blocks) b += scale * delta;
                    F_gamma = F_ext_gamma;
                }
            } else {
                // Below diis_start_iter nothing extrapolated reaches the
                // SCF, so every branch's return is discarded. Same
                // produced-vs-consumed retraction as the hybrid above.
                ediis.discard_last_extrapolation();
                adiis.discard_last_extrapolation();
            }
        }
        if (opts.fock_mixing != 0.0) {
            if (have_prev_fock) {
                mix_lattice_fock_set(
                    F_set, F_prev_mixed, opts.fock_mixing);
            }
            F_prev_mixed = F_set;
            have_prev_fock = true;
            F_gamma.setZero();
            for (const auto& b : F_set.blocks) F_gamma += b;
        }

        result.scf_trace.push_back(SCFIteration{
            iter, E_total, (iter == 1) ? 0.0 : dE, grad_norm, diis_subspace});

        VIBEQC_DIAG("progress", vibeqc::DiagLevel::STANDARD,
            "iter=%d energy=%.10f grad=%.3e diis=%d",
            iter, E_total, grad_norm, diis_subspace);

        // Saunders-Hillier level shift for this iteration, resolved by the
        // same helper the molecular and Γ periodic drivers use (base shift +
        // warm-up, or an explicit schedule) and applied per k by the shared
        // ``apply_level_shift_k`` operator. It raises the virtual bands by b
        // and leaves the occupied manifold fixed, so it is inert at the
        // converged density: it changes the SCF path, never the fixed point.
        // ``P_used`` is the closed-shell total density, hence TOTAL.
        const double b_shift = level_shift_at_iter(opts, iter);
        if (b_shift > 0.0) {
            VIBEQC_DIAG("periodic-rhf", vibeqc::DiagLevel::DEBUG,
                "level shift b=%.3e at iter %d", b_shift, iter);
        }

        // Diagonalize F(k) at every k to get new MOs, build P_new(g).
        for (std::size_t ik = 0; ik < nk; ++ik) {
            ComplexMatrix F_k = bloch_sum_fast(F_set, kmesh.kpoints[ik]);
            F_k = 0.5 * (F_k + F_k.adjoint().eval());
            // Guarded so the P(k) Bloch sum itself is skipped, not merely the
            // shift arithmetic, on any cycle whose resolved shift is 0.
            if (b_shift != 0.0) {
                const ComplexMatrix P_k =
                    bloch_sum_fast(P_used, kmesh.kpoints[ik]);
                F_k = apply_level_shift_k(F_k, S_k[ik], P_k, b_shift,
                                          LevelShiftDensity::TOTAL);
            }
            const auto dk = diag_k(F_k, X_k[ik], dav_ptr);
            C_k[ik] = dk.C;
            result.mo_energies_k[ik] = dk.energies;
            if (static_cast<std::size_t>(ik) == 0) {
                // Store Γ MO eigenvalues & coefficients as the "canonical"
                // output (convention used by most periodic codes that
                // report one MO list).
                result.mo_energies = dk.energies;
                result.mo_coeffs = dk.C.real();  // real at Γ
            }
        }
        P_prev = P_used;
        P_set = real_space_density_from_kpoints(
            C_k, n_occ_per_k, kmesh, S_set.cells);

        result.density = P_used.blocks[0];  // g=0 block at Γ
        result.fock = F_gamma;
        result.energy = E_total;
        result.e_electronic = E_elec;
        result.n_iter = iter;

        if (converged) {
            result.converged = true;
            break;
        }

        // Dynamic-damping update (Zerner-Hehenberger 1979). Inactive
        // unless opts.dynamic_damping is true.
        if (opts.dynamic_damping) {
            current_damping = update_dynamic_damping(
                current_damping, E_total, E_prev, have_prev_E,
                opts.dynamic_damping_min, opts.dynamic_damping_max);
        }
        have_prev_E = true;

        E_prev = E_total;
    }
    // A zero-iteration result has no SCF Bloch orbital state to export.
    if (result.n_iter == 0) {
        result.mo_energies_k.clear();
        return result;
    }
    result.mo_coeffs_k = C_k;
    for (const auto& c : C_k) {
        Eigen::VectorXd occ = Eigen::VectorXd::Zero(c.cols());
        occ.head(nocc).setConstant(2.0);
        result.occupations_k.push_back(std::move(occ));
    }
    return result;
}

}  // namespace

PeriodicRHFResult run_rhf_periodic(const PeriodicSystem& system,
                                   const BasisSet& basis,
                                   const BlochKMesh& kmesh,
                                   const PeriodicSCFOptions& opts) {
    return run_rhf_periodic_impl(system, basis, kmesh, opts, nullptr);
}

PeriodicRHFResult run_rhf_periodic_with_state_capture(
    const PeriodicSystem& system,
    const BasisSet& basis,
    const BlochKMesh& kmesh,
    const PeriodicSCFOptions& opts,
    PeriodicRHFStateCaptureRequest request) {
    return run_rhf_periodic_impl(system, basis, kmesh, opts, &request);
}

// ---------------------------------------------------------------------------
// Multi-k periodic KS-DFT
// ---------------------------------------------------------------------------

PeriodicKSResult run_rks_periodic(const PeriodicSystem& system,
                                  const BasisSet& basis,
                                  const BlochKMesh& kmesh,
                                  const PeriodicKSOptions& opts) {
    validate_guess_kmesh(kmesh);
    std::vector<InitialGuess> guess_capabilities{
        InitialGuess::HCORE, InitialGuess::SAD, InitialGuess::PATOM,
        InitialGuess::HUECKEL, InitialGuess::MINAO, InitialGuess::READ};
    if (system.dim == 3) guess_capabilities.push_back(InitialGuess::SAP);
    const InitialGuess effective_guess = GuessEngine::resolve_auto_for_molecule(
        system.unit_cell_molecule(), opts.initial_guess, true, false, guess_capabilities);
    const auto guess_mol = system.unit_cell_molecule();
    const auto guess_ecp = periodic_guess_ecp_context(opts, &guess_mol);
    validate_guess_ecp(effective_guess, guess_ecp, &guess_mol);
    if (!opts.atomic_spins.empty()) {
        throw GuessCapabilityError("native periodic RKS cannot represent an atomic spin seed");
    }
    if (opts.spinlock_mode != SpinlockMode::OFF && opts.spinlock_iterations > 0) {
        throw GuessCapabilityError("native periodic RKS cannot execute a spin schedule");
    }
    const int n_elec = system.n_electrons();
    const bool _rks_has_ecp = !opts.ecp_primitive_blocks.empty();
    const int n_elec_valence_rks = _rks_has_ecp
        ? n_elec - opts.ecp_total_ncore
        : n_elec;
    if (n_elec_valence_rks % 2 != 0) {
        throw std::invalid_argument(
            "run_rks_periodic: closed-shell KS requires even electron "
            "count per unit cell, got " + std::to_string(n_elec_valence_rks) +
            (_rks_has_ecp ? " (valence, with ECP ncore=" +
             std::to_string(opts.ecp_total_ncore) + ")" : ""));
    }
    if (system.multiplicity != 1) {
        throw std::invalid_argument(
            "run_rks_periodic: closed-shell KS requires multiplicity = 1");
    }
    if (kmesh.kpoints.empty()) {
        throw std::invalid_argument("run_rks_periodic: empty k-mesh");
    }
    validate_fraction_01("run_rks_periodic: damping", opts.damping);
    validate_fraction_01("run_rks_periodic: fock_mixing",
                         opts.fock_mixing);
    const int nocc = n_elec_valence_rks / 2;

    Functional func(opts.functional, /*spin=*/1);
    const double exchange_scale = func.hf_exchange_fraction();

    // ---- One-electron lattice integrals ---------------------------------
    const auto S_set = on_eri_cells(compute_overlap_lattice(basis, system, opts.lattice_opts),
                                    basis, system, opts.lattice_opts);
    const bool use_davidson_here_rks =
        opts.use_davidson && S_set.nbf >= opts.davidson_min_dim;
    DavidsonOptions dav_opts_rks = opts.davidson;
    if (use_davidson_here_rks) {
        if (dav_opts_rks.n_eig == 0) dav_opts_rks.n_eig = S_set.nbf;
    }
    const DavidsonOptions* dav_ptr_rks =
        use_davidson_here_rks ? &dav_opts_rks : nullptr;
    const auto T_set = on_eri_cells(compute_kinetic_lattice(basis, system, opts.lattice_opts),
                                    basis, system, opts.lattice_opts);
    LatticeMatrixSet V_set;
    LatticeMatrixSet V_ecp_set;
    if (_rks_has_ecp) {
        V_set = on_eri_cells(compute_nuclear_lattice_with_charges(
            basis, system, opts.lattice_opts, opts.ecp_effective_charges), basis, system, opts.lattice_opts);
        V_ecp_set = on_eri_cells(compute_ecp_lattice_from_primitives(
            basis, system, opts.lattice_opts,
            opts.ecp_home_centers, opts.ecp_primitive_blocks), basis, system, opts.lattice_opts);
    } else {
        V_set = on_eri_cells(compute_nuclear_lattice(basis, system, opts.lattice_opts), basis, system, opts.lattice_opts);
    }
    LatticeMatrixSet Hcore_set;
    Hcore_set.nbf = S_set.nbf;
    Hcore_set.cells = S_set.cells;
    Hcore_set.blocks.resize(S_set.cells.size());
    for (std::size_t c = 0; c < S_set.cells.size(); ++c) {
        Hcore_set.blocks[c] = T_set.blocks[c] + V_set.blocks[c];
        if (_rks_has_ecp && c < V_ecp_set.blocks.size()) {
            Hcore_set.blocks[c] += V_ecp_set.blocks[c];
        }
    }

    // Atom-major quadrature with the same physical point partition used
    // by the Python periodic DFT routes.
    const Grid grid = opts.use_periodic_becke
        ? build_periodic_point_grid(system, opts.becke_image_radius_bohr, opts.grid)
        : build_grid(system.unit_cell_molecule(), opts.grid);

    // ---- Precompute per-k S(k) and X(k)=S^{-1/2} ------------------------
    const std::size_t nk = kmesh.kpoints.size();
    std::vector<ComplexMatrix> S_k(nk), X_k(nk), H_k(nk);
    Eigen::MatrixXd S_effective = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        S_k[ik] = bloch_sum_fast(S_set, kmesh.kpoints[ik]);
        S_k[ik] = 0.5 * (S_k[ik] + S_k[ik].adjoint().eval());
        S_effective += kmesh.weights[ik] * S_k[ik].real();
        X_k[ik] = s_inverse_sqrt_complex(S_k[ik]);
        H_k[ik] = bloch_sum_fast(Hcore_set, kmesh.kpoints[ik]);
        H_k[ik] = 0.5 * (H_k[ik] + H_k[ik].adjoint().eval());
    }

    // ---- Initial guess ---------------------------------------------------
    std::vector<ComplexMatrix> C_k(nk);
    std::vector<int> n_occ_per_k(nk, nocc);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        const auto dk = diag_k(H_k[ik], X_k[ik], dav_ptr_rks);
        C_k[ik] = dk.C;
    }
    LatticeMatrixSet P_set =
        real_space_density_from_kpoints(C_k, n_occ_per_k, kmesh, S_set.cells);

    if (effective_guess == InitialGuess::READ) {
        P_set = read_initial_density_multik(opts.read_density_k, S_k, kmesh,
                                          system, S_set.cells, 2 * nocc);
    } else if (effective_guess == InitialGuess::SAP) {
        // Periodic SAP Fock-mode guess, wired through the multi-k RKS path:
        // F_SAP(k) = T(k) + V_SAP(k), diagonalised at each k.
        P_set = sap_initial_density_multik(
            basis, system, T_set, S_set.cells, kmesh, X_k, nocc,
            opts.lattice_opts, guess_ecp);
    } else if (effective_guess == InitialGuess::HUECKEL) {
        // Periodic HUECKEL Fock-mode guess:
        // F_GWH(g) from S(g) and atomic AO energies, diagonalised at each k.
        const auto F_huckel = compute_huckel_fock_lattice(
            system.unit_cell_molecule(), basis, S_set, guess_ecp);
        P_set = fock_initial_density_multik(
            F_huckel, S_set.cells, kmesh, X_k, nocc);
    } else if (effective_guess == InitialGuess::MINAO) {
        // Periodic MINAO density-mode guess: project the ANO-RCC reference
        // density onto the working basis with the lattice-summed Γ overlaps,
        // place it on the g=0 cell (Bloch-summing recovers a k-independent
        // D(k) for the first Fock build), exactly like SAD.
        Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
        for (const auto& b : S_set.blocks) S_gamma += b;
        const Eigen::MatrixXd D_minao = compute_minao_density_periodic(
            system.unit_cell_molecule(), basis, system, S_gamma,
            2 * nocc, opts.lattice_opts, guess_ecp);
        P_set = local_density_from_home_block(
            normalize_guess_density(D_minao, S_effective, 2.0 * nocc),
            S_set.cells);
    } else if (effective_guess == InitialGuess::PATOM) {
        P_set = patom_initial_density_multik(
            basis, system, Hcore_set, S_set.cells, kmesh, X_k, nocc,
            opts.lattice_opts, S_effective, guess_ecp);
    } else {
        const auto g = GuessEngine::build_closed_shell(
            system.unit_cell_molecule(), basis, nocc,
            effective_guess,
            /*S=*/&S_effective, /*Hcore=*/nullptr, /*jk=*/nullptr,
            SystemHints{/*is_periodic=*/true}, 1e-7, guess_ecp);
        if (g.D.size() != 0) {
            for (auto& b : P_set.blocks) b.setZero();
            P_set.blocks[0] = g.D;
        }
    }

    double e_nuc;
    if (_rks_has_ecp && !opts.ecp_effective_charges.empty()) {
        const auto rep_cells = direct_lattice_cells(
            system, opts.lattice_opts.nuclear_cutoff_bohr);
        double enuc = 0.0;
        const auto& atoms = system.unit_cell;
        for (const auto& rc : rep_cells) {
            for (std::size_t i = 0; i < atoms.size(); ++i) {
                double qi = i < opts.ecp_effective_charges.size()
                    ? opts.ecp_effective_charges[i]
                    : static_cast<double>(atoms[i].Z);
                std::size_t j0 = (rc.r_cart.squaredNorm() < 1e-24) ? i + 1 : 0;
                for (std::size_t j = j0; j < atoms.size(); ++j) {
                    double qj = j < opts.ecp_effective_charges.size()
                        ? opts.ecp_effective_charges[j]
                        : static_cast<double>(atoms[j].Z);
                    double dx = atoms[i].xyz[0] + rc.r_cart[0] - atoms[j].xyz[0];
                    double dy = atoms[i].xyz[1] + rc.r_cart[1] - atoms[j].xyz[1];
                    double dz = atoms[i].xyz[2] + rc.r_cart[2] - atoms[j].xyz[2];
                    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (r > 1e-12) enuc += qi * qj / r;
                }
            }
        }
        e_nuc = 0.5 * enuc;
    } else {
        e_nuc = nuclear_repulsion_per_cell(system, opts.lattice_opts);
    }

    // Γ-folded overlap for DIIS / gradient measure.
    Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
    for (const auto& b : S_set.blocks) S_gamma += b;

    PeriodicKSResult result;
    result.mo_energies_k.resize(nk);
    result.restart_basis = basis;
    result.restart_lattice = system.lattice;
    result.guess_selection = {opts.initial_guess, effective_guess, effective_guess};
    result.restart_kpoints.resize(kmesh.kpoints.size(), 3);
    result.restart_weights.resize(kmesh.weights.size());
    for (std::size_t ik = 0; ik < kmesh.kpoints.size(); ++ik) {
        result.restart_kpoints.row(ik) = kmesh.kpoints[ik].transpose();
        result.restart_weights(ik) = kmesh.weights[ik];
    }

    result.functional = opts.functional;
    result.e_nuclear = e_nuc;
    result.overlap = S_gamma;

    DIIS diis = make_diis(opts);
    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);
    LatticeMatrixSet P_prev = P_set;
    LatticeMatrixSet F_prev_mixed;
    bool have_prev_fock = false;
    double E_prev = 0.0;
    double current_damping = opts.damping;
    bool have_prev_E = false;

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter;

        LatticeMatrixSet P_used = P_set;
        if (iter > 1 && current_damping > 0.0 && !diis_active) {
            for (std::size_t c = 0; c < P_set.blocks.size(); ++c) {
                P_used.blocks[c] =
                    current_damping * P_prev.blocks[c]
                    + (1.0 - current_damping) * P_set.blocks[c];
            }
        }

        // Two-electron contributions.
        // exchange_scale = 0 for pure DFT skips the K lattice sum entirely.
        const auto F_HFpart_set = build_fock_2e_real_space(
            basis, system, opts.lattice_opts, P_used, exchange_scale);
        const auto xc_contrib = build_xc_periodic(
            basis, system, grid, func, P_used, opts.lattice_opts);

        // F(g) = Hcore(g) + F^{HF-part}(g) + V_xc(g)
        LatticeMatrixSet F_set;
        F_set.nbf = S_set.nbf;
        F_set.cells = S_set.cells;
        F_set.blocks.resize(S_set.cells.size());
        // Hcore is zero-extended onto the exchange/density enclosure.
        if (F_HFpart_set.blocks.size() > S_set.cells.size()) {
            throw std::runtime_error(
                "run_rks_periodic: 2e Fock cell list is longer than Hcore's");
        }
        for (std::size_t c = 0; c < S_set.cells.size(); ++c) {
            F_set.blocks[c] =
                Hcore_set.blocks[c] + xc_contrib.V_xc.blocks[c];
            if (c < F_HFpart_set.blocks.size()) {
                F_set.blocks[c] += F_HFpart_set.blocks[c];
            }
        }

        // Energy:  E_elec = Σ_g tr(P(g)·Hcore(g))
        //                   + (1/2) Σ_g tr(P(g)·F^{HF-part}(g))  + E_xc
        //
        // where F^{HF-part} = J − (α/2) K. V_xc is reported through E_xc
        // rather than a trace, so no double-counting subtraction is needed.
        // The Dudarev +U contribution is also added separately (below),
        // not folded into the per-cell trace — see cpp/src/rhf.cpp for
        // the molecular analog and the double-counting note in
        // cpp/include/vibeqc/dft_plus_u.hpp.
        double E_elec = xc_contrib.e_xc;
        for (std::size_t c = 0; c < F_set.blocks.size(); ++c) {
            E_elec += (P_used.blocks[c].array()
                       * Hcore_set.blocks[c].array()).sum();
            if (c < F_HFpart_set.blocks.size()) {
                E_elec += 0.5 * (P_used.blocks[c].array()
                                 * F_HFpart_set.blocks[c].array()).sum();
            }
        }

        // Dudarev +U at multi-k (Increment 4c). Generalises the
        // Γ-only path: the AO occupation matrix is k-averaged
        //   n^A_l = (1/2) Σ_k w_k Re[(S(k) P(k) S(k))_(A,l)]
        // (per-spin, closed-shell convention with P_σ = P/2). The
        // variational per-k Fock contribution is S(k) V_AO S(k);
        // adding the **real-space convolution**
        //   X(g) = Σ_{g1+g2=g} S(g1) V_AO S(g2)
        // to F_set.blocks recovers ``S(k) V_AO S(k)`` at every k
        // automatically via the Bloch sum (so DIIS, the FDS-SDF
        // commutator error vector, and the per-k diagonalisation all
        // see the +U term consistently).
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            std::vector<Eigen::MatrixXcd> P_k_for_U(nk);
            for (std::size_t ik = 0; ik < nk; ++ik) {
                P_k_for_U[ik] =
                    bloch_sum_fast(P_used, kmesh.kpoints[ik]);
                P_k_for_U[ik] = 0.5 * (P_k_for_U[ik]
                                        + P_k_for_U[ik].adjoint().eval());
            }
            const auto vu_mk = compute_dft_plus_u_multi_k_closed_shell(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                S_k, P_k_for_U, kmesh.weights);
            e_dft_plus_u = vu_mk.energy_total;
            // Compute X(g) = Σ_{g1+g2=g} S(g1) V_AO S(g2) and add
            // to F_set. Look up cell indices via integer offsets.
            const Eigen::MatrixXd& V_AO = vu_mk.V_AO_per_spin;
            std::map<std::array<int, 3>, std::size_t> cell_index_of;
            for (std::size_t i = 0; i < S_set.cells.size(); ++i) {
                cell_index_of[{
                    S_set.cells[i].index[0],
                    S_set.cells[i].index[1],
                    S_set.cells[i].index[2],
                }] = i;
            }
            for (std::size_t i1 = 0; i1 < S_set.cells.size(); ++i1) {
                for (std::size_t i2 = 0; i2 < S_set.cells.size(); ++i2) {
                    const auto& g1 = S_set.cells[i1].index;
                    const auto& g2 = S_set.cells[i2].index;
                    const std::array<int, 3> g_sum = {
                        g1[0] + g2[0], g1[1] + g2[1], g1[2] + g2[2],
                    };
                    auto it = cell_index_of.find(g_sum);
                    if (it == cell_index_of.end()) {
                        continue;  // outside the lattice cutoff
                    }
                    F_set.blocks[it->second].noalias() +=
                        S_set.blocks[i1] * V_AO * S_set.blocks[i2];
                }
            }
        }

        const double E_total = E_elec + e_nuc + e_dft_plus_u;

        // Γ-folded matrices for convergence test & DIIS.
        Eigen::MatrixXd F_gamma = Eigen::MatrixXd::Zero(S_set.nbf, S_set.nbf);
        Eigen::MatrixXd P_gamma = F_gamma;
        for (std::size_t c = 0; c < F_set.blocks.size(); ++c) {
            F_gamma += F_set.blocks[c];
            P_gamma += P_used.blocks[c];
        }
        const Eigen::MatrixXd error =
            F_gamma * P_gamma * S_gamma - S_gamma * P_gamma * F_gamma;
        const double grad_norm = error.norm();
        const double dE = E_total - E_prev;

        VIBEQC_DIAG("periodic-rks", vibeqc::DiagLevel::VERBOSE,
            "iter %3d  E=% .10f  dE=%+.3e  |grad|=%.3e",
            iter, E_total, (iter == 1 ? 0.0 : dE), grad_norm);

        const bool converged = is_scf_converged(
            iter, dE, grad_norm,
            opts.conv_tol_energy, opts.conv_tol_grad);
        if (converged) {
            VIBEQC_DIAG("periodic-rks", vibeqc::DiagLevel::DEBUG,
                "converged at iter %d: |dE|=%.3e < %.0e, |grad|=%.3e < %.0e",
                iter, std::abs(dE), opts.conv_tol_energy,
                grad_norm, opts.conv_tol_grad);
        }

        int diis_subspace = 0;
        if (opts.use_diis) {
            // See run_rhf_periodic above for the per-cell EDIIS / ADIIS
            // rationale — the QP cross-term sums over the real-space
            // cell list, matching the periodic energy bilinear form.
            Eigen::MatrixXd F_ext_gamma;
            std::vector<Eigen::MatrixXd> F_ext_blocks;
            bool have_ext_per_cell = false;
            bool have_ext_gamma = false;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the DIIS extrapolation; the
                // adaptive depth policy is baked into ``diis`` (make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    F_ext_gamma = diis.extrapolate(F_gamma, error);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    have_ext_gamma = true;
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    throw std::runtime_error(
                        "run_rks_periodic: KDIIS not yet supported for "
                        "multi-k SCF (needs a per-k MO-basis design). "
                        "Use DIIS, EDIIS, or EDIIS_DIIS for multi-k.");
                }
                case SCFAccelerator::EDIIS: {
                    F_ext_blocks = ediis.extrapolate(
                        F_set.blocks, P_used.blocks, E_total);
                    diis_subspace = static_cast<int>(ediis.subspace_size());
                    have_ext_per_cell = true;
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    F_ext_gamma = diis.extrapolate(F_gamma, error);
                    auto F_e_blocks = ediis.extrapolate(
                        F_set.blocks, P_used.blocks, E_total);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    // Intensive RMS metric (ediis.hpp) -- the raw
                    // Frobenius norm is size-extensive and over-holds
                    // the EDIIS regime (ddd8d325 defect class).
                    if (ediis_diis_switch_metric(grad_norm, F_gamma.rows())
                            > opts.ediis_diis_switch_threshold) {
                        F_ext_blocks = std::move(F_e_blocks);
                        have_ext_per_cell = true;
                    } else {
                        // A DIIS-branch cycle discards F_e_blocks; retract
                        // it so the anti-replay guard keys on consumed
                        // returns only.
                        ediis.discard_last_extrapolation();
                        have_ext_gamma = true;
                    }
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    F_ext_blocks = adiis.extrapolate(
                        F_set.blocks, P_used.blocks);
                    diis_subspace = static_cast<int>(adiis.subspace_size());
                    have_ext_per_cell = true;
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    // ADIIS analogue of EDIIS_DIIS: ADIIS (per-cell) far
                    // from convergence, DIIS (Γ-folded) in the asymptotic
                    // regime. See the EDIIS_DIIS case above.
                    F_ext_gamma = diis.extrapolate(F_gamma, error);
                    auto F_a_blocks = adiis.extrapolate(
                        F_set.blocks, P_used.blocks);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    if (ediis_diis_switch_metric(grad_norm, F_gamma.rows())
                            > opts.ediis_diis_switch_threshold) {
                        F_ext_blocks = std::move(F_a_blocks);
                        have_ext_per_cell = true;
                    } else {
                        adiis.discard_last_extrapolation();
                        have_ext_gamma = true;
                    }
                    break;
                }
            }
            if (diis_active) {
                if (have_ext_per_cell) {
                    F_set.blocks = std::move(F_ext_blocks);
                    F_gamma.setZero();
                    for (const auto& b : F_set.blocks) F_gamma += b;
                } else if (have_ext_gamma) {
                    const Eigen::MatrixXd delta = F_ext_gamma - F_gamma;
                    const double scale = 1.0
                        / static_cast<double>(F_set.blocks.size());
                    for (auto& b : F_set.blocks) b += scale * delta;
                    F_gamma = F_ext_gamma;
                }
            }
        }
        if (opts.fock_mixing != 0.0) {
            if (have_prev_fock) {
                mix_lattice_fock_set(
                    F_set, F_prev_mixed, opts.fock_mixing);
            }
            F_prev_mixed = F_set;
            have_prev_fock = true;
            F_gamma.setZero();
            for (const auto& b : F_set.blocks) F_gamma += b;
        }

        result.scf_trace.push_back(SCFIteration{
            iter, E_total, (iter == 1) ? 0.0 : dE, grad_norm, diis_subspace});

        VIBEQC_DIAG("progress", vibeqc::DiagLevel::STANDARD,
            "iter=%d energy=%.10f grad=%.3e diis=%d",
            iter, E_total, grad_norm, diis_subspace);

        // Re-diagonalize at every k with the updated F(g). The +U
        // term (Increment 4c) is already inside F_set via the
        // real-space convolution above, so the Bloch sum picks up
        // S(k) V_AO S(k) at every k automatically.
        // Saunders-Hillier level shift for this iteration; see the identical
        // note in the multi-k RHF loop above. Inert at the converged density;
        // the P(k) Bloch sum + S·P·S products are skipped when b is 0.
        const double b_shift = level_shift_at_iter(opts, iter);

        for (std::size_t ik = 0; ik < nk; ++ik) {
            ComplexMatrix F_k = bloch_sum_fast(F_set, kmesh.kpoints[ik]);
            F_k = 0.5 * (F_k + F_k.adjoint().eval());
            if (b_shift != 0.0) {
                const ComplexMatrix P_k =
                    bloch_sum_fast(P_used, kmesh.kpoints[ik]);
                F_k = apply_level_shift_k(F_k, S_k[ik], P_k, b_shift,
                                          LevelShiftDensity::TOTAL);
            }
            const auto dk = diag_k(F_k, X_k[ik], dav_ptr_rks);
            C_k[ik] = dk.C;
            result.mo_energies_k[ik] = dk.energies;
            if (ik == 0) {
                result.mo_energies = dk.energies;
                result.mo_coeffs = dk.C.real();
            }
        }
        P_prev = P_used;
        P_set = real_space_density_from_kpoints(
            C_k, n_occ_per_k, kmesh, S_set.cells);

        // Populate energy breakdown (last iteration's contributions).
        double e_hf_K = 0.0;
        if (exchange_scale != 0.0) {
            // Fast approximation of -(α/2) tr(P·K): decompose F^{HF-part} to
            // separate J and K. Rather than recomputing, leverage the fact
            // that F^{HF-part} = J − ½·α·K and compute J directly via the
            // exchange_scale=0 path on demand. For the trace report we
            // accept a small overhead once per iteration.
            const auto J_only_set = build_fock_2e_real_space(
                basis, system, opts.lattice_opts, P_used, 0.0);
            // The two-electron sets ride the plain |g| ball, a prefix of
            // the density's (pair-complete under pair_complete_1e, #429)
            // list, and are zero beyond their own support: trace them
            // over their own length, never over F_set's.
            double trJ = 0.0;
            for (std::size_t c = 0; c < J_only_set.blocks.size(); ++c) {
                trJ += (P_used.blocks[c].array()
                        * J_only_set.blocks[c].array()).sum();
            }
            // tr(P·(F^{HF-part})) = tr(P·J) − ½·α·tr(P·K)
            double trFhf = 0.0;
            for (std::size_t c = 0; c < F_HFpart_set.blocks.size(); ++c) {
                trFhf += (P_used.blocks[c].array()
                          * F_HFpart_set.blocks[c].array()).sum();
            }
            const double trK = (trJ - trFhf) * 2.0 / exchange_scale;
            result.e_coulomb = 0.5 * trJ;
            e_hf_K = -0.5 * exchange_scale * trK;
        } else {
            double trJ = 0.0;
            for (std::size_t c = 0; c < F_HFpart_set.blocks.size(); ++c) {
                trJ += (P_used.blocks[c].array()
                        * F_HFpart_set.blocks[c].array()).sum();
            }
            result.e_coulomb = 0.5 * trJ;
        }
        result.e_hf_exchange = e_hf_K;
        result.e_xc = xc_contrib.e_xc;
        result.e_dft_plus_u = e_dft_plus_u;
        result.density = P_used.blocks[0];
        result.fock = F_gamma;
        result.energy = E_total;
        result.e_electronic = E_elec;
        result.n_iter = iter;

        if (converged) {
            result.converged = true;
            break;
        }

        if (opts.dynamic_damping) {
            current_damping = update_dynamic_damping(
                current_damping, E_total, E_prev, have_prev_E,
                opts.dynamic_damping_min, opts.dynamic_damping_max);
        }
        have_prev_E = true;

        E_prev = E_total;
    }
    // A zero-iteration result has no SCF Bloch orbital state to export.
    if (result.n_iter == 0) {
        result.mo_energies_k.clear();
        return result;
    }
    result.mo_coeffs_k = C_k;
    for (const auto& c : C_k) {
        Eigen::VectorXd occ = Eigen::VectorXd::Zero(c.cols());
        occ.head(nocc).setConstant(2.0);
        result.occupations_k.push_back(std::move(occ));
    }
    return result;
}

}  // namespace vibeqc
