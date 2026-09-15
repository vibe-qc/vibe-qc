#include "vibeqc/bloch.hpp"

#include "vibeqc/crystal.hpp"
#include "vibeqc/kmesh_address.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace vibeqc {

ComplexMatrix bloch_sum(const LatticeMatrixSet& real_space,
                        const Eigen::Vector3d& k_cart) {
    const int nbf = real_space.nbf;
    ComplexMatrix M = ComplexMatrix::Zero(nbf, nbf);
    for (std::size_t c = 0; c < real_space.cells.size(); ++c) {
        const double phase = k_cart.dot(real_space.cells[c].r_cart);
        const std::complex<double> z(std::cos(phase), std::sin(phase));
        // Add z * block. Eigen complex<->real promotion works naturally.
        M += z * real_space.blocks[c];
    }
    return M;
}

BlochKMesh monkhorst_pack(const PeriodicSystem& system,
                          std::array<int, 3> mesh,
                          std::array<int, 3> is_shift,
                          bool use_symmetry) {
    for (int i = 0; i < 3; ++i) {
        if (mesh[i] < 1) {
            throw std::runtime_error("monkhorst_pack: mesh[i] must be ≥ 1");
        }
        // Non-periodic axes are forced to a single Γ-point slice.
        if (i >= system.dim) {
            mesh[i] = 1;
            is_shift[i] = 0;
        }
    }

    // Validate the shift AFTER the normalisation above, so the two branches
    // below accept and reject exactly the same inputs. Before #691 only the
    // unreduced branch checked, and it did so implicitly by constructing a
    // RegularKMesh; use_symmetry=true went to spglib unchecked and built a
    // silently different grid. Validating here rather than only in the
    // descriptor keeps the error attributed to the function the caller
    // named, and deliberately still tolerates a stray shift on a
    // non-periodic axis, which the normalisation has already zeroed.
    for (int i = 0; i < 3; ++i) {
        if (is_shift[i] != 0 && is_shift[i] != 1) {
            throw std::runtime_error(
                "monkhorst_pack: is_shift[" + std::to_string(i) + "] = " +
                std::to_string(is_shift[i]) +
                " must be 0 or 1; it is a half-step flag, not a "
                "displacement");
        }
    }

    const Eigen::Matrix3d B = system.reciprocal_lattice();  // columns = b_i

    BlochKMesh out;
    out.mesh = mesh;
    out.is_shift = is_shift;

    if (use_symmetry) {
        if (!system.symmetry.has_value()) {
            throw std::runtime_error(
                "monkhorst_pack: use_symmetry=true requires system.symmetry "
                "to be populated via attach_symmetry(system)");
        }
        // Re-derive the Crystal from the PeriodicSystem for spglib.
        Crystal crystal;
        crystal.lattice = system.lattice;
        crystal.fractional_coords.resize(3, system.unit_cell.size());
        crystal.species.resize(system.unit_cell.size());
        const auto lat_inv = system.lattice.inverse();
        for (std::size_t i = 0; i < system.unit_cell.size(); ++i) {
            const auto& a = system.unit_cell[i];
            Eigen::Vector3d r(a.xyz[0], a.xyz[1], a.xyz[2]);
            crystal.fractional_coords.col(i) = lat_inv * r;
            crystal.species[i] = a.Z;
        }
        const auto ir = irreducible_kpoints(crystal, mesh, is_shift);
        out.kpoints.reserve(ir.fractional_kpoints.size());
        for (const auto& kf : ir.fractional_kpoints) {
            out.kpoints.push_back(B * kf);
        }
        out.weights = ir.weights;
        out.ir_mapping = ir.ir_mapping;
        return out;
    }

    // Full mesh, no reduction. The grid, its order, and its coordinates are
    // owned by RegularKMesh (kmesh_address.hpp): index
    // (m_0 * N_1 + m_1) * N_2 + m_2 with fractional coordinate
    // a_d / (2 N_d), a_d = 2 m_d + s_d.
    //
    // Byte-neutral against the (m_d + s_d/2) / N_d form this replaced.
    // Numerator and denominator are both exactly twice the old ones, so the
    // real quotient is unchanged; every value involved (m_d, s_d/2, N_d,
    // 2 m_d + s_d, 2 N_d) is exactly representable, and IEEE-754 division is
    // correctly rounded. The two expressions therefore produce the same
    // double, not merely nearby ones -- pinned by
    // tests/test_kmesh_address.py::test_bloch_mesh_matches_legacy_formula_bitwise.
    //
    // The constructor also validates what this function only documented:
    // is_shift outside {0, 1}, and a division product that overflows the
    // point count, are now refused here rather than silently producing a
    // grid that is not the one the caller named.
    const RegularKMesh addressing(mesh, is_shift);
    const std::size_t n_total = addressing.size();
    out.kpoints.reserve(n_total);
    out.weights.assign(n_total, 1.0 / static_cast<double>(n_total));
    for (std::size_t i = 0; i < n_total; ++i) {
        out.kpoints.push_back(B * addressing.fractional_at(i));
    }
    return out;
}

BandDiag diagonalize_bloch(const ComplexMatrix& F_k,
                           const ComplexMatrix& S_k,
                           double lindep_threshold) {
    const int n = static_cast<int>(F_k.rows());
    if (S_k.rows() != n || S_k.cols() != n || F_k.cols() != n) {
        throw std::runtime_error("diagonalize_bloch: F and S shape mismatch");
    }

    // Symmetric orthogonalisation: S = U diag(s) U†, X = U diag(s^{-½}).
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> es_S(S_k);
    if (es_S.info() != Eigen::Success) {
        throw std::runtime_error("diagonalize_bloch: S eigensolver failed");
    }
    const Eigen::VectorXd s_eig = es_S.eigenvalues();
    const double s_min = s_eig.minCoeff();
    if (s_min < lindep_threshold) {
        throw std::runtime_error(
            "diagonalize_bloch: S(k) is near-singular (min eigenvalue "
            + std::to_string(s_min) + " < threshold "
            + std::to_string(lindep_threshold) + ")");
    }

    const ComplexMatrix& U = es_S.eigenvectors();
    ComplexVector s_half_inv = s_eig.cwiseInverse().cwiseSqrt().cast<std::complex<double>>();
    ComplexMatrix X = U * s_half_inv.asDiagonal();

    // F' = X† F X, then diagonalize.
    ComplexMatrix F_prime = X.adjoint() * F_k * X;
    // Symmetrize numerically to damp tiny non-Hermitian drift from the
    // complex arithmetic (imaginary part of diagonal etc.).
    F_prime = 0.5 * (F_prime + F_prime.adjoint().eval());

    Eigen::SelfAdjointEigenSolver<ComplexMatrix> es_F(F_prime);
    if (es_F.info() != Eigen::Success) {
        throw std::runtime_error("diagonalize_bloch: F eigensolver failed");
    }

    BandDiag out;
    out.energies = es_F.eigenvalues();
    out.coefficients = X * es_F.eigenvectors();
    return out;
}

}  // namespace vibeqc
