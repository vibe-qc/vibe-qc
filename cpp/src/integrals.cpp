#include "vibeqc/integrals.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/engine.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vibeqc {

namespace {

// Fill both triangles of a symmetric matrix from libint engine output for a
// single 1-electron operator (overlap, kinetic, or nuclear attraction).
Eigen::MatrixXd compute_1e_matrix(
    const BasisSet& basis,
    libint2::Operator op,
    const std::vector<std::pair<double, std::array<double, 3>>>* nuclei) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto nbf = basis.nbasis();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine prototype(op, shells.max_nprim(), shells.max_l(), 0);
    if (op == libint2::Operator::nuclear && nuclei != nullptr) {
        prototype.set_params(*nuclei);
    }
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    // Parallelise the outer shell loop. Each thread writes to a distinct
    // set of rows (determined by its s1 value), so the scatter to M is
    // free of races without any locking or per-thread accumulation. For
    // the symmetric (s2, s1) scatter the target is never the same row as
    // the thread's own s1, so it lands in another thread's region — but
    // different threads' (s1, s2, s2_sym) writes never target the same
    // matrix element because the (s1, s2) quartet is unique per thread.
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();

            engine.compute(shells[s1], shells[s2]);
            const double* block = buf[0];
            if (!block) continue;

            for (std::size_t i = 0; i < n1; ++i) {
                for (std::size_t j = 0; j < n2; ++j) {
                    const double val = block[i * n2 + j];
                    M(bf1 + i, bf2 + j) = val;
                    if (s1 != s2) {
                        M(bf2 + j, bf1 + i) = val;
                    }
                }
            }
        }
    }
    return M;
}

}  // namespace

Eigen::MatrixXd compute_overlap(const BasisSet& basis) {
    return compute_1e_matrix(basis, libint2::Operator::overlap, nullptr);
}

Eigen::MatrixXd compute_overlap_two_basis(const BasisSet& basis1,
                                          const BasisSet& basis2) {
    ensure_libint_initialized();
    const auto& sh1 = basis1.libint();
    const auto& sh2 = basis2.libint();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(basis1.nbasis(), basis2.nbasis());

    // One engine sized for the larger of the two bases. No permutational
    // symmetry to exploit (the two index ranges are distinct), so we walk
    // the full shell1 x shell2 product.
    libint2::Engine engine(
        libint2::Operator::overlap,
        std::max(sh1.max_nprim(), sh2.max_nprim()),
        std::max(sh1.max_l(), sh2.max_l()), 0);
    const auto& buf = engine.results();
    const auto s2bf1 = sh1.shell2bf();
    const auto s2bf2 = sh2.shell2bf();

    for (std::size_t i1 = 0; i1 < sh1.size(); ++i1) {
        const auto bf1 = s2bf1[i1];
        const auto n1 = sh1[i1].size();
        for (std::size_t i2 = 0; i2 < sh2.size(); ++i2) {
            const auto bf2 = s2bf2[i2];
            const auto n2 = sh2[i2].size();
            engine.compute(sh1[i1], sh2[i2]);
            const double* block = buf[0];
            if (block == nullptr) continue;
            for (std::size_t a = 0; a < n1; ++a) {
                for (std::size_t b = 0; b < n2; ++b) {
                    M(bf1 + a, bf2 + b) = block[a * n2 + b];
                }
            }
        }
    }
    return M;
}

Eigen::MatrixXd compute_kinetic(const BasisSet& basis) {
    return compute_1e_matrix(basis, libint2::Operator::kinetic, nullptr);
}

Eigen::MatrixXd compute_nuclear(const BasisSet& basis, const Molecule& mol) {
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        q.emplace_back(static_cast<double>(a.Z), a.xyz);
    }
    return compute_1e_matrix(basis, libint2::Operator::nuclear, &q);
}

DipoleIntegrals compute_dipole(const BasisSet& basis,
                               const std::array<double, 3>& origin) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto nbf = basis.nbasis();

    DipoleIntegrals out;
    out.x = Eigen::MatrixXd::Zero(nbf, nbf);
    out.y = Eigen::MatrixXd::Zero(nbf, nbf);
    out.z = Eigen::MatrixXd::Zero(nbf, nbf);

    // emultipole1 returns 4 result buffers per shell pair in order
    // (overlap, x, y, z). We drop the overlap — the caller already has
    // it via compute_overlap — and fold the three dipole components into
    // the output matrices.
    //
    // Engine-per-thread pool, same pattern as the other 1e operators.
    // Per-shell-pair scatter targets are disjoint across (s1, s2)
    // values, so different threads' writes never collide.
    libint2::Engine prototype(libint2::Operator::emultipole1,
                              shells.max_nprim(), shells.max_l(), 0);
    prototype.set_params(origin);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();

            engine.compute(shells[s1], shells[s2]);
            // buf[0] = overlap, buf[1..3] = ⟨μ | x-O_x | ν⟩ etc.
            const double* block_x = buf[1];
            const double* block_y = buf[2];
            const double* block_z = buf[3];
            if (!block_x) continue;

            for (std::size_t i = 0; i < n1; ++i) {
                for (std::size_t j = 0; j < n2; ++j) {
                    const double vx = block_x[i * n2 + j];
                    const double vy = block_y[i * n2 + j];
                    const double vz = block_z[i * n2 + j];
                    out.x(bf1 + i, bf2 + j) = vx;
                    out.y(bf1 + i, bf2 + j) = vy;
                    out.z(bf1 + i, bf2 + j) = vz;
                    if (s1 != s2) {
                        // ⟨μ|x-O|ν⟩ = ⟨ν|x-O|μ⟩ (real, symmetric operator).
                        out.x(bf2 + j, bf1 + i) = vx;
                        out.y(bf2 + j, bf1 + i) = vy;
                        out.z(bf2 + j, bf1 + i) = vz;
                    }
                }
            }
        }
    }
    return out;
}

Eri4D compute_eri(const BasisSet& basis) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto nbf = basis.nbasis();

    Eri4D eri;
    eri.n = nbf;
    eri.data.assign(nbf * nbf * nbf * nbf, 0.0);

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(),
                              shells.max_l(),
                              0);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    // Restrict the shell-quartet loop to the unique 1/8 dictated by the
    // permutational symmetry of (mu nu | lambda sigma):
    //   s1 >= s2, s3 >= s4, and (s1, s2) >= (s3, s4) in dictionary order.
    // Each computed block is scattered into all 8 equivalent positions.
    //
    // Parallel safety: each unique (s1, s2, s3, s4) quartet produces 8
    // scattered writes to eri(·,·,·,·). Because the restricted loop
    // produces each unique quartet exactly once (no two (s1, s2, s3, s4)
    // tuples from different threads give the same set of 8 permutations),
    // the scattered writes are disjoint across threads. No atomics or
    // per-thread accumulators are needed.
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            for (int s3 = 0; s3 <= s1; ++s3) {
                const auto bf3 = shell2bf[s3];
                const auto n3 = shells[s3].size();
                const auto s4_max = (s3 == s1) ? s2 : s3;
                for (int s4 = 0; s4 <= s4_max; ++s4) {
                    const auto bf4 = shell2bf[s4];
                    const auto n4 = shells[s4].size();

                    engine.compute(shells[s1], shells[s2],
                                   shells[s3], shells[s4]);
                    const double* block = buf[0];
                    if (!block) continue;  // screened to zero

                    for (std::size_t i = 0; i < n1; ++i) {
                        const auto mu = bf1 + i;
                        for (std::size_t j = 0; j < n2; ++j) {
                            const auto nu = bf2 + j;
                            for (std::size_t k = 0; k < n3; ++k) {
                                const auto lam = bf3 + k;
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const auto sig = bf4 + l;
                                    const double v = block[
                                        ((i * n2 + j) * n3 + k) * n4 + l];

                                    // 8-fold permutational symmetry of
                                    // (mu nu | lambda sigma):
                                    eri(mu,  nu,  lam, sig) = v;
                                    eri(nu,  mu,  lam, sig) = v;
                                    eri(mu,  nu,  sig, lam) = v;
                                    eri(nu,  mu,  sig, lam) = v;
                                    eri(lam, sig, mu,  nu ) = v;
                                    eri(sig, lam, mu,  nu ) = v;
                                    eri(lam, sig, nu,  mu ) = v;
                                    eri(sig, lam, nu,  mu ) = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return eri;
}

Eigen::MatrixXd eri_mo_pair_transform(const Eri4D& eri,
                                      const Eigen::MatrixXd& C_bra,
                                      const Eigen::MatrixXd& C_ket) {
    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>;
    const auto nao = static_cast<Eigen::Index>(eri.n);
    if (C_bra.rows() != nao || C_ket.rows() != nao)
        throw std::invalid_argument(
            "eri_mo_pair_transform: MO coefficient row count must match the "
            "AO dimension of the ERI tensor");
    if (C_bra.cols() < 1 || C_ket.cols() < 1)
        throw std::invalid_argument(
            "eri_mo_pair_transform: MO coefficient matrices must be non-empty");
    const Eigen::Index nb = C_bra.cols();
    const Eigen::Index nk = C_ket.cols();

    // Ket-pair half transform: H(mu*nao + nu, z*nk + w) = (mu nu | z w).
    // eri.data is row-major over (mu, nu, lambda, sigma), so each AO-pair
    // row is a contiguous (nao, nao) block over (lambda, sigma).
    RowMat H(nao * nao, nk * nk);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index r = 0; r < nao * nao; ++r) {
        Eigen::Map<const RowMat> M(eri.data.data() + r * nao * nao, nao, nao);
        const RowMat t = C_ket.transpose() * M * C_ket;  // (nk, nk)
        H.row(r) = Eigen::Map<const Eigen::RowVectorXd>(t.data(), nk * nk);
    }

    // Bra-pair half transform on H^T rows:
    // Wt(z*nk + w, x*nb + y) = (x y | z w).
    const RowMat Ht = H.transpose();
    RowMat Wt(nk * nk, nb * nb);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index r = 0; r < nk * nk; ++r) {
        Eigen::Map<const RowMat> M(Ht.data() + r * nao * nao, nao, nao);
        const RowMat t = C_bra.transpose() * M * C_bra;  // (nb, nb)
        Wt.row(r) = Eigen::Map<const Eigen::RowVectorXd>(t.data(), nb * nb);
    }
    return Wt.transpose();
}

}  // namespace vibeqc
