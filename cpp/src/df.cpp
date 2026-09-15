// Density-fitting integral kernels. See vibeqc/df.hpp for the math
// derivation and references; this translation unit is purely the
// libint2 driver loop for the two-centre Coulomb metric (P|Q) and the
// three-centre tensor (P|μν).

#include "vibeqc/df.hpp"
#include "vibeqc/build_config.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/thread_pool.hpp"
#include "vibeqc/diagnostics.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <libint2/atom.h>
#include <libint2/engine.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {

namespace {

// Fixed 24-column partitions have been measured to retain the monolithic
// GEMM/TRSM bytes on these two linked backends.  Other BLAS implementations
// and Eigen's generic kernels remain on the legacy serial calls until their
// packing boundaries have an equally explicit positive control.
constexpr bool kExactDfBlockingBackend =
#if defined(EIGEN_USE_BLAS)
    build_config::kBlasLibraries.find("openblas")
        != std::string_view::npos
    || build_config::kBlasLibraries.find("OpenBLAS")
        != std::string_view::npos
    || build_config::kBlasLibraries.find("Accelerate")
        != std::string_view::npos;
#else
    false;
#endif

}  // namespace

Eigen::MatrixXd compute_2c_eri(const BasisSet& aux) {
    ensure_libint_initialized();

    const auto& shells = aux.libint();
    const auto n = aux.nbasis();
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n),
                                              static_cast<Eigen::Index>(n));

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(),
                              shells.max_l(),
                              0);
    // BraKet::xs_xs: 2-centre integral (P|Q). The "s" slots are unit
    // shells supplied internally by libint2 — see engine.impl.h:169
    // for the variadic compute() padding.
    prototype.set(libint2::BraKet::xs_xs);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    // Outer P, restrict Q <= P; mirror to (Q, P) on the way out.
    // Each thread owns a contiguous row block of V (rows in
    // shell sP); the mirror writes target columns in another thread's
    // row range, but the (sP, sQ) → (sQ, sP) reflection is unique per
    // unordered pair, so no two threads write to the same matrix
    // element.
    #pragma omp parallel for schedule(dynamic)
    for (int sP = 0; sP < n_shells; ++sP) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bfP = shell2bf[sP];
        const auto nP = shells[sP].size();
        for (int sQ = 0; sQ <= sP; ++sQ) {
            const auto bfQ = shell2bf[sQ];
            const auto nQ = shells[sQ].size();

            engine.compute(shells[sP], shells[sQ]);
            const double* block = buf[0];
            if (!block) continue;  // screened to zero

            for (std::size_t i = 0; i < nP; ++i) {
                for (std::size_t j = 0; j < nQ; ++j) {
                    const double v = block[i * nQ + j];
                    V(static_cast<Eigen::Index>(bfP + i),
                      static_cast<Eigen::Index>(bfQ + j)) = v;
                    if (sP != sQ) {
                        V(static_cast<Eigen::Index>(bfQ + j),
                          static_cast<Eigen::Index>(bfP + i)) = v;
                    }
                }
            }
            // sP == sQ: the libint block is itself symmetric in (i, j)
            // (V is SPD), so the "full block" iteration above already
            // wrote every element of the diagonal block correctly —
            // no mirror needed.
        }
    }
    return V;
}

Eri3D compute_3c_eri(const BasisSet& orbital, const BasisSet& aux) {
    ensure_libint_initialized();

    const auto& orb_sh = orbital.libint();
    const auto& aux_sh = aux.libint();
    const auto n_orb = orbital.nbasis();
    const auto n_aux = aux.nbasis();

    Eri3D T;
    T.n_aux = n_aux;
    T.n_orb = n_orb;
    T.data.assign(n_aux * n_orb * n_orb, 0.0);

    libint2::Engine prototype(
        libint2::Operator::coulomb,
        std::max(orb_sh.max_nprim(), aux_sh.max_nprim()),
        std::max(orb_sh.max_l(), aux_sh.max_l()),
        0);
    // BraKet::xs_xx: aux on the bra (3 shells total: P, μ, ν).
    prototype.set(libint2::BraKet::xs_xx);
    auto engines = make_engine_pool(prototype);

    const auto orb_shell2bf = orb_sh.shell2bf();
    const auto aux_shell2bf = aux_sh.shell2bf();
    const int n_aux_shells = static_cast<int>(aux_sh.size());
    const int n_orb_shells = static_cast<int>(orb_sh.size());

    // The P-shell loop is the natural parallel axis: each thread writes
    // a disjoint slab of P-rows into the (n_aux, n_orb, n_orb) tensor.
    // Inner (sM >= sN) restriction with mirror exploits orbital-pair
    // symmetry (P|μν) = (P|νμ); the libint block returned for (sM, sN)
    // is mirrored to T(P, ν, μ) explicitly when sM != sN.
    #pragma omp parallel for schedule(dynamic)
    for (int sP = 0; sP < n_aux_shells; ++sP) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bfP = aux_shell2bf[sP];
        const auto nP = aux_sh[sP].size();
        for (int sM = 0; sM < n_orb_shells; ++sM) {
            const auto bfM = orb_shell2bf[sM];
            const auto nM = orb_sh[sM].size();
            for (int sN = 0; sN <= sM; ++sN) {
                const auto bfN = orb_shell2bf[sN];
                const auto nN = orb_sh[sN].size();

                engine.compute(aux_sh[sP], orb_sh[sM], orb_sh[sN]);
                const double* block = buf[0];
                if (!block) continue;

                for (std::size_t ip = 0; ip < nP; ++ip) {
                    const auto Pidx = bfP + ip;
                    for (std::size_t im = 0; im < nM; ++im) {
                        const auto mu = bfM + im;
                        for (std::size_t in = 0; in < nN; ++in) {
                            const auto nu = bfN + in;
                            const double v = block[(ip * nM + im) * nN + in];
                            T(Pidx, mu, nu) = v;
                            if (sM != sN) {
                                T(Pidx, nu, mu) = v;
                            }
                        }
                    }
                    // sM == sN: the libint block (im, in) is symmetric,
                    // and the inner loop scans the full nM × nN range,
                    // so T(P, μ, ν) and T(P, ν, μ) are both filled
                    // automatically (with equal values).
                }
            }
        }
    }
    return T;
}

namespace detail {

std::pair<Eigen::MatrixXd, int> solve_df_half_transform(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& T_flat,
    bool force_blocked) {
    if (L.rows() != L.cols() || L.rows() != T_flat.rows()) {
        throw std::invalid_argument(
            "solve_df_half_transform: L must be square and match the "
            "T_flat auxiliary dimension");
    }

    constexpr Eigen::Index kMinRhsBlockColumns = 256;
    // OpenBLAS's supported double-precision N micro-kernels have widths
    // 2/4/6/8. Starting every sub-call on their least-common multiple keeps
    // its packed-column groups identical to the legacy monolithic dtrsm;
    // Accelerate follows the same exact partition in the regression lane.
    constexpr Eigen::Index kRhsColumnAlignment = 24;
    constexpr long double kParallelMinFlops = 1.0e9L;
    // During this solve T, T_flat, and B_flat are simultaneously live. The
    // construction preflight charges four tensor extents, leaving one extent
    // for BLAS packing buffers; 32 concurrent calls keep that backend-owned
    // scratch bounded while still covering full-node molecular targets.
    constexpr int kMaxWorkers = 32;
    const Eigen::Index n_rhs = T_flat.cols();
    // At most one block per possible worker. Besides bounding OpenBLAS's
    // internal buffer-pool pressure, this makes OMP=1 and OMP=N call BLAS
    // with the same dimension-only partition; only block ownership changes.
    const Eigen::Index n_blocks = std::min<Eigen::Index>(
        kMaxWorkers,
        std::max<Eigen::Index>(1, n_rhs / kMinRhsBlockColumns));
    const Eigen::Index aligned_groups = n_blocks > 1
        ? n_rhs / kRhsColumnAlignment
        : 0;
    const Eigen::Index groups_per_block = n_blocks > 1
        ? aligned_groups / n_blocks
        : 0;
    const Eigen::Index extra_groups = n_blocks > 1
        ? aligned_groups % n_blocks
        : 0;
    const long double solve_flops =
        static_cast<long double>(L.rows())
        * static_cast<long double>(L.rows())
        * static_cast<long double>(n_rhs);
    // Generic Eigen and unmeasured BLAS backends retain the monolithic solve.
    constexpr bool kAllowParallelSolve = kExactDfBlockingBackend;
    const bool use_blocks = force_blocked
        || (kAllowParallelSolve && solve_flops >= kParallelMinFlops);
    const int workers = use_blocks && kAllowParallelSolve
        ? omp_workers_for(static_cast<std::size_t>(n_blocks), kMaxWorkers)
        : 1;

    if (!use_blocks) {
        return {
            L.triangularView<Eigen::Lower>().solve(T_flat),
            1,
        };
    }

    // RHS columns are mathematically independent. Keep the block boundaries
    // fixed across thread counts: OMP=1 and OMP=N therefore send identical
    // shapes to BLAS and differ only in which worker owns a block. B_flat is
    // allocated before the region, and every block is a disjoint output.
    Eigen::MatrixXd B_flat = T_flat;
    auto solve_block = [&](Eigen::Index block_index) {
        const Eigen::Index col0 = kRhsColumnAlignment * (
            block_index * groups_per_block
            + std::min(block_index, extra_groups));
        const Eigen::Index block_groups = groups_per_block
            + (block_index < extra_groups ? 1 : 0);
        const Eigen::Index col1 = block_index + 1 == n_blocks
            ? n_rhs
            : col0 + kRhsColumnAlignment * block_groups;
        const Eigen::Index width = col1 - col0;
        auto rhs = B_flat.middleCols(col0, width);
        L.triangularView<Eigen::Lower>().solveInPlace(rhs);
    };

    if (workers == 1) {
        for (Eigen::Index block = 0; block < n_blocks; ++block) {
            solve_block(block);
        }
        return {std::move(B_flat), 1};
    }

    int actual_workers = 1;
    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        {
            actual_workers = omp_team_threads();
        }
        #pragma omp for schedule(static)
        for (Eigen::Index block = 0; block < n_blocks; ++block) {
            solve_block(block);
        }
    }
    return {std::move(B_flat), actual_workers};
}

}  // namespace detail

// -----------------------------------------------------------------------------
// DensityFitting class — the in-language SCF-driver consumer.
// -----------------------------------------------------------------------------

DensityFitting::DensityFitting(const BasisSet& orbital, const BasisSet& aux) {
    orbital_basis_ = &orbital;
    aux_basis_     = &aux;
    n_orb_ = orbital.nbasis();
    n_aux_ = aux.nbasis();

    VIBEQC_DIAG("df", vibeqc::DiagLevel::VERBOSE,
        "n_orb=%zu  n_aux=%zu  ratio=%.2f",
        n_orb_, n_aux_, static_cast<double>(n_aux_) / static_cast<double>(n_orb_));

    // 1. Two-centre Coulomb metric V_PQ = (P|Q).
    Eigen::MatrixXd V = compute_2c_eri(aux);

    // 2. Cholesky V = L L^T. Eigen's LLT signals failure via info().
    Eigen::LLT<Eigen::MatrixXd> llt(V);
    if (llt.info() != Eigen::Success) {
        // Diagnose: smallest / largest eigenvalues, so the user sees
        // the actual problem (negative eig vs zero eig vs huge cond).
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
        const double min_eig = es.eigenvalues()[0];
        const double max_eig = es.eigenvalues()[V.rows() - 1];
        throw std::runtime_error(
            "DensityFitting: 2-centre Coulomb metric V = (P|Q) is not "
            "positive-definite (min eig = " + std::to_string(min_eig) +
            ", max eig = " + std::to_string(max_eig) + "). "
            "Either the auxiliary basis is degenerate or the geometry "
            "has duplicated atomic centres. Pick a different aux basis "
            "or fix the geometry.");
    }
    L_ = llt.matrixL();

    // 3. Three-centre tensor T_{P, μν} = (P | μν).
    Eri3D T = compute_3c_eri(orbital, aux);

    // 4. Half-transform: B^P_{μν} = Σ_Q [L^{-1}]_{PQ} T_{Q, μν}.
    // Equivalent to one triangular solve over the P-axis. Pack T as a
    // (n_aux, n_orb²) row-major matrix, solve L · B = T (in place over
    // rows), unpack into the per-P block list.
    T_flat_.resize(static_cast<Eigen::Index>(n_aux_),
                   static_cast<Eigen::Index>(n_orb_ * n_orb_));
    constexpr Eigen::Index kPackColumnBlock = 32;
    constexpr long double kParallelCopyMinTerms = 1.0e7L;
    constexpr int kCopyMaxWorkers = 64;
    const Eigen::Index n_aux = static_cast<Eigen::Index>(n_aux_);
    const Eigen::Index n_orb = static_cast<Eigen::Index>(n_orb_);
    const Eigen::Index n_pair = n_orb * n_orb;
    const Eigen::Index n_pack_blocks =
        (n_pair + kPackColumnBlock - 1) / kPackColumnBlock;
    const long double copy_terms =
        static_cast<long double>(n_aux_) * static_cast<long double>(n_orb_)
        * static_cast<long double>(n_orb_);
    const int pack_workers = copy_terms >= kParallelCopyMinTerms
        ? omp_workers_for(static_cast<std::size_t>(n_pack_blocks),
                          kCopyMaxWorkers)
        : 1;
    const double* t_data = T.data.data();
    double* t_flat_data = T_flat_.data();
    auto pack_block = [&](Eigen::Index block) {
        const Eigen::Index col0 = block * kPackColumnBlock;
        const Eigen::Index col1 = std::min(
            n_pair, col0 + kPackColumnBlock);
        for (Eigen::Index P = 0; P < n_aux; ++P) {
            const double* source = t_data + P * n_pair;
            for (Eigen::Index col = col0; col < col1; ++col) {
                t_flat_data[col * n_aux + P] = source[col];
            }
        }
    };
    if (pack_workers == 1) {
        for (Eigen::Index block = 0; block < n_pack_blocks; ++block) {
            pack_block(block);
        }
    } else {
        int actual_workers = 1;
        #pragma omp parallel num_threads(pack_workers)
        {
            #pragma omp single
            {
                actual_workers = omp_team_threads();
            }
            #pragma omp for schedule(static)
            for (Eigen::Index block = 0; block < n_pack_blocks; ++block) {
                pack_block(block);
            }
        }
        last_df_pack_workers_used_ = actual_workers;
    }
    auto transformed = detail::solve_df_half_transform(L_, T_flat_);
    Eigen::MatrixXd B_flat = std::move(transformed.first);
    last_df_transform_workers_used_ = transformed.second;

    // Unpack into B_per_P_ for per-P access in the K builders.
    B_per_P_.resize(n_aux_);
    for (std::size_t P = 0; P < n_aux_; ++P) {
        B_per_P_[P].resize(n_orb, n_orb);
    }
    std::vector<double*> b_per_p_data;
    b_per_p_data.reserve(n_aux_);
    for (auto& Bp : B_per_P_) {
        b_per_p_data.push_back(Bp.data());
    }
    const double* b_flat_data = B_flat.data();
    auto unpack_column = [&](Eigen::Index nu) {
        for (Eigen::Index mu = 0; mu < n_orb; ++mu) {
            const double* source =
                b_flat_data + (mu * n_orb + nu) * n_aux;
            const Eigen::Index destination = mu + nu * n_orb;
            for (Eigen::Index P = 0; P < n_aux; ++P) {
                b_per_p_data[static_cast<std::size_t>(P)][destination] =
                    source[P];
            }
        }
    };
    const int unpack_workers = copy_terms >= kParallelCopyMinTerms
        ? omp_workers_for(n_orb_, kCopyMaxWorkers)
        : 1;
    if (unpack_workers == 1) {
        for (Eigen::Index nu = 0; nu < n_orb; ++nu) {
            unpack_column(nu);
        }
    } else {
        int actual_workers = 1;
        #pragma omp parallel num_threads(unpack_workers)
        {
            #pragma omp single
            {
                actual_workers = omp_team_threads();
            }
            #pragma omp for schedule(static)
            for (Eigen::Index nu = 0; nu < n_orb; ++nu) {
                unpack_column(nu);
            }
        }
        last_df_unpack_workers_used_ = actual_workers;
    }
}

Eigen::MatrixXd DensityFitting::build_J(const Eigen::MatrixXd& D) const {
    // γ_P = sum_μν B^P_μν D_μν = (B^P : D) — a Frobenius inner product.
    Eigen::VectorXd gamma(static_cast<Eigen::Index>(n_aux_));
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_orb_),
        static_cast<Eigen::Index>(n_orb_));

    // At small dimensions one OpenMP team costs more than the RI-J
    // contractions. Large molecular jobs are different: both contractions
    // contain n_aux * n_orb^2 independent scalar terms and were previously
    // serial even when the SCF owned a full-node allocation.
    constexpr long double kParallelMinTerms = 1.0e7L;
    const long double contraction_terms =
        static_cast<long double>(n_aux_)
        * static_cast<long double>(n_orb_)
        * static_cast<long double>(n_orb_);
    const std::size_t work_items = std::min(n_aux_, n_orb_);
    const int workers = contraction_terms >= kParallelMinTerms
        ? omp_workers_for(work_items, omp_max_threads())
        : 1;
    last_j_workers_used_.store(1, std::memory_order_relaxed);

    if (workers == 1) {
        for (std::size_t P = 0; P < n_aux_; ++P) {
            gamma(static_cast<Eigen::Index>(P)) =
                (B_per_P_[P].array() * D.array()).sum();
        }
        for (std::size_t P = 0; P < n_aux_; ++P) {
            J.noalias() +=
                gamma(static_cast<Eigen::Index>(P)) * B_per_P_[P];
        }
        return J;
    }

    int actual_workers = 1;
    const Eigen::Index n_aux = static_cast<Eigen::Index>(n_aux_);
    const Eigen::Index n_orb = static_cast<Eigen::Index>(n_orb_);
    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        {
            actual_workers = omp_team_threads();
        }

        // Each auxiliary component owns one complete Frobenius reduction.
        // Its μν association is therefore identical to the serial path.
        #pragma omp for schedule(static)
        for (Eigen::Index P = 0; P < n_aux; ++P) {
            gamma(P) =
                (B_per_P_[static_cast<std::size_t>(P)].array()
                 * D.array()).sum();
        }

        // Columns are disjoint outputs. Every J(μ,ν) still visits P in
        // ascending order, so this changes ownership but not the scientific
        // reduction order or any individual multiply/add sequence.
        #pragma omp for schedule(static)
        for (Eigen::Index nu = 0; nu < n_orb; ++nu) {
            for (Eigen::Index P = 0; P < n_aux; ++P) {
                J.col(nu).noalias() += gamma(P)
                    * B_per_P_[static_cast<std::size_t>(P)].col(nu);
            }
        }
    }
    last_j_workers_used_.store(actual_workers, std::memory_order_relaxed);
    return J;
}

Eigen::MatrixXd DensityFitting::build_J_screened(
    const Eigen::MatrixXd& D, double screen_tol) const {
    const auto& orb_sh = orbital_basis_->libint();
    const auto sh2bf = orb_sh.shell2bf();
    const int n_sh = static_cast<int>(orb_sh.size());
    const Eigen::Index n_orb_idx = static_cast<Eigen::Index>(n_orb_);

    // Per-shell-pair density max |D_μν| for screening.  Shell pairs
    // with max |D| below the tolerance contribute < tol · B_max to γ,
    // where B_max = max_{μ,ν,P} |B^P_μν| ≈ O(1).  The default
    // screen_tol = 1e-10 keeps the J-build bit-identical to build_J
    // while skipping 30−70 % of shell-pair work on extended systems.
    std::vector<double> D_sh_max(n_sh * n_sh, 0.0);
    for (int ish = 0; ish < n_sh; ++ish) {
        const int bfi = sh2bf[ish];
        const int ni  = static_cast<int>(orb_sh[ish].size());
        for (int jsh = 0; jsh < n_sh; ++jsh) {
            const int bfj = sh2bf[jsh];
            const int nj  = static_cast<int>(orb_sh[jsh].size());
            double mx = 0.0;
            for (int i = 0; i < ni; ++i)
                for (int j = 0; j < nj; ++j)
                    mx = std::max(mx, std::abs(D(bfi + i, bfj + j)));
            D_sh_max[ish * n_sh + jsh] = mx;
        }
    }

    // Screened γ and J accumulation.  Uses the same B-per-P loop as
    // build_J but with an inner shell-pair skip when D_sh < tol.
    // γ is indexed by the auxiliary index P below, so it must be sized
    // n_aux_ and not n_orb_ (build_J does this correctly above). An
    // auxiliary basis is routinely 3-4x the orbital basis, so sizing it
    // n_orb_ made every P >= n_orb_ a heap write past the end.
    Eigen::VectorXd gamma(static_cast<Eigen::Index>(n_aux_));
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_orb_idx, n_orb_idx);
    for (std::size_t P = 0; P < n_aux_; ++P) {
        const auto& Bp = B_per_P_[P];
        double gP = 0.0;
        for (int ish = 0; ish < n_sh; ++ish) {
            const int bfi = sh2bf[ish];
            const int ni  = static_cast<int>(orb_sh[ish].size());
            for (int jsh = 0; jsh < n_sh; ++jsh) {
                if (D_sh_max[ish * n_sh + jsh] < screen_tol) continue;
                const int bfj = sh2bf[jsh];
                const int nj  = static_cast<int>(orb_sh[jsh].size());
                for (int i = 0; i < ni; ++i)
                    for (int j = 0; j < nj; ++j) {
                        const double v = Bp(bfi + i, bfj + j);
                        gP += v * D(bfi + i, bfj + j);
                    }
            }
        }
        gamma(static_cast<Eigen::Index>(P)) = gP;
        if (std::abs(gP) > screen_tol)
            J.noalias() += gP * Bp;
    }
    return J;
}

Eigen::MatrixXd DensityFitting::build_K_density(
    const Eigen::MatrixXd& D) const {
    // K_μν = Σ_λσ D_λσ (μλ|νσ) ≈ Σ_P (B^P · D · B^P)_{μν}.
    // The triple product B^P D B^P is symmetric whenever both factors are
    // (matrix algebra: (ABA)^T = A^T B^T A^T = A B A when A and B are
    // symmetric), so K is symmetric whenever D is.
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_orb_),
        static_cast<Eigen::Index>(n_orb_));
    constexpr long double kParallelMinFlops = 3.0e10L;
    constexpr int kMaxWorkers = 64;
    constexpr Eigen::Index kOutputColumnBlock = 24;
    const long double single_flops = 4.0L
        * static_cast<long double>(n_aux_)
        * static_cast<long double>(n_orb_)
        * static_cast<long double>(n_orb_)
        * static_cast<long double>(n_orb_);
    const Eigen::Index n_orb = static_cast<Eigen::Index>(n_orb_);
    const Eigen::Index n_column_blocks =
        (n_orb + kOutputColumnBlock - 1) / kOutputColumnBlock;
    const int workers = kExactDfBlockingBackend
            && single_flops >= kParallelMinFlops
        ? omp_workers_for(
              std::max<std::size_t>(
                  n_aux_, static_cast<std::size_t>(n_column_blocks)),
              kMaxWorkers)
        : 1;
    last_k_workers_used_.store(1, std::memory_order_relaxed);
    if (workers == 1) {
        for (std::size_t P = 0; P < n_aux_; ++P) {
            K.noalias() += B_per_P_[P] * D * B_per_P_[P];
        }
        return K;
    }

    // Keep each legacy full-shape X_P = B_P D call intact. K columns are
    // independent, so fixed 24-column blocks can accumulate P in the exact
    // legacy order while separate workers own disjoint outputs. The boundary
    // is aligned to the supported BLAS N micro-kernels, preserving their
    // monolithic packed-column groups on OpenBLAS and Accelerate.
    const std::size_t p_per_wave = std::min<std::size_t>(
        n_aux_, static_cast<std::size_t>(kMaxWorkers));
    std::vector<Eigen::MatrixXd> X;
    X.reserve(p_per_wave);
    for (std::size_t local = 0; local < p_per_wave; ++local) {
        X.emplace_back(n_orb, n_orb);
    }

    int actual_workers = 1;
    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        {
            actual_workers = omp_team_threads();
        }
        for (std::size_t wave = 0; wave < n_aux_; wave += p_per_wave) {
            const std::size_t wave_size =
                std::min(p_per_wave, n_aux_ - wave);
            #pragma omp for schedule(static, 1)
            for (Eigen::Index local = 0;
                 local < static_cast<Eigen::Index>(wave_size); ++local) {
                const auto P = wave + static_cast<std::size_t>(local);
                X[static_cast<std::size_t>(local)].noalias() =
                    B_per_P_[P] * D;
            }

            #pragma omp for schedule(static, 1)
            for (Eigen::Index block = 0;
                 block < n_column_blocks; ++block) {
                const Eigen::Index col0 = block * kOutputColumnBlock;
                const Eigen::Index width =
                    std::min(kOutputColumnBlock, n_orb - col0);
                for (std::size_t local = 0; local < wave_size; ++local) {
                    K.middleCols(col0, width).noalias() +=
                        X[local]
                        * B_per_P_[wave + local].middleCols(col0, width);
                }
            }
        }
    }
    last_k_workers_used_.store(actual_workers, std::memory_order_relaxed);
    return K;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
DensityFitting::build_K_density_pair(
    const Eigen::MatrixXd& D_alpha,
    const Eigen::MatrixXd& D_beta) const {
    // The legacy K path materialises X_P = B_P D and then executes
    // K += X_P B_P with beta=1, in ascending P.  Keep those two operations
    // and the complete beta=1 chain unchanged for each spin.  Only the
    // independent X_P products and the independent alpha/beta chains run
    // concurrently.
    constexpr long double kParallelMinFlops = 6.0e10L;
    constexpr int kMaxWorkers = 64;
    const long double pair_flops = 8.0L
        * static_cast<long double>(n_aux_)
        * static_cast<long double>(n_orb_)
        * static_cast<long double>(n_orb_)
        * static_cast<long double>(n_orb_);
    const int workers = kExactDfBlockingBackend
            && pair_flops >= kParallelMinFlops
        ? omp_workers_for(2 * n_aux_, kMaxWorkers)
        : 1;
    last_k_pair_workers_used_.store(1, std::memory_order_relaxed);
    if (workers < 2) {
        return {build_K_density(D_alpha), build_K_density(D_beta)};
    }

    Eigen::MatrixXd K_alpha = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_orb_),
        static_cast<Eigen::Index>(n_orb_));
    Eigen::MatrixXd K_beta = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_orb_),
        static_cast<Eigen::Index>(n_orb_));

    // At most min(kMaxWorkers, 2*n_aux) full matrices are live, so this
    // scratch fits inside the two tensor extents already charged above the
    // resident DF layout by the construction-peak memory preflight.
    // Allocate it serially so allocation failure propagates normally instead
    // of escaping an OpenMP region, and reuse one team and the same matrices
    // across every wave.
    const std::size_t p_per_wave = std::min<std::size_t>(
        n_aux_, static_cast<std::size_t>(kMaxWorkers / 2));
    const Eigen::Index n_orb = static_cast<Eigen::Index>(n_orb_);
    std::vector<Eigen::MatrixXd> X_alpha;
    std::vector<Eigen::MatrixXd> X_beta;
    X_alpha.reserve(p_per_wave);
    X_beta.reserve(p_per_wave);
    for (std::size_t local = 0; local < p_per_wave; ++local) {
        X_alpha.emplace_back(n_orb, n_orb);
        X_beta.emplace_back(n_orb, n_orb);
    }

    int actual_workers = 1;
    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        {
            actual_workers = omp_team_threads();
        }
        for (std::size_t wave = 0; wave < n_aux_; wave += p_per_wave) {
            const std::size_t wave_size =
                std::min(p_per_wave, n_aux_ - wave);
            const std::size_t tasks = 2 * wave_size;
            #pragma omp for schedule(static, 1)
            for (Eigen::Index task = 0;
                 task < static_cast<Eigen::Index>(tasks); ++task) {
                const auto slot = static_cast<std::size_t>(task);
                const bool beta = slot >= wave_size;
                const std::size_t local = beta ? slot - wave_size : slot;
                const auto P = wave + local;
                if (beta) {
                    X_beta[local].noalias() = B_per_P_[P] * D_beta;
                } else {
                    X_alpha[local].noalias() = B_per_P_[P] * D_alpha;
                }
            }

            // Each spin/column block owns disjoint output. Within a block the
            // auxiliary index still advances exactly as in the legacy
            // beta=1 chain; fixed BLAS-aligned boundaries preserve the full
            // call's packed-column groups.
            constexpr Eigen::Index kOutputColumnBlock = 24;
            const Eigen::Index n_column_blocks =
                (n_orb + kOutputColumnBlock - 1) / kOutputColumnBlock;
            const Eigen::Index accumulation_tasks = 2 * n_column_blocks;
            #pragma omp for schedule(static, 1)
            for (Eigen::Index task = 0;
                 task < accumulation_tasks; ++task) {
                const bool beta = task >= n_column_blocks;
                const Eigen::Index block = beta
                    ? task - n_column_blocks
                    : task;
                const Eigen::Index col0 = block * kOutputColumnBlock;
                const Eigen::Index width =
                    std::min(kOutputColumnBlock, n_orb - col0);
                if (!beta) {
                    for (std::size_t local = 0; local < wave_size; ++local) {
                        K_alpha.middleCols(col0, width).noalias() +=
                            X_alpha[local]
                            * B_per_P_[wave + local].middleCols(col0, width);
                    }
                } else {
                    for (std::size_t local = 0; local < wave_size; ++local) {
                        K_beta.middleCols(col0, width).noalias() +=
                            X_beta[local]
                            * B_per_P_[wave + local].middleCols(col0, width);
                    }
                }
            }
        }
    }
    last_k_pair_workers_used_.store(actual_workers,
                                    std::memory_order_relaxed);
    return {std::move(K_alpha), std::move(K_beta)};
}

Eigen::MatrixXd DensityFitting::build_K_mo(
    const Eigen::MatrixXd& C_occ) const {
    // B_occ^P = B^P · C_occ          (n_orb × n_occ)
    // K       = Σ_P B_occ^P · (B_occ^P)^T
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_orb_),
        static_cast<Eigen::Index>(n_orb_));
    for (std::size_t P = 0; P < n_aux_; ++P) {
        const Eigen::MatrixXd B_occ_P = B_per_P_[P] * C_occ;
        K.noalias() += B_occ_P * B_occ_P.transpose();
    }
    return K;
}

Eigen::MatrixXd DensityFitting::build_J_streaming(
    const Eigen::MatrixXd& D, int block_size) const {
    const Eigen::Index n_orb_idx = static_cast<Eigen::Index>(n_orb_);
    const Eigen::Index n_aux_idx = static_cast<Eigen::Index>(n_aux_);
    const Eigen::Index n2 = n_orb_idx * n_orb_idx;

    // Step 1: γ_P = (B^P : D) via blocked B-tensor solve.
    // γ̃ = L^{-T} · γ  so J = Σ_Q γ̃_Q · T_Q later.
    Eigen::VectorXd gamma = Eigen::VectorXd::Zero(n_aux_idx);
    Eigen::MatrixXd B_block;  // reused per block
    for (Eigen::Index p0 = 0; p0 < n_aux_idx; p0 += block_size) {
        const Eigen::Index p1 = std::min(p0 + block_size, n_aux_idx);
        const Eigen::Index nb = p1 - p0;
        // Extract T block and solve L · B_block = T_block.
        B_block = L_.triangularView<Eigen::Lower>()
            .solve(T_flat_.middleRows(p0, nb));
        // Contract with D.
        for (Eigen::Index p = 0; p < nb; ++p) {
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                Eigen::Dynamic, Eigen::RowMajor>>
                Bp(B_block.data() + p * n2, n_orb_idx, n_orb_idx);
            gamma(p0 + p) = (Bp.array() * D.array()).sum();
        }
    }

    // Step 2: J = Σ_Q γ̃_Q · T_Q where γ̃ = L^{-T} · γ.
    Eigen::VectorXd gamma_tilde =
        L_.transpose().triangularView<Eigen::Upper>().solve(gamma);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_orb_idx, n_orb_idx);
    for (Eigen::Index Q = 0; Q < n_aux_idx; ++Q) {
        const double gQ = gamma_tilde(Q);
        if (std::abs(gQ) < 1e-15) continue;
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
            Eigen::Dynamic, Eigen::RowMajor>>
            TQ(T_flat_.data() + Q * n2, n_orb_idx, n_orb_idx);
        J.noalias() += gQ * TQ;
    }
    return J;
}

Eigen::MatrixXd DensityFitting::build_K_streaming(
    const Eigen::MatrixXd& D, int block_size) const {
    const Eigen::Index n_orb_idx = static_cast<Eigen::Index>(n_orb_);
    const Eigen::Index n_aux_idx = static_cast<Eigen::Index>(n_aux_);
    const Eigen::Index n2 = n_orb_idx * n_orb_idx;

    // K = Σ_P B^P · D · B^P via blocked B-tensor solve.
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_orb_idx, n_orb_idx);
    Eigen::MatrixXd B_block;
    for (Eigen::Index p0 = 0; p0 < n_aux_idx; p0 += block_size) {
        const Eigen::Index p1 = std::min(p0 + block_size, n_aux_idx);
        const Eigen::Index nb = p1 - p0;
        B_block = L_.triangularView<Eigen::Lower>()
            .solve(T_flat_.middleRows(p0, nb));
        for (Eigen::Index p = 0; p < nb; ++p) {
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                Eigen::Dynamic, Eigen::RowMajor>>
                Bp(B_block.data() + p * n2, n_orb_idx, n_orb_idx);
            K.noalias() += Bp * D * Bp;
        }
    }
    return K;
}

Eigen::MatrixXd DensityFitting::build_g_rhf(
    const Eigen::MatrixXd& D) const {
    // F = Hcore + G with G = J - K/2 in the closed-shell convention
    // D = 2 C_occ C_occ^T (so trace D·S = n_electrons).
    return build_J(D) - 0.5 * build_K_density(D);
}

Eigen::MatrixXd DensityFitting::project_through_B(
    const Eigen::MatrixXd& M) const {
    // result = Σ_P B^P · M · B^P
    // Used by robust COSX to compute K_XvQ = project(K_XvX).
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_orb_),
        static_cast<Eigen::Index>(n_orb_));
    for (std::size_t P = 0; P < n_aux_; ++P)
        R.noalias() += B_per_P_[P] * M * B_per_P_[P];
    return R;
}

Eigen::MatrixXd DensityFitting::compute_B_metric() const {
    // S_PQ = (B^P : B^Q) = Σ_μν B^P_μν B^Q_μν
    //      = (L^{-1} · W · L^{-T})_{PQ}
    // where W_RS = (T_R : T_S) = T_flat row R · T_flat row S^T.
    const Eigen::Index n = static_cast<Eigen::Index>(n_aux_);
    const Eigen::Index n2 = static_cast<Eigen::Index>(n_orb_ * n_orb_);
    Eigen::MatrixXd W = T_flat_ * T_flat_.transpose();  // (n_aux, n_aux)
    // S = L^{-1} · W · L^{-T}
    Eigen::MatrixXd L_inv_W = L_.triangularView<Eigen::Lower>().solve(W);
    // L^{-T} = (L^{-1})^T; solve L^T · X = L_inv_W^T → X^T = L^{-1} · W · L^{-T}
    Eigen::MatrixXd S = L_.transpose()
        .triangularView<Eigen::Upper>()
        .solve(L_inv_W.transpose())
        .transpose();
    return S;
}

Eigen::VectorXd DensityFitting::contract_B_with(
    const Eigen::MatrixXd& M) const {
    // α = L^{-1} · T_flat · vec(M)
    const Eigen::Index n_orb_idx = static_cast<Eigen::Index>(n_orb_);
    const Eigen::Index n_aux_idx = static_cast<Eigen::Index>(n_aux_);
    const Eigen::Index n2 = n_orb_idx * n_orb_idx;
    Eigen::VectorXd Tm(n_aux_idx);
    Eigen::Map<const Eigen::VectorXd> M_vec(M.data(), n2);
    Tm = T_flat_ * M_vec;
    return L_.triangularView<Eigen::Lower>().solve(Tm);
}

// -----------------------------------------------------------------------------
// Derivative kernels (libint2 deriv_order=1) — feed the DF analytic
// gradient assembly. Each kernel computes the *contracted* gradient
// contribution (per-atom, n_atoms × 3) directly, avoiding the full
// (n_atoms · 3, n_aux, ...) shell-derivative tensors.
// -----------------------------------------------------------------------------

namespace {

// Helper: shell index → atom index (0-based). Mirrors the private
// shell_to_atom() in gradient.cpp; duplicated here to avoid coupling
// gradient.cpp's internals into df.cpp.
std::vector<long> shell_to_atom_index(
    const BasisSet& basis, const Molecule& mol) {
    std::vector<libint2::Atom> atoms;
    atoms.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        libint2::Atom la;
        la.atomic_number = a.Z;
        la.x = a.xyz[0];
        la.y = a.xyz[1];
        la.z = a.xyz[2];
        atoms.push_back(la);
    }
    return basis.libint().shell2atom(atoms);
}

}  // namespace

Eigen::MatrixXd compute_2c_eri_gradient_weighted(
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::MatrixXd& omega) {
    ensure_libint_initialized();

    const auto& shells = aux.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom_index(aux, mol);
    const std::size_t N = mol.atoms().size();
    const std::size_t n_aux = aux.nbasis();

    if (static_cast<std::size_t>(omega.rows()) != n_aux
        || static_cast<std::size_t>(omega.cols()) != n_aux) {
        throw std::invalid_argument(
            "compute_2c_eri_gradient_weighted: omega shape does not "
            "match the auxiliary basis dimension");
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(N), 3);

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(),
                              shells.max_l(),
                              1 /*deriv_order*/);
    prototype.set(libint2::BraKet::xs_xs);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    const int n_shells = static_cast<int>(shells.size());

    // 6 derivative buffers per (P, Q) shell pair: 3 per shell × 2 centres.
    #pragma omp parallel for schedule(dynamic)
    for (int sP = 0; sP < n_shells; ++sP) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const auto bfP = shell2bf[sP];
        const auto nP = shells[sP].size();
        const long atomP = s2a[sP];
        for (int sQ = 0; sQ < n_shells; ++sQ) {
            const auto bfQ = shell2bf[sQ];
            const auto nQ = shells[sQ].size();
            const long atomQ = s2a[sQ];

            engine.compute(shells[sP], shells[sQ]);
            const long centres[2] = {atomP, atomQ};
            for (int icenter = 0; icenter < 2; ++icenter) {
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < nP; ++i) {
                        for (std::size_t j = 0; j < nQ; ++j) {
                            acc += omega(
                                static_cast<Eigen::Index>(bfP + i),
                                static_cast<Eigen::Index>(bfQ + j))
                                * block[i * nQ + j];
                        }
                    }
                    grad_local(centres[icenter], d) += acc;
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

Eigen::MatrixXd compute_2c_eri_gradient_contribution(
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::VectorXd& gamma) {
    // Materialise γ γ^T (n_aux²) and forward to the matrix-form kernel.
    // n_aux² doubles is a small extra allocation at chemistry sizes
    // (~50 KB at n_aux=80; ~8 MB at n_aux=1000) compared to the
    // libint-derivative-call cost, so the convenience is worth it.
    const Eigen::MatrixXd omega = gamma * gamma.transpose();
    return compute_2c_eri_gradient_weighted(aux, mol, omega);
}

Eigen::MatrixXd compute_3c_eri_gradient_weighted(
    const BasisSet& orbital,
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& W) {
    ensure_libint_initialized();

    const auto& orb_sh = orbital.libint();
    const auto& aux_sh = aux.libint();
    const auto orb_shell2bf = orb_sh.shell2bf();
    const auto aux_shell2bf = aux_sh.shell2bf();
    const auto orb_s2a = shell_to_atom_index(orbital, mol);
    const auto aux_s2a = shell_to_atom_index(aux, mol);
    const std::size_t N = mol.atoms().size();
    const std::size_t n_orb = orbital.nbasis();
    const std::size_t n_aux = aux.nbasis();

    if (static_cast<std::size_t>(W.rows()) != n_aux
        || static_cast<std::size_t>(W.cols()) != n_orb * n_orb) {
        throw std::invalid_argument(
            "compute_3c_eri_gradient_weighted: W shape does not match "
            "(n_aux, n_orb * n_orb)");
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(N), 3);

    libint2::Engine prototype(
        libint2::Operator::coulomb,
        std::max(orb_sh.max_nprim(), aux_sh.max_nprim()),
        std::max(orb_sh.max_l(), aux_sh.max_l()),
        1 /*deriv_order*/);
    prototype.set(libint2::BraKet::xs_xx);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    const int n_aux_sh = static_cast<int>(aux_sh.size());
    const int n_orb_sh = static_cast<int>(orb_sh.size());

    // 9 derivative buffers per (P, μν) triplet: 3 per shell × 3 centres.
    //
    // **Iteration convention.** Iterate the **full** (sM, sN) cartesian
    // product of orbital shells with a **fresh libint engine per call**.
    // Previous variants (unique-pair sN ≤ sM + factor-2 doubling; or
    // full iteration with one engine per (sP, sM) group) leaked
    // ~5-15 mHa per heavy-atom component on systems with two adjacent
    // same-l heavy atoms (HCOOH and glycine with def2-tzvp /
    // def2-tzvp-jk; reproducer: ``scripts/df_3c_residual_probe.py``).
    // The factor-2 doubling assumes both shell-pair orderings give the
    // same per-atom derivative-buffer attribution, which is only true
    // if libint's per-call attribution is consistent across calls;
    // empirically it is not for ``xs_xx`` with ``deriv_order=1`` when
    // the engine is reused. Full iteration with one engine per
    // (sP, sM) group cuts the leak in half but does not eliminate it.
    //
    // Fresh engine per (sP, sM, sN) call costs an additional
    // engine-copy per inner-loop iter (~µs each on M-class hardware),
    // but the gradient kernel runs once per SCF (not per iter), so the
    // wall-clock impact is tolerable. The residual drops to the FD
    // truncation floor (~1e-8 Ha/bohr).
    //
    // **L-canonical reorder.** Kept for the mixed-l case (lM ≠ lN).
    // When lM < lN, swap so the higher-l shell is in libint's ket-1
    // slot — bypasses libint's ``swap_tket`` + ``DerivMapGenerator``
    // mis-routing for high-l mixed-l ket pairs (the v0.7.3
    // 3c-gradient bug). When lM == lN no swap is needed.
    //
    // **Atom routing.** After the reorder, ``centres[icenter]`` for
    // ``icenter ∈ {1, 2}`` is the atom of the shell *passed* to
    // libint at that position (not the original sM / sN). The W
    // indexing uses the canonical-order basis ranges, which by W's
    // (μ↔ν) symmetry picks up the same contracted contribution as
    // the original-order indexing would.
    #pragma omp parallel for schedule(dynamic)
    for (int sP = 0; sP < n_aux_sh; ++sP) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& grad_local = grad_tls[tid];

        const auto bfP = aux_shell2bf[sP];
        const auto nP = aux_sh[sP].size();
        const long atomP = aux_s2a[sP];

        for (int sM = 0; sM < n_orb_sh; ++sM) {
            for (int sN = 0; sN < n_orb_sh; ++sN) {
                // Fresh engine per call — eliminates the libint state
                // leak documented above.
                libint2::Engine engine = prototype;
                const auto& buf = engine.results();

                // L-canonical reorder of the ket pair so libint's
                // swap_tket = false. sM_canon is the shell vibe-qc
                // hands libint as the *first* ket; sN_canon is the
                // second.
                const int lM = orb_sh[sM].contr[0].l;
                const int lN = orb_sh[sN].contr[0].l;
                const bool swap = (lM < lN);
                const int sM_canon = swap ? sN : sM;
                const int sN_canon = swap ? sM : sN;

                const auto bfM_canon = orb_shell2bf[sM_canon];
                const auto nM_canon  = orb_sh[sM_canon].size();
                const auto bfN_canon = orb_shell2bf[sN_canon];
                const auto nN_canon  = orb_sh[sN_canon].size();
                const long atomM_canon = orb_s2a[sM_canon];
                const long atomN_canon = orb_s2a[sN_canon];

                engine.compute(aux_sh[sP],
                                orb_sh[sM_canon], orb_sh[sN_canon]);
                const long centres[3] = {
                    atomP, atomM_canon, atomN_canon};
                for (int icenter = 0; icenter < 3; ++icenter) {
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t ip = 0; ip < nP; ++ip) {
                            const auto Pidx = bfP + ip;
                            for (std::size_t im = 0; im < nM_canon; ++im) {
                                const auto mu = bfM_canon + im;
                                for (std::size_t in = 0; in < nN_canon; ++in) {
                                    const auto nu = bfN_canon + in;
                                    acc += W(
                                        static_cast<Eigen::Index>(Pidx),
                                        static_cast<Eigen::Index>(
                                            mu * n_orb + nu))
                                        * block[(ip * nM_canon + im)
                                                * nN_canon + in];
                                }
                            }
                        }
                        grad_local(centres[icenter], d) += acc;
                    }
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

Eigen::MatrixXd compute_3c_eri_gradient_contribution(
    const BasisSet& orbital,
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    const Eigen::VectorXd& gamma) {
    // Materialise W^P_{μν} = γ_P · D_{μν} (rank-1 over P axis) and
    // forward to the matrix-form kernel.
    const std::size_t n_orb = orbital.nbasis();
    const std::size_t n_aux = aux.nbasis();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        W(static_cast<Eigen::Index>(n_aux),
          static_cast<Eigen::Index>(n_orb * n_orb));
    for (std::size_t P = 0; P < n_aux; ++P) {
        const double gP = gamma(static_cast<Eigen::Index>(P));
        for (std::size_t mu = 0; mu < n_orb; ++mu) {
            for (std::size_t nu = 0; nu < n_orb; ++nu) {
                W(static_cast<Eigen::Index>(P),
                  static_cast<Eigen::Index>(mu * n_orb + nu)) =
                    gP * D(static_cast<Eigen::Index>(mu),
                           static_cast<Eigen::Index>(nu));
            }
        }
    }
    return compute_3c_eri_gradient_weighted(orbital, aux, mol, W);
}

// Internal helper: compute γ = V^{-1} ρ from (B-tensor, D).
//   ρ_P = Σ_Q L_PQ (Σ_μν B^Q_μν D_μν)
//   γ   = V^{-1} ρ = L^{-T} (B : D)  (one upper-triangular solve)
static Eigen::VectorXd compute_gamma_from_density(
    const std::vector<Eigen::MatrixXd>& B_per_P,
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& D,
    std::size_t n_aux) {
    Eigen::VectorXd b_dot_d(static_cast<Eigen::Index>(n_aux));
    for (std::size_t P = 0; P < n_aux; ++P) {
        b_dot_d(static_cast<Eigen::Index>(P)) =
            (B_per_P[P].array() * D.array()).sum();
    }
    return L.triangularView<Eigen::Lower>().transpose().solve(b_dot_d);
}

Eigen::MatrixXd DensityFitting::compute_j_gradient(
    const Molecule& mol, const Eigen::MatrixXd& D) const {
    if (orbital_basis_ == nullptr || aux_basis_ == nullptr) {
        throw std::runtime_error(
            "DensityFitting::compute_j_gradient: orbital/aux BasisSets "
            "are not stored — was this DensityFitting object built via "
            "the standard constructor?");
    }
    if (D.rows() != static_cast<Eigen::Index>(n_orb_)
        || D.cols() != static_cast<Eigen::Index>(n_orb_)) {
        throw std::invalid_argument(
            "DensityFitting::compute_j_gradient: D shape does not match "
            "the orbital basis dimension");
    }

    const Eigen::VectorXd gamma = compute_gamma_from_density(
        B_per_P_, L_, D, n_aux_);

    // 3-centre contribution: Σ_{P, μν} γ_P D_μν ∂(P|μν)/∂R.
    // 2-centre contribution: -(1/2) Σ_{PQ} γ_P γ_Q ∂V_{PQ}/∂R.
    Eigen::MatrixXd grad_3c = compute_3c_eri_gradient_contribution(
        *orbital_basis_, *aux_basis_, mol, D, gamma);
    Eigen::MatrixXd grad_2c = compute_2c_eri_gradient_contribution(
        *aux_basis_, mol, gamma);
    return grad_3c - 0.5 * grad_2c;
}

Eigen::MatrixXd DensityFitting::compute_k_gradient(
    const Molecule& mol,
    const Eigen::MatrixXd& C_occ,
    double alpha_hf) const {
    if (orbital_basis_ == nullptr || aux_basis_ == nullptr) {
        throw std::runtime_error(
            "DensityFitting::compute_k_gradient: orbital/aux BasisSets "
            "are not stored");
    }
    if (C_occ.rows() != static_cast<Eigen::Index>(n_orb_)) {
        throw std::invalid_argument(
            "DensityFitting::compute_k_gradient: C_occ.rows() does not "
            "match the orbital basis dimension");
    }
    const auto n_occ = static_cast<std::size_t>(C_occ.cols());
    if (n_occ == 0 || alpha_hf == 0.0) {
        // No occupied space, or pure DFT (no K contribution).
        return Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(mol.atoms().size()), 3);
    }

    // Build the half-MO B-tensor B_M^P = C_occ^T · B^P · C_occ.
    // Equivalently: B_M^P = (C_occ^T T^P C_occ) followed by L^{-1} on
    // the P axis (since T = L · B). Doing it per P:
    //   B_M^P_{ij} = (C_occ^T B^P C_occ)_{ij}
    //
    // Then η^P = L^{-T} B_M^P (one upper-triangular solve over the P
    // axis with the (n_occ², n_aux) flattened layout).
    //
    // From η we form:
    //   ω_{PQ} = (η^P : η^Q) = Σ_{ij} η^P_{ij} η^Q_{ij}
    //   Y^P_{μν} = (C_occ · η^P · C_occ^T)_{μν}
    // and contract with the 2c / 3c gradient kernels.

    // Stack B_M^P along columns (P-row × ij-col, row-major).
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;
    MatRM B_M(static_cast<Eigen::Index>(n_aux_),
              static_cast<Eigen::Index>(n_occ * n_occ));
    for (std::size_t P = 0; P < n_aux_; ++P) {
        const Eigen::MatrixXd Bm =
            C_occ.transpose() * B_per_P_[P] * C_occ;     // (n_occ, n_occ)
        for (std::size_t i = 0; i < n_occ; ++i) {
            for (std::size_t j = 0; j < n_occ; ++j) {
                B_M(static_cast<Eigen::Index>(P),
                    static_cast<Eigen::Index>(i * n_occ + j)) =
                    Bm(static_cast<Eigen::Index>(i),
                       static_cast<Eigen::Index>(j));
            }
        }
    }

    // η = L^{-T} B_M (one upper-triangular solve over the P axis).
    // solve(B_M) treats each column of B_M independently.
    MatRM eta = L_.triangularView<Eigen::Lower>().transpose().solve(B_M);

    // ω_{PQ} = (η^P : η^Q) = (eta · eta^T)_{PQ} (Frobenius product
    // over the (i, j) MO axis is exactly the GEMM over the n_occ²
    // flat axis).
    const Eigen::MatrixXd omega = eta * eta.transpose();   // (n_aux, n_aux)

    // Y^P_{μν} = C_occ · η^P · C_occ^T. Build flat row-major
    // (n_aux, n_orb²) for the 3c kernel.
    MatRM Y(static_cast<Eigen::Index>(n_aux_),
            static_cast<Eigen::Index>(n_orb_ * n_orb_));
    for (std::size_t P = 0; P < n_aux_; ++P) {
        // Reshape eta row P into (n_occ, n_occ).
        Eigen::MatrixXd eta_P(static_cast<Eigen::Index>(n_occ),
                              static_cast<Eigen::Index>(n_occ));
        for (std::size_t i = 0; i < n_occ; ++i) {
            for (std::size_t j = 0; j < n_occ; ++j) {
                eta_P(static_cast<Eigen::Index>(i),
                      static_cast<Eigen::Index>(j)) = eta(
                    static_cast<Eigen::Index>(P),
                    static_cast<Eigen::Index>(i * n_occ + j));
            }
        }
        const Eigen::MatrixXd Y_P = C_occ * eta_P * C_occ.transpose();
        for (std::size_t mu = 0; mu < n_orb_; ++mu) {
            for (std::size_t nu = 0; nu < n_orb_; ++nu) {
                Y(static_cast<Eigen::Index>(P),
                  static_cast<Eigen::Index>(mu * n_orb_ + nu)) = Y_P(
                    static_cast<Eigen::Index>(mu),
                    static_cast<Eigen::Index>(nu));
            }
        }
    }

    // K-gradient assembly:
    //   ∂E_K/∂R = +α_HF Σ_{PQ} ω_{PQ} ∂V_{PQ}/∂R
    //              − 2 α_HF Σ_{P, μν} Y^P_{μν} ∂(P|μν)/∂R
    const Eigen::MatrixXd grad_2c = compute_2c_eri_gradient_weighted(
        *aux_basis_, mol, omega);
    const Eigen::MatrixXd grad_3c = compute_3c_eri_gradient_weighted(
        *orbital_basis_, *aux_basis_, mol, Y);
    return alpha_hf * grad_2c - 2.0 * alpha_hf * grad_3c;
}

Eigen::MatrixXd DensityFitting::compute_jk_gradient(
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    const Eigen::MatrixXd& C_occ,
    double alpha_hf) const {
    if (alpha_hf == 0.0) {
        // Pure DFT — only J contributes.
        return compute_j_gradient(mol, D);
    }
    // For HF / hybrid we could compose J and K via two kernel calls
    // each (4 total). Folding into a single matrix-weighted call per
    // axis halves the libint work at the price of materialising
    // (n_aux, n_aux) and (n_aux, n_orb²) intermediates that we'd build
    // anyway for K — net win.

    const auto n_occ = static_cast<std::size_t>(C_occ.cols());
    if (C_occ.rows() != static_cast<Eigen::Index>(n_orb_)) {
        throw std::invalid_argument(
            "DensityFitting::compute_jk_gradient: C_occ.rows() does not "
            "match the orbital basis dimension");
    }

    const Eigen::VectorXd gamma = compute_gamma_from_density(
        B_per_P_, L_, D, n_aux_);

    // Build the K-side intermediates (B_M, η, ω, Y) — same as
    // compute_k_gradient.
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;
    MatRM B_M(static_cast<Eigen::Index>(n_aux_),
              static_cast<Eigen::Index>(n_occ * n_occ));
    for (std::size_t P = 0; P < n_aux_; ++P) {
        const Eigen::MatrixXd Bm =
            C_occ.transpose() * B_per_P_[P] * C_occ;
        for (std::size_t i = 0; i < n_occ; ++i) {
            for (std::size_t j = 0; j < n_occ; ++j) {
                B_M(static_cast<Eigen::Index>(P),
                    static_cast<Eigen::Index>(i * n_occ + j)) =
                    Bm(static_cast<Eigen::Index>(i),
                       static_cast<Eigen::Index>(j));
            }
        }
    }
    MatRM eta = L_.triangularView<Eigen::Lower>().transpose().solve(B_M);
    const Eigen::MatrixXd omega = eta * eta.transpose();

    MatRM Y(static_cast<Eigen::Index>(n_aux_),
            static_cast<Eigen::Index>(n_orb_ * n_orb_));
    for (std::size_t P = 0; P < n_aux_; ++P) {
        Eigen::MatrixXd eta_P(static_cast<Eigen::Index>(n_occ),
                              static_cast<Eigen::Index>(n_occ));
        for (std::size_t i = 0; i < n_occ; ++i) {
            for (std::size_t j = 0; j < n_occ; ++j) {
                eta_P(static_cast<Eigen::Index>(i),
                      static_cast<Eigen::Index>(j)) = eta(
                    static_cast<Eigen::Index>(P),
                    static_cast<Eigen::Index>(i * n_occ + j));
            }
        }
        const Eigen::MatrixXd Y_P = C_occ * eta_P * C_occ.transpose();
        for (std::size_t mu = 0; mu < n_orb_; ++mu) {
            for (std::size_t nu = 0; nu < n_orb_; ++nu) {
                Y(static_cast<Eigen::Index>(P),
                  static_cast<Eigen::Index>(mu * n_orb_ + nu)) = Y_P(
                    static_cast<Eigen::Index>(mu),
                    static_cast<Eigen::Index>(nu));
            }
        }
    }

    // Combined weights.
    //   2c:  Ω = -(1/2) γ γ^T + α_HF · ω
    //   3c:  W^P_{μν} = γ_P · D_{μν}  −  2 α_HF · Y^P_{μν}
    Eigen::MatrixXd Omega = -0.5 * gamma * gamma.transpose()
                            + alpha_hf * omega;

    MatRM W(static_cast<Eigen::Index>(n_aux_),
            static_cast<Eigen::Index>(n_orb_ * n_orb_));
    for (std::size_t P = 0; P < n_aux_; ++P) {
        const double gP = gamma(static_cast<Eigen::Index>(P));
        for (std::size_t mu = 0; mu < n_orb_; ++mu) {
            for (std::size_t nu = 0; nu < n_orb_; ++nu) {
                const auto col = static_cast<Eigen::Index>(
                    mu * n_orb_ + nu);
                W(static_cast<Eigen::Index>(P), col) =
                    gP * D(static_cast<Eigen::Index>(mu),
                           static_cast<Eigen::Index>(nu))
                    - 2.0 * alpha_hf * Y(
                        static_cast<Eigen::Index>(P), col);
            }
        }
    }

    const Eigen::MatrixXd grad_2c = compute_2c_eri_gradient_weighted(
        *aux_basis_, mol, Omega);
    const Eigen::MatrixXd grad_3c = compute_3c_eri_gradient_weighted(
        *orbital_basis_, *aux_basis_, mol, W);
    return grad_2c + grad_3c;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
DensityFitting::mo_transform(const Eigen::MatrixXd& C_left,
                              const Eigen::MatrixXd& C_right) const {
    const auto nl = static_cast<std::size_t>(C_left.cols());
    const auto nr = static_cast<std::size_t>(C_right.cols());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        result(static_cast<Eigen::Index>(n_aux_),
               static_cast<Eigen::Index>(nl * nr));

    // Per-P MO transform B^P_{pq} = (C_left^T · B^P · C_right)_{pq}.
    // Two GEMMs per P (n_orb² × n_left then n_left × n_right contractions).
    // Outer-P loop is embarrassingly parallel — each thread writes into
    // its own row of `result`.
    #pragma omp parallel for schedule(static)
    for (std::size_t P = 0; P < n_aux_; ++P) {
        const Eigen::MatrixXd B_mo =
            C_left.transpose() * B_per_P_[P] * C_right;  // (nl, nr)
        for (std::size_t p = 0; p < nl; ++p) {
            for (std::size_t q = 0; q < nr; ++q) {
                result(static_cast<Eigen::Index>(P),
                       static_cast<Eigen::Index>(p * nr + q)) = B_mo(
                    static_cast<Eigen::Index>(p),
                    static_cast<Eigen::Index>(q));
            }
        }
    }
    return result;
}

}  // namespace vibeqc
