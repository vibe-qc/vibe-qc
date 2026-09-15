// Periodic 2c / 3c Coulomb integrals on auxiliary bases. See
// vibeqc/aux_eri.hpp for the math contract; this file is the
// libint2-driven implementation.

#include "vibeqc/aux_eri.hpp"

#include "vibeqc/init.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace vibeqc {

namespace {

// Clone a libint shell vector and translate every origin by dr. Same
// helper as the one in cpp/src/lattice_integrals.cpp; duplicated here
// to keep aux_eri.cpp self-contained (the lattice_integrals one lives
// in an anonymous namespace).
std::vector<libint2::Shell> shift_shells(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr) {
    std::vector<libint2::Shell> out(shells.begin(), shells.end());
    for (auto& s : out) {
        s.O[0] += dr[0];
        s.O[1] += dr[1];
        s.O[2] += dr[2];
    }
    return out;
}

// max_shell_offset / pair_complete_cells / pair_in_range live in
// vibeqc/lattice_pair_cells.hpp. That header is the ONE definition of
// the enumeration contract a periodic two-centre lattice sum has to
// honour, and it carries the rationale plus the measured consequences
// of getting it wrong. Do not re-copy them into a second file.

}  // namespace

Eigen::MatrixXd compute_2c_eri_lattice(const BasisSet& aux,
                                       const PeriodicSystem& system,
                                       const LatticeSumOptions& opts) {
    ensure_libint_initialized();

    const auto& shells_ref = aux.libint();
    const auto n = aux.nbasis();
    // Pair-complete enumeration (see pair_complete_cells): bounding |g|
    // alone is not translation-invariant and breaks the Gram structure
    // of this metric on cells with a large intra-cell offset.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, shells_ref, shells_ref);

    // Per-thread accumulator: one (n × n) double matrix per OpenMP thread.
    // Threads parallelise across cells; each cell's contribution is added
    // to its own thread-local M, then we sum the locals at the end. This
    // avoids races that the (sP, sQ) shell-pair scatter would otherwise
    // introduce when two cells happen to write to the same matrix element
    // (which they do on the diagonal block).
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> M_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n),
                              static_cast<Eigen::Index>(n)));

    // libint2 BraKet::xs_xs prototype — same flavour as the molecular
    // compute_2c_eri kernel; the Coulomb operator just sees a different
    // shell pair on each call.
    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells_ref.max_nprim(),
                              shells_ref.max_l(),
                              0);
    prototype.set(libint2::BraKet::xs_xs);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells_ref.shell2bf();
    const int n_shells = static_cast<int>(shells_ref.size());
    const int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& M_local = M_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);

        // Full (sP, sQ) loop — no triangular shortcut, since the (Q_g)
        // shells are at a different origin than (P_0) so the integral
        // is NOT symmetric in (sP, sQ) at the per-cell level. The
        // symmetry holds only after summing over cells (g and -g
        // contribute the transposed pair). We rely on the cell loop
        // to enumerate both g and -g and let the elementwise sum
        // produce a symmetric matrix.
        for (int sP = 0; sP < n_shells; ++sP) {
            const auto bfP = shell2bf[sP];
            const auto nP = shells_ref[sP].size();
            for (int sQ = 0; sQ < n_shells; ++sQ) {
                const auto bfQ = shell2bf[sQ];
                const auto nQ = shells_g[sQ].size();

                if (!pair_in_range(shells_ref[sP], shells_g[sQ],
                                   opts.cutoff_bohr)) continue;

                engine.compute(shells_ref[sP], shells_g[sQ]);
                const double* block = buf[0];
                if (!block) continue;  // screened to zero

                for (std::size_t i = 0; i < nP; ++i) {
                    for (std::size_t j = 0; j < nQ; ++j) {
                        M_local(static_cast<Eigen::Index>(bfP + i),
                                static_cast<Eigen::Index>(bfQ + j)) +=
                            block[i * nQ + j];
                    }
                }
            }
        }
    }

    // Reduce: sum thread-local accumulators.
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n),
                                              static_cast<Eigen::Index>(n));
    for (const auto& m : M_tls) M += m;

    // Symmetrise: cell-list quadrature noise can leave (M − M^T) of order
    // 1e-15; symmetrising costs nothing and matches the GDF metric's
    // analytic property.
    M = 0.5 * (M + M.transpose());
    return M;
}

Lattice2CEriBlockSet compute_2c_eri_lattice_blocks(
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts) {
    ensure_libint_initialized();

    const auto& shells_ref = aux.libint();
    const auto n = aux.nbasis();
    // Same pair-complete enumeration as compute_2c_eri_lattice: the
    // documented contract is that summing these blocks reproduces the
    // Gamma metric, so the two must enumerate the identical term set.
    // The block set carries its own `cells`, so consumers stay aligned.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, shells_ref, shells_ref);

    Lattice2CEriBlockSet out;
    out.n_aux = n;
    out.cells = cells;
    out.blocks.assign(
        cells.size(),
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n),
                              static_cast<Eigen::Index>(n)));

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells_ref.max_nprim(),
                              shells_ref.max_l(),
                              0);
    prototype.set(libint2::BraKet::xs_xs);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells_ref.shell2bf();
    const int n_shells = static_cast<int>(shells_ref.size());
    const int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& M_block = out.blocks[static_cast<std::size_t>(c)];

        const Eigen::Vector3d& g = cells[static_cast<std::size_t>(c)].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);

        for (int sP = 0; sP < n_shells; ++sP) {
            const auto bfP = shell2bf[sP];
            const auto nP = shells_ref[sP].size();
            for (int sQ = 0; sQ < n_shells; ++sQ) {
                const auto bfQ = shell2bf[sQ];
                const auto nQ = shells_g[sQ].size();

                if (!pair_in_range(shells_ref[sP], shells_g[sQ],
                                   opts.cutoff_bohr)) continue;

                engine.compute(shells_ref[sP], shells_g[sQ]);
                const double* block = buf[0];
                if (!block) continue;

                for (std::size_t i = 0; i < nP; ++i) {
                    for (std::size_t j = 0; j < nQ; ++j) {
                        M_block(static_cast<Eigen::Index>(bfP + i),
                                static_cast<Eigen::Index>(bfQ + j)) +=
                            block[i * nQ + j];
                    }
                }
            }
        }
    }

    return out;
}

Eri3D compute_3c_eri_lattice(const BasisSet& orbital,
                             const BasisSet& aux,
                             const PeriodicSystem& system,
                             const LatticeSumOptions& opts) {
    ensure_libint_initialized();

    const auto& orb_sh = orbital.libint();
    const auto& aux_sh = aux.libint();
    const auto n_orb = orbital.nbasis();
    const auto n_aux = aux.nbasis();

    // Pair-complete enumeration (see pair_complete_cells). Only the ket
    // AO shell is translated, so the separation that decides whether a
    // term survives is the AO-pair one, |O_M - O_N - g|; the bra aux
    // shell rides along untranslated and does not bound the cell list.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, orb_sh, orb_sh);

    // Per-thread accumulator: one Eri3D per thread, then reduced. Same
    // pattern as compute_2c_eri_lattice — the (sM, sN_g) scatter pattern
    // crosses thread boundaries because two different cells g, g' can
    // both write to the same T(P, μ, ν) when only ν's libint shell is
    // shifted (the destination index is the unit-cell ν index).
    const int n_threads = omp_max_threads();
    std::vector<Eri3D> T_tls(n_threads);
    for (auto& t : T_tls) {
        t.n_aux = n_aux;
        t.n_orb = n_orb;
        t.data.assign(n_aux * n_orb * n_orb, 0.0);
    }

    libint2::Engine prototype(
        libint2::Operator::coulomb,
        std::max(orb_sh.max_nprim(), aux_sh.max_nprim()),
        std::max(orb_sh.max_l(), aux_sh.max_l()),
        0);
    prototype.set(libint2::BraKet::xs_xx);
    auto engines = make_engine_pool(prototype);

    const auto orb_shell2bf = orb_sh.shell2bf();
    const auto aux_shell2bf = aux_sh.shell2bf();
    const int n_aux_shells = static_cast<int>(aux_sh.size());
    const int n_orb_shells = static_cast<int>(orb_sh.size());
    const int n_cells = static_cast<int>(cells.size());

    // Outer parallel axis: lattice cells. Each thread does its own
    // (sP, sM_0, sN_g) loop and writes to a private T_tls entry; no
    // synchronisation needed inside the loop.
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& T_local = T_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto orb_shells_g = shift_shells(orb_sh, g);

        // Iterate ALL (sM, sN) pairs (no triangle shortcut): sM is in the
        // reference cell, sN is in cell g, so the integral is in general
        // not symmetric in (sM, sN). The symmetry T(P, μ, ν) = T(P, ν, μ)
        // emerges only from the full cell loop (translating by -g and
        // applying inversion gives the transpose).
        for (int sP = 0; sP < n_aux_shells; ++sP) {
            const auto bfP = aux_shell2bf[sP];
            const auto nP = aux_sh[sP].size();
            for (int sM = 0; sM < n_orb_shells; ++sM) {
                const auto bfM = orb_shell2bf[sM];
                const auto nM = orb_sh[sM].size();
                for (int sN = 0; sN < n_orb_shells; ++sN) {
                    const auto bfN = orb_shell2bf[sN];
                    const auto nN = orb_shells_g[sN].size();

                    if (!pair_in_range(orb_sh[sM], orb_shells_g[sN],
                                       opts.cutoff_bohr)) continue;

                    engine.compute(aux_sh[sP], orb_sh[sM], orb_shells_g[sN]);
                    const double* block = buf[0];
                    if (!block) continue;

                    for (std::size_t ip = 0; ip < nP; ++ip) {
                        const auto Pidx = bfP + ip;
                        for (std::size_t im = 0; im < nM; ++im) {
                            const auto mu = bfM + im;
                            for (std::size_t in = 0; in < nN; ++in) {
                                const auto nu = bfN + in;
                                T_local(Pidx, mu, nu) +=
                                    block[(ip * nM + im) * nN + in];
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce: sum thread-local Eri3D's.
    Eri3D T;
    T.n_aux = n_aux;
    T.n_orb = n_orb;
    T.data.assign(n_aux * n_orb * n_orb, 0.0);
    for (const auto& t : T_tls) {
        for (std::size_t k = 0; k < T.data.size(); ++k) {
            T.data[k] += t.data[k];
        }
    }

    // Symmetrise (μ, ν) — the cell sum has produced a numerically
    // symmetric tensor up to quadrature noise. Forcing it cleans up
    // ~1e-15 asymmetry and matches the analytic property.
    for (std::size_t P = 0; P < n_aux; ++P) {
        for (std::size_t mu = 0; mu < n_orb; ++mu) {
            for (std::size_t nu = mu + 1; nu < n_orb; ++nu) {
                const double avg = 0.5 * (T(P, mu, nu) + T(P, nu, mu));
                T(P, mu, nu) = avg;
                T(P, nu, mu) = avg;
            }
        }
    }
    return T;
}

Lattice3CEriBlockSet compute_3c_eri_lattice_blocks(
    const BasisSet& orbital,
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts) {
    ensure_libint_initialized();

    const auto& orb_sh = orbital.libint();
    const auto& aux_sh = aux.libint();
    const auto n_orb = orbital.nbasis();
    const auto n_aux = aux.nbasis();
    // Same pair-complete enumeration as compute_3c_eri_lattice (see the
    // 2c blocks note above); summing these blocks must reproduce the
    // Gamma tensor.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, orb_sh, orb_sh);

    Lattice3CEriBlockSet out;
    out.n_aux = n_aux;
    out.n_orb = n_orb;
    out.cells = cells;
    out.blocks.resize(cells.size());
    for (auto& t : out.blocks) {
        t.n_aux = n_aux;
        t.n_orb = n_orb;
        t.data.assign(n_aux * n_orb * n_orb, 0.0);
    }

    libint2::Engine prototype(
        libint2::Operator::coulomb,
        std::max(orb_sh.max_nprim(), aux_sh.max_nprim()),
        std::max(orb_sh.max_l(), aux_sh.max_l()),
        0);
    prototype.set(libint2::BraKet::xs_xx);
    auto engines = make_engine_pool(prototype);

    const auto orb_shell2bf = orb_sh.shell2bf();
    const auto aux_shell2bf = aux_sh.shell2bf();
    const int n_aux_shells = static_cast<int>(aux_sh.size());
    const int n_orb_shells = static_cast<int>(orb_sh.size());
    const int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& T_block = out.blocks[static_cast<std::size_t>(c)];

        const Eigen::Vector3d& g = cells[static_cast<std::size_t>(c)].r_cart;
        const auto orb_shells_g = shift_shells(orb_sh, g);

        for (int sP = 0; sP < n_aux_shells; ++sP) {
            const auto bfP = aux_shell2bf[sP];
            const auto nP = aux_sh[sP].size();
            for (int sM = 0; sM < n_orb_shells; ++sM) {
                const auto bfM = orb_shell2bf[sM];
                const auto nM = orb_sh[sM].size();
                for (int sN = 0; sN < n_orb_shells; ++sN) {
                    const auto bfN = orb_shell2bf[sN];
                    const auto nN = orb_shells_g[sN].size();

                    if (!pair_in_range(orb_sh[sM], orb_shells_g[sN],
                                       opts.cutoff_bohr)) continue;

                    engine.compute(aux_sh[sP], orb_sh[sM], orb_shells_g[sN]);
                    const double* block = buf[0];
                    if (!block) continue;

                    for (std::size_t ip = 0; ip < nP; ++ip) {
                        const auto Pidx = bfP + ip;
                        for (std::size_t im = 0; im < nM; ++im) {
                            const auto mu = bfM + im;
                            for (std::size_t in = 0; in < nN; ++in) {
                                const auto nu = bfN + in;
                                T_block(Pidx, mu, nu) +=
                                    block[(ip * nM + im) * nN + in];
                            }
                        }
                    }
                }
            }
        }
    }

    return out;
}

// ============================================================
// RSGDF — short-range halves
// ============================================================
// These are the SR pieces of the range-separated split
//   1/r₁₂ = erfc(ω r₁₂)/r₁₂  +  erf(ω r₁₂)/r₁₂
// computed via libint's Operator::erfc_coulomb. Same cell-loop and
// shell-shifting structure as the bare kernels above; the ONLY
// difference is the libint operator and the engine.set_params(ω)
// call. The LR pieces are computed analytically in Python from
// Gaussian Fourier transforms — see vibeqc.aux_basis.

Eigen::MatrixXd compute_2c_eri_lattice_sr(const BasisSet& aux,
                                          const PeriodicSystem& system,
                                          const LatticeSumOptions& opts,
                                          double omega) {
    ensure_libint_initialized();

    const auto& shells_ref = aux.libint();
    const auto n = aux.nbasis();
    // Pair-complete enumeration, as every bare kernel above uses. The
    // 2026-08-03 fix (5a40626a8) converted those and missed this SR pair.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, shells_ref, shells_ref);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> M_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n),
                              static_cast<Eigen::Index>(n)));

    // Operator::erfc_coulomb takes ω as a single scalar parameter.
    // ω = 0 reduces to the bare Coulomb kernel; ω → ∞ to the zero
    // potential. Typical RSGDF values: 0.1–0.4 bohr⁻¹.
    libint2::Engine prototype(libint2::Operator::erfc_coulomb,
                              shells_ref.max_nprim(),
                              shells_ref.max_l(),
                              0);
    prototype.set(libint2::BraKet::xs_xs);
    using erfc_coul_params = libint2::operator_traits<
        libint2::Operator::erfc_coulomb>::oper_params_type;
    prototype.set_params(erfc_coul_params{omega});
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells_ref.shell2bf();
    const int n_shells = static_cast<int>(shells_ref.size());
    const int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& M_local = M_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);

        for (int sP = 0; sP < n_shells; ++sP) {
            const auto bfP = shell2bf[sP];
            const auto nP = shells_ref[sP].size();
            for (int sQ = 0; sQ < n_shells; ++sQ) {
                const auto bfQ = shell2bf[sQ];
                const auto nQ = shells_g[sQ].size();

                if (!pair_in_range(shells_ref[sP], shells_g[sQ],
                                   opts.cutoff_bohr)) continue;

                engine.compute(shells_ref[sP], shells_g[sQ]);
                const double* block = buf[0];
                if (!block) continue;

                for (std::size_t i = 0; i < nP; ++i) {
                    for (std::size_t j = 0; j < nQ; ++j) {
                        M_local(static_cast<Eigen::Index>(bfP + i),
                                static_cast<Eigen::Index>(bfQ + j)) +=
                            block[i * nQ + j];
                    }
                }
            }
        }
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n),
                                              static_cast<Eigen::Index>(n));
    for (const auto& m : M_tls) M += m;
    M = 0.5 * (M + M.transpose());
    return M;
}

Eri3D compute_3c_eri_lattice_sr(const BasisSet& orbital,
                                const BasisSet& aux,
                                const PeriodicSystem& system,
                                const LatticeSumOptions& opts,
                                double omega) {
    ensure_libint_initialized();

    const auto& orb_sh = orbital.libint();
    const auto& aux_sh = aux.libint();
    const auto n_orb = orbital.nbasis();
    const auto n_aux = aux.nbasis();

    // Pair-complete enumeration, as every bare kernel above uses. The
    // 2026-08-03 fix (5a40626a8) converted those and missed this SR pair.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, orb_sh, orb_sh);

    const int n_threads = omp_max_threads();
    std::vector<Eri3D> T_tls(n_threads);
    for (auto& t : T_tls) {
        t.n_aux = n_aux;
        t.n_orb = n_orb;
        t.data.assign(n_aux * n_orb * n_orb, 0.0);
    }

    libint2::Engine prototype(
        libint2::Operator::erfc_coulomb,
        std::max(orb_sh.max_nprim(), aux_sh.max_nprim()),
        std::max(orb_sh.max_l(), aux_sh.max_l()),
        0);
    prototype.set(libint2::BraKet::xs_xx);
    using erfc_coul_params = libint2::operator_traits<
        libint2::Operator::erfc_coulomb>::oper_params_type;
    prototype.set_params(erfc_coul_params{omega});
    auto engines = make_engine_pool(prototype);

    const auto orb_shell2bf = orb_sh.shell2bf();
    const auto aux_shell2bf = aux_sh.shell2bf();
    const int n_aux_shells = static_cast<int>(aux_sh.size());
    const int n_orb_shells = static_cast<int>(orb_sh.size());
    const int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& T_local = T_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto orb_shells_g = shift_shells(orb_sh, g);

        for (int sP = 0; sP < n_aux_shells; ++sP) {
            const auto bfP = aux_shell2bf[sP];
            const auto nP = aux_sh[sP].size();
            for (int sM = 0; sM < n_orb_shells; ++sM) {
                const auto bfM = orb_shell2bf[sM];
                const auto nM = orb_sh[sM].size();
                for (int sN = 0; sN < n_orb_shells; ++sN) {
                    const auto bfN = orb_shell2bf[sN];
                    const auto nN = orb_shells_g[sN].size();

                    if (!pair_in_range(orb_sh[sM], orb_shells_g[sN],
                                       opts.cutoff_bohr)) continue;

                    engine.compute(aux_sh[sP], orb_sh[sM], orb_shells_g[sN]);
                    const double* block = buf[0];
                    if (!block) continue;

                    for (std::size_t ip = 0; ip < nP; ++ip) {
                        const auto Pidx = bfP + ip;
                        for (std::size_t im = 0; im < nM; ++im) {
                            const auto mu = bfM + im;
                            for (std::size_t in = 0; in < nN; ++in) {
                                const auto nu = bfN + in;
                                T_local(Pidx, mu, nu) +=
                                    block[(ip * nM + im) * nN + in];
                            }
                        }
                    }
                }
            }
        }
    }

    Eri3D T;
    T.n_aux = n_aux;
    T.n_orb = n_orb;
    T.data.assign(n_aux * n_orb * n_orb, 0.0);
    for (const auto& t : T_tls) {
        for (std::size_t k = 0; k < T.data.size(); ++k) {
            T.data[k] += t.data[k];
        }
    }
    for (std::size_t P = 0; P < n_aux; ++P) {
        for (std::size_t mu = 0; mu < n_orb; ++mu) {
            for (std::size_t nu = mu + 1; nu < n_orb; ++nu) {
                const double avg = 0.5 * (T(P, mu, nu) + T(P, nu, mu));
                T(P, mu, nu) = avg;
                T(P, nu, mu) = avg;
            }
        }
    }
    return T;
}

// ============================================================
// Periodic GDF analytic gradient kernels
// ============================================================

namespace {

// Shell index → atom index (0-based) for a given basis + molecule.
// Same as the anonymous-namespace helper in df.cpp; duplicated to
// keep aux_eri.cpp self-contained.
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

Eigen::MatrixXd compute_2c_eri_lattice_gradient_weighted(
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const Eigen::MatrixXd& omega) {
    ensure_libint_initialized();

    const auto& shells_ref = aux.libint();
    const auto n = aux.nbasis();
    const Molecule mol = system.unit_cell_molecule();
    const auto s2a = shell_to_atom_index(aux, mol);
    const std::size_t N = mol.atoms().size();
    // Must enumerate exactly as compute_2c_eri_lattice, whose sum this
    // differentiates; otherwise the analytic gradient is the derivative
    // of a different quantity than the energy.
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, shells_ref, shells_ref);

    if (static_cast<std::size_t>(omega.rows()) != n
        || static_cast<std::size_t>(omega.cols()) != n) {
        throw std::invalid_argument(
            "compute_2c_eri_lattice_gradient_weighted: omega shape does not "
            "match the auxiliary basis dimension");
    }

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells_ref.max_nprim(),
                              shells_ref.max_l(),
                              1 /*deriv_order*/);
    prototype.set(libint2::BraKet::xs_xs);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    const auto shell2bf = shells_ref.shell2bf();
    const int n_shells = static_cast<int>(shells_ref.size());
    const int n_cells = static_cast<int>(cells.size());

    // Outer loop over lattice cells. For each cell T, translate the Q
    // (ket-side) shells by T. The P-shells stay at the reference cell.
    // Derivative buffers: buf[0]=d/dR_P, buf[1]=d/dR_Q; both accumulate
    // on unit-cell atom indices (same physical atom).
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);

        for (int sP = 0; sP < n_shells; ++sP) {
            const auto bfP = shell2bf[sP];
            const auto nP = shells_ref[sP].size();
            const long atomP = s2a[sP];
            for (int sQ = 0; sQ < n_shells; ++sQ) {
                const auto bfQ = shell2bf[sQ];
                const auto nQ = shells_g[sQ].size();
                const long atomQ = s2a[sQ];  // unit-cell atom, not shifted

                if (!pair_in_range(shells_ref[sP], shells_g[sQ],
                                   opts.cutoff_bohr)) continue;

                engine.compute(shells_ref[sP], shells_g[sQ]);
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
    }

    // Reduce thread-local accumulators.
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(N), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

Eigen::MatrixXd compute_3c_eri_lattice_gradient_weighted(
    const BasisSet& orbital,
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& W) {
    ensure_libint_initialized();

    const auto& orb_sh = orbital.libint();
    const auto& aux_sh = aux.libint();
    const auto n_orb = orbital.nbasis();
    const auto n_aux = aux.nbasis();
    const Molecule mol = system.unit_cell_molecule();
    const auto orb_s2a = shell_to_atom_index(orbital, mol);
    const auto aux_s2a = shell_to_atom_index(aux, mol);
    const std::size_t N = mol.atoms().size();
    // Same enumeration as compute_3c_eri_lattice (see above).
    const auto cells =
        pair_complete_cells(system, opts.cutoff_bohr, orb_sh, orb_sh);

    if (static_cast<std::size_t>(W.rows()) != n_aux
        || static_cast<std::size_t>(W.cols()) != n_orb * n_orb) {
        throw std::invalid_argument(
            "compute_3c_eri_lattice_gradient_weighted: W shape does not match "
            "(n_aux, n_orb * n_orb)");
    }

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

    const auto orb_shell2bf = orb_sh.shell2bf();
    const auto aux_shell2bf = aux_sh.shell2bf();
    const int n_aux_shells = static_cast<int>(aux_sh.size());
    const int n_orb_shells = static_cast<int>(orb_sh.size());
    const int n_cells = static_cast<int>(cells.size());

    // Outer loop over lattice cells. P and μ stay in the reference cell;
    // ν is shifted by T. Three derivative centres: aux (P), orbital ket-1
    // (μ), orbital ket-2 (ν). All accumulate on unit-cell atom indices.
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& grad_local = grad_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto orb_shells_g = shift_shells(orb_sh, g);

        for (int sP = 0; sP < n_aux_shells; ++sP) {
            const auto bfP = aux_shell2bf[sP];
            const auto nP = aux_sh[sP].size();
            const long atomP = aux_s2a[sP];

            for (int sM = 0; sM < n_orb_shells; ++sM) {
                for (int sN = 0; sN < n_orb_shells; ++sN) {
                    // Same AO-pair range filter as compute_3c_eri_lattice,
                    // whose sum this differentiates. Applied on the
                    // pre-canonical (sM, sN) pair, exactly as there, so
                    // the two enumerate the identical term set.
                    if (!pair_in_range(orb_sh[sM], orb_shells_g[sN],
                                       opts.cutoff_bohr)) continue;

                    // Fresh engine per call — prevents libint state leak
                    // (same pattern as the molecular 3c gradient, df.cpp).
                    libint2::Engine engine = prototype;
                    const auto& buf = engine.results();

                    // L-canonical reorder (same as molecular 3c gradient).
                    const int lM = orb_sh[sM].contr[0].l;
                    const int lN = orb_sh[sN].contr[0].l;
                    const bool swap = (lM < lN);
                    const int sM_canon = swap ? sN : sM;
                    const int sN_canon = swap ? sM : sN;

                    const auto bfM_canon = orb_shell2bf[sM_canon];
                    const auto nM_canon  = orb_sh[sM_canon].size();
                    const auto bfN_canon = orb_shell2bf[sN_canon];
                    const auto nN_canon  = orb_shells_g[sN_canon].size();
                    const long atomM_canon = orb_s2a[sM_canon];
                    const long atomN_canon = orb_s2a[sN_canon];

                    engine.compute(aux_sh[sP],
                                    orb_sh[sM_canon], orb_shells_g[sN_canon]);
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
    }

    // Reduce thread-local accumulators.
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(N), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

}  // namespace vibeqc
