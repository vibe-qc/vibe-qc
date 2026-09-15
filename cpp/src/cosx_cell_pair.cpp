#include "vibeqc/cosx_cell_pair.hpp"

#include <Eigen/Dense>
#include <libint2/engine.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/cosx.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/schwarz.hpp"
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {

namespace {

// AO screening tolerance — same value and same semantics as the
// molecular COSX kernel (cpp/src/cosx.cpp): values below this are
// treated as zero in the chi / Dχ / Schwarz screens.
constexpr double AO_TOL = 1e-7;

// Hash for the integer lattice-shift key (same FNV-ish pattern as
// periodic_fock.cpp's CellIndexMap — cells live in a small box).
struct ShiftHash {
    std::size_t operator()(const Eigen::Vector3i& v) const noexcept {
        const auto a = static_cast<std::uint64_t>(v[0] + 1024);
        const auto b = static_cast<std::uint64_t>(v[1] + 1024);
        const auto c = static_cast<std::uint64_t>(v[2] + 1024);
        return static_cast<std::size_t>((a * 2654435761ULL) ^
                                        (b * 40503ULL) ^ c);
    }
};
struct ShiftEq {
    bool operator()(const Eigen::Vector3i& a,
                    const Eigen::Vector3i& b) const noexcept {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};

}  // namespace

CosxCellPairCaches build_cosx_cell_pair_caches(
    const BasisSet& basis,
    const std::vector<LatticeCell>& cells,
    double schwarz_drop_tol,
    double omega) {
    ensure_libint_initialized();

    if (cells.empty()) {
        throw std::runtime_error(
            "build_cosx_cell_pair_caches: empty cell list");
    }
    if (omega < 0.0) {
        throw std::runtime_error(
            "build_cosx_cell_pair_caches: omega must be >= 0");
    }
    bool has_zero = false;
    for (const auto& c : cells) {
        if (c.r_cart.norm() < 1e-12) { has_zero = true; break; }
    }
    if (!has_zero) {
        throw std::runtime_error(
            "build_cosx_cell_pair_caches: cell list lacks the zero "
            "cell (pass the direct_lattice_cells output unmodified)");
    }

    const auto& shells = basis.libint();
    const auto n_shells = static_cast<int>(shells.size());
    const int n_cells = static_cast<int>(cells.size());

    CosxCellPairCaches caches;
    caches.omega = omega;
    caches.cells = cells;
    caches.delta_index.assign(
        static_cast<std::size_t>(n_cells) * n_cells, -1);

    // Distinct integer shifts δ = index(g) − index(p), with the first
    // (p, g) realisation remembered for the Cartesian shift. The
    // Cartesian difference is identical for every realisation of the
    // same integer shift (lattice linearity).
    std::unordered_map<Eigen::Vector3i, int, ShiftHash, ShiftEq>
        shift_to_delta;
    std::vector<Eigen::Vector3d> delta_carts;
    std::vector<std::vector<std::pair<int, int>>> realisations;
    for (int p = 0; p < n_cells; ++p) {
        for (int g = 0; g < n_cells; ++g) {
            const Eigen::Vector3i key = cells[static_cast<std::size_t>(g)].index
                                      - cells[static_cast<std::size_t>(p)].index;
            auto it = shift_to_delta.find(key);
            int k;
            if (it == shift_to_delta.end()) {
                k = static_cast<int>(delta_carts.size());
                shift_to_delta.emplace(key, k);
                delta_carts.push_back(
                    cells[static_cast<std::size_t>(g)].r_cart
                    - cells[static_cast<std::size_t>(p)].r_cart);
                realisations.emplace_back();
            } else {
                k = it->second;
            }
            realisations[static_cast<std::size_t>(k)].emplace_back(p, g);
        }
    }

    // Per-δ Schwarz tables over (ν_δ, σ_0) pairs; a shift whose whole
    // table is below the drop tolerance contributes < tol · ‖Dχ‖ per
    // grid point and is dropped wholesale — this is what bounds the
    // ket-cell sum to AO-pair-overlap neighbours. With ω > 0 the
    // metric is the erfc-screened Coulomb, matching the kernel the
    // caches will feed. Note the Schwarz factor is the pair density's
    // SELF-interaction — it decays with |δ| through the pair OVERLAP
    // only (erfc(ω·0) = 1 at zero self-distance), so the δ set is the
    // same as for the full kernel; the SR locality shows up in the
    // K(g) block decay (bra-ket coupling), not in the cache size.
    libint2::Engine engine(
        omega > 0.0 ? libint2::Operator::erfc_coulomb
                    : libint2::Operator::coulomb,
        shells.max_nprim(), shells.max_l(), 0);
    if (omega > 0.0) {
        engine.set_params(omega);
    }
    const auto& buf = engine.results();

    // Home shell list as a plain vector — the σ side of every
    // two-set pair cache.
    const std::vector<libint2::Shell> home_vec(shells.begin(),
                                               shells.end());

    for (std::size_t k = 0; k < delta_carts.size(); ++k) {
        const Eigen::Vector3d& dr = delta_carts[k];
        std::vector<libint2::Shell> shifted =
            shift_shells_to_cell(shells, dr);

        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n_shells, n_shells);
        double q_max = 0.0;
        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto n1 = shifted[static_cast<std::size_t>(s1)].size();
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const auto n2 = shells[static_cast<std::size_t>(s2)].size();
                // (ν_δ σ_0 | ν_δ σ_0): Coulomb-self of the shifted pair.
                engine.compute(shifted[static_cast<std::size_t>(s1)],
                               shells[static_cast<std::size_t>(s2)],
                               shifted[static_cast<std::size_t>(s1)],
                               shells[static_cast<std::size_t>(s2)]);
                const double* block = buf[0];
                if (block == nullptr) continue;
                const std::size_t n_elem = n1 * n2 * n1 * n2;
                double mx = 0.0;
                for (std::size_t i = 0; i < n_elem; ++i) {
                    mx = std::max(mx, std::abs(block[i]));
                }
                const double q = std::sqrt(mx);
                Q(s1, s2) = q;
                q_max = std::max(q_max, q);
            }
        }
        if (q_max < schwarz_drop_tol) {
            continue;  // realisations keep delta_index = -1
        }

        const int d = static_cast<int>(caches.deltas.size());
        caches.deltas.push_back(dr);
        caches.schwarz_delta.push_back(std::move(Q));
        caches.pp_delta.push_back(
            build_primitive_pair_cache_two_set(shifted, home_vec));
        caches.shells_delta.push_back(std::move(shifted));
        for (const auto& pg : realisations[k]) {
            caches.delta_index[
                static_cast<std::size_t>(pg.first) * n_cells
                + pg.second] = d;
        }
    }

    // SR pair cutoff tables (omega > 0): probe the validated kernel
    // itself at increasing pseudo-nucleus distances and record, per
    // (δ, s_ν, s_σ), the radius beyond which the SR A-block is below
    // the screening floor. Probing the real kernel makes the table
    // exact w.r.t. both the erfc range and the pair-density extent —
    // including diffuse shells, where analytic extent models
    // under- or over-shoot. Cost: n_δ × n_shells² × (directions ×
    // distance grid) kernel calls at setup — sub-second to seconds.
    if (omega > 0.0) {
        // Floor: one decade below the engine's AO_TOL screen so the
        // dchi multiplier (typically ≤ 10) cannot resurrect a
        // skipped pair.
        constexpr double kCutFloor = 1e-9;
        constexpr double kStep = 1.0;     // bohr, distance grid
        constexpr double kMargin = 1.0;   // bohr, safety on top
        const double d_max_probe =
            60.0 + 10.0 / omega;          // generous upper bound

        const auto boys = build_boys_table(shells.max_l());
        const int max_l_shell = shells.max_l();
        CosxKernelWorkspace ws;
        ws.reserve(max_l_shell, max_l_shell);
        const int max_n_per_shell =
            (max_l_shell + 1) * (max_l_shell + 2) / 2;
        std::vector<double> block(
            static_cast<std::size_t>(max_n_per_shell) * max_n_per_shell);

        caches.r_cut_delta.reserve(caches.deltas.size());
        caches.r_cut_max_delta.reserve(caches.deltas.size());
        for (std::size_t kd = 0; kd < caches.deltas.size(); ++kd) {
            const auto& sh_nu = caches.shells_delta[kd];
            const auto& pp = caches.pp_delta[kd];
            Eigen::MatrixXd rc =
                Eigen::MatrixXd::Zero(n_shells, n_shells);
            double rc_max = 0.0;
            for (int s1 = 0; s1 < n_shells; ++s1) {
                const auto n1 = sh_nu[static_cast<std::size_t>(s1)].size();
                for (int s2 = 0; s2 < n_shells; ++s2) {
                    if (caches.schwarz_delta[kd](s1, s2)
                            < schwarz_drop_tol) {
                        // Pair dead by overlap at any distance.
                        rc(s1, s2) = 0.0;
                        continue;
                    }
                    const auto n2 =
                        shells[static_cast<std::size_t>(s2)].size();
                    // Pair reference point: midpoint of the two shell
                    // centres. Probe along two orthogonal-ish
                    // directions and keep the worst (the block decay
                    // is near-isotropic at SR-dead distances; two
                    // directions guard the pair-axis anisotropy).
                    const auto& Oa =
                        sh_nu[static_cast<std::size_t>(s1)].O;
                    const auto& Ob =
                        shells[static_cast<std::size_t>(s2)].O;
                    const Eigen::Vector3d mid(
                        0.5 * (Oa[0] + Ob[0]),
                        0.5 * (Oa[1] + Ob[1]),
                        0.5 * (Oa[2] + Ob[2]));
                    Eigen::Vector3d axis(
                        Ob[0] - Oa[0], Ob[1] - Oa[1], Ob[2] - Oa[2]);
                    if (axis.norm() < 1e-8) axis = Eigen::Vector3d(1, 0, 0);
                    axis.normalize();
                    // A second direction least aligned with the axis.
                    Eigen::Vector3d perp =
                        axis.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
                    if (perp.norm() < 1e-8) {
                        perp = axis.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
                    }
                    perp.normalize();

                    // The engine screens on the MIN distance from the
                    // pseudo-nucleus to either shell centre, so the
                    // recorded cutoff uses that same metric (the
                    // min-centre distance of the breaking probe point,
                    // maxed over directions, + margin).
                    auto min_center_dist =
                        [&](const Eigen::Vector3d& Cv) {
                            const double da = (Cv - Eigen::Vector3d(
                                Oa[0], Oa[1], Oa[2])).norm();
                            const double db = (Cv - Eigen::Vector3d(
                                Ob[0], Ob[1], Ob[2])).norm();
                            return std::min(da, db);
                        };
                    double r_cut = d_max_probe;
                    for (double dprobe = kStep; dprobe <= d_max_probe;
                         dprobe += kStep) {
                        double mx = 0.0;
                        double mc = 0.0;
                        for (const auto& dir : {axis, perp}) {
                            const Eigen::Vector3d Cv =
                                mid + dprobe * dir;
                            const std::array<double, 3> C = {
                                Cv[0], Cv[1], Cv[2]};
                            cosx_nuclear_pair_into(
                                sh_nu[static_cast<std::size_t>(s1)],
                                shells[static_cast<std::size_t>(s2)],
                                pp.pairs[static_cast<std::size_t>(s1)
                                         * n_shells + s2],
                                C, boys, ws, block.data(), omega);
                            for (std::size_t e = 0; e < n1 * n2; ++e) {
                                mx = std::max(mx, std::abs(block[e]));
                            }
                            mc = std::max(mc, min_center_dist(Cv));
                        }
                        if (mx < kCutFloor) {
                            r_cut = mc + kMargin;
                            break;
                        }
                    }
                    rc(s1, s2) = r_cut;
                    rc_max = std::max(rc_max, r_cut);
                }
            }
            caches.r_cut_delta.push_back(std::move(rc));
            caches.r_cut_max_delta.push_back(rc_max);
        }
    }

    return caches;
}

LatticeMatrixSet compute_cosx_k_blocks(
    const BasisSet& basis,
    const LatticeMatrixSet& P_real_space,
    const Grid& cosx_grid,
    const CosxCellPairCaches& caches,
    const Eigen::MatrixXd& q_cached,
    const BoysTable* boys_table) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const std::size_t n_shells = shells.size();
    const int n_bf = static_cast<int>(basis.nbasis());
    const int n_pts = static_cast<int>(cosx_grid.points.rows());
    const int n_cells = static_cast<int>(caches.cells.size());

    if (P_real_space.nbf != n_bf) {
        throw std::runtime_error(
            "compute_cosx_k_blocks: density nbf mismatch");
    }

    // Result blocks live on the caches' cell list — the same output
    // convention as build_jk_2e_real_space_explicit.
    LatticeMatrixSet K_set;
    K_set.nbf = n_bf;
    K_set.cells = caches.cells;
    K_set.blocks.assign(static_cast<std::size_t>(n_cells),
                        Eigen::MatrixXd::Zero(n_bf, n_bf));

    // Density-block lookup by integer cell difference, precomputed as
    // a (c_λ, c_σ) table: p_idx[c_λ * n_cells + c_σ] = block index in
    // P_real_space, or −1 when the difference is outside the density
    // set (zero block by truncation — same convention as the
    // direct-ERI reference's ``p_block`` lambda in periodic_fock.cpp).
    std::unordered_map<Eigen::Vector3i, int, ShiftHash, ShiftEq>
        p_cell_index;
    for (std::size_t i = 0; i < P_real_space.cells.size(); ++i) {
        p_cell_index.emplace(P_real_space.cells[i].index,
                             static_cast<int>(i));
    }
    std::vector<int> p_idx(
        static_cast<std::size_t>(n_cells) * n_cells, -1);
    for (int cl = 0; cl < n_cells; ++cl) {
        for (int cs = 0; cs < n_cells; ++cs) {
            const Eigen::Vector3i h =
                caches.cells[static_cast<std::size_t>(cs)].index
                - caches.cells[static_cast<std::size_t>(cl)].index;
            auto it = p_cell_index.find(h);
            if (it != p_cell_index.end()) {
                p_idx[static_cast<std::size_t>(cl) * n_cells + cs] =
                    it->second;
            }
        }
    }

    // Per-cell shifted AO tables χ(r_G − c) over the whole grid.
    // Cells with no support anywhere are released and skipped.
    // Memory: n_active_cells × n_pts × n_bf doubles — fine at
    // validation scale; chunk for production.
    std::vector<Eigen::MatrixXd> ao_at_cell(
        static_cast<std::size_t>(n_cells));
    int i_home = -1;
    for (int p = 0; p < n_cells; ++p) {
        const Eigen::Vector3d& r_cell =
            caches.cells[static_cast<std::size_t>(p)].r_cart;
        if (r_cell.norm() < 1e-12) i_home = p;
        Eigen::MatrixX3d pts = cosx_grid.points;
        pts.rowwise() -= r_cell.transpose();
        Eigen::MatrixXd ao = evaluate_ao(basis, pts);
        if (ao.size() == 0 || ao.cwiseAbs().maxCoeff() < AO_TOL) {
            continue;  // leave 0×0 — bra-inactive cell
        }
        ao_at_cell[static_cast<std::size_t>(p)] = std::move(ao);
    }
    if (i_home < 0) {
        throw std::runtime_error(
            "compute_cosx_k_blocks: caches.cells lacks the zero cell");
    }
    const Eigen::MatrixXd& ao_home =
        ao_at_cell[static_cast<std::size_t>(i_home)];
    if (ao_home.size() == 0) {
        // No basis support anywhere on the grid — K is exactly zero.
        return K_set;
    }

    // Boys table: caller-cached or local one-time build.
    const BoysTable* boys_use = boys_table;
    BoysTable local_boys;
    if (boys_use == nullptr) {
        local_boys = build_boys_table(shells.max_l());
        boys_use = &local_boys;
    }
    const int max_l_shell = shells.max_l();

    // Per-δ Schwarz maxima for the cell-level skip.
    std::vector<double> schwarz_max_delta(caches.schwarz_delta.size());
    for (std::size_t k = 0; k < caches.schwarz_delta.size(); ++k) {
        schwarz_max_delta[k] =
            caches.schwarz_delta[k].size() == 0
                ? 0.0
                : caches.schwarz_delta[k].maxCoeff();
    }

    const std::size_t n_threads =
        static_cast<std::size_t>(std::max(1, omp_max_threads()));
    // Per-thread accumulators: one block per output cell.
    std::vector<std::vector<Eigen::MatrixXd>> K_thread(
        n_threads,
        std::vector<Eigen::MatrixXd>(
            static_cast<std::size_t>(n_cells),
            Eigen::MatrixXd::Zero(n_bf, n_bf)));

    #pragma omp parallel
    {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& K_local = K_thread[tid];

        CosxKernelWorkspace ws;
        ws.reserve(max_l_shell, max_l_shell);
        const int max_n_per_shell =
            (max_l_shell + 1) * (max_l_shell + 2) / 2;
        std::vector<double> kernel_block(
            static_cast<std::size_t>(max_n_per_shell) * max_n_per_shell);

        // Thread-local per-point scratch, reused across points.
        std::vector<double> v_max_local(n_shells, 0.0);
        std::vector<double> dist_sigma(n_shells, 0.0);
        std::vector<double> dist_nu(n_shells, 0.0);
        // V^{(c_σ)} vectors and F^{(g)} accumulators, one column per
        // cell. ``f_active`` flags which output cells received work
        // at this point.
        Eigen::MatrixXd V(n_bf, n_cells);
        Eigen::MatrixXd F(n_bf, n_cells);
        std::vector<char> f_active(static_cast<std::size_t>(n_cells));

        #pragma omp for schedule(guided)
        for (int g_pt = 0; g_pt < n_pts; ++g_pt) {
            const double w_g = cosx_grid.weights(g_pt);
            if (w_g == 0.0) continue;
            const Eigen::VectorXd chi_home =
                ao_home.row(g_pt).transpose();
            // Bra-row screen: every K(g) row scales with χ_μ(r_G).
            if (chi_home.cwiseAbs().maxCoeff() < AO_TOL) continue;

            const double rg_x = cosx_grid.points(g_pt, 0);
            const double rg_y = cosx_grid.points(g_pt, 1);
            const double rg_z = cosx_grid.points(g_pt, 2);

            // ---- V build: V^{(c_σ)}_σ = Σ_{c_λ} Σ_λ P(c_σ−c_λ)_{λσ}
            //      · χ_λ(r_G − c_λ)  —  i.e. V += P(h)ᵀ · χ. ----
            // P's ROW index is the λ (bra) index and its COLUMN the σ
            // (ket) index — the direct-ERI reference contracts
            // ``P(bf3, bf4) · (μ_0 λ_{c_λ} | ν_g σ_{c_σ})`` with bf3
            // on λ. For symmetric blocks (cell-diagonal density) the
            // transpose is invisible; for off-diagonal blocks
            // (P(h) ≠ P(h)ᵀ in general, only P(h) = P(−h)ᵀ holds) it
            // is essential. Loop bra cells once (rigorous χ screen),
            // scatter into every σ-cell column with a surviving
            // density block.
            V.setZero();
            bool any_v = false;
            for (int cl = 0; cl < n_cells; ++cl) {
                const Eigen::MatrixXd& ao_p =
                    ao_at_cell[static_cast<std::size_t>(cl)];
                if (ao_p.size() == 0) continue;
                const Eigen::VectorXd chi_l =
                    ao_p.row(g_pt).transpose();
                if (chi_l.cwiseAbs().maxCoeff() < AO_TOL) continue;
                for (int cs = 0; cs < n_cells; ++cs) {
                    const int pb = p_idx[
                        static_cast<std::size_t>(cl) * n_cells + cs];
                    if (pb < 0) continue;
                    const Eigen::MatrixXd& P_block =
                        P_real_space.blocks[static_cast<std::size_t>(pb)];
                    if (P_block.size() == 0) continue;
                    V.col(cs).noalias() +=
                        P_block.transpose() * chi_l;
                    any_v = true;
                }
            }
            if (!any_v) continue;

            F.setZero();
            std::fill(f_active.begin(), f_active.end(), 0);
            bool any_work = false;

            // ---- σ-cell loop: A^{(δ)}(r_G − c_σ) · V^{(c_σ)} ----
            for (int cs = 0; cs < n_cells; ++cs) {
                // Per-shell max |V| (σ-side screen) + overall max.
                double v_overall = 0.0;
                for (std::size_t s = 0; s < n_shells; ++s) {
                    const auto bf0 = shell2bf[s];
                    const auto ns = shells[s].size();
                    double m = 0.0;
                    for (std::size_t i = 0; i < ns; ++i) {
                        m = std::max(m, std::abs(
                            V(static_cast<Eigen::Index>(bf0 + i), cs)));
                    }
                    v_max_local[s] = m;
                    v_overall = std::max(v_overall, m);
                }
                if (v_overall < AO_TOL) continue;

                // Pseudo-nucleus for this σ cell: C = r_G − c_σ.
                const Eigen::Vector3d& r_cs =
                    caches.cells[static_cast<std::size_t>(cs)].r_cart;
                const double cx = rg_x - r_cs[0];
                const double cy = rg_y - r_cs[1];
                const double cz = rg_z - r_cs[2];
                const std::array<double, 3> C_g = {cx, cy, cz};

                // Home-σ distances to C — shared across output cells.
                for (std::size_t s = 0; s < n_shells; ++s) {
                    const auto& O = shells[s].O;
                    const double dx = O[0] - cx;
                    const double dy = O[1] - cy;
                    const double dz = O[2] - cz;
                    dist_sigma[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
                }

                // ---- output-cell loop (g): δ = g − c_σ lookup ----
                for (int g_c = 0; g_c < n_cells; ++g_c) {
                    const int d = caches.delta_index[
                        static_cast<std::size_t>(cs) * n_cells + g_c];
                    if (d < 0) continue;  // Schwarz-dropped shift
                    if (schwarz_max_delta[static_cast<std::size_t>(d)]
                            * v_overall < AO_TOL) {
                        continue;
                    }
                    const auto& sh_nu = caches.shells_delta[
                        static_cast<std::size_t>(d)];
                    const auto& pp = caches.pp_delta[
                        static_cast<std::size_t>(d)];
                    const auto& schwarz = caches.schwarz_delta[
                        static_cast<std::size_t>(d)];

                    // ν_δ-shell distances to C (distance damping for
                    // the Schwarz screen, capped at 1 as in the
                    // molecular kernel).
                    double min_dist_nu = 1e300;
                    for (std::size_t s = 0; s < n_shells; ++s) {
                        const auto& O = sh_nu[s].O;
                        const double dx = O[0] - cx;
                        const double dy = O[1] - cy;
                        const double dz = O[2] - cz;
                        dist_nu[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
                        min_dist_nu = std::min(min_dist_nu, dist_nu[s]);
                    }

                    // SR cell-level cutoff (omega > 0): if every pair
                    // of this δ is beyond its tabulated SR radius —
                    // bounded below by the min ν_δ-centre distance —
                    // the whole (c_σ, δ) combination is dead.
                    if (!caches.r_cut_max_delta.empty()
                        && min_dist_nu > caches.r_cut_max_delta[
                               static_cast<std::size_t>(d)]) {
                        continue;
                    }

                    // Rectangular (ν_δ, σ_0) pair loop. One-directional
                    // contraction — only σ carries the density weight:
                    //   F^{(g)}_ν += A^{(δ)}_{νσ}(C) · V^{(c_σ)}_σ
                    // (The (c_σ, g) ↔ (g, c_σ) transpose pairing could
                    // halve the kernel calls — perf follow-up.)
                    const Eigen::MatrixXd* r_cut =
                        caches.r_cut_delta.empty()
                            ? nullptr
                            : &caches.r_cut_delta[
                                  static_cast<std::size_t>(d)];
                    for (std::size_t s2 = 0; s2 < n_shells; ++s2) {
                        const double v_s = v_max_local[s2];
                        if (v_s < AO_TOL) continue;
                        const auto bf2 = shell2bf[s2];
                        const auto n2 = shells[s2].size();
                        for (std::size_t s1 = 0; s1 < n_shells; ++s1) {
                            const double d_min = std::min(
                                dist_nu[s1], dist_sigma[s2]);
                            // SR per-pair cutoff (omega > 0): the
                            // tabulated radius reflects kernel range
                            // AND pair extent — the screen that makes
                            // basis-extent-sized cell domains
                            // affordable (M3b-6).
                            if (r_cut != nullptr
                                && d_min > (*r_cut)(
                                       static_cast<Eigen::Index>(s1),
                                       static_cast<Eigen::Index>(s2))) {
                                continue;
                            }
                            const double dist_factor =
                                1.0 / std::max(1.0, d_min);
                            const double bound =
                                schwarz(static_cast<Eigen::Index>(s1),
                                        static_cast<Eigen::Index>(s2))
                                * v_s * dist_factor;
                            if (bound < AO_TOL) continue;

                            const auto bf1 = shell2bf[s1];
                            const auto n1 = sh_nu[s1].size();
                            // Kernel range follows the caches: full
                            // Coulomb (omega = 0) or erfc-SR
                            // (omega > 0, M3b-4a).
                            cosx_nuclear_pair_into(
                                sh_nu[s1],
                                shells[s2],
                                pp.pairs[s1 * n_shells + s2],
                                C_g, *boys_use, ws,
                                kernel_block.data(),
                                caches.omega);
                            Eigen::Map<const Eigen::Matrix<
                                double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
                                buf_mat(kernel_block.data(),
                                        static_cast<Eigen::Index>(n1),
                                        static_cast<Eigen::Index>(n2));
                            F.col(g_c).segment(bf1, n1).noalias() +=
                                buf_mat * V.col(cs).segment(bf2, n2);
                            f_active[static_cast<std::size_t>(g_c)] = 1;
                            any_work = true;
                        }
                    }
                }
            }
            if (!any_work) continue;

            // K(g) += w_g · χ_home · F^{(g)ᵀ} — row-screened sparse
            // update below 33 % active rows, single GEMM above (same
            // calibration as the molecular kernel, cosx.cpp).
            int n_active_rows = 0;
            for (int mu = 0; mu < n_bf; ++mu)
                if (std::abs(chi_home(mu)) >= AO_TOL) ++n_active_rows;
            for (int g_c = 0; g_c < n_cells; ++g_c) {
                if (!f_active[static_cast<std::size_t>(g_c)]) continue;
                auto& K_g = K_local[static_cast<std::size_t>(g_c)];
                if (n_active_rows * 3 < n_bf) {
                    for (int mu = 0; mu < n_bf; ++mu) {
                        const double cmu = chi_home(mu);
                        if (std::abs(cmu) < AO_TOL) continue;
                        K_g.row(mu).noalias() +=
                            (w_g * cmu) * F.col(g_c).transpose();
                    }
                } else {
                    K_g.noalias() +=
                        (w_g * chi_home) * F.col(g_c).transpose();
                }
            }
        }
    }  // omp parallel

    for (const auto& Kt : K_thread) {
        for (int g_c = 0; g_c < n_cells; ++g_c) {
            K_set.blocks[static_cast<std::size_t>(g_c)] +=
                Kt[static_cast<std::size_t>(g_c)];
        }
    }

    // Empty Q → raw quadrature blocks (the validation contract).
    if (q_cached.size() == 0) {
        return K_set;
    }

    // Bra-side Q-junction from the left + pairwise enforcement of the
    // exact block symmetry K(g) = K(−g)ᵀ — the block generalisation
    // of the molecular symmetrise + Q post-processing. Cells whose −g
    // partner is outside the list keep the unsymmetrised left
    // correction (only possible with a non-ball cell list).
    std::vector<int> minus_index(static_cast<std::size_t>(n_cells), -1);
    {
        std::unordered_map<Eigen::Vector3i, int, ShiftHash, ShiftEq>
            cell_map;
        for (int c = 0; c < n_cells; ++c) {
            cell_map.emplace(
                caches.cells[static_cast<std::size_t>(c)].index, c);
        }
        for (int c = 0; c < n_cells; ++c) {
            const Eigen::Vector3i mi =
                -caches.cells[static_cast<std::size_t>(c)].index;
            auto it = cell_map.find(mi);
            if (it != cell_map.end()) minus_index[
                static_cast<std::size_t>(c)] = it->second;
        }
    }
    std::vector<Eigen::MatrixXd> corrected(
        static_cast<std::size_t>(n_cells));
    for (int c = 0; c < n_cells; ++c) {
        const Eigen::MatrixXd QK =
            q_cached * K_set.blocks[static_cast<std::size_t>(c)];
        const int m = minus_index[static_cast<std::size_t>(c)];
        if (m >= 0) {
            corrected[static_cast<std::size_t>(c)] =
                0.5 * (QK
                       + (q_cached
                          * K_set.blocks[static_cast<std::size_t>(m)])
                             .transpose());
        } else {
            corrected[static_cast<std::size_t>(c)] = QK;
        }
    }
    K_set.blocks = std::move(corrected);
    return K_set;
}

Eigen::MatrixXd compute_cosx_k_cell_pair(
    const BasisSet& basis,
    const Eigen::MatrixXd& D,
    const Grid& cosx_grid,
    const CosxCellPairCaches& caches,
    const Eigen::MatrixXd& q_cached,
    const BoysTable* boys_table) {
    const int n_bf = static_cast<int>(D.rows());

    // Cell-diagonal density = the single-block set P = {0 ↦ D}.
    LatticeMatrixSet P_diag;
    P_diag.nbf = n_bf;
    P_diag.cells.push_back(LatticeCell{});  // zero index, zero r_cart
    P_diag.blocks.push_back(D);

    // Raw blocks from the engine; the Γ-folded K is their plain sum
    // (K(Γ) = Σ_g K(g)), then the molecular symmetrise + Q-junction
    // post-processing is applied to the fold.
    LatticeMatrixSet K_set = compute_cosx_k_blocks(
        basis, P_diag, cosx_grid, caches,
        /*q_cached=*/Eigen::MatrixXd(), boys_table);

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_bf, n_bf);
    for (const auto& B : K_set.blocks) K += B;

    // Symmetrise (Neese 2009 §2.3) + overlap-fit Q-junction (§2.4) —
    // identical post-processing to the molecular kernel. The Q
    // corrector is a home-basis + grid invariant; the bra side here is
    // the same home-χ quadrature it was derived for.
    Eigen::MatrixXd K_naive = 0.5 * (K + K.transpose());

    Eigen::MatrixXd Q_local;
    const Eigen::MatrixXd* Q_ptr = &q_cached;
    if (q_cached.size() == 0) {
        Q_local = build_cosx_q(basis, cosx_grid);
        Q_ptr = &Q_local;
    }
    if (Q_ptr->size() == 0) {
        return K_naive;
    }
    return (*Q_ptr) * K_naive * Q_ptr->transpose();
}

}  // namespace vibeqc
