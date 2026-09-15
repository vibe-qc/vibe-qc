#include "vibeqc/cosx_one_center.hpp"

#include <Eigen/Dense>
#include <libint2.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "vibeqc/cosx_kernel.hpp"
#include "vibeqc/init.hpp"

namespace vibeqc {

namespace {

Eigen::MatrixXd build_one_center_eri(
    const std::vector<libint2::Shell>& atom_shells,
    const std::vector<int>& local_shell2bf,
    int na,
    double omega) {

    const int n_sh = static_cast<int>(atom_shells.size());
    const int na2 = na * na;
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(na2, na2);
    if (n_sh == 0) return E;

    int max_nprim = 0, max_l = 0;
    for (const auto& sh : atom_shells) {
        max_nprim = std::max(max_nprim, static_cast<int>(sh.alpha.size()));
        for (const auto& c : sh.contr)
            max_l = std::max(max_l, static_cast<int>(c.l));
    }

    // omega > 0: erfc(omega*r)/r short-range ERIs — the analytic side of
    // the replacement must use the SAME kernel as the quadrature side it
    // substitutes (the range-separated multi-k engine's SR part).
    libint2::Engine engine(
        omega > 0.0 ? libint2::Operator::erfc_coulomb
                    : libint2::Operator::coulomb,
        max_nprim, max_l, 0);
    if (omega > 0.0) {
        using erfc_coul_params = libint2::operator_traits<
            libint2::Operator::erfc_coulomb>::oper_params_type;
        engine.set_params(erfc_coul_params{omega});
    }
    const auto& buf = engine.results();

    for (int s1 = 0; s1 < n_sh; ++s1) {
        const auto& sh1 = atom_shells[s1];
        const int bf1 = local_shell2bf[s1], n1 = static_cast<int>(sh1.size());
        for (int s3 = 0; s3 < n_sh; ++s3) {
            const auto& sh3 = atom_shells[s3];
            const int bf3 = local_shell2bf[s3], n3 = static_cast<int>(sh3.size());
            const int idx13 = s1 * n_sh + s3;
            for (int s2 = 0; s2 < n_sh; ++s2) {
                const auto& sh2 = atom_shells[s2];
                const int bf2 = local_shell2bf[s2], n2 = static_cast<int>(sh2.size());
                const int s4_start = (s1 == s2) ? s3 : 0;
                for (int s4 = s4_start; s4 < n_sh; ++s4) {
                    const int idx24 = s2 * n_sh + s4;
                    if (idx13 < idx24) continue;
                    const auto& sh4 = atom_shells[s4];
                    const int bf4 = local_shell2bf[s4], n4 = static_cast<int>(sh4.size());

                    engine.compute(sh1, sh3, sh2, sh4);
                    const double* block = buf[0];
                    if (block == nullptr) continue;

                    for (int a = 0; a < n1; ++a) {
                        const int imu = bf1 + a;
                        for (int b = 0; b < n3; ++b) {
                            const int ila = bf3 + b;
                            const int row_ml = imu * na + ila;
                            const int row_lm = ila * na + imu;
                            for (int c = 0; c < n2; ++c) {
                                const int inu = bf2 + c;
                                for (int d = 0; d < n4; ++d) {
                                    const int isi = bf4 + d;
                                    const double v = block[
                                        ((a * n3 + b) * n2 + c) * n4 + d];
                                    const int col_ns = inu * na + isi;
                                    const int col_sn = isi * na + inu;
                                    E(row_ml, col_ns) = v;
                                    E(row_lm, col_ns) = v;
                                    E(row_ml, col_sn) = v;
                                    E(row_lm, col_sn) = v;
                                    E(col_ns, row_ml) = v;
                                    E(col_sn, row_ml) = v;
                                    E(col_ns, row_lm) = v;
                                    E(col_sn, row_lm) = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return E;
}

}  // namespace

OneCenterCorrection::OneCenterCorrection(const BasisSet& basis,
                                         double omega)
    : omega_(omega), basis_(&basis) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    n_bf_ = static_cast<int>(basis.nbasis());

    const auto shell_infos = basis.shells();
    const int n_sh = static_cast<int>(shell_infos.size());
    if (n_sh == 0) return;

    int n_atoms = 0;
    for (const auto& si : shell_infos)
        n_atoms = std::max(n_atoms, si.atom_index + 1);

    std::vector<std::vector<int>> sh_by_atom(n_atoms);
    for (int s = 0; s < n_sh; ++s)
        sh_by_atom[shell_infos[s].atom_index].push_back(s);

    atom_bfs_.resize(n_atoms);
    atom_shells_.resize(n_atoms);
    eri_blocks_.resize(n_atoms);

    for (int a = 0; a < n_atoms; ++a) {
        const auto& sh_idx = sh_by_atom[a];
        if (sh_idx.empty()) continue;
        atom_shells_[a] = sh_idx;

        int na = 0;
        for (int si : sh_idx) na += static_cast<int>(shells[si].size());

        std::vector<int> bf_list;
        bf_list.reserve(na);
        for (int si : sh_idx) {
            const int bf0 = shell2bf[si];
            const int ns = static_cast<int>(shells[si].size());
            for (int i = 0; i < ns; ++i) bf_list.push_back(bf0 + i);
        }
        atom_bfs_[a] = std::move(bf_list);

        std::vector<libint2::Shell> atom_sh;
        atom_sh.reserve(sh_idx.size());
        for (int si : sh_idx) atom_sh.push_back(shells[si]);

        std::vector<int> lcl_s2bf(sh_idx.size());
        int cnt = 0;
        for (std::size_t i = 0; i < sh_idx.size(); ++i) {
            lcl_s2bf[i] = cnt;
            cnt += static_cast<int>(atom_sh[i].size());
        }
        eri_blocks_[a] = build_one_center_eri(atom_sh, lcl_s2bf, na,
                                              omega_);
    }
}

Eigen::MatrixXd OneCenterCorrection::apply(
    const Eigen::MatrixXd& D,
    const GridBatches& grid_batches,
    const BoysTable& boys,
    const PrimitivePairCache& pp_cache,
    const std::vector<double>& shell_cutoffs) const {

    const int n_atoms = static_cast<int>(atom_bfs_.size());
    Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(n_bf_, n_bf_);
    if (n_bf_ == 0) return correction;

    const auto& shells = basis_->libint();
    const std::size_t n_total_sh = shells.size();

    for (int a = 0; a < n_atoms; ++a) {
        const auto& bf = atom_bfs_[a];
        const auto& sh_idx = atom_shells_[a];
        const int na = static_cast<int>(bf.size());
        const int n_sh_a = static_cast<int>(sh_idx.size());
        if (na == 0 || n_sh_a == 0) continue;

        // ---- Exact K from stored ERI ---------------------------------
        const Eigen::MatrixXd& E = eri_blocks_[a];
        Eigen::MatrixXd D_sub(na, na);
        for (int i = 0; i < na; ++i)
            for (int j = 0; j < na; ++j)
                D_sub(i, j) = D(bf[i], bf[j]);

        Eigen::MatrixXd K_exact = Eigen::MatrixXd::Zero(na, na);
        for (int mu = 0; mu < na; ++mu)
            for (int nu = 0; nu < na; ++nu) {
                double acc = 0.0;
                for (int la = 0; la < na; ++la)
                    for (int si = 0; si < na; ++si)
                        acc += D_sub(la, si)
                             * E(mu * na + la, nu * na + si);
                K_exact(mu, nu) = acc;
            }

        // ---- COSX K restricted to atom A (interior only) --------------
        // Local shell-to-bf offset map (0..na-1).
        std::vector<int> lcl_bf0(n_sh_a);
        {
            int off = 0;
            for (int i = 0; i < n_sh_a; ++i) {
                lcl_bf0[i] = off;
                off += static_cast<int>(shells[sh_idx[i]].size());
            }
        }

        int max_l_a = 0;
        for (int si : sh_idx)
            for (const auto& c : shells[si].contr)
                max_l_a = std::max(max_l_a, static_cast<int>(c.l));
        CosxKernelWorkspace ws;
        ws.reserve(max_l_a, max_l_a);
        const int max_bf = (max_l_a + 1) * (max_l_a + 2) / 2;
        std::vector<double> kernel_block(
            static_cast<std::size_t>(max_bf) * max_bf);

        // Map from full basis index to atom-local index.
        std::vector<int> full_to_lcl(n_bf_, -1);
        for (int i = 0; i < na; ++i) full_to_lcl[bf[i]] = i;

        Eigen::MatrixXd K_cosx_atom = Eigen::MatrixXd::Zero(na, na);

        for (const auto& batch : grid_batches.batches) {
            const int n_p = batch.n_points;
            for (int gp = 0; gp < n_p; ++gp) {
                const double w_g = batch.weights(gp);
                if (w_g == 0.0) continue;

                const double rx = batch.points(gp, 0);
                const double ry = batch.points(gp, 1);
                const double rz = batch.points(gp, 2);

                // Quick reject: is this point within any cutoff?
                bool near = false;
                for (int si : sh_idx) {
                    const auto& O = shells[si].O;
                    const double dx = O[0] - rx, dy = O[1] - ry,
                                 dz = O[2] - rz;
                    if (std::sqrt(dx*dx + dy*dy + dz*dz)
                        < shell_cutoffs[si]) { near = true; break; }
                }
                if (!near) continue;

                // Gather chi values for the atom's BFs from the
                // GridBatches chi_primary cache.
                Eigen::VectorXd chi_a = Eigen::VectorXd::Zero(na);
                {
                    const int n_prim =
                        static_cast<int>(batch.primary_bfs.size());
                    for (int c = 0; c < n_prim; ++c) {
                        const int fb = batch.primary_bfs[c];
                        const int lb = full_to_lcl[fb];
                        if (lb >= 0)
                            chi_a(lb) = batch.chi_primary(gp, c);
                    }
                }
                const double chi_max = chi_a.cwiseAbs().maxCoeff();
                if (chi_max < 1e-7) continue;

                Eigen::VectorXd Dchi_a = D_sub * chi_a;

                Eigen::VectorXd F_g = Eigen::VectorXd::Zero(na);
                for (int si = 0; si < n_sh_a; ++si) {
                    const int s1 = sh_idx[si];
                    const auto& sh1 = shells[s1];
                    const int bf1 = lcl_bf0[si], n1 =
                        static_cast<int>(sh1.size());
                    for (int sj = 0; sj <= si; ++sj) {
                        const int s2 = sh_idx[sj];
                        const auto& sh2 = shells[s2];
                        const int bf2 = lcl_bf0[sj], n2 =
                            static_cast<int>(sh2.size());
                        const double dmax = std::max(
                            Dchi_a.segment(bf2, n2).cwiseAbs().maxCoeff(),
                            Dchi_a.segment(bf1, n1).cwiseAbs().maxCoeff());
                        if (dmax < 1e-7) continue;

                        const std::array<double,3> C={rx,ry,rz};
                        const auto& pd =
                            pp_cache.pairs[s1*n_total_sh + s2];
                        cosx_nuclear_pair_into(
                            sh1, sh2, pd, C, boys, ws,
                            kernel_block.data(), omega_);
                        Eigen::Map<const Eigen::Matrix<double,
                            Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
                            A(kernel_block.data(), n1, n2);
                        F_g.segment(bf1, n1).noalias() +=
                            A * Dchi_a.segment(bf2, n2);
                        if (s1 != s2)
                            F_g.segment(bf2, n2).noalias() +=
                                A.transpose()*Dchi_a.segment(bf1, n1);
                    }
                }

                for (int mu = 0; mu < na; ++mu) {
                    const double cmu = chi_a(mu);
                    if (std::abs(cmu) < 1e-7) continue;
                    K_cosx_atom.row(mu).noalias() +=
                        (w_g * cmu) * F_g.transpose();
                }
            }
        }

        // Correction for atom A.
        for (int i = 0; i < na; ++i)
            for (int j = 0; j < na; ++j)
                correction(bf[i], bf[j]) =
                    K_exact(i, j) - K_cosx_atom(i, j);
    }
    return correction;
}

}  // namespace vibeqc
