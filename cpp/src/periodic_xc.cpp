#include "vibeqc/periodic_xc.hpp"

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/atom.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace vibeqc {

// Value/Fock workspace includes active AO tables, concurrent contractions,
// point vectors and shifted grids. Output lattice matrices, input density,
// pair metadata and external-provider full-grid fields are reserved separately.
std::size_t periodic_xc_value_batch_size(
    std::size_t nbf, std::size_t n_active, bool need_grad,
    bool open_shell, std::size_t n_threads) {
    if (nbf == 0 || n_active == 0 || n_threads == 0)
        throw std::invalid_argument("periodic XC value batch dimensions must be positive");
    const long double basis = nbf, threads = n_threads;
    const long double tables = need_grad ? 4 : 1;
    const long double per_point = sizeof(double) * (
        tables * n_active * basis + threads * (6 * basis + 20) + 64);
    const long double fixed = sizeof(double) * threads * basis * basis
        * (open_shell ? 6 : 4);
    constexpr std::size_t workspace_bytes = 256ULL * 1024 * 1024;
    if (!std::isfinite(fixed + per_point) || fixed + per_point > workspace_bytes)
        throw std::runtime_error("periodic XC value cannot fit one grid point in its workspace");
    return std::min<std::size_t>(4096, (workspace_bytes - fixed) / per_point);
}


namespace {

// Shell → atom map for the home-cell basis (mirrors gradient.cpp).
std::vector<std::vector<int>> atom_bf_indices(const BasisSet& basis,
                                              const Molecule& mol) {
    std::vector<libint2::Atom> la;
    la.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        libint2::Atom x;
        x.atomic_number = a.Z;
        x.x = a.xyz[0];
        x.y = a.xyz[1];
        x.z = a.xyz[2];
        la.push_back(x);
    }
    const auto s2a = basis.libint().shell2atom(la);
    const auto shell2bf = basis.libint().shell2bf();
    const auto& shells = basis.libint();
    std::vector<std::vector<int>> atom_bf(mol.atoms().size());
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const int A = static_cast<int>(s2a[s]);
        const int bf0 = static_cast<int>(shell2bf[s]);
        const int bf1 = bf0 + static_cast<int>(shells[s].size());
        for (int bf = bf0; bf < bf1; ++bf) atom_bf[A].push_back(bf);
    }
    return atom_bf;
}

int hessian_component(int a, int b) {
    if (a > b) std::swap(a, b);
    static constexpr int idx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return idx[a][b];
}

// Enforce the von Weizsacker lower bound on the kinetic energy density,
// tau >= tau_W = |grad rho|^2 / (8 rho) = sigma / (8 rho). tau = 1/2 sum_i
// |grad psi_i|^2 is positive-definite and bounded below by tau_W (the exact
// single-orbital value); a real density always satisfies it. The periodic
// CROSS-CELL tau assembly (sum over image cells, with signed density-matrix
// blocks) can produce tau slightly BELOW tau_W -- even negative -- from
// roundoff at diffuse grid points. Unregularised meta-GGAs (TPSS, M06-L, ...)
// are SINGULAR there: libxc returns v_sigma / v_tau ~ 1e16, which contaminates
// the V_xc Fock's virtual subspace (eigenvalues ~1e8) and can drive the SCF to
// a spurious stationary point, while the energy integral stays ~correct (the
// offending points carry negligible exc*w). SCAN/r2SCAN self-regularise and
// were unaffected. Clamping tau to tau_W restores the physical input at exactly
// those points and is a no-op everywhere tau >= tau_W already, so it does not
// perturb the converged energy -- it is the energy-consistent root-cause fix,
// applied identically in the energy, the SCF Fock, and the analytic gradient so
// FD and analytic forces stay synchronised. Clamping tau to 0 is NOT enough
// (libxc still diverges at tau=0 with finite sigma); the bound must be tau_W.
inline void enforce_von_weizsaecker_floor(Eigen::VectorXd& tau,
                                          const Eigen::VectorXd& rho,
                                          const Eigen::VectorXd& sigma) {
    const Eigen::Index n = tau.size();
    for (Eigen::Index i = 0; i < n; ++i) {
        if (rho[i] <= 0.0) {
            tau[i] = 0.0;
        } else {
            const double tau_w = sigma[i] / (8.0 * rho[i]);
            if (tau[i] < tau_w) tau[i] = tau_w;
        }
    }
}

// Map a lattice-cell triple (n1,n2,n3) -> its position in the cell array, so
// a bra/ket cell pair (a, s) can resolve the density-matrix block P(s−a).
std::map<std::array<int, 3>, int> build_cell_index_map(
    const std::vector<LatticeCell>& cells) {
    std::map<std::array<int, 3>, int> m;
    for (std::size_t c = 0; c < cells.size(); ++c) {
        const Eigen::Vector3i& idx = cells[c].index;
        m[{idx[0], idx[1], idx[2]}] = static_cast<int>(c);
    }
    return m;
}

// Active bra/ket cells for the cross-cell density sum: cells whose image
// atoms come within `radius` of a home-cell atom, so their lattice-shifted
// AOs overlap the home-atom Becke grid region. The home cell (g=0) is always
// active. Cells beyond `radius` contribute only at far, vanishing-weight grid
// points and are excluded by the same real-space cutoff the S/T/Fock lattice
// sums use (`opts.cutoff_bohr`). The MgO/STO-3G LDA E_xc(radius) study
// converges to < 0.1 mHa by ~12 bohr, well inside the production 14-15 bohr
// cutoff. The direct regression lives in tests/test_periodic_xc_cross_cell.py.
std::vector<int> active_density_cells(const std::vector<LatticeCell>& cells,
                                      const PeriodicSystem& system,
                                      double radius) {
    const auto& atoms = system.unit_cell;
    const double r2 = radius * radius;
    std::vector<int> active;
    active.reserve(cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c) {
        if (cells[c].index.isZero()) {
            active.push_back(static_cast<int>(c));
            continue;
        }
        const Eigen::Vector3d& R = cells[c].r_cart;
        bool near = false;
        for (const auto& a : atoms) {
            const Eigen::Vector3d pa(a.xyz[0] + R[0], a.xyz[1] + R[1],
                                     a.xyz[2] + R[2]);
            for (const auto& b : atoms) {
                const Eigen::Vector3d pb(b.xyz[0], b.xyz[1], b.xyz[2]);
                if ((pa - pb).squaredNorm() <= r2) { near = true; break; }
            }
            if (near) break;
        }
        if (near) active.push_back(static_cast<int>(c));
    }
    return active;
}

// Bra/ket cell pair whose lattice difference s−a resolves to density cell gp.
struct BraKetPair {
    int a;   // bra cell position
    int s;   // ket cell position
    int gp;  // density / V_xc cell position for g = idx(s) − idx(a)
};

// Enumerate the (bra, ket) pairs whose lattice difference s−a is a valid
// density cell. `bra_cells` is the home cell only for a molecular-limit
// density (no cross-cell blocks) or the full active set for a genuine periodic
// density (P(g≠0) populated); `ket_cells` is always the active set.
std::vector<BraKetPair> build_braket_pairs(
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& bra_cells,
    const std::vector<int>& ket_cells,
    const std::map<std::array<int, 3>, int>& cell_pos,
    bool require_difference_closure = false) {
    std::vector<BraKetPair> pairs;
    pairs.reserve(bra_cells.size() * ket_cells.size());
    for (int a : bra_cells) {
        const Eigen::Vector3i ia = cells[a].index;
        for (int s : ket_cells) {
            const Eigen::Vector3i is = cells[s].index;
            const std::array<int, 3> g = {is[0] - ia[0], is[1] - ia[1],
                                          is[2] - ia[2]};
            auto it = cell_pos.find(g);
            if (it == cell_pos.end()) {
                if (require_difference_closure) {
                    throw std::invalid_argument(
                        "build_xc_periodic: explicit periodic-lattice "
                        "density domain is not closed over active AO-image "
                        "differences");
                }
                continue;
            }
            pairs.push_back(BraKetPair{a, s, it->second});
        }
    }
    return pairs;
}

// True iff any non-home density block is non-negligible. A density with only
// the home (g=0) block populated is a molecular-limit density (the GDF Γ-cell
// convention, a Γ-only SCF, etc.): its physical grid density is the home-cell
// form χ_0 P(0) χ_0, NOT the cross-cell (bra-image) replication — feeding the
// latter a P(0)-only density over-counts (replicated density on the molecular
// Becke grid). The cross-cell sum is the physical periodic density only when
// the full real-space fold P(g≠0) is present (multi-k BIPOLE).
bool density_has_cross_cell_blocks(const LatticeMatrixSet& P, double tol) {
    for (std::size_t c = 0; c < P.cells.size(); ++c) {
        if (P.cells[c].index.isZero()) continue;
        if (P.blocks[c].size() != 0 &&
            P.blocks[c].cwiseAbs().maxCoeff() > tol) {
            return true;
        }
    }
    return false;
}

bool use_periodic_density_bras(PeriodicXCDensityDomain density_domain,
                               bool auto_detected_periodic) {
    switch (density_domain) {
        case PeriodicXCDensityDomain::AUTO:
            return auto_detected_periodic;
        case PeriodicXCDensityDomain::MOLECULAR_HOME:
            return false;
        case PeriodicXCDensityDomain::PERIODIC_LATTICE:
            return true;
    }
    throw std::invalid_argument(
        "build_xc_periodic: invalid periodic XC density domain");
}

// Home-cell position in `active` (always present — active_density_cells pushes
// the g=0 cell first).
std::vector<int> home_only_bra(const std::vector<LatticeCell>& cells,
                               const std::vector<int>& active) {
    for (int c : active) {
        if (cells[c].index.isZero()) return {c};
    }
    return active.empty() ? std::vector<int>{} : std::vector<int>{active.front()};
}

// Evaluate lattice-shifted AO values (and gradients when `need_grad`) for the
// active cells only, via SHIFTED GRID POINTS:  χ_c(r) = χ_μ(r − R_c) is
// obtained by evaluating the home basis at r − R_c — identical numerics to a
// basis recentred at +R_c, but without rebuilding a BasisSet per cell. Fills
// chi[c] / dchi[c] for c in `active`; other entries stay empty.
void evaluate_active_shifted_ao(const BasisSet& basis,
                                const Eigen::MatrixX3d& points,
                                const std::vector<LatticeCell>& cells,
                                const std::vector<int>& active, bool need_grad,
                                std::vector<Eigen::MatrixXd>& chi,
                                std::vector<std::array<Eigen::MatrixXd, 3>>& dchi) {
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t ai = 0; ai < active.size(); ++ai) {
        const int c = active[ai];
        const Eigen::Vector3d& R = cells[c].r_cart;
        if (R.isZero()) {
            if (need_grad) {
                AOValues v = evaluate_ao_with_gradient(basis, points);
                chi[c] = std::move(v.values);
                dchi[c] = std::move(v.gradients);
            } else {
                chi[c] = evaluate_ao(basis, points);
            }
            continue;
        }
        Eigen::MatrixX3d pts = points;
        pts.rowwise() -= R.transpose();
        if (need_grad) {
            AOValues v = evaluate_ao_with_gradient(basis, pts);
            chi[c] = std::move(v.values);
            dchi[c] = std::move(v.gradients);
        } else {
            chi[c] = evaluate_ao(basis, pts);
        }
    }
}

struct ExternalPeriodicFields {
    Eigen::VectorXd rho;
    Eigen::MatrixX3d grad;
    Eigen::VectorXd tau;
};

// Assemble one spin channel on a grid slice from the same cross-cell
// bra/ket convention used by the libxc path.  External nonlocal XC needs the
// complete vectors, so callers repeat this bounded-AO operation over slices
// before invoking the provider once.
ExternalPeriodicFields build_external_periodic_fields_batch(
    const LatticeMatrixSet& density,
    const std::vector<BraKetPair>& pairs,
    const std::vector<Eigen::MatrixXd>& chi_h,
    const std::vector<std::array<Eigen::MatrixXd, 3>>& dchi_h,
    Eigen::Index n_points) {
    ExternalPeriodicFields fields;
    fields.rho = Eigen::VectorXd::Zero(n_points);
    fields.grad = Eigen::MatrixX3d::Zero(n_points, 3);
    fields.tau = Eigen::VectorXd::Zero(n_points);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::VectorXd> rho_t(
        n_threads, Eigen::VectorXd::Zero(n_points));
    std::array<std::vector<Eigen::VectorXd>, 3> grad_t;
    for (int c = 0; c < 3; ++c) {
        grad_t[c].assign(n_threads, Eigen::VectorXd::Zero(n_points));
    }
    std::vector<Eigen::VectorXd> tau_t(
        n_threads, Eigen::VectorXd::Zero(n_points));

    // schedule(static), not dynamic: each thread sums its pairs into a
    // private accumulator and the partials are added in thread order
    // below. A dynamic partition changes which pairs land in which
    // partial from run to run, and with it the rounding of the total, so
    // the same density gave XC energies scattered at 1e-9 (#81). Every
    // pair costs one chi_a * P product, so a static partition loses no
    // balance and the result is reproducible for a fixed thread count.
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        const int tid = omp_thread_index();
        const BraKetPair& pr = pairs[k];
        const Eigen::MatrixXd& chi_a = chi_h[pr.a];
        const Eigen::MatrixXd& chi_s = chi_h[pr.s];
        const Eigen::MatrixXd& P = density.blocks[pr.gp];
        const Eigen::MatrixXd chi_aP = chi_a * P;
        rho_t[tid].array() +=
            (chi_aP.array() * chi_s.array()).rowwise().sum();
        const auto& da = dchi_h[pr.a];
        const auto& ds = dchi_h[pr.s];
        for (int c = 0; c < 3; ++c) {
            grad_t[c][tid].array() +=
                (chi_aP.array() * ds[c].array()).rowwise().sum();
            grad_t[c][tid].array() +=
                ((da[c] * P).array() * chi_s.array()).rowwise().sum();
            tau_t[tid].array() +=
                ((da[c] * P).array() * ds[c].array()).rowwise().sum();
        }
    }
    for (int t = 0; t < n_threads; ++t) {
        fields.rho += rho_t[t];
        fields.tau += tau_t[t];
        for (int c = 0; c < 3; ++c) {
            fields.grad.col(c) += grad_t[c][t];
        }
    }
    fields.tau *= 0.5;
    return fields;
}

void project_external_periodic_batch(
    const std::vector<BraKetPair>& pairs,
    const std::vector<std::vector<int>>& pairs_by_gp,
    const std::vector<Eigen::MatrixXd>& chi_h,
    const std::vector<std::array<Eigen::MatrixXd, 3>>& dchi_h,
    const Eigen::VectorXd& q_rho,
    const Eigen::MatrixX3d& q_grad,
    const Eigen::VectorXd& q_tau,
    std::vector<Eigen::MatrixXd>& output_blocks) {
    const int nbf = output_blocks.empty()
        ? 0 : static_cast<int>(output_blocks.front().rows());
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t gp = 0; gp < output_blocks.size(); ++gp) {
        if (pairs_by_gp[gp].empty()) continue;
        Eigen::MatrixXd Vg = Eigen::MatrixXd::Zero(nbf, nbf);
        for (int k : pairs_by_gp[gp]) {
            const BraKetPair& pr = pairs[static_cast<std::size_t>(k)];
            const Eigen::MatrixXd& chi_a = chi_h[pr.a];
            const Eigen::MatrixXd& chi_s = chi_h[pr.s];
            Vg.noalias() +=
                chi_a.transpose() * q_rho.asDiagonal() * chi_s;
            const auto& da = dchi_h[pr.a];
            const auto& ds = dchi_h[pr.s];
            for (int c = 0; c < 3; ++c) {
                const Eigen::VectorXd qg = q_grad.col(c);
                Vg.noalias() +=
                    da[c].transpose() * qg.asDiagonal() * chi_s;
                Vg.noalias() +=
                    chi_a.transpose() * qg.asDiagonal() * ds[c];
                const Eigen::VectorXd qt = 0.5 * q_tau;
                Vg.noalias() +=
                    da[c].transpose() * qt.asDiagonal() * ds[c];
            }
        }
        output_blocks[gp].noalias() += Vg;
    }
}

}  // namespace

std::array<std::size_t, 2> periodic_xc_domain_counts(
    const PeriodicSystem& system, const std::vector<LatticeCell>& cells,
    const LatticeSumOptions& opts, PeriodicXCDensityDomain density_domain) {
    const auto active = active_density_cells(cells, system, opts.cutoff_bohr);
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr)
        : opts.cutoff_bohr;
    const auto bras = use_periodic_density_bras(density_domain, true)
        ? active_density_cells(cells, system, bra_radius)
        : home_only_bra(cells, active);
    return {active.size(), bras.size()};
}

PeriodicXCContribution build_xc_periodic(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const Grid& grid,
                                         const Functional& func,
                                         const LatticeMatrixSet& P_real_space,
                                         const LatticeSumOptions& opts,
                                         PeriodicXCDensityDomain density_domain) {
    const int nbf = static_cast<int>(basis.nbasis());
    if (P_real_space.nbf != nbf) {
        throw std::runtime_error("build_xc_periodic: density nbf mismatch");
    }
    const bool is_gga  = (func.kind() == XCKind::GGA);
    const bool is_mgga = (func.kind() == XCKind::MGGA);
    const bool need_grad = is_gga || is_mgga;

    const std::size_t n_cells = P_real_space.cells.size();
    const auto n_pts = static_cast<Eigen::Index>(grid.points.rows());

    // --- Cross-cell density: BOTH AOs range over lattice cells ------------
    //
    // The physical periodic density on the home-atom Becke grid is the full
    // crystal density, summed over BOTH a bra cell `a` and a ket cell `s`,
    // with the real-space density matrix taken at their difference g = s − a:
    //
    //   ρ(r) = Σ_a Σ_s χ_a(r) · P(s−a) · χ_s(r)          (χ_c(r) ≡ χ(r − R_c))
    //
    // The shipped code summed only the home-bra slice  ρ(r)=Σ_s χ_0 P(s) χ_s,
    // which is NOT lattice-periodic (probe: ρ_code(r+a)−ρ_code(r) ≈ 18 e/bohr³
    // on MgO) and pointwise wrong by O(1) in the bonding region of dense
    // crystals — while STILL integrating to N electrons, so ∫ρ dr cannot
    // see it. That omission of the absolute lattice translation (bra in an
    // image cell) is the +428 mHa MgO/STO-3G LDA error vs sealed CRYSTAL23
    // (−270.49836635 Ha/FU) and PySCF.pbc GDF (−270.49937 Ha/FU), both ~1 mHa
    // apart, while the shipped code gave −270.07046. The PBE / r²SCAN siblings
    // shared the same defect (~+399 / +384 mHa)
    // because LDA is ρ-only — the error is in the density assembly every
    // functional shares, not the σ / τ Fock terms.
    //
    // Bra and ket cells are restricted to those whose lattice-shifted AOs
    // overlap the home-atom grid region (active_density_cells, governed by
    // opts.cutoff_bohr — the same converged real-space cutoff the S/T/Fock
    // lattice sums use; MgO/STO-3G LDA E_xc converges < 0.1 mHa by ~12 bohr).
    const auto cell_pos = build_cell_index_map(P_real_space.cells);
    // Ket / AO-evaluation set: cells within the density cutoff.
    const std::vector<int> active =
        active_density_cells(P_real_space.cells, system, opts.cutoff_bohr);
    // Bra (image) cells: every lattice translation whose shifted AOs still
    // reach the home-cell grid, i.e. the same AO-reach radius as the kets
    // (opts.cutoff_bohr), unless a driver narrowed it explicitly through
    // LatticeSumOptions::becke_image_radius_bohr (then capped at cutoff_bohr
    // so the bra set stays inside the AO-evaluated `active` set). The density
    // at a grid point is a property of the point, not of the Becke partition:
    // the partition only decides which atom's weight the point carries, so
    // stopping the bra sum at the partition reach drops real density. With
    // the old fixed 10-bohr fallback the Li/STO-3G 6-bohr cell lost 0.4 % of
    // its density to the diffuse 2sp shell (exponent 0.048) and E_xc came out
    // 8 mHa short against PySCF (#265). Only summed over the image bra cells
    // for a genuine periodic density (P(g≠0) populated); a molecular-limit
    // density (home block only) keeps the home-cell bra so ρ = χ_0 P(0) χ_0.
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr)
        : opts.cutoff_bohr;
    const std::vector<int> bra_active =
        active_density_cells(P_real_space.cells, system, bra_radius);
    const bool use_periodic_bras = use_periodic_density_bras(
        density_domain,
        density_has_cross_cell_blocks(P_real_space, 1e-12));
    const std::vector<int> bra_cells = use_periodic_bras
        ? bra_active
        : home_only_bra(P_real_space.cells, active);
    const std::vector<BraKetPair> pairs =
        build_braket_pairs(
            P_real_space.cells, bra_cells, active, cell_pos,
            density_domain == PeriodicXCDensityDomain::PERIODIC_LATTICE);

    // ---- Output accumulators (filled incrementally over grid batches) -----
    PeriodicXCContribution out;
    out.V_xc.nbf = nbf;
    out.V_xc.cells = P_real_space.cells;
    out.V_xc.blocks.assign(n_cells, Eigen::MatrixXd::Zero(nbf, nbf));
    double e_xc = 0.0;

    // Output cell gp -> the bra/ket pairs that map to it (batch-independent).
    std::vector<std::vector<int>> pairs_by_gp(n_cells);
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        pairs_by_gp[pairs[k].gp].push_back(static_cast<int>(k));
    }

    if (func.is_external()) {
        if (grid.atomic_weights.size() != n_pts
            || static_cast<Eigen::Index>(grid.atom_of_point.size()) != n_pts
            || grid.atom_coords.cols() != 3) {
            throw std::invalid_argument(
                "external periodic XC requires atom-major grid ownership, "
                "raw atomic weights, and atom coordinates");
        }
        ExternalXCInput input;
        input.functional = func.name();
        input.grid_profile = atomic_grid_profile_name(grid.atomic_grid_profile);
        input.points = grid.points;
        input.grid_weights = grid.weights;
        input.atomic_grid_weights = grid.atomic_weights;
        input.atom_of_point = grid.atom_of_point;
        input.atom_coords = grid.atom_coords;
        input.atomic_numbers = grid.atomic_numbers;
        input.periodic = true;
        input.periodic_dimension = system.dim;
        input.lattice = system.lattice;
        input.rho_alpha.resize(n_pts);
        input.rho_beta.resize(n_pts);
        input.grad_alpha.resize(n_pts, 3);
        input.grad_beta.resize(n_pts, 3);
        input.tau_alpha.resize(n_pts);
        input.tau_beta.resize(n_pts);

        const Eigen::Index kExternalXcGridBatch = static_cast<Eigen::Index>(
            periodic_xc_value_batch_size(nbf, active.size(), true,
                                         false, omp_max_threads()));
        for (Eigen::Index g0 = 0; g0 < n_pts;
             g0 += kExternalXcGridBatch) {
            const Eigen::Index nb = std::min<Eigen::Index>(
                kExternalXcGridBatch, n_pts - g0);
            const Eigen::MatrixX3d pts_b = grid.points.middleRows(g0, nb);
            std::vector<Eigen::MatrixXd> chi_h(n_cells);
            std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
            evaluate_active_shifted_ao(
                basis, pts_b, P_real_space.cells, active,
                /*need_grad=*/true, chi_h, dchi_h);
            const ExternalPeriodicFields fields =
                build_external_periodic_fields_batch(
                    P_real_space, pairs, chi_h, dchi_h, nb);
            input.rho_alpha.segment(g0, nb) = 0.5 * fields.rho;
            input.rho_beta.segment(g0, nb) = 0.5 * fields.rho;
            input.grad_alpha.middleRows(g0, nb) = 0.5 * fields.grad;
            input.grad_beta.middleRows(g0, nb) = 0.5 * fields.grad;
            // Keep the raw linear tau feature; see the spin-resolved path
            // below for why external providers do not inherit libxc's
            // tau_W regularization.
            input.tau_alpha.segment(g0, nb) = 0.5 * fields.tau;
            input.tau_beta.segment(g0, nb) = 0.5 * fields.tau;
        }

        const ExternalXCEvaluation ev = func.eval_external(input);
        const Eigen::VectorXd q_rho =
            0.5 * (ev.v_rho_alpha + ev.v_rho_beta);
        const Eigen::MatrixX3d q_grad =
            0.5 * (ev.v_grad_alpha + ev.v_grad_beta);
        const Eigen::VectorXd q_tau =
            0.5 * (ev.v_tau_alpha + ev.v_tau_beta);
        for (Eigen::Index g0 = 0; g0 < n_pts;
             g0 += kExternalXcGridBatch) {
            const Eigen::Index nb = std::min<Eigen::Index>(
                kExternalXcGridBatch, n_pts - g0);
            const Eigen::MatrixX3d pts_b = grid.points.middleRows(g0, nb);
            std::vector<Eigen::MatrixXd> chi_h(n_cells);
            std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
            evaluate_active_shifted_ao(
                basis, pts_b, P_real_space.cells, active,
                /*need_grad=*/true, chi_h, dchi_h);
            project_external_periodic_batch(
                pairs, pairs_by_gp, chi_h, dchi_h,
                q_rho.segment(g0, nb), q_grad.middleRows(g0, nb),
                q_tau.segment(g0, nb), out.V_xc.blocks);
        }
        out.e_xc = ev.energy;
        return out;
    }

    // ---- Grid-batched XC quadrature --------------------------------------
    // The shifted-AO tables chi_h / dchi_h for the active cells are the
    // dominant allocation -- O(n_active * n_pts * nbf), tens of GB on an
    // image-rich slab. libxc is pointwise, so the whole rho -> functional ->
    // V_xc(g) chain factorises over disjoint grid-point batches: each batch
    // evaluates the active-cell AOs on its own points, builds rho/grad/tau,
    // evaluates the functional, and accumulates e_xc and V_xc(g). Peak AO
    // memory drops to O(n_active * batch * nbf). Per-batch math is identical
    // to the whole-grid path (libxc is pointwise; e_xc and V_xc are sums over
    // grid points), so the result matches to floating-point reassociation.
    const Eigen::Index kXcGridBatch = static_cast<Eigen::Index>(
            periodic_xc_value_batch_size(nbf, active.size(), need_grad,
                                         false, omp_max_threads()));
    for (Eigen::Index g0 = 0; g0 < n_pts; g0 += kXcGridBatch) {
        const Eigen::Index nb = std::min(kXcGridBatch, n_pts - g0);
        const Eigen::MatrixX3d pts_b = grid.points.middleRows(g0, nb);
        const Eigen::VectorXd w_b = grid.weights.segment(g0, nb);

        // Shifted AO values (and gradients) for the active cells, batch points.
        std::vector<Eigen::MatrixXd> chi_h(n_cells);
        std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
        evaluate_active_shifted_ao(basis, pts_b, P_real_space.cells, active,
                                   need_grad, chi_h, dchi_h);

        Eigen::VectorXd rho = Eigen::VectorXd::Zero(nb);
        Eigen::VectorXd gx, gy, gz, tau;
        if (need_grad) {
            gx = Eigen::VectorXd::Zero(nb);
            gy = Eigen::VectorXd::Zero(nb);
            gz = Eigen::VectorXd::Zero(nb);
        }
        if (is_mgga) {
            tau = Eigen::VectorXd::Zero(nb);
        }

        // Accumulate ρ / ∇ρ / τ over all active bra–ket pairs. Each thread
        // owns private accumulators; a final reduction sums them (no races).
        {
            const int n_threads = omp_max_threads();
            std::vector<Eigen::VectorXd> rho_t(
                n_threads, Eigen::VectorXd::Zero(nb));
            std::vector<Eigen::VectorXd> gx_t, gy_t, gz_t, tau_t;
            if (need_grad) {
                gx_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gy_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gz_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
            }
            if (is_mgga) {
                tau_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
            }

            // schedule(static), not dynamic: each thread sums its pairs into a
            // private accumulator and the partials are added in thread order
            // below. A dynamic partition changes which pairs land in which
            // partial from run to run, and with it the rounding of the total, so
            // the same density gave XC energies scattered at 1e-9 (#81). Every
            // pair costs one chi_a * P product, so a static partition loses no
            // balance and the result is reproducible for a fixed thread count.
            #pragma omp parallel for schedule(static)
            for (std::size_t k = 0; k < pairs.size(); ++k) {
                const int tid = omp_thread_index();
                const BraKetPair& pr = pairs[k];
                const Eigen::MatrixXd& chi_a = chi_h[pr.a];   // bra χ(r − R_a)
                const Eigen::MatrixXd& chi_s = chi_h[pr.s];   // ket χ(r − R_s)
                const Eigen::MatrixXd& P = P_real_space.blocks[pr.gp];
                const Eigen::MatrixXd chi_aP = chi_a * P;     // (nb, nbf)
                // ρ(r) += Σ_{μν} χ_a(r)_μ P(s−a)_{μν} χ_s(r)_ν
                rho_t[tid].array() +=
                    (chi_aP.array() * chi_s.array()).rowwise().sum();

                if (need_grad) {
                    // ∂ρ/∂α = Σ_{μν} P_{μν}[∂χ_a,μ χ_s,ν + χ_a,μ ∂χ_s,ν]
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    gx_t[tid].array() +=
                        (chi_aP.array() * ds[0].array()).rowwise().sum();
                    gy_t[tid].array() +=
                        (chi_aP.array() * ds[1].array()).rowwise().sum();
                    gz_t[tid].array() +=
                        (chi_aP.array() * ds[2].array()).rowwise().sum();
                    gx_t[tid].array() +=
                        ((da[0] * P).array() * chi_s.array()).rowwise().sum();
                    gy_t[tid].array() +=
                        ((da[1] * P).array() * chi_s.array()).rowwise().sum();
                    gz_t[tid].array() +=
                        ((da[2] * P).array() * chi_s.array()).rowwise().sum();
                }

                if (is_mgga) {
                    // τ(r) += ½ Σ_d Σ_{μν} P_{μν} ∂_dχ_a,μ ∂_dχ_s,ν; ×½ after.
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    for (int d = 0; d < 3; ++d) {
                        tau_t[tid].array() +=
                            ((da[d] * P).array() * ds[d].array())
                                .rowwise().sum();
                    }
                }
            }

            // Reduce per-thread accumulators.
            for (int t = 0; t < n_threads; ++t) {
                rho += rho_t[t];
                if (need_grad) {
                    gx += gx_t[t];
                    gy += gy_t[t];
                    gz += gz_t[t];
                }
                if (is_mgga) {
                    tau += tau_t[t];
                }
            }
        }
        if (is_mgga) {
            tau *= 0.5;   // match rks.cpp:124
        }

        // Numerical hygiene: tiny negative densities from roundoff break libxc.
        for (Eigen::Index i = 0; i < nb; ++i) {
            if (rho[i] < 0.0) rho[i] = 0.0;
        }

        Eigen::VectorXd sigma;
        if (need_grad) {
            sigma = gx.array().square() + gy.array().square()
                  + gz.array().square();
        } else {
            sigma.resize(0);
        }

        // tau >= tau_W (von Weizsacker): the cross-cell tau sum can dip below
        // it from roundoff, making unregularised meta-GGAs singular (see the
        // helper). No-op where the physical bound already holds.
        if (is_mgga) enforce_von_weizsaecker_floor(tau, rho, sigma);

        Eigen::VectorXd exc, v_rho, v_sigma, v_tau;
        if (is_mgga) {
            func.eval_unpolarised_mgga(rho, sigma, tau,
                                       exc, v_rho, v_sigma, v_tau);
        } else {
            func.eval_unpolarised(rho, sigma, exc, v_rho, v_sigma);
        }

        // E_xc per unit cell (accumulated across batches).
        e_xc += w_b.dot(exc);

        // --- V_xc(g) += ∂E_xc/∂P(g) for this batch ------------------------
        //   V_xc(g)_{μν} = Σ_{(a,s): s−a = g} Σ_r w(r) v(r) χ_a(r)_μ χ_s(r)_ν
        //                  (+ GGA / mGGA gradient terms with bra=a, ket=s)
        // Each output cell gp sums the pairs that map to it (grouped so the
        // per-gp blocks are written race-free in parallel within the batch).
        const Eigen::VectorXd w_vrho = w_b.array() * v_rho.array();
        Eigen::VectorXd w2vs_gx, w2vs_gy, w2vs_gz;
        if (need_grad) {
            const Eigen::VectorXd w2vs = 2.0 * w_b.array() * v_sigma.array();
            w2vs_gx = w2vs.array() * gx.array();
            w2vs_gy = w2vs.array() * gy.array();
            w2vs_gz = w2vs.array() * gz.array();
        }
        Eigen::VectorXd w_vtau;
        if (is_mgga) {
            w_vtau = 0.5 * w_b.array() * v_tau.array();
        }

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t gp = 0; gp < n_cells; ++gp) {
            if (pairs_by_gp[gp].empty()) continue;
            Eigen::MatrixXd Vg = Eigen::MatrixXd::Zero(nbf, nbf);
            for (int k : pairs_by_gp[gp]) {
                const BraKetPair& pr = pairs[k];
                const Eigen::MatrixXd& chi_a = chi_h[pr.a];
                const Eigen::MatrixXd& chi_s = chi_h[pr.s];
                // LDA piece: χ_aᵀ · diag(w·v_ρ) · χ_s
                Vg.noalias() += chi_a.transpose() * w_vrho.asDiagonal() * chi_s;

                if (need_grad) {
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    // Term 1: (∇ρ · ∇χ_a) χ_s
                    Eigen::MatrixXd A =
                        da[0].array().colwise() * w2vs_gx.array();
                    A.array() += da[1].array().colwise() * w2vs_gy.array();
                    A.array() += da[2].array().colwise() * w2vs_gz.array();
                    Vg.noalias() += A.transpose() * chi_s;
                    // Term 2: χ_a (∇ρ · ∇χ_s)
                    Eigen::MatrixXd B =
                        ds[0].array().colwise() * w2vs_gx.array();
                    B.array() += ds[1].array().colwise() * w2vs_gy.array();
                    B.array() += ds[2].array().colwise() * w2vs_gz.array();
                    Vg.noalias() += chi_a.transpose() * B;
                }

                if (is_mgga) {
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    for (int d = 0; d < 3; ++d) {
                        Vg.noalias() +=
                            da[d].transpose() * w_vtau.asDiagonal() * ds[d];
                    }
                }
            }
            out.V_xc.blocks[gp].noalias() += Vg;   // accumulate across batches
        }
    }

    out.e_xc = e_xc;
    return out;
}


static Eigen::MatrixXd xc_lattice_gradient_batch(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_real_space,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain) {
    if (func.is_external()) {
        throw std::runtime_error(
            "xc_lattice_gradient_contribution: analytic gradients for "
            "full-grid external XC require grid/image/partition response "
            "terms that are not implemented");
    }
    const int nbf = static_cast<int>(basis.nbasis());
    if (P_real_space.nbf != nbf) {
        throw std::runtime_error("xc_lattice_gradient_contribution: nbf mismatch");
    }
    const bool is_gga = (func.kind() == XCKind::GGA);
    const bool is_mgga = (func.kind() == XCKind::MGGA);
    // meta-GGA reuses the GGA σ-Pulay machinery (AO Hessians + shifted-AO
    // gradients) — it depends on ∇ρ too — and additionally carries the
    // τ-Pulay term (added below). Drive both branches with need_hess.
    const bool need_hess = is_gga || is_mgga;

    // Home-cell AO values + gradients on the grid. GGA/meta-GGA additionally
    // need AO Hessians for the sigma-Pulay (and meta-GGA tau-Pulay) terms.
    Eigen::MatrixXd chi_ref;
    std::array<Eigen::MatrixXd, 3> dchi_ref;
    std::array<Eigen::MatrixXd, 6> hess_ref;
    if (need_hess) {
        AOValuesWithHessian ao_ref = evaluate_ao_with_hessian(basis, grid.points);
        chi_ref = std::move(ao_ref.values);
        dchi_ref = std::move(ao_ref.gradients);
        hess_ref = std::move(ao_ref.hessians);
    } else {
        AOValues ao_ref = evaluate_ao_with_gradient(basis, grid.points);
        chi_ref = std::move(ao_ref.values);
        dchi_ref = std::move(ao_ref.gradients);
    }

    // Per-cell shifted AO values + gradients (the gradient needs the shifted
    // AO gradients for the ν-side term even for LDA).
    const std::vector<int> active =
        active_density_cells(P_real_space.cells, system, opts.cutoff_bohr);
    const std::size_t n_cells = P_real_space.cells.size();
    std::vector<Eigen::MatrixXd> chi_h(n_cells);
    std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
    std::vector<std::array<Eigen::MatrixXd, 6>> hess_h;
    if (need_hess) hess_h.resize(n_cells);
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t ai = 0; ai < active.size(); ++ai) {
        const int c = active[ai];
        if (P_real_space.cells[c].index.isZero()) {
            chi_h[c] = chi_ref;
            dchi_h[c] = dchi_ref;
            if (need_hess) hess_h[c] = hess_ref;
            continue;
        }
        // chi(r; R + h) = chi(r - h; R): preserve the exact shells,
        // contractions and normalization supplied by the caller.
        const Eigen::MatrixX3d shifted_points =
            grid.points.rowwise() - P_real_space.cells[c].r_cart.transpose();
        if (need_hess) {
            AOValuesWithHessian aov =
                evaluate_ao_with_hessian(basis, shifted_points);
            chi_h[c] = std::move(aov.values);
            dchi_h[c] = std::move(aov.gradients);
            hess_h[c] = std::move(aov.hessians);
        } else {
            AOValues aov = evaluate_ao_with_gradient(basis, shifted_points);
            chi_h[c] = std::move(aov.values);
            dchi_h[c] = std::move(aov.gradients);
        }
    }

    const auto n_pts = static_cast<Eigen::Index>(grid.points.rows());

    // CROSS-CELL bra/ket pairs — IDENTICAL screening to build_xc_periodic, so
    // the analytic XC Pulay force is the derivative of exactly the E_xc this
    // density produces. The shipped kernel summed only the home-bra slice
    // ρ = Σ_s χ_0 P(s) χ_s while build_xc_periodic uses the full cross-cell
    // ρ = Σ_a Σ_s χ_a P(s−a) χ_s (the dense-XC P0 fix); on a periodic density
    // (P(g≠0) populated) the two disagree, so the kernel missed the bra-image
    // motion (H₂/STO-3G 6-bohr: 9.25e-3 Ha/bohr vs fixed-grid FD; home-only
    // density was already exact). Now the bra ranges over image cells too.
    const auto cell_pos = build_cell_index_map(P_real_space.cells);
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr)
        : opts.cutoff_bohr;
    const std::vector<int> bra_active =
        active_density_cells(P_real_space.cells, system, bra_radius);
    const std::vector<int> bra_cells =
        use_periodic_density_bras(density_domain,
            density_has_cross_cell_blocks(P_real_space, 1e-12))
            ? bra_active
            : home_only_bra(P_real_space.cells, active);
    const std::vector<BraKetPair> pairs =
        build_braket_pairs(P_real_space.cells, bra_cells, active, cell_pos,
            density_domain == PeriodicXCDensityDomain::PERIODIC_LATTICE);

    // ρ(r), (GGA/MGGA) ∇ρ(r) and (MGGA) τ(r) — bra χ_a, ket χ_s, P(s−a).
    Eigen::VectorXd rho = Eigen::VectorXd::Zero(n_pts);
    Eigen::VectorXd gx, gy, gz, tau;
    if (need_hess) {
        gx = Eigen::VectorXd::Zero(n_pts);
        gy = Eigen::VectorXd::Zero(n_pts);
        gz = Eigen::VectorXd::Zero(n_pts);
    }
    if (is_mgga) tau = Eigen::VectorXd::Zero(n_pts);
    for (const BraKetPair& pr : pairs) {
        const Eigen::MatrixXd& chi_a = chi_h[pr.a];   // bra χ(r − R_a)
        const Eigen::MatrixXd& chi_s = chi_h[pr.s];   // ket χ(r − R_s)
        const Eigen::MatrixXd& Pg = P_real_space.blocks[pr.gp];
        const Eigen::MatrixXd chi_P = chi_a * Pg;     // ν→(r,nbf)
        rho.array() += (chi_P.array() * chi_s.array()).rowwise().sum();
        if (need_hess) {
            const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
            const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
            gx.array() += (chi_P.cwiseProduct(ds[0])).rowwise().sum().array();
            gy.array() += (chi_P.cwiseProduct(ds[1])).rowwise().sum().array();
            gz.array() += (chi_P.cwiseProduct(ds[2])).rowwise().sum().array();
            gx.array() += ((da[0] * Pg).cwiseProduct(chi_s)).rowwise().sum().array();
            gy.array() += ((da[1] * Pg).cwiseProduct(chi_s)).rowwise().sum().array();
            gz.array() += ((da[2] * Pg).cwiseProduct(chi_s)).rowwise().sum().array();
        }
        if (is_mgga) {
            const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
            const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
            // τ(r) = ½ Σ_d (∂_d χ_a · P(s−a) · ∂_d χ_s)(r); ×½ after.
            for (int d = 0; d < 3; ++d) {
                tau.array() +=
                    ((da[d] * Pg).cwiseProduct(ds[d])).rowwise().sum().array();
            }
        }
    }
    for (Eigen::Index i = 0; i < n_pts; ++i)
        if (rho[i] < 0.0) rho[i] = 0.0;
    if (is_mgga) tau *= 0.5;  // match build_xc_periodic / rks.cpp:124

    Eigen::VectorXd sigma;
    if (need_hess)
        sigma = gx.array().square() + gy.array().square() + gz.array().square();
    else
        sigma.resize(0);
    // tau >= tau_W (von Weizsacker) — identical to build_xc_periodic so the
    // analytic gradient differentiates exactly the SCF energy surface.
    if (is_mgga) enforce_von_weizsaecker_floor(tau, rho, sigma);
    Eigen::VectorXd exc, v_rho, v_sigma, v_tau;
    if (is_mgga)
        func.eval_unpolarised_mgga(rho, sigma, tau, exc, v_rho, v_sigma, v_tau);
    else
        func.eval_unpolarised(rho, sigma, exc, v_rho, v_sigma);

    const Eigen::VectorXd w_vrho = grid.weights.array() * v_rho.array();
    // GGA/MGGA pre-weighted ∇ρ (2·w·v_σ·∇ρ), reused per cell.
    std::array<Eigen::VectorXd, 3> w2vs_grad;
    if (need_hess) {
        // σ-Pulay pre-weight 2·w·v_σ·∇ρ.  Use the *raw* v_σ — do NOT
        // magnitude-clip it.  B88/LYP/SCAN carry a legitimately large (but
        // finite) v_σ in the low-density tail, and the SCF energy / FD
        // reference are built from exactly that unclipped value, so clamping
        // it here desynchronises the analytic gradient from its own energy
        // (B3LYP/BLYP H₂ drifted ~1e-4 Ha/bohr, SCAN ~2e-3 vs FD; PBE-family
        // was unaffected only because its v_σ never reached the clamp).  The
        // bare 2·w·v_σ form was exact at the G1b gauge fix; only guard a
        // non-finite v_σ (libxc can return 0/0 → NaN at ρ→0 points).
        Eigen::VectorXd vs = v_sigma;
        for (Eigen::Index i = 0; i < n_pts; ++i)
            if (!std::isfinite(vs[i])) vs[i] = 0.0;
        const Eigen::VectorXd w2vs = 2.0 * grid.weights.array() * vs.array();
        w2vs_grad[0] = w2vs.array() * gx.array();
        w2vs_grad[1] = w2vs.array() * gy.array();
        w2vs_grad[2] = w2vs.array() * gz.array();
    }
    // MGGA τ-Pulay pre-weight: ½·w·v_τ. The energy carries τ = ½Σ_d P·∂χ·∂χ
    // and E_xc^τ = Σ_r w v_τ τ, so ∂E/∂R picks up ½·w·v_τ per τ derivative.
    Eigen::VectorXd w_vtau_half;
    if (is_mgga) {
        w_vtau_half.resize(n_pts);
        // ½·w·v_τ. With the tau >= tau_W floor above, v_τ is bounded and
        // physical (no more 1e13 outliers), so the previous ±10 magnitude clip
        // + ρ-screen band-aids are removed — they desynchronised the analytic
        // gradient from its own (unclipped) SCF energy. Keep only a NaN guard.
        for (Eigen::Index i = 0; i < n_pts; ++i) {
            const double vt = std::isfinite(v_tau[i]) ? v_tau[i] : 0.0;
            w_vtau_half[i] = 0.5 * grid.weights[i] * vt;
        }
    }

    std::vector<Atom> cell_atoms(system.unit_cell.begin(), system.unit_cell.end());
    const Molecule mol(std::move(cell_atoms), system.charge, system.multiplicity);
    const std::vector<std::vector<int>> atom_bf = atom_bf_indices(basis, mol);
    const std::size_t N = mol.atoms().size();

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_t(
        n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));
    // Scatter helper: accumulate −Σ_r wv·(dchi⊙coeff)[r,bf] onto atom(bf).
    std::vector<int> bf2atom(nbf, 0);
    for (std::size_t A = 0; A < N; ++A)
        for (int bf : atom_bf[A]) bf2atom[bf] = static_cast<int>(A);

    // Cross-cell pair loop: bra μ in cell a (∂ → atom μ), ket ν in cell s
    // (∂ → atom ν), density block P(s−a). A lattice shift R_a is constant under
    // the atom displacement, so ∂χ_a/∂R_atom(μ) = −∇χ_a = −dchi_h[a]; the same
    // bf2atom scatter is correct for any cell.
    // schedule(static), not dynamic: each thread sums its pairs into a
    // private accumulator and the partials are added in thread order
    // below. A dynamic partition changes which pairs land in which
    // partial from run to run, and with it the rounding of the total, so
    // the same density gave XC energies scattered at 1e-9 (#81). Every
    // pair costs one chi_a * P product, so a static partition loses no
    // balance and the result is reproducible for a fixed thread count.
    #pragma omp parallel for schedule(static)
    for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
        const int tid = omp_thread_index();
        const BraKetPair& pr = pairs[pi];
        const Eigen::MatrixXd& Pg = P_real_space.blocks[pr.gp];
        const Eigen::MatrixXd A1 = chi_h[pr.s] * Pg.transpose();  // (n_pts,nbf), μ
        const Eigen::MatrixXd A2 = chi_h[pr.a] * Pg;              // (n_pts,nbf), ν
        std::array<Eigen::MatrixXd, 3> B1;
        std::array<Eigen::MatrixXd, 3> B2;
        if (need_hess) {
            for (int d = 0; d < 3; ++d) {
                B1[d] = dchi_h[pr.s][d] * Pg.transpose();
                B2[d] = dchi_h[pr.a][d] * Pg;
            }
        }
        for (int co = 0; co < 3; ++co) {
            // Term1 (bra AO μ in cell a): t1[μ] = Σ_r wv · ∂_co χ_a[μ] · A1[μ]
            Eigen::VectorXd t1 =
                (dchi_h[pr.a][co].cwiseProduct(A1)).transpose() * w_vrho;
            // Term2 (ket AO ν in cell s): t2[ν] = Σ_r wv · ∂_co χ_s[ν] · A2[ν]
            Eigen::VectorXd t2 =
                (dchi_h[pr.s][co].cwiseProduct(A2)).transpose() * w_vrho;
            if (need_hess) {
                for (int d = 0; d < 3; ++d) {
                    const int hd = hessian_component(co, d);
                    // Bra AO μ (cell a):
                    //   f_d · [∂_co∂_d χ_a(μ) · (χ_s P^T)(μ)
                    //          + ∂_co χ_a(μ) · (∂_dχ_s P^T)(μ)]
                    t1.noalias() +=
                        (hess_h[pr.a][hd].cwiseProduct(A1)).transpose()
                        * w2vs_grad[d];
                    t1.noalias() +=
                        (dchi_h[pr.a][co].cwiseProduct(B1[d])).transpose()
                        * w2vs_grad[d];
                    // Ket AO ν (cell s):
                    //   f_d · [∂_coχ_s(ν) · (∂_dχ_a P)(ν)
                    //          + ∂_co∂_dχ_s(ν) · (χ_a P)(ν)]
                    t2.noalias() +=
                        (dchi_h[pr.s][co].cwiseProduct(B2[d])).transpose()
                        * w2vs_grad[d];
                    t2.noalias() +=
                        (hess_h[pr.s][hd].cwiseProduct(A2)).transpose()
                        * w2vs_grad[d];
                }
            }
            if (is_mgga) {
                // τ-Pulay. τ = ½ Σ_d Σ_μν P_μν ∂_dχ_a,μ(r) ∂_dχ_s,ν(r); moving
                // the bra AO μ gives −∂_co∂_dχ_a, the ket AO ν −∂_co∂_dχ_s.
                for (int d = 0; d < 3; ++d) {
                    const int hd = hessian_component(co, d);
                    t1.noalias() +=
                        (hess_h[pr.a][hd].cwiseProduct(B1[d])).transpose()
                        * w_vtau_half;
                    t2.noalias() +=
                        (hess_h[pr.s][hd].cwiseProduct(B2[d])).transpose()
                        * w_vtau_half;
                }
            }
            // scatter (negated): grad(atom,co) -= t1[μ] + t2[ν]
            Eigen::MatrixXd& g = grad_t[tid];
            for (int bf = 0; bf < nbf; ++bf) {
                g(bf2atom[bf], co) -= t1[bf] + t2[bf];
            }
        }
    }
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (int t = 0; t < n_threads; ++t) grad += grad_t[t];
    return grad;
}


static Eigen::MatrixXd xc_lattice_gradient_batch_uks(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_alpha,
    const LatticeMatrixSet& P_beta,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain) {
    if (func.is_external()) {
        throw std::runtime_error(
            "xc_lattice_gradient_contribution_uks: analytic gradients for "
            "full-grid external XC require grid/image/partition response "
            "terms that are not implemented");
    }
    const int nbf = static_cast<int>(basis.nbasis());
    if (P_alpha.nbf != nbf || P_beta.nbf != nbf) {
        throw std::runtime_error(
            "xc_lattice_gradient_contribution_uks: density nbf mismatch");
    }
    if (P_alpha.cells.size() != P_beta.cells.size()) {
        throw std::runtime_error(
            "xc_lattice_gradient_contribution_uks: P_alpha and P_beta "
            "must use the same cell list");
    }
    const bool is_gga = (func.kind() == XCKind::GGA);
    const bool is_mgga = (func.kind() == XCKind::MGGA);
    // meta-GGA reuses the σ-Pulay machinery and adds the per-spin τ-Pulay.
    const bool need_hess = is_gga || is_mgga;

    Eigen::MatrixXd chi_ref;
    std::array<Eigen::MatrixXd, 3> dchi_ref;
    std::array<Eigen::MatrixXd, 6> hess_ref;
    if (need_hess) {
        AOValuesWithHessian ao_ref = evaluate_ao_with_hessian(basis, grid.points);
        chi_ref = std::move(ao_ref.values);
        dchi_ref = std::move(ao_ref.gradients);
        hess_ref = std::move(ao_ref.hessians);
    } else {
        AOValues ao_ref = evaluate_ao_with_gradient(basis, grid.points);
        chi_ref = std::move(ao_ref.values);
        dchi_ref = std::move(ao_ref.gradients);
    }

    const std::vector<int> active =
        active_density_cells(P_alpha.cells, system, opts.cutoff_bohr);
    const std::size_t n_cells = P_alpha.cells.size();
    std::vector<Eigen::MatrixXd> chi_h(n_cells);
    std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
    std::vector<std::array<Eigen::MatrixXd, 6>> hess_h;
    if (need_hess) hess_h.resize(n_cells);
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t ai = 0; ai < active.size(); ++ai) {
        const int c = active[ai];
        if (P_alpha.cells[c].index.isZero()) {
            chi_h[c] = chi_ref;
            dchi_h[c] = dchi_ref;
            if (need_hess) hess_h[c] = hess_ref;
            continue;
        }
        // chi(r; R + h) = chi(r - h; R): preserve the exact shells,
        // contractions and normalization supplied by the caller.
        const Eigen::MatrixX3d shifted_points =
            grid.points.rowwise() - P_alpha.cells[c].r_cart.transpose();
        if (need_hess) {
            AOValuesWithHessian aov =
                evaluate_ao_with_hessian(basis, shifted_points);
            chi_h[c] = std::move(aov.values);
            dchi_h[c] = std::move(aov.gradients);
            hess_h[c] = std::move(aov.hessians);
        } else {
            AOValues aov = evaluate_ao_with_gradient(basis, shifted_points);
            chi_h[c] = std::move(aov.values);
            dchi_h[c] = std::move(aov.gradients);
        }
    }

    // CROSS-CELL bra/ket pairs — IDENTICAL screening to build_xc_periodic_uks
    // (bra ranges over image cells too), so this per-spin XC Pulay force is the
    // derivative of exactly the E_xc that energy produces. The shipped kernel
    // summed only the home-bra slice ρ_σ = Σ_s χ_0 P_σ(s) χ_s, missing the
    // bra-image motion on a periodic density (the RKS sibling's bug).
    const auto cell_pos = build_cell_index_map(P_alpha.cells);
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr)
        : opts.cutoff_bohr;
    const std::vector<int> bra_active =
        active_density_cells(P_alpha.cells, system, bra_radius);
    const bool cross_cell = density_has_cross_cell_blocks(P_alpha, 1e-12) ||
                            density_has_cross_cell_blocks(P_beta, 1e-12);
    const std::vector<int> bra_cells =
        use_periodic_density_bras(density_domain, cross_cell)
            ? bra_active : home_only_bra(P_alpha.cells, active);
    const std::vector<BraKetPair> pairs =
        build_braket_pairs(P_alpha.cells, bra_cells, active, cell_pos,
            density_domain == PeriodicXCDensityDomain::PERIODIC_LATTICE);

    const auto n_pts = static_cast<Eigen::Index>(grid.points.rows());
    Eigen::VectorXd rho_a = Eigen::VectorXd::Zero(n_pts);
    Eigen::VectorXd rho_b = Eigen::VectorXd::Zero(n_pts);
    Eigen::VectorXd gax, gay, gaz, gbx, gby, gbz, tau_a, tau_b;
    if (need_hess) {
        gax = Eigen::VectorXd::Zero(n_pts);
        gay = Eigen::VectorXd::Zero(n_pts);
        gaz = Eigen::VectorXd::Zero(n_pts);
        gbx = Eigen::VectorXd::Zero(n_pts);
        gby = Eigen::VectorXd::Zero(n_pts);
        gbz = Eigen::VectorXd::Zero(n_pts);
    }
    if (is_mgga) {
        tau_a = Eigen::VectorXd::Zero(n_pts);
        tau_b = Eigen::VectorXd::Zero(n_pts);
    }

    for (const BraKetPair& pr : pairs) {
        const Eigen::MatrixXd& chi_a = chi_h[pr.a];   // bra χ(r − R_a)
        const Eigen::MatrixXd& chi_s = chi_h[pr.s];   // ket χ(r − R_s)
        const Eigen::MatrixXd& Pa = P_alpha.blocks[pr.gp];
        const Eigen::MatrixXd& Pb = P_beta.blocks[pr.gp];
        const Eigen::MatrixXd chi_Pa = chi_a * Pa;
        const Eigen::MatrixXd chi_Pb = chi_a * Pb;
        rho_a.array() += (chi_Pa.array() * chi_s.array()).rowwise().sum();
        rho_b.array() += (chi_Pb.array() * chi_s.array()).rowwise().sum();

        if (is_mgga) {
            const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
            const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
            for (int d = 0; d < 3; ++d) {
                tau_a.array() += ((da[d] * Pa)
                                      .cwiseProduct(ds[d])).rowwise().sum().array();
                tau_b.array() += ((da[d] * Pb)
                                      .cwiseProduct(ds[d])).rowwise().sum().array();
            }
        }
        if (is_gga || is_mgga) {
            const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
            const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
            gax.array() += (chi_Pa.cwiseProduct(ds[0])).rowwise().sum().array();
            gay.array() += (chi_Pa.cwiseProduct(ds[1])).rowwise().sum().array();
            gaz.array() += (chi_Pa.cwiseProduct(ds[2])).rowwise().sum().array();
            gax.array() += ((da[0] * Pa).cwiseProduct(chi_s)).rowwise().sum().array();
            gay.array() += ((da[1] * Pa).cwiseProduct(chi_s)).rowwise().sum().array();
            gaz.array() += ((da[2] * Pa).cwiseProduct(chi_s)).rowwise().sum().array();

            gbx.array() += (chi_Pb.cwiseProduct(ds[0])).rowwise().sum().array();
            gby.array() += (chi_Pb.cwiseProduct(ds[1])).rowwise().sum().array();
            gbz.array() += (chi_Pb.cwiseProduct(ds[2])).rowwise().sum().array();
            gbx.array() += ((da[0] * Pb).cwiseProduct(chi_s)).rowwise().sum().array();
            gby.array() += ((da[1] * Pb).cwiseProduct(chi_s)).rowwise().sum().array();
            gbz.array() += ((da[2] * Pb).cwiseProduct(chi_s)).rowwise().sum().array();
        }
    }

    for (Eigen::Index i = 0; i < n_pts; ++i) {
        if (rho_a[i] < 0.0) rho_a[i] = 0.0;
        if (rho_b[i] < 0.0) rho_b[i] = 0.0;
    }
    if (is_mgga) {
        tau_a *= 0.5;
        tau_b *= 0.5;
    }

    Eigen::VectorXd sigma_aa, sigma_ab, sigma_bb;
    if (need_hess) {
        sigma_aa = gax.array().square()
                 + gay.array().square()
                 + gaz.array().square();
        sigma_bb = gbx.array().square()
                 + gby.array().square()
                 + gbz.array().square();
        sigma_ab = gax.array() * gbx.array()
                 + gay.array() * gby.array()
                 + gaz.array() * gbz.array();
    } else {
        sigma_aa.resize(0);
        sigma_ab.resize(0);
        sigma_bb.resize(0);
    }

    // tau_σ >= tau_W,σ = σ_σσ / (8 ρ_σ) per spin (von Weizsacker) — identical
    // to build_xc_periodic_uks so the analytic gradient matches its energy.
    if (is_mgga) {
        enforce_von_weizsaecker_floor(tau_a, rho_a, sigma_aa);
        enforce_von_weizsaecker_floor(tau_b, rho_b, sigma_bb);
    }

    Eigen::VectorXd exc;
    Eigen::VectorXd v_rho_a, v_rho_b;
    Eigen::VectorXd v_sig_aa, v_sig_ab, v_sig_bb;
    Eigen::VectorXd v_tau_a, v_tau_b;
    if (is_mgga) {
        func.eval_polarised_mgga(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                                 tau_a, tau_b,
                                 exc, v_rho_a, v_rho_b,
                                 v_sig_aa, v_sig_ab, v_sig_bb,
                                 v_tau_a, v_tau_b);
    } else {
        func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                            exc, v_rho_a, v_rho_b,
                            v_sig_aa, v_sig_ab, v_sig_bb);
    }

    const Eigen::VectorXd w_vra = grid.weights.array() * v_rho_a.array();
    const Eigen::VectorXd w_vrb = grid.weights.array() * v_rho_b.array();
    // Per-spin τ-Pulay pre-weights ½·w·v_τσ. With the τ_σ >= τ_W floor above,
    // v_τσ is bounded; the previous ±10 clip + ρ-screen band-aids are removed
    // (they desynced the analytic gradient from its energy). NaN guard only.
    Eigen::VectorXd w_vtau_half_a, w_vtau_half_b;
    if (is_mgga) {
        w_vtau_half_a.resize(n_pts);
        w_vtau_half_b.resize(n_pts);
        for (Eigen::Index i = 0; i < n_pts; ++i) {
            const double vta = std::isfinite(v_tau_a[i]) ? v_tau_a[i] : 0.0;
            const double vtb = std::isfinite(v_tau_b[i]) ? v_tau_b[i] : 0.0;
            w_vtau_half_a[i] = 0.5 * grid.weights[i] * vta;
            w_vtau_half_b[i] = 0.5 * grid.weights[i] * vtb;
        }
    }
    std::array<Eigen::VectorXd, 3> w_fa;
    std::array<Eigen::VectorXd, 3> w_fb;
    if (need_hess) {
        // σ-Pulay pre-weights.  Use the *raw* v_σ (αα/αβ/ββ) — do NOT
        // magnitude-clip: B88/LYP/SCAN carry a legitimately large (but
        // finite) v_σ in the low-density tail and the energy / FD reference
        // use that unclipped value (see the RKS path above for the ~1e-4
        // Ha/bohr desync a clamp causes).  Only guard a non-finite v_σ.
        Eigen::VectorXd vs_aa = v_sig_aa;
        Eigen::VectorXd vs_ab = v_sig_ab;
        Eigen::VectorXd vs_bb = v_sig_bb;
        for (Eigen::Index i = 0; i < n_pts; ++i) {
            if (!std::isfinite(vs_aa[i])) vs_aa[i] = 0.0;
            if (!std::isfinite(vs_ab[i])) vs_ab[i] = 0.0;
            if (!std::isfinite(vs_bb[i])) vs_bb[i] = 0.0;
        }
        w_fa[0] = grid.weights.array()
                * (2.0 * vs_aa.array() * gax.array()
                   + vs_ab.array() * gbx.array());
        w_fa[1] = grid.weights.array()
                * (2.0 * vs_aa.array() * gay.array()
                   + vs_ab.array() * gby.array());
        w_fa[2] = grid.weights.array()
                * (2.0 * vs_aa.array() * gaz.array()
                   + vs_ab.array() * gbz.array());
        w_fb[0] = grid.weights.array()
                * (2.0 * vs_bb.array() * gbx.array()
                   + vs_ab.array() * gax.array());
        w_fb[1] = grid.weights.array()
                * (2.0 * vs_bb.array() * gby.array()
                   + vs_ab.array() * gay.array());
        w_fb[2] = grid.weights.array()
                * (2.0 * vs_bb.array() * gbz.array()
                   + vs_ab.array() * gaz.array());
    }

    std::vector<Atom> cell_atoms(system.unit_cell.begin(), system.unit_cell.end());
    const Molecule mol(std::move(cell_atoms), system.charge, system.multiplicity);
    const std::vector<std::vector<int>> atom_bf = atom_bf_indices(basis, mol);
    const std::size_t N = mol.atoms().size();

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_t(
        n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));
    std::vector<int> bf2atom(nbf, 0);
    for (std::size_t A = 0; A < N; ++A)
        for (int bf : atom_bf[A]) bf2atom[bf] = static_cast<int>(A);

    // schedule(static), not dynamic: each thread sums its pairs into a
    // private accumulator and the partials are added in thread order
    // below. A dynamic partition changes which pairs land in which
    // partial from run to run, and with it the rounding of the total, so
    // the same density gave XC energies scattered at 1e-9 (#81). Every
    // pair costs one chi_a * P product, so a static partition loses no
    // balance and the result is reproducible for a fixed thread count.
    #pragma omp parallel for schedule(static)
    for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
        const int tid = omp_thread_index();
        const BraKetPair& pr = pairs[pi];
        const Eigen::MatrixXd& Pa = P_alpha.blocks[pr.gp];
        const Eigen::MatrixXd& Pb = P_beta.blocks[pr.gp];
        const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];  // bra grads
        const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];  // ket grads
        const Eigen::MatrixXd A1a = chi_h[pr.s] * Pa.transpose();  // μ (bra)
        const Eigen::MatrixXd A2a = chi_h[pr.a] * Pa;              // ν (ket)
        const Eigen::MatrixXd A1b = chi_h[pr.s] * Pb.transpose();
        const Eigen::MatrixXd A2b = chi_h[pr.a] * Pb;
        std::array<Eigen::MatrixXd, 3> B1a;
        std::array<Eigen::MatrixXd, 3> B2a;
        std::array<Eigen::MatrixXd, 3> B1b;
        std::array<Eigen::MatrixXd, 3> B2b;
        if (need_hess) {
            for (int d = 0; d < 3; ++d) {
                B1a[d] = ds[d] * Pa.transpose();
                B2a[d] = da[d] * Pa;
                B1b[d] = ds[d] * Pb.transpose();
                B2b[d] = da[d] * Pb;
            }
        }
        for (int co = 0; co < 3; ++co) {
            // t1σ: bra AO μ in cell a (∂ → atom μ); t2σ: ket AO ν in cell s.
            Eigen::VectorXd t1a =
                (da[co].cwiseProduct(A1a)).transpose() * w_vra;
            Eigen::VectorXd t2a =
                (ds[co].cwiseProduct(A2a)).transpose() * w_vra;
            Eigen::VectorXd t1b =
                (da[co].cwiseProduct(A1b)).transpose() * w_vrb;
            Eigen::VectorXd t2b =
                (ds[co].cwiseProduct(A2b)).transpose() * w_vrb;
            if (need_hess) {
                for (int d = 0; d < 3; ++d) {
                    const int hd = hessian_component(co, d);
                    t1a.noalias() +=
                        (hess_h[pr.a][hd].cwiseProduct(A1a)).transpose()
                        * w_fa[d];
                    t1a.noalias() +=
                        (da[co].cwiseProduct(B1a[d])).transpose()
                        * w_fa[d];
                    t2a.noalias() +=
                        (ds[co].cwiseProduct(B2a[d])).transpose()
                        * w_fa[d];
                    t2a.noalias() +=
                        (hess_h[pr.s][hd].cwiseProduct(A2a)).transpose()
                        * w_fa[d];

                    t1b.noalias() +=
                        (hess_h[pr.a][hd].cwiseProduct(A1b)).transpose()
                        * w_fb[d];
                    t1b.noalias() +=
                        (da[co].cwiseProduct(B1b[d])).transpose()
                        * w_fb[d];
                    t2b.noalias() +=
                        (ds[co].cwiseProduct(B2b[d])).transpose()
                        * w_fb[d];
                    t2b.noalias() +=
                        (hess_h[pr.s][hd].cwiseProduct(A2b)).transpose()
                        * w_fb[d];
                }
            }
            if (is_mgga) {
                // Per-spin τ-Pulay: t1σ += ½w·v_τσ·∂_co∂_dχ_a·(∂_dχ_s P_σ^T),
                //                   t2σ += ½w·v_τσ·∂_co∂_dχ_s·(∂_dχ_a P_σ).
                for (int d = 0; d < 3; ++d) {
                    const int hd = hessian_component(co, d);
                    t1a.noalias() +=
                        (hess_h[pr.a][hd].cwiseProduct(B1a[d])).transpose()
                        * w_vtau_half_a;
                    t2a.noalias() +=
                        (hess_h[pr.s][hd].cwiseProduct(B2a[d])).transpose()
                        * w_vtau_half_a;
                    t1b.noalias() +=
                        (hess_h[pr.a][hd].cwiseProduct(B1b[d])).transpose()
                        * w_vtau_half_b;
                    t2b.noalias() +=
                        (hess_h[pr.s][hd].cwiseProduct(B2b[d])).transpose()
                        * w_vtau_half_b;
                }
            }
            Eigen::MatrixXd& g = grad_t[tid];
            for (int bf = 0; bf < nbf; ++bf) {
                g(bf2atom[bf], co) -= t1a[bf] + t2a[bf] + t1b[bf] + t2b[bf];
            }
        }
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (int t = 0; t < n_threads; ++t) grad += grad_t[t];
    return grad;
}


// ---------------------------------------------------------------------------
// Open-shell (UKS) periodic XC. Mirrors build_xc_periodic but with two
// densities (P_α, P_β) and produces two V_xc matrices (V_α, V_β) plus a
// total E_xc. The kernel logic follows the molecular build_uks_xc in
// uks.cpp; the only difference is the per-cell AO contraction loop, which
// matches build_xc_periodic exactly.
// ---------------------------------------------------------------------------
PeriodicUKSXCContribution build_xc_periodic_uks(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_alpha,
    const LatticeMatrixSet& P_beta,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain) {
    const int nbf = static_cast<int>(basis.nbasis());
    if (P_alpha.nbf != nbf || P_beta.nbf != nbf) {
        throw std::runtime_error("build_xc_periodic_uks: density nbf mismatch");
    }
    if (P_alpha.cells.size() != P_beta.cells.size()) {
        throw std::runtime_error(
            "build_xc_periodic_uks: P_alpha and P_beta must use the "
            "same cell list");
    }
    const bool is_gga  = (func.kind() == XCKind::GGA);
    const bool is_mgga = (func.kind() == XCKind::MGGA);
    const bool need_grad = is_gga || is_mgga;

    const std::size_t n_cells = P_alpha.cells.size();
    const auto n_pts = static_cast<Eigen::Index>(grid.points.rows());

    // Cross-cell density (both AOs over lattice cells), per spin σ:
    //   ρ_σ(r) = Σ_a Σ_s χ_a(r) · P_σ(s−a) · χ_s(r)
    // identical structure to the closed-shell build_xc_periodic fix — the
    // shipped home-bra-only sum dropped the absolute lattice translation and
    // gave the dense-crystal periodic-XC error pinned by
    // tests/test_periodic_xc_cross_cell.py.
    const auto cell_pos = build_cell_index_map(P_alpha.cells);
    const std::vector<int> active =
        active_density_cells(P_alpha.cells, system, opts.cutoff_bohr);
    // Bra (image) cells screened by the periodic Becke partition reach (see
    // LatticeSumOptions::becke_image_radius_bohr and the closed-shell path).
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr)
        : opts.cutoff_bohr;
    const std::vector<int> bra_active =
        active_density_cells(P_alpha.cells, system, bra_radius);
    // Cross-cell bra only for a genuine periodic density (either spin carries
    // P(g≠0)); a molecular-limit density keeps the home-cell bra.
    const bool has_xc = use_periodic_density_bras(
        density_domain,
        density_has_cross_cell_blocks(P_alpha, 1e-12) ||
            density_has_cross_cell_blocks(P_beta, 1e-12));
    const std::vector<int> bra_cells =
        has_xc ? bra_active : home_only_bra(P_alpha.cells, active);
    const std::vector<BraKetPair> pairs =
        build_braket_pairs(
            P_alpha.cells, bra_cells, active, cell_pos,
            density_domain == PeriodicXCDensityDomain::PERIODIC_LATTICE);

    // ---- Output accumulators (filled incrementally over grid batches) -----
    PeriodicUKSXCContribution out;
    out.V_alpha.nbf = nbf;
    out.V_alpha.cells = P_alpha.cells;
    out.V_alpha.blocks.assign(n_cells, Eigen::MatrixXd::Zero(nbf, nbf));
    out.V_beta.nbf = nbf;
    out.V_beta.cells = P_alpha.cells;
    out.V_beta.blocks.assign(n_cells, Eigen::MatrixXd::Zero(nbf, nbf));
    double e_xc = 0.0;

    // Output cell gp -> the bra/ket pairs that map to it (batch-independent).
    std::vector<std::vector<int>> pairs_by_gp(n_cells);
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        pairs_by_gp[pairs[k].gp].push_back(static_cast<int>(k));
    }

    if (func.is_external()) {
        if (grid.atomic_weights.size() != n_pts
            || static_cast<Eigen::Index>(grid.atom_of_point.size()) != n_pts
            || grid.atom_coords.cols() != 3) {
            throw std::invalid_argument(
                "external periodic UKS XC requires atom-major grid "
                "ownership, raw atomic weights, and atom coordinates");
        }
        ExternalXCInput input;
        input.functional = func.name();
        input.grid_profile = atomic_grid_profile_name(grid.atomic_grid_profile);
        input.points = grid.points;
        input.grid_weights = grid.weights;
        input.atomic_grid_weights = grid.atomic_weights;
        input.atom_of_point = grid.atom_of_point;
        input.atom_coords = grid.atom_coords;
        input.atomic_numbers = grid.atomic_numbers;
        input.periodic = true;
        input.periodic_dimension = system.dim;
        input.lattice = system.lattice;
        input.rho_alpha.resize(n_pts);
        input.rho_beta.resize(n_pts);
        input.grad_alpha.resize(n_pts, 3);
        input.grad_beta.resize(n_pts, 3);
        input.tau_alpha.resize(n_pts);
        input.tau_beta.resize(n_pts);

        const Eigen::Index kExternalXcGridBatch = static_cast<Eigen::Index>(
            periodic_xc_value_batch_size(nbf, active.size(), true,
                                         true, omp_max_threads()));
        for (Eigen::Index g0 = 0; g0 < n_pts;
             g0 += kExternalXcGridBatch) {
            const Eigen::Index nb = std::min<Eigen::Index>(
                kExternalXcGridBatch, n_pts - g0);
            const Eigen::MatrixX3d pts_b = grid.points.middleRows(g0, nb);
            std::vector<Eigen::MatrixXd> chi_h(n_cells);
            std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
            evaluate_active_shifted_ao(
                basis, pts_b, P_alpha.cells, active,
                /*need_grad=*/true, chi_h, dchi_h);
            const ExternalPeriodicFields fa =
                build_external_periodic_fields_batch(
                    P_alpha, pairs, chi_h, dchi_h, nb);
            const ExternalPeriodicFields fb =
                build_external_periodic_fields_batch(
                    P_beta, pairs, chi_h, dchi_h, nb);
            input.rho_alpha.segment(g0, nb) = fa.rho;
            input.rho_beta.segment(g0, nb) = fb.rho;
            input.grad_alpha.middleRows(g0, nb) = fa.grad;
            input.grad_beta.middleRows(g0, nb) = fb.grad;
            // Preserve the provider's linear density-matrix feature map.
            // Unlike singular pointwise libxc meta-GGAs, the published SKALA
            // preprocessing accepts raw tau through log(abs(tau) + eps).
            // Applying the tau_W floor here would require its matching
            // piecewise adjoint and would define a different functional.
            input.tau_alpha.segment(g0, nb) = fa.tau;
            input.tau_beta.segment(g0, nb) = fb.tau;
        }

        const ExternalXCEvaluation ev = func.eval_external(input);
        for (Eigen::Index g0 = 0; g0 < n_pts;
             g0 += kExternalXcGridBatch) {
            const Eigen::Index nb = std::min<Eigen::Index>(
                kExternalXcGridBatch, n_pts - g0);
            const Eigen::MatrixX3d pts_b = grid.points.middleRows(g0, nb);
            std::vector<Eigen::MatrixXd> chi_h(n_cells);
            std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
            evaluate_active_shifted_ao(
                basis, pts_b, P_alpha.cells, active,
                /*need_grad=*/true, chi_h, dchi_h);
            project_external_periodic_batch(
                pairs, pairs_by_gp, chi_h, dchi_h,
                ev.v_rho_alpha.segment(g0, nb),
                ev.v_grad_alpha.middleRows(g0, nb),
                ev.v_tau_alpha.segment(g0, nb), out.V_alpha.blocks);
            project_external_periodic_batch(
                pairs, pairs_by_gp, chi_h, dchi_h,
                ev.v_rho_beta.segment(g0, nb),
                ev.v_grad_beta.middleRows(g0, nb),
                ev.v_tau_beta.segment(g0, nb), out.V_beta.blocks);
        }
        out.e_xc = ev.energy;
        return out;
    }

    // ---- Grid-batched XC quadrature (spin-polarised; see build_xc_periodic)
    // chi_h / dchi_h for the active cells dominate memory; libxc is pointwise,
    // so the rho -> functional -> V_σ(g) chain factorises over disjoint
    // grid-point batches, accumulating e_xc and V_alpha/V_beta(g). Peak AO
    // memory becomes O(n_active * batch * nbf), independent of grid size.
    // Per-batch math is identical to the whole-grid path, so energies match
    // to floating-point reassociation.
    const Eigen::Index kXcGridBatch = static_cast<Eigen::Index>(
            periodic_xc_value_batch_size(nbf, active.size(), need_grad,
                                         true, omp_max_threads()));
    for (Eigen::Index g0 = 0; g0 < n_pts; g0 += kXcGridBatch) {
        const Eigen::Index nb = std::min(kXcGridBatch, n_pts - g0);
        const Eigen::MatrixX3d pts_b = grid.points.middleRows(g0, nb);
        const Eigen::VectorXd w_b = grid.weights.segment(g0, nb);

        std::vector<Eigen::MatrixXd> chi_h(n_cells);
        std::vector<std::array<Eigen::MatrixXd, 3>> dchi_h(n_cells);
        evaluate_active_shifted_ao(basis, pts_b, P_alpha.cells, active,
                                   need_grad, chi_h, dchi_h);

        // Per-spin densities and gradients (batch).
        Eigen::VectorXd rho_a = Eigen::VectorXd::Zero(nb);
        Eigen::VectorXd rho_b = Eigen::VectorXd::Zero(nb);
        Eigen::VectorXd gax, gay, gaz, gbx, gby, gbz, tau_a, tau_b;
        if (need_grad) {
            gax = Eigen::VectorXd::Zero(nb);
            gay = Eigen::VectorXd::Zero(nb);
            gaz = Eigen::VectorXd::Zero(nb);
            gbx = Eigen::VectorXd::Zero(nb);
            gby = Eigen::VectorXd::Zero(nb);
            gbz = Eigen::VectorXd::Zero(nb);
        }
        if (is_mgga) {
            tau_a = Eigen::VectorXd::Zero(nb);
            tau_b = Eigen::VectorXd::Zero(nb);
        }

        {
            const int n_threads = omp_max_threads();
            std::vector<Eigen::VectorXd> rho_a_t(n_threads,
                Eigen::VectorXd::Zero(nb));
            std::vector<Eigen::VectorXd> rho_b_t(n_threads,
                Eigen::VectorXd::Zero(nb));
            std::vector<Eigen::VectorXd> gax_t, gay_t, gaz_t,
                                         gbx_t, gby_t, gbz_t, tau_a_t, tau_b_t;
            if (need_grad) {
                gax_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gay_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gaz_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gbx_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gby_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                gbz_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
            }
            if (is_mgga) {
                tau_a_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
                tau_b_t.assign(n_threads, Eigen::VectorXd::Zero(nb));
            }

            // schedule(static), not dynamic: each thread sums its pairs into a
            // private accumulator and the partials are added in thread order
            // below. A dynamic partition changes which pairs land in which
            // partial from run to run, and with it the rounding of the total, so
            // the same density gave XC energies scattered at 1e-9 (#81). Every
            // pair costs one chi_a * P product, so a static partition loses no
            // balance and the result is reproducible for a fixed thread count.
            #pragma omp parallel for schedule(static)
            for (std::size_t k = 0; k < pairs.size(); ++k) {
                const int tid = omp_thread_index();
                const BraKetPair& pr = pairs[k];
                const Eigen::MatrixXd& chi_a = chi_h[pr.a];   // bra χ(r − R_a)
                const Eigen::MatrixXd& chi_s = chi_h[pr.s];   // ket χ(r − R_s)
                const Eigen::MatrixXd chi_Pa = chi_a * P_alpha.blocks[pr.gp];
                const Eigen::MatrixXd chi_Pb = chi_a * P_beta.blocks[pr.gp];
                rho_a_t[tid].array() +=
                    (chi_Pa.array() * chi_s.array()).rowwise().sum();
                rho_b_t[tid].array() +=
                    (chi_Pb.array() * chi_s.array()).rowwise().sum();

                if (need_grad) {
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    // ket-gradient term  χ_a P_σ ⊙ ∂χ_s
                    gax_t[tid].array() +=
                        (chi_Pa.array() * ds[0].array()).rowwise().sum();
                    gay_t[tid].array() +=
                        (chi_Pa.array() * ds[1].array()).rowwise().sum();
                    gaz_t[tid].array() +=
                        (chi_Pa.array() * ds[2].array()).rowwise().sum();
                    gbx_t[tid].array() +=
                        (chi_Pb.array() * ds[0].array()).rowwise().sum();
                    gby_t[tid].array() +=
                        (chi_Pb.array() * ds[1].array()).rowwise().sum();
                    gbz_t[tid].array() +=
                        (chi_Pb.array() * ds[2].array()).rowwise().sum();
                    // bra-gradient term  (∂χ_a P_σ) ⊙ χ_s
                    gax_t[tid].array() +=
                        ((da[0] * P_alpha.blocks[pr.gp]).array()
                         * chi_s.array()).rowwise().sum();
                    gay_t[tid].array() +=
                        ((da[1] * P_alpha.blocks[pr.gp]).array()
                         * chi_s.array()).rowwise().sum();
                    gaz_t[tid].array() +=
                        ((da[2] * P_alpha.blocks[pr.gp]).array()
                         * chi_s.array()).rowwise().sum();
                    gbx_t[tid].array() +=
                        ((da[0] * P_beta.blocks[pr.gp]).array()
                         * chi_s.array()).rowwise().sum();
                    gby_t[tid].array() +=
                        ((da[1] * P_beta.blocks[pr.gp]).array()
                         * chi_s.array()).rowwise().sum();
                    gbz_t[tid].array() +=
                        ((da[2] * P_beta.blocks[pr.gp]).array()
                         * chi_s.array()).rowwise().sum();
                }

                if (is_mgga) {
                    // τ_σ += ½ Σ_d (∂_dχ_a · P_σ(s−a) · ∂_dχ_s); ×½ after.
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    for (int d = 0; d < 3; ++d) {
                        tau_a_t[tid].array() +=
                            ((da[d] * P_alpha.blocks[pr.gp]).array()
                             * ds[d].array()).rowwise().sum();
                        tau_b_t[tid].array() +=
                            ((da[d] * P_beta.blocks[pr.gp]).array()
                             * ds[d].array()).rowwise().sum();
                    }
                }
            }

            for (int t = 0; t < n_threads; ++t) {
                rho_a += rho_a_t[t];
                rho_b += rho_b_t[t];
                if (need_grad) {
                    gax += gax_t[t]; gay += gay_t[t]; gaz += gaz_t[t];
                    gbx += gbx_t[t]; gby += gby_t[t]; gbz += gbz_t[t];
                }
                if (is_mgga) {
                    tau_a += tau_a_t[t];
                    tau_b += tau_b_t[t];
                }
            }
        }
        if (is_mgga) {
            tau_a *= 0.5;   // match uks.cpp:149-150
            tau_b *= 0.5;
        }

        // Roundoff hygiene.
        for (Eigen::Index i = 0; i < nb; ++i) {
            if (rho_a[i] < 0.0) rho_a[i] = 0.0;
            if (rho_b[i] < 0.0) rho_b[i] = 0.0;
        }

        Eigen::VectorXd sigma_aa, sigma_ab, sigma_bb;
        if (need_grad) {
            sigma_aa = gax.array().square()
                     + gay.array().square()
                     + gaz.array().square();
            sigma_bb = gbx.array().square()
                     + gby.array().square()
                     + gbz.array().square();
            sigma_ab = gax.array() * gbx.array()
                     + gay.array() * gby.array()
                     + gaz.array() * gbz.array();
        } else {
            sigma_aa.resize(0);
            sigma_ab.resize(0);
            sigma_bb.resize(0);
        }

        // tau_σ >= tau_W,σ = σ_σσ / (8 ρ_σ) per spin (von Weizsacker): the
        // cross-cell per-spin τ sum can dip below the bound from roundoff,
        // making unregularised meta-GGAs singular (see the helper). No-op where
        // the physical bound already holds.
        if (is_mgga) {
            enforce_von_weizsaecker_floor(tau_a, rho_a, sigma_aa);
            enforce_von_weizsaecker_floor(tau_b, rho_b, sigma_bb);
        }

        Eigen::VectorXd exc;
        Eigen::VectorXd v_rho_a, v_rho_b;
        Eigen::VectorXd v_sig_aa, v_sig_ab, v_sig_bb;
        Eigen::VectorXd v_tau_a, v_tau_b;
        if (is_mgga) {
            func.eval_polarised_mgga(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                                     tau_a, tau_b,
                                     exc, v_rho_a, v_rho_b,
                                     v_sig_aa, v_sig_ab, v_sig_bb,
                                     v_tau_a, v_tau_b);
        } else {
            func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                                exc, v_rho_a, v_rho_b,
                                v_sig_aa, v_sig_ab, v_sig_bb);
        }

        e_xc += w_b.dot(exc);

        // --- V_σ(g) += per-batch contribution -----------------------------
        //   V_σ(g)_{μν} = Σ_{(a,s): s−a=g} Σ_r w(r) v_ρσ(r) χ_a(r)_μ χ_s(r)_ν
        //                 (+ GGA flow / mGGA τ terms). Grouped by gp so each
        //   spin's block is written race-free in parallel within the batch.
        const Eigen::VectorXd w_vra = w_b.array() * v_rho_a.array();
        const Eigen::VectorXd w_vrb = w_b.array() * v_rho_b.array();
        // Per-spin GGA "flow vectors":
        //   f_α = 2 v_σαα ∇ρ_α + v_σαβ ∇ρ_β ;  f_β = 2 v_σββ ∇ρ_β + v_σαβ ∇ρ_α
        Eigen::VectorXd w_fa_x, w_fa_y, w_fa_z, w_fb_x, w_fb_y, w_fb_z;
        if (need_grad) {
            w_fa_x = w_b.array() * (2.0 * v_sig_aa.array() * gax.array()
                                    + v_sig_ab.array() * gbx.array());
            w_fa_y = w_b.array() * (2.0 * v_sig_aa.array() * gay.array()
                                    + v_sig_ab.array() * gby.array());
            w_fa_z = w_b.array() * (2.0 * v_sig_aa.array() * gaz.array()
                                    + v_sig_ab.array() * gbz.array());
            w_fb_x = w_b.array() * (2.0 * v_sig_bb.array() * gbx.array()
                                    + v_sig_ab.array() * gax.array());
            w_fb_y = w_b.array() * (2.0 * v_sig_bb.array() * gby.array()
                                    + v_sig_ab.array() * gay.array());
            w_fb_z = w_b.array() * (2.0 * v_sig_bb.array() * gbz.array()
                                    + v_sig_ab.array() * gaz.array());
        }
        Eigen::VectorXd w_vta, w_vtb;
        if (is_mgga) {
            w_vta = 0.5 * w_b.array() * v_tau_a.array();
            w_vtb = 0.5 * w_b.array() * v_tau_b.array();
        }

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t gp = 0; gp < n_cells; ++gp) {
            if (pairs_by_gp[gp].empty()) continue;
            Eigen::MatrixXd Va = Eigen::MatrixXd::Zero(nbf, nbf);
            Eigen::MatrixXd Vb = Eigen::MatrixXd::Zero(nbf, nbf);
            for (int k : pairs_by_gp[gp]) {
                const BraKetPair& pr = pairs[k];
                const Eigen::MatrixXd& chi_a = chi_h[pr.a];
                const Eigen::MatrixXd& chi_s = chi_h[pr.s];
                // LDA pieces.
                Va.noalias() += chi_a.transpose() * w_vra.asDiagonal() * chi_s;
                Vb.noalias() += chi_a.transpose() * w_vrb.asDiagonal() * chi_s;

                if (need_grad) {
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    // V_σ(g) += [w·f_σ · ∇χ_a]^T χ_s + χ_a^T [w·f_σ · ∇χ_s]
                    Eigen::MatrixXd Aa = da[0].array().colwise() * w_fa_x.array();
                    Aa.array() += da[1].array().colwise() * w_fa_y.array();
                    Aa.array() += da[2].array().colwise() * w_fa_z.array();
                    Va.noalias() += Aa.transpose() * chi_s;
                    Eigen::MatrixXd Ba = ds[0].array().colwise() * w_fa_x.array();
                    Ba.array() += ds[1].array().colwise() * w_fa_y.array();
                    Ba.array() += ds[2].array().colwise() * w_fa_z.array();
                    Va.noalias() += chi_a.transpose() * Ba;

                    Eigen::MatrixXd Ab = da[0].array().colwise() * w_fb_x.array();
                    Ab.array() += da[1].array().colwise() * w_fb_y.array();
                    Ab.array() += da[2].array().colwise() * w_fb_z.array();
                    Vb.noalias() += Ab.transpose() * chi_s;
                    Eigen::MatrixXd Bb = ds[0].array().colwise() * w_fb_x.array();
                    Bb.array() += ds[1].array().colwise() * w_fb_y.array();
                    Bb.array() += ds[2].array().colwise() * w_fb_z.array();
                    Vb.noalias() += chi_a.transpose() * Bb;
                }

                if (is_mgga) {
                    const std::array<Eigen::MatrixXd, 3>& da = dchi_h[pr.a];
                    const std::array<Eigen::MatrixXd, 3>& ds = dchi_h[pr.s];
                    for (int d = 0; d < 3; ++d) {
                        Va.noalias() +=
                            da[d].transpose() * w_vta.asDiagonal() * ds[d];
                        Vb.noalias() +=
                            da[d].transpose() * w_vtb.asDiagonal() * ds[d];
                    }
                }
            }
            out.V_alpha.blocks[gp].noalias() += Va;
            out.V_beta.blocks[gp].noalias() += Vb;
        }
    }

    out.e_xc = e_xc;
    return out;
}


// Conservative numerical workspace per point: retained AO tables, the
// largest concurrent pair contractions, point vectors, and shifted grids.
// Input density, the input grid, cell-pair metadata, and atom accumulators
// are accounted separately by the caller's whole-calculation memory model.
std::size_t periodic_xc_gradient_batch_size(
    std::size_t nbf, std::size_t n_cells, bool need_hess,
    bool open_shell, std::size_t n_threads) {
    if (nbf == 0 || n_cells == 0 || n_threads == 0) {
        throw std::invalid_argument("periodic XC gradient batch dimensions must be positive");
    }
    const long double tables = need_hess ? 10 : 4;
    const long double scratch = open_shell
        ? (need_hess ? 18 : 6) : (need_hess ? 10 : 4);
    const long double vectors = open_shell ? 40 : 24;
    const long double per_point = sizeof(double) * (
        nbf * (tables * (n_cells + 1.0L) + n_threads * scratch)
        + vectors + 3 * n_threads + 4);
    constexpr std::size_t workspace_bytes = 256ULL * 1024 * 1024;
    if (!std::isfinite(per_point) || per_point > workspace_bytes) {
        throw std::runtime_error("periodic XC gradient cannot fit one grid point in its workspace");
    }
    return std::min<std::size_t>(4096, workspace_bytes / per_point);
}

Eigen::MatrixXd xc_lattice_gradient_contribution(
    const BasisSet& basis, const PeriodicSystem& system,
    const Grid& grid, const Functional& func,
    const LatticeMatrixSet& density, const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain) {
    if (func.is_external()) {
        throw std::runtime_error(
            "xc_lattice_gradient_contribution: analytic gradients for "
            "full-grid external XC require grid/image/partition response "
            "terms that are not implemented");
    }
    if (grid.points.rows() != grid.weights.size()
        || !grid.points.allFinite() || !grid.weights.allFinite()) {
        throw std::invalid_argument("periodic XC gradient requires a finite aligned quadrature");
    }
    const auto batch_size = static_cast<Eigen::Index>(periodic_xc_gradient_batch_size(
        basis.nbasis(), active_density_cells(density.cells, system, opts.cutoff_bohr).size(),
        func.kind() != XCKind::LDA,
        false, omp_max_threads()));
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(system.unit_cell.size(), 3);
    for (Eigen::Index first = 0; first < grid.points.rows(); first += batch_size) {
        const auto count = std::min(batch_size, grid.points.rows() - first);
        Grid batch;
        batch.points = grid.points.middleRows(first, count);
        batch.weights = grid.weights.segment(first, count);
        gradient += xc_lattice_gradient_batch(
            basis, system, batch, func, density, opts, density_domain);
    }
    return gradient;
}


Eigen::MatrixXd xc_lattice_gradient_contribution_uks(
    const BasisSet& basis, const PeriodicSystem& system,
    const Grid& grid, const Functional& func,
    const LatticeMatrixSet& alpha, const LatticeMatrixSet& beta, const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain) {
    if (func.is_external()) {
        throw std::runtime_error(
            "xc_lattice_gradient_contribution_uks: analytic gradients for "
            "full-grid external XC require grid/image/partition response "
            "terms that are not implemented");
    }
    if (grid.points.rows() != grid.weights.size()
        || !grid.points.allFinite() || !grid.weights.allFinite()) {
        throw std::invalid_argument("periodic XC gradient requires a finite aligned quadrature");
    }
    const auto batch_size = static_cast<Eigen::Index>(periodic_xc_gradient_batch_size(
        basis.nbasis(), active_density_cells(alpha.cells, system, opts.cutoff_bohr).size(),
        func.kind() != XCKind::LDA,
        true, omp_max_threads()));
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(system.unit_cell.size(), 3);
    for (Eigen::Index first = 0; first < grid.points.rows(); first += batch_size) {
        const auto count = std::min(batch_size, grid.points.rows() - first);
        Grid batch;
        batch.points = grid.points.middleRows(first, count);
        batch.weights = grid.weights.segment(first, count);
        gradient += xc_lattice_gradient_batch_uks(
            basis, system, batch, func, alpha, beta, opts, density_domain);
    }
    return gradient;
}

}  // namespace vibeqc
