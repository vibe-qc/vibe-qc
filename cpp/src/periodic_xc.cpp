#include "vibeqc/periodic_xc.hpp"

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/bloch.hpp"
#include "vibeqc/periodic_bloch.hpp"
#include "vibeqc/ao_eval_subset.hpp"
#include "vibeqc/schwarz.hpp"
#include "vibeqc/thread_pool.hpp"
#include "vibeqc/detail/native_phase_timing.hpp"

#include <libint2/atom.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
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


// ---- AO-reach screening of lattice images (vibe-qc#268, #299, #53) -------
//
// The value kernels used to evaluate every AO of every active image cell at
// every point of a grid batch and to contract every bra-ket cell pair with
// dense nbf x nbf density blocks, so their cost grew with (cells within
// cutoff_bohr)^2 x nbf^2 however far each shell actually reaches. The image
// DOMAIN (active_density_cells, the bra radius, cutoff_bohr) is unchanged
// here; the screen only drops shells of an image whose Gaussian extent
// cannot reach the batch: shell s of image cell c is kept iff
//   dist(O_s + R_c, box(batch points)) < r_cut[s],
// with r_cut from compute_shell_radial_cutoffs at the screening tolerance
// (an AO amplitude bound, as in the molecular COSX builders). An image with
// no kept shell drops out of the batch, and so do its bra-ket pairs. A
// tolerance <= 0 disables the screen: every shell is kept, every column and
// contraction is the dense one, and the result is bitwise the dense kernel.
struct ImageShellScreen {
    bool enabled = false;
    std::vector<Eigen::Vector3d> centre;   // home-cell shell centres
    std::vector<double> cut2;              // squared radial cutoffs
    std::vector<int> bf0;                  // first basis function of shell
    std::vector<int> nbf;                  // functions in shell
    // Primitive data for the amplitude bound (shell_amplitude_bound).
    std::vector<std::vector<double>> alpha;
    std::vector<std::vector<double>> abs_coeff;
    std::vector<int> l;
    double pair_tol = 0.0;                 // bound on a pair's |rho| term
};

// Upper bound on |chi| and |grad chi| of any function of shell s at distance
// >= d from its centre: sum_p |c_p| r^l exp(-a_p r^2) is bounded by its value
// at max(d, r_peak) per primitive; the Cartesian monomial is <= r^l, a
// spherical combination of them is bounded by the number of Cartesians
// times 2, and the gradient adds at most a factor (2 a_p r + l / r).
double shell_amplitude_bound(const ImageShellScreen& scr, std::size_t s,
                             double d) {
    const int l = scr.l[s];
    double b = 0.0;
    for (std::size_t p = 0; p < scr.alpha[s].size(); ++p) {
        const double a = scr.alpha[s][p];
        const double r_peak = l > 0 ? std::sqrt((l + 1) / (2.0 * a)) : 0.0;
        const double r = std::max(d, r_peak);
        const double val = std::pow(r, l) * std::exp(-a * r * r);
        const double grad = 2.0 * a * r + (r > 0.0 ? l / r : 0.0) + 1.0;
        b += scr.abs_coeff[s][p] * val * grad;
    }
    const double n_cart = 0.5 * (l + 1) * (l + 2);
    return b * 2.0 * n_cart;
}

ImageShellScreen make_image_shell_screen(const BasisSet& basis, double tol) {
    ImageShellScreen scr;
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    scr.enabled = std::isfinite(tol) && tol > 0.0;
    std::vector<double> cut;
    if (scr.enabled) cut = compute_shell_radial_cutoffs(shells, tol);
    const std::size_t n = shells.size();
    scr.centre.resize(n);
    scr.cut2.resize(n);
    scr.bf0.resize(n);
    scr.nbf.resize(n);
    for (std::size_t s = 0; s < n; ++s) {
        scr.centre[s] = Eigen::Vector3d(shells[s].O[0], shells[s].O[1],
                                        shells[s].O[2]);
        scr.cut2[s] = scr.enabled
            ? cut[s] * cut[s] : std::numeric_limits<double>::infinity();
        scr.bf0[s] = static_cast<int>(shell2bf[s]);
        scr.nbf[s] = static_cast<int>(shells[s].size());
        const auto& contr = shells[s].contr[0];
        scr.l.push_back(contr.l);
        scr.alpha.emplace_back(shells[s].alpha.begin(), shells[s].alpha.end());
        std::vector<double> c;
        for (double x : contr.coeff) c.push_back(std::abs(x));
        scr.abs_coeff.push_back(std::move(c));
    }
    // A pair is dropped when its bound on |rho| (and |grad rho|) at every
    // batch point is below tol: the scale of one AO at its cutoff amplitude
    // against an O(1) partner, the term the shell screen already drops.
    scr.pair_tol = scr.enabled ? tol : 0.0;
    return scr;
}

// AO table of one lattice image restricted to its live shells in one batch.
struct ScreenedImageAO {
    std::vector<int> shells;   // live shells, increasing
    std::vector<int> bfs;      // their basis functions (home numbering)
    bool full = false;         // every shell live: bfs == 0..nbf-1
    double bound = 0.0;        // sum over live functions of |chi| bounds
    Eigen::MatrixXd chi;                   // (nb, bfs.size())
    std::array<Eigen::MatrixXd, 3> dchi;   // gradients, same shape
    bool live() const { return !bfs.empty(); }
};

// Shells of the image at R that reach the box [lo, hi].
void select_live_shells(const ImageShellScreen& scr, const Eigen::Vector3d& R,
                        const Eigen::Vector3d& lo, const Eigen::Vector3d& hi,
                        ScreenedImageAO& img) {
    img.shells.clear();
    img.bfs.clear();
    img.bound = 0.0;
    const std::size_t n = scr.centre.size();
    for (std::size_t s = 0; s < n; ++s) {
        if (scr.enabled) {
            const Eigen::Vector3d X = scr.centre[s] + R;
            double d2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double d = std::max({0.0, lo[k] - X[k], X[k] - hi[k]});
                d2 += d * d;
            }
            if (!(d2 < scr.cut2[s])) continue;
            img.bound += scr.nbf[s] * shell_amplitude_bound(scr, s, std::sqrt(d2));
        }
        img.shells.push_back(static_cast<int>(s));
        for (int f = 0; f < scr.nbf[s]; ++f) img.bfs.push_back(scr.bf0[s] + f);
    }
    img.full = img.shells.size() == n;
}

void batch_box(const Eigen::MatrixX3d& pts, Eigen::Vector3d& lo,
               Eigen::Vector3d& hi) {
    if (pts.rows() == 0) {
        lo.setZero();
        hi.setZero();
        return;
    }
    lo = pts.colwise().minCoeff().transpose();
    hi = pts.colwise().maxCoeff().transpose();
}

// Screened counterpart of evaluate_active_shifted_ao: fills tab[c] for the
// active cells whose image reaches the batch; other entries stay empty.
void evaluate_screened_shifted_ao(const BasisSet& basis,
                                  const ImageShellScreen& scr,
                                  const Eigen::MatrixX3d& points,
                                  const std::vector<LatticeCell>& cells,
                                  const std::vector<int>& active,
                                  bool need_grad,
                                  std::vector<ScreenedImageAO>& tab) {
    Eigen::Vector3d lo, hi;
    batch_box(points, lo, hi);
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t ai = 0; ai < active.size(); ++ai) {
        const int c = active[ai];
        const Eigen::Vector3d& R = cells[c].r_cart;
        ScreenedImageAO& img = tab[c];
        select_live_shells(scr, R, lo, hi, img);
        if (!img.live()) continue;
        if (R.isZero()) {
            evaluate_ao_shell_subset(basis, points, img.shells, img.chi,
                                     need_grad ? &img.dchi : nullptr);
            continue;
        }
        Eigen::MatrixX3d pts = points;
        pts.rowwise() -= R.transpose();
        evaluate_ao_shell_subset(basis, pts, img.shells, img.chi,
                                 need_grad ? &img.dchi : nullptr);
    }
}

// P(g) restricted to the live functions of a bra/ket image pair. Returns the
// block itself when both images are complete (no copy, dense arithmetic).
const Eigen::MatrixXd& pair_density_block(const Eigen::MatrixXd& P,
                                          const ScreenedImageAO& A,
                                          const ScreenedImageAO& S,
                                          Eigen::MatrixXd& buffer) {
    if (A.full && S.full) return P;
    buffer = P(A.bfs, S.bfs);
    return buffer;
}

// Pairs whose bra and ket images both reach the batch and whose term can
// exceed the pair tolerance, in pair order (so the static partition over
// them is reproducible, #81). For the density, |chi_a P(g) chi_s| <=
// B_a max|P(g)| B_s. The potential V(g) = dE/dP(g) does not scale with P(g)
// (a vanishing P(g), e.g. a home-cell density, still has a V(g) block), so
// its pairs are screened on B_a B_s alone: pass p_max = nullptr.
std::vector<int> live_pair_indices(const std::vector<BraKetPair>& pairs,
                                   const std::vector<ScreenedImageAO>& tab,
                                   const ImageShellScreen& scr,
                                   const std::vector<double>* p_max) {
    std::vector<int> live;
    live.reserve(pairs.size());
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        const ScreenedImageAO& a = tab[pairs[k].a];
        const ScreenedImageAO& s = tab[pairs[k].s];
        if (!a.live() || !s.live()) continue;
        if (scr.enabled) {
            const double scale = p_max ? (*p_max)[pairs[k].gp] : 1.0;
            if (a.bound * s.bound * scale < scr.pair_tol) continue;
        }
        live.push_back(static_cast<int>(k));
    }
    return live;
}

std::vector<double> block_abs_max(const LatticeMatrixSet& P,
                                  const LatticeMatrixSet* Q = nullptr) {
    std::vector<double> m(P.blocks.size(), 0.0);
    for (std::size_t c = 0; c < P.blocks.size(); ++c) {
        if (P.blocks[c].size()) m[c] = P.blocks[c].cwiseAbs().maxCoeff();
        if (Q && Q->blocks[c].size()) {
            m[c] = std::max(m[c], Q->blocks[c].cwiseAbs().maxCoeff());
        }
    }
    return m;
}

// Edge (bohr) of the cubic bins that order the screened points.
constexpr double kScreenedXcBinBohr = 2.0;

// Points carrying a nonzero quadrature weight, in spatially compact order.
// The libxc value kernels are pointwise, so a zero-weight point contributes
// to neither E_xc nor V_xc, and the order of the points only reassociates
// the sums. The atom-major grid lists whole radial shells (spheres up to
// ~15 bohr), so contiguous batches of it have boxes as large as the image
// domain and the screen could drop nothing; sorting the points by cubic bin
// makes each batch a compact region. Only used with the screen enabled; the
// disabled path keeps every point in grid order so that it stays bitwise the
// dense kernel.
void nonzero_weight_points(const Grid& grid, Eigen::MatrixX3d& points,
                           Eigen::VectorXd& weights) {
    const Eigen::Index n = grid.points.rows();
    struct Key { std::array<long long, 3> bin; Eigen::Index index; };
    std::vector<Key> keys;
    keys.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        if (grid.weights[i] == 0.0) continue;
        Key k;
        for (int c = 0; c < 3; ++c) {
            k.bin[c] = static_cast<long long>(
                std::floor(grid.points(i, c) / kScreenedXcBinBohr));
        }
        k.index = i;
        keys.push_back(k);
    }
    std::stable_sort(keys.begin(), keys.end(),
                     [](const Key& a, const Key& b) { return a.bin < b.bin; });
    const auto m = static_cast<Eigen::Index>(keys.size());
    points.resize(m, 3);
    weights.resize(m);
    for (Eigen::Index j = 0; j < m; ++j) {
        const Eigen::Index i = keys[static_cast<std::size_t>(j)].index;
        points.row(j) = grid.points.row(i);
        weights[j] = grid.weights[i];
    }
}

// Upper bound on points per batch when screening: smaller batches give
// tighter boxes (the grid is atom-major, radial shell by radial shell).
constexpr Eigen::Index kScreenedXcBatch = 512;

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
            // The gradient and kinetic density share this matrix product.
            const Eigen::MatrixXd gradient_density = da[c] * P;
            grad_t[c][tid].array() +=
                (chi_aP.array() * ds[c].array()).rowwise().sum();
            grad_t[c][tid].array() +=
                (gradient_density.array() * chi_s.array()).rowwise().sum();
            tau_t[tid].array() +=
                (gradient_density.array() * ds[c].array()).rowwise().sum();
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

// Finite-torus regrouping is used only by the direct-k wrappers below. The
// ordinary V(g) API never exposes its compressed projection intermediates.
struct XCTorusContext {
    bool compatible = false;
    bool used = false;
    std::vector<std::array<int, 3>> residue;
    std::vector<double> sign;
    std::map<std::array<int, 3>, int> cell_pos;
    // Complete torus density (opt-in, #519 RG-1): every pair of live images
    // through the density of its torus residue, not only the retained pair
    // graph. Requires an unshifted mesh (all signs +1) and a density block
    // for every residue; `rep` is the first cell of each residue.
    bool complete = false;
    bool unit_signs = true;
    std::array<int, 3> mesh{1, 1, 1};
    std::map<std::array<int, 3>, int> rep;
};

XCTorusContext xc_torus_context(const PeriodicSystem& system,
                                const BlochKMesh& mesh,
                                const LatticeMatrixSet& p,
                                const LatticeMatrixSet* beta = nullptr) {
    XCTorusContext ctx;
    if (mesh.kpoints.empty())
        throw std::invalid_argument("periodic k-point XC requires k-points");
    for (const auto& k : mesh.kpoints)
        if (!k.allFinite())
            throw std::invalid_argument("periodic k-point XC requires finite k-points");
    // Ad hoc lists, reduced meshes and unusual weightings retain the pair path.
    std::size_t nk = 1;
    for (int d = 0; d < 3; ++d) {
        if (mesh.mesh[d] <= 0 || mesh.is_shift[d] < 0 || mesh.is_shift[d] > 1)
            return ctx;
        if (d >= system.dim && (mesh.mesh[d] != 1 || mesh.is_shift[d] != 0))
            return ctx;
        if (nk > mesh.kpoints.size() / static_cast<std::size_t>(mesh.mesh[d]))
            return ctx;
        nk *= static_cast<std::size_t>(mesh.mesh[d]);
    }
    if (nk != mesh.kpoints.size() || mesh.weights.size() != nk) return ctx;
    if (!mesh.ir_mapping.empty()) {
        if (mesh.ir_mapping.size() != nk) return ctx;
        for (std::size_t k = 0; k < nk; ++k)
            if (mesh.ir_mapping[k] != static_cast<int>(k)) return ctx;
    }
    const auto expected = monkhorst_pack(system, mesh.mesh, mesh.is_shift, false);
    for (std::size_t k = 0; k < nk; ++k) {
        if (!std::isfinite(mesh.weights[k]) ||
            std::abs(mesh.weights[k] - 1.0 / nk) > 1e-14 ||
            (mesh.kpoints[k] - expected.kpoints[k]).norm() > 1e-12)
            return ctx;
    }
    if (p.nbf <= 0 || p.cells.size() != p.blocks.size() || p.cells.empty())
        return ctx;
    if (beta && (beta->nbf != p.nbf || beta->cells.size() != p.cells.size() ||
                 beta->blocks.size() != p.blocks.size())) return ctx;
    const auto density_scale = [](const LatticeMatrixSet& density) {
        double scale = 0.0;
        for (const auto& block : density.blocks) {
            if (block.rows() != density.nbf || block.cols() != density.nbf || !block.allFinite())
                return -1.0;
            scale = std::max(scale, block.cwiseAbs().maxCoeff());
        }
        return scale;
    };
    const double scale_alpha = density_scale(p);
    const double scale_beta = beta ? density_scale(*beta) : 0.0;
    if (scale_alpha < 0.0 || scale_beta < 0.0) return ctx;
    std::map<std::array<int, 3>, std::size_t> representatives;
    for (std::size_t i = 0; i < p.cells.size(); ++i) {
        const auto& cell = p.cells[i];
        const auto& block = p.blocks[i];
        if (block.rows() != p.nbf || block.cols() != p.nbf || !block.allFinite())
            return ctx;
        if (!cell.r_cart.allFinite() ||
            (cell.r_cart - system.lattice * cell.index.cast<double>()).norm() > 1e-12)
            return ctx;
        std::array<int, 3> residue, index;
        int parity = 0;
        for (int d = 0; d < 3; ++d) {
            index[d] = cell.index[d];
            // Floor division, including negative lattice indices.
            residue[d] = index[d] % mesh.mesh[d];
            if (residue[d] < 0) residue[d] += mesh.mesh[d];
            const auto quotient = (static_cast<long long>(index[d]) - residue[d]) / mesh.mesh[d];
            parity ^= static_cast<int>((quotient % 2 + 2) % 2) * mesh.is_shift[d];
        }
        if (!ctx.cell_pos.emplace(index, static_cast<int>(i)).second) return ctx;
        ctx.residue.push_back(residue);
        ctx.sign.push_back(parity ? -1.0 : 1.0);
        if (parity) ctx.unit_signs = false;
        if (beta && (beta->cells[i].index != cell.index ||
                     (beta->cells[i].r_cart - cell.r_cart).norm() > 1e-12 ||
                     beta->blocks[i].rows() != p.nbf || beta->blocks[i].cols() != p.nbf ||
                     !beta->blocks[i].allFinite())) return ctx;
        const auto [it, inserted] = representatives.emplace(residue, i);
        if (inserted) continue;
        const auto ref = it->second;
        const double factor = ctx.sign[i] * ctx.sign[ref];
        const auto agrees = [factor](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                      double scale) {
            // Fourier phases can span several periods. Admit only roundoff
            // relative to this spin's density scale, including near-zero
            // residue blocks; a tiny spin channel gets no unit-sized floor.
            return (a - factor * b).cwiseAbs().maxCoeff() <=
                256.0 * std::numeric_limits<double>::epsilon() * scale;
        };
        if (!agrees(block, p.blocks[ref], scale_alpha) ||
            (beta && !agrees(beta->blocks[i], beta->blocks[ref], scale_beta))) return ctx;
    }
    for (const auto& entry : representatives)
        ctx.rep.emplace(entry.first, static_cast<int>(entry.second));
    for (int d = 0; d < 3; ++d) {
        ctx.mesh[d] = mesh.mesh[d];
        if (mesh.is_shift[d] != 0) ctx.unit_signs = false;
    }
    ctx.compatible = true;
    return ctx;
}

struct XCTorusRectangle {
    std::vector<int> bra, ket;
    int gp;
    double sign;
};
using XCTorusPlan = std::vector<XCTorusRectangle>;

// The complete torus plan (opt-in, #519 RG-1): one rectangle per pair of torus
// residues over all images that reach the batch, so the density is
// sum_{a,s} chi_a^T P(res(s) - res(a)) chi_s over every pair of live images.
// It is the domain-converged density: on a domain that closes the density
// (the #465 bound) it differs from the retained pair graph by the pairs
// beyond the domain only, 7e-12 Ha in E_xc on LiH/STO-3G (1,1,1) and 3e-10 Ha
// on an H2/STO-3G (2,1,1) cube. Its AO sums cost (images x residues) instead
// of (images x pair-ball size). Returns {} (the caller's regular plan) when a
// residue has no density block or a sign is -1.
XCTorusPlan xc_torus_complete_plan(const XCTorusContext& ctx,
                                   const std::vector<ScreenedImageAO>& tab) {
    if (!ctx.unit_signs) return {};
    std::map<std::array<int, 3>, std::vector<int>> by_residue;
    for (std::size_t c = 0; c < tab.size(); ++c)
        if (tab[c].live()) by_residue[ctx.residue[c]].push_back(static_cast<int>(c));
    XCTorusPlan plan;
    for (const auto& bra : by_residue)
        for (const auto& ket : by_residue) {
            std::array<int, 3> d;
            for (int k = 0; k < 3; ++k)
                d[k] = ((ket.first[k] - bra.first[k]) % ctx.mesh[k] + ctx.mesh[k]) % ctx.mesh[k];
            const auto found = ctx.rep.find(d);
            if (found == ctx.rep.end()) return {};
            XCTorusRectangle r;
            r.bra = bra.second;
            r.ket = ket.second;
            r.gp = found->second;
            r.sign = 1.0;
            plan.push_back(std::move(r));
        }
    return plan;
}

// A disjoint cover of the RETAINED pair graph, partitioned by both residues.
// No complement, new pair, larger image domain or altered screen is involved.
XCTorusPlan xc_torus_plan(XCTorusContext* ctx,
                          const std::vector<BraKetPair>& pairs,
                          const std::vector<int>& live,
                          const std::vector<ScreenedImageAO>* tab,
                          int nbf) {
    if (ctx && ctx->complete && tab) {
        auto complete = xc_torus_complete_plan(*ctx, *tab);
        if (!complete.empty()) {
            ctx->used = true;
            return complete;
        }
    }
    if (!ctx || live.empty()) return {};
    using Key = std::pair<std::array<int, 3>, std::array<int, 3>>;
    std::map<Key, std::vector<int>> blocks;
    // Uniform small meshes revisit the same residue pairs many times. Index
    // their map values directly, retaining the map's original ordering and
    // stable nodes. Bound scratch to 4096 pointers and avoid sparse indexes
    // with more slots than retained pairs; larger meshes keep map lookup.
    std::size_t residues = 1;
    for (int m : ctx->mesh) {
        if (residues > 64 / static_cast<std::size_t>(m)) {
            residues = 0;
            break;
        }
        residues *= static_cast<std::size_t>(m);
    }
    std::vector<std::vector<int>*> index;
    if (residues && residues * residues <= live.size())
        index.assign(residues * residues, nullptr);
    const auto residue_index = [&](const std::array<int, 3>& r) {
        return (static_cast<std::size_t>(r[0]) * ctx->mesh[1] + r[1])
            * ctx->mesh[2] + r[2];
    };
    long double pair_work = 0.0;
    for (int i : live) {
        const auto& p = pairs[i];
        if (index.empty()) {
            blocks[{ctx->residue[p.a], ctx->residue[p.s]}].push_back(i);
        } else {
            auto*& block = index[residue_index(ctx->residue[p.a]) * residues
                                 + residue_index(ctx->residue[p.s])];
            if (!block) block = &blocks[{ctx->residue[p.a], ctx->residue[p.s]}];
            block->push_back(i);
        }
        pair_work += tab ? (*tab)[p.a].bfs.size() * (*tab)[p.s].bfs.size()
                         : static_cast<long double>(nbf) * nbf;
    }
    XCTorusPlan plan;
    for (const auto& entry : blocks) {
        auto cover = [&](bool transpose) {
            std::map<int, std::vector<int>> neighbors;
            for (int i : entry.second) {
                const auto& p = pairs[i];
                neighbors[transpose ? p.s : p.a].push_back(transpose ? p.a : p.s);
            }
            std::map<std::vector<int>, std::vector<int>> equal_rows;
            for (auto& row : neighbors) {
                std::sort(row.second.begin(), row.second.end());
                equal_rows[row.second].push_back(row.first);
            }
            XCTorusPlan result;
            for (const auto& row : equal_rows) {
                XCTorusRectangle r;
                r.bra = transpose ? row.first : row.second;
                r.ket = transpose ? row.second : row.first;
                // Every rectangle edge exists. Find its representative pair
                // without constructing a dense image-by-image lookup table.
                const int a = r.bra.front(), s = r.ket.front();
                const auto found = std::find_if(entry.second.begin(), entry.second.end(),
                    [&](int i) { return pairs[i].a == a && pairs[i].s == s; });
                r.gp = pairs[*found].gp;
                r.sign = ctx->sign[a] * ctx->sign[s];
                result.push_back(std::move(r));
            }
            return result;
        };
        auto rows = cover(false), columns = cover(true);
        auto& best = rows.size() <= columns.size() ? rows : columns;
        for (auto& rectangle : best) plan.push_back(std::move(rectangle));
    }
    // Full-basis rectangle sums can lose to sparse tiny pairs. Include their
    // additions in a conservative work estimate before choosing the path.
    long double grouped_work = static_cast<long double>(plan.size()) * nbf * nbf;
    for (const auto& r : plan) grouped_work +=
        static_cast<long double>(r.bra.size() + r.ket.size()) * nbf;
    if (grouped_work * 1.25L >= pair_work) return {};
    ctx->used = true;
    return plan;
}

struct XCTorusAOView {
    const Eigen::MatrixXd& chi;
    const std::array<Eigen::MatrixXd, 3>& grad;
    const std::vector<int>* bfs; // null means the complete basis
};
struct XCTorusAO {
    Eigen::MatrixXd chi;
    std::array<Eigen::MatrixXd, 3> grad;
};
template<class View>
XCTorusAO xc_torus_sum(const std::vector<int>& cells, const XCTorusContext& ctx,
                       View view, Eigen::Index points, int nbf, bool gradient) {
    XCTorusAO out;
    out.chi = Eigen::MatrixXd::Zero(points, nbf);
    if (gradient) for (auto& g : out.grad) g = Eigen::MatrixXd::Zero(points, nbf);
    for (int c : cells) {
        const auto ao = view(c);
        if (!ao.bfs) {
            out.chi += ctx.sign[c] * ao.chi;
            if (gradient) for (int d = 0; d < 3; ++d) out.grad[d] += ctx.sign[c] * ao.grad[d];
        } else {
            for (std::size_t j = 0; j < ao.bfs->size(); ++j) {
                const int bf = (*ao.bfs)[j];
                out.chi.col(bf) += ctx.sign[c] * ao.chi.col(j);
                if (gradient) for (int d = 0; d < 3; ++d)
                    out.grad[d].col(bf) += ctx.sign[c] * ao.grad[d].col(j);
            }
        }
    }
    return out;
}

template<std::size_t Spins, class View>
std::array<ExternalPeriodicFields, Spins> xc_torus_fields_channels(
    const std::array<const LatticeMatrixSet*, Spins>& densities,
    const XCTorusPlan& plan, const XCTorusContext& ctx, View view,
    Eigen::Index points, bool gradient, bool kinetic) {
    const int nt = omp_max_threads();
    const int nbf = densities[0]->nbf;
    std::vector<std::array<ExternalPeriodicFields, Spins>> local(nt);
    for (auto& channels : local) for (auto& f : channels) {
        f.rho = Eigen::VectorXd::Zero(points);
        if (gradient) f.grad = Eigen::MatrixX3d::Zero(points, 3);
        if (kinetic) f.tau = Eigen::VectorXd::Zero(points);
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < plan.size(); ++j) {
        auto& channels = local[omp_thread_index()];
        const auto& r = plan[j];
        const auto a = xc_torus_sum(r.bra, ctx, view, points, nbf, gradient);
        // Identical ordered image lists give identical signed AO sums.
        const bool same_images = r.bra == r.ket;
        // Avoid GCC 12's aggregate-temporary noexcept ICE in this template.
        XCTorusAO separate_s;
        if (!same_images)
            separate_s = xc_torus_sum(r.ket, ctx, view, points, nbf, gradient);
        const auto& s = same_images ? a : separate_s;
        // Alpha/beta use identical signed AO sums but independent density
        // products. Reuse only this rectangle's immutable AO tables; keep
        // each spin's static partition and accumulation order unchanged.
        // Poschel et al., arXiv:2608.19033, Eq. (2), applies per spin.
        for (std::size_t spin = 0; spin < Spins; ++spin) {
            auto& f = channels[spin];
            const Eigen::MatrixXd density = r.sign * densities[spin]->blocks[r.gp];
            const Eigen::MatrixXd ap = a.chi * density;
            f.rho.array() += (ap.array() * s.chi.array()).rowwise().sum();
            // Differentiate the AO-density bilinear form (Poschel et al.,
            // arXiv:2608.19033, Eq. (2)). For GGA, one adjoint product replaces
            // three gradient-density products: (da P) . s = da . (s P^T).
            // Individual lattice density blocks need not be symmetric. The
            // kinetic-density path still needs all three da P products.
            Eigen::MatrixXd sp;
            if (gradient && !kinetic) sp.noalias() = s.chi * density.transpose();
            if (gradient) for (int d = 0; d < 3; ++d) {
                f.grad.col(d).array() += (ap.array() * s.grad[d].array()).rowwise().sum();
                if (kinetic) {
                    const Eigen::MatrixXd dp = a.grad[d] * density;
                    f.grad.col(d).array() += (dp.array() * s.chi.array()).rowwise().sum();
                    f.tau.array() += 0.5 * (dp.array() * s.grad[d].array()).rowwise().sum();
                } else {
                    f.grad.col(d).array() += (a.grad[d].array() * sp.array()).rowwise().sum();
                }
            }
        }
    }
    for (int t = 1; t < nt; ++t) for (std::size_t spin = 0; spin < Spins; ++spin) {
        local[0][spin].rho += local[t][spin].rho;
        if (gradient) local[0][spin].grad += local[t][spin].grad;
        if (kinetic) local[0][spin].tau += local[t][spin].tau;
    }
    return std::move(local[0]);
}

template<class View>
ExternalPeriodicFields xc_torus_fields(const LatticeMatrixSet& p,
    const XCTorusPlan& plan, const XCTorusContext& ctx, View view,
    Eigen::Index points, bool gradient, bool kinetic) {
    auto fields = xc_torus_fields_channels(
        std::array<const LatticeMatrixSet*, 1>{&p}, plan, ctx, view,
        points, gradient, kinetic);
    return std::move(fields[0]);
}

struct XCTorusProjectionChannel {
    const Eigen::VectorXd& qr;
    const Eigen::MatrixX3d& qg;
    const Eigen::VectorXd& qt;
    std::vector<Eigen::MatrixXd>& blocks;
};

template<std::size_t Spins, class View>
void xc_torus_project_channels(const XCTorusPlan& plan, const XCTorusContext& ctx,
    View view, const std::array<XCTorusProjectionChannel, Spins>& channels) {
    static_assert(Spins == 1 || Spins == 2);
    const int nbf = static_cast<int>(channels[0].blocks.front().rows());
    const auto points = channels[0].qr.size();
    const bool gradient = channels[0].qg.rows() != 0, kinetic = channels[0].qt.size() != 0;
    // Repeated ordered image lists have exactly the same signed AO sums.
    // The AO-density and potential adjoint remain those of Poschel et al.,
    // arXiv:2608.19033, Eqs. (2), (9); this is host-side reuse only.
    // Keep only a bounded prefix of candidates and cache actual reuse, not
    // density values or grid state across calls. The planner reserves sixteen
    // extra AO images per worker; at most four cached six-table groups per
    // worker leave more than half that scratch for the original contractions.
    struct CachedGroup {
        const std::vector<int>* cells;
        std::size_t uses = 0;
        XCTorusAO ao;
        std::array<Eigen::MatrixXd, Spins> weighted_gradient;
    };
    const auto less = [](const std::vector<int>* a, const std::vector<int>* b) {
        return *a < *b;
    };
    std::map<const std::vector<int>*, std::size_t, decltype(less)> group_index(less);
    std::vector<CachedGroup> cache;
    const long double entry_bytes = sizeof(CachedGroup) + 128.0L
        + sizeof(double) * static_cast<long double>(points) * nbf
          * (gradient ? 4 + Spins : 1);
    const auto capacity = static_cast<std::size_t>(std::min<long double>(
        std::min<std::size_t>(64, 4 * static_cast<std::size_t>(omp_max_threads())),
        (64ULL * 1024 * 1024) / entry_bytes));
    cache.reserve(capacity);
    const auto count = [&](const std::vector<int>& cells) {
        const auto found = group_index.find(&cells);
        if (found != group_index.end()) {
            ++cache[found->second].uses;
        } else if (cache.size() < capacity) {
            group_index.emplace(&cells, cache.size());
            cache.push_back(CachedGroup{&cells, 1, {}, {}});
        }
    };
    if (capacity) for (const auto& rectangle : plan) {
        count(rectangle.bra);
        if (rectangle.bra != rectangle.ket) count(rectangle.ket);
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < cache.size(); ++i) {
        auto& c = cache[i];
        if (c.uses < 2) continue;
        c.ao = xc_torus_sum(*c.cells, ctx, view, points, nbf, gradient);
        if (gradient) for (std::size_t spin = 0; spin < Spins; ++spin) {
            const auto& qg = channels[spin].qg;
            auto& weighted = c.weighted_gradient[spin];
            weighted = c.ao.grad[0].array().colwise() * qg.col(0).array();
            for (int d = 1; d < 3; ++d)
                weighted.array() +=
                    c.ao.grad[d].array().colwise() * qg.col(d).array();
        }
    }
    const auto cached = [&](const std::vector<int>& cells) -> const CachedGroup* {
        const auto found = group_index.find(&cells);
        if (found == group_index.end() || cache[found->second].uses < 2) return nullptr;
        return &cache[found->second];
    };
    std::map<int, std::vector<int>> grouped;
    for (std::size_t j = 0; j < plan.size(); ++j) grouped[plan[j].gp].push_back(static_cast<int>(j));
    std::vector<std::pair<int, std::vector<int>>> outputs(grouped.begin(), grouped.end());
    // Output blocks are independent, while their retained rectangles can
    // differ greatly in count and image support. Balance whole blocks without
    // changing the signed AO or per-block rectangle accumulation order.
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        // At most two accumulated outputs and one contraction temporary,
        // within the open-shell planner's six fixed matrices per worker.
        std::array<Eigen::MatrixXd, Spins> values;
        for (auto& value : values) value = Eigen::MatrixXd::Zero(nbf, nbf);
        for (int j : outputs[i].second) {
            const auto& r = plan[j];
            const auto* cached_a = cached(r.bra);
            const bool same_images = r.bra == r.ket;
            const auto* cached_s = same_images ? cached_a : cached(r.ket);
            // Keep the same GCC 12 workaround as the density contraction.
            XCTorusAO separate_a, separate_s;
            if (!cached_a)
                separate_a = xc_torus_sum(r.bra, ctx, view, points, nbf, gradient);
            const auto& a = cached_a ? cached_a->ao : separate_a;
            if (!same_images && !cached_s)
                separate_s = xc_torus_sum(r.ket, ctx, view, points, nbf, gradient);
            const auto& s = same_images ? a : cached_s ? cached_s->ao : separate_s;
            // The signed AO sums are spin-independent. Apply each spin's
            // potential adjoint separately (Poschel et al., Eqs. (9), (10));
            // never share density-dependent weights between channels.
            for (std::size_t spin = 0; spin < Spins; ++spin) {
                const auto& qr = channels[spin].qr;
                const auto& qg = channels[spin].qg;
                const auto& qt = channels[spin].qt;
                Eigen::MatrixXd v = a.chi.transpose() * qr.asDiagonal() * s.chi;
                if (gradient) {
                    Eigen::MatrixXd separate_da, separate_ds;
                    if (!cached_a) {
                        separate_da = a.grad[0].array().colwise() * qg.col(0).array();
                        for (int d = 1; d < 3; ++d)
                            separate_da.array() += a.grad[d].array().colwise() * qg.col(d).array();
                    }
                    if (!cached_s) {
                        separate_ds = s.grad[0].array().colwise() * qg.col(0).array();
                        for (int d = 1; d < 3; ++d)
                            separate_ds.array() += s.grad[d].array().colwise() * qg.col(d).array();
                    }
                    const auto& da = cached_a ? cached_a->weighted_gradient[spin] : separate_da;
                    const auto& ds = cached_s ? cached_s->weighted_gradient[spin] : separate_ds;
                    v.noalias() += da.transpose() * s.chi;
                    v.noalias() += a.chi.transpose() * ds;
                    if (kinetic) for (int d = 0; d < 3; ++d)
                        v.noalias() += a.grad[d].transpose() * qt.asDiagonal() * s.grad[d];
                }
                values[spin] += r.sign * v;
            }
        }
        // These are compressed projection intermediates. Only their Bloch
        // sums on the verified mesh are returned by the direct-k wrappers.
        for (std::size_t spin = 0; spin < Spins; ++spin)
            channels[spin].blocks[outputs[i].first] += values[spin];
    }
}

template<class View>
void xc_torus_project(const XCTorusPlan& plan, const XCTorusContext& ctx,
    View view, const Eigen::VectorXd& qr, const Eigen::MatrixX3d& qg,
    const Eigen::VectorXd& qt, std::vector<Eigen::MatrixXd>& blocks) {
    xc_torus_project_channels(plan, ctx, view,
        std::array<XCTorusProjectionChannel, 1>{{{qr, qg, qt, blocks}}});
}

std::size_t xc_value_batch_size(int nbf, std::size_t active, bool gradient,
                                bool uks, XCTorusContext*& ctx, bool screened = false) {
    const auto threads = static_cast<std::size_t>(omp_max_threads());
    auto batch = periodic_xc_value_batch_size(nbf, active, gradient, uks, threads);
    if (screened) batch = std::min(batch, static_cast<std::size_t>(kScreenedXcBatch));
    if (!ctx) return batch;
    // Preserve the original batch boxes: changing them changes which shells
    // and pairs pass screening. If signed sums do not fit alongside the old
    // batch, use its original contractions instead of shrinking that batch.
    constexpr std::size_t extra_images_per_worker = 16;
    const auto scratch_images = extra_images_per_worker * threads;
    if (active > std::numeric_limits<std::size_t>::max() - scratch_images) {
        ctx = nullptr;
        return batch;
    }
    try {
        const auto admitted = periodic_xc_value_batch_size(
            nbf, active + scratch_images, gradient, uks, threads);
        if (admitted < batch) ctx = nullptr;
    } catch (const std::runtime_error&) {
        ctx = nullptr;
    }
    return batch;
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

PeriodicXCScreeningStats periodic_xc_screening_stats(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const LatticeMatrixSet& P_real_space,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain,
    double ao_screening_tol,
    bool open_shell,
    bool need_grad) {
    const int nbf = static_cast<int>(basis.nbasis());
    if (P_real_space.nbf != nbf) {
        throw std::runtime_error(
            "periodic_xc_screening_stats: density nbf mismatch");
    }
    // Same image domain and pair list as the value kernels.
    const auto cell_pos = build_cell_index_map(P_real_space.cells);
    const std::vector<int> active =
        active_density_cells(P_real_space.cells, system, opts.cutoff_bohr);
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr)
        : opts.cutoff_bohr;
    const bool use_periodic_bras = use_periodic_density_bras(
        density_domain, density_has_cross_cell_blocks(P_real_space, 1e-12));
    const std::vector<int> bra_cells = use_periodic_bras
        ? active_density_cells(P_real_space.cells, system, bra_radius)
        : home_only_bra(P_real_space.cells, active);
    const std::vector<BraKetPair> pairs = build_braket_pairs(
        P_real_space.cells, bra_cells, active, cell_pos,
        density_domain == PeriodicXCDensityDomain::PERIODIC_LATTICE);

    const ImageShellScreen screen =
        make_image_shell_screen(basis, ao_screening_tol);
    Eigen::MatrixX3d q_points;
    Eigen::VectorXd q_weights;
    if (screen.enabled) nonzero_weight_points(grid, q_points, q_weights);
    const Eigen::MatrixX3d& grid_pts = screen.enabled ? q_points : grid.points;
    const Eigen::Index n_q = grid_pts.rows();
    Eigen::Index batch = static_cast<Eigen::Index>(
        periodic_xc_value_batch_size(nbf, active.size(), need_grad,
                                     open_shell, omp_max_threads()));
    if (screen.enabled) batch = std::min(batch, kScreenedXcBatch);

    PeriodicXCScreeningStats st;
    st.n_points = static_cast<std::size_t>(grid.points.rows());
    st.n_points_evaluated = static_cast<std::size_t>(n_q);
    const std::size_t n_shells = screen.centre.size();
    std::vector<ScreenedImageAO> tab(P_real_space.cells.size());
    for (Eigen::Index g0 = 0; g0 < n_q; g0 += batch) {
        const Eigen::Index nb = std::min(batch, n_q - g0);
        Eigen::Vector3d lo, hi;
        batch_box(grid_pts.middleRows(g0, nb), lo, hi);
        for (auto& img : tab) {
            img.shells.clear();
            img.bfs.clear();
        }
        for (int c : active) {
            select_live_shells(screen, P_real_space.cells[c].r_cart, lo, hi,
                               tab[c]);
            st.image_shells_live += tab[c].shells.size();
        }
        st.image_shells_dense += active.size() * n_shells;
        st.pairs_dense += pairs.size();
        // The potential's pair list is the superset (no P(g) factor).
        st.pairs_live += live_pair_indices(pairs, tab, screen, nullptr).size();
        ++st.n_batches;
    }
    return st;
}

static PeriodicXCContribution build_xc_periodic_impl(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const Grid& grid,
                                         const Functional& func,
                                         const LatticeMatrixSet& P_real_space,
                                         const LatticeSumOptions& opts,
                                         PeriodicXCDensityDomain density_domain,
                                         double ao_screening_tol, XCTorusContext* torus) {
    detail::NativePhaseCall<8> profile("periodic_xc.closed_shell",
        {"setup", "ao_collocation", "density", "functional",
         "potential_projection", "external_provider_route", "pair_plan",
         "torus_potential_projection"},
        {{"nbasis", basis.nbasis()}, {"points", static_cast<std::uint64_t>(grid.points.rows())},
         {"cells", P_real_space.cells.size()}});
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
    out.periodic_density_bras = use_periodic_bras;
    double e_xc = 0.0;
    double n_grid = 0.0;

    // Output cell gp -> the bra/ket pairs that map to it (batch-independent).
    std::vector<std::vector<int>> pairs_by_gp(n_cells);
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        pairs_by_gp[pairs[k].gp].push_back(static_cast<int>(k));
    }

    profile.elapsed.finish(0);
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
            xc_value_batch_size(nbf, active.size(), true, false, torus));
        std::vector<int> all_pairs;
        if (torus) {
            all_pairs.resize(pairs.size());
            std::iota(all_pairs.begin(), all_pairs.end(), 0);
        }
        const auto external_plan = xc_torus_plan(torus, pairs, all_pairs, nullptr, nbf);
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
            const auto view = [&](int i) -> XCTorusAOView {
                return {chi_h[i], dchi_h[i], nullptr};
            };
            const ExternalPeriodicFields fields =
                external_plan.empty() ? build_external_periodic_fields_batch(
                    P_real_space, pairs, chi_h, dchi_h, nb)
                : xc_torus_fields(P_real_space, external_plan, *torus, view, nb, true, true);
            n_grid += grid.weights.segment(g0, nb).dot(fields.rho);
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
            const auto view = [&](int i) -> XCTorusAOView {
                return {chi_h[i], dchi_h[i], nullptr};
            };
            if (!external_plan.empty()) {
                xc_torus_project(external_plan, *torus, view,
                    q_rho.segment(g0, nb), q_grad.middleRows(g0, nb),
                    0.5 * q_tau.segment(g0, nb), out.V_xc.blocks);
            } else {
            project_external_periodic_batch(
                pairs, pairs_by_gp, chi_h, dchi_h,
                q_rho.segment(g0, nb), q_grad.middleRows(g0, nb),
                q_tau.segment(g0, nb), out.V_xc.blocks);
            }
        }
        profile.elapsed.finish(5);
        out.e_xc = ev.energy;
        out.n_electrons_grid = n_grid;
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
    //
    // AO-reach screening (see ImageShellScreen): with the screen enabled the
    // batches run over the nonzero-weight points only, each image carries
    // only the shells that reach the batch box, and pairs with an image that
    // reaches nothing are skipped. Disabled (tol <= 0), every point, shell
    // and pair is kept and the arithmetic is the dense kernel's.
    const ImageShellScreen screen =
        make_image_shell_screen(basis, ao_screening_tol);
    Eigen::MatrixX3d q_points;
    Eigen::VectorXd q_weights;
    if (screen.enabled) nonzero_weight_points(grid, q_points, q_weights);
    const Eigen::MatrixX3d& grid_pts = screen.enabled ? q_points : grid.points;
    const Eigen::VectorXd& grid_w = screen.enabled ? q_weights : grid.weights;
    const Eigen::Index n_q = grid_pts.rows();
    Eigen::Index kXcGridBatch = static_cast<Eigen::Index>(
            xc_value_batch_size(nbf, active.size(), need_grad, false, torus, screen.enabled));
    if (screen.enabled) kXcGridBatch = std::min(kXcGridBatch, kScreenedXcBatch);
    const std::vector<double> p_max = block_abs_max(P_real_space);
    profile.elapsed.finish(0);
    for (Eigen::Index g0 = 0; g0 < n_q; g0 += kXcGridBatch) {
        const Eigen::Index nb = std::min(kXcGridBatch, n_q - g0);
        const Eigen::MatrixX3d pts_b = grid_pts.middleRows(g0, nb);
        const Eigen::VectorXd w_b = grid_w.segment(g0, nb);

        // Shifted AO values (and gradients) of the live shells of the active
        // images at the batch points, and the pairs that survive the screen.
        std::vector<ScreenedImageAO> tab(n_cells);
        evaluate_screened_shifted_ao(basis, screen, pts_b, P_real_space.cells,
                                     active, need_grad, tab);
        profile.elapsed.finish(1);
        const std::vector<int> live =
            live_pair_indices(pairs, tab, screen, &p_max);
        const std::vector<int> live_v =
            live_pair_indices(pairs, tab, screen, nullptr);
        const auto density_plan = xc_torus_plan(torus, pairs, live, &tab, nbf);
        const auto potential_plan = xc_torus_plan(torus, pairs, live_v, &tab, nbf);
        profile.elapsed.finish(6);
        const auto view = [&](int i) -> XCTorusAOView {
            return {tab[i].chi, tab[i].dchi, tab[i].full ? nullptr : &tab[i].bfs};
        };

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
        if (!density_plan.empty()) {
            const auto f = xc_torus_fields(P_real_space, density_plan, *torus,
                                           view, nb, need_grad, is_mgga);
            rho = f.rho;
            if (need_grad) { gx = f.grad.col(0); gy = f.grad.col(1); gz = f.grad.col(2); }
            if (is_mgga) tau = f.tau;
        } else {
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
            for (std::size_t kk = 0; kk < live.size(); ++kk) {
                const int tid = omp_thread_index();
                const BraKetPair& pr = pairs[live[kk]];
                const ScreenedImageAO& img_a = tab[pr.a];
                const ScreenedImageAO& img_s = tab[pr.s];
                const Eigen::MatrixXd& chi_a = img_a.chi;     // bra χ(r − R_a)
                const Eigen::MatrixXd& chi_s = img_s.chi;     // ket χ(r − R_s)
                Eigen::MatrixXd P_buffer;
                const Eigen::MatrixXd& P = pair_density_block(
                    P_real_space.blocks[pr.gp], img_a, img_s, P_buffer);
                const Eigen::MatrixXd chi_aP = chi_a * P;     // (nb, n_live_s)
                // ρ(r) += Σ_{μν} χ_a(r)_μ P(s−a)_{μν} χ_s(r)_ν
                rho_t[tid].array() +=
                    (chi_aP.array() * chi_s.array()).rowwise().sum();

                if (need_grad) {
                    // ∂ρ/∂α = Σ_{μν} P_{μν}[∂χ_a,μ χ_s,ν + χ_a,μ ∂χ_s,ν]
                    const std::array<Eigen::MatrixXd, 3>& da = img_a.dchi;
                    const std::array<Eigen::MatrixXd, 3>& ds = img_s.dchi;
                    gx_t[tid].array() +=
                        (chi_aP.array() * ds[0].array()).rowwise().sum();
                    gy_t[tid].array() +=
                        (chi_aP.array() * ds[1].array()).rowwise().sum();
                    gz_t[tid].array() +=
                        (chi_aP.array() * ds[2].array()).rowwise().sum();
                    // Reuse each bra-gradient product for tau as well.
                    // Keep only one product alive per worker and preserve
                    // each field's pair/direction accumulation order.
                    const std::array<Eigen::VectorXd*, 3> gradients = {
                        &gx_t[tid], &gy_t[tid], &gz_t[tid]};
                    for (int d = 0; d < 3; ++d) {
                        const Eigen::MatrixXd gradient_density = da[d] * P;
                        gradients[d]->array() +=
                            (gradient_density.array() * chi_s.array())
                                .rowwise().sum();
                        if (is_mgga) {
                            tau_t[tid].array() +=
                                (gradient_density.array() * ds[d].array())
                                    .rowwise().sum();
                        }
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
        if (is_mgga && density_plan.empty()) {
            tau *= 0.5;   // match rks.cpp:124
        }

        n_grid += w_b.dot(rho);   // before the clamp: the density as summed

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

        profile.elapsed.finish(2);
        Eigen::VectorXd exc, v_rho, v_sigma, v_tau;
        if (is_mgga) {
            func.eval_unpolarised_mgga(rho, sigma, tau,
                                       exc, v_rho, v_sigma, v_tau);
        } else {
            func.eval_unpolarised(rho, sigma, exc, v_rho, v_sigma);
        }

        profile.elapsed.finish(3);
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

        if (!potential_plan.empty()) {
            Eigen::MatrixX3d qg;
            if (need_grad) { qg.resize(nb, 3); qg << w2vs_gx, w2vs_gy, w2vs_gz; }
            xc_torus_project(potential_plan, *torus, view, w_vrho, qg,
                             w_vtau, out.V_xc.blocks);
            profile.elapsed.finish(7);
            continue;
        }

        std::vector<std::vector<int>> live_by_gp(n_cells);
        for (int k : live_v) live_by_gp[pairs[k].gp].push_back(k);

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t gp = 0; gp < n_cells; ++gp) {
            if (live_by_gp[gp].empty()) continue;
            Eigen::MatrixXd Vg = Eigen::MatrixXd::Zero(nbf, nbf);
            for (int k : live_by_gp[gp]) {
                const BraKetPair& pr = pairs[k];
                const ScreenedImageAO& img_a = tab[pr.a];
                const ScreenedImageAO& img_s = tab[pr.s];
                const Eigen::MatrixXd& chi_a = img_a.chi;
                const Eigen::MatrixXd& chi_s = img_s.chi;
                // Complete images accumulate straight into Vg (the dense
                // arithmetic); screened ones into their live sub-block.
                const bool full = img_a.full && img_s.full;
                Eigen::MatrixXd sub;
                if (!full) {
                    sub.setZero(static_cast<Eigen::Index>(img_a.bfs.size()),
                                static_cast<Eigen::Index>(img_s.bfs.size()));
                }
                Eigen::MatrixXd& Vt = full ? Vg : sub;
                // LDA piece: χ_aᵀ · diag(w·v_ρ) · χ_s
                Vt.noalias() += chi_a.transpose() * w_vrho.asDiagonal() * chi_s;

                if (need_grad) {
                    const std::array<Eigen::MatrixXd, 3>& da = img_a.dchi;
                    const std::array<Eigen::MatrixXd, 3>& ds = img_s.dchi;
                    // Term 1: (∇ρ · ∇χ_a) χ_s
                    Eigen::MatrixXd A =
                        da[0].array().colwise() * w2vs_gx.array();
                    A.array() += da[1].array().colwise() * w2vs_gy.array();
                    A.array() += da[2].array().colwise() * w2vs_gz.array();
                    Vt.noalias() += A.transpose() * chi_s;
                    // Term 2: χ_a (∇ρ · ∇χ_s)
                    Eigen::MatrixXd B =
                        ds[0].array().colwise() * w2vs_gx.array();
                    B.array() += ds[1].array().colwise() * w2vs_gy.array();
                    B.array() += ds[2].array().colwise() * w2vs_gz.array();
                    Vt.noalias() += chi_a.transpose() * B;
                }

                if (is_mgga) {
                    const std::array<Eigen::MatrixXd, 3>& da = img_a.dchi;
                    const std::array<Eigen::MatrixXd, 3>& ds = img_s.dchi;
                    for (int d = 0; d < 3; ++d) {
                        Vt.noalias() +=
                            da[d].transpose() * w_vtau.asDiagonal() * ds[d];
                    }
                }
                if (!full) Vg(img_a.bfs, img_s.bfs) += sub;
            }
            out.V_xc.blocks[gp].noalias() += Vg;   // accumulate across batches
        }
        profile.elapsed.finish(4);
    }

    out.e_xc = e_xc;
    out.n_electrons_grid = n_grid;
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
static PeriodicUKSXCContribution build_xc_periodic_uks_impl(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_alpha,
    const LatticeMatrixSet& P_beta,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain,
    double ao_screening_tol, XCTorusContext* torus) {
    detail::NativePhaseCall<8> profile("periodic_xc.open_shell",
        {"setup", "ao_collocation", "density", "functional",
         "potential_projection", "external_provider_route", "pair_plan",
         "torus_potential_projection"},
        {{"nbasis", basis.nbasis()}, {"points", static_cast<std::uint64_t>(grid.points.rows())},
         {"cells", P_alpha.cells.size()}});
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
    out.periodic_density_bras = has_xc;
    double e_xc = 0.0;
    double n_grid = 0.0;

    // Output cell gp -> the bra/ket pairs that map to it (batch-independent).
    std::vector<std::vector<int>> pairs_by_gp(n_cells);
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        pairs_by_gp[pairs[k].gp].push_back(static_cast<int>(k));
    }

    profile.elapsed.finish(0);
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
            xc_value_batch_size(nbf, active.size(), true, true, torus));
        std::vector<int> all_pairs;
        if (torus) {
            all_pairs.resize(pairs.size());
            std::iota(all_pairs.begin(), all_pairs.end(), 0);
        }
        const auto external_plan = xc_torus_plan(torus, pairs, all_pairs, nullptr, nbf);
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
            const auto view = [&](int i) -> XCTorusAOView {
                return {chi_h[i], dchi_h[i], nullptr};
            };
            const auto fields = external_plan.empty()
                ? std::array<ExternalPeriodicFields, 2>{
                    build_external_periodic_fields_batch(P_alpha, pairs, chi_h, dchi_h, nb),
                    build_external_periodic_fields_batch(P_beta, pairs, chi_h, dchi_h, nb)}
                : xc_torus_fields_channels(
                    std::array<const LatticeMatrixSet*, 2>{&P_alpha, &P_beta},
                    external_plan, *torus, view, nb, true, true);
            const auto& fa = fields[0];
            const auto& fb = fields[1];
            n_grid += grid.weights.segment(g0, nb).dot(fa.rho + fb.rho);
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
            const auto view = [&](int i) -> XCTorusAOView {
                return {chi_h[i], dchi_h[i], nullptr};
            };
            if (!external_plan.empty()) {
                const Eigen::VectorXd qa = ev.v_rho_alpha.segment(g0, nb);
                const Eigen::VectorXd qb = ev.v_rho_beta.segment(g0, nb);
                const Eigen::MatrixX3d ga = ev.v_grad_alpha.middleRows(g0, nb);
                const Eigen::MatrixX3d gb = ev.v_grad_beta.middleRows(g0, nb);
                const Eigen::VectorXd ta = 0.5 * ev.v_tau_alpha.segment(g0, nb);
                const Eigen::VectorXd tb = 0.5 * ev.v_tau_beta.segment(g0, nb);
                xc_torus_project_channels(external_plan, *torus, view,
                    std::array<XCTorusProjectionChannel, 2>{{
                        {qa, ga, ta, out.V_alpha.blocks},
                        {qb, gb, tb, out.V_beta.blocks}}});
            } else {
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
        }
        profile.elapsed.finish(5);
        out.e_xc = ev.energy;
        out.n_electrons_grid = n_grid;
        return out;
    }

    // ---- Grid-batched XC quadrature (spin-polarised; see build_xc_periodic)
    // chi_h / dchi_h for the active cells dominate memory; libxc is pointwise,
    // so the rho -> functional -> V_σ(g) chain factorises over disjoint
    // grid-point batches, accumulating e_xc and V_alpha/V_beta(g). Peak AO
    // memory becomes O(n_active * batch * nbf), independent of grid size.
    // Per-batch math is identical to the whole-grid path, so energies match
    // to floating-point reassociation.
    // AO-reach screening as in build_xc_periodic (ImageShellScreen).
    const ImageShellScreen screen =
        make_image_shell_screen(basis, ao_screening_tol);
    Eigen::MatrixX3d q_points;
    Eigen::VectorXd q_weights;
    if (screen.enabled) nonzero_weight_points(grid, q_points, q_weights);
    const Eigen::MatrixX3d& grid_pts = screen.enabled ? q_points : grid.points;
    const Eigen::VectorXd& grid_w = screen.enabled ? q_weights : grid.weights;
    const Eigen::Index n_q = grid_pts.rows();
    Eigen::Index kXcGridBatch = static_cast<Eigen::Index>(
            xc_value_batch_size(nbf, active.size(), need_grad, true, torus, screen.enabled));
    if (screen.enabled) kXcGridBatch = std::min(kXcGridBatch, kScreenedXcBatch);
    const std::vector<double> p_max = block_abs_max(P_alpha, &P_beta);
    profile.elapsed.finish(0);
    for (Eigen::Index g0 = 0; g0 < n_q; g0 += kXcGridBatch) {
        const Eigen::Index nb = std::min(kXcGridBatch, n_q - g0);
        const Eigen::MatrixX3d pts_b = grid_pts.middleRows(g0, nb);
        const Eigen::VectorXd w_b = grid_w.segment(g0, nb);

        std::vector<ScreenedImageAO> tab(n_cells);
        evaluate_screened_shifted_ao(basis, screen, pts_b, P_alpha.cells,
                                     active, need_grad, tab);
        profile.elapsed.finish(1);
        const std::vector<int> live =
            live_pair_indices(pairs, tab, screen, &p_max);
        const std::vector<int> live_v =
            live_pair_indices(pairs, tab, screen, nullptr);
        const auto density_plan = xc_torus_plan(torus, pairs, live, &tab, nbf);
        const auto potential_plan = xc_torus_plan(torus, pairs, live_v, &tab, nbf);
        const auto view = [&](int i) -> XCTorusAOView {
            return {tab[i].chi, tab[i].dchi, tab[i].full ? nullptr : &tab[i].bfs};
        };

        profile.elapsed.finish(6);
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

        if (!density_plan.empty()) {
            const auto fields = xc_torus_fields_channels(
                std::array<const LatticeMatrixSet*, 2>{&P_alpha, &P_beta},
                density_plan, *torus, view, nb, need_grad, is_mgga);
            const auto& a = fields[0];
            const auto& b = fields[1];
            rho_a = a.rho; rho_b = b.rho;
            if (need_grad) {
                gax = a.grad.col(0); gay = a.grad.col(1); gaz = a.grad.col(2);
                gbx = b.grad.col(0); gby = b.grad.col(1); gbz = b.grad.col(2);
            }
            if (is_mgga) { tau_a = a.tau; tau_b = b.tau; }
        } else {
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
            for (std::size_t kk = 0; kk < live.size(); ++kk) {
                const int tid = omp_thread_index();
                const BraKetPair& pr = pairs[live[kk]];
                const ScreenedImageAO& img_a = tab[pr.a];
                const ScreenedImageAO& img_s = tab[pr.s];
                const Eigen::MatrixXd& chi_a = img_a.chi;     // bra χ(r − R_a)
                const Eigen::MatrixXd& chi_s = img_s.chi;     // ket χ(r − R_s)
                Eigen::MatrixXd PA_buffer, PB_buffer;
                const Eigen::MatrixXd& PA = pair_density_block(
                    P_alpha.blocks[pr.gp], img_a, img_s, PA_buffer);
                const Eigen::MatrixXd& PB = pair_density_block(
                    P_beta.blocks[pr.gp], img_a, img_s, PB_buffer);
                const Eigen::MatrixXd chi_Pa = chi_a * PA;
                const Eigen::MatrixXd chi_Pb = chi_a * PB;
                rho_a_t[tid].array() +=
                    (chi_Pa.array() * chi_s.array()).rowwise().sum();
                rho_b_t[tid].array() +=
                    (chi_Pb.array() * chi_s.array()).rowwise().sum();

                if (need_grad) {
                    const std::array<Eigen::MatrixXd, 3>& da = img_a.dchi;
                    const std::array<Eigen::MatrixXd, 3>& ds = img_s.dchi;
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
                    // Reuse the bra-gradient product for the same spin's
                    // kinetic density without retaining extra image tables.
                    const std::array<Eigen::VectorXd*, 3> gradients_a = {
                        &gax_t[tid], &gay_t[tid], &gaz_t[tid]};
                    const std::array<Eigen::VectorXd*, 3> gradients_b = {
                        &gbx_t[tid], &gby_t[tid], &gbz_t[tid]};
                    for (int d = 0; d < 3; ++d) {
                        {
                            const Eigen::MatrixXd gradient_density = da[d] * PA;
                            gradients_a[d]->array() +=
                                (gradient_density.array() * chi_s.array())
                                    .rowwise().sum();
                            if (is_mgga) {
                                tau_a_t[tid].array() +=
                                    (gradient_density.array() * ds[d].array())
                                        .rowwise().sum();
                            }
                        }
                        {
                            const Eigen::MatrixXd gradient_density = da[d] * PB;
                            gradients_b[d]->array() +=
                                (gradient_density.array() * chi_s.array())
                                    .rowwise().sum();
                            if (is_mgga) {
                                tau_b_t[tid].array() +=
                                    (gradient_density.array() * ds[d].array())
                                        .rowwise().sum();
                            }
                        }
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
        if (is_mgga && density_plan.empty()) {
            tau_a *= 0.5;   // match uks.cpp:149-150
            tau_b *= 0.5;
        }

        n_grid += w_b.dot(rho_a + rho_b);   // before the clamp

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

        profile.elapsed.finish(2);
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

        profile.elapsed.finish(3);
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

        if (!potential_plan.empty()) {
            Eigen::MatrixX3d qa, qb;
            if (need_grad) {
                qa.resize(nb, 3); qa << w_fa_x, w_fa_y, w_fa_z;
                qb.resize(nb, 3); qb << w_fb_x, w_fb_y, w_fb_z;
            }
            xc_torus_project_channels(potential_plan, *torus, view,
                std::array<XCTorusProjectionChannel, 2>{{
                    {w_vra, qa, w_vta, out.V_alpha.blocks},
                    {w_vrb, qb, w_vtb, out.V_beta.blocks}}});
            profile.elapsed.finish(7);
            continue;
        }

        std::vector<std::vector<int>> live_by_gp(n_cells);
        for (int k : live_v) live_by_gp[pairs[k].gp].push_back(k);

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t gp = 0; gp < n_cells; ++gp) {
            if (live_by_gp[gp].empty()) continue;
            Eigen::MatrixXd Va_g = Eigen::MatrixXd::Zero(nbf, nbf);
            Eigen::MatrixXd Vb_g = Eigen::MatrixXd::Zero(nbf, nbf);
            for (int k : live_by_gp[gp]) {
                const BraKetPair& pr = pairs[k];
                const ScreenedImageAO& img_a = tab[pr.a];
                const ScreenedImageAO& img_s = tab[pr.s];
                const Eigen::MatrixXd& chi_a = img_a.chi;
                const Eigen::MatrixXd& chi_s = img_s.chi;
                // Complete images accumulate straight into the cell blocks
                // (the dense arithmetic); screened ones into live sub-blocks.
                const bool full = img_a.full && img_s.full;
                Eigen::MatrixXd sub_a, sub_b;
                if (!full) {
                    const auto na = static_cast<Eigen::Index>(img_a.bfs.size());
                    const auto ns = static_cast<Eigen::Index>(img_s.bfs.size());
                    sub_a.setZero(na, ns);
                    sub_b.setZero(na, ns);
                }
                Eigen::MatrixXd& Va = full ? Va_g : sub_a;
                Eigen::MatrixXd& Vb = full ? Vb_g : sub_b;
                // LDA pieces.
                Va.noalias() += chi_a.transpose() * w_vra.asDiagonal() * chi_s;
                Vb.noalias() += chi_a.transpose() * w_vrb.asDiagonal() * chi_s;

                if (need_grad) {
                    const std::array<Eigen::MatrixXd, 3>& da = img_a.dchi;
                    const std::array<Eigen::MatrixXd, 3>& ds = img_s.dchi;
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
                    const std::array<Eigen::MatrixXd, 3>& da = img_a.dchi;
                    const std::array<Eigen::MatrixXd, 3>& ds = img_s.dchi;
                    for (int d = 0; d < 3; ++d) {
                        Va.noalias() +=
                            da[d].transpose() * w_vta.asDiagonal() * ds[d];
                        Vb.noalias() +=
                            da[d].transpose() * w_vtb.asDiagonal() * ds[d];
                    }
                }
                if (!full) {
                    Va_g(img_a.bfs, img_s.bfs) += sub_a;
                    Vb_g(img_a.bfs, img_s.bfs) += sub_b;
                }
            }
            out.V_alpha.blocks[gp].noalias() += Va_g;
            out.V_beta.blocks[gp].noalias() += Vb_g;
        }
        profile.elapsed.finish(4);
    }

    out.e_xc = e_xc;
    out.n_electrons_grid = n_grid;
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

PeriodicXCContribution build_xc_periodic(const BasisSet& basis,
    const PeriodicSystem& system, const Grid& grid, const Functional& func,
    const LatticeMatrixSet& p, const LatticeSumOptions& opts,
    PeriodicXCDensityDomain domain, double tol) {
    return build_xc_periodic_impl(basis, system, grid, func, p, opts, domain, tol, nullptr);
}

PeriodicUKSXCContribution build_xc_periodic_uks(const BasisSet& basis,
    const PeriodicSystem& system, const Grid& grid, const Functional& func,
    const LatticeMatrixSet& a, const LatticeMatrixSet& b, const LatticeSumOptions& opts,
    PeriodicXCDensityDomain domain, double tol) {
    return build_xc_periodic_uks_impl(basis, system, grid, func, a, b, opts, domain, tol, nullptr);
}

static std::vector<Eigen::MatrixXcd> fold_xc_projection(
    const LatticeMatrixSet& projection, const BlochKMesh& mesh) {
    std::vector<Eigen::Vector3d> origins;
    origins.reserve(projection.cells.size());
    for (const auto& cell : projection.cells) origins.push_back(cell.r_cart);
    return bloch_sum_multi_k(projection.blocks, origins, mesh.kpoints);
}

PeriodicKXCContribution build_xc_periodic_kpoints(const BasisSet& basis,
    const PeriodicSystem& system, const Grid& grid, const Functional& func,
    const LatticeMatrixSet& p, const BlochKMesh& mesh, const LatticeSumOptions& opts,
    PeriodicXCDensityDomain domain, double tol, bool torus_complete) {
    auto ctx = xc_torus_context(system, mesh, p);
    ctx.complete = torus_complete;
    // The internal projection may combine many physical V(g) blocks into a
    // representative image. Only the per-k matrices escape this wrapper.
    auto value = build_xc_periodic_impl(basis, system, grid, func, p, opts, domain, tol,
                                        ctx.compatible ? &ctx : nullptr);
    PeriodicKXCContribution out;
    out.V_xc = fold_xc_projection(value.V_xc, mesh);
    out.e_xc = value.e_xc;
    out.n_electrons_grid = value.n_electrons_grid;
    out.periodic_density_bras = value.periodic_density_bras;
    out.torus_density_compatible = ctx.compatible;
    out.used_torus_grouping = ctx.used;
    return out;
}

PeriodicKUKSXCContribution build_xc_periodic_uks_kpoints(const BasisSet& basis,
    const PeriodicSystem& system, const Grid& grid, const Functional& func,
    const LatticeMatrixSet& a, const LatticeMatrixSet& b, const BlochKMesh& mesh,
    const LatticeSumOptions& opts, PeriodicXCDensityDomain domain, double tol,
    bool torus_complete) {
    auto ctx = xc_torus_context(system, mesh, a, &b);
    ctx.complete = torus_complete;
    auto value = build_xc_periodic_uks_impl(basis, system, grid, func, a, b, opts, domain, tol,
                                            ctx.compatible ? &ctx : nullptr);
    PeriodicKUKSXCContribution out;
    out.V_alpha = fold_xc_projection(value.V_alpha, mesh);
    out.V_beta = fold_xc_projection(value.V_beta, mesh);
    out.e_xc = value.e_xc;
    out.n_electrons_grid = value.n_electrons_grid;
    out.periodic_density_bras = value.periodic_density_bras;
    out.torus_density_compatible = ctx.compatible;
    out.used_torus_grouping = ctx.used;
    return out;
}

}  // namespace vibeqc
