#include "vibeqc/uks.hpp"
#include "vibeqc/orbital_scf.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/diis.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/dynamic_damping.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_staged.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/kdiis.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/mom.hpp"
#include "vibeqc/linear_dependence.hpp"
#include "vibeqc/newton.hpp"
#include "vibeqc/quadratic_scf.hpp"
#include "vibeqc/scf_mixing.hpp"
#include "vibeqc/scf_convergence.hpp"
#include "vibeqc/scf_restart.hpp"
#include "vibeqc/soscf.hpp"
#include "vibeqc/trah.hpp"
#include "vibeqc/thread_pool.hpp"
#include "vibeqc/davidson.hpp"
#include "vibeqc/xc.hpp"
#include "vibeqc/xc_kernel.hpp"
#include "vibeqc/vv10.hpp"

namespace vibeqc {

namespace {

Eigen::MatrixXd build_spin_density(const Eigen::MatrixXd& C, int nocc) {
    if (nocc <= 0) return Eigen::MatrixXd::Zero(C.rows(), C.rows());
    return C.leftCols(nocc) * C.leftCols(nocc).transpose();
}

// A nominally spin-restricted singlet can acquire a few ulps of alpha/beta
// density drift because the two eigensolves are performed independently.
// Treat only that roundoff envelope as spin-degenerate so an expensive
// exchange builder is not called twice for the same physical density. A
// genuinely broken-symmetry singlet remains on the ordinary two-build path.
constexpr double kSpinDegenerateDensityTolerance = 1e-13;

bool spin_densities_are_degenerate(const Eigen::MatrixXd& D_alpha,
                                   const Eigen::MatrixXd& D_beta,
                                   int n_alpha,
                                   int n_beta) {
    return n_alpha == n_beta
        && D_alpha.rows() == D_beta.rows()
        && D_alpha.cols() == D_beta.cols()
        && (D_alpha - D_beta).cwiseAbs().maxCoeff()
               <= kSpinDegenerateDensityTolerance;
}

// <S^2> for UHF / UKS:
//   <S^2> = S(S+1) + n_β − Σ_{i∈occ_α, j∈occ_β} |<α_i | S | β_j>|²
double compute_s_squared(
    const Eigen::MatrixXd& Ca, int n_alpha,
    const Eigen::MatrixXd& Cb, int n_beta,
    const Eigen::MatrixXd& S) {
    const double dS = 0.5 * (n_alpha - n_beta);
    const double ideal = dS * (dS + 1);
    if (n_alpha == 0 || n_beta == 0) return ideal;
    const Eigen::MatrixXd Cao = Ca.leftCols(n_alpha);
    const Eigen::MatrixXd Cbo = Cb.leftCols(n_beta);
    const Eigen::MatrixXd M = Cao.transpose() * S * Cbo;
    const double sum_sq = (M.array() * M.array()).sum();
    return ideal + static_cast<double>(n_beta) - sum_sq;
}

struct UksXc {
    Eigen::MatrixXd V_alpha;
    Eigen::MatrixXd V_beta;
    double energy = 0.0;
    int batch_workers = 1;
};

using UksXcFunctionalPool = std::vector<std::unique_ptr<Functional>>;

UksXcFunctionalPool make_uks_xc_functional_pool(
    const std::string& name,
    Eigen::Index n_points) {
    const std::size_t n_batches = static_cast<std::size_t>(
        (n_points + kMolecularXcGridBatchSize - 1)
        / kMolecularXcGridBatchSize);
    const int workers = omp_workers_for(
        n_batches, kMolecularXcGridMaxWorkers);
    UksXcFunctionalPool pool;
    if (workers <= 1) return pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        pool.push_back(std::make_unique<Functional>(name, /*spin=*/2));
    }
    return pool;
}

Grid uks_xc_grid_slice(const Grid& grid,
                       Eigen::Index start,
                       Eigen::Index count) {
    Grid slice;
    slice.atomic_grid_profile = grid.atomic_grid_profile;
    slice.points = grid.points.middleRows(start, count);
    slice.weights = grid.weights.segment(start, count);
    if (grid.atomic_weights.size() == grid.weights.size()) {
        slice.atomic_weights = grid.atomic_weights.segment(start, count);
    }
    slice.atom_coords = grid.atom_coords;
    slice.atomic_numbers = grid.atomic_numbers;
    if (grid.atom_of_point.size()
        == static_cast<std::size_t>(grid.points.rows())) {
        const auto first = grid.atom_of_point.begin()
                         + static_cast<std::ptrdiff_t>(start);
        slice.atom_of_point.assign(first, first + count);
    }
    return slice;
}

AOValues evaluate_uks_xc_ao(const BasisSet& basis,
                            const Eigen::MatrixX3d& points,
                            bool need_gradient) {
    if (need_gradient) {
        return evaluate_ao_with_gradient(basis, points);
    }
    AOValues ao;
    ao.values = evaluate_ao(basis, points);
    return ao;
}

struct UksDensityFields {
    Eigen::VectorXd rho_a;
    Eigen::VectorXd rho_b;
    Eigen::VectorXd gax;
    Eigen::VectorXd gay;
    Eigen::VectorXd gaz;
    Eigen::VectorXd gbx;
    Eigen::VectorXd gby;
    Eigen::VectorXd gbz;
    Eigen::VectorXd sigma_aa;
    Eigen::VectorXd sigma_ab;
    Eigen::VectorXd sigma_bb;
    Eigen::VectorXd tau_a;
    Eigen::VectorXd tau_b;
};

UksDensityFields build_uks_density_fields(
    const Functional& func,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& Da,
    const Eigen::MatrixXd& Db,
    bool build_tau = true) {
    const XCKind kind = func.kind();
    const bool is_mgga = (kind == XCKind::MGGA);
    const bool need_grad = (kind == XCKind::GGA) || is_mgga
                        || func.needs_vv10();
    const auto n_pts = chi.rows();
    const Eigen::MatrixXd chi_Da = chi * Da;
    const Eigen::MatrixXd chi_Db = chi * Db;

    UksDensityFields fields;
    fields.rho_a.resize(n_pts);
    fields.rho_b.resize(n_pts);
    if (need_grad) {
        fields.gax.resize(n_pts);
        fields.gay.resize(n_pts);
        fields.gaz.resize(n_pts);
        fields.gbx.resize(n_pts);
        fields.gby.resize(n_pts);
        fields.gbz.resize(n_pts);
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            const auto cDa = chi_Da.row(g);
            const auto cDb = chi_Db.row(g);
            const auto c = chi.row(g);
            const auto dx = dchi[0].row(g);
            const auto dy = dchi[1].row(g);
            const auto dz = dchi[2].row(g);
            fields.rho_a(g) = cDa.dot(c);
            fields.rho_b(g) = cDb.dot(c);
            fields.gax(g) = 2.0 * cDa.dot(dx);
            fields.gay(g) = 2.0 * cDa.dot(dy);
            fields.gaz(g) = 2.0 * cDa.dot(dz);
            fields.gbx(g) = 2.0 * cDb.dot(dx);
            fields.gby(g) = 2.0 * cDb.dot(dy);
            fields.gbz(g) = 2.0 * cDb.dot(dz);
        }
        fields.sigma_aa = fields.gax.array().square()
                        + fields.gay.array().square()
                        + fields.gaz.array().square();
        fields.sigma_ab = fields.gax.array() * fields.gbx.array()
                        + fields.gay.array() * fields.gby.array()
                        + fields.gaz.array() * fields.gbz.array();
        fields.sigma_bb = fields.gbx.array().square()
                        + fields.gby.array().square()
                        + fields.gbz.array().square();
    } else {
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            const auto c = chi.row(g);
            fields.rho_a(g) = chi_Da.row(g).dot(c);
            fields.rho_b(g) = chi_Db.row(g).dot(c);
        }
    }

    if (is_mgga && build_tau) {
        fields.tau_a = Eigen::VectorXd::Zero(n_pts);
        fields.tau_b = Eigen::VectorXd::Zero(n_pts);
        std::array<Eigen::MatrixXd, 3> dchi_Da;
        std::array<Eigen::MatrixXd, 3> dchi_Db;
        for (int c = 0; c < 3; ++c) {
            dchi_Da[c] = dchi[c] * Da;
            dchi_Db[c] = dchi[c] * Db;
        }
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            double ta = 0.0;
            double tb = 0.0;
            for (int c = 0; c < 3; ++c) {
                ta += dchi_Da[c].row(g).dot(dchi[c].row(g));
                tb += dchi_Db[c].row(g).dot(dchi[c].row(g));
            }
            fields.tau_a(g) = 0.5 * ta;
            fields.tau_b(g) = 0.5 * tb;
        }
    }
    return fields;
}

UksXc build_external_xc_uks(
    const Functional& func,
    const BasisSet& basis,
    const Grid& grid,
    const Eigen::MatrixXd& Da,
    const Eigen::MatrixXd& Db) {
    const Eigen::Index n_pts = grid.points.rows();
    const Eigen::Index n_bf = static_cast<Eigen::Index>(basis.nbasis());
    UksXc out{
        Eigen::MatrixXd::Zero(n_bf, n_bf),
        Eigen::MatrixXd::Zero(n_bf, n_bf),
        0.0,
        1,
    };
    if (n_pts == 0) return out;
    if (grid.atomic_weights.size() != n_pts
        || static_cast<Eigen::Index>(grid.atom_of_point.size()) != n_pts
        || grid.atom_coords.cols() != 3) {
        throw std::invalid_argument(
            "external XC requires atom-major grid ownership, raw atomic "
            "weights, and atom coordinates");
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
    input.rho_alpha.resize(n_pts);
    input.rho_beta.resize(n_pts);
    input.grad_alpha.resize(n_pts, 3);
    input.grad_beta.resize(n_pts, 3);
    input.tau_alpha.resize(n_pts);
    input.tau_beta.resize(n_pts);

    for (Eigen::Index g0 = 0; g0 < n_pts;
         g0 += kMolecularXcGridBatchSize) {
        const Eigen::Index count = std::min<Eigen::Index>(
            kMolecularXcGridBatchSize, n_pts - g0);
        const Grid slice = uks_xc_grid_slice(grid, g0, count);
        AOValues ao = evaluate_uks_xc_ao(basis, slice.points, true);
        const UksDensityFields fields = build_uks_density_fields(
            func, ao.values, ao.gradients, Da, Db, /*build_tau=*/true);
        input.rho_alpha.segment(g0, count) = fields.rho_a;
        input.rho_beta.segment(g0, count) = fields.rho_b;
        input.grad_alpha.col(0).segment(g0, count) = fields.gax;
        input.grad_alpha.col(1).segment(g0, count) = fields.gay;
        input.grad_alpha.col(2).segment(g0, count) = fields.gaz;
        input.grad_beta.col(0).segment(g0, count) = fields.gbx;
        input.grad_beta.col(1).segment(g0, count) = fields.gby;
        input.grad_beta.col(2).segment(g0, count) = fields.gbz;
        input.tau_alpha.segment(g0, count) = fields.tau_a;
        input.tau_beta.segment(g0, count) = fields.tau_b;
    }

    const ExternalXCEvaluation ev = func.eval_external(input);
    out.energy = ev.energy;
    for (Eigen::Index g0 = 0; g0 < n_pts;
         g0 += kMolecularXcGridBatchSize) {
        const Eigen::Index count = std::min<Eigen::Index>(
            kMolecularXcGridBatchSize, n_pts - g0);
        const Eigen::MatrixX3d points = grid.points.middleRows(g0, count);
        AOValues ao = evaluate_uks_xc_ao(basis, points, true);
        const Eigen::VectorXd qra = ev.v_rho_alpha.segment(g0, count);
        const Eigen::VectorXd qrb = ev.v_rho_beta.segment(g0, count);
        out.V_alpha.noalias() +=
            ao.values.transpose() * qra.asDiagonal() * ao.values;
        out.V_beta.noalias() +=
            ao.values.transpose() * qrb.asDiagonal() * ao.values;
        const Eigen::VectorXd qta =
            0.5 * ev.v_tau_alpha.segment(g0, count);
        const Eigen::VectorXd qtb =
            0.5 * ev.v_tau_beta.segment(g0, count);
        for (int c = 0; c < 3; ++c) {
            const Eigen::VectorXd qga =
                ev.v_grad_alpha.col(c).segment(g0, count);
            const Eigen::VectorXd qgb =
                ev.v_grad_beta.col(c).segment(g0, count);
            const Eigen::MatrixXd Ga =
                ao.gradients[c].transpose() * qga.asDiagonal() * ao.values;
            const Eigen::MatrixXd Gb =
                ao.gradients[c].transpose() * qgb.asDiagonal() * ao.values;
            out.V_alpha.noalias() += Ga + Ga.transpose();
            out.V_beta.noalias() += Gb + Gb.transpose();
            out.V_alpha.noalias() += ao.gradients[c].transpose()
                                   * qta.asDiagonal() * ao.gradients[c];
            out.V_beta.noalias() += ao.gradients[c].transpose()
                                  * qtb.asDiagonal() * ao.gradients[c];
        }
    }
    return out;
}

// Build V_xc,α / V_xc,β matrices and E_xc for UKS from grid data.
//
// ρ_σ(g)   = Σ_{μν} (D_σ)_{μν} χ_μ(g) χ_ν(g)
// ∇ρ_σ(g) = 2 Σ_{μν} (D_σ)_{μν} χ_μ(g) ∇χ_ν(g)
// σ_αα = |∇ρ_α|²    σ_αβ = ∇ρ_α·∇ρ_β    σ_ββ = |∇ρ_β|²
// τ_σ(g) = ½ Σ_{μν} (D_σ)_{μν} Σ_c ∂_c χ_μ(g) ∂_c χ_ν(g)   (MGGA only)
//
// The "flow vector" for spin σ at each grid point is:
//   f_σ(g) = 2 v_σ_σσ(g) ∇ρ_σ(g) + v_σ_αβ(g) ∇ρ_{¬σ}(g)
// With F_σ(g, μ) = f_σ(g) · ∇χ_μ(g), the XC matrix elements are:
//   V_xc,σ_{μν} = Σ_g w_g v_ρ_σ(g) χ_μ(g) χ_ν(g)
//               + Σ_g w_g [F_σ(g,μ) χ_ν(g) + χ_μ(g) F_σ(g,ν)]
//               + ½ Σ_g w_g v_τ_σ(g) Σ_c ∂_c χ_μ(g) ∂_c χ_ν(g)   (MGGA)
UksXc build_uks_xc_slice(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& Da,
    const Eigen::MatrixXd& Db,
    const Eigen::VectorXd* vv10_v_rho = nullptr,
    const Eigen::VectorXd* vv10_v_sigma = nullptr) {
    const XCKind kind = func.kind();
    const bool is_gga  = (kind == XCKind::GGA);
    const bool is_mgga = (kind == XCKind::MGGA);
    const bool need_grad = is_gga || is_mgga || func.needs_vv10();
    const auto n_pts = chi.rows();
    UksDensityFields fields =
        build_uks_density_fields(func, chi, dchi, Da, Db);

    Eigen::VectorXd exc, v_rho_a, v_rho_b;
    Eigen::VectorXd v_sig_aa, v_sig_ab, v_sig_bb;
    Eigen::VectorXd v_tau_a, v_tau_b;
    if (is_mgga) {
        func.eval_polarised_mgga(
                                 fields.rho_a, fields.rho_b,
                                 fields.sigma_aa, fields.sigma_ab,
                                 fields.sigma_bb,
                                 fields.tau_a, fields.tau_b,
                                 exc, v_rho_a, v_rho_b,
                                 v_sig_aa, v_sig_ab, v_sig_bb,
                                 v_tau_a, v_tau_b);
    } else {
        func.eval_polarised(
                            fields.rho_a, fields.rho_b,
                            fields.sigma_aa, fields.sigma_ab,
                            fields.sigma_bb,
                            exc, v_rho_a, v_rho_b,
                            v_sig_aa, v_sig_ab, v_sig_bb);
    }

    double E_xc = grid.weights.dot(exc);

    // The wrapper evaluates VV10's full-grid double sum once.  Apply its
    // matching potential slice to both spin densities here, with the total-
    // sigma chain-rule factors unchanged from the dense implementation.
    // When VV10 is projected from a separate coarse grid (vv10_grid_factor
    // > 1), the caller passes nullptr here and this path skips VV10.
    if (func.needs_vv10() && vv10_v_rho != nullptr && vv10_v_sigma != nullptr) {
        if (vv10_v_rho->size() != n_pts || vv10_v_sigma->size() != n_pts) {
            throw std::invalid_argument(
                "build_uks_xc_slice: VV10 potential slice has the wrong "
                "grid length");
        }
        v_rho_a += *vv10_v_rho;
        v_rho_b += *vv10_v_rho;
        v_sig_aa += *vv10_v_sigma;
        v_sig_ab += 2.0 * *vv10_v_sigma;
        v_sig_bb += *vv10_v_sigma;
    }

    // LDA V pieces.
    Eigen::VectorXd w_vra = grid.weights.array() * v_rho_a.array();
    Eigen::VectorXd w_vrb = grid.weights.array() * v_rho_b.array();
    Eigen::MatrixXd V_a = chi.transpose() * w_vra.asDiagonal() * chi;
    Eigen::MatrixXd V_b = chi.transpose() * w_vrb.asDiagonal() * chi;

    if (need_grad) {
        // Flow vectors f_σ(g).
        Eigen::VectorXd fa_x =
            2.0 * v_sig_aa.array() * fields.gax.array()
          +       v_sig_ab.array() * fields.gbx.array();
        Eigen::VectorXd fa_y =
            2.0 * v_sig_aa.array() * fields.gay.array()
          +       v_sig_ab.array() * fields.gby.array();
        Eigen::VectorXd fa_z =
            2.0 * v_sig_aa.array() * fields.gaz.array()
          +       v_sig_ab.array() * fields.gbz.array();
        Eigen::VectorXd fb_x =
            2.0 * v_sig_bb.array() * fields.gbx.array()
          +       v_sig_ab.array() * fields.gax.array();
        Eigen::VectorXd fb_y =
            2.0 * v_sig_bb.array() * fields.gby.array()
          +       v_sig_ab.array() * fields.gay.array();
        Eigen::VectorXd fb_z =
            2.0 * v_sig_bb.array() * fields.gbz.array()
          +       v_sig_ab.array() * fields.gaz.array();

        // Build Fa / Fb in a single OMP-parallel row pass — the
        // pre-fix expression ``f_x.asDiagonal()*dchi[0] + ...``
        // materialised three (n_pts × n_bf) intermediates per spin
        // before summing. Fuse to one loop, two intermediates total.
        const Eigen::Index n_bf = chi.cols();
        Eigen::MatrixXd Fa(n_pts, n_bf);
        Eigen::MatrixXd Fb(n_pts, n_bf);
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            const auto dx = dchi[0].row(g);
            const auto dy = dchi[1].row(g);
            const auto dz = dchi[2].row(g);
            Fa.row(g) = fa_x(g) * dx + fa_y(g) * dy + fa_z(g) * dz;
            Fb.row(g) = fb_x(g) * dx + fb_y(g) * dy + fb_z(g) * dz;
        }

        Eigen::MatrixXd Fa_w = Fa.transpose() * grid.weights.asDiagonal() * chi;
        Eigen::MatrixXd Fb_w = Fb.transpose() * grid.weights.asDiagonal() * chi;
        V_a += Fa_w + Fa_w.transpose();
        V_b += Fb_w + Fb_w.transpose();
    }

    if (is_mgga) {
        // Per-spin τ Fock contribution:
        //   V^τ_σ_{μν} = ½ Σ_g w_g v_τ_σ(g) Σ_c ∂_c χ_μ(g) ∂_c χ_ν(g)
        const Eigen::VectorXd w_vta = 0.5 * grid.weights.array() * v_tau_a.array();
        const Eigen::VectorXd w_vtb = 0.5 * grid.weights.array() * v_tau_b.array();
        for (int c = 0; c < 3; ++c) {
            V_a += dchi[c].transpose() * w_vta.asDiagonal() * dchi[c];
            V_b += dchi[c].transpose() * w_vtb.asDiagonal() * dchi[c];
        }
    }
    return {V_a, V_b, E_xc, 1};
}

// Spin-polarised counterpart of the bounded RKS quadrature.  All semilocal
// quantities are pointwise and accumulated over contiguous slices of the
// existing Becke grid.  VV10 retains its full cross-grid coupling through a
// first density pass and one global kernel call, followed by batched AO
// projection of the resulting potentials.
//
// When ``vv10_grid`` is provided (non-null), VV10 is evaluated on that
// coarser grid and projected independently — the O(N^2) double sum runs on
// ~9x fewer points (factor-3 coarsening), making VV10-paired functionals
// production-usable.  When null, VV10 uses the full XC grid (legacy).
UksXc build_uks_xc(const Functional& func,
                    const BasisSet& basis,
                    const Grid& grid,
                    const Eigen::MatrixXd& Da,
                    const Eigen::MatrixXd& Db,
                    const Grid* vv10_grid,
                    const UksXcFunctionalPool& worker_functionals) {
    const Eigen::Index n_pts = grid.points.rows();
    const Eigen::Index n_bf = static_cast<Eigen::Index>(basis.nbasis());
    UksXc total{
        Eigen::MatrixXd::Zero(n_bf, n_bf),
        Eigen::MatrixXd::Zero(n_bf, n_bf),
        0.0,
        1,
    };
    if (n_pts == 0) return total;
    if (func.is_external()) {
        return build_external_xc_uks(func, basis, grid, Da, Db);
    }

    const bool need_gradient = func.kind() != XCKind::LDA
                            || func.needs_vv10();
    const bool vv10_on_coarse = (vv10_grid != nullptr) && func.needs_vv10();
    VV10Result vv10;

    // ---- VV10 nonlocal correlation (evaluated once per SCF iteration) ----
    if (func.needs_vv10()) {
        const Grid& vv10_g = vv10_on_coarse ? *vv10_grid : grid;
        const Eigen::Index vv10_n = vv10_g.points.rows();
        if (vv10_n > 0) {
            // Gather total density and sigma on the VV10 grid.
            Eigen::VectorXd rho_total(vv10_n);
            Eigen::VectorXd sigma_total(vv10_n);
            for (Eigen::Index g0 = 0; g0 < vv10_n;
                 g0 += kMolecularXcGridBatchSize) {
                const Eigen::Index count = std::min<Eigen::Index>(
                    kMolecularXcGridBatchSize, vv10_n - g0);
                const Grid slice = uks_xc_grid_slice(vv10_g, g0, count);
                AOValues ao = evaluate_uks_xc_ao(
                    basis, slice.points, true);
                const UksDensityFields fields = build_uks_density_fields(
                    func, ao.values, ao.gradients, Da, Db,
                    /*build_tau=*/false);
                rho_total.segment(g0, count) =
                    fields.rho_a + fields.rho_b;
                sigma_total.segment(g0, count) =
                    fields.sigma_aa.array()
                  + 2.0 * fields.sigma_ab.array()
                  + fields.sigma_bb.array();
            }
            vv10 = compute_vv10(vv10_g.points, vv10_g.weights,
                                rho_total, sigma_total,
                                func.vv10_b(), func.vv10_C());
            total.energy += vv10.energy;

            // When VV10 is on a separate coarse grid, project its V_xc
            // contribution independently.
            if (vv10_on_coarse) {
                for (Eigen::Index g0 = 0; g0 < vv10_n;
                     g0 += kMolecularXcGridBatchSize) {
                    const Eigen::Index count = std::min<Eigen::Index>(
                        kMolecularXcGridBatchSize, vv10_n - g0);
                    const Grid slice = uks_xc_grid_slice(
                        vv10_g, g0, count);
                    AOValues ao = evaluate_uks_xc_ao(
                        basis, slice.points, true);
                    const UksDensityFields fields =
                        build_uks_density_fields(
                            func, ao.values, ao.gradients, Da, Db,
                            /*build_tau=*/false);

                    const Eigen::VectorXd vr =
                        vv10.v_rho.segment(g0, count);
                    const Eigen::VectorXd vs =
                        vv10.v_sigma.segment(g0, count);

                    // rho term: V_σ += χᵀ · diag(w · v_rho) · χ
                    // VV10 acts on total density → same v_rho for both
                    // spins.
                    Eigen::VectorXd w_vrho =
                        slice.weights.array() * vr.array();
                    Eigen::MatrixXd V_rho =
                        ao.values.transpose()
                        * w_vrho.asDiagonal() * ao.values;
                    total.V_alpha.noalias() += V_rho;
                    total.V_beta.noalias() += V_rho;

                    // sigma / GGA term for the VV10 kernel.
                    // VV10 depends on the total σ = |∇ρ_tot|², so the
                    // effective flow vector for *both* spins is
                    //   2 · v_sigma · (∇ρ_α + ∇ρ_β).
                    // Build F from the total gradient once and add
                    // the resulting V piece to both V_α and V_β.
                    {
                        Eigen::MatrixXd F(count, n_bf);
                        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
                        for (Eigen::Index g = 0; g < count; ++g) {
                            const double gx = fields.gax(g) + fields.gbx(g);
                            const double gy = fields.gay(g) + fields.gby(g);
                            const double gz = fields.gaz(g) + fields.gbz(g);
                            F.row(g) =
                                gx * ao.gradients[0].row(g)
                              + gy * ao.gradients[1].row(g)
                              + gz * ao.gradients[2].row(g);
                        }
                        Eigen::VectorXd u =
                            2.0 * slice.weights.array() * vs.array();
                        Eigen::MatrixXd Fu =
                            F.transpose() * u.asDiagonal() * ao.values;
                        Eigen::MatrixXd V_sigma = Fu + Fu.transpose();
                        total.V_alpha.noalias() += V_sigma;
                        total.V_beta.noalias() += V_sigma;
                    }
                }
                }
            }
        }

    // ---- Semilocal XC on the full grid (ordered parallel batches) ----
    // Each worker owns its libxc Functional. Batch GEMM dimensions and the
    // linked-BLAS route stay unchanged; results are reduced in their original
    // grid order, one memory-bounded wave at a time.
    const auto evaluate_batch = [&](const Functional& batch_func,
                                    Eigen::Index g0) {
        const Eigen::Index count = std::min<Eigen::Index>(
            kMolecularXcGridBatchSize, n_pts - g0);
        const Grid slice = uks_xc_grid_slice(grid, g0, count);
        AOValues ao = evaluate_uks_xc_ao(
            basis, slice.points, need_gradient);
        Eigen::VectorXd vv10_v_rho;
        Eigen::VectorXd vv10_v_sigma;
        const Eigen::VectorXd* vv10_v_rho_ptr = nullptr;
        const Eigen::VectorXd* vv10_v_sigma_ptr = nullptr;
        if (batch_func.needs_vv10() && !vv10_on_coarse) {
            vv10_v_rho = vv10.v_rho.segment(g0, count);
            vv10_v_sigma = vv10.v_sigma.segment(g0, count);
            vv10_v_rho_ptr = &vv10_v_rho;
            vv10_v_sigma_ptr = &vv10_v_sigma;
        }
        return build_uks_xc_slice(
            batch_func, slice, ao.values, ao.gradients, Da, Db,
            vv10_v_rho_ptr, vv10_v_sigma_ptr);
    };

    const std::size_t n_batches = static_cast<std::size_t>(
        (n_pts + kMolecularXcGridBatchSize - 1)
        / kMolecularXcGridBatchSize);
    const int workers = omp_workers_for(
        n_batches, kMolecularXcGridMaxWorkers);
    total.batch_workers = 1;

    if (workers == 1) {
        for (std::size_t batch = 0; batch < n_batches; ++batch) {
            const Eigen::Index g0 = static_cast<Eigen::Index>(batch)
                                  * kMolecularXcGridBatchSize;
            UksXc part = evaluate_batch(func, g0);
            total.V_alpha.noalias() += part.V_alpha;
            total.V_beta.noalias() += part.V_beta;
            total.energy += part.energy;
        }
        return total;
    }

    if (worker_functionals.size() < static_cast<std::size_t>(workers)) {
        throw std::logic_error(
            "build_uks_xc: worker Functional pool is smaller than the "
            "resolved XC batch team");
    }
    for (std::size_t wave = 0; wave < n_batches;
         wave += static_cast<std::size_t>(workers)) {
        const int wave_size = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(workers), n_batches - wave));
        std::vector<UksXc> parts(static_cast<std::size_t>(wave_size));
        std::vector<std::exception_ptr> worker_errors(
            static_cast<std::size_t>(wave_size));
        int actual_workers = 1;
        #pragma omp parallel num_threads(wave_size)
        {
            #pragma omp single
            {
                actual_workers = omp_team_threads();
            }
            #pragma omp for schedule(static, 1)
            for (int local_batch = 0; local_batch < wave_size; ++local_batch) {
                try {
                    const Eigen::Index g0 = static_cast<Eigen::Index>(
                        wave + static_cast<std::size_t>(local_batch))
                        * kMolecularXcGridBatchSize;
                    const auto worker = static_cast<std::size_t>(
                        omp_thread_index());
                    parts[static_cast<std::size_t>(local_batch)] =
                        evaluate_batch(*worker_functionals[worker], g0);
                } catch (...) {
                    worker_errors[static_cast<std::size_t>(local_batch)] =
                        std::current_exception();
                }
            }
        }
        total.batch_workers = std::max(
            total.batch_workers, actual_workers);
        for (int local_batch = 0; local_batch < wave_size; ++local_batch) {
            const auto slot = static_cast<std::size_t>(local_batch);
            if (worker_errors[slot]) {
                std::rethrow_exception(worker_errors[slot]);
            }
            const auto& part = parts[slot];
            total.V_alpha.noalias() += part.V_alpha;
            total.V_beta.noalias() += part.V_beta;
            total.energy += part.energy;
        }
    }
    return total;
}

}  // namespace

UKSXCPotential evaluate_uks_xc_potential(
    const Functional& functional,
    const BasisSet& basis,
    const Grid& grid,
    const Eigen::MatrixXd& density_alpha,
    const Eigen::MatrixXd& density_beta) {
    if (density_alpha.rows() != static_cast<Eigen::Index>(basis.nbasis())
        || density_alpha.cols() != density_alpha.rows()
        || density_beta.rows() != density_alpha.rows()
        || density_beta.cols() != density_alpha.cols()) {
        throw std::invalid_argument(
            "evaluate_uks_xc_potential: density matrices must both be "
            "square with one row per basis function");
    }
    const UksXc xc = build_uks_xc(
        functional, basis, grid, density_alpha, density_beta,
        /*vv10_grid=*/nullptr, /*worker_functionals=*/{});
    return UKSXCPotential{xc.V_alpha, xc.V_beta, xc.energy};
}

OrbitalXCFunction make_orbital_uks_xc(const BasisSet& basis, const Grid& grid, const std::string& name) {
    auto f = std::make_shared<Functional>(name, 2);
    validate_orbital_xc(*f);
    auto ao = std::make_shared<AOValues>(evaluate_ao_with_gradient(basis, grid.points));
    auto pool = std::make_shared<UksXcFunctionalPool>(
        make_uks_xc_functional_pool(name, grid.points.rows()));
    return [basis, grid, f, ao, pool](const Eigen::MatrixXd& da, const Eigen::MatrixXd& db, bool response) {
        OrbitalXC out;
        const auto value = build_uks_xc(*f, basis, grid, da, db, nullptr, *pool);
        out.energy = value.energy; out.alpha = value.V_alpha; out.beta = value.V_beta;
        if (response) out.unrestricted_kernel = make_polarised_xc_kernel_builder(
            *f, grid, ao->values, ao->gradients, da, db);
        return out;
    };
}

UKSResult run_uks(const Molecule& mol,
                  const BasisSet& basis,
                  const UKSOptions& opts) {
    validate_orbital_optimizer(opts);
    if (opts.orbital_optimizer == "opentrustregion"
        && (opts.spinlock_mode != SpinlockMode::OFF || !opts.atomic_spins.empty()))
        throw std::invalid_argument("OpenTrustRegion does not support spin schedules, MOM holds or targeted atomic spin states; select native");
    validate_initial_guess(opts.initial_guess);
    validate_guess_ecp(opts.initial_guess, molecular_guess_ecp_context(opts), &mol);
    validate_scf_max_iter(opts.max_iter, "run_uks");
    validate_atomic_spin_selection(
        &mol, GuessEngine::resolve_auto_for_molecule(mol, opts.initial_guess, false, true),
        opts.atomic_spins);
    const auto ecp_input = validate_molecular_ecp_dispatch(
        opts.ecp_centers, opts.ecp_library, opts.ecp_primitive_blocks,
        opts.ecp_primitive_centers, opts.ecp_effective_charges,
        opts.ecp_total_ncore, "run_uks");
    // ---- Integrals ----
    const Eigen::MatrixXd S = compute_overlap(basis);
    const Eigen::MatrixXd T = compute_kinetic(basis);
    const auto ecp_h = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
        ? compute_ecp_one_electron_from_primitives(
              basis, mol, opts.ecp_primitive_centers,
              opts.ecp_primitive_blocks, opts.ecp_effective_charges,
              opts.ecp_total_ncore)
        : compute_ecp_one_electron(
              basis, mol, opts.ecp_centers, opts.ecp_library);
    const Eigen::MatrixXd Hcore = T + ecp_h.V + ecp_h.V_ecp;
    const double E_nuc = ecp_h.E_nuc;

    // Effective electron count: subtract ECP-replaced core electrons.
    const int n_elec = mol.n_electrons() - ecp_h.total_ncore;
    if (n_elec < 0) {
        throw std::invalid_argument(
            "run_uks: ECP cores remove more electrons ("
            + std::to_string(ecp_h.total_ncore)
            + ") than the molecule has ("
            + std::to_string(mol.n_electrons())
            + "). Molecule.n_electrons() must be the full physical electron "
              "count; the SCF subtracts the ECP core itself.");
    }
    const auto with_ecp_provenance =
        [ncore = ecp_h.total_ncore,
         operator_applied = ecp_input != MolecularECPInput::NONE,
         xml_centers = ecp_input == MolecularECPInput::XML_LIBRARY
             ? opts.ecp_centers : std::vector<ECPCenter>{},
         xml_library = ecp_input == MolecularECPInput::XML_LIBRARY
             ? (opts.ecp_library.empty() ? std::string("ecp10mdf")
                                         : opts.ecp_library)
             : std::string{},
         primitive_blocks = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
             ? opts.ecp_primitive_blocks : std::vector<ECPPrimitiveBlock>{},
         primitive_centers = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
             ? opts.ecp_primitive_centers
             : std::vector<std::array<double, 3>>{},
         effective_charges = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
             ? opts.ecp_effective_charges : std::vector<double>{}](
            UKSResult result) {
            result.ecp_operator_applied = operator_applied;
            result.ecp_provenance_verified = true;
            result.ecp_xml_centers = xml_centers;
            result.ecp_xml_library = xml_library;
            result.ecp_primitive_blocks = primitive_blocks;
            result.ecp_primitive_centers = primitive_centers;
            result.ecp_effective_charges = effective_charges;
            result.ecp_total_ncore = ncore;
            return result;
        };
    const int mult = mol.multiplicity();
    const int n_alpha = (n_elec + mult - 1) / 2;
    const int n_beta  = (n_elec - mult + 1) / 2;

    if (n_alpha + n_beta != n_elec || n_alpha - n_beta != mult - 1) {
        throw std::invalid_argument(
            "UKS: multiplicity and electron count are inconsistent");
    }
    if (n_beta < 0) {
        throw std::invalid_argument("UKS: negative beta electron count");
    }
    validate_fraction_01("UKS: damping", opts.damping);
    validate_fraction_01("UKS: fock_mixing", opts.fock_mixing);

    // ---- XC grid + (optional) COSX grid ----
    Grid grid = build_grid(mol, opts.grid);

    // VV10 nonlocal correlation grid (when the functional needs it).
    // Same logic as run_rks — builds a coarser grid for the O(N^2)
    // VV10 double sum.  See rks.cpp for the detailed rationale.
    std::unique_ptr<Grid> vv10_grid_owner;
    const Functional functional_uks(opts.functional, 2);
    if (functional_uks.needs_vv10() && opts.grid.vv10_grid_factor > 0.0
        && opts.grid.vv10_grid_factor != 1.0) {
        GridOptions vv10_opts = opts.grid;
        const double f = opts.grid.vv10_grid_factor;
        vv10_opts.n_radial = std::max(5, static_cast<int>(
            std::lround(opts.grid.n_radial / f)));
        if (vv10_opts.angular == AngularScheme::Lebedev) {
            int order = vv10_opts.lebedev_order;
            if (order > 11) {
                int steps = std::max(1, static_cast<int>(std::lround(
                    static_cast<double>(order - 11) / f)));
                order = std::max(11, order - steps * 6);
                if (order <= 14) order = 11;
                else if (order <= 20) order = 17;
                else if (order <= 26) order = 23;
                else if (order <= 32) order = 29;
                else if (order <= 38) order = 35;
            }
            vv10_opts.lebedev_order = order;
        } else {
            vv10_opts.n_theta = std::max(3, static_cast<int>(
                std::lround(opts.grid.n_theta / f)));
            vv10_opts.n_phi = std::max(4, static_cast<int>(
                std::lround(opts.grid.n_phi / f)));
        }
        vv10_grid_owner = std::make_unique<Grid>(build_grid(mol, vv10_opts));
    }
    const Grid* vv10_grid = vv10_grid_owner.get();
    // COSX supplies only exact exchange. For a pure functional alpha_HF is
    // zero, so building or staging a COSX grid cannot change the KS surface.
    // Route that documented no-op through the ordinarily configured JK
    // builder instead: otherwise the staged wrapper's one-iteration final-grid
    // recompute would overwrite the supported UKS stability verdict and its
    // #205 timing metadata. COSX also requires the density-fit/auxiliary-basis
    // route; a bare ``cosx=true`` on a direct job is inactive.
    const bool cosx_exchange_active =
        opts.density_fit && opts.cosx
        && functional_uks.hf_exchange_fraction() != 0.0;
    Grid cosx_grid_built;
    CosxVariant cosx_variant_resolved = CosxVariant::FITTED;
    std::vector<GridOptions> cosx_stages;  // COSX grid progression (GridX)
    bool cosx_multistage = false;
    const bool spinlock_active =
        (opts.spinlock_mode == SpinlockMode::SPIN_SCHEDULE
         && opts.spinlock_iterations > 0 && opts.max_iter > 0);
    if (cosx_exchange_active) {
        const int cosx_card = cosx_basis_cardinality_from_name(basis.name());
        cosx_variant_resolved = resolve_cosx_variant(
            opts.cosx_variant, opts.thresh_cosx, cosx_card,
            opts.cosx_grid_level);
        if (!cosx_use_gridx(opts.cosx_grid_level, opts.conv_tol_grad)) {
            // Legacy single COSX grid: explicit opt-out or the auto fallback
            // for a conv_tol_grad tighter than the GridX commutator floor.
            cosx_grid_built = build_grid(
                mol, resolve_cosx_grid_options(opts.cosx_grid, 0, cosx_card));
        } else if (spinlock_active) {
            // SPINLOCK runs two sequential SCFs; use the single fine grid
            // rather than the multi-stage progression.
            cosx_grid_built = build_grid(
                mol, cosx_grid_options_for_level(
                         resolve_cosx_grid_level(opts.cosx_grid_level,
                                                 cosx_card)));
        } else {
            // GridX multi-stage progression (P2): stage-0 (coarse) grid
            // seeds the guess; finer stages are built on the fly.
            cosx_stages = cosx_grid_stages_for_level(
                resolve_cosx_grid_level(opts.cosx_grid_level, cosx_card));
            cosx_grid_built = build_grid(mol, cosx_stages.front());
            cosx_multistage = true;
        }
    }

    // Range-separated hybrids need the erf-attenuated K build — only
    // the direct-SCF builder has it. See run_rks for the rationale.
    const bool functional_is_rsh = functional_uks.is_range_separated();

    // ---- JKBuilder dispatch (FourIndex / DF / COSX) ------------------------
    std::unique_ptr<BasisSet> aux;
    std::unique_ptr<JKBuilder> jk;
    if (opts.density_fit) {
        if (functional_is_rsh) {
            throw std::invalid_argument(
                "run_uks: range-separated hybrids (ωB97X, …) are not yet "
                "supported with density_fit=true — run with "
                "density_fit=false (the direct-SCF path handles RSH).");
        }
        if (opts.aux_basis.empty()) {
            throw std::invalid_argument(
                "run_uks: density_fit=true requires aux_basis to be set "
                "(e.g. \"def2-svp-jk\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"jk\") "
                "for autodetection.");
        }
        aux = std::make_unique<BasisSet>(mol, opts.aux_basis);
        if (cosx_exchange_active) {
            jk = make_cosx_jk_builder(basis, *aux, cosx_grid_built,
                                      cosx_variant_resolved);
        } else {
            jk = make_df_jk_builder(basis, *aux);
        }
    } else {
        const SCFMode mode = functional_is_rsh
            ? SCFMode::DIRECT
            : resolve_scf_mode(
                  opts.scf_mode, static_cast<int>(basis.nbasis()),
                  opts.scf_mode_auto_threshold);
        if (mode == SCFMode::DIRECT) {
            // Two-phase Schwarz: see run_rhf.
            const double initial_thr =
                (opts.schwarz_threshold_loose > opts.schwarz_threshold)
                    ? opts.schwarz_threshold_loose
                    : opts.schwarz_threshold;
            jk = make_direct_jk_builder(
                basis, initial_thr,
                opts.incremental_fock, opts.incremental_fock_reset_freq);
        } else {
            jk = make_four_index_jk_builder(basis);
        }
    }

    auto prepared = prepare_open_guess(
        &mol, basis, n_alpha, n_beta, opts.initial_guess, S, Hcore, *jk,
        {}, {}, opts.read_density_alpha, opts.read_density_beta,
        opts.atomic_spins, opts.linear_dep_threshold, true, nullptr, molecular_guess_ecp_context(opts));
    const auto& gDa = prepared.alpha;
    const auto& gDb = prepared.beta;

    // SPINLOCK spin-schedule (mode A): mirror of run_uhf. Converge at a locked
    // n_alpha-n_beta for the first spinlock_iterations cycles, then restart at
    // the multiplicity target from that density (two sequential SCFs).
    if (spinlock_active) {
        const int n_elec = n_alpha + n_beta;
        if (opts.spinlock_value > n_elec || -opts.spinlock_value > n_elec
            || ((n_elec + opts.spinlock_value) % 2) != 0) {
            throw std::runtime_error(
                "UKS SPINLOCK: spinlock_value=" + std::to_string(opts.spinlock_value)
                + " is incompatible with " + std::to_string(n_elec)
                + " electrons (need |value| <= n_elec and matching parity)");
        }
        const int na_lock = (n_elec + opts.spinlock_value) / 2;
        const int nb_lock = (n_elec - opts.spinlock_value) / 2;
        UKSOptions opts_lock = opts;
        opts_lock.spinlock_mode = SpinlockMode::OFF;
        opts_lock.max_iter = opts.spinlock_iterations;
        const auto locked = run_uks_scf_with_jk(
            basis, na_lock, nb_lock, S, Hcore, E_nuc, *jk, grid, opts_lock,
            gDa, gDb, vv10_grid, &mol, &prepared.selection);
        prepared.selection.transport = InitialGuess::READ;
        UKSOptions opts_release = opts;
        opts_release.spinlock_mode = SpinlockMode::OFF;
        return with_ecp_provenance(run_uks_scf_with_jk(
            basis, n_alpha, n_beta, S, Hcore, E_nuc, *jk, grid, opts_release,
            locked.density_alpha, locked.density_beta, vv10_grid, &mol, &prepared.selection));
    }
    if (cosx_multistage) {
        Eigen::MatrixXd Da = gDa, Db = gDb;
        auto seg = [&](const JKBuilder& seg_jk,
                       const UKSOptions& so) -> UKSResult {
            UKSResult r = run_uks_scf_with_jk(basis, n_alpha, n_beta, S, Hcore,
                                              E_nuc, seg_jk, grid, so, Da, Db,
                                              vv10_grid, &mol, &prepared.selection);
            prepared.selection.transport = InitialGuess::READ;
            Da = r.density_alpha;
            Db = r.density_beta;
            return r;
        };
        return with_ecp_provenance(run_cosx_staged_scf<UKSResult>(
            mol, cosx_stages, opts, *jk, seg));
    }
    return with_ecp_provenance(run_uks_scf_with_jk(
        basis, n_alpha, n_beta, S, Hcore, E_nuc, *jk, grid, opts, gDa, gDb,
        vv10_grid, &mol, &prepared.selection));
}

UKSResult run_uks_scf_once(const BasisSet& basis,
                              int n_alpha,
                              int n_beta,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const JKBuilder& jk,
                              const Grid& grid,
                              const UKSOptions& opts,
                              const Eigen::MatrixXd& init_alpha,
                              const Eigen::MatrixXd& init_beta,
                              const Grid* vv10_grid = nullptr,
                              const Eigen::MatrixXd* mom_anchor_alpha = nullptr,
                              const Eigen::MatrixXd* mom_anchor_beta = nullptr,
                              int mom_anchor_iters = 0) {
    validate_orbital_optimizer(opts);
    validate_scf_max_iter(opts.max_iter, "run_uks_scf_with_jk");
    if (n_alpha < 0 || n_beta < 0) {
        throw std::invalid_argument(
            "run_uks_scf_once: per-spin electron counts must be "
            "non-negative");
    }
    validate_fraction_01("run_uks_scf_once: damping", opts.damping);
    validate_fraction_01(
        "run_uks_scf_once: fock_mixing", opts.fock_mixing);
    if (S.rows() != Hcore.rows() || S.cols() != Hcore.cols()
        || S.rows() != static_cast<Eigen::Index>(basis.nbasis())) {
        throw std::invalid_argument(
            "run_uks_scf_once: S / Hcore shapes do not match "
            "basis.nbasis()");
    }
    if ((init_alpha.size() == 0) != (init_beta.size() == 0)) {
        throw std::invalid_argument(
            "run_uks_scf_once: init_alpha and init_beta must both be "
            "supplied or both be empty");
    }

    // Canonical orthogonalization; see rhf.cpp for the shared notes.
    const auto orth = canonical_orthogonalizer(S, opts.linear_dep_threshold);
    if (orth.n_kept == 0) {
        throw std::runtime_error(
            "UKS: AO basis has no non-null directions above the "
            "linear-dependence threshold (threshold = "
            + std::to_string(opts.linear_dep_threshold) + ")");
    }
    if (std::max(n_alpha, n_beta) > orth.n_kept) {
        throw std::runtime_error(
            "UKS: canonical orthogonalization dropped too many basis "
            "directions for this electron count (max(n_α, n_β) = "
            + std::to_string(std::max(n_alpha, n_beta)) + ", n_kept = "
            + std::to_string(orth.n_kept) + ")");
    }
    const Eigen::MatrixXd& X = orth.X;

    // ---- Davidson iterative diagonalization setup -------------------------
    const bool use_davidson_here =
        opts.use_davidson && orth.n_kept >= opts.davidson_min_dim;
    DavidsonOptions dav_opts = opts.davidson;
    if (use_davidson_here) {
        if (dav_opts.n_eig == 0) dav_opts.n_eig = orth.n_kept;
    }
    Eigen::MatrixXd dav_guess_ortho;  // for eigenvector recycling

    // ---- Functional in spin-polarized mode ----
    Functional functional(opts.functional, /*spin=*/2);
    const UksXcFunctionalPool xc_functionals =
        make_uks_xc_functional_pool(opts.functional, grid.points.rows());
    const double alpha_hf = functional.hf_exchange_fraction();
    // Range-separated (CAM) hybrid bookkeeping — see run_rks for the
    // rationale. For a global hybrid cam_a == alpha_hf, cam_b == 0.
    const bool is_rsh = functional.is_range_separated();
    const double cam_a = functional.cam_alpha();
    const double cam_b = functional.cam_beta();
    const double rsh_omega = functional.rsh_omega();
    const bool need_k = (cam_a != 0.0 || cam_b != 0.0);

    // The opt-in Newton/TRAH f_xc builders still cache dense whole-grid AO
    // tables.  Delay that fallback until a second-order phase is entered;
    // ordinary UKS XC uses the bounded batch path below.  The memory
    // estimator keeps the dense charge when either threshold is enabled.
    std::unique_ptr<AOValues> xc_kernel_ao;
    auto make_xc_kernel = [&](const Eigen::MatrixXd& density_alpha,
                              const Eigen::MatrixXd& density_beta) {
        if (!xc_kernel_ao) {
            xc_kernel_ao = std::make_unique<AOValues>(
                evaluate_ao_with_gradient(basis, grid.points));
        }
        return make_polarised_xc_kernel_builder(
            functional, grid, xc_kernel_ao->values,
            xc_kernel_ao->gradients, density_alpha, density_beta);
    };

    // ---- Initial α/β densities ----
    // Diagonalize the guess Fock (Hcore or J(D_SAD) on top of Hcore), then
    // form D_σ by occupying the n_σ lowest orbitals. The rank asymmetry
    // (n_α > n_β for open shell) gives genuine spin polarization from
    // iteration 1 — a pure scaling of a closed-shell density does not,
    // and leaves UKS stuck at the wrong local minimum for pathological
    // systems (OH / LDA being the canonical example).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fock_solver;
    auto diagonalize = [&](const Eigen::MatrixXd& F)
        -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
        const Eigen::MatrixXd Fp = X.transpose() * F * X;
        if (use_davidson_here) {
            if (dav_guess_ortho.size() != 0) {
                dav_opts.guess_vectors = dav_guess_ortho;
            }
            DavidsonResult dres = davidson_solve(Fp, dav_opts);
            if (!dres.converged) {
                throw std::runtime_error(
                    "UKS: Davidson diagonalization did not converge after "
                    + std::to_string(dres.n_iter) + " iterations");
            }
            dav_guess_ortho = dres.eigenvectors;
            return {X * dres.eigenvectors, dres.eigenvalues};
        }
        fock_solver.compute(Fp);
        if (fock_solver.info() != Eigen::Success) {
            throw std::runtime_error("UKS: Fock diagonalization failed");
        }
        return {X * fock_solver.eigenvectors(), fock_solver.eigenvalues()};
    };

    // ---- Initial densities ----
    // The SAD-fock-guess construction (Hcore + J(D_sad) − ½K(D_sad)
    // → diagonalise → split per-spin) lives in the run_uks wrapper —
    // it's molecule-specific. By the time the SCF core sees them,
    // init_alpha / init_beta are either user-supplied or the wrapper-
    // computed SAD densities; falling through to Hcore-diag is the
    // periodic-Γ / arbitrary-Hcore caller's path.
    Eigen::MatrixXd D_alpha, D_beta;
    Eigen::MatrixXd C0;
    Eigen::VectorXd eps0;
    std::tie(C0, eps0) = diagonalize(Hcore);
    if (init_alpha.size() != 0) {
        if (init_alpha.rows() != S.rows() || init_alpha.cols() != S.cols()
            || init_beta.rows()  != S.rows() || init_beta.cols()  != S.cols()) {
            throw std::invalid_argument(
                "run_uks_scf_once: init_alpha / init_beta shapes do "
                "not match S");
        }
        D_alpha = init_alpha;
        D_beta  = init_beta;
    } else {
        D_alpha = build_spin_density(C0, n_alpha);
        D_beta  = build_spin_density(C0, n_beta);
    }
    Eigen::MatrixXd Da_prev = D_alpha;
    Eigen::MatrixXd Db_prev = D_beta;

    // Track per-spin MO frame between iterations so the C1c Newton step
    // has a current MO basis to operate in. Updated each iteration after
    // either diagonalisation or the quadratic step.
    Eigen::MatrixXd C_alpha_prev_mo = C0;
    Eigen::MatrixXd C_beta_prev_mo  = C0;
    Eigen::VectorXd eps_alpha_prev_mo = eps0;
    Eigen::VectorXd eps_beta_prev_mo  = eps0;

    UKSResult result;
    result.restart_basis = basis;
    result.functional = opts.functional;
    result.e_nuclear = E_nuc;
    {
        const double dS = 0.5 * (n_alpha - n_beta);
        result.s_squared_ideal = dS * (dS + 1.0);
    }

    if (opts.orbital_optimizer == "opentrustregion") {
        if (opts.spinlock_mode != SpinlockMode::OFF || !opts.atomic_spins.empty())
            throw std::invalid_argument("OpenTrustRegion does not support targeted spin states or spinlock; select native");
        validate_orbital_xc(functional);
        OrbitalXCFunction xc = [&](const Eigen::MatrixXd& da, const Eigen::MatrixXd& db, bool response) {
            const auto value = build_uks_xc(functional, basis, grid, da, db, vv10_grid, xc_functionals);
            result.xc_batch_workers_used = std::max(result.xc_batch_workers_used, value.batch_workers);
            OrbitalXC out; out.energy = value.energy; out.alpha = value.V_alpha; out.beta = value.V_beta;
            if (response) out.unrestricted_kernel = make_xc_kernel(da, db);
            return out;
        };
        const auto run = run_orbital_scf(opts, X, S, Hcore, E_nuc, jk,
            D_alpha, D_beta, n_alpha, n_beta, false, alpha_hf, xc);
        assign_orbital_unrestricted(result, run, E_nuc, S, n_alpha, n_beta);
        result.e_coulomb = run.state.coulomb;
        result.e_hf_exchange = run.state.exchange;
        result.e_xc = run.state.xc.energy;
        return result;
    }

    // ORCA NOITER compatibility: evaluate the initial spin densities once,
    // without taking an SCF update or recording an iteration.
    if (opts.max_iter == 0) {
        const Eigen::MatrixXd D_total = D_alpha + D_beta;
        const Eigen::MatrixXd J = jk.build_J_slot(D_total, 0);
        Eigen::MatrixXd K_a, K_b;
        if (need_k) {
            K_a = cam_a * jk.build_K_slot(D_alpha, 0);
            const bool spin_degenerate = spin_densities_are_degenerate(
                D_alpha, D_beta, n_alpha, n_beta);
            if (spin_degenerate) {
                K_b = K_a;
            } else {
                K_b = cam_a * jk.build_K_slot(D_beta, 1);
            }
            if (cam_b != 0.0 && spin_degenerate) {
                K_a.noalias() += cam_b * jk.build_K_erf(D_alpha, rsh_omega);
                K_b = K_a;
            } else if (cam_b != 0.0) {
                K_a.noalias() += cam_b * jk.build_K_erf(D_alpha, rsh_omega);
                K_b.noalias() += cam_b * jk.build_K_erf(D_beta, rsh_omega);
            }
        }
        UksXc xc = build_uks_xc(
            functional, basis, grid, D_alpha, D_beta, vv10_grid,
            xc_functionals);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc.batch_workers);
        Eigen::MatrixXd F_alpha = Hcore + J + xc.V_alpha;
        Eigen::MatrixXd F_beta = Hcore + J + xc.V_beta;
        if (need_k) {
            F_alpha -= K_a;
            F_beta -= K_b;
        }

        const double E_core = (D_total.array() * Hcore.array()).sum();
        const double E_J = 0.5 * (D_total.array() * J.array()).sum();
        double E_K = 0.0;
        if (need_k) {
            E_K = -0.5 * (
                (D_alpha.array() * K_a.array()).sum()
                + (D_beta.array() * K_b.array()).sum());
        }
        const double E_xc = xc.energy;
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu_a = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                D_alpha, S);
            const auto vu_b = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                D_beta, S);
            F_alpha += vu_a.V;
            F_beta += vu_b.V;
            e_dft_plus_u = vu_a.energy + vu_b.energy;
        }
        auto [Ca, eps_a] = diagonalize(F_alpha);
        auto [Cb, eps_b] = diagonalize(F_beta);
        result.mo_energies_alpha = eps_a;
        result.mo_energies_beta = eps_b;
        result.mo_coeffs_alpha = Ca;
        result.mo_coeffs_beta = Cb;
        result.density_alpha = D_alpha;
        result.density_beta = D_beta;
        result.fock_alpha = F_alpha;
        result.fock_beta = F_beta;
        result.energy = E_core + E_J + E_K + E_xc + E_nuc
                      + e_dft_plus_u;
        result.e_electronic = E_core + E_J + E_K + E_xc;
        result.e_coulomb = E_J;
        result.e_hf_exchange = E_K;
        result.e_xc = E_xc;
        result.e_dft_plus_u = e_dft_plus_u;
        result.s_squared = compute_s_squared(Ca, n_alpha, Cb, n_beta, S);
        return result;
    }

    // One spin-coupled Pulay history (see DIIS::extrapolate_spin_coupled),
    // same convention as uhf.cpp: the spins share J(D_α + D_β), so per-spin
    // histories can stall the SCF tail. The adaptive depth policy
    // (R_CDIIS / AD_CDIIS) is baked in via make_diis.
    DIIS diis = make_diis(opts);
    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);
    KDIIS kdiis(opts.diis_subspace_size);
    UHFSOSCF usoscf(opts.soscf_opts);  // stateful L-BFGS SOSCF accelerator

    // Dynamic-damping state (see RHF). Inactive by default.
    double current_damping = opts.damping;
    bool have_prev_E = false;
    bool full_fock_validation_pending = false;
    int fock_map_epoch_start_iter = 1;

    double E_prev = 0.0;
    Eigen::MatrixXd F_alpha_prev_mixed;
    Eigen::MatrixXd F_beta_prev_mixed;
    bool have_prev_fock = false;

    // Auto-level-shift-on-oscillation state — see rhf.cpp for the full
    // description. Applied per-spin.
    bool oscillation_engaged = false;
    double effective_level_shift = opts.level_shift;
    int effective_warmup = opts.level_shift_warmup_cycles;

    int num_restarts = 0;
    int stall_iters = 0;
    double best_grad_norm = 1e300;
    double best_energy = 0.0;
    Eigen::MatrixXd best_Ca = C0;
    Eigen::MatrixXd best_Cb = C0;
    Eigen::VectorXd best_eps_a = eps0;
    Eigen::VectorXd best_eps_b = eps0;
    Eigen::MatrixXd best_Da = D_alpha;
    Eigen::MatrixXd best_Db = D_beta;

    // Phase D2c/D2d/D2e-KS-UHF state (mirror of rks.cpp).
    double trah_trust_radius   = opts.trah_opts.initial_trust_radius;
    double trah_predicted_decrease = 0.0;
    double trah_kappa_norm     = 0.0;   // ‖κ‖ of the previous TRAH step
    bool   trah_active_prev    = false;
    // Two-phase direct-SCF Schwarz tightening — see run_rhf comment.
    const bool schwarz_two_phase =
        opts.schwarz_threshold_loose > opts.schwarz_threshold;
    bool schwarz_tightened = !schwarz_two_phase;
    // Per-iteration wall clock for the SCFIteration trace
    // (BUG87-A truthful-timers fix): each trace row is charged
    // with the wall time since the previous row was recorded.
    auto t_iter_prev = std::chrono::steady_clock::now();
    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        const bool validating_first_full_build =
            full_fock_validation_pending;
        full_fock_validation_pending = false;

        // Quadratic-fallback phase: per-spin Newton step replaces the
        // diagonalize-F update once iter > quadratic_fallback_iter. DIIS
        // and damping are skipped during the quadratic phase, same as RHF.
        const bool in_quadratic_phase =
            opts.quadratic_fallback_iter > 0
            && iter > opts.quadratic_fallback_iter;

        // Phase D2c/D2d/D2e-KS-UHF activation flags (recomputed after
        // grad_norm is known below).
        bool in_newton_phase = false;
        bool in_trah_phase   = false;
        bool in_soscf_phase  = false;

        // SPINLOCK PATTERN_HOLD: the accelerator is suspended (no history
        // recorded, no extrapolation, damping stays live) while the hold
        // is active. Fock extrapolation across held-window iterates steers
        // the SCF toward the symmetric attractor by continuous orbital
        // rotation, a collapse the occupation-selecting MOM hold cannot
        // see, and poisons the post-release history with out-of-basin
        // iterates. The history starts fresh at release. Mirrors the
        // periodic drivers (periodic_uks_multi_k_ewald.py & siblings).
        const bool hold_active =
            opts.spinlock_mode == SpinlockMode::PATTERN_HOLD
            && opts.spinlock_iterations > 0
            && iter <= opts.spinlock_iterations;
        // Stability-escape MOM anchor hold: keep the occupation pattern
        // from the rotated reference while the mean field self-consists
        // around it (mirrors UHF run_uhf_scf_once).
        const bool anchor_hold_active =
            mom_anchor_alpha != nullptr && mom_anchor_beta != nullptr
            && mom_anchor_iters > 0 && iter <= mom_anchor_iters;

        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter
            && !in_quadratic_phase && !hold_active && !anchor_hold_active;

        // Gate on the iteration-local current_damping, not opts.damping:
        // with damping=0.0 + dynamic_damping=true the dynamic update is
        // the only source of a non-zero mixing factor (matches rhf.cpp).
        const bool density_mixed_this_iter =
            iter != 1 && current_damping != 0.0 && !diis_active
            && !in_quadratic_phase;
        auto damp = [&](const Eigen::MatrixXd& now,
                        const Eigen::MatrixXd& prev) {
            if (!density_mixed_this_iter) return now;
            return Eigen::MatrixXd(current_damping * prev
                                   + (1.0 - current_damping) * now);
        };
        const Eigen::MatrixXd Da_used = damp(D_alpha, Da_prev);
        const Eigen::MatrixXd Db_used = damp(D_beta,  Db_prev);

        // J slot 0, K_α slot 0, K_β slot 1 — per-spin incremental
        // caches need distinct slots (see DirectJKBuilder); the
        // _slot wrappers delegate to the stateless path on non-
        // Direct builders + when incremental_fock=false.
        const Eigen::MatrixXd D_total = Da_used + Db_used;
        const Eigen::MatrixXd J = jk.build_J_slot(D_total, 0);

        // Per-spin effective exchange K_σ = cam_a·K(D_σ) + cam_b·K_erf(D_σ).
        // For a global hybrid cam_b == 0 → K_σ = α·K(D_σ).
        Eigen::MatrixXd K_a, K_b;
        if (need_k) {
            K_a = cam_a * jk.build_K_slot(Da_used, 0);
            const bool spin_degenerate = spin_densities_are_degenerate(
                Da_used, Db_used, n_alpha, n_beta);
            if (spin_degenerate) {
                K_b = K_a;
            } else {
                K_b = cam_a * jk.build_K_slot(Db_used, 1);
            }
            if (cam_b != 0.0 && spin_degenerate) {
                K_a.noalias() += cam_b * jk.build_K_erf(Da_used, rsh_omega);
                K_b = K_a;
            } else if (cam_b != 0.0) {
                K_a.noalias() += cam_b * jk.build_K_erf(Da_used, rsh_omega);
                K_b.noalias() += cam_b * jk.build_K_erf(Db_used, rsh_omega);
            }
        }

        UksXc xc = build_uks_xc(
            functional, basis, grid, Da_used, Db_used, vv10_grid,
            xc_functionals);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc.batch_workers);

        Eigen::MatrixXd F_alpha = Hcore + J + xc.V_alpha;
        Eigen::MatrixXd F_beta  = Hcore + J + xc.V_beta;
        if (need_k) {
            // K_a / K_b already carry the cam_a / cam_b mixing.
            F_alpha -= K_a;
            F_beta  -= K_b;
        }

        // Energy decomposition (KS pieces only — Dudarev +U is added
        // separately below; same double-counting discipline as
        // RHF/RKS/UHF).
        const double E_core = (D_total.array() * Hcore.array()).sum();
        const double E_J = 0.5 * (D_total.array() * J.array()).sum();
        double E_K = 0.0;
        if (need_k) {
            E_K = -0.5 * (
                (Da_used.array() * K_a.array()).sum()
                + (Db_used.array() * K_b.array()).sum());
        }
        const double E_xc = xc.energy;

        // Dudarev +U Fock contribution — per-spin convention,
        // kernel called once per spin with that spin's density.
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu_a = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                Da_used, S);
            const auto vu_b = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                Db_used, S);
            F_alpha += vu_a.V;
            F_beta  += vu_b.V;
            e_dft_plus_u = vu_a.energy + vu_b.energy;
        }

        const double E_total = E_core + E_J + E_K + E_xc + E_nuc
                             + e_dft_plus_u;

        // Pulay, J. Comput. Chem. 3, 556 (1982), Eq. (4) and p. 557:
        // e_σ^AO = F_σ D_σ S - S D_σ F_σ, followed by the orthonormal-basis
        // error e_σ = X^T e_σ^AO X with X^T S X = I. Use that same balanced
        // metric for both the coupled-spin DIIS Gram matrix and convergence.
        // Reference regression (SH radical, B3LYP/def2-SVP, DefGrid3): the
        // terminal beta AO norm 1.339028317e-6 becomes 9.103544775e-7 after
        // projection, below the unchanged conv_tol_grad = 1e-6 contract.
        const Eigen::MatrixXd err_a_ao =
            F_alpha * Da_used * S - S * Da_used * F_alpha;
        const Eigen::MatrixXd err_b_ao =
            F_beta * Db_used * S - S * Db_used * F_beta;
        const Eigen::MatrixXd err_a = X.transpose() * err_a_ao * X;
        const Eigen::MatrixXd err_b = X.transpose() * err_b_ao * X;
        const double grad_norm = std::max(err_a.norm(), err_b.norm());
        const double dE = E_total - E_prev;

        VIBEQC_DIAG("uks", vibeqc::DiagLevel::VERBOSE,
            "iter %3d  E=% .10f  dE=%+.3e  |grad|=%.3e",
            iter, E_total, (iter == 1 ? 0.0 : dE), grad_norm);

        // Two-phase Schwarz tightening; see run_rhf for full comment.
        bool schwarz_tightened_this_iter = false;
        if (!schwarz_tightened
            && grad_norm < opts.schwarz_threshold_tighten_at) {
            jk.set_schwarz_threshold(opts.schwarz_threshold);
            jk.reset_state();
            schwarz_tightened = true;
            schwarz_tightened_this_iter =
                jk.uses_runtime_schwarz_threshold();
            if (schwarz_tightened_this_iter) {
                full_fock_validation_pending = true;
                fock_map_epoch_start_iter = iter + 1;
            }
        }

        const bool coarse_fock_active =
            jk.incremental_active()
            || schwarz_tightened_this_iter
            || (!schwarz_tightened
                && jk.uses_runtime_schwarz_threshold());
        const bool begin_full_fock_refinement =
            needs_full_fock_refinement(
                iter, grad_norm, opts.conv_tol_grad,
                coarse_fock_active);
        const bool density_converged =
            !validating_first_full_build
            && !begin_full_fock_refinement
            && is_scf_converged(
                iter, dE, grad_norm,
                opts.conv_tol_energy, opts.conv_tol_grad);
        if (density_converged) {
            VIBEQC_DIAG("uks", vibeqc::DiagLevel::DEBUG,
                "density gates cleared at iter %d: |dE|=%.3e < %.0e, "
                "|grad|=%.3e < %.0e; validating the returned determinant",
                iter, std::abs(dE), opts.conv_tol_energy,
                grad_norm, opts.conv_tol_grad);
        }
        Eigen::MatrixXd density_gate_Ca_occ;
        Eigen::MatrixXd density_gate_Cb_occ;
        if (density_converged) {
            density_gate_Ca_occ = C_alpha_prev_mo.leftCols(n_alpha);
            density_gate_Cb_occ = C_beta_prev_mo.leftCols(n_beta);
        }

        // D2c/D2d/D2e-KS-UHF activation. Same mutual-exclusion priority
        // as RKS / RHF: quadratic > Newton > TRAH > SOSCF.
        // Newton / TRAH are disabled for range-separated hybrids — their
        // single-``alpha`` orbital-Hessian exchange response cannot
        // represent the two-term (cam_a·K + cam_b·K_erf) RSH response.
        // See run_rks for the full rationale.
        in_newton_phase =
            opts.newton_threshold > 0.0
            && grad_norm < opts.newton_threshold
            && !in_quadratic_phase
            && !is_rsh;
        in_trah_phase =
            opts.trah_threshold > 0.0
            && grad_norm < opts.trah_threshold
            && !in_quadratic_phase
            && !in_newton_phase
            && !is_rsh;
        in_soscf_phase =
            opts.soscf_threshold > 0.0
            && grad_norm < opts.soscf_threshold
            && !in_quadratic_phase
            && !in_newton_phase
            && !in_trah_phase;
        const bool in_second_order_phase =
            in_newton_phase || in_trah_phase || in_soscf_phase;

        // Accelerator history is skipped entirely (not even recorded)
        // while the PATTERN_HOLD window is active; see the hold_active
        // note above.
        int diis_sub = 0;
        // Extrapolation that actually reached F this iteration (#682); see
        // SCFIteration::accelerator_step for the codes.
        int accelerator_step_this_iter = 0;
        if (opts.use_diis && !in_quadratic_phase && !in_second_order_phase
            && !hold_active) {
            Eigen::MatrixXd Fa_ext = F_alpha;
            Eigen::MatrixXd Fb_ext = F_beta;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the spin-coupled DIIS
                // extrapolation; the adaptive depth policy is baked into
                // ``diis`` at construction (see make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    auto d = diis.extrapolate_spin_coupled(
                        F_alpha, F_beta, err_a, err_b);
                    Fa_ext = std::move(d.first);
                    Fb_ext = std::move(d.second);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    accelerator_step_this_iter = 1;
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    auto k = kdiis.extrapolate(
                        F_alpha, F_beta,
                        C_alpha_prev_mo, C_beta_prev_mo,
                        eps_alpha_prev_mo, eps_beta_prev_mo,
                        n_alpha, n_beta);
                    Fa_ext = std::move(k.first);
                    Fb_ext = std::move(k.second);
                    diis_sub = static_cast<int>(kdiis.subspace_size());
                    accelerator_step_this_iter = 2;
                    break;
                }
                case SCFAccelerator::EDIIS: {
                    auto e = ediis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used, E_total);
                    Fa_ext = std::move(e.first);
                    Fb_ext = std::move(e.second);
                    diis_sub = static_cast<int>(ediis.subspace_size());
                    accelerator_step_this_iter = 3;
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    auto d = diis.extrapolate_spin_coupled(
                        F_alpha, F_beta, err_a, err_b);
                    Eigen::MatrixXd Fa_d = std::move(d.first);
                    Eigen::MatrixXd Fb_d = std::move(d.second);
                    auto e = ediis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used, E_total);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm,
                            static_cast<Eigen::Index>(F_alpha.rows()),
                            2);
                    if (switch_metric > opts.ediis_diis_switch_threshold) {
                        Fa_ext = std::move(e.first);
                        Fb_ext = std::move(e.second);
                        accelerator_step_this_iter = 3;
                    } else {
                        // A DIIS-branch cycle discards the EDIIS pair;
                        // retract it so the anti-replay guard keys on
                        // consumed returns only.
                        ediis.discard_last_extrapolation();
                        Fa_ext = std::move(Fa_d);
                        Fb_ext = std::move(Fb_d);
                        accelerator_step_this_iter = 1;
                    }
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    auto a = adiis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used);
                    Fa_ext = std::move(a.first);
                    Fb_ext = std::move(a.second);
                    diis_sub = static_cast<int>(adiis.subspace_size());
                    accelerator_step_this_iter = 4;
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    auto d = diis.extrapolate_spin_coupled(
                        F_alpha, F_beta, err_a, err_b);
                    Eigen::MatrixXd Fa_d = std::move(d.first);
                    Eigen::MatrixXd Fb_d = std::move(d.second);
                    auto a = adiis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm,
                            static_cast<Eigen::Index>(F_alpha.rows()),
                            2);
                    if (switch_metric > opts.ediis_diis_switch_threshold) {
                        Fa_ext = std::move(a.first);
                        Fb_ext = std::move(a.second);
                        accelerator_step_this_iter = 4;
                    } else {
                        adiis.discard_last_extrapolation();
                        Fa_ext = std::move(Fa_d);
                        Fb_ext = std::move(Fb_d);
                        accelerator_step_this_iter = 1;
                    }
                    break;
                }
            }
            if (diis_active) {
                F_alpha = Fa_ext;
                F_beta  = Fb_ext;
            } else {
                // Below diis_start_iter nothing extrapolated reaches the
                // SCF, so every branch's return is discarded. Same
                // produced-vs-consumed retraction as the hybrid above.
                ediis.discard_last_extrapolation();
                adiis.discard_last_extrapolation();
                accelerator_step_this_iter = 0;
            }
        }
        if (!in_quadratic_phase && opts.fock_mixing != 0.0) {
            if (have_prev_fock) {
                F_alpha = mix_fock_matrices(
                    F_alpha, F_alpha_prev_mixed, opts.fock_mixing);
                F_beta = mix_fock_matrices(
                    F_beta, F_beta_prev_mixed, opts.fock_mixing);
            }
            F_alpha_prev_mixed = F_alpha;
            F_beta_prev_mixed = F_beta;
            have_prev_fock = true;
        }

        result.scf_trace.push_back(SCFIteration{
            iter, E_total, (iter == 1) ? 0.0 : dE, grad_norm, diis_sub,
        });
        result.scf_trace.back().accelerator_step = accelerator_step_this_iter;
        {
            // Python fans this precision-preserving diagnostic out to the
            // live terminal, NDJSON, and vq manifest sinks.
            VIBEQC_DIAG("progress", vibeqc::DiagLevel::QUIET,
                "iter=%d energy=%.17g dE=%.17g grad=%.17g diis=%d",
                iter, E_total, result.scf_trace.back().delta_e, grad_norm,
                diis_sub);
        }
        {
            // Truthful per-iteration wall (BUG87-A): everything since the
            // previous trace row, so Fock build + energy + DIIS +
            // diagonalization are all covered.
            const auto t_iter_now = std::chrono::steady_clock::now();
            result.scf_trace.back().wall_s =
                std::chrono::duration<double>(t_iter_now - t_iter_prev)
                    .count();
            t_iter_prev = t_iter_now;
        }

        // Auto-level-shift-on-oscillation detection — see rhf.cpp for the
        // full description.
        const bool refresh_restart_baseline =
            iter == 1 || validating_first_full_build
            || E_total < best_energy;
        const int projected_stall_iters = stall_iters
            + ((!refresh_restart_baseline
                && grad_norm > best_grad_norm * 1.01) ? 1 : 0);
        const bool coarse_stall_transition_imminent =
            opts.restart_opts.enabled
            && !in_newton_phase && !in_trah_phase
            && projected_stall_iters >= opts.restart_opts.max_stall_iters
            && !density_converged
            && coarse_fock_active;
        if (opts.auto_level_shift_on_oscillation && !oscillation_engaged
            && iter >= opts.oscillation_detect_start_iter
            && !in_quadratic_phase && !in_second_order_phase
            && !coarse_stall_transition_imminent
            && !validating_first_full_build
            && !begin_full_fock_refinement
            && iter - fock_map_epoch_start_iter + 1
                   >= opts.oscillation_window
            && detect_scf_oscillation(result.scf_trace,
                                      opts.oscillation_window,
                                      opts.oscillation_min_sign_flips,
                                      opts.conv_tol_energy)) {
            oscillation_engaged = true;
            effective_level_shift = opts.auto_level_shift_value;
            effective_warmup = 0;  // persistent — held every iteration
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            VIBEQC_DIAG("uks", vibeqc::DiagLevel::STANDARD,
                "auto level-shift engaged at iter %d (b=%.2f Ha)",
                iter, effective_level_shift);
        }

        // Per-spin Saunders-Hillier level shift (C1a-2). Same as UHF
        // since UKS uses the unscaled D_σ = C_occ C_occ^T convention.
        Eigen::MatrixXd Ca_new, Cb_new;
        Eigen::VectorXd eps_a_new, eps_b_new;
        if (in_newton_phase) {
            // D2c-KS-UHF — build the polarised XC kernel pinned at the
            // current per-spin densities. The dispatcher picks the LDA
            // or (Phase 17e) GGA / hybrid-GGA builder by functional
            // kind; meta-GGA raises with a roadmap pointer.
            auto xc_kernel = make_xc_kernel(Da_used, Db_used);
            auto step = uhf_newton_step(F_alpha, F_beta,
                                         C_alpha_prev_mo, C_beta_prev_mo,
                                         eps_alpha_prev_mo, eps_beta_prev_mo,
                                         n_alpha, n_beta, jk,
                                         opts.newton_opts,
                                         xc_kernel.get(), alpha_hf);
            Ca_new = std::move(step.C_alpha);
            Cb_new = std::move(step.C_beta);
            eps_a_new = std::move(step.eps_alpha);
            eps_b_new = std::move(step.eps_beta);
        } else if (in_trah_phase) {
            // D2e-KS-UHF — same polarised XC kernel as Newton; LDA /
            // GGA / hybrid-GGA dispatched, meta-GGA raises.
            auto xc_kernel = make_xc_kernel(Da_used, Db_used);
            auto step = uhf_trah_step(F_alpha, F_beta,
                                       C_alpha_prev_mo, C_beta_prev_mo,
                                       eps_alpha_prev_mo, eps_beta_prev_mo,
                                       n_alpha, n_beta, jk,
                                       opts.trah_opts, trah_trust_radius,
                                       xc_kernel.get(), alpha_hf);
            Ca_new = std::move(step.C_alpha);
            Cb_new = std::move(step.C_beta);
            eps_a_new = std::move(step.eps_alpha);
            eps_b_new = std::move(step.eps_beta);
            trah_predicted_decrease = step.predicted_decrease;
            trah_kappa_norm = step.kappa_norm;
            // The trace row for this iteration was pushed above (before
            // the step); record the TRAH level shift into it now.
            result.scf_trace.back().trah_level_shift = step.level_shift;
        } else if (in_soscf_phase) {
            // D2d-KS-UHF Neese SOSCF — same per-spin AH eigsolve as UHF.
            // No XC kernel needed; F_σ already carries V_xc,σ.
            auto step = usoscf.step(F_alpha, F_beta,
                                    C_alpha_prev_mo, C_beta_prev_mo,
                                    eps_alpha_prev_mo, eps_beta_prev_mo,
                                    n_alpha, n_beta);
            Ca_new = std::move(step.C_alpha);
            Cb_new = std::move(step.C_beta);
            eps_a_new = std::move(step.eps_alpha);
            eps_b_new = std::move(step.eps_beta);
        } else if (in_quadratic_phase) {
            auto step_a = quadratic_step(F_alpha, C_alpha_prev_mo,
                                         eps_alpha_prev_mo, n_alpha,
                                         opts.quadratic_fallback_shift,
                                         opts.quadratic_fallback_max_step);
            auto step_b = quadratic_step(F_beta, C_beta_prev_mo,
                                         eps_beta_prev_mo, n_beta,
                                         opts.quadratic_fallback_shift,
                                         opts.quadratic_fallback_max_step);
            Ca_new = std::move(step_a.C);
            eps_a_new = std::move(step_a.eps);
            Cb_new = std::move(step_b.C);
            eps_b_new = std::move(step_b.eps);
        } else {
            // Standard per-spin diagonalize with the unified auto-reducing
            // Saunders-Hillier shift resolved for this iteration.
            // ``apply_level_shift`` returns F untouched when b is 0, so the
            // S·Dσ·S matmuls are skipped once the warm-up releases.
            // Da_used / Db_used are *spin* densities, hence SPIN: weight 1.
            // When oscillation has been auto-detected, the shift is resolved
            // from the engaged effective value rather than opts.level_shift.
            const double b = level_shift_at_iter(
                oscillation_engaged ? effective_level_shift
                                    : opts.level_shift,
                oscillation_engaged ? effective_warmup
                                    : opts.level_shift_warmup_cycles,
                opts.level_shift_schedule,
                opts.max_iter,
                iter);
            std::tie(Ca_new, eps_a_new) = diagonalize(apply_level_shift(
                F_alpha, S, Da_used, b, LevelShiftDensity::SPIN));
            std::tie(Cb_new, eps_b_new) = diagonalize(apply_level_shift(
                F_beta, S, Db_used, b, LevelShiftDensity::SPIN));
            // SPINLOCK pattern-hold: mirror of run_uhf_scf_with_jk. Hold the
            // seeded broken-symmetry occupation via maximum overlap (MOM) for
            // the first spinlock_iterations cycles, then release to aufbau, so
            // an ATOMSPIN / atomic_spins seed does not collapse to the
            // symmetric solution early. iter 1 sets the pattern by aufbau; MOM
            // holds it for iters 2..spinlock_iterations.
            if (opts.spinlock_mode == SpinlockMode::PATTERN_HOLD
                && opts.spinlock_iterations > 0
                && iter <= opts.spinlock_iterations && iter > 1
                && C_alpha_prev_mo.cols() >= n_alpha
                && C_beta_prev_mo.cols() >= n_beta) {
                mom_reorder_occupied(Ca_new, eps_a_new, S,
                                     C_alpha_prev_mo.leftCols(n_alpha), n_alpha);
                mom_reorder_occupied(Cb_new, eps_b_new, S,
                                     C_beta_prev_mo.leftCols(n_beta), n_beta);
            }
            // Stability-escape MOM anchor hold (mirrors UHF).
            if (anchor_hold_active) {
                mom_reorder_occupied(Ca_new, eps_a_new, S,
                                     *mom_anchor_alpha, n_alpha);
                mom_reorder_occupied(Cb_new, eps_b_new, S,
                                     *mom_anchor_beta, n_beta);
            }
        }
        C_alpha_prev_mo   = Ca_new;
        C_beta_prev_mo    = Cb_new;
        eps_alpha_prev_mo = eps_a_new;
        eps_beta_prev_mo  = eps_b_new;
        Da_prev = Da_used;
        Db_prev = Db_used;
        D_alpha = build_spin_density(Ca_new, n_alpha);
        D_beta  = build_spin_density(Cb_new, n_beta);

        // ---- deterministic orbital-rotation restart (BUG 64) --------
        if (iter == 1) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
        } else if (validating_first_full_build) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_new;
            best_Cb = Cb_new;
            best_eps_a = eps_a_new;
            best_eps_b = eps_b_new;
            best_Da = D_alpha;
            best_Db = D_beta;
        } else if (E_total < best_energy) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_new;
            best_Cb = Cb_new;
            best_eps_a = eps_a_new;
            best_eps_b = eps_b_new;
            best_Da = D_alpha;
            best_Db = D_beta;
        }
        if (grad_norm <= best_grad_norm * 1.01) {
            if (grad_norm < best_grad_norm) {
                best_grad_norm = grad_norm;
            }
        } else {
            ++stall_iters;
        }
        // IID 129 follow-up: an energy-only stall must still end the coarse
        // Fock phase; orbital rotation keeps its gradient guard below.
        const bool restart_stall_detected =
            opts.restart_opts.enabled
            && !in_newton_phase && !in_trah_phase
            && stall_iters >= opts.restart_opts.max_stall_iters
            && !density_converged;
        const bool full_fock_refinement_from_stall =
            restart_stall_detected && coarse_fock_active;
        const bool transition_to_full_fock =
            begin_full_fock_refinement || full_fock_refinement_from_stall;
        const bool fock_map_changed_this_iter =
            transition_to_full_fock || schwarz_tightened_this_iter;
        if (transition_to_full_fock) {
            if (!schwarz_tightened) {
                jk.set_schwarz_threshold(opts.schwarz_threshold);
                schwarz_tightened = true;
            }
            jk.set_incremental(false);
            jk.reset_state();
            stall_iters = 0;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_new;
            best_Cb = Cb_new;
            best_eps_a = eps_a_new;
            best_eps_b = eps_b_new;
            best_Da = D_alpha;
            best_Db = D_beta;
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            usoscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            full_fock_validation_pending = true;
            fock_map_epoch_start_iter = iter + 1;
            if (begin_full_fock_refinement) {
                VIBEQC_DIAG("uks", vibeqc::DiagLevel::STANDARD,
                    "direct-SCF coarse Fock phase ended at iter %d after "
                    "gradient convergence; certifying energy on consecutive "
                    "tight-screened full-density builds "
                    "(|dE|=%.3e, grad=%.3e)",
                    iter, std::abs(dE), grad_norm);
            } else {
                VIBEQC_DIAG("uks", vibeqc::DiagLevel::STANDARD,
                    "direct-SCF coarse Fock phase ended at iter %d after "
                    "drift stall; certifying energy on consecutive "
                    "tight-screened "
                    "full-density builds (grad=%.3e)",
                    iter, grad_norm);
            }
        } else if (restart_stall_detected
                   && grad_norm > opts.conv_tol_grad
                   && num_restarts < opts.restart_opts.max_restarts) {
            ++num_restarts;
            stall_iters = 0;
            VIBEQC_DIAG("uks", vibeqc::DiagLevel::STANDARD,
                "restart %d/%d at iter %d: grad=%.3e, best_grad=%.3e",
                num_restarts, opts.restart_opts.max_restarts,
                iter, grad_norm, best_grad_norm);
            auto [Ca_rot, eps_a_rot] = rotate_orbitals_in_subspaces(
                best_Ca, best_eps_a, n_alpha, &F_alpha,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4 + 1);
            auto [Cb_rot, eps_b_rot] = rotate_orbitals_in_subspaces(
                best_Cb, best_eps_b, n_beta, &F_beta,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4 + 2,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4 + 3);
            D_alpha = build_spin_density(Ca_rot, n_alpha);
            D_beta  = build_spin_density(Cb_rot, n_beta);
            Ca_new = Ca_rot;
            Cb_new = Cb_rot;
            eps_a_new = eps_a_rot;
            eps_b_new = eps_b_rot;
            // The restart replaces the determinant selected earlier in this
            // iteration.  Keep the previous-MO state synchronized with that
            // replacement: the next iteration's convergence snapshot, MOM
            // reference, and second-order step must all describe D_alpha /
            // D_beta, not the pre-restart occupied subspace.
            C_alpha_prev_mo = Ca_rot;
            C_beta_prev_mo = Cb_rot;
            eps_alpha_prev_mo = eps_a_rot;
            eps_beta_prev_mo = eps_b_rot;
            diis.clear(); ediis.clear(); adiis.clear(); kdiis.clear();
            usoscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_rot;
            best_Cb = Cb_rot;
            best_eps_a = eps_a_rot;
            best_eps_b = eps_b_rot;
            best_Da = D_alpha;
            best_Db = D_beta;
        }
        if (schwarz_tightened_this_iter && !transition_to_full_fock) {
            stall_iters = 0;
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            usoscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
        }
        if (fock_map_changed_this_iter) {
            trah_trust_radius = opts.trah_opts.initial_trust_radius;
            trah_predicted_decrease = 0.0;
            trah_kappa_norm = 0.0;
        }

        result.mo_energies_alpha = eps_a_new;
        result.mo_energies_beta  = eps_b_new;
        result.mo_coeffs_alpha = Ca_new;
        result.mo_coeffs_beta  = Cb_new;
        result.density_alpha = Da_used;
        result.density_beta  = Db_used;
        result.fock_alpha = F_alpha;
        result.fock_beta  = F_beta;
        result.energy = E_total;
        // +U-free KS electronic energy; see the matching note in
        // cpp/src/rks.cpp. Same convention across all four mean-field
        // drivers (RHF/UHF/RKS/UKS) post-Increment 4b.
        result.e_electronic = E_total - E_nuc - e_dft_plus_u;
        result.e_coulomb = E_J;
        result.e_hf_exchange = E_K;
        result.e_xc = E_xc;
        result.e_dft_plus_u = e_dft_plus_u;
        result.n_iter = iter;

        if (density_converged) {
            // Final self-consistency pass on the freshly built determinant.
            // Pulay, J. Comput. Chem. 3, 556 (1982), Eq. (4), makes the
            // convergence statement about one matching (D, F[D]) pair.  The
            // gates above certified Da_used / Db_used.  When density damping
            // was inactive, that pair is already the idempotent determinant
            // represented by density_gate_C{a,b}_occ, so preserve it rather
            // than advance to an untested DIIS image.  When damping made
            // D_used non-idempotent, validate the freshly diagonalized
            // D_alpha / D_beta determinant instead.  If either gate fails,
            // continue the ordinary SCF loop; never label an untested
            // determinant converged or return a damped density as an SCF
            // wavefunction.
            // result.energy and the e_coulomb / e_hf_exchange / e_xc
            // decomposition are likewise recomputed on the rebuilt
            // (J_f, K_f, xc_f) pair so the returned energy reproduces
            // from the returned matrices (2026-05-18 audit P3; see
            // tests/test_scf_final_consistency.py). The energy therefore
            // differs from scf_trace[-1].energy by at most ~conv_tol;
            // test_scf_log.py asserts that band.
            //
            // E_K_f uses the SCF's iterated K convention, i.e. it is
            // computed BEFORE the COSX one-center correction: the
            // correction upgrades only the returned Fock / MOs (see the
            // matching note in rhf.cpp; no-op for non-COSX builders).
            const Eigen::MatrixXd& Da_final = density_mixed_this_iter
                ? D_alpha : Da_used;
            const Eigen::MatrixXd& Db_final = density_mixed_this_iter
                ? D_beta : Db_used;
            const Eigen::MatrixXd Ca_final_occ = density_mixed_this_iter
                ? Eigen::MatrixXd(Ca_new.leftCols(n_alpha))
                : density_gate_Ca_occ;
            const Eigen::MatrixXd Cb_final_occ = density_mixed_this_iter
                ? Eigen::MatrixXd(Cb_new.leftCols(n_beta))
                : density_gate_Cb_occ;
            const Eigen::MatrixXd D_total_f = Da_final + Db_final;
            const Eigen::MatrixXd J_f = jk.build_J_slot(D_total_f, 0);
            Eigen::MatrixXd Ka_f, Kb_f;   // effective per-spin exchange
            double E_K_f = 0.0;
            if (need_k) {
                const bool spin_degenerate = spin_densities_are_degenerate(
                    Da_final, Db_final, n_alpha, n_beta);
                Ka_f = cam_a * jk.build_K_slot(Da_final, 0);
                if (spin_degenerate) {
                    Kb_f = Ka_f;
                } else {
                    Kb_f = cam_a * jk.build_K_slot(Db_final, 1);
                }
                if (cam_b != 0.0 && spin_degenerate) {
                    Ka_f.noalias() +=
                        cam_b * jk.build_K_erf(Da_final, rsh_omega);
                    Kb_f = Ka_f;
                } else if (cam_b != 0.0) {
                    Ka_f.noalias() +=
                        cam_b * jk.build_K_erf(Da_final, rsh_omega);
                    Kb_f.noalias() +=
                        cam_b * jk.build_K_erf(Db_final, rsh_omega);
                }
                // Ka_f / Kb_f already carry the cam_a / cam_b mixing, so
                // the exchange-energy coefficient is a bare -0.5
                // (matches the per-iteration E_K formula above).
                E_K_f = -0.5 * (
                      (Da_final.array() * Ka_f.array()).sum()
                    + (Db_final.array() * Kb_f.array()).sum());
            }
            UksXc xc_f = build_uks_xc(
                functional, basis, grid, Da_final, Db_final, vv10_grid,
                xc_functionals);
            result.xc_batch_workers_used = std::max(
                result.xc_batch_workers_used, xc_f.batch_workers);
            // Fa_scf_f / Fb_scf_f are the Fock matrices of the iterated SCF
            // energy map.  COSX's one-center replacement is a final orbital-
            // energy correction and is deliberately not part of that map;
            // validate against the former, while preserving the corrected
            // Fa_f / Fb_f result convention below.  They are identical for
            // every non-COSX builder.
            Eigen::MatrixXd Fa_scf_f = Hcore + J_f + xc_f.V_alpha;
            Eigen::MatrixXd Fb_scf_f = Hcore + J_f + xc_f.V_beta;
            if (need_k) {
                Fa_scf_f -= Ka_f;
                Fb_scf_f -= Kb_f;
            }
            if (!opts.dft_plus_u_sites.empty()) {
                const auto vu_a_f = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    Da_final, S);
                const auto vu_b_f = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    Db_final, S);
                Fa_scf_f += vu_a_f.V;
                Fb_scf_f += vu_b_f.V;
                result.e_dft_plus_u = vu_a_f.energy + vu_b_f.energy;
            }
            Eigen::MatrixXd Fa_f = Fa_scf_f;
            Eigen::MatrixXd Fb_f = Fb_scf_f;
            if (need_k) {
                Eigen::MatrixXd Ka_corrected = Ka_f;
                Eigen::MatrixXd Kb_corrected = Kb_f;
                jk.apply_one_center_correction(Ka_corrected, Da_final);
                if (spin_densities_are_degenerate(
                        Da_final, Db_final, n_alpha, n_beta)) {
                    Kb_corrected = Ka_corrected;
                } else {
                    jk.apply_one_center_correction(Kb_corrected, Db_final);
                }
                Fa_f.noalias() += Ka_f - Ka_corrected;
                Fb_f.noalias() += Kb_f - Kb_corrected;
            }
            auto [Ca_f, eps_a_f] = diagonalize(Fa_f);
            auto [Cb_f, eps_b_f] = diagonalize(Fb_f);
            // Keep the canonical orbitals on the determinant that is being
            // validated.  A plain aufbau slice may cross an occupied/virtual
            // near-degeneracy and silently report MOs for a different
            // stationary state even when [F,DS] is small.  The maximum-
            // overlap selection of Gilbert, Besley & Gill, J. Phys. Chem. A
            // 112, 13164 (2008), preserves the candidate occupied subspace;
            // it changes only column order, not the eigenvectors or energies.
            mom_reorder_occupied(Ca_f, eps_a_f, S,
                                 Ca_final_occ, n_alpha);
            mom_reorder_occupied(Cb_f, eps_b_f, S,
                                 Cb_final_occ, n_beta);
            const double E_core_f = (D_total_f.array() * Hcore.array()).sum();
            const double E_J_f    = 0.5 * (D_total_f.array() * J_f.array()).sum();
            const double E_xc_f = xc_f.energy;
            // result.e_dft_plus_u was re-reported on the fresh densities
            // in the +U branch above (0.0 when +U is off).
            const double E_total_f = E_core_f + E_J_f + E_K_f + E_xc_f
                                   + E_nuc + result.e_dft_plus_u;
            const Eigen::MatrixXd err_a_f_ao =
                Fa_scf_f * Da_final * S - S * Da_final * Fa_scf_f;
            const Eigen::MatrixXd err_b_f_ao =
                Fb_scf_f * Db_final * S - S * Db_final * Fb_scf_f;
            const double grad_norm_f = std::max(
                (X.transpose() * err_a_f_ao * X).norm(),
                (X.transpose() * err_b_f_ao * X).norm());
            const double dE_f = E_total_f - E_total;
            const bool returned_density_converged = is_scf_converged(
                iter + 1, dE_f, grad_norm_f,
                opts.conv_tol_energy, opts.conv_tol_grad);
            result.mo_energies_alpha = eps_a_f;
            result.mo_energies_beta  = eps_b_f;
            result.mo_coeffs_alpha = Ca_f;
            result.mo_coeffs_beta  = Cb_f;
            result.fock_alpha = Fa_f;
            result.fock_beta  = Fb_f;
            result.density_alpha = Da_final;
            result.density_beta  = Db_final;
            result.energy = E_total_f;
            // +U-free KS electronic energy; see the matching note in the
            // per-iteration block above.
            result.e_electronic = E_total_f - E_nuc - result.e_dft_plus_u;
            result.e_coulomb = E_J_f;
            result.e_hf_exchange = E_K_f;
            result.e_xc = E_xc_f;
            result.s_squared = compute_s_squared(Ca_f, n_alpha,
                                                  Cb_f, n_beta, S);
            if (returned_density_converged) {
                result.converged = true;
                return result;
            }
            VIBEQC_DIAG("uks", vibeqc::DiagLevel::STANDARD,
                "terminal determinant rejected at iter %d: "
                "|dE_next|=%.3e (tol %.0e), |grad_next|=%.3e (tol %.0e); "
                "continuing SCF",
                iter, std::abs(dE_f), opts.conv_tol_energy,
                grad_norm_f, opts.conv_tol_grad);
        }

        // Powell-ρ TRAH trust-radius update (mirror of rks.cpp).
        if (trah_active_prev && iter > 1
            && !fock_map_changed_this_iter) {
            const double actual_decrease = E_prev - E_total;
            trah_trust_radius = update_trust_radius(
                trah_trust_radius, actual_decrease, trah_predicted_decrease,
                trah_kappa_norm,   // ‖κ‖ of the previous TRAH step
                opts.trah_opts);
        }
        trah_active_prev = fock_map_changed_this_iter
            ? false : in_trah_phase;

        if (opts.dynamic_damping && !fock_map_changed_this_iter) {
            current_damping = update_dynamic_damping(
                current_damping, E_total, E_prev, have_prev_E,
                opts.dynamic_damping_min, opts.dynamic_damping_max);
        }
        have_prev_E = !fock_map_changed_this_iter;

        E_prev = E_total;
    }
    result.converged = false;
    return result;
}

// ============================================================================
// Stability-escape helpers (mirror uhf.cpp)
// ============================================================================

struct UksDensityEnergy {
    double energy = 0.0;
    int xc_batch_workers = 1;
};

// Complete KS energy of arbitrary per-spin densities for the stability
// line search. Bauernschmitt and Ahlrichs, J. Chem. Phys. 104, 9047 (1996),
// doi:10.1063/1.471637, Eqs. (2)-(6): the variational surface contains the
// one-electron, Coulomb, and *full* E_xc terms. A trial value without E_xc
// cannot be compared with UKSResult::energy and used to decide whether an
// unstable-mode rotation is downhill.
UksDensityEnergy uks_energy_of_densities(
    const Eigen::MatrixXd& Hcore,
    double E_nuc,
    const JKBuilder& jk,
    const Functional& functional,
    const BasisSet& basis,
    const Grid& grid,
    const Grid* vv10_grid,
    const UksXcFunctionalPool& worker_functionals,
    const Eigen::MatrixXd& Da,
    const Eigen::MatrixXd& Db,
    double alpha_hf) {
    const Eigen::MatrixXd D_tot = Da + Db;
    const Eigen::MatrixXd J = jk.build_J(D_tot);
    const double e_core = (D_tot.array() * Hcore.array()).sum();
    const double e_J = 0.5 * (D_tot.array() * J.array()).sum();
    double e_K = 0.0;
    if (alpha_hf != 0.0) {
        const Eigen::MatrixXd Ka = jk.build_K(Da);
        const Eigen::MatrixXd Kb = jk.build_K(Db);
        e_K = -0.5 * alpha_hf * (
            (Da.array() * Ka.array()).sum()
          + (Db.array() * Kb.array()).sum());
    }
    const UksXc xc = build_uks_xc(
        functional, basis, grid, Da, Db, vv10_grid,
        worker_functionals);
    return {
        e_core + e_J + e_K + xc.energy + E_nuc,
        xc.batch_workers,
    };
}

// Rotate one spin's occupied space along the occ-vir generator.
// Same as uhf.cpp:rotated_occupied.
Eigen::MatrixXd rotated_occupied_uks(const Eigen::MatrixXd& C,
                                     const Eigen::MatrixXd& kappa_ov,
                                     int n_occ,
                                     double theta) {
    const Eigen::Index n_kept = C.cols();
    if (n_occ <= 0) return Eigen::MatrixXd(C.rows(), 0);
    if (kappa_ov.size() == 0) return C.leftCols(n_occ);
    const Eigen::Index n_vir = n_kept - n_occ;
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_kept, n_kept);
    K.bottomLeftCorner(n_vir, n_occ) = theta * kappa_ov;
    K.topRightCorner(n_occ, n_vir) = -theta * kappa_ov.transpose();
    // Use the shared, scaling-and-squaring skew exponential. Diagonalising
    // K^2 here is numerically unsafe: rank-deficient occ-vir generators have
    // exact null modes whose rounded eigenvalues can be slightly positive,
    // making sqrt(-mu) NaN for any basis larger than the two-orbital H2 case.
    return (C * expm_skew(K)).leftCols(n_occ);
}

// ============================================================================
// run_uks_scf_with_jk — public entry point with stability analysis
// ============================================================================
UKSResult run_uks_scf_with_jk(const BasisSet& basis,
                              int n_alpha,
                              int n_beta,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const JKBuilder& jk,
                              const Grid& grid,
                              const UKSOptions& opts,
                              const Eigen::MatrixXd& init_alpha,
                              const Eigen::MatrixXd& init_beta,
                              const Grid* vv10_grid,
                              const Molecule* guess_molecule,
                              const GuessSelection* prepared_guess) {
    validate_scf_max_iter(opts.max_iter, "run_uks_scf_with_jk");
    if (opts.spinlock_mode == SpinlockMode::SPIN_SCHEDULE
        && opts.spinlock_iterations > 0 && opts.max_iter > 0) {
        throw std::invalid_argument(
            "run_uks_scf_with_jk does not implement SPIN_SCHEDULE phases; "
            "use run_uks for a locked/released population schedule");
    }
    auto constructed = prepare_open_guess(
        guess_molecule, basis, n_alpha, n_beta, opts.initial_guess, S, Hcore,
        jk, init_alpha, init_beta, opts.read_density_alpha, opts.read_density_beta,
        opts.atomic_spins, opts.linear_dep_threshold, true, prepared_guess, molecular_guess_ecp_context(opts));

    UKSResult result = run_uks_scf_once(
        basis, n_alpha, n_beta, S, Hcore, E_nuc, jk, grid, opts,
        constructed.alpha, constructed.beta, vv10_grid);
    result.guess_selection = constructed.selection;
    if (opts.orbital_optimizer == "opentrustregion") return result;
    const int n_iter_before_stability = result.n_iter;
    result.n_iter_before_stability = n_iter_before_stability;

    // Post-convergence stability-phase timing (issue #205). On large
    // open-shell jobs this phase measured 0.8-3.0x the wall of the entire
    // SCF loop it certifies, and nothing recorded it: the output layer
    // divided the whole driver wall by n_iter, so "SCF avg. per iteration"
    // silently absorbed it and every wall planned from a pre-analysis run
    // under-sized by 2-4x. Wall comes from steady_clock; CPU from
    // std::clock(), which on POSIX accumulates every OpenMP thread -- the
    // same convention as the Python PerfScope's time.process_time(), so
    // the two are directly comparable in the perf log.
    const auto t_stability_wall0 = std::chrono::steady_clock::now();
    const std::clock_t t_stability_cpu0 = std::clock();
    const auto charge_stability_phase = [&](UKSResult& charged) {
        charged.stability_wall_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_stability_wall0).count();
        charged.stability_cpu_s =
            static_cast<double>(std::clock() - t_stability_cpu0)
            / static_cast<double>(CLOCKS_PER_SEC);
    };

    // ---- Internal stability analysis + corrective restart ----------------
    if (!opts.stability_check || !result.converged) {
        return result;
    }

    // Build the XC kernel infrastructure (needed for Hessian matvec).
    // Same pattern as the Newton/TRAH kernel builder in the SCF setup.
    Functional functional(opts.functional, /*spin=*/2);
    const auto unsupported_stability = [&](bool unsupported,
                                           const char* missing_response,
                                           const char* opt_out) {
        if (!unsupported) return false;
        if (opts.stability_check_explicit) {
            throw std::runtime_error(
                std::string("UKS internal stability analysis was explicitly ")
                + "requested, but " + missing_response
                + " is not yet plumbed. Set stability_check=False to run "
                + opt_out + " without a stability verdict.");
        }
        // The default-on check must not manufacture a verdict from an
        // incomplete orbital Hessian. Preserve the converged first-order
        // result and leave stability_checked=false.
        return true;
    };

    if (unsupported_stability(
            !opts.dft_plus_u_sites.empty(),
            "the DFT+U response",
            "the supported first-order DFT+U SCF")) {
        return result;
    }
    if (functional.kind() == XCKind::MGGA) {
        if (unsupported_stability(
                true,
                "tau-dependent polarised fxc",
                "the supported first-order meta-GGA SCF")) {
            return result;
        }
    }
    if (unsupported_stability(
            functional.is_range_separated(),
            "the range-separated exact-exchange response",
            "the supported first-order range-separated-hybrid SCF")) {
        return result;
    }
    if (unsupported_stability(
            functional.needs_vv10(),
            "the VV10 nonlocal correlation response",
            "the supported first-order VV10 SCF")) {
        return result;
    }
    const double alpha_hf = functional.hf_exchange_fraction();
    if (unsupported_stability(
            alpha_hf != 0.0 && jk.has_post_scf_exchange_correction(),
            "the COSX one-centre exchange correction response and a "
            "consistent corrected-orbital energy surface",
            "the supported first-order hybrid COSX SCF")) {
        return result;
    }
    const UksXcFunctionalPool line_search_functionals =
        make_uks_xc_functional_pool(opts.functional, grid.points.rows());
    // Memory-bounded kernel: this analysis runs by default after every
    // converged UKS SCF, so it must not cache dense whole-grid
    // (n_pts × n_bf) AO tables the way the opt-in Newton/TRAH builders
    // do — that cost ~15 concurrent dense extents (+4.1 GB on a 261-bf
    // / 1.3e5-point witness) invisible to the memory preflight. The
    // batched builder streams kMolecularXcGridBatchSize-point slices,
    // matching the SCF loop's own XC quadrature.
    auto make_xc_kernel = [&](const Eigen::MatrixXd& density_alpha,
                              const Eigen::MatrixXd& density_beta) {
        return make_batched_polarised_xc_kernel_builder(
            functional, grid, basis, density_alpha, density_beta,
            kMolecularXcKernelMaxWorkers);
    };

    UHFStabilityOptions sopts;
    sopts.max_iter = opts.stability_davidson_max_iter;
    // A residual looser than the instability threshold cannot certify the
    // sign of the reported eigenvalue. This only selects the UKS caller's
    // requested accuracy; the shared Davidson implementation and the UHF
    // path (including issue #398) are unchanged.
    sopts.residual_tol = std::min(1e-5, 0.1 * opts.stability_tol);

    // Deliberate state-targeting: report but don't escape.
    const bool deliberate_state_targeting =
        (opts.spinlock_mode != SpinlockMode::OFF
         && opts.spinlock_iterations > 0)
        || !opts.atomic_spins.empty();
    if (deliberate_state_targeting) {
        auto xc_k = make_xc_kernel(result.density_alpha, result.density_beta);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc_k->batch_workers_used());
        const auto stab = uhf_internal_stability_lowest(
            result.mo_coeffs_alpha, result.mo_coeffs_beta,
            result.mo_energies_alpha, result.mo_energies_beta,
            n_alpha, n_beta, jk, sopts, xc_k.get(), alpha_hf);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc_k->batch_workers_used());
        // The Hessian builder retains only O(n_grid) scalar response state,
        // but it need not overlap the complete-XC line search or a restart
        // SCF. Release it before either phase so the #18 preflight remains a
        // max of bounded phases rather than their sum.
        xc_k.reset();
        result.stability_checked = true;
        result.stability_analysis_converged = stab.converged;
        result.stability_eigenvalue = stab.lowest_eigenvalue;
        result.internal_instability =
            stab.converged
            && stab.lowest_eigenvalue < -opts.stability_tol;
        charge_stability_phase(result);
        return result;
    }

    for (int restart = 0; ; ++restart) {
        auto xc_k = make_xc_kernel(result.density_alpha, result.density_beta);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc_k->batch_workers_used());
        const auto stab = uhf_internal_stability_lowest(
            result.mo_coeffs_alpha, result.mo_coeffs_beta,
            result.mo_energies_alpha, result.mo_energies_beta,
            n_alpha, n_beta, jk, sopts, xc_k.get(), alpha_hf);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc_k->batch_workers_used());
        result.stability_checked = true;
        result.stability_analysis_converged = stab.converged;
        result.stability_eigenvalue = stab.lowest_eigenvalue;
        result.internal_instability =
            stab.converged
            && stab.lowest_eigenvalue < -opts.stability_tol;
        if (!stab.converged) break;
        if (!result.internal_instability) break;
        if (restart >= opts.stability_max_retries) break;
        // The response builder is no longer needed once the owned Davidson
        // result has been extracted. Do not retain its whole-grid scalar
        // state across line-search XC batches or a corrective SCF.
        xc_k.reset();

        // Search and reconverge BOTH signs of the internal-instability mode.
        // Seeger and Pople, J. Chem. Phys. 66, 3045 (1977),
        // doi:10.1063/1.434318, Section IV, Eqs. (37)-(41): unlike an
        // external instability, the two signed internal descents need not
        // be equivalent and can terminate at different minima.
        static constexpr double kThetaMagnitudes[] = {
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0,
        };
        struct EscapeSeed {
            bool found = false;
            double energy = 0.0;
            Eigen::MatrixXd C_alpha_occ;
            Eigen::MatrixXd C_beta_occ;
        };
        auto line_search = [&](double sign) {
            EscapeSeed seed;
            seed.energy = result.energy;
            for (const double magnitude : kThetaMagnitudes) {
                const double theta = sign * magnitude;
                const Eigen::MatrixXd Ca_occ = rotated_occupied_uks(
                    result.mo_coeffs_alpha, stab.kappa_alpha,
                    n_alpha, theta);
                const Eigen::MatrixXd Cb_occ = rotated_occupied_uks(
                    result.mo_coeffs_beta, stab.kappa_beta,
                    n_beta, theta);
                const Eigen::MatrixXd Da = Ca_occ * Ca_occ.transpose();
                const Eigen::MatrixXd Db = Cb_occ * Cb_occ.transpose();
                const UksDensityEnergy trial = uks_energy_of_densities(
                    Hcore, E_nuc, jk, functional, basis, grid, vv10_grid,
                    line_search_functionals, Da, Db, alpha_hf);
                result.xc_batch_workers_used = std::max(
                    result.xc_batch_workers_used,
                    trial.xc_batch_workers);
                if (trial.energy < seed.energy) {
                    seed.found = true;
                    seed.energy = trial.energy;
                    seed.C_alpha_occ = Ca_occ;
                    seed.C_beta_occ = Cb_occ;
                }
            }
            return seed;
        };

        // Fixed-reference MOM protects sparse occupation patterns during the
        // first drive. If that hold over-constrains a collective mode, retry
        // the same variationally downhill seed with ordinary Aufbau filling.
        constexpr int kAnchorHoldIters = 15;
        constexpr int kMaxRestartChains = 5;
        auto reconverge = [&](const EscapeSeed& seed, int anchor_iters) {
            const Eigen::MatrixXd Da =
                seed.C_alpha_occ * seed.C_alpha_occ.transpose();
            const Eigen::MatrixXd Db =
                seed.C_beta_occ * seed.C_beta_occ.transpose();
            const Eigen::MatrixXd* anchor_alpha =
                anchor_iters > 0 ? &seed.C_alpha_occ : nullptr;
            const Eigen::MatrixXd* anchor_beta =
                anchor_iters > 0 ? &seed.C_beta_occ : nullptr;
            UKSResult candidate = run_uks_scf_once(
                basis, n_alpha, n_beta, S, Hcore, E_nuc, jk, grid, opts,
                Da, Db, vv10_grid,
                anchor_alpha, anchor_beta, anchor_iters);
            double chain_energy = seed.energy;
            for (int chain = 0;
                 !candidate.converged && chain < kMaxRestartChains; ++chain) {
                if (candidate.energy >= chain_energy - 1e-9) break;
                chain_energy = candidate.energy;
                candidate = run_uks_scf_once(
                    basis, n_alpha, n_beta, S, Hcore, E_nuc, jk, grid, opts,
                    candidate.density_alpha, candidate.density_beta,
                    vv10_grid);
            }
            return candidate;
        };

        bool found_reconverged_descent = false;
        UKSResult retried;
        for (const double sign : {-1.0, 1.0}) {
            const EscapeSeed seed = line_search(sign);
            if (!seed.found) continue;
            for (const int anchor_iters : {kAnchorHoldIters, 0}) {
                UKSResult candidate = reconverge(seed, anchor_iters);
                candidate.xc_batch_workers_used = std::max(
                    candidate.xc_batch_workers_used,
                    result.xc_batch_workers_used);
                VIBEQC_DIAG(
                    "uks-stability", vibeqc::DiagLevel::DEBUG,
                    "event=escape_candidate restart=%d sign=%+.0f "
                    "anchor_iters=%d converged=%d seed_energy=%.16e "
                    "energy=%.16e",
                    restart, sign, anchor_iters,
                    candidate.converged ? 1 : 0,
                    seed.energy, candidate.energy);
                if (candidate.converged
                    && candidate.energy < result.energy - 1e-10) {
                    if (!found_reconverged_descent
                        || candidate.energy < retried.energy) {
                        found_reconverged_descent = true;
                        retried = std::move(candidate);
                    }
                    break;
                }
            }
        }
        if (!found_reconverged_descent) break;
        // ``result`` accumulated the maximum team observed by every signed
        // trial-energy build, including a branch that was not selected.
        retried.xc_batch_workers_used = std::max(
            retried.xc_batch_workers_used,
            result.xc_batch_workers_used);
        retried.stability_checked = true;
        retried.stability_analysis_converged = true;
        retried.stability_eigenvalue = stab.lowest_eigenvalue;
        retried.n_stability_restarts = result.n_stability_restarts + 1;
        retried.n_iter_before_stability = n_iter_before_stability;
        result = retried;
        result.guess_selection = constructed.selection;
        result.guess_selection->transport = InitialGuess::READ;
    }
    charge_stability_phase(result);
    return result;
}

}  // namespace vibeqc
