#include "vibeqc/rks.hpp"
#include "vibeqc/orbital_scf.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/diis.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/dynamic_damping.hpp"
#include "vibeqc/kdiis.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_staged.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/jk_builder.hpp"
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

Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int nocc) {
    if (nocc <= 0) return Eigen::MatrixXd::Zero(C.rows(), C.rows());
    return 2.0 * C.leftCols(nocc) * C.leftCols(nocc).transpose();
}

struct XcContribution {
    Eigen::MatrixXd V;       // V_xc matrix, n_bf × n_bf
    double energy = 0.0;     // E_xc
    int batch_workers = 1;
};

using XcFunctionalPool = std::vector<std::unique_ptr<Functional>>;

XcFunctionalPool make_xc_functional_pool(const std::string& name,
                                          int spin,
                                          Eigen::Index n_points) {
    const std::size_t n_batches = static_cast<std::size_t>(
        (n_points + kMolecularXcGridBatchSize - 1)
        / kMolecularXcGridBatchSize);
    const int workers = omp_workers_for(
        n_batches, kMolecularXcGridMaxWorkers);
    XcFunctionalPool pool;
    if (workers <= 1) return pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        pool.push_back(std::make_unique<Functional>(name, spin));
    }
    return pool;
}

Grid xc_grid_slice(const Grid& grid, Eigen::Index start, Eigen::Index count) {
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

AOValues evaluate_xc_ao(const BasisSet& basis,
                        const Eigen::MatrixX3d& points,
                        bool need_gradient) {
    if (need_gradient) {
        return evaluate_ao_with_gradient(basis, points);
    }
    AOValues ao;
    ao.values = evaluate_ao(basis, points);
    return ao;
}

struct RksDensityFields {
    Eigen::VectorXd rho;
    Eigen::VectorXd gx;
    Eigen::VectorXd gy;
    Eigen::VectorXd gz;
    Eigen::VectorXd sigma;
    Eigen::VectorXd tau;
};

RksDensityFields build_rks_density_fields(
    const Functional& func,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D,
    bool build_tau = true) {
    const auto n_pts = chi.rows();
    const Eigen::MatrixXd chiD = chi * D;
    const XCKind kind = func.kind();
    const bool is_mgga = (kind == XCKind::MGGA);
    const bool need_grad = (kind == XCKind::GGA) || is_mgga
                        || func.needs_vv10();

    RksDensityFields fields;
    fields.rho.resize(n_pts);
    if (need_grad) {
        fields.gx.resize(n_pts);
        fields.gy.resize(n_pts);
        fields.gz.resize(n_pts);
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            const auto cD = chiD.row(g);
            fields.rho(g) = cD.dot(chi.row(g));
            fields.gx(g) = 2.0 * cD.dot(dchi[0].row(g));
            fields.gy(g) = 2.0 * cD.dot(dchi[1].row(g));
            fields.gz(g) = 2.0 * cD.dot(dchi[2].row(g));
        }
        fields.sigma = fields.gx.array().square()
                     + fields.gy.array().square()
                     + fields.gz.array().square();
    } else {
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            fields.rho(g) = chiD.row(g).dot(chi.row(g));
        }
        fields.sigma.resize(0);
    }

    if (is_mgga && build_tau) {
        fields.tau = Eigen::VectorXd::Zero(n_pts);
        std::array<Eigen::MatrixXd, 3> dchiD;
        for (int c = 0; c < 3; ++c) {
            dchiD[c] = dchi[c] * D;
        }
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            double value = 0.0;
            for (int c = 0; c < 3; ++c) {
                value += dchiD[c].row(g).dot(dchi[c].row(g));
            }
            fields.tau(g) = 0.5 * value;
        }
    }
    return fields;
}

// Full-grid nonlocal-XC path.  The provider returns derivatives of the
// integrated scalar energy, so projection deliberately contains no second
// quadrature-weight factor.  For a restricted density the two model spin
// channels are rho/2, grad(rho)/2, tau/2; the 1/2 chain-rule factor is
// applied when their adjoints are combined below.
XcContribution build_external_xc_rks(
    const Functional& func,
    const BasisSet& basis,
    const Grid& grid,
    const Eigen::MatrixXd& D) {
    const Eigen::Index n_pts = grid.points.rows();
    const Eigen::Index n_bf = static_cast<Eigen::Index>(basis.nbasis());
    XcContribution out{Eigen::MatrixXd::Zero(n_bf, n_bf), 0.0, 1};
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

    // Pass 1: assemble the complete feature vectors in memory-bounded AO
    // slices.  The neural/nonlocal model itself is called only after every
    // atom block is present.
    for (Eigen::Index g0 = 0; g0 < n_pts;
         g0 += kMolecularXcGridBatchSize) {
        const Eigen::Index count = std::min<Eigen::Index>(
            kMolecularXcGridBatchSize, n_pts - g0);
        const Grid slice = xc_grid_slice(grid, g0, count);
        AOValues ao = evaluate_xc_ao(basis, slice.points, true);
        const RksDensityFields fields = build_rks_density_fields(
            func, ao.values, ao.gradients, D, /*build_tau=*/true);
        input.rho_alpha.segment(g0, count) = 0.5 * fields.rho;
        input.rho_beta.segment(g0, count) = 0.5 * fields.rho;
        input.grad_alpha.col(0).segment(g0, count) = 0.5 * fields.gx;
        input.grad_alpha.col(1).segment(g0, count) = 0.5 * fields.gy;
        input.grad_alpha.col(2).segment(g0, count) = 0.5 * fields.gz;
        input.grad_beta.middleRows(g0, count) =
            input.grad_alpha.middleRows(g0, count);
        input.tau_alpha.segment(g0, count) = 0.5 * fields.tau;
        input.tau_beta.segment(g0, count) = 0.5 * fields.tau;
    }

    const ExternalXCEvaluation ev = func.eval_external(input);
    out.energy = ev.energy;
    const Eigen::VectorXd q_rho =
        0.5 * (ev.v_rho_alpha + ev.v_rho_beta);
    const Eigen::MatrixX3d q_grad =
        0.5 * (ev.v_grad_alpha + ev.v_grad_beta);
    const Eigen::VectorXd q_tau =
        0.5 * (ev.v_tau_alpha + ev.v_tau_beta);

    // Pass 2: project the full-energy feature adjoints back to the AO density
    // matrix without retaining O(G*N_AO) tables.
    for (Eigen::Index g0 = 0; g0 < n_pts;
         g0 += kMolecularXcGridBatchSize) {
        const Eigen::Index count = std::min<Eigen::Index>(
            kMolecularXcGridBatchSize, n_pts - g0);
        const Eigen::MatrixX3d points = grid.points.middleRows(g0, count);
        AOValues ao = evaluate_xc_ao(basis, points, true);
        const Eigen::VectorXd qr = q_rho.segment(g0, count);
        out.V.noalias() +=
            ao.values.transpose() * qr.asDiagonal() * ao.values;
        for (int c = 0; c < 3; ++c) {
            const Eigen::VectorXd qg = q_grad.col(c).segment(g0, count);
            const Eigen::MatrixXd G =
                ao.gradients[c].transpose() * qg.asDiagonal() * ao.values;
            out.V.noalias() += G + G.transpose();
            const Eigen::VectorXd qt =
                0.5 * q_tau.segment(g0, count);
            out.V.noalias() += ao.gradients[c].transpose()
                             * qt.asDiagonal() * ao.gradients[c];
        }
    }
    return out;
}

// Build V_xc (matrix) and E_xc (scalar) for RKS from the grid data.
//
// ρ(g)     = Σ_{μν} D_{μν} χ_μ(g) χ_ν(g)
// ∇ρ(g)   = 2 Σ_{μν} D_{μν} χ_μ(g) ∇χ_ν(g)
// σ(g)    = |∇ρ(g)|²
// τ(g)    = ½ Σ_{μν} D_{μν} Σ_c ∂_c χ_μ(g) ∂_c χ_ν(g)   (MGGA only)
// exc(g)  = ρ(g) · ε_xc(ρ, σ[, τ])  (from libxc; returned as the integrand)
//
// V_xc_{μν} = Σ_g w_g [v_ρ(g) χ_μ(g) χ_ν(g)
//                      + 2 v_σ(g) · ( ∇ρ(g)·∇χ_μ(g) χ_ν(g)
//                                    + χ_μ(g) ∇ρ(g)·∇χ_ν(g) )
//                      + ½ v_τ(g) Σ_c ∂_c χ_μ(g) ∂_c χ_ν(g) ]   (MGGA only)
XcContribution build_xc_slice(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D,
    const Eigen::VectorXd* vv10_v_rho = nullptr,
    const Eigen::VectorXd* vv10_v_sigma = nullptr) {
    const auto n_pts = chi.rows();
    const auto n_bf = chi.cols();

    const XCKind kind = func.kind();
    const bool is_gga  = (kind == XCKind::GGA);
    const bool is_mgga = (kind == XCKind::MGGA);
    const bool need_grad = is_gga || is_mgga || func.needs_vv10();
    RksDensityFields fields =
        build_rks_density_fields(func, chi, dchi, D);

    Eigen::VectorXd exc, v_rho, v_sigma, v_tau;
    if (is_mgga) {
        func.eval_unpolarised_mgga(fields.rho, fields.sigma, fields.tau,
                                   exc, v_rho, v_sigma, v_tau);
    } else {
        func.eval_unpolarised(fields.rho, fields.sigma,
                              exc, v_rho, v_sigma);
    }

    // E_xc = Σ_g w_g · exc[g] (exc is ρ·ε_xc).
    double E_xc = grid.weights.dot(exc);

    // VV10 itself is a full-grid double sum, not a pointwise functional.
    // The batched wrapper evaluates it once and passes the matching potential
    // slices here for AO projection.  Calling compute_vv10 per slice would
    // drop every cross-batch pair from Vydrov-Van Voorhis Eq. (13).
    //
    // When the VV10 potential is evaluated on a separate coarse grid
    // (GridOptions::vv10_grid_factor > 1), the caller projects it
    // independently and passes nullptr here — this path then skips VV10.
    if (func.needs_vv10() && vv10_v_rho != nullptr && vv10_v_sigma != nullptr) {
        if (vv10_v_rho->size() != n_pts || vv10_v_sigma->size() != n_pts) {
            throw std::invalid_argument(
                "build_xc_slice: VV10 potential slice has the wrong grid length");
        }
        v_rho += *vv10_v_rho;
        v_sigma += *vv10_v_sigma;
    }

    // LDA piece of V:   V_{μν} += Σ_g w_g v_ρ(g) χ_μ(g) χ_ν(g)
    //                           = χᵀ · diag(w · v_ρ) · χ
    Eigen::VectorXd w_vrho = grid.weights.array() * v_rho.array();
    Eigen::MatrixXd V = chi.transpose() * w_vrho.asDiagonal() * chi;

    if (need_grad) {
        // GGA piece: V_{μν} += Σ_g w_g · 2 v_σ · [ (∇ρ·∇χ_μ) χ_ν + χ_μ (∇ρ·∇χ_ν) ]
        // Define F_μ(g) = Σ_c grho_c(g) ∂_c χ_μ(g).
        // Then V_{μν} += Σ_g u(g) [ F_μ(g) χ_ν(g) + χ_μ(g) F_ν(g) ]
        //              = F^T · diag(u) · χ + χ^T · diag(u) · F
        // Build F in a single OMP-parallel row pass — the pre-fix
        // expression ``grho_x.asDiagonal()*dchi[0] + ...`` materialised
        // three (n_pts × n_bf) intermediates before summing.
        Eigen::MatrixXd F(n_pts, n_bf);
        #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
        for (Eigen::Index g = 0; g < n_pts; ++g) {
            F.row(g) = fields.gx(g) * dchi[0].row(g)
                     + fields.gy(g) * dchi[1].row(g)
                     + fields.gz(g) * dchi[2].row(g);
        }
        Eigen::VectorXd u = 2.0 * grid.weights.array() * v_sigma.array();
        Eigen::MatrixXd Fu = F.transpose() * u.asDiagonal() * chi;
        V += Fu + Fu.transpose();
    }

    if (is_mgga) {
        // MGGA τ piece:  V^τ_{μν} = ½ Σ_g w_g v_τ(g) Σ_c ∂_c χ_μ(g) ∂_c χ_ν(g)
        //                         = ½ Σ_c (∂_c χ)ᵀ · diag(w · v_τ) · (∂_c χ)
        // Naturally symmetric (∂_c χ_μ ∂_c χ_ν symmetric under μ↔ν).
        const Eigen::VectorXd w_vtau = 0.5 * grid.weights.array() * v_tau.array();
        for (int c = 0; c < 3; ++c) {
            V += dchi[c].transpose() * w_vtau.asDiagonal() * dchi[c];
        }
    }
    return {V, E_xc, 1};
}

// Evaluate molecular XC in fixed-size slices of the already-constructed
// Becke grid.  Becke's quadrature is a weighted sum over independent points,
// so disjoint batching preserves every point, weight, density invariant, and
// Fock term; it changes only floating-point association.
//
// VV10 (when needed) is handled separately from the semilocal XC:
//   - If ``vv10_grid`` is provided (non-null), VV10 is evaluated on that
//     coarser grid and its V_xc contribution is projected independently.
//     This is the performance path — the VV10 double sum is O(N²) and the
//     coarse grid (factor 3 in each dimension → ~9× fewer points → ~81×
//     fewer pairs) makes VV10-paired functionals production-usable.
//   - If ``vv10_grid`` is null, VV10 is evaluated on the full XC grid
//     (legacy behaviour, preserved for bitwise reproducibility and for
//     callers that don't build a separate VV10 grid).
XcContribution build_xc(const Functional& func,
                        const BasisSet& basis,
                        const Grid& grid,
                        const Eigen::MatrixXd& D,
                        const Grid* vv10_grid,
                        const XcFunctionalPool& worker_functionals) {
    const Eigen::Index n_pts = grid.points.rows();
    const Eigen::Index n_bf = static_cast<Eigen::Index>(basis.nbasis());
    XcContribution total{Eigen::MatrixXd::Zero(n_bf, n_bf), 0.0, 1};
    if (n_pts == 0) return total;
    if (func.is_external()) {
        return build_external_xc_rks(func, basis, grid, D);
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
            // Gather total density rho and sigma on the VV10 grid.
            Eigen::VectorXd rho(vv10_n);
            Eigen::VectorXd sigma(vv10_n);
            for (Eigen::Index g0 = 0; g0 < vv10_n;
                 g0 += kMolecularXcGridBatchSize) {
                const Eigen::Index count = std::min<Eigen::Index>(
                    kMolecularXcGridBatchSize, vv10_n - g0);
                const Grid slice = xc_grid_slice(vv10_g, g0, count);
                AOValues ao = evaluate_xc_ao(basis, slice.points, true);
                const RksDensityFields fields = build_rks_density_fields(
                    func, ao.values, ao.gradients, D, /*build_tau=*/false);
                rho.segment(g0, count) = fields.rho;
                sigma.segment(g0, count) = fields.sigma;
            }
            vv10 = compute_vv10(vv10_g.points, vv10_g.weights, rho, sigma,
                                func.vv10_b(), func.vv10_C());
            total.energy += vv10.energy;

            // When VV10 is on a separate coarse grid, project its V_xc
            // contribution independently.  The projection formulas are the
            // same as in build_xc_slice (rho term + sigma/GGA term) but
            // without the libxc semilocal pieces.
            if (vv10_on_coarse) {
                for (Eigen::Index g0 = 0; g0 < vv10_n;
                     g0 += kMolecularXcGridBatchSize) {
                    const Eigen::Index count = std::min<Eigen::Index>(
                        kMolecularXcGridBatchSize, vv10_n - g0);
                    const Grid slice = xc_grid_slice(vv10_g, g0, count);
                    AOValues ao = evaluate_xc_ao(basis, slice.points, true);
                    const RksDensityFields fields =
                        build_rks_density_fields(
                            func, ao.values, ao.gradients, D,
                            /*build_tau=*/false);

                    const Eigen::VectorXd vr = vv10.v_rho.segment(g0, count);
                    const Eigen::VectorXd vs = vv10.v_sigma.segment(g0, count);

                    // rho term:  V += χᵀ · diag(w · v_rho) · χ
                    Eigen::VectorXd w_vrho =
                        slice.weights.array() * vr.array();
                    total.V.noalias() +=
                        ao.values.transpose() * w_vrho.asDiagonal()
                        * ao.values;

                    // sigma / GGA term:
                    //   F_μ(g) = Σ_c ∇_c ρ(g) · ∂_c χ_μ(g)
                    //   V += Fᵀ · diag(u) · χ + χᵀ · diag(u) · F
                    //   with u = 2 · w · v_sigma
                    Eigen::MatrixXd F(count, n_bf);
                    #pragma omp parallel for schedule(static) if(!omp_in_parallel_region())
                    for (Eigen::Index g = 0; g < count; ++g) {
                        F.row(g) =
                            fields.gx(g) * ao.gradients[0].row(g)
                          + fields.gy(g) * ao.gradients[1].row(g)
                          + fields.gz(g) * ao.gradients[2].row(g);
                    }
                    Eigen::VectorXd u =
                        2.0 * slice.weights.array() * vs.array();
                    Eigen::MatrixXd Fu =
                        F.transpose() * u.asDiagonal() * ao.values;
                    total.V.noalias() += Fu + Fu.transpose();
                }
            }
        }
    }

    // ---- Semilocal XC on the full grid (ordered parallel batches) ----
    // Every batch owns its Functional because libxc evaluation mutates the
    // underlying xc_func_type workspace.  Per-batch matrix products keep the
    // same full dimensions and linked-BLAS path as the serial implementation;
    // only independent grid slices run concurrently.  Results are accumulated
    // below in their original grid order so the scientific reduction is
    // unchanged.  Waves bound live scratch to at most the exported worker cap.
    const auto evaluate_batch = [&](const Functional& batch_func,
                                    Eigen::Index g0) {
        const Eigen::Index count = std::min<Eigen::Index>(
            kMolecularXcGridBatchSize, n_pts - g0);
        const Grid slice = xc_grid_slice(grid, g0, count);
        AOValues ao = evaluate_xc_ao(basis, slice.points, need_gradient);
        // VV10 potentials are only passed to build_xc_slice when they live
        // on the same grid (legacy path).  When VV10 was projected from a
        // coarse grid above, we pass nullptr so build_xc_slice skips VV10.
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
        return build_xc_slice(
            batch_func, slice, ao.values, ao.gradients, D,
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
            XcContribution part = evaluate_batch(func, g0);
            total.V.noalias() += part.V;
            total.energy += part.energy;
        }
        return total;
    }

    if (worker_functionals.size() < static_cast<std::size_t>(workers)) {
        throw std::logic_error(
            "build_xc: worker Functional pool is smaller than the "
            "resolved XC batch team");
    }
    for (std::size_t wave = 0; wave < n_batches;
         wave += static_cast<std::size_t>(workers)) {
        const int wave_size = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(workers), n_batches - wave));
        std::vector<XcContribution> parts(
            static_cast<std::size_t>(wave_size));
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
            total.V.noalias() += part.V;
            total.energy += part.energy;
        }
    }
    return total;
}

}  // namespace

OrbitalXCFunction make_orbital_rks_xc(const BasisSet& basis, const Grid& grid, const std::string& name) {
    auto f = std::make_shared<Functional>(name, 1);
    validate_orbital_xc(*f);
    auto ao = std::make_shared<AOValues>(evaluate_ao_with_gradient(basis, grid.points));
    auto pool = std::make_shared<XcFunctionalPool>(
        make_xc_functional_pool(name, 1, grid.points.rows()));
    return [basis, grid, f, ao, pool](const Eigen::MatrixXd& da, const Eigen::MatrixXd& db, bool response) {
        OrbitalXC out;
        (void)db;
        const auto value = build_xc(*f, basis, grid, da, nullptr, *pool);
        out.energy = value.energy; out.alpha = value.V;
        if (response) out.restricted_kernel = make_unpolarised_xc_kernel_builder(
            *f, grid, ao->values, ao->gradients, da);
        return out;
    };
}

RKSResult run_rks(const Molecule& mol,
                  const BasisSet& basis,
                  const RKSOptions& opts) {
    validate_orbital_optimizer(opts);
    validate_initial_guess(opts.initial_guess);
    validate_guess_ecp(opts.initial_guess, molecular_guess_ecp_context(opts), &mol);
    validate_scf_max_iter(opts.max_iter, "run_rks");
    if (mol.n_electrons() % 2 != 0) {
        throw std::invalid_argument(
            "run_rks: closed-shell DFT requires even electron count, got "
            + std::to_string(mol.n_electrons()));
    }
    if (mol.multiplicity() != 1) {
        throw std::invalid_argument(
            "run_rks: RKS requires multiplicity = 1, got "
            + std::to_string(mol.multiplicity()));
    }
    validate_fraction_01("run_rks: damping", opts.damping);
    validate_fraction_01("run_rks: fock_mixing", opts.fock_mixing);

    const auto ecp_input = validate_molecular_ecp_dispatch(
        opts.ecp_centers, opts.ecp_library, opts.ecp_primitive_blocks,
        opts.ecp_primitive_centers, opts.ecp_effective_charges,
        opts.ecp_total_ncore, "run_rks");

    // ---- One-electron integrals ----
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
            "run_rks: ECP cores remove more electrons ("
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
            RKSResult result) {
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

    // ---- XC grid + (when needed) COSX grid ---------------------------------
    Grid grid = build_grid(mol, opts.grid);

    // VV10 nonlocal correlation grid (when the functional needs it).
    // Built at a coarser resolution than the semilocal XC grid because
    // the VV10 double integral is smooth (1/R^6 kernel) and converges
    // rapidly with grid density.  A separate grid avoids the O(N^2)
    // pairwise evaluation on the full ~46k-points/atom XC grid.
    std::unique_ptr<Grid> vv10_grid_owner;
    const Functional functional(opts.functional);
    if (functional.needs_vv10() && opts.grid.vv10_grid_factor > 0.0
        && opts.grid.vv10_grid_factor != 1.0) {
        GridOptions vv10_opts = opts.grid;
        // Scale radial and angular resolution down by the factor.
        const double f = opts.grid.vv10_grid_factor;
        vv10_opts.n_radial = std::max(5, static_cast<int>(
            std::lround(opts.grid.n_radial / f)));
        if (vv10_opts.angular == AngularScheme::Lebedev) {
            // Step down to the next lower bundled Lebedev order (29→17,
            // 35→23, etc.).  The bundled orders are 11, 17, 23, 29, 35,
            // 41, 47, 53.
            int order = vv10_opts.lebedev_order;
            if (order > 11) {
                // Reduce by roughly factor f; at least one tier down.
                int steps = std::max(1, static_cast<int>(std::lround(
                    static_cast<double>(order - 11) / f)));
                order = std::max(11, order - steps * 6);
                // Snap to the nearest bundled order.
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
    Grid cosx_grid_built;
    CosxVariant cosx_variant_resolved = CosxVariant::FITTED;
    std::vector<GridOptions> cosx_stages;  // COSX grid progression (GridX)
    bool cosx_multistage = false;
    if (opts.cosx) {
        const int cosx_card = cosx_basis_cardinality_from_name(basis.name());
        cosx_variant_resolved = resolve_cosx_variant(
            opts.cosx_variant, opts.thresh_cosx, cosx_card,
            opts.cosx_grid_level);
        if (!cosx_use_gridx(opts.cosx_grid_level, opts.conv_tol_grad)) {
            // Legacy single COSX grid: explicit opt-out or the auto fallback
            // for a conv_tol_grad tighter than the GridX commutator floor.
            cosx_grid_built = build_grid(
                mol, resolve_cosx_grid_options(opts.cosx_grid, 0, cosx_card));
        } else {
            // GridX multi-stage progression (P2): stage-0 (coarse) grid
            // seeds the guess; finer stages are built on the fly.
            cosx_stages = cosx_grid_stages_for_level(
                resolve_cosx_grid_level(opts.cosx_grid_level, cosx_card));
            cosx_grid_built = build_grid(mol, cosx_stages.front());
            cosx_multistage = true;
        }
    }

    // Range-separated hybrids (ωB97X, …) need the erf-attenuated K
    // build, which only the direct-SCF Fock builder implements. Probe
    // the functional once to drive the JKBuilder dispatch below.
    const bool functional_is_rsh = functional.is_range_separated();

    // ---- Two-electron infrastructure (JKBuilder dispatch) ------------------
    std::unique_ptr<BasisSet> aux;
    std::unique_ptr<JKBuilder> jk;
    if (opts.density_fit) {
        if (functional_is_rsh) {
            throw std::invalid_argument(
                "run_rks: range-separated hybrids (ωB97X, …) are not yet "
                "supported with density_fit=true — the erf-attenuated "
                "exchange needs erf 3-centre integrals that the RI path "
                "does not yet build. Run with density_fit=false (the "
                "direct-SCF path handles RSH).");
        }
        if (opts.aux_basis.empty()) {
            throw std::invalid_argument(
                "run_rks: density_fit=true requires aux_basis to be set "
                "(e.g. \"def2-svp-jk\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"jk\") "
                "for autodetection.");
        }
        aux = std::make_unique<BasisSet>(mol, opts.aux_basis);
        if (opts.cosx) {
            jk = make_cosx_jk_builder(basis, *aux, cosx_grid_built,
                                      cosx_variant_resolved);
        } else {
            jk = make_df_jk_builder(basis, *aux);
        }
    } else {
        // RSH forces the direct path — only DirectJKBuilder has the
        // erf-attenuated K kernel (FourIndexJKBuilder would throw in
        // build_K_erf). resolve_scf_mode otherwise picks by basis size.
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

    auto prepared = prepare_closed_guess(
        &mol, basis, n_elec / 2, opts.initial_guess, S, Hcore, *jk,
        {}, opts.read_density, opts.linear_dep_threshold, nullptr, molecular_guess_ecp_context(opts));
    const auto& guess_D = prepared.density;

    if (cosx_multistage) {
        Eigen::MatrixXd D = guess_D;
        auto seg = [&](const JKBuilder& seg_jk,
                       const RKSOptions& so) -> RKSResult {
            RKSResult r = run_rks_scf_with_jk(basis, n_elec, S, Hcore, E_nuc,
                                              seg_jk, grid, so, D,
                                              vv10_grid, nullptr, nullptr, &mol, &prepared.selection);
            prepared.selection.transport = InitialGuess::READ;
            D = r.density;
            return r;
        };
        return with_ecp_provenance(run_cosx_staged_scf<RKSResult>(
            mol, cosx_stages, opts, *jk, seg));
    }
    return with_ecp_provenance(run_rks_scf_with_jk(
        basis, n_elec, S, Hcore, E_nuc, *jk, grid, opts, guess_D,
        vv10_grid, nullptr, nullptr, &mol, &prepared.selection));
}

RKSResult run_rks_scf_with_jk(const BasisSet& basis,
                              int n_electrons,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const JKBuilder& jk,
                              const Grid& grid,
                              const RKSOptions& opts,
                              const Eigen::MatrixXd& initial_density,
                              const Grid* vv10_grid,
                              const std::vector<Eigen::MatrixXd>* warm_fock_history,
                              const std::vector<Eigen::MatrixXd>* warm_error_history,
                              const Molecule* guess_molecule,
                              const GuessSelection* prepared_guess) {
    validate_initial_guess(opts.initial_guess);
    validate_guess_ecp(opts.initial_guess, molecular_guess_ecp_context(opts), guess_molecule);
    validate_orbital_optimizer(opts);
    validate_scf_max_iter(opts.max_iter, "run_rks_scf_with_jk");
    if (n_electrons < 0 || n_electrons % 2 != 0) {
        throw std::invalid_argument("closed-shell SCF requires a nonnegative even electron count");
    }
    auto constructed = prepare_closed_guess(
        guess_molecule, basis, n_electrons / 2, opts.initial_guess, S, Hcore,
        jk, initial_density, opts.read_density, opts.linear_dep_threshold,
        prepared_guess, molecular_guess_ecp_context(opts));


    validate_fraction_01("run_rks_scf_with_jk: damping", opts.damping);
    validate_fraction_01(
        "run_rks_scf_with_jk: fock_mixing", opts.fock_mixing);
    if (S.rows() != Hcore.rows() || S.cols() != Hcore.cols()
        || S.rows() != static_cast<Eigen::Index>(basis.nbasis())) {
        throw std::invalid_argument(
            "run_rks_scf_with_jk: S / Hcore shapes do not match "
            "basis.nbasis()");
    }
    const int nocc = n_electrons / 2;

    // Canonical orthogonalization; see rhf.cpp for the shared notes.
    const auto orth = canonical_orthogonalizer(S, opts.linear_dep_threshold);
    if (orth.n_kept == 0) {
        throw std::runtime_error(
            "RKS: AO basis has no non-null directions above the "
            "linear-dependence threshold (threshold = "
            + std::to_string(opts.linear_dep_threshold) + ")");
    }
    if (nocc > orth.n_kept) {
        throw std::runtime_error(
            "RKS: canonical orthogonalization dropped too many basis "
            "directions for this electron count (n_occ = "
            + std::to_string(nocc) + ", n_kept = "
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

    Functional functional(opts.functional);
    const XcFunctionalPool xc_functionals = make_xc_functional_pool(
        opts.functional, /*spin=*/1, grid.points.rows());
    const double alpha = functional.hf_exchange_fraction();
    // Range-separated (CAM) hybrid bookkeeping. For a *global* hybrid
    // cam_a == alpha and cam_b == 0, so the exchange assembly below
    // collapses to the plain −½·α·K path; for an RSH hybrid (ωB97X, …)
    // the effective exchange is cam_a·K + cam_b·K_erf(ω).
    const bool is_rsh = functional.is_range_separated();
    const double cam_a = functional.cam_alpha();
    const double cam_b = functional.cam_beta();
    const double rsh_omega = functional.rsh_omega();
    const bool need_k = (cam_a != 0.0 || cam_b != 0.0);

    // Newton/TRAH are opt-in and their f_xc builders still retain dense
    // whole-grid AO tables.  Build that fallback lazily only if either phase
    // is actually entered; ordinary SCF XC stays on the bounded batched path.
    // The Python estimator continues to charge the dense route whenever one
    // of these thresholds is enabled.
    std::unique_ptr<AOValues> xc_kernel_ao;
    auto make_xc_kernel = [&](const Eigen::MatrixXd& density) {
        if (!xc_kernel_ao) {
            xc_kernel_ao = std::make_unique<AOValues>(
                evaluate_ao_with_gradient(basis, grid.points));
        }
        return make_unpolarised_xc_kernel_builder(
            functional, grid, xc_kernel_ao->values,
            xc_kernel_ao->gradients, density);
    };

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fock_solver;
    auto diagonalize_fock = [&](const Eigen::MatrixXd& F)
        -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
        const Eigen::MatrixXd Fp = X.transpose() * F * X;
        if (use_davidson_here) {
            if (dav_guess_ortho.size() != 0) {
                dav_opts.guess_vectors = dav_guess_ortho;
            }
            DavidsonResult dres = davidson_solve(Fp, dav_opts);
            if (!dres.converged) {
                throw std::runtime_error(
                    "RKS: Davidson diagonalization did not converge after "
                    + std::to_string(dres.n_iter) + " iterations");
            }
            dav_guess_ortho = dres.eigenvectors;
            return {X * dres.eigenvectors, dres.eigenvalues};
        }
        fock_solver.compute(Fp);
        if (fock_solver.info() != Eigen::Success) {
            throw std::runtime_error("RKS: Fock diagonalization failed");
        }
        return {X * fock_solver.eigenvectors(), fock_solver.eigenvalues()};
    };

    // ---- Initial density (and a seed MO frame for the C1c Newton step) ----
    auto [C0, eps0] = diagonalize_fock(Hcore);
    Eigen::MatrixXd D;
    if (constructed.density.size() != 0) {
        if (constructed.density.rows() != S.rows()
            || constructed.density.cols() != S.cols()) {
            throw std::invalid_argument(
                "run_rks_scf_with_jk: initial_density shape does not "
                "match S");
        }
        D = constructed.density;
    } else {
        D = build_density(C0, nocc);
    }
    Eigen::MatrixXd D_prev = D;

    // Track MO basis between iterations so the C1c Newton step has a
    // current MO frame to operate in. Updated each iteration after either
    // diagonalisation or the quadratic step.
    Eigen::MatrixXd C_prev_mo = C0;
    Eigen::VectorXd eps_prev_mo = eps0;

    RKSResult result;
    result.guess_selection = constructed.selection;
    result.restart_basis = basis;
    result.functional = opts.functional;
    result.e_nuclear = E_nuc;

    if (opts.orbital_optimizer == "opentrustregion") {
        validate_orbital_xc(functional);
        OrbitalXCFunction xc = [&](const Eigen::MatrixXd& da, const Eigen::MatrixXd&, bool response) {
            const auto value = build_xc(functional, basis, grid, da, vv10_grid, xc_functionals);
            result.xc_batch_workers_used = std::max(result.xc_batch_workers_used, value.batch_workers);
            OrbitalXC out; out.energy = value.energy; out.alpha = value.V;
            if (response) out.restricted_kernel = make_xc_kernel(da);
            return out;
        };
        const auto run = run_orbital_scf(opts, X, S, Hcore, E_nuc, jk,
            D, {}, nocc, nocc, true, alpha, xc);
        assign_orbital_restricted(result, run, E_nuc);
        result.e_coulomb = run.state.coulomb;
        result.e_hf_exchange = run.state.exchange;
        result.e_xc = run.state.xc.energy;
        return result;
    }

    // ORCA NOITER compatibility: evaluate the initial density once, but do
    // not take an SCF update or append an iteration trace entry.
    if (opts.max_iter == 0) {
        const Eigen::MatrixXd J = jk.build_J_slot(D, 0);
        Eigen::MatrixXd K_mat;
        if (need_k) {
            K_mat = cam_a * jk.build_K_slot(D, 0);
            if (cam_b != 0.0) {
                K_mat.noalias() += cam_b * jk.build_K_erf(D, rsh_omega);
            }
        }
        XcContribution xc = build_xc(
            functional, basis, grid, D, vv10_grid, xc_functionals);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc.batch_workers);
        Eigen::MatrixXd F = Hcore + J + xc.V;
        if (need_k) F -= 0.5 * K_mat;

        const double E_core = (D.array() * Hcore.array()).sum();
        const double E_J = 0.5 * (D.array() * J.array()).sum();
        const double E_K = need_k
            ? -0.25 * (D.array() * K_mat.array()).sum()
            : 0.0;
        const double E_xc = xc.energy;
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                0.5 * D, S);
            F += vu.V;
            e_dft_plus_u = 2.0 * vu.energy;
        }
        auto [C, eps] = diagonalize_fock(F);
        result.mo_energies = eps;
        result.mo_coeffs = C;
        result.density = D;
        result.fock = F;
        result.energy = E_core + E_J + E_K + E_xc + E_nuc
                      + e_dft_plus_u;
        result.e_electronic = E_core + E_J + E_K + E_xc;
        result.e_coulomb = E_J;
        result.e_hf_exchange = E_K;
        result.e_xc = E_xc;
        result.e_dft_plus_u = e_dft_plus_u;
        return result;
    }

    DIIS diis = make_diis(opts);

    // CPCM warm-start (BUG 100): pre-populate DIIS from a previous
    // inner SCF's (Fock, error) history so the convergence doesn't
    // restart from scratch when only Hcore changed by V_q.
    if (warm_fock_history != nullptr && warm_error_history != nullptr
        && warm_fock_history->size() == warm_error_history->size()) {
        for (std::size_t i = 0; i < warm_fock_history->size(); ++i) {
            // extrapolate() pushes the pair into history and returns
            // an extrapolated Fock we discard here — we only want the
            // side effect of pre-populating the subspace.
            diis.extrapolate((*warm_fock_history)[i],
                            (*warm_error_history)[i]);
        }
    }

    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);
    KDIIS kdiis(opts.diis_subspace_size);
    SOSCF soscf(opts.soscf_opts);  // stateful L-BFGS SOSCF accelerator

    // Dynamic-damping state (see RHF). Inactive by default.
    double current_damping = opts.damping;
    bool have_prev_E = false;
    bool full_fock_validation_pending = false;
    int fock_map_epoch_start_iter = 1;

    double E_prev = 0.0;
    Eigen::MatrixXd F_prev_mixed;
    bool have_prev_fock = false;

    // Auto-level-shift-on-oscillation state — see rhf.cpp for the full
    // description.
    bool oscillation_engaged = false;
    double effective_level_shift = opts.level_shift;
    int effective_warmup = opts.level_shift_warmup_cycles;

    int num_restarts = 0;
    int stall_iters = 0;
    double best_grad_norm = 1e300;
    double best_energy = 0.0;
    Eigen::MatrixXd best_C = C0;
    Eigen::VectorXd best_eps = eps0;
    Eigen::MatrixXd best_D = D;

    // Phase D2c/D2d/D2e-KS state (mirror of rhf.cpp).
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

        // Quadratic-fallback phase: Newton step in MO space replaces the
        // diagonalize-F update once iter > quadratic_fallback_iter. DIIS
        // and damping are skipped during the quadratic phase, same as RHF.
        const bool in_quadratic_phase =
            opts.quadratic_fallback_iter > 0
            && iter > opts.quadratic_fallback_iter;

        // Phase D2c/D2d/D2e-KS activation. We can only judge after the
        // first iter (grad_norm needed) so initial guess for these flags
        // is false; we recompute after computing the gradient below.
        bool in_newton_phase = false;
        bool in_trah_phase   = false;
        bool in_soscf_phase  = false;

        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter
            && !in_quadratic_phase;

        // Gate on the iteration-local current_damping, not opts.damping:
        // with damping=0.0 + dynamic_damping=true the dynamic update is
        // the only source of a non-zero mixing factor (matches rhf.cpp).
        const Eigen::MatrixXd D_used =
            (iter == 1 || current_damping == 0.0 || diis_active
             || in_quadratic_phase)
                ? D
                : current_damping * D_prev + (1.0 - current_damping) * D;

        // Coulomb and (if hybrid) HF exchange. Slot 0 on both J and K
        // — RKS issues at most one call per (J, K) per iter, so a
        // single slot is enough to feed the incremental-Fock ΔD cache
        // (DirectJKBuilder). ``build_J_slot`` / ``build_K_slot`` fall
        // back to the stateless path on non-Direct builders + when
        // ``incremental_fock = false``.
        const Eigen::MatrixXd J = jk.build_J_slot(D_used, 0);
        // Effective exchange matrix K_eff = cam_a·K + cam_b·K_erf(ω).
        // For a global hybrid cam_b == 0 → K_eff = α·K; for a pure
        // functional need_k is false and K_eff stays empty.
        Eigen::MatrixXd K_mat;
        if (need_k) {
            K_mat = cam_a * jk.build_K_slot(D_used, 0);
            if (cam_b != 0.0) {
                K_mat.noalias() += cam_b * jk.build_K_erf(D_used, rsh_omega);
            }
        }

        // XC on the grid.
        XcContribution xc = build_xc(
            functional, basis, grid, D_used, vv10_grid, xc_functionals);
        result.xc_batch_workers_used = std::max(
            result.xc_batch_workers_used, xc.batch_workers);

        Eigen::MatrixXd F = Hcore + J + xc.V;
        if (need_k) {
            // For UHF-style Fock you subtract (1/2) K; here D has the
            // closed-shell "2·" convention baked in, and K_mat already
            // carries the cam_a / cam_b mixing, so the coefficient is ½.
            F -= 0.5 * K_mat;
        }

        // Energy (KS core + J + K + xc + nuc). Computed from the
        // +U-free F (Hcore + J ± K + V_xc) so the double-counting
        // correction in compute_dudarev_energy gives the right answer;
        // see cpp/include/vibeqc/dft_plus_u.hpp + the equivalent block
        // in cpp/src/rhf.cpp:278+ for the math note.
        const double E_core = (D_used.array() * Hcore.array()).sum();
        const double E_J    = 0.5 * (D_used.array() * J.array()).sum();
        const double E_K    = need_k
            ? -0.25 * (D_used.array() * K_mat.array()).sum()
            : 0.0;
        const double E_xc   = xc.energy;

        // Dudarev +U Fock contribution. Per-spin convention: pass
        // P_σ = D_used / 2 (closed-shell ⇒ each spin sees half the
        // total density). V_U is identical for both spins so we add
        // it to the single RKS Fock once; the energy doubles to sum
        // over both spins.
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                0.5 * D_used, S);
            F += vu.V;
            e_dft_plus_u = 2.0 * vu.energy;
        }

        const double E_total = E_core + E_J + E_K + E_xc + E_nuc
                             + e_dft_plus_u;

        // Orbital gradient.
        const Eigen::MatrixXd error = F * D_used * S - S * D_used * F;
        const double grad_norm = error.norm();
        const double dE = E_total - E_prev;

        VIBEQC_DIAG("rks", vibeqc::DiagLevel::VERBOSE,
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
        const bool converged =
            !validating_first_full_build
            && !begin_full_fock_refinement
            && is_scf_converged(
                iter, dE, grad_norm,
                opts.conv_tol_energy, opts.conv_tol_grad);
        if (converged) {
            VIBEQC_DIAG("rks", vibeqc::DiagLevel::DEBUG,
                "converged at iter %d: |dE|=%.3e < %.0e, |grad|=%.3e < %.0e",
                iter, std::abs(dE), opts.conv_tol_energy,
                grad_norm, opts.conv_tol_grad);
        }

        // D2c/D2d/D2e-KS activation. Same mutual-exclusion priority as
        // RHF: quadratic > Newton > TRAH > SOSCF. Each second-order
        // phase replaces "diagonalize F" with an orbital-rotation step;
        // DIIS / damping / level-shift are skipped during these phases.
        // Newton / TRAH build a full orbital-Hessian matvec whose
        // exchange response is scaled by a single ``alpha``; that
        // cannot represent a range-separated hybrid's two-term
        // (cam_a·K + cam_b·K_erf) response, so they are disabled for
        // RSH functionals. Plain DIIS + the F/ε-only SOSCF / quadratic
        // fallback (both RSH-safe) carry the convergence. An
        // RSH-aware second-order Hessian is a follow-up — see
        // handovers/HANDOVER_META_GGA.md § 4.
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

        // Skip DIIS during any second-order phase — the orbital step IS
        // the update mechanism, mixing with Fock extrapolation undoes
        // the trust-region cap. Same logic as the quadratic phase.
        const bool in_second_order_phase =
            in_newton_phase || in_trah_phase || in_soscf_phase;

        int diis_sub = 0;
        // Extrapolation that actually reached F this iteration (#682); see
        // SCFIteration::accelerator_step for the codes.
        int accelerator_step_this_iter = 0;
        if (opts.use_diis && !in_quadratic_phase && !in_second_order_phase) {
            Eigen::MatrixXd F_ext = F;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the DIIS extrapolation; the
                // adaptive depth policy is baked into ``diis`` at
                // construction (see make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    F_ext = diis.extrapolate(F, error);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    accelerator_step_this_iter = 1;
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    F_ext = kdiis.extrapolate(F, C_prev_mo,
                                              eps_prev_mo, nocc);
                    diis_sub = static_cast<int>(kdiis.subspace_size());
                    accelerator_step_this_iter = 2;
                    break;
                }
                case SCFAccelerator::EDIIS: {
                    F_ext = ediis.extrapolate(F, D_used, E_total);
                    diis_sub = static_cast<int>(ediis.subspace_size());
                    accelerator_step_this_iter = 3;
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    Eigen::MatrixXd F_d = diis.extrapolate(F, error);
                    Eigen::MatrixXd F_e =
                        ediis.extrapolate(F, D_used, E_total);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm, static_cast<Eigen::Index>(F.rows()));
                    const bool take_ediis =
                        switch_metric > opts.ediis_diis_switch_threshold;
                    // A DIIS-branch cycle discards F_e; retract it so the
                    // anti-replay guard keys on consumed returns only.
                    if (!take_ediis) ediis.discard_last_extrapolation();
                    F_ext = take_ediis ? F_e : F_d;
                    accelerator_step_this_iter = take_ediis ? 3 : 1;
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    F_ext = adiis.extrapolate(F, D_used);
                    diis_sub = static_cast<int>(adiis.subspace_size());
                    accelerator_step_this_iter = 4;
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    Eigen::MatrixXd F_d = diis.extrapolate(F, error);
                    Eigen::MatrixXd F_a = adiis.extrapolate(F, D_used);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm, static_cast<Eigen::Index>(F.rows()));
                    const bool take_adiis =
                        switch_metric > opts.ediis_diis_switch_threshold;
                    if (!take_adiis) adiis.discard_last_extrapolation();
                    F_ext = take_adiis ? F_a : F_d;
                    accelerator_step_this_iter = take_adiis ? 4 : 1;
                    break;
                }
            }
            if (diis_active) {
                F = F_ext;
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
                F = mix_fock_matrices(F, F_prev_mixed, opts.fock_mixing);
            }
            F_prev_mixed = F;
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
            && !converged
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
            VIBEQC_DIAG("rks", vibeqc::DiagLevel::STANDARD,
                "auto level-shift engaged at iter %d (b=%.2f Ha)",
                iter, effective_level_shift);
        }

        // Saunders-Hillier level shift (C1a-2) applied to F before
        // diagonalisation; same closed-shell formula as RHF since RKS
        // uses the same D = 2·C_occ·C_occ^T convention.
        Eigen::MatrixXd C_new;
        Eigen::VectorXd eps_new;
        if (in_newton_phase) {
            // D2c-KS Newton step. Build the unpolarised LDA/GGA kernel
            // pinned at D_used (the current iteration density), then a
            // full-Hessian CG against (orbital Hessian + W^XC).
            auto xc_kernel = make_xc_kernel(D_used);
            auto step = newton_step(F, C_prev_mo, eps_prev_mo, nocc, jk,
                                   opts.newton_opts, xc_kernel.get(), alpha);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
        } else if (in_trah_phase) {
            // D2e-KS TRAH — same matvec as Newton + Powell-ρ trust radius.
            // ρ update lands at the bottom of the loop.
            auto xc_kernel = make_xc_kernel(D_used);
            auto step = trah_step(F, C_prev_mo, eps_prev_mo, nocc, jk,
                                  opts.trah_opts, trah_trust_radius,
                                  xc_kernel.get(), alpha);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
            trah_predicted_decrease = step.predicted_decrease;
            trah_kappa_norm = step.kappa_norm;
            // The trace row for this iteration was pushed above (before
            // the step); record the TRAH level shift into it now.
            result.scf_trace.back().trah_level_shift = step.level_shift;
        } else if (in_soscf_phase) {
            // D2d-KS Neese SOSCF — no XC kernel needed; F already
            // contains V_xc.
            auto step = soscf.step(F, C_prev_mo, eps_prev_mo, nocc);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
        } else if (in_quadratic_phase) {
            auto step = quadratic_step(F, C_prev_mo, eps_prev_mo, nocc,
                                       opts.quadratic_fallback_shift,
                                       opts.quadratic_fallback_max_step);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
        } else {
            // Standard diagonalize with the unified auto-reducing
            // Saunders-Hillier shift resolved for this iteration.
            // ``apply_level_shift`` returns F untouched when b is 0, so the
            // S·D·S matmuls are skipped once the warm-up releases.
            // ``D_used`` is the closed-shell total density, hence TOTAL.
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
            std::tie(C_new, eps_new) = diagonalize_fock(apply_level_shift(
                F, S, D_used, b, LevelShiftDensity::TOTAL));
        }
        C_prev_mo = C_new;
        eps_prev_mo = eps_new;
        D_prev = D_used;
        D = build_density(C_new, nocc);

        // ---- deterministic orbital-rotation restart (BUG 64) --------
        if (iter == 1) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
        } else if (validating_first_full_build) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_new;
            best_eps = eps_new;
            best_D = D;
        } else if (E_total < best_energy) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_new;
            best_eps = eps_new;
            best_D = D;
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
            && !converged;
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
            best_C = C_new;
            best_eps = eps_new;
            best_D = D;
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            soscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            full_fock_validation_pending = true;
            fock_map_epoch_start_iter = iter + 1;
            if (begin_full_fock_refinement) {
                VIBEQC_DIAG("rks", vibeqc::DiagLevel::STANDARD,
                    "direct-SCF coarse Fock phase ended at iter %d after "
                    "gradient convergence; certifying energy on consecutive "
                    "tight-screened full-density builds "
                    "(|dE|=%.3e, grad=%.3e)",
                    iter, std::abs(dE), grad_norm);
            } else {
                VIBEQC_DIAG("rks", vibeqc::DiagLevel::STANDARD,
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
            VIBEQC_DIAG("rks", vibeqc::DiagLevel::STANDARD,
                "restart %d/%d at iter %d: grad=%.3e, best_grad=%.3e",
                num_restarts, opts.restart_opts.max_restarts,
                iter, grad_norm, best_grad_norm);
            auto [C_rot, eps_rot] = rotate_orbitals_in_subspaces(
                best_C, best_eps, nocc, &F,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 2,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 2 + 1);
            D = build_density(C_rot, nocc);
            C_new = C_rot;
            eps_new = eps_rot;
            diis.clear(); ediis.clear(); adiis.clear(); kdiis.clear();
            soscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_rot;
            best_eps = eps_rot;
            best_D = D;
        }
        if (schwarz_tightened_this_iter && !transition_to_full_fock) {
            stall_iters = 0;
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            soscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
        }
        if (fock_map_changed_this_iter) {
            trah_trust_radius = opts.trah_opts.initial_trust_radius;
            trah_predicted_decrease = 0.0;
            trah_kappa_norm = 0.0;
        }

        result.mo_energies = eps_new;
        result.mo_coeffs = C_new;
        result.density = D_used;
        result.fock = F;
        result.energy = E_total;
        // e_electronic is the +U-free KS electronic energy
        // (E_core + E_J + E_K + E_xc) — matches the RHF/UHF +U-free
        // convention. The Dudarev +U contribution is reported
        // separately in e_dft_plus_u and added to `energy`.
        result.e_electronic = E_total - E_nuc - e_dft_plus_u;
        result.e_coulomb = E_J;
        result.e_hf_exchange = E_K;
        result.e_xc = E_xc;
        result.e_dft_plus_u = e_dft_plus_u;
        result.n_iter = iter;

        if (converged) {
            // Final self-consistency pass (same convention as RHF):
            // rebuild F on the fresh D so the returned MOs and Fock
            // correspond to F(D_final), not F(D_{final-1}).
            // result.energy and the e_coulomb / e_hf_exchange / e_xc
            // decomposition are likewise recomputed on the rebuilt
            // (J_f, K_f, xc_f, D) so the returned energy reproduces
            // from the returned matrices (2026-05-18 audit P3; see
            // tests/test_scf_final_consistency.py). The energy therefore
            // differs from scf_trace[-1].energy by at most ~conv_tol;
            // test_scf_log.py asserts that band.
            //
            // E_K_f uses the SCF's iterated K convention, i.e. it is
            // computed BEFORE the COSX one-center correction: the
            // correction upgrades only the returned Fock / MOs. Folding
            // it into the energy would shift COSX totals off the
            // iterated SCF surface (~1e-4 Ha) and break the parity
            // decomposition self-consistency (test_parity_hf_dft.py);
            // a no-op for non-COSX builders.
            const Eigen::MatrixXd J_f = jk.build_J_slot(D, 0);
            Eigen::MatrixXd K_f;   // effective exchange cam_a·K + cam_b·K_erf
            double E_K_f = 0.0;
            if (need_k) {
                K_f = cam_a * jk.build_K_slot(D, 0);
                if (cam_b != 0.0) {
                    K_f.noalias() += cam_b * jk.build_K_erf(D, rsh_omega);
                }
                // K_f already carries the cam_a / cam_b mixing, so the
                // exchange-energy coefficient is a bare -0.25 (matches
                // the per-iteration E_K formula above).
                E_K_f = -0.25 * (D.array() * K_f.array()).sum();
                jk.apply_one_center_correction(K_f, D);
            }
            XcContribution xc_f = build_xc(
                functional, basis, grid, D, vv10_grid, xc_functionals);
            result.xc_batch_workers_used = std::max(
                result.xc_batch_workers_used, xc_f.batch_workers);
            Eigen::MatrixXd F_f = Hcore + J_f + xc_f.V;
            if (need_k) F_f -= 0.5 * K_f;
            if (!opts.dft_plus_u_sites.empty()) {
                const auto vu_f = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    0.5 * D, S);
                F_f += vu_f.V;
                result.e_dft_plus_u = 2.0 * vu_f.energy;
            }
            auto [C_f, eps_f] = diagonalize_fock(F_f);
            const double E_core_f = (D.array() * Hcore.array()).sum();
            const double E_J_f    = 0.5 * (D.array() * J_f.array()).sum();
            const double E_xc_f   = xc_f.energy;
            // result.e_dft_plus_u was re-reported on the fresh density in
            // the +U branch above (0.0 when +U is off).
            const double E_total_f = E_core_f + E_J_f + E_K_f + E_xc_f
                                   + E_nuc + result.e_dft_plus_u;
            result.mo_energies = eps_f;
            result.mo_coeffs = C_f;
            result.fock = F_f;
            result.density = D;
            result.energy = E_total_f;
            result.e_electronic = E_total_f - E_nuc - result.e_dft_plus_u;
            result.e_coulomb = E_J_f;
            result.e_hf_exchange = E_K_f;
            result.e_xc = E_xc_f;
            result.converged = true;
            // Save DIIS history for CPCM warm-start (BUG 100).
            {
                const auto& fh = diis.fock_history();
                const auto& eh = diis.error_history();
                result.diis_fock_history.assign(fh.begin(), fh.end());
                result.diis_error_history.assign(eh.begin(), eh.end());
            }

            // ---- Restricted-stability VERDICT (issue #144) ------------
            // Verdict only — report the lowest orbital-rotation Hessian
            // eigenvalue of the converged restricted solution, never
            // rotate / escape / promote. The policy for what to do with
            // a negative verdict is the maintainer decision in
            // agentic-loop/asks/ask-scf144-restricted-stability-contract-
            // 2026-08-27.md; this phase only makes it visible (mirrors
            // UHF/UKS: skip silently on unsupported routes by default,
            // fail closed on an explicit request).
            if (opts.stability_check) {
                const auto t_stab_wall0 = std::chrono::steady_clock::now();
                const std::clock_t t_stab_cpu0 = std::clock();
                const auto charge_stability_phase = [&]() {
                    result.stability_wall_s = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_stab_wall0).count();
                    result.stability_cpu_s =
                        static_cast<double>(std::clock() - t_stab_cpu0)
                        / static_cast<double>(CLOCKS_PER_SEC);
                };
                const auto unsupported_stability = [&](bool unsupported,
                                                       const char* missing_response,
                                                       const char* opt_out) {
                    if (!unsupported) return false;
                    if (opts.stability_check_explicit) {
                        throw std::runtime_error(
                            std::string("RKS internal stability analysis was "
                                        "explicitly requested, but ")
                            + missing_response
                            + " is not yet plumbed. Set stability_check=False "
                              "to run "
                            + opt_out + " without a stability verdict.");
                    }
                    // Default-on check must not manufacture a verdict from
                    // an incomplete orbital Hessian. Preserve the converged
                    // first-order result, leave stability_checked=false.
                    return true;
                };
                const double alpha_hf = functional.hf_exchange_fraction();
                const bool unsupported_route =
                    unsupported_stability(
                        !opts.dft_plus_u_sites.empty(),
                        "the DFT+U response",
                        "the supported first-order DFT+U SCF")
                    || (functional.kind() == XCKind::MGGA
                        && unsupported_stability(
                            true,
                            "tau-dependent unpolarised fxc",
                            "the supported first-order meta-GGA SCF"))
                    || unsupported_stability(
                        functional.is_range_separated(),
                        "the range-separated exact-exchange response",
                        "the supported first-order RSH SCF")
                    || unsupported_stability(
                        functional.needs_vv10(),
                        "the VV10 nonlocal correlation response",
                        "the supported first-order VV10 SCF")
                    || unsupported_stability(
                        alpha_hf != 0.0
                            && jk.has_post_scf_exchange_correction(),
                        "the COSX one-centre exchange correction response and "
                        "a consistent corrected-orbital energy surface",
                        "the supported first-order hybrid COSX SCF");
                if (!unsupported_route) {
                    UHFStabilityOptions sopts;
                    sopts.max_iter = opts.stability_davidson_max_iter;
                    // A residual looser than the instability threshold
                    // cannot certify the sign of the reported eigenvalue.
                    sopts.residual_tol =
                        std::min(1e-5, 0.1 * opts.stability_tol);
                    // Equal spin blocks on the restricted solution with
                    // the polarised XC kernel pinned at (D/2, D/2): the
                    // coupled per-spin orbital Hessian decomposes into
                    // the closed-shell KS singlet and triplet sectors
                    // (Bauernschmitt & Ahlrichs 1996, doi:10.1063/1.471637;
                    // HF limit: Seeger & Pople 1977, doi:10.1063/1.434318),
                    // so the lowest eigenvalue over the combined space is
                    // the restricted stability verdict. Memory-bounded
                    // batched kernel, same as the UKS default path. The
                    // kernel builder evaluates polarised fxc, which
                    // requires a spin=2 Functional view (same pattern as
                    // the UKS driver).
                    Functional functional_pol(opts.functional, /*spin=*/2);
                    auto xc_k = make_batched_polarised_xc_kernel_builder(
                        functional_pol, grid, basis, 0.5 * D, 0.5 * D,
                        kMolecularXcKernelMaxWorkers);
                    result.xc_batch_workers_used = std::max(
                        result.xc_batch_workers_used,
                        xc_k->batch_workers_used());
                    const auto stab = uhf_internal_stability_lowest(
                        C_f, C_f, eps_f, eps_f, nocc, nocc, jk, sopts,
                        xc_k.get(), alpha_hf);
                    result.xc_batch_workers_used = std::max(
                        result.xc_batch_workers_used,
                        xc_k->batch_workers_used());
                    // The Hessian builder retains only O(n_grid) scalar
                    // response state, but it need not overlap the complete
                    // XC phase. Release it before returning.
                    xc_k.reset();
                    result.stability_checked = true;
                    result.stability_analysis_converged = stab.converged;
                    result.stability_eigenvalue = stab.lowest_eigenvalue;
                    result.internal_instability =
                        stab.converged
                        && stab.lowest_eigenvalue < -opts.stability_tol;
                }
                charge_stability_phase();
            }
            return result;
        }

        // Powell-ρ trust-radius update for TRAH. Apply now that this
        // iteration's actual ΔE is known (relative to E_prev). Skip the
        // update on the first iteration (no prior E_prev) and on iters
        // where TRAH wasn't the chosen step (predicted_decrease stale).
        if (trah_active_prev && iter > 1
            && !fock_map_changed_this_iter) {
            const double actual_decrease = E_prev - E_total;  // positive = good
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
    // Save DIIS history even on non-convergence — the partial subspace
    // may still accelerate the next CPCM macro-iteration (BUG 100).
    {
        const auto& fh = diis.fock_history();
        const auto& eh = diis.error_history();
        result.diis_fock_history.assign(fh.begin(), fh.end());
        result.diis_error_history.assign(eh.begin(), eh.end());
    }
    return result;
}

}  // namespace vibeqc
