// Periodic exchange-correlation Fock contribution.
//
// Generalises the molecular build_xc from rks.cpp to a real-space density
// matrix P(h):
//
//   ρ(r)   = Σ_{μν, h} P_μν(h) χ_μ(r) χ_ν(r − h)
//   ∇ρ(r) = Σ_{μν, h} P_μν(h) [ ∇χ_μ(r) χ_ν(r − h) + χ_μ(r) ∇χ_ν(r − h) ]
//
// V_xc lives in real space as a LatticeMatrixSet:
//
//   V_xc^μν(g) = ∫ w(r) [ v_ρ(r) χ_μ(r) χ_ν(r − g)
//                          + 2 v_σ(r) (∇ρ(r) · ∇χ_μ(r) χ_ν(r − g)
//                                       + ∇ρ(r) · χ_μ(r) ∇χ_ν(r − g)) ] dr
//
// E_xc per unit cell = Σ_r w(r) · ρ(r) · ε_xc(ρ(r), σ(r)).
//
// Numerical integration runs on the caller-provided ``Grid``. For
// production periodic DFT, callers build that grid with
// :func:`build_periodic_becke_grid` (Python) /
// ``build_grid_periodic`` (C++) so the Becke partition denominator
// honours image atoms (Phase 12f). The PeriodicKSOptions driver
// defaults wire this in automatically since v0.9.x
// (use_periodic_becke = true). A plain molecular ``build_grid`` over
// the unit-cell atoms is still accepted — it integrates correctly
// only in the molecular-limit regime where no image atom sits
// within the partition reach of a home-cell grid point.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>

#include "basis.hpp"
#include "grid.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"
#include "xc.hpp"

namespace vibeqc {

// Shared admission for the bounded AO workspace used by value/force kernels.
// Density, quadrature, lattice outputs and pair metadata remain separate costs.
std::size_t periodic_xc_value_batch_size(
    std::size_t nbf, std::size_t n_active, bool need_grad,
    bool open_shell, std::size_t n_threads);
std::size_t periodic_xc_gradient_batch_size(
    std::size_t nbf, std::size_t n_cells, bool need_hess,
    bool open_shell, std::size_t n_threads);

struct PeriodicXCContribution {
    LatticeMatrixSet V_xc;   // real-space V_xc(g), same cell list as P
    double e_xc = 0.0;       // E_xc per unit cell (Hartree)
};

// Interpretation of a periodic real-space density supplied to the XC
// quadrature.  AUTO preserves the historical data-dependent behaviour for
// existing callers.  PERIODIC_LATTICE is the variational finite-torus
// contract: both AO indices range over lattice images even when every
// non-home P(g) block happens to vanish (for example a constant D(k)).
// MOLECULAR_HOME keeps only the home-cell bra and is intended for explicit
// molecular-limit embeddings.
enum class PeriodicXCDensityDomain {
    AUTO,
    MOLECULAR_HOME,
    PERIODIC_LATTICE,
};

// Counts of evaluated AO images and bra images, without AO/grid allocation.
// AUTO uses the periodic upper bound because no density is supplied here.
std::array<std::size_t, 2> periodic_xc_domain_counts(
    const PeriodicSystem& system, const std::vector<LatticeCell>& cells,
    const LatticeSumOptions& opts, PeriodicXCDensityDomain density_domain);

// Build the XC Fock contribution from a real-space density.
//
// ``density_cutoff`` determines which P(h) entries contribute to ρ(r) and
// ∇ρ(r); for molecular-limit tests it may be set equal to the cell list
// length so all provided density blocks are used.
PeriodicXCContribution build_xc_periodic(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const Grid& grid,
                                         const Functional& func,
                                         const LatticeMatrixSet& P_real_space,
                                         const LatticeSumOptions& opts,
                                         PeriodicXCDensityDomain density_domain =
                                             PeriodicXCDensityDomain::AUTO);

// Open-shell (UKS) periodic XC.
//
// Two real-space densities ``P_alpha(g)``, ``P_beta(g)`` (one-particle —
// no factor of 2). Returns separate V_xc_alpha(g), V_xc_beta(g) lattice
// matrix sets plus the spin-resolved E_xc per unit cell.
//
//   ρ_σ(r)   = Σ_{μν, h} (P_σ)_μν(h) χ_μ(r) χ_ν(r − h)
//   ∇ρ_σ(r) = Σ_{μν, h} (P_σ)_μν(h) [∇χ_μ χ_ν + χ_μ ∇χ_ν]
//   σ_αα = |∇ρ_α|²,  σ_αβ = ∇ρ_α · ∇ρ_β,  σ_ββ = |∇ρ_β|²
//
// libxc's spin-polarized eval gives (v_ρ_σ, v_σ_αα, v_σ_αβ, v_σ_ββ) at
// every grid point; the V_xc_σ matrix in cell g is assembled with the
// closed-shell formula on the LDA piece (with v_ρ → v_ρ_σ) and the
// per-spin "flow vector" formulation on the GGA piece (matches the
// molecular UKS path in cpp/src/uks.cpp build_uks_xc).
struct PeriodicUKSXCContribution {
    LatticeMatrixSet V_alpha;   // V_xc^α(g)
    LatticeMatrixSet V_beta;    // V_xc^β(g)
    double e_xc = 0.0;          // total E_xc per unit cell (Hartree)
};

PeriodicUKSXCContribution build_xc_periodic_uks(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_alpha_real_space,
    const LatticeMatrixSet& P_beta_real_space,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain =
        PeriodicXCDensityDomain::AUTO);

// ---------------------------------------------------------------------------
// Periodic XC Pulay atomic gradient (the analytic V_xc force).
//
// Differentiates E_xc = Σ_r w(r) ε_xc(ρ(r)) w.r.t. the home-cell atom
// positions, holding the integration grid fixed (the Becke grid-weight
// derivative ∂w/∂R is neglected — the same approximation the molecular
// xc_pulay_gradient_* kernels make; valid to grid accuracy for fine grids).
//
// Both the home AO χ_ref (μ ∈ A) and the image AO χ_h (ν ∈ A, shifted by
// cell h) move with R_A, so ∂ρ/∂R_A carries two terms — mirroring the ∇ρ
// assembly in build_xc_periodic:
//
//   (∂E_xc/∂R_A)_c = −Σ_h [ Σ_{μ∈A} Σ_r w·v_ρ · ∂_c χ_ref(r)_μ (χ_h·P(h)ᵀ)(r)_μ
//                          + Σ_{ν∈A} Σ_r w·v_ρ · ∂_c χ_h(r)_ν (χ_ref·P(h))(r)_ν ]
//                    + GGA σ-terms (∇ρ-coupled, via the AO Hessian).
//
// Reduces to the molecular xc_pulay_gradient_rks_* (the −2 prefactor) on a
// single home cell. Returns (n_atoms, 3). LDA exact; GGA sigma-Pulay terms
// included; meta-GGA τ terms are out of scope until periodic XC assembles τ.
Eigen::MatrixXd xc_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_real_space,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain = PeriodicXCDensityDomain::AUTO);

// Open-shell companion of xc_lattice_gradient_contribution.  Holds the
// grid fixed and differentiates the alpha + beta XC density with per-spin
// v_rho and GGA sigma potentials from the spin-polarized functional.  LDA
// exact; GGA sigma-Pulay terms included; meta-GGA τ terms are out of scope
// until periodic XC assembles τ.
Eigen::MatrixXd xc_lattice_gradient_contribution_uks(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const Grid& grid,
    const Functional& func,
    const LatticeMatrixSet& P_alpha_real_space,
    const LatticeMatrixSet& P_beta_real_space,
    const LatticeSumOptions& opts,
    PeriodicXCDensityDomain density_domain = PeriodicXCDensityDomain::AUTO);

}  // namespace vibeqc
