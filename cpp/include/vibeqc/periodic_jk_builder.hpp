// Periodic-Γ adapter that lets molecular SCF drivers build the
// Fock-matrix two-electron piece on a PeriodicSystem.
//
// Implements the molecular ``JKBuilder`` interface (see
// jk_builder.hpp) on top of ``build_jk_gamma_molecular_limit`` (see
// periodic_fock.hpp). Once the molecular SCF drivers are factored to
// take a JKBuilder (the v0.7.x jk_builder.cpp refactor), passing one
// of these into the same SCF loop drives a Γ-only periodic-RHF /
// periodic-RKS calculation without changing any driver code. The
// only remaining periodic-specific pieces are the one-electron
// matrix (lattice-summed T + V_ne) and the nuclear repulsion energy
// (Ewald 3D / direct truncated for lower-dim), which the driver
// already takes as inputs in the JKBuilder formulation.
//
// Currently wraps the direct four-index lattice ERI path; future
// concrete classes — ``make_periodic_gamma_df_jk_builder`` (RI-J
// with Ewald-screened metric), ``make_periodic_gamma_cosx_jk_builder``
// (RI-J + COSX-K with periodic-Becke grid) — slot in alongside this
// one without touching the SCF drivers.

#pragma once

#include <memory>

#include "basis.hpp"
#include "jk_builder.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// Γ-only, molecular-limit periodic J/K builder. The two lattice
// indices in ``build_jk_gamma_molecular_limit`` are both capped by
// ``opts.cutoff_bohr``; ``omega`` selects the Coulomb kernel
// (``omega == 0`` → full 1/r_12, ``omega > 0`` → erfc-screened
// short-range Ewald piece, see periodic_fock.hpp).
//
// The returned builder owns ``opts`` (a small POD struct) by value
// and ``basis`` / ``system`` by raw pointer — the caller must keep
// them alive for the builder's lifetime.
std::unique_ptr<JKBuilder> make_periodic_gamma_jk_builder(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    double omega = 0.0);

// CCM (Cyclic Cluster Model) — WSSC-weighted periodic-Γ JKBuilder.
//
// Wraps ``build_jk_ccm_weighted`` in the same JKBuilder interface so the
// existing ``run_rhf_scf_with_jk`` SCF driver can run a CCM HF calculation
// without changes. ``weight_cells`` and ``weight_matrices`` carry the WSSC
// two-center pair weights (see periodic_fock.hpp for the format). The
// builder owns copies of the weight data; the caller keeps ``basis`` and
// ``system`` alive for the builder's lifetime.
std::unique_ptr<JKBuilder> make_periodic_gamma_ccm_jk_builder(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<Eigen::Vector3i>& weight_cells,
    const std::vector<Eigen::MatrixXd>& weight_matrices,
    const std::string& method = "bra_home",
    double omega = 0.0);

}  // namespace vibeqc
