// Second-order Møller-Plesset (MP2) correlation energy on a closed-shell
// RHF reference.
//
// Canonical RMP2 formula (Szabo-Ostlund eq. 6.71) in spatial-orbital form:
//   E_MP2 = Σ_{ij ∈ occ} Σ_{ab ∈ virt}  (ia|jb) · [2 (ia|jb) − (ib|ja)]
//                                       / (ε_i + ε_j − ε_a − ε_b)
//
// Closed-shell only — a spin-unrestricted counterpart (UMP2) on UHF
// references is a separate phase.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <string>

#include "basis.hpp"
#include "molecule.hpp"
#include "rhf.hpp"

namespace vibeqc {

struct MP2Options {
    // Frozen-core convention. -1 selects vibe-qc's published chemical-core
    // count (ORCA 6.1 Table 2.69, converted from electrons to spatial
    // orbitals); 0 requests all-electron correlation; a positive value
    // freezes that many lowest-energy occupied spatial MOs. The published
    // policy is count-only and does not reproduce ORCA's additional
    // orbital-character reorder for unusual canonical orderings.
    int n_frozen_core = -1;

    // Density fitting (resolution of the identity). When ``density_fit
    // = true``, the AO→MO ERI transformation goes through the DF
    // factorisation
    //   (ia|jb) ≈ Σ_P B^P_{ia} B^P_{jb}
    // (Vahtras-Almlöf-Feyereisen, CPL 213, 514 (1993)) instead of the
    // explicit four-index ERI. Cost drops from O(n^5) to
    // O(n_aux · n_occ · n_vir · n_orb) for the half-transform plus
    // O(n_aux · n_occ²·n_vir²) for the (ia|jb) build — the ratio
    // scales as n_aux / n_orb² ≈ 1/3 for typical orbital + JK aux
    // pairings.
    //
    // ``aux_basis`` is a libint-recognised auxiliary basis name (e.g.
    // "def2-svp-rifit", "cc-pvtz-ri"). For MP2 the per-zeta RIfit
    // family (def2-svp-rifit … def2-qzvppd-rifit, cc-pvNz-ri) is the
    // right choice; the universal-jkfit and per-zeta-jkfit families
    // also work but are over-fit for correlation. Empty + density_fit
    // = true raises with a hint at vibeqc.default_aux_basis_for(name,
    // kind="ri").
    bool density_fit = false;
    std::string aux_basis;

    // Scaled-spin-component coefficients (Grimme JCP 118, 9095 (2003)).
    // The correlation energy is reported as
    //   e_correlation = c_os · E_os + c_ss · E_ss
    // with E_os the singlet (αβ) pair contribution Σ (ia|jb)²/Δ and
    // E_ss the triplet (αα+ββ) pair contribution
    // Σ [(ia|jb)² − (ia|jb)(ib|ja)]/Δ. The defaults c_os = c_ss = 1
    // recover canonical RMP2. Named recipes:
    //   * SCS-MP2 (Grimme 2003):           c_os = 6/5,  c_ss = 1/3
    //   * SOS-MP2 (Jung et al. 2004):      c_os = 1.3,  c_ss = 0
    //   * B2PLYP MP2 correction (a_c=0.27): c_os = c_ss = 0.27
    // Scaling composes identically with ``density_fit`` — both paths
    // share the same (E_os, E_ss) accumulation and apply the scales
    // at the final sum.
    double c_os = 1.0;
    double c_ss = 1.0;

    // When true, store the half-transformed B-mo intermediate
    // (n_aux × n_occ·n_vir) as single-precision float instead of
    // double.  Halves the memory with no measurable accuracy loss
    // — the V⁻¹/² contraction stays in double, and the final
    // (ia|jb) GEMM accumulates in double.
    bool use_float_intermediates = false;

    // Diagnostic: when ``density_fit = true``, also build the canonical
    // four-index (ia|jb) tensor and report the per-bucket residual on
    // ``MP2Result::{e_os_ri_residual, e_ss_ri_residual}``. Off by
    // default because it defeats the speed advantage of RI (it forces
    // the O(n^5) canonical AO→MO transform alongside the O(n_aux ⋯ )
    // RI build). Intended use is sanity-checking aux-basis adequacy:
    // the Coulomb-metric Dunlap fit produces structurally biased
    // residuals (E_os over-estimated, E_ss under-estimated) that
    // partially cancel in the unscaled sum but survive SCS/SOS
    // scaling; this flag surfaces the per-bucket fingerprint without
    // requiring a second user-facing run.
    bool report_ri_residual = false;

    // Bounded-memory execution. ``requested_memory_bytes == 0`` preserves
    // the historical behaviour (the auto mode selects the in-core kernel).
    // With a non-zero budget, auto selects the in-core kernel only when its
    // modeled peak fits; otherwise it selects the integral-direct/panel
    // kernel. ``disk`` uses the same exact integral route as ``direct`` but
    // spills occupied-orbital MO slabs beneath ``scratch_directory`` and
    // removes its private scratch directory on every exit path.
    std::size_t requested_memory_bytes = 0;
    std::string memory_mode = "auto";  // auto | incore | direct | disk
    std::string scratch_directory;
};

struct MP2Result {
    double e_hf = 0.0;           // Underlying RHF total energy (Hartree)
    double e_correlation = 0.0;  // c_os · e_os + c_ss · e_ss
    double e_total = 0.0;        // e_hf + e_correlation
    int n_frozen_core = 0;       // Resolved frozen occupied spatial MOs

    // Grimme-convention spin-component decomposition (Grimme 2003).
    // Both numbers are reported *unscaled* — multiply by the
    // ``MP2Options::c_os`` / ``c_ss`` chosen for the run to reproduce
    // ``e_correlation``. At canonical c_os = c_ss = 1 the two terms
    // sum directly to ``e_correlation`` (and to E_MP2 from
    // Szabo-Ostlund eq. 6.71).
    double e_os = 0.0;  // Opposite-spin (αβ pair):   Σ (ia|jb)²/Δ
    double e_ss = 0.0;  // Same-spin (αα+ββ pairs):
                        //   Σ [(ia|jb)² − (ia|jb)(ib|ja)]/Δ

    // RI fit residual diagnostic, populated only when
    // ``MP2Options::report_ri_residual && density_fit`` was set. The
    // residuals are ``e_{os,ss}^{RI} − e_{os,ss}^{canonical}``: positive
    // means the RI bucket over-shoots canonical, negative under-shoots.
    // For Coulomb-metric Dunlap fits with a per-zeta RIfit aux, the
    // characteristic signature is ``e_os_ri_residual > 0`` and
    // ``e_ss_ri_residual < 0`` (partial cancellation in the unscaled
    // sum). ``ri_residual_reported`` distinguishes "not requested" from
    // "requested and computed to zero".
    double e_os_ri_residual = 0.0;
    double e_ss_ri_residual = 0.0;
    bool   ri_residual_reported = false;

    // Bounded-memory execution telemetry. ``workspace_bytes`` is the
    // modeled peak of MP2-owned integral/transform storage for the resolved
    // kernel; ``disk_bytes`` is the maximum scratch-file extent (zero unless
    // disk mode was selected).
    std::string memory_mode_used = "incore";
    std::size_t workspace_bytes = 0;
    std::size_t disk_bytes = 0;
};

// Compute RMP2 on a converged RHF reference. `rhf.mo_energies`,
// `rhf.mo_coeffs` and `rhf.density` are read; they must correspond to
// the `mol` / `basis` passed here.
MP2Result run_mp2(const Molecule& mol,
                  const BasisSet& basis,
                  const RHFResult& rhf,
                  const MP2Options& options = {});

// Internal substrate shared by RMP2 and UMP2. It evaluates exact AO shell
// quartets with libint and contracts only the requested bra-occupied slab
// into OV blocks. No AO n^4 or complete OVOV tensor is materialised.
// Kept in this header (rather than duplicated in ump2.cpp) so restricted and
// unrestricted direct routes have one permutation convention.
namespace mp2_detail {

using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;

RowMatrix build_direct_ovov_i_batch(
    const BasisSet& basis,
    const Eigen::MatrixXd& C_bra_occ,
    const Eigen::MatrixXd& C_bra_vir,
    const Eigen::MatrixXd& C_ket_occ,
    const Eigen::MatrixXd& C_ket_vir,
    std::size_t first_bra_occupied,
    std::size_t occupied_count);

}  // namespace mp2_detail

}  // namespace vibeqc
