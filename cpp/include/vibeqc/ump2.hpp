// Unrestricted second-order Møller-Plesset (UMP2) correlation energy on
// a UHF reference.
//
// Spin-orbital formula grouped by spin channel (α = up, β = down):
//
//   E_UMP2 = (1/4) Σ_{ij∈occ_α, ab∈vir_α}
//               |⟨ij||ab⟩_αα|² / (ε_i^α + ε_j^α − ε_a^α − ε_b^α)
//          + (1/4) Σ_{ij∈occ_β, ab∈vir_β}
//               |⟨ij||ab⟩_ββ|² / (ε_i^β + ε_j^β − ε_a^β − ε_b^β)
//          +       Σ_{i∈occ_α, a∈vir_α, j∈occ_β, b∈vir_β}
//                 (ia|jb)_αβ²   / (ε_i^α + ε_j^β − ε_a^α − ε_b^β)
//
// with antisymmetrised same-spin integrals
//   ⟨ij||ab⟩_σσ = (ia|jb)_σσ − (ib|ja)_σσ
// and opposite-spin integrals
//   (ia|jb)_αβ  = ∫∫ φ^α_i(1) φ^α_a(1) · 1/r12 · φ^β_j(2) φ^β_b(2) d1 d2.
//
// Reduces to RMP2 when D_α = D_β and α,β spatial orbitals coincide.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <string>

#include "basis.hpp"
#include "molecule.hpp"
#include "uhf.hpp"

namespace vibeqc {

struct UMP2Options {
    // Frozen-core convention. -1 selects vibe-qc's published chemical-core
    // count (ORCA 6.1 Table 2.69, converted from electrons to spatial
    // orbitals); 0 requests all-electron correlation; a positive value
    // freezes that many lowest-energy doubly occupied spatial MOs. The
    // published policy is count-only and does not reorder orbitals by
    // atomic-orbital character.
    int n_frozen_core = -1;

    // Density fitting (resolution of the identity). When ``density_fit
    // = true``, all three (αα, ββ, αβ) AO→MO transforms route through
    // the DF factorisation, with α and β MO B-tensors built once each
    // and the three OVOV blocks formed as
    //   αα : B_α^T · B_α
    //   ββ : B_β^T · B_β
    //   αβ : B_α^T · B_β
    // via single GEMM calls. See MP2Options::density_fit and
    // cpp/include/vibeqc/df.hpp for the math.
    bool density_fit = false;
    std::string aux_basis;

    // Scaled-spin-component coefficients (Grimme JCP 118, 9095 (2003)).
    // The correlation energy is reported as
    //   e_correlation = c_os · e_ab + c_ss · (e_aa + e_bb)
    // — i.e. the opposite-spin channel is scaled by ``c_os`` and the
    // sum of the same-spin channels by ``c_ss``. Defaults c_os =
    // c_ss = 1 recover canonical UMP2. See ``MP2Options`` for the
    // named SCS / SOS / B2PLYP recipes; the same factors apply
    // identically for the unrestricted reference.
    double c_os = 1.0;
    double c_ss = 1.0;

    // Diagnostic: when ``density_fit = true``, also build the canonical
    // four-index OVOV tensors for all three (αα, ββ, αβ) channels and
    // report the per-channel residual on
    // ``UMP2Result::{e_aa_ri_residual, e_bb_ri_residual,
    // e_ab_ri_residual}``. Off by default — it doubles the runtime of
    // an RI-UMP2 call (forces the O(N⁵) canonical AO→MO transform). The
    // closed-shell counterpart is ``MP2Options::report_ri_residual``;
    // see that field for the rationale. For UMP2 the Coulomb-metric
    // Dunlap fit biases the αβ channel like the RMP2 opposite-spin
    // bucket and the αα / ββ channels like the same-spin bucket.
    bool report_ri_residual = false;

    // Same bounded-memory contract as MP2Options. A zero budget keeps the
    // legacy in-core auto default; a non-zero budget makes auto choose a
    // direct occupied-orbital slab route when the in-core channel tensors do
    // not fit. Disk scratch never changes the integral approximation.
    std::size_t requested_memory_bytes = 0;
    std::string memory_mode = "auto";  // auto | incore | direct | disk
    std::string scratch_directory;
};

struct UMP2Result {
    double e_hf = 0.0;           // Underlying UHF total energy
    double e_correlation = 0.0;  // c_os · e_ab + c_ss · (e_aa + e_bb)
    double e_total = 0.0;
    int n_frozen_core = 0;       // Resolved frozen occupied spatial MOs

    double e_aa = 0.0;           // αα same-spin channel
    double e_bb = 0.0;           // ββ same-spin channel
    double e_ab = 0.0;           // αβ opposite-spin channel

    // RI fit residual diagnostic, populated only when
    // ``UMP2Options::report_ri_residual && density_fit`` was set. Each
    // residual is ``e_channel^{RI} − e_channel^{canonical}`` at the
    // same UHF reference: positive means the RI channel over-shoots
    // canonical, negative under-shoots. ``ri_residual_reported``
    // distinguishes "not requested" from "requested and computed to
    // zero".
    double e_aa_ri_residual = 0.0;
    double e_bb_ri_residual = 0.0;
    double e_ab_ri_residual = 0.0;
    bool   ri_residual_reported = false;

    std::string memory_mode_used = "incore";
    std::size_t workspace_bytes = 0;
    std::size_t disk_bytes = 0;
};

UMP2Result run_ump2(const Molecule& mol,
                    const BasisSet& basis,
                    const UHFResult& uhf,
                    const UMP2Options& options = {});

}  // namespace vibeqc
