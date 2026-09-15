#pragma once

// Selected finite-image SR Bloch contribution, not a BIPOLE Hamiltonian or
// correlation source certificate. Sun2023, doi:10.1063/5.0155815,
// Eqs.13-18/45-47, with the inverse-Bloch convention of bipole_ewald_gram:
//   V[L,R] = sum_(g,p,s) exp[i kL.g + i(q+kR).p - i kR.s]
//                    (mu_0 nu_g | lambda_p sigma_s)_erfc(omega*r12)/r12.
// q is a RegularKMesh transfer; kL/kR are its possibly shifted grid points.
// Integer phases are reduced modulo each doubled mesh extent before any
// conversion to binary64. No Cartesian k.R, hidden Nk/spin factor, density,
// q=0 subtraction, image completion, screening or permutation repair.
// Repeated image triples are repeated contributions, in caller order.

#include "vibeqc/bipole_erfc_panel.hpp"
#include "vibeqc/bipole_ewald_gram.hpp"
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace vibeqc {

struct BipoleErfcBlochOptions {
    BipoleErfcPanelOptions raw;
    // Exact image-multiset closure under all eight real-ERI permutations,
    // each reanchored at its first AO's cell. No completion or averaging.
    // This does not cover AO masks, spatial symmetry, or a matched HF source.
    bool require_image_permutation_closure = false;
};
struct BipoleErfcBlochInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    // Includes the raw backend's required positive opaque allowance; it is
    // charged once, not subtracted from known engine numeric arrays.
    std::uint64_t backend_margin_bytes_per_replica = 0;
};
struct BipoleErfcBlochCaps {
    BipoleErfcPanelCaps raw;
    std::uint64_t maximum_kpoints = 0, maximum_phase_evaluations = 0;
    // Required positive only when image-permutation closure is requested.
    std::uint64_t maximum_support_image_comparisons = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_node_bytes = 0, maximum_work_units = 0;
};
struct BipoleErfcBlochPlan {
    std::uint64_t n_basis = 0, n_kpoints = 0, image_count = 0;
    BipoleEwaldGramSelection selection;
    std::uint64_t output_elements = 0, retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t phase_evaluations = 0, folded_scalar_terms = 0;
    std::uint64_t support_image_comparisons_upper_bound = 0, support_work_units = 0;
    std::uint64_t fixed_numeric_workspace_bytes = 0, wrapper_control_storage_bytes = 0;
    std::uint64_t raw_phase_owned_numerical_bytes = 0, fold_phase_owned_numerical_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, borrowed_numerical_bytes = 0;
    std::uint64_t control_storage_bytes = 0, per_replica_inventoried_bytes = 0;
    std::uint64_t required_node_inventoried_bytes = 0;
    std::uint64_t phase_work_units = 0, folding_work_units = 0, identity_work_units = 0;
    std::uint64_t work_units_upper_bound = 0;
    // Already admitted with wrapper output+compensation+scalar workspace and
    // wrapper controls added to raw.other_live. Never subtract these twice.
    BipoleErfcPanelPlan raw;
};
struct BipoleErfcBlochDiagnostics {
    std::uint64_t evaluated_images = 0, folded_scalar_terms = 0;
    std::uint64_t support_image_comparisons = 0;
    double maximum_phase_modulus_residual = 0, maximum_integral_magnitude = 0;
    BipoleErfcPanelDiagnostics raw;
};
class BipoleErfcBlochResult {
public:
    BipoleErfcBlochResult(const BipoleErfcBlochResult&) = delete;
    BipoleErfcBlochResult& operator=(const BipoleErfcBlochResult&) = delete;
    BipoleErfcBlochResult(BipoleErfcBlochResult&&) noexcept;
    BipoleErfcBlochResult& operator=(BipoleErfcBlochResult&&) noexcept;
    const BipoleErfcBlochPlan& memory() const noexcept { return plan_; }
    const BipoleErfcBlochDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::complex<double>* data() const;
    std::complex<double> element(std::uint64_t left, std::uint64_t right) const;
    const std::string& input_identity_sha256() const;
    const std::string& source_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    const std::string& raw_input_identity_sha256() const;
    const std::string& raw_source_identity_sha256() const;
    const std::string& raw_payload_identity_sha256() const;
    bool image_permutation_support_certified() const;
    // Empty unless the requested multiset check succeeded on the exact input.
    const std::string& image_permutation_support_identity_sha256() const;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
private:
    BipoleErfcBlochResult() = default;
    BipoleErfcBlochPlan plan_;
    BipoleErfcBlochDiagnostics diagnostics_;
    std::vector<std::complex<double>> values_;
    std::string input_identity_, source_identity_, payload_identity_;
    std::string raw_input_identity_, raw_source_identity_, raw_payload_identity_;
    std::string image_permutation_support_identity_;
    friend BipoleErfcBlochResult make_bipole_erfc_bloch(
        const BasisSet&, const PeriodicSystem&, const RegularKMesh&, BipoleErfcImageView,
        const BipoleEwaldGramSelection&, const BipoleErfcBlochOptions&,
        const BipoleErfcBlochInventory&, const BipoleErfcBlochCaps&);
};

// Count-only composition; no image/primitive/lattice floating payload scans
// or raw backend engine allocation. Known outer phase/fold counts are capped
// before the raw metadata planner. Full child AND enclosing memory/work
// admission precedes make's payload reads. Known raw arrays and opaque backend
// margin are both retained in the census, not interchangeable reservations.
// Peak = complex output+compensation + wrapper fixed scalar workspace + raw
// peak. During folding the engine is gone but one whole supplied image-slab
// raw result remains; it is released before the final result is published.
// No all-image domain, all-AO square, phase table or full-k array is invented.
// Inputs are borrowed immutable for the entire synchronous call; no callbacks.
BipoleErfcBlochPlan plan_bipole_erfc_bloch(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&, BipoleErfcImageView,
    const BipoleEwaldGramSelection&, const BipoleErfcBlochOptions&,
    const BipoleErfcBlochInventory&, const BipoleErfcBlochCaps&);
BipoleErfcBlochResult make_bipole_erfc_bloch(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&, BipoleErfcImageView,
    const BipoleEwaldGramSelection&, const BipoleErfcBlochOptions&,
    const BipoleErfcBlochInventory&, const BipoleErfcBlochCaps&);

} // namespace vibeqc
