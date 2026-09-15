#pragma once

// Raw real-space SR Coulomb integrals, Sun2023, doi:10.1063/5.0155815,
// Eqs.18 and 45. For each explicitly supplied integer triple (g,p,s):
//   V[image,mu*nao+nu,lambda*nao+sigma]
//     = (mu_0 nu_g | lambda_p sigma_s)_erfc(omega*r12)/r12.
// This is NOT a finite BIPOLE operator: no O/C/D domain predicate, density,
// Bloch phase, 1/Nk, q=0 subtraction, screening by density, or permutation
// averaging is applied. Repeated image triples are legal repeated requests.
// The original normalized basis is used, with one contraction per shell,
// pure L<=min(6, generated ERI backend limit) or Cartesian s. No
// general-contraction reinterpretation occurs.

#include "vibeqc/basis.hpp"
#include "vibeqc/periodic.hpp"
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vibeqc {

struct BipoleErfcImageView {
    const std::int64_t* indices = nullptr; // contiguous [image,3,3]: g,p,s
    std::uint64_t image_count = 0, element_count = 0;
};
struct BipoleErfcPanelSelection {
    std::uint64_t left_pair_begin = 0, left_pair_count = 0;
    std::uint64_t right_pair_begin = 0, right_pair_count = 0;
};
struct BipoleErfcPanelOptions {
    double omega = std::numeric_limits<double>::quiet_NaN();
};
struct BipoleErfcPanelInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    // Required explicit allowance of at least 8 MiB for the reviewed
    // libint initialization/shared Boys
    // tables, allocator capacity/alignment and opaque backend controls.
    // Known engine arrays are counted separately; neither is an RSS proof.
    std::uint64_t backend_margin_bytes_per_replica = 0;
};
struct BipoleErfcPanelCaps {
    std::uint64_t maximum_images = 0, maximum_shell_quartet_calls = 0;
    std::uint64_t maximum_primitive_quartet_visits = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_node_bytes = 0, maximum_work_units = 0;
};
struct BipoleErfcPanelPlan {
    std::uint64_t n_basis = 0, n_shells = 0, image_count = 0;
    BipoleErfcPanelSelection selection;
    std::uint64_t maximum_primitives = 0, maximum_angular_momentum = 0;
    std::uint64_t backend_maximum_angular_momentum = 0;
    std::uint64_t output_elements = 0, retained_output_bytes = 0;
    std::uint64_t shell_quartet_calls_upper_bound = 0, primitive_quartet_visits_upper_bound = 0;
    std::uint64_t maximum_shell_quartet_elements = 0;
    std::uint64_t engine_primitive_bytes = 0, engine_pair_bytes = 0;
    std::uint64_t engine_stack_bytes = 0, engine_scratch_bytes = 0, engine_core_numeric_bytes = 0;
    std::uint64_t shell_copy_numeric_bytes = 0, fixed_numeric_workspace_bytes = 0;
    std::uint64_t borrowed_basis_numeric_bytes = 0, borrowed_image_bytes = 0;
    std::uint64_t borrowed_geometry_numeric_bytes = 0, borrowed_numerical_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, control_storage_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t metadata_work_units = 0, validation_work_units = 0;
    std::uint64_t contraction_work_units = 0, work_units_upper_bound = 0;
};
struct BipoleErfcPanelDiagnostics {
    std::uint64_t shell_quartet_calls = 0, null_shell_quartet_buffers = 0;
    double maximum_integral_magnitude = 0;
    // Primitive screening target is exactly zero. A null libint buffer is
    // recorded and returned as zero, not certified as an exact analytic zero.
    double primitive_screening_precision = 0;
};
class BipoleErfcPanelResult {
public:
    BipoleErfcPanelResult(const BipoleErfcPanelResult&) = delete;
    BipoleErfcPanelResult& operator=(const BipoleErfcPanelResult&) = delete;
    BipoleErfcPanelResult(BipoleErfcPanelResult&&) noexcept = default;
    BipoleErfcPanelResult& operator=(BipoleErfcPanelResult&&) noexcept = default;
    const BipoleErfcPanelPlan& memory() const noexcept { return plan_; }
    const BipoleErfcPanelDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const double* data() const;
    double element(std::uint64_t image, std::uint64_t left, std::uint64_t right) const;
    const std::string& input_identity_sha256() const;
    const std::string& source_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
private:
    BipoleErfcPanelResult() = default;
    BipoleErfcPanelPlan plan_;
    BipoleErfcPanelDiagnostics diagnostics_;
    std::vector<double> values_;
    std::string input_identity_, source_identity_, payload_identity_;
    friend BipoleErfcPanelResult make_bipole_erfc_panel(
        const BasisSet&, const PeriodicSystem&, BipoleErfcImageView,
        const BipoleErfcPanelSelection&, const BipoleErfcPanelOptions&,
        const BipoleErfcPanelInventory&, const BipoleErfcPanelCaps&);
};

// Count-only descriptor census: no primitive/image-label scan, numeric
// allocation or engine construction. Full admission precedes all payload work.
// Work is a conservative abstract census, not a backend instruction counter.
BipoleErfcPanelPlan plan_bipole_erfc_panel(
    const BasisSet&, const PeriodicSystem&, BipoleErfcImageView,
    const BipoleErfcPanelSelection&, const BipoleErfcPanelOptions&,
    const BipoleErfcPanelInventory&, const BipoleErfcPanelCaps&);
BipoleErfcPanelResult make_bipole_erfc_panel(
    const BasisSet&, const PeriodicSystem&, BipoleErfcImageView,
    const BipoleErfcPanelSelection&, const BipoleErfcPanelOptions&,
    const BipoleErfcPanelInventory&, const BipoleErfcPanelCaps&);

} // namespace vibeqc
