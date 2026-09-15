#pragma once

// An actual, density-independent finite reciprocal source, authenticated by
// native inputs and exact accepted records. This shares the v1 state-based
// fixed-FMA sphere predicate and certified box; it does not manufacture an
// SCF state or change v1 source identities. Sun et al. (2017),
// doi:10.1063/1.4998644, Eqs. 13, 16, 20-21.

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/periodic_gaussian_source_context.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicGaussianReciprocalSourceVersion = 1U;

struct PeriodicGaussianReciprocalSourceCaps {
    // All positive. Fixed storage includes the output native source, retained
    // context object, and explicitly sized factory workspace. This is an
    // inventory of these objects, NOT total stack/RSS. Shared-pointer control,
    // allocator, SHA/string, return temporaries and the existing interval
    // certifier's fixed scalar/control workspace are separate backend costs.
    std::uint64_t maximum_fixed_storage_bytes = 0;
    std::uint64_t maximum_candidates_per_source = 0;
    std::uint64_t maximum_candidate_evaluations = 0;
    // Applied to EACH exact source SHA wire after count, before hash.
    std::uint64_t maximum_source_wire_bytes = 0;
};

struct PeriodicGaussianReciprocalSourceInventory {
    std::uint64_t fixed_source_storage_bytes = 0;
    std::uint64_t retained_context_storage_bytes = 0;
    std::uint64_t factory_workspace_storage_bytes = 0;
    std::uint64_t inventoried_fixed_storage_bytes = 0;
    std::uint64_t variable_owned_numeric_bytes = 0; // exactly zero
    // q != qbar: upper=4*Cq+2*Cqbar, actual=3*Cq+2*Cqbar+Nq.
    // q == qbar: reuse count/hash; upper=4*Cq, actual=3*Cq+Nq.
    // These count candidate traversal slots plus direct partner evaluations,
    // not CPU instructions, Fourier calls or an external callback's work.
    std::uint64_t candidate_evaluations_upper_bound = 0;
    std::uint64_t candidate_evaluations_performed = 0;
    std::uint64_t source_wire_bytes = 0;
    std::uint64_t conjugate_source_wire_bytes = 0;
};

class PeriodicGaussianReciprocalSource {
public:
    PeriodicGaussianReciprocalSource(const PeriodicGaussianReciprocalSource&) = delete;
    PeriodicGaussianReciprocalSource& operator=(const PeriodicGaussianReciprocalSource&) = delete;
    PeriodicGaussianReciprocalSource(PeriodicGaussianReciprocalSource&&) noexcept = default;
    PeriodicGaussianReciprocalSource& operator=(PeriodicGaussianReciprocalSource&&) = delete;
    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianReciprocalSourceVersion; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    bool reciprocal_conjugate_closure_certified() const noexcept { return static_cast<bool>(context_); }
    bool ao_image_source_certified() const noexcept { return false; }
    std::uint64_t q_index() const noexcept { return q_.index; }
    std::uint64_t conjugate_q_index() const noexcept { return conjugate_q_index_; }
    bool self_conjugate_transfer() const noexcept { return q_.self_conjugate; }
    const std::array<int, 3>& centered_doubled_numerator() const noexcept { return q_.centered_doubled_numerator; }
    const std::array<int, 3>& centered_reciprocal_wrap() const noexcept { return q_.centered_reciprocal_wrap; }
    const Eigen::Vector3d& q_fractional() const noexcept { return numeric_.geometry.q_fractional; }
    const Eigen::Vector3d& q_cartesian() const noexcept { return numeric_.q_cartesian; }
    const Eigen::Matrix3d& reciprocal_lattice() const noexcept { return numeric_.geometry.reciprocal_lattice; }
    double reciprocal_energy_cutoff() const;
    double maximum_reciprocal_radius() const noexcept { return numeric_.maximum_radius; }
    double radial_boundary_tolerance() const noexcept { return numeric_.boundary_tolerance; }
    double cell_volume_bohr3() const noexcept { return numeric_.geometry.cell_volume; }
    const std::array<std::int64_t, 3>& lower_bounds() const noexcept { return numeric_.geometry.lower_bounds; }
    const std::array<std::int64_t, 3>& upper_bounds() const noexcept { return numeric_.geometry.upper_bounds; }
    std::uint64_t candidate_count() const noexcept { return numeric_.enumeration.candidate_count; }
    std::uint64_t conjugate_candidate_count() const noexcept { return conjugate_candidate_count_; }
    std::uint64_t accepted_vector_count() const noexcept { return accepted_count_; }
    std::uint64_t zero_mode_excluded_count() const noexcept { return q_.gamma ? 1U : 0U; }
    std::uint64_t conjugacy_audited_vector_count() const noexcept { return audited_count_; }
    const PeriodicGaussianReciprocalSourceInventory& inventory() const noexcept { return inventory_; }
    std::string source_identity_sha256() const;
    std::string conjugate_source_identity_sha256() const;
    std::string source_context_identity_sha256() const;

private:
    PeriodicGaussianReciprocalSource() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    detail::PeriodicReciprocalNumericPlan numeric_;
    PeriodicGaussianTransferRecord q_;
    std::uint64_t conjugate_q_index_ = 0;
    std::uint64_t conjugate_candidate_count_ = 0;
    std::uint64_t accepted_count_ = 0;
    std::uint64_t audited_count_ = 0;
    std::array<char, 64> source_digest_{};
    std::array<char, 64> conjugate_digest_{};
    PeriodicGaussianReciprocalSourceInventory inventory_;
    friend PeriodicGaussianReciprocalSource make_periodic_gaussian_reciprocal_source(
        std::shared_ptr<const PeriodicGaussianSourceContext>, std::uint64_t,
        const PeriodicGaussianReciprocalSourceCaps&);
    friend std::uint64_t visit_periodic_gaussian_reciprocal_source(
        const PeriodicGaussianReciprocalSource&, std::uint64_t,
        detail::PeriodicReciprocalRecordCallback, void*);
};

// Both opposite-q plans are capped before traversal. Count and SHA passes
// retain no G-vector lists. Equal cardinalities and the exact injective
// nbar=-n-(fq+fqbar) map must preserve opposite p and identical p2/w bitwise.
// An asymmetric radial boundary fails closed; no cutoff/slack is changed.
// Only one source is returned, retaining a shared immutable context. The
// context's Gaussian basis identities are already authenticated; later
// Fourier consumers MUST recheck their borrowed BasisSet content separately.
PeriodicGaussianReciprocalSource make_periodic_gaussian_reciprocal_source(
    std::shared_ptr<const PeriodicGaussianSourceContext> context,
    std::uint64_t q_index, const PeriodicGaussianReciprocalSourceCaps& caps);

// Source wire: length-prefixed domain
// "vibeqc.periodic.gaussian-reciprocal-source", u32 source version, u32
// numerical-source version, u32 context version, context SHA string, u64 q;
// centered numerator and wrap (three i32 each), B row-major (nine binary64),
// Ecut/radius/slack/Omega (four binary64), lower/upper labels (three i64 each),
// candidate/accepted/zero-excluded counts (three u64), then accepted records
// in canonical order: i64[3] label and binary64[5] px/py/pz/p2/w.
// All integers big-endian, strings u64-length-prefixed, signed zeros +0.
// Own and opposite wires are independent (no circular digest), identical to
// constructing each q separately. Context identity binds actual original A,
// both bases and scientific policies; no state/schedule/allocation is hashed.
std::uint64_t periodic_gaussian_reciprocal_source_wire_bytes(std::uint64_t accepted_count);

// Cap before one traversal; immutable-source SHA is recomputed during replay
// and checked at completion. The callback borrows transient label/lanes, must
// not retain their references, and may throw. A failed callback/hash has no
// success count; external sink effects require caller-owned rollback. This
// is NOT a sink transaction commit. No callback wrapper or G list allocated.
std::uint64_t visit_periodic_gaussian_reciprocal_source(
    const PeriodicGaussianReciprocalSource& source,
    std::uint64_t maximum_candidates,
    detail::PeriodicReciprocalRecordCallback callback, void* user);

} // namespace vibeqc
