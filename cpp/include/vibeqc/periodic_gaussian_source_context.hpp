#pragma once

// Density-independent inputs for the finite Gaussian electron-electron
// source. This is a compiled-policy context, NOT a per-q enumerated-source
// manifest, numerical factor payload, nuclear/Hcore certificate, or SCF state.
// Sun et al., JCP 147, 164119 (2017), doi:10.1063/1.4998644, Eqs. 3, 13,
// 16, 21: positive ket Bloch phase, q=ket-bra, and principal metric whitening.
// Only 3D Gamma meshes and the explicitly finite all-reciprocal/image policy
// are supported. No tail error, correlation auxiliary quality, or production
// DLPNO accuracy is certified by authenticating these inputs.

#include <array>
#include <cstdint>
#include <string>

#include "vibeqc/basis.hpp"
#include "vibeqc/kmesh_address.hpp"
#include "vibeqc/periodic.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicGaussianSourceContextVersion = 1U;
inline constexpr char kPeriodicGaussianSourceGeometryPolicy[] =
    "direct-columns;reciprocal=PeriodicSystem::reciprocal_lattice;"
    "2*pi*inverse(direct).transpose;binary64-v1";
inline constexpr char kPeriodicGaussianSourceHamiltonianPolicy[] =
    "3d;full-coulomb-all-reciprocal;gamma-mesh;G0-omit;exxdiv=None;"
    "short-range=disabled;integral-screening=none;positive-ket-phase;"
    "q=ket-bra;centered-half-open-negative-Nyquist;"
    "original-auxiliary-principal-Hermitian-pseudoinverse-root;"
    "strict-eigenvalue-greater-than-rank-cutoff;finite-image-reference-v1";
inline constexpr char kPeriodicGaussianSourceReciprocalPolicy[] =
    "p=B*(n+centered-q);lexicographic-n;fixed-reverse-index-fma;"
    "Gamma-p2<=2*Ecut;nonGamma-radius=sqrt(2*Ecut)+"
    "128*epsilon*max(sqrt(2*Ecut)+norm(q),1);"
    "weight=(4*pi/Omega)/p2;Gamma-zero-by-integer-label;"
    "Omega=(2*pi/maxabs(B))^3/abs(det_fma(B/maxabs(B)));"
    "per-q-enumeration-and-conjugate-closure-audits-required-v1";

struct PeriodicGaussianSourceOptions {
    // Required finite positive controls; zero is deliberately not a default.
    double reciprocal_energy_cutoff = 0.0;
    double ao_pair_image_cutoff_bohr = 0.0;
    double metric_absolute_eigenvalue_threshold = 0.0;
    double metric_negative_tolerance = 0.0; // <= eigenvalue threshold
};

struct PeriodicGaussianSourceCaps {
    // All caps must be positive. Counts below combine both basis roles even
    // if the caller passes the same BasisSet object for AO and auxiliary.
    std::uint64_t maximum_context_storage_bytes = 0;
    std::uint64_t maximum_kpoint_count = 0;
    std::uint64_t maximum_shell_count = 0;
    std::uint64_t maximum_contraction_count = 0;
    std::uint64_t maximum_primitive_numeric_lanes = 0;
    std::uint64_t maximum_basis_content_wire_bytes = 0;
    std::uint64_t maximum_borrowed_active_numeric_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicGaussianBasisInventory {
    std::uint64_t function_count = 0;
    std::uint64_t shell_count = 0;
    std::uint64_t contraction_count = 0;
    std::uint64_t exponent_count = 0; // once per libint shell
    std::uint64_t coefficient_count = 0; // once per contraction
    // Exact active numeric payload, NOT allocator capacity/owner/control:
    // 8*(3*shell_count + exponent_count + coefficient_count).
    std::uint64_t borrowed_active_numeric_bytes = 0;
    // Existing content-digest v1 wire: 75 + 8*S + 45*C + 16*P_c.
    std::uint64_t content_wire_bytes = 0;
};

struct PeriodicGaussianSourceInventory {
    std::uint64_t fixed_context_storage_bytes = 0; // sizeof(native context)
    std::uint64_t variable_owned_numeric_bytes = 0; // exactly zero
    PeriodicGaussianBasisInventory ao;
    PeriodicGaussianBasisInventory auxiliary;
    std::uint64_t combined_borrowed_active_numeric_bytes = 0;
    // Conservative abstract work bound, not a timing/FLOP promise:
    // 128*(combined content wire bytes) + 64*(S+C) + 4096.
    // Metadata is capped incrementally before full finite/hash scans.
    std::uint64_t work_units_upper_bound = 0;
};

struct PeriodicGaussianKRecord {
    std::uint64_t index = 0;
    std::array<int, 3> modular_doubled_address = {0, 0, 0};
    std::array<double, 3> fractional = {0.0, 0.0, 0.0};
    std::array<double, 3> cartesian = {0.0, 0.0, 0.0};
};

struct PeriodicGaussianTransferRecord {
    std::uint64_t index = 0;
    std::array<int, 3> modular_doubled_address = {0, 0, 0};
    std::array<int, 3> centered_doubled_numerator = {0, 0, 0};
    // modular_fractional = centered_fractional + reciprocal_wrap exactly.
    std::array<int, 3> centered_reciprocal_wrap = {0, 0, 0};
    std::array<double, 3> fractional = {0.0, 0.0, 0.0};
    std::array<double, 3> cartesian = {0.0, 0.0, 0.0};
    bool gamma = false;
    bool self_conjugate = false;
};

class PeriodicGaussianSourceContext {
public:
    PeriodicGaussianSourceContext(const PeriodicGaussianSourceContext&) = delete;
    PeriodicGaussianSourceContext& operator=(const PeriodicGaussianSourceContext&) = delete;
    PeriodicGaussianSourceContext(PeriodicGaussianSourceContext&&) noexcept = default;
    PeriodicGaussianSourceContext& operator=(PeriodicGaussianSourceContext&&) = delete;

    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianSourceContextVersion; }
    bool density_independent() const noexcept { return true; }
    bool enumerated_source_certified() const noexcept { return false; }
    bool ao_image_source_certified() const noexcept { return false; }
    bool nuclear_hcore_certified() const noexcept { return false; }
    const Eigen::Matrix3d& direct_lattice() const noexcept { return direct_; }
    const Eigen::Matrix3d& reciprocal_lattice() const noexcept { return reciprocal_; }
    const RegularKMesh& mesh() const noexcept { return mesh_; }
    const PeriodicGaussianSourceOptions& options() const noexcept { return options_; }
    const PeriodicGaussianSourceInventory& inventory() const noexcept { return inventory_; }
    std::string ao_basis_identity_sha256() const;
    std::string auxiliary_basis_identity_sha256() const;
    std::string reciprocal_producer_identity_sha256() const;
    std::string factorization_backend_identity_sha256() const;
    std::string source_context_identity_sha256() const;
    // Native allocation-free immutable receipt view for downstream owner
    // checks. The context must remain alive; no mutable storage is exposed.
    const std::array<char, 64>& source_context_identity_ascii() const noexcept { return context_digest_; }
    PeriodicGaussianKRecord k_record(std::uint64_t index) const;
    PeriodicGaussianTransferRecord transfer_record(std::uint64_t index) const;
    std::uint64_t ket_index(std::uint64_t bra_index, std::uint64_t q_index) const;
    // The context does not retain BasisSet pointers. Every later numerical
    // builder must revalidate its borrowed physical bases using this seam.
    // This checks content, not name, and returns the new per-role inventory.
    PeriodicGaussianSourceInventory verify_bases(
        const BasisSet& ao, const BasisSet& auxiliary,
        const PeriodicGaussianSourceCaps& caps) const;

private:
    PeriodicGaussianSourceContext() = default;
    Eigen::Matrix3d direct_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d reciprocal_ = Eigen::Matrix3d::Zero();
    RegularKMesh mesh_{{1, 1, 1}};
    PeriodicGaussianSourceOptions options_;
    PeriodicGaussianSourceInventory inventory_;
    std::array<char, 64> ao_digest_{};
    std::array<char, 64> auxiliary_digest_{};
    std::array<char, 64> producer_digest_{};
    std::array<char, 64> backend_digest_{};
    std::array<char, 64> context_digest_{};
    friend PeriodicGaussianSourceContext make_periodic_gaussian_source_context(
        const PeriodicSystem&, const BasisSet&, const BasisSet&,
        const RegularKMesh&, const PeriodicGaussianSourceOptions&,
        const PeriodicGaussianSourceCaps&);
};

// Only system.dim/lattice are read. Atom list, charge, multiplicity and space
// group are NOT electron-electron source inputs; basis centers are hashed.
// Both A and derived B are sealed, with columns as lattice vectors and
// A^T B=2*pi I checked scale-aware. No all-k/all-q table is constructed.
// Context retains only constant-size fields: zero variable numerical heap.
// Existing SHA helpers return constant-length strings transiently; string
// allocator/control overhead and ordinary scalar stack are not reported as
// numerical payload. Caller must separately budget full borrowed basis owner
// capacities, allocator overhead, Python copies and other live objects.
//
// Identity wire: length-prefixed UTF-8 domain
// "vibeqc.periodic.gaussian-source-context", u32 version and dimension;
// geometry, Hamiltonian, finite-image and reciprocal policy strings; u32
// reciprocal-source contract version and basis-digest version;
// AO/aux/producer/backend SHA strings; mesh and shift (three u32
// each); A and B row-major (nine binary64 each); options in declaration order;
// AO/aux function counts (u64 each). All integers big-endian; finite doubles
// canonicalize signed zero. Caps/inventory are deliberately not scientific
// identity fields. The address and fixed FMA Cartesian policy is versioned
// by this contract. No per-q accepted vector/count is claimed or hashed here.
PeriodicGaussianSourceContext make_periodic_gaussian_source_context(
    const PeriodicSystem& system, const BasisSet& ao, const BasisSet& auxiliary,
    const RegularKMesh& mesh, const PeriodicGaussianSourceOptions& options,
    const PeriodicGaussianSourceCaps& caps);

} // namespace vibeqc
