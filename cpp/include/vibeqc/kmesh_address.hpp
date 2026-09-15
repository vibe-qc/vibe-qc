// Exact addressing and crystal-momentum algebra on a regular k-mesh.
//
// Scope. This is the integer foundation the periodic correlated methods
// stand on. It owns exactly two things: the bijection between a linear
// k-index and the grid point that index names, and the modular algebra
// crystal-momentum conservation is written in. It knows nothing about
// lattices, basis functions, symmetry, density fitting, or SCF; the only
// floating-point arithmetic it performs is the final conversion of an
// address to a fractional coordinate.
//
// Deliberately out of scope: symmetry reduction (that is spglib's job, see
// crystal.hpp), Cartesian k-vectors (multiply a fractional coordinate by
// PeriodicSystem::reciprocal_lattice(), as bloch.cpp does), and any table
// indexed by more than one k at a time.
//
//
// Doubled addresses
// -----------------
// A regular mesh with divisions N_d and per-axis shift s_d in {0, 1} places
// its points at
//
//     k_frac[d] = (m_d + s_d/2) / N_d,      m_d = 0 ... N_d - 1.
//
// Multiplying through by 2 N_d makes every coordinate an integer:
//
//     a_d = 2 m_d + s_d   (mod M_d),        M_d = 2 N_d,
//
// with k_frac[d] = a_d / M_d exactly. Crystal momentum is defined modulo a
// reciprocal-lattice vector, i.e. modulo 1 in fractional coordinates, i.e.
// modulo M_d in a_d. Momentum algebra is therefore integer arithmetic in
// Z_{M_0} x Z_{M_1} x Z_{M_2} and is exact by construction: no tolerance,
// no nearest-neighbour search, no hash of a rounded float.
//
// The doubling is what makes a shifted grid representable at all. With
// s_d = 1 the coordinate is a half-integer multiple of 1/N_d, which an
// undoubled index cannot name. It also fixes the parity of a_d at s_d for
// every point of the grid, which is why a k address and a transfer-momentum
// address are separate types below.
//
// The float-keyed alternative -- rounding k_i - k_a + k_j to a fixed number
// of decimals and looking the key up in a hash of the k list -- can serve a
// general, possibly irregular character mesh. For a regular grid it is both
// slower and strictly less exact, so production regular-grid consumers use
// this module instead.
//
//
// Ordering
// --------
// The linear index is last-axis-fast,
//
//     index = (m_0 * N_1 + m_1) * N_2 + m_2,
//
// which is the order bloch.cpp's monkhorst_pack() has always emitted and
// the order spglib's grid addresses use. Every consumer of a BlochKMesh
// already depends on it; it is a fixed contract, not an implementation
// detail.
//
//
// Monkhorst-Pack convention: the half-step offset is the EVEN-mesh case
// --------------------------------------------------------------------
// Monkhorst & Pack, Phys. Rev. B 13, 5188 (1976),
// doi:10.1103/PhysRevB.13.5188, Eq. (3) defines the per-axis sequence
//
//     u_r = (2 r - q - 1) / (2 q),          r = 1, 2, ..., q,
//
// so the doubled numerators 2r - q - 1 run over q integers spaced by 2 and
// centred on 0. For ODD q that set contains 0: the classical mesh already
// contains Gamma and coincides with this module's s = 0 grid (each point up
// to a reciprocal-lattice wrap). For EVEN q it does not contain 0: the
// classical mesh sits half a step off Gamma and coincides with this
// module's s = 1 grid.
//
// So s_d = 1 reproduces the classical Monkhorst-Pack set on an even mesh,
// not on an odd one. The comments this header replaces had that backwards.
// vibe-qc's own project reference mesh is explicitly Gamma-centred --
// mesh = (8, 8, 8), is_shift = (0, 0, 0) -- and s_d = 1 is supported per
// axis for callers that want the classical even-mesh set or a staggered
// grid. See python/vibeqc/periodic_runner.py's KMESH_CONVENTION_* strings
// and tests/test_kmesh_convention_provenance.py for the user-facing half of
// this, and docs/user_guide/ for the narrative.
//
//
// Crystal-momentum conservation
// -----------------------------
// Translational symmetry forces every amplitude of a periodic correlated
// method to conserve crystal momentum up to a reciprocal-lattice vector G.
// For the doubles amplitude t^{a k_a, b k_b}_{i k_i, j k_j},
//
//     k_a + k_b - k_i - k_j = G
//
// (McClain, Sun, Chan & Berkelbach, J. Chem. Theory Comput. 13, 1209
// (2017), doi:10.1021/acs.jctc.7b00049, text below Eq. (26)), which fixes
// the fourth index and is what removes one factor of N_k from the cost.
// This module writes the same statement as
//
//     k_i - k_a + k_j = k_b + G,
//
// with G an integer vector in fractional reciprocal-lattice units. The G
// here is the negative of the G in the sentence above; both are
// reciprocal-lattice vectors and the sign is only a labelling choice, but
// the one this module returns is the one that makes every reduction read
// the same way:
//
//     <exact integer combination, as a fraction> = <reduced address> + G.
//
// Callers whose quantities are G-periodic can ignore G. Callers carrying an
// explicit e^{i G.r} factor (structure factors, AO-pair Fourier transforms,
// anything that lands on a plane-wave grid) need it, which is why every
// reduction here can return it exactly.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vibeqc {

// Which group an address belongs to. The distinction is not cosmetic: on a
// shifted mesh a k address has odd doubled components and a transfer
// address has even ones, so silently passing one where the other is
// expected names a point that is not on the grid at all.
enum class KAddressKind {
    // A point of the k-grid itself: doubled component parity == is_shift[d].
    GRID,
    // A transfer momentum q = k_j - k_i. The shift cancels in the
    // difference, so q always lives on the ZERO-shift mesh of the same
    // divisions and every doubled component is even, whatever is_shift is.
    TRANSFER,
};

// A point of Z_{M_0} x Z_{M_1} x Z_{M_2}, tagged by the group it lives in.
// ``doubled[d]`` is the reduced doubled coordinate a_d, always in
// [0, M_d). Two addresses of different kinds are different C++ types and do
// not convert into one another.
template <KAddressKind Kind>
struct ModularKAddress {
    static constexpr KAddressKind kind = Kind;

    std::array<int, 3> doubled = {0, 0, 0};

    bool operator==(const ModularKAddress& other) const noexcept {
        return doubled == other.doubled;
    }
    bool operator!=(const ModularKAddress& other) const noexcept {
        return !(*this == other);
    }
};

using KGridAddress = ModularKAddress<KAddressKind::GRID>;
using KTransferAddress = ModularKAddress<KAddressKind::TRANSFER>;

// A reduction result: the reduced address plus the reciprocal-lattice
// vector that was subtracted to get there, in integer fractional units.
// ``wrap`` is exact -- it is an integer, never a rounded float.
template <KAddressKind Kind>
struct WrappedKAddress {
    ModularKAddress<Kind> address;
    Eigen::Vector3i wrap = Eigen::Vector3i::Zero();
};

using WrappedKGridAddress = WrappedKAddress<KAddressKind::GRID>;
using WrappedKTransferAddress = WrappedKAddress<KAddressKind::TRANSFER>;

// A validated regular k-mesh descriptor plus its exact address algebra.
//
// Construction validates and never allocates: the object is three small
// integer triples and a size. Every mesh whose point count would overflow
// std::size_t is rejected at construction, before any consumer has a chance
// to size a container from it.
class RegularKMesh {
public:
    // Largest per-axis division count. Bounded so the doubled modulus
    // 2 * N_d still fits an int; the intermediate sums of the momentum
    // algebra are carried in std::int64_t and cannot overflow within it.
    static constexpr int max_divisions =
        (std::numeric_limits<int>::max() - 1) / 2;

    // Throws std::runtime_error if any mesh[d] < 1 or > max_divisions, if
    // any is_shift[d] is outside {0, 1}, or if the product of the divisions
    // exceeds std::size_t.
    RegularKMesh(std::array<int, 3> mesh, std::array<int, 3> is_shift);

    // Gamma-centred mesh: the project reference convention.
    explicit RegularKMesh(std::array<int, 3> mesh)
        : RegularKMesh(mesh, {0, 0, 0}) {}

    const std::array<int, 3>& mesh() const noexcept { return mesh_; }
    const std::array<int, 3>& is_shift() const noexcept { return is_shift_; }

    // M_d = 2 * N_d, the modulus each doubled coordinate lives in.
    const std::array<int, 3>& doubled_modulus() const noexcept {
        return doubled_modulus_;
    }

    // N_k = N_0 * N_1 * N_2. Also the number of distinct transfer momenta.
    std::size_t size() const noexcept { return size_; }

    // ---- index <-> address, an exact bijection --------------------------

    // Grid point named by ``index``, which must be < size().
    KGridAddress address(std::size_t index) const;

    // Inverse of address(). ``k`` must be reduced and carry this mesh's
    // parity; both are checked.
    std::size_t index(const KGridAddress& k) const;

    // The same bijection for the transfer group. transfer_address(i) and
    // the zero-shift mesh's address(i) carry the same doubled triple, which
    // is the sense in which a transfer momentum "is" a zero-shift k-point.
    KTransferAddress transfer_address(std::size_t index) const;
    std::size_t transfer_index(const KTransferAddress& q) const;

    // ---- fractional coordinates -----------------------------------------

    // a_d / M_d, exactly the value bloch.cpp's monkhorst_pack() emits for
    // the same grid point: numerator and denominator are both twice the
    // historical (m_d + s_d/2) / N_d form, the real quotient is unchanged,
    // and IEEE-754 division is correctly rounded, so the double is
    // bit-identical rather than merely close.
    Eigen::Vector3d fractional(const KGridAddress& k) const;
    Eigen::Vector3d fractional(const KTransferAddress& q) const;

    // fractional(address(index)), the form the mesh builder wants.
    Eigen::Vector3d fractional_at(std::size_t index) const;

    // ---- exact modular algebra -------------------------------------------

    // -k. Unlike the two operations below this one does depend on the
    // shift: on a zero-shift mesh the partner of m is (N - m) mod N, on a
    // shifted mesh it is N - 1 - m.
    KGridAddress negate(const KGridAddress& k) const;
    WrappedKGridAddress negate_with_wrap(const KGridAddress& k) const;
    KTransferAddress negate(const KTransferAddress& q) const;
    WrappedKTransferAddress negate_with_wrap(const KTransferAddress& q) const;

    // Transfer momentum q = k_j - k_i. The shift cancels, so both the
    // reduced q and its wrap are independent of is_shift.
    KTransferAddress transfer(const KGridAddress& ki,
                              const KGridAddress& kj) const;
    WrappedKTransferAddress transfer_with_wrap(const KGridAddress& ki,
                                               const KGridAddress& kj) const;

    // Translate one grid point by a transfer momentum: k_j = k_i + q - G.
    // The transfer has even doubled parity, so the result remains on this
    // grid for both Gamma-centred and half-shifted meshes.  The returned wrap
    // satisfies k_i + q = k_j + G exactly.
    KGridAddress add_transfer(const KGridAddress& ki,
                              const KTransferAddress& q) const;
    WrappedKGridAddress add_transfer_with_wrap(
        const KGridAddress& ki,
        const KTransferAddress& q) const;

    // The conserved fourth index: k_b with k_i - k_a + k_j = k_b + G. The
    // shift cancels here too -- it appears three times with a net single
    // power, which is exactly the parity k_b carries.
    KGridAddress conserved(const KGridAddress& ki,
                           const KGridAddress& ka,
                           const KGridAddress& kj) const;
    WrappedKGridAddress conserved_with_wrap(const KGridAddress& ki,
                                            const KGridAddress& ka,
                                            const KGridAddress& kj) const;

    // ---- on-demand index lookups ----------------------------------------
    //
    // These are the production entry points for a correlated method. They
    // are O(1) in time and O(1) in memory: nothing here materialises a
    // table over pairs or triples of k. An N_k^3 conservation table is a
    // sound idea for a handful of k-points and a bad one past that -- at
    // the project reference mesh (8, 8, 8) it is 512^3 = 134,217,728
    // entries, over a gigabyte at 8 bytes each, to store a number three
    // integer divisions can produce on the spot.

    std::size_t negate_index(std::size_t index) const;
    std::size_t transfer_index(std::size_t ki, std::size_t kj) const;
    std::size_t add_transfer_index(std::size_t ki,
                                   std::size_t q) const;
    std::size_t conserved_index(std::size_t ki,
                                std::size_t ka,
                                std::size_t kj) const;

private:
    // Per-axis divisions (m_0, m_1, m_2) of a linear index.
    std::array<int, 3> divisions(std::size_t index) const;

    // Reduce a raw doubled triple into [0, M_d) and report the exact
    // integer wrap: raw_d = reduced_d + M_d * wrap_d.
    void reduce(const std::array<std::int64_t, 3>& raw,
                std::array<int, 3>& reduced,
                Eigen::Vector3i& wrap) const;

    void require_grid(const KGridAddress& k, const char* context) const;
    void require_transfer(const KTransferAddress& q, const char* context) const;
    void require_index(std::size_t index, const char* context) const;

    std::array<int, 3> mesh_ = {1, 1, 1};
    std::array<int, 3> is_shift_ = {0, 0, 0};
    std::array<int, 3> doubled_modulus_ = {2, 2, 2};
    std::size_t size_ = 1;
};

}  // namespace vibeqc
