#include "vibeqc/kmesh_address.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace vibeqc {

namespace {

std::string triple(const std::array<int, 3>& v) {
    return "(" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", " +
           std::to_string(v[2]) + ")";
}

// N_0 * N_1 * N_2 without wrapping. Carried in std::uint64_t and compared
// against the destination type's maximum before every multiply, so an
// oversized mesh is rejected while it is still three integers -- no
// consumer ever gets a wrapped count to size a container from.
std::size_t checked_mesh_size(const std::array<int, 3>& mesh) {
    constexpr std::uint64_t limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    std::uint64_t total = 1;
    for (int d = 0; d < 3; ++d) {
        const std::uint64_t n = static_cast<std::uint64_t>(mesh[d]);
        if (total > limit / n) {
            throw std::runtime_error(
                "RegularKMesh: mesh " + triple(mesh) +
                " has more points than std::size_t can count");
        }
        total *= n;
    }
    return static_cast<std::size_t>(total);
}

}  // namespace

// A transfer momentum must not be usable where a k-point is expected, and
// vice versa: on a shifted mesh their doubled parities differ, so the
// substitution names a point that is not on the grid. The separation is
// carried by the type system and therefore has to be asserted at compile
// time -- there is no runtime call that could fail for it.
static_assert(!std::is_convertible<KTransferAddress, KGridAddress>::value,
              "KTransferAddress must not convert to KGridAddress");
static_assert(!std::is_convertible<KGridAddress, KTransferAddress>::value,
              "KGridAddress must not convert to KTransferAddress");

RegularKMesh::RegularKMesh(std::array<int, 3> mesh,
                           std::array<int, 3> is_shift) {
    for (int d = 0; d < 3; ++d) {
        if (mesh[d] < 1) {
            throw std::runtime_error(
                "RegularKMesh: mesh " + triple(mesh) +
                " must have strictly positive divisions on every axis");
        }
        if (mesh[d] > max_divisions) {
            throw std::runtime_error(
                "RegularKMesh: mesh " + triple(mesh) + " exceeds the " +
                std::to_string(max_divisions) +
                " divisions per axis the doubled address representation "
                "supports");
        }
        if (is_shift[d] != 0 && is_shift[d] != 1) {
            throw std::runtime_error(
                "RegularKMesh: is_shift " + triple(is_shift) +
                " must be 0 or 1 on every axis; a shift is a half-step "
                "flag, not a displacement");
        }
    }

    size_ = checked_mesh_size(mesh);
    mesh_ = mesh;
    is_shift_ = is_shift;
    for (int d = 0; d < 3; ++d) {
        doubled_modulus_[d] = 2 * mesh[d];
    }
}

void RegularKMesh::require_index(std::size_t index, const char* context) const {
    if (index >= size_) {
        throw std::runtime_error(
            std::string(context) + ": index " + std::to_string(index) +
            " is outside the " + std::to_string(size_) +
            " points of mesh " + triple(mesh_));
    }
}

void RegularKMesh::require_grid(const KGridAddress& k,
                                const char* context) const {
    for (int d = 0; d < 3; ++d) {
        const int a = k.doubled[d];
        if (a < 0 || a >= doubled_modulus_[d]) {
            throw std::runtime_error(
                std::string(context) + ": doubled address " +
                triple(k.doubled) + " is not reduced into [0, " +
                triple(doubled_modulus_) + ")");
        }
        if ((a % 2) != is_shift_[d]) {
            throw std::runtime_error(
                std::string(context) + ": doubled address " +
                triple(k.doubled) + " has axis " + std::to_string(d) +
                " parity " + std::to_string(a % 2) + ", but is_shift " +
                triple(is_shift_) + " puts every k-point of this mesh at "
                "parity " + std::to_string(is_shift_[d]) +
                ". A transfer momentum q = k_j - k_i is always even and "
                "belongs to the transfer group, not here");
        }
    }
}

void RegularKMesh::require_transfer(const KTransferAddress& q,
                                    const char* context) const {
    for (int d = 0; d < 3; ++d) {
        const int a = q.doubled[d];
        if (a < 0 || a >= doubled_modulus_[d]) {
            throw std::runtime_error(
                std::string(context) + ": doubled address " +
                triple(q.doubled) + " is not reduced into [0, " +
                triple(doubled_modulus_) + ")");
        }
        if ((a % 2) != 0) {
            throw std::runtime_error(
                std::string(context) + ": doubled address " +
                triple(q.doubled) + " is odd on axis " + std::to_string(d) +
                "; a transfer momentum lives on the zero-shift mesh and is "
                "even on every axis whatever is_shift is");
        }
    }
}

std::array<int, 3> RegularKMesh::divisions(std::size_t index) const {
    // index = (m_0 * N_1 + m_1) * N_2 + m_2, last axis fastest.
    const std::size_t n1 = static_cast<std::size_t>(mesh_[1]);
    const std::size_t n2 = static_cast<std::size_t>(mesh_[2]);
    const std::size_t m2 = index % n2;
    const std::size_t rest = index / n2;
    const std::size_t m1 = rest % n1;
    const std::size_t m0 = rest / n1;
    return {static_cast<int>(m0), static_cast<int>(m1), static_cast<int>(m2)};
}

void RegularKMesh::reduce(const std::array<std::int64_t, 3>& raw,
                          std::array<int, 3>& reduced,
                          Eigen::Vector3i& wrap) const {
    for (int d = 0; d < 3; ++d) {
        const std::int64_t m = static_cast<std::int64_t>(doubled_modulus_[d]);
        std::int64_t r = raw[d] % m;
        if (r < 0) {
            r += m;
        }
        // Exact by construction: raw - r is a multiple of m, so the
        // division is not a rounding but an identity.
        const std::int64_t g = (raw[d] - r) / m;
        reduced[d] = static_cast<int>(r);
        wrap[d] = static_cast<int>(g);
    }
}

KGridAddress RegularKMesh::address(std::size_t index) const {
    require_index(index, "RegularKMesh::address");
    const std::array<int, 3> m = divisions(index);
    KGridAddress k;
    for (int d = 0; d < 3; ++d) {
        k.doubled[d] = 2 * m[d] + is_shift_[d];
    }
    return k;
}

std::size_t RegularKMesh::index(const KGridAddress& k) const {
    require_grid(k, "RegularKMesh::index");
    std::size_t out = 0;
    for (int d = 0; d < 3; ++d) {
        const int m = (k.doubled[d] - is_shift_[d]) / 2;
        out = out * static_cast<std::size_t>(mesh_[d]) +
              static_cast<std::size_t>(m);
    }
    return out;
}

KTransferAddress RegularKMesh::transfer_address(std::size_t index) const {
    require_index(index, "RegularKMesh::transfer_address");
    const std::array<int, 3> m = divisions(index);
    KTransferAddress q;
    for (int d = 0; d < 3; ++d) {
        q.doubled[d] = 2 * m[d];
    }
    return q;
}

std::size_t RegularKMesh::transfer_index(const KTransferAddress& q) const {
    require_transfer(q, "RegularKMesh::transfer_index");
    std::size_t out = 0;
    for (int d = 0; d < 3; ++d) {
        out = out * static_cast<std::size_t>(mesh_[d]) +
              static_cast<std::size_t>(q.doubled[d] / 2);
    }
    return out;
}

Eigen::Vector3d RegularKMesh::fractional(const KGridAddress& k) const {
    require_grid(k, "RegularKMesh::fractional");
    Eigen::Vector3d f;
    for (int d = 0; d < 3; ++d) {
        f[d] = static_cast<double>(k.doubled[d]) /
               static_cast<double>(doubled_modulus_[d]);
    }
    return f;
}

Eigen::Vector3d RegularKMesh::fractional(const KTransferAddress& q) const {
    require_transfer(q, "RegularKMesh::fractional");
    Eigen::Vector3d f;
    for (int d = 0; d < 3; ++d) {
        f[d] = static_cast<double>(q.doubled[d]) /
               static_cast<double>(doubled_modulus_[d]);
    }
    return f;
}

Eigen::Vector3d RegularKMesh::fractional_at(std::size_t index) const {
    require_index(index, "RegularKMesh::fractional_at");
    return fractional(address(index));
}

WrappedKGridAddress RegularKMesh::negate_with_wrap(
    const KGridAddress& k) const {
    require_grid(k, "RegularKMesh::negate");
    const std::array<std::int64_t, 3> raw = {
        -static_cast<std::int64_t>(k.doubled[0]),
        -static_cast<std::int64_t>(k.doubled[1]),
        -static_cast<std::int64_t>(k.doubled[2]),
    };
    WrappedKGridAddress out;
    reduce(raw, out.address.doubled, out.wrap);
    return out;
}

KGridAddress RegularKMesh::negate(const KGridAddress& k) const {
    return negate_with_wrap(k).address;
}

WrappedKTransferAddress RegularKMesh::negate_with_wrap(
    const KTransferAddress& q) const {
    require_transfer(q, "RegularKMesh::negate");
    const std::array<std::int64_t, 3> raw = {
        -static_cast<std::int64_t>(q.doubled[0]),
        -static_cast<std::int64_t>(q.doubled[1]),
        -static_cast<std::int64_t>(q.doubled[2]),
    };
    WrappedKTransferAddress out;
    reduce(raw, out.address.doubled, out.wrap);
    return out;
}

KTransferAddress RegularKMesh::negate(const KTransferAddress& q) const {
    return negate_with_wrap(q).address;
}

WrappedKTransferAddress RegularKMesh::transfer_with_wrap(
    const KGridAddress& ki, const KGridAddress& kj) const {
    require_grid(ki, "RegularKMesh::transfer (ki)");
    require_grid(kj, "RegularKMesh::transfer (kj)");
    std::array<std::int64_t, 3> raw;
    for (int d = 0; d < 3; ++d) {
        raw[d] = static_cast<std::int64_t>(kj.doubled[d]) -
                 static_cast<std::int64_t>(ki.doubled[d]);
    }
    WrappedKTransferAddress out;
    reduce(raw, out.address.doubled, out.wrap);
    return out;
}

KTransferAddress RegularKMesh::transfer(const KGridAddress& ki,
                                        const KGridAddress& kj) const {
    return transfer_with_wrap(ki, kj).address;
}

WrappedKGridAddress RegularKMesh::add_transfer_with_wrap(
    const KGridAddress& ki,
    const KTransferAddress& q) const {
    require_grid(ki, "RegularKMesh::add_transfer (ki)");
    require_transfer(q, "RegularKMesh::add_transfer (q)");
    std::array<std::int64_t, 3> raw;
    for (int d = 0; d < 3; ++d) {
        raw[d] = static_cast<std::int64_t>(ki.doubled[d])
            + static_cast<std::int64_t>(q.doubled[d]);
    }
    WrappedKGridAddress out;
    reduce(raw, out.address.doubled, out.wrap);
    return out;
}

KGridAddress RegularKMesh::add_transfer(
    const KGridAddress& ki,
    const KTransferAddress& q) const {
    return add_transfer_with_wrap(ki, q).address;
}

WrappedKGridAddress RegularKMesh::conserved_with_wrap(
    const KGridAddress& ki,
    const KGridAddress& ka,
    const KGridAddress& kj) const {
    require_grid(ki, "RegularKMesh::conserved (ki)");
    require_grid(ka, "RegularKMesh::conserved (ka)");
    require_grid(kj, "RegularKMesh::conserved (kj)");
    std::array<std::int64_t, 3> raw;
    for (int d = 0; d < 3; ++d) {
        raw[d] = static_cast<std::int64_t>(ki.doubled[d]) -
                 static_cast<std::int64_t>(ka.doubled[d]) +
                 static_cast<std::int64_t>(kj.doubled[d]);
    }
    WrappedKGridAddress out;
    reduce(raw, out.address.doubled, out.wrap);
    return out;
}

KGridAddress RegularKMesh::conserved(const KGridAddress& ki,
                                     const KGridAddress& ka,
                                     const KGridAddress& kj) const {
    return conserved_with_wrap(ki, ka, kj).address;
}

// ---- on-demand index lookups --------------------------------------------
//
// Written directly on the per-axis divisions rather than by composing
// address() and index(). The shift drops out of both maps below --
// (2 m_j + s) - (2 m_i + s) = 2 (m_j - m_i) and
// (2 m_i + s) - (2 m_a + s) + (2 m_j + s) = 2 (m_i - m_a + m_j) + s -- so
// the index arithmetic is the same for every shift, and the parity that
// distinguishes the two groups is restored only when an address is built.

std::size_t RegularKMesh::negate_index(std::size_t index) const {
    require_index(index, "RegularKMesh::negate_index");
    // Not shift-independent: -(2m) reduces to 2((N - m) mod N), while
    // -(2m + 1) reduces to 2(N - 1 - m) + 1.
    return this->index(negate(address(index)));
}

std::size_t RegularKMesh::transfer_index(std::size_t ki,
                                         std::size_t kj) const {
    require_index(ki, "RegularKMesh::transfer_index (ki)");
    require_index(kj, "RegularKMesh::transfer_index (kj)");
    const std::array<int, 3> mi = divisions(ki);
    const std::array<int, 3> mj = divisions(kj);
    std::size_t out = 0;
    for (int d = 0; d < 3; ++d) {
        int m = (mj[d] - mi[d]) % mesh_[d];
        if (m < 0) {
            m += mesh_[d];
        }
        out = out * static_cast<std::size_t>(mesh_[d]) +
              static_cast<std::size_t>(m);
    }
    return out;
}

std::size_t RegularKMesh::add_transfer_index(std::size_t ki,
                                             std::size_t q) const {
    require_index(ki, "RegularKMesh::add_transfer_index (ki)");
    require_index(q, "RegularKMesh::add_transfer_index (q)");
    const std::array<int, 3> mi = divisions(ki);
    const std::array<int, 3> mq = divisions(q);
    std::size_t out = 0;
    for (int d = 0; d < 3; ++d) {
        const int m = (mi[d] + mq[d]) % mesh_[d];
        out = out * static_cast<std::size_t>(mesh_[d])
            + static_cast<std::size_t>(m);
    }
    return out;
}

std::size_t RegularKMesh::conserved_index(std::size_t ki,
                                          std::size_t ka,
                                          std::size_t kj) const {
    require_index(ki, "RegularKMesh::conserved_index (ki)");
    require_index(ka, "RegularKMesh::conserved_index (ka)");
    require_index(kj, "RegularKMesh::conserved_index (kj)");
    const std::array<int, 3> mi = divisions(ki);
    const std::array<int, 3> ma = divisions(ka);
    const std::array<int, 3> mj = divisions(kj);
    std::size_t out = 0;
    for (int d = 0; d < 3; ++d) {
        // |m_i - m_a + m_j| < 2 N_d <= 2 * max_divisions < INT_MAX, so the
        // sum is representable before the reduction.
        int m = (mi[d] - ma[d] + mj[d]) % mesh_[d];
        if (m < 0) {
            m += mesh_[d];
        }
        out = out * static_cast<std::size_t>(mesh_[d]) +
              static_cast<std::size_t>(m);
    }
    return out;
}

}  // namespace vibeqc
