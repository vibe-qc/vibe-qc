// Bounded one-center auxiliary-Gaussian Fourier panels for periodic GDF.
//
// This is the native counterpart of the libint-calibrated auxiliary transform
// used by the range-separated periodic density-fitting path.  It evaluates
//
//   F_P(p) = integral xi_P(r) exp(-i p.r) dr
//
// for caller-supplied Cartesian reciprocal vectors p.  The implementation is
// deliberately a one-panel primitive: the caller owns reciprocal-space
// enumeration and must pass an explicit byte cap before this routine allocates
// its only dense A x n_vector complex result.

#pragma once

#include <Eigen/Core>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "basis.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kAuxiliaryBasisContentDigestVersion = 1U;

// Dense row-major panel.  The layout intentionally matches a C-contiguous
// Python array with shape (n_auxiliary, n_vectors): P is the slow axis and the
// reciprocal-vector index is the unit-stride axis.
struct AuxiliaryFourierPanel {
    std::vector<std::complex<double>> data;
    std::size_t n_auxiliary = 0;
    std::size_t n_vectors = 0;
    std::uint64_t output_bytes = 0;

    std::complex<double>& operator()(std::size_t auxiliary,
                                     std::size_t vector) noexcept {
        return data[auxiliary * n_vectors + vector];
    }

    std::complex<double> operator()(std::size_t auxiliary,
                                    std::size_t vector) const noexcept {
        return data[auxiliary * n_vectors + vector];
    }
};

// Non-owning, independently-strided Cartesian-vector lanes.  A factor builder
// can point this directly at the px/py/pz lanes of its admitted five-lane
// reciprocal panel, while the row-major convenience overload below uses three
// interleaved lanes with stride 3.  Pointers may be null only when count == 0;
// every stride is measured in doubles and must be positive.
struct AuxiliaryFourierVectorView {
    const double* x = nullptr;
    const double* y = nullptr;
    const double* z = nullptr;
    std::size_t count = 0;
    std::ptrdiff_t x_stride = 1;
    std::ptrdiff_t y_stride = 1;
    std::ptrdiff_t z_stride = 1;
};

// Return the immutable numerical-content identity of a BasisSet.
//
// Canonical v1 wire, in exact order:
//
//   u64 byte length + ASCII domain
//       "vibeqc.periodic.auxiliary-basis-content"
//   u32 digest version
//   u64 nbasis
//   u64 nshells (the underlying libint shell count)
//   u64 contraction-record count
//   for every underlying libint shell in native order:
//     u64 contractions in this shell
//     for each contraction in native order:
//       u64 atom index
//       u32 angular momentum L
//       u8  pure flag (0 or 1)
//       u64 primitive count
//       primitive-count IEEE-754 binary64 exponents
//       primitive-count IEEE-754 binary64 contraction coefficients
//       three IEEE-754 binary64 origin coordinates (x, y, z)
//
// Every integer and binary64 payload is big-endian.  Both signs of binary64
// zero are encoded as +0.0.  Non-finite values and non-positive exponents are
// rejected.  The mutable/display-only BasisSet::name() is intentionally absent.
std::string auxiliary_basis_content_identity_sha256(const BasisSet& basis);

// Evaluate one bounded auxiliary Fourier panel.
//
// Shell convention:
//   * L=0 uses the libint convention with Y_00 already absorbed, so its angular
//     polynomial is 1;
//   * pure real-spherical shells L=1..cart_to_sph_data::kMaxL are evaluated
//     through the generated Cartesian solid-harmonic polynomial tables;
//   * Cartesian shells with L>0 and shells beyond the generated table are
//     rejected before numerical evaluation.
//
// The supplied matrix has one finite Cartesian reciprocal vector per row in
// inverse bohr.  output_byte_cap is mandatory and positive.  The required
// A*n_vectors*sizeof(complex<double>) extent is checked for integer and address
// space overflow and against the cap before shell inspection or transcendental
// work, then allocated exactly once.  The kernel uses no size-dependent heap
// workspace beyond that result.  Squared reciprocal norms use the fixed nested
// binary64 order
//
//   fma(px, px, fma(py, py, fma(pz, pz, 0)))
//
// shared by the reciprocal-source manifest, so its sealed p2 and the Gaussian
// radial factor cannot diverge through compiler-dependent contraction.
// Evaluation is deterministic and serial.
AuxiliaryFourierPanel auxiliary_gaussian_fourier_panel(
    const BasisSet& basis,
    AuxiliaryFourierVectorView vectors,
    std::uint64_t output_byte_cap);

// Zero-copy convenience overload for an interleaved row-major n x 3 matrix.
// It routes to the strided-view implementation and creates no staging matrix.
AuxiliaryFourierPanel auxiliary_gaussian_fourier_panel(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& vectors,
    std::uint64_t output_byte_cap);

}  // namespace vibeqc
