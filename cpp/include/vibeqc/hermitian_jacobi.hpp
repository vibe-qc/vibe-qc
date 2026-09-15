// Allocation-transparent reference eigensolver for complex Hermitian matrices.
//
// Cyclic two-sided Jacobi rotations, using the Hermitian specialization of
// T. Hahn, "Routines for the diagonalization of complex matrices" (2007
// revision), Secs. 3 and 4.1, Eqs. 10-17, arXiv:physics/0607103v2,
// doi:10.48550/arXiv.physics/0607103.
// Eigenvectors are columns here, the adjoint of Hahn's row convention.

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace vibeqc {

enum class HermitianJacobiStatus : std::uint32_t {
    Success = 0,
    InvalidInput = 1,
    NonFiniteInput = 2,
    NotHermitian = 3,
    NoConvergence = 4,
    NumericalFailure = 5,
};

struct HermitianJacobiOptions {
    // Both controls must be supplied explicitly. They describe numerical
    // diagonalization, never the scientific rank cutoff of a later consumer.
    std::uint64_t max_sweeps = 0;
    // Finite and strictly between zero and one. A completed sweep converges
    // when ||offdiag(M_scaled)||_F <= tolerance * ||M_input_scaled||_F.
    double relative_offdiagonal_tolerance = 0.0;
};

struct HermitianJacobiResult {
    HermitianJacobiStatus status = HermitianJacobiStatus::InvalidInput;
    std::uint64_t sweeps = 0;
    std::uint64_t rotations = 0;
    // M_scaled = scalbn(M_input, -input_scale_exponent). The scale is an
    // exactly representable positive power of two, including subnormals.
    int input_scale_exponent = 0;
    double input_scale = 1.0;
    double scaled_input_frobenius_norm = 0.0;
    double scaled_offdiagonal_frobenius_norm = 0.0;
    double orthogonality_frobenius_error = 0.0;
    // Nonzero real/imaginary input components rounded to zero by scaling.
    // This is diagnostic accounting, not a claimed backward-error bound.
    std::uint64_t scaling_underflow_components = 0;
};

// Solve M U = U diag(lambda), with no allocation, exceptions, Eigen or BLAS.
// The caller owns three disjoint arrays: row-major M and U, each with at
// least dimension^2 elements, and lambda with at least dimension elements.
// The arrays must actually contain the declared accessible storage; this
// function checks nulls, alignment, active extent/address overflow, overlap,
// dimension > 0 and numerical options before dereferencing or mutating them.
// IEEE-754 binary64 without fast-math is required at compile time; a runtime
// floating-point rounding mode other than FE_TONEAREST, or lack of gradual
// underflow, returns InvalidInput.
// Non-finite input or any non-real diagonal / non-Hermitian element is rejected
// before mutation. Hermiticity uses exact numerical equality; either sign of
// zero is accepted. Input extents need not be equal to the active extents.
//
// On Success, lambda is finite and sorted ascending (stable for equal
// values), U contains the matching eigenvectors in its columns, and M holds
// the scaled, transformed matrix with the same final column ordering. The
// retained offdiagonal entries are NOT cleared to pretend exact convergence.
// Input M is consumed. On NoConvergence or NumericalFailure, buffers contain
// partial work and must not be used as a successful eigensystem.
//
// Each sweep visits (p,q) in row-cyclic order, p=0..n-2, q=p+1..n-1. Zero
// pivots are skipped. A nonzero pivot whose rotation underflows is left in M
// so it remains visible to the convergence test. The solver reports failure
// if its sweep budget is exhausted or a finite eigenvalue is not representable.
//
// The offdiagonal norm is an internal convergence diagnostic, NOT a certified
// residual against original M: rounding and input scaling alter the accumulated
// similarity relation. Independent validation must retain or reconstruct small
// original matrices and check ||M_original U - U diag(lambda)||_F/||M_original||_F
// as well as ||U^H U-I||_F. No original-matrix copy is retained by this routine.
HermitianJacobiResult hermitian_jacobi_in_place(
    std::complex<double>* matrix,
    std::size_t matrix_elements,
    std::complex<double>* eigenvectors,
    std::size_t eigenvector_elements,
    double* eigenvalues,
    std::size_t eigenvalue_elements,
    std::size_t dimension,
    const HermitianJacobiOptions& options) noexcept;

}  // namespace vibeqc
