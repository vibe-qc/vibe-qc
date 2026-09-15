// Spin-adapted pair densities and pair natural orbitals in one real,
// orthonormal virtual domain. This is a numerical primitive, not a DLPNO
// driver, a periodic Hamiltonian, or a threshold-preset implementation.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vibeqc/hermitian_jacobi.hpp"

namespace vibeqc {

enum class RestrictedPairKind : std::uint32_t {
    Diagonal = 0,
    OffDiagonal = 1,
};

struct RestrictedPairMP2MemoryPlan {
    std::uint64_t domain_dimension = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
};

struct RestrictedPairMP2Result {
    RestrictedPairMP2Result() = default;
    RestrictedPairMP2Result(const RestrictedPairMP2Result&) = delete;
    RestrictedPairMP2Result& operator=(const RestrictedPairMP2Result&) = delete;
    RestrictedPairMP2Result(RestrictedPairMP2Result&&) noexcept = default;
    RestrictedPairMP2Result& operator=(RestrictedPairMP2Result&&) noexcept = default;
    std::size_t domain_dimension = 0;
    std::vector<double> amplitudes;
    // Ordered spatial-pair contribution sum_ab (2 G_ab-G_ba) T_ab.
    // NO diagonal/off-diagonal, translation, symmetry or cell multiplicity.
    double ordered_pair_energy = 0.0;
    double minimum_denominator = 0.0;
    double maximum_denominator = 0.0;
    double maximum_absolute_amplitude = 0.0;
    double maximum_residual = 0.0;
    double residual_frobenius_norm = 0.0;
    std::uint64_t energy_term_underflow_count = 0;
    std::uint64_t energy_input_scaling_underflow_count = 0;
    RestrictedPairMP2MemoryPlan memory;
};

RestrictedPairMP2MemoryPlan plan_restricted_pair_semicanonical_mp2(
    std::uint64_t domain_dimension);

// Initial semi-canonical amplitudes, not the coupled local MP2 solution:
// Nejad et al., JCP 163, 214107 (2025), doi:10.1063/5.0290816, Eq.37.
// G_ab=(i a|j b), Delta_ab=eps_a+eps_b-f_ii-f_jj, T_ab=-G_ab/Delta_ab.
// Real orthonormal semicanonical virtual space only. Complex factors must
// first pass an explicit upstream real-space integral gate; no cast here.
// The positive denominator_floor is an explicit rejection boundary, never
// a level shift or clipping rule. All Delta must be strictly greater.
// Inputs are borrowed contiguous row-major G[n,n] and eps[n]; cap covers
// exactly one owned n*n amplitude matrix, checked before reads/allocation.
// Non-finite inputs/results, denominator input-scaling loss and amplitude
// underflow fail closed. Energy-only input-scaling/term underflow is counted.
// No pair screening, preset, DIIS or PNO step.
RestrictedPairMP2Result restricted_pair_semicanonical_mp2(
    const double* exchange_integrals, std::size_t integral_elements,
    const double* virtual_energies, std::size_t energy_elements,
    std::size_t domain_dimension, double occupied_fock_ii,
    double occupied_fock_jj, double denominator_floor,
    std::uint64_t numerical_byte_cap);

struct RestrictedPairPNOMemoryPlan {
    std::uint64_t domain_dimension = 0;
    // Caller-owned input. This is reported, but not included in the owned
    // payload cap below: the primitive never copies the amplitude matrix.
    std::uint64_t input_bytes = 0;
    // Density, all occupations, and at most n*n retained PNO coefficients.
    std::uint64_t output_bytes_upper_bound = 0;
    // The two n*n complex arrays consumed by the allocation-free eigensolver.
    std::uint64_t eigensolver_workspace_bytes = 0;
    // Exact maximum simultaneously owned numerical payload: 40*n*n + 8*n.
    // Allocator bookkeeping, constant-size objects and caller data are not
    // numerical payload. There is no Eigen/BLAS or other hidden heap scratch.
    std::uint64_t peak_owned_numerical_bytes = 0;
};

struct RestrictedPairPNOResult {
    RestrictedPairPNOResult() = default;
    RestrictedPairPNOResult(const RestrictedPairPNOResult&) = delete;
    RestrictedPairPNOResult& operator=(const RestrictedPairPNOResult&) = delete;
    RestrictedPairPNOResult(RestrictedPairPNOResult&&) noexcept = default;
    RestrictedPairPNOResult& operator=(RestrictedPairPNOResult&&) noexcept = default;

    RestrictedPairKind pair_kind = RestrictedPairKind::OffDiagonal;
    std::size_t domain_dimension = 0;
    std::size_t retained_dimension = 0;
    double occupation_cutoff = 0.0;
    // Row-major n*n density and n*n_retained coefficient matrix. PNOs are
    // coefficient columns, ordered by descending occupation. The largest
    // absolute coefficient in each column has a nonnegative sign.
    std::vector<double> density;
    std::vector<double> coefficients;
    // All n eigenvalues in descending order, including roundoff-scale
    // negative values. No occupation is silently clipped or rescaled.
    std::vector<double> occupations;
    double density_trace = 0.0;
    double discarded_occupation_sum = 0.0;
    double minimum_occupation = 0.0;
    std::uint64_t negative_occupation_count = 0;
    // The diagonal-pair symmetry gate is ||(T-T^T)/2||_F / ||T||_F
    // <= 64*epsilon. The accepted antisymmetric part is discarded only for
    // diagonal pairs, and its physical Frobenius norm is reported here.
    double input_relative_antisymmetric_norm = 0.0;
    double discarded_antisymmetric_norm = 0.0;
    // Floating-point range diagnostics, not rigorous error certificates.
    std::uint64_t amplitude_scaling_underflow_count = 0;
    std::uint64_t density_underflow_entry_count = 0;
    // Independent residual against the original, returned density. These
    // diagnostics are evaluated over the complete eigensystem before PNO
    // truncation, using scalar loops and constant-size scratch.
    double eigensystem_relative_residual = 0.0;
    double eigenvector_orthogonality_error = 0.0;
    HermitianJacobiResult eigensolver;
    RestrictedPairPNOMemoryPlan memory;
    std::uint64_t output_numerical_bytes = 0;
};

// Count-only checked plan. Requires n > 0 and rejects integer/address-space
// overflow. No data-dependent or size-dependent allocation is performed.
RestrictedPairPNOMemoryPlan plan_restricted_pair_pnos(
    std::uint64_t domain_dimension);

// Construct the real restricted pair density of Riplinger and Neese,
// J. Chem. Phys. 138, 034106 (2013), p. 034106-7, Eqs. (23)-(25),
// doi:10.1063/1.4773581:
//
//   Ttilde = (4*T - 2*T^T)/(1+delta_ij)
//   D = Ttilde*T^T + Ttilde^T*T.
//
// With S=(T+T^T)/2 and A=(T-T^T)/2 this is exactly
//   D = (4*S*S^T + 12*A*A^T)/(1+delta_ij),
// a positive-semidefinite Gram sum. It is evaluated in that form with
// power-of-two input scaling and compensated scalar sums. A diagonal pair
// requires symmetric T up to the scale-aware gate documented on the result.
// PairKind changes the density normalization only; no pair-energy or
// translation multiplicity is applied by this primitive.
//
// amplitudes points to a real, contiguous row-major n*n matrix. The supplied
// accessible element count may be larger, but must cover the active matrix.
// The positive numerical_byte_cap is checked against the complete owned
// numerical peak before any size-dependent allocation or amplitude read.
// All amplitudes, the nonnegative cutoff, and returned numerical quantities
// must be finite. Materially negative eigenvalues, significant complex
// eigenvectors for this real matrix, nonconvergence and failed independent
// residual/orthogonality checks are errors. Tiny negative eigenvalues are
// retained verbatim in occupations and counted, never silently clipped.
//
// A positive cutoff retains exactly occupations > occupation_cutoff, as in
// the paper; a rank-zero result is valid. Exactly zero cutoff is a deliberate
// complete-domain diagnostic convention: it retains every column, including
// zero-occupation and roundoff-negative directions. No PNO is force-retained
// for a positive cutoff. No semicanonicalization or energy correction occurs.
RestrictedPairPNOResult restricted_pair_pnos(
    const double* amplitudes,
    std::size_t amplitude_elements,
    std::size_t domain_dimension,
    RestrictedPairKind pair_kind,
    double occupation_cutoff,
    std::uint64_t numerical_byte_cap,
    const HermitianJacobiOptions& eigensolver_options);

struct RestrictedPairSemicanonicalMemoryPlan {
    std::uint64_t domain_dimension = 0;
    std::uint64_t selected_dimension = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t projection_phase_bytes = 0;
    std::uint64_t factorization_phase_bytes = 0;
    std::uint64_t rotation_phase_bytes = 0;
    std::uint64_t validation_phase_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
};

struct RestrictedPairSemicanonicalResult {
    RestrictedPairSemicanonicalResult() = default;
    RestrictedPairSemicanonicalResult(const RestrictedPairSemicanonicalResult&) = delete;
    RestrictedPairSemicanonicalResult& operator=(
        const RestrictedPairSemicanonicalResult&) = delete;
    RestrictedPairSemicanonicalResult(RestrictedPairSemicanonicalResult&&) noexcept = default;
    RestrictedPairSemicanonicalResult& operator=(
        RestrictedPairSemicanonicalResult&&) noexcept = default;

    std::size_t domain_dimension = 0;
    std::size_t selected_dimension = 0;
    // Row-major n*r C'=C*U; columns ordered by ascending semicanonical energy.
    // The largest absolute coefficient in each column is nonnegative.
    std::vector<double> coefficients;
    std::vector<double> energies;
    double input_orthonormality_error = 0.0;
    double output_orthonormality_error = 0.0;
    double reduced_eigensystem_relative_residual = 0.0;
    double projected_fock_relative_residual = 0.0;
    double subspace_projector_frobenius_error = 0.0;
    // ||F C'-C' eps||_F / ||F||_F is reported only, never convergence-gated:
    // the selected PNO space generally is NOT an invariant space of full F.
    double full_space_relative_residual = 0.0;
    // Residual denominators use one if the corresponding Fock norm is zero.
    int fock_scale_exponent = 0;
    std::uint64_t fock_scaling_underflow_entries = 0;
    bool eigensolver_performed = false;
    HermitianJacobiResult eigensolver;
    RestrictedPairSemicanonicalMemoryPlan memory;
};

// Count-only allocation-transparent plan for n>0 and 0<=r<=n. Inputs remain
// caller-owned. For r>0, exact owned numerical phase payloads are:
// projection 8*r*r+8*n; factorization 40*r*r+8*r;
// rotation 8*n*r+16*r*r+8*r; validation 8*n*r+8*r+8*n.
// The cap covers their maximum, excluding allocator/control overhead. r=0
// requires no owned numerical payload, and an explicit zero cap is valid.
RestrictedPairSemicanonicalMemoryPlan plan_restricted_pair_semicanonicalization(
    std::uint64_t domain_dimension, std::uint64_t selected_dimension);

// Recanonicalization following Riplinger and Neese (2013), p.034106-7,
// text immediately after Eqs.24-25, doi:10.1063/1.4773581:
// diagonalize only the selected-space Fock matrix C^T F C and return C'=C U.
// No selection, density construction, orbital forcing, or outside-domain
// orbital mixing occurs. C and F are readonly contiguous row-major arrays of
// n*r and n*n real doubles. F must be finite and exactly symmetric, and C
// must be finite with ||C^T C-I||_F <= input_orthonormality_tolerance (explicit,
// finite, strictly between zero and one). Pointer extents and a sufficient
// numerical_byte_cap are checked before allocation or numerical input reads.
//
// Power-of-two Fock scaling avoids intermediate overflow; scale losses are
// reported. Numerical audits use the original projected Fock matrix and an
// independent reconstruction C'^T F C', never an incorrect full-space
// invariance requirement. Empty selected spaces are valid, but n=0 is not.
RestrictedPairSemicanonicalResult restricted_pair_semicanonicalize(
    const double* coefficients,
    std::size_t coefficient_elements,
    const double* virtual_fock,
    std::size_t fock_elements,
    std::size_t domain_dimension,
    std::size_t selected_dimension,
    double input_orthonormality_tolerance,
    std::uint64_t numerical_byte_cap,
    const HermitianJacobiOptions& eigensolver_options);

}  // namespace vibeqc
