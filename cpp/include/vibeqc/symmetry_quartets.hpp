#pragma once

#include "vibeqc/symmetry_shared.hpp"
#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

namespace vibeqc {
// Provider-independent algebra only. Source, group, orbit coverage and
// screening admission remain the caller's responsibility.
SymmetryPlan plan_symmetry_quartet_accumulator(std::uint64_t n,
    std::uint64_t shells, std::uint64_t operations, std::uint64_t matrix_elements,
    std::uint64_t maximum_shell, const SymmetryBudget&, std::uint64_t density_count = 1,
    std::uint64_t j_count = std::numeric_limits<std::uint64_t>::max(),
    std::uint64_t k_count = std::numeric_limits<std::uint64_t>::max());

struct SymmetryQuartetCounts {
    std::uint64_t representatives = 0, scalar_products = 0;
    std::uint64_t block_contractions = 0, transport_scalar_products = 0;
};
struct SymmetryQuartetTargets {
    std::uint64_t dimension = 0, j_count = 1, k_count = 1;
    std::vector<std::complex<double>> j, k;
    SymmetryQuartetCounts counts;
};

class SymmetryQuartetAccumulator {
public:
    SymmetryQuartetAccumulator(std::uint64_t n, std::uint64_t shells,
        std::uint64_t operations, std::uint64_t matrix_elements,
        std::uint64_t maximum_shell, const std::int64_t* offsets,
        const double* inverse_matrices, const std::complex<double>* density,
        const SymmetryBudget&, std::uint64_t density_count = 1,
        std::uint64_t j_count = std::numeric_limits<std::uint64_t>::max(),
        std::uint64_t k_count = std::numeric_limits<std::uint64_t>::max(),
        const std::uint8_t* targets = nullptr);
    // members is row-major [operation, permutation, shell0, ..., shell3].
    // The ordered real tile is borrowed only until this call returns.
    void accumulate(const std::array<std::int64_t,4>& representative,
        const std::array<std::uint64_t,4>& shape, const double* tile,
        std::uint64_t members, const std::int64_t* images);
    // Expand one admitted real shell orbit in first-occurrence order, then
    // reuse accumulate. mapped_shells[g,axis] is the shell action on this
    // representative only. mode: 0 identity, 1 ERI permutations, 2 spatial.
    // Index algebra, not a source or group-action certificate. One orbit's
    // map, unique-image set and rows fit the existing metadata reservation.
    void accumulate_orbit(const std::array<std::int64_t,4>& representative,
        const std::array<std::uint64_t,4>& shape, const double* tile,
        std::uint64_t operations, const std::int64_t* mapped_shells,
        std::uint64_t identity, std::uint8_t mode);
    SymmetryQuartetTargets finish();
    // Binding-level descriptor failures occur before accumulate is entered.
    void invalidate() noexcept { valid_ = false; }
    const SymmetryPlan& memory() const { return memory_; }
    const std::vector<std::uint64_t>& shell_offsets() const { return offsets_; }
    std::uint64_t operations() const { return operations_; }
private:
    SymmetryPlan memory_;
    std::uint64_t n_, shells_, operations_, matrix_elements_, maximum_shell_;
    bool valid_ = true, finished_ = false;
    std::vector<std::uint64_t> offsets_, sizes_, matrix_offsets_;
    std::vector<std::uint64_t> j_channels_, k_channels_;
    std::vector<double> matrices_, first_, second_;
    std::vector<std::uint8_t> identity_blocks_;
    std::vector<std::complex<double>> density_, j_, k_;
    SymmetryQuartetCounts counts_;
};
} // namespace vibeqc
