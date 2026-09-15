#pragma once

// Shared mathematical symmetry primitives. No geometry, operator or state
// certification, solver admission or approximation policy is implied.
#include <array>
#include <complex>
#include <cstdint>
#include <vector>

namespace vibeqc {

struct SymmetryBudget {
    std::uint64_t maximum_bytes = 0;
    std::uint64_t maximum_work = 0;
};
struct SymmetryPlan {
    // Logical simultaneous borrowed + owned payload and conservative control
    // reservation; not an allocator/RSS guarantee. No numerical replicas.
    std::uint64_t bytes = 0, work = 0;
};

class SymmetryGroup {
public:
    std::uint64_t order() const { return order_; }
    std::uint64_t identity() const { return identity_; }
    std::uint64_t product(std::uint64_t g, std::uint64_t h) const;
    std::uint64_t inverse(std::uint64_t g) const;
    bool antiunitary(std::uint64_t g) const;
    std::array<std::int64_t,3> cocycle(std::uint64_t g, std::uint64_t h) const;
    const SymmetryPlan& memory() const { return memory_; }
private:
    SymmetryGroup() = default;
    std::uint64_t order_ = 0, identity_ = 0;
    SymmetryPlan memory_;
    std::vector<std::int64_t> products_, inverses_, cocycles_;
    std::vector<std::uint8_t> antiunitary_;
    friend SymmetryGroup make_symmetry_group(std::uint64_t, std::uint64_t,
        const std::int64_t*, const std::uint8_t*, const std::int64_t*,
        const std::int64_t*, const SymmetryBudget&);
};
SymmetryPlan plan_symmetry_group(std::uint64_t order, bool lattice,
                                const SymmetryBudget&);
// products[g,h] acts h first. Optional rotations[g,3,3] and
// cocycles[g,h,3] occur together; g h = T_cocycle(g,h) product(g,h).
// Scalar time reversal is the Z2 grading. Spin double groups are not inferred.
SymmetryGroup make_symmetry_group(std::uint64_t order, std::uint64_t identity,
    const std::int64_t* products, const std::uint8_t* antiunitary,
    const std::int64_t* rotations, const std::int64_t* cocycles,
    const SymmetryBudget&);

SymmetryPlan plan_symmetry_blocks(std::uint64_t dimension, std::uint64_t blocks,
    std::uint64_t matrix_elements, std::uint64_t columns, const SymmetryBudget&);
// Compact block-permutation action. offsets partitions [0,dimension),
// destinations is a bijection of equal-sized blocks. Matrices are packed
// row-major, in source block order. Arbitrary complex invertible or singular
// blocks are accepted: this is transport, not a representation certificate.
// antiunitary applies U conjugate(C); its REAL-inner-product adjoint is
// conjugate(U^dagger C), not U^dagger conjugate(C).
std::vector<std::complex<double>> apply_symmetry_blocks(
    std::uint64_t dimension, std::uint64_t blocks, std::uint64_t matrix_elements,
    std::uint64_t columns, const std::int64_t* offsets,
    const std::int64_t* destinations, const std::complex<double>* matrices,
    const std::complex<double>* coefficients, bool antiunitary, bool adjoint,
    const SymmetryBudget&);

} // namespace vibeqc
