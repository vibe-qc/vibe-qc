#pragma once

#include "vibeqc/symmetry_shared.hpp"
#include <complex>
#include <cstdint>
#include <vector>

namespace vibeqc {
// Provider-independent contraction of a supplied block tensor. This object
// does not qualify a producer, selection rule, retained space or state.
SymmetryPlan plan_symmetry_ri_blocks(std::uint64_t n, std::uint64_t rank,
    std::uint64_t groups, std::uint64_t blocks, std::uint64_t values,
    const SymmetryBudget&);

struct SymmetryRIResult {
    std::vector<std::complex<double>> j, k;
    std::uint64_t scalar_products = 0, block_contractions = 0;
};

class SymmetryRIBlocks {
public:
    // Offsets partition the adapted AO and auxiliary coordinates; empty
    // groups are allowed. Keys [p,q,A] describe nonoverlapping dense blocks,
    // packed in key order with row-major [A,p,q] payloads.
    SymmetryRIBlocks(std::uint64_t n, std::uint64_t rank,
        std::uint64_t groups, std::uint64_t blocks, std::uint64_t values,
        const std::int64_t* orbital_offsets, const std::int64_t* auxiliary_offsets,
        const std::int64_t* keys, const std::complex<double>* factors,
        const SymmetryBudget&);
    SymmetryRIResult contract(const std::complex<double>* density,
        bool make_j, bool make_k, const SymmetryBudget&) const;
    std::uint64_t dimension() const { return n_; }
    const SymmetryPlan& memory() const { return memory_; }
private:
    struct Block { std::uint64_t p, q, a, offset; };
    std::uint64_t n_, rank_, groups_;
    SymmetryPlan memory_;
    std::vector<std::uint64_t> orbital_, auxiliary_;
    std::vector<Block> blocks_;
    std::vector<std::vector<std::uint64_t>> by_auxiliary_;
    std::vector<std::complex<double>> factors_;
};
} // namespace vibeqc
