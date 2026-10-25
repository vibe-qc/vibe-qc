#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace vibeqc::aopair_ft_detail {

// One Fourier call owns this table. Only exp(-|G|^2/(4 gamma)) is shared:
// origins, contraction coefficients, phases, weights and Hermite reductions
// stay with their original primitive/cell task. Exact gamma keys, no rounding.
// McMurchie-Davidson, J. Comput. Phys. 26, 218 (1978), Sec. 2;
// Sun et al., J. Chem. Phys. 147, 164119 (2017), Appendix B.
class RadialGaussianCache {
public:
    static constexpr std::size_t byte_limit = 64 * 1024 * 1024;
    static constexpr std::size_t row_limit = 128;
    // Conservative allowance for the fixed row inventory and map descriptors.
    static constexpr std::size_t descriptor_bytes = row_limit * 128;

    explicit RadialGaussianCache(const std::vector<double>& squared_vectors,
                                 std::size_t budget = byte_limit)
        : squared_vectors_(squared_vectors),
          capacity_(squared_vectors.empty() ? 0 : std::min(
              row_limit, (std::min(budget, byte_limit) > descriptor_bytes
                  ? std::min(budget, byte_limit) - descriptor_bytes : 0) / sizeof(double)
                         / squared_vectors.size())) {}

    // Admission is serial, before workers start. Payload is allocated here so
    // lazy initialization in an OpenMP worker cannot throw on allocation.
    // At most 64 MiB of doubles and 128 map/once-flag descriptors per call.
    void admit(double gamma) {
        if (rows_.size() >= capacity_ || !std::isfinite(gamma) || gamma <= 0.0
            || rows_.find(gamma) != rows_.end()) return;
        try {
            auto& row = rows_.try_emplace(gamma).first->second;
            row.values.reset(new double[squared_vectors_.size()]);
        } catch (const std::bad_alloc&) {
            // Reuse is optional: allocation pressure preserves the scalar path.
            rows_.clear();
            capacity_ = 0;
        }
    }

    // The row inventory never changes after admission. Each row is filled at
    // first actual use, so fully screened shell pairs evaluate no exponentials.
    // call_once publishes its immutable values to all participating workers.
    const double* find(double gamma) const {
        if (!std::isfinite(gamma) || gamma <= 0.0) return nullptr;
        const auto it = rows_.find(gamma);
        if (it == rows_.end()) return nullptr;
        const auto& row = it->second;
        std::call_once(row.ready, [&] {
            const double inv_gamma = 1.0 / gamma;
            const double quarter_inv_gamma = 0.25 * inv_gamma;
            for (std::size_t k = 0; k < squared_vectors_.size(); ++k)
                row.values[k] = std::exp(-squared_vectors_[k] * quarter_inv_gamma);
        });
        return row.values.get();
    }

    std::size_t rows() const noexcept { return rows_.size(); }
    std::size_t payload_bytes() const noexcept {
        return rows_.size() * squared_vectors_.size() * sizeof(double);
    }

private:
    struct Row {
        std::unique_ptr<double[]> values;
        mutable std::once_flag ready;
    };
    const std::vector<double>& squared_vectors_;
    std::size_t capacity_;
    std::map<double, Row> rows_;
};
} // namespace vibeqc::aopair_ft_detail
