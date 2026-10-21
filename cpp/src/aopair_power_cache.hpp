#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace vibeqc::aopair_ft_detail {

// G-only Cartesian factors of the Gaussian-product Fourier transform.
// The original left-associated complex products are cached, never the weights,
// primitive prefactors or reductions. A call owns this bounded table; separate
// folded-z and explicit-z rows preserve both existing arithmetic paths.
// McMurchie-Davidson, J. Comput. Phys. 26, 218 (1978), Sec. 2;
// Sun et al., J. Chem. Phys. 147, 164119 (2017), Appendix B.
class ReciprocalPowerCache {
public:
    static constexpr std::size_t byte_limit = 64 * 1024 * 1024;
    static constexpr std::size_t max_powers = 13;
    static constexpr std::size_t max_rows =
        max_powers * max_powers * (max_powers + 1);
    static constexpr std::size_t descriptor_bytes = max_rows * 64;

    ReciprocalPowerCache(const std::vector<std::complex<double>>& x,
                         const std::vector<std::complex<double>>& y,
                         const std::vector<std::complex<double>>& z,
                         std::size_t n_G, std::size_t n_powers,
                         std::size_t budget = byte_limit)
        : x_(x), y_(y), z_(z), n_G_(n_G), n_powers_(n_powers) {
        const auto limit = std::min(budget, byte_limit);
        if (!n_G_ || !n_powers_ || n_powers_ > max_powers
            || limit <= descriptor_bytes
            || n_G_ > x_.size() / n_powers_
            || n_G_ > y_.size() / n_powers_
            || n_G_ > z_.size() / n_powers_) return;
        capacity_ = (limit - descriptor_bytes) / sizeof(std::complex<double>) / n_G_;
        if (!capacity_) return;
        // Low total orders are shared by the most shell pairs. Higher orders
        // not admitted under the byte bound use the original arithmetic.
        try {
            rows_.resize(n_powers_ * n_powers_ * (n_powers_ + 1));
            for (std::size_t order = 0; order < n_powers_; ++order)
                for (std::size_t t = 0; t <= order; ++t)
                    for (std::size_t u = 0; u <= order - t; ++u) {
                        const auto v = order - t - u;
                        admit(t, u, v, false);
                        if (v == 0) admit(t, u, 0, true);
                    }
        } catch (const std::bad_alloc&) {
            rows_.clear();
            count_ = capacity_ = 0;
        }
    }

    const std::complex<double>* find(std::size_t t, std::size_t u,
                                     std::size_t v, bool folded_z) const {
        if (rows_.empty() || t >= n_powers_ || u >= n_powers_ || v >= n_powers_
            || n_powers_ > max_powers || (folded_z && v != 0)) return nullptr;
        const auto& row = rows_[index(t, u, v, folded_z)];
        if (!row) return nullptr;
        std::call_once(row->ready, [&] {
            const auto* gx = x_.data() + t * n_G_;
            const auto* gy = y_.data() + u * n_G_;
            const auto* gz = z_.data() + v * n_G_;
            for (std::size_t k = 0; k < n_G_; ++k) {
                const std::complex<double> xy = gx[k] * gy[k];
                row->values[k] = folded_z ? xy : xy * gz[k];
            }
        });
        return row->values.get();
    }
    std::size_t rows() const noexcept { return count_; }
    std::size_t payload_bytes() const noexcept {
        return count_ * n_G_ * sizeof(std::complex<double>);
    }
private:
    struct Row {
        std::unique_ptr<std::complex<double>[]> values;
        mutable std::once_flag ready;
    };
    std::size_t index(std::size_t t, std::size_t u, std::size_t v,
                      bool folded_z) const {
        const auto plane = n_powers_ * n_powers_;
        return folded_z ? plane * n_powers_ + t * n_powers_ + u
                        : (t * n_powers_ + u) * n_powers_ + v;
    }
    void admit(std::size_t t, std::size_t u, std::size_t v, bool folded_z) {
        if (count_ >= capacity_) return;
        auto row = std::make_unique<Row>();
        row->values.reset(new std::complex<double>[n_G_]);
        rows_[index(t, u, v, folded_z)] = std::move(row);
        ++count_;
    }
    const std::vector<std::complex<double>>& x_;
    const std::vector<std::complex<double>>& y_;
    const std::vector<std::complex<double>>& z_;
    std::size_t n_G_, n_powers_, capacity_ = 0, count_ = 0;
    std::vector<std::unique_ptr<Row>> rows_;
};
} // namespace vibeqc::aopair_ft_detail
