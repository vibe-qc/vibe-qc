// Bounded observations of periodic shell-pair contraction; no numerical policy.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace vibeqc {
// identities: output cell, lambda cell, sigma cell, density cell,
//             s1,s2,s3,s4, mu,nu,rho,sigma AO, channel (0=J,1=K), sequence.
// The three cell positions index the returned J/K cells; density cell indexes
// the supplied density. A null integral uses UINT64_MAX for all four AOs.
// values: returned integral, density, separately rounded product, sum before,
//         sum after. Null integral records contain zeros in these columns.
struct PeriodicJKTraceRecord {
    std::array<std::uint64_t, 14> identities{};
    std::array<double, 5> values{};
};

namespace detail {
class PeriodicJKTraceBuffer {
public:
    static constexpr std::size_t byte_limit = 128 * 1024 * 1024;
    explicit PeriodicJKTraceBuffer(std::size_t capacity) {
        if (!capacity || capacity > byte_limit / sizeof(PeriodicJKTraceRecord))
            throw std::invalid_argument("periodic J/K trace requires a positive record cap within 128 MiB");
        records_.resize(capacity);
    }
    // Every worker claims its own preallocated element. No allocation or
    // exception is allowed inside an OpenMP contraction. Overflow invalidates
    // the whole trace at finish(), never returns a silently truncated prefix.
    void record(const PeriodicJKTraceRecord& record) noexcept {
        const auto slot = next_.fetch_add(1, std::memory_order_relaxed);
        if (slot < records_.size()) records_[slot] = record;
    }
    std::vector<PeriodicJKTraceRecord> finish() {
        const auto count = next_.load(std::memory_order_relaxed);
        if (count > records_.size())
            throw std::length_error("periodic J/K trace record cap exceeded; no partial trace returned");
        records_.resize(count);
        std::sort(records_.begin(), records_.end(), [](const auto& a, const auto& b) {
            // Each output shell/channel owns a sequential accumulator. Restore
            // its actual order independently of worker completion order.
            const auto& x = a.identities; const auto& y = b.identities;
            return std::tie(x[0],x[4],x[5],x[12],x[13])
                 < std::tie(y[0],y[4],y[5],y[12],y[13]);
        });
        return std::move(records_);
    }
private:
    std::vector<PeriodicJKTraceRecord> records_;
    std::atomic<std::size_t> next_{0};
};
} // namespace detail
} // namespace vibeqc
