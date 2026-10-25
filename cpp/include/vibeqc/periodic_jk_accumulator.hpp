#pragma once

#include <cmath>

namespace vibeqc::detail {

// Neumaier compensated summation of signed, separately rounded products.
// Higham, SIAM J. Sci. Comput. 14, 783-799 (1993), Sec. 3, pp. 790-791,
// doi:10.1137/0914050. Keep the magnitude comparison and parenthesization:
// reassociation under fast-math would discard the rounding correction.
// Two doubles per output AO pair; no retained list of contraction terms.
class PeriodicJKAccumulator {
public:
    void add(double value) noexcept {
        const double updated = sum_ + value;
        correction_ += std::abs(sum_) >= std::abs(value)
            ? (sum_ - updated) + value : (value - updated) + sum_;
        sum_ = updated;
    }
    double value() const noexcept { return sum_ + correction_; }
private:
    double sum_ = 0.0;
    double correction_ = 0.0;
};

} // namespace vibeqc::detail
