// Collision-resistant identity for native semiempirical parameter snapshots.
//
// The canonical encoder is deliberately binary, versioned, and independent
// of locale or host byte order. Parameter-set implementations append every
// stored, execution-relevant field in a documented schema order and then pin
// the resulting SHA-256 in their published-content allowlist.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace semiempirical {

// Mixed into native result records so provenance is bound to the immutable
// parameter copy that actually executed, rather than reconstructed from a
// mutable caller-owned object after the calculation.
struct ParameterIdentifiedResult {
    std::string parameter_identity;
    std::string parameter_sha256;
};

namespace detail {

class CanonicalParameterHasher {
public:
    explicit CanonicalParameterHasher(const std::string& schema) {
        add_string("vibeqc.semiempirical.parameter-content");
        add_string(schema);
    }

    void add_bool(bool value) { add_u64(value ? 1U : 0U); }

    void add_int(std::int64_t value) {
        add_u64(static_cast<std::uint64_t>(value));
    }

    void add_size(std::size_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "semiempirical parameter collection is too large to hash");
        }
        add_u64(static_cast<std::uint64_t>(value));
    }

    void add_double(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "semiempirical parameter identity rejects non-finite values");
        }
        if (value == 0.0) value = 0.0;  // Canonicalize negative zero.
        static_assert(std::numeric_limits<double>::is_iec559,
                      "parameter identity requires IEEE-754 arithmetic");
        static_assert(sizeof(double) == sizeof(std::uint64_t),
                      "parameter identity requires IEEE-754 binary64");
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        add_u64(bits);
    }

    void add_string(const std::string& value) {
        add_size(value.size());
        hasher_.update(
            reinterpret_cast<const std::uint8_t*>(value.data()),
            value.size());
    }

    std::string finish() { return hasher_.finish_hex(); }

private:
    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            encoded[i] = static_cast<std::uint8_t>(
                value >> (56U - 8U * i));
        }
        hasher_.update(encoded.data(), encoded.size());
    }

    ::vibeqc::detail::Sha256 hasher_;
};

inline std::string custom_parameter_identity(const std::string& sha256) {
    return "custom:" + sha256;
}

}  // namespace detail
}  // namespace semiempirical
}  // namespace vibeqc
