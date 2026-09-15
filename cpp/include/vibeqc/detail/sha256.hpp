// Minimal streaming SHA-256 used by native, versioned content identities.
//
// This is deliberately only the raw hash primitive. Each owning subsystem
// defines its own canonical wire format, domain separator, and schema version.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace vibeqc {
namespace detail {

class Sha256 {
public:
    Sha256() { reset(); }

    void update(const std::uint8_t* data, std::size_t size) {
        total_size_ += size;
        while (size > 0) {
            const std::size_t available = block_.size() - block_size_;
            const std::size_t take = size < available ? size : available;
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                process_block(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish_hex() {
        const std::uint64_t bit_size = total_size_ * 8U;
        const std::uint8_t one = 0x80U;
        update(&one, 1);
        const std::uint8_t zero = 0;
        while (block_size_ != 56) update(&zero, 1);
        std::array<std::uint8_t, 8> encoded_size{};
        for (std::size_t i = 0; i < encoded_size.size(); ++i) {
            encoded_size[i] = static_cast<std::uint8_t>(
                bit_size >> (56U - 8U * i));
        }
        update(encoded_size.data(), encoded_size.size());

        static constexpr char hex[] = "0123456789abcdef";
        std::string result(64, '0');
        for (std::size_t i = 0; i < state_.size(); ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                const auto byte = static_cast<std::uint8_t>(
                    state_[i] >> (24U - 8U * j));
                result[8U * i + 2U * j] = hex[byte >> 4U];
                result[8U * i + 2U * j + 1U] = hex[byte & 0x0fU];
            }
        }
        return result;
    }

private:
    static std::uint32_t rotate_right(std::uint32_t value,
                                      std::uint32_t count) noexcept {
        return (value >> count) | (value << (32U - count));
    }

    void reset() noexcept {
        state_ = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
        };
        total_size_ = 0;
        block_size_ = 0;
    }

    void process_block(const std::uint8_t* data) noexcept {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(data[4U * i]) << 24U)
                | (static_cast<std::uint32_t>(data[4U * i + 1U]) << 16U)
                | (static_cast<std::uint32_t>(data[4U * i + 2U]) << 8U)
                | static_cast<std::uint32_t>(data[4U * i + 3U]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const std::uint32_t s0 = rotate_right(words[i - 15U], 7U)
                ^ rotate_right(words[i - 15U], 18U)
                ^ (words[i - 15U] >> 3U);
            const std::uint32_t s1 = rotate_right(words[i - 2U], 17U)
                ^ rotate_right(words[i - 2U], 19U)
                ^ (words[i - 2U] >> 10U);
            words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t sum1 = rotate_right(e, 6U)
                ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + sum1 + choose
                + constants[i] + words[i];
            const std::uint32_t sum0 = rotate_right(a, 2U)
                ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::uint64_t total_size_ = 0;
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
};

}  // namespace detail
}  // namespace vibeqc
