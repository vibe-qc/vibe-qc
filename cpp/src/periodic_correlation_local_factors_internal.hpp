#pragma once

// Private shared leaves. Callers must perform object/cap/view admission before
// invoking numeric routines. No caller-labelled factor or public array route.
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_correlation_local_factors.hpp"

namespace vibeqc {
namespace periodic_correlation_local_detail {

void validate_reference(const PeriodicCorrelationAdmittedReference& reference);
void validate_wannier(const PeriodicCorrelationAdmittedReference& reference,
                      const PeriodicCorrelationWannier& wannier);
void validate_pao(const PeriodicCorrelationAdmittedReference& reference,
                  const PeriodicCorrelationPAODomain& domain,
                  const PeriodicCorrelationPAOSpace& space);
void validate_reader(const PeriodicCorrelationAdmittedReference& reference,
                     const PeriodicCorrelationFactorStreamSchedule& schedule,
                     const PeriodicCorrelationPrivateFactorReader& reader);
std::string validate_gauges(const PeriodicCorrelationWannier& wannier,
                            const std::complex<double>* gauges, std::size_t count);

// Outputs are contiguous columns. Both virtual scratch columns have nao
// entries and are reused between output columns. No hidden allocations.
void fill_occupied_column(const PeriodicRestrictedMeanFieldState& state,
                          const std::complex<double>* gauges,
                          std::uint64_t occupied_index, std::uint64_t cell,
                          std::size_t k, std::complex<double>* output);
void fill_virtual_columns(const PeriodicRestrictedMeanFieldState& state,
                         const PeriodicCorrelationPAODomain& domain,
                         const PeriodicCorrelationPAOSpace& space,
                         std::uint64_t begin, std::uint64_t count,
                         std::uint64_t translation_cell, std::size_t k,
                         std::complex<double>* output, std::complex<double>* scratch);

// Exact scalar wire shared by the localized consumers, never std::hash or
// native object bytes. Callers preflight the SHA bit-length domain.
class Digest {
public:
    explicit Digest(const char* domain) { string(domain); u32(1); }
    void u32(std::uint32_t x) {
        std::array<std::uint8_t, 4> b{};
        for (unsigned i = 0; i < 4; ++i) b[i] = x >> (24U - 8U * i);
        hash_.update(b.data(), b.size());
    }
    void u64(std::uint64_t x) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i < 8; ++i) b[i] = x >> (56U - 8U * i);
        hash_.update(b.data(), b.size());
    }
    void real(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("local factor gauge/digest has a non-finite lane");
        if (x == 0.0) x = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x, sizeof(bits));
        u64(bits);
    }
    void complex(std::complex<double> x) { real(x.real()); real(x.imag()); }
    void string(const std::string& x) {
        u64(x.size());
        hash_.update(reinterpret_cast<const std::uint8_t*>(x.data()), x.size());
    }
    void selection(const PeriodicCorrelationLocalOrbitalSelection& s) {
        u64(s.occupied_index); u64(s.occupied_cell); u64(s.virtual_begin);
        u64(s.virtual_count); u64(s.virtual_translation_cell);
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
};

}  // namespace periodic_correlation_local_detail
}  // namespace vibeqc
