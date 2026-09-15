#pragma once

// Private shared leaves for finite Gaussian pair integrals. The Fourier
// implementation remains the owner of AO lookup, image boxes and MD tables.
// This traversal is the existing v1 binary64 source predicate, NOT an
// interval-certified enumeration or an infinite-image truncation bound.

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <Eigen/Core>
#include "vibeqc/basis.hpp"

namespace vibeqc::aopair_ft_detail {

struct Ao {
    const libint2::Shell* shell = nullptr;
    const libint2::Shell::Contraction* contraction = nullptr;
    std::size_t component = 0;
};
struct ImageBox {
    std::array<std::int64_t, 3> lower{};
    std::array<std::int64_t, 3> upper{};
    std::uint64_t candidates = 1;
};
using LongMatrix = Eigen::Matrix<long double, 3, 3>;
Ao ao(const BasisSet&, std::uint64_t);
ImageBox image_box(const Ao&, const Ao&, const LongMatrix&, double cutoff);
// Caller supplies (la+1)*(lb+1)*(la+lb+1) doubles; unnormalized
// Hermite polynomial coefficients, with E(0,0,0)=1. No allocation.
void md_coefficients(int la, int lb, double gamma, double pa, double pb, double*);

// One translated AO pair and reciprocal vector, including its ket Bloch
// phase and physical solid-harmonic normalization. Entries are value,
// d/dA_x,y,z and d/dB_x,y,z. The same AO lookup, image policy and MD
// recurrence as the value panels are used; this leaf allocates no heap.
std::array<std::complex<double>, 7> value_and_center_derivatives(
    const Ao& bra, const Ao& ket, const Eigen::Vector3d& p,
    const Eigen::Vector3d& k_cart, const Eigen::Vector3d& translation,
    const Eigen::Vector3d& separation, double separation_squared);
std::size_t center_derivative_numeric_workspace_bytes() noexcept;

inline double norm2(double x, double y, double z) {
    return std::fma(x, x, std::fma(y, y, std::fma(z, z, 0.0)));
}
inline void require_finite(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("AO-pair Fourier panel requires finite values");
}
template<class Function>
std::uint64_t walk_images(const Ao& bra, const Ao& ket,
    const Eigen::Matrix3d& lattice, const ImageBox& box,
    double cutoff_squared, Function&& function) {
    std::uint64_t retained = 0;
    for (std::int64_t i = box.lower[0]; i <= box.upper[0]; ++i) {
        for (std::int64_t j = box.lower[1]; j <= box.upper[1]; ++j) {
            for (std::int64_t k = box.lower[2]; k <= box.upper[2]; ++k) {
                Eigen::Vector3d translation;
                Eigen::Vector3d separation;
                for (int d = 0; d < 3; ++d) {
                    translation[d] = std::fma(
                        static_cast<double>(i), lattice(d, 0),
                        std::fma(static_cast<double>(j), lattice(d, 1),
                                 static_cast<double>(k) * lattice(d, 2)));
                    separation[d] = bra.shell->O[d] - ket.shell->O[d]
                                    - translation[d];
                    require_finite(translation[d]);
                    require_finite(separation[d]);
                }
                const double squared = norm2(separation[0], separation[1], separation[2]);
                require_finite(squared);
                if (squared <= cutoff_squared) {
                    ++retained;
                    function(std::array<std::int64_t, 3>{i, j, k},
                             translation, separation, squared);
                }
            }
        }
    }
    return retained;
}
} // namespace vibeqc::aopair_ft_detail
