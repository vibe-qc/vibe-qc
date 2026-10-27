// Copyright (c) vibe-qc contributors.
// SPDX-License-Identifier: MPL-2.0
// Standalone SCC mixer algebra checks; also registered in tests/native.
// Build with the Eigen headers used by the native core:
// c++ -std=c++17 -O1 -Icpp/include -I<eigen-include> \
//   tests/test_charge_diis.cpp cpp/src/semiempirical/core/charge_mixer.cpp \
//   -o <external-build>/charge-diis
#include "vibeqc/semiempirical/core/charge_mixer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using vibeqc::semiempirical::ChargeDIISMixer;
using Vector = Eigen::VectorXd;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void weighted_residual_minimum(double scale) {
    ChargeDIISMixer mixer(6, .5, 3);
    Vector x = Vector::Zero(2);
    Vector error(2);
    error << scale, 0.;
    mixer.mix(x, (x + error).eval(), 1);
    x << 1., 2.;
    error << 0., .1 * scale;
    mixer.mix(x, (x + error).eval(), 2);
    // Minimize c1^2 + .01*c2^2 with c1+c2=1. Individually
    // normalized errors incorrectly give c1=c2=.5 instead.
    Vector expected(2);
    // Apply those coefficients to g(x_i) = x_i + error_i.
    expected << (1. + .01 * scale) / 1.01,
                (2. + .1 * scale) / 1.01;
    require((x - expected).cwiseAbs().maxCoeff() < 1e-10,
            "DIIS must retain relative residual magnitudes");
}

void refresh_parallel_history() {
    ChargeDIISMixer mixer(6, .5, 3);
    Vector x = Vector::Zero(2), error(2);
    error << 1., 0.;
    mixer.mix(x, (x + error).eval(), 1);
    x << .9, 0.;
    error << .1, 0.;
    mixer.mix(x, (x + error).eval(), 2);
    x << .95, .2;
    error << 0., .1;
    mixer.mix(x, (x + error).eval(), 3);
    Vector expected(2);
    expected << .975, .15;
    require((x - expected).cwiseAbs().maxCoeff() < 1e-12,
            "parallel history must keep the current paired input and error");
    mixer.reset();
    x.setZero();
    error << 1., 0.;
    mixer.mix(x, error, 1);
    require(std::abs(x(0) - .5) < 1e-15 && x(1) == 0.,
            "reset must discard both input and residual history");
}

void contracting_map() {
    // Two independent affine responses with the known fixed point (1, 1e-6).
    // The first residual contracts rapidly; the second eventually rotates
    // the error direction. Stale, individually normalized history used to
    // jump from a residual below 1e-6 back to .5 and stay there indefinitely.
    ChargeDIISMixer mixer(6, .5, 3);
    Vector x = Vector::Zero(2), next(2);
    for (int iteration = 1; iteration <= 80; ++iteration) {
        next << 1., 1e-7 + .9 * x(1);
        mixer.mix(x, next, iteration);
    }
    next << 1., 1e-7 + .9 * x(1);
    require((next - x).cwiseAbs().maxCoeff() < 1e-12,
            "contracting affine map must reach a small residual");
    require(std::abs(x(0) - 1.) < 1e-12 && std::abs(x(1) - 1e-6) < 1e-11,
            "contracting affine map must reach its independent fixed point");
}

void oscillating_map() {
    // The known fixed point is (1, 2e-8). Simple half mixing is unstable
    // in the second mode. Input-only extrapolation stalls with a residual
    // around 3e-5 because its next input stays inside the same affine span.
    ChargeDIISMixer mixer(6, .5, 3);
    Vector x = Vector::Zero(2), next(2);
    for (int iteration = 1; iteration <= 80; ++iteration) {
        next << 1., 1e-7 - 4. * x(1);
        mixer.mix(x, next, iteration);
    }
    next << 1., 1e-7 - 4. * x(1);
    require((next - x).cwiseAbs().maxCoeff() < 1e-12,
            "DIIS must advance a nonzero residual beyond the input span");
    require(std::abs(x(0) - 1.) < 1e-12 && std::abs(x(1) - 2e-8) < 1e-12,
            "oscillating affine map must reach its independent fixed point");
}

int main() {
    try {
        for (double scale : {1e-6, 1., 1e6}) weighted_residual_minimum(scale);
        refresh_parallel_history();
        contracting_map();
        oscillating_map();
        std::cout << "Charge DIIS: residual weights, scale invariance, history, reset and affine maps passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
