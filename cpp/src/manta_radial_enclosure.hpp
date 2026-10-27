#pragma once
#include "vibeqc/manta_md.hpp"
#include "manta_ri_interval.hpp"
namespace vibeqc::manta {
// Shared radial-kernel proof; callers admit order/work and validate the kernel.
// Ordinary Boys and positive erfc implementations remain owned by the metric
// producer. This wrapper also retains its erf and endpoint conventions.
std::vector<spectral_interval::Interval> radial_kernel_enclosure(int,
    spectral_interval::Interval,spectral_interval::Interval,md::RadialKernel,double);
} // namespace vibeqc::manta
