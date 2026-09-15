#pragma once

// Private arithmetic test seam. This constructs no physical source, panel,
// reference or certificate. It reuses the production consumer's Boys table.
#include <cstdint>
#include <vector>

namespace vibeqc::periodic_gaussian_nuclear_detail {
std::vector<double> boys_diagnostic(int order, const double* values,
    std::uint64_t count, std::uint64_t maximum_owned_numeric_bytes,
    std::uint64_t maximum_work_units);
}
