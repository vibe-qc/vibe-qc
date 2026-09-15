// One-time libint2 initialization.
//
// libint2::initialize() must be called exactly once per process before any
// integral engine is created; libint2::finalize() is intentionally NOT
// called on module unload (letting the OS reclaim memory is cheaper and
// sidesteps static-destruction-order headaches when pybind11 tears down).

#pragma once

namespace vibeqc {

void ensure_libint_initialized();

}  // namespace vibeqc
