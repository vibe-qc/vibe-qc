// Thin header-only helpers for OpenMP parallelism.
//
// Purpose: a single place to query thread count and to build the
// canonical "one libint2::Engine per thread" pool used by every
// integral-heavy kernel. Keeps each site from duplicating
//
//     const int nt = omp_get_max_threads();
//     std::vector<libint2::Engine> engines(nt, prototype);
//
// and the corresponding `engine = engines[omp_get_thread_num()]`
// access inside the parallel region.
//
// When OpenMP is disabled (unlikely in our build, but keeps the
// header correct when compiled without `-fopenmp`) the helpers
// return sensible single-threaded defaults.

#pragma once

#include <libint2/engine.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_OPENMP)
  #include <omp.h>
#endif

namespace vibeqc {

// Maximum number of OpenMP threads in the current parallel region
// or (if we are not inside one) the maximum the user has requested
// via OMP_NUM_THREADS / omp_set_num_threads.
inline int omp_max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Zero-based thread index inside the innermost enclosing parallel
// region, or 0 if we are not in one.
inline int omp_thread_index() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Size of the active team, or one outside a parallel region.
inline int omp_team_threads() {
#if defined(_OPENMP)
    return omp_get_num_threads();
#else
    return 1;
#endif
}

// True inside an active OpenMP parallel region.  Kernels that can run both
// as top-level work and inside a coarser parallel decomposition use this to
// suppress nested teams while retaining their ordinary standalone scaling.
inline bool omp_in_parallel_region() {
#if defined(_OPENMP)
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

// Bound an outer decomposition by both the requested OpenMP team and the
// number of independent work items.  ``cap`` is a memory-safety ceiling for
// kernels whose per-worker scratch is substantial.
inline int omp_workers_for(std::size_t work_items, int cap) {
    if (work_items == 0 || cap <= 0 || omp_in_parallel_region()) return 1;
    const auto bounded = std::min(
        work_items,
        static_cast<std::size_t>(std::min(omp_max_threads(), cap)));
    return std::max(1, static_cast<int>(bounded));
}

// Build a pool of libint2::Engines, one per thread, all cloned from
// a single prototype so they share configuration (operator, max_l,
// max_nprim, parameter set). libint2::Engine is copy-constructible
// but not thread-safe, so each thread must own its own engine.
//
// Usage:
//     auto engines = make_engine_pool(prototype);
//     #pragma omp parallel for
//     for (...) {
//         auto& engine = engines[omp_thread_index()];
//         engine.compute(...);
//     }
inline std::vector<libint2::Engine>
make_engine_pool(const libint2::Engine& prototype) {
    return std::vector<libint2::Engine>(
        static_cast<std::size_t>(omp_max_threads()), prototype);
}

// Set the maximum number of threads. ``n <= 0`` restores the default:
// OMP_NUM_THREADS from the environment if set, otherwise the number
// of hardware logical cores (``omp_get_num_procs``). Returns the
// thread count that will be used on the next parallel region.
//
// When OpenMP is not available this is a no-op and always returns 1.
inline int set_num_threads(int n) {
#if defined(_OPENMP)
    if (n <= 0) {
        const char* env = std::getenv("OMP_NUM_THREADS");
        int target = omp_get_num_procs();
        if (env != nullptr) {
            try {
                const int parsed = std::stoi(env);
                if (parsed > 0) target = parsed;
            } catch (...) {
                // ignore malformed env; fall through to num_procs
            }
        }
        omp_set_num_threads(target);
    } else {
        omp_set_num_threads(n);
    }
    return omp_get_max_threads();
#else
    (void)n;
    return 1;
#endif
}

}  // namespace vibeqc
