// Configuration witness only. No integral evaluation and no libint execution.
// libint2.h exposes the generated capability macros and matching public C API.
#include <libint2.h>

#define VIBEQC_STRINGIFY_IMPL(value) #value
#define VIBEQC_STRINGIFY(value) VIBEQC_STRINGIFY_IMPL(value)
#ifdef LIBINT2_SUPPORT_ONEBODY
static_assert(LIBINT2_SUPPORT_ONEBODY == 1,
              "LIBINT2_SUPPORT_ONEBODY=" VIBEQC_STRINGIFY(LIBINT2_SUPPORT_ONEBODY)
              "; vibe-qc requires 1");
#else
#error "LIBINT2_SUPPORT_ONEBODY=unknown; vibe-qc requires 1"
#endif
#ifdef LIBINT2_DERIV_ONEBODY_ORDER
static_assert(LIBINT2_DERIV_ONEBODY_ORDER >= 1,
              "LIBINT2_DERIV_ONEBODY_ORDER=" VIBEQC_STRINGIFY(LIBINT2_DERIV_ONEBODY_ORDER)
              "; vibe-qc requires >=1");
#else
#error "LIBINT2_DERIV_ONEBODY_ORDER=unknown; vibe-qc requires >=1"
#endif

#if defined(LIBINT2_SUPPORT_ONEBODY) && LIBINT2_SUPPORT_ONEBODY == 1 && \
    defined(LIBINT2_DERIV_ONEBODY_ORDER) && LIBINT2_DERIV_ONEBODY_ORDER >= 1
int main() {
    // Volatile pointer stores retain link references even under optimization.
    // Do not hard-code table dimensions or call an uninitialized build table.
    auto* volatile overlap = &libint2_build_overlap1;
    auto* volatile kinetic = &libint2_build_kinetic1;
    auto* volatile elecpot = &libint2_build_elecpot1;
    auto* volatile overlap_memory = &libint2_need_memory_overlap1;
    auto* volatile kinetic_memory = &libint2_need_memory_kinetic1;
    auto* volatile elecpot_memory = &libint2_need_memory_elecpot1;
    return overlap == nullptr || kinetic == nullptr || elecpot == nullptr ||
           overlap_memory == nullptr || kinetic_memory == nullptr ||
           elecpot_memory == nullptr;
}
#endif
