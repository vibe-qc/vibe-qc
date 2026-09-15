#include "vibeqc/init.hpp"

#include <libint2.hpp>
#include <mutex>
#include <stdexcept>

namespace vibeqc {

void ensure_libint_initialized() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        libint2::initialize();

        // Pin the solid-harmonic ordering invariant documented in
        // vibeqc/basis.hpp. libint's *runtime* setting -- not the
        // LIBINT_SHGSHELL_ORDERING build macro -- is what
        // SolidHarmonicsCoefficients::init() consults when it builds the
        // Cartesian->pure transform the integral engine applies
        // (third_party/libint/install/include/libint2/solidharmonics.h).
        // So this is the check that actually protects ao_eval.cpp's
        // hardcoded m = -l..+l loop, the molden m-permutation table, and
        // the QVF writer's `m = ao_local - l`.
        //
        // libint2::initialize() leaves the accessor at its default
        // (SHGShellOrdering_Standard) and set_solid_harmonics_ordering()
        // must be called before the first Engine is constructed, so
        // checking once here -- immediately after initialize(), before any
        // vibe-qc Engine exists -- catches both a changed upstream default
        // and any stray setter call that beat us to it. Throwing beats
        // silently emitting permuted orbitals.
        if (libint2::solid_harmonics_ordering()
                != libint2::SHGShellOrdering_Standard) {
            throw std::runtime_error(
                "vibe-qc requires libint's STANDARD solid-harmonic ordering "
                "(m = -l..+l), but libint2::solid_harmonics_ordering() "
                "reports GAUSSIAN/MOLDEN ordering (m = 0, +1, -1, +2, -2, "
                "...). Every pure shell with l >= 1 would be silently "
                "permuted in AO evaluation, the molden writer, and the QVF "
                "writer. Do not call libint2::set_solid_harmonics_ordering() "
                "with a non-standard ordering.");
        }
    });
}

}  // namespace vibeqc
