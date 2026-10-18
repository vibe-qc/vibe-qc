// Minimal valence basis set for semiempirical methods.
//
// Builds a libint2 BasisSet from STO-NG expansions (N=6) with
// element-specific Slater exponents.  The basis is minimal valence.
// Works with both DFTB-specific SemiempiricalParameters and the
// generic CoreParameterSet interface.

#pragma once

#include <vector>

#include "vibeqc/basis.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/core/parameters.hpp"

namespace vibeqc {
namespace semiempirical {

struct ValenceShell {
    int atom_index;
    int l;
    bool pure;
};

class SemiempiricalBasis {
public:
    // Build from DFTB-specific parameters.
    static BasisSet build(const Molecule& mol,
                          const SemiempiricalParameters& params,
                          int n_primitives = 6);

    // Build from generic CoreParameterSet (for xTB, PM7, etc.).
    // n_primitives == 0 is the GFN2-xTB "auto" sentinel: resolves to the
    // element-dependent primitive count (H,He -> 3; else -> 4) per atom,
    // matching xtb's setGFN2NumberOfPrimitives.
    static BasisSet build(const Molecule& mol,
                          const CoreParameterSet& params,
                          int n_primitives = 6);

    static std::vector<ValenceShell> valence_shells(
        const Molecule& mol, const SemiempiricalParameters& params);

    static std::vector<ShellInfo> build_shell_info(
        const Molecule& mol, const SemiempiricalParameters& params,
        int n_primitives = 6);

    // See build() above for the n_primitives == 0 "auto" sentinel.
    static std::vector<ShellInfo> build_shell_info(
        const Molecule& mol, const CoreParameterSet& params,
        int n_primitives = 6);

private:
    // (zeta, n, l) -> contracted Gaussian expansion of the STO with
    // n_primitives in {3, 4, 6}; every row is keyed by principal quantum
    // number n and angular momentum l (Stewart 1970, xtb slater.f90).
    static std::pair<std::vector<double>, std::vector<double>>
    sto_ng_expansion(double zeta, int n, int l, int n_primitives = 6);
};

}  // namespace semiempirical
}  // namespace vibeqc
