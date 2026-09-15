// Atomic-orbital basis set, backed by libint2's BasisSet.

#pragma once

#include <libint2/basis.h>
#include <libint2/config.h>
#include <libint2/shgshell_ordering.h>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "molecule.hpp"

// --- Solid-harmonic ordering invariant -------------------------------------
//
// vibe-qc assumes libint emits the 2l+1 functions of a pure shell in
// STANDARD order, m = -l, -l+1, ..., +l. That assumption is hardcoded in
// at least three places, none of which consult libint to check it:
//
//   * cpp/src/ao_eval.cpp -- cart_to_pure_transform() builds its rows with
//     a literal `for (int m = -l; m <= l; ++m)` loop.
//   * python/vibeqc/output/formats/molden.py -- the molden<-libint
//     m-permutation table.
//   * python/vibeqc/output/formats/qvf.py -- emits `m = ao_local - l`, and
//     the QVF spec Appendix A.2 pins m = -l..+l for `pure: true`.
//
// Under GAUSSIAN ordering (m = 0, +1, -1, +2, -2, ...) every pure shell
// with l >= 1 would silently permute: no error, just wrong orbitals.
//
// The compile-time macro below is only half the guard, and is NOT the half
// the integral engine reads. Upstream deprecated it (libint2/config2.h marks
// it "retired Jan 2023"); it now seeds only the legacy FOR_SOLIDHARM /
// INT_SOLIDHARMINDEX helpers. libint2::solidharmonics::SolidHarmonicsCoefficients
// -- what the engine actually transforms with -- reads the *runtime* setting
// libint2::solid_harmonics_ordering() instead. The load-bearing check is
// therefore the runtime one in ensure_libint_initialized()
// (cpp/src/init.cpp); this static_assert just refuses a vendored libint
// configured the wrong way at build time, and documents the assumption at
// the header every shell-touching translation unit already includes.
#ifdef LIBINT_SHGSHELL_ORDERING
static_assert(
    LIBINT_SHGSHELL_ORDERING == LIBINT_SHGSHELL_ORDERING_STANDARD,
    "vibe-qc requires libint built with STANDARD solid-harmonic ordering "
    "(m = -l..+l). This libint is configured for GAUSSIAN ordering, which "
    "would silently permute every pure shell with l >= 1 in ao_eval.cpp, "
    "the molden writer, and the QVF writer. Rebuild libint with "
    "LIBINT_SHGSHELL_ORDERING=LIBINT_SHGSHELL_ORDERING_STANDARD.");
#endif

// --- Cartesian in-shell ordering invariant ---------------------------------
//
// vibe-qc equally assumes libint's STANDARD Cartesian component ordering
// (FOR_CART: lx = l..0 descending, then ly = l-lx..0 descending), hardcoded
// independently in at least six places, none of which consult libint:
//
//   * cpp/src/ao_eval.cpp -- enumerate_cartesians()
//   * cpp/src/cosx_kernel.cpp -- the Obara-Saika component loops
//   * python/vibeqc/output/formats/qvf.py -- _basis_shell_payload, and the
//     QVF spec Appendix A.2 pins the same order for `pure: false`
//   * python/vibeqc/_aopair_ft.py -- cartesian_components_for_l
//   * cpp/include/vibeqc/cart_to_sph_data.hpp -- generated column order
//   * vibe-view's renderer -- _cartesian_factors
//
// Unlike the solid-harmonic ordering above, this macro has NO runtime
// accessor and no deprecation shim -- libint's generated engine code is
// specialized to it at build time -- so this compile-time check is fully
// load-bearing, not a secondary guard. Under LIBINT_CGSHELL_ORDERING_ORCA
// or _GAMESS every surface above would silently permute the components of
// each Cartesian shell: no error, just wrong densities. Not every listed
// surface includes this header directly (cosx_kernel.cpp does not); the
// guard still covers them, because a misconfigured libint fails the build
// in the TUs that do -- the check is per-build, not per-TU.
#ifdef LIBINT_CGSHELL_ORDERING
static_assert(
    LIBINT_CGSHELL_ORDERING == LIBINT_CGSHELL_ORDERING_STANDARD,
    "vibe-qc requires libint built with STANDARD Cartesian (CGShell) "
    "ordering (lx descending, then ly descending). This libint is "
    "configured with a different in-shell Cartesian ordering, which would "
    "silently permute the components of every Cartesian shell in "
    "ao_eval.cpp, cosx_kernel.cpp, the QVF writer, and the "
    "Cartesian->spherical transform tables. Rebuild libint with its "
    "default (standard) Cartesian Gaussian ordering.");
#endif

namespace vibeqc {

// Python-facing description of one contracted Gaussian shell.
// Every field is plain data so the struct is copy-by-value trivially
// exposable through pybind11. One instance per (libint shell ×
// contraction) — for the common segmented case this is 1:1 with shells.
struct ShellInfo {
    int atom_index;                    // 0-based into Molecule.atoms()
    int l;                             // angular momentum: 0=s, 1=p, 2=d, ...
    bool pure;                         // true = spherical harmonics (2L+1);
                                       // false = Cartesian ((L+1)(L+2)/2)
    std::vector<double> exponents;     // primitive Gaussian exponents
    std::vector<double> coefficients;  // contraction coefficients
                                       // (libint-normalized, same length
                                       // as exponents)
    std::array<double, 3> origin;      // Cartesian origin in bohr
};

class BasisSet {
public:
    // basis_name is a libint2-recognized basis identifier ("sto-3g",
    // "6-31g*", "cc-pvdz", ...). Case is normalized internally.
    //
    // ``require_all_atoms`` (default true) refuses a basis file that has no
    // block for one of the molecule's elements. libint2 does not raise for
    // that case: it returns the atom with zero shells, and an SCF then runs
    // with the missing atom's electrons in the other atoms' functions
    // (measured: Ag+ over benzene in def2-TZVP, which stops at Kr, placed
    // 88 electrons in ligand functions under the HCORE guess). Pass false
    // only for coverage diagnostics that want to inspect the gaps
    // (vibeqc.density_fitting.aux_basis_coverage_gaps). A Z=0 ghost
    // centre is exempt.
    BasisSet(const Molecule& mol, const std::string& basis_name,
             bool require_all_atoms = true);

    // Build a BasisSet from explicit per-shell metadata. Each ShellInfo
    // contributes one libint shell at the supplied origin with the given
    // primitives + contraction coefficients.
    //
    // ``coefficients_pre_normalized`` controls how the coefficients are
    // interpreted:
    //
    //   true  (default) — coefficients are taken AS GIVEN. Use this when
    //         the ShellInfo list comes from another BasisSet's
    //         ``shells()`` (the natural form for round-tripping and
    //         in-place coefficient modification — e.g. modrho
    //         renormalisation in vibeqc.aux_basis).
    //   false — coefficients are raw .g94 / Gaussian-style (assume
    //         primitives are unnormalized). libint multiplies in
    //         primitive normalisation factors at construction
    //         (``Shell::renorm()``). Use this when feeding shells parsed
    //         from a basis-set file format that uses the Gaussian
    //         convention.
    //
    // The shell-to-atom map is built by matching shell origins to atom
    // positions (exact equality — same logic as the name-loaded
    // constructor). Every shell origin must coincide with an atom in
    // ``mol``; otherwise the constructor throws.
    BasisSet(const Molecule& mol, const std::vector<ShellInfo>& shells,
             const std::string& name = "<custom>",
             bool coefficients_pre_normalized = true);

    std::size_t nbasis() const;
    std::size_t nshells() const noexcept { return shells_.size(); }
    const std::string& name() const noexcept { return name_; }

    // Owning atom of one underlying libint shell.  This is a zero-allocation
    // accessor for native streaming kernels that must not materialise the
    // copy-by-value ShellInfo inventory.  Throws std::out_of_range when
    // shell_index >= nshells().
    int shell_atom_index(std::size_t shell_index) const;

    // Access the underlying libint basis (needed to construct Engines).
    const libint2::BasisSet& libint() const noexcept { return shells_; }

    // Flattened per-contraction description of every shell in the basis.
    // Returned by value; cheap enough for typical basis sizes.
    std::vector<ShellInfo> shells() const;

private:
    libint2::BasisSet shells_;
    std::string name_;
    std::vector<int> shell_to_atom_;
};

// Build a "softened" BasisSet by dropping tight primitives (exponent >
// ``exponent_cutoff``) from each shell of ``basis``. Native fast path for
// vibeqc.periodic_gapw_augment.softened_basis, kept bit-for-bit identical
// to that function's pure-Python fallback: a primitive survives iff its
// exponent <= cutoff; a shell with no surviving primitives is dropped
// entirely (not collapsed to a loosest primitive); surviving contraction
// coefficients are carried through unchanged (pre-normalised, no renorm);
// an all-pruned basis is returned unchanged. See basis.cpp for the full
// contract + the parity test that pins it.
BasisSet prune_tight_primitives(const BasisSet& basis,
                                const Molecule& mol,
                                double exponent_cutoff);

}  // namespace vibeqc
