#include "vibeqc/basis.hpp"

#include "vibeqc/init.hpp"

#include <libint2/atom.h>
#include <libint2/shell.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string basis_lookup_key(const std::string& name) {
    std::string key;
    key.reserve(name.size());
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            key.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == '+') {
            key.push_back('+');
        }
    }
    return key;
}

std::string basis_data_name(const std::string& basis_name) {
    // Alias resolution for the *file* libint opens. This must agree with
    // vibeqc.basis_registry.canonical_basis_name, which resolves the same
    // spellings for metadata; the two used to disagree, so a name the
    // registry knew (``sto3g``) still failed in BasisSet because this
    // returned it verbatim (#743). tests/test_basis_alias_parity.py pins
    // the agreement, so an alias added to registry.toml without a line
    // here fails rather than drifting.
    //
    // basis_lookup_key strips punctuation and case, so one entry covers
    // every spelling of a name: "6-311+G(3df,2p)", "6311+g3df2p" and
    // "6-311+g3df2p" share a key.
    const std::string key = basis_lookup_key(basis_name);

    // Punctuated spellings whose file ships under a different stem.
    if (key == "6311+g3df2p") return "6-311+g3df2p";
    if (key == "sto3g") return "sto-3g";
    if (key == "sto6g") return "sto-6g";

    // Gaussian's (d) / (d,p) polarisation spellings for the Pople sets are
    // the same basis as the star form libint stores.
    if (key == "631gd") return "6-31g*";
    if (key == "631gdp") return "6-31g**";
    if (key == "6311gdp") return "6-311g**";
    if (key == "631+gdp") return "6-31+g**";
    if (key == "6311+gdp") return "6-311+g**";
    // No entry for the single-(d) forms of 6-311G, 6-31+G and 6-311+G: the
    // library ships no 6-311g*, 6-31+g* or 6-311+g* file, and mapping onto a
    // missing stem would only relabel the failure. They keep the plain
    // lowercase path and fail naming what the user asked for.

    return to_lower(basis_name);
}

std::vector<libint2::Atom> to_libint_atoms(const vibeqc::Molecule& mol) {
    std::vector<libint2::Atom> atoms;
    atoms.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        libint2::Atom la;
        la.atomic_number = a.Z;
        la.x = a.xyz[0];
        la.y = a.xyz[1];
        la.z = a.xyz[2];
        atoms.push_back(la);
    }
    return atoms;
}

// Construct a libint2::BasisSet defensively. Two failure modes need
// to be tamed before the libint2 ctor fires from inside our member-
// initializer list:
//
//  1. libint2's globals must be initialized — without an upstream
//     ``libint2::initialize()`` call, BasisSet construction is
//     undefined behavior. ``ensure_libint_initialized()`` is
//     idempotent (std::call_once) so calling it from every BasisSet
//     ctor is cheap.
//
//  2. libint2 throws (rather than returning empty) when
//     LIBINT_DATA_PATH is bogus or when a basis-file's G94 element
//     symbol isn't in the periodic table — the messages are useful
//     but bare. Wrap them with our directive context (LIBINT_DATA_PATH,
//     basis-library workflow pointer) so the user can act on the
//     error without spelunking through libint internals.
libint2::BasisSet make_libint_basis(const std::string& name_lower,
                                    const std::vector<libint2::Atom>& atoms,
                                    const std::string& original_name) {
    vibeqc::ensure_libint_initialized();
    try {
        return libint2::BasisSet(name_lower, atoms);
    } catch (const std::exception& e) {
        const char* env = std::getenv("LIBINT_DATA_PATH");
        throw std::runtime_error(
            std::string("BasisSet: failed to load basis '") + original_name +
            "'. libint2 reported: " + e.what() +
            ". LIBINT_DATA_PATH=" + (env ? env : "<unset>") +
            ". Check the spelling of the basis name (libint expects the "
            "BSE-canonical form), or place a custom .g94 file under "
            "basis_library/custom/ and rerun scripts/setup_basis_library.sh.");
    }
}

// Map each shell to its owning atom index by matching the shell's origin
// to the atom Cartesian position. libint2 constructs shell.O from the
// atoms we hand it, so exact double equality is the stable match.
// Uses tolerance-based comparison (1e-10 bohr^2) to handle
// floating-point drift in displaced geometries (optimizer steps).
std::vector<int> build_shell_to_atom(const libint2::BasisSet& shells,
                                     const vibeqc::Molecule& mol) {
    const auto& atoms = mol.atoms();
    std::vector<int> out;
    out.reserve(shells.size());
    for (const auto& shell : shells) {
        int idx = -1;
        for (std::size_t a = 0; a < atoms.size(); ++a) {
            double dx = shell.O[0] - atoms[a].xyz[0];
            double dy = shell.O[1] - atoms[a].xyz[1];
            double dz = shell.O[2] - atoms[a].xyz[2];
            if (dx*dx + dy*dy + dz*dz < 1e-10) {
                idx = static_cast<int>(a);
                break;
            }
        }
        if (idx < 0) {
            throw std::runtime_error(
                "BasisSet: shell origin does not match any atom position");
        }
        out.push_back(idx);
    }
    return out;
}

}  // namespace

namespace vibeqc {

BasisSet::BasisSet(const Molecule& mol, const std::string& basis_name,
                   bool require_all_atoms)
    : shells_(make_libint_basis(basis_data_name(basis_name),
                                to_libint_atoms(mol),
                                basis_name)),
      name_(basis_name) {
    // libint2 silently returns an empty BasisSet when no .g94 file under
    // LIBINT_DATA_PATH matches the requested name. Downstream code (integral
    // evaluators, SCF) then dereferences empty containers and segfaults.
    // Reject the empty case up front with a helpful message instead.
    if (shells_.size() == 0 || shells_.nbf() == 0) {
        throw std::runtime_error(
            "BasisSet: no shells loaded for basis '" + basis_name +
            "'. libint2's data directory does not contain a matching "
            ".g94 file for this name and the given atoms. Check the "
            "spelling, try the canonical libint/BSE form, or place the "
            "file under basis_library/custom/ and rerun "
            "scripts/setup_basis_library.sh.");
    }

    // Force pure spherical harmonics for d and higher (5d, 7f, 9g, ...),
    // overriding libint's historical Pople-convention default that forces
    // 6 Cartesian d-functions for 6-31G* / 6-31G** etc. Modern convention
    // (PySCF, Psi4, ORCA) uses spherical d — the extra Cartesian s-component
    // has no variational value and just clutters the basis.
    //
    // set_pure(true) sets pure = true on every shell's primary contraction;
    // for L <= 1 this is a no-op, for L >= 2 it switches to solid harmonics.
    shells_.set_pure(true);

    shell_to_atom_ = build_shell_to_atom(shells_, mol);

    // Every real atom must own at least one shell. libint2 hands back an
    // atom with zero shells when the .g94 file has no block for its
    // element, and nothing downstream notices: the SCF simply places that
    // atom's electrons in the other atoms' functions.
    if (require_all_atoms) {
        const auto& atoms = mol.atoms();
        std::vector<bool> covered(atoms.size(), false);
        for (int a : shell_to_atom_) covered[static_cast<std::size_t>(a)] = true;
        // build_shell_to_atom attributes a shell to the first atom at its
        // origin, so a centre coinciding with an earlier atom of the same
        // element shows as bare here although libint built its shells.
        // A duplicated centre is not a coverage gap (the DF metric guard
        // diagnoses it); borrow the coincident atom's coverage.
        for (std::size_t a = 0; a < atoms.size(); ++a) {
            if (covered[a]) continue;
            for (std::size_t b = 0; b < atoms.size(); ++b) {
                if (b == a || !covered[b] || atoms[a].Z != atoms[b].Z) continue;
                const double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
                const double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
                const double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
                if (dx * dx + dy * dy + dz * dz < 1e-10) {
                    covered[a] = true;
                    break;
                }
            }
        }
        std::vector<int> missing_z;
        for (std::size_t a = 0; a < atoms.size(); ++a) {
            if (covered[a] || atoms[a].Z <= 0) continue;
            if (std::find(missing_z.begin(), missing_z.end(), atoms[a].Z)
                == missing_z.end()) {
                missing_z.push_back(atoms[a].Z);
            }
        }
        if (!missing_z.empty()) {
            std::string listed;
            for (int z : missing_z) {
                if (!listed.empty()) listed += ", ";
                listed += "Z=" + std::to_string(z);
            }
            throw std::runtime_error(
                "BasisSet: basis '" + basis_name + "' has no functions for " +
                listed + ": the bundled file carries no block for this "
                "element, so the atom would enter the calculation with zero "
                "basis functions. Choose a basis that covers every element "
                "(see python/vibeqc/basis_library/basis/), or place a block "
                "under basis_library/custom/ and rerun "
                "scripts/setup_basis_library.sh.");
        }
    }
}

// Build libint2::BasisSet from an explicit ShellInfo list.
//
// For each ShellInfo we materialise one libint2::Shell at the supplied
// origin with one libint2::Contraction (vibe-qc emits one contraction
// per shell via shells(); the round-trip preserves that structure).
// ``embed_normalization_into_coefficients`` controls whether libint
// re-runs Shell::renorm() on construction or trusts the coefficients
// as-given.
namespace {

libint2::BasisSet make_libint_basis_from_shells(
    const std::vector<vibeqc::ShellInfo>& shells,
    bool coefficients_pre_normalized) {
    vibeqc::ensure_libint_initialized();

    // libint2::BasisSet inherits *privately* from std::vector<Shell>; we can't
    // emplace_back into it directly. Build the underlying vector first, then
    // move-construct the BasisSet — that ctor calls init() internally to fill
    // in nbf_ / max_l_ / max_nprim_ / shell2bf_ from the shell list.
    std::vector<libint2::Shell> shell_vec;
    shell_vec.reserve(shells.size());
    for (const auto& s : shells) {
        if (s.exponents.size() != s.coefficients.size()) {
            throw std::runtime_error(
                "BasisSet(<shells>): exponents and coefficients have "
                "different sizes (" + std::to_string(s.exponents.size()) +
                " vs " + std::to_string(s.coefficients.size()) +
                ") on a shell with l=" + std::to_string(s.l));
        }
        if (s.exponents.empty()) {
            throw std::runtime_error(
                "BasisSet(<shells>): shell with l=" + std::to_string(s.l) +
                " has no primitives.");
        }
        // libint2's small-vector type for the primitive arrays.
        libint2::svector<libint2::Shell::real_t> alpha(
            s.exponents.begin(), s.exponents.end());
        libint2::svector<libint2::Shell::real_t> coeff(
            s.coefficients.begin(), s.coefficients.end());
        // Contraction is nested inside Shell.
        libint2::Shell::Contraction contr{
            /*l=*/s.l,
            /*pure=*/s.pure,
            /*coeff=*/std::move(coeff)
        };
        libint2::svector<libint2::Shell::Contraction> contrs;
        contrs.push_back(std::move(contr));
        std::array<libint2::Shell::real_t, 3> origin = {
            s.origin[0], s.origin[1], s.origin[2]
        };
        // The fourth ctor arg flips the renormalisation behaviour:
        //   true  → libint multiplies coeffs by primitive norm factors
        //          (Gaussian / .g94 input convention).
        //   false → coeffs are taken as-given (libint-internal convention).
        // The vibe-qc API uses ``coefficients_pre_normalized`` (semantically
        // inverted) so the default mirrors what shells() emits — pass
        // false to libint when the caller has marked them pre-normalized.
        const bool embed = !coefficients_pre_normalized;
        shell_vec.emplace_back(std::move(alpha), std::move(contrs),
                               origin, embed);
    }
    return libint2::BasisSet(std::move(shell_vec));
}

// Build the shell-to-atom map by exact origin matching, sharing logic
// with build_shell_to_atom but taking a libint::BasisSet (not the
// vibeqc::BasisSet which doesn't exist yet at construction time).
std::vector<int> build_shell_to_atom_libint(
    const libint2::BasisSet& shells, const vibeqc::Molecule& mol) {
    const auto& atoms = mol.atoms();
    std::vector<int> out;
    out.reserve(shells.size());
    for (const auto& shell : shells) {
        int idx = -1;
        for (std::size_t a = 0; a < atoms.size(); ++a) {
            double dx = shell.O[0] - atoms[a].xyz[0];
            double dy = shell.O[1] - atoms[a].xyz[1];
            double dz = shell.O[2] - atoms[a].xyz[2];
            if (dx*dx + dy*dy + dz*dz < 1e-10) {
                idx = static_cast<int>(a);
                break;
            }
        }
        if (idx < 0) {
            throw std::runtime_error(
                "BasisSet(<shells>): shell origin does not match any "
                "atom position in the supplied Molecule. Shell origin = ["
                + std::to_string(shell.O[0]) + ", "
                + std::to_string(shell.O[1]) + ", "
                + std::to_string(shell.O[2]) + "].");
        }
        out.push_back(idx);
    }
    return out;
}

}  // namespace

BasisSet::BasisSet(const Molecule& mol,
                   const std::vector<ShellInfo>& shells,
                   const std::string& name,
                   bool coefficients_pre_normalized)
    : shells_(make_libint_basis_from_shells(
          shells, coefficients_pre_normalized)),
      name_(name) {
    if (shells_.size() == 0 || shells_.nbf() == 0) {
        throw std::runtime_error(
            "BasisSet(<shells>): no shells loaded. Pass a non-empty "
            "ShellInfo list with primitives in each shell.");
    }
    shell_to_atom_ = build_shell_to_atom_libint(shells_, mol);
}

std::size_t BasisSet::nbasis() const {
    return shells_.nbf();
}

int BasisSet::shell_atom_index(std::size_t shell_index) const {
    return shell_to_atom_.at(shell_index);
}

std::vector<ShellInfo> BasisSet::shells() const {
    std::vector<ShellInfo> out;
    out.reserve(shells_.size());
    for (std::size_t s = 0; s < shells_.size(); ++s) {
        const auto& shell = shells_[s];
        for (const auto& contr : shell.contr) {
            ShellInfo info;
            info.atom_index = shell_to_atom_[s];
            info.l = contr.l;
            info.pure = contr.pure;
            // libint2 stores alpha / coeff as svector (small-buffer
            // optimized), not std::vector — copy element-wise.
            info.exponents.assign(shell.alpha.begin(), shell.alpha.end());
            info.coefficients.assign(contr.coeff.begin(), contr.coeff.end());
            info.origin = {shell.O[0], shell.O[1], shell.O[2]};
            out.push_back(std::move(info));
        }
    }
    return out;
}

// Build a "softened" BasisSet by dropping tight primitives (exponent >
// ``exponent_cutoff``) from every contracted shell of ``basis``. This is
// the native fast path for ``vibeqc.periodic_gapw_augment.softened_basis``
// and is contractually IDENTICAL to that function's pure-Python fallback
// (see python/vibeqc/periodic_gapw_augment.py) — same primitives kept,
// same shells dropped, contraction coefficients carried through unchanged:
//
//   * a primitive is kept iff its exponent <= exponent_cutoff;
//   * a shell whose primitives are ALL tight is dropped entirely (it
//     contributes no soft density) — it is NOT collapsed to a single
//     "loosest" primitive;
//   * surviving coefficients are taken AS-IS: the derived basis is built
//     with coefficients_pre_normalized=true, so libint does not renormalise
//     them and the soft density matches the fallback bit-for-bit;
//   * if every shell is dropped (cutoff below the loosest exponent in the
//     whole basis), the original basis is returned unchanged.
//
// Keeping these semantics in lock-step with the Python fallback is what
// lets softened_basis() prefer this path transparently — flipping between
// the C++ and Python implementations must not move any SCF number. The
// equivalence is pinned by tests/test_basis_prune_tight_primitives.py.
// (NB: the fallback also emits a GAPWExperimentalWarning in the all-pruned
// case; the native path returns the same basis but cannot raise that Python
// warning — an informational-only difference in a pathological cutoff.)
BasisSet prune_tight_primitives(const BasisSet& basis,
                                const Molecule& mol,
                                double exponent_cutoff) {
    auto all_shells = basis.shells();
    std::vector<ShellInfo> pruned_shells;
    pruned_shells.reserve(all_shells.size());
    for (const auto& sh : all_shells) {
        ShellInfo kept;
        kept.atom_index = sh.atom_index;
        kept.l = sh.l;
        kept.pure = sh.pure;
        kept.origin = sh.origin;
        kept.exponents.reserve(sh.exponents.size());
        kept.coefficients.reserve(sh.coefficients.size());
        for (std::size_t i = 0; i < sh.exponents.size(); ++i) {
            if (sh.exponents[i] <= exponent_cutoff) {
                kept.exponents.push_back(sh.exponents[i]);
                kept.coefficients.push_back(sh.coefficients[i]);
            }
        }
        if (kept.exponents.empty()) {
            continue;  // shell has no soft primitives — drop it
        }
        pruned_shells.push_back(std::move(kept));
    }
    if (pruned_shells.empty()) {
        // Every shell was fully pruned. Mirror the fallback's "return the
        // original basis unchanged" by rebuilding it verbatim from its own
        // shells (coefficients already libint-normalised → pre_normalized).
        return BasisSet(mol, all_shells, basis.name(),
                        /*coefficients_pre_normalized=*/true);
    }
    return BasisSet(mol, pruned_shells,
                    basis.name() + "-soft",
                    /*coefficients_pre_normalized=*/true);
}

}  // namespace vibeqc
