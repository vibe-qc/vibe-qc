// Shared published conventions for molecular correlated methods.
//
// The frozen-core counts below transcribe ORCA 6.1, Table 2.69
// ("Number of frozen core electrons").  ORCA additionally identifies core
// orbitals by atomic-orbital character when the canonical energy ordering is
// unusual.  vibe-qc currently implements the published *count* only: each
// correlated kernel drops that many lowest-energy occupied spatial MOs.  Do
// not describe this helper as reproducing ORCA's orbital-character reorder.

#pragma once

#include <stdexcept>
#include <string>

#include "molecule.hpp"

namespace vibeqc {

// Number of frozen electrons assigned to one atom by ORCA 6.1 Table 2.69.
// The table ends at copernicium (Z=112), matching the manual's supported
// range. Z=0 is vibe-qc's conventional zero-electron ghost-centre extension.
// Counts are always even because they describe doubly occupied core shells in
// the molecular frozen-core convention.
inline int published_frozen_core_electrons_for_atomic_number(int z) {
    // Z=0 is the conventional ghost-centre placeholder.  It contributes
    // basis functions but no electrons, so it cannot contribute a frozen
    // core count in a counterpoise-style molecular inventory.
    if (z == 0) return 0;
    if (z < 0 || z > 112) {
        throw std::invalid_argument(
            "published frozen-core convention supports atomic numbers 0..112");
    }
    if (z <= 4) return 0;    // H-He, Li-Be
    if (z <= 12) return 2;   // B-Ne, Na-Mg
    if (z <= 30) return 10;  // Al-Ar, K-Zn
    if (z <= 38) return 18;  // Ga-Kr, Rb-Sr
    if (z <= 48) return 28;  // Y-Cd
    if (z <= 70) return 36;  // In-Xe, Cs-Ba, La-Yb
    if (z <= 80) return 46;  // Lu, Hf-Hg
    if (z <= 103) return 68; // Tl-Rn, Fr-Ra, Ac-Lr
    return 100;              // Rf-Cn
}

inline int published_frozen_core_orbitals_for_atomic_number(int z) {
    return published_frozen_core_electrons_for_atomic_number(z) / 2;
}

inline int published_frozen_core_orbital_count(const Molecule& molecule) {
    int orbitals = 0;
    for (const auto& atom : molecule.atoms()) {
        orbitals += published_frozen_core_orbitals_for_atomic_number(atom.Z);
    }
    return orbitals;
}

// Variational electron count of an SCF reference: the physical count minus
// the electrons an ECP removed.  Every correlated kernel partitions its
// occupied space from this number, never from Molecule::n_electrons().
inline int effective_electron_count(const Molecule& mol, int ecp_total_ncore,
                                    const char* route) {
    if (ecp_total_ncore < 0) {
        throw std::invalid_argument(
            std::string(route) + ": invalid negative ecp_total_ncore provenance");
    }
    const int physical = mol.n_electrons();
    if (ecp_total_ncore > physical) {
        throw std::invalid_argument(
            std::string(route) + ": ecp_total_ncore (" +
            std::to_string(ecp_total_ncore) +
            ") exceeds the molecule's physical electron count (" +
            std::to_string(physical) + ")");
    }
    return physical - ecp_total_ncore;
}

// Frozen-core count a native kernel applies.  The published default (-1) is
// element-based and does not know which cores an ECP already removed, so it
// is applied natively only to an all-electron reference; the Python wrappers
// resolve the ECP-aware count per atom
// (vibeqc.correlation_conventions.resolve_frozen_core_count(..., reference=))
// and pass it explicitly, and an ECP reference reaching -1 here fails rather
// than freezing valence orbitals.
inline int resolve_native_frozen_core(const Molecule& mol, int requested,
                                      int ecp_total_ncore, const char* route) {
    if (requested == -1) {
        if (ecp_total_ncore > 0) {
            throw std::invalid_argument(
                std::string(route) +
                ": n_frozen_core=-1 (published default) cannot be applied "
                "natively to an ECP reference (" +
                std::to_string(ecp_total_ncore) +
                " core electrons already removed); resolve the count with "
                "vibeqc.correlation_conventions.resolve_frozen_core_count("
                "molecule, None, reference=scf_result) and pass it");
        }
        return published_frozen_core_orbital_count(mol);
    }
    if (requested < 0) {
        throw std::invalid_argument(
            std::string(route) +
            ": n_frozen_core must be -1 (published default) or >= 0");
    }
    return requested;
}

}  // namespace vibeqc
