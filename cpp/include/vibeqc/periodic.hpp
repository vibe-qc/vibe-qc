// Periodic system: unit cell + lattice + dimensionality.
//
// Dimensionality convention (applies throughout the periodic stack):
//   dim = 1 → polymer; only lattice.col(0) is a lattice vector, cols 1..2
//            are non-periodic directions (atoms at whatever Cartesian
//            positions they occupy; no lattice sum along those axes).
//   dim = 2 → slab / layer; cols 0..1 are the in-plane lattice vectors and
//            col 2 is a synthesized *bookkeeping* normal (see below), NOT a
//            physical vacuum gap. Atoms carry their real Cartesian z.
//   dim = 3 → bulk; all three columns are lattice vectors.
//
// The lattice matrix is always full-rank 3×3 regardless of dim. This keeps
// AO integrals (which operate in full 3-space) and spglib (which demands a
// 3D lattice) happy with no special-casing. Lattice sums iterate only over
// the first `dim` lattice axes.
//
// For dim < 3 the columns dim..2 are NOT a periodicity/vacuum parameter: the
// rigorous low-dimensional Coulomb gauges (SLAB_EWALD_2D for slabs; the 1D
// wire sum) are provably invariant to them. Build a dim=2 system from two
// in-plane vectors via `vibeqc.slab_2d(...)`, which auto-synthesizes col 2
// along the normal n̂=(a1×a2)/|a1×a2| at a bookkeeping length (~30 bohr or
// atom-extent + padding). A large col-2 vector is only ever needed as a grid
// / density-cube / viewer extent, never for the SCF. (The earlier convention
// that documented col 2 as a load-bearing ~30-bohr vacuum gap was a
// plane-wave idiom and is retired: a Gaussian slab needs no vacuum.)

#pragma once

#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "crystal.hpp"
#include "molecule.hpp"

namespace vibeqc {

struct PeriodicSystem {
    // 1, 2, or 3. Number of periodic dimensions.
    int dim = 3;

    // Full 3×3 lattice matrix in bohr. Columns = Cartesian lattice vectors.
    // For dim < 3, columns dim..2 are non-physical: they are synthesized to
    // keep the lattice full-rank for the AO integrals and spglib. They are
    // not cell edges and not repeat vectors, and the total energy is
    // invariant to their length (see header comment).
    Eigen::Matrix3d lattice = Eigen::Matrix3d::Identity() * 30.0;

    // Unit-cell atoms. Positions are Cartesian (bohr), not fractional —
    // matches the Molecule convention used everywhere else in the code.
    std::vector<Atom> unit_cell;

    // Total charge per unit cell and spin multiplicity (the latter only
    // meaningful at Γ for a supercell-like calculation; formally ignored
    // in bulk HF).
    int charge = 0;
    int multiplicity = 1;

    // Space-group analysis, populated lazily via attach_symmetry().
    // When set, downstream code may use it to reduce the k-mesh to the
    // IBZ and (in later phases) reduce real-space cell / pair lists.
    std::optional<SpaceGroup> symmetry;

    // Convenience: pack the unit-cell atoms into a Molecule so we can reuse
    // BasisSet, compute_overlap, etc. unchanged.
    Molecule unit_cell_molecule() const {
        int n_elec = -charge;
        for (const auto& atom : unit_cell) n_elec += atom.Z;

        int mol_multiplicity = multiplicity;
        const int unpaired = mol_multiplicity - 1;
        const int paired = n_elec - unpaired;
        if (mol_multiplicity < 1 || n_elec < 0 ||
            paired < 0 || paired % 2 != 0) {
            mol_multiplicity = (n_elec % 2 == 0) ? 1 : 2;
        }
        return Molecule(unit_cell, charge, mol_multiplicity);
    }

    // Reciprocal lattice (with 2π factor), columns = b1, b2, b3.
    // For dim < 3, reciprocal vectors for vacuum axes are formally defined
    // but unused in k-point and Bloch-sum machinery.
    Eigen::Matrix3d reciprocal_lattice() const;

    // Number of electrons per unit cell, derived from species + charge.
    int n_electrons() const;
};

// Attach a spglib-derived space group analysis to a PeriodicSystem. The
// input lattice/positions are converted to spglib's conventions internally.
// For dim < 3, the vacuum directions are passed through to spglib and the
// resulting space group may include 1D/2D subperiodic operations (spglib
// treats all input as 3D; filtering is a later-phase concern).
void attach_symmetry(PeriodicSystem& system, double symprec = 1.0e-5);

}  // namespace vibeqc
