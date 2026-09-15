// Crystal structure + space-group analysis via spglib.
//
// Scope (M1 of periodic work):
//   - Crystal: lattice + fractional atomic positions + atomic numbers.
//   - analyze(): call spglib and return a SpaceGroup describing number,
//     international symbol, Hall number, point group, and the full list of
//     symmetry operations in fractional coordinates.
//   - to_primitive(): reduce an input cell to its primitive cell.
//
// Deliberately out of scope here: irreducible k-point mesh, Wyckoff letter
// lookup, magnetic space groups, layer groups, site-symmetry tables, CIF I/O.
// Add when a consumer exists.
//
// Convention. Lattice vectors are stored as the *columns* of a 3×3 matrix,
// so ``lattice.col(i)`` is the i-th lattice vector in Cartesian bohr.
// Fractional coordinates are stored one atom per column of a 3×N matrix.
// (Both choices mirror Eigen's default column-major layout and let us hand
// matrices to spglib via a single `(i,j)` copy at the FFI boundary.)

#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

namespace vibeqc {

struct Crystal {
    // Cartesian lattice vectors in bohr, columns a, b, c.
    Eigen::Matrix3d lattice = Eigen::Matrix3d::Identity();
    // Fractional coordinates, one atom per column.
    Eigen::Matrix3Xd fractional_coords;
    // Atomic numbers (one per atom). Distinct Zs → distinct spglib types.
    std::vector<int> species;

    int n_atoms() const {
        return static_cast<int>(species.size());
    }
};

struct SymmetryOp {
    // Rotation matrix in the fractional basis (integer entries, as spglib
    // returns them) and translation in fractional coordinates.
    Eigen::Matrix3i rotation = Eigen::Matrix3i::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

struct SpaceGroup {
    int number = 0;                          // International Tables 1..230
    std::string international_symbol;        // e.g. "Fm-3m"
    int hall_number = 0;
    std::string point_group;                 // e.g. "m-3m"
    std::vector<SymmetryOp> operations;      // Full list (order of the group)
    std::vector<int> equivalent_atoms;       // Index of symmetry representative
                                             // for each input atom.
};

// Analyze the space-group of a Crystal. ``symprec`` is the Cartesian distance
// tolerance in bohr for atomic-position matching; spglib's default is 1e-5 Å,
// we expose the same number converted to bohr.
//
// Throws std::runtime_error if spglib fails (e.g. atoms too close, degenerate
// lattice).
SpaceGroup analyze(const Crystal& crystal, double symprec = 1.0e-5);

// Reduce to primitive cell. Returns a new Crystal; the input is unchanged.
// ``symprec`` is the same tolerance used by spglib internally.
Crystal to_primitive(const Crystal& crystal, double symprec = 1.0e-5);

// Irreducible reciprocal-mesh grid points, returned as fractional coordinates
// in the reciprocal lattice basis. ``mesh[i]`` is the number of divisions
// along reciprocal axis i; ``is_shift[i] ∈ {0,1}`` shifts the grid by half
// a step, i.e. to (m + ½)/N instead of m/N.
//
// That half-step offset is the EVEN-mesh case. Monkhorst & Pack, Phys. Rev.
// B 13, 5188 (1976), Eq. (3) puts the classical points at
// u_r = (2r − q − 1)/(2q): for odd q the set contains 0 and already agrees
// with is_shift = 0, and only for even q does the classical mesh sit off
// Γ and agree with is_shift = 1.
//
// The output ``ir_mapping`` has length ∏ mesh[i] and maps each full-mesh
// point to an index into the returned unique list.
struct IrreducibleKMesh {
    std::vector<Eigen::Vector3d> fractional_kpoints; // (3, N_ir)
    std::vector<double> weights;                     // sum to 1
    std::vector<int> ir_mapping;                     // length ∏ mesh[i]
};
IrreducibleKMesh irreducible_kpoints(const Crystal& crystal,
                                     const std::array<int, 3>& mesh,
                                     const std::array<int, 3>& is_shift = {0, 0, 0},
                                     double symprec = 1.0e-5);

// Version string of the linked spglib library ("x.y.z").
std::string spglib_version();

}  // namespace vibeqc
