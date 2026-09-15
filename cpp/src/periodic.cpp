#include "vibeqc/periodic.hpp"

#include <stdexcept>

namespace vibeqc {

Eigen::Matrix3d PeriodicSystem::reciprocal_lattice() const {
    const double det = lattice.determinant();
    if (std::abs(det) < 1.0e-14) {
        throw std::runtime_error(
            "PeriodicSystem::reciprocal_lattice: singular lattice matrix");
    }
    // b = 2π · (a⁻¹)ᵀ so that a_i · b_j = 2π δ_ij.
    return 2.0 * M_PI * lattice.inverse().transpose();
}

int PeriodicSystem::n_electrons() const {
    int z_total = 0;
    for (const auto& a : unit_cell) z_total += a.Z;
    return z_total - charge;
}

void attach_symmetry(PeriodicSystem& system, double symprec) {
    // Build a Crystal with fractional coordinates and call analyze().
    // Fractional = lattice⁻¹ · cartesian.
    const auto lat_inv = system.lattice.inverse();
    Crystal c;
    c.lattice = system.lattice;
    c.fractional_coords.resize(3, system.unit_cell.size());
    c.species.resize(system.unit_cell.size());
    for (std::size_t i = 0; i < system.unit_cell.size(); ++i) {
        const auto& a = system.unit_cell[i];
        Eigen::Vector3d r(a.xyz[0], a.xyz[1], a.xyz[2]);
        c.fractional_coords.col(i) = lat_inv * r;
        c.species[i] = a.Z;
    }
    system.symmetry = analyze(c, symprec);
}

}  // namespace vibeqc
