#include "vibeqc/molecule.hpp"

#include <cmath>
#include <stdexcept>

namespace vibeqc {

Molecule::Molecule(std::vector<Atom> atoms, int charge, int multiplicity)
    : atoms_(std::move(atoms)), charge_(charge), multiplicity_(multiplicity) {
    if (multiplicity_ < 1) {
        throw std::invalid_argument("multiplicity must be >= 1");
    }
    if (n_electrons() < 0) {
        throw std::invalid_argument("net electron count is negative");
    }
    // Consistency: n_electrons parity must match multiplicity parity.
    // (multiplicity = 2S + 1, so (n_electrons - (mult - 1)) must be even and >= 0.)
    const int unpaired = multiplicity_ - 1;
    const int paired = n_electrons() - unpaired;
    if (paired < 0 || paired % 2 != 0) {
        throw std::invalid_argument(
            "n_electrons and multiplicity are inconsistent");
    }
}

int Molecule::n_electrons() const noexcept {
    int z_total = 0;
    for (const auto& a : atoms_) z_total += a.Z;
    return z_total - charge_;
}

double Molecule::nuclear_repulsion() const {
    constexpr double eps = 1e-12;
    double e_nuc = 0.0;
    for (std::size_t i = 0; i < atoms_.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms_.size(); ++j) {
            const double dx = atoms_[i].xyz[0] - atoms_[j].xyz[0];
            const double dy = atoms_[i].xyz[1] - atoms_[j].xyz[1];
            const double dz = atoms_[i].xyz[2] - atoms_[j].xyz[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < eps) {
                throw std::runtime_error("atoms coincide (distance ~ 0)");
            }
            e_nuc += static_cast<double>(atoms_[i].Z * atoms_[j].Z) / r;
        }
    }
    return e_nuc;
}

}  // namespace vibeqc
