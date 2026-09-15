// A molecule: a list of atoms plus charge and spin multiplicity.
// All positions are stored in bohr.

#pragma once

#include <array>
#include <vector>

namespace vibeqc {

struct Atom {
    int Z;                        // atomic number
    std::array<double, 3> xyz;    // position in bohr
};

class Molecule {
public:
    Molecule(std::vector<Atom> atoms, int charge = 0, int multiplicity = 1);

    const std::vector<Atom>& atoms() const noexcept { return atoms_; }
    int charge() const noexcept { return charge_; }
    int multiplicity() const noexcept { return multiplicity_; }
    int n_electrons() const noexcept;
    double nuclear_repulsion() const;

private:
    std::vector<Atom> atoms_;
    int charge_;
    int multiplicity_;
};

}  // namespace vibeqc
