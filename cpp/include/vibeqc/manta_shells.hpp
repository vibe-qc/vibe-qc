#pragma once

#include "vibeqc/basis.hpp"
#include "vibeqc/manta_md.hpp"
#include <utility>
#include <cstdint>

namespace vibeqc::manta::md {
using Charges = std::vector<std::pair<double,Point>>;
struct SharedCoulombTiles {
    std::vector<Matrix> values;
    std::uint64_t primitive_evaluations=0;
};
struct CoulombDerivative {
    Matrix values;
    std::uint64_t primitive_evaluations=0;
};

// Internal contracted-shell kernels. The provider must admit shapes, work,
// memory and numerical payload before calling these routines. Coefficients
// already include shell-wide axial primitive normalization and contraction.
// Null slots denote an internal constant factor for 2c/3c Coulomb kernels.
Matrix shell_coulomb(const std::array<const libint2::Shell*,4>&,
                     RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
SharedCoulombTiles shell_coulomb_shared(const std::vector<std::array<const libint2::Shell*,4>>&,
                                      RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
CoulombDerivative shell_coulomb_derivative(const std::array<const libint2::Shell*,4>&,int center,int axis,
                                           RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
Matrix shell_one_electron(const libint2::Shell&,const libint2::Shell&,
                          const std::string& operation,const Charges& charges={},
                          const Point& origin={},const Index& powers={},
                          RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
Matrix shell_moments(const libint2::Shell&,const libint2::Shell&,const Point& origin,int maximum_degree);
Matrix shell_auxiliary_moments(const libint2::Shell&,const Point& origin,int maximum_degree);
Matrix shell_one_electron_derivative(const libint2::Shell&,const libint2::Shell&,
                          const std::string& operation,const Charges&,const Point&,const Index&,
                          const std::string& target,int axis,std::size_t charge_index,
                          RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
ComplexMatrix shell_fourier_panel(const libint2::Shell*,const libint2::Shell*,const Matrix& wave_vectors);
Eigen::VectorXcd shell_fourier(const libint2::Shell&,const libint2::Shell&,const Point&);
} // namespace vibeqc::manta::md
