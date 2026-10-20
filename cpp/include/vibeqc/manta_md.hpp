#pragma once

// Independently implemented McMurchie-Davidson primitives. No integral
// library is used here. The contraction/provider layer owns normalization.
#include <Eigen/Dense>
#include <array>
#include <complex>
#include <vector>

namespace vibeqc::manta::md {
using Index = std::array<int,3>;
using Point = std::array<double,3>;
using Matrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using ComplexMatrix = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
constexpr int maximum_order = 32;

std::vector<double> boys(int maximum_n, double argument);
enum class RadialKernel { Coulomb, Erf, Erfc };
std::vector<double> radial_boys(int maximum_n,double argument,double rho,RadialKernel,double omega);
std::vector<Index> cartesians(int l);
std::vector<Index> hermites(int l);
Matrix representation(int l, bool pure);

// E rows enumerate Cartesian pairs (a,b), b contiguous; columns enumerate
// Hermite triples by total degree. Unit factors for 2c/3c have l=0, exponent=0
// internally; they are not normalized orbitals and never enter a user basis.
struct Pair {
    double exponent = 0;
    Point center{};
    std::vector<Index> indices;
    Matrix coefficients;
};
Pair pair(int la, int lb, double alpha, double beta, const Point& a, const Point& b);
Matrix coulomb(const Pair&, const Pair&,RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
Matrix overlap(int la, int lb, double alpha, double beta, const Point& a, const Point& b);
Matrix kinetic(int la, int lb, double alpha, double beta, const Point& a, const Point& b);
Matrix potential(const Pair&, const Point& charge_position, double charge,
                 RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
Matrix potential_charge_derivative(const Pair&,const Point& charge_position,double charge,int axis,
                                   RadialKernel kernel=RadialKernel::Coulomb,double omega=0.);
Matrix moment(int la, int lb, double alpha, double beta, const Point& a, const Point& b,
              const Point& origin, const Index& powers);
Matrix moments(int la,int lb,double alpha,double beta,const Point& a,const Point& b,
               const Point& origin,int maximum_degree);
ComplexMatrix fourier_panel(const Pair&,const Matrix& wave_vectors);
Eigen::VectorXcd fourier(const Pair&, const Point& wave_vector);
} // namespace vibeqc::manta::md
