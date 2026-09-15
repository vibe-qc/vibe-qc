#include "vibeqc/periodic_ao_bloch_transport.hpp"

#include <libint2/solidharmonics.h>
#include <algorithm>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace vibeqc {
namespace {

using U = std::uint64_t;
using I = std::int64_t;
using Z = std::complex<double>;
constexpr int Lmax = kPeriodicAOBlochTransportMaximumAngularMomentum;
constexpr int Nc = (Lmax + 1) * (Lmax + 2) / 2;
constexpr int Np = 2 * Lmax + 1;
constexpr int Side = Lmax + 1;
constexpr int Poly = Side * Side * Side;
constexpr double TwoPi = 6.283185307179586476925286766559005768;

[[noreturn]] void invalid(const char* text) {
    throw std::invalid_argument(text);
}
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("AO Bloch transport count overflow");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("AO Bloch transport count overflow");
    return a * b;
}
I iadd(I a, I b) {
    if ((b > 0 && a > std::numeric_limits<I>::max() - b) ||
        (b < 0 && a < std::numeric_limits<I>::min() - b))
        throw std::overflow_error("AO Bloch transport exact integer overflow");
    return a + b;
}
I ineg(I a) {
    if (a == std::numeric_limits<I>::min())
        throw std::overflow_error("AO Bloch transport exact integer overflow");
    return -a;
}
I imul(I a, I b) {
    if (a == 0 || b == 0) return 0;
    if ((a == -1 && b == std::numeric_limits<I>::min()) ||
        (b == -1 && a == std::numeric_limits<I>::min()))
        throw std::overflow_error("AO Bloch transport exact integer overflow");
    if (a > 0 ? (b > 0 ? a > std::numeric_limits<I>::max() / b
                          : b < std::numeric_limits<I>::min() / a)
              : (b > 0 ? a < std::numeric_limits<I>::min() / b
                          : a < std::numeric_limits<I>::max() / b))
        throw std::overflow_error("AO Bloch transport exact integer overflow");
    return a * b;
}
I imod(I a, I m) {
    const I r = a % m;
    return r < 0 ? r + m : r;
}
void cap(U value, U bound, const char* message) {
    if (value > bound) invalid(message);
}
double finite(double x) {
    if (!std::isfinite(x))
        throw std::overflow_error("AO Bloch transport nonfinite numerical value");
    return x;
}
void finite_complex(Z x) {
    finite(x.real());
    finite(x.imag());
}
void controls(const PeriodicAOBlochTransportOptions& o,
              const PeriodicAOBlochTransportInventory& live) {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    invalid("AO Bloch transport requires strict floating point compilation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || sizeof(double) != 8 ||
        std::numeric_limits<double>::digits != 53 ||
        std::numeric_limits<double>::max_exponent != 1024 || FLT_RADIX != 2 ||
        sizeof(Z) != 16 || std::fegetround() != FE_TONEAREST)
        invalid("AO Bloch transport requires binary64 round-to-nearest arithmetic");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.0, zero = 0.0;
    volatile double smallest_normal = std::numeric_limits<double>::min(), half = 0.5;
    if (!(tiny > 0) || tiny * one != tiny || tiny + zero != tiny ||
        std::fma(tiny, one, zero) != tiny || std::nextafter(0.0, 1.0) != tiny ||
        smallest_normal * half == 0)
        invalid("AO Bloch transport requires gradual underflow");
    const double tolerances[] = {
        o.maximum_atom_mapping_residual_bohr,
        o.maximum_basis_origin_residual_bohr,
        o.maximum_rotation_orthogonality_residual,
        o.maximum_polynomial_reconstruction_residual,
        o.maximum_pure_rotation_unitarity_residual};
    for (double x : tolerances)
        if (!std::isfinite(x) || x < 0)
            invalid("AO Bloch transport explicit finite nonnegative audit controls required");
    if (!std::isfinite(o.minimum_relative_lattice_volume) ||
        o.minimum_relative_lattice_volume <= 0 || o.minimum_relative_lattice_volume >= 1)
        invalid("AO Bloch transport relative lattice volume floor must be in (0,1)");
    if (!live.numerical_replicas || !live.backend_margin_bytes_per_replica)
        invalid("AO Bloch transport positive replica count and backend margin required");
}

using AtomMap = detail::PeriodicAOAtomMap;
using ShellMap = detail::PeriodicAOShellMap;
struct Work {
    std::array<std::array<int,3>,Nc> powers{};
    std::array<double,Nc*Nc> cart{}, rotation{};
    std::array<double,Np*Nc> pure{}, transformed{};
    std::array<double,Np*Np> cholesky{};
    std::array<double,Poly> polynomial{}, next{};
    std::array<double,Np> rhs{};
    // A, A^-1, R and R^-1. All matrices are row-major scalar arrays.
    std::array<double,9> a{}, inverse{}, r{}, inverse_r{};
    std::array<double,3> translation{};
};
// Additional simultaneously live geometry temporaries (two 3x3 arrays,
// three 3-vectors) and compensated scalar accumulators. This explicit
// conservative numerical reservation is separate from descriptor controls.
constexpr U FixedScalarNumericalReservation = 512;

// W^-T = cofactor(W)/det(W). No floating inverse or rounded k address.
std::array<I,9> reciprocal_rotation(const SymmetryOp& op) {
    std::array<I,9> cof{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int a = (i + 1) % 3, b = (i + 2) % 3;
            const int c = (j + 1) % 3, d = (j + 2) % 3;
            cof[3*i+j] = iadd(imul(op.rotation(a,c), op.rotation(b,d)),
                              ineg(imul(op.rotation(a,d), op.rotation(b,c))));
        }
    }
    I determinant = 0;
    for (int j = 0; j < 3; ++j)
        determinant = iadd(determinant, imul(op.rotation(0,j), cof[j]));
    if (determinant != 1 && determinant != -1)
        invalid("AO Bloch transport requires a unimodular integer Seitz rotation");
    if (determinant == -1)
        for (I& x : cof) x = ineg(x);
    // Independent adjugate identity catches implementation/convention errors.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            I x = 0;
            for (int k = 0; k < 3; ++k)
                x = iadd(x, imul(op.rotation(i,k), cof[3*j+k]));
            if (x != (i == j ? 1 : 0))
                invalid("AO Bloch transport integer inverse identity failed");
        }
    return cof;
}

void mesh_action(PeriodicAOBlochTransportPlan& p, const PeriodicSystem& system,
                 const SymmetryOp& op, const RegularKMesh& mesh) {
    if (system.dim < 1 || system.dim > 3)
        invalid("AO Bloch transport invalid periodic dimension");
    if (p.source_index >= mesh.size())
        throw std::out_of_range("AO Bloch transport source k index");
    for (int d = system.dim; d < 3; ++d) {
        if (mesh.mesh()[d] != 1 || mesh.is_shift()[d] != 0)
            invalid("AO Bloch transport nonperiodic axis must have a Gamma-only mesh");
        for (int e = 0; e < system.dim; ++e)
            if (op.rotation(d,e) != 0)
                invalid("AO Bloch transport rotation does not preserve periodic subspace");
    }
    const auto v = reciprocal_rotation(op);
    std::array<I,9> action{};
    for (int d = 0; d < 3; ++d) {
        I parity = 0;
        for (int e = 0; e < 3; ++e) {
            const I numerator = imul(mesh.mesh()[d], v[3*d+e]);
            if (numerator % mesh.mesh()[e] != 0)
                invalid("AO Bloch transport rotation is incompatible with the whole mesh");
            I x = numerator / mesh.mesh()[e];
            if (p.time_reversal) x = ineg(x);
            action[3*d+e] = x;
            parity = iadd(parity, imul(x, mesh.is_shift()[e]));
        }
        if (imod(parity, 2) != mesh.is_shift()[d])
            invalid("AO Bloch transport operation is incompatible with whole-mesh shift parity");
    }
    const auto source = mesh.address(p.source_index);
    KGridAddress target;
    for (int d = 0; d < 3; ++d) {
        I raw = 0;
        for (int e = 0; e < 3; ++e)
            raw = iadd(raw, imul(action[3*d+e], source.doubled[e]));
        const I modulus = mesh.doubled_modulus()[d];
        const I reduced = imod(raw, modulus);
        target.doubled[d] = static_cast<int>(reduced);
        // Avoid raw-reduced overflow at INT64_MIN.
        p.reciprocal_wrap[d] = raw / modulus - (raw % modulus < 0 ? 1 : 0);
    }
    p.target_doubled_address = target.doubled;
    p.target_index = mesh.index(target);
}

void sum(double& s, double& c, double x) {
    finite(x);
    const double t = finite(s + x);
    c = finite(c + (std::abs(s) >= std::abs(x) ? (s-t)+x : (x-t)+s));
    s = t;
}
double dot3(const double* a, const double* b) {
    double s = 0, c = 0;
    for (int d = 0; d < 3; ++d) sum(s,c,finite(a[d]*b[d]));
    return finite(s+c);
}
double norm3(const std::array<double,3>& x) {
    return finite(std::hypot(x[0],x[1],x[2]));
}

void geometry(Work& w, const PeriodicSystem& system, const SymmetryOp& op,
              const PeriodicAOBlochTransportOptions& options,
              PeriodicAOBlochTransportDiagnostics& diag) {
    double scale = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            w.a[3*i+j] = finite(system.lattice(i,j));
            scale = std::max(scale,std::abs(w.a[3*i+j]));
        }
    if (scale == 0) invalid("AO Bloch transport singular lattice");
    std::array<double,9> a{}, cof{};
    for (int k = 0; k < 9; ++k) a[k] = w.a[k]/scale;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const int r = (i+1)%3, s = (i+2)%3;
            const int t = (j+1)%3, u = (j+2)%3;
            cof[3*i+j] = finite(a[3*r+t]*a[3*s+u]-a[3*r+u]*a[3*s+t]);
        }
    const double det = dot3(a.data(),cof.data());
    if (std::abs(det) < options.minimum_relative_lattice_volume)
        invalid("AO Bloch transport relative lattice volume floor");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            w.inverse[3*i+j] = finite((cof[3*j+i]/det)/scale);
    const auto v = reciprocal_rotation(op);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double r = 0, rc = 0, ri = 0, ric = 0;
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) {
                    sum(r,rc,finite(finite(w.a[3*i+k]*op.rotation(k,l))*w.inverse[3*l+j]));
                    sum(ri,ric,finite(finite(w.a[3*i+k]*static_cast<double>(v[3*l+k]))*
                                      w.inverse[3*l+j]));
                }
            w.r[3*i+j] = finite(r+rc);
            w.inverse_r[3*i+j] = finite(ri+ric);
        }
        double t = 0, tc = 0;
        for (int j = 0; j < 3; ++j)
            sum(t,tc,finite(w.a[3*i+j]*finite(op.translation[j])));
        w.translation[i] = finite(t+tc);
    }
    double error = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0, c = 0;
            for (int k = 0; k < 3; ++k) sum(s,c,w.r[3*k+i]*w.r[3*k+j]);
            error = std::max(error,std::abs(finite(s+c)-(i == j ? 1.0 : 0.0)));
        }
    diag.rotation_orthogonality_residual = error;
    if (error > options.maximum_rotation_orthogonality_residual)
        invalid("AO Bloch transport Cartesian rotation orthogonality audit");
    for (const auto& atom : system.unit_cell) {
        if (atom.Z <= 0) invalid("AO Bloch transport invalid atomic species");
        for (double x : atom.xyz) finite(x);
    }
}

void atom_mapping(const Work& w, const PeriodicSystem& system,
                  const PeriodicAOBlochTransportOptions& options,
                  std::vector<AtomMap>& map, PeriodicAOBlochTransportDiagnostics& diag) {
    for (U a = 0; a < map.size(); ++a) {
        std::array<double,3> moved{};
        for (int d = 0; d < 3; ++d)
            moved[d] = finite(dot3(&w.r[3*d],system.unit_cell[a].xyz.data())+w.translation[d]);
        U matches = 0;
        AtomMap found;
        double found_error = 0;
        for (U b = 0; b < map.size(); ++b) {
            if (system.unit_cell[a].Z != system.unit_cell[b].Z) continue;
            std::array<double,3> delta{}, error{};
            for (int d = 0; d < 3; ++d) delta[d] = finite(system.unit_cell[b].xyz[d]-moved[d]);
            std::array<I,3> ell{};
            bool periodic = true;
            for (int d = 0; d < 3; ++d) {
                const double frac = dot3(&w.inverse[3*d],delta.data());
                const double nearest = std::nearbyint(frac);
                // Every integer used in Cartesian reconstruction remains exactly representable.
                if (std::abs(nearest) > 4503599627370496.0)
                    throw std::overflow_error("AO Bloch transport atom lattice shift range");
                ell[d] = static_cast<I>(nearest);
                if (d >= system.dim && ell[d] != 0) periodic = false;
            }
            if (!periodic) continue;
            for (int d = 0; d < 3; ++d) {
                double s = 0, c = 0;
                for (int e = 0; e < 3; ++e) sum(s,c,w.a[3*d+e]*static_cast<double>(ell[e]));
                error[d] = finite(delta[d]-finite(s+c));
            }
            const double residual = norm3(error);
            if (residual <= options.maximum_atom_mapping_residual_bohr) {
                ++matches;
                found = {b,ell};
                found_error = residual;
            }
        }
        if (matches != 1)
            invalid("AO Bloch transport atom mapping must be unique and species preserving");
        for (U previous = 0; previous < a; ++previous)
            if (map[previous].destination == found.destination)
                invalid("AO Bloch transport atom mapping is not bijective");
        map[a] = found;
        diag.maximum_atom_mapping_residual_bohr =
            std::max(diag.maximum_atom_mapping_residual_bohr,found_error);
    }
    diag.mapped_atoms = map.size();
}

bool equal_radial(const libint2::Shell& a, const libint2::Shell& b) {
    if (a.alpha.size() != b.alpha.size() || a.contr.size() != b.contr.size()) return false;
    for (U p = 0; p < a.alpha.size(); ++p) if (a.alpha[p] != b.alpha[p]) return false;
    for (U c = 0; c < a.contr.size(); ++c) {
        const auto& x = a.contr[c];
        const auto& y = b.contr[c];
        if (x.l != y.l || x.pure != y.pure || x.coeff.size() != y.coeff.size()) return false;
        for (U p = 0; p < x.coeff.size(); ++p) if (x.coeff[p] != y.coeff[p]) return false;
    }
    return true;
}
void shell_mapping(const BasisSet& basis, const PeriodicSystem& system,
                   const std::vector<AtomMap>& atoms,
                   const PeriodicAOBlochTransportOptions& options,
                   std::vector<ShellMap>& map, PeriodicAOBlochTransportDiagnostics& diag) {
    const auto& shells = basis.libint();
    for (U s = 0; s < shells.size(); ++s) {
        const auto& sh = shells[s];
        const auto atom = static_cast<U>(basis.shell_atom_index(s));
        std::array<double,3> e{};
        for (int d = 0; d < 3; ++d) e[d] = finite(finite(sh.O[d])-system.unit_cell[atom].xyz[d]);
        const double residual = norm3(e);
        diag.maximum_basis_origin_residual_bohr =
            std::max(diag.maximum_basis_origin_residual_bohr,residual);
        if (residual > options.maximum_basis_origin_residual_bohr)
            invalid("AO Bloch transport basis origin does not match its owning atom");
        for (double exponent : sh.alpha)
            if (!std::isfinite(exponent) || exponent <= 0)
                invalid("AO Bloch transport Gaussian exponent must be finite and positive");
        for (const auto& c : sh.contr)
            for (double coefficient : c.coeff) finite(coefficient);
    }
    for (U s = 0; s < shells.size(); ++s) {
        const auto destination_atom = atoms[basis.shell_atom_index(s)].destination;
        bool matched = false;
        for (U t = 0; t < shells.size(); ++t) {
            if (static_cast<U>(basis.shell_atom_index(t)) != destination_atom) continue;
            bool used = false;
            for (U previous = 0; previous < s; ++previous)
                if (map[previous].destination == t) { used = true; break; }
            if (used || !equal_radial(shells[s],shells[t])) continue;
            map[s] = {t,static_cast<U>(shells.shell2bf()[s]),static_cast<U>(shells.shell2bf()[t])};
            matched = true;
            break;
        }
        if (!matched)
            invalid("AO Bloch transport requires a bijective exact radial/angular/pure shell match");
    }
    diag.mapped_shells = map.size();
}

int poly_index(int x, int y, int z) { return (x*Side+y)*Side+z; }

// Polynomial substitution is in the native axial-normalized Cartesian basis.
// M[u,v] expands monomial_u(R^-1 x) into monomial_v(x).
void rotation_block(Work& w, int l, bool pure,
                    const PeriodicAOBlochTransportOptions& options,
                    PeriodicAOBlochTransportDiagnostics& diag) {
    const int nc = (l+1)*(l+2)/2, np = 2*l+1;
    int at = 0;
    for (int x = l; x >= 0; --x)
        for (int y = l-x; y >= 0; --y) w.powers[at++] = {x,y,l-x-y};
    for (int u = 0; u < nc; ++u) {
        w.polynomial.fill(0);
        w.polynomial[0] = 1;
        int degree = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int repeat = 0; repeat < w.powers[u][axis]; ++repeat) {
                w.next.fill(0);
                for (int x = degree; x >= 0; --x)
                    for (int y = degree-x; y >= 0; --y) {
                        const int z = degree-x-y;
                        const double p = w.polynomial[poly_index(x,y,z)];
                        for (int d = 0; d < 3; ++d) {
                            const int index = poly_index(x+(d==0),y+(d==1),z+(d==2));
                            w.next[index] = finite(w.next[index]+finite(p*w.inverse_r[3*axis+d]));
                        }
                    }
                w.polynomial.swap(w.next);
                ++degree;
            }
        }
        for (int v = 0; v < nc; ++v)
            w.cart[u*Nc+v] = w.polynomial[poly_index(w.powers[v][0],w.powers[v][1],w.powers[v][2])];
    }
    if (!pure) {
        for (int a = 0; a < nc; ++a)
            for (int b = 0; b < nc; ++b) w.rotation[a*Nc+b] = w.cart[b*Nc+a];
        ++diag.cartesian_rotation_blocks;
        return; // Cartesian polynomial reconstruction is the identity, not an orthogonal projection.
    }
    using libint2::solidharmonics::SolidHarmonicsCoefficients;
    for (int a = 0; a < np; ++a)
        for (int v = 0; v < nc; ++v)
            w.pure[a*Nc+v] = finite(SolidHarmonicsCoefficients<double>::coeff(
                l,a-l,w.powers[v][0],w.powers[v][1],w.powers[v][2]));
    for (int a = 0; a < np; ++a)
        for (int v = 0; v < nc; ++v) {
            double s = 0, c = 0;
            for (int u = 0; u < nc; ++u) sum(s,c,w.pure[a*Nc+u]*w.cart[u*Nc+v]);
            w.transformed[a*Nc+v] = finite(s+c);
        }
    // Algebraic coefficient Gram T T^t, not the physical Cartesian AO metric.
    for (int a = 0; a < np; ++a)
        for (int b = 0; b <= a; ++b) {
            double s = 0, c = 0;
            for (int v = 0; v < nc; ++v) sum(s,c,w.pure[a*Nc+v]*w.pure[b*Nc+v]);
            for (int k = 0; k < b; ++k) sum(s,c,-w.cholesky[a*Np+k]*w.cholesky[b*Np+k]);
            const double x = finite(s+c);
            if (a == b) {
                if (x <= 0) invalid("AO Bloch transport singular native pure polynomial basis");
                w.cholesky[a*Np+b] = std::sqrt(x);
            } else w.cholesky[a*Np+b] = finite(x/w.cholesky[b*Np+b]);
        }
    for (int source = 0; source < np; ++source) {
        for (int a = 0; a < np; ++a) {
            double s = 0, c = 0;
            for (int v = 0; v < nc; ++v) sum(s,c,w.pure[a*Nc+v]*w.transformed[source*Nc+v]);
            for (int k = 0; k < a; ++k) sum(s,c,-w.cholesky[a*Np+k]*w.rhs[k]);
            w.rhs[a] = finite(finite(s+c)/w.cholesky[a*Np+a]);
        }
        for (int a = np-1; a >= 0; --a) {
            double s = w.rhs[a], c = 0;
            for (int k = a+1; k < np; ++k) sum(s,c,-w.cholesky[k*Np+a]*w.rotation[k*Nc+source]);
            w.rotation[a*Nc+source] = finite(finite(s+c)/w.cholesky[a*Np+a]);
        }
    }
    double reconstruction = 0, unitarity = 0;
    for (int source = 0; source < np; ++source)
        for (int v = 0; v < nc; ++v) {
            double s = 0, c = 0;
            for (int a = 0; a < np; ++a) sum(s,c,w.rotation[a*Nc+source]*w.pure[a*Nc+v]);
            reconstruction = std::max(reconstruction,std::abs(finite(s+c)-w.transformed[source*Nc+v]));
        }
    for (int a = 0; a < np; ++a)
        for (int b = 0; b < np; ++b) {
            double s = 0, c = 0;
            for (int k = 0; k < np; ++k) sum(s,c,w.rotation[k*Nc+a]*w.rotation[k*Nc+b]);
            unitarity = std::max(unitarity,std::abs(finite(s+c)-(a==b ? 1.0 : 0.0)));
        }
    diag.maximum_polynomial_reconstruction_residual =
        std::max(diag.maximum_polynomial_reconstruction_residual,reconstruction);
    diag.maximum_pure_rotation_unitarity_residual =
        std::max(diag.maximum_pure_rotation_unitarity_residual,unitarity);
    if (reconstruction > options.maximum_polynomial_reconstruction_residual)
        invalid("AO Bloch transport pure polynomial reconstruction audit");
    if (unitarity > options.maximum_pure_rotation_unitarity_residual)
        invalid("AO Bloch transport pure rotation unitarity audit");
    ++diag.pure_rotation_blocks;
}

Z atom_phase(const AtomMap& a, const RegularKMesh& mesh,
             const PeriodicAOBlochTransportPlan& p) {
    double turns = 0, compensation = 0;
    for (int d = 0; d < 3; ++d) {
        const I modulus = mesh.doubled_modulus()[d];
        const I numerator = imod(imul(imod(a.ell[d],modulus),p.target_doubled_address[d]),modulus);
        sum(turns,compensation,static_cast<double>(numerator)/static_cast<double>(modulus));
    }
    turns = finite(turns+compensation);
    turns -= std::nearbyint(turns);
    if (turns == 0) return {1,0};
    if (std::abs(turns) == 0.5) return {-1,0};
    if (turns == 0.25) return {0,1};
    if (turns == -0.25) return {0,-1};
    return {std::cos(TwoPi*turns),std::sin(TwoPi*turns)};
}

} // namespace

PeriodicAOBlochTransportPlan plan_periodic_ao_bloch_transport(
    const BasisSet& basis, const PeriodicSystem& system, const SymmetryOp& op,
    const RegularKMesh& mesh, U source_index, bool time_reversal, U columns,
    const PeriodicAOBlochTransportOptions& options,
    const PeriodicAOBlochTransportInventory& live,
    const PeriodicAOBlochTransportCaps& caps) {
    controls(options,live);
    PeriodicAOBlochTransportPlan p;
    p.n_atoms = system.unit_cell.size(); p.n_shells = basis.nshells();
    p.n_columns = columns; p.n_kpoints = mesh.size(); p.source_index = source_index;
    p.mesh = mesh.mesh(); p.is_shift = mesh.is_shift(); p.time_reversal = time_reversal;
    if (!p.n_atoms || !p.n_shells) invalid("AO Bloch transport requires a nonempty atom/basis inventory");
    cap(p.n_atoms,caps.maximum_atoms,"AO Bloch transport atom cap");
    cap(p.n_shells,caps.maximum_shells,"AO Bloch transport shell cap");
    cap(columns,caps.maximum_columns,"AO Bloch transport column cap");
    mesh_action(p,system,op,mesh);
    U descriptor_bytes = add(sizeof(BasisSet),sizeof(PeriodicSystem));
    descriptor_bytes = add(descriptor_bytes,add(sizeof(SymmetryOp),sizeof(RegularKMesh)));
    descriptor_bytes = add(descriptor_bytes,add(basis.name().size(),1));
    descriptor_bytes = add(descriptor_bytes,mul(p.n_atoms,sizeof(Atom)));
    descriptor_bytes = add(descriptor_bytes,mul(p.n_shells,sizeof(libint2::Shell)+sizeof(int)+sizeof(std::size_t)));
    const auto& shells = basis.libint();
    if (shells.shell2bf().size() != p.n_shells)
        invalid("AO Bloch transport native shell offset metadata");
    for (U s = 0; s < p.n_shells; ++s) {
        const auto& sh = shells[s];
        const int atom = basis.shell_atom_index(s);
        if (atom < 0 || static_cast<U>(atom) >= p.n_atoms || sh.alpha.empty() || sh.contr.empty())
            invalid("AO Bloch transport invalid shell/atom/contraction metadata");
        if (shells.shell2bf()[s] != p.n_basis)
            invalid("AO Bloch transport native AO offset mismatch");
        p.basis_numeric_lanes = add(p.basis_numeric_lanes,add(3,add(sh.alpha.size(),sh.max_ln_coeff.size())));
        cap(p.basis_numeric_lanes,caps.maximum_basis_numeric_lanes,"AO Bloch transport basis numeric lane cap");
        p.n_contractions = add(p.n_contractions,sh.contr.size());
        cap(p.n_contractions,caps.maximum_contractions,"AO Bloch transport contraction cap");
        descriptor_bytes = add(descriptor_bytes,mul(sh.contr.size(),sizeof(libint2::Shell::Contraction)));
        for (const auto& c : sh.contr) {
            if (c.l < 0 || c.l > Lmax)
                invalid("AO Bloch transport unsupported angular momentum (native bound l<=6)");
            if (c.coeff.size() != sh.alpha.size())
                invalid("AO Bloch transport primitive/coefficient extent mismatch");
            p.basis_numeric_lanes = add(p.basis_numeric_lanes,c.coeff.size());
            cap(p.basis_numeric_lanes,caps.maximum_basis_numeric_lanes,"AO Bloch transport basis numeric lane cap");
            p.n_basis = add(p.n_basis,c.size());
            cap(p.n_basis,caps.maximum_basis_functions,"AO Bloch transport basis-function cap");
        }
    }
    if (!p.n_basis || p.n_basis != basis.nbasis()) invalid("AO Bloch transport native AO count mismatch");
    // The optional SpaceGroup remains owned by the borrowed System even though
    // the numerical leaf uses only the explicitly supplied single operation.
    if (system.symmetry) {
        const auto& g = *system.symmetry;
        descriptor_bytes = add(descriptor_bytes,mul(g.operations.size(),sizeof(SymmetryOp)));
        descriptor_bytes = add(descriptor_bytes,mul(g.equivalent_atoms.size(),sizeof(int)));
        descriptor_bytes = add(descriptor_bytes,
            add(add(g.international_symbol.size(),1),add(g.point_group.size(),1)));
    }
    p.retained_output_bytes = mul(16,mul(p.n_basis,columns));
    p.borrowed_coefficient_bytes = p.retained_output_bytes;
    p.borrowed_basis_numeric_bytes = mul(8,p.basis_numeric_lanes);
    p.borrowed_geometry_numeric_bytes = add(96,mul(24,p.n_atoms));
    p.borrowed_numerical_bytes = add(p.borrowed_coefficient_bytes,
        add(p.borrowed_basis_numeric_bytes,p.borrowed_geometry_numeric_bytes));
    p.mapping_workspace_bytes = add(mul(p.n_atoms,sizeof(AtomMap)),mul(p.n_shells,sizeof(ShellMap)));
    // The same buffers become retained output; no second mapping copy exists.
    p.retained_mapping_bytes = p.mapping_workspace_bytes;
    p.fixed_numerical_workspace_bytes = add(sizeof(Work),FixedScalarNumericalReservation);
    p.peak_owned_numerical_bytes = add(p.retained_output_bytes,
        add(p.mapping_workspace_bytes,p.fixed_numerical_workspace_bytes));
    // Logical payload/descriptor census, not malloc-capacity/RSS certification.
    // Full borrowed descriptor sizeofs conservatively include any inline
    // numerical fields also charged above. The fixed control allowance covers
    // integer/scalar control temporaries and standard control objects; the
    // caller separately supplies a positive backend/RSS margin.
    p.control_storage_bytes = add(live.other_live_control_bytes_per_replica,
        add(descriptor_bytes,8192+sizeof(PeriodicAOBlochTransportResult)+
            3*sizeof(std::vector<Z>)+2*sizeof(PeriodicAOBlochTransportPlan)+
            sizeof(PeriodicAOBlochTransportDiagnostics)+sizeof(options)+sizeof(live)+sizeof(caps)));
    // Shell search may compare a complete radial profile for each candidate,
    // and the used-shell test is cubic in shell count in the worst case.
    U work = add(65536,mul(256,mul(p.n_atoms,p.n_atoms)));
    work = add(work,mul(32,mul(mul(p.n_shells,p.n_shells),p.n_shells)));
    work = add(work,mul(64,mul(p.n_shells,p.basis_numeric_lanes)));
    work = add(work,mul(2000000,p.n_contractions));
    work = add(work,mul(128*Nc,mul(p.n_basis,columns)));
    p.work_units_upper_bound = work;
    p.per_replica_inventoried_bytes = add(p.peak_owned_numerical_bytes,
        add(p.borrowed_numerical_bytes,add(live.other_live_numerical_bytes_per_replica,
            add(p.control_storage_bytes,live.backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes = add(live.external_node_bytes,
        mul(live.numerical_replicas,p.per_replica_inventoried_bytes));
    cap(p.borrowed_numerical_bytes,caps.maximum_borrowed_numerical_bytes,"AO Bloch transport borrowed numerical byte cap");
    cap(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"AO Bloch transport owned numerical byte cap");
    cap(p.control_storage_bytes,caps.maximum_control_storage_bytes,"AO Bloch transport control storage cap");
    cap(p.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,"AO Bloch transport per-replica inventoried byte cap");
    cap(p.required_node_inventoried_bytes,caps.maximum_node_inventoried_bytes,"AO Bloch transport node inventoried byte cap");
    cap(p.work_units_upper_bound,caps.maximum_work_units,"AO Bloch transport work cap");
    if (p.retained_output_bytes > std::numeric_limits<std::size_t>::max() ||
        p.mapping_workspace_bytes > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("AO Bloch transport allocation address extent");
    return p;
}

PeriodicAOBlochTransportResult apply_periodic_ao_bloch_operation(
    const BasisSet& basis, const PeriodicSystem& system, const SymmetryOp& op,
    const RegularKMesh& mesh, U source_index, bool time_reversal,
    const Z* coefficients, U coefficient_count, U columns,
    const PeriodicAOBlochTransportOptions& options,
    const PeriodicAOBlochTransportInventory& live,
    const PeriodicAOBlochTransportCaps& caps) {
    const auto p = plan_periodic_ao_bloch_transport(basis,system,op,mesh,source_index,
                                                  time_reversal,columns,options,live,caps);
    if (coefficient_count != mul(p.n_basis,columns) ||
        (coefficient_count && (!coefficients || reinterpret_cast<std::uintptr_t>(coefficients)%alignof(Z))))
        invalid("AO Bloch transport requires an aligned exact coefficient extent");
    Work w;
    PeriodicAOBlochTransportResult result;
    result.plan_ = p;
    geometry(w,system,op,options,result.diagnostics_);
    result.atoms_.resize(p.n_atoms);
    result.shells_.resize(p.n_shells);
    auto& atoms = result.atoms_;
    auto& shells = result.shells_;
    atom_mapping(w,system,options,atoms,result.diagnostics_);
    shell_mapping(basis,system,atoms,options,shells,result.diagnostics_);
    for (U i = 0; i < coefficient_count; ++i) finite_complex(coefficients[i]);
    result.coefficients_.resize(coefficient_count);
    const auto& raw = basis.libint();
    for (U s = 0; s < p.n_shells; ++s) {
        U source_ao = shells[s].source_ao, destination_ao = shells[s].destination_ao;
        const Z phase = atom_phase(atoms[basis.shell_atom_index(s)],mesh,p);
        for (const auto& contraction : raw[s].contr) {
            rotation_block(w,contraction.l,contraction.pure,options,result.diagnostics_);
            const U n = contraction.size();
            for (U a = 0; a < n; ++a)
                for (U column = 0; column < columns; ++column) {
                    double real = 0, imag = 0, real_c = 0, imag_c = 0;
                    for (U b = 0; b < n; ++b) {
                        Z x = coefficients[(source_ao+b)*columns+column];
                        if (time_reversal) x = std::conj(x);
                        const double d = w.rotation[a*Nc+b];
                        sum(real,real_c,d*x.real()); sum(imag,imag_c,d*x.imag());
                    }
                    const Z value = phase*Z(finite(real+real_c),finite(imag+imag_c));
                    finite_complex(value);
                    result.coefficients_[(destination_ao+a)*columns+column] = value;
                    result.diagnostics_.maximum_output_magnitude =
                        std::max(result.diagnostics_.maximum_output_magnitude,finite(std::abs(value)));
                }
            source_ao += n; destination_ao += n;
            ++result.diagnostics_.rotated_contractions;
        }
    }
    return result;
}

const Z* PeriodicAOBlochTransportResult::data() const {
    if (plan_.n_basis == 0 || coefficients_.size() != mul(plan_.n_basis,plan_.n_columns) ||
        atoms_.size() != plan_.n_atoms || shells_.size() != plan_.n_shells)
        throw std::logic_error("AO Bloch transport result is moved from or has invalid storage");
    return coefficients_.data();
}
Z PeriodicAOBlochTransportResult::coefficient(U ao, U column) const {
    const auto* values = data();
    if (ao >= plan_.n_basis || column >= plan_.n_columns)
        throw std::out_of_range("AO Bloch transport coefficient index");
    return values[ao*plan_.n_columns+column];
}
U PeriodicAOBlochTransportResult::atom_destination(U atom) const {
    (void)data();
    if (atom >= atoms_.size()) throw std::out_of_range("AO Bloch transport atom index");
    return atoms_[atom].destination;
}
std::array<I,3> PeriodicAOBlochTransportResult::atom_lattice_shift(U atom) const {
    (void)atom_destination(atom);
    return atoms_[atom].ell;
}
U PeriodicAOBlochTransportResult::shell_destination(U shell) const {
    (void)data();
    if (shell >= shells_.size()) throw std::out_of_range("AO Bloch transport shell index");
    return shells_[shell].destination;
}

} // namespace vibeqc
