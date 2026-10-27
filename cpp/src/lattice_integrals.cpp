#include "vibeqc/lattice_integrals.hpp"

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/cart_to_sph_data.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/engine.h>
#include <libint2/solidharmonics.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vibeqc {

namespace {

// Clone a shell vector and translate every origin by dr (bohr). The
// contraction coefficients / primitive exponents are unchanged — only
// the origin moves, which is what a lattice translation does.
std::vector<libint2::Shell> shift_shells(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr) {
    std::vector<libint2::Shell> out(shells.begin(), shells.end());
    for (auto& s : out) {
        s.O[0] += dr[0];
        s.O[1] += dr[1];
        s.O[2] += dr[2];
    }
    return out;
}

// The lattice cells a translation-invariant 1e sum needs: the ball of
// radius ``cutoff`` padded by the largest intra-cell shell offset. The
// per-pair filter inside the builder then restores the exact term set
//   {(p, q, g) : |O_p - O_q - g| <= cutoff}
// which is what translation invariance requires; see
// lattice_pair_cells.hpp for why bounding |g| instead is a wrong answer
// and not a tolerance.
//
// Deliberately a plain SPHERE and not the tighter union of per-pair balls.
// A sphere in the direct lattice is closed under the crystal's point group,
// which ``identify_lattice_orbits`` / ``identify_atom_pair_orbits`` require
// of any cell list they partition; the reachability-pruned union is not
// (a rotation maps an atom-pair offset onto an image of another offset, so
// the cell INDEX picks up the difference of two lattice translations even
// though the physical term set maps onto itself). The cells the padding
// adds and no pair reaches cost one distance test per shell row and stay
// exactly zero, so the price is blocks of zeros, not integrals.
//
// The zero cell stays first, as ``direct_lattice_cells`` guarantees and the
// SCF drivers rely on.
std::vector<LatticeCell> pair_cells_for_1e(const libint2::BasisSet& shells,
                                           const PeriodicSystem& system,
                                           double cutoff) {
    return pair_complete_cells(system, cutoff, shells, shells);
}

// Apply the same physical AO-pair support to a dense reciprocal/grid block
// as to its analytical real-space partner.
void filter_pair_block(Eigen::MatrixXd& block,
                       const libint2::BasisSet& shells,
                       const Eigen::Vector3d& g, double cutoff) {
    const auto offsets = shells.shell2bf();
    for (std::size_t p = 0; p < shells.size(); ++p) {
        for (std::size_t q = 0; q < shells.size(); ++q) {
            if (!pair_in_range(shells[p], shells[q], g, cutoff)) {
                block.block(offsets[p], offsets[q], shells[p].size(),
                            shells[q].size()).setZero();
            }
        }
    }
}

// Core driver: for every lattice cell g, compute ⟨ χ_μ(0) | Op | χ_ν(g) ⟩.
// The Op is a libint 1-body operator (overlap, kinetic, or nuclear). For
// nuclear attraction, caller supplies the already-lattice-summed point
// charge list via ``nuclei``.
//
// ``pair_cutoff`` bounds the PHYSICAL separation |O_μ - O_ν - g| of each
// term. Pass <= 0 to compute every pair of every supplied cell (the
// caller-controlled explicit-cell contract).
// Forward-declared so the classic entry point can delegate.
LatticeMatrixSet compute_1e_lattice_matrix_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    libint2::Operator op,
    const std::vector<std::pair<double, std::array<double, 3>>>* nuclei,
    double pair_cutoff = 0.0,
    const PairNuclearImageSelector* source_selector = nullptr,
    const std::vector<double>* source_weights = nullptr);
LatticeMatrixSet compute_1e_lattice_matrix_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    libint2::Operator op,
    const std::vector<std::pair<double, std::array<double, 3>>>* nuclei,
    double pair_cutoff,
    const PairNuclearImageSelector* source_selector,
    const std::vector<double>* source_weights) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());

    libint2::Engine prototype(op, shells_ref.max_nprim(), shells_ref.max_l(), 0);
    if (nuclei != nullptr) {
        using nuc_params =
            libint2::operator_traits<libint2::Operator::nuclear>::oper_params_type;
        prototype.set_params(nuc_params{*nuclei});
    }
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();

    LatticeMatrixSet set;
    set.nbf = nbf;
    set.cells = cells;
    set.blocks.assign(cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));

    // Parallelise over (cell, bra shell) rather than over cells alone. With
    // the pair filter on, near cells carry every pair while far cells carry
    // a handful, so a per-cell schedule leaves most threads idle behind the
    // few dense ones; the collapsed index space is ~n_shells× finer and the
    // rows of a far cell that reach nothing cost only their distance tests.
    // Every iteration writes a disjoint (block, row-range) slice, so the
    // result does not depend on the schedule: unlike the accumulating
    // aux-ERI kernels, these builders stay bit-reproducible run to run.
    const std::int64_t n_cells = static_cast<std::int64_t>(cells.size());
    const std::int64_t n_shells = static_cast<std::int64_t>(shells_ref.size());

    #pragma omp parallel
    {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        // One shifted-shell buffer per thread, re-pointed (not reallocated)
        // when a thread crosses into a new cell. The collapsed space is
        // cell-major and handed out in chunks, so a thread stays inside one
        // cell for long runs.
        std::vector<libint2::Shell> shells_g(shells_ref.begin(),
                                             shells_ref.end());
        std::int64_t cached_cell = -1;
        std::optional<PairNuclearImageCache> sources;
        if (source_selector) sources.emplace(*source_selector, source_weights);

        #pragma omp for collapse(2) schedule(dynamic, 16)
        for (std::int64_t c = 0; c < n_cells; ++c) {
            for (std::int64_t s1i = 0; s1i < n_shells; ++s1i) {
                if (c != cached_cell) {
                    cached_cell = c;
                    const Eigen::Vector3d& g =
                        cells[static_cast<std::size_t>(c)].r_cart;
                    for (std::size_t s = 0; s < shells_g.size(); ++s) {
                        shells_g[s].O[0] = shells_ref[s].O[0] + g[0];
                        shells_g[s].O[1] = shells_ref[s].O[1] + g[1];
                        shells_g[s].O[2] = shells_ref[s].O[2] + g[2];
                    }
                }
                const auto s1 = static_cast<std::size_t>(s1i);
                const auto bf1 = shell2bf[s1];
                const auto n1 = shells_ref[s1].size();
                Eigen::MatrixXd& block =
                    set.blocks[static_cast<std::size_t>(c)];

                for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                    if (pair_cutoff > 0.0 &&
                        !pair_in_range(shells_ref[s1], shells_g[s2],
                                       pair_cutoff)) {
                        continue;
                    }
                    const auto bf2 = shell2bf[s2];
                    const auto n2 = shells_g[s2].size();

                    if (sources) {
                        const auto& selected = sources->get(shells_ref[s1], shells_g[s2]);
                        if (selected.charges.empty()) continue;
                        engine.set_params(selected.charges);
                    }
                    engine.compute(shells_ref[s1], shells_g[s2]);
                    const double* tile = buf[0];
                    if (!tile) continue;

                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            block(bf1 + i, bf2 + j) = tile[i * n2 + j];
                        }
                    }
                }
            }
        }
    }
    return set;
}

LatticeMatrixSet compute_1e_lattice_matrix(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    libint2::Operator op,
    const std::vector<std::pair<double, std::array<double, 3>>>* nuclei) {
    // Pair-complete enumeration (see lattice_pair_cells.hpp): what decides
    // whether a term matters is |O_μ - O_ν - g|, so the cell list is padded
    // by the largest intra-cell offset and each pair is then filtered on
    // its true separation. Bounding |g| alone is not translation
    // invariant -- on LiH rocksalt / def2-svp at the default 15-bohr cutoff
    // it moved the Gamma overlap by 7.2e-01 when the same crystal was
    // re-described with the H one lattice vector away.
    if (!opts.pair_complete_1e) {
        // #429 stage 2: the switch is off by default; the plain |g| ball and
        // the unfiltered pair set are the pre-2026-09 term set, bit for bit.
        const auto plain = direct_lattice_cells(system, opts.cutoff_bohr);
        return compute_1e_lattice_matrix_explicit(
            basis, system, plain, op, nuclei, 0.0);
    }
    const auto cells =
        pair_cells_for_1e(basis.libint(), system, opts.cutoff_bohr);
    std::optional<PairNuclearImageSelector> sources;
    if (op == libint2::Operator::nuclear)
        sources.emplace(system, opts.nuclear_cutoff_bohr);
    return compute_1e_lattice_matrix_explicit(
        basis, system, cells, op, nuclei, opts.cutoff_bohr,
        sources ? &*sources : nullptr);
}

// Build the lattice-summed point-charge list for nuclear attraction: every
// atom of the unit cell, replicated over every lattice cell within
// opts.nuclear_cutoff_bohr.
std::vector<std::pair<double, std::array<double, 3>>>
build_periodic_nuclear_charges(const PeriodicSystem& system,
                               const LatticeSumOptions& opts) {
    const auto cells =
        direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.reserve(cells.size() * system.unit_cell.size());
    for (const auto& c : cells) {
        for (const auto& a : system.unit_cell) {
            std::array<double, 3> r = {
                a.xyz[0] + c.r_cart[0],
                a.xyz[1] + c.r_cart[1],
                a.xyz[2] + c.r_cart[2],
            };
            q.emplace_back(static_cast<double>(a.Z), r);
        }
    }
    return q;
}

}  // namespace

std::vector<LatticeCell> pair_complete_lattice_cells(
    const BasisSet& basis,
    const PeriodicSystem& system,
    double cutoff_bohr) {
    return pair_cells_for_1e(basis.libint(), system, cutoff_bohr);
}

std::vector<LatticeCell> physical_eri_lattice_cells(
    const BasisSet& basis, const PeriodicSystem& system,
    double pair_cutoff_bohr, double interaction_cutoff_bohr) {
    LatticeSumOptions opts;
    opts.cutoff_bohr = pair_cutoff_bohr;
    opts.eri_interaction_cutoff_bohr = interaction_cutoff_bohr;
    return pair_complete_eri_cells(system, opts, basis.libint());
}

std::vector<std::vector<std::array<int, 4>>> physical_nuclear_source_indices(
    const PeriodicSystem& system, const Eigen::MatrixXd& midpoints,
    double cutoff_bohr) {
    if (midpoints.cols() != 3 || !midpoints.allFinite())
        throw std::invalid_argument("Nuclear support midpoints must be finite (n,3)");
    PairNuclearImageSelector selector(system, cutoff_bohr);
    constexpr std::size_t term_limit = 2000000;
    // Bound stored rows and selected labels, not a Cartesian box containing
    // each sphere. Enforce the remaining budget inside the shared traversal.
    if (static_cast<std::size_t>(midpoints.rows()) > term_limit)
        throw std::length_error("Nuclear support record exceeds bounded term budget");
    std::size_t terms = static_cast<std::size_t>(midpoints.rows());
    std::vector<std::vector<std::array<int, 4>>> result;
    for (Eigen::Index row = 0; row < midpoints.rows(); ++row) {
        std::vector<std::array<int, 4>> labels;
        selector.select(midpoints.row(row).transpose(), nullptr, &labels,
                        term_limit - terms, true);
        terms += labels.size();
        result.push_back(std::move(labels));
    }
    return result;
}

LatticeMatrixSet compute_overlap_lattice(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const LatticeSumOptions& opts) {
    return compute_1e_lattice_matrix(
        basis, system, opts, libint2::Operator::overlap, nullptr);
}

LatticeMatrixSet compute_kinetic_lattice(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const LatticeSumOptions& opts) {
    return compute_1e_lattice_matrix(
        basis, system, opts, libint2::Operator::kinetic, nullptr);
}

LatticeMatrixSet compute_nuclear_lattice(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const LatticeSumOptions& opts) {
    const auto nuclei = build_periodic_nuclear_charges(system, opts);
    return compute_1e_lattice_matrix(
        basis, system, opts, libint2::Operator::nuclear, &nuclei);
}

LatticeMatrixSet compute_nuclear_lattice_with_charges(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<double>& effective_charges) {
    if (opts.pair_complete_1e) {
        const PairNuclearImageSelector sources(system, opts.nuclear_cutoff_bohr);
        const auto cells = pair_cells_for_1e(basis.libint(), system, opts.cutoff_bohr);
        return compute_1e_lattice_matrix_explicit(
            basis, system, cells, libint2::Operator::nuclear, nullptr,
            opts.cutoff_bohr, &sources, &effective_charges);
    }
    const auto cells =
        direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.reserve(cells.size() * system.unit_cell.size());
    for (const auto& c : cells) {
        for (std::size_t a = 0; a < system.unit_cell.size(); ++a) {
            const auto& atom = system.unit_cell[a];
            std::array<double, 3> r = {
                atom.xyz[0] + c.r_cart[0],
                atom.xyz[1] + c.r_cart[1],
                atom.xyz[2] + c.r_cart[2],
            };
            double qval = a < effective_charges.size()
                ? effective_charges[a]
                : static_cast<double>(atom.Z);
            q.emplace_back(qval, r);
        }
    }
    return compute_1e_lattice_matrix(
        basis, system, opts, libint2::Operator::nuclear, &q);
}

namespace {

// ---------------------------------------------------------------------------
// Per-product nuclear source radius for the erfc short-range sum (#196)
// ---------------------------------------------------------------------------
//
// The pair-complete erfc sum takes every nuclear image within
// ``nuclear_cutoff_bohr`` of each AO-pair midpoint M. Its integrand decays
// with the erfc range of the PRODUCT, not with that radius: for a Gaussian
// product of exponent p centred at P the attraction of one point charge at
// distance R is (pi/p)^{3/2} [erf(sqrt(p) R) - erf(mu R)] / R with
// mu = omega sqrt(p / (p + omega^2)), which is at most (pi/p)^{3/2}
// erfc(mu R) / R. This is the nuclear (point-charge) limit of the
// short-range three-centre analysis of Ye & Berkelbach, J. Chem. Phys. 154,
// 131104 (2021), doi:10.1063/5.0046617, SI Sec. S4 Eqs. (S8)-(S13), with
// the auxiliary exponent taken to infinity (eta_1 = sqrt(p), eta_2 = mu).
//
// Their Eq. (S18) turns the sum over images beyond R_c into an integral
// with density 4 pi R^2 / Omega under asymptotic assumptions. Here the same
// step is made a bound. erfc(mu r)/r is subharmonic away from r = 0: it is
// 1/r minus erf(mu r)/r, the potential of a positive Gaussian charge, so its
// Laplacian is +4 pi times that charge density. Its value at an image C is
// therefore at most its average over the ball of radius r0 around C, where
// r0 is half the shortest lattice vector, so the balls of one atom's images
// are disjoint. Every ball of an image beyond R_c lies beyond x = R_c - r0,
// and with erfc(z) <= exp(-z^2)/(sqrt(pi) z)
//
//   sum_{|C-P|>R_c} erfc(mu|C-P|)/|C-P|
//       <= (4 pi / V_ball) int_x^inf s erfc(mu s) ds
//       <= (2 pi / (V_ball mu^2)) erfc(mu x),   V_ball = 4 pi r0^3 / 3.
//
// Angular factors are bounded by |u|^l exp(-a u^2/2) <= (l/(e a))^{l/2},
// which costs half of each exponent (p -> p/2 and the pair prefactor
// exp(-ab/p |AB|^2) -> its square root); the solid-harmonic combination by
// the largest absolute row sum of its Cartesian coefficients. P lies within
// |P - M| of the selection centre M. The returned radius never exceeds the
// caller's cutoff, so the selection is the subset of the unscreened one
// whose dropped attraction is bounded, element by element, by ``tol``
// (Hartree).
struct ShellSourceBound {
    int l = 0;
    double tau = 1.0;  // bound on |angular part| / |r|^l
    std::vector<double> alpha;
    std::vector<double> coeff;  // max |coefficient| per primitive
};

ShellSourceBound make_shell_source_bound(const libint2::Shell& shell) {
    ShellSourceBound out;
    if (shell.contr.empty())
        throw std::invalid_argument("nuclear source radius: shell without contraction");
    out.l = shell.contr[0].l;
    for (const auto& c : shell.contr) {
        if (c.l != out.l)
            throw std::invalid_argument("nuclear source radius: mixed-l shells are not supported");
    }
    if (shell.contr[0].pure && out.l > 0) {
        using libint2::solidharmonics::SolidHarmonicsCoefficients;
        const auto& coefs = SolidHarmonicsCoefficients<double>::instance(
            static_cast<unsigned int>(out.l));
        double worst = 0.0;
        for (int r = 0; r < 2 * out.l + 1; ++r) {
            const double* values = coefs.row_values(static_cast<std::size_t>(r));
            double sum = 0.0;
            for (unsigned int k = 0; k < coefs.nnz(static_cast<std::size_t>(r)); ++k)
                sum += std::abs(values[k]);
            worst = std::max(worst, sum);
        }
        out.tau = worst;
    }
    out.alpha.assign(shell.alpha.begin(), shell.alpha.end());
    out.coeff.assign(shell.alpha.size(), 0.0);
    for (const auto& c : shell.contr)
        for (std::size_t k = 0; k < c.coeff.size(); ++k)
            out.coeff[k] = std::max(out.coeff[k], std::abs(c.coeff[k]));
    return out;
}

// max_u u^l exp(-a u^2 / 2) = (l / (e a))^{l/2}
double angular_peak(int l, double a) {
    if (l == 0) return 1.0;
    return std::pow(static_cast<double>(l) / (std::exp(1.0) * a), 0.5 * l);
}

// Length of the shortest nonzero lattice vector, exactly: a vector n.L of
// length <= l has |n_i| <= l ||(L^-1)_i||, so enumerating that box with l the
// shortest basis vector finds it.
double shortest_lattice_vector(const Eigen::Matrix3d& lattice) {
    const Eigen::Matrix3d inverse = lattice.inverse();
    double best = std::min({lattice.col(0).norm(), lattice.col(1).norm(), lattice.col(2).norm()});
    if (!inverse.allFinite() || !std::isfinite(best) || !(best > 0.0)) return 0.0;
    std::array<int, 3> reach;
    for (int axis = 0; axis < 3; ++axis) {
        const double r = std::ceil(best * inverse.row(axis).norm() + 1e-9);
        if (!(r < 64.0)) {
            // Too skewed to enumerate: the Gram-Schmidt lengths bound the
            // shortest vector from below, which keeps the balls disjoint.
            const Eigen::HouseholderQR<Eigen::Matrix3d> qr(lattice);
            const Eigen::Matrix3d R = qr.matrixQR().triangularView<Eigen::Upper>();
            return std::min({std::abs(R(0, 0)), std::abs(R(1, 1)), std::abs(R(2, 2))});
        }
        reach[axis] = static_cast<int>(r);
    }
    for (int i = -reach[0]; i <= reach[0]; ++i)
    for (int j = -reach[1]; j <= reach[1]; ++j)
    for (int k = -reach[2]; k <= reach[2]; ++k) {
        if (i == 0 && j == 0 && k == 0) continue;
        best = std::min(best, (lattice * Eigen::Vector3d(i, j, k)).norm());
    }
    return best;
}

double nuclear_source_radius(const ShellSourceBound& sa, const Eigen::Vector3d& A,
                             const ShellSourceBound& sb, const Eigen::Vector3d& B,
                             double omega, double z_abs, double packing_radius,
                             double tol, double cutoff) {
    constexpr double pi = 3.14159265358979323846;
    struct Term { double weight, mu, reach; };
    std::vector<Term> terms;
    terms.reserve(sa.alpha.size() * sb.alpha.size());
    const Eigen::Vector3d M = 0.5 * (A + B);
    const double ab2 = (A - B).squaredNorm();
    const bool split = sa.l > 0 || sb.l > 0;
    const double ball = 4.0 * pi * packing_radius * packing_radius * packing_radius / 3.0;
    for (std::size_t i = 0; i < sa.alpha.size(); ++i) {
        for (std::size_t j = 0; j < sb.alpha.size(); ++j) {
            const double a = sa.alpha[i], b = sb.alpha[j];
            const double c = sa.coeff[i] * sb.coeff[j];
            if (c == 0.0) continue;
            const double p = a + b;
            const double reduced = a * b / p;
            const double pe = split ? 0.5 * p : p;
            const double log_k = split ? -0.5 * reduced * ab2 : -reduced * ab2;
            const double mu = omega * std::sqrt(pe / (pe + omega * omega));
            const double weight = sa.tau * sb.tau * c *
                                  angular_peak(sa.l, a) * angular_peak(sb.l, b) *
                                  std::exp(log_k) * std::pow(pi / pe, 1.5) *
                                  z_abs * 2.0 * pi / (ball * mu * mu);
            const Eigen::Vector3d P = (a * A + b * B) / p;
            terms.push_back({weight, mu, (P - M).norm() + packing_radius});
        }
    }
    auto bound = [&](double radius) {
        double total = 0.0;
        for (const auto& t : terms) {
            const double x = radius - t.reach;
            if (!(x > 0.0)) return std::numeric_limits<double>::infinity();
            total += t.weight * std::erfc(t.mu * x);
        }
        return total;
    };
    if (terms.empty()) return 0.0;
    if (!(bound(cutoff) <= tol)) return cutoff;
    double lo = 0.0, hi = cutoff;
    for (int it = 0; it < 32; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (bound(mid) <= tol) hi = mid; else lo = mid;
    }
    return hi;
}

}  // namespace

LatticeMatrixSet compute_nuclear_erfc_lattice(const BasisSet& basis,
                                              const PeriodicSystem& system,
                                              double omega,
                                              const LatticeSumOptions& opts,
                                              double source_tol) {
    ensure_libint_initialized();
    if (!std::isfinite(source_tol) || source_tol < 0.0)
        throw std::invalid_argument("compute_nuclear_erfc_lattice: source_tol must be finite and nonnegative");
    const bool screen_sources = source_tol > 0.0;
    if (screen_sources && (!opts.pair_complete_1e || system.dim != 3))
        throw std::invalid_argument(
            "compute_nuclear_erfc_lattice: source_tol needs pair_complete_1e on a 3-D lattice");

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    const auto nuclei = opts.pair_complete_1e
        ? std::vector<std::pair<double, std::array<double, 3>>>{}
        : build_periodic_nuclear_charges(system, opts);
    const PairNuclearImageSelector selector(system, opts.nuclear_cutoff_bohr);

    // libint's erfc_nuclear takes a tuple (ω, point-charge list). ω is the
    // erfc attenuation parameter: the kernel is erfc(ω · r_C) / r_C, which
    // reduces to 1/r_C at ω=0 and vanishes as ω → ∞.
    libint2::Engine prototype(libint2::Operator::erfc_nuclear,
                              shells_ref.max_nprim(),
                              shells_ref.max_l(), 0);
    using erfc_params =
        libint2::operator_traits<libint2::Operator::erfc_nuclear>::oper_params_type;
    if (!opts.pair_complete_1e) prototype.set_params(erfc_params{omega, nuclei});
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    std::vector<PairNuclearImageCache> source_caches;
    for (int tid = 0; tid < omp_max_threads(); ++tid) source_caches.emplace_back(selector);

    // Per-product source radii (``source_tol`` > 0); see nuclear_source_radius.
    std::vector<ShellSourceBound> source_bounds;
    double z_abs = 0.0, packing_radius = 0.0;
    if (screen_sources) {
        for (const auto& shell : shells_ref) source_bounds.push_back(make_shell_source_bound(shell));
        for (const auto& atom : system.unit_cell) z_abs += std::abs(static_cast<double>(atom.Z));
        packing_radius = 0.5 * shortest_lattice_vector(system.lattice);
        if (!(packing_radius > 0.0) || !std::isfinite(packing_radius))
            throw std::invalid_argument("compute_nuclear_erfc_lattice: degenerate lattice for source_tol");
    }
    struct ScreenedSources {
        Eigen::Vector3d center;
        double radius;
        PairNuclearImageSet images;
    };
    std::vector<std::vector<ScreenedSources>> screened_caches(
        static_cast<std::size_t>(omp_max_threads()));

    LatticeMatrixSet set;
    set.nbf = nbf;
    // Both Ewald halves and their gradients use this same AO-pair support.
    set.cells = opts.pair_complete_1e
        ? pair_cells_for_1e(shells_ref, system, opts.cutoff_bohr)
        : direct_lattice_cells(system, opts.cutoff_bohr);
    set.blocks.assign(set.cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));

    const int n_cells = static_cast<int>(set.cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();

        const Eigen::Vector3d& g = set.cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);

        Eigen::MatrixXd block = Eigen::MatrixXd::Zero(nbf, nbf);
        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                if (opts.pair_complete_1e &&
                    !pair_in_range(shells_ref[s1], shells_g[s2], opts.cutoff_bohr))
                    continue;
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();

                if (opts.pair_complete_1e && screen_sources) {
                    const Eigen::Vector3d A(shells_ref[s1].O[0], shells_ref[s1].O[1], shells_ref[s1].O[2]);
                    const Eigen::Vector3d B(shells_g[s2].O[0], shells_g[s2].O[1], shells_g[s2].O[2]);
                    const double radius = nuclear_source_radius(
                        source_bounds[s1], A, source_bounds[s2], B, omega, z_abs,
                        packing_radius, source_tol, opts.nuclear_cutoff_bohr);
                    const Eigen::Vector3d center = 0.5 * (A + B);
                    auto& cache = screened_caches[static_cast<std::size_t>(omp_thread_index())];
                    const PairNuclearImageSet* selected = nullptr;
                    for (const auto& entry : cache) {
                        if (entry.radius == radius &&
                            (entry.center.array() == center.array()).all()) {
                            selected = &entry.images;
                            break;
                        }
                    }
                    if (!selected) {
                        if (cache.size() >= 16) cache.clear();
                        cache.push_back({center, radius, selector.select_within(center, radius)});
                        selected = &cache.back().images;
                    }
                    if (selected->charges.empty()) continue;
                    engine.set_params(erfc_params{omega, selected->charges});
                } else if (opts.pair_complete_1e) {
                    const auto& selected = source_caches[omp_thread_index()].get(
                        shells_ref[s1], shells_g[s2]);
                    if (selected.charges.empty()) continue;
                    engine.set_params(erfc_params{omega, selected.charges});
                }
                engine.compute(shells_ref[s1], shells_g[s2]);
                const double* tile = buf[0];
                if (!tile) continue;

                for (std::size_t i = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        block(bf1 + i, bf2 + j) = tile[i * n2 + j];
                    }
                }
            }
        }
        set.blocks[c] = std::move(block);
    }
    return set;
}

// ---------------------------------------------------------------------------
// Phase 12e-c: full Ewald nuclear-attraction lattice sum via grid integration
// ---------------------------------------------------------------------------

namespace {

// Translate the actual AO shells, including custom contractions and ECP
// valence bases. A display name need not identify a reloadable basis file.
BasisSet shifted_basis_for_cell(const BasisSet& ref,
                                const PeriodicSystem& system,
                                const Eigen::Vector3d& dr) {
    std::vector<Atom> shifted;
    shifted.reserve(system.unit_cell.size());
    for (const auto& a : system.unit_cell) {
        shifted.push_back(Atom{
            a.Z,
            {a.xyz[0] + dr[0], a.xyz[1] + dr[1], a.xyz[2] + dr[2]},
        });
    }
    Molecule mol(std::move(shifted), system.charge, system.multiplicity);
    auto shells = ref.shells();
    for (auto& shell : shells) {
        for (int axis = 0; axis < 3; ++axis) shell.origin[axis] += dr[axis];
    }
    return BasisSet(mol, shells, ref.name(), true);
}

}  // namespace

LatticeMatrixSet compute_nuclear_lattice_ewald(const BasisSet& basis,
                                               const PeriodicSystem& system,
                                               const Grid& grid,
                                               const LatticeSumOptions& opts,
                                               const EwaldOptions& ewald_opts) {
    if (system.dim != 3) {
        throw std::invalid_argument(
            "compute_nuclear_lattice_ewald: 3D Ewald requires dim == 3. "
            "Use compute_nuclear_lattice (DIRECT_TRUNCATED) for 1D / 2D.");
    }

    // Resolve α (same auto-rule the Ewald engine uses).
    const double alpha = (ewald_opts.alpha > 0.0)
        ? ewald_opts.alpha
        : std::sqrt(-std::log(ewald_opts.tolerance))
              / ewald_opts.real_cutoff_bohr;

    // -----------------------------------------------------------------
    //   V(g) = V_short(g) + V_long(g)
    //
    //   V_short(g): analytical erfc-screened nuclear-attraction via
    //               libint (Phase 12e-b's compute_nuclear_erfc_lattice).
    //               Captures the sharp 1/r spike at each nucleus with
    //               full libint accuracy.
    //   V_long(g):  grid integral of the smooth, bounded long-range
    //               potential ~erf(α r)/r.
    // -----------------------------------------------------------------

    // Short-range component — analytical. compute_nuclear_erfc_lattice
    // already emits the correctly-signed electronic potential
    // (−Z erfc/r integrated against the AOs), so the sign is right.
    LatticeMatrixSet V_short =
        compute_nuclear_erfc_lattice(basis, system, alpha, opts);

    // Long-range component — grid quadrature. Evaluate the *smooth*
    // long-range Ewald potential at every grid point; the short-range
    // part is omitted (handled above analytically).
    const Eigen::VectorXd v_long_r =
        ewald_nuclear_potential(system, grid.points, ewald_opts,
                                /*include_short_range=*/false);

    const Eigen::MatrixXd chi_ref = evaluate_ao(basis, grid.points);
    const int nbf = static_cast<int>(chi_ref.cols());
    const Eigen::VectorXd wv = grid.weights.array() * v_long_r.array();
    const Eigen::MatrixXd chi_wv = chi_ref.array().colwise() * wv.array();

    LatticeMatrixSet set;
    set.nbf = nbf;
    // Adopt the short-range part's cell list outright rather than
    // re-deriving it. Both halves are the same μν lattice sum, so they must
    // share the pair-complete enumeration (see lattice_pair_cells.hpp);
    // re-deriving it was how the two could silently disagree.
    set.cells = V_short.cells;
    const int n_cells = static_cast<int>(set.cells.size());
    set.blocks.assign(set.cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const BasisSet basis_g =
            shifted_basis_for_cell(basis, system, set.cells[c].r_cart);
        const Eigen::MatrixXd chi_g = evaluate_ao(basis_g, grid.points);
        //   V_long_μν(g) = Σ_r χ_μ(r) · w(r) · v_long(r) · χ_ν(r − g)
        const Eigen::MatrixXd V_long_block = chi_wv.transpose() * chi_g;
        set.blocks[c] = V_short.blocks[c] + V_long_block;
        if (opts.pair_complete_1e) {
            filter_pair_block(set.blocks[c], basis.libint(),
                              set.cells[c].r_cart, opts.cutoff_bohr);
        }
    }

    return set;
}

// ---------------------------------------------------------------------------
// Lattice-summed SAP guess potential (Ewald split: analytical short-range +
// smooth long-range grid quadrature)
// ---------------------------------------------------------------------------
//
// The molecular SAP potential, as built by build_v_sap_matrix /
// compute_sap_potential_molecular, is V_SAP(r) = +Σ_A Σ_i c_i^A erf(ω_i^A r_A)/r_A
// (attractive: the oscillating fit coefficients give Σ_i c_i √α_i < 0, so
// V_SAP(0) = (2/√π) Σ_i c_i √α_i < 0; libint's erf_nuclear applies the −q
// nuclear sign, and the molecular code passes q = −c_i). Split each erf term
// with a single moderate Ewald parameter η via
//   erf(ω r)/r = erf(η r)/r + erfc(η r)/r − erfc(ω r)/r, which gives
//
//   V_SAP = −Σ_A Σ_i c_i erfc(ω_i r)/r           (short, analytical)
//          + Σ_A Q_A erfc(η r)/r                  (short, analytical)
//          + Σ_A Q_A erf(η r)/r ,   Q_A = Σ_i c_i^A (long, smooth grid).
//
// The two erfc pieces are short-ranged real-space lattice sums computed
// analytically with libint's erfc_nuclear engine — this resolves the sharp
// near-nucleus structure that pure grid quadrature gets wrong (the
// reciprocal-only first cut produced an asymmetric, wrong-signed V_SAP). The
// remaining erf(η r)/r piece is smooth (moderate η) and grid-quadratured,
// reusing compute_nuclear_lattice_ewald's long-range path with the net
// per-atom effective charges Q_A. The decomposition is exact (the three pieces
// reconstruct +Σ c_i erf(ω_i)/r) and the moderate η keeps both the real-space
// erfc sums and the reciprocal erf sum cheap for any element. The G=0 /
// jellium background is a constant shift, which leaves the guess density
// invariant.

namespace {

// Accumulate the analytical erfc-screened lattice matrix into ``set``:
//   set.blocks[g]_μν += −Σ_C q_C ⟨χ_μ(0)| erfc(ω |r − R_C|)/|r − R_C| |χ_ν(g)⟩.
// libint's erfc_nuclear applies the nuclear-attraction sign (charge q →
// −q·erfc/r), the same convention build_periodic_nuclear_charges relies on.
// ``charges`` is the already-image-expanded (q, position) list. No-op if empty.
void accumulate_erfc_lattice(
    LatticeMatrixSet& set,
    const libint2::BasisSet& shells_ref,
    double omega,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges,
    double pair_cutoff,
    const PairNuclearImageSelector* source_selector,
    const std::vector<double>& source_weights) {
    if (!source_selector && charges.empty()) return;
    libint2::Engine prototype(libint2::Operator::erfc_nuclear,
                              shells_ref.max_nprim(), shells_ref.max_l(), 0);
    using erfc_params =
        libint2::operator_traits<libint2::Operator::erfc_nuclear>::oper_params_type;
    if (!source_selector) prototype.set_params(erfc_params{omega, charges});
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    std::vector<PairNuclearImageCache> source_caches;
    if (source_selector) {
        for (int tid = 0; tid < omp_max_threads(); ++tid)
            source_caches.emplace_back(*source_selector, &source_weights);
    }
    const int n_cells = static_cast<int>(set.cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto shells_g = shift_shells(shells_ref, set.cells[c].r_cart);
        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                if (pair_cutoff > 0.0 &&
                    !pair_in_range(shells_ref[s1], shells_g[s2], pair_cutoff))
                    continue;
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                if (source_selector) {
                    const auto& selected = source_caches[omp_thread_index()].get(
                        shells_ref[s1], shells_g[s2]);
                    if (selected.charges.empty()) continue;
                    engine.set_params(erfc_params{omega, selected.charges});
                }
                engine.compute(shells_ref[s1], shells_g[s2]);
                const double* tile = buf[0];
                if (!tile) continue;
                for (std::size_t i = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        set.blocks[c](bf1 + i, bf2 + j) += tile[i * n2 + j];
                    }
                }
            }
        }
    }
}

// Per-AO factor taking the raw AO-pair FT convention to libint's: 1 for s,
// sqrt(4π/(2l+1)) for l > 0 (vibeqc.bipole_ext_el_pole.
// _libint_ylm_correction_per_ao). Empty when a shell lies outside the native
// pair-FT kernel's contract (pure shells through kMaxL, or s).
std::vector<double> pair_ft_ao_scales(const libint2::BasisSet& shells) {
    std::vector<double> scales;
    scales.reserve(shells.nbf());
    for (const auto& shell : shells) {
        for (const auto& contraction : shell.contr) {
            const int l = contraction.l;
            if (l > cart_to_sph_data::kMaxL || (l > 0 && !contraction.pure)) {
                return {};
            }
            const double factor =
                l == 0 ? 1.0 : std::sqrt(4.0 * M_PI / (2.0 * l + 1.0));
            scales.insert(scales.end(), contraction.size(), factor);
        }
    }
    return scales;
}

// Reciprocal-lattice vectors 0 < |G| <= k_max, rows of the returned matrix.
// |n_i| = |G·a_i| / 2π <= k_max |a_i| / 2π bounds each Miller index.
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
reciprocal_vectors_within(const PeriodicSystem& system, double k_max) {
    const Eigen::Matrix3d B = system.reciprocal_lattice();
    std::array<int, 3> n_max{};
    for (int i = 0; i < 3; ++i) {
        n_max[i] = static_cast<int>(
            std::ceil(k_max * system.lattice.col(i).norm() / (2.0 * M_PI)));
    }
    std::vector<Eigen::Vector3d> kept;
    const double k_max_sq = k_max * k_max;
    for (int n0 = -n_max[0]; n0 <= n_max[0]; ++n0) {
        for (int n1 = -n_max[1]; n1 <= n_max[1]; ++n1) {
            for (int n2 = -n_max[2]; n2 <= n_max[2]; ++n2) {
                if (n0 == 0 && n1 == 0 && n2 == 0) continue;
                const Eigen::Vector3d G = n0 * B.col(0) + n1 * B.col(1) + n2 * B.col(2);
                if (G.squaredNorm() <= k_max_sq) kept.push_back(G);
            }
        }
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> out(
        static_cast<Eigen::Index>(kept.size()), 3);
    for (std::size_t i = 0; i < kept.size(); ++i) {
        out.row(static_cast<Eigen::Index>(i)) = kept[i].transpose();
    }
    return out;
}

}  // namespace

LatticeMatrixSet compute_vsap_lattice(const BasisSet& basis,
                                      const PeriodicSystem& system,
                                      const Grid& grid,
                                      const std::string& sap_basis_name,
                                      const LatticeSumOptions& opts,
                                      const EwaldOptions& ewald_opts) {
    if (system.dim != 3) {
        throw std::invalid_argument(
            "compute_vsap_lattice: 3D SAP requires dim == 3. 1D / 2D SAP "
            "guesses are a separate follow-up.");
    }
    ensure_libint_initialized();
    const std::map<int, SAPExpansion>& sap_table =
        sap_expansions(sap_basis_name);

    // Per-element net effective charge Q_A = Σ_i c_i; validate SAP coverage.
    auto net_charge = [&](int Z) {
        const auto it = sap_table.find(Z);
        if (it == sap_table.end()) {
            throw std::runtime_error(
                "compute_vsap_lattice: no SAP data for Z=" + std::to_string(Z)
                + " in the chosen SAP basis. Periodic SAP currently requires "
                  "every element to have SAP data (bare-nuclear fallback is a "
                  "follow-up); use SAD or HCORE for this system.");
        }
        double q = 0.0;
        for (double c : it->second.coeffs) q += c;
        return q;
    };
    for (const auto& atom : system.unit_cell) (void)net_charge(atom.Z);

    // Moderate Ewald split parameter η (same auto-rule as the Ewald engine).
    const double eta = (ewald_opts.alpha > 0.0)
        ? ewald_opts.alpha
        : std::sqrt(-std::log(ewald_opts.tolerance)) / ewald_opts.real_cutoff_bohr;

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());

    LatticeMatrixSet set;
    set.nbf = nbf;
    set.cells = opts.pair_complete_1e
        ? pair_cells_for_1e(shells_ref, system, opts.cutoff_bohr)
        : direct_lattice_cells(system, opts.cutoff_bohr);
    set.blocks.assign(set.cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));

    // Image cells for the short-range erfc point-charge lists.
    const auto charge_cells =
        direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    auto image_charges = [&](const std::vector<std::size_t>& atom_idx,
                             double q_each) {
        std::vector<std::pair<double, std::array<double, 3>>> q;
        q.reserve(charge_cells.size() * atom_idx.size());
        for (const auto& cell : charge_cells) {
            for (std::size_t a : atom_idx) {
                const auto& at = system.unit_cell[a];
                q.emplace_back(q_each, std::array<double, 3>{
                    at.xyz[0] + cell.r_cart[0],
                    at.xyz[1] + cell.r_cart[1],
                    at.xyz[2] + cell.r_cart[2]});
            }
        }
        return q;
    };

    // ---- Short-range, analytical libint erfc ----
    // libint's erfc_nuclear applies the −q nuclear sign, so a point charge q
    // contributes −q·erfc/r.
    // (a) η-erfc with charges −Q_A  → +Σ_A Q_A erfc(η)/r.
    std::vector<std::pair<double, std::array<double, 3>>> eta_q;
    eta_q.reserve(charge_cells.size() * system.unit_cell.size());
    for (const auto& cell : charge_cells) {
        for (const auto& at : system.unit_cell) {
            eta_q.emplace_back(-net_charge(at.Z), std::array<double, 3>{
                at.xyz[0] + cell.r_cart[0],
                at.xyz[1] + cell.r_cart[1],
                at.xyz[2] + cell.r_cart[2]});
        }
    }
    const PairNuclearImageSelector selector(system, opts.nuclear_cutoff_bohr);
    const auto* source_selector = opts.pair_complete_1e ? &selector : nullptr;
    std::vector<double> source_weights;
    for (const auto& atom : system.unit_cell) source_weights.push_back(-net_charge(atom.Z));
    accumulate_erfc_lattice(set, shells_ref, eta, eta_q,
                            opts.pair_complete_1e ? opts.cutoff_bohr : 0.0,
                            source_selector, source_weights);

    // (b) per-(element, primitive) erfc with charges +c_i  → −Σ c_i erfc(ω_i)/r.
    //     Atoms grouped by element so one erfc call covers all of them.
    std::map<int, std::vector<std::size_t>> atoms_by_z;
    for (std::size_t a = 0; a < system.unit_cell.size(); ++a) {
        atoms_by_z[system.unit_cell[a].Z].push_back(a);
    }
    for (const auto& [z, idx] : atoms_by_z) {
        const SAPExpansion& e = sap_table.at(z);
        for (std::size_t i = 0; i < e.alphas.size(); ++i) {
            source_weights.assign(system.unit_cell.size(), 0.0);
            for (std::size_t atom : idx) source_weights[atom] = e.coeffs[i];
            accumulate_erfc_lattice(set, shells_ref, std::sqrt(e.alphas[i]),
                                    image_charges(idx, e.coeffs[i]),
                                    opts.pair_complete_1e ? opts.cutoff_bohr : 0.0,
                                    source_selector, source_weights);
        }
    }

    // ---- Long-range, smooth erf(η r)/r, analytic in reciprocal space ----
    // v_long(r) = +Σ_A Q_A erf(η r)/r (reciprocal lattice sum). Each block
    // integrates one AO pair against that periodic potential's Fourier series,
    //   V_long(g) = (1/Ω) Re Σ_{G≠0} w(G) conj(FT_μν(G; g)),
    //   w(G) = (4π/G²) e^{−G²/4η²} Σ_A Q_A e^{−iG·R_A},
    // the per-cell form of vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_lattice
    // (Sun et al., JCP 147, 164119 (2017), doi:10.1063/1.4998644). The G = 0
    // background −π ΣQ/(η²Ω)·S(g) of the Ewald potential is kept; it shifts
    // F_SAP by c·S and leaves the guess density unchanged. A molecular grid of
    // the home-cell atoms used to integrate this term, so the blocks changed
    // when an atom moved by a lattice vector (0.29 Ha on MgO/def2-SVP, #483).
    // The grid is used only for shells outside the pair-FT kernel's contract.
    const std::vector<double> ao_scales = pair_ft_ao_scales(shells_ref);
    if (!ao_scales.empty()) {
        const double volume = std::abs(system.lattice.determinant());
        const double k_max = (ewald_opts.recip_cutoff_bohr_inv > 0.0)
            ? ewald_opts.recip_cutoff_bohr_inv
            : 2.0 * eta * std::sqrt(-std::log(ewald_opts.tolerance));
        const auto G = reciprocal_vectors_within(system, k_max);
        double q_total = 0.0;
        Eigen::VectorXcd weights = Eigen::VectorXcd::Zero(G.rows());
        for (const auto& atom : system.unit_cell) {
            const double q = net_charge(atom.Z);
            q_total += q;
            const Eigen::Vector3d R(atom.xyz[0], atom.xyz[1], atom.xyz[2]);
            for (Eigen::Index k = 0; k < G.rows(); ++k) {
                weights[k] += q * std::polar(1.0, -G.row(k).dot(R));
            }
        }
        for (Eigen::Index k = 0; k < G.rows(); ++k) {
            const double G2 = G.row(k).squaredNorm();
            weights[k] *= 4.0 * M_PI / G2 * std::exp(-G2 / (4.0 * eta * eta));
        }
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> R_cells(
            static_cast<Eigen::Index>(set.cells.size()), 3);
        for (std::size_t c = 0; c < set.cells.size(); ++c) {
            R_cells.row(static_cast<Eigen::Index>(c)) = set.cells[c].r_cart.transpose();
        }
        const std::vector<double> v_long = G.rows() == 0
            ? std::vector<double>(set.cells.size() * nbf * nbf, 0.0)
            : ao_pair_fourier_transform_weighted_per_cell(basis, G, R_cells, weights);
        const auto overlap = compute_overlap_lattice_explicit(
            basis, system, set.cells, opts.pair_complete_1e ? opts.cutoff_bohr : 0.0);
        const double background = -M_PI * q_total / (eta * eta * volume);
        const std::size_t stride = static_cast<std::size_t>(nbf) * nbf;
        for (std::size_t c = 0; c < set.cells.size(); ++c) {
            auto& block = set.blocks[c];
            for (int m = 0; m < nbf; ++m) {
                for (int n = 0; n < nbf; ++n) {
                    block(m, n) += ao_scales[m] * ao_scales[n] / volume
                        * v_long[c * stride + static_cast<std::size_t>(m) * nbf + n];
                }
            }
            block += background * overlap.blocks[c];
            if (opts.pair_complete_1e) {
                filter_pair_block(block, shells_ref, set.cells[c].r_cart, opts.cutoff_bohr);
            }
        }
        return set;
    }
    if (grid.points.rows() == 0) {
        throw std::invalid_argument(
            "compute_vsap_lattice: the analytic long-range term needs pure "
            "shells through L=6 (or s); pass a grid for other bases");
    }
    const std::size_t n_atoms = system.unit_cell.size();
    Eigen::Matrix3Xd qpos(3, n_atoms);
    Eigen::VectorXd qval(n_atoms);
    for (std::size_t a = 0; a < n_atoms; ++a) {
        const auto& at = system.unit_cell[a];
        qpos(0, a) = at.xyz[0];
        qpos(1, a) = at.xyz[1];
        qpos(2, a) = at.xyz[2];
        qval(a) = net_charge(at.Z);
    }
    EwaldOptions eo = ewald_opts;
    eo.alpha = eta;
    const Eigen::VectorXd v_long_r = ewald_point_charge_potential(
        system.lattice, qpos, qval, grid.points, eo,
        /*include_short_range=*/false);

    const Eigen::MatrixXd chi_ref = evaluate_ao(basis, grid.points);
    const Eigen::VectorXd wv = grid.weights.array() * v_long_r.array();
    const Eigen::MatrixXd chi_wv = chi_ref.array().colwise() * wv.array();

    const int n_cells = static_cast<int>(set.cells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const BasisSet basis_g =
            shifted_basis_for_cell(basis, system, set.cells[c].r_cart);
        const Eigen::MatrixXd chi_g = evaluate_ao(basis_g, grid.points);
        set.blocks[c] += chi_wv.transpose() * chi_g;
        if (opts.pair_complete_1e) {
            filter_pair_block(set.blocks[c], shells_ref,
                              set.cells[c].r_cart, opts.cutoff_bohr);
        }
    }

    return set;
}

// ---- Public explicit-cell wrappers (Phase SYM3b, vibeqc scope) -----------

LatticeMatrixSet compute_overlap_lattice_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    double pair_cutoff) {
    return compute_1e_lattice_matrix_explicit(
        basis, system, cells, libint2::Operator::overlap, nullptr,
        pair_cutoff);
}

LatticeMatrixSet compute_kinetic_lattice_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    double pair_cutoff) {
    return compute_1e_lattice_matrix_explicit(
        basis, system, cells, libint2::Operator::kinetic, nullptr,
        pair_cutoff);
}

LatticeMatrixSet compute_nuclear_lattice_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<LatticeCell>& cells) {
    const auto nuclei = build_periodic_nuclear_charges(system, opts);
    const PairNuclearImageSelector selector(system, opts.nuclear_cutoff_bohr);
    return compute_1e_lattice_matrix_explicit(
        basis, system, cells, libint2::Operator::nuclear, &nuclei,
        opts.pair_complete_1e ? opts.cutoff_bohr : 0.0,
        opts.pair_complete_1e ? &selector : nullptr);
}

}  // namespace vibeqc
