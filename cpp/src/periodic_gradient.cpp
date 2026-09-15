#include "vibeqc/periodic_gradient.hpp"

#include <libint2/atom.h>
#include <libint2/engine.h>
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vibeqc/ewald.hpp"               // Ewald nuclear-gradient primitives
#include "vibeqc/init.hpp"
#include "vibeqc/lattice_integrals.hpp"   // direct_lattice_cells
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/schwarz.hpp"
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {

namespace {

// Same helper used by gradient.cpp and lattice_integrals.cpp — kept
// private to avoid coupling.
std::vector<libint2::Atom> to_libint_atoms(const std::vector<Atom>& atoms) {
    std::vector<libint2::Atom> out;
    out.reserve(atoms.size());
    for (const auto& a : atoms) {
        libint2::Atom la;
        la.atomic_number = a.Z;
        la.x = a.xyz[0];
        la.y = a.xyz[1];
        la.z = a.xyz[2];
        out.push_back(la);
    }
    return out;
}

// shell → atom index (inside the unit cell). Same ordering as the
// molecular helper.
std::vector<long> shell_to_atom(const BasisSet& basis,
                                  const std::vector<Atom>& atoms) {
    return basis.libint().shell2atom(to_libint_atoms(atoms));
}

// Translate every shell origin by dr (bohr). Contraction coefficients
// + primitive exponents stay; only origin moves. Mirrors the helper
// in lattice_integrals.cpp.
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

// Cell-index-map type matching periodic_fock.cpp's helper. We re-
// declare it here (private) rather than expose it from the Fock
// implementation file to avoid coupling.
struct LatticeIdxHash {
    std::size_t operator()(const Eigen::Vector3i& v) const noexcept {
        const auto a = static_cast<std::size_t>(v[0]) + 0x9e3779b9ULL;
        const auto b = static_cast<std::size_t>(v[1]) + 0x85ebca6bULL;
        const auto c = static_cast<std::size_t>(v[2]) + 0xc2b2ae35ULL;
        return static_cast<std::size_t>((a * 2654435761ULL) ^
                                        (b * 40503ULL) ^ c);
    }
};
struct LatticeIdxEq {
    bool operator()(const Eigen::Vector3i& a,
                    const Eigen::Vector3i& b) const noexcept {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};
using CellIdxMap = std::unordered_map<Eigen::Vector3i, int,
                                       LatticeIdxHash, LatticeIdxEq>;

CellIdxMap build_cell_idx_map(const std::vector<LatticeCell>& cells) {
    CellIdxMap m;
    m.reserve(cells.size() * 2);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        m[cells[i].index] = static_cast<int>(i);
    }
    return m;
}

}  // namespace

// =======================================================================
// Nuclear-repulsion gradient (per-cell convention)
// =======================================================================

Eigen::MatrixXd nuclear_repulsion_gradient_per_cell(
    const PeriodicSystem& system,
    const LatticeSumOptions& opts) {
    if (opts.coulomb_method == CoulombMethod::SLAB_EWALD_2D ||
        opts.coulomb_method == CoulombMethod::NEUTRALIZED_1D) {
        throw std::runtime_error(
            "nuclear_repulsion_gradient_per_cell: 1D / 2D Ewald variants "
            "are not yet dispatched here. Use DIRECT_TRUNCATED or EWALD_3D.");
    }
    if (opts.coulomb_method == CoulombMethod::EWALD_3D) {
        // Match nuclear_repulsion_per_cell's EWALD_3D option mapping exactly:
        // the Ewald defaults choose alpha / reciprocal cutoff, and the lattice
        // nuclear cutoff controls the real-space sum. Falling through to the
        // direct 1/r^3 derivative mixes gauges with the Ewald E_nn energy.
        EwaldOptions eopts;
        eopts.real_cutoff_bohr = opts.nuclear_cutoff_bohr;
        return ewald_nuclear_repulsion_gradient(system, eopts);
    }

    const auto cells = direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    const auto& atoms = system.unit_cell;
    const std::size_t N = atoms.size();
    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    if (opts.pair_complete_1e) {
        const PairNuclearImageSelector selector(system, opts.nuclear_cutoff_bohr);
        for (std::size_t a = 0; a < N; ++a) {
            const Eigen::Vector3d center(atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
            const auto images = selector.select(center);
            for (std::size_t image = 0; image < images.charges.size(); ++image) {
                const auto& charge = images.charges[image];
                const auto b = images.atoms[image];
                if (static_cast<long>(a) == b) continue;
                const Eigen::Vector3d delta = center - Eigen::Vector3d(
                    charge.second[0], charge.second[1], charge.second[2]);
                const double r2 = delta.squaredNorm();
                if (r2 < 1e-28) continue;
                const Eigen::Vector3d contribution =
                    (-0.5 * atoms[a].Z * charge.first / (r2 * std::sqrt(r2))) * delta;
                grad.row(a) += contribution.transpose();
                grad.row(b) -= contribution.transpose();
            }
        }
        return grad;
    }

    // F_A = +Σ_{B≠A, all c} Z_A Z_B (r_A - r_B - c) / |r_A - r_B - c|^3.
    //
    // = the *negative* of ∂E_nuc^pc/∂r_A (we return the gradient, so
    //   the sign in our buffer matches molecular nuclear_repulsion_gradient
    //   which computes ∂E/∂r_A directly — that comes out negative when
    //   atoms are pushed apart by Coulomb repulsion).
    //
    // Self-image pairs (B=A, c≠0) cancel exactly when both ∂/∂r_A
    // contributions are summed (see derivation in the design notes);
    // the cleanest implementation is to skip the (B=A, all c) self-
    // pair entirely.
    for (const auto& cell : cells) {
        for (std::size_t a = 0; a < N; ++a) {
            for (std::size_t b = 0; b < N; ++b) {
                if (a == b) continue;   // self-pair (any c) → zero force
                const double dx = atoms[a].xyz[0] - atoms[b].xyz[0]
                                   - cell.r_cart[0];
                const double dy = atoms[a].xyz[1] - atoms[b].xyz[1]
                                   - cell.r_cart[1];
                const double dz = atoms[a].xyz[2] - atoms[b].xyz[2]
                                   - cell.r_cart[2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < 1.0e-28) continue;   // numerical guard
                const double r3 = r2 * std::sqrt(r2);
                const double pref =
                    -static_cast<double>(atoms[a].Z) *
                     static_cast<double>(atoms[b].Z) / r3;
                grad(a, 0) += pref * dx;
                grad(a, 1) += pref * dy;
                grad(a, 2) += pref * dz;
            }
        }
    }
    return grad;
}

// The analytic partner of periodic_rhf.cpp::nuclear_repulsion_slab_ewald_2d:
// gradient of {background-neutralised 2D Ewald at z_b = 0} minus {the exact
// uniform-sheet term}, with the SCF's actual α and real cutoff. See
// periodic_gradient.hpp and the term-by-term derivation in
// ewald.cpp::ewald_2d_point_charge_gradient_with_background. (Parry, Surf.
// Sci. 49, 433 (1975); de Leeuw & Perram, Mol. Phys. 37, 1313 (1979).)
Eigen::MatrixXd nuclear_repulsion_slab_ewald_2d_gradient(
    const PeriodicSystem& system,
    const LatticeSumOptions& opts) {
    if (system.dim != 2) {
        throw std::invalid_argument(
            "nuclear_repulsion_slab_ewald_2d_gradient: SLAB_EWALD_2D "
            "requires dim == 2");
    }

    const auto& atoms = system.unit_cell;
    const int n = static_cast<int>(atoms.size());
    if (n == 0) return Eigen::MatrixXd::Zero(0, 3);

    Eigen::Matrix3Xd positions(3, n);
    Eigen::VectorXd charges(n);
    for (int i = 0; i < n; ++i) {
        positions.col(i) = Eigen::Vector3d(
            atoms[i].xyz[0], atoms[i].xyz[1], atoms[i].xyz[2]);
        charges[i] = static_cast<double>(atoms[i].Z);
    }

    // Mirror the energy's parameterisation exactly (differentiate what the
    // SCF evaluates): α from slab_ewald_alpha (default 0.4), real cutoff
    // from nuclear_cutoff_bohr, z_background = 0.
    EwaldOptions eopts;
    eopts.alpha = opts.slab_ewald_alpha > 0.0 ? opts.slab_ewald_alpha : 0.4;
    eopts.real_cutoff_bohr = opts.nuclear_cutoff_bohr;

    const Eigen::Vector3d a1 = system.lattice.col(0);
    const Eigen::Vector3d a2 = system.lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double area = cross.norm();
    if (area < 1.0e-14) {
        throw std::invalid_argument(
            "nuclear_repulsion_slab_ewald_2d_gradient: degenerate in-plane "
            "lattice vectors");
    }
    const Eigen::Vector3d nhat = cross / area;

    const double z_background = 0.0;
    Eigen::Matrix3Xd grad = ewald_2d_point_charge_gradient_with_background(
        system.lattice, positions, charges, z_background, eopts);

    // Remove the sheet term's gradient to recover the bare four-term block
    // (the energy subtracts e_sheet = −(2π/A) q_bg Σ_i Z_i |z_i − z_b|, so
    // the gradient subtracts −(2π/A) q_bg Z_i sgn(z_i − z_b) n̂).
    const double q_background = -charges.sum();
    const double sheet_pref = -(2.0 * M_PI / area) * q_background;
    for (int i = 0; i < n; ++i) {
        const double dz = positions.col(i).dot(nhat) - z_background;
        const double sgn = (dz > 0.0) ? 1.0 : ((dz < 0.0) ? -1.0 : 0.0);
        grad.col(i) -= (sheet_pref * charges[i] * sgn) * nhat;
    }

    // Return N × 3 (row = atom), matching nuclear_repulsion_gradient_per_cell.
    return grad.transpose();
}

// =======================================================================
// Lattice-summed 1-e gradient contributions
// =======================================================================
//
// All three (overlap / kinetic / nuclear-attraction) follow the same
// pattern: outer loop over cell vectors g, build shifted shells_g
// (origin += g), call libint with deriv_order=1 on the shell pair
// (s1 in the reference cell, s2 in the shifted cell), and contract
// with the supplied real-space density matrix block at cell g.
//
// Important convention. When atom A in the *reference* cell moves by
// δR, every periodic image of A also moves by δR. So if a derivative
// buffer is "with respect to the origin of shell s2" — and s2 lives
// in the shifted cell on atom A's *image* — the contribution still
// goes onto atom A (not onto a separate "image atom"). The assignment
// `atom = (icenter == 0) ? atom1 : atom2` already reflects this,
// because shell2atom returns the *unit-cell* atom index for both s2
// (in the reference cell) and s2's image (in the shifted cell): they
// are the same logical atom.

// ----- Overlap lattice gradient (∂S/∂R contracted with W) ---------------

Eigen::MatrixXd overlap_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& W_set,
    const LatticeSumOptions& opts,
    bool apply_pair_filter) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const auto shell2bf = shells_ref.shell2bf();
    const auto s2a = shell_to_atom(basis, system.unit_cell);
    const std::size_t N = system.unit_cell.size();

    libint2::Engine prototype(libint2::Operator::overlap,
                              shells_ref.max_nprim(), shells_ref.max_l(),
                              1 /*deriv_order*/);
    auto engines = make_engine_pool(prototype);

    // Use the cell list from W_set itself — it must match the cell
    // list compute_overlap_lattice produced in the same SCF run, so
    // the per-cell W blocks line up with the integrals we evaluate
    // here.
    const auto& cells = W_set.cells;
    const int n_cells = static_cast<int>(cells.size());
    const int n_shells = static_cast<int>(shells_ref.size());

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);
        const Eigen::MatrixXd& W = W_set.blocks[c];

        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                // Differentiate EXACTLY the sum the energy builds: the
                // one-electron lattice sums bound the physical pair
                // separation |O_mu - O_nu - g|, not |g| (see
                // vibeqc/lattice_pair_cells.hpp). A derivative that
                // enumerated a different term set would not be the
                // derivative of the energy, and shows up as an
                // analytic-vs-FD residual at the dropped pairs.
                if (apply_pair_filter && opts.pair_complete_1e &&
                    !pair_in_range(shells_ref[s1], shells_g[s2],
                                   opts.cutoff_bohr)) {
                    continue;
                }
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                const long atom2 = s2a[s2];

                engine.compute(shells_ref[s1], shells_g[s2]);
                // 6 derivative buffers: (x,y,z) for center 1, then center 2.
                for (int icenter = 0; icenter < 2; ++icenter) {
                    const long atom = (icenter == 0) ? atom1 : atom2;
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t i = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                acc += W(bf1 + i, bf2 + j)
                                       * block[i * n2 + j];
                            }
                        }
                        // E contribution is -tr(W · dS); gradient
                        // picks up the sign.
                        grad_local(atom, d) -= acc;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

// ----- Kinetic lattice gradient (∂T/∂R contracted with D) ---------------

Eigen::MatrixXd kinetic_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const auto shell2bf = shells_ref.shell2bf();
    const auto s2a = shell_to_atom(basis, system.unit_cell);
    const std::size_t N = system.unit_cell.size();

    libint2::Engine prototype(libint2::Operator::kinetic,
                              shells_ref.max_nprim(), shells_ref.max_l(),
                              1 /*deriv_order*/);
    auto engines = make_engine_pool(prototype);

    const auto& cells = D_set.cells;
    const int n_cells = static_cast<int>(cells.size());
    const int n_shells = static_cast<int>(shells_ref.size());

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);
        const Eigen::MatrixXd& D = D_set.blocks[c];

        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                // Differentiate EXACTLY the sum the energy builds: the
                // one-electron lattice sums bound the physical pair
                // separation |O_mu - O_nu - g|, not |g| (see
                // vibeqc/lattice_pair_cells.hpp). A derivative that
                // enumerated a different term set would not be the
                // derivative of the energy, and shows up as an
                // analytic-vs-FD residual at the dropped pairs.
                if (opts.pair_complete_1e &&
                    !pair_in_range(shells_ref[s1], shells_g[s2],
                                   opts.cutoff_bohr)) {
                    continue;
                }
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                const long atom2 = s2a[s2];

                engine.compute(shells_ref[s1], shells_g[s2]);
                for (int icenter = 0; icenter < 2; ++icenter) {
                    const long atom = (icenter == 0) ? atom1 : atom2;
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t i = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                acc += D(bf1 + i, bf2 + j)
                                       * block[i * n2 + j];
                            }
                        }
                        grad_local(atom, d) += acc;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

// ----- Nuclear-attraction lattice gradient ------------------------------

Eigen::MatrixXd nuclear_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const auto shell2bf = shells_ref.shell2bf();
    const auto s2a = shell_to_atom(basis, system.unit_cell);
    const std::size_t N = system.unit_cell.size();

    // Build the lattice-summed nuclear point-charge list. Each entry
    // is (Z, position) for one atom-replica within nuclear_cutoff.
    const auto charge_cells =
        direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    // Track which unit-cell atom each nuclear-point-charge index
    // belongs to (so derivative buffers from the libint nuclear-
    // engine can be routed back to the right unit-cell atom).
    std::vector<long> q_atom;
    q.reserve(charge_cells.size() * N);
    q_atom.reserve(charge_cells.size() * N);
    for (const auto& cc : charge_cells) {
        for (std::size_t a = 0; a < N; ++a) {
            const auto& at = system.unit_cell[a];
            std::array<double, 3> r = {
                at.xyz[0] + cc.r_cart[0],
                at.xyz[1] + cc.r_cart[1],
                at.xyz[2] + cc.r_cart[2],
            };
            q.emplace_back(static_cast<double>(at.Z), r);
            q_atom.push_back(static_cast<long>(a));
        }
    }
    const PairNuclearImageSelector selector(system, opts.nuclear_cutoff_bohr);
    std::vector<PairNuclearImageCache> source_caches;
    for (int tid = 0; tid < omp_max_threads(); ++tid) source_caches.emplace_back(selector);

    libint2::Engine prototype(libint2::Operator::nuclear,
                              shells_ref.max_nprim(), shells_ref.max_l(),
                              1 /*deriv_order*/);
    prototype.set_params(q);
    auto engines = make_engine_pool(prototype);

    const auto& cells = D_set.cells;
    const int n_cells = static_cast<int>(cells.size());
    const int n_shells = static_cast<int>(shells_ref.size());

    // Number of derivative buffers per shell pair: 3 · (2 basis
    // centers + N_q point charges).


    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);
        const Eigen::MatrixXd& D = D_set.blocks[c];

        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                // Differentiate EXACTLY the sum the energy builds: the
                // one-electron lattice sums bound the physical pair
                // separation |O_mu - O_nu - g|, not |g| (see
                // vibeqc/lattice_pair_cells.hpp). A derivative that
                // enumerated a different term set would not be the
                // derivative of the energy, and shows up as an
                // analytic-vs-FD residual at the dropped pairs.
                if (opts.pair_complete_1e &&
                    !pair_in_range(shells_ref[s1], shells_g[s2],
                                   opts.cutoff_bohr)) {
                    continue;
                }
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                const long atom2 = s2a[s2];

                const std::vector<long>* parents = &q_atom;
                if (opts.pair_complete_1e) {
                    const auto& selected = source_caches[tid].get(shells_ref[s1], shells_g[s2]);
                    if (selected.charges.empty()) continue;
                    engine.set_params(selected.charges);
                    parents = &selected.atoms;
                }
                const std::size_t ncenters = 2 + parents->size();
                engine.compute(shells_ref[s1], shells_g[s2]);
                for (std::size_t icenter = 0; icenter < ncenters; ++icenter) {
                    long atom;
                    if (icenter == 0) atom = atom1;
                    else if (icenter == 1) atom = atom2;
                    else {
                        // Nuclear point charges 0..N_q-1; q_atom maps
                        // each one back to its unit-cell atom index
                        // (image atoms in cell c contribute to the
                        // same unit-cell atom because they translate
                        // together with their reference-cell sibling).
                        atom = (*parents)[icenter - 2];
                    }
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t i = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                acc += D(bf1 + i, bf2 + j)
                                       * block[i * n2 + j];
                            }
                        }
                        grad_local(atom, d) += acc;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

// =======================================================================
// Screened (erfc) nuclear-attraction lattice gradient — V_short piece of
// the Ewald V_ne gradient. Mirrors nuclear_lattice_gradient_contribution
// with libint's erfc_nuclear operator (params = {ω, point-charge list}).
// =======================================================================

Eigen::MatrixXd nuclear_erfc_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts,
    double omega) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const auto shell2bf = shells_ref.shell2bf();
    const auto s2a = shell_to_atom(basis, system.unit_cell);
    const std::size_t N = system.unit_cell.size();

    // Lattice-summed nuclear point-charge list (same cutoff as the energy
    // V_short, compute_nuclear_erfc_lattice → build_periodic_nuclear_charges
    // uses opts.nuclear_cutoff_bohr).
    const auto charge_cells =
        direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    std::vector<long> q_atom;
    q.reserve(charge_cells.size() * N);
    q_atom.reserve(charge_cells.size() * N);
    for (const auto& cc : charge_cells) {
        for (std::size_t a = 0; a < N; ++a) {
            const auto& at = system.unit_cell[a];
            std::array<double, 3> r = {
                at.xyz[0] + cc.r_cart[0],
                at.xyz[1] + cc.r_cart[1],
                at.xyz[2] + cc.r_cart[2],
            };
            q.emplace_back(static_cast<double>(at.Z), r);
            q_atom.push_back(static_cast<long>(a));
        }
    }
    const PairNuclearImageSelector selector(system, opts.nuclear_cutoff_bohr);
    std::vector<PairNuclearImageCache> source_caches;
    for (int tid = 0; tid < omp_max_threads(); ++tid) source_caches.emplace_back(selector);

    libint2::Engine prototype(libint2::Operator::erfc_nuclear,
                              shells_ref.max_nprim(), shells_ref.max_l(),
                              1 /*deriv_order*/);
    using erfc_params =
        libint2::operator_traits<libint2::Operator::erfc_nuclear>::oper_params_type;
    prototype.set_params(erfc_params{omega, q});
    auto engines = make_engine_pool(prototype);

    const auto& cells = D_set.cells;
    const int n_cells = static_cast<int>(cells.size());
    const int n_shells = static_cast<int>(shells_ref.size());

    // 3 · (2 basis centers + N_q point charges) derivative buffers.


    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const Eigen::Vector3d& g = cells[c].r_cart;
        const auto shells_g = shift_shells(shells_ref, g);
        const Eigen::MatrixXd& D = D_set.blocks[c];

        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                if (opts.pair_complete_1e &&
                    !pair_in_range(shells_ref[s1], shells_g[s2], opts.cutoff_bohr))
                    continue;
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                const long atom2 = s2a[s2];

                const std::vector<long>* parents = &q_atom;
                if (opts.pair_complete_1e) {
                    const auto& selected = source_caches[tid].get(shells_ref[s1], shells_g[s2]);
                    if (selected.charges.empty()) continue;
                    engine.set_params(erfc_params{omega, selected.charges});
                    parents = &selected.atoms;
                }
                const std::size_t ncenters = 2 + parents->size();
                engine.compute(shells_ref[s1], shells_g[s2]);
                for (std::size_t icenter = 0; icenter < ncenters; ++icenter) {
                    long atom;
                    if (icenter == 0) atom = atom1;
                    else if (icenter == 1) atom = atom2;
                    else atom = (*parents)[icenter - 2];
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t i = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                acc += D(bf1 + i, bf2 + j)
                                       * block[i * n2 + j];
                            }
                        }
                        grad_local(atom, d) += acc;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

// =======================================================================
// 2-electron ERI lattice gradient
// =======================================================================
//
// Σ_{c_g, c_lam, c_sig} Σ_μνλσ Γ(c_g, c_lam, c_sig)_μνλσ ·
//        ∂(μ_0 ν_{c_g} | λ_{c_lam} σ_{c_sig}) / ∂R
//
// with Γ_J = (1/2) D(c_g)_μν D(c_sig - c_lam)_λσ.
// The full-Coulomb K path keeps the true-periodic exchange-density slots
// validated by tests/test_periodic_gradient_g1a.py. The screened K_SR
// path used by the corrected BIPOLE gauge differentiates the real-space
// K_SR energy directly with D(c_g)_μν D(c_sig - c_lam)_λσ on the permuted
// exchange integral (μ_0 λ_{c_lam} | ν_{c_g} σ_{c_sig}).
//
// Mirrors build_fock_2e_real_space's cell-quartet structure; replaces
// each ``engine.compute(...)`` with a deriv_order=1 engine and
// iterates the 12 derivative buffers per quartet (3 × 4 centers).
// Atom-mapping rule: each shell-center derivative routes back to the
// unit-cell atom that the shell *belongs to* (image atoms move with
// their reference siblings, so the same atom index applies regardless
// of cell offset).

Eigen::MatrixXd eri_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts,
    double alpha_hf,
    double j_scale,
    double omega,
    bool exchange_energy_convention,
    const std::vector<LatticeCell>& internal_cells,
    const std::vector<LatticeCell>& requested_output_cells,
    const std::vector<std::vector<uint8_t>>& output_shell_masks) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (D_set.nbf != nbf) {
        throw std::runtime_error(
            "eri_lattice_gradient_contribution: density nbf mismatch");
    }

    const auto& density_cells = D_set.cells;
    const auto automatic_outputs = opts.pair_complete_1e && requested_output_cells.empty()
        ? pair_complete_eri_cells(system, opts, shells_ref)
        : std::vector<LatticeCell>{};
    const auto& output_cells = requested_output_cells.empty()
        ? (opts.pair_complete_1e ? automatic_outputs : density_cells)
        : requested_output_cells;
    const auto automatic_internal = opts.pair_complete_1e && internal_cells.empty()
        ? pair_complete_eri_cells(system, opts, shells_ref)
        : std::vector<LatticeCell>{};
    const auto& cells = internal_cells.empty()
        ? (opts.pair_complete_1e ? automatic_internal : output_cells) : internal_cells;
    const bool legacy_single_domain = !opts.pair_complete_1e && internal_cells.empty() &&
        requested_output_cells.empty() && output_shell_masks.empty();
    const double interaction_cutoff = eri_interaction_cutoff(opts);
    const std::size_t n_g = output_cells.size();
    const std::size_t n_c = cells.size();

    // The padded M5 traversal keeps the Fock-output / contracting-density
    // cell list separate from the density support, while extending the
    // internal lambda/sigma image sum. Every output cell must therefore be
    // present in both the internal list and the density support.
    const auto internal_idx = build_cell_idx_map(cells);
    const auto density_idx = build_cell_idx_map(density_cells);
    std::vector<int> output_internal_positions(n_g, -1);
    std::vector<int> output_density_positions(n_g, -1);
    for (std::size_t c_g = 0; c_g < n_g; ++c_g) {
        auto internal_it = internal_idx.find(output_cells[c_g].index);
        if (internal_it == internal_idx.end()) {
            throw std::runtime_error(
                "eri_lattice_gradient_contribution: output cell is "
                "absent from internal_cells");
        }
        auto density_it = density_idx.find(output_cells[c_g].index);
        if (density_it == density_idx.end()) {
            throw std::runtime_error(
                "eri_lattice_gradient_contribution: output cell is absent "
                "from the density support");
        }
        output_internal_positions[c_g] = internal_it->second;
        output_density_positions[c_g] = density_it->second;
    }

    // Pre-shift shells once per cell.
    std::vector<std::vector<libint2::Shell>> shells_at(n_c);
    for (std::size_t c = 0; c < n_c; ++c) {
        shells_at[c] = shift_shells(shells_ref, cells[c].r_cart);
    }

    // Cell-index lookup: given a cell-index difference Δ, return the
    // index into D_set.blocks (or sentinel -1 if outside the cutoff).
    auto block_for_diff = [&](const Eigen::Vector3i& d) -> const Eigen::MatrixXd* {
        auto it = density_idx.find(d);
        if (it == density_idx.end()) return nullptr;
        return &D_set.blocks[it->second];
    };

    // ω > 0 → short-range erfc-attenuated Coulomb (BIPOLE J_SR); else 1/r.
    const libint2::Operator op = (omega > 0.0)
        ? libint2::Operator::erfc_coulomb
        : libint2::Operator::coulomb;
    libint2::Engine prototype(op,
                              shells_ref.max_nprim(), shells_ref.max_l(),
                              1 /*deriv_order*/);
    if (omega > 0.0) prototype.set_params(omega);
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const auto s2a = shell_to_atom(basis, system.unit_cell);
    const std::size_t N_atoms = system.unit_cell.size();
    const std::size_t nshells = shells_ref.size();
    const bool use_output_masks = !output_shell_masks.empty();
    if (use_output_masks && output_shell_masks.size() != n_g) {
        throw std::runtime_error(
            "eri_lattice_gradient_contribution: output_shell_masks must "
            "contain one mask per output cell");
    }
    if (use_output_masks) {
        for (const auto& mask : output_shell_masks) {
            if (mask.size() != nshells * nshells) {
                throw std::runtime_error(
                    "eri_lattice_gradient_contribution: every output shell "
                    "mask must contain nshells*nshells entries");
            }
        }
    }

    const bool j_on = (j_scale != 0.0);
    const bool screened_k = (omega > 0.0);
    const bool direct_exchange_energy = opts.pair_complete_1e || screened_k || exchange_energy_convention;
    const double j_coef = 0.5 * j_scale;
    // Direct exchange energy in build_jk_2e_real_space is
    //   E_x = -alpha/4 sum_g D(g) : K_g,
    //   K_g(mu,nu) = sum_lam,sig D(sig-lam)(la,si)
    //                 (mu_0 la_lam | nu_g si_sig).
    // Differentiate that exact energy contraction for K_SR and for BIPOLE's
    // legacy full-Coulomb energy path when exchange_energy_convention is set.
    // The default full-Coulomb DIRECT_TRUNCATED path keeps the established
    // true-periodic D(lam) * D(sig-g) contraction and symmetry halving.
    const double k_coef_direct_energy = 0.25 * alpha_hf;
    const double k_coef_full = 0.125 * alpha_hf;

    // ---- Cauchy–Schwarz pre-pass ------------------------------------------
    //
    // For the gradient pass we use the same per-cell Schwarz factors
    //
    //   Q[c][s_a, s_b] = √( max_{ij} | ⟨s_a_0 s_b_c | s_a_0 s_b_c⟩_{ij,ij} | )
    //
    // computed with a deriv_order=0 Coulomb engine — the bound
    // applies to the integral magnitude, not to its derivative
    // directly. Plain Schwarz on the gradient is therefore non-rigorous
    // (the derivative can be larger than the integral by a factor
    // proportional to the Gaussian exponent), so we use a *separate
    // tighter threshold* (``opts.schwarz_threshold_forces``, default
    // 1e-14) which is the standard CP2K / CRYSTAL convention.
    //
    // J integral: (μ_0 ν_g | λ_λ σ_σ)
    //   bra pair displacement = c_g
    //   ket pair displacement = c_σ − c_λ
    // K integral: (μ_0 λ_λ | ν_g σ_σ)
    //   bra pair displacement = c_λ
    //   ket pair displacement = c_σ − c_g
    libint2::Engine prototype_q(libint2::Operator::coulomb,
                                shells_ref.max_nprim(), shells_ref.max_l(),
                                0 /*deriv_order — magnitudes only*/);
    const double schwarz_thr = opts.schwarz_threshold_forces;
    const bool screen = (schwarz_thr > 0.0);
    const auto Q = screen
        ? compute_schwarz_factors_per_cell(shells_ref, shells_at, prototype_q)
        : std::vector<std::vector<double>>{};
    const auto Q_max = screen
        ? max_q_per_cell(Q)
        : std::vector<double>{};
    const double D_max = screen ? density_envelope(D_set.blocks) : 0.0;

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N_atoms), 3));

    const int n_g_i = static_cast<int>(n_g);

    // Parallelise over the outer cell axis. Each thread accumulates
    // into its own grad_tls[tid]; we merge after the loop.
    #pragma omp parallel for schedule(dynamic) collapse(1)
    for (int c_g = 0; c_g < n_g_i; ++c_g) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const int c_g_internal = output_internal_positions[c_g];
        const auto& shells_g = shells_at[c_g_internal];
        const Eigen::Vector3i cell_g = output_cells[c_g].index;
        const Eigen::MatrixXd* D_g =
            &D_set.blocks[output_density_positions[c_g]];
        const std::vector<uint8_t>* output_mask = use_output_masks
            ? &output_shell_masks[static_cast<std::size_t>(c_g)]
            : nullptr;

        for (std::size_t c_lam = 0; c_lam < n_c; ++c_lam) {
            const auto& shells_lam = shells_at[c_lam];
            const Eigen::Vector3i cell_lam = cells[c_lam].index;
            const Eigen::MatrixXd* D_lam = block_for_diff(cell_lam);

            for (std::size_t c_sig = 0; c_sig < n_c; ++c_sig) {
                const auto& shells_sig = shells_at[c_sig];
                const Eigen::Vector3i cell_sig = cells[c_sig].index;

                // J density block: D(c_sig - c_lam)
                const Eigen::MatrixXd* D_J =
                    block_for_diff(cell_sig - cell_lam);
                // Full-Coulomb true-periodic K density block: D(c_sig - c_g).
                // Direct-energy K (screened K_SR, or BIPOLE full-K with
                // exchange_energy_convention) uses D_J instead because it
                // differentiates -alpha/4 sum_g D(g):K(g).
                const Eigen::MatrixXd* D_K =
                    block_for_diff(cell_sig - cell_g);
                const bool k_density_available =
                    direct_exchange_energy
                    ? (D_J != nullptr)
                    : (D_lam != nullptr && D_K != nullptr);
                if (!(j_on && D_J) && !(alpha_hf != 0.0 && k_density_available))
                    continue;

                // Schwarz cell-displacement indices for J (ket = σ − λ)
                // and K (ket = σ − g). When the displacement is outside
                // the cell list the corresponding Q is zero by truncation
                // and the bound vanishes → skip the libint call.
                int c_J_idx = -1;
                int c_K_idx = -1;
                if (screen) {
                    auto it_J = internal_idx.find(cell_sig - cell_lam);
                    if (it_J != internal_idx.end()) c_J_idx = it_J->second;
                    if (alpha_hf != 0.0) {
                        auto it_K = internal_idx.find(cell_sig - cell_g);
                        if (it_K != internal_idx.end()) c_K_idx = it_K->second;
                    }
                }

                // Cell-level Schwarz: skip the entire shell-quartet loop
                // when neither J nor K can possibly contribute above the
                // threshold (cuts another 10-100× on real crystals where
                // many cell triples have negligible shell-pair overlap).
                if (screen) {
                    const bool j_possible = j_on && D_J && (c_J_idx >= 0) &&
                        (Q_max[c_g_internal] * Q_max[c_J_idx] * D_max
                            >= schwarz_thr);
                    // Preserve the historical true-periodic low-level
                    // contraction when no extended-domain option is supplied.
                    // That path used J-density availability as part of its
                    // Schwarz prefilter and is pinned by G1a. Padded and
                    // pair-resolved domains instead use the actual K-density
                    // availability required by their energy contraction.
                    const bool k_possible =
                        (legacy_single_domain ? (D_J != nullptr)
                                              : k_density_available) &&
                        (alpha_hf != 0.0) &&
                        (c_K_idx >= 0) &&
                        (Q_max[c_lam] * Q_max[c_K_idx] * D_max *
                            0.5 * std::fabs(alpha_hf) >= schwarz_thr);
                    if (!j_possible && !k_possible) continue;
                }

                for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                    const auto bf1 = shell2bf[s1];
                    const auto n1 = shells_ref[s1].size();
                    const long atom1 = s2a[s1];
                    for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                        if (output_mask &&
                            (*output_mask)[s1 * nshells + s2] == 0)
                            continue;
                        const auto bf2 = shell2bf[s2];
                        const auto n2 = shells_g[s2].size();
                        const long atom2 = s2a[s2];
                        // J bra: (s1_0, s2_g). Schwarz factor.
                        const double q12_J = screen
                            ? Q[c_g_internal][s1 * nshells + s2]
                            : 0.0;
                        for (std::size_t s3 = 0; s3 < shells_lam.size(); ++s3) {
                            const auto bf3 = shell2bf[s3];
                            const auto n3 = shells_lam[s3].size();
                            const long atom3 = s2a[s3];
                            // K bra: (s1_0, s3_λ). Schwarz factor.
                            const double q13_K = (screen && alpha_hf != 0.0)
                                ? Q[c_lam][s1 * nshells + s3]
                                : 0.0;
                            for (std::size_t s4 = 0; s4 < shells_sig.size(); ++s4) {
                                const auto bf4 = shell2bf[s4];
                                const auto n4 = shells_sig[s4].size();
                                const long atom4 = s2a[s4];

                                // ----- J piece -----
                                // Compute ∂(μ_0 ν_{c_g} | λ_{c_lam} σ_{c_sig})/∂R
                                // and contract with J Γ = (1/2) D(c_g)_μν · D(c_sig - c_lam)_λσ.
                                // 12 derivative buffers per quartet (3 × 4 centers).
                                bool do_J = j_on && D_J != nullptr &&
                                    (!opts.pair_complete_1e || pair_products_in_range(
                                        shells_ref[s1], shells_g[s2], shells_lam[s3], shells_sig[s4],
                                        opts.cutoff_bohr, interaction_cutoff));
                                if (do_J && screen) {
                                    if (c_J_idx < 0) {
                                        do_J = false;
                                    } else {
                                        const double q34_J =
                                            Q[c_J_idx][s3 * nshells + s4];
                                        if (q12_J * q34_J * D_max < schwarz_thr)
                                            do_J = false;
                                    }
                                }
                                if (do_J) {
                                    engine.compute(
                                        shells_ref[s1], shells_g[s2],
                                        shells_lam[s3], shells_sig[s4]);

                                    const long centers_J[4] = {
                                        atom1, atom2, atom3, atom4};
                                    // Libint clears only the first target when
                                    // the whole quartet is screened. The other
                                    // pointers can retain the previous quartet.
                                    for (int icenter = 0; buf[0] && icenter < 4; ++icenter) {
                                        for (int d = 0; d < 3; ++d) {
                                            const double* block =
                                                buf[icenter * 3 + d];
                                            if (!block) continue;
                                            double acc = 0.0;
                                            for (std::size_t i = 0; i < n1; ++i) {
                                                const auto mu = bf1 + i;
                                                for (std::size_t j = 0; j < n2; ++j) {
                                                    const auto nu = bf2 + j;
                                                    const double d_g_mn = (*D_g)(mu, nu);
                                                    for (std::size_t k = 0; k < n3; ++k) {
                                                        const auto lam = bf3 + k;
                                                        for (std::size_t l = 0; l < n4; ++l) {
                                                            const auto sig = bf4 + l;
                                                            const double gamma_J =
                                                                j_coef * d_g_mn
                                                                * (*D_J)(lam, sig);
                                                            acc += gamma_J * block[
                                                                ((i * n2 + j) * n3 + k) * n4 + l];
                                                        }
                                                    }
                                                }
                                            }
                                            grad_local(centers_J[icenter], d) += acc;
                                        }
                                    }
                                }

                                // ----- K piece -----
                                // Periodic K cannot reuse the J integral via
                                // dummy-index + cell relabeling (the molecular
                                // single-integral trick fails because swapping
                                // c_g ↔ c_lam in the K integral yields a
                                // different periodic integral, not the same
                                // one). So we compute the K integral with
                                // permuted shell args:
                                //   (μ_0 λ_{c_lam} | ν_{c_g} σ_{c_sig})
                                // and contract either with the direct
                                // real-space exchange-energy density
                                //   K Γ = -(α_HF/4)
                                //          D(c_g)_μν · D(c_sig - c_lam)_λσ,
                                // or, for full Coulomb, with the legacy
                                // true-periodic density slots
                                //   K Γ = -(α_HF/8)
                                //         D(c_lam)_μλ · D(c_sig - c_g)_νσ.
                                //
                                // Atom mapping: with shell ordering
                                // (s1, s3@c_lam, s2@c_g, s4@c_sig), libint's
                                // 4 derivative-center buffers correspond to
                                // (atom1, atom3, atom2, atom4) in that order.
                                // The integral block strides are
                                // [μ][λ][ν][σ] = (n1, n3, n2, n4).
                                bool do_K = alpha_hf != 0.0 && k_density_available &&
                                    (!opts.pair_complete_1e || pair_products_in_range(
                                        shells_ref[s1], shells_lam[s3], shells_g[s2], shells_sig[s4],
                                        opts.cutoff_bohr, interaction_cutoff));
                                if (do_K && screen) {
                                    if (c_K_idx < 0) {
                                        do_K = false;
                                    } else {
                                        const double q24_K =
                                            Q[c_K_idx][s2 * nshells + s4];
                                        const double k_scale =
                                            0.25 * std::fabs(alpha_hf);
                                        if (q13_K * q24_K * D_max * k_scale
                                            < schwarz_thr)
                                            do_K = false;
                                    }
                                }
                                if (do_K) {
                                    engine.compute(
                                        shells_ref[s1], shells_lam[s3],
                                        shells_g[s2], shells_sig[s4]);

                                    const long centers_K[4] = {
                                        atom1, atom3, atom2, atom4};
                                    // Check the whole-quartet sentinel for K
                                    // independently of the preceding J build.
                                    for (int icenter = 0; buf[0] && icenter < 4; ++icenter) {
                                        for (int d = 0; d < 3; ++d) {
                                            const double* block =
                                                buf[icenter * 3 + d];
                                            if (!block) continue;
                                            double acc = 0.0;
                                            for (std::size_t i = 0; i < n1; ++i) {
                                                const auto mu = bf1 + i;
                                                for (std::size_t k = 0; k < n3; ++k) {
                                                    const auto lam = bf3 + k;
                                                    for (std::size_t j = 0; j < n2; ++j) {
                                                        const auto nu = bf2 + j;
                                                        const double d_left =
                                                            direct_exchange_energy
                                                            ? (*D_g)(mu, nu)
                                                            : (*D_lam)(mu, lam);
                                                        for (std::size_t l = 0; l < n4; ++l) {
                                                            const auto sig = bf4 + l;
                                                            const double d_right =
                                                                direct_exchange_energy
                                                                ? (*D_J)(lam, sig)
                                                                : (*D_K)(nu, sig);
                                                            const double k_coef =
                                                                direct_exchange_energy
                                                                ? k_coef_direct_energy
                                                                : k_coef_full;
                                                            const double gamma_K =
                                                                -k_coef * d_left
                                                                * d_right;
                                                            acc += gamma_K * block[
                                                                ((i * n3 + k) * n2 + j) * n4 + l];
                                                        }
                                                    }
                                                }
                                            }
                                            grad_local(centers_K[icenter], d) += acc;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N_atoms), 3);
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

}  // namespace vibeqc
