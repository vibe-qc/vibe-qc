#include "vibeqc/lattice_integrals.hpp"

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/engine.h>
#include <array>
#include <cmath>
#include <cstdint>
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
    double pair_cutoff = 0.0);
LatticeMatrixSet compute_1e_lattice_matrix_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    libint2::Operator op,
    const std::vector<std::pair<double, std::array<double, 3>>>* nuclei,
    double pair_cutoff) {
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
    return compute_1e_lattice_matrix_explicit(
        basis, system, cells, op, nuclei, opts.cutoff_bohr);
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

LatticeMatrixSet compute_nuclear_erfc_lattice(const BasisSet& basis,
                                              const PeriodicSystem& system,
                                              double omega,
                                              const LatticeSumOptions& opts) {
    ensure_libint_initialized();
    require_plain_ball_for_ewald_nuclear(opts, "compute_nuclear_erfc_lattice");

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    const auto nuclei = build_periodic_nuclear_charges(system, opts);

    // libint's erfc_nuclear takes a tuple (ω, point-charge list). ω is the
    // erfc attenuation parameter: the kernel is erfc(ω · r_C) / r_C, which
    // reduces to 1/r_C at ω=0 and vanishes as ω → ∞.
    libint2::Engine prototype(libint2::Operator::erfc_nuclear,
                              shells_ref.max_nprim(),
                              shells_ref.max_l(), 0);
    using erfc_params =
        libint2::operator_traits<libint2::Operator::erfc_nuclear>::oper_params_type;
    prototype.set_params(erfc_params{omega, nuclei});
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();

    LatticeMatrixSet set;
    set.nbf = nbf;
    // Plain |g| ball, NOT the pair-complete enumeration the overlap /
    // kinetic / direct-nuclear builders use. This is the erfc half of an
    // Ewald split whose erf half is a reciprocal AO-pair FT or a grid
    // quadrature built on ``direct_lattice_cells`` at the same cutoff
    // (compute_nuclear_lattice_ewald, compute_v_ne_ewald_3d_ft_*,
    // compute_vsap_lattice). Both halves must enumerate the same term set
    // or the split stops reconstructing V; widening only this one moved
    // the FT V_ne analytic-vs-FD gradient by 2.2e-05 Ha/bohr. Callers that
    // pair these blocks with a one-electron matrix index this shorter
    // list's prefix (direct_lattice_cells stable-sorts by |r|).
    set.cells = direct_lattice_cells(system, opts.cutoff_bohr);
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
            // NO pair filter here, deliberately. This is one half of an
            // Ewald split: the erf complement is a grid quadrature
            // (compute_nuclear_lattice_ewald) or a reciprocal AO-pair FT
            // (compute_v_ne_ewald_3d_ft_*) that contracts every pair of
            // these cells densely. Filtering only the erfc half would drop
            // terms the erf half still carries, and the split would no
            // longer reconstruct V. The cell list IS pair-complete, so both
            // halves see the same, larger, image set than before.
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();

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

// Construct a BasisSet whose shells live at translated atomic positions.
// We reuse BasisSet's own constructor — libint will reassemble the shell
// list from the shifted Molecule exactly as it did for the original.
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
    return BasisSet(mol, ref.name());
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
    require_plain_ball_for_ewald_nuclear(opts, "compute_nuclear_lattice_ewald");

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
    double pair_cutoff) {
    if (charges.empty()) return;
    libint2::Engine prototype(libint2::Operator::erfc_nuclear,
                              shells_ref.max_nprim(), shells_ref.max_l(), 0);
    using erfc_params =
        libint2::operator_traits<libint2::Operator::erfc_nuclear>::oper_params_type;
    prototype.set_params(erfc_params{omega, charges});
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const int n_cells = static_cast<int>(set.cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto shells_g = shift_shells(shells_ref, set.cells[c].r_cart);
        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            // No pair filter: same Ewald-split argument as
            // compute_nuclear_erfc_lattice. ``pair_cutoff`` is accepted so
            // the signature records which cutoff the cell list came from.
            (void)pair_cutoff;
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
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
    require_plain_ball_for_ewald_nuclear(opts, "compute_vsap_lattice");
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
    // Plain |g| ball, for the same Ewald-split reason as
    // compute_nuclear_erfc_lattice: the long-range erf piece below is a
    // grid quadrature over this cell list.
    set.cells = direct_lattice_cells(system, opts.cutoff_bohr);
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
    accumulate_erfc_lattice(set, shells_ref, eta, eta_q,
                            opts.pair_complete_1e ? opts.cutoff_bohr : 0.0);

    // (b) per-(element, primitive) erfc with charges +c_i  → −Σ c_i erfc(ω_i)/r.
    //     Atoms grouped by element so one erfc call covers all of them.
    std::map<int, std::vector<std::size_t>> atoms_by_z;
    for (std::size_t a = 0; a < system.unit_cell.size(); ++a) {
        atoms_by_z[system.unit_cell[a].Z].push_back(a);
    }
    for (const auto& [z, idx] : atoms_by_z) {
        const SAPExpansion& e = sap_table.at(z);
        for (std::size_t i = 0; i < e.alphas.size(); ++i) {
            accumulate_erfc_lattice(set, shells_ref, std::sqrt(e.alphas[i]),
                                    image_charges(idx, e.coeffs[i]),
                                    opts.pair_complete_1e ? opts.cutoff_bohr : 0.0);
        }
    }

    // ---- Long-range, smooth erf(η r)/r via grid quadrature ----
    // v_long(r) = +Σ_A Q_A erf(η r)/r (reciprocal lattice sum);
    // ewald_point_charge_potential returns exactly that unsigned long-range
    // potential of the effective charges Q_A.
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
    return compute_1e_lattice_matrix_explicit(
        basis, system, cells, libint2::Operator::nuclear, &nuclei,
        opts.pair_complete_1e ? opts.cutoff_bohr : 0.0);
}

}  // namespace vibeqc
