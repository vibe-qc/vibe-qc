// Pair-distance-correct image selection for the semiempirical periodic
// lattice sums (issue #316).
//
// A semiempirical two-centre lattice-sum term - overlap / H0 image
// blocks, SCC gamma, AES multipole kernels, repulsive and core-core
// pairs - depends on the lattice translation g ONLY through the physical
// pair separation |R_A - R_B - g|.  An interaction cutoff therefore
// bounds the PAIR distance, not the translation length:
//
//     kept terms = { (A, B, g) : |R_A - R_B - g| <= R_cut }.        (*)
//
// Selecting cells by |g| <= R_cut instead (what direct_lattice_cells
// alone provides) breaks (*) in both directions: a Gamma supercell wider
// than R_cut keeps no image at all and silently degrades to a
// free-boundary cluster (issue #316: MgO conventional n>=2 at the
// 15-bohr default, Gamma-ladder error up to +686 uHa/cell healing as a
// ~1/n surface term), and even a narrower cell drops corner pairs whose
// separation is inside R_cut while their translation is not (falsifier-B
// measured 19.3-153.1 uHa on n=1 MgO Gamma cells labelled converged).
//
// Literature basis for (*):
//
//  * Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001),
//    doi:10.1002/1096-987X(20010115)22:1<89::AID-JCC9>3.0.CO;2-7,
//    pp. 90-91 (Eqs. 2-5 and "Implementation in MSINDO"): the cyclic
//    cluster interaction region is constructed symmetrically AROUND EACH
//    ATOM - an interaction is admitted by the interatomic distance
//    r(X_i Y_j), with the translation chosen to bring the pair into
//    range; earlier translation-based implementations are called out for
//    dropping exactly these boundary interactions.
//  * Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
//    doi:10.1002/jcc.23550: the same atom-centred interaction-region
//    semantics at the Hartree-Fock CCM level.
//  * Sharma & Beylkin, J. Chem. Theory Comput. 17, 3916 (2021),
//    doi:10.1021/acs.jctc.0c01195, Eqs. 17-18: the lattice-dependent
//    factor of a two-centre sum depends on the translation only through
//    T(P) ~ |A - B - P|^2, so the contributing translations form a ball
//    centred on the intra-cell offset, NOT on the origin.  This is the
//    tree's canonical statement of the same fact for the Gaussian stack
//    (cpp/include/vibeqc/lattice_pair_cells.hpp).
//
// Implementation pattern (same as lattice_pair_cells.hpp, per-atom
// instead of per-shell): enumerate a superset of translations by padding
// the radius with the largest intra-cell pair offset,
//
//     |g| <= R_cut + max_{A,B} |R_A - R_B|                          (**)
//
// (triangle inequality: any g satisfying (*) for some pair satisfies
// (**)), then filter every (A, B, g) term on the true pair separation.
// The resulting term set is exactly (*): translation invariant by
// construction, and it folds term-for-term onto the matched primitive
// k-point sum, which is what makes the band-folding identity
// Gamma-supercell(n^3) == n^3 Monkhorst-Pack exact.
//
// Cells with no in-range pair are pruned from the returned list.  This
// keeps n_cells an honest count of actual image coupling: a genuinely
// isolated molecular-limit box (tiny extent, huge cell) still yields the
// origin-only list, so the issue-#316 gen-1 fail-closed guard
// (require_nonzero_lattice_image) and the explicit gamma_only_0
// molecular mode keep their exact semantics.
//
// The Gaussian/ab-initio stack's image selection is deliberately NOT
// changed here - direct_lattice_cells keeps its plain |g|-ball contract
// for every other caller (see the issue-#316 caller audit).

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/periodic.hpp"

namespace vibeqc {
namespace semiempirical {

// Largest intra-cell atom-pair separation |R_A - R_B| in bohr: how far
// the ball of contributing translations can be displaced from the
// origin.  Zero for empty or single-atom cells.
inline double max_atom_pair_offset(const PeriodicSystem& system) {
    const auto& atoms = system.unit_cell;
    double d2_max = 0.0;
    for (std::size_t a = 0; a + 1 < atoms.size(); ++a) {
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            const double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            const double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            const double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            d2_max = std::max(d2_max, dx * dx + dy * dy + dz * dz);
        }
    }
    return std::sqrt(d2_max);
}

// Is the (A, B, g) pair term within the interaction cutoff?  ``b`` is
// the untranslated ket atom; the term couples A in the home cell to B
// translated by +g, matching the S_munu(g) = <chi_mu(r)|chi_nu(r-g)>
// convention used throughout the semiempirical lattice sums.
inline bool atom_pair_in_cutoff(const Atom& a, const Atom& b,
                                const Eigen::Vector3d& g,
                                double cutoff_bohr) {
    const double dx = a.xyz[0] - b.xyz[0] - g[0];
    const double dy = a.xyz[1] - b.xyz[1] - g[1];
    const double dz = a.xyz[2] - b.xyz[2] - g[2];
    return dx * dx + dy * dy + dz * dz <= cutoff_bohr * cutoff_bohr;
}

// Lattice translations carrying at least one atom-pair interaction
// within ``cutoff_bohr`` under rule (*): the padded enumeration (**),
// pruned to cells with an in-range pair.  The zero cell is always first
// (ascending-|g| stable order is inherited from direct_lattice_cells),
// so the ``gamma_only_0`` molecular mode's ``cells[0]`` contract holds.
inline std::vector<LatticeCell> atom_pair_interaction_cells(
    const PeriodicSystem& system, double cutoff_bohr) {
    const auto padded = direct_lattice_cells(
        system, cutoff_bohr + max_atom_pair_offset(system));
    const auto& atoms = system.unit_cell;

    std::vector<LatticeCell> cells;
    cells.reserve(padded.size());
    for (const auto& cell : padded) {
        bool has_pair = atoms.empty()
            && (cell.index.array() == 0).all();  // keep g=0 for empty cells
        for (std::size_t a = 0; !has_pair && a < atoms.size(); ++a) {
            for (std::size_t b = 0; b < atoms.size(); ++b) {
                if (atom_pair_in_cutoff(atoms[a], atoms[b], cell.r_cart,
                                        cutoff_bohr)) {
                    has_pair = true;
                    break;
                }
            }
        }
        if (has_pair) cells.push_back(cell);
    }
    return cells;
}

// Zero every AO-pair entry of per-cell lattice blocks whose atom-pair
// separation exceeds the cutoff.  Applying rule (*) once at block-build
// time makes every downstream consumer - Gamma sums, Bloch phases,
// Wolfsberg-Helmholtz / EHT H0 built proportional to S(g), and the
// M(g)-against-dS(g) derivative contractions - inherit the filtered term
// set.  The filter is symmetric under (mu <-> nu, g <-> -g), so
// Hermiticity of H(k)/S(k) is preserved.
inline void mask_blocks_beyond_pair_cutoff(
    std::vector<Eigen::MatrixXd>& blocks,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    const std::vector<Atom>& atoms,
    double cutoff_bohr) {
    const double cutoff_sq = cutoff_bohr * cutoff_bohr;
    for (std::size_t ci = 0; ci < cells.size() && ci < blocks.size(); ++ci) {
        const Eigen::Vector3d& g = cells[ci].r_cart;
        auto& block = blocks[ci];
        for (Eigen::Index mu = 0; mu < block.rows(); ++mu) {
            const auto& ra = atoms[ao_atom[static_cast<std::size_t>(mu)]];
            for (Eigen::Index nu = 0; nu < block.cols(); ++nu) {
                const auto& rb = atoms[ao_atom[static_cast<std::size_t>(nu)]];
                const double dx = ra.xyz[0] - rb.xyz[0] - g[0];
                const double dy = ra.xyz[1] - rb.xyz[1] - g[1];
                const double dz = ra.xyz[2] - rb.xyz[2] - g[2];
                if (dx * dx + dy * dy + dz * dz > cutoff_sq) {
                    block(mu, nu) = 0.0;
                }
            }
        }
    }
}

}  // namespace semiempirical
}  // namespace vibeqc
