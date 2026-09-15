// Direct-lattice iteration and storage of real-space periodic matrices.
//
// A LatticeCell is an integer shift (n1, n2, n3) together with its Cartesian
// realization n1·a1 + n2·a2 + n3·a3 in bohr. For dim < 3 the unused index
// components are always zero.
//
// Two related containers:
//   - std::vector<LatticeCell>: the list of cells considered in a lattice
//     sum (subject to a cutoff).
//   - LatticeMatrixSet: a real-space periodic matrix M(g), stored as one
//     dense block per lattice cell. Used for S(g), T(g), V(g), H(g), F(g),
//     and (later) the real-space density matrix P(g).

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "periodic.hpp"

namespace vibeqc {

struct LatticeCell {
    Eigen::Vector3i index = Eigen::Vector3i::Zero();  // (n1, n2, n3)
    Eigen::Vector3d r_cart = Eigen::Vector3d::Zero(); // Cartesian, bohr
};

// Enumerate all integer lattice cells whose Cartesian shift has Euclidean
// norm ≤ cutoff_bohr. The returned list always includes the zero cell and
// is unordered with respect to the index, but is grouped by ascending |r|.
//
// For dim=1 only n1 varies; for dim=2 (n1,n2); for dim=3 all three.
std::vector<LatticeCell> direct_lattice_cells(const PeriodicSystem& system,
                                              double cutoff_bohr);

// Fail closed when a caller that promises periodic image coupling receives
// only the home cell.  ``direct_lattice_cells`` deliberately remains a plain
// geometric enumerator: an origin-only result is valid for explicitly
// molecular ``gamma_only_0`` routes, but silently treating that result as a
// periodic Gamma calculation turns a wide supercell into a free-boundary
// cluster.  Periodic drivers call this helper unless the user selected such an
// explicit molecular mode.
void require_nonzero_lattice_image(const std::vector<LatticeCell>& cells,
                                   double cutoff_bohr,
                                   const char* context);

// Coulomb lattice-sum method. Only DIRECT_TRUNCATED is implemented in
// Phase 12a–12b; later phases fill in the rest.
enum class CoulombMethod {
    DIRECT_TRUNCATED = 0,
    EWALD_3D         = 1,
    SLAB_EWALD_2D    = 2,
    NEUTRALIZED_1D   = 3,
};

struct LatticeSumOptions {
    // Cutoff for the μν overlap / kinetic / Fock lattice sums (bohr).
    // Cells with |g| > cutoff_bohr are excluded from the real-space sum.
    double cutoff_bohr = 15.0;

    // Separate, typically larger cutoff for the nuclear-attraction lattice
    // sum — the potential of a unit-cell of point charges falls off as
    // 1/r, not with the basis-function overlap.
    double nuclear_cutoff_bohr = 25.0;

    // Reach of the periodic Becke partition of the DFT grid the caller built
    // (`build_periodic_becke_grid`'s ``image_radius_bohr``). build_xc_periodic
    // screens the cross-cell (bra-image) density sum by THIS radius, not
    // cutoff_bohr: the cross-cell density is only correctly normalised over a
    // cell up to the partition reach (image atoms beyond it are not in the
    // partition, so replicating their density over-counts on the resulting
    // molecular grid — the dilute / box>image_radius failure mode). 0.0 means
    // "unset" and falls back to cutoff_bohr (the molecular / single-cell grid
    // case where they coincide). Drivers set it to the grid's image_radius.
    double becke_image_radius_bohr = 0.0;

    // How to treat the infinite Coulomb series in the Fock build. Phase 12a
    // does not consume this field but it is stored so calling code can
    // specify its intent now and have it forwarded to later phases.
    CoulombMethod coulomb_method = CoulombMethod::DIRECT_TRUNCATED;

    // Gaussian split parameter for the rigorous 2D slab Ewald blocks. The
    // Gamma slab SCF uses one shared value for E_nn, V_ne, and J so the
    // separately net-charged block gauges cancel in the neutral total.
    double slab_ewald_alpha = 0.4;

    // Future screening thresholds (placeholders — unused in 12a).
    double screening_overlap_threshold = 0.0;
    double screening_exchange_threshold = 0.0;

    // Schwarz screening threshold for the periodic 2-e Fock build
    // (build_fock_2e_real_space). For each shell quartet × cell triple
    // we compute the Schwarz bound
    //   |⟨μ ν_g | λ_λ σ_σ⟩|  ≤  √|⟨μν|μν⟩_g| · √|⟨λσ|λσ⟩_{σ-λ}|
    // and skip the libint quartet call if it falls below this. Set to
    // 0.0 to disable (legacy behavior pre-v0.5.3 — extremely slow on
    // realistic k-meshes; the unscreened cost is O(n_c³ · n_shells⁴)).
    // Default 1e-12 Ha matches PySCF's pbc.scf default and stays well
    // inside production-quality SCF accuracy.
    double schwarz_threshold = 1.0e-12;

    // Schwarz screening threshold for the periodic 2-e Fock derivative
    // pass (eri_lattice_gradient_contribution). Plain Cauchy–Schwarz on
    // the gradient is non-rigorous (the bound applies to the *integral*
    // not its derivative; the derivative magnitude can be larger by a
    // Gaussian-exponent factor), so the standard fix is a tighter
    // threshold than the energy side — typically 100× tighter, matching
    // CP2K's ``EPS_SCHWARZ_FORCES`` default of 1e-9 (when EPS_SCHWARZ is
    // 1e-7) or, scaled to vibe-qc's tighter energy default,
    // ``schwarz_threshold_forces = 1e-14`` for ``schwarz_threshold = 1e-12``.
    // Set to 0.0 to disable (very slow on real crystals).
    double schwarz_threshold_forces = 1.0e-14;

    // Distance-aware screening for erfc J/K. Charge-pair Schwarz bounds
    // (Sun 2023, Eq. 51, doi:10.1063/5.0155815) include angular polynomials
    // and absolute contraction coefficients through positive Gaussian
    // envelopes. Positive product envelopes also retain overlap decay.
    // The sparse lattice searches enclose all terms that pass
    // the same quartet predicate. The former QQR attenuation omitted
    // angular tails (#755). False retains ordinary Schwarz screening for
    // independent numerical comparisons. Only affects omega > 0 builds;
    // the production padded-image policy enables it.
    bool sr_range_screening = false;
    // GitLab #429, stage 2 (Family B). When set, the cutoff-driven
    // one-electron builders (compute_{overlap,kinetic,nuclear}_lattice and
    // the erfc / vsap nuclear sums) enumerate the translation-invariant term
    // set {(mu, nu, g) : |O_mu - O_nu - g| <= cutoff_bohr} -- the cell list
    // padded by the largest intra-cell shell offset, every pair filtered on
    // its physical separation (lattice_pair_cells.hpp; Sharma & Beylkin,
    // JCTC 17, 3916 (2021), Eqs. 17-18) -- and the overlap / kinetic /
    // nuclear gradient partners differentiate that same set. Off (the
    // default) keeps the |g| <= cutoff_bohr ball, which is NOT translation
    // invariant (LiH rocksalt / def2-svp moves by 7.2e-01 in S at the
    // shipped cutoff) but is what every consumer that pairs a one-electron
    // cell list index-for-index with the two-electron ball expects. The
    // default flips per consumer family once each family's pins are
    // measured; see handovers/HANDOVER_OPEN_BUGS_V015.md.
    bool pair_complete_1e = false;
    // Physical ERI product-midpoint radius when pair completion is enabled.
    // Zero uses cutoff_bohr. Padding changes this radius, not AO-pair support.
    double eri_interaction_cutoff_bohr = 0.0;

    // Generate only potentially contributing cell pairs before the erfc
    // traversal (#21). Preserves the existing screening decisions and sum
    // order. False retains exhaustive enumeration for numerical comparisons.
    bool sr_sparse_traversal = true;
};

// Real-space periodic matrix: one nbf × nbf block per lattice cell.
struct LatticeMatrixSet {
    int nbf = 0;
    std::vector<LatticeCell> cells;         // cells[i].r_cart aligned with
    std::vector<Eigen::MatrixXd> blocks;    // blocks[i] = M(g_i), nbf × nbf

    std::size_t size() const noexcept { return cells.size(); }
};

}  // namespace vibeqc
