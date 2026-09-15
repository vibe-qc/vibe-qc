#include "vibeqc/periodic_cosx_jk_builder.hpp"

#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_one_center.hpp"
#include "vibeqc/df.hpp"
#include "vibeqc/grid_batch.hpp"
#include "vibeqc/schwarz.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace vibeqc {

namespace {

class PeriodicGammaCOSXJKBuilder final : public JKBuilder {
public:
    PeriodicGammaCOSXJKBuilder(const BasisSet& basis,
                               const BasisSet& aux_basis,
                               const PeriodicSystem& system,
                               const LatticeSumOptions& opts,
                               const GridOptions& cosx_grid_opts)
        : df_(basis, aux_basis),
          basis_(&basis),
          lattice_(system.lattice),
          // Image cells for the M3a COSX-K lattice summation: the
          // same truncated direct-lattice list the rest of the
          // periodic stack uses. For vacuum-padded molecular-limit
          // boxes the list collapses to the home cell and
          // compute_cosx_k keeps the single-pass batched kernel.
          image_cells_(direct_lattice_cells(system, opts.cutoff_bohr)),
          // Home-only J/K uses molecular quadrature (#707). Applying a
          // periodic partition there would remove density from a model
          // that has no image electrons to supply the missing weight.
          cosx_grid_(image_cells_.size() == 1
              ? build_grid(system.unit_cell_molecule(), cosx_grid_opts)
              : build_periodic_cosx_grid(system, cosx_grid_opts)),
          cosx_q_(build_cosx_q(basis, cosx_grid_)),
          cosx_schwarz_(build_cosx_schwarz(basis)),
          shell_cutoffs_(
              compute_shell_radial_cutoffs(basis.libint(), 1e-12)),
          grid_batches_(
              build_grid_batches(basis, cosx_grid_, 4096,
                                 shell_cutoffs_, false)),
          boys_table_(build_boys_table(basis.libint().max_l())),
          pp_cache_(build_primitive_pair_cache(basis.libint())),
          one_center_(basis) {}

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        return df_.build_J(D);
    }
    int last_df_transform_workers_used() const noexcept override {
        return df_.last_df_transform_workers_used();
    }
    int last_df_pack_workers_used() const noexcept override {
        return df_.last_df_pack_workers_used();
    }
    int last_df_unpack_workers_used() const noexcept override {
        return df_.last_df_unpack_workers_used();
    }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        Eigen::MatrixXd K = compute_cosx_k(
            *basis_, D, cosx_grid_,
            cosx_q_, cosx_schwarz_, shell_cutoffs_,
            &grid_batches_,
            &boys_table_, &pp_cache_,
            &lattice_, &image_cells_);
        return K;
    }
    void apply_one_center_correction(
        Eigen::Ref<Eigen::MatrixXd> K,
        const Eigen::MatrixXd& D) const override {
        // Match the molecular COSX lifecycle: iterate on the smooth
        // seminumerical K surface and apply the analytic one-center
        // replacement only to the post-convergence Fock. Feeding the
        // quadrature residual back into every SCF iteration prevents tight
        // convergence for some grids.
        K.noalias() += one_center_.apply(
            D, grid_batches_, boys_table_, pp_cache_, shell_cutoffs_);
    }
    bool has_post_scf_exchange_correction() const noexcept override {
        return true;
    }

private:
    DensityFitting df_;
    const BasisSet* basis_;
    Eigen::Matrix3d lattice_;
    std::vector<LatticeCell> image_cells_;
    Grid cosx_grid_;
    Eigen::MatrixXd cosx_q_;
    Eigen::MatrixXd cosx_schwarz_;
    std::vector<double> shell_cutoffs_;
    GridBatches grid_batches_;
    BoysTable boys_table_;
    PrimitivePairCache pp_cache_;
    OneCenterCorrection one_center_;

    static Grid build_periodic_cosx_grid(
        const PeriodicSystem& system,
        const GridOptions& opts) {
        return build_periodic_point_grid(system, 10.0, opts);
    }

};

// -----------------------------------------------------------------------
// Tight-cell builder: periodic GDF J from precomputed Lpq + COSX-K.
//
// Same COSX setup as PeriodicGammaCOSXJKBuilder, but the Coulomb J
// is built from the periodic-correct Lpq tensor (compcell or RSGDF
// GDF factorization, computed once in Python) instead of the molecular
// DensityFitting.  The COSX K sums exchange over the truncated image-
// cell list (M3a single-lattice-sum model; Γ-folded GDF exchange
// parity is still open — handovers/HANDOVER_RIJCOSX_M3A.md).  No exxdiv='ewald'
// Madelung shift is applied — the COSX K is entirely real-space
// (direct-truncated lattice sum) and the GDF J from Lpq already
// handles the G=0 gauge through the compcell/RSGDF construction.
// -----------------------------------------------------------------------

class PeriodicTightCellCOSXJKBuilder final : public JKBuilder {
public:
    PeriodicTightCellCOSXJKBuilder(
        const BasisSet& basis,
        std::vector<Eigen::MatrixXd> lpq,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const GridOptions& cosx_grid_opts)
        : lpq_(std::move(lpq)),
          n_kept_(lpq_.size()),
          n_orb_(basis.nbasis()),
          basis_(&basis),
          lattice_(system.lattice),
          // Image cells for the M3a COSX-K lattice summation (same
          // truncated direct-lattice list as the one-electron lattice
          // sums). On tight cells this adds the home-bra → image-ket
          // exchange class on top of the home-cell-only M3 kernel
          // (H-chain anchor, D = I: ||K|| 1.74 → 3.94; Γ-folded GDF
          // parity needs more — handovers/HANDOVER_RIJCOSX_M3A.md).
          image_cells_(direct_lattice_cells(system, opts.cutoff_bohr)),
          cosx_grid_(build_periodic_cosx_grid(system, cosx_grid_opts)),
          cosx_q_(build_cosx_q(basis, cosx_grid_)),
          cosx_schwarz_(build_cosx_schwarz(basis)),
          shell_cutoffs_(
              compute_shell_radial_cutoffs(basis.libint(), 1e-12)),
          grid_batches_(
              build_grid_batches(basis, cosx_grid_, 4096,
                                 shell_cutoffs_, false)),
          boys_table_(build_boys_table(basis.libint().max_l())),
          pp_cache_(build_primitive_pair_cache(basis.libint())),
          one_center_(basis) {
        if (lpq_.empty()) {
            throw std::runtime_error(
                "PeriodicTightCellCOSXJKBuilder: Lpq tensor is empty");
        }
        for (std::size_t p = 0; p < lpq_.size(); ++p) {
            if (lpq_[p].rows() != static_cast<Eigen::Index>(n_orb_) ||
                lpq_[p].cols() != static_cast<Eigen::Index>(n_orb_)) {
                throw std::runtime_error(
                    "PeriodicTightCellCOSXJKBuilder: Lpq[" +
                    std::to_string(p) + "] has wrong shape (" +
                    std::to_string(lpq_[p].rows()) + "x" +
                    std::to_string(lpq_[p].cols()) + ", expected " +
                    std::to_string(n_orb_) + "x" + std::to_string(n_orb_) +
                    ")");
            }
        }
    }

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        // γ_P = Σ_{μν} Lpq[P, μ, ν] · D[μ, ν]  (Frobenius inner product)
        // J    = Σ_P γ_P · Lpq[P]
        const Eigen::Index n = static_cast<Eigen::Index>(n_orb_);
        const Eigen::Index nk = static_cast<Eigen::Index>(n_kept_);
        Eigen::VectorXd gamma(nk);
        for (Eigen::Index P = 0; P < nk; ++P) {
            gamma(P) = (lpq_[static_cast<std::size_t>(P)].array()
                        * D.array()).sum();
        }
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index P = 0; P < nk; ++P) {
            J.noalias() += gamma(P) * lpq_[static_cast<std::size_t>(P)];
        }
        return J;
    }

    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        return compute_cosx_k(
            *basis_, D, cosx_grid_,
            cosx_q_, cosx_schwarz_, shell_cutoffs_,
            &grid_batches_,
            &boys_table_, &pp_cache_,
            &lattice_, &image_cells_);
    }

    void apply_one_center_correction(
        Eigen::Ref<Eigen::MatrixXd> K,
        const Eigen::MatrixXd& D) const override {
        // Keep the analytic replacement out of the iterated K for the same
        // smooth-SCF contract as the molecular COSX builder above.
        K.noalias() += one_center_.apply(
            D, grid_batches_, boys_table_, pp_cache_, shell_cutoffs_);
    }
    bool has_post_scf_exchange_correction() const noexcept override {
        return true;
    }

private:
    std::vector<Eigen::MatrixXd> lpq_;
    std::size_t n_kept_;
    std::size_t n_orb_;
    const BasisSet* basis_;
    Eigen::Matrix3d lattice_;
    std::vector<LatticeCell> image_cells_;
    Grid cosx_grid_;
    Eigen::MatrixXd cosx_q_;
    Eigen::MatrixXd cosx_schwarz_;
    std::vector<double> shell_cutoffs_;
    GridBatches grid_batches_;
    BoysTable boys_table_;
    PrimitivePairCache pp_cache_;
    OneCenterCorrection one_center_;

    static Grid build_periodic_cosx_grid(
        const PeriodicSystem& system,
        const GridOptions& opts) {
        return build_periodic_point_grid(system, 10.0, opts);
    }

};

}  // namespace

std::unique_ptr<JKBuilder> make_periodic_gamma_cosx_jk_builder(
    const BasisSet& basis,
    const BasisSet& aux_basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const GridOptions& cosx_grid_opts) {
    return std::make_unique<PeriodicGammaCOSXJKBuilder>(
        basis, aux_basis, system, opts, cosx_grid_opts);
}

std::unique_ptr<JKBuilder> make_periodic_tight_cosx_jk_builder(
    const BasisSet& basis,
    const std::vector<Eigen::MatrixXd>& lpq,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const GridOptions& cosx_grid_opts) {
    return std::make_unique<PeriodicTightCellCOSXJKBuilder>(
        basis, lpq, system, opts, cosx_grid_opts);
}

}  // namespace vibeqc
