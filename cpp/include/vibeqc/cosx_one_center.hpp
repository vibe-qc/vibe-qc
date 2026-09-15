#pragma once

#include <Eigen/Dense>

#include <vector>

#include "basis.hpp"
#include "cosx_kernel.hpp"
#include "grid_batch.hpp"

namespace vibeqc {

class OneCenterCorrection {
public:
    // ``omega`` (default 0) selects the Coulomb kernel of BOTH sides of
    // the replacement: 0 uses the full 1/r kernel (molecular COSX and
    // the Gamma periodic builders); ``omega > 0`` uses the
    // erfc(omega*r)/r short-range kernel (libint
    // ``Operator::erfc_coulomb`` for the analytic same-atom ERIs, the
    // M3b-4a attenuated seeding for the subtracted quadrature block) —
    // the variant the range-separated multi-k COSX engine needs, where
    // only the real-space SR part carries same-atom cusp quadrature
    // error (the reciprocal-space LR-erf complement is smooth).
    explicit OneCenterCorrection(const BasisSet& basis,
                                 double omega = 0.0);

    Eigen::MatrixXd apply(const Eigen::MatrixXd& D,
                          const GridBatches& grid_batches,
                          const BoysTable& boys,
                          const PrimitivePairCache& pp_cache,
                          const std::vector<double>& shell_cutoffs) const;

    std::size_t n_atoms() const noexcept { return atom_bfs_.size(); }

private:
    int n_bf_ = 0;
    double omega_ = 0.0;
    const BasisSet* basis_ = nullptr;
    std::vector<std::vector<int>> atom_bfs_;
    std::vector<std::vector<int>> atom_shells_;
    std::vector<Eigen::MatrixXd> eri_blocks_;
};

}  // namespace vibeqc
