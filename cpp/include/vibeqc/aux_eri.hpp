// Periodic 2-centre and 3-centre Coulomb integrals on auxiliary bases.
//
// These extend the molecular `compute_2c_eri` / `compute_3c_eri` kernels in
// vibeqc/df.hpp to crystals: each integral is summed over a real-space lattice
// of cell shifts.
//
// At the Γ point, GDF requires
//
//   M_PQ      = Σ_T (P_0 | Q_T)                ← 2c periodic metric
//   T_{P,μν}  = Σ_T (P_0 | μ_0 ν_T)            ← 3c periodic ERI w/ ket image sum
//
// where P, Q are auxiliary functions in the unit cell, μ, ν are AOs in the
// unit cell, and T runs over direct-lattice translations within
// LatticeSumOptions::cutoff_bohr. These are the building blocks for the
// Cholesky-fitted Lpq tensor that drives the periodic J/K contractions.
//
// Convergence note. The bare lattice sum of (P_0 | Q_T) over T is logarithmically
// divergent for an aux basis with non-zero monopoles, because (1/r) does not
// integrate to a finite limit over an infinite charge distribution. PySCF's
// GDF handles this with a charge-compensation trick (see make_modrho_basis):
// each aux primitive is paired with a smooth Gaussian of opposite monopole, so
// the (aux − comp) pair has zero net charge and the sum converges in real
// space. These C++ kernels are agnostic to that convention — they compute
// whatever lattice sum the input aux basis prescribes. The Python wrapper
// (vibeqc.aux_basis.make_aux_basis_set) is responsible for constructing a
// charge-compensated aux basis when bit-equivalent parity vs PySCF GDF is
// the goal. For tight aux bases on ionic crystals the bare sum converges fast
// enough that the difference is below SCF tolerance — but you should verify
// the convergence with a cutoff sweep before trusting it.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "basis.hpp"
#include "df.hpp"            // for Eri3D (shared with molecular DF)
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// 2-centre periodic Coulomb metric, Γ-summed:
//   M_PQ = Σ_T (P_0 | Q_T)   (auxiliary basis, no cell-decomposition stored)
//
// Output is (n_aux, n_aux), symmetric (numerically symmetrized by averaging
// (M + M^T)/2 to absorb cell-list quadrature noise).
//
// libint2 BraKet::xs_xs (same as the molecular 2c kernel), wrapped in the
// LatticeSumOptions::cutoff_bohr cell loop.
Eigen::MatrixXd compute_2c_eri_lattice(const BasisSet& aux,
                                       const PeriodicSystem& system,
                                       const LatticeSumOptions& opts);

struct Lattice2CEriBlockSet {
    std::size_t n_aux = 0;
    std::vector<LatticeCell> cells;
    std::vector<Eigen::MatrixXd> blocks;  // blocks[i] = (P_0 | Q_T_i)
};

// Cell-resolved 2-centre periodic Coulomb metric blocks.
//
// This is the storage boundary needed by native multi-k GDF: callers can
// recover the Γ metric by summing all blocks, or apply Bloch phases to build
// q-dependent auxiliary metrics without recomputing libint shell pairs.
Lattice2CEriBlockSet compute_2c_eri_lattice_blocks(
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts);

// 3-centre periodic ERI, Γ-summed with image sum on the orbital ν (ket-side):
//   T_{P, μ, ν} = Σ_T (P_0 | μ_0 ν_T)
//
// Aux P and orbital μ stay anchored at the reference cell; only ν is shifted
// by the lattice translation. By translation invariance of the bare 1/r₁₂
// kernel the resulting tensor is symmetric in (μ, ν).
//
// Output is Eri3D with extents (n_aux, n_orb, n_orb), row-major
// (row index = aux), shape-compatible with the molecular compute_3c_eri.
//
// libint2 BraKet::xs_xx, wrapped in the LatticeSumOptions::cutoff_bohr cell
// loop on the ν shells.
Eri3D compute_3c_eri_lattice(const BasisSet& orbital,
                             const BasisSet& aux,
                             const PeriodicSystem& system,
                             const LatticeSumOptions& opts);

struct Lattice3CEriBlockSet {
    std::size_t n_aux = 0;
    std::size_t n_orb = 0;
    std::vector<LatticeCell> cells;
    std::vector<Eri3D> blocks;  // blocks[i] = (P_0 | μ_0 ν_T_i)
};

// Cell-resolved 3-centre periodic ERI blocks.
//
// The summed Γ tensor is recovered by summing over ``blocks``. Individual
// blocks are intentionally not symmetrised in (μ,ν); that symmetry emerges
// only after the full ±T lattice sum and must not be baked in before k-phase
// transforms.
Lattice3CEriBlockSet compute_3c_eri_lattice_blocks(
    const BasisSet& orbital,
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts);

// ============================================================
// RSGDF — range-separated Gaussian density fitting
// ============================================================
//
// Ye & Berkelbach, J. Chem. Phys. 154, 131104 (2021),
// DOI 10.1063/5.0046617. The core idea: range-separate the Coulomb
// kernel itself rather than the densities,
//
//   1/r₁₂ = erfc(ω r₁₂)/r₁₂  +  erf(ω r₁₂)/r₁₂
//                  └─ SR ─┘     └─── LR ───┘
//
// The SR part decays as exp(-ω²r²) at large r so the lattice sum
// Σ_T (P_0 | erfc(ω·)/|·| | Q_T) converges absolutely **with no
// compensating-charge bookkeeping needed**. The LR part has Fourier
// transform (4π/G²) exp(-G²/(4ω²)). The Gaussian damping controls the
// large-G tail but does not remove the Coulomb pole at G=0; a periodic
// assembly must omit or otherwise handle that mode according to its explicit
// electrostatic/finite-size convention. Away from G=0 the exponential decay
// permits a compact reciprocal-space sum.
//
// These kernels compute the SR halves. The LR halves are built in
// Python from analytical Gaussian Fourier transforms (no FFT needed)
// — see vibeqc.aux_basis.build_lpq_rsgdf for the full assembly.
//
// Recommended ω choice: ω ≈ min(0.4, 0.5 / r_nn) bohr⁻¹, where r_nn
// is the smallest nearest-neighbour distance (Ye 2021 §III).

// 2-centre SR Coulomb metric, image-summed on the auxiliary ket:
//   M^SR_PQ(ω) = Σ_T (P_0 | erfc(ω r)/r | Q_T)
// libint2 BraKet::xs_xs with Operator::erfc_coulomb and the
// attenuation parameter ω passed as set_params(omega).
Eigen::MatrixXd compute_2c_eri_lattice_sr(const BasisSet& aux,
                                          const PeriodicSystem& system,
                                          const LatticeSumOptions& opts,
                                          double omega);

// Gamma-only 3-centre SR ERI, image-summed on the second orbital factor:
//   T^SR_{P, μ, ν}(ω) = Σ_T (P_0 | erfc(ω r)/r | μ_0 ν_T)
// libint2 BraKet::xs_xx with Operator::erfc_coulomb.
// This is not the general multi-k RSGDF source, whose Bloch transform has
// independent translations on both AO factors (Ye and Berkelbach 2021,
// Eq. 12). A multi-k factor producer must use a double-cell source and must
// not substitute this single-translation Gamma sum.
Eri3D compute_3c_eri_lattice_sr(const BasisSet& orbital,
                                const BasisSet& aux,
                                const PeriodicSystem& system,
                                const LatticeSumOptions& opts,
                                double omega);

// ============================================================
// Periodic GDF analytic gradient kernels
// ============================================================
//
// Lattice-summed analogues of the molecular 2c / 3c gradient kernels
// in df.hpp (compute_2c_eri_gradient_weighted / compute_3c_eri_gradient_weighted).
// Each kernel adds an outer loop over direct-lattice cells T, translates
// the relevant shell origins by T, and calls libint with deriv_order=1.
//
// Image-cell derivative contributions are mapped back onto the unit-cell
// atoms: for (P_0 | Q_T), the P-centre derivative accumulates on atom A
// (unit cell), and the Q-centre derivative on atom B (unit cell — the
// same physical atom, since R_B(image) = R_B(unit) + T, ∂/∂R_B(image) =
// ∂/∂R_B(unit)).
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr.

// Lattice-summed 2-centre metric gradient:
//   grad(A, c) = Σ_{PQ} Ω_{PQ} Σ_T ∂(P_0|Q_T)/∂R_{A,c}
// Calls libint2 BraKet::xs_xs, deriv_order=1.
Eigen::MatrixXd compute_2c_eri_lattice_gradient_weighted(
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const Eigen::MatrixXd& omega);

// Lattice-summed 3-centre ERI gradient:
//   grad(A, c) = Σ_{P, μν} W^P_{μν} Σ_T ∂(P_0|μ_0 ν_T)/∂R_{A,c}
// Calls libint2 BraKet::xs_xx, deriv_order=1.
// W is (n_aux, n_orb²) row-major, same convention as the molecular kernel.
Eigen::MatrixXd compute_3c_eri_lattice_gradient_weighted(
    const BasisSet& orbital,
    const BasisSet& aux,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& W);

}  // namespace vibeqc
