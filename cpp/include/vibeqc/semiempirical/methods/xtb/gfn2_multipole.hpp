// GFN2-xTB multipole integral infrastructure.
//
// Computes dipole and quadrupole AO integrals using libint's
// emultipole1 / emultipole2 operators on the STO-6G semiempirical
// basis.  These feed the anisotropic electrostatics (AES) — the
// defining physics of GFN2-xTB (Bannwarth, Ehlert, Grimme, JCTC 2019).
//
// Convention:
//   Dipole:    d^A_c = −Σ_{μ∈A} Σ_ν D_{μν} ⟨μ | r_c−R^A_c | ν⟩
//   Quadrupole (traceless):
//     Θ^A_c = −Σ_{μ∈A} Σ_ν D_{μν} ⟨μ | ³⁄₂(r−R^A)_c₁(r−R^A)_c₂
//                                         − ½ δ_{c₁c₂}|r−R^A|² | ν⟩
//   where c ∈ {xx, xy, xz, yy, yz, zz} are the 6 unique components.
//
// Reference: CP2K PR #5131, tblite multipole Ewald contributions.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <functional>
#include <vector>

#include "vibeqc/basis.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/molecule.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// ---------------------------------------------------------------------------
// Per-atom multipole AO-matrices (3 dipole + 6 quadrupole components).
// ---------------------------------------------------------------------------

struct GFN2MultipoleAtom {
    // Dipole integrals ⟨μ|r_c − R^A_c|ν⟩ for c = 0,1,2 (x,y,z).
    // Origin is the atom centre R_A.
    Eigen::MatrixXd dip_x;
    Eigen::MatrixXd dip_y;
    Eigen::MatrixXd dip_z;

    // Quadrupole integrals in the GFN2 traceless convention
    //   q_ij = ³⁄₂(r_i−R^A_i)(r_j−R^A_j) − ½ δ_ij |r−R^A|²
    // stored in the 6-component order xx, xy, xz, yy, yz, zz.
    Eigen::MatrixXd quad_xx;
    Eigen::MatrixXd quad_xy;
    Eigen::MatrixXd quad_xz;
    Eigen::MatrixXd quad_yy;
    Eigen::MatrixXd quad_yz;
    Eigen::MatrixXd quad_zz;

    // Convenience: return dipole matrix for component c = 0,1,2.
    const Eigen::MatrixXd& dipole(int c) const {
        switch (c) {
            case 0: return dip_x;
            case 1: return dip_y;
            case 2: return dip_z;
        }
        return dip_x;  // unreachable, silence warning
    }
    Eigen::MatrixXd& dipole(int c) {
        switch (c) {
            case 0: return dip_x;
            case 1: return dip_y;
            case 2: return dip_z;
        }
        return dip_x;
    }
};

// Per-molecule collection of per-atom multipole matrices.
struct GFN2MultipoleSet {
    std::vector<GFN2MultipoleAtom> atoms;  // atoms[a]
    int n_basis = 0;
};

// ---------------------------------------------------------------------------
// Shell-resolved multipole moments (AES variables).
// ---------------------------------------------------------------------------

struct GFN2ShellMoments {
    Eigen::VectorXd q;              // shell charges  Δq_s  (n_shells)
    Eigen::VectorXd ox;             // shell dipoles  d^x_s (n_shells)
    Eigen::VectorXd oy;
    Eigen::VectorXd oz;
    Eigen::VectorXd txx;            // shell quadrupoles  Θ^{xx}_s (n_shells)
    Eigen::VectorXd txy;
    Eigen::VectorXd txz;
    Eigen::VectorXd tyy;
    Eigen::VectorXd tyz;
    Eigen::VectorXd tzz;

    // Total size = 1 + 3 + 6 = 10 per shell
    int n_shells() const { return static_cast<int>(q.size()); }

    void resize(int n) {
        int N = n;
        q.resize(N);  ox.resize(N); oy.resize(N); oz.resize(N);
        txx.resize(N); txy.resize(N); txz.resize(N);
        tyy.resize(N); tyz.resize(N); tzz.resize(N);
    }
};

// ---------------------------------------------------------------------------
// Build per-atom multipole AO-integral matrices.
// ---------------------------------------------------------------------------

GFN2MultipoleSet build_gfn2_multipole_integrals(
    const BasisSet& basis,
    const Molecule& mol,
    bool compute_quadrupole = true);

// Raw Cartesian quadrupole integrals <p|r_k r_l|q> about the GLOBAL origin,
// component order (xx, yy, zz, xy, xz, yz) — xtb build_SDQH0's qpint
// convention, needed by the faithful-AES Fock (setvsdq/builidIsoAnisotropicH1
// contract vq against these, not the traceless atom-centred set).
std::vector<Eigen::MatrixXd> build_gfn2_raw_global_quadrupole_integrals(
    const BasisSet& basis, const Molecule& mol);

// ---------------------------------------------------------------------------
// Contract the density matrix against per-atom multipole integrals to
// produce shell-resolved multipole moments.
// ---------------------------------------------------------------------------

GFN2ShellMoments compute_shell_multipole_moments(
    const Eigen::MatrixXd& density,       // D_{μν}
    const BasisSet& basis,
    const Molecule& mol,
    const GFN2MultipoleSet& mp_int,
    const std::vector<int>& ao_shell,     // AO index → shell index
    const Eigen::VectorXd& n0_shell,      // neutral reference population
    int n_shells);

// ---------------------------------------------------------------------------
// Image-summed global-origin multipole integrals (Gamma-point).
//
// For a Gamma-point density the same P couples every AO pair (kappa in the
// home cell, lambda in cell g), so every quantity that is linear in the
// integrals can be assembled from lattice moments of the per-cell blocks:
//   S0 = sum_g S(g)          S1_i = sum_g g_i S(g)      S2_ij = sum_g g_i g_j S(g)
//   D0_j = sum_g D_j(g)      D1_ij = sum_g g_i D_j(g)
//   Q0_k = sum_g Q_k(g)      (raw global-origin Cartesian quadrupoles)
// with D_j(g) = <kappa| r_j |lambda(r - g)> and Q_k(g) = <kappa| r_i r_j |lambda_g>
// about the fixed global origin.  The g-moments are what the ket-side origin
// shift R_B + g of the AES Fock (gfn2_aes.hpp) expands into, so a Gamma SCC
// needs these 28 matrices once and no per-cell integral storage.
// Component orders: D (x, y, z); Q and S2 (xx, xy, xz, yy, yz, zz), the
// libint emultipole2 order.  Entries of AO pairs beyond the pair-distance
// cutoff are zero in every component (pair_lattice.hpp rule (*)).
// ---------------------------------------------------------------------------

struct GFN2MultipoleLatticeSums {
    int n_basis = 0;
    Eigen::MatrixXd S0;
    std::array<Eigen::MatrixXd, 3> S1;
    std::array<Eigen::MatrixXd, 6> S2;
    std::array<Eigen::MatrixXd, 3> D0;
    std::array<std::array<Eigen::MatrixXd, 3>, 3> D1;
    std::array<Eigen::MatrixXd, 6> Q0;
};

// Weight of the (bra atom a, ket atom b, cell) block: 1 inside a pair
// cutoff for the Gamma-periodic driver, the fractional Wigner-Seitz
// ownership for the SECCM adapters, 0 for an absent pair.
using GFN2PairWeight = std::function<double(int a, int b, int cell_index)>;

GFN2MultipoleLatticeSums build_gfn2_multipole_lattice_sums(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    const GFN2PairWeight& pair_weight);

// Convenience: unit weight inside the pair-distance cutoff (rule (*)).
GFN2MultipoleLatticeSums build_gfn2_multipole_lattice_sums(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    double pair_cutoff_bohr);

// Molecular case: the zero cell only, no cutoff (S1, S2, D1 vanish).
GFN2MultipoleLatticeSums build_gfn2_multipole_molecular_sums(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<int>& ao_atom);

// Contraction of first geometric derivatives of the per-cell blocks with
// caller-supplied weights:
//   gradient(atom, i) = sum_g sum_{kappa lambda} [ wS d S + sum_j wD_j d D_j
//                       + sum_k wQ_k d Q_k ]_{kappa lambda}(g) / d R_atom,i
// where the bra derivative goes to the bra atom and the ket derivative to the
// ket atom; image_virial(i, j) = sum_g f_ket,i(g) g_j accumulates the
// ket-side force of every image block against its translation, the explicit
// lattice-vector part of the strain derivative.  The filler produces the
// weight matrices for one cell (its shift is passed so ket-origin-dependent
// coefficients can be formed); AO pairs beyond the pair cutoff are skipped.
struct GFN2MultipoleDerivativeContraction {
    Eigen::MatrixXd gradient;
    Eigen::Matrix3d image_virial = Eigen::Matrix3d::Zero();
};

using GFN2MultipoleWeightFiller = std::function<void(
    int cell_index, const Eigen::Vector3d& shift,
    Eigen::MatrixXd& weight_S,
    std::array<Eigen::MatrixXd, 3>& weight_D,
    std::array<Eigen::MatrixXd, 6>& weight_Q)>;

GFN2MultipoleDerivativeContraction contract_gfn2_multipole_lattice_derivatives(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    const GFN2PairWeight& pair_weight,
    const GFN2MultipoleWeightFiller& filler);

GFN2MultipoleDerivativeContraction contract_gfn2_multipole_lattice_derivatives(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    double pair_cutoff_bohr,
    const GFN2MultipoleWeightFiller& filler);

// Test seam for the libint emultipole2 first-derivative buffer layout: two
// single-primitive s shells at the origin and at (1.3, 0.3, -0.2) bohr.
// Rows 0-5 hold the 60 derivative buffers as libint lays them out (row d,
// column o = buffer d*10 + o); rows 6-17 hold the plain 10 integrals with
// the bra (rows 6-11) or ket (rows 12-17) displaced by +h/-h along x, y, z
// in that order, so a test can identify the ordering by finite differences.
Eigen::MatrixXd libint_emultipole2_derivative_layout_probe(double h);

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
