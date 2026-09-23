#include "vibeqc/semiempirical/seccm/gfn2.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/basin_gates.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"
#include "vibeqc/semiempirical/methods/indo/ccm_engine.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_aes.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_multipole.hpp"

#include "ewald_1d.h"
#include "seccm_common.h"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

namespace {

Eigen::Index expected_gfn2_shell_charge_count(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params) {
    Eigen::Index count = 0;
    for (const auto& atom : mol.atoms()) {
        const auto* element = params.element_data(atom.Z);
        if (element == nullptr) {
            // The common parameter validation below owns the missing-element
            // diagnostic. A restart length cannot be interpreted until every
            // atom has parameter data.
            return -1;
        }
        for (const auto& shell : element->shells) {
            // Mirror SemiempiricalBasis::build_shell_info(), which omits a
            // shell whose STO exponent is not positive.
            if (shell.zeta > 0.0) {
                ++count;
            }
        }
    }
    return count;
}

// ===========================================================================
// WS-weighted GFN2 supercell assembly.  Private to this adapter: the
// molecular GFN2 driver and the Gamma-periodic driver stay frozen, so the
// image-block arithmetic is consumed here through their public builders
// (build_gfn2_hamiltonian_zero_image / gfn2_shell_gamma_at_distance /
// compute_overlap_lattice_explicit).
// ===========================================================================

struct GFN2Supercell {
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd h0;
    Eigen::MatrixXd gamma_shell;
    double cyclic_repulsive = 0.0;
    int n_records = 0;
    // Faithful-AES geometry data (built only when aes_faithful): the
    // WS-weighted image-summed global-origin multipole integrals and the
    // CN-dependent per-atom AES parameters.  Both consume the same record
    // inventory as S and H0, which is what makes the AES channel
    // representative-independent (issue #348).
    xtb::GFN2MultipoleLatticeSums aes_sums;
    xtb::GFN2AesAtomParameters aes_params;
    Eigen::VectorXd cyclic_cn;
};

// ---------------------------------------------------------------------------
// Shared-kernel SECCM shell gamma (issue #444)
// ---------------------------------------------------------------------------
//
// The Wigner-Seitz image inventory is already a directed image-record set
// `(a, b, shift, weight)`, which is exactly what
// semiempirical::build_periodic_shell_gamma consumes; the Gamma-periodic
// GFN2 driver feeds the same function with unit weights over its pair
// cutoff.  Routing SECCM through it replaces three hand-rolled Ewald
// implementations (3-D, 2-D Parry/Heyes, 1-D wire) with the one that is
// pinned against the Madelung constants of rocksalt, the square lattice and
// the alternating chain, and -- the point of #444 -- makes
// ShellGammaForm::Elstner available here.  The Klopman-Ohno remainder
// summed over the WS shell is finite but has no thermodynamic limit
// (-eta^2/2R^3 with pair-dependent amplitude); the Elstner remainder decays
// exponentially, so it does.
//
// Two conventions are preserved bit-for-bit from the kernels this replaces:
//
//   * The record shift is `disp - (R_b - R_a)`, so the kernel's
//     `R_b + shift - R_a` reproduces the record's own physical displacement
//     and a boundary site keeps its ownership geometry whatever lattice
//     representative the caller typed.
//   * `molecular_onsite` (the IID 150 channel-isolation knob) removes the
//     Ewald lattice self potential from the same-atom block afterwards
//     rather than inside the kernel.  Phi(0, exclude_self) is one number for
//     the whole cell, so subtracting it from every same-atom entry is
//     identical to never having added it.
xtb::GFN2PairWeight seccm_pair_weight(
    const WSTopology& topology,
    const std::vector<detail::ShellKey>& shell_keys) {
    return [&topology, &shell_keys](int a, int b, int cell_index) -> double {
        const detail::ShellKey& label =
            shell_keys[static_cast<std::size_t>(cell_index)];
        if (a == b) {
            const bool home = label[0] == 0 && label[1] == 0 && label[2] == 0;
            return home ? 1.0 : 0.0;
        }
        double weight = 0.0;
        for (const auto& image :
             topology.cells[static_cast<std::size_t>(a)]) {
            if (image.origin != b) continue;
            if (image.image_shell_label != label) continue;
            weight += image.weight;
        }
        return weight;
    };
}

}  // namespace

Eigen::MatrixXd seccm_shell_gamma(
    const std::vector<GFN2ShellInfo>& shell_info,
    const std::vector<Eigen::Vector3d>& atom_coords,
    const WSTopology& topology,
    ShellGammaForm form,
    bool molecular_onsite,
    double alpha) {
    std::vector<GammaSite> sites;
    sites.reserve(shell_info.size());
    for (const auto& shell : shell_info) {
        GammaSite site;
        site.atom = shell.atom_idx;
        site.hardness = std::max(shell.hardness, 1.0e-6);
        sites.push_back(site);
    }
    ShellGammaSpec spec;
    spec.form = form;
    spec.ko_average = KlopmanOhnoAverage::HardnessMean;

    const EwaldCoulombKernel ewald(
        topology.translations,
        alpha > 0.0 ? alpha : detail::seccm_ewald_alpha(topology));
    Eigen::MatrixXd gamma = build_periodic_shell_gamma(
        sites, atom_coords, spec, ewald,
        detail::seccm_image_records(topology, atom_coords));

    if (molecular_onsite) {
        const double self_potential =
            ewald.potential(Eigen::Vector3d::Zero(), true);
        const int n_shells = static_cast<int>(shell_info.size());
        for (int si = 0; si < n_shells; ++si) {
            for (int sj = 0; sj < n_shells; ++sj) {
                if (shell_info[si].atom_idx == shell_info[sj].atom_idx) {
                    gamma(si, sj) -= self_potential;
                }
            }
        }
    }
    return gamma;
}

namespace {

// One-center blocks are the molecular ones; every directed two-center block
// is the GFN2 H0 image block at the record displacement, accumulated with
// the record's fractional weight (the Bredow-Geudtner-Jug construction).
//
// The H0 self-energies depend on the atom-local coordination number; on the
// finite torus that number is assembled ONCE from the directed WS records
// (issue #354) and reused for every image block, so it sees the cyclic
// images and is independent of which coordinate representative the caller
// typed for a boundary site.
GFN2Supercell assemble_gfn2_supercell(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const WSTopology& topology,
    const BasisSet& basis,
    const std::vector<GFN2ShellInfo>& shell_info,
    bool ewald_gamma,
    bool ewald_gamma_molecular_onsite,
    ShellGammaForm gamma_form,
    bool aes_faithful) {
    const int n_basis = static_cast<int>(basis.nbasis());
    const int n_atoms = static_cast<int>(mol.atoms().size());
    const auto& atoms = mol.atoms();

    const std::vector<detail::ShellKey> shell_keys =
        detail::collect_shell_keys(topology);
    const std::vector<LatticeCell> lattice_cells =
        detail::shell_lattice_cells(shell_keys, topology.translations);

    PeriodicSystem system;
    system.dim = static_cast<int>(topology.translations.size());
    system.unit_cell = mol.atoms();
    system.charge = mol.charge();
    system.multiplicity = mol.multiplicity();
    for (std::size_t axis = 0; axis < topology.translations.size(); ++axis) {
        system.lattice.col(static_cast<Eigen::Index>(axis)) =
            topology.translations[axis];
    }
    const auto overlap_blocks =
        compute_overlap_lattice_explicit(basis, system, lattice_cells);
    std::map<detail::ShellKey, std::size_t> block_by_label;
    for (std::size_t index = 0; index < shell_keys.size(); ++index) {
        block_by_label.emplace(shell_keys[index], index);
    }

    // Cyclic Eq. 18 coordination numbers.  Bannwarth, Ehlert & Grimme,
    // J. Chem. Theory Comput. 15, 1652 (2019), doi:10.1021/acs.jctc.8b01176,
    // Eq. 18:
    //   CN'_A = sum_{B != A} f_CN(R_AB),
    //   f_CN = [1 + exp(-10 (4(R_A,cov + R_B,cov)/(3 R_AB) - 1))]^-1
    //        * [1 + exp(-20 ((4(R_A,cov + R_B,cov)/3 + 2)/R_AB - 1))]^-1
    // (gfn2_h0_coordination_pair_contribution, with the shared 15-bohr
    // cutoff).  On the finite torus the sum over "B != A" is the directed
    // Wigner-Seitz inventory of atom A: every record contributes at its
    // physical image displacement |d| with its two-center ownership weight
    // omega = 1/n (Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
    // doi:10.1002/jcc.23550, Eqs. 4-5 - the same weight that assembles the
    // cyclic overlap above), and there is no 1/2: Eq. 18 is a directed
    // central-atom sum, not a pair energy.  The trivial one-replica record
    // set (zero-translation images, unit weights, origin order) reproduces
    // gfn2_h0_coordination_numbers term for term, so the molecular limit is
    // unchanged.  Before #354 every image block recomputed the CN
    // molecularly from the typed home-cell atoms, which omitted the cyclic
    // images and broke the torus translation symmetry of H0.
    Eigen::VectorXd cyclic_cn = Eigen::VectorXd::Zero(n_atoms);
    for (int central = 0; central < n_atoms; ++central) {
        for (const auto& image :
             topology.cells[static_cast<std::size_t>(central)]) {
            const double dx = image.disp[0];
            const double dy = image.disp[1];
            const double dz = image.disp[2];
            const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
            cyclic_cn(central) += image.weight
                * gfn2_h0_coordination_pair_contribution(
                    atoms[central].Z, atoms[image.origin].Z, R);
        }
    }

    // Per-shell-label H0 blocks.  Image blocks keep the H(g) = H(-g)^T
    // nonsymmetric block convention of build_gfn2_hamiltonian_zero_image;
    // the validated reverse-image topology reassembles them into a
    // symmetric supercell matrix.  Every block consumes the one cyclic CN
    // vector above.
    std::vector<Eigen::MatrixXd> h0_blocks(lattice_cells.size());
    for (std::size_t ci = 0; ci < lattice_cells.size(); ++ci) {
        h0_blocks[ci] = build_gfn2_hamiltonian_zero_image_with_cn(
            basis, overlap_blocks.blocks[ci], mol, params,
            lattice_cells[ci].r_cart, cyclic_cn);
    }

    const std::vector<detail::AOInfo> ao_info = detail::ao_info_for_basis(basis);

    GFN2Supercell super;
    super.overlap = Eigen::MatrixXd::Zero(n_basis, n_basis);
    super.h0 = Eigen::MatrixXd::Zero(n_basis, n_basis);
    const Eigen::MatrixXd& home_overlap =
        overlap_blocks.blocks[block_by_label.at(detail::ShellKey{0, 0, 0})];
    const Eigen::MatrixXd& home_h0 =
        h0_blocks[block_by_label.at(detail::ShellKey{0, 0, 0})];
    for (int mu = 0; mu < n_basis; ++mu) {
        const int central = ao_info[static_cast<std::size_t>(mu)].atom;
        for (int nu = 0; nu < n_basis; ++nu) {
            const int origin = ao_info[static_cast<std::size_t>(nu)].atom;
            if (central == origin) {
                super.overlap(mu, nu) = home_overlap(mu, nu);
                super.h0(mu, nu) = home_h0(mu, nu);
                continue;
            }
            for (const auto& image :
                 topology.cells[static_cast<std::size_t>(central)]) {
                if (image.origin != origin) continue;
                const std::size_t block_index =
                    block_by_label.at(image.image_shell_label);
                super.overlap(mu, nu) += image.weight
                    * overlap_blocks.blocks[block_index](mu, nu);
                super.h0(mu, nu) += image.weight * h0_blocks[block_index](mu, nu);
            }
        }
    }

    // Shell-resolved gamma: the WS-truncated Klopman-Ohno sum by default
    // (same-atom block is the molecular one; every other atom contributes
    // through its WS records at the record distances), or the Ewald-split
    // shell kernel when ewald_gamma is on. The 3-D KO remainder has no
    // thermodynamic limit (#444): production entry rejects that route,
    // while the internal kernel remains available for diagnostics.
    const int n_shells = static_cast<int>(shell_info.size());
    super.gamma_shell = Eigen::MatrixXd::Zero(n_shells, n_shells);
    if (ewald_gamma) {
        super.gamma_shell = seccm_shell_gamma(
            shell_info, detail::seccm_atom_coords(mol), topology, gamma_form,
            ewald_gamma_molecular_onsite);
    } else {
        for (int si = 0; si < n_shells; ++si) {
            for (int sj = 0; sj < n_shells; ++sj) {
                if (shell_info[si].atom_idx == shell_info[sj].atom_idx) {
                    super.gamma_shell(si, sj) = gfn2_shell_gamma_at_distance(
                        shell_info[si], shell_info[sj], 0.0);
                }
            }
        }
        for (int central = 0; central < n_atoms; ++central) {
            for (const auto& image :
                 topology.cells[static_cast<std::size_t>(central)]) {
                const int origin = image.origin;
                const double R = image.disp.norm();
                for (int si = 0; si < n_shells; ++si) {
                    if (shell_info[si].atom_idx != central) continue;
                    for (int sj = 0; sj < n_shells; ++sj) {
                        if (shell_info[sj].atom_idx != origin) continue;
                        super.gamma_shell(si, sj) += image.weight
                            * gfn2_shell_gamma_at_distance(
                                  shell_info[si], shell_info[sj], R);
                    }
                }
            }
        }
    }

    // WS-weighted pair repulsion: 0.5 per directed record, the two validated
    // directions of each pair carrying equal weights (SCC-DFTB-SECCM
    // convention).
    for (int central = 0; central < n_atoms; ++central) {
        for (const auto& image :
             topology.cells[static_cast<std::size_t>(central)]) {
            super.cyclic_repulsive += 0.5 * image.weight
                * params.repulsive_energy(
                    atoms[central].Z, atoms[image.origin].Z,
                    image.disp.norm());
            ++super.n_records;
        }
    }

    // Faithful AES (issue #348): the multipole integrals are image-summed
    // over the same Wigner-Seitz inventory as S and H0, so a boundary site
    // retyped by a lattice vector produces the same moments.  The shipped
    // ad-hoc shell-resolved path builds them from the home-cell molecule
    // instead and is representative-dependent; the two cannot be reconciled
    // because that kernel consumes shell-resolved moments and the pair
    // inventory produces atom-resolved CAMM.
    super.cyclic_cn = cyclic_cn;
    if (aes_faithful) {
        std::vector<int> ao_atom(static_cast<std::size_t>(n_basis), 0);
        for (int mu = 0; mu < n_basis; ++mu) {
            ao_atom[static_cast<std::size_t>(mu)] =
                ao_info[static_cast<std::size_t>(mu)].atom;
        }
        super.aes_sums = xtb::build_gfn2_multipole_lattice_sums(
            basis, atoms, lattice_cells, ao_atom,
            seccm_pair_weight(topology, shell_keys));
        super.aes_params =
            xtb::gfn2_aes_atom_parameters(atoms, params, cyclic_cn);
    }
    return super;
}

bool has_extended_period_element(const Molecule& mol) {
    for (const auto& atom : mol.atoms())
        if (atom.Z > 18) return true;
    return false;
}

// Everything the supercell SCC loop needs, precomputed once per run.
struct SupercellContext {
    const Molecule* mol = nullptr;
    const WSTopology* topology = nullptr;
    int group_order = 0;
    double gap_tolerance = 1.0e-8;
    double conv_tol = 1.0e-6;
    GFN2Supercell super;
    std::vector<GFN2ShellInfo> shell_info;
    std::vector<int> ao_shell;
    std::vector<int> ao_atom;
    Eigen::VectorXd n0_shell;
    Eigen::VectorXd gam3_shell;
    xtb::GFN2MultipoleSet mp_int;
    int n_basis = 0;
    int n_atoms = 0;
    int n_shells = 0;
    int n_occ = 0;
    bool has_gam3 = false;
    // Opt-in Madelung/Ewald embedding state (empty when disabled).
    bool madelung = false;
    // Experimental knobs: S-weighted Madelung deposit and self-term
    // removal (IID 150).
    bool madelung_s_weighted = false;
    bool madelung_no_self = false;
    // Experimental validation knob: disable the WS-folded AES channel
    // (see GFN2SECCMOptions::include_aes).
    bool include_aes = true;
    // Faithful record-resolved AES (issue #348): the Bannwarth 2019 model
    // on cumulative atomic multipole moments built from the Wigner-Seitz
    // image inventory, replacing the ad-hoc shell-resolved gamma^3/gamma^5
    // kernel on home-cell moments.
    bool aes_faithful = false;
    double aes_damping = 0.5;
    std::vector<Eigen::Vector3d> positions;
    const WSTopology* topology_for_records = nullptr;
    // Canonical orthogonalizer of the supercell overlap (Peintinger-Bredow
    // 2014 screening remedy). Empty when the overlap passed the
    // positive-definiteness gate: the legacy generalized solve is then
    // bit-for-bit the shipped arithmetic. Non-empty when the cyclic
    // overlap had non-positive modes; the charge map then solves the
    // ordinary eigenproblem in the X-transformed basis.
    Eigen::MatrixXd ortho;
    Eigen::MatrixXd madkonst;
    std::vector<std::vector<indo::WSNeighbor>> ews;
};

// Shell population tolerance for the physical-basin gate. The bounds and
// predicates live in one place, shared with the periodic Gamma driver
// (issue #411): see vibeqc/semiempirical/core/basin_gates.hpp.
bool shell_populations_physical(
    const SupercellContext& ctx, const Eigen::VectorXd& dq_shell) {
    return shell_populations_within_basin(
        ctx.shell_info, ctx.n0_shell, dq_shell);
}

bool molecular_shell_populations_physical(
    const Molecule& mol,
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::VectorXd& dq_shell) {
    if (dq_shell.size() != static_cast<Eigen::Index>(shell_info.size())) {
        return false;
    }
    Eigen::VectorXd n0_shell(dq_shell.size());
    for (std::size_t si = 0; si < shell_info.size(); ++si) {
        const GFN2ShellInfo& shell = shell_info[si];
        n0_shell(static_cast<Eigen::Index>(si)) =
            xtb::gfn2_reference_occupation(mol.atoms()[shell.atom_idx].Z,
                                           shell.l);
    }
    return shell_populations_within_basin(shell_info, n0_shell, dq_shell);
}

Eigen::VectorXd effective_madelung_potential(
    const SupercellContext& ctx, const Eigen::VectorXd& dq_atom) {
    Eigen::VectorXd potential = indo::_madelung_potential_ewald(
        dq_atom, ctx.madkonst, ctx.ews);
    if (ctx.madelung_no_self) {
        for (int atom = 0; atom < ctx.n_atoms; ++atom) {
            potential(atom) -=
                ctx.madkonst(atom, atom) * dq_atom(atom);
        }
    }
    return potential;
}

// One GFN2-SECCM charge-map evaluation: build H(dq) from the fixed
// supercell matrices (H0, S, shell gamma, opt-in Madelung/Ewald
// embedding), solve the generalized eigenproblem, build the density
// (Fermi-Dirac occupations at the same temperature as the engine, or the
// hard-Aufbau T = 0 path), and return the Mulliken shell fluctuations.
// This is the map dq -> dq_new whose fixed point the SCC loop iterates
// and whose Jacobian the Newton mixer finite-differences; the loop and
// the mixer call it so the iteration, the line search, and the Jacobian
// share the same arithmetic bit-for-bit. `moments` supplies the lagged
// shell multipole moments for the AES potential (nullptr /
// have_moments = false before the first density exists); the FD
// Jacobian freezes them, making the Newton step a quasi-Newton chord on
// the charge subspace (the moments refresh each iteration from the
// mixed density). The `iter` argument only labels the diagnostic
// exceptions.
struct GFN2ChargeMapOutput {
    Eigen::VectorXd dq_new;
    Eigen::MatrixXd density;
    Eigen::VectorXd eps;
    Eigen::MatrixXd coefficients;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd V_mp;
    double entropy = 0.0;
};

// Reduced SCC state of the SECCM engine: the quantities the Fock matrix
// depends on self-consistently, packed as one vector so a charge mixer acts
// on all of them together (issue #409).
//
// On the shipped ad-hoc AES path that is the shell charge fluctuations
// alone -- its multipole potential is rebuilt from each new density and is
// not an independent iterate -- so the state is [dq] and every existing
// trajectory is unchanged.  On the faithful path the atom-resolved CAMM
// enter the Fock through the AES potentials, exactly as they do in xtb, so
// the state is [dq; mu; theta].  Mixing charges while merely damping the
// moments is what made xtb trajectory parity structurally unreachable: the
// two are one fixed-point problem, not a fixed-point problem plus a lag.
struct SeccmReducedState {
    int n_shells = 0;
    int n_atoms = 0;      // 0 when the moments are not part of the state
    Eigen::VectorXd x;

    void resize(int shells, int atoms) {
        n_shells = shells;
        n_atoms = atoms;
        x = Eigen::VectorXd::Zero(shells + 9 * atoms);
    }
    bool has_moments() const { return n_atoms > 0; }
    Eigen::VectorXd dq_shell() const { return x.head(n_shells); }
    void set_dq(const Eigen::VectorXd& dq) { x.head(n_shells) = dq; }

    xtb::GFN2AesMoments moments(
        const std::vector<GFN2ShellInfo>& shell_info) const {
        xtb::GFN2AesMoments moments;
        moments.q = Eigen::VectorXd::Zero(n_atoms);
        moments.mu = Eigen::MatrixXd::Zero(n_atoms, 3);
        moments.theta = Eigen::MatrixXd::Zero(n_atoms, 6);
        // Atomic partial charge is the negated sum of its shell
        // fluctuations, the periodic driver's convention.
        for (int si = 0; si < n_shells; ++si) {
            moments.q(shell_info[static_cast<std::size_t>(si)].atom_idx)
                -= x(si);
        }
        for (int a = 0; a < n_atoms; ++a) {
            for (int k = 0; k < 3; ++k) {
                moments.mu(a, k) = x(n_shells + 3 * a + k);
            }
            for (int k = 0; k < 6; ++k) {
                moments.theta(a, k) =
                    x(n_shells + 3 * n_atoms + 6 * a + k);
            }
        }
        return moments;
    }

    void set(const Eigen::VectorXd& dq, const xtb::GFN2AesMoments& moments) {
        set_dq(dq);
        for (int a = 0; a < n_atoms; ++a) {
            for (int k = 0; k < 3; ++k) {
                x(n_shells + 3 * a + k) = moments.mu(a, k);
            }
            for (int k = 0; k < 6; ++k) {
                x(n_shells + 3 * n_atoms + 6 * a + k) = moments.theta(a, k);
            }
        }
    }
};

// Record-resolved cumulative atomic multipole moments of a supercell
// density (issue #348).  The partial charges are the Mulliken charges of
// the same density against the WS-weighted overlap, so the moments and the
// charges the AES functional pairs them with come from one state.
xtb::GFN2AesMoments seccm_aes_moments(
    const SupercellContext& ctx, const Eigen::MatrixXd& density) {
    const Eigen::MatrixXd DS = density * ctx.super.overlap;
    Eigen::VectorXd q_atom = Eigen::VectorXd::Zero(ctx.n_atoms);
    for (int mu = 0; mu < ctx.n_basis; ++mu) {
        q_atom(ctx.ao_atom[static_cast<std::size_t>(mu)]) += DS(mu, mu);
    }
    const auto& atoms = ctx.mol->atoms();
    for (int a = 0; a < ctx.n_atoms; ++a) {
        q_atom(a) = static_cast<double>(
            xtb::gfn2_valence_electrons(atoms[static_cast<std::size_t>(a)].Z))
            - q_atom(a);
    }
    return xtb::gfn2_aes_moments(
        density, ctx.super.aes_sums, ctx.ao_atom, ctx.positions, q_atom);
}

GFN2ChargeMapOutput gfn2_seccm_charge_map(
    const SupercellContext& ctx,
    const Eigen::VectorXd& dq_shell,
    const xtb::GFN2ShellMoments* moments,
    bool have_moments,
    double electronic_temperature,
    int iter,
    const xtb::GFN2AesPotentials* aes_potentials = nullptr) {
    // Isotropic second-order (shell-resolved) potential plus the
    // shell-resolved on-site third-order term (V3 = -G dq^2).
    Eigen::VectorXd V_shell = ctx.super.gamma_shell * dq_shell;
    for (int si = 0; si < ctx.n_shells; ++si) {
        const double G = ctx.gam3_shell(si);
        if (G != 0.0) {
            V_shell(si) += -G * dq_shell(si) * dq_shell(si);
        }
    }

    // Anisotropic second-order (ad-hoc shell-resolved AES) potential,
    // WS-weighted over the records: each directed record carries its
    // fractional weight through the potential sums.  The record
    // displacement d = r_b_image - r_central maps to the molecular
    // driver's r_a - r_b = -d.
    Eigen::VectorXd V_mp = Eigen::VectorXd::Zero(ctx.n_shells);
    if (ctx.include_aes && ctx.aes_faithful) {
        // The faithful channel is atom-resolved and enters the Fock matrix
        // directly (Eqs. 39-44), not through a shell potential, so V_mp
        // stays zero and the AES energy is the Eq. 25 functional rather
        // than 1/2 dq.V_mp.  Handled after H_scc is formed below.
    } else if (ctx.include_aes && have_moments) {
        for (int central = 0; central < ctx.n_atoms; ++central) {
            for (const auto& image :
                 ctx.topology->cells[static_cast<std::size_t>(central)]) {
                const int b = image.origin;
                const double w = image.weight;
                const double dx = -image.disp[0];
                const double dy = -image.disp[1];
                const double dz = -image.disp[2];
                const double R2 = dx * dx + dy * dy + dz * dz;
                const double R = std::sqrt(R2);
                if (R < 1e-12) continue;
                const double invR3 = 1.0 / (R2 * R);
                const double invR5 = invR3 / R2;
                const double qxx = 3 * dx * dx - R2;
                const double qxy = 3 * dx * dy;
                const double qxz = 3 * dx * dz;
                const double qyy = 3 * dy * dy - R2;
                const double qyz = 3 * dy * dz;
                const double qzz = 3 * dz * dz - R2;
                for (int si = 0; si < ctx.n_shells; ++si) {
                    if (ctx.shell_info[si].atom_idx != central) continue;
                    for (int sj = 0; sj < ctx.n_shells; ++sj) {
                        if (ctx.shell_info[sj].atom_idx != b) continue;
                        const double g = gfn2_shell_gamma_at_distance(
                            ctx.shell_info[si], ctx.shell_info[sj], R);
                        if (g < 1e-15) continue;
                        const double g3 = g * g * g;
                        const double g5 = g3 * g * g;
                        const double pot_dip = -g3
                            * (moments->ox(sj) * dx
                               + moments->oy(sj) * dy
                               + moments->oz(sj) * dz)
                            * invR3;
                        const double pot_quad = 0.5 * g5
                            * (moments->txx(sj) * qxx
                               + moments->txy(sj) * qxy * 2
                               + moments->txz(sj) * qxz * 2
                               + moments->tyy(sj) * qyy
                               + moments->tyz(sj) * qyz * 2
                               + moments->tzz(sj) * qzz)
                            * invR5;
                        V_mp(si) += w * (pot_dip + pot_quad);
                    }
                }
            }
        }
    }
    V_shell += V_mp;

    Eigen::MatrixXd H_scc = ctx.super.h0;
    for (int mu = 0; mu < ctx.n_basis; ++mu) {
        const int sm = ctx.ao_shell[mu];
        for (int nu = 0; nu < ctx.n_basis; ++nu) {
            const int sn = ctx.ao_shell[nu];
            H_scc(mu, nu) += 0.5 * ctx.super.overlap(mu, nu)
                * (V_shell(sm) + V_shell(sn));
        }
    }

    // Faithful AES Fock (Eqs. 39-44) through the WS-weighted image-summed
    // integrals: the bra side sits at R_A and the ket side at the record's
    // own image origin, which is the whole point of building the sums from
    // the record inventory.  Lagged one iteration like the molecular
    // driver's, and damped for the same reason (the on-site DPOL/QPOL
    // feedback is stiff).
    if (ctx.include_aes && ctx.aes_faithful && aes_potentials != nullptr) {
        xtb::gfn2_aes_add_fock(
            H_scc, ctx.super.aes_sums, ctx.ao_atom, ctx.positions,
            *aes_potentials);
    }

    // Opt-in Madelung/Ewald embedding (MSINDO CCM convention
    // F_add(k,k) = -V_mad): the Coulombic image-sum tail beyond the
    // WS cell, acting on the atomic Mulliken fluctuations of the
    // current shell iterate (all shells of an atom sit at the same
    // site, so the shell-resolved potential collapses to the atomic
    // one). The S-weighted deposit (0.5 S (V_A + V_B), the gamma
    // channel's form) was prototyped as a stability fix for
    // strongly ionic cells and REVERTED: it converges corundum but
    // over-polarizes it (q_rms 0.68 vs xtb 0.41) and removes the
    // MgO minimum that the diagonal convention produces. Neither
    // deposit is uniformly right; the root cause is the abrupt
    // WS-boundary switch between the damped Klopman-Ohno kernel and
    // the bare Coulomb tail, which needs the self-consistent
    // image-summed gamma treatment, not a deposit-convention tweak.
    if (ctx.madelung) {
        Eigen::VectorXd dq_atom_iter =
            Eigen::VectorXd::Zero(ctx.n_atoms);
        for (int si = 0; si < ctx.n_shells; ++si) {
            dq_atom_iter(ctx.shell_info[si].atom_idx) -= dq_shell(si);
        }
        const Eigen::VectorXd V_mad_eff = effective_madelung_potential(
            ctx, dq_atom_iter);
        if (!V_mad_eff.allFinite()) {
            throw std::runtime_error(
                "GFN2-SECCM embedding potential produced non-finite "
                "values");
        }
        if (ctx.madelung_s_weighted) {
            // S-weighted deposit (0.5 S (V_A + V_B), the gamma
            // channel's form): the variational convention for the
            // non-orthogonal basis, consistent with the S-weighted
            // Mulliken response. Experimental knob for IID 150 (the
            // diagonal form's mismatch with the Mulliken response is
            // violent on the synthetic aligned-plane stress map).
            for (int mu = 0; mu < ctx.n_basis; ++mu) {
                const double V_mu = V_mad_eff(ctx.ao_atom[mu]);
                for (int nu = 0; nu < ctx.n_basis; ++nu) {
                    H_scc(mu, nu) -= 0.5 * ctx.super.overlap(mu, nu)
                        * (V_mu + V_mad_eff(ctx.ao_atom[nu]));
                }
            }
        } else {
            for (int mu = 0; mu < ctx.n_basis; ++mu) {
                H_scc(mu, mu) -= V_mad_eff(ctx.ao_atom[mu]);
            }
        }
    }

    // Diagonalize in the canonical-orthogonalized basis when the cyclic
    // overlap needed screening (Peintinger-Bredow 2014 remedy); otherwise
    // the legacy generalized solve, bit-for-bit the shipped arithmetic.
    GFN2ChargeMapOutput out;
    if (ctx.ortho.size() > 0) {
        const detail::CanonicalEigensolution canonical =
            detail::solve_canonical_orthogonalized(
                H_scc, ctx.ortho, "GFN2-SECCM");
        out.eps = canonical.eigenvalues;
        out.coefficients = canonical.coefficients;
    } else {
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            H_scc, ctx.super.overlap);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "GFN2-SECCM generalized eigensolver failed at iter "
                + std::to_string(iter));
        }
        out.eps = solver.eigenvalues();
        out.coefficients = solver.eigenvectors();
    }
    out.hamiltonian = std::move(H_scc);

    Eigen::MatrixXd D;
    if (electronic_temperature > 0.0) {
        KPointOccupationOptions occupation_options;
        occupation_options.smearing_temperature = electronic_temperature;
        auto occupation = compute_closed_shell_kpoint_occupations(
            {out.eps}, {1.0}, 2.0 * ctx.n_occ, ctx.n_occ, occupation_options);
        out.entropy = occupation.entropy;
        D = out.coefficients * occupation.occupations_per_k.front().asDiagonal()
            * out.coefficients.transpose();
    } else {
        Eigen::VectorXd occ = Eigen::VectorXd::Zero(out.eps.size());
        for (int i = 0; i < ctx.n_occ; ++i) occ(i) = 2.0;
        D = out.coefficients * occ.asDiagonal() * out.coefficients.transpose();
    }
    Eigen::MatrixXd DS = D * ctx.super.overlap;

    out.dq_new = Eigen::VectorXd::Zero(ctx.n_shells);
    for (int mu = 0; mu < ctx.n_basis; ++mu) {
        const int si = ctx.ao_shell[mu];
        if (si >= 0) out.dq_new(si) += DS(mu, mu);
    }
    for (int si = 0; si < ctx.n_shells; ++si) {
        out.dq_new(si) -= ctx.n0_shell(si);
    }
    out.density = std::move(D);
    out.V_mp = std::move(V_mp);
    return out;
}

// One line-searched Newton step on the charge-map fixed point f(dq) = dq
// (the probe_slab_jacobian.py loop, studies/seccm-bulk3d). The Jacobian
// J = df/dq is built by forward finite differences with the lagged shell
// moments frozen (a quasi-Newton chord on the charge subspace: the
// moments refresh each iteration from the mixed density, so the frozen
// chord still converges to the true fixed point). The step solves the
// plain Newton system (J - I) x = residual: the map conserves the total
// supercell charge, so u^T J = 0 for u = ones/sqrt(n) and any solution
// of u^T(J - I)x = -u^T x = u^T residual = 0 automatically lies on the
// zero-sum manifold (the step's residual uniform component is projected
// out as roundoff cleanup). A residual-reduction line search halves the
// damping until |f(q_new) - q_new|_inf drops below the pre-step
// residual (the probe's exact loop); if no trial reduces it the step
// falls back to damped simple mixing so a failed Newton probe cannot
// strand the iteration outside the physical basin. (A regularizer of
// the form J - I + lambda u u^T was prototyped and REVERTED: its right
// null vector is (J - I)^{-1} u, not u, so the minimal-norm solve
// mixes a spurious uniform component into the step and the map is not
// invariant under uniform charge shifts.)
struct NewtonStepOutput {
    Eigen::VectorXd dq_next;
    Eigen::MatrixXd density;   // density at the accepted state
    bool accepted = false;
};

// Trust-region cap on the Newton step (electrons): the plain solve can
// produce unbounded steps along near-degenerate directions of J - I;
// the direction is kept and the length capped so the residual-reduction
// line search operates at the physical shell-charge scale.
constexpr double kNewtonStepCap = 0.5;

NewtonStepOutput gfn2_seccm_newton_step(
    const SupercellContext& ctx,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& f_dq,
    const xtb::GFN2ShellMoments* moments,
    bool have_moments,
    double electronic_temperature,
    double simple_step,
    int iter,
    double max_change) {
    NewtonStepOutput out;
    const int n = ctx.n_shells;
    const Eigen::VectorXd residual = f_dq - dq;

    // FD Jacobian (forward differences, the probe's delta).
    const double fd_delta = 1.0e-4;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd qp = dq;
        qp(j) += fd_delta;
        GFN2ChargeMapOutput map_p = gfn2_seccm_charge_map(
            ctx, qp, moments, have_moments, electronic_temperature, iter);
        J.col(j) = (map_p.dq_new - f_dq) / fd_delta;
    }

    // Plain Newton solve (J - I) x = residual. ColPivHouseholderQR is
    // robust for the numerically non-symmetric system and returns the
    // minimal-norm solution when the matrix is rank-deficient; the
    // zero-sum property of the step follows from the solve itself
    // (u^T residual = 0 exactly because the map conserves charge).
    bool step_ok = J.allFinite();
    Eigen::VectorXd step = Eigen::VectorXd::Zero(n);
    if (step_ok) {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(
            J - Eigen::MatrixXd::Identity(n, n));
        step = qr.solve(residual);
        step_ok = step.allFinite();
        if (step_ok) {
            // Roundoff cleanup: pin the step's uniform component to
            // zero so the iterate stays on the zero-sum charge manifold
            // exactly (the map is not invariant under uniform shifts).
            const Eigen::VectorXd u = Eigen::VectorXd::Constant(
                n, 1.0 / std::sqrt(static_cast<double>(n)));
            step -= u * u.dot(step);
            // Trust-region cap: the GAM3 term makes J - I nearly
            // singular along stiff modes, so the plain solve can
            // produce unbounded steps there. The direction is kept and
            // the length capped at the physical shell-charge scale so
            // the residual-reduction line search below never wastes
            // its halvings on a wild step (the unbounded step was the
            // GAM3-map crawl: 300+ iterations without reaching the
            // 4-layer fixed point).
            const double step_inf = step.cwiseAbs().maxCoeff();
            if (step_inf > kNewtonStepCap) {
                step *= kNewtonStepCap / step_inf;
            }
        }
    }

    if (step_ok) {
        double damping = 1.0;
        for (int k = 0; k < 12; ++k) {
            Eigen::VectorXd q_trial = dq - damping * step;
            GFN2ChargeMapOutput map_t = gfn2_seccm_charge_map(
                ctx, q_trial, moments, have_moments,
                electronic_temperature, iter);
            const double r_trial =
                (map_t.dq_new - q_trial).cwiseAbs().maxCoeff();
            if (std::isfinite(r_trial) && r_trial < max_change) {
                out.dq_next = std::move(q_trial);
                out.density = std::move(map_t.density);
                out.accepted = true;
                return out;
            }
            damping *= 0.5;
        }
    }

    // Fallback: the damped simple-mixing step the default path would
    // take, so a failed Newton probe cannot strand the iteration.
    out.dq_next = simple_step * f_dq + (1.0 - simple_step) * dq;
    out.accepted = false;
    return out;
}

// One supercell SCC solve with a fixed (charge_mixing, max_iter,
// electronic_temperature, restart) tuple.  Mirrors the molecular driver's loop
// structure: the ad-hoc shell-resolved AES potential joins the Fock from the
// first iteration that has moments, the energy functional is evaluated at
// the converged density's own Mulliken charges, and the per-cell fields are
// the raw finite-cluster totals divided by the finite-group order.
Eigen::VectorXd seccm_residual_trace_vector(
    const std::vector<double>& values) {
    Eigen::VectorXd trace(static_cast<Eigen::Index>(values.size()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        trace(static_cast<Eigen::Index>(i)) = values[i];
    }
    return trace;
}

// Budget-independent stabilisation checkpoint, the #244 pattern of the
// periodic Gamma driver (and the #3 advisory checkpoint of the molecular
// driver): the primary attempt keeps its whole budget while it is still
// contracting, and is handed to the ladder only when the best residual of
// the last window no longer beats the best of the window before. The old
// rule capped the primary at 500 iterations whenever max_iter >= 2500, so
// raising the budget by one changed the trajectory and every default run
// (max_iter = 3600) abandoned a converging solve at 500.
constexpr int kStabilityCheckpoint = 500;
constexpr int kProgressWindow = 25;

bool seccm_still_contracting(const std::vector<double>& trace) {
    const int n = static_cast<int>(trace.size());
    if (n < 2 * kProgressWindow) return false;
    const auto end = trace.end();
    const double recent = *std::min_element(end - kProgressWindow, end);
    const double prior = *std::min_element(
        end - 2 * kProgressWindow, end - kProgressWindow);
    return recent < prior;
}

void stamp_supercell_attempt(
    GFN2SECCMResult& result,
    int allocated_max_iter,
    double charge_mixing,
    double electronic_temperature,
    SCCMixer scc_mixer,
    bool restart_supplied,
    const std::vector<double>& residuals,
    const std::string& exit_reason) {
    result.scc_max_change_trace = seccm_residual_trace_vector(residuals);
    xtb::GFN2SCCAttempt attempt;
    switch (scc_mixer) {
        case SCCMixer::DIIS: attempt.solver = "supercell_diis"; break;
        case SCCMixer::Broyden: attempt.solver = "supercell_broyden"; break;
        case SCCMixer::BroydenEyert:
            attempt.solver = "supercell_broyden_eyert";
            break;
        case SCCMixer::Newton: attempt.solver = "supercell_newton"; break;
        default: attempt.solver = "supercell_simple"; break;
    }
    attempt.allocated_max_iter = allocated_max_iter;
    attempt.n_iter = result.n_iter;
    attempt.ladder_charge_mixing = charge_mixing;
    attempt.electronic_temperature = electronic_temperature;
    attempt.scc_mixer = scc_mixer;
    attempt.restart_supplied = restart_supplied;
    attempt.exit_reason = exit_reason;
    // Both budget exhaustion and a stalled stabilization checkpoint (#294)
    // end an attempt that never reached the residual tolerance; every other
    // reason describes a fixed point the gates then judged.
    attempt.scc_converged = exit_reason != "iteration_limit"
        && exit_reason != "stalled_checkpoint";
    attempt.physical_basin = result.physical_basin;
    attempt.max_change_trace = result.scc_max_change_trace;
    result.attempts = {std::move(attempt)};
    result.selected_attempt_index =
        result.converged && result.physical_basin ? 0 : -1;
}

void prepend_attempt_history(
    GFN2SECCMResult& retry,
    const GFN2SECCMResult& prefix) {
    Eigen::VectorXd trace(
        prefix.scc_max_change_trace.size()
        + retry.scc_max_change_trace.size());
    trace.head(prefix.scc_max_change_trace.size()) =
        prefix.scc_max_change_trace;
    trace.tail(retry.scc_max_change_trace.size()) =
        retry.scc_max_change_trace;
    retry.scc_max_change_trace = std::move(trace);

    const int attempt_offset = static_cast<int>(prefix.attempts.size());
    std::vector<xtb::GFN2SCCAttempt> attempts;
    attempts.reserve(prefix.attempts.size() + retry.attempts.size());
    attempts.insert(
        attempts.end(), prefix.attempts.begin(), prefix.attempts.end());
    attempts.insert(
        attempts.end(), retry.attempts.begin(), retry.attempts.end());
    retry.attempts = std::move(attempts);
    if (retry.selected_attempt_index >= 0) {
        retry.selected_attempt_index += attempt_offset;
    }
    retry.n_iter += prefix.n_iter;
}

GFN2SECCMResult run_supercell_scc(
    const SupercellContext& ctx,
    const BasisSet& basis,
    SCCMixer scc_mixer,
    double charge_mixing,
    int max_iter,
    bool advisory_checkpoint,
    double electronic_temperature,
    const Eigen::VectorXd& initial_shell_charges) {
    bool stalled_at_checkpoint = false;
    const Molecule& mol = *ctx.mol;
    const auto& atoms = mol.atoms();

    Eigen::VectorXd dq_shell = Eigen::VectorXd::Zero(ctx.n_shells);
    if (initial_shell_charges.size() == ctx.n_shells) {
        dq_shell = initial_shell_charges;
    }
    xtb::GFN2ShellMoments moments;
    moments.resize(ctx.n_shells);
    bool have_moments = false;
    // The reduced SCC state (issue #409).  On the faithful path it carries
    // the CAMM alongside the shell charges, so the mixer sees the whole
    // fixed-point problem and the AES potentials are built from the *mixed*
    // moments rather than from the latest density with a damping lag.
    const bool joint_state = ctx.include_aes && ctx.aes_faithful;
    if (joint_state && scc_mixer == SCCMixer::Newton) {
        throw std::invalid_argument(
            "GFN2-SECCM scc_mixer='newton' is unavailable with "
            "aes_faithful=True: its finite-difference Jacobian is defined on "
            "the charge subspace, and the faithful AES makes the multipole "
            "moments part of the iterate (#409)");
    }
    SeccmReducedState state;
    state.resize(ctx.n_shells, joint_state ? ctx.n_atoms : 0);
    state.set_dq(dq_shell);

    // GFN2's third-order term always lands the molecular driver on damped
    // simple mixing (capped at 0.1); without it the driver's default mixer
    // is simple mixing at the requested fraction. Charge-DIIS was prototyped
    // as an accelerator for stiff embedded ionic cells and REVERTED: it does
    // not contract the embedded corundum map and it slows the embedded MgO
    // cell 13x (2940 vs 218 iterations) - the third-order map's oscillatory
    // modes defeat raw charge-vector DIIS, exactly the molecular driver's
    // rationale for keeping simple mixing. Broyden and DIIS remain opt-in
    // for maps whose fixed point simple mixing orbits (IID 141: the
    // four-layer alternating-plane slab's repelling fixed point). Newton is
    // the line-searched frozen-multipole chord/quasi-Newton mixer that
    // converges that numerical probe; its step is taken
    // in-line below, not through the ChargeMixer hierarchy, because it
    // needs the charge map itself (FD Jacobian + line search).
    const double step = ctx.has_gam3
        ? std::min(charge_mixing, 0.1)
        : charge_mixing;
    std::unique_ptr<ChargeMixer> mixer;
    if (scc_mixer == SCCMixer::Broyden) {
        mixer = std::make_unique<BroydenMixer>(6, charge_mixing);
    } else if (scc_mixer == SCCMixer::BroydenEyert) {
        // The tblite/xtb Eyert scheme (broyden.f90): tblite sizes the
        // Broyden history to the full iteration budget (its default
        // memory is maxiter), not a sliding window; a shorter window
        // deviates from the xtb trajectory once the iteration count
        // passes it. The mixing parameter arrives validated in (0, 1];
        // the Python facade maps an unset value to xtb's bromix
        // default 0.4 for this mixer.
        mixer = std::make_unique<EyertBroydenMixer>(
            std::max(max_iter, 1), charge_mixing);
    } else if (scc_mixer == SCCMixer::DIIS) {
        mixer = std::make_unique<ChargeDIISMixer>(6, 0.5, 3);
    } else {
        mixer = std::make_unique<SimpleMixer>(step);
    }

    double occupation_entropy = 0.0;
    std::vector<double> max_change_trace;
    Eigen::VectorXd dq_new = Eigen::VectorXd::Zero(ctx.n_shells);
    Eigen::VectorXd V_mp = Eigen::VectorXd::Zero(ctx.n_shells);
    Eigen::MatrixXd H_scc = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_basis);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ctx.n_basis, ctx.n_basis);
    Eigen::VectorXd eps;
    Eigen::MatrixXd C;

    GFN2SECCMResult result;
    for (int iter = 1; iter <= max_iter; ++iter) {
        // One charge-map evaluation at the current shell iterate: the
        // shared arithmetic for the loop, the Newton mixer's FD
        // Jacobian, and its line search (bit-for-bit identical to the
        // shipped inlined map).
        xtb::GFN2AesPotentials aes_potentials;
        if (joint_state) {
            aes_potentials = xtb::gfn2_aes_potentials(
                ctx.positions, state.moments(ctx.shell_info),
                ctx.super.aes_params,
                detail::seccm_image_records(*ctx.topology_for_records, ctx.positions));
        }
        GFN2ChargeMapOutput map = gfn2_seccm_charge_map(
            ctx, dq_shell, have_moments ? &moments : nullptr, have_moments,
            electronic_temperature, iter,
            joint_state ? &aes_potentials : nullptr);
        dq_new = std::move(map.dq_new);
        H_scc = std::move(map.hamiltonian);
        D = std::move(map.density);
        eps = std::move(map.eps);
        C = std::move(map.coefficients);
        V_mp = std::move(map.V_mp);
        occupation_entropy = map.entropy;

        // Residual of the whole reduced state.  On the faithful path the
        // moments are part of the Hamiltonian, so a stationary charge vector
        // with drifting moments is not a converged SCC (issue #409).
        SeccmReducedState state_new;
        state_new.resize(ctx.n_shells, joint_state ? ctx.n_atoms : 0);
        if (joint_state) {
            state_new.set(dq_new, seccm_aes_moments(ctx, D));
        } else {
            state_new.set_dq(dq_new);
        }
        const double max_change =
            (state_new.x - state.x).cwiseAbs().maxCoeff();
        max_change_trace.push_back(max_change);

        if (max_change < ctx.conv_tol) {
            // Converged.  Evaluate the GFN2 energy functional at the
            // converged density's own Mulliken charges (molecular driver
            // convention): bare-H0 band + 2nd order + AES + 3rd + repulsion.
            dq_shell = dq_new;
            Eigen::MatrixXd DS = D * ctx.super.overlap;
            Eigen::VectorXd dq_atom = Eigen::VectorXd::Zero(ctx.n_atoms);
            for (int mu = 0; mu < ctx.n_basis; ++mu) {
                dq_atom(ctx.ao_atom[mu]) += DS(mu, mu);
            }
            for (int a = 0; a < ctx.n_atoms; ++a) {
                dq_atom(a) = static_cast<double>(
                    xtb::gfn2_valence_electrons(atoms[a].Z)) - dq_atom(a);
            }

            const double E_band0 = (D.cwiseProduct(ctx.super.h0)).sum();
            const double E_es =
                0.5 * dq_shell.dot(ctx.super.gamma_shell * dq_shell);
            // Faithful: the Eq. 25 functional of the converged density's
            // own record-resolved CAMM.  Ad-hoc: 1/2 dq . V_mp.
            const double E_aes = (ctx.include_aes && ctx.aes_faithful)
                ? xtb::gfn2_aes_energy(
                      ctx.positions, seccm_aes_moments(ctx, D),
                      ctx.super.aes_params,
                      detail::seccm_image_records(
                          *ctx.topology_for_records, ctx.positions))
                : 0.5 * dq_shell.dot(V_mp);
            double E_3rd = 0.0;
            for (int si = 0; si < ctx.n_shells; ++si) {
                const double G = ctx.gam3_shell(si);
                if (G == 0.0) continue;
                const double q = -dq_shell(si);
                E_3rd += G * q * q * q / 3.0;
            }
            const double E_rep = ctx.super.cyclic_repulsive;
            // Embedding self-energy at the converged charges (MSINDO CCM
            // convention: 1/2 dq . V_mad with the SMADEL subtraction).
            const double E_mad = ctx.madelung
                ? 0.5 * dq_atom.dot(effective_madelung_potential(
                      ctx, dq_atom))
                : 0.0;
            // Mermin -T*S entropy term at finite electronic temperature
            // (molecular run_gfn2_xtb convention; zero at T = 0).
            const double entropy_term =
                electronic_temperature * occupation_entropy;
            const double total = E_band0 + E_es + E_aes + E_3rd + E_mad
                + E_rep - entropy_term;
            const double normalization = static_cast<double>(ctx.group_order);

            result.energy = total / normalization;
            result.free_energy = result.energy;
            result.entropy = occupation_entropy / normalization;
            result.smearing_temperature = electronic_temperature;
            result.e_electronic =
                (E_band0 + E_es + E_aes + E_3rd + E_mad - entropy_term)
                / normalization;
            result.e_repulsive = E_rep / normalization;
            result.e_scc = E_es / normalization;
            result.e_band0 = E_band0 / normalization;
            result.e_aes = E_aes / normalization;
            result.e_3rd = E_3rd / normalization;
            result.e_madelung = E_mad / normalization;
            result.total_cyclic_energy = total;
            result.cyclic_electronic_energy =
                E_band0 + E_es + E_aes + E_3rd + E_mad - entropy_term;
            result.cyclic_repulsive_energy = E_rep;
            result.cyclic_scc_energy = E_es;
            result.cyclic_aes_energy = E_aes;
            result.cyclic_3rd_energy = E_3rd;
            result.cyclic_band0_energy = E_band0;
            result.cyclic_madelung_energy = E_mad;
            // Gap check before the eigenvalue vector moves out of scope.  A
            // rejected fixed point is returned as a normal, fully described
            // attempt so the outer ladder can retain its trace and budget.
            // The frontier gap is measured and recorded whenever a
            // frontier exists, at every electronic temperature: a smeared
            // run must not report a placeholder 0.0 for a state that is
            // in fact gapped.  Only the *rejection* is temperature-gated
            // -- finite temperature resolves the occupation of a
            // sub-epsilon frontier uniquely, so such a state is admitted
            // with the waiver recorded rather than rejected.
            bool gap_rejected = false;
            if (ctx.n_occ > 0 && ctx.n_occ < ctx.n_basis) {
                const double gap = detail::finite_torus_homo_lumo_gap(
                    eps, ctx.n_occ);
                result.homo_lumo_gap = gap;
                const bool gap_below_tolerance =
                    !std::isfinite(gap) || gap <= ctx.gap_tolerance;
                if (gap_below_tolerance
                    && (!std::isfinite(gap) || electronic_temperature <= 0.0)) {
                    gap_rejected = true;
                } else {
                    result.gap_guard_waived = gap_below_tolerance;
                }
            }
            result.mo_energies = std::move(eps);
            result.mo_coeffs = std::move(C);
            result.density = std::move(D);
            result.overlap = ctx.super.overlap;
            result.hamiltonian = std::move(H_scc);
            const bool physical = shell_populations_physical(ctx, dq_shell)
                && gfn2_seccm_local_charge_state_is_physical(
                    dq_atom, dq_shell);
            result.charges = std::move(dq_atom);
            result.shell_charges = dq_shell;
            result.physical_basin = physical;
            result.n_basis = ctx.n_basis;
            result.n_occ = ctx.n_occ;
            result.n_iter = iter;
            result.group_order = ctx.group_order;
            result.n_records = ctx.super.n_records;
            result.converged = !gap_rejected;
            std::string exit_reason = "converged";
            if (gap_rejected) {
                exit_reason = "gap_rejected";
            } else if (!physical) {
                exit_reason = "unphysical_basin";
            }
            stamp_supercell_attempt(
                result,
                max_iter,
                charge_mixing,
                electronic_temperature,
                scc_mixer,
                initial_shell_charges.size() != 0,
                max_change_trace,
                exit_reason);
            return result;
        }

        if (advisory_checkpoint && iter % kStabilityCheckpoint == 0
                && !seccm_still_contracting(max_change_trace)) {
            stalled_at_checkpoint = true;
            break;
        }

        // Not converged: advance the charges and refresh the shell
        // moments for the next iteration's AES potential (skipped when
        // the AES channel is disabled by the validation knob). The
        // Newton mixer replaces the linear mix with the line-searched
        // Newton step (IID 141's repelling fixed point); a rejected
        // step falls back to the damped simple-mixing advance, and the
        // moments then refresh from the pre-step density exactly like
        // the default path.
        if (joint_state) {
            // One mixer over the whole reduced state: charges and moments
            // advance together, which is what the shipped path could not do
            // and what xtb does (issue #409).  The Newton mixer is refused
            // for this state at entry, since its finite-difference Jacobian
            // is defined on the charge subspace only.
            mixer->mix(state.x, state_new.x, iter);
            dq_shell = state.dq_shell();
        } else if (scc_mixer == SCCMixer::Newton) {
            NewtonStepOutput nstep = gfn2_seccm_newton_step(
                ctx, dq_shell, dq_new, have_moments ? &moments : nullptr,
                have_moments, electronic_temperature, step, iter,
                max_change);
            dq_shell = std::move(nstep.dq_next);
            if (nstep.accepted) {
                D = std::move(nstep.density);
            }
            state.set_dq(dq_shell);
        } else {
            mixer->mix(dq_shell, dq_new, iter);
            state.set_dq(dq_shell);
        }
        if (ctx.include_aes && !joint_state) {
            moments = xtb::compute_shell_multipole_moments(
                D, basis, mol, ctx.mp_int, ctx.ao_shell,
                ctx.n0_shell, ctx.n_shells);
            moments.q = dq_shell;
            have_moments = true;
        }
    }

    // Failed exit: expose the last iterated state so the SCC map can be
    // analyzed from Python. With max_iter = 1 the loop built H_scc from
    // the neutral start, so hamiltonian is the supercell H0 exactly and
    // shell_charges is the raw first response f(0) = pop(H0) - n0.
    // Count executed iterations: a checkpoint stop ends the attempt early.
    result.n_iter = static_cast<int>(max_change_trace.size());
    result.smearing_temperature = electronic_temperature;
    result.converged = false;
    result.overlap = ctx.super.overlap;
    result.hamiltonian = std::move(H_scc);
    result.shell_charges = dq_new;
    Eigen::VectorXd dq_atom_last = Eigen::VectorXd::Zero(ctx.n_atoms);
    for (int si = 0; si < ctx.n_shells; ++si) {
        dq_atom_last(ctx.shell_info[si].atom_idx) -= dq_new(si);
    }
    const bool physical = shell_populations_physical(ctx, dq_new)
        && gfn2_seccm_local_charge_state_is_physical(
            dq_atom_last, dq_new);
    result.charges = std::move(dq_atom_last);
    result.n_basis = ctx.n_basis;
    result.n_occ = ctx.n_occ;
    result.group_order = ctx.group_order;
    result.n_records = ctx.super.n_records;
    result.physical_basin = physical;
    stamp_supercell_attempt(
        result,
        max_iter,
        charge_mixing,
        electronic_temperature,
        scc_mixer,
        initial_shell_charges.size() != 0,
        max_change_trace,
        stalled_at_checkpoint ? "stalled_checkpoint" : "iteration_limit");
    return result;
}

}  // namespace

bool gfn2_seccm_local_charge_state_is_physical(
    const Eigen::VectorXd& dq_atom,
    const Eigen::VectorXd& dq_shell) {
    // Shared calibration: vibeqc/semiempirical/core/basin_gates.hpp (#411).
    return charge_state_within_basin(dq_atom, dq_shell);
}

GFN2SECCMResult run_gfn2_seccm(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    const GFN2SECCMOptions& opts,
    double gap_tolerance) {
    const bool restart_requested = opts.initial_shell_charges.size() != 0;
    const bool molecular_delegation =
        detail::is_trivial_molecular_records(topology)
        && !opts.madelung && opts.include_aes;
    if (restart_requested) {
        if (!opts.initial_shell_charges.allFinite()) {
            throw std::invalid_argument(
                "GFN2-SECCM initial_shell_charges must contain only finite "
                "values");
        }
        const Eigen::Index expected_shell_charges =
            expected_gfn2_shell_charge_count(mol, params);
        if (expected_shell_charges >= 0) {
            if (opts.initial_shell_charges.size()
                != expected_shell_charges) {
                throw std::invalid_argument(
                    "GFN2-SECCM initial_shell_charges must be empty or have "
                    "exactly n_shells entries");
            }
            if (molecular_delegation) {
                throw std::invalid_argument(
                    "GFN2-SECCM initial_shell_charges are unavailable on a "
                    "molecular-delegation topology because the molecular "
                    "driver cannot consume a shell-charge restart");
            }
        }
    }

    detail::validate_common_inputs(
        mol,
        [&params](int Z) { return params.has_element(Z); },
        topology,
        group_order,
        geometry_tolerance,
        1.0e-10,
        gap_tolerance,
        "GFN2-SECCM",
        [](int) {},
        [](int, int) {});
    if (opts.max_iter < 1 || !std::isfinite(opts.conv_tol_charge)
        || opts.conv_tol_charge <= 0.0 || !std::isfinite(opts.charge_mixing)
        || opts.charge_mixing <= 0.0 || opts.charge_mixing > 1.0
        || !std::isfinite(opts.electronic_temperature)
        || opts.electronic_temperature < 0.0) {
        throw std::invalid_argument(
            "GFN2-SECCM options must be positive and finite, with "
            "charge_mixing in (0, 1] and electronic_temperature >= 0");
    }
    if (opts.madelung && opts.ewald_gamma) {
        throw std::invalid_argument(
            "GFN2-SECCM madelung and ewald_gamma are mutually exclusive: "
            "the Ewald-summed shell gamma subsumes the Coulomb tail");
    }
    if ((opts.madelung_s_weighted || opts.madelung_no_self)
        && !opts.madelung) {
        throw std::invalid_argument(
            "GFN2-SECCM madelung_s_weighted and madelung_no_self require "
            "madelung=true");
    }
    if (opts.ewald_gamma_molecular_onsite && !opts.ewald_gamma) {
        throw std::invalid_argument(
            "GFN2-SECCM ewald_gamma_molecular_onsite requires "
            "ewald_gamma=true");
    }

    const auto stamp_requested_options = [&opts, gap_tolerance](
        GFN2SECCMResult& result) {
        result.requested_electronic_temperature =
            opts.electronic_temperature;
        result.requested_madelung = opts.madelung;
        result.requested_madelung_s_weighted = opts.madelung_s_weighted;
        result.requested_madelung_no_self = opts.madelung_no_self;
        result.requested_ewald_gamma = opts.ewald_gamma;
        result.requested_include_aes = opts.include_aes;
        result.requested_ewald_gamma_molecular_onsite =
            opts.ewald_gamma_molecular_onsite;
        result.requested_max_iter = opts.max_iter;
        result.requested_conv_tol_charge = opts.conv_tol_charge;
        result.requested_charge_mixing = opts.charge_mixing;
        result.requested_scc_mixer = opts.scc_mixer;
        result.requested_ewald_gamma_k0_global =
            opts.ewald_gamma_k0_global;
        result.requested_finite_torus_gap_tolerance = gap_tolerance;
        result.requested_initial_shell_charges =
            opts.initial_shell_charges;
    };

    if (molecular_delegation && opts.scc_mixer == SCCMixer::Newton) {
        throw std::invalid_argument(
            "GFN2-SECCM scc_mixer='newton' is unavailable on an exact "
            "molecular-delegation topology; use the molecular simple, DIIS, "
            "or Broyden mixer");
    }
    if (molecular_delegation) {
        // The cyclic cluster is the ordinary molecule; the validated
        // molecular driver is the exact CCM result for the single-replica
        // torus (bit-for-bit parity by delegation).
        xtb::XTBSccOptions driver_options;
        driver_options.max_iter = opts.max_iter;
        driver_options.conv_tol_charge = opts.conv_tol_charge;
        driver_options.charge_mixing = opts.charge_mixing;
        driver_options.electronic_temperature = opts.electronic_temperature;
        driver_options.scc_mixer = opts.scc_mixer;
        // The faithful AES is a model choice, not a boundary choice, so the
        // delegated molecular solve must make the same one; otherwise the
        // single-replica torus would silently return the ad-hoc energy while
        // the caller asked for the Bannwarth model.
        driver_options.aes_faithful = opts.aes_faithful;
        driver_options.aes_damping = opts.aes_damping;
        const xtb::GFN2Result molecular =
            xtb::run_gfn2_xtb(mol, params, driver_options);

        if (!molecular.converged) {
            // Preserve the native fail-closed contract for an explicitly
            // capped molecular solve.  In particular, do not attempt entropy
            // or shell-population reconstruction from empty result matrices.
            GFN2SECCMResult failed;
            failed.n_iter = molecular.n_iter;
            failed.n_basis = molecular.n_basis;
            failed.n_occ = molecular.n_occ;
            failed.smearing_temperature =
                molecular.smearing_temperature;
            failed.group_order = group_order;
            failed.molecular_delegated = true;
            for (const auto& cell : topology.cells) {
                failed.n_records += static_cast<int>(cell.size());
            }
            failed.scc_max_change_trace = molecular.scc_max_change_trace;
            failed.attempts = molecular.attempts;
            for (auto& attempt : failed.attempts) {
                attempt.molecular_delegated = true;
            }
            failed.selected_attempt_index =
                molecular.selected_attempt_index;
            stamp_requested_options(failed);
            return failed;
        }

        // The molecular driver reports the temperature actually used by its
        // bounded stabilization ladder.  That can differ from the requested
        // value, so consume its thermodynamic metadata rather than relabeling
        // a finite-T retry as an Aufbau result.
        const double cyclic_entropy = molecular.entropy;
        const double actual_temperature = molecular.smearing_temperature;
        const double entropy_term = actual_temperature * cyclic_entropy;
        const double cyclic_free_energy = molecular.free_energy;
        const double cyclic_free_electronic =
            molecular.e_electronic - entropy_term;
        const double normalization = static_cast<double>(group_order);

        GFN2SECCMResult result;
        result.energy = cyclic_free_energy / normalization;
        result.free_energy = result.energy;
        result.entropy = cyclic_entropy / normalization;
        result.smearing_temperature = actual_temperature;
        result.e_electronic = cyclic_free_electronic / normalization;
        result.e_repulsive = molecular.e_repulsive / normalization;
        result.e_scc = molecular.e_scc / normalization;
        result.e_band0 = molecular.e_band0 / normalization;
        result.e_aes = molecular.e_aes / normalization;
        result.e_3rd = molecular.e_3rd / normalization;
        result.total_cyclic_energy = cyclic_free_energy;
        result.cyclic_electronic_energy = cyclic_free_electronic;
        result.cyclic_repulsive_energy = molecular.e_repulsive;
        result.cyclic_scc_energy = molecular.e_scc;
        result.cyclic_aes_energy = molecular.e_aes;
        result.cyclic_3rd_energy = molecular.e_3rd;
        result.cyclic_band0_energy = molecular.e_band0;
        result.mo_energies = molecular.mo_energies;
        result.mo_coeffs = molecular.mo_coeffs;
        result.density = molecular.density;
        result.overlap = molecular.overlap;
        result.hamiltonian = molecular.hamiltonian;
        result.charges = molecular.charges;

        // Reconstruct the shell-resolved diagnostic from the molecular
        // density.  The molecular result exposes atomic charges only, but the
        // SECCM contract exposes dq_shell = population - n0 on every path.
        BasisSet molecular_basis = SemiempiricalBasis::build(
            mol, params, 0);
        const std::vector<GFN2ShellInfo> shell_info =
            gfn2_enumerate_shells(molecular_basis, mol, params);
        std::vector<int> ao_shell(molecular.n_basis, -1);
        for (int si = 0; si < static_cast<int>(shell_info.size()); ++si) {
            for (int local = 0; local < shell_info[si].n_funcs; ++local) {
                ao_shell[shell_info[si].bf_start + local] = si;
            }
        }
        result.shell_charges = Eigen::VectorXd::Zero(shell_info.size());
        const Eigen::MatrixXd density_overlap =
            molecular.density * molecular.overlap;
        for (int mu = 0; mu < molecular.n_basis; ++mu) {
            result.shell_charges(ao_shell[mu]) += density_overlap(mu, mu);
        }
        for (int si = 0; si < static_cast<int>(shell_info.size()); ++si) {
            const int Z = mol.atoms()[shell_info[si].atom_idx].Z;
            result.shell_charges(si) -=
                xtb::gfn2_reference_occupation(Z, shell_info[si].l);
        }
        result.n_basis = molecular.n_basis;
        result.n_occ = molecular.n_occ;
        result.n_iter = molecular.n_iter;
        result.scc_max_change_trace = molecular.scc_max_change_trace;
        result.attempts = molecular.attempts;
        for (auto& attempt : result.attempts) {
            attempt.molecular_delegated = true;
        }
        result.selected_attempt_index = molecular.selected_attempt_index;
        result.group_order = group_order;
        for (const auto& cell : topology.cells) {
            result.n_records += static_cast<int>(cell.size());
        }
        result.converged = molecular.converged;
        result.molecular_delegated = true;
        result.physical_basin = molecular_shell_populations_physical(
            mol, shell_info, result.shell_charges)
            && gfn2_seccm_local_charge_state_is_physical(
                result.charges, result.shell_charges);
        // Same contract as the supercell engine above: always measure and
        // record the frontier gap; gate only the rejection on temperature.
        bool gap_rejected = false;
        if (molecular.n_occ > 0 && molecular.n_occ < molecular.n_basis) {
            result.homo_lumo_gap =
                molecular.mo_energies(molecular.n_occ)
                - molecular.mo_energies(molecular.n_occ - 1);
            const bool gap_below_tolerance =
                !std::isfinite(result.homo_lumo_gap)
                || result.homo_lumo_gap <= gap_tolerance;
            if (gap_below_tolerance
                && (!std::isfinite(result.homo_lumo_gap)
                    || actual_temperature <= 0.0)) {
                gap_rejected = true;
            } else {
                result.gap_guard_waived = gap_below_tolerance;
            }
        }
        // Validation precedence preserves the historical public gap error
        // when a numerically converged state violates both acceptance gates.
        if (result.selected_attempt_index >= 0
            && (gap_rejected || !result.physical_basin)) {
            auto& selected = result.attempts[static_cast<std::size_t>(
                result.selected_attempt_index)];
            selected.exit_reason = gap_rejected
                ? "gap_rejected" : "unphysical_basin";
            selected.scc_converged = true;
            selected.physical_basin = result.physical_basin;
            result.converged = false;
            result.selected_attempt_index = -1;
        }
        stamp_requested_options(result);
        return result;
    }

    // A finite WS sum is not a thermodynamic-limit regularization when the
    // remainder it truncates has a power-law tail: the Klopman-Ohno
    // remainder decays as -eta^2/2R^3 with pair-dependent amplitude, whose
    // charge-weighted 3-D sum need not cancel under neutrality.  That is a
    // property of the kernel, not of the boundary, so the refusal is scoped
    // to the kernel (#444).  The Elstner form's remainder is the
    // exponentially decaying -S(tau_a, tau_b, R), whose image sum converges
    // absolutely in three dimensions as it does in one and two, so the
    // route is available with it.  Do not repair the Klopman-Ohno route by
    // inventing a different GFN2 hardness mapping; select the form instead.
    if (opts.ewald_gamma && topology.translations.size() == 3
        && opts.gamma_form != ShellGammaForm::Elstner) {
        throw std::invalid_argument(
            "GFN2-SECCM 3-D ewald_gamma is unavailable with the Klopman-Ohno "
            "shell gamma on non-molecular cyclic topologies: its WS-folded "
            "remainder has no thermodynamic limit. Pass gamma_form=Elstner "
            "for a kernel that does (#444).");
    }

    // ---- Multi-replica WS-weighted supercell engine ----

    // n_primitives=0: GFN2-xTB per-element auto (H,He->3; else->4).
    BasisSet basis = SemiempiricalBasis::build(mol, params, 0);
    const int n_basis = static_cast<int>(basis.nbasis());
    const int n_atoms = static_cast<int>(mol.atoms().size());
    const auto& atoms = mol.atoms();

    SupercellContext ctx;
    ctx.mol = &mol;
    ctx.topology = &topology;
    ctx.group_order = group_order;
    ctx.gap_tolerance = gap_tolerance;
    ctx.conv_tol = opts.conv_tol_charge;
    ctx.n_basis = n_basis;
    ctx.n_atoms = n_atoms;
    ctx.shell_info = gfn2_enumerate_shells(basis, mol, params);
    ctx.n_shells = static_cast<int>(ctx.shell_info.size());
    ctx.super = assemble_gfn2_supercell(
        mol, params, topology, basis, ctx.shell_info, opts.ewald_gamma,
        opts.ewald_gamma_molecular_onsite, opts.gamma_form,
        opts.include_aes && opts.aes_faithful);

    if (!ctx.super.overlap.allFinite() || !ctx.super.h0.allFinite()
        || !ctx.super.gamma_shell.allFinite()) {
        throw std::runtime_error(
            "GFN2-SECCM matrix assembly produced non-finite values");
    }
    if (detail::max_abs(
            ctx.super.overlap - ctx.super.overlap.transpose()) > 1.0e-10
        || detail::max_abs(ctx.super.h0 - ctx.super.h0.transpose()) > 1.0e-10
        || detail::max_abs(
               ctx.super.gamma_shell - ctx.super.gamma_shell.transpose())
            > 1.0e-10) {
        throw std::runtime_error(
            "GFN2-SECCM reverse-image assembly is not Hermitian");
    }
    int n_val_elec = -mol.charge();
    for (int a = 0; a < n_atoms; ++a) {
        n_val_elec += xtb::gfn2_valence_electrons(atoms[a].Z);
    }
    ctx.n_occ = std::min(std::max(n_val_elec, 0) / 2, n_basis);
    // The WS-weighted cyclic overlap can have non-positive eigenvalues
    // (Peintinger-Bredow 2014, the documented C-point failure mode for
    // boundary-heavy cells). The paper's remedy is canonical
    // orthogonalization: screen the non-positive subspace and solve in the
    // transformed basis. Cells that pass the gate keep the legacy
    // generalized solve bit-for-bit; a cell whose screened subspace cannot
    // represent the occupied manifold still fails closed.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_solver(
        ctx.super.overlap);
    if (overlap_solver.info() != Eigen::Success
        || overlap_solver.eigenvalues().minCoeff() <= 1.0e-10) {
        ctx.ortho = detail::canonical_orthogonalizer(
            ctx.super.overlap, 1.0e-10, ctx.n_occ, "GFN2-SECCM");
        if (ctx.ortho.size() == 0) {
            throw std::runtime_error(
                "GFN2-SECCM overlap matrix is not positive definite");
        }
    }

    ctx.ao_shell.assign(n_basis, -1);
    for (int si = 0; si < ctx.n_shells; ++si) {
        for (int i = 0; i < ctx.shell_info[si].n_funcs; ++i) {
            ctx.ao_shell[ctx.shell_info[si].bf_start + i] = si;
        }
    }

    ctx.ao_atom.assign(n_basis, 0);
    for (int s = 0; s < static_cast<int>(basis.shells().size()); ++s) {
        const int nf = basis.libint().shells()[s].size();
        const int bf0 = basis.libint().shell2bf()[s];
        for (int i = 0; i < nf; ++i) {
            ctx.ao_atom[bf0 + i] = basis.shells()[s].atom_index;
        }
    }

    ctx.n0_shell = Eigen::VectorXd::Zero(ctx.n_shells);
    for (int si = 0; si < ctx.n_shells; ++si) {
        const int Z = atoms[ctx.shell_info[si].atom_idx].Z;
        ctx.n0_shell(si) =
            xtb::gfn2_reference_occupation(Z, ctx.shell_info[si].l);
    }

    // GFN2 globpar l-dependent GAM3 scaling (molecular driver values).
    static const double gam3_l_scale[4] = {1.0, 0.5, 0.25, 0.25};
    ctx.gam3_shell = Eigen::VectorXd::Zero(ctx.n_shells);
    for (int si = 0; si < ctx.n_shells; ++si) {
        const auto* e = params.element_data(
            atoms[ctx.shell_info[si].atom_idx].Z);
        const double gam3_raw = e ? e->gam3 : 0.0;
        const int l = ctx.shell_info[si].l;
        ctx.gam3_shell(si) = gam3_raw * gam3_l_scale[std::min(l, 3)];
    }
    ctx.has_gam3 = ctx.gam3_shell.cwiseAbs().maxCoeff() > 0.0;

    // ---- Opt-in Madelung/Ewald embedding state (shared kernels with the
    // SCC-DFTB-SECCM adapter: 1-D background-corrected wire Ewald, 2-D
    // Parry/Heyes, 3-D Ewald, all through indo::_madelung_potential_ewald
    // so the SMADEL short-range subtraction stays consistent).
    ctx.madelung = opts.madelung;
    ctx.madelung_s_weighted = opts.madelung_s_weighted;
    ctx.madelung_no_self = opts.madelung_no_self;
    ctx.include_aes = opts.include_aes;
    ctx.aes_faithful = opts.aes_faithful;
    ctx.aes_damping = opts.aes_damping;
    ctx.positions = detail::seccm_atom_coords(mol);
    ctx.topology_for_records = &topology;
    if (ctx.madelung) {
        auto madelung_state = detail::build_seccm_madelung_state(
            topology, n_atoms, "GFN2-SECCM");
        ctx.ews = std::move(madelung_state.ews);
        ctx.madkonst = std::move(madelung_state.madkonst);
    }

    if (ctx.include_aes) {
        ctx.mp_int = xtb::build_gfn2_multipole_integrals(basis, mol, true);
    }

    // max_iter is the total public SCC budget (molecular driver contract).
    // The stabilization ladder mirrors run_gfn2_xtb: one slow T=0 restart
    // and one mild finite-T restart, plus a high-T retry for post-Ar
    // elements, all bounded by the remaining budget. The primary attempt
    // may use the whole budget; it is handed to the ladder from a
    // budget-independent checkpoint when it stops contracting, so raising
    // max_iter never changes its trajectory (see seccm_still_contracting).
    const int total_max_iter = std::max(0, opts.max_iter);
    const bool use_stabilization =
        ctx.has_gam3 && total_max_iter >= 2500;
    GFN2SECCMResult result = run_supercell_scc(
        ctx,
        basis,
        opts.scc_mixer,
        opts.charge_mixing,
        total_max_iter,
        use_stabilization,
        opts.electronic_temperature,
        opts.initial_shell_charges);
    // The ladder also reacts to a converged-but-unphysical fixed point:
    // the over-polarized intra-atomic shell-transfer basin (see the
    // physical-basin gate above) is a fixed point of the SCC map, so the
    // ordinary non-convergence trigger never sees it. The same neutral
    // restarts that recover stiff insulators also escape that basin.
    const bool primary_accepted =
        result.converged && result.physical_basin;
    if (!primary_accepted && use_stabilization) {
        struct Retry {
            double charge_mixing;
            int max_iter;
            double electronic_temperature;
        };
        std::vector<Retry> retries = {
            {0.01, 2000, 0.0},
            {0.05, 500, 0.005},
        };
        if (has_extended_period_element(mol)) {
            retries.push_back({0.05, 600, 0.05});
        }
        for (const auto& retry : retries) {
            const int remaining_iters = total_max_iter - result.n_iter;
            if (remaining_iters <= 0) break;
            // Automatic recovery deliberately starts from the neutral charge
            // state. A caller-supplied restart is primary-attempt input, not a
            // persistent mutation of the reusable supercell context.
            GFN2SECCMResult retry_result = run_supercell_scc(
                ctx,
                basis,
                SCCMixer::Simple,
                retry.charge_mixing,
                std::min(retry.max_iter, remaining_iters),
                false,
                retry.electronic_temperature,
                Eigen::VectorXd());
            prepend_attempt_history(retry_result, result);
            if (retry_result.converged && retry_result.physical_basin) {
                stamp_requested_options(retry_result);
                return retry_result;
            }
            result = std::move(retry_result);
        }
    }
    // Fail closed on an unphysical fixed point: a converged over-polarized
    // state is a wrong answer, never a result. The ladder above already
    // searched for the physical basin when the budget allowed it.
    if (result.converged && !result.physical_basin) {
        result.converged = false;
    }
    stamp_requested_options(result);
    return result;
}

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
