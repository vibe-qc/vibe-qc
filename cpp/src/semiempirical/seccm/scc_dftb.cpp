#include "vibeqc/semiempirical/seccm/scc_dftb.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vibeqc/diis.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/periodic_gradient.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"
#include "vibeqc/semiempirical/methods/indo/ccm_engine.hpp"

#include "ewald_1d.h"
#include "seccm_common.h"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

namespace {

// One SCC-DFTB-SECCM charge-map evaluation: build H(dq) from the fixed
// supercell matrices (H0, S, gamma, and the opt-in Madelung embedding),
// solve the generalized eigenproblem, build the density (Fermi-Dirac
// occupations at the same temperature as the engine, or the hard-Aufbau
// T = 0 path), and return the Mulliken fluctuations. This is the map
// dq -> dq_new whose fixed point the SCF loop iterates and whose
// loop below calls it so every accelerator shares the same arithmetic.
// The `iter` argument only labels diagnostic exceptions.
// Record the occupation the accepted density was actually built from, and
// how far it is from the integer Aufbau occupation (issue 302).
//
// The T = 0 branch of the charge map builds D = 2 C_occ C_occ^T directly and
// never forms an occupation vector, so the Aufbau vector is reconstructed
// here rather than threaded through the map; the density is untouched either
// way.  On the smeared branch the vector is the Fermi-Dirac occupation the
// map applied.
//
// Weinert and Davenport, Phys. Rev. B 45, 13709 (1992), Eq. (8):
//   f_i = 1 / (exp(beta (eps_i - mu)) + 1)
// and Eq. (10'): the term that makes the fractional-occupation functional
// variational is (1/beta) sum_i [f_i ln f_i + (1 - f_i) ln(1 - f_i)], "the
// same as the entropy contribution to the free energy for noninteracting
// particles, -T S_s[n]" -- i.e. exactly the Mermin term this route
// subtracts.  So the Aufbau and fractional rows are stationary points of two
// different functionals, and
//   deviation = max_i |f_i - f_i^Aufbau|
// is zero exactly when they coincide.  That is why the record is this
// deviation and not a bound on the gap or on a charge: it needs no scale.
//
// The state this catches is the one the positive-gap guard structurally
// cannot see.  That guard compares the frontier gap to an absolute epsilon in
// Ha, while the scale that resolves an occupation on the smeared branch is
// kT; a gap far above the epsilon can still be a vanishing fraction of the
// smearing width, and the frontier is then equally occupied -- metallic --
// with the guard reporting an ordinary gapped row.
void record_applied_occupation(
    SCCDFTBSECCMResult& result,
    const Eigen::VectorXd& smeared_occupations,
    Eigen::Index n_orbitals,
    int n_occ,
    bool smeared) {
    if (n_orbitals <= 0) return;
    Eigen::VectorXd applied;
    if (smeared) {
        if (smeared_occupations.size() != n_orbitals) return;
        applied = smeared_occupations;
    } else {
        applied = Eigen::VectorXd::Zero(n_orbitals);
        applied.head(n_occ).setConstant(2.0);
    }
    double deviation = 0.0;
    for (Eigen::Index i = 0; i < applied.size(); ++i) {
        const double aufbau = i < static_cast<Eigen::Index>(n_occ) ? 2.0 : 0.0;
        deviation = std::max(deviation, std::abs(applied(i) - aufbau));
    }
    result.occupations = std::move(applied);
    result.aufbau_occupation_deviation = deviation;
    result.aufbau_occupation =
        deviation <= result.aufbau_occupation_tolerance;
}

struct SCCChargeMapOutput {
    Eigen::VectorXd dq_new;
    Eigen::MatrixXd density;
    Eigen::VectorXd eps;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd occupations;
    Eigen::MatrixXd hamiltonian;
    double entropy = 0.0;
};

SCCChargeMapOutput scc_seccm_charge_map(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const BasisSet& basis,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& ortho,
    const Eigen::MatrixXd& h0,
    const Eigen::MatrixXd& gamma,
    const Eigen::MatrixXd& madkonst,
    const std::vector<std::vector<indo::WSNeighbor>>& ews,
    const std::vector<detail::AOInfo>& ao_info,
    const Eigen::VectorXd& dq,
    int n_occ,
    int n_valence,
    const SCCDFTBSECCMOptions& opts,
    int iter) {
    const bool smeared = opts.electronic_temperature > 0.0;
    const int n_basis = static_cast<int>(basis.nbasis());

    // Every quadratic Mulliken-charge interaction has the same variational
    // derivative (Elstner et al., PRB 58, 7260, Eqs. 20-22):
    // H^SCC_mu,nu = H0_mu,nu - 1/2 S_mu,nu (V_A + V_B).
    Eigen::VectorXd potential = gamma * dq;
    if (opts.madelung) {
        const Eigen::VectorXd V_mad = indo::_madelung_potential_ewald(dq, madkonst, ews);
        if (!V_mad.allFinite()) {
            throw std::runtime_error(
                "SCC-DFTB-SECCM embedding potential produced "
                "non-finite values");
        }
        potential += V_mad;
    }
    Eigen::MatrixXd hamiltonian = h0;
    for (int mu = 0; mu < n_basis; ++mu) {
        const double V_mu = potential(ao_info[static_cast<std::size_t>(mu)].atom);
        for (int nu = 0; nu < n_basis; ++nu) {
            const double V_nu = potential(ao_info[static_cast<std::size_t>(nu)].atom);
            hamiltonian(mu, nu) -= 0.5 * overlap(mu, nu) * (V_mu + V_nu);
        }
    }

    if (!hamiltonian.allFinite() || !overlap.allFinite()) {
        throw std::runtime_error(
            "SCC-DFTB-SECCM Hamiltonian became non-finite at iter "
            + std::to_string(iter));
    }
    // Canonical-orthogonalized solve when the cyclic overlap needed
    // screening (Peintinger-Bredow 2014 remedy); otherwise the legacy
    // generalized solve, bit-for-bit the shipped arithmetic.
    SCCChargeMapOutput out;
    if (ortho.size() > 0) {
        const detail::CanonicalEigensolution canonical =
            detail::solve_canonical_orthogonalized(
                hamiltonian, ortho, "SCC-DFTB-SECCM");
        out.eps = canonical.eigenvalues;
        out.coefficients = canonical.coefficients;
    } else {
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            hamiltonian, overlap);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "SCC-DFTB-SECCM eigendecomposition failed at iter "
                + std::to_string(iter));
        }
        out.eps = solver.eigenvalues();
        out.coefficients = solver.eigenvectors();
    }
    if (smeared) {
        // Fractional Fermi-Dirac occupations through the shared kernel.
        // Every charge accelerator consumes this same finite-temperature
        // map, including its Fermi-window response.
        KPointOccupationOptions occupation_options;
        occupation_options.smearing_temperature =
            opts.electronic_temperature;
        auto occupation = compute_closed_shell_kpoint_occupations(
            {out.eps}, {1.0}, n_valence, n_occ, occupation_options);
        out.occupations = occupation.occupations_per_k.front();
        out.entropy = occupation.entropy;
        out.density = out.coefficients * out.occupations.asDiagonal()
            * out.coefficients.transpose();
    } else {
        const Eigen::MatrixXd occupied = out.coefficients.leftCols(n_occ);
        out.density = 2.0 * occupied * occupied.transpose();
    }

    // Gross Mulliken charges (Mulliken, J. Chem. Phys. 23, 1833 (1955),
    // Eqs. (6)/(6') and (8)):
    //   N(k) = sum_{mu in k} (D S)_{mu mu},   Q(k) = N_0(k) - N(k)
    // with N_0(k) the free-atom valence count.  These are NOT bounded by the
    // atom's shell capacity, and #302 must not be "fixed" by gating on them:
    // Sec. 3, p. 1835 of that paper records that a gross population ideally
    // "would never be less than zero" and for a single AO "should never
    // exceed the number 2.00 of electrons in a closed atomic sub-shell", but
    // that small negative values and slight excesses do occur, and that the
    // invariant quantity is the total, "necessarily an integer".  The total
    // is what mulliken_charges conserves; a per-atom excursion of a few
    // milli-electrons here is the partitioning, not a defective density.  On
    // the filed Mg/O rocksalt(100) L3 row the overlap is positive definite
    // (so no screening is engaged), tr(D S) is exact, and the natural
    // occupations of S^(1/2) D S^(1/2) lie inside [0, 2].  What that row
    // actually violates is the Aufbau occupation, which is recorded by
    // record_applied_occupation above.
    out.dq_new = SemiempiricalHamiltonianBuilder::mulliken_charges(
        basis, out.density, overlap, mol, params);
    out.hamiltonian = std::move(hamiltonian);
    return out;
}

}  // namespace

SCCDFTBSECCMResult run_scc_dftb_seccm(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    int group_order,
    const std::array<int, 3>& replicas,
    double geometry_tolerance,
    const SCCDFTBSECCMOptions& opts,
    double hermiticity_tolerance,
    double gap_tolerance,
    bool compute_gradient) {
    detail::validate_common_inputs(
        mol,
        [&params](int Z) { return params.has_element(Z); },
        topology,
        group_order,
        geometry_tolerance,
        hermiticity_tolerance,
        gap_tolerance,
        "SCC-DFTB-SECCM",
        [](int) {
            // Any element present in the parameter set is in scope; the
            // shared validator rejects missing elements itself.
        },
        [](int, int) {
            // The molecular SCC-DFTB driver accepts fallback repulsives
            // (R^-12 default from Hubbard U) for pairs without an explicit
            // entry; SECCM keeps that scope.
        },
        opts.madelung);
    if (opts.madelung && opts.ewald_gamma) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM madelung and ewald_gamma are mutually exclusive: "
            "the Ewald-summed gamma already carries the Coulomb tail");
    }
    // The refusal is a property of the kernel, not of the boundary (#211):
    // it is the Klopman-Ohno remainder's R^-3 tail, with a pair-dependent
    // amplitude, whose Wigner-Seitz truncation has no thermodynamic limit in
    // three dimensions.  The Elstner remainder decays exponentially and does,
    // so three dimensions are available with it.
    if (opts.madelung && topology.translations.size() == 3
        && opts.gamma_form != ShellGammaForm::Elstner) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM 3-D Madelung embedding is unavailable with the "
            "Klopman-Ohno gamma: its Ohno correction has no thermodynamic "
            "limit. Pass gamma_form=Elstner for a kernel that does (#211).");
    }
    int replica_order = 1;
    for (std::size_t axis = 0; axis < replicas.size(); ++axis) {
        const int replica = replicas[axis];
        if (replica <= 0) {
            throw std::invalid_argument(
                "SCC-DFTB-SECCM replicas must be positive");
        }
        if (axis >= topology.translations.size() && replica != 1) {
            throw std::invalid_argument(
                "SCC-DFTB-SECCM inactive replicas must equal one");
        }
        if (replica > group_order / replica_order) {
            throw std::invalid_argument(
                "SCC-DFTB-SECCM replica product must equal group order");
        }
        replica_order *= replica;
    }
    if (replica_order != group_order) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM replica product must equal group order");
    }
    if (topology.translations.size() == 3
        && std::any_of(
            replicas.begin(),
            replicas.end(),
            [](int replica) { return replica % 2 == 0; })) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM 3-D even-replica/Nyquist ownership is not "
            "validated; all active 3-D replica counts must be odd");
    }
    if (opts.max_iter < 1 || opts.conv_tol_charge <= 0.0
        || !std::isfinite(opts.conv_tol_charge)
        || !std::isfinite(opts.charge_mixing)
        || opts.charge_mixing <= 0.0 || opts.charge_mixing > 1.0) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM options must be positive and finite, with "
            "charge_mixing in (0, 1]");
    }
    if (opts.use_diis && (opts.diis_subspace < 2 || opts.diis_subspace > 12)) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM DIIS subspace must be between 2 and 12");
    }
    if (opts.use_broyden
        && (opts.broyden_memory < 1 || opts.broyden_memory > 20
            || !std::isfinite(opts.broyden_damping)
            || opts.broyden_damping <= 0.0 || opts.broyden_damping > 1.0)) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM Broyden options require memory in [1, 20] and "
            "damping in (0, 1]");
    }
    if (opts.use_diis && opts.use_broyden) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM DIIS and Broyden accelerators are mutually "
            "exclusive");
    }
    if (!std::isfinite(opts.electronic_temperature)
        || opts.electronic_temperature < 0.0) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM electronic_temperature must be finite and "
            "non-negative");
    }
    const bool smeared = opts.electronic_temperature > 0.0;

    const int n_valence =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);
    if (n_valence <= 0 || n_valence % 2 != 0) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM requires a positive even valence-electron count");
    }

    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    const int n_occ = n_valence / 2;
    if (n_valence > 2 * n_basis || n_occ >= n_basis) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM occupancy does not fit the finite-cluster AO "
            "space or leaves no LUMO");
    }
    const int n_atoms = static_cast<int>(mol.atoms().size());

    // WS-weighted supercell S and H0 (shared DFTB-family assembly).
    const std::pair<Eigen::MatrixXd, Eigen::MatrixXd> assembled =
        detail::assemble_weighted_dftb_h0(mol, params, topology, basis, n_basis);
    const Eigen::MatrixXd& overlap = assembled.first;
    const Eigen::MatrixXd& h0 = assembled.second;
    if (!overlap.allFinite() || !h0.allFinite()) {
        throw std::runtime_error(
            "SCC-DFTB-SECCM matrix assembly produced non-finite values");
    }
    if (detail::max_abs(overlap - overlap.transpose()) > hermiticity_tolerance
        || detail::max_abs(h0 - h0.transpose()) > hermiticity_tolerance) {
        throw std::runtime_error(
            "SCC-DFTB-SECCM reverse-image assembly is not Hermitian");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_solver(overlap);
    Eigen::MatrixXd ortho;
    if (overlap_solver.info() != Eigen::Success
        || overlap_solver.eigenvalues().minCoeff()
            <= hermiticity_tolerance) {
        // WS-weighted cyclic overlap with non-positive modes: screen the
        // non-positive subspace (Peintinger-Bredow 2014) instead of hard
        // failing; the kept subspace must still cover the occupied
        // manifold, otherwise the adapter fails closed as before.
        ortho = detail::canonical_orthogonalizer(
            overlap, hermiticity_tolerance, n_occ, "SCC-DFTB-SECCM");
    }

    // Supercell gamma: the Ewald-split kernel when the caller asks for the
    // embedded route, otherwise the untailed WS-weighted sum.
    Eigen::MatrixXd gamma = opts.ewald_gamma
        ? detail::ewald_dftb_gamma(mol, params, topology, opts.gamma_form)
        : detail::weighted_dftb_gamma(
              mol, params, topology, opts.gamma_form);
    if (!gamma.allFinite()) {
        throw std::runtime_error(
            "SCC-DFTB-SECCM gamma assembly produced non-finite values");
    }
    if (detail::max_abs(gamma - gamma.transpose()) > hermiticity_tolerance) {
        throw std::runtime_error(
            "SCC-DFTB-SECCM gamma assembly is not Hermitian");
    }

    // AO-to-atom map for the SCC potential contraction.
    const std::vector<detail::AOInfo> ao_info = detail::ao_info_for_basis(basis);

    // ---- Opt-in Madelung/Ewald embedding ----
    // The embedding couples to the supercell Mulliken fluctuations: the
    // Madelung potential of dq shifts the diagonal of H^SCC (same sign
    // convention as indo::run_ccm_core: F_add(k,k) = -V_mad) and its
    // classical self-energy 1/2 dq . V_mad joins the total energy. The
    // ews set mirrors indo::_ewald_ws_cells: WS records plus the atom
    // itself (weight 1, zero displacement).
    //
    // 1-D uses the converged background-corrected wire Ewald
    // (detail::wire_madkonst_1d, Parry-type; see ewald_1d.h), not the
    // truncated ±2-shell bare sum of the MSINDO kernel, which leaves a
    // nearly singular even-N charge response (documented period-2 limit
    // cycle). All three dimensions contract through
    // indo::_madelung_potential_ewald so the SMADEL short-range
    // subtraction stays consistent across dimensions.
    const int dim = static_cast<int>(topology.translations.size());
    std::vector<std::vector<indo::WSNeighbor>> ews;
    Eigen::MatrixXd madkonst;
    auto madelung_potential = [&](const Eigen::VectorXd& charges) {
        return indo::_madelung_potential_ewald(charges, madkonst, ews);
    };
    if (opts.madelung) {
        ews = indo::_ewald_ws_cells(topology);
        if (dim == 1) {
            madkonst = detail::wire_madkonst_1d(ews, topology.translations[0], n_atoms);
        } else if (dim == 2) {
            madkonst = indo::_madkonst_2d(ews, topology.translations, n_atoms);
        } else if (dim == 3) {
            madkonst = indo::_madkonst_3d(ews, topology.translations, n_atoms);
        }
        if (!madkonst.allFinite()) {
            throw std::runtime_error(
                "SCC-DFTB-SECCM Madelung-constant matrix produced "
                "non-finite values");
        }
    }

    // ---- Charge SCF loop (molecular run_scc_dftb pattern, T = 0) ----
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n_atoms);
    const double mix = opts.charge_mixing;
    double mix_cur = mix;
    Eigen::VectorXd resid_prev;

    // Fermi-occupation state (opt-in smearing): the occupations and entropy
    // come from the shared charge-map helper each iteration.
    Eigen::VectorXd occupations;
    double occupation_entropy = 0.0;

    SCCDFTBSECCMResult result;
    // The applied guard epsilon is recorded on every returned result --
    // converged or not -- so a consumer can reproduce the accept/reject
    // decision from the record alone.
    result.finite_torus_gap_tolerance = gap_tolerance;
    result.aufbau_occupation_tolerance = kAufbauOccupationTolerance;
    Eigen::VectorXd eps;
    Eigen::MatrixXd coefficients;
    Eigen::MatrixXd density;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd dq_new;

    // Pulay DIIS on the charge vector (periodic run_scc_dftb_gamma pattern).
    // The stiff charge response of an embedded charged supercell can leave
    // plain Aitken relaxation in a limit cycle; DIIS on (dq, resid) pairs
    // breaks it. Disabled by default for parity with the molecular
    // non-DIIS path; the Python wrapper enables it for madelung runs.
    std::unique_ptr<DIIS> charge_diis;
    if (opts.use_diis) {
        charge_diis = std::make_unique<DIIS>(opts.diis_subspace);
    }
    // Modified Broyden quasi-Newton mixer (core/charge_mixer.hpp). Operates
    // on the same (dq, resid) iteration as the Aitken/DIIS paths; the mixed
    // iterate is projected back onto the total-charge manifold exactly like
    // the DIIS extrapolation below.
    std::unique_ptr<BroydenMixer> charge_broyden;
    if (opts.use_broyden) {
        charge_broyden = std::make_unique<BroydenMixer>(
            opts.broyden_memory, opts.broyden_damping);
    }

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        // One charge-map evaluation: the exact arithmetic below lives in
        // the shared helper so every accelerator iterates the same map
        // (bit-for-bit parity pinned by the convergence regressions).
        SCCChargeMapOutput map = scc_seccm_charge_map(
            mol, params, basis, overlap, ortho, h0, gamma, madkonst, ews,
            ao_info, dq, n_occ, n_valence, opts, iter);
        eps = std::move(map.eps);
        coefficients = std::move(map.coefficients);
        occupations = std::move(map.occupations);
        density = std::move(map.density);
        hamiltonian = std::move(map.hamiltonian);
        occupation_entropy = map.entropy;
        dq_new = std::move(map.dq_new);

        Eigen::VectorXd resid = dq_new - dq;
        const double max_change = resid.cwiseAbs().maxCoeff();
        const bool converged = max_change < opts.conv_tol_charge;

        if (charge_broyden) {
            // Modified Broyden quasi-Newton step on the charge vector
            // (CP2K/tblite pattern), then projected back onto the
            // total-charge manifold exactly like the DIIS extrapolation.
            // Rejected extrapolations (non-finite or unphysically large
            // charge fluctuations) fall back to a conservative damped
            // residual step so the iteration never leaves the physical
            // basin. Broyden accelerates ordinary stiff cases (the 2-cell
            // embedded chain converges ~5x faster than Aitken); the
            // converged 1-D Ewald kernel (ewald_1d.h) has closed the
            // former even-replica charged-chain limit cycle.
            const Eigen::VectorXd dq_before = dq;
            charge_broyden->mix(dq, dq_new, iter);
            const double drift =
                (dq.sum() - static_cast<double>(mol.charge()))
                / static_cast<double>(n_atoms);
            dq.array() -= drift;
            bool physical = dq.allFinite();
            if (physical) {
                for (int a = 0; a < n_atoms; ++a) {
                    const int Z_a = mol.atoms()[a].Z;
                    const int n_val = params.valence_electrons(Z_a);
                    if (std::abs(dq(a))
                        > static_cast<double>(n_val) + 2.0) {
                        physical = false;
                        break;
                    }
                }
            }
            if (!physical) {
                dq = dq_before;
                dq += std::min(0.5, mix_cur) * resid;
            }
        } else if (charge_diis) {
            // DIIS extrapolation on the charge vector (periodic driver
            // pattern), damped by the current mixing fraction. Rejected
            // extrapolations fall back to the Aitken path below.
            Eigen::MatrixXd dq_mat = dq_new;
            Eigen::MatrixXd err_mat = resid;
            Eigen::MatrixXd dq_diis_mat =
                charge_diis->extrapolate(dq_mat, err_mat);

            bool diis_used = false;
            if (charge_diis->subspace_size() >= 2) {
                Eigen::VectorXd dq_candidate = dq_diis_mat.col(0);
                if (dq_candidate.allFinite()) {
                    bool physical = true;
                    for (int a = 0; a < n_atoms; ++a) {
                        int Z_a = mol.atoms()[a].Z;
                        int n_val = params.valence_electrons(Z_a);
                        if (std::abs(dq_candidate(a))
                            > static_cast<double>(n_val) + 2.0) {
                            physical = false;
                            break;
                        }
                    }
                    if (physical) {
                        dq += mix_cur * (dq_candidate - dq);
                        diis_used = true;
                    }
                }
            }

            if (diis_used) {
                // Enforce total charge conservation on the extrapolated
                // iterate.
                double drift =
                    (dq.sum() - static_cast<double>(mol.charge()))
                    / static_cast<double>(n_atoms);
                dq.array() -= drift;
            } else {
                // Fall back to vector Aitken relaxation for this step.
                if (resid_prev.size() > 0 && mix > 0.0) {
                    const Eigen::VectorXd delta_resid = resid - resid_prev;
                    const double denom = delta_resid.squaredNorm();
                    if (denom > 1e-24) {
                        const double estimate =
                            -mix_cur * resid_prev.dot(delta_resid) / denom;
                        if (std::isfinite(estimate) && estimate > 0.0) {
                            const double mix_floor = std::min(0.01, mix);
                            mix_cur = std::max(
                                mix_floor, std::min(mix, estimate));
                        } else if (resid.dot(resid_prev) < 0.0) {
                            mix_cur = std::max(
                                0.5 * mix_cur, std::min(0.01, mix));
                        }
                    }
                }
                dq += mix_cur * resid;
            }
            resid_prev = resid;
        } else {
            // Vector Aitken relaxation (Irons & Tuck 1969), identical to
            // the molecular non-DIIS path.
            if (resid_prev.size() > 0 && mix > 0.0) {
                const Eigen::VectorXd delta_resid = resid - resid_prev;
                const double denom = delta_resid.squaredNorm();
                if (denom > 1e-24) {
                    const double estimate =
                        -mix_cur * resid_prev.dot(delta_resid) / denom;
                    if (std::isfinite(estimate) && estimate > 0.0) {
                        const double mix_floor = std::min(0.01, mix);
                        mix_cur = std::max(mix_floor, std::min(mix, estimate));
                    } else if (resid.dot(resid_prev) < 0.0) {
                        mix_cur = std::max(0.5 * mix_cur, std::min(0.01, mix));
                    }
                }
            }
            resid_prev = resid;
            dq += mix_cur * resid;
        }

        if (!converged) continue;

        const double gap = eps(n_occ) - eps(n_occ - 1);
        // Positive-gap guard.  The finite torus must have a frontier gap
        // for its occupation to be well defined; the epsilon is the
        // `gap_tolerance` argument (1e-8 Ha by default -- seven orders
        // below any chemically meaningful gap, and far enough above the
        // ~1e-15 Ha degeneracies these tori produce that it separates a
        // physical small gap from a numerical zero).
        //
        // A non-finite frontier is never admissible.  At T = 0 the Aufbau
        // occupation of a sub-epsilon frontier is ambiguous, so the route
        // fails closed.  Finite electronic temperature does resolve the
        // occupation uniquely (Fermi-Dirac), so the requirement is waived
        // rather than fatal there -- but the waiver is recorded, because a
        // smeared degenerate frontier must not be indistinguishable from a
        // genuinely gapped state on the result record.
        const bool gap_below_tolerance =
            !std::isfinite(gap) || gap <= gap_tolerance;
        if (!std::isfinite(gap) || (!smeared && gap_below_tolerance)) {
            throw std::runtime_error(
                "SCC-DFTB-SECCM requires a positive finite-torus HOMO-LUMO "
                "gap");
        }
        result.gap_guard_waived = smeared && gap_below_tolerance;
        // Variational band energy tr(D H0) plus the Mermin -T*S entropy
        // term at finite electronic temperature (molecular run_scc_dftb
        // convention; zero at T = 0 so the Aufbau path is unchanged).
        const double cyclic_entropy_term =
            opts.electronic_temperature * occupation_entropy;
        const double cyclic_electronic = density.cwiseProduct(h0).sum();
        const double cyclic_scc = 0.5 * dq_new.dot(gamma * dq_new);
        const double cyclic_madelung = opts.madelung
            ? 0.5 * dq_new.dot(madelung_potential(dq_new))
            : 0.0;
        result.e_electronic = cyclic_electronic - cyclic_entropy_term;
        result.e_scc = cyclic_scc;
        result.e_madelung = cyclic_madelung;
        result.energy = cyclic_electronic - cyclic_entropy_term
            + cyclic_scc + cyclic_madelung;
        result.total_cyclic_energy = result.energy;
        result.cyclic_electronic_energy =
            cyclic_electronic - cyclic_entropy_term;
        result.cyclic_scc_energy = cyclic_scc;
        result.cyclic_madelung_energy = cyclic_madelung;
        result.homo_lumo_gap = gap;
        record_applied_occupation(result, occupations, eps.size(), n_occ,
                                  smeared);
        result.entropy = occupation_entropy;
        result.smearing_temperature = opts.electronic_temperature;
        result.mo_energies = std::move(eps);
        result.mo_coeffs = std::move(coefficients);
        result.density = std::move(density);
        result.overlap = overlap;
        result.hamiltonian = std::move(hamiltonian);
        result.charges = dq_new;
        result.n_basis = n_basis;
        result.n_occ = n_occ;
        result.n_iter = iter;
        result.group_order = group_order;
        result.converged = true;
        break;
    }

    if (!result.converged) {
        // Evaluate the functional consistently at the last-iterate density
        // (molecular run_scc_dftb convention: use the Mulliken charges of
        // that density, not the mixed iterate).
        const double cyclic_entropy_term =
            opts.electronic_temperature * occupation_entropy;
        const double cyclic_electronic = density.cwiseProduct(h0).sum();
        const double cyclic_scc = 0.5 * dq_new.dot(gamma * dq_new);
        const double cyclic_madelung = opts.madelung
            ? 0.5 * dq_new.dot(madelung_potential(dq_new))
            : 0.0;
        result.e_electronic = cyclic_electronic - cyclic_entropy_term;
        result.e_scc = cyclic_scc;
        result.e_madelung = cyclic_madelung;
        result.energy = cyclic_electronic - cyclic_entropy_term
            + cyclic_scc + cyclic_madelung;
        result.total_cyclic_energy = result.energy;
        result.cyclic_electronic_energy =
            cyclic_electronic - cyclic_entropy_term;
        result.cyclic_scc_energy = cyclic_scc;
        result.cyclic_madelung_energy = cyclic_madelung;
        record_applied_occupation(result, occupations, eps.size(), n_occ,
                                  smeared);
        result.entropy = occupation_entropy;
        result.smearing_temperature = opts.electronic_temperature;
        result.n_iter = opts.max_iter;
        // Diagnostic matrices from the last iterate.
        result.mo_energies = eps;
        result.mo_coeffs = coefficients;
        result.density = density;
        result.overlap = overlap;
        result.hamiltonian = hamiltonian;
        result.charges = dq_new;
        result.n_basis = n_basis;
        result.n_occ = n_occ;
        result.group_order = group_order;
    }

    // Repulsive energy (WS-weighted, like DFTB0-SECCM).
    const auto& atoms = mol.atoms();
    double cyclic_repulsive = 0.0;
    int n_records = 0;
    for (int central = 0; central < n_atoms; ++central) {
        for (const auto& image : topology.cells[central]) {
            cyclic_repulsive += 0.5 * image.weight
                * params.repulsive_energy(
                    atoms[central].Z,
                    atoms[image.origin].Z,
                    image.disp.norm());
            ++n_records;
        }
    }
    result.e_repulsive = cyclic_repulsive;
    result.cyclic_repulsive_energy = cyclic_repulsive;
    result.total_cyclic_energy += cyclic_repulsive;
    result.energy += cyclic_repulsive;

    // Normalize to per-primitive-cell energies. The total_cyclic_energy and
    // cyclic_* fields keep the raw finite-cluster totals (DFTB0-SECCM
    // convention).
    const double normalization = static_cast<double>(group_order);
    result.energy /= normalization;
    result.e_electronic /= normalization;
    result.e_repulsive /= normalization;
    result.e_scc /= normalization;
    result.e_madelung /= normalization;
    result.free_energy = result.energy;
    result.entropy /= normalization;
    result.n_records = n_records;
    if (compute_gradient) {
        if (!result.converged) {
            throw std::runtime_error(
                "SCC-DFTB-SECCM gradient requires a converged SCF");
        }
        result.gradient = compute_scc_dftb_seccm_gradient(
            mol, params, topology, result, group_order, geometry_tolerance,
            opts);
        result.has_gradient = true;
    }
    return result;
}

Eigen::MatrixXd compute_scc_dftb_seccm_gradient(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    const SCCDFTBSECCMResult& result,
    int group_order,
    double geometry_tolerance,
    const SCCDFTBSECCMOptions& opts) {
    detail::validate_common_inputs(
        mol,
        [&params](int Z) { return params.has_element(Z); },
        topology,
        group_order,
        geometry_tolerance,
        1.0e-10,
        1.0e-8,
        "SCC-DFTB-SECCM",
        [](int) {},
        [](int, int) {},
        opts.madelung);
    if (opts.madelung && opts.ewald_gamma) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM madelung and ewald_gamma are mutually exclusive: "
            "the Ewald-summed gamma already carries the Coulomb tail");
    }
    // The refusal is a property of the kernel, not of the boundary (#211):
    // it is the Klopman-Ohno remainder's R^-3 tail, with a pair-dependent
    // amplitude, whose Wigner-Seitz truncation has no thermodynamic limit in
    // three dimensions.  The Elstner remainder decays exponentially and does,
    // so three dimensions are available with it.
    if (opts.madelung && topology.translations.size() == 3
        && opts.gamma_form != ShellGammaForm::Elstner) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM 3-D Madelung embedding is unavailable with the "
            "Klopman-Ohno gamma: its Ohno correction has no thermodynamic "
            "limit. Pass gamma_form=Elstner for a kernel that does (#211).");
    }
    const int n_atoms = static_cast<int>(mol.atoms().size());
    if (result.group_order != group_order || result.n_basis <= 0
        || result.n_occ <= 0 || result.n_occ >= result.n_basis
        || result.density.rows() != result.n_basis
        || result.density.cols() != result.n_basis
        || result.mo_coeffs.rows() != result.n_basis
        || result.mo_coeffs.cols() != result.n_basis
        || result.mo_energies.size() != result.n_basis
        || result.charges.size() != n_atoms) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM gradient requires a matching finite-cluster "
            "result");
    }
    if (!result.density.allFinite() || !result.mo_coeffs.allFinite()
        || !result.mo_energies.allFinite() || !result.charges.allFinite()) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM gradient result matrices must be finite");
    }
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    if (static_cast<int>(basis.nbasis()) != result.n_basis) {
        throw std::invalid_argument(
            "SCC-DFTB-SECCM gradient basis does not match the energy result");
    }
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_info = basis.shells();
    std::vector<int> ao_atom(static_cast<std::size_t>(result.n_basis));
    for (std::size_t shell = 0; shell < shell_info.size(); ++shell) {
        for (std::size_t local = 0; local < shells[shell].size(); ++local) {
            ao_atom[shell2bf[shell] + local] = shell_info[shell].atom_index;
        }
    }
    const auto& atoms = mol.atoms();
    const double kappa = params.kappa();
    const int n_valence =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);

    // Energy-weighted density from the converged supercell MOs. At finite
    // electronic temperature the occupied weights are the Fermi-Dirac
    // occupations of the Mermin fixed point (recomputed through the same
    // kernel as the SCF loop; the occupation response itself is absorbed by
    // the Mermin stationarity, so no explicit entropy derivative appears in
    // the fixed-charge force). The T = 0 Aufbau path is unchanged.
    Eigen::MatrixXd energy_weighted = Eigen::MatrixXd::Zero(
        result.n_basis, result.n_basis);
    if (opts.electronic_temperature > 0.0) {
        KPointOccupationOptions occupation_options;
        occupation_options.smearing_temperature =
            opts.electronic_temperature;
        auto occupation = compute_closed_shell_kpoint_occupations(
            {result.mo_energies}, {1.0}, n_valence, result.n_occ,
            occupation_options);
        const Eigen::VectorXd occ = occupation.occupations_per_k.front();
        for (int orbital = 0; orbital < result.n_basis; ++orbital) {
            energy_weighted += result.mo_energies(orbital) * occ(orbital)
                * (result.mo_coeffs.col(orbital)
                   * result.mo_coeffs.col(orbital).transpose());
        }
    } else {
        const Eigen::MatrixXd occupied =
            result.mo_coeffs.leftCols(result.n_occ);
        for (int orbital = 0; orbital < result.n_occ; ++orbital) {
            energy_weighted += result.mo_energies(orbital)
                * (2.0 * occupied.col(orbital)
                   * occupied.col(orbital).transpose());
        }
    }

    // WS-weighted supercell gamma and total SCC potential, rebuilt
    // identically to the energy engine. The Madelung channel is another
    // quadratic Mulliken-charge interaction and therefore joins the same
    // overlap-weighted variational operator.
    const Eigen::MatrixXd gamma = opts.ewald_gamma
        ? detail::ewald_dftb_gamma(mol, params, topology, opts.gamma_form)
        : detail::weighted_dftb_gamma(
              mol, params, topology, opts.gamma_form);
    Eigen::VectorXd potential = gamma * result.charges;
    const int dim = static_cast<int>(topology.translations.size());
    std::vector<std::vector<indo::WSNeighbor>> ews;
    Eigen::MatrixXd madkonst;
    if (opts.madelung) {
        ews = indo::_ewald_ws_cells(topology);
        if (dim == 1) {
            madkonst = detail::wire_madkonst_1d(
                ews, topology.translations[0], n_atoms);
        } else {
            madkonst = indo::_madkonst_2d(
                ews, topology.translations, n_atoms);
        }
        const Eigen::VectorXd V_mad =
            indo::_madelung_potential_ewald(
                result.charges, madkonst, ews);
        if (!V_mad.allFinite()) {
            throw std::runtime_error(
                "SCC-DFTB-SECCM embedding potential produced "
                "non-finite values in the gradient");
        }
        potential += V_mad;
    }

    // M matrix, molecular compute_scc_dftb_gradient convention:
    // M = -W + D . 1/2 (kappa hbar_avg - V_A - V_B) on cross-atom pairs.
    // The Elstner PRB 58, 7260 (1998) Eq. 25 sign reasoning carries over
    // unchanged from the molecular driver (dq > 0 = cation convention).
    Eigen::MatrixXd effective = -energy_weighted;
    for (int mu = 0; mu < result.n_basis; ++mu) {
        const int central = ao_atom[static_cast<std::size_t>(mu)];
        for (int nu = 0; nu < result.n_basis; ++nu) {
            const int origin = ao_atom[static_cast<std::size_t>(nu)];
            if (central == origin) continue;
            const double hbar_avg = 0.5
                * (params.average_on_site(atoms[central].Z)
                   + params.average_on_site(atoms[origin].Z));
            effective(mu, nu) += result.density(mu, nu) * 0.5
                * (kappa * hbar_avg - potential(central) - potential(origin));
        }
    }

    // Scatter M into the WS-record blocks (DFTB0-SECCM gradient pattern).
    using detail::ShellKey;
    std::vector<ShellKey> shell_keys{{0, 0, 0}};
    for (const auto& cell : topology.cells) {
        for (const auto& image : cell) {
            if (std::find(
                    shell_keys.begin(),
                    shell_keys.end(),
                    image.image_shell_label)
                == shell_keys.end()) {
                shell_keys.push_back(image.image_shell_label);
            }
        }
    }
    LatticeMatrixSet weighted_effective;
    weighted_effective.nbf = result.n_basis;
    weighted_effective.cells.reserve(shell_keys.size());
    weighted_effective.blocks.assign(
        shell_keys.size(),
        Eigen::MatrixXd::Zero(result.n_basis, result.n_basis));
    std::map<ShellKey, std::size_t> block_by_label;
    for (std::size_t index = 0; index < shell_keys.size(); ++index) {
        LatticeCell cell;
        cell.index = Eigen::Vector3i(
            shell_keys[index][0],
            shell_keys[index][1],
            shell_keys[index][2]);
        cell.r_cart = detail::shell_translation(
            shell_keys[index], topology.translations);
        weighted_effective.cells.push_back(cell);
        block_by_label.emplace(shell_keys[index], index);
    }
    for (int mu = 0; mu < result.n_basis; ++mu) {
        const int central = ao_atom[static_cast<std::size_t>(mu)];
        for (int nu = 0; nu < result.n_basis; ++nu) {
            const int origin = ao_atom[static_cast<std::size_t>(nu)];
            if (central == origin) continue;
            for (const auto& image : topology.cells[central]) {
                if (image.origin != origin) continue;
                const auto block = block_by_label.at(image.image_shell_label);
                weighted_effective.blocks[block](mu, nu) =
                    -image.weight * effective(mu, nu);
            }
        }
    }

    PeriodicSystem system;
    system.dim = static_cast<int>(topology.translations.size());
    system.unit_cell = atoms;
    system.charge = mol.charge();
    system.multiplicity = mol.multiplicity();
    for (std::size_t axis = 0; axis < topology.translations.size(); ++axis) {
        system.lattice.col(static_cast<Eigen::Index>(axis)) =
            topology.translations[axis];
    }
    LatticeSumOptions unused_options;
    Eigen::MatrixXd gradient = overlap_lattice_gradient_contribution(
        basis, system, weighted_effective, unused_options);

    // WS-weighted repulsive derivative (DFTB0-SECCM pattern).
    for (int central = 0; central < n_atoms; ++central) {
        for (const auto& image : topology.cells[central]) {
            const double distance = image.disp.norm();
            const double derivative = params.repulsive_derivative(
                atoms[central].Z, atoms[image.origin].Z, distance);
            if (derivative == 0.0 || distance < 1.0e-12) continue;
            const Eigen::Vector3d contribution =
                0.5 * image.weight * derivative * image.disp / distance;
            gradient.row(central) -= contribution.transpose();
            gradient.row(image.origin) += contribution.transpose();
        }
    }

    // SCC gamma derivative: dE_scc/dR = 1/2 dq . dgamma/dR . dq over the
    // live record displacements. gamma(a,b) accumulates weighted screened
    // Coulomb kernels g(R) = 1/sqrt(R^2 + eta^2) with dg/dr_a = -d g^3,
    // dg/dr_b = +d g^3 (d = image displacement, r_central - r_image); the
    // 1/2 of E_scc = 1/2 dq.gamma.dq is cancelled by the two symmetric
    // appearances, and both ends are deposited per record (self-images
    // cancel exactly: their displacement is rigid under
    // rebuild_displacements).
    //
    // The kernel is whatever the energy used, so the derivative asks the
    // shared kernel for it rather than hard-coding one form.  Writing it
    // through shell_gamma_pair_derivative keeps the Klopman-Ohno path
    // bit-identical: there dgamma/dR = -R g^3, and -dgamma/dR / R = g^3, the
    // factor this loop used to write out.
    ShellGammaSpec gamma_spec;
    gamma_spec.form = opts.gamma_form;
    gamma_spec.ko_average = KlopmanOhnoAverage::InverseHardnessMean;
    if (opts.ewald_gamma) {
        // Embedded route: the Ewald half has its own derivative, so the
        // whole term comes from the kernel that assembled the matrix.
        const std::vector<Eigen::Vector3d> coords =
            detail::seccm_atom_coords(mol);
        const EwaldCoulombKernel ewald(
            topology.translations, detail::seccm_ewald_alpha(topology));
        gradient += periodic_shell_gamma_gradient(
            detail::dftb_gamma_sites(mol, params), coords, gamma_spec, ewald,
            detail::seccm_image_records(topology, coords), result.charges);
    } else {
    for (int central = 0; central < n_atoms; ++central) {
        const int Z_a = atoms[central].Z;
        double Ua = params.hubbard_u(Z_a);
        if (Ua <= 0.0) Ua = 0.4;
        for (const auto& image : topology.cells[central]) {
            const int origin = image.origin;
            const int Z_b = atoms[origin].Z;
            double Ub = params.hubbard_u(Z_b);
            if (Ub <= 0.0) Ub = 0.4;
            const double R = image.disp.norm();
            if (R < 1.0e-12) continue;
            // E_scc = 1/2 dq.gamma.dq: each directed WS record contributes
            // +- 1/2 w dq_a dq_b (-dgamma/dR / R) d to its two ends (the
            // symmetric reverse record supplies the matching half). disp
            // points from the central atom to the image (r_origin -
            // r_central), so dR/dr_central = -disp/R.
            const double radial =
                -shell_gamma_pair_derivative(gamma_spec, Ua, Ub, R) / R;
            const Eigen::Vector3d contribution = 0.5 * image.weight
                * result.charges(central) * result.charges(origin)
                * radial * image.disp;
            gradient.row(central) += contribution.transpose();
            gradient.row(origin) -= contribution.transpose();
        }
    }
    }

    if (opts.madelung) {
        // The total SCC Hamiltonian is now the variational derivative of
        // 1/2 dq . (gamma + M_eff) . dq, so the self-consistent charge
        // response cancels by stationarity. Only the explicit fixed-charge
        // derivative 1/2 dq . dM_eff/dR . dq remains.
        //
        // M_eff = madkonst - SMADEL (the 1/d subtraction inside
        // _madelung_potential_ewald). Each directed record contributes
        // +-0.5 w dq_i dq_o gh to its two ends. Self-images carry zero
        // displacement and contribute nothing.
        if (dim == 1) {
            for (int central = 0; central < n_atoms; ++central) {
                for (const auto& image : topology.cells[central]) {
                    const int origin = image.origin;
                    const double d = image.disp.norm();
                    Eigen::Vector3d gh = -detail::wire_kernel_gradient(
                        -image.disp, topology.translations[0]);
                    if (d > 1.0e-12) gh += image.disp / (d * d * d);
                    const Eigen::Vector3d contribution = 0.5 * image.weight
                        * result.charges(central)
                        * result.charges(origin) * gh;
                    gradient.row(central) -= contribution.transpose();
                    gradient.row(origin) += contribution.transpose();
                }
            }
        } else {
            // The 2-D Ewald matrix depends on geometry through the live
            // record displacements. Reuse the validated CCM deposit, which
            // includes the reciprocal-space and SMADEL derivatives.
            std::vector<std::array<double, 3>> mad_grad(
                static_cast<std::size_t>(n_atoms), {0.0, 0.0, 0.0});
            std::vector<Eigen::Vector3d> coords;
            coords.reserve(static_cast<std::size_t>(n_atoms));
            for (const auto& atom : atoms) {
                coords.emplace_back(atom.xyz[0], atom.xyz[1], atom.xyz[2]);
            }
            indo::_add_madelung_gradient(
                mad_grad, result.charges, coords, topology.translations,
                topology);
            for (int a = 0; a < n_atoms; ++a) {
                gradient(a, 0) += mad_grad[static_cast<std::size_t>(a)][0];
                gradient(a, 1) += mad_grad[static_cast<std::size_t>(a)][1];
                gradient(a, 2) += mad_grad[static_cast<std::size_t>(a)][2];
            }
        }
    }
    gradient /= static_cast<double>(group_order);
    if (!gradient.allFinite()) {
        throw std::runtime_error(
            "SCC-DFTB-SECCM gradient produced non-finite values");
    }
    return gradient;
}

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
