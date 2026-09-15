#include "vibeqc/semiempirical/methods/xtb/periodic_gfn2.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/basin_gates.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

namespace {

bool is_noble_gas(int Z) {
    return Z == 2 || Z == 10 || Z == 18 || Z == 36 || Z == 54 || Z == 86;
}

// A nonzero image label alone is not a representation-invariant periodicity
// test: an isolated molecule straddling a cell face moves one of its internal
// bonds from g=0 to g!=0.  The pair graph is genuinely periodic only when its
// image labels contain a noncontractible cycle.  Equivalently, if integer
// representatives n_A can make every in-cutoff edge satisfy n_B=n_A+g, all
// edges belong to finite clusters and the ordinary periodic route must retain
// the molecular-limit fail-closed guard.
bool periodic_pair_graph_spans_lattice(
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    double cutoff_bohr) {
    struct PairImageEdge {
        int atom_index;
        Eigen::Vector3i image_index;
    };

    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());
    std::vector<std::vector<PairImageEdge>> adjacency(
        static_cast<std::size_t>(n_atoms));
    for (const auto& cell : cells) {
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = 0; b < n_atoms; ++b) {
                if (atom_pair_in_cutoff(
                        atoms[static_cast<std::size_t>(a)],
                        atoms[static_cast<std::size_t>(b)],
                        cell.r_cart, cutoff_bohr)) {
                    adjacency[static_cast<std::size_t>(a)].push_back(
                        {b, cell.index});
                }
            }
        }
    }

    std::vector<Eigen::Vector3i> representative_image(
        static_cast<std::size_t>(n_atoms), Eigen::Vector3i::Zero());
    std::vector<bool> assigned(static_cast<std::size_t>(n_atoms), false);
    std::vector<int> stack;
    for (int root = 0; root < n_atoms; ++root) {
        if (assigned[static_cast<std::size_t>(root)]) continue;
        assigned[static_cast<std::size_t>(root)] = true;
        stack.push_back(root);
        while (!stack.empty()) {
            const int a = stack.back();
            stack.pop_back();
            for (const PairImageEdge& edge :
                 adjacency[static_cast<std::size_t>(a)]) {
                const Eigen::Vector3i expected =
                    representative_image[static_cast<std::size_t>(a)]
                    + edge.image_index;
                const std::size_t b =
                    static_cast<std::size_t>(edge.atom_index);
                if (!assigned[b]) {
                    representative_image[b] = expected;
                    assigned[b] = true;
                    stack.push_back(edge.atom_index);
                } else if ((representative_image[b].array()
                            != expected.array()).any()) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Physical-basin gate for the periodic SCC fixed point. Same calibrated
// bounds and rationale as the SECCM adapter's gate, shared in one place
// (vibeqc/semiempirical/core/basin_gates.hpp, issue #411): the GFN2
// third-order term admits a spurious over-polarized intra-atomic
// shell-transfer fixed point. A converged-but-unphysical state is a wrong
// answer, never a result.
bool periodic_gfn2_charge_state_physical(
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::VectorXd& n0_shell,
    const Eigen::VectorXd& dq_shell,
    const Eigen::VectorXd& dq_atom) {
    if (!shell_populations_within_basin(shell_info, n0_shell, dq_shell)) {
        return false;
    }
    return charge_state_within_basin(dq_atom, dq_shell);
}

// Reduced SCC state: shell charge fluctuations, CAMM dipoles, CAMM
// traceless quadrupoles, packed as one vector [dq; mu; theta] so the
// charge mixers act on charges and moments together (the tblite state
// vector; issue #409).
struct ReducedState {
    int n_shells = 0;
    int n_atoms = 0;
    Eigen::VectorXd x;

    static int size(int n_shells, int n_atoms) {
        return n_shells + 9 * n_atoms;
    }
    void resize(int ns, int na) {
        n_shells = ns;
        n_atoms = na;
        x = Eigen::VectorXd::Zero(size(ns, na));
    }
    Eigen::VectorXd dq_shell() const { return x.head(n_shells); }
    Eigen::MatrixXd mu() const {
        Eigen::MatrixXd m(n_atoms, 3);
        for (int a = 0; a < n_atoms; ++a)
            for (int k = 0; k < 3; ++k) m(a, k) = x(n_shells + 3 * a + k);
        return m;
    }
    Eigen::MatrixXd theta() const {
        Eigen::MatrixXd t(n_atoms, 6);
        const int offset = n_shells + 3 * n_atoms;
        for (int a = 0; a < n_atoms; ++a)
            for (int k = 0; k < 6; ++k) t(a, k) = x(offset + 6 * a + k);
        return t;
    }
    void set(const Eigen::VectorXd& dq, const GFN2AesMoments& moments) {
        x.head(n_shells) = dq;
        for (int a = 0; a < n_atoms; ++a) {
            for (int k = 0; k < 3; ++k) x(n_shells + 3 * a + k) = moments.mu(a, k);
            for (int k = 0; k < 6; ++k)
                x(n_shells + 3 * n_atoms + 6 * a + k) = moments.theta(a, k);
        }
    }
};

}  // namespace

namespace {

ImageRecordSource pair_cutoff_records(
    const std::vector<Eigen::Vector3d>& positions_copy,
    const std::vector<LatticeCell>& cells_copy,
    double cutoff) {
    return [positions_copy, cells_copy, cutoff](const ImageRecordSink& sink) {
        const int n = static_cast<int>(positions_copy.size());
        const double cutoff_sq = cutoff * cutoff;
        for (const auto& cell : cells_copy) {
            const bool zero_cell = (cell.index.array() == 0).all();
            for (int a = 0; a < n; ++a) {
                for (int b = 0; b < n; ++b) {
                    if (zero_cell && a == b) continue;
                    const Eigen::Vector3d d = positions_copy[static_cast<std::size_t>(b)]
                        + cell.r_cart - positions_copy[static_cast<std::size_t>(a)];
                    if (d.squaredNorm() > cutoff_sq) continue;
                    ImageRecord record;
                    record.a = a;
                    record.b = b;
                    record.shift = cell.r_cart;
                    record.weight = 1.0;
                    sink(record);
                }
            }
        }
    };
}

}  // namespace

ImageRecordSource PeriodicGFN2Assembly::records() const {
    return pair_cutoff_records(positions, cells, cutoff_bohr);
}

ImageRecordSource PeriodicGFN2Assembly::gamma_records() const {
    return pair_cutoff_records(positions, gamma_cells, gamma_cutoff_bohr);
}

EwaldCoulombKernel PeriodicGFN2Assembly::ewald() const {
    return EwaldCoulombKernel(lattice_translations);
}

PeriodicGFN2Assembly assemble_periodic_gfn2(
    const PeriodicSystem& system,
    const GFN2ParameterSet& params,
    double cutoff_bohr,
    ShellGammaForm gamma_form) {
    PeriodicGFN2Assembly A{
        system.unit_cell_molecule(),
        SemiempiricalBasis::build(system.unit_cell_molecule(), params, 0)};
    A.cutoff_bohr = cutoff_bohr;
    A.gamma_spec.form = gamma_form;
    A.gamma_spec.ko_average = KlopmanOhnoAverage::HardnessMean;
    for (int axis = 0; axis < system.dim; ++axis) {
        A.lattice_translations.push_back(system.lattice.col(axis));
    }

    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    A.cells = atom_pair_interaction_cells(system, cutoff_bohr);
    require_nonzero_lattice_image(A.cells, cutoff_bohr, "run_gfn2_xtb_gamma");
    if (!periodic_pair_graph_spans_lattice(system, A.cells, cutoff_bohr)) {
        require_nonzero_lattice_image({}, cutoff_bohr, "run_gfn2_xtb_gamma");
    }

    const auto& atoms = system.unit_cell;
    A.n_atoms = static_cast<int>(atoms.size());
    A.n_basis = static_cast<int>(A.basis.nbasis());
    A.shell_info = gfn2_enumerate_shells(A.basis, A.mol, params);
    A.n_shells = static_cast<int>(A.shell_info.size());
    A.positions.reserve(atoms.size());
    for (const auto& atom : atoms) {
        A.positions.emplace_back(atom.xyz[0], atom.xyz[1], atom.xyz[2]);
    }

    A.ao_shell.assign(A.n_basis, -1);
    A.ao_atom.assign(A.n_basis, 0);
    for (int si = 0; si < A.n_shells; ++si) {
        for (int i = 0; i < A.shell_info[si].n_funcs; ++i) {
            const int mu = A.shell_info[si].bf_start + i;
            A.ao_shell[mu] = si;
            A.ao_atom[mu] = A.shell_info[si].atom_idx;
        }
    }
    A.gamma_sites.reserve(A.shell_info.size());
    for (const auto& shell : A.shell_info) {
        A.gamma_sites.push_back({shell.atom_idx,
                                 shell.hardness > 0.0 ? shell.hardness : 0.5});
    }

    A.n_valence_electrons = -system.charge;
    for (const auto& atom : atoms) {
        A.n_valence_electrons += gfn2_valence_electrons(atom.Z);
    }
    A.n_occ = std::min(std::max(A.n_valence_electrons, 0) / 2, A.n_basis);

    A.n0_shell = Eigen::VectorXd::Zero(A.n_shells);
    // GFN2 globpar l-dependent GAM3 scales: s=1.0, p=0.5, d=0.25.
    static const double gam3_l_scale[4] = {1.0, 0.5, 0.25, 0.25};
    A.gam3_shell = Eigen::VectorXd::Zero(A.n_shells);
    for (int si = 0; si < A.n_shells; ++si) {
        const int Z = atoms[A.shell_info[si].atom_idx].Z;
        A.n0_shell(si) = gfn2_reference_occupation(Z, A.shell_info[si].l);
        const auto* e = params.element_data(Z);
        A.gam3_shell(si) = (e ? e->gam3 : 0.0)
            * gam3_l_scale[std::min(A.shell_info[si].l, 3)];
    }
    A.has_gam3 = A.gam3_shell.cwiseAbs().maxCoeff() > 0.0;

    // Overlap and H0 image blocks on the interaction set (*).
    A.overlap_blocks = compute_overlap_lattice_explicit(A.basis, system, A.cells);
    mask_blocks_beyond_pair_cutoff(
        A.overlap_blocks.blocks, A.cells, A.ao_atom, atoms, cutoff_bohr);
    A.coordination_numbers = gfn2_h0_periodic_coordination_numbers(system);
    A.S_gamma = Eigen::MatrixXd::Zero(A.n_basis, A.n_basis);
    A.H0_gamma = Eigen::MatrixXd::Zero(A.n_basis, A.n_basis);
    for (std::size_t ci = 0; ci < A.cells.size(); ++ci) {
        A.S_gamma += A.overlap_blocks.blocks[ci];
        A.H0_gamma += build_gfn2_hamiltonian_zero_image_with_cn(
            A.basis, A.overlap_blocks.blocks[ci], A.mol, params,
            A.cells[ci].r_cart, A.coordination_numbers);
    }

    // Lattice-summed shell gamma: Ewald point Coulomb plus the short-range
    // remainder over its own converged pair range, on-site block from the
    // parameters.
    A.gamma_cutoff_bohr = cutoff_bohr;
    for (std::size_t i = 0; i < A.gamma_sites.size(); ++i) {
        for (std::size_t j = i; j < A.gamma_sites.size(); ++j) {
            const double range = shell_gamma_remainder_range(
                A.gamma_spec, A.gamma_sites[i].hardness,
                A.gamma_sites[j].hardness, kGammaRemainderTolerance);
            if (std::isfinite(range)) {
                A.gamma_cutoff_bohr = std::max(A.gamma_cutoff_bohr, range);
            }
        }
    }
    A.gamma_cells = A.gamma_cutoff_bohr > cutoff_bohr
        ? atom_pair_interaction_cells(system, A.gamma_cutoff_bohr)
        : A.cells;
    A.gamma_shell = build_periodic_shell_gamma(
        A.gamma_sites, A.positions, A.gamma_spec, A.ewald(), A.gamma_records());

    // Image-summed multipole integrals and CN-dependent AES parameters.
    A.multipole_sums = build_gfn2_multipole_lattice_sums(
        A.basis, atoms, A.cells, A.ao_atom, cutoff_bohr);
    A.aes_params = gfn2_aes_atom_parameters(
        atoms, params, A.coordination_numbers);

    // Lattice-summed pair repulsion.
    A.e_repulsive = 0.0;
    for (const auto& cell : A.cells) {
        const bool zero_cell = (cell.index.array() == 0).all();
        for (int a = 0; a < A.n_atoms; ++a) {
            const int first_b = zero_cell ? a + 1 : 0;
            for (int b = first_b; b < A.n_atoms; ++b) {
                const double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + cell.r_cart[0]);
                const double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + cell.r_cart[1]);
                const double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + cell.r_cart[2]);
                const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (R < 1e-12 || R > cutoff_bohr) continue;
                A.e_repulsive += (zero_cell ? 1.0 : 0.5)
                    * params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }
    return A;
}

void validate_periodic_gfn2_physics_domain(const PeriodicSystem& system) {
    if (system.unit_cell.empty()) return;
    const bool rare_gas_only = std::all_of(
        system.unit_cell.begin(), system.unit_cell.end(),
        [](const Atom& atom) { return is_noble_gas(atom.Z); });
    if (rare_gas_only) {
        throw std::runtime_error(
            "Periodic GFN2-xTB rare-gas cells are unavailable: the current "
            "H0 coordination-number implementation produces a nonphysical "
            "short-range attractive-collapse curve");
    }
}

PeriodicGFN2Result run_gfn2_xtb_gamma(
    const PeriodicSystem& system, const GFN2ParameterSet& params,
    const XTBSccOptions& sopts, double cutoff_bohr) {

    if (system.multiplicity != 1)
        throw std::invalid_argument("run_gfn2_xtb_gamma: closed-shell only");
    validate_periodic_gfn2_physics_domain(system);
    // Default-on frontier smearing: unless the caller set the electronic
    // temperature explicitly, use the small default width so a Gamma frontier
    // crossing on a lattice sweep cannot flip the hard-Aufbau occupations
    // between SCC branches. An explicit electronic_temperature=0 keeps the
    // exact zero-temperature Aufbau occupations.
    const double smearing_temperature = sopts.electronic_temperature_explicit
        ? sopts.electronic_temperature
        : kPeriodicGFN2DefaultElectronicTemperature;
    KPointOccupationOptions occupation_options;
    occupation_options.smearing_temperature = smearing_temperature;
    validate_kpoint_occupation_options(
        occupation_options, "run_gfn2_xtb_gamma");
    const ShellGammaForm gamma_form = sopts.gamma_form_explicit
        ? sopts.gamma_form
        : kPeriodicGFN2DefaultGammaForm;

    const PeriodicGFN2Assembly A = assemble_periodic_gfn2(
        system, params, cutoff_bohr, gamma_form);
    const auto& atoms = system.unit_cell;
    const int n_basis = A.n_basis;
    const int n_atoms = A.n_atoms;
    const int n_shells = A.n_shells;
    const int n_occ = A.n_occ;
    const ImageRecordSource records = A.records();

    const double simple_step = sopts.charge_mixing > 0.0
        ? std::min(sopts.charge_mixing, 0.1)
        : 0.1;

    // Mixer: damped simple mixing when third-order is active (see molecular
    // gfn2_driver.cpp for the full rationale: the 3rd-order term creates a
    // spurious over-polarized fixed point that Broyden/DIIS lock onto).  The
    // mixed vector is the full reduced state (charges and CAMM moments).
    std::unique_ptr<ChargeMixer> mixer;
    // Unset (Simple) selects the joint-state Eyert Broyden default unless
    // the caller asked for the damped simple/Aitken path explicitly through
    // a nonzero mixer_damping (the stabilization restart does).
    const bool eyert_default = sopts.scc_mixer == SCCMixer::Simple
        && !(sopts.mixer_damping > 0.0);
    const bool aitken_simple =
        A.has_gam3 && sopts.scc_mixer == SCCMixer::Simple && !eyert_default;
    if (eyert_default) {
        mixer = std::make_unique<EyertBroydenMixer>(
            std::min(std::max(sopts.max_iter, 1), kPeriodicGFN2BroydenMemory),
            kPeriodicGFN2BroydenMixing);
    } else if (sopts.scc_mixer == SCCMixer::DIIS) {
        mixer = std::make_unique<ChargeDIISMixer>(
            std::max(2, sopts.mixer_memory),
            sopts.mixer_damping > 0 ? sopts.mixer_damping : 0.5, 3);
    } else if (sopts.scc_mixer == SCCMixer::Broyden) {
        mixer = std::make_unique<BroydenMixer>(
            std::max(1, sopts.mixer_memory),
            sopts.mixer_damping > 0 ? sopts.mixer_damping : sopts.charge_mixing);
    } else if (sopts.scc_mixer == SCCMixer::BroydenEyert) {
        // tblite/xtb broyden.f90 on the full reduced state (charges and
        // CAMM moments), history sized to the iteration budget.
        mixer = std::make_unique<EyertBroydenMixer>(
            std::min(std::max(sopts.max_iter, 1), kPeriodicGFN2BroydenMemory),
            sopts.charge_mixing > 0.0 ? sopts.charge_mixing : kPeriodicGFN2BroydenMixing);
    } else if (sopts.scc_mixer == SCCMixer::Newton) {
        throw std::invalid_argument(
            "run_gfn2_xtb_gamma: SCCMixer::Newton is SECCM-only");
    } else {
        mixer = std::make_unique<SimpleMixer>(
            A.has_gam3 ? simple_step : sopts.charge_mixing);
    }

    ReducedState state;
    state.resize(n_shells, n_atoms);
    Eigen::VectorXd residual_prev;
    std::vector<double> max_change_trace;
    double step_cur = simple_step;
    PeriodicGFN2Result result;
    result.n_occ = n_occ;
    result.n_basis = n_basis;
    result.cutoff_bohr = cutoff_bohr;
    result.smearing_temperature = smearing_temperature;
    result.gamma_form = gamma_form;

    // The primary solve may use the full public budget. A fixed reserve
    // used to cut its allowance from 2499 to 500 when max_iter crossed
    // 2500, abandoning even a contracting SCC (#244). Decide whether to
    // restart from the residual history at budget-independent checkpoints,
    // so raising the cap does not change the primary trajectory. As in the
    // molecular driver's advisory checkpoint, compare successive window
    // minima; leave a still-improving solve on its current branch.
    const bool use_stabilization = A.has_gam3 && sopts.auto_stabilize;
    constexpr int kStabilityCheckpoint = 500;
    constexpr int kProgressWindow = 25;
    const auto still_contracting = [&max_change_trace]() {
        const int n = static_cast<int>(max_change_trace.size());
        if (n < 2 * kProgressWindow) return false;
        const auto end = max_change_trace.end();
        const double recent = *std::min_element(end - kProgressWindow, end);
        const double prior = *std::min_element(
            end - 2 * kProgressWindow, end - kProgressWindow);
        return recent < prior;
    };

    for (int iter = 1; iter <= sopts.max_iter; ++iter) {
        const Eigen::VectorXd dq_shell = state.dq_shell();
        // Isotropic second-order (shell-resolved) potential with the
        // lattice-summed gamma, plus the shell-resolved third-order on-site
        // potential (xtb thirdorder.f90 shellGam branch: V3_l = -G_l dq_l^2).
        Eigen::VectorXd V_shell = A.gamma_shell * dq_shell;
        for (int si = 0; si < n_shells; ++si) {
            const double G = A.gam3_shell(si);
            if (G != 0.0) V_shell(si) += -G * dq_shell(si) * dq_shell(si);
        }
        // Atomic partial charges (n0 - pop) and the mixed CAMM moments.
        Eigen::VectorXd q_atom = Eigen::VectorXd::Zero(n_atoms);
        for (int si = 0; si < n_shells; ++si)
            q_atom(A.shell_info[si].atom_idx) -= dq_shell(si);
        GFN2AesMoments moments;
        moments.q = q_atom;
        moments.mu = state.mu();
        moments.theta = state.theta();
        const GFN2AesPotentials potentials = gfn2_aes_potentials(
            A.positions, moments, A.aes_params, records);

        Eigen::MatrixXd H_scc = A.H0_gamma;
        for (int mu = 0; mu < n_basis; ++mu) {
            const int sm = A.ao_shell[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                const int sn = A.ao_shell[nu];
                H_scc(mu, nu) += 0.5 * A.S_gamma(mu, nu) * (V_shell(sm) + V_shell(sn));
            }
        }
        gfn2_aes_add_fock(H_scc, A.multipole_sums, A.ao_atom, A.positions, potentials);

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, A.S_gamma);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Periodic GFN2: diag failed");

        Eigen::VectorXd eps = solver.eigenvalues();
        auto C = solver.eigenvectors();
        // Default-on Fermi smearing (see the option resolution at the top of
        // this driver): a positive electronic temperature uses the
        // particle-conserving Fermi map n_i = 2 / (1 + exp((eps_i - mu) / T)).
        // At T=0 the shared occupation kernel reproduces the exact
        // hard-Aufbau density.
        auto occupation = compute_closed_shell_kpoint_occupations(
            {eps}, {1.0}, 2.0 * n_occ, n_occ, occupation_options);
        Eigen::MatrixXd D = C * occupation.occupations_per_k.front().asDiagonal()
            * C.transpose();
        Eigen::MatrixXd DS = D * A.S_gamma;

        Eigen::VectorXd dq_new = Eigen::VectorXd::Zero(n_shells);
        for (int mu = 0; mu < n_basis; ++mu) {
            const int si = A.ao_shell[mu];
            if (si >= 0) dq_new(si) += DS(mu, mu);
        }
        dq_new -= A.n0_shell;
        Eigen::VectorXd q_atom_new = Eigen::VectorXd::Zero(n_atoms);
        for (int si = 0; si < n_shells; ++si)
            q_atom_new(A.shell_info[si].atom_idx) -= dq_new(si);
        const GFN2AesMoments moments_new = gfn2_aes_moments(
            D, A.multipole_sums, A.ao_atom, A.positions, q_atom_new);
        ReducedState state_new;
        state_new.resize(n_shells, n_atoms);
        state_new.set(dq_new, moments_new);

        // AES is part of the self-consistent Hamiltonian: require the
        // complete reduced state used by the Fock build (charges and
        // moments) to be stationary (issue #338).
        const Eigen::VectorXd residual = state_new.x - state.x;
        const double max_change = residual.cwiseAbs().maxCoeff();
        max_change_trace.push_back(max_change);

        if (max_change < sopts.conv_tol_charge) {
            // Converged: evaluate the GFN2 explicit energy functional at the
            // converged density's own Mulliken charges and moments.
            Eigen::VectorXd dq_atom = Eigen::VectorXd::Zero(n_atoms);
            for (int mu = 0; mu < n_basis; ++mu) dq_atom(A.ao_atom[mu]) += DS(mu, mu);
            for (int a = 0; a < n_atoms; ++a)
                dq_atom(a) = static_cast<double>(gfn2_valence_electrons(atoms[a].Z)) - dq_atom(a);

            // Physical-basin gate (see the helper above): a converged
            // over-polarized shell-transfer state is a wrong answer, never a
            // result. Break so the bounded stabilization restart below can
            // search the physical branch, and fail closed when it cannot.
            if (!periodic_gfn2_charge_state_physical(
                    A.shell_info, A.n0_shell, dq_new, dq_atom)) {
                break;
            }

            // GFN2-xTB total energy as an explicit functional (JCTC 2019,
            // Eqs. 1-9): bare-H0 band + 2nd order (isotropic and
            // anisotropic) + 3rd order + repulsion.
            const double E_band0 = (D.cwiseProduct(A.H0_gamma)).sum();
            const double E_es = 0.5 * dq_new.dot(A.gamma_shell * dq_new);
            const double E_aes = gfn2_aes_energy(
                A.positions, moments_new, A.aes_params, records);
            double E_3rd = 0.0;
            for (int si = 0; si < n_shells; ++si) {
                const double G = A.gam3_shell(si);
                if (G == 0.0) continue;
                const double q = -dq_new(si);  // shell partial charge n0-pop
                E_3rd += G * q * q * q / 3.0;
            }

            result.energy = E_band0 + E_es + E_aes + E_3rd + A.e_repulsive;
            // Bannwarth, Ehlert & Grimme, JCTC 2019, Eqs. 8 and 10:
            // G_Fermi = -T*S is required for a variational solution with
            // fractional Fermi-Dirac occupations (Mermin, Phys. Rev. 137,
            // A1441 (1965)).
            result.fermi_level = occupation.fermi_level;
            result.entropy = occupation.entropy;
            result.free_energy = result.energy
                - smearing_temperature * result.entropy;
            result.e_electronic = E_band0 + E_es + E_aes + E_3rd;
            result.e_repulsive = A.e_repulsive;
            result.e_scc = E_es;
            result.e_band0 = E_band0;
            result.e_aes = E_aes;
            result.e_3rd = E_3rd;
            result.mo_energies = eps;
            result.mo_coeffs = C;
            result.density = D;
            result.overlap_gamma = A.S_gamma;
            result.hamiltonian_gamma = H_scc;
            result.charges = dq_atom;
            result.dq_shell = dq_new;
            result.atom_dipoles = moments_new.mu;
            result.atom_quadrupoles = moments_new.theta;
            result.occupations = occupation.occupations_per_k.front();
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_shells = n_shells;
            result.n_cells = static_cast<int>(A.cells.size());
            result.n_iter = iter;
            result.converged = true;
            result.scc_max_change_trace = Eigen::Map<const Eigen::VectorXd>(
                max_change_trace.data(), static_cast<Eigen::Index>(max_change_trace.size()));
            return result;
        }

        // Not converged: advance the reduced state.  The third-order GFN2
        // fixed point can acquire a strongly negative charge-response
        // eigenvalue in polar periodic cells. Bounded vector Aitken
        // relaxation reduces (but never extrapolates beyond) the
        // physical-branch simple-mixing step when consecutive residuals
        // reveal that alternating mode.
        if (aitken_simple) {
            if (residual_prev.size() > 0) {
                const Eigen::VectorXd delta = residual - residual_prev;
                const double denominator = delta.squaredNorm();
                if (denominator > 1e-24) {
                    const double estimate = -step_cur
                        * residual_prev.dot(delta) / denominator;
                    const double step_floor = std::min(0.005, simple_step);
                    if (std::isfinite(estimate) && estimate > 0.0) {
                        step_cur = std::max(
                            step_floor, std::min(simple_step, estimate));
                    } else if (residual.dot(residual_prev) < 0.0) {
                        step_cur = std::max(0.5 * step_cur, step_floor);
                    }
                }
            }
            residual_prev = residual;
            state.x += step_cur * residual;
        } else {
            mixer->mix(state.x, state_new.x, iter);
        }

        if (use_stabilization && iter % kStabilityCheckpoint == 0
                && !still_contracting()) {
            break;
        }
    }
    // A physical-basin rejection or stalled checkpoint may end the primary
    // early. Count executed iterations, rather than its unused allowance.
    result.n_iter = static_cast<int>(max_change_trace.size());
    result.scc_max_change_trace = Eigen::Map<const Eigen::VectorXd>(
        max_change_trace.data(), static_cast<Eigen::Index>(max_change_trace.size()));
    const int remaining = sopts.max_iter - result.n_iter;
    if (use_stabilization && remaining > 0) {
        // Match the molecular GFN2 contract: a failed physical-branch solve
        // receives one bounded restart from the neutral state with the
        // damped simple/Aitken path (mixer_damping > 0 selects it).
        XTBSccOptions retry_options = sopts;
        retry_options.charge_mixing = 0.01;
        retry_options.mixer_damping = 1.0;
        retry_options.max_iter = std::min(2000, remaining);
        retry_options.scc_mixer = SCCMixer::Simple;
        retry_options.auto_stabilize = false;
        retry_options.gamma_form = gamma_form;
        retry_options.gamma_form_explicit = true;
        PeriodicGFN2Result retry = run_gfn2_xtb_gamma(
            system, params, retry_options, cutoff_bohr);
        retry.n_iter += result.n_iter;
        Eigen::VectorXd trace(result.scc_max_change_trace.size()
                              + retry.scc_max_change_trace.size());
        trace << result.scc_max_change_trace, retry.scc_max_change_trace;
        retry.scc_max_change_trace = trace;
        return retry;
    }
    return result;
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
