#include "vibeqc/gradient.hpp"

#include <libint2/atom.h>
#include <libint2/engine.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/df.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/thread_pool.hpp"
#include "vibeqc/xc.hpp"

namespace vibeqc {

namespace {

// Same helper used by integrals.cpp — kept private here to avoid coupling.
std::vector<libint2::Atom> to_libint_atoms(const Molecule& mol) {
    std::vector<libint2::Atom> atoms;
    atoms.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        libint2::Atom la;
        la.atomic_number = a.Z;
        la.x = a.xyz[0];
        la.y = a.xyz[1];
        la.z = a.xyz[2];
        atoms.push_back(la);
    }
    return atoms;
}

// Returns, for each shell, the index of the atom it belongs to.
std::vector<long> shell_to_atom(const BasisSet& basis, const Molecule& mol) {
    return basis.libint().shell2atom(to_libint_atoms(mol));
}

// Sum of the three classical-Hcore-derived gradient pieces:
//   nuclear repulsion + tr(D · ∂(T+V_ne)/∂R) + (∂V_ECP/∂R if ECPs).
// When the options carry an ECP — either inline primitive blocks or XML
// library centres — the SCF replaced V_ne with effective-charges
// V_ne(Z_eff) and added V_ECP, so the gradient must follow suit on all
// three pieces to differentiate the same Hamiltonian as the energy. The
// public gradient entry point validates that XML and inline inputs are
// mutually exclusive before any occupancy or integral work. With neither set
// this reduces to the bare-Z all-electron path bit-for-bit.
Eigen::MatrixXd classical_hcore_grad_pieces(
    const Molecule& mol, const BasisSet& basis,
    const Eigen::MatrixXd& D,
    const GradientOptions& options) {
    // The public entry point has already proved that this is the one operative
    // ECP route. ``ecp_effective_charges`` is per-atom and already carries bare
    // Z on atoms with no ECP, so it is the Z_eff vector directly (#574).
    if (!options.ecp_primitive_blocks.empty()) {
        const auto& Z_eff_inline = options.ecp_effective_charges;
        if (Z_eff_inline.size() != mol.atoms().size()) {
            throw std::invalid_argument(
                "classical_hcore_grad_pieces: ecp_effective_charges must have "
                "one entry per atom when ecp_primitive_blocks is set (got "
                + std::to_string(Z_eff_inline.size()) + " for "
                + std::to_string(mol.atoms().size())
                + " atoms). Copy the values from the SCF options.");
        }
        return nuclear_repulsion_gradient(mol, Z_eff_inline)
             + one_electron_gradient_contribution(basis, mol, D, Z_eff_inline)
             + compute_ecp_gradient_contribution_from_primitives(
                   basis, mol, options.ecp_primitive_centers,
                   options.ecp_primitive_blocks, D);
    }
    if (options.ecp_centers.empty()) {
        return nuclear_repulsion_gradient(mol)
             + one_electron_gradient_contribution(basis, mol, D);
    }
    const std::string lib =
        options.ecp_library.empty() ? std::string("ecp10mdf")
                                    : options.ecp_library;
    const auto Z_eff = ecp_effective_charges(
        mol, options.ecp_centers, lib, /*share_dir=*/"");
    return nuclear_repulsion_gradient(mol, Z_eff)
         + one_electron_gradient_contribution(basis, mol, D, Z_eff)
         + compute_ecp_gradient_contribution(
               basis, mol, options.ecp_centers, D, lib, /*share_dir=*/"");
}

// Number of core electrons replaced by the ECPs named on ``options``
// (0 when no ECPs). The SCF fills valence-only orbitals
// (run_rhf: n_elec = mol.n_electrons() - ecp_h.total_ncore), so the
// gradient drivers must subtract the same count when sizing the
// occupied block for W / C_occ — otherwise the energy-weighted
// density sums over orbitals the SCF never occupied (probe: 0.41
// Ha/bohr error on the overlap-gradient piece of Zn²⁺-H/ecp10mdf).
int ecp_replaced_core_electrons(const Molecule& mol,
                                const GradientOptions& options) {
    // Inline path (mutually exclusive with XML after entry validation): the
    // SCF subtracted exactly this count
    // (run_rhf: n_elec = mol.n_electrons() - ecp_h.total_ncore, with
    // ECPHcore.total_ncore = opts.ecp_total_ncore), so the gradient must
    // size its occupied block with the same number (#574).
    if (!options.ecp_primitive_blocks.empty()) {
        return options.ecp_total_ncore;
    }
    if (options.ecp_centers.empty()) return 0;
    const std::string lib =
        options.ecp_library.empty() ? std::string("ecp10mdf")
                                    : options.ecp_library;
    const auto Z_eff = ecp_effective_charges(
        mol, options.ecp_centers, lib, /*share_dir=*/"");
    const auto& atoms = mol.atoms();
    double n_core = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        n_core += static_cast<double>(atoms[i].Z) - Z_eff[i];
    }
    return static_cast<int>(std::lround(n_core));
}

// Keep the native C++ gradient boundary aligned with the high-level SCF
// dispatch. In particular, direct users of ``_vibeqc_core.compute_gradient*``
// must not bypass the Python provenance wrapper and revive the historical
// inline-over-XML precedence. This runs at the start of every public molecular
// gradient entry point, before occupied-space sizing, grid construction, or
// derivative-integral work.
void validate_gradient_ecp_dispatch(const GradientOptions& options,
                                    const std::string& route) {
    (void)validate_molecular_ecp_dispatch(
        options.ecp_centers,
        options.ecp_library,
        options.ecp_primitive_blocks,
        options.ecp_primitive_centers,
        options.ecp_effective_charges,
        options.ecp_total_ncore,
        route);
}

// The XC-Pulay kernels below are linear sums over independent quadrature
// points. Keep the molecular grid itself (points, weights, and ownership) in
// its original order, but evaluate AO values and their derivatives only for
// one bounded contiguous slice at a time. This caps the otherwise dominant
// n_grid x n_basis AO/gradient/Hessian storage without changing the
// quadrature or any functional inputs.
template <typename Kernel>
Eigen::MatrixXd accumulate_molecular_xc_grid_batches(
    const Molecule& mol, const Grid& grid, Kernel&& kernel) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(mol.atoms().size()), 3);
    const Eigen::Index n_points = grid.points.rows();
    const Eigen::Index batch_size =
        static_cast<Eigen::Index>(kMolecularXcGridBatchSize);
    for (Eigen::Index first = 0; first < n_points; first += batch_size) {
        const Eigen::Index count = std::min(batch_size, n_points - first);
        Grid batch;
        batch.points = grid.points.middleRows(first, count);
        batch.weights = grid.weights.segment(first, count);
        const auto first_owner = grid.atom_of_point.begin()
            + static_cast<std::ptrdiff_t>(first);
        batch.atom_of_point.assign(
            first_owner, first_owner + static_cast<std::ptrdiff_t>(count));
        result += kernel(batch);
    }
    return result;
}

}  // namespace

Eigen::MatrixXd nuclear_repulsion_gradient(
    const Molecule& mol,
    const std::vector<double>& effective_charges) {
    const auto& atoms = mol.atoms();
    const std::size_t N = atoms.size();
    if (effective_charges.size() != N) {
        throw std::invalid_argument(
            "nuclear_repulsion_gradient: effective_charges length "
            + std::to_string(effective_charges.size())
            + " != mol.atoms() size " + std::to_string(N));
    }
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    // dE_nuc/dR_A = -Σ_{B≠A} Z_A Z_B (R_A - R_B) / |R_A - R_B|^3
    // With ECPs active, ``effective_charges[A] = Z_A − n_core_A``;
    // mirrors the SCF's effective E_nuc convention in
    // ``compute_ecp_one_electron``.
    for (std::size_t A = 0; A < N; ++A) {
        for (std::size_t B = 0; B < N; ++B) {
            if (A == B) continue;
            const double dx = atoms[A].xyz[0] - atoms[B].xyz[0];
            const double dy = atoms[A].xyz[1] - atoms[B].xyz[1];
            const double dz = atoms[A].xyz[2] - atoms[B].xyz[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double r3 = r2 * std::sqrt(r2);
            const double pref =
                -effective_charges[A] * effective_charges[B] / r3;
            grad(A, 0) += pref * dx;
            grad(A, 1) += pref * dy;
            grad(A, 2) += pref * dz;
        }
    }
    return grad;
}

Eigen::MatrixXd nuclear_repulsion_gradient(const Molecule& mol) {
    const auto& atoms = mol.atoms();
    std::vector<double> Z(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        Z[i] = static_cast<double>(atoms[i].Z);
    }
    return nuclear_repulsion_gradient(mol, Z);
}

Eigen::MatrixXd overlap_gradient_contribution(const BasisSet& basis,
                                              const Molecule& mol,
                                              const Eigen::MatrixXd& W) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(),
                              1 /*deriv_order*/);
    auto engines = make_engine_pool(prototype);

    // Per-thread gradient accumulator — each thread writes to its own
    // copy, then the driver sums them after the parallel region. The
    // alternative (atomic-add into a shared ``grad``) was ruled out on
    // performance; the reduction is cheap compared to the integrals.
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    const int n_shells = static_cast<int>(shells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];

            engine.compute(shells[s1], shells[s2]);
            // 6 derivative buffers: (x,y,z) for center 1, then center 2.
            for (int icenter = 0; icenter < 2; ++icenter) {
                const long atom = (icenter == 0) ? atom1 : atom2;
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            acc += W(bf1 + i, bf2 + j) * block[i * n2 + j];
                        }
                    }
                    // E contribution is -tr(W · dS); gradient picks up the sign.
                    grad_local(atom, d) -= acc;
                }
            }
        }
    }

    for (const auto& g : grad_tls) grad += g;
    return grad;
}

Eigen::MatrixXd one_electron_gradient_contribution(
    const BasisSet& basis, const Molecule& mol,
    const Eigen::MatrixXd& D,
    const std::vector<double>& effective_charges) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();
    if (effective_charges.size() != N) {
        throw std::invalid_argument(
            "one_electron_gradient_contribution: effective_charges length "
            + std::to_string(effective_charges.size())
            + " != mol.atoms() size " + std::to_string(N));
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    // --- Kinetic: only basis-center derivatives, 6 buffers per shell pair.
    {
        libint2::Engine prototype(libint2::Operator::kinetic,
                                  shells.max_nprim(), shells.max_l(), 1);
        auto engines = make_engine_pool(prototype);
        const int n_threads = omp_max_threads();
        std::vector<Eigen::MatrixXd> grad_tls(
            n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));
        const int n_shells = static_cast<int>(shells.size());

        #pragma omp parallel for schedule(dynamic)
        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& grad_local = grad_tls[tid];

            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                const long atom2 = s2a[s2];
                engine.compute(shells[s1], shells[s2]);
                for (int icenter = 0; icenter < 2; ++icenter) {
                    const long atom = (icenter == 0) ? atom1 : atom2;
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t i = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                acc += D(bf1 + i, bf2 + j) * block[i * n2 + j];
                            }
                        }
                        grad_local(atom, d) += acc;
                    }
                }
            }
        }
        for (const auto& g : grad_tls) grad += g;
    }

    // --- Nuclear attraction: basis centers (2) + all nuclei (Natoms) give
    //     3 * (2 + Natoms) derivative buffers per shell pair.
    // When ECPs replace core electrons the (q_A, R_A) list seen by
    // libint must use ``Z_eff_A = Z_A − n_core_A`` to match the SCF's
    // ``compute_nuclear_with_charges`` Hcore build.
    {
        std::vector<std::pair<double, std::array<double, 3>>> q;
        q.reserve(N);
        const auto& atoms = mol.atoms();
        for (std::size_t i = 0; i < N; ++i) {
            q.emplace_back(effective_charges[i], atoms[i].xyz);
        }
        libint2::Engine prototype(libint2::Operator::nuclear,
                                  shells.max_nprim(), shells.max_l(), 1);
        prototype.set_params(q);
        auto engines = make_engine_pool(prototype);
        const int n_threads = omp_max_threads();
        std::vector<Eigen::MatrixXd> grad_tls(
            n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));
        const int ncenters = 2 + static_cast<int>(N);
        const int n_shells = static_cast<int>(shells.size());

        #pragma omp parallel for schedule(dynamic)
        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& grad_local = grad_tls[tid];

            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                const long atom2 = s2a[s2];
                engine.compute(shells[s1], shells[s2]);
                for (int icenter = 0; icenter < ncenters; ++icenter) {
                    long atom;
                    if (icenter == 0) atom = atom1;
                    else if (icenter == 1) atom = atom2;
                    else atom = static_cast<long>(icenter - 2);  // nucleus index = atom index
                    for (int d = 0; d < 3; ++d) {
                        const double* block = buf[icenter * 3 + d];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t i = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                acc += D(bf1 + i, bf2 + j) * block[i * n2 + j];
                            }
                        }
                        grad_local(atom, d) += acc;
                    }
                }
            }
        }
        for (const auto& g : grad_tls) grad += g;
    }
    return grad;
}

Eigen::MatrixXd one_electron_gradient_contribution(const BasisSet& basis,
                                                    const Molecule& mol,
                                                    const Eigen::MatrixXd& D) {
    const auto& atoms = mol.atoms();
    std::vector<double> Z(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        Z[i] = static_cast<double>(atoms[i].Z);
    }
    return one_electron_gradient_contribution(basis, mol, D, Z);
}

ExternalChargeGradient compute_external_charge_density_gradient(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    const std::vector<double>& charges,
    const std::vector<std::array<double, 3>>& positions) {
    ensure_libint_initialized();

    if (charges.size() != positions.size()) {
        throw std::invalid_argument(
            "compute_external_charge_density_gradient: charges and "
            "positions must have the same length");
    }

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N_atoms = mol.atoms().size();
    const std::size_t N_pts = charges.size();

    ExternalChargeGradient out;
    out.atom_grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N_atoms), 3);
    out.point_grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N_pts), 3);
    if (N_pts == 0) return out;

    // libint's nuclear-attraction operator takes the charge list as
    // (charge, position) pairs — fractional charges are supported
    // (the same path integrals.cpp uses for compute_nuclear). With
    // deriv order 1 the engine emits 3·(2 + N_pts) buffers per shell
    // pair: 3 for the bra center, 3 for the ket center, 3 per charge.
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.reserve(N_pts);
    for (std::size_t i = 0; i < N_pts; ++i) {
        q.emplace_back(charges[i], positions[i]);
    }

    libint2::Engine prototype(libint2::Operator::nuclear,
                              shells.max_nprim(), shells.max_l(), 1);
    prototype.set_params(q);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    // Per-thread accumulators — the parallel scatter writes only into
    // the owning thread's matrix, so the cross-shell-pair writes to a
    // shared atom / point index are race-free without locking.
    std::vector<Eigen::MatrixXd> atom_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N_atoms), 3));
    std::vector<Eigen::MatrixXd> point_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N_pts), 3));

    const int ncenters = 2 + static_cast<int>(N_pts);
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& atom_local = atom_tls[tid];
        auto& point_local = point_tls[tid];

        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];
            engine.compute(shells[s1], shells[s2]);
            for (int icenter = 0; icenter < ncenters; ++icenter) {
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            acc += D(bf1 + i, bf2 + j) * block[i * n2 + j];
                        }
                    }
                    if (icenter == 0) {
                        atom_local(atom1, d) += acc;
                    } else if (icenter == 1) {
                        atom_local(atom2, d) += acc;
                    } else {
                        point_local(icenter - 2, d) += acc;
                    }
                }
            }
        }
    }
    for (const auto& g : atom_tls) out.atom_grad += g;
    for (const auto& g : point_tls) out.point_grad += g;

    // libint's nuclear operator is V_μν = −Σ_i q_i ⟨μ|1/|r−s_i||ν⟩,
    // so Tr(D·V) = −E_ext with E_ext = Σ_i q_i Tr(D·M_i) and M_i the
    // positive Coulomb kernel. The gradient engine therefore returns
    // −dE_ext/dR. Negate once so the caller gets +dE_ext/dR.
    out.atom_grad *= -1.0;
    out.point_grad *= -1.0;
    return out;
}

Eigen::MatrixXd two_electron_gradient_contribution(const BasisSet& basis,
                                                    const Molecule& mol,
                                                    const Eigen::MatrixXd& D,
                                                    double alpha_hf) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    // Fix C (v0.7.4) — permutationally-unique 1/8 shell-quartet loop +
    // angular-momentum-canonical reorder before libint, fixing the
    // v0.7.3 f-shell gradient bug.
    //
    // Mathematical formula:
    //   dE_2/dR = Σ_μνλσ ∂(μν|λσ)/∂R · Γ_μνλσ
    //   Γ_μνλσ = (1/2) D_μν D_λσ − (α_HF/4) D_μλ D_νσ
    //
    // Restricting to canonical shell quartets (s1≥s2, s3≥s4,
    // (s1,s2)≥(s3,s4)) with weight s1234_deg ∈ {1,2,4,8} and using the
    // symmetrised γ averaged over the 8 ERI permutations:
    //   Γ_avg = (1/2) D_μν D_λσ − (α_HF/8)(D_μλ D_νσ + D_νλ D_μσ)
    //
    // Then:
    //   dE_2/dR[A, d] = Σ_canonical s1234_deg · Σ_basis Γ_avg
    //                    · ∂(μν|λσ)/∂R[A, d]
    //
    // The 23a7951 attempt used 2·Γ_avg (which gave 2× error on
    // fully-unique quartets); Γ_avg = (1/8)·Σ_8 perm-gammas is correct.
    //
    // L-canonical reorder before engine.compute() avoids libint's
    // internal swap_tbra / swap_tket / swap_braket + DerivMapGenerator
    // unscramble path, which has a buggy derivative-to-atom routing for
    // high-l mixed-l shell quartets (the v0.7.3 f-shell bug; see
    // HANDOVER_0.7.4.md and the keen-edison investigation report). For
    // libint's BraKet::xx_xx with standard angular-momentum ordering
    // (third_party/libint/.../engine.impl.h:1215-1219):
    //   swap_tbra = (l1 < l2)         — want l1 >= l2
    //   swap_tket = (l3 < l4)         — want l3 >= l4
    //   swap_braket = (l1+l2 > l3+l4) — want l1+l2 <= l3+l4
    //
    // The same fix was applied to the 3c DF kernel in commit 2196345.
    //
    // For alpha_hf = 1 this is plain HF. For hybrid DFT (e.g. B3LYP)
    // pass the functional's HF-exchange fraction. For pure DFT pass 0 —
    // only the Coulomb half of Γ survives.
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        for (int s2 = 0; s2 <= s1; ++s2) {
            for (int s3 = 0; s3 <= s1; ++s3) {
                const int s4_max = (s1 == s3) ? s2 : s3;
                for (int s4 = 0; s4 <= s4_max; ++s4) {
                    // Permutational degeneracy of the canonical quartet.
                    const double s12_deg = (s1 == s2) ? 1.0 : 2.0;
                    const double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                    const double s12_34_deg =
                        (s1 == s3) ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
                    const double weight = s12_deg * s34_deg * s12_34_deg;

                    // L-canonical reorder so libint's swap_tbra /
                    // swap_tket / swap_braket are all false. Permute
                    // (s1,s2,s3,s4) → (sa,sb,sc,sd).
                    int sa = s1, sb = s2, sc = s3, sd = s4;
                    int la = shells[sa].contr[0].l;
                    int lb = shells[sb].contr[0].l;
                    int lc = shells[sc].contr[0].l;
                    int ld = shells[sd].contr[0].l;
                    if (la < lb) {
                        std::swap(sa, sb); std::swap(la, lb);
                    }
                    if (lc < ld) {
                        std::swap(sc, sd); std::swap(lc, ld);
                    }
                    if (la + lb > lc + ld) {
                        std::swap(sa, sc); std::swap(sb, sd);
                        std::swap(la, lc); std::swap(lb, ld);
                    }

                    const auto bfa = shell2bf[sa];
                    const auto nA  = shells[sa].size();
                    const auto bfb = shell2bf[sb];
                    const auto nB  = shells[sb].size();
                    const auto bfc = shell2bf[sc];
                    const auto nC  = shells[sc].size();
                    const auto bfd = shell2bf[sd];
                    const auto nD  = shells[sd].size();

                    engine.compute(shells[sa], shells[sb],
                                   shells[sc], shells[sd]);
                    if (buf[0] == nullptr) continue;  // screened.

                    // Derivative buffers are indexed in the same order
                    // as the shells passed to engine.compute (the
                    // l-canonical order), so atom routing follows that.
                    const long centers[4] = {
                        s2a[sa], s2a[sb], s2a[sc], s2a[sd]};

                    for (int icenter = 0; icenter < 4; ++icenter) {
                        for (int d = 0; d < 3; ++d) {
                            const double* block = buf[icenter * 3 + d];
                            if (!block) continue;
                            double acc = 0.0;
                            for (std::size_t i = 0; i < nA; ++i) {
                                const auto mu = bfa + i;
                                for (std::size_t j = 0; j < nB; ++j) {
                                    const auto nu = bfb + j;
                                    const double d_mu_nu = D(mu, nu);
                                    for (std::size_t k = 0; k < nC; ++k) {
                                        const auto lam = bfc + k;
                                        const double d_mu_lam = D(mu, lam);
                                        const double d_nu_lam = D(nu, lam);
                                        for (std::size_t l = 0; l < nD; ++l) {
                                            const auto sig = bfd + l;
                                            // Γ_avg = (1/2) D_μν D_λσ
                                            //   − (α/8)(D_μλ D_νσ
                                            //          + D_νλ D_μσ)
                                            const double gamma =
                                                0.5 * d_mu_nu * D(lam, sig)
                                                - 0.125 * alpha_hf * (
                                                    d_mu_lam * D(nu, sig)
                                                  + d_nu_lam * D(mu, sig));
                                            acc += gamma * block[
                                                ((i * nB + j) * nC + k) * nD + l];
                                        }
                                    }
                                }
                            }
                            grad_local(centers[icenter], d) += weight * acc;
                        }
                    }
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

Eigen::MatrixXd compute_gradient(const Molecule& mol,
                                 const BasisSet& basis,
                                 const RHFResult& result,
                                 const GradientOptions& options) {
    validate_gradient_ecp_dispatch(options, "compute_gradient");
    if (!result.converged) {
        throw std::runtime_error(
            "compute_gradient: RHF result is not converged");
    }

    // Energy-weighted density W = 2 Σ_{i∈occ} ε_i C_μi C_νi
    // (valence-only occupied count when ECPs replace core electrons;
    // matches the SCF's n_elec = mol.n_electrons() - total_ncore).
    const int nocc =
        (mol.n_electrons() - ecp_replaced_core_electrons(mol, options)) / 2;
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
    for (int i = 0; i < nocc; ++i) {
        const Eigen::VectorXd Ci = result.mo_coeffs.col(i);
        W += 2.0 * result.mo_energies(i) * Ci * Ci.transpose();
    }
    const Eigen::MatrixXd& D = result.density;

    // Two paths for the two-electron gradient:
    //   density_fit=false: four-index ERI derivative (default).
    //   density_fit=true : DF J + α_HF·K via the precomputed B-tensor
    //                       (Weigend, PCCP 4, 4285 (2002)). For HF
    //                       α_HF = 1; the K piece is wrapped into the
    //                       same DensityFitting::compute_jk_gradient
    //                       call.
    Eigen::MatrixXd grad_2e;
    if (options.density_fit) {
        if (options.aux_basis.empty()) {
            throw std::invalid_argument(
                "compute_gradient: density_fit=true requires aux_basis "
                "to be set (e.g. \"def2-svp-jk\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"jk\") "
                "for autodetection.");
        }
        const BasisSet aux(mol, options.aux_basis);
        const DensityFitting df(basis, aux);
        const Eigen::MatrixXd C_occ = result.mo_coeffs.leftCols(nocc);
        if (options.cosx) {
            // RIJCOSX RHF: J via DF, K via seminumerical chain-of-spheres.
            const int cosx_card =
                cosx_basis_cardinality_from_name(basis.name());
            const Grid cosx_grid_built = build_grid(
                mol, resolve_cosx_grid_options(
                         options.cosx_grid, options.cosx_grid_level, cosx_card));
            const bool cosx_fit =
                resolve_cosx_variant(options.cosx_variant, options.thresh_cosx,
                                     cosx_card, options.cosx_grid_level)
                    != CosxVariant::STANDARD;
            grad_2e = df.compute_j_gradient(mol, D)
                    + compute_cosx_k_gradient_contribution(
                        mol, basis, D, cosx_grid_built, /*alpha_hf=*/1.0,
                        Eigen::MatrixXd(), cosx_fit);
        } else {
            grad_2e = df.compute_jk_gradient(mol, D, C_occ, /*alpha_hf=*/1.0);
        }
    } else {
        grad_2e = two_electron_gradient_contribution(basis, mol, D);
    }

    return classical_hcore_grad_pieces(mol, basis, D, options)
         + overlap_gradient_contribution(basis, mol, W)
         + grad_2e;
}

Eigen::MatrixXd two_electron_gradient_contribution_uhf(
    const BasisSet& basis, const Molecule& mol,
    const Eigen::MatrixXd& D_alpha, const Eigen::MatrixXd& D_beta,
    double alpha_hf) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();

    const Eigen::MatrixXd D_total = D_alpha + D_beta;

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    // Fix C (v0.7.4) — UHF/UKS counterpart of the canonical 1/8 +
    // l-canonical reorder fix in the RHF kernel above. See that
    // function's commentary for the bug context.
    //
    // All-pairs Γ_UHF (this branch's pre-Fix-C formula):
    //   Γ_UHF_μνλσ = (1/2)(D_tot)_μν (D_tot)_λσ
    //              − (α_HF/2)(D_α)_μλ (D_α)_νσ
    //              − (α_HF/2)(D_β)_μλ (D_β)_νσ
    //
    // Canonical-loop Γ_avg (averaged over the 8 ERI permutations):
    //   J part: (1/2) D_tot_μν D_tot_λσ  (unchanged — 8 perms are equal)
    //   K part (per spin σ ∈ {α,β}):
    //      − (α_HF/4)(D_σ_μλ D_σ_νσ + D_σ_νλ D_σ_μσ)
    //
    // For closed-shell (D_α = D_β = D/2) this reduces bit-for-bit to
    // the RHF Γ_avg in the function above.
    //
    // α_HF = 1 → plain UHF. Hybrid DFT passes the functional's HF
    // fraction. Pure DFT passes 0 (only Coulomb survives).
    const double x_coef = 0.25 * alpha_hf;  // (α_HF/4) per spin
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        for (int s2 = 0; s2 <= s1; ++s2) {
            for (int s3 = 0; s3 <= s1; ++s3) {
                const int s4_max = (s1 == s3) ? s2 : s3;
                for (int s4 = 0; s4 <= s4_max; ++s4) {
                    const double s12_deg = (s1 == s2) ? 1.0 : 2.0;
                    const double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                    const double s12_34_deg =
                        (s1 == s3) ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
                    const double weight = s12_deg * s34_deg * s12_34_deg;

                    // L-canonical reorder before libint (see RHF kernel
                    // for rationale).
                    int sa = s1, sb = s2, sc = s3, sd = s4;
                    int la = shells[sa].contr[0].l;
                    int lb = shells[sb].contr[0].l;
                    int lc = shells[sc].contr[0].l;
                    int ld = shells[sd].contr[0].l;
                    if (la < lb) {
                        std::swap(sa, sb); std::swap(la, lb);
                    }
                    if (lc < ld) {
                        std::swap(sc, sd); std::swap(lc, ld);
                    }
                    if (la + lb > lc + ld) {
                        std::swap(sa, sc); std::swap(sb, sd);
                        std::swap(la, lc); std::swap(lb, ld);
                    }

                    const auto bfa = shell2bf[sa];
                    const auto nA  = shells[sa].size();
                    const auto bfb = shell2bf[sb];
                    const auto nB  = shells[sb].size();
                    const auto bfc = shell2bf[sc];
                    const auto nC  = shells[sc].size();
                    const auto bfd = shell2bf[sd];
                    const auto nD  = shells[sd].size();

                    engine.compute(shells[sa], shells[sb],
                                   shells[sc], shells[sd]);
                    if (buf[0] == nullptr) continue;

                    const long centers[4] = {
                        s2a[sa], s2a[sb], s2a[sc], s2a[sd]};
                    for (int icenter = 0; icenter < 4; ++icenter) {
                        for (int d = 0; d < 3; ++d) {
                            const double* block = buf[icenter * 3 + d];
                            if (!block) continue;
                            double acc = 0.0;
                            for (std::size_t i = 0; i < nA; ++i) {
                                const auto mu = bfa + i;
                                for (std::size_t j = 0; j < nB; ++j) {
                                    const auto nu = bfb + j;
                                    const double d_tot_mu_nu = D_total(mu, nu);
                                    for (std::size_t k = 0; k < nC; ++k) {
                                        const auto lam = bfc + k;
                                        const double d_a_mu_lam = D_alpha(mu, lam);
                                        const double d_a_nu_lam = D_alpha(nu, lam);
                                        const double d_b_mu_lam = D_beta (mu, lam);
                                        const double d_b_nu_lam = D_beta (nu, lam);
                                        for (std::size_t l = 0; l < nD; ++l) {
                                            const auto sig = bfd + l;
                                            // Γ_avg per-spin K:
                                            //  −(α/4)(D_σ_μλ D_σ_νσ
                                            //        + D_σ_νλ D_σ_μσ)
                                            const double gamma =
                                                0.5 * d_tot_mu_nu * D_total(lam, sig)
                                                - x_coef * (
                                                    d_a_mu_lam * D_alpha(nu, sig)
                                                  + d_a_nu_lam * D_alpha(mu, sig)
                                                  + d_b_mu_lam * D_beta (nu, sig)
                                                  + d_b_nu_lam * D_beta (mu, sig));
                                            acc += gamma * block[
                                                ((i * nB + j) * nC + k) * nD + l];
                                        }
                                    }
                                }
                            }
                            grad_local(centers[icenter], d) += weight * acc;
                        }
                    }
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

namespace {

// LDA-only XC Pulay force contribution for RKS:
//   (dE_xc/dR_A)_c = −2 Σ_{μ∈A} Σ_ν D_μν Σ_g w_g v_ρ(g) ∂_c χ_μ(g) χ_ν(g)
// arising from the basis-function derivative ∂χ_μ/∂R_A = −∂_c χ_μ when μ
// is centered on A.
//
// GGAs add a contribution involving ∇ρ · ∂(χ_μ χ_ν)/∂R_A, which requires
// the second spatial derivatives of χ on the grid — handled in a
// follow-up commit.
Eigen::MatrixXd xc_pulay_gradient_rks_lda(
    const Molecule& mol, const BasisSet& basis,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D,
    const Functional& func) {

    const auto n_bf = static_cast<Eigen::Index>(basis.nbasis());
    const std::size_t N = mol.atoms().size();

    // Density on grid.
    const Eigen::MatrixXd chiD = chi * D;
    Eigen::VectorXd rho = (chiD.array() * chi.array()).rowwise().sum();

    // XC potential (LDA).
    Eigen::VectorXd sigma(0);
    Eigen::VectorXd exc, v_rho, v_sigma;
    func.eval_unpolarised(rho, sigma, exc, v_rho, v_sigma);

    // W_c_{μν} = ∂_c χ^T · diag(w · v_ρ) · χ     [not symmetric in μ,ν]
    Eigen::VectorXd w_vrho = grid.weights.array() * v_rho.array();
    std::array<Eigen::MatrixXd, 3> W;
    for (int c = 0; c < 3; ++c) {
        W[c] = dchi[c].transpose() * w_vrho.asDiagonal() * chi;
    }

    // Build list of basis-function indices per atom.
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    std::vector<std::vector<int>> atom_bf(N);
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const int A = static_cast<int>(s2a[s]);
        const int bf_start = static_cast<int>(shell2bf[s]);
        const int bf_end = bf_start + static_cast<int>(shells[s].size());
        for (int bf = bf_start; bf < bf_end; ++bf) {
            atom_bf[A].push_back(bf);
        }
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);
    for (std::size_t A = 0; A < N; ++A) {
        for (int c = 0; c < 3; ++c) {
            double acc = 0.0;
            for (const int mu : atom_bf[A]) {
                // Σ_ν D_{μν} W_c(μ, ν)
                acc += (D.row(mu).array() * W[c].row(mu).array()).sum();
            }
            grad(A, c) = -2.0 * acc;
        }
    }
    return grad;
}

}  // namespace

namespace {

// Helper: list of basis-function indices per atom.
std::vector<std::vector<int>> basis_functions_per_atom(
    const BasisSet& basis, const Molecule& mol) {
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    std::vector<std::vector<int>> atom_bf(mol.atoms().size());
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const int A = static_cast<int>(s2a[s]);
        const int bf_start = static_cast<int>(shell2bf[s]);
        const int bf_end = bf_start + static_cast<int>(shells[s].size());
        for (int bf = bf_start; bf < bf_end; ++bf) atom_bf[A].push_back(bf);
    }
    return atom_bf;
}

// GGA / meta-GGA XC Pulay force for RKS. Includes the LDA piece, the GGA
// pieces that involve ∇ρ and ∂²χ (via the Hessian), and — for meta-GGA
// functionals — the kinetic-energy-density τ piece. Formula:
//
//   T_{μν}^c = −Σ_g w_g v_ρ(g) ∂_c χ_μ(g) χ_ν(g)                    (1)
//            −Σ_g w_g 2 v_σ(g) ∂_c χ_μ(g) (∇ρ(g)·∇χ_ν(g))          (2)
//            −Σ_g w_g 2 v_σ(g) (∇ρ(g)·∂_c∇χ_μ(g)) χ_ν(g)           (3)
//            −Σ_g w_g v_τ(g) Σ_d ∂_c∂_d χ_μ(g) ∂_d χ_ν(g)          (4)
//
// Then (dE_xc^Pulay/dR_A)_c = 2 Σ_{μ∈A} Σ_ν D_μν T_{μν}^c.
//
// Term (4) is the τ piece. With τ(g) = ½ Σ_μν D_μν Σ_d ∂_dχ_μ ∂_dχ_ν,
// the ½ prefactor cancels the μ↔ν symmetry factor of 2, so the τ-Pulay
// matrix carries an explicit ½ to match the universal −2 scatter
// prefactor below (M1's ρ-term has no ½, hence the −2 is correct for
// it directly; M4 = ½ · [Σ_d hessᵀ diag(w v_τ) dchi_d]).
Eigen::MatrixXd xc_pulay_gradient_rks_gga_mgga(
    const Molecule& mol, const BasisSet& basis,
    const Grid& grid,
    const AOValuesWithHessian& ao,
    const Eigen::MatrixXd& D,
    const Functional& func) {

    const auto n_pts = ao.values.rows();
    const auto n_bf = ao.values.cols();
    const std::size_t N = mol.atoms().size();
    const bool is_mgga = (func.kind() == XCKind::MGGA);

    const Eigen::MatrixXd& chi = ao.values;
    const auto& dchi = ao.gradients;
    const auto& hess = ao.hessians;   // xx, xy, xz, yy, yz, zz

    // ρ and ∇ρ on grid.
    const Eigen::MatrixXd chiD = chi * D;
    Eigen::VectorXd rho = (chiD.array() * chi.array()).rowwise().sum();
    Eigen::VectorXd gx = 2.0 * chiD.cwiseProduct(dchi[0]).rowwise().sum();
    Eigen::VectorXd gy = 2.0 * chiD.cwiseProduct(dchi[1]).rowwise().sum();
    Eigen::VectorXd gz = 2.0 * chiD.cwiseProduct(dchi[2]).rowwise().sum();
    Eigen::VectorXd sigma = gx.array().square() + gy.array().square()
                          + gz.array().square();

    // τ(g) = ½ Σ_μν D_μν Σ_d ∂_dχ_μ ∂_dχ_ν  (MGGA only).
    Eigen::VectorXd tau;
    if (is_mgga) {
        tau = Eigen::VectorXd::Zero(n_pts);
        for (int d = 0; d < 3; ++d) {
            const Eigen::MatrixXd dchiD = dchi[d] * D;
            tau.array() += 0.5
                * dchiD.cwiseProduct(dchi[d]).rowwise().sum().array();
        }
    }

    Eigen::VectorXd exc, v_rho, v_sigma, v_tau;
    if (is_mgga) {
        func.eval_unpolarised_mgga(rho, sigma, tau,
                                   exc, v_rho, v_sigma, v_tau);
    } else {
        func.eval_unpolarised(rho, sigma, exc, v_rho, v_sigma);
    }

    const Eigen::VectorXd w     = grid.weights;
    const Eigen::VectorXd w_vr  = w.array() * v_rho.array();
    const Eigen::VectorXd w_2vs = 2.0 * w.array() * v_sigma.array();
    // ½ · w · v_τ — the ½ cancels τ's μ↔ν symmetry factor (see header).
    Eigen::VectorXd w_vtau_half;
    if (is_mgga) w_vtau_half = 0.5 * w.array() * v_tau.array();

    // --- Term 1 (LDA-like): M1_{μν} = Σ_g w v_ρ ∂_c χ_μ χ_ν  for each c
    // --- Term 2 (GGA, ∂χ × ∇χ): M2_{μν} = Σ_g 2 w v_σ (∇ρ·∇χ_ν) ∂_c χ_μ  for each c
    // --- Term 3 (GGA, Hessian): M3_{μν} = Σ_g 2 w v_σ (∇ρ·∂_c∇χ_μ) χ_ν for each c
    // --- Term 4 (MGGA, τ):      M4_{μν} = ½ Σ_g w v_τ Σ_d ∂_c∂_d χ_μ ∂_d χ_ν
    //
    // We accumulate -2 × Σ_{μ∈A} Σ_ν D_μν (M1 + M2 + M3 + M4)_{μν}^c into grad[A, c].

    // Pre-compute  ∇ρ · ∇χ_ν(g) for each (g, ν) :
    //   dot_gradrho_gradchi(g, ν) = gx χ'_x(g,ν) + gy χ'_y(g,ν) + gz χ'_z(g,ν)
    Eigen::MatrixXd dot_grc =
          gx.asDiagonal() * dchi[0]
        + gy.asDiagonal() * dchi[1]
        + gz.asDiagonal() * dchi[2];   // (n_pts, n_bf)

    auto hess_entry = [&](int c, int d) -> const Eigen::MatrixXd& {
        // Map ordered pair (c, d) to storage index using symmetry:
        //   xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
        if (c > d) std::swap(c, d);
        static constexpr int idx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        return hess[idx[c][d]];
    };

    const auto atom_bf = basis_functions_per_atom(basis, mol);
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    // Precompute the c-specific matrices that we scatter into atoms.
    for (int c = 0; c < 3; ++c) {
        // For term 3 we need the "directional Hessian" Y_μ^c(g) =
        //   Σ_d ∂_d ρ(g) × ∂_c∂_d χ_μ(g)
        Eigen::MatrixXd Y = gx.asDiagonal() * hess_entry(c, 0)
                          + gy.asDiagonal() * hess_entry(c, 1)
                          + gz.asDiagonal() * hess_entry(c, 2);

        // M^c_{μν} = (∂_c χ)^T (w v_ρ) χ           (term 1)
        //          + (∂_c χ)^T (2 w v_σ) · (∇ρ·∇χ) (term 2)
        //          + Y^T (2 w v_σ) χ               (term 3)
        Eigen::MatrixXd M1 = dchi[c].transpose() * w_vr.asDiagonal()  * chi;
        Eigen::MatrixXd M2 = dchi[c].transpose() * w_2vs.asDiagonal() * dot_grc;
        Eigen::MatrixXd M3 = Y.transpose()       * w_2vs.asDiagonal() * chi;
        Eigen::MatrixXd Mc = M1 + M2 + M3;

        if (is_mgga) {
            // Term 4: M4_{μν}^c = ½ Σ_d (∂_c∂_d χ_μ) (w v_τ) (∂_d χ_ν).
            for (int d = 0; d < 3; ++d) {
                Mc.noalias() += hess_entry(c, d).transpose()
                              * w_vtau_half.asDiagonal() * dchi[d];
            }
        }

        for (std::size_t A = 0; A < N; ++A) {
            double acc = 0.0;
            for (const int mu : atom_bf[A]) {
                acc += (D.row(mu).array() * Mc.row(mu).array()).sum();
            }
            grad(A, c) = -2.0 * acc;
        }
    }
    return grad;
}

}  // namespace

Eigen::MatrixXd compute_gradient_rks(const Molecule& mol,
                                     const BasisSet& basis,
                                     const RKSResult& result,
                                     const GridOptions& grid_options,
                                     const GradientOptions& options) {
    validate_gradient_ecp_dispatch(options, "compute_gradient_rks");
    if (!result.converged) {
        throw std::runtime_error(
            "compute_gradient_rks: RKS result is not converged");
    }

    Functional functional(result.functional);
    if (functional.is_external()) {
        throw std::runtime_error(
            "compute_gradient_rks: analytic gradients for full-grid "
            "external XC functionals are unavailable because derivatives "
            "of grid coordinates, partition weights, and atom coarse "
            "centers are required; use whole-SCF finite differences");
    }
    const XCKind xc_kind = functional.kind();
    // GGA and meta-GGA both need the basis Hessian (∂²χ) on the grid.
    const bool need_hessian = (xc_kind == XCKind::GGA
                               || xc_kind == XCKind::MGGA);
    const double alpha_hf = functional.hf_exchange_fraction();

    // Energy-weighted density (RHF convention: W = 2 Σ_i ε_i C_μi C_νi;
    // valence-only occupied count when ECPs replace core electrons).
    const int nocc =
        (mol.n_electrons() - ecp_replaced_core_electrons(mol, options)) / 2;
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
    for (int i = 0; i < nocc; ++i) {
        const Eigen::VectorXd Ci = result.mo_coeffs.col(i);
        W += 2.0 * result.mo_energies(i) * Ci * Ci.transpose();
    }
    const Eigen::MatrixXd& D = result.density;

    Grid grid = build_grid(mol, grid_options);

    const Eigen::MatrixXd xc_pulay = accumulate_molecular_xc_grid_batches(
        mol, grid, [&](const Grid& batch) {
            if (need_hessian) {
                AOValuesWithHessian ao =
                    evaluate_ao_with_hessian(basis, batch.points);
                return xc_pulay_gradient_rks_gga_mgga(
                    mol, basis, batch, ao, D, functional);
            }
            AOValues ao = evaluate_ao_with_gradient(basis, batch.points);
            return xc_pulay_gradient_rks_lda(
                mol, basis, batch, ao.values, ao.gradients, D, functional);
        });

    Eigen::MatrixXd grad_2e;
    if (options.density_fit) {
        if (options.aux_basis.empty()) {
            throw std::invalid_argument(
                "compute_gradient_rks: density_fit=true requires "
                "aux_basis to be set (e.g. \"def2-svp-jk\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"jk\") "
                "for autodetection.");
        }
        const BasisSet aux(mol, options.aux_basis);
        const DensityFitting df(basis, aux);
        const Eigen::MatrixXd C_occ = result.mo_coeffs.leftCols(nocc);
        if (options.cosx && alpha_hf != 0.0) {
            // RIJCOSX: J via DF, K via seminumerical chain-of-spheres.
            // Build the COSX grid once and feed it to the K-gradient
            // kernel; the J-grad kernel is unchanged from RIJK.
            const int cosx_card =
                cosx_basis_cardinality_from_name(basis.name());
            const Grid cosx_grid_built = build_grid(
                mol, resolve_cosx_grid_options(
                         options.cosx_grid, options.cosx_grid_level, cosx_card));
            const bool cosx_fit =
                resolve_cosx_variant(options.cosx_variant, options.thresh_cosx,
                                     cosx_card, options.cosx_grid_level)
                    != CosxVariant::STANDARD;
            grad_2e = df.compute_j_gradient(mol, D)
                    + compute_cosx_k_gradient_contribution(
                        mol, basis, D, cosx_grid_built, alpha_hf,
                        Eigen::MatrixXd(), cosx_fit);
        } else {
            grad_2e = df.compute_jk_gradient(mol, D, C_occ, alpha_hf);
        }
    } else {
        grad_2e = two_electron_gradient_contribution(basis, mol, D, alpha_hf);
    }

    return classical_hcore_grad_pieces(mol, basis, D, options)
         + overlap_gradient_contribution(basis, mol, W)
         + grad_2e
         + xc_pulay;
}

Eigen::MatrixXd xc_pulay_gradient_rks(const Molecule& mol,
                                      const BasisSet& basis,
                                      const Eigen::MatrixXd& D,
                                      const std::string& functional_name,
                                      const GridOptions& grid_options) {
    // The XC-Pulay piece of compute_gradient_rks (the basis-function derivatives
    // of E_xc[ρ] on a *fixed* Becke grid — same grid-fixed approximation, no Becke
    // weight derivative), factored out and exposed for the Γ-CCM force driver. The
    // CCM XC term is the ordinary molecular functional of the cluster (supercell)
    // density on a molecular Becke grid; it does NOT fold against the WSSC weights
    // (unlike h^CCM / J^CCM / S^CCM), so the CCM XC gradient is exactly this
    // molecular XC-Pulay evaluated on the supercell. LDA / GGA / meta-GGA.
    Functional functional(functional_name);
    if (functional.is_external()) {
        throw std::runtime_error(
            "xc_pulay_gradient_rks: full-grid external XC gradients are "
            "not implemented");
    }
    const XCKind xc_kind = functional.kind();
    const bool need_hessian = (xc_kind == XCKind::GGA || xc_kind == XCKind::MGGA);
    Grid grid = build_grid(mol, grid_options);
    return accumulate_molecular_xc_grid_batches(
        mol, grid, [&](const Grid& batch) {
            if (need_hessian) {
                AOValuesWithHessian ao =
                    evaluate_ao_with_hessian(basis, batch.points);
                return xc_pulay_gradient_rks_gga_mgga(
                    mol, basis, batch, ao, D, functional);
            }
            AOValues ao = evaluate_ao_with_gradient(basis, batch.points);
            return xc_pulay_gradient_rks_lda(
                mol, basis, batch, ao.values, ao.gradients, D, functional);
        });
}

namespace {

// UKS XC Pulay force. Covers LDA, GGA, and meta-GGA in a single pass.
// For spin σ:
//   T_σ^{c}_{μν} = -v_ρ_σ ∂_c χ_μ χ_ν
//                  - (GGA terms involving ∇ρ_σ, ∇ρ_{¬σ} with the same
//                     ∂_c∇χ_μ χ_ν / ∂_c χ_μ ∇χ_ν structure as RKS GGA)
//                  - (meta-GGA τ term: v_τ_σ Σ_d ∂_c∂_d χ_μ ∂_d χ_ν)
// with per-spin "flow" vector
//   f_α = 2 v_σ_αα · ∇ρ_α + v_σ_αβ · ∇ρ_β
//   f_β = 2 v_σ_ββ · ∇ρ_β + v_σ_αβ · ∇ρ_α
// so that each spin's contribution has the RKS-GGA shape with ∇ρ → f_σ
// and v_ρ → v_ρ_σ. The σ_αβ cross-term is baked into the flow vectors.
// The τ term mirrors the RKS one: τ_σ = ½ Σ_μν (D_σ)_μν Σ_d ∂_dχ_μ ∂_dχ_ν,
// the ½ cancels the μ↔ν symmetry factor, M4_σ carries an explicit ½.
Eigen::MatrixXd xc_pulay_gradient_uks(
    const Molecule& mol, const BasisSet& basis,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const std::array<Eigen::MatrixXd, 6>* hess,   // null for LDA
    const Eigen::MatrixXd& D_alpha,
    const Eigen::MatrixXd& D_beta,
    const Functional& func) {

    const auto n_pts = chi.rows();
    const auto n_bf = chi.cols();
    const std::size_t N = mol.atoms().size();
    const XCKind xc_kind = func.kind();
    const bool is_gga  = (xc_kind == XCKind::GGA);
    const bool is_mgga = (xc_kind == XCKind::MGGA);
    if ((is_gga || is_mgga) && hess == nullptr) {
        throw std::runtime_error(
            "xc_pulay_gradient_uks: GGA / meta-GGA needs the basis Hessian");
    }

    // Densities and their gradients on grid.
    const Eigen::MatrixXd chi_Da = chi * D_alpha;
    const Eigen::MatrixXd chi_Db = chi * D_beta;
    Eigen::VectorXd rho_a = (chi_Da.array() * chi.array()).rowwise().sum();
    Eigen::VectorXd rho_b = (chi_Db.array() * chi.array()).rowwise().sum();

    Eigen::VectorXd gax, gay, gaz, gbx, gby, gbz;
    Eigen::VectorXd sigma_aa, sigma_ab, sigma_bb;
    if (is_gga || is_mgga) {
        gax = 2.0 * chi_Da.cwiseProduct(dchi[0]).rowwise().sum();
        gay = 2.0 * chi_Da.cwiseProduct(dchi[1]).rowwise().sum();
        gaz = 2.0 * chi_Da.cwiseProduct(dchi[2]).rowwise().sum();
        gbx = 2.0 * chi_Db.cwiseProduct(dchi[0]).rowwise().sum();
        gby = 2.0 * chi_Db.cwiseProduct(dchi[1]).rowwise().sum();
        gbz = 2.0 * chi_Db.cwiseProduct(dchi[2]).rowwise().sum();
        sigma_aa = gax.array().square() + gay.array().square() + gaz.array().square();
        sigma_ab = gax.array()*gbx.array() + gay.array()*gby.array() + gaz.array()*gbz.array();
        sigma_bb = gbx.array().square() + gby.array().square() + gbz.array().square();
    }

    // Per-spin kinetic-energy density τ_σ = ½ Σ_μν (D_σ)_μν Σ_d ∂_dχ_μ ∂_dχ_ν.
    Eigen::VectorXd tau_a, tau_b;
    if (is_mgga) {
        tau_a = Eigen::VectorXd::Zero(n_pts);
        tau_b = Eigen::VectorXd::Zero(n_pts);
        for (int d = 0; d < 3; ++d) {
            const Eigen::MatrixXd dchi_Da = dchi[d] * D_alpha;
            const Eigen::MatrixXd dchi_Db = dchi[d] * D_beta;
            tau_a.array() += 0.5
                * dchi_Da.cwiseProduct(dchi[d]).rowwise().sum().array();
            tau_b.array() += 0.5
                * dchi_Db.cwiseProduct(dchi[d]).rowwise().sum().array();
        }
    }

    Eigen::VectorXd exc, v_rho_a, v_rho_b;
    Eigen::VectorXd v_sig_aa, v_sig_ab, v_sig_bb;
    Eigen::VectorXd v_tau_a, v_tau_b;
    if (is_mgga) {
        func.eval_polarised_mgga(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                                 tau_a, tau_b,
                                 exc, v_rho_a, v_rho_b,
                                 v_sig_aa, v_sig_ab, v_sig_bb,
                                 v_tau_a, v_tau_b);
    } else {
        func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                            exc, v_rho_a, v_rho_b,
                            v_sig_aa, v_sig_ab, v_sig_bb);
    }

    // Flow vectors f_σ_x/y/z on grid.
    Eigen::VectorXd fa_x, fa_y, fa_z, fb_x, fb_y, fb_z;
    Eigen::MatrixXd dot_fa_grad_chi, dot_fb_grad_chi;
    if (is_gga || is_mgga) {
        // Flow vectors: f_α = 2 v_σαα ∇ρ_α + v_σαβ ∇ρ_β  (and α↔β).
        fa_x = 2.0 * v_sig_aa.array() * gax.array() + v_sig_ab.array() * gbx.array();
        fa_y = 2.0 * v_sig_aa.array() * gay.array() + v_sig_ab.array() * gby.array();
        fa_z = 2.0 * v_sig_aa.array() * gaz.array() + v_sig_ab.array() * gbz.array();
        fb_x = 2.0 * v_sig_bb.array() * gbx.array() + v_sig_ab.array() * gax.array();
        fb_y = 2.0 * v_sig_bb.array() * gby.array() + v_sig_ab.array() * gay.array();
        fb_z = 2.0 * v_sig_bb.array() * gbz.array() + v_sig_ab.array() * gaz.array();

        // f_σ · ∇χ_ν(g) — one vector per (g, ν).
        dot_fa_grad_chi = fa_x.asDiagonal() * dchi[0]
                       + fa_y.asDiagonal() * dchi[1]
                       + fa_z.asDiagonal() * dchi[2];
        dot_fb_grad_chi = fb_x.asDiagonal() * dchi[0]
                       + fb_y.asDiagonal() * dchi[1]
                       + fb_z.asDiagonal() * dchi[2];
    }

    auto hess_entry = [&](int c, int d) -> const Eigen::MatrixXd& {
        if (c > d) std::swap(c, d);
        static constexpr int idx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        return (*hess)[idx[c][d]];
    };

    const Eigen::VectorXd w = grid.weights;
    const Eigen::VectorXd w_vra = w.array() * v_rho_a.array();
    const Eigen::VectorXd w_vrb = w.array() * v_rho_b.array();
    // ½ · w · v_τ_σ — the ½ cancels τ's μ↔ν symmetry factor (see header).
    Eigen::VectorXd w_vta_half, w_vtb_half;
    if (is_mgga) {
        w_vta_half = 0.5 * w.array() * v_tau_a.array();
        w_vtb_half = 0.5 * w.array() * v_tau_b.array();
    }

    const auto atom_bf = basis_functions_per_atom(basis, mol);
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    for (int c = 0; c < 3; ++c) {
        // Alpha spin matrix: M_a_{μν}^c
        Eigen::MatrixXd Ma = dchi[c].transpose() * w_vra.asDiagonal() * chi;
        if (is_gga || is_mgga) {
            // Term 2: ∂_c χ_μ × (f_a · ∇χ_ν)
            Ma.noalias() += dchi[c].transpose() * w.asDiagonal()
                            * dot_fa_grad_chi;
            // Term 3: Y_μ^c · χ_ν   where Y = f_a · ∂_c∇χ
            Eigen::MatrixXd Ya = fa_x.asDiagonal() * hess_entry(c, 0)
                               + fa_y.asDiagonal() * hess_entry(c, 1)
                               + fa_z.asDiagonal() * hess_entry(c, 2);
            Ma.noalias() += Ya.transpose() * w.asDiagonal() * chi;
        }
        if (is_mgga) {
            // Term 4 (τ): M4_a_{μν}^c = ½ Σ_d ∂_c∂_dχ_μ (w v_τα) ∂_dχ_ν.
            for (int d = 0; d < 3; ++d) {
                Ma.noalias() += hess_entry(c, d).transpose()
                              * w_vta_half.asDiagonal() * dchi[d];
            }
        }
        // Beta spin matrix: M_b_{μν}^c
        Eigen::MatrixXd Mb = dchi[c].transpose() * w_vrb.asDiagonal() * chi;
        if (is_gga || is_mgga) {
            Mb.noalias() += dchi[c].transpose() * w.asDiagonal()
                            * dot_fb_grad_chi;
            Eigen::MatrixXd Yb = fb_x.asDiagonal() * hess_entry(c, 0)
                               + fb_y.asDiagonal() * hess_entry(c, 1)
                               + fb_z.asDiagonal() * hess_entry(c, 2);
            Mb.noalias() += Yb.transpose() * w.asDiagonal() * chi;
        }
        if (is_mgga) {
            for (int d = 0; d < 3; ++d) {
                Mb.noalias() += hess_entry(c, d).transpose()
                              * w_vtb_half.asDiagonal() * dchi[d];
            }
        }

        for (std::size_t A = 0; A < N; ++A) {
            double acc = 0.0;
            for (const int mu : atom_bf[A]) {
                acc += (D_alpha.row(mu).array() * Ma.row(mu).array()).sum();
                acc += (D_beta .row(mu).array() * Mb.row(mu).array()).sum();
            }
            grad(A, c) = -2.0 * acc;
        }
    }
    return grad;
}

}  // namespace

// UHF / UKS DF two-electron gradient. Composes
//   ∂E_2e/∂R = ∂E_J(D_α + D_β)/∂R
//            + ∂E_K^α(C_α^occ, α_HF/2)/∂R + ∂E_K^β(C_β^occ, α_HF/2)/∂R
// from the existing closed-shell DF gradient kernels. Per-spin α_HF/2
// rescaling — the closed-shell ``compute_k_gradient`` was derived for
// the RHF energy ``E_K^RHF = -2 α_HF Tr[C^T K(CC^T) C]`` (factor of 4
// from D = 2 CC^T squared, ½ from the closed-shell K energy prefactor);
// the per-spin UHF energy ``E_K^σ = -(α_HF/2) Tr[C_σ^T K(C_σ C_σ^T)
// C_σ]`` is exactly ¼ of that at the same C_occ, so calling the
// closed-shell kernel with ``alpha_hf/2`` per spin and summing
// reproduces the right UHF gradient (the closed-shell limit
// C_α = C_β = C_RHF gives 2 · compute_k_gradient(C, α/2) =
// compute_k_gradient(C, α), the RHF expression).
//
// When ``options.cosx`` is set, the K piece routes through the
// seminumerical chain-of-spheres kernel per spin instead of the DF
// K gradient, mirroring the RHF RIJCOSX branch in
// ``compute_gradient``. The COSX kernel takes the spin density and
// computes ∂/∂R of E_K_σ = −(α_HF/2) tr(D_σ · K_cosx[D_σ]); summing
// over spins gives the UHF/UKS RIJCOSX K-gradient. J still goes
// through ``df.compute_j_gradient`` on the total density.
//
// Three (2c+3c) kernel calls total rather than one fused call —
// future fusion optimisation tracked.
static Eigen::MatrixXd uhf_df_two_electron_gradient(
    const Molecule& mol,
    const BasisSet& basis,
    const Eigen::MatrixXd& D_alpha,
    const Eigen::MatrixXd& D_beta,
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    int n_alpha,
    int n_beta,
    double alpha_hf,
    const GradientOptions& options) {
    if (options.aux_basis.empty()) {
        throw std::invalid_argument(
            "compute_gradient_uhf/_uks: density_fit=true requires "
            "aux_basis to be set (e.g. \"def2-svp-jk\"). Use "
            "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"jk\") "
            "for autodetection.");
    }
    const BasisSet aux(mol, options.aux_basis);
    const DensityFitting df(basis, aux);

    const Eigen::MatrixXd D_total = D_alpha + D_beta;
    Eigen::MatrixXd grad_2e = df.compute_j_gradient(mol, D_total);
    if (alpha_hf != 0.0) {
        if (options.cosx) {
            // UHF/UKS RIJCOSX: J via DF (above), K via seminumerical
            // chain-of-spheres on each spin density. The COSX
            // kernel differentiates E_K = −(α_kernel/2) tr(D ·
            // K_cosx) where K_cosx has the closed-shell 1/2 baked
            // into its definition (cosx.cpp). For unrestricted, the
            // per-spin exchange energy is
            //   E_K_σ = −α_HF · tr(D_σ · K_cosx[D_σ])
            // (twice the closed-shell formula per density because
            // D_σ is the 1× per-spin density rather than the 2×
            // closed-shell density). Passing α_kernel = 2 · α_HF
            // recovers the right per-spin gradient and reduces to
            // the RHF expression in the closed-shell limit
            // D_α = D_β = D_RHF/2.
            const int cosx_card =
                cosx_basis_cardinality_from_name(basis.name());
            const Grid cosx_grid_built = build_grid(
                mol, resolve_cosx_grid_options(
                         options.cosx_grid, options.cosx_grid_level, cosx_card));
            const bool cosx_fit =
                resolve_cosx_variant(options.cosx_variant, options.thresh_cosx,
                                     cosx_card, options.cosx_grid_level)
                    != CosxVariant::STANDARD;
            const double cosx_alpha = 2.0 * alpha_hf;
            if (n_alpha > 0) {
                grad_2e += compute_cosx_k_gradient_contribution(
                    mol, basis, D_alpha, cosx_grid_built, cosx_alpha,
                    Eigen::MatrixXd(), cosx_fit);
            }
            if (n_beta > 0) {
                grad_2e += compute_cosx_k_gradient_contribution(
                    mol, basis, D_beta, cosx_grid_built, cosx_alpha,
                    Eigen::MatrixXd(), cosx_fit);
            }
        } else {
            if (n_alpha > 0) {
                const Eigen::MatrixXd C_occ_a = C_alpha.leftCols(n_alpha);
                grad_2e += df.compute_k_gradient(mol, C_occ_a, alpha_hf * 0.5);
            }
            if (n_beta > 0) {
                const Eigen::MatrixXd C_occ_b = C_beta.leftCols(n_beta);
                grad_2e += df.compute_k_gradient(mol, C_occ_b, alpha_hf * 0.5);
            }
        }
    }
    return grad_2e;
}

Eigen::MatrixXd compute_gradient_uks(const Molecule& mol,
                                     const BasisSet& basis,
                                     const UKSResult& result,
                                     const GridOptions& grid_options,
                                     const GradientOptions& options) {
    validate_gradient_ecp_dispatch(options, "compute_gradient_uks");
    if (!result.converged) {
        throw std::runtime_error(
            "compute_gradient_uks: UKS result is not converged");
    }

    Functional functional(result.functional, /*spin=*/2);
    if (functional.is_external()) {
        throw std::runtime_error(
            "compute_gradient_uks: analytic gradients for full-grid "
            "external XC functionals are unavailable because derivatives "
            "of grid coordinates, partition weights, and atom coarse "
            "centers are required; use whole-SCF finite differences");
    }
    const XCKind xc_kind = functional.kind();
    const bool need_hessian = (xc_kind == XCKind::GGA
                               || xc_kind == XCKind::MGGA);
    const double alpha_hf = functional.hf_exchange_fraction();

    // Valence-only electron count when ECPs replace core electrons
    // (matches run_uhf/run_uks: n_elec = mol.n_electrons() - total_ncore).
    const int n_elec =
        mol.n_electrons() - ecp_replaced_core_electrons(mol, options);
    const int mult = mol.multiplicity();
    const int n_alpha = (n_elec + mult - 1) / 2;
    const int n_beta  = (n_elec - mult + 1) / 2;

    // Per-spin energy-weighted density: W_σ = Σ_{i∈occ_σ} ε_{σ,i} C_σ_i C_σ_i^T
    const auto nb = basis.nbasis();
    Eigen::MatrixXd W_alpha = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd W_beta  = Eigen::MatrixXd::Zero(nb, nb);
    for (int i = 0; i < n_alpha; ++i) {
        const Eigen::VectorXd Ci = result.mo_coeffs_alpha.col(i);
        W_alpha += result.mo_energies_alpha(i) * Ci * Ci.transpose();
    }
    for (int i = 0; i < n_beta; ++i) {
        const Eigen::VectorXd Ci = result.mo_coeffs_beta.col(i);
        W_beta  += result.mo_energies_beta (i) * Ci * Ci.transpose();
    }
    const Eigen::MatrixXd W_total = W_alpha + W_beta;

    const Eigen::MatrixXd& D_alpha = result.density_alpha;
    const Eigen::MatrixXd& D_beta  = result.density_beta;
    const Eigen::MatrixXd D_total  = D_alpha + D_beta;

    Grid grid = build_grid(mol, grid_options);
    const Eigen::MatrixXd xc_pulay = accumulate_molecular_xc_grid_batches(
        mol, grid, [&](const Grid& batch) {
            if (need_hessian) {
                AOValuesWithHessian ao =
                    evaluate_ao_with_hessian(basis, batch.points);
                return xc_pulay_gradient_uks(
                    mol, basis, batch, ao.values, ao.gradients, &ao.hessians,
                    D_alpha, D_beta, functional);
            }
            AOValues ao = evaluate_ao_with_gradient(basis, batch.points);
            return xc_pulay_gradient_uks(
                mol, basis, batch, ao.values, ao.gradients, nullptr,
                D_alpha, D_beta, functional);
        });

    Eigen::MatrixXd grad_2e;
    if (options.density_fit) {
        grad_2e = uhf_df_two_electron_gradient(
            mol, basis, D_alpha, D_beta,
            result.mo_coeffs_alpha, result.mo_coeffs_beta,
            n_alpha, n_beta, alpha_hf, options);
    } else {
        grad_2e = two_electron_gradient_contribution_uhf(
            basis, mol, D_alpha, D_beta, alpha_hf);
    }

    return classical_hcore_grad_pieces(mol, basis, D_total, options)
         + overlap_gradient_contribution(basis, mol, W_total)
         + grad_2e
         + xc_pulay;
}

Eigen::MatrixXd xc_pulay_gradient_uks(const Molecule& mol,
                                      const BasisSet& basis,
                                      const Eigen::MatrixXd& D_alpha,
                                      const Eigen::MatrixXd& D_beta,
                                      const std::string& functional_name,
                                      const GridOptions& grid_options) {
    // The XC-Pulay piece of compute_gradient_uks (fixed Becke grid, no weight
    // derivative), factored out and exposed for the Γ-CCM UKS force driver — the
    // open-shell counterpart of xc_pulay_gradient_rks. The CCM XC is the ordinary
    // molecular functional of the supercell spin densities, so it does NOT fold.
    // Overloads (by arity) the internal same-named LDA/GGA/mGGA kernel it wraps.
    Functional functional(functional_name, /*spin=*/2);
    if (functional.is_external()) {
        throw std::runtime_error(
            "xc_pulay_gradient_uks: full-grid external XC gradients are "
            "not implemented");
    }
    const XCKind xc_kind = functional.kind();
    const bool need_hessian = (xc_kind == XCKind::GGA || xc_kind == XCKind::MGGA);
    Grid grid = build_grid(mol, grid_options);
    return accumulate_molecular_xc_grid_batches(
        mol, grid, [&](const Grid& batch) {
            if (need_hessian) {
                AOValuesWithHessian ao =
                    evaluate_ao_with_hessian(basis, batch.points);
                return xc_pulay_gradient_uks(
                    mol, basis, batch, ao.values, ao.gradients, &ao.hessians,
                    D_alpha, D_beta, functional);
            }
            AOValues ao = evaluate_ao_with_gradient(basis, batch.points);
            return xc_pulay_gradient_uks(
                mol, basis, batch, ao.values, ao.gradients, nullptr,
                D_alpha, D_beta, functional);
        });
}

Eigen::MatrixXd compute_gradient_uhf(const Molecule& mol,
                                     const BasisSet& basis,
                                     const UHFResult& result,
                                     const GradientOptions& options) {
    validate_gradient_ecp_dispatch(options, "compute_gradient_uhf");
    if (!result.converged) {
        throw std::runtime_error(
            "compute_gradient_uhf: UHF result is not converged");
    }

    // Valence-only electron count when ECPs replace core electrons
    // (matches run_uhf/run_uks: n_elec = mol.n_electrons() - total_ncore).
    const int n_elec =
        mol.n_electrons() - ecp_replaced_core_electrons(mol, options);
    const int mult = mol.multiplicity();
    const int n_alpha = (n_elec + mult - 1) / 2;
    const int n_beta  = (n_elec - mult + 1) / 2;

    // Per-spin energy-weighted densities (no factor of 2 — UHF densities are
    // one-particle per spin):
    //   W_σ = Σ_{i∈occ_σ} ε_{σ,i} C_{σ,i} C_{σ,i}^T
    const auto nb = basis.nbasis();
    Eigen::MatrixXd W_alpha = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd W_beta  = Eigen::MatrixXd::Zero(nb, nb);
    for (int i = 0; i < n_alpha; ++i) {
        const Eigen::VectorXd Ci = result.mo_coeffs_alpha.col(i);
        W_alpha += result.mo_energies_alpha(i) * Ci * Ci.transpose();
    }
    for (int i = 0; i < n_beta; ++i) {
        const Eigen::VectorXd Ci = result.mo_coeffs_beta.col(i);
        W_beta  += result.mo_energies_beta (i) * Ci * Ci.transpose();
    }
    const Eigen::MatrixXd W_total = W_alpha + W_beta;

    const Eigen::MatrixXd& D_alpha = result.density_alpha;
    const Eigen::MatrixXd& D_beta  = result.density_beta;
    const Eigen::MatrixXd D_total = D_alpha + D_beta;

    Eigen::MatrixXd grad_2e;
    if (options.density_fit) {
        // UHF: α_HF = 1 (no functional scaling).
        grad_2e = uhf_df_two_electron_gradient(
            mol, basis, D_alpha, D_beta,
            result.mo_coeffs_alpha, result.mo_coeffs_beta,
            n_alpha, n_beta, /*alpha_hf=*/1.0, options);
    } else {
        grad_2e = two_electron_gradient_contribution_uhf(
            basis, mol, D_alpha, D_beta);
    }

    return classical_hcore_grad_pieces(mol, basis, D_total, options)
         + overlap_gradient_contribution(basis, mol, W_total)
         + grad_2e;
}

Eigen::MatrixXd two_electron_gradient_casscf(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::Ref<const Eigen::VectorXd>& gamma_ao_flat) {
    // Analytic nuclear gradient of the two-electron part of the CASSCF energy
    // at a general AO 2-particle density Γ_μνλσ.  Same shell-quartet
    // derivative-integral loop as two_electron_gradient_contribution, but
    // replaces the RHF density-product Γ with the supplied Γ_μνλσ.
    //
    // Eq. (12.5.7) of Helgaker, Jørgensen & Olsen, "Molecular Electronic-
    // Structure Theory":
    //   dE_2/dR[A, d] = Σ_μνλσ Γ_μνλσ · ∂(μν|λσ)/∂R[A, d]
    //
    // gamma_ao_flat is row-major over the 4 AO indices:
    //   idx = ((mu * nb + nu) * nb + lam) * nb + sig
    // where nb = basis.nbasis().
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();
    const auto nb = static_cast<Eigen::Index>(basis.nbasis());

    // Validate the supplied tensor size.
    const Eigen::Index expected = nb * nb * nb * nb;
    if (gamma_ao_flat.size() != expected) {
        throw std::invalid_argument(
            "two_electron_gradient_casscf: gamma_ao_flat has size "
            + std::to_string(gamma_ao_flat.size()) + ", expected "
            + std::to_string(expected) + " (= nb^4 with nb = "
            + std::to_string(nb) + ")");
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3);

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), 3));

    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        for (int s2 = 0; s2 <= s1; ++s2) {
            for (int s3 = 0; s3 <= s1; ++s3) {
                const int s4_max = (s1 == s3) ? s2 : s3;
                for (int s4 = 0; s4 <= s4_max; ++s4) {
                    const double s12_deg = (s1 == s2) ? 1.0 : 2.0;
                    const double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                    const double s12_34_deg =
                        (s1 == s3) ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
                    const double weight = s12_deg * s34_deg * s12_34_deg;

                    // L-canonical reorder (mirrors two_electron_gradient_contribution).
                    int sa = s1, sb = s2, sc = s3, sd = s4;
                    int la = shells[sa].contr[0].l;
                    int lb = shells[sb].contr[0].l;
                    int lc = shells[sc].contr[0].l;
                    int ld = shells[sd].contr[0].l;
                    if (la < lb) { std::swap(sa, sb); std::swap(la, lb); }
                    if (lc < ld) { std::swap(sc, sd); std::swap(lc, ld); }
                    if (la + lb > lc + ld) {
                        std::swap(sa, sc); std::swap(sb, sd);
                        std::swap(la, lc); std::swap(lb, ld);
                    }

                    const auto bfa = shell2bf[sa];
                    const auto nA  = shells[sa].size();
                    const auto bfb = shell2bf[sb];
                    const auto nB  = shells[sb].size();
                    const auto bfc = shell2bf[sc];
                    const auto nC  = shells[sc].size();
                    const auto bfd = shell2bf[sd];
                    const auto nD  = shells[sd].size();

                    engine.compute(shells[sa], shells[sb],
                                   shells[sc], shells[sd]);
                    if (buf[0] == nullptr) continue;

                    const long centers[4] = {
                        s2a[sa], s2a[sb], s2a[sc], s2a[sd]};

                    for (int icenter = 0; icenter < 4; ++icenter) {
                        for (int d = 0; d < 3; ++d) {
                            const double* block = buf[icenter * 3 + d];
                            if (!block) continue;
                            double acc = 0.0;
                            for (std::size_t i = 0; i < nA; ++i) {
                                const auto mu = bfa + i;
                                const Eigen::Index mu_nb = mu * nb;
                                for (std::size_t j = 0; j < nB; ++j) {
                                    const auto nu = bfb + j;
                                    const Eigen::Index mu_nb_nu = mu_nb + nu;
                                    const Eigen::Index mu_nb_nu_nb =
                                        (mu_nb_nu) * nb;
                                    for (std::size_t k = 0; k < nC; ++k) {
                                        const auto lam = bfc + k;
                                        const Eigen::Index base =
                                            (mu_nb_nu_nb + lam) * nb;
                                        for (std::size_t l = 0; l < nD; ++l) {
                                            const auto sig = bfd + l;
                                            // Γ_μνλσ from the flat tensor.
                                            const double gamma =
                                                gamma_ao_flat[base + sig];
                                            acc += gamma * block[
                                                ((i * nB + j) * nC + k) * nD
                                                + l];
                                        }
                                    }
                                }
                            }
                            grad_local(centers[icenter], d) += weight * acc;
                        }
                    }
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;
    return grad;
}

}  // namespace vibeqc
