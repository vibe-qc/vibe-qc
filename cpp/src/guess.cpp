#include "vibeqc/guess.hpp"

#include <Eigen/Eigenvalues>
#include <libint2/atom.h>
#include <libint2/basis.h>
#include <libint2/engine.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fstream>
#include <sstream>

#include <libint2/chemistry/elements.h>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/fock.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/lattice_sum.hpp"   // direct_lattice_cells, LatticeSumOptions
#include "vibeqc/lattice_pair_cells.hpp"  // pair_complete_cells, pair_in_range
#include "vibeqc/linear_dependence.hpp"
#include "vibeqc/periodic.hpp"      // PeriodicSystem (periodic MINAO guess)

namespace vibeqc {

namespace {

// Shell-to-atom mapping helper (mirror of the private one in gradient.cpp
// — kept local to avoid a cross-compilation-unit dependency).
std::vector<long> shell_to_atom_map(const BasisSet& basis, const Molecule& mol) {
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
    return basis.libint().shell2atom(atoms);
}

// Basis-function ranges [bf_first, bf_last) that each atom owns.
std::vector<std::pair<std::size_t, std::size_t>>
atom_basis_ranges(const BasisSet& basis, const Molecule& mol) {
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom_map(basis, mol);
    const std::size_t natoms = mol.atoms().size();

    std::vector<std::pair<std::size_t, std::size_t>> ranges(
        natoms, {basis.nbasis(), 0});
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const long atom = s2a[s];
        if (atom < 0 || static_cast<std::size_t>(atom) >= natoms) continue;
        const std::size_t first = shell2bf[s];
        const std::size_t last  = first + shells[s].size();
        auto& r = ranges[atom];
        if (first < r.first) r.first = first;
        if (last  > r.second) r.second = last;
    }
    // Atoms with no basis functions (shouldn't happen with standard
    // basis sets but defensive) collapse to [0,0).
    for (auto& r : ranges) {
        if (r.first == basis.nbasis()) r = {0, 0};
    }
    return ranges;
}

// Aufbau-fill an occupation vector using eigenvalue groups: orbitals with
// energies within `degeneracy_tol` of each other are treated as a single
// group and receive identical fractional occupation.
//
//   * n electrons available (not #orbitals — the total "number of electrons"
//     to distribute, with each orbital able to hold 2).
//   * eps (ascending) — orbital energies.
//
// Returns the occupation of each orbital.
Eigen::VectorXd aufbau_fractional_occupations(
    const Eigen::VectorXd& eps, double n_electrons,
    double degeneracy_tol = 1e-6) {
    const auto m = eps.size();
    Eigen::VectorXd occ = Eigen::VectorXd::Zero(m);
    double remaining = n_electrons;

    Eigen::Index i = 0;
    while (i < m && remaining > 1e-12) {
        // Find the extent of the degenerate group starting at i.
        Eigen::Index j = i + 1;
        while (j < m && std::abs(eps(j) - eps(i)) < degeneracy_tol) ++j;
        const double group_size = static_cast<double>(j - i);
        const double group_capacity = 2.0 * group_size;
        if (remaining >= group_capacity) {
            for (Eigen::Index k = i; k < j; ++k) occ(k) = 2.0;
            remaining -= group_capacity;
        } else {
            const double each = remaining / group_size;
            for (Eigen::Index k = i; k < j; ++k) occ(k) = each;
            remaining = 0.0;
        }
        i = j;
    }
    return occ;
}

// Ground-state electron configuration of a neutral atom, expressed as
// per-angular-momentum electron totals {n_s, n_p, n_d, n_f}. Used to pin
// SAD atomic occupations by l-channel instead of by a global eigenvalue
// sort.
//
// The baseline is the Madelung (Aufbau) filling order; the experimentally
// observed neutral-atom ground states deviate from it for the standard
// set of "anomalous" d/f-block elements (Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag,
// La, Ce, Gd, Pt, Au, Ac, Th, Pa, U, Np, Cm, Lr), which are pinned
// explicitly below. This mirrors what PySCF (scf.atom_hf via
// elements.CONFIGURATION), Psi4 and ORCA do for their atomic guesses:
// fill tabulated per-subshell occupations rather than sorting bare-atom
// eigenvalues, which the 3d/4s/4p near-degeneracy makes unreliable.
//
// Neutral-atom ground-state configurations are CODATA/NIST
// shared-background data (NIST Atomic Spectra Database); there is no
// single citable publication, so no citation-database entry is required
// (CLAUDE.md §8).
std::array<int, 4> ground_config_per_l(int Z) {
    // Anomalous ground states (deviation from strict Madelung order).
    // Value = full {s, p, d, f} electron totals; comment = the valence
    // configuration the totals encode.
    static const std::map<int, std::array<int, 4>> anomalies = {
        {24,  {7, 12, 5, 0}},     // Cr  [Ar] 3d5 4s1
        {29,  {7, 12, 10, 0}},    // Cu  [Ar] 3d10 4s1
        {41,  {9, 18, 14, 0}},    // Nb  [Kr] 4d4 5s1
        {42,  {9, 18, 15, 0}},    // Mo  [Kr] 4d5 5s1
        {44,  {9, 18, 17, 0}},    // Ru  [Kr] 4d7 5s1
        {45,  {9, 18, 18, 0}},    // Rh  [Kr] 4d8 5s1
        {46,  {8, 18, 20, 0}},    // Pd  [Kr] 4d10 5s0
        {47,  {9, 18, 20, 0}},    // Ag  [Kr] 4d10 5s1
        {57,  {12, 24, 21, 0}},   // La  [Xe] 5d1 6s2
        {58,  {12, 24, 21, 1}},   // Ce  [Xe] 4f1 5d1 6s2
        {64,  {12, 24, 21, 7}},   // Gd  [Xe] 4f7 5d1 6s2
        {78,  {11, 24, 29, 14}},  // Pt  [Xe] 4f14 5d9 6s1
        {79,  {11, 24, 30, 14}},  // Au  [Xe] 4f14 5d10 6s1
        {89,  {14, 30, 31, 14}},  // Ac  [Rn] 6d1 7s2
        {90,  {14, 30, 32, 14}},  // Th  [Rn] 6d2 7s2
        {91,  {14, 30, 31, 16}},  // Pa  [Rn] 5f2 6d1 7s2
        {92,  {14, 30, 31, 17}},  // U   [Rn] 5f3 6d1 7s2
        {93,  {14, 30, 31, 18}},  // Np  [Rn] 5f4 6d1 7s2
        {96,  {14, 30, 31, 21}},  // Cm  [Rn] 5f7 6d1 7s2
        {103, {14, 31, 30, 28}},  // Lr  [Rn] 5f14 7s2 7p1
    };
    const auto it = anomalies.find(Z);
    if (it != anomalies.end()) return it->second;

    // Madelung order, by l-channel (subshells in fill order):
    //   1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p
    // — covers Z up to 118. Each subshell holds 2(2l+1) electrons.
    static const std::array<int, 19> fill_l = {
        0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1};
    std::array<int, 4> per_l = {0, 0, 0, 0};
    int remaining = Z;
    for (const int l : fill_l) {
        if (remaining <= 0) break;
        const int cap = 2 * (2 * l + 1);
        const int fill = std::min(cap, remaining);
        per_l[static_cast<std::size_t>(l)] += fill;
        remaining -= fill;
    }
    return per_l;
}

// l-channel label of every MO via its largest Mulliken population. For a
// spherically-averaged atom S and the Fock are block-diagonal in l, so a
// converged atomic MO is a pure-l function and its dominant population names
// its channel. Shared by the SAD occupation fill and the Hund spin split.
std::vector<int> mulliken_mo_l(const Eigen::MatrixXd& C,
                               const Eigen::MatrixXd& S,
                               const std::vector<int>& ao_l) {
    const Eigen::Index m = C.cols();
    const Eigen::MatrixXd SC = S * C;
    std::vector<int> mo_l(static_cast<std::size_t>(m), 0);
    for (Eigen::Index j = 0; j < m; ++j) {
        std::array<double, 8> pop{};
        for (Eigen::Index mu = 0; mu < C.rows(); ++mu) {
            const int l = ao_l[static_cast<std::size_t>(mu)];
            if (l >= 0 && l < 8) {
                pop[static_cast<std::size_t>(l)] += C(mu, j) * SC(mu, j);
            }
        }
        int best = 0;
        for (int l = 1; l < 8; ++l) {
            if (pop[static_cast<std::size_t>(l)] >
                pop[static_cast<std::size_t>(best)]) best = l;
        }
        mo_l[static_cast<std::size_t>(j)] = best;
    }
    return mo_l;
}

// Hund's-rule spin split of the spherically-averaged total occupations into
// (occ_alpha, occ_beta) for a MAJORITY-ALPHA atom. Within each l-channel the
// open subshell carries `unpaired` singly-occupied orbitals (Hund's first
// rule); everything else pairs. We resolve that net moment over the
// channel's occupied MOs in proportion to their total occupancy, so the
// per-spin density stays spherical, occ_alpha + occ_beta == occ exactly, and
// trace((D_alpha - D_beta)·S) == sum_l unpaired_l = the free-atom moment
// (e.g. Fe d6 -> 4, Cr 3d5 4s1 -> 6, O 2p4 -> 2). This is the per-atom seed
// CRYSTAL's ATOMSPIN sets; the caller applies the sign (swap for majority
// beta, average for a non-magnetic tag).
std::pair<Eigen::VectorXd, Eigen::VectorXd> spin_resolve_occupations(
    const Eigen::MatrixXd& C, const Eigen::MatrixXd& S,
    const std::vector<int>& ao_l, const std::array<int, 4>& per_l,
    const Eigen::VectorXd& occ) {
    const Eigen::Index m = occ.size();
    const std::vector<int> mo_l = mulliken_mo_l(C, S, ao_l);
    Eigen::VectorXd occ_a = 0.5 * occ;   // default: every orbital paired
    Eigen::VectorXd occ_b = 0.5 * occ;
    for (int l = 0; l < 4; ++l) {
        const int N = per_l[static_cast<std::size_t>(l)];
        if (N <= 0) continue;
        const int cap = 2 * (2 * l + 1);     // closed-subshell capacity
        const int half = 2 * l + 1;          // orbitals per subshell
        const int r = N % cap;               // electrons in the open subshell
        const int unpaired = (r <= half) ? r : (cap - r);
        if (unpaired == 0) continue;         // closed channel stays paired
        double Nl = 0.0;                      // actual occ summed in channel
        for (Eigen::Index j = 0; j < m; ++j) {
            if (mo_l[static_cast<std::size_t>(j)] == l) Nl += occ(j);
        }
        if (Nl < 1e-12) continue;
        const double moment = std::min(static_cast<double>(unpaired), Nl);
        const double a_frac = 0.5 * (1.0 + moment / Nl);
        const double b_frac = 0.5 * (1.0 - moment / Nl);
        for (Eigen::Index j = 0; j < m; ++j) {
            if (mo_l[static_cast<std::size_t>(j)] != l) continue;
            occ_a(j) = occ(j) * a_frac;
            occ_b(j) = occ(j) * b_frac;
        }
    }
    return {occ_a, occ_b};
}

// Assign isolated-atom SAD occupations per angular-momentum channel from a
// ground-state configuration, instead of one global eigenvalue sort.
//
// For a spherically-averaged atom both S and the Fock are block-diagonal
// in l, so every MO is a pure-l function. We identify each MO's l via a
// Mulliken projection, then within each l-block Aufbau-fill the
// lowest-energy orbitals up to the tabulated electron count for that l.
// Near-degenerate MOs inside a block share fractional occupation equally,
// keeping the atomic density spherical. This fixes the transition-metal
// mis-occupation a global eigenvalue sort produces from the 3d/4s/4p
// near-degeneracy (e.g. Ni → 3d8 4s2, not 3d10 4s0).
//
// If the basis is too small to hold the tabulated configuration (some
// l-block lacks enough functions), we fall back to a global Aufbau fill so
// the electron count is still exactly Z.
Eigen::VectorXd configuration_occupations(
    const Eigen::MatrixXd& C, const Eigen::VectorXd& eps,
    const Eigen::MatrixXd& S, const std::vector<int>& ao_l,
    const std::array<int, 4>& per_l_counts,
    double degeneracy_tol = 1e-6) {
    const Eigen::Index m = eps.size();
    Eigen::VectorXd occ = Eigen::VectorXd::Zero(m);

    // l-channel of each MO via its Mulliken population (a pure-l MO lands in
    // one channel; MOs dominated by l >= 4 get an out-of-range label and are
    // skipped by the fill).
    const std::vector<int> mo_l = mulliken_mo_l(C, S, ao_l);

    // Per-l Aufbau fill up to the tabulated electron count.
    for (int l = 0; l < 4; ++l) {
        double remaining =
            static_cast<double>(per_l_counts[static_cast<std::size_t>(l)]);
        if (remaining <= 0.0) continue;
        std::vector<Eigen::Index> idx;
        for (Eigen::Index j = 0; j < m; ++j) {
            if (mo_l[static_cast<std::size_t>(j)] == l) idx.push_back(j);
        }
        std::size_t i = 0;
        while (i < idx.size() && remaining > 1e-12) {
            std::size_t k = i + 1;
            while (k < idx.size() &&
                   std::abs(eps(idx[k]) - eps(idx[i])) < degeneracy_tol) {
                ++k;
            }
            const double group = static_cast<double>(k - i);
            const double capacity = 2.0 * group;
            if (remaining >= capacity) {
                for (std::size_t t = i; t < k; ++t) occ(idx[t]) = 2.0;
                remaining -= capacity;
            } else {
                const double each = remaining / group;
                for (std::size_t t = i; t < k; ++t) occ(idx[t]) = each;
                remaining = 0.0;
            }
            i = k;
        }
    }

    // Electron-count guard: if the basis couldn't represent the tabulated
    // configuration, degrade to the global Aufbau fill (exactly Z).
    double target = 0.0;
    for (const int n : per_l_counts) target += n;
    if (std::abs(occ.sum() - target) > 1e-6) {
        return aufbau_fractional_occupations(eps, target, degeneracy_tol);
    }
    return occ;
}

// Converged spherically-averaged atomic SCF for one isolated atom.
// ``C`` / ``eps`` are the atomic MO coefficients + orbital energies in
// the element's AO ordering; ``occ`` the fractional occupations; ``D``
// the total density (trace D·S = Z). HUECKEL consumes C + eps to build
// the GWH effective Hamiltonian; SAD/MINAO use only D.
struct AtomicSCF {
    Eigen::MatrixXd C;
    Eigen::VectorXd eps;
    Eigen::VectorXd occ;
    Eigen::MatrixXd D;
    Eigen::VectorXd occ_alpha;  // Hund spin split (majority-alpha); see
    Eigen::VectorXd occ_beta;   // spin_resolve_occupations
    Eigen::MatrixXd S;
};

// RHF-style SCF on a single isolated atom with fractional occupations
// (spherically averaged). Returns the converged orbitals + total density
// (each diagonal element of D = orbital occupancy with both spins
// included).
std::array<int, 4> valence_config_per_l(int Z, int ncore) {
    // Closed cores of the admitted semilocal ECP families. In particular,
    // 28 is [Ar]3d10, NOT the neutral Ni configuration, and 60 is
    // [Kr]4d10 4f14. Andrae et al. (1990), Sec. 1-2, retain ns/np/nd/(n+1)s
    // for the 28/60-core transition series. Never change the chemical Z.
    static const std::map<int, std::array<int, 4>> cores{
        {0, {0, 0, 0, 0}}, {2, {2, 0, 0, 0}},
        {10, {4, 6, 0, 0}}, {18, {6, 12, 0, 0}},
        {28, {6, 12, 10, 0}}, {36, {8, 18, 10, 0}},
        {46, {8, 18, 20, 0}}, {54, {10, 24, 20, 0}},
        {60, {8, 18, 20, 14}}, {68, {10, 24, 20, 14}},
        {78, {10, 24, 30, 14}}, {80, {12, 24, 30, 14}},
        {86, {12, 30, 30, 14}},
    };
    const auto core = cores.find(ncore);
    if (core == cores.end()) throw GuessCapabilityError(
        "initial guess: no closed-core atomic occupation contract for ECP ncore="
        + std::to_string(ncore) + "; use HCORE or a validated READ source");
    auto result = ground_config_per_l(Z);
    for (int l = 0; l < 4; ++l) {
        result[l] -= core->second[l];
        if (result[l] < 0) throw std::invalid_argument(
            "initial guess: ECP closed core is incompatible with this element");
    }
    return result;
}

AtomicSCF atomic_scf(int Z, const std::string& basis_name,
                     const BasisSet* actual_basis = nullptr, int ncore = 0,
                     const Eigen::MatrixXd* v_ecp = nullptr) {
    ensure_libint_initialized();

    // One-atom molecule at origin. The Molecule ctor would reject it for
    // lone atoms with odd Z + default multiplicity 1, so we bypass that
    // check by constructing the libint basis directly and skipping our
    // own Molecule for the atomic SCF.
    std::vector<libint2::Atom> atoms{libint2::Atom{Z, 0.0, 0.0, 0.0}};
    std::string name_lower = basis_name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    libint2::BasisSet shells;
    try {
        shells = actual_basis ? actual_basis->libint()
                              : libint2::BasisSet(name_lower, atoms);
    } catch (const std::exception& e) {
        // libint throws (rather than returning empty) for bogus
        // LIBINT_DATA_PATH and malformed .g94 parses. Translate into a
        // SAD-context error message so users hitting it from the SCF
        // initial-guess path get a directive trace.
        throw std::runtime_error(
            std::string("SAD: failed to load basis '") + basis_name +
            "' for Z=" + std::to_string(Z) + ". libint2 reported: " +
            e.what());
    }
    if (!actual_basis) shells.set_pure(true);

    const std::size_t nbf = shells.nbf();
    if (nbf == 0) {
        throw std::runtime_error(
            "SAD: basis " + basis_name + " has no functions for Z=" +
            std::to_string(Z));
    }

    // Angular momentum l of each basis function, in libint's bf ordering
    // (functions grouped by shell, then by contraction within the shell).
    // The spherically-averaged atomic Fock is block-diagonal in l, so SAD
    // occupations are pinned per l-channel — see configuration_occupations.
    std::vector<int> ao_l;
    ao_l.reserve(nbf);
    for (std::size_t s = 0; s < shells.size(); ++s) {
        for (const auto& c : shells[s].contr) {
            const int nf = c.pure ? (2 * c.l + 1)
                                  : ((c.l + 1) * (c.l + 2) / 2);
            for (int i = 0; i < nf; ++i) ao_l.push_back(c.l);
        }
    }

    if (actual_basis) {
        const auto populations = valence_config_per_l(Z, ncore);
        for (int l = 0; l < 4; ++l) {
            const auto capacity = 2 * std::count(ao_l.begin(), ao_l.end(), l);
            if (populations[l] > capacity) throw GuessCapabilityError(
                "initial guess: actual atomic basis cannot represent ECP valence "
                "occupation in l=" + std::to_string(l));
        }
    }

    // Compute one-electron integrals directly with libint (we can't use
    // compute_overlap etc. because those need a vibeqc::BasisSet which
    // requires a vibeqc::Molecule).
    auto compute_1e = [&](libint2::Operator op,
                          const std::vector<std::pair<double,
                                                      std::array<double,3>>>*
                              nuclei = nullptr) {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbf, nbf);
        libint2::Engine engine(op, shells.max_nprim(), shells.max_l(), 0);
        if (op == libint2::Operator::nuclear && nuclei) {
            engine.set_params(*nuclei);
        }
        const auto& buf = engine.results();
        const auto shell2bf = shells.shell2bf();
        for (std::size_t s1 = 0; s1 < shells.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            for (std::size_t s2 = 0; s2 <= s1; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                engine.compute(shells[s1], shells[s2]);
                const double* block = buf[0];
                if (!block) continue;
                for (std::size_t i = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        const double v = block[i * n2 + j];
                        M(bf1 + i, bf2 + j) = v;
                        if (s1 != s2) M(bf2 + j, bf1 + i) = v;
                    }
                }
            }
        }
        return M;
    };

    const Eigen::MatrixXd S = compute_1e(libint2::Operator::overlap);
    const Eigen::MatrixXd T = compute_1e(libint2::Operator::kinetic);
    std::vector<std::pair<double, std::array<double, 3>>> q{
        {static_cast<double>(Z - ncore), {0.0, 0.0, 0.0}}};
    const Eigen::MatrixXd V = compute_1e(libint2::Operator::nuclear, &q);
    Eigen::MatrixXd Hcore = T + V;
    if (v_ecp) Hcore += *v_ecp;

    // ERI via our helper: build a vibeqc::BasisSet that points at the
    // libint BasisSet. Easier: reuse the shell list through a small
    // vibeqc::BasisSet-like adapter. Cleanest: make Eri4D from a direct
    // libint engine loop (mirrors compute_eri but using local `shells`).
    Eri4D eri;
    eri.n = nbf;
    eri.data.assign(nbf * nbf * nbf * nbf, 0.0);
    {
        libint2::Engine engine(libint2::Operator::coulomb,
                               shells.max_nprim(), shells.max_l(), 0);
        const auto& buf = engine.results();
        const auto shell2bf = shells.shell2bf();
        for (std::size_t s1 = 0; s1 < shells.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            for (std::size_t s2 = 0; s2 <= s1; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                for (std::size_t s3 = 0; s3 <= s1; ++s3) {
                    const auto bf3 = shell2bf[s3];
                    const auto n3 = shells[s3].size();
                    const auto s4_max = (s3 == s1) ? s2 : s3;
                    for (std::size_t s4 = 0; s4 <= s4_max; ++s4) {
                        const auto bf4 = shell2bf[s4];
                        const auto n4 = shells[s4].size();
                        engine.compute(shells[s1], shells[s2],
                                       shells[s3], shells[s4]);
                        const double* block = buf[0];
                        if (!block) continue;
                        for (std::size_t i = 0; i < n1; ++i) {
                            const auto mu = bf1 + i;
                            for (std::size_t j = 0; j < n2; ++j) {
                                const auto nu = bf2 + j;
                                for (std::size_t k = 0; k < n3; ++k) {
                                    const auto lam = bf3 + k;
                                    for (std::size_t l = 0; l < n4; ++l) {
                                        const auto sig = bf4 + l;
                                        const double v = block[
                                            ((i * n2 + j) * n3 + k) * n4 + l];
                                        eri(mu,  nu,  lam, sig) = v;
                                        eri(nu,  mu,  lam, sig) = v;
                                        eri(mu,  nu,  sig, lam) = v;
                                        eri(nu,  mu,  sig, lam) = v;
                                        eri(lam, sig, mu,  nu)  = v;
                                        eri(sig, lam, mu,  nu)  = v;
                                        eri(lam, sig, nu,  mu)  = v;
                                        eri(sig, lam, nu,  mu)  = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // S^{-1/2}
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> se(S);
    const Eigen::VectorXd s_eigs = se.eigenvalues();
    if (s_eigs.minCoeff() < 1e-10) {
        throw std::runtime_error("SAD: atomic overlap is near-singular");
    }
    const Eigen::VectorXd s_inv_sqrt = s_eigs.unaryExpr(
        [](double v) { return 1.0 / std::sqrt(v); });
    const Eigen::MatrixXd X = se.eigenvectors() * s_inv_sqrt.asDiagonal()
                              * se.eigenvectors().transpose();

    // SCF with fractional Aufbau occupations.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fsolver;
    auto diag = [&](const Eigen::MatrixXd& F)
        -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
        const Eigen::MatrixXd Fp = X.transpose() * F * X;
        fsolver.compute(Fp);
        return {X * fsolver.eigenvectors(), fsolver.eigenvalues()};
    };

    // Start from Hcore. Occupations are pinned per angular-momentum
    // channel from the element's ground-state configuration (not a single
    // global eigenvalue sort), so the 3d/4s/4p near-degeneracy doesn't
    // mis-occupy transition-metal atoms — see configuration_occupations /
    // ground_config_per_l.
    const std::array<int, 4> per_l = valence_config_per_l(Z, ncore);
    auto [C, eps] = diag(Hcore);
    Eigen::VectorXd occ = configuration_occupations(C, eps, S, ao_l, per_l);
    Eigen::MatrixXd D = C * occ.asDiagonal() * C.transpose();

    const int max_iter = 50;
    const double tol = 1e-8;
    double E_prev = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        // RHF-style Fock with the total (spin-summed) density D.
        const Eigen::MatrixXd G = build_fock_g(eri, D);
        const Eigen::MatrixXd F = Hcore + G;
        const double E = 0.5 * (D.array() * (Hcore + F).array()).sum();
        if (iter > 0 && std::abs(E - E_prev) < tol) break;
        E_prev = E;

        std::tie(C, eps) = diag(F);
        occ = configuration_occupations(C, eps, S, ao_l, per_l);
        D = C * occ.asDiagonal() * C.transpose();
    }
    auto [occ_a, occ_b] = spin_resolve_occupations(C, S, ao_l, per_l, occ);
    return {C, eps, occ, D, occ_a, occ_b, S};  // trace D·S = Z - ncore
}

// Validate atom/operator correspondence before constructing any atomic integral.
void validate_ecp_context(const Molecule* mol, const GuessECPContext& ecp,
                          bool atomic) {
    // A solvent inner-SCF context can carry a derived XML core count. It is
    // checked against the library below, not mistaken for an inline operator.
    validate_molecular_ecp_dispatch(
        ecp.xml_centers, ecp.xml_library, ecp.primitive_blocks,
        ecp.primitive_centers, ecp.effective_charges,
        ecp.xml_centers.empty() ? ecp.total_ncore : 0, "initial guess");
    if (ecp.total_ncore < 0) throw std::invalid_argument(
        "initial guess: ECP core count must be nonnegative");
    if (!ecp.active() || !mol) return;
    const bool use_inline = !ecp.primitive_blocks.empty();
    if (use_inline && (ecp.primitive_blocks.size() != ecp.primitive_centers.size()
                      || ecp.effective_charges.size() != mol->atoms().size()))
        throw std::invalid_argument("initial guess: ECP operator/center/charge sizes disagree");
    std::vector<int> source(mol->atoms().size(), -1);
    const std::size_t count = use_inline ? ecp.primitive_centers.size()
                                         : ecp.xml_centers.size();
    std::map<int, int> core_by_z;
    if (!use_inline) {
        std::vector<int> zs;
        for (const auto& c : ecp.xml_centers) zs.push_back(c.Z);
        core_by_z = ecp_core_electrons(zs, ecp.xml_library.empty() ? "ecp10mdf" : ecp.xml_library);
    }
    int total_core = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& xyz = use_inline ? ecp.primitive_centers[i] : ecp.xml_centers[i].xyz;
        int matched = -1;
        for (std::size_t a = 0; a < mol->atoms().size(); ++a) {
            double r2 = 0.0;
            for (int d = 0; d < 3; ++d) r2 += std::pow(mol->atoms()[a].xyz[d] - xyz[d], 2);
            if (r2 < 1e-12) {
                if (matched >= 0 || source[a] >= 0)
                    throw std::invalid_argument("initial guess: ambiguous ECP atom placement");
                matched = static_cast<int>(a);
            }
        }
        if (matched < 0) throw std::invalid_argument("initial guess: ECP center does not match an atom");
        const auto& atom = mol->atoms()[matched];
        int core;
        if (use_inline) {
            const double charge = ecp.effective_charges[matched];
            if (!std::isfinite(charge) || charge <= 0 || charge > atom.Z
                || std::abs(charge - std::round(charge)) > 1e-10)
                throw std::invalid_argument("initial guess: ECP effective charge must be a positive integral valence charge");
            core = atom.Z - static_cast<int>(std::round(charge));
        } else {
            if (ecp.xml_centers[i].Z != atom.Z || !core_by_z.count(atom.Z))
                throw std::invalid_argument("initial guess: XML ECP does not cover its atom");
            core = core_by_z.at(atom.Z);
        }
        if (atomic) valence_config_per_l(atom.Z, core);
        source[matched] = static_cast<int>(i);
        total_core += core;
    }
    if (use_inline) {
        for (std::size_t a = 0; a < source.size(); ++a)
            if (source[a] < 0 && ecp.effective_charges[a] != mol->atoms()[a].Z)
                throw std::invalid_argument("initial guess: reduced charge has no ECP operator");
    }
    if ((use_inline || ecp.total_ncore != 0) && total_core != ecp.total_ncore)
        throw std::invalid_argument("initial guess: total ECP core count disagrees with atom operators");
}

// Atom-local corrections follow Van Lenthe (2006), Procedure p. 927.
// ECP atoms use their actual shells and one-center operator. Element-only
// caching is retained for the unchanged AE path; ECP states are keyed by
// atom so different cores/operators/bases on equal-Z atoms cannot alias.
const AtomicSCF& atomic_state_for_atom(
    const Molecule& mol, const BasisSet& basis, std::size_t a,
    const GuessECPContext& ecp, std::map<int, AtomicSCF>& cache) {
    const auto& atom = mol.atoms().at(a);
    int source = -1;
    bool use_inline = !ecp.primitive_blocks.empty();
    if (ecp.active()) {
        const std::size_t n = use_inline ? ecp.primitive_centers.size()
                                         : ecp.xml_centers.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& xyz = use_inline ? ecp.primitive_centers[i]
                                         : ecp.xml_centers[i].xyz;
            double r2 = 0.0;
            for (int d = 0; d < 3; ++d) r2 += std::pow(atom.xyz[d] - xyz[d], 2);
            if (r2 < 1e-12) {
                if (source >= 0 || (!use_inline && ecp.xml_centers[i].Z != atom.Z))
                    throw std::invalid_argument("initial guess: ambiguous ECP atom placement");
                source = static_cast<int>(i);
            }
        }
    }
    const int key = !ecp.active() ? atom.Z : -static_cast<int>(a + 1);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    if (!ecp.active()) return cache.emplace(key, atomic_scf(atom.Z, basis.name())).first->second;

    Molecule atom_mol({Atom{atom.Z, {0.0, 0.0, 0.0}}}, 0, atom.Z % 2 + 1);
    std::vector<ShellInfo> atom_shells;
    for (auto shell : basis.shells()) {
        if (shell.atom_index != static_cast<int>(a)) continue;
        if (!shell.pure && shell.l >= 2) throw GuessCapabilityError(
            "initial guess: ECP atomic averaging requires pure d/higher shells");
        shell.atom_index = 0;
        shell.origin = {0.0, 0.0, 0.0};
        atom_shells.push_back(std::move(shell));
    }
    BasisSet atom_basis(atom_mol, atom_shells, basis.name(), true);
    ECPHcore h;
    if (source < 0) {
        h.total_ncore = 0;
        h.V_ecp = Eigen::MatrixXd::Zero(atom_basis.nbasis(), atom_basis.nbasis());
    } else if (use_inline) {
        if (ecp.effective_charges.size() != mol.atoms().size())
            throw std::invalid_argument("initial guess: ECP charges must cover every atom");
        const double removed = atom.Z - ecp.effective_charges[a];
        if (!std::isfinite(removed) || std::abs(removed - std::round(removed)) > 1e-10)
            throw std::invalid_argument("initial guess: ECP core count must be integral");
        const int core = static_cast<int>(std::round(removed));
        valence_config_per_l(atom.Z, core);
        h = compute_ecp_one_electron_from_primitives(
            atom_basis, atom_mol, {{0.0, 0.0, 0.0}},
            {ecp.primitive_blocks.at(source)}, {ecp.effective_charges[a]}, core);
    } else {
        const auto library = ecp.xml_library.empty() ? "ecp10mdf" : ecp.xml_library;
        h = compute_ecp_one_electron(atom_basis, atom_mol,
                                     {{atom.Z, {0.0, 0.0, 0.0}}}, library);
        valence_config_per_l(atom.Z, h.total_ncore);
    }
    return cache.emplace(key, atomic_scf(atom.Z, basis.name(), &atom_basis,
                                         h.total_ncore, &h.V_ecp)).first->second;
}

// Thin wrapper: SAD only needs the converged total density.
Eigen::MatrixXd atomic_sad_density(int Z, const std::string& basis_name) {
    return atomic_scf(Z, basis_name).D;
}

}  // namespace

void validate_guess_ecp(InitialGuess kind, const GuessECPContext& ecp, const Molecule* mol) {
    validate_ecp_context(mol, ecp, kind != InitialGuess::HCORE && kind != InitialGuess::READ);
}

Eigen::MatrixXd sad_density(const Molecule& mol, const BasisSet& basis,
                            const GuessECPContext& ecp) {
    validate_ecp_context(&mol, ecp, true);
    const std::size_t nbf = basis.nbasis();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nbf, nbf);

    // Cache atomic densities per Z so we only do each element's atomic
    // SCF once even for molecules with repeated atoms.
    std::map<int, AtomicSCF> atomic;
    const auto ranges = atom_basis_ranges(basis, mol);

    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const int Z = mol.atoms()[a].Z;
        const auto& D_at = atomic_state_for_atom(mol, basis, a, ecp, atomic).D;
        const auto [first, last] = ranges[a];
        const auto n_at = last - first;
        if (n_at == 0) continue;
        if (n_at != static_cast<std::size_t>(D_at.rows())) {
            throw std::runtime_error(
                "SAD: atomic basis size mismatch for Z=" + std::to_string(Z));
        }
        D.block(first, first, n_at, n_at) = D_at;
    }
    return D;
}

// ============================================================================
//                       SAP: Superposition of Atomic Potentials
// ============================================================================
//
// Lehtola, Visscher, Engel, *J. Chem. Phys.* 152, 144105 (2020).
// DOI:10.1063/5.0004046. Atomic potential V_atom_A(r) is fit as a sum
// of Gaussian-charge Coulomb potentials:
//
//   V_atom_A(r) = -Σ_i c_i^A · erf(√α_i^A · r) / r
//
// (negative sign in front of the sum → attractive; .g94 coefficients
// are stored positive for the dominant attractive contributions).
// V_SAP replaces V_ne entirely — F_SAP = T + V_SAP. The matrix element
// of one (α, R_A) primitive is computed via libint2's erf_nuclear
// engine with attenuation ω = √α and a single point charge at R_A.

namespace {

// SAPExpansion now lives in guess.hpp (shared with the periodic SAP guess);
// see ``vibeqc::SAPExpansion`` / ``sap_expansions``.

// Build a symbol → Z table from libint2's element database. Cached.
const std::map<std::string, int>& symbol_to_z_table() {
    static const std::map<std::string, int> table = [] {
        std::map<std::string, int> t;
        for (const auto& el : libint2::chemistry::get_element_info()) {
            t.emplace(el.symbol, static_cast<int>(el.Z));
        }
        return t;
    }();
    return table;
}

// Parse a SAP .g94 file. The format (per libint convention) is:
//   ! comment lines start with '!'
//   ****                       (separator)
//   <Symbol> 0                 (element header — charge always 0 here)
//   S <N> <scale>              (S-shell with N primitives, scale always 1.00)
//   <α_1> <c_1>
//   ...
//   <α_N> <c_N>
//   ****
//
// All SAP files use a single S-shell per element with the per-element
// primitive count varying. Returns the per-element (α, c) lists.
std::map<int, SAPExpansion> parse_sap_g94(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(
            "SAP: cannot open atomic-potential file " + path);
    }
    const auto& sym2z = symbol_to_z_table();

    std::map<int, SAPExpansion> table;
    std::string line;
    enum class State { LookingForElement, ExpectingShell, ReadingPrimitives };
    State state = State::LookingForElement;
    int current_Z = -1;
    int n_primitives_remaining = 0;

    while (std::getline(in, line)) {
        // Strip trailing whitespace + CR
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '
                                 || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '!') continue;
        if (line.rfind("****", 0) == 0) {
            // Separator: reset to look for next element
            state = State::LookingForElement;
            current_Z = -1;
            n_primitives_remaining = 0;
            continue;
        }
        std::istringstream iss(line);
        switch (state) {
            case State::LookingForElement: {
                std::string symbol;
                int charge;
                if (iss >> symbol >> charge) {
                    auto it = sym2z.find(symbol);
                    if (it == sym2z.end()) {
                        // Unknown symbol — should not happen with bundled files;
                        // skip the element block.
                        state = State::LookingForElement;
                        continue;
                    }
                    current_Z = it->second;
                    state = State::ExpectingShell;
                }
                break;
            }
            case State::ExpectingShell: {
                char shell_type;
                int n;
                double scale;
                if (iss >> shell_type >> n >> scale) {
                    if (shell_type != 'S') {
                        throw std::runtime_error(
                            "SAP: only S-type shells supported, got '"
                            + std::string(1, shell_type) + "' for Z="
                            + std::to_string(current_Z) + " in " + path);
                    }
                    n_primitives_remaining = n;
                    state = State::ReadingPrimitives;
                }
                break;
            }
            case State::ReadingPrimitives: {
                double alpha, coeff;
                if (iss >> alpha >> coeff) {
                    auto& expansion = table[current_Z];
                    expansion.alphas.push_back(alpha);
                    expansion.coeffs.push_back(coeff);
                    --n_primitives_remaining;
                    if (n_primitives_remaining == 0) {
                        // End of this element's shell; next line should
                        // be **** to confirm.
                        state = State::LookingForElement;
                    }
                }
                break;
            }
        }
    }
    return table;
}

// Locate a SAP atomic-potential ``.g94`` table on disk, at RUNTIME.
//
// libint2's imported CMake target bakes ``LIBINT_DATADIR`` in as a
// compile-time macro pointing at the *build* tree. That directory need not
// exist when the job runs — relocatable bundles, wheels installed elsewhere,
// and multi-stage containers all move the data out from under it — so
// resolving the path from the macro alone made SAP (and AUTO, which
// ``resolve_auto`` maps to SAP for molecular closed-shell) unusable on any
// relocated install.
//
// The precedence below mirrors ``libint2::BasisSet::data_path()``, so SAP
// tables resolve exactly where ordinary basis sets already do:
//
//   1. ``$LIBINT_DATA_PATH`` — the runtime override. vibe-qc's Python
//      package also points this at its bundled ``basis_library/`` at import
//      time (``_install_basis_library``), so a relocated wheel resolves with
//      no action from the user.
//   2. ``LIBINT_DATADIR`` — the compile-time value, fallback only.
//   3. the bare filename in the working directory — last-ditch, as before.
//
// Each root is probed as ``<root>/basis/<name>.g94`` and then
// ``<root>/<name>.g94``, matching libint's own "try without /basis"
// fallback. A genuinely missing table reports every location searched.
std::string resolve_sap_data_file(const std::string& sap_basis_name) {
    const std::string leaf = sap_basis_name + ".g94";

    std::vector<std::string> roots;
    // Single path, not a PATH-style list: libint reads LIBINT_DATA_PATH the
    // same way, and it is libint that loads the orbital basis alongside this.
    if (const char* env = std::getenv("LIBINT_DATA_PATH")) {
        if (*env != '\0') roots.emplace_back(env);
    }
#ifdef LIBINT_DATADIR
    roots.emplace_back(LIBINT_DATADIR);
#endif

    std::vector<std::string> tried;
    for (const auto& root : roots) {
        for (const std::string& candidate : {root + "/basis/" + leaf,
                                             root + "/" + leaf}) {
            if (std::ifstream(candidate)) return candidate;
            tried.push_back(candidate);
        }
    }
    if (std::ifstream(leaf)) return leaf;
    tried.push_back(leaf);

    std::string message =
        "SAP: cannot open atomic-potential file '" + leaf + "'. Searched:";
    for (const auto& candidate : tried) message += "\n  " + candidate;
    message +=
        "\nSet LIBINT_DATA_PATH to the directory holding 'basis/" + leaf
        + "' (the same directory libint loads orbital basis sets from).";
    throw std::runtime_error(message);
}

// Cache (resolved path) → table to avoid re-parsing the file each call.
// Keyed on the resolved path rather than the basis name because the same
// name can resolve to different files across LIBINT_DATA_PATH changes within
// one process (basis-library overlays, tests). Re-resolving is a couple of
// stat calls — negligible against the SCF that follows.
const std::map<int, SAPExpansion>& sap_table_for(
    const std::string& sap_basis_name) {
    static std::map<std::string, std::map<int, SAPExpansion>> cache;
    const std::string path = resolve_sap_data_file(sap_basis_name);
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;

    cache[path] = parse_sap_g94(path);
    return cache[path];
}

// Build V_SAP_{μν} = Σ_A Σ_i (-c_i^A) · <χ_μ | erf(√α_i^A · r_A) / r_A | χ_ν>
// via libint2's erf_nuclear engine. Each (atom A, primitive i) pair is
// one engine call — for typical molecules this is 5–30 primitives per
// atom × n_atoms, all cheap once-only at the start of SCF.
Eigen::MatrixXd build_v_sap_matrix(
    const Molecule& mol, const BasisSet& basis,
    const std::map<int, SAPExpansion>& sap_table) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const std::size_t nbf = shells.nbf();
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine engine(libint2::Operator::erf_nuclear,
                           shells.max_nprim(), shells.max_l(), 0);
    const auto& buf = engine.results();
    const auto shell2bf = shells.shell2bf();
    using ParamsType = libint2::operator_traits<
        libint2::Operator::erf_nuclear>::oper_params_type;

    for (const auto& atom : mol.atoms()) {
        const auto it = sap_table.find(atom.Z);
        if (it == sap_table.end()) {
            throw GuessCapabilityError(
                "SAP: no atomic potential data for Z=" + std::to_string(atom.Z));
        }
        const auto& expansion = it->second;
        for (std::size_t i = 0; i < expansion.alphas.size(); ++i) {
            const double omega = std::sqrt(expansion.alphas[i]);
            const double q = -expansion.coeffs[i];  // attractive
            ParamsType params;
            std::get<0>(params) = omega;
            std::get<1>(params) = {
                {q, {atom.xyz[0], atom.xyz[1], atom.xyz[2]}}};
            engine.set_params(params);
            for (std::size_t s1 = 0; s1 < shells.size(); ++s1) {
                const auto bf1 = shell2bf[s1];
                const auto n1 = shells[s1].size();
                for (std::size_t s2 = 0; s2 <= s1; ++s2) {
                    const auto bf2 = shell2bf[s2];
                    const auto n2 = shells[s2].size();
                    engine.compute(shells[s1], shells[s2]);
                    const double* block = buf[0];
                    if (!block) continue;
                    for (std::size_t mu = 0; mu < n1; ++mu) {
                        for (std::size_t nu = 0; nu < n2; ++nu) {
                            const double v = block[mu * n2 + nu];
                            V(bf1 + mu, bf2 + nu) += v;
                            if (s1 != s2) V(bf2 + nu, bf1 + mu) += v;
                        }
                    }
                }
            }
        }
    }
    return V;
}

// SAP path: diagonalise F_SAP = T + V_SAP in a canonically orthogonalised
// AO basis. Lehtola, Visscher, and Engel, JCP 152, 144105 (2020), Sec. II A,
// define this spin-independent one-particle Hamiltonian as the source of the
// approximate molecular orbitals. Closed- and open-shell callers occupy the
// same ordered orbital set with their respective electron counts.
Eigen::MatrixXd sap_orbitals(
    const Molecule& mol, const BasisSet& basis,
    int n_orbitals_required,
    const Eigen::MatrixXd& S,
    double linear_dep_threshold,
    const std::string& sap_basis_name, const GuessECPContext& ecp) {
    const Eigen::MatrixXd T = compute_kinetic(basis);
    const auto& table = sap_table_for(sap_basis_name);
    const Eigen::MatrixXd V_SAP = ecp.active()
        ? compute_vsap_ecp(mol, basis, ecp)
        : build_v_sap_matrix(mol, basis, table);
    const Eigen::MatrixXd F_guess = T + V_SAP;

    const auto orth = canonical_orthogonalizer(S, linear_dep_threshold);
    if (orth.n_kept < n_orbitals_required) {
        throw std::runtime_error(
            "SAP: canonical orthogonaliser dropped too many basis "
            "directions for n_occ=" + std::to_string(n_orbitals_required)
            + " (n_kept=" + std::to_string(orth.n_kept) + ")");
    }
    const Eigen::MatrixXd& X = orth.X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
        X.transpose() * F_guess * X);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("SAP: F_SAP diagonalisation failed");
    }
    return X * es.eigenvectors();
}

// Produce a closed-shell SAP density from the first n_occ orbitals. Caller is
// responsible for passing a valid S matrix; linear_dep_threshold drives the
// orthogonalisation cutoff.
Eigen::MatrixXd sap_density(
    const Molecule& mol, const BasisSet& basis,
    int n_occ,
    const Eigen::MatrixXd& S,
    double linear_dep_threshold,
    const std::string& sap_basis_name, const GuessECPContext& ecp) {
    const Eigen::MatrixXd C = sap_orbitals(
        mol, basis, n_occ, S, linear_dep_threshold, sap_basis_name, ecp);
    const auto C_occ = C.leftCols(n_occ);
    return 2.0 * C_occ * C_occ.transpose();
}

}  // namespace

namespace {
// Exchange-only local SAP on spherical valence HF densities, following the
// LDA-on-HF construction of Lehtola, Visscher & Engel (JCP 152, 144105).
// The actual one-center ECP defines the HF reference. No all-electron core
// density, potential fit, or effective-Z element substitution enters here.
struct ValenceAtomicPotential {
    std::array<double, 3> center;
    Eigen::VectorXd radius, radial_potential; // stores r * v(r)
    double log_min, log_step;
    double at(double r) const {
        r = std::max(r, radius(0));
        if (r >= radius(radius.size()-1)) return 0;
        const double index = (std::log(r) - log_min) / log_step;
        const int lo = std::max(0, std::min(int(radius.size())-2, int(index)));
        const double fraction = (r-radius(lo))/(radius(lo+1)-radius(lo));
        return ((1-fraction)*radial_potential(lo) + fraction*radial_potential(lo+1))/r;
    }
};

std::vector<ValenceAtomicPotential> valence_atomic_potentials(
    const Molecule& mol, const BasisSet& basis, const GuessECPContext& ecp) {
    validate_guess_ecp(InitialGuess::SAP, ecp, &mol);
    std::map<int, AtomicSCF> cache;
    std::vector<ValenceAtomicPotential> potentials;
    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const auto& state = atomic_state_for_atom(mol, basis, a, ecp, cache);
        const double electrons = (state.D * state.S).trace();
        Molecule atom({Atom{mol.atoms()[a].Z, {0,0,0}}}, 0, mol.atoms()[a].Z % 2 + 1);
        std::vector<ShellInfo> shells;
        double min_exp = 1;
        for (auto shell : basis.shells()) {
            if (shell.atom_index != int(a)) continue;
            shell.atom_index = 0; shell.origin = {0,0,0};
            for (double exponent : shell.exponents) min_exp = std::min(min_exp, exponent);
            shells.push_back(std::move(shell));
        }
        BasisSet atomic_basis(atom, shells, basis.name(), true);
        constexpr int n = 8192;
        ValenceAtomicPotential pot;
        pot.center = mol.atoms()[a].xyz;
        pot.log_min = std::log(1e-8);
        pot.log_step = (std::log(std::max(40.0, std::sqrt(40.0/min_exp))) - pot.log_min)/(n-1);
        pot.radius.resize(n);
        Eigen::MatrixX3d points = Eigen::MatrixX3d::Zero(n,3);
        for (int i=0; i<n; ++i) points(i,2) = pot.radius(i) = std::exp(pot.log_min+i*pot.log_step);
        const Eigen::MatrixXd values = evaluate_ao(atomic_basis, points);
        Eigen::VectorXd rho = ((values * state.D).array()*values.array()).rowwise().sum();
        rho = rho.cwiseMax(0);
        Eigen::VectorXd enclosed = Eigen::VectorXd::Zero(n), tail = Eigen::VectorXd::Zero(n);
        const double four_pi = 4*std::acos(-1.0);
        for (int i=1; i<n; ++i) {
            const double r0=pot.radius(i-1), r1=pot.radius(i);
            enclosed(i)=enclosed(i-1)+four_pi*.5*(r1-r0)*(rho(i-1)*r0*r0+rho(i)*r1*r1);
        }
        if (!rho.allFinite() || enclosed(n-1) <= 0 || std::abs(enclosed(n-1)-electrons) > 1e-3)
            throw GuessCapabilityError("SAP: spherical ECP reference failed radial charge integration");
        // Enforce atomic neutrality in the far field; discretization must
        // never leave a spurious 1/r tail in a periodic potential sum.
        const double scale = electrons/enclosed(n-1);
        rho *= scale; enclosed *= scale;
        for (int i=n-2; i>=0; --i) {
            const double r0=pot.radius(i), r1=pot.radius(i+1);
            tail(i)=tail(i+1)+four_pi*.5*(r1-r0)*(rho(i)*r0+rho(i+1)*r1);
        }
        pot.radial_potential.resize(n);
        for (int i=0; i<n; ++i)
            pot.radial_potential(i) = enclosed(i)-electrons + pot.radius(i)*(
                tail(i)-std::cbrt(3*rho(i)/std::acos(-1.0)));
        potentials.push_back(std::move(pot));
    }
    return potentials;
}

Eigen::VectorXd valence_sap_on_grid(
    const std::vector<ValenceAtomicPotential>& atoms, const Grid& grid,
    const std::vector<Eigen::Vector3d>& images) {
    Eigen::VectorXd potential = Eigen::VectorXd::Zero(grid.points.rows());
    for (const auto& atom : atoms) for (const auto& image : images) {
        const Eigen::Vector3d center(atom.center[0], atom.center[1], atom.center[2]);
        for (Eigen::Index i=0; i<potential.size(); ++i)
            potential(i) += atom.at((grid.points.row(i).transpose()-center-image).norm());
    }
    return potential;
}
} // namespace

Eigen::MatrixXd compute_vsap_ecp(
    const Molecule& mol, const BasisSet& basis, const GuessECPContext& ecp) {
    const auto atoms = valence_atomic_potentials(mol, basis, ecp);
    GridOptions grid_options; grid_options.n_radial = 100; grid_options.lebedev_order = 29;
    const auto grid = build_grid(mol, grid_options);
    const Eigen::VectorXd potential = valence_sap_on_grid(atoms, grid, {Eigen::Vector3d::Zero()});
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
    for (Eigen::Index start=0; start<grid.points.rows(); start+=512) {
        const Eigen::Index count = std::min<Eigen::Index>(512, grid.points.rows()-start);
        const Eigen::MatrixXd ao = evaluate_ao(basis, grid.points.middleRows(start,count));
        const Eigen::VectorXd weights = grid.weights.segment(start,count).array()*potential.segment(start,count).array();
        result += ao.transpose() * (ao.array().colwise()*weights.array()).matrix();
    }
    if (!ecp.primitive_blocks.empty())
        result += compute_ecp_one_electron_from_primitives(basis, mol,
            ecp.primitive_centers, ecp.primitive_blocks, ecp.effective_charges, ecp.total_ncore).V_ecp;
    else if (!ecp.xml_centers.empty())
        result += compute_ecp_one_electron(basis, mol, ecp.xml_centers,
            ecp.xml_library.empty() ? "ecp10mdf" : ecp.xml_library).V_ecp;
    return (0.5*(result+result.transpose())).eval();
}

LatticeMatrixSet compute_vsap_ecp_lattice(
    const BasisSet& basis, const PeriodicSystem& system,
    const LatticeSumOptions& opts, const GuessECPContext& ecp) {
    const auto mol = system.unit_cell_molecule();
    const auto atoms = valence_atomic_potentials(mol, basis, ecp);
    if (!ecp.xml_centers.empty()) throw GuessCapabilityError("periodic SAP requires inline ECP operators");
    GridOptions grid_options; grid_options.n_radial = 100; grid_options.lebedev_order = 29;
    const auto grid = build_grid(mol, grid_options);
    std::vector<Eigen::Vector3d> images;
    for (const auto& cell : direct_lattice_cells(system, opts.nuclear_cutoff_bohr)) images.push_back(cell.r_cart);
    const Eigen::VectorXd potential = valence_sap_on_grid(atoms, grid, images);
    auto result = compute_ecp_lattice_from_primitives(basis, system, opts, ecp.primitive_centers, ecp.primitive_blocks);
    for (Eigen::Index start=0; start<grid.points.rows(); start+=512) {
        const Eigen::Index count = std::min<Eigen::Index>(512, grid.points.rows()-start);
        const Eigen::MatrixX3d points = grid.points.middleRows(start,count);
        const Eigen::MatrixXd ao = evaluate_ao(basis, points);
        const Eigen::VectorXd weights = grid.weights.segment(start,count).array()*potential.segment(start,count).array();
        const Eigen::MatrixXd weighted = (ao.array().colwise()*weights.array()).matrix();
        for (std::size_t c=0; c<result.cells.size(); ++c) {
            const Eigen::MatrixX3d shifted = points.rowwise()-result.cells[c].r_cart.transpose();
            result.blocks[c] += weighted.transpose()*evaluate_ao(basis, shifted);
        }
    }
    return result;
}

// Public accessor for the cached per-element SAP expansions (declared in
// guess.hpp). Lets the periodic lattice-summed SAP guess reuse the same fitted
// (α_i, c_i) data the molecular SAP guess parses, instead of re-reading the
// .g94 file.
const std::map<int, SAPExpansion>& sap_expansions(
    const std::string& sap_basis_name) {
    return sap_table_for(sap_basis_name);
}

Eigen::MatrixXd compute_sap_potential_molecular(
    const BasisSet& basis, const Molecule& mol,
    const std::string& sap_basis_name) {
    return build_v_sap_matrix(mol, basis, sap_table_for(sap_basis_name));
}

// ============================================================================
//   PATOM / HUECKEL / MINAO - molecular guesses plus periodic helpers
// ============================================================================
//
// All three are *density-mode*: they return a density matrix the SCF
// driver consumes directly, exactly like SAD (and like SAP, which builds a
// Fock and diagonalises it internally before returning D). Keeping them
// density-mode means no SCF-driver change is needed — the engine does the
// one-off diagonalise / project step, and the driver just starts from D.
//
// The molecular GuessEngine path remains density-mode. Periodic closed-shell
// RHF/RKS wires HUECKEL separately as a lattice Fock-mode guess; periodic
// paths that still reach the engine get a clear "not yet implemented for
// this periodic driver" error, mirroring SAP/MINAO.

namespace {

// libint2::Atom list for a vibeqc::Molecule (mirror of the conversion in
// shell_to_atom_map; pulled out for the MINAO reference basis build).
std::vector<libint2::Atom> molecule_libint_atoms(const Molecule& mol) {
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

// ---- PATOM ----------------------------------------------------------------
//
// "Polarised atom" guess (ORCA PAtom). Start from the SAD density, then take
// a small number of molecular mean-field Fock steps so the superposed
// free-atom densities re-polarise in the actual molecular field. With one
// step this is "SAD + one HF SCF iteration"; the SCF loop then continues
// from a density that already feels its neighbours. Builds on SAD (Van
// Lenthe et al., J. Comput. Chem. 27, 926 (2006)); the in-field step uses
// the HF-like Fock F = Hcore + J(D) − ½K(D) regardless of the target method
// (a guess does not need the XC potential). Needs Hcore + a JKBuilder.
Eigen::MatrixXd patom_density_closed(
    const Molecule& mol, const BasisSet& basis, int n_occ,
    const Eigen::MatrixXd& S, const Eigen::MatrixXd& Hcore,
    const JKBuilder& jk, double linear_dep_threshold, int n_iter,
    const GuessECPContext& ecp) {
    Eigen::MatrixXd D = normalize_guess_density(
        sad_density(mol, basis, ecp), S, 2.0 * n_occ);
    if (n_occ <= 0) return D;
    const auto orth = canonical_orthogonalizer(S, linear_dep_threshold);
    if (orth.n_kept < n_occ) {
        throw std::runtime_error(
            "PATOM: canonical orthogonaliser dropped too many basis "
            "directions for n_occ=" + std::to_string(n_occ)
            + " (n_kept=" + std::to_string(orth.n_kept) + ")");
    }
    const Eigen::MatrixXd& X = orth.X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    for (int it = 0; it < std::max(1, n_iter); ++it) {
        const Eigen::MatrixXd F =
            Hcore + jk.build_J(D) - 0.5 * jk.build_K(D);
        es.compute(X.transpose() * F * X);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error(
                "PATOM: in-field Fock diagonalisation failed");
        }
        const Eigen::MatrixXd C = X * es.eigenvectors();
        const auto C_occ = C.leftCols(n_occ);
        D = 2.0 * C_occ * C_occ.transpose();
    }
    return D;
}

// Open-shell PATOM: Hund-split atomic SAD start, then per-spin in-field
// steps with Fσ = Hcore + J(Dα + Dβ) − K(Dσ).
//
// The Hund split applies only when n_α ≠ n_β. A spin-balanced system must
// start from spin-averaged atomic occupations: a Hund-split start on a
// singlet (any atom with an open-shell Hund ground state — C, N, O, …)
// leaves O(1e-7) spurious α/β polarisation in the converged densities at
// default tolerances, breaks the UKS/RKS singlet parity contract, and
// defeats the spin-degenerate shared-exchange gate in uhf/uks (its 1e-13
// envelope never engages). With a symmetric start and n_α == n_β the
// per-spin refinement below stays bit-identical across spins.
//
// Policy note (guess-spin-symmetry investigation, 2026-08-15; see
// handovers/HANDOVER_GUESS_SPIN_SYMMETRY.md): n_α == n_β is NOT a
// spin-restricted-state proxy — the broken-symmetry singlet class (H2
// past the Coulson-Fischer point, ozone, twisted ethylene, p-benzyne,
// AFM dimers) genuinely breaks symmetry. The default guess is
// deliberately NOT the state-selection mechanism for that class: the
// default-on stability escape (Seeger-Pople 1977; Lehtola 2020 §10)
// descends from the symmetric saddle to the BS minimum, measured
// identical before/after this change on the full set. A deliberate
// AFM pattern is seeded via ATOMSPIN (handled above). The symmetric
// default is the Van Lenthe 2006 standard start.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> patom_density_open(
    const Molecule& mol, const BasisSet& basis, int n_alpha, int n_beta,
    const Eigen::MatrixXd& S, const Eigen::MatrixXd& Hcore,
    const JKBuilder& jk, double linear_dep_threshold, int n_iter,
    const GuessECPContext& ecp) {
    // Build Hund-split per-atom densities (same as the default SAD path).
    const bool spin_balanced = (n_alpha == n_beta);
    const std::size_t nbf = basis.nbasis();
    Eigen::MatrixXd Da = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Db = Eigen::MatrixXd::Zero(nbf, nbf);
    const auto ranges = atom_basis_ranges(basis, mol);
    std::map<int, AtomicSCF> cache;
    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const int Z = mol.atoms()[a].Z;
        const AtomicSCF& at = atomic_state_for_atom(mol, basis, a, ecp, cache);
        const Eigen::VectorXd oa =
            spin_balanced ? Eigen::VectorXd(0.5 * at.occ) : at.occ_alpha;
        const Eigen::VectorXd ob =
            spin_balanced ? Eigen::VectorXd(0.5 * at.occ) : at.occ_beta;
        const Eigen::MatrixXd da =
            at.C * oa.asDiagonal() * at.C.transpose();
        const Eigen::MatrixXd db =
            at.C * ob.asDiagonal() * at.C.transpose();
        const auto [first, last] = ranges[a];
        const auto n_at = last - first;
        if (n_at == 0) continue;
        if (n_at != static_cast<std::size_t>(da.rows())) {
            throw std::runtime_error(
                "PATOM: atomic basis size mismatch for Z="
                + std::to_string(Z));
        }
        Da.block(first, first, n_at, n_at) = da;
        Db.block(first, first, n_at, n_at) = db;
    }
    auto normalized = normalize_guess_spin_densities(Da, Db, S, n_alpha, n_beta);
    Da = std::move(normalized.first);
    Db = std::move(normalized.second);
    const auto orth = canonical_orthogonalizer(S, linear_dep_threshold);
    if (orth.n_kept < std::max(n_alpha, n_beta)) {
        throw std::runtime_error(
            "PATOM: canonical orthogonaliser dropped too many basis "
            "directions for the occupied space");
    }
    const Eigen::MatrixXd& X = orth.X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    auto rebuild = [&](const Eigen::MatrixXd& F, int n_occ) -> Eigen::MatrixXd {
        es.compute(X.transpose() * F * X);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error(
                "PATOM: in-field Fock diagonalisation failed");
        }
        const Eigen::MatrixXd C = X * es.eigenvectors();
        const auto C_occ = C.leftCols(std::max(0, n_occ));
        return C_occ * C_occ.transpose();
    };
    for (int it = 0; it < std::max(1, n_iter); ++it) {
        const Eigen::MatrixXd J = jk.build_J(Da + Db);
        auto [K_alpha, K_beta] = jk.build_K_pair(Da, Db);
        const Eigen::MatrixXd Fa = Hcore + J - K_alpha;
        const Eigen::MatrixXd Fb = Hcore + J - K_beta;
        Da = rebuild(Fa, n_alpha);
        Db = rebuild(Fb, n_beta);
    }
    return {Da, Db};
}

// ---- HUECKEL --------------------------------------------------------------
//
// Parameter-free extended-Hückel / generalised Wolfsberg-Helmholz (GWH)
// guess. For each atom we run the spherically-averaged atomic SCF (the same
// machinery SAD uses) and take its MO coefficients C^A and orbital energies
// ε^A. The occupied atomic orbitals of every atom, embedded in the molecular
// AO space, form the minimal "atomic-orbital basis" Φ (one column per
// occupied atomic MO). Restricting to the occupied set — rather than all
// atomic orbitals incl. virtuals — keeps the atomic-orbital overlap
// Φ^T S Φ well-conditioned on richer bases and matches the classic minimal-
// basis extended-Hückel construction. The GWH effective Hamiltonian is
//
//   H_ii = ε_i,
//   H_ij = ½ K S_ij (ε_i + ε_j)   (i ≠ j),   K = 1.75,
//
// where S_ij = (Φ^T S Φ)_ij is the overlap of atomic orbitals in the
// molecular metric. Solving the generalised symmetric eigenproblem
// H x = S x E and back-transforming C = Φ x yields molecular orbitals with
// C^T S C = I; the first n_occ build the density.
//
// The GWH off-diagonal form is Wolfsberg & Helmholz, J. Chem. Phys. 20, 837
// (1952) with Hoffmann's K = 1.75 (J. Chem. Phys. 39, 1397 (1963)); using
// computed atomic-orbital energies in place of tabulated VSIPs is the
// parameter-free variant assessed by Lehtola, J. Chem. Theory Comput. 15,
// 1593 (2019). Molecular-only (needs the molecular overlap S).
Eigen::MatrixXd huckel_mo_coefficients(
    const Molecule& mol, const BasisSet& basis,
    const Eigen::MatrixXd& S, Eigen::VectorXd* mo_energies_out,
    const GuessECPContext& ecp) {
    const std::size_t nbf = basis.nbasis();
    const auto ranges = atom_basis_ranges(basis, mol);

    // Collect every atom's occupied atomic orbitals, embedded in the full
    // molecular AO space, as the columns of Φ. occ > occ_tol selects the
    // occupied set (spherically-averaged atoms fractionally occupy a whole
    // valence shell, so this keeps the full valence space).
    const double occ_tol = 1e-8;
    std::vector<Eigen::VectorXd> cols;
    std::vector<double> col_eps;
    std::map<int, AtomicSCF> cache;
    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const int Z = mol.atoms()[a].Z;
        const AtomicSCF& at = atomic_state_for_atom(mol, basis, a, ecp, cache);
        const auto [first, last] = ranges[a];
        const auto n_at = last - first;
        if (n_at == 0) continue;
        if (n_at != static_cast<std::size_t>(at.C.rows())) {
            throw std::runtime_error(
                "HUECKEL: atomic basis size mismatch for Z="
                + std::to_string(Z));
        }
        for (Eigen::Index i = 0; i < at.C.cols(); ++i) {
            if (at.occ(i) <= occ_tol) continue;
            Eigen::VectorXd col = Eigen::VectorXd::Zero(nbf);
            col.segment(first, n_at) = at.C.col(i);
            cols.push_back(std::move(col));
            col_eps.push_back(at.eps(i));
        }
    }
    const std::size_t m = cols.size();
    if (m == 0) {
        throw std::runtime_error(
            "HUECKEL: no occupied atomic orbitals found for this molecule");
    }
    Eigen::MatrixXd Phi(nbf, m);
    Eigen::VectorXd eps(m);
    for (std::size_t i = 0; i < m; ++i) {
        Phi.col(static_cast<Eigen::Index>(i)) = cols[i];
        eps(static_cast<Eigen::Index>(i)) = col_eps[i];
    }

    // GWH effective Hamiltonian in the occupied-atomic-orbital basis.
    const Eigen::MatrixXd s = Phi.transpose() * S * Phi;
    const double K = 1.75;  // Wolfsberg-Helmholz / Hoffmann factor
    const Eigen::Index mm = static_cast<Eigen::Index>(m);
    Eigen::MatrixXd H(mm, mm);
    for (Eigen::Index i = 0; i < mm; ++i) {
        H(i, i) = eps(i);
        for (Eigen::Index j = i + 1; j < mm; ++j) {
            const double v = 0.5 * K * s(i, j) * (eps(i) + eps(j));
            H(i, j) = v;
            H(j, i) = v;
        }
    }

    // Generalised symmetric eigenproblem H x = s x E (s is SPD: the occupied
    // atomic orbitals are linearly independent and S is SPD).
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(H, s);
    if (ges.info() != Eigen::Success) {
        throw std::runtime_error(
            "HUECKEL: generalised eigensolve failed (near-singular atomic-"
            "orbital overlap — try SAD or SAP for this basis)");
    }
    if (mo_energies_out) *mo_energies_out = ges.eigenvalues();
    return Phi * ges.eigenvectors();  // nbf × m molecular MOs, C^T S C = I
}

Eigen::VectorXd huckel_ao_energies(
    const Molecule& mol, const BasisSet& basis, const GuessECPContext& ecp) {
    const std::size_t nbf = basis.nbasis();
    Eigen::VectorXd eps_ao = Eigen::VectorXd::Zero(nbf);
    const auto ranges = atom_basis_ranges(basis, mol);
    std::map<int, AtomicSCF> cache;
    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const int Z = mol.atoms()[a].Z;
        const AtomicSCF& at = atomic_state_for_atom(mol, basis, a, ecp, cache);
        const auto [first, last] = ranges[a];
        const auto n_at = last - first;
        if (n_at == 0) continue;
        if (n_at != static_cast<std::size_t>(at.C.rows())) {
            throw std::runtime_error(
                "HUECKEL: atomic basis size mismatch for Z="
                + std::to_string(Z));
        }
        const Eigen::MatrixXd F_at =
            at.S * at.C * at.eps.asDiagonal() * at.C.transpose() * at.S;
        eps_ao.segment(first, n_at) = F_at.diagonal();
    }
    return eps_ao;
}

// ---- MINAO ----------------------------------------------------------------
//
// Project free-atom reference densities, built in a minimal AO basis, onto
// the target basis (the minimal-AO reference idea is Knizia, J. Chem. Theory
// Comput. 9, 4834 (2013); the projection guess is PySCF's init_guess_by_minao,
// Sun et al. 2018; reference data are the ANO-RCC minimal contractions of
// Roos/Widmark). For each atom the reference density D_ref^A is the
// spherically-averaged atomic SCF density in the minimal basis (well-
// conditioned, cheap). Assemble block-diagonal D_ref in the reference basis,
// then project to the target basis with
//
//   P = S_tt^{-1} S_tm,    D = P D_ref P^T,
//
// where S_tt is the target overlap and S_tm = <χ_target | χ_ref> the cross-
// basis overlap (PySCF's project_dm_nr2nr). Molecular-only.

// Per-atom basis-function ranges for a raw libint basis (mirror of
// atom_basis_ranges, which needs a vibeqc::BasisSet + Molecule).
std::vector<std::pair<std::size_t, std::size_t>>
libint_atom_ranges(const libint2::BasisSet& shells,
                   const std::vector<libint2::Atom>& atoms) {
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shells.shell2atom(atoms);
    const std::size_t natoms = atoms.size();
    const std::size_t nbf = shells.nbf();
    std::vector<std::pair<std::size_t, std::size_t>> ranges(natoms, {nbf, 0});
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const long atom = s2a[s];
        if (atom < 0 || static_cast<std::size_t>(atom) >= natoms) continue;
        const std::size_t first = shell2bf[s];
        const std::size_t last = first + shells[s].size();
        auto& r = ranges[static_cast<std::size_t>(atom)];
        if (first < r.first) r.first = first;
        if (last > r.second) r.second = last;
    }
    for (auto& r : ranges) {
        if (r.first == nbf) r = {0, 0};
    }
    return ranges;
}

// Rectangular cross-basis overlap <χ_μ^(1) | χ_ν^(2)>.
Eigen::MatrixXd cross_basis_overlap(const libint2::BasisSet& b1,
                                    const libint2::BasisSet& b2) {
    ensure_libint_initialized();
    const std::size_t n1 = b1.nbf();
    const std::size_t n2 = b2.nbf();
    Eigen::MatrixXd S12 = Eigen::MatrixXd::Zero(n1, n2);
    libint2::Engine engine(
        libint2::Operator::overlap,
        std::max(b1.max_nprim(), b2.max_nprim()),
        std::max(b1.max_l(), b2.max_l()), 0);
    const auto& buf = engine.results();
    const auto sh2bf1 = b1.shell2bf();
    const auto sh2bf2 = b2.shell2bf();
    for (std::size_t s1 = 0; s1 < b1.size(); ++s1) {
        const auto bf1 = sh2bf1[s1];
        const auto nn1 = b1[s1].size();
        for (std::size_t s2 = 0; s2 < b2.size(); ++s2) {
            const auto bf2 = sh2bf2[s2];
            const auto nn2 = b2[s2].size();
            engine.compute(b1[s1], b2[s2]);
            const double* block = buf[0];
            if (!block) continue;
            for (std::size_t i = 0; i < nn1; ++i) {
                for (std::size_t j = 0; j < nn2; ++j) {
                    S12(bf1 + i, bf2 + j) = block[i * nn2 + j];
                }
            }
        }
    }
    return S12;
}

// Lattice-summed cross-basis overlap, folded to Γ:
//   S_tm(Γ)_{μν} = Σ_g ⟨ χ_target_μ(0) | χ_ref_ν(g) ⟩ ,
// the periodic generalisation of cross_basis_overlap. Only the Γ point (k=0)
// is needed for the MINAO density guess, so every cell phase e^{ik·g} is 1 and
// the per-cell rectangular blocks are summed directly. The reference shells
// are translated by each direct-lattice vector within the overlap cutoff
// (mirrors lattice_integrals.cpp's shift_shells).
Eigen::MatrixXd cross_overlap_lattice_gamma(
    const libint2::BasisSet& target, const libint2::BasisSet& ref,
    const PeriodicSystem& system, const LatticeSumOptions& opts) {
    ensure_libint_initialized();
    const std::size_t nt = target.nbf();
    const std::size_t nr = ref.nbf();
    Eigen::MatrixXd S_tm = Eigen::MatrixXd::Zero(nt, nr);
    // #429: under pair_complete_1e the contributing images are set by the
    // physical separation |O_target - O_ref - g| (lattice_pair_cells.hpp);
    // bra and ket are different bases here, so the padding uses their cross
    // offsets and the per-pair filter below restores the cutoff. Off by
    // default: the plain |g| ball, bit-identical to before.
    const auto cells = opts.pair_complete_1e
        ? pair_complete_cells(system, opts.cutoff_bohr, target, ref)
        : direct_lattice_cells(system, opts.cutoff_bohr);

    libint2::Engine engine(
        libint2::Operator::overlap,
        std::max(target.max_nprim(), ref.max_nprim()),
        std::max(target.max_l(), ref.max_l()), 0);
    const auto& buf = engine.results();
    const auto sh2bf_t = target.shell2bf();
    const auto sh2bf_r = ref.shell2bf();

    for (const auto& cell : cells) {
        // Translate every reference shell origin by the lattice vector g.
        std::vector<libint2::Shell> ref_g(ref.begin(), ref.end());
        for (auto& s : ref_g) {
            s.O[0] += cell.r_cart[0];
            s.O[1] += cell.r_cart[1];
            s.O[2] += cell.r_cart[2];
        }
        for (std::size_t s1 = 0; s1 < target.size(); ++s1) {
            const auto bf1 = sh2bf_t[s1];
            const auto n1 = target[s1].size();
            for (std::size_t s2 = 0; s2 < ref_g.size(); ++s2) {
                if (opts.pair_complete_1e &&
                    !pair_in_range(target[s1], ref_g[s2], opts.cutoff_bohr)) {
                    continue;
                }
                const auto bf2 = sh2bf_r[s2];
                const auto n2 = ref_g[s2].size();
                engine.compute(target[s1], ref_g[s2]);
                const double* tile = buf[0];
                if (!tile) continue;
                for (std::size_t i = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        S_tm(bf1 + i, bf2 + j) += tile[i * n2 + j];
                    }
                }
            }
        }
    }
    return S_tm;
}

Eigen::MatrixXd minao_density(
    const Molecule& mol, const BasisSet& basis, const Eigen::MatrixXd& S,
    int n_elec, const GuessECPContext& ecp) {
    if (n_elec < 0) throw std::invalid_argument("MINAO: electron count must be nonnegative");
    if (n_elec == 0) return Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
    ensure_libint_initialized();
    const std::string ref_name = "ano-rcc-mb";  // ANO-RCC minimal contraction
    const auto atoms = molecule_libint_atoms(mol);

    libint2::BasisSet ref;
    try {
        ref = libint2::BasisSet(ref_name, atoms);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("MINAO: could not build the ano-rcc-mb reference "
                        "basis for this molecule (element outside the "
                        "Z=1-96 reference set?). libint2 reported: ")
            + e.what()
            + ". Use SAD or SAP for this system instead.");
    }
    ref.set_pure(true);
    const std::size_t n_ref = ref.nbf();

    // Block-diagonal reference density (one atomic SCF per element, cached).
    Eigen::MatrixXd D_ref = Eigen::MatrixXd::Zero(n_ref, n_ref);
    const auto ref_ranges = libint_atom_ranges(ref, atoms);
    // ECP atoms solve the valence-only atomic Hamiltonian in the reference
    // basis. Core occupations are removed by angular channel, and the exact
    // operator/center contract is preserved (including mixed AE/ECP cells).
    const BasisSet reference_basis(mol, ref_name, true);
    std::map<int, AtomicSCF> atomic;
    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const int Z = mol.atoms()[a].Z;
        const auto& D_at = atomic_state_for_atom(mol, reference_basis, a, ecp, atomic).D;
        const auto [first, last] = ref_ranges[a];
        const auto n_at = last - first;
        if (n_at == 0) continue;
        if (n_at != static_cast<std::size_t>(D_at.rows())) {
            throw std::runtime_error(
                "MINAO: reference basis size mismatch for Z="
                + std::to_string(Z));
        }
        D_ref.block(first, first, n_at, n_at) = D_at;
    }

    // Project: P = S_tt^{-1} S_tm ; D = P D_ref P^T.
    const Eigen::MatrixXd S_tm = cross_basis_overlap(basis.libint(), ref);
    const Eigen::MatrixXd P = S.ldlt().solve(S_tm);  // (nbf x n_ref)
    Eigen::MatrixXd D = P * D_ref * P.transpose();

    // Guess hygiene: the projection is not exactly norm-conserving when the
    // target basis does not fully span the reference. Rescale so the
    // electron count is exact — a harmless scalar on a starting density.
    if (n_elec > 0) {
        const double n_proj = (D.array() * S.array()).sum();  // tr(D·S)
        if (n_proj > 1e-8) D *= static_cast<double>(n_elec) / n_proj;
    }
    return D;
}

}  // namespace

// Periodic MINAO density guess. This mirrors the molecular minao_density above
// (same ANO-RCC-MB reference, same block-diagonal atomic reference density,
// same project-then-rescale recipe) but replaces the two overlaps with their
// lattice-summed Γ folds: S → S_gamma (supplied by the driver, which already
// folded the target overlap to Γ) and the cross-basis overlap S_tm →
// cross_overlap_lattice_gamma (summed over direct-lattice images of the
// reference shells). The resulting D is the on-site (g=0) density block; the
// driver Bloch-sums it to recover D(k).
Eigen::MatrixXd compute_minao_density_periodic(
    const Molecule& mol, const BasisSet& basis,
    const PeriodicSystem& system, const Eigen::MatrixXd& S_gamma,
    int n_elec, const LatticeSumOptions& lattice_opts, const GuessECPContext& ecp) {
    validate_guess_ecp(InitialGuess::MINAO, ecp, &mol);
    if (n_elec < 0) throw std::invalid_argument("MINAO: electron count must be nonnegative");
    if (n_elec == 0) return Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
    ensure_libint_initialized();
    const std::string ref_name = "ano-rcc-mb";  // ANO-RCC minimal contraction
    const auto atoms = molecule_libint_atoms(mol);

    libint2::BasisSet ref;
    try {
        ref = libint2::BasisSet(ref_name, atoms);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("periodic MINAO: could not build the ano-rcc-mb "
                        "reference basis for this cell (element outside the "
                        "Z=1-96 reference set?). libint2 reported: ")
            + e.what()
            + ". Use SAD or SAP for this system instead.");
    }
    ref.set_pure(true);
    const std::size_t n_ref = ref.nbf();

    // Block-diagonal reference density (one atomic SCF per element, cached) —
    // identical to the molecular path; the reference density is on-site.
    Eigen::MatrixXd D_ref = Eigen::MatrixXd::Zero(n_ref, n_ref);
    const auto ref_ranges = libint_atom_ranges(ref, atoms);
    // ECP atoms solve the valence-only atomic Hamiltonian in the reference
    // basis. Core occupations are removed by angular channel, and the exact
    // operator/center contract is preserved (including mixed AE/ECP cells).
    const BasisSet reference_basis(mol, ref_name, true);
    std::map<int, AtomicSCF> atomic;
    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        const int Z = mol.atoms()[a].Z;
        const auto& D_at = atomic_state_for_atom(mol, reference_basis, a, ecp, atomic).D;
        const auto [first, last] = ref_ranges[a];
        const auto n_at = last - first;
        if (n_at == 0) continue;
        if (n_at != static_cast<std::size_t>(D_at.rows())) {
            throw std::runtime_error(
                "periodic MINAO: reference basis size mismatch for Z="
                + std::to_string(Z));
        }
        D_ref.block(first, first, n_at, n_at) = D_at;
    }

    // Project: P = S(Γ)^{-1} S_tm(Γ) ; D = P D_ref P^T, with the cross-basis
    // overlap lattice-summed to Γ.
    const Eigen::MatrixXd S_tm =
        cross_overlap_lattice_gamma(basis.libint(), ref, system, lattice_opts);
    const Eigen::MatrixXd P = S_gamma.ldlt().solve(S_tm);  // (nbf x n_ref)
    Eigen::MatrixXd D = P * D_ref * P.transpose();

    // Rescale so tr(D·S(Γ)) = n_elec exactly (harmless scalar on a guess).
    if (n_elec > 0) {
        const double n_proj = (D.array() * S_gamma.array()).sum();
        if (n_proj > 1e-8) D *= static_cast<double>(n_elec) / n_proj;
    }
    return D;
}

LatticeMatrixSet compute_huckel_fock_lattice(
    const Molecule& mol, const BasisSet& basis,
    const LatticeMatrixSet& S_set, const GuessECPContext& ecp) {
    validate_guess_ecp(InitialGuess::HUECKEL, ecp, &mol);
    const Eigen::VectorXd eps = huckel_ao_energies(mol, basis, ecp);
    if (static_cast<int>(eps.size()) != S_set.nbf) {
        throw std::runtime_error(
            "periodic HUECKEL: AO-energy vector size does not match "
            "the overlap lattice basis dimension");
    }

    constexpr double K = 1.75;  // Wolfsberg-Helmholz / Hoffmann factor
    LatticeMatrixSet F = S_set;
    for (std::size_t c = 0; c < F.blocks.size(); ++c) {
        const bool home_cell =
            (c < F.cells.size() && F.cells[c].r_cart.squaredNorm() < 1.0e-24);
        Eigen::MatrixXd& block = F.blocks[c];
        for (Eigen::Index mu = 0; mu < block.rows(); ++mu) {
            for (Eigen::Index nu = 0; nu < block.cols(); ++nu) {
                if (home_cell && mu == nu) {
                    block(mu, nu) = eps(mu);
                } else {
                    block(mu, nu) =
                        0.5 * K * S_set.blocks[c](mu, nu)
                        * (eps(mu) + eps(nu));
                }
            }
        }
    }
    return F;
}

// ============================================================================
//                              GuessEngine
// ============================================================================
//
// Single dispatch point for every SCF driver. Each ``kind`` is handled
// by a small free function in this file (HCORE / SAD / SAP ship in
// v0.9.x; dedicated periodic drivers may short-circuit for lattice variants
// such as SAP / HUECKEL / MINAO before this engine is reached.

namespace {

const char* guess_name(InitialGuess k) {
    switch (k) {
        case InitialGuess::AUTO:    return "AUTO";
        case InitialGuess::HCORE:   return "HCORE";
        case InitialGuess::SAD:     return "SAD";
        case InitialGuess::SAP:     return "SAP";
        case InitialGuess::PATOM:   return "PATOM";
        case InitialGuess::HUECKEL: return "HUECKEL";
        case InitialGuess::MINAO:   return "MINAO";
        case InitialGuess::READ:    return "READ";
        case InitialGuess::FRAGMO:  return "FRAGMO";
    }
    return "?";
}

[[noreturn]] void throw_not_implemented(InitialGuess kind, const char* roadmap) {
    throw std::runtime_error(
        std::string("GuessEngine: kind=") + guess_name(kind)
        + " not yet implemented — see " + roadmap);
}

// Transition-metal detection by Z. d-block: Sc–Zn, Y–Cd,
// La/Hf–Hg, Ac/Rf–Cn. f-block (lanthanides + actinides) also
// counted — they share SAD/PATOM's "shell structure matters" failure
// mode against SAP on the first iteration.
bool has_transition_metal(const Molecule& mol) {
    for (const auto& a : mol.atoms()) {
        const int Z = a.Z;
        if ((Z >= 21 && Z <= 30) ||   // Sc–Zn
            (Z >= 39 && Z <= 48) ||   // Y–Cd
            (Z == 57) || (Z >= 72 && Z <= 80) ||   // La + Hf–Hg
            (Z >= 58 && Z <= 71) ||   // Ce–Lu (lanthanides)
            (Z == 89) || (Z >= 104 && Z <= 112) ||  // Ac + Rf–Cn
            (Z >= 90 && Z <= 103)) {  // Th–Lr (actinides)
            return true;
        }
    }
    return false;
}

// Fill in derived hints from the Molecule. Caller-supplied hints
// take precedence — only fields the caller didn't set get filled.
SystemHints enrich_hints(const Molecule& mol, const SystemHints& in,
                          bool detected_open_shell) {
    SystemHints out = in;
    if (!out.has_transition_metal) {
        out.has_transition_metal = has_transition_metal(mol);
    }
    if (!out.is_open_shell) {
        out.is_open_shell = detected_open_shell;
    }
    return out;
}

}  // namespace


namespace {

template <typename Matrix>
Matrix normalize_density_impl(
    const Matrix& density, const Matrix& metric, double electrons) {
    if (density.rows() != density.cols() || metric.rows() != metric.cols()
        || density.rows() != metric.rows() || !density.allFinite()
        || !metric.allFinite() || !std::isfinite(electrons) || electrons < 0
        || (metric - metric.adjoint()).norm()
            > 1e-10 * std::max(1.0, metric.norm())) {
        throw std::invalid_argument(
            "initial guess: invalid density, overlap metric or population");
    }
    Matrix result = 0.5 * (density + density.adjoint()).eval();
    if (electrons == 0) return Matrix::Zero(result.rows(), result.cols());
    const double population = std::real((result * metric).trace());
    if (!result.allFinite() || !std::isfinite(population) || !(population > 0)) {
        throw std::invalid_argument(
            "initial guess: positive electron count needs a finite positive metric trace");
    }
    if (std::abs(population - electrons) > 1e-12) {
        result *= electrons / population;
    }
    if (!result.allFinite()) {
        throw std::invalid_argument("initial guess: density normalization overflow");
    }
    return result;
}

template <typename Matrix>
std::pair<Matrix, Matrix> normalize_spin_impl(
    const Matrix& alpha, const Matrix& beta, const Matrix& metric,
    int n_alpha, int n_beta) {
    if (n_alpha < 0 || n_beta < 0 || alpha.rows() != beta.rows()
        || alpha.cols() != beta.cols()) {
        throw std::invalid_argument("initial guess: invalid spin populations");
    }
    const int electrons = n_alpha + n_beta;
    Matrix total = normalize_density_impl(
        Matrix(alpha + beta), metric, double(electrons));
    if (electrons == 0) return {total, total};
    // Do not perturb a valid seed by roundoff. Exact equality of singlet
    // channels is also part of the spin-symmetry contract.
    if (std::abs(std::real((alpha * metric).trace()) - n_alpha) <= 1e-12
        && std::abs(std::real((beta * metric).trace()) - n_beta) <= 1e-12) {
        return {Matrix(0.5 * (alpha + alpha.adjoint())),
                Matrix(0.5 * (beta + beta.adjoint()))};
    }
    if (n_alpha == n_beta && (alpha.array() == beta.array()).all()) {
        Matrix half = normalize_density_impl(alpha, metric, n_alpha);
        return {half, half};
    }
    const double raw_count = std::real(((alpha + beta) * metric).trace());
    Matrix da = (0.5 * (alpha + alpha.adjoint())).eval() * (electrons / raw_count);
    Matrix db = total - da;
    double na = std::real((da * metric).trace());
    double nb = std::real((db * metric).trace());

    // Preserve the total density and existing Hund spin pattern whenever its
    // magnitude can be reduced by a convex mixture (#119 / BUG 88). If charge
    // or spin requires amplification, transfer a positive fraction of the
    // other spin density instead of amplifying signed occupations.
    const double spin = na - nb;
    const double target_spin = n_alpha - n_beta;
    if (std::abs(spin) > 1e-12 && std::abs(target_spin / spin) <= 1.0) {
        const Matrix difference = (target_spin / spin) * (da - db);
        da = 0.5 * (total + difference);
        db = total - da;
    } else if (n_alpha > na && nb > 0) {
        const Matrix transfer = ((n_alpha - na) / nb) * db;
        da += transfer;
        db -= transfer;
    } else if (n_alpha < na && na > 0) {
        const Matrix transfer = ((na - n_alpha) / na) * da;
        db += transfer;
        da -= transfer;
    }
    return {
        normalize_density_impl(da, metric, n_alpha),
        normalize_density_impl(db, metric, n_beta),
    };
}

}  // namespace

Eigen::MatrixXd normalize_guess_density(
    const Eigen::MatrixXd& density, const Eigen::MatrixXd& metric,
    double electrons) {
    return normalize_density_impl(density, metric, electrons);
}

Eigen::MatrixXcd normalize_guess_density(
    const Eigen::MatrixXcd& density, const Eigen::MatrixXcd& metric,
    double electrons) {
    return normalize_density_impl(density, metric, electrons);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> normalize_guess_spin_densities(
    const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta,
    const Eigen::MatrixXd& metric, int n_alpha, int n_beta) {
    return normalize_spin_impl(
        alpha, beta, metric, n_alpha, n_beta);
}

std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> normalize_guess_spin_densities(
    const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
    const Eigen::MatrixXcd& metric, int n_alpha, int n_beta) {
    return normalize_spin_impl(
        alpha, beta, metric, n_alpha, n_beta);
}

void validate_guess_atomic_spins(
    const Molecule& mol, const BasisSet& basis,
    const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
    const Eigen::MatrixXcd& metric, const std::vector<int>& atomic_spins) {
    if (atomic_spins.empty()) return;
    if (atomic_spins.size() != mol.atoms().size()) {
        throw std::invalid_argument("ATOMSPIN: one tag per atom is required");
    }
    const Eigen::VectorXd spin = ((alpha - beta) * metric).diagonal().real();
    const auto ranges = atom_basis_ranges(basis, mol);
    for (std::size_t atom = 0; atom < atomic_spins.size(); ++atom) {
        const auto [first, last] = ranges[atom];
        const double population = spin.segment(first, last - first).sum();
        const int tag = atomic_spins[atom];
        const bool compatible = tag == 0 ? std::abs(population) <= 1e-9
            : (tag == 1 || tag == -1) && tag * population > 1e-9;
        if (!compatible) {
            throw std::invalid_argument(
                "ATOMSPIN: requested populations are incompatible with the "
                "SAD spin pattern at atom " + std::to_string(atom)
                + "; choose compatible atomic_spins or multiplicity");
        }
    }
}

std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> normalize_atomic_guess(
    const Molecule& mol, const BasisSet& basis,
    const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
    const Eigen::MatrixXcd& metric, int n_alpha, int n_beta,
    const std::vector<int>& atomic_spins) {
    if (n_alpha < 0 || n_beta < 0 || atomic_spins.size() != mol.atoms().size()) {
        throw std::invalid_argument("ATOMSPIN: invalid populations or tag count");
    }
    const auto nbf = static_cast<Eigen::Index>(basis.nbasis());
    if (alpha.rows() != nbf || alpha.cols() != nbf
        || beta.rows() != nbf || beta.cols() != nbf
        || !alpha.allFinite() || !beta.allFinite()
        || (alpha - alpha.adjoint()).norm() > 1e-10
        || (beta - beta.adjoint()).norm() > 1e-10) {
        throw std::invalid_argument("ATOMSPIN: expected Hermitian atomic densities");
    }
    const auto ranges = atom_basis_ranges(basis, mol);
    Eigen::MatrixXcd off_alpha = alpha, off_beta = beta;
    for (const auto& range : ranges) {
        const auto [first, last] = range;
        off_alpha.block(first, first, last - first, last - first).setZero();
        off_beta.block(first, first, last - first, last - first).setZero();
    }
    if (off_alpha.norm() > 1e-10 || off_beta.norm() > 1e-10) {
        throw std::invalid_argument("ATOMSPIN: expected block-diagonal atomic densities");
    }
    const int electrons = n_alpha + n_beta;
    const Eigen::MatrixXcd total = normalize_guess_density(
        Eigen::MatrixXcd(alpha + beta), metric, electrons);
    if (electrons == 0) {
        validate_guess_atomic_spins(mol, basis, total, total, metric, atomic_spins);
        return {total, total};
    }
    const double raw_count = std::real(((alpha + beta) * metric).trace());
    Eigen::MatrixXcd spin = (alpha - beta) * (electrons / raw_count);
    const Eigen::VectorXd local_spin = (spin * metric).diagonal().real();
    double positive = 0.0, negative = 0.0;
    for (std::size_t a = 0; a < atomic_spins.size(); ++a) {
        const auto [first, last] = ranges[a];
        const double value = local_spin.segment(first, last - first).sum();
        if (atomic_spins[a] == 1) positive += value;
        else if (atomic_spins[a] == -1) negative -= value;
        else if (atomic_spins[a] != 0 || std::abs(value) > 1e-9) {
            throw std::invalid_argument("ATOMSPIN: invalid unpolarized atomic seed");
        }
    }
    // Preserve each sign and the total atomic density. Reduce only the
    // overrepresented sign group, via convex alpha/beta mixing on its
    // atomic blocks. This retains AFM order even when target net spin is zero.
    const double target = n_alpha - n_beta;
    double positive_scale = 1.0, negative_scale = 1.0;
    if (target < positive - negative - 1e-12) {
        positive_scale = positive > 0 ? (target + negative) / positive : -1.0;
    } else if (target > positive - negative + 1e-12) {
        negative_scale = negative > 0 ? (positive - target) / negative : -1.0;
    }
    if (positive_scale <= 0 || positive_scale > 1.0 + 1e-12
        || negative_scale <= 0 || negative_scale > 1.0 + 1e-12) {
        throw std::invalid_argument(
            "ATOMSPIN: requested spin exceeds the sign-preserving atomic "
            "density pattern; choose compatible atomic_spins or multiplicity");
    }
    for (std::size_t a = 0; a < atomic_spins.size(); ++a) {
        const auto [first, last] = ranges[a];
        spin.block(first, first, last - first, last - first) *=
            atomic_spins[a] > 0 ? positive_scale : negative_scale;
    }
    Eigen::MatrixXcd da = 0.5 * (total + spin);
    Eigen::MatrixXcd db = total - da;
    validate_guess_atomic_spins(mol, basis, da, db, metric, atomic_spins);
    return {da, db};
}

void validate_initial_guess(InitialGuess kind) {
    switch (kind) {
        case InitialGuess::AUTO:
        case InitialGuess::HCORE:
        case InitialGuess::SAD:
        case InitialGuess::SAP:
        case InitialGuess::PATOM:
        case InitialGuess::HUECKEL:
        case InitialGuess::MINAO:
        case InitialGuess::READ:
        case InitialGuess::FRAGMO:
            return;
    }
    throw std::invalid_argument("unknown initial_guess enum value");
}

void validate_atomic_spin_selection(
    const Molecule* mol, InitialGuess effective, const std::vector<int>& tags) {
    if (tags.empty()) return;
    if (!mol || tags.size() != mol->atoms().size()) {
        throw std::invalid_argument("GuessEngine: atomic_spins length must match atom count");
    }
    if (std::any_of(tags.begin(), tags.end(), [](int tag) { return tag < -1 || tag > 1; })) {
        throw std::invalid_argument("atomic_spins tags must be -1, 0 or 1");
    }
    if (effective != InitialGuess::SAD) {
        throw std::runtime_error(
            std::string("atomic_spins requires the SAD guess; resolved guess is ")
            + guess_name(effective));
    }
}

InitialGuess GuessEngine::resolve_auto(InitialGuess kind,
                                       const SystemHints& hints) {
    validate_initial_guess(kind);
    if (kind != InitialGuess::AUTO) return kind;
    // Policy table (docs/roadmap.md §G2e — partial; smearing /
    // Saunders–Hillier shift integration with AUTO lands later):
    //
    //   periodic, any system        → SAD   (NaCl/MgO bombing fix;
    //                                        explicit periodic SAP is
    //                                        available through route-local
    //                                        lattice-summed drivers)
    //   molecular, open-shell       → SAD   (SCF asymmetry develops
    //                                        from per-spin Fock build)
    //   molecular, transition metal → SAD   (closed-shell SAP can land
    //                                        the wrong d-shell occ; select
    //                                        PATOM explicitly when its
    //                                        in-field step is desired)
    //   molecular, closed-shell    → PATOM  (preserve the established
    //                                        molecular default basin)
    if (hints.is_periodic) return InitialGuess::SAD;
    if (hints.is_open_shell) return InitialGuess::SAD;
    if (hints.has_transition_metal) return InitialGuess::SAD;
    // Retain the established molecular closed-shell default basin while AUTO
    // becomes the public literal default. This is compatibility policy.
    return InitialGuess::PATOM;
}

InitialGuess GuessEngine::resolve_auto_for_molecule(
    const Molecule& mol, InitialGuess kind, bool is_periodic, bool is_open_shell,
    const std::optional<std::vector<InitialGuess>>& supported) {
    validate_initial_guess(kind);
    SystemHints hints;
    hints.is_periodic = is_periodic;
    hints.is_open_shell = is_open_shell;
    InitialGuess effective = resolve_auto(kind, enrich_hints(mol, hints, is_open_shell));
    if (kind == InitialGuess::AUTO && !is_periodic && is_open_shell
        && mol.multiplicity() != 1 && mol.atoms().size() == 1) {
        effective = InitialGuess::PATOM;
    }
    if (supported) {
        for (auto item : *supported) validate_initial_guess(item);
        auto accepts = [&](InitialGuess item) {
            return std::find(supported->begin(), supported->end(), item) != supported->end();
        };
        if (!accepts(effective)) {
            if (kind == InitialGuess::AUTO && accepts(InitialGuess::SAD)) {
                // PATOM's unrefined parent remains available without J/K.
                effective = InitialGuess::SAD;
            } else if (kind == InitialGuess::AUTO && accepts(InitialGuess::HCORE)) {
                effective = InitialGuess::HCORE;
            } else {
                throw GuessCapabilityError(
                    std::string("initial_guess=") + guess_name(kind) + " resolves to "
                    + guess_name(effective) + "; the route does not implement this construction");
            }
        }
    }
    return effective;
}

namespace {

GuessSelection select_prepared_guess(
    const Molecule* mol, InitialGuess kind, bool open_shell,
    bool supplied, const GuessSelection* prepared) {
    validate_initial_guess(kind);
    if (prepared) {
        validate_initial_guess(prepared->requested);
        validate_initial_guess(prepared->effective);
        validate_initial_guess(prepared->transport);
        if (prepared->requested != kind || prepared->effective == InitialGuess::AUTO
            || prepared->transport == InitialGuess::AUTO) {
            throw std::invalid_argument("initial guess: inconsistent prepared provenance");
        }
        if ((prepared->transport == InitialGuess::READ && !supplied)
            || (prepared->transport != InitialGuess::READ
                && prepared->transport != prepared->effective)
            || (kind != InitialGuess::AUTO && prepared->effective != kind
                && prepared->effective != InitialGuess::READ)) {
            throw std::invalid_argument("initial guess: provenance does not match the supplied construction");
        }
        return *prepared;
    }
    const InitialGuess effective = supplied ? InitialGuess::READ
        : mol ? GuessEngine::resolve_auto_for_molecule(*mol, kind, false, open_shell)
        : kind == InitialGuess::AUTO ? InitialGuess::HCORE : kind;
    if (!supplied && !mol && effective != InitialGuess::HCORE
        && effective != InitialGuess::READ && effective != InitialGuess::FRAGMO) {
        throw GuessCapabilityError(
            std::string("initial_guess=") + guess_name(kind)
            + " requires molecule= on this external-JK construction route");
    }
    return {kind, effective, supplied ? InitialGuess::READ : effective};
}

void validate_prepared_density(const Eigen::MatrixXd& density, const BasisSet& basis) {
    const auto nbf = static_cast<Eigen::Index>(basis.nbasis());
    if (density.rows() != nbf || density.cols() != nbf || !density.allFinite()
        || (density - density.transpose()).norm() > 1e-8) {
        throw std::invalid_argument(
            "initial guess: prepared density must be finite, Hermitian and match the basis");
    }
    if (density.rows() > 0) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(density, Eigen::EigenvaluesOnly);
        if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() < -1e-8) {
            throw std::invalid_argument("initial guess: density must be positive semidefinite");
        }
    }
}

Eigen::MatrixXd normalize_prepared_restart(
    const Eigen::MatrixXd& density, const Eigen::MatrixXd& overlap, int electrons) {
    // The shared normalizer preserves valid populations while removing
    // accepted antisymmetric roundoff, including on the no-rescaling path.
    return normalize_guess_density(density, overlap, electrons);
}

}  // namespace

PreparedClosedGuess prepare_closed_guess(
    const Molecule* mol, const BasisSet& basis, int n_occ, InitialGuess kind,
    const Eigen::MatrixXd& overlap, const Eigen::MatrixXd& hcore,
    const JKBuilder& jk, const Eigen::MatrixXd& initial_density,
    const Eigen::MatrixXd& read_density, double linear_dep_threshold,
    const GuessSelection* prepared, const GuessECPContext& ecp) {
    if (n_occ < 0) throw std::invalid_argument("initial guess: negative occupation");
    if (overlap.rows() != basis.nbasis() || overlap.cols() != basis.nbasis()
        || hcore.rows() != overlap.rows() || hcore.cols() != overlap.cols()
        || !overlap.allFinite() || !hcore.allFinite()) {
        throw std::invalid_argument("initial guess: invalid overlap or Hcore shape/data");
    }
    validate_initial_guess(kind);
    validate_guess_ecp(kind, ecp, mol);
    PreparedClosedGuess result;
    result.selection = select_prepared_guess(mol, kind, false, initial_density.size() != 0, prepared);
    validate_guess_ecp(result.selection.effective, ecp, mol);
    if (initial_density.size()) result.density = initial_density;
    else if (result.selection.effective == InitialGuess::READ
             || result.selection.effective == InitialGuess::FRAGMO) result.density = read_density;
    else if (result.selection.effective != InitialGuess::HCORE) {
        if (!mol) throw GuessCapabilityError("initial guess: molecule is required to construct a density");
        if (n_occ == 0) result.density = Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
        else result.density = GuessEngine::build_closed_shell(
            *mol, basis, n_occ, result.selection.effective, &overlap, &hcore,
            &jk, {}, linear_dep_threshold, ecp).D;
    }
    if (result.density.size() || result.selection.effective != InitialGuess::HCORE) {
        validate_prepared_density(result.density, basis);
        result.density = normalize_prepared_restart(result.density, overlap, 2 * n_occ);
    }
    return result;
}

PreparedOpenGuess prepare_open_guess(
    const Molecule* mol, const BasisSet& basis, int n_alpha, int n_beta,
    InitialGuess kind, const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& hcore, const JKBuilder& jk,
    const Eigen::MatrixXd& initial_alpha, const Eigen::MatrixXd& initial_beta,
    const Eigen::MatrixXd& read_alpha, const Eigen::MatrixXd& read_beta,
    const std::vector<int>& atomic_spins, double linear_dep_threshold,
    bool refine_sad, const GuessSelection* prepared, const GuessECPContext& ecp) {
    if (n_alpha < 0 || n_beta < 0 || ((initial_alpha.size() == 0) != (initial_beta.size() == 0))) {
        throw std::invalid_argument("initial guess: invalid spin counts or incomplete density pair");
    }
    if (overlap.rows() != basis.nbasis() || overlap.cols() != basis.nbasis()
        || hcore.rows() != overlap.rows() || hcore.cols() != overlap.cols()
        || !overlap.allFinite() || !hcore.allFinite()) {
        throw std::invalid_argument("initial guess: invalid overlap or Hcore shape/data");
    }
    validate_initial_guess(kind);
    validate_guess_ecp(kind, ecp, mol);
    PreparedOpenGuess result;
    result.selection = select_prepared_guess(mol, kind, true, initial_alpha.size() != 0, prepared);
    validate_guess_ecp(result.selection.effective, ecp, mol);
    validate_atomic_spin_selection(mol, result.selection.effective, atomic_spins);
    if (initial_alpha.size()) {
        result.alpha = initial_alpha;
        result.beta = initial_beta;
    } else if (result.selection.effective == InitialGuess::READ
               || result.selection.effective == InitialGuess::FRAGMO) {
        result.alpha = read_alpha;
        result.beta = read_beta;
    } else if (result.selection.effective != InitialGuess::HCORE) {
        if (!mol) throw GuessCapabilityError("initial guess: molecule is required to construct a density");
        const JKBuilder* guess_jk = refine_sad || result.selection.effective == InitialGuess::PATOM ? &jk : nullptr;
        auto built = GuessEngine::build_open_shell(
            *mol, basis, n_alpha, n_beta, result.selection.effective,
            &overlap, &hcore, guess_jk, SystemHints{false, false, true},
            linear_dep_threshold, atomic_spins.empty() ? nullptr : &atomic_spins, ecp);
        result.alpha = built.D_alpha;
        result.beta = built.D_beta;
        if (n_alpha + n_beta == 0) {
            result.alpha = Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
            result.beta = result.alpha;
        }
    }
    if (result.alpha.size() || result.selection.effective != InitialGuess::HCORE) {
        validate_prepared_density(result.alpha, basis);
        validate_prepared_density(result.beta, basis);
        result.alpha = (0.5 * (result.alpha + result.alpha.transpose())).eval();
        result.beta = (0.5 * (result.beta + result.beta.transpose())).eval();
        const double population_alpha = (result.alpha * overlap).trace();
        const double population_beta = (result.beta * overlap).trace();
        if (!std::isfinite(population_alpha) || !std::isfinite(population_beta)
            || std::abs(population_alpha - n_alpha) > 1e-12
            || std::abs(population_beta - n_beta) > 1e-12) {
            // A spin-schedule restart may populate an initially empty spin
            // channel. Transfer density between spins before normalization;
            // independent scaling cannot create the missing population.
            auto normalized = normalize_guess_spin_densities(
                result.alpha, result.beta, overlap, n_alpha, n_beta);
            result.alpha = std::move(normalized.first);
            result.beta = std::move(normalized.second);
        }
    }
    return result;
}

GuessClosedShellResult GuessEngine::build_closed_shell(
    const Molecule& mol, const BasisSet& basis,
    int n_occ,
    InitialGuess kind,
    const Eigen::MatrixXd* S,
    const Eigen::MatrixXd* Hcore,
    const JKBuilder* jk,
    const SystemHints& hints,
    double linear_dep_threshold, const GuessECPContext& ecp) {
    GuessClosedShellResult result;
    const InitialGuess resolved = resolve_auto_for_molecule(
        mol, kind, hints.is_periodic, false,
        kind == InitialGuess::AUTO && (!S || !Hcore || !jk)
            ? std::optional<std::vector<InitialGuess>>({
                InitialGuess::HCORE, InitialGuess::SAD, InitialGuess::SAP,
                InitialGuess::HUECKEL, InitialGuess::MINAO})
            : std::nullopt);
    validate_guess_ecp(resolved, ecp, &mol);
    result.resolved_kind = resolved;

    switch (resolved) {
        case InitialGuess::HCORE:
            // Empty D + empty F_guess means "driver diagonalises Hcore
            // itself" — preserves the existing per-driver fallback
            // path. SAP will populate F_guess; we keep HCORE empty
            // so callers don't need to know about V_ne lattice-sum
            // separation on the periodic side.
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> HCORE"
                : "HCORE";
            return result;
        case InitialGuess::SAD:
            result.D = normalize_guess_density(
                sad_density(mol, basis, ecp), S ? *S : compute_overlap(basis),
                2.0 * n_occ);
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> SAD (superposition of atomic densities)"
                : "SAD (superposition of atomic densities)";
            return result;
        case InitialGuess::SAP: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: periodic SAP requires the route's lattice "
                    "context to build F_SAP(k) = T(k) + V_SAP(k). This "
                    "context-free entry point cannot construct it; use a "
                    "periodic SCF driver that supplies the PeriodicSystem and "
                    "lattice options.");
            }
            if (S == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: SAP requires S (overlap matrix) for "
                    "canonical orthogonalisation; caller passed nullptr");
            }
            result.D = sap_density(mol, basis, n_occ, *S,
                                   linear_dep_threshold,
                                   "sap_helfem_large", ecp);
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> SAP (Lehtola/Visscher/Engel 2020, sap_helfem_large)"
                : "SAP (Lehtola/Visscher/Engel 2020, sap_helfem_large)";
            if (ecp.active()) result.provenance =
                std::string(kind == InitialGuess::AUTO ? "AUTO -> " : "")
                + "SAP (numerical LDA on valence HF, actual ECP operator)";
            return result;
        }
        case InitialGuess::PATOM: {
            if (S == nullptr || Hcore == nullptr || jk == nullptr) {
                if (hints.is_periodic) {
                    throw std::runtime_error(
                        "GuessEngine: periodic PATOM requires S, Hcore and a "
                        "periodic JKBuilder for the in-field re-polarisation "
                        "step; this periodic driver does not yet build them. "
                        "Use SAD or HCORE for that path.");
                }
                throw std::runtime_error(
                    "GuessEngine: PATOM requires S, Hcore and a JKBuilder for "
                    "the in-field re-polarisation step; caller passed nullptr");
            }
            result.D = patom_density_closed(mol, basis, n_occ, *S, *Hcore, *jk,
                                            linear_dep_threshold, /*n_iter=*/1, ecp);
            if (hints.is_periodic) {
                result.provenance =
                    "PATOM (periodic Gamma, SAD + in-field re-polarisation)";
            } else {
                result.provenance = (kind == InitialGuess::AUTO)
                    ? "AUTO -> PATOM (SAD + in-field re-polarisation)"
                    : "PATOM (SAD + in-field re-polarisation)";
            }
            return result;
        }
        case InitialGuess::HUECKEL: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: periodic HUECKEL is built as a Fock-mode "
                    "GWH lattice guess and is wired into the closed-shell "
                    "periodic RHF/RKS drivers (Γ and multi-k), which "
                    "short-circuit before this engine. Reaching here means "
                    "HUECKEL was requested on a periodic path that does not "
                    "yet build it; use SAD or HCORE for that path.");
            }
            if (S == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: HUECKEL requires S (overlap matrix); caller "
                    "passed nullptr");
            }
            Eigen::VectorXd eps;
            const Eigen::MatrixXd C =
                huckel_mo_coefficients(mol, basis, *S, &eps, ecp);
            if (C.cols() < n_occ) {
                throw std::runtime_error(
                    "HUECKEL: fewer occupied atomic orbitals than molecular "
                    "occupied orbitals (a highly charged anion in a small "
                    "basis?) — use SAD or SAP for this system");
            }
            const auto C_occ = C.leftCols(n_occ);
            result.D = 2.0 * C_occ * C_occ.transpose();
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> HUECKEL (parameter-free GWH; Hoffmann 1963 / Lehtola 2019)"
                : "HUECKEL (parameter-free GWH; Hoffmann 1963 / Lehtola 2019)";
            return result;
        }
        case InitialGuess::MINAO: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: periodic MINAO is built as a density-mode "
                    "guess (compute_minao_density_periodic — the ANO-RCC "
                    "reference density projected with lattice-summed Γ "
                    "overlaps) and is wired into the closed-shell periodic "
                    "RHF/RKS drivers (Γ and multi-k), which short-circuit "
                    "before this engine. Reaching here means MINAO was "
                    "requested on a periodic path that does not yet build it; "
                    "use SAD or HCORE for that path.");
            }
            if (S == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: MINAO requires S (overlap matrix); caller "
                    "passed nullptr");
            }
            result.D = minao_density(mol, basis, *S, /*n_elec=*/2 * n_occ, ecp);
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> MINAO (ANO-RCC-MB reference projection)"
                : "MINAO (ANO-RCC-MB reference projection)";
            return result;
        }
        case InitialGuess::READ:
            throw_not_implemented(resolved, "docs/roadmap.md §G2 (v0.9.x)");
        case InitialGuess::FRAGMO:
            // FRAGMO assembles the guess density in the Python runner
            // (guess_fragmo.py) and injects it via opts.read_density, exactly
            // like READ — the C++ driver short-circuits before this engine
            // call, so reaching here means a driver fed FRAGMO straight to the
            // engine without populating read_density.
            throw std::runtime_error(
                "GuessEngine: FRAGMO is assembled in the Python runner "
                "(run_rhf/run_rks fragments=...) and injected via "
                "opts.read_density; the C++ engine is not reached for it. "
                "Calling the engine with FRAGMO directly is unsupported.");
        case InitialGuess::AUTO:
            // Unreachable: resolve_auto handled it above.
            throw std::logic_error("GuessEngine: AUTO leaked past resolve_auto");
    }
    throw std::logic_error("GuessEngine: unhandled InitialGuess kind");
}

GuessOpenShellResult GuessEngine::build_open_shell(
    const Molecule& mol, const BasisSet& basis,
    int n_alpha, int n_beta,
    InitialGuess kind,
    const Eigen::MatrixXd* S,
    const Eigen::MatrixXd* Hcore,
    const JKBuilder* jk,
    const SystemHints& hints,
    double linear_dep_threshold,
    const std::vector<int>* atomic_spins, const GuessECPContext& ecp) {
    GuessOpenShellResult result;
    if (n_alpha < 0 || n_beta < 0) {
        throw std::invalid_argument("GuessEngine: spin populations must be nonnegative");
    }
    // Keep the raw engine entry point on the same molecule-aware AUTO policy
    // as the production UHF/UKS drivers. In particular, an isolated
    // spin-polarised atom needs PATOM's in-field re-polarisation rather than
    // the generic open-shell SAD choice.
    InitialGuess resolved = resolve_auto_for_molecule(
        mol, kind, hints.is_periodic, true,
        kind == InitialGuess::AUTO && (!S || !Hcore || !jk)
            ? std::optional<std::vector<InitialGuess>>({
                InitialGuess::HCORE, InitialGuess::SAD, InitialGuess::SAP,
                InitialGuess::HUECKEL, InitialGuess::MINAO})
            : std::nullopt);
    // Spin seeds modify SAD only; AUTO is resolved before validating that
    // contract. An explicit PATOM request cannot lose its in-field step.
    const bool seed_present =
        atomic_spins != nullptr && !atomic_spins->empty();
    validate_guess_ecp(resolved, ecp, &mol);
    result.resolved_kind = resolved;
    if (seed_present) validate_atomic_spin_selection(&mol, resolved, *atomic_spins);
    const int n_elec = n_alpha + n_beta;

    switch (resolved) {
        case InitialGuess::HCORE:
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> HCORE"
                : "HCORE";
            return result;
        case InitialGuess::SAD: {
            if (n_elec <= 0) {
                result.D_alpha = Eigen::MatrixXd::Zero(basis.nbasis(), basis.nbasis());
                result.D_beta = result.D_alpha;
                if (seed_present) {
                    validate_guess_atomic_spins(
                        mol, basis, result.D_alpha.cast<std::complex<double>>(),
                        result.D_beta.cast<std::complex<double>>(),
                        (S ? *S : compute_overlap(basis)).cast<std::complex<double>>(),
                        *atomic_spins);
                }
                result.provenance = "SAD (no electrons; zero density)";
                return result;
            }
            // ATOMSPIN: per-atom spin-seeded broken-symmetry start. When the
            // caller tags atoms (+1 majority alpha, -1 majority beta, 0
            // unpolarised), assemble per-atom Hund-split spin densities
            // block-diagonally instead of the spin-symmetric split below, so
            // an AFM / ferrimagnetic sublattice pattern is seeded. Density-
            // mode for both UHF and UKS; the SCF refines the pattern and
            // enforces the global n_alpha / n_beta.
            if (atomic_spins != nullptr && !atomic_spins->empty()) {
                if (atomic_spins->size() != mol.atoms().size()) {
                    throw std::runtime_error(
                        "GuessEngine: atomic_spins length ("
                        + std::to_string(atomic_spins->size())
                        + ") must match the atom count ("
                        + std::to_string(mol.atoms().size()) + ")");
                }
                const std::size_t nbf = basis.nbasis();
                Eigen::MatrixXd Da = Eigen::MatrixXd::Zero(nbf, nbf);
                Eigen::MatrixXd Db = Eigen::MatrixXd::Zero(nbf, nbf);
                const auto ranges = atom_basis_ranges(basis, mol);
                std::map<int, AtomicSCF> cache;
                for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
                    const int Z = mol.atoms()[a].Z;
                    const AtomicSCF& at = atomic_state_for_atom(mol, basis, a, ecp, cache);
                    const int sgn = (*atomic_spins)[a];
                    Eigen::VectorXd oa, ob;
                    if (sgn > 0)      { oa = at.occ_alpha; ob = at.occ_beta; }
                    else if (sgn < 0) { oa = at.occ_beta;  ob = at.occ_alpha; }
                    else              { oa = 0.5 * at.occ;  ob = 0.5 * at.occ; }
                    const Eigen::MatrixXd da =
                        at.C * oa.asDiagonal() * at.C.transpose();
                    const Eigen::MatrixXd db =
                        at.C * ob.asDiagonal() * at.C.transpose();
                    const auto [first, last] = ranges[a];
                    const auto n_at = last - first;
                    if (n_at == 0) continue;
                    if (n_at != static_cast<std::size_t>(da.rows())) {
                        throw std::runtime_error(
                            "ATOMSPIN: atomic basis size mismatch for Z="
                            + std::to_string(Z));
                    }
                    Da.block(first, first, n_at, n_at) = da;
                    Db.block(first, first, n_at, n_at) = db;
                }
                auto normalized = normalize_atomic_guess(
                    mol, basis, Da.cast<std::complex<double>>(),
                    Db.cast<std::complex<double>>(),
                    (S ? *S : compute_overlap(basis)).cast<std::complex<double>>(),
                    n_alpha, n_beta, *atomic_spins);
                result.D_alpha = normalized.first.real();
                result.D_beta = normalized.second.real();
                result.provenance =
                    (kind == InitialGuess::AUTO)
                        ? "AUTO -> SAD (per-atom spin-seeded; ATOMSPIN)"
                        : "SAD (per-atom spin-seeded; ATOMSPIN)";
                return result;
            }
            // Hund-split atomic densities (default for open-shell).
            // Each atom contributes its natural Hund's-rule spin-polarised
            // density (occ_alpha / occ_beta from spin_resolve_occupations),
            // assembled block-diagonally.  This replaces the old proportional
            // spin split of a closed-shell SAD density, which gave transition
            // metals a spherically-symmetric spin density and caused the SCF
            // to land on excited solutions (BUG 88 — FeCl₃ UHF/cc-pVDZ).
            //
            // Spin-balanced systems (n_α == n_β) take spin-averaged atomic
            // occupations instead: a Hund-split singlet start leaves
            // convergence-limited α/β polarisation in the final densities
            // and defeats the shared-exchange spin-degenerate gate. A
            // deliberate broken-symmetry singlet start remains available
            // via ATOMSPIN (handled above). Broken-symmetry singlets are
            // still reachable under defaults: the stability escape, not
            // the guess, descends from the symmetric saddle (verified
            // H2/O3/C2H4/pC6H4; handovers/HANDOVER_GUESS_SPIN_SYMMETRY.md).
            //
            // The UKS path (jk != nullptr) first builds per-spin densities
            // this way, then Fock-diagonalises for further refinement.
            const bool sad_spin_balanced = (n_alpha == n_beta);
            const std::size_t nbf = basis.nbasis();
            Eigen::MatrixXd Da_hund = Eigen::MatrixXd::Zero(nbf, nbf);
            Eigen::MatrixXd Db_hund = Eigen::MatrixXd::Zero(nbf, nbf);
            const auto ranges = atom_basis_ranges(basis, mol);
            std::map<int, AtomicSCF> cache;
            for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
                const int Z = mol.atoms()[a].Z;
                const AtomicSCF& at = atomic_state_for_atom(mol, basis, a, ecp, cache);
                // Natural Hund majority-alpha configuration (spin-averaged
                // when the molecular spin counts balance).
                const Eigen::VectorXd oa = sad_spin_balanced
                    ? Eigen::VectorXd(0.5 * at.occ) : at.occ_alpha;
                const Eigen::VectorXd ob = sad_spin_balanced
                    ? Eigen::VectorXd(0.5 * at.occ) : at.occ_beta;
                const Eigen::MatrixXd da =
                    at.C * oa.asDiagonal() * at.C.transpose();
                const Eigen::MatrixXd db =
                    at.C * ob.asDiagonal() * at.C.transpose();
                const auto [first, last] = ranges[a];
                const auto n_at = last - first;
                if (n_at == 0) continue;
                if (n_at != static_cast<std::size_t>(da.rows())) {
                    throw std::runtime_error(
                        "SAD: atomic basis size mismatch for Z="
                        + std::to_string(Z));
                }
                Da_hund.block(first, first, n_at, n_at) = da;
                Db_hund.block(first, first, n_at, n_at) = db;
            }
            auto normalized = normalize_guess_spin_densities(
                Da_hund, Db_hund, S ? *S : compute_overlap(basis),
                n_alpha, n_beta);
            Da_hund = std::move(normalized.first);
            Db_hund = std::move(normalized.second);
            if (jk != nullptr && S != nullptr && Hcore != nullptr) {
                // UKS-style packaging: build a per-spin F_SAD from the
                // Hund-split densities and refine via Fock-diagonalise.
                const Eigen::MatrixXd D_tot = Da_hund + Db_hund;
                const Eigen::MatrixXd F_guess =
                    *Hcore + jk->build_J(D_tot) - 0.5 * jk->build_K(D_tot);
                const auto orth =
                    canonical_orthogonalizer(*S, linear_dep_threshold);
                if (orth.n_kept >= std::max(n_alpha, n_beta)) {
                    const Eigen::MatrixXd& X = orth.X;
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
                        X.transpose() * F_guess * X);
                    const Eigen::MatrixXd C = X * es.eigenvectors();
                    const auto Cao = C.leftCols(n_alpha);
                    const auto Cbo = C.leftCols(n_beta);
                    result.D_alpha = Cao * Cao.transpose();
                    result.D_beta  = Cbo * Cbo.transpose();
                    result.provenance =
                        (kind == InitialGuess::AUTO)
                        ? "AUTO -> SAD (Hund-split + Fock-diagonalise)"
                        : "SAD (Hund-split + Fock-diagonalise)";
                    return result;
                }
            }
            result.D_alpha = std::move(Da_hund);
            result.D_beta  = std::move(Db_hund);
            result.provenance =
                (kind == InitialGuess::AUTO)
                ? "AUTO -> SAD (Hund-split atomic densities)"
                : "SAD (Hund-split atomic densities)";
            return result;
        }
        case InitialGuess::SAP: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: periodic SAP requires the route's lattice "
                    "context to build F_SAP(k) = T(k) + V_SAP(k). This "
                    "context-free entry point cannot construct it; use a "
                    "periodic SCF driver that supplies the PeriodicSystem and "
                    "lattice options.");
            }
            if (S == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: SAP requires S for canonical orth");
            }
            if (n_elec <= 0) {
                result.provenance = "SAP (no electrons; empty density)";
                return result;
            }
            // The SAP potential is spin-independent: diagonalise it once and
            // occupy the common orbital set through n_alpha and n_beta. A
            // proportional split of a doubly occupied max(n_alpha, n_beta)
            // density would contain 2*max(n_alpha, n_beta) electrons instead
            // of n_alpha+n_beta for every genuine open shell (issue 666).
            const Eigen::MatrixXd C = sap_orbitals(
                mol, basis, std::max(n_alpha, n_beta), *S,
                linear_dep_threshold, "sap_helfem_large", ecp);
            const auto C_alpha_occ = C.leftCols(n_alpha);
            const auto C_beta_occ = C.leftCols(n_beta);
            result.D_alpha = C_alpha_occ * C_alpha_occ.transpose();
            result.D_beta = C_beta_occ * C_beta_occ.transpose();
            result.provenance =
                (kind == InitialGuess::AUTO)
                ? "AUTO -> SAP (Lehtola/Visscher/Engel 2020, spin-resolved occupations)"
                : "SAP (Lehtola/Visscher/Engel 2020, spin-resolved occupations)";
            if (ecp.active()) result.provenance =
                std::string(kind == InitialGuess::AUTO ? "AUTO -> " : "")
                + "SAP (numerical LDA on valence HF, actual ECP operator, spin-resolved occupations)";
            return result;
        }
        case InitialGuess::PATOM: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: PATOM not yet implemented for periodic "
                    "systems; use SAD or HCORE instead");
            }
            if (S == nullptr || Hcore == nullptr || jk == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: PATOM requires S, Hcore and a JKBuilder for "
                    "the in-field re-polarisation step; caller passed nullptr "
                    "(the UHF wrapper forwards its JKBuilder only for PATOM)");
            }
            if (n_elec <= 0) {
                result.provenance = "PATOM (no electrons; empty density)";
                return result;
            }
            auto [Da, Db] = patom_density_open(
                mol, basis, n_alpha, n_beta, *S, *Hcore, *jk,
                linear_dep_threshold, /*n_iter=*/1, ecp);
            result.D_alpha = std::move(Da);
            result.D_beta = std::move(Db);
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> PATOM (SAD + per-spin in-field re-polarisation)"
                : "PATOM (SAD + per-spin in-field re-polarisation)";
            return result;
        }
        case InitialGuess::HUECKEL: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: periodic HUECKEL is wired for the "
                    "closed-shell RHF/RKS periodic drivers (Γ and multi-k); "
                    "the open-shell periodic path does not yet build the "
                    "per-spin GWH Fock. Use SAD or HCORE for periodic "
                    "open-shell.");
            }
            if (S == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: HUECKEL requires S (overlap matrix); caller "
                    "passed nullptr");
            }
            if (n_elec <= 0) {
                result.provenance = "HUECKEL (no electrons; empty density)";
                return result;
            }
            Eigen::VectorXd eps;
            const Eigen::MatrixXd C =
                huckel_mo_coefficients(mol, basis, *S, &eps, ecp);
            if (C.cols() < std::max(n_alpha, n_beta)) {
                throw std::runtime_error(
                    "HUECKEL: fewer occupied atomic orbitals than molecular "
                    "occupied orbitals (a highly charged anion in a small "
                    "basis?) — use SAD or SAP for this system");
            }
            const auto Ca = C.leftCols(n_alpha);
            const auto Cb = C.leftCols(n_beta);
            result.D_alpha = Ca * Ca.transpose();
            result.D_beta = Cb * Cb.transpose();
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> HUECKEL (parameter-free GWH; Hoffmann 1963 / Lehtola 2019)"
                : "HUECKEL (parameter-free GWH; Hoffmann 1963 / Lehtola 2019)";
            return result;
        }
        case InitialGuess::MINAO: {
            if (hints.is_periodic) {
                throw std::runtime_error(
                    "GuessEngine: MINAO not yet implemented for periodic "
                    "systems; use SAD or HCORE instead");
            }
            if (S == nullptr) {
                throw std::runtime_error(
                    "GuessEngine: MINAO requires S (overlap matrix); caller "
                    "passed nullptr");
            }
            if (n_elec <= 0) {
                result.provenance = "MINAO (no electrons; empty density)";
                return result;
            }
            const Eigen::MatrixXd D_tot = minao_density(mol, basis, *S, n_elec, ecp);
            const double fa = static_cast<double>(n_alpha) / n_elec;
            const double fb = static_cast<double>(n_beta) / n_elec;
            result.D_alpha = fa * D_tot;
            result.D_beta = fb * D_tot;
            result.provenance = (kind == InitialGuess::AUTO)
                ? "AUTO -> MINAO (ANO-RCC-MB reference projection, proportional spin split)"
                : "MINAO (ANO-RCC-MB reference projection, proportional spin split)";
            return result;
        }
        case InitialGuess::READ:
            throw_not_implemented(resolved, "docs/roadmap.md §G2 (v0.9.x)");
        case InitialGuess::FRAGMO:
            // Assembled in the Python runner (guess_fragmo.py) and injected via
            // opts.read_density_{alpha,beta}; the C++ driver short-circuits
            // before this engine call (see build_closed_shell's FRAGMO case).
            throw std::runtime_error(
                "GuessEngine: FRAGMO is assembled in the Python runner "
                "(run_uhf/run_uks fragments=...) and injected via "
                "opts.read_density_{alpha,beta}; the C++ engine is not reached "
                "for it. Calling the engine with FRAGMO directly is unsupported.");
        case InitialGuess::AUTO:
            throw std::logic_error("GuessEngine: AUTO leaked past resolve_auto");
    }
    throw std::logic_error("GuessEngine: unhandled InitialGuess kind");
}

}  // namespace vibeqc
