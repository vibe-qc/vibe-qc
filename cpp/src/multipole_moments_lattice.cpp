#include "vibeqc/multipole_moments_lattice.hpp"

#include "vibeqc/init.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/atom.h>
#include <libint2/engine.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibeqc {

namespace {

// Clone a shell vector and translate every origin by dr. Same helper as
// in lattice_integrals.cpp; duplicated here to avoid a public header
// dependency (the helper is private to lattice_integrals.cpp).
std::vector<libint2::Shell> shift_shells_for_multipole(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr) {
    std::vector<libint2::Shell> out(shells.begin(), shells.end());
    for (auto& s : out) {
        s.O[0] += dr[0];
        s.O[1] += dr[1];
        s.O[2] += dr[2];
    }
    return out;
}

libint2::Operator multipole_operator_for_L_max(int L_max, bool& spherical) {
    switch (L_max) {
        case 1: spherical = false; return libint2::Operator::emultipole1;
        case 2: spherical = false; return libint2::Operator::emultipole2;
        case 3: spherical = false; return libint2::Operator::emultipole3;
        case 4: spherical = true;  return libint2::Operator::sphemultipole;
        default:
            throw std::invalid_argument(
                "compute_multipole_moments_lattice: L_max must be 1, 2, 3, or 4 "
                "(libint emultipole{1,2,3} for Cartesian, sphemultipole for L=4 "
                "spherical harmonic moments per Saunders 1992, Sec. 3).");
    }
}

int multipole_n_components_for_L_max(int L_max, bool spherical) {
    if (spherical) {
        return (L_max + 1) * (L_max + 1);  // (L+1)^2 spherical harmonic components
    }
    return cartesian_multipole_n_components(L_max);
}

}  // namespace

LatticeMultipoleSet compute_multipole_moments_lattice(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    int L_max,
    const std::array<double, 3>& origin) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    bool is_spherical = false;
    const libint2::Operator op = multipole_operator_for_L_max(L_max, is_spherical);
    const int n_components = multipole_n_components_for_L_max(L_max, is_spherical);

    libint2::Engine prototype(op, shells_ref.max_nprim(), shells_ref.max_l(), 0);
    prototype.set_params(origin);
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();

    LatticeMultipoleSet set;
    set.nbf = nbf;
    set.L_max = L_max;
    set.spherical = is_spherical;
    set.origin = origin;
    set.cells = direct_lattice_cells(system, opts.cutoff_bohr);

    const int n_cells = static_cast<int>(set.cells.size());
    set.blocks.assign(n_cells, {});
    for (int c = 0; c < n_cells; ++c) {
        set.blocks[c].assign(
            n_components, Eigen::MatrixXd::Zero(nbf, nbf));
    }

    // Parallelise over lattice cells (same pattern as the 1e
    // lattice integrals): each thread writes to disjoint set.blocks[c],
    // no locking required.
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();

        const Eigen::Vector3d& g = set.cells[c].r_cart;
        const auto shells_g = shift_shells_for_multipole(shells_ref, g);

        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();

                engine.compute(shells_ref[s1], shells_g[s2]);

                // Scatter all components for this shell pair into the
                // per-component blocks.
                for (int comp = 0; comp < n_components; ++comp) {
                    const double* tile = buf[comp];
                    if (!tile) continue;
                    auto& block = set.blocks[c][comp];
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            block(bf1 + i, bf2 + j) = tile[i * n2 + j];
                        }
                    }
                }
            }
        }
    }

    return set;
}

// ---------------------------------------------------------------------------
// Disabled native EXT EL-SPHEROPOLE gradient (emultipole2 deriv_order=1)
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_ext_el_spheropole_gradient_lattice(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const LatticeMatrixSet& P_real,
    int n_electrons) {
    ensure_libint_initialized();
    (void)basis;
    (void)system;
    (void)opts;
    (void)P_real;
    (void)n_electrons;

    throw std::runtime_error(
        "compute_ext_el_spheropole_gradient_lattice: native libint "
        "emultipole2 derivative path is disabled because the current "
        "vendored libint build can segfault in that kernel. Use "
        "vibeqc.bipole_gradient._spheropole_ewald_gradient, which "
        "central-differences the exact spheropole energy at fixed density.");
}

}  // namespace vibeqc
