// Phase 14a/14b — libecpint wrapper.
//
// 14a: ``libecpint_version()`` smoke test.
// 14b: ``compute_ecp_matrix(basis, ecp_centers, library_name, share_dir)``
//      computes the AO-basis ECP matrix elements
//        V_ECP_{μν} = ⟨χ_μ | V_ECP | χ_ν⟩
//      via libecpint's built-in XML library, then transforms each
//      shell-pair sub-block from Cartesian to spherical (real-solid-
//      harmonic) basis using libint's ``solidharmonics::tform_*``
//      primitives. vibe-qc's BasisSet forces ``set_pure(true)``, so
//      the spherical matrix is what every downstream operator (Hcore,
//      gradient, ...) expects.
//
// Reference: libecpint's set_ecp_basis_from_library API (api.hpp,
// HAS_PUGIXML branch). The XML files ship alongside the vendored
// libecpint install under share/libecpint/xml/.

#include "vibeqc/ecp.hpp"

#include <libecpint/api.hpp>
#include <libecpint/config.hpp>
#include <libecpint/ecp.hpp>
#include <libecpint/gaussquad.hpp>
#include <libint2.hpp>
#include <libint2/solidharmonics.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/lattice_integrals.hpp"

namespace vibeqc {

namespace {

// Path to libecpint's XML share dir baked at build time via
// cpp/CMakeLists.txt (vendored ``third_party/libecpint/install/share/
// libecpint`` for development builds, or the discovered system /
// Homebrew path). Falls back to the empty string for release-mode
// wheel builds where the bake-time path doesn't exist on the user's
// machine — those installs resolve at runtime via the
// ``VIBEQC_ECP_SHARE_DIR`` env var, which the Python ``__init__.py``
// sets to the wheel-bundled ``python/vibeqc/ecp_library/`` location.
#ifndef VIBEQC_LIBECPINT_SHARE_DIR
#define VIBEQC_LIBECPINT_SHARE_DIR ""
#endif

constexpr const char* kVendoredShareDir = VIBEQC_LIBECPINT_SHARE_DIR;

// Resolve a usable XML share_dir at call time. Order:
//   1. Caller-supplied non-empty ``share_dir``.
//   2. ``$VIBEQC_ECP_SHARE_DIR`` (set by Python __init__ to the
//      bundled path).
//   3. The bake-time ``VIBEQC_LIBECPINT_SHARE_DIR``.
// Returns the empty string if none resolves — caller's responsibility
// to throw a directive error.
std::string resolve_ecp_share_dir(const std::string& share_dir) {
    if (!share_dir.empty()) return share_dir;
    if (const char* env = std::getenv("VIBEQC_ECP_SHARE_DIR")) {
        if (env[0] != '\0') return std::string(env);
    }
    return std::string(kVendoredShareDir);
}

std::vector<int> validated_xml_ecp_atom_cores(
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const std::map<int, int>& core_map,
    const char* route) {
    const auto& atoms = mol.atoms();
    std::vector<int> atom_cores(atoms.size(), 0);
    std::vector<bool> atom_matched(atoms.size(), false);
    for (const auto& center : ecp_centers) {
        std::size_t match_index = atoms.size();
        int match_count = 0;
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            const double dx = atoms[i].xyz[0] - center.xyz[0];
            const double dy = atoms[i].xyz[1] - center.xyz[1];
            const double dz = atoms[i].xyz[2] - center.xyz[2];
            if (atoms[i].Z == center.Z
                && dx * dx + dy * dy + dz * dz < 1e-12) {
                match_index = i;
                ++match_count;
            }
        }
        if (match_count != 1) {
            throw std::invalid_argument(
                std::string(route) + ": each ECP center must match exactly "
                "one molecule atom by atomic number and position");
        }
        if (atom_matched[match_index]) {
            throw std::invalid_argument(
                std::string(route) + ": duplicate ECP centers match the "
                "same molecule atom");
        }
        const auto core = core_map.find(center.Z);
        if (core == core_map.end()) {
            throw std::invalid_argument(
                std::string(route) + ": the selected ECP library has no "
                "core-electron entry for atomic number "
                + std::to_string(center.Z));
        }
        if (core->second < 0 || core->second > center.Z) {
            throw std::invalid_argument(
                std::string(route) + ": invalid ECP core-electron count for "
                "atomic number " + std::to_string(center.Z));
        }
        atom_matched[match_index] = true;
        atom_cores[match_index] = core->second;
    }
    return atom_cores;
}

void validate_primitive_ecp_centers(
    const Molecule& mol,
    const std::vector<std::array<double, 3>>& ecp_centers,
    const char* route) {
    const auto& atoms = mol.atoms();
    std::vector<bool> atom_matched(atoms.size(), false);
    for (const auto& center : ecp_centers) {
        std::size_t match_index = atoms.size();
        int match_count = 0;
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            const double dx = atoms[i].xyz[0] - center[0];
            const double dy = atoms[i].xyz[1] - center[1];
            const double dz = atoms[i].xyz[2] - center[2];
            if (dx * dx + dy * dy + dz * dz < 1e-12) {
                match_index = i;
                ++match_count;
            }
        }
        if (match_count != 1) {
            throw std::invalid_argument(
                std::string(route) + ": each primitive ECP center must match "
                "exactly one molecule atom by position");
        }
        if (atom_matched[match_index]) {
            throw std::invalid_argument(
                std::string(route) + ": duplicate primitive ECP centers "
                "match the same molecule atom");
        }
        atom_matched[match_index] = true;
    }
}

}  // namespace

std::pair<double, bool> libecpint_gaussian_quadrature(
    int npoints, double exponent, double center, double tolerance) {
    if (npoints < 4 || npoints > 65536 || !std::isfinite(tolerance)
        || tolerance <= 0.0) {
        throw std::invalid_argument("invalid radial quadrature diagnostic grid");
    }
    libecpint::GCQuadrature grid;
    grid.initGrid(npoints, libecpint::ONEPOINT);
    std::function<double(double, const double*, int)> gaussian =
        [=](double x, const double*, int) {
            return std::exp(-exponent * (x - center) * (x - center));
        };
    const auto integral = grid.integrate(
        gaussian, nullptr, tolerance, 0, grid.getN() - 1);
    return {integral.first, integral.second != 0};
}

std::string libecpint_version() {
    // Returns the bare version string + MAX_L compile-time bound, no
    // "libecpint " prefix. The banner's "linked: ..." formatter adds the
    // library-name prefix itself (matching the libint / libxc / spglib
    // accessors which return just "2.13.1" / "7.0.0" / "2.7.0"); having
    // this accessor also include "libecpint " would double-print as
    // ``libecpint libecpint 1.0.7 ...`` on the banner.
    std::ostringstream ss;
    ss << "1.0.7 (vendored, MAX_L=" << LIBECPINT_MAX_L << ")";
    libecpint::ECPIntegrator dummy;
    (void)dummy.basis_is_set;
    return ss.str();
}

Eigen::MatrixXd compute_ecp_matrix(
    const BasisSet& basis,
    const std::vector<ECPCenter>& ecp_centers,
    const std::string& library_name,
    const std::string& share_dir) {
    const std::string xml_share_dir = resolve_ecp_share_dir(share_dir);
    if (xml_share_dir.empty()) {
        throw std::runtime_error(
            "compute_ecp_matrix: share_dir must be supplied (the build "
            "did not bake VIBEQC_LIBECPINT_SHARE_DIR; pass it via the "
            "share_dir argument).");
    }

    const auto& libint_basis = basis.libint();
    const std::size_t n_shells = libint_basis.size();

    // -----------------------------------------------------------------
    // Pack libint shells into the flat arrays libecpint expects.
    // -----------------------------------------------------------------
    std::vector<double> coords;
    coords.reserve(3 * n_shells);
    std::vector<double> exponents;
    std::vector<double> coefs;
    std::vector<int> ams;
    ams.reserve(n_shells);
    std::vector<int> shell_lengths;
    shell_lengths.reserve(n_shells);

    // Cartesian and spherical AO offsets per shell — needed for the
    // transform pass below.
    std::vector<std::size_t> cart_offsets(n_shells + 1, 0);
    std::vector<std::size_t> sph_offsets(n_shells + 1, 0);

    for (std::size_t s = 0; s < n_shells; ++s) {
        const auto& shell = libint_basis[s];
        coords.push_back(shell.O[0]);
        coords.push_back(shell.O[1]);
        coords.push_back(shell.O[2]);
        // vibe-qc constructs segmented basis sets (one contraction
        // per shell), so contr[0] holds the contraction.
        const auto& contr = shell.contr[0];
        ams.push_back(contr.l);
        shell_lengths.push_back(static_cast<int>(shell.alpha.size()));
        for (std::size_t p = 0; p < shell.alpha.size(); ++p) {
            exponents.push_back(shell.alpha[p]);
            coefs.push_back(contr.coeff[p]);
        }
        const std::size_t cart_size =
            ((contr.l + 1) * (contr.l + 2)) / 2;
        const std::size_t sph_size = 2 * contr.l + 1;
        cart_offsets[s + 1] = cart_offsets[s] + cart_size;
        sph_offsets[s + 1] = sph_offsets[s] + sph_size;
    }

    // -----------------------------------------------------------------
    // Pack ECP center streams.
    // -----------------------------------------------------------------
    const std::size_t n_ecps = ecp_centers.size();
    std::vector<double> ecp_coords;
    ecp_coords.reserve(3 * n_ecps);
    std::vector<int> ecp_charges;
    ecp_charges.reserve(n_ecps);
    std::vector<std::string> ecp_names(n_ecps, library_name);

    for (const auto& center : ecp_centers) {
        ecp_coords.push_back(center.xyz[0]);
        ecp_coords.push_back(center.xyz[1]);
        ecp_coords.push_back(center.xyz[2]);
        ecp_charges.push_back(center.Z);
    }

    // -----------------------------------------------------------------
    // Run the integrator.
    // -----------------------------------------------------------------
    libecpint::ECPIntegrator integrator;
    integrator.set_gaussian_basis(
        static_cast<int>(n_shells),
        coords.data(), exponents.data(), coefs.data(),
        ams.data(), shell_lengths.data());
#ifdef HAS_PUGIXML
    if (n_ecps > 0) {
        integrator.set_ecp_basis_from_library(
            static_cast<int>(n_ecps),
            ecp_coords.data(), ecp_charges.data(),
            ecp_names, xml_share_dir);
    }
#else
    static_assert(false, "libecpint must be built with pugixml support; "
                         "see scripts/build_libecpint.sh");
#endif
    integrator.init(0);  // 0 = no derivatives
    integrator.compute_integrals();

    // -----------------------------------------------------------------
    // Read out integrals (Cartesian basis) and transform per shell-pair
    // to spherical via libint's solidharmonics primitives.
    // -----------------------------------------------------------------
    const int ncart = integrator.ncart;
    const std::size_t nbf = basis.nbasis();
    Eigen::MatrixXd sph_matrix = Eigen::MatrixXd::Zero(nbf, nbf);

    // libint's tform_* helpers expect row-major source / target
    // buffers per shell-pair. Use scratch vectors per pair (small).
    for (std::size_t s1 = 0; s1 < n_shells; ++s1) {
        const int l1 = libint_basis[s1].contr[0].l;
        const std::size_t cart_1 = ((l1 + 1) * (l1 + 2)) / 2;
        const std::size_t sph_1 = 2 * l1 + 1;
        const std::size_t off_c1 = cart_offsets[s1];
        const std::size_t off_s1 = sph_offsets[s1];
        for (std::size_t s2 = 0; s2 < n_shells; ++s2) {
            const int l2 = libint_basis[s2].contr[0].l;
            const std::size_t cart_2 = ((l2 + 1) * (l2 + 2)) / 2;
            const std::size_t sph_2 = 2 * l2 + 1;
            const std::size_t off_c2 = cart_offsets[s2];
            const std::size_t off_s2 = sph_offsets[s2];

            // Pack source block (row-major).
            std::vector<double> source(cart_1 * cart_2);
            for (std::size_t i = 0; i < cart_1; ++i) {
                for (std::size_t j = 0; j < cart_2; ++j) {
                    // Bounds check: integrator.integrals is
                    // (ncart × ncart) row-major.
                    source[i * cart_2 + j] = integrator.integrals(
                        static_cast<int>(off_c1 + i),
                        static_cast<int>(off_c2 + j));
                }
            }
            (void)ncart;

            // Step 1: transform column dimension (l2: cart → sph).
            std::vector<double> mid(cart_1 * sph_2);
            libint2::solidharmonics::tform_cols(
                cart_1, l2, source.data(), mid.data());

            // Step 2: transform row dimension (l1: cart → sph).
            std::vector<double> target(sph_1 * sph_2);
            libint2::solidharmonics::tform_rows(
                l1, sph_2, mid.data(), target.data());

            // Scatter into the full spherical matrix.
            for (std::size_t i = 0; i < sph_1; ++i) {
                for (std::size_t j = 0; j < sph_2; ++j) {
                    sph_matrix(off_s1 + i, off_s2 + j) =
                        target[i * sph_2 + j];
                }
            }
        }
    }

    // V_ECP is symmetric; clean up tiny numerical asymmetry.
    sph_matrix = 0.5 * (sph_matrix + sph_matrix.transpose());
    return sph_matrix;
}

std::vector<double> ecp_effective_charges(
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const std::string& library_name,
    const std::string& share_dir) {
    const auto& atoms = mol.atoms();
    std::vector<double> Z_eff(atoms.size());
    if (ecp_centers.empty()) {
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            Z_eff[i] = static_cast<double>(atoms[i].Z);
        }
        return Z_eff;
    }
    // Same (Z, position) atom-matching convention as
    // compute_ecp_one_electron — keep the two in sync.
    std::vector<int> ecp_charges;
    ecp_charges.reserve(ecp_centers.size());
    for (const auto& c : ecp_centers) ecp_charges.push_back(c.Z);
    const auto core_map = ecp_core_electrons(ecp_charges, library_name, share_dir);
    const auto atom_cores = validated_xml_ecp_atom_cores(
        mol, ecp_centers, core_map, "ecp_effective_charges");
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        Z_eff[i] = static_cast<double>(atoms[i].Z - atom_cores[i]);
    }
    return Z_eff;
}

namespace {

// Packed libint-shell streams in libecpint's expected layout, plus the
// per-shell Cartesian / spherical offsets needed to fold a Cartesian
// derivative block back into the spherical AO basis. Shared by the
// XML-library and inline-primitive ECP gradient paths so both
// differentiate through identical machinery (#574).
struct EcpGaussianStream {
    std::vector<double> coords;
    std::vector<double> exponents;
    std::vector<double> coefs;
    std::vector<int> ams;
    std::vector<int> shell_lengths;
    std::vector<std::size_t> cart_offsets;
    std::vector<std::size_t> sph_offsets;
    std::size_t n_shells = 0;
};

// libecpint clusters shells into atoms by unique coordinates in the order
// they appear, which matches ``mol.atoms()`` (vibe-qc emits per-atom shells
// in molecule order).
EcpGaussianStream pack_ecp_gaussian_stream(const BasisSet& basis) {
    const auto& libint_basis = basis.libint();
    EcpGaussianStream gs;
    gs.n_shells = libint_basis.size();
    gs.coords.reserve(3 * gs.n_shells);
    gs.ams.reserve(gs.n_shells);
    gs.shell_lengths.reserve(gs.n_shells);
    gs.cart_offsets.assign(gs.n_shells + 1, 0);
    gs.sph_offsets.assign(gs.n_shells + 1, 0);
    for (std::size_t s = 0; s < gs.n_shells; ++s) {
        const auto& shell = libint_basis[s];
        gs.coords.push_back(shell.O[0]);
        gs.coords.push_back(shell.O[1]);
        gs.coords.push_back(shell.O[2]);
        const auto& contr = shell.contr[0];
        gs.ams.push_back(contr.l);
        gs.shell_lengths.push_back(static_cast<int>(shell.alpha.size()));
        for (std::size_t p = 0; p < shell.alpha.size(); ++p) {
            gs.exponents.push_back(shell.alpha[p]);
            gs.coefs.push_back(contr.coeff[p]);
        }
        gs.cart_offsets[s + 1] =
            gs.cart_offsets[s] + ((contr.l + 1) * (contr.l + 2)) / 2;
        gs.sph_offsets[s + 1] =
            gs.sph_offsets[s] + 2 * static_cast<std::size_t>(contr.l) + 1;
    }
    return gs;
}

// Per-component (atom, direction) contraction:
//   dE_ECP/dR_{A,c} = sum_munu D_munu * (dV_ECP_munu / dR_{A,c}).
// libecpint returns the Cartesian-basis derivative for each (A, c);
// transform Cartesian -> spherical per shell-pair via libint's
// solidharmonics helpers (same recipe as compute_ecp_matrix) and
// contract with D. ``integrator`` must already have had init(1) and
// compute_first_derivs() called on it.
Eigen::MatrixXd contract_ecp_first_derivs(
    const BasisSet& basis,
    const libecpint::ECPIntegrator& integrator,
    const EcpGaussianStream& gs,
    const Eigen::MatrixXd& D,
    const int n_atoms,
    const char* fn_name) {
    if (static_cast<int>(integrator.natoms) != n_atoms) {
        throw std::runtime_error(
            std::string(fn_name) + ": libecpint reports "
            + std::to_string(integrator.natoms)
            + " atoms but the molecule has " + std::to_string(n_atoms)
            + ". This means the gaussian basis omits a shell for one of "
            "the atoms (ghost / empty-basis atoms are not supported by "
            "the ECP gradient path).");
    }
    const auto& libint_basis = basis.libint();
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);
    for (int A = 0; A < n_atoms; ++A) {
        for (int c = 0; c < 3; ++c) {
            const auto& deriv_cart = integrator.first_derivs[3 * A + c];
            double acc = 0.0;
            for (std::size_t s1 = 0; s1 < gs.n_shells; ++s1) {
                const int l1 = libint_basis[s1].contr[0].l;
                const std::size_t cart_1 = ((l1 + 1) * (l1 + 2)) / 2;
                const std::size_t sph_1 = 2 * l1 + 1;
                const std::size_t off_c1 = gs.cart_offsets[s1];
                const std::size_t off_s1 = gs.sph_offsets[s1];
                for (std::size_t s2 = 0; s2 < gs.n_shells; ++s2) {
                    const int l2 = libint_basis[s2].contr[0].l;
                    const std::size_t cart_2 = ((l2 + 1) * (l2 + 2)) / 2;
                    const std::size_t sph_2 = 2 * l2 + 1;
                    const std::size_t off_c2 = gs.cart_offsets[s2];
                    const std::size_t off_s2 = gs.sph_offsets[s2];

                    std::vector<double> source(cart_1 * cart_2);
                    for (std::size_t i = 0; i < cart_1; ++i) {
                        for (std::size_t j = 0; j < cart_2; ++j) {
                            source[i * cart_2 + j] = deriv_cart(
                                static_cast<int>(off_c1 + i),
                                static_cast<int>(off_c2 + j));
                        }
                    }
                    std::vector<double> mid(cart_1 * sph_2);
                    libint2::solidharmonics::tform_cols(
                        cart_1, l2, source.data(), mid.data());
                    std::vector<double> target(sph_1 * sph_2);
                    libint2::solidharmonics::tform_rows(
                        l1, sph_2, mid.data(), target.data());

                    for (std::size_t i = 0; i < sph_1; ++i) {
                        for (std::size_t j = 0; j < sph_2; ++j) {
                            acc += D(static_cast<Eigen::Index>(off_s1 + i),
                                     static_cast<Eigen::Index>(off_s2 + j))
                                 * target[i * sph_2 + j];
                        }
                    }
                }
            }
            grad(A, c) = acc;
        }
    }
    return grad;
}

}  // namespace

Eigen::MatrixXd compute_ecp_gradient_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const Eigen::MatrixXd& D,
    const std::string& library_name,
    const std::string& share_dir) {
    const int n_atoms = static_cast<int>(mol.atoms().size());
    if (ecp_centers.empty()) {
        return Eigen::MatrixXd::Zero(n_atoms, 3);
    }
    // The raw derivative integrator accepts arbitrary centers. Validate the
    // molecular one-to-one placement contract before producing a gradient
    // that could not correspond to the SCF Hamiltonian.
    (void)ecp_effective_charges(
        mol, ecp_centers, library_name, share_dir);
    const std::string xml_share_dir = resolve_ecp_share_dir(share_dir);
    if (xml_share_dir.empty()) {
        throw std::runtime_error(
            "compute_ecp_gradient_contribution: share_dir must be supplied "
            "(the build did not bake VIBEQC_LIBECPINT_SHARE_DIR; pass via "
            "share_dir).");
    }

    const EcpGaussianStream gs = pack_ecp_gaussian_stream(basis);

    const std::size_t n_ecps = ecp_centers.size();
    std::vector<double> ecp_coords;
    ecp_coords.reserve(3 * n_ecps);
    std::vector<int> ecp_charges;
    ecp_charges.reserve(n_ecps);
    std::vector<std::string> ecp_names(n_ecps, library_name);
    for (const auto& center : ecp_centers) {
        ecp_coords.push_back(center.xyz[0]);
        ecp_coords.push_back(center.xyz[1]);
        ecp_coords.push_back(center.xyz[2]);
        ecp_charges.push_back(center.Z);
    }

    libecpint::ECPIntegrator integrator;
    integrator.set_gaussian_basis(
        static_cast<int>(gs.n_shells),
        gs.coords.data(), gs.exponents.data(), gs.coefs.data(),
        gs.ams.data(), gs.shell_lengths.data());
#ifdef HAS_PUGIXML
    integrator.set_ecp_basis_from_library(
        static_cast<int>(n_ecps),
        ecp_coords.data(), ecp_charges.data(),
        ecp_names, xml_share_dir);
#else
    static_assert(false, "libecpint must be built with pugixml support; "
                         "see scripts/build_libecpint.sh");
#endif
    integrator.init(1);  // first derivatives
    integrator.compute_first_derivs();

    return contract_ecp_first_derivs(
        basis, integrator, gs, D, n_atoms,
        "compute_ecp_gradient_contribution");
}

Eigen::MatrixXd compute_ecp_gradient_contribution_from_primitives(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<std::array<double, 3>>& ecp_centers,
    const std::vector<ECPPrimitiveBlock>& primitives,
    const Eigen::MatrixXd& D) {
    const int n_atoms = static_cast<int>(mol.atoms().size());
    if (primitives.empty()) {
        return Eigen::MatrixXd::Zero(n_atoms, 3);
    }
    if (ecp_centers.size() != primitives.size()) {
        throw std::invalid_argument(
            "compute_ecp_gradient_contribution_from_primitives: center/block "
            "count mismatch");
    }
    validate_primitive_ecp_centers(
        mol,
        ecp_centers,
        "compute_ecp_gradient_contribution_from_primitives");

    const EcpGaussianStream gs = pack_ecp_gaussian_stream(basis);

    std::vector<double> center_xyz;
    center_xyz.reserve(3 * ecp_centers.size());
    for (const auto& xyz : ecp_centers) {
        center_xyz.push_back(xyz[0]);
        center_xyz.push_back(xyz[1]);
        center_xyz.push_back(xyz[2]);
    }
    std::vector<double> ecp_exp;
    std::vector<double> ecp_coef;
    std::vector<int> ecp_ams;
    std::vector<int> ecp_ns;
    std::vector<int> n_prim;
    n_prim.reserve(primitives.size());
    for (const auto& p : primitives) {
        n_prim.push_back(p.n_primitive);
        ecp_exp.insert(ecp_exp.end(), p.exponents.begin(), p.exponents.end());
        ecp_coef.insert(ecp_coef.end(), p.coefficients.begin(),
                        p.coefficients.end());
        ecp_ams.insert(ecp_ams.end(), p.ams.begin(), p.ams.end());
        ecp_ns.insert(ecp_ns.end(), p.ns.begin(), p.ns.end());
    }

    libecpint::ECPIntegrator integrator;
    integrator.set_gaussian_basis(
        static_cast<int>(gs.n_shells),
        gs.coords.data(), gs.exponents.data(), gs.coefs.data(),
        gs.ams.data(), gs.shell_lengths.data());
    integrator.set_ecp_basis(
        static_cast<int>(primitives.size()), center_xyz.data(),
        ecp_exp.data(), ecp_coef.data(), ecp_ams.data(), ecp_ns.data(),
        n_prim.data());
    integrator.init(1);  // first derivatives
    integrator.compute_first_derivs();

    return contract_ecp_first_derivs(
        basis, integrator, gs, D, n_atoms,
        "compute_ecp_gradient_contribution_from_primitives");
}

std::map<int, int> ecp_core_electrons(
    const std::vector<int>& charges,
    const std::string& library_name,
    const std::string& share_dir) {
    const std::string xml_share_dir = resolve_ecp_share_dir(share_dir);
    if (xml_share_dir.empty()) {
        throw std::runtime_error(
            "ecp_core_electrons: share_dir must be supplied (build did "
            "not bake VIBEQC_LIBECPINT_SHARE_DIR).");
    }
    if (charges.empty()) {
        return {};
    }

    // libecpint's set_ecp_basis_from_library wants a coords stream too;
    // the positions don't affect the core_electrons map populated by
    // load. Use the origin for every entry — purely a vehicle for the
    // XML lookup.
    const std::size_t n_ecps = charges.size();
    std::vector<double> coords(3 * n_ecps, 0.0);
    std::vector<int> ecp_charges(charges.begin(), charges.end());
    std::vector<std::string> ecp_names(n_ecps, library_name);

    libecpint::ECPIntegrator integrator;
    // We need ANY gaussian basis to call init() — but for ncore lookup
    // alone we don't need init() at all; just load the ECPs and read
    // the map from integrator.ecps.
#ifdef HAS_PUGIXML
    integrator.set_ecp_basis_from_library(
        static_cast<int>(n_ecps),
        coords.data(), ecp_charges.data(),
        ecp_names, xml_share_dir);
#else
    static_assert(false, "libecpint must be built with pugixml support");
#endif

    std::map<int, int> result;
    for (const auto& kv : integrator.ecps.core_electrons) {
        result[kv.first] = kv.second;
    }
    return result;
}

MolecularECPInput validate_molecular_ecp_dispatch(
    const std::vector<ECPCenter>& xml_centers,
    const std::string& xml_library,
    const std::vector<ECPPrimitiveBlock>& primitive_blocks,
    const std::vector<std::array<double, 3>>& primitive_centers,
    const std::vector<double>& effective_charges,
    int total_ncore,
    const std::string& route) {
    if (total_ncore < 0) {
        throw std::invalid_argument(
            route + ": ecp_total_ncore must be non-negative");
    }

    const bool has_xml_input =
        !xml_centers.empty() || !xml_library.empty();
    const bool has_primitive_input =
        !primitive_blocks.empty() || !primitive_centers.empty()
        || !effective_charges.empty() || total_ncore != 0;

    if (has_xml_input && has_primitive_input) {
        throw std::invalid_argument(
            route + ": XML-library and inline-primitive ECP inputs are "
                    "mutually exclusive");
    }
    if (xml_centers.empty() && !xml_library.empty()) {
        throw std::invalid_argument(
            route + ": ecp_library requires non-empty ecp_centers");
    }
    if (primitive_blocks.empty()
        && (!primitive_centers.empty() || !effective_charges.empty()
            || total_ncore != 0)) {
        throw std::invalid_argument(
            route + ": inline-primitive ECP metadata requires non-empty "
                    "ecp_primitive_blocks");
    }

    if (!primitive_blocks.empty()) {
        return MolecularECPInput::INLINE_PRIMITIVES;
    }
    if (!xml_centers.empty()) {
        return MolecularECPInput::XML_LIBRARY;
    }
    return MolecularECPInput::NONE;
}

ECPHcore compute_ecp_one_electron(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const std::string& ecp_library) {
    ECPHcore out;
    if (ecp_centers.empty()) {
        out.V = compute_nuclear(basis, mol);
        out.V_ecp = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(basis.nbasis()),
            static_cast<Eigen::Index>(basis.nbasis()));
        out.E_nuc = mol.nuclear_repulsion();
        out.total_ncore = 0;
        return out;
    }

    const std::string lib =
        ecp_library.empty() ? std::string("ecp10mdf") : ecp_library;

    // Look up ncore per Z via libecpint's library.
    std::vector<int> ecp_charges;
    ecp_charges.reserve(ecp_centers.size());
    for (const auto& c : ecp_centers) ecp_charges.push_back(c.Z);
    const auto core_map = ecp_core_electrons(ecp_charges, lib, "");
    const auto atom_cores = validated_xml_ecp_atom_cores(
        mol, ecp_centers, core_map, "compute_ecp_one_electron");

    // Build per-atom positions + effective charges, and accumulate
    // the total number of replaced-core electrons (used by the SCF
    // to compute valence-only nocc).
    const auto& atoms = mol.atoms();
    std::vector<std::array<double, 3>> positions(atoms.size());
    std::vector<double> Z_eff(atoms.size());
    int total_ncore = 0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        positions[i] = atoms[i].xyz;
        const int ncore_i = atom_cores[i];
        Z_eff[i] = static_cast<double>(atoms[i].Z - ncore_i);
        total_ncore += ncore_i;
    }
    out.total_ncore = total_ncore;

    out.V = compute_nuclear_with_charges(basis, positions, Z_eff);
    out.V_ecp = compute_ecp_matrix(basis, ecp_centers, lib, "");

    // Effective nuclear repulsion: Σ_{A<B} Z_eff_A · Z_eff_B / R_AB.
    double E_nuc = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            const double dx = atoms[i].xyz[0] - atoms[j].xyz[0];
            const double dy = atoms[i].xyz[1] - atoms[j].xyz[1];
            const double dz = atoms[i].xyz[2] - atoms[j].xyz[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            E_nuc += Z_eff[i] * Z_eff[j] / r;
        }
    }
    out.E_nuc = E_nuc;
    return out;
}

ECPHcore compute_ecp_one_electron_from_primitives(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<std::array<double, 3>>& ecp_centers,
    const std::vector<ECPPrimitiveBlock>& primitives,
    const std::vector<double>& effective_charges,
    int total_ncore) {
    ECPHcore out;
    if (primitives.empty()) {
        if (!ecp_centers.empty() || !effective_charges.empty()
            || total_ncore != 0) {
            throw std::invalid_argument(
                "compute_ecp_one_electron_from_primitives: empty primitive "
                "blocks require empty centers/effective_charges and zero "
                "total_ncore");
        }
        out.V = compute_nuclear(basis, mol);
        out.V_ecp = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(basis.nbasis()),
            static_cast<Eigen::Index>(basis.nbasis()));
        out.E_nuc = mol.nuclear_repulsion();
        out.total_ncore = 0;
        return out;
    }
    if (ecp_centers.size() != primitives.size()) {
        throw std::invalid_argument(
            "compute_ecp_one_electron_from_primitives: center/block count "
            "mismatch");
    }
    const auto& atoms = mol.atoms();
    if (effective_charges.size() != atoms.size()) {
        throw std::invalid_argument(
            "compute_ecp_one_electron_from_primitives: effective_charges "
            "must have one entry per atom");
    }
    if (total_ncore < 0) {
        throw std::invalid_argument(
            "compute_ecp_one_electron_from_primitives: total_ncore is negative");
    }
    validate_primitive_ecp_centers(
        mol,
        ecp_centers,
        "compute_ecp_one_electron_from_primitives");
    double observed_ncore = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const double effective = effective_charges[i];
        if (!std::isfinite(effective)
            || effective < -1e-12
            || effective > static_cast<double>(atoms[i].Z) + 1e-12) {
            throw std::invalid_argument(
                "compute_ecp_one_electron_from_primitives: effective nuclear "
                "charges must be finite and lie between zero and atom.Z");
        }
        observed_ncore += static_cast<double>(atoms[i].Z) - effective;
    }
    if (std::abs(observed_ncore - static_cast<double>(total_ncore)) > 1e-8) {
        throw std::invalid_argument(
            "compute_ecp_one_electron_from_primitives: effective nuclear "
            "charges are inconsistent with total_ncore");
    }

    std::vector<std::array<double, 3>> positions(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) positions[i] = atoms[i].xyz;

    out.V = compute_nuclear_with_charges(basis, positions, effective_charges);

    std::vector<double> center_xyz;
    center_xyz.reserve(3 * ecp_centers.size());
    for (const auto& xyz : ecp_centers) {
        center_xyz.push_back(xyz[0]);
        center_xyz.push_back(xyz[1]);
        center_xyz.push_back(xyz[2]);
    }
    out.V_ecp = compute_ecp_matrix_from_primitives(basis, center_xyz, primitives);
    out.total_ncore = total_ncore;

    double E_nuc = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            const double dx = atoms[i].xyz[0] - atoms[j].xyz[0];
            const double dy = atoms[i].xyz[1] - atoms[j].xyz[1];
            const double dz = atoms[i].xyz[2] - atoms[j].xyz[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            E_nuc += effective_charges[i] * effective_charges[j] / r;
        }
    }
    out.E_nuc = E_nuc;
    return out;
}

Eigen::MatrixXd compute_nuclear_with_charges(
    const BasisSet& basis,
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<double>& charges) {
    if (positions.size() != charges.size()) {
        throw std::invalid_argument(
            "compute_nuclear_with_charges: positions and charges must "
            "have the same length");
    }
    ensure_libint_initialized();

    // Build the (charge, xyz) pair list in the shape libint expects.
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        q.emplace_back(charges[i], positions[i]);
    }

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto n_basis = static_cast<Eigen::Index>(basis.nbasis());
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n_basis, n_basis);

    libint2::Engine engine(libint2::Operator::nuclear,
                           shells.max_nprim(), shells.max_l(), 0);
    engine.set_params(q);
    const auto& buf = engine.results();
    for (std::size_t s1 = 0; s1 != shells.size(); ++s1) {
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (std::size_t s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            engine.compute(shells[s1], shells[s2]);
            const double* block = buf[0];
            if (block == nullptr) continue;
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>
                buf_mat(block, n1, n2);
            matrix.block(bf1, bf2, n1, n2) = buf_mat;
            if (s1 != s2) {
                matrix.block(bf2, bf1, n2, n1) = buf_mat.transpose();
            }
        }
    }
    return matrix;
}

// ---- Phase 14g: inline-primitive ECP integrals -------------------------

Eigen::MatrixXd compute_ecp_matrix_from_primitives(
    const BasisSet& basis,
    const std::vector<double>& center_xyz,
    const std::vector<ECPPrimitiveBlock>& primitives) {
    const std::size_t n_centers = primitives.size();
    if (n_centers == 0) {
        const Eigen::Index nbf = static_cast<Eigen::Index>(basis.nbasis());
        return Eigen::MatrixXd::Zero(nbf, nbf);
    }
    if (center_xyz.size() != 3 * n_centers) {
        throw std::invalid_argument(
            "compute_ecp_matrix_from_primitives: center_xyz must have "
            "length 3*n_centers");
    }
    const auto& libint_basis = basis.libint();
    const std::size_t n_shells = libint_basis.size();
    std::vector<double> coords; coords.reserve(3*n_shells);
    std::vector<double> exponents; std::vector<double> coefs;
    std::vector<int> ams; ams.reserve(n_shells);
    std::vector<int> shell_lengths; shell_lengths.reserve(n_shells);
    std::vector<std::size_t> cart_offsets(n_shells+1,0);
    std::vector<std::size_t> sph_offsets(n_shells+1,0);
    for (std::size_t s=0; s<n_shells; ++s) {
        const auto& shell=libint_basis[s];
        coords.push_back(shell.O[0]); coords.push_back(shell.O[1]); coords.push_back(shell.O[2]);
        const auto& contr=shell.contr[0]; ams.push_back(contr.l);
        shell_lengths.push_back(static_cast<int>(shell.alpha.size()));
        for (std::size_t p=0; p<shell.alpha.size(); ++p) {
            exponents.push_back(shell.alpha[p]); coefs.push_back(contr.coeff[p]);
        }
        cart_offsets[s+1]=cart_offsets[s]+((contr.l+1)*(contr.l+2))/2;
        sph_offsets[s+1]=sph_offsets[s]+2*contr.l+1;
    }
    std::vector<double> ecp_exp, ecp_coef; std::vector<int> ecp_ams, ecp_ns;
    std::vector<int> n_prim; n_prim.reserve(n_centers);
    for (const auto& p:primitives) {
        n_prim.push_back(p.n_primitive);
        ecp_exp.insert(ecp_exp.end(), p.exponents.begin(), p.exponents.end());
        ecp_coef.insert(ecp_coef.end(), p.coefficients.begin(), p.coefficients.end());
        ecp_ams.insert(ecp_ams.end(), p.ams.begin(), p.ams.end());
        ecp_ns.insert(ecp_ns.end(), p.ns.begin(), p.ns.end());
    }
    libecpint::ECPIntegrator integrator;
    integrator.set_gaussian_basis(static_cast<int>(n_shells), coords.data(), exponents.data(), coefs.data(), ams.data(), shell_lengths.data());
    integrator.set_ecp_basis(static_cast<int>(n_centers), center_xyz.data(), ecp_exp.data(), ecp_coef.data(), ecp_ams.data(), ecp_ns.data(), n_prim.data());
    integrator.init(0); integrator.compute_integrals();
    const int ncart=integrator.ncart; const std::size_t nbf=basis.nbasis();
    Eigen::MatrixXd sph_matrix=Eigen::MatrixXd::Zero(nbf,nbf);
    for (std::size_t s1=0; s1<n_shells; ++s1) {
        const int l1=libint_basis[s1].contr[0].l;
        const std::size_t cart_1=((l1+1)*(l1+2))/2, sph_1=2*l1+1;
        const std::size_t off_c1=cart_offsets[s1], off_s1=sph_offsets[s1];
        for (std::size_t s2=0; s2<n_shells; ++s2) {
            const int l2=libint_basis[s2].contr[0].l;
            const std::size_t cart_2=((l2+1)*(l2+2))/2, sph_2=2*l2+1;
            const std::size_t off_c2=cart_offsets[s2], off_s2=sph_offsets[s2];
            std::vector<double> source(cart_1*cart_2);
            for (std::size_t i=0; i<cart_1; ++i)
                for (std::size_t j=0; j<cart_2; ++j)
                    source[i*cart_2+j]=integrator.integrals(static_cast<int>(off_c1+i), static_cast<int>(off_c2+j));
            (void)ncart;
            std::vector<double> mid(cart_1*sph_2);
            libint2::solidharmonics::tform_cols(cart_1,l2,source.data(),mid.data());
            std::vector<double> target(sph_1*sph_2);
            libint2::solidharmonics::tform_rows(l1,sph_2,mid.data(),target.data());
            for (std::size_t i=0; i<sph_1; ++i)
                for (std::size_t j=0; j<sph_2; ++j)
                    sph_matrix(off_s1+i,off_s2+j)=target[i*sph_2+j];
        }
    }
    sph_matrix=0.5*(sph_matrix+sph_matrix.transpose());
    return sph_matrix;
}

namespace {
libecpint::GaussianShell ecp_gaussian_shell(
    const libint2::Shell& shell, const Eigen::Vector3d& shift) {
    const std::array<double, 3> center = {
        shell.O[0] + shift[0], shell.O[1] + shift[1], shell.O[2] + shift[2]};
    libecpint::GaussianShell result(center, shell.contr[0].l);
    for (std::size_t p = 0; p < shell.alpha.size(); ++p) {
        result.addPrim(shell.alpha[p], shell.contr[0].coeff[p]);
    }
    return result;
}
struct PeriodicEcpPotentials {
    std::vector<libecpint::ECP> potentials;
    int max_l = 0;
};

PeriodicEcpPotentials prepare_periodic_ecp(
    const std::vector<std::array<double, 3>>& home_ecp_centers,
    const std::vector<ECPPrimitiveBlock>& home_primitives) {
    if (home_ecp_centers.size() != home_primitives.size()) {
        throw std::invalid_argument("periodic ECP centers and primitive blocks differ in length");
    }
    // Validate before entering either the library or an OpenMP region.
    std::vector<libecpint::ECP> potentials;
    int max_ecp_l = 0;
    for (std::size_t a = 0; a < home_primitives.size(); ++a) {
        const auto& p = home_primitives[a];
        const auto n = p.exponents.size();
        if (p.n_primitive <= 0 || static_cast<std::size_t>(p.n_primitive) != n
            || p.coefficients.size() != n || p.ams.size() != n || p.ns.size() != n) {
            throw std::invalid_argument("periodic ECP primitive arrays must have matching positive lengths");
        }
        for (double x : home_ecp_centers[a]) {
            if (!std::isfinite(x)) throw std::invalid_argument("periodic ECP centers must be finite");
        }
        libecpint::ECP potential(home_ecp_centers[a].data());
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(p.exponents[i]) || p.exponents[i] <= 0.0
                || !std::isfinite(p.coefficients[i]) || p.ams[i] < 0
                || p.ams[i] > LIBECPINT_MAX_L || p.ns[i] < 0) {
                throw std::invalid_argument("invalid periodic ECP primitive exponent, coefficient, angular momentum or radial power");
            }
            potential.addPrimitive(p.ns[i], p.ams[i], p.exponents[i], p.coefficients[i], false);
            max_ecp_l = std::max(max_ecp_l, p.ams[i]);
        }
        potential.sort();
        potentials.push_back(std::move(potential));
    }
    return {std::move(potentials), max_ecp_l};
}

}  // namespace

LatticeMatrixSet compute_ecp_lattice_from_primitives(
    const BasisSet& basis, const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<std::array<double, 3>>& home_ecp_centers,
    const std::vector<ECPPrimitiveBlock>& home_primitives) {
    const auto prepared = prepare_periodic_ecp(home_ecp_centers, home_primitives);
    const auto& potentials = prepared.potentials;
    const int max_ecp_l = prepared.max_l;
    const auto cells = opts.pair_complete_1e
        ? pair_complete_lattice_cells(basis, system, opts.cutoff_bohr)
        : direct_lattice_cells(system, opts.cutoff_bohr);
    const int nbf = static_cast<int>(basis.nbasis());
    LatticeMatrixSet result;
    result.nbf = nbf;
    result.cells = cells;
    result.blocks.assign(cells.size(), Eigen::MatrixXd::Zero(nbf, nbf));
    if (potentials.empty()) return result;
    ensure_libint_initialized();
    const auto images = direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    int max_basis_l = 0;
    for (const auto& shell : shells) max_basis_l = std::max(max_basis_l, shell.contr[0].l);
    std::exception_ptr failure;
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t c = 0; c < cells.size(); ++c) {
        try {
            libecpint::ECPIntegral engine(max_basis_l, max_ecp_l);
            auto& block = result.blocks[c];
            // V(g) = <mu_0 | sum_T U_T | nu_g>. It is generally NOT
            // symmetric at fixed g. Shifting both orbitals instead repeats
            // a home-cell matrix in every off-cell block.
            for (std::size_t s1 = 0; s1 < shells.size(); ++s1) {
                const auto bra = ecp_gaussian_shell(shells[s1], Eigen::Vector3d::Zero());
                const int l1 = bra.am(), nc1 = bra.ncartesian(), ns1 = 2*l1 + 1;
                for (std::size_t s2 = 0; s2 < shells.size(); ++s2) {
                    const auto ket = ecp_gaussian_shell(shells[s2], cells[c].r_cart);
                    const int l2 = ket.am(), nc2 = ket.ncartesian(), ns2 = 2*l2 + 1;
                    std::vector<double> cartesian(nc1*nc2, 0.0);
                    libecpint::TwoIndex<double> values;
                    // Stream one projector image and shell pair. No full
                    // Cartesian supercell matrix or replicated ECP stream.
                    for (const auto& image : images) {
                        for (const auto& home : potentials) {
                            auto potential = home;
                            potential.setPos(home.center_[0] + image.r_cart[0],
                                             home.center_[1] + image.r_cart[1],
                                             home.center_[2] + image.r_cart[2]);
                            engine.compute_shell_pair(potential, bra, ket, values);
                            for (int i = 0; i < nc1; ++i) {
                                for (int j = 0; j < nc2; ++j) cartesian[i*nc2 + j] += values(i, j);
                            }
                        }
                    }
                    std::vector<double> intermediate(nc1*ns2), spherical(ns1*ns2);
                    libint2::solidharmonics::tform_cols(nc1, l2, cartesian.data(), intermediate.data());
                    libint2::solidharmonics::tform_rows(l1, ns2, intermediate.data(), spherical.data());
                    for (int i = 0; i < ns1; ++i) {
                        for (int j = 0; j < ns2; ++j) {
                            block(shell2bf[s1] + i, shell2bf[s2] + j) = spherical[i*ns2 + j];
                        }
                    }
                }
            }
        } catch (...) {
            #pragma omp critical(vibeqc_periodic_ecp_failure)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
    return result;
}

// Differentiate the same finite projector/image sum as the value builder.
// The caller supplies the accepted lattice density; no home-cell or fixed-g
// symmetry is inferred. All translated copies move with their home atom.
Eigen::MatrixXd ecp_lattice_gradient_contribution_from_primitives(
    const BasisSet& basis, const PeriodicSystem& system,
    const LatticeMatrixSet& density, const LatticeSumOptions& opts,
    const std::vector<std::array<double, 3>>& home_ecp_centers,
    const std::vector<ECPPrimitiveBlock>& home_primitives) {
    const auto prepared = prepare_periodic_ecp(home_ecp_centers, home_primitives);
    const auto& potentials = prepared.potentials;
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const int nbf = static_cast<int>(basis.nbasis());
    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());
    if (density.nbf != nbf || density.cells.size() != density.blocks.size()) {
        throw std::invalid_argument("periodic ECP gradient density shape mismatch");
    }
    std::map<std::array<int, 3>, std::size_t> density_index;
    for (std::size_t c = 0; c < density.cells.size(); ++c) {
        const auto& cell = density.cells[c];
        const auto& block = density.blocks[c];
        if (block.rows() != nbf || block.cols() != nbf || !block.allFinite()
            || !cell.r_cart.allFinite()
            || (cell.r_cart - system.lattice * cell.index.cast<double>()).norm() > 1e-9) {
            throw std::invalid_argument("periodic ECP gradient requires finite aligned density cells");
        }
        const std::array<int, 3> key = {cell.index[0], cell.index[1], cell.index[2]};
        if (!density_index.emplace(key, c).second) {
            throw std::invalid_argument("periodic ECP gradient density has duplicate cells");
        }
    }
    const auto cells = opts.pair_complete_1e
        ? pair_complete_lattice_cells(basis, system, opts.cutoff_bohr)
        : direct_lattice_cells(system, opts.cutoff_bohr);
    std::vector<std::size_t> density_positions;
    for (const auto& cell : cells) {
        const auto found = density_index.find({cell.index[0], cell.index[1], cell.index[2]});
        if (found == density_index.end()) {
            throw std::invalid_argument("periodic ECP gradient density does not cover the energy image domain");
        }
        density_positions.push_back(found->second);
    }
    std::vector<int> shell_atoms(shells.size());
    int max_basis_l = 0;
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const int a = basis.shell_atom_index(s);
        if (a < 0 || a >= n_atoms) {
            throw std::invalid_argument("periodic ECP gradient shell has no home atom");
        }
        for (int d = 0; d < 3; ++d) {
            if (std::abs(shells[s].O[d] - system.unit_cell[a].xyz[d]) > 1e-9) {
                throw std::invalid_argument("periodic ECP gradient shell is not on its home atom");
            }
        }
        shell_atoms[s] = a;
        max_basis_l = std::max(max_basis_l, shells[s].contr[0].l);
    }
    std::vector<int> projector_atoms;
    for (const auto& center : home_ecp_centers) {
        int owner = -1;
        for (Eigen::Index a = 0; a < n_atoms; ++a) {
            double distance = 0.0;
            for (int d = 0; d < 3; ++d) {
                distance += std::abs(center[d] - system.unit_cell[a].xyz[d]);
            }
            if (distance <= 1e-9) {
                if (owner >= 0) {
                    throw std::invalid_argument("periodic ECP gradient projector has ambiguous home atom");
                }
                owner = static_cast<int>(a);
            }
        }
        if (owner < 0) {
            throw std::invalid_argument("periodic ECP gradient projector is not on a home atom");
        }
        projector_atoms.push_back(owner);
    }
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(n_atoms, 3);
    if (potentials.empty()) return gradient;
    ensure_libint_initialized();
    const auto images = direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    // One small nuclear accumulator per density cell allows deterministic
    // reduction without storing derivative AO matrices for the image sum.
    std::vector<Eigen::MatrixXd> cell_gradients(
        cells.size(), Eigen::MatrixXd::Zero(n_atoms, 3));
    std::exception_ptr failure;
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t c = 0; c < cells.size(); ++c) {
        try {
            libecpint::ECPIntegral engine(max_basis_l, prepared.max_l, 1);
            auto& local = cell_gradients[c];
            const auto& D = density.blocks[density_positions[c]];
            for (std::size_t s1 = 0; s1 < shells.size(); ++s1) {
                const auto bra = ecp_gaussian_shell(shells[s1], Eigen::Vector3d::Zero());
                const int l1 = bra.am(), nc1 = bra.ncartesian(), ns1 = 2*l1 + 1;
                for (std::size_t s2 = 0; s2 < shells.size(); ++s2) {
                    const auto ket = ecp_gaussian_shell(shells[s2], cells[c].r_cart);
                    const int l2 = ket.am(), nc2 = ket.ncartesian(), ns2 = 2*l2 + 1;
                    const auto D_pair = D.block(shell2bf[s1], shell2bf[s2], ns1, ns2);
                    if (D_pair.isZero(0.0)) continue;
                    std::array<libecpint::TwoIndex<double>, 9> derivatives;
                    std::vector<double> cartesian(nc1*nc2), intermediate(nc1*ns2), spherical(ns1*ns2);
                    for (const auto& image : images) {
                        for (std::size_t a = 0; a < potentials.size(); ++a) {
                            const auto& home = potentials[a];
                            auto potential = home;
                            potential.setPos(home.center_[0] + image.r_cart[0],
                                             home.center_[1] + image.r_cart[1],
                                             home.center_[2] + image.r_cart[2]);
                            engine.compute_shell_pair_derivative(potential, bra, ket, derivatives);
                            const std::array<int, 3> owners = {
                                shell_atoms[s1], shell_atoms[s2], projector_atoms[a]};
                            // libecpint combines coincident-centre partials in
                            // these nine slots. Add every slot to its owner;
                            // do not discard or double a coincident projector.
                            for (int center = 0; center < 3; ++center) {
                                for (int d = 0; d < 3; ++d) {
                                    const auto& value = derivatives[3*center + d];
                                    for (int i = 0; i < nc1; ++i) {
                                        for (int j = 0; j < nc2; ++j) cartesian[i*nc2 + j] = value(i, j);
                                    }
                                    libint2::solidharmonics::tform_cols(nc1, l2, cartesian.data(), intermediate.data());
                                    libint2::solidharmonics::tform_rows(l1, ns2, intermediate.data(), spherical.data());
                                    double contribution = 0.0;
                                    for (int i = 0; i < ns1; ++i) {
                                        for (int j = 0; j < ns2; ++j) contribution += D_pair(i, j) * spherical[i*ns2 + j];
                                    }
                                    local(owners[center], d) += contribution;
                                }
                            }
                        }
                    }
                }
            }
        } catch (...) {
            #pragma omp critical(vibeqc_periodic_ecp_failure)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
    for (const auto& local : cell_gradients) gradient += local;
    return gradient;
}

}  // namespace vibeqc
