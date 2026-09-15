// pybind11 bindings for the vibe-qc C++ core.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <libint2/engine.h>
#include <libint2/util/configuration.h>
#include <xc.h>

#include "vibeqc/build_config.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/basis.hpp"
#include "vibeqc/solvation_cpcm.hpp"
#include "vibeqc/basis_param_gradient.hpp"
#include "vibeqc/ao_eval.hpp"
#include "vibeqc/bloch.hpp"
#include "vibeqc/kmesh_address.hpp"
#include "vibeqc/casci.hpp"
#include "vibeqc/coop_cohp.hpp"
#include "vibeqc/correlation_conventions.hpp"
#include "vibeqc/casscf_gradient.hpp"
#include "vibeqc/ccsd_triples.hpp"
#include "vibeqc/dlpno_mp2.hpp"
#include "vibeqc/nevpt2_ops.hpp"
#include "vibeqc/pt2_transform.hpp"
#include "vibeqc/selected_ci.hpp"
#include "vibeqc/lobpcg.hpp"
#include "vibeqc/jd.hpp"
#include "vibeqc/gplhr.hpp"
#include "vibeqc/geomopt_kernels.hpp"
#include "vibeqc/crystal.hpp"
#include "vibeqc/dft_plus_u.hpp"
#include "vibeqc/dispersion.hpp"
#include "vibeqc/eeq_charges.hpp"
#include "vibeqc/semiempirical/dftb0.hpp"
#include "vibeqc/semiempirical/gradient.hpp"
#include "vibeqc/semiempirical/periodic_dftb0.hpp"
#include "vibeqc/semiempirical/periodic_scc_dftb.hpp"
#include "vibeqc/semiempirical/kpoints_dftb0.hpp"
#include "vibeqc/semiempirical/kpoints_scc_dftb.hpp"
#include "vibeqc/semiempirical/kpoints_dftb_fd_batch.hpp"
#include "vibeqc/semiempirical/native_facade.hpp"
#include "vibeqc/semiempirical/repulsive_spline.hpp"
#include "vibeqc/semiempirical/core/method_registry.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"
#include "vibeqc/semiempirical/core/parameters.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"
#include "vibeqc/semiempirical/methods/xtb/kpoints_gfn2.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/periodic_gfn2.hpp"
#include "vibeqc/semiempirical/methods/xtb/ugfn2.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/omx_parameters.hpp"
#include "vibeqc/semiempirical/methods/nddo/omx_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/periodic_pm6.hpp"
#include "vibeqc/semiempirical/methods/nddo/periodic_omx.hpp"
#include "vibeqc/semiempirical/methods/nddo/periodic_fd_batch.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/methods/indo/indo_engine.hpp"
#include "vibeqc/semiempirical/methods/indo/ccm_engine.hpp"
#include "vibeqc/semiempirical/methods/indo/msindo_gepol.hpp"
#include "vibeqc/semiempirical/methods/indo/msindo_gradient.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/scc_dftb.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/seccm/dftb0.hpp"
#include "vibeqc/semiempirical/seccm/scc_dftb.hpp"
#include "vibeqc/semiempirical/seccm/pm6.hpp"
#include "vibeqc/semiempirical/seccm/omx.hpp"
#include "vibeqc/semiempirical/seccm/gfn2.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"
#include "vibeqc/ecp.hpp"
#include "vibeqc/ewald.hpp"
#include "vibeqc/fft_poisson.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/multipole_moments_lattice.hpp"
#include "vibeqc/adjoined_pair_moments.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/periodic_bloch.hpp"
#include "vibeqc/periodic_fock.hpp"
#include "vibeqc/periodic_jk_builder.hpp"
#include "vibeqc/periodic_cosx_jk_builder.hpp"
#include "vibeqc/periodic_rhf.hpp"
#include "vibeqc/periodic_scf.hpp"
#include "vibeqc/periodic_xc.hpp"
#include "vibeqc/thread_pool.hpp"
#include "vibeqc/fock.hpp"
#include "vibeqc/gradient.hpp"
#include "vibeqc/periodic_gradient.hpp"
#include "vibeqc/hessian_integrals.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_cell_pair.hpp"
#include "vibeqc/cosx_one_center.hpp"
#include "vibeqc/cosx_robust.hpp"
#include "vibeqc/cosx_kernel.hpp"
#include "vibeqc/grid_batch.hpp"
#include "vibeqc/schwarz.hpp"
#include "vibeqc/bipole_dispatch.hpp"
#include "vibeqc/bipole_far_field_kernel.hpp"
#include "vibeqc/bipole_contractor.hpp"
#include "vibeqc/bipole_spherical_buffer.hpp"
#include "vibeqc/bipole_pair_moments.hpp"
#include "vibeqc/bipole_moment_derivatives.hpp"
#include "vibeqc/bipole_far_field_gradient.hpp"
#include "vibeqc/bipole_fock_kernel_builder.hpp"
#include "vibeqc/bipole_multipole.hpp"
#include "vibeqc/polipo_moments.hpp"
#include "vibeqc/df.hpp"
#include "vibeqc/aux_eri.hpp"
#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/brent.hpp"
#include "vibeqc/diis.hpp"
#include "vibeqc/kdiis.hpp"
#include "vibeqc/mp2.hpp"
#include "vibeqc/ccsd.hpp"
#include "vibeqc/uccsd.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/newton.hpp"
#include "vibeqc/rhf.hpp"
#include "vibeqc/rks.hpp"
#include "vibeqc/scf_convergence.hpp"
#include "vibeqc/uhf.hpp"
#include "vibeqc/uks.hpp"
#include "vibeqc/ump2.hpp"
#include "vibeqc/xc.hpp"
#include "vibeqc/xc_kernel.hpp"
#include "vibeqc/vv10.hpp"
#include "vibeqc/b97m_energy.hpp"
#include "vibeqc/diagnostics.hpp"

namespace py = pybind11;

// Element symbols 0..118 for the ergonomic Atom.symbol accessor (ASE
// parity: users reach for .symbol / .number, not just .Z). Index 0 is a
// dummy/ghost placeholder ("X").
namespace {
const char* const kElementSymbols[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
std::string element_symbol_for_z(int z) {
    constexpr int n = static_cast<int>(sizeof(kElementSymbols) /
                                       sizeof(kElementSymbols[0]));
    if (z < 0 || z >= n)
        throw std::invalid_argument(
            "Atom.symbol: atomic number " + std::to_string(z) +
            " is out of the tabulated range 0..118");
    return kElementSymbols[z];
}

template <typename OptionsT>
void set_molecular_scf_max_iter(OptionsT& options,
                                py::handle value,
                                const char* context) {
    if (py::isinstance<py::bool_>(value)) {
        throw py::value_error(
            std::string(context)
            + ": max_iter must be a non-negative integer");
    }
    try {
        options.max_iter = py::cast<int>(value);
    } catch (const py::cast_error&) {
        throw py::value_error(
            std::string(context)
            + ": max_iter must fit in a C++ int");
    }
}

// ---- C++ → Python diagnostics trampoline ---------------------------------

// Python callback object.  Stored once by ``_set_diagnostics_sink``;
// must not be mutated while a C++ computation is in flight.
py::object g_py_diag_callback;

/// Trampoline: called from ``vibeqc::diagnostic()`` / ``VIBEQC_DIAG``.
/// Acquires the GIL (the SCF loop typically releases it) and forwards
/// ``(tag, level_int, msg)`` to the registered Python callable.
void diag_trampoline(const char* tag, int level, const char* msg) {
    py::gil_scoped_acquire gil;
    if (g_py_diag_callback.is_none())
        return;
    try {
        g_py_diag_callback(tag, level, msg);
    } catch (py::error_already_set& e) {
        // A diagnostic callback failure must never abort the
        // computation — the output channel may be closed, the
        // Python callback may have been garbage-collected, etc.
        e.restore();
        PyErr_Clear();
    }
}

}  // namespace

#define VIBEQC_BINDINGS_HAS_PY_ALIAS 1
#include "symmetry_shared_bindings.cpp"
#include "periodic_ao_bloch_transport_bindings.cpp"
#include "periodic_orbital_sewing_bindings.cpp"
#include "bipole_ewald_gram_bindings.cpp"
#include "aiccm2026dev_b_kernels_bindings.cpp"
#include "periodic_correlation_admitted_reference_bindings.cpp"
#include "periodic_correlation_factor_build_admission_bindings.cpp"
#include "periodic_correlation_factor_stream_bindings.cpp"
#include "periodic_correlation_reciprocal_metric_bindings.cpp"
#include "periodic_correlation_metric_factorization_bindings.cpp"
#include "periodic_correlation_three_center_bindings.cpp"
#include "periodic_correlation_factor_store_bindings.cpp"
#include "periodic_correlation_wannier_bindings.cpp"
#include "periodic_correlation_diabatic_seed_bindings.cpp"
#include "periodic_correlation_bloch_iao_bindings.cpp"
#include "periodic_correlation_iao_pm_bindings.cpp"
#include "periodic_correlation_iao_optimizer_bindings.cpp"
#include "periodic_correlation_occupied_fock_bindings.cpp"
#include "periodic_correlation_pao_domain_bindings.cpp"
#include "periodic_correlation_pao_space_bindings.cpp"
#include "periodic_correlation_real_pao_space_bindings.cpp"
#include "periodic_correlation_real_pao_embedding_bindings.cpp"
#include "periodic_correlation_occupied_pao_domain_bindings.cpp"
#include "periodic_correlation_pair_pao_domain_bindings.cpp"
#include "periodic_correlation_local_factors_bindings.cpp"
#include "periodic_correlation_density_factors_bindings.cpp"
#include "periodic_correlation_local_orbital_factors_bindings.cpp"
#include "periodic_correlation_real_local_basis_bindings.cpp"
#include "periodic_correlation_real_local_provider_bindings.cpp"
#include "periodic_gaussian_source_context_bindings.cpp"
#include "periodic_gaussian_reciprocal_source_bindings.cpp"
#include "periodic_gaussian_metric_bindings.cpp"
#include "periodic_gaussian_three_center_bindings.cpp"
#include "periodic_gaussian_bloch_iao_bindings.cpp"
#include "periodic_gaussian_localization_bindings.cpp"
#include "periodic_gaussian_occupied_pao_domain_bindings.cpp"
#include "periodic_gaussian_pair_pao_domain_bindings.cpp"
#include "periodic_gaussian_pair_domain_builder_bindings.cpp"
#include "periodic_gaussian_one_electron_bindings.cpp"
#include "periodic_gaussian_nuclear_bindings.cpp"
#include "periodic_gaussian_fock_bindings.cpp"
#include "bounded_periodic_rhf_solver_bindings.cpp"
#include "periodic_gaussian_rhf_bindings.cpp"
#include "periodic_gaussian_local_orbital_factors_bindings.cpp"
#include "periodic_gaussian_density_gram_bindings.cpp"
#include "periodic_gaussian_mixed_pair_factors_bindings.cpp"
#include "periodic_gaussian_pair_interaction_bindings.cpp"
#include "periodic_gaussian_pair_particle_hole_bindings.cpp"
#include "periodic_gaussian_pair_ccsd_physical_particle_hole_bindings.cpp"
#include "periodic_gaussian_real_local_provider_bindings.cpp"
#include "periodic_gaussian_selected_local_ccsd_t_bindings.cpp"
#include "periodic_gaussian_pair_pnos_bindings.cpp"
#include "periodic_gaussian_embedded_pair_pnos_bindings.cpp"
#include "periodic_gaussian_pair_space_bindings.cpp"
#include "periodic_gaussian_gram_pair_pnos_bindings.cpp"
#include "bounded_restricted_pair_mp2_solver_bindings.cpp"
#include "bounded_restricted_pair_ccsd_amplitudes_bindings.cpp"
#include "bounded_restricted_ccsd_energy_bindings.cpp"
#include "bounded_restricted_pair_ccsd_solver_bindings.cpp"
#include "bounded_restricted_pair_ccsd_interaction_bindings.cpp"
#include "bounded_restricted_pair_ccsd_particle_hole_bindings.cpp"
#include "periodic_gaussian_pair_mp2_bindings.cpp"
#include "periodic_gaussian_domain_pair_mp2_bindings.cpp"
#include "periodic_gaussian_pair_ccsd_bindings.cpp"
#include "periodic_gaussian_triple_spaces_bindings.cpp"
#include "periodic_gaussian_local_triples_bindings.cpp"
#include "periodic_correlation_pao_overlap_bindings.cpp"
#include "periodic_correlation_pair_integrals_bindings.cpp"
#include "periodic_correlation_pair_residual_bindings.cpp"
#include "bounded_restricted_ccsd_target_bindings.cpp"
#include "bounded_restricted_ccsd_solver_bindings.cpp"
#include "bounded_restricted_triples_target_bindings.cpp"
#include "bounded_restricted_triples_moments_bindings.cpp"
#include "bounded_restricted_triples_solver_bindings.cpp"
#include "bounded_restricted_local_triples_solver_bindings.cpp"
#include "bounded_restricted_local_triples_moments_bindings.cpp"
#include "bounded_restricted_triple_natural_orbitals_bindings.cpp"
#include "periodic_correlation_real_local_ccsd_t_bindings.cpp"
#include "periodic_aopair_fourier_bindings.cpp"
#include "periodic_aopair_cross_fourier_bindings.cpp"
#include "hermitian_jacobi_bindings.cpp"
#include "pair_natural_orbitals_bindings.cpp"
#include "periodic_correlation_pair_topology_bindings.cpp"
#include "periodic_correlation_resource_bindings.cpp"
#include "periodic_auxiliary_fourier_bindings.cpp"
#include "periodic_gdf_short_range_bindings.cpp"
#include "periodic_mean_field_state_bindings.cpp"
#include "periodic_rhf_state_capture_bindings.cpp"
#undef VIBEQC_BINDINGS_HAS_PY_ALIAS

// v0.12 R3 — pybind11 trampoline for ``vibeqc::JKBuilder``. Lets a
// Python subclass implement ``build_J`` / ``build_K`` and be passed to
// any C++ SCF entry point that consumes a ``const JKBuilder&`` (in
// particular ``run_rhf_scf_with_jk``). The GPW route uses this so the
// Python ``GpwJBuilder`` (FFT-Poisson Hartree-J on the smooth grid)
// can drive the production C++ SCF host — DIIS, dynamic damping,
// level-shift, Newton/SOSCF — without a dedicated C++ port of the
// GPW kernel. ``PYBIND11_OVERRIDE_PURE`` re-acquires the GIL inside
// the override; the SCF driver releases it for the C++ build_J / K
// closures, which is harmless here because the override macro
// handles GIL state itself.
class PyJKBuilder : public vibeqc::JKBuilder {
public:
    using MatrixPair = std::pair<Eigen::MatrixXd, Eigen::MatrixXd>;
    using vibeqc::JKBuilder::JKBuilder;

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd,
            vibeqc::JKBuilder,
            build_J,
            D);
    }

    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd,
            vibeqc::JKBuilder,
            build_K,
            D);
    }

    MatrixPair build_K_pair(const Eigen::MatrixXd& D_alpha,
                            const Eigen::MatrixXd& D_beta) const override {
        PYBIND11_OVERRIDE(
            MatrixPair,
            vibeqc::JKBuilder,
            build_K_pair,
            D_alpha, D_beta);
    }

    Eigen::MatrixXd build_K_erf(const Eigen::MatrixXd& D,
                                double omega) const override {
        PYBIND11_OVERRIDE(
            Eigen::MatrixXd,
            vibeqc::JKBuilder,
            build_K_erf,
            D, omega);
    }

    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                double alpha_hf) const override {
        PYBIND11_OVERRIDE(
            Eigen::MatrixXd,
            vibeqc::JKBuilder,
            build_g_rhf,
            D, alpha_hf);
    }

    void set_schwarz_threshold(double threshold) const override {
        PYBIND11_OVERRIDE(
            void,
            vibeqc::JKBuilder,
            set_schwarz_threshold,
            threshold);
    }

    bool uses_runtime_schwarz_threshold() const override {
        PYBIND11_OVERRIDE(
            bool,
            vibeqc::JKBuilder,
            uses_runtime_schwarz_threshold);
    }

    void reset_state() const override {
        PYBIND11_OVERRIDE(
            void,
            vibeqc::JKBuilder,
            reset_state);
    }

    void set_incremental(bool enabled) const override {
        PYBIND11_OVERRIDE(
            void,
            vibeqc::JKBuilder,
            set_incremental,
            enabled);
    }

    bool incremental_active() const override {
        PYBIND11_OVERRIDE(
            bool,
            vibeqc::JKBuilder,
            incremental_active);
    }
};

// Full-grid XC callback holder.  The native SCF drivers release the GIL, so
// the one provider invocation per XC build reacquires it here.  Inputs and
// outputs are deliberately plain dictionaries of NumPy/Eigen-compatible
// arrays: the Python side can stay lazy (including importing PyTorch only on
// first SKALA use) without introducing a native libtorch dependency.
class PythonExternalXCProvider final : public vibeqc::ExternalXCProvider {
public:
    explicit PythonExternalXCProvider(py::object callback)
        : callback_(std::move(callback)) {}

    py::object callback() const { return callback_; }

    vibeqc::ExternalXCEvaluation evaluate(
        const vibeqc::ExternalXCInput& input) override {
        // One provider instance may be shared by simultaneous SCF contexts.
        // TorchScript inference/autograd uses mutable runtime state, so keep
        // a single callback invocation in flight per registered instance.
        std::lock_guard<std::mutex> callback_lock(callback_mutex_);
        py::gil_scoped_acquire gil;
        const Eigen::Index n = input.points.rows();
        if (input.points.cols() != 3
            || input.grid_weights.size() != n
            || input.atomic_grid_weights.size() != n
            || static_cast<Eigen::Index>(input.atom_of_point.size()) != n
            || input.rho_alpha.size() != n || input.rho_beta.size() != n
            || input.grad_alpha.rows() != n || input.grad_alpha.cols() != 3
            || input.grad_beta.rows() != n || input.grad_beta.cols() != 3
            || input.tau_alpha.size() != n || input.tau_beta.size() != n) {
            throw std::invalid_argument(
                "external XC input has inconsistent full-grid shapes");
        }

        // Validate atom-major contiguity and derive the exact block lengths
        // expected by atom-nonlocal models.  Every atom must own one complete
        // non-empty block and blocks must occur in atom order.
        const Eigen::Index n_atoms = input.atom_coords.rows();
        if (input.atom_coords.cols() != 3
            || static_cast<Eigen::Index>(input.atomic_numbers.size())
                   != n_atoms) {
            throw std::invalid_argument(
                "external XC atom metadata has inconsistent shapes");
        }
        std::vector<std::int64_t> atomic_grid_sizes(
            static_cast<std::size_t>(n_atoms), 0);
        int previous = -1;
        for (Eigen::Index g = 0; g < n; ++g) {
            const int owner = input.atom_of_point[static_cast<std::size_t>(g)];
            if (owner < 0 || owner >= n_atoms) {
                throw std::invalid_argument(
                    "external XC atom_of_point index is out of range");
            }
            if (owner != previous) {
                if (owner != previous + 1) {
                    throw std::invalid_argument(
                        "external XC grid must contain contiguous atom-major "
                        "blocks in atom order");
                }
                previous = owner;
            }
            ++atomic_grid_sizes[static_cast<std::size_t>(owner)];
        }
        if (n_atoms > 0 && previous != n_atoms - 1) {
            throw std::invalid_argument(
                "external XC grid is missing one or more atom blocks");
        }

        py::dict args;
        args["functional"] = input.functional;
        args["grid_profile"] = input.grid_profile;
        args["points"] = input.points;
        args["grid_weights"] = input.grid_weights;
        args["atomic_grid_weights"] = input.atomic_grid_weights;
        args["atomic_grid_sizes"] = atomic_grid_sizes;
        args["atom_coords"] = input.atom_coords;
        args["atomic_numbers"] = input.atomic_numbers;
        args["periodic"] = input.periodic;
        args["periodic_dimension"] = input.periodic_dimension;
        args["lattice"] = input.lattice;
        args["rho_alpha"] = input.rho_alpha;
        args["rho_beta"] = input.rho_beta;
        args["grad_alpha"] = input.grad_alpha;
        args["grad_beta"] = input.grad_beta;
        args["tau_alpha"] = input.tau_alpha;
        args["tau_beta"] = input.tau_beta;

        py::object raw = callback_(std::move(args));
        if (!py::isinstance<py::dict>(raw)) {
            throw std::invalid_argument(
                "external XC callback must return a dict");
        }
        const py::dict result = raw.cast<py::dict>();
        auto required = [&](const char* key) -> py::handle {
            if (!result.contains(key)) {
                throw std::invalid_argument(
                    std::string("external XC callback omitted '") + key
                    + "'");
            }
            return result[key];
        };

        vibeqc::ExternalXCEvaluation out;
        out.energy = py::cast<double>(required("energy"));
        out.v_rho_alpha =
            py::cast<Eigen::VectorXd>(required("v_rho_alpha"));
        out.v_rho_beta =
            py::cast<Eigen::VectorXd>(required("v_rho_beta"));
        out.v_grad_alpha =
            py::cast<Eigen::MatrixX3d>(required("v_grad_alpha"));
        out.v_grad_beta =
            py::cast<Eigen::MatrixX3d>(required("v_grad_beta"));
        out.v_tau_alpha =
            py::cast<Eigen::VectorXd>(required("v_tau_alpha"));
        out.v_tau_beta =
            py::cast<Eigen::VectorXd>(required("v_tau_beta"));
        if (!std::isfinite(out.energy)
            || out.v_rho_alpha.size() != n || out.v_rho_beta.size() != n
            || out.v_grad_alpha.rows() != n
            || out.v_grad_alpha.cols() != 3
            || out.v_grad_beta.rows() != n
            || out.v_grad_beta.cols() != 3
            || out.v_tau_alpha.size() != n || out.v_tau_beta.size() != n
            || !out.v_rho_alpha.allFinite() || !out.v_rho_beta.allFinite()
            || !out.v_grad_alpha.allFinite() || !out.v_grad_beta.allFinite()
            || !out.v_tau_alpha.allFinite() || !out.v_tau_beta.allFinite()) {
            throw std::invalid_argument(
                "external XC callback returned a non-finite value or an "
                "array with the wrong full-grid shape");
        }
        return out;
    }

private:
    py::object callback_;
    std::mutex callback_mutex_;
};

namespace {

// Issue #342: a periodic k-route SCF that exhausts its iteration budget
// must be loud by default.  ``py::register_exception`` translates this to
// the Python exception ``SCFNonConvergenceError`` (a RuntimeError
// subclass, registered on the semiempirical submodule) so callers that
// screen on "did not raise" can never admit an unconverged record.
// Diagnostics callers opt back into the returned record with
// ``allow_unconverged=True``; the record then still carries
// ``converged=False``.
struct SCFNonConvergenceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string scf_nonconvergence_message(
    const char* route_label,
    int n_iter,
    double conv_tol_charge) {
    std::ostringstream msg;
    msg << route_label << " SCF did not converge after " << n_iter
        << " iterations (conv_tol_charge=" << conv_tol_charge
        << "); refusing to return the energy as a successful result. "
           "Pass allow_unconverged=True to receive the unconverged "
           "result for diagnostics (issue #342).";
    return msg.str();
}

template <typename Parameters>
void validate_parameter_snapshot_type(const Parameters&) {}

void validate_parameter_snapshot_type(
    const vibeqc::semiempirical::nddo::PM6ParameterSet& parameters) {
    // OMx derives from PM6ParameterSet to reuse basis machinery.  Copying it
    // through a PM6-typed binding would slice away the OMx registry and could
    // turn the synchronized base projection into a falsely labelled PM6 set.
    if (parameters.method_name() != "pm6") {
        throw std::invalid_argument(
            "PM6 drivers require PM6 parameters; PM6 driver requires PM6 "
            "parameters, not " + parameters.method_name());
    }
}

template <typename Parameters, typename Callback>
auto run_with_parameter_snapshot(
    const Parameters& parameters,
    Callback&& callback) {
    // This copy and its validation happen while the Python GIL is still held.
    // Native work then sees one immutable value object even if another Python
    // thread mutates the caller-owned builder after the GIL is released.
    validate_parameter_snapshot_type(parameters);
    const Parameters snapshot(parameters);
    (void)snapshot.content_sha256();
    py::gil_scoped_release release;
    return callback(snapshot);
}

template <typename Parameters, typename Callback>
auto run_with_identified_parameter_snapshot(
    const Parameters& parameters,
    Callback&& callback) {
    validate_parameter_snapshot_type(parameters);
    const Parameters snapshot(parameters);
    const std::string sha256 = snapshot.content_sha256();
    const std::string identity = snapshot.parameter_identity();
    auto result = [&]() {
        py::gil_scoped_release release;
        return callback(snapshot);
    }();
    result.parameter_sha256 = sha256;
    result.parameter_identity = identity;
    return result;
}

template <typename Result, typename Parameters>
void require_matching_parameter_snapshot(
    const Result& result,
    const Parameters& snapshot,
    const char* operation) {
    const std::string sha256 = snapshot.content_sha256();
    if (result.parameter_sha256 != sha256) {
        throw std::invalid_argument(
            std::string(operation)
            + " requires the same immutable parameter snapshot as the "
              "energy result");
    }
}

template <typename Parameters, typename Result, typename Callback>
auto run_with_matching_parameter_snapshot(
    const Parameters& parameters,
    const Result& result,
    const char* operation,
    Callback&& callback) {
    validate_parameter_snapshot_type(parameters);
    const Parameters snapshot(parameters);
    require_matching_parameter_snapshot(result, snapshot, operation);
    py::gil_scoped_release release;
    return callback(snapshot);
}

bool is_single_gamma_gfn2_mesh(const vibeqc::BlochKMesh& kmesh) {
    if (kmesh.kpoints.size() != 1) return false;
    const Eigen::Vector3d& kpoint = kmesh.kpoints.front();
    if (!kpoint.allFinite() || !kpoint.isZero(0.0)) return false;
    return kmesh.weights.empty()
        || (kmesh.weights.size() == 1
            && std::isfinite(kmesh.weights.front())
            && kmesh.weights.front() == 1.0);
}

Eigen::Index expected_gfn2_shell_charge_count(
    const vibeqc::Molecule& mol,
    const vibeqc::semiempirical::xtb::GFN2ParameterSet& params) {
    Eigen::Index count = 0;
    for (const auto& atom : mol.atoms()) {
        const auto* element = params.element_data(atom.Z);
        if (element == nullptr) {
            return -1;
        }
        for (const auto& shell : element->shells) {
            if (shell.zeta > 0.0) {
                ++count;
            }
        }
    }
    return count;
}

bool gfn2_seccm_records_are_trivial_molecular(
    const std::vector<int>& central,
    const std::vector<int>& origin,
    const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
    const std::vector<double>& weights,
    const std::vector<int>& multiplicities,
    const Eigen::Ref<const Eigen::MatrixXd>& displacements) {
    const std::size_t count = central.size();
    if (origin.size() != count || weights.size() != count
        || multiplicities.size() != count
        || shell_labels.rows() != static_cast<Eigen::Index>(count)
        || shell_labels.cols() != 3
        || displacements.rows() != static_cast<Eigen::Index>(count)
        || displacements.cols() != 3) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const Eigen::Index row = static_cast<Eigen::Index>(index);
        if (shell_labels(row, 0) != 0 || shell_labels(row, 1) != 0
            || shell_labels(row, 2) != 0
            || !std::isfinite(weights[index])
            || std::abs(weights[index] - 1.0) > 1.0e-12) {
            return false;
        }
    }
    return true;
}

// Shared SECCM record-array reconstruction: Python passes the validated
// frozen WS records as flat arrays (see python/vibeqc/semiempirical/seccm/),
// and every SECCM adapter entry point rebuilds the same WSTopology from
// them.
vibeqc::semiempirical::seccm::WSTopology
seccm_topology_from_records(
    const vibeqc::Molecule& mol,
    const Eigen::Ref<const Eigen::MatrixXd>& translations,
    const std::vector<int>& central,
    const std::vector<int>& origin,
    const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
    const std::vector<double>& weights,
    const std::vector<int>& multiplicities,
    const Eigen::Ref<const Eigen::MatrixXd>& displacements) {
    const std::size_t count = central.size();
    if (translations.cols() != 3 || translations.rows() < 1
        || translations.rows() > 3) {
        throw std::invalid_argument(
            "SECCM translations must have shape (D, 3) with D in 1..3");
    }
    if (origin.size() != count || weights.size() != count
        || multiplicities.size() != count
        || shell_labels.rows() != static_cast<Eigen::Index>(count)
        || shell_labels.cols() != 3
        || displacements.rows() != static_cast<Eigen::Index>(count)
        || displacements.cols() != 3) {
        throw std::invalid_argument(
            "SECCM record arrays must have matching lengths");
    }
    vibeqc::semiempirical::seccm::WSTopology topology;
    topology.cells.resize(mol.atoms().size());
    for (Eigen::Index row = 0; row < translations.rows(); ++row) {
        topology.translations.push_back(
            translations.row(row).transpose());
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (central[index] < 0
            || central[index] >= static_cast<int>(topology.cells.size())) {
            throw std::invalid_argument(
                "SECCM record has an out-of-range central atom");
        }
        topology.cells[static_cast<std::size_t>(central[index])].push_back({
            origin[index],
            weights[index],
            displacements.row(static_cast<Eigen::Index>(index)).transpose(),
            {
                shell_labels(static_cast<Eigen::Index>(index), 0),
                shell_labels(static_cast<Eigen::Index>(index), 1),
                shell_labels(static_cast<Eigen::Index>(index), 2),
            },
            multiplicities[index],
        });
    }
    return topology;
}

// Structural validation for the declared finite group. A coherent choice of
// primitive vectors and replica counts is part of the model definition; in a
// defect-safe group-only topology it cannot be inferred uniquely from atoms.
// The public Python topology owns that declaration, while this private seam
// rejects malformed evidence and derives normalization without a free scalar.
struct ValidatedSECCMTopology {
    vibeqc::semiempirical::seccm::WSTopology topology;
    int group_order = 0;
};

ValidatedSECCMTopology seccm_validated_topology_from_records(
    const vibeqc::Molecule& mol,
    const Eigen::Ref<const Eigen::MatrixXd>& translations,
    const std::vector<int>& central,
    const std::vector<int>& origin,
    const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
    const std::vector<double>& weights,
    const std::vector<int>& multiplicities,
    const Eigen::Ref<const Eigen::MatrixXd>& displacements,
    const Eigen::Ref<const Eigen::MatrixXd>& primitive_vectors,
    const std::vector<int>& replicas,
    double geometry_tolerance,
    const std::string& route_name) {
    auto topology = seccm_topology_from_records(
        mol,
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements);
    const Eigen::Index dimensionality = translations.rows();
    if (!std::isfinite(geometry_tolerance) || geometry_tolerance <= 0.0) {
        throw std::invalid_argument(
            route_name
            + " finite-group geometry tolerance must be positive and finite");
    }
    if (primitive_vectors.rows() != dimensionality
        || primitive_vectors.cols() != 3) {
        throw std::invalid_argument(
            route_name
            + " primitive vectors must have shape (D, 3) matching the "
              "cyclic translations");
    }
    if (!primitive_vectors.allFinite()) {
        throw std::invalid_argument(
            route_name + " primitive vectors must be finite");
    }
    if (replicas.size() != 3) {
        throw std::invalid_argument(
            route_name + " replicas must contain exactly three integers");
    }
    for (int replica : replicas) {
        if (replica <= 0) {
            throw std::invalid_argument(
                route_name + " replicas must be positive");
        }
    }
    for (std::size_t axis = static_cast<std::size_t>(dimensionality);
         axis < replicas.size(); ++axis) {
        if (replicas[axis] != 1) {
            throw std::invalid_argument(
                route_name + " inactive replicas must equal one");
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        primitive_vectors, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular_values = svd.singularValues();
    const double vector_scale = singular_values[0];
    const double rank_threshold = std::max(
        geometry_tolerance,
        std::numeric_limits<double>::epsilon()
            * static_cast<double>(
                std::max(primitive_vectors.rows(), primitive_vectors.cols()))
            * vector_scale);
    if (singular_values[singular_values.size() - 1] <= rank_threshold) {
        throw std::invalid_argument(
            route_name
            + " primitive vectors are linearly dependent within the "
              "declared tolerance");
    }

    int group_order = 1;
    for (int replica : replicas) {
        if (replica > std::numeric_limits<int>::max() / group_order) {
            throw std::overflow_error(
                route_name + " finite-group order overflows int");
        }
        group_order *= replica;
    }
    for (Eigen::Index axis = 0; axis < dimensionality; ++axis) {
        const auto translation =
            topology.translations[static_cast<std::size_t>(axis)];
        if (!translation.allFinite()) {
            throw std::invalid_argument(
                route_name + " cyclic translations must be finite");
        }
        const Eigen::Vector3d expected =
            static_cast<double>(replicas[static_cast<std::size_t>(axis)])
            * primitive_vectors.row(axis).transpose();
        if (!expected.allFinite()
            || (translation - expected).cwiseAbs().maxCoeff()
                > geometry_tolerance) {
            throw std::invalid_argument(
                route_name
                + " cyclic translations do not equal replica-scaled "
                  "primitive vectors");
        }
    }
    return {std::move(topology), group_order};
}

}  // namespace

PYBIND11_MODULE(_vibeqc_core, m) {
    m.doc() = "vibeqc native core (C++ backend)";
    bind_aiccm2026dev_b_kernels(m);
    bind_periodic_correlation_resource(m);
    bind_periodic_mean_field_state(m);
    bind_symmetry_shared(m);
    bind_periodic_ao_bloch_transport(m);
    bind_periodic_orbital_sewing(m);
    bind_bipole_ewald_gram(m);
    bind_bipole_erfc_panel(m);
    bind_bipole_erfc_bloch(m);
    bind_periodic_correlation_admitted_reference(m);
    bind_periodic_correlation_factor_stream(m);
    bind_periodic_correlation_factor_build_admission(m);
    bind_periodic_auxiliary_fourier(m);
    bind_periodic_gdf_short_range(m);
    bind_periodic_correlation_reciprocal_metric(m);
    bind_periodic_correlation_metric_factorization(m);
    bind_periodic_correlation_three_center(m);
    bind_periodic_correlation_factor_store(m);
    bind_periodic_correlation_wannier(m);
    bind_periodic_correlation_diabatic_seed(m);
    bind_periodic_correlation_bloch_iao(m);
    bind_periodic_correlation_iao_pm(m);
    bind_periodic_correlation_iao_optimizer(m);
    bind_periodic_correlation_occupied_fock(m);
    bind_periodic_correlation_pao_domain(m);
    bind_periodic_correlation_pao_space(m);
    bind_periodic_correlation_real_pao_space(m);
    bind_periodic_correlation_real_pao_embedding(m);
    bind_periodic_correlation_occupied_pao_domain(m);
    bind_periodic_correlation_pair_pao_domain(m);
    bind_periodic_correlation_local_factors(m);
    bind_periodic_correlation_density_factors(m);
    bind_periodic_correlation_local_orbital_factors(m);
    bind_periodic_correlation_real_local_basis(m);
    bind_periodic_correlation_real_local_provider(m);
    bind_periodic_gaussian_source_context(m);
    bind_periodic_gaussian_reciprocal_source(m);
    bind_periodic_gaussian_metric(m);
    bind_periodic_gaussian_three_center(m);
    bind_periodic_gaussian_bloch_iao(m);
    bind_periodic_gaussian_localization(m);
    bind_periodic_gaussian_occupied_pao_domain(m);
    bind_periodic_gaussian_pair_pao_domain(m);
    bind_periodic_gaussian_pair_domain_builder(m);
    bind_periodic_gaussian_one_electron(m);
    bind_periodic_gaussian_nuclear(m);
    bind_periodic_gaussian_fock(m);
    bind_bounded_periodic_rhf_solver(m);
    bind_periodic_gaussian_rhf(m);
    bind_periodic_gaussian_local_orbital_factors(m);
    bind_periodic_gaussian_density_gram(m);
    bind_periodic_gaussian_mixed_pair_factors(m);
    bind_periodic_gaussian_pair_interaction(m);
    bind_periodic_gaussian_pair_particle_hole(m);
    bind_periodic_gaussian_pair_ccsd_physical_particle_hole(m);
    bind_periodic_gaussian_real_local_provider(m);
    bind_periodic_gaussian_selected_local_ccsd_t(m);
    bind_periodic_gaussian_pair_pnos(m);
    bind_periodic_gaussian_embedded_pair_pnos(m);
    bind_periodic_gaussian_pair_space(m);
    bind_periodic_gaussian_gram_pair_pnos(m);
    bind_bounded_restricted_pair_mp2_solver(m);
    bind_bounded_restricted_pair_ccsd_amplitudes(m);
    bind_bounded_restricted_ccsd_energy(m);
    bind_bounded_restricted_pair_ccsd_solver(m);
    bind_bounded_restricted_pair_ccsd_interaction(m);
    bind_bounded_restricted_pair_ccsd_particle_hole(m);
    bind_periodic_gaussian_pair_mp2(m);
    bind_periodic_gaussian_domain_pair_mp2(m);
    bind_periodic_gaussian_pair_ccsd(m);
    bind_periodic_gaussian_triple_spaces(m);
    bind_periodic_gaussian_local_triples(m);
    bind_periodic_correlation_pao_overlap(m);
    bind_periodic_correlation_pair_integrals(m);
    bind_periodic_correlation_pair_residual(m);
    bind_bounded_restricted_ccsd_target(m);
    bind_bounded_restricted_ccsd_solver(m);
    bind_bounded_restricted_triples_target(m);
    bind_bounded_restricted_triples_moments(m);
    bind_bounded_restricted_triples_solver(m);
    bind_bounded_restricted_local_triples_solver(m);
    bind_bounded_restricted_local_triples_moments(m);
    bind_bounded_restricted_triple_natural_orbitals(m);
    bind_periodic_correlation_real_local_ccsd_t(m);
    bind_periodic_aopair_fourier(m);
    bind_periodic_aopair_cross_fourier(m);
    bind_hermitian_jacobi(m);
    bind_pair_natural_orbitals(m);
    bind_periodic_correlation_pair_topology(m);
    bind_periodic_rhf_state_capture(m);

    // ----- Smoke / diagnostics -------------------------------------------
    m.def("hello",
          []() { return std::string{"vibeqc core alive"}; },
          "Smoke test: returns a static greeting.");

    // C++ diagnostics channel — gated, callback-driven, no direct I/O.
    // Docs: cpp/include/vibeqc/diagnostics.hpp
    m.def("_set_diagnostics_level",
          [](int level) {
              vibeqc::g_diag_level.store(level, std::memory_order_relaxed);
          },
          py::arg("level"),
          "Set the C++ diagnostics verbosity threshold (0=QUIET … "
          "3=DEBUG). Silences any C++ diagnostic with ``level > "
          "threshold``.");

    m.def("_set_diagnostics_sink",
          [](py::object callback) {
              g_py_diag_callback = callback;
              if (callback.is_none()) {
                  vibeqc::g_diag_sink.store(
                      nullptr, std::memory_order_relaxed);
              } else {
                  vibeqc::g_diag_sink.store(
                      diag_trampoline, std::memory_order_relaxed);
              }
          },
          py::arg("callback"),
          "Install (or clear) the Python callback that receives C++ "
          "diagnostics.  Pass ``None`` to disable.  Must be called "
          "before any parallel computation that may emit diagnostics.");

    m.def("_get_diagnostics_level",
          []() {
              return vibeqc::g_diag_level.load(std::memory_order_relaxed);
          },
          "Return the current C++ diagnostics verbosity threshold.");

    m.def("libint_version",
          []() { return libint2::libint_version_string(false); },
          "libint2 version string, e.g. \"2.13.1\".");

    m.def("cpcm_build_A_matrix",
          &vibeqc::build_cpcm_A_matrix,
          py::arg("cavity_points"), py::arg("cavity_weights"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the dense CPCM/COSMO cavity self-interaction matrix.");
    m.def("cpcm_build_capped_A_matrix",
          &vibeqc::build_cpcm_capped_A_matrix,
          py::arg("cavity_points"), py::arg("a_diag"),
          py::call_guard<py::gil_scoped_release>(),
          "Build a dense COSMO A matrix with capped off-diagonal interactions.");
    m.def("cpcm_dielectric_factor",
          &vibeqc::cpcm_dielectric_factor,
          py::arg("epsilon"), py::arg("variant") = std::string("cpcm"),
          "Return the CPCM/COSMO conductor-screening dielectric factor.");
    py::class_<vibeqc::CPCMSolveResult>(m, "CPCMSolveResult")
        .def_readonly("q", &vibeqc::CPCMSolveResult::q)
        .def_readonly("V", &vibeqc::CPCMSolveResult::V)
        .def_readonly("e_solv", &vibeqc::CPCMSolveResult::e_solv)
        .def_readonly("epsilon", &vibeqc::CPCMSolveResult::epsilon);
    m.def("cpcm_solve_apparent_charges",
          &vibeqc::solve_cpcm_apparent_charges,
          py::arg("A"), py::arg("V_at_cavity"), py::arg("epsilon"),
          py::arg("variant") = std::string("cpcm"),
          py::call_guard<py::gil_scoped_release>(),
          "Solve A q = -f(e) V for CPCM/COSMO apparent surface charges.");

    m.def("libxc_version",
          []() { return std::string{xc_version_string()}; },
          "libxc version string, e.g. \"7.0.0\".");

    m.def("blas_info",
          []() {
              py::dict info;
              info["libraries"] = std::string{
                  vibeqc::build_config::kBlasLibraries};
              info["blas_enabled"] = vibeqc::build_config::kBlasEnabled;
              info["lapacke_enabled"] =
                  vibeqc::build_config::kLapackeEnabled;
              return info;
          },
          "Build-time BLAS/LAPACK linkage info: dict with keys "
          "'libraries' (raw BLAS_LIBRARIES string from CMake's FindBLAS — "
          "e.g. '-framework Accelerate' on macOS, '/usr/lib/libblas.so' "
          "on Linux netlib, empty when no BLAS was linked), "
          "'blas_enabled' (True iff EIGEN_USE_BLAS was set), and "
          "'lapacke_enabled' (True iff EIGEN_USE_LAPACKE was set; "
          "controls whether Eigen's dense solvers delegate to LAPACKE).");

    m.def("_libecpint_gaussian_quadrature",
          &vibeqc::libecpint_gaussian_quadrature,
          py::arg("npoints"), py::arg("exponent"), py::arg("center"),
          py::arg("tolerance") = 1e-12,
          "Diagnostic: integral and convergence of a Gaussian on [-1,1].");
    m.def("libecpint_version",
          &vibeqc::libecpint_version,
          "libecpint version string. Phase 14 ECP integrals depend on "
          "this library; vibe-qc vendors v1.0.7 under "
          "third_party/libecpint/ alongside its pugixml + libcerf "
          "dependencies for byte-identical builds across machines.");

    m.def("fftw3_version",
          &vibeqc::fftw3_version,
          "FFTW3 version string, e.g. \"3.3.10-sse2-avx\". The "
          "FFT-Poisson long-range Hartree solver "
          "(cpp/src/fft_poisson.cpp) and the upcoming GAPW route "
          "both depend on FFTW3; surfaced through the banner per "
          "CLAUDE.md § 6.");

    py::class_<vibeqc::ECPCenter>(m, "ECPCenter",
        "Per-atom ECP placement: atomic number Z + Cartesian "
        "position (bohr).")
        .def(py::init<>())
        .def(py::init([](int Z, std::array<double, 3> xyz) {
            return vibeqc::ECPCenter{Z, xyz};
        }), py::arg("Z"), py::arg("xyz"))
        .def_readwrite("Z", &vibeqc::ECPCenter::Z)
        .def_readwrite("xyz", &vibeqc::ECPCenter::xyz);

    py::class_<vibeqc::ECPPrimitiveBlock>(m, "ECPPrimitiveBlock",
        "Inline ECP primitive data for one ECP centre (Phase 14g).")
        .def(py::init<>())
        .def_readwrite("n_primitive", &vibeqc::ECPPrimitiveBlock::n_primitive)
        .def_readwrite("exponents", &vibeqc::ECPPrimitiveBlock::exponents)
        .def_readwrite("coefficients", &vibeqc::ECPPrimitiveBlock::coefficients)
        .def_readwrite("ams", &vibeqc::ECPPrimitiveBlock::ams)
        .def_readwrite("ns", &vibeqc::ECPPrimitiveBlock::ns);

    m.def("compute_ecp_matrix_from_primitives",
          &vibeqc::compute_ecp_matrix_from_primitives,
          py::arg("basis"), py::arg("center_xyz"), py::arg("primitives"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute V_ECP matrix from inline primitive data (Phase 14g).");

    m.def("compute_ecp_lattice_from_primitives",
          &vibeqc::compute_ecp_lattice_from_primitives,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("home_ecp_centers"), py::arg("home_primitives"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed V_ECP(g) from inline primitives (Phase 14g).");

    m.def("ecp_lattice_gradient_contribution_from_primitives",
          &vibeqc::ecp_lattice_gradient_contribution_from_primitives,
          py::arg("basis"), py::arg("system"), py::arg("density"),
          py::arg("options"), py::arg("home_ecp_centers"),
          py::arg("home_primitives"),
          py::call_guard<py::gil_scoped_release>(),
          "Periodic ECP atomic gradient on the value builder's image domain.");

    m.def("compute_ecp_matrix", &vibeqc::compute_ecp_matrix,
          py::arg("basis"), py::arg("ecp_centers"),
          py::arg("library_name") = std::string("ecp10mdf"),
          py::arg("share_dir") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Compute the AO-basis ECP matrix V_ECP_{μν} = ⟨χ_μ|V_ECP|χ_ν⟩ "
          "via libecpint's built-in XML library (ecp10mdf, ecp28mdf, "
          "ecp46mdf, ecp60mdf, ecp78mdf, lanl2dz). Output is in "
          "spherical (real-solid-harmonic) basis. share_dir defaults "
          "to the vendored third_party/libecpint/install/share/libecpint.");

    m.def("get_num_threads",
          &vibeqc::omp_max_threads,
          "Maximum number of OpenMP threads the next parallel region "
          "will use.");

    m.def("set_num_threads",
          &vibeqc::set_num_threads,
          py::arg("n"),
          "Set the OpenMP thread count; ``n <= 0`` restores the "
          "OMP_NUM_THREADS-derived default. Returns the actual thread "
          "count that will be used.");

    // ----- Atom ----------------------------------------------------------
    py::class_<vibeqc::Atom>(m, "Atom")
        .def(py::init<>())
        .def(py::init([](int Z, std::array<double, 3> xyz) {
                 return vibeqc::Atom{Z, xyz};
             }),
             py::arg("Z"), py::arg("xyz"),
             "Create an Atom with atomic number Z and position in bohr.")
        .def_readwrite("Z", &vibeqc::Atom::Z)
        .def_readwrite("xyz", &vibeqc::Atom::xyz)
        // ASE-parity ergonomic aliases: .number is the atomic number
        // (== Z), .symbol the element symbol. Read-only; set .Z to change.
        .def_property_readonly(
            "number", [](const vibeqc::Atom& a) { return a.Z; },
            "Atomic number (alias for .Z; read-only).")
        .def_property_readonly(
            "symbol",
            [](const vibeqc::Atom& a) { return element_symbol_for_z(a.Z); },
            "Element symbol for this atom's atomic number (read-only).")
        .def("__repr__", [](const vibeqc::Atom& a) {
            return "Atom(Z=" + std::to_string(a.Z) + ", xyz=[" +
                   std::to_string(a.xyz[0]) + ", " +
                   std::to_string(a.xyz[1]) + ", " +
                   std::to_string(a.xyz[2]) + "])";
        });

    // ----- Molecule ------------------------------------------------------
    py::class_<vibeqc::Molecule>(m, "Molecule")
        .def(py::init<std::vector<vibeqc::Atom>, int, int>(),
             py::arg("atoms"),
             py::arg("charge") = 0,
             py::arg("multiplicity") = 1)
        .def_property_readonly("atoms", &vibeqc::Molecule::atoms)
        .def_property_readonly("charge", &vibeqc::Molecule::charge)
        .def_property_readonly("multiplicity", &vibeqc::Molecule::multiplicity)
        .def("n_electrons", &vibeqc::Molecule::n_electrons)
        .def("nuclear_repulsion", &vibeqc::Molecule::nuclear_repulsion,
             "Point-charge nuclear repulsion energy in Hartree.");

    m.def(
        "_published_frozen_core_orbitals_for_atomic_number",
        &vibeqc::published_frozen_core_orbitals_for_atomic_number,
        py::arg("atomic_number"),
        "Private lookup for the published frozen-core spatial-orbital count "
        "(ORCA 6.1 Table 2.69), extended by zero for a Z=0 ghost centre. "
        "This is a count-only convention; it does not implement ORCA's "
        "orbital-character reorder.");

    // ----- BasisSet ------------------------------------------------------
    py::class_<vibeqc::ShellInfo>(m, "ShellInfo",
        "One contracted Gaussian shell: angular momentum, primitive "
        "exponents, contraction coefficients, and the atom it sits on. "
        "Produced by BasisSet.shells(); also constructible from Python so "
        "callers can build modrho-renormalised aux bases or any other "
        "custom-coefficient shell list and feed it to "
        "BasisSet(molecule, shells, name, coefficients_pre_normalized).")
        .def(py::init<>())
        .def(py::init([](int atom_index, int l, bool pure,
                         std::vector<double> exponents,
                         std::vector<double> coefficients,
                         std::array<double, 3> origin) {
                 vibeqc::ShellInfo info;
                 info.atom_index = atom_index;
                 info.l = l;
                 info.pure = pure;
                 info.exponents = std::move(exponents);
                 info.coefficients = std::move(coefficients);
                 info.origin = origin;
                 return info;
             }),
             py::arg("atom_index"), py::arg("l"), py::arg("pure"),
             py::arg("exponents"), py::arg("coefficients"),
             py::arg("origin"),
             "Construct a ShellInfo from explicit per-shell metadata. "
             "Field semantics match the read-only properties; see "
             "BasisSet(molecule, shells, ...) for how this is consumed.")
        .def_readwrite("atom_index", &vibeqc::ShellInfo::atom_index,
                      "Index of the owning atom in Molecule.atoms() (0-based).")
        .def_readwrite("l", &vibeqc::ShellInfo::l,
                      "Angular momentum: 0=s, 1=p, 2=d, 3=f, ...")
        .def_readwrite("pure", &vibeqc::ShellInfo::pure,
                      "True: spherical harmonics (2L+1 AOs). "
                      "False: Cartesian ((L+1)(L+2)/2 AOs).")
        .def_readwrite("exponents", &vibeqc::ShellInfo::exponents,
                      "Primitive Gaussian exponents α_i (bohr^-2).")
        .def_readwrite("coefficients", &vibeqc::ShellInfo::coefficients,
                      "Contraction coefficients c_i (same length as "
                      "exponents). When read from BasisSet.shells() these "
                      "include libint's primitive normalisation; pass "
                      "coefficients_pre_normalized=True (default) when "
                      "constructing a derived BasisSet from a modified "
                      "shell list to keep them as-is.")
        .def_readwrite("origin", &vibeqc::ShellInfo::origin,
                      "Cartesian origin (bohr).")
        .def("__repr__", [](const vibeqc::ShellInfo& s) {
            const char lch[] = "spdfghikl";
            char lc = (s.l >= 0 && s.l < 9) ? lch[s.l] : '?';
            return std::string{"ShellInfo("} + lc
                 + ", atom=" + std::to_string(s.atom_index)
                 + ", n_prim=" + std::to_string(s.exponents.size())
                 + (s.pure ? ", pure" : ", cart") + ")";
        });

    py::class_<vibeqc::BasisSet>(m, "BasisSet")
        .def(py::init<const vibeqc::Molecule&, const std::string&, bool>(),
             py::arg("molecule"),
             py::arg("name"),
             py::arg("require_all_atoms") = true,
             "Load a bundled basis by name for every atom of the molecule. "
             "require_all_atoms=True (default) refuses a file that has no "
             "block for one of the molecule's elements instead of letting "
             "that atom enter with zero basis functions; pass False only "
             "for coverage diagnostics (density_fitting.aux_basis_coverage_gaps).")
        // Build from an explicit ShellInfo list. coefficients_pre_normalized
        // defaults to True (round-trip from .shells() preserves them as-is);
        // set False when feeding shells parsed from a Gaussian-style file
        // format that needs libint to embed primitive normalisation.
        .def(py::init<const vibeqc::Molecule&,
                      const std::vector<vibeqc::ShellInfo>&,
                      const std::string&,
                      bool>(),
             py::arg("molecule"),
             py::arg("shells"),
             py::arg("name") = std::string("<custom>"),
             py::arg("coefficients_pre_normalized") = true,
             "Build a BasisSet from explicit per-shell metadata. "
             "Each ShellInfo contributes one libint shell at the supplied "
             "origin with the given primitives + contraction "
             "coefficients. Use this to construct modrho-renormalised aux "
             "bases or any other custom-coefficient basis without "
             "round-tripping through a .g94 file. "
             "coefficients_pre_normalized=True (default) takes the "
             "coefficients as-given (the natural form for round-tripping "
             "shells extracted via .shells()); set False to have libint "
             "embed primitive normalisation factors (the Gaussian / .g94 "
             "convention).")
        .def_property_readonly("name", &vibeqc::BasisSet::name)
        .def_property_readonly("nbasis", &vibeqc::BasisSet::nbasis,
             "Number of basis functions (n_bf) — sum over shells of "
             "(2L+1) for spherical / (L+1)(L+2)/2 for Cartesian. "
             "Property (no parens). Was a method until v0.4 polish.")
        .def_property_readonly("nshells", &vibeqc::BasisSet::nshells,
             "Number of contracted shells. Property (no parens). Was "
             "a method until v0.4 polish.")
        .def("shells", &vibeqc::BasisSet::shells,
             "List of ShellInfo structs — one per (shell × contraction) "
             "pair, in libint's native shell order. Use this to export "
             "the basis to external formats (molden, NWChem, ...). "
             "Method, not property — returns a freshly-built list.");

    m.def("prune_tight_primitives", &vibeqc::prune_tight_primitives,
          py::arg("basis"), py::arg("molecule"), py::arg("cutoff"),
          "Build a softened BasisSet by dropping tight primitives "
          "(exponent > cutoff) from each shell of ``basis``. Native fast "
          "path for vibeqc.periodic_gapw_augment.softened_basis, kept "
          "identical to its pure-Python fallback: a primitive survives iff "
          "its exponent <= cutoff; a shell left with no surviving "
          "primitives is dropped entirely; surviving contraction "
          "coefficients are taken as-is (pre-normalised, no renorm); an "
          "all-pruned basis is returned unchanged.");

    // ----- Integrals -----------------------------------------------------
    // Eigen matrices are returned; pybind11 delivers them as NumPy arrays.
    // Release the GIL for the duration of the native compute — Python isn't
    // doing anything during integral evaluation.
    m.def("compute_overlap", &vibeqc::compute_overlap,
          py::arg("basis"),
          py::call_guard<py::gil_scoped_release>(),
          "AO overlap matrix S_{mu nu} = <mu|nu>.");

    m.def("overlap_exponent_derivative", &vibeqc::overlap_exponent_derivative,
          py::arg("basis"), py::arg("shell_idx"), py::arg("prim_idx"),
          py::call_guard<py::gil_scoped_release>(),
          "Analytic dS/dalpha: derivative of the AO overlap matrix w.r.t. the "
          "prim_idx-th primitive exponent of contracted shell shell_idx "
          "(Phase-1 basis-optimisation energy gradient).");

    m.def("kinetic_exponent_derivative", &vibeqc::kinetic_exponent_derivative,
          py::arg("basis"), py::arg("shell_idx"), py::arg("prim_idx"),
          py::call_guard<py::gil_scoped_release>(),
          "Analytic dT/dalpha: derivative of the AO kinetic-energy matrix "
          "w.r.t. the prim_idx-th primitive exponent of shell shell_idx.");

    m.def("nuclear_exponent_derivative", &vibeqc::nuclear_exponent_derivative,
          py::arg("basis"), py::arg("molecule"), py::arg("shell_idx"),
          py::arg("prim_idx"),
          py::call_guard<py::gil_scoped_release>(),
          "Analytic dV/dalpha: derivative of the AO nuclear-attraction matrix "
          "w.r.t. the prim_idx-th primitive exponent of shell shell_idx.");

    m.def("eri_exponent_derivative", &vibeqc::eri_exponent_derivative,
          py::arg("basis"), py::arg("shell_idx"), py::arg("prim_idx"),
          py::call_guard<py::gil_scoped_release>(),
          "Analytic d(mu nu|lam sig)/dalpha w.r.t. the prim_idx-th primitive "
          "exponent of shell shell_idx. Returns the flat (nbf^4, row-major) "
          "tensor; reshape to (nbf,nbf,nbf,nbf).");

    m.def("compute_overlap_two_basis", &vibeqc::compute_overlap_two_basis,
          py::arg("basis1"), py::arg("basis2"),
          py::call_guard<py::gil_scoped_release>(),
          "Cross-basis AO overlap S_{mu nu} = <mu^(1)|nu^(2)> between two "
          "basis sets (possibly different contraction and/or geometry). "
          "Shape (basis1.nbasis, basis2.nbasis); not symmetric. Used by the "
          "READ initial guess to project a prior density onto a new basis.");

    m.def("compute_kinetic", &vibeqc::compute_kinetic,
          py::arg("basis"),
          py::call_guard<py::gil_scoped_release>(),
          "AO kinetic-energy matrix T_{mu nu}.");

    m.def("compute_nuclear", &vibeqc::compute_nuclear,
          py::arg("basis"), py::arg("molecule"),
          py::call_guard<py::gil_scoped_release>(),
          "AO nuclear-attraction matrix V_{mu nu}.");

    // compute_nuclear_with_charges — nuclear-attraction matrix with
    // caller-supplied per-atom charges (e.g. Z_eff = Z − n_core for
    // ECP-aware solvation calculations).
    m.def("compute_nuclear_with_charges",
          &vibeqc::compute_nuclear_with_charges,
          py::arg("basis"), py::arg("positions"), py::arg("charges"),
          py::call_guard<py::gil_scoped_release>(),
          "AO nuclear-attraction matrix V_{mu nu} = Σ_A −q_A · "
          "⟨χ_μ|1/|r − R_A||χ_ν⟩ with caller-supplied charges q_A.");

    // ecp_core_electrons — lookup n_core per Z from a libecpint XML
    // library. Returns a dict {Z: n_core} for the given list of
    // atomic numbers. Used by the CPCM solvation driver to compute
    // effective nuclear charges Z_eff = Z − n_core for the cavity
    // electrostatic potential.
    m.def("ecp_core_electrons", &vibeqc::ecp_core_electrons,
          py::arg("charges"), py::arg("library_name"),
          py::arg("share_dir") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Look up the number of core electrons replaced by each ECP "
          "atom in a given libecpint XML library. Returns dict "
          "{Z: n_core} covering every charge in ``charges`` that "
          "appears in the library.");

    py::class_<vibeqc::DipoleIntegrals>(m, "DipoleIntegrals",
        "x, y, z components of the dipole matrix <μ|r_c − O_c|ν>.")
        .def_readonly("x", &vibeqc::DipoleIntegrals::x)
        .def_readonly("y", &vibeqc::DipoleIntegrals::y)
        .def_readonly("z", &vibeqc::DipoleIntegrals::z);

    m.def("compute_dipole", &vibeqc::compute_dipole,
          py::arg("basis"),
          py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
          py::call_guard<py::gil_scoped_release>(),
          "Dipole-integral matrices <μ|r − O|ν> about ``origin`` (bohr). "
          "Used to compute dipole moments via "
          "<r> = -tr(P·M) + Σ Z_A (R_A − O).");

    // Compute the dense 4-index ERI tensor. We hand ownership of the
    // std::vector buffer to NumPy via a capsule so there's no copy on the
    // way back to Python (important as nbasis grows).
    m.def("compute_eri",
          [](const vibeqc::BasisSet& basis) {
              vibeqc::Eri4D eri;
              {
                  py::gil_scoped_release unlock_gil;
                  eri = vibeqc::compute_eri(basis);
              }
              const auto n = static_cast<py::ssize_t>(eri.n);
              // Move the vector onto the heap so the capsule can outlive this
              // scope and own it through NumPy's lifetime.
              auto* owner = new std::vector<double>(std::move(eri.data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<double>*>(p);
              });
              return py::array_t<double>(
                  {n, n, n, n},
                  {static_cast<py::ssize_t>(n * n * n * sizeof(double)),
                   static_cast<py::ssize_t>(n * n * sizeof(double)),
                   static_cast<py::ssize_t>(n * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("basis"),
          "AO electron-repulsion integral tensor (mu nu | lambda sigma), "
          "chemists' notation. Shape (n, n, n, n) with n = nbasis.");

    // AO -> MO four-index ERI transformation for TD-DFT.
    // 4-step half-transform in row-major (matching numpy C layout).
    m.def("eri_ao_to_mo_blocks",
          [](const py::array_t<double, py::array::c_style>& eri,
             const Eigen::MatrixXd& C,
             int n_occ) -> py::tuple {
              const auto n = static_cast<Eigen::Index>(eri.shape(0));
              const Eigen::Index n_virt = n - n_occ;
              if (n_occ <= 0 || n_virt <= 0) {
                  throw std::invalid_argument(
                      "eri_ao_to_mo_blocks: n_occ must be in (0, n_basis)");
              }
              using RM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>;
              using RMap = Eigen::Map<const RM>;

              // Convert MO coefficients to row-major for consistent layout.
              RM Co_rm = C.leftCols(n_occ);   // (n, n_occ) row-major
              RM Cv_rm = C.rightCols(n_virt); // (n, n_virt) row-major

              const Eigen::Index n3 = n * n * n;

              // Step 1: mu -> i  (shared)
              //   h1(i, nu*lam*sig) = Σ_mu C_occ(mu,i) * eri(mu, nu*lam*sig)
              RMap eri2d(eri.data(), n, n3);
              RM h1 = Co_rm.transpose() * eri2d;  // (n_occ, n³) RM

              // ---- ovov (ia|jb): steps 2a-4a ---------
              // Step 2a: nu -> a  — for each i, Cv^T * h1_i
              RM h2_ovov(n_occ * n_virt, n * n);
              for (Eigen::Index i = 0; i < n_occ; ++i) {
                  RMap h1_i(h1.data() + i * n3, n, n * n);
                  h2_ovov.middleRows(i * n_virt, n_virt) =
                      Cv_rm.transpose() * h1_i;
              }
              // Step 3a: lam -> j  — for each (i,a), Co^T * h2_ia
              RM h3_ovov(n_occ * n_virt * n_occ, n);
              for (Eigen::Index i = 0; i < n_occ; ++i) {
                  for (Eigen::Index a = 0; a < n_virt; ++a) {
                      RMap h2_ia(
                          h2_ovov.data() + (i * n_virt + a) * (n * n), n, n);
                      h3_ovov.middleRows((i * n_virt + a) * n_occ, n_occ) =
                          Co_rm.transpose() * h2_ia;
                  }
              }
              // Step 4a: sig -> b  — for each (i,a,j), rowvec * Cv
              RM ovov(n_occ * n_virt * n_occ, n_virt);
              for (Eigen::Index i = 0; i < n_occ; ++i) {
                  for (Eigen::Index a = 0; a < n_virt; ++a) {
                      const Eigen::Index ia_row_off = (i * n_virt + a) * n_occ;
                      for (Eigen::Index j = 0; j < n_occ; ++j) {
                          ovov.row(ia_row_off + j) =
                              h3_ovov.row(ia_row_off + j) * Cv_rm;
                      }
                  }
              }

              // ---- oovv (ij|ab): steps 2b-4b ---------
              // Step 2b: nu -> j  — for each i, Co^T * h1_i
              RM h2_oovv(n_occ * n_occ, n * n);
              for (Eigen::Index i = 0; i < n_occ; ++i) {
                  RMap h1_i(h1.data() + i * n3, n, n * n);
                  h2_oovv.middleRows(i * n_occ, n_occ) =
                      Co_rm.transpose() * h1_i;
              }
              // Step 3b: lam -> a  — for each (i,j), Cv^T * h2_ij
              RM h3_oovv(n_occ * n_occ * n_virt, n);
              for (Eigen::Index i = 0; i < n_occ; ++i) {
                  const auto row_off = i * n_occ;
                  for (Eigen::Index j = 0; j < n_occ; ++j) {
                      RMap h2_ij(
                          h2_oovv.data() + (row_off + j) * (n * n), n, n);
                      h3_oovv.middleRows((row_off + j) * n_virt, n_virt) =
                          Cv_rm.transpose() * h2_ij;
                  }
              }
              // Step 4b: sig -> b  — for each (i,j,a), rowvec * Cv
              RM oovv(n_occ * n_occ * n_virt, n_virt);
              for (Eigen::Index i = 0; i < n_occ; ++i) {
                  for (Eigen::Index j = 0; j < n_occ; ++j) {
                      const Eigen::Index ij_row_off = (i * n_occ + j) * n_virt;
                      for (Eigen::Index a = 0; a < n_virt; ++a) {
                          oovv.row(ij_row_off + a) =
                              h3_oovv.row(ij_row_off + a) * Cv_rm;
                      }
                  }
              }

              // Return as NumPy arrays.
              auto _make_arr = [](const RM& src,
                                  const std::vector<py::ssize_t>& shape,
                                  const std::vector<py::ssize_t>& strides)
                  -> py::array_t<double> {
                  auto* buf = new double[src.size()];
                  std::copy_n(src.data(), src.size(), buf);
                  return py::array_t<double>(
                      shape, strides, buf,
                      py::capsule(buf,
                                  [](void* p) { delete[] static_cast<double*>(p); }));
              };

              py::array_t<double> a_ovov = _make_arr(
                  ovov,
                  {n_occ, n_virt, n_occ, n_virt},
                  {static_cast<py::ssize_t>(n_virt * n_occ * n_virt
                                            * sizeof(double)),
                   static_cast<py::ssize_t>(n_occ * n_virt * sizeof(double)),
                   static_cast<py::ssize_t>(n_virt * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))});

              py::array_t<double> a_oovv = _make_arr(
                  oovv,
                  {n_occ, n_occ, n_virt, n_virt},
                  {static_cast<py::ssize_t>(n_occ * n_virt * n_virt
                                            * sizeof(double)),
                   static_cast<py::ssize_t>(n_virt * n_virt * sizeof(double)),
                   static_cast<py::ssize_t>(n_virt * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))});

              return py::make_tuple(a_ovov, a_oovv);
          },
          py::arg("eri"), py::arg("mo_coeff"), py::arg("n_occ"),
          "AO-to-MO 4-index ERI transform.  Returns (ovov, oovv).\n"
          "ovov = (ia|jb), shape (n_occ,n_virt,n_occ,n_virt)\n"
          "oovv = (ij|ab), shape (n_occ,n_occ,n_virt,n_virt)");

    m.def("eri_ao_to_mo_cross",
          [](const py::array_t<double, py::array::c_style>& eri,
             const Eigen::MatrixXd& Ca,
             const Eigen::MatrixXd& Cb,
             int n_occ_a,
             int n_occ_b) -> py::array_t<double> {
              if (eri.ndim() != 4) {
                  throw std::invalid_argument(
                      "eri_ao_to_mo_cross: eri must be a rank-4 AO tensor");
              }
              const auto n = static_cast<Eigen::Index>(eri.shape(0));
              for (int axis = 1; axis < 4; ++axis) {
                  if (eri.shape(axis) != n) {
                      throw std::invalid_argument(
                          "eri_ao_to_mo_cross: eri must have shape (n,n,n,n)");
                  }
              }
              if (Ca.rows() != n || Cb.rows() != n) {
                  throw std::invalid_argument(
                      "eri_ao_to_mo_cross: MO coefficient row count must match eri");
              }
              const Eigen::Index n_virt_a = Ca.cols() - n_occ_a;
              const Eigen::Index n_virt_b = Cb.cols() - n_occ_b;
              if (n_occ_a <= 0 || n_occ_b <= 0
                  || n_virt_a <= 0 || n_virt_b <= 0) {
                  throw std::invalid_argument(
                      "eri_ao_to_mo_cross: n_occ_alpha and n_occ_beta must leave "
                      "at least one occupied and one virtual orbital");
              }

              using RM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>;
              using RMap = Eigen::Map<const RM>;

              RM ovov;
              {
                  py::gil_scoped_release unlock_gil;

                  // Convert MO coefficients to row-major for consistent layout.
                  RM Co_a = Ca.leftCols(n_occ_a);
                  RM Cv_a = Ca.rightCols(n_virt_a);
                  RM Co_b = Cb.leftCols(n_occ_b);
                  RM Cv_b = Cb.rightCols(n_virt_b);

                  const Eigen::Index n3 = n * n * n;

                  // Step 1: mu -> i_alpha.
                  RMap eri2d(eri.data(), n, n3);
                  RM h1 = Co_a.transpose() * eri2d;

                  // Step 2: nu -> a_alpha.
                  RM h2(n_occ_a * n_virt_a, n * n);
                  for (Eigen::Index i = 0; i < n_occ_a; ++i) {
                      RMap h1_i(h1.data() + i * n3, n, n * n);
                      h2.middleRows(i * n_virt_a, n_virt_a) =
                          Cv_a.transpose() * h1_i;
                  }

                  // Step 3: lambda -> j_beta.
                  RM h3(n_occ_a * n_virt_a * n_occ_b, n);
                  for (Eigen::Index i = 0; i < n_occ_a; ++i) {
                      for (Eigen::Index a = 0; a < n_virt_a; ++a) {
                          RMap h2_ia(
                              h2.data() + (i * n_virt_a + a) * (n * n),
                              n, n);
                          h3.middleRows((i * n_virt_a + a) * n_occ_b,
                                        n_occ_b) =
                              Co_b.transpose() * h2_ia;
                      }
                  }

                  // Step 4: sigma -> b_beta.
                  ovov.resize(n_occ_a * n_virt_a * n_occ_b, n_virt_b);
                  for (Eigen::Index i = 0; i < n_occ_a; ++i) {
                      for (Eigen::Index a = 0; a < n_virt_a; ++a) {
                          const Eigen::Index ia_row_off =
                              (i * n_virt_a + a) * n_occ_b;
                          for (Eigen::Index j = 0; j < n_occ_b; ++j) {
                              ovov.row(ia_row_off + j) =
                                  h3.row(ia_row_off + j) * Cv_b;
                          }
                      }
                  }
              }

              auto* buf = new double[ovov.size()];
              std::copy_n(ovov.data(), ovov.size(), buf);
              std::vector<py::ssize_t> shape = {
                  static_cast<py::ssize_t>(n_occ_a),
                  static_cast<py::ssize_t>(n_virt_a),
                  static_cast<py::ssize_t>(n_occ_b),
                  static_cast<py::ssize_t>(n_virt_b),
              };
              std::vector<py::ssize_t> strides = {
                  static_cast<py::ssize_t>(n_virt_a * n_occ_b * n_virt_b
                                           * sizeof(double)),
                  static_cast<py::ssize_t>(n_occ_b * n_virt_b
                                           * sizeof(double)),
                  static_cast<py::ssize_t>(n_virt_b * sizeof(double)),
                  static_cast<py::ssize_t>(sizeof(double)),
              };
              return py::array_t<double>(
                  shape, strides, buf,
                  py::capsule(buf,
                              [](void* p) { delete[] static_cast<double*>(p); }));
          },
          py::arg("eri"),
          py::arg("mo_coeff_alpha"),
          py::arg("mo_coeff_beta"),
          py::arg("n_occ_alpha"),
          py::arg("n_occ_beta"),
          "AO-to-MO cross-spin ERI transform. Returns (ia_alpha|jb_beta) "
          "with shape (n_occ_alpha,n_virt_alpha,n_occ_beta,n_virt_beta).");

    // Density-fitting integral kernels. Math + references in
    // cpp/include/vibeqc/df.hpp; the Python side
    // (vibeqc.density_fitting.DensityFitting) consumes these and owns
    // the Cholesky factorisation, B-tensor, and J / K / MO contractions.
    m.def("compute_2c_eri",
          &vibeqc::compute_2c_eri,
          py::arg("aux"),
          py::call_guard<py::gil_scoped_release>(),
          "Two-centre Coulomb metric V_PQ = (P|Q) on an auxiliary basis. "
          "Returns a symmetric positive-definite (n_aux, n_aux) matrix.");

    m.def("compute_2c_eri_gradient_weighted",
          &vibeqc::compute_2c_eri_gradient_weighted,
          py::arg("aux"), py::arg("molecule"), py::arg("omega"),
          py::call_guard<py::gil_scoped_release>(),
          "Per-atom Σ_PQ Ω_PQ ∂V_PQ/∂R for an arbitrary (n_aux, n_aux) "
          "weight Ω. The DF-J gradient passes -(1/2) γ γ^T; the DF-K "
          "gradient passes ω = (η^P : η^Q). Returns (n_atoms, 3).");

    m.def("compute_2c_eri_gradient_contribution",
          &vibeqc::compute_2c_eri_gradient_contribution,
          py::arg("aux"), py::arg("molecule"), py::arg("gamma"),
          py::call_guard<py::gil_scoped_release>(),
          "Per-atom Σ_PQ γ_P γ_Q ∂V_PQ/∂R (rank-1 special case of "
          "compute_2c_eri_gradient_weighted; convenience for the J path).");

    m.def("compute_3c_eri_gradient_weighted",
          &vibeqc::compute_3c_eri_gradient_weighted,
          py::arg("orbital"), py::arg("aux"), py::arg("molecule"),
          py::arg("W"),
          py::call_guard<py::gil_scoped_release>(),
          "Per-atom Σ_{P, μν} W^P_μν ∂(P|μν)/∂R for an arbitrary "
          "(n_aux, n_orb*n_orb) row-major weight tensor W. The DF-J "
          "gradient passes γ_P · D_μν; the DF-K gradient passes "
          "-2 Y^P_μν. Returns (n_atoms, 3).");

    m.def("compute_3c_eri_gradient_contribution",
          &vibeqc::compute_3c_eri_gradient_contribution,
          py::arg("orbital"), py::arg("aux"), py::arg("molecule"),
          py::arg("D"), py::arg("gamma"),
          py::call_guard<py::gil_scoped_release>(),
          "Per-atom Σ_P γ_P · Σ_μν D_μν ∂(P|μν)/∂R (convenience for "
          "the J path; equivalent to "
          "compute_3c_eri_gradient_weighted with W^P_μν = γ_P · D_μν).");

    m.def("compute_df_j_gradient",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux,
             const vibeqc::Molecule& mol,
             const Eigen::MatrixXd& D) {
              vibeqc::DensityFitting df(orbital, aux);
              return df.compute_j_gradient(mol, D);
          },
          py::arg("orbital"), py::arg("aux"),
          py::arg("molecule"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Assemble the complete DF Coulomb-energy gradient at fixed D: "
          "Σ_P γ_P ∂ρ_P/∂R - (1/2) Σ_PQ γ_P γ_Q ∂V_PQ/∂R. For pure DFT "
          "(α_HF = 0) this is the *complete* DF analytic two-electron "
          "gradient. For HF / hybrid DFT pair with the K-gradient "
          "contribution.");

    m.def("compute_df_k_gradient",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux,
             const vibeqc::Molecule& mol,
             const Eigen::MatrixXd& C_occ,
             double alpha_hf) {
              vibeqc::DensityFitting df(orbital, aux);
              return df.compute_k_gradient(mol, C_occ, alpha_hf);
          },
          py::arg("orbital"), py::arg("aux"),
          py::arg("molecule"), py::arg("C_occ"),
          py::arg("alpha_hf") = 1.0,
          py::call_guard<py::gil_scoped_release>(),
          "Assemble the DF Exchange-energy gradient at fixed (C_occ, "
          "α_HF). E_K = -(α_HF/4) tr(D · K_DF) with D = 2 C_occ C_occ^T "
          "(closed shell). For pure DFT (α_HF = 0) returns zero.");

    m.def("compute_df_jk_gradient",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux,
             const vibeqc::Molecule& mol,
             const Eigen::MatrixXd& D,
             const Eigen::MatrixXd& C_occ,
             double alpha_hf) {
              vibeqc::DensityFitting df(orbital, aux);
              return df.compute_jk_gradient(mol, D, C_occ, alpha_hf);
          },
          py::arg("orbital"), py::arg("aux"),
          py::arg("molecule"), py::arg("D"), py::arg("C_occ"),
          py::arg("alpha_hf") = 1.0,
          py::call_guard<py::gil_scoped_release>(),
          "Combined DF J + α_HF·K gradient — the full DF analytic "
          "two-electron gradient for HF / hybrid DFT closed-shell. "
          "Folds both contractions into one 2c + 3c kernel call each, "
          "halving the libint derivative work compared to "
          "compute_df_j_gradient + compute_df_k_gradient.");

    m.def("build_cosx_q",
          &vibeqc::build_cosx_q,
          py::arg("basis"), py::arg("cosx_grid"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the basis-only overlap-fit Q-junction matrix used by "
          "the COSX K corrector (Neese 2009 §2.4). Q = S · S_grid^-1 "
          "is a function of the AO basis and the cosx grid only — "
          "compute it once at SCF setup and pass it to every "
          "``compute_cosx_k`` call to skip the per-iter Q rebuild. "
          "Returns 0×0 on LDLT failure of S_grid (signal to fall back "
          "to the uncorrected K_naive).");

    m.def("build_cosx_schwarz",
          &vibeqc::build_cosx_schwarz,
          py::arg("basis"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the basis-only Schwarz upper-bound table "
          "Q(s1, s2) = sqrt(max |(μν|μν)|) for the COSX shell-pair "
          "screen. Symmetric (n_shells, n_shells). Pass to "
          "``compute_cosx_k`` as ``schwarz_cached`` to enable the "
          "tighter screen: drops pairs whose contribution to K is "
          "below tolerance before paying for the libint "
          "nuclear-attraction call. Basis-only and SCF-invariant.");

    m.def("compute_shell_radial_cutoffs",
          [](const vibeqc::BasisSet& basis, double tol) {
              return vibeqc::compute_shell_radial_cutoffs(
                  basis.libint(), tol);
          },
          py::arg("basis"), py::arg("tol") = 1e-12,
          "Per-shell radial extent (bohr): smallest r* such that "
          "|χ_μ(r)| < tol for any μ in the shell and |r - O_s| > r*. "
          "Conservative (Gaussian-extent) bound derived from each "
          "shell's slowest-decaying primitive. Pass to "
          "``compute_cosx_k`` as ``shell_cutoffs`` to enable the "
          "active-shell pre-prune in the K-build pair loop.");

    // --- GridBatch / GridBatches: per-batch view of the integration grid.
    // The structs are exposed read-only — they are infrastructure for
    // COSX-K / batched XC builders, not a user-facing API. Surfacing
    // them in Python is needed only to validate the batching from
    // ``tests/test_grid_batch.py``.
    py::class_<vibeqc::GridBatch>(m, "GridBatch")
        .def_readonly("start_index",    &vibeqc::GridBatch::start_index)
        .def_readonly("n_points",       &vibeqc::GridBatch::n_points)
        .def_readonly("points",         &vibeqc::GridBatch::points)
        .def_readonly("weights",        &vibeqc::GridBatch::weights)
        .def_readonly("primary_shells", &vibeqc::GridBatch::primary_shells)
        .def_readonly("primary_bfs",    &vibeqc::GridBatch::primary_bfs)
        .def_readonly("chi_primary",    &vibeqc::GridBatch::chi_primary)
        .def_readonly("has_gradient",   &vibeqc::GridBatch::has_gradient);

    py::class_<vibeqc::GridBatches>(m, "GridBatches")
        .def_readonly("batches",         &vibeqc::GridBatches::batches)
        .def_readonly("n_pts_total",     &vibeqc::GridBatches::n_pts_total)
        .def_readonly("n_bf_total",      &vibeqc::GridBatches::n_bf_total)
        .def_readonly("batch_size_hint", &vibeqc::GridBatches::batch_size_hint);

    m.attr("MOLECULAR_XC_GRID_BATCH_SIZE") =
        vibeqc::kMolecularXcGridBatchSize;
    m.attr("MOLECULAR_XC_GRID_MAX_WORKERS") =
        vibeqc::kMolecularXcGridMaxWorkers;
    m.attr("MOLECULAR_XC_KERNEL_MAX_WORKERS") =
        vibeqc::kMolecularXcKernelMaxWorkers;

    m.def("build_grid_batches",
          &vibeqc::build_grid_batches,
          py::arg("basis"), py::arg("grid"),
          py::arg("batch_size"),
          py::arg("shell_cutoffs"),
          py::arg("need_gradient") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Build a batched view of an integration grid with per-batch "
          "primary-shell pruning + a basis-only chi cache. Returns a "
          "``GridBatches`` carrying ``batches[i].chi_primary`` already "
          "evaluated. ``shell_cutoffs`` should come from "
          "``compute_shell_radial_cutoffs(basis, AO_TOL)``. SCF-"
          "invariant — build once at setup; reuse across iterations.");

    m.def("compute_cosx_one_center_correction",
          [](const vibeqc::BasisSet& basis,
             const Eigen::MatrixXd& D,
             const vibeqc::Grid& grid,
             double omega) {
              vibeqc::ensure_libint_initialized();
              const auto& shells = basis.libint();
              const auto cutoffs =
                  vibeqc::compute_shell_radial_cutoffs(shells, 1e-12);
              const auto batches = vibeqc::build_grid_batches(
                  basis, grid, 4096, cutoffs, false);
              const auto boys =
                  vibeqc::build_boys_table(shells.max_l());
              const auto pp_cache =
                  vibeqc::build_primitive_pair_cache(shells);
              const vibeqc::OneCenterCorrection one_center(basis, omega);
              return one_center.apply(
                  D, batches, boys, pp_cache, cutoffs);
          },
          py::arg("basis"), py::arg("D"), py::arg("grid"),
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "COSX analytic one-center correction matrix "
          "K_exact^(1c) − K_quadrature^(1c) for the same-atom exchange "
          "blocks on ``grid``, contracted with the home-cell density "
          "``D``. ``omega > 0`` evaluates BOTH sides with the "
          "erfc(omega*r)/r short-range kernel (libint "
          "``Operator::erfc_coulomb`` analytic ERIs + the M3b-4a "
          "attenuated quadrature seeding) — the variant the "
          "range-separated multi-k COSX bridge applies "
          "post-convergence. Add the result to the final K "
          "(molecular/Gamma-builder lifecycle, commit 40dad042: the "
          "SCF iterates on the uncorrected seminumerical surface; only "
          "the returned Fock/MOs are upgraded).");

    m.def("compute_robust_cosx_k",
          [](const Eigen::MatrixXd& D,
             const Eigen::MatrixXd& K_XvX,
             const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux) {
              vibeqc::DensityFitting df(orbital, aux);
              return vibeqc::compute_robust_cosx_k(D, K_XvX, df);
          },
          py::arg("D"), py::arg("K_XvX"),
          py::arg("orbital"), py::arg("aux"),
          py::call_guard<py::gil_scoped_release>(),
          "Robust Dunlap-fit COSX exchange. K_robust = K_XvX + K_QvQ "
          "− K_XvQ − K_QvX where K_XvQ = Σ_P B^P · K_XvX · B^P. "
          "Constructs a C++ DensityFitting from orbital+aux internally "
          "for the B-tensor. (Neese 2009 §2.5).");

m.def("compute_cosx_k",
      [](const vibeqc::BasisSet& basis,
         const Eigen::MatrixXd& density,
         const vibeqc::Grid& cosx_grid,
         const Eigen::MatrixXd& q_cached,
         const Eigen::MatrixXd& schwarz_cached,
         const std::vector<double>& shell_cutoffs,
         const vibeqc::GridBatches* grid_batches,
         const std::optional<Eigen::Matrix3d>& lattice_opt,
         const std::optional<std::vector<vibeqc::LatticeCell>>&
             image_cells_opt) {
              const Eigen::Matrix3d* lattice_ptr = nullptr;
              Eigen::Matrix3d lattice_copy;
              if (lattice_opt.has_value()) {
                  lattice_copy = *lattice_opt;
                  lattice_ptr = &lattice_copy;
              }
              const std::vector<vibeqc::LatticeCell>* cells_ptr = nullptr;
              if (image_cells_opt.has_value()) {
                  cells_ptr = &*image_cells_opt;
              }
              return vibeqc::compute_cosx_k(
              basis, density, cosx_grid, q_cached,
              schwarz_cached, shell_cutoffs, grid_batches,
              /*boys_table=*/nullptr,
              /*pp_cache=*/nullptr,
              lattice_ptr, cells_ptr);
      },
      py::arg("basis"), py::arg("density"), py::arg("cosx_grid"),
      py::arg("q_cached") = Eigen::MatrixXd(),
      py::arg("schwarz_cached") = Eigen::MatrixXd(),
      py::arg("shell_cutoffs") = std::vector<double>(),
      py::arg("grid_batches") = static_cast<const vibeqc::GridBatches*>(nullptr),
      py::arg("lattice") = std::nullopt,
      py::arg("image_cells") = std::nullopt,
      py::call_guard<py::gil_scoped_release>(),
      "Seminumerical exchange matrix K via Neese's chain-of-spheres "
      "(COSX) algorithm. K_{μν} ≈ Σ_g w_g χ_μ(r_g) (A(r_g)·D·χ(r_g))_ν "
      "where A(r_g) is the analytic 1/|r-r_g| Coulomb-attraction "
      "matrix at the grid point. Returns the symmetrised K. Pair "
      "with RI-J for the standard RIJCOSX hybrid-DFT acceleration. "
      "Optional basis-only / SCF-invariant caches — build once at "
      "setup: ``q_cached`` from ``build_cosx_q`` (skip Q rebuild); "
      "``schwarz_cached`` from ``build_cosx_schwarz`` (tighter "
      "shell-pair screen); ``shell_cutoffs`` from "
      "``compute_shell_radial_cutoffs`` (active-shell pre-prune). "
      "``lattice`` (optional 3×3 matrix) enables periodic minimum-"
      "image convention for shell-distance screening — pass the "
      "unit-cell lattice vectors (columns, bohr) for tight crystals; "
      "omitting it (default) falls back to plain Euclidean distance "
      "for molecular systems. ``image_cells`` (optional list of "
      "LatticeCell from ``direct_lattice_cells``; requires "
      "``lattice``) enables the M3a image-cell exchange summation: "
      "the per-grid-point kernel accumulates the A-operator over all "
      "cells R (home cell included), F_ν(r_g) = Σ_R Σ_σ "
      "A_{νσ}(r_g − R)·(D·χ(r_g))_σ — the home-bra → image-ket "
      "exchange class for tight cells (single-lattice-sum model; "
      "Γ-folded GDF exchange parity needs more — see "
      "handovers/HANDOVER_RIJCOSX_M3A.md). A home-only list keeps the "
      "single-pass kernel; when image summation is active "
      "``grid_batches`` is ignored (unbatched path).");

    // ---- M3b-1: cell-pair double-sum periodic COSX-K ------------------
    py::class_<vibeqc::CosxCellPairCaches>(
        m, "CosxCellPairCaches",
        "Basis-only / SCF-invariant per-relative-cell-shift caches "
        "for ``compute_cosx_k_cell_pair`` (shifted shells, primitive-"
        "pair data, Schwarz tables). Build once per geometry via "
        "``build_cosx_cell_pair_caches``; opaque holder on the Python "
        "side.")
        .def_property_readonly(
            "n_cells",
            [](const vibeqc::CosxCellPairCaches& c) {
                return c.cells.size();
            },
            "Number of lattice cells the (p, g) double sum runs over.")
        .def_property_readonly(
            "n_deltas",
            [](const vibeqc::CosxCellPairCaches& c) {
                return c.deltas.size();
            },
            "Number of Schwarz-surviving relative shifts δ = g − p.")
        .def_readonly(
            "omega", &vibeqc::CosxCellPairCaches::omega,
            "Coulomb-kernel range separation the caches were built "
            "for (0 = full 1/r; ω > 0 = erfc(ω·r)/r short-range).");

    m.def("build_cosx_cell_pair_caches",
          &vibeqc::build_cosx_cell_pair_caches,
          py::arg("basis"), py::arg("cells"),
          py::arg("schwarz_drop_tol") = 1e-10,
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Build the per-δ caches for the cell-pair periodic COSX-K "
          "kernel: for every surviving relative shift δ = g − p of "
          "the supplied cell list (``direct_lattice_cells`` output, "
          "must contain the zero cell), the δ-translated shell set, "
          "the (ν_δ, σ_0) primitive-pair data, and the Häser-Ahlrichs "
          "Schwarz table. Shifts whose whole Schwarz table falls "
          "below ``schwarz_drop_tol`` are dropped wholesale — this "
          "bounds the ket-cell sum to AO-pair-overlap neighbours. "
          "``omega`` selects the kernel (0 = full Coulomb; ω > 0 = "
          "erfc-SR, M3b-4a): the Schwarz metric matches and the "
          "kernels consuming the caches inherit the value. (The δ "
          "set itself is overlap-bounded either way — SR locality "
          "shows up in the K(g) block decay, not the cache size.)");

    m.def("compute_cosx_k_cell_pair",
          [](const vibeqc::BasisSet& basis,
             const Eigen::MatrixXd& density,
             const vibeqc::Grid& cosx_grid,
             const vibeqc::CosxCellPairCaches& caches,
             const Eigen::MatrixXd& q_cached) {
              return vibeqc::compute_cosx_k_cell_pair(
                  basis, density, cosx_grid, caches, q_cached,
                  /*boys_table=*/nullptr);
          },
          py::arg("basis"), py::arg("density"), py::arg("cosx_grid"),
          py::arg("caches"),
          py::arg("q_cached") = Eigen::MatrixXd(),
          py::call_guard<py::gil_scoped_release>(),
          "Cell-pair double-sum periodic COSX exchange (M3b-1): "
          "K_{μν} = Σ_{g,p} Σ_{λσ} D_{λσ} (μ_0 λ_p | ν_g σ_p) — the "
          "Γ-point cell-diagonal-density exchange, the same object as "
          "the direct-ERI ``build_jk_gamma_molecular_limit`` K, "
          "evaluated seminumerically: per grid point and bra cell p, "
          "Dχ_p from the shifted AOs χ(r_G − p), and per ket cell g "
          "the analytic A-block over (ν shifted by δ = g − p, σ home) "
          "with the pseudo-nucleus at r_G − p. Returns the "
          "symmetrised, Q-corrected K. Agrees with the direct-ERI "
          "reference to COSX grid-quadrature accuracy and reduces to "
          "the molecular ``compute_cosx_k`` for a home-only cell "
          "list. Building block for multi-k periodic COSX "
          "(handovers/HANDOVER_RIJCOSX_M3A.md § 'M3b direction'); D must be "
          "symmetric (per-spin densities fine), generalised response "
          "densities are an M3b follow-up.");

    m.def("compute_cosx_k_blocks",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::LatticeMatrixSet& P_real_space,
             const vibeqc::Grid& cosx_grid,
             const vibeqc::CosxCellPairCaches& caches,
             const Eigen::MatrixXd& q_cached) {
              return vibeqc::compute_cosx_k_blocks(
                  basis, P_real_space, cosx_grid, caches, q_cached,
                  /*boys_table=*/nullptr);
          },
          py::arg("basis"), py::arg("P_real_space"),
          py::arg("cosx_grid"), py::arg("caches"),
          py::arg("q_cached") = Eigen::MatrixXd(),
          py::call_guard<py::gil_scoped_release>(),
          "Real-space exchange blocks from real-space density blocks "
          "(M3b-2) — the multi-k periodic COSX engine: "
          "K(g)_{μν} = Σ_{c_λ,c_σ} Σ_{λσ} P(c_σ−c_λ)_{λσ} "
          "(μ_0 λ_{c_λ} | ν_g σ_{c_σ}), the same object and summation "
          "domain as the direct-ERI ``build_jk_2e_real_space_explicit``"
          " K (density looked up by integer cell difference into "
          "``P_real_space.cells``; output LatticeMatrixSet on "
          "``caches.cells``). Per grid point: bra-cell contraction "
          "V^{(c_σ)} = Σ_{c_λ} P(c_σ−c_λ)ᵀ·χ(r_G − c_λ), then the "
          "analytic A-block over (ν shifted by δ = g − c_σ, σ home) "
          "pairs at the pseudo-nucleus r_G − c_σ. ``q_cached`` empty "
          "(default) returns the RAW quadrature blocks; non-empty "
          "applies the bra-side left Q-junction with pairwise "
          "K(g) = K(−g)ᵀ symmetrisation. The density blocks must "
          "satisfy P(h) = P(−h)ᵀ. For insulators the per-point work "
          "is bounded by density decay × AO-pair overlap — "
          "independent of the k-mesh size.");

    m.def("compute_cosx_k_gradient_contribution",
          &vibeqc::compute_cosx_k_gradient_contribution,
          py::arg("molecule"), py::arg("basis"), py::arg("density"),
          py::arg("cosx_grid"), py::arg("alpha_hf"),
          py::arg("q_cached") = Eigen::MatrixXd(),
          py::arg("apply_overlap_fit") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Analytic nuclear gradient of the COSX K-energy term: "
          "dE_K/dR with E_K = -(α_hf/2) tr(D · K_cosx). Frozen "
          "grid / weights / Q-fit; per-atom error ~µHa/bohr on neutral "
          "organics with the default sparse grid. Pass the cached Q "
          "matrix from the K-build to skip redundant Q recomputation.");

    // --- Test-only entry points for the in-tree COSX nuclear-
    // attraction kernel. These exercise the kernel one shell pair at
    // a time so tests can pin it against libint at ULP. Not part of
    // the user-facing API — prefix with underscore.
    m.def("_cosx_nuclear_pair_custom",
          [](const vibeqc::BasisSet& basis, int s1, int s2,
             const std::array<double, 3>& C, double omega) {
              vibeqc::ensure_libint_initialized();
              const auto& shells = basis.libint();
              const int n_shells = static_cast<int>(shells.size());
              if (s1 < 0 || s1 >= n_shells || s2 < 0 || s2 >= n_shells) {
                  throw std::out_of_range(
                      "_cosx_nuclear_pair_custom: shell index out of range");
              }
              auto boys = vibeqc::build_boys_table(shells.max_l());
              auto cache = vibeqc::build_primitive_pair_cache(shells);
              const auto& pp = cache.pairs[
                  static_cast<std::size_t>(s1) * n_shells + s2];
              return vibeqc::cosx_nuclear_pair(
                  shells[s1], shells[s2], pp, C, boys, omega);
          },
          py::arg("basis"), py::arg("s1"), py::arg("s2"), py::arg("C"),
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Test-only: in-tree nuclear-attraction kernel on a single "
          "shell pair (s1, s2) with one pseudo-nucleus of charge q = −1 "
          "at C (bohr). Returns the (n1, n2) contracted block. "
          "``omega > 0`` evaluates the erfc-attenuated short-range "
          "kernel erfc(ω·r)/r (M3b-4a).");

    m.def("_cosx_nuclear_pair_libint",
          [](const vibeqc::BasisSet& basis, int s1, int s2,
             const std::array<double, 3>& C, double omega) {
              vibeqc::ensure_libint_initialized();
              const auto& shells = basis.libint();
              const int n_shells = static_cast<int>(shells.size());
              if (s1 < 0 || s1 >= n_shells || s2 < 0 || s2 >= n_shells) {
                  throw std::out_of_range(
                      "_cosx_nuclear_pair_libint: shell index out of range");
              }
              std::vector<std::pair<double, std::array<double, 3>>>
                  charges{{-1.0, C}};
              const auto n1 = static_cast<int>(shells[s1].size());
              const auto n2 = static_cast<int>(shells[s2].size());
              Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor> out =
                  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>::Zero(n1, n2);
              const double* block = nullptr;
              if (omega > 0.0) {
                  libint2::Engine engine(
                      libint2::Operator::erfc_nuclear,
                      shells.max_nprim(), shells.max_l(), 0);
                  engine.set_params(std::make_tuple(omega, charges));
                  engine.compute(shells[s1], shells[s2]);
                  block = engine.results()[0];
                  if (block != nullptr) {
                      std::memcpy(out.data(), block,
                                  sizeof(double)
                                      * static_cast<std::size_t>(n1)
                                      * static_cast<std::size_t>(n2));
                  }
              } else {
                  libint2::Engine engine(libint2::Operator::nuclear,
                                         shells.max_nprim(),
                                         shells.max_l(), 0);
                  engine.set_params(charges);
                  engine.compute(shells[s1], shells[s2]);
                  block = engine.results()[0];
                  if (block != nullptr) {
                      std::memcpy(out.data(), block,
                                  sizeof(double)
                                      * static_cast<std::size_t>(n1)
                                      * static_cast<std::size_t>(n2));
                  }
              }
              return out;
          },
          py::arg("basis"), py::arg("s1"), py::arg("s2"), py::arg("C"),
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Test-only: libint Operator::nuclear (or erfc_nuclear when "
          "``omega > 0``) reference for the same (s1, s2, C, q = −1) "
          "input as ``_cosx_nuclear_pair_custom``. Used by "
          "tests/test_cosx_kernel.py to pin the in-tree kernel to ULP "
          "against libint.");

    m.def("compute_3c_eri",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux) {
              vibeqc::Eri3D T;
              {
                  py::gil_scoped_release unlock_gil;
                  T = vibeqc::compute_3c_eri(orbital, aux);
              }
              const auto na = static_cast<py::ssize_t>(T.n_aux);
              const auto no = static_cast<py::ssize_t>(T.n_orb);
              auto* owner = new std::vector<double>(std::move(T.data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<double>*>(p);
              });
              return py::array_t<double>(
                  {na, no, no},
                  {static_cast<py::ssize_t>(no * no * sizeof(double)),
                   static_cast<py::ssize_t>(no * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("orbital"),
          py::arg("aux"),
          "Three-centre ERI tensor (P | mu nu) with shape "
          "(n_aux, n_orb, n_orb). Symmetric in (mu, nu); both off-diagonal "
          "positions are filled so callers iterate the full extent.");

    m.def("_df_half_transform_for_test",
          [](const Eigen::MatrixXd& L, const Eigen::MatrixXd& T_flat) {
              Eigen::MatrixXd legacy;
              std::pair<Eigen::MatrixXd, int> blocked;
              {
                  py::gil_scoped_release unlock_gil;
                  legacy = L.triangularView<Eigen::Lower>().solve(T_flat);
                  blocked = vibeqc::detail::solve_df_half_transform(
                      L, T_flat, /*force_blocked=*/true);
              }
              return py::make_tuple(
                  std::move(legacy),
                  std::move(blocked.first),
                  blocked.second);
          },
          py::arg("L"), py::arg("T_flat"),
          "Test-only legacy and fixed-block DF half transforms plus the "
          "actual blocked-path worker count.");

    // ----- Periodic DF kernels ------------------------------------------
    // Periodic 2c / 3c Coulomb integrals for the GDF Lpq build. See
    // cpp/include/vibeqc/aux_eri.hpp for the math contract; the Python
    // consumer is vibeqc.aux_basis.build_lpq_native.
    m.def("compute_2c_eri_lattice",
          &vibeqc::compute_2c_eri_lattice,
          py::arg("aux"), py::arg("system"), py::arg("opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Periodic 2-centre Coulomb metric M_PQ = Σ_T (P_0 | Q_T) on an "
          "auxiliary basis. Image sum runs over direct-lattice cells with "
          "|g| ≤ opts.cutoff_bohr. Returns a symmetric (n_aux, n_aux) "
          "matrix. Bare lattice sum — for general convergence on "
          "non-charge-compensated aux bases use a charge-compensated aux "
          "basis (see vibeqc.aux_basis.make_aux_basis_set).");

    m.def("compute_2c_eri_lattice_blocks",
          [](const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts) {
              vibeqc::Lattice2CEriBlockSet blocks;
              {
                  py::gil_scoped_release unlock_gil;
                  blocks = vibeqc::compute_2c_eri_lattice_blocks(
                      aux, system, opts);
              }
              const auto nc = static_cast<py::ssize_t>(blocks.cells.size());
              const auto na = static_cast<py::ssize_t>(blocks.n_aux);
              py::array_t<int> indices({nc, py::ssize_t{3}});
              py::array_t<double> vectors({nc, py::ssize_t{3}});
              py::array_t<double> data({nc, na, na});
              auto idx = indices.mutable_unchecked<2>();
              auto vec = vectors.mutable_unchecked<2>();
              auto arr = data.mutable_unchecked<3>();
              for (py::ssize_t c = 0; c < nc; ++c) {
                  const auto& cell = blocks.cells[static_cast<std::size_t>(c)];
                  for (py::ssize_t a = 0; a < 3; ++a) {
                      idx(c, a) = cell.index[static_cast<int>(a)];
                      vec(c, a) = cell.r_cart[static_cast<int>(a)];
                  }
                  const auto& M = blocks.blocks[static_cast<std::size_t>(c)];
                  for (py::ssize_t p = 0; p < na; ++p) {
                      for (py::ssize_t q = 0; q < na; ++q) {
                          arr(c, p, q) = M(
                              static_cast<Eigen::Index>(p),
                              static_cast<Eigen::Index>(q));
                      }
                  }
              }
              return py::make_tuple(indices, vectors, data);
          },
          py::arg("aux"), py::arg("system"), py::arg("opts"),
          "Cell-resolved periodic 2-centre Coulomb metric blocks. "
          "Returns (cell_indices, cell_vectors_bohr, blocks), where "
          "blocks[c, P, Q] = (P_0 | Q_Tc). Summing over c reproduces "
          "compute_2c_eri_lattice at Γ.");

    m.def("compute_3c_eri_lattice",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts) {
              vibeqc::Eri3D T;
              {
                  py::gil_scoped_release unlock_gil;
                  T = vibeqc::compute_3c_eri_lattice(orbital, aux, system, opts);
              }
              const auto na = static_cast<py::ssize_t>(T.n_aux);
              const auto no = static_cast<py::ssize_t>(T.n_orb);
              auto* owner = new std::vector<double>(std::move(T.data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<double>*>(p);
              });
              return py::array_t<double>(
                  {na, no, no},
                  {static_cast<py::ssize_t>(no * no * sizeof(double)),
                   static_cast<py::ssize_t>(no * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("orbital"), py::arg("aux"),
          py::arg("system"), py::arg("opts"),
          "Periodic 3-centre ERI tensor T_{P, μ, ν} = Σ_T (P_0 | μ_0 ν_T). "
          "Aux P and orbital μ stay anchored at the reference cell; the "
          "second orbital ν is shifted across direct-lattice translations "
          "with |g| ≤ opts.cutoff_bohr. Symmetric in (μ, ν). Shape "
          "(n_aux, n_orb, n_orb).");

    m.def("compute_3c_eri_lattice_blocks",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts) {
              vibeqc::Lattice3CEriBlockSet blocks;
              {
                  py::gil_scoped_release unlock_gil;
                  blocks = vibeqc::compute_3c_eri_lattice_blocks(
                      orbital, aux, system, opts);
              }
              const auto nc = static_cast<py::ssize_t>(blocks.cells.size());
              const auto na = static_cast<py::ssize_t>(blocks.n_aux);
              const auto no = static_cast<py::ssize_t>(blocks.n_orb);
              py::array_t<int> indices({nc, py::ssize_t{3}});
              py::array_t<double> vectors({nc, py::ssize_t{3}});
              py::array_t<double> data({nc, na, no, no});
              auto idx = indices.mutable_unchecked<2>();
              auto vec = vectors.mutable_unchecked<2>();
              auto arr = data.mutable_unchecked<4>();
              for (py::ssize_t c = 0; c < nc; ++c) {
                  const auto& cell = blocks.cells[static_cast<std::size_t>(c)];
                  for (py::ssize_t a = 0; a < 3; ++a) {
                      idx(c, a) = cell.index[static_cast<int>(a)];
                      vec(c, a) = cell.r_cart[static_cast<int>(a)];
                  }
                  const auto& T = blocks.blocks[static_cast<std::size_t>(c)];
                  for (py::ssize_t p = 0; p < na; ++p) {
                      for (py::ssize_t mu = 0; mu < no; ++mu) {
                          for (py::ssize_t nu = 0; nu < no; ++nu) {
                              arr(c, p, mu, nu) = T(
                                  static_cast<std::size_t>(p),
                                  static_cast<std::size_t>(mu),
                                  static_cast<std::size_t>(nu));
                          }
                      }
                  }
              }
              return py::make_tuple(indices, vectors, data);
          },
          py::arg("orbital"), py::arg("aux"),
          py::arg("system"), py::arg("opts"),
          "Cell-resolved periodic 3-centre ERI blocks. Returns "
          "(cell_indices, cell_vectors_bohr, blocks), where "
          "blocks[c, P, mu, nu] = (P_0 | mu_0 nu_Tc). Individual "
          "blocks are not symmetrized in (mu,nu); summing over c "
          "reproduces compute_3c_eri_lattice at Γ.");

    // RSGDF — short-range halves (Ye & Berkelbach, J. Chem. Phys. 154,
    // 131104 (2021), DOI 10.1063/5.0046617). Same kernels as above but
    // with the Coulomb operator replaced by erfc(ω r)/r.
    m.def("compute_2c_eri_lattice_sr",
          &vibeqc::compute_2c_eri_lattice_sr,
          py::arg("aux"), py::arg("system"), py::arg("opts"), py::arg("omega"),
          py::call_guard<py::gil_scoped_release>(),
          "RSGDF short-range 2-centre metric "
          "M^SR_{PQ}(ω) = Σ_T (P_0 | erfc(ωr)/r | Q_T). The lattice sum "
          "converges absolutely (no compensating-charge bookkeeping "
          "needed) because the erfc kernel decays as exp(-ω²r²). The "
          "long-range half is built in Python from analytical Gaussian "
          "Fourier transforms — see vibeqc.aux_basis.build_lpq_rsgdf.");

    m.def("compute_3c_eri_lattice_sr",
          [](const vibeqc::BasisSet& orbital,
             const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts,
             double omega) {
              vibeqc::Eri3D T;
              {
                  py::gil_scoped_release unlock_gil;
                  T = vibeqc::compute_3c_eri_lattice_sr(
                      orbital, aux, system, opts, omega);
              }
              const auto na = static_cast<py::ssize_t>(T.n_aux);
              const auto no = static_cast<py::ssize_t>(T.n_orb);
              auto* owner = new std::vector<double>(std::move(T.data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<double>*>(p);
              });
              return py::array_t<double>(
                  {na, no, no},
                  {static_cast<py::ssize_t>(no * no * sizeof(double)),
                   static_cast<py::ssize_t>(no * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("orbital"), py::arg("aux"),
          py::arg("system"), py::arg("opts"), py::arg("omega"),
          "RSGDF short-range 3-centre tensor "
          "T^SR_{P, μν}(ω) = Σ_T (P_0 | erfc(ωr)/r | μ_0 ν_T). "
          "Image-summed on the second orbital factor; converges "
          "absolutely thanks to the erfc decay. Shape (n_aux, n_orb, "
          "n_orb), symmetric in (μ, ν).");

    // ----- Periodic GDF analytic gradient kernels ------------------------
    // Lattice-summed analogues of the molecular 2c/3c gradient kernels in
    // df.hpp. Outer loop over lattice cells T; libint deriv_order=1 on the
    // shifted shell origins.
    m.def("compute_2c_eri_lattice_gradient_weighted",
          &vibeqc::compute_2c_eri_lattice_gradient_weighted,
          py::arg("aux"), py::arg("system"), py::arg("opts"),
          py::arg("omega"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed 2-centre metric gradient: "
          "grad(A, c) = Σ_{PQ} Ω_{PQ} Σ_T ∂(P_0|Q_T)/∂R_{A,c}. "
          "Returns (n_atoms, 3) in Hartree/bohr.");

    m.def("compute_3c_eri_lattice_gradient_weighted",
          &vibeqc::compute_3c_eri_lattice_gradient_weighted,
          py::arg("orbital"), py::arg("aux"),
          py::arg("system"), py::arg("opts"), py::arg("W"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed 3-centre ERI gradient: "
          "grad(A, c) = Σ_{P, μν} W^P_{μν} Σ_T "
          "∂(P_0|μ_0 ν_T)/∂R_{A,c}. "
          "W is (n_aux, n_orb²) row-major. Returns (n_atoms, 3) in "
          "Hartree/bohr.");

    // ----- AO-pair FT (Bloch-summed) ------------------------------------
    // Math + algorithm in cpp/include/vibeqc/aopair_ft.hpp. The Python
    // wrapper at python/vibeqc/_aopair_ft.py dispatches to this kernel
    // when the basis is s-only and falls back to the pure-Python
    // McMurchie-Davidson path for higher L (until the general-L kernel
    // lands — Item 1b in the integrals-chat handover). Output is
    // complex128 with shape (n_orb, n_orb, n_G).
    m.def("ao_pair_fourier_transform_bloch_ss",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& k_cart) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              if (k_cart.ndim() != 1 || k_cart.shape(0) != 3) {
                  throw std::invalid_argument(
                      "k_cart must have shape (3,)");
              }
              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              Eigen::Vector3d k_vec(k_cart.data()[0],
                                    k_cart.data()[1],
                                    k_cart.data()[2]);

              vibeqc::AOPairFTTensor T;
              {
                  py::gil_scoped_release unlock_gil;
                  T = vibeqc::ao_pair_fourier_transform_bloch_ss(
                      basis, G_map, R_map, k_vec);
              }
              const auto no = static_cast<py::ssize_t>(T.n_orb);
              const auto nG = static_cast<py::ssize_t>(T.n_G);
              using cplx = std::complex<double>;
              auto* owner = new std::vector<cplx>(std::move(T.data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<cplx>*>(p);
              });
              return py::array_t<cplx>(
                  {no, no, nG},
                  {static_cast<py::ssize_t>(no * nG * sizeof(cplx)),
                   static_cast<py::ssize_t>(nG * sizeof(cplx)),
                   static_cast<py::ssize_t>(sizeof(cplx))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("k_cart"),
          "Bloch-summed AO-pair FT — s-only fast path. "
          "out_{μν}(G) = Σ_R exp(+i k·R) · ∫ χ_μ(r) χ_ν(r − R) "
          "e^{−iG·r} dr. Returns (n_orb, n_orb, n_G) complex128. "
          "Throws if any shell has L > 0 — see "
          "``ao_pair_fourier_transform_bloch_cxx`` for the general-L "
          "kernel.");

    // General-L Bloch-summed AO-pair FT — McMurchie-Davidson chain
    // + Cartesian-to-spherical transform. Dispatcher in
    // python/vibeqc/_aopair_ft.py picks this kernel over the SS fast
    // path when the basis has any L > 0.
    m.def("ao_pair_fourier_transform_bloch_cxx",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& k_cart,
             double screen_tol) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              if (k_cart.ndim() != 1 || k_cart.shape(0) != 3) {
                  throw std::invalid_argument(
                      "k_cart must have shape (3,)");
              }
              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              Eigen::Vector3d k_vec(k_cart.data()[0],
                                    k_cart.data()[1],
                                    k_cart.data()[2]);

              vibeqc::AOPairFTTensor T;
              {
                  py::gil_scoped_release unlock_gil;
                  T = vibeqc::ao_pair_fourier_transform_bloch(
                      basis, G_map, R_map, k_vec, screen_tol);
              }
              const auto no = static_cast<py::ssize_t>(T.n_orb);
              const auto nG = static_cast<py::ssize_t>(T.n_G);
              using cplx = std::complex<double>;
              auto* owner = new std::vector<cplx>(std::move(T.data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<cplx>*>(p);
              });
              return py::array_t<cplx>(
                  {no, no, nG},
                  {static_cast<py::ssize_t>(no * nG * sizeof(cplx)),
                   static_cast<py::ssize_t>(nG * sizeof(cplx)),
                   static_cast<py::ssize_t>(sizeof(cplx))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("k_cart"),
          py::arg("screen_tol") = 0.0,
          "Bloch-summed AO-pair FT — general L via McMurchie-Davidson "
          "Hermite chain + Cartesian-to-spherical transform. Supports "
          "any pure-spherical basis up to L = 6 — the extent of the "
          "table at cpp/include/vibeqc/cart_to_sph_data.hpp, which is "
          "also the ceiling on libint's configurable build-time max_am "
          "(scripts/_libint_max_am.sh). Throws on Cartesian (non-pure) "
          "L > 0 shells.");

    m.def("ao_pair_fourier_transform_weighted_per_cell",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<std::complex<double>,
                               py::array::c_style |
                               py::array::forcecast>& reciprocal_weights,
             double screen_tol) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              if (reciprocal_weights.ndim() != 1) {
                  throw std::invalid_argument(
                      "reciprocal_weights must have shape (n_G,)");
              }
              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              const auto W_n =
                  static_cast<Eigen::Index>(reciprocal_weights.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              Eigen::Map<const Eigen::VectorXcd> W_map(
                  reciprocal_weights.data(), W_n);

              std::vector<double> data;
              {
                  py::gil_scoped_release unlock_gil;
                  data = vibeqc::ao_pair_fourier_transform_weighted_per_cell(
                      basis, G_map, R_map, W_map, screen_tol);
              }
              const auto no =
                  static_cast<py::ssize_t>(basis.nbasis());
              const auto ng = static_cast<py::ssize_t>(R_n);
              auto* owner = new std::vector<double>(std::move(data));
              py::capsule keep_alive(owner, [](void* p) {
                  delete static_cast<std::vector<double>*>(p);
              });
              return py::array_t<double>(
                  {ng, no, no},
                  {static_cast<py::ssize_t>(no * no * sizeof(double)),
                   static_cast<py::ssize_t>(no * sizeof(double)),
                   static_cast<py::ssize_t>(sizeof(double))},
                  owner->data(),
                  keep_alive);
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("reciprocal_weights"),
          py::arg("screen_tol") = 0.0,
          "Per-cell reciprocal-weight-contracted AO-pair FT: "
          "out[g,m,n] = Re(sum_G w(G) conj(FT_mn(G; R_g))). Returns a "
          "(n_g, n_orb, n_orb) real array. Never materialises the dense "
          "(n_orb, n_orb, n_G) tensor and parallelises over the whole "
          "flattened (shell-pair, cell) task space.");

    m.def("ao_pair_fourier_transform_bloch_multi_cxx",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& k_carts,
             double screen_tol,
             const std::optional<py::array_t<double,
                                             py::array::c_style |
                                             py::array::forcecast>>&
                 pair_weights) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              if (k_carts.ndim() != 2 || k_carts.shape(1) != 3) {
                  throw std::invalid_argument(
                      "k_carts must have shape (n_k, 3)");
              }
              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              const auto k_n = static_cast<Eigen::Index>(k_carts.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  k_map(k_carts.data(), k_n, 3);

              std::vector<double> weights;
              if (pair_weights.has_value()) {
                  const auto& w = pair_weights.value();
                  const auto n_orb =
                      static_cast<py::ssize_t>(basis.nbasis());
                  if (w.ndim() != 2 || w.shape(0) != n_orb
                          || w.shape(1) != n_orb) {
                      throw std::invalid_argument(
                          "pair_weights must have shape (n_orb, n_orb)");
                  }
                  weights.assign(w.data(), w.data() + w.size());
              }

              std::vector<vibeqc::AOPairFTTensor> tensors;
              {
                  py::gil_scoped_release unlock_gil;
                  tensors = vibeqc::ao_pair_fourier_transform_bloch_multi(
                      basis, G_map, R_map, k_map, screen_tol, weights);
              }
              using cplx = std::complex<double>;
              py::list result;
              for (auto& T : tensors) {
                  const auto no = static_cast<py::ssize_t>(T.n_orb);
                  const auto nG = static_cast<py::ssize_t>(T.n_G);
                  auto* owner = new std::vector<cplx>(std::move(T.data));
                  py::capsule keep_alive(owner, [](void* p) {
                      delete static_cast<std::vector<cplx>*>(p);
                  });
                  result.append(py::array_t<cplx>(
                      {no, no, nG},
                      {static_cast<py::ssize_t>(no * nG * sizeof(cplx)),
                       static_cast<py::ssize_t>(nG * sizeof(cplx)),
                       static_cast<py::ssize_t>(sizeof(cplx))},
                      owner->data(),
                      keep_alive));
              }
              return result;
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("k_carts"),
          py::arg("screen_tol") = 0.0,
          py::arg("pair_weights") = py::none(),
          "Multi-k batched Bloch-summed AO-pair FT: one k-independent "
          "per-cell McMurchie-Davidson pass shared across every row of "
          "k_carts, folded per k with its Bloch phase. Returns a list "
          "of (n_orb, n_orb, n_G) complex arrays, one per k. Same "
          "shell/L contract as ao_pair_fourier_transform_bloch_cxx. "
          "Optional (n_orb, n_orb) real pair_weights is applied as the "
          "result is stored, bit-identically to scaling the returned "
          "tensors with NumPy afterwards; an exactly-zero weight stores "
          "+0.0 (a masked pair).");

    m.def("ao_pair_fourier_transform_gamma_gradient_weighted",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& pair_weights,
             const py::array_t<std::complex<double>, py::array::c_style |
                                                     py::array::forcecast>&
                 reciprocal_weights,
             std::size_t n_atoms) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              const auto no = static_cast<py::ssize_t>(basis.nbasis());
              if (pair_weights.ndim() != 2
                      || pair_weights.shape(0) != no
                      || pair_weights.shape(1) != no) {
                  throw std::invalid_argument(
                      "pair_weights must have shape (n_orb, n_orb)");
              }
              if (reciprocal_weights.ndim() != 1
                      || reciprocal_weights.shape(0) != G_vectors.shape(0)) {
                  throw std::invalid_argument(
                      "reciprocal_weights must have shape (n_G,)");
              }

              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>
                  W_map(pair_weights.data(),
                        static_cast<Eigen::Index>(no),
                        static_cast<Eigen::Index>(no));
              Eigen::Map<const Eigen::VectorXcd> q_map(
                  reciprocal_weights.data(), G_n);
              Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
                  gradient;
              {
                  py::gil_scoped_release unlock_gil;
                  gradient =
                      vibeqc::ao_pair_fourier_transform_gamma_gradient_weighted(
                          basis, G_map, R_map, W_map, q_map, n_atoms);
              }
              return gradient;
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("pair_weights"),
          py::arg("reciprocal_weights"), py::arg("n_atoms"),
          "Weighted Gamma-point AO-pair FT centre gradient. Computes "
          "d/dR_A Re sum_G q_G conj(sum_mn W_mn sum_R FT_mn(G;R)) "
          "without materialising AO-pair derivative tensors. Returns "
          "(n_atoms, 3).");

    m.def("ao_pair_fourier_transform_gamma_gradient_gweighted",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<std::complex<double>, py::array::c_style |
                                                     py::array::forcecast>&
                 pair_weights_g,
             std::size_t n_atoms) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              const auto no = static_cast<py::ssize_t>(basis.nbasis());
              if (pair_weights_g.ndim() != 3
                      || pair_weights_g.shape(0) != no
                      || pair_weights_g.shape(1) != no
                      || pair_weights_g.shape(2) != G_vectors.shape(0)) {
                  throw std::invalid_argument(
                      "pair_weights_g must have shape (n_orb, n_orb, n_G)");
              }

              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              const std::complex<double>* Q_ptr = pair_weights_g.data();
              Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
                  gradient;
              {
                  py::gil_scoped_release unlock_gil;
                  gradient =
                      vibeqc::ao_pair_fourier_transform_gamma_gradient_gweighted(
                          basis, G_map, R_map, Q_ptr, n_atoms);
              }
              return gradient;
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("pair_weights_g"),
          py::arg("n_atoms"),
          "G-resolved-weighted Gamma-point AO-pair FT centre gradient. "
          "Computes d/dR_A Re sum_G sum_mn Q_mn(G) conj(sum_R FT_mn(G;R)) "
          "with one complex (n_orb, n_orb, n_G) pair weight replacing the "
          "separable pair_weights x reciprocal_weights product. Returns "
          "(n_atoms, 3).");

    m.def("ao_pair_fourier_transform_bloch_gradient_gweighted",
          [](const vibeqc::BasisSet& basis,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& G_vectors,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& R_g_list,
             const py::array_t<double, py::array::c_style |
                                       py::array::forcecast>& k_cart,
             const py::array_t<std::complex<double>, py::array::c_style |
                                                     py::array::forcecast>&
                 pair_weights_g,
             std::size_t n_atoms) {
              if (G_vectors.ndim() != 2 || G_vectors.shape(1) != 3) {
                  throw std::invalid_argument(
                      "G_vectors must have shape (n_G, 3)");
              }
              if (R_g_list.ndim() != 2 || R_g_list.shape(1) != 3) {
                  throw std::invalid_argument(
                      "R_g_list must have shape (n_g, 3)");
              }
              if (k_cart.ndim() != 1 || k_cart.shape(0) != 3) {
                  throw std::invalid_argument(
                      "k_cart must have shape (3,)");
              }
              const auto no = static_cast<py::ssize_t>(basis.nbasis());
              if (pair_weights_g.ndim() != 3
                      || pair_weights_g.shape(0) != no
                      || pair_weights_g.shape(1) != no
                      || pair_weights_g.shape(2) != G_vectors.shape(0)) {
                  throw std::invalid_argument(
                      "pair_weights_g must have shape (n_orb, n_orb, n_G)");
              }

              const auto G_n = static_cast<Eigen::Index>(G_vectors.shape(0));
              const auto R_n = static_cast<Eigen::Index>(R_g_list.shape(0));
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  G_map(G_vectors.data(), G_n, 3);
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>>
                  R_map(R_g_list.data(), R_n, 3);
              Eigen::Vector3d k_vec(k_cart.data()[0],
                                    k_cart.data()[1],
                                    k_cart.data()[2]);
              const std::complex<double>* Q_ptr = pair_weights_g.data();
              Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
                  gradient;
              {
                  py::gil_scoped_release unlock_gil;
                  gradient =
                      vibeqc::ao_pair_fourier_transform_bloch_gradient_gweighted(
                          basis, G_map, R_map, k_vec, Q_ptr, n_atoms);
              }
              return gradient;
          },
          py::arg("basis"), py::arg("G_vectors"),
          py::arg("R_g_list"), py::arg("k_cart"),
          py::arg("pair_weights_g"), py::arg("n_atoms"),
          "Bloch-phased G-resolved-weighted AO-pair FT centre gradient "
          "(HANDOVER_GDF_GRADIENT_DEFERRED.md section 4, rung 1). Computes "
          "d/dR_A Re sum_G sum_mn Q_mn(G) "
          "conj(sum_R exp(+i k.R) FT_mn(G;R)); k_cart = 0 reduces to the "
          "Gamma kernel bit-for-bit. Returns (n_atoms, 3).");

    // ----- RHF -----------------------------------------------------------
    py::enum_<vibeqc::InitialGuess>(m, "InitialGuess")
        .value("AUTO",    vibeqc::InitialGuess::AUTO)
        .value("HCORE",   vibeqc::InitialGuess::HCORE)
        .value("SAD",     vibeqc::InitialGuess::SAD)
        .value("SAP",     vibeqc::InitialGuess::SAP)
        .value("PATOM",   vibeqc::InitialGuess::PATOM)
        .value("HUECKEL", vibeqc::InitialGuess::HUECKEL)
        .value("HUCKEL",  vibeqc::InitialGuess::HUECKEL)
        .value("MINAO",   vibeqc::InitialGuess::MINAO)
        .value("READ",    vibeqc::InitialGuess::READ)
        .value("FRAGMO",  vibeqc::InitialGuess::FRAGMO);

    py::register_exception<vibeqc::GuessCapabilityError>(
        m, "GuessCapabilityError", PyExc_NotImplementedError);

    py::class_<vibeqc::GuessSelection>(m, "GuessSelection")
        .def(py::init([](vibeqc::InitialGuess requested,
                         vibeqc::InitialGuess effective,
                         vibeqc::InitialGuess transport) {
            vibeqc::validate_initial_guess(requested);
            vibeqc::validate_initial_guess(effective);
            vibeqc::validate_initial_guess(transport);
            return vibeqc::GuessSelection{requested, effective, transport};
        }), py::arg("requested"), py::arg("effective"), py::arg("transport"))
        .def_readonly("requested", &vibeqc::GuessSelection::requested)
        .def_readonly("effective", &vibeqc::GuessSelection::effective)
        .def_readonly("transport", &vibeqc::GuessSelection::transport);

    py::enum_<vibeqc::SpinlockMode>(m, "SpinlockMode")
        .value("OFF",           vibeqc::SpinlockMode::OFF)
        .value("SPIN_SCHEDULE", vibeqc::SpinlockMode::SPIN_SCHEDULE)
        .value("PATTERN_HOLD",  vibeqc::SpinlockMode::PATTERN_HOLD);

    m.def("_validate_guess_atomic_spins", &vibeqc::validate_guess_atomic_spins,
          py::arg("mol"), py::arg("basis"), py::arg("alpha"), py::arg("beta"),
          py::arg("metric"), py::arg("atomic_spins"));

    m.def("_normalize_atomic_guess", &vibeqc::normalize_atomic_guess,
          py::arg("mol"), py::arg("basis"), py::arg("alpha"), py::arg("beta"),
          py::arg("metric"), py::arg("n_alpha"), py::arg("n_beta"),
          py::arg("atomic_spins"));

    m.def("_normalize_guess_density",
          [](const Eigen::MatrixXcd& density, const Eigen::MatrixXcd& metric,
             double electrons) {
              return vibeqc::normalize_guess_density(density, metric, electrons);
          }, py::arg("density"), py::arg("metric"), py::arg("electrons"),
          "Normalize a density-mode guess in the actual Hermitian SCF metric.");
    m.def("_normalize_guess_spin_densities",
          [](const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
             const Eigen::MatrixXcd& metric, int n_alpha, int n_beta) {
              return vibeqc::normalize_guess_spin_densities(
                  alpha, beta, metric, n_alpha, n_beta);
          }, py::arg("alpha"), py::arg("beta"), py::arg("metric"),
          py::arg("n_alpha"), py::arg("n_beta"),
          "Normalize spin populations while preserving the total/Hund pattern "
          "or a deliberate per-atom seed.");

    m.def("_resolve_initial_guess_for_molecule",
          &vibeqc::GuessEngine::resolve_auto_for_molecule,
          py::arg("molecule"), py::arg("kind"),
          py::arg("is_periodic") = false,
          py::arg("is_open_shell") = false,
          py::arg("supported") = py::none(),
          "Resolve AUTO with the same molecule-derived hints used by the "
          "production GuessEngine; non-AUTO kinds pass through unchanged.");

    py::class_<vibeqc::GuessECPContext>(m, "_GuessECPContext")
        .def(py::init<>())
        .def_readwrite("xml_centers", &vibeqc::GuessECPContext::xml_centers)
        .def_readwrite("xml_library", &vibeqc::GuessECPContext::xml_library)
        .def_readwrite("primitive_blocks", &vibeqc::GuessECPContext::primitive_blocks)
        .def_readwrite("primitive_centers", &vibeqc::GuessECPContext::primitive_centers)
        .def_readwrite("effective_charges", &vibeqc::GuessECPContext::effective_charges)
        .def_readwrite("total_ncore", &vibeqc::GuessECPContext::total_ncore)
        .def_property_readonly("active", &vibeqc::GuessECPContext::active);

    m.def("_validate_guess_ecp", &vibeqc::validate_guess_ecp,
          py::arg("kind"), py::arg("ecp_context"), py::arg("molecule") = py::none());

    // ---- Engine helpers used by the Python periodic drivers ------------
    // See python/vibeqc/guess.py for the Python-facing wrappers. These
    // private bindings call into GuessEngine with S/Hcore/JK = nullptr —
    // matching what the Python periodic drivers can supply today. The
    // future Fock-mode guesses (SAP, HUECKEL) will need an overload
    // that takes Hcore + a periodic JKBuilder; out of scope for v0.9.x
    // Commit 1, which is the scaffolding-only step.
    m.def("_guess_closed_shell_density",
          [](const vibeqc::Molecule& mol, const vibeqc::BasisSet& basis,
             int n_occ, vibeqc::InitialGuess kind,
             bool is_periodic, py::object overlap, const vibeqc::GuessECPContext& ecp) -> py::object {
              Eigen::MatrixXd metric;
              if (!overlap.is_none()) metric = overlap.cast<Eigen::MatrixXd>();
              vibeqc::SystemHints hints;
              hints.is_periodic = is_periodic;
              auto g = vibeqc::GuessEngine::build_closed_shell(
                  mol, basis, n_occ, kind,
                  /*S=*/overlap.is_none() ? nullptr : &metric,
                  /*Hcore=*/nullptr, /*jk=*/nullptr,
                  hints, 1e-7, ecp);
              if (g.D.size() == 0) return py::none();
              return py::cast(g.D);
          },
          py::arg("mol"), py::arg("basis"), py::arg("n_occ"),
          py::arg("kind"), py::arg("is_periodic") = true,
          py::arg("overlap") = py::none(),
          py::arg("ecp_context") = vibeqc::GuessECPContext{},
          "Closed-shell initial density via GuessEngine. Returns None "
          "for HCORE (and any future Fock-mode kind reached without a "
          "JKBuilder).");

    m.def("compute_vsap_ecp", &vibeqc::compute_vsap_ecp,
          py::arg("mol"), py::arg("basis"), py::arg("ecp_context"),
          py::call_guard<py::gil_scoped_release>());
    m.def("compute_vsap_ecp_lattice", &vibeqc::compute_vsap_ecp_lattice,
          py::arg("basis"), py::arg("system"), py::arg("options"), py::arg("ecp_context"),
          py::call_guard<py::gil_scoped_release>());

    m.def("compute_minao_density_periodic",
          &vibeqc::compute_minao_density_periodic,
          py::arg("mol"), py::arg("basis"), py::arg("system"),
          py::arg("S_gamma"), py::arg("n_electrons"),
          py::arg("options"), py::arg("ecp_context") = vibeqc::GuessECPContext{},
          py::call_guard<py::gil_scoped_release>(),
          "Periodic MINAO density-mode guess. Projects the ANO-RCC minimal "
          "reference density onto the working periodic basis using the "
          "Gamma-folded target overlap and lattice-summed cross overlap.");

    m.def("compute_huckel_fock_lattice",
          &vibeqc::compute_huckel_fock_lattice,
          py::arg("mol"), py::arg("basis"), py::arg("overlap_lattice"),
          py::arg("ecp_context") = vibeqc::GuessECPContext{},
          py::call_guard<py::gil_scoped_release>(),
          "Periodic HUECKEL Fock-mode lattice guess. Builds the GWH Fock "
          "blocks from the lattice overlap and per-AO atomic energies.");

    m.def("_guess_open_shell_density",
          [](const vibeqc::Molecule& mol, const vibeqc::BasisSet& basis,
             int n_alpha, int n_beta, vibeqc::InitialGuess kind,
             bool is_periodic,
             const std::vector<int>& atomic_spins,
             py::object overlap, const vibeqc::GuessECPContext& ecp) -> py::object {
              Eigen::MatrixXd metric;
              if (!overlap.is_none()) metric = overlap.cast<Eigen::MatrixXd>();
              vibeqc::SystemHints hints;
              hints.is_periodic = is_periodic;
              hints.is_open_shell = true;
              // ATOMSPIN: a per-atom spin seed assembles a block-diagonal
              // broken-symmetry SAD density. The engine is is_periodic-
              // agnostic here — at Γ the g=0 cell density Bloch-sums to a
              // broken-symmetry D(k). nullptr/empty = spin-symmetric split.
              auto g = vibeqc::GuessEngine::build_open_shell(
                  mol, basis, n_alpha, n_beta, kind,
                  /*S=*/overlap.is_none() ? nullptr : &metric,
                  /*Hcore=*/nullptr, /*jk=*/nullptr,
                  hints, /*linear_dep_threshold=*/1e-7,
                  atomic_spins.empty() ? nullptr : &atomic_spins, ecp);
              if (g.D_alpha.size() == 0) return py::none();
              return py::make_tuple(g.D_alpha, g.D_beta);
          },
          py::arg("mol"), py::arg("basis"),
          py::arg("n_alpha"), py::arg("n_beta"),
          py::arg("kind"), py::arg("is_periodic") = true,
          py::arg("atomic_spins") = std::vector<int>{},
          py::arg("overlap") = py::none(),
          py::arg("ecp_context") = vibeqc::GuessECPContext{},
          "Open-shell per-spin initial densities via GuessEngine. "
          "Returns None for HCORE. ``atomic_spins`` (per-atom +1/-1/0) "
          "seeds an ATOMSPIN broken-symmetry SAD start.");

    m.def("_guess_closed_shell_density_with_jk",
          [](const vibeqc::Molecule& mol, const vibeqc::BasisSet& basis,
             int n_occ, vibeqc::InitialGuess kind,
             const Eigen::MatrixXd& S, const Eigen::MatrixXd& Hcore,
             const std::shared_ptr<vibeqc::JKBuilder>& jk,
             double linear_dep_threshold, const vibeqc::GuessECPContext& ecp) -> py::object {
              if (!jk) throw std::invalid_argument(
                  "_guess_closed_shell_density_with_jk: jk_builder must "
                  "not be None");
              vibeqc::SystemHints hints;
              auto g = vibeqc::GuessEngine::build_closed_shell(
                  mol, basis, n_occ, kind, &S, &Hcore, jk.get(), hints,
                  linear_dep_threshold, ecp);
              if (g.D.size() == 0) return py::none();
              return py::cast(g.D);
          },
          py::arg("mol"), py::arg("basis"), py::arg("n_occ"),
          py::arg("kind"), py::arg("S"), py::arg("Hcore"),
          py::arg("jk_builder"), py::arg("linear_dep_threshold") = 1e-7,
          py::arg("ecp_context") = vibeqc::GuessECPContext{},
          "Molecular closed-shell initial density via GuessEngine with the "
          "overlap, Hcore, and JKBuilder required by Fock-mode guesses.");

    m.def("_guess_open_shell_density_with_jk",
          [](const vibeqc::Molecule& mol, const vibeqc::BasisSet& basis,
             int n_alpha, int n_beta, vibeqc::InitialGuess kind,
             const Eigen::MatrixXd& S, const Eigen::MatrixXd& Hcore,
             const std::shared_ptr<vibeqc::JKBuilder>& jk,
             bool use_jk_for_density_mode,
             const std::vector<int>& atomic_spins,
             double linear_dep_threshold, const vibeqc::GuessECPContext& ecp) -> py::object {
              if (!jk) throw std::invalid_argument(
                  "_guess_open_shell_density_with_jk: jk_builder must not "
                  "be None");
              vibeqc::SystemHints hints;
              hints.is_open_shell = true;
              const vibeqc::InitialGuess effective_kind =
                  vibeqc::GuessEngine::resolve_auto_for_molecule(
                      mol, kind, /*is_periodic=*/false,
                      /*is_open_shell=*/true);
              const vibeqc::JKBuilder* guess_jk =
                  (use_jk_for_density_mode
                   || effective_kind == vibeqc::InitialGuess::PATOM)
                      ? jk.get() : nullptr;
              auto g = vibeqc::GuessEngine::build_open_shell(
                  mol, basis, n_alpha, n_beta, kind, &S, &Hcore, guess_jk,
                  hints, linear_dep_threshold,
                  atomic_spins.empty() ? nullptr : &atomic_spins, ecp);
              if (g.D_alpha.size() == 0) return py::none();
              return py::make_tuple(g.D_alpha, g.D_beta);
          },
          py::arg("mol"), py::arg("basis"), py::arg("n_alpha"),
          py::arg("n_beta"), py::arg("kind"), py::arg("S"),
          py::arg("Hcore"), py::arg("jk_builder"),
          py::arg("use_jk_for_density_mode"),
          py::arg("atomic_spins") = std::vector<int>{},
          py::arg("linear_dep_threshold") = 1e-7,
          py::arg("ecp_context") = vibeqc::GuessECPContext{},
          "Molecular open-shell initial densities via GuessEngine with the "
          "overlap, Hcore, and JKBuilder required by Fock-mode guesses.");

    py::enum_<vibeqc::DIISDepthPolicy>(m, "DIISDepthPolicy",
          "History-depth management policy for the DIIS accelerator. "
          "FIXED = Pulay's classic FIFO window (default). RESTART = "
          "Chupin et al. 2021 (ESAIM: M2AN 55, 2785) Algorithm 3: grow "
          "the depth until the stored error differences become nearly "
          "linearly dependent, then restart (τ = restart parameter). "
          "ADAPTIVE = their Algorithm 4: keep only recent iterates whose "
          "residual is within 1/δ of the current one.")
        .value("FIXED",    vibeqc::DIISDepthPolicy::FIXED)
        .value("RESTART",  vibeqc::DIISDepthPolicy::RESTART)
        .value("ADAPTIVE", vibeqc::DIISDepthPolicy::ADAPTIVE);

    py::class_<vibeqc::DIIS>(m, "DIIS",
          "Pulay's Direct Inversion of the Iterative Subspace SCF "
          "accelerator (Chem. Phys. Lett. 73, 393 (1980); J. Comput. "
          "Chem. 3, 556 (1982)). Maintains a rolling history of "
          "(Fock, error) pairs; on each call solves the small (n+1)-"
          "dim Pulay linear system for coefficients c_i that minimise "
          "‖Σ c_i e_i‖ subject to Σ c_i = 1, and returns the "
          "extrapolated Fock Σ c_i F_i. Used internally by every "
          "molecular and periodic SCF driver — this Python binding "
          "exists so periodic Python SCF loops can call into the "
          "same canonical implementation instead of re-deriving "
          "Pulay's recurrence in numpy. ``policy`` selects the "
          "adaptive-depth variants of Chupin et al. 2021 (RESTART / "
          "ADAPTIVE); ``adaptive_param`` is their τ (RESTART) or δ "
          "(ADAPTIVE) and is ignored for FIXED.")
        .def(py::init<std::size_t, vibeqc::DIISDepthPolicy, double>(),
             py::arg("max_subspace") = 8,
             py::arg("policy") = vibeqc::DIISDepthPolicy::FIXED,
             py::arg("adaptive_param") = 1.0e-4,
             "Construct an empty DIIS with the given history cap "
             "(must be ≥ 2; default 8), depth policy (default FIXED), "
             "and adaptive parameter (τ/δ; default 1e-4).")
        .def("clear", &vibeqc::DIIS::clear,
             "Drop all (F, error) history. Used for level-shift "
             "warm-up restarts and similar SCF resets.")
        .def_property_readonly("subspace_size",
             &vibeqc::DIIS::subspace_size,
             "Current history size (0 immediately after construction "
             "or ``clear()``; capped at ``max_subspace``).")
        .def_property_readonly("policy", &vibeqc::DIIS::policy,
             "The DIISDepthPolicy this instance was constructed with.")
        .def("extrapolate", &vibeqc::DIIS::extrapolate,
             py::arg("fock"), py::arg("error"),
             "Append ``(fock, error)`` to the history and return the "
             "Pulay-extrapolated Fock Σ_i c_i F_i. Returns ``fock`` "
             "unchanged before the history has at least two entries. "
             "``error`` is conventionally the commutator "
             "``F D S − S D F``, evaluated at the density that "
             "produced ``fock``.")
        .def("extrapolate_spin_coupled",
             &vibeqc::DIIS::extrapolate_spin_coupled,
             py::arg("fock_alpha"), py::arg("fock_beta"),
             py::arg("error_alpha"), py::arg("error_beta"),
             "Spin-coupled open-shell extrapolation: one B matrix built "
             "from the stacked (e_α; e_β) error and a single coefficient "
             "set applied to both Fock matrices. Returns "
             "(Σ c_i F_α,i, Σ c_i F_β,i).")
        .def("extrapolate_blocks", &vibeqc::DIIS::extrapolate_blocks,
             py::arg("fock_blocks"), py::arg("error_blocks"),
             "Block-vector extrapolation. Stores an (F, error) iterate "
             "whose Fock and error are each a list of real blocks, and "
             "returns Σ_i c_i F_i block-by-block. The Pulay inner "
             "product ⟨e_i | e_j⟩ = Σ_b (e_i[b] ⊙ e_j[b]).sum() sums "
             "over all blocks, so a multi-k periodic history whose "
             "blocks are √w_k·Re M(k) and √w_k·Im M(k) reproduces the "
             "k-weighted form Σ_k w_k Re Tr[e_i(k)† e_j(k)] exactly. "
             "Fock and error block lists are independent histories and "
             "may differ in shape (KDIIS pairs nbf×nbf Fock blocks with "
             "n_vir×n_occ gradient blocks). Open-shell histories append "
             "the β blocks to both lists. Block layout must be stable "
             "across pushed iterates. Returns ``fock_blocks`` unchanged "
             "when the history holds fewer than two iterates.");

    py::enum_<vibeqc::SCFAccelerator>(m, "SCFAccelerator",
          "SCF Fock-extrapolation accelerator. "
          "DIIS = Pulay's commutator DIIS (historical default); "
          "KDIIS = orbital-rotation gradient error vector (Kollmar, Int. "
          "J. Quantum Chem. 62, 617 (1997); ORCA's opt-in ``!KDIIS``); "
          "EDIIS = energy-DIIS (Kudin / Scuseria / Cancès, J. Chem. "
          "Phys. 116, 8255 (2002)); "
          "EDIIS_DIIS = production hybrid (Garza / Scuseria, J. Chem. "
          "Phys. 137, 054110 (2012)) — EDIIS while the commutator "
          "norm is above ``ediis_diis_switch_threshold``, then plain "
          "DIIS for the asymptotic regime; "
          "ADIIS = augmented-Roothaan-Hall DIIS (Hu & Yang, J. Chem. "
          "Phys. 132, 054109 (2010)) — EDIIS sibling, near-identical "
          "at the HF level; "
          "R_CDIIS = restarted commutator-DIIS (Chupin / Dupuy / "
          "Legendre / Séré, ESAIM: M2AN 55, 2785 (2021)) — adaptive "
          "depth with restart on near-linear-dependence (τ = "
          "``diis_restart_tau``); "
          "AD_CDIIS = adaptive-depth commutator-DIIS (same paper) — "
          "depth shrinks near convergence via a residual-ratio test "
          "(δ = ``diis_adaptive_delta``).")
        .value("DIIS",       vibeqc::SCFAccelerator::DIIS)
        .value("KDIIS",      vibeqc::SCFAccelerator::KDIIS)
        .value("EDIIS",      vibeqc::SCFAccelerator::EDIIS)
        .value("EDIIS_DIIS", vibeqc::SCFAccelerator::EDIIS_DIIS)
        .value("ADIIS",      vibeqc::SCFAccelerator::ADIIS)
        .value("ADIIS_DIIS", vibeqc::SCFAccelerator::ADIIS_DIIS)
        .value("R_CDIIS",    vibeqc::SCFAccelerator::R_CDIIS)
        .value("AD_CDIIS",   vibeqc::SCFAccelerator::AD_CDIIS);

    py::enum_<vibeqc::SCFMode>(m, "SCFMode",
          "SCF Fock-build mode. CONVENTIONAL = in-core 4-index ERI "
          "tensor (fast for ≤~150 BF, O(n_bf⁴) memory). DIRECT = "
          "on-the-fly Schwarz-screened libint quartet evaluation "
          "(the only path that survives at >250 BF). AUTO (default) "
          "selects on basis size (DIRECT when n_bf > "
          "scf_mode_auto_threshold). Orthogonal to density_fit + "
          "cosx — those supersede it when enabled.")
        .value("AUTO",         vibeqc::SCFMode::AUTO)
        .value("CONVENTIONAL", vibeqc::SCFMode::CONVENTIONAL)
        .value("DIRECT",       vibeqc::SCFMode::DIRECT);

    py::enum_<vibeqc::CosxVariant>(m, "CosxVariant",
          "COSX K-matrix variant selector.")
        .value("AUTO",     vibeqc::CosxVariant::AUTO)
        .value("STANDARD", vibeqc::CosxVariant::STANDARD)
        .value("FITTED",   vibeqc::CosxVariant::FITTED);

    py::class_<vibeqc::EDIIS>(m, "EDIIS",
          "Energy-DIIS extrapolator (Kudin / Scuseria / Cancès, "
          "J. Chem. Phys. 116, 8255 (2002)). Minimises a quadratic energy "
          "functional on the convex hull of stored (F, D) pairs subject "
          "to Σ c_i = 1, c_i ≥ 0 — a small simplex-constrained QP. The "
          "SCF drivers use this internally when scf_accelerator = EDIIS "
          "or EDIIS_DIIS; the class is exposed primarily for testing "
          "the coefficient solver against hand-constructed inputs. The "
          "exact nonconvex simplex solve supports max_subspace 2..12; "
          "larger values fail on first use.")
        .def(py::init<std::size_t>(), py::arg("max_subspace") = 8)
        .def("clear", &vibeqc::EDIIS::clear)
        .def("subspace_size", &vibeqc::EDIIS::subspace_size)
        .def("last_coeffs",
             [](const vibeqc::EDIIS& e) { return e.last_coeffs(); },
             "Coefficients from the most recent extrapolate() call. "
             "Empty before the first call. Sum to 1, all ≥ 0.")
        .def("last_extrapolated_density",
             &vibeqc::EDIIS::last_extrapolated_density,
             "The density whose Fock the most recent extrapolate() call "
             "returned: Σ_i c_i D_i per block, combined with the same "
             "coefficients as the Fock and over the history that "
             "survived the anti-replay guard. At the HF level the "
             "returned Fock is exactly F(this density), so a driver "
             "that couples the extrapolated Fock to density-derived "
             "projectors (the ROHF/ROKS Roothaan effective Fock) must "
             "use this density, never the current iterate's (GitLab "
             "#487). List of blocks; empty before the first call.")
        .def("replay_guard_erasures",
             &vibeqc::EDIIS::replay_guard_erasures,
             "Number of history entries the anti-replay guard has "
             "erased since construction / clear(). Diagnostic.")
        .def("discard_last_extrapolation",
             &vibeqc::EDIIS::discard_last_extrapolation,
             "Declare that the most recent extrapolate() return was NOT "
             "handed to the SCF (the EDIIS_DIIS hybrid took the DIIS "
             "branch, or the driver is still below diis_start_iter). The "
             "anti-replay guard keys on the previously CONSUMED return, "
             "so a dropped result must be retracted or a later first use "
             "of the same Fock is misread as a replay.")
        .def("extrapolate",
             py::overload_cast<const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                double>(&vibeqc::EDIIS::extrapolate),
             py::arg("fock"), py::arg("density"), py::arg("energy"),
             "Closed-shell extrapolation: store (F, D, energy) and "
             "return Σ_i c_i F_i. Returns F unchanged when n < 2.")
        .def("extrapolate_uhf",
             py::overload_cast<const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                double>(&vibeqc::EDIIS::extrapolate),
             py::arg("fock_alpha"), py::arg("fock_beta"),
             py::arg("density_alpha"), py::arg("density_beta"),
             py::arg("energy"),
             "Open-shell extrapolation: store the four matrices + total "
             "energy and return (Σ c_i F_α,i, Σ c_i F_β,i). The same "
             "coefficient set applies to both spins.")
        .def("extrapolate_blocks",
             py::overload_cast<const std::vector<Eigen::MatrixXd>&,
                                const std::vector<Eigen::MatrixXd>&,
                                double>(&vibeqc::EDIIS::extrapolate),
             py::arg("fock_blocks"), py::arg("density_blocks"),
             py::arg("energy"),
             "Block-vector extrapolation. Stores an (F, D, energy) "
             "iterate whose Fock and density are each a list of "
             "nbf×nbf real blocks and returns Σ_i c_i F_i block-by-"
             "block. The QP cross-term ⟨F_i | D_j⟩ = Σ_b (F_i[b] ⊙ "
             "D_j[b]).sum() sums over all blocks. Used by multi-k "
             "periodic SCF (blocks = real-space cells of a "
             "LatticeMatrixSet) and by the open-shell UHF kernel "
             "(blocks = α + β). Block layout (count, ordering, "
             "dimensions) must match across pushed iterates. Returns "
             "``fock_blocks`` unchanged when n < 2.");

    py::class_<vibeqc::ADIIS>(m, "ADIIS",
          "Augmented-Roothaan-Hall DIIS extrapolator (Hu & Yang, "
          "J. Chem. Phys. 132, 054109 (2010)). Minimises the ARH "
          "energy model expanded about the most recent iterate, on the "
          "convex hull of stored (F, D) pairs subject to Σ c_i = 1, "
          "c_i ≥ 0 — the same simplex-constrained QP solver as EDIIS. "
          "Needs no per-iterate energy. The SCF drivers use this when "
          "scf_accelerator = ADIIS or ADIIS_DIIS; the class is exposed "
          "primarily for "
          "testing the coefficient solver against hand-constructed "
          "inputs. The exact nonconvex simplex solve supports "
          "max_subspace 2..12; larger values fail on first use.")
        .def(py::init<std::size_t>(), py::arg("max_subspace") = 8)
        .def("clear", &vibeqc::ADIIS::clear)
        .def("subspace_size", &vibeqc::ADIIS::subspace_size)
        .def("last_coeffs",
             [](const vibeqc::ADIIS& a) { return a.last_coeffs(); },
             "Coefficients from the most recent extrapolate() call. "
             "Empty before the first call. Sum to 1, all ≥ 0.")
        .def("last_extrapolated_density",
             &vibeqc::ADIIS::last_extrapolated_density,
             "ADIIS sibling of EDIIS.last_extrapolated_density: "
             "Σ_i c_i D_i per block over the surviving history, with the "
             "coefficients of the most recent extrapolate() return "
             "(GitLab #487). List of blocks; empty before the first call.")
        .def("replay_guard_erasures",
             &vibeqc::ADIIS::replay_guard_erasures,
             "ADIIS sibling of EDIIS.replay_guard_erasures.")
        .def("discard_last_extrapolation",
             &vibeqc::ADIIS::discard_last_extrapolation,
             "ADIIS sibling of EDIIS.discard_last_extrapolation: declare "
             "that the most recent extrapolate() return was NOT handed "
             "to the SCF (ADIIS_DIIS took the DIIS branch, or the driver "
             "is below diis_start_iter).")
        .def("extrapolate",
             py::overload_cast<const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&>(
                 &vibeqc::ADIIS::extrapolate),
             py::arg("fock"), py::arg("density"),
             "Closed-shell extrapolation: store (F, D) and return "
             "Σ_i c_i F_i. Returns F unchanged when n < 2.")
        .def("extrapolate_uhf",
             py::overload_cast<const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&>(
                 &vibeqc::ADIIS::extrapolate),
             py::arg("fock_alpha"), py::arg("fock_beta"),
             py::arg("density_alpha"), py::arg("density_beta"),
             "Open-shell extrapolation: store the four matrices and "
             "return (Σ c_i F_α,i, Σ c_i F_β,i). The same coefficient "
             "set applies to both spins.")
        .def("extrapolate_blocks",
             py::overload_cast<const std::vector<Eigen::MatrixXd>&,
                                const std::vector<Eigen::MatrixXd>&>(
                 &vibeqc::ADIIS::extrapolate),
             py::arg("fock_blocks"), py::arg("density_blocks"),
             "Block-vector extrapolation. Stores an (F, D) iterate "
             "whose Fock and density are each a list of nbf×nbf real "
             "blocks and returns Σ_i c_i F_i block-by-block. The ARH "
             "trace cross-term ⟨D_i − D_n | F_j − F_n⟩ = Σ_b "
             "((D_i − D_n)[b] ⊙ (F_j − F_n)[b]).sum() sums over all "
             "blocks. Used by multi-k periodic SCF (blocks = real-"
             "space cells) and by the open-shell UHF kernel (blocks "
             "= α + β). Block layout must match across pushed "
             "iterates. Returns ``fock_blocks`` unchanged when "
             "n < 2.");

    py::class_<vibeqc::KDIIS>(m, "KDIIS",
          "Kollmar's DIIS — orbital-rotation-gradient DIIS (C. Kollmar, "
          "Int. J. Quantum Chem. 62, 617 (1997); ORCA's opt-in "
          "``!KDIIS`` keyword). Replaces the AO-basis commutator error "
          "e = F·D·S − S·D·F with the canonical-MO-basis occ-vir block "
          "g_{ai} = (C^T F C)_{ai}. The Pulay least-squares extrapolation "
          "is identical (Σ c_i F_i with Σ c_i = 1 minimising "
          "‖Σ c_i e_i‖_F); only the error metric differs. Particularly "
          "robust on transition-metal complexes and open-shell systems "
          "where the AO-commutator error is dominated by the wrong "
          "eigenmodes. The SCF drivers use this when "
          "scf_accelerator = KDIIS; the class is exposed primarily for "
          "testing and for Python periodic SCF loops to call into the "
          "canonical implementation instead of re-deriving the recurrence "
          "in numpy.")
        .def(py::init<std::size_t>(), py::arg("max_subspace") = 8)
        .def("clear", &vibeqc::KDIIS::clear)
        .def_property_readonly("subspace_size",
             &vibeqc::KDIIS::subspace_size)
        .def("extrapolate",
             py::overload_cast<const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::VectorXd&,
                                int>(&vibeqc::KDIIS::extrapolate),
             py::arg("fock"), py::arg("mo_coeffs"), py::arg("mo_energies"),
             py::arg("n_occ"),
             "Closed-shell extrapolation. Builds g_{ai} = (C^T F C)_{ai} "
             "(occ-vir block in the canonical MO basis), pushes (F, g) "
             "onto the rolling history, solves the Pulay least-squares, "
             "and returns the extrapolated Fock in the AO basis. Returns "
             "``fock`` unchanged before the history has at least two "
             "entries.")
        .def("extrapolate_uhf",
             py::overload_cast<const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::MatrixXd&,
                                const Eigen::VectorXd&,
                                const Eigen::VectorXd&,
                                int, int>(&vibeqc::KDIIS::extrapolate),
             py::arg("fock_alpha"), py::arg("fock_beta"),
             py::arg("mo_coeffs_alpha"), py::arg("mo_coeffs_beta"),
             py::arg("mo_energies_alpha"), py::arg("mo_energies_beta"),
             py::arg("n_alpha"), py::arg("n_beta"),
             "Open-shell extrapolation. Builds per-spin orbital-gradient "
             "errors, concatenates them into a single error matrix for "
             "the Pulay B-matrix solve, and returns the extrapolated "
             "(F_α, F_β) pair. The same coefficient set applies to both "
             "spins (spin-coupled extrapolation, matching the EDIIS/ADIIS "
             "open-shell convention).");

    m.def("sad_density", &vibeqc::sad_density,
          py::arg("molecule"), py::arg("basis"),
          py::arg("ecp_context") = vibeqc::GuessECPContext{},
          "Superposition of atomic densities (SAD) initial-guess density "
          "matrix. Builds an isolated-atom RHF SCF (with fractional Aufbau "
          "occupations) for each unique element in the molecule, then "
          "assembles the full molecular density matrix block-diagonally "
          "by AO range. The atomic SCFs are cached per (Z, basis_name) "
          "for the lifetime of the call. Returns a (n_bf, n_bf) numpy "
          "array. tr(D · S) ≈ n_electrons. Standard reference: "
          "Van Lenthe et al., J. Comput. Chem. 27, 926 (2006).");

    py::class_<vibeqc::RHFOptions>(m, "RHFOptions")
        .def(py::init<>())
        .def(py::init<const vibeqc::RHFOptions&>(), py::arg("other"),
             "Value copy. Used by routes (CPCM solvation, SAD retries) that "
             "must mutate a per-call options view without touching the "
             "caller's object, including the issue-#144 stability fields.")
        .def_property(
            "max_iter",
            [](const vibeqc::RHFOptions& options) { return options.max_iter; },
            [](vibeqc::RHFOptions& options, py::handle value) {
                set_molecular_scf_max_iter(options, value, "RHFOptions");
            })
        .def_readwrite("conv_tol_energy",
                       &vibeqc::RHFOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad", &vibeqc::RHFOptions::conv_tol_grad)
        .def_readwrite("damping", &vibeqc::RHFOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::RHFOptions::dynamic_damping,
                       "Adaptive density-mixing α (Zerner-Hehenberger 1979). "
                       "When True, ``damping`` is the initial α; the SCF "
                       "loop adjusts it within "
                       "[``dynamic_damping_min``, ``dynamic_damping_max``] "
                       "based on energy decrease. Default True "
                       "(ORCA-style adaptive damping).")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::RHFOptions::dynamic_damping_min,
                       "Lower bound for dynamic-damping α. Default 0.0.")
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::RHFOptions::dynamic_damping_max,
                       "Upper bound for dynamic-damping α. Default 0.95.")
        .def_readwrite("fock_mixing",
                       &vibeqc::RHFOptions::fock_mixing,
                       "Fock/Kohn-Sham matrix mixing fraction. This is "
                       "the weight of the previous Fock matrix in the "
                       "matrix diagonalised to form the next density. "
                       "CRYSTAL FMIXING 30 corresponds to 0.30. "
                       "Default 0.0 (off).")
        .def_readwrite("use_diis", &vibeqc::RHFOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::RHFOptions::diis_start_iter)
        .def_readwrite("diis_subspace_size",
                       &vibeqc::RHFOptions::diis_subspace_size)
        .def_readwrite("scf_accelerator",
                       &vibeqc::RHFOptions::scf_accelerator,
                       "SCF Fock-extrapolation accelerator. See "
                       "SCFAccelerator. Default EDIIS_DIIS; "
                       "this production hybrid is "
                       "recommended by Garza / Scuseria 2012 for "
                       "transition metals and broken-symmetry cases.")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::RHFOptions::ediis_diis_switch_threshold,
                       "EDIIS → DIIS commutator-norm threshold for "
                       "SCFAccelerator.EDIIS_DIIS. Use EDIIS while "
                       "‖F D S − S D F‖_F is above this value; switch "
                       "to plain DIIS once below. Default 1e-1.")
        .def_readwrite("diis_restart_tau",
                       &vibeqc::RHFOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ ∈ (0,1) (Chupin et "
                       "al. 2021). Used only for scf_accelerator=R_CDIIS; "
                       "smaller ⇒ larger mean DIIS depth. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::RHFOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ > 0 (Chupin et al. "
                       "2021). Used only for scf_accelerator=AD_CDIIS; "
                       "smaller ⇒ larger mean DIIS depth. Default 1e-4.")
        .def_readwrite("initial_guess",
                       &vibeqc::RHFOptions::initial_guess)
        .def_readwrite("read_density",
                       &vibeqc::RHFOptions::read_density,
                       "Starting density for initial_guess=READ (closed-"
                       "shell, in the current AO basis). Normally set by the "
                       "vibeqc.run_rhf wrapper from read_path / read_from, "
                       "not by hand.")
        .def_readwrite("read_path",
                       &vibeqc::RHFOptions::read_path,
                       "READ source file (.qvf / .molden) for the run_rhf "
                       "wrapper; empty = restart from an in-memory read_from.")
        .def_readwrite("linear_dep_threshold",
                       &vibeqc::RHFOptions::linear_dep_threshold)
        .def_readwrite("ecp_centers",
                       &vibeqc::RHFOptions::ecp_centers,
                       "List[ECPCenter] — atoms carrying an effective "
                       "core potential. Empty (default) = all-electron "
                       "calculation. See Phase 14c.")
        .def_readwrite("ecp_library",
                       &vibeqc::RHFOptions::ecp_library,
                       "ECP XML library name (default 'ecp10mdf' when "
                       "ecp_centers is non-empty and this is empty).")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::RHFOptions::ecp_primitive_blocks,
                       "Inline primitive ECP blocks. Mutually exclusive with "
                       "ecp_centers/ecp_library on high-level SCF routes.")
        .def_readwrite("ecp_primitive_centers",
                       &vibeqc::RHFOptions::ecp_primitive_centers,
                       "Cartesian centers (bohr) matching ecp_primitive_blocks.")
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::RHFOptions::ecp_effective_charges,
                       "Per-atom effective nuclear charges for inline ECPs.")
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::RHFOptions::ecp_total_ncore,
                       "Total number of core electrons replaced by inline ECPs.")
        .def_readwrite("level_shift",
                       &vibeqc::RHFOptions::level_shift,
                       "Phase C1a-2 — Saunders-Hillier level shift "
                       "(Hartree). Adds F + b·S − (b/2)·S·D·S to F "
                       "before diagonalisation, raising virtual orbital "
                       "eigenvalues by b. Inert at the converged "
                       "density (SCF fixed point unchanged). Useful "
                       "for small-HOMO-LUMO-gap molecules where DIIS "
                       "oscillates. Default 0.0 (no shift). Typical "
                       "values: 0.1 – 0.5 Hartree.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::RHFOptions::level_shift_warmup_cycles,
                       "Auto-reducing level-shift warm-up. -1 = auto "
                       "(shifted for up to five startup cycles, then "
                       "released with ≥1 unshifted tail cycle); 0 = "
                       "persistent (held every iteration, the legacy "
                       "behaviour); N > 0 = explicit warm-up length. "
                       "Default -1. Ignored when level_shift == 0 or when "
                       "level_shift_schedule is non-empty.")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::RHFOptions::level_shift_schedule,
                       "Explicit per-iteration Saunders-Hillier shift "
                       "curve (CRYSTAL LEVSHIFT B IRESET style). Empty ⇒ "
                       "derive from level_shift + level_shift_warmup_cycles. "
                       "Non-empty ⇒ use directly: entry i is the shift at "
                       "iteration i+1, the last entry is reused for later "
                       "iterations (set it to 0.0 to release). Lower a "
                       "vibeqc.LevelShiftSchedule via .as_list().")
        .def_readwrite("auto_level_shift_on_oscillation",
                       &vibeqc::RHFOptions::auto_level_shift_on_oscillation,
                       "Auto level-shift on SCF oscillation. When True "
                       "(default), the SCF loop monitors the energy history "
                       "and automatically engages a persistent "
                       "Saunders-Hillier level shift when oscillation is "
                       "detected. This protects transition-metal systems, "
                       "heterocycles, and small-gap organics from DIIS "
                       "limit-cycles. Mirrors the ROHF Python driver's "
                       "auto_level_shift_on_oscillation.")
        .def_readwrite("auto_level_shift_value",
                       &vibeqc::RHFOptions::auto_level_shift_value,
                       "Level shift magnitude (Hartree) engaged when "
                       "auto_level_shift_on_oscillation detects oscillation. "
                       "Default 0.3 Ha.")
        .def_readwrite("oscillation_detect_start_iter",
                       &vibeqc::RHFOptions::oscillation_detect_start_iter,
                       "First iteration at which the oscillation detector "
                       "activates. Default 10.")
        .def_readwrite("oscillation_window",
                       &vibeqc::RHFOptions::oscillation_window,
                       "Number of recent energy deltas examined for "
                       "oscillation. Default 10.")
        .def_readwrite("oscillation_min_sign_flips",
                       &vibeqc::RHFOptions::oscillation_min_sign_flips,
                       "Minimum dE sign flips within the oscillation "
                       "window to trigger auto-level-shift. Default 5.")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::RHFOptions::quadratic_fallback_iter,
                       "Phase C1c — second-order SCF fallback "
                       "activation iteration. When > 0, after this "
                       "iteration RHF switches from 'diagonalize F' to "
                       "a Newton step in MO space with diagonal-Hessian "
                       "preconditioning. Use for small-gap systems "
                       "where DIIS + level shift fail to converge. "
                       "Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::RHFOptions::quadratic_fallback_shift,
                       "Damping shift λ in the orbital-rotation "
                       "denominator (ε_a − ε_i + λ) of the C1c "
                       "Newton step. Default 0.1 Hartree. Larger "
                       "values produce more conservative steps.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::RHFOptions::quadratic_fallback_max_step,
                       "Trust-region cap on ‖κ‖ per C1c Newton "
                       "step. Default 0.1. Larger values produce "
                       "more aggressive rotations; smaller values "
                       "are safer.")
        .def_readwrite("newton_threshold",
                       &vibeqc::RHFOptions::newton_threshold,
                       "Phase D2c — Newton (full orbital-Hessian Newton "
                       "via preconditioned CG) activation threshold on "
                       "‖F D S − S D F‖_F. When > 0 and the gradient "
                       "drops below this value, the SCF replaces "
                       "diagonalize-F with a Newton step on the "
                       "orbital-rotation manifold. 1.0 (Fischer-Almlöf "
                       "1992 convention) is a reasonable production "
                       "default. Default 0.0 (disabled — SCF behaves "
                       "exactly as before).")
        .def_readwrite("newton_opts",
                       &vibeqc::RHFOptions::newton_opts,
                       "Knobs for the Newton step (CG cap, tolerance, "
                       "trust radius). See NewtonOptions.")
        .def_readwrite("soscf_threshold",
                       &vibeqc::RHFOptions::soscf_threshold,
                       "Phase D2d — Neese SOSCF (approximate second-"
                       "order SCF: augmented-Hessian step with diagonal-"
                       "dominant Hessian) activation threshold on "
                       "‖F D S − S D F‖_F. Cheaper per step than Newton "
                       "(no Fock build), linearly convergent. Mutually "
                       "exclusive with Newton; if both thresholds are "
                       "set, Newton wins. Default 0.0 (disabled; opt-in "
                       "L-BFGS quasi-Newton finalizer when > 0).")
        .def_readwrite("soscf_opts",
                       &vibeqc::RHFOptions::soscf_opts,
                       "Knobs for the SOSCF step (trust radius). See "
                       "SOSCFOptions.")
        .def_readwrite("trah_threshold",
                       &vibeqc::RHFOptions::trah_threshold,
                       "Phase D2e TRAH (Trust-Region Augmented Hessian; "
                       "Helmich-Paris 2022) activation threshold on "
                       "‖F D S − S D F‖_F. Full Hessian like Newton, "
                       "with an adaptive trust radius driven by Powell's "
                       "rho test. Mutual exclusion with Newton + SOSCF "
                       "(priority: quadratic > Newton > TRAH > SOSCF). "
                       "Default 0.0 (disabled).")
        .def_readwrite("trah_opts",
                       &vibeqc::RHFOptions::trah_opts,
                       "Knobs for the TRAH step (CG cap + trust-region "
                       "schedule). See TRAHOptions.")
        .def_readwrite("restart_opts",
                       &vibeqc::RHFOptions::restart_opts,
                       "Deterministic orbital-rotation restart options. "
                       "See SCFRestartOptions.")
        .def_readwrite("density_fit",
                       &vibeqc::RHFOptions::density_fit,
                       "Enable density fitting (RI) for the J + K "
                       "Fock build. Replaces the four-index ERI tensor "
                       "with the Cholesky-factorised B-tensor "
                       "(Whitten 1973, Eichkorn et al. 1995). Reduces "
                       "the per-iter Fock cost from O(n^4) to "
                       "O(n^2 · n_aux). Requires aux_basis to be set. "
                       "Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::RHFOptions::aux_basis,
                       "Auxiliary basis name for density fitting. "
                       "libint-recognised name, e.g. \"def2-svp-jk\", "
                       "\"cc-pvtz-jkfit\". Use "
                       "vibeqc.default_aux_basis_for(orbital_basis_name, "
                       "kind=\"jk\") for autodetection. Empty + "
                       "density_fit=True raises.")
        .def_readwrite("scf_mode",
                       &vibeqc::RHFOptions::scf_mode,
                       "SCF Fock-build mode (SCFMode). AUTO = pick by "
                       "basis size (default). CONVENTIONAL = in-core "
                       "4-index ERI. DIRECT = Schwarz-screened on-the-"
                       "fly libint quartet evaluation. Orthogonal to "
                       "density_fit + cosx.")
        .def_readwrite("scf_mode_auto_threshold",
                       &vibeqc::RHFOptions::scf_mode_auto_threshold,
                       "AUTO cutoff: pick DIRECT when n_bf > "
                       "threshold, else CONVENTIONAL. Default 140 (BUG 87).")
        .def_readwrite("schwarz_threshold",
                       &vibeqc::RHFOptions::schwarz_threshold,
                       "Per-quartet Schwarz skip bound for the DIRECT "
                       "kernel (|⟨μν|λσ⟩| ≤ Q_μν · Q_λσ). Default "
                       "1e-10 (ORCA convention). 0 disables screening.")
        .def_readwrite("incremental_fock",
                       &vibeqc::RHFOptions::incremental_fock,
                       "Enable Almlöf-style incremental ΔP Fock build "
                       "for the DIRECT path. Caches D_prev + G_2e_prev; "
                       "per iter, builds G_2e[ΔD] and adds to prev — "
                       "3-10× total-SCF speedup at large basis. "
                       "Default True (BUG 87).")
        .def_readwrite("incremental_fock_reset_freq",
                       &vibeqc::RHFOptions::incremental_fock_reset_freq,
                       "Full-rebuild frequency for incremental_fock — "
                       "every N iterations the ΔP cache is discarded "
                       "and rebuilt from scratch to bound floating-"
                       "point drift. Default 8 (Almlöf incremental-build reset frequency).")
        .def_readwrite("schwarz_threshold_loose",
                       &vibeqc::RHFOptions::schwarz_threshold_loose,
                       "Two-phase Schwarz threshold for DIRECT mode: "
                       "start at this looser cutoff (more aggressive "
                       "screening), tighten to ``schwarz_threshold`` "
                       "once SCF grad-norm < schwarz_threshold_tighten_at. "
                       "Default 1e-7 (ORCA convention); set ≤ "
                       "schwarz_threshold to disable.")
        .def_readwrite("schwarz_threshold_tighten_at",
                       &vibeqc::RHFOptions::schwarz_threshold_tighten_at,
                       "Gradient-norm cutoff that triggers the loose→"
                       "tight Schwarz transition. Default 1e-3.")
        .def_readwrite("cosx",
                       &vibeqc::RHFOptions::cosx,
                       "Use chain-of-spheres exchange (COSX) for the K "
                       "build. Pair with density_fit=True for the "
                       "standard RIJCOSX hybrid-DFT acceleration. "
                       "Default False.")
        .def_readwrite("cosx_grid",
                       &vibeqc::RHFOptions::cosx_grid,
                       "GridOptions for the COSX integration grid (only "
                       "used when cosx=True). Default = same as the DFT "
                       "XC defaults; tune for the speed-accuracy "
                       "trade-off.")
        .def_readwrite("cosx_variant",
                       &vibeqc::RHFOptions::cosx_variant,
                       "COSX K-matrix variant selector. AUTO resolves to "
                       "the production FITTED path unless explicitly "
                       "overridden.")
        .def_readwrite("cosx_grid_level",
                       &vibeqc::RHFOptions::cosx_grid_level,
                       "COSX GridX selector: -1 AUTO, 0 legacy cosx_grid, "
                       "1..4 explicit GridX tier.")
        .def_readwrite("dft_plus_u_sites",
                       &vibeqc::RHFOptions::dft_plus_u_sites,
                       "Internal: list of _HubbardSiteCxx (Hartree). "
                       "Populated by the vibeqc.run_rhf Python wrapper "
                       "from the user-facing dft_plus_u=[HubbardSite(...)] "
                       "kwarg. Empty by default — empty ≡ no +U.")
        .def_readwrite("dft_plus_u_ao_groups",
                       &vibeqc::RHFOptions::dft_plus_u_ao_groups,
                       "Internal: parallel array of AO-index lists, one "
                       "per Hubbard site. Precomputed once at SCF setup "
                       "(see python/vibeqc/dft_plus_u.py::ao_group_indices) "
                       "so the SCF loop pays the shell walk zero times.")
        .def_readwrite("use_davidson",
                       &vibeqc::RHFOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::RHFOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::RHFOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.")
        .def_property(
            "stability_check",
            [](const vibeqc::RHFOptions& opts) {
                return opts.stability_check;
            },
            [](vibeqc::RHFOptions& opts, bool enabled) {
                opts.stability_check = enabled;
                opts.stability_check_explicit = true;
            },
            "Restricted-stability VERDICT (issue #144): after a "
            "converged RHF SCF, report the lowest orbital-rotation "
            "Hessian eigenvalue at the returned solution (Seeger-Pople "
            "1977 singlet + triplet sectors, matrix-free Davidson). "
            "Verdict only — no rotation, escape, or promotion; what to "
            "do with a negative verdict is a pending maintainer "
            "decision. Default True; set False for legacy single-SCF "
            "behaviour. Setting it marks the request explicit, so an "
            "unsupported route fails closed instead of skipping "
            "silently.")
        .def_property_readonly(
            "_stability_check_explicit",
            [](const vibeqc::RHFOptions& opts) {
                return opts.stability_check_explicit;
            },
            "Internal flag distinguishing the automatic stability default "
            "from an explicit user request.")
        .def_readwrite("stability_tol",
                       &vibeqc::RHFOptions::stability_tol,
                       "Instability threshold: unstable when the lowest "
                       "Hessian eigenvalue < -stability_tol. Default 1e-4.")
        .def_readwrite("stability_davidson_max_iter",
                       &vibeqc::RHFOptions::stability_davidson_max_iter,
                       "Davidson matvec cap for the stability solve. "
                       "Default 60.");

    py::class_<vibeqc::NewtonOptions>(m, "NewtonOptions",
          "Knobs for the Newton Newton step (Phase D2c). See "
          "cpp/include/vibeqc/newton.hpp for the math + references "
          "(Bacskay 1981, Fischer/Almlöf 1992, Neese 2000, "
          "Helmich-Paris 2022).")
        .def(py::init<>())
        .def_readwrite("cg_max_iter", &vibeqc::NewtonOptions::cg_max_iter,
                       "CG iteration cap. 50 is generous; Newton normally "
                       "converges in 5-15 iters once active. Default 50.")
        .def_readwrite("cg_tol", &vibeqc::NewtonOptions::cg_tol,
                       "Relative residual tolerance on the CG solver: "
                       "‖A κ + g‖ ≤ tol · ‖g‖. Default 1e-4.")
        .def_readwrite("trust_radius", &vibeqc::NewtonOptions::trust_radius,
                       "Trust-region cap on ‖κ‖_F. Default 0.3.");

    py::class_<vibeqc::SOSCFOptions>(m, "SOSCFOptions",
          "Knobs for the Neese SOSCF step (Phase D2d) — augmented-"
          "Hessian eigsolve with diagonal-dominant orbital Hessian. "
          "See cpp/include/vibeqc/soscf.hpp for the math + reference "
          "(Neese 2000, Chem. Phys. Lett. 325, 93). Distinct from "
          "NewtonOptions (D2c, full Hessian via CG).")
        .def(py::init<>())
        .def_readwrite("trust_radius",
                       &vibeqc::SOSCFOptions::trust_radius,
                       "Trust-region cap on ‖κ‖_F. Default 0.3 — same "
                       "default as NewtonOptions for consistency.");

    py::class_<vibeqc::TRAHOptions>(m, "TRAHOptions",
          "Knobs for the TRAH (Trust-Region Augmented Hessian) step "
          "(Phase D2e — Helmich-Paris 2022). Same full-Hessian CG as "
          "Newton; differs in the adaptive trust radius driven by "
          "Powell's rho test (actual / model-predicted dE). See "
          "cpp/include/vibeqc/trah.hpp.")
        .def(py::init<>())
        .def_readwrite("cg_max_iter", &vibeqc::TRAHOptions::cg_max_iter)
        .def_readwrite("cg_tol", &vibeqc::TRAHOptions::cg_tol)
        .def_readwrite("initial_trust_radius",
                       &vibeqc::TRAHOptions::initial_trust_radius)
        .def_readwrite("max_trust_radius",
                       &vibeqc::TRAHOptions::max_trust_radius)
        .def_readwrite("min_trust_radius",
                       &vibeqc::TRAHOptions::min_trust_radius)
        .def_readwrite("rho_shrink",
                       &vibeqc::TRAHOptions::rho_shrink)
        .def_readwrite("rho_expand",
                       &vibeqc::TRAHOptions::rho_expand)
        .def_readwrite("trust_shrink_factor",
                       &vibeqc::TRAHOptions::trust_shrink_factor)
        .def_readwrite("trust_expand_factor",
                       &vibeqc::TRAHOptions::trust_expand_factor)
        .def_readwrite("ms_max_iter", &vibeqc::TRAHOptions::ms_max_iter,
                       "More-Sorensen secular-root-find iteration cap for "
                       "the boundary level shift lambda.")
        .def_readwrite("ms_tol", &vibeqc::TRAHOptions::ms_tol,
                       "Relative tolerance on ||kappa(lambda)|| = Delta for "
                       "the More-Sorensen secular root-find.");

    py::class_<vibeqc::SCFRestartOptions>(m, "SCFRestartOptions",
          "Deterministic SCF restart via randomized orbital rotations. "
          "When the SCF stalls, rotates MOs within each subspace and "
          "restarts from the resulting density. See cpp/include/vibeqc/"
          "scf_restart.hpp.")
        .def(py::init<>())
        .def_readwrite("enabled",
                       &vibeqc::SCFRestartOptions::enabled,
                       "Enable deterministic restart. Default true.")
        .def_readwrite("max_stall_iters",
                       &vibeqc::SCFRestartOptions::max_stall_iters,
                       "Consecutive stall iterations before restart. Default 15.")
        .def_readwrite("seed",
                       &vibeqc::SCFRestartOptions::seed,
                       "PRNG seed for deterministic rotation. Default 42.")
        .def_readwrite("max_restarts",
                       &vibeqc::SCFRestartOptions::max_restarts,
                       "Maximum restarts per SCF run. Default 3.");

    py::class_<vibeqc::DavidsonOptions>(m, "DavidsonOptions",
          "Knobs for the Davidson iterative diagonalizer — a block-"
          "Davidson algorithm that extracts the lowest n_eig eigenpairs "
          "of a real symmetric matrix without full diagonalization. "
          "See cpp/include/vibeqc/davidson.hpp for the math + references "
          "(Davidson 1975; Liu 1978; Kresse & Furthmueller 1996).")
        .def(py::init<>())
        .def_readwrite("n_eig", &vibeqc::DavidsonOptions::n_eig,
                       "Number of lowest eigenvalues to extract. "
                       "0 = auto (n_kept orthogonal basis functions).")
        .def_readwrite("n_guess", &vibeqc::DavidsonOptions::n_guess,
                       "Initial subspace dimension. 0 = auto "
                       "(max(n_eig + 5, 2*n_eig)).")
        .def_readwrite("max_subspace", &vibeqc::DavidsonOptions::max_subspace,
                       "Subspace size budget before collapse. "
                       "0 = auto (min(6*n_eig, n_basis)).")
        .def_readwrite("conv_tol", &vibeqc::DavidsonOptions::conv_tol,
                       "Convergence threshold on residual norm "
                       "||r_i||_2. Default 1e-7.")
        .def_readwrite("max_iter", &vibeqc::DavidsonOptions::max_iter,
                       "Maximum outer (subspace-expansion) iterations. "
                       "Default 200.")
        .def_readwrite("max_refine", &vibeqc::DavidsonOptions::max_refine,
                       "Maximum inner lock-refinement iterations. Default 5.")
        .def_readwrite("preshift", &vibeqc::DavidsonOptions::preshift,
                       "Preconditioner shift (Hartree) to avoid divergence "
                       "when lambda_i is close to a diagonal element. "
                       "Default 1e-6.")
        .def_readwrite("verbosity", &vibeqc::DavidsonOptions::verbosity,
                       "0 = silent, 1 = summary, 2 = per-iteration.")
        .def_readwrite("guess_vectors",
                       &vibeqc::DavidsonOptions::guess_vectors,
                       "Optional initial guess vectors (n_basis x n_cols). "
                       "When non-empty, used as the starting subspace. "
                       "Key for SCF eigenvector recycling.")
        .def_readwrite("guess_vectors_cplx",
                       &vibeqc::DavidsonOptions::guess_vectors_cplx,
                       "Complex variant of guess_vectors for the "
                       "Hermitian solver.");

    py::class_<vibeqc::DavidsonResult>(m, "DavidsonResult",
          "Result of a real-symmetric Davidson diagonalization.")
        .def(py::init<>())
        .def_readonly("eigenvalues",
                      &vibeqc::DavidsonResult::eigenvalues)
        .def_readonly("eigenvectors",
                      &vibeqc::DavidsonResult::eigenvectors)
        .def_readonly("n_iter", &vibeqc::DavidsonResult::n_iter)
        .def_readonly("subspace_dim",
                      &vibeqc::DavidsonResult::subspace_dim)
        .def_readonly("converged",
                      &vibeqc::DavidsonResult::converged);

    py::class_<vibeqc::DavidsonResultComplex>(
        m, "DavidsonResultComplex",
        "Result of a complex-Hermitian Davidson diagonalization.")
        .def(py::init<>())
        .def_readonly("eigenvalues",
                      &vibeqc::DavidsonResultComplex::eigenvalues)
        .def_readonly("eigenvectors",
                      &vibeqc::DavidsonResultComplex::eigenvectors)
        .def_readonly("n_iter",
                      &vibeqc::DavidsonResultComplex::n_iter)
        .def_readonly("subspace_dim",
                      &vibeqc::DavidsonResultComplex::subspace_dim)
        .def_readonly("converged",
                      &vibeqc::DavidsonResultComplex::converged);

    m.def("davidson_solve", &vibeqc::davidson_solve,
          py::arg("A"), py::arg("options"),
          "Solve Ax = lambda x for the lowest eigenvalues of a real "
          "symmetric matrix using the blocked Davidson algorithm.");

    m.def("davidson_solve_matvec", &vibeqc::davidson_solve_matvec,
          py::arg("n_basis"), py::arg("matvec"), py::arg("diag"),
          py::arg("options"),
          "Solve Ax = lambda x for the lowest eigenvalues of a real "
          "symmetric matrix using a matrix-vector product callback.");

    m.def("davidson_solve_hermitian",
          &vibeqc::davidson_solve_hermitian,
          py::arg("A_H"), py::arg("options"),
          "Solve A x = lambda x for the lowest eigenvalues of a complex "
          "Hermitian matrix using the blocked Davidson algorithm.");

    m.def("davidson_solve_hermitian_matvec",
          &vibeqc::davidson_solve_hermitian_matvec,
          py::arg("n_basis"), py::arg("matvec"), py::arg("diag"),
          py::arg("options"),
          "Solve A x = lambda x for the lowest eigenvalues of a complex "
          "Hermitian matrix using a matrix-vector product callback.");

    m.def("davidson_summary",
          py::overload_cast<const vibeqc::DavidsonResult&>(
              &vibeqc::davidson_summary),
          py::arg("result"),
          "Return a one-line summary string for a DavidsonResult.");

    // ---- LOBPCG eigensolver (Knyazev 2001) ----
    py::class_<vibeqc::LOBPCGOptions>(m, "LOBPCGOptions",
          "Options for the LOBPCG eigensolver (Knyazev 2001) — a "
          "block preconditioned conjugate-gradient algorithm that "
          "extracts the lowest n_eig eigenpairs of a real symmetric "
          "matrix using a 3-term recurrence.  See "
          "cpp/include/vibeqc/lobpcg.hpp for the algorithm and references.")
        .def(py::init<>())
        .def_readwrite("n_eig", &vibeqc::LOBPCGOptions::n_eig,
                       "Number of lowest eigenvalues to extract.  Default 5.")
        .def_readwrite("block_size", &vibeqc::LOBPCGOptions::block_size,
                       "Block size (number of trial vectors).  0 = auto "
                       "(2 * n_eig).  Must be >= n_eig.")
        .def_readwrite("max_iter", &vibeqc::LOBPCGOptions::max_iter,
                       "Maximum outer iterations.  Default 200.")
        .def_readwrite("tol", &vibeqc::LOBPCGOptions::tol,
                       "Convergence threshold on residual norm "
                       "||r_i||_2.  Default 1e-7.")
        .def_readwrite("verbosity", &vibeqc::LOBPCGOptions::verbosity,
                       "0 = silent, 1 = summary, 2 = per-iteration.")
        .def_readwrite("preshift", &vibeqc::LOBPCGOptions::preshift,
                       "Preconditioner shift to avoid blow-up when "
                       "lambda_i is close to a diagonal element.  "
                       "Default 1e-6.");

    py::class_<vibeqc::LOBPCGResult>(m, "LOBPCGResult",
          "Result of a LOBPCG diagonalization.")
        .def(py::init<>())
        .def_readonly("eigenvalues",
                      &vibeqc::LOBPCGResult::eigenvalues)
        .def_readonly("eigenvectors",
                      &vibeqc::LOBPCGResult::eigenvectors)
        .def_readonly("n_iter", &vibeqc::LOBPCGResult::n_iter)
        .def_readonly("converged",
                      &vibeqc::LOBPCGResult::converged);

    m.def("lobpcg_solve", &vibeqc::lobpcg_solve,
          py::arg("A"), py::arg("options"),
          "Solve Ax = lambda x for the lowest eigenvalues of a real "
          "symmetric matrix using the LOBPCG algorithm (Knyazev 2001).");

    m.def("lobpcg_solve_matvec", &vibeqc::lobpcg_solve_matvec,
          py::arg("n"), py::arg("matvec"), py::arg("diag"),
          py::arg("options"),
          "Matrix-free LOBPCG: matvec(v) returns A·v.  diag is the "
          "diagonal of A (for the Jacobi preconditioner).");



    // ---- Jacobi-Davidson (Phase D4) ------------------------------------
    py::class_<vibeqc::JDOptions>(m, "JDOptions",
          "Options for the Jacobi-Davidson eigensolver "
          "(Sleijpen & Van der Vorst 1996).")
        .def(py::init<>())
        .def_readwrite("n_eig", &vibeqc::JDOptions::n_eig)
        .def_readwrite("tol", &vibeqc::JDOptions::tol)
        .def_readwrite("max_iter", &vibeqc::JDOptions::max_iter)
        .def_readwrite("max_inner", &vibeqc::JDOptions::max_inner)
        .def_readwrite("inner_tol", &vibeqc::JDOptions::inner_tol)
        .def_readwrite("max_subspace", &vibeqc::JDOptions::max_subspace)
        .def_readwrite("sigma", &vibeqc::JDOptions::sigma,
                       "Target shift for interior eigenvalues. 0 = extremal.")
        .def_readwrite("preshift", &vibeqc::JDOptions::preshift)
        .def_readwrite("verbosity", &vibeqc::JDOptions::verbosity);

    py::class_<vibeqc::JDResult>(m, "JDResult",
          "Result of a Jacobi-Davidson diagonalization.")
        .def(py::init<>())
        .def_readonly("eigenvalues", &vibeqc::JDResult::eigenvalues)
        .def_readonly("eigenvectors", &vibeqc::JDResult::eigenvectors)
        .def_readonly("n_iter", &vibeqc::JDResult::n_iter)
        .def_readonly("subspace_dim", &vibeqc::JDResult::subspace_dim)
        .def_readonly("converged", &vibeqc::JDResult::converged);

    m.def("jd_solve", &vibeqc::jd_solve,
          py::arg("A"), py::arg("options"),
          "Jacobi-Davidson eigensolver (Sleijpen & Van der Vorst 1996): "
          "find eigenvalues of a real symmetric matrix using MINRES "
          "correction-equation solver.");


    // ---- GPLHR (Phase D5) ----------------------------------------------
    py::class_<vibeqc::GPLHROptions>(m, "GPLHROptions",
          "Options for the GPLHR interior eigensolver "
          "(Zuev et al. 2015).")
        .def(py::init<>())
        .def_readwrite("sigma", &vibeqc::GPLHROptions::sigma,
                       "Target shift for interior eigenvalues.")
        .def_readwrite("n_eig", &vibeqc::GPLHROptions::n_eig)
        .def_readwrite("block_size", &vibeqc::GPLHROptions::block_size)
        .def_readwrite("tol", &vibeqc::GPLHROptions::tol)
        .def_readwrite("max_iter", &vibeqc::GPLHROptions::max_iter)
        .def_readwrite("max_subspace", &vibeqc::GPLHROptions::max_subspace)
        .def_readwrite("preshift", &vibeqc::GPLHROptions::preshift)
        .def_readwrite("verbosity", &vibeqc::GPLHROptions::verbosity);

    py::class_<vibeqc::GPLHRResult>(m, "GPLHRResult")
        .def(py::init<>())
        .def_readonly("eigenvalues", &vibeqc::GPLHRResult::eigenvalues)
        .def_readonly("eigenvectors", &vibeqc::GPLHRResult::eigenvectors)
        .def_readonly("n_iter", &vibeqc::GPLHRResult::n_iter)
        .def_readonly("subspace_dim", &vibeqc::GPLHRResult::subspace_dim)
        .def_readonly("converged", &vibeqc::GPLHRResult::converged);

    m.def("gplhr_solve", &vibeqc::gplhr_solve,
          py::arg("A"), py::arg("options"),
          "GPLHR interior eigensolver (Zuev et al. 2015): "
          "find eigenvalues nearest sigma using harmonic Ritz extraction.");

    m.def("jd_summary",
          py::overload_cast<const vibeqc::JDResult&>(&vibeqc::jd_summary),
          py::arg("result"),
          "Return a one-line summary string for a JDResult.");

    m.def("lobpcg_summary", &vibeqc::lobpcg_summary,
          py::arg("result"),
          "Return a one-line summary string for a LOBPCGResult.");

    // ---- Brent 1-D minimisation ----
    py::class_<vibeqc::BrentResult>(m, "BrentResult")
        .def(py::init<>())
        .def_readonly("x_min", &vibeqc::BrentResult::x_min)
        .def_readonly("f_min", &vibeqc::BrentResult::f_min)
        .def_readonly("n_eval", &vibeqc::BrentResult::n_eval);

    m.def("brent_minimize", &vibeqc::brent_minimize,
          py::arg("f"), py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("tol") = 1e-5, py::arg("max_iter") = 100);

    py::class_<vibeqc::SCFIteration>(m, "SCFIteration")
        .def(py::init<>())
        .def(py::init([](int iter, double energy, double delta_e,
                          double grad_norm, int diis_subspace,
                          int newton_cg_iter, double trah_level_shift,
                          double wall_s, int accelerator_step) {
            vibeqc::SCFIteration it;
            it.iter = iter;
            it.energy = energy;
            it.delta_e = delta_e;
            it.grad_norm = grad_norm;
            it.diis_subspace = diis_subspace;
            it.newton_cg_iter = newton_cg_iter;
            it.trah_level_shift = trah_level_shift;
            it.wall_s = wall_s;
            it.accelerator_step = accelerator_step;
            return it;
        }),
            py::arg("iter") = 0,
            py::arg("energy") = 0.0,
            py::arg("delta_e") = 0.0,
            py::arg("grad_norm") = 0.0,
            py::arg("diis_subspace") = 0,
            py::arg("newton_cg_iter") = 0,
            py::arg("trah_level_shift") = 0.0,
            py::arg("wall_s") = 0.0,
            py::arg("accelerator_step") = 0,
            "Construct an SCFIteration record. Used by Python SCF drivers "
            "(periodic Ewald RHF / UHF / UKS) so every backend's "
            "``scf_trace`` carries the same item type — instead of plain "
            "tuples on the Ewald path and SCFIteration on the C++ path.")
        .def_readonly("iter", &vibeqc::SCFIteration::iter)
        .def_readonly("energy", &vibeqc::SCFIteration::energy,
                      "Total energy at this iteration (Hartree).")
        .def_readonly("delta_e", &vibeqc::SCFIteration::delta_e,
                      "E[k] - E[k-1]; 0 on the first iteration.")
        .def_readonly("grad_norm", &vibeqc::SCFIteration::grad_norm,
                      "||F D S - S D F||_Frobenius; zero at convergence.")
        .def_readonly("diis_subspace", &vibeqc::SCFIteration::diis_subspace,
                      "DIIS history size used this iteration (0 if off).")
        .def_readonly("accelerator_step",
                      &vibeqc::SCFIteration::accelerator_step,
                      "Fock extrapolation that actually reached the SCF this "
                      "iteration: 0 none, 1 Pulay DIIS (incl. the adaptive-"
                      "depth R_CDIIS / AD_CDIIS policies), 2 KDIIS, 3 EDIIS, "
                      "4 ADIIS. The EDIIS+DIIS / ADIIS+DIIS hybrids record the "
                      "branch their switch metric selected (GitLab #682).")
        .def_readonly("newton_cg_iter", &vibeqc::SCFIteration::newton_cg_iter,
                      "Newton CG iterations consumed this step (0 unless "
                      "the Newton Newton update was active this iter).")
        .def_readonly("trah_level_shift",
                      &vibeqc::SCFIteration::trah_level_shift,
                      "TRAH level shift λ applied this step. 0 unless the "
                      "TRAH update took a trust-region boundary step; "
                      "> 0 ⇒ the step landed on the ‖κ‖ = Δ boundary "
                      "with κ(λ) = −(A + λI)⁻¹g.")
        .def_readonly("wall_s", &vibeqc::SCFIteration::wall_s,
                      "Measured wall-clock seconds charged to this "
                      "iteration (time since the previous trace row). "
                      "Exactly 0.0 means unmeasured -- render as '--', "
                      "not as a 0.000 s measurement.")
        .def("__repr__", [](const vibeqc::SCFIteration& s) {
            return std::string{"SCFIteration(iter="} + std::to_string(s.iter)
                   + ", energy=" + std::to_string(s.energy)
                   + ", dE=" + std::to_string(s.delta_e)
                   + ", grad=" + std::to_string(s.grad_norm) + ")";
        });

    m.def(
        "_detect_scf_oscillation_for_testing",
        [](const std::vector<vibeqc::SCFIteration>& trace,
           int window,
           int min_sign_flips,
           double conv_tol_energy) {
            return vibeqc::detect_scf_oscillation(
                trace, window, min_sign_flips, conv_tol_energy);
        },
        py::arg("trace"),
        py::arg("window"),
        py::arg("min_sign_flips"),
        py::arg("conv_tol_energy"),
        "Private regression hook for the molecular SCF oscillation detector.");

    py::class_<vibeqc::RHFResult>(m, "RHFResult")
        .def_readonly("restart_basis", &vibeqc::RHFResult::restart_basis)
        .def_readwrite("guess_selection", &vibeqc::RHFResult::guess_selection)
        .def_readonly("energy", &vibeqc::RHFResult::energy,
                      "Total HF energy (Hartree).")
        .def_readonly("e_electronic", &vibeqc::RHFResult::e_electronic,
                      "Electronic energy = total - nuclear_repulsion.")
        .def_readonly("ecp_operator_applied",
                      &vibeqc::RHFResult::ecp_operator_applied,
                      "True when the SCF Hamiltonian included an ECP "
                      "operator, including a zero-core model potential.")
        .def_readonly("ecp_provenance_verified",
                      &vibeqc::RHFResult::ecp_provenance_verified,
                      "True when the high-level molecular driver built the "
                      "Hamiltonian and stamped exact ECP provenance; false "
                      "means unknown for a caller-supplied Hcore.")
        .def_readonly("ecp_xml_centers", &vibeqc::RHFResult::ecp_xml_centers,
                      "Exact XML ECP centers used by the SCF; empty for "
                      "all-electron and inline-primitive references.")
        .def_readonly("ecp_xml_library", &vibeqc::RHFResult::ecp_xml_library,
                      "Normalized XML ECP library used by the SCF; empty "
                      "for all-electron and inline-primitive references.")
        .def_readonly("ecp_primitive_blocks",
                      &vibeqc::RHFResult::ecp_primitive_blocks,
                      "Exact inline ECP primitive blocks used by the SCF.")
        .def_readonly("ecp_primitive_centers",
                      &vibeqc::RHFResult::ecp_primitive_centers,
                      "Inline ECP centers paired with ecp_primitive_blocks.")
        .def_readonly("ecp_effective_charges",
                      &vibeqc::RHFResult::ecp_effective_charges,
                      "Per-atom effective charges for the inline ECP route.")
        .def_readonly("ecp_total_ncore", &vibeqc::RHFResult::ecp_total_ncore,
                      "Physical core electrons replaced by the molecular "
                      "ECP used for this SCF reference; zero otherwise.")
        .def_readonly("e_dft_plus_u", &vibeqc::RHFResult::e_dft_plus_u,
                      "Dudarev DFT+U contribution to the total energy "
                      "(Hartree). 0 when no Hubbard sites are configured "
                      "in RHFOptions.dft_plus_u_sites.")
        .def_readonly("n_iter", &vibeqc::RHFResult::n_iter)
        .def_readonly("converged", &vibeqc::RHFResult::converged)
        .def_readonly("mo_energies", &vibeqc::RHFResult::mo_energies,
                      "Orbital energies (Hartree), ascending.")
        .def_readonly("mo_coeffs", &vibeqc::RHFResult::mo_coeffs,
                      "MO coefficient matrix: columns are MOs in the AO basis.")
        .def_readonly("density", &vibeqc::RHFResult::density,
                      "Converged density matrix D = 2 C_occ C_occ^T.")
        .def_readonly("fock", &vibeqc::RHFResult::fock,
                      "Converged Fock matrix F = Hcore + G(D).")
        .def_readonly("scf_trace", &vibeqc::RHFResult::scf_trace,
                      "Per-iteration SCF trace (list of SCFIteration).")
        .def_readonly("stability_checked",
                      &vibeqc::RHFResult::stability_checked,
                      "True when the restricted stability analysis ran on "
                      "the returned solution (verdict only; issue #144).")
        .def_readonly("stability_analysis_converged",
                      &vibeqc::RHFResult::stability_analysis_converged,
                      "True when the stability Davidson solve converged "
                      "(stability_eigenvalue is then trustworthy).")
        .def_readonly("stability_eigenvalue",
                      &vibeqc::RHFResult::stability_eigenvalue,
                      "Lowest orbital-rotation Hessian eigenvalue at the "
                      "returned solution; >= -stability_tol means stable.")
        .def_readonly("internal_instability",
                      &vibeqc::RHFResult::internal_instability,
                      "True when the returned solution is internally or "
                      "externally UNSTABLE (excited SCF solution) — "
                      "reported loudly by the output layer.")
        .def_readonly("stability_wall_s",
                      &vibeqc::RHFResult::stability_wall_s,
                      "Wall seconds spent in the stability phase.")
        .def_readonly("stability_cpu_s",
                      &vibeqc::RHFResult::stability_cpu_s,
                      "CPU seconds spent in the stability phase.")
        .def("__repr__", [](const vibeqc::RHFResult& r) {
            return std::string{"RHFResult(energy="}
                   + std::to_string(r.energy)
                   + ", n_iter=" + std::to_string(r.n_iter)
                   + ", converged=" + (r.converged ? "True" : "False") + ")";
        });

    m.def("run_rhf", &vibeqc::run_rhf,
          py::arg("molecule"), py::arg("basis"),
          py::arg("options") = vibeqc::RHFOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run restricted Hartree-Fock SCF on a closed-shell molecule. "
          "Returns an RHFResult.");

    m.def("run_rhf_scf_with_jk",
          [](const vibeqc::BasisSet& basis,
             int n_electrons,
             const Eigen::MatrixXd& S,
             const Eigen::MatrixXd& Hcore,
             double E_nuc,
             const std::shared_ptr<vibeqc::JKBuilder>& jk,
             const vibeqc::RHFOptions& options,
             const Eigen::MatrixXd& initial_density,
             const vibeqc::Molecule* molecule,
             const vibeqc::GuessSelection* guess_selection) {
              if (!jk) throw std::invalid_argument(
                  "run_rhf_scf_with_jk: jk_builder must not be None");
              return vibeqc::run_rhf_scf_with_jk(
                  basis, n_electrons, S, Hcore, E_nuc, *jk, options,
                  initial_density, molecule, guess_selection);
          },
          py::arg("basis"), py::arg("n_electrons"),
          py::arg("S"), py::arg("Hcore"), py::arg("E_nuc"),
          py::arg("jk_builder"),
          py::arg("options") = vibeqc::RHFOptions{},
          py::arg("initial_density") = Eigen::MatrixXd{},
          py::arg("molecule") = py::none(),
          py::arg("guess_selection") = py::none(),
          py::call_guard<py::gil_scoped_release>(),
          "Lower-level closed-shell RHF SCF entry point. Takes the AO "
          "overlap S, the one-electron Hamiltonian Hcore, the system's "
          "nuclear repulsion E_nuc, and a JKBuilder for the two-electron "
          "Fock piece. Drives the same SCF iteration loop as run_rhf "
          "(canonical orthogonalisation, DIIS, optional damping / level-"
          "shift / quadratic fallback). Use it to drive periodic-Γ "
          "RHF or any SCF where Hcore is built externally — pair with "
          "make_periodic_gamma_jk_builder for periodic-Γ inputs. "
          "Pass ``initial_density`` (closed-shell density matrix) to "
          "seed D directly — empty (default) falls back to "
          "diagonalising Hcore.");

    // ----- Gradient -------------------------------------------------------
    py::class_<vibeqc::GradientOptions>(m, "GradientOptions")
        .def(py::init<>())
        .def_readwrite("density_fit",
                       &vibeqc::GradientOptions::density_fit,
                       "Enable density fitting for the two-electron "
                       "gradient. Replaces the four-index ERI-derivative "
                       "path with the J + α_HF·K factorisation through "
                       "the precomputed B-tensor (Weigend 2002). Hcore, "
                       "overlap, and (for DFT) XC pieces are unchanged. "
                       "Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::GradientOptions::aux_basis,
                       "Auxiliary basis name for density fitting. "
                       "See RHFOptions.aux_basis.")
        .def_readwrite("cosx",
                       &vibeqc::GradientOptions::cosx,
                       "Use chain-of-spheres exchange (COSX) for the K "
                       "piece of the two-electron gradient. Pair with "
                       "density_fit=True for the standard RIJCOSX path. "
                       "Default False.")
        .def_readwrite("cosx_grid",
                       &vibeqc::GradientOptions::cosx_grid,
                       "GridOptions for the COSX integration grid (only "
                       "used when cosx=True). Sparser default than the "
                       "DFT XC grid; see "
                       "vibeqc.cosx.default_cosx_grid_options.")
        .def_readwrite("cosx_variant",
                       &vibeqc::GradientOptions::cosx_variant,
                       "COSX K-matrix variant selector for analytic "
                       "gradients.")
        .def_readwrite("cosx_grid_level",
                       &vibeqc::GradientOptions::cosx_grid_level,
                       "COSX GridX selector for analytic gradients: 0 uses "
                       "the legacy cosx_grid; positive values select GridX.")
        .def_readwrite("thresh_cosx",
                       &vibeqc::GradientOptions::thresh_cosx,
                       "COSX accuracy threshold mirrored from SCF options. "
                       "Used by AUTO cosx_variant resolution.")
        .def_readwrite("ecp_centers",
                       &vibeqc::GradientOptions::ecp_centers,
                       "ECP centers — must mirror the SCF call's "
                       "RHFOptions/RKSOptions/UHFOptions/UKSOptions::"
                       "ecp_centers so the gradient differentiates the "
                       "same Hamiltonian as the SCF energy. When non-"
                       "empty, the gradient drivers apply Z_eff = "
                       "Z − n_core to nuclear repulsion + V_ne pieces "
                       "and add the ∂V_ECP/∂R contribution.")
        .def_readwrite("ecp_library",
                       &vibeqc::GradientOptions::ecp_library,
                       "ECP library name (e.g. \"ecp10mdf\"). Must "
                       "match the SCF call's ecp_library. Defaults to "
                       "\"ecp10mdf\" when empty.")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::GradientOptions::ecp_primitive_blocks,
                       "Inline primitive ECP blocks. When non-empty these "
                       "take precedence over ecp_centers/ecp_library, "
                       "mirroring the SCF drivers' selection. Must mirror "
                       "the SCF call's ecp_primitive_blocks so the gradient "
                       "differentiates the same Hamiltonian as the energy.")
        .def_readwrite("ecp_primitive_centers",
                       &vibeqc::GradientOptions::ecp_primitive_centers,
                       "Cartesian centers (bohr) matching "
                       "ecp_primitive_blocks.")
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::GradientOptions::ecp_effective_charges,
                       "Per-ATOM effective nuclear charges for inline ECPs "
                       "(bare Z on atoms carrying no ECP). One entry per "
                       "atom.")
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::GradientOptions::ecp_total_ncore,
                       "Total number of core electrons replaced by inline "
                       "ECPs. Sizes the occupied block the same way the SCF "
                       "did.");

    m.def("compute_gradient", &vibeqc::compute_gradient,
          py::arg("molecule"), py::arg("basis"), py::arg("result"),
          py::arg("options") = vibeqc::GradientOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Analytic RHF nuclear gradient. Returns an (n_atoms, 3) Eigen "
          "matrix in Hartree/bohr. `result` must come from a converged "
          "run_rhf(...) call. Pass options.density_fit=True with "
          "options.aux_basis to use the DF analytic gradient (Weigend "
          "2002, PCCP 4, 4285).");

    m.def("compute_gradient_uhf", &vibeqc::compute_gradient_uhf,
          py::arg("molecule"), py::arg("basis"), py::arg("result"),
          py::arg("options") = vibeqc::GradientOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Analytic UHF nuclear gradient. Returns an (n_atoms, 3) matrix "
          "in Hartree/bohr. `result` must come from a converged "
          "run_uhf(...) call. Pass options.density_fit=True with "
          "options.aux_basis to use the DF analytic gradient (composes "
          "J on D_α+D_β with per-spin K at α_HF/2). Required to opt "
          "into the v0.7.x f-shell direct-gradient bug workaround for "
          "open-shell systems.");

    // compute_gradient_rks has a GridOptions default argument; see below,
    // registered after GridOptions.

    // ----- Numerical integration grid (DFT prerequisite) ------------------
    py::enum_<vibeqc::AngularScheme>(m, "AngularScheme",
        "Angular-grid family. ``Lebedev`` (Lebedev-Laikov) is the modern "
        "default for ~half the points at the same XC accuracy; "
        "``ProductGaussLegendre`` is the legacy n_θ × n_φ grid kept for "
        "parity-matrix bit-reproducibility.")
        .value("Lebedev", vibeqc::AngularScheme::Lebedev)
        .value("ProductGaussLegendre",
               vibeqc::AngularScheme::ProductGaussLegendre);

    py::enum_<vibeqc::AtomicPartition>(m, "AtomicPartition",
        "Atomic-partition cell-function family. ``Becke`` is the "
        "iterated-polynomial 1988 scheme (J. Chem. Phys. 88, 2547); "
        "``Stratmann`` is the 1996 piecewise polynomial with |μ| ≥ a "
        "hard cutoff (Chem. Phys. Lett. 257, 213) — modern default in "
        "PySCF / NWChem / Q-Chem, asymptotically linear per grid point.")
        .value("Becke",     vibeqc::AtomicPartition::Becke)
        .value("Stratmann", vibeqc::AtomicPartition::Stratmann);

    py::enum_<vibeqc::AngularPruning>(m, "AngularPruning",
        "Angular-order pruning scheme for Lebedev grids. ``None`` (the "
        "default) uses ``lebedev_order`` at every radial shell. "
        "``NWChem`` is the 3-tier Murray-Handy-Laming 1993 / SG-1 "
        "(Gill-Johnson-Pople 1993) scheme — inner core and outer tail "
        "use a smaller angular order, the chemically-active middle "
        "band uses the full order. Typical 25-40 % point-count "
        "reduction with sub-µHa accuracy cost on first-row organics.")
        .value("None",   vibeqc::AngularPruning::None)
        .value("NWChem", vibeqc::AngularPruning::NWChem);

    py::enum_<vibeqc::AtomicGridProfile>(m, "AtomicGridProfile",
        "Complete atom-grid protocol. ``Generic`` preserves the individual "
        "GridOptions controls. ``PySCFLevel3`` selects the PySCF-2.14 "
        "level-3 parity profile pinned by vibe-qc for SKALA-1.1.")
        .value("Generic", vibeqc::AtomicGridProfile::Generic)
        .value("PySCFLevel3", vibeqc::AtomicGridProfile::PySCFLevel3);

    py::class_<vibeqc::GridOptions>(m, "GridOptions")
        .def(py::init<>())
        .def_property("atomic_grid_profile",
            [](const vibeqc::GridOptions& o) {
                return o.atomic_grid_profile;
            },
            [](vibeqc::GridOptions& o, const py::object& v) {
                if (py::isinstance<vibeqc::AtomicGridProfile>(v)) {
                    o.atomic_grid_profile =
                        v.cast<vibeqc::AtomicGridProfile>();
                    return;
                }
                std::string s = v.cast<std::string>();
                std::transform(s.begin(), s.end(), s.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                if (s == "generic") {
                    o.atomic_grid_profile =
                        vibeqc::AtomicGridProfile::Generic;
                } else if (s == "pyscf-level3" || s == "pyscf_level3"
                           || s == "pyscflevel3" || s == "skala") {
                    o.atomic_grid_profile =
                        vibeqc::AtomicGridProfile::PySCFLevel3;
                } else {
                    throw std::invalid_argument(
                        "GridOptions.atomic_grid_profile: unknown value '"
                        + s + "' (use 'generic' or 'pyscf-level3')");
                }
            },
            "Complete atom-grid protocol. Accepts an AtomicGridProfile enum "
            "or 'generic' / 'pyscf-level3'. The latter is the exact "
            "PySCF-2.14 level-3 parity profile accepted by vibe-qc for "
            "SKALA-1.1.")
        .def_readwrite("n_radial", &vibeqc::GridOptions::n_radial)
        // Angular grid selector + per-scheme parameters.
        .def_property("angular",
            [](const vibeqc::GridOptions& o) { return o.angular; },
            [](vibeqc::GridOptions& o, const py::object& v) {
                // Accept either the enum directly or a case-insensitive
                // string ("lebedev" / "product" / "product_gauss_legendre")
                // — the string form is what most user code wants.
                if (py::isinstance<vibeqc::AngularScheme>(v)) {
                    o.angular = v.cast<vibeqc::AngularScheme>();
                    return;
                }
                std::string s = v.cast<std::string>();
                std::transform(s.begin(), s.end(), s.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                if (s == "lebedev") {
                    o.angular = vibeqc::AngularScheme::Lebedev;
                } else if (s == "product" || s == "product_gauss_legendre"
                           || s == "productgausslegendre") {
                    o.angular = vibeqc::AngularScheme::ProductGaussLegendre;
                } else {
                    throw std::invalid_argument(
                        "GridOptions.angular: unknown value '" + s
                        + "' (use 'lebedev' or 'product')");
                }
            },
            "Angular-grid family — either an AngularScheme enum or a "
            "case-insensitive string ('lebedev' / 'product').")
        .def_readwrite("lebedev_order",
                       &vibeqc::GridOptions::lebedev_order,
                       "Lebedev algebraic order. Bundled tiers: "
                       "5/7/11/15/17/23/27/29/31/35/41/47/53 "
                       "(14/26/50/86/110/194/266/302/350/434/590/770/974 "
                       "points). Used only when angular=Lebedev.")
        .def_readwrite("n_theta",  &vibeqc::GridOptions::n_theta,
                       "Gauss-Legendre θ nodes per shell — used only "
                       "with angular=ProductGaussLegendre.")
        .def_readwrite("n_phi",    &vibeqc::GridOptions::n_phi,
                       "Uniform φ nodes per shell — used only with "
                       "angular=ProductGaussLegendre.")
        // Atomic-partition selector + per-scheme parameters.
        .def_property("partition",
            [](const vibeqc::GridOptions& o) { return o.partition; },
            [](vibeqc::GridOptions& o, const py::object& v) {
                if (py::isinstance<vibeqc::AtomicPartition>(v)) {
                    o.partition = v.cast<vibeqc::AtomicPartition>();
                    return;
                }
                std::string s = v.cast<std::string>();
                std::transform(s.begin(), s.end(), s.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                if (s == "becke") {
                    o.partition = vibeqc::AtomicPartition::Becke;
                } else if (s == "stratmann") {
                    o.partition = vibeqc::AtomicPartition::Stratmann;
                } else {
                    throw std::invalid_argument(
                        "GridOptions.partition: unknown value '" + s
                        + "' (use 'becke' or 'stratmann')");
                }
            },
            "Atomic-partition cell-function — either an AtomicPartition "
            "enum or a case-insensitive string ('becke' / 'stratmann').")
        .def_readwrite("becke_k",  &vibeqc::GridOptions::becke_k,
                       "Iterated-polynomial smoothing order — used only "
                       "with partition=Becke.")
        .def_property("angular_pruning",
            [](const vibeqc::GridOptions& o) { return o.angular_pruning; },
            [](vibeqc::GridOptions& o, const py::object& v) {
                if (py::isinstance<vibeqc::AngularPruning>(v)) {
                    o.angular_pruning = v.cast<vibeqc::AngularPruning>();
                    return;
                }
                std::string s = v.cast<std::string>();
                std::transform(s.begin(), s.end(), s.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                if (s == "none") {
                    o.angular_pruning = vibeqc::AngularPruning::None;
                } else if (s == "nwchem") {
                    o.angular_pruning = vibeqc::AngularPruning::NWChem;
                } else {
                    throw std::invalid_argument(
                        "GridOptions.angular_pruning: unknown value '" + s
                        + "' (use 'none' or 'nwchem')");
                }
            },
            "Angular-order pruning scheme for Lebedev grids — either "
            "an AngularPruning enum or a case-insensitive string "
            "('none' / 'nwchem'). Only effective when angular=Lebedev.")
        .def_readwrite("orca_angular_points",
                       &vibeqc::GridOptions::orca_angular_points,
                       "ORCA-style 5-region AngularGrid point counts. Empty "
                       "uses the legacy Lebedev/grid-pruning path.")
        .def_readwrite("vv10_grid_factor",
                       &vibeqc::GridOptions::vv10_grid_factor,
                       "VV10 nonlocal correlation grid coarsening factor. "
                       "When > 1.0, a separate coarser grid is built for "
                       "the VV10 double sum (O(N^2) cost). Factor 3.0 "
                       "(default) yields ~81x VV10 kernel speedup with "
                       "sub-mHa energy accuracy. Set to 1.0 for legacy "
                       "behaviour (VV10 on the full XC grid). <=0 disables "
                       "the separate VV10 grid.");

    m.def("cosx_basis_cardinality_from_name",
          &vibeqc::cosx_basis_cardinality_from_name,
          py::arg("basis_name"),
          "Heuristic COSX basis cardinality from a basis-set name.");

    m.def("resolve_cosx_variant",
          &vibeqc::resolve_cosx_variant,
          py::arg("variant"),
          py::arg("thresh_cosx"),
          py::arg("basis_cardinality"),
          py::arg("grid_level"),
          "Resolve a COSX variant selector to a concrete variant.");

    m.def("resolve_cosx_grid_level",
          &vibeqc::resolve_cosx_grid_level,
          py::arg("grid_level"),
          py::arg("basis_cardinality"),
          "Resolve a COSX grid-level selector to a concrete GridX tier.");

    m.def("cosx_grid_options_for_level",
          &vibeqc::cosx_grid_options_for_level,
          py::arg("level"),
          "Return COSX GridX options for level 1..4.");

    m.def("cosx_grid_stages_for_level",
          &vibeqc::cosx_grid_stages_for_level,
          py::arg("level"),
          "Return COSX coarse-to-fine grid stages for level 1..4.");

    py::class_<vibeqc::Grid>(m, "Grid")
        .def(py::init([](const Eigen::MatrixX3d& points,
                         const Eigen::VectorXd& weights) {
            if (points.rows() != weights.size()
                || !points.allFinite() || !weights.allFinite())
                throw std::invalid_argument("Grid requires finite aligned points and weights");
            vibeqc::Grid grid;
            grid.points = points;
            grid.weights = weights;
            return grid;
        }), py::arg("points"), py::arg("weights"),
        "Construct a fixed quadrature without owning-atom metadata; "
        "suitable for semilocal XC, not atom-nonlocal providers.")
        .def_property_readonly("n_points", [](const vibeqc::Grid& grid) {
            return grid.points.rows();
        }, "Quadrature size without copying the points or weights.")
        .def_readonly("atomic_grid_profile",
                      &vibeqc::Grid::atomic_grid_profile,
                      "Complete construction profile actually used for this "
                      "grid (Generic or PySCFLevel3).")
        .def_readonly("points",  &vibeqc::Grid::points,
                      "(N, 3) grid points in bohr.")
        .def_readonly("weights", &vibeqc::Grid::weights,
                      "(N,) integration weights (bohr^3).")
        .def_readonly("atomic_weights", &vibeqc::Grid::atomic_weights,
                      "(N,) unpartitioned owning-atom quadrature weights "
                      "(bohr^3), used by atom-nonlocal XC descriptors.")
        .def_readonly("atom_of_point", &vibeqc::Grid::atom_of_point,
                      "(N,) index of the atom that 'owns' each point.")
        .def_readonly("atom_coords", &vibeqc::Grid::atom_coords,
                      "(n_atoms, 3) owning-atom coordinates in bohr.")
        .def_readonly("atomic_numbers", &vibeqc::Grid::atomic_numbers,
                      "(n_atoms,) nuclear charges in owning-atom order.");

    m.def("build_grid", &vibeqc::build_grid,
          py::arg("molecule"),
          py::arg("options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Build the DFT numerical-integration grid for the given molecule.");

    m.def("grid_atomic_point_counts", &vibeqc::grid_atomic_point_counts,
          py::arg("molecule"),
          py::arg("options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Return exact per-atom raw-grid point counts without building "
          "coordinates, partitions, or AO tables.");

    m.def("build_periodic_point_grid", &vibeqc::build_periodic_point_grid,
          py::arg("system"), py::arg("image_radius_bohr"),
          py::arg("options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Build an atom-major periodic grid using point-centered physical "
          "image neighborhoods, preserving raw atomic quadrature metadata.");

    m.def("build_grid_periodic",
          py::overload_cast<
              const vibeqc::Molecule&,
              const std::vector<Eigen::Vector3d>&,
              const vibeqc::GridOptions&>(&vibeqc::build_grid_periodic),
          py::arg("grid_molecule"),
          py::arg("partition_atom_positions"),
          py::arg("options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Build a DFT integration grid with the periodic-Becke partition. "
          "Points are generated around ``grid_molecule`` atoms (home cell), "
          "but the Becke fuzzy-cell partition denominator includes every "
          "atom in ``partition_atom_positions`` (home + images). The first "
          "len(grid_molecule.atoms()) entries of partition_atom_positions "
          "must equal the home-cell atom positions in order. Reduces to "
          "build_grid when partition_atom_positions == grid_molecule.");

    m.def("build_grid_periodic",
          py::overload_cast<
              const vibeqc::Molecule&,
              const std::vector<Eigen::Vector3d>&,
              const std::vector<int>&,
              const vibeqc::GridOptions&>(&vibeqc::build_grid_periodic),
          py::arg("grid_molecule"),
          py::arg("partition_atom_positions"),
          py::arg("partition_atomic_numbers"),
          py::arg("options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Atomic-number-aware periodic grid builder. Required by the "
          "PySCF-level3 profile so heteroatomic radius adjustment also "
          "covers image atoms.");

    // Depends on GridOptions being registered (default arg), so it lands
    // here rather than beside the other gradient bindings.
    m.def("compute_gradient_rks", &vibeqc::compute_gradient_rks,
          py::arg("molecule"), py::arg("basis"), py::arg("result"),
          py::arg("grid_options") = vibeqc::GridOptions{},
          py::arg("options") = vibeqc::GradientOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Analytic RKS (closed-shell DFT) nuclear gradient. Supports "
          "LDA, GGA, and hybrid functionals. Pass options.density_fit"
          "=True with options.aux_basis for the DF analytic gradient. "
          "Returns (n_atoms, 3) in Hartree/bohr.");

    m.def("xc_pulay_gradient_rks", &vibeqc::xc_pulay_gradient_rks,
          py::arg("molecule"), py::arg("basis"), py::arg("D"),
          py::arg("functional"),
          py::arg("grid_options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "XC-Pulay piece of the RKS gradient on a fixed Becke grid (the "
          "molecular XC force; LDA / GGA / meta-GGA, no Becke weight "
          "derivative). Exposed for the Γ-CCM force driver, whose XC term "
          "is the ordinary molecular functional of the supercell density "
          "and therefore does not carry the WSSC fold (unlike h^CCM / "
          "J^CCM). Returns (n_atoms, 3) in Hartree/bohr.");

    m.def("compute_gradient_uks", &vibeqc::compute_gradient_uks,
          py::arg("molecule"), py::arg("basis"), py::arg("result"),
          py::arg("grid_options") = vibeqc::GridOptions{},
          py::arg("options") = vibeqc::GradientOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Analytic UKS (open-shell DFT) nuclear gradient. Supports LDA, "
          "GGA, and hybrid functionals. Returns (n_atoms, 3) in Hartree/"
          "bohr. Pass options.density_fit=True with options.aux_basis "
          "for the DF analytic gradient (Fix D workaround for the "
          "v0.7.x f-shell direct-gradient bug; matches "
          "compute_gradient_uhf's per-spin J/K composition + XC Pulay "
          "piece unchanged).");

    m.def("xc_pulay_gradient_uks",
          py::overload_cast<const vibeqc::Molecule&, const vibeqc::BasisSet&,
                            const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                            const std::string&, const vibeqc::GridOptions&>(
              &vibeqc::xc_pulay_gradient_uks),
          py::arg("molecule"), py::arg("basis"),
          py::arg("D_alpha"), py::arg("D_beta"),
          py::arg("functional"),
          py::arg("grid_options") = vibeqc::GridOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Open-shell counterpart of xc_pulay_gradient_rks: the XC-Pulay "
          "piece of the UKS gradient on a fixed Becke grid (LDA / GGA / "
          "meta-GGA, no Becke weight derivative). Exposed for the Γ-CCM UKS "
          "force driver, whose XC term is the ordinary molecular functional "
          "of the supercell spin densities (no WSSC fold). Returns "
          "(n_atoms, 3) in Hartree/bohr.");

    // ---- AO Fock-builder primitives ----
    // Exposed so the analytic Hessian (Phase 17b-3) can build F at a
    // displaced geometry with a frozen reference density (no SCF), and
    // so external code can probe Coulomb / exchange contributions.
    m.def("build_fock_g",
          [](const py::array_t<double, py::array::c_style>& eri,
             const Eigen::MatrixXd& D) {
              if (eri.ndim() != 4) {
                  throw std::invalid_argument(
                      "build_fock_g: eri must be a 4-D ndarray");
              }
              vibeqc::Eri4D eri_obj;
              eri_obj.n = static_cast<std::size_t>(eri.shape(0));
              eri_obj.data.assign(eri.data(),
                                   eri.data() + eri.size());
              return vibeqc::build_fock_g(eri_obj, D);
          },
          py::arg("eri"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Closed-shell Fock 2-electron contribution G[D] = J[D] − ½ K[D]. "
          "``eri`` is the (n, n, n, n) AO integral tensor from compute_eri.");

    m.def("build_coulomb",
          [](const py::array_t<double, py::array::c_style>& eri,
             const Eigen::MatrixXd& D) {
              vibeqc::Eri4D eri_obj;
              eri_obj.n = static_cast<std::size_t>(eri.shape(0));
              eri_obj.data.assign(eri.data(), eri.data() + eri.size());
              return vibeqc::build_coulomb(eri_obj, D);
          },
          py::arg("eri"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Coulomb matrix J[D]_μν = Σ_λσ (μν|λσ) D_λσ.");

    m.def("build_exchange",
          [](const py::array_t<double, py::array::c_style>& eri,
             const Eigen::MatrixXd& D) {
              vibeqc::Eri4D eri_obj;
              eri_obj.n = static_cast<std::size_t>(eri.shape(0));
              eri_obj.data.assign(eri.data(), eri.data() + eri.size());
              return vibeqc::build_exchange(eri_obj, D);
          },
          py::arg("eri"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Exchange matrix K[D]_μν = Σ_λσ (μλ|νσ) D_λσ.");

    // ---- Per-contribution gradient bindings ----
    // These are the building blocks the high-level `compute_gradient_*`
    // wrap. Exposed so the analytic-Hessian assembly (Phase 17b-3) can
    // FD them with frozen weight matrices, and so cross-check tests can
    // verify each piece against PySCF independently.
    m.def("nuclear_repulsion_gradient",
          py::overload_cast<const vibeqc::Molecule&>(
              &vibeqc::nuclear_repulsion_gradient),
          py::arg("molecule"),
          py::call_guard<py::gil_scoped_release>(),
          "Closed-form ∂E_nuc/∂R per atom (bare Z). Returns "
          "(n_atoms, 3) in Hartree/bohr.");

    m.def("nuclear_repulsion_gradient",
          py::overload_cast<const vibeqc::Molecule&,
                            const std::vector<double>&>(
              &vibeqc::nuclear_repulsion_gradient),
          py::arg("molecule"), py::arg("effective_charges"),
          py::call_guard<py::gil_scoped_release>(),
          "Effective-charges overload — pass Z_eff = Z − n_core per "
          "atom for ECP-aware gradients. Reduces to the bare-Z "
          "overload when effective_charges == [Z for atoms].");

    m.def("overlap_gradient_contribution",
          &vibeqc::overlap_gradient_contribution,
          py::arg("basis"), py::arg("molecule"), py::arg("W"),
          py::call_guard<py::gil_scoped_release>(),
          "−Σ_μν W ∂S/∂R contraction (per atom). Pass the energy-"
          "weighted density W. Returns (n_atoms, 3) in Hartree/bohr.");

    m.def("one_electron_gradient_contribution",
          py::overload_cast<const vibeqc::BasisSet&,
                            const vibeqc::Molecule&,
                            const Eigen::MatrixXd&>(
              &vibeqc::one_electron_gradient_contribution),
          py::arg("basis"), py::arg("molecule"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Σ_μν D ∂(T+V)/∂R contraction (per atom, bare Z). Pass the "
          "closed-shell total density D. Returns (n_atoms, 3) in "
          "Hartree/bohr.");

    m.def("one_electron_gradient_contribution",
          py::overload_cast<const vibeqc::BasisSet&,
                            const vibeqc::Molecule&,
                            const Eigen::MatrixXd&,
                            const std::vector<double>&>(
              &vibeqc::one_electron_gradient_contribution),
          py::arg("basis"), py::arg("molecule"), py::arg("D"),
          py::arg("effective_charges"),
          py::call_guard<py::gil_scoped_release>(),
          "Effective-charges overload — uses ``effective_charges`` "
          "for the V_ne derivative kernel (kinetic piece unchanged). "
          "Required for gradients run with ECPs to match the SCF's "
          "compute_nuclear_with_charges Hcore.");

    m.def("compute_ecp_gradient_contribution",
          &vibeqc::compute_ecp_gradient_contribution,
          py::arg("basis"), py::arg("molecule"), py::arg("ecp_centers"),
          py::arg("D"),
          py::arg("library_name") = std::string("ecp10mdf"),
          py::arg("share_dir") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Σ_μν D · (∂V_ECP/∂R) contraction per atom, via libecpint's "
          "first-derivative API. Returns (n_atoms, 3) in Hartree/bohr; "
          "an empty ``ecp_centers`` returns a zero matrix.");

    m.def("compute_ecp_gradient_contribution_from_primitives",
          &vibeqc::compute_ecp_gradient_contribution_from_primitives,
          py::arg("basis"), py::arg("molecule"), py::arg("ecp_centers"),
          py::arg("primitives"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Inline-primitive counterpart of "
          "``compute_ecp_gradient_contribution`` (#574). Same "
          "Σ_μν D · (∂V_ECP/∂R) contraction, with the ECP supplied as "
          "primitive blocks instead of an XML library name. "
          "``ecp_centers`` and ``primitives`` are per-ECP-centre and must "
          "be the same length. Returns (n_atoms, 3) in Hartree/bohr; empty "
          "``primitives`` returns a zero matrix.");

    m.def("ecp_effective_charges",
          &vibeqc::ecp_effective_charges,
          py::arg("molecule"), py::arg("ecp_centers"),
          py::arg("library_name") = std::string("ecp10mdf"),
          py::arg("share_dir") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Return per-atom Z_eff = Z − n_core (or bare Z for atoms "
          "without an ECP). Order matches mol.atoms().");

    m.def("two_electron_gradient_contribution",
          &vibeqc::two_electron_gradient_contribution,
          py::arg("basis"), py::arg("molecule"), py::arg("D"),
          py::arg("alpha_hf") = 1.0,
          py::call_guard<py::gil_scoped_release>(),
          "Σ_μνλσ Γ ∂(μν|λσ)/∂R contraction (per atom). For plain HF "
          "use alpha_hf = 1; for hybrid DFT pass the functional's HF-"
          "exchange fraction. Returns (n_atoms, 3) in Hartree/bohr.");

        m.def("two_electron_gradient_contribution_uhf",
              &vibeqc::two_electron_gradient_contribution_uhf,
              py::arg("basis"), py::arg("molecule"),
              py::arg("D_alpha"), py::arg("D_beta"),
              py::arg("alpha_hf") = 1.0,
              py::call_guard<py::gil_scoped_release>(),
              "UHF/UKS counterpart of two_electron_gradient_contribution.");

        m.def("two_electron_gradient_casscf",
              &vibeqc::two_electron_gradient_casscf,
              py::arg("basis"), py::arg("molecule"),
              py::arg("gamma_ao_flat"),
              py::call_guard<py::gil_scoped_release>(),
              "CASSCF two-electron nuclear gradient at a general AO "
              "2-particle density.  ``gamma_ao_flat`` is a flat row-major "
              "vector of length nb^4, indexed as "
              "``((mu*nb + nu)*nb + lam)*nb + sig``.  Uses the same "
              "shell-quartet derivative-integral loop as "
              "two_electron_gradient_contribution but replaces the RHF "
              "density-product with the supplied tensor.  Returns "
              "(n_atoms, 3) in Hartree/bohr.");

    // CPCM solvation-gradient helper (v0.9.1). Analytic per-center
    // derivatives of the electron-density / external-point-charge
    // interaction — see vibeqc/gradient.hpp for the math.
    py::class_<vibeqc::ExternalChargeGradient>(
        m, "ExternalChargeGradient",
        "Per-center derivatives of E_ext = Σ_i q_i Tr(D·M_i) where "
        "M_i,μν = ⟨μ|1/|r−s_i||ν⟩. ``atom_grad`` (n_atoms, 3) carries "
        "the bra+ket basis-center derivatives; ``point_grad`` "
        "(n_points, 3) carries the per-external-charge derivatives. "
        "Both in Hartree/bohr with the sign of dE_ext/dR.")
        .def_readonly("atom_grad", &vibeqc::ExternalChargeGradient::atom_grad)
        .def_readonly("point_grad",
                      &vibeqc::ExternalChargeGradient::point_grad);

    m.def("compute_external_charge_density_gradient",
          &vibeqc::compute_external_charge_density_gradient,
          py::arg("basis"), py::arg("molecule"), py::arg("D"),
          py::arg("charges"), py::arg("positions"),
          py::call_guard<py::gil_scoped_release>(),
          "Analytic per-center gradient of the electron density's "
          "interaction with external point charges — the CPCM "
          "electronic-ESP-derivative kernel. ``charges`` and "
          "``positions`` are the apparent surface charges q_i and "
          "cavity points s_i; ``D`` is the SCF density. Returns an "
          "ExternalChargeGradient (atom_grad + point_grad). Drives "
          "libint's nuclear-attraction gradient engine in one pass — "
          "the closed-form replacement for the integral-FD path in "
          "vibeqc.solvation.cpcm_gradient.");

    // ---- Phase G1a — periodic gradient primitives -------------------
    // Lattice-summed analogues of the molecular gradient primitives
    // above, exposed individually so callers (and tests) can FD each
    // piece in isolation against the molecular reference.
    m.def("nuclear_repulsion_gradient_per_cell",
          &vibeqc::nuclear_repulsion_gradient_per_cell,
          py::arg("system"), py::arg("opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Per-cell nuclear-repulsion gradient. Returns (n_atoms, 3) "
          "in Hartree/bohr. Same lattice-sum convention as "
          "nuclear_repulsion_per_cell.");

    m.def("nuclear_repulsion_slab_ewald_2d_gradient",
          &vibeqc::nuclear_repulsion_slab_ewald_2d_gradient,
          py::arg("system"), py::arg("opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Analytic gradient of exactly what nuclear_repulsion_per_cell "
          "computes under CoulombMethod.SLAB_EWALD_2D: the bare Parry / de "
          "Leeuw-Perram 2D-Ewald four-term block (background-neutralised "
          "primitive at z_b = 0 minus the exact uniform-sheet term), with "
          "the SCF's actual α (slab_ewald_alpha or 0.4) and real cutoff "
          "(nuclear_cutoff_bohr). Returns (n_atoms, 3) in Hartree/bohr; "
          "dim == 2 only.");

    m.def("overlap_lattice_gradient_contribution",
          &vibeqc::overlap_lattice_gradient_contribution,
          py::arg("basis"), py::arg("system"), py::arg("W_set"),
          py::arg("opts"), py::arg("apply_pair_filter") = true,
          py::call_guard<py::gil_scoped_release>(),
          "−Σ_g W(g) · ∂S(g)/∂R contraction (per atom). Pass the real-"
          "space energy-weighted density on the same cell list as the "
          "matching compute_overlap_lattice call. Returns (n_atoms, 3). "
          "With opts.pair_complete_1e the derivative runs over "
          "compute_overlap_lattice's pair-complete term set; "
          "apply_pair_filter=False differentiates the all-pairs sum over "
          "the supplied cells instead (the Ewald-split V_ne G=0 "
          "correction needs that).");

    m.def("kinetic_lattice_gradient_contribution",
          &vibeqc::kinetic_lattice_gradient_contribution,
          py::arg("basis"), py::arg("system"), py::arg("D_set"),
          py::arg("opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Σ_g D(g) · ∂T(g)/∂R contraction (per atom). Pass the real-"
          "space density on the same cell list as compute_kinetic_lattice. "
          "Returns (n_atoms, 3).");

    m.def("nuclear_lattice_gradient_contribution",
          &vibeqc::nuclear_lattice_gradient_contribution,
          py::arg("basis"), py::arg("system"), py::arg("D_set"),
          py::arg("opts"),
          py::call_guard<py::gil_scoped_release>(),
          "Σ_g D(g) · ∂V(g)/∂R contraction (per atom). Includes the "
          "Σ_h-over-nuclear-images derivative — basis-center derivatives "
          "(2) plus one buffer per nuclear point charge in the lattice "
          "sum. Returns (n_atoms, 3).");

    m.def("nuclear_erfc_lattice_gradient_contribution",
          &vibeqc::nuclear_erfc_lattice_gradient_contribution,
          py::arg("basis"), py::arg("system"), py::arg("D_set"),
          py::arg("opts"), py::arg("omega"),
          py::call_guard<py::gil_scoped_release>(),
          "Σ_g D(g) · ∂V_short(g)/∂R for the erfc-screened (erfc(ω·r)/r) "
          "nuclear attraction — the short-range piece of the BIPOLE Ewald "
          "V_ne gradient. Same buffer layout as "
          "nuclear_lattice_gradient_contribution. Returns (n_atoms, 3).");

    m.def("eri_lattice_gradient_contribution",
          &vibeqc::eri_lattice_gradient_contribution,
          py::arg("basis"), py::arg("system"), py::arg("D_set"),
          py::arg("opts"), py::arg("alpha_hf") = 1.0,
          py::arg("j_scale") = 1.0, py::arg("omega") = 0.0,
          py::arg("exchange_energy_convention") = false,
          py::arg("internal_cells") = std::vector<vibeqc::LatticeCell>{},
          py::arg("output_cells") = std::vector<vibeqc::LatticeCell>{},
          py::arg("output_shell_masks") =
              std::vector<std::vector<uint8_t>>{},
          py::call_guard<py::gil_scoped_release>(),
          "Periodic 2-electron ERI gradient: Σ_{g_a,g_b,g_c} Σ_μνλσ "
          "Γ(g_a,g_b,g_c) · ∂(μ_0 ν_{g_a} | λ_{g_b} σ_{g_c})/∂R with "
          "Γ = (j_scale/2) D(g_a)_μν D(g_c-g_b)_λσ - (α_HF/4) D(g_b)_μλ "
          "D(g_c-g_a)_νσ. For pure DFT pass alpha_hf = 0; for plain "
          "RHF pass 1; for hybrid DFT pass the functional's HF-exchange "
          "fraction. j_scale=0 gives an exchange-only gradient; omega>0 "
          "switches the 2e operator to erfc(ω·r)/r (BIPOLE J_SR). "
          "exchange_energy_convention=True differentiates the direct "
          "-alpha/4 D:K exchange-energy contraction for the K term. "
          "internal_cells optionally extends the lambda/sigma image "
          "traversal; output_cells optionally selects the c_g Fock-output "
          "domain while D_set.cells remains the density support; "
          "output_shell_masks optionally selects the emitted shell pairs "
          "for each output cell. "
          "Returns (n_atoms, 3).");

    // ---- Phase 17b-2 — Hessian skeleton-derivative contractions ----
    m.def("compute_overlap_hessian_contribution",
          &vibeqc::compute_overlap_hessian_contribution,
          py::arg("basis"), py::arg("molecule"), py::arg("W"),
          py::call_guard<py::gil_scoped_release>(),
          "Skeleton −Σ W ∂²S/∂R_α∂R_β contraction. Returns a (3N, 3N) "
          "symmetric matrix in Hartree/bohr². Pass the energy-weighted "
          "density W = 2 C_occ · diag(ε_occ) · C_occ^T (closed-shell "
          "convention). Used internally by the analytic RHF Hessian.");

    m.def("compute_kinetic_nuclear_hessian_contribution",
          &vibeqc::compute_kinetic_nuclear_hessian_contribution,
          py::arg("basis"), py::arg("molecule"), py::arg("D"),
          py::call_guard<py::gil_scoped_release>(),
          "Skeleton Σ D ∂²(T+V)/∂R_α∂R_β contraction. Returns a (3N, 3N) "
          "symmetric matrix in Hartree/bohr². Pass the closed-shell total "
          "density D = 2 C_occ · C_occ^T. Includes nuclear-attraction "
          "second derivatives w.r.t. both basis-function and nuclear "
          "centers.");

    m.def("compute_eri_hessian_contribution",
          &vibeqc::compute_eri_hessian_contribution,
          py::arg("basis"), py::arg("molecule"), py::arg("D"),
          py::arg("alpha_hf") = 1.0,
          py::call_guard<py::gil_scoped_release>(),
          "Skeleton Σ Γ ∂²(μν|λσ)/∂R_α∂R_β contraction with the closed-"
          "shell two-particle density Γ = (1/2) D D − (α_HF/4) D D' "
          "(exchange permutation). For plain HF use α_HF = 1; for hybrid "
          "DFT pass the functional's HF-exchange fraction.");

    m.def("compute_eri_hessian_contribution_uhf",
          &vibeqc::compute_eri_hessian_contribution_uhf,
          py::arg("basis"), py::arg("molecule"),
          py::arg("D_alpha"), py::arg("D_beta"),
          py::arg("alpha_hf") = 1.0,
          py::call_guard<py::gil_scoped_release>(),
          "UHF / UKS counterpart of compute_eri_hessian_contribution. "
          "Two-particle density: Γ = (1/2)(D_α+D_β)(D_α+D_β) − "
          "(α_HF/2)(D_α D_α + D_β D_β) (per-spin exchange).");

    m.def("nuclear_repulsion_hessian",
          &vibeqc::nuclear_repulsion_hessian,
          py::arg("molecule"),
          py::call_guard<py::gil_scoped_release>(),
          "Closed-form ∂²E_nuc/∂R_α∂R_β for the nuclear-repulsion "
          "energy. Returns a (3N, 3N) symmetric matrix.");

    m.def("evaluate_ao", &vibeqc::evaluate_ao,
          py::arg("basis"), py::arg("points"),
          py::call_guard<py::gil_scoped_release>(),
          "Evaluate χ_μ(r_g) for every basis function and every grid point. "
          "Returns an (n_points, n_basis) matrix.");

    m.def("evaluate_bloch_ao", &vibeqc::evaluate_bloch_ao,
          py::arg("basis"), py::arg("points"), py::arg("k_cart"),
          py::arg("lattice_translations"),
          py::call_guard<py::gil_scoped_release>(),
          "Bloch-summed AO matrix χ_μ^k(r_g) = Σ_T exp(i k·T) χ_μ(r_g − T) "
          "on a flat (n_points, 3) array of Cartesian grid points (bohr). "
          "``lattice_translations`` is an (n_T, 3) array of Cartesian "
          "lattice shifts T (bohr) — usually obtained by stacking the "
          "``r_cart`` of cells from ``direct_lattice_cells(system, cutoff)``. "
          "Returns a complex (n_points, n_basis) matrix. Contracting against "
          "a column of the multi-k SCF coefficient matrix C(k) recovers the "
          "Bloch crystalline orbital ψ_{n,k}(r); use the Python wrapper "
          "``vibeqc.evaluate_bloch_orbital`` for that one-liner.");

    // ----- XC functional (libxc wrapper) ----------------------------------
    py::enum_<vibeqc::XCKind>(m, "XCKind")
        .value("LDA",  vibeqc::XCKind::LDA)
        .value("GGA",  vibeqc::XCKind::GGA)
        .value("MGGA", vibeqc::XCKind::MGGA);

    py::class_<vibeqc::Functional>(m, "Functional")
        .def(py::init<const std::string&, int>(),
             py::arg("name"), py::arg("spin") = 1,
             "Construct an XC functional by name ('LDA', 'PBE', 'BLYP', "
             "'B3LYP') or by comma-separated libxc IDs. 'b3lyp' is the "
             "ORCA-convention B3LYP (libxc HYB_GGA_XC_B3LYP5, VWN5 "
             "local correlation; explicit spelling 'b3lyp5'); "
             "'b3lyp/g' / 'b3lypg' select the Gaussian-convention "
             "variant (HYB_GGA_XC_B3LYP, VWN-RPA) that Gaussian, "
             "PySCF, and Psi4 mean by the bare name.")
        .def_property_readonly("name", &vibeqc::Functional::name)
        .def_property_readonly("kind", &vibeqc::Functional::kind)
        .def_property_readonly("is_external",
                               &vibeqc::Functional::is_external,
                               "True for a registered full-grid XC backend "
                               "rather than a libxc functional.")
        .def_property_readonly(
            "external_capabilities",
            [](const vibeqc::Functional& functional) -> py::object {
                if (!functional.is_external()) return py::none();
                const auto& capabilities =
                    functional.external_capabilities();
                py::dict result;
                result["version"] = capabilities.version;
                result["required_grid_profile"] =
                    capabilities.required_grid_profile;
                return result;
            },
            "Immutable external-provider capability record, or None for a "
            "libxc functional. Version 1 declares required_grid_profile.")
        .def_property_readonly("is_hybrid", &vibeqc::Functional::is_hybrid)
        .def_property_readonly("hf_exchange_fraction",
                               &vibeqc::Functional::hf_exchange_fraction)
        .def_property_readonly("is_range_separated",
                               &vibeqc::Functional::is_range_separated,
                               "True for a range-separated (CAM / RSH) "
                               "hybrid — ωB97X, ωB97X-D, CAM-B3LYP, "
                               "HSE06, …. The exact-exchange admixture "
                               "is position-dependent: EXX(r) = "
                               "cam_alpha + cam_beta·erf(rsh_omega·r).")
        .def_property_readonly("rsh_omega", &vibeqc::Functional::rsh_omega,
                               "Range-separation parameter ω (bohr⁻¹) of "
                               "a CAM/RSH hybrid; 0 for non-RSH.")
        .def_property_readonly("cam_alpha", &vibeqc::Functional::cam_alpha,
                               "Full-range (always-on) HF-exchange "
                               "fraction. Equals hf_exchange_fraction "
                               "for a global hybrid; the constant CAM "
                               "term α for an RSH hybrid.")
        .def_property_readonly("cam_beta", &vibeqc::Functional::cam_beta,
                               "Long-range-only HF-exchange fraction β "
                               "(switched on by erf(ω·r)). Zero for a "
                               "global hybrid; for ωB97X-family "
                               "long-range-corrected functionals "
                               "cam_alpha + cam_beta = 1.")
        .def_property_readonly("is_double_hybrid",
                               &vibeqc::Functional::is_double_hybrid,
                               "True if this is a double-hybrid functional "
                               "(B2PLYP, DSD-PBEP86, …). Double hybrids "
                               "add a scaled-MP2 correlation correction "
                               "on top of the SCF hybrid-DFT step; see "
                               "``mp2_c_os`` / ``mp2_c_ss`` for the "
                               "scaling coefficients and "
                               "``vibeqc.run_b2plyp`` for the orchestrator.")
        .def_property_readonly("mp2_c_os", &vibeqc::Functional::mp2_c_os,
                               "Opposite-spin MP2-correction coefficient "
                               "for a double-hybrid functional. Zero for "
                               "non-double-hybrid functionals. See "
                               "``MP2Options.c_os``.")
        .def_property_readonly("mp2_c_ss", &vibeqc::Functional::mp2_c_ss,
                               "Same-spin MP2-correction coefficient for "
                               "a double-hybrid functional. Zero for "
                               "non-double-hybrid functionals. See "
                               "``MP2Options.c_ss``.")
        .def_property_readonly("needs_vv10", &vibeqc::Functional::needs_vv10,
                               "True for a VV10-paired functional (ωB97X-V, "
                               "ωB97M-V, the standalone VV10, …). libxc "
                               "supplies only the semilocal part; the RKS / "
                               "UKS SCF adds the VV10 nonlocal correlation "
                               "(see vibeqc.compute_vv10) self-consistently.")
        .def_property_readonly("vv10_b", &vibeqc::Functional::vv10_b,
                               "VV10 nonlocal parameter b (e.g. 6.0 for "
                               "ωB97X-V / ωB97M-V, 5.9 for VV10). Zero when "
                               "needs_vv10 is False.")
        .def_property_readonly("vv10_C", &vibeqc::Functional::vv10_C,
                               "VV10 nonlocal parameter C (e.g. 0.01 for "
                               "ωB97X-V / ωB97M-V, 0.0093 for VV10). Zero "
                               "when needs_vv10 is False.")
        .def("eval_unpolarised",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho,
                const Eigen::VectorXd& sigma) {
                 Eigen::VectorXd exc, v_rho, v_sigma;
                 f.eval_unpolarised(rho, sigma, exc, v_rho, v_sigma);
                 return py::make_tuple(std::move(exc),
                                       std::move(v_rho),
                                       std::move(v_sigma));
             },
             py::arg("rho"), py::arg("sigma"),
             "Evaluate XC energy density and potentials on a grid. Returns "
             "(exc, v_rho, v_sigma). For LDA, sigma may be empty and v_sigma "
             "will be empty too.")
        .def("eval_unpolarised_fxc",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho,
                const Eigen::VectorXd& sigma) {
                 Eigen::VectorXd v2rho2, v2rhosigma, v2sigma2;
                 f.eval_unpolarised_fxc(rho, sigma, v2rho2, v2rhosigma,
                                          v2sigma2);
                 return py::make_tuple(std::move(v2rho2),
                                       std::move(v2rhosigma),
                                       std::move(v2sigma2));
             },
             py::arg("rho"), py::arg("sigma"),
             "Phase 17d — XC kernel for analytic KS Hessian / CPKS. Returns "
             "(v2rho2, v2rhosigma, v2sigma2). For LDA, only v2rho2 = "
             "∂²f/∂ρ² is populated; v2rhosigma and v2sigma2 are empty.")
        .def("eval_polarised_lda_fxc",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho_a,
                const Eigen::VectorXd& rho_b) {
                 Eigen::VectorXd v_aa, v_ab, v_bb;
                 f.eval_polarised_lda_fxc(rho_a, rho_b, v_aa, v_ab, v_bb);
                 return py::make_tuple(std::move(v_aa),
                                       std::move(v_ab),
                                       std::move(v_bb));
             },
             py::arg("rho_a"), py::arg("rho_b"),
             "LDA-only polarized XC kernel for UKS analytic Hessian. "
             "Returns (v2rho2_αα, v2rho2_αβ, v2rho2_ββ).")
        .def("eval_polarised_gga_fxc",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho_a,
                const Eigen::VectorXd& rho_b,
                const Eigen::VectorXd& sigma_aa,
                const Eigen::VectorXd& sigma_ab,
                const Eigen::VectorXd& sigma_bb) {
                 vibeqc::PolarisedGGAFxc out;
                 f.eval_polarised_gga_fxc(
                     rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, out);
                 py::dict d;
                 d["v2rho2_aa"] = out.v2rho2_aa;
                 d["v2rho2_ab"] = out.v2rho2_ab;
                 d["v2rho2_bb"] = out.v2rho2_bb;
                 d["v2rhosigma_a_aa"] = out.v2rhosigma_a_aa;
                 d["v2rhosigma_a_ab"] = out.v2rhosigma_a_ab;
                 d["v2rhosigma_a_bb"] = out.v2rhosigma_a_bb;
                 d["v2rhosigma_b_aa"] = out.v2rhosigma_b_aa;
                 d["v2rhosigma_b_ab"] = out.v2rhosigma_b_ab;
                 d["v2rhosigma_b_bb"] = out.v2rhosigma_b_bb;
                 d["v2sigma2_aa_aa"] = out.v2sigma2_aa_aa;
                 d["v2sigma2_aa_ab"] = out.v2sigma2_aa_ab;
                 d["v2sigma2_aa_bb"] = out.v2sigma2_aa_bb;
                 d["v2sigma2_ab_ab"] = out.v2sigma2_ab_ab;
                 d["v2sigma2_ab_bb"] = out.v2sigma2_ab_bb;
                 d["v2sigma2_bb_bb"] = out.v2sigma2_bb_bb;
                 return d;
             },
             py::arg("rho_a"), py::arg("rho_b"),
             py::arg("sigma_aa"), py::arg("sigma_ab"),
             py::arg("sigma_bb"),
             "Polarized LDA/GGA XC kernel for UKS analytic Hessian. "
             "Returns a dict with the 15 libxc second-derivative "
             "blocks. Raises on meta-GGA functionals.")
        .def("eval_polarised",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho_a,
                const Eigen::VectorXd& rho_b,
                const Eigen::VectorXd& sigma_aa,
                const Eigen::VectorXd& sigma_ab,
                const Eigen::VectorXd& sigma_bb) {
                 Eigen::VectorXd exc, v_rho_a, v_rho_b;
                 Eigen::VectorXd v_sigma_aa, v_sigma_ab, v_sigma_bb;
                 f.eval_polarised(rho_a, rho_b,
                                   sigma_aa, sigma_ab, sigma_bb,
                                   exc, v_rho_a, v_rho_b,
                                   v_sigma_aa, v_sigma_ab, v_sigma_bb);
                 return py::make_tuple(std::move(exc),
                                       std::move(v_rho_a),
                                       std::move(v_rho_b),
                                       std::move(v_sigma_aa),
                                       std::move(v_sigma_ab),
                                       std::move(v_sigma_bb));
             },
             py::arg("rho_a"), py::arg("rho_b"),
             py::arg("sigma_aa"), py::arg("sigma_ab"), py::arg("sigma_bb"),
             "Polarized XC energy density + first derivatives on a grid. "
             "Returns (exc, v_rho_a, v_rho_b, v_sigma_aa, v_sigma_ab, "
             "v_sigma_bb). For LDA, sigma_* may be empty and the v_sigma_* "
             "outputs will be empty too. The Functional must be constructed "
             "with spin=2.")
        .def("eval_unpolarised_mgga",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho,
                const Eigen::VectorXd& sigma,
                const Eigen::VectorXd& tau) {
                 Eigen::VectorXd exc, v_rho, v_sigma, v_tau;
                 f.eval_unpolarised_mgga(rho, sigma, tau,
                                         exc, v_rho, v_sigma, v_tau);
                 return py::make_tuple(std::move(exc),
                                       std::move(v_rho),
                                       std::move(v_sigma),
                                       std::move(v_tau));
             },
             py::arg("rho"), py::arg("sigma"), py::arg("tau"),
             "Meta-GGA evaluation (τ-dependent). Returns "
             "(exc, v_rho, v_sigma, v_tau). τ = ½ Σ_i |∇ψ_i(r)|² is the "
             "total (both spins) for the unpolarised path.")
        .def("eval_polarised_mgga",
             [](const vibeqc::Functional& f,
                const Eigen::VectorXd& rho_a,
                const Eigen::VectorXd& rho_b,
                const Eigen::VectorXd& sigma_aa,
                const Eigen::VectorXd& sigma_ab,
                const Eigen::VectorXd& sigma_bb,
                const Eigen::VectorXd& tau_a,
                const Eigen::VectorXd& tau_b) {
                 Eigen::VectorXd exc, v_rho_a, v_rho_b;
                 Eigen::VectorXd v_sigma_aa, v_sigma_ab, v_sigma_bb;
                 Eigen::VectorXd v_tau_a, v_tau_b;
                 f.eval_polarised_mgga(rho_a, rho_b,
                                       sigma_aa, sigma_ab, sigma_bb,
                                       tau_a, tau_b,
                                       exc, v_rho_a, v_rho_b,
                                       v_sigma_aa, v_sigma_ab, v_sigma_bb,
                                       v_tau_a, v_tau_b);
                 return py::make_tuple(std::move(exc),
                                       std::move(v_rho_a),
                                       std::move(v_rho_b),
                                       std::move(v_sigma_aa),
                                       std::move(v_sigma_ab),
                                       std::move(v_sigma_bb),
                                       std::move(v_tau_a),
                                       std::move(v_tau_b));
             },
             py::arg("rho_a"), py::arg("rho_b"),
             py::arg("sigma_aa"), py::arg("sigma_ab"), py::arg("sigma_bb"),
             py::arg("tau_a"), py::arg("tau_b"),
             "Polarized meta-GGA evaluation. Returns "
             "(exc, v_rho_a, v_rho_b, v_sigma_aa, v_sigma_ab, v_sigma_bb, "
             "v_tau_a, v_tau_b). τ_σ = ½ Σ_iσ |∇ψ_iσ(r)|² is per-spin.");

    m.def("evaluate_uks_xc_potential",
          [](const vibeqc::Functional& functional,
             const vibeqc::BasisSet& basis,
             const vibeqc::Grid& grid,
             const Eigen::MatrixXd& density_alpha,
             const Eigen::MatrixXd& density_beta) {
              vibeqc::UKSXCPotential xc;
              {
                  py::gil_scoped_release release;
                  xc = vibeqc::evaluate_uks_xc_potential(
                      functional, basis, grid,
                      density_alpha, density_beta);
              }
              return py::make_tuple(
                  std::move(xc.V_alpha), std::move(xc.V_beta), xc.energy);
          },
          py::arg("functional"), py::arg("basis"), py::arg("grid"),
          py::arg("density_alpha"), py::arg("density_beta"),
          "Project one spin-resolved XC energy and its first derivatives "
          "onto AO matrices. Supports both libxc and registered full-grid "
          "providers; returns (V_alpha, V_beta, E_xc).");

    // ---- Runtime functional-alias registration ---------------------
    m.def("define_functional", &vibeqc::register_functional_alias,
          py::arg("name"), py::arg("components"),
          py::arg("hf_exchange_fraction") = 0.0,
          "Register a user-defined functional alias at runtime.\n"
          "components is a list of (libxc_name, weight) pairs.\n"
          "hf_exchange_fraction is the global HF-exchange fraction.\n"
          "After registration, Functional(name) resolves the alias.");

    m.def("_dynamic_alias_available", &vibeqc::dynamic_alias_available,
          py::arg("name"),
          "Return whether name belongs to the libxc-style dynamic-alias "
          "registry. Full-grid external providers are intentionally not "
          "reported by this compatibility query.");

    m.def("define_external_functional",
          [](const std::string& name, py::object callback,
             double hf_exchange_fraction, int capability_version,
             const std::string& required_grid_profile) {
              if (!PyCallable_Check(callback.ptr())) {
                  throw std::invalid_argument(
                      "define_external_functional: callback must be callable");
              }
              vibeqc::ExternalXCCapabilities capabilities;
              capabilities.version = capability_version;
              capabilities.required_grid_profile = required_grid_profile;
              vibeqc::register_external_functional(
                  name,
                  std::make_shared<PythonExternalXCProvider>(
                      std::move(callback)),
                  hf_exchange_fraction,
                  std::move(capabilities));
          },
          py::arg("name"), py::arg("callback"),
          py::arg("hf_exchange_fraction") = 0.0,
          py::kw_only(),
          py::arg("capability_version") =
              vibeqc::kExternalXCCapabilityVersion,
          py::arg("required_grid_profile") = "",
          "Register a full-grid nonlocal XC callback. The callback receives "
          "one atom-major feature dict and returns the scalar energy plus "
          "full-energy adjoints for rho, Cartesian grad(rho), and tau in "
          "both spin channels. Capability protocol version 1 optionally "
          "declares required_grid_profile ('generic' or 'pyscf-level3'); the "
          "common provider boundary rejects a Grid built with another "
          "profile before invoking the callback.");

    m.def("_define_external_functional_family",
          [](const std::vector<std::string>& names, py::object callback,
             double hf_exchange_fraction, int capability_version,
             const std::string& required_grid_profile) {
              if (!PyCallable_Check(callback.ptr())) {
                  throw std::invalid_argument(
                      "_define_external_functional_family: callback must be "
                      "callable");
              }
              vibeqc::ExternalXCCapabilities capabilities;
              capabilities.version = capability_version;
              capabilities.required_grid_profile = required_grid_profile;
              vibeqc::register_external_functional_family(
                  names,
                  std::make_shared<PythonExternalXCProvider>(
                      std::move(callback)),
                  hf_exchange_fraction,
                  std::move(capabilities));
          },
          py::arg("names"), py::arg("callback"),
          py::arg("hf_exchange_fraction") = 0.0,
          py::kw_only(),
          py::arg("capability_version") =
              vibeqc::kExternalXCCapabilityVersion,
          py::arg("required_grid_profile") = "",
          "Atomically register one full-grid nonlocal XC callback under an "
          "immutable family of names. This private adapter surface either "
          "registers every name or leaves the registry unchanged.");

    m.def("_external_functional_registration",
          [](const std::string& name) -> py::object {
              const auto registration =
                  vibeqc::find_external_functional_registration(name);
              if (!registration.has_value()) return py::none();

              py::dict result;
              const auto python_provider =
                  std::dynamic_pointer_cast<PythonExternalXCProvider>(
                      registration->provider);
              if (python_provider) {
                  result["callback"] = python_provider->callback();
              } else {
                  result["callback"] = py::none();
              }
              result["hf_exchange_fraction"] =
                  registration->hf_exchange_fraction;
              result["capability_version"] =
                  registration->capabilities.version;
              result["required_grid_profile"] =
                  registration->capabilities.required_grid_profile;
              return result;
          },
          py::arg("name"),
          "Return an immutable snapshot of a registered external XC "
          "provider, including its Python callback when applicable. This "
          "private query supports safe provider adoption after a Python "
          "module purge; it cannot replace or remove registrations.");

    // VV10 nonlocal correlation (Vydrov-Van Voorhis 2010). Exposed for
    // unit-validating the nonlocal kernel against an independent
    // reference on an arbitrary grid; the SCF drivers call the same C++
    // routine internally when Functional.needs_vv10 is set.
    m.def("compute_vv10",
          [](const Eigen::MatrixX3d& points,
             const Eigen::VectorXd& weights,
             const Eigen::VectorXd& rho,
             const Eigen::VectorXd& sigma,
             double b, double C) {
              const vibeqc::VV10Result r =
                  vibeqc::compute_vv10(points, weights, rho, sigma, b, C);
              return py::make_tuple(r.energy, r.v_rho, r.v_sigma);
          },
          py::arg("points"), py::arg("weights"), py::arg("rho"),
          py::arg("sigma"), py::arg("b"), py::arg("C"),
          "Evaluate VV10 nonlocal correlation on a grid. Inputs: points "
          "(n,3) bohr, weights (n,), rho (n,), sigma = |grad rho|^2 (n,), "
          "and the VV10 parameters b, C. Returns (energy, v_rho, v_sigma) "
          "where energy is the scalar E_c^nl and v_rho / v_sigma are the "
          "per-point potentials in libxc integrand convention (not "
          "weight-multiplied).");

    // B97M-form semilocal XC energy density (energy-only) for the ωB97M(2)
    // xDH double hybrid. terms_* are lists of (coeff, w_power, u_power).
    m.def("compute_b97m_semilocal_exc",
          [](const Eigen::VectorXd& rho_a, const Eigen::VectorXd& rho_b,
             const Eigen::VectorXd& sigma_aa, const Eigen::VectorXd& sigma_ab,
             const Eigen::VectorXd& sigma_bb,
             const Eigen::VectorXd& tau_a, const Eigen::VectorXd& tau_b,
             double omega, double gamma_x, double gamma_ss, double gamma_os,
             const std::vector<std::tuple<double, int, int>>& terms_x,
             const std::vector<std::tuple<double, int, int>>& terms_ss,
             const std::vector<std::tuple<double, int, int>>& terms_os) {
              auto conv = [](const std::vector<std::tuple<double, int, int>>& in) {
                  std::vector<vibeqc::B97MTerm> out;
                  out.reserve(in.size());
                  for (const auto& t : in) {
                      out.push_back({std::get<0>(t), std::get<1>(t),
                                     std::get<2>(t)});
                  }
                  return out;
              };
              vibeqc::B97MParamsStatic sp;
              sp.omega = omega;
              sp.gamma_x = gamma_x; sp.gamma_ss = gamma_ss; sp.gamma_os = gamma_os;
              sp.terms_x = conv(terms_x);
              sp.terms_ss = conv(terms_ss);
              sp.terms_os = conv(terms_os);
              return vibeqc::b97m_semilocal_exc(
                  rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, sp);
          },
          py::arg("rho_a"), py::arg("rho_b"), py::arg("sigma_aa"),
          py::arg("sigma_ab"), py::arg("sigma_bb"), py::arg("tau_a"),
          py::arg("tau_b"), py::arg("omega"), py::arg("gamma_x"),
          py::arg("gamma_ss"), py::arg("gamma_os"), py::arg("terms_x"),
          py::arg("terms_ss"), py::arg("terms_os"),
          "Generic B97M-form semilocal XC energy density (energy-only) for "
          "the ωB97M(2) xDH double hybrid. terms_* = lists of "
          "(coeff, w_power, u_power). Returns exc(g) = ρ(g)·ε_xc^semilocal "
          "(integrand form). Validate by feeding ωB97M-V's coefficients and "
          "matching Functional('wb97m-v').eval_polarised_mgga.");

    // B97M semilocal potential (v_rho/v_sigma/v_tau) for the
    // self-consistent ωB97M(2) variant. Same args as the energy version;
    // returns (v_rho_a, v_rho_b, v_sigma_aa, v_sigma_bb, v_tau_a, v_tau_b).
    m.def("compute_b97m_semilocal_vxc",
          [](const Eigen::VectorXd& rho_a, const Eigen::VectorXd& rho_b,
             const Eigen::VectorXd& sigma_aa, const Eigen::VectorXd& sigma_ab,
             const Eigen::VectorXd& sigma_bb,
             const Eigen::VectorXd& tau_a, const Eigen::VectorXd& tau_b,
             double omega, double gamma_x, double gamma_ss, double gamma_os,
             const std::vector<std::tuple<double, int, int>>& terms_x,
             const std::vector<std::tuple<double, int, int>>& terms_ss,
             const std::vector<std::tuple<double, int, int>>& terms_os) {
              auto conv = [](const std::vector<std::tuple<double, int, int>>& in) {
                  std::vector<vibeqc::B97MTerm> out;
                  out.reserve(in.size());
                  for (const auto& t : in)
                      out.push_back({std::get<0>(t), std::get<1>(t), std::get<2>(t)});
                  return out;
              };
              vibeqc::B97MParamsStatic sp;
              sp.omega = omega;
              sp.gamma_x = gamma_x; sp.gamma_ss = gamma_ss; sp.gamma_os = gamma_os;
              sp.terms_x = conv(terms_x);
              sp.terms_ss = conv(terms_ss);
              sp.terms_os = conv(terms_os);
              const vibeqc::B97MVxc v = vibeqc::b97m_semilocal_vxc(
                  rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, sp);
              return py::make_tuple(v.v_rho_a, v.v_rho_b, v.v_sigma_aa,
                                    v.v_sigma_bb, v.v_tau_a, v.v_tau_b);
          },
          py::arg("rho_a"), py::arg("rho_b"), py::arg("sigma_aa"),
          py::arg("sigma_ab"), py::arg("sigma_bb"), py::arg("tau_a"),
          py::arg("tau_b"), py::arg("omega"), py::arg("gamma_x"),
          py::arg("gamma_ss"), py::arg("gamma_os"), py::arg("terms_x"),
          py::arg("terms_ss"), py::arg("terms_os"),
          "B97M semilocal potential (finite-difference of the energy kernel) "
          "for self-consistent ωB97M(2). Returns (v_rho_a, v_rho_b, "
          "v_sigma_aa, v_sigma_bb, v_tau_a, v_tau_b). Validate against "
          "Functional('wb97m-v').eval_polarised_mgga's potentials.");

    // Phase D2c-KS τ_σ = ½ Σ_iσ |∇ψ_iσ(r)|² is per-spin.");

    // Phase D2c-KS — XC kernel matvec (W^XC[D_pert]) for the second-order
    // KS SCF path. Closed-shell (RKS) only in this commit; UKS path
    // exists as a factory stub that throws with a roadmap pointer. See
    // cpp/include/vibeqc/xc_kernel.hpp for the design + math.
    py::class_<vibeqc::XCKernelBuilder>(m, "XCKernelBuilder",
        "Closed-shell unpolarised XC kernel builder. Pin the reference "
        "density at construction (via make_unpolarised_xc_kernel_builder); "
        "each ``apply(D_pert)`` returns W^XC[D_pert] in AO basis.")
        .def("apply", &vibeqc::XCKernelBuilder::apply,
             py::arg("D_pert"),
             "AO-basis W^XC[D_pert] matvec. Symmetric output.");

    m.def("make_unpolarised_xc_kernel_builder",
          [](const vibeqc::Functional& func,
             const vibeqc::Grid& grid,
             const Eigen::MatrixXd& chi,
             const Eigen::MatrixXd& dchi_x,
             const Eigen::MatrixXd& dchi_y,
             const Eigen::MatrixXd& dchi_z,
             const Eigen::MatrixXd& D_used) {
              std::array<Eigen::MatrixXd, 3> dchi = {dchi_x, dchi_y, dchi_z};
              return vibeqc::make_unpolarised_xc_kernel_builder(
                  func, grid, chi, dchi, D_used);
          },
          py::arg("functional"),
          py::arg("grid"),
          py::arg("chi"),
          py::arg("dchi_x"), py::arg("dchi_y"), py::arg("dchi_z"),
          py::arg("D_used"),
          "Phase D2c-KS — build an unpolarised LDA / GGA XC kernel for "
          "the Hessian matvec at reference density D_used. The grid + "
          "chi + dchi triple comes from build_grid + "
          "evaluate_ao_with_gradient. Returns a pinned-state builder; "
          "call .apply(D_pert) for each Newton-CG matvec.");

    // Open-shell (polarised LDA) XC kernel.
    py::class_<vibeqc::UHFXCKernelBuilder>(m, "UHFXCKernelBuilder",
        "Open-shell polarised XC kernel builder. Per-spin matvec "
        "couples α and β via the αβ v2rho2 cross-term.")
        .def("apply",
             [](const vibeqc::UHFXCKernelBuilder& k,
                const Eigen::MatrixXd& D_pert_alpha,
                const Eigen::MatrixXd& D_pert_beta) {
                 auto out = k.apply(D_pert_alpha, D_pert_beta);
                 return py::make_tuple(std::move(out.alpha),
                                       std::move(out.beta));
             },
             py::arg("D_pert_alpha"), py::arg("D_pert_beta"),
             "Returns (W^XC_α, W^XC_β), each (n_bf, n_bf), symmetric.")
        .def_property_readonly(
             "batch_workers_used",
             &vibeqc::UHFXCKernelBuilder::batch_workers_used,
             "Maximum concurrent grid batches actually used.");

    m.def("make_polarised_lda_xc_kernel_builder",
          &vibeqc::make_polarised_lda_xc_kernel_builder,
          py::arg("functional"),
          py::arg("grid"),
          py::arg("chi"),
          py::arg("D_used_alpha"),
          py::arg("D_used_beta"),
          "Phase D2c-KS-UHF — build an open-shell polarised LDA XC "
          "kernel for the Hessian matvec at reference per-spin "
          "densities. Raises on GGA functionals (polarised GGA fxc "
          "isn't plumbed in Functional yet — Phase 17e).");

    m.def("make_polarised_gga_xc_kernel_builder",
          [](const vibeqc::Functional& func,
             const vibeqc::Grid& grid,
             const Eigen::MatrixXd& chi,
             const Eigen::MatrixXd& dchi_x,
             const Eigen::MatrixXd& dchi_y,
             const Eigen::MatrixXd& dchi_z,
             const Eigen::MatrixXd& D_used_alpha,
             const Eigen::MatrixXd& D_used_beta) {
              std::array<Eigen::MatrixXd, 3> dchi = {dchi_x, dchi_y, dchi_z};
              return vibeqc::make_polarised_gga_xc_kernel_builder(
                  func, grid, chi, dchi, D_used_alpha, D_used_beta);
          },
          py::arg("functional"),
          py::arg("grid"),
          py::arg("chi"),
          py::arg("dchi_x"), py::arg("dchi_y"), py::arg("dchi_z"),
          py::arg("D_used_alpha"),
          py::arg("D_used_beta"),
          "Phase D2c-KS-UHF — build an open-shell polarised GGA XC "
          "kernel for the Hessian matvec at reference per-spin "
          "densities.");

    m.def("make_polarised_xc_kernel_builder",
          [](const vibeqc::Functional& func,
             const vibeqc::Grid& grid,
             const Eigen::MatrixXd& chi,
             const Eigen::MatrixXd& dchi_x,
             const Eigen::MatrixXd& dchi_y,
             const Eigen::MatrixXd& dchi_z,
             const Eigen::MatrixXd& D_used_alpha,
             const Eigen::MatrixXd& D_used_beta) {
              std::array<Eigen::MatrixXd, 3> dchi = {dchi_x, dchi_y, dchi_z};
              return vibeqc::make_polarised_xc_kernel_builder(
                  func, grid, chi, dchi, D_used_alpha, D_used_beta);
          },
          py::arg("functional"),
          py::arg("grid"),
          py::arg("chi"),
          py::arg("dchi_x"), py::arg("dchi_y"), py::arg("dchi_z"),
          py::arg("D_used_alpha"),
          py::arg("D_used_beta"),
          "Build an open-shell polarised XC kernel builder, dispatching "
          "to LDA or GGA based on the functional kind.");

    m.def("make_batched_polarised_xc_kernel_builder",
          &vibeqc::make_batched_polarised_xc_kernel_builder,
          py::arg("functional"),
          py::arg("grid"),
          py::arg("basis"),
          py::arg("D_used_alpha"),
          py::arg("D_used_beta"),
          py::arg("max_batch_workers") = 1,
          "Memory-bounded open-shell polarised XC kernel builder used by "
          "the default-on UKS internal stability analysis. Algebraically "
          "identical to make_polarised_xc_kernel_builder, but never holds "
          "whole-grid (n_pts, n_bf) AO tables: construction and every "
          "apply() re-evaluate AOs in kMolecularXcGridBatchSize-point "
          "slices. Takes the basis instead of precomputed chi/dchi.");

    m.def("evaluate_ao_with_gradient",
          [](const vibeqc::BasisSet& basis,
             const Eigen::MatrixX3d& points) {
              auto r = vibeqc::evaluate_ao_with_gradient(basis, points);
              // Return values and the three gradient matrices as a plain
              // Python tuple. Construct outside the GIL-released region so
              // pybind11 can convert Eigen -> NumPy safely.
              py::gil_scoped_acquire gil;
              return py::make_tuple(r.values,
                                    r.gradients[0],
                                    r.gradients[1],
                                    r.gradients[2]);
          },
          py::arg("basis"), py::arg("points"),
          "Evaluate AO values AND first spatial derivatives. Returns a tuple "
          "(values, grad_x, grad_y, grad_z), each an (n_points, n_basis) matrix.");

    m.def("evaluate_ao_with_hessian",
          [](const vibeqc::BasisSet& basis,
             const Eigen::MatrixX3d& points) {
              auto r = vibeqc::evaluate_ao_with_hessian(basis, points);
              py::gil_scoped_acquire gil;
              return py::make_tuple(r.values,
                                    r.gradients[0],
                                    r.gradients[1],
                                    r.gradients[2],
                                    r.hessians[0],
                                    r.hessians[1],
                                    r.hessians[2],
                                    r.hessians[3],
                                    r.hessians[4],
                                    r.hessians[5]);
          },
          py::arg("basis"), py::arg("points"),
          "Evaluate AO values, first AND second derivatives. "
          "Returns tuple (values, grad_x, grad_y, grad_z, "
          "hess_xx, hess_xy, hess_xz, hess_yy, hess_yz, hess_zz), "
          "each (n_points, n_basis).");

    // ----- UHF -----------------------------------------------------------
    py::class_<vibeqc::UHFOptions>(m, "UHFOptions")
        .def(py::init<>())
        .def(py::init<const vibeqc::UHFOptions&>(), py::arg("other"),
             "Value copy. Used by alternate-guess retries so ECP, SCF, and "
             "implicit option state remain identical across candidates.")
        .def_property(
            "max_iter",
            [](const vibeqc::UHFOptions& options) { return options.max_iter; },
            [](vibeqc::UHFOptions& options, py::handle value) {
                set_molecular_scf_max_iter(options, value, "UHFOptions");
            })
        .def_readwrite("conv_tol_energy",
                       &vibeqc::UHFOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad", &vibeqc::UHFOptions::conv_tol_grad)
        .def_readwrite("damping", &vibeqc::UHFOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::UHFOptions::dynamic_damping,
                       "See RHFOptions::dynamic_damping. Default True.")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::UHFOptions::dynamic_damping_min)
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::UHFOptions::dynamic_damping_max)
        .def_readwrite("fock_mixing",
                       &vibeqc::UHFOptions::fock_mixing,
                       "Per-spin Fock matrix mixing fraction. See "
                       "RHFOptions.fock_mixing.")
        .def_readwrite("use_diis", &vibeqc::UHFOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::UHFOptions::diis_start_iter)
        .def_readwrite("diis_subspace_size",
                       &vibeqc::UHFOptions::diis_subspace_size)
        .def_readwrite("scf_accelerator",
                       &vibeqc::UHFOptions::scf_accelerator,
                       "SCF Fock-extrapolation accelerator. See "
                       "RHFOptions.scf_accelerator. The EDIIS / "
                       "EDIIS_DIIS variants apply a single coefficient "
                       "set to both spins (the energy functional couples "
                       "them through E_total).")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::UHFOptions::ediis_diis_switch_threshold,
                       "See RHFOptions.ediis_diis_switch_threshold. "
                       "Default 1e-1.")
        .def_readwrite("diis_restart_tau",
                       &vibeqc::UHFOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ. See "
                       "RHFOptions.diis_restart_tau. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::UHFOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ. See "
                       "RHFOptions.diis_adaptive_delta. Default 1e-4.")
        .def_readwrite("initial_guess",
                       &vibeqc::UHFOptions::initial_guess)
        .def_readwrite("read_density_alpha",
                       &vibeqc::UHFOptions::read_density_alpha,
                       "Alpha starting density for initial_guess=READ. Set by "
                       "the vibeqc.run_uhf wrapper from read_path / read_from.")
        .def_readwrite("read_density_beta",
                       &vibeqc::UHFOptions::read_density_beta,
                       "Beta starting density for initial_guess=READ.")
        .def_readwrite("read_path",
                       &vibeqc::UHFOptions::read_path,
                       "READ source file (.qvf / .molden) for the run_uhf "
                       "wrapper; empty restarts from an in-memory read_from.")
        .def_readwrite("atomic_spins",
                       &vibeqc::UHFOptions::atomic_spins,
                       "ATOMSPIN: per-atom spin seed (list of +1/-1/0 in atom "
                       "order) for a broken-symmetry SAD start. Empty = "
                       "spin-symmetric. Requires the SAD guess.")
        .def_readwrite("spinlock_mode",
                       &vibeqc::UHFOptions::spinlock_mode,
                       "SPINLOCK mode: OFF (default), SPIN_SCHEDULE (lock "
                       "n_alpha-n_beta=spinlock_value for spinlock_iterations "
                       "cycles then release), or PATTERN_HOLD (MOM-hold the "
                       "seeded broken-symmetry pattern for spinlock_iterations).")
        .def_readwrite("spinlock_iterations",
                       &vibeqc::UHFOptions::spinlock_iterations,
                       "SPINLOCK: number of initial SCF cycles to hold (both "
                       "modes). 0 = off.")
        .def_readwrite("spinlock_value",
                       &vibeqc::UHFOptions::spinlock_value,
                       "SPINLOCK SPIN_SCHEDULE: the n_alpha-n_beta to lock "
                       "during the first spinlock_iterations cycles.")
        .def_readwrite("linear_dep_threshold",
                       &vibeqc::UHFOptions::linear_dep_threshold)
        .def_readwrite("ecp_centers",
                       &vibeqc::UHFOptions::ecp_centers,
                       "List[ECPCenter]. See RHFOptions.ecp_centers.")
        .def_readwrite("ecp_library",
                       &vibeqc::UHFOptions::ecp_library,
                       "ECP XML library name. See RHFOptions.ecp_library.")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::UHFOptions::ecp_primitive_blocks,
                       "Inline primitive ECP blocks. See RHFOptions.")
        .def_readwrite("ecp_primitive_centers",
                       &vibeqc::UHFOptions::ecp_primitive_centers,
                       "Cartesian centers (bohr) matching ecp_primitive_blocks.")
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::UHFOptions::ecp_effective_charges,
                       "Per-atom effective nuclear charges for inline ECPs.")
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::UHFOptions::ecp_total_ncore,
                       "Total number of core electrons replaced by inline ECPs.")
        .def_readwrite("level_shift",
                       &vibeqc::UHFOptions::level_shift,
                       "Phase C1a-2 — per-spin Saunders-Hillier level "
                       "shift (Hartree). Formula: "
                       "F_σ + b·S − b·(S·D_σ·S). See "
                       "RHFOptions.level_shift. Default 0.0.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::UHFOptions::level_shift_warmup_cycles,
                       "Auto-reducing level-shift warm-up (per-spin). "
                       "See RHFOptions.level_shift_warmup_cycles. "
                       "Default -1 (auto).")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::UHFOptions::level_shift_schedule,
                       "Explicit per-iteration shift curve (per-spin). "
                       "See RHFOptions.level_shift_schedule.")
        .def_readwrite("auto_level_shift_on_oscillation",
                       &vibeqc::UHFOptions::auto_level_shift_on_oscillation,
                       "Auto level-shift on SCF oscillation. "
                       "See RHFOptions.auto_level_shift_on_oscillation. "
                       "Applied per-spin. Default True.")
        .def_readwrite("auto_level_shift_value",
                       &vibeqc::UHFOptions::auto_level_shift_value,
                       "Level shift engaged on detected oscillation. "
                       "See RHFOptions.auto_level_shift_value. "
                       "Default 0.3 Ha.")
        .def_readwrite("oscillation_detect_start_iter",
                       &vibeqc::UHFOptions::oscillation_detect_start_iter,
                       "See RHFOptions.oscillation_detect_start_iter. "
                       "Default 10.")
        .def_readwrite("oscillation_window",
                       &vibeqc::UHFOptions::oscillation_window,
                       "See RHFOptions.oscillation_window. Default 10.")
        .def_readwrite("oscillation_min_sign_flips",
                       &vibeqc::UHFOptions::oscillation_min_sign_flips,
                       "See RHFOptions.oscillation_min_sign_flips. "
                       "Default 5.")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::UHFOptions::quadratic_fallback_iter,
                       "Phase C1c — per-spin second-order SCF "
                       "fallback. See RHFOptions.quadratic_fallback_iter; "
                       "the Newton step runs independently on the α "
                       "and β MO sets. Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::UHFOptions::quadratic_fallback_shift,
                       "C1c orbital-rotation denominator shift. "
                       "See RHFOptions.quadratic_fallback_shift.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::UHFOptions::quadratic_fallback_max_step,
                       "C1c per-step ‖κ‖ trust-region cap. "
                       "See RHFOptions.quadratic_fallback_max_step.")
        .def_readwrite("newton_threshold",
                       &vibeqc::UHFOptions::newton_threshold,
                       "Phase D2c — Newton activation threshold "
                       "(max-over-spins commutator norm). See "
                       "RHFOptions.newton_threshold. The open-shell "
                       "Newton step couples α and β rotations through "
                       "the shared J in the orbital Hessian. "
                       "Default 0.0 (disabled).")
        .def_readwrite("newton_opts",
                       &vibeqc::UHFOptions::newton_opts,
                       "Knobs for the open-shell Newton step (CG cap, "
                       "tolerance, trust radius). Same NewtonOptions "
                       "type as RHF; the trust radius applies to the "
                       "combined ‖(κ_α, κ_β)‖_F.")
        .def_readwrite("soscf_threshold",
                       &vibeqc::UHFOptions::soscf_threshold,
                       "Phase D2d Neese SOSCF activation threshold "
                       "for open-shell. Diagonal-dominant Hessian → "
                       "per-spin AH eigsolves. Mutually exclusive "
                       "with Newton. Default 0.0 (disabled).")
        .def_readwrite("soscf_opts",
                       &vibeqc::UHFOptions::soscf_opts,
                       "Knobs for the open-shell SOSCF step (trust "
                       "radius). Same SOSCFOptions type as RHF.")
        .def_readwrite("trah_threshold",
                       &vibeqc::UHFOptions::trah_threshold,
                       "Phase D2e TRAH activation threshold for open-"
                       "shell. Same coupled per-spin CG as Newton + "
                       "adaptive trust radius. Priority: quadratic > "
                       "Newton > TRAH > SOSCF. Default 0.0 (disabled).")
        .def_readwrite("trah_opts",
                       &vibeqc::UHFOptions::trah_opts,
                       "Knobs for the open-shell TRAH step (CG cap + "
                       "trust-region schedule). Same TRAHOptions type "
                       "as RHF; trust radius applies to combined "
                       "‖(κ_α, κ_β)‖_F.")
        .def_readwrite("restart_opts",
                       &vibeqc::UHFOptions::restart_opts,
                       "Deterministic orbital-rotation restart options. "
                       "See SCFRestartOptions.")
        .def_readwrite("density_fit",
                       &vibeqc::UHFOptions::density_fit,
                       "Enable density fitting (RI) for the per-spin "
                       "Fock build: J(D_total) and K(D_α/D_β) via "
                       "the precomputed B-tensor instead of the four-"
                       "index ERI. See RHFOptions.density_fit. "
                       "Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::UHFOptions::aux_basis,
                       "Auxiliary basis name for density fitting. "
                       "See RHFOptions.aux_basis.")
        .def_readwrite("scf_mode",
                       &vibeqc::UHFOptions::scf_mode,
                       "SCF Fock-build mode (SCFMode). See "
                       "RHFOptions.scf_mode.")
        .def_readwrite("scf_mode_auto_threshold",
                       &vibeqc::UHFOptions::scf_mode_auto_threshold,
                       "AUTO cutoff (n_bf). See "
                       "RHFOptions.scf_mode_auto_threshold.")
        .def_readwrite("schwarz_threshold",
                       &vibeqc::UHFOptions::schwarz_threshold,
                       "Per-quartet Schwarz skip bound for DIRECT mode. "
                       "See RHFOptions.schwarz_threshold.")
        .def_readwrite("incremental_fock",
                       &vibeqc::UHFOptions::incremental_fock,
                       "Enable incremental ΔP Fock build for DIRECT. "
                       "See RHFOptions.incremental_fock.")
        .def_readwrite("incremental_fock_reset_freq",
                       &vibeqc::UHFOptions::incremental_fock_reset_freq,
                       "Full-rebuild frequency for incremental_fock. "
                       "See RHFOptions.incremental_fock_reset_freq.")
        .def_readwrite("schwarz_threshold_loose",
                       &vibeqc::UHFOptions::schwarz_threshold_loose,
                       "Two-phase Schwarz loose threshold. "
                       "See RHFOptions.schwarz_threshold_loose.")
        .def_readwrite("schwarz_threshold_tighten_at",
                       &vibeqc::UHFOptions::schwarz_threshold_tighten_at,
                       "Grad-norm cutoff for the loose→tight transition. "
                       "See RHFOptions.schwarz_threshold_tighten_at.")
        .def_readwrite("cosx",
                       &vibeqc::UHFOptions::cosx,
                       "Use chain-of-spheres exchange for K. See "
                       "RHFOptions.cosx. Default False.")
        .def_readwrite("cosx_grid",
                       &vibeqc::UHFOptions::cosx_grid,
                       "GridOptions for the COSX grid. See "
                       "RHFOptions.cosx_grid.")
        .def_readwrite("cosx_variant",
                       &vibeqc::UHFOptions::cosx_variant,
                       "COSX K-matrix variant selector. See "
                       "RHFOptions.cosx_variant.")
        .def_readwrite("cosx_grid_level",
                       &vibeqc::UHFOptions::cosx_grid_level,
                       "COSX GridX selector. See RHFOptions.cosx_grid_level.")
        .def_readwrite("dft_plus_u_sites",
                       &vibeqc::UHFOptions::dft_plus_u_sites,
                       "Internal: list of _HubbardSiteCxx. Populated by "
                       "the vibeqc.run_uhf Python wrapper.")
        .def_readwrite("dft_plus_u_ao_groups",
                       &vibeqc::UHFOptions::dft_plus_u_ao_groups,
                       "Internal: parallel AO-index lists per site.")
        .def_readwrite("use_davidson",
                       &vibeqc::UHFOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::UHFOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::UHFOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.")
        .def_readwrite("stability_check",
                       &vibeqc::UHFOptions::stability_check,
                       "Post-convergence internal (real UHF->UHF) "
                       "stability analysis with corrective "
                       "rotate-and-reconverge restarts (Lehtola 2020 "
                       "Sec. 10 / Seeger-Pople 1977). Default True; set "
                       "False for the legacy single-SCF behaviour.")
        .def_readwrite("stability_tol",
                       &vibeqc::UHFOptions::stability_tol,
                       "Instability threshold: unstable when the lowest "
                       "internal-Hessian eigenvalue < -stability_tol. "
                       "Default 1e-4.")
        .def_readwrite("stability_max_retries",
                       &vibeqc::UHFOptions::stability_max_retries,
                       "Maximum rotate-and-reconverge escapes before the "
                       "instability is reported unresolved. Default 3.")
        .def_readwrite("stability_davidson_max_iter",
                       &vibeqc::UHFOptions::stability_davidson_max_iter,
                       "Matvec cap for each stability Davidson solve "
                       "(one J + two K builds per matvec). Default 60.")
        .def_readwrite("multi_guess_seeds",
                       &vibeqc::UHFOptions::multi_guess_seeds,
                       "Additional PRNG seeds for multi-start convergence "
                       "(BUG 88). Each seed triggers a deterministic "
                       "restart; lowest-energy stable solution returned. "
                       "Empty (default) = single SCF.");

    py::class_<vibeqc::UHFResult>(m, "UHFResult")
        .def_readonly("restart_basis", &vibeqc::UHFResult::restart_basis)
        .def_readwrite("guess_selection", &vibeqc::UHFResult::guess_selection)
        .def_readonly("energy", &vibeqc::UHFResult::energy,
                      "Total HF energy (Hartree).")
        .def_readonly("e_electronic", &vibeqc::UHFResult::e_electronic)
        .def_readonly("ecp_operator_applied",
                      &vibeqc::UHFResult::ecp_operator_applied,
                      "True when the SCF Hamiltonian included an ECP "
                      "operator, including a zero-core model potential.")
        .def_readonly("ecp_provenance_verified",
                      &vibeqc::UHFResult::ecp_provenance_verified,
                      "True when exact high-level molecular ECP provenance "
                      "is available; false for a caller-supplied Hcore.")
        .def_readonly("ecp_xml_centers", &vibeqc::UHFResult::ecp_xml_centers,
                      "Exact XML ECP centers used by the SCF; empty for "
                      "all-electron and inline-primitive references.")
        .def_readonly("ecp_xml_library", &vibeqc::UHFResult::ecp_xml_library,
                      "Normalized XML ECP library used by the SCF; empty "
                      "for all-electron and inline-primitive references.")
        .def_readonly("ecp_primitive_blocks",
                      &vibeqc::UHFResult::ecp_primitive_blocks,
                      "Exact inline ECP primitive blocks used by the SCF.")
        .def_readonly("ecp_primitive_centers",
                      &vibeqc::UHFResult::ecp_primitive_centers,
                      "Inline ECP centers paired with ecp_primitive_blocks.")
        .def_readonly("ecp_effective_charges",
                      &vibeqc::UHFResult::ecp_effective_charges,
                      "Per-atom effective charges for the inline ECP route.")
        .def_readonly("ecp_total_ncore", &vibeqc::UHFResult::ecp_total_ncore,
                      "Physical core electrons replaced by the molecular "
                      "ECP used for this SCF reference; zero otherwise.")
        .def_readonly("e_dft_plus_u", &vibeqc::UHFResult::e_dft_plus_u,
                      "Dudarev DFT+U contribution to the total energy "
                      "(Hartree). 0 unless UHFOptions.dft_plus_u_sites "
                      "is non-empty. Sum of per-spin contributions.")
        .def_readonly("n_iter", &vibeqc::UHFResult::n_iter)
        .def_readonly("converged", &vibeqc::UHFResult::converged)
        .def_readonly("s_squared", &vibeqc::UHFResult::s_squared,
                      "<S^2> expectation value (spin-contamination diagnostic).")
        .def_readonly("s_squared_ideal",
                      &vibeqc::UHFResult::s_squared_ideal,
                      "S(S+1) for the requested multiplicity.")
        .def_property_readonly(
            "s_squared_deviation",
            [](const vibeqc::UHFResult& r) {
                return r.s_squared - r.s_squared_ideal;
            },
            "Spin contamination: <S^2> - S(S+1).")
        .def_readonly("mo_energies_alpha",
                      &vibeqc::UHFResult::mo_energies_alpha)
        .def_readonly("mo_coeffs_alpha",
                      &vibeqc::UHFResult::mo_coeffs_alpha)
        .def_readonly("density_alpha", &vibeqc::UHFResult::density_alpha)
        .def_readonly("fock_alpha", &vibeqc::UHFResult::fock_alpha)
        .def_readonly("mo_energies_beta",
                      &vibeqc::UHFResult::mo_energies_beta)
        .def_readonly("mo_coeffs_beta",
                      &vibeqc::UHFResult::mo_coeffs_beta)
        .def_readonly("density_beta", &vibeqc::UHFResult::density_beta)
        .def_readonly("fock_beta", &vibeqc::UHFResult::fock_beta)
        .def_readonly("scf_trace", &vibeqc::UHFResult::scf_trace)
        .def_readonly("stability_checked",
                      &vibeqc::UHFResult::stability_checked,
                      "True when the internal stability analysis ran on "
                      "the returned solution.")
        .def_readonly("stability_analysis_converged",
                      &vibeqc::UHFResult::stability_analysis_converged,
                      "True when the stability Davidson solve converged "
                      "(stability_eigenvalue is then trustworthy).")
        .def_readonly("stability_eigenvalue",
                      &vibeqc::UHFResult::stability_eigenvalue,
                      "Lowest internal orbital-rotation Hessian "
                      "eigenvalue at the returned solution; >= "
                      "-stability_tol means internally stable.")
        .def_readonly("n_stability_restarts",
                      &vibeqc::UHFResult::n_stability_restarts,
                      "Rotate-and-reconverge escapes taken to reach the "
                      "returned solution.")
        .def_readonly("internal_instability",
                      &vibeqc::UHFResult::internal_instability,
                      "True when the returned solution is internally "
                      "UNSTABLE (excited SCF solution) and was not / must "
                      "not be escaped -- reported loudly by the output "
                      "layer.")
        .def_readonly("stability_wall_s",
                      &vibeqc::UHFResult::stability_wall_s,
                      "Wall seconds spent in the post-convergence "
                      "internal-stability phase; 0.0 when it did not run.")
        .def_readonly("stability_cpu_s",
                      &vibeqc::UHFResult::stability_cpu_s,
                      "CPU seconds (summed over OpenMP threads) spent in "
                      "the post-convergence internal-stability phase.")
        .def("__repr__", [](const vibeqc::UHFResult& r) {
            return std::string{"UHFResult(energy="}
                   + std::to_string(r.energy)
                   + ", n_iter=" + std::to_string(r.n_iter)
                   + ", converged=" + (r.converged ? "True" : "False")
                   + ", <S^2>=" + std::to_string(r.s_squared) + ")";
        });

    // ----- RKS (closed-shell DFT) ----------------------------------------
    py::class_<vibeqc::RKSOptions>(m, "RKSOptions")
        .def(py::init<>())
        .def(py::init<const vibeqc::RKSOptions&>(), py::arg("other"),
             "Value copy. Used by routes (CPCM solvation, SAD retries) that "
             "must mutate a per-call options view without touching the "
             "caller's object, including the issue-#144 stability fields.")
        .def_readwrite("functional",  &vibeqc::RKSOptions::functional)
        .def_readwrite("grid",        &vibeqc::RKSOptions::grid)
        .def_property(
            "max_iter",
            [](const vibeqc::RKSOptions& options) { return options.max_iter; },
            [](vibeqc::RKSOptions& options, py::handle value) {
                set_molecular_scf_max_iter(options, value, "RKSOptions");
            })
        .def_readwrite("conv_tol_energy",
                       &vibeqc::RKSOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad",
                       &vibeqc::RKSOptions::conv_tol_grad)
        .def_readwrite("damping",     &vibeqc::RKSOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::RKSOptions::dynamic_damping,
                       "See RHFOptions::dynamic_damping. Default True.")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::RKSOptions::dynamic_damping_min)
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::RKSOptions::dynamic_damping_max)
        .def_readwrite("fock_mixing",
                       &vibeqc::RKSOptions::fock_mixing,
                       "Kohn-Sham matrix mixing fraction. See "
                       "RHFOptions.fock_mixing.")
        .def_readwrite("use_diis",    &vibeqc::RKSOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::RKSOptions::diis_start_iter)
        .def_readwrite("diis_subspace_size",
                       &vibeqc::RKSOptions::diis_subspace_size)
        .def_readwrite("scf_accelerator",
                       &vibeqc::RKSOptions::scf_accelerator,
                       "SCF Fock-extrapolation accelerator. See "
                       "RHFOptions.scf_accelerator.")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::RKSOptions::ediis_diis_switch_threshold,
                       "See RHFOptions.ediis_diis_switch_threshold. "
                       "Default 1e-1.")
        .def_readwrite("diis_restart_tau",
                       &vibeqc::RKSOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ. See "
                       "RHFOptions.diis_restart_tau. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::RKSOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ. See "
                       "RHFOptions.diis_adaptive_delta. Default 1e-4.")
        .def_readwrite("initial_guess",
                       &vibeqc::RKSOptions::initial_guess)
        .def_readwrite("read_density",
                       &vibeqc::RKSOptions::read_density,
                       "Starting density for initial_guess=READ. Set by the "
                       "vibeqc.run_rks wrapper from read_path / read_from.")
        .def_readwrite("read_path",
                       &vibeqc::RKSOptions::read_path,
                       "READ source file (.qvf / .molden) for the run_rks "
                       "wrapper; empty restarts from an in-memory read_from.")
        .def_readwrite("linear_dep_threshold",
                       &vibeqc::RKSOptions::linear_dep_threshold)
        .def_readwrite("ecp_centers",
                       &vibeqc::RKSOptions::ecp_centers,
                       "List[ECPCenter]. See RHFOptions.ecp_centers.")
        .def_readwrite("ecp_library",
                       &vibeqc::RKSOptions::ecp_library,
                       "ECP XML library name. See RHFOptions.ecp_library.")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::RKSOptions::ecp_primitive_blocks,
                       "Inline primitive ECP blocks. See RHFOptions.")
        .def_readwrite("ecp_primitive_centers",
                       &vibeqc::RKSOptions::ecp_primitive_centers,
                       "Cartesian centers (bohr) matching ecp_primitive_blocks.")
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::RKSOptions::ecp_effective_charges,
                       "Per-atom effective nuclear charges for inline ECPs.")
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::RKSOptions::ecp_total_ncore,
                       "Total number of core electrons replaced by inline ECPs.")
        .def_readwrite("level_shift",
                       &vibeqc::RKSOptions::level_shift,
                       "Phase C1a-2 — Saunders-Hillier level shift "
                       "(Hartree). Same formula as RHF (closed-shell "
                       "D = 2·C_occ·C_occ^T). See "
                       "RHFOptions.level_shift. Default 0.0.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::RKSOptions::level_shift_warmup_cycles,
                       "Auto-reducing level-shift warm-up. "
                       "See RHFOptions.level_shift_warmup_cycles. "
                       "Default -1 (auto).")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::RKSOptions::level_shift_schedule,
                       "Explicit per-iteration shift curve. "
                       "See RHFOptions.level_shift_schedule.")
        .def_readwrite("auto_level_shift_on_oscillation",
                       &vibeqc::RKSOptions::auto_level_shift_on_oscillation,
                       "Auto level-shift on SCF oscillation. "
                       "See RHFOptions.auto_level_shift_on_oscillation. "
                       "Default True.")
        .def_readwrite("auto_level_shift_value",
                       &vibeqc::RKSOptions::auto_level_shift_value,
                       "Level shift engaged on detected oscillation. "
                       "See RHFOptions.auto_level_shift_value. "
                       "Default 0.3 Ha.")
        .def_readwrite("oscillation_detect_start_iter",
                       &vibeqc::RKSOptions::oscillation_detect_start_iter,
                       "See RHFOptions.oscillation_detect_start_iter. "
                       "Default 10.")
        .def_readwrite("oscillation_window",
                       &vibeqc::RKSOptions::oscillation_window,
                       "See RHFOptions.oscillation_window. Default 10.")
        .def_readwrite("oscillation_min_sign_flips",
                       &vibeqc::RKSOptions::oscillation_min_sign_flips,
                       "See RHFOptions.oscillation_min_sign_flips. "
                       "Default 5.")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::RKSOptions::quadratic_fallback_iter,
                       "Phase C1c — second-order SCF fallback. "
                       "See RHFOptions.quadratic_fallback_iter. "
                       "Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::RKSOptions::quadratic_fallback_shift,
                       "C1c orbital-rotation denominator shift. "
                       "See RHFOptions.quadratic_fallback_shift.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::RKSOptions::quadratic_fallback_max_step,
                       "C1c per-step ‖κ‖ trust-region cap. "
                       "See RHFOptions.quadratic_fallback_max_step.")
        .def_readwrite("newton_threshold",
                       &vibeqc::RKSOptions::newton_threshold,
                       "Phase D2c-KS Newton activation threshold. "
                       "See RHFOptions.newton_threshold. Default 0.0 "
                       "(disabled). XC kernel built via "
                       "make_unpolarised_xc_kernel_builder; raises on "
                       "meta-GGA functionals (Phase 17e+ classification).")
        .def_readwrite("newton_opts",
                       &vibeqc::RKSOptions::newton_opts,
                       "NewtonOptions (CG knobs + trust-region cap). "
                       "See NewtonOptions.")
        .def_readwrite("soscf_threshold",
                       &vibeqc::RKSOptions::soscf_threshold,
                       "Phase D2d-KS Neese SOSCF activation threshold. "
                       "Default 0.0. No XC kernel needed (F already "
                       "carries V_xc).")
        .def_readwrite("soscf_opts",
                       &vibeqc::RKSOptions::soscf_opts,
                       "SOSCFOptions. See SOSCFOptions.")
        .def_readwrite("trah_threshold",
                       &vibeqc::RKSOptions::trah_threshold,
                       "Phase D2e-KS TRAH activation threshold. Default "
                       "0.0. Same XC kernel as Newton. Mutual-exclusion "
                       "priority: quadratic > Newton > TRAH > SOSCF.")
        .def_readwrite("trah_opts",
                       &vibeqc::RKSOptions::trah_opts,
                       "TRAHOptions (CG + Powell-ρ trust schedule).")
        .def_readwrite("restart_opts",
                       &vibeqc::RKSOptions::restart_opts,
                       "Deterministic orbital-rotation restart options. "
                       "See SCFRestartOptions.")
        .def_readwrite("density_fit",
                       &vibeqc::RKSOptions::density_fit,
                       "Enable density fitting (RI) for the J build "
                       "(and K build for hybrid functionals). See "
                       "RHFOptions.density_fit. Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::RKSOptions::aux_basis,
                       "Auxiliary basis name for density fitting. "
                       "See RHFOptions.aux_basis.")
        .def_readwrite("scf_mode",
                       &vibeqc::RKSOptions::scf_mode,
                       "SCF Fock-build mode (SCFMode). See "
                       "RHFOptions.scf_mode.")
        .def_readwrite("scf_mode_auto_threshold",
                       &vibeqc::RKSOptions::scf_mode_auto_threshold,
                       "AUTO cutoff (n_bf). See "
                       "RHFOptions.scf_mode_auto_threshold.")
        .def_readwrite("schwarz_threshold",
                       &vibeqc::RKSOptions::schwarz_threshold,
                       "Per-quartet Schwarz skip bound for DIRECT mode. "
                       "See RHFOptions.schwarz_threshold.")
        .def_readwrite("incremental_fock",
                       &vibeqc::RKSOptions::incremental_fock,
                       "Enable incremental ΔP Fock build for DIRECT. "
                       "See RHFOptions.incremental_fock.")
        .def_readwrite("incremental_fock_reset_freq",
                       &vibeqc::RKSOptions::incremental_fock_reset_freq,
                       "Full-rebuild frequency for incremental_fock. "
                       "See RHFOptions.incremental_fock_reset_freq.")
        .def_readwrite("schwarz_threshold_loose",
                       &vibeqc::RKSOptions::schwarz_threshold_loose,
                       "Two-phase Schwarz loose threshold. "
                       "See RHFOptions.schwarz_threshold_loose.")
        .def_readwrite("schwarz_threshold_tighten_at",
                       &vibeqc::RKSOptions::schwarz_threshold_tighten_at,
                       "Grad-norm cutoff for the loose→tight transition. "
                       "See RHFOptions.schwarz_threshold_tighten_at.")
        .def_readwrite("cosx",
                       &vibeqc::RKSOptions::cosx,
                       "Use chain-of-spheres exchange (COSX) for the K "
                       "build. Pair with density_fit=True for the "
                       "standard RIJCOSX hybrid-DFT acceleration. "
                       "No-op for pure DFT (α_HF = 0). Default False.")
        .def_readwrite("cosx_grid",
                       &vibeqc::RKSOptions::cosx_grid,
                       "GridOptions for the COSX integration grid (only "
                       "used when cosx=True). Defaults to a sparser "
                       "tier than the XC grid; see "
                       "vibeqc.cosx.default_cosx_grid_options.")
        .def_readwrite("cosx_variant",
                       &vibeqc::RKSOptions::cosx_variant,
                       "COSX K-matrix variant selector. AUTO resolves to "
                       "the production FITTED path unless explicitly "
                       "overridden.")
        .def_readwrite("cosx_grid_level",
                       &vibeqc::RKSOptions::cosx_grid_level,
                       "COSX GridX selector: -1 AUTO, 0 legacy cosx_grid, "
                       "1..4 explicit GridX tier.")
        .def_readwrite("dft_plus_u_sites",
                       &vibeqc::RKSOptions::dft_plus_u_sites,
                       "Internal: list of _HubbardSiteCxx (Hartree). "
                       "Populated by the vibeqc.run_rks Python wrapper "
                       "from the user-facing dft_plus_u=[HubbardSite(...)] "
                       "kwarg.")
        .def_readwrite("dft_plus_u_ao_groups",
                       &vibeqc::RKSOptions::dft_plus_u_ao_groups,
                       "Internal: parallel array of AO-index lists, one "
                       "per Hubbard site. Precomputed by the Python "
                       "wrapper from ao_group_indices(basis).")
        .def_readwrite("use_davidson",
                       &vibeqc::RKSOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::RKSOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::RKSOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.")
        .def_property(
            "stability_check",
            [](const vibeqc::RKSOptions& opts) {
                return opts.stability_check;
            },
            [](vibeqc::RKSOptions& opts, bool enabled) {
                opts.stability_check = enabled;
                opts.stability_check_explicit = true;
            },
            "Restricted-stability VERDICT (issue #144): after a "
            "converged RKS SCF, report the lowest orbital-rotation "
            "Hessian eigenvalue at the returned solution (closed-shell "
            "KS singlet + triplet sectors, Bauernschmitt-Ahlrichs 1996; "
            "matrix-free Davidson with the polarised XC kernel at "
            "(D/2, D/2)). Verdict only — no rotation, escape, or "
            "promotion; what to do with a negative verdict is a pending "
            "maintainer decision. Default True; set False for legacy "
            "single-SCF behaviour. Setting it marks the request "
            "explicit, so an unsupported route (+U, meta-GGA, RSH, VV10, "
            "COSX hybrids) fails closed instead of skipping silently.")
        .def_property_readonly(
            "_stability_check_explicit",
            [](const vibeqc::RKSOptions& opts) {
                return opts.stability_check_explicit;
            },
            "Internal flag distinguishing the automatic stability default "
            "from an explicit user request.")
        .def_readwrite("stability_tol",
                       &vibeqc::RKSOptions::stability_tol,
                       "Instability threshold: unstable when the lowest "
                       "Hessian eigenvalue < -stability_tol. Default 1e-4.")
        .def_readwrite("stability_davidson_max_iter",
                       &vibeqc::RKSOptions::stability_davidson_max_iter,
                       "Davidson matvec cap for the stability solve. "
                       "Default 60.");

    py::class_<vibeqc::RKSResult>(m, "RKSResult")
        .def_readonly("restart_basis", &vibeqc::RKSResult::restart_basis)
        .def_readwrite("guess_selection", &vibeqc::RKSResult::guess_selection)
        .def_readonly("energy", &vibeqc::RKSResult::energy)
        .def_readonly("e_electronic", &vibeqc::RKSResult::e_electronic)
        .def_readonly("ecp_operator_applied",
                      &vibeqc::RKSResult::ecp_operator_applied,
                      "True when the SCF Hamiltonian included an ECP "
                      "operator, including a zero-core model potential.")
        .def_readonly("ecp_provenance_verified",
                      &vibeqc::RKSResult::ecp_provenance_verified,
                      "True when exact high-level molecular ECP provenance "
                      "is available; false for a caller-supplied Hcore.")
        .def_readonly("ecp_xml_centers", &vibeqc::RKSResult::ecp_xml_centers,
                      "Exact XML ECP centers used by the SCF; empty for "
                      "all-electron and inline-primitive references.")
        .def_readonly("ecp_xml_library", &vibeqc::RKSResult::ecp_xml_library,
                      "Normalized XML ECP library used by the SCF; empty "
                      "for all-electron and inline-primitive references.")
        .def_readonly("ecp_primitive_blocks",
                      &vibeqc::RKSResult::ecp_primitive_blocks,
                      "Exact inline ECP primitive blocks used by the SCF.")
        .def_readonly("ecp_primitive_centers",
                      &vibeqc::RKSResult::ecp_primitive_centers,
                      "Inline ECP centers paired with ecp_primitive_blocks.")
        .def_readonly("ecp_effective_charges",
                      &vibeqc::RKSResult::ecp_effective_charges,
                      "Per-atom effective charges for the inline ECP route.")
        .def_readonly("ecp_total_ncore", &vibeqc::RKSResult::ecp_total_ncore,
                      "Physical core electrons replaced by the molecular "
                      "ECP used for this SCF reference; zero otherwise.")
        .def_readonly("e_coulomb", &vibeqc::RKSResult::e_coulomb)
        .def_readonly("e_hf_exchange", &vibeqc::RKSResult::e_hf_exchange)
        .def_readonly("e_xc", &vibeqc::RKSResult::e_xc)
        .def_readonly("e_nuclear", &vibeqc::RKSResult::e_nuclear)
        .def_readonly("e_dft_plus_u", &vibeqc::RKSResult::e_dft_plus_u,
                      "Dudarev DFT+U contribution to the total energy "
                      "(Hartree). 0 when RKSOptions.dft_plus_u_sites is "
                      "empty.")
        .def_readonly("n_iter", &vibeqc::RKSResult::n_iter)
        .def_readonly("converged", &vibeqc::RKSResult::converged)
        .def_readonly("xc_batch_workers_used",
                      &vibeqc::RKSResult::xc_batch_workers_used,
                      "Maximum concurrent molecular XC grid batches used.")
        .def_readonly("mo_energies", &vibeqc::RKSResult::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::RKSResult::mo_coeffs)
        .def_readonly("density", &vibeqc::RKSResult::density)
        .def_readonly("fock", &vibeqc::RKSResult::fock)
        .def_readonly("scf_trace", &vibeqc::RKSResult::scf_trace)
        .def_readonly("functional", &vibeqc::RKSResult::functional)
        .def_readonly("stability_checked",
                      &vibeqc::RKSResult::stability_checked,
                      "True when the restricted stability analysis ran on "
                      "the returned solution (verdict only; issue #144).")
        .def_readonly("stability_analysis_converged",
                      &vibeqc::RKSResult::stability_analysis_converged,
                      "True when the stability Davidson solve converged "
                      "(stability_eigenvalue is then trustworthy).")
        .def_readonly("stability_eigenvalue",
                      &vibeqc::RKSResult::stability_eigenvalue,
                      "Lowest orbital-rotation Hessian eigenvalue at the "
                      "returned solution; >= -stability_tol means stable.")
        .def_readonly("internal_instability",
                      &vibeqc::RKSResult::internal_instability,
                      "True when the returned solution is internally or "
                      "externally UNSTABLE (excited SCF solution) — "
                      "reported loudly by the output layer.")
        .def_readonly("stability_wall_s",
                      &vibeqc::RKSResult::stability_wall_s,
                      "Wall seconds spent in the stability phase.")
        .def_readonly("stability_cpu_s",
                      &vibeqc::RKSResult::stability_cpu_s,
                      "CPU seconds spent in the stability phase.")
        .def_readonly("diis_fock_history",
                      &vibeqc::RKSResult::diis_fock_history,
                      "Final DIIS Fock-matrix history for CPCM warm-start "
                      "(BUG 100).")
        .def_readonly("diis_error_history",
                      &vibeqc::RKSResult::diis_error_history,
                      "Final DIIS error-vector history for CPCM warm-start "
                      "(BUG 100).")
        .def("__repr__", [](const vibeqc::RKSResult& r) {
            return std::string{"RKSResult(energy="}
                   + std::to_string(r.energy)
                   + ", functional=" + r.functional
                   + ", converged=" + (r.converged ? "True" : "False") + ")";
        });

    m.def("run_rks", &vibeqc::run_rks,
          py::arg("molecule"), py::arg("basis"),
          py::arg("options") = vibeqc::RKSOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run restricted Kohn-Sham DFT on a closed-shell molecule.");

    m.def("rhf_result_from_rks", &vibeqc::rhf_result_from_rks,
          py::arg("rks_result"),
          "Adapter that returns an RHFResult populated from an "
          "RKSResult — the same orbitals, energy, density, and Fock "
          "matrix, just typed as an RHF reference so it can be passed "
          "to ``run_mp2``. The intended consumer is the double-hybrid "
          "dispatch (``vibeqc.run_b2plyp`` etc.), which runs RKS with "
          "the hybrid SCF piece of a double-hybrid functional and then "
          "feeds the KS orbitals into MP2 with the functional's "
          "``mp2_c_os`` / ``mp2_c_ss`` scaling.");

    m.def("run_rks_scf_with_jk",
          [](const vibeqc::BasisSet& basis,
             int n_electrons,
             const Eigen::MatrixXd& S,
             const Eigen::MatrixXd& Hcore,
             double E_nuc,
             const std::shared_ptr<vibeqc::JKBuilder>& jk,
             const vibeqc::Grid& xc_grid,
             const vibeqc::RKSOptions& options,
             const Eigen::MatrixXd& initial_density,
             const std::vector<Eigen::MatrixXd>& warm_fock_history,
             const std::vector<Eigen::MatrixXd>& warm_error_history,
             const vibeqc::Molecule* molecule,
             const vibeqc::GuessSelection* guess_selection) {
              if (!jk) throw std::invalid_argument(
                  "run_rks_scf_with_jk: jk_builder must not be None");
              return vibeqc::run_rks_scf_with_jk(
                  basis, n_electrons, S, Hcore, E_nuc, *jk, xc_grid,
                  options, initial_density, nullptr,
                  warm_fock_history.empty() ? nullptr : &warm_fock_history,
                  warm_error_history.empty() ? nullptr : &warm_error_history,
                  molecule, guess_selection);
          },
          py::arg("basis"), py::arg("n_electrons"),
          py::arg("S"), py::arg("Hcore"), py::arg("E_nuc"),
          py::arg("jk_builder"), py::arg("xc_grid"),
          py::arg("options") = vibeqc::RKSOptions{},
          py::arg("initial_density") = Eigen::MatrixXd{},
          py::arg("warm_fock_history") = std::vector<Eigen::MatrixXd>{},
          py::arg("warm_error_history") = std::vector<Eigen::MatrixXd>{},
          py::arg("molecule") = py::none(),
          py::arg("guess_selection") = py::none(),
          py::call_guard<py::gil_scoped_release>(),
          "Lower-level closed-shell KS-DFT SCF entry point. Mirrors "
          "run_rhf_scf_with_jk but adds a pre-built ``xc_grid`` for "
          "the numerical XC integration. See run_rhf_scf_with_jk and "
          "RKSOptions.functional for the rest.\n\n"
          "CPCM warm-start (BUG 100): ``warm_fock_history`` / "
          "``warm_error_history`` pre-populate the DIIS subspace "
          "from a previous inner SCF so convergence does not restart "
          "from scratch when only Hcore changed by V_q.");

    // ----- UKS (open-shell DFT) ------------------------------------------
    py::class_<vibeqc::UKSOptions>(m, "UKSOptions")
        .def(py::init<>())
        .def(py::init<const vibeqc::UKSOptions&>(), py::arg("other"),
             "Value-copy an existing UKSOptions object, including internal "
             "implicit-versus-explicit stability-request state.")
        .def_readwrite("functional",  &vibeqc::UKSOptions::functional)
        .def_readwrite("grid",        &vibeqc::UKSOptions::grid)
        .def_property(
            "max_iter",
            [](const vibeqc::UKSOptions& options) { return options.max_iter; },
            [](vibeqc::UKSOptions& options, py::handle value) {
                set_molecular_scf_max_iter(options, value, "UKSOptions");
            })
        .def_readwrite("conv_tol_energy",
                       &vibeqc::UKSOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad", &vibeqc::UKSOptions::conv_tol_grad)
        .def_readwrite("damping",     &vibeqc::UKSOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::UKSOptions::dynamic_damping,
                       "See RHFOptions::dynamic_damping. Default True.")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::UKSOptions::dynamic_damping_min)
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::UKSOptions::dynamic_damping_max)
        .def_readwrite("fock_mixing",
                       &vibeqc::UKSOptions::fock_mixing,
                       "Per-spin Kohn-Sham matrix mixing fraction. See "
                       "RHFOptions.fock_mixing.")
        .def_readwrite("use_diis",    &vibeqc::UKSOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::UKSOptions::diis_start_iter)
        .def_readwrite("diis_subspace_size",
                       &vibeqc::UKSOptions::diis_subspace_size)
        .def_readwrite("scf_accelerator",
                       &vibeqc::UKSOptions::scf_accelerator,
                       "SCF Fock-extrapolation accelerator. See "
                       "RHFOptions.scf_accelerator and "
                       "UHFOptions.scf_accelerator.")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::UKSOptions::ediis_diis_switch_threshold,
                       "See RHFOptions.ediis_diis_switch_threshold. "
                       "Default 1e-1.")
        .def_readwrite("diis_restart_tau",
                       &vibeqc::UKSOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ. See "
                       "RHFOptions.diis_restart_tau. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::UKSOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ. See "
                       "RHFOptions.diis_adaptive_delta. Default 1e-4.")
        .def_readwrite("initial_guess",
                       &vibeqc::UKSOptions::initial_guess)
        .def_readwrite("read_density_alpha",
                       &vibeqc::UKSOptions::read_density_alpha,
                       "Alpha starting density for initial_guess=READ. Set by "
                       "the vibeqc.run_uks wrapper from read_path / read_from.")
        .def_readwrite("read_density_beta",
                       &vibeqc::UKSOptions::read_density_beta,
                       "Beta starting density for initial_guess=READ.")
        .def_readwrite("read_path",
                       &vibeqc::UKSOptions::read_path,
                       "READ source file (.qvf / .molden) for the run_uks "
                       "wrapper; empty restarts from an in-memory read_from.")
        .def_readwrite("atomic_spins",
                       &vibeqc::UKSOptions::atomic_spins,
                       "ATOMSPIN: per-atom spin seed (list of +1/-1/0 in atom "
                       "order) for a broken-symmetry SAD start. Empty = "
                       "spin-symmetric. Requires the SAD guess.")
        .def_readwrite("spinlock_mode",
                       &vibeqc::UKSOptions::spinlock_mode,
                       "SPINLOCK mode: OFF (default), SPIN_SCHEDULE (lock "
                       "n_alpha-n_beta=spinlock_value for spinlock_iterations "
                       "cycles then release), or PATTERN_HOLD (MOM-hold the "
                       "seeded broken-symmetry pattern for spinlock_iterations).")
        .def_readwrite("spinlock_iterations",
                       &vibeqc::UKSOptions::spinlock_iterations,
                       "SPINLOCK: number of initial SCF cycles to hold (both "
                       "modes). 0 = off.")
        .def_readwrite("spinlock_value",
                       &vibeqc::UKSOptions::spinlock_value,
                       "SPINLOCK SPIN_SCHEDULE: the n_alpha-n_beta to lock "
                       "during the first spinlock_iterations cycles.")
        .def_readwrite("linear_dep_threshold",
                       &vibeqc::UKSOptions::linear_dep_threshold)
        .def_readwrite("ecp_centers",
                       &vibeqc::UKSOptions::ecp_centers,
                       "List[ECPCenter]. See RHFOptions.ecp_centers.")
        .def_readwrite("ecp_library",
                       &vibeqc::UKSOptions::ecp_library,
                       "ECP XML library name. See RHFOptions.ecp_library.")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::UKSOptions::ecp_primitive_blocks,
                       "Inline primitive ECP blocks. See RHFOptions.")
        .def_readwrite("ecp_primitive_centers",
                       &vibeqc::UKSOptions::ecp_primitive_centers,
                       "Cartesian centers (bohr) matching ecp_primitive_blocks.")
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::UKSOptions::ecp_effective_charges,
                       "Per-atom effective nuclear charges for inline ECPs.")
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::UKSOptions::ecp_total_ncore,
                       "Total number of core electrons replaced by inline ECPs.")
        .def_readwrite("level_shift",
                       &vibeqc::UKSOptions::level_shift,
                       "Phase C1a-2 — per-spin Saunders-Hillier level "
                       "shift (Hartree). Same formula as UHF. See "
                       "UHFOptions.level_shift. Default 0.0.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::UKSOptions::level_shift_warmup_cycles,
                       "Auto-reducing level-shift warm-up (per-spin). "
                       "See RHFOptions.level_shift_warmup_cycles. "
                       "Default -1 (auto).")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::UKSOptions::level_shift_schedule,
                       "Explicit per-iteration shift curve (per-spin). "
                       "See RHFOptions.level_shift_schedule.")
        .def_readwrite("auto_level_shift_on_oscillation",
                       &vibeqc::UKSOptions::auto_level_shift_on_oscillation,
                       "Auto level-shift on SCF oscillation. "
                       "See RHFOptions.auto_level_shift_on_oscillation. "
                       "Applied per-spin. Default True.")
        .def_readwrite("auto_level_shift_value",
                       &vibeqc::UKSOptions::auto_level_shift_value,
                       "Level shift engaged on detected oscillation. "
                       "See RHFOptions.auto_level_shift_value. "
                       "Default 0.3 Ha.")
        .def_readwrite("oscillation_detect_start_iter",
                       &vibeqc::UKSOptions::oscillation_detect_start_iter,
                       "See RHFOptions.oscillation_detect_start_iter. "
                       "Default 10.")
        .def_readwrite("oscillation_window",
                       &vibeqc::UKSOptions::oscillation_window,
                       "See RHFOptions.oscillation_window. Default 10.")
        .def_readwrite("oscillation_min_sign_flips",
                       &vibeqc::UKSOptions::oscillation_min_sign_flips,
                       "See RHFOptions.oscillation_min_sign_flips. "
                       "Default 5.")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::UKSOptions::quadratic_fallback_iter,
                       "Phase C1c — per-spin second-order SCF "
                       "fallback. See RHFOptions.quadratic_fallback_iter. "
                       "Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::UKSOptions::quadratic_fallback_shift,
                       "C1c orbital-rotation denominator shift. "
                       "See RHFOptions.quadratic_fallback_shift.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::UKSOptions::quadratic_fallback_max_step,
                       "C1c per-step ‖κ‖ trust-region cap. "
                       "See RHFOptions.quadratic_fallback_max_step.")
        .def_readwrite("newton_threshold",
                       &vibeqc::UKSOptions::newton_threshold,
                       "Phase D2c-KS-UHF Newton activation threshold. "
                       "Default 0.0. XC kernel via polarised LDA fxc; "
                       "raises on GGA functionals (Phase 17e pending).")
        .def_readwrite("newton_opts",
                       &vibeqc::UKSOptions::newton_opts,
                       "NewtonOptions. See NewtonOptions.")
        .def_readwrite("soscf_threshold",
                       &vibeqc::UKSOptions::soscf_threshold,
                       "Phase D2d-KS-UHF Neese SOSCF threshold. "
                       "Default 0.0. No XC kernel needed.")
        .def_readwrite("soscf_opts",
                       &vibeqc::UKSOptions::soscf_opts,
                       "SOSCFOptions. See SOSCFOptions.")
        .def_readwrite("trah_threshold",
                       &vibeqc::UKSOptions::trah_threshold,
                       "Phase D2e-KS-UHF TRAH threshold. Default 0.0. "
                       "Same XC kernel as Newton.")
        .def_readwrite("trah_opts",
                       &vibeqc::UKSOptions::trah_opts,
                       "TRAHOptions. See TRAHOptions.")
        .def_readwrite("restart_opts",
                       &vibeqc::UKSOptions::restart_opts,
                       "Deterministic orbital-rotation restart options. "
                       "See SCFRestartOptions.")
        .def_readwrite("density_fit",
                       &vibeqc::UKSOptions::density_fit,
                       "Enable density fitting (RI) for the per-spin "
                       "Fock build. See RHFOptions.density_fit. "
                       "Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::UKSOptions::aux_basis,
                       "Auxiliary basis name for density fitting. "
                       "See RHFOptions.aux_basis.")
        .def_readwrite("scf_mode",
                       &vibeqc::UKSOptions::scf_mode,
                       "SCF Fock-build mode (SCFMode). See "
                       "RHFOptions.scf_mode.")
        .def_readwrite("scf_mode_auto_threshold",
                       &vibeqc::UKSOptions::scf_mode_auto_threshold,
                       "AUTO cutoff (n_bf). See "
                       "RHFOptions.scf_mode_auto_threshold.")
        .def_readwrite("schwarz_threshold",
                       &vibeqc::UKSOptions::schwarz_threshold,
                       "Per-quartet Schwarz skip bound for DIRECT mode. "
                       "See RHFOptions.schwarz_threshold.")
        .def_readwrite("incremental_fock",
                       &vibeqc::UKSOptions::incremental_fock,
                       "Enable incremental ΔP Fock build for DIRECT. "
                       "See RHFOptions.incremental_fock.")
        .def_readwrite("incremental_fock_reset_freq",
                       &vibeqc::UKSOptions::incremental_fock_reset_freq,
                       "Full-rebuild frequency for incremental_fock. "
                       "See RHFOptions.incremental_fock_reset_freq.")
        .def_readwrite("schwarz_threshold_loose",
                       &vibeqc::UKSOptions::schwarz_threshold_loose,
                       "Two-phase Schwarz loose threshold. "
                       "See RHFOptions.schwarz_threshold_loose.")
        .def_readwrite("schwarz_threshold_tighten_at",
                       &vibeqc::UKSOptions::schwarz_threshold_tighten_at,
                       "Grad-norm cutoff for the loose→tight transition. "
                       "See RHFOptions.schwarz_threshold_tighten_at.")
        .def_readwrite("cosx",
                       &vibeqc::UKSOptions::cosx,
                       "Use chain-of-spheres exchange for K. See "
                       "RKSOptions.cosx. Default False.")
        .def_readwrite("cosx_grid",
                       &vibeqc::UKSOptions::cosx_grid,
                       "GridOptions for the COSX grid. See "
                       "RKSOptions.cosx_grid.")
        .def_readwrite("cosx_variant",
                       &vibeqc::UKSOptions::cosx_variant,
                       "COSX K-matrix variant selector. See "
                       "RKSOptions.cosx_variant.")
        .def_readwrite("cosx_grid_level",
                       &vibeqc::UKSOptions::cosx_grid_level,
                       "COSX GridX selector. See RKSOptions.cosx_grid_level.")
        .def_readwrite("dft_plus_u_sites",
                       &vibeqc::UKSOptions::dft_plus_u_sites,
                       "Internal: list of _HubbardSiteCxx. Populated by "
                       "the vibeqc.run_uks Python wrapper.")
        .def_readwrite("dft_plus_u_ao_groups",
                       &vibeqc::UKSOptions::dft_plus_u_ao_groups,
                       "Internal: parallel AO-index lists per site.")
        .def_readwrite("use_davidson",
                       &vibeqc::UKSOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::UKSOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::UKSOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.")
        .def_property(
            "stability_check",
            [](const vibeqc::UKSOptions& opts) {
                return opts.stability_check;
            },
            [](vibeqc::UKSOptions& opts, bool enabled) {
                opts.stability_check = enabled;
                opts.stability_check_explicit = true;
            },
            "Internal SCF stability analysis (UKS, default ON where the "
            "complete polarised response is supported). The implicit "
            "automatic check is skipped for meta-GGA, range-separated, "
            "VV10, DFT+U, active hybrid RIJCOSX, and coupled-solvent jobs; explicitly "
            "requesting it fails closed until the missing response terms "
            "are available. "
            "See UHFOptions.stability_check.")
        .def_property_readonly(
            "_stability_check_explicit",
            [](const vibeqc::UKSOptions& opts) {
                return opts.stability_check_explicit;
            },
            "Internal flag distinguishing the automatic stability default "
            "from an explicit user request.")
        .def_readwrite("stability_tol",
                       &vibeqc::UKSOptions::stability_tol,
                       "Instability threshold on the lowest Hessian "
                       "eigenvalue. Default 1e-4.")
        .def_readwrite("stability_max_retries",
                       &vibeqc::UKSOptions::stability_max_retries,
                       "Maximum rotate-and-reconverge escapes. Default 3.")
        .def_readwrite("stability_davidson_max_iter",
                       &vibeqc::UKSOptions::stability_davidson_max_iter,
                       "Davidson matvec cap for the stability solve. "
                       "Default 60.")
        .def_readwrite("multi_guess_seeds",
                       &vibeqc::UKSOptions::multi_guess_seeds,
                       "Additional PRNG seeds for multi-start convergence "
                       "(BUG 88). See UHFOptions.multi_guess_seeds.");

    py::class_<vibeqc::UKSResult>(m, "UKSResult")
        .def_readonly("restart_basis", &vibeqc::UKSResult::restart_basis)
        .def_readwrite("guess_selection", &vibeqc::UKSResult::guess_selection)
        .def_readonly("energy",       &vibeqc::UKSResult::energy)
        .def_readonly("e_electronic", &vibeqc::UKSResult::e_electronic)
        .def_readonly("ecp_operator_applied",
                       &vibeqc::UKSResult::ecp_operator_applied,
                       "True when the SCF Hamiltonian included an ECP "
                       "operator, including a zero-core model potential.")
        .def_readonly("ecp_provenance_verified",
                       &vibeqc::UKSResult::ecp_provenance_verified,
                       "True when exact high-level molecular ECP provenance "
                       "is available; false for a caller-supplied Hcore.")
        .def_readonly("ecp_xml_centers", &vibeqc::UKSResult::ecp_xml_centers,
                      "Exact XML ECP centers used by the SCF; empty for "
                      "all-electron and inline-primitive references.")
        .def_readonly("ecp_xml_library", &vibeqc::UKSResult::ecp_xml_library,
                      "Normalized XML ECP library used by the SCF; empty "
                      "for all-electron and inline-primitive references.")
        .def_readonly("ecp_primitive_blocks",
                       &vibeqc::UKSResult::ecp_primitive_blocks,
                       "Exact inline ECP primitive blocks used by the SCF.")
        .def_readonly("ecp_primitive_centers",
                       &vibeqc::UKSResult::ecp_primitive_centers,
                       "Inline ECP centers paired with ecp_primitive_blocks.")
        .def_readonly("ecp_effective_charges",
                       &vibeqc::UKSResult::ecp_effective_charges,
                       "Per-atom effective charges for the inline ECP route.")
        .def_readonly("ecp_total_ncore", &vibeqc::UKSResult::ecp_total_ncore,
                      "Physical core electrons replaced by the molecular "
                      "ECP used for this SCF reference; zero otherwise.")
        .def_readonly("e_coulomb",    &vibeqc::UKSResult::e_coulomb)
        .def_readonly("e_hf_exchange",&vibeqc::UKSResult::e_hf_exchange)
        .def_readonly("e_dft_plus_u", &vibeqc::UKSResult::e_dft_plus_u,
                      "Dudarev DFT+U contribution (Hartree). Sum of "
                      "per-spin contributions; 0 when +U is not active.")
        .def_readonly("e_xc",         &vibeqc::UKSResult::e_xc)
        .def_readonly("e_nuclear",    &vibeqc::UKSResult::e_nuclear)
        .def_readonly("n_iter",       &vibeqc::UKSResult::n_iter)
        .def_readonly("converged",    &vibeqc::UKSResult::converged)
        .def_readonly("xc_batch_workers_used",
                       &vibeqc::UKSResult::xc_batch_workers_used,
                       "Maximum concurrent molecular XC grid batches used.")
        .def_readonly("s_squared",    &vibeqc::UKSResult::s_squared)
        .def_readonly("s_squared_ideal",
                       &vibeqc::UKSResult::s_squared_ideal)
        .def_property_readonly(
            "s_squared_deviation",
            [](const vibeqc::UKSResult& r) {
                return r.s_squared - r.s_squared_ideal;
            },
            "Spin contamination: <S^2> - S(S+1).")
        .def_readonly("mo_energies_alpha",
                       &vibeqc::UKSResult::mo_energies_alpha)
        .def_readonly("mo_coeffs_alpha",
                       &vibeqc::UKSResult::mo_coeffs_alpha)
        .def_readonly("density_alpha",&vibeqc::UKSResult::density_alpha)
        .def_readonly("fock_alpha",   &vibeqc::UKSResult::fock_alpha)
        .def_readonly("mo_energies_beta",
                       &vibeqc::UKSResult::mo_energies_beta)
        .def_readonly("mo_coeffs_beta",
                       &vibeqc::UKSResult::mo_coeffs_beta)
        .def_readonly("density_beta", &vibeqc::UKSResult::density_beta)
        .def_readonly("fock_beta",    &vibeqc::UKSResult::fock_beta)
        .def_readonly("scf_trace",    &vibeqc::UKSResult::scf_trace)
        .def_readonly("functional",   &vibeqc::UKSResult::functional)
        .def_readonly("stability_checked",
                       &vibeqc::UKSResult::stability_checked,
                       "True when the internal stability analysis ran on "
                       "the returned solution.")
        .def_readonly("stability_analysis_converged",
                       &vibeqc::UKSResult::stability_analysis_converged,
                       "True when the stability Davidson solve converged.")
        .def_readonly("stability_eigenvalue",
                       &vibeqc::UKSResult::stability_eigenvalue,
                       "Lowest internal orbital-rotation Hessian "
                       "eigenvalue at the returned solution.")
        .def_readonly("n_stability_restarts",
                       &vibeqc::UKSResult::n_stability_restarts,
                       "Rotate-and-reconverge escapes taken to reach the "
                       "returned UKS solution.")
        .def_readonly("internal_instability",
                       &vibeqc::UKSResult::internal_instability,
                       "True when the returned solution is internally "
                       "UNSTABLE (excited SCF solution).")
        .def_readonly("stability_wall_s",
                       &vibeqc::UKSResult::stability_wall_s,
                       "Wall seconds spent in the post-convergence "
                       "internal-stability phase; 0.0 when it did not run.")
        .def_readonly("stability_cpu_s",
                       &vibeqc::UKSResult::stability_cpu_s,
                       "CPU seconds (summed over OpenMP threads) spent in "
                       "the post-convergence internal-stability phase.")
        .def_readonly("n_iter_before_stability",
                       &vibeqc::UKSResult::n_iter_before_stability,
                       "Iterations in the initial SCF loop whose wall is "
                       "outside the post-convergence stability phase.")
        .def("__repr__", [](const vibeqc::UKSResult& r) {
            return std::string{"UKSResult(energy="}
                   + std::to_string(r.energy)
                   + ", functional=" + r.functional
                   + ", converged=" + (r.converged ? "True" : "False")
                   + ", <S^2>=" + std::to_string(r.s_squared) + ")";
        });

    m.def("run_uks", &vibeqc::run_uks,
          py::arg("molecule"), py::arg("basis"),
          py::arg("options") = vibeqc::UKSOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run unrestricted Kohn-Sham DFT on an open-shell molecule. "
          "Alpha/beta occupations follow from multiplicity.");

    m.def("run_uhf", &vibeqc::run_uhf,
          py::arg("molecule"), py::arg("basis"),
          py::arg("options") = vibeqc::UHFOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run unrestricted Hartree-Fock SCF on an open- or closed-shell "
          "molecule. Alpha/beta occupations follow from molecule.multiplicity. "
          "Returns a UHFResult.");

    m.def("run_uhf_scf_with_jk",
          [](const vibeqc::BasisSet& basis,
             int n_alpha, int n_beta,
             const Eigen::MatrixXd& S,
             const Eigen::MatrixXd& Hcore,
             double E_nuc,
             const std::shared_ptr<vibeqc::JKBuilder>& jk,
             const vibeqc::UHFOptions& options,
             const Eigen::MatrixXd& init_alpha,
             const Eigen::MatrixXd& init_beta,
             const vibeqc::Molecule* molecule,
             const vibeqc::GuessSelection* guess_selection) {
              if (!jk) throw std::invalid_argument(
                  "run_uhf_scf_with_jk: jk_builder must not be None");
              return vibeqc::run_uhf_scf_with_jk(
                  basis, n_alpha, n_beta, S, Hcore, E_nuc, *jk, options,
                  init_alpha, init_beta, molecule, guess_selection);
          },
          py::arg("basis"), py::arg("n_alpha"), py::arg("n_beta"),
          py::arg("S"), py::arg("Hcore"), py::arg("E_nuc"),
          py::arg("jk_builder"),
          py::arg("options") = vibeqc::UHFOptions{},
          py::arg("init_alpha") = Eigen::MatrixXd{},
          py::arg("init_beta")  = Eigen::MatrixXd{},
          py::arg("molecule") = py::none(),
          py::arg("guess_selection") = py::none(),
          py::call_guard<py::gil_scoped_release>(),
          "Lower-level UHF SCF entry point. Mirrors run_rhf_scf_with_jk "
          "but with per-spin electron counts. ``init_alpha`` / "
          "``init_beta`` are optional density-matrix seeds (both supplied "
          "or both empty); empty falls back to a Hcore-diag guess.");

    m.def("run_uks_scf_with_jk",
          [](const vibeqc::BasisSet& basis,
             int n_alpha, int n_beta,
             const Eigen::MatrixXd& S,
             const Eigen::MatrixXd& Hcore,
             double E_nuc,
             const std::shared_ptr<vibeqc::JKBuilder>& jk,
             const vibeqc::Grid& xc_grid,
             const vibeqc::UKSOptions& options,
             const Eigen::MatrixXd& init_alpha,
             const Eigen::MatrixXd& init_beta,
             const vibeqc::Molecule* molecule,
             const vibeqc::GuessSelection* guess_selection) {
              if (!jk) throw std::invalid_argument(
                  "run_uks_scf_with_jk: jk_builder must not be None");
              return vibeqc::run_uks_scf_with_jk(
                  basis, n_alpha, n_beta, S, Hcore, E_nuc, *jk, xc_grid,
                  options, init_alpha, init_beta, nullptr, molecule, guess_selection);
          },
          py::arg("basis"), py::arg("n_alpha"), py::arg("n_beta"),
          py::arg("S"), py::arg("Hcore"), py::arg("E_nuc"),
          py::arg("jk_builder"), py::arg("xc_grid"),
          py::arg("options") = vibeqc::UKSOptions{},
          py::arg("init_alpha") = Eigen::MatrixXd{},
          py::arg("init_beta")  = Eigen::MatrixXd{},
          py::arg("molecule") = py::none(),
          py::arg("guess_selection") = py::none(),
          py::call_guard<py::gil_scoped_release>(),
          "Lower-level UKS SCF entry point. Mirrors run_rks_scf_with_jk "
          "but with per-spin electron counts and per-spin initial "
          "densities. The ``xc_grid`` is the numerical-integration grid "
          "for the XC piece — pass a periodic grid for periodic-Γ "
          "open-shell DFT.");

    // ----- MP2 (second-order Moller-Plesset) -----------------------------
    py::class_<vibeqc::MP2Options>(m, "MP2Options")
        .def(py::init<>())
        .def_property(
            "n_frozen_core",
            [](const vibeqc::MP2Options& options) -> py::object {
                if (options.n_frozen_core == -1) return py::none();
                return py::int_(options.n_frozen_core);
            },
            [](vibeqc::MP2Options& options, const py::object& requested) {
                if (requested.is_none()) {
                    options.n_frozen_core = -1;
                    return;
                }
                const int count = requested.cast<int>();
                if (count < 0) {
                    throw py::value_error(
                        "n_frozen_core must be None or a non-negative integer");
                }
                options.n_frozen_core = count;
            },
            "Frozen occupied spatial orbitals. None (default) resolves the "
            "published ORCA 6.1 Table 2.69 count; 0 requests all-electron "
            "correlation; positive values are explicit counts. The published "
            "policy is count-only and does not reorder orbitals by atomic "
            "character.")
        .def_readwrite("density_fit",
                       &vibeqc::MP2Options::density_fit,
                       "Enable density fitting for the AO→MO ERI "
                       "transformation. Replaces the four-index ERI "
                       "and O(n^5) ao_to_mo_ovov transform with the "
                       "DF factorisation (ia|jb) ≈ Σ_P B^P_ia B^P_jb "
                       "(Vahtras-Almlöf-Feyereisen 1993). Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::MP2Options::aux_basis,
                       "Auxiliary basis name for density fitting "
                       "(libint-recognised, e.g. \"def2-tzvp-rifit\", "
                       "\"cc-pvtz-ri\"). Use "
                       "vibeqc.default_aux_basis_for(orbital_basis_name, "
                       "kind=\"ri\") for autodetection. Empty + "
                       "density_fit=True raises.")
        .def_readwrite("c_os", &vibeqc::MP2Options::c_os,
                       "Opposite-spin (singlet pair) scaling coefficient "
                       "in the Grimme JCP 118, 9095 (2003) decomposition. "
                       "Default 1.0 (canonical MP2). 6/5 for SCS-MP2, "
                       "1.3 for SOS-MP2, 0.27 for the B2PLYP MP2 "
                       "correction.")
        .def_readwrite("c_ss", &vibeqc::MP2Options::c_ss,
                       "Same-spin (triplet pair) scaling coefficient. "
                       "Default 1.0 (canonical MP2). 1/3 for SCS-MP2, "
                       "0.0 for SOS-MP2, 0.27 for the B2PLYP MP2 "
                       "correction.")
        .def_readwrite("use_float_intermediates",
                       &vibeqc::MP2Options::use_float_intermediates,
                       "When density_fit=True, store the half-transformed "
                       "B-mo intermediate as single-precision float "
                       "instead of double.  Halves the memory for the "
                       "(n_aux × n_occ·n_vir) tensor.  Default False.")
        .def_readwrite("report_ri_residual",
                       &vibeqc::MP2Options::report_ri_residual,
                       "Diagnostic: when density_fit=True, also build "
                       "the canonical four-index (ia|jb) tensor and "
                       "report (e_os_RI − e_os_canonical) / "
                       "(e_ss_RI − e_ss_canonical) on the MP2Result. "
                       "Doubles the runtime cost — opt-in, intended "
                       "for verifying aux-basis adequacy. Default False.")
        .def_readwrite("requested_memory_bytes",
                       &vibeqc::MP2Options::requested_memory_bytes,
                       "Maximum MP2 workspace in bytes. Zero preserves the "
                       "low-level in-core auto default; run_job derives a "
                       "nonzero cap from the process allocation. The "
                       "direct and disk strategies honor this bound.")
        .def_readwrite("memory_mode",
                       &vibeqc::MP2Options::memory_mode,
                       "Integral-storage strategy: 'auto', 'incore', "
                       "'direct', or 'disk'. Default 'auto'.")
        .def_readwrite("scratch_directory",
                       &vibeqc::MP2Options::scratch_directory,
                       "Directory for bounded semi-direct scratch files. "
                       "Empty uses the platform temporary directory.");

    py::class_<vibeqc::MP2Result>(m, "MP2Result")
        .def_readonly("e_hf",          &vibeqc::MP2Result::e_hf)
        .def_readonly("e_correlation", &vibeqc::MP2Result::e_correlation,
                      "Spin-component-scaled correlation energy: "
                      "c_os · e_os + c_ss · e_ss. At c_os = c_ss = 1 "
                      "this is canonical RMP2.")
        .def_readonly("e_total",       &vibeqc::MP2Result::e_total)
        .def_readonly("n_frozen_core", &vibeqc::MP2Result::n_frozen_core,
                      "Resolved number of frozen occupied spatial orbitals.")
        .def_readonly("e_os",          &vibeqc::MP2Result::e_os,
                      "Unscaled opposite-spin (αβ singlet pair) energy "
                      "Σ (ia|jb)²/Δ (Grimme JCP 118, 9095 (2003)). "
                      "Multiply by ``MP2Options.c_os`` for the scaled "
                      "contribution to e_correlation.")
        .def_readonly("e_ss",          &vibeqc::MP2Result::e_ss,
                      "Unscaled same-spin (αα+ββ triplet pair) energy "
                      "Σ [(ia|jb)² − (ia|jb)(ib|ja)]/Δ. Multiply by "
                      "``MP2Options.c_ss`` for the scaled contribution "
                      "to e_correlation.")
        .def_readonly("e_os_ri_residual",
                      &vibeqc::MP2Result::e_os_ri_residual,
                      "RI fit residual e_os_RI − e_os_canonical, "
                      "populated only when ``MP2Options.density_fit && "
                      "report_ri_residual``. 0.0 otherwise; check "
                      "``ri_residual_reported`` to distinguish.")
        .def_readonly("e_ss_ri_residual",
                      &vibeqc::MP2Result::e_ss_ri_residual,
                      "RI fit residual e_ss_RI − e_ss_canonical, "
                      "populated only when ``MP2Options.density_fit && "
                      "report_ri_residual``.")
        .def_readonly("ri_residual_reported",
                      &vibeqc::MP2Result::ri_residual_reported,
                      "True iff ``report_ri_residual`` was set on the "
                      "MP2Options for this run AND density_fit was on. "
                      "Use this to distinguish 'residual not requested' "
                      "from 'residual requested and happens to be 0'.")
        .def_readonly("memory_mode_used", &vibeqc::MP2Result::memory_mode_used,
                      "Resolved integral-storage strategy used by the run.")
        .def_readonly("workspace_bytes", &vibeqc::MP2Result::workspace_bytes,
                      "Modeled peak MP2 workspace allocated by the kernel.")
        .def_readonly("disk_bytes", &vibeqc::MP2Result::disk_bytes,
                      "Peak scratch-file extent used by semi-direct MP2.")
        .def("__repr__", [](const vibeqc::MP2Result& r) {
            return std::string{"MP2Result(e_total="}
                   + std::to_string(r.e_total)
                   + ", e_corr=" + std::to_string(r.e_correlation) + ")";
        });

    m.def("run_mp2", &vibeqc::run_mp2,
          py::arg("molecule"), py::arg("basis"), py::arg("rhf_result"),
          py::arg("options") = vibeqc::MP2Options{},
          py::call_guard<py::gil_scoped_release>(),
          "Run closed-shell RMP2 correlation on a converged RHF reference. "
          "Pass options.density_fit=True with options.aux_basis set to "
          "use density fitting (RI-MP2).");

    // ----- Direct determinant CAS-CI (string sigma + Davidson) -----------
    py::class_<vibeqc::CASCIDirectOptions>(m, "CASCIDirectOptions")
        .def(py::init<>())
        .def_readwrite("nroots", &vibeqc::CASCIDirectOptions::nroots,
                       "Number of lowest eigenpairs to converge.")
        .def_readwrite("tol", &vibeqc::CASCIDirectOptions::tol,
                       "Davidson residual-norm convergence per root.")
        .def_readwrite("max_iter", &vibeqc::CASCIDirectOptions::max_iter)
        .def_readwrite("max_subspace",
                       &vibeqc::CASCIDirectOptions::max_subspace,
                       "Subspace size at which Davidson collapses to the "
                       "current Ritz vectors.")
        .def_readwrite("chunk", &vibeqc::CASCIDirectOptions::chunk,
                       "Determinant-index chunk width for the sigma build.");

    py::class_<vibeqc::CASCIDirectResult>(m, "CASCIDirectResult")
        .def_readonly("eigenvalues",
                      &vibeqc::CASCIDirectResult::eigenvalues,
                      "Active-space CI eigenvalues (no core constant).")
        .def_readonly("ci", &vibeqc::CASCIDirectResult::ci,
                      "(ndet, nroots) CI vectors, alpha-major determinant "
                      "order matching "
                      "solvers._determinant.generate_determinants.")
        .def_readonly("converged", &vibeqc::CASCIDirectResult::converged)
        .def_readonly("n_iter", &vibeqc::CASCIDirectResult::n_iter)
        .def_readonly("n_det", &vibeqc::CASCIDirectResult::n_det);

    m.def(
        "casci_direct_solve",
        [](const Eigen::MatrixXd& h1,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               eri_chem,
           int n_act, int n_alpha, int n_beta,
           const vibeqc::CASCIDirectOptions& opts,
           const Eigen::MatrixXd& guess) {
            std::vector<double> eri(eri_chem.data(),
                                    eri_chem.data() + eri_chem.size());
            py::gil_scoped_release rel;
            return vibeqc::casci_direct_solve(h1, eri, n_act, n_alpha,
                                              n_beta, opts, guess);
        },
        py::arg("h1"), py::arg("eri_chem"), py::arg("n_act"),
        py::arg("n_alpha"), py::arg("n_beta"),
        py::arg("options") = vibeqc::CASCIDirectOptions{},
        py::arg("guess") = Eigen::MatrixXd(),
        "Direct determinant CAS-CI: lowest eigenpairs of the active-space "
        "CI Hamiltonian via string-based sigma builds (Knowles-Handy 1984) "
        "+ block Davidson (Davidson 1975), without materializing H. h1 is "
        "the (dressed) active one-electron matrix; eri_chem the active "
        "(pq|rs) integrals in CHEMIST notation, any shape with n_act^4 "
        "elements. guess (ndet x k) warm-starts the Davidson subspace, "
        "e.g. with the previous CASSCF macro-iteration's CI vector(s).");

    m.def(
        "casci_direct_rdm12",
        [](const Eigen::VectorXd& ci, int n_act, int n_alpha, int n_beta) {
            Eigen::MatrixXd rdm1;
            std::vector<double> rdm2;
            {
                py::gil_scoped_release rel;
                vibeqc::casci_direct_rdm12(ci, n_act, n_alpha, n_beta, rdm1,
                                           rdm2);
            }
            py::array_t<double> r2(
                std::vector<py::ssize_t>{n_act, n_act, n_act, n_act});
            std::memcpy(r2.mutable_data(), rdm2.data(),
                        rdm2.size() * sizeof(double));
            return py::make_tuple(rdm1, r2);
        },
        py::arg("ci"), py::arg("n_act"), py::arg("n_alpha"),
        py::arg("n_beta"),
        "Spin-summed (rdm1, rdm2) of a CI vector in the direct engine's "
        "determinant order; conventions match solvers._rdm.make_rdm12 "
        "(rdm1[p,q] = <a+_p a_q>, rdm2[p,q,r,s] = <a+_p a+_r a_s a_q>).");

    m.def(
        "casci_direct_sigma",
        [](const Eigen::VectorXd& ci, const Eigen::MatrixXd& h1,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               eri_chem,
           int n_act, int n_alpha, int n_beta) {
            std::vector<double> eri(eri_chem.data(),
                                    eri_chem.data() + eri_chem.size());
            py::gil_scoped_release rel;
            return vibeqc::casci_direct_sigma(ci, h1, eri, n_act, n_alpha,
                                              n_beta);
        },
        py::arg("ci"), py::arg("h1"), py::arg("eri_chem"), py::arg("n_act"),
        py::arg("n_alpha"), py::arg("n_beta"),
        "sigma = H*c in the determinant basis (testing hook for the direct "
        "CAS-CI engine).");

    // ----- NEVPT2 sparse determinant operations (performance-critical) -----
    m.def(
        "apply_1body_cpp",
        [](const std::unordered_map<uint64_t, double>& state,
           const Eigen::MatrixXd& h1,
           int norb,
           const std::optional<std::vector<int>>& idx) {
            py::gil_scoped_release rel;
            return vibeqc::apply_1body_cpp(
                state, h1, norb, idx ? &*idx : nullptr);
        },
        py::arg("state"), py::arg("h1"), py::arg("norb"),
        py::arg("idx") = py::none(),
        "Apply sum_{pq} h1[p,q] sum_s a^dagger_{ps} a_{qs} to a sparse "
        "determinant state dict {bitmask -> coeff}.");

    m.def(
        "apply_2body_cpp",
        [](const std::unordered_map<uint64_t, double>& state,
           py::array_t<double, py::array::c_style | py::array::forcecast> eri,
           int norb,
           const std::optional<std::vector<int>>& idx) {
            std::vector<double> eri_vec(eri.data(), eri.data() + eri.size());
            py::gil_scoped_release rel;
            return vibeqc::apply_2body_cpp(
                state, eri_vec, norb, idx ? &*idx : nullptr);
        },
        py::arg("state"), py::arg("eri"), py::arg("norb"),
        py::arg("idx") = py::none(),
        "Apply 1/2 sum_{pqrs} eri(p,q,r,s) sum_{st} a^dagger_{ps} a^dagger_{rt} "
        "a_{st} a_{qs} to a sparse determinant state (chemist's convention, flat).");

    m.def(
        "dot_cpp",
        &vibeqc::dot_cpp,
        py::arg("a"), py::arg("b"),
        "Dot product of two sparse determinant states.");

    m.def(
        "add_cpp",
        &vibeqc::add_cpp,
        py::arg("a"), py::arg("b"),
        "Add two sparse determinant states (a + b).");

    m.def(
        "state_rdm12_cpp",
        [](const std::unordered_map<uint64_t, double>& state,
           int norb) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXd rdm1(norb, norb);
            Eigen::MatrixXd rdm2(norb * norb, norb * norb);
            {
                py::gil_scoped_release rel;
                vibeqc::state_rdm12_cpp(state, norb, rdm1, rdm2);
            }
            return {rdm1, rdm2};
        },
        py::arg("state"), py::arg("norb"),
        "Compute 1-RDM and flat 2-RDM from a sparse determinant state.");

    m.def(
        "state_rdm3_cpp",
        [](const std::unordered_map<uint64_t, double>& state,
           int norb, int n_act) -> std::vector<double> {
            std::vector<double> rdm3;
            {
                py::gil_scoped_release rel;
                vibeqc::state_rdm3_cpp(state, norb, n_act, rdm3);
            }
            return rdm3;
        },
        py::arg("state"), py::arg("norb"), py::arg("n_act"),
        "Compute flat active-space 3-RDM from a sparse determinant state.");

    // ----- 4-index MO transform (CASPT2/NEVPT2 dE^2/dk kernel) --------
    m.def(
        "transform_4index_mo",
        [](const Eigen::VectorXd& g_phys_flat,
           const Eigen::MatrixXd& U,
           int norb) -> Eigen::VectorXd {
            Eigen::VectorXd result;
            {
                py::gil_scoped_release rel;
                result = vibeqc::transform_4index_mo(g_phys_flat, U, norb);
            }
            return result;
        },
        py::arg("g_phys_flat"), py::arg("U"), py::arg("norb"),
        "Transform 4-index MO integrals g_new[p,r,q,s] = U^T g U. "
        "g_phys is in physicist's (pr|qs) flat row-major. Returns flat (norb^4) vector.");

    // ----- Selected CI over arbitrary SpinDet lists (roadmap 25i) --------
    py::class_<vibeqc::SelectedCIOptionsCpp>(m, "SelectedCIOptionsCpp")
        .def(py::init<>())
        .def_readwrite("nroots", &vibeqc::SelectedCIOptionsCpp::nroots)
        .def_readwrite("max_cycles",
                       &vibeqc::SelectedCIOptionsCpp::max_cycles)
        .def_readwrite("target_size",
                       &vibeqc::SelectedCIOptionsCpp::target_size)
        .def_readwrite("max_new_per_cycle",
                       &vibeqc::SelectedCIOptionsCpp::max_new_per_cycle)
        .def_readwrite("conv_tol_energy",
                       &vibeqc::SelectedCIOptionsCpp::conv_tol_energy)
        .def_readwrite("pt2_threshold",
                       &vibeqc::SelectedCIOptionsCpp::pt2_threshold)
        .def_readwrite("significant_coeff",
                       &vibeqc::SelectedCIOptionsCpp::significant_coeff)
        .def_readwrite("select_eps",
                       &vibeqc::SelectedCIOptionsCpp::select_eps,
                       "Heat-bath prefilter on |H_DI c_I| during selection "
                       "(0 = exact CIPSI scoring).")
        .def_readwrite("use_heat_bath_walk",
                       &vibeqc::SelectedCIOptionsCpp::use_heat_bath_walk,
                       "Walk presorted |H|-ordered double-excitation lists "
                       "when select_eps > 0 (identical candidate set; "
                       "false = brute-force prefilter, the equivalence-"
                       "test hook).")
        .def_readwrite("spin_complete",
                       &vibeqc::SelectedCIOptionsCpp::spin_complete)
        .def_readwrite("davidson_max_iter",
                       &vibeqc::SelectedCIOptionsCpp::davidson_max_iter)
        .def_readwrite("davidson_tol",
                       &vibeqc::SelectedCIOptionsCpp::davidson_tol);

    py::class_<vibeqc::SelectedCIResultCpp>(m, "SelectedCIResultCpp")
        .def_readonly("eigenvalues",
                      &vibeqc::SelectedCIResultCpp::eigenvalues,
                      "Lowest-nroots active-space eigenvalues in the "
                      "selected space (no core constant).")
        .def_readonly("ci", &vibeqc::SelectedCIResultCpp::ci)
        .def_readonly("n_cycles", &vibeqc::SelectedCIResultCpp::n_cycles)
        .def_readonly("converged", &vibeqc::SelectedCIResultCpp::converged)
        .def_property_readonly(
            "dets_a",
            [](const vibeqc::SelectedCIResultCpp& r) {
                py::array_t<std::uint64_t> out(
                    py::ssize_t(r.dets_a.size()));
                std::memcpy(out.mutable_data(), r.dets_a.data(),
                            r.dets_a.size() * sizeof(std::uint64_t));
                return out;
            },
            "Alpha occupation bitmasks of the selected determinants.")
        .def_property_readonly(
            "dets_b",
            [](const vibeqc::SelectedCIResultCpp& r) {
                py::array_t<std::uint64_t> out(
                    py::ssize_t(r.dets_b.size()));
                std::memcpy(out.mutable_data(), r.dets_b.data(),
                            r.dets_b.size() * sizeof(std::uint64_t));
                return out;
            },
            "Beta occupation bitmasks (index-aligned with dets_a).");

    m.def(
        "selected_ci_solve",
        [](const Eigen::MatrixXd& h1,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               eri_chem,
           int n_act, int n_alpha, int n_beta,
           const vibeqc::SelectedCIOptionsCpp& opts,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               guess_a,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               guess_b) {
            std::vector<double> eri(eri_chem.data(),
                                    eri_chem.data() + eri_chem.size());
            std::vector<std::uint64_t> ga(guess_a.data(),
                                          guess_a.data() + guess_a.size());
            std::vector<std::uint64_t> gb(guess_b.data(),
                                          guess_b.data() + guess_b.size());
            py::gil_scoped_release rel;
            return vibeqc::selected_ci_solve(h1, eri, n_act, n_alpha,
                                             n_beta, opts, ga, gb);
        },
        py::arg("h1"), py::arg("eri_chem"), py::arg("n_act"),
        py::arg("n_alpha"), py::arg("n_beta"),
        py::arg("options") = vibeqc::SelectedCIOptionsCpp{},
        py::arg("guess_a") = py::array_t<std::uint64_t>(0),
        py::arg("guess_b") = py::array_t<std::uint64_t>(0),
        "CIPSI selected CI in an active space over SpinDet bitmasks: "
        "grow-and-diagonalize with coherent multi-root selection scores, "
        "sparse Slater-Condon H + block Davidson. guess_a/guess_b "
        "(index-aligned uint64 masks) seed the variational space, e.g. "
        "the previous CASSCF macro-iteration's selected set. Conventions "
        "match solvers._selected_ci.selected_casci (the Python oracle).");

    m.def(
        "selected_ci_rdm12",
        [](py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               dets_a,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               dets_b,
           const Eigen::VectorXd& ci, int n_act) {
            std::vector<std::uint64_t> da(dets_a.data(),
                                          dets_a.data() + dets_a.size());
            std::vector<std::uint64_t> db(dets_b.data(),
                                          dets_b.data() + dets_b.size());
            Eigen::MatrixXd rdm1;
            std::vector<double> rdm2;
            {
                py::gil_scoped_release rel;
                vibeqc::selected_ci_rdm12(da, db, ci, n_act, rdm1, rdm2);
            }
            py::array_t<double> r2(
                std::vector<py::ssize_t>{n_act, n_act, n_act, n_act});
            std::memcpy(r2.mutable_data(), rdm2.data(),
                        rdm2.size() * sizeof(double));
            return py::make_tuple(rdm1, r2);
        },
        py::arg("dets_a"), py::arg("dets_b"), py::arg("ci"),
        py::arg("n_act"),
        "Spin-summed (rdm1, rdm2) of a CI vector over an ARBITRARY "
        "(truncated) SpinDet bitmask list; PySCF conventions matching "
        "solvers._rdm.make_rdm12; exact for selected-CI wavefunctions "
        "(direct two-body Slater-Condon between list determinants).");

    m.def(
        "casscf_orbital_gradient",
        [](const Eigen::MatrixXd& h1,
           const Eigen::VectorXd& eri_chem_flat,
           const Eigen::MatrixXd& dm1,
           const Eigen::VectorXd& dm2_flat,
           int n_core, int n_act, int norb,
           const std::vector<std::pair<int, int>>& pairs) {
            vibeqc::CasscfGradientResult res;
            {
                py::gil_scoped_release rel;
                res = vibeqc::casscf_orbital_gradient(
                    h1, eri_chem_flat, dm1, dm2_flat,
                    n_core, n_act, norb, pairs);
            }
            return py::make_tuple(
                res.gradient, res.F, res.Favg, res.H_diag);
        },
        py::arg("h1"), py::arg("eri_chem_flat"),
        py::arg("dm1"), py::arg("dm2_flat"),
        py::arg("n_core"), py::arg("n_act"), py::arg("norb"),
        py::arg("pairs"),
        "CASSCF orbital gradient, generalized Fock, and diagonal Hessian. "
        "Returns (grad, F, Favg, H_diag). eri_chem is flat-packed (pq|rs) "
        "row-major over indices p,q,r,s. dm2 is flat-packed (t,u,v,w) over "
        "active indices. pairs is a list of (p,q) rotation-pair indices.");

    m.def("ccsd_t_triples",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> t1,
             py::array_t<double, py::array::c_style | py::array::forcecast> t2,
             py::array_t<double, py::array::c_style | py::array::forcecast> eri_as,
             py::array_t<double, py::array::c_style | py::array::forcecast> eps_o,
             py::array_t<double, py::array::c_style | py::array::forcecast> eps_v,
             int n_occ, int n_vir, int nso) {
              if (n_occ < 0 || n_vir < 0 || nso < 0) {
                  throw std::invalid_argument(
                      "ccsd_t_triples: n_occ, n_vir, and nso must be non-negative");
              }

              auto expect_size = [](const py::array& arr,
                                    py::ssize_t expected,
                                    const char* name) {
                  if (arr.size() != expected) {
                      throw std::invalid_argument(
                          std::string("ccsd_t_triples: ") + name
                          + " has " + std::to_string(arr.size())
                          + " elements; expected " + std::to_string(expected));
                  }
              };

              const py::ssize_t n_occ_ss = static_cast<py::ssize_t>(n_occ);
              const py::ssize_t n_vir_ss = static_cast<py::ssize_t>(n_vir);
              const py::ssize_t nso_ss = static_cast<py::ssize_t>(nso);

              expect_size(t1, n_vir_ss * n_occ_ss, "t1");
              expect_size(t2, n_vir_ss * n_vir_ss * n_occ_ss * n_occ_ss, "t2");
              expect_size(eri_as, nso_ss * nso_ss * nso_ss * nso_ss, "eri_as");
              expect_size(eps_o, n_occ_ss, "eps_o");
              expect_size(eps_v, n_vir_ss, "eps_v");

              double et = 0.0;
              {
                  py::gil_scoped_release rel;
                  et = vibeqc::ccsd_t_triples(
                      t1.data(), t2.data(), eri_as.data(),
                      eps_o.data(), eps_v.data(),
                      n_occ, n_vir, nso);
              }
              return et;
          },
          py::arg("t1"), py::arg("t2"),
          py::arg("eri_as"), py::arg("eps_o"), py::arg("eps_v"),
          py::arg("n_occ"), py::arg("n_vir"), py::arg("nso"),
          "CCSD(T) triples energy correction. On-the-fly contraction "
          "avoids 6-index intermediate tensors. OpenMP-parallel.");

    m.def("dlpno_mp2_iterate",
          [](const std::vector<int>& pair_i,
             const std::vector<int>& pair_j,
             const std::vector<Eigen::MatrixXd>& K,
             const std::vector<Eigen::VectorXd>& eps,
             const std::vector<Eigen::MatrixXd>& T,
             const std::vector<Eigen::MatrixXd>& V,
             const Eigen::MatrixXd& F_oo,
             const Eigen::MatrixXd& S_ao,
             double damping) {
              vibeqc::DLPNOMP2IterResult res;
              {
                  py::gil_scoped_release rel;
                  res = vibeqc::dlpno_mp2_iterate(
                      pair_i, pair_j, K, eps, T, V, F_oo, S_ao, damping);
              }
              return py::make_tuple(
                  res.T_new, res.max_residual, res.e_corr);
          },
          py::arg("pair_i"), py::arg("pair_j"),
          py::arg("K"), py::arg("eps"), py::arg("T"),
          py::arg("V"), py::arg("F_oo"), py::arg("S_ao"),
          py::arg("damping"),
          "One DLPNO-MP2 LMP2 amplitude iteration. "
          "Returns (T_new, max_residual, e_corr).");

    m.def(
        "selected_ci_en_pt2_deterministic",
        [](const Eigen::MatrixXd& h1,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               eri_chem,
           int n_act,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               dets_a,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               dets_b,
           const Eigen::VectorXd& ci, double e0, double eps2,
           bool use_heat_bath_walk) {
            std::vector<double> g(eri_chem.data(),
                                  eri_chem.data() + eri_chem.size());
            std::vector<std::uint64_t> da(dets_a.data(),
                                          dets_a.data() + dets_a.size());
            std::vector<std::uint64_t> db(dets_b.data(),
                                          dets_b.data() + dets_b.size());
            double e2 = 0.0;
            std::int64_t n_pert = 0;
            {
                py::gil_scoped_release rel;
                e2 = vibeqc::selected_ci_en_pt2_deterministic(
                    h1, g, n_act, da, db, ci, e0, eps2,
                    use_heat_bath_walk, &n_pert);
            }
            return py::make_tuple(e2, n_pert);
        },
        py::arg("h1"), py::arg("eri_chem"), py::arg("n_act"),
        py::arg("dets_a"), py::arg("dets_b"), py::arg("ci"),
        py::arg("e0"), py::arg("eps2") = 0.0,
        py::arg("use_heat_bath_walk") = true,
        "Deterministic Epstein-Nesbet PT2 on a selected wavefunction "
        "(Sharma et al JCTC 13, 1595 (2017) Eq. 5): coherent perturber "
        "numerators screened at |H_ai c_i| >= eps2, heat-bath-walked "
        "double enumeration. e0 is the ACTIVE-SPACE variational "
        "eigenvalue. Returns (e2, n_perturbers).");

    m.def(
        "selected_ci_en_pt2_stochastic",
        [](const Eigen::MatrixXd& h1,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               eri_chem,
           int n_act,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               dets_a,
           py::array_t<std::uint64_t,
                       py::array::c_style | py::array::forcecast>
               dets_b,
           const Eigen::VectorXd& ci, double e0, double eps2,
           double eps2_loose, int n_samples, int sample_size,
           std::uint64_t seed, bool use_heat_bath_walk) {
            std::vector<double> g(eri_chem.data(),
                                  eri_chem.data() + eri_chem.size());
            std::vector<std::uint64_t> da(dets_a.data(),
                                          dets_a.data() + dets_a.size());
            std::vector<std::uint64_t> db(dets_b.data(),
                                          dets_b.data() + dets_b.size());
            double mean = 0.0, sem = 0.0;
            {
                py::gil_scoped_release rel;
                vibeqc::selected_ci_en_pt2_stochastic(
                    h1, g, n_act, da, db, ci, e0, eps2, eps2_loose,
                    n_samples, sample_size, seed, use_heat_bath_walk,
                    &mean, &sem);
            }
            return py::make_tuple(mean, sem);
        },
        py::arg("h1"), py::arg("eri_chem"), py::arg("n_act"),
        py::arg("dets_a"), py::arg("dets_b"), py::arg("ci"),
        py::arg("e0"), py::arg("eps2"), py::arg("eps2_loose"),
        py::arg("n_samples"), py::arg("sample_size"),
        py::arg("seed") = 0, py::arg("use_heat_bath_walk") = true,
        "Stochastic difference term of the semistochastic EN-PT2 "
        "(Sharma et al JCTC 13, 1595 (2017) Eqs. 7-11): mean +/- stderr "
        "over n_samples batches of the unbiased estimator of "
        "E2[eps2] - E2[eps2_loose], each batch drawing sample_size "
        "determinants with replacement from p_i ~ |c_i|. Returns "
        "(mean, stderr).");

    // ----- UMP2 (open-shell MP2) -----------------------------------------
    py::class_<vibeqc::UMP2Options>(m, "UMP2Options")
        .def(py::init<>())
        .def_property(
            "n_frozen_core",
            [](const vibeqc::UMP2Options& options) -> py::object {
                if (options.n_frozen_core == -1) return py::none();
                return py::int_(options.n_frozen_core);
            },
            [](vibeqc::UMP2Options& options, const py::object& requested) {
                if (requested.is_none()) {
                    options.n_frozen_core = -1;
                    return;
                }
                const int count = requested.cast<int>();
                if (count < 0) {
                    throw py::value_error(
                        "n_frozen_core must be None or a non-negative integer");
                }
                options.n_frozen_core = count;
            },
            "Frozen doubly occupied spatial orbitals. None (default) resolves "
            "the published ORCA 6.1 Table 2.69 count; 0 requests all-electron "
            "correlation; positive values are explicit counts. The published "
            "policy is count-only and does not reorder orbitals by atomic "
            "character.")
        .def_readwrite("density_fit",
                       &vibeqc::UMP2Options::density_fit,
                       "Enable density fitting for the AO→MO ERI "
                       "transform. The αα / ββ / αβ OVOV tensors all "
                       "route through the DF factorisation; α and β "
                       "MO B-tensors are built once each and the three "
                       "blocks form via single GEMMs. See "
                       "MP2Options.density_fit. Default False.")
        .def_readwrite("aux_basis",
                       &vibeqc::UMP2Options::aux_basis,
                       "Auxiliary basis name for density fitting. "
                       "See MP2Options.aux_basis.")
        .def_readwrite("c_os", &vibeqc::UMP2Options::c_os,
                       "Opposite-spin (αβ channel) scaling coefficient. "
                       "Default 1.0 (canonical UMP2). See "
                       "``MP2Options.c_os`` for the named SCS / SOS / "
                       "B2PLYP recipes.")
        .def_readwrite("c_ss", &vibeqc::UMP2Options::c_ss,
                       "Same-spin (αα+ββ channels) scaling coefficient. "
                       "Default 1.0 (canonical UMP2). See "
                       "``MP2Options.c_ss``.")
        .def_readwrite("report_ri_residual",
                       &vibeqc::UMP2Options::report_ri_residual,
                       "Diagnostic: when density_fit=True, also build "
                       "the canonical four-index OVOV tensors for all "
                       "three (αα, ββ, αβ) channels and report the "
                       "per-channel residual on the UMP2Result. Doubles "
                       "the runtime cost — opt-in, intended for "
                       "verifying aux-basis adequacy. See "
                       "MP2Options.report_ri_residual. Default False.")
        .def_readwrite("requested_memory_bytes",
                       &vibeqc::UMP2Options::requested_memory_bytes,
                       "Maximum UMP2 workspace in bytes. Zero preserves the "
                       "low-level in-core auto default; run_job derives a "
                       "nonzero cap from the process allocation.")
        .def_readwrite("memory_mode",
                       &vibeqc::UMP2Options::memory_mode,
                       "Integral-storage strategy: 'auto', 'incore', "
                       "'direct', or 'disk'. Default 'auto'.")
        .def_readwrite("scratch_directory",
                       &vibeqc::UMP2Options::scratch_directory,
                       "Directory for bounded semi-direct scratch files.");

    py::class_<vibeqc::UMP2Result>(m, "UMP2Result")
        .def_readonly("e_hf",          &vibeqc::UMP2Result::e_hf)
        .def_readonly("e_correlation", &vibeqc::UMP2Result::e_correlation)
        .def_readonly("e_total",       &vibeqc::UMP2Result::e_total)
        .def_readonly("n_frozen_core", &vibeqc::UMP2Result::n_frozen_core,
                      "Resolved number of frozen occupied spatial orbitals.")
        .def_readonly("e_aa",          &vibeqc::UMP2Result::e_aa,
                      "αα same-spin channel")
        .def_readonly("e_bb",          &vibeqc::UMP2Result::e_bb,
                      "ββ same-spin channel")
        .def_readonly("e_ab",          &vibeqc::UMP2Result::e_ab,
                      "αβ opposite-spin channel")
        .def_readonly("e_aa_ri_residual",
                      &vibeqc::UMP2Result::e_aa_ri_residual,
                      "RI fit residual e_aa_RI − e_aa_canonical, "
                      "populated only when ``UMP2Options.density_fit && "
                      "report_ri_residual``. 0.0 otherwise; check "
                      "``ri_residual_reported`` to distinguish.")
        .def_readonly("e_bb_ri_residual",
                      &vibeqc::UMP2Result::e_bb_ri_residual,
                      "RI fit residual e_bb_RI − e_bb_canonical, "
                      "populated only when ``UMP2Options.density_fit && "
                      "report_ri_residual``.")
        .def_readonly("e_ab_ri_residual",
                      &vibeqc::UMP2Result::e_ab_ri_residual,
                      "RI fit residual e_ab_RI − e_ab_canonical, "
                      "populated only when ``UMP2Options.density_fit && "
                      "report_ri_residual``.")
        .def_readonly("ri_residual_reported",
                      &vibeqc::UMP2Result::ri_residual_reported,
                      "True iff ``report_ri_residual`` was set on the "
                      "UMP2Options for this run AND density_fit was on.")
        .def_readonly("memory_mode_used", &vibeqc::UMP2Result::memory_mode_used,
                      "Resolved integral-storage strategy used by the run.")
        .def_readonly("workspace_bytes", &vibeqc::UMP2Result::workspace_bytes,
                      "Modeled peak UMP2 workspace allocated by the kernel.")
        .def_readonly("disk_bytes", &vibeqc::UMP2Result::disk_bytes,
                      "Peak scratch-file extent used by semi-direct UMP2.")
        .def("__repr__", [](const vibeqc::UMP2Result& r) {
            return std::string{"UMP2Result(e_total="}
                   + std::to_string(r.e_total)
                   + ", e_corr=" + std::to_string(r.e_correlation) + ")";
        });

    m.def("run_ump2", &vibeqc::run_ump2,
          py::arg("molecule"), py::arg("basis"), py::arg("uhf_result"),
          py::arg("options") = vibeqc::UMP2Options{},
          py::call_guard<py::gil_scoped_release>(),
          "Run open-shell UMP2 correlation on a converged UHF reference. "
          "Pass options.density_fit=True with options.aux_basis set to "
          "use density fitting (RI-UMP2).");


    // ----- CCSD / CCSD(T) (coupled-cluster) -----------------------------
    py::class_<vibeqc::CCSDOptions>(m, "CCSDOptions")
        .def(py::init<>())
        .def_readwrite("density_fit",
                       &vibeqc::CCSDOptions::density_fit,
                       "Enable density fitting for the two-electron integrals. "
                       "Default True — DF is the standard approach for CC.")
        .def_readwrite("aux_basis",
                               &vibeqc::CCSDOptions::aux_basis,
                               "Auxiliary basis name for density fitting "
                               "(libint-recognised, e.g. \"def2-tzvp-rifit\"). "
                               "Empty + density_fit=True raises.")
        .def_readwrite("max_iter",
                       &vibeqc::CCSDOptions::max_iter,
                       "Maximum CCSD iterations. Default 100.")
        .def_readwrite("conv_tol_energy",
                       &vibeqc::CCSDOptions::conv_tol_energy,
                       "Convergence threshold on energy change (Ha). Default 1e-8.")
        .def_readwrite("conv_tol_residual",
                       &vibeqc::CCSDOptions::conv_tol_residual,
                       "Convergence threshold on ||R1|| + ||R2||. Default 1e-7.")
        .def_readwrite("diis_subspace_size",
                       &vibeqc::CCSDOptions::diis_subspace_size,
                       "DIIS subspace size for amplitude extrapolation. "
                       "Set to 0 to disable DIIS. Default 6.")
        .def_property(
            "n_frozen_core",
            [](const vibeqc::CCSDOptions& options) -> py::object {
                if (options.n_frozen_core == -1) return py::none();
                return py::int_(options.n_frozen_core);
            },
            [](vibeqc::CCSDOptions& options, const py::object& requested) {
                if (requested.is_none()) {
                    options.n_frozen_core = -1;
                    return;
                }
                const int count = requested.cast<int>();
                if (count < 0) {
                    throw py::value_error(
                        "n_frozen_core must be None or a non-negative integer");
                }
                options.n_frozen_core = count;
            },
            "Frozen occupied spatial orbitals. None (default) resolves the "
            "published ORCA 6.1 Table 2.69 count; 0 requests all-electron "
            "correlation; positive values are explicit counts. The published "
            "policy is count-only and does not reorder orbitals by atomic "
            "character.")
        .def_readwrite("compute_triples",
                       &vibeqc::CCSDOptions::compute_triples,
                       "Compute the perturbative (T) correction after CCSD "
                       "converges. Default True.")
        .def_readwrite("triples_memory_mode",
                       &vibeqc::CCSDOptions::triples_memory_mode,
                       "Triples memory strategy: 'auto' (default), 'fast', "
                       "'blocked', 'direct', or 'disk'. Legacy 'low' is an "
                       "alias for 'blocked'.")
        .def_readwrite("requested_memory_bytes",
                       &vibeqc::CCSDOptions::requested_memory_bytes,
                       "Maximum modeled triples-phase peak in bytes, including "
                       "retained triples inputs and aggregate worker scratch. "
                       "Zero leaves the low-level planner uncapped; run_job "
                       "derives a nonzero cap from the process allocation.")
        .def_readwrite("triples_tile_size",
                       &vibeqc::CCSDOptions::triples_tile_size,
                       "Optional external-virtual tile size. Zero lets the "
                       "budget planner choose.")
        .def_readwrite("triples_max_threads",
                       &vibeqc::CCSDOptions::triples_max_threads,
                       "Maximum worker threads for triples. Zero uses the "
                       "OpenMP limit subject to the aggregate memory budget.")
        .def_readwrite("triples_scratch_directory",
                       &vibeqc::CCSDOptions::triples_scratch_directory,
                       "Directory for disk-backed triples factors. Empty "
                       "uses the platform temporary directory.")
        .def_readwrite("cc_variant",
                       &vibeqc::CCSDOptions::cc_variant,
                       "Approximate coupled-cluster / coupled-pair variant "
                       "of the amplitude equations: \"ccsd\" (default), "
                       "\"cc2\", \"ccd\", \"bccd\" (caller must "
                       "supply Brueckner orbitals), \"lccd\", \"lccsd\" "
                       "(= \"cepa(0)\"), \"cepa(1)\", \"cepa(2)\", "
                       "\"cepa(3)\", \"qcisd\".  Closed-shell kernel only; "
                       "compute_triples is supported for CCSD, BCCD, and "
                       "QCISD.")
        .def_readwrite("triples_variant",
                       &vibeqc::CCSDOptions::triples_variant,
                       "Perturbative-triples correction selected into "
                       "result.e_t when compute_triples is True: \"(t)\" "
                       "(default, standard Raghavachari CCSD(T)) or "
                       "\"[t]\" (bracket CCSD[T] = CCSD+T(CCSD), the "
                       "fourth-order piece only). Closed-shell kernel "
                       "only.");

    py::class_<vibeqc::CCSDIteration>(m, "CCSDIteration",
        "One row of the CCSD iteration trace.")
        .def_readonly("iter",          &vibeqc::CCSDIteration::iter)
        .def_readonly("energy",        &vibeqc::CCSDIteration::energy)
        .def_readonly("delta_e",       &vibeqc::CCSDIteration::delta_e)
        .def_readonly("r1_norm",       &vibeqc::CCSDIteration::r1_norm)
        .def_readonly("r2_norm",       &vibeqc::CCSDIteration::r2_norm)
        .def_readonly("diis_subspace", &vibeqc::CCSDIteration::diis_subspace);

    py::class_<vibeqc::CCSDResult>(m, "CCSDResult")
        .def_readonly("e_hf",                &vibeqc::CCSDResult::e_hf,
                      "Underlying RHF total energy (Hartree).")
        .def_readonly("e_ccsd_correlation",   &vibeqc::CCSDResult::e_ccsd_correlation,
                      "CCSD correlation energy contribution.")
        .def_readonly("e_ccsd",              &vibeqc::CCSDResult::e_ccsd,
                      "e_hf + e_ccsd_correlation.")
        .def_readonly("e_t",                 &vibeqc::CCSDResult::e_t,
                      "Selected perturbative triples correction (0 if not "
                      "computed): E[T]+E_ST for triples_variant \"(t)\", "
                      "E[T] for \"[t]\".")
        .def_readonly("e_t4",                &vibeqc::CCSDResult::e_t4,
                      "Fourth-order connected-triples piece E[T] "
                      "(the CCSD[T] correction).")
        .def_readonly("e_t5_st",             &vibeqc::CCSDResult::e_t5_st,
                      "Fifth-order singles-triples piece E_ST "
                      "(E(T) = E[T] + E_ST).")
        .def_readonly("e_ccsd_t",            &vibeqc::CCSDResult::e_ccsd_t,
                      "e_ccsd + e_t.")
        .def_readonly("e_total",             &vibeqc::CCSDResult::e_total,
                      "Total CCSD(T) energy (alias for e_ccsd_t).")
        .def_readonly("n_iter",              &vibeqc::CCSDResult::n_iter)
        .def_readonly("converged",           &vibeqc::CCSDResult::converged)
        .def_readonly("t1_norm",             &vibeqc::CCSDResult::t1_norm,
                      "Frobenius norm of converged T1 amplitudes.")
        .def_readonly("t2_norm",             &vibeqc::CCSDResult::t2_norm,
                      "Frobenius norm of converged T2 amplitudes.")
        .def_readonly("t1_amplitudes",       &vibeqc::CCSDResult::t1_amplitudes,
                      "Final T1 amplitudes with shape "
                      "(n_occ_active, n_vir).")
        .def_readonly("cc_trace",            &vibeqc::CCSDResult::cc_trace,
                      "Per-iteration CCSD trace.")
        .def_readonly("triples_memory_mode_used",
                      &vibeqc::CCSDResult::triples_memory_mode_used,
                      "Resolved triples memory strategy used by the run.")
        .def_readonly("triples_tile_size_used",
                      &vibeqc::CCSDResult::triples_tile_size_used,
                      "External-virtual tile size used by triples.")
        .def_readonly("triples_threads_used",
                      &vibeqc::CCSDResult::triples_threads_used,
                      "Worker count used by the triples kernel.")
        .def_readonly("triples_workspace_bytes",
                      &vibeqc::CCSDResult::triples_workspace_bytes,
                      "Full modeled triples-phase peak in bytes: retained "
                      "inputs plus aggregate active-worker scratch.")
        .def_readonly("triples_disk_bytes",
                      &vibeqc::CCSDResult::triples_disk_bytes,
                      "Peak scratch-file extent used by disk-backed triples.")
        .def("__repr__", [](const vibeqc::CCSDResult& r) {
            return std::string{"CCSDResult(e_total="}
                   + std::to_string(r.e_total)
                   + ", e_corr=" + std::to_string(r.e_ccsd_correlation)
                   + ", e_t=" + std::to_string(r.e_t) + ")";
        });

    m.def("run_ccsd", &vibeqc::run_ccsd,
          py::arg("molecule"), py::arg("basis"), py::arg("rhf_result"),
          py::arg("options") = vibeqc::CCSDOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run closed-shell CCSD (and optionally CCSD(T)) on a "
          "converged RHF reference.  Density-fitted integrals by default "
          "(density_fit=True with an auxiliary basis set); "
          "density_fit=False selects the canonical exact four-index "
          "route.");

    m.def("run_ccsd_from_mos", &vibeqc::run_ccsd_from_mos,
          py::arg("molecule"), py::arg("basis"),
          py::arg("mo_coeffs"), py::arg("fock"), py::arg("e_hf"),
          py::arg("ecp_total_ncore"),
          py::arg("options") = vibeqc::CCSDOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run closed-shell CCSD(T) from explicit MO coefficients and "
          "an AO Fock matrix — the entry point for FNO-CCSD(T), where the "
          "virtual space is truncated to the dominant MP2 natural orbitals "
          "and semicanonicalized.  C is (n_ao x n_occ+n_vir_kept) with the "
          "occupied columns first; F is the AO Fock (passed unchanged).  "
          "ecp_total_ncore is mandatory SCF provenance: the occupied space "
          "is partitioned from n_electrons - ecp_total_ncore, and the "
          "published frozen-core default is refused for a nonzero count "
          "(pass the ECP-aware count).  Orbital energies are diag(C^T F C).  "
          "Density-fitted by default; "
          "density_fit=False selects the canonical exact four-index "
          "route.");

    m.def("run_uccsd", &vibeqc::run_uccsd,
          py::arg("molecule"), py::arg("basis"), py::arg("uhf_result"),
          py::arg("options") = vibeqc::CCSDOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run open-shell UCCSD (and optionally UCCSD(T)) on a "
          "converged UHF reference.  Spin-orbital formulation; "
          "density-fitted by default, canonical exact four-index route "
          "with density_fit=False.  Returns a "
          "CCSDResult (same fields as the closed-shell kernel).");

    m.def("run_uccsd_from_mos", &vibeqc::run_uccsd_from_mos,
          py::arg("molecule"), py::arg("basis"),
          py::arg("mo_coeffs_alpha"), py::arg("mo_coeffs_beta"),
          py::arg("fock_alpha"), py::arg("fock_beta"),
          py::arg("e_hf"), py::arg("ecp_total_ncore"),
          py::arg("options") = vibeqc::CCSDOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Run open-shell UCCSD(T) from explicit MO-coefficient and "
          "per-spin Fock matrices — the entry point for ROHF-reference "
          "CCSD and any other non-UHF reference.  C_alpha / C_beta are "
          "AO-to-MO coefficient matrices (n_ao x n_orb); for ROHF these "
          "are identical.  F_alpha / F_beta are the per-spin AO Fock "
          "matrices.  e_hf is the SCF reference energy.  "
          "ecp_total_ncore is mandatory provenance; the occupied space is "
          "partitioned from n_electrons - ecp_total_ncore.  Density-fitted by "
          "default, canonical exact four-index route with "
          "density_fit=False.  Returns a "
          "CCSDResult (same fields as the closed-shell kernel).");

    using RowMatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>;
    m.def(
        "dlpno_uccsd_pair_residual",
        [](const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const RowMatD& b_so, const Eigen::MatrixXd& f_so) {
            Eigen::MatrixXd R1, R2_flat;
            {
                // Release the GIL only while the native integral build and
                // residual contraction run; tuple conversion needs it.
                py::gil_scoped_release release;
                vibeqc::dlpno_uccsd_pair_residual(
                    t1, t2_flat, b_so, f_so, R1, R2_flat);
            }
            return py::make_tuple(R1, R2_flat);
        },
        py::arg("t1"), py::arg("t2_flat"), py::arg("b_so"), py::arg("f_so"),
        "Single caller-supplied local spin-orbital UCCSD residual. The "
        "basis is ordered [occupied | virtual], b_so pair columns are "
        "flattened as p*n+q, and t2_flat uses rows i*n_occ+j and columns "
        "a*n_vir+b. Returns (R1, R2_flat).");

    m.def(
        "dlpno_spin_orbital_triple_energy",
        [](Eigen::Index i, Eigen::Index j, Eigen::Index k,
           const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const RowMatD& eri_vovv, const RowMatD& eri_ovoo,
           const RowMatD& eri_oovv, const Eigen::VectorXd& eps_o,
           const Eigen::VectorXd& eps_v) {
            double e_t = 0.0;
            {
                py::gil_scoped_release release;
                e_t = vibeqc::dlpno_spin_orbital_triple_energy(
                    i, j, k, t1, t2_flat, eri_vovv, eri_ovoo, eri_oovv,
                    eps_o, eps_v);
            }
            return e_t;
        },
        py::arg("i"), py::arg("j"), py::arg("k"), py::arg("t1"),
        py::arg("t2_flat"), py::arg("eri_vovv"), py::arg("eri_ovoo"),
        py::arg("eri_oovv"), py::arg("eps_o"), py::arg("eps_v"),
        "Spin-orbital CCSD(T) energy for one distinct occupied triple in a "
        "caller-supplied virtual domain. Native counterpart of "
        "vibeqc.dlpno._ccsd_ref.so_triple_energy.");

    m.def(
        "dlpno_spatial_triples_correction",
        [](const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const RowMatD& b_ov, const RowMatD& b_oo, const RowMatD& b_vv,
           const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v) {
            double e_t = 0.0;
            {
                // Release the GIL around the OpenMP triples contraction;
                // pybind's scalar return conversion reacquires it below.
                py::gil_scoped_release release;
                e_t = vibeqc::dlpno_spatial_triples_correction(
                    t1, t2_flat, b_ov, b_oo, b_vv, eps_o, eps_v);
            }
            return e_t;
        },
        py::arg("t1"), py::arg("t2_flat"), py::arg("b_ov"), py::arg("b_oo"),
        py::arg("b_vv"), py::arg("eps_o"), py::arg("eps_v"),
        "Closed-shell spatial CCSD(T) triples correction from caller-supplied "
        "DF blocks. Native OpenMP counterpart of "
        "vibeqc.dlpno._ccsd_cs.cs_triples_correction.");

    m.def(
        "dlpno_spatial_triple_energy",
        [](Eigen::Index i, Eigen::Index j, Eigen::Index k,
           const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const RowMatD& ov_vv, const RowMatD& oo_ov, const RowMatD& ov_ov,
           const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v) {
            double e_t = 0.0;
            {
                py::gil_scoped_release release;
                e_t = vibeqc::dlpno_spatial_triple_energy(
                    i, j, k, t1, t2_flat, ov_vv, oo_ov, ov_ov, eps_o, eps_v);
            }
            return e_t;
        },
        py::arg("i"), py::arg("j"), py::arg("k"),
        py::arg("t1"), py::arg("t2_flat"), py::arg("ov_vv"),
        py::arg("oo_ov"), py::arg("ov_ov"), py::arg("eps_o"),
        py::arg("eps_v"),
        "Closed-shell spatial CCSD(T) energy for one ordered occupied triple "
        "from caller-supplied local integral blocks. Native counterpart of "
        "vibeqc.dlpno._ccsd_cs.cs_triple_energy.");

    m.def(
        "dlpno_project_tno_amplitudes",
        [](const Eigen::MatrixXd& V_T, int n_act,
           const std::vector<int>& pair_i,
           const std::vector<int>& pair_j,
           const std::vector<Eigen::MatrixXd>& U_pno,
           const std::vector<Eigen::MatrixXd>& T2_pno,
           const std::vector<int>& t1_idx,
           const std::vector<Eigen::VectorXd>& t1_vec) {
            Eigen::MatrixXd t1_out, T2_out;
            {
                py::gil_scoped_release release;
                vibeqc::dlpno_project_tno_amplitudes(
                    V_T, n_act, pair_i, pair_j, U_pno, T2_pno,
                    t1_idx, t1_vec, t1_out, T2_out);
            }
            return py::make_tuple(t1_out, T2_out);
        },
        py::arg("V_T"), py::arg("n_act"),
        py::arg("pair_i"), py::arg("pair_j"),
        py::arg("U_pno"), py::arg("T2_pno"),
        py::arg("t1_idx"), py::arg("t1_vec"),
        "Project DLPNO pair amplitudes from per-pair PNO bases into a "
        "common TNO basis.  Returns (t1_out, T2_out) where t1_out is "
        "(n_act, n_T) and T2_out is (n_act*n_act, n_T*n_T) row-major.");

    m.def(
        "dlpno_build_tno_density",
        [](const Eigen::MatrixXd& V_T,
           const std::vector<int>& distinct,
           const std::vector<int>& pair_i,
           const std::vector<int>& pair_j,
           const std::vector<Eigen::MatrixXd>& U_pno,
           const std::vector<Eigen::MatrixXd>& T2_pno,
           double tcut_tno) {
            Eigen::MatrixXd result;
            {
                py::gil_scoped_release release;
                result = vibeqc::dlpno_build_tno_density(
                    V_T, distinct, pair_i, pair_j, U_pno, T2_pno, tcut_tno);
            }
            return result;
        },
        py::arg("V_T"), py::arg("distinct"),
        py::arg("pair_i"), py::arg("pair_j"),
        py::arg("U_pno"), py::arg("T2_pno"), py::arg("tcut_tno"),
        "Build TNO occupation density from per-pair amplitudes, diagonalise, "
        "and truncate V_T by occupation threshold. Returns truncated V_T.");

    m.def(
        "dlpno_pair_residual",
        [](const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const RowMatD& b_ov, const RowMatD& b_oo, const RowMatD& b_vv,
           const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
           const Eigen::MatrixXd& f_ov, bool include_ladder,
           bool include_ring, bool return_ring) -> py::tuple {
            Eigen::MatrixXd R1, R2_flat, W1, W2, WX;
            {
                // Release the GIL only around the pure-C++ kernel; the
                // py::make_tuple return conversion below needs the GIL.
                py::gil_scoped_release release;
                vibeqc::dlpno_pair_residual(
                    t1, t2_flat, b_ov, b_oo, b_vv, f_oo, f_vv, f_ov,
                    R1, R2_flat, include_ladder, include_ring,
                    return_ring ? &W1 : nullptr,
                    return_ring ? &W2 : nullptr,
                    return_ring ? &WX : nullptr);
            }
            if (return_ring)
                return py::make_tuple(R1, R2_flat,
                                      py::make_tuple(W1, W2, WX));
            return py::make_tuple(R1, R2_flat);
        },
        py::arg("t1"), py::arg("t2_flat"), py::arg("b_ov"), py::arg("b_oo"),
        py::arg("b_vv"), py::arg("f_oo"), py::arg("f_vv"), py::arg("f_ov"),
        py::arg("include_ladder") = true, py::arg("include_ring") = true,
        py::arg("return_ring") = false,
        "Single closed-shell DF-CCSD residual (the DLPNO local-solver "
        "per-pair kernel). Returns (R1, R2_flat), or (R1, R2_flat, "
        "(W1, W2, WX)) when return_ring=True. include_ladder=False omits "
        "the particle-particle W_abef term and its O(nv^4) (ae|bf) build; "
        "include_ring=False omits the three ring contractions from the "
        "Ph-symmetrised half. In both cases the caller must contract the "
        "omitted term elsewhere (issue #700).");

    m.def(
        "dlpno_target_pair_residual",
        [](const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const RowMatD& b_ov, const RowMatD& b_oo, const RowMatD& b_vv,
           const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
           const Eigen::MatrixXd& f_ov,
           Eigen::Index target_i, Eigen::Index target_j,
           bool include_ladder, bool include_ring,
           bool return_ring) -> py::tuple {
            Eigen::MatrixXd R1_target, R2_target, W1, W2, WX;
            {
                py::gil_scoped_release release;
                vibeqc::dlpno_target_pair_residual(
                    t1, t2_flat, b_ov, b_oo, b_vv, f_oo, f_vv, f_ov,
                    target_i, target_j, R1_target, R2_target, include_ladder,
                    include_ring,
                    return_ring ? &W1 : nullptr,
                    return_ring ? &W2 : nullptr,
                    return_ring ? &WX : nullptr);
            }
            if (return_ring)
                return py::make_tuple(R1_target, R2_target,
                                      py::make_tuple(W1, W2, WX));
            return py::make_tuple(R1_target, R2_target);
        },
        py::arg("t1"), py::arg("t2_flat"), py::arg("b_ov"), py::arg("b_oo"),
        py::arg("b_vv"), py::arg("f_oo"), py::arg("f_vv"), py::arg("f_ov"),
        py::arg("target_i"), py::arg("target_j"),
        py::arg("include_ladder") = true, py::arg("include_ring") = true,
        py::arg("return_ring") = false,
        "Target-selective closed-shell DF-CCSD residual for one DLPNO pair. "
        "Returns (R1[target_i], R2[target_i,target_j]), or that plus "
        "(W1, W2, WX) when return_ring=True. include_ladder=False omits the "
        "W_abef term and its O(nv^4) build; include_ring=False omits the "
        "three ring contractions (issue #700).");

    // ----- Spatial (Python-bindable) CCSD residual + solver ---------------
    // These are direct pybind11 wrappers for the ccsd.cpp spatial
    // residual and solver; they accept the same flat-matrix integral-block
    // layout Python callers already build (reshape 4D numpy arrays to 2D
    // without copy).  GIL-released: the residual and solver loops are pure
    // C++ + OpenMP.

    m.def(
        "spatial_ccsd_residual",
        [](const Eigen::MatrixXd& t1, const Eigen::MatrixXd& t2_flat,
           const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
           const Eigen::MatrixXd& f_ov,
           const Eigen::MatrixXd& ovov_flat, const Eigen::MatrixXd& oooo_flat,
           const Eigen::MatrixXd& ooov_flat, const Eigen::MatrixXd& oovv_flat,
           const Eigen::MatrixXd& ovvv_flat, const Eigen::MatrixXd& vvvv_flat) {
            Eigen::MatrixXd R1, R2_flat;
            vibeqc::spatial_ccsd_residual(
                t1, t2_flat, f_oo, f_vv, f_ov,
                ovov_flat, oooo_flat, ooov_flat, oovv_flat,
                ovvv_flat, vvvv_flat, R1, R2_flat);
            return py::make_tuple(R1, R2_flat);
        },
        py::arg("t1"), py::arg("t2_flat"),
        py::arg("f_oo"), py::arg("f_vv"), py::arg("f_ov"),
        py::arg("ovov_flat"), py::arg("oooo_flat"),
        py::arg("ooov_flat"), py::arg("oovv_flat"),
        py::arg("ovvv_flat"), py::arg("vvvv_flat"),
        "Single closed-shell spatial CCSD residual (numpy-array entry point). "
        "Integral blocks are (no*nv, no*nv) flat matrices — reshape 4D numpy "
        "arrays to 2D before calling (no copy for C-contiguous data with "
        "RowMajor Eigen maps). Returns (R1, R2_flat).");

    m.def(
        "spatial_ccsd_solve",
        [](const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_vv,
           const Eigen::MatrixXd& f_ov,
           const Eigen::MatrixXd& ovov_flat, const Eigen::MatrixXd& oooo_flat,
           const Eigen::MatrixXd& ooov_flat, const Eigen::MatrixXd& oovv_flat,
           const Eigen::MatrixXd& ovvv_flat, const Eigen::MatrixXd& vvvv_flat,
           double e_hf,
           int max_iter, double conv_tol_energy, double conv_tol_residual,
           int diis_size) {
            vibeqc::SpatialCCSDResult r =
                vibeqc::spatial_ccsd_solve(
                    f_oo, f_vv, f_ov,
                    ovov_flat, oooo_flat, ooov_flat, oovv_flat,
                    ovvv_flat, vvvv_flat,
                    e_hf, max_iter, conv_tol_energy, conv_tol_residual,
                    diis_size);
            return py::make_tuple(
                r.T1, r.T2_flat, r.e_corr, r.n_iter, r.converged);
        },
        py::arg("f_oo"), py::arg("f_vv"), py::arg("f_ov"),
        py::arg("ovov_flat"), py::arg("oooo_flat"),
        py::arg("ooov_flat"), py::arg("oovv_flat"),
        py::arg("ovvv_flat"), py::arg("vvvv_flat"),
        py::arg("e_hf") = 0.0,
        py::arg("max_iter") = 100,
        py::arg("conv_tol_energy") = 1e-8,
        py::arg("conv_tol_residual") = 1e-7,
        py::arg("diis_size") = 6,
        "Full closed-shell spatial CCSD solver from pre-built integral blocks. "
        "Returns (T1, T2_flat, E_corr, n_iter, converged).");

    m.def(
        "spatial_lambda_triples",
        [](const Eigen::MatrixXd& T1, const Eigen::MatrixXd& T2_flat,
           const Eigen::MatrixXd& L1, const Eigen::MatrixXd& L2_flat,
           const Eigen::MatrixXd& f_ov,
           const Eigen::MatrixXd& ov_vv, const Eigen::MatrixXd& oo_ov,
           const Eigen::MatrixXd& ov_ov,
           const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v) {
            double e_t = vibeqc::spatial_lambda_triples(
                T1, T2_flat, L1, L2_flat, f_ov,
                ov_vv, oo_ov, ov_ov, eps_o, eps_v);
            return e_t;
        },
        py::arg("t1"), py::arg("t2_flat"),
        py::arg("l1"), py::arg("l2_flat"),
        py::arg("f_ov"),
        py::arg("ov_vv"), py::arg("oo_ov"), py::arg("ov_ov"),
        py::arg("eps_o"), py::arg("eps_v"),
        "Closed-shell A-CCSD(T) Lambda triples correction. "
        "OpenMP-parallel over occupied triples. "
        "Returns the Lambda (T) energy contribution.");

    // ----- Crystal / space-group analysis (spglib) -----------------------
    py::class_<vibeqc::Crystal>(m, "Crystal")
        .def(py::init<>())
        .def(py::init([](const Eigen::Matrix3d& lattice,
                         const Eigen::Matrix3Xd& fractional_coords,
                         const std::vector<int>& species) {
                 vibeqc::Crystal c;
                 c.lattice = lattice;
                 c.fractional_coords = fractional_coords;
                 c.species = species;
                 if (static_cast<int>(fractional_coords.cols()) !=
                     static_cast<int>(species.size())) {
                     throw std::runtime_error(
                         "Crystal: fractional_coords must have one column "
                         "per species entry");
                 }
                 return c;
             }),
             py::arg("lattice"), py::arg("fractional_coords"),
             py::arg("species"),
             "Construct a Crystal. ``lattice`` columns are the Cartesian "
             "lattice vectors in bohr; ``fractional_coords`` is a 3×N matrix "
             "with one atom per column.")
        .def_readwrite("lattice",            &vibeqc::Crystal::lattice)
        .def_readwrite("fractional_coords",  &vibeqc::Crystal::fractional_coords)
        .def_readwrite("species",            &vibeqc::Crystal::species)
        .def_property_readonly("n_atoms",    &vibeqc::Crystal::n_atoms)
        .def("__repr__", [](const vibeqc::Crystal& c) {
            return std::string{"Crystal(n_atoms="}
                   + std::to_string(c.n_atoms()) + ")";
        });

    py::class_<vibeqc::SymmetryOp>(m, "SymmetryOp")
        .def_readonly("rotation",    &vibeqc::SymmetryOp::rotation,
                      "Integer rotation matrix in the fractional basis.")
        .def_readonly("translation", &vibeqc::SymmetryOp::translation,
                      "Translation in fractional coordinates.")
        .def("__repr__", [](const vibeqc::SymmetryOp&) {
            return std::string{"SymmetryOp(...)"};
        });

    py::class_<vibeqc::SpaceGroup>(m, "SpaceGroup")
        .def_readonly("number",                &vibeqc::SpaceGroup::number)
        .def_readonly("international_symbol",
                      &vibeqc::SpaceGroup::international_symbol)
        .def_readonly("hall_number",           &vibeqc::SpaceGroup::hall_number)
        .def_readonly("point_group",           &vibeqc::SpaceGroup::point_group)
        .def_readonly("operations",            &vibeqc::SpaceGroup::operations)
        .def_readonly("equivalent_atoms",      &vibeqc::SpaceGroup::equivalent_atoms)
        .def_property_readonly("order", [](const vibeqc::SpaceGroup& sg) {
            return static_cast<int>(sg.operations.size());
        })
        .def("__repr__", [](const vibeqc::SpaceGroup& sg) {
            return std::string{"SpaceGroup(number="}
                   + std::to_string(sg.number) + ", symbol='"
                   + sg.international_symbol + "', order="
                   + std::to_string(sg.operations.size()) + ")";
        });

    m.def("analyze_symmetry", &vibeqc::analyze,
          py::arg("crystal"), py::arg("symprec") = 1.0e-5,
          py::call_guard<py::gil_scoped_release>(),
          "Run spglib space-group analysis on a Crystal.");

    m.def("to_primitive", &vibeqc::to_primitive,
          py::arg("crystal"), py::arg("symprec") = 1.0e-5,
          py::call_guard<py::gil_scoped_release>(),
          "Return the primitive cell of an input Crystal.");

    m.def("spglib_version", &vibeqc::spglib_version,
          "Version of the linked spglib library.");

    py::class_<vibeqc::IrreducibleKMesh>(m, "IrreducibleKMesh")
        .def_readonly("fractional_kpoints",
                      &vibeqc::IrreducibleKMesh::fractional_kpoints)
        .def_readonly("weights",     &vibeqc::IrreducibleKMesh::weights)
        .def_readonly("ir_mapping",  &vibeqc::IrreducibleKMesh::ir_mapping);

    m.def("irreducible_kpoints", &vibeqc::irreducible_kpoints,
          py::arg("crystal"), py::arg("mesh"),
          py::arg("is_shift") = std::array<int, 3>{0, 0, 0},
          py::arg("symprec") = 1.0e-5,
          py::call_guard<py::gil_scoped_release>(),
          "Irreducible-Brillouin-zone k-point list for a Monkhorst–Pack mesh "
          "(fractional coordinates in the reciprocal lattice).");

    // ----- Periodic systems + lattice sums -------------------------------
    py::class_<vibeqc::PeriodicSystem>(m, "PeriodicSystem")
        .def(py::init<>())
        .def(py::init([](int dim,
                         const Eigen::Matrix3d& lattice,
                         const std::vector<vibeqc::Atom>& unit_cell,
                         int charge, int multiplicity) {
                 vibeqc::PeriodicSystem s;
                 s.dim = dim;
                 s.lattice = lattice;
                 s.unit_cell = unit_cell;
                 s.charge = charge;
                 s.multiplicity = multiplicity;
                 if (dim < 1 || dim > 3) {
                     throw std::runtime_error(
                         "PeriodicSystem: dim must be 1, 2, or 3");
                 }
                 return s;
             }),
             py::arg("dim"), py::arg("lattice"), py::arg("unit_cell"),
             py::arg("charge") = 0, py::arg("multiplicity") = 1,
             "Construct a PeriodicSystem. ``lattice`` COLUMNS are the "
             "Cartesian lattice vectors in bohr: ``lattice[:, i]`` is a_i. "
             "A row-oriented matrix is a different (sheared) crystal that "
             "runs and converges without error -- issue #445 -- so when "
             "building from three vectors prefer the keyword form "
             "``PeriodicSystem(dim, lattice_vectors=[a1, a2, a3], "
             "unit_cell=...)``, which never states a matrix orientation. "
             "The matrix is always 3×3 regardless of ``dim``; columns "
             "beyond ``dim`` are non-physical, synthesized to keep the "
             "lattice full-rank, and the energy is invariant to their "
             "length.")
        // Issue #445: the declared-orientation constructor. The caller
        // names the three vectors; the binding stacks them into COLUMNS, so
        // there is no matrix orientation for the caller to get wrong.
        .def(py::init([](int dim,
                         const std::vector<Eigen::Vector3d>& lattice_vectors,
                         const std::vector<vibeqc::Atom>& unit_cell,
                         int charge, int multiplicity) {
                 if (lattice_vectors.size() != 3) {
                     throw std::invalid_argument(
                         "PeriodicSystem: lattice_vectors must hold exactly "
                         "three Cartesian lattice vectors (a1, a2, a3) in "
                         "bohr; got " + std::to_string(lattice_vectors.size()));
                 }
                 Eigen::Matrix3d lattice;
                 for (int i = 0; i < 3; ++i) {
                     lattice.col(i) = lattice_vectors[static_cast<std::size_t>(i)];
                 }
                 vibeqc::PeriodicSystem s;
                 s.dim = dim;
                 s.lattice = lattice;
                 s.unit_cell = unit_cell;
                 s.charge = charge;
                 s.multiplicity = multiplicity;
                 if (dim < 1 || dim > 3) {
                     throw std::runtime_error(
                         "PeriodicSystem: dim must be 1, 2, or 3");
                 }
                 return s;
             }),
             py::arg("dim"), py::kw_only(), py::arg("lattice_vectors"),
             py::arg("unit_cell"),
             py::arg("charge") = 0, py::arg("multiplicity") = 1,
             "Construct a PeriodicSystem from its three lattice VECTORS "
             "``lattice_vectors=[a1, a2, a3]`` (each a 3-vector in bohr). "
             "This form is orientation-free: the vectors are stacked into "
             "the columns of ``lattice`` for you (issue #445).")
        .def_readwrite("dim",          &vibeqc::PeriodicSystem::dim)
        .def_readwrite("lattice",      &vibeqc::PeriodicSystem::lattice)
        .def_readwrite("unit_cell",    &vibeqc::PeriodicSystem::unit_cell)
        .def_readwrite("charge",       &vibeqc::PeriodicSystem::charge)
        .def_readwrite("multiplicity", &vibeqc::PeriodicSystem::multiplicity)
        .def_readwrite("symmetry",     &vibeqc::PeriodicSystem::symmetry)
        .def("reciprocal_lattice",
             &vibeqc::PeriodicSystem::reciprocal_lattice)
        .def("n_electrons",   &vibeqc::PeriodicSystem::n_electrons)
        .def("unit_cell_molecule",
             &vibeqc::PeriodicSystem::unit_cell_molecule)
        .def("__repr__", [](const vibeqc::PeriodicSystem& s) {
            return std::string{"PeriodicSystem(dim="}
                   + std::to_string(s.dim) + ", n_atoms="
                   + std::to_string(s.unit_cell.size()) + ")";
        });

    m.def("attach_symmetry", &vibeqc::attach_symmetry,
          py::arg("system"), py::arg("symprec") = 1.0e-5,
          py::call_guard<py::gil_scoped_release>(),
          "Populate system.symmetry with a spglib analysis of the unit cell.");

    py::class_<vibeqc::LatticeCell>(m, "LatticeCell")
        .def(py::init([](const vibeqc::PeriodicSystem& system,
                        const Eigen::Vector3i& index) {
            if (system.dim < 1 || system.dim > 3) {
                throw std::invalid_argument("LatticeCell: invalid periodic dimension");
            }
            for (int axis = system.dim; axis < 3; ++axis) {
                if (index[axis] != 0) {
                    throw std::invalid_argument(
                        "LatticeCell: nonperiodic index must be zero");
                }
            }
            vibeqc::LatticeCell cell;
            cell.index = index;
            cell.r_cart = system.lattice * index.cast<double>();
            if (!cell.r_cart.allFinite()) {
                throw std::invalid_argument(
                    "LatticeCell: Cartesian translation must be finite");
            }
            return cell;
        }), py::arg("system"), py::arg("index"),
        "Construct one exact integer cell without enumerating a sphere.")
        .def_readonly("index",  &vibeqc::LatticeCell::index)
        .def_readonly("r_cart", &vibeqc::LatticeCell::r_cart);

    py::enum_<vibeqc::CoulombMethod>(m, "CoulombMethod")
        .value("DIRECT_TRUNCATED", vibeqc::CoulombMethod::DIRECT_TRUNCATED)
        .value("EWALD_3D",         vibeqc::CoulombMethod::EWALD_3D)
        .value("SLAB_EWALD_2D",    vibeqc::CoulombMethod::SLAB_EWALD_2D)
        .value("NEUTRALIZED_1D",   vibeqc::CoulombMethod::NEUTRALIZED_1D);

    py::class_<vibeqc::LatticeSumOptions>(m, "LatticeSumOptions")
        .def(py::init<>())
        .def_readwrite("cutoff_bohr",
                       &vibeqc::LatticeSumOptions::cutoff_bohr)
        .def_readwrite("eri_interaction_cutoff_bohr",
                       &vibeqc::LatticeSumOptions::eri_interaction_cutoff_bohr,
                       "Physical AO-product midpoint separation cutoff in bohr. "
                       "Used with pair_complete_1e; zero uses cutoff_bohr.")
        .def_readwrite("nuclear_cutoff_bohr",
                       &vibeqc::LatticeSumOptions::nuclear_cutoff_bohr)
        .def_readwrite("becke_image_radius_bohr",
                       &vibeqc::LatticeSumOptions::becke_image_radius_bohr,
                       "Reach (bohr) of the periodic Becke partition of the "
                       "DFT grid the caller built. build_xc_periodic screens "
                       "the cross-cell (bra-image) density sum by this "
                       "radius, not cutoff_bohr. 0.0 (default) means unset "
                       "and falls back to the built-in partition reach. "
                       "Drivers set it to the grid's image_radius_bohr.")
        .def_readwrite("coulomb_method",
                       &vibeqc::LatticeSumOptions::coulomb_method)
        .def_readwrite("slab_ewald_alpha",
                       &vibeqc::LatticeSumOptions::slab_ewald_alpha,
                       "Gaussian split parameter shared by the 2D slab "
                       "Ewald E_nn, V_ne, and J blocks.")
        .def_readwrite("screening_overlap_threshold",
                       &vibeqc::LatticeSumOptions::screening_overlap_threshold)
        .def_readwrite("screening_exchange_threshold",
                       &vibeqc::LatticeSumOptions::screening_exchange_threshold)
        .def_readwrite("schwarz_threshold",
                       &vibeqc::LatticeSumOptions::schwarz_threshold,
                       "Schwarz screening threshold for the periodic 2-e "
                       "Fock build. Default 1e-10 Ha. Set to 0.0 to disable "
                       "(unscreened, O(n_c^3 * n_shells^4) — very slow).")
        .def_readwrite("schwarz_threshold_forces",
                       &vibeqc::LatticeSumOptions::schwarz_threshold_forces,
                       "Schwarz screening threshold for the periodic 2-e "
                       "Fock derivative pass (gradient ERIs). Default "
                       "1e-14 — 100× tighter than schwarz_threshold "
                       "because plain Schwarz on derivatives is non-rigorous "
                       "(integral bound is not a derivative bound). Set to "
                       "0.0 to disable.")
        .def_readwrite("sr_sparse_traversal",
                       &vibeqc::LatticeSumOptions::sr_sparse_traversal,
                       "Generate erfc cell-pair candidates before traversal. "
                       "Default True; False retains exhaustive enumeration "
                       "with identical screening for validation.")
        .def_readwrite("pair_complete_1e",
                       &vibeqc::LatticeSumOptions::pair_complete_1e,
                       "#429 stage 2: enumerate the one-electron lattice "
                       "sums (S, T, V and their gradient partners) over the "
                       "translation-invariant pair set "
                       "{(mu, nu, g): |O_mu - O_nu - g| <= cutoff_bohr} "
                       "instead of the |g| <= cutoff_bohr ball. Default "
                       "False (bit-identical to the historical sums); the "
                       "default flips per consumer family once measured.")
        .def_readwrite("sr_range_screening",
                       &vibeqc::LatticeSumOptions::sr_range_screening,
                       "Distance-aware charge-pair Schwarz and positive "
                       "Gaussian-product bounds for the erfc(w*r)/r "
                       "short-range direct J/K build. Includes angular "
                       "polynomials and absolute contraction coefficients. "
                       "Rejects density-weighted quartet bounds below "
                       "schwarz_threshold; this per-quartet threshold is "
                       "not a bound on the total lattice-sum error. Default "
                       "False retains ordinary Schwarz screening for "
                       "independent comparisons; only affects omega > 0.");

    py::class_<vibeqc::LatticeMatrixSet>(m, "LatticeMatrixSet")
        .def(py::init<>())
        .def_readonly("nbf",    &vibeqc::LatticeMatrixSet::nbf)
        .def_readonly("cells",  &vibeqc::LatticeMatrixSet::cells)
        .def_readonly("blocks", &vibeqc::LatticeMatrixSet::blocks,
                      "Per-cell ``(n_bf, n_bf)`` block matrices. "
                      "Note: ``.blocks`` returns a fresh Python list "
                      "each access (pybind11's default for "
                      "std::vector<MatrixXd>); element assignment to "
                      "the returned list does NOT propagate to C++. "
                      "Use ``set_block(i, M)`` to mutate the underlying "
                      "C++ vector in place.")
        .def("set_block",
             [](vibeqc::LatticeMatrixSet& s, int i,
                const Eigen::MatrixXd& M) {
                 if (i < 0 || static_cast<std::size_t>(i) >= s.blocks.size()) {
                     throw py::index_error(
                         "LatticeMatrixSet.set_block: index "
                         + std::to_string(i) + " out of range (size "
                         + std::to_string(s.blocks.size()) + ")");
                 }
                 if (static_cast<int>(M.rows()) != s.nbf ||
                     static_cast<int>(M.cols()) != s.nbf) {
                     throw py::value_error(
                         "LatticeMatrixSet.set_block: matrix shape "
                         "must equal (nbf, nbf)");
                 }
                 s.blocks[static_cast<std::size_t>(i)] = M;
             },
             py::arg("index"), py::arg("matrix"),
             "Replace the block at the given cell index with ``matrix``. "
             "Mutates the underlying C++ storage in place — unlike "
             "``self.blocks[i] = matrix`` which writes to a transient "
             "Python-list copy and silently no-ops at the C++ level.")
        .def("__len__", [](const vibeqc::LatticeMatrixSet& s) {
            return s.size();
        });

    m.def(
        "make_lattice_matrix_set",
        [](int nbf,
           const std::vector<vibeqc::LatticeCell>& cells,
           py::list py_blocks) -> vibeqc::LatticeMatrixSet {
            vibeqc::LatticeMatrixSet s;
            s.nbf = nbf;
            s.cells = cells;
            s.blocks.reserve(py_blocks.size());
            for (auto& item : py_blocks) {
                s.blocks.push_back(
                    py::cast<Eigen::MatrixXd>(item));
            }
            return s;
        },
        py::arg("nbf"), py::arg("cells"), py::arg("blocks"),
        "Construct a LatticeMatrixSet from Python lists. "
        "cells[i].r_cart must align with blocks[i].");

    m.def("direct_lattice_cells", &vibeqc::direct_lattice_cells,
          py::arg("system"), py::arg("cutoff_bohr"),
          py::call_guard<py::gil_scoped_release>(),
          "Enumerate integer lattice cells within a Cartesian cutoff.");

    m.def("pair_complete_lattice_cells",
          &vibeqc::pair_complete_lattice_cells,
          py::arg("basis"), py::arg("system"), py::arg("cutoff_bohr"),
          py::call_guard<py::gil_scoped_release>(),
          "Cell list the one-electron lattice builders enumerate under "
          "LatticeSumOptions.pair_complete_1e: the translations reaching any "
          "shell pair whose physical separation |O_mu - O_nu - g| is within "
          "cutoff_bohr, not those with |g| <= cutoff_bohr (which is not "
          "translation invariant). A superset of "
          "direct_lattice_cells(system, cutoff_bohr) whose leading entries "
          "are exactly that list, in the same order (#429).");

    m.def("physical_eri_lattice_cells", &vibeqc::physical_eri_lattice_cells,
          py::arg("basis"), py::arg("system"), py::arg("pair_cutoff_bohr"),
          py::arg("interaction_cutoff_bohr"),
          py::call_guard<py::gil_scoped_release>(),
          "Union of pair-centered lattice neighborhoods enclosing the physical "
          "ERI product support. Align operators by cell key, not list prefix.");

    m.def("compute_overlap_lattice", &vibeqc::compute_overlap_lattice,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed overlap integrals S_μν(g).");

    m.def("compute_kinetic_lattice", &vibeqc::compute_kinetic_lattice,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed kinetic-energy integrals T_μν(g).");

    m.def("compute_nuclear_lattice", &vibeqc::compute_nuclear_lattice,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed nuclear-attraction integrals V_μν(g).");

    m.def("compute_nuclear_lattice_with_charges",
          &vibeqc::compute_nuclear_lattice_with_charges,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("effective_charges"),
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed nuclear-attraction with Z_eff (ECP-aware).");

    m.def("compute_overlap_lattice_explicit",
          &vibeqc::compute_overlap_lattice_explicit,
          py::arg("basis"), py::arg("system"), py::arg("cells"),
          py::arg("pair_cutoff") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Overlap S(g) on caller-supplied cell list (Phase SYM3b). "
          "pair_cutoff bounds the physical separation |O_mu - O_nu - g| of "
          "each term, as the cutoff-driven builder does with "
          "pair_complete_1e; 0.0 computes every pair of every supplied cell.");

    m.def("compute_kinetic_lattice_explicit",
          &vibeqc::compute_kinetic_lattice_explicit,
          py::arg("basis"), py::arg("system"), py::arg("cells"),
          py::arg("pair_cutoff") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Kinetic T(g) on caller-supplied cell list (Phase SYM3b). "
          "pair_cutoff as for compute_overlap_lattice_explicit.");

    m.def("compute_nuclear_lattice_explicit",
          &vibeqc::compute_nuclear_lattice_explicit,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("cells"),
          py::call_guard<py::gil_scoped_release>(),
          "Nuclear V(g) on caller-supplied cell list (Phase SYM3b).");

    m.def("compute_nuclear_erfc_lattice",
          &vibeqc::compute_nuclear_erfc_lattice,
          py::arg("basis"), py::arg("system"),
          py::arg("omega"), py::arg("options"),
          py::call_guard<py::gil_scoped_release>(),
          "Short-range (erfc-screened) component of the nuclear-attraction "
          "lattice sum. Ewald building block: exponentially convergent real-"
          "space sum for any ω > 0. Combine with the reciprocal-space long-"
          "range part (Phase 12e-c) for the full 3D Ewald treatment of V(g).");

    // ----- Lattice multipole moments (BIPOLE Phase 1) ---------------------
    py::class_<vibeqc::LatticeMultipoleSet>(m, "LatticeMultipoleSet")
        .def_readonly("nbf",     &vibeqc::LatticeMultipoleSet::nbf)
        .def_readonly("L_max",   &vibeqc::LatticeMultipoleSet::L_max)
        .def_readonly("spherical", &vibeqc::LatticeMultipoleSet::spherical,
                      "True if output is spherical (sphemultipole for L=4), "
                      "False if Cartesian (emultipole for L<=3).")
        .def_readonly("cells",   &vibeqc::LatticeMultipoleSet::cells)
        .def_readonly("origin",  &vibeqc::LatticeMultipoleSet::origin)
        .def_readonly("blocks",  &vibeqc::LatticeMultipoleSet::blocks,
                      "Per-cell, per-component nbf×nbf moment matrix. "
                      "Layout: blocks[c][component] is the (nbf, nbf) "
                      "matrix of ⟨μ_0 | x^i y^j z^k | ν_c⟩ where (i, j, k) "
                      "encodes the Cartesian multipole order. Components "
                      "follow libint emultipole{1,2,3} ordering: "
                      "[S, μ_x, μ_y, μ_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz, ...].")
        .def("__len__", [](const vibeqc::LatticeMultipoleSet& s) {
            return s.cells.size();
        });

    m.def("cartesian_multipole_n_components",
          &vibeqc::cartesian_multipole_n_components,
          py::arg("L_max"),
          "Number of Cartesian-multipole components for moments up "
          "through L_max (libint convention): emultipole1→4, "
          "emultipole2→10, emultipole3→20.");

    m.def("compute_multipole_moments_lattice",
          &vibeqc::compute_multipole_moments_lattice,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("L_max") = 2,
          py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
          py::call_guard<py::gil_scoped_release>(),
          "Compute shell-pair multipole moments at every lattice cell "
          "out to options.cutoff_bohr. Foundation for the BIPOLE "
          "quartet prototype; the periodic expansion source is Pisani-"
          "Dovesi-Roetti (1988), Ch. II.4c. L_max ∈ {1, 2, 3} "
          "→ Cartesian via libint emultipole{1,2,3} (spherical=false); "
          "L_max = 4 → spherical harmonic moments via libint sphemultipole "
          "(spherical=true, (L_max+1)² = 25 components). The single "
          "expansion origin (typically (0,0,0)) is shared across all "
          "shell pairs; adjoined-Gaussian product centers are "
          "constructed via analytical polynomial-shift in Python "
          "(see vibeqc.bipole_pair_moments).");

    m.def("extend_lattice_moments_to_L4",
          &vibeqc::extend_lattice_moments_to_L4,
          py::arg("M3"), py::arg("basis"),
          py::call_guard<py::gil_scoped_release>(),
          "Extend L=3 Cartesian multipole moments to L=4 using an "
          "implementation-specific isotropic Gaussian model. "
          "Returns a new LatticeMultipoleSet with 35 components.");

    // ---- POLIPO -- native multipole engine (post-v0.15) ------------------
    py::class_<vibeqc::PolipoShellInfo>(m, "PolipoShellInfo",
        "Shell descriptor for the POLIPO native multipole engine.")
        .def(py::init<>())
        .def_readwrite("l", &vibeqc::PolipoShellInfo::l)
        .def_readwrite("pure", &vibeqc::PolipoShellInfo::pure)
        .def_readwrite("origin", &vibeqc::PolipoShellInfo::origin)
        .def_readwrite("exponents", &vibeqc::PolipoShellInfo::exponents)
        .def_readwrite("coeffs", &vibeqc::PolipoShellInfo::coeffs)
        .def_readwrite("bf_offset", &vibeqc::PolipoShellInfo::bf_offset)
        .def_readwrite("n_bf", &vibeqc::PolipoShellInfo::n_bf);

    py::class_<vibeqc::PolipoCellInfo>(m, "PolipoCellInfo",
        "Lattice cell descriptor for POLIPO.")
        .def(py::init<>())
        .def_readwrite("r_cart", &vibeqc::PolipoCellInfo::r_cart)
        .def_readwrite("index", &vibeqc::PolipoCellInfo::index);

    py::class_<vibeqc::PolipoOptions>(m, "PolipoOptions",
        "Computation options for POLIPO.")
        .def(py::init<>())
        .def_readwrite("cutoff_bohr", &vibeqc::PolipoOptions::cutoff_bohr)
        .def_readwrite("L_max", &vibeqc::PolipoOptions::L_max)
        .def_readwrite("spherical", &vibeqc::PolipoOptions::spherical)
        .def_readwrite("use_ryz", &vibeqc::PolipoOptions::use_ryz)
        .def_readwrite("num_threads", &vibeqc::PolipoOptions::num_threads);

    py::class_<vibeqc::PolipoMultipoleSet>(m, "PolipoMultipoleSet",
        "POLIPO multipole moment result set.")
        .def(py::init<>())
        .def_readonly("nbf", &vibeqc::PolipoMultipoleSet::nbf)
        .def_readonly("L_max", &vibeqc::PolipoMultipoleSet::L_max)
        .def_readonly("spherical", &vibeqc::PolipoMultipoleSet::spherical)
        .def_readonly("cells", &vibeqc::PolipoMultipoleSet::cells)
        .def_readonly("origin", &vibeqc::PolipoMultipoleSet::origin)
        .def_readonly("blocks", &vibeqc::PolipoMultipoleSet::blocks);

    m.def("compute_polipo_moments_lattice",
          &vibeqc::compute_polipo_moments_lattice,
          py::arg("shells"), py::arg("cells"), py::arg("options"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute shell-pair multipole moments using the POLIPO native "
          "engine (Hermite expansion).  Self-contained C++ path, no "
          "libint dependency.  Supports L_max up to 12.  Selectable via "
          "multipole_engine='polipo' runtime parameter.");

    m.def("extend_lattice_moments",
          &vibeqc::extend_lattice_moments,
          py::arg("M3"), py::arg("basis"), py::arg("L_target"),
          py::call_guard<py::gil_scoped_release>(),
          "Extend L=3 Cartesian multipole moments to arbitrary L_target "
          "(4, 5, or 6) using the isotropic s-Gaussian analytical formulas. "
          "Odd orders (L=5) are identically zero.  Returns a new "
          "LatticeMultipoleSet with L_max = L_target.");

    m.def("compute_ext_el_spheropole_gradient_lattice",
          &vibeqc::compute_ext_el_spheropole_gradient_lattice,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("P_real"), py::arg("n_electrons"),
          py::call_guard<py::gil_scoped_release>(),
          "Disabled native atomic-position gradient of the EXT "
          "EL-SPHEROPOLE energy. The emultipole2 deriv_order=1 path is "
          "currently process-unsafe with the vendored libint build; use "
          "vibeqc.bipole_gradient, which central-differences this energy "
          "term at fixed density while the native derivative port is "
          "repaired.");

    // (The spheropole kernels — compute_spheropole_{bare_integrals,
    // kernel_lattice,kernel_grad_lattice} — were removed 2026-06-01: the
    // EXT EL-SPHEROPOLE energy now uses the exact emultipole2
    // bond-symmetrised second moment, via compute_multipole_moments_lattice
    // in vibeqc.bipole_ext_el_pole.compute_ext_el_spheropole.)

    // ----- Bloch sums & k-mesh -------------------------------------------
    py::class_<vibeqc::BlochKMesh>(m, "BlochKMesh")
        .def_readonly("mesh",       &vibeqc::BlochKMesh::mesh)
        .def_readonly("is_shift",   &vibeqc::BlochKMesh::is_shift)
        .def_readonly("kpoints",    &vibeqc::BlochKMesh::kpoints)
        .def_readonly("weights",    &vibeqc::BlochKMesh::weights)
        .def_readonly("ir_mapping", &vibeqc::BlochKMesh::ir_mapping)
        .def("__len__", [](const vibeqc::BlochKMesh& m) { return m.size(); });

    m.def("level_shift_at_iter",
          [](double level_shift, int level_shift_warmup_cycles,
             const std::vector<double>& level_shift_schedule,
             int max_iter, int iter) {
              return vibeqc::level_shift_at_iter(
                  level_shift, level_shift_warmup_cycles,
                  level_shift_schedule, max_iter, iter);
          },
          py::arg("level_shift"),
          py::arg("level_shift_warmup_cycles"),
          py::arg("level_shift_schedule"),
          py::arg("max_iter"),
          py::arg("iter"),
          "Resolve the Saunders-Hillier level shift for one SCF "
          "iteration (1-indexed). Single source of truth shared by the "
          "molecular C++ drivers and the periodic Python drivers: an "
          "explicit level_shift_schedule takes precedence, otherwise the "
          "shift is derived from level_shift + level_shift_warmup_cycles "
          "(-1 auto / 0 persistent / N explicit warm-up).");

    py::enum_<vibeqc::LevelShiftDensity>(m, "LevelShiftDensity",
          "Which density is being handed to apply_level_shift, which fixes "
          "the weight w in F + b*(S - w*S*D*S). SPIN: a single-spin density "
          "D_s with occupations in {0, 1}; it is idempotent in the overlap "
          "metric, so S*D_s*S is the occupied-space projector and w = 1. "
          "TOTAL: a closed-shell total density D = 2P with occupations in "
          "{0, 2}, so w = 1/2 recovers the same projector. Either way the "
          "occupied eigenvalues do not move and every virtual rises by "
          "exactly b. This is not a tuning knob.")
        .value("SPIN", vibeqc::LevelShiftDensity::SPIN)
        .value("TOTAL", vibeqc::LevelShiftDensity::TOTAL);

    m.def("apply_level_shift", &vibeqc::apply_level_shift,
          py::arg("fock"), py::arg("overlap"), py::arg("density"),
          py::arg("b"), py::arg("convention"),
          "Saunders-Hillier shift operator, real (molecular or Γ-point): "
          "F + b*(S - w*S*D*S), symmetrized, with w set by ``convention``. "
          "Returns ``fock`` untouched when b == 0, so the S*D*S products "
          "are skipped once the warm-up releases. Single source of truth "
          "for the weight; the C++ SCF drivers and the periodic Python "
          "drivers both call it rather than open-coding the expression.");

    m.def("apply_level_shift_k", &vibeqc::apply_level_shift_k,
          py::arg("fock_k"), py::arg("overlap_k"), py::arg("density_k"),
          py::arg("b"), py::arg("convention"),
          "Per-k Hermitian analogue of apply_level_shift: "
          "F(k) + b*(S(k) - w*S(k)*P(k)*S(k)), Hermitized.");

    m.def("monkhorst_pack", &vibeqc::monkhorst_pack,
          py::arg("system"), py::arg("mesh"),
          py::arg("is_shift") = std::array<int, 3>{0, 0, 0},
          py::arg("use_symmetry") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Monkhorst–Pack k-point mesh, optionally reduced to the IBZ.");

    // K4 — factory for building a BlochKMesh from an explicit Cartesian
    // k-point list + matching weights. Used by KPoints.to_bloch_kmesh()
    // when kind ∈ {"explicit", "band-path"} so user-supplied k-lists
    // and HPKOT band paths can feed any periodic SCF driver that
    // expects a BlochKMesh.
    m.def("bloch_kmesh_from_lists",
          [](const std::vector<Eigen::Vector3d>& kpoints_cart,
             const std::vector<double>& weights) {
              if (kpoints_cart.size() != weights.size()) {
                  throw std::runtime_error(
                      "bloch_kmesh_from_lists: kpoints_cart and weights "
                      "must have the same length.");
              }
              vibeqc::BlochKMesh bm;
              bm.mesh = {1, 1, 1};
              bm.is_shift = {0, 0, 0};
              bm.kpoints = kpoints_cart;
              bm.weights = weights;
              bm.ir_mapping.clear();
              return bm;
          },
          py::arg("kpoints_cart"), py::arg("weights"),
          "Build a BlochKMesh from explicit (Cartesian k-vector, weight) "
          "lists. mesh / is_shift / ir_mapping are left as 1×1×1 / 0 / "
          "empty since those are MP-specific concepts that don't apply "
          "to explicit lists or band paths.");

    // ----- Regular k-mesh addressing (diagnostic surface) ----------------
    //
    // cpp/include/vibeqc/kmesh_address.hpp is a C++ layer: it exists for the
    // periodic correlated methods, which call it from C++, and nothing in
    // python/vibeqc/ needs it. The only reason it has a Python face at all is
    // that this repository's change-safety inventory is pytest
    // (tests/test_kmesh_address.py), so an exact-integer module with no
    // Python caller would otherwise ship untested.
    //
    // Accordingly this is a DIAGNOSTIC surface, not an API: underscore
    // prefixed, deliberately NOT re-exported from python/vibeqc/__init__.py,
    // no stability guarantee, and no user-facing wrapper. It hands back plain
    // integer triples rather than address objects so that no addressing logic
    // is duplicated on the Python side -- the algebra stays in C++, and the
    // KGridAddress / KTransferAddress separation stays a compile-time
    // property (static_assert in kmesh_address.cpp) with its runtime half
    // visible here as the parity checks in ``index`` and
    // ``index_of_transfer``.
    py::class_<vibeqc::RegularKMesh> regular_kmesh(m, "_RegularKMesh");
    regular_kmesh
        .def(py::init<std::array<int, 3>, std::array<int, 3>>(),
             py::arg("mesh"),
             py::arg("is_shift") = std::array<int, 3>{0, 0, 0})
        .def_property_readonly("mesh", &vibeqc::RegularKMesh::mesh)
        .def_property_readonly("is_shift", &vibeqc::RegularKMesh::is_shift)
        .def_property_readonly("doubled_modulus",
                               &vibeqc::RegularKMesh::doubled_modulus)
        .def("__len__",
             [](const vibeqc::RegularKMesh& self) { return self.size(); })
        .def("address",
             [](const vibeqc::RegularKMesh& self, std::size_t index) {
                 return self.address(index).doubled;
             },
             py::arg("index"))
        .def("index",
             [](const vibeqc::RegularKMesh& self, std::array<int, 3> doubled) {
                 return self.index(vibeqc::KGridAddress{doubled});
             },
             py::arg("doubled"))
        .def("transfer_address",
             [](const vibeqc::RegularKMesh& self, std::size_t index) {
                 return self.transfer_address(index).doubled;
             },
             py::arg("index"))
        .def("index_of_transfer",
             [](const vibeqc::RegularKMesh& self, std::array<int, 3> doubled) {
                 return self.transfer_index(vibeqc::KTransferAddress{doubled});
             },
             py::arg("doubled"))
        .def("fractional",
             [](const vibeqc::RegularKMesh& self, std::array<int, 3> doubled) {
                 return self.fractional(vibeqc::KGridAddress{doubled});
             },
             py::arg("doubled"))
        .def("transfer_fractional",
             [](const vibeqc::RegularKMesh& self, std::array<int, 3> doubled) {
                 return self.fractional(vibeqc::KTransferAddress{doubled});
             },
             py::arg("doubled"))
        .def("fractional_at", &vibeqc::RegularKMesh::fractional_at,
             py::arg("index"))
        .def("negate",
             [](const vibeqc::RegularKMesh& self, std::array<int, 3> doubled) {
                 const auto out =
                     self.negate_with_wrap(vibeqc::KGridAddress{doubled});
                 return py::make_tuple(out.address.doubled, out.wrap);
             },
             py::arg("doubled"),
             "(-k reduced, wrap G) with -k_frac = reduced_frac + G.")
        .def("transfer",
             [](const vibeqc::RegularKMesh& self,
                std::array<int, 3> ki, std::array<int, 3> kj) {
                 const auto out = self.transfer_with_wrap(
                     vibeqc::KGridAddress{ki}, vibeqc::KGridAddress{kj});
                 return py::make_tuple(out.address.doubled, out.wrap);
             },
             py::arg("ki"), py::arg("kj"),
             "(q = k_j - k_i reduced, wrap G) with "
             "k_j - k_i = q_reduced + G.")
        .def("add_transfer",
             [](const vibeqc::RegularKMesh& self,
                std::array<int, 3> ki, std::array<int, 3> q) {
                 const auto out = self.add_transfer_with_wrap(
                     vibeqc::KGridAddress{ki},
                     vibeqc::KTransferAddress{q});
                 return py::make_tuple(out.address.doubled, out.wrap);
             },
             py::arg("ki"), py::arg("q"),
             "(k_j = k_i + q reduced, wrap G) with "
             "k_i + q = k_j + G.")
        .def("conserved",
             [](const vibeqc::RegularKMesh& self,
                std::array<int, 3> ki, std::array<int, 3> ka,
                std::array<int, 3> kj) {
                 const auto out = self.conserved_with_wrap(
                     vibeqc::KGridAddress{ki}, vibeqc::KGridAddress{ka},
                     vibeqc::KGridAddress{kj});
                 return py::make_tuple(out.address.doubled, out.wrap);
             },
             py::arg("ki"), py::arg("ka"), py::arg("kj"),
             "(k_b reduced, wrap G) with k_i - k_a + k_j = k_b + G.")
        .def("negate_index", &vibeqc::RegularKMesh::negate_index,
             py::arg("index"))
        .def("transfer_index",
             [](const vibeqc::RegularKMesh& self,
                std::size_t ki, std::size_t kj) {
                 return self.transfer_index(ki, kj);
             },
             py::arg("ki"), py::arg("kj"))
        .def("add_transfer_index", &vibeqc::RegularKMesh::add_transfer_index,
             py::arg("ki"), py::arg("q"),
             "O(1) lookup of k_j for q = k_j - k_i.")
        .def("conserved_index", &vibeqc::RegularKMesh::conserved_index,
             py::arg("ki"), py::arg("ka"), py::arg("kj"),
             "O(1) lookup of the conserved fourth index. There is "
             "deliberately no table builder anywhere in this class: at the "
             "project reference mesh (8,8,8) an N_k^3 table is 512^3 "
             "entries, over a gigabyte, for a number three integer "
             "divisions produce on the spot.");
    regular_kmesh.attr("max_divisions") = vibeqc::RegularKMesh::max_divisions;

    m.def("bloch_sum", &vibeqc::bloch_sum,
          py::arg("real_space"), py::arg("k_cart"),
          py::call_guard<py::gil_scoped_release>(),
          "Bloch sum M(k) = Σ_g exp(i k·g) M(g) for a single k-point.");

    m.def("bloch_sum_multi_k", &vibeqc::bloch_sum_multi_k,
          py::arg("blocks"), py::arg("cell_r"), py::arg("k_vectors"),
          py::call_guard<py::gil_scoped_release>(),
          "Bloch sum over all k-points at once. "
          "F(k) = Σ_g exp(+i k·R_g) · block[g]. "
          "OpenMP-parallel over the k-loop.");

    m.def("assemble_fock_multi_k", &vibeqc::assemble_fock_multi_k,
          py::arg("f2e_blocks"), py::arg("cell_r"),
          py::arg("k_vectors"), py::arg("hcore_k"),
          py::call_guard<py::gil_scoped_release>(),
          "Per-k Fock assembly: Bloch sum + Hcore addition. "
          "F(k) = Σ_g exp(+i k·R_g) · F_2e(g) + H_core(k).");

    m.def("inverse_bloch_multi_k", &vibeqc::inverse_bloch_multi_k,
          py::arg("blocks"), py::arg("cell_r"), py::arg("k_vectors"),
          py::call_guard<py::gil_scoped_release>(),
          "Inverse Bloch transform over all k-points at once. "
          "D(k) = ½[Σ_g exp(+i k·R_g) D(g) + h.c.].");

    py::class_<vibeqc::BandDiag>(m, "BandDiag")
        .def_readonly("energies",     &vibeqc::BandDiag::energies)
        .def_readonly("coefficients", &vibeqc::BandDiag::coefficients);

    m.def("diagonalize_bloch", &vibeqc::diagonalize_bloch,
          py::arg("F_k"), py::arg("S_k"),
          py::arg("lindep_threshold") = 1.0e-9,
          py::call_guard<py::gil_scoped_release>(),
          "Solve F·C = S·C·diag(ε) at a single k-point by symmetric "
          "orthogonalisation.");

    // ----- COOP/COHP bonding-analysis kernels -------------------------
    m.def(
        "coop_weights_k",
        [](const vibeqc::ComplexMatrix& C_k,
           const vibeqc::ComplexMatrix& S_k,
           const std::vector<std::pair<std::vector<int>, std::vector<int>>>&
               pairs_ao) -> Eigen::MatrixXd {
            return vibeqc::coop_weights_k(C_k, S_k, pairs_ao);
        },
        py::arg("C_k"), py::arg("S_k"), py::arg("pairs_ao"),
        py::call_guard<py::gil_scoped_release>(),
        "Compute COOP weights for all atom-pairs at one k-point.\n"
        "w_{AB,n} = Re[ Σ_{μ∈A,ν∈B}  C*_{μn} · S_{μν} · C_{νn} ].\n"
        "Returns real (n_pairs, n_bands).");

    m.def(
        "cohp_weights_k",
        [](const vibeqc::ComplexMatrix& C_k,
           const vibeqc::ComplexMatrix& H_k,
           const std::vector<std::pair<std::vector<int>, std::vector<int>>>&
               pairs_ao) -> Eigen::MatrixXd {
            return vibeqc::cohp_weights_k(C_k, H_k, pairs_ao);
        },
        py::arg("C_k"), py::arg("H_k"), py::arg("pairs_ao"),
        py::call_guard<py::gil_scoped_release>(),
        "Compute COHP weights for all atom-pairs at one k-point.\n"
        "w_{AB,n} = Re[ Σ_{μ∈A,ν∈B}  C*_{μn} · H_{μν} · C_{νn} ].\n"
        "Returns real (n_pairs, n_bands).");

    m.def(
        "gaussian_broaden_projected",
        &vibeqc::gaussian_broaden_projected,
        py::arg("energies_per_k"), py::arg("weights_per_k"),
        py::arg("k_weights"), py::arg("energy_grid"), py::arg("sigma"),
        py::call_guard<py::gil_scoped_release>(),
        "Gaussian-broaden per-pair orbital weights onto an energy grid.\n"
        "Returns (n_pairs, n_e) real projection.");

    // ----- JKBuilder (J / K Fock-build strategy) -------------------------
    // Polymorphic interface used by the molecular SCF drivers and (via
    // the periodic factory below) the periodic Γ-only adapter. Concrete
    // factories return std::unique_ptr<JKBuilder>; pybind11 holds them
    // as ``std::shared_ptr`` for Python-side ownership convenience. The
    // ``PyJKBuilder`` trampoline (v0.12 R3) makes the abstract class
    // sub-classable from Python so the GPW Γ-only Hartree-J builder
    // (``GpwJBuilder``) can be plugged into the C++ SCF host
    // ``run_rhf_scf_with_jk`` without a dedicated C++ implementation —
    // unlocks DIIS / dynamic damping / level-shift / Newton for the
    // GPW route from C++.
    py::class_<vibeqc::JKBuilder, PyJKBuilder,
               std::shared_ptr<vibeqc::JKBuilder>>(
        m, "JKBuilder")
        .def(py::init<>())
        .def("build_J",
             &vibeqc::JKBuilder::build_J,
             py::arg("density"),
             "Coulomb matrix J(D)_{μν} = Σ_{λρ} D_{λρ} (μν|λρ).")
        .def_property_readonly(
             "last_j_workers_used",
             &vibeqc::JKBuilder::last_j_workers_used,
             "Actual OpenMP team used by the most recent J build.")
        .def_property_readonly(
             "last_df_transform_workers_used",
             &vibeqc::JKBuilder::last_df_transform_workers_used,
             "Actual OpenMP team used by the DF B-tensor transform.")
        .def_property_readonly(
             "last_df_pack_workers_used",
             &vibeqc::JKBuilder::last_df_pack_workers_used,
             "Actual OpenMP team used to pack the raw DF tensor.")
        .def_property_readonly(
             "last_df_unpack_workers_used",
             &vibeqc::JKBuilder::last_df_unpack_workers_used,
             "Actual OpenMP team used to unpack the transformed DF tensor.")
        .def("build_K",
             &vibeqc::JKBuilder::build_K,
             py::arg("density"),
             "Exchange matrix K(D)_{μν} = Σ_{λρ} D_{λρ} (μλ|νρ).")
        .def("build_J_slot",
             &vibeqc::JKBuilder::build_J_slot,
             py::arg("density"), py::arg("slot"),
             "Build J(D) through a named incremental-cache slot. Stateless "
             "builders ignore the slot.")
        .def("build_K_slot",
             &vibeqc::JKBuilder::build_K_slot,
             py::arg("density"), py::arg("slot"),
             "Build K(D) through a named incremental-cache slot. Stateless "
             "builders ignore the slot.")
        .def_property_readonly(
             "last_k_workers_used",
             &vibeqc::JKBuilder::last_k_workers_used,
             "Actual OpenMP team used by the latest single-density K build.")
        .def("build_K_pair",
             &vibeqc::JKBuilder::build_K_pair,
             py::arg("density_alpha"), py::arg("density_beta"),
             "Build independent alpha/beta exchange matrices together.")
        .def_property_readonly(
             "last_k_pair_workers_used",
             &vibeqc::JKBuilder::last_k_pair_workers_used,
             "Largest actual OpenMP team used by the latest paired K build.")
        .def("build_K_erf",
             &vibeqc::JKBuilder::build_K_erf,
             py::arg("density"), py::arg("omega"),
             "Long-range exchange matrix K_erf(D; omega) with "
             "erf(omega*r12)/r12. Implemented by the direct builder.")
        .def("build_g_rhf",
             &vibeqc::JKBuilder::build_g_rhf,
             py::arg("density"), py::arg("alpha_hf") = 1.0,
             "Closed-shell Fock 2-electron piece G(D) = J − ½·α_HF·K. "
             "Implementations may fuse the J and K kernels and override "
             "this; the default falls back to build_J + build_K. COSX "
             "one-center corrections are deliberately post-SCF and are "
             "not part of this iterated contribution.")
        .def("reset_state",
             &vibeqc::JKBuilder::reset_state,
             "Discard cross-iteration builder caches.")
        .def("set_schwarz_threshold",
             &vibeqc::JKBuilder::set_schwarz_threshold,
             py::arg("threshold"),
             "Update the live Schwarz screening threshold when supported.")
        .def("uses_runtime_schwarz_threshold",
             &vibeqc::JKBuilder::uses_runtime_schwarz_threshold,
             "Whether the builder has a live runtime Schwarz threshold.")
        .def("set_incremental",
             &vibeqc::JKBuilder::set_incremental,
             py::arg("enabled"),
             "Enable or disable the incremental density-difference cache.")
        .def("incremental_active",
             &vibeqc::JKBuilder::incremental_active,
             "Return whether incremental density-difference builds are active.");

    m.def("make_four_index_jk_builder",
          [](const vibeqc::BasisSet& basis) {
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_four_index_jk_builder(basis));
          },
          py::arg("basis"),
          "Direct four-index ERI JKBuilder. Materialises the full "
          "(μν|λρ) tensor at construction; reuses it across SCF "
          "iterations.");

    m.def("make_direct_jk_builder",
          [](const vibeqc::BasisSet& basis, double schwarz_threshold,
             bool incremental, int reset_freq) {
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_direct_jk_builder(
                      basis, schwarz_threshold, incremental, reset_freq));
          },
          py::arg("basis"), py::arg("schwarz_threshold") = 1e-10,
          py::arg("incremental") = false, py::arg("reset_freq") = 8,
          "Direct (integral-driven) JKBuilder. Caches the Schwarz Q "
          "matrix at construction; per SCF iter, walks an 8-fold-"
          "symmetric shell-quartet loop and evaluates only the "
          "surviving quartets via libint. O(n_shells² + n_bf²) "
          "memory instead of O(n_bf⁴) — closes the ~50-atom / "
          "def2-SVP OOM wall.");

    m.def("make_df_jk_builder",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::BasisSet& aux) {
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_df_jk_builder(basis, aux));
          },
          py::arg("basis"), py::arg("aux"),
          "Density-fitting (RIJK) JKBuilder. Owns a DensityFitting "
          "instance with the V-metric Cholesky and B-tensor.");

    m.def("make_cosx_jk_builder",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::BasisSet& aux,
             const vibeqc::Grid& cosx_grid) {
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_cosx_jk_builder(basis, aux, cosx_grid));
          },
          py::arg("basis"), py::arg("aux"), py::arg("cosx_grid"),
          "RIJCOSX JKBuilder. RI-J for the Coulomb piece, "
          "seminumerical chain-of-spheres for the K piece on the "
          "supplied grid.");

    m.def("make_periodic_gamma_jk_builder",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts,
             double omega) {
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_periodic_gamma_jk_builder(
                      basis, system, opts, omega));
          },
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("omega") = 0.0,
          "Γ-only molecular-limit periodic JKBuilder. Wraps "
          "build_jk_gamma_molecular_limit so the molecular SCF "
          "loop body can drive a periodic-Γ calculation through "
          "the same JKBuilder interface.");

    m.def("make_periodic_gamma_cosx_jk_builder",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts,
             const vibeqc::GridOptions& cosx_grid_opts) {
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_periodic_gamma_cosx_jk_builder(
                      basis, aux, system, opts, cosx_grid_opts));
          },
          py::arg("basis"), py::arg("aux"), py::arg("system"),
          py::arg("options"),
          py::arg("cosx_grid_options") = vibeqc::default_cosx_grid_options(),
          "Periodic Γ-only RIJCOSX JKBuilder. RI-J from density "
          "fitting (molecular DensityFitting, correct for Γ-folded "
          "density), seminumerical COSX-K on a periodic Becke grid "
          "with exchange summed over the truncated image-cell list "
          "(direct_lattice_cells at options.cutoff_bohr). For vacuum-"
          "padded molecular-limit cells the cell list collapses to "
          "the home cell — zero overhead. Note the molecular DF-J is "
          "only valid in the molecular limit; use "
          "make_periodic_tight_cosx_jk_builder for tight crystals.");

    m.def("make_periodic_tight_cosx_jk_builder",
          [](const vibeqc::BasisSet& basis,
             py::array_t<double, py::array::c_style | py::array::forcecast> lpq_array,
             const vibeqc::PeriodicSystem& system,
             const vibeqc::LatticeSumOptions& opts,
             const vibeqc::GridOptions& cosx_grid_opts) {
              // Convert 3D numpy array (n_kept, n_orb, n_orb) into
              // vector<Eigen::MatrixXd> for the C++ builder.
              py::buffer_info buf = lpq_array.request();
              if (buf.ndim != 3) {
                  throw std::runtime_error(
                      "make_periodic_tight_cosx_jk_builder: lpq must be "
                      "a 3D array (n_kept, n_orb, n_orb), got ndim=" +
                      std::to_string(buf.ndim));
              }
              const auto n_kept = static_cast<std::size_t>(buf.shape[0]);
              const auto n_orb  = static_cast<std::size_t>(buf.shape[1]);
              const auto n_col  = static_cast<std::size_t>(buf.shape[2]);
              if (n_orb != n_col) {
                  throw std::runtime_error(
                      "make_periodic_tight_cosx_jk_builder: lpq last two "
                      "dims must be equal (square), got (" +
                      std::to_string(n_orb) + ", " +
                      std::to_string(n_col) + ")");
              }
              const double* data = static_cast<const double*>(buf.ptr);
              const Eigen::Index stride = static_cast<Eigen::Index>(n_orb * n_orb);
              std::vector<Eigen::MatrixXd> lpq;
              lpq.reserve(n_kept);
              for (std::size_t p = 0; p < n_kept; ++p) {
                  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>
                      slice(data + p * stride,
                            static_cast<Eigen::Index>(n_orb),
                            static_cast<Eigen::Index>(n_orb));
                  lpq.push_back(slice);
              }
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_periodic_tight_cosx_jk_builder(
                      basis, lpq, system, opts, cosx_grid_opts));
          },
          py::arg("basis"), py::arg("lpq"), py::arg("system"),
          py::arg("options"),
          py::arg("cosx_grid_options") = vibeqc::default_cosx_grid_options(),
          "Periodic Γ-only RIJCOSX JKBuilder for tight cells. "
          "Periodic-correct GDF Coulomb J from a precomputed Lpq tensor "
          "(compcell or RSGDF factorization, built once in Python via "
          "``build_lpq_compcell`` or ``build_lpq_native_fft``). "
          "Seminumerical COSX-K on a periodic Becke grid with exchange "
          "summed over the truncated image-cell list "
          "(direct_lattice_cells at options.cutoff_bohr) — the M3a "
          "single-lattice-sum kernel (home-bra → image-ket exchange "
          "class; Γ-folded GDF exchange parity is still open, see "
          "handovers/HANDOVER_RIJCOSX_M3A.md). No exxdiv='ewald' Madelung "
          "K-shift (unnecessary — COSX-K is a real-space direct-"
          "truncated sum).");


    // ---- CCM-weighted periodic-Γ JKBuilder --------------------------------
    m.def("make_periodic_gamma_ccm_jk_builder",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::PeriodicSystem& system,
             py::array_t<int, py::array::c_style> weight_cells_np,
             py::array_t<double, py::array::c_style> weight_matrices_np,
             const vibeqc::LatticeSumOptions& opts,
             const std::string& method,
             double omega) {
              auto wc_buf = weight_cells_np.unchecked<2>();
              if (wc_buf.ndim() != 2 || wc_buf.shape(1) != 3)
                  throw std::runtime_error(
                      "make_periodic_gamma_ccm_jk_builder: weight_cells must be (n, 3)");
              std::vector<Eigen::Vector3i> weight_cells(wc_buf.shape(0));
              for (py::ssize_t i = 0; i < wc_buf.shape(0); ++i)
                  weight_cells[i] = Eigen::Vector3i(
                      wc_buf(i, 0), wc_buf(i, 1), wc_buf(i, 2));
              auto wm_buf = weight_matrices_np.unchecked<3>();
              if (wm_buf.ndim() != 3)
                  throw std::runtime_error(
                      "make_periodic_gamma_ccm_jk_builder: weight_matrices must be 3D");
              std::vector<Eigen::MatrixXd> weight_matrices(wm_buf.shape(0));
              const int n_atoms = static_cast<int>(wm_buf.shape(1));
              for (py::ssize_t i = 0; i < wm_buf.shape(0); ++i) {
                  weight_matrices[i] = Eigen::MatrixXd(n_atoms, n_atoms);
                  for (py::ssize_t a = 0; a < n_atoms; ++a)
                      for (py::ssize_t b = 0; b < n_atoms; ++b)
                          weight_matrices[i](a, b) = wm_buf(i, a, b);
              }
              return std::shared_ptr<vibeqc::JKBuilder>(
                  vibeqc::make_periodic_gamma_ccm_jk_builder(
                      basis, system, opts, weight_cells, weight_matrices, method, omega));
          },
          py::arg("basis"), py::arg("system"),
          py::arg("weight_cells").noconvert(),
          py::arg("weight_matrices").noconvert(),
          py::arg("options"),
          py::arg("method") = "bra_home",
          py::arg("omega") = 0.0,
          "CCM WSSC-weighted periodic-Î JKBuilder. Wraps "
          "build_jk_ccm_weighted in the JKBuilder interface so "
          "run_rhf_scf_with_jk can drive a CCM HF calculation. "
          "weight_cells: (n_cells, 3) int array. "
          "weight_matrices: (n_cells, n_atoms, n_atoms) float array.");

    // ----- Γ-only periodic RHF -------------------------------------------
    py::class_<vibeqc::JKMatrices>(m, "JKMatrices")
        .def_readonly("J", &vibeqc::JKMatrices::J)
        .def_readonly("K", &vibeqc::JKMatrices::K);

    m.def("build_jk_gamma_molecular_limit",
          &vibeqc::build_jk_gamma_molecular_limit,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Direct-SCF J and K matrices at \u0393 for a \u0393-only molecular-limit "
          "density. Set ``omega > 0`` to use the erfc-screened Coulomb "
          "kernel (short-range Ewald split); default ``omega=0`` uses "
          "the full 1/r_12 kernel.");

    m.def("build_jk_gamma_molecular_limit_explicit",
          &vibeqc::build_jk_gamma_molecular_limit_explicit,
          py::arg("basis"), py::arg("system"), py::arg("cells"),
          py::arg("options"), py::arg("density"), py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Same as build_jk_gamma_molecular_limit, but with caller-supplied "
          "cell list (Phase SYM3b — symmetry-reduced Fock build).");

    // ----- Phase M3b — per-cell-pair J/K contributions -------------------
    py::class_<vibeqc::PairJKContribution>(m, "PairJKContribution")
        .def_readonly("c_g", &vibeqc::PairJKContribution::c_g)
        .def_readonly("c_p", &vibeqc::PairJKContribution::c_p)
        .def_readonly("J_contrib", &vibeqc::PairJKContribution::J_contrib)
        .def_readonly("K_contrib", &vibeqc::PairJKContribution::K_contrib);

    m.def("build_jk_pair_contributions",
          &vibeqc::build_jk_pair_contributions,
          py::arg("basis"), py::arg("system"), py::arg("cells"),
          py::arg("pairs"), py::arg("options"), py::arg("density"),
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Per-pair J/K contributions for symmetry-reduced Fock build (M3b). "
          "Runs the same Γ-only molecular-limit kernel as "
          "build_jk_gamma_molecular_limit_explicit but stores each (c_g, c_p) "
          "cell pair's J/K block separately instead of summing. Summing over "
          "the full n_c × n_c pair set reproduces the explicit kernel's J/K.");

    // ---- CCM (Cyclic Cluster Model) WSSC-weighted J/K build --------------
    m.def("build_jk_ccm_weighted",
          [](const vibeqc::BasisSet& basis,
             const vibeqc::PeriodicSystem& system,
             py::array_t<int, py::array::c_style> weight_cells_np,
             py::array_t<double, py::array::c_style> weight_matrices_np,
             const vibeqc::LatticeSumOptions& opts,
             const Eigen::MatrixXd& P,
             const std::string& method,
             double omega) {
              // Convert weight_cells: shape (n_wcells, 3) -> vector<Vector3i>
              auto wc_buf = weight_cells_np.unchecked<2>();
              if (wc_buf.ndim() != 2 || wc_buf.shape(1) != 3)
                  throw std::runtime_error(
                      "build_jk_ccm_weighted: weight_cells must be (n, 3)");
              std::vector<Eigen::Vector3i> weight_cells(wc_buf.shape(0));
              for (py::ssize_t i = 0; i < wc_buf.shape(0); ++i)
                  weight_cells[i] = Eigen::Vector3i(
                      wc_buf(i, 0), wc_buf(i, 1), wc_buf(i, 2));
              // Convert weight_matrices: shape (n_wcells, n_atoms, n_atoms)
              auto wm_buf = weight_matrices_np.unchecked<3>();
              if (wm_buf.ndim() != 3)
                  throw std::runtime_error(
                      "build_jk_ccm_weighted: weight_matrices must be 3D");
              if (wm_buf.shape(0) != wc_buf.shape(0))
                  throw std::runtime_error(
                      "build_jk_ccm_weighted: n_wcells mismatch");
              std::vector<Eigen::MatrixXd> weight_matrices(wm_buf.shape(0));
              const int n_atoms = static_cast<int>(wm_buf.shape(1));
              for (py::ssize_t i = 0; i < wm_buf.shape(0); ++i) {
                  weight_matrices[i] = Eigen::MatrixXd(n_atoms, n_atoms);
                  for (py::ssize_t a = 0; a < n_atoms; ++a)
                      for (py::ssize_t b = 0; b < n_atoms; ++b)
                          weight_matrices[i](a, b) = wm_buf(i, a, b);
              }
              const auto cells = vibeqc::direct_lattice_cells(
                  system, opts.cutoff_bohr);
              return vibeqc::build_jk_ccm_weighted(
                  basis, system, cells, opts, P,
                  weight_cells, weight_matrices, method, omega);
          },
          py::arg("basis"), py::arg("system"),
          py::arg("weight_cells").noconvert(),
          py::arg("weight_matrices").noconvert(),
          py::arg("options"), py::arg("density"),
          py::arg("method") = "bra_home",
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "CCM WSSC-weighted J/K build. Like "
          "build_jk_gamma_molecular_limit but applies per-atom-pair "
          "four-center WSSC weights during the shell-quartet loop. "
          "weight_cells: (n_cells, 3) int array of cell indices. "
          "weight_matrices: (n_cells, n_atoms, n_atoms) float array "
          "of two-center pair weights.");

    m.def("nuclear_repulsion_per_cell",
          &vibeqc::nuclear_repulsion_per_cell,
          py::arg("system"), py::arg("options"),
          py::call_guard<py::gil_scoped_release>(),
          "Nuclear-nuclear repulsion per unit cell. Dispatches on "
          "``options.coulomb_method``: DIRECT_TRUNCATED (half-summed "
          "direct truncation — conditionally convergent in 3D) or "
          "EWALD_3D (Ewald summation — absolutely convergent) or "
          "SLAB_EWALD_2D (bare 2D slab Ewald nuclear block for the "
          "Gamma slab SCF).");

    // ----- Ewald summation (Phase 12e-a) ---------------------------------
    py::class_<vibeqc::EwaldOptions>(m, "EwaldOptions")
        .def(py::init<>())
        .def_readwrite("alpha",
                       &vibeqc::EwaldOptions::alpha,
                       "Gaussian screening parameter; ≤ 0 auto-selects.")
        .def_readwrite("real_cutoff_bohr",
                       &vibeqc::EwaldOptions::real_cutoff_bohr)
        .def_readwrite("recip_cutoff_bohr_inv",
                       &vibeqc::EwaldOptions::recip_cutoff_bohr_inv,
                       "Reciprocal-space cutoff in bohr⁻¹; ≤ 0 auto-selects.")
        .def_readwrite("tolerance",
                       &vibeqc::EwaldOptions::tolerance);

    m.def("ewald_point_charge_energy",
          &vibeqc::ewald_point_charge_energy,
          py::arg("lattice"), py::arg("positions_cart"), py::arg("charges"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Ewald-summed Madelung energy for an arbitrary 3D lattice of "
          "point charges. Handles non-neutral cells via the jellium "
          "(uniform compensating background) convention.");

    m.def("ewald_nuclear_repulsion",
          &vibeqc::ewald_nuclear_repulsion,
          py::arg("system"), py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Nuclear Madelung energy for a PeriodicSystem via Ewald summation "
          "(dim == 3 only).");

    m.def("ewald_point_charge_gradient",
          &vibeqc::ewald_point_charge_gradient,
          py::arg("lattice"), py::arg("positions_cart"), py::arg("charges"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Analytic gradient ∂E/∂R_C of the Ewald point-charge Madelung "
          "energy, returned as a 3 × N matrix (column C is the gradient on "
          "charge C). Matches a finite difference of "
          "ewald_point_charge_energy on the same α / cutoffs.");

    m.def("ewald_nuclear_repulsion_gradient",
          &vibeqc::ewald_nuclear_repulsion_gradient,
          py::arg("system"), py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Nuclear Ewald-repulsion gradient ∂E_nn/∂R_A for a PeriodicSystem, "
          "returned as an N × 3 matrix (row A is the gradient on atom A; "
          "dim == 3 only).");

    m.def("ewald_point_charge_potential",
          &vibeqc::ewald_point_charge_potential,
          py::arg("lattice"), py::arg("charge_positions_cart"),
          py::arg("charges"), py::arg("eval_points_cart"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::arg("include_short_range") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Ewald-summed Coulomb potential of a periodic lattice of point "
          "charges, evaluated at a list of Cartesian points. Set "
          "``include_short_range=False`` to omit the erfc 1/r spikes at "
          "each charge — used when the short-range part will be added "
          "analytically via libint's erfc_nuclear.");

    m.def("ewald_nuclear_potential",
          &vibeqc::ewald_nuclear_potential,
          py::arg("system"), py::arg("eval_points_cart"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::arg("include_short_range") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Electronic nuclear-attraction potential V_nuc(r) = -Σ Z_A/|r-R_A| "
          "summed over a 3D-periodic lattice via Ewald.");

    m.def("ewald_2d_point_charge_energy",
          &vibeqc::ewald_2d_point_charge_energy,
          py::arg("lattice"), py::arg("positions_cart"), py::arg("charges"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Rigorous 2D (slab) Ewald Madelung energy (Parry 1975 / de Leeuw-"
          "Perram 1979) for a charge distribution periodic in the plane "
          "spanned by lattice columns 0,1 and finite along the normal. The "
          "true 2D lattice sum — no vacuum padding, no inter-image dipole "
          "artefact. Requires a charge-neutral cell (Σ q = 0); a net-charged "
          "periodic plane has a divergent energy and is rejected.");

    m.def("ewald_2d_point_charge_potential",
          &vibeqc::ewald_2d_point_charge_potential,
          py::arg("lattice"), py::arg("charge_positions_cart"),
          py::arg("charges"), py::arg("eval_points_cart"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::arg("include_short_range") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Rigorous 2D (slab) Ewald electrostatic potential (Parry 1975 / de "
          "Leeuw-Perram 1979) of a charge distribution periodic in lattice "
          "columns 0,1 and finite along the normal, evaluated at a list of "
          "Cartesian points. The long-range building block for the "
          "electron-nuclear attraction in a slab SCF. Set "
          "``include_short_range=False`` to omit the erfc 1/r spikes at the "
          "charges. Requires a charge-neutral cell.");

    m.def("ewald_2d_point_charge_energy_with_background",
          &vibeqc::ewald_2d_point_charge_energy_with_background,
          py::arg("lattice"), py::arg("positions_cart"), py::arg("charges"),
          py::arg("z_background"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Rigorous 2D (slab) Ewald energy of a net-charged layer plus a "
          "co-located uniform in-plane neutralising sheet (total charge "
          "-sum(q) per cell) at normal coordinate ``z_background`` (Parry 1975 "
          "/ de Leeuw-Perram 1979). The sheet couples only through the g=0 "
          "channel; on a charge-neutral input it vanishes and the result "
          "equals ewald_2d_point_charge_energy. No neutrality guard — a "
          "net-charged ``charges`` is the intended input. The per-block "
          "building block for the slab SCF energy decomposition.");

    m.def("ewald_2d_point_charge_gradient_with_background",
          &vibeqc::ewald_2d_point_charge_gradient_with_background,
          py::arg("lattice"), py::arg("positions_cart"), py::arg("charges"),
          py::arg("z_background"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Analytic gradient ∂E/∂R_C of "
          "ewald_2d_point_charge_energy_with_background, returned as a 3 × N "
          "matrix (column C is the gradient on charge C). Differentiates the "
          "same truncated 2D-Ewald sums the energy evaluates (identical α and "
          "cutoffs): the real-space pair term, the z-resolved Parry "
          "reciprocal kernel (in-plane force from the phase, normal force "
          "from ∂F/∂z), the g=0 slab term (f'(z) = erf(αz)), and the "
          "neutralising-sheet term (-(2π/A) q_b q_C sgn(z_C - z_b) along the "
          "normal). On a neutral cell this is the gradient of "
          "ewald_2d_point_charge_energy.");

    m.def("ewald_2d_point_charge_potential_with_background",
          &vibeqc::ewald_2d_point_charge_potential_with_background,
          py::arg("lattice"), py::arg("charge_positions_cart"),
          py::arg("charges"), py::arg("z_background"),
          py::arg("eval_points_cart"),
          py::arg("options") = vibeqc::EwaldOptions{},
          py::arg("include_short_range") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Rigorous 2D (slab) Ewald electrostatic potential of a net-charged "
          "layer plus a co-located uniform neutralising sheet (charge "
          "-sum(q) at normal coordinate ``z_background``), evaluated at a list "
          "of Cartesian points. The exact uniform-sheet field "
          "-(2*pi/A) q_b |z - z_b| is added to the bare-source potential; zero "
          "on neutral input. No neutrality "
          "guard. The long-range building block for the slab-SCF electron-"
          "nuclear attraction with non-neutral nuclear blocks.");

    m.def("compute_nuclear_lattice_ewald",
          &vibeqc::compute_nuclear_lattice_ewald,
          py::arg("basis"), py::arg("system"), py::arg("grid"),
          py::arg("options"),
          py::arg("ewald_options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Full 3D Ewald-summed nuclear-attraction lattice sum V_μν(g) via "
          "numerical integration on the molecular grid. Replacement for "
          "compute_nuclear_lattice in 3D bulk where the point-charge sum "
          "is conditionally convergent.");

    m.def("compute_vsap_lattice",
          &vibeqc::compute_vsap_lattice,
          py::arg("basis"), py::arg("system"), py::arg("grid"),
          py::arg("sap_basis_name") = std::string("sap_helfem_large"),
          py::arg("options") = vibeqc::LatticeSumOptions{},
          py::arg("ewald_options") = vibeqc::EwaldOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Lattice-summed SAP (superposition-of-atomic-potentials) guess "
          "potential V^SAP_μν(g) on the periodic Becke grid. Each erf-screened "
          "SAP term is summed in reciprocal space (the smooth Ewald piece); the "
          "G=0 background is dropped (leaves the guess density invariant). For "
          "the periodic SAP initial guess (F_SAP = T + V_SAP).");

    m.def("compute_sap_potential_molecular",
          &vibeqc::compute_sap_potential_molecular,
          py::arg("basis"), py::arg("mol"),
          py::arg("sap_basis_name") = std::string("sap_helfem_large"),
          py::call_guard<py::gil_scoped_release>(),
          "Molecular SAP potential matrix V_SAP_μν (analytical, libint). "
          "Parity reference for compute_vsap_lattice.");

    py::class_<vibeqc::PeriodicRHFOptions>(m, "PeriodicRHFOptions")
        .def(py::init<>())
        .def_readwrite("max_iter",
                       &vibeqc::PeriodicRHFOptions::max_iter)
        .def_readwrite("conv_tol_energy",
                       &vibeqc::PeriodicRHFOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad",
                       &vibeqc::PeriodicRHFOptions::conv_tol_grad)
        .def_readwrite("damping",
                       &vibeqc::PeriodicRHFOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::PeriodicRHFOptions::dynamic_damping,
                       "See RHFOptions.dynamic_damping. Default True.")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::PeriodicRHFOptions::dynamic_damping_min)
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::PeriodicRHFOptions::dynamic_damping_max)
        .def_readwrite("fock_mixing",
                       &vibeqc::PeriodicRHFOptions::fock_mixing,
                       "Fock matrix mixing fraction. CRYSTAL FMIXING "
                       "30 corresponds to 0.30. Default 0.0 (off).")
        .def_readwrite("use_diis",
                       &vibeqc::PeriodicRHFOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::PeriodicRHFOptions::diis_start_iter)
        .def_readwrite("diis_subspace_size",
                       &vibeqc::PeriodicRHFOptions::diis_subspace_size)
        .def_readwrite("scf_accelerator",
                       &vibeqc::PeriodicRHFOptions::scf_accelerator,
                       "SCFAccelerator family member (DIIS / KDIIS / "
                       "EDIIS / EDIIS_DIIS / ADIIS). Default EDIIS_DIIS "
                       "(production hybrid).")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::PeriodicRHFOptions::ediis_diis_switch_threshold)
        .def_readwrite("diis_restart_tau",
                       &vibeqc::PeriodicRHFOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ. See "
                       "RHFOptions.diis_restart_tau. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::PeriodicRHFOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ. See "
                       "RHFOptions.diis_adaptive_delta. Default 1e-4.")
        .def_readwrite("initial_guess",
                       &vibeqc::PeriodicRHFOptions::initial_guess)
        .def_readwrite("level_shift",
                       &vibeqc::PeriodicRHFOptions::level_shift,
                       "Saunders-Hillier level shift (Hartree). Adds "
                       "b · (S − ½ S D S) to F before diagonalisation, "
                       "raising virtual orbital eigenvalues by b. "
                       "Default 0.0 (no shift). Useful when DIIS "
                       "oscillates on small-HOMO–LUMO-gap insulators.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::PeriodicRHFOptions::level_shift_warmup_cycles,
                       "CRYSTAL-style level-shift warm-up cycles. "
                       "-1 auto, 0 persistent shift, positive values "
                       "request an explicit shifted startup length.")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::PeriodicRHFOptions::level_shift_schedule,
                       "Explicit per-iteration Saunders-Hillier shift "
                       "curve (CRYSTAL LEVSHIFT B IRESET style). Empty ⇒ "
                       "derive from level_shift + level_shift_warmup_cycles. "
                       "Non-empty ⇒ use directly (entry i is the shift at "
                       "iteration i+1, last entry reused past its length). "
                       "Lower a vibeqc.LevelShiftSchedule via .as_list().")
        .def_readwrite("smearing_temperature",
                       &vibeqc::PeriodicRHFOptions::smearing_temperature,
                       "Fermi-Dirac smearing temperature (Hartree). "
                       "T > 0 replaces hard Aufbau with smooth "
                       "occupations and reports the free energy "
                       "A = E − T·S. Default 0.0 (no smearing).")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::PeriodicRHFOptions::quadratic_fallback_iter,
                       "Phase C1c — second-order SCF fallback "
                       "activation iteration. When > 0, after this "
                       "iteration the Ewald drivers switch from "
                       "'diagonalize F' to a Newton step in MO space "
                       "with diagonal-Hessian preconditioning. Use "
                       "for small-gap systems where DIIS + level "
                       "shift fail to converge. Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::PeriodicRHFOptions::quadratic_fallback_shift,
                       "Damping shift λ in the orbital-rotation "
                       "denominator (ε_a − ε_i + λ) of the C1c "
                       "Newton step. Default 0.1 Hartree. Larger "
                       "values produce more conservative steps.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::PeriodicRHFOptions::quadratic_fallback_max_step,
                       "Trust-region cap on ‖κ‖ per C1c Newton "
                       "step. Default 0.1. Larger values produce "
                       "more aggressive rotations; smaller values "
                       "are safer.")
        .def_readwrite("lattice_opts",
                       &vibeqc::PeriodicRHFOptions::lattice_opts)
        .def_readwrite("dft_plus_u_sites",
                       &vibeqc::PeriodicRHFOptions::dft_plus_u_sites,
                       "Internal: list of _HubbardSiteCxx. Populated by "
                       "the Python run_rhf_periodic_gamma wrapper.")
        .def_readwrite("dft_plus_u_ao_groups",
                       &vibeqc::PeriodicRHFOptions::dft_plus_u_ao_groups,
                       "Internal: parallel AO-index lists per site.")
        .def_readwrite("read_density",
                       &vibeqc::PeriodicRHFOptions::read_density,
                       "READ restart: closed-shell g=0 cell density (RHF), "
                       "projected onto this cell's basis. Filled by the "
                       "run_periodic_job wrapper; the driver injects it at "
                       "g=0. Empty = not a READ start.")
        .def_readwrite("read_density_alpha",
                       &vibeqc::PeriodicRHFOptions::read_density_alpha,
                       "READ restart (open-shell UHF): per-spin g=0 alpha "
                       "density. See read_density.")
        .def_readwrite("read_density_beta",
                       &vibeqc::PeriodicRHFOptions::read_density_beta,
                       "READ restart (open-shell UHF): per-spin g=0 beta "
                       "density. See read_density.")
        .def_readwrite("read_path",
                       &vibeqc::PeriodicRHFOptions::read_path,
                       "READ source file (.qvf / .molden) for the periodic "
                       "wrapper; empty restarts from an in-memory read_from.")
        .def_readwrite("atomic_spins",
                       &vibeqc::PeriodicRHFOptions::atomic_spins,
                       "ATOMSPIN: per-atom spin seed (list of +1/-1/0 in "
                       "unit-cell atom order) for a broken-symmetry SAD "
                       "start (open-shell UHF). Empty = spin-symmetric.")
        .def_readwrite("spinlock_mode",
                       &vibeqc::PeriodicRHFOptions::spinlock_mode,
                       "SPINLOCK mode (open-shell UHF): OFF (default), "
                       "SPIN_SCHEDULE (lock n_alpha-n_beta=spinlock_value for "
                       "spinlock_iterations cycles then release), or "
                       "PATTERN_HOLD (MOM-hold the seeded broken-symmetry "
                       "pattern for spinlock_iterations).")
        .def_readwrite("spinlock_value",
                       &vibeqc::PeriodicRHFOptions::spinlock_value,
                       "SPINLOCK SPIN_SCHEDULE: the n_alpha-n_beta to lock "
                       "during the first spinlock_iterations cycles.")
        .def_readwrite("spinlock_iterations",
                       &vibeqc::PeriodicRHFOptions::spinlock_iterations,
                       "SPINLOCK: number of initial SCF cycles to hold (both "
                       "modes). 0 = off.")
        .def_readwrite("use_davidson",
                       &vibeqc::PeriodicRHFOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::PeriodicRHFOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::PeriodicRHFOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::PeriodicRHFOptions::ecp_primitive_blocks,
                       "Inline primitive ECP blocks, one per ECP-bearing "
                       "home atom; run_periodic_job fills them from the "
                       "basis' .ecp sidecar or the pob-TZVP-rev2 CRYSTAL "
                       "records. Consumed by the k-point GDF drivers.")
        .def_readwrite("ecp_home_centers",
                       &vibeqc::PeriodicRHFOptions::ecp_home_centers,
                       "Home-cell centres (bohr) matching "
                       "ecp_primitive_blocks.")
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::PeriodicRHFOptions::ecp_effective_charges,
                       "Per-atom effective nuclear charges Z - n_core.")
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::PeriodicRHFOptions::ecp_total_ncore,
                       "Core electrons replaced by the ECPs; the driver "
                       "fills n_electrons - ecp_total_ncore.");

    py::class_<vibeqc::PeriodicRHFResult>(m, "PeriodicRHFResult")
        .def_readonly("restart_basis", &vibeqc::PeriodicRHFResult::restart_basis)
        .def_readonly("restart_lattice", &vibeqc::PeriodicRHFResult::restart_lattice)
        .def_readonly("restart_kpoints", &vibeqc::PeriodicRHFResult::restart_kpoints)
        .def_readonly("restart_weights", &vibeqc::PeriodicRHFResult::restart_weights)
        .def_readwrite("guess_selection", &vibeqc::PeriodicRHFResult::guess_selection)
        .def_readonly("energy",       &vibeqc::PeriodicRHFResult::energy)
        .def_readonly("e_electronic", &vibeqc::PeriodicRHFResult::e_electronic)
        .def_readonly("e_nuclear",    &vibeqc::PeriodicRHFResult::e_nuclear)
        .def_readonly("e_dft_plus_u", &vibeqc::PeriodicRHFResult::e_dft_plus_u,
                      "Dudarev DFT+U contribution per unit cell (Hartree). "
                      "0 unless PeriodicRHFOptions.dft_plus_u_sites is "
                      "non-empty.")
        .def_readonly("n_iter",       &vibeqc::PeriodicRHFResult::n_iter)
        .def_readonly("converged",    &vibeqc::PeriodicRHFResult::converged)
        .def_readonly("mo_energies",  &vibeqc::PeriodicRHFResult::mo_energies)
        .def_readonly("mo_coeffs",    &vibeqc::PeriodicRHFResult::mo_coeffs)
        .def_readonly("density",      &vibeqc::PeriodicRHFResult::density)
        .def_readonly("mo_coeffs_k", &vibeqc::PeriodicRHFResult::mo_coeffs_k)
        .def_readonly("occupations_k", &vibeqc::PeriodicRHFResult::occupations_k)
        .def_readonly("mo_energies_k", &vibeqc::PeriodicRHFResult::mo_energies_k)
        .def_readonly("fock",         &vibeqc::PeriodicRHFResult::fock)
        .def_readonly("overlap",      &vibeqc::PeriodicRHFResult::overlap)
        .def_property_readonly(
            "has_mean_field_state",
            [](const vibeqc::PeriodicRHFResult& result) {
                return static_cast<bool>(result.mean_field_state);
            })
        .def_property_readonly(
            "mean_field_state",
            [](const vibeqc::PeriodicRHFResult& result)
                -> std::shared_ptr<
                    vibeqc::PeriodicRestrictedMeanFieldState> {
                return std::const_pointer_cast<
                    vibeqc::PeriodicRestrictedMeanFieldState>(
                        result.mean_field_state);
            },
            "Experimental full-k physical RHF state, or None. The returned "
            "immutable Python surface shares ownership with this result and "
            "offers only one-k-at-a-time accessors.")
        .def_readonly("scf_trace",    &vibeqc::PeriodicRHFResult::scf_trace)
        .def("__repr__", [](const vibeqc::PeriodicRHFResult& r) {
            return std::string{"PeriodicRHFResult(energy="}
                   + std::to_string(r.energy)
                   + ", converged=" + (r.converged ? "True" : "False")
                   + ", n_iter=" + std::to_string(r.n_iter) + ")";
        });

    m.def("run_rhf_periodic_gamma", &vibeqc::run_rhf_periodic_gamma,
          py::arg("system"), py::arg("basis"),
          py::arg("options") = vibeqc::PeriodicRHFOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Γ-only closed-shell RHF for a periodic system (molecular-limit "
          "regime).");

    // ----- Multi-k periodic RHF (Phase 12c) ------------------------------
    m.def("build_fock_2e_real_space", &vibeqc::build_fock_2e_real_space,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("exchange_scale") = 1.0,
          py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "General multi-k two-electron Fock F^{2e}(g) = J(g) − "
          "½·exchange_scale·K(g) from a real-space density matrix. "
          "exchange_scale=0 skips the K computation (pure DFT). "
          "omega > 0 swaps the 1/r_12 kernel for erfc(ω·r_12)/r_12 "
          "(short-range Ewald split).");

    py::class_<vibeqc::JKLatticeMatrixSets>(m, "JKLatticeMatrixSets")
        .def_readonly("cell_triples_considered",
                      &vibeqc::JKLatticeMatrixSets::cell_triples_considered)
        .def_readonly("cell_triples_possible",
                      &vibeqc::JKLatticeMatrixSets::cell_triples_possible)
        .def_readonly("shell_quartets_considered",
                      &vibeqc::JKLatticeMatrixSets::shell_quartets_considered)
        .def_readonly("shell_pair_outputs",
                      &vibeqc::JKLatticeMatrixSets::shell_pair_outputs)
        .def_readonly("cell_sparse_outputs",
                      &vibeqc::JKLatticeMatrixSets::cell_sparse_outputs)
        .def_readonly("exhaustive_outputs",
                      &vibeqc::JKLatticeMatrixSets::exhaustive_outputs)
        .def_readonly("shell_pair_fallback_outputs",
                      &vibeqc::JKLatticeMatrixSets::shell_pair_fallback_outputs)
        .def_readonly("cell_sparse_fallback_outputs",
                      &vibeqc::JKLatticeMatrixSets::cell_sparse_fallback_outputs)
        .def_readonly("charge_screening_available",
                      &vibeqc::JKLatticeMatrixSets::charge_screening_available)
        .def_readonly("product_screening_available",
                      &vibeqc::JKLatticeMatrixSets::product_screening_available)
        .def_readonly("J", &vibeqc::JKLatticeMatrixSets::J)
        .def_readonly("K", &vibeqc::JKLatticeMatrixSets::K)
        .def_readonly(
            "J_gamma_energy_derivative",
            &vibeqc::JKLatticeMatrixSets::J_gamma_energy_derivative)
        .def_readonly(
            "K_gamma_energy_derivative",
            &vibeqc::JKLatticeMatrixSets::K_gamma_energy_derivative)
        .def_readonly(
            "J_density_energy_derivative",
            &vibeqc::JKLatticeMatrixSets::J_density_energy_derivative)
        .def_readonly(
            "K_density_energy_derivative",
            &vibeqc::JKLatticeMatrixSets::K_density_energy_derivative);

    m.def("build_jk_2e_real_space", &vibeqc::build_jk_2e_real_space,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "General multi-k two-electron component build returning "
          "separate real-space J(g) and full K(g) blocks. K has no "
          "-1/2 prefactor; callers form J - 1/2*exchange_scale*K.");

    m.def("build_jk_2e_real_space_explicit",
          &vibeqc::build_jk_2e_real_space_explicit,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("cells"), py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Same as build_jk_2e_real_space, but with caller-supplied "
          "cell list (Phase SYM3b — symmetry-reduced Fock build).");

    m.def("build_jk_2e_real_space_output_subset",
          &vibeqc::build_jk_2e_real_space_output_subset,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("output_indices"), py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Phase SYM3b point-group reduction: build J(g)/K(g) ONLY for the "
          "bra cells in output_indices (orbit representatives), with the full "
          "internal lattice sum — each emitted block is exact; non-emitted "
          "blocks are left zero for caller-side point-group reconstruction.");

    m.def("build_jk_2e_real_space_output_subset_masked",
          &vibeqc::build_jk_2e_real_space_output_subset_masked,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("output_indices"),
          py::arg("output_shell_masks"), py::arg("omega") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Phase SYM3b shell-pair mask: like build_jk_2e_real_space_output_"
          "subset, but builds only the atom-pair-orbit representative "
          "sub-blocks within each output cell. output_shell_masks is parallel "
          "to output_indices; each is an nshells*nshells row-major uint8 flag "
          "array (build the (s1_home, s2_cell) output shell pair iff non-zero). "
          "Internal lattice sum stays full, so each emitted sub-block is exact.");

    m.def("build_jk_2e_real_space_domains",
          &vibeqc::build_jk_2e_real_space_domains,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("cells"),
          py::arg("output_indices") = std::vector<int>{},
          py::arg("output_shell_masks") =
              std::vector<std::vector<uint8_t>>{},
          py::arg("omega") = 0.0, py::arg("compute_exchange") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Pair-resolved truncation (M1): full truncation-domain control "
          "over the direct J/K build. ``cells`` is the internal (c_g, c_lam, "
          "c_sig) summation cell list (the other entry points derive it "
          "radially from options.cutoff_bohr); ``output_indices`` restricts "
          "the emitted bra cells (positions into ``cells``; empty = all); "
          "``output_shell_masks`` (parallel to ``output_indices``, one "
          "nshells*nshells row-major uint8 array each) restricts the emitted "
          "output shell pairs. The density support is data-level: P(h) "
          "lookups follow the density's own cell list. Each emitted "
          "(sub-)block is the exact sum over ``cells``. "
          "``compute_exchange=False`` skips the K contraction (J-only; K "
          "blocks return zero).");

    // ---- Dormant BIPOLE quartet-classifier prototype --------------------
    py::class_<vibeqc::BipolarQuartetEntry>(m, "BipolarQuartetEntry",
        "One far-field shell quartet: its shell indices, cell indices, "
        "multipole truncation order, and bra/ket product-distribution widths.")
        .def(py::init<>())
        .def_readwrite("s1", &vibeqc::BipolarQuartetEntry::s1)
        .def_readwrite("s2", &vibeqc::BipolarQuartetEntry::s2)
        .def_readwrite("s3", &vibeqc::BipolarQuartetEntry::s3)
        .def_readwrite("s4", &vibeqc::BipolarQuartetEntry::s4)
        .def_readwrite("c_bra", &vibeqc::BipolarQuartetEntry::c_bra)
        .def_readwrite("c_ket", &vibeqc::BipolarQuartetEntry::c_ket)
        .def_readwrite("truncation_order",
                        &vibeqc::BipolarQuartetEntry::truncation_order)
        .def_readwrite("bra_width",
                        &vibeqc::BipolarQuartetEntry::bra_width)
        .def_readwrite("ket_width",
                        &vibeqc::BipolarQuartetEntry::ket_width);

    py::class_<vibeqc::BipoleShellInfo>(m, "BipoleShellInfo",
        "Shell descriptor for the penetration dispatch: minimum primitive "
        "exponent, Cartesian origin, and atom index for symmetry mapping.")
        .def(py::init<>())
        .def_readwrite("min_exponent",
                        &vibeqc::BipoleShellInfo::min_exponent)
        .def_readwrite("origin", &vibeqc::BipoleShellInfo::origin)
        .def_readwrite("atom_index",
                        &vibeqc::BipoleShellInfo::atom_index);

    py::class_<vibeqc::BipoleCellInfo>(m, "BipoleCellInfo",
        "Lattice cell descriptor: Cartesian vector (bohr) and integer "
        "lattice index.")
        .def(py::init<>())
        .def_readwrite("r_cart", &vibeqc::BipoleCellInfo::r_cart)
        .def_readwrite("index", &vibeqc::BipoleCellInfo::index);

    py::class_<vibeqc::BipoleDispatchParams>(m, "BipoleDispatchParams",
        "Parameters for the implementation-specific quartet classifier: "
        "inverse cell length scale, dispatch slope, prototype overlap "
        "bound, and maximum multipole order. No one-to-one TOLINTEG "
        "mapping is documented.")
        .def(py::init<>())
        .def_readwrite("cell_length_scale_inv",
                        &vibeqc::BipoleDispatchParams::cell_length_scale_inv)
        .def_readwrite("dispatch_slope",
                        &vibeqc::BipoleDispatchParams::dispatch_slope)
        .def_readwrite("overlap_threshold",
                        &vibeqc::BipoleDispatchParams::overlap_threshold)
        .def_readwrite("max_multipole_order",
                        &vibeqc::BipoleDispatchParams::max_multipole_order);

    m.def("compute_bipolar_penetration_dispatch",
          &vibeqc::compute_bipolar_penetration_dispatch,
          py::arg("shells"), py::arg("cells"), py::arg("params"),
          py::call_guard<py::gil_scoped_release>(),
          "Enumerate entries selected by the implementation-specific "
          "geometric quartet classifier. OpenMP-parallelised over the "
          "(c_g, c_lam) cell-pair loop. A positive truncation order is "
          "a prototype assignment, not a paper-derived validity guarantee.");

    // ---- Dormant prototype far-field Fock kernel -------------------------
    py::class_<vibeqc::FarFieldFockKernelEntry>(
        m, "FarFieldFockKernelEntry",
        "Pre-computed far-field Fock kernel entry.  Stores the effective "
        "linear operator K_matrix and the AO sub-range indices for the "
        "bra and ket blocks.")
        .def(py::init<>())
        .def_readwrite("bra_cell_ix",
                        &vibeqc::FarFieldFockKernelEntry::bra_cell_ix)
        .def_readwrite("bra_cell_iy",
                        &vibeqc::FarFieldFockKernelEntry::bra_cell_iy)
        .def_readwrite("bra_cell_iz",
                        &vibeqc::FarFieldFockKernelEntry::bra_cell_iz)
        .def_readwrite("bra_b1",
                        &vibeqc::FarFieldFockKernelEntry::bra_b1)
        .def_readwrite("bra_b1e",
                        &vibeqc::FarFieldFockKernelEntry::bra_b1e)
        .def_readwrite("bra_b2",
                        &vibeqc::FarFieldFockKernelEntry::bra_b2)
        .def_readwrite("bra_b2e",
                        &vibeqc::FarFieldFockKernelEntry::bra_b2e)
        .def_readwrite("ket_cell_ix",
                        &vibeqc::FarFieldFockKernelEntry::ket_cell_ix)
        .def_readwrite("ket_cell_iy",
                        &vibeqc::FarFieldFockKernelEntry::ket_cell_iy)
        .def_readwrite("ket_cell_iz",
                        &vibeqc::FarFieldFockKernelEntry::ket_cell_iz)
        .def_readwrite("ket_b3",
                        &vibeqc::FarFieldFockKernelEntry::ket_b3)
        .def_readwrite("ket_b3e",
                        &vibeqc::FarFieldFockKernelEntry::ket_b3e)
        .def_readwrite("ket_b4",
                        &vibeqc::FarFieldFockKernelEntry::ket_b4)
        .def_readwrite("ket_b4e",
                        &vibeqc::FarFieldFockKernelEntry::ket_b4e)
        .def_readwrite("K_matrix",
                        &vibeqc::FarFieldFockKernelEntry::K_matrix);

    py::class_<vibeqc::FarFieldFockResult>(
        m, "FarFieldFockResult",
        "Result of applying the far-field Fock kernel to a density matrix.")
        .def(py::init<>())
        .def_readonly("fock_matrices",
                       &vibeqc::FarFieldFockResult::fock_matrices)
        .def_readonly("cell_ix",
                       &vibeqc::FarFieldFockResult::cell_ix)
        .def_readonly("cell_iy",
                       &vibeqc::FarFieldFockResult::cell_iy)
        .def_readonly("cell_iz",
                       &vibeqc::FarFieldFockResult::cell_iz)
        .def_readonly("coulomb_energy_far",
                       &vibeqc::FarFieldFockResult::coulomb_energy_far)
        .def_readonly("n_quartets_applied",
                       &vibeqc::FarFieldFockResult::n_quartets_applied);

    m.def("apply_far_field_fock_kernel",
          &vibeqc::apply_far_field_fock_kernel,
          py::arg("entries"),
          py::arg("density_cell_keys_flat"),
          py::arg("density_blocks"),
          py::arg("nbf"),
          py::call_guard<py::gil_scoped_release>(),
          "Apply a pre-computed far-field Fock kernel to density blocks. "
          "OpenMP-parallelised over kernel entries.  Returns a "
          "FarFieldFockResult with per-cell Fock contributions and "
          "the Coulomb far-field energy.");

    m.def("build_far_field_k_matrices_batch",
          &vibeqc::build_far_field_k_matrices_batch,
          py::arg("bra_moments_2d"),
          py::arg("ket_moments_2d"),
          py::arg("T_matrix"),
          py::call_guard<py::gil_scoped_release>(),
          "Batched K-matrix builder for the far-field Fock kernel. "
          "For each quartet in a group sharing the same interaction "
          "tensor T, computes K = bra @ T @ ket^T via Eigen matmul. "
          "OpenMP-parallelised over quartets.");

    // ---- Direct contractor for the dormant quartet prototype -------------
    py::class_<vibeqc::BipoleMomentBufferCpp>(m, "BipoleMomentBufferCpp",
        "Spherical moment buffer in C++-friendly flat layout.  Contains "
        "per-shell-pair spherical multipole moments, adjoined-Gaussian "
        "product-distribution centres, and shell-index metadata.")
        .def(py::init<>())
        .def_readwrite("L_max", &vibeqc::BipoleMomentBufferCpp::L_max)
        .def_readwrite("n_sph", &vibeqc::BipoleMomentBufferCpp::n_sph)
        .def_readwrite("entries", &vibeqc::BipoleMomentBufferCpp::entries)
        .def_readwrite("cell_positions",
                        &vibeqc::BipoleMomentBufferCpp::cell_positions)
        .def_readwrite("cell_indices",
                        &vibeqc::BipoleMomentBufferCpp::cell_indices)
        .def_readwrite("shell_slices",
                        &vibeqc::BipoleMomentBufferCpp::shell_slices);

    py::class_<vibeqc::BipolePairMomentEntry>(m, "BipolePairMomentEntry",
        "One shell-pair entry in the spherical moment buffer: flat moment "
        "matrix (n_ao_bra * n_ao_ket, n_sph), pair centre, and AO offsets.")
        .def(py::init<>())
        .def_readwrite("shell_index_1",
                        &vibeqc::BipolePairMomentEntry::shell_index_1)
        .def_readwrite("shell_index_2",
                        &vibeqc::BipolePairMomentEntry::shell_index_2)
        .def_readwrite("cell_index",
                        &vibeqc::BipolePairMomentEntry::cell_index)
        .def_readwrite("bf_offset_1",
                        &vibeqc::BipolePairMomentEntry::bf_offset_1)
        .def_readwrite("bf_count_1",
                        &vibeqc::BipolePairMomentEntry::bf_count_1)
        .def_readwrite("bf_offset_2",
                        &vibeqc::BipolePairMomentEntry::bf_offset_2)
        .def_readwrite("bf_count_2",
                        &vibeqc::BipolePairMomentEntry::bf_count_2)
        .def_readwrite("moments_flat",
                        &vibeqc::BipolePairMomentEntry::moments_flat)
        .def_readwrite("centre_x",
                        &vibeqc::BipolePairMomentEntry::centre_x)
        .def_readwrite("centre_y",
                        &vibeqc::BipolePairMomentEntry::centre_y)
        .def_readwrite("centre_z",
                        &vibeqc::BipolePairMomentEntry::centre_z);

    py::class_<vibeqc::BipoleQuartetEntryCpp>(m, "BipoleQuartetEntryCpp",
        "One far-field quartet entry: shell indices, cell indices, "
        "truncation order, and product-distribution widths.")
        .def(py::init<>())
        .def_readwrite("shell_1", &vibeqc::BipoleQuartetEntryCpp::shell_1)
        .def_readwrite("shell_2", &vibeqc::BipoleQuartetEntryCpp::shell_2)
        .def_readwrite("cell_bra", &vibeqc::BipoleQuartetEntryCpp::cell_bra)
        .def_readwrite("shell_3", &vibeqc::BipoleQuartetEntryCpp::shell_3)
        .def_readwrite("shell_4", &vibeqc::BipoleQuartetEntryCpp::shell_4)
        .def_readwrite("cell_ket", &vibeqc::BipoleQuartetEntryCpp::cell_ket)
        .def_readwrite("truncation_order",
                        &vibeqc::BipoleQuartetEntryCpp::truncation_order)
        .def_readwrite("bra_width",
                        &vibeqc::BipoleQuartetEntryCpp::bra_width)
        .def_readwrite("ket_width",
                        &vibeqc::BipoleQuartetEntryCpp::ket_width);

    py::class_<vibeqc::DensityBlockCpp>(m, "DensityBlockCpp",
        "Density matrix block for one lattice cell.")
        .def(py::init<>())
        .def_readwrite("cell_ix", &vibeqc::DensityBlockCpp::cell_ix)
        .def_readwrite("cell_iy", &vibeqc::DensityBlockCpp::cell_iy)
        .def_readwrite("cell_iz", &vibeqc::DensityBlockCpp::cell_iz)
        .def_readwrite("density", &vibeqc::DensityBlockCpp::density);

    py::class_<vibeqc::BipoleFarFieldResultCpp>(m, "BipoleFarFieldResultCpp",
        "Result of the C++ far-field Coulomb contractor.")
        .def_readonly("fock_blocks",
                       &vibeqc::BipoleFarFieldResultCpp::fock_blocks)
        .def_readonly("fock_cell_ix",
                       &vibeqc::BipoleFarFieldResultCpp::fock_cell_ix)
        .def_readonly("fock_cell_iy",
                       &vibeqc::BipoleFarFieldResultCpp::fock_cell_iy)
        .def_readonly("fock_cell_iz",
                       &vibeqc::BipoleFarFieldResultCpp::fock_cell_iz)
        .def_readonly("coulomb_energy_far",
                       &vibeqc::BipoleFarFieldResultCpp::coulomb_energy_far)
        .def_readonly("quartets_processed",
                       &vibeqc::BipoleFarFieldResultCpp::quartets_processed);

    m.def("compute_bipolar_coulomb_far_field_cpp",
          &vibeqc::compute_bipolar_coulomb_far_field_cpp,
          py::arg("moment_buffer"),
          py::arg("dispatch"),
          py::arg("density_blocks"),
          py::arg("ewald_omega"),
          py::arg("nbf_total"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute the Coulomb far-field Fock contribution for all quartets "
          "in the penetration dispatch.  OpenMP-parallelised over quartets "
          "with per-thread interaction-tensor caching.  Returns per-cell "
          "Fock blocks and the far-field Coulomb energy.\n\n"
          "Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the periodic "
          "quartet-expansion source. This contractor is a prototype.");

    // ---- Cartesian → spherical buffer conversion -------------------------
    py::class_<vibeqc::SphericalMomentBufferCpp>(m, "SphericalMomentBufferCpp",
        "Spherical multipole moment buffer in C++ layout.  Contains "
        "per-shell-pair flat spherical moments and adjoined pair centres.")
        .def(py::init<>())
        .def_readonly("L_max", &vibeqc::SphericalMomentBufferCpp::L_max)
        .def_readonly("n_sph", &vibeqc::SphericalMomentBufferCpp::n_sph)
        .def_readonly("cells", &vibeqc::SphericalMomentBufferCpp::cells)
        .def_readonly("shell_slices",
                       &vibeqc::SphericalMomentBufferCpp::shell_slices)
        .def_readonly("entries",
                       &vibeqc::SphericalMomentBufferCpp::entries);

    py::class_<vibeqc::SphericalMomentBufferCpp::Entry>(
        m, "SphericalMomentBufferEntry",
        "One shell-pair entry: flat spherical moment matrix, AO offsets, "
        "and adjoined-Gaussian pair centre.")
        .def(py::init<>())
        .def_readwrite("shell_1",
                        &vibeqc::SphericalMomentBufferCpp::Entry::shell_1)
        .def_readwrite("shell_2",
                        &vibeqc::SphericalMomentBufferCpp::Entry::shell_2)
        .def_readwrite("cell_index",
                        &vibeqc::SphericalMomentBufferCpp::Entry::cell_index)
        .def_readwrite("bf_offset_1",
                        &vibeqc::SphericalMomentBufferCpp::Entry::bf_offset_1)
        .def_readwrite("bf_count_1",
                        &vibeqc::SphericalMomentBufferCpp::Entry::bf_count_1)
        .def_readwrite("bf_offset_2",
                        &vibeqc::SphericalMomentBufferCpp::Entry::bf_offset_2)
        .def_readwrite("bf_count_2",
                        &vibeqc::SphericalMomentBufferCpp::Entry::bf_count_2)
        .def_readwrite("moments_flat",
                        &vibeqc::SphericalMomentBufferCpp::Entry::moments_flat)
        .def_readwrite("centre_x",
                        &vibeqc::SphericalMomentBufferCpp::Entry::centre_x)
        .def_readwrite("centre_y",
                        &vibeqc::SphericalMomentBufferCpp::Entry::centre_y)
        .def_readwrite("centre_z",
                        &vibeqc::SphericalMomentBufferCpp::Entry::centre_z);

    m.def("convert_cartesian_to_spherical_buffer",
          &vibeqc::convert_cartesian_to_spherical_buffer,
          py::arg("cartesian_blocks"),
          py::arg("C_matrix"),
          py::arg("shell_slices"),
          py::arg("cells"),
          py::arg("pair_centres"),
          py::call_guard<py::gil_scoped_release>(),
          "Convert per-shell-pair Cartesian multipole moments to spherical "
          "harmonic moments via matrix multiplication with the Cartesian→"
          "spherical conversion matrix.  OpenMP-parallelised over shell pairs.\n\n"
          "The conversion uses the documented Stone real-solid-harmonic "
          "convention; Saunders (1992) does not derive this API.");

    // ---- Adjoined-Gaussian pair-centre moment shift ----------------------
    py::class_<vibeqc::PairMultipoleMomentsCpp>(m, "PairMultipoleMomentsCpp",
        "Shifted Cartesian multipole moments about adjoined-Gaussian pair "
        "centres.  Same layout as LatticeMultipoleSet: blocks[c][comp].")
        .def(py::init<>())
        .def_readonly("nbf", &vibeqc::PairMultipoleMomentsCpp::nbf)
        .def_readonly("L_max", &vibeqc::PairMultipoleMomentsCpp::L_max)
        .def_readonly("cells", &vibeqc::PairMultipoleMomentsCpp::cells)
        .def_readonly("blocks", &vibeqc::PairMultipoleMomentsCpp::blocks)
        .def_readonly("centres_flat",
                       &vibeqc::PairMultipoleMomentsCpp::centres_flat)
        .def_readonly("shell_slices",
                       &vibeqc::PairMultipoleMomentsCpp::shell_slices);

    m.def("shift_multipole_moments_to_pair_centres",
          &vibeqc::shift_multipole_moments_to_pair_centres,
          py::arg("M_lat"), py::arg("basis"), py::arg("L_target"),
          py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
          py::call_guard<py::gil_scoped_release>(),
          "Shift global-origin Cartesian multipole moments to per-shell-pair "
          "adjoined-Gaussian product centres. OpenMP-parallelised over cells. "
          "Pisani-Dovesi (1980), Sec. 4, supplies the diffuse s-Gaussian "
          "convention; the centre and shift follow from standard Gaussian-"
          "product and binomial identities.");

    // ---- Moment-derivative buffer for the dormant prototype --------------
    py::class_<vibeqc::MomentDerivativeEntry>(m, "MomentDerivativeEntryCpp",
        "One shell pair's spherical moment derivatives dM_sph/dC_a for axes x,y,z.")
        .def(py::init<>())
        .def_readonly("shell_1", &vibeqc::MomentDerivativeEntry::shell_1)
        .def_readonly("shell_2", &vibeqc::MomentDerivativeEntry::shell_2)
        .def_readonly("cell_index", &vibeqc::MomentDerivativeEntry::cell_index)
        .def_readonly("bf_count_1", &vibeqc::MomentDerivativeEntry::bf_count_1)
        .def_readonly("bf_count_2", &vibeqc::MomentDerivativeEntry::bf_count_2)
        .def_readonly("bf_offset_1", &vibeqc::MomentDerivativeEntry::bf_offset_1)
        .def_readonly("bf_offset_2", &vibeqc::MomentDerivativeEntry::bf_offset_2)
        .def_readonly("gradient_flat",
                       &vibeqc::MomentDerivativeEntry::gradient_flat);

    py::class_<vibeqc::MomentDerivativeBufferCpp>(m, "MomentDerivativeBufferCpp",
        "Container for precomputed spherical moment derivatives for all shell pairs.")
        .def(py::init<>())
        .def_readonly("max_cart_L", &vibeqc::MomentDerivativeBufferCpp::max_cart_L)
        .def_readonly("n_sph_deriv", &vibeqc::MomentDerivativeBufferCpp::n_sph_deriv)
        .def_readonly("entries", &vibeqc::MomentDerivativeBufferCpp::entries);

    m.def("compute_moment_derivatives_for_buffer",
          &vibeqc::compute_moment_derivatives_for_buffer,
          py::arg("cartesian_blocks"),
          py::arg("C_matrix"),
          py::arg("shell_slices"),
          py::arg("cells"),
          py::arg("n_cart"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute dM_sph/dC_a for every shell pair.  OpenMP-parallelised "
          "over cells and shell pairs.  The derivative of Cartesian moment "
          "M^{(i,j,k)}(C) w.r.t. C_a is -i*M^{(i-1)}(C) (a=x).\n\n"
          "This follows by differentiating the standard binomial moment "
          "translation; no cited source derives the surrounding prototype.");

    // ---- Gradient contractor for the dormant quartet prototype -----------
    m.def("compute_bipolar_far_field_gradient_cpp",
          &vibeqc::compute_bipolar_far_field_gradient_cpp,
          py::arg("moment_buffer"),
          py::arg("dispatch"),
          py::arg("density_blocks"),
          py::arg("ewald_omega"),
          py::arg("n_atoms"),
          py::arg("shell_to_atom"),
          py::arg("include_moment_derivative") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Compute the far-field nuclear gradient via C++ OpenMP.\n\n"
          "For each far-field quartet, contracts the interaction-tensor "
          "gradient dT/dR with pair moments and density, then distributes "
          "forces to atoms via the shell-to-atom mapping.  Optionally "
          "includes sub-dominant dM/dA terms.\n\n"
          "Saunders (1992), Sec. 5.3, Eqs. (90)-(92), supports the radial "
          "derivative recurrence but not this full quartet algorithm.");

    // ---- Fock-kernel builder for the dormant prototype -------------------
    m.def("build_far_field_fock_kernel_cpp",
          &vibeqc::build_far_field_fock_kernel_cpp,
          py::arg("moment_buffer"), py::arg("dispatch"),
          py::arg("ewald_omega"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the far-field Fock kernel from the moment buffer and "
          "penetration dispatch.  Groups quartets by (R_sep, L_order), "
          "computes interaction tensors once per group, and builds "
          "K matrices via batched matmul.\n\n"
          "The grouping and cache strategy are implementation prototypes.");

    // ---- PDR 1988 Ch. II.4c multipole interaction-tensor prototype --------
    m.def("multipole_cartesian_to_spherical_matrix",
          &vibeqc::multipole_cartesian_to_spherical_matrix,
          py::arg("L_max"),
          py::call_guard<py::gil_scoped_release>(),
          "Cartesian -> spherical conversion matrix C (n_sph, n_cart). "
          "M_sph = C @ M_cart in the Stone real-solid-harmonic convention.");

    m.def("multipole_interaction_tensor",
          &vibeqc::multipole_interaction_tensor,
          py::arg("L_max_A"), py::arg("L_max_B"),
          py::arg("rx"), py::arg("ry"), py::arg("rz"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the bare-Coulomb spherical interaction tensor "
          "T_{l1,m1; l2,m2}(R) from exact Cartesian derivatives of "
          "1/|R|.  Returns (n_A, n_B) matrix with Stone real-solid-"
          "harmonic convention.");

    m.def("multipole_erfc_interaction_tensor",
          &vibeqc::multipole_erfc_interaction_tensor,
          py::arg("L_max_A"), py::arg("L_max_B"),
          py::arg("rx"), py::arg("ry"), py::arg("rz"),
          py::arg("mu"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the erfc-screened spherical interaction tensor "
          "for the short-range Ewald piece erfc(sqrt(mu)*r)/r.");

    m.def("multipole_erf_interaction_tensor",
          &vibeqc::multipole_erf_interaction_tensor,
          py::arg("L_max_A"), py::arg("L_max_B"),
          py::arg("rx"), py::arg("ry"), py::arg("rz"),
          py::arg("mu"),
          py::call_guard<py::gil_scoped_release>(),
          "Build the erf-screened spherical interaction tensor "
          "for the long-range Ewald piece erf(sqrt(mu)*r)/r.");

    m.def("multipole_interaction_tensor_gradient",
          &vibeqc::multipole_interaction_tensor_gradient,
          py::arg("L_max_A"), py::arg("L_max_B"),
          py::arg("rx"), py::arg("ry"), py::arg("rz"),
          py::call_guard<py::gil_scoped_release>(),
          "Gradient of the bare-Coulomb spherical interaction tensor. "
          "Returns a list of 3 matrices (d/dx, d/dy, d/dz), each of "
          "shape (n_A, n_B).  Computed from the order+1 Cartesian "
          "derivative via ∂/∂R_a d^gamma(1/r) = d^{gamma+1_a}(1/r).");

    m.def("_multipole_cartesian_derivative_inverse_r",
          &vibeqc::multipole_cartesian_derivative_inverse_r,
          py::arg("ix"), py::arg("iy"), py::arg("iz"),
          py::arg("rx"), py::arg("ry"), py::arg("rz"),
          "Internal regression hook for Cartesian derivatives of 1/r. "
          "Supports total derivative order through five; not a public "
          "far-field activation surface.");

    m.def("multipole_erfc_interaction_tensor_gradient",
          &vibeqc::multipole_erfc_interaction_tensor_gradient,
          py::arg("L_max_A"), py::arg("L_max_B"),
          py::arg("rx"), py::arg("ry"), py::arg("rz"),
          py::arg("mu"),
          py::call_guard<py::gil_scoped_release>(),
          "Gradient of the erfc-screened spherical interaction tensor. "
          "Returns a list of 3 matrices (d/dx, d/dy, d/dz).  Supports "
          "L_max ≤ 3.  Falls back to bare Coulomb gradient when mu = 0.");

    m.def("multipole_n_components",
          &vibeqc::multipole_n_components,
          py::arg("L_max"),
          "Number of real spherical harmonic components up to L_max.");

    m.def("build_jk_2e_real_space_bipolar_dispatch",
          &vibeqc::build_jk_2e_real_space_bipolar_dispatch,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("bipolar_skip_mask"),
          py::arg("omega") = 0.0, py::arg("compute_exchange") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Exact-ERI skip surface for the dormant PDR 1988 Ch. II.4c "
          "quartet prototype: builds J(g)/K(g) "
          "with per-shell-quartet skip control. ``bipolar_skip_mask`` is a "
          "vector (length n_c * n_c) of sorted integer vectors; each inner "
          "vector lists shell-quartet flat indices s1*nsh^3 + s2*nsh^2 + "
          "s3*nsh + s4 that the prototype marks for skipping from exact ERI "
          "evaluation). The Python multipole contractor supplies these "
          "contributions separately. Quartets NOT in the mask are near-"
          "field and evaluated exactly. The default empty mask gives the "
          "same result as build_jk_2e_real_space.");

    m.def("build_jk_2e_real_space_domains_gamma_derivative",
          &vibeqc::build_jk_2e_real_space_domains_gamma_derivative,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("cells"),
          py::arg("output_indices") = std::vector<int>{},
          py::arg("omega") = 0.0, py::arg("compute_exchange") = true,
          py::arg("output_shell_masks") =
              std::vector<std::vector<uint8_t>>{},
          py::arg("density_shell_masks") =
              std::vector<std::vector<uint8_t>>{},
          py::arg("gamma_density") = Eigen::MatrixXd(),
          py::call_guard<py::gil_scoped_release>(),
          "Gamma-only finite-domain density adjoint for the J/K energy "
          "bilinears. Uses the same internal, output, and density-support "
          "domains as build_jk_2e_real_space_domains and additionally "
          "returns J_gamma_energy_derivative and "
          "K_gamma_energy_derivative. Pair-resolved callers pass both shell "
          "mask families and the unmasked homogeneous gamma_density.");

    m.def("build_jk_2e_real_space_domains_density_derivative",
          &vibeqc::build_jk_2e_real_space_domains_density_derivative,
          py::arg("basis"), py::arg("system"), py::arg("options"),
          py::arg("density"), py::arg("cells"),
          py::arg("output_indices") = std::vector<int>{},
          py::arg("omega") = 0.0, py::arg("compute_exchange") = true,
          py::arg("output_shell_masks") =
              std::vector<std::vector<uint8_t>>{},
          py::arg("density_shell_masks") =
              std::vector<std::vector<uint8_t>>{},
          py::call_guard<py::gil_scoped_release>(),
          "General finite-domain density adjoint for the J/K energy "
          "bilinears. Uses the same internal, output, and density-support "
          "domains as build_jk_2e_real_space_domains and returns one "
          "J_density_energy_derivative / K_density_energy_derivative block "
          "for every input real-space density cell.");

    m.def("real_space_density_from_kpoints",
          &vibeqc::real_space_density_from_kpoints,
          py::arg("C_per_k"), py::arg("n_occ_per_k"),
          py::arg("kmesh"), py::arg("cells"),
          py::call_guard<py::gil_scoped_release>(),
          "Fold per-k MO coefficients into a real-space density matrix P(g).");

    m.def("real_space_density_from_kpoints_fractional",
          &vibeqc::real_space_density_from_kpoints_fractional,
          py::arg("C_per_k"), py::arg("occ_per_k"),
          py::arg("kmesh"), py::arg("cells"),
          py::call_guard<py::gil_scoped_release>(),
          "Fractional-occupation variant of real_space_density_from_kpoints. "
          "``occ_per_k`` is a list of double arrays — one occupation per MO "
          "per k-point, typically in [0, 2] for closed-shell RHF. Used by "
          "Fermi-Dirac-smearing-driven SCF (Phase C1b).");

    py::class_<vibeqc::PeriodicSCFOptions>(m, "PeriodicSCFOptions")
        .def(py::init<>())
        .def_readwrite("max_iter",
                       &vibeqc::PeriodicSCFOptions::max_iter)
        .def_readwrite("conv_tol_energy",
                       &vibeqc::PeriodicSCFOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad",
                       &vibeqc::PeriodicSCFOptions::conv_tol_grad)
        .def_readwrite("damping",
                       &vibeqc::PeriodicSCFOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::PeriodicSCFOptions::dynamic_damping,
                       "See RHFOptions.dynamic_damping. Default True.")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::PeriodicSCFOptions::dynamic_damping_min)
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::PeriodicSCFOptions::dynamic_damping_max)
        .def_readwrite("fock_mixing",
                       &vibeqc::PeriodicSCFOptions::fock_mixing,
                       "Fock matrix mixing fraction. See "
                       "RHFOptions.fock_mixing.")
        .def_readwrite("use_diis",
                       &vibeqc::PeriodicSCFOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::PeriodicSCFOptions::diis_start_iter)
        .def_readwrite("diis_subspace_size",
                       &vibeqc::PeriodicSCFOptions::diis_subspace_size)
        .def_readwrite("scf_accelerator",
                       &vibeqc::PeriodicSCFOptions::scf_accelerator,
                       "SCFAccelerator family. Default EDIIS_DIIS "
                       "(production hybrid).")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::PeriodicSCFOptions::ediis_diis_switch_threshold)
        .def_readwrite("diis_restart_tau",
                       &vibeqc::PeriodicSCFOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ. See "
                       "RHFOptions.diis_restart_tau. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::PeriodicSCFOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ. See "
                       "RHFOptions.diis_adaptive_delta. Default 1e-4.")
        .def_readwrite("initial_guess",
                       &vibeqc::PeriodicSCFOptions::initial_guess)
        .def_readwrite("read_density_k",
                       &vibeqc::PeriodicSCFOptions::read_density_k)
        .def_readwrite("level_shift",
                       &vibeqc::PeriodicSCFOptions::level_shift,
                       "Saunders-Hillier level shift (Hartree). Default 0.0.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::PeriodicSCFOptions::level_shift_warmup_cycles,
                       "CRYSTAL-style level-shift warm-up cycles. "
                       "-1 auto, 0 persistent shift, positive values "
                       "request an explicit shifted startup length.")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::PeriodicSCFOptions::level_shift_schedule,
                       "Explicit per-iteration shift curve. See "
                       "PeriodicRHFOptions.level_shift_schedule.")
        .def_readwrite("smearing_temperature",
                       &vibeqc::PeriodicSCFOptions::smearing_temperature,
                       "Fermi-Dirac smearing temperature (Hartree). Default 0.0.")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::PeriodicSCFOptions::quadratic_fallback_iter,
                       "C1c second-order SCF fallback activation iter. "
                       "Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::PeriodicSCFOptions::quadratic_fallback_shift,
                       "C1c shift λ in (ε_a − ε_i + λ). Default 0.1.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::PeriodicSCFOptions::quadratic_fallback_max_step,
                       "C1c trust-region cap on ‖κ‖ per Newton step. "
                       "Default 0.1.")
        .def_readwrite("lattice_opts",
                       &vibeqc::PeriodicSCFOptions::lattice_opts)
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::PeriodicSCFOptions::ecp_primitive_blocks)
        .def_readwrite("ecp_home_centers",
                       &vibeqc::PeriodicSCFOptions::ecp_home_centers)
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::PeriodicSCFOptions::ecp_effective_charges)
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::PeriodicSCFOptions::ecp_total_ncore)
        .def_readwrite("use_davidson",
                       &vibeqc::PeriodicSCFOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::PeriodicSCFOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::PeriodicSCFOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.");

    m.def("run_rhf_periodic", &vibeqc::run_rhf_periodic,
          py::arg("system"), py::arg("basis"), py::arg("kmesh"),
          py::arg("options") = vibeqc::PeriodicSCFOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Multi-k closed-shell RHF for a periodic system — real-space "
          "density matrix, general three-lattice-index Fock build.");

    m.def("_run_rhf_periodic_with_state_capture",
          &vibeqc::run_rhf_periodic_with_state_capture,
          py::arg("system"), py::arg("basis"), py::arg("kmesh"),
          py::arg("options"), py::arg("request"),
          py::call_guard<py::gil_scoped_release>(),
          "Experimental direct-truncated multi-k RHF producer for native "
          "correlation development. The request caps retained state bytes; "
          "it is not total-SCF resource admission.");

    // ----- Multi-k periodic KS-DFT (Phase 12d) ---------------------------
    py::class_<vibeqc::PeriodicXCContribution>(m, "PeriodicXCContribution")
        .def_readonly("V_xc", &vibeqc::PeriodicXCContribution::V_xc)
        .def_readonly("e_xc", &vibeqc::PeriodicXCContribution::e_xc);

    py::enum_<vibeqc::PeriodicXCDensityDomain>(m,
                                                "PeriodicXCDensityDomain")
        .value("AUTO", vibeqc::PeriodicXCDensityDomain::AUTO)
        .value("MOLECULAR_HOME",
               vibeqc::PeriodicXCDensityDomain::MOLECULAR_HOME)
        .value("PERIODIC_LATTICE",
               vibeqc::PeriodicXCDensityDomain::PERIODIC_LATTICE);

    m.def("periodic_xc_domain_counts", &vibeqc::periodic_xc_domain_counts,
          py::arg("system"), py::arg("cells"), py::arg("options"),
          py::arg("density_domain") = vibeqc::PeriodicXCDensityDomain::AUTO,
          py::call_guard<py::gil_scoped_release>());
    m.def("periodic_xc_value_batch_size", &vibeqc::periodic_xc_value_batch_size,
          py::arg("n_basis"), py::arg("n_active"), py::arg("need_grad"),
          py::arg("open_shell"), py::arg("n_threads"));
    m.def("periodic_xc_gradient_batch_size", &vibeqc::periodic_xc_gradient_batch_size,
          py::arg("n_basis"), py::arg("n_cells"), py::arg("need_hess"),
          py::arg("open_shell"), py::arg("n_threads"));

    m.def("build_xc_periodic", &vibeqc::build_xc_periodic,
          py::arg("basis"), py::arg("system"), py::arg("grid"),
          py::arg("functional"), py::arg("density"), py::arg("options"),
          py::arg("density_domain") =
              vibeqc::PeriodicXCDensityDomain::AUTO,
          py::call_guard<py::gil_scoped_release>(),
          "Periodic XC Fock contribution V_xc(g) and E_xc per unit cell.");

    m.def("xc_lattice_gradient_contribution",
          &vibeqc::xc_lattice_gradient_contribution,
          py::arg("basis"), py::arg("system"), py::arg("grid"),
          py::arg("functional"), py::arg("density"), py::arg("options"),
          py::arg("density_domain") =
              vibeqc::PeriodicXCDensityDomain::AUTO,
          py::call_guard<py::gil_scoped_release>(),
          "Periodic XC Pulay atomic gradient (n_atoms, 3): the analytic "
          "V_xc force. Holds the grid fixed (neglects the Becke weight "
          "derivative, like the molecular xc_pulay_gradient_* kernels). "
          "LDA, GGA sigma-Pulay, and meta-GGA tau-Pulay terms included.");

    m.def("xc_lattice_gradient_contribution_uks",
          &vibeqc::xc_lattice_gradient_contribution_uks,
          py::arg("basis"), py::arg("system"), py::arg("grid"),
          py::arg("functional"), py::arg("density_alpha"),
          py::arg("density_beta"), py::arg("options"),
          py::arg("density_domain") =
              vibeqc::PeriodicXCDensityDomain::AUTO,
          py::call_guard<py::gil_scoped_release>(),
          "Open-shell periodic XC Pulay atomic gradient (n_atoms, 3): "
          "the analytic alpha + beta V_xc force. Holds the grid fixed. "
          "LDA, GGA sigma-Pulay, and meta-GGA tau-Pulay terms included.");

    // Open-shell periodic XC — companion of build_xc_periodic for UKS.
    py::class_<vibeqc::PeriodicUKSXCContribution>(m, "PeriodicUKSXCContribution")
        .def_readonly("V_alpha", &vibeqc::PeriodicUKSXCContribution::V_alpha,
                      "V_xc^α(g) as a LatticeMatrixSet — per-cell α-spin "
                      "XC potential matrices.")
        .def_readonly("V_beta", &vibeqc::PeriodicUKSXCContribution::V_beta,
                      "V_xc^β(g) as a LatticeMatrixSet — per-cell β-spin "
                      "XC potential matrices.")
        .def_readonly("e_xc", &vibeqc::PeriodicUKSXCContribution::e_xc,
                      "Total E_xc per unit cell (Hartree).");

    m.def("build_xc_periodic_uks", &vibeqc::build_xc_periodic_uks,
          py::arg("basis"), py::arg("system"), py::arg("grid"),
          py::arg("functional"), py::arg("density_alpha"),
          py::arg("density_beta"), py::arg("options"),
          py::arg("density_domain") =
              vibeqc::PeriodicXCDensityDomain::AUTO,
          py::call_guard<py::gil_scoped_release>(),
          "Open-shell (UKS) periodic XC: takes per-spin LatticeMatrixSet "
          "densities P_α(g), P_β(g) and returns per-spin V_xc matrices + "
          "total E_xc per unit cell. The Functional must be constructed "
          "with spin=2 (eval_polarised path); spin=1 will throw.");

    py::class_<vibeqc::PeriodicKSOptions>(m, "PeriodicKSOptions")
        .def(py::init<>())
        .def_readwrite("functional",
                       &vibeqc::PeriodicKSOptions::functional)
        .def_readwrite("grid",
                       &vibeqc::PeriodicKSOptions::grid)
        .def_readwrite("max_iter",
                       &vibeqc::PeriodicKSOptions::max_iter)
        .def_readwrite("conv_tol_energy",
                       &vibeqc::PeriodicKSOptions::conv_tol_energy)
        .def_readwrite("conv_tol_grad",
                       &vibeqc::PeriodicKSOptions::conv_tol_grad)
        .def_readwrite("damping",
                       &vibeqc::PeriodicKSOptions::damping)
        .def_readwrite("dynamic_damping",
                       &vibeqc::PeriodicKSOptions::dynamic_damping,
                       "See RHFOptions.dynamic_damping. Default True.")
        .def_readwrite("dynamic_damping_min",
                       &vibeqc::PeriodicKSOptions::dynamic_damping_min)
        .def_readwrite("dynamic_damping_max",
                       &vibeqc::PeriodicKSOptions::dynamic_damping_max)
        .def_readwrite("fock_mixing",
                       &vibeqc::PeriodicKSOptions::fock_mixing,
                       "Kohn-Sham matrix mixing fraction. See "
                       "RHFOptions.fock_mixing.")
        .def_readwrite("use_diis",
                       &vibeqc::PeriodicKSOptions::use_diis)
        .def_readwrite("diis_start_iter",
                       &vibeqc::PeriodicKSOptions::diis_start_iter)
        .def_readwrite("scf_accelerator",
                       &vibeqc::PeriodicKSOptions::scf_accelerator,
                       "SCFAccelerator family. Default EDIIS_DIIS "
                       "(production hybrid).")
        .def_readwrite("ediis_diis_switch_threshold",
                       &vibeqc::PeriodicKSOptions::ediis_diis_switch_threshold)
        .def_readwrite("diis_restart_tau",
                       &vibeqc::PeriodicKSOptions::diis_restart_tau,
                       "R_CDIIS restart parameter τ. See "
                       "RHFOptions.diis_restart_tau. Default 1e-4.")
        .def_readwrite("diis_adaptive_delta",
                       &vibeqc::PeriodicKSOptions::diis_adaptive_delta,
                       "AD_CDIIS depth parameter δ. See "
                       "RHFOptions.diis_adaptive_delta. Default 1e-4.")
        .def_readwrite("diis_subspace_size",
                       &vibeqc::PeriodicKSOptions::diis_subspace_size)
        .def_readwrite("initial_guess",
                       &vibeqc::PeriodicKSOptions::initial_guess)
        .def_readwrite("read_density_k",
                       &vibeqc::PeriodicKSOptions::read_density_k)
        .def_readwrite("level_shift",
                       &vibeqc::PeriodicKSOptions::level_shift,
                       "Saunders-Hillier level shift (Hartree). Default 0.0.")
        .def_readwrite("level_shift_warmup_cycles",
                       &vibeqc::PeriodicKSOptions::level_shift_warmup_cycles,
                       "CRYSTAL-style level-shift warm-up cycles. "
                       "-1 auto, 0 persistent shift, positive values "
                       "request an explicit shifted startup length.")
        .def_readwrite("level_shift_schedule",
                       &vibeqc::PeriodicKSOptions::level_shift_schedule,
                       "Explicit per-iteration shift curve. See "
                       "PeriodicRHFOptions.level_shift_schedule.")
        .def_readwrite("smearing_temperature",
                       &vibeqc::PeriodicKSOptions::smearing_temperature,
                       "Fermi-Dirac smearing temperature (Hartree). Default 0.0.")
        .def_readwrite("quadratic_fallback_iter",
                       &vibeqc::PeriodicKSOptions::quadratic_fallback_iter,
                       "C1c second-order SCF fallback activation iter. "
                       "Default 0 (disabled).")
        .def_readwrite("quadratic_fallback_shift",
                       &vibeqc::PeriodicKSOptions::quadratic_fallback_shift,
                       "C1c shift λ in (ε_a − ε_i + λ). Default 0.1.")
        .def_readwrite("quadratic_fallback_max_step",
                       &vibeqc::PeriodicKSOptions::quadratic_fallback_max_step,
                       "C1c trust-region cap on ‖κ‖ per Newton step. "
                       "Default 0.1.")
        .def_readwrite("lattice_opts",
                       &vibeqc::PeriodicKSOptions::lattice_opts)
        .def_readwrite("use_periodic_becke",
                       &vibeqc::PeriodicKSOptions::use_periodic_becke,
                       "When true (default, since v0.9.x), the DFT "
                       "integration grid uses the periodic Becke "
                       "partition (extends the partition denominator "
                       "over image atoms within becke_image_radius_bohr). "
                       "Reduces exactly to the molecular partition in the "
                       "molecular-limit regime, so it is safe to leave on "
                       "for any periodic calculation. Set false only to "
                       "reproduce v0.8.x numerics — silently wrong on "
                       "tight cells.")
        .def_readwrite("becke_image_radius_bohr",
                       &vibeqc::PeriodicKSOptions::becke_image_radius_bohr,
                       "Radius (bohr) within which image atoms are "
                       "included in the periodic Becke partition. Only "
                       "consulted when use_periodic_becke = true. "
                       "Default 10 bohr.")
        .def_readwrite("dft_plus_u_sites",
                       &vibeqc::PeriodicKSOptions::dft_plus_u_sites,
                       "Internal: list of _HubbardSiteCxx for the Γ-only "
                       "periodic RKS +U path (Increment 4b). Populated by "
                       "the Python run_rks_periodic wrapper; multi-k +U "
                       "support is queued for Increment 4c.")
        .def_readwrite("dft_plus_u_ao_groups",
                       &vibeqc::PeriodicKSOptions::dft_plus_u_ao_groups,
                       "Internal: parallel AO-index lists per site.")
        .def_readwrite("read_density",
                       &vibeqc::PeriodicKSOptions::read_density,
                       "READ restart: closed-shell g=0 cell density (RKS), "
                       "projected onto this cell's basis. See "
                       "PeriodicRHFOptions.read_density.")
        .def_readwrite("read_density_alpha",
                       &vibeqc::PeriodicKSOptions::read_density_alpha,
                       "READ restart (open-shell UKS): per-spin g=0 alpha "
                       "density. See PeriodicRHFOptions.read_density.")
        .def_readwrite("read_density_beta",
                       &vibeqc::PeriodicKSOptions::read_density_beta,
                       "READ restart (open-shell UKS): per-spin g=0 beta "
                       "density. See PeriodicRHFOptions.read_density.")
        .def_readwrite("read_path",
                       &vibeqc::PeriodicKSOptions::read_path,
                       "READ source file (.qvf / .molden) for the periodic "
                       "wrapper; empty restarts from an in-memory read_from.")
        .def_readwrite("atomic_spins",
                       &vibeqc::PeriodicKSOptions::atomic_spins,
                       "ATOMSPIN: per-atom spin seed (list of +1/-1/0 in "
                       "unit-cell atom order) for a broken-symmetry SAD "
                       "start (open-shell UKS). Empty = spin-symmetric.")
        .def_readwrite("spinlock_mode",
                       &vibeqc::PeriodicKSOptions::spinlock_mode,
                       "SPINLOCK mode (open-shell UKS): OFF (default), "
                       "SPIN_SCHEDULE (lock n_alpha-n_beta=spinlock_value for "
                       "spinlock_iterations cycles then release), or "
                       "PATTERN_HOLD (MOM-hold the seeded broken-symmetry "
                       "pattern for spinlock_iterations).")
        .def_readwrite("spinlock_value",
                       &vibeqc::PeriodicKSOptions::spinlock_value,
                       "SPINLOCK SPIN_SCHEDULE: the n_alpha-n_beta to lock "
                       "during the first spinlock_iterations cycles.")
        .def_readwrite("spinlock_iterations",
                       &vibeqc::PeriodicKSOptions::spinlock_iterations,
                       "SPINLOCK: number of initial SCF cycles to hold (both "
                       "modes). 0 = off.")
        .def_readwrite("ecp_primitive_blocks",
                       &vibeqc::PeriodicKSOptions::ecp_primitive_blocks)
        .def_readwrite("ecp_home_centers",
                       &vibeqc::PeriodicKSOptions::ecp_home_centers)
        .def_readwrite("ecp_effective_charges",
                       &vibeqc::PeriodicKSOptions::ecp_effective_charges)
        .def_readwrite("ecp_total_ncore",
                       &vibeqc::PeriodicKSOptions::ecp_total_ncore)
        .def_readwrite("use_davidson",
                       &vibeqc::PeriodicKSOptions::use_davidson,
                       "Phase D3 -- use Davidson for Fock diagonalization. "
                       "Default False.")
        .def_readwrite("davidson",
                       &vibeqc::PeriodicKSOptions::davidson,
                       "DavidsonOptions knobs (only consulted when "
                       "use_davidson=True).")
        .def_readwrite("davidson_min_dim",
                       &vibeqc::PeriodicKSOptions::davidson_min_dim,
                       "Minimum AO basis dimension for Davidson to "
                       "activate. Default 100.");

    py::class_<vibeqc::PeriodicKSResult>(m, "PeriodicKSResult")
        .def_readonly("restart_basis", &vibeqc::PeriodicKSResult::restart_basis)
        .def_readonly("restart_lattice", &vibeqc::PeriodicKSResult::restart_lattice)
        .def_readonly("restart_kpoints", &vibeqc::PeriodicKSResult::restart_kpoints)
        .def_readonly("restart_weights", &vibeqc::PeriodicKSResult::restart_weights)
        .def_readwrite("guess_selection", &vibeqc::PeriodicKSResult::guess_selection)
        .def_readonly("energy",        &vibeqc::PeriodicKSResult::energy)
        .def_readonly("e_electronic",  &vibeqc::PeriodicKSResult::e_electronic)
        .def_readonly("e_coulomb",     &vibeqc::PeriodicKSResult::e_coulomb)
        .def_readonly("e_hf_exchange", &vibeqc::PeriodicKSResult::e_hf_exchange)
        .def_readonly("e_xc",          &vibeqc::PeriodicKSResult::e_xc)
        .def_readonly("e_nuclear",     &vibeqc::PeriodicKSResult::e_nuclear)
        .def_readonly("e_dft_plus_u",  &vibeqc::PeriodicKSResult::e_dft_plus_u,
                       "Dudarev DFT+U contribution per unit cell "
                       "(Hartree). 0 unless dft_plus_u_sites is set.")
        .def_readonly("n_iter",        &vibeqc::PeriodicKSResult::n_iter)
        .def_readonly("converged",     &vibeqc::PeriodicKSResult::converged)
        .def_readonly("mo_energies",   &vibeqc::PeriodicKSResult::mo_energies)
        .def_readonly("mo_coeffs",     &vibeqc::PeriodicKSResult::mo_coeffs)
        .def_readonly("density",       &vibeqc::PeriodicKSResult::density)
        .def_readonly("mo_coeffs_k", &vibeqc::PeriodicKSResult::mo_coeffs_k)
        .def_readonly("occupations_k", &vibeqc::PeriodicKSResult::occupations_k)
        .def_readonly("mo_energies_k", &vibeqc::PeriodicKSResult::mo_energies_k)
        .def_readonly("fock",          &vibeqc::PeriodicKSResult::fock)
        .def_readonly("overlap",       &vibeqc::PeriodicKSResult::overlap)
        .def_readonly("scf_trace",     &vibeqc::PeriodicKSResult::scf_trace)
        .def_readonly("functional",    &vibeqc::PeriodicKSResult::functional)
        .def("__repr__", [](const vibeqc::PeriodicKSResult& r) {
            return std::string{"PeriodicKSResult(energy="}
                   + std::to_string(r.energy)
                   + ", functional=" + r.functional
                   + ", converged=" + (r.converged ? "True" : "False") + ")";
        });

    m.def("run_rks_periodic", &vibeqc::run_rks_periodic,
          py::arg("system"), py::arg("basis"), py::arg("kmesh"),
          py::arg("options") = vibeqc::PeriodicKSOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Multi-k closed-shell Kohn-Sham DFT for a periodic system.");

    // ---- DFT+U (Dudarev) — C++ kernel surface --------------------------
    //
    // ``HubbardSiteCxx`` is the internal POD used by the C++ SCF Fock
    // builder. Users construct ``vibeqc.HubbardSite`` (Python frozen
    // dataclass, eV inputs); the Python wrapper converts to this struct
    // (eV → Hartree, ao_groups precomputed) before calling into C++.
    py::class_<vibeqc::HubbardSiteCxx>(m, "_HubbardSiteCxx",
        "Internal POD for the C++ DFT+U kernel. Mirrors "
        "vibeqc.HubbardSite but stores U_eff in Hartree (the unit "
        "conversion happens at the Python boundary). Not intended for "
        "direct user construction — use vibeqc.HubbardSite.")
        .def(py::init<>())
        .def(py::init([](int atom_index, int l, double U_eff_au) {
                  return vibeqc::HubbardSiteCxx{atom_index, l, U_eff_au};
              }),
              py::arg("atom_index"), py::arg("l"), py::arg("U_eff_au"))
        .def_readwrite("atom_index", &vibeqc::HubbardSiteCxx::atom_index)
        .def_readwrite("l", &vibeqc::HubbardSiteCxx::l)
        .def_readwrite("U_eff_au", &vibeqc::HubbardSiteCxx::U_eff_au);

    m.def("_compute_dft_plus_u_cxx",
          [](const std::vector<vibeqc::HubbardSiteCxx>& sites,
             const std::vector<std::vector<int>>& ao_groups,
             const Eigen::MatrixXd& P,
             const Eigen::MatrixXd& S) {
              auto r = vibeqc::compute_dft_plus_u(sites, ao_groups, P, S);
              return py::make_tuple(r.energy, std::move(r.V));
          },
          py::arg("sites"), py::arg("ao_groups"),
          py::arg("P"), py::arg("S"),
          "Internal: per-spin Dudarev +U energy + AO Fock contribution. "
          "Returns (E_U_sigma, V_U). Spin convention is the caller's "
          "responsibility (pass P_sigma for the per-spin formula).");

    m.def("_compute_dft_plus_u_multi_k_per_spin_cxx",
          [](const std::vector<vibeqc::HubbardSiteCxx>& sites,
             const std::vector<std::vector<int>>& ao_groups,
             const std::vector<Eigen::MatrixXcd>& S_k,
             const std::vector<Eigen::MatrixXcd>& P_sigma_k,
             const std::vector<double>& weights) {
              auto r = vibeqc::compute_dft_plus_u_multi_k_per_spin(
                  sites, ao_groups, S_k, P_sigma_k, weights);
              return py::make_tuple(r.energy_total,
                                     std::move(r.V_AO_per_spin));
          },
          py::arg("sites"), py::arg("ao_groups"),
          py::arg("S_k"), py::arg("P_sigma_k"), py::arg("weights"),
          "Internal: per-spin multi-k Dudarev +U kernel for the "
          "BIPOLE Python SCF drivers (UHF / UKS). Returns "
          "(E_sigma, V_AO_per_spin). Caller multiplies V_AO by S(k) "
          "on both sides per k-point and adds to the per-k per-spin "
          "Fock; caller sums energies over spins for the total.");

    // ---- Dispersion (D3-BJ) --------------------------------------------
    py::class_<vibeqc::D3BJParams>(m, "D3BJParams",
        "Damping parameters of the D3 Becke-Johnson correction.\n\n"
        "s6/s8 scale the C6/C8 two-body terms; a1/a2 are the BJ damping\n"
        "constants. s9 scales the Axilrod-Teller-Muto three-body term and\n"
        "defaults to 0.0 -- published per-functional damping sets are fit\n"
        "two-body-only. Methods defined to include ATM (PBEh-3c, HSE-3c)\n"
        "set s9 = 1.0.")
        .def(py::init<>())
        .def(py::init([](double s6, double s8, double a1, double a2,
                         double s9) {
                  return vibeqc::D3BJParams{s6, s8, a1, a2, s9};
              }),
              py::arg("s6") = 1.0, py::arg("s8") = 0.0,
              py::arg("a1") = 0.0, py::arg("a2") = 0.0,
              py::arg("s9") = 0.0)
        .def_readwrite("s6", &vibeqc::D3BJParams::s6)
        .def_readwrite("s8", &vibeqc::D3BJParams::s8)
        .def_readwrite("a1", &vibeqc::D3BJParams::a1)
        .def_readwrite("a2", &vibeqc::D3BJParams::a2)
        .def_readwrite("s9", &vibeqc::D3BJParams::s9)
        .def("__repr__", [](const vibeqc::D3BJParams& p) {
            return "D3BJParams(s6=" + std::to_string(p.s6)
                 + ", s8=" + std::to_string(p.s8)
                 + ", a1=" + std::to_string(p.a1)
                 + ", a2=" + std::to_string(p.a2)
                 + ", s9=" + std::to_string(p.s9) + ")";
        });

    m.def("d3_r0ab", &vibeqc::d3_r0ab, py::arg("ZA"), py::arg("ZB"),
          "DFT-D3 ab-initio cut-off radius r0ab(ZA, ZB) in bohr, used by "
          "the ATM three-body zero-damping function. NaN outside Z <= 18.");

    py::class_<vibeqc::DispersionResult>(m, "DispersionResult",
        "Dispersion energy and (optional) atomic gradient.")
        .def_readonly("energy", &vibeqc::DispersionResult::energy)
        .def_readonly("gradient", &vibeqc::DispersionResult::gradient);

    m.def("d3bj_params_for",
          [](const std::string& functional) -> py::object {
              auto p = vibeqc::d3bj_params_for(functional);
              if (!p) return py::none();
              return py::cast(*p);
          },
          py::arg("functional"),
          "Look up D3-BJ damping parameters for a DFT functional "
          "(case-insensitive). Returns None if the functional is not "
          "in the shipped parameter table.");

    m.def("compute_d3bj", &vibeqc::compute_d3bj,
          py::arg("molecule"), py::arg("params"),
          py::arg("with_gradient") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Pairwise D3(BJ) dispersion energy (and optionally gradient) "
          "for a molecule given the functional's damping parameters.");

    m.def("d3_coordination_numbers",
          [](const vibeqc::Molecule& mol) {
              return vibeqc::coordination_numbers(mol).cn;
          },
          py::arg("molecule"),
          py::call_guard<py::gil_scoped_release>(),
          "Smooth D3 coordination numbers (Grimme counting function) for "
          "every atom in the molecule.");

    m.def("d3_r2r4", &vibeqc::r2r4, py::arg("Z"),
          "r2r4 atomic ratio sqrt(<r^4>/<r^2>) used in the D3 C8 scaling.");
    m.def("d3_rcov", &vibeqc::rcov, py::arg("Z"),
          "D3 covalent radius in bohr (Pyykko single-bond radius scaled "
          "by k2 = 4/3).");
    m.def("d3_max_supported_Z", &vibeqc::max_supported_Z,
          "Highest atomic number covered by the native D3 tables.");

    // ---- Chai-Head-Gordon dispersion — the ωB97X-D "-D" term ----------
    m.def("compute_chg_dispersion", &vibeqc::compute_chg_dispersion,
          py::arg("molecule"), py::arg("with_gradient") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Chai-Head-Gordon ('CHG' / DFT-D2-type) empirical dispersion "
          "energy — the intrinsic '-D' correction of ωB97X-D. "
          "E = -s6 Σ_{A<B} C6_AB/R^6 · 1/(1 + d·(R/R0_AB)^-12), with "
          "s6 = 1, d = 6, the Grimme-2006 D2 C6 / vdW-radius tables "
          "(H–Xe). Returns a DispersionResult (.energy, .gradient).");
    m.def("chg_max_supported_Z", &vibeqc::chg_max_supported_Z,
          "Highest atomic number in the ωB97X-D / DFT-D2 dispersion "
          "table (54 = Xe).");

    // ---- EEQ atomic-charge model (Caldeweyher 2019) -------------------
    //
    // The algorithmically distinctive piece of D4 over D3: an Apache-2.0
    // port of Grimme group's multicharge::eeq2019 model. See
    // cpp/include/vibeqc/eeq_charges.hpp for the full method
    // description.
    py::class_<vibeqc::EEQResult>(m, "EEQResult",
        "Result of an EEQ charge calculation. ``charges`` is an "
        "ndarray of n atomic partial charges (electrons); "
        "``chemical_potential`` is the Lagrange multiplier λ of the "
        "charge-conservation constraint (== the system chemical "
        "potential).")
        .def_readonly("charges", &vibeqc::EEQResult::charges)
        .def_readonly("chemical_potential",
                       &vibeqc::EEQResult::chemical_potential)
        .def("__repr__", [](const vibeqc::EEQResult& r) {
            return std::string{"EEQResult(n_atoms="}
                   + std::to_string(r.charges.size())
                   + ", chemical_potential="
                   + std::to_string(r.chemical_potential) + ")";
        });

    py::class_<vibeqc::EEQOptions>(m, "EEQOptions",
        "EEQ-CN counting + solver knobs. Defaults match "
        "multicharge::new_eeq2019_model (cn_cutoff = 25.0 bohr, "
        "cn_exp = 7.5, cn_max = 8.0).")
        .def(py::init<>())
        .def_readwrite("cn_cutoff", &vibeqc::EEQOptions::cn_cutoff)
        .def_readwrite("cn_exp",    &vibeqc::EEQOptions::cn_exp)
        .def_readwrite("cn_max",    &vibeqc::EEQOptions::cn_max);

    m.def("eeq_charges", &vibeqc::eeq_charges,
          py::arg("molecule"), py::arg("total_charge") = 0.0,
          py::arg("options") = vibeqc::EEQOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Compute electronegativity-equilibration (EEQ, Caldeweyher "
          "2019) atomic partial charges for a molecule. Returns an "
          "EEQResult. Apache-2.0 port of Grimme group's "
          "multicharge::eeq2019 model. The algorithmically "
          "distinctive piece of D4 dispersion over D3 — full D4 "
          "dispersion energy in vibe-qc's native backend will land "
          "in a follow-on (Phase D4b); for production D4 numbers, "
          "use vibeqc.compute_d4 (which wraps the optional dftd4 "
          "Python package).");

    m.def("eeq_coordination_numbers",
          &vibeqc::eeq_coordination_numbers,
          py::arg("molecule"), py::arg("options") = vibeqc::EEQOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "EEQ-specific coordination numbers (erf counting function, "
          "kcn = 7.5, bare Pyykko covalent radii). Different from "
          "vibeqc.d3_coordination_numbers, which uses the D3 "
          "counting function + 4/3-scaled radii.");

    // ---- Reciprocal-space periodic Poisson (Phase 12e-c-3a) -----------
    //
    // Accept + return a flat numpy array rather than a custom
    // ScalarField3D pybind11 class; the C++-side struct is just a
    // shape + std::vector pair, not worth exposing. Users pass a
    // (nx, ny, nz) numpy array in and get a same-shaped one out.
    auto numpy_to_field = [](py::array_t<double, py::array::c_style | py::array::forcecast> arr)
        -> vibeqc::ScalarField3D {
        if (arr.ndim() != 3) {
            throw std::invalid_argument(
                "solve_poisson: rho must be a 3D numpy array");
        }
        vibeqc::ScalarField3D out;
        out.resize(static_cast<std::size_t>(arr.shape(0)),
                    static_cast<std::size_t>(arr.shape(1)),
                    static_cast<std::size_t>(arr.shape(2)));
        const double* src = arr.data();
        std::copy(src, src + out.data.size(), out.data.begin());
        return out;
    };
    auto field_to_numpy = [](const vibeqc::ScalarField3D& f) -> py::array_t<double> {
        py::array_t<double> out({f.nx, f.ny, f.nz});
        auto buf = out.mutable_unchecked<3>();
        for (std::size_t i = 0; i < f.nx; ++i)
            for (std::size_t j = 0; j < f.ny; ++j)
                for (std::size_t k = 0; k < f.nz; ++k)
                    buf(i, j, k) = f(i, j, k);
        return out;
    };

    m.def("solve_poisson_erf_screened",
          [numpy_to_field, field_to_numpy](
              py::array_t<double, py::array::c_style | py::array::forcecast> rho,
              const Eigen::Matrix3d& lattice,
              double omega) {
              auto rho_f = numpy_to_field(rho);
              auto V = vibeqc::solve_poisson_erf_screened(rho_f, lattice, omega);
              return field_to_numpy(V);
          },
          py::arg("rho"), py::arg("lattice"), py::arg("omega"),
          "Periodic Poisson solver with the erf-screened Coulomb kernel "
          "K(r) = erf(omega r) / r. Uses FFTW3 to apply Ṽ(G) = (4π/G²)"
          " exp(-G²/4ω²) ρ̃(G). Returned V has zero mean (G=0 component "
          "dropped).");

    m.def("solve_poisson_coulomb",
          [numpy_to_field, field_to_numpy](
              py::array_t<double, py::array::c_style | py::array::forcecast> rho,
              const Eigen::Matrix3d& lattice) {
              auto rho_f = numpy_to_field(rho);
              auto V = vibeqc::solve_poisson_coulomb(rho_f, lattice);
              return field_to_numpy(V);
          },
          py::arg("rho"), py::arg("lattice"),
          "Periodic Poisson solver with the unscreened Coulomb kernel "
          "K(r) = 1/r. Ṽ(G) = (4π/G²) ρ̃(G). Returned V has zero mean.");

    m.def("hartree_energy_on_grid",
          [numpy_to_field](
              py::array_t<double, py::array::c_style | py::array::forcecast> rho,
              py::array_t<double, py::array::c_style | py::array::forcecast> V,
              double cell_volume_bohr3) {
              auto rho_f = numpy_to_field(rho);
              auto V_f = numpy_to_field(V);
              return vibeqc::hartree_energy(rho_f, V_f, cell_volume_bohr3);
          },
          py::arg("rho"), py::arg("V"), py::arg("cell_volume_bohr3"),
          "E_H = (1/2) integral rho(r) V(r) dr computed on the grid. "
          "Test utility — most callers compute this themselves.");
    // ================================================================
    // Semiempirical DFTB0 (Stage 1)
    // ================================================================
    py::module_ m_semi = m.def_submodule("semiempirical",
        "Semiempirical tight-binding methods (DFTB0, SCC-DFTB, ...)");

    // Kernel-form enums are registered first: pybind11 resolves an
    // enum-typed default argument at def() time, so any def below that
    // defaults to one would fail at import if these came later.
    // ---- Translation-covariant lattice-summed shell gamma (core) ----------
    py::enum_<vibeqc::semiempirical::ShellGammaForm>(
        m_semi, "ShellGammaForm",
        "Functional form of the second-order charge kernel: Klopman-Ohno "
        "(GFN2 Eq. 21) or Elstner 1998 Eq. 17/18.")
        .value("KlopmanOhno", vibeqc::semiempirical::ShellGammaForm::KlopmanOhno)
        .value("Elstner", vibeqc::semiempirical::ShellGammaForm::Elstner)
        .export_values();
    py::enum_<vibeqc::semiempirical::KlopmanOhnoAverage>(
        m_semi, "KlopmanOhnoAverage")
        .value("HardnessMean",
               vibeqc::semiempirical::KlopmanOhnoAverage::HardnessMean)
        .value("InverseHardnessMean",
               vibeqc::semiempirical::KlopmanOhnoAverage::InverseHardnessMean)
        .export_values();

    // Loud k-route non-convergence (issue #342).  RuntimeError base keeps
    // every existing ``except RuntimeError`` consumer catching.
    py::register_exception<SCFNonConvergenceError>(
        m_semi, "SCFNonConvergenceError", PyExc_RuntimeError);

    py::class_<vibeqc::semiempirical::BroydenMixer>(
        m_semi, "_BroydenMixer",
        "Internal native charge-mixer test surface.")
        .def(py::init<int, double>(), py::arg("memory") = 6,
             py::arg("damping") = 0.4)
        .def(
            "mix",
            [](vibeqc::semiempirical::BroydenMixer& mixer,
               const Eigen::VectorXd& dq_out,
               const Eigen::VectorXd& dq_in,
               int iter) {
                if (dq_out.size() != dq_in.size()) {
                    throw std::invalid_argument(
                        "Broyden mixer input vectors must have equal sizes");
                }
                Eigen::VectorXd mixed = dq_out;
                mixer.mix(mixed, dq_in, iter);
                return mixed;
            },
            py::arg("dq_out"), py::arg("dq_in"), py::arg("iter"))
        .def("reset", &vibeqc::semiempirical::BroydenMixer::reset)
        .def_property_readonly(
            "error", &vibeqc::semiempirical::BroydenMixer::error);

    py::enum_<vibeqc::semiempirical::SemiempiricalNativeMethod>(
        m_semi, "NativeMethod")
        .value("DFTB0", vibeqc::semiempirical::SemiempiricalNativeMethod::DFTB0)
        .value("SCCDFTB", vibeqc::semiempirical::SemiempiricalNativeMethod::SCCDFTB)
        .value("GFN2XTB", vibeqc::semiempirical::SemiempiricalNativeMethod::GFN2XTB)
        .value("PM6", vibeqc::semiempirical::SemiempiricalNativeMethod::PM6)
        .value("OM1", vibeqc::semiempirical::SemiempiricalNativeMethod::OM1)
        .value("OM2", vibeqc::semiempirical::SemiempiricalNativeMethod::OM2)
        .value("OM3", vibeqc::semiempirical::SemiempiricalNativeMethod::OM3)
        .value("MSINDO", vibeqc::semiempirical::SemiempiricalNativeMethod::MSINDO)
        .value("MSINDONDDO",
               vibeqc::semiempirical::SemiempiricalNativeMethod::MSINDONDDO);

    py::enum_<vibeqc::semiempirical::SemiempiricalNativeSpin>(
        m_semi, "NativeSpin")
        .value("ClosedShell",
               vibeqc::semiempirical::SemiempiricalNativeSpin::ClosedShell)
        .value("Unrestricted",
               vibeqc::semiempirical::SemiempiricalNativeSpin::Unrestricted);

    py::class_<vibeqc::semiempirical::SemiempiricalNativeRoute>(
        m_semi, "NativeRoute")
        .def(py::init<>())
        .def_readwrite("method",
                       &vibeqc::semiempirical::SemiempiricalNativeRoute::method)
        .def_readwrite("spin",
                       &vibeqc::semiempirical::SemiempiricalNativeRoute::spin)
        .def_readwrite("max_iter",
                       &vibeqc::semiempirical::SemiempiricalNativeRoute::max_iter)
        .def_readwrite("conv_tol",
                       &vibeqc::semiempirical::SemiempiricalNativeRoute::conv_tol)
        .def_readwrite(
            "charge_mixing",
            &vibeqc::semiempirical::SemiempiricalNativeRoute::charge_mixing);

    py::class_<vibeqc::semiempirical::SemiempiricalNativeResult>(
        m_semi, "NativeResult")
        .def(py::init<>())
        .def_readonly("route",
                      &vibeqc::semiempirical::SemiempiricalNativeResult::route)
        .def_readonly("energy",
                      &vibeqc::semiempirical::SemiempiricalNativeResult::energy)
        .def_readonly(
            "e_electronic",
            &vibeqc::semiempirical::SemiempiricalNativeResult::e_electronic)
        .def_readonly(
            "e_repulsive",
            &vibeqc::semiempirical::SemiempiricalNativeResult::e_repulsive)
        .def_readonly("e_core",
                      &vibeqc::semiempirical::SemiempiricalNativeResult::e_core)
        .def_readonly("e_scc",
                      &vibeqc::semiempirical::SemiempiricalNativeResult::e_scc)
        .def_readonly(
            "binding_energy",
            &vibeqc::semiempirical::SemiempiricalNativeResult::binding_energy)
        .def_readonly("n_basis",
                      &vibeqc::semiempirical::SemiempiricalNativeResult::n_basis)
        .def_readonly("n_iter",
                      &vibeqc::semiempirical::SemiempiricalNativeResult::n_iter)
        .def_readonly(
            "converged",
            &vibeqc::semiempirical::SemiempiricalNativeResult::converged)
        .def_readonly(
            "retained_result_bytes",
            &vibeqc::semiempirical::SemiempiricalNativeResult::retained_result_bytes)
        .def_readonly(
            "workspace_bytes",
            &vibeqc::semiempirical::SemiempiricalNativeResult::workspace_bytes)
        .def_readonly(
            "memory_counters_complete",
            &vibeqc::semiempirical::SemiempiricalNativeResult::memory_counters_complete)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::SemiempiricalNativeResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::SemiempiricalNativeResult& result) {
                return result.parameter_sha256;
            });

        // --- SemiempiricalBasis ---
    py::class_<vibeqc::semiempirical::SemiempiricalBasis>(
        m_semi, "SemiempiricalBasis",
        "Minimal valence STO-NG basis builder.")
        .def_static("build",
             py::overload_cast<const vibeqc::Molecule&, const vibeqc::semiempirical::CoreParameterSet&, int>(
                 &vibeqc::semiempirical::SemiempiricalBasis::build),
             py::arg("mol"), py::arg("params"), py::arg("n_primitives") = 6,
             "Build basis from CoreParameterSet.");


py::class_<vibeqc::semiempirical::SemiempiricalParameters>(
        m_semi, "SemiempiricalParameters",
        "DFTB0 / SCC-DFTB parameter set: on-site energies, STO exponents, "
        "repulsive pair potentials. Use SemiempiricalParameters.dftb0_default() "
        "for the built-in H/C/N/O table.")
        .def(py::init<>())
        .def_static("dftb0_production",&vibeqc::semiempirical::SemiempiricalParameters::dftb0_production,
                    "Production DFTB parameter set. Currently identical to "
                    "dftb0_default(): the spline-fitted production "
                    "repulsives are deferred pending a validated DFT "
                    "reference dataset (issue #306).").def_static("dftb0_default",
                    &vibeqc::semiempirical::SemiempiricalParameters::dftb0_default,
                    "Return the built-in DFTB0 parameter set for H, C, N, O.")
        .def("add_element", [](vibeqc::semiempirical::SemiempiricalParameters& p,
                                  int Z, const std::vector<double>& on_site,
                                  const std::vector<double>& zeta,
                                  double U, int nval) {
            vibeqc::semiempirical::ElementData e;
            e.Z = Z; e.on_site = on_site; e.zeta = zeta;
            e.hubbard_u = U; e.valence_electrons = nval;
            p.add_element(e);
        }, py::arg("Z"), py::arg("on_site"), py::arg("zeta"),
           py::arg("hubbard_u"), py::arg("valence_electrons"),
           "Add an element to the parameter set.")
        .def("has_element",
             &vibeqc::semiempirical::SemiempiricalParameters::has_element,
             py::arg("Z"),
             "True if element Z is in the parameter set.")
        .def("on_site_energy",
             &vibeqc::semiempirical::SemiempiricalParameters::on_site_energy,
             py::arg("Z"), py::arg("l"),
             "On-site energy for element Z, ang.mom. l (Hartree).")
        .def("sto_exponent",
             &vibeqc::semiempirical::SemiempiricalParameters::sto_exponent,
             py::arg("Z"), py::arg("l"),
             "STO exponent for element Z, ang.mom. l (a0^-1).")
        .def("average_on_site",
             &vibeqc::semiempirical::SemiempiricalParameters::average_on_site,
             py::arg("Z"),
             "Average on-site energy for element Z (Hartree).")
        .def("hubbard_u",
             &vibeqc::semiempirical::SemiempiricalParameters::hubbard_u,
             py::arg("Z"),
             "Hubbard U parameter for element Z (Hartree).")
        .def("valence_electrons",
             &vibeqc::semiempirical::SemiempiricalParameters::valence_electrons,
             py::arg("Z"),
             "Neutral-atom valence electron count for element Z.")
        .def("gamma_onsite",
             &vibeqc::semiempirical::SemiempiricalParameters::gamma_onsite,
             py::arg("Z"),
             "On-site gamma = U_A (Hartree).")
        .def("set_repulsive_pair_analytic",
             [](vibeqc::semiempirical::SemiempiricalParameters& p,
                int Z1, int Z2, double A, double B) {
                 vibeqc::semiempirical::RepulsivePairV2 rp;
                 rp.A = A;
                 rp.B = B;
                 p.set_repulsive_pair(Z1, Z2, rp);
             },
             py::arg("Z1"), py::arg("Z2"), py::arg("A"),
             py::arg("B") = 0.0,
             "Set analytic repulsive pair: A/R^12 if B=0, else A*exp(-B*R).")
        .def("set_repulsive_pair_spline",
             [](vibeqc::semiempirical::SemiempiricalParameters& p,
                int Z1, int Z2,
                const std::vector<double>& R, const std::vector<double>& V) {
                 vibeqc::semiempirical::RepulsivePairV2 rp;
                 rp.spline.build(R, V);
                 p.set_repulsive_pair(Z1, Z2, rp);
             },
             py::arg("Z1"), py::arg("Z2"), py::arg("R"), py::arg("V"),
             "Set spline-fitted repulsive pair from knot arrays (R, V). "
             "R and V must be the same length, R strictly increasing.")
        .def("has_repulsive_pair",
             [](const vibeqc::semiempirical::SemiempiricalParameters& p,
                int Z1, int Z2) -> bool {
                 return p.repulsive_pair(Z1, Z2) != nullptr;
             },
             py::arg("Z1"), py::arg("Z2"),
             "True if an explicit repulsive pair is set for (Z1,Z2).")
        .def("repulsive_energy",
             &vibeqc::semiempirical::SemiempiricalParameters::repulsive_energy,
             py::arg("Z1"), py::arg("Z2"), py::arg("R"),
             "Evaluate V_rep(R) for pair (Z1, Z2) at distance R (bohr).")
        .def_property("kappa",
                      &vibeqc::semiempirical::SemiempiricalParameters::kappa,
                      &vibeqc::semiempirical::SemiempiricalParameters::set_kappa,
                      "Wolfsberg-Helmholtz scaling constant (default 1.75).")
        .def("content_sha256",
             &vibeqc::semiempirical::SemiempiricalParameters::content_sha256,
             "Canonical SHA-256 of every stored execution parameter.")
        .def("parameter_identity",
             &vibeqc::semiempirical::SemiempiricalParameters::parameter_identity,
             "Allowlisted built-in identity or custom:<sha256>.")
//         .def("parameter_fingerprint",
//              &vibeqc::semiempirical::SemiempiricalParameters::parameter_fingerprint,
;  // parameter_fingerprint removed: v0.15.116 compat

    py::class_<vibeqc::semiempirical::DFTB0Result>(
        m_semi, "DFTB0Result",
        "Result of a non-self-consistent DFTB0 energy calculation.")
        .def(py::init<>())
        .def_readonly("energy",
                      &vibeqc::semiempirical::DFTB0Result::energy)
        .def_readonly("e_electronic",
                      &vibeqc::semiempirical::DFTB0Result::e_electronic)
        .def_readonly("e_repulsive",
                      &vibeqc::semiempirical::DFTB0Result::e_repulsive)
        .def_readonly("mo_energies",
                      &vibeqc::semiempirical::DFTB0Result::mo_energies)
        .def_readonly("mo_coeffs",
                      &vibeqc::semiempirical::DFTB0Result::mo_coeffs)
        .def_readonly("density",
                      &vibeqc::semiempirical::DFTB0Result::density)
        .def_readonly("overlap",
                      &vibeqc::semiempirical::DFTB0Result::overlap)
        .def_readonly("hamiltonian",
                      &vibeqc::semiempirical::DFTB0Result::hamiltonian)
        .def_readonly("charges",
                      &vibeqc::semiempirical::DFTB0Result::charges)
        .def_readonly("n_basis",
                      &vibeqc::semiempirical::DFTB0Result::n_basis)
        .def_readonly("n_occ",
                      &vibeqc::semiempirical::DFTB0Result::n_occ)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::DFTB0Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::DFTB0Result& result) {
                return result.parameter_sha256;
            });

    py::class_<vibeqc::semiempirical::seccm::DFTB0SECCMResult>(
        m_semi, "DFTB0SECCMResult",
        "Internal result of frozen-topology DFTB0-SECCM assembly.")
        .def_readonly(
            "energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::energy)
        .def_readonly(
            "electronic_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::electronic_energy)
        .def_readonly(
            "repulsive_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::repulsive_energy)
        .def_readonly(
            "long_range_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::long_range_energy)
        .def_readonly(
            "dispersion_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::dispersion_energy)
        .def_readonly(
            "specific_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::specific_energy)
        .def_readonly(
            "total_cyclic_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::total_cyclic_energy)
        .def_readonly(
            "cyclic_electronic_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::cyclic_electronic_energy)
        .def_readonly(
            "cyclic_repulsive_energy",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::cyclic_repulsive_energy)
        .def_readonly(
            "homo_lumo_gap",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::homo_lumo_gap)
        .def_readonly(
            "mo_energies",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::mo_energies)
        .def_readonly(
            "mo_coeffs",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::mo_coeffs)
        .def_readonly(
            "density",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::density)
        .def_readonly(
            "overlap",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::overlap)
        .def_readonly(
            "hamiltonian",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::hamiltonian)
        .def_readonly(
            "gradient",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::gradient)
        .def_readonly(
            "has_gradient",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::has_gradient)
        .def_readonly(
            "n_basis",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::n_basis)
        .def_readonly(
            "n_occ",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::n_occ)
        .def_readonly(
            "group_order",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::group_order)
        .def_readonly(
            "n_records",
            &vibeqc::semiempirical::seccm::DFTB0SECCMResult::n_records)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::seccm::DFTB0SECCMResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::seccm::DFTB0SECCMResult& result) {
                return result.parameter_sha256;
            });

    py::class_<vibeqc::semiempirical::seccm::SCCDFTBSECCMResult>(
        m_semi, "SCCDFTBSECCMResult",
        "Internal result of frozen-topology SCC-DFTB-SECCM assembly.")
        .def_readonly(
            "energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::energy)
        .def_readonly(
            "e_electronic",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::e_electronic)
        .def_readonly(
            "e_repulsive",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::e_repulsive)
        .def_readonly(
            "e_scc",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::e_scc)
        .def_readonly(
            "e_madelung",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::e_madelung)
        .def_readonly(
            "total_cyclic_energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::total_cyclic_energy)
        .def_readonly(
            "cyclic_electronic_energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::cyclic_electronic_energy)
        .def_readonly(
            "cyclic_repulsive_energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::cyclic_repulsive_energy)
        .def_readonly(
            "cyclic_scc_energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::cyclic_scc_energy)
        .def_readonly(
            "cyclic_madelung_energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::cyclic_madelung_energy)
        .def_readonly(
            "homo_lumo_gap",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::homo_lumo_gap)
        .def_readonly(
            "finite_torus_gap_tolerance",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::
                finite_torus_gap_tolerance)
        .def_readonly(
            "gap_guard_waived",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::gap_guard_waived)
        .def_readonly(
            "occupations",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::occupations)
        .def_readonly(
            "aufbau_occupation_deviation",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::
                aufbau_occupation_deviation)
        .def_readonly(
            "aufbau_occupation_tolerance",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::
                aufbau_occupation_tolerance)
        .def_readonly(
            "aufbau_occupation",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::
                aufbau_occupation)
        .def_readonly(
            "free_energy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::free_energy)
        .def_readonly(
            "entropy",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::entropy)
        .def_readonly(
            "smearing_temperature",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::smearing_temperature)
        .def_readonly(
            "mo_energies",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::mo_energies)
        .def_readonly(
            "mo_coeffs",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::mo_coeffs)
        .def_readonly(
            "density",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::density)
        .def_readonly(
            "overlap",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::overlap)
        .def_readonly(
            "hamiltonian",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::hamiltonian)
        .def_readonly(
            "charges",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::charges)
        .def_readonly(
            "gradient",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::gradient)
        .def_readonly(
            "has_gradient",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::has_gradient)
        .def_readonly(
            "n_basis",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::n_basis)
        .def_readonly(
            "n_occ",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::n_occ)
        .def_readonly(
            "n_iter",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::n_iter)
        .def_readonly(
            "group_order",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::group_order)
        .def_readonly(
            "n_records",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::n_records)
        .def_readonly(
            "converged",
            &vibeqc::semiempirical::seccm::SCCDFTBSECCMResult::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::seccm::SCCDFTBSECCMResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::seccm::SCCDFTBSECCMResult& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "run_dftb0",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_dftb0(mol, snapshot);
                });
        },
        py::arg("mol"), py::arg("params"),
        "Run a non-self-consistent DFTB0 energy calculation.");

    m_semi.def(
        "_run_dftb0_seccm_from_records",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const Eigen::Ref<const Eigen::MatrixXd>& translations,
           const std::vector<int>& central,
           const std::vector<int>& origin,
           const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
           const std::vector<double>& weights,
           const std::vector<int>& multiplicities,
           const Eigen::Ref<const Eigen::MatrixXd>& displacements,
           const Eigen::Ref<const Eigen::MatrixXd>& primitive_vectors,
           const std::vector<int>& replicas,
           double geometry_tolerance,
           bool compute_gradient) {
            const auto validated = seccm_validated_topology_from_records(
                    mol,
                    translations,
                    central,
                    origin,
                    shell_labels,
                    weights,
                    multiplicities,
                    displacements,
                    primitive_vectors,
                    replicas,
                    geometry_tolerance,
                    "DFTB0-SECCM");
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::seccm::run_dftb0_seccm(
                        mol,
                        snapshot,
                        validated.topology,
                        validated.group_order,
                        geometry_tolerance,
                        1.0e-10,
                        1.0e-8,
                        compute_gradient);
                });
        },
        py::arg("mol"),
        py::arg("params"),
        py::arg("translations"),
        py::arg("central"),
        py::arg("origin"),
        py::arg("shell_labels"),
        py::arg("weights"),
        py::arg("multiplicities"),
        py::arg("displacements"),
        py::arg("primitive_vectors"),
        py::arg("replicas"),
        py::arg("geometry_tolerance"),
        py::arg("compute_gradient") = false,
        "Internal DFTB0-SECCM frozen-record adapter.");

    m_semi.def(
        "_run_scc_dftb_seccm_from_records",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const Eigen::Ref<const Eigen::MatrixXd>& translations,
           const std::vector<int>& central,
           const std::vector<int>& origin,
           const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
           const std::vector<double>& weights,
           const std::vector<int>& multiplicities,
           const Eigen::Ref<const Eigen::MatrixXd>& displacements,
           const Eigen::Ref<const Eigen::MatrixXd>& primitive_vectors,
           const std::vector<int>& replicas,
           double geometry_tolerance,
           int max_iter,
           double conv_tol_charge,
           double charge_mixing,
           bool madelung,
           vibeqc::semiempirical::ShellGammaForm gamma_form,
           bool ewald_gamma,
           bool use_diis,
           int diis_subspace,
           bool use_broyden,
           int broyden_memory,
           double broyden_damping,
           double electronic_temperature,
           bool compute_gradient) {
            const auto validated = seccm_validated_topology_from_records(
                    mol,
                    translations,
                    central,
                    origin,
                    shell_labels,
                    weights,
                    multiplicities,
                    displacements,
                    primitive_vectors,
                    replicas,
                    geometry_tolerance,
                    "SCC-DFTB-SECCM");
            const std::array<int, 3> replica_counts{
                replicas[0], replicas[1], replicas[2]};
            vibeqc::semiempirical::seccm::SCCDFTBSECCMOptions options;
            options.max_iter = max_iter;
            options.conv_tol_charge = conv_tol_charge;
            options.charge_mixing = charge_mixing;
            options.madelung = madelung;
            options.gamma_form = gamma_form;
            options.ewald_gamma = ewald_gamma;
            options.use_diis = use_diis;
            options.diis_subspace = diis_subspace;
            options.use_broyden = use_broyden;
            options.broyden_memory = broyden_memory;
            options.broyden_damping = broyden_damping;
            options.electronic_temperature = electronic_temperature;
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::seccm::run_scc_dftb_seccm(
                        mol,
                        snapshot,
                        validated.topology,
                        validated.group_order,
                        replica_counts,
                        geometry_tolerance,
                        options,
                        1.0e-10,
                        1.0e-8,
                        compute_gradient);
                });
        },
        py::arg("mol"),
        py::arg("params"),
        py::arg("translations"),
        py::arg("central"),
        py::arg("origin"),
        py::arg("shell_labels"),
        py::arg("weights"),
        py::arg("multiplicities"),
        py::arg("displacements"),
        py::arg("primitive_vectors"),
        py::arg("replicas"),
        py::arg("geometry_tolerance"),
        py::arg("max_iter") = 500,
        py::arg("conv_tol_charge") = 1.0e-8,
        py::arg("charge_mixing") = 0.2,
        py::arg("madelung") = false,
        py::arg("gamma_form") =
            vibeqc::semiempirical::ShellGammaForm::KlopmanOhno,
        py::arg("ewald_gamma") = false,
        py::arg("use_diis") = false,
        py::arg("diis_subspace") = 6,
        py::arg("use_broyden") = false,
        py::arg("broyden_memory") = 6,
        py::arg("broyden_damping") = 0.4,
        py::arg("electronic_temperature") = 0.0,
        py::arg("compute_gradient") = false,
        "Internal SCC-DFTB-SECCM frozen-record adapter.");

    py::class_<vibeqc::semiempirical::seccm::PM6SECCMResult>(
        m_semi, "PM6SECCMResult",
        "Internal result of frozen-topology PM6-SECCM assembly.")
        .def_readonly(
            "energy",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::energy)
        .def_readonly(
            "e_electronic",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::e_electronic)
        .def_readonly(
            "e_core",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::e_core)
        .def_readonly(
            "total_cyclic_energy",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::total_cyclic_energy)
        .def_readonly(
            "cyclic_core_energy",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::cyclic_core_energy)
        .def_readonly(
            "e_madelung",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::e_madelung)
        .def_readonly(
            "cyclic_madelung_energy",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::cyclic_madelung_energy)
        .def_readonly(
            "homo_lumo_gap",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::homo_lumo_gap)
        .def_readonly(
            "mo_energies",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::mo_energies)
        .def_readonly(
            "mo_coeffs",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::mo_coeffs)
        .def_readonly(
            "density",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::density)
        .def_readonly(
            "n_basis",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::n_basis)
        .def_readonly(
            "n_occ",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::n_occ)
        .def_readonly(
            "n_iter",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::n_iter)
        .def_readonly(
            "group_order",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::group_order)
        .def_readonly(
            "n_records",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::n_records)
        .def_readonly(
            "converged",
            &vibeqc::semiempirical::seccm::PM6SECCMResult::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::seccm::PM6SECCMResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::seccm::PM6SECCMResult& result) {
                return result.parameter_sha256;
            });

    py::class_<vibeqc::semiempirical::seccm::OMxSECCMResult>(
        m_semi, "OMxSECCMResult",
        "Internal result of the WS-weighted OMx-SECCM adapter.")
        .def_readonly(
            "energy",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::energy)
        .def_readonly(
            "e_electronic",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::e_electronic)
        .def_readonly(
            "e_core",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::e_core)
        .def_readonly(
            "total_cyclic_energy",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::total_cyclic_energy)
        .def_readonly(
            "cyclic_core_energy",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::cyclic_core_energy)
        .def_readonly(
            "homo_lumo_gap",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::homo_lumo_gap)
        .def_readonly(
            "mo_energies",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::mo_energies)
        .def_readonly(
            "mo_coeffs",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::mo_coeffs)
        .def_readonly(
            "density",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::density)
        .def_readonly(
            "n_basis",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::n_basis)
        .def_readonly(
            "n_occ",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::n_occ)
        .def_readonly(
            "n_iter",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::n_iter)
        .def_readonly(
            "group_order",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::group_order)
        .def_readonly(
            "n_records",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::n_records)
        .def_readonly(
            "three_center_weighting",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::
                three_center_weighting)
        .def_readonly(
            "converged",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::converged)
        .def_readonly(
            "truncated_electrostatics_acknowledged",
            &vibeqc::semiempirical::seccm::OMxSECCMResult::
                truncated_electrostatics_acknowledged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::seccm::OMxSECCMResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::seccm::OMxSECCMResult& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "_omx_seccm_three_center_endpoint_weight",
        [](const std::string& weighting, double x_mu_c, double x_nu_c) {
            return vibeqc::semiempirical::seccm::
                omx_three_center_endpoint_weight(
                    vibeqc::semiempirical::seccm::
                        parse_omx_three_center_weighting(weighting),
                    x_mu_c,
                    x_nu_c);
        },
        py::arg("three_center_weighting"),
        py::arg("x_mu_c"),
        py::arg("x_nu_c"),
        "Internal equation-level seam for OMx-SECCM three-center weights.");

    // Registered before the GFN2-SECCM adapter bindings (its
    // _run_gfn2_seccm_from_records default uses the enum).
    py::enum_<vibeqc::semiempirical::SCCMixer>(m_semi, "SCCMixer")
        .value("Simple", vibeqc::semiempirical::SCCMixer::Simple)
        .value("DIIS", vibeqc::semiempirical::SCCMixer::DIIS)
        .value("Broyden", vibeqc::semiempirical::SCCMixer::Broyden)
        .value(
            "BroydenEyert", vibeqc::semiempirical::SCCMixer::BroydenEyert)
        .value("Newton", vibeqc::semiempirical::SCCMixer::Newton)
        .export_values();

    py::class_<vibeqc::semiempirical::xtb::GFN2SCCAttempt>(
        m_semi, "GFN2SCCAttempt",
        "Internal immutable-by-convention record of one bounded GFN2 SCC "
        "attempt.")
        .def_readonly(
            "solver",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::solver)
        .def_readonly(
            "allocated_max_iter",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::allocated_max_iter)
        .def_readonly(
            "n_iter",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::n_iter)
        .def_readonly(
            "ladder_charge_mixing",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::
                ladder_charge_mixing)
        .def_readonly(
            "electronic_temperature",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::
                electronic_temperature)
        .def_readonly(
            "scc_mixer",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::scc_mixer)
        .def_readonly(
            "restart_supplied",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::restart_supplied)
        .def_readonly(
            "molecular_delegated",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::
                molecular_delegated)
        .def_readonly(
            "exit_reason",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::exit_reason)
        .def_readonly(
            "scc_converged",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::scc_converged)
        .def_readonly(
            "physical_basin",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::physical_basin)
        .def_readonly(
            "max_change_trace",
            &vibeqc::semiempirical::xtb::GFN2SCCAttempt::max_change_trace);

    py::class_<vibeqc::semiempirical::seccm::GFN2SECCMResult>(
        m_semi, "GFN2SECCMResult",
        "Internal result of the WS-weighted GFN2-SECCM adapter.")
        .def_readonly(
            "energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::energy)
        .def_readonly(
            "free_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::free_energy)
        .def_readonly(
            "entropy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::entropy)
        .def_readonly(
            "smearing_temperature",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::smearing_temperature)
        .def_readonly(
            "e_electronic",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_electronic)
        .def_readonly(
            "e_repulsive",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_repulsive)
        .def_readonly(
            "e_scc",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_scc)
        .def_readonly(
            "e_band0",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_band0)
        .def_readonly(
            "e_aes",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_aes)
        .def_readonly(
            "e_3rd",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_3rd)
        .def_readonly(
            "e_madelung",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::e_madelung)
        .def_readonly(
            "total_cyclic_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::total_cyclic_energy)
        .def_readonly(
            "cyclic_electronic_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_electronic_energy)
        .def_readonly(
            "cyclic_repulsive_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_repulsive_energy)
        .def_readonly(
            "cyclic_scc_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_scc_energy)
        .def_readonly(
            "cyclic_aes_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_aes_energy)
        .def_readonly(
            "cyclic_3rd_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_3rd_energy)
        .def_readonly(
            "cyclic_band0_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_band0_energy)
        .def_readonly(
            "cyclic_madelung_energy",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::cyclic_madelung_energy)
        .def_readonly(
            "homo_lumo_gap",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::homo_lumo_gap)
        .def_readonly(
            "gap_guard_waived",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::gap_guard_waived)
        .def_readonly(
            "mo_energies",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::mo_energies)
        .def_readonly(
            "mo_coeffs",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::mo_coeffs)
        .def_readonly(
            "density",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::density)
        .def_readonly(
            "overlap",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::overlap)
        .def_readonly(
            "hamiltonian",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::hamiltonian)
        .def_readonly(
            "charges",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::charges)
        .def_readonly(
            "shell_charges",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::shell_charges)
        .def_readonly(
            "scc_max_change_trace",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::scc_max_change_trace)
        .def_readonly(
            "n_basis",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::n_basis)
        .def_readonly(
            "n_occ",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::n_occ)
        .def_readonly(
            "n_iter",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::n_iter)
        .def_readonly(
            "group_order",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::group_order)
        .def_readonly(
            "n_records",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::n_records)
        .def_readonly(
            "converged",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::converged)
        .def_readonly(
            "attempts",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::attempts)
        .def_readonly(
            "selected_attempt_index",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                selected_attempt_index)
        .def_readonly(
            "requested_electronic_temperature",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_electronic_temperature)
        .def_readonly(
            "requested_madelung",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_madelung)
        .def_readonly(
            "requested_madelung_s_weighted",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_madelung_s_weighted)
        .def_readonly(
            "requested_madelung_no_self",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_madelung_no_self)
        .def_readonly(
            "requested_ewald_gamma",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_ewald_gamma)
        .def_readonly(
            "requested_include_aes",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_include_aes)
        .def_readonly(
            "requested_ewald_gamma_molecular_onsite",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_ewald_gamma_molecular_onsite)
        .def_readonly(
            "requested_max_iter",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_max_iter)
        .def_readonly(
            "requested_conv_tol_charge",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_conv_tol_charge)
        .def_readonly(
            "requested_charge_mixing",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_charge_mixing)
        .def_readonly(
            "requested_scc_mixer",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_scc_mixer)
        .def_readonly(
            "requested_ewald_gamma_k0_global",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_ewald_gamma_k0_global)
        .def_readonly(
            "requested_finite_torus_gap_tolerance",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_finite_torus_gap_tolerance)
        .def_readonly(
            "requested_initial_shell_charges",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::
                requested_initial_shell_charges)
        .def_readonly(
            "molecular_delegated",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::molecular_delegated)
        .def_readonly(
            "physical_basin",
            &vibeqc::semiempirical::seccm::GFN2SECCMResult::physical_basin)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::seccm::GFN2SECCMResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::seccm::GFN2SECCMResult& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "_gfn2_seccm_local_charge_state_is_physical",
        &vibeqc::semiempirical::seccm::
            gfn2_seccm_local_charge_state_is_physical,
        py::arg("dq_atom"),
        py::arg("dq_shell"),
        "Internal native predicate for localized GFN2-SECCM charge states.");

    m_semi.def(
        "_gfn2_seccm_ewald_shell_gamma_from_records",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const Eigen::Ref<const Eigen::MatrixXd>& translations,
           const std::vector<int>& central,
           const std::vector<int>& origin,
           const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
           const std::vector<double>& weights,
           const std::vector<int>& multiplicities,
           const Eigen::Ref<const Eigen::MatrixXd>& displacements,
           double alpha,
           bool molecular_onsite,
           bool k0_global,
           vibeqc::semiempirical::ShellGammaForm form) {
            if (!std::isfinite(alpha) || !(alpha > 0.0)) {
                throw std::invalid_argument(
                    "GFN2-SECCM Ewald test kernel requires alpha > 0");
            }
            const vibeqc::semiempirical::seccm::WSTopology topology =
                seccm_topology_from_records(
                    mol,
                    translations,
                    central,
                    origin,
                    shell_labels,
                    weights,
                    multiplicities,
                    displacements);
            const auto basis =
                vibeqc::semiempirical::SemiempiricalBasis::build(
                    mol, params, 0);
            const auto shell_info =
                vibeqc::semiempirical::gfn2_enumerate_shells(
                    basis, mol, params);
            std::vector<Eigen::Vector3d> atom_coords;
            atom_coords.reserve(mol.atoms().size());
            for (const auto& atom : mol.atoms()) {
                atom_coords.emplace_back(atom.xyz[0], atom.xyz[1], atom.xyz[2]);
            }
            const std::size_t dimension = topology.translations.size();
            if (dimension < 1 || dimension > 3) {
                throw std::invalid_argument(
                    "GFN2-SECCM Ewald test kernel requires a 1-D, 2-D, or "
                    "3-D topology");
            }
            (void)k0_global;  // deprecated no-op, see GFN2SECCMOptions
            return vibeqc::semiempirical::seccm::seccm_shell_gamma(
                shell_info,
                atom_coords,
                topology,
                form,
                molecular_onsite,
                alpha);
        },
        py::arg("mol"),
        py::arg("params"),
        py::arg("translations"),
        py::arg("central"),
        py::arg("origin"),
        py::arg("shell_labels"),
        py::arg("weights"),
        py::arg("multiplicities"),
        py::arg("displacements"),
        py::arg("alpha"),
        py::arg("molecular_onsite") = false,
        py::arg("k0_global") = false,
        py::arg("form") = vibeqc::semiempirical::ShellGammaForm::KlopmanOhno,
        "Internal production-kernel seam for GFN2-SECCM Ewald tests.");

    m_semi.def(
        "_run_pm6_seccm_from_records",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           const Eigen::Ref<const Eigen::MatrixXd>& translations,
           const std::vector<int>& central,
           const std::vector<int>& origin,
           const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
           const std::vector<double>& weights,
           const std::vector<int>& multiplicities,
           const Eigen::Ref<const Eigen::MatrixXd>& displacements,
           const Eigen::Ref<const Eigen::MatrixXd>& primitive_vectors,
           const std::vector<int>& replicas,
           double geometry_tolerance,
           int max_iter,
           double conv_tol,
           bool madelung,
           bool allow_truncated_electrostatics) {
            const auto validated = seccm_validated_topology_from_records(
                    mol,
                    translations,
                    central,
                    origin,
                    shell_labels,
                    weights,
                    multiplicities,
                    displacements,
                    primitive_vectors,
                    replicas,
                    geometry_tolerance,
                    "PM6-SECCM");
            vibeqc::semiempirical::seccm::PM6SECCMOptions options;
            options.max_iter = max_iter;
            options.conv_tol = conv_tol;
            options.madelung = madelung;
            options.allow_truncated_electrostatics =
                allow_truncated_electrostatics;
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::seccm::run_pm6_seccm(
                        mol,
                        snapshot,
                        validated.topology,
                        validated.group_order,
                        geometry_tolerance,
                        options,
                        1.0e-8);
                });
        },
        py::arg("mol"),
        py::arg("params"),
        py::arg("translations"),
        py::arg("central"),
        py::arg("origin"),
        py::arg("shell_labels"),
        py::arg("weights"),
        py::arg("multiplicities"),
        py::arg("displacements"),
        py::arg("primitive_vectors"),
        py::arg("replicas"),
        py::arg("geometry_tolerance"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1.0e-7,
        py::arg("madelung") = false,
        py::arg("allow_truncated_electrostatics") = false,
        "Internal PM6-SECCM frozen-record adapter.");

    m_semi.def(
        "_run_omx_seccm_from_records",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           const Eigen::Ref<const Eigen::MatrixXd>& translations,
           const std::vector<int>& central,
           const std::vector<int>& origin,
           const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
           const std::vector<double>& weights,
           const std::vector<int>& multiplicities,
           const Eigen::Ref<const Eigen::MatrixXd>& displacements,
           const Eigen::Ref<const Eigen::MatrixXd>& primitive_vectors,
           const std::vector<int>& replicas,
           double geometry_tolerance,
           int max_iter,
           double conv_tol,
           const std::string& three_center_weighting,
           bool allow_truncated_electrostatics) {
            const auto validated = seccm_validated_topology_from_records(
                    mol,
                    translations,
                    central,
                    origin,
                    shell_labels,
                    weights,
                    multiplicities,
                    displacements,
                    primitive_vectors,
                    replicas,
                    geometry_tolerance,
                    "OMx-SECCM");
            vibeqc::semiempirical::seccm::OMxSECCMOptions options;
            options.max_iter = max_iter;
            options.conv_tol = conv_tol;
            options.three_center_weighting =
                vibeqc::semiempirical::seccm::parse_omx_three_center_weighting(
                    three_center_weighting);
            options.allow_truncated_electrostatics =
                allow_truncated_electrostatics;
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::seccm::run_omx_seccm(
                        mol,
                        snapshot,
                        validated.topology,
                        validated.group_order,
                        geometry_tolerance,
                        options,
                        1.0e-8);
                });
        },
        py::arg("mol"),
        py::arg("params"),
        py::arg("translations"),
        py::arg("central"),
        py::arg("origin"),
        py::arg("shell_labels"),
        py::arg("weights"),
        py::arg("multiplicities"),
        py::arg("displacements"),
        py::arg("primitive_vectors"),
        py::arg("replicas"),
        py::arg("geometry_tolerance"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1.0e-7,
        py::arg("three_center_weighting") = "peintinger_eq13",
        py::arg("allow_truncated_electrostatics") = false,
        "Internal WS-weighted OMx-SECCM frozen-record adapter.");

    m_semi.def(
        "_run_gfn2_seccm_from_records",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const Eigen::Ref<const Eigen::MatrixXd>& translations,
           const std::vector<int>& central,
           const std::vector<int>& origin,
           const Eigen::Ref<const Eigen::MatrixXi>& shell_labels,
           const std::vector<double>& weights,
           const std::vector<int>& multiplicities,
           const Eigen::Ref<const Eigen::MatrixXd>& displacements,
           const Eigen::Ref<const Eigen::MatrixXd>& primitive_vectors,
           const std::vector<int>& replicas,
           double geometry_tolerance,
           int max_iter,
           double conv_tol_charge,
           double charge_mixing,
           double electronic_temperature,
           bool madelung,
           bool madelung_s_weighted,
           bool madelung_no_self,
           bool ewald_gamma,
           bool include_aes,
           bool ewald_gamma_molecular_onsite,
           bool ewald_gamma_k0_global,
           vibeqc::semiempirical::ShellGammaForm gamma_form,
           bool aes_faithful,
           double aes_damping,
           vibeqc::semiempirical::SCCMixer scc_mixer,
           const Eigen::Ref<const Eigen::VectorXd>& initial_shell_charges) {
            if (initial_shell_charges.size() != 0) {
                if (!initial_shell_charges.allFinite()) {
                    throw std::invalid_argument(
                        "GFN2-SECCM initial_shell_charges must contain only "
                        "finite values");
                }
                const Eigen::Index expected_shell_charges =
                    expected_gfn2_shell_charge_count(mol, params);
                if (expected_shell_charges >= 0) {
                    if (initial_shell_charges.size()
                        != expected_shell_charges) {
                        throw std::invalid_argument(
                            "GFN2-SECCM initial_shell_charges must be empty "
                            "or have exactly n_shells entries");
                    }
                    if (!madelung && include_aes
                        && gfn2_seccm_records_are_trivial_molecular(
                            central,
                            origin,
                            shell_labels,
                            weights,
                            multiplicities,
                            displacements)) {
                        throw std::invalid_argument(
                            "GFN2-SECCM initial_shell_charges are unavailable "
                            "on a molecular-delegation topology because the "
                            "molecular driver cannot consume a shell-charge "
                            "restart");
                    }
                }
            }
            const auto validated = seccm_validated_topology_from_records(
                    mol,
                    translations,
                    central,
                    origin,
                    shell_labels,
                    weights,
                    multiplicities,
                    displacements,
                    primitive_vectors,
                    replicas,
                    geometry_tolerance,
                    "GFN2-SECCM");
            vibeqc::semiempirical::seccm::GFN2SECCMOptions options;
            options.max_iter = max_iter;
            options.conv_tol_charge = conv_tol_charge;
            options.charge_mixing = charge_mixing;
            options.electronic_temperature = electronic_temperature;
            options.madelung = madelung;
            options.madelung_s_weighted = madelung_s_weighted;
            options.madelung_no_self = madelung_no_self;
            options.ewald_gamma = ewald_gamma;
            options.include_aes = include_aes;
            options.ewald_gamma_molecular_onsite =
                ewald_gamma_molecular_onsite;
            options.ewald_gamma_k0_global = ewald_gamma_k0_global;
            options.gamma_form = gamma_form;
            options.aes_faithful = aes_faithful;
            options.aes_damping = aes_damping;
            options.scc_mixer = scc_mixer;
            options.initial_shell_charges = initial_shell_charges;
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::seccm::run_gfn2_seccm(
                        mol,
                        snapshot,
                        validated.topology,
                        validated.group_order,
                        geometry_tolerance,
                        options,
                        1.0e-8);
                });
        },
        py::arg("mol"),
        py::arg("params"),
        py::arg("translations"),
        py::arg("central"),
        py::arg("origin"),
        py::arg("shell_labels"),
        py::arg("weights"),
        py::arg("multiplicities"),
        py::arg("displacements"),
        py::arg("primitive_vectors"),
        py::arg("replicas"),
        py::arg("geometry_tolerance"),
        py::arg("max_iter") = 3600,
        py::arg("conv_tol_charge") = 1.0e-6,
        py::arg("charge_mixing") = 0.1,
        py::arg("electronic_temperature") = 0.0,
        py::arg("madelung") = false,
        py::arg("madelung_s_weighted") = false,
        py::arg("madelung_no_self") = false,
        py::arg("ewald_gamma") = false,
        py::arg("include_aes") = true,
        py::arg("ewald_gamma_molecular_onsite") = false,
        py::arg("ewald_gamma_k0_global") = false,
        py::arg("gamma_form") =
            vibeqc::semiempirical::ShellGammaForm::KlopmanOhno,
        py::arg("aes_faithful") = false,
        py::arg("aes_damping") = 0.5,
        py::arg("scc_mixer") = vibeqc::semiempirical::SCCMixer::Simple,
        py::arg("initial_shell_charges") = Eigen::VectorXd(),
        "Internal WS-weighted GFN2-SECCM frozen-record adapter.");



    m_semi.def(
        "compute_dftb0_gradient",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::DFTB0Result& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_matching_parameter_snapshot(
                params, result, "DFTB0 gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_dftb0_gradient(
                        mol, result, snapshot);
                });
        },
        py::arg("mol"), py::arg("result"), py::arg("params"),
        "Compute the analytic nuclear gradient for a DFTB0 result. "
        "Returns (n_atoms, 3) matrix in Hartree/bohr.");


    m_semi.def(
        "compute_scc_dftb_gradient_response",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SCCDFTBResult& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_matching_parameter_snapshot(
                params, result, "SCC-DFTB response gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::
                        compute_scc_dftb_gradient_response(
                            mol, result, snapshot);
                });
        },
        py::arg("mol"), py::arg("result"), py::arg("params"),
        "SCC-DFTB gradient with CP-SCC charge-response correction.");

    m_semi.def(
        "compute_scc_dftb_gradient",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SCCDFTBResult& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_matching_parameter_snapshot(
                params, result, "SCC-DFTB gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_scc_dftb_gradient(
                        mol, result, snapshot);
                });
        },
        py::arg("mol"), py::arg("result"), py::arg("params"),
        "Compute the analytic nuclear gradient for an SCC-DFTB result.");

    m_semi.def(
        "compute_uscc_dftb_gradient",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::USCCDFTBResult& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_matching_parameter_snapshot(
                params, result, "unrestricted SCC-DFTB gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_uscc_dftb_gradient(
                        mol, result, snapshot);
                });
        },
        py::arg("mol"), py::arg("result"), py::arg("params"),
        "Compute the analytic nuclear gradient for an unrestricted SCC-DFTB result.");

    // --- Periodic gradients ---
    m_semi.def(
        "compute_periodic_dftb0_gradient",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::PeriodicDFTB0Result& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            // Preserve the derivative API's legacy provenance error before
            // checking the newly added parameter snapshot provenance.
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_dftb0_gradient: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic DFTB0 gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::
                        compute_periodic_dftb0_gradient(
                            system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
        "Periodic DFTB0 atomic gradient at Gamma-point.");

    m_semi.def(
        "compute_periodic_scc_dftb_gradient",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::PeriodicSCCDFTBResult& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            if (!result.converged) {
                throw std::invalid_argument(
                    "compute_periodic_scc_dftb_gradient requires a converged "
                    "SCC result");
            }
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_scc_dftb_gradient: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic SCC-DFTB gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::
                        compute_periodic_scc_dftb_gradient(
                            system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
        "Periodic SCC-DFTB atomic gradient at Gamma-point.");

    m_semi.def(
        "compute_periodic_udftb0_gradient",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::PeriodicUDFTB0Result& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_udftb0_gradient: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic unrestricted DFTB0 gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::
                        compute_periodic_udftb0_gradient(
                            system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
        "Periodic UDFTB0 atomic gradient at Gamma-point.");

    m_semi.def(
        "compute_periodic_uscc_dftb_gradient",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::PeriodicUSCCDFTBResult& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            if (!result.converged) {
                throw std::invalid_argument(
                    "compute_periodic_uscc_dftb_gradient requires a converged "
                    "SCC result");
            }
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_uscc_dftb_gradient: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic unrestricted SCC-DFTB gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::
                        compute_periodic_uscc_dftb_gradient(
                            system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
        "Periodic USCC-DFTB atomic gradient at Gamma-point.");

    // --- Periodic stress ---
    m_semi.def(
        "compute_periodic_dftb0_stress",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::PeriodicDFTB0Result& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_dftb0_stress: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic DFTB0 stress",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_periodic_dftb0_stress(
                        system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
        "Periodic DFTB0 analytic stress tensor at Gamma-point (3x3, Ha/bohr^3).");

    m_semi.def(
        "compute_periodic_scc_dftb_stress",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::PeriodicSCCDFTBResult& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            if (!result.converged) {
                throw std::invalid_argument(
                    "compute_periodic_scc_dftb_stress requires a converged "
                    "SCC result");
            }
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_scc_dftb_stress: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic SCC-DFTB stress",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::
                        compute_periodic_scc_dftb_stress(
                            system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
        "Periodic SCC-DFTB analytic stress tensor at Gamma-point (3x3, Ha/bohr^3).");



    m_semi.def(
        "dftb0_repulsive_gradient",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::dftb0_repulsive_gradient(
                        mol, snapshot);
                });
        },
        py::arg("mol"), py::arg("params"),
        "Repulsive-energy gradient only (for testing). "
        "Returns (n_atoms, 3) matrix in Hartree/bohr.");


    // --- SCC-DFTB (Stage 3) ---
    py::class_<vibeqc::semiempirical::SCCOptions>(
        m_semi, "SCCOptions",
        "SCC-DFTB SCF options.")
        .def(py::init<>())
        .def_readwrite("max_iter",
                       &vibeqc::semiempirical::SCCOptions::max_iter)
        .def_readwrite("conv_tol_charge",
                       &vibeqc::semiempirical::SCCOptions::conv_tol_charge)
        .def_readwrite("charge_mixing",
                       &vibeqc::semiempirical::SCCOptions::charge_mixing)
        .def_readwrite(
            "electronic_temperature",
            &vibeqc::semiempirical::SCCOptions::electronic_temperature)
        .def_readwrite("initial_charges",
                       &vibeqc::semiempirical::SCCOptions::initial_charges)
        .def_readwrite("use_diis",
                       &vibeqc::semiempirical::SCCOptions::use_diis)
        .def_readwrite("diis_subspace",
                       &vibeqc::semiempirical::SCCOptions::diis_subspace)
        .def_readwrite("gamma_form",
                       &vibeqc::semiempirical::SCCOptions::gamma_form,
                       "Functional form of the SCC-DFTB gamma (D1).")
        /* .def_readwrite("use_dftb3",
                       &vibeqc::semiempirical::SCCOptions::use_dftb3) */;

    py::class_<vibeqc::semiempirical::SCCDFTBResult>(
        m_semi, "SCCDFTBResult",
        "Result of an SCC-DFTB calculation.")
        .def(py::init<>())
        .def_readonly("energy",
                      &vibeqc::semiempirical::SCCDFTBResult::energy)
        .def_readonly("e_electronic",
                      &vibeqc::semiempirical::SCCDFTBResult::e_electronic)
        .def_readonly("e_repulsive",
                      &vibeqc::semiempirical::SCCDFTBResult::e_repulsive)
        .def_readonly("e_scc",
                      &vibeqc::semiempirical::SCCDFTBResult::e_scc)
        /* .def_readonly("e_gamma3",
                      &vibeqc::semiempirical::SCCDFTBResult::e_gamma3) */
        .def_readonly("mo_energies",
                      &vibeqc::semiempirical::SCCDFTBResult::mo_energies)
        .def_readonly("mo_coeffs",
                      &vibeqc::semiempirical::SCCDFTBResult::mo_coeffs)
        .def_readonly("density",
                      &vibeqc::semiempirical::SCCDFTBResult::density)
        .def_readonly("overlap",
                      &vibeqc::semiempirical::SCCDFTBResult::overlap)
        .def_readonly("hamiltonian",
                      &vibeqc::semiempirical::SCCDFTBResult::hamiltonian)
        .def_readonly("charges",
                      &vibeqc::semiempirical::SCCDFTBResult::charges)
        .def_readonly("n_basis",
                      &vibeqc::semiempirical::SCCDFTBResult::n_basis)
        .def_readonly("n_occ",
                      &vibeqc::semiempirical::SCCDFTBResult::n_occ)
        .def_readonly("n_iter",
                      &vibeqc::semiempirical::SCCDFTBResult::n_iter)
        .def_readonly("converged",
                      &vibeqc::semiempirical::SCCDFTBResult::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::SCCDFTBResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::SCCDFTBResult& result) {
                return result.parameter_sha256;
            })
        /* .def_readonly("parameter_fingerprint",
                      &vibeqc::semiempirical::SCCDFTBResult::parameter_fingerprint) */;

    m_semi.def(
        "run_scc_dftb",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::semiempirical::SCCOptions& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_scc_dftb(
                        mol, snapshot, options);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::SCCOptions{},
        "Run an SCC-DFTB calculation with charge self-consistency.");

    // Add Hubbard U accessor to SemiempiricalParameters (already defined above,
    // just need to add the binding). We patch it in via a follow-up .def call.


    // --- Periodic DFTB0 (Stage 4) ---
    py::class_<vibeqc::semiempirical::PeriodicDFTB0Options>(
        m_semi, "PeriodicDFTB0Options",
        "Options for periodic Gamma-point DFTB0.")
        .def(py::init<>())
        .def_readwrite("cutoff_bohr",
                       &vibeqc::semiempirical::PeriodicDFTB0Options::cutoff_bohr)
        .def_readwrite("gamma_only_0",
                       &vibeqc::semiempirical::PeriodicDFTB0Options::gamma_only_0);

    py::class_<vibeqc::semiempirical::PeriodicDFTB0Result>(
        m_semi, "PeriodicDFTB0Result",
        "Result of a periodic Gamma-point DFTB0 calculation.")
        .def(py::init<>())
        .def_readonly("energy",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::energy)
        .def_readonly("e_electronic",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::e_electronic)
        .def_readonly("e_repulsive",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::e_repulsive)
        .def_readonly("mo_energies",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::mo_energies)
        .def_readonly("mo_coeffs",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::mo_coeffs)
        .def_readonly("density",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::density)
        .def_readonly("occupations",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::occupations)
        .def_readonly("overlap_gamma",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::overlap_gamma)
        .def_readonly("hamiltonian_gamma",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::hamiltonian_gamma)
        .def_readonly("n_basis",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::n_basis)
        .def_readonly("n_occ",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::n_occ)
        .def_readonly("n_cells",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::n_cells)
        .def_readonly("cutoff_bohr",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::cutoff_bohr)
        .def_readonly("gamma_only_0",
                      &vibeqc::semiempirical::PeriodicDFTB0Result::gamma_only_0)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::PeriodicDFTB0Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::PeriodicDFTB0Result& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "atom_pair_interaction_cells",
        &vibeqc::semiempirical::atom_pair_interaction_cells,
        py::arg("system"), py::arg("cutoff_bohr"),
        py::call_guard<py::gil_scoped_release>(),
        "Lattice translations carrying at least one atom-pair interaction "
        "within the cutoff (issue #316 pair-distance image selection): the "
        "image list the semiempirical periodic routes actually sum over.");

    m_semi.def(
        "run_dftb0_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::semiempirical::PeriodicDFTB0Options& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_dftb0_gamma(
                        system, snapshot, options);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::PeriodicDFTB0Options{},
        "Run periodic Gamma-point DFTB0 energy calculation.");


        // --- Periodic UDFTB0 ---
    py::class_<vibeqc::semiempirical::PeriodicUDFTB0Result>(
        m_semi, "PeriodicUDFTB0Result",
        "Result of an unrestricted periodic Gamma-point DFTB0 calculation.")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::PeriodicUDFTB0Result::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::PeriodicUDFTB0Result::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::PeriodicUDFTB0Result::e_repulsive)
        .def_readonly("mo_energies", &vibeqc::semiempirical::PeriodicUDFTB0Result::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::PeriodicUDFTB0Result::mo_coeffs)
        .def_readonly("density_alpha", &vibeqc::semiempirical::PeriodicUDFTB0Result::density_alpha)
        .def_readonly(
            "occupations_alpha",
            &vibeqc::semiempirical::PeriodicUDFTB0Result::occupations_alpha)
        .def_readonly(
            "occupations_beta",
            &vibeqc::semiempirical::PeriodicUDFTB0Result::occupations_beta)
        .def_readonly("density_beta", &vibeqc::semiempirical::PeriodicUDFTB0Result::density_beta)
        .def_readonly("overlap_gamma", &vibeqc::semiempirical::PeriodicUDFTB0Result::overlap_gamma)
        .def_readonly("hamiltonian_gamma", &vibeqc::semiempirical::PeriodicUDFTB0Result::hamiltonian_gamma)
        .def_readonly("n_basis", &vibeqc::semiempirical::PeriodicUDFTB0Result::n_basis)
        .def_readonly("n_alpha", &vibeqc::semiempirical::PeriodicUDFTB0Result::n_alpha)
        .def_readonly("n_beta", &vibeqc::semiempirical::PeriodicUDFTB0Result::n_beta)
        .def_readonly("n_cells", &vibeqc::semiempirical::PeriodicUDFTB0Result::n_cells)
        .def_readonly("cutoff_bohr", &vibeqc::semiempirical::PeriodicUDFTB0Result::cutoff_bohr)
        .def_readonly("gamma_only_0", &vibeqc::semiempirical::PeriodicUDFTB0Result::gamma_only_0)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::PeriodicUDFTB0Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::PeriodicUDFTB0Result& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "run_udftb0_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::semiempirical::PeriodicDFTB0Options& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_udftb0_gamma(
                        system, snapshot, options);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::PeriodicDFTB0Options{},
        "Run unrestricted periodic Gamma-point DFTB0 calculation.");


// --- Periodic SCC-DFTB (Stage 5) ---
    py::class_<vibeqc::semiempirical::PeriodicSCCOptions>(
        m_semi, "PeriodicSCCOptions")
        .def(py::init<>())
        .def_readwrite("cutoff_bohr",
                       &vibeqc::semiempirical::PeriodicSCCOptions::cutoff_bohr)
        .def_readwrite("max_iter",
                       &vibeqc::semiempirical::PeriodicSCCOptions::max_iter)
        .def_readwrite("conv_tol_charge",
                       &vibeqc::semiempirical::PeriodicSCCOptions::conv_tol_charge)
        .def_readwrite("charge_mixing",
                       &vibeqc::semiempirical::PeriodicSCCOptions::charge_mixing)
        .def_readwrite("use_diis",
                       &vibeqc::semiempirical::PeriodicSCCOptions::use_diis)
        .def_readwrite("diis_subspace",
                       &vibeqc::semiempirical::PeriodicSCCOptions::diis_subspace)
        .def_readwrite("gamma_form",
                       &vibeqc::semiempirical::PeriodicSCCOptions::gamma_form,
                       "Functional form of the SCC-DFTB gamma (D1)."); 

    py::class_<vibeqc::semiempirical::PeriodicSCCDFTBResult>(
        m_semi, "PeriodicSCCDFTBResult")
        .def(py::init<>())
        .def_readonly("energy",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::energy)
        .def_readonly("e_electronic",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::e_electronic)
        .def_readonly("e_repulsive",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::e_repulsive)
        .def_readonly("e_scc",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::e_scc)
        .def_readonly("mo_energies",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::mo_energies)
        .def_readonly("mo_coeffs",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::mo_coeffs)
        .def_readonly("density",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::density)
        .def_readonly("occupations",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::occupations)
        .def_readonly("overlap_gamma",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::overlap_gamma)
        .def_readonly("hamiltonian_gamma",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::hamiltonian_gamma)
        .def_readonly("charges",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::charges)
        .def_readonly("n_basis",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::n_basis)
        .def_readonly("n_occ",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::n_occ)
        .def_readonly("n_cells",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::n_cells)
        .def_readonly("n_iter",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::n_iter)
        .def_readonly("converged",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::converged)
        .def_readonly("cutoff_bohr",
                      &vibeqc::semiempirical::PeriodicSCCDFTBResult::cutoff_bohr)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::PeriodicSCCDFTBResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::PeriodicSCCDFTBResult& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "run_scc_dftb_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::semiempirical::PeriodicSCCOptions& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_scc_dftb_gamma(
                        system, snapshot, options);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::PeriodicSCCOptions{},
        "Run periodic SCC-DFTB Gamma-point calculation.");

    // --- Periodic USCC-DFTB ---
    py::class_<vibeqc::semiempirical::PeriodicUSCCDFTBResult>(
        m_semi, "PeriodicUSCCDFTBResult")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::e_repulsive)
        .def_readonly("e_scc", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::e_scc)
        .def_readonly("mo_energies", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::mo_coeffs)
        .def_readonly("density_alpha", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::density_alpha)
        .def_readonly(
            "occupations_alpha",
            &vibeqc::semiempirical::PeriodicUSCCDFTBResult::occupations_alpha)
        .def_readonly(
            "occupations_beta",
            &vibeqc::semiempirical::PeriodicUSCCDFTBResult::occupations_beta)
        .def_readonly("density_beta", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::density_beta)
        .def_readonly("overlap_gamma", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::overlap_gamma)
        .def_readonly("hamiltonian_gamma", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::hamiltonian_gamma)
        .def_readonly("charges", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::charges)
        .def_readonly("n_basis", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::n_basis)
        .def_readonly("n_alpha", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::n_alpha)
        .def_readonly("n_beta", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::n_beta)
        .def_readonly("n_cells", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::n_cells)
        .def_readonly("n_iter", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::converged)
        .def_readonly("cutoff_bohr", &vibeqc::semiempirical::PeriodicUSCCDFTBResult::cutoff_bohr)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::PeriodicUSCCDFTBResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::PeriodicUSCCDFTBResult& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "run_uscc_dftb_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::semiempirical::PeriodicSCCOptions& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_uscc_dftb_gamma(
                        system, snapshot, options);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::PeriodicSCCOptions{},
        "Run unrestricted periodic SCC-DFTB Gamma-point calculation.");



    // --- Unrestricted DFTB0 ---
    py::class_<vibeqc::semiempirical::UDFTB0Result>(
        m_semi, "UDFTB0Result",
        "Result of an unrestricted DFTB0 calculation.")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::UDFTB0Result::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::UDFTB0Result::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::UDFTB0Result::e_repulsive)
        .def_readonly("mo_energies", &vibeqc::semiempirical::UDFTB0Result::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::UDFTB0Result::mo_coeffs)
        .def_readonly("density_alpha", &vibeqc::semiempirical::UDFTB0Result::density_alpha)
        .def_readonly("density_beta", &vibeqc::semiempirical::UDFTB0Result::density_beta)
        .def_readonly("overlap", &vibeqc::semiempirical::UDFTB0Result::overlap)
        .def_readonly("hamiltonian", &vibeqc::semiempirical::UDFTB0Result::hamiltonian)
        .def_readonly("charges", &vibeqc::semiempirical::UDFTB0Result::charges)
        .def_readonly("n_basis", &vibeqc::semiempirical::UDFTB0Result::n_basis)
        .def_readonly("n_alpha", &vibeqc::semiempirical::UDFTB0Result::n_alpha)
        .def_readonly("n_beta", &vibeqc::semiempirical::UDFTB0Result::n_beta)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::UDFTB0Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::UDFTB0Result& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "run_udftb0",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_udftb0(mol, snapshot);
                });
        },
        py::arg("mol"), py::arg("params"),
        "Run unrestricted DFTB0 for open-shell molecules.");


    py::class_<vibeqc::semiempirical::USCCDFTBResult>(
        m_semi, "USCCDFTBResult")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::USCCDFTBResult::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::USCCDFTBResult::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::USCCDFTBResult::e_repulsive)
        .def_readonly("e_scc", &vibeqc::semiempirical::USCCDFTBResult::e_scc)
        .def_readonly("mo_energies", &vibeqc::semiempirical::USCCDFTBResult::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::USCCDFTBResult::mo_coeffs)
        .def_readonly("density_alpha", &vibeqc::semiempirical::USCCDFTBResult::density_alpha)
        .def_readonly("density_beta", &vibeqc::semiempirical::USCCDFTBResult::density_beta)
        .def_readonly("overlap", &vibeqc::semiempirical::USCCDFTBResult::overlap)
        .def_readonly("hamiltonian", &vibeqc::semiempirical::USCCDFTBResult::hamiltonian)
        .def_readonly("charges", &vibeqc::semiempirical::USCCDFTBResult::charges)
        .def_readonly("n_basis", &vibeqc::semiempirical::USCCDFTBResult::n_basis)
        .def_readonly("n_alpha", &vibeqc::semiempirical::USCCDFTBResult::n_alpha)
        .def_readonly("n_beta", &vibeqc::semiempirical::USCCDFTBResult::n_beta)
        .def_readonly("n_iter", &vibeqc::semiempirical::USCCDFTBResult::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::USCCDFTBResult::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::USCCDFTBResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::USCCDFTBResult& result) {
                return result.parameter_sha256;
            });
    m_semi.def(
        "run_uscc_dftb",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::semiempirical::SCCOptions& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_uscc_dftb(
                        mol, snapshot, options);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::SCCOptions{},
        "Run unrestricted SCC-DFTB for open-shell molecules.");


    m_semi.def(
        "compute_udftb0_gradient",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::UDFTB0Result& result,
           const vibeqc::semiempirical::SemiempiricalParameters& params) {
            return run_with_matching_parameter_snapshot(
                params, result, "unrestricted DFTB0 gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_udftb0_gradient(
                        mol, result, snapshot);
                });
        },
        py::arg("mol"), py::arg("result"), py::arg("params"),
        "Analytic gradient for unrestricted DFTB0.");


    // --- k-point DFTB0 (Stage 10) ---
    py::class_<vibeqc::semiempirical::KPointOccupationOptions>(
        m_semi, "KPointOccupationOptions")
        .def(py::init<>())
        .def_readwrite(
            "smearing_temperature",
            &vibeqc::semiempirical::KPointOccupationOptions::smearing_temperature)
        .def_readwrite(
            "min_resolvable_frontier_gap",
            &vibeqc::semiempirical::KPointOccupationOptions::
                min_resolvable_frontier_gap);

    // Roundoff-degeneracy grouping constants of the T = 0 fills (#544),
    // exported so the Python twin (vibeqc.smearing.apply) can be pinned to
    // the same numbers by a test instead of by a comment.
    m_semi.attr("zero_temperature_degeneracy_ulps") =
        vibeqc::semiempirical::kZeroTemperatureDegeneracyUlps;
    m_semi.attr("degenerate_group_span_factor") =
        vibeqc::semiempirical::kDegenerateGroupSpanFactor;

    py::class_<vibeqc::semiempirical::KPointOccupationResult>(
        m_semi, "KPointOccupationResult")
        .def(py::init<>())
        .def_readonly(
            "occupations_per_k",
            &vibeqc::semiempirical::KPointOccupationResult::occupations_per_k)
        .def_readonly(
            "fermi_level",
            &vibeqc::semiempirical::KPointOccupationResult::fermi_level)
        .def_readonly(
            "entropy",
            &vibeqc::semiempirical::KPointOccupationResult::entropy)
        .def_readonly(
            "frontier_cut_unresolved",
            &vibeqc::semiempirical::KPointOccupationResult::
                frontier_cut_unresolved)
        .def_readonly(
            "frontier_gap",
            &vibeqc::semiempirical::KPointOccupationResult::frontier_gap);

    m_semi.def(
        "compute_closed_shell_kpoint_occupations",
        &vibeqc::semiempirical::compute_closed_shell_kpoint_occupations,
        py::arg("eps_per_k"), py::arg("weights"), py::arg("n_electrons_per_cell"),
        py::arg("n_occ_each"),
        py::arg("options") = vibeqc::semiempirical::KPointOccupationOptions{},
        "Closed-shell k-point occupations: the T = 0 global-Aufbau fill "
        "(one weighted particle constraint over the whole mesh, transitive "
        "roundoff-degeneracy grouping, Weinert-Davenport T -> 0 ensemble on "
        "a cut group) or the Fermi-Dirac fill at smearing_temperature > 0. "
        "The kernel the k-point DFTB0 / SCC-DFTB drivers use; exposed for "
        "tests and for parity with vibeqc.smearing.apply (#544).");

    m_semi.def(
        "gamma_degenerate_frontier_occupations",
        &vibeqc::semiempirical::gamma_degenerate_frontier_occupations,
        py::arg("eps"), py::arg("n_occ"), py::arg("energy_tolerance") = -1.0,
        py::arg("max_occupation") = 2.0,
        "Single-spectrum Gamma occupations: hard Aufbau unless the integer "
        "frontier cuts a roundoff-degenerate manifold, which then receives "
        "equal fractional occupations (issues #339, #544). energy_tolerance "
        "< 0 selects the kernel's own roundoff scale.");

    py::class_<vibeqc::semiempirical::KPointDFTB0Result>(
        m_semi, "KPointDFTB0Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::KPointDFTB0Result::energy)
        .def_readonly(
            "free_energy",
            &vibeqc::semiempirical::KPointDFTB0Result::free_energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::KPointDFTB0Result::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::KPointDFTB0Result::e_repulsive)
        .def_readonly(
            "fermi_level",
            &vibeqc::semiempirical::KPointDFTB0Result::fermi_level)
        .def_readonly("entropy", &vibeqc::semiempirical::KPointDFTB0Result::entropy)
        .def_readonly(
            "smearing_temperature",
            &vibeqc::semiempirical::KPointDFTB0Result::smearing_temperature)
        .def_readonly("band_energies", &vibeqc::semiempirical::KPointDFTB0Result::band_energies)
        .def_readonly("eps_per_k", &vibeqc::semiempirical::KPointDFTB0Result::eps_per_k)
        .def_readonly(
            "occupations_per_k",
            &vibeqc::semiempirical::KPointDFTB0Result::occupations_per_k)
        // Measured band edges + gaps from the occupations actually used
        // (issue #426): valence n > 1e-8, conduction n < 2 - 1e-8, a
        // fractional state in both classes, gaps unclamped. NaN / -1 only
        // when an edge does not exist in the model space.
        .def_readonly(
            "valence_band_max",
            &vibeqc::semiempirical::KPointDFTB0Result::valence_band_max,
            "Highest occupied energy across all k (occupation > 1e-8); "
            "NaN when no occupied state exists.")
        .def_readonly(
            "conduction_band_min",
            &vibeqc::semiempirical::KPointDFTB0Result::conduction_band_min,
            "Lowest conduction-capable energy across all k (occupation "
            "< 2 - 1e-8); NaN when the valence basis is completely "
            "filled.")
        .def_readonly(
            "indirect_gap",
            &vibeqc::semiempirical::KPointDFTB0Result::indirect_gap,
            "conduction_band_min - valence_band_max, unclamped: a "
            "fractional Fermi group (metal / smeared frontier) measures "
            "<= 0, band overlap measures its negative value; NaN only "
            "when an edge does not exist in the model space.")
        .def_readonly(
            "direct_gap",
            &vibeqc::semiempirical::KPointDFTB0Result::direct_gap,
            "min over k of the per-k gap; always >= indirect_gap.")
        .def_readonly(
            "gap_above_fermi_manifold",
            &vibeqc::semiempirical::KPointDFTB0Result::gap_above_fermi_manifold,
            "Distance to the next entirely empty state: "
            "min(eps | n <= 1e-8) - max(eps | n > 1e-8). This is NOT "
            "the band gap -- at a fractionally occupied edge it steps "
            "over the whole partially filled manifold and is strictly "
            "positive. Exposed under its own name so it can never be "
            "mistaken for indirect_gap (issue #426 / sec8-r4 F1).")
        .def_readonly(
            "is_metallic",
            &vibeqc::semiempirical::KPointDFTB0Result::is_metallic,
            "True when E_F is pinned inside a partially filled "
            "manifold (some occupation is fractional) or the measured "
            "indirect_gap is <= 0. The structural gapless flag: screen "
            "on this rather than on the sign of a float, since a "
            "zero-temperature global aufbau fill can never produce a "
            "negative pooled gap.")
        .def_readonly(
            "valence_band_max_k",
            &vibeqc::semiempirical::KPointDFTB0Result::valence_band_max_k)
        .def_readonly(
            "conduction_band_min_k",
            &vibeqc::semiempirical::KPointDFTB0Result::conduction_band_min_k)
        .def_readonly(
            "direct_gap_k",
            &vibeqc::semiempirical::KPointDFTB0Result::direct_gap_k)
        .def_readonly(
            "valence_max_per_k",
            &vibeqc::semiempirical::KPointDFTB0Result::valence_max_per_k)
        .def_readonly(
            "conduction_min_per_k",
            &vibeqc::semiempirical::KPointDFTB0Result::conduction_min_per_k)
        .def_readonly("n_basis", &vibeqc::semiempirical::KPointDFTB0Result::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::KPointDFTB0Result::n_occ)
        .def_readonly("n_kpoints", &vibeqc::semiempirical::KPointDFTB0Result::n_kpoints)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::KPointDFTB0Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::KPointDFTB0Result& result) {
                return result.parameter_sha256;
            });
    m_semi.def(
        "run_dftb0_kpoints",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::BlochKMesh& kmesh,
           double cutoff_bohr,
           const vibeqc::semiempirical::KPointOccupationOptions& occupations) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_dftb0_kpoints(
                        system, snapshot, kmesh, cutoff_bohr, occupations);
                });
        },
        py::arg("system"), py::arg("params"), py::arg("kmesh"),
        py::arg("cutoff_bohr") = 15.0,
        py::arg("occupation_options") =
            vibeqc::semiempirical::KPointOccupationOptions{},
        "Run periodic DFTB0 on a k-point mesh.");
    m_semi.def(
        "run_dftb0_bandpath",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const std::vector<Eigen::Vector3d>& kpath,
           double cutoff_bohr) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_dftb0_bandpath(
                        system, snapshot, kpath, cutoff_bohr);
                });
        },
        py::arg("system"), py::arg("params"), py::arg("kpath"),
        py::arg("cutoff_bohr") = 15.0,
        "Run DFTB0 along a band-structure k-path.");


    py::class_<vibeqc::semiempirical::KPointSCCDFTBResult>(
        m_semi, "KPointSCCDFTBResult")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::KPointSCCDFTBResult::energy)
        .def_readonly(
            "free_energy",
            &vibeqc::semiempirical::KPointSCCDFTBResult::free_energy)
        .def_readonly(
            "e_electronic",
            &vibeqc::semiempirical::KPointSCCDFTBResult::e_electronic)
        .def_readonly(
            "e_repulsive",
            &vibeqc::semiempirical::KPointSCCDFTBResult::e_repulsive)
        .def_readonly("e_scc", &vibeqc::semiempirical::KPointSCCDFTBResult::e_scc)
        .def_readonly(
            "fermi_level",
            &vibeqc::semiempirical::KPointSCCDFTBResult::fermi_level)
        .def_readonly("entropy", &vibeqc::semiempirical::KPointSCCDFTBResult::entropy)
        .def_readonly(
            "smearing_temperature",
            &vibeqc::semiempirical::KPointSCCDFTBResult::smearing_temperature)
        .def_readonly("charges", &vibeqc::semiempirical::KPointSCCDFTBResult::charges)
        .def_readonly("band_energies", &vibeqc::semiempirical::KPointSCCDFTBResult::band_energies)
        .def_readonly("eps_per_k", &vibeqc::semiempirical::KPointSCCDFTBResult::eps_per_k)
        .def_readonly(
            "occupations_per_k",
            &vibeqc::semiempirical::KPointSCCDFTBResult::occupations_per_k)
        // Measured band edges + gaps (issue #426; same convention as
        // KPointDFTB0Result). NaN also on unconverged diagnostics
        // records: nothing was measured.
        .def_readonly(
            "valence_band_max",
            &vibeqc::semiempirical::KPointSCCDFTBResult::valence_band_max,
            "Highest occupied energy across all k (occupation > 1e-8); "
            "NaN when no occupied state exists or the SCC loop did not "
            "converge.")
        .def_readonly(
            "conduction_band_min",
            &vibeqc::semiempirical::KPointSCCDFTBResult::conduction_band_min,
            "Lowest conduction-capable energy across all k (occupation "
            "< 2 - 1e-8); NaN when the valence basis is completely "
            "filled or the SCC loop did not converge.")
        .def_readonly(
            "indirect_gap",
            &vibeqc::semiempirical::KPointSCCDFTBResult::indirect_gap,
            "conduction_band_min - valence_band_max, unclamped: a "
            "fractional Fermi group (metal / smeared frontier) measures "
            "<= 0, band overlap measures its negative value; NaN only "
            "when an edge does not exist in the model space or the SCC "
            "loop did not converge.")
        .def_readonly(
            "direct_gap",
            &vibeqc::semiempirical::KPointSCCDFTBResult::direct_gap,
            "min over k of the per-k gap; always >= indirect_gap.")
        .def_readonly(
            "gap_above_fermi_manifold",
            &vibeqc::semiempirical::KPointSCCDFTBResult::gap_above_fermi_manifold,
            "Distance to the next entirely empty state: "
            "min(eps | n <= 1e-8) - max(eps | n > 1e-8). This is NOT "
            "the band gap -- at a fractionally occupied edge it steps "
            "over the whole partially filled manifold and is strictly "
            "positive. Exposed under its own name so it can never be "
            "mistaken for indirect_gap (issue #426 / sec8-r4 F1).")
        .def_readonly(
            "is_metallic",
            &vibeqc::semiempirical::KPointSCCDFTBResult::is_metallic,
            "True when E_F is pinned inside a partially filled "
            "manifold (some occupation is fractional) or the measured "
            "indirect_gap is <= 0. The structural gapless flag: screen "
            "on this rather than on the sign of a float, since a "
            "zero-temperature global aufbau fill can never produce a "
            "negative pooled gap.")
        .def_readonly(
            "valence_band_max_k",
            &vibeqc::semiempirical::KPointSCCDFTBResult::valence_band_max_k)
        .def_readonly(
            "conduction_band_min_k",
            &vibeqc::semiempirical::KPointSCCDFTBResult::conduction_band_min_k)
        .def_readonly(
            "direct_gap_k",
            &vibeqc::semiempirical::KPointSCCDFTBResult::direct_gap_k)
        .def_readonly(
            "valence_max_per_k",
            &vibeqc::semiempirical::KPointSCCDFTBResult::valence_max_per_k)
        .def_readonly(
            "conduction_min_per_k",
            &vibeqc::semiempirical::KPointSCCDFTBResult::conduction_min_per_k)
        .def_readonly("n_basis", &vibeqc::semiempirical::KPointSCCDFTBResult::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::KPointSCCDFTBResult::n_occ)
        .def_readonly("n_kpoints", &vibeqc::semiempirical::KPointSCCDFTBResult::n_kpoints)
        .def_readonly("n_iter", &vibeqc::semiempirical::KPointSCCDFTBResult::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::KPointSCCDFTBResult::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::KPointSCCDFTBResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::KPointSCCDFTBResult& result) {
                return result.parameter_sha256;
            });
    m_semi.def(
        "run_scc_dftb_kpoints",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::BlochKMesh& kmesh,
           const vibeqc::semiempirical::SCCOptions& scc_options,
           double cutoff_bohr,
           const vibeqc::semiempirical::KPointOccupationOptions& occupations,
           bool allow_unconverged) {
            auto result = run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_scc_dftb_kpoints(
                        system, snapshot, kmesh, scc_options, cutoff_bohr,
                        occupations);
                });
            // Fail loud on SCC budget exhaustion (issue #342): callers
            // that screen on "did not raise" must never admit an
            // unconverged record.  The C++ driver keeps its return-object
            // contract; opt-in diagnostics callers receive the record
            // still flagged converged=False.
            if (!result.converged && !allow_unconverged) {
                throw SCFNonConvergenceError(scf_nonconvergence_message(
                    "periodic full-k SCC-DFTB", result.n_iter,
                    scc_options.conv_tol_charge));
            }
            return result;
        },
        py::arg("system"), py::arg("params"), py::arg("kmesh"),
        py::arg("scc_opts") = vibeqc::semiempirical::SCCOptions{},
        py::arg("cutoff_bohr") = 15.0,
        py::arg("occupation_options") =
            vibeqc::semiempirical::KPointOccupationOptions{},
        py::arg("allow_unconverged") = false,
        "Run periodic SCC-DFTB on a k-point mesh. Raises "
        "SCFNonConvergenceError when the SCC loop exhausts max_iter "
        "unless allow_unconverged=True.");

    py::class_<vibeqc::semiempirical::KPointDFTBFDBatchOptions>(
        m_semi, "KPointDFTBFDBatchOptions")
        .def(py::init<>())
        .def_readwrite(
            "coordinate_step",
            &vibeqc::semiempirical::KPointDFTBFDBatchOptions::coordinate_step)
        .def_readwrite(
            "strain_step",
            &vibeqc::semiempirical::KPointDFTBFDBatchOptions::strain_step)
        .def_readwrite(
            "compute_gradient",
            &vibeqc::semiempirical::KPointDFTBFDBatchOptions::compute_gradient)
        .def_readwrite(
            "compute_stress",
            &vibeqc::semiempirical::KPointDFTBFDBatchOptions::compute_stress)
        .def_readwrite(
            "strain_positions",
            &vibeqc::semiempirical::KPointDFTBFDBatchOptions::strain_positions);

    py::class_<vibeqc::semiempirical::KPointDFTBFDBatchResult>(
        m_semi, "KPointDFTBFDBatchResult")
        .def(py::init<>())
        .def_readonly(
            "gradient",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::gradient)
        .def_readonly(
            "stress",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::stress)
        .def_readonly(
            "energy_evaluations",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::energy_evaluations)
        .def_readonly(
            "system_workspace_copies",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::system_workspace_copies)
        .def_readonly(
            "kmesh_workspace_copies",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::kmesh_workspace_copies)
        .def_readonly(
            "lattice_cell_setups",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::lattice_cell_setups)
        .def_readonly(
            "workspace_bytes",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::workspace_bytes)
        .def_readonly(
            "memory_counters_complete",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::memory_counters_complete)
        .def_readonly(
            "differentiated_free_energy",
            &vibeqc::semiempirical::KPointDFTBFDBatchResult::differentiated_free_energy)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::KPointDFTBFDBatchResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::KPointDFTBFDBatchResult& result) {
                return result.parameter_sha256;
            });

    m_semi.def(
        "compute_dftb0_kpoints_fd_batch",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::BlochKMesh& kmesh,
           double cutoff_bohr,
           const vibeqc::semiempirical::KPointDFTBFDBatchOptions& fd_options,
           const vibeqc::semiempirical::KPointOccupationOptions& occupations) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_dftb0_kpoints_fd_batch(
                        system, snapshot, kmesh, cutoff_bohr, fd_options,
                        occupations);
                });
        },
        py::arg("system"), py::arg("params"), py::arg("kmesh"),
        py::arg("cutoff_bohr") = 15.0,
        py::arg("fd_options") =
            vibeqc::semiempirical::KPointDFTBFDBatchOptions{},
        py::arg("occupation_options") =
            vibeqc::semiempirical::KPointOccupationOptions{},
        "Compute full-k DFTB0 gradient/stress by native finite differences.");
    m_semi.def(
        "compute_scc_dftb_kpoints_fd_batch",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::SemiempiricalParameters& params,
           const vibeqc::BlochKMesh& kmesh,
           const vibeqc::semiempirical::SCCOptions& scc_options,
           double cutoff_bohr,
           const vibeqc::semiempirical::KPointDFTBFDBatchOptions& fd_options,
           const vibeqc::semiempirical::KPointOccupationOptions& occupations) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_scc_dftb_kpoints_fd_batch(
                        system, snapshot, kmesh, scc_options, cutoff_bohr,
                        fd_options, occupations);
                });
        },
        py::arg("system"), py::arg("params"), py::arg("kmesh"),
        py::arg("scc_options") = vibeqc::semiempirical::SCCOptions{},
        py::arg("cutoff_bohr") = 15.0,
        py::arg("fd_options") =
            vibeqc::semiempirical::KPointDFTBFDBatchOptions{},
        py::arg("occupation_options") =
            vibeqc::semiempirical::KPointOccupationOptions{},
        "Compute full-k SCC-DFTB gradient/stress by native finite differences.");

    // --- Method registry ---
    py::enum_<vibeqc::semiempirical::MethodFamily>(m_semi, "MethodFamily")
        .value("DFTB", vibeqc::semiempirical::MethodFamily::DFTB)
        .value("XTB", vibeqc::semiempirical::MethodFamily::XTB)
        .value("NDDO", vibeqc::semiempirical::MethodFamily::NDDO)
        .value("INDO", vibeqc::semiempirical::MethodFamily::INDO)
        .value("ML", vibeqc::semiempirical::MethodFamily::ML)
        .value("LEGACY", vibeqc::semiempirical::MethodFamily::LEGACY);

    py::enum_<vibeqc::semiempirical::PeriodicTier>(m_semi, "PeriodicTier")
        .value("NATIVE", vibeqc::semiempirical::PeriodicTier::Native)
        .value("GENERALIZED", vibeqc::semiempirical::PeriodicTier::Generalized)
        .value("EXPERIMENTAL", vibeqc::semiempirical::PeriodicTier::Experimental);

    py::class_<vibeqc::semiempirical::SemiempiricalMethodConfig>(
        m_semi, "SemiempiricalMethodConfig")
        .def(py::init<>())
        .def_readwrite("name", &vibeqc::semiempirical::SemiempiricalMethodConfig::name)
        .def_readwrite("display_name", &vibeqc::semiempirical::SemiempiricalMethodConfig::display_name)
        .def_readwrite("family", &vibeqc::semiempirical::SemiempiricalMethodConfig::family)
        .def_readwrite("description", &vibeqc::semiempirical::SemiempiricalMethodConfig::description)
        .def_readwrite("supports_open_shell", &vibeqc::semiempirical::SemiempiricalMethodConfig::supports_open_shell)
        .def_readwrite("supports_periodic", &vibeqc::semiempirical::SemiempiricalMethodConfig::supports_periodic)
        .def_readwrite("periodic_tier", &vibeqc::semiempirical::SemiempiricalMethodConfig::periodic_tier)
        .def_readwrite("gradient_quality", &vibeqc::semiempirical::SemiempiricalMethodConfig::gradient_quality)
        .def_readwrite("supports_stress", &vibeqc::semiempirical::SemiempiricalMethodConfig::supports_stress)
        .def_readwrite("supports_kpoints", &vibeqc::semiempirical::SemiempiricalMethodConfig::supports_kpoints)
        .def_readwrite("has_dispersion", &vibeqc::semiempirical::SemiempiricalMethodConfig::has_dispersion)
        .def_readwrite("parameter_version", &vibeqc::semiempirical::SemiempiricalMethodConfig::parameter_version);

            py::class_<vibeqc::semiempirical::CoreElementData>(
        m_semi, "CoreElementData")
        .def(py::init<>())
        .def_readwrite("Z", &vibeqc::semiempirical::CoreElementData::Z)
        .def_readwrite("valence_electrons", &vibeqc::semiempirical::CoreElementData::valence_electrons)
        .def_readwrite("hubbard_u", &vibeqc::semiempirical::CoreElementData::hubbard_u)
        .def_readonly("on_site", &vibeqc::semiempirical::CoreElementData::on_site)
        .def_readonly("zeta", &vibeqc::semiempirical::CoreElementData::zeta);

    py::class_<vibeqc::semiempirical::CoreParameterSet,
                 std::unique_ptr<vibeqc::semiempirical::CoreParameterSet, py::nodelete>>(
        m_semi, "CoreParameterSet",
        "Base class for semiempirical parameter sets.")
        .def("has_element", &vibeqc::semiempirical::CoreParameterSet::has_element)
        .def("element", [](const vibeqc::semiempirical::CoreParameterSet& p,
                            int Z) {
            return p.element(Z);
        }, py::arg("Z"), "Return a detached base-element snapshot.")
        .def("repulsive_energy", &vibeqc::semiempirical::CoreParameterSet::repulsive_energy)
        .def("repulsive_derivative", &vibeqc::semiempirical::CoreParameterSet::repulsive_derivative)
        .def("method_name", &vibeqc::semiempirical::CoreParameterSet::method_name)
        .def("parameter_version", &vibeqc::semiempirical::CoreParameterSet::parameter_version)
        .def("n_elements", &vibeqc::semiempirical::CoreParameterSet::n_elements);

py::class_<vibeqc::semiempirical::ParameterSetMetadata>(
        m_semi, "ParameterSetMetadata")
        .def(py::init<>())
        .def_readwrite("method_name", &vibeqc::semiempirical::ParameterSetMetadata::method_name)
        .def_readwrite("version", &vibeqc::semiempirical::ParameterSetMetadata::version)
        .def_readwrite("origin", &vibeqc::semiempirical::ParameterSetMetadata::origin)
        .def_readwrite("license", &vibeqc::semiempirical::ParameterSetMetadata::license)
        .def_readwrite("n_elements", &vibeqc::semiempirical::ParameterSetMetadata::n_elements)
        .def_readwrite("element_list", &vibeqc::semiempirical::ParameterSetMetadata::element_list)
        .def_readwrite("doi_or_url", &vibeqc::semiempirical::ParameterSetMetadata::doi_or_url)
        .def_readwrite("fit_dataset", &vibeqc::semiempirical::ParameterSetMetadata::fit_dataset)
        .def_readwrite("parameter_hash", &vibeqc::semiempirical::ParameterSetMetadata::parameter_hash);

py::class_<vibeqc::semiempirical::SemiempiricalMethodRegistry>(
        m_semi, "SemiempiricalMethodRegistry")
        .def_static("register_dftb", []() {
            auto& reg = vibeqc::semiempirical::SemiempiricalMethodRegistry::instance();
            // DFTB0
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "dftb0";
                p.config.display_name = "DFTB0";
                p.config.family = vibeqc::semiempirical::MethodFamily::DFTB;
                p.config.description = "Non-self-consistent tight-binding";
                p.config.supports_open_shell = true;
                p.config.supports_periodic = true;
                p.config.periodic_tier = vibeqc::semiempirical::PeriodicTier::Native;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::Exact;
                p.config.supports_stress = true;
                p.config.supports_kpoints = true;
                p.config.has_dispersion = true;
                p.config.parameter_version = "in-house-85-2026";
                            reg.register_method(p);
            }
            // PM6
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "pm6";
                p.config.display_name = "PM6";
                p.config.family = vibeqc::semiempirical::MethodFamily::NDDO;
                p.config.description = "Parameterized Model 6 (Stewart 2007)";
                p.config.supports_open_shell = true;
                p.config.supports_periodic = true;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::FiniteDifference;
                p.config.periodic_tier = vibeqc::semiempirical::PeriodicTier::Experimental;
                p.config.supports_stress = true;
                p.config.parameter_version = "pm6-2007";
                reg.register_method(p);
            }
            // MSINDO (Bredow/Geudtner/Jug INDO) — RHF s/p/d + UHF s/p
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "msindo";
                p.config.display_name = "MSINDO";
                p.config.family = vibeqc::semiempirical::MethodFamily::INDO;
                p.config.description =
                    "MSINDO INDO (Bredow/Geudtner/Jug); RHF s/p/d + open-shell "
                    "UHF s/p";
                // C++ engine now covers full Z=1..35 (H–Br) RHF s/p/d + UHF s/p.
                p.config.supports_open_shell = true;
                p.config.gradient_quality =
                    vibeqc::semiempirical::GradientQuality::FiniteDifference;
                p.config.periodic_tier =
                    vibeqc::semiempirical::PeriodicTier::Experimental;
                p.config.supported_elements = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36};
                p.config.parameter_version = "msindo-2025";
                reg.register_method(p);
            }
            // OM1
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "om1";
                p.config.display_name = "OM1";
                p.config.family = vibeqc::semiempirical::MethodFamily::NDDO;
                p.config.description = "Orthogonalization Model 1 (Kolb/Thiel 1993)";
                p.config.supports_open_shell = true;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::FiniteDifference;
                p.config.parameter_version = "om1-1993";
                reg.register_method(p);
            }
            // OM2
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "om2";
                p.config.display_name = "OM2";
                p.config.family = vibeqc::semiempirical::MethodFamily::NDDO;
                p.config.description = "Orthogonalization Model 2 (Weber/Thiel 2000)";
                p.config.supports_open_shell = true;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::FiniteDifference;
                p.config.parameter_version = "om2-2000";
                reg.register_method(p);
            }
            // OM3
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "om3";
                p.config.display_name = "OM3";
                p.config.family = vibeqc::semiempirical::MethodFamily::NDDO;
                p.config.description = "Orthogonalization Model 3 (Scholten 2003)";
                p.config.supports_open_shell = true;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::FiniteDifference;
                p.config.parameter_version = "om3-2003";
                reg.register_method(p);
            }
            // GFN2-xTB
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "gfn2-xtb";
                p.config.display_name = "GFN2-xTB";
                p.config.family = vibeqc::semiempirical::MethodFamily::XTB;
                p.config.description = "Extended tight-binding (Grimme 2019)";
                p.config.supports_open_shell = true;
                p.config.supports_periodic = true;
                p.config.periodic_tier = vibeqc::semiempirical::PeriodicTier::Generalized;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::Exact;
                p.config.supports_stress = true;
                p.config.supports_kpoints = true;
                p.config.has_dispersion = true;
                p.config.parameter_version = "gfn2-xtb-2019";
                reg.register_method(p);
            }
            // SCC-DFTB
            {
                vibeqc::semiempirical::SemiempiricalMethodPlugin p;
                p.config.name = "scc-dftb";
                p.config.display_name = "SCC-DFTB";
                p.config.family = vibeqc::semiempirical::MethodFamily::DFTB;
                p.config.description = "Self-consistent-charge tight-binding";
                p.config.supports_open_shell = true;
                p.config.supports_periodic = true;
                p.config.periodic_tier = vibeqc::semiempirical::PeriodicTier::Native;
                p.config.gradient_quality = vibeqc::semiempirical::GradientQuality::Exact;
                p.config.supports_stress = true;
                p.config.supports_kpoints = true;
                p.config.has_dispersion = true;
                p.config.parameter_version = "in-house-85-2026";
                reg.register_method(p);
            }
        })
        .def_static("instance", &vibeqc::semiempirical::SemiempiricalMethodRegistry::instance,
                    py::return_value_policy::reference)
        .def("find_config", [](const vibeqc::semiempirical::SemiempiricalMethodRegistry& reg,
                                  const std::string& name) -> vibeqc::semiempirical::SemiempiricalMethodConfig {
                 const auto* p = reg.find(name);
                 if (p) return p->config;
                 throw std::runtime_error("Method not found: " + name);
             })
        .def("method_names", &vibeqc::semiempirical::SemiempiricalMethodRegistry::method_names)
        .def("family_methods", &vibeqc::semiempirical::SemiempiricalMethodRegistry::family_methods);

    // --- GFN2-xTB parameters ---
    py::module_ m_xtb = m_semi.def_submodule("xtb", "GFN-xTB method family");

    // Same class object as vibeqc._vibeqc_core.semiempirical's: xtb k-route
    // callers catch the identical exception (issue #342).
    m_xtb.attr("SCFNonConvergenceError") =
        m_semi.attr("SCFNonConvergenceError");

    py::class_<vibeqc::semiempirical::xtb::GFN2ElementData::ShellData>(
        m_xtb, "GFN2ShellData",
        "Per-shell data for GFN2-xTB.")
        .def(py::init<>())
        .def_readwrite("n", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::n)
        .def_readwrite("l", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::l)
        .def_readwrite("en", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::en)
        .def_readwrite("zeta", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::zeta)
        .def_readwrite("k_en", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::k_en)
        .def_readwrite("kcn", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::kcn)
        .def_readwrite("poly", &vibeqc::semiempirical::xtb::GFN2ElementData::ShellData::poly);

    py::class_<vibeqc::semiempirical::xtb::GFN2ElementData>(
        m_xtb, "GFN2ElementData",
        "Per-element data for GFN2-xTB.")
        .def(py::init<>())
        .def_readwrite("Z", &vibeqc::semiempirical::xtb::GFN2ElementData::Z)
        .def_property("shells",
            [](vibeqc::semiempirical::xtb::GFN2ElementData& ed) -> std::vector<vibeqc::semiempirical::xtb::GFN2ElementData::ShellData>& {
                return ed.shells;
            },
            [](vibeqc::semiempirical::xtb::GFN2ElementData& ed,
               const std::vector<vibeqc::semiempirical::xtb::GFN2ElementData::ShellData>& v) {
                ed.shells = v;
            })
        .def_readwrite("gam", &vibeqc::semiempirical::xtb::GFN2ElementData::gam)
        .def_readwrite("gam3", &vibeqc::semiempirical::xtb::GFN2ElementData::gam3)
        .def_readwrite("alpha", &vibeqc::semiempirical::xtb::GFN2ElementData::alpha)
        .def_readwrite("dpol", &vibeqc::semiempirical::xtb::GFN2ElementData::dpol)
        .def_readwrite("qpol", &vibeqc::semiempirical::xtb::GFN2ElementData::qpol)
        .def_readwrite("mp_rad", &vibeqc::semiempirical::xtb::GFN2ElementData::mp_rad)
        .def_readwrite("mp_vcn", &vibeqc::semiempirical::xtb::GFN2ElementData::mp_vcn)
        .def("add_shell", [](vibeqc::semiempirical::xtb::GFN2ElementData& ed,
                              int l, double en, double zeta, double k_en,
                              double kcn, double poly, int n) {
            vibeqc::semiempirical::xtb::GFN2ElementData::ShellData sd;
            sd.n = (n > 0) ? n : (l + 1); sd.l = l; sd.en = en;
            sd.zeta = zeta; sd.k_en = k_en;
            sd.kcn = kcn; sd.poly = poly;
            ed.shells.push_back(sd);
        },
        py::arg("l"), py::arg("en"), py::arg("zeta"), py::arg("k_en"),
        py::arg("kcn") = 0.0, py::arg("poly") = 0.0, py::arg("n") = 0);

    py::class_<vibeqc::semiempirical::xtb::GFN2RepulsivePair>(
        m_xtb, "GFN2RepulsivePair",
        "Repulsive pair data for GFN2-xTB.")
        .def(py::init<>())
        .def_readwrite("alpha", &vibeqc::semiempirical::xtb::GFN2RepulsivePair::alpha)
        .def_readwrite("k_ab", &vibeqc::semiempirical::xtb::GFN2RepulsivePair::k_ab);

    py::class_<vibeqc::semiempirical::xtb::GFN2ParameterSet,
                 vibeqc::semiempirical::CoreParameterSet>(
        m_xtb, "GFN2ParameterSet",
        "GFN2-xTB parameter set (Grimme group, 2019).")
        .def(py::init<>())
        .def_readwrite("d4_s8", &vibeqc::semiempirical::xtb::GFN2ParameterSet::d4_s8)
        .def_readwrite("d4_s9", &vibeqc::semiempirical::xtb::GFN2ParameterSet::d4_s9)
        .def_readwrite("d4_a1", &vibeqc::semiempirical::xtb::GFN2ParameterSet::d4_a1)
        .def_readwrite("d4_a2", &vibeqc::semiempirical::xtb::GFN2ParameterSet::d4_a2)
        .def("has_element", &vibeqc::semiempirical::xtb::GFN2ParameterSet::has_element)
        .def("n_elements", &vibeqc::semiempirical::xtb::GFN2ParameterSet::n_elements)
        .def(
            "element_data",
            [](const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
               int Z) {
                const auto* element = params.element_data(Z);
                if (element == nullptr) {
                    throw py::key_error(
                        "GFN2ParameterSet: element Z=" + std::to_string(Z)
                        + " not found");
                }
                return *element;
            },
            py::arg("Z"),
            "Return a detached copy of one authoritative element record.")
        .def(
            "repulsive_pair",
            [](const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
               int Z1, int Z2) {
                const auto* pair = params.repulsive_pair(Z1, Z2);
                if (pair == nullptr) {
                    throw py::key_error(
                        "GFN2ParameterSet: repulsive pair not found");
                }
                return *pair;
            },
            py::arg("Z1"), py::arg("Z2"),
            "Return a detached copy of one authoritative repulsive pair.")
        .def("repulsive_energy", &vibeqc::semiempirical::xtb::GFN2ParameterSet::repulsive_energy)
        .def("add_element", &vibeqc::semiempirical::xtb::GFN2ParameterSet::add_element)
        .def("set_repulsive_pair", &vibeqc::semiempirical::xtb::GFN2ParameterSet::set_repulsive_pair)
        .def("metadata", &vibeqc::semiempirical::xtb::GFN2ParameterSet::metadata)
        .def("content_sha256",
             &vibeqc::semiempirical::xtb::GFN2ParameterSet::content_sha256)
        .def("parameter_identity",
             &vibeqc::semiempirical::xtb::GFN2ParameterSet::parameter_identity);

    // --- GFN2-xTB driver ---
    py::class_<vibeqc::semiempirical::xtb::GFN2Result>(
        m_xtb, "GFN2Result",
        "Result of a GFN2-xTB calculation.")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::xtb::GFN2Result::energy)
        .def_readonly("free_energy",
                      &vibeqc::semiempirical::xtb::GFN2Result::free_energy)
        .def_readonly("entropy",
                      &vibeqc::semiempirical::xtb::GFN2Result::entropy)
        .def_readonly(
            "smearing_temperature",
            &vibeqc::semiempirical::xtb::GFN2Result::smearing_temperature)
        .def_readonly("e_electronic", &vibeqc::semiempirical::xtb::GFN2Result::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::xtb::GFN2Result::e_repulsive)
        .def_readonly("e_scc", &vibeqc::semiempirical::xtb::GFN2Result::e_scc)
        .def_readonly("e_band0", &vibeqc::semiempirical::xtb::GFN2Result::e_band0)
        .def_readonly("e_aes", &vibeqc::semiempirical::xtb::GFN2Result::e_aes)
        .def_readonly("e_3rd", &vibeqc::semiempirical::xtb::GFN2Result::e_3rd)
        .def_readonly("vmp_rms", &vibeqc::semiempirical::xtb::GFN2Result::vmp_rms)
        .def_readonly("shell_dip_rms", &vibeqc::semiempirical::xtb::GFN2Result::shell_dip_rms)
        .def_readonly("charges", &vibeqc::semiempirical::xtb::GFN2Result::charges)
        .def_readonly("mo_energies", &vibeqc::semiempirical::xtb::GFN2Result::mo_energies)
        .def_readonly("n_basis", &vibeqc::semiempirical::xtb::GFN2Result::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::xtb::GFN2Result::n_occ)
        .def_readonly("converged", &vibeqc::semiempirical::xtb::GFN2Result::converged)
        .def_readonly("n_iter", &vibeqc::semiempirical::xtb::GFN2Result::n_iter)
        .def_readonly(
            "scc_max_change_trace",
            &vibeqc::semiempirical::xtb::GFN2Result::scc_max_change_trace)
        .def_readonly(
            "attempts",
            &vibeqc::semiempirical::xtb::GFN2Result::attempts)
        .def_readonly(
            "selected_attempt_index",
            &vibeqc::semiempirical::xtb::GFN2Result::
                selected_attempt_index)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::xtb::GFN2Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::xtb::GFN2Result& result) {
                return result.parameter_sha256;
            });

    py::class_<vibeqc::semiempirical::xtb::XTBSccOptions>(
        m_xtb, "XTBSccOptions")
        .def(py::init<>())
        .def_readwrite("max_iter", &vibeqc::semiempirical::xtb::XTBSccOptions::max_iter)
        .def_readwrite("conv_tol_charge", &vibeqc::semiempirical::xtb::XTBSccOptions::conv_tol_charge)
        .def_readwrite("charge_mixing", &vibeqc::semiempirical::xtb::XTBSccOptions::charge_mixing)
        .def_readwrite("scc_mixer", &vibeqc::semiempirical::xtb::XTBSccOptions::scc_mixer)
        .def_readwrite("mixer_memory", &vibeqc::semiempirical::xtb::XTBSccOptions::mixer_memory)
        .def_readwrite("mixer_damping", &vibeqc::semiempirical::xtb::XTBSccOptions::mixer_damping)
        // Property (not a plain field): assignment marks the temperature as
        // explicitly requested. The periodic GFN2-xTB drivers smear the
        // frontier by the default width when it is left unset, and honour an
        // explicit value exactly (0 = zero-temperature Aufbau). The molecular
        // driver runs its primary attempt at the value as given; its
        // auto-stabilisation ladder reads the flag and falls back to a
        // finite temperature only when none was requested (vibe-qc#3).
        .def_property(
            "electronic_temperature",
            [](const vibeqc::semiempirical::xtb::XTBSccOptions& self) {
                return self.electronic_temperature;
            },
            [](vibeqc::semiempirical::xtb::XTBSccOptions& self, double value) {
                self.electronic_temperature = value;
                self.electronic_temperature_explicit = true;
            })
        .def_readonly("electronic_temperature_explicit",
                      &vibeqc::semiempirical::xtb::XTBSccOptions::electronic_temperature_explicit)
        .def_readwrite("auto_stabilize",
                       &vibeqc::semiempirical::xtb::XTBSccOptions::auto_stabilize)
        .def_readwrite("aes_faithful", &vibeqc::semiempirical::xtb::XTBSccOptions::aes_faithful)
        .def_readwrite("aes_damping", &vibeqc::semiempirical::xtb::XTBSccOptions::aes_damping)
        .def_property(
            "gamma_form",
            [](const vibeqc::semiempirical::xtb::XTBSccOptions& self) {
                return self.gamma_form;
            },
            [](vibeqc::semiempirical::xtb::XTBSccOptions& self,
               vibeqc::semiempirical::ShellGammaForm value) {
                self.gamma_form = value;
                self.gamma_form_explicit = true;
            },
            "Isotropic second-order kernel form (ShellGammaForm). Unset: the "
            "molecular driver uses Klopman-Ohno and the periodic driver the "
            "Elstner 1998 form; setting it makes every driver honour it.")
        .def_readonly("gamma_form_explicit",
                      &vibeqc::semiempirical::xtb::XTBSccOptions::gamma_form_explicit);

    m_xtb.def(
        "run_gfn2_xtb",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const vibeqc::semiempirical::xtb::XTBSccOptions& options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::xtb::run_gfn2_xtb(
                        mol, snapshot, options);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::xtb::XTBSccOptions{},
        "Run GFN2-xTB SCC energy calculation.");

        // GFN2 gradient
    m_semi.def(
        "compute_gfn2_gradient",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::xtb::GFN2Result& result,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params) {
            return run_with_matching_parameter_snapshot(
                params, result, "GFN2-xTB gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_gfn2_gradient(
                        mol, result, snapshot);
                });
        },
        py::arg("mol"), py::arg("result"), py::arg("params"),
        "GFN2-xTB analytic gradient (fixed-charge approximation).");

        // Multi-k GFN2-xTB
    py::class_<vibeqc::semiempirical::xtb::KPointGFN2Result>(
        m_xtb, "KPointGFN2Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::xtb::KPointGFN2Result::energy)
        .def_readonly("free_energy", &vibeqc::semiempirical::xtb::KPointGFN2Result::free_energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::xtb::KPointGFN2Result::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::xtb::KPointGFN2Result::e_repulsive)
        .def_readonly("e_scc", &vibeqc::semiempirical::xtb::KPointGFN2Result::e_scc)
        .def_readonly("e_band0", &vibeqc::semiempirical::xtb::KPointGFN2Result::e_band0)
        .def_readonly("e_aes", &vibeqc::semiempirical::xtb::KPointGFN2Result::e_aes)
        .def_readonly("e_3rd", &vibeqc::semiempirical::xtb::KPointGFN2Result::e_3rd)
        .def_readonly("fermi_level", &vibeqc::semiempirical::xtb::KPointGFN2Result::fermi_level)
        .def_readonly("entropy", &vibeqc::semiempirical::xtb::KPointGFN2Result::entropy)
        .def_readonly("smearing_temperature", &vibeqc::semiempirical::xtb::KPointGFN2Result::smearing_temperature)
        .def_readonly("band_energies", &vibeqc::semiempirical::xtb::KPointGFN2Result::band_energies)
        .def_readonly("eps_per_k", &vibeqc::semiempirical::xtb::KPointGFN2Result::eps_per_k)
        .def_readonly(
            "occupations_per_k",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::occupations_per_k,
            "Occupations actually used, per k-point (issue #426); empty "
            "on an unconverged diagnostics record.")
        // Measured band edges + gaps (issue #426; same convention as
        // KPointDFTB0Result). NaN also on unconverged diagnostics
        // records: nothing was measured.
        .def_readonly(
            "valence_band_max",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::valence_band_max,
            "Highest occupied energy across all k (occupation > 1e-8); "
            "NaN when no occupied state exists or the SCC loop did not "
            "converge.")
        .def_readonly(
            "conduction_band_min",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::conduction_band_min,
            "Lowest conduction-capable energy across all k (occupation "
            "< 2 - 1e-8); NaN when the valence basis is completely "
            "filled or the SCC loop did not converge.")
        .def_readonly(
            "indirect_gap",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::indirect_gap,
            "conduction_band_min - valence_band_max, unclamped: a "
            "fractional Fermi group (metal / smeared frontier) measures "
            "<= 0, band overlap measures its negative value; NaN only "
            "when an edge does not exist in the model space or the SCC "
            "loop did not converge.")
        .def_readonly(
            "direct_gap",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::direct_gap,
            "min over k of the per-k gap; always >= indirect_gap.")
        .def_readonly(
            "gap_above_fermi_manifold",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::gap_above_fermi_manifold,
            "Distance to the next entirely empty state: "
            "min(eps | n <= 1e-8) - max(eps | n > 1e-8). This is NOT "
            "the band gap -- at a fractionally occupied edge it steps "
            "over the whole partially filled manifold and is strictly "
            "positive. Exposed under its own name so it can never be "
            "mistaken for indirect_gap (issue #426 / sec8-r4 F1).")
        .def_readonly(
            "is_metallic",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::is_metallic,
            "True when E_F is pinned inside a partially filled "
            "manifold (some occupation is fractional) or the measured "
            "indirect_gap is <= 0. The structural gapless flag: screen "
            "on this rather than on the sign of a float, since a "
            "zero-temperature global aufbau fill can never produce a "
            "negative pooled gap.")
        .def_readonly(
            "valence_band_max_k",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::valence_band_max_k)
        .def_readonly(
            "conduction_band_min_k",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::conduction_band_min_k)
        .def_readonly(
            "direct_gap_k",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::direct_gap_k)
        .def_readonly(
            "valence_max_per_k",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::valence_max_per_k)
        .def_readonly(
            "conduction_min_per_k",
            &vibeqc::semiempirical::xtb::KPointGFN2Result::conduction_min_per_k)
        .def_readonly("n_kpoints", &vibeqc::semiempirical::xtb::KPointGFN2Result::n_kpoints)
        .def_readonly("n_basis", &vibeqc::semiempirical::xtb::KPointGFN2Result::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::xtb::KPointGFN2Result::n_occ)
        .def_readonly("n_iter", &vibeqc::semiempirical::xtb::KPointGFN2Result::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::xtb::KPointGFN2Result::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::xtb::KPointGFN2Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::xtb::KPointGFN2Result& result) {
                return result.parameter_sha256;
            });

    m_xtb.def(
        "run_gfn2_xtb_kpoints",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const vibeqc::BlochKMesh& kmesh,
           const vibeqc::semiempirical::xtb::XTBSccOptions& scc_opts,
           double cutoff_bohr,
           bool allow_unconverged) {
            // Preserve the fail-closed #351/domain error precedence.  Those
            // paths intentionally do not consume params, so malformed dormant
            // parameter builders must not mask their public route error.
            auto result = [&]() {
                if (system.multiplicity != 1) {
                    py::gil_scoped_release release;
                    return vibeqc::semiempirical::xtb::run_gfn2_xtb_kpoints(
                        system, params, kmesh, scc_opts, cutoff_bohr);
                }
                vibeqc::semiempirical::xtb::
                    validate_periodic_gfn2_physics_domain(system);
                if (!is_single_gamma_gfn2_mesh(kmesh)) {
                    py::gil_scoped_release release;
                    return vibeqc::semiempirical::xtb::run_gfn2_xtb_kpoints(
                        system, params, kmesh, scc_opts, cutoff_bohr);
                }
                return run_with_identified_parameter_snapshot(
                    params,
                    [&](const auto& snapshot) {
                        return vibeqc::semiempirical::xtb::run_gfn2_xtb_kpoints(
                            system, snapshot, kmesh, scc_opts, cutoff_bohr);
                    });
            }();
            // Fail loud on SCC budget exhaustion (issue #342): callers
            // that screen on "did not raise" must never admit an
            // unconverged record.
            if (!result.converged && !allow_unconverged) {
                throw SCFNonConvergenceError(scf_nonconvergence_message(
                    "periodic GFN2-xTB k-point", result.n_iter,
                    scc_opts.conv_tol_charge));
            }
            return result;
        },
        py::arg("system"), py::arg("params"), py::arg("kmesh"),
        py::arg("scc_opts") = vibeqc::semiempirical::xtb::XTBSccOptions{},
        py::arg("cutoff_bohr") = 15.0,
        py::arg("allow_unconverged") = false,
        "Run GFN2-xTB through a one-point Gamma mesh. Non-Gamma "
        "meshes fail closed until the full complex/AES model exists. "
        "Raises SCFNonConvergenceError when the SCC loop exhausts "
        "max_iter unless allow_unconverged=True.");

    m_xtb.def(
        "run_gfn2_xtb_bandpath",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const std::vector<Eigen::Vector3d>& kpath,
           const vibeqc::BlochKMesh& reference_mesh,
           const vibeqc::semiempirical::xtb::XTBSccOptions& scc_opts,
           double cutoff_bohr) {
            // Band paths are unconditionally fail-closed (#351) after their
            // parameter-independent argument checks and never execute params.
            // Keep that exact contract instead of hashing irrelevant input.
            py::gil_scoped_release release;
            return vibeqc::semiempirical::xtb::run_gfn2_xtb_bandpath(
                system,
                params,
                kpath,
                reference_mesh,
                scc_opts,
                cutoff_bohr);
        },
        py::arg("system"), py::arg("params"), py::arg("kpath"),
        py::arg("reference_mesh"),
        py::arg("scc_opts") = vibeqc::semiempirical::xtb::XTBSccOptions{},
        py::arg("cutoff_bohr") = 15.0,
        "Fail closed for GFN2-xTB band paths until full complex Bloch "
        "phases, phase-resolved energy, and AES parity are implemented.");

    // Periodic GFN2-xTB
    py::class_<vibeqc::semiempirical::xtb::PeriodicGFN2Result>(
        m_xtb, "PeriodicGFN2Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::energy)
        .def_readonly("free_energy", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::free_energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::e_electronic)
        .def_readonly("e_repulsive", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::e_repulsive)
        .def_readonly("e_scc", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::e_scc)
        .def_readonly("e_band0", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::e_band0)
        .def_readonly("e_aes", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::e_aes)
        .def_readonly("e_3rd", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::e_3rd)
        .def_readonly("fermi_level", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::fermi_level)
        .def_readonly("entropy", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::entropy)
        .def_readonly("smearing_temperature", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::smearing_temperature)
        .def_readonly("occupations", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::occupations)
        .def_readonly("converged", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::converged)
        .def_readonly("n_iter", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::n_iter)
        .def_readonly("n_cells", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::n_cells)
        .def_readonly("cutoff_bohr", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::cutoff_bohr)
        // Read-only diagnostics of the Gamma-point assembly (SCC-map
        // analysis probes, IID 130/300; a max_iter=1 run from the
        // neutral start exposes the bare Bloch-fold H0/S).
        .def_readonly("mo_energies", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::mo_energies)
        .def_readonly("overlap_gamma", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::overlap_gamma)
        .def_readonly("hamiltonian_gamma", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::hamiltonian_gamma)
        .def_readonly("charges", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::charges)
        .def_readonly("dq_shell", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::dq_shell)
        .def_readonly("atom_dipoles",
                      &vibeqc::semiempirical::xtb::PeriodicGFN2Result::atom_dipoles,
                      "Converged CAMM dipoles (n_atoms x 3).")
        .def_readonly("atom_quadrupoles",
                      &vibeqc::semiempirical::xtb::PeriodicGFN2Result::atom_quadrupoles,
                      "Converged traceless CAMM quadrupoles (n_atoms x 6: xx, xy, xz, yy, yz, zz).")
        .def_readonly("gamma_form",
                      &vibeqc::semiempirical::xtb::PeriodicGFN2Result::gamma_form)
        .def_readonly("scc_max_change_trace",
                      &vibeqc::semiempirical::xtb::PeriodicGFN2Result::scc_max_change_trace,
                      "Per-iteration max |X_new - X| over charges and moments.")
        .def_readonly("n_basis", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::n_occ)
        .def_readonly("n_shells", &vibeqc::semiempirical::xtb::PeriodicGFN2Result::n_shells)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::xtb::PeriodicGFN2Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::xtb::PeriodicGFN2Result& result) {
                return result.parameter_sha256;
            });

    m_xtb.def(
        "run_gfn2_xtb_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const vibeqc::semiempirical::xtb::XTBSccOptions& scc_opts,
           double cutoff_bohr) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::xtb::run_gfn2_xtb_gamma(
                        system, snapshot, scc_opts, cutoff_bohr);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("scc_opts") = vibeqc::semiempirical::xtb::XTBSccOptions{},
        py::arg("cutoff_bohr") = 15.0,
        "Run periodic GFN2-xTB SCC calculation at Gamma-point.");

    m_semi.def(
        "compute_periodic_gfn2_gradient",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::xtb::PeriodicGFN2Result& result,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params) {
            // Preserve the derivative API's lattice provenance error before
            // checking the newly added parameter snapshot provenance.
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_gfn2_gradient: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic GFN2-xTB gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_periodic_gfn2_gradient(
                        system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
               "Periodic GFN2-xTB atomic gradient at Gamma-point.");

    m_semi.def(
        "compute_periodic_gfn2_stress",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::xtb::PeriodicGFN2Result& result,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params) {
            if (!(result.cutoff_bohr > 0.0)) {
                throw std::invalid_argument(
                    "compute_periodic_gfn2_stress: result carries no "
                    "lattice-cutoff provenance; re-run the SCF with the "
                    "current driver");
            }
            return run_with_matching_parameter_snapshot(
                params, result, "periodic GFN2-xTB stress",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::compute_periodic_gfn2_stress(
                        system, result, snapshot);
                });
        },
        py::arg("system"), py::arg("result"), py::arg("params"),
                "Periodic GFN2-xTB stress tensor at Gamma-point.");

        // Unrestricted GFN2-xTB
    py::class_<vibeqc::semiempirical::xtb::UGFN2Result>(
        m_xtb, "UGFN2Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::xtb::UGFN2Result::energy)
        .def_readonly("n_alpha", &vibeqc::semiempirical::xtb::UGFN2Result::n_alpha)
        .def_readonly("n_beta", &vibeqc::semiempirical::xtb::UGFN2Result::n_beta)
        .def_readonly("n_iter", &vibeqc::semiempirical::xtb::UGFN2Result::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::xtb::UGFN2Result::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::xtb::UGFN2Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::xtb::UGFN2Result& result) {
                return result.parameter_sha256;
            });

    m_xtb.def(
        "run_ugfn2_xtb",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           const vibeqc::semiempirical::xtb::XTBSccOptions& opts) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::xtb::run_ugfn2_xtb(
                        mol, snapshot, opts);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::xtb::XTBSccOptions{},
        "Run unrestricted GFN2-xTB SCC calculation.");

    // --- NDDO methods ---
    py::module_ m_nddo = m_semi.def_submodule("nddo", "NDDO-family methods (MNDO/AM1/PMx/OMx)");

    // --- INDO family (MSINDO) ---
    py::module_ m_indo = m_semi.def_submodule("indo", "INDO-family methods (MSINDO)");

    // MsindoParameterSet — full Z=1..54 parameter table.
    py::class_<vibeqc::semiempirical::indo::MsindoParameterSet>(
        m_indo, "MsindoParameterSet")
        .def(py::init<>())
        .def_readwrite("MUS", &vibeqc::semiempirical::indo::MsindoParameterSet::MUS)
        .def_readwrite("MUP", &vibeqc::semiempirical::indo::MsindoParameterSet::MUP)
        .def_readwrite("MUD", &vibeqc::semiempirical::indo::MsindoParameterSet::MUD)
        .def_readwrite("MUSE", &vibeqc::semiempirical::indo::MsindoParameterSet::MUSE)
        .def_readwrite("MUPE", &vibeqc::semiempirical::indo::MsindoParameterSet::MUPE)
        .def_readwrite("MUDE", &vibeqc::semiempirical::indo::MsindoParameterSet::MUDE)
        .def_readwrite("KSS", &vibeqc::semiempirical::indo::MsindoParameterSet::KSS)
        .def_readwrite("KPS", &vibeqc::semiempirical::indo::MsindoParameterSet::KPS)
        .def_readwrite("KPP", &vibeqc::semiempirical::indo::MsindoParameterSet::KPP)
        .def_readwrite("KDS", &vibeqc::semiempirical::indo::MsindoParameterSet::KDS)
        .def_readwrite("KDP", &vibeqc::semiempirical::indo::MsindoParameterSet::KDP)
        .def_readwrite("KDD", &vibeqc::semiempirical::indo::MsindoParameterSet::KDD)
        .def_readwrite("IPOTS", &vibeqc::semiempirical::indo::MsindoParameterSet::IPOTS)
        .def_readwrite("IPOTP", &vibeqc::semiempirical::indo::MsindoParameterSet::IPOTP)
        .def_readwrite("IPOTD", &vibeqc::semiempirical::indo::MsindoParameterSet::IPOTD)
        .def_readwrite("TAU1S", &vibeqc::semiempirical::indo::MsindoParameterSet::TAU1S)
        .def_readwrite("TAU2S", &vibeqc::semiempirical::indo::MsindoParameterSet::TAU2S)
        .def_readwrite("TAU2P", &vibeqc::semiempirical::indo::MsindoParameterSet::TAU2P)
        .def_readwrite("TAU3S", &vibeqc::semiempirical::indo::MsindoParameterSet::TAU3S)
        .def_readwrite("TAU3P", &vibeqc::semiempirical::indo::MsindoParameterSet::TAU3P)
        .def_readwrite("TAU3D", &vibeqc::semiempirical::indo::MsindoParameterSet::TAU3D)
        .def_readwrite("FCP1S", &vibeqc::semiempirical::indo::MsindoParameterSet::FCP1S)
        .def_readwrite("FCP2S", &vibeqc::semiempirical::indo::MsindoParameterSet::FCP2S)
        .def_readwrite("FCP2P", &vibeqc::semiempirical::indo::MsindoParameterSet::FCP2P)
        .def_readwrite("FCP3S", &vibeqc::semiempirical::indo::MsindoParameterSet::FCP3S)
        .def_readwrite("FCP3P", &vibeqc::semiempirical::indo::MsindoParameterSet::FCP3P)
        .def_readwrite("FCP3D", &vibeqc::semiempirical::indo::MsindoParameterSet::FCP3D)
        .def_readwrite("SCP3D", &vibeqc::semiempirical::indo::MsindoParameterSet::SCP3D)
        .def_readwrite("SCP4S", &vibeqc::semiempirical::indo::MsindoParameterSet::SCP4S)
        .def_readwrite("SCP4P", &vibeqc::semiempirical::indo::MsindoParameterSet::SCP4P)
        .def_readwrite("LS", &vibeqc::semiempirical::indo::MsindoParameterSet::LS)
        .def_readwrite("MP", &vibeqc::semiempirical::indo::MsindoParameterSet::MP)
        .def_readwrite("ND", &vibeqc::semiempirical::indo::MsindoParameterSet::ND)
        .def_static("embedded_hf", &vibeqc::semiempirical::indo::MsindoParameterSet::embedded_hf,
                    "Return the built-in H–F embedded parameter set (backward compatibility).");

    // JSON loaders
    m_indo.def("load_params_from_json",
               &vibeqc::semiempirical::indo::load_msindo_params_from_json,
               py::arg("json_str"),
               "Load a MsindoParameterSet from msindo_params.json content.");
    m_indo.def("merge_nddo_overrides",
               &vibeqc::semiempirical::indo::merge_msindo_nddo_overrides,
               py::arg("base"), py::arg("json_str"),
               "Merge NDDO parameter overrides from msindo_params_nddo.json into base set.");

    // MsindoResult
    py::class_<vibeqc::semiempirical::indo::MsindoResult>(m_indo, "MsindoResult")
        .def(py::init<>())
        .def_readonly("total_energy", &vibeqc::semiempirical::indo::MsindoResult::total_energy)
        .def_readonly("electronic_energy", &vibeqc::semiempirical::indo::MsindoResult::electronic_energy)
        .def_readonly("binding_energy", &vibeqc::semiempirical::indo::MsindoResult::binding_energy)
        .def_readonly("mo_energies", &vibeqc::semiempirical::indo::MsindoResult::mo_energies)
        .def_readonly("density", &vibeqc::semiempirical::indo::MsindoResult::density)
        .def_readonly("n_iter", &vibeqc::semiempirical::indo::MsindoResult::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::indo::MsindoResult::converged);

    // Backward-compatible run_msindo (embedded H-F, no params arg).
    m_indo.def("run_msindo",
               py::overload_cast<const std::vector<int>&,
                                 const std::vector<std::array<double, 3>>&,
                                 int, double, int>(
                   &vibeqc::semiempirical::indo::run_msindo_core),
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("max_iter") = 200, py::arg("conv_tol") = 1e-9,
               py::arg("charge") = 0,
               py::call_guard<py::gil_scoped_release>(),
               "Run closed-shell MSINDO SCF (H-F, s/p shells, embedded params). "
               "coords_angstrom is a list of [x,y,z] in Angstrom.");

    // Full parameter-driven run_msindo (RHF, s/p/d, optional closed-shell NDDO).
    m_indo.def("run_msindo_full",
               py::overload_cast<const std::vector<int>&,
                                 const std::vector<std::array<double, 3>>&,
                                 const vibeqc::semiempirical::indo::MsindoParameterSet&,
                                 int, double, bool, int>(
                   &vibeqc::semiempirical::indo::run_msindo_core),
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("params"), py::arg("max_iter") = 200,
               py::arg("conv_tol") = 1e-9, py::arg("nddo") = false,
               py::arg("charge") = 0,
               py::call_guard<py::gil_scoped_release>(),
               "Run closed-shell MSINDO SCF with a full parameter set (s/p/d, "
               "full Z=1..54 INDO scope). Set nddo=True for closed-shell NDDO "
               "(H, Li-F, and Na-Cl, including the Al-Cl SPDD terms). "
               "coords_angstrom is a list of [x,y,z] in Angstrom.");

    // Open-shell UHF driver.
    m_indo.def("run_msindo_uhf",
               [](const std::vector<int>& Z_in,
                  const std::vector<std::array<double, 3>>& coords_in,
                  const vibeqc::semiempirical::indo::MsindoParameterSet& params_in,
                  int max_iter_in, double conv_tol_in, int mult_in, bool nddo_in,
                  int charge_in) {
                   return vibeqc::semiempirical::indo::run_msindo_core_uhf(
                       Z_in, coords_in, params_in, max_iter_in, conv_tol_in,
                       mult_in, nddo_in, charge_in);
               },
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("params"), py::arg("max_iter") = 200,
               py::arg("conv_tol") = 1e-9, py::arg("multiplicity") = 2,
               py::arg("nddo") = false,
               py::arg("charge") = 0,
               py::call_guard<py::gil_scoped_release>(),
               "Run open-shell MSINDO INDO UHF SCF (s/p/d elements); NDDO is "
               "closed-shell only and nddo=True is rejected.");

    // --- Integral kernels (exposed for gradient testing) ---
    m_indo.def("s2int", &vibeqc::semiempirical::indo::s2int,
               py::arg("n1"), py::arg("l1"), py::arg("m1"), py::arg("z1"),
               py::arg("n2"), py::arg("l2"), py::arg("m2"), py::arg("z2"),
               py::arg("R"),
               "Two-center overlap <mu_a|nu_b>.");
    m_indo.def("ds2int", &vibeqc::semiempirical::indo::ds2int,
               py::arg("n1"), py::arg("l1"), py::arg("m1"), py::arg("z1"),
               py::arg("n2"), py::arg("l2"), py::arg("m2"), py::arg("z2"),
               py::arg("R"),
               "Derivative d/dR of two-center overlap.");
    m_indo.def("c2int", &vibeqc::semiempirical::indo::c2int,
               py::arg("n1"), py::arg("l1"), py::arg("m1"), py::arg("z1"),
               py::arg("n2"), py::arg("l2"), py::arg("m2"), py::arg("z2"),
               py::arg("R"),
               "Two-center Coulomb integral (monopole gamma).");
    m_indo.def("dc2int", &vibeqc::semiempirical::indo::dc2int,
               py::arg("n1"), py::arg("l1"), py::arg("m1"), py::arg("z1"),
               py::arg("n2"), py::arg("l2"), py::arg("m2"), py::arg("z2"),
               py::arg("R"),
               "Derivative d/dR of two-center Coulomb integral.");
    m_indo.def("v2int", &vibeqc::semiempirical::indo::v2int,
               py::arg("n1"), py::arg("l1"), py::arg("m1"), py::arg("z1"),
               py::arg("R"),
               "One-center nuclear attraction <mu_a|1/r_b|mu_a>.");
    m_indo.def("dv2int", &vibeqc::semiempirical::indo::dv2int,
               py::arg("n1"), py::arg("l1"), py::arg("m1"), py::arg("z1"),
               py::arg("R"),
               "Derivative d/dR of one-center nuclear attraction.");
    m_indo.def("cosmo_b_matrix",
               [](const std::vector<int>& Z,
                  const Eigen::Ref<const Eigen::MatrixXd>& coords_bohr,
                  const Eigen::Ref<const Eigen::MatrixXd>& cavity_points,
                  const vibeqc::semiempirical::indo::MsindoParameterSet& params) {
                   auto res = vibeqc::semiempirical::indo::build_cosmo_b_matrix(
                       Z, coords_bohr, cavity_points, params);
                   return std::make_tuple(std::move(res.B), std::move(res.pairs), res.nsto);
               },
               py::arg("atomic_numbers"), py::arg("coords_bohr"),
               py::arg("cavity_points"), py::arg("params"),
               py::call_guard<py::gil_scoped_release>(),
               "Build the MSINDO COSMO distributed-multipole B matrix. "
               "Returns (B, pairs, nsto), where pairs are (component, mu, nu).");
    m_indo.def("cosmo_charge_vector",
               &vibeqc::semiempirical::indo::cosmo_charge_vector,
               py::arg("density"), py::arg("nsto"), py::arg("pairs"),
               py::call_guard<py::gil_scoped_release>(),
               "Build the MSINDO COSMO electronic charge-component vector.");
    m_indo.def("cosmo_esp_at_cavity",
               &vibeqc::semiempirical::indo::cosmo_esp_at_cavity,
               py::arg("B"), py::arg("density"), py::arg("nsto"),
               py::arg("pairs"),
               py::call_guard<py::gil_scoped_release>(),
               "Apply the MSINDO COSMO B^T Q electronic surface potential.");
    m_indo.def("cosmo_fock_contribution",
               &vibeqc::semiempirical::indo::cosmo_fock_contribution,
               py::arg("B"), py::arg("charges"), py::arg("nsto"),
               py::arg("pairs"),
               py::call_guard<py::gil_scoped_release>(),
               "Scatter the MSINDO COSMO B q reaction field into an AO matrix.");
    m_indo.def("cosmo_core_potential_at_cavity",
               &vibeqc::semiempirical::indo::cosmo_core_potential_at_cavity,
               py::arg("coords_bohr"), py::arg("core_charges"),
               py::arg("cavity_points"),
               py::call_guard<py::gil_scoped_release>(),
               "Build the MSINDO COSMO effective-core potential on cavity points.");
    py::class_<vibeqc::semiempirical::indo::MsindoCosmoReactionField>(
        m_indo, "MsindoCosmoReactionField")
        .def_readonly("fock", &vibeqc::semiempirical::indo::MsindoCosmoReactionField::fock)
        .def_readonly("q", &vibeqc::semiempirical::indo::MsindoCosmoReactionField::q)
        .def_readonly("V_total", &vibeqc::semiempirical::indo::MsindoCosmoReactionField::V_total)
        .def_readonly("e_add", &vibeqc::semiempirical::indo::MsindoCosmoReactionField::e_add)
        .def_readonly("e_pol", &vibeqc::semiempirical::indo::MsindoCosmoReactionField::e_pol);
    m_indo.def("cosmo_reaction_field",
               &vibeqc::semiempirical::indo::cosmo_reaction_field,
               py::arg("B"), py::arg("A"), py::arg("core_potential"),
               py::arg("density"), py::arg("nsto"), py::arg("pairs"),
               py::arg("epsilon"), py::arg("variant") = "cosmo",
               py::call_guard<py::gil_scoped_release>(),
               "Build the MSINDO COSMO reaction-field Fock contribution for "
               "one density: V=B^TQ+Vcore, Aq=-fV, F=Bq. The screening factor "
               "f comes from the shared cpcm_dielectric_factor, so 'variant' "
               "selects CPCM or COSMO screening rather than the COSMO form "
               "being assumed.");
    py::class_<vibeqc::semiempirical::indo::MsindoGepolCavity>(
        m_indo, "MsindoGepolCavity")
        .def_readonly("points", &vibeqc::semiempirical::indo::MsindoGepolCavity::points)
        .def_readonly("a_diag", &vibeqc::semiempirical::indo::MsindoGepolCavity::a_diag)
        .def_readonly("area", &vibeqc::semiempirical::indo::MsindoGepolCavity::area)
        .def_readonly("seg_atom", &vibeqc::semiempirical::indo::MsindoGepolCavity::seg_atom);
    m_indo.def("gepol_build_cavity",
               &vibeqc::semiempirical::indo::build_msindo_gepol_cavity,
               py::arg("atomic_numbers"), py::arg("coords_bohr"),
               py::arg("coarse") = 0, py::arg("fine") = 3,
               py::arg("rsolve") = 0.0, py::arg("rscale") = 0.1,
               py::call_guard<py::gil_scoped_release>(),
               "Build the MSINDO GEPOL/SAS COSMO cavity. "
               "Coordinates are in bohr; rsolve is in Angstrom. Returns points, "
               "diagonal self-terms, segment areas, and owners.");

    // --- Periodic CCM ---
    m_indo.def(
        "_build_seccm_topology_records_for_testing",
        [](const Eigen::Ref<const Eigen::MatrixXd>& coords_bohr,
           const Eigen::Ref<const Eigen::MatrixXd>& translations_bohr) {
            if (coords_bohr.cols() != 3 || coords_bohr.rows() < 1) {
                throw std::invalid_argument(
                    "SECCM test topology coordinates must have shape (N, 3) "
                    "with N >= 1");
            }
            if (translations_bohr.cols() != 3
                || translations_bohr.rows() < 1
                || translations_bohr.rows() > 3) {
                throw std::invalid_argument(
                    "SECCM test topology translations must have shape (D, 3) "
                    "with 1 <= D <= 3");
            }
            if (!coords_bohr.allFinite() || !translations_bohr.allFinite()) {
                throw std::invalid_argument(
                    "SECCM test topology inputs must contain only finite values");
            }

            const double translation_scale =
                translations_bohr.rowwise().norm().maxCoeff();
            const double rank_tolerance = std::max(
                1.0e-12,
                64.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, translation_scale));
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                translations_bohr,
                Eigen::ComputeThinU | Eigen::ComputeThinV);
            if (translation_scale <= rank_tolerance
                || svd.singularValues().minCoeff() <= rank_tolerance) {
                throw std::invalid_argument(
                    "SECCM test topology translations must be nonzero and "
                    "linearly independent (absolute rank tolerance 1e-12, "
                    "with a scale-aware floating-point floor)");
            }

            std::vector<Eigen::Vector3d> coords;
            coords.reserve(static_cast<std::size_t>(coords_bohr.rows()));
            for (Eigen::Index row = 0; row < coords_bohr.rows(); ++row) {
                coords.push_back(coords_bohr.row(row).transpose());
            }
            std::vector<Eigen::Vector3d> translations;
            translations.reserve(
                static_cast<std::size_t>(translations_bohr.rows()));
            for (Eigen::Index row = 0; row < translations_bohr.rows(); ++row) {
                translations.push_back(translations_bohr.row(row).transpose());
            }

            const auto topology =
                vibeqc::semiempirical::seccm::build_wigner_seitz(
                    coords, translations);
            py::dict result;
            result["schema"] = "seccm-ws-cxx-records-v1";
            py::list records;
            py::list total_weights;
            for (std::size_t central = 0; central < topology.cells.size();
                 ++central) {
                const auto& cell = topology.cells[central];
                for (std::size_t record_index = 0;
                     record_index < cell.size(); ++record_index) {
                    const auto& image = cell[record_index];
                    py::dict record;
                    record["central"] = central;
                    record["record_index"] = record_index;
                    record["origin"] = image.origin;
                    record["shell_label"] = py::make_tuple(
                        image.image_shell_label[0],
                        image.image_shell_label[1],
                        image.image_shell_label[2]);
                    record["ownership_multiplicity"] =
                        image.ownership_multiplicity;
                    record["weight"] = image.weight;
                    record["displacement"] = py::make_tuple(
                        image.disp.x(), image.disp.y(), image.disp.z());
                    records.append(std::move(record));
                }
                total_weights.append(topology.total_weight(central));
            }
            result["records"] = std::move(records);
            result["total_weights"] = std::move(total_weights);
            result["valid"] = topology.is_valid(
                static_cast<int>(coords.size()));
            return result;
        },
        py::arg("coords_bohr"), py::arg("translations_bohr"),
        "Private deterministic SECCM Wigner-Seitz record bridge for tests.");

    m_indo.def("run_ccm",
               &vibeqc::semiempirical::indo::run_ccm_core,
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("translations_angstrom"), py::arg("params"),
               py::arg("madelung") = false,
               py::arg("max_iter") = 200, py::arg("conv_tol") = 1e-9,
               py::arg("charge") = 0,
               py::call_guard<py::gil_scoped_release>(),
               "Run periodic MSINDO SCF in the Cyclic Cluster Model (CCM). "
               "coords_angstrom / translations_angstrom are lists of [x,y,z] in Angstrom. "
               "1 vector → CCM1D (polymer), 2 → CCM2D (surface), 3 → CCM3D (bulk).");

    m_indo.def("ccm_gradient_fd",
               &vibeqc::semiempirical::indo::ccm_gradient_fd,
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("translations_angstrom"), py::arg("params"),
               py::arg("madelung") = false,
               py::arg("atoms") = std::vector<int>{},
               py::arg("max_iter") = 200, py::arg("conv_tol") = 1e-10,
               py::arg("step") = 1e-3, py::arg("charge") = 0,
               py::call_guard<py::gil_scoped_release>(),
               "Finite-difference CCM nuclear gradient (Ha/bohr). "
               "Holds Wigner-Seitz topology fixed at the reference geometry.");

    m_indo.def("ccm_gradient_analytic",
               &vibeqc::semiempirical::indo::ccm_gradient_analytic,
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("translations_angstrom"), py::arg("params"),
               py::arg("madelung") = false, py::arg("charge") = 0,
               py::arg("max_iter") = 200, py::arg("conv_tol") = 1e-10,
               py::call_guard<py::gil_scoped_release>(),
               "Analytic CCM nuclear gradient (Ha/bohr), closed-shell RHF. "
               "WS topology fixed at the reference geometry; direct-assembly "
               "Madelung gradient (1-D / 2-D Parry-Heyes / 3-D Ewald).");

    // --- Analytic molecular gradient (Phase 4) ---
    m_indo.def("gradient_analytic",
               &vibeqc::semiempirical::indo::msindo_gradient_analytic_core,
               py::arg("atomic_numbers"), py::arg("coords_angstrom"),
               py::arg("params"), py::arg("max_iter") = 200,
               py::arg("conv_tol") = 1e-10,
               py::arg("charge") = 0,
               py::call_guard<py::gil_scoped_release>(),
               "Analytic nuclear gradient (Ha/bohr) for RHF MSINDO.");

    // PairBlocksDeriv struct (for testing)
    py::class_<vibeqc::semiempirical::indo::PairBlocksDeriv>(m_indo, "PairBlocksDeriv")
        .def(py::init<>())
        .def_readonly("dHK1_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::dHK1_local)
        .def_readonly("dHL1_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::dHL1_local)
        .def_readonly("dHKL2_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::dHKL2_local)
        .def_readonly("d_gamma_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::d_gamma_local)
        .def_readonly("HK1_DR", &vibeqc::semiempirical::indo::PairBlocksDeriv::HK1_DR)
        .def_readonly("HL1_DR", &vibeqc::semiempirical::indo::PairBlocksDeriv::HL1_DR)
        .def_readonly("HKL2_DR", &vibeqc::semiempirical::indo::PairBlocksDeriv::HKL2_DR)
        .def_readonly("M_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::M_local)
        .def_readonly("dM_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::dM_local)
        .def_readonly("S_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::S_local)
        .def_readonly("dS_local", &vibeqc::semiempirical::indo::PairBlocksDeriv::dS_local)
        .def_readonly("R", &vibeqc::semiempirical::indo::PairBlocksDeriv::R);

    m_indo.def("pair_blocks_deriv",
               &vibeqc::semiempirical::indo::pair_blocks_deriv,
               py::arg("zk"), py::arg("zl"), py::arg("rk"), py::arg("rl"),
               py::arg("params"),
               py::call_guard<py::gil_scoped_release>(),
               "Compute per-pair integral derivatives.");

    py::class_<vibeqc::semiempirical::nddo::NDDOElementData>(
        m_nddo, "NDDOElementData")
        .def(py::init<>())
        .def_readwrite("Z", &vibeqc::semiempirical::nddo::NDDOElementData::Z)
        .def_readwrite("uss", &vibeqc::semiempirical::nddo::NDDOElementData::uss)
        .def_readwrite("upp", &vibeqc::semiempirical::nddo::NDDOElementData::upp)
        .def_readwrite("udd", &vibeqc::semiempirical::nddo::NDDOElementData::udd)
        .def_readwrite("betas", &vibeqc::semiempirical::nddo::NDDOElementData::betas)
        .def_readwrite("betap", &vibeqc::semiempirical::nddo::NDDOElementData::betap)
        .def_readwrite("betad", &vibeqc::semiempirical::nddo::NDDOElementData::betad)
        .def_readwrite("zs", &vibeqc::semiempirical::nddo::NDDOElementData::zs)
        .def_readwrite("zp", &vibeqc::semiempirical::nddo::NDDOElementData::zp)
        .def_readwrite("zd", &vibeqc::semiempirical::nddo::NDDOElementData::zd)
        .def_readwrite("zsn", &vibeqc::semiempirical::nddo::NDDOElementData::zsn)
        .def_readwrite("zpn", &vibeqc::semiempirical::nddo::NDDOElementData::zpn)
        .def_readwrite("zdn", &vibeqc::semiempirical::nddo::NDDOElementData::zdn)
        .def_readwrite("gss", &vibeqc::semiempirical::nddo::NDDOElementData::gss)
        .def_readwrite("gpp", &vibeqc::semiempirical::nddo::NDDOElementData::gpp)
        .def_readwrite("gsp", &vibeqc::semiempirical::nddo::NDDOElementData::gsp)
        .def_readwrite("gp2", &vibeqc::semiempirical::nddo::NDDOElementData::gp2)
        .def_readwrite("hsp", &vibeqc::semiempirical::nddo::NDDOElementData::hsp)
        .def_readwrite("f0sd", &vibeqc::semiempirical::nddo::NDDOElementData::f0sd)
        .def_readwrite("g2sd", &vibeqc::semiempirical::nddo::NDDOElementData::g2sd)
        .def_readwrite("alpha", &vibeqc::semiempirical::nddo::NDDOElementData::alpha)
        .def_readwrite("pcore", &vibeqc::semiempirical::nddo::NDDOElementData::pcore)
        .def_readwrite("polvo", &vibeqc::semiempirical::nddo::NDDOElementData::polvo)
        .def_readwrite("cpe_zet", &vibeqc::semiempirical::nddo::NDDOElementData::cpe_zet)
        .def_readwrite("cpe_z0", &vibeqc::semiempirical::nddo::NDDOElementData::cpe_z0)
        .def_readwrite("cpe_b", &vibeqc::semiempirical::nddo::NDDOElementData::cpe_b)
        .def_readwrite("cpe_xlo", &vibeqc::semiempirical::nddo::NDDOElementData::cpe_xlo)
        .def_readwrite("cpe_xhi", &vibeqc::semiempirical::nddo::NDDOElementData::cpe_xhi)
        .def_readwrite("has_cpe", &vibeqc::semiempirical::nddo::NDDOElementData::has_cpe)
        .def_readwrite("has_d", &vibeqc::semiempirical::nddo::NDDOElementData::has_d)
        .def_readwrite("n_orbitals", &vibeqc::semiempirical::nddo::NDDOElementData::n_orbitals)
                .def("add_gamma_term", [](vibeqc::semiempirical::nddo::NDDOElementData& ed, double coeff, double exponent, double factor) {
                    ed.gamma_terms.push_back({coeff, exponent, factor});
                })
                .def("add_gaussian_term", &vibeqc::semiempirical::nddo::NDDOElementData::add_gaussian_term,
                     py::arg("coeff"), py::arg("exponent"), py::arg("factor"));

    py::class_<vibeqc::semiempirical::nddo::PM6ParameterSet,
                 vibeqc::semiempirical::CoreParameterSet>(
        m_nddo, "PM6ParameterSet")
        .def(py::init<std::string>(), py::arg("method_name") = "pm6")
        .def("has_element", &vibeqc::semiempirical::nddo::PM6ParameterSet::has_element)
        .def("n_elements", &vibeqc::semiempirical::nddo::PM6ParameterSet::n_elements)
        .def("element", [](const vibeqc::semiempirical::nddo::PM6ParameterSet& p,
                            int Z) {
            return p.element(Z);
        }, py::arg("Z"), "Return a detached base-element snapshot.")
        .def("element_data", [](const vibeqc::semiempirical::nddo::PM6ParameterSet& p,
                                 int Z) {
            const auto* element = p.element_data(Z);
            if (element == nullptr)
                throw py::key_error("element Z=" + std::to_string(Z));
            return *element;
        }, py::arg("Z"), "Return a detached snapshot of one element record.")
        .def("add_element", &vibeqc::semiempirical::nddo::PM6ParameterSet::add_element)
        .def("add_diatomic", [](vibeqc::semiempirical::nddo::PM6ParameterSet& p,
                                  int Z1, int Z2, double alpb, double xfac) {
            vibeqc::semiempirical::nddo::NDDODiatomicParams dp;
            dp.d1 = alpb; dp.d2 = xfac;
            p.set_diatomic(Z1, Z2, dp);
        }, py::arg("Z1"), py::arg("Z2"), py::arg("alpb"), py::arg("xfac"))
        .def("metadata", &vibeqc::semiempirical::nddo::PM6ParameterSet::metadata)
        .def("content_sha256",
             &vibeqc::semiempirical::nddo::PM6ParameterSet::content_sha256)
        .def("parameter_identity",
             &vibeqc::semiempirical::nddo::PM6ParameterSet::parameter_identity);

    m_nddo.def(
        "pm6_electron_core_gamma",
        &vibeqc::semiempirical::nddo::pm6_electron_core_gamma,
        py::arg("R"), py::arg("electron_atom"), py::arg("core_atom"),
        "Directed PM6 electron-core Klopman-Ohno monopole (Hartree)."
    );

    py::class_<vibeqc::semiempirical::nddo::PM6SPHydrogenPair>(
        m_nddo, "PM6SPHydrogenPair")
        .def_readonly(
            "electron_repulsion",
            &vibeqc::semiempirical::nddo::PM6SPHydrogenPair::electron_repulsion)
        .def_readonly(
            "heavy_core",
            &vibeqc::semiempirical::nddo::PM6SPHydrogenPair::heavy_core)
        .def_readonly(
            "hydrogen_core",
            &vibeqc::semiempirical::nddo::PM6SPHydrogenPair::hydrogen_core);

    m_nddo.def(
        "pm6_sp_hydrogen_pair",
        &vibeqc::semiempirical::nddo::pm6_sp_hydrogen_pair,
        py::arg("Z_heavy"),
        py::arg("heavy_to_hydrogen"),
        py::arg("heavy_atom"),
        py::arg("hydrogen_atom"),
        "Source-verifiable PM6 s/p-heavy--hydrogen multipole block."
    );

    py::class_<vibeqc::semiempirical::nddo::PM6SPSPPair>(
        m_nddo, "PM6SPSPPair")
        .def_readonly(
            "electron_repulsion",
            &vibeqc::semiempirical::nddo::PM6SPSPPair::electron_repulsion)
        .def_readonly(
            "first_core",
            &vibeqc::semiempirical::nddo::PM6SPSPPair::first_core)
        .def_readonly(
            "second_core",
            &vibeqc::semiempirical::nddo::PM6SPSPPair::second_core);

    m_nddo.def(
        "pm6_sp_sp_pair",
        &vibeqc::semiempirical::nddo::pm6_sp_sp_pair,
        py::arg("Z_first"),
        py::arg("Z_second"),
        py::arg("first_to_second"),
        py::arg("first_atom"),
        py::arg("second_atom"),
        "Source-verifiable PM6 s/p-heavy--s/p-heavy multipole block."
    );

    // --- OMx parameter set ---
    py::class_<vibeqc::semiempirical::nddo::OMxElementData>(
        m_nddo, "OMxElementData")
        .def(py::init<>())
        .def_readwrite("Z", &vibeqc::semiempirical::nddo::OMxElementData::Z)
        .def_readwrite("uss", &vibeqc::semiempirical::nddo::OMxElementData::uss)
        .def_readwrite("upp", &vibeqc::semiempirical::nddo::OMxElementData::upp)
        .def_readwrite("gss", &vibeqc::semiempirical::nddo::OMxElementData::gss)
        .def_readwrite("gpp", &vibeqc::semiempirical::nddo::OMxElementData::gpp)
        .def_readwrite("gsp", &vibeqc::semiempirical::nddo::OMxElementData::gsp)
        .def_readwrite("gp2", &vibeqc::semiempirical::nddo::OMxElementData::gp2)
        .def_readwrite("hsp", &vibeqc::semiempirical::nddo::OMxElementData::hsp)
        .def_readwrite("beta_s", &vibeqc::semiempirical::nddo::OMxElementData::beta_s)
        .def_readwrite("beta_p", &vibeqc::semiempirical::nddo::OMxElementData::beta_p)
        .def_readwrite("beta_pi", &vibeqc::semiempirical::nddo::OMxElementData::beta_pi)
        .def_readwrite("alpha_s", &vibeqc::semiempirical::nddo::OMxElementData::alpha_s)
        .def_readwrite("alpha_p", &vibeqc::semiempirical::nddo::OMxElementData::alpha_p)
        .def_readwrite("alpha_pi", &vibeqc::semiempirical::nddo::OMxElementData::alpha_pi)
        .def_readwrite("F1", &vibeqc::semiempirical::nddo::OMxElementData::F1)
        .def_readwrite("G1", &vibeqc::semiempirical::nddo::OMxElementData::G1)
        .def_readwrite("F2", &vibeqc::semiempirical::nddo::OMxElementData::F2)
        .def_readwrite("G2", &vibeqc::semiempirical::nddo::OMxElementData::G2)
        .def_readwrite("zeta", &vibeqc::semiempirical::nddo::OMxElementData::zeta)
        .def_readwrite("zs", &vibeqc::semiempirical::nddo::OMxElementData::zs)
        .def_readwrite("zp", &vibeqc::semiempirical::nddo::OMxElementData::zp)
        .def_readwrite("alpha", &vibeqc::semiempirical::nddo::OMxElementData::alpha)
        .def_readwrite("n_orbitals", &vibeqc::semiempirical::nddo::OMxElementData::n_orbitals)
        .def_readwrite("beta_s_xh", &vibeqc::semiempirical::nddo::OMxElementData::beta_s_xh)
        .def_readwrite("beta_p_xh", &vibeqc::semiempirical::nddo::OMxElementData::beta_p_xh)
        .def_readwrite("alpha_s_xh", &vibeqc::semiempirical::nddo::OMxElementData::alpha_s_xh)
        .def_readwrite("alpha_p_xh", &vibeqc::semiempirical::nddo::OMxElementData::alpha_p_xh)
        .def_readwrite("zeta_alpha", &vibeqc::semiempirical::nddo::OMxElementData::zeta_alpha)
        .def_readwrite("F_alpha_alpha", &vibeqc::semiempirical::nddo::OMxElementData::F_alpha_alpha)
        .def_readwrite("beta_alpha", &vibeqc::semiempirical::nddo::OMxElementData::beta_alpha)
        .def_readwrite("alpha_alpha", &vibeqc::semiempirical::nddo::OMxElementData::alpha_alpha);

    py::class_<vibeqc::semiempirical::nddo::OMxParameterSet,
                 vibeqc::semiempirical::nddo::PM6ParameterSet>(
        m_nddo, "OMxParameterSet")
        .def(py::init<vibeqc::semiempirical::nddo::OMxVariant>(),
             py::arg("variant"))
        .def("variant", &vibeqc::semiempirical::nddo::OMxParameterSet::variant)
        .def("method_name", &vibeqc::semiempirical::nddo::OMxParameterSet::method_name)
        .def("metadata", &vibeqc::semiempirical::nddo::OMxParameterSet::metadata)
        .def("content_sha256",
             &vibeqc::semiempirical::nddo::OMxParameterSet::content_sha256)
        .def("parameter_identity",
             &vibeqc::semiempirical::nddo::OMxParameterSet::parameter_identity)
        .def(
            "omx_element_data",
            [](const vibeqc::semiempirical::nddo::OMxParameterSet& params,
               int atomic_number) {
                const auto* element = params.omx_data(atomic_number);
                if (element == nullptr) {
                    throw py::key_error(
                        "No OMx parameters for atomic number "
                        + std::to_string(atomic_number));
                }
                // Return a detached builder record.  Mutating it cannot
                // change the parameter set until add_omx_element() is called.
                return *element;
            },
            py::arg("atomic_number"))
        .def("add_omx_element",
             &vibeqc::semiempirical::nddo::OMxParameterSet::add_omx_element);

    // --- PM6 driver ---
    py::class_<vibeqc::semiempirical::nddo::PM6Result>(
        m_nddo, "PM6Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::nddo::PM6Result::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::nddo::PM6Result::e_electronic)
        .def_readonly("e_core", &vibeqc::semiempirical::nddo::PM6Result::e_core)
        .def_readonly("density", &vibeqc::semiempirical::nddo::PM6Result::density)
        .def_readonly("n_basis", &vibeqc::semiempirical::nddo::PM6Result::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::nddo::PM6Result::n_occ)
        .def_readonly("converged", &vibeqc::semiempirical::nddo::PM6Result::converged)
        .def_readonly("n_iter", &vibeqc::semiempirical::nddo::PM6Result::n_iter)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::nddo::PM6Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::nddo::PM6Result& result) {
                return result.parameter_sha256;
            });

    m_nddo.def(
        "compute_pm6_gradient_fd",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::compute_pm6_gradient_fd(
                        mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference PM6 gradient (Hartree/bohr).");

    m_nddo.def(
        "compute_pm6_gradient_fd_from_result",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           const vibeqc::semiempirical::nddo::PM6Result& base,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_matching_parameter_snapshot(
                params, base, "PM6 gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_pm6_gradient_fd_from_result(
                            mol, snapshot, base, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("base"),
        py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference PM6 gradient continued from a converged base.");

    m_nddo.def(
        "compute_upm6_gradient_fd",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::compute_upm6_gradient_fd(
                        mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference UPM6 gradient (Hartree/bohr).");

    m_nddo.def(
        "compute_upm6_gradient_fd_from_result",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           const vibeqc::semiempirical::nddo::UPM6Result& base,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_matching_parameter_snapshot(
                params, base, "UPM6 gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_upm6_gradient_fd(
                            mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("base"),
        py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference UPM6 gradient bound to an energy snapshot.");

    m_nddo.def(
        "run_pm6",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           int max_iter,
           double conv_tol) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_pm6(
                        mol, snapshot, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1e-7,
        "Run PM6 SCF energy calculation.");

    m_nddo.def(
        "run_pm6_with_density",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           const Eigen::MatrixXd& initial_density,
           int max_iter,
           double conv_tol) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_pm6_with_density(
                        mol, snapshot, initial_density, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("initial_density"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1e-7,
        "Run PM6 SCF from a prior converged density.");

    m_nddo.def(
        "run_pm6_with_smearing",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           double electronic_temperature,
           int max_iter,
           double conv_tol) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_pm6_with_smearing(
                        mol,
                        snapshot,
                        electronic_temperature,
                        max_iter,
                        conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("electronic_temperature"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1e-7,
        "Run PM6 SCF with finite-temperature frontier occupations.");

    py::class_<vibeqc::semiempirical::nddo::UPM6Result>(m_nddo, "UPM6Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::nddo::UPM6Result::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::nddo::UPM6Result::e_electronic)
        .def_readonly("e_core", &vibeqc::semiempirical::nddo::UPM6Result::e_core)
        .def_readonly("mo_energies", &vibeqc::semiempirical::nddo::UPM6Result::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::nddo::UPM6Result::mo_coeffs)
        .def_readonly("density_alpha", &vibeqc::semiempirical::nddo::UPM6Result::density_alpha)
        .def_readonly("density_beta", &vibeqc::semiempirical::nddo::UPM6Result::density_beta)
        .def_readonly("n_alpha", &vibeqc::semiempirical::nddo::UPM6Result::n_alpha)
        .def_readonly("n_beta", &vibeqc::semiempirical::nddo::UPM6Result::n_beta)
        .def_readonly("n_basis", &vibeqc::semiempirical::nddo::UPM6Result::n_basis)
        .def_readonly("n_iter", &vibeqc::semiempirical::nddo::UPM6Result::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::nddo::UPM6Result::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::nddo::UPM6Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::nddo::UPM6Result& result) {
                return result.parameter_sha256;
            });

    m_nddo.def(
        "run_upm6",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           int max_iter,
           double conv_tol) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_upm6(
                        mol, snapshot, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1e-7,
        "Run unrestricted PM6 SCF energy calculation.");

    // --- OMx driver ---
    m_nddo.def(
        "run_omx_v2",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           int max_iter,
           double conv_tol) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_omx_v2(
                        mol, snapshot, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1e-7,
        "Run OMx SCF with the proper OM1/OM2/OM3 Hamiltonian "
        "(eq-17 resonance integrals, ORT corrections, OM2/OM3 ECP).");

    m_nddo.def(
        "run_uomx_v2",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           int max_iter,
           double conv_tol) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_uomx_v2(
                        mol, snapshot, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"),
        py::arg("max_iter") = 100,
        py::arg("conv_tol") = 1e-7,
        "Run unrestricted OMx SCF on the proper OM1/OM2/OM3 "
        "Hamiltonian (UHF, stacked-spin DIIS).");

    m_nddo.def(
        "compute_omx_v2_gradient_fd",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::compute_omx_v2_gradient_fd(
                        mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference OMx v2 gradient (Hartree/bohr).");

    m_nddo.def(
        "compute_uomx_v2_gradient_fd",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_uomx_v2_gradient_fd(
                            mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference UOMx v2 gradient (Hartree/bohr).");

    m_nddo.def(
        "compute_omx_v2_gradient_fd_from_result",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           const vibeqc::semiempirical::nddo::PM6Result& base,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_matching_parameter_snapshot(
                params, base, "OMx gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_omx_v2_gradient_fd(
                            mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("base"),
        py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference OMx gradient bound to an energy snapshot.");

    m_nddo.def(
        "compute_uomx_v2_gradient_fd_from_result",
        [](const vibeqc::Molecule& mol,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           const vibeqc::semiempirical::nddo::UPM6Result& base,
           double h,
           int max_iter,
           double conv_tol) {
            return run_with_matching_parameter_snapshot(
                params, base, "unrestricted OMx gradient",
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_uomx_v2_gradient_fd(
                            mol, snapshot, h, max_iter, conv_tol);
                });
        },
        py::arg("mol"), py::arg("params"), py::arg("base"),
        py::arg("h") = 0.001,
        py::arg("max_iter") = 100, py::arg("conv_tol") = 1e-7,
        "Finite-difference unrestricted OMx gradient bound to an energy "
        "snapshot.");

    m_nddo.def("omx_resonance_integral",
              &vibeqc::semiempirical::nddo::omx_resonance_integral,
              py::arg("t_mu"), py::arg("t_nu"),
              py::arg("Z_mu"), py::arg("Z_nu"),
              py::arg("R"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
              py::arg("params"),
              "OMx resonance integral beta_munu in the molecular frame (Ha); "
              "eq 17 of Dral 2016. Reproduces Weber 2000 Table 3.");

        // OMx
    py::enum_<vibeqc::semiempirical::nddo::OMxVariant>(m_nddo, "OMxVariant")
        .value("OM1", vibeqc::semiempirical::nddo::OMxVariant::OM1)
        .value("OM2", vibeqc::semiempirical::nddo::OMxVariant::OM2)
        .value("OM3", vibeqc::semiempirical::nddo::OMxVariant::OM3);

    // --- Periodic PM6 ---
    py::class_<vibeqc::semiempirical::nddo::PeriodicPM6Options>(
        m_nddo, "PeriodicPM6Options")
        .def(py::init<>())
        .def_readwrite("cutoff_bohr", &vibeqc::semiempirical::nddo::PeriodicPM6Options::cutoff_bohr)
        .def_readwrite("conv_tol", &vibeqc::semiempirical::nddo::PeriodicPM6Options::conv_tol)
        .def_readwrite("max_iter", &vibeqc::semiempirical::nddo::PeriodicPM6Options::max_iter)
        .def_readwrite("gamma_only_0", &vibeqc::semiempirical::nddo::PeriodicPM6Options::gamma_only_0);

    py::class_<vibeqc::semiempirical::nddo::PeriodicPM6Result>(
        m_nddo, "PeriodicPM6Result")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::nddo::PeriodicPM6Result::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::nddo::PeriodicPM6Result::e_electronic)
        .def_readonly("e_core", &vibeqc::semiempirical::nddo::PeriodicPM6Result::e_core)
        .def_readonly("mo_energies", &vibeqc::semiempirical::nddo::PeriodicPM6Result::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::nddo::PeriodicPM6Result::mo_coeffs)
        .def_readonly("density", &vibeqc::semiempirical::nddo::PeriodicPM6Result::density)
        .def_readonly("fock_gamma", &vibeqc::semiempirical::nddo::PeriodicPM6Result::fock_gamma)
        .def_readonly("n_basis", &vibeqc::semiempirical::nddo::PeriodicPM6Result::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::nddo::PeriodicPM6Result::n_occ)
        .def_readonly("n_cells", &vibeqc::semiempirical::nddo::PeriodicPM6Result::n_cells)
        .def_readonly("n_iter", &vibeqc::semiempirical::nddo::PeriodicPM6Result::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::nddo::PeriodicPM6Result::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::nddo::PeriodicPM6Result& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::nddo::PeriodicPM6Result& result) {
                return result.parameter_sha256;
            });

    m_nddo.def(
        "run_pm6_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           const vibeqc::semiempirical::nddo::PeriodicPM6Options& opts) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_pm6_gamma(
                        system, snapshot, opts);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::nddo::PeriodicPM6Options{},
        "Run periodic PM6 SCF calculation at Gamma-point.");

    // --- Periodic OMx ---
    py::class_<vibeqc::semiempirical::nddo::PeriodicOMxOptions>(
        m_nddo, "PeriodicOMxOptions")
        .def(py::init<>())
        .def_readwrite("cutoff_bohr", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::cutoff_bohr)
        .def_readwrite("conv_tol", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::conv_tol)
        .def_readwrite("max_iter", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::max_iter)
        .def_readwrite("gamma_only_0", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::gamma_only_0)
        .def_readwrite("warmup_iters", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::warmup_iters)
        .def_readwrite("density_mixing", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::density_mixing)
        .def_readwrite("eval_floor", &vibeqc::semiempirical::nddo::PeriodicOMxOptions::eval_floor);

    py::class_<vibeqc::semiempirical::nddo::PeriodicOMxResult>(
        m_nddo, "PeriodicOMxResult")
        .def(py::init<>())
        .def_readonly("energy", &vibeqc::semiempirical::nddo::PeriodicOMxResult::energy)
        .def_readonly("e_electronic", &vibeqc::semiempirical::nddo::PeriodicOMxResult::e_electronic)
        .def_readonly("e_core", &vibeqc::semiempirical::nddo::PeriodicOMxResult::e_core)
        .def_readonly("mo_energies", &vibeqc::semiempirical::nddo::PeriodicOMxResult::mo_energies)
        .def_readonly("mo_coeffs", &vibeqc::semiempirical::nddo::PeriodicOMxResult::mo_coeffs)
        .def_readonly("density", &vibeqc::semiempirical::nddo::PeriodicOMxResult::density)
        .def_readonly("fock_gamma", &vibeqc::semiempirical::nddo::PeriodicOMxResult::fock_gamma)
        .def_readonly("overlap_gamma", &vibeqc::semiempirical::nddo::PeriodicOMxResult::overlap_gamma)
        .def_readonly("n_basis", &vibeqc::semiempirical::nddo::PeriodicOMxResult::n_basis)
        .def_readonly("n_occ", &vibeqc::semiempirical::nddo::PeriodicOMxResult::n_occ)
        .def_readonly("n_cells", &vibeqc::semiempirical::nddo::PeriodicOMxResult::n_cells)
        .def_readonly("n_iter", &vibeqc::semiempirical::nddo::PeriodicOMxResult::n_iter)
        .def_readonly("converged", &vibeqc::semiempirical::nddo::PeriodicOMxResult::converged)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::nddo::PeriodicOMxResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::nddo::PeriodicOMxResult& result) {
                return result.parameter_sha256;
            });

    m_nddo.def(
        "run_omx_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           const vibeqc::semiempirical::nddo::PeriodicOMxOptions& opts) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::run_omx_gamma(
                        system, snapshot, opts);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("opts") = vibeqc::semiempirical::nddo::PeriodicOMxOptions{},
        "Reject the retired Bloch-periodic OMx prototype.");

    py::class_<vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions>(
        m_nddo, "PeriodicNDDOFDBatchOptions")
        .def(py::init<>())
        .def_readwrite(
            "coordinate_step",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions::coordinate_step)
        .def_readwrite(
            "strain_step",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions::strain_step)
        .def_readwrite(
            "compute_gradient",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions::compute_gradient)
        .def_readwrite(
            "compute_stress",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions::compute_stress)
        .def_readwrite(
            "strain_positions",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions::strain_positions);

    py::class_<vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult>(
        m_nddo, "PeriodicNDDOFDBatchResult")
        .def_readonly(
            "gradient",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::gradient)
        .def_readonly(
            "stress",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::stress)
        .def_readonly(
            "energy_evaluations",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::energy_evaluations)
        .def_readonly(
            "system_workspace_copies",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::system_workspace_copies)
        .def_readonly(
            "lattice_cell_setups",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::lattice_cell_setups)
        .def_readonly(
            "workspace_bytes",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::workspace_bytes)
        .def_readonly(
            "memory_counters_complete",
            &vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult::memory_counters_complete)
        .def_property_readonly(
            "parameter_identity",
            [](const vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult& result) {
                return result.parameter_identity;
            })
        .def_property_readonly(
            "parameter_sha256",
            [](const vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchResult& result) {
                return result.parameter_sha256;
            });

    m_nddo.def(
        "compute_pm6_gamma_fd_batch",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& params,
           const vibeqc::semiempirical::nddo::PeriodicPM6Options& scf_options,
           const vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions&
               fd_options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_pm6_gamma_fd_batch(
                            system, snapshot, scf_options, fd_options);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("scf_options") =
            vibeqc::semiempirical::nddo::PeriodicPM6Options{},
        py::arg("fd_options") =
            vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions{},
        "Compute periodic PM6 gradient and stress in one native FD batch.");
    m_nddo.def(
        "compute_omx_gamma_fd_batch",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::nddo::OMxParameterSet& params,
           const vibeqc::semiempirical::nddo::PeriodicOMxOptions& scf_options,
           const vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions&
               fd_options) {
            return run_with_identified_parameter_snapshot(
                params,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::nddo::
                        compute_omx_gamma_fd_batch(
                            system, snapshot, scf_options, fd_options);
                });
        },
        py::arg("system"), py::arg("params"),
        py::arg("scf_options") =
            vibeqc::semiempirical::nddo::PeriodicOMxOptions{},
        py::arg("fd_options") =
            vibeqc::semiempirical::nddo::PeriodicNDDOFDBatchOptions{},
        "Reject derivatives for the retired Bloch-periodic OMx prototype.");

    m_semi.def(
        "run_native",
        [](const vibeqc::semiempirical::SemiempiricalNativeRoute& route,
           const vibeqc::Molecule& molecule,
           const vibeqc::semiempirical::SemiempiricalParameters& parameters) {
            return run_with_identified_parameter_snapshot(
                parameters,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_native(
                        route, molecule, snapshot);
                });
        },
        py::arg("route"), py::arg("molecule"), py::arg("parameters"),
        "Run a molecular semiempirical energy through the unified native facade.");
    m_semi.def(
        "run_native",
        [](const vibeqc::semiempirical::SemiempiricalNativeRoute& route,
           const vibeqc::Molecule& molecule,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& parameters) {
            return run_with_identified_parameter_snapshot(
                parameters,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_native(
                        route, molecule, snapshot);
                });
        },
        py::arg("route"), py::arg("molecule"), py::arg("parameters"));
    m_semi.def(
        "run_native",
        [](const vibeqc::semiempirical::SemiempiricalNativeRoute& route,
           const vibeqc::Molecule& molecule,
           const vibeqc::semiempirical::nddo::OMxParameterSet& parameters) {
            return run_with_identified_parameter_snapshot(
                parameters,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_native(
                        route, molecule, snapshot);
                });
        },
        py::arg("route"), py::arg("molecule"), py::arg("parameters"));
    m_semi.def(
        "run_native",
        [](const vibeqc::semiempirical::SemiempiricalNativeRoute& route,
           const vibeqc::Molecule& molecule,
           const vibeqc::semiempirical::nddo::PM6ParameterSet& parameters) {
            return run_with_identified_parameter_snapshot(
                parameters,
                [&](const auto& snapshot) {
                    return vibeqc::semiempirical::run_native(
                        route, molecule, snapshot);
                });
        },
        py::arg("route"), py::arg("molecule"), py::arg("parameters"));
    m_semi.def(
        "run_native",
        py::overload_cast<
            const vibeqc::semiempirical::SemiempiricalNativeRoute&,
            const vibeqc::Molecule&,
            const vibeqc::semiempirical::indo::MsindoParameterSet&>(
            &vibeqc::semiempirical::run_native),
        py::arg("route"), py::arg("molecule"), py::arg("parameters"),
        py::call_guard<py::gil_scoped_release>());

    // --- Hamiltonian builders ---
    m_semi.def("build_gfn2_hamiltonian_zero",
               &vibeqc::semiempirical::build_gfn2_hamiltonian_zero,
               py::arg("basis"), py::arg("S"), py::arg("mol"), py::arg("params"),
               "Build GFN2-xTB H⁰ from overlap and parameters.");

    m_semi.def("build_gfn2_hamiltonian_zero_image",
               &vibeqc::semiempirical::build_gfn2_hamiltonian_zero_image,
               py::arg("basis"), py::arg("S"), py::arg("mol"), py::arg("params"),
               py::arg("image_shift"),
               "Build a GFN2-xTB H0 block for a translated periodic image.");

    m_semi.def("_gfn2_h0_periodic_coordination_numbers",
               &vibeqc::semiempirical::gfn2_h0_periodic_coordination_numbers,
               py::arg("system"),
               "Build pair-complete periodic GFN2 H0 coordination numbers.");

    m_semi.def("_libint_emultipole2_derivative_layout_probe",
               &vibeqc::semiempirical::xtb::libint_emultipole2_derivative_layout_probe,
               py::arg("h") = 1.0e-4,
               "Test seam: libint emultipole2 first-derivative buffer layout.");
    m_semi.def(
        "_periodic_gfn2_shell_gamma",
        [](const vibeqc::PeriodicSystem& system,
           const vibeqc::semiempirical::xtb::GFN2ParameterSet& params,
           double cutoff_bohr, vibeqc::semiempirical::ShellGammaForm form) {
            return vibeqc::semiempirical::xtb::assemble_periodic_gfn2(
                       system, params, cutoff_bohr, form).gamma_shell;
        },
        py::arg("system"), py::arg("params"), py::arg("cutoff_bohr"),
        py::arg("form"),
        "Test seam: the lattice-summed periodic GFN2 shell gamma.");

    m_semi.def("build_gfn2_gamma",
               &vibeqc::semiempirical::build_gfn2_gamma,
               py::arg("mol"), py::arg("params"),
               "Build GFN2-xTB gamma (Klopman-Ohno) matrix.");

    // ---- Translation-covariant lattice-summed shell gamma (core) ----------
    py::class_<vibeqc::semiempirical::ShellGammaSpec>(m_semi, "ShellGammaSpec")
        .def(py::init<>())
        .def_readwrite("form", &vibeqc::semiempirical::ShellGammaSpec::form)
        .def_readwrite("ko_average",
                       &vibeqc::semiempirical::ShellGammaSpec::ko_average);
    m_semi.def("elstner_short_range",
               &vibeqc::semiempirical::elstner_short_range,
               py::arg("tau_a"), py::arg("tau_b"), py::arg("R"),
               "Elstner 1998 short-range S(tau_a, tau_b, R).");
    m_semi.def("elstner_short_range_derivative",
               &vibeqc::semiempirical::elstner_short_range_derivative,
               py::arg("tau_a"), py::arg("tau_b"), py::arg("R"));
    m_semi.def("shell_gamma_onsite", &vibeqc::semiempirical::shell_gamma_onsite,
               py::arg("spec"), py::arg("U_a"), py::arg("U_b"));
    m_semi.def("shell_gamma_pair", &vibeqc::semiempirical::shell_gamma_pair,
               py::arg("spec"), py::arg("U_a"), py::arg("U_b"), py::arg("R"));
    m_semi.def("shell_gamma_pair_derivative",
               &vibeqc::semiempirical::shell_gamma_pair_derivative,
               py::arg("spec"), py::arg("U_a"), py::arg("U_b"), py::arg("R"));
    m_semi.def("shell_gamma_remainder",
               &vibeqc::semiempirical::shell_gamma_remainder,
               py::arg("spec"), py::arg("U_a"), py::arg("U_b"), py::arg("R"));
    m_semi.def("shell_gamma_remainder_derivative",
               &vibeqc::semiempirical::shell_gamma_remainder_derivative,
               py::arg("spec"), py::arg("U_a"), py::arg("U_b"), py::arg("R"));
    m_semi.def("shell_gamma_remainder_range",
               &vibeqc::semiempirical::shell_gamma_remainder_range,
               py::arg("spec"), py::arg("U_a"), py::arg("U_b"),
               py::arg("tolerance") = 1.0e-10);
    py::class_<vibeqc::semiempirical::EwaldCoulombKernel>(
        m_semi, "EwaldCoulombKernel",
        "Ewald-summed 1/R lattice potential (1-D wire, 2-D slab, 3-D).")
        .def(py::init([](const Eigen::Ref<const Eigen::MatrixXd>& translations,
                         double alpha) {
                 if (translations.cols() != 3 || translations.rows() < 1
                     || translations.rows() > 3) {
                     throw std::invalid_argument(
                         "translations must have shape (D, 3) with D in 1..3");
                 }
                 std::vector<Eigen::Vector3d> t;
                 for (Eigen::Index r = 0; r < translations.rows(); ++r) {
                     t.push_back(translations.row(r).transpose());
                 }
                 return vibeqc::semiempirical::EwaldCoulombKernel(t, alpha);
             }),
             py::arg("translations"), py::arg("alpha") = 0.0)
        .def_property_readonly(
            "dimension", &vibeqc::semiempirical::EwaldCoulombKernel::dimension)
        .def_property_readonly(
            "alpha", &vibeqc::semiempirical::EwaldCoulombKernel::alpha)
        .def("potential",
             [](const vibeqc::semiempirical::EwaldCoulombKernel& self,
                const Eigen::Vector3d& d, bool exclude_self) {
                 return self.potential(d, exclude_self);
             },
             py::arg("d"), py::arg("exclude_self") = false)
        .def("gradient",
             [](const vibeqc::semiempirical::EwaldCoulombKernel& self,
                const Eigen::Vector3d& d, bool exclude_self) {
                 return self.gradient(d, exclude_self);
             },
             py::arg("d"), py::arg("exclude_self") = false);
    {
        using vibeqc::semiempirical::GammaSite;
        using vibeqc::semiempirical::ImageRecord;
        using vibeqc::semiempirical::ImageRecordSink;
        const auto make_sites = [](const std::vector<int>& site_atom,
                                   const std::vector<double>& hardness) {
            if (site_atom.size() != hardness.size()) {
                throw std::invalid_argument(
                    "site_atom and hardness must have the same length");
            }
            std::vector<GammaSite> sites;
            for (std::size_t i = 0; i < site_atom.size(); ++i) {
                sites.push_back({site_atom[i], hardness[i]});
            }
            return sites;
        };
        const auto make_positions =
            [](const Eigen::Ref<const Eigen::MatrixXd>& positions) {
                if (positions.cols() != 3) {
                    throw std::invalid_argument(
                        "positions must have shape (n_atoms, 3)");
                }
                std::vector<Eigen::Vector3d> out;
                for (Eigen::Index r = 0; r < positions.rows(); ++r) {
                    out.push_back(positions.row(r).transpose());
                }
                return out;
            };
        const auto make_records = [](const std::vector<int>& rec_a,
                                     const std::vector<int>& rec_b,
                                     const Eigen::MatrixXd& shifts,
                                     const std::vector<double>& weights) {
            if (rec_a.size() != rec_b.size() || rec_a.size() != weights.size()
                || shifts.rows() != static_cast<Eigen::Index>(rec_a.size())
                || shifts.cols() != 3) {
                throw std::invalid_argument(
                    "record arrays must have matching lengths");
            }
            return vibeqc::semiempirical::ImageRecordSource(
                [rec_a, rec_b, shifts, weights](const ImageRecordSink& sink) {
                    for (std::size_t i = 0; i < rec_a.size(); ++i) {
                        ImageRecord record;
                        record.a = rec_a[i];
                        record.b = rec_b[i];
                        record.shift =
                            shifts.row(static_cast<Eigen::Index>(i)).transpose();
                        record.weight = weights[i];
                        sink(record);
                    }
                });
        };
        m_semi.def(
            "_molecular_shell_gamma",
            [make_sites, make_positions](
                const std::vector<int>& site_atom,
                const std::vector<double>& hardness,
                const Eigen::Ref<const Eigen::MatrixXd>& positions,
                const vibeqc::semiempirical::ShellGammaSpec& spec) {
                return vibeqc::semiempirical::build_molecular_shell_gamma(
                    make_sites(site_atom, hardness), make_positions(positions),
                    spec);
            },
            py::arg("site_atom"), py::arg("hardness"), py::arg("positions"),
            py::arg("spec"));
        m_semi.def(
            "_periodic_shell_gamma_from_records",
            [make_sites, make_positions, make_records](
                const std::vector<int>& site_atom,
                const std::vector<double>& hardness,
                const Eigen::Ref<const Eigen::MatrixXd>& positions,
                const vibeqc::semiempirical::ShellGammaSpec& spec,
                const vibeqc::semiempirical::EwaldCoulombKernel& ewald,
                const std::vector<int>& rec_a, const std::vector<int>& rec_b,
                const Eigen::MatrixXd& shifts,
                const std::vector<double>& weights) {
                return vibeqc::semiempirical::build_periodic_shell_gamma(
                    make_sites(site_atom, hardness), make_positions(positions),
                    spec, ewald, make_records(rec_a, rec_b, shifts, weights));
            },
            py::arg("site_atom"), py::arg("hardness"), py::arg("positions"),
            py::arg("spec"), py::arg("ewald"), py::arg("record_a"),
            py::arg("record_b"), py::arg("record_shift"),
            py::arg("record_weight"),
            "Test seam: record-driven lattice-summed shell gamma.");
        m_semi.def(
            "_periodic_shell_gamma_gradient_from_records",
            [make_sites, make_positions, make_records](
                const std::vector<int>& site_atom,
                const std::vector<double>& hardness,
                const Eigen::Ref<const Eigen::MatrixXd>& positions,
                const vibeqc::semiempirical::ShellGammaSpec& spec,
                const vibeqc::semiempirical::EwaldCoulombKernel& ewald,
                const std::vector<int>& rec_a, const std::vector<int>& rec_b,
                const Eigen::MatrixXd& shifts,
                const std::vector<double>& weights,
                const Eigen::VectorXd& dq) {
                return vibeqc::semiempirical::periodic_shell_gamma_gradient(
                    make_sites(site_atom, hardness), make_positions(positions),
                    spec, ewald, make_records(rec_a, rec_b, shifts, weights),
                    dq);
            },
            py::arg("site_atom"), py::arg("hardness"), py::arg("positions"),
            py::arg("spec"), py::arg("ewald"), py::arg("record_a"),
            py::arg("record_b"), py::arg("record_shift"),
            py::arg("record_weight"), py::arg("dq"));
        m_semi.def(
            "_periodic_shell_gamma_strain_from_records",
            [make_sites, make_positions, make_records](
                const std::vector<int>& site_atom,
                const std::vector<double>& hardness,
                const Eigen::Ref<const Eigen::MatrixXd>& positions,
                const vibeqc::semiempirical::ShellGammaSpec& spec,
                const vibeqc::semiempirical::EwaldCoulombKernel& ewald,
                const std::vector<int>& rec_a, const std::vector<int>& rec_b,
                const Eigen::MatrixXd& shifts,
                const std::vector<double>& weights,
                const Eigen::VectorXd& dq) {
                return Eigen::MatrixXd(
                    vibeqc::semiempirical::periodic_shell_gamma_strain_derivative(
                        make_sites(site_atom, hardness),
                        make_positions(positions), spec, ewald,
                        make_records(rec_a, rec_b, shifts, weights), dq));
            },
            py::arg("site_atom"), py::arg("hardness"), py::arg("positions"),
            py::arg("spec"), py::arg("ewald"), py::arg("record_a"),
            py::arg("record_b"), py::arg("record_shift"),
            py::arg("record_weight"), py::arg("dq"));
    }

    // Shell-resolved gamma (H¹ foundation)
    py::class_<vibeqc::semiempirical::GFN2ShellInfo>(m_semi, "GFN2ShellInfo")
        .def(py::init<>())
        .def_readonly("atom_idx", &vibeqc::semiempirical::GFN2ShellInfo::atom_idx)
        .def_readonly("Z", &vibeqc::semiempirical::GFN2ShellInfo::Z)
        .def_readonly("l", &vibeqc::semiempirical::GFN2ShellInfo::l)
        .def_readonly("bf_start", &vibeqc::semiempirical::GFN2ShellInfo::bf_start)
        .def_readonly("n_funcs", &vibeqc::semiempirical::GFN2ShellInfo::n_funcs)
        .def_readonly("k_en", &vibeqc::semiempirical::GFN2ShellInfo::k_en)
        .def_readonly("hardness", &vibeqc::semiempirical::GFN2ShellInfo::hardness)
        .def_readonly("kcn", &vibeqc::semiempirical::GFN2ShellInfo::kcn)
        .def_readonly("poly", &vibeqc::semiempirical::GFN2ShellInfo::poly);

    m_semi.def("gfn2_enumerate_shells",
               &vibeqc::semiempirical::gfn2_enumerate_shells,
               py::arg("basis"), py::arg("mol"), py::arg("params"),
               "Enumerate shells from a GFN2-xTB basis set.");

    m_semi.def("build_gfn2_shell_gamma",
               &vibeqc::semiempirical::build_gfn2_shell_gamma,
               py::arg("shells"), py::arg("mol"), py::arg("params"),
               "Build shell-resolved gamma matrix for H¹.");


// --- Repulsive spline ---
    py::class_<vibeqc::semiempirical::RepulsiveSpline>(m_semi, "RepulsiveSpline")
        .def(py::init<>())
        .def("build", &vibeqc::semiempirical::RepulsiveSpline::build)
        .def("evaluate", &vibeqc::semiempirical::RepulsiveSpline::evaluate)
        .def("derivative", &vibeqc::semiempirical::RepulsiveSpline::derivative)
        .def("n_knots", &vibeqc::semiempirical::RepulsiveSpline::n_knots)
        .def("cutoff", &vibeqc::semiempirical::RepulsiveSpline::cutoff)
        .def("empty", &vibeqc::semiempirical::RepulsiveSpline::empty);

// ---------------------------------------------------------------------------
// Geometry optimisation kernels
// ---------------------------------------------------------------------------
    m.def("lbfgs_two_loop", &vibeqc::geomopt::lbfgs_two_loop,
          py::arg("s_list"), py::arg("y_list"), py::arg("g"),
          py::arg("gamma0") = 1.0,
          "L-BFGS two-loop recursion: return (direction, success).");

    m.def("bfgs_inverse_update", &vibeqc::geomopt::bfgs_inverse_update,
          py::arg("H_inv"), py::arg("s"), py::arg("y"),
          "BFGS Sherman-Morrison inverse-Hessian update.");

    m.def("hessian_update_bfgs", &vibeqc::geomopt::hessian_update_bfgs,
          py::arg("H"), py::arg("s"), py::arg("y"),
          "BFGS Hessian (not inverse) update.");

    m.def("hessian_update_sr1", &vibeqc::geomopt::hessian_update_sr1,
          py::arg("H"), py::arg("s"), py::arg("y"),
          "SR1 Hessian update.");

    m.def("hessian_update_powell", &vibeqc::geomopt::hessian_update_powell,
          py::arg("H"), py::arg("s"), py::arg("y"),
          "Powell-symmetric-Broyden (PSB) Hessian update.");

    m.def("hessian_update_bofill", &vibeqc::geomopt::hessian_update_bofill,
          py::arg("H"), py::arg("s"), py::arg("y"),
          "Bofill weighted Hessian update.");

    m.def("hessian_update_ms", &vibeqc::geomopt::hessian_update_ms,
          py::arg("H"), py::arg("s"), py::arg("y"),
          "Murtagh-Sargent symmetric rank-2 Hessian update.");

    m.def("trust_region_step", &vibeqc::geomopt::trust_region_step,
          py::arg("g"), py::arg("B"), py::arg("delta"),
          py::arg("max_iter") = 30, py::arg("tol") = 1e-8,
          "More-Sorensen trust-region subproblem solver. "
          "Returns (step, level_shift, interior_flag).");

    m.def("max_abs_component", &vibeqc::geomopt::max_abs_component,
          py::arg("v"), "Maximum absolute component of a vector.");

    m.def("rms", &vibeqc::geomopt::rms,
          py::arg("v"), "Root-mean-square of a vector.");

    m.def("max_atom_displacement", &vibeqc::geomopt::max_atom_displacement,
          py::arg("x_old"), py::arg("x_new"),
          "Maximum atomic displacement between two geometries.");

    m.def("rms_displacement", &vibeqc::geomopt::rms_displacement,
          py::arg("x_old"), py::arg("x_new"),
          "RMS atomic displacement between two geometries.");

}
