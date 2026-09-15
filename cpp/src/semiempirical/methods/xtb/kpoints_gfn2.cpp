#include "vibeqc/semiempirical/methods/xtb/kpoints_gfn2.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "vibeqc/semiempirical/kpoints_occupations.hpp"
#include "vibeqc/semiempirical/methods/xtb/periodic_gfn2.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {
namespace {

bool is_single_gamma_mesh(const BlochKMesh& kmesh) {
    if (kmesh.kpoints.size() != 1) {
        return false;
    }
    const Eigen::Vector3d& kpoint = kmesh.kpoints.front();
    if (!kpoint.allFinite() || !kpoint.isZero(0.0)) {
        return false;
    }
    return kmesh.weights.empty()
        || (kmesh.weights.size() == 1
            && std::isfinite(kmesh.weights.front())
            && kmesh.weights.front() == 1.0);
}

KPointGFN2Result kpoint_result_from_gamma(PeriodicGFN2Result gamma) {
    KPointGFN2Result result;
    result.energy = gamma.energy;
    result.free_energy = gamma.free_energy;
    result.e_electronic = gamma.e_electronic;
    result.e_repulsive = gamma.e_repulsive;
    result.e_scc = gamma.e_scc;
    result.e_band0 = gamma.e_band0;
    result.e_aes = gamma.e_aes;
    result.e_3rd = gamma.e_3rd;
    result.fermi_level = gamma.fermi_level;
    result.entropy = gamma.entropy;
    result.smearing_temperature = gamma.smearing_temperature;
    result.band_energies.reserve(
        static_cast<std::size_t>(gamma.mo_energies.size()));
    for (Eigen::Index i = 0; i < gamma.mo_energies.size(); ++i) {
        result.band_energies.push_back(gamma.mo_energies(i));
    }
    result.eps_per_k.push_back(std::move(gamma.mo_energies));
    // Measured band edges + gaps from the occupations actually used
    // (issue #426; convention documented on KPointBandEdges). The Gamma
    // driver stores its converged occupation vector only on success, so an
    // unconverged diagnostics record keeps empty occupations_per_k and NaN
    // edge fields: nothing was measured.
    if (gamma.occupations.size() > 0 && !result.eps_per_k.empty() &&
        gamma.occupations.size() == result.eps_per_k.front().size()) {
        result.occupations_per_k.push_back(std::move(gamma.occupations));
        store_kpoint_band_edges(
            result,
            compute_kpoint_band_edges(
                result.eps_per_k, result.occupations_per_k));
    }
    result.density = std::move(gamma.density);
    result.overlap_gamma = std::move(gamma.overlap_gamma);
    result.charges = std::move(gamma.charges);
    result.dq_shell = std::move(gamma.dq_shell);
    result.n_basis = gamma.n_basis;
    result.n_occ = gamma.n_occ;
    result.n_kpoints = 1;
    result.n_shells = gamma.n_shells;
    result.n_iter = gamma.n_iter;
    result.converged = gamma.converged;
    return result;
}

[[noreturn]] void throw_gfn2_bandpath_disabled() {
    throw std::runtime_error(
        "run_gfn2_xtb_bandpath: non-Gamma GFN2 band paths are disabled "
        "(issue #351) because the current implementation lacks full complex "
        "Bloch phases, phase-resolved bare-band energy, and AES parity");
}

}  // namespace

KPointGFN2Result run_gfn2_xtb_kpoints(
    const PeriodicSystem& system, const GFN2ParameterSet& params,
    const BlochKMesh& kmesh, const XTBSccOptions& sopts,
    double cutoff_bohr) {
    if (system.multiplicity != 1) {
        throw std::invalid_argument("run_gfn2_xtb_kpoints: closed-shell only");
    }
    validate_periodic_gfn2_physics_domain(system);

    // Issue #351: the former non-Gamma path used only cos(k.g), contracted
    // the weighted density against H0(Gamma), and omitted AES. Preserve the
    // public one-point Gamma spelling by delegating to the supported Gamma
    // driver, but do not expose a plausible wrong answer for any other mesh.
    if (is_single_gamma_mesh(kmesh)) {
        return kpoint_result_from_gamma(
            run_gfn2_xtb_gamma(system, params, sopts, cutoff_bohr));
    }
    throw std::runtime_error(
        "run_gfn2_xtb_kpoints: non-Gamma GFN2 is disabled (issue #351) "
        "because the current implementation lacks full complex Bloch phases, "
        "phase-resolved bare-band energy, and AES parity; use "
        "run_gfn2_xtb_gamma for Gamma-only calculations");
}

KPointGFN2Result run_gfn2_xtb_bandpath(
    const PeriodicSystem& system, const GFN2ParameterSet&,
    const std::vector<Eigen::Vector3d>& kpath,
    const BlochKMesh& reference_mesh,
    const XTBSccOptions&,
    double) {
    if (system.multiplicity != 1) {
        throw std::invalid_argument("run_gfn2_xtb_bandpath: closed-shell only");
    }
    if (kpath.empty()) {
        throw std::invalid_argument(
            "run_gfn2_xtb_bandpath: the k-path is empty");
    }
    if (reference_mesh.size() == 0) {
        throw std::invalid_argument(
            "run_gfn2_xtb_bandpath: the reference k-mesh is empty");
    }

    // Useful band paths contain non-Gamma points. The previous real-only
    // diagonalization dropped their imaginary phase and omitted AES.
    throw_gfn2_bandpath_disabled();
}

KPointGFN2Result run_gfn2_xtb_bandpath(
    const PeriodicSystem&, const GFN2ParameterSet&,
    const std::vector<Eigen::Vector3d>&, double) {
    throw_gfn2_bandpath_disabled();
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
