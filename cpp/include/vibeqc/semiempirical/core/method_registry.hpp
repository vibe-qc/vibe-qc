// Shared semiempirical core — method registry and interfaces.
//
// Phase 0: Provides the SemiempiricalMethodConfig struct for
// method metadata and a plan for extracting shared components
// from the DFTB implementation (ADR-001).
//
// Executable methods (DFTB, GFN-xTB, PM6, ...) register themselves here
// and provide Hamiltonian builder, gamma function, repulsive
// potential, and parameter loading as plugin functions.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace vibeqc {

// Forward declarations
class Molecule;
class BasisSet;
struct PeriodicSystem;
struct LatticeSumOptions;

namespace semiempirical {

// Forward declarations
class SemiempiricalParameters;

// ---------------------------------------------------------------------------
// Method metadata
// ---------------------------------------------------------------------------

enum class MethodFamily {
    DFTB,   // Density-Functional Tight-Binding
    XTB,    // GFN-xTB family
    NDDO,   // MNDO/AM1/PMx/OMx family
    INDO,   // MSINDO (Bredow/Geudtner/Jug)
    ML,     // ML-corrected semiempirical
    LEGACY, // AM1, MNDO, PM3 (regression baselines only)
};

enum class PeriodicTier {
    Native = 1,      // Production periodic support
    Generalized = 2, // Validated periodic with caveats
    Experimental = 3 // Behind feature flag, not validated
};

enum class GradientQuality {
    Exact,            // Matches FD to machine precision
    Approximate,      // Fixed-charge or other approximation
    FiniteDifference, // No analytic gradient
};

struct SemiempiricalMethodConfig {
    std::string name;                     // "dftb0", "scc-dftb", "gfn2-xtb", "pm6"
    std::string display_name;             // "DFTB0", "SCC-DFTB", "GFN2-xTB"
    MethodFamily family;
    std::string description;

    // Capabilities
    bool supports_open_shell = false;
    bool supports_periodic = false;
    PeriodicTier periodic_tier = PeriodicTier::Experimental;
    GradientQuality gradient_quality = GradientQuality::FiniteDifference;
    bool supports_stress = false;
    bool supports_kpoints = false;
    bool has_dispersion = false;

    // Element support (empty = all elements in parameter set)
    std::vector<int> supported_elements;

    // Parameter version for reproducibility
    std::string parameter_version;
};

// ---------------------------------------------------------------------------
// Hamiltonian builder plugin interface
// ---------------------------------------------------------------------------

using HamiltonianBuilder = std::function<Eigen::MatrixXd(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const SemiempiricalParameters& params)>;

// ---------------------------------------------------------------------------
// Gamma matrix builder plugin interface
// ---------------------------------------------------------------------------

using GammaBuilder = std::function<Eigen::MatrixXd(
    const Molecule& mol,
    const SemiempiricalParameters& params)>;

// ---------------------------------------------------------------------------
// SCC Hamiltonian correction plugin interface
// ---------------------------------------------------------------------------

using SCCHamiltonianCorrection = std::function<Eigen::MatrixXd(
    const Eigen::MatrixXd& H0,
    const Eigen::MatrixXd& S,
    const BasisSet& basis,
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const Eigen::MatrixXd& gamma,
    const Eigen::VectorXd& delta_q)>;

// ---------------------------------------------------------------------------
// Repulsive pair potential plugin interface
// ---------------------------------------------------------------------------

using RepulsiveEnergyFunc = std::function<double(
    int Z1, int Z2, double R, const SemiempiricalParameters& params)>;

using RepulsiveDerivativeFunc = std::function<double(
    int Z1, int Z2, double R, const SemiempiricalParameters& params)>;

// ---------------------------------------------------------------------------
// Periodic gamma matrix builder (lattice-summed)
// ---------------------------------------------------------------------------

using PeriodicGammaBuilder = std::function<Eigen::MatrixXd(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    double cutoff_bohr)>;

// ---------------------------------------------------------------------------
// Method plugin bundle
// ---------------------------------------------------------------------------

struct SemiempiricalMethodPlugin {
    SemiempiricalMethodConfig config;

    // Required: build the zero-order Hamiltonian
    HamiltonianBuilder build_hamiltonian_zero;

    // Required: build the gamma (electrostatic interaction) matrix
    GammaBuilder build_gamma;

    // Optional: SCC Hamiltonian correction (defaults to DFTB-style)
    SCCHamiltonianCorrection build_scc_correction = nullptr;

    // Required: repulsive pair potential
    RepulsiveEnergyFunc repulsive_energy;
    RepulsiveDerivativeFunc repulsive_derivative;

    // Optional: periodic gamma builder (defaults to Ohno-Klopman sum)
    PeriodicGammaBuilder build_periodic_gamma = nullptr;
};

// ---------------------------------------------------------------------------
// Method registry
// ---------------------------------------------------------------------------

class SemiempiricalMethodRegistry {
public:
    static SemiempiricalMethodRegistry& instance();

    // Register a method plugin
    void register_method(const SemiempiricalMethodPlugin& plugin);

    // Look up a method by name
    const SemiempiricalMethodPlugin* find(const std::string& name) const;

    // List all registered methods
    std::vector<std::string> method_names() const;

    // Get config for all methods in a family
    std::vector<SemiempiricalMethodConfig> family_methods(MethodFamily family) const;

private:
    SemiempiricalMethodRegistry() = default;
    std::vector<SemiempiricalMethodPlugin> methods_;
};

}  // namespace semiempirical
}  // namespace vibeqc
