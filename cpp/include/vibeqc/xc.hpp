// Exchange-correlation functional wrapper over libxc.
//
// A `Functional` is a (possibly composite) XC functional — e.g. "LDA" =
// Slater exchange + VWN5 correlation, "B3LYP" = a single built-in hybrid,
// "BLYP" = B88 exchange + LYP correlation. Internally we hold one or more
// xc_func_type instances from libxc; all their energy densities and
// potentials are summed.
//
// On evaluation, given a grid of n_points with density ρ(r) (and ∇ρ(r) for
// GGAs, plus the kinetic-energy density τ(r) for meta-GGAs), libxc returns:
//   exc(r)         — energy density per unit volume (Hartree / bohr^3 per electron)
//   v_ρ(r)         — ∂f/∂ρ
//   v_σ(r)         — ∂f/∂σ   (σ = |∇ρ|², GGAs and meta-GGAs)
//   v_τ(r)         — ∂f/∂τ   (meta-GGAs only)
//
// The MGGA path uses ``eval_unpolarised_mgga`` / ``eval_polarised_mgga``;
// the LDA/GGA path keeps the leaner ``eval_unpolarised`` / ``eval_polarised``
// signatures unchanged. Calling the non-mgga overload on an MGGA functional
// raises with a roadmap pointer to the right method. Functionals that
// additionally depend on ∇²ρ (the laplacian) are rejected at construction
// — vibe-qc's MGGA grid populates τ but not ∇²ρ.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vibeqc {

enum class XCKind {
    LDA,    // depends only on ρ
    GGA,    // depends on ρ and ∇ρ
    MGGA,   // depends on ρ, ∇ρ, and τ (kinetic-energy density)
};

// Versioned capability record for full-grid external XC providers. Version 1
// intentionally starts with the smallest cross-route requirement: a provider
// may require one complete atomic-grid construction protocol. Empty means that
// any Grid profile is accepted.
inline constexpr int kExternalXCCapabilityVersion = 1;
struct ExternalXCCapabilities {
    int version = kExternalXCCapabilityVersion;
    std::string required_grid_profile;
};

// Full-grid, spin-resolved feature contract for nonlocal XC providers.
// Unlike libxc's semilocal interface, an external provider sees the entire
// atom-major quadrature at once.  This is required when the energy at one
// point depends on descriptors integrated over other points (SKALA is the
// first consumer).  All numerical values use atomic units.
struct ExternalXCInput {
    std::string functional;
    std::string grid_profile;              // actual Grid construction profile
    Eigen::MatrixX3d points;             // (G,3), bohr
    Eigen::VectorXd grid_weights;         // (G,), final energy quadrature
    Eigen::VectorXd atomic_grid_weights;  // (G,), unpartitioned atom grid
    std::vector<int> atom_of_point;       // (G,), contiguous atom-major blocks
    Eigen::MatrixX3d atom_coords;         // (A,3), bohr
    std::vector<int> atomic_numbers;      // (A,)
    bool periodic = false;
    int periodic_dimension = 0;
    Eigen::Matrix3d lattice = Eigen::Matrix3d::Zero(); // columns, bohr

    Eigen::VectorXd rho_alpha, rho_beta;  // (G,)
    Eigen::MatrixX3d grad_alpha, grad_beta; // (G,3)
    Eigen::VectorXd tau_alpha, tau_beta;  // (G,)
};

// First derivatives of the *scalar, already quadrature-integrated* XC
// energy.  Consequently these are energy adjoints, not libxc-style local
// potentials, and callers must not multiply by grid weights a second time.
// Cartesian gradient adjoints make the interface general for nonlocal
// functionals whose chain rule cannot be represented by v_sigma alone.
struct ExternalXCEvaluation {
    double energy = 0.0;
    Eigen::VectorXd v_rho_alpha, v_rho_beta;
    Eigen::MatrixX3d v_grad_alpha, v_grad_beta;
    Eigen::VectorXd v_tau_alpha, v_tau_beta;
};

class ExternalXCProvider {
public:
    virtual ~ExternalXCProvider() = default;
    virtual ExternalXCEvaluation evaluate(const ExternalXCInput& input) = 0;
};

// Read-only snapshot used by language bindings which need to adopt a
// process-lifetime provider after their module-local state was re-created.
// Registration remains insert-only: this surface cannot replace or remove a
// provider, so duplicate-name collision safety is unchanged.
struct ExternalXCRegistration {
    std::shared_ptr<ExternalXCProvider> provider;
    double hf_exchange_fraction = 0.0;
    ExternalXCCapabilities capabilities;
};

// Phase 17e — spin-polarised GGA second derivatives of the XC energy
// density (the "fxc kernel"), used by the open-shell Kohn-Sham
// orbital-Hessian matvec (UKS Newton / TRAH) and by CPKS linear
// response. libxc's ``xc_gga_fxc`` returns 15 distinct pieces for
// spin = 2; bundling them in one struct keeps the call site readable
// (the alternative is a 15-out-parameter signature).
//
// Component naming follows libxc's derivative-index convention:
//   v2rho2_st       = ∂²f / ∂ρ_s ∂ρ_t            (s,t ∈ {a,b})
//   v2rhosigma_s_uv = ∂²f / ∂ρ_s ∂σ_uv           (uv ∈ {aa,ab,bb})
//   v2sigma2_uv_xy  = ∂²f / ∂σ_uv ∂σ_xy
// where σ_aa = ∇ρ_a·∇ρ_a, σ_ab = ∇ρ_a·∇ρ_b, σ_bb = ∇ρ_b·∇ρ_b.
// Only the symmetric-unique pieces are stored (3 + 6 + 6 = 15);
// the transposes (v2rho2_ba, v2sigma2_ab_aa, …) equal their stored
// counterparts. Each vector has length n_points.
struct PolarisedGGAFxc {
    // ∂²f / ∂ρ_s ∂ρ_t — 3 unique pieces.
    Eigen::VectorXd v2rho2_aa, v2rho2_ab, v2rho2_bb;
    // ∂²f / ∂ρ_s ∂σ_uv — 6 pieces (ρ ≠ σ, so the full 2×3 block).
    Eigen::VectorXd v2rhosigma_a_aa, v2rhosigma_a_ab, v2rhosigma_a_bb;
    Eigen::VectorXd v2rhosigma_b_aa, v2rhosigma_b_ab, v2rhosigma_b_bb;
    // ∂²f / ∂σ_uv ∂σ_xy — 6 unique pieces (upper triangle of the
    // symmetric 3×3 over {aa,ab,bb}).
    Eigen::VectorXd v2sigma2_aa_aa, v2sigma2_aa_ab, v2sigma2_aa_bb;
    Eigen::VectorXd v2sigma2_ab_ab, v2sigma2_ab_bb, v2sigma2_bb_bb;
};

class Functional {
public:
    // Accepts common aliases ("LDA", "SVWN", "PBE", "BLYP", "B3LYP"),
    // or an explicit comma-separated list of libxc XC_… integer IDs.
    // spin = 1 (unpolarised) for now — UKS path will accept spin = 2.
    explicit Functional(const std::string& name, int spin = 1);
    ~Functional();

    Functional(const Functional&) = delete;
    Functional& operator=(const Functional&) = delete;

    const std::string& name() const { return name_; }
    XCKind kind() const { return kind_; }
    bool is_hybrid() const {
        return hf_exchange_fraction_ != 0.0 || cam_beta_ != 0.0;
    }
    double hf_exchange_fraction() const { return hf_exchange_fraction_; }

    // Range-separated (CAM / RSH) hybrid parameters. A range-separated
    // hybrid — ωB97X, ωB97X-D, ωB97X-V, CAM-B3LYP, HSE06, … — makes the
    // exact-exchange admixture position-dependent:
    //
    //   EXX(r₁₂) = cam_alpha + cam_beta · erf(rsh_omega · r₁₂)
    //
    // i.e. the exchange operator is
    //
    //   cam_alpha · (1/r₁₂)  +  cam_beta · (erf(ω·r₁₂)/r₁₂).
    //
    // ``cam_alpha`` is the full-range (always-on) HF fraction;
    // ``cam_beta`` switches on the *additional* long-range HF via the
    // error function (β > 0 long-range-corrected, β < 0 screened like
    // HSE). For a long-range-corrected functional (ωB97X family)
    // cam_alpha + cam_beta = 1 (100 % HF at long range).
    //
    // For a *global* hybrid (B3LYP, PBE0, …) is_range_separated() is
    // false, cam_alpha() == hf_exchange_fraction(), cam_beta() == 0. For
    // a pure functional all three are zero. SCF drivers that build the
    // exchange matrix branch on is_range_separated(): when true they
    // additionally build the erf-attenuated exchange K_erf(ω) and
    // assemble  −½(cam_alpha·K + cam_beta·K_erf); when false the single
    // −½·cam_alpha·K term reproduces the global-hybrid path exactly.
    bool is_range_separated() const { return is_range_separated_; }
    double rsh_omega() const { return rsh_omega_; }
    double cam_alpha() const { return cam_alpha_; }
    double cam_beta() const { return cam_beta_; }

    // Double-hybrid MP2-correction coefficients (Grimme 2006 family).
    // Non-zero for known double hybrids (B2PLYP: c_os = c_ss = 0.27;
    // DSD-PBEP86: c_os ≠ c_ss). For non-double-hybrid functionals
    // (LDA / GGA / hybrid-GGA) both are zero. The intended use is:
    //   if (func.is_double_hybrid()) {
    //       // run RKS with the SCF piece (the libxc components +
    //       // ``hf_exchange_fraction()`` HF mixing), then run RI-MP2
    //       // with options.c_os = mp2_c_os(), options.c_ss = mp2_c_ss()
    //       // on the converged KS orbitals; the total energy is
    //       //   E_DH = E_rks + c_os · E_os + c_ss · E_ss.
    //   }
    // The Python wrapper ``vibeqc.run_b2plyp`` orchestrates this for
    // B2PLYP; the dispatcher pattern is intended to generalise to
    // DSD-PBEP86, PWPB95, ωB97M(2), etc. once each is registered with
    // the appropriate ``mp2_c_os`` / ``mp2_c_ss`` pair.
    bool is_double_hybrid() const {
        return mp2_c_os_ != 0.0 || mp2_c_ss_ != 0.0;
    }
    double mp2_c_os() const { return mp2_c_os_; }
    double mp2_c_ss() const { return mp2_c_ss_; }

    // VV10 nonlocal correlation (Vydrov-Van Voorhis 2010). libxc supplies
    // the *semilocal* part of a VV10-paired functional (ωB97X-V, ωB97M-V,
    // the standalone VV10, …) and the two empirical parameters (b, C) via
    // ``xc_nlc_coef``, but it does NOT evaluate the nonlocal double
    // integral — that is the host's job (see cpp/src/vv10.cpp,
    // ``compute_vv10``). ``needs_vv10()`` is true iff any libxc component
    // carries the ``XC_FLAGS_VV10`` flag; the RKS / UKS V_xc builders then
    // add the nonlocal energy + self-consistent potential on the same grid
    // alongside the semilocal pieces. ``vv10_b()`` / ``vv10_C()`` are the
    // parameters libxc reports (e.g. b = 6.0, C = 0.01 for ωB97X-V /
    // ωB97M-V; b = 5.9, C = 0.0093 for the original VV10). Zero / false for
    // a functional with no nonlocal component. (All VV10-paired aliases use
    // a single weight-1.0 component, so the nonlocal term enters at unit
    // weight.)
    bool needs_vv10() const { return needs_vv10_; }
    double vv10_b() const { return vv10_b_; }
    double vv10_C() const { return vv10_C_; }

    // True for a full-grid provider registered with
    // ``register_external_functional`` rather than a libxc composition.
    bool is_external() const;
    // Query the immutable capability record captured when an external
    // provider was registered. Throws for a libxc-backed functional.
    const ExternalXCCapabilities& external_capabilities() const;
    ExternalXCEvaluation eval_external(const ExternalXCInput& input) const;

    // Unpolarised evaluation. For LDA, only rho is used (sigma may be an
    // empty vector). For GGA, sigma must be the pointwise |∇ρ|².
    //
    // On output:
    //   exc     (n_points,)          — ρ·ε_xc (integrand form: E_xc = Σ w·exc)
    //   v_rho   (n_points,)          — ∂f/∂ρ
    //   v_sigma (n_points,)          — ∂f/∂σ (only filled for GGA; resized to 0 for LDA)
    void eval_unpolarised(const Eigen::VectorXd& rho,
                          const Eigen::VectorXd& sigma,
                          Eigen::VectorXd& exc,
                          Eigen::VectorXd& v_rho,
                          Eigen::VectorXd& v_sigma) const;

    // Polarized evaluation for UKS. Requires the functional to have been
    // constructed with spin=2.
    //
    // Inputs  (all size n_points):
    //   rho_a, rho_b
    //   sigma_aa = |∇ρ_α|²   sigma_ab = ∇ρ_α·∇ρ_β   sigma_bb = |∇ρ_β|²
    //   (sigma_* may be empty for LDA)
    //
    // Outputs (all size n_points):
    //   exc            — ρ_total · ε_xc (integrand form)
    //   v_rho_a, v_rho_b
    //   v_sigma_aa, v_sigma_ab, v_sigma_bb (empty for LDA)
    void eval_polarised(const Eigen::VectorXd& rho_a,
                        const Eigen::VectorXd& rho_b,
                        const Eigen::VectorXd& sigma_aa,
                        const Eigen::VectorXd& sigma_ab,
                        const Eigen::VectorXd& sigma_bb,
                        Eigen::VectorXd& exc,
                        Eigen::VectorXd& v_rho_a,
                        Eigen::VectorXd& v_rho_b,
                        Eigen::VectorXd& v_sigma_aa,
                        Eigen::VectorXd& v_sigma_ab,
                        Eigen::VectorXd& v_sigma_bb) const;

    // Phase 17d — XC kernel (second derivative) for unpolarised LDA / GGA.
    // Used by analytic Kohn-Sham Hessian / CPKS (linear-response).
    //
    // LDA: ``v2rho2(g) = ∂²f_xc(ρ)/∂ρ² (ρ(g))``. ``v2rhosigma`` and
    // ``v2sigma2`` are resized to 0.
    //
    // GGA: ``v2rho2(g) = ∂²f/∂ρ²``,
    //      ``v2rhosigma(g) = ∂²f/∂ρ∂σ``,
    //      ``v2sigma2(g) = ∂²f/∂σ²``,
    //      all evaluated at (ρ(g), σ(g)).
    void eval_unpolarised_fxc(const Eigen::VectorXd& rho,
                                const Eigen::VectorXd& sigma,
                                Eigen::VectorXd& v2rho2,
                                Eigen::VectorXd& v2rhosigma,
                                Eigen::VectorXd& v2sigma2) const;

    // Polarized LDA XC kernel. For LDA polarized:
    //   v2rho2(g) is a (3,)-tuple per grid point in libxc's order
    //   (αα, αβ, ββ). We return three vectors of length n_points.
    void eval_polarised_lda_fxc(const Eigen::VectorXd& rho_a,
                                  const Eigen::VectorXd& rho_b,
                                  Eigen::VectorXd& v2rho2_aa,
                                  Eigen::VectorXd& v2rho2_ab,
                                  Eigen::VectorXd& v2rho2_bb) const;

    // Phase 17e — polarised GGA XC kernel (spin-polarised second
    // derivatives). Wraps libxc's ``xc_gga_fxc`` for spin = 2; fills
    // all 15 pieces of ``out`` (see PolarisedGGAFxc). LDA components
    // inside a composite functional contribute only to v2rho2_*; their
    // v2rhosigma / v2sigma2 entries stay zero. Accepts pure-LDA
    // functionals too (every σ piece returns zero) so callers can take
    // a single uniform path. Raises on meta-GGA — τ-dependent second
    // derivatives are a later phase.
    //
    // σ inputs follow the static-V_xc convention: σ_aa = |∇ρ_a|²,
    // σ_ab = ∇ρ_a·∇ρ_b, σ_bb = |∇ρ_b|² (libxc order, no factor 2 on
    // the cross term).
    void eval_polarised_gga_fxc(const Eigen::VectorXd& rho_a,
                                  const Eigen::VectorXd& rho_b,
                                  const Eigen::VectorXd& sigma_aa,
                                  const Eigen::VectorXd& sigma_ab,
                                  const Eigen::VectorXd& sigma_bb,
                                  PolarisedGGAFxc& out) const;

    // ----- Meta-GGA evaluation (τ-dependent, used by TPSS/TPSSh,
    //       M06-L/M06-2X, etc.) ------------------------------------------
    //
    // Inputs (all size n_points, spin = 1):
    //   rho      — density ρ(r)
    //   sigma    — |∇ρ|²
    //   tau      — kinetic-energy density τ(r) = ½ Σ_i |∇ψ_i(r)|²
    //              (the total, not per-spin; libxc unpolarised convention)
    //
    // Outputs (all size n_points):
    //   exc      — ρ·ε_xc (integrand form: E_xc = Σ w·exc)
    //   v_rho    — ∂f/∂ρ
    //   v_sigma  — ∂f/∂σ
    //   v_tau    — ∂f/∂τ
    //
    // LDA/GGA components inside a composite are evaluated via their own
    // libxc calls (they contribute zero to v_tau). The Functional must
    // have ``kind() == MGGA`` (or LDA/GGA with v_tau returned as zero —
    // permitted as a convenience for callers that uniformly take the MGGA
    // path). Throws if the functional needs ∇²ρ.
    void eval_unpolarised_mgga(const Eigen::VectorXd& rho,
                                const Eigen::VectorXd& sigma,
                                const Eigen::VectorXd& tau,
                                Eigen::VectorXd& exc,
                                Eigen::VectorXd& v_rho,
                                Eigen::VectorXd& v_sigma,
                                Eigen::VectorXd& v_tau) const;

    // Spin-polarized meta-GGA. tau_a, tau_b are per-spin (each = ½ Σ_iσ
    // |∇ψ_iσ|²); libxc's polarized MGGA convention takes them separately.
    void eval_polarised_mgga(const Eigen::VectorXd& rho_a,
                              const Eigen::VectorXd& rho_b,
                              const Eigen::VectorXd& sigma_aa,
                              const Eigen::VectorXd& sigma_ab,
                              const Eigen::VectorXd& sigma_bb,
                              const Eigen::VectorXd& tau_a,
                              const Eigen::VectorXd& tau_b,
                              Eigen::VectorXd& exc,
                              Eigen::VectorXd& v_rho_a,
                              Eigen::VectorXd& v_rho_b,
                              Eigen::VectorXd& v_sigma_aa,
                              Eigen::VectorXd& v_sigma_ab,
                              Eigen::VectorXd& v_sigma_bb,
                              Eigen::VectorXd& v_tau_a,
                              Eigen::VectorXd& v_tau_b) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string name_;
    XCKind kind_ = XCKind::LDA;
    double hf_exchange_fraction_ = 0.0;
    double mp2_c_os_ = 0.0;
    double mp2_c_ss_ = 0.0;
    bool is_range_separated_ = false;
    double rsh_omega_ = 0.0;
    double cam_alpha_ = 0.0;
    double cam_beta_ = 0.0;
    bool needs_vv10_ = false;
    double vv10_b_ = 0.0;
    double vv10_C_ = 0.0;
};

/// Check whether a functional alias has been registered dynamically
/// (via register_functional_alias).
bool dynamic_alias_available(const std::string& name);

/// Register a user-defined functional alias at runtime.
///
/// components is a list of (libxc_name, weight) pairs, e.g.
/// ("GGA_X_PW91", 0.80).  hf_exchange_fraction is the global
/// HF-exchange fraction (e.g. 0.20 for PW1PW).  After registration
/// Functional(name) resolves the alias exactly like a built-in.
///
/// Thread-safe; intended to be called once at startup from Python.
void register_functional_alias(
    const std::string& name,
    const std::vector<std::pair<std::string, double>>& components,
    double hf_exchange_fraction);

/// Register a full-grid nonlocal XC backend under a functional name.
/// The registry owns the provider, so Functional objects remain valid for
/// the duration of the process.  Thread-safe; registration is normally done
/// once by a lazy Python adapter at package import time.
void register_external_functional(
    const std::string& name,
    std::shared_ptr<ExternalXCProvider> provider,
    double hf_exchange_fraction = 0.0,
    ExternalXCCapabilities capabilities = {});

/// Atomically register one full-grid nonlocal XC backend under several names.
///
/// Every name is validated before the process-lifetime registry changes. If
/// any name is blank, duplicated, or already resolves, no member of the
/// family is registered. All names retain the same provider instance and
/// capability record.
void register_external_functional_family(
    const std::vector<std::string>& names,
    std::shared_ptr<ExternalXCProvider> provider,
    double hf_exchange_fraction = 0.0,
    ExternalXCCapabilities capabilities = {});

/// Return a read-only snapshot of an existing full-grid provider registration,
/// or std::nullopt when ``name`` is not in the external registry. This is an
/// adoption/query surface only; callers cannot mutate the process-lifetime
/// registry through the returned value.
std::optional<ExternalXCRegistration>
find_external_functional_registration(const std::string& name);

}  // namespace vibeqc
