#include "vibeqc/xc.hpp"

#include <xc.h>
#include <xc_funcs.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vibeqc {

struct Functional::Impl {
    std::vector<xc_func_type> funcs;
    // Per-component multiplicative weight applied to exc / v_rho / v_sigma /
    // fxc when accumulating libxc outputs. Default 1.0; non-unit values come
    // from custom-mix aliases like PW1PW (0.80·PW91-X + 1.00·PW91-C).
    std::vector<double> weights;
    // Number of entries in `funcs` that have been successfully initialized
    // via xc_func_init. Only those may be handed to xc_func_end — calling
    // xc_func_end on an uninitialized (zero-filled) xc_func_type segfaults.
    // This matters when the Functional ctor throws partway through the
    // init loop (invalid id, or unsupported family for a later id).
    std::size_t n_init = 0;
    int spin;
    std::shared_ptr<ExternalXCProvider> external_provider;
    ExternalXCCapabilities external_capabilities;

    ~Impl() {
        for (std::size_t i = 0; i < n_init; ++i) xc_func_end(&funcs[i]);
    }
};

namespace {

std::string to_lower_copy(const std::string& s) {
    std::string r(s);
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return r;
}

// One libxc component plus the multiplicative weight under which its
// energy density and potentials contribute to the composite functional.
struct AliasComponent {
    int libxc_id;
    double weight;
    // Optional libxc external-parameter overrides, applied right after
    // xc_func_init via xc_func_set_ext_params_name. Empty for stock
    // functionals; non-empty only for *reparametrised* components —
    // currently just PWPB95's PW6-modified mPW91 exchange and its
    // re-tuned B95 correlation (Goerigk-Grimme 2011). This stays
    // aggregate-init compatible: the existing two-field component
    // literals (``{XC_GGA_X_PBE, 0.31}``) leave it default-empty.
    std::vector<std::pair<std::string, double>> ext_params {};
};

// Full alias spec: weighted component list, plus an extra HF-exchange
// fraction that is *not* implied by libxc (used for hand-mixed hybrids
// like PW1PW whose underlying components are pure GGAs and therefore
// report ``xc_hyb_exx_coef = 0``), plus optional double-hybrid MP2-
// correction coefficients (non-zero only for B2PLYP / DSD-PBEP86 /
// PWPB95 / … — see ``Functional::is_double_hybrid``).
struct AliasSpec {
    std::vector<AliasComponent> components;
    double extra_hf_fraction = 0.0;
    double mp2_c_os = 0.0;
    double mp2_c_ss = 0.0;
};

// Defined below after the built-in alias table.  Registration calls it to
// reject names which already resolve to a built-in or dynamic libxc recipe.
AliasSpec resolve_alias(const std::string& name_in);

// Thread-safe dynamic alias registry.  User-defined functionals registered
// via register_functional_alias live here and are consulted by
// resolve_alias after the built-in static map.
namespace {
std::mutex g_dynamic_aliases_mutex;
std::unordered_map<std::string, AliasSpec> g_dynamic_aliases;

struct ExternalSpec {
    std::shared_ptr<ExternalXCProvider> provider;
    double hf_exchange_fraction = 0.0;
    ExternalXCCapabilities capabilities;
};

// Intentionally process-lifetime storage.  A Python-backed provider owns a
// py::object in the bindings translation unit; destroying that object during
// C++ static teardown can run after Python has finalized and is unsafe.  The
// tiny registry is therefore leaked at process exit, just like CPython's own
// immortal module state, while individual Functional objects retain ordinary
// shared ownership during calculations.
std::mutex& external_functionals_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}
std::unordered_map<std::string, ExternalSpec>& external_functionals() {
    static auto* registry =
        new std::unordered_map<std::string, ExternalSpec>();
    return *registry;
}

std::string _dynamic_lower(const std::string& s) {
    std::string r(s);
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return r;
}

std::string canonical_external_grid_profile(const std::string& profile) {
    std::string key = _dynamic_lower(profile);
    key.erase(std::remove_if(key.begin(), key.end(),
                             [](unsigned char c) { return std::isspace(c); }),
              key.end());
    std::replace(key.begin(), key.end(), '_', '-');
    if (key.empty()) return {};
    if (key == "generic") return "generic";
    if (key == "pyscf-level3" || key == "pyscflevel3"
        || key == "skala") {
        return "pyscf-level3";
    }
    throw std::invalid_argument(
        "register_external_functional: unsupported required_grid_profile '"
        + profile + "' (use '', 'generic', or 'pyscf-level3')");
}
}  // namespace

}  // namespace (closes the outer anonymous namespace from line 36)

bool dynamic_alias_available(const std::string& name) {
    const auto key = _dynamic_lower(name);
    std::lock_guard<std::mutex> lock(g_dynamic_aliases_mutex);
    return g_dynamic_aliases.find(key) != g_dynamic_aliases.end();
}

std::optional<ExternalXCRegistration>
find_external_functional_registration(const std::string& name) {
    const auto key = _dynamic_lower(name);
    std::lock_guard<std::mutex> lock(external_functionals_mutex());
    const auto it = external_functionals().find(key);
    if (it == external_functionals().end()) return std::nullopt;
    return ExternalXCRegistration{
        it->second.provider,
        it->second.hf_exchange_fraction,
        it->second.capabilities,
    };
}

void register_functional_alias(
    const std::string& name,
    const std::vector<std::pair<std::string, double>>& components,
    double hf_exchange_fraction) {
    const auto key = _dynamic_lower(name);
    AliasSpec spec;
    spec.extra_hf_fraction = hf_exchange_fraction;
    for (const auto& kv : components) {
        const auto& comp_name = kv.first;
        double weight = kv.second;
        int id = xc_functional_get_number(comp_name.c_str());
        if (id <= 0) {
            throw std::invalid_argument(
                "register_functional_alias: unknown libxc component name '"
                + comp_name + "'");
        }
        spec.components.push_back({id, weight});
    }
    if (spec.components.empty()) {
        throw std::invalid_argument(
            "register_functional_alias: component list must not be empty for '"
            + name + "'");
    }
    // Preserve the public alias API's historical replace-in-place semantics.
    // Use one lock order for both registries so a dynamic alias and an
    // external provider cannot race to claim the same case-folded name; an
    // external provider is insert-only and therefore cannot be displaced.
    std::scoped_lock lock(
        g_dynamic_aliases_mutex, external_functionals_mutex());
    if (external_functionals().find(key)
        != external_functionals().end()) {
        throw std::invalid_argument(
            "register_functional_alias: name '" + name
            + "' is already registered as an external functional");
    }
    g_dynamic_aliases[key] = std::move(spec);
}

void register_external_functional(
    const std::string& name,
    std::shared_ptr<ExternalXCProvider> provider,
    double hf_exchange_fraction,
    ExternalXCCapabilities capabilities) {
    register_external_functional_family(
        {name}, std::move(provider), hf_exchange_fraction,
        std::move(capabilities));
}

void register_external_functional_family(
    const std::vector<std::string>& names,
    std::shared_ptr<ExternalXCProvider> provider,
    double hf_exchange_fraction,
    ExternalXCCapabilities capabilities) {
    if (names.empty()) {
        throw std::invalid_argument(
            "register_external_functional_family: names must not be empty");
    }
    if (!provider) {
        throw std::invalid_argument(
            "register_external_functional: provider must not be null");
    }
    if (!std::isfinite(hf_exchange_fraction)
        || hf_exchange_fraction < 0.0 || hf_exchange_fraction > 1.0) {
        throw std::invalid_argument(
            "register_external_functional: hf_exchange_fraction must be "
            "between 0 and 1");
    }
    if (capabilities.version != kExternalXCCapabilityVersion) {
        throw std::invalid_argument(
            "register_external_functional: unsupported capability version "
            + std::to_string(capabilities.version) + " (this core supports "
            + std::to_string(kExternalXCCapabilityVersion) + ")");
    }
    capabilities.required_grid_profile = canonical_external_grid_profile(
        capabilities.required_grid_profile);

    std::vector<std::string> keys;
    keys.reserve(names.size());
    for (const auto& name : names) {
        if (name.empty()) {
            throw std::invalid_argument(
                "register_external_functional: name must not be empty");
        }
        if (name.find_first_not_of(" \t\r\n") == std::string::npos) {
            throw std::invalid_argument(
                "register_external_functional: name must not be blank");
        }
        const auto key = _dynamic_lower(name);
        if (std::find(keys.begin(), keys.end(), key) != keys.end()) {
            throw std::invalid_argument(
                "register_external_functional: duplicate family name '"
                + name + "'");
        }
        keys.push_back(key);
    }

    // External providers are resolved ahead of libxc in Functional's
    // constructor.  Refuse an existing alias instead of permitting a plugin
    // to silently hijack a built-in such as PBE.  Rejecting re-registration
    // also keeps every Python-backed provider retained by the intentionally
    // process-lifetime registry; no displaced py::object can then be
    // destroyed on a native thread without the GIL.
    for (const auto& name : names) {
        bool already_resolves = false;
        try {
            (void)resolve_alias(name);
            already_resolves = true;
        } catch (const std::invalid_argument&) {
            // Unknown names are eligible for external registration.
        }
        if (already_resolves) {
            throw std::invalid_argument(
                "register_external_functional: name '" + name
                + "' already resolves to an existing functional");
        }
    }

    std::scoped_lock lock(
        g_dynamic_aliases_mutex, external_functionals_mutex());
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (g_dynamic_aliases.find(keys[i]) != g_dynamic_aliases.end()
            || external_functionals().find(keys[i])
                   != external_functionals().end()) {
            throw std::invalid_argument(
                "register_external_functional: name '" + names[i]
                + "' is already registered");
        }
    }

    // Build a complete replacement before touching the immortal registry.
    // Copy/insert failures therefore leave it unchanged, while swap is
    // noexcept for this standard-allocator unordered_map.
    auto updated = external_functionals();
    updated.reserve(updated.size() + names.size());
    for (const auto& key : keys) {
        updated.emplace(
            key,
            ExternalSpec{provider, hf_exchange_fraction, capabilities});
    }
    external_functionals().swap(updated);
}

namespace {

// PWPB95 (Goerigk-Grimme 2011) reparametrises libxc's mPW91 exchange
// into the "PW6" modified-PW form. The exponential-term parameter is
// published as c_pw / X2S², where X2S = 1/(2·(6π²)^(1/3)) is libxc's
// reduced-gradient conversion constant (|∇ρ|/ρ^(4/3) → s). The Psi4
// reference implementation derives libxc's ``_alpha`` external
// parameter the same way, so this value transplants directly into the
// vendored libxc (7.0.0); the ORCA cross-check in tests/test_pwpb95.py
// confirms it end-to-end.
constexpr double kPWPB95_X2S =
    0.1282782438530421943003109254455883701296;
constexpr double kPWPB95_MPW_ALPHA =
    0.32620 / (kPWPB95_X2S * kPWPB95_X2S);

// Resolve a high-level alias to a weighted list of libxc XC_… IDs and an
// optional explicit HF-exchange fraction. At evaluation time each
// component's exc / v_ρ / v_σ output is multiplied by its weight before
// being summed into the running totals.
AliasSpec resolve_alias(const std::string& name_in) {
    const auto name = to_lower_copy(name_in);

    static const std::unordered_map<std::string, AliasSpec> aliases = {
        {"lda",   {{{XC_LDA_X, 1.0}, {XC_LDA_C_VWN, 1.0}}, 0.0}},
        {"svwn",  {{{XC_LDA_X, 1.0}, {XC_LDA_C_VWN, 1.0}}, 0.0}},
        {"slater",{{{XC_LDA_X, 1.0}}, 0.0}},           // exchange only
        {"pbe",   {{{XC_GGA_X_PBE, 1.0}, {XC_GGA_C_PBE, 1.0}}, 0.0}},
        // PBEsol — Perdew et al. PRL 100, 136406 (2008). The PBE
        // re-parameterisation for solids / surfaces (restores the
        // gradient expansion for exchange). Pure GGA, no HF.
        {"pbesol",  {{{XC_GGA_X_PBE_SOL, 1.0}, {XC_GGA_C_PBE_SOL, 1.0}}, 0.0}},
        {"pbe_sol", {{{XC_GGA_X_PBE_SOL, 1.0}, {XC_GGA_C_PBE_SOL, 1.0}}, 0.0}},
        {"blyp",  {{{XC_GGA_X_B88, 1.0}, {XC_GGA_C_LYP, 1.0}}, 0.0}},
        // B3LYP — the "B3LYP" recipe (HF + Slater + B88 exchange,
        // VWN + LYP correlation, Becke's 3 weights; Stephens, Devlin,
        // Chabalowski & Frisch, J. Phys. Chem. 98, 11623 (1994)) is
        // fixed; codes differ only in *which* Vosko-Wilk-Nusair
        // local-correlation parametrisation fills the 0.19-weighted
        // LSDA-correlation slot (Hertwig & Koch, Chem. Phys. Lett.
        // 268, 345 (1997)):
        //   XC_HYB_GGA_XC_B3LYP5  (id 475): VWN5 (Ceperley-Alder
        //     fit) — the ORCA / TURBOMOLE / ADF / CRYSTAL convention.
        //   XC_HYB_GGA_XC_B3LYP   (id 402): the VWN fit to the RPA
        //     electron-gas data — what Gaussian calls "VWN(III)".
        //     Convention of Gaussian, libxc, PySCF (b3lyp AND
        //     b3lypg as of 2.13), Psi4 (>= 1.2), NWChem, Q-Chem.
        // **vibe-qc ships the ORCA definition**: bare "b3lyp" is the
        // VWN5 variant (maintainer ruling, reaffirmed 2026-06-11
        // after a full cross-code audit). This is also internally
        // consistent — vibe-qc's "lda" / "svwn" already use VWN5.
        // "b3lyp5" is an explicit spelling of the same flavor (use
        // it in cross-code tables to be unambiguous); "b3lyp/g"
        // (ORCA spelling) and "b3lypg" (PySCF spelling) select the
        // Gaussian-compatible variant.
        //
        // Flavor gap, H2/STO-3G at 1.4 bohr (RKS, conv 1e-12):
        //   b3lyp == b3lyp5 (id 475) = -1.1586001482 Ha
        //     == PySCF 2.13 'b3lyp5' (libxc 7.0.0) to 1e-10,
        //     == CRYSTAL14 "B3LYP" -1.1586001474 (XLGRID),
        //     == ORCA "B3LYP" -1.158669 (DefGrid2);
        //   b3lyp/g == b3lypg (id 402) = -1.1654009284 Ha
        //     == PySCF 2.13 'b3lyp'/'b3lypg' to 1e-10,
        //     == ORCA "B3LYP/G" -1.165470 (DefGrid2) to ~1 µHa;
        //   gap = 0.19·(E_c[VWN_RPA] − E_c[VWN5]) = -6.80 mHa.
        // No "b3lyp3" alias on purpose: libxc's LDA_C_VWN_3 is the
        // Ceperley-Alder fit III, which equals VWN5 on closed-shell
        // densities — it is NOT Gaussian's "VWN(III)" (= the RPA
        // fit, libxc LDA_C_VWN_RPA), so the name would select the
        // wrong flavor exactly when users reach for it.
        // All four are libxc hybrids, so xc_hyb_exx_coef supplies
        // the 0.20 HF fraction (extra_hf_fraction = 0.0).
        {"b3lyp",   {{{XC_HYB_GGA_XC_B3LYP5, 1.0}}, 0.0}},
        {"b3lyp5",  {{{XC_HYB_GGA_XC_B3LYP5, 1.0}}, 0.0}},
        {"b3lyp/g", {{{XC_HYB_GGA_XC_B3LYP,  1.0}}, 0.0}},
        {"b3lypg",  {{{XC_HYB_GGA_XC_B3LYP,  1.0}}, 0.0}},
        // PBE0 (a.k.a. PBE1PBE, PBEh): 25% HF exchange + 75% PBE-X +
        // PBE-C. libxc's name is XC_HYB_GGA_XC_PBEH (id 406) — the
        // Adamo / Barone 1999 non-empirical hybrid, generally a
        // stronger hybrid than B3LYP for first-row-organic
        // thermochemistry. XC_HYB_GGA_XC_PBEH is a libxc hybrid, so
        // xc_hyb_exx_coef supplies its 0.25 HF fraction automatically
        // (extra_hf_fraction = 0.0, exactly as for b3lyp above).
        {"pbe0",  {{{XC_HYB_GGA_XC_PBEH, 1.0}}, 0.0}},
        // PW1PW — Bredow & Gerson 1-parameter global hybrid
        // (Phys. Rev. B 61, 5194 (2000), §III.D "Hybrid approach
        // based on the PWGGA method"; PW1PW is the post-2013
        // CRYSTAL-community shorthand).
        //
        //   E_x  = 0.20 · E_x^HF + 0.80 · E_x^PW91
        //   E_c  =                    1.00 · E_c^PW91
        //
        // The 0.20 HF mixing was empirically optimized on
        // MgO / NiO / CoO bulk properties; it is the reference
        // functional behind the pob-TZVP basis-set papers
        // (Peintinger-Vilela Oliveira-Bredow 2013) and the
        // canonical solid-state hybrid for ionic / wide-gap systems.
        //
        // Neither GGA_X_PW91 (id 109) nor GGA_C_PW91 (id 134) is a
        // libxc hybrid, so ``xc_hyb_exx_coef`` returns 0 for both;
        // the 0.20 HF fraction is registered explicitly via
        // ``extra_hf_fraction`` and flows through
        // ``Functional::hf_exchange_fraction()`` into the SCF / DF
        // / gradient drivers exactly like B3LYP's 0.20 or PBE0's
        // 0.25 do.
        {"pw1pw", {{{XC_GGA_X_PW91, 0.80},
                    {XC_GGA_C_PW91, 1.00}}, 0.20}},
        // B2PLYP (Grimme, J. Chem. Phys. 124, 034108 (2006)) — the
        // first practical double hybrid:
        //
        //   E_xc^B2PLYP = 0.53 · E_x^HF + 0.47 · E_x^B88
        //               + 0.73 · E_c^LYP + 0.27 · E_c^PT2
        //
        // The first three terms (HF + B88 exchange + LYP correlation)
        // are evaluated at the SCF level — this is the ``Functional``
        // piece, a custom-weighted hybrid GGA registered exactly like
        // PW1PW. The fourth term is an MP2 correction on the converged
        // KS orbitals with c_os = c_ss = 0.27, signalled by the
        // ``mp2_c_os`` / ``mp2_c_ss`` AliasSpec fields and exposed via
        // ``Functional::mp2_c_os()`` / ``mp2_c_ss()`` /
        // ``is_double_hybrid()``. Neither GGA_X_B88 (id 106) nor
        // GGA_C_LYP (id 131) is a libxc hybrid, so the 0.53 HF mixing
        // is registered explicitly via ``extra_hf_fraction`` (same
        // pattern as PW1PW). The dispatcher that turns this Functional
        // into an actual run is ``vibeqc.run_b2plyp`` (Python).
        {"b2plyp", {{{XC_GGA_X_B88, 0.47},
                     {XC_GGA_C_LYP, 0.73}}, 0.53, 0.27, 0.27}},
        // DSD-PBEP86 (Kozuch & Martin, Phys. Chem. Chem. Phys. 13,
        // 20104 (2011), doi:10.1039/c1cp22592h) — the
        // **D**ispersion-corrected, **S**pin-component-scaled,
        // **D**ouble-hybrid built on PBE exchange and Perdew 86
        // correlation. Asymmetric MP2 scaling (c_os ≠ c_ss) is the
        // distinguishing feature; this is exactly the case the SCS-MP2
        // surface (c_os, c_ss on MP2Options) makes natural.
        //
        // Final recommended D3(BJ)-fit parameter set, Kozuch-Martin
        // 2011, p. 20106 ("The final and recommended DSD-PBEP86
        // parameters with the D3BJ dispersion scheme are c_S = 0.25,
        // c_O = 0.53, c_X = 0.30, c_C = 0.43"; eqn (1) there defines
        // c_X as the *DFT*-exchange fraction, so HF exchange = 0.70):
        //
        //   E_xc^DSD-PBEP86 = 0.70 · E_x^HF + 0.30 · E_x^PBE
        //                   + 0.43 · E_c^P86
        //                   + 0.53 · E_c^PT2,os + 0.25 · E_c^PT2,ss
        //
        // This is byte-for-byte the recipe ORCA's ``DSD-PBEP86``
        // keyword prints (ScalHFX 0.700, ScalDFX 0.300, ScalDFC =
        // ScalLDAC 0.430, MP2 aa/bb 0.25, ab 0.53) and it pairs with
        // the matching 2011 D3(BJ) damping fit (s6 = 0.418, a1 = 0,
        // s8 = 0, a2 = 5.65) that dispersion.py routes via its
        // "dsd-pbep86" → "dsdpbep86_2011" alias.
        //
        // TWO same-name traps live under this name (BUG
        // DSDPBEP86-DFT-PART-1P6MHA):
        //
        // (1) Parameter revision. Kozuch & Martin *revised* every
        //     coefficient in J. Comput. Chem. 34, 2327 (2013),
        //     doi:10.1002/jcc.23391 (SI, DSD-DFT-D3BJ table, pbe/p86
        //     column: c_x,HF = 0.69, c_c = 0.44, c_s = 0.22,
        //     c_o = 0.52, s6 = 0.48, a2 = 5.6). vibe-qc's
        //     "dsd-pbep86" is the 2011 set above, matching ORCA and
        //     the 2011 dispersion fit. A pre-fix revision of this
        //     entry mixed the two (2013 SCF piece 0.69/0.31/0.44)
        //     with MP2 scalings 0.55/0.09 transcribed from *adjacent
        //     columns* of the 2013 SI table (0.55 is DSD-PBEPBE's
        //     c_o, 0.09 DSD-PBEB95's c_s).
        //
        // (2) P86 local-part flavor. libxc's GGA_C_P86 (id 132) is
        //     Perdew's original 1986 form on the PZ81 local
        //     correlation; GGA_C_P86VWN (id 252, "Perdew 86 based on
        //     the VWN5 LDA") is the flavor ORCA composes. Measured on
        //     H2O/def2-SVP (case mb014, conventional ORCA 6.1.1
        //     reference, 2026-08-06 regression check): PZ81
        //     flavor deviates +1.85 mHa in the SCF part, VWN5 flavor
        //     +0.005 mHa (grid residual). GGA_C_P86VWN_FT (exact
        //     ftilde) lands at -0.029 mHa, so ORCA uses the 1.745
        //     approximation — GGA_C_P86VWN it is.
        //
        // Combined pre-fix defect: -1.58 mHa in the DFT part alone on
        // mb014 plus unpublished MP2 scalings. Pinned by
        // tests/test_dsd_pbep86.py against the conventional ORCA
        // 6.1.1 reference.
        //
        // The "no-D" variant landed here is the XC + MP2 part without
        // dispersion — pair with ``vibeqc.compute_d3bj`` (Phase D1)
        // for the dispersion-augmented total, and watch for the
        // dedicated D4 parameterisation for DSD-PBEP86 when it ships.
        //
        // Neither GGA_X_PBE (id 101) nor GGA_C_P86VWN (id 252) is a
        // libxc hybrid, so the 0.70 HF mixing is registered explicitly
        // via ``extra_hf_fraction``. The dispatcher is
        // ``vibeqc.run_dsd_pbep86`` (Python).
        {"dsd-pbep86", {{{XC_GGA_X_PBE, 0.30},
                         {XC_GGA_C_P86VWN, 0.43}}, 0.70, 0.53, 0.25}},
        // Alias without the dash for codes / scripts that prefer
        // ``DSDPBEP86`` (matches the underscore-less style of e.g.
        // ``b3lyp``).
        {"dsdpbep86",  {{{XC_GGA_X_PBE, 0.30},
                         {XC_GGA_C_P86VWN, 0.43}}, 0.70, 0.53, 0.25}},
        // revDSD-PBEP86-D4 (Santra, Sylvetsky & Martin, J. Phys. Chem. A
        // 123, 5129 (2019), doi:10.1021/acs.jpca.9b03157) — the
        // GMTKN55-retrained revision of DSD-PBEP86. Same primitive
        // components as DSD-PBEP86 (PBE exchange + Perdew-86 correlation),
        // but every linear coefficient is re-optimised:
        //
        //   E_xc^revDSD-PBEP86-D4 = 0.69 · E_x^HF + 0.31 · E_x^PBE
        //                         + 0.4210 · E_c^P86
        //                         + 0.5922 · E_c^PT2,os + 0.0636 · E_c^PT2,ss
        //
        // Coefficients are the **D4** parametrisation from Santra-Sylvetsky-
        // Martin 2019 Table 4 (c_x,HF = 0.69, c_c,DFT = 0.4210,
        // c_2ab = 0.5922, c_2ss = 0.0636). The matching D4 damping
        // (s6 = 0.5132, s8 = 0, a1 = 0.44, a2 = 3.60) lives in
        // python/vibeqc/dispersion_d4_parameters.py ("revdsdpbep86") and is
        // folded in by run_double_hybrid(..., dispersion="d4"). NB the
        // -D3BJ member uses a *different* XC/MP2 fit (c_c = 0.4296,
        // c_2ab = 0.5785, c_2ss = 0.0799 with s6 = 0.4377, a1 = 0, a2 = 5.5),
        // so the two are not interchangeable — this alias is specifically
        // the D4 member. Neither GGA_X_PBE (id 101) nor GGA_C_P86VWN
        // (id 252) is a libxc hybrid, so the 0.69 HF mixing is registered
        // via extra_hf_fraction (same pattern as DSD-PBEP86). The
        // dispatcher is vibeqc.run_revdsd_pbep86 (Python).
        //
        // P86 local-part flavor: VWN5-local GGA_C_P86VWN (id 252), the
        // same flavor as "dsd-pbep86" above. The 2019 refit itself was run
        // in Q-Chem 5.1 (Santra 2019, "Electronic structure details"), but
        // the paper's own Electronic Supporting Information settles the
        // intended composition: the ORCA sample input for
        // revDSD-PBEP86-D3(BJ) (ESI p. S22, deck 3) specifies
        //
        //     Exchange X_PBE / Correlation C_P86
        //     ScalHFX 0.69 / ScalDFX 0.31 / ScalGGAC = ScalLDAC = 0.4296
        //
        // and ORCA composes that keyword on VWN5 — it prints "LDA part of
        // GGA corr.  LDAOpt .... VWN-5". The authors' Gaussian deck
        // (ESI p. S20, deck 3, iop(3/78=0429604296)) scales local and
        // nonlocal correlation by the same 0.4296, i.e. one uniform c_c
        // over the whole composed correlation, which is what the single
        // 0.4210 coefficient below expresses for the D4 member. The ESI
        // decks are for the D3(BJ) member, but they fix the *primitive
        // choice* (which P86), not the linear coefficients, so the flavor
        // verdict carries over to the D4 member unchanged.
        //
        // Pre-fix this entry used PZ81-local GGA_C_P86 (id 132), which
        // matches neither the authors' prescription nor any code that
        // implements revDSD. Measured on witness M19 (H2O/cc-pVTZ, ORCA
        // 6.1.1 reference rp163-orca-m19-revdsd): the flavor swap alone
        // moves the SCF half by -1.7969 mHa (+2.4582 -> +0.6613 mHa vs
        // ORCA). See tests/test_revdsd_pbep86.py for the pinned
        // decomposition (BUG DSDPBEP86-DFT-PART-1P6MHA follow-up, GitLab
        // issue 32).
        //
        // REVISION TRAP — 2019 vs 2021. The coefficients below are the
        // 2019 Table 4 D4 set. ORCA's "REVDSD-PBEP86-D4/2021" keyword is a
        // *later* re-parametrisation and prints ScalDFC = ScalLDAC =
        // 0.4224, not 0.4210; on M19 that difference is worth a further
        // -0.5095 mHa in the SCF half. vibe-qc ships the 2019 set that its
        // citation route claims (santra_martin_revdsd_2019) and does NOT
        // guess the 2021 values — the defining 2021 paper is not in the
        // reference library (requested in ../library/maintenance/
        // manual-dois.txt). Do not "close the last 0.5 mHa" against an
        // ORCA /2021 reference by editing these numbers; that would leave
        // the alias citing a paper it no longer implements. A 2021 member
        // needs its own alias, its own D4 damping set, and its own
        // citation entry.
        {"revdsd-pbep86", {{{XC_GGA_X_PBE, 0.31},
                            {XC_GGA_C_P86VWN, 0.4210}}, 0.69, 0.5922, 0.0636}},
        {"revdsdpbep86",  {{{XC_GGA_X_PBE, 0.31},
                            {XC_GGA_C_P86VWN, 0.4210}}, 0.69, 0.5922, 0.0636}},
        // PWPB95 (Goerigk & Grimme, J. Chem. Theory Comput. 7, 291
        // (2011)) — the spin-opposite-scaled (SOS) double-hybrid
        // *meta-GGA*:
        //
        //   E_xc^PWPB95 = 0.50 · E_x^HF   + 0.50 · E_x^mPW(PW6)
        //               + 0.731 · E_c^B95 + 0.269 · E_c^PT2,os
        //
        // Two things make PWPB95 unlike B2PLYP / DSD-PBEP86:
        //
        //  1. It is the *SOS* member of the line — the same-spin PT2
        //     coefficient is zero (mp2_c_ss = 0), so the post-SCF
        //     correction is opposite-spin MP2 only. ``is_double_hybrid``
        //     still fires because mp2_c_os ≠ 0.
        //  2. It does not mix *stock* libxc components. The exchange is
        //     the "PW6" reparametrisation of mPW91 (re-tuned
        //     _bt / _alpha / _expo) and the correlation is B95 with
        //     re-tuned same-/opposite-spin _css / _copp. Those values
        //     are pushed into libxc through the per-component
        //     ``ext_params`` overrides below (the first alias to use
        //     them — see ``AliasComponent::ext_params``).
        //
        // Neither GGA_X_MPW91 (id 119) nor MGGA_C_BC95 (id 240) is a
        // libxc hybrid, so the 0.50 HF mixing is registered explicitly
        // via ``extra_hf_fraction`` (same pattern as B2PLYP / PW1PW).
        // BC95 is a meta-GGA, so the composite resolves to XCKind::MGGA
        // and the SCF step runs through the τ-dependent KS path. The
        // dispatcher is ``vibeqc.run_pwpb95`` (Python).
        {"pwpb95", {{{XC_GGA_X_MPW91, 0.50,
                      {{"_bt", 0.004440},
                       {"_alpha", kPWPB95_MPW_ALPHA},
                       {"_expo", 3.7868}}},
                     {XC_MGGA_C_BC95, 0.731,
                      {{"_css", 0.03241},
                       {"_copp", 0.00250}}}}, 0.50, 0.269, 0.0}},
        // ---- Meta-GGA (τ-dependent) functionals --------------------------
        //
        // TPSS — Tao, Perdew, Staroverov, Scuseria PRL 91, 146401 (2003).
        // The original non-empirical meta-GGA: exchange + correlation
        // satisfy more exact constraints than PBE without empirical
        // fitting. ``XC_MGGA_X_TPSS`` (id 202) + ``XC_MGGA_C_TPSS`` (id
        // 231) are both pure meta-GGAs (no HF mixing).
        {"tpss", {{{XC_MGGA_X_TPSS, 1.0},
                   {XC_MGGA_C_TPSS, 1.0}}, 0.0}},
        // revTPSS — Perdew, Ruzsinszky, Csonka, Constantin, Sun PRL 103,
        // 026403 (2009). The revised TPSS meta-GGA (better lattice
        // constants / surface energies). Pure meta-GGA, no HF.
        // ``XC_MGGA_X_REVTPSS`` (id 212) + ``XC_MGGA_C_REVTPSS`` (id 241).
        {"revtpss", {{{XC_MGGA_X_REVTPSS, 1.0},
                      {XC_MGGA_C_REVTPSS, 1.0}}, 0.0}},
        {"rev-tpss", {{{XC_MGGA_X_REVTPSS, 1.0},
                       {XC_MGGA_C_REVTPSS, 1.0}}, 0.0}},
        // TPSSh — Staroverov, Scuseria, Tao, Perdew JCP 119, 12129
        // (2003). The 10% global hybrid of TPSS. ``XC_HYB_MGGA_XC_TPSSH``
        // (id 457) is a libxc-builtin hybrid combining both exchange and
        // correlation in one functional ID; ``xc_hyb_exx_coef`` returns
        // 0.10 automatically (extra_hf_fraction = 0.0). One of the
        // best-tested hybrid meta-GGAs for thermochemistry / kinetics
        // and the natural meta-GGA upgrade path from PBE0 / B3LYP for
        // molecular work.
        {"tpssh", {{{XC_HYB_MGGA_XC_TPSSH, 1.0}}, 0.0}},
        // M06-L — Zhao & Truhlar JCP 125, 194101 (2006). The pure
        // (non-hybrid) Minnesota meta-GGA; fast, no exact exchange,
        // strong performance on transition-metal thermochemistry.
        // ``XC_MGGA_X_M06_L`` (id 203) + ``XC_MGGA_C_M06_L`` (id 233).
        {"m06-l", {{{XC_MGGA_X_M06_L, 1.0},
                    {XC_MGGA_C_M06_L, 1.0}}, 0.0}},
        // M06-2X — Zhao & Truhlar Theor Chem Acc 120, 215 (2008). The
        // 54% global-hybrid Minnesota meta-GGA, parameterised for
        // main-group thermochemistry + non-covalent interactions. The
        // exchange piece ``XC_HYB_MGGA_X_M06_2X`` (id 450) reports its
        // 0.54 HF fraction through ``xc_hyb_exx_coef``; the correlation
        // piece ``XC_MGGA_C_M06_2X`` (id 236) is pure meta-GGA.
        // extra_hf_fraction = 0.0 — both libxc components together
        // already encode the full HF mixing.
        {"m06-2x", {{{XC_HYB_MGGA_X_M06_2X, 1.0},
                     {XC_MGGA_C_M06_2X,     1.0}}, 0.0}},
        // Alias without the dash for codes/scripts that prefer
        // ``m062x`` (matches the ``dsdpbep86`` precedent above).
        {"m062x",  {{{XC_HYB_MGGA_X_M06_2X, 1.0},
                     {XC_MGGA_C_M06_2X,     1.0}}, 0.0}},
        {"m06l",   {{{XC_MGGA_X_M06_L, 1.0},
                     {XC_MGGA_C_M06_L, 1.0}}, 0.0}},
        // ---- SCAN / r²SCAN family ----------------------------------------
        //
        // SCAN — Sun, Ruzsinszky, Perdew PRL 115, 036402 (2015). The
        // "strongly constrained and appropriately normed" meta-GGA:
        // satisfies all 17 known exact constraints a semilocal
        // functional can. τ-dependent (via the iso-orbital indicator
        // α = (τ − τ_W)/τ_unif) but laplacian-free by design — it is a
        // pure meta-GGA reachable through vibe-qc's τ-only MGGA grid.
        // ``XC_MGGA_X_SCAN`` (id 263) + ``XC_MGGA_C_SCAN`` (id 267).
        {"scan", {{{XC_MGGA_X_SCAN, 1.0},
                   {XC_MGGA_C_SCAN, 1.0}}, 0.0}},
        // r²SCAN — Furness, Kaplan, Ning, Perdew, Sun JPCL 11, 8208
        // (2020). The re-regularized SCAN: same exact-constraint
        // satisfaction with a much smoother enhancement factor →
        // far better numerical stability and grid convergence than
        // SCAN, at no accuracy cost. The recommended default of the
        // family. ``XC_MGGA_X_R2SCAN`` (id 497) + ``XC_MGGA_C_R2SCAN``
        // (id 498). Also laplacian-free.
        {"r2scan", {{{XC_MGGA_X_R2SCAN, 1.0},
                     {XC_MGGA_C_R2SCAN, 1.0}}, 0.0}},
        // r²SCAN01. Furness, Kaplan, Ning, Perdew, Sun JCP 156,
        // 034109 (2022). A re-regularization of r²SCAN with a larger
        // value of the η regularization parameter that restores the
        // fourth-order term of the slowly-varying density-gradient
        // expansion which r²SCAN had traded away for smoothness. Aimed
        // at slowly-varying densities (bulk metals, bonding regions)
        // where r²SCAN slightly underbinds. Still τ-only (laplacian-
        // free), so it rides the same MGGA grid path as r²SCAN.
        // ``XC_MGGA_X_R2SCAN01`` (id 645) + ``XC_MGGA_C_R2SCAN01``
        // (id 642).
        {"r2scan01", {{{XC_MGGA_X_R2SCAN01, 1.0},
                       {XC_MGGA_C_R2SCAN01, 1.0}}, 0.0}},
        // r²SCAN0 — Bursch, Neugebauer, Grimme et al; the 25%-HF global
        // hybrid of r²SCAN (PBE0-like mixing). ``XC_HYB_MGGA_XC_R2SCAN0``
        // (id 660) is a libxc-builtin hybrid; ``xc_hyb_exx_coef``
        // supplies the 0.25 fraction automatically.
        {"r2scan0", {{{XC_HYB_MGGA_XC_R2SCAN0, 1.0}}, 0.0}},
        // r²SCANh — the 10%-HF global hybrid of r²SCAN (TPSSh-like
        // mixing). ``XC_HYB_MGGA_XC_R2SCANH`` (id 659).
        {"r2scanh", {{{XC_HYB_MGGA_XC_R2SCANH, 1.0}}, 0.0}},
        // ---- Range-separated hybrids -------------------------------------
        //
        // ωB97X — Chai & Head-Gordon JCP 128, 084106 (2008). A
        // long-range-corrected hybrid GGA: the exact-exchange admixture
        // ramps from a fixed short-range fraction up to 100 % HF at long
        // range via the error function. ``XC_HYB_GGA_XC_WB97X`` (id 464)
        // is a libxc-builtin range-separated functional — the CAM
        // parameters (ω, α, β) are read at construction via
        // ``xc_hyb_cam_coef`` and surfaced through
        // ``Functional::rsh_omega() / cam_alpha() / cam_beta()``. The
        // libxc ``xc_gga_exc_vxc`` call returns the (short-range) DFT
        // exchange + correlation; the SCF driver adds the position-
        // dependent HF exchange via a second erf-attenuated K build.
        // (ωB97X-D — the dispersion-corrected re-parameterisation,
        // libxc id 471 — registers separately once the Chai-Head-Gordon
        // dispersion term is wired; see handovers/HANDOVER_META_GGA.md § 4.)
        {"wb97x", {{{XC_HYB_GGA_XC_WB97X, 1.0}}, 0.0}},
        // ωB97 — Chai & Head-Gordon, JCP 128, 084106 (2008). The parent
        // range-separated hybrid of the ωB97X/ωB97X-D family (100%
        // long-range HF, no short-range HF). libxc builtin
        // ``XC_HYB_GGA_XC_WB97`` (id 463) reports its RSH parameters via
        // xc_hyb_cam_coef; extra_hf_fraction = 0.0.
        {"wb97", {{{XC_HYB_GGA_XC_WB97, 1.0}}, 0.0}},
        // ωB97X-D — Chai & Head-Gordon, PCCP 10, 6615 (2008). The
        // dispersion-corrected re-parameterisation of ωB97X.
        // ``XC_HYB_GGA_XC_WB97X_D`` (id 471) is the *XC* part — a
        // range-separated hybrid GGA (ω = 0.2, 22.2 % HF short-range
        // → 100 % long-range), evaluated through the same RSH path as
        // ``wb97x``. The intrinsic "-D" empirical dispersion (the
        // Chai-Head-Gordon DFT-D2-type term) is **not** part of the
        // libxc functional — it is a geometry-only additive
        // correction (``vibeqc::compute_chg_dispersion``). This alias
        // therefore resolves to the XC/SCF piece only; the
        // ``vibeqc.run_wb97x_d`` dispatcher adds the dispersion for
        // the complete ωB97X-D total energy (same split as the
        // ``b2plyp`` alias vs ``run_b2plyp``).
        {"wb97x-d", {{{XC_HYB_GGA_XC_WB97X_D, 1.0}}, 0.0}},
        {"wb97xd",  {{{XC_HYB_GGA_XC_WB97X_D, 1.0}}, 0.0}},
        // ωB97X-V (Mardirossian-Head-Gordon, Phys. Chem. Chem. Phys. 16,
        // 9904 (2014)) — the VV10-paired range-separated hybrid GGA;
        // functional parent of the v0.9.0 ωB97X-3c composite. Same RSH
        // machinery as ωB97X above. libxc id 466 carries the
        // XC_FLAGS_VV10 flag, so ``needs_vv10()`` fires and the SCF V_xc
        // builders add the VV10 nonlocal correlation (b = 6.0, C = 0.01
        // from xc_nlc_coef) self-consistently — this is the *complete*
        // ωB97X-V functional, not the semilocal-only part. (Inside the
        // ωB97X-3c recipe the cheaper D4 dispersion stands in for VV10;
        // that is the 3c composite's choice, handled in vibeqc.composites,
        // not here.)
        {"wb97x-v", {{{XC_HYB_GGA_XC_WB97X_V, 1.0}}, 0.0}},
        // VV10 — the original Vydrov-Van Voorhis functional (J. Chem.
        // Phys. 133, 244103 (2010)): rPW86 exchange + PBE correlation +
        // the VV10 nonlocal term (b = 5.9, C = 0.0093). libxc id 255
        // provides the semilocal part and carries XC_FLAGS_VV10, so the
        // nonlocal double integral is added by compute_vv10 exactly as for
        // ωB97X-V. A clean standalone target for validating the nonlocal
        // kernel in isolation.
        {"vv10", {{{XC_GGA_XC_VV10, 1.0}}, 0.0}},
        // ωB97M-V (Mardirossian-Head-Gordon, J. Chem. Phys. 144, 214110
        // (2016)) — the VV10-paired, range-separated *meta*-GGA hybrid;
        // the most accurate of the ωB97-V line and a leading general-
        // purpose functional. libxc id 531 (XC_HYB_MGGA_XC_WB97M_V) is a
        // CAM meta-GGA: it resolves to XCKind::MGGA (τ-dependent KS path),
        // reports its range-separation (ω = 0.3, cam_alpha = 0.15,
        // cam_beta = 0.85 → 100 % HF at long range) via xc_hyb_cam_coef,
        // and carries XC_FLAGS_VV10, so the same compute_vv10 machinery as
        // ωB97X-V adds the nonlocal correlation (b = 6.0, C = 0.01)
        // self-consistently. All three pieces — RSH erf-K, meta-GGA τ, and
        // VV10 — compose through the existing direct-SCF path; this is the
        // complete ωB97M-V functional.
        {"wb97m-v", {{{XC_HYB_MGGA_XC_WB97M_V, 1.0}}, 0.0}},
        // HSE06 (Heyd-Scuseria-Ernzerhof 2003/2006) — the screened
        // (short-range-only) hybrid; α = 0 long-range HF, 0.25 short-
        // range HF, ω = 0.11 bohr⁻¹. Solid-state workhorse; functional
        // parent of the v0.9.0 HSE-3c composite. In vibe-qc's erf-form
        // CAM representation cam_alpha = +0.25, cam_beta = −0.25
        // (EXX(r→0) = 0.25, EXX(r→∞) = 0) — handled by the milestone-A
        // RSH erf-attenuated-K machinery with a negative long-range
        // coefficient; no separate erfc kernel needed.
        {"hse06", {{{XC_HYB_GGA_XC_HSE06, 1.0}}, 0.0}},
        // CAM-B3LYP (Yanai, Tew, Handy 2004): Coulomb-attenuating B3LYP
        // with 19% short-range HF exchange and 65% long-range HF exchange.
        {"cam-b3lyp", {{{XC_HYB_GGA_XC_CAM_B3LYP, 1.0}}, 0.0}},
        {"camb3lyp",  {{{XC_HYB_GGA_XC_CAM_B3LYP, 1.0}}, 0.0}},
        // BHandHLYP (a.k.a. BHLYP) — Becke's "half-and-half" global
        // hybrid: 50% HF + 50% B88 exchange, LYP correlation. libxc
        // builtin ``XC_HYB_GGA_XC_BHANDHLYP`` (id 436) encodes the 0.5 HF
        // fraction (reported via xc_hyb_exx_coef); extra_hf_fraction = 0.0.
        {"bhandhlyp", {{{XC_HYB_GGA_XC_BHANDHLYP, 1.0}}, 0.0}},
        {"bhlyp",     {{{XC_HYB_GGA_XC_BHANDHLYP, 1.0}}, 0.0}},
        // LC-wPBE / LC-omegaPBE (Vydrov-Scuseria 2006):
        // long-range-corrected PBE, 0% short-range HF exchange and
        // 100% long-range HF exchange.
        {"lc-wpbe", {{{XC_HYB_GGA_XC_LC_WPBE, 1.0}}, 0.0}},
        {"lcwpbe",  {{{XC_HYB_GGA_XC_LC_WPBE, 1.0}}, 0.0}},
        // MN15 (Yu-He-Li-Truhlar 2016): global-hybrid Minnesota meta-GGA.
        {"mn15", {{{XC_HYB_MGGA_X_MN15, 1.0},
                   {XC_MGGA_C_MN15, 1.0}}, 0.0}},
        // ----------------------------------------------------------
        // Composite-3c parent functionals (v0.9.0). These are custom-
        // mixed hybrids registered for the corresponding 3c recipe and
        // are *only* meaningful at that recipe's native basis +
        // dispersion + gCP stack — see vibeqc.composites.
        // ----------------------------------------------------------
        // PBEh-3c modified PBE0 (Grimme, Brandenburg, Bannwarth,
        // Hansen, J. Chem. Phys. 143, 054107 (2015)): 42% HF exchange
        // + 58% PBE-X + PBE-C. Non-libxc-hybrid components, so the
        // 0.42 HF mixing is carried via extra_hf_fraction (same
        // pattern as PW1PW).
        {"pbeh-3c", {{{XC_GGA_X_PBE, 0.58},
                      {XC_GGA_C_PBE, 1.00}}, 0.42}},
        // B97-3c re-tuned Becke-97 GGA (Brandenburg, Bannwarth,
        // Hansen, Grimme, J. Chem. Phys. 148, 064104 (2018)) — pure
        // GGA, libxc id XC_GGA_XC_B97_3C.
        {"b97-3c", {{{XC_GGA_XC_B97_3C, 1.0}}, 0.0}},
    };

    auto it = aliases.find(name);
    if (it != aliases.end()) return it->second;

    // Check the dynamic (user-registered) alias table.
    {
        std::lock_guard<std::mutex> lock(g_dynamic_aliases_mutex);
        auto dit = g_dynamic_aliases.find(name);
        if (dit != g_dynamic_aliases.end()) return dit->second;
    }

    // ωB97M(2) (Mardirossian & Head-Gordon, J. Chem. Phys. 148, 241736
    // (2018)) is a named roadmap target, but its SCF base is a *re-fit* of
    // the B97M power-series functional (different linear coefficients from
    // ωB97M-V) plus a scaled-MP2 correction. libxc 7.0.0 ships ωB97M-V
    // (id 531) but NOT the ωB97M(2) base, so — unlike the other double
    // hybrids, which compose stock libxc components — it cannot be wired
    // as an alias here. Give a specific pointer instead of the generic
    // "unknown name" so a user copying the name from a paper knows why.
    if (name == "wb97m(2)" || name == "wb97m2" || name == "wb97m-2"
        || name == "wb97m(2)-v") {
        throw std::invalid_argument(
            "Functional: 'wb97m(2)' (ωB97M(2), Mardirossian-Head-Gordon "
            "2018) is not yet implemented — libxc 7.0.0 provides ωB97M-V "
            "('wb97m-v') but not the re-optimised B97M(2) semilocal base "
            "that ωB97M(2) requires, so it needs a dedicated from-scratch "
            "meta-GGA kernel rather than an alias. Use 'wb97m-v' for the "
            "(non-double-hybrid) parent functional.");
    }

    // Otherwise try to parse as a comma-separated list of numeric IDs
    // (each implicitly weight 1.0, no extra HF fraction).
    AliasSpec spec;
    std::stringstream ss(name_in);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Trim whitespace.
        auto l = tok.find_first_not_of(" \t");
        auto r = tok.find_last_not_of(" \t");
        if (l == std::string::npos) continue;
        tok = tok.substr(l, r - l + 1);
        try {
            spec.components.push_back({std::stoi(tok), 1.0});
        } catch (...) {
            throw std::invalid_argument(
                "Functional: unknown name '" + name_in + "'");
        }
    }
    if (spec.components.empty()) {
        throw std::invalid_argument(
            "Functional: empty or unrecognized name '" + name_in + "'");
    }
    return spec;
}

}  // namespace

Functional::Functional(const std::string& name, int spin)
    : impl_(std::make_unique<Impl>()), name_(name) {
    if (spin != 1 && spin != 2) {
        throw std::invalid_argument("Functional: spin must be 1 or 2");
    }
    impl_->spin = spin;

    // Full-grid providers are resolved before the libxc alias machinery.
    // They deliberately present as MGGA because the host must construct
    // rho, Cartesian grad(rho), and positive orbital tau for both spins.
    {
        const auto key = _dynamic_lower(name);
        std::lock_guard<std::mutex> lock(external_functionals_mutex());
        const auto it = external_functionals().find(key);
        if (it != external_functionals().end()) {
            impl_->external_provider = it->second.provider;
            impl_->external_capabilities = it->second.capabilities;
            kind_ = XCKind::MGGA;
            hf_exchange_fraction_ = it->second.hf_exchange_fraction;
            cam_alpha_ = hf_exchange_fraction_;
            return;
        }
    }

    const auto spec = resolve_alias(name);
    impl_->funcs.resize(spec.components.size());
    impl_->weights.resize(spec.components.size());
    const int libxc_spin = (spin == 1) ? XC_UNPOLARIZED : XC_POLARIZED;

    // Seed the HF-exchange fraction with any explicit contribution from the
    // alias (used by hand-mixed hybrids like PW1PW where the underlying
    // libxc components are pure GGAs). Any libxc-reported hybrid fractions
    // pick up below, scaled by their per-component weight.
    hf_exchange_fraction_ = spec.extra_hf_fraction;

    // Double-hybrid MP2-correction coefficients. Non-zero only for
    // B2PLYP / DSD-PBEP86 / etc.; see ``is_double_hybrid``.
    mp2_c_os_ = spec.mp2_c_os;
    mp2_c_ss_ = spec.mp2_c_ss;

    bool any_gga = false;
    bool any_mgga = false;
    for (std::size_t i = 0; i < spec.components.size(); ++i) {
        const int id = spec.components[i].libxc_id;
        const double w = spec.components[i].weight;
        if (xc_func_init(&impl_->funcs[i], id, libxc_spin) != 0) {
            throw std::runtime_error(
                "xc_func_init failed for functional id "
                + std::to_string(id));
        }
        impl_->weights[i] = w;
        // Mark this entry as initialized *before* any further check that
        // could throw, so ~Impl only runs xc_func_end on valid entries.
        impl_->n_init = i + 1;
        // Apply any libxc external-parameter overrides for this
        // component (reparametrised functionals — see
        // ``AliasComponent::ext_params`` and the PWPB95 alias). Each
        // ``xc_func_set_ext_params_name`` call re-runs libxc's
        // ``set_ext_params`` callback, so libxc's dependent internal
        // constants are re-derived; after the final call every override
        // is in effect together. Empty for every stock alias, so this
        // loop is a no-op on the B2PLYP / PBE / B3LYP / … paths.
        for (const auto& kv : spec.components[i].ext_params) {
            xc_func_set_ext_params_name(
                &impl_->funcs[i], kv.first.c_str(), kv.second);
        }
        // VV10 nonlocal correlation: libxc flags the component but leaves
        // the nonlocal double integral to the host. Pull the (b, C)
        // parameters here; the SCF V_xc builders evaluate compute_vv10 on
        // the grid when needs_vv10() is set. (All current VV10-paired
        // aliases use a single weight-1.0 component, so the nonlocal term
        // enters at unit weight; a weighted VV10 mix is not yet supported.)
        if (impl_->funcs[i].info->flags & XC_FLAGS_VV10) {
            double nlc_b = 0.0, nlc_C = 0.0;
            xc_nlc_coef(&impl_->funcs[i], &nlc_b, &nlc_C);
            needs_vv10_ = true;
            vv10_b_ = nlc_b;
            vv10_C_ = nlc_C;
        }
        const auto fam = impl_->funcs[i].info->family;
        if (fam == XC_FAMILY_GGA || fam == XC_FAMILY_HYB_GGA) any_gga = true;
        if (fam == XC_FAMILY_MGGA || fam == XC_FAMILY_HYB_MGGA) any_mgga = true;
        if (!(fam == XC_FAMILY_LDA
              || fam == XC_FAMILY_GGA
              || fam == XC_FAMILY_HYB_GGA
              || fam == XC_FAMILY_MGGA
              || fam == XC_FAMILY_HYB_MGGA)) {
            throw std::runtime_error(
                "Functional: only LDA / GGA / hybrid-GGA / meta-GGA / "
                "hybrid-meta-GGA supported (id " + std::to_string(id)
                + ", family " + std::to_string(fam) + ")");
        }
        // Meta-GGA functionals that additionally require the density
        // laplacian ∇²ρ — e.g. some BR-derived MGGAs — are not yet
        // supported. The kinetic-energy density τ alone is computed by
        // ``build_xc`` from the AO gradients; populating ∇²ρ needs the
        // AO second derivatives on the grid (a parallel workstream).
        if (impl_->funcs[i].info->flags & XC_FLAGS_NEEDS_LAPLACIAN) {
            throw std::runtime_error(
                "Functional: libxc id " + std::to_string(id) + " requires "
                "the density laplacian ∇²ρ, which vibe-qc's MGGA grid does "
                "not yet populate (only τ). Please use a τ-only meta-GGA "
                "such as TPSS, TPSSh, M06-L, or M06-2X.");
        }
        // Hybrid exchange mixing. Two cases:
        //
        //  * Range-separated (CAM) component — the EXX admixture is
        //    position-dependent: EXX(r) = α + β·erf(ω·r). libxc reports
        //    (ω, α, β) via xc_hyb_cam_coef; we accumulate the weighted
        //    α / β and record ω. ``hf_exchange_fraction`` picks up the
        //    full-range part α only (the long-range β term is carried
        //    separately in cam_beta_ and handled by the SCF driver's
        //    erf-attenuated K build).
        //  * Plain global hybrid — a single scalar fraction from
        //    xc_hyb_exx_coef. Scale by the component weight: a
        //    0.5*B3LYP mix contributes 0.5 × 0.20 to the total.
        if (impl_->funcs[i].info->flags & XC_FLAGS_HYB_CAM) {
            // libxc's CAM convention returns (ω, a, b) for the
            // exact-exchange operator written with the *erfc*
            // (short-range) attenuator:
            //
            //   K_xx = a · (1/r)  +  b · (erfc(ω·r)/r)
            //
            // e.g. ωB97X reports a = 1.0, b = −0.842294 (→ 100 % HF at
            // long range, 1 + b = 15.77 % at short range). vibe-qc's
            // SCF driver builds the *erf* (long-range) attenuated K,
            // so convert with erfc = 1 − erf:
            //
            //   K_xx = (a + b)·(1/r)  +  (−b)·(erf(ω·r)/r)
            //
            // giving cam_alpha = a + b (the always-on / short-range
            // limit HF fraction) and cam_beta = −b (the long-range
            // addition switched on by erf).
            double omega = 0.0, cam_a = 0.0, cam_b = 0.0;
            xc_hyb_cam_coef(&impl_->funcs[i], &omega, &cam_a, &cam_b);
            is_range_separated_ = true;
            // ω is a property of the range-separation, the same for
            // every component of a composite; the last one wins (all
            // agree for any sane mix).
            rsh_omega_ = omega;
            cam_alpha_ += w * (cam_a + cam_b);
            cam_beta_  += w * (-cam_b);
            hf_exchange_fraction_ += w * (cam_a + cam_b);
        } else {
            const double alpha = xc_hyb_exx_coef(&impl_->funcs[i]);
            hf_exchange_fraction_ += w * alpha;
        }
    }
    if (any_mgga)      kind_ = XCKind::MGGA;
    else if (any_gga)  kind_ = XCKind::GGA;
    else               kind_ = XCKind::LDA;

    // For a non-range-separated functional, cam_alpha mirrors the
    // single HF fraction and cam_beta is zero — so SCF drivers can use
    // (cam_alpha, cam_beta) uniformly and the global-hybrid path falls
    // out as the cam_beta == 0 special case.
    if (!is_range_separated_) {
        cam_alpha_ = hf_exchange_fraction_;
        cam_beta_  = 0.0;
    }
}

Functional::~Functional() = default;

bool Functional::is_external() const {
    return static_cast<bool>(impl_->external_provider);
}

const ExternalXCCapabilities& Functional::external_capabilities() const {
    if (!impl_->external_provider) {
        throw std::runtime_error(
            "Functional::external_capabilities called on a libxc functional");
    }
    return impl_->external_capabilities;
}

ExternalXCEvaluation Functional::eval_external(
    const ExternalXCInput& input) const {
    if (!impl_->external_provider) {
        throw std::runtime_error(
            "Functional::eval_external called on a libxc functional");
    }
    const std::string& required =
        impl_->external_capabilities.required_grid_profile;
    if (!required.empty() && input.grid_profile != required) {
        const std::string actual = input.grid_profile.empty()
            ? std::string{"<missing>"} : input.grid_profile;
        throw std::invalid_argument(
            "External XC functional '" + name_ + "' requires grid profile '"
            + required + "', but the supplied Grid reports '" + actual
            + "'. Rebuild the grid with the required atomic_grid_profile.");
    }
    return impl_->external_provider->evaluate(input);
}

void Functional::eval_unpolarised(const Eigen::VectorXd& rho,
                                  const Eigen::VectorXd& sigma,
                                  Eigen::VectorXd& exc,
                                  Eigen::VectorXd& v_rho,
                                  Eigen::VectorXd& v_sigma) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC functionals require full-grid evaluation; "
            "pointwise eval_unpolarised is unavailable");
    }
    if (impl_->spin != 1) {
        throw std::runtime_error(
            "Functional::eval_unpolarised called on spin-polarized functional");
    }
    if (kind_ == XCKind::MGGA) {
        throw std::runtime_error(
            "Functional::eval_unpolarised called on a meta-GGA functional "
            "(use Functional::eval_unpolarised_mgga, which also takes τ).");
    }
    const auto n = rho.size();
    if (kind_ == XCKind::GGA && sigma.size() != n) {
        throw std::invalid_argument(
            "Functional: sigma length must equal rho length for GGA");
    }
    exc   = Eigen::VectorXd::Zero(n);
    v_rho = Eigen::VectorXd::Zero(n);
    if (kind_ == XCKind::GGA) {
        v_sigma = Eigen::VectorXd::Zero(n);
    } else {
        v_sigma.resize(0);
    }

    Eigen::VectorXd comp_exc(n), comp_vr(n), comp_vs;
    if (kind_ == XCKind::GGA) comp_vs.resize(n);

    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam == XC_FAMILY_LDA) {
            xc_lda_exc_vxc(&func, static_cast<int>(n),
                           rho.data(), comp_exc.data(), comp_vr.data());
            exc.array() += w * rho.array() * comp_exc.array();  // w·ρ·ε_xc
            v_rho += w * comp_vr;
        } else {
            // GGA or hybrid-GGA. Same interface; the hybrid's X piece is the
            // GGA, the HF exchange is injected separately into the Fock build.
            xc_gga_exc_vxc(&func, static_cast<int>(n),
                           rho.data(), sigma.data(),
                           comp_exc.data(), comp_vr.data(), comp_vs.data());
            exc.array() += w * rho.array() * comp_exc.array();
            v_rho   += w * comp_vr;
            v_sigma += w * comp_vs;
        }
    }
    // NOTE: exc is returned as the integrand ρ(r)·ε_xc(r), so that
    // E_xc = ∫ exc(r) dr = Σ_g w_g exc(r_g) directly at the grid level.
    // Callers that want the per-electron quantity can divide by ρ(r).
}

void Functional::eval_polarised(const Eigen::VectorXd& rho_a,
                                const Eigen::VectorXd& rho_b,
                                const Eigen::VectorXd& sigma_aa,
                                const Eigen::VectorXd& sigma_ab,
                                const Eigen::VectorXd& sigma_bb,
                                Eigen::VectorXd& exc,
                                Eigen::VectorXd& v_rho_a,
                                Eigen::VectorXd& v_rho_b,
                                Eigen::VectorXd& v_sigma_aa,
                                Eigen::VectorXd& v_sigma_ab,
                                Eigen::VectorXd& v_sigma_bb) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC functionals require full-grid evaluation; "
            "pointwise eval_polarised is unavailable");
    }
    if (impl_->spin != 2) {
        throw std::runtime_error(
            "Functional::eval_polarised called on unpolarised functional");
    }
    if (kind_ == XCKind::MGGA) {
        throw std::runtime_error(
            "Functional::eval_polarised called on a meta-GGA functional "
            "(use Functional::eval_polarised_mgga, which also takes τ_α / τ_β).");
    }
    const auto n = rho_a.size();
    if (rho_b.size() != n) {
        throw std::invalid_argument(
            "Functional: rho_a and rho_b must have same length");
    }
    const bool is_gga = (kind_ == XCKind::GGA);
    if (is_gga) {
        if (sigma_aa.size() != n || sigma_ab.size() != n
            || sigma_bb.size() != n) {
            throw std::invalid_argument(
                "Functional: sigma_* length must equal rho length for GGA");
        }
    }

    exc       = Eigen::VectorXd::Zero(n);
    v_rho_a   = Eigen::VectorXd::Zero(n);
    v_rho_b   = Eigen::VectorXd::Zero(n);
    if (is_gga) {
        v_sigma_aa = Eigen::VectorXd::Zero(n);
        v_sigma_ab = Eigen::VectorXd::Zero(n);
        v_sigma_bb = Eigen::VectorXd::Zero(n);
    } else {
        v_sigma_aa.resize(0);
        v_sigma_ab.resize(0);
        v_sigma_bb.resize(0);
    }

    // libxc wants rho interleaved (ρ_α, ρ_β) of length 2*n, and sigma
    // interleaved (σ_αα, σ_αβ, σ_ββ) of length 3*n. Same for the outputs.
    Eigen::VectorXd rho_int (2 * n);
    Eigen::VectorXd sigma_int;
    Eigen::VectorXd vrho_int(2 * n);
    Eigen::VectorXd vsigma_int;
    Eigen::VectorXd comp_exc(n);
    if (is_gga) {
        sigma_int.resize(3 * n);
        vsigma_int.resize(3 * n);
    }
    for (Eigen::Index g = 0; g < n; ++g) {
        rho_int(2 * g    ) = rho_a(g);
        rho_int(2 * g + 1) = rho_b(g);
        if (is_gga) {
            sigma_int(3 * g    ) = sigma_aa(g);
            sigma_int(3 * g + 1) = sigma_ab(g);
            sigma_int(3 * g + 2) = sigma_bb(g);
        }
    }

    const Eigen::VectorXd rho_total = rho_a + rho_b;
    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam == XC_FAMILY_LDA) {
            xc_lda_exc_vxc(&func, static_cast<int>(n),
                           rho_int.data(), comp_exc.data(), vrho_int.data());
        } else {
            xc_gga_exc_vxc(&func, static_cast<int>(n),
                           rho_int.data(), sigma_int.data(),
                           comp_exc.data(), vrho_int.data(),
                           vsigma_int.data());
        }
        exc.array() += w * rho_total.array() * comp_exc.array();
        for (Eigen::Index g = 0; g < n; ++g) {
            v_rho_a(g) += w * vrho_int(2 * g    );
            v_rho_b(g) += w * vrho_int(2 * g + 1);
        }
        if (fam != XC_FAMILY_LDA) {
            for (Eigen::Index g = 0; g < n; ++g) {
                v_sigma_aa(g) += w * vsigma_int(3 * g    );
                v_sigma_ab(g) += w * vsigma_int(3 * g + 1);
                v_sigma_bb(g) += w * vsigma_int(3 * g + 2);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Phase 17d — XC kernel (second derivatives) for analytic KS Hessian.
// ----------------------------------------------------------------------

void Functional::eval_unpolarised_fxc(const Eigen::VectorXd& rho,
                                        const Eigen::VectorXd& sigma,
                                        Eigen::VectorXd& v2rho2,
                                        Eigen::VectorXd& v2rhosigma,
                                        Eigen::VectorXd& v2sigma2) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC second derivatives are not available; use an "
            "ordinary first-order SCF algorithm");
    }
    if (impl_->spin != 1) {
        throw std::runtime_error(
            "Functional::eval_unpolarised_fxc called on spin-polarized "
            "functional");
    }
    if (kind_ == XCKind::MGGA) {
        throw std::runtime_error(
            "Functional::eval_unpolarised_fxc: meta-GGA second derivatives "
            "(needed for analytic KS Hessian / CPKS on MGGA functionals) "
            "are not yet plumbed. SCF is fine; the analytic Hessian / "
            "linear-response path is restricted to LDA/GGA/hybrid-GGA.");
    }
    const auto n = rho.size();
    if (kind_ == XCKind::GGA && sigma.size() != n) {
        throw std::invalid_argument(
            "Functional: sigma length must equal rho length for GGA");
    }
    v2rho2 = Eigen::VectorXd::Zero(n);
    if (kind_ == XCKind::GGA) {
        v2rhosigma = Eigen::VectorXd::Zero(n);
        v2sigma2 = Eigen::VectorXd::Zero(n);
    } else {
        v2rhosigma.resize(0);
        v2sigma2.resize(0);
    }

    Eigen::VectorXd comp_v2rho2(n);
    Eigen::VectorXd comp_v2rhosigma, comp_v2sigma2;
    if (kind_ == XCKind::GGA) {
        comp_v2rhosigma.resize(n);
        comp_v2sigma2.resize(n);
    }

    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam == XC_FAMILY_LDA) {
            xc_lda_fxc(&func, static_cast<int>(n),
                       rho.data(), comp_v2rho2.data());
            v2rho2 += w * comp_v2rho2;
        } else {
            // GGA / hybrid-GGA: returns (v2rho2, v2rhosigma, v2sigma2).
            xc_gga_fxc(&func, static_cast<int>(n),
                       rho.data(), sigma.data(),
                       comp_v2rho2.data(), comp_v2rhosigma.data(),
                       comp_v2sigma2.data());
            v2rho2 += w * comp_v2rho2;
            v2rhosigma += w * comp_v2rhosigma;
            v2sigma2 += w * comp_v2sigma2;
        }
    }
}

void Functional::eval_polarised_lda_fxc(const Eigen::VectorXd& rho_a,
                                          const Eigen::VectorXd& rho_b,
                                          Eigen::VectorXd& v2rho2_aa,
                                          Eigen::VectorXd& v2rho2_ab,
                                          Eigen::VectorXd& v2rho2_bb) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC second derivatives are not available; use an "
            "ordinary first-order SCF algorithm");
    }
    if (impl_->spin != 2) {
        throw std::runtime_error(
            "Functional::eval_polarised_lda_fxc called on unpolarised functional");
    }
    if (kind_ != XCKind::LDA) {
        throw std::runtime_error(
            "Functional::eval_polarised_lda_fxc only supports LDA functionals "
            "(GGA second-derivatives have 6 σ-coupled pieces; not yet implemented).");
    }
    const auto n = rho_a.size();
    if (rho_b.size() != n) {
        throw std::invalid_argument(
            "Functional: rho_a and rho_b must have same length");
    }
    v2rho2_aa = Eigen::VectorXd::Zero(n);
    v2rho2_ab = Eigen::VectorXd::Zero(n);
    v2rho2_bb = Eigen::VectorXd::Zero(n);

    // libxc spin-polarized LDA fxc: rho interleaved (ρ_α, ρ_β); v2rho2
    // interleaved (αα, αβ, ββ).
    Eigen::VectorXd rho_int(2 * n);
    Eigen::VectorXd v2_int(3 * n);
    for (Eigen::Index g = 0; g < n; ++g) {
        rho_int(2 * g    ) = rho_a(g);
        rho_int(2 * g + 1) = rho_b(g);
    }

    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam != XC_FAMILY_LDA) {
            throw std::runtime_error(
                "eval_polarised_lda_fxc: non-LDA functional in mix; "
                "use eval_polarised_gga_fxc instead.");
        }
        xc_lda_fxc(&func, static_cast<int>(n),
                   rho_int.data(), v2_int.data());
        for (Eigen::Index g = 0; g < n; ++g) {
            v2rho2_aa(g) += w * v2_int(3 * g    );
            v2rho2_ab(g) += w * v2_int(3 * g + 1);
            v2rho2_bb(g) += w * v2_int(3 * g + 2);
        }
    }
}

void Functional::eval_polarised_gga_fxc(const Eigen::VectorXd& rho_a,
                                          const Eigen::VectorXd& rho_b,
                                          const Eigen::VectorXd& sigma_aa,
                                          const Eigen::VectorXd& sigma_ab,
                                          const Eigen::VectorXd& sigma_bb,
                                          PolarisedGGAFxc& out) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC second derivatives are not available; use an "
            "ordinary first-order SCF algorithm");
    }
    if (impl_->spin != 2) {
        throw std::runtime_error(
            "Functional::eval_polarised_gga_fxc called on unpolarised "
            "functional");
    }
    if (kind_ == XCKind::MGGA) {
        throw std::runtime_error(
            "Functional::eval_polarised_gga_fxc: meta-GGA second "
            "derivatives (τ-dependent) are not yet plumbed. UKS second-"
            "order SCF / linear response is restricted to LDA / GGA / "
            "hybrid-GGA.");
    }
    const auto n = rho_a.size();
    if (rho_b.size() != n) {
        throw std::invalid_argument(
            "Functional: rho_a and rho_b must have same length");
    }
    const bool is_gga = (kind_ == XCKind::GGA);
    if (is_gga) {
        if (sigma_aa.size() != n || sigma_ab.size() != n
            || sigma_bb.size() != n) {
            throw std::invalid_argument(
                "Functional: sigma_* length must equal rho length for "
                "polarised GGA fxc");
        }
    }

    // All 15 pieces start at zero — composite components accumulate.
    out.v2rho2_aa = Eigen::VectorXd::Zero(n);
    out.v2rho2_ab = Eigen::VectorXd::Zero(n);
    out.v2rho2_bb = Eigen::VectorXd::Zero(n);
    out.v2rhosigma_a_aa = Eigen::VectorXd::Zero(n);
    out.v2rhosigma_a_ab = Eigen::VectorXd::Zero(n);
    out.v2rhosigma_a_bb = Eigen::VectorXd::Zero(n);
    out.v2rhosigma_b_aa = Eigen::VectorXd::Zero(n);
    out.v2rhosigma_b_ab = Eigen::VectorXd::Zero(n);
    out.v2rhosigma_b_bb = Eigen::VectorXd::Zero(n);
    out.v2sigma2_aa_aa = Eigen::VectorXd::Zero(n);
    out.v2sigma2_aa_ab = Eigen::VectorXd::Zero(n);
    out.v2sigma2_aa_bb = Eigen::VectorXd::Zero(n);
    out.v2sigma2_ab_ab = Eigen::VectorXd::Zero(n);
    out.v2sigma2_ab_bb = Eigen::VectorXd::Zero(n);
    out.v2sigma2_bb_bb = Eigen::VectorXd::Zero(n);

    // libxc spin-polarised layout:
    //   rho        interleaved 2n  (ρ_a, ρ_b)
    //   sigma      interleaved 3n  (σ_aa, σ_ab, σ_bb)
    //   v2rho2          3n         (aa, ab, bb)
    //   v2rhosigma      6n         (a·aa, a·ab, a·bb, b·aa, b·ab, b·bb)
    //   v2sigma2        6n         (aa·aa, aa·ab, aa·bb, ab·ab, ab·bb,
    //                               bb·bb)
    Eigen::VectorXd rho_int(2 * n);
    for (Eigen::Index g = 0; g < n; ++g) {
        rho_int(2 * g    ) = rho_a(g);
        rho_int(2 * g + 1) = rho_b(g);
    }
    Eigen::VectorXd sigma_int;
    if (is_gga) {
        sigma_int.resize(3 * n);
        for (Eigen::Index g = 0; g < n; ++g) {
            sigma_int(3 * g    ) = sigma_aa(g);
            sigma_int(3 * g + 1) = sigma_ab(g);
            sigma_int(3 * g + 2) = sigma_bb(g);
        }
    }

    Eigen::VectorXd v2rho2_int(3 * n);
    Eigen::VectorXd v2rhosigma_int, v2sigma2_int;
    if (is_gga) {
        v2rhosigma_int.resize(6 * n);
        v2sigma2_int.resize(6 * n);
    }

    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam == XC_FAMILY_LDA) {
            // An LDA component (e.g. VWN5 correlation inside a hybrid):
            // only the ρ-ρ block is non-zero.
            xc_lda_fxc(&func, static_cast<int>(n),
                       rho_int.data(), v2rho2_int.data());
            for (Eigen::Index g = 0; g < n; ++g) {
                out.v2rho2_aa(g) += w * v2rho2_int(3 * g    );
                out.v2rho2_ab(g) += w * v2rho2_int(3 * g + 1);
                out.v2rho2_bb(g) += w * v2rho2_int(3 * g + 2);
            }
        } else {
            xc_gga_fxc(&func, static_cast<int>(n),
                       rho_int.data(), sigma_int.data(),
                       v2rho2_int.data(), v2rhosigma_int.data(),
                       v2sigma2_int.data());
            for (Eigen::Index g = 0; g < n; ++g) {
                out.v2rho2_aa(g) += w * v2rho2_int(3 * g    );
                out.v2rho2_ab(g) += w * v2rho2_int(3 * g + 1);
                out.v2rho2_bb(g) += w * v2rho2_int(3 * g + 2);

                out.v2rhosigma_a_aa(g) += w * v2rhosigma_int(6 * g    );
                out.v2rhosigma_a_ab(g) += w * v2rhosigma_int(6 * g + 1);
                out.v2rhosigma_a_bb(g) += w * v2rhosigma_int(6 * g + 2);
                out.v2rhosigma_b_aa(g) += w * v2rhosigma_int(6 * g + 3);
                out.v2rhosigma_b_ab(g) += w * v2rhosigma_int(6 * g + 4);
                out.v2rhosigma_b_bb(g) += w * v2rhosigma_int(6 * g + 5);

                out.v2sigma2_aa_aa(g) += w * v2sigma2_int(6 * g    );
                out.v2sigma2_aa_ab(g) += w * v2sigma2_int(6 * g + 1);
                out.v2sigma2_aa_bb(g) += w * v2sigma2_int(6 * g + 2);
                out.v2sigma2_ab_ab(g) += w * v2sigma2_int(6 * g + 3);
                out.v2sigma2_ab_bb(g) += w * v2sigma2_int(6 * g + 4);
                out.v2sigma2_bb_bb(g) += w * v2sigma2_int(6 * g + 5);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Meta-GGA evaluation (τ-dependent).
//
// libxc's xc_mgga_exc_vxc signature for spin = 1 (unpolarised):
//   inputs : rho[n], sigma[n], lapl[n], tau[n]
//   outputs: zk[n], vrho[n], vsigma[n], vlapl[n], vtau[n]
//
// For spin = 2 (polarised), arrays are interleaved per libxc convention:
//   rho  : 2n  (α, β)
//   sigma: 3n  (αα, αβ, ββ)
//   lapl : 2n  (α, β)
//   tau  : 2n  (α, β)
//   vrho : 2n  vsigma: 3n  vlapl: 2n  vtau: 2n
//
// vibe-qc's MGGA grid populates ρ, |∇ρ|², and τ but not ∇²ρ (the
// laplacian). At construction we reject any libxc functional whose
// XC_FLAGS_NEEDS_LAPLACIAN flag is set (see Functional::Functional),
// so here we always pass a zero-filled `lapl` input and discard `vlapl`.
// ----------------------------------------------------------------------

void Functional::eval_unpolarised_mgga(const Eigen::VectorXd& rho,
                                        const Eigen::VectorXd& sigma,
                                        const Eigen::VectorXd& tau,
                                        Eigen::VectorXd& exc,
                                        Eigen::VectorXd& v_rho,
                                        Eigen::VectorXd& v_sigma,
                                        Eigen::VectorXd& v_tau) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC functionals require full-grid evaluation; "
            "pointwise eval_unpolarised_mgga is unavailable");
    }
    if (impl_->spin != 1) {
        throw std::runtime_error(
            "Functional::eval_unpolarised_mgga called on spin-polarized "
            "functional");
    }
    const auto n = rho.size();
    if (sigma.size() != n) {
        throw std::invalid_argument(
            "Functional: sigma length must equal rho length");
    }
    if (tau.size() != n) {
        throw std::invalid_argument(
            "Functional: tau length must equal rho length");
    }

    exc     = Eigen::VectorXd::Zero(n);
    v_rho   = Eigen::VectorXd::Zero(n);
    v_sigma = Eigen::VectorXd::Zero(n);
    v_tau   = Eigen::VectorXd::Zero(n);

    Eigen::VectorXd comp_exc(n), comp_vr(n), comp_vs(n), comp_vl(n), comp_vt(n);
    Eigen::VectorXd zero_lapl = Eigen::VectorXd::Zero(n);

    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam == XC_FAMILY_LDA) {
            // LDA component inside a mixed-family MGGA — contributes only
            // to exc and v_rho.
            xc_lda_exc_vxc(&func, static_cast<int>(n),
                           rho.data(), comp_exc.data(), comp_vr.data());
            exc.array() += w * rho.array() * comp_exc.array();
            v_rho       += w * comp_vr;
        } else if (fam == XC_FAMILY_GGA || fam == XC_FAMILY_HYB_GGA) {
            xc_gga_exc_vxc(&func, static_cast<int>(n),
                           rho.data(), sigma.data(),
                           comp_exc.data(), comp_vr.data(), comp_vs.data());
            exc.array() += w * rho.array() * comp_exc.array();
            v_rho       += w * comp_vr;
            v_sigma     += w * comp_vs;
        } else {
            // MGGA / hybrid-MGGA.
            xc_mgga_exc_vxc(&func, static_cast<int>(n),
                            rho.data(), sigma.data(),
                            zero_lapl.data(), tau.data(),
                            comp_exc.data(), comp_vr.data(),
                            comp_vs.data(), comp_vl.data(),
                            comp_vt.data());
            exc.array() += w * rho.array() * comp_exc.array();
            v_rho       += w * comp_vr;
            v_sigma     += w * comp_vs;
            v_tau       += w * comp_vt;
        }
    }
}

void Functional::eval_polarised_mgga(const Eigen::VectorXd& rho_a,
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
                                      Eigen::VectorXd& v_tau_b) const {
    if (is_external()) {
        throw std::runtime_error(
            "External XC functionals require full-grid evaluation; "
            "pointwise eval_polarised_mgga is unavailable");
    }
    if (impl_->spin != 2) {
        throw std::runtime_error(
            "Functional::eval_polarised_mgga called on unpolarised functional");
    }
    const auto n = rho_a.size();
    if (rho_b.size() != n
        || sigma_aa.size() != n || sigma_ab.size() != n || sigma_bb.size() != n
        || tau_a.size() != n    || tau_b.size() != n) {
        throw std::invalid_argument(
            "Functional: all rho/sigma/tau inputs must have length n_points");
    }

    exc        = Eigen::VectorXd::Zero(n);
    v_rho_a    = Eigen::VectorXd::Zero(n);
    v_rho_b    = Eigen::VectorXd::Zero(n);
    v_sigma_aa = Eigen::VectorXd::Zero(n);
    v_sigma_ab = Eigen::VectorXd::Zero(n);
    v_sigma_bb = Eigen::VectorXd::Zero(n);
    v_tau_a    = Eigen::VectorXd::Zero(n);
    v_tau_b    = Eigen::VectorXd::Zero(n);

    Eigen::VectorXd rho_int(2 * n);
    Eigen::VectorXd sigma_int(3 * n);
    Eigen::VectorXd tau_int(2 * n);
    Eigen::VectorXd lapl_int = Eigen::VectorXd::Zero(2 * n);
    Eigen::VectorXd vrho_int(2 * n);
    Eigen::VectorXd vsigma_int(3 * n);
    Eigen::VectorXd vtau_int(2 * n);
    Eigen::VectorXd vlapl_int(2 * n);  // discarded
    Eigen::VectorXd comp_exc(n);

    for (Eigen::Index g = 0; g < n; ++g) {
        rho_int(2 * g    ) = rho_a(g);
        rho_int(2 * g + 1) = rho_b(g);
        sigma_int(3 * g    ) = sigma_aa(g);
        sigma_int(3 * g + 1) = sigma_ab(g);
        sigma_int(3 * g + 2) = sigma_bb(g);
        tau_int(2 * g    ) = tau_a(g);
        tau_int(2 * g + 1) = tau_b(g);
    }

    const Eigen::VectorXd rho_total = rho_a + rho_b;
    for (std::size_t i = 0; i < impl_->funcs.size(); ++i) {
        auto& func = impl_->funcs[i];
        const double w = impl_->weights[i];
        const auto fam = func.info->family;
        if (fam == XC_FAMILY_LDA) {
            xc_lda_exc_vxc(&func, static_cast<int>(n),
                           rho_int.data(), comp_exc.data(), vrho_int.data());
            exc.array() += w * rho_total.array() * comp_exc.array();
            for (Eigen::Index g = 0; g < n; ++g) {
                v_rho_a(g) += w * vrho_int(2 * g    );
                v_rho_b(g) += w * vrho_int(2 * g + 1);
            }
        } else if (fam == XC_FAMILY_GGA || fam == XC_FAMILY_HYB_GGA) {
            xc_gga_exc_vxc(&func, static_cast<int>(n),
                           rho_int.data(), sigma_int.data(),
                           comp_exc.data(), vrho_int.data(),
                           vsigma_int.data());
            exc.array() += w * rho_total.array() * comp_exc.array();
            for (Eigen::Index g = 0; g < n; ++g) {
                v_rho_a(g)    += w * vrho_int(2 * g    );
                v_rho_b(g)    += w * vrho_int(2 * g + 1);
                v_sigma_aa(g) += w * vsigma_int(3 * g    );
                v_sigma_ab(g) += w * vsigma_int(3 * g + 1);
                v_sigma_bb(g) += w * vsigma_int(3 * g + 2);
            }
        } else {
            xc_mgga_exc_vxc(&func, static_cast<int>(n),
                            rho_int.data(), sigma_int.data(),
                            lapl_int.data(), tau_int.data(),
                            comp_exc.data(), vrho_int.data(),
                            vsigma_int.data(), vlapl_int.data(),
                            vtau_int.data());
            exc.array() += w * rho_total.array() * comp_exc.array();
            for (Eigen::Index g = 0; g < n; ++g) {
                v_rho_a(g)    += w * vrho_int(2 * g    );
                v_rho_b(g)    += w * vrho_int(2 * g + 1);
                v_sigma_aa(g) += w * vsigma_int(3 * g    );
                v_sigma_ab(g) += w * vsigma_int(3 * g + 1);
                v_sigma_bb(g) += w * vsigma_int(3 * g + 2);
                v_tau_a(g)    += w * vtau_int(2 * g    );
                v_tau_b(g)    += w * vtau_int(2 * g + 1);
            }
        }
    }
}

}  // namespace vibeqc
