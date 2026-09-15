#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace vibeqc {
namespace semiempirical {
namespace xtb {

namespace {

constexpr int kMinimumAtomicNumber = 1;
constexpr int kMaximumAtomicNumber = 86;

// Canonical digest measured independently from the published loader.
// Update it only with the loader and its regression oracle; names, element
// counts, and source labels never grant published provenance by themselves.
// 2026-08-27 (issue #43): re-measured after the per-angular-momentum
// kcn/poly projection fix in the TOML converter - the previous digest
// (71b83ab40d8f09081094ca6fedc18dc95e6ff3b325b23c3b70a10679e285b69e)
// hashed the rotated d-first transition-metal parameters. The identity
// string is unchanged: this is still the published 2019 set, now
// loaded faithfully (regression oracle:
// tests/test_gfn2_xtb.py::TestGFN2DFirstParameterProjection).
// 2026-08-27 (issue #446): re-attested against the publication, not
// merely re-measured. The per-l keying is the paper's own - Bannwarth,
// Ehlert & Grimme, J. Chem. Theory Comput. 15, 1652-1671 (2019),
// doi:10.1021/acs.jctc.8b01176, eq 17
// (H_kk = H^l_A - H^l_CN * CN'_A, for kappa in l in A) and eq 19
// (k^poly_{A,l}) - and the loaded per-shell values match Supporting
// Information Table S52 ("Element-specific shell parameters employed
// in GFN2-xTB"), which tabulates k^poly_{A,l}, the CN-dependent level
// enhancement, H^l_A and zeta_l per shell label. Cu is the worked
// example: SI Table S52 gives k^poly = 0.177983 / 0.149778 / -0.265089
// for 4s / 4p / 3d, and the loaded set carries those on l = 0 / 1 / 2.
// The digest moved because a mapping was corrected, not a value:
// rotating only kcn/poly back to shell-positional order on the 41
// d-first elements reproduces the previous digest exactly. The
// tblite H0 bisection that located the defect is corroboration from an
// implementation, not from the publication. Full record and the
// re-run recipe: tests/test_gfn2_parameter_identity.py at
// GFN2_PUBLISHED_SHA256, and docs/user_guide/semiempirical.md
// ("Published-parameter identity").
constexpr const char* kGFN2XTB2019Sha256 =
    "0b3c70a5a7a8dec49953a59e72e8709a9b6e9a58ec55fb063f7d08a3067cc8ef";
constexpr const char* kGFN2XTB2019Identity =
    "published:gfn2-xtb-2019-v1";

void validate_atomic_number(int Z) {
    if (Z < kMinimumAtomicNumber || Z > kMaximumAtomicNumber) {
        throw std::invalid_argument(
            "GFN2-xTB atomic number must be in [1, 86]");
    }
}

void validate_element(const GFN2ElementData& elem) {
    validate_atomic_number(elem.Z);
    if (elem.shells.empty()) {
        throw std::invalid_argument(
            "GFN2-xTB element parameters require at least one shell");
    }
    const double scalar_fields[] = {
        elem.gam, elem.gam3, elem.alpha,
        elem.dpol, elem.qpol, elem.mp_rad, elem.mp_vcn,
    };
    for (double value : scalar_fields) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "GFN2-xTB element parameters must be finite");
        }
    }
    if (elem.gam <= 0.0 || elem.alpha <= 0.0
        || elem.mp_rad < 0.0 || elem.mp_vcn < 0.0) {
        throw std::invalid_argument(
            "GFN2-xTB element hardness and charge scaling must be positive, "
            "and multipole radius/CN parameters must be non-negative");
    }

    std::set<int> angular_momenta;
    std::set<std::pair<int, int>> shell_keys;
    for (const auto& shell : elem.shells) {
        if (shell.n < 1 || shell.n > 7
            || shell.l < 0 || shell.l > 3
            || shell.n <= shell.l) {
            throw std::invalid_argument(
                "GFN2-xTB shell quantum numbers require 1 <= n <= 7, "
                "0 <= l <= 3, and n > l");
        }
        if (!angular_momenta.insert(shell.l).second
            || !shell_keys.emplace(shell.n, shell.l).second) {
            throw std::invalid_argument(
                "GFN2-xTB element parameters reject duplicate shell "
                "angular-momentum channels");
        }
        const double shell_fields[] = {
            shell.en, shell.zeta, shell.k_en, shell.kcn, shell.poly,
        };
        for (double value : shell_fields) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "GFN2-xTB shell parameters must be finite");
            }
        }
        if (shell.zeta <= 0.0 || shell.k_en <= 0.0) {
            throw std::invalid_argument(
                "GFN2-xTB shell zeta and electronegativity scaling must be "
                "positive");
        }
    }
}

CoreElementData regenerate_core_element(const GFN2ElementData& elem) {
    CoreElementData base;
    base.Z = elem.Z;
    base.hubbard_u = elem.gam;
    // Every consumer of these arrays indexes them BY ANGULAR MOMENTUM
    // (SemiempiricalBasis::build_shell_info reads elem.zeta[l] /
    // elem.on_site[l]; CoreParameterSet::on_site(Z, l) / zeta(Z, l)
    // likewise). The xtb parameter file lists shells in occupation
    // order, which is d-FIRST for the transition-metal blocks
    // (ao=3d4s4p: Z 21-29, 39-47, 57-79), so a positional copy hands
    // the 4s shell the 3d exponent and level (the issue #43 Cu2 H0
    // parity defect, +36 to +140 mHa). Project by l instead; absent l
    // slots stay 0.0, which both build paths already treat as
    // shell-not-present.
    int max_l = -1;
    for (const auto& shell : elem.shells) {
        max_l = std::max(max_l, shell.l);
    }
    base.on_site.assign(static_cast<std::size_t>(max_l + 1), 0.0);
    base.zeta.assign(static_cast<std::size_t>(max_l + 1), 0.0);
    for (const auto& shell : elem.shells) {
        const auto slot = static_cast<std::size_t>(shell.l);
        if (base.zeta[slot] != 0.0) {
            throw std::logic_error(
                "GFN2-xTB element Z=" + std::to_string(elem.Z)
                + " carries two shells with angular momentum l="
                + std::to_string(shell.l)
                + "; the l-projected core arrays cannot represent that");
        }
        base.on_site[slot] = shell.en;
        base.zeta[slot] = shell.zeta;
        base.valence_electrons += 2 * (2 * shell.l + 1);
    }
    return base;
}

void validate_core_projection(const GFN2ElementData& elem) {
    const CoreElementData expected = regenerate_core_element(elem);
    const CoreElementData& actual = elem.base;
    if (actual.Z != expected.Z
        || actual.valence_electrons != expected.valence_electrons
        || actual.hubbard_u != expected.hubbard_u
        || actual.on_site != expected.on_site
        || actual.zeta != expected.zeta) {
        throw std::logic_error(
            "GFN2-xTB derived CoreElementData is inconsistent with its "
            "authoritative element record");
    }
}

void validate_repulsive_pair(const GFN2RepulsivePair& pair) {
    if (!std::isfinite(pair.alpha) || !std::isfinite(pair.k_ab)) {
        throw std::invalid_argument(
            "GFN2-xTB repulsive-pair parameters must be finite");
    }
    if (pair.alpha <= 0.0 || pair.k_ab < 0.0) {
        throw std::invalid_argument(
            "GFN2-xTB repulsive-pair alpha must be positive and k_ab must "
            "be non-negative");
    }
}

void append_element_identity(
    detail::CanonicalParameterHasher& hasher,
    const GFN2ElementData& elem) {
    hasher.add_int(elem.Z);
    hasher.add_size(elem.shells.size());
    for (const auto& shell : elem.shells) {
        // Shell order is execution-semantic: CoreElementData vectors are
        // currently regenerated in this exact order.
        hasher.add_int(shell.n);
        hasher.add_int(shell.l);
        hasher.add_double(shell.en);
        hasher.add_double(shell.zeta);
        hasher.add_double(shell.k_en);
        hasher.add_double(shell.kcn);
        hasher.add_double(shell.poly);
    }
    hasher.add_double(elem.gam);
    hasher.add_double(elem.gam3);
    hasher.add_double(elem.alpha);
    hasher.add_double(elem.dpol);
    hasher.add_double(elem.qpol);
    hasher.add_double(elem.mp_rad);
    hasher.add_double(elem.mp_vcn);
}

std::string parameter_identity_for_hash(const std::string& sha256) {
    if (sha256 == kGFN2XTB2019Sha256) return kGFN2XTB2019Identity;
    return detail::custom_parameter_identity(sha256);
}

}  // namespace

GFN2ParameterSet::GFN2ParameterSet() = default;

bool GFN2ParameterSet::has_element(int Z) const {
    return elements_.find(Z) != elements_.end();
}

const CoreElementData& GFN2ParameterSet::element(int Z) const {
    auto it = elements_.find(Z);
    if (it == elements_.end()) {
        throw std::runtime_error(
            "GFN2ParameterSet: element Z=" + std::to_string(Z)
            + " not found");
    }
    return it->second.base;
}

int GFN2ParameterSet::shell_principal_quantum_number(int Z, int l) const {
    const GFN2ElementData* e = element_data(Z);
    if (e == nullptr) {
        throw std::runtime_error(
            "GFN2ParameterSet: element Z=" + std::to_string(Z)
            + " not found for shell n query");
    }
    for (const auto& sh : e->shells) {
        if (sh.l == l) return sh.n;
    }
    // No shell of this l: fall back to the historical convention.
    return l + 1;
}

double GFN2ParameterSet::repulsive_energy(int Z1, int Z2, double R) const {
    if (R < 1e-12) return 0.0;
    const auto* rp = repulsive_pair(Z1, Z2);
    if (rp == nullptr) return 0.0;
    // GFN2-xTB repulsion (Bannwarth, Ehlert & Grimme, JCTC 2019, Eq. 9):
    //   E_rep(R) = (Z_Aeff * Z_Beff / R) * exp(-sqrt(alpha_A*alpha_B) * R^krep).
    // krep is 1.0 for H/He pairs and 3/2 otherwise.
    // The pair fields carry the *combined* parameters:
    //   k_ab = Zᴬ_eff · Zᴮ_eff      alpha = √(αᴬ · αᴮ)
    // The R^krep in the exponent is what makes the repulsion decay. The
    // pre-fix form k_ab·exp(−α·R) (with a ~0.04 exponent) decayed far too
    // slowly — still ~0.08 Ha per H–H pair at 40 bohr — which broke
    // size-consistency (two non-interacting fragments did not sum).
    const double k_f = (Z1 <= 2 && Z2 <= 2) ? 1.0 : 1.5;
    double damp = std::exp(-rp->alpha * std::pow(R, k_f));
    double e_rep = rp->k_ab / R * damp;

    // This published repulsion is correct as written — do NOT add a short-range
    // wall here.  The historical GFN2 water-PES collapse was NOT a repulsion
    // problem (and the basis is valence-only, not all-electron): it was an SCC
    // bug in gfn2_driver.cpp — a wrong-sign, shell-resolved third-order term plus
    // a double-counted band-energy assembly — and was fixed there.  With the bare
    // H0 plus this repulsion the water O–H PES already has its minimum at the
    // correct bond length; an empirical wall would only distort it.
    return e_rep;
}

double GFN2ParameterSet::repulsive_derivative(int Z1, int Z2, double R) const {
    if (R < 1e-12) return 0.0;
    const auto* rp = repulsive_pair(Z1, Z2);
    if (rp == nullptr) return 0.0;
    // dE/dR = -k_ab*damp*(1/R^2 + krep*alpha*R^(krep-2))
    const double k_f = (Z1 <= 2 && Z2 <= 2) ? 1.0 : 1.5;
    double damp = std::exp(-rp->alpha * std::pow(R, k_f));
    return -rp->k_ab * damp
        * (1.0 / (R * R) + k_f * rp->alpha * std::pow(R, k_f - 2.0));
}

int GFN2ParameterSet::n_elements() const {
    return static_cast<int>(elements_.size());
}

const GFN2ElementData* GFN2ParameterSet::element_data(int Z) const {
    auto it = elements_.find(Z);
    return (it != elements_.end()) ? &it->second : nullptr;
}

const GFN2RepulsivePair* GFN2ParameterSet::repulsive_pair(
    int Z1, int Z2) const {
    auto it = repulsive_pairs_.find(pair_key(Z1, Z2));
    return (it != repulsive_pairs_.end()) ? &it->second : nullptr;
}

void GFN2ParameterSet::add_element(const GFN2ElementData& elem) {
    validate_element(elem);
    elements_[elem.Z] = elem;
    elements_[elem.Z].base = regenerate_core_element(elements_[elem.Z]);
}

void GFN2ParameterSet::set_repulsive_pair(
    int Z1, int Z2, const GFN2RepulsivePair& rp) {
    validate_atomic_number(Z1);
    validate_atomic_number(Z2);
    validate_repulsive_pair(rp);
    repulsive_pairs_[pair_key(Z1, Z2)] = rp;
}

ParameterSetMetadata GFN2ParameterSet::metadata() const {
    ParameterSetMetadata m;
    m.method_name = "gfn2-xtb";
    m.version = version_;
    m.n_elements = n_elements();
    m.element_list.reserve(elements_.size());
    for (const auto& entry : elements_) {
        m.element_list.push_back(entry.first);
    }
    m.parameter_hash = content_sha256();
    const std::string identity = parameter_identity_for_hash(m.parameter_hash);
    if (identity == kGFN2XTB2019Identity) {
        m.origin = "published";
        m.license = "LGPL-3.0 (original parameter set)";
        m.doi_or_url = "10.1021/acs.jctc.8b01176";
    } else {
        m.origin = identity;
    }
    return m;
}

std::string GFN2ParameterSet::content_sha256() const {
    detail::CanonicalParameterHasher hasher("xtb.gfn2.v1");
    hasher.add_string(method_name());
    hasher.add_string(version_);

    hasher.add_size(elements_.size());
    for (const auto& entry : elements_) {
        if (entry.first != entry.second.Z) {
            throw std::logic_error(
                "GFN2-xTB element-map key disagrees with authoritative Z");
        }
        validate_element(entry.second);
        validate_core_projection(entry.second);
        append_element_identity(hasher, entry.second);
    }

    hasher.add_size(repulsive_pairs_.size());
    for (const auto& entry : repulsive_pairs_) {
        const int Z1 = entry.first / 1000;
        const int Z2 = entry.first % 1000;
        validate_atomic_number(Z1);
        validate_atomic_number(Z2);
        if (Z1 > Z2 || pair_key(Z1, Z2) != entry.first) {
            throw std::logic_error(
                "GFN2-xTB repulsive-pair map contains a non-canonical key");
        }
        validate_repulsive_pair(entry.second);
        hasher.add_int(Z1);
        hasher.add_int(Z2);
        hasher.add_double(entry.second.alpha);
        hasher.add_double(entry.second.k_ab);
    }

    // d4_* are intentionally excluded.  They are compatibility plumbing,
    // not inputs to any native GFN2 energy or derivative.  The molecular
    // runner executes the independently versioned Python ``gfn2xtb`` D4 row
    // and records that row's own frozen provenance.  Including these dormant
    // members would make this digest claim a change in executed native
    // parameters when the numerical model was unchanged.
    return hasher.finish();
}

std::string GFN2ParameterSet::parameter_identity() const {
    return parameter_identity_for_hash(content_sha256());
}

int GFN2ParameterSet::pair_key(int Z1, int Z2) const {
    validate_atomic_number(Z1);
    validate_atomic_number(Z2);
    int lo = std::min(Z1, Z2);
    int hi = std::max(Z1, Z2);
    return lo * 1000 + hi;
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
