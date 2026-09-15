#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vibeqc {
namespace semiempirical {
namespace nddo {

namespace {

// Canonical loader digests measured from the rebuilt native extension.
// Update them only with the loaders and regression oracles; element count,
// construction path, and source-file name never confer published provenance.
// Both allowlisted PM6 digests are MOPAC-derived (#440). The five-element
// inline set in python/vibeqc/semiempirical/methods/pm6_params.py was copied
// from MOPAC's Apache-2.0 parameters_for_PM6_C.F90, not transcribed from
// Stewart's published tables: all 118 shipped values (88 element + 30
// diatomic) reproduce that file exactly at its six published decimals, while
// none of them occurs anywhere in Stewart, J. Mol. Model. 13, 1173 (2007).
// The DOI below therefore names the *method*; the license field names the
// terms under which the numbers are redistributed, and those are Apache-2.0
// for both sets.
constexpr const char* kPM6MopacInlineSha256 =
    "b29a0554f7c311eea66dfa5b2368ccf75a20a7ae78bed548ef685a1ec43d4273";
constexpr const char* kPM6MopacFullSha256 =
    "105bb194eff3f4902758ea642839b3f296839ea87a9ccb8217b2ffe53c1e2d57";

constexpr const char* kPM6MopacInlineIdentity =
    "published:pm6-mopac-inline-v1";
constexpr const char* kPM6MopacFullIdentity =
    "published:pm6-mopac-full-v1";

// PM7 (#440, maintainer decision 2026-09-06): both shipped PM7 sets are
// likewise MOPAC-derived. The inline five-element set in
// python/vibeqc/semiempirical/methods/pm7_params.py reproduces MOPAC's
// Apache-2.0 parameters_for_PM7_C.F90 exactly: 65 element values from its
// data statements and 30 diatomic alpb/xfac values from its
// alpb_and_xfac_pm7 subroutine. The 28 pair values of unrecorded origin
// that shipped before 2026-09-06 were replaced by MOPAC's, which is why the
// inline digest below is not the one the pre-fix loader produced. The full
// set is the bundled pm7_mopac_params.toml cache. These identities describe
// parameter provenance only; the PM7 Hamiltonian stays gated (see
// repulsive_energy below and methods/pm7.py) until its feathered
// electrostatics are implemented.
constexpr const char* kPM7MopacInlineSha256 =
    "7e51f94bdc4ff7a0bd5b086ff390b14b97c1bf97cdc54ec6ab8e3d36f2525191";
constexpr const char* kPM7MopacFullSha256 =
    "ee028a756f7bc26a559796c97dad0f2b2f090aceb25de90403989e3d090b1359";

constexpr const char* kPM7MopacInlineIdentity =
    "published:pm7-mopac-inline-v1";
constexpr const char* kPM7MopacFullIdentity =
    "published:pm7-mopac-full-v1";

std::string pm6_parameter_identity_for_hash(
    const std::string& method_name,
    const std::string& sha256) {
    if (method_name == "pm6") {
        if (sha256 == kPM6MopacInlineSha256) {
            return kPM6MopacInlineIdentity;
        }
        if (sha256 == kPM6MopacFullSha256) {
            return kPM6MopacFullIdentity;
        }
    }
    if (method_name == "pm7") {
        if (sha256 == kPM7MopacInlineSha256) {
            return kPM7MopacInlineIdentity;
        }
        if (sha256 == kPM7MopacFullSha256) {
            return kPM7MopacFullIdentity;
        }
    }
    return detail::custom_parameter_identity(sha256);
}

void validate_finite_nddo_element(const NDDOElementData& elem) {
    const double scalar_fields[] = {
        elem.uss, elem.upp, elem.udd,
        elem.gss, elem.gpp, elem.gsp, elem.gp2, elem.hsp,
        elem.zs, elem.zp, elem.zd,
        elem.zsn, elem.zpn, elem.zdn,
        elem.betas, elem.betap, elem.betad,
        elem.f0sd, elem.g2sd,
        elem.alpha, elem.pcore, elem.polvo,
        elem.cpe_zet, elem.cpe_z0, elem.cpe_b,
        elem.cpe_xlo, elem.cpe_xhi,
    };
    for (double value : scalar_fields) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "NDDO element parameters must be finite");
        }
    }
    for (const auto& term : elem.gamma_terms) {
        if (!std::isfinite(term.coeff)
            || !std::isfinite(term.exponent)
            || !std::isfinite(term.factor)) {
            throw std::invalid_argument(
                "NDDO gamma terms must be finite");
        }
    }
    for (const auto& term : elem.gaussian_terms) {
        if (!std::isfinite(term.coeff)
            || !std::isfinite(term.exponent)
            || !std::isfinite(term.factor)) {
            throw std::invalid_argument(
                "NDDO Gaussian terms must be finite");
        }
    }
    if (elem.n_orbitals < 1 || elem.n_orbitals > 9) {
        throw std::invalid_argument(
            "NDDO element orbital count must be in [1, 9]");
    }
}

void append_nddo_element_identity(
    detail::CanonicalParameterHasher& hasher,
    const NDDOElementData& elem) {
    hasher.add_int(elem.Z);

    hasher.add_double(elem.uss);
    hasher.add_double(elem.upp);
    hasher.add_double(elem.udd);

    hasher.add_double(elem.gss);
    hasher.add_double(elem.gpp);
    hasher.add_double(elem.gsp);
    hasher.add_double(elem.gp2);
    hasher.add_double(elem.hsp);

    hasher.add_double(elem.zs);
    hasher.add_double(elem.zp);
    hasher.add_double(elem.zd);
    hasher.add_double(elem.zsn);
    hasher.add_double(elem.zpn);
    hasher.add_double(elem.zdn);

    hasher.add_double(elem.betas);
    hasher.add_double(elem.betap);
    hasher.add_double(elem.betad);
    hasher.add_double(elem.f0sd);
    hasher.add_double(elem.g2sd);

    hasher.add_double(elem.alpha);
    hasher.add_double(elem.pcore);
    hasher.add_double(elem.polvo);
    hasher.add_double(elem.cpe_zet);
    hasher.add_double(elem.cpe_z0);
    hasher.add_double(elem.cpe_b);
    hasher.add_double(elem.cpe_xlo);
    hasher.add_double(elem.cpe_xhi);
    hasher.add_bool(elem.has_cpe);

    hasher.add_size(elem.gamma_terms.size());
    for (const auto& term : elem.gamma_terms) {
        hasher.add_double(term.coeff);
        hasher.add_double(term.exponent);
        hasher.add_double(term.factor);
    }

    hasher.add_size(elem.gaussian_terms.size());
    for (const auto& term : elem.gaussian_terms) {
        hasher.add_double(term.coeff);
        hasher.add_double(term.exponent);
        hasher.add_double(term.factor);
    }

    hasher.add_int(elem.n_orbitals);
    hasher.add_bool(elem.has_d);
}

}  // namespace

int pm6_core_charge(int Z) {
    // Stewart, J. Mol. Model. 13, 1173 (2007): Z_A in the PM6
    // core-core expression is the number of explicitly treated valence
    // electrons.  These branches reproduce MOPAC's neutral-atom
    // ``tore = ios + iop + iod`` convention.  Filled d/f shells are frozen
    // for main-group atoms, while open transition-metal d shells are
    // explicit.  Rare gases use the published (np,(n+1)s) six-electron
    // valence space (He is the exception).
    if (Z == 1) return 1;
    if (Z == 2) return 2;
    if (Z >= 3 && Z <= 9) return Z - 2;
    if (Z == 10) return 6;
    if (Z >= 11 && Z <= 17) return Z - 10;
    if (Z == 18) return 6;
    if (Z >= 19 && Z <= 20) return Z - 18;
    if (Z >= 21 && Z <= 29) return Z - 18;
    if (Z == 30) return 2;
    if (Z >= 31 && Z <= 35) return Z - 28;
    if (Z == 36) return 6;
    if (Z >= 37 && Z <= 38) return Z - 36;
    if (Z >= 39 && Z <= 47) return Z - 36;
    if (Z == 48) return 2;
    if (Z >= 49 && Z <= 53) return Z - 46;
    if (Z == 54) return 6;
    if (Z >= 55 && Z <= 56) return Z - 54;
    if (Z >= 57 && Z <= 71) return 3;
    if (Z >= 72 && Z <= 79) return Z - 68;
    if (Z == 80) return 2;
    if (Z >= 81 && Z <= 85) return Z - 78;
    if (Z == 86) return 6;

    // PM6 has sparse actinide coverage.  MOPAC treats the f shell as core
    // in this range; the explicit s/p/d occupancies are therefore small.
    if (Z == 87 || Z == 88) return 1;
    if (Z == 89) return 3;
    if (Z == 90) return 4;
    if (Z >= 91 && Z <= 97) return 2;
    if (Z == 98) return 1;

    // Z=99+ entries in the upstream parameter arrays are pseudo-atoms and
    // charge markers, not chemical elements accepted by the PM6 model.
    return 0;
}

bool pm6_has_complete_base_parameters(const NDDOElementData& element) {
    auto finite_nonzero = [](double value) {
        return std::isfinite(value) && std::abs(value) > 1.0e-12;
    };
    return finite_nonzero(element.uss)
        && finite_nonzero(element.betas)
        && finite_nonzero(element.zs)
        && finite_nonzero(element.gss);
}

bool pm6_has_complete_sp_parameters(const NDDOElementData& element) {
    auto finite_nonzero = [](double value) {
        return std::isfinite(value) && std::abs(value) > 1.0e-12;
    };
    if (!pm6_has_complete_base_parameters(element)) return false;
    if (element.Z <= 2) return true;
    return element.n_orbitals == 4 && !element.has_d
        && finite_nonzero(element.upp)
        && finite_nonzero(element.betap)
        && finite_nonzero(element.zp)
        && finite_nonzero(element.gpp)
        && finite_nonzero(element.gsp)
        && finite_nonzero(element.gp2)
        && finite_nonzero(element.hsp);
}

PM6ParameterSet::PM6ParameterSet(std::string method_name)
    : method_name_(std::move(method_name)) {
    if (method_name_ != "pm6" && method_name_ != "pm7") {
        throw std::invalid_argument(
            "NDDO parameter method must be 'pm6' or 'pm7'");
    }
    version_ = method_name_ + "-mopac";
}

bool PM6ParameterSet::has_element(int Z) const {
    return elements_.find(Z) != elements_.end();
}

const CoreElementData& PM6ParameterSet::element(int Z) const {
    auto it = elements_.find(Z);
    if (it == elements_.end()) {
        throw std::runtime_error("PM6: element Z=" + std::to_string(Z) + " not found");
    }
    return it->second.base;
}

double PM6ParameterSet::repulsive_energy(int Z1, int Z2, double R) const {
    const std::string active_method = method_name();
    if (active_method != "pm6") {
        if (active_method != "pm7") {
            throw std::logic_error(
                active_method
                + " repulsive energy is unavailable through the PM6 core "
                  "model; use that method's Hamiltonian-specific API");
        }
        throw std::logic_error(
            "PM7 repulsive energy is unavailable until its feathered "
            "electrostatic core model is implemented");
    }
    if (!std::isfinite(R) || R <= 1.0e-12)
        throw std::invalid_argument(
            "PM6 repulsive energy requires a positive finite distance");
    const auto* e1 = element_data(Z1);
    const auto* e2 = element_data(Z2);
    if (e1 == nullptr || e2 == nullptr) {
        throw std::invalid_argument(
            "PM6 repulsive energy requires both element records");
    }
    if (Z1 < 1 || Z1 > 86 || Z2 < 1 || Z2 > 86
        || !pm6_has_complete_base_parameters(*e1)
        || !pm6_has_complete_base_parameters(*e2)) {
        throw std::invalid_argument(
            "PM6 repulsive energy requires complete executable element "
            "records");
    }
    return pm6_core_core_repulsion(
        Z1, Z2, R, *e1, *e2, diatomic(Z1, Z2));
}

double PM6ParameterSet::repulsive_derivative(int Z1, int Z2, double R) const {
    if (!std::isfinite(R) || R <= 1.0e-12)
        throw std::invalid_argument(
            "PM6 repulsive derivative requires a positive finite distance");
    const double step = std::min(
        1.0e-5 * std::max(1.0, R), 0.25 * R);
    return (
        repulsive_energy(Z1, Z2, R + step)
        - repulsive_energy(Z1, Z2, R - step)) / (2.0 * step);
}

int PM6ParameterSet::n_elements() const {
    return static_cast<int>(elements_.size());
}

const NDDOElementData* PM6ParameterSet::element_data(int Z) const {
    auto it = elements_.find(Z);
    return (it != elements_.end()) ? &it->second : nullptr;
}

const NDDODiatomicParams* PM6ParameterSet::diatomic(int Z1, int Z2) const {
    auto it = diatomic_.find(pair_key(Z1, Z2));
    return (it != diatomic_.end()) ? &it->second : nullptr;
}

void PM6ParameterSet::add_element(const NDDOElementData& elem) {
    const std::string active_method = method_name();
    if (active_method != "pm6" && active_method != "pm7") {
        throw std::logic_error(
            "PM6-family add_element cannot mutate " + active_method
            + " parameters; use that method's synchronized element API");
    }
    store_element(elem);
}

void PM6ParameterSet::store_element(const NDDOElementData& elem) {
    if (elem.Z < 1 || elem.Z > 98) {
        throw std::invalid_argument(
            "NDDO element atomic number must be in [1, 98]");
    }
    validate_finite_nddo_element(elem);
    elements_[elem.Z] = elem;
    auto& stored = elements_[elem.Z];
    stored.base.Z = elem.Z;
    stored.base.valence_electrons = pm6_core_charge(elem.Z);
    constexpr double ev_to_hartree = 1.0 / 27.2114;
    stored.base.hubbard_u = elem.gss * ev_to_hartree;
    stored.base.on_site.clear();
    stored.base.zeta.clear();
    stored.base.on_site.push_back(elem.uss * ev_to_hartree);
    stored.base.zeta.push_back(elem.zs);
    if (elem.Z > 2 && elem.n_orbitals >= 4) {
        stored.base.on_site.push_back(elem.upp * ev_to_hartree);
        stored.base.zeta.push_back(elem.zp);
    }
    if (elem.has_d && elem.n_orbitals >= 9) {
        stored.base.on_site.push_back(elem.udd * ev_to_hartree);
        stored.base.zeta.push_back(elem.zd);
    }
}

void PM6ParameterSet::set_diatomic(int Z1, int Z2, const NDDODiatomicParams& dp) {
    const std::string active_method = method_name();
    if (active_method != "pm6" && active_method != "pm7") {
        throw std::logic_error(
            "PM6-family set_diatomic cannot mutate " + active_method
            + " parameters because its core model is method-specific");
    }
    if (!std::isfinite(dp.d1) || !std::isfinite(dp.d2)) {
        throw std::invalid_argument(
            "NDDO diatomic parameters must be finite");
    }
    diatomic_[pair_key(Z1, Z2)] = dp;
}

int PM6ParameterSet::shell_principal_quantum_number(int Z, int l) const {
    int principal = 1;
    if (Z > 2) principal = 2;
    if (Z > 10) principal = 3;
    if (Z > 18) principal = 4;
    if (Z > 36) principal = 5;
    if (Z > 54) principal = 6;
    if (Z > 86) principal = 7;
    return l == 2 ? std::max(3, principal - 1) : principal;
}

ParameterSetMetadata PM6ParameterSet::metadata() const {
    ParameterSetMetadata m;
    m.method_name = method_name_;
    m.version = version_;
    m.n_elements = n_elements();
    m.element_list.reserve(elements_.size());
    for (const auto& entry : elements_) {
        m.element_list.push_back(entry.first);
    }
    m.parameter_hash = content_sha256();
    const std::string identity = pm6_parameter_identity_for_hash(
        method_name_, m.parameter_hash);
    if (identity == kPM6MopacInlineIdentity) {
        m.origin = "published";
        m.license = "Apache-2.0";
        m.doi_or_url = "10.1007/s00894-007-0233-4";
    } else if (identity == kPM6MopacFullIdentity) {
        m.origin = "published";
        m.license = "Apache-2.0";
        m.doi_or_url = "10.1007/s00894-007-0233-4";
    } else if (identity == kPM7MopacInlineIdentity
               || identity == kPM7MopacFullIdentity) {
        // Stewart, J. Mol. Model. 19, 1 (2013) names the PM7 *method*; the
        // values are MOPAC's and redistributed under Apache-2.0 (#440).
        m.origin = "published";
        m.license = "Apache-2.0";
        m.doi_or_url = "10.1007/s00894-012-1667-x";
    } else {
        m.origin = identity;
    }
    return m;
}

std::string PM6ParameterSet::content_sha256() const {
    detail::CanonicalParameterHasher hasher("nddo.pm6-family.v1");
    hasher.add_string(method_name_);

    hasher.add_size(elements_.size());
    for (const auto& entry : elements_) {
        // entry.first and elem.Z are the same authoritative key; serialize it
        // once through the record and never serialize the derived base view.
        append_nddo_element_identity(hasher, entry.second);
    }

    hasher.add_size(diatomic_.size());
    for (const auto& entry : diatomic_) {
        hasher.add_int(entry.first / 1000);
        hasher.add_int(entry.first % 1000);
        hasher.add_double(entry.second.d1);
        hasher.add_double(entry.second.d2);
    }
    return hasher.finish();
}

std::string PM6ParameterSet::parameter_identity() const {
    const std::string sha256 = content_sha256();
    return pm6_parameter_identity_for_hash(method_name_, sha256);
}

int PM6ParameterSet::pair_key(int Z1, int Z2) const {
    if (Z1 < 1 || Z1 > 98 || Z2 < 1 || Z2 > 98) {
        throw std::invalid_argument(
            "NDDO diatomic atomic numbers must be in [1, 98]");
    }
    return std::min(Z1, Z2) * 1000 + std::max(Z1, Z2);
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
