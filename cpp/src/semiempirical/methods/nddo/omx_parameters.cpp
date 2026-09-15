#include "vibeqc/semiempirical/methods/nddo/omx_parameters.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"

#include <cmath>
#include <stdexcept>

namespace vibeqc {
namespace semiempirical {
namespace nddo {

namespace {

// Canonical loader digests measured from the rebuilt native extension.
// Update them only with the loaders and regression oracles; published
// provenance comes from these literal allowlists, never names or counts.
constexpr const char* kOM1DralSha256 =
    "be0bb9541df6628dd399236d27c0591417da1e4b5469c32aa7347b89845436e1";
constexpr const char* kOM2DralSha256 =
    "9c73ec5520b0736b6f6215e8debbbd845a4c4edce1ee56eeb0e25f712a2096f7";
constexpr const char* kOM3DralSha256 =
    "26a937d0ada13634eb05487279a958ad8f2ddc7f5e91306bc2cc0a07e787575d";

constexpr const char* kOM1DralIdentity =
    "published:om1-dral-2016-v1";
constexpr const char* kOM2DralIdentity =
    "published:om2-dral-2016-v1";
constexpr const char* kOM3DralIdentity =
    "published:om3-dral-2016-v1";

std::string omx_parameter_identity_for_hash(
    OMxVariant variant,
    const std::string& sha256) {
    switch (variant) {
        case OMxVariant::OM1:
            if (sha256 == kOM1DralSha256) return kOM1DralIdentity;
            break;
        case OMxVariant::OM2:
            if (sha256 == kOM2DralSha256) return kOM2DralIdentity;
            break;
        case OMxVariant::OM3:
            if (sha256 == kOM3DralSha256) return kOM3DralIdentity;
            break;
    }
    return detail::custom_parameter_identity(sha256);
}

void append_omx_element_identity(
    detail::CanonicalParameterHasher& hasher,
    const OMxElementData& element) {
    hasher.add_int(element.Z);

    hasher.add_double(element.uss);
    hasher.add_double(element.upp);
    hasher.add_double(element.gss);
    hasher.add_double(element.gpp);
    hasher.add_double(element.gsp);
    hasher.add_double(element.gp2);
    hasher.add_double(element.hsp);

    hasher.add_double(element.zeta);
    hasher.add_double(element.beta_s);
    hasher.add_double(element.beta_p);
    hasher.add_double(element.beta_pi);
    hasher.add_double(element.alpha_s);
    hasher.add_double(element.alpha_p);
    hasher.add_double(element.alpha_pi);

    hasher.add_double(element.beta_s_xh);
    hasher.add_double(element.beta_p_xh);
    hasher.add_double(element.alpha_s_xh);
    hasher.add_double(element.alpha_p_xh);

    hasher.add_double(element.F1);
    hasher.add_double(element.F2);
    hasher.add_double(element.G1);
    hasher.add_double(element.G2);

    hasher.add_double(element.zeta_alpha);
    hasher.add_double(element.F_alpha_alpha);
    hasher.add_double(element.beta_alpha);
    hasher.add_double(element.alpha_alpha);

    hasher.add_double(element.zs);
    hasher.add_double(element.zp);
    hasher.add_double(element.alpha);
    hasher.add_int(element.n_orbitals);
}

}  // namespace

bool omx_has_complete_parameters(
    const OMxElementData& element,
    OMxVariant variant) {
    auto finite = [](double value) { return std::isfinite(value); };
    auto finite_nonzero = [&finite](double value) {
        return finite(value) && std::abs(value) > 1.0e-12;
    };
    auto finite_positive = [&finite](double value) {
        return finite(value) && value > 1.0e-12;
    };
    auto finite_zero = [&finite](double value) {
        return finite(value) && std::abs(value) <= 1.0e-12;
    };

    const double fields[] = {
        element.uss, element.upp, element.gss, element.gpp,
        element.gsp, element.gp2, element.hsp, element.zeta,
        element.beta_s, element.beta_p, element.beta_pi,
        element.alpha_s, element.alpha_p, element.alpha_pi,
        element.beta_s_xh, element.beta_p_xh,
        element.alpha_s_xh, element.alpha_p_xh,
        element.F1, element.F2, element.G1, element.G2,
        element.zeta_alpha, element.F_alpha_alpha,
        element.beta_alpha, element.alpha_alpha,
        element.zs, element.zp, element.alpha,
    };
    for (double value : fields) {
        if (!finite(value)) return false;
    }

    const bool supported_element = element.Z == 1 || element.Z == 6
        || element.Z == 7 || element.Z == 8 || element.Z == 9;
    if (!supported_element
        || !finite_nonzero(element.uss)
        || !finite_positive(element.gss)
        || !finite_positive(element.zeta)
        || !finite_positive(element.zs)
        || !finite_nonzero(element.beta_s)
        || !finite_positive(element.alpha_s)
        || !finite_nonzero(element.F1)) {
        return false;
    }

    const bool s_only = element.Z == 1;
    if (element.n_orbitals != (s_only ? 1 : 4)) return false;
    if (!s_only
        && (!finite_nonzero(element.upp)
            || !finite_positive(element.gpp)
            || !finite_positive(element.gsp)
            || !finite_positive(element.gp2)
            || !finite_positive(element.hsp)
            || !finite_positive(element.zp)
            || !finite_nonzero(element.beta_p)
            || !finite_nonzero(element.beta_pi)
            || !finite_positive(element.alpha_p)
            || !finite_positive(element.alpha_pi)
            || !finite_positive(element.alpha))) {
        return false;
    }

    // These are the nonzero correction channels in the published OM1/OM2/OM3
    // parameterizations.  A zero is meaningful only where the corresponding
    // fitted model omits that channel.
    if ((variant == OMxVariant::OM1 || variant == OMxVariant::OM2)
        && !finite_nonzero(element.F2)) {
        return false;
    }
    if ((variant == OMxVariant::OM2 || variant == OMxVariant::OM3)
        && !finite_nonzero(element.G1)) {
        return false;
    }
    if (variant == OMxVariant::OM2 && !finite_nonzero(element.G2)) {
        return false;
    }

    if (variant == OMxVariant::OM1
        && (!finite_zero(element.G1) || !finite_zero(element.G2))) {
        return false;
    }
    if (variant == OMxVariant::OM3
        && (!finite_zero(element.F2) || !finite_zero(element.G2))) {
        return false;
    }

    // OM2/OM3 use the fitted ECP channel on heavy atoms; hydrogen has no core.
    if (!s_only && variant != OMxVariant::OM1
        && (!finite_positive(element.zeta_alpha)
            || !finite_nonzero(element.F_alpha_alpha)
            || !finite_nonzero(element.beta_alpha)
            || !finite_positive(element.alpha_alpha))) {
        return false;
    }
    const bool ecp_is_zero = finite_zero(element.zeta_alpha)
        && finite_zero(element.F_alpha_alpha)
        && finite_zero(element.beta_alpha)
        && finite_zero(element.alpha_alpha);
    if ((s_only || variant == OMxVariant::OM1) && !ecp_is_zero) {
        return false;
    }

    const bool xh_is_zero = finite_zero(element.beta_s_xh)
        && finite_zero(element.beta_p_xh)
        && finite_zero(element.alpha_s_xh)
        && finite_zero(element.alpha_p_xh);
    const bool xh_is_complete = finite_nonzero(element.beta_s_xh)
        && finite_nonzero(element.beta_p_xh)
        && finite_positive(element.alpha_s_xh)
        && finite_positive(element.alpha_p_xh);
    if (s_only && !xh_is_zero) return false;
    if (!s_only) {
        const bool has_published_xh = variant != OMxVariant::OM1
            || element.Z == 7 || element.Z == 8;
        if (has_published_xh ? !xh_is_complete : !xh_is_zero) return false;
    }
    return true;
}

OMxParameterSet::OMxParameterSet(OMxVariant variant) : variant_(variant) {
    switch (variant_) {
        case OMxVariant::OM1:
        case OMxVariant::OM2:
        case OMxVariant::OM3:
            return;
    }
    throw std::invalid_argument("invalid OMx parameter variant");
}

std::string OMxParameterSet::method_name() const {
    switch (variant_) {
        case OMxVariant::OM1: return "om1";
        case OMxVariant::OM2: return "om2";
        case OMxVariant::OM3: return "om3";
    }
    return "omx";
}

std::string OMxParameterSet::parameter_version() const {
    switch (variant_) {
        case OMxVariant::OM1: return "om1-1993";
        case OMxVariant::OM2: return "om2-2000";
        case OMxVariant::OM3: return "om3-2003";
    }
    return "omx";
}

ParameterSetMetadata OMxParameterSet::metadata() const {
    ParameterSetMetadata m;
    m.method_name = method_name();
    m.version = parameter_version();
    m.n_elements = static_cast<int>(omx_elements_.size());
    m.element_list.reserve(omx_elements_.size());
    for (const auto& entry : omx_elements_) {
        m.element_list.push_back(entry.first);
    }
    m.parameter_hash = content_sha256();
    const std::string identity = omx_parameter_identity_for_hash(
        variant_, m.parameter_hash);
    if (identity == kOM1DralIdentity
        || identity == kOM2DralIdentity
        || identity == kOM3DralIdentity) {
        m.origin = "published";
        m.license = "published parameter table";
        m.doi_or_url = "10.1021/acs.jctc.5b01046";
    } else {
        m.origin = identity;
    }
    return m;
}

std::string OMxParameterSet::content_sha256() const {
    detail::CanonicalParameterHasher hasher("nddo.omx.v1");
    hasher.add_string(method_name());
    hasher.add_size(omx_elements_.size());
    for (const auto& entry : omx_elements_) {
        // The OMx record is authoritative.  Its synchronized inherited
        // NDDO/Core projections are intentionally absent from this schema.
        append_omx_element_identity(hasher, entry.second);
    }
    return hasher.finish();
}

std::string OMxParameterSet::parameter_identity() const {
    const std::string sha256 = content_sha256();
    return omx_parameter_identity_for_hash(variant_, sha256);
}

const OMxElementData* OMxParameterSet::omx_data(int Z) const {
    auto it = omx_elements_.find(Z);
    return (it != omx_elements_.end()) ? &it->second : nullptr;
}

void OMxParameterSet::add_omx_element(const OMxElementData& ed) {
    if (!omx_has_complete_parameters(ed, variant_)) {
        throw std::invalid_argument(
            method_name() + " element Z=" + std::to_string(ed.Z)
            + " is not a complete executable OMx parameter record");
    }
    omx_elements_[ed.Z] = ed;

    // Also register the NDDO base parameters so element_data() works
    NDDOElementData nddo_ed;
    nddo_ed.Z = ed.Z;
    nddo_ed.uss = ed.uss;
    nddo_ed.upp = ed.upp;
    nddo_ed.gss = ed.gss;
    nddo_ed.gpp = ed.gpp;
    nddo_ed.gsp = ed.gsp;
    nddo_ed.gp2 = ed.gp2;
    nddo_ed.hsp = ed.hsp;
    nddo_ed.zs = ed.zs;
    nddo_ed.zp = ed.zp;
    nddo_ed.betas = ed.beta_s;   // OMx βs maps to PM6 betas
    nddo_ed.betap = ed.beta_p;   // OMx βp maps to PM6 betap
    nddo_ed.alpha = ed.alpha;
    nddo_ed.n_orbitals = ed.n_orbitals;
    nddo_ed.has_d = false;
    // Gamma terms: empty → use simple Ohno-Klopman (the new run_omx_v2
    // builds its own two-centre repulsion)
    store_element(nddo_ed);
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
